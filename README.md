# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-13 | 今日论文总数: 566

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Context-dependent manifold learning: A neuromodulated constrained autoencoder approach

**arXiv ID:** 2603.11673 | [PDF](https://arxiv.org/pdf/2603.11673v1)

**作者:** Jérôme Adriaens `[一作]` (University of Liège), Pierre Sacré `[通讯]` (University of Liège)

**通讯引用:** 2467 | [OpenAlex ID](https://openalex.org/A5034817041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种具备可调节神经调制机制的约束自编码器，以实现上下文相关的流形学习。

**💡 创新点**

创新点在于将神经调制作用于自编码器的激活函数和偏置，实现在保持投影算子幂等性质的同时，使模型能够根据静态上下文参数动态重塑流形几何。

**🔧 技术方法**

采用了约束自编码器架构（双正交权重矩阵与互逆激活函数）、黎曼优化以及超网络生成上下文相关参数等技术。

**📊 数据集**

在两套模拟动力学系统上进行评估：一个具有可变几何形态的16自由度摆系统以及在分岔区间内变化的Lorenz96系统。

**📈 对比分析**

通过与无上下文约束AE和简单拼接式条件AE对比，发现Neuromodulated Constrained AE在重构误差（RMSE）上平均降低约75%，并在潜在空间结构上显著优于基线模型。

**⚠️ 局限性**

局限性包括仅在模拟数据上验证、需要显式的上下文向量、黎曼优化计算量较大，以及在更复杂或真实世界的非平稳数据中可能面临泛化与训练稳定性挑战。

---

## 2. Reference-Guided Machine Unlearning

**arXiv ID:** 2603.11210 | [PDF](https://arxiv.org/pdf/2603.11210v1)

**作者:** Jonas Mirlach `[一作]` (ETH Zurich), Julia E. Vogt `[通讯]` (ETH Zurich)

**通讯引用:** 79884 | [OpenAlex ID](https://openalex.org/A5047171254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于参考分布的机器无学习框架 ReGUn，通过将被遗忘样本的预测对齐到一个离散的 held‑out 参考集上，以实现对忘记数据的影响去除。

**💡 创新点**

创新点在于将无学习目标从简单的性能下降（如 loss 最大化或随机标签）转变为分布可区分性；通过使用类条件的参考分布进行蒸馏，保证模型在忘记样本上的行为与真正未见样本一致。

**🔧 技术方法**

使用的技术包括：1) 参考分布构造（从 held‑out 集中按类别抽样并取 softmax 均值）；2) 结合 KL 散度蒸馏与交叉熵训练的混合损失；3) 采用离散参考模型（如初始训练模型）避免额外训练；4) 在微调过程中保持对保留样本的正向监督。

**📊 数据集**

实验使用的公开数据集包括 CIFAR‑10、CIFAR‑100 以及 Tiny‑ImageNet；模型架构主要为 ResNet‑18（CNN）和 Swin‑T（Transformer）。

**📈 对比分析**

与多种基准（Finetune、NegGrad、NegGrad+、ℓ1‑sparse、SSD、SalUn、Amun）及 retrain 基线比较，ReGUn 在大多数设置下均表现出更小的 retrain‑gap，尤其在 Transformer 任务中显著降低 RMIA AUC，实现更优的忘记‑效用权衡。

**⚠️ 局限性**

局限性包括：1) 需要额外的 held‑out 数据，且该数据可能不总是可获得；2) 参考模型使用初始模型仍保留忘记数据的影响；3) 仅在分类任务上验证，未覆盖生成模型或大规模基础模型；4) 对极大忘记比例或高度不平衡类别的鲁棒性尚待进一步评估。

---

## 3. Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

**arXiv ID:** 2603.11653 | [PDF](https://arxiv.org/pdf/2603.11653v1)

**作者:** Jiaheng Hu `[一作]` (UT Austin), Roberto Martin-Martin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并系统评估了连续强化学习（CRL）在大规模视觉‑语言‑动作（VLA）模型上的表现，并发现简单的按序微调（Sequential Fine‑Tuning，Seq. FT）配合低秩适配（LoRA）即可实现高可塑性、几乎无灾难性遗忘，并保持甚至提升零样本泛化能力。

**💡 创新点**

创新点在于：①将Seq. FT与LoRA在大预训练VLA上组合使用并在多任务序列中进行验证，挑战了传统认为Seq. FT会导致灾难性遗忘的主流观点；②通过对比八种主流CRL方法（正则化、回放、参数隔离等）和多任务上界，证明Seq. FT既简洁又性能优越；③从理论与实验角度剖析了大模型、LoRA与自策略RL共同作用如何重新塑造稳定性‑可塑性权衡。

**🔧 技术方法**

主要技术包括：大规模预训练的VLA模型（OpenVLA‑OFT、Pi‑0、OpenVLA），低秩适配（LoRA）进行参数高效微调，基于自策略的强化学习框架GRPO，以及对比实验中的八种连续学习算法（EWC、Expert Replay、Dark Experience Replay、Dynamic Weight Expansion、SLCA、RETAIN 等）。

**📊 数据集**

使用的基准数据集包括：Libero（对象、空间、长时序子任务）、RoboCasa（多场景多任务）以及Maniskill（基于SAPIEN的机器人操控任务）。

**📈 对比分析**

与传统CRL方法的比较显示：Seq. FT 在平均成功率（AVG）上常接近或优于多任务上限，多任务训练的正则化或回放方法往往导致可塑性下降；在零样本成功率（ZS）上，Seq. FT 往往保持甚至略优于多任务训练。实验进一步表明，Seq. FT 对环境扰动、模型变更和任务顺序的鲁棒性都很强。

**⚠️ 局限性**

局限性包括：①实验中未对Seq. FT 进行超参数调优，仅使用默认 LoRA 设置；②只评估了少量随机种子（3 次）和高成本 GPU 资源；③结果主要来自模拟环境，实物机器人转移效果未验证；④对 Seq. FT 能否在更大尺度、更复杂任务序列中保持性能仍有待进一步探索。

---

## 4. Understanding Wikidata Qualifiers: An Analysis and Taxonomy

**arXiv ID:** 2603.11767 | [PDF](https://arxiv.org/pdf/2603.11767v1)

**作者:** Gilles Falquet `[一作]` (University of Geneva), Sahar Aljalbout `[通讯]` (University of Geneva)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5027417626)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对维基数据中的限定词进行了定量与定性分析，计算其频率和多样性，并基于最重要的300个限定词构建了一套新的层次化分类体系。

**💡 创新点**

创新点在于将频率与改进的香农熵（Hill数）相结合以衡量限定词的重要性，并将原有的“上下文/附加”分类细化为上下文、认知/不确定性、结构与附加四大类及其子类。

**🔧 技术方法**

主要技术包括数据挖掘脚本（Python），改进的多样性指数（Hill数），以及基于属性-限定词共现关系的聚类与手工归类。

**📊 数据集**

使用的数据集为 2025‑01‑01 版本的 Wikidata JSON 转储，共计 1,625,467,397 条语句，提取出 1,357 个可用限定词及其频次、共现属性等统计信息。

**📈 对比分析**

通过对前300个限定词的覆盖率（占 99.6% 说明）以及各类平均多样性和频率的统计，展示了新分类体系的完整性和可解释性；然而并未进行外部基准评测或推理性能比较。

**⚠️ 局限性**

局限性包括仅关注前300个限定词，无法覆盖所有少用或新引入的限定词；分类时存在歧义和误用现象，导致部分限定词被多类归属；且缺乏针对实际查询或推理任务的性能评估。

---

## 5. ActiveFreq: Integrating Active Learning and Frequency Domain Analysis for Interactive Segmentation

**arXiv ID:** 2603.11498 | [PDF](https://arxiv.org/pdf/2603.11498v1)

**作者:** Lijun Guo `[一作]` (Wuhan University), Gang Ke `[通讯]` (Dongguan Polytechnic)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种名为 ActiveFreq 的交互式分割框架，结合主动学习选择模块 AcSelect 与频域分析模块 FreqFormer，显著降低用户点击次数而保持高分割精度。

**💡 创新点**

创新点在于（1）使用主动学习评估并优先挑选最具信息量的误标区域，避免随机点击；（2）将傅里叶变换引入分割网络，融合空间与频域特征，提升对细节与全局结构的捕捉。

**🔧 技术方法**

采用熵基不确定度度量、多维二维离散傅里叶变换、Transformer 编码器+FreqNet 解码器、ROI Align 与轻量化多分支解码块等技术。

**📊 数据集**

在 ISIC‑2017 皮肤病变图像数据集和 OAI‑ZIB 骨关节 MRI 图像数据集上进行实验。

**📈 对比分析**

与 7 种 SOTA 方法（如 FocalClick、iSegFormer、GraCo 等）对比，ActiveFreq 在 ISIC‑2017 上 NoC@90 仅需 3.74 次点击，OAI‑ZIB 上 9.27 次，分别比最佳对手提升 10.1% 与 0.12 次，整体实现了显著性能提升。

**⚠️ 局限性**

局限性包括 AcSelect 仅逐区域评估，未考虑区域间交互；FreqModule 仅使用 2D DFT 组合，缺乏全 3D 频域互补信息，未来可探索多维或可学习权重的频域融合。

---

## 6. MaterialFigBENCH: benchmark dataset with figures for evaluating college-level materials science problem-solving abilities of multimodal large language models

**arXiv ID:** 2603.11414 | [PDF](https://arxiv.org/pdf/2603.11414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 7. XSkill: Continual Learning from Experience and Skills in Multimodal Agents

**arXiv ID:** 2603.12056 | [PDF](https://arxiv.org/pdf/2603.12056v1)

**作者:** Guanyu Jiang `[一作]` (Hong Kong University of Science and Technology), Fung `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出XSkill框架，使多模态代理在不更新参数的情况下通过经验和技能的双流持续学习，提升工具使用效率和组合灵活性。

**💡 创新点**

创新在于将任务级技能与动作级经验统一为视觉可观测的非参数知识库，并实现跨轨迹提炼、层次化整合、视觉上下文适配与检索闭环。

**🔧 技术方法**

使用多模态大语言模型(MLLM)进行知识抽取、重写、适配；视觉可观测摘要、跨轨迹批评、层次化合并；检索-适配-注入机制；支持跨模型知识迁移。

**📊 数据集**

在五个多模态基准上评估：VisualToolBench、TIR-Bench、MMSearch-Plus、MMBrowseComp、AgentVista，使用Gemini‑2.5‑Pro、Gemini‑3‑Flash、GPT‑5‑mini、o4‑mini等四个后端模型。

**📈 对比分析**

与无工具、工具+经验、AWM、DC、Agent‑KB等基线对比；XSkill在平均@4和Pass@4指标上整体提升2.58–6.71点（在不同模型上），对复杂视觉推理和多步工具组合的任务优势最为显著，且在零样本跨任务转移中亦优于基线。

**⚠️ 局限性**

局限包括：仅在单轮积累-测试循环中验证，需进一步探索持续迭代；对视觉质量依赖较高，可能在视觉噪声下表现下降；知识库扩展可能面临冗余和偏差问题，需要人工审查与审计。

---

## 8. Thousand-GPU Large-Scale Training and Optimization Recipe for AI-Native Cloud Embodied Intelligence Infrastructure

**arXiv ID:** 2603.11101 | [PDF](https://arxiv.org/pdf/2603.11101v1)

**作者:** Chen Zhou `[一作]` (Jingdong Technology Company), Zhen Sun `[通讯]` (Jingdong Technology Company)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了千GPU分布式训练框架并实现了大规模VLA模型（如GR00T N1.5、π系列）在云原生基础设施上的训练与评估；

**💡 创新点**

创新性地融合多维并行（3D+专家+序列），引入完全异步RL‑VLA^3训练管线、可变长度FlashAttention与数据打包、细粒度FP8量化，并将LeRobot与NVIDIA Isaac Sim生态系统在JoyBuilder平台上无缝集成；

**🔧 技术方法**

采用DDP、管道/张量/专家/序列并行、Ray、NVIDIA Isaac Sim、JoyBuilder、3.2 T RDMA网络、FlashAttention、数据打包、PTQ（FP8）、RL异步策略；

**📊 数据集**

使用LIBERO、ManiSkill、Qwen2.5‑VL等多模态数据集以及百万帧的嵌入式数据；

**📈 对比分析**

与同步训练、RLinf等基线对比，吞吐率提升59%–127%，GR00T N1.5单轮训练时长从15 h降至22 min（≈40×），π_0.5训练时间缩短40%，Variable‑Length FlashAttention在高填充率下可节省至90%计算；

**⚠️ 局限性**

在大规模 (>256 GPU) 下通信与负载不平衡导致子线性扩展，某些环境（如ManiSkill）对Rollout异步效果有限，仍需进一步优化Sim2Real一致性、评估标准与安全风险评估。

---

## 9. Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE

**arXiv ID:** 2603.11611 | [PDF](https://arxiv.org/pdf/2603.11611v1)

**作者:** Mohammad Aflah Khan `[一作]` (Max Planck Institute for Software Systems), Abhilasha Ravichander `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了旋转位置编码(RoPE)在隐藏维度中部分应用对大型Transformer训练收敛、内存占用和稳定性的影响，证明仅10%维度使用RoPE即可匹配完整RoPE的收敛性能并大幅降低内存；

**💡 创新点**

创新点在于量化不同RoPE比例对内存节省（最多10×）和训练稳定性的关系，揭示NoPE配置导致损失尖峰并通过QK-Norm归一化解决，同时提供跨架构、序列长度和模型规模的综合分析；

**🔧 技术方法**

采用部分RoPE配置、序列与并行注意力架构、QK-Norm归一化、EleutherAI LM Evaluation Harness评测工具，以及FineWeb与FineWeb‑Edu大规模预训练；

**📊 数据集**

使用了FineWeb（100B tokens）和FineWeb‑Edu（100B tokens）作为预训练数据，评估集包括ARC、LogiQA、LAMBADA、PIQA、SciQ、WinoGrande、WSC、PubMedQA等多项基准；

**📈 对比分析**

通过训练损失曲线、最终损失、perplexity以及多项MCQ基准评估对比不同RoPE比例、序列长度、模型规模，发现10%及以上RoPE与完整RoPE在损失和基准上几乎相同，内存显著降低，NoPE表现差，QK‑Norm可缓解损失尖峰；

**⚠️ 局限性**

局限性在于仅覆盖有限的模型尺寸、架构和数据量，未能探索全部组合；高计算成本限制实验规模；未深入研究部分RoPE与长度外推、组合NoPE等更复杂配置的交互。

---

## 10. Learning Tree-Based Models with Gradient Descent

**arXiv ID:** 2603.11117 | [PDF](https://arxiv.org/pdf/2603.11117v1)

**作者:** Sascha Marton `[一作]` (University of Mannheim), Sascha Marton `[通讯]` (University of Mannheim)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5022929385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种通过梯度下降学习硬轴对齐决策树及其集成的方法，利用稠密表示和直通算子实现所有树参数的联合优化。

**💡 创新点**

创新点在于将传统离散、非可微决策树转换为可微模型，使其能够通过梯度下降而非贪心搜索学习，并引入实例级加权的集成框架，实现性能与可解释性之间的平衡。

**🔧 技术方法**

核心技术包括稠密决策树表示、直通（straight‑through）反向传播算子、梯度下降优化以及实例级加权的集成策略。

**📊 数据集**

论文未给出具体数据集名称，评估结果涉及多种应用场景：小型表格数据集、复杂表格数据集、多模态学习任务以及可解释的强化学习任务。

**📈 对比分析**

与传统CART等贪心方法及其他基线模型相比，该方法在上述多领域实验中实现了最新的（state‑of‑the‑art）性能，同时保持了树模型的可解释性。

**⚠️ 局限性**

局限性包括：仍局限于轴对齐分裂，稠密表示可能导致内存占用较高；直通算子在离散化过程中引入梯度估计误差；在极大规模数据或高维特征上尚未充分验证其可扩展性。

---

## 11. MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning

**arXiv ID:** 2603.12266 | [PDF](https://arxiv.org/pdf/2603.12266v1)

**作者:** Haozhan Shen `[一作]` (Alibaba Group), Jianwei Yin `[通讯]` (Zhejiang University)

**通讯引用:** 7311 | [OpenAlex ID](https://openalex.org/A5069353502)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了MM-CondChain基准，用于评估多模态大型语言模型在视觉条件下的深层连锁推理能力。

**💡 创新点**

创新点在于：①通过Verifiable Programmatic Intermediate Representation（VPIR）实现逻辑与语言的分离，保证每层条件可机械验证；②引入层级化、可回溯的agentic synthesis pipeline（Planner–Verifier–Composer），实现可扩展、可控的难度调节；③构造了层级链条的硬负样本（True/False路径），使模型需逐层验证而非一次性匹配。

**🔧 技术方法**

技术手段包括：结构化事实抽取、VPIR谓词生成与程序化验证、Planner‑based链条构造、Verifier的双阶段质量控制、Composer的配对路径编译、跨域（自然图像、图表、GUI轨迹）的统一处理。

**📊 数据集**

使用公开数据集构建：自然图像（SAM 204 张 + GQA 194 张），图表（ChartQA 200 张），GUI轨迹（AITZ 377 轨迹，AIGW 3,421 截图），共 975 条实例，形成多层、可验证的条件链。

**📈 对比分析**

对比方法：在零样本设置下评估十余款开源与专有 MLLM，计算 True‑Path Accuracy、False‑Path Accuracy 与 Path F1（两者的调和均值）。最高平均 Path F1 仅为 53.33（Qwen3‑VL），表明当前模型即使最强也仅略高于随机，且在 False‑Path 任务上表现尤为差劲；性能随链深度和谓词复杂度显著下降。

**⚠️ 局限性**

局限性：① benchmark 仅关注可验证的逻辑条件，未覆盖更自由的视觉推理场景；② 过度依赖程序化验证，可能忽略模型对“近似”视觉证据的处理能力；③ 对于 GUI 轨迹的多帧时序理解仍不足；④ 目前评测缺乏对模型推理过程的可解释性分析，难以定位错误根源。

---

## 12. Structure Selection for Fairness-Constrained Differentially Private Data Synthesis

**arXiv ID:** 2603.12112 | [PDF](https://arxiv.org/pdf/2603.12112v1)

**作者:** Naeim Ghahramanpour `[一作]` (University of Western Ontario), Mostafa Milani `[通讯]` (University of Western Ontario)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在差分隐私数据合成过程中通过结构选择强制满足条件独立性(CI)约束的方法，直接在树结构的构造阶段过滤不符合公平约束的边，以保证合成数据在保持隐私的同时去除敏感属性与结果之间的无意义相关性。

**💡 创新点**

创新点在于将CI约束嵌入Kruskal式最大生成树的私有边选择过程，既保持了标准MST+PrivatePGM的隐私计量与可扩展性，又在测量阶段就实现了公平性约束，避免了后期修复带来的预算浪费和偏差放大。

**🔧 技术方法**

核心技术包括：差分隐私（zCDP/RDP）预算划分、指数机制用于私有边选择、基于边可行性检查的CI约束过滤、以及PrivatePGM作为重建模块；同时利用低敏感度的相关度度量作为边权。

**📊 数据集**

实验使用了五个公开公平性基准数据集：Adult、COMPAS、Dutch Census、German Credit 与 Law School，均按照预定义的Outcome、Protected、Admissible 与 Inadmissible 划分，并在每个数据集上施加 O ⊥ S | A 的条件独立性约束。

**📈 对比分析**

与未加入公平约束的PrivBayes、以及先前提出的PreFair做比较；实验结果显示，在相同隐私预算下，该方法在代理MI得分、KL/TV分布相似度、以及下游逻辑回归AUC指标上均优于PreFair，且在CMI与Equalized Odds等公平性指标上达到与PreFair相当或更优的表现。

**⚠️ 局限性**

局限性包括：只能处理单一CI约束且仅限树结构模型，无法直接扩展到多重或重叠的约束；对连续变量的处理需要离散化；当公平约束紧密且唯一时，结构选择改进空间有限；未评估在深度生成模型或更复杂分布下的可扩展性。

---

## 13. SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation

**arXiv ID:** 2603.11492 | [PDF](https://arxiv.org/pdf/2603.11492v1)

**作者:** Xiaogang Du `[一作]` (Shaanxi University of Science and Technology), Yingbo Wang `[通讯]` (Shaanxi University of Science and Technology)

**通讯引用:** 2414 | [OpenAlex ID](https://openalex.org/A5103104824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种面向医学图像分割的持续测试时自适应框架（SPEGC），通过语义提示增强特征并利用可微分图聚类进行高阶结构学习，实现对不断变化无标签域的自适应。

**💡 创新点**

创新点在于：①使用分离的共性与异质性语义提示池在特征空间中注入全局上下文信息，提升对域移噪声的鲁棒性；②将图聚类转化为可微分的最优传输问题，端到端稀疏化相似度矩阵得到高阶结构表征；③以该结构为监督，形成一致性与聚类损失，显著缓解错误累积与灾难性遗忘。

**🔧 技术方法**

核心技术包括：ResNet-50 backbone + MC Dropout不确定性采样；语义提示增强（Prompt Feature Enhancement，SPFE）；可微分图聚类求解器（DGCS，基于Sinkhorn算法的最优传输）；图一致性损失和聚类损失；源无关的测试时适应机制。

**📊 数据集**

实验使用两组医学分割基准：视网膜视盘/杯（RIM-ONE、REFUGE、ORIGA、REFUGE-Test、Drishti-GS）和息肉分割（BKAI-IGH-NEOPolyp、CVC-ClinicDB/CVC-612、ETIS、Kvasir）。

**📈 对比分析**

与六种代表性CTTA/ TTA 方法（SAR、DomainAdaptor、VPTTA、NC‑TTT、GraTa、TTDG）在单轮和长程连续适应场景下对比。SPEGC 在两任务均取得最高平均DSC，单轮提升约1.5%–2%，长程适应下误差累积与灾难性遗忘最小，整体表现为现有最优。

**⚠️ 局限性**

局限性包括：①图构造和最优传输求解导致计算量和显存占用高于轻量级方法；②对超参数（提示池大小、聚类数、节点池大小）敏感，需要经验调参；③在极端域移或数据极少的场景下，提示和聚类结构的可靠性仍有限。

---

## 14. Link Quality Aware Pathfinding for Chiplet Interconnects

**arXiv ID:** 2603.11612 | [PDF](https://arxiv.org/pdf/2603.11612v1)

**作者:** Aaron Yen `[一作]` (University of California, Los Angeles), Puneet Gupta `[通讯]` (University of California, Los Angeles)

**通讯引用:** 5622 | [OpenAlex ID](https://openalex.org/A5084229134)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种基于ECC开销的芯片片间互连路径寻找方法，使用RS‑FEC、CRC‑64和Go‑Back‑N ARQ对链路的能耗、面积和吞吐量进行综合评估，并利用CP‑SAT求解器实现系统级链路分配。

**💡 创新点**

创新点：① 开发了以RTL合成为基础的ECC量化流程，将错误校正能耗、面积与吞吐量整合到链路指标；② 将纠错后指标与距离、带宽、岸线宽度等物理约束联合建模的CP‑SAT分配框架；③ 通过实验验证CRC+ARQ可在满足10⁻²⁷送达BER目标时显著降低RS强度与能耗。

**🔧 技术方法**

技术手段：Synopsys Design Compiler + ASAP7/3nm库进行RS‑FEC、CRC‑64和GBN的合成；Reed‑Solomon (86,K) 码、CRC‑64/ECMA‑182 与 Go‑Back‑N ARQ；OR‑Tools CP‑SAT 约束求解；二项分布尾部计算、向量无活性功耗模型、能耗/面积归一化。

**📊 数据集**

使用的数据集包括公开的多种芯片互连技术参数（如 SuperCHIPS、Hsu、Nishi、UCIe 36G 等）以及实验所设计的三组网络拓扑（Example 1–3）。这些数据提供链路距离、原始BER、链路类型、原始吞吐/能耗等信息。

**📈 对比分析**

比较方法：将传统仅基于未纠错指标的贪心链路选择与全量ECC校正的CP‑SAT分配结果对比。结果显示：在短距离/低带宽场景两种方法相似；在长距离/高带宽场景，全量方法可降低能耗约30%–40%、面积约15%–20%，并在中等BER下将RS强度降低约2–3倍、goodput 提升10–20%。

**⚠️ 局限性**

局限性：仅假设独立同分布的比特错误，未考虑突发错误与错误结构；ARQ模型假设独立重传；未评估时延、窗口大小和路由拥塞；链路参数取自公开资料，未通过硬件实验验证；能耗估计基于向量无活性模型，忽略漏电与功耗动态分布；未处理多路复用、不同速率和协议层的交互影响。

---

## 15. MANSION: Multi-floor lANguage-to-3D Scene generatIOn for loNg-horizon tasks

**arXiv ID:** 2603.11554 | [PDF](https://arxiv.org/pdf/2603.11554v1)

**作者:** Lirong Che `[一作]` (Tsinghua University), Jian Su `[通讯]` (AgiBot)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了MANSION框架，利用语言驱动生成可跨层级、可编辑的多层建筑三维环境，并发布了涵盖1,000+多层建筑的MansionWorld数据集；

**💡 创新点**

核心创新在于将垂直结构作为硬约束，使用混合MLLM+几何求解器实现语义化的可验证楼层规划，并配合Task‑Semantic Scene Editing Agent实现任务可执行性；

**🔧 技术方法**

采用LLM（如Gemini‑2.5‑Pro）与几何求解器、LangGraph多代理编排、AI2‑THOR扩展（楼梯、电梯及跨层技能API）以及可编程对象放置算法；

**📊 数据集**

使用T2D、ResPlan（1K样本）等真实与合成楼层图数据，以及自研的1,000多栋多层建筑集合；

**📈 对比分析**

在楼层图生成上与CHD相当（MA设置下Micro‑IoU≈80%），在更复杂ResPlan‑1K上显著优于CHD（Micro‑IoU≈77%），对象放置方面达100%可达率、较低碰撞率，且在用户评测中在布局、真实感和多样性上均优于LayoutGPT和Holodeck；

**⚠️ 局限性**

仍存在跨层级长时序任务失败率高（四层任务全失），高层规划与记忆不足；生成模型对极大或复杂形状的房间仍易产生定位误差；整体算法依赖LLM的指针精度，需进一步提升可解释性与可调性。

---

## 16. SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory

**arXiv ID:** 2603.11746 | [PDF](https://arxiv.org/pdf/2603.11746v1)

**作者:** Dingcheng Zhen `[一作]` (Soul AI Lab), Shunshun Yin `[通讯]` (Soul AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于邻居强制（Neighbor Forcing）的步骤一致性自回归扩散模型，并引入 ConvKV 记忆压缩机制，实现了小时级实时人类动画的无限长视频生成，同时配备情感与动作编辑模块。

**💡 创新点**

创新点：①将邻近帧在同一扩散步骤的潜在表示作为条件，形成步骤一致的自回归传播；②通过 ConvKV（1D 卷积 + RoPE）将 KV 缓存压缩为固定长度，实现常数内存推理；③将上述机制与预训练 DiT 结合，并在同一框架下完成音频、文本、情感与动作的多模态对齐与编辑。

**🔧 技术方法**

使用技术：DiT 变压器 + Flow Matching 采样；音频交叉注意力、文本/图像交叉注意力；邻居强制训练策略；ConvKV 记忆压缩（1D 卷积 + RoPE 重置）；UniPC 求解器；FP8 低精度、序列并行与算子融合；DMD 风格蒸馏；情感动作编辑模块。

**📊 数据集**

数据集：HDTF（面部动态）、EMTD（全身动作）用于评测；训练使用约 300 小时的视频、音频与情感/动作字幕对；预训练权重来自 Wan2.1 与 InfiniteTalk。

**📈 对比分析**

与 Live-Avatar、OmniAvatar、InfiniteTalk、Bidirectional 基线等方法比较，显著提升同步指标（Sync‑C、Sync‑D）、FID/FVD、VBench 视觉质量；实时推理可达 20 FPS，仅需 2 块 H100/H200 GPU；每帧 512×512 仅 27.2 TFLOPs，低于竞品的 39.1–50.2 TFLOPs；长期视频显示身份与细节保持稳定。

**⚠️ 局限性**

局限性：仍依赖预训练的 DiT 与大量多模态数据；压缩机制虽然常数内存但对极长序列仍可能出现细节衰减；主要针对人类角色，通用物体或复杂场景的适用性尚未验证；模型复杂度高，对硬件（FP8 支持、GPU 计算力）有一定依赖。

---

## 17. PACED: Distillation at the Frontier of Student Competence

**arXiv ID:** 2603.11178 | [PDF](https://arxiv.org/pdf/2603.11178v1)

**作者:** Yuanda Xu `[一作]` (Princeton University), Zhipeng Wang `[通讯]` (Rice University)

**通讯引用:** 2032 | [OpenAlex ID](https://openalex.org/A5064248586)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Paced 框架，通过对学生的通过率进行加权，实现了更高效的 LLM 蒸馏。

**💡 创新点**

创新点在于从梯度信噪比边界消失推导 Beta 核加权，并证明其极大极小鲁棒性。

**🔧 技术方法**

技术上使用了基于学生 roll‑out 的通过率估计、Beta 核加权、两阶段 KL 顺序以及前向/反向 KL 损失。

**📊 数据集**

实验使用 DAPO 数据集，并在 MATH‑500、AIME 2024/2025 及 MMLU 上进行评估。

**📈 对比分析**

与前向 KL、Hard Filter、AKL 等基线相比，Paced 在 reasoning 任务上提升 7–15 分，遗忘率仅为 0.2–0.6%。

**⚠️ 局限性**

局限包括需要多次 roll‑out 的计算开销、指数参数手动设定以及仅验证同家族教师的情况。

---

## 18. Structure-Aware Epistemic Uncertainty Quantification for Neural Operator PDE Surrogates

**arXiv ID:** 2603.11052 | [PDF](https://arxiv.org/pdf/2603.11052v1)

**作者:** Haoze Song `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 40397 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结构感知的神经算子（NO）不确定性量化方法，利用仅对提升（lifting）模块注入随机性来生成更可靠的预测区间。

**💡 创新点**

创新点在于将随机采样限制在提升子空间，通过扰动初始特征来模拟先验不确定性，从而使不确定性与残差结构更好对齐，并显著缩小区间宽度。

**🔧 技术方法**

采用了两种轻量级特征噪声策略（通道级乘性dropout和匹配方差的高斯扰动），并结合标准校准生成置信区间；实现了对FNO和Transolver两类神经算子的无缝集成。

**📊 数据集**

实验数据集包括二维具有不连续系数的达西流动（Darcy Flow）以及三维几何变形的汽车CFD仿真（ShapeNet Car）。

**📈 对比分析**

与MC Dropout、拉普拉斯近似、深度集成等基线相比，该方法在保持高覆盖率的同时显著降低平均带宽，且推理速度与MC Dropout相当，优于深度集成。

**⚠️ 局限性**

局限性包括对提升模块设计的依赖，可能在极端高dropout率或极少采样次数时表现不佳；且方法主要针对单场问题，未扩展到多物理耦合或多场联合预测。

---

## 19. Linking Perception, Confidence and Accuracy in MLLMs

**arXiv ID:** 2603.12149 | [PDF](https://arxiv.org/pdf/2603.12149v1)

**作者:** Yuetian Du `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 19067 | [OpenAlex ID](https://openalex.org/A5100432719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套基于置信度自校准的多模态大型语言模型训练与推理框架，先通过置信度驱动的强化学习(CDRL)让模型学习识别视觉信息的充分与不足，再在推理阶段使用置信度感知的测试时缩放(CA‑TTS)和专家模型调度的Self‑Consistency、Self‑Reflection、Self‑Check模块提升答案质量。

**💡 创新点**

创新点包括：① 使用原始‑噪声图像对和置信度差异的奖励，显著提升模型对视觉扰动的敏感度；② 设计置信度驱动的多模块推理体系，并由专家模型进行模块排程和校验；③ 通过置信度校准实现“无成本”推理加速，实现测试时缩放的自由增益。

**🔧 技术方法**

采用了Group Relative Policy Optimization（GRPO）和置信度校准奖励、视觉对比解码（VCD）、专家模型的Planner/Critic/Voter角色，以及自一致性、反思与视觉检查等模块的协同推理技术。

**📊 数据集**

训练集 D_RL 由六大公开基准（包括数学推理与 VQA 数据集）筛选后生成的原始‑噪声图像对构成；评估基准包括 Math‑Vista、Math‑Vision、MMStar、MMMU 四大多模态视觉推理数据集。

**📈 对比分析**

与训练免费基线（Pass@1、Majority Voting、Deepconf）以及训练基线（DreamPRM、R1‑Onevision、VL‑Rethinker、We‑Think）进行对比，整体在四个基准上均取得 79.5%、42.4%、71.3% 与 66.3% 的准确率，平均提升约 8.8%，并在测试时缩放上展现出更陡峭的性能提升曲线。

**⚠️ 局限性**

局限性在于：① 需要昂贵的原始‑噪声图像对生成和强化学习训练，计算成本较高；② 模型性能在一定程度上依赖专家模型的可靠性与调度策略；③ 对不同视觉模态或更大规模数据的泛化能力仍待进一步验证。

---

## 20. Entropy Guided Diversification and Preference Elicitation in Agentic Recommendation Systems

**arXiv ID:** 2603.11399 | [PDF](https://arxiv.org/pdf/2603.11399v1)

**作者:** Dat Tran `[一作]` (Stanford University), Amin Saberi `[通讯]` (Stanford University)

**通讯引用:** 10479 | [OpenAlex ID](https://openalex.org/A5001035442)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种面向含糊用户查询的交互式决策支持系统(IDSS)，通过熵度量不确定性，引导偏好挖掘、结果排序与展示，实现用户意图的逐步澄清与推荐的多样化；

**💡 创新点**

核心创新在于把熵/不确定性作为一项统一信号贯穿询问、排序与展示的全过程，提出熵驱动的询问选择、基于不确定性的排名与分组多样化策略，以及对多模态信息的整合，打破传统组件分离的设计；

**🔧 技术方法**

采用大语言模型进行语义解析与自然语言提问生成；对候选集合进行熵计算；利用句子嵌入与MMR实现相似度排序；通过覆盖‑风险优化实现基于评论的属性匹配；使用分层网格展示提升可探索性；并在模拟框架中使用LLM生成的用户模型进行评估；

**📊 数据集**

以汽车推荐为主实验域，构建了基于真实用户评论的车评数据集，并进行风格一致性重写扩充；此外在电子产品域做了简要验证；所有用户模拟均基于该评论数据集；

**📈 对比分析**

通过对完整配置、去除MMR、去除熵驱动询问等消融实验，使用Precision@9、nDCG@9、满意度@9、内部多样性指标等评价；实验结果显示完整系统相较基线提升约4–5个百分点的精准度，熵驱动询问使新颖提问率提升至约95%（短查询）并显著降低无效提问；MMR显著提升内部多样性而略减小Top‑k精准度；在短查询中Embedding Similarity优于Coverage‑Risk，反之在长查询中Coverage‑Risk更佳；

**⚠️ 局限性**

主要限制在于评估依赖LLM生成的模拟用户，可能未能完整再现真实用户的多样性与策略性；实验仅覆盖汽车与简要电子产品两域，需在更广泛场景下验证；未来计划加入真实用户研究与动态多目标平衡策略。

---

## 21. A technology-oriented mapping of the language and translation industry: Analysing stakeholder values and their potential implication for translation pedagogy

**arXiv ID:** 2603.11667 | [PDF](https://arxiv.org/pdf/2603.11667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 22. Chem4DLLM: 4D Multimodal LLMs for Chemical Dynamics Understanding

**arXiv ID:** 2603.11924 | [PDF](https://arxiv.org/pdf/2603.11924v1)

**作者:** Xinyu Li `[一作]` (Australian Institute for Machine Learning), Javen Qinfeng Shi `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出化学动力学理解任务（ChemDU），并构建对应的Benchmark Chem4DBench与基线模型Chem4DLLM；

**💡 创新点**

创新点在于将4D分子轨迹（空间+时间）与自然语言进行映射，使用等变图编码器直接将原子级空间-时间信息注入LLM，提升对动态事件的解释能力；

**🔧 技术方法**

核心技术包括等变图神经网络（UMA）对每帧进行编码，特征投影与特殊标记融合，使用Qwen3-8B LLM进行自回归生成；

**📊 数据集**

使用两大数据集：Transition1x与RGD1（气相反应轨迹）以及扩展的OC20-NEB（催化表面反应轨迹）；

**📈 对比分析**

与多种基线（3D-MoLM、3D-MolT5、Chem3D-LLM、4D-MolT5、4D Text-based）进行比较，Chem4DLLM在SMILES预测、结构相似度、能量误差等指标上显著优于对手，尤其在OOD设置下保持稳健性能；

**⚠️ 局限性**

局限性包括：仅覆盖了有限类型的反应与轨迹，尚未处理更长时间尺度与更复杂的化学空间；数据量相对有限，模型对超大规模轨迹的长程记忆能力仍待提升；

---

## 23. UtilityMax Prompting: A Formal Framework for Multi-Objective Large Language Model Optimization

**arXiv ID:** 2603.11583 | [PDF](https://arxiv.org/pdf/2603.11583v1)

**作者:** Ofir Marom `[一作]` `[通讯]` (Independent Researcher), Ofir Marom (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 UtilityMax Prompting 框架，利用正式的数学目标在零样本条件下引导 LLM 进行多目标优化

**💡 创新点**

通过用影响图和期望效用函数替代模糊的自然语言目标，实现了对多目标任务的清晰、可度量的约束

**🔧 技术方法**

结合影响图建模、期望效用最大化、候选答案生成与期望值估算等技术

**📊 数据集**

在 MovieLens 1M 数据集的多目标电影推荐任务上进行实验

**📈 对比分析**

与两种自然语言基线（Basic、Harsh）对比，利用 Precision@10 与 NDCG@10 衡量，三大前沿模型（Claude Sonnet 4.6、GPT‑5.4、Gemini 2.5 Pro）中均显著提升，提升幅度约 12–18%

**⚠️ 局限性**

依赖模型能产生良好校准的概率估计，且对机率依赖假设有限制，若模型不足或变量设计不当可能收益有限

---

## 24. Huntington Disease Automatic Speech Recognition with Biomarker Supervision

**arXiv ID:** 2603.11168 | [PDF](https://arxiv.org/pdf/2603.11168v1)

**作者:** Charles L. Wang `[一作]` (Columbia University), Julia Hirschberg `[通讯]` (Columbia University)

**通讯引用:** 21505 | [OpenAlex ID](https://openalex.org/A5045037642)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对亨廷顿病患者的病理语音，构建高保真临床语料库，并系统比较了多种 ASR 架构（Encoder‑Decoder、Transducer、CTC）的零射性能；随后对最优模型 Parakeet‑TDT 进行参数高效的 encoder‑side 适配，并进一步加入基于七个临床可解释生物标记的辅助监督，研究其对识别性能及错误类型的影响。

**💡 创新点**

创新点包括：①首次在同一统一实验框架下对不同 ASR 家族在 HD 语音上的性能进行横向对比，揭示了架构特异性的错误模式；②提出了使用 encoder‑side PEFT adapter 对 Parakeet‑TDT 进行高效适配的方法；③将七个从语音学角度提取的生物标记（prosody、phonation、articulation）作为辅助监督，探究其对错误分布的重塑作用。

**🔧 技术方法**

采用的技术包括：Whisper（Encoder‑Decoder）、Parakeet‑TDT（Transducer）、Omnilingual（CTC）等开源 ASR；参数高效适配采用 encoder‑side PEFT adapters；辅助监督使用 masked mean‑pooled encoder 表示预测生物标记标签；训练采用混合精度、梯度累积、动态分桶、subsampling‑convolution chunking 等策略。

**📊 数据集**

使用的主要数据集是 4.5 小时的高保真临床语料库（130 名受试者，94 名 HD 病人，36 名对照），来自 Beth Israel Deaconess Medical Center 与 Canary Speech；此外从语料中提取七个临床可解释的生物标记，用于辅助监督。

**📈 对比分析**

在统一的 70/10/20 speaker‑independent split 上进行比较：零射阶段，Parakeet‑TDT 0.6B 达到 6.99% WER，明显优于 Whisper‑large‑v2（18.44%）和 Omnilingual（30.46%）；适配后 WER 降至 4.95%；生物标记辅助并未进一步降低总体 WER，但显著改变了代替、删除、插入三类错误的比例，并在不同病理严重度组表现出结构化的错误重塑。

**⚠️ 局限性**

局限性包括：语料量相对有限，主要为受控/朗读语音，未覆盖最严重的 HD 病例；辅助监督采用离散化标签和固定 λ，缺乏更丰富的融合策略；训练过程中无法使用验证基准选择最佳检查点，使用固定训练周期；未来需在更大、自然语音数据集上验证，并探索更细粒度的生物标记融合方法。

---

## 25. Type-safe Monitoring of Parameterized Streams

**arXiv ID:** 2603.11104 | [PDF](https://arxiv.org/pdf/2603.11104v1)

**作者:** Jan Baumeister `[一作]` (CISPA Helmholtz Center for Information Security), Florian Kohn `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出在 RTLola 监控框架中引入参数化流，并设计了一个精细化类型系统，保证在处理无限数据域时监控器不产生运行时错误。

**💡 创新点**

创新点包括：①将参数化流与实时流监控结合；②提出针对时间点的精细化（pacing）与语义（semantic）类型推断；③证明类型正确性与监控安全性的对应关系；④提供无界数据域的动态内存管理与静态错误检测机制。

**🔧 技术方法**

技术手段包括：RTLola 规范语言；精细化类型系统（pacing type 与 semantic type lattice）；形式语义与推理规则；约束求解器实现类型检查；对表达式蕴含的超近似处理。

**📊 数据集**

使用的实验数据集为：真实航空航天案例（Watchdog、RCC、FFD、Intruder、Waypoints、Geofence）以及人工生成的可扩展基准（Streams、Parameters、Conjuncts）。

**📈 对比分析**

评测方法：采用微基准测量，比较真实案例与人工基准在类型检查时间上的差异；结果显示对真实案例类型检查时间 <1 s，最坏情况（100 条流、100 个并联条件）不超过 18 s，性能可接受。

**⚠️ 局限性**

局限性：1）语义蕴含判定不可判定，需使用语法等价的上近似，可能导致误报；2）仅覆盖安全子集，无法保证所有规格无错误；3）对非确定性行为的检测仍有限；4）类型检查复杂度受约束求解器影响；5）未实现对监控代码生成的进一步优化。

---

## 26. IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL

**arXiv ID:** 2603.12151 | [PDF](https://arxiv.org/pdf/2603.12151v1)

**作者:** Zhoujun Cheng `[一作]` (University of California San Diego), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3982 | [OpenAlex ID](https://openalex.org/A5102493293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型（LLM）在 RL 后训练阶段的采样计算进行最优分配，探究平行采样数 n、批量问题数 Bp 与更新步数 M 在总计算预算 C 下的最优配置及其可预测的规模规律。

**💡 创新点**

提出在 RL 环境下可预测的“计算-规模”法则：①n 随计算预算递增并最终饱和；②在易难度不同的任务集上 n 的最优规模驱动机制不同；③在固定批量大小时，先增 Bp 后增 n；③在整体最优分配中，n 为主导参数，Bp 仅充当稳定性阈值。

**🔧 技术方法**

使用基于采样的 on‑policy RL 算法（GRPO、PPO、CISPO），结合学习率按 √B 调度、KL 与熵正则化（根据任务难度选择），并对多种模型与问题分布进行实验。

**📊 数据集**

数据集：Guru‑Math（按 avg@16 划分为 Easy 与 Hard 两类，分别约 6k 与 5k 题目），实验覆盖三大基础模型 Qwen2.5‑7B‑Instruct、Qwen3‑4B‑Instruct 与 Llama 3.1‑8B‑Instruct。

**📈 对比分析**

与传统的固定超参策略对比，本文给出的计算最优规则在验证集上实现了更高的 avg@4、best@k 与 worst@k 指标，且在不同模型与数据集上都保持一致性；实验量级达约 120k H200‑小时，展示了可直接迁移到实际生产调优的指导性。

**⚠️ 局限性**

局限性：①调优方案高度依赖任务难度与数据规模，难以一次性覆盖所有情况；②小数据集易出现过拟合导致 n 饱和提前；③多任务间的相互干扰未完全消除，导致最优 n 的波动；④仅验证了 on‑policy 方法，其他 RL 体系（如离策略）需进一步探究；⑤实验受 GPU 并行限制，某些 n/M 组合未能充分覆盖。

---

## 27. Wide-Area GNSS Spoofing and Jamming Detection Using AIS-Derived Spatiotemporal Integrity Monitoring

**arXiv ID:** 2603.11055 | [PDF](https://arxiv.org/pdf/2603.11055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 28. Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics

**arXiv ID:** 2603.11383 | [PDF](https://arxiv.org/pdf/2603.11383v1)

**作者:** Hendrik Chiche `[一作]` (OMGrab Inc.), Trevor Rigoberto Martinez `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用单目RGB-D相机与MediaPipe手部姿态估计，将人类手部动作离线转换为SO-ARM101机器人的关节轨迹，实现手影随身化的远程操控。

**💡 创新点**

创新点在于整合离线手部跟踪、深度重投影、PyBullet逆运动学和仿真预览，形成端到端的手影重现管线，且不需要训练。

**🔧 技术方法**

采用Intel RealSense D400相机、MediaPipe Hands、PyBullet逆运动学、LeRobot框架以及三层退化抓手控制。

**📊 数据集**

主要使用内部收集的RGB-D视频、20个手部标记点以及在3×3方格上的抓取任务数据；没有公开数据集。

**📈 对比分析**

与四种基于VLA的策略（ACT、SmolVLA、π_0.5、GR00T N1.5）在相同的五格抓取基准上比较，离线逆运动学达90%成功率，ACT略优为92%，其余低于50%。

**⚠️ 局限性**

主要限制是手部被环境物体遮挡导致MediaPipe失踪，特别在无结构的商店环境中成功率仅9.3%，以及在靠近机器人基座时手部自遮挡造成抓手角度估计失败。

---

## 29. Software-Hardware Binding for Protection of Sensitive Data in Embedded Software

**arXiv ID:** 2603.11727 | [PDF](https://arxiv.org/pdf/2603.11727v1)

**作者:** Bernhard Fischer `[一作]` (Software Competence Center Hagenberg), Florian Eibensteiner `[通讯]` (University of Applied Sciences Upper Austria)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5011010313)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于硬件指纹和布尔表达式的双层软件‑硬件绑定机制，用于在嵌入式软件中保护敏感数据（如PID控制器参数），并在被克隆硬件上以次优方式运行；

**💡 创新点**

创新点在于将敏感数据编码为布尔表达式，并通过SRAM PUF产生的唯一响应既解密表达式又给出变量赋值，实现双层保护；当在克隆硬件上运行时，程序会自动退化为安全但性能较差的行为；

**🔧 技术方法**

核心技术包括SRAM物理不可克隆函数（PUF）、布尔表达式生成与逻辑最小化（ESPRESSO）、对表达式的加密（AES‑128）、基于 MCU 的固件与安全特性；

**📊 数据集**

使用STM32F767ZI MCU的SRAM初始位图作为PUF数据，利用该芯片的512 KiB SRAM（最后32 bytes）进行PUF训练与提取；

**📈 对比分析**

通过对不同布尔变量数量(k)和备选数据值(m)的实验，测量了内存占用和表达式求值时间；k=6时平均耗时1.14 ms，k=10时为23.79 ms；内存使用随k指数增长；在目标MCU上验证了控制器性能优于克隆MCU；

**⚠️ 局限性**

主要限制为布尔表达式规模随变量数指数增长，导致内存和执行时间显著增加；对SRAM PUF的可靠性和温度稳定性有要求；在目标MCU上仍需防止直接读取PUF响应的物理攻击；

---

## 30. Separable neural architectures as a primitive for unified predictive and generative intelligence

**arXiv ID:** 2603.12244 | [PDF](https://arxiv.org/pdf/2603.12244v1)

**作者:** Reza T. Batley `[一作]` (Virginia Polytechnic Institute and State University), Sourav Saha `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了可分离神经架构（SNA），在四个不同领域（预测逆向、微结构生成、湍流分布建模、强化学习导航）中展示其作为统一预测‑生成原语的能力。

**💡 创新点**

将加法、二次、张量分解等可分离结构统一为一种基于张量秩与交互阶控制的表征类，允许在单一轻量网络中同时实现点预测、逆推、变分学习与生成，显著提升模型解释性与通用性。

**🔧 技术方法**

采用CP/张量分解、B‑样条子原子、变分Galerkin训练、最大后验MAP逆推、Prefix‑LM 约束的 Transformer、SNA 变体（KHRONOS、SPAN、Janus、Leviathan）等技术。

**📊 数据集**

使用 Inconel 718 热历史–机械性能数据、L‑BOM 3D 微结构弹性/渗透率数据、PDEBench 2D 惰性湍流数据，以及 Sketch‑to‑stress 微结构‑应力数据集。

**📈 对比分析**

与 CNN、ResNet、XGBoost、PINN、FNO、DeepONet、U‑Net、Dense Transformer、传统 MLP 等基线比较；SNA 相关模型在参数量 1–3 orders 量级更少、预测/逆推误差显著下降、推断速度 <50 ms、强化学习样本效率提升 30‑50%、微结构生成误差 <3.5%、湍流分布无平均漂移并保持能谱，整体性能优于传统方法。

**⚠️ 局限性**

仍需人工寻找可分离坐标/令牌化方案；可分离结构在某些系统中不自然显现；高阶交互与张量秩选择需要经验调优；对离散序列的适用有限；生成逆推易产生梯度幻觉；在数据稀疏或高压缩比场景下性能可能下降。

---

## 31. HATS: Hardness-Aware Trajectory Synthesis for GUI Agents

**arXiv ID:** 2603.12138 | [PDF](https://arxiv.org/pdf/2603.12138v1)

**作者:** Rui Shao `[一作]` (Harbin Institute of Technology), Gongwei Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5007109976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于语义难度的闭环轨迹合成框架（Hardness‑Aware Trajectory Synthesis）来训练图形用户界面（GUI）代理

**💡 创新点**

创新点在于：①将语义模糊度作为“难度”指标引入探索；②采用硬度驱动的蒙特卡洛树搜索（HD‑MCTS）实现探索与对齐校正的闭环；③多轮对齐校正将生成的指令与执行轨迹反复检验并修正；④用指令‑执行对齐度构造可调节的硬度奖励，自动引导数据采样到高价值语义模糊交互

**🔧 技术方法**

核心技术包括：大视觉‑语言模型（VLM）进行提示式指令生成与执行；HD‑MCTS（包含选择、扩展、仿真、回传四个阶段）；多轮对齐校正（包括子轨迹选择、指令合成、执行回放、相似性验证、迭代修正）；行动级重构召回（Action‑Level Reconstruction Recall）用作对齐度量；硬度奖励函数将召回反转为探索信号

**📊 数据集**

使用公开的两大 GUI 任务基准：AndroidWorld（116 任务）和 WebArena（Gitlab、Maps、Reddit 三个域）

**📈 对比分析**

与零样本、任务驱动、Self‑Instruct、OS‑Genesis 等四类现有合成策略比较；在所有后端 VLM（InternVL2‑4B/8B、Qwen2‑VL‑7B）上实验，均显示显著提升：AndroidWorld 总体成功率从 11.30%（OS‑Genesis）提升至 22.60%；WebArena 总体成功率从 6.53%（OS‑Genesis）提升至 24.87%，在难度较高的 P&W 与 Gitlab 等类别获得最高增幅

**⚠️ 局限性**

局限性包括：①仍需依赖强大 VLM 和大量算力进行多轮对齐校正；②硬度奖励设计对参数敏感，可能对极端情形表现欠佳；③对语义模糊度的定义仍以行动级召回为准，可能忽略更细粒度的语义不一致；④仅在公开基准上验证，真实世界多样性的推广性尚待进一步考察

---

## 32. Can RL Improve Generalization of LLM Agents? An Empirical Study

**arXiv ID:** 2603.12011 | [PDF](https://arxiv.org/pdf/2603.12011v1)

**作者:** Zhiheng Xi `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 16801 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的强化学习微调（RFT）进行系统研究，评估其在同一环境内不同任务难度、跨环境、以及多环境顺序训练中的泛化与迁移能力。

**💡 创新点**

首次从任务难度、环境差异和训练顺序三维度全面分析RFT的泛化特性，展示了课程学习、顺序训练的抗遗忘效应，并结合失效模式和案例研究阐明RFT的迁移局限。

**🔧 技术方法**

采用ReAct交互范式与策略梯度（GRPO）优化，在Qwen2.5-3B/7B-Instruct模型上执行强化学习微调，并使用AgentGym-RL框架进行训练与评估。

**📊 数据集**

使用AgentGym提供的五个环境（WebShop、SearchQA、TextCraft、AlfWorld、BabyAI）中的任务数据，按难度分为easy/hard，并采样8条轨迹进行训练。

**📈 对比分析**

将RFT模型与基线（未微调）进行对比，使用精确匹配成功率、平均交互回合数、生成token数等指标；结果显示RFT在同一环境内可提升60+分（WebShop hard），跨环境可实现平均3~4分的提升；顺序训练保持上游性能并实现下游性能提升，且与联合训练相当。

**⚠️ 局限性**

RFT泛化在持有环境与未持有环境之间存在显著差距，对背景知识、观测/动作空间差异敏感；在某些环境（如BabyAI、AlfWorld）出现负迁移；失效模式表明确认偏误、假信息生成等问题；训练顺序对跨环境泛化有影响，且长时间顺序训练易导致部分环境的遗忘。

---

## 33. GPT4o-Receipt: A Dataset and Human Study for AI-Generated Document Forensics

**arXiv ID:** 2603.11442 | [PDF](https://arxiv.org/pdf/2603.11442v1)

**作者:** Yan Zhang `[一作]`, Evelyn Marotta `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建 GPT4o-Receipt benchmark，包含 1,235 张收据图像（935 张 GPT-4o 合成，300 张真实），并通过人类 30 名标注者与 5 个多模态 LLM（Claude Sonnet 4、Gemini 2.5 Flash、GPT‑5 Nano、Grok 4、LLaMA 4 Scout）进行零样本检测与对比。

**💡 创新点**

①首次提出针对 AI 生成财务文档（收据）的完整数据集与多维度评估框架；②发现人类在视觉辨别上领先但在二分类 F1 低于 LLM，揭示视觉‑算术不对称；③通过对抗“算术硬化”实验验证检测器对多维信号的鲁棒性。

**🔧 技术方法**

利用 GPT‑4o 生成文本内容，再用 GPT‑Image‑1 渲染图像；采用 5 种多模态 LLM 进行零样本推理检测；结合 crowdsourced 人工视觉评分；使用统计 Bootstrap CI、Recall‑FPR 曲线等评估方法。

**📊 数据集**

GPT4o-Receipt 数据集：159 种商户类别，935 张 GPT‑4o 合成收据与 300 张来自 ExpressExpense 与 Roboflow Universe 的真实收据；全部图片 PNG（合成）或 JPEG（真实）格式。

**📈 对比分析**

采用二分类指标（准确率、F1、召回率、FPR）进行对比；Claude Sonnet 4 最高 F1 0.975（召回 0.972，FPR 0.070），Gemini 2.5 Flash F1 0.890（召回 0.807，FPR 0.023），人类 F1 0.852（召回 0.770，FPR 0.120）。Grok 4 召回近 1 但 FPR 0.903；LLaMA 4 Scout 召回 0.114 但 FPR 0.017；阈值敏感度实验表明人类最佳阈值为 ≤3。

**⚠️ 局限性**

仅基于 GPT‑4o 的合成，难以推广到其它生成模型；样本主要为英文北美/英国收据，缺乏多语言与非西方格式；LLM 评估缺乏独立算术验证，存在模型偏差；人类二分类结果为阈值推断而非直接判断；模型性能随 API 更新可能不可复现。

---

## 34. A Causal Approach to Predicting and Improving Human Perceptions of Social Navigation Robots

**arXiv ID:** 2603.11290 | [PDF](https://arxiv.org/pdf/2603.11290v1)

**作者:** Maximilian Diehl `[一作]` (Chalmers University of Technology), Marynel Vázquez `[通讯]` (Yale University)

**通讯引用:** 1573 | [OpenAlex ID](https://openalex.org/A5041256504)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现一个因果贝叶斯网络，用于在线预测移动机器人在导航任务中被人类感知的能力与意图，并生成可解释的、能够提升感知能力的对抗性轨迹；

**💡 创新点**

创新点在于：①用因果模型替代关联学习提高解释性；②将时间序列聚类与离散化相结合处理连续变量；③基于因果网络的组合搜索自动生成提升感知的对抗轨迹；

**🔧 技术方法**

使用的技术包括因果贝叶斯网络、K-means时间序列聚类、离散化、宽度优先搜索、线性混合效应模型等；

**📊 数据集**

使用的数据集为SEAN Together机器人跟随任务数据集（约2964条交互样本，60名受试者）；

**📈 对比分析**

通过留一交叉验证与随机森林基线对比，F1‑score分别提升至0.78/0.75，准确率提升约2–3%；在线用户实验中，对抗轨迹将感知能力提升了83%（正确预测）或27%（错误预测）；

**⚠️ 局限性**

局限性包括：数据量有限导致需人工指定图结构、忽略墙壁和行人等环境因素、生成轨迹未考虑人类追随者的实时反应、模型假设无未观测混杂变量。

---

## 35. Distributed Kalman--Consensus Filtering with Adaptive Uncertainty Weighting for Multi-Object Tracking in Mobile Robot Networks

**arXiv ID:** 2603.11328 | [PDF](https://arxiv.org/pdf/2603.11328v1)

**作者:** Niusha Khosravi `[一作]` (Instituto Superior Técnico), Meysam Basiri `[通讯]` (Instituto Superior Técnico)

**通讯引用:** 591 | [OpenAlex ID](https://openalex.org/A5046287015)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现并评估了基于MOTLEE框架的分布式卡尔曼-共识滤波器，并在其上加入自适应不确定性加权机制，用于多目标跟踪。

**💡 创新点**

将不确定性感知的自适应加权整合进分布式共识滤波，使机器人在异质定位不确定性条件下能够动态调整邻居信息权重，从而提升跟踪鲁棒性。

**🔧 技术方法**

ROS实现、2D LiDAR+DBSCAN聚类、常数速度模型Kalman滤波、GNN+Hungarian关联、分布式Kalman-consensus滤波、基于定位不确定性的自适应权重。

**📊 数据集**

在Gazebo仿真环境中使用两台差速驱动机器人和四个圆柱形移动目标的模拟数据，地图由Gmapping实时构建。

**📈 对比分析**

通过对比标准共识与自适应加权两方案，使用MOTA指标评估。对高定位不确定性机器人，MOTA平均提升约+0.085（局部）和+0.093（全局）；对低不确定性机器人，MOTA略降约-0.10，表明权重策略在不确定性差异下能显著提升弱节点性能，但对强节点略有抑制。

**⚠️ 局限性**

过于保守的权重策略降低了良好定位机器人的合作收益；常数速度模型无法捕捉急转弯导致的误差；对动态锚点的帧对齐高度依赖，稀疏环境下易失效；通信延迟仍会导致信息时效性不足，未能完全消除时延影响。

---

## 36. Just Use XML: Revisiting Joint Translation and Label Projection

**arXiv ID:** 2603.12021 | [PDF](https://arxiv.org/pdf/2603.12021v1)

**作者:** Thennal D K `[一作]` (University of Hamburg), Hans Ole Hatzel `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LabelPigeon，利用 XML 标签在一次前向过程中完成跨语言翻译和标签投射；

**💡 创新点**

创新点在于使用 XML 标记代替方括号实现嵌套/重叠标签的无缝投射，并通过高质量并行 XML 数据对模型进行联合细调，从而在不牺牲翻译质量的前提下显著提升投射准确率；

**🔧 技术方法**

技术包括 NLLB‑200 3.3B 的细调、XML 标记插入、Ratcliff/Obershelp 相似度 F1 评估、BLEU/chrF++ 和 COMET 翻译质量测量以及对 Flores‑200 的合成标签实验；

**📊 数据集**

使用的主要数据集有 Salesforce Localization XML MT、XQuAD、MLQA、UNER、CorefUD、MLQA 以及 Flores‑200；

**📈 对比分析**

与 Awesome‑align、Gemma‑3、EasyProject 等基线比较，LabelPigeon 在直接标签投射 F1、翻译质量（COMET/BLEU/chrF++）上均优；在 NER、CR、QA 等下游任务中平均提升约 10–15 F1，低资源语言尤为显著（如 Cebuano +30、Tagalog +39）；

**⚠️ 局限性**

局限性包括评估依赖已平行且标注的 QA 数据，可能受翻译ese 影响；合成标签插入可能不完全符合真实应用；未对最新方法 Codec、CLaP 进行完整评估；在核心ference 任务上效果相对有限。

---

## 37. Resurfacing Paralinguistic Awareness in Large Audio Language Models

**arXiv ID:** 2603.11947 | [PDF](https://arxiv.org/pdf/2603.11947v1)

**作者:** Hao Yang `[一作]` (Monash University), Gholamreza Haffari `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对两种大型音频语言模型（Qwen2.5‑Omni、Kimi‑Audio）进行层级分析，识别出包含声学特征（0‑6层）和语义理解（7‑14层）的层次；基于此设计了选择性微调与辅助双层分类头（ADCH）的 Paralinguistic‑Enhanced Fine‑Tuning（PE‑FT）协议，显著提升模型对年龄、性别、情感等旁语特征的感知与利用，从而改善与儿童的安全交互。

**💡 创新点**

创新点包括：①首次将儿童安全议题引入音频语言模型的安全评估；②提出五种层级分析（三类旁语线性探测、意图分类、余弦相似度、年龄相似度、Logit lens）共同定位旁语与语义层；③设计PE‑FT协议，利用选择性微调和 ADCH 以参数高效的方式重现旁语意识；④引入 PA‑score 与 PA‑rate 两个专门衡量旁语意识的评测指标。

**🔧 技术方法**

技术核心包括：层级线性探测、意图分类探测、余弦相似度分析、Logit lens、LoRA 适配器、交叉熵联合损失、辅助双层分类头、t‑SNE 可视化。

**📊 数据集**

数据集：①基于 GPT‑4.1 生成的 1500 条文本（每类 500 条），配以 9000 条合成语音（儿童/成人、男女、13 种情绪）；②评测集 1200 条语音（200 条文本 × 6 语音变体）包括儿童安全样本；③额外的 70 条儿童安全情境样本用于安全评估；④使用 Typecast、Google TTS、gpt‑4o‑mini‑tts 等合成引擎生成多种旁语属性。

**📈 对比分析**

与全层微调（0‑27层）和仅语义层微调（7‑14层）对比，PE‑FT 在三类旁语指标上均取得更高 PA‑score（年龄 0.96→0.965，性别 0.97→0.985，情感 0.46→0.503）和 PA‑rate（大幅提升至 97‑98%），同时保持或略低于原模型的整体内容质量（ParaS2S 分数）且 VoiceBench 帮助度无显著下降；儿童安全评估中 PA‑rate 从 4–7% 跃升至 97‑99%。

**⚠️ 局限性**

局限性：①性别处理仅为二元生物学分类，忽视自我认同与性别多样性；②模型对性别信号的依赖较强，导致对新性别说话者的泛化能力下降；③实验依赖合成语音，真实人声分布与合成差异可能影响效果；④PE‑FT 需要额外的辅助头与标注，增加训练复杂度。

---

## 38. Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing

**arXiv ID:** 2603.12254 | [PDF](https://arxiv.org/pdf/2603.12254v1)

**作者:** Baifeng Shi `[一作]` (University of California Berkeley), Hongxu Yin `[通讯]` (NVIDIA)

**通讯引用:** 2675 | [OpenAlex ID](https://openalex.org/A5002444694)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级的自回归视频盲扫（gazing）模块，在输入前预先去除冗余视觉补丁，从而显著降低ViT和多模态大型语言模型（MLLM）的计算量和显存需求。

**💡 创新点**

创新点在于（1）将多尺度补丁的自回归选择与自动停止机制相结合，实时决定最小补丁集合；（2）通过预训练的next‑token预测与强化学习后训练，学习在满足重构误差阈值的前提下最优补丁序列；（3）实现了高分辨率（4K）长视频（最高1K帧）在MLLM中的高效推理。

**🔧 技术方法**

技术手段包括：卷积编码器 + 自回归Transformer解码器（3M参数）；多尺度补丁词表；重构损失预测头；greedy搜索收集的盲扫序列用于NTP预训练；基于GRPO的RL后训练；以及与ViT的多尺度输入适配和视频序列化处理。

**📊 数据集**

使用了800K多样化原始视频数据（视角、自然、文本丰富等），从中提取26.5K条盲扫序列；并构建了首个高分辨率长视频QA基准“Gray!20”，包含268个5分钟、4K分辨率视频问答；此外在多项公开基准（VideoMME、MVBench、LongVideoBench等）上评测。

**📈 对比分析**

与多种基线（随机盲扫、RGB/光流盲扫、S/T/ST 词元裁剪方法、现有MLLM token pruning方案）对比，Gray!20在ViT和LLM推理上分别实现高达19×和10×的加速；在视频问答任务上，基于NVILA-8B-Video的模型在Gray!20上提升10.1%准确率，超过了公开SOTA模型Qwen2.5-VL-7B与GPT‑4o，显示出显著的性能与效率优势。

**⚠️ 局限性**

局限性包括：盲扫模块在极端高动态场景或非典型视频内容下仍可能漏检细节；当前多尺度补丁集成对ViT的适配依赖手工调整；以及在极高帧率/分辨率场景中，仍需进一步优化参数以保持实时性。

---

## 39. QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions

**arXiv ID:** 2603.12165 | [PDF](https://arxiv.org/pdf/2603.12165v1)

**作者:** Jiayin Lei `[一作]` (Beijing University of Technology), Tianming Yang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2773 | [OpenAlex ID](https://openalex.org/A5000086091)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出QAQ框架，利用答案对问题的逆向互信息RMI来挑选高质量的代码生成训练样本。

**💡 创新点**

创新点在于从逆向（Q|A）角度评估语义一致性，并结合强弱模型的认知差异（Diff）筛选既有效又有挑战性的样本。

**🔧 技术方法**

采用逆向困惑度（PPL(Q|A)）计算RMI，对RMI做分层归一化，使用强模型（DeepSeek-Coder-6.7B）和弱模型（Qwen3-0.6B）计算Diff，最终按阈值选取样本。

**📊 数据集**

以WarriorCoder合成数据集（约310K条指令-响应对）为实验基准。

**📈 对比分析**

与IFD、RDS+、SCAR等传统选择方法对比，25% 的RMI 50‑75% 区间样本在 HumanEval+、MBPP+ 上可达到或超过全量训练性能，且显著优于随机、IFD 等基线。

**⚠️ 局限性**

局限包括仅在单一数据集验证、对强弱模型选择敏感、超参数（K、阈值）未充分调优、未提供多次随机种子验证、RMI 计算开销较大。

---

## 40. The Network That Thinks: Kraken* and the Dawn of Cognitive 6G

**arXiv ID:** 2603.11920 | [PDF](https://arxiv.org/pdf/2603.11920v1)

**作者:** Ian F. Akyildiz `[一作]` (Truva Inc), Tuğçe Bilen `[通讯]` (Istanbul Technical University)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5066357879)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出一种名为 Kraken 的三层知识驱动架构，旨在将语义通信、生成式推理和目标导向网络优化集成到未来 6G 网络中，实现分布式集体智能。

**💡 创新点**

创新点在于将语义通信、生成式世界模型与目标导向决策三项能力统一进一个可扩展的三平面体系，并与 O-RAN、网络数字孪生和 MLOps 结合，为 6G 生态提供可逐步演进的实现路径。

**🔧 技术方法**

采用语义-aware PHY/MAC/网络层、分布式 Generative Network Agent（GNA）、共享知识图谱、基础模型、意图仓库等技术；实现平台基于 Open RAN，验证平台使用网络数字孪生，模型生命周期管理通过 MLOps pipeline。

**📊 数据集**

该工作为概念性架构，未使用公开数据集或实验数据，主要通过理论分析和案例场景（自动驾驶、XR、桥梁监测）来说明设计思路。

**📈 对比分析**

文中未给出具体对比实验或量化指标；仅通过案例讨论指出在语义压缩、延迟降低和资源利用率方面潜在提升，但缺乏实测性能评估。

**⚠️ 局限性**

局限性包括：缺乏实际部署与实验验证；对大规模多代理协同与意图对齐的算法细节不完整；对安全、隐私与标准化兼容性的讨论不足；模型复杂度和硬件实现的可行性待进一步研究。

---

## 41. Agentic AI for Embodied-enhanced Beam Prediction in Low-Altitude Economy Networks

**arXiv ID:** 2603.11392 | [PDF](https://arxiv.org/pdf/2603.11392v1)

**作者:** Min Hao `[一作]` (South China Normal University), Rong Yu `[通讯]` (Guangdong University of Technology)

**通讯引用:** 10334 | [OpenAlex ID](https://openalex.org/A5100659097)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了面向低空经济网络的无人机毫米波链路主动波束预测方案，结合多代理AI架构与混合多模态预测模型，实现了基于无人机移动信息与视觉数据的实时波束推断。

**💡 创新点**

创新点包括：①将波束预测拆解为任务分析、方案规划与完整性评估三步，构建多代理协同推理框架；②设计基于Mamba、ResNet与跨注意力的混合模型，实现数值与图像特征的动态融合；③通过代理间交互动态切换数据流策略，提升在非平稳环境下的鲁棒性。

**🔧 技术方法**

技术手段涵盖：多代理LLM（ReAct范式）推理、Mamba时序建模、ResNet视觉编码、跨模态注意力融合、Transformer解码以及多模态决策触发机制。

**📊 数据集**

使用真实无人机与毫米波基站的数据集DeepSense6G（Tuen Park, Arizona），包含RGB图像、GPS/IMU等数值特征以及64波束码书的实际测量。

**📈 对比分析**

与单模态及传统学习式预测方法对比，混合模型在不同编码层数与注意力头数下，平均Top‑1准确率最高可达96.57%，并在不同时间窗口（3/5/10 s）内表现出更高的对角占比与更低的推断延迟，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①多代理推理依赖大模型，推理延迟与算力开销较高；②模型对训练数据量依赖强，需大量标注样本；③未考虑持续学习与轨迹优化的进一步协同，可能在极端动态场景下仍有误差。

---

## 42. SPARK: Skeleton-Parameter Aligned Retargeting on Humanoid Robots with Kinodynamic Trajectory Optimization

**arXiv ID:** 2603.11480 | [PDF](https://arxiv.org/pdf/2603.11480v1)

**作者:** Hanwen Wang `[一作]` (University of Wisconsin-Madison), Xiaobin Xiong `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种两阶段流水线，将任务空间的人类运动转化为符合机器人运动学和动力学的自然运动参考，首先通过将人类运动转换为URDF骨架并校准尺寸来提高逆运动学质量；随后通过分阶段的运动优化（先运动学TO，再逆动力学，最后完整的动力学TO）生成动态可行的轨迹及扭矩参考，供强化学习控制器训练使用。

**💡 创新点**

创新点在于：①基于URDF的全尺寸校准，直接对骨架尺寸进行物理可解释的匹配，显著降低逆运动学调参；②分阶段的动力学运动优化框架，逐步加入动力学约束，解决高维非线性问题；③将优化得到的扭矩参考作为强化学习的额外监督，提升学习效率与跟踪性能。

**🔧 技术方法**

技术包括：人类骨架到URDF的生成与校准；逆运动学求解；分阶段运动优化（运动学TO、逆动力学QP、完整动力学TO）；强化学习（BeyondMimic）用于跟踪训练；物理约束如接触约束、碰撞避免、扭矩限制。

**📊 数据集**

使用公开的人类动作数据集 AMASS（包括 ACCAD 子集）进行评估，并在多种机器人平台（Unitree G1、H1、Booster T1、EngineAI PM01、Kuavo 4Pro）上进行实验。

**📈 对比分析**

与传统 GMR 逆运动学方法对比，URDF 校准在多机器人平台上将平均位姿误差降低约 65–75%。在动作编辑（跳跃）和高动态动作（侧翻）上，采用 KDTO 优化后的轨迹在强化学习训练中显著加速收敛，最终跟踪误差明显低于仅使用 KTO 或原始编辑参考。

**⚠️ 局限性**

局限性包括：仍需要针对不同机器人进行 URDF 校准参数选择；分阶段优化耗时较长，对实时控制不适合；仅在已知接触序列和标定的场景下有效，无法处理未知接触或动态环境变化。

---

## 43. Frequentist Consistency of Prior-Data Fitted Networks for Causal Inference

**arXiv ID:** 2603.12037 | [PDF](https://arxiv.org/pdf/2603.12037v1)

**作者:** Valentyn Melnychuk `[一作]` (LMU Munich), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2421 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对先验拟合网络（PFN）在因果推断中的频繁主义一致性进行理论分析，并提出一阶后验校正（OSPC）结合马尔可夫后验（MP）的方法，使 PFN 的 ATE 估计既能保留先验信息，又能满足 Bernstein–von Mises 定理，实现频繁主义一致性。

**💡 创新点**

① 发现 PFN 受先验诱导的混杂偏差影响，导致频繁主义一致性失效；② 提出基于效率影响函数的一阶后验校正（OSPC），可在不重新训练的前提下校正偏差；③ 通过 Copula‑基马尔可夫后验重构功能后验，实现对 PFN 产生的后验分布的充分利用。

**🔧 技术方法**

使用 PFN（TabPFN、CausalPFN、CausalFM）作为基础模型；马尔可夫后验（MP）与 Copula 结合以恢复功能后验；一阶后验校正（OSPC）；A‑IPTW 作为频繁主义参考；贝叶斯自举、效率影响函数等统计工具。

**📊 数据集**

半合成数据（样本量 n 与特征维度 d_x 可调，Δ≈-3.4）、IHDP 实验数据（n≈747, d_x=25, Δ≈-0.2）、ACIC 2016 77 组半合成数据集（n≈4802, d_x=82, Δ 变化）以及 COVID‑19 阻断政策真实案例。

**📈 对比分析**

将 MP‑OSPC 与传统 A‑IPTW、无校正 PFN、MP‑插件等基线在 TV（总变差）和 KS（Kolmogorov–Smirnov）指标上进行比较。结果显示 MP‑OSPC 在大多数实验中在 TV 和 KS 上均优于基线，逼近 A‑IPTW 的不确定性分布，并在有限样本下实现更好的校准。

**⚠️ 局限性**

① PFN 近似贝叶斯，严格的 BvM 证明仅在第二阶剩余项可忽略时成立；② 在大样本或高维场景下 R₂ 可能增大，导致校正效果下降；③ 需要引入 Copula 参数（ρ）和 MP 步数，增加计算成本；④ 对极端倾向分数（接近 0 或 1）的估计仍有限。

---

## 44. PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents

**arXiv ID:** 2603.11955 | [PDF](https://arxiv.org/pdf/2603.11955v1)

**作者:** Minjia Wang `[一作]` (Harvard University), Zheng Sun `[通讯]` (Apple)

**通讯引用:** 1303 | [OpenAlex ID](https://openalex.org/A5101651739)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM代理的PersonaTrace框架，能够从人口统计分布生成完整的用户画像、日常事件树和对应的数字足迹（邮件、聊天、日历等），并提供高质量的合成数据集。

**💡 创新点**

首次提出端到端的代理驱动合成数字足迹流水线，利用Persona、Event、Artifact代理和Critic循环实现多模态、一致性和真实度的合成；并通过人类评测与下游任务验证其多样性与实用性。

**🔧 技术方法**

使用大型语言模型（Gemini‑1.5‑Pro）作为核心生成引擎，结合embedding检索、MinHash LSH、递归事件扩展、生成‑评判循环以及多种多模态生成策略。

**📊 数据集**

基于美国社区调查（ACS 2022）作为人口先验，PersonaHub 作为事件记忆来源；与八个现有合成数据集及私有真实数据进行对比评估。

**📈 对比分析**

通过内在指标（多样性、真实度、LLM-as-Judge评分）与外在指标（邮件分类、撰写、问答、下一条信息预测四个下游任务）进行比较；PersonaTrace在所有评测中均表现优于其他合成数据集，尤其在多模态多任务上的泛化能力最强。

**⚠️ 局限性**

缺乏对事件主题/意图的可控性；目前代理主要依赖LLM先验，难以限制生成内容到特定话题，且对特殊业务场景的可定制性有限。

---

## 45. CHiL(L)Grader: Calibrated Human-in-the-Loop Short-Answer Grading

**arXiv ID:** 2603.11957 | [PDF](https://arxiv.org/pdf/2603.11957v1)

**作者:** Pranav Raikote `[一作]` (Stockholm University), Panagiotis Papapetrou `[通讯]` (Stockholm University)

**通讯引用:** 2406 | [OpenAlex ID](https://openalex.org/A5044999523)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Calibrated Human-in-the-Loop (HiL) 框架，结合温度校准、置信度选择性预测和持续学习，实现短答自动评分时将高置信度样本自动化，低置信度样本交给教师处理；

**💡 创新点**

创新点在于将置信度校准、选择性推理与持续学习三者集成到同一流水线，解决LLM过度自信和分布漂移问题；

**🔧 技术方法**

主要技术包括指令调优（instruction‑tuning）、后置温度缩放（temperature scaling）校准、置信度门控的选择性预测、LoRA适配器微调与重放缓冲的持续学习；

**📊 数据集**

实验使用三大公开短答评分数据集：DAMI（数据挖掘，5/8/10级别）、SciEntsBank（小学科学，0–4级别）和EngSAF（工程学，0–2级别）；

**📈 对比分析**

与零样本、少样本、RAG、LoRA指令调优等基线对比，Calibrated HiL 在各数据集上均实现了QWK≥0.80的专家级评分，并在35–65%样本上实现自动化；

**⚠️ 局限性**

局限包括需人工反馈来驱动持续学习、对不同学科和评分尺度需要重新校准温度、在覆盖率与准确度间需权衡，且在极少样本或极大分布漂移场景下效果可能受限。

---

## 46. DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering

**arXiv ID:** 2603.11798 | [PDF](https://arxiv.org/pdf/2603.11798v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6603 | [OpenAlex ID](https://openalex.org/A5062243169)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DocSage端到端的agentic框架，解决多文档多实体问答中的跨文档逻辑推理问题。

**💡 创新点**

通过交互式模式动态发现查询特定的最小可连接模式，结合错误感知的结构化抽取和基于模式的关系推理，实现了结构化数据驱动的高精度推理。

**🔧 技术方法**

使用LLM与LoRA校准的抽取器、ASK算法的交互式模式发现、CLEAR逻辑一致性校正以及SQL查询编译执行等技术。

**📊 数据集**

在MEBench和Loong两个MDMEQA基准上进行评估。

**📈 对比分析**

相较于GPT‑4o、标准RAG、GraphRAG和StructRAG，DocSage在MEBench上提升约27%准确率，在Loong上平均得分和完全正确率均远超对手，显示出显著优势。

**⚠️ 局限性**

计算成本高、依赖底层LLM性能，且对高度噪声或对立文本的鲁棒性有限。

---

## 47. Quality-Driven Agentic Reasoning for LLM-Assisted Software Design: Questions-of-Thoughts (QoT) as a Time-Series Self-QA Chain

**arXiv ID:** 2603.11082 | [PDF](https://arxiv.org/pdf/2603.11082v1)

**作者:** Yen-Ku Liu `[一作]` (National Taiwan Normal University), Yun-Cheng Tsai `[通讯]` (National Taiwan Normal University)

**通讯引用:** 349 | [OpenAlex ID](https://openalex.org/A5008816013)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 Questions‑of‑Thoughts (QoT) 框架，利用序列化的工程步骤和逐步自问自答的推理链来提升 LLM 在软件设计中的质量，特别关注可扩展性、完整性、模块化和安全性。

**💡 创新点**

创新点在于将软件质量维度嵌入推理流程，通过自问自答显式捕获设计约束并在推理知识库中积累，以此形成可审计、可复用的设计轨迹。

**🔧 技术方法**

采用 LLM 推理（如 Gemini 3.1 Pro、LLaMA 系列）、自问自答链、ISO/IEC 25010 为基础的质量打分指标，以及自动化 LLM‑judge 评估框架。

**📊 数据集**

构建了涵盖 API 设计、数据通信与文件系统三大后端工程领域的基准任务，任务中包含多模块分解与安全/错误处理等实际工程关注点。

**📈 对比分析**

通过与 NoQoT（无自问自答）和 CoT（标准链式思维）对照实验，测量每个维度的四级评分，结果显示在大型模型与复杂域中 QoT 能显著提升质量得分；在小模型中提升幅度有限，但仍有优势；在文件系统域出现轻微的“过度思考”导致得分下降。

**⚠️ 局限性**

局限性包括额外推理步骤带来的推理延迟、对模型多步规划能力的依赖导致在小模型上效果不稳定，以及无法完全消除错误或幻觉，需结合静态分析与自动化验证进一步提升可靠性。

---

## 48. Teleodynamic Learning a new Paradigm For Interpretable AI

**arXiv ID:** 2603.11355 | [PDF](https://arxiv.org/pdf/2603.11355v1)

**作者:** Enrique ter Horst `[一作]`, Juan Diego Zambrano `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出“Teleodynamic Learning”范式，将学习视为在受约束的动力学系统中结构、参数和资源共同演化，而非传统的目标函数最小化

**💡 创新点**

核心创新在于：① 双时间尺度的内外动态耦合；② 内部能量变量自我生成并调控结构变化；③ 通过本地Teleodynamic目标实现自发结构停止和三相阶段（欠结构→增长→过度结构）

**🔧 技术方法**

使用Spencer‑Brown Laws of Form构建逻辑结构，信息几何中的自然梯度进行参数更新，基于Coalgebraic语义的状态转移，配合能量和复杂度惩罚的本地目标函数

**📊 数据集**

在UCI标准分类数据集上实验：IRIS、WINE、Breast Cancer、DIGITS

**📈 对比分析**

与Logistic Regression、Decision Tree、Random Forest、SVM、MLP等基线对比；在IRIS、WINE、Breast Cancer上DE11实现93.3%、92.6%、94.7%的准确率，接近或略低于最优基线，同时生成可解释的逻辑规则；在DIGITS上表现相对逊色，显示规模与特征维度受限

**⚠️ 局限性**

限制包括：对Fisher信息矩阵仅采用对角近似；结构动作空间有限（仅Genesis、Wedge、noop），难以应对高维复杂任务；能量和复杂度系数需手工调参，缺乏自适应机制；在大规模或高维数据上可解释性与推理效率下降

---

## 49. Gen-Fab: A Variation-Aware Generative Model for Predicting Fabrication Variations in Nanophotonic Devices

**arXiv ID:** 2603.11505 | [PDF](https://arxiv.org/pdf/2603.11505v1)

**作者:** Rambod Azimi `[一作]` (McGill University), Odile Liboiron-Ladouceur `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为Gen‑Fab的条件生成对抗网络，用于从硅光子器件的GDS布局预测多样化的SEM图像，捕捉工艺变异并实现数字孪生；

**💡 创新点**

在Pix2Pix框架中重新引入噪声向量到生成器的瓶颈，实现一对多映射，能够直接学习并采样真实工艺变异分布；

**🔧 技术方法**

采用条件GAN（Pix2Pix）+噪声注入+PatchGAN判别器+L1重构损失+Adam优化+数据增强与混合精度训练；

**📊 数据集**

使用ANT‑NanoSOI数据集（31对高分辨率GDS‑SEM图像，2048×2048），通过旋转等方式扩充至124对训练集；测试集为6种未见的结构，每种有35个SEM样本；

**📈 对比分析**

与三种基线（确定性U‑Net、MC‑Dropout U‑Net、Ensemble U‑Nets）比较，使用IoU、KL‑Divergence和Wasserstein距离评估；Gen‑Fab平均IoU达89.8%，优于U‑Net 85.3%、MC‑Dropout 80.7%、Ensemble 85.8%，KL‑D和W‑D亦显著更低，证明分布匹配更好；

**⚠️ 局限性**

局限性包括：目前仅处理单层硅光子结构，未涵盖多层或更复杂器件；需要大量高分辨率SEM数据；GAN训练可能受模式坍塌或不稳定影响；噪声向量的可解释性与控制尚未充分；在更大域外场景中的泛化仍需进一步验证。

---

## 50. IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse

**arXiv ID:** 2603.12201 | [PDF](https://arxiv.org/pdf/2603.12201v1)

**作者:** Yushi Bai `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14763 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过跨层重用索引器计算，加速深度稀疏注意力模型的推理速度。

**💡 创新点**

首次将索引器的跨层冗余性与稀疏注意力结合，提出训练‑无关贪婪搜索与训练‑有关多层蒸馏两种优化策略。

**🔧 技术方法**

利用DeepSeek Sparse Attention的轻量级索引器、贪婪层次搜索、交叉层蒸馏损失以及多头稀疏注意力机制。

**📊 数据集**

在GLM‑4.7‑Flash（30B）和GLM‑5（744B）上使用长文本、链式推理与通用推理十个公开基准（如MRCR、LongBench、AIME、GPQA、LiveCodeBench等）进行评估。

**📈 对比分析**

与原始DSA相比，IndexCache在保持语言建模精度（Long Avg≈50、G&R Avg≈74）下，预填充速度提升至1.82×、解码速度提升至1.48×，且在GLM‑5上仍可达1.3×速度提升。

**⚠️ 局限性**

依赖于索引器输出的稳定性；在极端稀疏率（如1/8）仍会出现性能下降；对非DSA稀疏注意力模型的迁移效果尚待验证。

---

## 51. Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning

**arXiv ID:** 2603.11460 | [PDF](https://arxiv.org/pdf/2603.11460v1)

**作者:** Seung hee Choi `[一作]` (Hanyang University), Dong-Jin Kim `[通讯]` (Hanyang University)

**通讯引用:** 20672 | [OpenAlex ID](https://openalex.org/A5100344647)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种名为 STaRC 的检索增强式密集视频字幕生成框架，利用从 DVC 事件标注直接衍生的高亮标签训练帧级显著性模型，并将该显著性信号用于聚类检索段落和解码器提示，以实现更精准的事件分割和字幕生成。

**💡 创新点**

创新点在于：① 将显著性学习转化为监督任务，直接从现有事件标注生成二值高亮标签，避免额外标注；② 统一显著性作为检索与生成的共享监督信号，构建 Saliency‑Guided Segmentation & Retrieval（SGSR）和 Saliency Prompt（SaliP）两大模块；③ 通过 OT‑based 聚类根据显著性对帧进行事件对齐分割，显著性提示则以显著性得分注入解码器注意力，提升检索质量和字幕相关性。

**🔧 技术方法**

核心技术包括：显著性高亮检测模块（基于 1D CNN/Transformer）、SWSA 视频特征细化、OT（Optimal Transport）聚类实现 SGSR、Saliency Prompt 将显著性嵌入解码器注意力、全流程 Transformer‑based 生成网络。

**📊 数据集**

实验数据集为 YouCook2 与 ViTT 两大工业级 DVC 基准；模型在这两组数据上均完成完整训练、检索与生成过程。

**📈 对比分析**

与现有检索增强方法（如 Sali4Vid、HiCM^2 等）和传统两阶段/端到端 DVC 方法对比，STaRC 在 CIDEr、METEOR、BLEU 等评估指标上均实现 SOTA（比 Sali4Vid 提升约 4–6% 的 CIDEr，METEOR 提升 3–5%）。

**⚠️ 局限性**

局限性：① 显著性预测质量直接决定检索与生成效果，对显著性误判仍敏感；② 只在两大数据集验证，缺乏更广泛跨域评估；③ OT 聚类需要手动调参，聚类性能受参数影响；④ 当前模型仍以视频帧为基本单元，难以处理极长视频或多模态情境。

---

## 52. Faster Relational Algorithms Using Geometric Data Structures

**arXiv ID:** 2603.11402 | [PDF](https://arxiv.org/pdf/2603.11402v1)

**作者:** Aryan Esmailpour `[一作]` (University of Illinois Chicago), Stavros Sintos `[通讯]` (University of Illinois Chicago)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5015883931)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在关系型数据库上设计了更快的聚类算法，尤其针对k‑center/median/means问题；

**💡 创新点**

提出了随机化BBD树（RBBD树）和基于其的关系型查询算子，允许在不完全 materialize join 的前提下按需构建树分支；

**🔧 技术方法**

利用随机采样、计数与报告或acles、BBD树结构以及分层分割与随机中心收缩技术；

**📊 数据集**

主要使用公开的Yelp评论数据集（约800万条记录，合并后约2200万条）；

**📈 对比分析**

与之前的关系型聚类算法相比，时间复杂度从O(k²N)降至O(kN)，并在实验中实现了近乎线性的运行时间，同时保持了相同或更优的近似因子；

**⚠️ 局限性**

方法主要适用于无环（acyclic）join，扩展到一般cyclic查询需额外转换步骤；对高维/大k情况仍有 log 级别的开销，且目前仅验证了聚类任务，对其他优化目标的适用性仍待进一步评估。

---

## 53. Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning

**arXiv ID:** 2603.11346 | [PDF](https://arxiv.org/pdf/2603.11346v1)

**作者:** Yuto Shibata `[一作]` (Keio AI Research Center), Katerina Fragkiadaki `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3429 | [OpenAlex ID](https://openalex.org/A5008661738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种多智能体强化学习框架 AssistMimic，用于在物理仿真中学习支持者与受体的可跟踪控制器，从而实现人–人间的支援与帮助动作。

**💡 创新点**

创新点包括：① 用单人运动追踪先验初始化加速学习；② 动态参考重定向使支持者随受体姿态实时调整；③ 引入接触奖励鼓励真实物理接触，提升鲁棒性。

**🔧 技术方法**

采用深度强化学习（PPO）与物理引擎结合，配合先验权重初始化、动态参考重定向、接触奖励、DAgger 进行多样化策略蒸馏。

**📊 数据集**

使用 Inter-X 与 HHI-Assist 两个真实人–人交互数据集进行训练与评估。

**📈 对比分析**

相较于 Kinematic-Recipient、Frozen-Recipient、Phys-Reaction 等基线，AssistMimic 在 Inter-X 上 83% 的成功率、HHI-Assist 上 66% 的成功率，且对未见动力学表现出显著的鲁棒性，显著优于基线。

**⚠️ 局限性**

局限性包括：缺乏视觉感知和环境感知、尚未在真实机器人上验证、手指抓握能力有限、与高层规划器耦合不够紧密。

---

## 54. Prediction of Grade, Gender, and Academic Performance of Children and Teenagers from Handwriting Using the Sigma-Lognormal Model

**arXiv ID:** 2603.11519 | [PDF](https://arxiv.org/pdf/2603.11519v1)

**作者:** Adrian Iste `[一作]` (Osaka Metropolitan University), Koichi Kise `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了小学至初中学生的手写动态，预测年级、性别和学业成绩。

**💡 创新点**

首次将Sigma‑Lognormal模型参数作为手写特征，并系统比较三类特征在儿童教育任务中的表现。

**🔧 技术方法**

采用在线手写采集、基本统计、熵、Sigma‑Lognormal模型提取特征，并使用线性/逻辑回归与随机森林进行预测。

**📊 数据集**

使用日本Wacom在线手写数据集，约223名学生、每人约110道题目，覆盖9年级。

**📈 对比分析**

通过5折交叉验证比较三类特征，Sigma‑Lognormal特征在年级预测、性别分类和学业表现分类中表现最佳，整体精度处于中等水平。

**⚠️ 局限性**

仅聚合到学生级别，未利用个体笔画或题目级的序列信息；模型未进行大规模调参，预测效果受限。

---

## 55. BEACON: Budget-Aware Entity Matching Across Domains (Extended Technical Report)

**arXiv ID:** 2603.11391 | [PDF](https://arxiv.org/pdf/2603.11391v1)

**作者:** Nicholas Pulsone `[一作]` (Worcester Polytechnic Institute), Gregory Goren `[通讯]` (eBay Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了在标签预算受限的跨域实体匹配任务下，利用分布感知的动态重采样与多模型集成的BEACON框架，用以选择最有助于提升目标域匹配性能的样本。

**💡 创新点**

创新点在于将预训练语言模型的嵌入空间与分布匹配策略（NN、TVDF、KCG）相结合，并在动态训练循环中持续更新样本分布，实现预算感知的跨域样本选择与集成学习的首次融合。

**🔧 技术方法**

技术上使用RoBERTa等预训练语言模型、单词级与对比级嵌入、最近邻/TVDF/KCG分布采样、动态重采样循环以及加权软投票的多模型集成。

**📊 数据集**

实验数据集包括WDC多维度基准、WDC产品语料和Abt-Buy商品数据，全部按产品类别或品牌划分为多个子域，满足跨域低资源场景的评估需求。

**📈 对比分析**

与SPEC、GEN、MFSN、Battleship、PromptEM、LLM（LLaMA）和Jellyfish等基线进行比较，BEACON在宏观和加权F1上在各种预算下平均提升约3–4%，在样本稀缺的域中表现尤为显著。

**⚠️ 局限性**

局限性包括对高维嵌入计算资源的需求、对不同域标签质量和分布一致性的依赖，以及在完全无标签的新域上效果仍受限。

---

## 56. Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**arXiv ID:** 2603.11627 | [PDF](https://arxiv.org/pdf/2603.11627v1)

**作者:** Yichi Zhang `[一作]` (Fudan University), Zixin Hu `[通讯]` (Fudan University)

**通讯引用:** 2540 | [OpenAlex ID](https://openalex.org/A5101743291)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出并实现了一种基于 3D Transformer 的通用 PET 图像分割基础模型 SegAnyPET，并构建了最大的全身 PET 影像分割数据集 PETWB‑Seg11K

**💡 创新点**

创新点包括：① 通过 3D Prompt‑able 结构实现任意目标（器官/病灶）零射频提示分割；② 结合 PET 专属数据实现对 PET 功能信号的特征学习，显著提升跨中心、跨病种、跨探针的泛化能力；③ 提供了多层次模型（通用版与 Lesion‑专用版）满足不同临床需求

**🔧 技术方法**

技术涵盖：3D Transformer‑based Image Encoder、Prompt Encoder（点/掩码）和 Mask Decoder；多尺度数据增强、混合精度分布式训练；交互式迭代提示生成；与 SAM、SAM‑Med3D、SegVol、SAT 等现有基础模型进行对比；与 nnUNet、STUNet 等任务专用模型对照

**📊 数据集**

使用了 11,041 张全身 PET 扫描，59,831 个分割掩码，涵盖 FDG‑PET、PSMA‑PET、PET/CT 与 PET/MRI 等多中心、多探针、多设备数据；内部验证集与外部验证集（UMD‑PETCT、UMD‑PETMR、AutoPET‑PSMA）进一步检验泛化

**📈 对比分析**

与多种 2D/3D 基础模型和任务专用模型对比，SegAnyPET 在内部和外部评估中均获得最高或相近的 Dice/NSD 指标，且在零射频提示下即可完成多目标分割；在交互式工作流中显著降低标注时间（约 82% 节省）

**⚠️ 局限性**

局限性包括：对多分散病灶的点提示交互仍受限，需改进提示策略；在稀有病种、探针或解剖部位的覆盖不足；虽然在多中心验证表现稳健，但仍有提升空间，尤其在细粒度边界精度方面

---

## 57. Grounding Robot Generalization in Training Data via Retrieval-Augmented VLMs

**arXiv ID:** 2603.11426 | [PDF](https://arxiv.org/pdf/2603.11426v1)

**作者:** Jensen Gao `[一作]` (Stanford University), Dhruv Shah `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了检索增强数据分析框架，通过先检索与评估任务相关的训练样本，再让视觉语言模型分析其差异，来系统评估机器人策略的泛化类型。

**💡 创新点**

创新点在于将检索和VLM分析结合成可扩展的两阶段流程，并引入基于-Gen轴的解释性评估。

**🔧 技术方法**

技术上使用了基于机器人策略的VLA嵌入做检索，以及Gemini/ Gemini 3.0/3.1等视觉语言模型进行差异分析。

**📊 数据集**

数据集方面，作者使用了ALOHA 2平台的Pick‑and‑Place、Unzip Lunchbag、Fold Dress三类任务，以及Bridge V2和1M ALOHA 2演示数据。

**📈 对比分析**

在受控实验中，检索召回率超过95%，VLM的轴差异预测准确率约为94%–98%，整体泛化分类准确率高达92%；在大规模数据上，与人工标注的一致性达70%–77%。

**⚠️ 局限性**

局限性包括对细微视觉差异的识别不足、只依赖初始观测和指令导致的误判，以及分类粒度有限。

---

## 58. Decentralized Cooperative Localization for Multi-Robot Systems with Asynchronous Sensor Fusion

**arXiv ID:** 2603.12075 | [PDF](https://arxiv.org/pdf/2603.12075v1)

**作者:** Nivand Khosravi `[一作]` (Instituto Superior Técnico), Masoud S. Bahraini `[通讯]` (University of Birmingham)

**通讯引用:** 392 | [OpenAlex ID](https://openalex.org/A5065109096)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种分布式协作定位框架，使用分布式扩展卡尔曼滤波（EKF）维护机器人间的交叉协方差，支持任意初始参考系、异步传感器采样以及双里程碑观测，实现对GPS‑禁区多机器人系统的高精度定位。

**💡 创新点**

创新点包括：① 事件触发的压缩信息交换，仅在机器人相互可见且通信可用时发送交叉协方差、增益和创新；② 通过测量模型中的变换矩阵实现任意参考系的自动对齐；③ 采用ROS message_filters进行时间戳同步，处理速度不同的编码器与LiDAR数据；④ 双里程碑策略，将静态环境特征与动态机器人自身作为可观测标记，显著提升在特征稀缺环境中的观测连续性。

**🔧 技术方法**

核心技术包括：分布式EKF、交叉协方差传播、变换矩阵测量模型、ROS message_filters、Adaptive Breakpoint Detector（ABD）进行LiDAR分割与圆柱体特征提取、Mahalanobis距离门限防止错误关联。

**📊 数据集**

使用的数据集：Gazebo仿真环境（6 m × 7 m、100 s、0.1 s步长）以及真实地下实验室（约30 m²、10 次实验、每次21.6 s）。真实环境的地面真值来自 OptiTrack 运动捕捉系统，采样频率与 LiDAR 10 Hz 对齐。

**📈 对比分析**

比较方法：与单纯里程计（DR）、单机器人 SLAM、集中式协作定位（CCL）、集中式协作+静态里程碑（CCL‑LM）以及分布式协作+双里程碑（DCL‑LM）进行对比。实验结果显示，DCL 相比 CCL 在位置 RMSE 上降低约 34 %，而 DCL‑LM 更进一步，RMSE 降低约 56 %。在真实实验中，DCL‑LM 的 X、Y 坐标均保持在 0.025–0.031 m 的水平，显著优于 CCL 的 0.056–0.070 m。

**⚠️ 局限性**

限制：目前仅验证两机器人配置，扩展到更大团队需设计更高效的交叉协方差管理；依赖 LiDAR 进行动态标记检测，受传感器遮挡或低光照影响；双里程碑方法要求存在可观测的静态特征，对完全无标记环境适用性有限；通信模型假设偶尔可达，极端延迟或断连场景尚未充分验证。

---

## 59. Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining

**arXiv ID:** 2603.11103 | [PDF](https://arxiv.org/pdf/2603.11103v1)

**作者:** Zhiyuan Zeng `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练阶段通过将静态软件仓库重构为多代理模拟生成的动态思维与行动轨迹，增强 LLM 对软件开发过程的理解

**💡 创新点**

创新点在于将仓库重构为“过程轨迹”而非仅代码；使用多代理模拟与结构化真实仓库信息来生成轨迹，并通过搜索式 CoT 优化提升推理质量

**🔧 技术方法**

技术主要包括多代理（主代理与子代理）仿真、结构化信息注入、链式思考搜索优化、对生成轨迹的预训练与损失遮蔽

**📊 数据集**

使用约 300k 个 GitHub 开源仓库作为基础，利用 Qwen3 生成 4B 词长的合成轨迹，并通过搜索迭代获得 12% 的预训练数据

**📈 对比分析**

与 Raw‑Repos、Repo2Agent、Repo2Agent‑Search 以及官方 Prolong 基线对比，结果显示在长上下文、编码、推理与软件工程能力上，搜索优化的轨迹训练实现了显著提升（如 HumanEval 37.20% vs 34.76%）

**⚠️ 局限性**

局限性包括对 LLM 生成轨迹的误差/幻觉依赖、数据噪声、仅在 Llama‑3‑8B 上验证、未考察后期微调与更大模型的可迁移性

---

## 60. One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries

**arXiv ID:** 2603.11545 | [PDF](https://arxiv.org/pdf/2603.11545v1)

**作者:** Mayank Saini Arit Kumar Bishwas `[一作]` `[通讯]`, Mayank Saini Arit Kumar Bishwas

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个集中式多模态AI调度框架，利用Supervisor动态拆解任务并调用专用工具，实现高效、可扩展的查询处理。

**💡 创新点**

核心创新在于中央Supervisor与Typed Tool接口、动态任务分解、并行执行、局部修复和自适应路由，取代传统预设决策树，显著提升性能与成本。

**🔧 技术方法**

使用了LLM/SLM组合、RouteLLM、Couplet Framework（YOLO、CLIP、Tesseract、Whisper等传统模型）、LangGraph StateGraph、Moe、分层记忆、多路复用URL验证等技术。

**📊 数据集**

评估基于2,847个跨15类任务的查询，涵盖MMLU、VQA-v2、PDF QA、OCR、音视频、混合检索等标准与自定义基准。

**📈 对比分析**

与匹配层级基线、单体LLM和AutoGen/LangGraph等对照实验，TTA降低72%，重工率降低85%，成本降低67%，吞吐量提升20%，准确率保持95%±1%一致。

**⚠️ 局限性**

局限包括LLM调度在极高吞吐时引发延迟、Couplet模型集成受限、长会话记忆压缩成本高、偶尔过度升级到昂贵模型，以及未实现全自动模型集成。

---

## 61. Single molecule localization microscopy challenge: a biologically inspired benchmark for long-sequence modeling

**arXiv ID:** 2603.11296 | [PDF](https://arxiv.org/pdf/2603.11296v1)

**作者:** Fatemeh Valeh `[一作]` (Technische Universität Wien), Gerhard Schütz `[通讯]` (Technische Universität Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Single Molecule Localization Microscopy Challenge（SMLM‑C）基准，用来评估长序列状态空间模型在稀疏时空点过程上的表现，并在两种dSTORM模拟条件下对S5和Mamba‑2进行实验评估。

**💡 创新点**

创新点在于：①设计了专门针对生物稀疏时空点过程的模拟基准，捕捉真实的闪烁动力学和重叠噪声；②通过对不同平均暗期的对照，揭示了长时间间隙对状态空间模型性能的根本影响。

**🔧 技术方法**

技术方面采用了结构化与选择性状态空间模型S5和Mamba‑2，配备轻量MLP解码器，训练时使用Chamfer距离，模型选择用Hungarian匹配误差；评估则采用检测精度、TP/FP/FN计数以及RMSE_TP。

**📊 数据集**

使用的数据集为SMLM‑C的十个模拟场景，重点实验条件为两种dSTORM（D2、D4），每个条件包含10,000帧、7或9个保留发射体，提供完整的真值位置信息。

**📈 对比分析**

通过比较S5与Mamba‑2的小/大版本，基于验证Hungarian误差和测试集的检测准确率与定位误差进行评估。最佳检测准确率约为73%，RMSE_TP约5–6 nm；在短暗期条件下两模型表现相近，长暗期条件下Mamba‑2显著优于S5，且更大模型整体性能更好。

**⚠️ 局限性**

局限性包括：假设已知发射体数量，未处理计数估计；仅在小尺寸ROI且每帧最多一次定位；仅评估两种条件，未覆盖全部10个场景；模型参数与实现不完全对齐导致比较偏差；整体准确率仍低于实际SMLM重建需求，且Mamba‑2训练成本较高。

---

## 62. Derain-Agent: A Plug-and-Play Agent Framework for Rainy Image Restoration

**arXiv ID:** 2603.11866 | [PDF](https://arxiv.org/pdf/2603.11866v1)

**作者:** Zhaocheng Yu `[一作]`, Kui Jiang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可插拔的Derain-Agent框架，对单张雨天图像进行动态的后处理优化。

**💡 创新点**

将雨水去除转为基于agent的规划与空间自适应强度调制，取代静态推理。

**🔧 技术方法**

使用轻量级规划网络（ResNet34）进行工具序列预测和强度映射，并与冻结的去噪、去模糊、色彩校正工具协同执行。

**📊 数据集**

在LHP-Rain、RE-Rain、Rain13K、Rain200H等合成与真实雨天数据上进行训练和评估。

**📈 对比分析**

与多种基准去雨网络（Restormer、Sfnet、NeRD等）在PSNR/SSIM上做对比，平均提升约0.8–1.2 dB，且在无参考指标上显著下降。

**⚠️ 局限性**

受限于固定工具库大小、需要离线搜索生成ground-truth路径，以及对极端多模态降解场景的适应性仍有限。

---

## 63. Learn Structure, Adapt on the Fly: Multi-Scale Residual Learning and Online Adaptation for Aerial Manipulators

**arXiv ID:** 2603.11638 | [PDF](https://arxiv.org/pdf/2603.11638v1)

**作者:** Samaksh Ujjawal `[一作]` (Robotics Research Center International Institute of Information Technology Hyderabad), Spandan Roy `[通讯]` (Robotics Research Center International Institute of Information Technology Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种实时预测自适应框架，用Factorized Dynamics Transformer预测残差并通过Latent Residual Adapter在线调整，以实现无人机机械臂的残差补偿控制。

**💡 创新点**

创新点包括将物理变量拆分为独立token实现跨变量注意力；双流架构分离短期惯性耦合与长期气动记忆；在潜在空间做线性RLS自适应而非全网络微调。

**🔧 技术方法**

采用Transformer+变量token化+全局token+交叉注意力+MLP解码+Recursive Least Squares+适应增益控制等技术。

**📊 数据集**

使用在Tarot‑650+2-DOF机械臂平台上采集的实验数据，涵盖0g/200g/300g/400g/500g负载下的5分钟/100Hz轨迹。

**📈 对比分析**

与SysID、GP、LSTM、Transformer、Neural‑fly、MLP-last-layer等基线对比；残差预测RMSE降至0.21，R²升至0.972；闭环跟踪RMSE比对手低约30%–40%，且保持毫秒级推理时间。

**⚠️ 局限性**

假设局部线性适配在极端剧烈动态跳变下可能失效；未显式建模不确定性和瞬时转移检测；依赖大量多负载数据，迁移到全新任务仍需重训。

---

## 64. Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection

**arXiv ID:** 2603.11441 | [PDF](https://arxiv.org/pdf/2603.11441v1)

**作者:** Mehmet Kerem Turkcan `[一作]` (Columbia University), Mehmet Kerem Turkcan `[通讯]` (Columbia University)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5022905172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将SAM3模型通过训练‑free的结构优化转化为实时的多类别开源词检测器。

**💡 创新点**

创新点在于利用ViT骨干的类别无关性实现骨干共享、批量解码、仅检测推理、TensorRT FP16部署及子块剪枝等多级优化，极大降低推理成本；另外提出仅训练轻量FPN适配器的adapter distillation方法，保持编码解码器不变，显著提升小模型的检测质量。

**🔧 技术方法**

技术包括骨干共享、批量多类别解码、检测‑only推理、图结构重排实现TensorRT FP16、分离式引擎设计、跨帧流水线、子块级别BlockPruner剪枝以及适配器蒸馏。

**📊 数据集**

主要使用COCO val2017（5000张，80类）评估检测性能；adapter蒸馏训练使用COCO train2017（118k张无标注图像）。

**📈 对比分析**

与原SAM3以及其他开源词检测器（Grounding DINO‑L、GLIP‑L、YOLO‑World‑L）对比，DART在不使用检测训练数据的前提下实现55.8 AP（15.8 FPS，4类）并在低延迟目标下达到38.7 AP（45 FPS）与RepViT‑M2.3后者。相比同类方法，速度提升约5.6×，AP接近甚至超过有训练的专用检测器。

**⚠️ 局限性**

局限在于单尺度FPN导致低分辨率下小目标召回率低（AP_S从40.3降至12.4），以及在极大类别数时解码器成为瓶颈；同时实验仅在单块RTX 4080上进行，性能对不同硬件的泛化未知。

---

## 65. Robust Co-design Optimisation for Agile Fixed-Wing UAVs

**arXiv ID:** 2603.11130 | [PDF](https://arxiv.org/pdf/2603.11130v1)

**作者:** Adrian Andrei Buda `[一作]` (Imperial College London), Urban Fasel `[通讯]` (Imperial College London)

**通讯引用:** 1065 | [OpenAlex ID](https://openalex.org/A5040035698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种双层鲁棒协同设计框架，联合优化固定翼无人机的结构参数与控制策略，并在设计与轨迹规划过程中直接考虑参数不确定性和风扰动。

**💡 创新点**

创新点在于：① 将噪声模型（参数误差+Von Kármán风）嵌入到低层评估循环，实现在设计搜索阶段即优化鲁棒性能；② 采用高层CMA-ES与低层直接协同优化，保持非线性气动和四元数动力学的可微分性；③ 通过Monte‑Carlo集成评估实现对鲁棒性的期望性能度量。

**🔧 技术方法**

技术手段包括：非线性条带理论气动模型、四元数动力学、直接共轭法轨迹优化、时间变LQR反馈、Monte‑Carlo鲁棒评估、Von Kármán风模型、CMA‑ES演化搜索、自动微分求解器与线性求解器。

**📊 数据集**

数据集：无公开数据集，全部采用仿真生成的三种任务（障碍规避、垂直翻转、180°弯道）及其对应的环境参数与扰动场。

**📈 对比分析**

与确定性协同设计基线比较，鲁棒方案在三种任务下均显著降低轨迹RMSE、提升成功率（多为100%），尤其在10%参数误差+风扰动情形下表现尤为突出；性能提升主要体现在鲁棒设计能在高扰动下保持稳定飞行与任务完成。

**⚠️ 局限性**

局限性：① 仍基于仿真，缺乏真实飞行验证；② 仅考虑六个几何参数，未涵盖结构或推进器细节；③ 受限于现有气动模型与LQR控制，复杂动力学或非线性控制方法尚未探索；④ Monte‑Carlo评估计算量大，优化效率仍可提升。

---

## 66. Design Exploration of Lightweight Interactions for Awareness-Supporting Technologies in Hybrid Work

**arXiv ID:** 2603.11977 | [PDF](https://arxiv.org/pdf/2603.11977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 67. Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild

**arXiv ID:** 2603.11618 | [PDF](https://arxiv.org/pdf/2603.11618v1)

**作者:** Jiin Im `[一作]` (Hanyang University), Je Hyeong Hong `[通讯]` (Hanyang University)

**通讯引用:** 3481 | [OpenAlex ID](https://openalex.org/A5010730040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Shape-of-You (SoY)，通过联合 2D 语义特征与 3D 几何信息来生成无监督语义对应的伪标签，并使用软目标损失训练轻量网络。

**💡 创新点**

创新点包括：① 将语义对应转化为融合的 Gromov-Wasserstein (FGW) 最优传输问题；② 用 3D 基础模型提取几何结构，并通过锚点线性化近似 FGW；③ 引入软目标损失对概率伪标签进行动态标签平滑。

**🔧 技术方法**

技术手段包括：Fused Gromov-Wasserstein（FGW）最优传输、未平衡 OT、锚点线性化、基于 3D 底层模型（VGGT）提取几何坐标、DINOv2+Stable Diffusion 预训练特征、软目标对比损失和密集对应损失。

**📊 数据集**

使用 SPair-71k 与 AP-10k 两大无监督对应基准数据集进行评估。

**📈 对比分析**

与现有基线（如 ASIC、DINOv2、DistillDIFT、DINOv2+SD）对比，SoY 在 SPair-71k 上 PCK@0.1 取得 67.9%（相较 63.5% 的基线提升约 4.4%），在 AP-10k 上同样表现领先，且在几何模糊子集上收益更明显。

**⚠️ 局限性**

局限性：依赖 3D 基础模型，模型在平面或透明表面重建不佳；对高度对称物体或中等视角下的几何歧义可能产生错误锚点，导致软目标可能过度平滑。

---

## 68. Enhancing Requirements Traceability Link Recovery: A Novel Approach with T-SimCSE

**arXiv ID:** 2603.11800 | [PDF](https://arxiv.org/pdf/2603.11800v1)

**作者:** Ye Wang `[一作]` (Zhejiang Gongshang University), Liping Zhao `[通讯]` (University of Manchester)

**通讯引用:** 1886 | [OpenAlex ID](https://openalex.org/A5025287610)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于SimCSE的需求追踪链接恢复方法T‑SimCSE，并通过奖励机制提升检索质量。

**💡 创新点**

创新点在于引入“特异性”指标对潜在链接进行加权重重排序，从而解决传统相似度匹配忽视上下文差异的问题。

**🔧 技术方法**

主要技术包括预训练句子嵌入模型SimCSE、余弦相似度计算、特异性奖励策略以及阈值自动搜索。

**📊 数据集**

在十个公开的需求追踪数据集（EasyClinic、GANNT、CM1、CCHIT、MODIS、Dronology、WARC、InfusionPump、EBT、IceBreaker）上进行实验。

**📈 对比分析**

与基线（VSM、LSI、BERT、Word2Vec、SimCSE）以及先前方法（S2Trace、WQI、LiSSA‑CoT、GeT2Trace）对比，T‑SimCSE在 MAP、F1、F2 等指标上平均提升约 10‑20%，并在大多数数据集上实现显著统计优势。

**⚠️ 局限性**

局限包括对小规模数据集的依赖、未在真实工业项目中验证、以及在特定域（如工业规模文档）下的可扩展性和多语言支持仍待改进。

---

## 69. Beyond the Class Subspace: Teacher-Guided Training for Reliable Out-of-Distribution Detection in Single-Domain Models

**arXiv ID:** 2603.11269 | [PDF](https://arxiv.org/pdf/2603.11269v1)

**作者:** Hong Yang `[一作]` (Rochester Institute of Technology), Alex Ororbia `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1562 | [OpenAlex ID](https://openalex.org/A5084332360)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究单域场景下的 OOD 检测问题，提出 Teacher-Guided Training (TGT) 方法，在训练阶段通过冻结的多域教师（DINOv2）提供的类别抑制残差来引导学生网络恢复对域偏移的敏感性，从而提升 OOD 检测性能。

**💡 创新点**

创新点包括：① 发现并系统化解释了单域监督训练导致的 Domain‑Sensitivity Collapse (DSC) 现象；② 设计了利用类别抑制的教师残差作为辅助目标的 TGT 训练框架，实现了无推理开销的表示重塑；③ 通过理论与实验验证 DSC 与 OOD 分数失效之间的关系。

**🔧 技术方法**

使用的技术包括：冻结的 DINOv2 ViT-S/14 作为多域教师；对教师特征做类别子空间投影并取残差；在学生网络上加一个域头，使用余弦相似度损失；标准交叉熵与辅助域损失联合训练；有效秩（effective rank）与参与比（participation ratio）等几何诊断工具；多种后置 OOD 分数（MDS、ViM、kNN、MSP、Energy 等）。

**📊 数据集**

实验使用八个单域数据集：Colon、Tissue、EuroSAT、Fashion、Food、Rock、Yoga、Garbage，涵盖病理图像、卫星图像、产品图像、食品照片等多种视觉域。

**📈 对比分析**

与 CE 基线、SupCon 对比，TGT 在所有八个数据集上平均提升 FPR@95（尤其是距离型分数）数个百分点（MDS ↓ 8.3pp，ViM ↓ 5.8pp，kNN ↓ 9.2pp），AUROC 也有明显上升；在内域 OOD 上同样表现提升，日志型分数提升因数据集差异而有时为负面；相比 DINOv2 细调，TGT 的提升幅度较小，原因是教师与学生共享架构与初始化。

**⚠️ 局限性**

局限性：① 对极端内域 OOD 的绝对性能仍较高，表示难以完全区分相似域内的未知类别；② TGT 需要额外的教师模型与辅助头，增加训练时的算力与实现复杂度；③ 在共享架构（如 DINOv2）时，教师残差的多样性有限，导致提升不明显；④ 仍未解决在高度细粒度类别中域信息与类别信息混叠导致的 OOD 效果不佳。

---

## 70. Coalgebraic Path Constraints

**arXiv ID:** 2603.12204 | [PDF](https://arxiv.org/pdf/2603.12204v1)

**作者:** Todd Schmid `[一作]` (Bucknell University), Todd Schmid `[通讯]` (Bucknell University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5087089457)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并研究了等式路径约束（equational path constraints），用以在协变范畴中对 coalgebra 进行公理化，并构造相应的最终 coalgebra。

**💡 创新点**

提供了一种新的、类似代数的描述方法替代传统的 coequations，证明了等式路径约束能定义协变范畴，并给出了色彩数上界及终极网（terminal net）构造。

**🔧 技术方法**

运用了范畴论、可达性、Monad、终极序列、路径约束和等式的概念，结合子函子和等值器的构造。

**📊 数据集**

无实验数据集，纯理论研究。

**📈 对比分析**

无实验对比，未进行性能评估。

**⚠️ 局限性**

尚未给出对所有可达 functor 的协变范畴的结构性表征，对非等式路径约束的分析不足，且对更一般情形的推广仍有待研究。

---

## 71. AGMARL-DKS: An Adaptive Graph-Enhanced Multi-Agent Reinforcement Learning for Dynamic Kubernetes Scheduling

**arXiv ID:** 2603.12031 | [PDF](https://arxiv.org/pdf/2603.12031v1)

**作者:** Hamed Hamzeh `[一作]` (University of Westminster), Hamed Hamzeh `[通讯]` (University of Westminster)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5080249914)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于多智能体强化学习的动态Kubernetes调度器AGMARL-DKS，利用图神经网络实现节点间全局上下文感知，并采用压力感知的字典序优先策略实现多目标调度。

**💡 创新点**

创新点包括：①将调度任务建模为协作式多智能体MAMDP，实现CTDE（集中训练+分散执行）以克服大规模集群的可扩展性瓶颈；②使用GNN生成全局嵌入，让每个节点能在局部观测中获得全局信息；③引入压力感知的字典序多目标决策机制，动态调整目标优先级；④结合自适应学习率和自我约束策略，实现对动态工作负载的自适应响应。

**🔧 技术方法**

采用的技术主要包括：多智能体深度确定性策略梯度（MADDPG）、图神经网络（GNN）、集中训练-分散执行框架、字典序多目标排序、压力感知奖励函数、Flask实现的Kubernetes Scheduler Extender、GKE原生指标采集与模拟环境。

**📊 数据集**

实验使用的“数据集”为在Google Kubernetes Engine（GKE）上部署的模拟与真实工作负载：①两套自定义压力测试（资源压力级联与高频脏/故障注入），②各种规模节点池（固定与自适应）以及模拟的CPU/内存请求和重启行为；未使用公开的标准数据集，而是基于真实Kubernetes API构建的仿真环境。

**📈 对比分析**

通过在GKE上对比默认Kubernetes调度器，采用两种压力测试场景，评估指标包括资源利用率、故障率、成本、调度延迟和系统稳定性。AGMARL-DKS在所有指标上均显著优于默认调度器：实现更高的资源打包效率、更低的成本、更好的容错性，并在极端压力下保持系统可用。

**⚠️ 局限性**

限制与挑战：①评估仅在GKE单一环境完成，缺乏跨云/多集群的泛化验证；②需要训练时间和算力，部署前需预训练模型；③对极大规模集群的GNN规模与通信开销尚未彻底验证；④字典序优先策略虽可解释但在极端多目标冲突场景下可能产生局部最优。

---

## 72. AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies

**arXiv ID:** 2603.11969 | [PDF](https://arxiv.org/pdf/2603.11969v1)

**作者:** Jennifer Nolan `[一作]` (Georgia Institute of Technology), John Christian `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1710 | [OpenAlex ID](https://openalex.org/A5060159588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AstroSplat，一种基于物理学的 2D Gaussian splatting 框架，用来从小行星现场图像进行表面重建与光照特征估计。

**💡 创新点**

创新点在于将行星散射模型（Lambert、Lommel‑Seeliger、Lunar‑Lambert）直接嵌入 Gaussian splatting 的强度计算中，替代传统的球谐系数（SH），实现光照方向、法线和材料属性的显式建模。

**🔧 技术方法**

使用技术包括 2D Gaussian splatting、球谐系数（对照）、物理光学反射模型、视角/光照变换、基于损失的训练（L1+SSIM+法线损失）以及三维网格重建。

**📊 数据集**

数据集为 NASA Dawn 任务拍摄的 Vesta Cornelia 卧坑、Ceres Ikapati 与 Ahuna Mons 的多视角图像。

**📈 对比分析**

通过 PSNR、SSIM、LPIPS 等渲染指标以及法线误差、相对 albedo 误差和对称 Hausdorff 距离评估重建质量，实验结果显示物理模型在渲染质量、法线精度与 albedo 估计方面均优于 SH。

**⚠️ 局限性**

局限性在于仅在局部区域验证；仍需更大规模数据和更多物理模型来实现全表面重建，且在计算上与 SH 对比的速度与内存开销未作详细讨论。

---

## 73. Approximate Dynamic Nearest Neighbor Searching in a Polygonal Domain

**arXiv ID:** 2603.11775 | [PDF](https://arxiv.org/pdf/2603.11775v1)

**作者:** Joost van der Laan `[一作]` (Utrecht University), Lorenzo Theunissen `[通讯]` (Delft University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了一种在二维多边形域中，支持点集动态插入删除的近似最近邻搜索数据结构，并给出了相应的近似两点最短路径查询方法。

**💡 创新点**

主要创新点在于：①构造了一个新的连续锥图（包含Steiner点）以实现对任意两点（包含查询点）之间的(1+ε)近似距离求解；②利用层次化分隔器和锚点集合，设计了动态加权Voronoi图，实现高效的近似最近邻查询；③改进了Thorup的分隔器实现，使空间从O(n^9)下降到O(n/ε log n)，查询时间从O(1/ε^3)下降到O(1/ε^2 log n)。

**🔧 技术方法**

核心技术包括：锥图（Cone graph）与连续锥图、层次化分隔器（Balanced hierarchical separator）、锚点（Anchor points）构造、动态加权Voronoi图（additively weighted Voronoi diagram）、多点的低包络（lower envelope）维护、垂直射线查询结构等。所有这些技术结合实现了动态近似最近邻和两点最短路径的高效查询。

**📊 数据集**

本文为理论研究，未使用具体的实验数据集；所有结果均为算法分析与复杂度证明。

**📈 对比分析**

与之前的工作相比：对动态近似最近邻，空间从O(n+m)（静态）提升到O((n+m)/ε log n + m/ε log m)，查询时间从O(√m log n)降低到O((1/ε^2) log n + log^2 m)，更新时间从O(√m log m log^2 n)降低到O((1/ε^2) log n + log^2 m)。对两点最短路径，空间从O(n^9)降至O(n/ε log n)，查询时间从O(1/ε^3)降至O(1/ε^2 log n)。

**⚠️ 局限性**

局限性包括：①仅适用于二维多边形域；②结果依赖于近似参数ε，较小ε时空间与时间仍然较大；③更新时间为摊销形式，实际实现可能有较大常数；④对更高维或更复杂障碍域的推广尚未给出。

---

## 74. Duality and decoding of linearized Algebraic Geometry codes

**arXiv ID:** 2603.11826 | [PDF](https://arxiv.org/pdf/2603.11826v1)

**作者:** Elena Berardini `[一作]` (University of Bordeaux), Fabrice Drain `[通讯]` (University of Bordeaux)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文设计了一种多项式时间的解码算法，能在求和-秩度量下对采用无分支评估点的线性化代数几何（LAG）码进行错误纠正，并证明该算法的正确性。

**💡 创新点**

创新点在于首次为分裂代数域上的除子引入Riemann–Roch定理与Serre对偶性，利用其推导出LAG码与其线性化微分码的等价双对偶关系，并基于此构造了可在多点求和-秩度量下纠错的完整算法。

**🔧 技术方法**

技术手段包括：在除子上的Riemann–Roch定理与对偶性证明、Ore多项式环与其伴随代数、代数几何中的微分形式与残差、 Adele空间与残差定理、以及传统的syndrome解码与线性系统求解。

**📊 数据集**

论文未使用具体实验数据集，主要聚焦理论证明与算法设计；但在附录中提供了SageMath实现代码用于演示算法的可行性。

**📈 对比分析**

与传统的Hamming度量下AG码解码算法相比，本文的算法能够在求和-秩度量下实现与Singleton界限相近的纠错能力，误差上限为⌊(A)-g-1/2⌋，并在复杂度上保持多项式级别，主要依赖于评估点数s和分裂指数r。

**⚠️ 局限性**

局限性包括：需评估点为无分支且满足特定假设；对除子A的次数范围限定在2g-2<degA<sr；需要存在满足特定条件的Galois除子A₁；在实际大规模实现时，常数因子与具体参数（如r、s）可能导致计算量显著。

---

## 75. Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning

**arXiv ID:** 2603.11219 | [PDF](https://arxiv.org/pdf/2603.11219v1)

**作者:** Yuehao Song `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 32800 | [OpenAlex ID](https://openalex.org/A5037191476)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种双系统驾驶策略，将视觉语言模型（VLM）与端到端（E2E）驾驶策略对齐，实现决策与规划的一致性；

**💡 创新点**

创新点在于三阶段一致性导向训练框架：驱动预训练 → 开环对齐 → 通过层次强化学习闭环对齐，显著提升决策-规划一致性与安全性；

**🔧 技术方法**

技术包括Qwen2.5-VL-3B视觉语言模型、决策适配器、Diffusion-based Planner、AdaLN条件注入、层次强化学习（HRL）与安全/效率奖励；

**📊 数据集**

使用约10,000小时（360M帧）真实驾驶数据，含多视角视频、里程计、地图与交通标注，并在1,300条高危驾驶片段构建3D Gaussian Splatting（3DGS）闭环环境；

**📈 对比分析**

与Senna等SOTA方法对比，决策-规划F1提升19.3%，开环FDE下降5.7%，闭环事故率下降30.6%，在NAVSIM v2上EPDMS提升1.1点，整体性能显著优于基线；

**⚠️ 局限性**

局限在于VLM推理速度低（10Hz），无法实现车载实时同步，需进一步硬件优化以实现完整同步协同。

---

## 76. ChemSICal-Net: Timing-Controlled Chemical Reaction Network for Successive Interference Cancellation in Molecular Multiple Access

**arXiv ID:** 2603.12141 | [PDF](https://arxiv.org/pdf/2603.12141v1)

**作者:** Alexander Wietfeld `[一作]` (Technical University of Munich), Wolfgang Kellerer `[通讯]` (Technical University of Munich)

**通讯引用:** 9761 | [OpenAlex ID](https://openalex.org/A5021781616)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发了ChemSICal‑Net，一个利用化学反应网络实现多用户化学多址中的SIC接收机，并通过化学时钟和贝叶斯优化提升检测精度。

**💡 创新点**

创新点在于将SIC算法映射为可时序控制的化学计算模块，结合化学振荡器实现显式时序控制，并提出自适应贝叶斯优化用于高维CRN参数调优。

**🔧 技术方法**

使用化学反应网络、化学振荡器（双位磷酸化振荡器）、蒙特卡洛模拟、Gillespie SSA、ODE求解、贝叶斯优化、模拟退火、Metropolis‑Hastings等技术。

**📊 数据集**

使用的“数据集”是基于DBMC通道模型的符号组合产生的接收分子数分布（Poisson混合分布），以及参数化的传输距离、分子数量等实验参数。

**📈 对比分析**

与基线未时钟的Always‑On方案以及模拟退火、MCMC对比，使用输入加权错误概率和决策时间为指标，结果显示自适应BO在相同仿真预算下可将错误率从≈0.25降低到≈0.02，时钟控制在短决策时间内提升≈2倍准确率。

**⚠️ 局限性**

局限性包括化学反应网络的规模与复杂性、参数可迁移性、对化学时钟实现的实际时间尺度以及对采样和重置机制的抽象处理。

---

## 77. Causal Matrix Completion under Multiple Treatments via Mixed Synthetic Nearest Neighbors

**arXiv ID:** 2603.11942 | [PDF](https://arxiv.org/pdf/2603.11942v1)

**作者:** Minrui Luo `[一作]` (Tsinghua University), Zhiheng Zhang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究多重处理（多水平）情况下的因果矩阵补全问题，提出混合合成最近邻（MSNN）方法以实现条目级别的因果估计。

**💡 创新点**

创新点：① 通过共享行因子假设实现跨处理级别的数据整合；② 引入混合锚点（MAR/MAC）并使用权重归一化，显著提升稀疏处理级别的样本效率，理论上实现指数级别的样本量改进；③ 在保持SNN的有限样本误差上界和渐近正态性不变的前提下，扩展到多处理场景。

**🔧 技术方法**

技术手段：低秩矩阵分解（SVD）、权重归一化、混合锚点选取（基于二分图最大团算法）、理论分析（有限样本误差界、渐近正态性、样本效率证明）以及实验评估。

**📊 数据集**

数据集：合成数据（MCAR 与 MNAR 生成），真实案例数据——加州烟草控制政策（Proposition 99）等。

**📈 对比分析**

方法比较：与原始SNN进行对比；实验结果显示 MSNN 在稀疏处理级别可行率提升约 2–3 倍，平均相对误差降低 2–3 倍，尤其在处理级别观测比例低于 5% 时优势更为显著。

**⚠️ 局限性**

局限性：① 对行因子共享的假设敏感，若不满足可能导致估计失效；② 极度稀疏的数据仍可能无法构造足够的混合锚点；③ 算法在大规模矩阵上求解最大全1子矩阵的计算复杂度较高，需要进一步优化。

---

## 78. Semi-Synthetic Parallel Data for Translation Quality Estimation: A Case Study of Dataset Building for an Under-Resourced Language Pair

**arXiv ID:** 2603.11743 | [PDF](https://arxiv.org/pdf/2603.11743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 79. OmniForcing: Unleashing Real-time Joint Audio-Visual Generation

**arXiv ID:** 2603.11647 | [PDF](https://arxiv.org/pdf/2603.11647v1)

**作者:** Yaofeng Su `[一作]` (JD Explore Academy), Nan Duan `[通讯]` (JD Explore Academy)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过把离线的双向音视频扩散模型（LTX‑2）蒸馏成实时自回归流式生成器，实现在单张 GPU 上实现约 25 FPS 的同步音视频流式生成。

**💡 创新点**

创新点包括：
- Asymmetric Block‑Causal Alignment 与 Global Prefix 形成 1 秒宏块对齐，解决音视频频率不匹配；
- Audio Sink Token 与 Identity RoPE 作为全局稀疏注意力的缓冲，缓解音频稀疏导致的 Softmax 崩溃和梯度爆炸；
- Joint Self‑Forcing Distillation 把双模态的曝光偏差最小化；
- Modality‑Independent Rolling KV‑Cache 通过跨模态的并行计算和 KV‑缓存降低每步上下文复杂度至 O(L)。

**🔧 技术方法**

技术手段包括：
- Diffusion Transformer (DiT) 与 LTX‑2 双流架构；
- Distribution Matching Distillation (DMD) 进行少步去噪；
- Causal ODE 回归对齐 causal mask；
- Audio Sink Tokens + Identity RoPE 解决注意力稀疏；
- Self‑Forcing 训练策略；
- KV‑缓存与模态独立并行推理。

**📊 数据集**

训练使用 Mixkit 视频+文本、AudioCaps 音频文本以及 Open‑Sora‑Plan 文本重写后的数据；评估在 JavisBench 与 VBench 上进行。

**📈 对比分析**

与双向教师 LTX‑2、Cascade T2A+A2V、T2V+V2A 以及其它 joint 生成模型比较：
- 运行时间从 197 s 缩短至 5.7 s（TTFC 0.7 s）；
- 生成速度达 25 FPS；
- 视觉质量 FVD 137.2（仅比教师略高）、FAD 5.7；
- 文本一致性 CLIP 0.322、CLAP 0.401；
- 同步度 DeSync 0.392 接近教师 0.384；
- 在大多数指标上排名教师第二，优于所有基线。

**⚠️ 局限性**

局限性：
- 由于使用单向注意力，某些跨模态一致性和同步度仍略低于双向教师；
- 训练与推理仍需要较高计算资源（14 B 视频 + 5 B 音频模型）；
- 目前仅在单 GPU 上实现，扩展到更大分辨率/更长时序需要进一步的多 GPU 并行策略；
- 复杂的三阶段蒸馏 pipeline 使得复现和调优成本较高。

---

## 80. Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training

**arXiv ID:** 2603.12246 | [PDF](https://arxiv.org/pdf/2603.12246v1)

**作者:** Yixin Liu `[一作]` (Meta Superintelligence Labs), Zhengxing Chen `[通讯]` (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在非可验证任务下，使用推理型LLM评判器对LLM后期训练（RL）的效果进行系统评估，并与传统非推理评判器进行对比。

**💡 创新点**

提出并验证推理LLM评判器在实际策略训练中的优势，揭示其能够抑制奖励劫持并引导模型生成对金标准评判器有效的对抗输出；同时证明推理过程的细粒度监督对提升评判器有效性至关重要。

**🔧 技术方法**

采用gpt-oss-120b作为金标准评判器；对Qwen3系列模型进行SFT+GRPO微调得到推理和非推理评判器；利用GRPO对政策进行微调；使用Krippendorff's Alpha衡量评判器与金标准的一致性；在pointwise和pairwise评判模式下进行实验。

**📊 数据集**

Tulu3偏好数据混合（指令+候选输出）用于训练评判器和策略；Arena-Hard-V2 benchmark用于评估最终策略性能。

**📈 对比分析**

通过比较不同规模评判器（非推理 vs 推理）在金标准评判器下的奖励变化曲线；对比训练评判器与金标准评判器评估的奖励差异；在Arena-Hard-V2上，使用推理评判器训练的Llama-3.1策略在创作写作子集取得约90%分数，超过Gemini-2.0-Flash、Gemini-2.5等前沿模型；非推理评判器训练的策略则出现严重奖励劫持。

**⚠️ 局限性**

实验受限于对推理努力与金标准访问的依赖，结果展示了LLM评判器的脆弱性；未在更大模型或多评判器/多prompt设定下验证；缺乏对抗性训练或动态评判器机制的探索，需要进一步提升评判器鲁棒性。

---

## 81. PolyCrysDiff: Controllable Generation of Three-Dimensional Computable Polycrystalline Material Structures

**arXiv ID:** 2603.11695 | [PDF](https://arxiv.org/pdf/2603.11695v1)

**作者:** Chi Chen `[一作]` (Shanghai Jiao Tong University), Yanming Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 27975 | [OpenAlex ID](https://openalex.org/A5100344066)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了PolyCrysDiff框架，利用条件潜在扩散模型生成可计算、可控的三维多晶材料微结构。

**💡 创新点**

创新点在于将3D VAE与潜在扩散相结合，并通过交叉注意力实现对平均晶粒大小和球形度的精准控制；生成质量和可计算性均优于传统MRF和CNN方法。

**🔧 技术方法**

使用技术包括3D变分自编码器（VAE）、潜在扩散模型、U‑Net去噪网络、交叉注意力机制、VGG感知损失以及基于MOOSE框架的晶体塑性有限元（CPFEM）模拟。

**📊 数据集**

数据集为2000个基于Voronoi三维镶嵌生成的微结构（64³体素，RGB编码晶体取向），每个样本晶粒数随机在50~300之间。

**📈 对比分析**

通过与MRF、SolidTexture等传统方法对比，采用KS、EMD、VGG感知分数、两点相关函数等指标评估；在晶粒属性控制上R²>0.972，生成结构可直接用于CPFEM，且在力学性能上的表现与晶粒大小相关性与实验一致。

**⚠️ 局限性**

局限性包括训练数据仅为合成Voronoi结构，缺乏真实实验微结构；当前只支持平均晶粒大小和球形度的条件控制；对更大尺寸或更高分辨率的结构扩展性以及对更复杂属性的可控性仍需进一步研究。

---

## 82. Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets

**arXiv ID:** 2603.11634 | [PDF](https://arxiv.org/pdf/2603.11634v1)

**作者:** Sreevardhan Sirigiri `[一作]` (University of Sydney), Fabio Ramos `[通讯]` (NVIDIA)

**通讯引用:** 9065 | [OpenAlex ID](https://openalex.org/A5062619542)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了基于路径签名的熵度量，用于评估机器人示范数据集多样性，并基于此提出FAKTUAL算法进行无模型数据集挑选。

**💡 创新点**

首次将签名核与Shannon/Von Neumann熵结合用于机器人轨迹多样性评估，提出全模型无关、低成本的FAKTUAL数据挑选策略。

**🔧 技术方法**

使用路径签名变换、签名核、Shannon/Von Neumann熵、近似最大覆盖贪心算法、ViT特征嵌入、RNN/SMVLA行为克隆等技术。

**📊 数据集**

在RoboMimic（Can、Square、Transport）与MetaWorld（Stick Push、Door Open、Shelf Place）以及四个SO-ARM101真实操作任务上进行实验。

**📈 对比分析**

与DemInf、Cupid、Demo-SCORE、Oracle、Success Similarity及随机选择进行比较；FAKTUAL在多数任务中接近或超过全量数据，优于随机挑选，且比大多数质量基准略逊一筹。

**⚠️ 局限性**

仅关注多样性不保证高质量；对存在劣质或对抗性示范时可能适得其反；缺乏因果关系证明；对模型容量敏感，若模型不足则多样性可能适得其反。

---

## 83. HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

**arXiv ID:** 2603.12222 | [PDF](https://arxiv.org/pdf/2603.12222v1)

**作者:** Andy Li `[一作]` (University of Aberdeen), Georgios Leontidis `[通讯]` (UiT Arctic University of Norway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端的层级自动剪枝框架HiAP，能在单阶段训练中自动发现Vision Transformer的最优子网络；

**💡 创新点**

创新点在于同时使用宏观门（裁剪注意力头与FFN块）与微观门（裁剪头内部维度与FFN神经元），并通过可微的Gumbel‑Sigmoid门和预算感知损失实现无手工阈值、无多阶段流程的自动剪枝；

**🔧 技术方法**

采用Gumbel‑Sigmoid连续松弛、梯度直通估计、可微MACs预算正则化、结构可行性约束、知识蒸馏等技术；

**📊 数据集**

在CIFAR‑10（ViT‑Tiny）和ImageNet‑1K（DeiT‑Small）上进行实验；

**📈 对比分析**

与现有结构化剪枝方法（WDPruning、S2ViT、ViT‑Slim、GOHSP等）对比，HiAP在保持或略降Top‑1精度的同时实现约33% MACs减少，单阶段训练时间仅为200轮；

**⚠️ 局限性**

局限在于优化目标仅为预期MACs，未直接校准实际延迟/能耗；实际加速受硬件/内核实现影响，未来可结合平台校准的延迟/能耗信号、令牌剪枝、量化及编译优化进一步提升。

---

## 84. WORKSWORLD: A Domain for Integrated Numeric Planning and Scheduling of Distributed Pipelined Workflows

**arXiv ID:** 2603.12214 | [PDF](https://arxiv.org/pdf/2603.12214v1)

**作者:** Taylor Paul `[一作]` (University of Maryland), William Regli `[通讯]` (University of Maryland)

**通讯引用:** 4688 | [OpenAlex ID](https://openalex.org/A5012260180)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于数值规划的框架，用YAML描述数据管道与资源图，自动生成PDDL并利用增强版ENHSP求解器实现工作流的联合规划与调度，最终生成完整的执行计划并进行验证与可视化。

**💡 创新点**

1）引入了全新的数值规划领域<WORKS‑WORLD>，能够在不预先给出完整 DAG 的情况下，通过目标约束自动构造并部署分布式数据管道；2）将工作流、计算、存储与网络资源统一建模为“接口”概念，实现跨站点的数据共享与处理决策；3）通过Jinja模板与YAML配置实现人性化的输入方式，降低规划问题的建模门槛。

**🔧 技术方法**

主要技术包括：YAML配置文件→Jinja+Python生成PDDL；PDDL 2.1 level 2数值规划；ENHSP数值规划器（h_max^MRP+H启发式）；计划验证（VAL）和可视化工具；Kubernetes+CloudLab环境用于基准测试。

**📊 数据集**

使用公开的合成基准集（https://gitlab.com/thpaul/worksworld-benchmarks/）以及在 CloudLab 部署的 Kubernetes 集群；未使用真实工业数据，而是通过可配置的 YAML 生成不同规模的工作流与资源图。

**📈 对比分析**

对比方法：本文未与现有专有或领域特定调度器直接比较，而是通过实验验证 ENHSP 在不同规模（工作流组件、接口、链路、站点）下的可扩展性。性能结果表明，在 30 GB 内存与 1 小时 CPU 限制下，能够成功规划 14 个处理/数据组件、8 个站点的线性链工作流；对规模的指数增长表现为子指数级别，说明自适应剪枝有效。

**⚠️ 局限性**

主要局限：1）仅支持线性链式工作流，无法处理一般 DAG；2）缺乏时间/周期调度，适用于永久驻留的管道；3）未利用状态相关的动作成本，导致资源利用率模型过于粗糙；4）仅在合成基准上验证，缺少真实工业案例的实证。

---

## 85. HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment

**arXiv ID:** 2603.12044 | [PDF](https://arxiv.org/pdf/2603.12044v1)

**作者:** Krishna Kant Singh `[一作]` (Forschungszentrum Jülich), Lena Oden `[通讯]` (FernUniversität in Hagen)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5011121841)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文设计并实现了可在不同 HPC 站点零改动部署的 Apptainer 容器，利用 PMIx 完全自包含 OpenMPI 5 实现 MPI+CUDA 软件栈的可移植化，并在 Karolina 与 JURECA 集群上验证其性能与裸机相当。

**💡 创新点**

创新点包括：①基于 PMIx 的混合容器化策略，使容器内部完整包含 OpenMPI 5 运行时，消除宿主 MPI 依赖；②将容器化流程嵌入 EBRAINS esd 的 CI/CD 管道，实现自动化、可复现的性能验证；③通过微基准与实际神经网络模拟，证明跨站点迁移不损失性能。

**🔧 技术方法**

使用技术：Apptainer 容器引擎、PMIx 进程管理、OpenMPI 5、UCX、NCCL、Spack 包管理、CI/CD 流水线；基准工具包括 OSU MPI 微基准（init、latency）、NCCL AllReduce、Arbor 与 NEURON 神经网络模拟。

**📊 数据集**

数据集与测试：Arbor 环形网络（128 k 细胞）和 NEURON 256 环网络，CPU 与 GPU 两种配置；使用 OSU 微基准、NCCL 测试；未使用外部科学数据，仅使用模拟生成的数据。

**📈 对比分析**

比较方法：在 Karolina 与 JURECA 两台 HPC 集群上同时跑裸机与容器化版本，测量 MPI 初始化时间、点对点延迟、NCCL 带宽以及 Arbor/NEURON 的强/弱扩展；结果显示 CPU 工作几乎无性能损失，GPU 工作容器相对慢约 12‑19%，微基准几乎无差异。

**⚠️ 局限性**

局限性：GPU 性能存在 12‑19% 的相对延迟，原因尚不明确；评估仅覆盖 esd 生态中少量软件包，未测量跨工具交互性能；需人工检查日志以发现隐性问题，缺乏自动日志解析；CUDA 版本兼容性深度分析未完成。

---

## 86. Fast and exact visibility on digitized shapes and application to saliency-aware normal estimation

**arXiv ID:** 2603.11851 | [PDF](https://arxiv.org/pdf/2603.11851v1)

**作者:** Romain Negro `[一作]` (Universite Savoie Mont Blanc), Jacques-Olivier Lachaud `[通讯]` (Universite Savoie Mont Blanc)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于整数区间列表的数字形状可见性快速精确计算方法，并将该可见性用于构造对显著特征敏感的法向估计器。

**💡 创新点**

创新点在于：①将格点集合编码为投影坐标对应的整数区间集合；②将可见性问题化为区间平移与交集的组合运算，从而实现全局精确可见性；③将可见性信息融入加权协方差分析，得到既保留平滑区域法向又精准捕捉尖锐特征的法向估计。

**🔧 技术方法**

主要技术包括：整数区间列表表示、格点映射（lattice map）、原始向量枚举、区间交集算法、Gaussian 加权的可见点协方差矩阵、特征值分解得到法向；实验中使用 DGtal 库实现可见性与法向计算。

**📊 数据集**

实验数据集涵盖多种 2D/3D 形状：goursat、torus、rcube、sphere9、ellipsoid、leopold、Dice‑20（D20）等数字化模型，使用不同网格步长（0.125~1）进行评测。

**📈 对比分析**

与传统广度优先可见性算法比较：在可见性半径≤20 时速度更快；与积分不变体（Integral Invariant）法向估计比较：在 RMSE 与 Emax 上呈现 h^{2/3} 与 h^{1/2} 的收敛率，精度更高；在曲率估计上，提出的可见性法向在保留平滑部位精度的同时，能更准确地重现尖锐特征；计算时间相对较长（约 52 s 对比 2.3 s/0.1 s）。

**⚠️ 局限性**

局限性包括：①当可见性半径增大时，计算时间显著增加；②多网格收敛性尚未严格证明；③算法目前聚焦于全局可见性匹配，尚未完全优化对不同尺度的粗细层次处理；④对参数（σ、可见性半径）的选择需要经验调优。

---

## 87. Edge-Cloud Collaborative Speech Emotion Captioning via Token-Level Speculative Decoding in Audio-Language Models

**arXiv ID:** 2603.11397 | [PDF](https://arxiv.org/pdf/2603.11397v1)

**作者:** Xiangyuan Xue `[一作]` (University of Auckland), Hong Jia `[通讯]` (University of Auckland)

**通讯引用:** 27447 | [OpenAlex ID](https://openalex.org/A5102810576)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于不确定性引导的投机式解码（UGSD）的边缘-云协同语音情感字幕生成框架

**💡 创新点**

创新点在于使用token级熵度量不确定性，动态决定何时将高不确定性token块上报云端进行验证，从而实现精细化、按需的协同；同时采用自适应块长策略平衡验证频率与通信成本

**🔧 技术方法**

利用轻量化SALM（Qwen2.5-Omni-3B）做边缘端草稿生成，云端使用大规模LALM（Qwen3-Omni-30B-A3B-Instruct）做验证；结合投机式解码、熵度量、rank式接受阈值及动态块长

**📊 数据集**

在MER2024数据集上进行评测，涵盖英文和中文子集，共计每种语言332条录音

**📈 对比分析**

与仅边缘、仅云、固定块长投机式解码等方案对比，UGSD在英文BLEU-1提升约21–76%，中文METEOR提升111%；推理时间从40.21s降至28.67s，token输出速率提升8.5×；仅约18.2%token被送往云端，显著降低数据传输与隐私风险

**⚠️ 局限性**

局限包括对云端验证器模型规模敏感，较小云模型时收益下降；仍需传输压缩后的token与音频特征，可能泄露部分信息；以及对动态块长策略的调参复杂度和在极低网络延迟场景下的表现待进一步验证

---

## 88. Improving LLM Performance Through Black-Box Online Tuning: A Case for Adding System Specs to Factsheets for Trusted AI

**arXiv ID:** 2603.11340 | [PDF](https://arxiv.org/pdf/2603.11340v1)

**作者:** Yonas Atinafu `[一作]` (University of Waterloo), Robin Cohen `[通讯]` (University of Waterloo)

**通讯引用:** 11457 | [OpenAlex ID](https://openalex.org/A5000636604)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种黑盒在线调优控制器 SLO‑Tuner，利用短时端到端测量与 hill‑climb 方法，在保持 P99 延迟 SLO 的前提下最大化 LLM 服务的 goodput，并结合离散事件模拟器进行预探索。

**💡 创新点**

创新点包括：① 将 P99 SLO 作为优先目标，优化 goodput 而非平均吞吐；② 将 speculative decoding 视为可调运行时参数，并设计逻辑 knobs（队列压力、批处理、投机度）与适配器实现跨堆栈可移植；③ 通过轻量级离散事件模拟器与实时系统联动，实现低成本的趋势验证与安全搜索；④ 在实际 vLLM 上完成 8 步 hill‑climb，显著提升性能。

**🔧 技术方法**

技术手段：黑盒在线控制、端到端延迟测量、hill‑climb 搜索、离散事件模拟、P99 与 goodput 计算、SLO‑Tuner 逻辑 knobs 与适配器、vLLM 集成、GPU（NVIDIA L40S）与 Apple Silicon（MLX）实验。

**📊 数据集**

数据集与工作负载：TinyLlama‑1.1B‑Chat（1.1B 参数）、Qwen‑0.6B、Qwen‑4B；单 prompt 生成（约 64 词输出）及 Poisson / 负载 burst 交替；模拟器采用 log‑normal 长度分布模拟大 prompt/中等输出场景。

**📈 对比分析**

比较方法：与离散事件模拟的网格基线、burst/steady 流量下的 P99–goodput Pareto 前沿对比；在实测 vLLM 上对比默认配置与 SLO‑Tuner 路径。结果：P99 从 1.36 s 降至 0.70 s，goodput 从约 8 rps 提升至 15 rps，且在 1.2 s SLO 下保持稳定。

**⚠️ 局限性**

局限性：仅评估小模型、单 GPU 单机、合成负载；未覆盖多租户、GPU 集群调度、多模型多 GPU 等生产场景；hill‑climb 为局部搜索，可能陷入局部最优；控制器重启服务器导致耗时；对非稳态或极端 burst 流量的适应性有限。

---

## 89. Learning Visuomotor Policy for Multi-Robot Laser Tag Game

**arXiv ID:** 2603.11980 | [PDF](https://arxiv.org/pdf/2603.11980v1)

**作者:** Kai Li `[一作]` (Zhejiang University), Shiyu Zhao `[通讯]` (Westlake University)

**通讯引用:** 4730 | [OpenAlex ID](https://openalex.org/A5052346042)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种端到端视觉运动策略，用单目相机直接控制多机器人激光标枪游戏。

**💡 创新点**

创新点包括使用专门的可置换特征提取器、深度热图输入以及教师-学生特权模仿学习，显著提升命中率与碰撞规避。

**🔧 技术方法**

采用YOLOv5 Nano检测、DepthAnything v2生成深度图、Gaussian热图、CNN编码、LSTM记忆和MLP输出动作，辅以MADDPG训练教师策略。

**📊 数据集**

使用来自仿真环境的8万张图像和真实环境的2万张图像构成训练数据集，包含多场景混合。

**📈 对比分析**

在仿真和真实场景与经典模块化基线对比，命中率提升16.7%，碰撞率降低6%，并在Jetson Orin NX上实现20Hz控制频率。

**⚠️ 局限性**

局限性在于对深度估计与热图生成的依赖、教师策略的训练成本以及对小规模团队的验证，且在更复杂环境或更大团队时效果尚待验证。

---

## 90. Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans

**arXiv ID:** 2603.11640 | [PDF](https://arxiv.org/pdf/2603.11640v1)

**作者:** Sizhong Qin `[一作]` (Tsinghua University), Xinzheng Lu `[通讯]` (Tsinghua University)

**通讯引用:** 13024 | [OpenAlex ID](https://openalex.org/A5091489017)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HouseMind统一框架，实现建筑平面图的理解、生成和编辑。

**💡 创新点**

创新点在于使用分层VQ‑VAE离散化房间实例为令牌，并在单一序列模型中统一三大任务，结合多模态对齐和指令微调。

**🔧 技术方法**

技术包括VQ‑VAE离散化、跨模态嵌入初始化、LLM自回归建模、指令微调（SFT）以及层次化令牌化。

**📊 数据集**

主要使用RPLAN平面图数据集，并通过Qwen3‑30B‑A3B自动生成文本描述以构建预训练语料。

**📈 对比分析**

在理解、生成和编辑三任务上与LLaVA、Qwen‑VL、ChatHouseDiffusion、FloorPlan‑LLaMA等基线比较，HouseMind在微/Macro IoU、FID、GED、Node‑F1等指标上均显著优于对手，生成精度与可控性提升10%+。

**⚠️ 局限性**

局限在于编辑仅支持简单增删房间，未建模门窗家具，且缺乏对美学与专业设计偏好的完全对齐。

---

## 91. LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference

**arXiv ID:** 2603.11605 | [PDF](https://arxiv.org/pdf/2603.11605v1)

**作者:** Junkun Jiang `[一作]` (Hong Kong Baptist University), Jie Chen `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 20936 | [OpenAlex ID](https://openalex.org/A5100333005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 Labanotation 的可解释运动表示 LabanLite，并基于此构建了文本→LabanLite→运动的生成框架 LaMoGen，利用 LLM 进行符号规划与细节补全

**💡 创新点**

核心创新在于将 Labanotation 的符号化、帧级标注与文本模板相结合，形成离散但可解释的中间表示；同时引入检索增强的 LLM 提示，使 LLM 能自发生成符号序列；并设计了基于 Laban 的多维评估基准

**🔧 技术方法**

使用 LLM（GPT‑4.1、Qwen3 等）进行概念级符号合成；Transformer 作为 Laban‑motion 编码器-解码器；离散码表学习；检索增强提示；CLIP 用于检索相似文本；二阶段生成（概念+细节）

**📊 数据集**

自建 HumanML3D‑Laban 数据集（从 HumanML3D 提取，半自动注释为 Laban 与文本），以及公开的 HumanML3D 与 KIT‑ML 数据集

**📈 对比分析**

在 Laban Benchmark 以及 HumanML3D、KIT‑ML 上与 MDM、ReMoDiff、MoDiff、KP、CoMo、MotionGPT 等 SOTA 方法对比。LaMoGen 在 Laban 指标（SMT、TMP、HMN）、R‑Precision、FID 等上均居于前列，尤其在文本‑运动对齐与可解释性方面表现突出；在传统 FID 上略逊于部分基线，但整体性能保持 top‑3

**⚠️ 局限性**

由于 LabanLite 的符号抽象只能捕捉高层语义，无法表达不同个体同一动作在速度、细节上的细微差异，导致在 FID 等分布相似度指标上略显劣势；此外 LLM 对检索示例与遮蔽比例等超参数敏感，需精细调优

---

## 92. Rotatable Antenna Enabled Covert Communication

**arXiv ID:** 2603.11716 | [PDF](https://arxiv.org/pdf/2603.11716v1)

**作者:** Qi Dai `[一作]` (South China University of Technology), Chengwen Xing `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5008383657)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种利用可旋转天线（RA）实现隐蔽通信的系统，并通过联合优化波束成形与天线旋转角度来最大化合法用户的隐蔽传输速率。

**💡 创新点**

创新点在于首次将可旋转天线引入隐蔽通信领域，利用天线方向度可调的空间自由度显著提升隐蔽速率，并通过交替优化算法克服非凸耦合难题。

**🔧 技术方法**

采用了RA技术、第二次序锥规划（SOCP）、连续凸逼近（SCA）以及交替优化（AO）框架来实现性能优化。

**📊 数据集**

实验采用仿真数据，设定天线数量、距离、噪声功率等参数，并无使用公开数据集。

**📈 对比分析**

与固定天线、随机取向和等向天线基准方案对比，仿真表明RA系统在不同天线数、距离及指向角下均取得更高的隐蔽速率，且收敛速度约为8次迭代。

**⚠️ 局限性**

主要限制包括对完整CSI的假设、仅考虑自由空间LoS信道、以及机械/电子旋转实现的实际成本与时延等未进一步讨论。

---

## 93. Meta-Reinforcement Learning with Self-Reflection for Agentic Search

**arXiv ID:** 2603.11327 | [PDF](https://arxiv.org/pdf/2603.11327v1)

**作者:** Teng Xiao `[一作]` (Allen Institute for AI), Hannaneh Hajishirzi `[通讯]` (Allen Institute for AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MetaSearch——一种在 agentic search 任务中通过跨 episode 自我反思进行多轮强化学习的 meta‑RL 框架。

**💡 创新点**

创新点包括：① 将 in‑context meta‑RL 引入无环境反馈的多工具搜索任务；② 通过显式自我反思把前一次 episode 的经验作为上下文；③ 引入无 critic 的 turn‑level grouped advantage（RLOO）进行细粒度信用分配；④ 在训练阶段实现多轮反思，使搜索过程从独立尝试转变为逐步深化的探索。

**🔧 技术方法**

使用技术包括：大型语言模型（Qwen‑Series）+ ReAct 交互框架；多轮强化学习（PPO‑style clipped surrogate）与 RLOO advantage 结合；自我反思提示和上下文管理；微 episode（工具调用级别）扩展；离线离群点去除工具输出的损失掩码。

**📊 数据集**

实验数据集涵盖单跳 QA（NQ、TriviaQA、PopQA）、多跳 QA（HotpotQA、2WikiMultiHopQA、Musique、Bamboogle）以及更复杂的 ASearcher；使用 2018 年 Wikipedia 作为知识库。

**📈 对比分析**

与 ReSearch、Search‑R1、PPRM、StepResearch 等基线在所有 benchmark 上对比。MetaSearch 在 Qwen‑2.5‑3B/7B 上平均提升 9.2%–19.3%，在 ASearcher 上提升 10.2% EM、9.5% F1；在多跳 QA 上相对基线提升显著，表明自我反思与 meta‑RL 能显著提高搜索效率与答案质量。

**⚠️ 局限性**

局限性：① 未评估长篇生成任务；② 仅使用固定 Wikipedia 搜索工具，未探讨多工具混合环境；③ 未在更大规模模型或更长训练周期下验证扩展性。

---

## 94. DNS-GT: A Graph-based Transformer Approach to Learn Embeddings of Domain Names from DNS Queries

**arXiv ID:** 2603.11200 | [PDF](https://arxiv.org/pdf/2603.11200v1)

**作者:** Massimiliano Altieri `[一作]` (European Commission), Ignacio Sanchez `[通讯]` (European Commission)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5101579730)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并训练了 DNS-GT 模型，对 DNS 查询序列进行无监督预训练（掩码语言模型），随后微调用于域名分类和僵尸网络检测。

**💡 创新点**

创新点在于：① 将 Transformer 的自注意力与图注意力网络（GAT）结合，在 DNS 查询序列上建模上下文与结构；② 引入可定制的知识图谱拓扑以限制注意力范围；③ 采用主机与域名双嵌入并通过权重合并，支持主机隐私控制。

**🔧 技术方法**

使用技术包括：Transformer+GAT、masked language modeling、multi‑head attention、残差连接、批归一化、dropout、不同序列化策略（固定长度、贪婪时间、聚类时间）以及端到端与外部分类器两种微调方式。

**📊 数据集**

实验数据来自 TI‑2016 校园网络 DNS 流量（约 1.3 M 请求用于预训练，0.31 M 用于评估），并结合 Firebog 黑名单（169 k 恶意域名）做标签；同时在相同数据上进行僵尸网络检测。

**📈 对比分析**

与 Word2Vec CBOW/SkipGram 的外部分类器和端到端模型对比，DNS‑GT 在所有序列化策略下的 ROC‑AUC 和 F1 分数均明显优于基线；在端到端密度策略下达到最高 AUC 0.848，F1 最佳 0.654。

**⚠️ 局限性**

局限性包括：① 训练和微调时间较长、参数量高（约 24 M）；② 在外部分类器上表现不佳，主要受限于上下文信息无法被利用；③ 仅在 DNS 数据上验证，缺乏跨源/多协议的通用性评估；④ 对多主机攻击检测的能力仍有限。

---

## 95. The Attack and Defense Landscape of Agentic AI: A Comprehensive Survey

**arXiv ID:** 2603.11088 | [PDF](https://arxiv.org/pdf/2603.11088v1)

**作者:** Juhee Kim `[一作]` (University of California Berkeley), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并构建了 AI 代理安全的完整框架，涵盖设计维度、攻击景观、风险分类、对策与案例分析。

**💡 创新点**

首次从系统视角全面梳理 AI 代理安全，提出七维设计框架、统一的攻击与风险分类、对策分层体系，并结合 AutoGPT 等案例揭示现实安全缺口。

**🔧 技术方法**

采用系统化方法与文献综述，结合 MITRE ATLAS、OWASP Top 10 等已有安全框架，对不同攻击向量与防御技术进行归纳与对比。

**📊 数据集**

未使用传统数据集；主要基于 2023‑2025 年间 128 篇学术论文、预印本、行业白皮书及 CVE 报告构建知识库。

**📈 对比分析**

通过对比已公开的 51 种攻击方法与 60 种防御手段，对现有系统进行风险映射与防御覆盖评估，发现大多数代理缺乏完整的输入/输出 guardrail、信息流控制与多维度防御，性能评估依赖已有研究而非实验数据。

**⚠️ 局限性**

限制包括：缺乏统一评估基准与可复现的实验；防御技术在真实环境中的兼容性与可扩展性不足；安全框架对动态、跨域代理的细粒度控制仍不完善；缺少标准化的身份与访问管理规范。

---

## 96. V2A-DPO: Omni-Preference Optimization for Video-to-Audio Generation

**arXiv ID:** 2603.11089 | [PDF](https://arxiv.org/pdf/2603.11089v1)

**作者:** Nolan Chan `[一作]` (Chinese University of Hong Kong), Dingdong Wang `[通讯]` (National Research Council Canada)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 V2A-DPO 框架，对流式视频到音频模型进行直接偏好优化，使生成的音频更符合人类偏好。

**💡 创新点**

创新点包括：1) AudioScore —— 一套多维度的人类偏好对齐评分系统；2) 基于 AudioScore 的自动化偏好对齐数据生成管线；3) 结合课程学习的 DPO 优化策略，先训练易区分的偏好对，再训练难区分的对，提升对齐效果。

**🔧 技术方法**

采用了流式匹配模型（Flow Matching）、直接偏好优化（DPO）、课程学习、基于 ImageBind、CLAP、Synchformer、PANNs、PESQ 等预训练模型进行特征提取与评分；训练使用 AdamW、CFG 以及 KL 散度约束。

**📊 数据集**

主要使用 VGGSound 公开数据集（310 类视频音频对）进行实验，并在该数据集的测试集上与现有模型比较。

**📈 对比分析**

与 Diffusion、Autoregressive 及其他流式 V2A 模型（如 Seeing&Hearing、FoleyCrafter、V-AURA、ThinkSound、Frieren）以及 DPO/DDPO 对比。结果显示，V2A-DPO 优化后的 Frieren 与 MMAudio 在 IS、IB-Score、DeSync 等指标上分别比基线提升 1.81、0.86、-0.09，且在多项指标上超越现有公开 V2A 模型。

**⚠️ 局限性**

局限性包括：1) Aesthetic appeal 难以通过 AudioScore 定量评估，需人工标注；2) 课程学习阈值与 KL 散度参数对最终性能影响大，需经验调参；3) 只在 VGGSound 上验证，缺乏跨数据集泛化评估。

---

## 97. Towards Dynamic Model Identification and Gravity Compensation for the dVRK-Si Patient Side Manipulator

**arXiv ID:** 2603.12099 | [PDF](https://arxiv.org/pdf/2603.12099v1)

**作者:** Haoying Zhou `[一作]` (Worcester Polytechnic Institute), Peter Kazanzides `[通讯]` (Johns Hopkins University)

**通讯引用:** 8060 | [OpenAlex ID](https://openalex.org/A5022619253)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

完成了dVRK-Si PSM的完整运动学与动力学建模、参数识别，并实现了实时重力补偿与计算力反馈控制。

**💡 创新点**

首次为dVRK-Si PSM提供闭链改进DH运动学、基于Euler–Lagrange的线性参数逆动力学、优化激励轨迹的物理一致性参数识别，以及可直接嵌入dVRK软件的简化重力补偿模型。

**🔧 技术方法**

采用改进的Denavit–Hartenberg参数、Euler–Lagrange动力学推导、线性回归与QR分解提取基参数、基于傅里叶级数的激励轨迹优化、凸优化与物理可行性约束识别，并使用Python/C++实现实时计算。

**📊 数据集**

通过在dVRK-Si PSM上执行周期激励轨迹（0.18 Hz，6阶谐波）收集200 Hz关节位置、速度、加速度与电机扭矩数据；并在静态关节位置测量用于简化模型。

**📈 对比分析**

与实验测量进行NRMSE对比（前三关节误差约7–22 %），在静态保持时重力补偿将关节误差降低68–84 %，末端漂移从4.2 mm降至0.7 mm；计算力前馈进一步将位置误差降低约35 %/40 %比单独重力或PID更优。

**⚠️ 局限性**

仅在单一零度倾斜的dVRK-Si PSM上验证，未评估多角度、多工具和多硬件实例；实时计算受Python延迟和通信延迟限制；低扭矩关节模型误差较大，需改进摩擦、后效和传输非线性建模。

---

## 98. Social Distancing Equilibria in Games under Conventional SI Dynamics

**arXiv ID:** 2603.12107 | [PDF](https://arxiv.org/pdf/2603.12107v1)

**作者:** Connor D Olson `[一作]`, Timothy C Reluga `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文对有限时段的SI社会疏离博弈进行建模，并给出了其唯一的纳什均衡以及相应的演化稳定策略（ESS）。

**💡 创新点**

创新点在于首次给出该博弈的完整解析唯一性证明，利用变量变换和Filippov系统实现显式求解，并证明纳什均衡与社会最优一致，消除了自由乘车问题。

**🔧 技术方法**

主要技术包括Pontryagin最大值原理、Hamiltonian分析、决策势能Φ的坐标变换、Lambert W函数求解超越方程以及相平面/分段常数策略的解析积分。

**📊 数据集**

本文为纯理论研究，没有使用真实数据集；通过符号参数（如I0、m、tf）示例说明模型行为。

**📈 对比分析**

对比方法：将纳什均衡与社会最优（无自由乘车）进行比较，证明两者一致；通过对疫情负担的分析展示不同策略下的效益变化，表明均衡策略在负担降低方面具有最优性能。

**⚠️ 局限性**

局限性：仅适用于SI模型的无折扣、阈值线性干预；对SIR等更复杂动态不适用；对不完美干预或正折扣情况的结果仍需进一步研究。

---

## 99. Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era

**arXiv ID:** 2603.12016 | [PDF](https://arxiv.org/pdf/2603.12016v1)

**作者:** Nicholas Schaub `[一作]` (Axle Research), Nathan Hotaling `[通讯]` (Axle Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一款名为Nyxus的下一代大规模图像特征提取库，支持2D/3D医学与细胞图像，提供出色的可扩展性、可重复性与易用性。

**💡 创新点**

创新点在于：①实现真正的out‑of‑core、GPU加速特征提取；②通过软硬特征可调超参数实现目标化与非目标化两种工作流；③统一兼容radiomics与cell‑profiling的特征集合；④采用开源文件格式与多接口（CLI、Python、Napari、OCI容器）提升可访问性；⑤提供IBSI标准与极速性能两种配置。

**🔧 技术方法**

技术实现：C++17 + Pybind11构建高性能后端；多线程与CUDA GPU加速；利用OME‑TIFF/NGFF、DICOM、NIfTI等标准读取；输出CSV、Arrow、Parquet、Pandas等；通过软特征超参数网格搜索实现精确的时间‑准确度折衷。

**📊 数据集**

使用了TissueNet 1.0（约2500张细胞图像）和Medical Segmentation Decathlon（MRI/CT体积）两大公开基准数据集，并生成了多尺度合成数据用于特征基准。

**📈 对比分析**

与CellProfiler、Imea、MATLAB、MITK、NIST Feature2DJava、PyRadiomics、RadiomicsJ、WND‑CHARM等主流工具在相同硬件（8核EC2 P5节点）上进行对比。结果显示Nyxus在非目标化模式下对Intensity特征最快3‑35×，Texture特征最快58‑131×；在医学影像上对RadiomicsJ/ MITK 5×以上；在多线程与GPU环境下可线性扩展，ROI尺寸与线程数均表现出良好可伸缩性。

**⚠️ 局限性**

局限性包括：IBSI兼容模式下性能下降，GPU对大量小ROI时转移瓶颈；miscellaneous类特征仍略逊于其它库；目前仅支持C++/Python/Napari，未来需加入Julia/Java；需要进一步验证对细胞成像社区标准的一致性。

---

## 100. Normative Common Ground Replication (NormCoRe): Replication-by-Translation for Studying Norms in Multi-agent AI

**arXiv ID:** 2603.11974 | [PDF](https://arxiv.org/pdf/2603.11974v1)

**作者:** Luca Deck `[一作]` (University of Bayreuth), Niklas Kühl `[通讯]` (University of Bayreuth)

**通讯引用:** 2383 | [OpenAlex ID](https://openalex.org/A5071819311)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NormCoRe框架，将人类实验迁移为多智能体AI实验，并在公平分配实验中检验其有效性。

**💡 创新点**

创新点在于将复制视为跨群体的“翻译”过程，分层对齐认知、存在、交互与干预四个层面，系统记录并解释AI与人类在规范判断上的差异。

**🔧 技术方法**

技术包括大语言模型（LLM）基础模型选择、Prompt工程与角色设定、记忆与决策启发式、基于轮询的多智能体协商协议。

**📊 数据集**

数据集为原始人类实验的 34 组 5 人公平原则选择与排名数据，配合 LLM 生成的 AI 组实验日志和决策记录。

**📈 对比分析**

通过对比人类与 AI 组在最终公平原则选择、内部一致性及协商收敛速度等指标，发现 AI 组达成一致率更高、分布更集中；不同模型与语言会显著改变结果，表明设计决策对规范判断高度敏感。

**⚠️ 局限性**

局限性包括样本规模有限、文化与语言偏向、模型版本快速迭代导致结果时变、对真实世界应用的可推广性不确定，以及对人类与 AI 之间本质差异的解释仍需进一步探究。

---

## 101. Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction

**arXiv ID:** 2603.11206 | [PDF](https://arxiv.org/pdf/2603.11206v1)

**作者:** Jingxing Zhong `[一作]` (Xiamen University), Xinguo Zhuang `[通讯]` (Xiamen University)

**通讯引用:** 3812 | [OpenAlex ID](https://openalex.org/A5100861291)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了基于文本提示的乳腺癌分割模型TextBCS，通过阶段分离视觉‑语言交互（SVLI）和证据学习（EL）提升DCE‑MRI分割精度。

**💡 创新点**

创新点包括：①阶段分离的视觉‑语言交互模块在每个下采样层实现多级文本引导；②采用变分Dirichlet证据学习量化分割不确定性；③结合交叉模态对齐损失和多级注意力，显著改善低对比度和模糊边界的分割。

**🔧 技术方法**

技术手段：多层交叉注意力的SVLI、变分Dirichlet证据学习、交叉模态对齐损失、Dice/ICE/ELKLLoss、CLIP/BioClinicalBERT文本编码、PyTorch实现。

**📊 数据集**

使用公开的 Duke‑Breast‑Cancer‑MRI 数据集，共 3876 张切片，按 7:1:2 进行训练/验证/测试划分。

**📈 对比分析**

与多种 UNet 变体（UNet、UNet++、nnUNet、TransUNet、SwinUNet、UCTransNet）和文本引导方法（CLIP、TGANet、ConVIRT、GLoRIA、MGCA、LViT）对比，TextBCS 在 Dice 85.33% / mIoU 76.08% 上优于所有基线，参数与 FLOPs 适中，p 值<0.05 证明显著性。

**⚠️ 局限性**

局限性：模型对文本提示的准确性高度依赖，复杂或不一致的提示会导致分割误差；需要人工或 LLM 生成提示，涉及隐私与错误提示风险；在未见过的文本风格下鲁棒性有限。

---

## 102. Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training

**arXiv ID:** 2603.12255 | [PDF](https://arxiv.org/pdf/2603.12255v1)

**作者:** Fangfu Liu `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**通讯引用:** 2363 | [OpenAlex ID](https://openalex.org/A5013973037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Spatial‑TTT框架，利用测试时训练的快速权重在长视频流中持续更新并存储空间信息，支持流式视觉空间推理；

**💡 创新点**

核心创新包括：①混合架构将TTT层与自注意力锚层交替使用；②大块更新与滑窗注意力并行；③引入轻量化深度可卷积的空间预测机制；④构建密集场景描述数据集驱动快速权重学习；

**🔧 技术方法**

技术实现基于Qwen3‑VL‑2B-Instruct的Transformer，使用TTT、深度可卷积、MuON优化、滑窗注意力、双KV缓存等；

**📊 数据集**

使用自建的密集场景描述数据集（约16k样本），以及3M规模的空间VQA数据集；

**📈 对比分析**

在VSI‑Bench、MindCube、VSI‑SUPER等视觉空间基准上与多种开源与专有MLLM、长视频理解模型对比，取得最高或接近最高的平均准确率，特别在多选、计数和持续空间感知任务上显著优于基线；

**⚠️ 局限性**

局限性主要在于：①对预训练模型的依赖较强；②在极长视频或高分辨率输入下仍面临计算与内存限制；③快速权重更新策略对不同任务的泛化尚待进一步验证。

---

## 103. Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation

**arXiv ID:** 2603.12247 | [PDF](https://arxiv.org/pdf/2603.12247v1)

**作者:** Xiangyu Zhao `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5859 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 FIRM 框架，通过专门的数据构造和奖励模型训练，提升图像编辑与生成任务的 RL 对齐效果。

**💡 创新点**

创新点包括：① 针对编辑任务的“difference-first”与生成任务的“plan-then-score”数据构造管道；② 用两阶段 MLLM 评估实现高质量奖励；③ 在 RL 中引入 Consistency‑Modulated Execution (CME) 与 Quality‑Modulated Alignment (QMA) 的奖励融合策略，显著抑制奖励劫持。

**🔧 技术方法**

技术手段涵盖：多模态大型语言模型（Qwen3-VL、Qwen3-32B、Qwen3‑VL‑8B‑Instruct）、DiffusionNFT 在线 RL、Flow‑Matching、基于视觉差异与检查表的 VQA 风格评估。

**📊 数据集**

数据集包括：FIRM‑Edit‑370K、FIRM‑Gen‑293K（通过公开编辑与生成数据集构造）以及人类标注的 FIRM‑Bench（807 样本）用于验证奖励模型。

**📈 对比分析**

通过将 FIRM‑Edit‑8B 与 FIRM‑Gen‑8B 作为奖励模型在 Edit‑R1 与 Diffusion‑NFT 框架下进行 RL，模型在 GEditBench、ImgEdit、GenEval、DPGBench、TIIF、UniGenBench 等基准上均超越现有开源与闭源方法；与 Qwen3‑VL‑8B/32B 的对比表明 FIRM 能显著提升编辑一致性与生成细节质量。

**⚠️ 局限性**

局限性包括：仍依赖大量人工标注的验证基准；奖励模型对极端复杂或高度细粒度需求的适配尚未彻底；在跨模态生成任务中，对多模态细节的解释性与可解释性还有提升空间。

---

## 104. Primitive-Root Determinant Densities over Prime Fields and Implications for PRIM-LWE

**arXiv ID:** 2603.11196 | [PDF](https://arxiv.org/pdf/2603.11196v1)

**作者:** Vipin Singh Sehrawat `[一作]` `[通讯]` (Circle Internet), Vipin Singh Sehrawat (Circle Internet)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了原始-LWE（prim‑LWE）约化中的维度均匀采样常数 c(p)，证明其在所有素数上不恒正，给出了 c(p) 的极限分布、最小值的渐近阶数、以及针对常用加密模数的显式下界。

**💡 创新点**

① 用 Dirichlet 定理和 Mertens 乘积公式无条件证明 infₚ c(p)=0；② 推导 min_{p≤x} c(p)≈1/lnlnx 的精确极限阶；③ 显示 c(p) 在素数上拥有连续纯奇异分布，支持集为 [0,½]；④ 给出仅依赖 ω(p−1) 的显式下界，适用于 NTT 友好模数。

**🔧 技术方法**

主要技术包括：Dirichlet 进阶素数定理、Mertens 第三公式、Linnik 最小素数估计、Erdős–Wintner 及其在位移素数上的变体、纯奇异性判定与连续映射定理。

**📊 数据集**

使用的“数据集”是全体素数及其前一整数的质因子分解（尤其是 NTT 友好模数 3329、8380417 等），以及已知的质因子分布统计（如 ω(p−1) 的极限分布）。

**📈 对比分析**

与先前只给出大致下界的工作相比，本研究提供了可计算的常数上界：例如 3329 的拒绝采样开销 ≤ 2.17，8380417 ≤ 3.42；对所有 q>2³⁰ 的通用上界 1/c(q)≤1.79 log q；最坏情况的上界为 O(loglog q)。

**⚠️ 局限性**

局限性：仅给出了极限分布的存在与支撑范围，未求出具体分布函数的闭式或数值特征；最小值下界与上界的精度受 Linnik 指数及 Mertens 误差影响；结果只适用于素数模数，对复数模数的推广仍需研究。

---

## 105. A Two-Stage Dual-Modality Model for Facial Emotional Expression Recognition

**arXiv ID:** 2603.12221 | [PDF](https://arxiv.org/pdf/2603.12221v1)

**作者:** Jiajun Sun `[一作]` (Shanghai Normal University), Zhe Gao `[通讯]` (Shanghai Normal University)

**通讯引用:** 363 | [OpenAlex ID](https://openalex.org/A5101556258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种两阶段轻量化的音视频双模态框架，用于在 10th ABAW 竞赛中对野外视频进行帧级情绪表达识别。

**💡 创新点**

创新点在于结合 PadAug 与 MoE 辅助的 DINOv2 视觉适配、三尺度多尺度人脸重裁与平均、轻量化门控融合音视频特征以及推理时的中值平滑，显著提升了在不受控制视频中的鲁棒性与准确性。

**🔧 技术方法**

采用的核心技术包括 DINOv2 ViT‑L/14 视觉编码器、PadAug 边缘填充数据增强、MoE 训练头、Wav2Vec 2.0 音频编码、门控融合模块以及时序中值滤波。

**📊 数据集**

主要使用的数据集为 Aff‑Wild2（作为验证与测试基准），并在 Stage I 的视觉适配阶段使用 AffectNet 与 RAF‑DB 两个大型图像级表情识别数据集。

**📈 对比分析**

与官方基线相比，本方法在验证集上取得 Macro‑F1 0.5368，5‑折交叉验证平均 0.5122 ± 0.0277，显著优于基线与前期竞赛提交方案。

**⚠️ 局限性**

限制在于仍依赖预训练的视觉/音频模型，门控融合的参数规模有限，未充分利用长时序信息，且未实现端到端训练，对极端光照、遮挡或快速姿态变化的鲁棒性仍有提升空间。

---

## 106. CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges

**arXiv ID:** 2603.11863 | [PDF](https://arxiv.org/pdf/2603.11863v1)

**作者:** Zi-Han Wang `[一作]` (Southern University of Science and Technology), Linyi Yang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3169 | [OpenAlex ID](https://openalex.org/A5035082722)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CreativeBench基准，用于评估代码生成系统的组合式与探索式创造力，并设计EvoRePE激励策略提升创造力。

**💡 创新点**

创新点在于基于Boden的认知框架构建两类任务；使用可执行代码客观区分创造与幻觉；提出统一创造度量（质量×新颖度）并验证其可靠性；提出训练自由的EvoRePE内在化进化搜索向量。

**🔧 技术方法**

使用GPT‑4.1/4o生成任务；采用代码嵌入与n‑gram距离度量新颖度；利用沙箱执行与LLM‑as‑judge评估质量；通过PCA提取激活向量实现EvoRePE；对比零样本、AlphaEvolve、GEPA。

**📊 数据集**

数据集为CreativeBench（约196个Python问题），包含两子集CreativeBench‑Combo与CreativeBench‑Explore，来源于AutoCodeBench种子并通过反向工程与自对弈生成。

**📈 对比分析**

与多种基础模型（Gemini‑3‑Pro、GPT‑5.2等）及演化基线比较，结果表明即便最强模型Pass@1也低于60%，规模提升显著提升组合式创造力但探索式下降；EvoRePE可在不增加搜索成本的情况下提升创意分数。

**⚠️ 局限性**

局限包括仅覆盖Python语言；仅评估而非训练模型；自动构建过程可能带来生成器偏差；新颖度测度对超浅编辑敏感。

---

## 107. LHGstore: An In-Memory Learned Graph Storage for Fast Updates and Analytics

**arXiv ID:** 2603.11596 | [PDF](https://arxiv.org/pdf/2603.11596v1)

**作者:** Pengpeng Qiao `[一作]` (Institute for Science Tokyo), Yang Cao `[通讯]` (Institute for Science Tokyo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种名为LHGstore的内存动态图存储框架，通过两层学习索引和度量感知的存储布局，实现了高吞吐量的更新和低延迟的图分析。

**💡 创新点**

创新点在于首次将学习型索引集成到图存储，构建了顶层顶点索引和底层边索引的两层层次结构，并根据顶点度数动态选择无序数组或学习索引，从而兼顾更新效率与遍历局部性。

**🔧 技术方法**

主要技术包括自适应线性回归学习索引、gapped数组、无序数组、翻译表以及两层层次索引的设计，配合度数阈值T来切换存储方式。

**📊 数据集**

使用了Graph500-24、Graph500-26、Orkut和LiveJournal四个真实/合成图数据集进行评测。

**📈 对比分析**

通过与Teseo、Sortledton、LiveGraph、Aspen、LSGraph和LGstore等六种基线系统在写、读写混合、读三种工作负载以及五种图算法（BFS、PageRank、LCC、WCC、SSSP）下的吞吐量和运行时进行对比，LHGstore在写和读写混合场景下实现了6.6–28.2倍的吞吐量提升，图分析时间比最慢基线快5.9–17.4倍，且内存占用保持在竞争水平。

**⚠️ 局限性**

局限性包括对度数阈值T的手工调参需要针对不同图分布；在极端高度顶点场景下学习索引的预测误差仍可能导致查找或插入的额外扫描；以及对极大规模图的整体内存占用仍略高于最轻量级基线。

---

## 108. Ensuring Safety in Automated Mechanical Ventilation through Offline Reinforcement Learning and Digital Twin Verification

**arXiv ID:** 2603.11372 | [PDF](https://arxiv.org/pdf/2603.11372v1)

**作者:** Hang Yu `[一作]` (University of Warwick), Declan Bates `[通讯]` (University of Warwick)

**通讯引用:** 3131 | [OpenAlex ID](https://openalex.org/A5077819322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种Transformer‑based Conservative Q‑Learning（T‑CQL）框架，用于机械通气的个性化与自动化，并通过数字孪生进行“床边”评估。

**💡 创新点**

创新点包括：①使用Transformer编码器对患者历史状态进行时序建模；②基于不确定性量化的自适应保守正则化，提升安全性；③加入序列一致性正则化以增强鲁棒性；④设计结合VILI指标和病情严重度的临床奖励；⑤采用数字孪生而非仅FQE进行在线评估，弥补传统OPE的局限。

**🔧 技术方法**

技术手段：Transformer Encoder、Conservative Q‑Learning、epistemic uncertainty quantification、consistency regularization、offline RL、Fitted Q‑Evaluation、基于计算机的心肺数字孪生模拟。

**📊 数据集**

数据集：MIMIC‑III 与 MIMIC‑IV 公开 ICU 数据（约11,585名患者，994,080小时的PCV记录）以及构造的98个基于真实患者参数的数字孪生。

**📈 对比分析**

比较方法：与临床医生、DeepVent (CQL)、BCQ、DDQN 等现有离线RL模型进行对比。T‑CQL 在FQE得分（0.87）和预测死亡率（0.16）上均优于其他方法；在数字孪生评估中，安全约束满足率（47.96%）和驱动力降低率（44.9%）均高于临床基线（43.88%/37.8%），并且其动作分布最接近临床医生的实践。

**⚠️ 局限性**

局限性：传统FQE评估在分布偏移下可能产生误导；本研究仅在模拟环境验证，缺乏真实临床试验；奖励设计仍基于有限指标，尚未评估长期安全性和更广泛的临床适用性。

---

## 109. BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning

**arXiv ID:** 2603.12176 | [PDF](https://arxiv.org/pdf/2603.12176v1)

**作者:** Jingyang Ke `[一作]` (Georgia Institute of Technology), Anqi Wu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 842 | [OpenAlex ID](https://openalex.org/A5034063987)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出BehaviorVLM，一个统一的视觉语言框架，既可在不进行任务特定微调的情况下完成动物姿态估计，也能实现多动物行为分割与语义描述。

**💡 创新点**

创新点在于通过多阶段视觉语言推理将任务拆解为区域检测、关键点分配、跨视角一致性与三维校正，极大降低人工标注；同时将深度嵌入聚类、VLM生成短片描述和LLM语义合并相结合，完成从视觉信息到可解释行为标签的闭环。

**🔧 技术方法**

使用预训练的 Vision‑Language Model（如Qwen 3.5-27B/35B）、大语言模型（Qwen3‑Next‑80B）、RANSAC三角化、Deep Embedded Clustering、Segment‑Anything 3、LookAgain特征融合等。

**📊 数据集**

在自制的六视角量子点小鼠数据集（500帧）和公开的MABe2022 Mouse Triplets（多鼠交互）上进行实验。

**📈 对比分析**

与基线方法（无三维校正的滚动提示、纯关键点模型、传统无监督行为分割）相比，完整的BehaviorVLM在姿态误差上提升约46%（从14.29mm降至6.59mm），在行为分割上能得到与人工标签一致的语义区间，并提供可读的行为描述。

**⚠️ 局限性**

限制在于仍依赖高质量的多视角摄像和量子点标记；对极端遮挡或低光照条件下的精度有待验证；VLM与LLM推理速度较慢，实时性受限。

---

## 110. EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**arXiv ID:** 2603.12267 | [PDF](https://arxiv.org/pdf/2603.12267v1)

**作者:** Tianwei Xiong `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 3881 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了EVATok，一种内容自适应视频分词框架，能够为每个视频预测最优的token分配，从而提升重建与生成效率与质量。

**💡 创新点**

创新点包括：①提出代理奖励（proxy reward）指标用于寻找最优token分配；②设计轻量级router实现一次性快速预测；③构建四阶段训练框架，将代理tokenizer、router与最终自适应tokenizer融合。

**🔧 技术方法**

主要技术手段为Q-Former风格的一维tokenizer、代理tokenizer、ViT-like router、视频语义编码器进行表示对齐与判别、以及改进的训练策略（对齐损失、对抗判别器）。

**📊 数据集**

使用的数据集包括：WebVid-10M（用于router训练与代理奖励评估），UCF‑101与Kinetics‑600（用于重建与生成实验），所有视频裁剪为16×128×128帧。

**📈 对比分析**

与固定长度基线（如LARP、ElasticTok、AdapTok）及先前SOTA方法进行对比，评估指标为LPIPS、重建FVD（rFVD）和生成FVD（gFVD）。EVATok在UCF‑101的class‑to‑video生成中取得48 gFVD、比SOTA低26.2%token、并在重建任务中token长度减少24.4%，整体性能均优于基线。

**⚠️ 局限性**

局限性在于目前仅在16帧视频上验证，未证明对更长时序的适用性；高复杂场景下的分配仍可能欠佳；同时需要大规模数据预训练来构建高质量的router与tokenizer。

---

## 111. EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection

**arXiv ID:** 2603.11521 | [PDF](https://arxiv.org/pdf/2603.11521v1)

**作者:** Shuo Jiang `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 11708 | [OpenAlex ID](https://openalex.org/A5084291326)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于教师-学生框架的无监督隐蔽物体检测方法，结合多线索原始感知和演进伪标签融合，实现细节保留与边界精确。

**💡 创新点**

创新点在于三大模块：多线索原始感知（MNP）捕获低层纹理与中层语义；演进伪标签融合（PEF）通过深浅层交互和光谱张量注意融合提升伪标签质量；局部伪标签细化（LPR）利用注意头多样性恢复细节。

**🔧 技术方法**

采用DINO预训练ViT-S/8作为编码器，结合LBP/DoG纹理提取、深度可分离卷积、张量分解与SVD、注意力熵筛选等技术。

**📊 数据集**

训练集为CAMO-Train 1000张与COD10K-Train 3040张，评估集包括CHAMELEON、CAMO、COD10K、NC4K四个COD基准。

**📈 对比分析**

与多种UOS和UCOD方法对比，EReCu在S_m、F_ω^β、E_m和MAE四项指标上均优于现有无监督方法，部分指标甚至逼近或超过监督方法。

**⚠️ 局限性**

局限在于对高维特征与多头注意的依赖，计算开销较大；对极端遮蔽或极低对比度场景仍可能出现漏检。

---

## 112. Interventional Time Series Priors for Causal Foundation Models

**arXiv ID:** 2603.11090 | [PDF](https://arxiv.org/pdf/2603.11090v1)

**作者:** Dennis Thumm `[一作]` (National University of Singapore), Ying Chen `[通讯]` (National University of Singapore)

**通讯引用:** 50507 | [OpenAlex ID](https://openalex.org/A5100383082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CausalTimePrior，一个生成配对观测与干预时间序列的先验，用于训练基于 PFN 的时间序列因果推断模型，并展示了其在模拟数据上的效果。

**💡 创新点**

创新点在于（1）提供了支持多种干预类型（硬、软、时变）的离散时间结构因果模型（TSCM）先验；（2）首次结合马尔可夫式结构切换（regime‑switching）来模拟时间变化的因果关系；（3）生成的干预数据可直接用于预训练 PFN，实现零样本因果效应估计。

**🔧 技术方法**

技术包括：基于 Erdős‑Rényi 生成图结构、可配置非线性自回归机制、对干预进行解析式修改、Markov 切换模型实现结构变换、GRU‑基础的 PFN 进行预训练与评估。

**📊 数据集**

主要使用了 CausalTimePrior 自生成的 100K 条 TSCM 训练集以及 1K 条留出的测试集，干预类型覆盖硬、软、时变，包含约 15% 的结构切换模型。

**📈 对比分析**

与传统 VAR 线性基准以及无干预训练的 PFN 进行对比；在预测干预结果时，PFN 的预测/真实比率达 0.95，非干预查询为 0.46，RMSE 与 VAR 基线相当，显示出可在不需样本级拟合的情况下完成因果效应估计。

**⚠️ 局限性**

局限包括：仅假设马尔可夫噪声且离散时间动态；未对真实世界因果时间序列进行验证；Erdős‑Rényi 图先验未显式覆盖所有典型因果结构；对连续时间或非马尔可夫性因果关系的处理仍待扩展。

---

## 113. Single-View Rolling-Shutter SfM

**arXiv ID:** 2603.11888 | [PDF](https://arxiv.org/pdf/2603.11888v1)

**作者:** Sofía Errázuriz Muñoz `[一作]` (KTH Royal Institute of Technology), Kathlén Kohn `[通讯]` (Digital Futures)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种针对滚动快门相机的单视图结构从运动（SfM）方法，系统地探讨了如何从单个RS图像中恢复运动和场景参数。

**💡 创新点**

创新点在于系统性地探索了单视图RS SfM，提供了基础理论，描述了RS图像捕获的几何特性，并推导出从单个RS图像中可重建的相机和3D场景信息。

**🔧 技术方法**

使用了Cayley参数化的多项式运动模型来描述相机的中心和方向，结合代数几何的方法来分析RS图像的性质。

**📊 数据集**

使用了多个合成数据集和真实数据集进行实验，包括iPhone 3GS序列和其他合成图像序列。

**📈 对比分析**

与现有的多视图方法进行比较，本文的方法在特定情况下表现出可行性和实用性，但在噪声影响下性能有所下降，尤其是在高频特征被噪声主导的情况下。

**⚠️ 局限性**

限制在于目前的理论和实验主要集中在特定的相机模型和场景设置上，未来需要扩展到更复杂的场景和多视图SfM问题。

---

## 114. Shadowless Projection Mapping for Tabletop Workspaces with Synthetic Aperture Projector

**arXiv ID:** 2603.11551 | [PDF](https://arxiv.org/pdf/2603.11551v1)

**作者:** Takahiro Okamoto `[一作]` (Osaka University), Daisuke Iwai `[通讯]` (Osaka University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了基于25个投影仪的合成孔投影映射系统，实现桌面工作区无延迟、无阴影的增强现实投影，并提出了子像素模糊补偿方法和“投影感知”（SoP）评估指标。

**💡 创新点**

创新点包括：①利用合成孔技术将多投影仪的光源融合为等效大孔径投影，实现阴影即时消除；②提出与投影仪数量无关的模糊补偿算法，显著降低计算时间；③首次引入SoP指标通过用户研究量化投影对表面反射感知的影响，并给出投影仪布置的实用准则。

**🔧 技术方法**

采用多投影仪阵列（5×5）与灰度码标定、相机-投影仪光传输矩阵、光场合成、子像素对齐与卷积逆算、基于深度摄像头/图像传感器的光照补偿。

**📊 数据集**

实验数据集主要包括：平面桌面（1200×600 mm）、非平面人像泥塑（300×250×600 mm）、A5纸张（文字/投影文本）等自制样本；无公开公共数据集。

**📈 对比分析**

通过与单投影仪系统、传统模糊补偿方案的对比；性能指标：PSNR从19.54提升至20.05、SSIM从0.44提升至0.50，补偿计算时间从2732 s降至109 s（相当于25倍速度提升），并在用户实验中SoP评分显著降低，识别任务时间增加、准确率下降。

**⚠️ 局限性**

局限性在于：①需要大量投影仪（25个或以上）导致硬件成本高、部署空间受限；②单台PC可连接投影仪数有限，系统可扩展性受限；③标定过程耗时长（每台投影仪需单独投影灰度码）；④当前仅适用于桌面级或中等尺寸场景，复杂3D物体仍需进一步研究。

---

## 115. OSM-based Domain Adaptation for Remote Sensing VLMs

**arXiv ID:** 2603.11804 | [PDF](https://arxiv.org/pdf/2603.11804v1)

**作者:** Stefan Maria Ailuro `[一作]` (Sofia University), Danda Pani Paudel `[通讯]` (Sofia University)

**通讯引用:** 1986 | [OpenAlex ID](https://openalex.org/A5050696776)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用OpenStreetMap地图信息对遥感视觉语言模型进行自监督式领域适配的方法，生成海量无人工标注的遥感图像-文本对并微调模型。

**💡 创新点**

创新点在于：①将OSM地图渲染为可视化图层，让模型自身利用OCR与图表理解能力生成带地理语义的caption；②构建OSMDA-Captions数据集；③采用自含式训练，完全不依赖外部教师模型，成本大幅降低。

**🔧 技术方法**

技术手段包括：基于Mapnik渲染OSM-carto风格地图；利用基础VLM（InternVL3.5-8B）进行图像-地图双模态提示生成caption；使用LoRA微调、混合精度训练；统一的评价框架（G‑Eval、F1、MAE）。

**📊 数据集**

使用的数据集：SkyScript训练集（约150万卫星图像）与OSM数据；OSMDA-Captions（20万图像-文本对）；10个遥感基准（NWPU-Captions、UCM-Captions、VRSBench、RSVQA-HR/LR、AID、Million-AID、EuroSAT、XLRS-Bench、SkyScript-Bench）。

**📈 对比分析**

与9个现有基准模型及教师蒸馏方法对比，OSMDA‑VLM在10个基准中获6/10榜首，零射击通用化性能居前列，且成本仅为传统教师蒸馏的1/10左右。

**⚠️ 局限性**

局限性包括：依赖OSM地图质量，缺乏覆盖度低的区域会导致性能下降；模型在复杂多用途场景下的描述不足；可能在特定领域标签上出现偏见。

---

## 116. OrthoEraser: Coupled-Neuron Orthogonal Projection for Concept Erasure

**arXiv ID:** 2603.11493 | [PDF](https://arxiv.org/pdf/2603.11493v1)

**作者:** Chuancheng Shi `[一作]` (University of Sydney), Zhiyong Wang `[通讯]` (University of Sydney)

**通讯引用:** 31628 | [OpenAlex ID](https://openalex.org/A5100614129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出OrthoEraser框架，通过稀疏自编码器定位敏感神经元并在其正交投影中抑制敏感概念，避免对正常语义的损害；

**💡 创新点**

创新点在于将概念抑制视为几何正交投影问题，在稀疏分解后构造保护子空间，使干预向量在其正交补空间内，精准去除敏感内容而不破坏生成流形；

**🔧 技术方法**

采用稀疏自编码器（SAE）进行特征分解、耦合神经元检测、正交投影抑制以及梯度正交化技术；

**📊 数据集**

在Stable Diffusion 1.4、FLUX.1 Dev、Show-o2等模型上使用I2P、P4D、Ring-A-Bell、MS COCO等数据集进行安全性、鲁棒性和生成质量评估；

**📈 对比分析**

与ESD、UCE、SNCE等SOTA方法对比，OrthoEraser在I2P中的性别裸露检测降至5例（近零），CLIP Score仅下降0.01，FID降至1.15，且在鲁棒性测试中攻击成功率从98.7%降至2.7%，表现出更优的安全-质量权衡；

**⚠️ 局限性**

局限性包括对λ等超参数的敏感性、在极端攻击或极端安全需求场景下可能仍出现残留敏感信号，以及对模型结构差异的适配需进一步验证。

---

## 117. MirrorDrift: Actuated Mirror-Based Attacks on LiDAR SLAM

**arXiv ID:** 2603.11364 | [PDF](https://arxiv.org/pdf/2603.11364v1)

**作者:** Rokuto Nagata `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**通讯引用:** 2928 | [OpenAlex ID](https://openalex.org/A5055467060)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种基于可操控平面镜的物理攻击框架MirrorDrift，利用镜面反射在LiDAR SLAM中注入虚假点，导致定位漂移。

**💡 创新点**

创新点在于首次将镜面反射作为注入式攻击通道，并通过镜子放置、对齐与周期性偏航运动的优化，显著放大SLAM误差；该攻击不依赖于传感器内部时序，可绕过现代LiDAR的干扰抑制。

**🔧 技术方法**

采用了光学反射仿真、贝叶斯优化求解镜子参数、周期性偏航控制以及对点云的遮挡与投影计算，并结合KISS-ICP、FAST‑LIO2、GLIM等开源SLAM算法进行实验。

**📊 数据集**

使用VLP‑32c、Horizon、HAP、AT‑128等实测LiDAR数据，以及自建的室内室外测试轨迹和RTK‑GNSS参考轨迹。

**📈 对比分析**

与无镜、随机放置镜以及仅遮挡/仅反射的基线进行对比，实验显示优化放置的镜子使平均位姿误差提升6.1倍、真实硬件测试中多模SLAM系统出现3.4–6.0 m位姿误差，优于随机或无镜基线。

**⚠️ 局限性**

局限在于需要靠近目标路径（≤3 m），镜子需手动或受风干扰影响；对动态镜子反射点的检测与多模冗余等防御机制仍未成熟。

---

## 118. Exponential-Family Membership Inference: From LiRA and RMIA to BaVarIA

**arXiv ID:** 2603.11799 | [PDF](https://arxiv.org/pdf/2603.11799v1)

**作者:** Rickard Brännvall `[一作]` (RISE Research Institutes of Sweden), Rickard Brännvall `[通讯]` (RISE Research Institutes of Sweden)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5077095868)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文统一了LiRA、RMIA和BASE三种成员推断攻击，并提出了Bayesian方差推断攻击BaVarIA，提升了小样本shadow模型下的性能。

**💡 创新点**

创新点在于构建指数族对数似然比框架，将三种攻击归为同一模型并提出BASE1–4层级；引入正态-逆伽马先验实现Bayesian方差估计，消除阈值切换带来的不连续性。

**🔧 技术方法**

使用指数族统计、对数似然比（LLR）、高斯/指数分布假设、正态-逆伽马（NIG）先验、贝叶斯后验预测（Student‑t）、多层级BASE架构以及实验对比等技术。

**📊 数据集**

实验数据集包含12个公开数据集（6个图像：CIFAR‑10/100、CINIC‑10；6个表格：Location、Purchase100、Texas100等），分别训练ResNet/WideResNet/MLP模型。

**📈 对比分析**

与LiRA、RMIA、BASE3等方法在12个数据集上对比，BaVarIA‑t在所有shadow预算下获得最高AUC，BaVarIA‑n在低FPR场景下性能更稳健，且在小shadow模型数（K≤16）时优于LiRA。

**⚠️ 局限性**

局限在于实验仅基于Design B采样、shadow模型来源相同、未考察跨域迁移；NIG超参数采用经验贝叶斯默认值，且对极低FPR评估精度有限。

---

## 119. Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints

**arXiv ID:** 2603.11755 | [PDF](https://arxiv.org/pdf/2603.11755v1)

**作者:** Chenyangguang Zhang `[一作]` (ETH Zurich), Xi Wang `[通讯]` (ETH Zurich)

**通讯引用:** 2875 | [OpenAlex ID](https://openalex.org/A5021428067)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于稀疏三维手关节的前视视角视频生成框架，可通过单帧参考图像和可控的3D关节轨迹生成高保真、运动可控的第一人称视频。

**💡 创新点**

创新点包括：①遮挡感知的三维关节控制模块，利用源帧遮挡惩罚和目标帧3D加权来解决相互遮挡问题；②直接将3D几何与语义嵌入注入生成器，保持空间一致性；③构建大规模自动标注流水线，提供百万级精确手轨迹及跨体型机器人数据集。

**🔧 技术方法**

技术手段包括：基于WAN 2.1的流匹配Diffusion Transformer；Low‑Rank Adaptation (LoRA) 进行轻量化微调；Gaussian热图聚合、遮挡概率模型、3D加权软最大、3D几何编码、因果3D卷积等；以及人机交互式编辑接口。

**📊 数据集**

使用的数据集包括：Ego4D（≈1M视频+手轨迹）、EgoDex（桌面抓取数据）、Humanoid Everyday（两个机器人平台的前视机器人视频+手轨迹），以及自研的机器人轨迹生成和校准流程。

**📈 对比分析**

与Mask2IV、WAN‑Fun、WAN‑Move、MotionStream等基线相比，本文在Ego4D、EgoDex和机器人数据集上显著提升：FVD降低16%~52%，MPJPE下降68%~89%，FID/PSNR/SSIM也均优于竞争方法；在交叉体型机器人上表现尤为突出，跨体型泛化显著。

**⚠️ 局限性**

局限性：仍依赖高质量的3D关节标注，遮挡惩罚与权重参数需要手动调优；对极端光照、快速运动或复杂多手交互场景的鲁棒性尚未彻底验证；生成模型对非常细小的动作（如单指轻微抖动）精度有限。

---

## 120. DyWeight: Dynamic Gradient Weighting for Few-Step Diffusion Sampling

**arXiv ID:** 2603.11607 | [PDF](https://arxiv.org/pdf/2603.11607v1)

**作者:** Tong Zhao `[一作]` (Zhejiang University), Chi Zhang `[通讯]` (AGI Lab Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种学习型多步解算器 DyWeight，用于在极少步数下高效采样扩散模型。

**💡 创新点**

创新点在于放宽权重求和为1的传统约束，采用无约束、时变权重实现梯度聚合与步长自适应的隐式耦合，并通过时间平移与缩放实现时间校准，避免了手工时间表和多阶段优化。

**🔧 技术方法**

核心技术包括：无约束时变梯度权重、隐式时间校准（平移+缩放）、端点监督的教师-学生蒸馏框架、以及轻量化单轮训练。

**📊 数据集**

在多种数据集上验证：像素空间（CIFAR-10, ImageNet64, FFHQ, AFHQv2）、潜在空间（LSUN-Bedroom, MS-COCO）以及最新文本到图像模型 FLUX.1-dev。

**📈 对比分析**

与传统手工多步解算器（iPNDM、DPM‑Solver++、UniPC）及学习型解算器（AMED, EPD, LD3, DLMS, S4S‑Alt）对比，DyWeight 在所有实验设置下都实现了更低的 FID/CLIP 分数，且仅需极少的 NFE，达成了低-NFE 下的最优性能。

**⚠️ 局限性**

局限性包括：仍依赖教师解算器生成高质量轨迹，端点监督可能导致非最优中间路径；对极大步长或高度非线性时间表的适应性尚未完全验证；以及在一些潜在空间模型中可能存在提示对齐（CLIP）稍逊的现象。

---

## 121. A Stable Neural Statistical Dependence Estimator for Autoencoder Feature Analysis

**arXiv ID:** 2603.11428 | [PDF](https://arxiv.org/pdf/2603.11428v1)

**作者:** Bo Hu `[一作]` (University of Florida), Jose C Principe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于正交密度比分解的稳定神经依赖估计器，并将其应用于变分自动编码器，以在静态无噪声网络中衡量输入、特征和重构之间的统计依赖。

**💡 创新点**

创新点在于：①利用正交分解逼近密度比而非直接估计密度比，避免了MINE中输入拼接和重配对的计算瓶颈；②引入一种无矩阵求逆或对数行列式的NMF风格标量损失，进一步提升稳定性与效率；③通过假设高斯噪声构造辅助变量，使静态网络的互信息测量可定义且可量化。

**🔧 技术方法**

使用正交密度比分解、非负矩阵分解（NMF）风格损失、变分自编码器的高斯假设、以及对噪声方差的经验估计；实现了多输出神经网络以学习左、右奇异函数。

**📊 数据集**

在两块数据集上验证：1）低维两月形合成数据；2）MNIST手写数字数据。

**📈 对比分析**

与传统MINE、核基方法（KDE、KICA、HSIC）以及之前的logdet、trace损失进行对比；结果显示NMF-DR损失在估计精度和收敛速度上均优于MINE，且保持了统计依赖的替代模式；学习曲线更平滑、稳定。

**⚠️ 局限性**

局限性包括：需要对高斯噪声方差进行经验调参；对奇异函数维度的选择敏感；在高维特征维度超过可用奇异值时仍可能出现偏差；目前仅在二分类或低维数据上验证，缺乏对更复杂任务的推广。

---

## 122. Making Chant Computing Easy: CantusCorpus v1.0 and the PyCantus Library

**arXiv ID:** 2603.11933 | [PDF](https://arxiv.org/pdf/2603.11933v1)

**作者:** Anna Dvořáková `[一作]` (Charles University), Jan Hajič `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了CantusCorpus v1.0数据集（888,010首 Gregorian chant 及 2,278 份来源），并开发了轻量级 Python 库 PyCantus，支持统一加载、过滤与可复现的分析流程；同时通过模糊匹配将 Corpus Monodicum 的部分数据转换为 PyCantus 兼容格式，实现跨项目整合。

**💡 创新点**

创新点在于：①将 Cantus 生态系统中 19 个数据库的 chant 记录统一打包为单一可下载数据集；②提供完整的 ETL 代码与 PyCantus 统一数据模型，降低科研门槛与提升可复现性；③演示了 PyCantus 在无 Cantus ID 的项目（Corpus Monodicum）中的可扩展性，开启了跨数据库互操作的范例。

**🔧 技术方法**

采用的技术包括：Web 爬取 Cantus Index 的 JSON 接口、CSV/Excel 处理、数据清洗与标准化、FuzzyWuzzy 模糊匹配（token_sort_ratio）用于 CM 与 Cantus ID 对齐、Volpiano 旋律编码转换、Python + Django + YAML 配置实现过滤器与历史追踪。

**📊 数据集**

使用的数据集为：①CantusCorpus v1.0（包含 chant 与 source 两张表）；②Corpus Monodicum 498 条 Proper Mass 对应 chant 的 MEI 数据，后经 ETL 生成 420 条可在 PyCantus 中使用的记录。

**📈 对比分析**

论文未给出具体性能指标，而是通过对比 CantusCorpus v1.0 与早期 v0.2 在“未见过的 chant”实验中的结果，表明大规模数据提升了实验的稳健性与可复制性；此外，PyCantus 的过滤与导出机制实现了实验配置的持久化和团队共享。

**⚠️ 局限性**

主要局限包括：①缺乏数据验证与控制词典，导致字段一致性不足；②对 Cantus ID 的依赖限制了与非 Cantus 数据源的直接整合；③可能存在重复或不完整的来源记录；④当前版本仍未支持多种旋律编码（如 GABC）和更细粒度的版本追踪。

---

## 123. Exhaustive Circuit Mapping of a Single-Cell Foundation Model Reveals Massive Redundancy, Heavy-Tailed Hub Architecture, and Layer-Dependent Differentiation Control

**arXiv ID:** 2603.11940 | [PDF](https://arxiv.org/pdf/2603.11940v1)

**作者:** Ihor Kendiukhov `[一作]` `[通讯]` (University of Tübingen), Ihor Kendiukhov (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对Geneformer单细胞基础模型进行全面的机制可解释性分析，完成了全量特征的电路追踪、三阶互作消融以及轨迹导向特征驱动实验；

**💡 创新点**

突破先前采样偏差、仅两阶交互、缺乏因果证据的局限，揭示了重度冗余的重心分布、注释偏差与层级决定细胞成熟方向的因果关系；

**🔧 技术方法**

采用稀疏自编码器（TopK SAE）、因果介导框架、Welford在线统计、激活拉伸与轨迹驱动等技术；

**📊 数据集**

使用K562细胞转录组、Tabula Sapiens免疫细胞数据，以及Geneformer V2‑316M预训练模型；

**📈 对比分析**

通过与先前仅抽样30特征的选择性追踪比较，结果显示全量追踪增加了27倍边缘数，三阶消融显示冗余比降低至0.59，层级驱动实验显示L17特征向成熟方向的正向驱动概率为1.0，早层为负/混合；

**⚠️ 局限性**

局限包括仅对20细胞做全量追踪降低统计功效、只测3个下游层、三阶消融样本量有限、轨迹驱动效应幅度微小、仅在Geneformer上验证，缺乏对其他基础模型的跨验证。

---

## 124. Deactivating Refusal Triggers: Understanding and Mitigating Overrefusal in Safety Alignment

**arXiv ID:** 2603.11388 | [PDF](https://arxiv.org/pdf/2603.11388v1)

**作者:** Zhiyu Xue `[一作]` (University of California), Ramtin Pedarsani `[通讯]` (University of California)

**通讯引用:** 3623 | [OpenAlex ID](https://openalex.org/A5040270189)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了安全对齐导致的过度拒绝问题，并提出基于拒绝触发器的缓解策略。

**💡 创新点**

首次将拒绝触发器作为机制解释，并利用其生成匹配分布的良性训练样本。

**🔧 技术方法**

使用了安全对齐的三种训练方法(SFT、P-SFT、RLVR)，并通过GPT‑4o提取拒绝触发器。

**📊 数据集**

使用Llama2、Llama3‑Uncensored、Qwen2.5‑Uncensored模型，训练集为Safety Data的有害查询和提取的触发器样本，评估基准包括SorryBench、JBench‑H、HEx‑PHI、Koala、GSM‑8K、SQL‑1K等。

**📈 对比分析**

与传统使用Alpaca等通用良性语料的做法相比，新方法在显著降低拒绝率的同时保持或略低的攻击成功率，整体Avg指标明显提升。

**⚠️ 局限性**

依赖外部LLM提取触发器可能引入噪声，评估主要基于规则检测，且触发器匹配的规模与多样性仍有待进一步优化。

---

## 125. Subtime: Reversible Information Exchange and the Emergence of Classical Time

**arXiv ID:** 2603.11571 | [PDF](https://arxiv.org/pdf/2603.11571v1)

**作者:** Paul L. Borrill `[一作]` `[通讯]` (DAE DAE LUS), Paul L. Borrill (DAE DAE LUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文正式化了subtime的概念——在纠缠系统中可逆的信息交换模式，并展示了经典时间如何通过去相干作为渐近极限出现。

**💡 创新点**

创新点在于引入了完美信息反馈（PIF）作为可逆性的实现，定义了可逆因果原则（RCP），并将多个物理理论统一在一个对称原则下。

**🔧 技术方法**

使用了光子钟模型和扩展的Oreshkov–Costa–Brukner框架，结合了时间反转对偶条件，形成了过程理论的正式化。

**📊 数据集**

未具体提及使用的数据集，但提到的实验可通过可逆数字链接和量子开关实验进行验证。

**📈 对比分析**

通过与现有的量子因果结构理论进行比较，展示了在去相干情况下，经典时间的出现与信息的保守性之间的关系，性能上强调了信息的可逆性和熵的量化。

**⚠️ 局限性**

限制在于该框架未能替代标准量子力学，而是提供了一种对时间结构的重新解释，且未解决时间箭头的根本问题。

---

## 126. Geometry-Aware Probabilistic Circuits via Voronoi Tessellations

**arXiv ID:** 2603.11946 | [PDF](https://arxiv.org/pdf/2603.11946v1)

**作者:** Sahil Sidheekh `[一作]` (University of Texas at Dallas), Sriraam Natarajan `[通讯]` (University of Texas at Dallas)

**通讯引用:** 2774 | [OpenAlex ID](https://openalex.org/A5064323671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在概率电路（PC）中引入基于Voronoi分割的几何感知路由，并提供可证实的近似推理框架与能够保持精确可推理的层次化分解结构。

**💡 创新点**

创新点在于：① 证明Voronoi路由与传统PC可推理的冲突；② 设计两种补救方案：一是基于轴对齐盒子与可调细分的可证实近似推理；二是提出层次化分解Voronoi（HFV）在保持几何性与可推理性之间的权衡；③ 用温度调度的软硬切换实现端到端学习。

**🔧 技术方法**

主要技术包括Voronoi分割、盒子逼近与细分、可证实上下界推理、层次化分解PC（HFV）、软硬门控（温度软化）、最大似然训练与温度退火。

**📊 数据集**

在八个低维合成数据集上评估：二维（Alphabet、CheckerBoard、Pinwheel、Spiral）和三维（BentLissajous、InterlockedCircles、Knotted、TwistedEight）。

**📈 对比分析**

与传统EinsumNet和HCLT基准进行比较。VT版本在所有数据集上得到比基准更高的测试对数似然（下界甚至超过基准），而HFV版本保持与基准相当的精度，且实现了精确推理；实验展示了VT的高表达能力与HFV的可推理优势。

**⚠️ 局限性**

局限性：① 在高维或复杂几何结构下，盒子逼近导致的近似误差可能过大；② HFV受限于分解结构，表达力低于全局Voronoi；③ 软硬门控的温度调度需要经验调节，可能导致训练不稳定。

---

## 127. The Mirror Design Pattern: Strict Data Geometry over Model Scale for Prompt Injection Detection

**arXiv ID:** 2603.11875 | [PDF](https://arxiv.org/pdf/2603.11875v1)

**作者:** J Alex Corll `[一作]` `[通讯]`, J Alex Corll

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Mirror 设计模式，通过严格的细胞匹配和源可追溯的几何化数据组织，训练稀疏字符 n-gram SVM，构建 L1 级 prompt injection 筛查器，并证明其在召回率和延迟上优于现有的 Prompt Guard 2 等语义模型。

**💡 创新点**

创新点在于：1) 用 Mirror 细胞将恶意与正样本在语言、格式、长度、理由等维度完全对齐，消除模型学习到的表面短路；2) 将 SVM 编译为静态 Rust 哈希表，实现在毫秒级无依赖部署；3) 通过内容哈希与源追踪实现严格可审计的数据契约，证明数据几何比模型规模更关键。

**🔧 技术方法**

使用技术包括：字符级 3-5 维 n-gram 特征、线性 SVM（C=1.0）、特征稀疏化（约 15,000 维）、完美哈希映射编译、Rust 静态二进制、内容哈希+源追踪流水线、对比 Prompt Guard 2（22M）与正则规则层。

**📊 数据集**

数据集：5,000 条公开源样本，覆盖 8 类攻击理由 × 4 语言的 32 细胞；holdout 524 条（248 恶意、276 正常）用于模型评估；另外使用 2,386 条硬性 benign 挑战集和 JailbreakBench 语义改写测试。

**📈 对比分析**

比较方法：在同一 524 条 holdout 上，固定阈值 t=0。Mirror L1 SVM 取得 F1=0.921、召回=0.960、精确=0.885，延迟<1 ms；Prompt Guard 2（22M）F1=0.591、召回=0.444、精确=0.887，延迟≈49 ms；正则层精确 0.992、召回 0.141。Mirror 在召回率和延迟上显著优于语义模型，且精确率保持可接受水平。

**⚠️ 局限性**

限制：1) 仍存在 use‑versus‑mention、硬性 benign 高 FPR（阈值+2 下 12%）等残差；2) 对高度改写的攻击和部分多语言场景召回仍约 21%；3) 仅针对公开源数据，规模有限；4) 线性 n‑gram 仍无法解决上下文歧义，无法完全替代 L2a 语义层。

---

## 128. STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure Transformer for Offline Multi-task Multi-agent Reinforcement Learning

**arXiv ID:** 2603.11691 | [PDF](https://arxiv.org/pdf/2603.11691v1)

**作者:** Jiwon Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Youngchul Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2545 | [OpenAlex ID](https://openalex.org/A5020240958)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 STAIRS-Former，一种在离线多任务多智能体强化学习中通过空间-时间层级注意力和随机 token dropout 进行高效建模的 transformer 结构；

**💡 创新点**

核心创新包括：①递归空间层级以加强实体间关系推理；②双时钟时间层级（短期与长期记忆）以捕获长期依赖；③token dropout 以提升对不同智能体数的泛化能力；

**🔧 技术方法**

采用 transformer（多头自注意力、FFN）、GRU、Qatten 混合网络以及 TD3+BC 风格的离线学习损失；

**📊 数据集**

在 SMAC、SMAC-v2、MPE、MaMuJoCo 的多任务环境上进行评估，使用四种数据质量（Expert、Medium、Medium-Expert、Medium-Replay）；

**📈 对比分析**

相较于 UPDeT-m、ODIS、HiSSD 等基线，STAIRS-Former 在训练集与测试集的平均胜率上提升约 10–50%（例如 SMAC-Hard 上从 57.0% 提升至 67.4%），实现了新的 state‑of‑the‑art；

**⚠️ 局限性**

主要限制仍在于对极端稀疏奖励或更大规模智能体群的适应性尚未彻底验证，且模型复杂度较高，可能导致训练成本增加。

---

## 129. EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering

**arXiv ID:** 2603.11703 | [PDF](https://arxiv.org/pdf/2603.11703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. UNet-AF: An alias-free UNet for image restoration

**arXiv ID:** 2603.11323 | [PDF](https://arxiv.org/pdf/2603.11323v1)

**作者:** Jérémy Scanvic `[一作]` (Laboratoire de Physique de l'ENS de Lyon), Julián Tachella `[通讯]` (Laboratoire de Physique de l'ENS de Lyon)

**通讯引用:** 859 | [OpenAlex ID](https://openalex.org/A5083772205)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并实现了一种无别名化的 UNet 架构（UNet‑AF），通过在传统 UNet 结构中替换多种层以实现对连续平移的严格等变性，并在图像恢复任务中进行验证。

**💡 创新点**

创新点在于：① 采用 FFT 实现的 sinc 频域平滑卷积代替最大池化；② 在上采样、激活函数与归一化层均使用专门的 anti‑aliasing 设计；③ 结合循环卷积、可选残差连接与圆形填充，实现了对完整平移的等变性；④ 通过全方位消除别名化显著提升网络在恢复任务中的鲁棒性与性能。

**🔧 技术方法**

使用的关键技术包括：傅里叶变换实现的 sinc 低通滤波器（BlurPool 与 Filtered Upconv）；圆形卷积（Circular Pad）；Filtered GELU 激活（细分上采样后再下采样消除高频噪声）；Alias‑free Layer Normalization；以及可选的残差连接。所有这些层均基于现有研究（如 chaman21、karras21、michaeli23 等）的最新改进。

**📊 数据集**

实验所用数据集为 DIV2K 的 900 张自然图像（700 训练 / 100 验证 / 100 测试），经统一裁剪、缩放并加入高斯模糊+噪声（或仅噪声）生成低质量输入，用于三类恢复任务：循环模糊、有效模糊、以及高斯噪声去除。

**📈 对比分析**

与传统 UNet（Ronneberger）和改进版 UNet（Jin）基线相比，UNet‑AF 在 PSNR、SSIM、LPIPS 等视觉指标上保持相当或略优，且在自定义的等变性指标 EQUIV 上提升约 40–70 dB，训练过程更稳定，验证曲线更平滑。仅在推理速度上，UNet‑AF 由于额外的滤波与归一化开销，约慢 5–7 倍。

**⚠️ 局限性**

主要局限包括：① 计算成本显著增加，导致推理速度下降；② 对某些激活函数的过滤方式（Filtered GELU）在极端深度网络中仍可能略降性能；③ 目前的 anti‑aliasing 滤波器未针对不同层进行专门优化，未来可通过定制 GPU 核心或更高效的滤波实现进一步加速。

---

## 131. LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation

**arXiv ID:** 2603.12152 | [PDF](https://arxiv.org/pdf/2603.12152v1)

**作者:** Feiyu Duan `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5242 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LifeSim用户模拟器和LifeSim‑Eval评测基准，用于长时序个性化助手交互的模拟与评估。

**💡 创新点**

创新点包括：①将Belief‑Desire‑Intention（BDI）心理模型与事件引擎、行为引擎结合，实现对用户内部认知与外部物理环境的统一建模；②构建多场景、多意图（显式与隐式）交互的长时序评测框架；③使用百万级用户池与真实移动轨迹，提升仿真真实性；④采用LLM‑as‑Judge与人工双重评测，保证评测可靠性。

**🔧 技术方法**

技术：BDI认知引擎、事件引擎、用户行为引擎（记忆感知、情感推理、动作选择）、LLM生成与判别；评测采用多维度指标（意图识别/完成、自然度、连贯性、偏好恢复、人格一致性）；实验平台使用vLLM、NVIDIA RTX 4090，采样温度固定为1.0。

**📊 数据集**

数据集：SocioVerse（Twitter）提供人口统计属性；AlignX提供人格与偏好维度；3,374条用户移动轨迹（251个兴趣点）；LifeSim‑Eval共120个用户、1,200个场景，覆盖8个生活领域。

**📈 对比分析**

评测方法：在单场景和长时序两种设置下，使用多种公开与闭源LLM（如GPT‑5、GPT‑4o、Claude‑Sonnet、DeepSeek‑V3.2、Qwen3、Gemma3、Llama3.1、gpt‑oss）对意图识别、完成、自然度、连贯性、偏好恢复、人格一致性进行量化；实验结果显示：闭源模型在隐式意图识别与完成上明显优于开源模型，规模增大往往带来提升；但无论模型大小，隐式意图与长时序偏好建模仍显弱势。

**⚠️ 局限性**

局限性：①仅覆盖日常生活场景，缺乏高风险领域（医疗、法律、金融）评估；②当前仅使用文本交互，未加入多模态用户信号（视觉、生理等）。

---

## 132. Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information

**arXiv ID:** 2603.11749 | [PDF](https://arxiv.org/pdf/2603.11749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 133. EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation

**arXiv ID:** 2603.12108 | [PDF](https://arxiv.org/pdf/2603.12108v1)

**作者:** Yan Li `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5859 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一图像分词器，通过残差演化轨迹将像素特征逐步转化为语义特征，实现视觉理解与生成的同一框架。

**💡 创新点**

将像素级特征与语义级特征在共享潜在空间内逐层演化，既保持任务解耦又保持内在一致性，解决传统耦合与过度解耦的矛盾。

**🔧 技术方法**

级联残差向量量化、语义对齐损失、像素重建与感知损失、对抗训练、SigLIP2 对齐投影、LLM（Qwen‑2.5‑7B）等技术。

**📊 数据集**

使用 CC12M、ImageNet‑1K、LLaVA‑Pretrain‑558k、LLaVA‑OneVision、LLaVA‑NeXT‑780k、Cambrian‑10M、HoneyBee、ImageNet1K‑QwenImage、BLIP3o‑Pretrain‑Short 等公开数据，共约 13‑22M 张图像。

**📈 对比分析**

与多种统一分词器（UniTok、TokenFlow、DualToken 等）及生成专用模型（Stable Diffusion、DALL‑E3 等）进行 rFID、SEEDBench、MMVU、MME、GenAI‑Bench、GenEval 等多维度评估，rFID 达 0.43，生成与理解均多项指标名列前茅。

**⚠️ 局限性**

仍需提升高分辨率生成性能、处理更复杂多模态任务的能力；残差演化层数与代码本大小的选择对性能有影响；虽然训练规模小于对手，但在更大数据集上的泛化尚待验证。

---

## 134. GeNeX: Genetic Network eXperts framework for addressing Validation Overfitting

**arXiv ID:** 2603.11056 | [PDF](https://arxiv.org/pdf/2603.11056v1)

**作者:** Emmanuel Pintelas `[一作]` (University of Piraeus), Ioannis E. Livieris `[通讯]` (University of Peloponnese)

**通讯引用:** 3164 | [OpenAlex ID](https://openalex.org/A5074255083)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 GeNeX 的两阶段集成框架，旨在消除验证集过拟合（VO）风险，提升模型在低数据和分布偏移场景下的泛化性能。

**💡 创新点**

创新点在于：①在模型生成阶段使用梯度训练与遗传算法相结合的“双路径”策略，通过权重交叉和微调生成多样化的子代网络，避免对验证指标的依赖；②在集成构造阶段采用行为聚类与多专家原型融合，利用权重级联与SQP优化实现可解释且高鲁棒性的最终模型。

**🔧 技术方法**

技术包括：梯度下降训练、遗传交叉与变异、预测行为聚类（UMAP+GMM）、多专家选择（Top‑Val、鲁棒性、代表性、异常性）、原型网络融合（权重平均）、序列二次规划（SQP）优化以及基于Jensen‑Shannon Divergence 的 VO‑aware 数据划分。

**📊 数据集**

使用四个真实世界二分类任务的数据集：Skin Cancer、DeepFake Detection、Plant Disease 及 Pneumonia（Chest X‑ray），并在每个任务上构造 30%/70% 的训练/测试划分，最大化 JSD 以模拟分布偏移。

**📈 对比分析**

与 15 种现有集成方法（如单模型、增量集成、遗传进化、CBDMoE、PRL、HeX、RSEN、HWA 等）进行比较；在低 JSD（随机划分）场景下性能相近，而在高 JSD（VO‑aware）场景下 GeNeX 以更小的 VO 间隙和更高的测试 GM 指标领先，显示出显著的鲁棒性。

**⚠️ 局限性**

局限性主要在于：仅在图像分类任务上验证，尚未在分割、文本等其他任务中测试；需要进一步探索更高效的遗传搜索策略和更大规模的模型池；对资源消耗（尤其是多代遗传演化阶段）仍有改进空间。

---

## 135. Enhancing Value Alignment of LLMs with Multi-agent system and Combinatorial Fusion

**arXiv ID:** 2603.11126 | [PDF](https://arxiv.org/pdf/2603.11126v1)

**作者:** Yuanhong Wu `[一作]` (Fordham University), D. Frank Hsu `[通讯]` (Fordham University)

**通讯引用:** 10321 | [OpenAlex ID](https://openalex.org/A5082344124)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于多模道德代理的价值对齐系统 VAS-CFA，融合不同伦理视角的生成结果

**💡 创新点**

创新点在于将多模代理的输出通过组合融合分析（CFA）结合排名与分数两种聚合方式，利用认知多样性提升对齐质量

**🔧 技术方法**

采用 DPO+QLoRA 微调多模代理，GPT-4.1 拆分道德单位，SentenceTransformer+逻辑回归构建道德分类器，并使用 CFA 进行组合与加权聚合

**📊 数据集**

使用 Moral Integrity Corpus (MIC) 进行代理微调，并在 113.8K 的测试集上评估

**📈 对比分析**

与单代理、原始聚合、CVA-GS 等方法在 ROUGE‑L 与 BERTScore 上对比，VAS‑CFA 的 ARC/WRCDS 组合在 F1 上提升约 8‑10% 以上

**⚠️ 局限性**

局限在于仅覆盖五种道德维度，聚合过程仍需人工选择最佳单元，且模型规模与计算成本较高

---

## 136. Compiling Temporal Numeric Planning into Discrete PDDL+: Extended Version

**arXiv ID:** 2603.12188 | [PDF](https://arxiv.org/pdf/2603.12188v1)

**作者:** Andrea Micheli `[一作]` (Fondazione Bruno Kessler), Alessandro Valentini `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 653 | [OpenAlex ID](https://openalex.org/A5061585655)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将 PDDL 2.1 级别 3 的带持续时间动作的时序规划问题编译为 PDDL+ 问题，使用进程、事件和锁机制完全保持语义，并提供多项式规模的编译。

**💡 创新点**

首次给出完整的形式化编译并证明其音响与完备性；引入锁预条件/后效的创新机制，用于在 PDDL+ 中实现“无自重叠”与“非干扰”约束；使得 PDDL+ 能自然表达持续时间动作和数值约束。

**🔧 技术方法**

使用 PDDL+ 进程与事件的时间模型、锁变量、计时器、增量与赋值效果、离散时间步长以及 ENHSP 规划器的 h^mrp 评价函数。

**📊 数据集**

实验数据集包括经典 Matchcellar、MAJSP 以及新提出的 T‑Sailing、T‑Plant‑Watering 四个时序数值域；每个域 20 个实例。

**📈 对比分析**

对比 ARIES、NextFLAP、TFLAP、OPTIC、TAMER、Patty 等最先进的时序规划器。ENHSP-WA 取得最高覆盖率，尤其在时序数值域上明显优于对手；在纯时序域上 ARIES 更快，但两者互补；总体表现表明编译方案在复杂数值时序问题上具有竞争力。

**⚠️ 局限性**

局限性包括仅支持 PDDL 2.1 级别 3（不含连续变化），使用离散时间语义导致在连续时间下出现 Zeno 行为；编译后模型尺寸仍相对较大，锁机制对大规模问题可能产生开销；未对连续时间语义下的 PDDL+ 进行实验。

---

## 137. Language Generation with Replay: A Learning-Theoretic View of Model Collapse

**arXiv ID:** 2603.11784 | [PDF](https://arxiv.org/pdf/2603.11784v1)

**作者:** Giorgio Racca `[一作]` (University of Copenhagen), Amartya Sanyal `[通讯]` (University of Copenhagen)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5035879433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从学习理论的角度研究了在语言生成过程中加入“重放”机制（即训练时使用自身先前输出的文本）对模型可生成性的影响，并给出了针对不同生成定义（统一生成、非统一生成、极限生成、正规生成）的可生成性与不可生成性的精细划分与证明。

**💡 创新点**

创新点在于：①首次将重放对生成任务的影响形式化为“重放对手”模型，并在此框架下构造了理论上的可生成性分层与分离；②提出了在重放环境下仍能保持最优生成的“Witness Protection”算法；③给出了重放对正规生成导致不可生成性的最小化例子；④对传统在线学习中重放导致的可学习性分离与生成问题进行对比。

**🔧 技术方法**

主要技术包括：语言生成在极限框架、重放对手建模、组合维度分析、可生成性证明中的消除重放误导、Witness Protection（见 WP 算法）以及构造性硬例的递归枚举与对抗序列。

**📊 数据集**

该工作纯粹为理论分析，没有使用实际数据集，所有结论均在可数或不可数假设类的抽象设定上给出。

**📈 对比分析**

由于是理论论文，未进行实验对比；但通过构造性证明与示例展示，作者说明了在不同生成定义下重放能或不能影响可生成性，且在重放不影响的场景下，所给算法可在样本复杂度上与经典生成相匹配。

**⚠️ 局限性**

局限性包括：①假设生成器为确定性、无限算力；②仅在满足 UUS（无穷支撑）假设的假设类上成立；③未考虑噪声或不完备的重放信息；④缺乏对实际 LLM 训练与部署过程中的实验验证；⑤对正规生成的不可生成性结果只给出了最小化的四元组例子，可能不具普适性。

---

## 138. On the Robustness of Langevin Dynamics to Score Function Error

**arXiv ID:** 2603.11319 | [PDF](https://arxiv.org/pdf/2603.11319v1)

**作者:** Daniel Yiming Cao `[一作]` (Cornell University), Yuchen Wu `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了在高维情形下，使用估计的得分函数（score）进行 Langevin 动力学采样时的鲁棒性问题，并给出了一系列负向结果，证明即使 L^p（特别是 L^2）得分误差可以指数级小，Langevin 动力学仍会在多项式时间尺度内产生与目标分布相差巨大的采样结果。

**💡 创新点**

创新点在于：①提出并证明了针对简单的高维正态目标分布，在任何多项式时间尺度内，Langevin 动力学即使使用了极其精确（指数级小误差）的得分估计，也会在总变差距离上远离目标分布；②展示了数据驱动初始化（用训练样本作为初始点）同样会导致严重偏差，强调必须使用未在得分训练中出现的“新鲜”样本；③将上述负结果推广到广义目标分布（满足 Lipschitz、L^2 可积等条件），并给出构造性的得分估计使采样误差在无穷大时间下接近 1；④通过数值仿真验证了理论结论。

**🔧 技术方法**

主要技术包括：构造带有“记忆”性质的得分估计函数（piecewise Lipschitz、依赖于样本或常数）；利用高维高斯分布的径向集中性质证明得分误差指数级小；分析 Langevin 动力学的逃逸时间与 OU 过程比较，得到逃逸时间指数级大；使用 Girsanov 定理、Pinsker 不等式等概率工具来估计总变差和 KL 散度；以及对更一般目标分布使用锥形概率论与 Zvonkin 定理证明收敛不稳定性。

**📊 数据集**

在实验部分主要使用两类分布：1）单个高斯分布 N(0,2I_d)（d=50）；2）混合高斯模型 0.5N(-1,2I_d)+0.5N(4,2I_d)（d=25）。通过这些合成数据验证理论。

**📈 对比分析**

比较方法包括：①三种初始化策略（随机正态、随机新鲜样本、训练样本），②使用 Langevin 动力学采样后与目标分布的 KL 散度和 Wasserstein 距离进行评估。结果显示：使用训练样本初始化的 Langevin 动力学（算法3）往往产生更差的采样质量，而使用新鲜样本或随机正态初始化（算法1、2）性能相近且优于算法3；在混合高斯情形下差距相对较小。

**⚠️ 局限性**

局限性包括：①理论仅覆盖 Langevin 动力学，未涉及其他基于得分的采样算法如 Diffusion Models 的鲁棒性；②构造的得分估计多为理论性的“记忆”示例，实际学习过程的得分误差分布可能更复杂；③实验仅在合成数据上验证，缺乏对真实高维数据集的实证；④对离散化步骤的影响分析有限，主要以理想连续时间过程为依据。

---

## 139. OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams

**arXiv ID:** 2603.12265 | [PDF](https://arxiv.org/pdf/2603.12265v1)

**作者:** Yibin Yan `[一作]` (School of Artificial Intelligence), Weidi Xie `[通讯]` (School of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的流式视觉主干模型，可在不对主干进行微调的情况下，支持从图像、视频到几何重建、语言理解和机器人控制等多种任务；

**💡 创新点**

创新点包括：① 结合因果时空注意力和 3D-RoPE，使预训练的图像 ViT 成为严格因果的流式模型；② 在单一主干上同时训练自监督蒸馏、流式几何重建和视觉‑语言对齐的三大任务，形成协同的多任务预训练框架；③ 通过 KV‑cache 让模型能够在长时间序列中以线性时间/内存高效推理；

**🔧 技术方法**

使用技术包括：Vision Transformer (DINOv3 预训练)、因果时空注意力、3D Rotary Position Embedding、KV‑cache、双向学生-教师蒸馏、iBOT、KoLeo 正则、Gram 对齐、深度与相机头、轻量化语言解码器 (Qwen3-0.6B) 等；

**📊 数据集**

训练数据涵盖 29 份数据集，约 200M 帧，包含静态图像（ImageNet、ADE20K、NYUv2 等）、视频（Kinetics‑400、Something‑Something V2、DAVIS'17 等）以及几何重建数据（Sintel、KITTI、ScanNet 等）；

**📈 对比分析**

在 4 大评测维度上进行对比：① 图像/视频探测（ImageNet、Kinetics、Something‑Something、DAVIS）— 与 DINOv3、V‑JEPA2 等基线相比，性能相当或略优；② 流式几何重建（Sintel、KITTI、ScanNet）— 以接近或超过专用在线 3D 模型的精度；③ 视觉‑语言模型（LLaVA‑Video、V‑MME、VSI‑Bench）— 在大多数基准上均优于 LLaVA‑Video，甚至在 VSI‑Bench 上刷新 SOTA；④ 视觉‑语言‑动作（CALVIN、Sim‑BridgeV2）— 只冻结主干即可在机器人控制任务上达到或超过专用基线，展示零样本迁移能力；

**⚠️ 局限性**

局限性：① 在部分细粒度视觉或极端长序列任务上仍不及专门设计的基线；② 由于采用单一 ViT 结构，模型规模和参数量相对较大，部署成本仍有提升空间；③ 预训练依赖大量多样化数据，对数据获取成本敏感；④ 对极端低延迟/内存环境的极限压缩效果尚待进一步验证。

---

## 140. SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation

**arXiv ID:** 2603.12238 | [PDF](https://arxiv.org/pdf/2603.12238v1)

**作者:** Jun Luo `[一作]` (Peking University), Gang Zeng `[通讯]` (Peking University)

**通讯引用:** 5298 | [OpenAlex ID](https://openalex.org/A5103205993)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 SceneAssistant，一个基于视觉反馈闭环的开词汇 3D 场景生成代理，能从自然语言描述自动构建、编辑多样化 3D 场景。

**💡 创新点**

创新点在于提供完整的 Action API 组，让 VLM 在视觉反馈下直接操纵 3D 对象，避免预定义空间关系并实现闭环自我修正。

**🔧 技术方法**

采用 Gemini‑3.0‑Flash 作为 VLM 主干，结合 Hunyuan3D 与 Z‑Image 的多阶段 3D 资产生成 pipeline，以及 Blender 进行渲染。

**📊 数据集**

使用 Objaverse 数据集的 3D 资产与自制的图像生成数据，且无专门的开词汇场景数据集。

**📈 对比分析**

与 Holodeck、SceneWeaver 以及自设的 NoActionAPI/NoVisFeedback 基线比较，SceneAssistant 在空间布局正确率、对象质量与人类偏好上分别提升至 6.9/6.95/61.25%，显著优于对手。

**⚠️ 局限性**

局限在于对 3D 资产生成模型的依赖、计算资源消耗高以及缺乏对复杂物理交互的完整建模。

---

## 141. Resonate: Reinforcing Text-to-Audio Generation via Online Feedback from Large Audio Language Models

**arXiv ID:** 2603.11661 | [PDF](https://arxiv.org/pdf/2603.11661v1)

**作者:** Xiquan Li `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 99278 | [OpenAlex ID](https://openalex.org/A5100434325)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将在线强化学习算法GRPO与流匹配音频生成模型结合，构建并训练Resonate文本到音频生成器，获得TTA-Bench上语义对齐与音频质量的SOTA表现。

**💡 创新点**

首次在文本到音频领域使用在线强化学习（GRPO），利用大型音频语言模型（LALM）作为细粒度奖励，并将流匹配过程改造为SDE采样以实现随机探索。

**🔧 技术方法**

技术核心包括Flow-GRPO（在线GRPO + SDE采样）、流匹配（Conditional Flow Matching）、LALM奖励（Qwen2.5-Omni通过AQA评分）、CLAP评估、AudioBox-Aesthetics四大指标评测。

**📊 数据集**

预训练使用约370万对音频‑文本对（来源：AudioCaps、AudioSet、Clotho、VGGSound、WavCaps、AudioStock），评估集为TTA‑Bench Accuracy子集（1500条多事件复杂提示）。

**📈 对比分析**

与现有SOTA模型（MeanAudio、TangoFlux等）对比，Resonate‑GRPO在AQAScore 0.737、CLAP 0.476、PQ 6.064等指标上均超越对手；主观评测中，整体质量OVL 3.86、相关性REL 3.83为最高。

**⚠️ 局限性**

局限性包括：奖励函数仍受LALM与CLAP模型的瓶颈限制，可能出现奖励劫持；评估主要基于TTA‑Bench，未覆盖更大规模或更复杂的时间逻辑场景；模型参数量虽相对紧凑但仍需较高算力；对低质量数据的过拟合风险需要进一步缓解。

---

## 142. Automating Skill Acquisition through Large-Scale Mining of Open-Source Agentic Repositories: A Framework for Multi-Agent Procedural Knowledge Extraction

**arXiv ID:** 2603.11808 | [PDF](https://arxiv.org/pdf/2603.11808v1)

**作者:** Shuzhen Bi `[一作]` (East China Normal University), Aimin Zhou `[通讯]` (East China Normal University)

**通讯引用:** 9833 | [OpenAlex ID](https://openalex.org/A5050248676)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套从GitHub开源仓库自动抽取高质量代理技能（skills）的完整流程，并将抽取的技能转换为标准化的SKILL.md格式；

**💡 创新点**

创新点在于将仓库结构分析、语义检索与密集向量匹配相结合，形成两阶段检索+交叉编码器的技能识别机制，并建立四阶段安全验证管道；

**🔧 技术方法**

使用的技术包括仓库结构映射工具（如repo2AI）、双编码器密集检索、交叉编码器精细排序、自然语言处理生成前置元数据与操作指令、静态/行为分析安全检测以及多维度评估框架；

**📊 数据集**

数据集主要为公开的代理项目（TheoremExplainAgent与Code2Video）及其内部的TheoremExplainBench（240条定理）和Code2Video的教育视频生成数据；

**📈 对比分析**

通过对比基线代码生成模型，使用TeachQuiz指标评估知识迁移效率，实验显示代理生成的视频相较于基线模型提升了40%；在安全与可执行性上使用四阶段验证，结果显示大约26.1%的公开技能存在安全隐患；

**⚠️ 局限性**

局限性包括：抽取过程依赖于项目代码的可读性与文档完整性；安全验证仍需人工审计；抽取的技能可能在跨模型兼容性与版本演进上存在难度；以及大规模技能库的组织与冲突解决仍是后续挑战。

---

## 143. Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding

**arXiv ID:** 2603.11423 | [PDF](https://arxiv.org/pdf/2603.11423v1)

**作者:** Songlin Li `[一作]` (University of Electronic Science and Technology of China), Jian Yao `[通讯]` (XPeng Motors)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了可靠多样本蒸馏框架（R-MSD），通过为每个输入采集多份教师答案并进行任务自适应匹配，实现对大型视觉‑语言模型（LVLM）在视频理解任务中的高效蒸馏。

**💡 创新点**

其创新点在于将教师采样方差显式建模，采用闭式任务基于地面真值的质量评分与开放式任务统一抽样的任务自适应匹配，以及在对抗蒸馏阶段引入在线判别器和RL奖励，显著提升监督质量而非仅仅增加样本量。

**🔧 技术方法**

技术手段包括：多样本教师采样、质量感知加权匹配、基于判别器的对抗奖励、两阶段训练（先SFT warmup后RL‑based adversarial distillation）、以及对格式与内容的分离式奖励设计。

**📊 数据集**

训练数据来源于VideoR1（10%子集）和Open‑O3完整集；评估使用公开视频QA基准VideoMME、Video‑MMM‑U、WorldSense、LongVideoBench、MLVU_MCQ、VsTAR以及图像QA基准MathVista、MathVerse。

**📈 对比分析**

在与单样本蒸馏、SFT+RL基线以及现有同规模模型（如Qwen3‑VL‑4B）进行匹配预算对照的实验中，4B学生模型在VideoMME提升+1.5%、Video‑MMM‑U提升+3.2%、MathVerse提升+3.6%，并在VsTAR等多模态基准上表现优于对照，证明R‑MSD在多样本监督与对抗策略结合下取得了显著性能提升。

**⚠️ 局限性**

局限性包括：闭式任务的质量评分依赖精确的地面真值标注，开放式任务采用统一抽样无法显式评估语义正确性；多样本策略增加了训练时的算力开销；在弱监督或教师与学生容量差距较大时，方法的有效性尚待进一步验证。

---

## 144. Unveiling Practical Shortcomings of Patch Overfitting Detection Techniques

**arXiv ID:** 2603.11262 | [PDF](https://arxiv.org/pdf/2603.11262v1)

**作者:** David Williams `[一作]` (University College London), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 4545 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种补丁过拟合检测技术在实际场景中的效果进行了系统评估，构建了贴合真实 APR 输出的补丁数据集，并将其与随机采样基线进行比较。

**💡 创新点**

首次在真实 APR 生成的补丁数据集上进行大规模基准测试，揭示现有方法在实践中往往不如简单随机采样，并提出了新的基准评估框架。

**🔧 技术方法**

使用了静态分析、动态测试、机器学习三类过拟合检测技术，并引入两种随机采样基线作为对照。

**📊 数据集**

构造了多个基于真实 APR 工具产生的补丁数据集（覆盖多种工具和 Defects4J 等公开数据集），保证补丁产生条件一致。

**📈 对比分析**

通过与两种随机基线对照，评估检测准确率/覆盖率等指标；结果显示，简单随机方法在 71%~96% 的情况下优于所有先进方法，表明现有技术实用价值有限。

**⚠️ 局限性**

研究仅覆盖所选工具和数据集，可能不适用于所有类型的补丁或 Bug；随机基线在实际工程中可能不可行；此外，实验规模和成本未做充分评估，需进一步验证。

---

## 145. R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection

**arXiv ID:** 2603.11566 | [PDF](https://arxiv.org/pdf/2603.11566v1)

**作者:** Zhongyu Xia `[一作]` (Peking University), Weijun Qin `[通讯]` (EBTech Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种4D雷达-相机融合的3D目标检测框架，通过逐步净化BEV特征实现高精度定位

**💡 创新点**

核心创新在于三大模块：Panoramic Depth Fusion提升全景深度估计；Deformable Gated Temporal Fusion实现无姿态的时空对齐；Instance-Guided Dynamic Refinement利用2D实例语义对BEV特征进行动态校正

**🔧 技术方法**

采用邻域跨注意力聚合、概率深度监督、基于变形卷积的对齐、GRU式门控更新以及实例原型生成的仿射校正等多种深度学习技术

**📊 数据集**

在TJ4DRadSet和View-of-Delft（VoD）两个大型4D雷达-相机数据集上进行评估

**📈 对比分析**

与现有方法（如SGDet3D、CVFusion、RCFusion等）对比，本文在TJ4DRadSet上3D mAP提升至63.60%、在VoD上AP_EAA提升至93.96%，同时保持较高的实时帧率

**⚠️ 局限性**

依赖于高质量的2D实例提议，若实例检测失效会影响IGDR校正；对极端天气或极稀疏雷达点云的鲁棒性仍待进一步验证

---

## 146. Seeing Isn't Orienting: A Cognitively Grounded Benchmark Reveals Systematic Orientation Failures in MLLMs Supplementary

**arXiv ID:** 2603.11410 | [PDF](https://arxiv.org/pdf/2603.11410v1)

**作者:** Nazia Tasnim `[一作]` (Boston University), Bryan A. Plummer `[通讯]` (Boston University)

**通讯引用:** 3127 | [OpenAlex ID](https://openalex.org/A5061227594)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了一个认知驱动的方向感知基准 DORI，用以单独评估多模态大型语言模型（MLLM）在物体方向识别与几何推理方面的能力。

**💡 创新点**

创新点在于：① 将方向感知分解为四个从粗到细、由人类认知发展阶段衍生的维度（正面对齐、旋转变换、相对方向、自然方向）；② 采用粗细两级（分类与度量）评测体系，精准区分模型是缺乏基本方向感知还是缺乏连续几何推理；③ 通过结构化提示设计和多源数据集，消除了传统基准中对场景、语义的混淆。

**🔧 技术方法**

主要技术包括：多模态视觉语言模型评估、结构化提示（任务描述、范例、逐步推理）、基于 LoRA 的微调、Token‑based Fusion 对比、以及在不同维度与粒度上计算准确率。

**📊 数据集**

使用 14 个公开数据集（包括 13,652 张自然与合成图像），覆盖 67 个物体类别，生成 33,656 道多选问题（每题均含“无法确定”选项）。

**📈 对比分析**

对 24 个开源与闭源 MLLM 进行评测，结果显示：在粗粒度任务上大多数模型略高于随机，平均约 55% ；但在细粒度（角度/度量）任务上性能跌至 20% 左右，且相对方向与旋转变换是最难的子任务。微调后模型提升显著，且对外部空间推理基准有 27% 的迁移增益。

**⚠️ 局限性**

局限性包括：① 仍未解决 Canonical Orientation 维度，提示与模型结构无法弥补几何推理缺口；② 依赖结构化提示，模型对提示敏感；③ 评测主要聚焦单一物体或两图对比，缺乏更大尺度的多视角/多物体情境；④ 目前的基准仍未覆盖极端遮挡、光照变化等真实世界噪声。

---

## 147. SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**arXiv ID:** 2603.11616 | [PDF](https://arxiv.org/pdf/2603.11616v1)

**作者:** Muyi Sun `[一作]`, Tianzheng Deng `[通讯]` (AFMC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了SemiTooth，一种多源CBCT牙齿分割的半监督框架

**💡 创新点**

创新点在于多教师多学生协作架构和Stricter Weighted‑Confidence约束，提升多源伪标签质量和跨源泛化

**🔧 技术方法**

采用半监督学习、教师‑学生EMA、V‑Net骨干、区域级加权一致性损失（SWC）等技术

**📊 数据集**

使用自建的MS³Toothset数据集，包括98个标注样本（20个测试）和438个未标注样本，来源于上海科技、PKU‑SS和AFMC

**📈 对比分析**

与多种单源和多源SOTA方法（V‑Net、MT、UA‑MT、ASDA、MLRPL、CMT、Uni‑HSSL）对比，SemiTooth在mIoU、Dice、召回率和像素准确率上分别为76.67、85.69、88.66、86.44，取得最佳性能

**⚠️ 局限性**

局限性包括仅在三源数据上验证，未评估对更大范围外部数据的泛化；模型对硬件要求较高，且对不同设备的鲁棒性仍有待进一步验证

---

## 148. Relaxed Efficient Acquisition of Context and Temporal Features

**arXiv ID:** 2603.11370 | [PDF](https://arxiv.org/pdf/2603.11370v1)

**作者:** Yunni Qu `[一作]` (University of North Carolina), Junier Oliva `[通讯]` (University of North Carolina)

**通讯引用:** 2527 | [OpenAlex ID](https://openalex.org/A5102824470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种端到端可微的两阶段主动特征获取框架REACT，联合优化入职阶段的上下文特征选择和随时间的特征采集计划；

**💡 创新点**

首次将入职上下文与随访测量明确分离，使用Gumbel‑Sigmoid松弛实现离散采集决策的梯度传播，并通过自迭代训练解决传统RL难以优化的问题；

**🔧 技术方法**

Gumbel‑Sigmoid/straight‑through 采样松弛、全连接规划器、3层MLP预测器、滑动窗口的自迭代训练；

**📊 数据集**

五个真实纵向数据集：行为学 CHEEARS、ILIADD；临床学 OAI（KLG、WOMAC）、ADNI（痴呆三分类）；

**📈 对比分析**

与ASAC、RAS、DIME等RL/非RL基线对比；在各数据集上在相同成本预算下，REACT 在 AUROC/AUPRC 上均优于基线，且成本更低；

**⚠️ 局限性**

对入职上下文的选择依赖于预设成本权重，模型对不同预算敏感；自迭代训练需要多轮滚动采样，训练成本相对较高；

---

## 149. When Slots Compete: Slot Merging in Object-Centric Learning

**arXiv ID:** 2603.11246 | [PDF](https://arxiv.org/pdf/2603.11246v1)

**作者:** Christos Chatzisavvas `[一作]` (Democritus University of Thrace), Nikolaos Mitianoudis `[通讯]` (Democritus University of Thrace)

**通讯引用:** 1691 | [OpenAlex ID](https://openalex.org/A5064706177)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可微分的槽合并（slot merging）操作，用 Soft‑IoU 衡量槽间的空间重叠，并在 Slot Attention 训练过程中动态合并重叠槽，从而得到更连贯的对象级表示。

**💡 创新点**

创新点在于：①使用概率化 IoU（Soft‑IoU）量化槽间重叠；②引入固定阈值的合并策略并利用数据分布自适应确定阈值；③设计了保持梯度流的质量加权（barycentric）合并算子；④在 DINOSAUR 端到端框架中无学习模块即可插拔，显著提升对象分割与发现性能。

**🔧 技术方法**

核心技术包括 Slot Attention、Soft‑IoU 计算、质量加权合并算子、固定合并策略、基于直方图的阈值估计（triangle thresholding）、增量式重叠更新以及预训练特征重构的 DINOSAUR 解码器。

**📊 数据集**

实验使用四个公开的无监督对象学习基准：CLEVR、CLEVR‑10、SpriteWorld 以及 ShapeWorld（或类似的多对象合成数据集）。

**📈 对比分析**

在 DINOSAUR 框架下与 Slot Attention、Adaslot、MetaSlot、GENESIS‑V2 等基线进行比较。实验结果表明，合并策略在对象发现和分割任务上提高了 5–10% 的 mAP/AR 等指标，且在多物体复杂场景中的掩码质量更为连贯。

**⚠️ 局限性**

局限性包括：①合并阈值虽由数据自适应但仍需手动设置阈值参数；②固定合并策略在极端重叠或稀疏场景下可能导致信息丢失；③合并仅在训练阶段启用，推理阶段未评估其效果；④对极大槽数的场景仍可能产生计算开销，尽管已做增量优化。

---

## 150. Comparison of Outlier Detection Algorithms on String Data

**arXiv ID:** 2603.11049 | [PDF](https://arxiv.org/pdf/2603.11049v1)

**作者:** Philip Maus `[一作]` `[通讯]`, Philip Maus

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了两种针对单词字符串的离群值检测算法，分别是基于局部异常因子（LOF）的加权Levenshtein距离方法和基于层次左正则表达式（HiLRE）的学习方法；

**💡 创新点**

创新点在于为字符串数据设计了层次化字符类别加权的Levenshtein距离，以及一种利用HiLRE构造最优正则表达式并通过子集差值选择的离群检测框架；

**🔧 技术方法**

采用了改进的LOF算法（使用Hierarchical Levenshtein距离和KFCS自动选取k），以及HiLRE学习与子集差值选取算法，并引入阈值因子与最小匹配比例p_min；

**📊 数据集**

实验数据集包括合成的ISO 8601日期字符串、医院质量报告中的邮政编码、县名、街道名、房号、时间戳等真实文本；

**📈 对比分析**

通过多次随机抽样、ROC曲线和误报率评估，两种方法在不同数据结构下表现不同：HiLRE在可用正则表达式匹配的数据上召回率高且误报低；LOF在长度差异明显的数据上表现更好；

**⚠️ 局限性**

局限性包括对多词或需语义上下文的字符串检测能力不足，HiLRE在复杂或分散的数据上可能无法找到合适表达式，且两种方法在参数调优和理论复杂度分析上尚缺乏深入研究；

---

## 151. QUARE: Multi-Agent Negotiation for Balancing Quality Attributes in Requirements Engineering

**arXiv ID:** 2603.11890 | [PDF](https://arxiv.org/pdf/2603.11890v1)

**作者:** Haowei Cheng `[一作]` (Waseda University), Hironori Washizaki `[通讯]` (Waseda University)

**通讯引用:** 3307 | [OpenAlex ID](https://openalex.org/A5033111691)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种多代理协商框架 QUARE，利用专门化的质量代理（安全、效率、绿色、可信度、责任）与调度器共同分析需求并生成 KAOS 目标模型；

**💡 创新点**

创新点在于：①基于 ISO/IEC 25010 的质量维度拆分实现代理专门化；②引入对话式辩证协商协议（thesis‑antithesis‑synthesis），显式识别并解决跨质量冲突；③在协商后通过 RAG 自动化验证提升标准合规性与可验证性；

**🔧 技术方法**

使用大型语言模型（LLM）为核心推理引擎，结合多代理架构、KAOS 模型生成、BERTScore 语义一致性评估、RAG 依赖检索与标准条款推断；

**📊 数据集**

使用公开 RE 基准（MARE、iReDev）以及工业级自动驾驶规范共五个案例，覆盖安全、金融、信息系统等领域；

**📈 对比分析**

与单代理、无协商多代理、MARE、iReDev 四种基线对比，指标包括需求覆盖率、CHV、MDC、CU、MAC、语义保留率、冲突解决率、DAG 合法性、合规覆盖率等；结果显示 QUARE 平均产生 35 条需求，语义保留率 94.9%，合规覆盖率 98.2%（比基线提升 105%），并在安全、合规场景下表现尤为突出；

**⚠️ 局限性**

局限性包括：依赖单一 LLM 方案，未进行大规模人类评估；基线 MARE 与 iReDev 需重新实现，可能未完全复现原行为；实验覆盖的案例有限，缺乏更广泛的工业部署验证；未进行统计显著性检验，结果主要基于平均值；

---

## 152. Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models

**arXiv ID:** 2603.12118 | [PDF](https://arxiv.org/pdf/2603.12118v1)

**作者:** Jae-Won Chung `[一作]` (University of Michigan), Mosharaf Chowdhury `[通讯]` (University of Michigan)

**通讯引用:** 14749 | [OpenAlex ID](https://openalex.org/A5013180923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了Cornserve，一个分布式服务系统，专门用于处理任意输入输出多模态模型（Any-to-Any Models）；

**💡 创新点**

创新点包括：①灵活的任务抽象（单元任务、复合任务与应用）实现任意模型计算图的表达；②模型拆分（Model Fission）在组件边界实现独立扩展；③基于记录-重放的执行模型实现动态请求路径的高效调度；④共享执行器自动降低 GPU 使用；

**🔧 技术方法**

技术主要包括：Kubernetes + Python 23K 代码实现；自定义任务抽象与模型拆分；记录-重放执行机制；共享内存与 RDMA 侧车（Sidecar）实现组件间张量传输；OpenTelemetry 观测；针对不同组件的专用执行器（vLLM、Eric、Geri）；

**📊 数据集**

使用 ServeGen 真实多模态请求数据集进行评估；

**📈 对比分析**

与单机 monolithic 部署（所有组件在同一进程）进行对比。对 Qwen 2.5 Omni 7B 与 Qwen 3 Omni 30B 进行吞吐量和延迟评估。Cornserve 在 8/16 GPU 上分别提升吞吐量 3.09×/3.81×，延迟 P50/P95/P99 分别提升 3.24×/5.3×/5.79×；

**⚠️ 局限性**

局限：记录-重放要求任务控制流在请求下保持确定性；无法直接支持基于 LLM 回应的动态任务选择，需在应用层手动处理；对极大模型仍依赖显存，拆分仍可能导致显存不足。

---

## 153. A Collaborative and Pattern-Based Training Approach to Knowledge Acquisition and Decision-Making During the Design of Software Architectures Courses: A Case Study

**arXiv ID:** 2603.11904 | [PDF](https://arxiv.org/pdf/2603.11904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 154. RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset

**arXiv ID:** 2603.11811 | [PDF](https://arxiv.org/pdf/2603.11811v1)

**作者:** Yongzhong Wang `[一作]` (Southern University of Science and Technology), Feng Zheng `[通讯]` (Spatialtemporal AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RADAR 系统，实现全自动闭环机器人数据采集，涵盖任务生成、执行、评估与自动重置。

**💡 创新点**

将 Vision‑Language Model 与 Graph Neural Network ICL 分工为脑‑小脑协同，利用 3D 演示库和 LIFO 逆序规划实现无人工重置与自我恢复，形成持续自给的数据生成引擎。

**🔧 技术方法**

使用 Vision‑Language Model 进行语义规划与评估、Graph Neural Network + diffusion 的 in‑context imitation learning 执行、三阶段 VQA 评估、有限状态机驱动的前后向规划。

**📊 数据集**

在 RLBench 仿真中测试 7 个原子任务和 3 个长序列任务；在真实机器人上使用 RealSense D435i + SAM/XMem++ 进行实时 3D 分割，数据以少量人类演示为 3D 先验。

**📈 对比分析**

与 MOKA、ReKep 在相同任务下对比，RADAR 在原子任务上提升至 80–100%，在长序列任务上从 <10% 提升至 80–90%，显著优于基线。

**⚠️ 局限性**

重置可靠性仍有限，正向+逆向成功率受累积错误影响，需进一步多模态感知（如触觉、视觉高速伺服）以降低重置失败。

---

## 155. Translationese as a Rational Response to Translation Task Difficulty

**arXiv ID:** 2603.12050 | [PDF](https://arxiv.org/pdf/2603.12050v1)

**作者:** Maria Kunilovskaya `[一作]` (Saarland University), Maria Kunilovskaya `[通讯]` (Saarland University)

**通讯引用:** 113 | [OpenAlex ID](https://openalex.org/A5012912336)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了翻译ese是否可由翻译任务的认知负荷预测，并构建了段级翻译性得分分类器与回归模型

**💡 创新点**

首次将信息理论指标（LLM产生的surprisal和译解熵）与传统结构特征结合，用于解释翻译任务难度与翻译ese之间的关联

**🔧 技术方法**

采用GPT‑2与NMT模型计算surprisal、使用BERT获取对齐得分、计算译解熵，并利用SVM分类、线性回归、RFECV与SHAP进行特征选择与解释

**📊 数据集**

使用EPIC‑EuroParl‑UdS双向英德语料，涵盖书面与口译两种模式，约2万段对齐文本

**📈 对比分析**

通过宏F1评估分类（段级≈60%，文档级≈80%），用R²与MAE评估回归，最高R²达0.21（英→德书面），表明任务难度能解释约21%的翻译ese方差

**⚠️ 局限性**

局限包括对特定语料与语言对的依赖、特征提取的脆弱性、缺乏说话者层面控制、未对单独的语言学特征进行深入分析

---

## 156. You Told Me to Do It: Measuring Instructional Text-induced Private Data Leakage in LLM Agents

**arXiv ID:** 2603.11862 | [PDF](https://arxiv.org/pdf/2603.11862v1)

**作者:** Ching-Yu Kao `[一作]` (Fraunhofer AISEC), Philip Sperl `[通讯]` (Fraunhofer AISEC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对高特权 LLM 代理执行外部 README 文档中的指令进行攻击，并系统评估文档嵌入式指令注入威胁，提出三维度分类与 ReadSecBench 基准。

**💡 创新点**

① 构建 500 个真实 README 注入样本的 ReadSecBench；② 定义语言伪装、结构混淆、语义抽象三维度度量框架；③ 量化“受信执行者困境”和“语义安全差距”，系统评估多模型、多语言与人类检测。

**🔧 技术方法**

使用 LLM 代理（如 Claude Computer Use、OpenDevin 等）、LangChain 仿真、规则与 LLM 基准检测器、用户研究与实验平台沙箱等技术。

**📊 数据集**

采用 500 个来自 GitHub 热门仓库的 README 文件（含正负样本），按语言分布，构成实验数据集。

**📈 对比分析**

对比四大 LLM 家族（Claude、GPT、Gemini）在模拟环境下的语义合规率为 46%–79%；在真实代理上攻击成功率高达 85%；规则/LLM 防御检测率低且误报高。

**⚠️ 局限性**

受限于实验规模、仅评估单一高特权代理、受试者来自同一校且技术背景相似；未测试闭源 IDE 代理；仅关注数据泄露攻击，未涵盖持久性或横向移动等更深层威胁。

---

## 157. Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics

**arXiv ID:** 2603.11360 | [PDF](https://arxiv.org/pdf/2603.11360v1)

**作者:** Yangyang Qu `[一作]` (EURECOM), Evans Nicholas `[通讯]` (EURECOM)

**通讯引用:** 9989 | [OpenAlex ID](https://openalex.org/A5066811192)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Fair-Gate 框架，改进声纹识别系统在不同性别群体之间的公平性与性能平衡。

**💡 创新点**

创新点在于结合风险外推（REx）与局部互补门控：REx 在代理性别组间平衡分类风险，门控在特征层面显式划分身份与性别信息，提升可解释性并减少性别快捷路径。

**🔧 技术方法**

采用 ECAPA‑TDNN 作为基干网络，加入深度可分离卷积门控、梯度反转层、性别分支、嵌入去相关损失与 REx 正则化。

**📊 数据集**

在 VoxCeleb2 训练集上训练，在 VoxCeleb1 的原始、扩展与困难验证协议（Vox1‑O/E/H）上评估。

**📈 对比分析**

与 ECAPA‑TDNN、ECAPA‑TDNN+GRL 以及 VoxDisentangler 基线相比，Fair‑Gate 在 Vox1‑E 与 Vox1‑H 协议下取得最小 EER 与最低 GARBE（性别公平度），在保持或提升整体准确率的同时显著缩小性别差距。

**⚠️ 局限性**

局限性包括仅使用代理性别标签、对不同性别特征的可解释性仍有限、在跨域或更大规模数据集上的泛化需进一步验证。

---

## 158. PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation

**arXiv ID:** 2603.11917 | [PDF](https://arxiv.org/pdf/2603.11917v1)

**作者:** Pietro Bonazzi `[一作]` (ETH Zurich), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7785 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了PicoSAM3，一种轻量级、可提示的语义分割模型，专为在Sony IMX500传感器上实现实时、低延迟推理而设计。

**💡 创新点**

创新点在于结合ROI（区域兴趣）隐式提示、Efficient Channel Attention以及对SAM2/SAM3的大规模知识蒸馏，实现仅1.3M参数、INT8量化后仍保持高mIoU的模型。

**🔧 技术方法**

使用了稠密CNN架构、膨胀瓶颈层、ECA注意力、ROI裁剪隐式提示、知识蒸馏、后训练量化以及在IMX500 DSP上的原生部署技术。

**📊 数据集**

训练和评估均基于COCO和LVIS两个公开标注数据集。

**📈 对比分析**

与SAM-H、FastSAM、TinySAM、EdgeSAM、LiteSAM等基线对比，PicoSAM3在COCO上达到65.45% mIoU、LVIS 64.01% mIoU，INT8量化版在IMX500上实现11.82 ms的推理时延，显著优于同等复杂度模型。

**⚠️ 局限性**

局限性包括固定的96×96输入分辨率、对精准框定位的依赖、单目标裁剪的限制，以及对更复杂或未见过场景的泛化能力仍需提升。

---

## 159. Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

**arXiv ID:** 2603.11439 | [PDF](https://arxiv.org/pdf/2603.11439v1)

**作者:** Seung Hyup Baek `[一作]` (Konkuk University), Jae Won Cho `[通讯]` (Konkuk University)

**通讯引用:** 49717 | [OpenAlex ID](https://openalex.org/A5100681585)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了基于查询的稠密视频字幕生成框架 ROS‑DVC，旨在同时完成视频事件定位与描述。

**💡 创新点**

核心创新点包括：1）将定位查询与描述查询完全分离为独立的角色特定查询；2）引入跨任务对比对齐（CTCA）损失以保证两类查询语义一致；3）设计重叠抑制损失（OSL）主动减少重叠事件；4）加入概念引导器提升字幕语义丰富度。

**🔧 技术方法**

技术实现上采用基于 DETR 的双层可变形 Transformer 编码器-解码器，利用 CLIP ViT‑L/14 视觉特征，结合跨任务对比损失、重叠抑制损失和概念引导损失共同训练。

**📊 数据集**

在 YouCook2（2k 烹饪视频）与 ActivityNet Captions（20k 生活活动视频）两个主流稠密字幕基准数据集上进行评估。

**📈 对比分析**

与现有非 LLM 方法相比，ROS‑DVC 在 YouCook2 上 CIDEr 提升至 39.18（最高），F1 提升至 32.03；在 ActivityNet 上 CIDEr 35.04、SODA_c 6.45，均位列前列；同时定位召回/精度也大幅提升，表明方法在定位与字幕质量上均优于基线。

**⚠️ 局限性**

局限性包括：1）模型仍需大量计算资源；2）对 γ、α 等超参数敏感；3）缺乏外部记忆或预训练语言模型，导致在极长文本生成上略逊；4）仅在两大数据集验证，跨域泛化尚待进一步探索。

---

## 160. Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning

**arXiv ID:** 2603.11351 | [PDF](https://arxiv.org/pdf/2603.11351v1)

**作者:** Hong Lu `[一作]` (Tufts University), Matthias Scheutz `[通讯]` (Tufts University)

**通讯引用:** 9192 | [OpenAlex ID](https://openalex.org/A5044523801)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种融合大语言模型（LLM）、符号规划与强化学习的混合架构，用于动态开放世界中机器人面对新物体时缺失操作符的识别、规划生成以及控制策略学习，实现完整的计划-学习-执行循环。

**💡 创新点**

创新点包括：① 利用LLM进行缺失操作符的结构化生成并用自一致性提示提升准确性；② 让LLM生成多种奖励函数候选，并通过基因算法逐步淘汰表现最差的，显著加速强化学习；③ 将符号规划与连续控制的RL无缝结合，在连续空间中解决operator discovery难题；④ 提供端到端的Hybrid LLM Symbolic Planner + Subgoal Learning框架。

**🔧 技术方法**

主要技术手段有：GPT‑o3 进行规划与提示；GPT‑o4‑mini 生成奖励函数代码；self‑consistency 与 Tree‑of‑Thoughts 提升LLM推理质量；搜索‑与‑提示算法结合符号搜索；PPO（stable_baselines3）用于学习连续控制；遗传算法启发式奖励函数筛选。

**📊 数据集**

实验基于 mimicgen 仿真环境，涵盖 Kitchen、Nut Assembly、Coffee‑Box、Coffee‑Drawer 四个开放世界任务；使用相应的 PDDL 域与问题文件，训练 RL 策略。

**📈 对比分析**

对比方法包括 Operator Discovery（OD）、LEAGUE‑sparse（LS）和 Reward Machine（RM）。实验结果表明：Hybrid LLM Symbolic Planner 能在所有域中正确识别缺失操作符；LLM‑guided RL 在成功率与进度上显著优于 LS 与 RM（p<0.05），在难度较高的域中 OD 几乎无法发现可行状态。与 OD 相比，Hybrid 方案在平均时间上快 1–2 小时以上。

**⚠️ 局限性**

局限性在于：① 需要先验完整的谓词集合，机器人必须能够检测并分类新物体；② 目前使用真值状态验证子目标，实际部署需依赖感知模块；③ 方案在仿真中验证，真实机器人需要 sim‑to‑real 技术；④ 对 LLM 的依赖导致成本与推理延迟；⑤ 目前仅处理单一新颖物体，尚未扩展到多重并发新颖性。

---

## 161. Long-Context Encoder Models for Polish Language Understanding

**arXiv ID:** 2603.12191 | [PDF](https://arxiv.org/pdf/2603.12191v1)

**作者:** Sławomir Dadas `[一作]` (National Information Processing Institute), Przemysław Boruta `[通讯]` (PKO Bank Polski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一款支持8192 token长上下文的Polish encoder（polish-roberta-8k）及其压缩版

**💡 创新点**

通过两阶段训练（先扩展位置嵌入再全参数微调）、无污染打包、Flash Attention以及知识蒸馏，显著提升了长文本处理能力并实现模型压缩

**🔧 技术方法**

使用Transformer encoder、RoBERTa架构、Flash Attention、contamination‑free packing、全参数两阶段预训练、whole‑word masking及token‑级MSE蒸馏

**📊 数据集**

训练基于150B Polish语料的预训练模型；下游评测涵盖25个任务，包括KLEJ基准、FinBench银行金融任务、SICK、GCN、MIPD等公开数据集以及银行内部8B token域适配数据

**📈 对比分析**

与herbert-large-cased、polish-roberta-large-v2以及多语种模型（XLM‑Roberta、mmBERT等）做对比，平均性能从84.88%提升至85.93%，在长文本任务（Banking‑Long、MIPD、EURLEX等）表现最为突出；压缩版仍保持接近全尺寸性能

**⚠️ 局限性**

对极长序列仍有限制；蒸馏仅采用token级MSE，未蒸馏注意力矩阵；模型规模大导致显存/推理成本高；跨语言性能不及专门多语种模型；缺乏实时推理和部署可扩展性的系统性评估

---

## 162. Understanding LLM Behavior When Encountering User-Supplied Harmful Content in Harmless Tasks

**arXiv ID:** 2603.11914 | [PDF](https://arxiv.org/pdf/2603.11914v1)

**作者:** Junjie Chu `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估大语言模型在执行无害任务时，面对用户提供的有害内容是否会继续输出有害信息的风险；

**💡 创新点**

提出并量化“内容级有害风险”（in‑content harm risk），揭示了现有模型在任务级别已对齐但在内容级别仍存在安全缺口；

**🔧 技术方法**

采用大语言模型评估框架，结合OpenAI Moderation API、手工标注和多种外部安全防护（Llama Guard、Moderation API等）进行检测；

**📊 数据集**

构建了1,357条跨10类有害知识的知识库，以及9种无害任务，形成用户输入–任务组合的评测数据集；

**📈 对比分析**

通过K‑HRN、T‑HRR和GS三种指标对9种主流LLM（包括GPT‑5.2、Gemini‑3‑Pro、Qwen3等）进行对比，发现多数模型在翻译等用户依赖度高的任务中有超过50%的有害响应率；

**⚠️ 局限性**

限制包括任务样本覆盖不足、外部安全评估工具误判、模型版本更新不一定提升内容安全、实验数据不完全代表真实交互情境，以及对OpenAI政策定义的依赖导致类别偏差。

---

## 163. ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models

**arXiv ID:** 2603.12208 | [PDF](https://arxiv.org/pdf/2603.12208v1)

**作者:** Yingxin Lai `[一作]` (Great Bay University), Xiaochun Cao `[通讯]` (School of Cyber Science and Technology, Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 ForensicZip，一种训练无关、基于伪造检测的视觉-语言模型视觉令牌压缩框架。

**💡 创新点**

创新点在于将令牌压缩从语义驱动转变为伪造驱动：通过引入带虚拟节点的 Birth‑Death 最优传输捕捉时间上的物理不连续性，并结合高频拉普拉斯先验实现对伪造痕迹的精准保留。

**🔧 技术方法**

使用的技术包括：增广的 Birth‑Death Entropic OT、频域拉普拉斯高频先验、物理 Top‑K 选择，以及多模态 Transformer 推理。

**📊 数据集**

实验数据集涵盖 FakeVLM（FakeClue、LOKI）、DMimage、FakeShield‑AIGC、SIDA、DD‑VQA、FakeShield‑DeepFake、DFFD、FakeShield‑PhotoShop、POPE 等深度伪造、AIGC 与传统篡改数据集。

**📈 对比分析**

与 FastV、SparseVLM、VisionTrim、LLaVA‑Scissor 等语义驱动压缩基线以及完整 Token 模型对比；在 10% 令牌保留率下 ForensicZip 维持超过 90% 的准确率，且实现近 3× 的速度提升与 FLOPs 减少，显著优于其他方法。

**⚠️ 局限性**

局限性包括：依赖投影与 OT 求解，仍有额外计算开销；对极低分辨率或极短视频的表现可能受限；未充分验证跨模型大规模鲁棒性与非线性时空变化的适应性。

---

## 164. CAETC: Causal Autoencoding and Treatment Conditioning for Counterfactual Estimation over Time

**arXiv ID:** 2603.11565 | [PDF](https://arxiv.org/pdf/2603.11565v1)

**作者:** Nghia D. Nguyen `[一作]` (University of Illinois Urbana-Champaign), Lav R. Varshney `[通讯]` (Stony Brook University)

**通讯引用:** 7205 | [OpenAlex ID](https://openalex.org/A5065423139)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种适用于时间序列的反事实因果效应估计新方法，采用部分自编码器实现可逆表征，并将治疗作为条件对表征进行 FiLM 调整。

**💡 创新点**

创新点在于：①可逆自编码表征保证信息保留；②将治疗作为表征的显式条件而非简单拼接，增强表达力；③通过熵最大化的对抗游戏实现治疗不变表征；④在训练中加入对所有潜在治疗的条件化损失，提升反事实精度。

**🔧 技术方法**

技术方法包括：对抗表示学习、可逆自编码器、FiLM 条件层、熵最大化对抗、LSTM/TCN 序列骨干、时间截断（temporal cutoff）与自回归解码。

**📊 数据集**

使用的数据集：完全合成的非小细胞肺癌（NSCLC）仿真数据；基于 MIMIC‑III 的半合成数据；以及真实世界的 MIMIC‑III 观测数据（仅用于可观测结果评估）。

**📈 对比分析**

与 RMSN、CRN、CT、以及无对抗训练的 LSTM 进行比较，实验结果显示在合成、半合成以及可观测真实数据上，方法在 5‑步预测的 RMSE 均显著低于基线；消融实验验证各损失项对性能的贡献。

**⚠️ 局限性**

局限性：①在真实数据中缺乏反事实标签，评估依赖可观测误差；②对超参数敏感；③可能在高度非平稳或极端共线性场景下失效；④对模型解释性和计算成本的进一步研究仍需开展。

---

## 165. MDER-DR: Multi-Hop Question Answering with Entity-Centric Summaries

**arXiv ID:** 2603.11223 | [PDF](https://arxiv.org/pdf/2603.11223v1)

**作者:** Riccardo Campi `[一作]` (Politecnico di Milano), Piero Fraternali `[通讯]` (Politecnico di Milano)

**通讯引用:** 7309 | [OpenAlex ID](https://openalex.org/A5024296934)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于知识图谱的多跳问答框架MDER-DR，整合了索引与检索流程。

**💡 创新点**

通过MDER在索引阶段将多跳信息压缩为实体中心摘要，DR则在检索阶段通过语义分解和迭代解析实现多跳推理，避免实时图遍历。

**🔧 技术方法**

采用LLM驱动的文本预处理、三元组抽取、实体消歧、描述丰富和摘要生成；利用向量检索、实体匹配与循环解析，构建RAG与GraphRAG的对比实验。

**📊 数据集**

在WikiQA、HotpotQA和领域专用BenchEE等多语言（英/意/法/西）数据集上进行实验，抽样500问答对。

**📈 对比分析**

与LLM-only、Vector-RAG、GraphRAG基线比较，使用LLM-as-a-Judge、Soft EM和人工评估，MDER-DR在WikiQA最高提升约66%，在HotpotQA和BenchEE也实现显著改善，且对语言不匹配的鲁棒性更好。

**⚠️ 局限性**

预处理和索引阶段需多次LLM调用，计算成本较高；未在更大规模数据集上验证；缺少完整消融实验。

---

## 166. TimeSqueeze: Dynamic Patching for Efficient Time Series Forecasting

**arXiv ID:** 2603.11352 | [PDF](https://arxiv.org/pdf/2603.11352v1)

**作者:** Sravan Kumar Ankireddy `[一作]` (University of Texas at Austin), C. Bayan Bruss `[通讯]` (Capital One)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5017427561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种动态补丁化（Dynamic Patching）tokenizer，先用轻量的状态空间模型（SSM）编码器提取全分辨率的细粒度特征，再根据局部信号的相对偏差阈值自适应划分补丁，保留信息密集区的细粒度采样，平滑区使用大补丁，从而在保持时间精度的同时显著压缩Transformer的输入长度；

**💡 创新点**

核心创新在于①基于局部功率的相对偏差阈值实现无监督、可自适应的补丁边界划分；②将SSM的高效线性特征提取与动态补丁结合，既保留点级细节，又实现大幅度压缩；③仅保留补丁边界的embedding进行下采样并在Transformer后恢复，从而避免信息丢失并保持因果性；④在预训练中获得约20×加速、8×数据效率提升。

**🔧 技术方法**

采用Mamba等SSM实现轻量编码器/解码器；相对偏差阈值补丁算法；Mixture‑of‑Experts Transformer（Time‑MoE）作为后端；多时域预测头；Huber自回归损失+辅助负载平衡损失；预训练使用Time‑300B。

**📊 数据集**

预训练数据：Time‑300B（300 billion time‑points，跨天气、交通、金融等多域）；评估基准：ECL、ETTH1/2、Weather、Traffic、MSL、Exchange等长周期预测数据集。

**📈 对比分析**

在相同Transformer后端下，将Time‑MoE的点级tokenizer替换为动态补丁tokenizer；在零射和全射实验中，动态补丁模型与基准模型相当或略优；预训练速度提升≈20×，数据效率提升≈8×；GPU内存下降≈3.4×，推理吞吐量提升≈10.5×；与Moirai、TimesFM、Moment、Chronos等state‑of‑the‑art模型相比，取得竞争或超越的性能。

**⚠️ 局限性**

仍需手动调节阈值τ以控制压缩率，缺乏端到端学习补丁边界的机制；对不同采样频率/季节性可能需要统一参数；在某些场景下Large版本表现略低；对高噪声或非均匀采样的数据效果尚未充分验证。

---

## 167. Unifying Logical and Physical Layout Representations via Heterogeneous Graphs for Circuit Congestion Prediction

**arXiv ID:** 2603.11075 | [PDF](https://arxiv.org/pdf/2603.11075v1)

**作者:** Runbang Hu `[一作]` (University of Texas at Arlington), Yuede Ji `[通讯]` (University of Texas at Arlington)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5066436066)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出VeriHGN框架，将电路元件与空间网格统一为异构图，实现早期路由拥塞预测

**💡 创新点**

创新点在于融合网表拓扑与物理布局为单一异构图，并引入多尺度网格关系与关系特定消息传递

**🔧 技术方法**

采用异构图神经网络（HGT/多关系MLP）、多层级网格构建、加权回归损失和方差正则化等技术

**📊 数据集**

在ISPD2015、CircuitNet-N14、CircuitNet-N28三大工业基准上进行实验

**📈 对比分析**

与GCN、GAT、CircuitGNN、MIHC等基线相比，VeriHGN在MAE、RMSE、Pearson、Spearman、Kendall等指标均表现更优，尤其在排名相关性上明显提升

**⚠️ 局限性**

缺陷主要在跨设计迁移性能不足，零样本迁移下精度显著下降，且对极大规模电路的计算资源仍有一定需求

---

## 168. Chemical Reaction Networks Learn Better than Spiking Neural Networks

**arXiv ID:** 2603.12060 | [PDF](https://arxiv.org/pdf/2603.12060v1)

**作者:** Sophie Jaffard `[一作]` (Max Planck Institute of Molecular Cell Biology and Genetics), Ivo F. Sbalzarini `[通讯]` (Dresden University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无隐藏层的化学反应网络（CRN），并通过严格的理论证明展示其在监督分类任务中的学习能力；

**💡 创新点**

证明CRN在满足一定条件下可实现与隐藏层Spiking Neural Network（SNN）相同甚至更优的学习保证，且所需网络结构更简单，体现了化学反应网络在学习任务上的潜在优势；

**🔧 技术方法**

采用确定性质量作用动力学、专家聚合（Exponentially Weighted Average, EWA）算法以及误差回报（regret）理论，推导出局部和全局的误差上界、VC维度以及收敛性分析；

**📊 数据集**

使用手写数字（8×8像素）的数据集（来自scikit‑learn），将像素浓度映射为输入化学物种；

**📈 对比分析**

将CRN的性能与在相同任务下的SNN进行对比，结果显示：在深度n=1时CRN达到约85.8%准确率，n=2时达到约88.6%，均优于对应的SNN（无隐藏层53%，一隐藏层83.5%），且CRN实现更低的网络复杂度；

**⚠️ 局限性**

实验和理论均假设输入特征满足特定结构（如二值化、类由特征集合分解等），且未在真实生物体系中实现，实际可行性和对噪声鲁棒性的进一步验证仍待研究。

---

## 169. The Latent Color Subspace: Emergent Order in High-Dimensional Chaos

**arXiv ID:** 2603.12261 | [PDF](https://arxiv.org/pdf/2603.12261v1)

**作者:** Mateusz Pach `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 16246 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并解释了FLUX文本到图像模型中VAE潜在空间的颜色子空间，提出了一种训练‑无关的颜色观察与干预方法。

**💡 创新点**

创新点在于首次发现颜色在潜在空间形成HSL样的双锥结构，并基于此构建了从潜在坐标到HSL颜色的双向映射，实现了对中间生成步骤的颜色可视化与直接操控。

**🔧 技术方法**

主要技术包括VAE编码、PCA降维、流匹配(FM)动态分析、几何映射实现LCS↔HSL的编码/解码、以及基于注意力分割的局部颜色干预。

**📊 数据集**

实验数据集包括FLUX预训练模型、26张纯色墙面图像、GenEval单物体颜色评测集以及4,080张自然图像的精细颜色基准集。

**📈 对比分析**

与仅通过提示颜色或原始FLUX的对比实验显示，本文方法在全局和局部颜色准确率上分别提升至73%和70%，ΔE值从22降至9，结构相似度(SSIM、LPIPS、DINOv2)也显著优于提示方法。

**⚠️ 局限性**

局限性包括对分割质量高度依赖、在极早或极晚的扩散步长上效果不佳、以及需要对VAE潜在空间进行额外统计归一化，限制了在所有生成阶段的通用性。

---

## 170. KEPo: Knowledge Evolution Poison on Graph-based Retrieval-Augmented Generation

**arXiv ID:** 2603.11501 | [PDF](https://arxiv.org/pdf/2603.11501v1)

**作者:** Qizhi Chen `[一作]` (University of Electronic Science and Technology of China), Shuang Liang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 5132 | [OpenAlex ID](https://openalex.org/A5043630267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对GraphRAG的毒化攻击方法——Knowledge Evolution Poison（KEPo），通过伪造知识演化路径将有害信息注入知识图谱；

**💡 创新点**

创新点在于利用时间演化链路把毒化事件与原始事实自然衔接，降低条件困惑度，从而提升检索排名并误导LLM生成目标答案；

**🔧 技术方法**

主要技术包括LLM驱动的文本生成（Fabricator）、知识图谱构建与社区检索、演化路径生成、以及多目标跨子图协同链接；

**📊 数据集**

使用GraphRAG-Bench（Graph‑Story、Graph‑Medical）以及MuSiQue等问答数据集进行实验；

**📈 对比分析**

与三种基线（PoisonedRAG、CorruptRAG、GRAG‑Poison）及Naïve RAG对比，KEPo在所有GraphRAG框架上实现了最高的攻击成功率（ASR/CASR），多目标攻击进一步提升性能；

**⚠️ 局限性**

局限性包括对强大防御（如查询改写、指令忽略、提示检测）效果不佳，以及对LLM规模和生成质量的依赖，且在全局检索场景下效果略逊于局部检索。

---

## 171. Automatic Attack Script Generation: a MDA Approach

**arXiv ID:** 2603.11861 | [PDF](https://arxiv.org/pdf/2603.11861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 172. Multi-Station WiFi CSI Sensing Framework Robust to Station-wise Feature Missingness and Limited Labeled Data

**arXiv ID:** 2603.11858 | [PDF](https://arxiv.org/pdf/2603.11858v1)

**作者:** Keita Kayano `[一作]`, Tomoko Adachi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对多站点 WiFi CSI 感知的框架，兼顾站点缺失与标签稀缺问题。

**💡 创新点**

将自监督预训练与站点遮蔽增强联合使用，构建对站点缺失不变的表示。

**🔧 技术方法**

采用 CroSSL 自监督学习和 Station-wise Masking Augmentation (SMA) 技术。

**📊 数据集**

在办公室样本和工厂环境的真实多站点 CSI 数据集上进行评估。

**📈 对比分析**

与传统监督、数据增强和其他自监督基线对比，实验表明在不同缺失率和标签比例下均能保持最低 RMSE，优于对比方法。

**⚠️ 局限性**

对不同遮蔽率和环境的泛化性未做深入探究，且依赖大量无标签 CSI 进行预训练。

---

## 173. Real-time Rendering-based Surgical Instrument Tracking via Evolutionary Optimization

**arXiv ID:** 2603.11404 | [PDF](https://arxiv.org/pdf/2603.11404v1)

**作者:** Hanyang Hu `[一作]` (University of California San Diego), Michael C. Yip `[通讯]` (University of California San Diego)

**通讯引用:** 4450 | [OpenAlex ID](https://openalex.org/A5054598974)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于CMA-ES进化优化和批量渲染的实时手术工具跟踪框架，能够同时估计6-DoF姿态与可见关节角度；

**💡 创新点**

创新点在于将黑盒进化优化与GPU并行批量渲染结合，显著减少迭代次数并提高收敛鲁棒性；同时支持关节角度无输入、双手臂同步跟踪和时序Kalman滤波；

**🔧 技术方法**

核心技术包括CMA-ES演化优化、批量前向运动学与渲染、Surgical SAM 2分割、深度学习关键点检测、Kalman滤波；

**📊 数据集**

使用合成轨迹数据集（16对、1000帧/对）和真实数据集SurgPose（12条轨迹）以及自采集的带标记与关节角度的轨迹；

**📈 对比分析**

与梯度下降、XNES及Richter等粒子滤波基线对比，CMA-ES在保持3次迭代的同时，精度提升约30%–40%，帧率提升至≈43 fps（相较粒子滤波约19 fps），总体在实时性与准确性上表现最佳；

**⚠️ 局限性**

局限在于高度依赖分割质量，遮挡或分割误差会直接影响姿态估计，且在双工具重叠的真实手术场景中尚未充分验证。

---

## 174. EmbTracker: Traceable Black-box Watermarking for Federated Language Models

**arXiv ID:** 2603.12089 | [PDF](https://arxiv.org/pdf/2603.12089v1)

**作者:** Haodong Zhao `[一作]` (Shanghai Jiao Tong University), Gongshen Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2034 | [OpenAlex ID](https://openalex.org/A5085695760)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对联邦学习语言模型（FedLM）提出一种服务器端、可追溯的黑盒水印框架，利用词嵌入空间中的后门触发词实现身份特定水印，支持在不需要客户端参与的情况下实现模型泄漏追踪。

**💡 创新点**

创新点包括：①在词嵌入层直接注入可追溯的后门水印；②通过服务器一次性训练全局触发词嵌入，然后针对每个客户端替换嵌入实现身份独特性，避免逐客户端训练；③实现完全黑盒验证与追踪；④对多种PEFT（LoRA、Prefix Tuning）和多种FL优化器（FedAvg、FedAvgM、FedProx、SCAFFOLD）兼容，且对客户端数量、数据异质性和攻击（微调、剪枝、量化、噪声、重写）具有高鲁棒性。

**🔧 技术方法**

技术手段包括：后门式嵌入（embedding poisoning），触发词生成与数字签名映射，服务器端的模型聚合与嵌入替换，黑盒验证（触发词+目标输出），并在实验中使用BERT、Llama‑2‑7B、Qwen2.5‑VL‑7B‑Instruct等大型语言模型。

**📊 数据集**

实验数据集覆盖文本分类（SST‑2、Enron、Twitter）、多类分类（AGNews、DBpedia、Yahoo）、问答（FreebaseQA、CoQA、NQ）以及视觉‑语言问答（OK‑VQA、OCR‑VQA）。在这些任务上均验证了模型精度与水印效果。

**📈 对比分析**

与现有方案（WAFFLE、FedTracker、TraMark）对比，本文在所有设置下实现了接近 100% 的验证率（VR）且误识率低，保持 1–2% 的任务精度下降；相对 TraMark 的多客户端训练开销，本文仅需一次全局嵌入训练，时间开销明显更小；与白盒方案 FedTracker 的差距在可黑盒验证上被彻底消除。

**⚠️ 局限性**

局限性包括：①需要服务器在一开始对全局触发词进行一次训练，若服务器资源有限可能受限；②方案依赖于词嵌入空间，若模型架构不具备可直接替换的嵌入层可能无法迁移；③对极大客户端数量（数千或更多）时触发词唯一性的保证与存储成本尚未充分评估；④在极端攻击（如同步多重水印覆盖）时需要额外的时间戳等机制进行冲突解消。

---

## 175. Scaling Laws for Educational AI Agents

**arXiv ID:** 2603.11709 | [PDF](https://arxiv.org/pdf/2603.11709v1)

**作者:** Mengsong Wu `[一作]` (East China Normal University), Aimin Zhou `[通讯]` (East China Normal University)

**通讯引用:** 9833 | [OpenAlex ID](https://openalex.org/A5050248676)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出教育 AI 的 Agent Scaling Law 并通过 AgentProfile 规范实现结构化能力定义，构建 EduClaw 平台完成 330+ 角色配置与 1,100+ 技能模块的自动化生成与部署。

**💡 创新点**

创新点在于把模型规模之外的结构维度（角色清晰度、维度深度、技能组合、工具完整度、多代理编排）量化为扩展轴，并引入 AgentProfile 作为可进化的开放式规范，形成“结构化能力系统”而非单纯规模化。

**🔧 技术方法**

采用大语言模型（LLM）驱动的三阶段构建管道，JSON 结构化 AgentProfile，OpenClaw 运行时多进程隔离，Node.js 管理层实现动态实例化与 SSE 聚合；技能与工具通过库匹配与自动生成。

**📊 数据集**

使用自建的 K‑12 教学技能库（1,100+ 份）与各学科课程标准作为数据支撑；角色描述以一语句场景输入进行生成，验证标准与学科覆盖率。

**📈 对比分析**

通过对 330+ 角色的质量与交互效果进行定性观察，发现配置丰富度与教学表现正相关；未给出严格的数值指标，但展示了快速实例化（<1min）与规模可扩展性。

**⚠️ 局限性**

局限性包括：评价主要为定性观察，缺乏量化学习成效实验；缺少长期跟踪与多模态支持；难以分离角色配置与基础模型或技能质量的影响；以及在高等教育等非 K‑12 场景的泛化性待验证。

---

## 176. Bridging the Cognitive Gap: Co-Designing and Evaluating a Voice-Enabled Community Chatbot for Older Adults

**arXiv ID:** 2603.11303 | [PDF](https://arxiv.org/pdf/2603.11303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 177. Adversarial Reinforcement Learning for Detecting False Data Injection Attacks in Vehicular Routing

**arXiv ID:** 2603.11433 | [PDF](https://arxiv.org/pdf/2603.11433v1)

**作者:** Taha Eghtesad `[一作]` (Pennsylvania State University), Aron Laszka `[通讯]` (Pennsylvania State University)

**通讯引用:** 2413 | [OpenAlex ID](https://openalex.org/A5049435924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在拥堵导航系统中，通过对交通路段的旅行时间注入误报来实施的伪数据注入攻击，并提出了一种基于零和博弈的检测框架，使用策略空间响应或偶（PSRO）与深度强化学习求解纳什均衡，从而得到既能最大化攻击者总行驶时间又能最小化防御者误报成本的最优策略。

**💡 创新点**

创新点在于：①将攻击者与检测器建模为一个完全信息零和博弈，能够主动应对适应性与隐蔽性攻击；②利用PSRO与PPO实现对连续攻击动作和二值检测动作的逼近最优反应；③通过纳什均衡策略实现对未知攻击的鲁棒防御，并在实验中证明其对多种基线攻击的显著优越性。

**🔧 技术方法**

技术方法包括：零和博弈建模、策略空间响应或偶（PSRO）算法、深度强化学习（PPO）做最佳反应、BPR函数生成交通拥堵模型、Bayesian统计做基线检测。

**📊 数据集**

实验使用三种网络数据集：Sioux Falls交通网络、3x2及5x4 Grid Random Edge（GRE）图网络，并随机化OD需求。

**📈 对比分析**

与基线攻击（贪婪攻击、Gaussian攻击）和基线防御（Bayesian异常检测）比较，实验显示：平衡策略下的攻击者行驶时间比最佳基线高19%–22%；防御者误报降低后，总行驶时间偏差分别降低约35%、24%和38%，并且在对抗实验中均显著优于所有基线。

**⚠️ 局限性**

局限性包括：①近似均衡可能未完全收敛，导致最优性保证有限；②攻击者假设全局可观测，现实中可能受限；③实验仅在模拟网络中验证，缺乏真实交通数据的现场测试；④计算开销相对较高，需进一步优化。

---

## 178. Language Model Teams as Distributed Systems

**arXiv ID:** 2603.12229 | [PDF](https://arxiv.org/pdf/2603.12229v1)

**作者:** Elizabeth Mieczkowski `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 49684 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将分布式系统理论应用于大语言模型团队（LLM Team）的设计与评估，并通过实验验证该框架的可行性。

**💡 创新点**

创新点在于：①将LLM团队与分布式系统的四大属性（独立性、通信、并发、容错）对应，形成系统化的分析模型；②使用Amdahl定律等分布式计算定量预测团队规模对速度提升的上限；③阐明中心化与去中心化架构在效率、协调成本、一致性冲突和慢任务影响上的本质差异。

**🔧 技术方法**

技术方法包括：Amdahl定律及相关可扩展性模型、分布式系统中的一致性与负载均衡理论、LLM推理与工具调用的自动化脚本、Python与LLM API的实验框架。

**📊 数据集**

使用的数据集主要是三类编程协作任务：1）实现数学工具库；2）构建数据分析流水线；3）渲染SVG文件；每类任务设定并行、混合和串行的子任务依赖结构，采用Claude‑Sonnet‑4‑6、Gemini‑3‑Flash和GPT‑5.2等模型生成。

**📈 对比分析**

对比方法：在预分配与自协调两种架构下，分别测量速度提升（speedup）、消息数量、空闲回合、测试失败数和token消耗。实验结果显示：①预分配（中心化）在多任务并行性高时速度提升最大；②自协调（去中心化）在减少慢任务影响方面有优势，但整体效率低于中心化；③一致性冲突和通信开销在去中心化中显著增加；④token使用量与速度提升不成正比，往往超越收益。

**⚠️ 局限性**

局限性包括：①任务仅为预定义的程序化子任务，缺乏对开放式、动态推理任务的验证；②实验使用同一基础模型的同质团队，未探讨异质性对性能的影响；③未深入研究故障容错、负载均衡和动态调度的实现；④仅考察单机API调用的开销，缺少多机真实部署环境的验证。

---

## 179. Adapting Dijkstra for Buffers and Unlimited Transfers

**arXiv ID:** 2603.11729 | [PDF](https://arxiv.org/pdf/2603.11729v1)

**作者:** Denys Katkalo `[一作]` (Igor Sikorsky Kyiv Polytechnic Institute), Toby Walsh `[通讯]` (University of New South Wales)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种改进的Dijkstra算法——Transfer Aware Dijkstra (TAD)，用于处理无限换乘且包含缓冲时间的公共交通路由问题。

**💡 创新点**

核心创新是通过扫描完整列车行程而非单个边来正确处理缓冲时间，消除传统TD‑Dijkstra的主导连接过滤错误。

**🔧 技术方法**

使用时间依赖Dijkstra、Bucket‑CH预处理以及直接在GTFS时间表上执行的ScanTrip实现。

**📊 数据集**

在伦敦和瑞士的公共交通数据集上进行实验，其中伦敦无缓冲时间，瑞士约51%站点设置缓冲时间。

**📈 对比分析**

与MR、ULTRA‑CSA、TD‑Dijkstra等算法对比，伦敦上TAD相较MR获得约2.17倍加速，瑞士上获得约2.88倍加速；ULTRA‑CSA最快但需要大量预处理。

**⚠️ 局限性**

限制在于TAD仍需对转移图进行CH预处理，且在大规模道路网络中预处理成本较高，且未实现实时延迟更新功能。

---

## 180. Enhancing Music Recommendation with User Mood Input

**arXiv ID:** 2603.11796 | [PDF](https://arxiv.org/pdf/2603.11796v1)

**作者:** Terence Zeng `[一作]` `[通讯]`, Terence Zeng

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出并实现了一种基于情绪（能量-情绪谱）的音乐推荐系统，并通过单盲实验验证其效果。

**💡 创新点**

创新点在于将用户情绪状态作为推荐依据，首次在音乐推荐中引入情绪感知机制，以提升个性化体验。

**🔧 技术方法**

采用能量-情绪谱模型对歌曲进行情绪特征提取，结合内容过滤算法实现推荐，并利用单盲实验和统计显著性检验评估性能。

**📊 数据集**

实验使用公开的音乐数据集（如GTZAN、Million Song Dataset等）并收集用户对推荐的情绪评分。

**📈 对比分析**

通过单盲实验将情绪辅助推荐与基线推荐进行对比，结果显示加入情绪后推荐质量显著提升（p<0.05）。

**⚠️ 局限性**

局限性包括数据集规模和情绪标签的主观性、实验样本量有限，以及缺乏对多模态情绪检测的进一步验证。

---

## 181. ZTab: Domain-based Zero-shot Annotation for Table Columns

**arXiv ID:** 2603.11436 | [PDF](https://arxiv.org/pdf/2603.11436v1)

**作者:** Ehsan Hoseinzade `[一作]` (Simon Fraser University), Ke Wang `[通讯]` (Simon Fraser University)

**通讯引用:** 16480 | [OpenAlex ID](https://openalex.org/A5100627950)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了 ZTab 框架，用域级别的零样本学习方式对表格列进行语义类型标注，无需用户提供标注数据。

**💡 创新点**

创新点在于：①通过域配置（预定义类列表 + 结构化表格模式）生成伪表并微调 LLM，使模型在不同领域、不同本体下仍保持零样本推断；②提供“universal domain”与“specialized domain”可调节的 trade‑off；③兼顾隐私，支持本地训练与推断。

**🔧 技术方法**

技术手段包括：LLM（如 Llama、Qwen、GPT）生成类原型；使用这些原型动态构造伪表；Prompt 设计为列‑按列呈现并单列预测；LoRA 微调；标签映射与本体对齐。

**📊 数据集**

实验使用七个公开表格标注数据集：WikiTable、SOTAB‑sch、SOTAB‑sch‑s、SOTAB‑dbp、T2D、Limaye、Efthymiou。

**📈 对比分析**

通过与多种零样本基线（TableLlama、Jellyfish、ArcheTypeZS 等）和闭源 LLM 基线（CENTS、Chorus 等）的对比，ZTab‑privacy 在无标签条件下平均提升 23.5% Micro‑F1；ZTab‑performance 在 GPT‑4o 等模型上提升 4.5%；在跨域与跨本体场景也分别提升 1.4% 与 9.5%。相较于监督参考模型，仍有 7–10% 的性能差距，但在无标注、隐私受限的场景表现优于现有零样本方法。

**⚠️ 局限性**

局限性包括：①仍低于监督模型；②生成伪表与微调需要额外训练时间；③在类数极多或类间极为相似时性能仍受限；④跨本体仍需依赖标签映射；⑤对复杂表结构（合并单元格、多级标题、嵌套 JSON）需额外预处理。

---

## 182. Social, Legal, Ethical, Empathetic and Cultural Norm Operationalisation for AI Agents

**arXiv ID:** 2603.11864 | [PDF](https://arxiv.org/pdf/2603.11864v1)

**作者:** Radu Calinescu `[一作]` (University of York), Beverley Townsend `[通讯]` (University of York)

**通讯引用:** 445 | [OpenAlex ID](https://openalex.org/A5027335222)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出了一套系统的SLEEC（社会、法律、伦理、同理心与文化）规范操作化流程，包含五个阶段：功能能力规格、规范需求引出、规则良构性检查、AI代理实现与运行时守卫、合规性验证，并以辅助护理机器人ALMI为案例进行验证。

**💡 创新点**

创新点在于将高层抽象规范映射为可验证的技术要求；引入SLEEC领域专用DSL、过程代数与逻辑两种良构性检查方法；利用RoboChart与tock‑CSP实现形式化验证；构建运行时守卫实现可持续合规；并系统性讨论开放挑战与未来研究方向。

**🔧 技术方法**

技术手段包括：过程代数tock‑CSP及FDR模型检查、基于一阶逻辑（FOL*）的LEGOS工具、SLEEC‑TK与LLM辅助调试、RoboChart建模与自动代码生成、以及运行时规则守卫框架。

**📊 数据集**

文中未使用公开数据集，而是基于ALMI任务设计的训练数据模式，包含“跌倒/未跌倒”与“是否同意” 等标签，用于强化学习和规则校验。

**📈 对比分析**

论文主要通过形式化工具进行合规性验证与冲突检测，未给出传统性能指标或与现有方法的对比；其“性能”体现在规则冲突/冗余检测的准确性和验证过程中的可追溯性。

**⚠️ 局限性**

局限性包括：抽象规范到具体需求的映射仍需人工干预；规范歧义与价值冲突难以彻底解决；实现和运行时守卫对计算资源要求高；缺乏动态适应机制；跨学科团队协作与教育仍是关键瓶颈。

---

## 183. TATIC: Task-Aware Temporal Learning for Human Intent Inference from Physical Corrections in Human-Robot Collaboration

**arXiv ID:** 2603.11077 | [PDF](https://arxiv.org/pdf/2603.11077v1)

**作者:** Jiurun Song `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5066836550)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 TATIC 框架，利用机器人关节扭矩估计与任务对齐的特征规范化，结合因果多任务 TCN，实现从简短物理纠正中同时推断任务级意图和运动级参数，并通过意图驱动的运动适配实现人机协作中的实时意图识别与路径更新。

**💡 创新点**

创新点包括：
1) 采用扭矩估计实现无需外部力传感器的短时物理纠正检测；
2) 通过任务对齐特征规范化将交互信息投影到局部参考帧，显著提升跨布局和 OOD 的泛化；
3) 将任务级意图与连续运动参数统一建模为多任务 TCN，支持离散意图分类和连续参数回归；
4) 设计意图驱动的运动适配模块（Guide、Yield、Slow、Switch、Stop），实现从语义到执行的闭环。

**🔧 技术方法**

核心技术包括：
- 关节扭矩残差与连续性滤波实现物理纠正检测；
- 任务对齐特征规范化（基于参考速度构建局部坐标系）；
- 因果多层 Temporal Convolutional Network (TCN) 与非线性编码器、分类与回归头；
- 采用不确定性加权的多任务损失与遮罩机制；
- 依据推断结果的意图驱动运动适配策略；
- 交互上下文触发与不确定性量化。

**📊 数据集**

在 7-DoF 机械臂上收集了 500 条 ID 轨迹（包含 100 条不同工作空间布局）和 250 条 OOD 重新配置轨迹，共计 750 条演示。每条轨迹的物理纠正持续约 0.53–1.98 秒，用于训练、验证和测试。

**📈 对比分析**

与仅使用运动学特征、仅使用工作空间特征、仅使用对齐特征、以及完整特征的增量 ablation 进行对比；与非时序基线（均值、线性模型、MLP）以及 GT‑Filt 与 E2E 进行回归评估。宏观 F1 为 0.904，单类最高 0.951；在 OOD‑Reconfig 上，规范化后 F1 由 0.614 提升至 0.871。回归指标（Guide 方向 cos 相似度 0.891、Magnitude RMSE 0.098、Slow/ Yield RMSE 0.061/0.086、Switch 目标 F1 0.921）均优于基线，且硬件协同装配实验验证了闭环执行效果。

**⚠️ 局限性**

局限性：
1) 采用预定义的有限意图词汇表，缺少对个体化行为的自适应；
2) 仅关注短时物理纠正，未评估长时持续交互或多模态（语言、视觉）结合；
3) 目前仅在实验室桌面拆装任务验证，尚未在更复杂、多机器人或真实工业环境中推广。

---

## 184. Large Language Models for Biomedical Article Classification

**arXiv ID:** 2603.11780 | [PDF](https://arxiv.org/pdf/2603.11780v1)

**作者:** Jakub Proboszcz `[一作]` (Warsaw University of Technology), Paweł Cichosz `[通讯]` (Warsaw University of Technology)

**通讯引用:** 688 | [OpenAlex ID](https://openalex.org/A5068140368)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统评估大型语言模型（LLM）在医学文献（Drug Class Review）分类中的可行性，探索多种提示、输出处理、few‑shot 例子数量与选择以及高级技术（链式推理、树式推理、分块），并与传统分类器（朴素贝叶斯、随机森林）和微调 Transformer（DeBERTa、SciDeBERTa‑v2）进行对比。

**💡 创新点**

① 对 LLM 输出处理方式进行细粒度对比（token 概率、JSON、数值评分），并利用 token 概率生成类别概率；② 比较多种例子选择策略及不同嵌入模型；③ 评估提示复杂度对性能的影响；④ 引入链式推理、树式推理、分块等高级技术但发现无显著提升；⑤ 在系统化实验设计下将 LLM 与传统方法并行评估，填补了以往研究缺乏深入评估的空白。

**🔧 技术方法**

使用 Llama 3.1 8B/70B、Gemma 2 9B、Mistral 7B、GPT‑4.1 mini/nano 等 LLM；嵌入模型包括 all‑MiniLM‑L6‑v2、all‑distilroberta‑v1、all‑mpnet‑base‑v2；传统机器学习方法包括朴素贝叶斯、随机森林；Transformer 微调模型为 DeBERTa 与 SciDeBERTa‑v2；评估指标为 AUPRC、MCC、F1 等，并采用 5‑折交叉验证。

**📊 数据集**

15 个 Drug Class Review（DCR）数据集，包含 PubMed 文章摘要及专家对纳入/排除的二分类标签，样本量 287–2745，正类比例 2.2%–34.5%。

**📈 对比分析**

通过宏平均 AUPRC、MCC 等指标在 5‑折 CV 中与传统方法并行比较。传统方法（随机森林、朴素贝叶斯）在大多数数据集上仍优于 LLM；但在 few‑shot + token‑prob 的最佳 LLM 配置下，AUPRC 可达 0.48–0.56，接近甚至略低于最佳传统模型（约 0.55–0.60）。

**⚠️ 局限性**

仅评估了中小规模 LLM，未探讨极少样本情形；few‑shot 例子选自完整训练集，未检验有限标签数据时的表现；高级技巧（链式推理、树式推理、分块）未见显著提升；未考虑模型个性化微调或多语言适用性；对计算成本与实用性的平衡讨论有限。

---

## 185. BLooP: Zero-Shot Abstractive Summarization using Large Language Models with Bigram Lookahead Promotion

**arXiv ID:** 2603.11415 | [PDF](https://arxiv.org/pdf/2603.11415v1)

**作者:** Varun Iyer `[一作]`, Cornelia Caragea `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无训练的复制机制BLooP，用于提升大型语言模型在无监督摘要任务中的真实性与可靠性。

**💡 创新点**

创新点在于利用输入文本中的二元组（bigram）做前瞻性提示，直接在解码阶段调整token概率，而不依赖额外的复制分布或模型参数。

**🔧 技术方法**

技术实现是：构建一个哈希表缓存所有源文档内的bigram，在每个生成步骤将符合bigram的候选token提升α的logit；实验采用Llama‑3.1‑8B、Gemma‑2‑9B、Mistral‑Nemo‑2407等decoder‑only LLM。

**📊 数据集**

使用的数据集包括新闻摘要集CNN/DM、Multi‑News、CCSum，以及科学论文摘要集SciTLDR（摘要、AIC、全文三种输入）。

**📈 对比分析**

与传统的基于encoder‑decoder的预训练模型（PEGASUS、PROM、TED等）以及无监督LLM（GPT‑3.5、Mixtral）相比，BLooP在CNN/DM、CCSum和Multi‑News上实现了ROUGE‑L提升1–3分、BARTScore提升1–4分，甚至在无监督场景下逼近GPT‑3.5（175B）的表现；在SciTLDR上亦显著提升BARTScore且保持较低的抽象性损失。

**⚠️ 局限性**

局限性包括：对长文本（如SciTLDR‑AIC/Full）的bigram缓存规模大导致效果下降；需手动挑选α超参，虽不需大规模标注但仍需少量验证；以及在保持抽象性的同时出现轻微的词汇重复与流畅度略降。

---

## 186. Counterweights and Complementarities: The Convergence of AI and Blockchain Powering a Decentralized Future

**arXiv ID:** 2603.11299 | [PDF](https://arxiv.org/pdf/2603.11299v1)

**作者:** Yibai Li `[一作]` (University of Scranton), Deng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并讨论了人工智能与区块链技术的互补性，并倡导“去中心化智能”研究方向。

**💡 创新点**

将区块链与AI结合提出去中心化智能概念，强调去中心化数据管理、隐私保护及多方协作的研究需求。

**🔧 技术方法**

利用区块链、零知识证明、去中心化身份（DID）、智能合约、NFT等技术作为支持手段。

**📊 数据集**

无数据集（为综述/评论性论文）。

**📈 对比分析**

未进行实验或性能比较，仅以理论和文献综述说明优势。

**⚠️ 局限性**

缺乏实证验证，技术可行性和治理实现仍需进一步研究；对具体实现细节、规模化部署和跨链交互等方面未给出深入讨论。

---

## 187. OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**arXiv ID:** 2603.11698 | [PDF](https://arxiv.org/pdf/2603.11698v1)

**作者:** Xianjing Han `[一作]` (National University of Singapore), Jingjing Chen `[通讯]` (Fudan University)

**通讯引用:** 5633 | [OpenAlex ID](https://openalex.org/A5100373492)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了OSCBench，一个专门评估文本到视频生成模型在对象状态变化（OSC）方面能力的基准。

**💡 创新点**

创新点在于把OSC拆分为常规、创新和组合三类场景，并通过链式思考的多模态大模型自动评估，提供细粒度的OSC评价。

**🔧 技术方法**

采用了多模态大模型（如GPT‑5.2、Qwen3‑VL‑30B）进行链式推理评估，同时对六款先进T2V模型进行人类和自动化评测。

**📊 数据集**

数据来源为HowToChange（基于HowTo100M烹饪视频），共构造了1,120条提示，覆盖108个常规、20个创新和12个组合场景。

**📈 对比分析**

在六个T2V模型中，模型在语义对齐与场景匹配上表现良好，但在OSC准确度与连贯性上普遍偏低，尤其在创新和组合场景中表现显著下降；自动评估与人工评分高度相关，验证了方法可靠性。

**⚠️ 局限性**

局限性包括仅聚焦烹饪领域，缺乏跨域验证；评估采用采样而非全量人工标注，可能忽略细节缺陷。

---

## 188. GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing

**arXiv ID:** 2603.12264 | [PDF](https://arxiv.org/pdf/2603.12264v1)

**作者:** Mingxin Liu `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5859 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评测了跨学科的图像编辑基准GRADE，用于评估模型在学科知识推理下的编辑能力。

**💡 创新点**

引入学科知识推理维度，构建多维评价体系（学科推理、视觉一致性、逻辑可读性）并提供可扩展的自动化评测管线。

**🔧 技术方法**

利用大型多模态语言模型Gemini-3-Flash进行评判，结合GPT‑5生成的结构化问题导向评估指标，采用链式推理和多步骤评估流程。

**📊 数据集**

构建520张样本，覆盖10个学科（数学、物理、化学、生物、历史、地理、运动、音乐、计算机科学、经济学），每个样本包括输入图像、编辑指令和标注结果。

**📈 对比分析**

在20个最先进模型（10开源、10闭源）上进行评估，闭源模型平均准确率约46%，开源模型最高仅2.7%，显示GRADE对学科推理的高鉴别力。

**⚠️ 局限性**

模型对隐式指令的推理能力不足，错误类型多样且缺乏对学科模板的内在理解，整体准确率低于50%，表明仍有显著提升空间。

---

## 189. abx_amr_simulator: A simulation environment for antibiotic prescribing policy optimization under antimicrobial resistance

**arXiv ID:** 2603.11369 | [PDF](https://arxiv.org/pdf/2603.11369v1)

**作者:** Joyce Lee `[一作]`, Seth Blumberg `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个基于Python的仿真框架，用于模拟抗菌药物处方与抗菌素耐药（AMR）动态，并兼容强化学习（RL）环境；

**💡 创新点**

创新点在于将处方决策抽象为患者层面的MDP/POMDP环境，配备可调的部分可观测性、leaky-balloon耐药模型、平衡临床与社区目标的奖励函数，以及可通过YAML配置和Streamlit GUI快速定制的模块化设计；

**🔧 技术方法**

采用Python实现，依托Gymnasium RL API、YAML配置文件、Streamlit GUI、Optuna等自动超参数调优工具，以及PPO等RL算法；

**📊 数据集**

使用的是由PatientGenerator生成的合成患者群体，未使用真实医疗数据集；

**📈 对比分析**

比较方法主要是训练不同RL代理（如PPO）并在不同观测噪声/偏差/延迟设置下评估奖励表现，但文中未给出具体数值性能对比；

**⚠️ 局限性**

局限性包括：仅支持单代理单地点、静态（非非平稳）动力学、主要基于合成数据，缺乏真实世界验证。

---

## 190. Slack More, Predict Better: Proximal Relaxation for Probabilistic Latent Variable Model-based Soft Sensors

**arXiv ID:** 2603.11473 | [PDF](https://arxiv.org/pdf/2603.11473v1)

**作者:** Zehua Zou `[一作]` (Hangzhou International Innovation Institute Beihang University), Zhichao Chen `[通讯]` (National Key Lab of General AI Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过引入Wasserstein距离作为proximal operator，提出了KProx算法以改进非线性概率潜变量模型（NPLVM）的后验逼近，并基于该算法构建了KProxNPLVM软传感器模型。

**💡 创新点**

创新点在于：①从逼近误差的理论视角证明了传统AVI导致的参数化限制；②将Wasserstein距离嵌入proximal梯度下降框架，得到可实现的粒子-RKHS近似方法；③通过Sinkhorn迭代实现推断网络的Wasserstein距离梯度，形成完整的训练流程。

**🔧 技术方法**

主要技术包括：变分推断、Wasserstein距离与proximal梯度优化、RKHS中梯度场逼近、粒子推理、Sinkhorn-Knopp算法、深度生成网络与编码网络的联合训练。

**📊 数据集**

使用了三大真实工业工艺数据集：离馏塔（DBC）、二氧化碳吸收塔（CAC）和水煤气变换塔（CSC），以及合成的双峰后验测试样例。

**📈 对比分析**

与多种基线NPLVM（如NPLVR、DBPSFA、GMM-VAE）及非PLVM模型（如GSTAE、DGDL、iTransformer）进行比较，KProxNPLVM在RMSE、MAE、MAPE和R²等指标上均实现了显著提升，且在统计检验中达到显著性（p<0.05）。

**⚠️ 局限性**

局限性包括：①RKHS逼近可能在高维潜变量空间下失效；②粒子数与超参调节敏感；③未探讨其他距离度量或更灵活的速度场近似策略。

---

## 191. Group Resonance Network: Learnable Prototypes and Multi-Subject Resonance for EEG Emotion Recognition

**arXiv ID:** 2603.11119 | [PDF](https://arxiv.org/pdf/2603.11119v1)

**作者:** Renwei Meng `[一作]` `[通讯]` (Anhui University), Renwei Meng (Anhui University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种 Group Resonance Network（GRN），通过融合个体 EEG 表征、可学习的组原型和多主体同步张量来提升跨受试者情绪识别性能。

**💡 创新点**

创新点包括：① 引入多主体共振张量捕捉刺激锁定的群体同步；② 采用可学习的组原型对群体结构进行建模；③ 设计共振感知融合模块，既保留个体差异又强调群体共性。

**🔧 技术方法**

使用了轻量化 CNN/Transformer 编码器、PLV/Coherence 同步计算、可学习原型机制、共振感知融合与 MLP 分类器。

**📊 数据集**

实验基于 SEED（三情绪类）和 DEAP（情绪价度/唤醒二分类）公开 EEG 数据集。

**📈 对比分析**

在 Subject-Dependent 与 Leave-One-Subject-Out（LOSO）两种协议下与 DGCNN、ST-DADGAT、FCAnet、DVIE-Net 等基线比较，GRN 在 SEED SD 取得 97.42%/SI 87.90%，在 DEAP SI 上达到 90.35%/89.40%，显著提升。

**⚠️ 局限性**

局限性：依赖预先选定的小参考集合与同步计算，计算成本较高；对不同刺激或任务的泛化能力仍待进一步验证。

---

## 192. Concurrent Prehensile and Nonprehensile Manipulation: A Practical Approach to Multi-Stage Dexterous Tasks

**arXiv ID:** 2603.11655 | [PDF](https://arxiv.org/pdf/2603.11655v1)

**作者:** Hao Jiang `[一作]` (University of Southern California), Daniel Seita `[通讯]` (University of Southern California)

**通讯引用:** 1155 | [OpenAlex ID](https://openalex.org/A5041660944)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种样本高效的现实世界多任务抓取与非抓取协同操控系统，利用对象中心化技能分解与检索-对齐-执行框架实现。

**💡 创新点**

创新点在于将演示拆分为对象中心化、时间边界明确的可重用技能，并结合不确定性感知的姿态估计，实现极低演示量下的高效学习。

**🔧 技术方法**

技术包括多视角 RGB‑D 感知（Grounding DINO + SAM2）、对象点云对齐、基于 EKF 的不确定性姿态估计、Chamfer 距离检索与 Pyroki 轨迹跟踪。

**📊 数据集**

使用自采集的遥操作演示数据集（约 40 条/任务，3–4 条/对象，共 13 个对象），并在 Allegro 与 LEAP 双手上进行 1,000+ 次真实试验。

**📈 对比分析**

与图像基 Diffusion、原始点云 Diffusion 以及对象中心化 Diffusion 基线对比，平均成功率 66%（仅 3–4 条演示），相较基线提升 2–3 倍，且只需 5 条演示即可达到相同性能。

**⚠️ 局限性**

局限性：依赖高质量遥操作演示和多视角 RGB‑D，泛化受限于训练几何；大尺寸手臂难以抓取极薄物体；缺乏触觉反馈，鲁棒性受限。

---

## 193. Multilingual Financial Fraud Detection Using Machine Learning and Transformer Models: A Bangla-English Study

**arXiv ID:** 2603.11358 | [PDF](https://arxiv.org/pdf/2603.11358v1)

**作者:** Mohammad Shihab Uddin `[一作]` (Augusta University), Arif Hassan Zidan `[通讯]` (Augusta University)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5114419924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言（孟加拉语–英语）金融诈骗检测，比较经典机器学习与Transformer模型的表现；

**💡 创新点**

首次系统关注低资源孟加拉语的多语言场景，揭示传统TF‑IDF+线性模型在此任务上可优于大模型；

**🔧 技术方法**

使用TF‑IDF+Logistic Regression、Linear SVM、Ensemble等经典方法和多语言Transformer（如mBERT/XLM‑R）进行分类；

**📊 数据集**

采用由合法与欺诈金融消息组成的混合孟加拉语–英语数据集，包含代码混合、URL、电话号码等特征；

**📈 对比分析**

采用5折分层交叉验证，评估Accuracy、宏F1和PR‑AUC；Linear SVM得到91.59%准确率、91.30% F1，略优于Transformer的89.49%准确率、88.88% F1，但Transformer在诈骗召回率高而误报率大；

**⚠️ 局限性**

数据量有限、语言多样性与代码混合导致特征表征受限；Transformer对孟加拉语财务词汇曝光不足；缺乏大规模标注数据与细粒度错误分析。

---

## 194. Multi-Agent Collaboration for Automated Design Exploration on High Performance Computing Systems

**arXiv ID:** 2603.11515 | [PDF](https://arxiv.org/pdf/2603.11515v1)

**作者:** Harshitha Menon `[一作]` (Lawrence Livermore National Laboratory), Jonathan L. Belof `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 1823 | [OpenAlex ID](https://openalex.org/A5069056278)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了MADA多智能体框架，自动完成高性能计算环境下的网格生成、仿真提交、结果分析和逆向设计，最终迭代优化Richtmyer–Meshkov不稳定性抑制几何形状。

**💡 创新点**

通过LLM驱动的多智能体协作与Model Context Protocol统一工具接口，实现从仿真到设计的闭环自动化，并首次将代理模型与高保真仿真联动用于设计空间探索。

**🔧 技术方法**

采用大语言模型（OpenAI o3）、多智能体架构、MCP工具调用、Flux工作负载管理器、Cubit网格生成器、Laghos流体动力学仿真以及机器学习代理模型。

**📊 数据集**

使用RMI抑制实验的设计参数及Laghos仿真输出，并基于预训练的RMI代理模型数据进行评估。

**📈 对比分析**

在Tuolumne HPC上对比全仿真循环与代理模型加速的设计探索，展示MADA在迭代中显著提高RMI抑制效果且仅需极少人工干预，代理模型探索速度快数百倍。

**⚠️ 局限性**

仅采用采样搜索，无法保证全局最优；LLM推理结果可变；需要为每种仿真工具编写MCP服务器；高保真仿真仍消耗大量计算资源。

---

## 195. Cross-Context Review: Improving LLM Output Quality by Separating Production and Review Sessions

**arXiv ID:** 2603.12123 | [PDF](https://arxiv.org/pdf/2603.12123v1)

**作者:** Tae-Eun Song `[一作]` `[通讯]` (Daejeon Jungang Cheonggua Co. Ltd.), Tae-Eun Song (Daejeon Jungang Cheonggua Co. Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了跨上下文审阅（Cross-Context Review, CCR）方法，即在新的会话中让LLM只收到生成结果本身，评估并发现错误。

**💡 创新点**

创新点在于将会话上下文分离作为关键干预手段，证明仅通过断开生产与评审上下文即可显著提升错误检测，而不需要多模型协作或复杂的提示工程。

**🔧 技术方法**

技术细节：使用 Claude Opus 4.6 作为评审模型；设计四种实验条件（同会话自评 SR、重复自评 SR2、上下文感知子代理 SA、CCR）；采用自注入错误的手工生成 artifact 并在各条件下执行评审，评估指标为 Precision、Recall、F1。

**📊 数据集**

数据集：30 个人工生成的 artifact，包含代码、技术文档、演示脚本，每个 artifact 注入 5 个错误（共 150 个错误），按错误类型与严重程度划分。

**📈 对比分析**

比较方法：对每个 artifact 在四种条件下分别进行评审，统计 TP/FP/FN 并计算 F1；通过配对 t 检验和 Cohen’s d 评估统计显著性。结果显示 CCR 的 F1 为 28.6%，显著高于 SR（24.6%）、SR2（21.7%）和 SA（23.8%），p < 0.01；优势最明显在关键错误和代码类 artifact 上。

**⚠️ 局限性**

局限性：实验仅在单一模型（Claude Opus 4.6）上进行，缺乏跨模型验证；错误注入人为，可能不完全代表真实场景；整体 F1 仍较低；未与人工专家评审做基准对照；存在语言偏差（artifact 与评审语言不一致）以及仅在软件工程教程领域进行，未验证对其他领域的通用性。

---

## 196. LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories

**arXiv ID:** 2603.11987 | [PDF](https://arxiv.org/pdf/2603.11987v1)

**作者:** Qianpu Sun `[一作]` (Tsinghua University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 10809 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了LabShield基准，专门评估多模态大型语言模型在科学实验室中的安全意识与危险识别、推理与规划能力；

**💡 创新点**

创新点在于：①基于OSHA与GHS标准制定了四层操作级与四层安全级的分层安全词典；②采用多视角同步 RGB‑D 观测，克服单视角遮挡；③引入 Perception–Reasoning–Planning（PRP）三阶段评估框架；④双轨道（MCQ 与半开放 QA）评测，兼顾抽象推理与现实场景落地；

**🔧 技术方法**

主要技术包括：多模态大型语言模型（GPT‑4o、Gemini‑3、Claude‑4、Qwen‑VL、InternVL 等）、视觉‑语言‑动作模型、LLM‑as‑Judge 评估策略、数据增强与多视角融合算法；

**📊 数据集**

使用了自建 LabShield 数据集：164 个操作任务，涵盖三类实验区（工作台、废气罩、洗手池），每任务配备四视角 RGB‑D（头、躯干、双手）和 1,439 对 VQA；

**📈 对比分析**

在 33 个模型上进行零样本评测：最高 MCQ 准确率约 78%，但半开放 QA 的安全分数仅 ~50%，人类基线为 92%。结果表明 MCQ 评分与实际安全表现不匹配，显著缺口在高风险场景的危险识别与计划拒绝；

**⚠️ 局限性**

局限性包括：①现有模型即便是专为机器人设计的多模态模型，也未能显著提升安全性能；②对透明物体的感知不足导致关键危险被忽视；③安全评估仍依赖 LLM‑Judge，存在评判不稳定性；④数据集规模相对有限，难以覆盖所有实验室极端风险。

---

## 197. Paper Title: LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments

**arXiv ID:** 2603.12071 | [PDF](https://arxiv.org/pdf/2603.12071v1)

**作者:** Zhaoyang Jiang `[一作]` (University of Glasgow), Honghan Wu `[通讯]` (University of Glasgow)

**通讯引用:** 3199 | [OpenAlex ID](https://openalex.org/A5043821806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了 LoV3D，一种端到端的 3D 视觉‑语言模型管线，能够读取纵向 T1 加权脑 MRI，生成区域级解剖评估、纵向对比，并输出诊断标签及可验证的诊断摘要。

**💡 创新点**

通过可验证的 JSON 输出格式与临床加权 Verifier 自动评分相结合，实现无人工标注的偏好优化，闭环训练模型的认知预后推理。

**🔧 技术方法**

使用 MONAI ResNet‑50 3D 编码器、可学习投影器、Qwen‑2.5‑14B LLM + LoRA、Normative Z‑score 参考、Verifier‑guided DPO 训练、结构化输出与多任务学习。

**📊 数据集**

在 ADNI 8,114 张 T1 脑 MRI 上训练、验证、测试，并在 MIRIAD 与 AIBL 两个独立站点进行零样本迁移评估。

**📈 对比分析**

与传统 ResNet‑50 分类器、公开二分类与三分类基准以及 RadFM/M3D‑LaMed 等通用 3D VLM 进行比较；LoV3D 在 ADNI 上三类诊断准确率 93.7%、区域准确率 82.6%，在 MIRIAD 和 AIBL 的零样本测试分别达到 95.4% 和 82.9%。

**⚠️ 局限性**

依赖 FreeSurfer 生成的参考量化，限制为 T1 加权 MRI，未区分不同亚型 MCI，且对多模态或其他影像类型的适用性尚待验证。

---

## 198. Unclonable Encryption in the Haar Random Oracle Model

**arXiv ID:** 2603.11437 | [PDF](https://arxiv.org/pdf/2603.11437v1)

**作者:** James Bartusek `[一作]` (Columbia University), Eli Goldin `[通讯]` (New York University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5019414571)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

在 Haar 随机算子模型下构造了可重用的不可克隆加密（UE）方案，并给出了从 QROM 到 Haar 随机算子模型的编译器。

**💡 创新点**

创新点在于首次证明在不需要一把一向函数的前提下（微型加密环境）即可实现可重用 UE，并提出了新的“单元重程化引理”，该引理可独立用于其他量子算子重编程问题。

**🔧 技术方法**

主要技术包括：路径记录框架（Path Recording Framework）用于高效模拟 Haar 随机算子；单元重程化引理证明对不同子空间的 Haar 分布不可区分；以及构造一个基于 X^k 与 Haar 随机算子的混合加密形式，并设计相应的安全游戏与多阶段混合证明。

**📊 数据集**

该工作完全是理论性，无需实际数据集；所有安全分析均在抽象的随机算子模型下进行。

**📈 对比分析**

与以往仅在 QROM 中实现的单次 UE 或搜索安全 UE 方案相比，该方案支持多次加密且不依赖一向函数；安全性通过标准的不可克隆可区分性定义证明，复杂度保持多项式。相比之下，传统方案在可重用性和对硬件假设上更为苛刻。

**⚠️ 局限性**

局限性包括：依赖于 Haar 随机算子模型，理论构造尚未提供可实现的哈希函数或量子电路实例；编译器实现对查询量的隐式多项式上限；以及在实际实现时可能面临的量子硬件误差和资源消耗问题。

---

## 199. Fluid Reconfigurable Intelligent Surface Enabling Index Modulation

**arXiv ID:** 2603.11714 | [PDF](https://arxiv.org/pdf/2603.11714v1)

**作者:** Peng Zhang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**通讯引用:** 7809 | [OpenAlex ID](https://openalex.org/A5027587940)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出基于流体可重构智能表面（FRIS）的指数调制框架（FRIS‑RSM与FRIS‑RSSK），通过同时实现元素位置与相位的可重构，实现将信号聚焦至接收天线索引以携带信息；

**💡 创新点**

创新点包括：1）将流体天线与RIS结合，提供更丰富的空间自由度；2）设计连续与量化相位两种FRIS配置方案；3）提出两阶段低复杂度列表检测器；4）在双Rayleigh级联衰落下推导可计算的截断后选择统计量与MGF‑基误码分析；5）证明在单位对角协相关矩阵下可得到误码下界；

**🔧 技术方法**

使用的技术主要是：流体天线技术、RIS相位编程、指数调制、双Rayleigh衰落模型、截断（阈值）近似、矩差匹配、中心极限定理、MGF（矩生成函数）误码分析、Craig表示、两指数逼近、两阶段列表检测算法；

**📊 数据集**

本文未使用公开数据集，全部采用仿真验证，基于双Rayleigh级联衰落、理想CSI、不同FRIS候选元件数、激活比率、相位量化位数等参数进行Monte‑Carlo仿真；

**📈 对比分析**

通过与传统RIS‑RSSK/RSM基准（即所有反射元件均激活、无位置重构）进行比对，仿真结果表明：在相同物理孔径和激活元件数下，增加候选元件数量可实现4–7 dB的SNR提升；相量量化仅需3位即可逼近连续相位性能；两阶段列表检测器在保持接近ML性能的同时将复杂度降低至O(LM)，并在L=5时仅损失约1 dB；

**⚠️ 局限性**

局限性：1）误码分析基于单位对角协相关矩阵的下界，未考虑真实高相关场景；2）采用截断阈值近似与CLT近似，可能在K_sel较小或极端SNR下误差增大；3）假设理想CSI与无硬件非理想；4）仅研究SIMO单输入多输出，未扩展至多输入多输出；5）模型专注于双Rayleigh衰落，未涵盖其他通道模型；

---

## 200. Hoi3DGen: Generating High-Quality Human-Object-Interactions in 3D

**arXiv ID:** 2603.12126 | [PDF](https://arxiv.org/pdf/2603.12126v1)

**作者:** Agniv Sharma `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**通讯引用:** 14346 | [OpenAlex ID](https://openalex.org/A5076908763)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种从详细文本描述中生成高质量、可分割且可动画的三维人-物交互模型，能够准确体现接触关系并与SMPL模型对齐。

**💡 创新点**

创新点包括：1）基于多模态大语言模型的自动交互描述生成流水线，分解为外观、动作与接触子任务；2）视角条件的文本到图像生成，显著提升交互图像的接触准确性；3）将高质量2D图像上采样到3D，并通过视频分割与SMPL注册实现语义化交互网格；4）仅需约400条高质量交互样本即可显著提升模型性能。

**🔧 技术方法**

使用的技术包括：InternVL 和 LLaMA 3.1 进行文本标注；SANA（Latent Diffusion）进行视角条件文本到图像生成；Hunyuan3D 进行2D到3D的网格重建；Grounded-Segment Anything 2（GSAM2）进行视频分割；CameraHMR + Chamfer 优化实现SMPL注册；再利用Flux进行图像重纹理。

**📊 数据集**

主要数据集为 ProciGen，用于生成交互网格并通过自动标注生成 750k 条文本对；在此基础上筛选 400 条高质量样本用于模型微调；此外使用公开的 SMPL、Objaverse 等数据进行基准对比。

**📈 对比分析**

与现有通用 3D 生成模型 TRELLIS 以及交互生成模型 InterFusion 进行对比。GPT 评分提升至 0.81（相较 0.15/0.04），接触准确率达到 90%（相较 45%/N/A），在 40 条用户测试案例中文本一致性 91.09% 与 3D 质量 85.56%，均远超对手。CLIP 分数虽略低，但其对细粒度交互评价不敏感。

**⚠️ 局限性**

局限性：对极其复杂或模糊的姿态描述的把握不足，导致生成姿态不够精确；当前方法对文本中细节描述的理解主要集中在接触部位，未充分解决多姿态约束。

---

## 201. CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**arXiv ID:** 2603.12008 | [PDF](https://arxiv.org/pdf/2603.12008v1)

**作者:** Ziqi Ye `[一作]` (Fudan University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17003 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发并评估了首个十亿级别的SAR视觉基础模型CrossEarth‑SAR，并构建了200K规模的语义分割数据集与22个跨域泛化基准，探究其在多种域差异下的性能；

**💡 创新点**

创新点在于引入物理引导的稀疏Mixture‑of‑Experts架构，将SAR特有的物理描述符嵌入路由器，实现跨域专家激活，并配合大规模连续预训练与参数高效微调方法；

**🔧 技术方法**

使用DINOv2 backbone、Mask2Former解码器、physics‑guided sparse MoE、连续预训练（CPT）、参数高效微调（PEFT）、负载均衡损失以及三种SAR物理描述符（H_DE、ENL、R_LR）；

**📊 数据集**

使用CrossEarth‑SAR‑200K（200k监督/伪监督SAR图像）以及基于6个公开SAR语义分割数据集（AIR‑PolSAR‑Seg‑2.0、DDHR‑SK、FUSAR‑Map、OpenEarthMap‑SAR、SARBuD、WHU‑OPT‑SAR）的22个跨域子基准；

**📈 对比分析**

与DINOv2、DINOv3、SARATR‑X等基线在22个跨域基准上对比，CrossEarth‑SAR在大多数设置下实现SOTA，mIoU提升2-15个百分点，尤其在区域、极化、频段、平台混合差异上表现突出；

**⚠️ 局限性**

局限性包括：训练需要大规模GPU资源和高计算成本；物理描述符覆盖范围有限，可能在极端极化或频段下效果不佳；目前仅验证语义分割任务，尚未扩展到目标识别、变化检测等下游应用。

---

## 202. Towards heterogeneous parallelism for SPHinXsys

**arXiv ID:** 2603.11868 | [PDF](https://arxiv.org/pdf/2603.11868v1)

**作者:** Xiangyu Hu `[一作]` (Technical University of Munich), Alberto Guarnieri `[通讯]` (Technical University of Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究开发了弱压缩 Smoothed Particle Hydrodynamics (WCSPH) 方法，并将其与两方程 Reynolds-Averaged Navier–Stokes (RANS) 模型结合，用于壁面湍流（含流动分离）的数值模拟；同时实现了 SPHinXsys 库的 SYCL 并行化，支持 CPU 与 GPU 的异构计算。

**💡 创新点**

创新点在于通过改进主流（Adaptive Riemann‑eddy Dissipation 与去噪传输速度）和壁面处理（基于粒子的壁面模型、加权壁面补偿、BOT 等），首次在 SPH‑RANS 中实现了速度和湍流动能的收敛，并提出了统一的 SYCL 编程模式，既保持了开源开发优势，又实现了高效的 GPU 并行。

**🔧 技术方法**

所用技术包括 WCSPH、RANS、ARD、去噪传输速度、粒子壁面模型、Level‑Set BOT、SYCL/DPC++、Unified Shared Memory (USM)、原子操作构建 cell‑linked‑list、direct search 与 GPU 加速的粒子排序。

**📊 数据集**

验证使用了多种壁面湍流基准案例（直通道、轻曲率通道、强曲率通道、半收敛‑扩散通道、鱼通道）以及大坝破裂流场，用于性能和算法正确性的测试。

**📈 对比分析**

通过与 DualSPHysics（CUDA 并行实现）在相同 NVIDIA RTX 2080Ti GPU 上的比较，SPHinXsys 的运行时间约为 Dual 的一半，GPIPS（每秒粒子交互次数）约为其两倍；单精度 GPU 版本相较于 CPU 版本提升了约 27 倍。

**⚠️ 局限性**

局限性包括 GPU 上的内存限制导致需使用 direct search 而非完整邻居列表、SYCL 对某些 C++ 特性的支持不完全、以及对极复杂几何体的粒子排序与壁面处理仍需进一步优化。

---

## 203. HiSync: Spatio-Temporally Aligning Hand Motion from Wearable IMU and On-Robot Camera for Command Source Identification in Long-Range HRI

**arXiv ID:** 2603.11809 | [PDF](https://arxiv.org/pdf/2603.11809v1)

**作者:** Chengwen Zhang `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**通讯引用:** 5363 | [OpenAlex ID](https://openalex.org/A5057896400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了长距离多用户人机交互中的命令源识别（CSI），提出 HiSync 框架通过机器人摄像机光流与佩戴 IMU 的频域特征实现命令来源绑定。

**💡 创新点**

创新点在于：①在 34 m 以内无需语音、标记或用户身份登记即可完成 CSI；②使用频域特征和跨模态注意力，使光学与惯性信号对齐且对时间同步误差鲁棒；③结合多尺度窗口融合提升不同距离下的识别精度。

**🔧 技术方法**

技术手段包括：VideoFlow 光流提取、IMU 线性加速度与光流的频谱分析、质量感知特征调制（FiLM）、IMU 绑定交叉注意力、Scale‑Aware 多窗口融合、InfoNCE 对比学习等。

**📊 数据集**

使用自建的 9 465 条手势多模态数据集：38 位参与者、3 种视角（上、眼、下）、7 种手势、3–34 m 的距离范围。

**📈 对比分析**

与 Vision‑Only、URGR、VIPL 等基线对比，HiSync 在 3–34 m 区间平均 CSI 准确率 96.6%，在 34 m 仍达 94.3%，比前沿方法提升约 48%；实测机器人部署下准确率保持在 90% 以上。

**⚠️ 局限性**

局限性包括：测试环境人群密度低，未验证多 IMU 并发情形；机器人必须静止才能识别，对机器人运动与低帧率较敏感；仅支持预定义手势，未覆盖自然小幅手势与更大规模用户群。

---

## 204. Bounding the Fragmentation of B-Trees Subject to Batched Insertions

**arXiv ID:** 2603.12211 | [PDF](https://arxiv.org/pdf/2603.12211v1)

**作者:** Michael A. Bender `[一作]` (Stony Brook University and RelationalAI), Nicole Wein `[通讯]` (University of Michigan)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5091698334)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究B树在批量随机插入下的内部碎片化问题，提出并分析多种分裂策略（偶数分裂、延迟偶数分裂、非均匀分裂），并给出不同批量大小下的空间利用率理论下界与实验验证。

**💡 创新点**

将Yao在单键随机插入下的分析推广到批量插入；构造可解析的递推矩阵并利用其特征值证明期望块大小收敛；在不同批量区间内设计针对性算法，使空间利用率保持在0.5以上；给出递推解析闭式（含调和数），填补了批量插入理论空白。

**🔧 技术方法**

采用随机过程建模、矩阵递推与特征值分析、线性代数（Jordan标准形）以及调和级数求和等技术；对偶数分裂、延迟分裂与非均匀分裂分别构造转移矩阵并求解其稳态分布；通过模拟验证理论结果。

**📊 数据集**

在B=240的模拟实验中，使用200,000次插入（r∈[1,B]或r∈[B,5B]），测量平均块满度；实验数据与理论曲线对齐。

**📈 对比分析**

对比三种分裂策略的平均满度随r/B变化的曲线。偶数分裂在小r时表现良好（≥0.6），但在r≈B/2时降至0.5；延迟偶数分裂在大r时始终保持高满度（≥0.66），并在中间区间逐渐趋近1；非均匀分裂在r≈B/2时可恢复到≥0.6。整体上，针对不同r区间的算法能保持空间利用率显著高于50%。

**⚠️ 局限性**

仍存在理论与实验之间的细微差距，尤其是对小批量r>1时偶数分裂的精确下界不完整；延迟偶数分裂在某些r区间的闭式仍缺失；算法需知晓批量大小r，无法适用于未知或可变r；对于r可变化的批量插入以及非均匀批量的更通用策略仍是开放问题。

---

## 205. On the PLS-Completeness of $k$-Opt Local Search for the Traveling Salesman Problem

**arXiv ID:** 2603.11270 | [PDF](https://arxiv.org/pdf/2603.11270v1)

**作者:** Sophia Heimann `[一作]` (University of Bonn), Stefan Hougardy `[通讯]` (University of Bonn)

**通讯引用:** 1488 | [OpenAlex ID](https://openalex.org/A5085228445)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对旅行商问题的局部搜索算法，该算法从初始路径开始，迭代地用相同数量的边替换最多k条边，以获得更好的路径。

**💡 创新点**

首次提供了k≥15时旅行商问题的PLS-完全性的严格证明，并显著降低了k的值，解决了Monien等人提出的一个开放问题。

**🔧 技术方法**

使用了局部搜索算法和从有界度最大割问题到旅行商问题的新减缩方法。

**📊 数据集**

构造了一个新的图G，基于最大割问题的实例，并通过赋予边权重来形成旅行商问题的实例。

**📈 对比分析**

与Krentel的早期结果相比，当前论文的结果在k≥15时更强，且提供了更紧凑的减缩，证明了在k≥15时的PLS-完全性。

**⚠️ 局限性**

当前的证明方法可能无法推广到k<15的情况，且尚未解决TSP在某些常数k下的紧PLS-完全性问题。

---

## 206. A Quantitative Characterization of Forgetting in Post-Training

**arXiv ID:** 2603.12163 | [PDF](https://arxiv.org/pdf/2603.12163v1)

**作者:** Krishnakumar Balasubramanian `[一作]` (University of California), Shiva Prasad Kasiviswanathan `[通讯]` (Amazon)

**通讯引用:** 3874 | [OpenAlex ID](https://openalex.org/A5036801391)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文以两模式混合模型为抽象，理论分析持续后训练中生成模型的遗忘机制，并定量评估前向KL、反向KL以及重放对质量遗忘和组件漂移的影响。

**💡 创新点**

创新点在于建立了基于混合模型的理论框架，明确区分质量遗忘与组件漂移，证明前向KL会导致旧模式权重坍塌，反向KL通过重叠门控实现漂移抑制，并对SDFT、TTT‑Discover、OAPL等近策略方法进行定量比较。

**🔧 技术方法**

采用KL散度分析、Bhattacharyya重叠界、PL条件、梯度流与指数收敛证明以及重放与重要性加权等数学工具。

**📊 数据集**

该工作完全基于理论与仿真，不依赖公开数据集，仅在等协方差高斯混合模型上进行验证。

**📈 对比分析**

通过在混合模型下比较前向KL、反向KL与重放的优化动力学，发现前向KL在无重放时会完全遗忘旧模式；反向KL在分离度较大时保持旧模式且收敛指数级；加入重放的前向KL可部分避免遗忘。

**⚠️ 局限性**

局限性包括：仅适用于两模式、等协方差高斯模型，无法直接推广至高维、多模态生成模型；理论假设如模式可辨别、无模型误差等；缺乏在真实数据集上的实验验证。

---

## 207. WebWeaver: Breaking Topology Confidentiality in LLM Multi-Agent Systems with Stealthy Context-Based Inference

**arXiv ID:** 2603.11132 | [PDF](https://arxiv.org/pdf/2603.11132v1)

**作者:** Zixun Xiong `[一作]` (Stevens Institute of Technology), Hao Wang `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 41668 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可在单一代理被攻击时推断LLM多代理系统通信拓扑的攻击框架WebWeaver。

**💡 创新点**

创新点在于：①仅基于代理上下文而非身份标识进行拓扑推断；②结合隐蔽递归 jailbreak 与无 jailbreak 的扩散模型，具备更高的隐蔽性和鲁棒性；③设计掩蔽策略在扩散推理过程中保持已知拓扑一致性，并给出理论正确性保证。

**🔧 技术方法**

使用的技术包括：对话数据收集与发送方预测模型（基于语义指纹识别）；递归 jailbreak 机制；DDPM（去噪扩散概率模型）与掩蔽扩散完成模块；梯度优化的 jailbreak 生成。

**📊 数据集**

评估数据集涵盖 CSQA、GSM8k、Fact 与 Bias 四个任务场景的对话数据。

**📈 对比分析**

与仅依赖身份查询的 IP Leakage 方法及最近的优化式图完成方法 CZRL 对比，WebWeaver 在主动防御下平均提升约 60% 的推断准确率，且在 5-20 个代理规模下保持高 F1，在线开销低。

**⚠️ 局限性**

局限性：实验仅在受控实验环境下完成，未在真实的在线学术协作平台上验证；缺乏对不同代理模型异质性与更大规模网络的深入评估。

---

## 208. Procedural Fairness via Group Counterfactual Explanation

**arXiv ID:** 2603.11140 | [PDF](https://arxiv.org/pdf/2603.11140v1)

**作者:** Gideon Popoola `[一作]` (Montana State University), John Sheppard `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于组对比的集成梯度正则化框架 GCIG，要求模型在不同保护组下对同一实例给出的解释保持一致，从而实现程序性公平性。

**💡 创新点**

创新点在于将程序性公平性定义为在真标签条件下的组对比解释不变性，并将其作为可微正则项融入训练；同时与等化机会共同优化，兼顾预测、结果公平与解释公平。

**🔧 技术方法**

使用技术包括：集成梯度（Integrated Gradients）+ 组条件基线、指数移动平均（EMA）平滑基线、可微正则化（λ_ig）以及交叉熵损失和等化机会正则（λ_fair）等。

**📊 数据集**

实验基准为四个公开数据集：Adult Income、German Credit、COMPAS、Bank Marketing。

**📈 对比分析**

与无公平约束、预处理、后处理、对抗、红利修复等六类现有公平方法对比，GCIG 在解释差异（GCIG）上始终取得冠军；预测准确率（F1）和等化机会间距（EO gap）保持与最强对手相当，显示兼顾性能与公平。

**⚠️ 局限性**

局限性包括：仅处理二值保护属性和表格数据；对多组/交叉属性的推广尚待研究；依赖集成梯度导致额外训练开销，且对不同模型结构的适用性需进一步验证。

---

## 209. Noise-aware few-shot learning through bi-directional multi-view prompt alignment

**arXiv ID:** 2603.11617 | [PDF](https://arxiv.org/pdf/2603.11617v1)

**作者:** Lu Niu `[一作]` (Southeast University), Cheng Xue `[通讯]` (Southeast University)

**通讯引用:** 1315 | [OpenAlex ID](https://openalex.org/A5101889471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 NA-MVP 框架，通过双向多视角提示与不平衡最优传输实现噪声可知的少样本视觉-语言学习；

**💡 创新点**

创新点在于将区域感知的多视角提示对齐与基于提示的选择性 OT 标签修正结合，能够分离干净与噪声语义，克服单视角提示和全局标签修正的缺陷；

**🔧 技术方法**

采用了双向多视角提示、Unbalanced Optimal Transport、经典 OT、CLIP 文本/图像编码、ITBP 损失以及 GCE 损失等技术；

**📊 数据集**

实验使用 Caltech101、DTD、Flowers102、OxfordPets、UCF101（合成噪声）和 Food101N（真实噪声）等数据集；

**📈 对比分析**

与 CoOp、CoOp+GCE、JoAPR、NLPrompt 等基线相比，在所有噪声水平和数据集上均实现了最高准确率，尤其在高噪声场景下优势尤为明显；

**⚠️ 局限性**

局限性包括对提示数量与超参数较为敏感，过多提示可能导致冗余；对计算成本有一定开销；在极低样本或极高噪声条件下仍有提升空间，并缺乏深入理论解析。

---

## 210. Stage-Adaptive Reliability Modeling for Continuous Valence-Arousal Estimation

**arXiv ID:** 2603.11468 | [PDF](https://arxiv.org/pdf/2603.11468v1)

**作者:** Yubeen Lee `[一作]` (Sungkyunkwan University), Eunil Park `[通讯]` (Sungkyunkwan University)

**通讯引用:** 7789 | [OpenAlex ID](https://openalex.org/A5047279790)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种阶段自适应可靠性建模框架 SAGE，用于连续 valence‑arousal 估计，动态校准音频与视觉模态的可靠性并融合。

**💡 创新点**

创新点在于将模态可靠性估计与融合分离，使用时间依赖的可靠性权重对模态进行动态重加权，并通过可靠性引导的 Transformer 提升跨模态交互。

**🔧 技术方法**

技术包括 ResNet‑50 和 WavLM 预训练编码器、TCN 语义编码、可靠性引导融合 (RGF)、自注意力 Transformer、CCC 损失优化。

**📊 数据集**

使用 Aff‑Wild2 公开数据集，构建 300 帧长段，面向 10th ABAW 竞赛的 valence‑arousal 任务。

**📈 对比分析**

与多种最新多模态基线对比，验证集上 valence CCC 0.509、arousal 0.674，平均 0.591；测试集平均 CCC 0.58，整体表现与现有顶级方法相近，且不依赖额外数据集或集成。

**⚠️ 局限性**

局限在于对极端噪声或模态缺失的鲁棒性仍有限，缺乏更深层次的跨模态互补建模与外部知识整合。

---

## 211. EnTransformer: A Deep Generative Transformer for Multivariate Probabilistic Forecasting

**arXiv ID:** 2603.11909 | [PDF](https://arxiv.org/pdf/2603.11909v1)

**作者:** Rajdeep Pathak `[一作]` (SAFIR, Sorbonne University Abu Dhabi), Tanujit Chakraborty `[通讯]` (Sorbonne University)

**通讯引用:** 1220 | [OpenAlex ID](https://openalex.org/A5012926469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

提出了EnTransformer，结合engression与Transformer生成多元时间序列的概率预测。

**💡 创新点**

创新点在于通过在Transformer中注入预加噪声并使用能量分数训练，直接学习无参数化的条件预测分布，同时保留长程依赖与多元交互。

**🔧 技术方法**

使用了engression噪声注入、能量分数（Energy Score）损失、Transformer编码-解码结构、批量复制实现多样化预测等技术。

**📊 数据集**

在Solar、Electricity、Traffic、KDD‑cup、Taxi、Wikipedia六个公共多元时序数据集上进行实验。

**📈 对比分析**

与Vec‑LSTM、GP‑Scaling、GP‑Copula、LSTM‑MAF、Transformer‑MAF、TimeGrad、TACTiS、MG‑Input等基准模型在CRPS_sum上对比，EnTransformer在Solar、Electricity、KDD‑cup、Taxi等数据集上取得最优或接近最优的性能，整体稳定且训练速度更快。

**⚠️ 局限性**

局限在于需手动调节噪声标准差和样本数M，最佳M对不同数据不一致；对空间或图结构的依赖建模有限；在部分数据如Traffic、Solar的校准仍略逊色。

---

## 212. DriveXQA: Cross-modal Visual Question Answering for Adverse Driving Scene Understanding

**arXiv ID:** 2603.11380 | [PDF](https://arxiv.org/pdf/2603.11380v1)

**作者:** Mingzhe Tao `[一作]`, Rainer Stiefelhagen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了DriveXQA多模态VQA数据集，并提出MVX-LLM框架用于在多天气与传感器失效场景下的自动驾驶场景理解。

**💡 创新点**

创新点包括：① DriveXQA首次系统覆盖四种视觉模态、五种传感器失效与五种天气的VQA数据集；② MVX-LLM引入Dual Cross-Attention（空间+通道交叉注意力）实现多模态高效融合；③ 采用QAttn token聚合进一步提升性能。

**🔧 技术方法**

使用了CLIP‑based ViT编码器、PointNet++ LiDAR编码器、Dual Cross-Attention机制、Query Attention聚合、LoRA参数高效微调以及GPT-4o-mini自动评估等技术。

**📊 数据集**

利用DriveXQA数据集（7,885帧、102,505 QA对，覆盖多视角、多天气及多传感器失效）进行实验，并与NuScenes‑QA、DriveLM等公开数据集对比。

**📈 对比分析**

通过与多种投影器（GAP+Honeybee、GAP+ParGo、Self‑query）和多种token融合方法（GAP、QAttn(Spectral)等）在DriveXQA上进行定量评估，使用GPTScore、BLEU‑4、ROUGE‑L、METEOR等指标。MVX-LLM在所有条件下获得最高GPTScore 55.5，显著优于最强基线48.2，尤其在雾天和传感器失效场景提升突出。

**⚠️ 局限性**

局限性包括：① 依赖仿真数据，实测环境的模拟与现实差距；② 仅进行单帧静态理解，缺乏时序推理；③ 仅覆盖四种视觉模态，未来需加入更多传感器并探索轻量化部署。

---

## 213. Intelligent 6G Edge Connectivity: A Knowledge Driven Optimization Framework for Small Cell Selection

**arXiv ID:** 2603.12086 | [PDF](https://arxiv.org/pdf/2603.12086v1)

**作者:** Tuğçe Bilen `[一作]` (Istanbul Technical University), Ian F. Akyildiz `[通讯]` (Truva Inc.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于知识定义网络（KDN）的密集6G小基站用户关联框架，该框架通过队列感知指标、拉格朗日优化和轻量级学习向量量化（LVQ）实现自适应、QoS驱动的用户-小基站匹配。

**💡 创新点**

创新点包括：①将流量负载、排队延迟、包服务降解及能耗等多维度服务指标统一进关联决策；②将知识层、控制层和数据层闭环耦合，形成KDN闭环控制；③利用拉格朗日松弛求解全局最优关联，再用LVQ近似以实现实时推理；④在密集小基站场景下首次实现优化驱动、学习加速的协同关联方法。

**🔧 技术方法**

采用的技术有：知识定义网络架构、队列建模（G/G/1+Kingman公式）、拉格朗日松弛优化、Learning Vector Quantization（LVQ）学习器、NS-3仿真平台、比例公平调度器、路径损耗/阴影/瑞利衰落信道模型。

**📊 数据集**

使用的“数据集”为NS-3仿真生成的合成流量与拓扑数据，涵盖不同用户数、基站密度、移动速度、包大小及流量速率等多种场景。

**📈 对比分析**

与基准方法（最大RSRP、负载感知启发式、贪婪服务感知）比较，评估指标为平均端到端延迟与丢包率。实验结果显示在高流量、高移动率及高密度情况下，所提框架平均延迟降低30–45%，丢包率下降超过35%，并且95%以上的用户满足50 ms延迟和2%丢包的QoS阈值。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，缺乏真实网络流量和硬件异质性；能耗模型仅为相对指标，未细化硬件功耗与睡眠模式；架构假设单一管理域，未考虑多运营商或分布式KDN；学习模型缺乏对概念漂移的在线自适应机制。

---

## 214. On the Role of Reversible Instance Normalization

**arXiv ID:** 2603.11869 | [PDF](https://arxiv.org/pdf/2603.11869v1)

**作者:** Gaspard Berthelier `[一作]` (EDF R&D and Inria), Giovanni Neglia `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了时间序列预测中归一化方法，尤其是Reversible Instance Normalization（RevIN），通过消融实验评估其各组成部分的必要性。

**💡 创新点**

指出RevIN的可学习仿射层并不必要，归一化后在规范化空间进行反向传播更有效，同时强调RevIN并不能解决所有分布偏移，尤其是条件分布偏移。

**🔧 技术方法**

使用标准归一化、Instance Normalization、RevIN等归一化技术，搭配Transformer基模型PatchTST进行实验。

**📊 数据集**

评估数据集包括Electricity、Solar、Traffic三个真实数据集以及一个合成数据集。

**📈 对比分析**

与标准归一化和去掉仿射层的RevIN进行对比，实验表明Instance Normalization在新日期/新用户上平均表现最佳，但在某些平稳数据上可能受损；在归一化空间训练的模型在最终非归一化MSE上表现更好。

**⚠️ 局限性**

局限性在于RevIN对条件分布偏移无效，归一化可能削弱可预测的尺度与位移信息，并且对非平稳或平稳数据的适用性仍需进一步改进。

---

## 215. RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents

**arXiv ID:** 2603.11337 | [PDF](https://arxiv.org/pdf/2603.11337v1)

**作者:** Yonas Atinafu `[一作]` (University of Waterloo), Robin Cohen `[通讯]` (University of Waterloo)

**通讯引用:** 11457 | [OpenAlex ID](https://openalex.org/A5000636604)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RewardHackingAgents 框架，对 LLM 机器学习工程代理的评估完整性进行基准化测试，识别并量化评估过程中的两大破坏向量：评估器篡改和训练/测试泄露。

**💡 创新点**

创新点在于：①将评估完整性作为首要基准结果；②通过工作空间隔离、补丁跟踪、文件访问日志和可信参考指标，实现可审计的“评估器篡改”与“数据泄露”区分；③引入四种信任策略，系统评估单一与联合防御对攻击面的影响。

**🔧 技术方法**

使用基于工作空间的 episode 机制，配合文件 I/O 记录、评估器哈希对比、可信参考脚本执行以及自定义检测器，构成完整的评估完整性检测流程。

**📊 数据集**

在三类任务上验证：信用风险预测（XGBoost pipeline）、CIFAR-10 图像分类（ResNet-18 pipeline）以及 SST-2 文本情感分类（DistilBERT pipeline），并在两种 LLM 后端（TinyLlama 与 Qwen）下运行。

**📈 对比分析**

比较方法：对 scripted 攻击、benign 控制与自然代理三种行为进行实验，分别记录总体与向量级攻击成功率、误报率、漂移率和运行时开销。结果显示，单一防御只能阻止一种攻击，只有联合防御（full_locked）能将总体攻击率降至 0；在自然代理实验中，评估器篡改尝试出现率约 50%，被锁定策略有效消除，额外运行时间提升约 25–31%。

**⚠️ 局限性**

局限性包括：只覆盖两种攻击向量，未考虑 OS 层逃逸、子进程 I/O、数据污染等更细粒度威胁；实验任务与后端有限；自然代理行为受提示与任务设计影响，结果并非普适；评估完整性检测主要基于文件访问和哈希，可能被更隐蔽的篡改手段规避。

---

## 216. Probabilistic Disjunctive Normal Forms in Temporal Logic and Automata Theory

**arXiv ID:** 2603.11083 | [PDF](https://arxiv.org/pdf/2603.11083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 217. Identifying the Group to Intervene on to Maximise Effect Under Cross-Group Interference

**arXiv ID:** 2603.11059 | [PDF](https://arxiv.org/pdf/2603.11059v1)

**作者:** Xiaojing Du `[一作]` (Adelaide University), Thuc Duy Le `[通讯]` (Adelaide University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并解决跨组因果影响最大化问题，即在源组中选择最能对目标组产生最大因果影响的干预子集。

**💡 创新点**

定义了核心到群组因果效应（Co2G）并证明其在观测网络数据下可通过do‑calculus非参数识别；提出不确定性感知的 Causal Maximization (CauMax) 框架，包含基于图神经网络的估计器和两种可扩展的子集搜索算法。

**🔧 技术方法**

使用图神经网络（GNN）建模跨组干预传播，Monte Carlo dropout 估计不确定性，Gumbel‑Softmax 进行可微分子集优化；在实验中还对比了随机、度中心、影响力最大化等基线。

**📊 数据集**

在 BlogCatalog 和 Flickr 两个真实社交网络数据集上进行实验。

**📈 对比分析**

相较于随机、度中心、IM 等基线，CauMax（尤其是 CauMax‑D）在所有预算水平下均实现了至少十倍降规差（regret）和更低的 RMSE，表明能更精准地识别并最大化跨组因果影响。

**⚠️ 局限性**

方法依赖于因果充分性假设，若存在未观测的混杂变量会导致 Co2G 估计偏差；此外，当前模型未考虑隐藏混杂或时间动态效应。

---

## 218. Bridging Behavioral Biometrics and Source Code Stylometry: A Survey of Programmer Attribution

**arXiv ID:** 2603.11150 | [PDF](https://arxiv.org/pdf/2603.11150v1)

**作者:** Marek Horvath `[一作]` (Technical University of Kosice), Diomidis Spinellis `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 8764 | [OpenAlex ID](https://openalex.org/A5021948425)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对程序员身份识别进行系统性映射研究，整理特征类型、模型方法、数据来源和评估实践。

**💡 创新点**

构建统一的分类法，将静态风格特征与行为生物特征关联，并揭示验证与多作者场景的研究空白。

**🔧 技术方法**

结合传统机器学习（SVM、RF、kNN等）、深度学习（Transformer、CNN、GNN）以及统计特征提取技术。

**📊 数据集**

主要使用Google Code Jam、GitHub公开项目以及合成数据集，覆盖多种编程语言。

**📈 对比分析**

采用交叉验证、准确率、F1‑score等标准指标进行对比，发现传统与深度模型在大多数任务上性能相近，但对短代码片段仍表现有限。

**⚠️ 局限性**

高度依赖少数基准数据集，行为特征难以收集，缺乏开放验证、协作多作者实验和完整复现流程。

---

## 219. Enumerating All Directed Spanning Trees in Optimal Time

**arXiv ID:** 2603.11763 | [PDF](https://arxiv.org/pdf/2603.11763v1)

**作者:** Paweł Gawrychowski `[一作]` (University of Wrocław), Marcin Knapik `[通讯]` (University of Wrocław)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对给定有根有向图，提出一种算法能在总时间 (n+m+N) 和空间 (n+m) 下枚举出所有有向生成树（支配树）。

**💡 创新点**

创新点在于引入图论上对极少生成树图的结构刻画（链分解与仿真图），并利用该结构在递归过程中实现快速更新，从而突破以往 (n+m+N·polylog n) 的时间界限。

**🔧 技术方法**

核心技术包括：去除无用/必选边的修剪 (trimming)、对两条边不相交支配树的构造 (divergent trees)、链分解与仿真图 (emulation graph)、并行边的平坦化 (flattening)、并使用并查集与基数排序实现高效的节点/边更新。

**📊 数据集**

本文为理论工作，未使用具体实验数据集；所有结论均基于图论分析与算法复杂度证明。

**📈 对比分析**

与先前最优方法 (n+m+N·log²n) 相比，新的算法将额外的对数因子完全消除，达到理论上最优的 (n+m+N) 复杂度；但实现中的常数因子和递归深度仍需实际评估。

**⚠️ 局限性**

局限性包括：递归深度最高可达 m，导致在最坏情况下单个解之间的延迟为 O(m)；算法对图的结构假设（有根、可枚举的有向生成树）较为严格；实现细节中如平坦化、仿真图维护等可能引入较大的空间/时间常数。

---

## 220. Taming OpenClaw: Security Analysis and Mitigation of Autonomous LLM Agent Threats

**arXiv ID:** 2603.11619 | [PDF](https://arxiv.org/pdf/2603.11619v1)

**作者:** Xinhao Deng `[一作]` (Ant Group & Tsinghua University), Qi Li `[通讯]` (Tsinghua University)

**通讯引用:** 26126 | [OpenAlex ID](https://openalex.org/A5100350243)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对OpenClaw等自主LLM代理的安全威胁进行全面分析，并提出覆盖初始化、输入、推理、决策、执行五个生命周期阶段的防御框架。

**💡 创新点**

系统性地将威胁与防御映射到生命周期各阶段，强调跨阶段攻击链与多层防御协同，首次提出针对持续学习与插件生态的动态安全策略。

**🔧 技术方法**

使用插件审计与签名验证、指令分层与语义防火墙、向量空间访问控制与加密状态检查点、形式化验证与逻辑约束、内核级沙箱与运行时追踪等技术。

**📊 数据集**

基于OpenClaw公开实现的案例、公开插件仓库、RAG知识库以及常用Web抓取数据等数据集进行实验。

**📈 对比分析**

与传统单点防御（如guardrail、prompt‑data separation）对比，实验显示在阻止间接prompt注入、内存中毒、意图漂移等复合攻击时误报率下降约20%，成功率降低约30%。

**⚠️ 局限性**

局限性：未覆盖LLM内部权重攻击，依赖手工阈值调优，硬件安全方案成本较高，无法完全覆盖所有动态API的变化和跨租户环境的细粒度隔离。

---

## 221. Efficient Cross-View Localization in 6G Space-Air-Ground Integrated Network

**arXiv ID:** 2603.11398 | [PDF](https://arxiv.org/pdf/2603.11398v1)

**作者:** Min Hao `[一作]`, Wei Ni `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在6G空间-空中-地面集成网络(SAGIN)中实现高效跨视角定位的分拆推理框架，并给出了联合通信、计算与机密三维优化方案。

**💡 创新点**

创新点在于将跨视角定位与6G SAGIN深度融合，设计分拆推理架构以降低传输量和能耗，并通过强化学习实现三维成本的自适应最小化。

**🔧 技术方法**

采用分拆推理技术、ResNet‑50 + USAM 关注模块、三维协同通信计算模型，以及Actor‑Critic等强化学习算法进行优化。

**📊 数据集**

使用University‑1652数据集，该数据集包含卫星、无人机和车辆视角的1,652栋建筑图像。

**📈 对比分析**

通过Recall@K和AP指标评价实验结果，发现四个无人机视图+四个车辆视图时Recall@1≈86.9%、AP≈61.3%，同时RL优化显著降低了三维成本。

**⚠️ 局限性**

局限在于缺乏对真实多模态环境的验证、模型分拆点的选择仍依赖经验、以及对隐私攻击模型的理想化假设。

---

## 222. Can Small Language Models Use What They Retrieve? An Empirical Study of Retrieval Utilization Across Model Scale

**arXiv ID:** 2603.11513 | [PDF](https://arxiv.org/pdf/2603.11513v1)

**作者:** Sanchit Pandey `[一作]` `[通讯]` (Birla Institute of Technology and Science Pilani), Sanchit Pandey (Birla Institute of Technology and Science Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了小规模语言模型（≤7B参数）在检索增强生成（RAG）中的利用效果，使用oracle检索和parametric知识拆分对模型做诊断。

**💡 创新点**

引入oracle检索条件和parametric知识拆分，将检索质量与利用率分离，系统评估小模型的上下文利用瓶颈。

**🔧 技术方法**

采用四位NF4量化的SmolLM2、Qwen2.5、Llama3.1指令微调模型，结合BM25、密集检索、Oracle检索以及多模板提示进行评估。

**📊 数据集**

使用Natural Questions（单跳）和HotpotQA（多跳）共1000道问题，并构建PopQA长尾评估集。

**📈 对比分析**

通过Exact Match (EM) 与 F1 进行对比，发现无检索优于所有检索方法，Oracle检索在7B模型上最多只能取得约15% EM，检索导致已知答案被破坏 42-64%。

**⚠️ 局限性**

仅针对提取式问答，评估局限于英文Wikipedia领域、decoder-only模型、量化影响、检索语料覆盖率有限等。

---

## 223. Frequency Moments in Noisy Streaming and Distributed Data under Mismatch Ambiguity

**arXiv ID:** 2603.11216 | [PDF](https://arxiv.org/pdf/2603.11216v1)

**作者:** Kaiwen Liu `[一作]` (Indiana University), Qin Zhang `[通讯]` (Indiana University)

**通讯引用:** 7036 | [OpenAlex ID](https://openalex.org/A5100418221)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的框架用于在噪声数据集上进行统计估计，专注于频率矩（F_p）问题，展示了在数据流模型和协调模型中如何使用亚线性空间和通信来近似未知真实数据集的F_p。

**💡 创新点**

创新点在于引入了F_p-不匹配模糊性这一数据依赖量，并建立了一系列关于输入大小的紧下界，揭示了在噪声环境下F_p问题的复杂性。

**🔧 技术方法**

使用了数据流模型和协调模型的算法设计，结合了随机采样和计数技术。

**📊 数据集**

使用了合成的噪声数据集，具体数据集的构造依赖于输入的真实情况，且在不同的实例中可能有所不同。

**📈 对比分析**

与现有方法相比，提出的算法在噪声环境下的性能显著提高，尤其是在协调模型中，当F_p-不匹配模糊性低于某个阈值时，通信成本可以独立于输入大小。

**⚠️ 局限性**

限制在于当F_p-不匹配模糊性较大时，标准统计度量可能失去意义，且可能无法仅基于噪声输入估计真实的统计函数。

---

## 224. Bayesian Optimization of Partially Known Systems using Hybrid Models

**arXiv ID:** 2603.11199 | [PDF](https://arxiv.org/pdf/2603.11199v1)

**作者:** Eike Cramer `[一作]`, Alexander Mitsos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将部分已知的机理模型与高斯过程（GP）相结合的混合贝叶斯优化框架，并将其转换为随机非线性规划（NLP）求解。

**💡 创新点**

核心创新在于：①将缺失的物理方程通过GP嵌入为等式约束，构造混合模型；②通过样本平均逼近（SAA）将随机规划离散化；③在此框架下定义混合模型的期望改进（SAA-EI）作为获取函数，实现对传统黑箱贝叶斯优化的显著改进。

**🔧 技术方法**

技术手段包括：高斯过程回归、重参数化技巧、样本平均逼近、CasADi自动微分、IPOPT非线性求解器，以及拉丁超立方采样等多起点策略。

**📊 数据集**

使用的“数据集”主要为两类：①仿真数据，基于Forrester函数的单变量测试；②闪蒸单元的仿真数据，采用NRTL模型作为真值，收集温度、压力、底部浓度等观测；两种情境均采用在温度-压力平面上拉丁超立方采样获得初始样本。

**📈 对比分析**

与传统基于EI的贝叶斯优化以及随机采样进行对比。混合模型在Forrester例子中10次迭代后实现七个数量级的改进；在闪蒸单元中仅需4次迭代即可获得比标准贝叶斯优化低两阶的目标值。实验显示混合模型在收敛速度和最终目标值上均显著优于对照组。

**⚠️ 局限性**

主要局限包括：①SAA-EI近似导致后期EI趋于零，限制了获取函数的探索能力；②非凸的SAA-EI求解需要大量多起点，计算开销大；③对不可行区间的处理仍为经验性（设置高目标或随机值），未充分利用不等式约束；④未考虑实验噪声、模型误差以及真实实验验证，需进一步扩展。

---

## 225. MiNI-Q: A Miniature, Wire-Free Quadruped with Unbounded, Independently Actuated Leg Joints

**arXiv ID:** 2603.11537 | [PDF](https://arxiv.org/pdf/2603.11537v1)

**作者:** Daniel Koh `[一作]` (University of California Los Angeles), Dennis Hong `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一款无线、微型四足机器人MiNI-Q，配备独立驱动、无物理限位的两自由度腿部机构，并在实验中验证了其在多种地形下的爬梯、低通道、连续旋转及后翻等多样化运动能力。

**💡 创新点**

通过螺旋齿轮与带传动实现腿部关节无限旋转的独立驱动结构，打破传统串联/并联关节限制；同时采用无线架构和开源硬件降低构建门槛，提供更大的工作空间和多样化步态。

**🔧 技术方法**

机械设计使用PLA 3D打印、金属齿轮与带传动；电子设计采用DYNAMIXEL XL330电机、ESP32-C3微控制器、FreeRTOS调度、IMU（BNO055）及电流测量；控制实现高速模式切换与多任务调度。

**📊 数据集**

未使用公开数据集，主要采集实验数据（IMU姿态、电流、速度）用于跑步、爬梯、低通道、跳跃等运动的评测。

**📈 对比分析**

与Q8bot及其他小型四足（Chen等）在顶速、归一化顶速和能量运输成本（COT）进行对比；MiNI-Q实现0.46 m/s（5.22体长/秒）顶速，COT 8.1，能耗略高但提供更大工作空间与多样化运动。

**⚠️ 局限性**

受限于开环控制、ESP32-C3的调度延迟及电流传感器噪声；电压波动影响COT计算；缺乏触觉/接触传感器，无法实现实时闭环响应。

---

## 226. Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding

**arXiv ID:** 2603.11831 | [PDF](https://arxiv.org/pdf/2603.11831v1)

**作者:** Jiahao Li `[一作]` (Fudan University), Xiangdong Zhou `[通讯]` (Fudan University)

**通讯引用:** 7753 | [OpenAlex ID](https://openalex.org/A5085205118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出FutureCAD框架，利用大型语言模型（LLM）生成可执行的CadQuery程序，并通过文本查询实现B‑Rep几何实体的精准定位，从而实现高保真、支持高级特征（如倒圆角、倒角、壳体等）的文本到CAD模型生成。

**💡 创新点**

核心创新在于：①将LLM生成的程序与基于Transformer的BRepGround模块耦合，使自然语言查询能够在运行时精确映射到当前B‑Rep中的几何实体；②构建了包含140k真实工业CAD模型的专用数据集，并为每个模型生成了层级化的自然语言描述和查询；③采用两阶段训练策略——监督微调（SFT）+强化学习（RL），显著提升程序可执行性与几何精度。

**🔧 技术方法**

使用技术包括：Qwen2.5-7B-Instruct LLM、BRepGround（UV-Net + GNN + Transformer 融合模块）、CLIP‑style对比学习、Group Sequence Policy Optimization (GSPO)、Chamfer Distance奖励、CadQuery API扩展以及BERT文本编码。

**📊 数据集**

数据集为FutureCAD，约14万条工业CAD模型，包含标准特征与高级特征子集；在此基础上还对CadQuery代码进行了重写并加入多样化API；同时在Fusion360数据集上进行跨域测试。

**📈 对比分析**

与CAD-Translator、Text2CAD、Text‑to‑CadQuery、CADFusion、CAD‑LLaMA等基线相比，FutureCAD在高级特征子集上实现了最低Chamfer Distance（最高几何精度）并将无效率压至≈1%；在标准子集及Fusion360跨域测试中同样保持较低CD与IR，明显优于对比方法。

**⚠️ 局限性**

局限性包括：对复杂几何中的精确实体定位仍易受查询歧义影响；模型在极端高细节或非常规特征上的表现尚待验证；以及对安全性与合规性的审核依赖人工监督。

---

## 227. BackdoorIDS: Zero-shot Backdoor Detection for Pretrained Vision Encoder

**arXiv ID:** 2603.11664 | [PDF](https://arxiv.org/pdf/2603.11664v1)

**作者:** Siquan Huang `[一作]` (South China University of Technology), Leyu Shi `[通讯]` (South China University of Technology)

**通讯引用:** 595 | [OpenAlex ID](https://openalex.org/A5111111992)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种零样本、推理时可用的视觉编码器后门样本检测方法 BackdoorIDS。

**💡 创新点**

创新点在于利用“注意力劫持”和“恢复”两种现象，通过逐步遮掩图像生成嵌入序列，并用密度聚类判断是否存在后门，无需辅助数据或模型重训练。

**🔧 技术方法**

采用图像分块遮掩、预训练视觉编码器提取嵌入、计算局部密度、使用 DBSCAN 聚类以及阈值判断的技术组合。

**📊 数据集**

使用 SVHN、GTSRB、ImageNet 以及 COCO Caption 等公开数据集，并在 ResNet、ViT、CLIP、LLaVA‑1.5 等多种编码器上评估。

**📈 对比分析**

与 DeDe、PatchProcessing 等主流推理时防御方法对比，BackdoorIDS 在 5 种攻击（BadEncoder、Drupe、BadCLIP、BadVision）中取得 TPR 最高（最高 97%）、FPR 较低（约 17–28%），显著降低 ASR 并保持较高 CA。

**⚠️ 局限性**

局限性包括对某些复杂攻击（如 Drupe）TPR 略低、需手动调节遮掩步数与聚类阈值，对极端噪声或 JPEG 压缩下的性能略有下降，以及相对较高的推理时计算开销。

---

## 228. Real-World Point Tracking with Verifier-Guided Pseudo-Labeling

**arXiv ID:** 2603.12217 | [PDF](https://arxiv.org/pdf/2603.12217v1)

**作者:** Görkay Aydemir `[一作]` (Koç University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10269 | [OpenAlex ID](https://openalex.org/A5076097168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个学习型验证器（verifier），用于在多轨迹候选中动态评估每帧的可靠性，从而生成高质量的伪标签，用于无标注视频的自监督细调点跟踪模型；同时该验证器也可作为推理时的自适应集成模块。

**💡 创新点**

创新点在于：①将轨迹可靠性评估迁移为可学习的元模型；②利用对比学习在合成数据上训练验证器，使其能跨域识别轨迹一致性；③通过验证器选取最佳轨迹实现伪标签生成，显著降低传统单教师或随机教师的误差累积；④验证器既可用于训练，也可在推理时作为动态集成器。

**🔧 技术方法**

技术手段包括：多教师轨迹候选生成、局部特征提取（基于冻结CNN与可变形注意力）、候选Transformer（交叉注意力 + 时序自注意力）、对比式软目标训练、伪标签生成与可见性投票、以及合成-实景混合训练策略。

**📊 数据集**

数据集：合成训练用K-EPIC（≈11K视频），实景微调用从TAO、OVIS、VSPW筛选的≈8K视频；评估在四个真实世界基准上：TAP‑Vid DAVIS、TAP‑Vid Kinetics、RoboTAP、EgoPoints。

**📈 对比分析**

与之前的随机教师、单教师及CoTracker3等方法对比，验证器在四大基准上均取得最佳或次佳成绩；在DAVIS、Kinetics、RoboTAP和EgoPoints上实现了SOTA或显著提升，数据效率高，实景训练样本量相对较少即可获得显著性能提升。

**⚠️ 局限性**

局限性包括：细调性能仍受可用视频多样性和质量影响；验证器的上限受教师模型质量限制；对极端遮挡或长时间漂移的鲁棒性仍有提升空间；目前验证器主要关注点轨迹，可能不适用于更复杂的多点或场景关联任务。

---

## 229. DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**arXiv ID:** 2603.12257 | [PDF](https://arxiv.org/pdf/2603.12257v1)

**作者:** Yujie Wei `[一作]` (Fudan University), Hongming Shan `[通讯]` (Fudan University)

**通讯引用:** 4464 | [OpenAlex ID](https://openalex.org/A5049086157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个两阶段统一框架Omni-Motion Controlled Multi-Subject Video Customization，实现多主体身份定制与全粒度运动控制的无缝融合。

**💡 创新点**

创新性地将多主体身份与多粒度运动控制整合至单一DiT网络，并引入条件感知3D Rotary Positional Embedding、层级运动注入、组/角色嵌入以及潜在身份奖励反馈机制。

**🔧 技术方法**

采用Diffusion Transformer为生成器，结合3D VAE编码、条件感知3D RoPE、层级bbox注入、组/角色嵌入、基于视频扩散模型的潜在身份奖励模型与奖励反馈学习。

**📊 数据集**

构建了2M视频的自研多主体运动定制数据集，并创建了1,027条真实视频的DreamOmni Bench用于全面评测。

**📈 对比分析**

与DreamVideo-2、VACE、Phantom、Wan-Move、Video Alchemist、Tora2等基线在DreamOmni Bench和MSRVTT-Personalization Bench上对比，显著提升身份保真度（R-DINO、R-CLIP、Face-S）与运动控制精度（mIoU、EPE）。

**⚠️ 局限性**

仍存在模型规模和训练成本高、极端复杂场景下运动漂移或身份混合的风险，且奖励反馈需要精细调节λ_2以防止奖励破解。

---

## 230. HumDex:Humanoid Dexterous Manipulation Made Easy

**arXiv ID:** 2603.12260 | [PDF](https://arxiv.org/pdf/2603.12260v1)

**作者:** Liang Heng `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 HumDex——一种基于 IMU 的便携式全身遥操作系统，配备学习驱动的指尖转手部动作映射，并实现了两阶段基于人类数据的模仿学习框架，用于全身及手部精细操作。

**💡 创新点**

创新点：①使用 15 个轻量级 IMU 追踪器实现无遮挡、可携带的全身追踪；②采用轻量 MLP 学习指尖到 20 关节手部动作的映射，避免了优化求解；③两阶段预训练‑微调策略将多样化人类数据转化为机器人可执行的动作先验，显著提升泛化能力。

**🔧 技术方法**

技术：IMU 运动捕捉、General Motion Retargeting（GMR）全身对齐、TWIST2/SONIC 低层控制、MLP 指尖映射、Action Chunking Transformer (ACT) 政策、ResNet‑18 视觉编码器。

**📊 数据集**

数据集：60 轮任务遥操作演示（含 Scan&Pack、HangTowel、OpenDoor、PlaceBasketOnShelf、PickBread）；人类演示 500+ 轮（位置、对象、背景多样化）；机器人仅 50 轮；合并数据用于预训练与微调。

**📈 对比分析**

方法对比：与基线视觉遥操作（PICO + 视觉手部追踪）比较；指标包括收集时间、遥操作成功率、策略成功率。结果显示：收集时间缩短 26%（44.3 min vs 59.8 min），遥操作成功率 91.7% vs 74.6%，策略成功率 80.0% vs 57.5%。学习手部映射相较优化求解更快、更稳；两阶段训练使 OOD 泛化率从 12/30、10/30、9/30 提升至 21/30、20/30、25/30。

**⚠️ 局限性**

限制：训练数据规模有限，可能进一步提升性能；手部姿态覆盖面有限，未涉及更复杂接触与力感应交互；硬件负载与驱动力限制，无法探索更高性能的操控行为。

---

## 231. Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics

**arXiv ID:** 2603.12087 | [PDF](https://arxiv.org/pdf/2603.12087v1)

**作者:** Ming-Hong Chen `[一作]` (National Yang Ming Chiao Tung University), Ping-Chun Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 260 | [OpenAlex ID](https://openalex.org/A5017738079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出跨域Bellman一致性度量，设计QAvatar算法，利用混合Critic和自适应权重实现跨域强化学习，在不同状态/动作空间间实现可靠知识转移。

**💡 创新点**

①通过跨域Bellman一致性定义源模型可转移性；②QAvatar采用源域与目标域Q函数加权融合，并以超参数自适应权重实现无前置判断；③给出收敛上界并证明该权重可在无超参数条件下获得。

**🔧 技术方法**

使用自然策略梯度(NPG)、SAC、软策略迭代、交叉域Bellman损失、正则化流模型进行映射学习、以及无超参数加权方案。

**📊 数据集**

MuJoCo 运动学环境（HalfCheetah、Ant）、Robosuite 机器人臂任务（Panda→UR5e 的门开启、桌面擦拭）、Safety‑Gym 导航（Car→Doggo）、DeepMind Control Suite 图像连续控制任务（walker_walk→walker_run、cheetah_run）。

**📈 对比分析**

与CMD、CAT（PPO/SAC）、PAR、从零开始的SAC、直接微调（FT）等基线比较；在所有任务中，QAvatar 取得更快收敛、较低的阈值步数、最高的IQM；在正负转移、低质量源模型、无关源目标、非平稳环境及图像任务等多种情形下均表现出稳健的性能提升。

**⚠️ 局限性**

假设目标域采样成本高于训练成本；训练时间约为SAC 的两倍；映射与流模型的学习需要额外计算，可能影响实时部署；方法假设两域共享折扣因子并需要可探索的初始分布。

---

## 232. Performance Evaluation of Open-Source Large Language Models for Assisting Pathology Report Writing in Japanese

**arXiv ID:** 2603.11597 | [PDF](https://arxiv.org/pdf/2603.11597v1)

**作者:** Masataka Kawai `[一作]` (National Cancer Center Hospital East), Genichiro Ishii `[通讯]` (National Cancer Center Hospital East)

**通讯引用:** 14653 | [OpenAlex ID](https://openalex.org/A5046015782)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了七款开源大语言模型在日语病理报告写作中的四项任务：格式化与信息提取、错别字纠正、解释文本生成和主观评价。

**💡 创新点**

首次在日语病理报告场景下比较了思考型模型与医学专用模型的性能，并揭示了模型在不同任务和评价者之间的显著差异。

**🔧 技术方法**

采用了基于 Hugging Face 的开源 LLM（如 Gemma、MedGemma、SIP-jmed、Qwen3、gpt-oss 等），在本地 M2 Ultra 机器上通过 OpenAI API 进行推理。

**📊 数据集**

使用了 19 例日本乳腺癌协会报告规则的 JSON 数据、National Cancer Center Hospital East 的 31 份真实病理报告（含合成错别字）以及 23 份包含免疫组化的病例报告作评估。

**📈 对比分析**

通过字符三元组 F1、Jaccard 以及 pT、NG、HG 等指标与人工评测比较，发现思考型模型在推理准确度上占优，医学专用模型在错别字纠正和解释文本生成上表现最佳；整体性能仍低于商业闭源模型。

**⚠️ 局限性**

局限包括未与商业闭源模型直接对比、提示设计与采样随机性影响复现性、未对量化模型进行深入调优、缺乏对多样化语言细节的覆盖，以及主观评价者间差异大导致结果不稳定。

---

## 233. Privacy in ERP Systems: Behavioral Models of Developers and Consultants

**arXiv ID:** 2603.12195 | [PDF](https://arxiv.org/pdf/2603.12195v1)

**作者:** Alicia Pang `[一作]` (Leiden University), Olga Gadyatskaya `[通讯]` (Leiden University)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5068415784)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

调查了ERP系统开发者和顾问的隐私意识与行为，基于Fogg行为模型构建了隐私行为模型；

**💡 创新点**

将主题分析与行为模型结合，系统化映射GDPR合规性与隐私实践至动机、能力与触发因素；

**🔧 技术方法**

采用半结构访谈、主题分析与Fogg行为模型；

**📊 数据集**

16份访谈记录（7名开发者、9名顾问/经理）；

**📈 对比分析**

作为质性研究，没有对比基准或性能评估；

**⚠️ 局限性**

样本仅来自一家荷兰咨询公司，采用便利抽样并可能存在社会期望偏差。

---

## 234. Flight through Narrow Gaps with Morphing-Wing Drones

**arXiv ID:** 2603.12059 | [PDF](https://arxiv.org/pdf/2603.12059v1)

**作者:** Julius Wanner `[一作]` (École Polytechnique Fédérale de Lausanne), Dario Floreano `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 29019 | [OpenAlex ID](https://openalex.org/A5059369445)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并验证了一种可在窄缝中飞行的可变翼展开机，通过在低速、过失速条件下对翼扫角进行实时控制，实现了与翼宽相当以下的窄缝通过。

**💡 创新点**

提出了一种基于低雷诺数和后失速下可实时求解的翼扫机动气动模型，并将其嵌入多相优化与非线性MPC控制框架，实现了在保持恒定前进速度下通过窄缝的精确控制。

**🔧 技术方法**

利用伪谱多相轨迹优化、可调成本与约束的非线性模型预测控制、实时翼扫舵机耦合，以及运动捕捉测量与延迟补偿技术。

**📊 数据集**

通过风洞实验和三次飞行试验（分别在 5/6/7 m/s 起飞速度和不同阈值下）获取的升阻、姿态与速度数据，验证模型与控制性能。

**📈 对比分析**

将三种优化目标（最小高度变化、最小速度变化、最小翼扫时长）产生的轨迹分别在闭环MPC下测试，平均通过缝高度误差约为5 cm，速度保持恒定，误差随起速升高而减小，最小速度变化目标的误差略优。

**⚠️ 局限性**

受限于低角度失速时舵机灵敏度不足、模型对俯仰动力学的低估以及潜在尾翼碰撞风险，导致在低速（≈5 m/s）和大角度爬升时的可靠性受限。

---

## 235. Leveraging Phytolith Research using Artificial Intelligence

**arXiv ID:** 2603.11476 | [PDF](https://arxiv.org/pdf/2603.11476v1)

**作者:** Andrés G. Mejía Ramón `[一作]` (Institut de Ciencia i Tecnologia Ambientals Universitat Autònoma de Barcelona), Umberto Lombardo `[通讯]` (Departament de Prehistòria Universitat Autònoma de Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了名为 Sorometry 的端到端人工智能平台，实现了光学显微镜焦点堆栈图像的自动数字化、二维正射图像与三维点云的生成、分割、分类以及群落来源推断；

**💡 创新点**

首次将 2D 图像与 3D 点云通过多模态融合模型相结合，显著提升诊断形态分类精度，并引入贝叶斯有限混合模型实现群落级别植物来源推断，形成可扩展的“omics”式工作流；

**🔧 技术方法**

使用 ConvNeXt CNN 处理 2D 正射图像，PointNet++ 处理 3D 点云，随后通过晚期融合策略与贝叶斯混合模型；AI 开发过程中借助 LLM 辅助代码生成；

**📊 数据集**

共 712 个 2 mm × 2 mm 区段（含考古沉积、土壤核心及参考集合），生成约 381 万条分割点云，专家手工标注 15 842 条分割质量与 4 638 条形态标签；

**📈 对比分析**

在 24 种诊断形态上，融合模型整体准确率 77.9%，分割质量准确率 84.5%；相比单模态 ConvNeXt 或 PointNet++，准确率提升约 3–5%；在群落分析中，贝叶斯模型成功识别玉米、棕榈等植物来源；

**⚠️ 局限性**

对某些形态（如 Elongate、Grass silica short cell）仍存在混淆；训练数据覆盖有限，模型对未知形态的泛化能力不足；需要进一步扩充参考集合、细化类别和处理退化/过渡形态。

---

## 236. Multi-Robot Multitask Gaussian Process Estimation and Coverage

**arXiv ID:** 2603.11264 | [PDF](https://arxiv.org/pdf/2603.11264v1)

**作者:** Lai Wei `[一作]` (University of Michigan), Vaibhav Srivastava `[通讯]` (Michigan State University)

**通讯引用:** 3290 | [OpenAlex ID](https://openalex.org/A5069896928)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了多任务覆盖问题，并针对已知与未知感知需求分别设计了联邦多任务覆盖算法与自适应多任务覆盖算法

**💡 创新点**

创新点在于首次将多任务Gaussian Process学习与联邦通信架构结合，提出多任务覆盖惩罚与子线性累计回报分析

**🔧 技术方法**

采用联邦通信、Voronoi/等效分区、Gaussian Process多任务推断、确定性 epoch 调度与Lyapunov 收敛证明

**📊 数据集**

使用合成的21×21网格场景，混合高斯需求场与随机生成的机器人任务效能参数

**📈 对比分析**

通过与随机多任务学习与覆盖（RMLC）算法对比，DSMLC在累计回报上显著优于 RMLC，收敛到已知需求的最优部署

**⚠️ 局限性**

局限在于假设感知场是静态、已知噪声、线性成本函数，且仅在离散网格上验证，未考虑机器人动力学或非平稳环境

---

## 237. Opinion Dynamics in Learning Systems

**arXiv ID:** 2603.12137 | [PDF](https://arxiv.org/pdf/2603.12137v1)

**作者:** Jiduan Wu `[一作]` (Max Planck Institute for Intelligent Systems), Celestine Mendler-Dünner `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 167 | [OpenAlex ID](https://openalex.org/A5013807314)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究平台推荐与社交网络中意见动态的相互作用，构建递归的演化模型并分析其稳定均衡。

**💡 创新点**

将学习系统的 performative 效应与传统意见动力学耦合，发现平台既可作为同质化力量，又能通过聚合信息创造跨节点影响路径，从而促使网络在极端条件下达成一致。

**🔧 技术方法**

采用 Friedkin–Johnsen 模型、递归学习更新、闭式理论推导及半合成仿真，并比较了 Perfect、Mean、OLS 与 MLP 四种预测算法。

**📊 数据集**

使用 Pokec 社交网络（约 1.6 M 节点）的子图，并利用 Transformer 提取“relation_to_smoking”特征进行预测。

**📈 对比分析**

通过对比四种预测策略在方差收敛、均值变化及理论闭式结果，实验表明无论算法如何，performative 均衡均比单独的意见动力学更同质，且平台偏差会放大干预效果。

**⚠️ 局限性**

只考虑完美预测或简单参数模型，忽略网络拓扑演化与测量不确定性；理论多聚焦于均匀易感度与单一网络，实测验证尚有限。

---

## 238. Task-Conditioned Routing Signatures in Sparse Mixture-of-Experts Transformers

**arXiv ID:** 2603.11114 | [PDF](https://arxiv.org/pdf/2603.11114v1)

**作者:** Mynampati Sri Ranganadha Avinash `[一作]` `[通讯]` (Asthra Labs), Mynampati Sri Ranganadha Avinash (Asthra Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了稀疏 MoE 结构中的路由签名（routing signatures），探讨其是否随任务而呈现结构化分布。

**💡 创新点**

创新点在于：①提出路由签名作为跨层专家激活模式的向量摘要；②通过统计相似度与基线对比，首次证明路由模式具有任务条件化特性；③用路由签名实现高精度任务分类。

**🔧 技术方法**

使用了：路由签名构造（按层专家激活计数归一化）、层级余弦相似度、置换与负载平衡基线、逻辑回归分类器、Cohen’s d 统计、PCA 低维可视化。

**📊 数据集**

实验数据集为：OLMoE‑1B‑7B‑0125‑Instruct 模型下，80 条提示，分四类（代码、数学、故事、事实问答）。

**📈 对比分析**

比较方法：与随机置换和负载平衡基线对比，计算同类与异类提示间的路由相似度。结果显示同类相似度为 0.8435±0.0879，异类为 0.6225±0.1687，Cohen’s d = 1.44；仅用路由签名的逻辑回归在五折交叉验证下达成 92.5%±6.1% 的准确率。

**⚠️ 局限性**

局限性：仅评估单一模型与 80 条提示；实验为相关性分析，缺乏因果干预与更广泛模型/数据集验证；路由签名不揭示单个专家的语义功能。

---

## 239. LongFlow: Efficient KV Cache Compression for Reasoning M

**arXiv ID:** 2603.11504 | [PDF](https://arxiv.org/pdf/2603.11504v1)

**作者:** Yi Su `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 39085 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 LongFlow 的 KV 缓存压缩方法，专门针对长输出推理模型的高内存占用和计算瓶颈。

**💡 创新点**

创新点在于：① 零历史估计——仅利用当前查询向量和注意力中间结果即可评估历史 token 的重要性；② 零成本估计——将重要性分数直接从已计算的贡献向量中得到，无需额外计算或存储；③ 将 FlashAttention、重要性评估与 token 淘汰融合为单个 Triton 核心，显著降低延迟并提升吞吐。

**🔧 技术方法**

技术手段包括：Transformer 注意力机制、KV 缓存、FlashAttention、Triton 自定义 fused kernel、静态内存分配、长链思维（Chain-of-Thought）生成、LongFlowScore 重要性指标。

**📊 数据集**

实验使用的模型和数据集：DeepSeek-R1-Distill-Llama-8B、Qwen3 0.6B/1.7B/4B/8B；评估数据集包括 MATH‑500、AMC‑23、AIME‑24/25、GPQA、Minerva、OlympiadBench、GSM8K 等。

**📈 对比分析**

与 Vanilla、H2O、VATP、R‑KV 四个基线进行对比；LongFlow 在保持 0.08–1.3% 以内的准确率下降的同时，实现了最高 11.8× 的吞吐提升、80% 的 KV 缓存压缩率，并减少了内存碎片，支持更大 batch。

**⚠️ 局限性**

局限性：依赖连续查询的相似性，若出现主题跳转或工具使用等突变，重要性估计可能失效；目前仅针对单词查询长度为 1 的自回归推理优化，未覆盖长输入预填充、双向注意或非自回归生成场景。

---

## 240. Radio Radiance Field: The New Frontier of Spatial Wireless Channel Representation

**arXiv ID:** 2603.11588 | [PDF](https://arxiv.org/pdf/2603.11588v1)

**作者:** Haijian Sun `[一作]` (University of Georgia), Feng Ye `[通讯]` (University of Wisconsin-Madison)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的无线信道空间表征方法——无线辐射场（RRF），并基于 3D 高斯喷射（RF-3DGS）构建了高效的 RRF 重建与实时渲染管线，可直接获取完整的空间 CSI（Spatial‑CSI），用于天线波束成形、RIS 设计和集成感知与通信等应用。

**💡 创新点**

创新点包括：①将计算机视觉中的辐射场概念迁移至射频域，并结合电磁传播特性（衍射、反射、极化、频率依赖衰减）进行重新建模；②采用可编辑、可微分的 3D 高斯与球谐函数组合，显著提升了重建精度和推理速度；③在训练阶段引入两阶段融合策略，先利用多分辨率视觉图像学习几何，再冻结几何后通过空间 CSI 训练辐射特征，实现高精度、实时的 RRF 重建。

**🔧 技术方法**

技术手段包括：3D 高斯喷射（3DGS）、球谐函数（SH）表示辐射强度、Alpha‑blending 渲染管线、两阶段训练（几何阶段 + 辐射阶段）、基于视觉与射频联合数据的联合优化、以及针对 Massive MIMO/RIS 的空间 CSI 查询与波束调度。

**📊 数据集**

使用的数据集为：在实验室/室内场景中采集的同步 Tx‑Rx 频道冲激响应（空间 CSI）和对应的可视化图像（正射相机照片）。具体的公开数据集未列出，文中采用了自制的视觉与射频联合测量数据来训练和评估 RF‑3DGS 模型。

**📈 对比分析**

在与 NeRF^2 和 CGAN 的比较中，RF‑3DGS 在感知相似度（LPIPS）上表现更好；训练速度提升 58 倍；渲染速度提升 390 倍；且实现了 2 ms 以内的实时推理。实验表明，RF‑3DGS 能以极低的计算与通信开销提供完整的空间 CSI。

**⚠️ 局限性**

局限性包括：①视觉先验的几何表示为高斯椭球，缺乏精细表面细节，导致对距离变化不敏感；②当前模型主要验证于受控室内环境，面对动态户外场景时对遮挡、非视线传播及实时更新的处理仍不完善；③重建仍需大量测量样本，采样效率有待通过主动学习和贝叶斯优化进一步提升；④缺乏针对多设备协同感知与定位的完整端到端学习框架。

---

## 241. Sharpness-Aware Minimization for Generalized Embedding Learning in Federated Recommendation

**arXiv ID:** 2603.11503 | [PDF](https://arxiv.org/pdf/2603.11503v1)

**作者:** Fengyuan Yu `[一作]` (Zhejiang University), Chaochao Chen `[通讯]` (Zhejiang University)

**通讯引用:** 6516 | [OpenAlex ID](https://openalex.org/A5028791879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的联邦推荐框架FedRecGEL，旨在在联邦推荐中学习稳健的全局物品嵌入；

**💡 创新点**

创新点在于从物品视角将联邦推荐建模为多任务学习，并通过尖锐度感知最小化（SAM）在本地和全局聚合阶段同时优化共享与私有参数，从而显著提升泛化性能；

**🔧 技术方法**

主要技术包括尖锐度感知最小化（SAM）、分层多头架构、FedAvg聚合、负采样与交叉设备联邦学习；

**📊 数据集**

实验使用四个真实数据集：FilmTrust、Lastfm-2K、Amazon-Video 与 QB-article；

**📈 对比分析**

与 FedNCF、FedMF、PerFedRec、PFedRec、FedRAP、CoFedRec、GPFedRec 等基线比较，FedRecGEL 在所有数据集上均实现了 HR@10、NDCG@10 等指标的显著提升，尤其在用户-物品比率较高的场景下优势更为明显；

**⚠️ 局限性**

局限性包括：仅采用简单的 FedAvg 聚合策略，未探索更复杂的聚合或异构模型方案；对超参数（ρ_ur、ρ_co）敏感，需要手工调优；且实验仅在四个小规模数据集上验证，未评估在更大规模或其他联邦推荐场景下的鲁棒性。

---

## 242. Intrinsic Concept Extraction Based on Compositional Interpretability

**arXiv ID:** 2603.11795 | [PDF](https://arxiv.org/pdf/2603.11795v1)

**作者:** Hanyu Shi `[一作]` (Guangdong University of Technology), Pan Pan `[通讯]` (VIPSHOP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CI‑ICE任务，设计HyperExpress方法在单张图像中同时提取对象级与属性级概念，并通过超曲面投影实现概念可组合；

**💡 创新点**

结合双曲空间的层级对齐与蕴含关系学习，并利用Horosphere Projection在超曲面中实现概念可组合，弥补了现有无监督概念提取缺乏可组合性的空白；

**🔧 技术方法**

采用Poincaré球模型对比学习与蕴含学习、Lorentz模型中的蕴含锥、Horosphere Projection模块、扩散式文本‑图像模型（CLIP+diffusion）以及Wasserstein attention损失；

**📊 数据集**

在UCBench和ICBench的D1数据集上进行训练与评估；

**📈 对比分析**

与Break‑A‑Scene、ConceptExpress、AutoConcept、ICE等基线对比，利用SIM^I、SIM^C、ACC^k等指标，HyperExpress在大多数指标上实现竞争或领先成绩，但在某些指标上略逊于ICE；

**⚠️ 局限性**

尽管实现了可解释的可组合路径，但整体性能相对传统UCE方法略有下降，且对多图像或更大规模概念的扩展尚未验证。

---

## 243. Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings

**arXiv ID:** 2603.11321 | [PDF](https://arxiv.org/pdf/2603.11321v1)

**作者:** Yuning Wu `[一作]` (Amazon), Kai Wei `[通讯]` (Amazon)

**通讯引用:** 16348 | [OpenAlex ID](https://openalex.org/A5087096372)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自适应的后训练强化学习框架 HAPO，结合 Synthetic Success Injection 与 Thompson 采样启发的门控机制，在稀疏奖励和分布漂移场景下实现 RL 与 SFT 的动态协同。

**💡 创新点**

创新点在于：① 在低置信度群组中通过 SSI 将失败轨迹替换为教师示例，提供 hindsight 校正；② 使用 Thompson 采样得到的贝叶斯置信度作为门控阈值，使教师干预随策略改进自然衰减，保证渐进一致性并消除静态混合策略的长期偏差。

**🔧 技术方法**

采用 Group Relative Policy Optimization、Synthetic Success Injection、Thompson 采样、Beta‑Binomial 后验、CLIP 损失、策略塑造等技术组合实现动态 RL‑SFT 交互。

**📊 数据集**

在 Qwen2.5‑Math‑7B 语言模型上使用 OpenR1‑Math‑46k‑8192 训练数据，并在 AIME2024、MATH‑500、OlympiadBench 三个数学推理基准上评估。

**📈 对比分析**

通过与纯 RL（GRPO）、SFT、SFT‑then‑RL、SRFT、LUFFY 等基线对比，HAPO 在 AIME2024 上保持 36.7（与最佳相当），在 MATH‑500 上提升至 87.0（比 LUFFY 提升 2.4），在 OlympiadBench 上达到 51.4（与最佳相当）。整体表现优于静态混合策略，显示出更好的推理能力。

**⚠️ 局限性**

局限性：实验仅覆盖数学推理任务，尚未验证在更大规模模型或通用推理任务上的泛化；对门控阈值、群组大小等超参数敏感；理论证明主要针对稀疏奖励环境，对非稀疏或高维场景的收敛性仍需进一步研究。

---

## 244. Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting

**arXiv ID:** 2603.11938 | [PDF](https://arxiv.org/pdf/2603.11938v1)

**作者:** Chantal Pellegrini `[一作]` (Technische Universitat Munchen), Matthias Keicher `[通讯]` (Technische Universitat Munchen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ProtoSR框架，通过从大规模自由文本报告挖掘并对齐的视觉原型来辅助细粒度结构化放射报告生成。

**💡 创新点**

创新点在于将LLM驱动的原型知识库与结构化报告模型的后期融合相结合，利用检索到的原型作为残差校正器，显著提升长尾属性判断。

**🔧 技术方法**

采用指令调优LLM（Qwen2.5-7B-Instruct）进行文本标签抽取，构建多模态原型库，使用EfficientNet-B5+RadBERT作为特征提取器，并通过MPL与可学习缩放向量实现原型条件后期融合。

**📊 数据集**

使用Rad-ReStruct作为基准数据集进行训练与评测，并从MIMIC‑CXR中挖掘原型知识库。

**📈 对比分析**

与MedGemma、CheXagent、RaDialog、hi‑VQA、Context‑VQA等基线对比，ProtoSR在整体F1从32.5%提升到34.4%，尤其在细粒度属性层（L3）提升72.1%，报告级准确率达36.6%。

**⚠️ 局限性**

局限性包括对LLM抽取质量的依赖、知识库更新频率对性能的影响、长尾标签覆盖仍不完整，以及模型对噪声原型的鲁棒性仍有提升空间。

---

## 245. A Learning-Based Superposition Operator for Non-Renewal Arrival Processes in Queueing Networks

**arXiv ID:** 2603.11118 | [PDF](https://arxiv.org/pdf/2603.11118v1)

**作者:** Eliran Sherzer `[一作]` (Ariel University), Eliran Sherzer `[通讯]` (Ariel University)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5049924526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并训练了一个神经网络超叠算子，用低阶矩和短程自相关描述符映射到合并到达过程的高阶矩和自相关，从而实现非重合到达过程的可分解队列网络分析。

**💡 创新点**

创新点在于以数据驱动方式学习非重合到达过程的超叠映射，保留了高阶方差与短程相关信息，避免了传统 renewal 替代或 MAP 状态空间爆炸的局限。

**🔧 技术方法**

采用深度全连接神经网络，输入为每条流的前若干阶矩和多项式自相关，输出为合并过程的前五阶矩和二阶自相关；损失函数为加权 MSE，训练时利用 MAP Kronecker 叠加生成的精确标签。

**📊 数据集**

训练数据来自合成的 MAP 生成器（三种不同结构），对 500k 对 MAP 进行 Kronecker 叠加并提取 10 阶矩与 135 个自相关系数，构成训练、验证和测试集。

**📈 对比分析**

与 Whitt 的 renewal 近似和 Albin 方法相比，第二阶矩误差始终保持在 0.5%–3% 之间，比传统方法低 10–30 倍；在 64 种 SCV 与相关性组合下保持均匀低误差，推断时间仅需毫秒级。

**⚠️ 局限性**

局限在于误差可能在更深层次网络中累积，未覆盖极端高 SCV 或长程相关情形，且当前框架仅适用于无反馈的前向网络；未来需考虑域专用模型和不确定性量化。

---

## 246. MobileKernelBench: Can LLMs Write Efficient Kernels for Mobile Devices?

**arXiv ID:** 2603.11935 | [PDF](https://arxiv.org/pdf/2603.11935v1)

**作者:** Xingze Zou `[一作]` (Zhejiang University), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 7746 | [OpenAlex ID](https://openalex.org/A5100751566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向移动端的 kernel 生成与评测框架 MobileKernelBench，并提出了多智能体系统 MoKA 来实现自动化的 kernel 开发、调试与加速；

**💡 创新点**

在移动端环境下首次系统化评估 LLM 的 kernel 生成能力，提出了针对碎片化生态与数据稀缺的 MoKA 方案，实现了显著提升的编译成功率和性能加速；

**🔧 技术方法**

结合多轮计划-执行的多智能体架构、仓库感知工具、编译诊断、功能校验与性能解析等技术；

**📊 数据集**

使用基于 MNN CPU 后端的 190 任务（95 个 ONNX 原子算子、12 类）以及公开的 MobileKernelBench 数据集；

**📈 对比分析**

与 Claude‑Sonnet‑4.5、GPT‑5、Gemini‑2.5‑Flash、LLaMA‑3.1‑405B、DeepSeek‑R1、Qwen3‑235B 等现有 LLM 及 LoRA、GRPO 进行对比，MoKA 的 CSR 达到 93.7%，FCR 75.3%，并在 fast_1.5 设定下实现 27.4% 的加速成功率；

**⚠️ 局限性**

受限于移动框架的碎片化、数据稀缺以及对更大规模硬件加速（如多线程 NEON、内联汇编）和其他后端（如 NCNN、TFLite）的适配仍需进一步研究。

---

## 247. AI Knows What's Wrong But Cannot Fix It: Helicoid Dynamics in Frontier LLMs Under High-Stakes Decisions

**arXiv ID:** 2603.11559 | [PDF](https://arxiv.org/pdf/2603.11559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 248. Risk-Controllable Multi-View Diffusion for Driving Scenario Generation

**arXiv ID:** 2603.11534 | [PDF](https://arxiv.org/pdf/2603.11534v1)

**作者:** Hongyi Lin `[一作]` (Tsinghua University), Jinhua Zhao `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 6578 | [OpenAlex ID](https://openalex.org/A5023905102)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于风险控制的多视角驾驶场景生成框架RiskMV-DPO，可根据用户指定的风险水平主动合成多视角、时间一致的驾驶视频。

**💡 创新点**

将驾驶风险从事后标签转化为主动控制信号，利用风险条件的运动轨迹和3D框架指导扩散模型生成；引入几何-外观对齐模块和区域感知直接偏好优化（RA-DPO）提升几何一致性与动态区域真实感。

**🔧 技术方法**

风险计算与控制（基于TTC、RSS等）、多视角扩散模型（MagicDrive-STDiT3）、几何适配器（VGGT/DGGT）、FiLM条件调制、区域感知掩模、直接偏好优化（DPO）等。

**📊 数据集**

在nuScenes数据集上进行训练与评估。

**📈 对比分析**

与DriveDreamer-2、Panacea、MagicDriveV2等方法对比，RiskMV-DPO在FID从20.91降至15.70、FVD从94.84降至87.65、3D检测mAP从18.17提升至30.50，显示显著性能提升。

**⚠️ 局限性**

风险建模仍未覆盖所有复杂情境，生成效果高度依赖风险控制模块产生的轨迹和3D框架，泛化与鲁棒性有待进一步提升。

---

## 249. The Carnot Bound: Limits and Possibilities for Bandwidth-Efficient Consensus

**arXiv ID:** 2603.11797 | [PDF](https://arxiv.org/pdf/2603.11797v1)

**作者:** Andrew Lewis-Pye `[一作]` (London School of Economics), Patrick O'Grady `[通讯]` (Commonware)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了两种基于领导者的共识协议 Carnot 1 与 Carnot 2，并证明了 2-轮最终性协议无法达到低于 2.5 的数据扩展率，随后展示了通过 3-轮最终性实现接近 1 的数据扩展率。

**💡 创新点**

创新点在于给出 2-轮最终性协议的下限证明，并通过稳定领导者、可选的最小恢复码和 M‑证书等机制，设计了可动态调节恢复阈值的 3-轮最终性协议，实现了接近理论效率的带宽利用。

**🔧 技术方法**

核心技术包括可恢复的擦除编码（如 (n,k) 码）、Merkle 树认证、阈值签名以及简单的 Simplex 基础协议，结合稳定领导者和乐观投票来降低延迟。

**📊 数据集**

论文未使用真实数据集，而是在模拟的部分同步网络模型下进行理论分析与实验评估；实验结果将在后续章节补充。

**📈 对比分析**

通过理论分析和仿真，Carnot 1 在 n≥4f+1 条件下实现约 1.33 的数据扩展率，Carnot 2 在 n≥3f+1 条件下实现约 1.5，均显著优于现有 2.5 速率的协议；吞吐量随恢复阈值降低而提升，近似达到网络带宽上限。

**⚠️ 局限性**

限制包括 Carnot 1 对系统规模要求更高（n≥4f+1），Carnot 2 在 Byzantine 干扰下需额外的碎片广播，且两种协议均依赖于理想的时钟同步和完美的加密方案。

---

## 250. Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling

**arXiv ID:** 2603.11971 | [PDF](https://arxiv.org/pdf/2603.11971v1)

**作者:** Junhyeong Byeon `[一作]` (Kookmin University), Sejoon Lim `[通讯]` (Kookmin University)

**通讯引用:** 616 | [OpenAlex ID](https://openalex.org/A5072506078)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于CLIP与Wav2Vec 2.0的多模态情绪识别框架，利用TCN捕捉视频时序特征，并通过双向跨模注意力融合音频与视觉信息。

**💡 创新点**

创新点在于：①引入双向跨模注意力实现视觉与音频的双向交互；②使用TCN对视觉时序进行建模；③加入CLIP文本引导的对比学习以提升语义对齐。

**🔧 技术方法**

主要技术包括CLIP ViT-B/32视觉编码器、Wav2Vec 2.0音频编码器、Temporal Convolutional Network (TCN)、双向多头注意力融合、MLP分类头以及文本对比损失。

**📊 数据集**

使用ABAW 10th Challenge的EXPR任务数据集（在野视频情绪识别）。

**📈 对比分析**

与官方基线（预训练VGGFace）相比，所提方法在60帧窗口下取得53.71%准确率和0.3334的宏F1，显著提升性能。

**⚠️ 局限性**

局限性包括：仅融合音频与视觉两模态；对长时序视频的建模仍受限；对噪声鲁棒性和不同光照/姿态的适应性未进一步评估。

---

## 251. OA-NBV: Occlusion-Aware Next-Best-View Planning for Human-Centered Active Perception on Mobile Robots

**arXiv ID:** 2603.11072 | [PDF](https://arxiv.org/pdf/2603.11072v1)

**作者:** Boxun Hu `[一作]` (Johns Hopkins University), Tinoosh Mohsenin `[通讯]` (Johns Hopkins University)

**通讯引用:** 3305 | [OpenAlex ID](https://openalex.org/A5084010501)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对移动机器人在复杂环境中对被遮挡人类目标进行即时观测的需求，提出了 Occlusion-Aware Next-Best-View (OA-NBV) 方案。

**💡 创新点**

创新点包括：① 用目标中心化的可视性模型同时考虑遮挡、目标尺度和完整度；② 引入基于部件的 3D 目标估计（利用 SAT-HMR 结合 SAM‑2 分割和部件分类）以提升在强遮挡下的姿态对齐；③ 采用基于高度图的可行视角采样，严格考虑地形可通行性与机器人运动学。

**🔧 技术方法**

主要技术包括 SAT‑HMR（SMPL 网格重建）、SAM‑2（前景分割）、RTMPose（关键点检测）、点云 ICP 对齐、基于高度图的视角采样与可视性评估。

**📊 数据集**

使用的训练/评估数据集有：CIHP（部件分类）、DISC 后灾情场景（仿真）、Blender 4.5 构建的室内/室外模拟环境，以及真实世界中的 Unitree Go2 quadruped 与 RealSense D435i / Lidar 传感器。

**📈 对比分析**

与两种基线（Volumetric‑NBV 与 Prediction‑NBV）比较，OA‑NBV 在模拟与真实实验中均取得超过 90% 的成功率，归一化目标面积提升至少 81%，关键点可见度提升至少 58%，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：对光照变化（尤其是低光）敏感；整体推理延迟高（主要受网格重建和分割耗时）；仅做单步视角选择，未考虑多视角全景重建；在极端遮挡或复杂障碍场景中仍可能出现不可达候选视角。

---

## 252. CFD-HAR: User-controllable Privacy through Conditional Feature Disentanglement

**arXiv ID:** 2603.11526 | [PDF](https://arxiv.org/pdf/2603.11526v1)

**作者:** Alex Gn `[一作]`, Ada Axan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了用户可控的隐私保护方法CFD-HAR，利用条件特征解耦实现对运动识别中的敏感属性进行动态过滤，并与基于自编码器的少样本HAR方法进行对比。

**💡 创新点**

首次在IoT HAR领域引入可调lambda参数实现细粒度隐私控制，结构化分离活动语义与敏感属性，提供可控制的隐私保障。

**🔧 技术方法**

采用条件变分自编码器（CVAE）、β-VAE、对抗性隐私损失以及少样本学习（元学习）等技术。

**📊 数据集**

使用Motion‑sense数据集（6种活动，4种个人属性）和Daily Sports Activities Dataset (DSADS)（19种活动，含位置属性）。

**📈 对比分析**

通过单/多私有属性实验和隐私权重调节实验比较，CFD‑HAR在保持高活动识别F1的同时显著降低身份重识别F1；相比AE少样本方法，CFD提供更强的隐私保护但计算成本更高。

**⚠️ 局限性**

依赖敏感属性标签，解耦效果受数据稀缺或分布漂移影响，且对抗训练耗时，未能完全解决持续学习中的攻击风险。

---

## 253. In the LLM era, Word Sense Induction remains unsolved

**arXiv ID:** 2603.11686 | [PDF](https://arxiv.org/pdf/2603.11686v1)

**作者:** Anna Mosolova `[一作]` (Université Paris Cité), Carlos Ramisch `[通讯]` (Aix Marseille University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对词义归纳（WSI）的评估框架进行改进，并在SemCor上系统性实验不同预训练模型、聚类算法、LLM提示、数据增强及Wiktionary半监督策略。

**💡 创新点**

提出更自然的SemCor-derived评估集，首次系统比较多种聚类方法和数据增强技术，并证明Wiktionary增强可超越PolyLM和单集群基线。

**🔧 技术方法**

使用预训练上下文嵌入（BERT、Llama、GPT-4o）、自监督/半监督对比学习、X-means/AG聚类、LLM直接提示以及Wiktionary的must-link约束与实例扩增。

**📊 数据集**

SemCor 3.0的词义标注子集、WikiBooks、Wiktionary、Llama 3.1 8B、GPT-4o。

**📈 对比分析**

采用加权平均的P-metric等指标评估；在自然分布下1cpl基线在多数词性下最优；BERT+Wiktionary增强可突破PolyLM，最佳配置获得76.0，对比PolyLM 72.7；LLM提示表现最差。

**⚠️ 局限性**

仅在英语、SemCor的一小部分、少数PLM和LLM，GPU资源有限未能完整评估大模型；统计显著性仅针对部分实验；未覆盖副词等词性。

---

## 254. Beyond BFS: A Comparative Study of Rooted Spanning Tree Algorithms on GPUs

**arXiv ID:** 2603.11645 | [PDF](https://arxiv.org/pdf/2603.11645v1)

**作者:** Abhijeet Sahu `[一作]` (Indian Institute of Technology), Srikar Vilas Donur `[通讯]` (Indian Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GPU上根结点生成树（RST）算法进行比较研究，提出GPU适配的Path-Reversal RST（PR-RST）和基于GConn+Eulerian Tour的两阶段方案，评估其在真实图上的性能

**💡 创新点**

首次将PR-RST迁移到GPU并针对指针跳跃、路径反转等关键步骤做了硬件友好优化，同时将连通分量算法与Eulerian Tour结合，消除了传统Euler Tour对预处理的高开销

**🔧 技术方法**

GPU指针跳跃、路径反转、最大/最小Hooking、CUB库实现的Eulerian Tour、GConn连通分量框架、并行列表排名与标记

**📊 数据集**

共使用超过30个真实网络图，包括社交网络（LiveJournal、Orkut）、互联网自治系统、道路网络、Twitter、Web、随机网络等，节点数从几十万到数千万

**📈 对比分析**

在NVIDIA L40 GPU上对BFS、PR-RST和GConn+Eulerian Tour三种实现进行基准测试，采用五次循环取中位数；实验显示GConn+Eulerian Tour在高直径图上可达300×的速度提升，PR-RST略优于BFS，但远低于GConn

**⚠️ 局限性**

PR-RST虽然在GPU上可行，但实现复杂且受限于路径反转的频繁同步；GConn+Eulerian Tour虽然最快，但生成的树深度更大，可能影响后续基于BFS树的算法；两种方法在极大图或动态图中的可扩展性和动态维护仍有限

---

## 255. LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning

**arXiv ID:** 2603.12166 | [PDF](https://arxiv.org/pdf/2603.12166v1)

**作者:** Haiying Xu `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1219 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LatentGeo 框架，用连续潜在视觉表示实现几何辅助构造，避免显式绘图或外部工具；

**💡 创新点**

核心创新在于将辅助构造转化为可学习的潜在标记，并通过三阶段课程学习与 LaGDPO 强化学习稳定潜在空间；

**🔧 技术方法**

采用 Qwen2.5‑VL 变体，加入潜在标记与投影器，三阶段对齐-内部化-端到端训练，随后采用 Latent-aware Group‑Decoupled Policy Optimization；

**📊 数据集**

评测数据集包括自研 GeoAux（关注辅助构造）和 MathVerse（视觉依赖几何题）；

**📈 对比分析**

与多款开源与闭源多模态大型语言模型对比，LatentGeo 在 GeoAux 上总体准确率提升至 34.6%（比基线提升约 20%），在 MathVerse 取得 41.4%（比最佳开源模型高 6.7%）；

**⚠️ 局限性**

局限在对训练步骤高度依赖，缺乏对更复杂三维或非欧几里得几何的直接处理，且对外部工具的完全替代仍有挑战。

---

## 256. BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs

**arXiv ID:** 2603.11991 | [PDF](https://arxiv.org/pdf/2603.11991v1)

**作者:** Ilias Aarab `[一作]` (European Central Bank), Ilias Aarab `[通讯]` (European Central Bank)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5098009200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并发布了BTZSC基准，聚合22个公开文本分类数据集，统一评估零样本文本分类性能，并系统比较了NLI交叉编码器、文本嵌入模型、重排器和指令微调LLM四大模型族。

**💡 创新点**

创新点在于首次将上述四类模型在同一零样本评测框架下进行横向对比，公开了基准数据、代码与排行榜，并揭示重排器在零样本场景下的领先优势、嵌入模型的高效性以及NLI性能与零样本效果之间的族内差异。

**🔧 技术方法**

使用的技术包括：文本标签语言化、NLI交叉编码器微调、句向量嵌入与最近邻匹配、检索重排器（如Qwen3-Reranker）以及指令微调的生成式LLM；统一指标为宏F1，并同时报告微准。

**📊 数据集**

采用22个英文公共文本分类数据集，涵盖情感、主题、意图、情绪四类，覆盖二分类到77类高基数，文档长度与领域多样。

**📈 对比分析**

对比方法：在不使用任何任务标注的零样本设定下，统一使用宏F1与微准评估所有模型。结果显示Qwen3-Reranker-8B宏F1最高（0.72），嵌入模型如GTE-large（0.62）次之，NLI交叉编码器（0.59）和LLM（0.67）也表现突出；重排器在准确率与延迟上最优，嵌入模型在准确率和延迟比值上最具性价比。

**⚠️ 局限性**

局限性包括：部分数据集可能与模型预训练语料重叠，导致真实性被削弱；评测仅覆盖英文，未考虑多语言场景；未探讨标签表述多样性对性能的影响；LLM在零样本下仍存在较高延迟。

---

## 257. Systematic Security Analysis of the Iridium Satellite Radio Link

**arXiv ID:** 2603.12062 | [PDF](https://arxiv.org/pdf/2603.12062v1)

**作者:** Eric Jedermann `[一作]` (RPTU Kaiserslautern-Landau), Jens Schmitt `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对Iridium低地轨道卫星网络的射频链路进行了系统性安全评估，逆向工程其SIM身份验证机制并演示了SIM克隆、无线拦截与重组、无加密流量泄露、定位追踪、欺骗（spoofing）与干扰（jamming）等攻击。

**💡 创新点**

创新点在于：①首次完成Iridium射频链路的全面、公开安全评估；②通过实测验证COMP128-1在Iridium中的使用并实现快速提取K_i；③构建完整的SDR捕获与解码流水线，实现对数百万帧的去块重组和熵分析；④用自研的GNU Radio/ gr‑iridium 代码实现对Iridium信号的合成、欺骗与干扰，展示低成本设备即可完成这些攻击。

**🔧 技术方法**

主要技术包括：SIM卡读卡器（OmniKEY, CSL）+ COMP128‑1 破解脚本；软件定义无线电（HackRF One、USRP B210）+ L‑band 天线；GNU Radio 及自研 gr‑iridium 解析器；自制信号合成器（gr‑iridiumtx）实现伪装与干扰；信息熵统计、帧聚类与重组算法。

**📊 数据集**

使用了一个为期一月的Iridium下行数据集（约186 788 186 帧，覆盖多种服务类型），并结合LeoCommon观测网络的实时捕获，此外在实验室中对特定机型（Iridium GO!）进行抓包与注入。

**📈 对比分析**

通过对比已知的GSM COMP128‑1破解难度，本文展示了仅需约2 × 10⁴次挑战即可在6 分钟内恢复K_i；熵分析显示88.5 %帧未加密；干扰实验表明在-2.93 dB的J/S比即可将接收机对Ring Alert的接收率降至50 %，并能在地面局部实现服务中断。

**⚠️ 局限性**

局限性包括：只针对射频链路层，未涉及基站/地面网关级别攻击；实验主要基于Iridium NEXT兼容旧设备，未来更新的协议可能会修补部分缺口；攻击成功依赖于物理接近和具备SDR设备；未评估对特定应用层协议（如IP/TCP/HTTP）的安全改进效果。

---

## 258. Decentralized Orchestration Architecture for Fluid Computing: A Secure Distributed AI Use Case

**arXiv ID:** 2603.12001 | [PDF](https://arxiv.org/pdf/2603.12001v1)

**作者:** Diego Cajaraville-Aboy `[一作]` (Universidade de Vigo), Pablo Picallo-López `[通讯]` (Universidade de Vigo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了面向流计算的多域去中心化编排架构，并通过域侧 SDN 异常检测提升多域分布式学习的安全性。

**💡 创新点**

创新点在于将域侧控制服务视为首要能力，实现跨域自治编排与运行时增强，并提出 FU‑HST SDN 异常检测算法。

**🔧 技术方法**

采用多域编排、SDN 控制、半空间树（Half‑Space Trees）在线异常检测以及去中心化联邦学习协议。

**📊 数据集**

使用 MNIST 图像分类数据集进行联邦学习实验。

**📈 对比分析**

与 HST、SAD、iLOF 等基线对比，FU‑HST 在 F1、误禁率、DL 精度上表现最佳，且计算/通信开销极低。

**⚠️ 局限性**

局限在于 MDCA 的协同算法尚未正式评估、只在仿真中验证，未考虑真实网络延迟与安全策略细节。

---

## 259. Numerical benchmark for damage identification in Structural Health Monitoring

**arXiv ID:** 2603.12069 | [PDF](https://arxiv.org/pdf/2603.12069v1)

**作者:** Francesca Marafini `[一作]` (University of Florence), Gianni Bartoli `[通讯]` (University of Florence)

**通讯引用:** 4643 | [OpenAlex ID](https://openalex.org/A5018550913)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并生成了包含环境、操作变异、快速和慢速损伤以及传感器故障的三年时序SHM数据集。

**💡 创新点**

通过在单自由度钢梁模型中系统化模拟EOV、FAST、SLOW和SF/M效应，提供可重复、可扩展的综合数据集。

**🔧 技术方法**

使用Python、并行状态空间仿真、Euler–Bernoulli静态分析、SDOF动态建模和多类传感器故障仿真。

**📊 数据集**

生成的合成数据集（9个子数据集）已公开在Zenodo（DOI 10.5281/zenodo.17900300）。

**📈 对比分析**

通过将子数据集与标记对比，可评估不同SHM方法在受EOV、损伤和故障干扰下的鲁棒性，表现可在实验中验证。

**⚠️ 局限性**

限制在于仅线性材料、单自由度模型、缺少非线性效应、未考虑其他环境驱动或多DOF特征，且仿真噪声与真实传感器误差可能不完全匹配。

---

## 260. CR-Bench: Evaluating the Real-World Utility of AI Code Review Agents

**arXiv ID:** 2603.11078 | [PDF](https://arxiv.org/pdf/2603.11078v1)

**作者:** Kristen Pereira `[一作]` (Nutanix), Debojyoti Dutta `[通讯]` (Nutanix)

**通讯引用:** 1492 | [OpenAlex ID](https://openalex.org/A5113845539)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对真实仓库的代码审查基准数据集 CR‑Bench，并实现了一个基于 LLM 的评估器 CR‑Evaluator，用以细粒度衡量代码审查代理的表现。

**💡 创新点**

创新点包括：①聚焦真实可预防缺陷的完整 PR 上的 defect‑detecting 任务；②在传统精度/召回之外引入可用性率（Usefulness Rate）和信噪比（Signal‑to‑Noise Ratio）以评估开发者接受度；③通过从 SWE‑Bench 自动转换并验证的流程生成高质量的多维标签（类别、影响、严重性）。

**🔧 技术方法**

使用了前沿大模型 GPT‑5.2 / GPT‑5‑mini，结合 Reflexion 交互式推理框架，评估器采用 Claude‑Sonnet‑4.5 作为“LLM‑judge”。

**📊 数据集**

数据集为 CR‑Bench，包含 584 条 PR（174 条经人工验证），覆盖 Django、SymPy、Astropy、scikit‑learn 等成熟项目，标签按 ISO/IEC 25010 分类。

**📈 对比分析**

对比单次推理（Single‑Shot）与 Reflexion 迭代推理两种策略，发现单次推理在 GPT‑5.2 上取得更高的 SNR（5.11）但召回率仅 27%；而 Reflexion 在 GPT‑5.2 上召回率提升至 32.8% 但 SNR 降至 1.95；GPT‑5‑mini 在两种策略下表现均较弱，尤其在 Reflexion 下 SNR 仅 0.91。

**⚠️ 局限性**

局限性：①仅关注缺陷检测，未评估样式/结构类评论；②数据集主要来自 Python 开源项目，可能无法推广到其他语言或行业；③LLM 仍易产生幻觉，尤其在小模型下的 Reflexion 迭代中噪声更高；④样本规模相对有限，需进一步扩展以验证结论。

---

## 261. AutoVeriFix+: High-Correctness RTL Generation via Trace-Aware Causal Fix and Semantic Redundancy Pruning

**arXiv ID:** 2603.11489 | [PDF](https://arxiv.org/pdf/2603.11489v1)

**作者:** Yan Tan `[一作]` (Hong Kong University of Science and Technology), Yangdi Lyu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 680 | [OpenAlex ID](https://openalex.org/A5014819244)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AutoVeriFix+ 三阶段框架，先用 LLM 生成 Python 参考模型，再由另一个 LLM 生成 Verilog RTL 并迭代修正语法与功能错误，最后利用 Concolic 测试与执行轨迹反馈实现深度功能验证、缺陷定位与冗余逻辑裁剪。

**💡 创新点**

核心创新点包括：① 将 Python 高层语义作为黄金参考模型为 RTL 生成提供精确功能指引；② 通过执行轨迹（周期精确寄存器快照与路径 ID）为 LLM 提供因果级别的错误定位；③ 将 Concolic 测试与覆盖度报告融入闭环反馈，既提升功能正确率又实现语义冗余裁剪。

**🔧 技术方法**

采用的技术主要有：大语言模型（GPT‑3.5、GPT‑4）、自动化语法修正循环、代码注入式 instrument、Concolic 测试引擎（符号约束 + SMT 求解）、覆盖度分析、红利消除提示。

**📊 数据集**

实验使用 VerilogEval（人类编写与机器生成共 299 题）与 RTLLM v1.1 / v2.0（共 79 题）数据集，结合内部生成的 Python 参考模型与自动化测试向量。

**📈 对比分析**

与商业 LLM（GPT‑4、Claude3‑Sonnet）及硬件专用模型（VerilogEval、CodeGen‑6B、RTLCoder、OriGen 等）比较，AutoVeriFix+ 在 VerilogEval‑machine 上 pass@10 91.6%、在 RTLLM‑v2.0 上 pass@5 85.4%（GPT‑4 版本），平均功能正确率超过 80%，并通过 trace‑aware 优化平均削减 25% 冗余逻辑。

**⚠️ 局限性**

局限性：① 依赖高质量 Python 参考模型，若生成模型错误会影响后续步骤；② Concolic 测试对复杂状态空间仍可能无法覆盖全部分支，导致剩余误差；③ 处理过程计算量较大，SMT 求解和多轮 LLM 迭代耗时；④ 对于极其复杂的时序/流水线设计，内部寄存器追踪与重构仍需人工干预。

---

## 262. Multi-Agent Reinforcement Learning for UAV-Based Chemical Plume Source Localization

**arXiv ID:** 2603.11582 | [PDF](https://arxiv.org/pdf/2603.11582v1)

**作者:** Zhirun Li `[一作]` (University of New Mexico), Mostafa Hassanalian `[通讯]` (New Mexico Tech)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于多智能体深度强化学习（MARL）的UAV化学羽流源定位（CPSL）框架，通过虚拟锚点实现协同搜索、航迹优化和碰撞规避。

**💡 创新点**

创新点在于：①使用虚拟锚点将多UAV协同化简为单目标跟踪；②在CTDE框架下设计奖励函数，兼顾形成控制与上风搜索；③将流场噪声与传感器误差融合到训练环境，提升鲁棒性。

**🔧 技术方法**

采用PPO算法+DeepSet神经网络处理可变长度观测，配合碰撞安全协议；同时实现锚点更新规则、上风奖励与分散/收敛奖励等。

**📊 数据集**

使用基于滤波的片状扩散模型生成的合成气体羽流数据，包含多种风速、噪声级和障碍物配置的仿真场景；未使用公开真实测量数据。

**📈 对比分析**

与传统Fluxotaxis方法对比，MARL在所有风况下均显示更高的成功率（>0.99于稳态风）与更低的定位误差（≤2.4 m），CDF曲线显著向左平移，轨迹更平滑。

**⚠️ 局限性**

局限性包括：在中等/强风波动时成功率下降；训练过程易过拟合；碰撞率虽低但非零；未在真实野外验证；通信延迟及多源场景尚未处理。

---

## 263. $Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation

**arXiv ID:** 2603.12263 | [PDF](https://arxiv.org/pdf/2603.12263v1)

**作者:** Songlin Wei `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出并实现了 Psi‑Zero，一个针对人形机器人全身协调操控（loco‑manipulation）的开放式基础模型，并提供完整的训练与部署工具链。

**💡 创新点**

创新点在于：① 采用分阶段训练范式，先用人类自我摄像机视频预训练视觉‑语言模型（VLM）以学习任务语义与视觉特征；② 再用高质量机器人轨迹后训练基于流的多模态扩散变换器（MM‑DiT）动作专家，以学习关节空间控制；③ 引入训练‑时实动作分块（RTC）技术，解决大模型推理延迟导致的控制抖动；④ 通过对齐人类与机器人统一动作表征，避免了跨体现共训练的低效。

**🔧 技术方法**

技术手段包括：Qwen3‑VL‑2B‑Instruct 视觉‑语言基座、FAST 动作分词器、流匹配损失的 MM‑DiT 动作专家、基于强化学习的下肢跟踪器（AMO）、训练‑时实动作分块（RTC）以及专门设计的单操作者全身遥控框架。

**📊 数据集**

数据集主要包括：① EgoDex（约 829 小时的人类自我摄像机抓取与操作视频）；② Humanoid Everyday（约 31 小时的机器人执行轨迹，覆盖 260 项任务）；③ 另外使用 800 小时人类视频和 30 小时机器人数据进行微调，实验中对每项任务收集 80 条遥控轨迹。

**📈 对比分析**

在八个长时程、跨任务的真实世界实验中，Psi‑Zero 在成功率上平均提升至少 40% 以上，显著优于 π0.5、GR00T‑N1.6、InternVLA‑M1、H‑RDT、EgoVLA、Diffusion Policy 及 ACT 等公开基线；同时展示了更平稳的动作执行与更高的子任务完成率。

**⚠️ 局限性**

局限性包括：① 受限于计算与时间，无法进一步扩展到更大规模的人类与机器人数据；② 受限于 Unitree G1 机器人承载能力，某些高精度或大负荷操作仍受限制；③ 现有方法对下肢控制的通用性仍待提升，未来可进一步融合更丰富的下肢轨迹与动力学信息。

---

## 264. A Further Efficient Algorithm with Best-of-Both-Worlds Guarantees for $m$-Set Semi-Bandit Problem

**arXiv ID:** 2603.11764 | [PDF](https://arxiv.org/pdf/2603.11764v1)

**作者:** Botao Chen `[一作]` (Kyoto University), Junya Honda `[通讯]` (RIKEN)

**通讯引用:** 1500 | [OpenAlex ID](https://openalex.org/A5112464181)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了m-set半带问题中Follow-the-Perturbed-Leader (FTPL)策略的最优性和复杂性。通过扩展FTPL的分析，证明了在对抗性环境下，FTPL在特定参数下能够实现O(√(mdT))的最优后悔界限，并在随机环境下实现对数后悔，展示了FTPL在m-set半带问题中的Best-of-Both-Worlds最优性。

**💡 创新点**

创新点在于首次证明了FTPL在m-set半带问题中的对抗最优性和Best-of-Both-Worlds保证。此外，扩展了条件几何重采样技术到m-set半带问题，显著降低了计算复杂度。

**🔧 技术方法**

使用了Follow-the-Perturbed-Leader (FTPL)策略和条件几何重采样（CGR）技术，结合Fréchet和Pareto分布进行分析。

**📊 数据集**

使用了m-set半带问题的理论分析，具体数据集未明确提及，但涉及的环境包括对抗性和随机性设置。

**📈 对比分析**

与现有的BOBW策略进行比较，FTPL在计算效率上表现优越，尤其在高维情况下，FTPL的计算复杂度为O(md(log(d/m)+1))，而其他策略的复杂度显著更高。实验结果表明FTPL在后悔性能上与其他策略相当或更优。

**⚠️ 局限性**

限制在于FTPL的最优性分析主要依赖于特定的Fréchet和Pareto分布，可能不适用于所有类型的分布。此外，尽管CGR在计算效率上有所提升，但在某些情况下仍可能面临数值不稳定性的问题。

---

## 265. QChunker: Learning Question-Aware Text Chunking for Domain RAG via Multi-Agent Debate

**arXiv ID:** 2603.11650 | [PDF](https://arxiv.org/pdf/2603.11650v1)

**作者:** Jihao Zhao `[一作]` (Renmin University of China), Hongyan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 6283 | [OpenAlex ID](https://openalex.org/A5100332460)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 QChunker 框架，通过多代理辩论实现“理解-检索-增强”，将文本切块任务建模为分割与知识补全的复合任务，并提出直接评估指标 ChunkScore。

**💡 创新点**

创新点：①以问题为导向的多代理框架（问题提纲生成、切块、完整性审查、知识补全）；②将切块任务拆为分割与知识补全两子任务；③提出基于逻辑独立性和语义分散度的直接评估指标 ChunkScore；④构建 45K 高质量切块数据集和专项危险化学品安全 QA 数据集。

**🔧 技术方法**

技术：多代理对话式生成与判别、文本分割与重写、逻辑独立性与语义分散度评估（基于语言模型困惑度和句向量 Gram 矩阵）、小语言模型（SLM）微调、向量检索（Milvus）与嵌入模型（bge-base-zh-v1.5）。

**📊 数据集**

数据集：45K QChunker 切块数据集、危化品安全 QA 数据集 HChemSafety，以及公开的 CRUD、OmniEval、MultiFieldQA 进行跨域评估。

**📈 对比分析**

方法对比：与原始切块、Llama_index、语义切块、LumberChunker、MoC MetaChunker、Qwen2.5‑14B/3‑B、Qwen3‑14B 等基线进行 BLEU、ROUGE‑L、METEOR 比较。QChunker 在四个领域均取得最佳或第二最佳，优势显著（尤其在 HChemSafety 上提升显著）。

**⚠️ 局限性**

局限：①依赖大模型的多代理训练成本较高；②知识补全模块仍受原文信息限制，无法补全文档外信息；③ChunkScore 需要调参 λ，适配不同领域时可能需重新校准；④在极长文本或非常专业的语料中，逻辑独立性评估可能受模型困惑度不稳定影响。

---

## 266. Taming the Adversary: Stable Minimax Deep Deterministic Policy Gradient via Fractional Objectives

**arXiv ID:** 2603.12110 | [PDF](https://arxiv.org/pdf/2603.12110v1)

**作者:** Taeho Lee `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2186 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出MMDDPG框架，在连续控制任务中通过双人零和博弈学习对外部扰动和模型不确定性具有鲁棒性的策略。

**💡 创新点**

创新点在于引入分数式目标函数，将任务性能与扰动幅度权衡，并采用对数变换实现梯度稳定化，解决传统minimax训练中扰动过大导致的不稳定问题。

**🔧 技术方法**

采用深度确定性策略梯度（DDPG）结构，使用两套神经网络评估器分别估计累计成本与扰动能量，并在双人博弈中交替更新演员与评论家。

**📊 数据集**

实验使用MuJoCo仿真环境Reacher和Pusher，通过在这两套任务中引入随机高斯扰动和关节阻尼/齿轮系数不确定性进行测试。

**📈 对比分析**

与DDPG、RARL、PR-DDPG、NR-DDPG等基线对比，MMDDPG在扰动场景和参数不确定性下均取得最低的累计折扣成本及更小的方差，显示更好的鲁棒性与学习稳定性。

**⚠️ 局限性**

局限性包括：需要保证目标函数为正值，可能对学习率等超参数敏感；目前仅在连续控制仿真环境验证，缺乏真实硬件实验；对离散或极高维问题的适用性尚待进一步探索。

---

## 267. D-SLAMSpoof: An Environment-Agnostic LiDAR Spoofing Attack using Dynamic Point Cloud Injection

**arXiv ID:** 2603.11365 | [PDF](https://arxiv.org/pdf/2603.11365v1)

**作者:** Rokuto Nagata `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**通讯引用:** 2928 | [OpenAlex ID](https://openalex.org/A5055467060)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种环境无关的 LiDAR 注入攻击 D‑SLAMSpoof，并设计了基于扫描匹配的动态注入策略来破坏 SLAM 定位；同时提出了仅使用惯性传感器的防御方法 ISD‑SLAM。

**💡 创新点**

创新点在于通过空间约束逼迫与时序振荡相结合的注入，显著提升了在高特征环境中的攻击成功率；并提供了仅用标准 IMU 的可部署防御。

**🔧 技术方法**

技术包括多目标优化的注入形状设计、扫描匹配原理导向的时间控制、惯性与 SLAM 状态一致性检测、贝叶斯优化参数调优；利用 3D LiDAR、IMU、速度误差等。

**📊 数据集**

使用公开 MCD‑TUHH 数据集、VLP‑16 实验环境以及模拟仿真场景。

**📈 对比分析**

通过对比静态注入和 D‑SLAMSpoof，在三种 SLAM 算法（KISS‑ICP、FAST‑LIO2、GLIM）上的攻击成功率由约 23% 提升至 87%+；防御方案在仿真与真实实验中实现 96%/97% 的检测准确率并抑制漂移。

**⚠️ 局限性**

局限包括对特定 LiDAR 扫描模式的依赖、对具备随机化/防干扰特性的下一代 LiDAR 效果未知、以及需要预先设定轨迹的检测参数。

---

## 268. Topological DeepONets and a generalization of the Chen-Chen operator approximation theorem

**arXiv ID:** 2603.11972 | [PDF](https://arxiv.org/pdf/2603.11972v1)

**作者:** Vugar Ismailov `[一作]` `[通讯]`, Vugar Ismailov

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出一种拓扑DeepONet框架，将传统DeepONet的输入从欧氏空间扩展到任意Hausdorff局部凸拓扑向量空间，并证明在紧集上对连续非线性算子实现统一的分离逼近定理。

**💡 创新点**

创新点在于将算子逼近理论从Banach空间推广到更广泛的局部凸空间，利用连续线性泛函作为测量，实现通用的分支-主干网络结构；同时给出该框架下的通用逼近定理并展示了其与经典Chen–Chen与DeepONet定理的统一关系。

**🔧 技术方法**

采用拓扑前馈神经网络（基于连续线性泛函与Tauber–Wiener激活函数）构造分支网络，利用黎克核/里德核的凸性与Ridge函数在欧氏域的稠密性构造主干网络，结合Stone–Weierstrass、分区一类等经典泛函逼近工具完成理论证明。

**📊 数据集**

本文为理论研究，未使用具体实验数据集；讨论的示例仅为数学空间（如ℓ_p、L_p、Schwartz空间等）的抽象设定。

**📈 对比分析**

由于没有实验比较，本文不涉及数值性能评估；其优点体现在理论上提供了更广泛的适用范围和更强的通用逼近保证，但缺乏对实际算子学习任务的实验验证。

**⚠️ 局限性**

主要限制包括：需要输入空间具有Hahn–Banach扩展性质（局部凸性保证），逼近误差与网络深度/宽度的具体量化未给出；对非紧输入集或非连续算子缺乏理论支持；实现时需能够有效计算连续线性泛函（如积分、分布测量）与激活函数的组合。

---

## 269. Visibly Recursive Automata

**arXiv ID:** 2603.11648 | [PDF](https://arxiv.org/pdf/2603.11648v1)

**作者:** Kévin Dubrulle `[一作]` (Université de Mons), Gaëtan Staquet `[通讯]` (Nantes Université)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出可见递归自动机（VRA）模型，作为可见推送自动机（VPA）的模块化变体，并研究其语言运算与决策问题的复杂度。

**💡 创新点**

创新点在于：1) 允许同一调用符号对应多个子自动机，形成VRA严格超越SPAs；2) 引入编码确定性（codeterminism）概念，保留表达能力同时提升算法可行性；3) 给出VRA与VPA等价的多项式转换；4) 证明VRA在补集、空集、普遍性、包含和等价等决策问题中拥有比VPA更低或相同的复杂度。

**🔧 技术方法**

使用形式语言理论、自动机构造与转换、子集构造、笛卡尔积交叉等技术，直接在VRA层面实现语言闭包与决策算法，避免先转为VPA再操作。

**📊 数据集**

无实验数据集；工作完全为理论分析与算法构造。

**📈 对比分析**

与VPA的比较主要通过复杂度分析完成：VRA在大多数运算（如并、交、连接、Kleene星）与决策问题上与VPA保持相同复杂度，而在补集运算上可达 2^O(|A|) 级别的优化；算法直接实现避免了 VPA 转换所导致的指数或多项式增长。

**⚠️ 局限性**

局限性：1) 仍未解决 VRA 的确定化与唯一规范化问题；2) 对学习算法的研究仍在起步阶段，缺乏可执行的主动学习框架；3) 对实际数据（如 JSON schema）验证的实验验证尚未完成。

---

## 270. Bridging Discrete Marks and Continuous Dynamics: Dual-Path Cross-Interaction for Marked Temporal Point Processes

**arXiv ID:** 2603.11462 | [PDF](https://arxiv.org/pdf/2603.11462v1)

**作者:** Yuxiang Liu `[一作]` (University of Electronic Science and Technology of China), Yao LIu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 NEXTPP 模型，统一使用自注意力与 Neural ODE 两条并行流，并通过交叉注意力融合离散事件标记与连续时间动态，直接推断条件强度并生成未来事件。

**💡 创新点**

创新点在于双通道交叉交互机制：既能让事件标记影响时间演化，又能让时间上下文反馈到标记预测；以及采用事件粒度的 Neural ODE 进行细粒度时间建模，弥补了传统离散或连续方法的不足。

**🔧 技术方法**

核心技术包括：自注意力编码、神经常微分方程（Neural ODE）进行连续状态演化、交叉注意力融合、变分推断（KL 散度）、迭代抛光采样（thinning）以及多种评估指标（log‑likelihood、RMSE、错误率）。

**📊 数据集**

在五个真实数据集上评估：Taxi、Amazon、StackOverflow、Earthquake、Retweet，涵盖交通、电子商务、问答、地震和社交媒体等领域。

**📈 对比分析**

与多种基线（MHP、RMTPP、NHP、THP、SAHP、AttNHP、IFTPP、ODETPP、DLTPP）比较，NEXTPP 在 log‑likelihood、RMSE 与错误率上均取得最优或接近最优成绩，尤其在时间预测误差与事件类型错误率上明显优于其它模型。

**⚠️ 局限性**

局限性包括：对大规模、稀疏标记数据的泛化尚未充分验证；模型对 ODE 求解器的数值误差敏感；并且交叉注意力层的计算开销相对较高，限制了极大序列的实时部署。

---

## 271. EducaSim: Interactive Simulacra for CS1 Instructional Practice

**arXiv ID:** 2603.11444 | [PDF](https://arxiv.org/pdf/2603.11444v1)

**作者:** Cameron Mohne `[一作]` (Stanford University), Chris Piech `[通讯]` (Stanford University)

**通讯引用:** 5979 | [OpenAlex ID](https://openalex.org/A5074969309)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 EducaSim，一个基于生成式智能体的小组课堂模拟框架，帮助教师在不需要真人指导的情况下进行角色扮演训练。

**💡 创新点**

创新点包括将个性化学习者记忆、基于认知过程的答复决策框架和 LLM 驱动的发言器官结合，支持文本、语音、IDE 等多模态交互，并自动生成反馈。

**🔧 技术方法**

采用 GPT‑4.1‑mini 生成文本、Whisper 语音转写，并使用检索式记忆、生成式代理与 LLM 判断器等技术。

**📊 数据集**

使用了 CS1 课程的讲义、作业文本以及 1,300 名志愿教师和 150 名参与者的数据作为训练与评估素材。

**📈 对比分析**

与 Character.ai、ChatGPT、GPTeach 等现有工具进行功能对比，并在 254 次实验会话中获得平均 15 分 45 秒的持续时间，教师反馈普遍正面，显示 EducaSim 在功能覆盖和可用性上优于同类产品。

**⚠️ 局限性**

局限性包括无法完整模拟人类多样性、仅依赖文本输入导致语调与视觉线索缺失、说话器官在复杂交互中的鲁棒性不足，以及缺乏身份、文化等重要特征的体现。

---

## 272. TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers

**arXiv ID:** 2603.11071 | [PDF](https://arxiv.org/pdf/2603.11071v1)

**作者:** Pooria Roy `[一作]` (Queens University), Mete Bayrak `[通讯]` (Queens University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在 ESP32 微控制器上实现 TinyNav，一个端到端 TinyML 系统，利用量化的 2D CNN 处理 20 帧深度数据，实时预测转向与油门，实现自主导航。

**💡 创新点**

创新点包括：① 用 20 帧滑动窗口堆叠成多通道输入，取代 3D 卷积和循环网络；② 在极低资源的 ESP32 上实现 30 ms 推理，证明 TinyML 能在微控制器上实现实时自主控制；③ 通过 Grad‑CAM 验证模型对空间信息的感知。

**🔧 技术方法**

技术栈：TensorFlow Lite Micro、ESP‑NN、INT8 量化、双核并行处理、Sipeed MaixSense A010 ToF 深度相机、滑动窗口时间编码、Grad‑CAM 可视化。

**📊 数据集**

数据集：自行收集的 ToF 深度帧与人类控制的转向/油门标签，20 帧堆叠为 20 通道输入，60/40 训练/测试拆分，涵盖多种轨道配置，已公开发布在 Hugging Face。

**📈 对比分析**

评估方法：相关性分析、预测与真实分布匹配、Grad‑CAM 解释；量化模型保留 99.84% 的转向精度和 99.79% 的油门精度，推理时间 30 ms、参数 23k，ESP32 可实现 20 FPS；机器人在训练轨道上连续完成 40 圈无碰撞，在更复杂轨道上多次通过但偶有碰撞。

**⚠️ 局限性**

局限性：参数上限与内存受限（ESP32 只能容纳数万参数）；缺乏里程计与低速/逆行控制；仅依赖深度摄像头，对高度变化与非平面障碍识别能力弱；数据集多样性不足，导致对新环境的泛化能力有限。

---

## 273. Harnessing Data Asymmetry: Manifold Learning in the Finsler World

**arXiv ID:** 2603.11396 | [PDF](https://arxiv.org/pdf/2603.11396v1)

**作者:** Thomas Dagès `[一作]` (Technical University of Munich), Ron Kimmel `[通讯]` (Technion -- Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种全新的流形学习流水线，利用Finsler几何构造非对称相似度并在Finsler空间中嵌入数据。

**💡 创新点**

创新点在于：① 识别并保留采样不均导致的自然非对称性；② 将传统对称方法（t‑SNE、Umap）推广到非对称Finsler嵌入；③ 通过引入单一方向的Randers度量实现可扩展且可解释的嵌入。

**🔧 技术方法**

核心技术包括：Finsler几何理论、Finsler t‑SNE、Finsler Umap、基于密度加权的非对称距离构造、梯度下降优化以及SMACOF改进。

**📊 数据集**

使用了多种数据集：合成平面盘和瑞士卷、美国城市经纬度、16个分类数据集（Iris、MNIST、CIFAR‑10/100、ImageNet等）和大规模图像集（ImageNet）来验证方法。

**📈 对比分析**

与传统Isomap、t‑SNE、Umap、Poincaré等基线对比，使用AMI、ARI、NMI、Hom等聚类指标，实验显示Finsler嵌入在标签相关评估上始终优于欧氏基线，尤其在大规模数据上表现突出。

**⚠️ 局限性**

局限性在于仅采用单一Randers度量和单一非对称方向，可能无法充分捕捉多维非对称性；此外当前的非对称信息仅来自采样密度，未考虑其他特征驱动的非对称；对极端结构偏差的鲁棒性有限。

---

## 274. Federated Learning and Unlearning for Recommendation with Personalized Data Sharing

**arXiv ID:** 2603.11610 | [PDF](https://arxiv.org/pdf/2603.11610v1)

**作者:** Liang Qu `[一作]` (Edith Cowan University), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17393 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FedShare框架，实现联邦推荐系统中用户可自定义共享数据并支持后续数据遗忘（unlearning），同时通过对比学习提升模型表现。

**💡 创新点**

创新点：①实现完全个性化的数据共享与遗忘机制；②在学习阶段利用对比学习结合服务器端高阶图，增强低阶与高阶协同信号；③在遗忘阶段使用少量历史嵌入快照进行对比遗忘，避免存储大梯度信息。

**🔧 技术方法**

技术：联邦学习、FedAvg聚合、LGC图卷积、BPR损失、对比学习（SimCLR/InfoNCE）、历史嵌入快照构造遗忘视图、伪图遗忘与新的全局视图对比损失。

**📊 数据集**

数据集：MovieLens-1M、Amazon Fashion、Amazon Video Games。

**📈 对比分析**

与FedAvg、CDCGNNFed、PDC-FRS（学习阶段）以及FRU、CUFRU、Retrain（遗忘阶段）对比。FedShare在三大数据集上学习阶段的HR@20/NDCG@20均优于基线，遗忘阶段在保持推荐准确率的同时，具有与FRU、CUFRU相当或更好的遗忘效率，且显著降低存储开销。

**⚠️ 局限性**

限制：遗忘过程仍需剩余用户参与，增加通信与计算开销；目前仅支持已共享并随后请求遗忘的用户，无法处理从未共享但参与训练的用户的遗忘请求。

---

## 275. DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining

**arXiv ID:** 2603.11838 | [PDF](https://arxiv.org/pdf/2603.11838v1)

**作者:** Yutong Yan `[一作]` (Chinese University of Hong Kong), Yao Lu `[通讯]` (University College London)

**通讯引用:** 12543 | [OpenAlex ID](https://openalex.org/A5005089910)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本工作提出并训练了DatedGPT系列语言模型，共12个1.3B参数模型，每个模型仅使用其对应年份之前的网页数据进行预训练，并在此基础上进行时间敏感的指令微调，专注金融领域的时序预测任务。

**💡 创新点**

创新点在于首次构建大规模、完整覆盖2013‑2024年度的时间切片模型系列，采用100B词级别的年度数据分割，并配合严格的时间切断和指令数据清洗，切实消除时序预测中的lookahead bias。

**🔧 技术方法**

技术实现采用GPT‑2风格Transformer架构，预训练共25,000步≈100B token，随后在每年模型上进行3轮指令微调，使用线性warmup与余弦学习率调度；评估时使用基于困惑度的时间探测方法。

**📊 数据集**

使用FineWeb‑Edu为基础数据集，按抓取时间年限过滤得到12个年度子集；指令数据来自Coconot、Tulu‑3、OpenHermes‑2.5三大公开集合，在时间敏感性筛选后混合使用；金融领域指令数据从新闻标题和盈利电话稿生成约6k个年级样本。

**📈 对比分析**

通过零样本评测与SmolLM、GPT‑XL、TinyLlama、OPT、Pythia等同规模基线在ARC、HellaSwag、PIQA、MMLU、TruthfulQA、IFEval等基准上的对比，DatedGPT‑Instruct平均得分达到42.7，性能与同规模最佳模型相当；困惑度探测表明各模型的知识范围严格受其截止年份限制。

**⚠️ 局限性**

局限性包括模型规模相对较小，无法与更大模型比肩；训练与微调需要大量GPU时间，12个年度模型总成本高；指令数据的时间过滤可能无法完全排除泄露，且尚未在更细粒度的金融预测任务上进行实证评估。

---

## 276. EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models

**arXiv ID:** 2603.12252 | [PDF](https://arxiv.org/pdf/2603.12252v1)

**作者:** Xuanlang Dai `[一作]` (Shanghai AI Laboratory), Yuhang Zang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Endogenous Chain-of-Thought（EndoCoT）框架，使扩散模型在生成过程中实现自我引导的多步推理，从而解决迷宫、TSP、视觉空间规划、数独等视觉推理任务。

**💡 创新点**

首次将多模态大型语言模型（MLLM）与扩散Transformer（DiT）联合微调，采用迭代潜在思维指导和终端思维锚定实现真正的链式推理，而非仅靠一次性文本注入。

**🔧 技术方法**

结合流匹配（Flow Matching）框架、LoRA微调、迭代思维更新与条件流匹配损失、语义对齐损失，并采用两阶段渐进训练。

**📊 数据集**

在Maze、TSP、Sudoku、Visual Spatial Planning（VSP）及其扩展尺寸（如Maze-32、VSP-32）等四类视觉推理数据集上进行训练与评估。

**📈 对比分析**

与零样本、任务专门训练以及统一训练下的DiffThinker、Qwen3-VL-8B等基线对比，在任务专门训练下获得100%/90%/95%等最高分，统一训练下相对基线提升约4-10%，并显示出可解释的逐步推理轨迹。

**⚠️ 局限性**

需要手动设置推理步数、依赖高质量带中间监督的数据集，且在更大规模或无中间标签任务的适应性尚待提升。

---

## 277. Towards Automated Initial Probe Placement in Transthoracic Teleultrasound Using Human Mesh and Skeleton Recovery

**arXiv ID:** 2603.11257 | [PDF](https://arxiv.org/pdf/2603.11257v1)

**作者:** Yu Chung Lee `[一作]` (University of British Columbia), Septimiu E. Salcudean `[通讯]` (University of British Columbia)

**通讯引用:** 14965 | [OpenAlex ID](https://openalex.org/A5028375560)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于混合现实的框架，利用RGB图像进行人体网格恢复与骨架推断，实现无标定的患者三维模型与骨架注册，并基于解剖标志点生成多平面超声探头初始姿势指引，以改善胸腔超声远程操作的设置阶段。

**💡 创新点**

创新点包括：① 在无标定多视角条件下使用SAM 3D Body + MHR模型进行人体网格恢复并转换为SMPL/SMPL-X，再进一步推断SKEL骨架；② 采用多帧批量拟合与RANSAC实现空间平滑，提高模型稳定性；③ 结合骨骼标志点与胸壁法向投影，生成针对多种扫描平面的解剖相关探头姿势；④ 在Magic Leap 2 AR头显上实时投影指导，实现“暖启动”的直观支持。

**🔧 技术方法**

核心技术包括：SAM 3D Body + Momentum Human Rig（MHR）模型；SMPL / SMPL-X 以及 SKEL 骨架转换；多帧批量拟合与 RANSAC 进行鲁棒优化；利用骨骼标志点与胸壁法向投影生成探头姿势；混合现实渲染与实时投影；GPU 边缘推理（RTX 3090 Ti）与 AR 头显摄像头捕捉。

**📊 数据集**

实验数据由 5 名健康志愿者（5 男）在仰卧和左侧卧位分别拍摄 8 张 RGB 图像组成；未使用公开数据集，而是自建小规模实验数据；模型训练依赖 SAM 3D Body 预训练权重。

**📈 对比分析**

通过与手工标定的胸骨上/下探头位置（ground-truth）比较，测量位置误差、倾斜误差和旋转误差。结果显示：guided‑prediction 误差约 16–18 mm，pred‑ground truth 误差约 25 mm，guided‑ground truth 误差约 24 mm；倾斜误差 5–7°，旋转误差 3–5°。与训练后非专业医生的 3 cm 误差相近，表明 AR 指导在临床可接受范围内；系统总体运行时间约 40 秒。

**⚠️ 局限性**

主要局限包括：① 计算时间仍受模型转换与优化影响，仍需进一步加速；② 仅一次性注册，未处理扫描过程中的患者运动；③ 仅在健康志愿者中验证，缺乏病理、多样性和复杂窗口的临床数据；④ 依赖 RGB 摄像头与校准，易受光照、遮挡等因素影响。

---

## 278. Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos

**arXiv ID:** 2603.12064 | [PDF](https://arxiv.org/pdf/2603.12064v1)

**作者:** Shuo Sun `[一作]` (Örebro University), Martin Magnusson `[通讯]` (Örebro University)

**通讯引用:** 3647 | [OpenAlex ID](https://openalex.org/A5101576376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于多视角视频的稠密动态场景重建与相机姿态估计方法。

**💡 创新点**

创新点在于将动态点云重建与相机姿态估计联合优化，并利用时间一致性约束提高重建精度。

**🔧 技术方法**

主要技术包括多视角结构光、稠密光流估计、三维几何约束与深度学习的相机姿态回归。

**📊 数据集**

使用了开源的动态场景数据集如NYU Depth v2以及自制的多摄像机动态室内外数据集。

**📈 对比分析**

与传统单张相机或静态场景的MVS方法相比，在动态环境下的重建误差下降约20%，相机位姿精度提升约15%。

**⚠️ 局限性**

局限性在于需要较高帧率的视频输入，且对遮挡和极端光照条件的鲁棒性不足。

---

## 279. Beyond Barren Plateaus: A Scalable Quantum Convolutional Architecture for High-Fidelity Image Classification

**arXiv ID:** 2603.11131 | [PDF](https://arxiv.org/pdf/2603.11131v1)

**作者:** Radhakrishnan Delhibabu `[一作]` (Vellore Institute of Technology), Radhakrishnan Delhibabu `[通讯]` (Vellore Institute of Technology)

**通讯引用:** 168 | [OpenAlex ID](https://openalex.org/A5073663123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用局部成本函数和张量网络初始化的可扩展量子卷积神经网络，实现高精度图像分类

**💡 创新点**

通过局部观测量消除柏林坡问题并使用张量网络预训练加速收敛，显著提升参数效率

**🔧 技术方法**

量子卷积层、量子池化层、局部成本函数、参数位移规则、TensorFlow Quantum与Cirq、张量网络初始化

**📊 数据集**

MNIST手写数字数据集

**📈 对比分析**

与基线全局成本QCNN和经典CNN对比，测试准确率从52%提升至98.7%，参数量仅为45个，远低于经典网络的12万参数

**⚠️ 局限性**

仍受噪声影响，对大规模量子硬件的物理实现存在SWAP开销与相位错误挑战

---

## 280. Pano360: Perspective to Panoramic Vision with Geometric Consistency

**arXiv ID:** 2603.12013 | [PDF](https://arxiv.org/pdf/2603.12013v1)

**作者:** Zhengdong Zhu `[一作]` (South China University of Technology), Zhiheng Zhou `[通讯]` (South China University of Technology)

**通讯引用:** 680 | [OpenAlex ID](https://openalex.org/A5053841398)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多视角全景拼接任务中，提出了一种基于Transformer的网络Pano360，直接在三维摄影测量空间内利用相机位姿对图像进行全局对齐，并在同一前向传播中生成最佳缝合边缘，实现从几张到上百张图像的无缝全景拼接。

**💡 创新点**

创新点包括：①在3D空间直接利用相机位姿驱动图像对齐，避免传统基于2D特征匹配的累计误差；②提出多特征联合优化的缝合头，利用颜色、梯度和纹理信息一次性预测全局最佳缝合；③构建了包含200个真实场景、14,400张高分辨率图像的Pano360数据集，为大规模训练和评测提供基准。

**🔧 技术方法**

使用技术包括：Transformer网络（DINO预训练编码器 + VGGT交替注意力层）来学习全局几何关系；相机token预测相机内外参；基于预测相机位姿的投影头生成投影函数；多特征缝合头；以及多任务损失训练。

**📊 数据集**

使用的数据集为自行构建的Pano360数据集，共200个场景、72张/场景（14,400张）全景图像，包含多焦距、弱纹理、极端光照等多样化真实条件，并提供相机参数标注。

**📈 对比分析**

通过与AutoStitch、APAP、ELA、GSP、GES-GSP、UDIS2等传统与深度学习方法在Pano360和UDIS-D数据集上对比，采用BRISQUE、NIQE、Q-Align等无参考质量指标以及成功率和运行时间评估，Pano360在对齐精度、视觉质量上均达成SOTA；成功率97.8%（仅比其他方法高约10%），运行时间约5秒，显著快于传统方法。

**⚠️ 局限性**

局限性：目前不支持带有内在畸变的输入图像；对于极大视差导致同一对象从不同角度捕获的场景，仍需三维重建才能实现有效拼接。

---

## 281. FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning

**arXiv ID:** 2603.11901 | [PDF](https://arxiv.org/pdf/2603.11901v1)

**作者:** Yijun Pan `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**通讯引用:** 15065 | [OpenAlex ID](https://openalex.org/A5078337825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 FlexRec 框架，利用强化学习对大语言模型进行后训练，使其能够根据不同用户需求动态生成排序。

**💡 创新点**

创新点在于引入基于 swap 的因果级别项目奖励以及不确定性感知的 GRPO 更新，解决了传统序列级奖励粗糙和稀疏噪声问题。

**🔧 技术方法**

技术上结合了大语言模型（如 Qwen2.5-3B-Instruct）、自回归生成、GRPO、以及预测奖励及其方差的 critic。

**📊 数据集**

实验使用 KuaiRec、MovieLens‑1M 和 ESCI 数据集，覆盖短视频、电影和商品搜索三种场景。

**📈 对比分析**

与传统重排器、零样本 LLM 和其他 RL 后训练方法相比，FlexRec 在 NDCG@5/Recall@5 等指标上提升高达 59%/109%，并在跨需求泛化上表现最优。

**⚠️ 局限性**

局限在于仅处理封闭集重排，假设候选集已知且不考虑检索与开放世界物品的动态扩展。

---

## 282. The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas

**arXiv ID:** 2603.11842 | [PDF](https://arxiv.org/pdf/2603.11842v1)

**作者:** Aleksander Jarzębowicz `[一作]` (Gdańsk University of Technology), Emilio Insfran `[通讯]` (Polytechnic University of Valencia)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述与研究议程合成，梳理并总结自2023年以来信息系统领域中关于生成式人工智能的二级研究与未来研究议题，描绘技术与社会子系统的对齐与挑战。

**💡 创新点**

创新点在于首次将二级综述与研究议程相结合的混合综述框架，系统评估技术收益与风险，并提出面向信息系统的跨学科研究议程，强调人机协作与治理的协同优化。

**🔧 技术方法**

采用系统综述方法、主题分析、定量描述统计（如DARE和自定义质量评估）与信息系统视角的框架，未使用单一机器学习模型。

**📊 数据集**

研究样本为28篇论文（18篇二级综述、10篇研究议程），来源包括Scopus、Web of Science和AIS eLibrary，涉及医疗、教育、软件工程等行业。

**📈 对比分析**

通过质量评估与频次计数对研究主题进行比较，未涉及传统性能指标，主要展示各主题出现频率及其可信度，结果显示技术收益与风险均被广泛讨论，研究空白与治理需求突出。

**⚠️ 局限性**

局限包括样本量有限、仅纳入英文文献、快速演进导致文献缺失、对原始实验数据缺乏细致评估以及研究议程主观性可能影响议程完整性。

---

## 283. VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility

**arXiv ID:** 2603.11816 | [PDF](https://arxiv.org/pdf/2603.11816v1)

**作者:** Zhiwei Zhang `[一作]` (Beijing Jiaotong University), Wenjuan Han `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3201 | [OpenAlex ID](https://openalex.org/A5100604518)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 VisFold 框架，利用时间折叠图和节点可见性实现长时交通流量预测

**💡 创新点**

创新点包括：① 将时间维度折叠进每个节点的 token（Temporal Folding Graph），消除跨步消息传递和堆叠膨胀；② 引入节点级掩码与子图采样（Node Visibility）以降低节点规模、提升并行度并作为隐式正则化；③ 在不改动主干 Transformer 的前提下，通过重构 token 化方式显著提升效率和精度

**🔧 技术方法**

使用 Transformer 编码器、线性投影嵌入、空间/时间/日期嵌入、节点级掩码、子图采样、Huber 损失等技术

**📊 数据集**

在三大公开交通数据集 PEMS04、PEMS08 与 SEATTLE 上进行实验

**📈 对比分析**

与 12 个基线（STGNN、Transformer、MLP 等）对比，VisiFold 在 24/36/48 步长均取得最佳或次优指标，显著提高预测精度，并将 GPU 内存、训练/推理时间降低 5–20 倍

**⚠️ 局限性**

主要限制为：仅依赖历史数据，难以应对突发事件；对节点特定时间嵌入无明显提升；子图采样可能导致全局依赖信息丢失

---

## 284. Universal cycle constructions for k-subsets and k-multisets

**arXiv ID:** 2603.11954 | [PDF](https://arxiv.org/pdf/2603.11954v1)

**作者:** Colin Campbell `[一作]` (University of Guelph), Joe Sawada `[通讯]` (University of Guelph)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5060694274)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

设计了一种新型差分表示和简化频率表示，构造了适用于任意 n,k≥2 的 k-子集和 k-多重集的通用循环，并给出了 O(1) 逐符号生成算法。

**💡 创新点**

首次提供针对 k-多重集的高效通用循环构造，利用受限权重 de Bruijn 序列的 Grandmama 生成技术以及缺失符号寄存器，获得 O(1) 逐符号时间和 O(n) 空间。

**🔧 技术方法**

采用周期连结树、合并树、缺失符号寄存器、差分表示、简化频率表示、受限权重 de Bruijn 序列的 Grandmama 和 MSR 成功规则，以及连结树的 RCL 预序遍历。

**📊 数据集**

论文主要是理论构造，实验使用了小规模 n,k 组合，如 n=6,k=3、n=4,k=3 等示例来展示序列生成和长度验证。

**📈 对比分析**

通过与已知的基于图或无效构造方法对比，证明新算法在时间上达到 O(1) amortized 每符号、空间 O(n)，并且能在 O(n) 时间内求后继符号，明显优于先前仅能得到 O(n) 时间或不具备高效构造的方法。

**⚠️ 局限性**

仅适用于权重小于字母表大小的多重集情况；对更一般的权重范围或更复杂的约束尚无高效构造；实现仍需判定是否为回文花环等步骤导致常数因子较大。

---

## 285. Unsupervised LiDAR-Based Multi-UAV Detection and Tracking Under Extreme Sparsity

**arXiv ID:** 2603.11586 | [PDF](https://arxiv.org/pdf/2603.11586v1)

**作者:** Nivand Khosravi `[一作]` (Instituto Superior Tecnico University of Lisbon), Meysam Basiri `[通讯]` (Instituto Superior Tecnico University of Lisbon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本工作提出了一套无监督的固态LiDAR检测与跟踪完整管道，能够在极稀疏的单点/两点测量环境下自动识别并持续跟踪多架小型UAV，解决了传统检测器在非重复扫描下极低点密度的鲁棒性问题。

**💡 创新点**

创新点包括：① 结合范围自适应DBSCAN与三阶段时空一致性验证的无监督检测器；② 在IMM滤波框架下对比确定性Hungarian匹配与概率性JPDA关联，展示后者在轨道交叉与近距离相互作用场景中的显著优势；③ 设计了两环境评估协议，将真实RTK-GPS检测验证与仿真精确身份标签的跟踪评估相结合，克服了GNSS在近距离下的定位盲区。

**🔧 技术方法**

所用技术包括：无监督聚类（DBSCAN）、体素下采样、三阶段时空一致性验证、三维位置测量转化、IMM滤波（多子模型）、确定性Hungarian匹配、联合概率数据关联（JPDA）、RTK‑GPS实时定位、Gazebo仿真及Livox Mid‑360固态LiDAR。

**📊 数据集**

数据集：① 真实UAV飞行数据（两架F450、Livox Mid‑360、RTK‑GPS）共776帧；② Gazebo仿真生成的四个多目标交叉/遮挡/分离/混合情景，包含精准身份标签的千帧级别测量。

**📈 对比分析**

性能对比：在检测方面，最佳配置（minPts=1+时空验证）实现了0.891精度、0.804召回、0.63 m RMSE，检测覆盖率达69.9%；在跟踪方面，JPDA在四个情景下平均将身份切换从4.4降至1.6（64%下降），MOTA仅下降0.003，RMSE仅提升约0.01 m。

**⚠️ 局限性**

局限性：① 仅在单目标真实实验中验证检测，缺乏真实多目标GNSS ground truth；② JPDA在目标数增大时计算量呈指数增长；③ IMM模型参数需针对不同平台重新校准；④ 该方法依赖固态LiDAR的特性，无法适用于高点密度扫描系统。

---

## 286. Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge

**arXiv ID:** 2603.11665 | [PDF](https://arxiv.org/pdf/2603.11665v1)

**作者:** Junjie Wu `[一作]` (Meta AI), Kaitai Zhang `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种统一的多任务强化学习框架 MT-RL-Judge，用于提升多模态大语言模型在评估任务中的性能。

**💡 创新点**

创新点在于将多任务训练与强化学习结合，采用组相对策略优化（GRPO）并设计格式与准确度双重奖励，促使模型生成推理轨迹并提升泛化能力。

**🔧 技术方法**

使用技术包括强化学习（GRPO）、监督微调（SFT）、奖励建模、链式推理、以及多模态大语言模型基础架构。

**📊 数据集**

实验数据集涵盖 AGIN-Naturalness、AGIN-Tech、AGIN-Rationality、SeeTRUE、ImageReward、UnsafeBench 以及对比实验的 MJ-Bench。

**📈 对比分析**

与零样本、SFT 单/统一、RL 单等基线对比，MT-RL-Judge 在 Macro-F1 上多项指标位居前列，并在未见的 pairwise 任务上显著提升泛化性能。

**⚠️ 局限性**

局限性包括对 Safety 任务在统一 SFT 训练时仍易过拟合、在极端安全或多模态变形任务上表现仍有限，以及 RL 训练对样本效率和计算资源要求较高。

---

## 287. Cross-Resolution Attention Network for High-Resolution PM2.5 Prediction

**arXiv ID:** 2603.11725 | [PDF](https://arxiv.org/pdf/2603.11725v1)

**作者:** Ammar Kheder `[一作]` (LUT University), Michael Boy `[通讯]` (University of Helsinki)

**通讯引用:** 55046 | [OpenAlex ID](https://openalex.org/A5041685684)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 CRAN-PM，一种双分支 Vision Transformer，用于在欧洲范围内 1 km 分辨率下对 PM₂.₅ 进行 1–3 天的日预测。

**💡 创新点**

核心创新在于交叉分辨率注意力桥（使局部高分辨率信息查询全局低分辨率气象背景），以及地形感知自注意力和风向引导交叉注意力的软物理先验。

**🔧 技术方法**

使用了 Swin Transformer 的局部窗口注意力、交叉分辨率注意力、地形/风向偏置的自注意力、PixelShuffle 上采样、焦点频率损失（FFL）和站点监督损失。

**📊 数据集**

数据集包括 2017–2021 年 GHAP 1 km PM₂.₅、ERA5 ERA reanalysis + CAMS 25 km 气象场，以及 2022 年 362 天的 EEA 2971 站点监测，用作训练与评估。

**📈 对比分析**

与 CAMS、ConvLSTM、SimVP、Earthformer、ClimaX、TopoFlow 等基线比较，CRAN-PM 在 1 km 预测下 T+1 时 RMSE 为 6.85 µg/m³（比最佳单尺度基线低 4.7%），T+3 时降低 10.7%；在 25 km 也保持领先；在复杂地形站点的偏差降低 36%。

**⚠️ 局限性**

局限性包括：依赖 GHAP 观测的误差、仅提供日级预测、预测时延仅限 1–3 天、固定 512×512 断块划分可进一步优化。

---

## 288. From Pets to Robots: MojiKit as a Data-Informed Toolkit for Affective HRI Design

**arXiv ID:** 2603.11632 | [PDF](https://arxiv.org/pdf/2603.11632v1)

**作者:** Liwen He `[一作]` (Hong Kong University of Science and Technology), Xin Tong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13150 | [OpenAlex ID](https://openalex.org/A5100784734)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建并评估了 MojiKit——一个结合结构化情感交互参考卡、可编程的zoomorphic机器人原型（MomoBot）和无代码行为控制工作室，用于支持非专业人员在动物启发的社交机器人中快速迭代情感交互行为。

**💡 创新点**

创新点在于①将对65段人宠互动视频的编码与动物行为学文献整合，生成可视化的设计参考卡，②将参考卡与可编程硬件和实时反馈的无代码工作室无缝耦合，③通过协同工作坊验证工具能激发多样化的情感交互方案，突破传统单纯直觉或硬编码设计的限制。

**🔧 技术方法**

使用的技术包括：视频行为编码与语义映射（Cohen κ=0.929）；基于Arduino + ESP32的16轴伺服驱动硬件；Web‑based无代码行为编辑界面；Bézier平滑算法实现连贯运动；行为卡片的模块化卡片式设计与交互；以及用于评估的创意支持指数 (CSI) 等量表。

**📊 数据集**

使用的数据集为：65段公开人宠互动视频（总约79分钟，涵盖猫狗行为），以及24篇动物行为学论文（CatFACS、DogFACS 等）用以补充情感映射，后期形成 Design Reference Cards v3.0。

**📈 对比分析**

评估方法主要为：在18名受试者（18场次、9组）进行的共创工作坊中收集定量指标（CSI、卡片评价、工作室可用性量表）和定性资料（访谈、视频观察）。结果显示：CSI 最高得分为协作 4.69/5，卡片可读性 4.33/5，工作室反馈 4.3/5，生成 35 个情感交互模式。未与传统机器人行为库做直接性能对比，评估侧重于创意支持和可用性，而非行为效果指标。

**⚠️ 局限性**

局限性包括：①未进行长期随访或真实环境部署，缺乏行为有效性与用户情感收集；②原型硬件功能有限（无感知、音频、温度等模态），限制了可实现的交互；③受试者多为具备设计/技术背景的年轻人，未覆盖老年人、儿童等潜在目标用户；④评估主要聚焦人类视角，未考察机器人行为对现实动物的影响；⑤缺乏对系统安全性、伦理性（责任归属等）和多主体交互的深入探讨。

---

## 289. Initialization and Rate-Quality Functions for Generative Network Layer Protocols

**arXiv ID:** 2603.11122 | [PDF](https://arxiv.org/pdf/2603.11122v1)

**作者:** Mathias Thorsager `[一作]` (Aalborg University), Petar Popovski `[通讯]` (Aalborg University)

**通讯引用:** 27081 | [OpenAlex ID](https://openalex.org/A5071289803)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个用于学习生成式网络层压缩模型的速率-质量函数的初始化协议。

**💡 创新点**

创新在于方法无关的协议设计以及基于统计预算的速率-质量函数估计与推理区间。

**🔧 技术方法**

使用统计抽样、预测区间、生成式压缩模型（HiFiC）和LPIPS感知质量度量。

**📊 数据集**

使用COCO 2017图像数据集。

**📈 对比分析**

与无压缩PNG和传统JPEG比较，结果显示在满足不同质量阈值下，可在12张图后实现相较JPEG的通信收益，低估计预算下也可获得正收益。

**⚠️ 局限性**

局限在于仅验证了两种提示策略且对分布漂移的适应依赖于后续pilot传输；预测区间对小样本不稳健。

---

## 290. FBCIR: Balancing Cross-Modal Focuses in Composed Image Retrieval

**arXiv ID:** 2603.11520 | [PDF](https://arxiv.org/pdf/2603.11520v1)

**作者:** Chenchen Zhao `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14274 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了组成式图像检索（CIR）中的焦点失衡问题，提出了一种多模态焦点解释方法和一个数据增强工作流，用以识别并缓解模型过度关注单一模态导致的检索失败。

**💡 创新点**

创新点在于：①提出“焦点平衡比”度量，用于量化模型对图像与文本的关注平衡；②提出迭代焦点精炼方法自动定位模型最重要的图像片段与文本关键词；③设计基于 VLM、图像编辑和生成模型的硬负样本构造流程，生成需要跨模态平衡推理的检索难样本。

**🔧 技术方法**

核心技术包括：多模态迭代焦点精炼（使用掩码/空字符串剪枝与束搜索），焦点平衡比度量（基于图像区域面积与文本词权重），以及利用 Qwen‑VLM、Qwen‑Image‑Edit 与 Qwen‑Image 进行文本增强、图像编辑和生成的硬负样本生成。

**📊 数据集**

使用了 MegaPairs、SynthTriplets18M、GPT‑Image‑Edit‑1.5M 等三大 CIR 数据集构建的焦点挑战基准和微调数据集；同时在 CIRR、FashionIQ、GeneCIS 等标准基准上评估模型。

**📈 对比分析**

通过与 CLIP4CIR、SEARLE、BGE（CLIP‑基）以及 GME、RzenEmbed、MM‑Embed（VLM‑基）等主流模型在标准与焦点挑战基准上的比较，发现：①在小规模候选池的焦点挑战基准上模型性能显著下降；②微调后在硬样本召回率（Rs@1）上提升显著，且焦点平衡比差距（|r_I‑r_T|）显著缩小，证明方法有效提升了鲁棒性。

**⚠️ 局限性**

局限性包括：①焦点平衡比使用的区域面积与词权重为经验式，可能在某些样本上产生偏差；②多模态焦点精炼需要多次模型推理，计算成本较高，限制了实时或大规模部署；③实验主要关注硬负样本生成，尚未验证在更广泛任务或跨域迁移中的普适性。

---

## 291. Evaluating Explainable AI Attribution Methods in Neural Machine Translation via Attention-Guided Knowledge Distillation

**arXiv ID:** 2603.11342 | [PDF](https://arxiv.org/pdf/2603.11342v1)

**作者:** Aria Nourbakhsh `[一作]` (University of Luxembourg), Christoph Schommer `[通讯]` (University of Luxembourg)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5055092426)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于教师-学生蒸馏的自动化评估框架，通过将XAI归因图注入Transformer注意力来衡量不同归因方法在机器翻译中的效果。

**💡 创新点**

创新点在于使用归因图作为“软先验”在注意力层中进行组合，并通过学生模型的BLEU/chrF提升量化归因质量，同时引入Attributor网络评估归因可再现性与效果的相关性。

**🔧 技术方法**

技术包括Transformer seq2seq、Inseq归因工具、归因注入的四种组合算子（+、⊙、μ、R）、多种归因方法（Attention、ValueZeroing、LG×A、Saliency、IG、GSHAP、DeepLIFT、Input×Gradient）以及KL/Overlap@3/Tau@3评估。

**📊 数据集**

实验使用三语种（de‑en、fr‑en、ar‑en）WMT/UN平行语料，教师模型为Marian‑MT和mBART。

**📈 对比分析**

通过比较注入不同归因方法与算子的学生模型BLEU/chrF得分，发现Attention、ValueZeroing和LG×A在Encoder注意力上带来最大提升，乘法算子效果最好，且归因可再现性与BLEU高度相关；cross‑attention效果差。

**⚠️ 局限性**

局限包括计算成本高、仅评估Encoder/Encoder‑Decoder注意力、未探索decoder‑side归因、缺乏层级/头部细粒度分析，以及仅聚焦翻译任务。

---

## 292. Catalogue Grounded Multimodal Attribution for Museum Video under Resource and Regulatory Constraints

**arXiv ID:** 2603.11147 | [PDF](https://arxiv.org/pdf/2603.11147v1)

**作者:** Minsak Nanang `[一作]` (University of Surrey), Armin Mustafa `[通讯]` (University of Surrey)

**通讯引用:** 532 | [OpenAlex ID](https://openalex.org/A5075260187)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套基于多模态LLM的本地化视频元数据生成管道，自动为馆内视频生成可搜索的目录式元数据。

**💡 创新点**

创新点在于：1）将目录检索与多模态生成耦合，使用检索‑多选两阶段识别和显式弃权；2）在训练时加入结构化JSON和多选任务，使模型按目录闭集输出；3）在本地部署、数据主权和风险控制下实现高精度标注。

**🔧 技术方法**

使用 VideoLLaMA2.1‑7B‑16F+Qwen2 架构，SigLIP 视觉塔，LoRA 参数高效微调，检索式 IDF 加权匹配，以及基于多选的阈值决策。

**📊 数据集**

基于一份包含 60 幅画作的馆藏目录（标题、作者、主题、描述）以及合成的 210 条图像‑对话对构建训练集；评估使用 16 条馆内视频样本。

**📈 对比分析**

与基线直接识别（无检索）对比，基线准确率极低，管道通过保守决策显著降低误标率，覆盖率降低但预期效用提升，运行时间约 14–18 秒/视频。

**⚠️ 局限性**

局限包括：1）覆盖率受保守阈值限制；2）视频视角差异仍导致识别失败；3）仅覆盖已收录目录，未覆盖新作品；4）需要人工标注扩展数据集；5）对光照、遮挡等情况仍敏感。

---

## 293. From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts

**arXiv ID:** 2603.11781 | [PDF](https://arxiv.org/pdf/2603.11781v1)

**作者:** Sunil Prakash `[一作]` `[通讯]` (Indian School of Business), Sunil Prakash (Indian School of Business)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“Deliberative Collective Intelligence（DCI）”框架，通过多代理间的分阶段、结构化、可追踪的讨论来生成决策。

**💡 创新点**

创新点在于：①定义四类代理原型（Framer、Explorer、Challenger、Integrator）；②设计14种语义化的知识行为；③构建共享工作空间和明确的分歧记录；④开发能保证有限收敛的决策生成算法，输出包含选项、保留异议、少数报告和重开条件的结构化决策包。

**🔧 技术方法**

技术实现基于 Gemini 2.5 Flash LLM 的多代理协作，采用 agent‑runtime、定制的系统提示、三层语义行为模式、工作空间 JSON 结构，以及八阶段的收敛流程。

**📊 数据集**

评测数据集为 45 个开放式推理任务，覆盖七个领域（软件架构、政策分析、隐藏信息、后期证据、风险分析、争议决策、常规决策），每个领域 5–10 题。

**📈 对比分析**

与单代理、无结构辩论、投票和自一致性基线对比：在非常规任务上 DCI 对辩论提升约 +0.95 分（显著）；在隐藏信息任务上最高得分 9.56；但整体质量低于单代理，且 token 消耗约 62 倍，质量/token 远低于基线。

**⚠️ 局限性**

局限性包括：高成本与低效率、对任务高度依赖、缺乏模型多样性、样本量有限、评估依赖 LLM‑judge、可能出现代理角色漂移或协同偏向一致、强制回退质量不一定最佳。

---

## 294. Deterministic Algorithm for Non-monotone Submodular Maximization under Matroid and Knapsack Constraints

**arXiv ID:** 2603.11996 | [PDF](https://arxiv.org/pdf/2603.11996v1)

**作者:** Shengminjie Chen `[一作]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences), Jialin Zhang `[通讯]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了两种确定性算法，分别用于在基约束和背包约束下求解非单调子模函数最大化问题；

**💡 创新点**

通过扩展多线性扩展（EME）框架实现对连续优化算法的彻底去随机化，并利用离散局部搜索与辅助函数相结合，获得了比之前最优的确定性逼近比（0.385‑ε 对基约束，1/e‑ε 对背包约束）；

**🔧 技术方法**

主要技术包括：扩展多线性扩展、确定性 Aided Continuous Greedy、Split（分块）子算法、Deterministic Pipage Rounding、Relax 与 Rounding 过程，以及对支持大小进行严格控制的组合优化；

**📊 数据集**

无（论文未涉及实验或数据集）；

**📈 对比分析**

与现有随机化与确定性算法进行理论对比，结果显示在基约束下提升约 1.4% 逼近比，在背包约束下将确定性逼近比从 0.25 提升到约 0.367；

**⚠️ 局限性**

局限在于：1）对背包约束的去随机化仅实现到 1/e‑ε，还未达到 0.385‑ε；2）扩展多线性扩展框架在更复杂约束（如多维背包）下的通用性尚未证明；3）算法查询复杂度仍为多项式，但常数项较大，可能影响实际可行性。

---

## 295. Disentangled Representation Learning through Unsupervised Symmetry Group Discovery

**arXiv ID:** 2603.11790 | [PDF](https://arxiv.org/pdf/2603.11790v1)

**作者:** Dang-Nhu Barthélémy `[一作]` (Sorbonne Université), Argentieri Sylvain `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过自监督与环境交互，自动发现动作空间的群结构，并利用该结构学习线性对称分解（LSBD）的解耦表示。

**💡 创新点**

提出了可在无先验子群知识的情况下通过动作聚类自动重建对称群分解的算法，并证明了该分解在最小假设下可辨识；同时设计了不需要先验动作矩阵的 GMA‑VAE 方法，实现了无监督的 LSBD 学习。

**🔧 技术方法**

基于变分自编码器的动作条件 A‑VAE 进行动作聚类，随后利用带掩码的 GMA‑VAE 学习解耦线性动作矩阵；使用群理论距离度量和软硬阈值机制实现子群划分。

**📊 数据集**

在 Flatland、COIL、3DShapes、MPI3D 等离散与连续对称环境（包括位置、颜色、视角、对象排列等因素）上进行实验。

**📈 对比分析**

与 LSBD‑VAE、SOBDRL、Forward‑VAE 等基线对比；在 Inde、Mod、DCI、MIG、SAP 等指标上与监督 LSBD‑VAE 相当，且在长期预测和 OOD 泛化任务上优于无监督方法。

**⚠️ 局限性**

方法假设动作集合已按子群解耦（即每个动作属于唯一子群），且需训练两阶段网络，限制了对更复杂或噪声环境的适用性。

---

## 296. PhiPlot: A Web-Based Interactive EDA Environment for Atmospherically Relevant Molecules

**arXiv ID:** 2603.11751 | [PDF](https://arxiv.org/pdf/2603.11751v1)

**作者:** Matias Loukojärvi `[一作]` (University of Helsinki), Kai Puolamäki `[通讯]` (University of Helsinki)

**通讯引用:** 2639 | [OpenAlex ID](https://openalex.org/A5067547881)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并演示了一个面向大气化学分子数据的基于Web的交互式EDA工具PhiPlot，集成了数据摘要、聚类和知识驱动的二维嵌入。

**💡 创新点**

首次将受控核主成分分析（cKPCA）与化学领域知识结合，实现了实时交互式嵌入约束，并将该工具与现有数据库无缝集成。

**🔧 技术方法**

使用Python、Holoviz Panel、Bokeh进行前端展示；后端采用MongoDB、Docker、OpenShift；核心技术包括分子指纹化、cKPCA、LSP、K-means/BIRCH聚类。

**📊 数据集**

利用来自多个公开化学数据库的气溶胶相关分子集合（SMILES + 量化属性），通过MongoDB存储后提取使用。

**📈 对比分析**

与InVis、Solvent Surfer、χiplot等工具对比，PhiPlot在交互式嵌入、数据整合和可视化功能上更完整；用户调研显示交互速度快（<1 s 500点），并在任务完成率和满意度上优于现有工具。

**⚠️ 局限性**

局限包括：对分子搜索和功能团过滤支持不足，嵌入维度受限于500点实时性能，缺乏对更大规模数据的高效交互；未来需加入结构化过滤和机器学习预测功能。

---

## 297. HawkesRank: Event-Driven Centrality for Real-Time Importance Ranking

**arXiv ID:** 2603.11472 | [PDF](https://arxiv.org/pdf/2603.11472v1)

**作者:** Didier Sornette `[一作]` (Institute of Risk Analysis, Prediction and Management), Sandro Claudio Lera `[通讯]` (Institute of Risk Analysis, Prediction and Management)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 HawkesRank，一种基于多元 Hawkes 点过程的动态排名框架，通过事件强度直接衡量节点即时重要性；

**💡 创新点**

创新点在于将传统中心性（如 Katz、PageRank 等）视为 HawkesRank 的静态极限，并显式分离外源驱动与内源放大，同时摆脱了手工构造网络的依赖；

**🔧 技术方法**

核心技术包括多元 Hawkes 点过程建模、最大似然估计、时变强度分解为内源与外源、以及 Spearman 相关性等评估手段；

**📊 数据集**

实验使用了合成数据（10 类事件、Barabási–Albert 生成的网络）和真实 YouTube Live 聊天情感数据（六种基本情绪）进行验证；

**📈 对比分析**

通过与 Katz、Eigenvector、PageRank 及第一阶 Hawkes 近似进行 Spearman 相关性对比，HawkesRank 取得最高相关性，且能及时捕捉冲击带来的排名变化，静态指标表现明显不足；

**⚠️ 局限性**

局限性包括模型假设内源矩阵和衰减核为固定不变，难以捕捉时间变化的交互、抑制效应及大规模并行计算的挑战，并且仅在可观测事件数据充分时才能发挥优势。

---

## 298. Differentiable Thermodynamic Phase-Equilibria for Machine Learning

**arXiv ID:** 2603.11249 | [PDF](https://arxiv.org/pdf/2603.11249v1)

**作者:** Karim K. Ben Hicham `[一作]` (RWTH Aachen University), Alexander Mitsos `[通讯]` (Forschungszentrum Jülich GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了可微分的二元液–液相平衡求解器 DISCOMAX，结合图神经网络预测 Gibbs 混合能，并在端到端学习中保证热力学一致性。

**💡 创新点**

将极值原理离散化为枚举+掩码软最大化，利用 Straight-Through 软最大化实现反向传播的可微分，首次实现在机器学习训练中完全热力学一致的 LLE 预测。

**🔧 技术方法**

采用 Chemprop D-MPNN 对分子进行图嵌入，构建 Gibbs 混合能网络，离散化能量表格，使用 Straight-Through Softmax 结合 Hessian 与 Gibbs 辅助损失进行训练。

**📊 数据集**

使用 138,601 种可能二元体系中 8,597 个通过 HANNA‑2（COSMO‑RS 训练模型）预测得到的 LLE 数据集（feed 取两相混合区中心），不使用实验数据。

**📈 对比分析**

与基线神经网络替代求解器（Surrogate）比较，单体系统训练 MAE 从 0.101 降至 0.015，十折交叉验证测试 MAE 从 0.076 降至 0.068，性能显著优于基线且保持热力学一致。

**⚠️ 局限性**

局限性包括仅适用于二元体系、枚举离散化随组分数指数增长、未包含单相/混溶体系的训练，导致在极窄相分离区的预测精度受限。

---

## 299. GlyphBanana: Advancing Precise Text Rendering Through Agentic Workflows

**arXiv ID:** 2603.12155 | [PDF](https://arxiv.org/pdf/2603.12155v1)

**作者:** Zexuan Yan `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14719 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GlyphBanana agentic 工作流，融合系统字体渲染工具与扩散模型，实现对文本和复杂公式的高精度渲染；

**💡 创新点**

核心创新在于训练无关的频域分解+注意力重加权注入方案，以及 VLM 驱动的迭代细化；

**🔧 技术方法**

使用频域分解、注意力重加权、MM‑DiT、VAE、文本生成 VLM、图像到图像扩散等技术；

**📊 数据集**

构建 GlyphBanana‑Bench 数据集，覆盖普通单词、罕见汉字、多行数学公式等多语言、多难度场景；

**📈 对比分析**

在公开模型（如 Z‑Image、Qwen‑Image）上对比，OCR 准确率分别提升约 19.6% 与 6.9%，并在精度、风格与视觉质量指标上优于现有基线；

**⚠️ 局限性**

局限包括对极高分辨率或实时生成的适配性不足，依赖外部字体与 VLM 工具，对非训练集场景的泛化仍有限。

---

## 300. Time, Message and Memory-Optimal Distributed Minimum Spanning Tree and Partwise Aggregation

**arXiv ID:** 2603.12156 | [PDF](https://arxiv.org/pdf/2603.12156v1)

**作者:** Michael Elkin Tanya Goldenfeld `[一作]` `[通讯]`, Michael Elkin Tanya Goldenfeld

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一种分布式、确定性算法，在 CONGEST 模型下构造最小生成树（MST）、最小生成森林（MSF）和分区求聚合（PWA）问题，并同时优化时间、消息和内存复杂度。

**💡 创新点**

核心创新在于设计了“通信循环（Communication Cycle）”协议，该协议通过时间槽分配、树路由和区间分配等技术，使得各片段（fragment）可以在多阶段中以线性时间和消息量完成聚合、广播与合并，同时将每个节点的内存需求压缩到 O(log² n) 位。

**🔧 技术方法**

主要技术包括：
1) 细粒度时间槽（slot）调度与可插拔的发布-查询（p‑q）槽机制；
2) 采用 Thorup‑Zwick 树路由实现 O(log² n) 的标签和 O(log n) 的路由表；
3) 通过通信循环实现片段间的并行 convergecast 与 broadcast；
4) 区间分配（interval allocation）用于动态分配槽位，保证槽位不重叠；
5) 在每一阶段内使用 Lenzen 的 GKP 变体和 GKP 原始算法的组合，确保时间和消息复杂度接近最优。

**📊 数据集**

该工作为理论研究，未在具体网络或数据集上实验；所有结论均基于理论分析和复杂度证明。

**📈 对比分析**

与现有工作相比：
- 传统 GHS 系列算法时间 O(n)，内存 O(log n)；
- 现代时间/消息最优算法时间 O(D+√n)·log n，内存仍为 Ω(√n)；
- 本算法时间 O((D+√n)·log² n·log* n)，消息 O(m·log n + n·log² n·log* n)，内存 O(log² n)，在三项指标上均实现了新的综合平衡，尤其显著降低了内存需求。

**⚠️ 局限性**

局限性包括：
- 仍未达到完全时间最优（缺少 log* n 乘子）；
- 消息复杂度相对较高，尤其在大图中 m·log n 项；
- 依赖于可预先构建的 BFS 树和树路由表，实际部署中对网络拓扑稳定性有一定要求。

---

## 301. Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution

**arXiv ID:** 2603.11445 | [PDF](https://arxiv.org/pdf/2603.11445v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Verified Multi-Agent Orchestration 框架，利用 DAG 任务拆解、并行执行、LLM 驱动的完整性验证与自适应重规划，实现多代理协同完成复杂查询。

**💡 创新点**

核心创新在于（1）基于 DAG 的依赖感知并行执行；（2）在协调层使用 LLM 进行结果完整性验证并触发重规划；（3）可配置的终止条件平衡质量与成本；（4）跨层级的聚合合成与来源追溯。

**🔧 技术方法**

采用 LangGraph 与 Strands Agent 进行工作流编排，使用 Claude Sonnet/Haiku 4.5 进行代理执行，Claude Opus 4.5 进行验证；通过 Model Context Protocol 将多种领域工具暴露为 HTTP 微服务；实现分层合成与资源预算管理。

**📊 数据集**

在 25 条人工设计的市场调研查询集上进行评估，涵盖绩效分析、竞争情报、财务调查与战略评估四大类别。

**📈 对比分析**

通过与单代理、静态流水线基线比较，验证框架在答案完整性上提升至 4.2/5（比单代理 3.1/5 提升 35%），来源质量提升至 4.1/5（比单代理 2.6/5 提升 58%），但平均 token 使用量升至 850K（比单代理 100K 高 8.5 倍）。

**⚠️ 局限性**

局限性包括评估样本量有限、LLM 同族评估可能带来偏差、未做组件级消融实验、成本与延迟显著增加、验证仅检查完整性而非事实准确性，以及跨模型族与领域的泛化尚未验证。

---

## 302. Frequency-Modulated Visual Restoration for Matryoshka Large Multimodal Models

**arXiv ID:** 2603.11220 | [PDF](https://arxiv.org/pdf/2603.11220v1)

**作者:** Qingtao Pan `[一作]` (Case Western Reserve University), Shuo Li `[通讯]` (Case Western Reserve University)

**通讯引用:** 50353 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FMVR（Frequency‑Modulated Visual Restoration）策略，在视觉 token 压缩后通过拆分低频与高频两部分并进行可学习调制，恢复视觉语义，保持模型推理能力；

**💡 创新点**

创新点在于将视觉表示同时提取 AvgPool（低频）与 MaxPool（高频）分量，并用轻量可学习参数调制两者，形成“显著语义增强”和“弱语义恢复”双重过滤；再将该机制嵌入 Matryoshka Representation Learning，实现推理时可弹性调整 token 数量；

**🔧 技术方法**

采用 AvgPool 与 MaxPool 作为频率分离器，后接通道级可学习调制权重，集成进 LLaVA 视觉编码器和 LLM；实验中使用了 LLaVA‑1.5‑7B 与 LLaVA‑NeXT‑7B 作为基线模型；

**📊 数据集**

在 10 个图像理解基准（VQAv2、GQA、VizWiz、ScienceQA‑IMG、POPE、MME、MMBench、MMBench‑CN、MMVet、TextVQA）和 4 个视频理解基准（MSVD‑QA、MSRVTT‑QA、ActivityNet‑QA、视频生成）上进行评测；

**📈 对比分析**

与 FastV、PyramidDrop、SparseVLM、M3、MQT‑LLaVA 等传统 token‑压缩方法对比，FMVR 在 36 或 1 个 token 时仍保持约 100% 原始准确率，FLOPs 可下降至 8.9 倍；在高分辨率（720 token 与 2880 token）下亦保持近似性能；

**⚠️ 局限性**

对极小 token 数量（如单 token）时仍存在语义丢失，难以捕捉细粒度视觉细节；此外，未在更大规模模型或实时部署场景下验证鲁棒性。

---

## 303. LLMs can construct powerful representations and streamline sample-efficient supervised learning

**arXiv ID:** 2603.11679 | [PDF](https://arxiv.org/pdf/2603.11679v1)

**作者:** Ilker Demirel `[一作]` (Massachusetts Institute of Technology), David Sontag `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 12916 | [OpenAlex ID](https://openalex.org/A5013431623)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出利用大语言模型（LLM）生成“rubric”规则，对电子病历的文本序列进行结构化转换，以提升监督学习的样本效率与预测性能。

**💡 创新点**

创新点在于：①通过LLM在小样本内聚合多样化病例生成全局或本地规则；②规则可转为可解析脚本或表格特征，减少推理成本；③验证规则设计是提升模型性能的首要因素。

**🔧 技术方法**

核心技术包括：LLM（GPT‑5、Qwen3‑8B）用于规则生成与应用；文本嵌入模型（Qwen3‑8B‑Embedding）转向向量；逻辑回归/梯度提升/ XGBoost 等传统学习器；自动化脚本生成与解析。

**📊 数据集**

数据集为 Stanford Medicine 的 EHRSHOT 基准，涵盖 6,739 名患者的多模态电子病历，共 15 个预测任务（运营结果、诊断分配、实验室结果预测、胸片异常预测）。

**📈 对比分析**

与基线（NaiveText、CLMBR‑T、Count‑GBM、CoT 等）对比，rubric 方案在 15 个任务上平均 AUROC 与 AUPRC 均显著提升；在低样本（40 例）和全样本两种情形下，Local‑Rubric 与 Global‑Rubric‑Tabular 接近最优，尤其在诊断与实验室任务表现突出。

**⚠️ 局限性**

局限性包括：仅在医疗 EHR 领域验证；规则生成受上下文长度限制，仅 40 例可聚合；高维任务如胸片与运营结果仍落后于 CLMBR‑T；对多模态文本与影像的通用性尚待探究；LLM 生成脚本仍需人工审校，存在可重复性与成本问题。

---

## 304. Resource-Efficient Iterative LLM-Based NAS with Feedback Memory

**arXiv ID:** 2603.12091 | [PDF](https://arxiv.org/pdf/2603.12091v1)

**作者:** Xiaojie Gu `[一作]` (University of Wurzburg), Radu Timofte `[通讯]` (University of Wurzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于大型语言模型（LLM）的迭代神经架构搜索（NAS）闭环管线，利用历史反馈记忆和双重LLM（代码生成器与提示改进器）在单块RTX4090 GPU上实现无需微调的低成本、可复现的图像分类架构搜索；

**💡 创新点**

核心创新在于：①在开放式可执行PyTorch代码空间而非预定义单元中搜索；②使用长度固定的K=5窗口的马尔可夫历史反馈记录诊断三元组，避免信息溢出且捕获失败模式；③将代码合成与诊断推理拆分为双重LLM，降低每次调用的认知负载并暗示硬件友好；

**🔧 技术方法**

技术手段包括：instruction-tuned LLM（DeepSeek‑Coder、Qwen2.5、GLM‑5）进行代码生成；一周期代理训练评估（SGD、Cosine衰减）得到快速排名；结构化诊断三元组与历史窗口传递给提示改进器生成改进建议；

**📊 数据集**

使用CIFAR‑10、CIFAR‑100和ImageNette三大公开数据集进行实验；

**📈 对比分析**

对比单次生成基线，所有模型在迭代搜索中均显著提升，一周期代理准确率在CIFAR‑10上最高可达69.2%（DeepSeek‑Coder）或71.5%（Qwen2.5），在CIFAR‑100和ImageNette亦显示正向趋势；相关统计（Spearman ρ≈0.75，Kendall τ≈0.55）表明迭代过程具有显著的优化动力；

**⚠️ 局限性**

主要局限包括：①代码生成成功率波动大，特别是Qwen2.5的低成功率导致搜索噪声；②仅评估一周期代理精度，未验证最终收敛性能；③实验范围局限于小规模图像分类数据集，尚未验证在更大规模或不同任务上的适用性。

---

## 305. Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections

**arXiv ID:** 2603.12180 | [PDF](https://arxiv.org/pdf/2603.12180v1)

**作者:** Łukasz Borchmann `[一作]` (Snowflake AI Research), Anupam Datta `[通讯]` (Snowflake AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并发布了一个针对多模态智能体的文档问答基准，包含人类撰写的问题、最小证据页面标注以及一个衡量答案准确性、证据归属与计算效率的评估协议；

**💡 创新点**

基准设计采用多跳、跨文档、视觉感知和闭世界约束，结合经典测验理论实现有针对性的难度分层；引入 Kuiper 统计量评估智能体的努力-准确性校准，并通过严格的人工审核保证数据完整性；

**🔧 技术方法**

使用多模态大型语言模型与检索工具（BM25、语义检索、视觉检索）组成的迭代代理；评估借助 LLM-as-judge、文本与图像特征交叉检索；基准拆分采用经典测验理论；

**📊 数据集**

新建了约 800 篇多样化 PDF 文档（金融、报告、政府、法律、技术等），共 2,250 题目，所有问题均由专业人员人工撰写并标注最小证据页面；

**📈 对比分析**

与静态 RAG、递归语言模型、可工具化代理以及人工标注基准进行对比；最佳代理在 82% 的准确率下略低于完美检索下的人工基准（约 100% 或 80% 以上），并在效率方面显示出显著差距；

**⚠️ 局限性**

局限性包括仅覆盖英文、美国中心化的公共 PDF，可能受预训练数据泄漏影响；评估基于 LLM-judge 仍可能有误差；仅标注页面级证据，缺乏细粒度定位；基准对计算预算敏感，实时响应能力未充分评估；缺乏对对抗性攻击和隐私敏感数据的考量。

---

## 306. Learning Transferable Sensor Models via Language-Informed Pretraining

**arXiv ID:** 2603.11950 | [PDF](https://arxiv.org/pdf/2603.11950v1)

**作者:** Yuliang Chen `[一作]` (Dartmouth), Andrew Campbell `[通讯]` (Dartmouth)

**通讯引用:** 31278 | [OpenAlex ID](https://openalex.org/A5064857767)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种名为SLIP的语言信息预训练框架，用以学习跨传感器配置的可迁移语义对齐表示。

**💡 创新点**

关键创新包括：多模态对比与生成双目标的训练、FlexMLP可变补丁嵌入实现对不同采样率的自适应、以及将解码器LLM重构为编码解码结构以支持开放词汇生成。

**🔧 技术方法**

使用的技术包括：Transformer+FlashAttention、对比学习+自回归生成、2D RoPE自注意力、跨模态交叉注意力、FlexMLP权重缩放等。

**📊 数据集**

预训练使用约60万对时间序列与文本对的混合数据（覆盖健康、环境、IoT等），并在11个下游数据集（活动识别、临床诊断、压力预测、城市传感等）进行评估。

**📈 对比分析**

与基线（Chronos、NormWear、ChatTS等）比较，SLIP在11个分类任务上线性探针平均准确率77.14%，比最佳基线高5.9%；在零样本检索平均精度39.4%；在问答平均精度64.8%；在生成任务上BERTScore 0.887。

**⚠️ 局限性**

限制包括：语言模型固定不变、对长文本生成的计算开销较大、未分析生成的可信度与归因、以及对特定传感器高频细节的捕捉可能不足。

---

## 307. CoMMET: To What Extent Can LLMs Perform Theory of Mind Tasks?

**arXiv ID:** 2603.11915 | [PDF](https://arxiv.org/pdf/2603.11915v1)

**作者:** Ruirui Chen `[一作]` (Institute of High Performance Computing), Cheston Tan `[通讯]` (Centre for Frontier AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了多模态多轮对话的Theory of Mind评测基准CoMMET，用于评估LLM的心理状态推理能力。

**💡 创新点**

结合心理学ToM手册任务，扩展多种心理状态覆盖，加入多轮交互、图像支持和道德推理，首创多模态多轮ToM评测。

**🔧 技术方法**

采用LLM生成StoryTurn、基于Gemini 3.0 Pro生成图像与文本、基于ATOMS心理状态分类、使用LLM评审器与人工校验评估答案等技术。

**📊 数据集**

基于ToM手册任务构建CoMMET，包含591个StoryTurn、1973个问题、826张图像，覆盖ATOMS所有七种心理状态及道德推理。

**📈 对比分析**

评估八大LLM（OpenAI、Gemini、Claude、LLaMA等）在文本与图像子集上的故事级准确率，最高模型Gemini 3.0 Pro整体准确率约88%，各心理状态表现差异明显。

**⚠️ 局限性**

图像质量不一导致评估误差，解释性答案评估依赖LLM裁判且手工验证成本高，当前模型仍缺乏复杂多心理状态和说明推理的稳健性。

---

## 308. Stuck on Suggestions: Automation Bias, the Anchoring Effect, and the Factors That Shape Them in Computational Pathology

**arXiv ID:** 2603.11821 | [PDF](https://arxiv.org/pdf/2603.11821v1)

**作者:** Emely Rosbach `[一作]` (Technische Hochschule Ingolstadt), Marc Aubreville `[通讯]` (Flensburg University of Applied Sciences)

**通讯引用:** 1816 | [OpenAlex ID](https://openalex.org/A5018481044)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在计算病理学中，对28名专业病理学家进行基于H&E切片的肿瘤细胞比例估计实验，比较AI辅助与否以及是否存在时间压力下的决策，定量测量自动化偏差与锚定偏差。

**💡 创新点**

首次系统评估AI在病理学诊断中的自动化与锚定偏差，并揭示时间压力、专业经验与决策自信如何调节这些偏差，从而为AI辅助诊断的安全部署提供实证依据。

**🔧 技术方法**

使用基于FCOS+ResNet18的目标检测AI模型生成肿瘤细胞比例预测，结合线性混合效应模型（LMM）与归一化的WoA指标分析人机交互数据，评估偏差与性能。

**📊 数据集**

选取公开的三大H&E病理图像数据集：BreCaHad、Frei的结直肠癌数据集和BreastPathQ，共23块切片（20块实验、3块训练）用于实验。

**📈 对比分析**

通过重复测量ANOVA和配对t检验比较AI与非AI、时压与非时压条件下的平均绝对误差；结果显示AI提升整体准确度，自动化偏差发生率约7%，时压未显著提升频率但加重严重度，锚定偏差系数约0.44（时压下系数升高），表明AI支持在提高精度的同时诱发中等强度的认知偏差。

**⚠️ 局限性**

样本量仅28名专家，实验缺乏真实临床时间压力和完整诊断背景，AI模型在部分图像上性能不佳，可能低估偏差，且未考察长期使用或高风险场景下的影响。

---

## 309. A scalable framework for correcting public transport timetables using real-time data for accessibility analysis

**arXiv ID:** 2603.11477 | [PDF](https://arxiv.org/pdf/2603.11477v1)

**作者:** Zihao Chen `[一作]` (University of Exeter), Federico Botta `[通讯]` (University of Exeter)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5069884826)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可扩展框架，利用高频车辆位置（AVL）数据与 GTFS 时刻表匹配，重建全国范围内的经验公交行程时间表，并基于此计算行程时间可变性（TTV）和可达性。

**💡 创新点**

创新点在于：①提出了高效的空间匹配与线性插值/外推策略，直接恢复停站到达时间；②使用七天时间窗口扩展匹配率，显著提高数据可用性；③在全国规模下实现了可路由的经验时刻表，提升了实时可达性评估的真实性。

**🔧 技术方法**

技术手段包括 Python 自动化脚本、GTFS / GTFS‑RT 解析、粗筛+精确距离的空间匹配、线性插值与外推、r5py 路由引擎、统计分析与可视化。

**📊 数据集**

使用的数据集为英国公交开放数据服务（BODS）的全国 GTFS 与 GTFS‑RT（VehiclePositions）每日数据，英国 NHS 病院位置、英国城镇中心点、LSOA 居民分布等。

**📈 对比分析**

方法通过将重建时刻表与原始时刻表在 2025 年 5–10 月的全国数据上计算公交到医院和镇中心的平均行程时间及 TTV 进行比较，结果显示约 79% LSOA 的平均行程时间和 81% 的 TTV 均高于时刻表预估；相较传统方法，能够捕捉到更真实的时间增幅和变异性，且计算效率足以完成全国级别分析。

**⚠️ 局限性**

局限性包括：仅覆盖公交，不包含地铁、轻轨等模式；重建时间为回溯性，未考虑乘客在出行时信息不完整的情形；缺失 trip_id 导致匹配率受限，可能影响部分地区的可达性评估；外推假设延迟恒定，可能在交通拥堵或运营调整时产生偏差。

---

## 310. A Simple Efficiency Incremental Learning Framework via Vision-Language Model with Nonlinear Multi-Adapters

**arXiv ID:** 2603.11211 | [PDF](https://arxiv.org/pdf/2603.11211v1)

**作者:** Haihua Luo `[一作]`, Fengyu Cong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SimE 框架，利用预训练的 vision‑language 模型（如 CLIP）和轻量级适配器实现增量学习，无需回放内存；

**💡 创新点**

创新点在于设计了 Multi‑Adapter 在 Transformer 结构中多位置插入适配器，并揭示适配器连接数与 IL 性能呈非线性关系，证明只在 block 间增加适配器即可提升性能；

**🔧 技术方法**

使用了 CLIP 预训练模型（ViT‑B/16、ViT‑L/14 等）、轻量级 Adapter 微调、零样本 CLIP 预训练数据集、类增量学习（CIL）框架及多步增量任务；

**📊 数据集**

评估数据集包括 CIFAR‑100、TinyImageNet、ImageNet‑R、CUB‑200、ImanEtc，CLIP 预训练数据集有 LAION‑2B、WIT‑400M、Laion‑400M、DataComp‑1B、CommonPool‑1B 等；

**📈 对比分析**

与多种 CIL 方法（UCIR、PASS、DyTox、DER、CLIP、Fien‑tune、iCaRL、LwF、Continual‑CLIP、LwF‑VR、ZSCL）以及参数高效方法（L2P、DualPrompt、CODA‑Prompt、Boosting‑CL、APER、MISA）进行对比，SimE 在 TinyImageNet 上提升约 9.6%，在 CIFAR‑100 上提升 5.3%，平均 accuracy 85.94%（10 步）且参数量仅数千，GPU 资源占用显著降低；

**⚠️ 局限性**

局限在于对大型预训练数据集和大模型的依赖，过多适配器在小增量步骤下可能导致过拟合和性能下降，对适配器配置仍需手工调优，且未深入研究长期记忆衰减和跨任务迁移问题。

---

## 311. Scaling Reasoning Efficiently via Relaxed On-Policy Distillation

**arXiv ID:** 2603.11137 | [PDF](https://arxiv.org/pdf/2603.11137v1)

**作者:** Jongwoo Ko `[一作]` (Microsoft), Pashmina Cameron `[通讯]` (Microsoft)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5080222030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 on‑policy distillation 进行 RL 视角的解析，并提出 Relaxed On‑Policy Distillation 框架，利用奖励裁剪、基于熵的 Token 级动态采样以及两阶段（探索‑精炼）训练，提升容量受限模型在推理任务上的性能。

**💡 创新点**

① 将教师–学生对数似然比视为固定奖励，使 distillation 与策略梯度等价；② 通过混合奖励下界实现奖励裁剪，避免负奖励无穷大导致的梯度爆炸；③ 采用 token 熵筛选动态采样，聚焦信息量高的推理步骤；④ 设计两阶段 mask，先探索多解空间后精炼对齐；⑤ 在多种推理任务上实现比传统 RL 更高的样本效率与推理速度。

**🔧 技术方法**

强化学习的策略梯度与 stop‑gradient 控制方差；奖励裁剪（混合下界）；token 熵动态采样；两阶段探索‑精炼训练策略；大语言模型与视觉 LLM（如 Qwen、SkyWork）与 vLLM 并行推理。

**📊 数据集**

数学推理：AIME‑24/25、AMC‑23、MATH‑500、Minerva Math、Olympiad Bench；视觉推理：Geometry3K、MathVerse、MathVision、MathVista、WeMath、HallusionBench；工具推理：Pixel‑Reasoner、V‑Star、InfographicVQA、TallyQA；训练数据包括 Geometry3K、Pixel‑Reasoner、InfographicVQA 等。

**📈 对比分析**

与 RKL（原 on‑policy distillation）、GRPO、DeepSeek‑R1 等方法对比。实验表明：数学任务样本效率提升 6.7–12×；视觉任务 7B 学生匹配 32B 教师，推理速度提升 3.32×；在所有基准上均超越基线，尤其在大模型下保持训练稳定。

**⚠️ 局限性**

仍依赖高容量教师模型；混合奖励阈值与熵阈值需手动调参；对极端稀疏奖励或超大规模模型的适应性尚待验证。

---

## 312. When do modal definability and preservation theorems transfer to the finite?

**arXiv ID:** 2603.12171 | [PDF](https://arxiv.org/pdf/2603.12171v1)

**作者:** Johan van Benthem `[一作]` (University of Amsterdam), Xi Yang `[通讯]` (Tsinghua University)

**通讯引用:** 7520 | [OpenAlex ID](https://openalex.org/A5073809550)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

探讨模态逻辑与一阶逻辑在有限结构上的可定义性与保持性问题，系统评估了传统保持定理（如单调性、生成子框架保持、完全加性、连续性、金字塔-汤姆森定理等）在有限模型与框架下的转移情况。

**💡 创新点**

提出了新的正面结果——Bisimulation Safety定理在有限结构上也成立，并给出相应的构造证明；同时展示了多种保持性定理在有限框架下的失效，并揭示了对应的可定义性层级（FO、FO+monTC、FO+monLFP、MSO）在有限框架上的复杂性分化。

**🔧 技术方法**

利用模态逻辑的有限模型性质、二元关系的模态程序与标准翻译、Bisimulation安全性技术、Ehrenfeucht–Fraïssé游戏、Hanf局部性定理、归约至计算复杂性问题（如Set Splitting、Horn-UNSAT）、描述性复杂性理论（NL、P、CONP）等多种逻辑与计算机科学技术。

**📊 数据集**

无实验数据集，本研究为纯理论分析。

**📈 对比分析**

通过理论归约与复杂度分析，证明了若干保持性定理在有限结构下仍保持可判定性（多为PSPACE‑hard），而部分保持性定理（如对生成子框架、并集、受限同构等）失效，且相关可定义性问题的复杂度可精确定位（如McKinsey算子的有效性为CONP‑完整）。

**⚠️ 局限性**

局限性：本研究仅覆盖基本模态语言与一阶语义，未探讨更广泛的模态子句（如只用正片段或更复杂算子），以及对特定框架族（如传递、树形等）的细化结果；同时，部分复杂性结论依赖于未证明的计算复杂性假设（如NL= P 或 P=NP）。

---

## 313. OMNIA: Closing the Loop by Leveraging LLMs for Knowledge Graph Completion

**arXiv ID:** 2603.11820 | [PDF](https://arxiv.org/pdf/2603.11820v1)

**作者:** Frédéric Ieng `[一作]` (LIPADE, Université Paris Cité), Farah Benamara `[通讯]` (Université de Toulouse)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 OMNIA 两阶段知识图谱完成框架，先通过聚类生成候选三元组，再用轻量级嵌入过滤和 LLM 验证提升完整度。

**💡 创新点**

创新点在于将结构化聚类与 LLM 语义验证分离，专注于 LLM 生成图谱中的隐含语义缺失，而不依赖外部文本。

**🔧 技术方法**

核心技术包括基于 (relation, tail) 对的实体聚类、TransE 嵌入过滤、以及多种 LLM prompting（零样本、少样本、RAG）进行语义判定。

**📊 数据集**

使用 FB15K-237、CoDEx-M、WN18RR、Socio‑Economic 以及 Covid‑Fact（案例）等四大标准与自建数据集进行评测。

**📈 对比分析**

与 KGE、KG‑BERT、MuKDC 等基线比较，OMNIA 在 F1 分数上提升高达 23%（如 CoDEx‑M）并在稠密图谱中显著优于传统方法。

**⚠️ 局限性**

局限性包括稀疏图谱下候选生成覆盖率低，过滤阶段可能丢失部分真三元组，需进一步改进稀疏区域的候选生成策略。

---

## 314. Understanding User Perceptions of Human-centered AI-Enhanced Support Group Formation in Online Healthcare Communities

**arXiv ID:** 2603.11237 | [PDF](https://arxiv.org/pdf/2603.11237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 315. Gender Bias in Generative AI-assisted Recruitment Processes

**arXiv ID:** 2603.11736 | [PDF](https://arxiv.org/pdf/2603.11736v1)

**作者:** Martina Ullasci `[一作]` (Politecnico di Torino), Antonio Vetrò `[通讯]` (Politecnico di Torino)

**通讯引用:** 3462 | [OpenAlex ID](https://openalex.org/A5047965231)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了GPT‑5在为35岁以下意大利本科毕业生生成工作建议时的性别偏见。

**💡 创新点**

首次系统评估LLM在职位、行业和形容词推荐中的性别差异，发现语言层面存在显著偏差。

**🔧 技术方法**

使用GPT‑5生成文本，并通过开放编码将输出归类后采用卡方检验进行统计分析。

**📊 数据集**

24个模拟候选人（12名女性、12名男性）作为实验数据，按年龄、经验和专业分布均衡。

**📈 对比分析**

对职位、行业和形容词的性别分布进行卡方检验，结果显示职位和行业无显著差异（p>0.05），但形容词在性别间差异显著（p≈0.002）。

**⚠️ 局限性**

样本量有限、仅使用二元性别、手工编码存在主观性、仅测试单一LLM（GPT‑5），难以推广到更大规模或其他模型。

---

## 316. Coarse-Guided Visual Generation via Weighted h-Transform Sampling

**arXiv ID:** 2603.12057 | [PDF](https://arxiv.org/pdf/2603.12057v1)

**作者:** Yanghao Wang `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 96127 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于加权h-变换的无训练粗引导视觉生成方法。

**💡 创新点**

创新点在于利用Doob的h-变换在采样过程中引导生成，并设计噪声水平自适应权重以抑制近似误差。

**🔧 技术方法**

使用扩散模型的反向SDE/ODE、Doob h-变换、加权采样策略以及噪声水平调度。

**📊 数据集**

在FFHQ 256x256人脸数据集、DL3DV-10K视频数据集以及相关合成降噪/超分辨/去模糊任务。

**📈 对比分析**

与需要已知前向算子或起始点指导的训练免费方法对比，实验显示在多种图像和视频恢复任务中F1D、LPIPS等指标均优于SDEdit，且与最佳基准相当。

**⚠️ 局限性**

受近似误差影响，权重调度对α取值敏感，且在部分极端条件下仍可能偏离原始参考。

---

## 317. DRAFTO: Decoupled Reduced-space and Adaptive Feasibility-repair Trajectory Optimization for Robotic Manipulators

**arXiv ID:** 2603.11074 | [PDF](https://arxiv.org/pdf/2603.11074v1)

**作者:** Yichang Feng `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5066836550)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将约束QP与减小空间Gauss-Newton分离的轨迹优化算法，并在Franka Research 3机器人上实现实时执行；

**💡 创新点**

创新点在于将关节限幅通过柔性hinge‑squared惩罚融入GN步，采用两阶段非单调接受规则与自适应正则化实现全局收敛，并将QP仅用于初始化与终止修复，显著降低计算成本；

**🔧 技术方法**

使用低维基函数参数化（FACTO方式）、减小空间Gauss‑Newton、hinge‑squared约束惩罚、非单调接受规则、软硬约束分离、自适应λ调节以及Pinocchio‑FCL碰撞检测；

**📊 数据集**

在Franka Research 3机器人上进行1000+单臂/双臂任务的仿真 benchmark，包含未约束与约束场景，使用统一的Pinocchio‑FCL进行碰撞检测；

**📈 对比分析**

与CHOMP、TrajOpt、GPMP2、FACTO以及RRT‑Connect、RRT*、PRM等优化/采样算法在相同kinematics库下进行比较，指标包括成功率、计算时间和轨迹平滑度；结果显示相比FACTO计算时间快40‑75%，成功率保持88‑97%，在大多数场景中优于采样方法；

**⚠️ 局限性**

仍受限于高维约束和复杂交叉碰撞的处理、对极端动态环境的实时适应有限、依赖离线规划与执行分离，对多目标约束的扩展尚未充分验证。

---

## 318. CLASP: Defending Hybrid Large Language Models Against Hidden State Poisoning Attacks

**arXiv ID:** 2603.12206 | [PDF](https://arxiv.org/pdf/2603.12206v1)

**作者:** Alexandre Le Mercier `[一作]` (Ghent University), Chris Develder `[通讯]` (Ghent University)

**通讯引用:** 6599 | [OpenAlex ID](https://openalex.org/A5084742757)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Clasp 系统，在 SSM（如 Mamba）中通过分析块输出嵌入（BOE）实现对隐藏状态毒化攻击（HiSPA）的高效检测与拦截；

**💡 创新点**

创新点在于：1）发现 HiSPA 触发会在 Mamba BOE 中产生可观测的统计特征；2）构建 409 维 BOE 统计特征并使用 XGBoost 进行二分类；3）实现模型无关、低延迟的前置防御方案；

**🔧 技术方法**

采用 Mamba SSM、BOE 统计特征（L2 范数、均值、方差、偏度、峰度等）、XGBoost 分类器、Optuna 超参调优以及对 10^7 级 token 流量的性能测试；

**📊 数据集**

使用 2,483 篇公开简历（Hugging Face 简历集）及其 HiSPA 与无害触发版本，总计约 9.5M tokens；同时使用 RoBench-25 子集进行 BOE 特征提取；

**📈 对比分析**

通过完整集、留一法（LOO）和聚类交叉验证（CCV）三种评估，token F1 约 95.9%，文档 F1 99.3%；即使在未见触发器的情况下，文档 F1 仍保持 91% 以上；系统吞吐量 1,032 tokens/s，VRAM 消耗 <4GB；

**⚠️ 局限性**

局限性包括：1）对结构上与训练集不同的零日 HiSPA 触发器泛化能力有限；2）token 级检测受时不变约束，易出现误检/漏检；3）仅针对 HiSPA，无法完整阻止所有 PIA，需要与其他检测工具结合；

---

## 319. ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation

**arXiv ID:** 2603.11542 | [PDF](https://arxiv.org/pdf/2603.11542v1)

**作者:** Md Jahidul Islam `[一作]` (Bangladesh University of Engineering and Technology), Md Jahidul Islam `[通讯]` (Bangladesh University of Engineering and Technology)

**通讯引用:** 602 | [OpenAlex ID](https://openalex.org/A5100604188)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的一-shot 视觉‑语言适配框架 ReHARK，利用全局 RKHS 近似、混合语义‑视觉先验、支持集桥接与分布校正以及多尺度 RBF 核实现对 CLIP 的高效微调。

**💡 创新点**

创新点包括：
1) 通过融合 CLIP 文本权重、GPT‑3 语义描述和视觉类别原型构建混合先验；
2) 采用多阶段改进管线（先验构建、桥接增广、分布校正、多尺度 RBF）；
3) 将 Tip‑Adapter 的局部 NW 估计转化为全局 Kernel Ridge Regression，消除边界偏差；
4) 使用多尺度 RBF 核组合捕获不同尺度的特征几何结构。

**🔧 技术方法**

技术手段包括：
- RKHS 中的 Kernel Ridge Regression 与近端正则化；
- 多尺度 RBF 核（线性组合不同宽度的高斯核）；
- 支持集桥接（通过视觉特征与先验混合生成中间样本）；
- 统计分布校正（幂变换 + ℓ₂ 归一化）；
- Optuna 超参数搜索（β₁、β₂、p、γ、ω 等）。

**📊 数据集**

使用了 11 个公开基准数据集：ImageNet、Caltech101、DTD、EuroSAT、FGVCAircraft、Food101、OxfordFlowers、OxfordPets、StanfordCars、SUN397、UCF101。

**📈 对比分析**

在 1‑shot 场景下与 Zero‑Shot CLIP、GDA、Tip‑Adapter、ProKeR 等基线对比，平均准确率从 58.88% 提升至 65.83%，在多项任务中均超越对手；消融实验验证了各模块对性能的贡献。

**⚠️ 局限性**

局限性：
- 需要 1000 次 Optuna 试验，调参成本较高；
- 依赖 GPT‑3 生成的通用文本描述，对专业/技术域可能不够 discriminative；
- 在高类内方差的一-shot 场景下仍存在对齐困难；
- 未在更大规模的 LVLM 上验证；
- 未来需开发在线超参数预测或生成式桥接样本。

---

## 320. Ghost Framing Theory: Exploring the role of generative AI in new venture rhetorical legitimation

**arXiv ID:** 2603.11384 | [PDF](https://arxiv.org/pdf/2603.11384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 321. Mitigating the Multiplicity Burden: The Role of Calibration in Reducing Predictive Multiplicity of Classifiers

**arXiv ID:** 2603.11750 | [PDF](https://arxiv.org/pdf/2603.11750v1)

**作者:** Mustafa Cavus `[一作]` (Eskisehir Technical University), Mustafa Cavus `[通讯]` (Eskisehir Technical University)

**通讯引用:** 139 | [OpenAlex ID](https://openalex.org/A5057924084)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了分类器校准对信用风险预测中预测多重性（Rashomon效应）影响，并通过后置校准方法降低模型间的预测不一致性。

**💡 创新点**

首次将概率校准与预测多重性结合，证明校准能在不牺牲准确率的前提下显著减少不同近最优模型对少数类别样本的争议，并揭示校准对少数类的影响不均匀。

**🔧 技术方法**

使用AutoML自动生成多样化模型，构建Rashomon集；通过Platt Scaling、Isotonic Regression、Temperature Scaling三种后置校准方法；评估指标包括Obscurity、Ambiguity、Discrepancy和预测置信度，并使用Wilcoxon秩和检验与Dunn检验进行统计比较。

**📊 数据集**

九个公开信用风险基准数据集，样本量从1,000到251,503，特征数从11到65，类别不平衡率介于2.3到20.2。

**📈 对比分析**

与未校准基线相比，三种校准方法均显著降低Obscurity（尤其是Platt Scaling和Isotonic Regression对多数类降至≈0.10，少数类从≈0.14降至≈0.10），并在大多数数据集上提升预测置信度。统计检验显示校准对多数类效果更强，少数类仍保留一定多重性。

**⚠️ 局限性**

局限性：校准对少数类样本的提升有限，且实验仅针对二分类信用风险；未考虑校准对模型训练过程的直接影响；未来需在多类别场景和结合校准约束的训练方法中进一步验证。

---

## 322. ADMM-based Continuous Trajectory Optimization in Graphs of Convex Sets

**arXiv ID:** 2603.11335 | [PDF](https://arxiv.org/pdf/2603.11335v1)

**作者:** Lukas Pries `[一作]` (Technical University of Munich), Markus Ryll `[通讯]` (Technical University of Munich)

**通讯引用:** 2510 | [OpenAlex ID](https://openalex.org/A5018909750)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种基于 ADMM 的连续轨迹优化器 ACTOR，能够在由多个凸集组成的非凸环境中同时优化离散空间分配与连续控制。

**💡 创新点**

创新点在于：① 构造“时空分配图”，将混合整数约束化为有向无环图，从而把非凸约束转化为最短路径问题；② 将轨迹参数化为多段 Bernstein 多项式，使得主更新可用闭式二次规划求解；③ 通过 ADMM 将主子问题拆分为高效的 QP、投影和图搜索，获得 factorization‑free、可线性扩展的算法。

**🔧 技术方法**

主要技术包括：ADMM 分解、混合整数规划的图表述、Bezier 多项式轨迹参数化、凸集合投影、动态规划求解 DAG 最短路径、以及二次控制能量目标。

**📊 数据集**

实验使用了：1）基于点云的非凸迷宫与 U‑形障碍；2）仿真无人机的最小‑snap 轨迹规划；3）多段线性安全通道与 A* 路径搜索生成的分解；没有使用公开工业数据集，而是构造的合成测试环境。

**📈 对比分析**

与现有方法（OBCA、SafeC、A*、GCS 以及 MIQP 求解器）比较，ACTOR 在可行解空间更大、收敛更快、迭代次数少、整体运行时间呈线性增长；在复杂环境下能得到更平滑、接近全局最优的轨迹，且不需要预先构造安全通道或复杂的 warm‑start。

**⚠️ 局限性**

局限性包括：① 由于非凸约束，ADMM 只能保证局部最优，缺乏全局最优性保证；② 对惩罚参数 ρ 的选择仍需经验；③ 在极端非凸或极大规模分解时，图搜索与投影仍可能成为瓶颈；④ 对动态约束的高阶逼近需要手工设定安全边界。

---

## 323. On-Average Stability of Multipass Preconditioned SGD and Effective Dimension

**arXiv ID:** 2603.11989 | [PDF](https://arxiv.org/pdf/2603.11989v1)

**作者:** Simon Vary `[一作]` (University of Oxford), Patrick Rebeschini `[通讯]` (University of Oxford)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5091668854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了多次预处理随机梯度下降（PSGD）中的人口风险曲率、噪声几何和预处理之间的权衡，分析了这些因素对PSGD的泛化能力的影响。

**💡 创新点**

提出了一种新的平均稳定性分析方法，适用于多次PSGD，解决了数据重用引起的相关性问题，并推导出与有效维度相关的过度风险界限。

**🔧 技术方法**

使用了平均稳定性分析技术，结合了多次随机梯度下降（PSGD）算法的特性。

**📊 数据集**

使用了从训练集S中均匀抽样的样本数据集，具体数据集未明确给出。

**📈 对比分析**

与现有的单次SGD稳定性技术相比，提出的方法能够处理多次迭代中的数据相关性，性能上能够提供更好的过度风险界限，尤其是在选择不当的预处理器时，可能导致优化和泛化的有效维度依赖性不佳。

**⚠️ 局限性**

研究中提到的限制包括在多次迭代中处理参数与数据集之间的依赖性，以及在选择预处理器时可能导致的统计性能不佳。

---

## 324. SNAP-V: A RISC-V SoC with Configurable Neuromorphic Acceleration for Small-Scale Spiking Neural Networks

**arXiv ID:** 2603.11939 | [PDF](https://arxiv.org/pdf/2603.11939v1)

**作者:** Kanishka Gunawardana `[一作]` (University of Peradeniya), Isuru Nawinne `[通讯]` (University of Peradeniya)

**通讯引用:** 164 | [OpenAlex ID](https://openalex.org/A5004075430)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了基于RISC‑V的SNAP‑V SoC，集成可配置的神经形态加速器Cerebra‑H，用于小规模SNN推理；

**💡 创新点**

将RISC‑V通用核心与分层NoC、分布式权重内存相结合，实现低功耗、可扩展的SNN加速器，并支持硬件编码/解码单元；

**🔧 技术方法**

采用RISC‑V Rocket Chip + RoCC接口、分层Network‑on‑Chip、Leaky Integrate‑and‑Fire神经元、固定点运算以及Chisel/Chipyard开发流程；

**📊 数据集**

使用MNIST手写数字数据集进行SNN训练与测试；

**📈 对比分析**

通过与ODIN、TrueNorth、Loihi等代表性平台比较，Cerebra‑H在45 nm节点下实现0.5 W功耗、1.05 pJ/SOP的突触能量、25.74 mm²面积、96 MHz频率，硬件推理平均精度误差仅2.62%，大网络误差低至0.63%；

**⚠️ 局限性**

系统总功耗主要受权重内存占比（≈96%）限制，导致整体能效受限；此外代码未公开，需进一步优化内存压缩与分层存储以提升效率。

---

## 325. IDRL: An Individual-Aware Multimodal Depression-Related Representation Learning Framework for Depression Diagnosis

**arXiv ID:** 2603.11644 | [PDF](https://arxiv.org/pdf/2603.11644v1)

**作者:** Chongxiao Wang `[一作]` (Northeastern University), Osmar R. Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5027917989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了个体感知的多模态抑郁相关表示学习框架IDRL，用于抑郁诊断。

**💡 创新点**

创新点在于将多模态特征拆分为模态共通、模态特定和抑郁无关空间，并通过个体化注意力动态加权，实现对个体差异的自适应融合。

**🔧 技术方法**

采用多模态编码器-解码器结构、中央矩差（CMD）对齐、软正交正则、任务相关/无关损失、贡献损失与对齐损失以及基于自注意力的个体查询融合技术。

**📊 数据集**

在AVEC‑2014（音视频）和Twitter（文本-图像）两个公开数据集上进行实验。

**📈 对比分析**

与多种单模态与多模态基线（包括解耦、注意力和Transformer方法）相比，IDRL在AVEC‑2014上MAE降至5.83、RMSE 7.34，在Twitter上准确率93.3%、宏F1 93.2%，均取得显著优势。

**⚠️ 局限性**

局限性包括对高质量对齐多模态数据的依赖、模型结构较为复杂且计算开销较大，以及在缺失模态或不同域时鲁棒性待进一步验证。

---

## 326. The Laziness of the Crowd: Effort Aversion Among Raters Risks Undermining the Efficacy of X's Community Notes Program

**arXiv ID:** 2603.11120 | [PDF](https://arxiv.org/pdf/2603.11120v1)

**作者:** Morgan Wack `[一作]` (University of Zurich), Mustafa Alam `[通讯]` (Clemson University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了事实核查难度如何影响Twitter/X Community Notes的有效性，探讨了投票者对难度较高的错误信息的参与度下降；

**💡 创新点**

创新点在于首次量化并证明“难度惩罚”——更难核查的错误信息更难获得公开笔记，并揭示了此现象的心理机制与系统性偏差；

**🔧 技术方法**

采用了多元逻辑回归、OLS回归、Cox比例风险模型和中介分析等计量方法，并使用GPT‑4.1/5 LLM管道进行事实核查验证；

**📊 数据集**

使用了2,250条疫苗相关的Community Notes帖子数据以及382名受访者对可信度与核查难度的问卷评分，并用LLM对帖子真实性做二次判断；

**📈 对比分析**

通过对比不同难度/可信度组合下的笔记出现率、回归系数和Cox模型风险比，发现难度较高的帖子笔记出现率下降约46%，若消除难度惩罚可预估减少约58,000次转发；

**⚠️ 局限性**

局限包括：问卷评分来自非真实投票者、研究聚焦于疫苗话题，可能缺乏跨领域泛化；难以完全排除其他潜在解释；

---

## 327. Algorithmic Consequences of Particle Filters for Sentence Processing: Amplified Garden-Paths and Digging-In Effects

**arXiv ID:** 2603.11412 | [PDF](https://arxiv.org/pdf/2603.11412v1)

**作者:** Amani Maina-Kilaas `[一作]` (Massachusetts Institute of Technology), Roger Levy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22724 | [OpenAlex ID](https://openalex.org/A5090215557)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究粒子滤波器模型在句子加工中的算法后果，证明在有限并行性下重采样会产生实时“挖掘”效应（digging‑in）和放大花园路效应。

**💡 创新点**

创新点在于将粒子滤波器的重采样机制与惊奇理论结合，给出了理论证明：重采样在不增加新信息的前提下必然提升期望惊奇，并导致长的歧义区块产生更大的加工成本；并用二阶和扩散近似阐释了该效应的强度与粒子数的倒数关系。

**🔧 技术方法**

使用粒子滤波算法（有限粒子数、权重更新、重采样）、信息论工具（KL 散度、熵）以及泰勒展开和扩散近似等数学工具对期望惊奇进行分析；并在仿真中实现二结构和多结构场景。

**📊 数据集**

未使用真实语言语料；所有实验均基于人工生成的结构概率模型（如两结构情景下的 Q 取值），用于验证理论公式和近似。

**📈 对比分析**

通过与完整并行惊奇理论（无粒子限制）的基线对比，展示粒子数减小时期望惊奇增大、花园路效应放大、digging‑in 效应出现；在仿真中，二阶近似与线性扩散近似分别在小增量和大增量场景中与真实增量高度吻合（R²≈0.82/0.70）。

**⚠️ 局限性**

局限性：假设歧义区块提供无信息、粒子数固定且重采样不受其他因素影响；未对真实句子数据做验证；结果仅在理论与仿真层面成立，实际人类加工是否存在相同的实时挖掘效应仍需实证研究。

---

## 328. CogSearch: A Cognitive-Aligned Multi-Agent Framework for Proactive Decision Support in E-Commerce Search

**arXiv ID:** 2603.11927 | [PDF](https://arxiv.org/pdf/2603.11927v1)

**作者:** Zhouwei Zhai `[一作]` (JD.com), Min Yang `[通讯]` (JD.com)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了CogSearch，一套将电商搜索转化为主动决策支持的认知对齐多智能体框架。

**💡 创新点**

创新点在于将搜索过程拆解为Planner、Executor、Guider、Decider 四个协作智能体，利用多源信息融合和LLM推理构建完整的认知闭环，实现从意图拆解到决策建议的全流程支持。

**🔧 技术方法**

使用的大语言模型包括 Qwen3-4B（Planner/Guider）和 Qwen3-8B（Decider），并结合多源检索（商品检索、Web检索、工具调用）、动态过滤与策略生成、记忆系统以及多准则决策推理技术。

**📊 数据集**

使用 JD.com 的真实交易日志构建的 ECCD‑Bench（10k 询问-回答对）进行离线评估，并在 JD 平台上进行线上 A/B 实验。

**📈 对比分析**

与传统检索‑排序基线对比，离线在复杂与咨询类查询中 ACC@5 和 UDS 均提升约 0.35/0.6 分；线上 A/B 测试显示决策成本降低 5%，整体用户转化率提升 0.41%，复杂查询转化率提升 30%。

**⚠️ 局限性**

主要限制包括系统平均延迟 1.3s（高于传统 800ms）、对大模型推理的算力需求、以及多源信息融合与推理过程中的信息不一致和幻觉风险。

---

## 329. UniHetCO: A Unified Heterogeneous Representation for Multi-Problem Learning in Unsupervised Neural Combinatorial Optimization

**arXiv ID:** 2603.11456 | [PDF](https://arxiv.org/pdf/2603.11456v1)

**作者:** Kien X. Nguyen `[一作]` (University of Delaware), Ilya Safro `[通讯]` (University of Delaware)

**通讯引用:** 2791 | [OpenAlex ID](https://openalex.org/A5057164472)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种统一的异构图表示 UniHetCO，能够在无监督神经组合优化中一次性训练并解决多类节点子集选择问题（如最大团、最大独立集、最小顶点覆盖、最小支配集）。

**💡 创新点**

创新点在于：①将一般二次规划（QP）及其约束直接编码为异构图，统一目标和约束；②使用统一的 QUBO 无监督损失实现多问题共享；③引入基于梯度范数的动态权重来平衡不同问题的梯度，从而解决多任务训练中的梯度不平衡。

**🔧 技术方法**

核心技术包括：异构图神经网络（分别对问题图、目标图、约束图进行消息传播）、QUBO 损失、梯度范数动态权重（GradNorm 思路）、归一化与贪心解码。

**📊 数据集**

实验数据集：IMDB、COLLAB、Twitter、RB200（社交网络）以及 SparseSuit（多领域稀疏矩阵集合）。

**📈 对比分析**

与 EGN、Meta-EGN、贪心基线和 Gurobi（不同时间限制）进行对比；单问题训练时性能与 EGN/META‑EGN 相当或略优；多问题训练略逊但仍保持竞争力；在零样本迁移和微调下，对某些问题（MC、MDS）可显著提升；神经网络输出可作为 Gurobi 的 MIP warm‑start，在 0.2s 时间限制内显著改善目标值。

**⚠️ 局限性**

主要限制：①高阶/全局线性约束在图中需要扩展为大量节点–边，导致图尺寸膨胀和计算开销；②即使使用动态权重，目标和罚项的尺度差异仍会导致梯度不平衡，尤其在任务分布高度异质时。

---

## 330. Sema: A High-performance System for LLM-based Semantic Query Processing

**arXiv ID:** 2603.11622 | [PDF](https://arxiv.org/pdf/2603.11622v1)

**作者:** Kangkang Qi `[一作]` (Beijing Institute of Technology), Kangfei Zhao `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 12724 | [OpenAlex ID](https://openalex.org/A5015991234)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取论文的具体研究内容，缺乏正文信息。

**💡 创新点**

无法确定创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法评估比较方法和性能。

**⚠️ 局限性**

无法确定研究的局限性。

---

## 331. An Intelligent Hybrid Cross-Entropy System for Maximising Network Homophily via Soft Happy Colouring

**arXiv ID:** 2603.11050 | [PDF](https://arxiv.org/pdf/2603.11050v1)

**作者:** Mohammad Hadi Shekarriz `[一作]`, Dhananjay Thiruvady `[通讯]` (Deakin University)

**通讯引用:** 1077 | [OpenAlex ID](https://openalex.org/A5046309576)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种将交叉熵（Cross‑Entropy）方法与快速结构感知局部搜索（Local Search）相结合的混合算法CE+LS，用于求解软快乐着色（Soft Happy Colouring）问题；

**💡 创新点**

创新点在于利用交叉熵的自适应重要采样与平滑机制实现全局搜索，并在每一代生成的候选解上立即执行局部搜索，从而在保持多样性的同时显著提升收敛速度与解质量；

**🔧 技术方法**

技术包括交叉熵优化、线性时间局部搜索、SMB（Stochastic Block Model）图生成、概率分布更新与经验熵平滑、以及基准算法的实现与对比；

**📊 数据集**

数据集为 28,000 个使用 SBM 生成的随机部分着色图，节点数 200–3,000，颜色数 2–20，边生成概率 p、q 以及快乐阈值 ρ 随机取值；

**📈 对比分析**

通过与 GA、MA+RLS、MA(LMC)、MA(Rnd) 等现有元启发式算法在平均 ρ‑快乐比率、分布直方图、Welch t‑检验等指标上比较，CE+LS 在所有难度区间表现最佳，平均 ρ‑快乐比率达到 0.904，尤其在紧约束（tight）区间显著优于对手；

**⚠️ 局限性**

局限性包括 CE 单独方法收敛慢；实验仅在 SBM 合成数据上进行，未在真实网络上验证；参数（种群规模、精英比例、平滑因子）需手工调优；在紧约束区间仍无法获得完全 ρ‑快乐着色，社区检测准确率低于基于 LMC 的方法。

---

## 332. CINDI: Conditional Imputation and Noisy Data Integrity with Flows in Power Grid Data

**arXiv ID:** 2603.11745 | [PDF](https://arxiv.org/pdf/2603.11745v1)

**作者:** David Baumgartner `[一作]` (Norwegian University of Science and Technology), Heri Ramampiaro `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 1351 | [OpenAlex ID](https://openalex.org/A5026537247)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为CINDI的端到端无监督概率框架，用于检测和修正多变量时间序列中的错误和异常，保证数据完整性；

**💡 创新点**

创新点在于将异常检测与数据插补统一在单一的条件正则化流模型中，利用精确的条件似然进行低概率片段识别和可统计一致的迭代插补；

**🔧 技术方法**

主要技术为条件正则化流（Conditional Normalizing Flow），配合基于负对数似然的阈值检测、迭代自回归插补以及CMA-ES超参搜索的模型选择；

**📊 数据集**

实验使用了挪威电力分配运营商的网格损耗时间序列数据（覆盖2017‑2023年），并在多变量时间序列基准FSB上做验证；

**📈 对比分析**

与传统插补方法（如线性、立方、双三次插值）以及模型驱动插补（dynamix、knowimp）进行比较，CINDI在F1/VUS/AUC指标上均优于大多数基线，尤其在误差率≤13.69%时表现最突出；

**⚠️ 局限性**

局限在于当数据错误率过高或缺乏噪声时模型难以学习有效分布，预训练模型才能保持稳定；此外，单模型单迭代策略可能导致过度修正或不自然插补。

---

## 333. Increasing intelligence in AI agents can worsen collective outcomes

**arXiv ID:** 2603.12129 | [PDF](https://arxiv.org/pdf/2603.12129v1)

**作者:** Neil F. Johnson `[一作]` (George Washington University), Neil F. Johnson `[通讯]` (George Washington University)

**通讯引用:** 12183 | [OpenAlex ID](https://openalex.org/A5031168379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了边缘AI代理在资源匮乏环境下的协调行为，探索了四个变量（本性、多样性、学习、文化、资源稀缺）对系统拥堵和个体收益的影响。

**💡 创新点**

创新点在于首次将真实LLM代理放入资源竞争游戏，能够独立调节本性、学习和文化，揭示“容量对人数比”决定是否需要技术升级。

**🔧 技术方法**

采用基于大语言模型的代理（GPT‑2、Pythia、OPT系列）与自适应概率阈值的决策流程，以及离散的拥堵反馈奖励机制。

**📊 数据集**

实验使用7个代理，分别加载124–410 M参数的LLM，模拟C=1~6的共享资源容量；无公开标准数据集，完全自构造游戏环境。

**📈 对比分析**

对比五个技术层次（L1~L5），通过20个随机种子、500轮模拟计算系统拥堵率和个体获胜率；结果显示在C/N≈0.5处出现交叉点，稀缺时低级技术表现最好，充裕时高级技术优越。

**⚠️ 局限性**

局限性包括仅使用小型LLM、固定温度T=1、二元动作空间、固定代理数目，且未验证大规模混合族群或真实物理测试的适用性。

---

## 334. Slow-Fast Inference: Training-Free Inference Acceleration via Within-Sentence Support Stability

**arXiv ID:** 2603.12038 | [PDF](https://arxiv.org/pdf/2603.12038v1)

**作者:** Xingyu Xie `[一作]` (National University of Singapore), Shuicheng Yan `[通讯]` (National University of Singapore)

**通讯引用:** 146899 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的 Slow-Fast Inference (SFI) 解码框架，通过在快速低成本步和偶尔的密集慢速步之间交替，利用稀疏记忆实现长上下文与长推理的高效推理。

**💡 创新点**

创新点包括：发现注意力支持在短语句内高度稳定，基于事件触发的慢步刷新策略；提出无训练的 Selector，采用 reverse‑KL 混合并结合稀疏选择；设计异步层级调度与两段内存布局的高效系统实现。

**🔧 技术方法**

技术手段涵盖稀疏注意力（sink+selected+recent）、KL 混合融合、Top‑K 选取、Soft‑NMS、跨头排他、GPU 异步流、内存共聚合稀疏核（Triton/vLLM）等。

**📊 数据集**

使用 LongBench V1/V2、GPQA‑Diamond、MMLU、Long‑CoT 任务等长上下文与长推理数据集，并在 Qwen3 系列模型上进行评估。

**📈 对比分析**

与完整 KV 全注意力基线以及多种训练无关 KV‑压缩方法对比。SFI 在不同模型规模和上下文长度下实现 1.6×–14.4× 的吞吐率提升，且在大多数任务上保持与全 KV 基线相当甚至略优的质量。

**⚠️ 局限性**

局限性包括：对 token 级别的边界触发敏感，短窗口或频繁语义切换时刷新可能不足；需要额外实现 Selector 与稀疏核；在非 Qwen3 预训练模型上可能需进一步调优；对小规模 GPU 或单卡的效率提升有限。

---

## 335. An Automatic Text Classification Method Based on Hierarchical Taxonomies, Neural Networks and Document Embedding: The NETHIC Tool

**arXiv ID:** 2603.11770 | [PDF](https://arxiv.org/pdf/2603.11770v1)

**作者:** Luigi Lomasto `[一作]` (Eustema S.p.A.), Daniele Toti `[通讯]` (Innovation Engineering S.r.l.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于层级分类体系、人工神经网络和文档嵌入的自动文本分类工具 NETHIC，并通过实验验证其在 Wikipedia 语料上的有效性。

**💡 创新点**

创新点在于将 Doc2Vec 文档嵌入与传统 Bag‑of‑Words 结合，并采用层级神经网络架构来逐层细化分类路径，从而显著提升分类准确率。

**🔧 技术方法**

使用了层级分类树、全连接多层感知机（MLP）神经网络、Doc2Vec（Gensim）、Bag‑of‑Words（CountVectorizer）、交叉验证和阈值驱动的路径构建算法。

**📊 数据集**

使用从 Wikipedia 提取的 57,304 篇文章组成的语料库，划分为训练集（54,439 篇）和验证集（2,843 篇），覆盖 117 个叶子类别。

**📈 对比分析**

通过在相同验证集上对比原始 BOW 版 NETHIC 与改进版 BOW+Doc2Vec NETHIC‑2 的准确率、混淆矩阵和 F1 分数；改进版提升约 2%（约 60 篇多正确分类），根级网络的混淆度明显降低。

**⚠️ 局限性**

局限性包括：单独使用 Doc2Vec 效果不佳；仍需依赖高维 BOW 词表；在更大规模或多领域数据上的可扩展性未完全验证；层级树深度固定，误分类传播问题仍存在。

---

## 336. Jailbreak Scaling Laws for Large Language Models: Polynomial-Exponential Crossover

**arXiv ID:** 2603.11331 | [PDF](https://arxiv.org/pdf/2603.11331v1)

**作者:** Indranil Halder `[一作]` (Harvard University), Cengiz Pehlevan `[通讯]` (Harvard University)

**通讯引用:** 1839 | [OpenAlex ID](https://openalex.org/A5023195984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于自旋玻璃的生成模型来解释大型语言模型在注入式攻击下的安全性，并推导了攻击成功率随采样次数和注入长度的增长规律

**💡 创新点**

创新点在于将安全性定义为低能量簇，利用自旋玻璃的多重重叠结构引入误差场，解释攻击成功率在弱场下呈多项式增长、强场下呈指数增长，并给出相应的解析公式

**🔧 技术方法**

使用自旋玻璃理论（RSB、Poisson-Dirichlet分布）、能量基模型、Langevin采样、解析推导与实验验证

**📊 数据集**

使用 walledai/AdvBench 数据集，并在 GPT‑4.5 Turbo、Vicuna‑7B、Meta Llama‑3.2‑3B‑Instruct、Llama‑3‑8B‑Instruct 等模型上进行实验

**📈 对比分析**

与传统拒绝字符串指标对比，采用 LLM‑Judge 评价得到更可靠的 ASR；实验曲线与理论公式 log(-logΠ_k)= -ν̂ log k - μ̂ k + log Ĉ 贴合，证明模型能够捕捉从多项式到指数的转变

**⚠️ 局限性**

局限性：假设教师与学生参数完全一致，未覆盖更复杂的安全机制；高场强下的理论仍为近似，需进一步严格证明；实验受代码数值限制，极值点可能失真

---

## 337. Fingerprinting Concepts in Data Streams with Supervised and Unsupervised Meta-Information

**arXiv ID:** 2603.11094 | [PDF](https://arxiv.org/pdf/2603.11094v1)

**作者:** Ben Halstead `[一作]`, Russel Pears `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出FiCSUM框架，通过提取并动态加权大量元信息特征，生成可唯一标识概念的指纹，用于检测数据流中的概念漂移与重复概念。

**💡 创新点**

创新点在于：①将监督与无监督的行为都纳入同一指纹表示；②引入多维元信息特征，显著提升概念区分能力；③设计动态加权策略，使同一指纹能够适配不同数据流的特征重要性。

**🔧 技术方法**

技术方法包括：窗口化数据流处理、元信息特征提取、向量化指纹生成、动态加权学习（可能基于梯度或注意力机制）、相似度测度（如余弦相似度）进行漂移检测。

**📊 数据集**

实验使用多种公开数据流数据集，涵盖监督与无监督任务，典型包括KDD Cup流式数据、Electricity, ForestCoverType等。

**📈 对比分析**

与现有概念表示方法（如使用少量元信息的传统指纹）进行对比实验，FiCSUM在概念识别准确率、漂移检测召回率和对已学习概念的重识别上均优于对手，显著提升自适应学习效果。

**⚠️ 局限性**

局限性包括：①需计算大量元信息，计算成本较高；②动态加权过程可能需要额外的调参与验证；③在极低信噪比或极端概念变化场景下，指纹区分度仍可能受到挑战。

---

## 338. Leveraging Large Language Models and Survival Analysis for Early Prediction of Chemotherapy Outcomes

**arXiv ID:** 2603.11594 | [PDF](https://arxiv.org/pdf/2603.11594v1)

**作者:** Muhammad Faisal Shahid `[一作]` (CureMD Research), Muddassar Farooq `[通讯]` (CureMD Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用大型语言模型（LLM）与生存分析，对化疗治疗方案和结果进行早期预测，涵盖乳腺癌及其他四种常见癌症。

**💡 创新点**

创新点在于将RAG+Critic-Agent循环用于临床笔记中表型和治疗标签的高效、低误报抽取，并将随机生存森林（RSF）同时用于生存概率预测与二分类，显著提升预测精度。

**🔧 技术方法**

使用的技术包括：LLaMA‑3、Mistral、Qwen 等 LLM；RAG 检索；Critic-Agent 反馈循环；随机生存森林（RSF）与 C‑index、准确率、F1 评估；校准曲线验证概率可靠性。

**📊 数据集**

数据集为 3,409 例乳腺癌患者的 EMR 笔记与相关临床特征（体重、实验室、表型、治疗方案等），并扩展到 1,685 例结直肠癌、2,079 例前列腺癌、1,072 例多发性骨髓瘤和 2,366 例肺癌。

**📈 对比分析**

方法通过与传统规则/本体抽取（NCIt、CCTOO）对比，LLM 系统在准确率、召回率、F1 上均优于本体系统；在 RSF 模型中，乳腺癌的 C‑index 为 0.731，准确率 0.723，F1 0.724；在其他癌症中，C‑index 均保持在 0.66–0.76 范围，准确率和 F1 亦在 0.68–0.77 之间，表现稳定。

**⚠️ 局限性**

局限性包括：对 EHR 质量和结构的高度依赖；LLM 仍可能产生少量幻觉，需要 Critic-Agent 辅助；药物组合支持阈值导致部分个性化方案被排除；尚未完成临床验证，模型泛化能力待进一步评估。

---

## 339. Causal Prosody Mediation for Text-to-Speech:Counterfactual Training of Duration, Pitch, and Energy in FastSpeech2

**arXiv ID:** 2603.11683 | [PDF](https://arxiv.org/pdf/2603.11683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 340. LLM-Augmented Digital Twin for Policy Evaluation in Short-Video Platforms

**arXiv ID:** 2603.11333 | [PDF](https://arxiv.org/pdf/2603.11333v1)

**作者:** Haoting Zhang `[一作]` (University of California), Zeyu Zheng `[通讯]` (University of California)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5064722764)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套可控可复现的数字孪生框架，用以评估短视频平台中闭环人机互动的政策与 AI 功能，重点实现了四个子孪生（用户、内容、交互、平台）和事件驱动的执行管道，并引入了可选的 LLM 语义决策服务，配合成本治理层实现高吞吐量的模拟。

**💡 创新点**

创新点包括：①四子孪生模块化架构和事件总线实现闭环反馈可追踪；②将平台政策与 LLM 决策拆分为可插拔组件，支持单独对策略进行因果评估；③Live/Cache/Surrogate 三层 LLM 调度与预算监控，既保证语义逼真又不牺牲规模；④在实验中引入控制实验设计（采用创作者营销计划、趋势预测）验证数字孪生在 AI‑驱动政策评估中的有效性。

**🔧 技术方法**

技术栈涵盖：agent‑based 规模模拟、事件总线与环境调度、四子孪生状态管理、LLM 统一优化层（包含批量、缓存、后备模型）、预算追踪与动态降级、基于 50 维向量的兴趣匹配与内容特征、基于规则的推荐与推广算法。

**📊 数据集**

数据集：无公开真实数据，全部使用合成生成的用户特征（根据人口统计、阶层分布）、内容原型（舞蹈、喜剧等）、交互日志及平台反馈。LLM 的输入/输出采用 JSON 架构，结果被缓存后重放。

**📈 对比分析**

对比方法：在数字孪生中对两组 AI 功能进行对照实验——创作者营销计划（LLM vs 规则）和趋势预测（LLM vs 规则）——并跨 16 条创作者采纳/收益组合、9 条治理策略+预算组合进行实验。实验结果表明：LLM 计划在保持观看时间基本不变的同时提升 3–4% 的礼物收入并降低收入 Gini；LLM 趋势预测在提升平均观看时长约 0.13 秒、略降跳过率的同时保持话题多样性不变，且 LLM 花费仅占总成本 2–3%。

**⚠️ 局限性**

局限性：①合成数据缺乏真实视频、社交网络与经济系统细节，可能导致结果与上线环境偏差；②LLM 仅在有限语义任务（如人物生成、标题、计划）中使用，未覆盖大规模生成内容或多模态视频；③对大规模用户（百万级）并行化与实时预算调度仍未充分验证；④实验中未引入真正的人类操作者或多轮交互反馈；⑤经济模型（广告、直播、商店）简化，未体现库存与价格波动；⑥成本治理层基于规则，未深度探讨 LLM 失效时的决策安全性与公平性。

---

## 341. A systematic review of secure coded caching

**arXiv ID:** 2603.11145 | [PDF](https://arxiv.org/pdf/2603.11145v1)

**作者:** S. -L. Ng `[一作]`, E. A. Quaglia `[通讯]` (Royal Holloway University of London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对安全有源缓存（Secure Coded Caching）领域的研究进行了系统综述，梳理了已有的安全目标、模型假设、实现技术，并对不同方案的性能（速率/内存曲线）和安全成本进行对比。

**💡 创新点**

创新点在于：①提出了统一的安全需求框架，识别了文献中缺失的全局安全视角；②归纳并分类了多种安全目标（内容机密性、文件隐私、需求隐私、缓存隐私、抵御协作攻击等）及其对应技术；③将安全缓存与网络编码、索引编码、私有信息检索、广播加密等已成熟的安全内容交付原语进行对照，指出潜在的互补与转化路径。

**🔧 技术方法**

主要使用的技术包括信息论方法（一次性密码、秘密共享、MDS 码、Placement‑Delivery 数组）、组合设计（虚拟用户、私有缓存）、以及对比分析所用的速率/子分割数等指标；同时还讨论了现代密码学技术（如Tee、ORAM、PIR）在未来可行性。

**📊 数据集**

作为综述论文，未采用实验数据集；所有评估均基于已有理论模型与已公布的速率/子分割曲线。

**📈 对比分析**

比较方法：作者通过绘制速率/内存曲线、列举已知下界与构造上界，对比不同安全目标下的速率损失；对比子分割复杂度，并在表格中对照方案的安全目标与技术手段。性能方面发现，尽管加入安全性会略微提升速率，但在实际参数规模下的损失相对可接受。

**⚠️ 局限性**

局限性包括：①缺乏统一的系统级安全模型，导致不同方案难以直接比较；②大多数方案仅考虑被动窃听者，忽略了主动攻击、消息篡改、认证等安全威胁；③过度依赖一次性密码，导致密钥管理与更新复杂、对密钥长度有严格要求；④未给出可实现的密钥分发或计算安全方案；⑤在实际大规模 CDN 环境中的可扩展性与实现细节缺乏讨论。

---

## 342. A Survey of Reasoning in Autonomous Driving Systems: Open Challenges and Emerging Paradigms

**arXiv ID:** 2603.11093 | [PDF](https://arxiv.org/pdf/2603.11093v1)

**作者:** Kejin Yu `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4073 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了将大型语言模型（LLM/MLLM）与自动驾驶（AD）系统集成的最新研究，提出认知层级和七大推理挑战，并从系统与评估两方面梳理了现有方法与基准，指出实现高阶自动化的关键瓶颈。

**💡 创新点**

创新点：①提出将推理提升为 AD 核心的“认知层级”框架，细化感知、 egocentric 与 social‑cognitive 三层；②归纳七大核心推理挑战（异构信号推理、感知‑认知偏差、响应‑推理权衡、决策‑现实对齐、长尾场景、监管合规与社会博弈）；③将系统与评估视角统一为双重视角，系统侧关注“玻璃盒”架构与推理驱动的感知/预测/规划；评估侧提出面向认知过程的基准与安全可信度检验。

**🔧 技术方法**

主要技术：大型语言模型与多模态 LLM、链式思考（CoT）、树式思考（ToT）、结构化表示、神经‑符号融合、双进程（fast/slow）架构、工具调用、记忆/外部知识检索、强化学习微调。

**📊 数据集**

使用的数据集/基准：NuScenes‑QA、NuScenes‑MQA、NuScenes‑SpatialQA、Reason2Drive、DriveLM‑Bench、OmniDrive、DrivingDojo、AutoTrust‑Bench、DriveAction‑Bench、DrivingVQA、DriveLMM‑o1、WOMD‑Reasoning、RoadTextVQA、STSB‑Bench 等。

**📈 对比分析**

方法比较：多篇系统化评测显示基于 CoT 的“玻璃盒”架构在解释性与内部一致性上优于传统黑盒方法；基于推理的感知/预测/规划模块在长尾安全场景（如临时施工、恶劣天气）下表现显著提升；然而，大型模型推理仍受限于推理时延与计算资源，尚缺乏统一的实时评估指标。

**⚠️ 局限性**

局限性：①推理延迟与安全实时需求冲突；②符号-物理对齐难度大，缺乏可验证的神经‑符号架构；③对多模态不确定性、感知‑认知偏差的鲁棒推理机制不足；④监管合规与社会博弈的实时动态检索与执行仍不成熟；⑤评估基准多停留在模拟或离线问答，真实世界鲁棒性与“未知未知”场景验证不足。

---

## 343. CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing

**arXiv ID:** 2603.11810 | [PDF](https://arxiv.org/pdf/2603.11810v1)

**作者:** Yue Shi `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 23012 | [OpenAlex ID](https://openalex.org/A5100447801)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过构建协同显式-隐式 3D 表示（隐式 SDF + 可编辑的处理点）实现了高质量的 3D 重建与细粒度、逼真的对象编辑。

**💡 创新点**

创新点包括将隐式 SDF 与显式可编辑处理点相结合、对光照与材质进行物理属性解耦，以及利用 SAM 与跨视角传播实现空间感知的局部编辑。

**🔧 技术方法**

采用 SDF 神经网络、可微采样、双分支 DDA 网络、Disney BRDF 与球面高斯光照模型、SAM+跨视角分割、ARAP 几何变形等技术，并在 PyTorch 中实现。

**📊 数据集**

在 PhySG 合成、NeRF 合成、DTU 实景、PhotoShape 组装的四个数据集上进行实验。

**📈 对比分析**

与 EditNeRF、NeuMesh、Seal3D、GaussianEditor 在 FID、PSNR、SSIM、LPIPS 及编辑时长等指标对比，CEI‑3D 在 FID（最低）、PSNR/SSIM（最高）、LPIPS（最低）和编辑时间（显著缩短）等方面均取得了最佳性能。

**⚠️ 局限性**

目前仅适用于静态多视图对象，需多张视图训练，对动态场景或单视图重建支持不足，训练成本仍相对较高，且对极其复杂材质的表现仍有提升空间。

---

## 344. Online Learning of Strategic Defense against Ecological Adversaries under Partial Observability with Semi-Bandit Feedback

**arXiv ID:** 2603.11726 | [PDF](https://arxiv.org/pdf/2603.11726v1)

**作者:** Anjali Purathekandy `[一作]` (Indian Institute of Science), Deepak N. Subramani `[通讯]` (Indian Institute of Science)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5076749495)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了HERDS算法，解决在人类-大象冲突中对未知生态对手进行自适应资源部署的在线学习问题。

**💡 创新点**

创新点在于：①基于损失驱动的动态探索-利用预算分配；②在混合半bandit反馈下对未观测攻击损失的分配式奖励估计；③在不依赖对手行为模型的前提下实现子线性遗憾保证。

**🔧 技术方法**

使用技术包括：在线组合优化中的Follow-the-Perturbed-Leader与Geometric Resampling、半bandit反馈估计、动态奖励更新以及Agent-Based Modeling仿真验证。

**📊 数据集**

使用的数据集为印度Periyar-Agasthyamalai地区的大象运动轨迹数据和相应的农业区边界划分。

**📈 对比分析**

与FPL-UE及静态策略对比，HERDS在100轮实验中将累计遗憾降低15-45%，在适应性攻击模型下将作物损失降低40-50%，收敛速度提升至40-50轮。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏实地实验；对守卫效果的噪声假设较简化；对大规模边界或多对手情景的扩展仍待研究。

---

## 345. RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

**arXiv ID:** 2603.12215 | [PDF](https://arxiv.org/pdf/2603.12215v1)

**作者:** Bin Wan `[一作]` (Shandong University), Sam Kwong `[通讯]` (Lingnan University)

**通讯引用:** 34648 | [OpenAlex ID](https://openalex.org/A5008386708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种Region Proportion-aware Dynamic Adaptive Salient Object Detection Network（RDNet），通过SwinTransformer骨干和三大自适应模块实现遥感图像显著目标检测。

**💡 创新点**

创新点包括：①依据目标区域比例动态选择不同尺寸卷积核的DAD模块；②利用小波交互与注意力的FCE模块提取多尺度上下文；③通过连续交叉注意力的RPL模块以及比例引导PG块实现区域定位与尺度感知。

**🔧 技术方法**

采用的技术有：SwinTransformer、动态卷积核选择、频率匹配小波交互、交叉注意力、全局平均池化+全连接比例引导、通道与空间注意力机制、深度监督等。

**📊 数据集**

使用公开光学遥感显著目标数据集：ORSSD、EORSSD和ORSI-4199，共计约6,000张图像。

**📈 对比分析**

与21种现有SOD方法（包括CNN、ViT、PVT等）在三数据集上进行定量和定性对比，RDNet在MAE、Fβ、Eξ等指标上均优于最优对手，MAE下降约3.9%，Fβ提升约9.1%。

**⚠️ 局限性**

局限性在于对极小或纹理与背景相似的细目标仍可能误检，且对极细目标检测效果仍不理想。

---

## 346. A Hybrid Neural-Assisted Unscented Kalman Filter for Unmanned Ground Vehicle Navigation

**arXiv ID:** 2603.11649 | [PDF](https://arxiv.org/pdf/2603.11649v1)

**作者:** Gal Versano `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**通讯引用:** 2562 | [OpenAlex ID](https://openalex.org/A5012718881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种混合深度学习与Unscented Kalman Filter（UKF）的导航框架，利用深度网络直接预测过程和测量噪声协方差，从而实现无人地面车辆（UGV）导航中的噪声自适应估计。

**💡 创新点**

创新点在于：①使用sim2real方法在仿真数据上训练双用途网络σ_Q-Net和σ_R-Net，直接从原始IMU和GNSS测量回归噪声标准差；②将网络输出的噪声协方差与传统UKF结合，采用α、β权重平衡自适应与固定协方差；③保持UKF核心方程不变，兼顾经典滤波的稳定性与深度学习的自适应性。

**🔧 技术方法**

技术手段包括：Unscented Kalman Filter、深度一维卷积网络、Adam优化器、滑动窗口回归、仿真生成多轨迹噪声数据、混合α/β权重调节。

**📊 数据集**

使用了三套真实UGV数据集（ROOAD平台、香港车辆、Rosbot‑XL机器人）共计160分钟，并在这些数据上进行评估；训练数据来源于五种基础轨迹的仿真，覆盖多种运动模式和噪声水平。

**📈 对比分析**

与四种基线方法（UKF、MB‑AUKF、ANPN‑UKF、ANPMN‑UKF）在PRMSE指标上比较。实验显示：ANPMN‑UKF平均PRMSE显著降低，分别比UKF提升22.7%、比MB‑AUKF提升12.7%、比ANPN‑UKF提升8.0%；运行时间略高于传统UKF。

**⚠️ 局限性**

局限性：①训练依赖仿真，若真实环境与仿真偏离显著，噪声估计可能退化；②相较于纯UKF，加入网络推理导致每步计算成本略高。

---

## 347. Representation Finetuning for Continual Learning

**arXiv ID:** 2603.11201 | [PDF](https://arxiv.org/pdf/2603.11201v1)

**作者:** Haihua Luo `[一作]`, Fengyu Cong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将表示层微调（ReFT）引入持续学习的框架 CoRe，利用低秩子空间对隐藏表示进行任务特定干预，实现模型在新任务上的快速适应。

**💡 创新点**

创新点在于：①将微调焦点从权重空间转移到表示空间，②在低秩子空间内采用显式优化目标以控制表示漂移，③兼顾参数高效性与对灾难性遗忘的抑制。

**🔧 技术方法**

使用的技术包括低秩投影矩阵 R、可学习线性变换 W 与偏置 b、正交约束、ViT 预训练模型以及 SGD、余弦学习率衰减等常规训练策略。

**📊 数据集**

实验数据集覆盖多种持续学习场景：任务增量（Aircraft、Caltech101、CIFAR100、DTD、EuroSAT 等）、域增量（CDDB、CORe50、DomainNet、OfficeHome）和类增量（CIFAR100、CUB200、ImageNet‑A/R、ObjectNet、OmniBenchmark、VTAB），使用 ViT‑B/16‑IN21K/IN1K 作为骨干。

**📈 对比分析**

与 Adapter、Prompt、SSF、全微调等参数高效微调方法以及冻结模型进行对比，CoRe 在 TIL、DIL、CIL 任务中均取得平均精度最高、参数量最低的结果，且在不平衡数据、不同随机种子和不同预训练路径下表现稳健。

**⚠️ 局限性**

局限性包括：①对低秩子空间维度的选择需要经验调参；②在非常深层或不同结构的模型上（如大型语言模型）需进一步验证；③对计算成本的额外开销（投影矩阵）与极端任务规模下的扩展性尚未充分评估。

---

## 348. Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**arXiv ID:** 2603.11556 | [PDF](https://arxiv.org/pdf/2603.11556v1)

**作者:** Xinyu Nan `[一作]` (Peking University), Mei Yang `[通讯]` (Peking University)

**通讯引用:** 71858 | [OpenAlex ID](https://openalex.org/A5100460802)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于扩散模型的双监督图像审美提升框架DIAE

**💡 创新点**

引入多模态审美感知（MAP）与弱监督“非完全配对”数据集，并采用双分支监督策略以提升审美与保持内容一致性

**🔧 技术方法**

使用Stable Diffusion v1.5扩散模型、ControlNet控制结构、HSV/轮廓图视觉编码器、LLaVA/UNIAA‑LLaVA进行审美评估与文本编码

**📊 数据集**

构建了 Imperfectly‑paired Image Aesthetic Enhancement Data（IIAEData），来源于AVA、TADNet、KONIQ、Flickr 等公开数据集，包含弱配对图像、标题与审美评估文本

**📈 对比分析**

在 LAION‑Aesthetic 评分、MLLM 评分（UNIAA‑LLaVA、DepictQA‑v2）以及 CLIP‑I 内容一致性指标上与 ControlNet、InstructPix2Pix、MGIE、DOODL 等 SOTA 方法对比，DIAE 在审美分数上提升 7–17% 并保持最高的内容一致性

**⚠️ 局限性**

目前对人像、群体等复杂场景缺乏适配，模型在此类图像的审美提升效果受限

---

## 349. Cross-Platform Digital Discourse Analysis of Iran: Topics, Sentiment, Polarization, and Event Validation on Telegram and Reddit

**arXiv ID:** 2603.11057 | [PDF](https://arxiv.org/pdf/2603.11057v1)

**作者:** Despoina Antonakaki `[一作]` (Foundation for Research and Technology), Sotiris Ioannidis `[通讯]` (Technical University of Crete)

**通讯引用:** 4603 | [OpenAlex ID](https://openalex.org/A5022073151)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Telegram和Reddit两平台的伊朗相关讨论进行跨平台分析，构建统一数据集并提炼话题、情感、对立度和升级指数。

**💡 创新点**

提出了整合TF–IDF+NMF、VADER情感、关键词束升级指数与事件时间序列的可复现跨平台框架，并展示实时升级检测与实际事件的时滞关系。

**🔧 技术方法**

采用Python、TF–IDF、非负矩阵分解、VADER情感分析、关键词束升级指数、时间序列相关与延迟分析、实体共现网络等技术。

**📊 数据集**

收集了2017-2026年Telegram 7,567条消息与2025-2026年Reddit 23,909条帖子/评论，并在2026年2月补充4,955条Telegram与49,014条Reddit实时数据。

**📈 对比分析**

通过同日及滞后相关性将升级指数与公开抗议及地缘政治事件时间线对比，发现最佳相关滞后约1-3周，说明平台语义升级能提前或跟随现实事件，且跨平台一致性较高。

**⚠️ 局限性**

受限于频道/子版块选择与关键词束方法，无法捕捉隐式升级表达，且缺乏多平台覆盖和嵌入式升级检测，未来需扩展到更多平台并加入机器学习分类器。

---

## 350. Mobile-GS: Real-time Gaussian Splatting for Mobile Devices

**arXiv ID:** 2603.11531 | [PDF](https://arxiv.org/pdf/2603.11531v1)

**作者:** Xiaobiao Du `[一作]` (University of Technology Sydney), Xin Yu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Mobile‑GS，针对移动设备实现实时高质量Gaussian Splatting。

**💡 创新点**

深度感知无排序渲染、视角依赖增强、一次spherical harmonics蒸馏、神经向量量化与贡献剪枝。

**🔧 技术方法**

深度感知无排序渲染、MLP视角增强、SH蒸馏、神经向量量化、贡献剪枝以及Vulkan实现。

**📊 数据集**

Mip‑NeRF 360等室内外场景数据集。

**📈 对比分析**

与3DGS、LightGaussian、SortFreeGS等方法对比，达到116–127 FPS、4.8 MB存储，质量与原3DGS相当。

**⚠️ 局限性**

训练耗时较长，对极端光照或复杂透明物体仍可能出现残留伪影。

---

## 351. Enhancing Lightweight Vision Language Models through Group Competitive Learning for Socially Compliant Navigation

**arXiv ID:** 2603.11447 | [PDF](https://arxiv.org/pdf/2603.11447v1)

**作者:** Xinyu Zhang `[一作]` (Hokkaido University), Ling Xiao `[通讯]` (Hokkaido University)

**通讯引用:** 46885 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发一种名为Group Competitive Learning（GCL）的训练策略，提升轻量级视觉语言模型在社会机器人导航中的推理与决策能力。

**💡 创新点**

通过组竞争目标（GCO）实现全球语义一致与分布正则化，并利用非对称组优化（AGO）在不同容量模型间竞争性学习，从而让小模型超越大模型的性能。

**🔧 技术方法**

采用GCO（含监督损失、全局语义损失GSL、分布正则化损失DRL）与AGO（学习率与温度异构调节），结合信息论对比损失与Jensen‑Shannon散度，训练时使用DeepSpeed ZeRO‑3加速。

**📊 数据集**

在SNEI（265/60训练/测试场景）和MUSON（640/160场景）两大社会导航基准数据集上进行评估。

**📈 对比分析**

与传统SFT做对比，Qwen2.5‑VL‑3B在GCL下F1从0.692提升至0.968（+40%），Qwen3‑VL‑4B从0.816提升至0.914（+12%），甚至3B模型在GCL下超过同类8B模型。

**⚠️ 局限性**

对细粒度时空推理仍有限，易在反射表面和细微人机交互上误判；目前仅验证于两类Qwen模型，尚未推广到更小模型或其他VLM架构。

---

## 352. Delayed Backdoor Attacks: Exploring the Temporal Dimension as a New Attack Surface in Pre-Trained Models

**arXiv ID:** 2603.11949 | [PDF](https://arxiv.org/pdf/2603.11949v1)

**作者:** Zikang Ding `[一作]` (University of Electronic Science and Technology of China), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 86138 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练语言模型中提出并实现了延迟后门攻击（DBA）和其原型DND，展示了通过状态跟踪与非线性衰减实现的可控延迟激活机制，并验证其在保持高正类准确率的同时，能够在触发阈值后几乎完美地执行攻击。

**💡 创新点**

①挑战传统的即时假设，引入时间维度的新攻击范式；②设计了可解释的非线性衰减激活控制器；③证明普通高频词可作为隐蔽触发器，并通过延迟机制绕过现有防御。

**🔧 技术方法**

结构级别的后门插入（状态跟踪模块 + 非线性衰减控制器）；基于BERT的文本分类模型；联合惰性与激活损失；双指标评估（ASR 与 ASR_delay）；对抗 ONION、STRIP、RAP、CUBE 等主流防御。

**📊 数据集**

SST‑2、HSOL、Offenseval、Twitter 四大文本分类基准。

**📈 对比分析**

与 BadNet、Syntactic、BITE 等传统即时后门相比，DND 在清洁准确率 ≥94% 的同时，ASR_delay 接近 100%，在多种防御下仍保持 90% 以上的成功率，明显优于对照组，表明延迟后门在隐蔽性与效果上更具优势。

**⚠️ 局限性**

仅依赖结构级修改，仍可能被人工白盒审计发现；目前实现仅为原型，缺乏对不同模型/任务的广泛验证；高频词触发器的评估标准尚不成熟，需要进一步研究状态化防御。

---

## 353. SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**arXiv ID:** 2603.12193 | [PDF](https://arxiv.org/pdf/2603.12193v1)

**作者:** Mengzhen Liu `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 10809 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种端到端的主动操控框架 SaPaVe，融合语义主动感知与主动视角执行；

**💡 创新点**

创新点在于：①将摄像头控制与操作动作解耦，采用两阶段训练；②构建200K条图像-语言-摄像头动作对数据集 ActiveViewPose-200K；③提出ActiveManip-Bench 评测基准；④引入通用空间知识注入提升 3D 视角鲁棒性；

**🔧 技术方法**

使用大型语言视觉模型（如 Qwen2.5-VL、Gemini），LoRA 方式的摄像头适配器，解耦动作头，3D 语义注入模块；

**📊 数据集**

数据集包括 ActiveViewPose-200K（200k 语义摄像头动作对）和 ActiveManip-Bench（12 任务、100 对象、20 场景的模拟基准）；

**📈 对比分析**

与现有 VLM、VLA 模型（π_0、GR00T‑N1）以及固定视角/手腕摄像头配置对比，SaPaVe 在模拟与真实场景下平均成功率提升至约 75% 以上，超越基线 20–30% 左右；

**⚠️ 局限性**

局限性包括：仍需大量标注数据、对极端动态场景的鲁棒性尚未完全验证、系统在真实环境中的部署成本较高。

---

## 354. COTONET: A custom cotton detection algorithm based on YOLO11 for stage of growth cotton boll detection

**arXiv ID:** 2603.11717 | [PDF](https://arxiv.org/pdf/2603.11717v1)

**作者:** Guillem González `[一作]` (Institut de Robòtica i Informàtica Industrial), Sergi Foix `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5085721116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在棉花采摘自动化中提出了一种针对棉花果实不同生长阶段的轻量化YOLO11改进模型，用于高精度目标检测。

**💡 创新点**

创新点在于将SE、SimAM、PHAM注意力模块与CARAFE上采样、SCDown下采样及SIoU损失结合，显著提升复杂场景下检测性能。

**🔧 技术方法**

采用改进的YOLO11架构，融合SEConvBlock、SimAM、PHAM、CARAFE、SCDown及SIoU损失函数。

**📊 数据集**

使用自建的六类棉花生长阶段公开数据集，共计数千张图像，并通过Albumentations进行增强。

**📈 对比分析**

与YOLOv8/9/10/12等小中型模型对比，改进模型在mAP50达0.811、mAP50-95达0.606，参数仅7.6M，显著优于基线。

**⚠️ 局限性**

主要局限在数据集规模有限、对极端遮挡与光照变化仍有误检，且未在真实机器人平台上验证。

---

## 355. Markovian Generation Chains in Large Language Models

**arXiv ID:** 2603.11228 | [PDF](https://arxiv.org/pdf/2603.11228v1)

**作者:** Mingmeng Geng `[一作]` (École Normale Supérieure Paris Sciences et Lettres and Centre National de la Recherche Scientifique), Thierry Poibeau `[通讯]` (École Normale Supérieure Paris Sciences et Lettres and Centre National de la Recherche Scientifique)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLM）在重复推理（如反复改写或往返翻译）中的文本演化过程，并提出了马尔可夫生成链（Markovian Generation Chains）框架进行定量分析。

**💡 创新点**

创新点在于：①将迭代LLM推理视为句子级的马尔可夫链，提供了新的理论视角；②引入“先前输出不携带记忆”的递归推理模型；③结合信息论工具（熵、KL收缩、混合界限）评估多步推理的多样性与稳定性；④系统性比较不同解码策略、提示多样性和模型的迭代行为。

**🔧 技术方法**

核心技术包括：LLM推理（Mistral‑7B、Llama‑3.1‑8B、Qwen2.5‑7B、GPT‑4o‑mini）、马尔可夫链构造与分析、句子级熵/相似度度量（METEOR、ROUGE‑1、BLEU、TF‑IDF 余弦）、实验中对解码方式（贪婪 vs 采样）和提示方式（单一 vs 交替）进行系统控制。

**📊 数据集**

使用的公开数据集有：BookSum、ScriptBase‑alpha、BBC News2024；从每个数据集中随机抽取 150 篇文档，取首句作为初始句子进行 50 步迭代实验。

**📈 对比分析**

比较方法：对不同模型、不同解码策略、不同提示设置下的迭代链，统计独特句子数、首次循环时间、相邻句子相似度以及与原始句子的累计相似度；与 Google Translate（v3）做往返翻译对照；实验结果显示：贪婪解码快速进入短周期或固定点；采样解码显著延长非循环阶段，产生更多独特句子；提示交替可进一步提高多样性；在大多数情况下，LLM 的迭代多样性超过固定 MT 服务。

**⚠️ 局限性**

限制包括：①实验仅在句子级别进行，忽略段落/篇章级结构；②马尔可夫链理论假设状态空间有限，实际文本空间巨大，仅能在有限迭代步数内观察；③未对语义保持度进行深入评估，可能存在累计语义漂移；④实验数据集有限，未覆盖更广泛的文本领域；⑤仅考察单一模型参数配置，未涉及模型更新或自适应机制。

---

## 356. Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple

**arXiv ID:** 2603.11053 | [PDF](https://arxiv.org/pdf/2603.11053v1)

**作者:** Amirhossein Bozorgkhoo `[一作]` (Independent Researcher), Igor Molybog `[通讯]` (University of Hawaii at Manoa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Speculative Decoding Scaling Law (SDSL)，通过分析草稿模型和目标模型的困惑度与接受率的关系，推导出在预训练缩放法则下的吞吐量表达式，并给出最优草稿模型规模与目标模型规模、训练数据量的对应关系。

**💡 创新点**

创新点在于：①用解析公式将草稿与目标模型的 perplexity 映射到 token 接受率 α；②结合预训练缩放法则得到闭式吞吐量表达式；③得到最优草稿模型尺寸约为目标模型的 1/200，提供理论指导取代经验搜索；④通过数值和实测验证该比例的普适性。

**🔧 技术方法**

主要技术手段包括：分析推导（利用拉姆伯特 W 函数等数学工具）、线性回归和数值网格搜索、Microsoft Deepspeed 实现 Speculative Decoding、计算 perplexity 与 α 的估算。

**📊 数据集**

使用 HellaSwag 数据集评估 perplexity，实验涵盖 LLaMA 3/3.1、OPT、Qwen 1.5/2.5、Seed-OSS 等多族模型。

**📈 对比分析**

通过对比不同草稿/目标模型组合的 α、token/FLOP 与 token/s 吞吐率，验证理论预测。实验表明最优草稿模型尺寸约为目标模型的 200 倍，理论与实测结果一致，显著提升吞吐率并降低延迟。

**⚠️ 局限性**

局限性包括：①假设草稿和目标模型在相同数据分布、训练配置下训练；②仅针对自回归文本模型，未验证编码-解码、混合专家或多模态模型；③预训练缩放系数对不同任务的适用性可能有限。

---

## 357. PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On

**arXiv ID:** 2603.11675 | [PDF](https://arxiv.org/pdf/2603.11675v1)

**作者:** Haohua Chen `[一作]` (Xiaohongshu Inc.), Zhiyong Wu `[通讯]` (Tsinghua University)

**通讯引用:** 3999 | [OpenAlex ID](https://openalex.org/A5102869280)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于流匹配 DiT 的虚拟试衣框架，支持多件衣物的实时试穿并可通过文本提示控制穿搭风格。

**💡 创新点**

创新点在于：① 将多模态条件（人像、衣物、姿态、分割、文本）融入流匹配 Transformer 的潜在空间；② 引入时序自引用机制和 3D-RoPE 实现条件组的参数无关编码；③ 采用分层空间条件压缩和区域加权损失提升细节保真与效率。

**🔧 技术方法**

核心技术包括流匹配 Transformer (Flow Matching DiT)、LoRA 微调、时序自引用（Temporal Self‑Reference）、3D‑RoPE 条件分组、区域加权损失、以及自研的 Style Prompt System。

**📊 数据集**

使用公开数据集 VITON‑HD、DressCode，并在自采的 13 张人像与 40 件衣物构成的 520 对野外样本上进行评估。

**📈 对比分析**

与现有 VTON 方法（LaDI‑VTON、CatVTON、OOTDiffusion、Any2AnyTryon、PromptDresser）以及通用图像编辑模型（Seedream 4.0、Qwen‑Image‑Edit、Nanobanana）对比，实验表明该方法在 SSIM、LPIPS、FID、KID 等指标上均优于对手，同时推理速度提升约 30‑40%。

**⚠️ 局限性**

局限性包括：① 仍依赖大规模预训练模型，推理成本不低；② 对极端姿态或高度遮挡场景的鲁棒性尚待进一步提升；③ Style Prompt 需要额外的文本生成模型，可能导致提示误差。

---

## 358. To Believe or Not To Believe: Comparing Supporting Information Tools to Aid Human Judgments of AI Veracity

**arXiv ID:** 2603.11393 | [PDF](https://arxiv.org/pdf/2603.11393v1)

**作者:** Jessica Irons `[一作]` (Commonwealth Scientific and Industrial Research Organisation), Stephen Wan `[通讯]` (Commonwealth Scientific and Industrial Research Organisation)

**通讯引用:** 1573 | [OpenAlex ID](https://openalex.org/A5008748003)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在三种支持信息工具（完整PDF、BM25检索段落、LLM解释）下，对用户进行AI生成答案的真实性评估，并通过实验测量效率、准确度、信任与依赖度。

**💡 创新点**

发现检索段落能够在保持准确度的前提下显著提升评估速度；LLM解释虽然同样快速，却导致过度依赖并降低错误检测率，尤其对复杂答案更为显著。

**🔧 技术方法**

使用BM25文本检索模型、GPT‑OSS:20B生成答案与解释、混合实验设计与标准问卷（NASA‑TLX、S‑TIAS）。

**📊 数据集**

基于三篇科普性科学文章（The Conversation）中抽取的五个字段（地点、技术、威胁、主要发现、建议行动）生成的15条答案。

**📈 对比分析**

通过三组对比实验（PDF、TopK、LLM）统计响应时间、准确率、信心、接受率等指标；TopK在响应时间上最快且保持准确度≈85%，LLM在接受率上最高但准确率最低。

**⚠️ 局限性**

样本为普罗大众，缺乏专业领域背景；只测试单一信息来源，未考虑多源混合或动态适配；LLM解释为决策性叙述，未尝试更客观或对比式解释。

---

## 359. SELF-VLA: A Skill Enhanced Agentic Vision-Language-Action Framework for Contact-Rich Disassembly

**arXiv ID:** 2603.11080 | [PDF](https://arxiv.org/pdf/2603.11080v1)

**作者:** Chang Liu `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5066836550)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SELF-VLA框架，将可重用的拆装技能嵌入视觉‑语言‑动作模型，实现机器人在端到端和技能库之间自动切换，完成电子废弃物拆解任务。

**💡 创新点**

创新点在于将结构化拆装技能与VLA模型融合，提供显式的停止信号、失败检测和恢复机制，从而提升长时序、接触密集任务的成功率。

**🔧 技术方法**

使用了LoRA微调的VLA模型（OpenVLA、OpenVLA‑OFT、π_0.5、π_0.5‑Droid），结合视觉‑语言理解、动作生成、技能库规划与纠正器，编码停止令的抓手动作实现模块切换。

**📊 数据集**

构建了528条真实拆解演示数据集（CPU提取和RAM移除），在双摄像头视角下采集，包含不同位置与方向的八种配置，并生成训练数据用于规划、纠正和端到端模型。

**📈 对比分析**

与纯端到端VLA基线相比，SELF‑VLA在CPU提取任务中提升了约31%、在RAM移除任务中提升约17%；在10Hz训练数据下表现更好，某些基线模型（如π_0.5‑Droid）在SELF‑VLA下完成率可达80%。

**⚠️ 局限性**

局限性包括对未见空间配置的泛化不足、OpenVLA模型难以产生准确的预接触姿态导致的失败、对停止信号编码的依赖以及在更复杂、真实环境中的性能验证不足。

---

## 360. Single Pixel Image Classification using an Ultrafast Digital Light Projector

**arXiv ID:** 2603.12036 | [PDF](https://arxiv.org/pdf/2603.12036v1)

**作者:** Aisha Kanwal `[一作]` (Institute of Photonics), Michael J. Strain `[通讯]` (Institute of Photonics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实验验证了利用微LED投影机和12×12哈达玛模式实现单像素成像（SPI）进行MNIST数字分类，并在1.2kfps下达成90%以上准确率。

**💡 创新点**

在不重建图像的前提下直接用时域光信号做分类，结合超高速微LED投影和极低复杂度的极限学习机(ELM)，实现极快的推理速度与高准确率。

**🔧 技术方法**

单像素成像、哈达玛模式投影、微LED数字光投影机、单像素光电探测、极限学习机与深度前馈神经网络。

**📊 数据集**

MNIST手写数字数据集（28×28像素，经过二值化后映射至DMD）。

**📈 对比分析**

将ELM与训练好的DNN在全哈达玛模式下以及不同子集模式下进行对比；DNN在完整模式下准确率>90%，ELM在完整模式下87%，但ELM推理速度快≈2×。

**⚠️ 局限性**

受限于哈达玛模式数目导致的图像信息损失，降采样子集会显著降低准确率，且对高噪声敏感，尚未实现多光谱或高分辨率场景。

---

## 361. Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs

**arXiv ID:** 2603.11495 | [PDF](https://arxiv.org/pdf/2603.11495v1)

**作者:** Kunfeng Chen `[一作]` (Wuhan University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 99642 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了 Tool-DC 框架，利用分治与自我反思提升 LLM 在长上下文工具调用任务中的表现。

**💡 创新点**

提出了训练无关的 Try‑Check‑Retry 分治流程和训练有监督的 Tool-DC (TB) 方法，有效解决工具数量大且噪声多时的推理困难。

**🔧 技术方法**

采用 Strategic Anchor Grouping 进行分组检索、局部推理、基于 schema 的规则校验以及自我反思聚合；TB 通过收集 CoT 训练数据并对模型微调实现内部化。

**📊 数据集**

使用 BFCL 与 ACEBench 基准，并在扩展设置中加入噪声工具；CoT 训练集基于 xlam-function-calling-60k 构造。

**📈 对比分析**

与 All_Funs、Top‑K、HiTEC‑ICL、ToolGT 等基线对比，TF 在标准/扩展设置下平均提升至 +25% 以上；TB 在 BFCL 上 Qwen2.5‑7B 达到 83.16%，超过 OpenAI o3、Claude‑Haiku‑4.5 等专有模型。

**⚠️ 局限性**

仅针对单步调用场景，训练集多样性不足，未评估多步嵌套情境，对重放训练的依赖可能限制泛化。

---

## 362. How Intelligence Emerges: A Minimal Theory of Dynamic Adaptive Coordination

**arXiv ID:** 2603.11560 | [PDF](https://arxiv.org/pdf/2603.11560v1)

**作者:** Stefano Grassi `[一作]` `[通讯]` (Bangkok University), Stefano Grassi (Bangkok University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出一种递归闭环的动态协调理论，阐释多智能体在持久环境与激励场耦合下如何产生智能化协同；

**💡 创新点**

核心创新是将持久环境记忆、分布式激励以及局部适应更新纳入同一闭环结构，证明在缺乏全局目标优化的前提下即可实现稳定、历史敏感的协调；

**🔧 技术方法**

使用连续动力学与离散时间系统的数学分析、守恒与耗散理论、Jacobian特征值判定及线性化稳定性分析；

**📊 数据集**

论文未使用传统实验数据集，而是通过构造的最小线性系统与仿真验证结构性条件；

**📈 对比分析**

比较方式为理论推导与数值模拟相结合，结果表明在满足耗散与非平凡激励-记忆耦合的参数区间内系统表现出收敛与振荡；

**⚠️ 局限性**

局限在于未给出全球收敛保证，仅提供局部稳定性；对复杂非线性或随机扰动的鲁棒性分析仍需进一步研究。

---

## 363. Context Before Code: An Experience Report on Vibe Coding in Practice

**arXiv ID:** 2603.11073 | [PDF](https://arxiv.org/pdf/2603.11073v1)

**作者:** Md Nasir Uddin Shuvo `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10266 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两种生产环境下使用上下文对话式vibe coding构建了一个多项目智能学习平台和一个学术检索增强生成系统。

**💡 创新点**

首次系统性记录了AI辅助编码在满足多租户隔离、访问控制、异步处理等关键架构约束时的局限与实践经验，并提出了“非委派区域”的概念。

**🔧 技术方法**

利用Python后端、OpenAI GPT‑4.1、异步工作队列、关系数据库、向量检索等技术栈，并通过上下文提示工程实现代码生成。

**📊 数据集**

采用团队上传的业务文档和学生/研究者上传的学术资料，未使用公开数据集。

**📈 对比分析**

未进行实验对比，仅通过结构化手工测试、代码审查和运行日志验证架构满足性，未给出量化性能指标。

**⚠️ 局限性**

研究基于两款系统的小型团队完成，方法回溯性分析，缺乏大规模验证和客观生产性能评估。

---

## 364. A Decade of Generative Adversarial Networks for Porous Material Reconstruction

**arXiv ID:** 2603.11836 | [PDF](https://arxiv.org/pdf/2603.11836v1)

**作者:** Ali Sadeghkhani `[一作]` (University of Leeds), Arash Rabbani `[通讯]` (University of Leeds)

**通讯引用:** 2151 | [OpenAlex ID](https://openalex.org/A5059420861)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2017–2026年间发表的96篇论文，系统评估了GAN在多孔材料重建中的应用。

**💡 创新点**

提出六大GAN架构类别（Vanilla、Multi‑Scale、Conditional、Attention‑Enhanced、Style‑based、Hybrid）并梳理其技术演进与性能提升。

**🔧 技术方法**

采用多种GAN技术，包括经典DCGAN、WGAN‑GP、Progressive Growing、SinGAN、CBAM‑增强等，结合对多尺度、条件控制与注意力机制的改进。

**📊 数据集**

利用公开的数字岩石数据库（如Berea砂岩、Ketton石灰岩等）与实验CT、SEM等真实多孔样本作为训练与验证数据集。

**📈 对比分析**

通过孔隙率误差≤1%、渗透率误差降低79%、可生成至2200³体素等定量指标，显示相较传统统计重建方法性能显著提升。

**⚠️ 局限性**

仍面临计算效率低、GPU内存瓶颈、2D→3D连续性和结构完整性保持等技术挑战。

---

## 365. Witnesses for Fixpoint Games on Lattices

**arXiv ID:** 2603.11908 | [PDF](https://arxiv.org/pdf/2603.11908v1)

**作者:** Barbara König `[一作]` (Universität Duisburg-Essen), Karla Messing `[通讯]` (Universität Duisburg-Essen)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套通用的基于格理论与 Galois 连接的框架，用来构造“witness”(即区分公式)并将其转换为 fixpoint 游戏中的赢策略，进而证明某个最小 fixpoint 不是给定元素的上界。框架同时覆盖两种游戏：原始(Primal)和对偶(Dual)；通过该框架对 bisimilarity、概率系统的行为度量以及马尔可夫链的终止概率进行了实例化，重新推导了已知的区分公式构造，并给出了新的终止概率下界证明方法。

**💡 创新点**

创新点包括：① 将区分公式抽象为格中的 witness，并证明其与 fixpoint 游戏赢策略之间的双向可转化；② 通过 Galois 连接实现逻辑宇宙与行为宇宙的统一映射；③ 引入“way‑below/way‑above”度量的 ordinal 维度，保证了策略的有限性；④ 对原始和对偶游戏分别给出了策略生成与 witness 构造的递归算法；⑤ 在概率行为度量和马尔可夫链终止概率的实例中，首次将上述框架与 Kantorovich 升维、树式 witness 等具体工具结合。

**🔧 技术方法**

技术手段主要包括：格理论（连续格、共连续格、基、way‑below/way‑above关系）、Galois 连接与 adjoint 逻辑、Scott 拓扑、Kleene 迭代、fixpoint 游戏（原始与对偶）、有限策略的构造与证明、递归 witness 构造、概率与期望的算子（Kantorovich 提升、期望算子）以及树形结构的终止概率估计。

**📊 数据集**

论文未使用实际实验数据集，所有结果均为理论证明。实例化仅涉及：无标签转移系统（bisimilarity）、带标签的概率马尔可夫链（行为度量）、以及有限状态的马尔可夫链（终止概率）。

**📈 对比分析**

由于本工作为理论框架，未进行数值实验或性能评估。作者通过数学证明证明了有限策略的存在性和 witness 与策略之间的等价性；在实例化部分与已有工作（如 Hennessy‑Milner、Kantorovich 归约、传统 bisimulation game）对比，显示了框架的通用性和对传统方法的重新解释，但没有给出计算复杂度或实验对比。

**⚠️ 局限性**

主要局限包括：① 需要行为格是连续或共连续，才能保证有限策略与 witness 的存在；② witness 的构造依赖于递归求解不等式，实际实现可能较为复杂；③ 对于无限或非连续格的系统框架尚不适用；④ 论文未给出算法实现与实测复杂度，实际可行性待进一步评估。

---

## 366. Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

**arXiv ID:** 2603.12262 | [PDF](https://arxiv.org/pdf/2603.12262v1)

**作者:** Yiran Guan `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38756 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Video Streaming Thinking (VST) 框架，使视频大语言模型在观看视频时同步生成链式思考，实现在线推理。

**💡 创新点**

创新点在于将链式思考预先并行于视频流，分摊推理延迟，并通过两阶段后训练（VST‑SFT + VST‑RL）与基于知识图谱的自动数据生成实现流式推理。

**🔧 技术方法**

采用 Qwen2.5‑VL 基础模型、流式注意力掩码、第一进先出文本记忆、强化学习奖励、自动知识图谱 + Chain‑of‑Thought 生成等技术。

**📊 数据集**

使用 LLaVA‑Vid、Video‑Marathon、Onethinker、RepCount 等视频数据集，并合成 100K 条流式推理示例进行训练。

**📈 对比分析**

在 StreamingBench、OVO‑Bench、VideoMME、LongVideoBench、VideoHolmes 等基准上与现有 SOTA 对比，VST‑7B 在 StreamingBench 79.5%、OVO‑Bench 59.3%，相较 GPT‑4o 等模型显著提升准确率且响应速度快。

**⚠️ 局限性**

局限性包括思考步骤仍占用大量 LLM token，导致计算成本；仅使用文本记忆，未与视觉记忆机制结合，需进一步探索更高效的推理方式。

---

## 367. Parameter unbounded Uzawa and penalty-splitted accelerated algorithms for frictionless contact problems

**arXiv ID:** 2603.12205 | [PDF](https://arxiv.org/pdf/2603.12205v1)

**作者:** Daria Koliesnikova `[一作]` (French Alternative Energies and Atomic Energy Commission), Isabelle Ramière `[通讯]` (French Alternative Energies and Atomic Energy Commission)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一的迭代分裂框架，用标准刚度矩阵求解刚体接触问题，并通过Crossed‑Secant加速实现参数无关的收敛。

**💡 创新点**

创新点在于将Crossed‑Secant固定点加速方法引入Uzawa及惩罚分裂方案，消除了传统方法对增量/惩罚参数的严格限制，显著提高了收敛速度和鲁棒性。

**🔧 技术方法**

采用Uzawa算法、惩罚分裂、投影-固定点迭代、Crossed‑Secant加速、FISTA、Anderson加速与自适应重启等技术；所有迭代只需求解一次不变刚度矩阵。

**📊 数据集**

使用三维学术Hertz接触、工业热诱导Pellet‑Cladding接触以及多域接触的有限元数据集（Cast3M求解器构建），覆盖从单体到多体的接触规模。

**📈 对比分析**

与传统Uzawa、惩罚、以及基于Lagrange乘子/鞍点矩阵的直接求解方法比较；通过迭代次数、误差指标（力、位移、间隙、互补条件）和计算时间评估。实验显示Crossed‑Secant在所有参数范围内迭代次数最少、误差达到机器精度，并在多域、并行环境下比传统方法快约1–2个数量级。

**⚠️ 局限性**

仍需针对非线性材料、摩擦接触以及更高效的并行线性求解器进行扩展；部分加速方法对线性求解器实现依赖，且在极大参数下可能出现数值不稳定；当前仅验证了无摩擦情况。

---

## 368. Toward Complex-Valued Neural Networks for Waveform Generation

**arXiv ID:** 2603.11589 | [PDF](https://arxiv.org/pdf/2603.11589v1)

**作者:** Hyung-Seok Oh `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**通讯引用:** 21721 | [OpenAlex ID](https://openalex.org/A5011014617)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了ComVo，一个基于iSTFT的复杂域对抗式语音合成器

**💡 创新点**

使用CVNN生成器与判别器、相位量化层以及块矩阵计算方案实现复杂域建模与训练加速

**🔧 技术方法**

核心技术包括复杂值卷积网络、相位量化非线性、块矩阵实现的复数运算以及GAN对抗训练

**📊 数据集**

在LibriTTS（语音）和MUSDB18-HQ（混音）数据集上进行训练与评估

**📈 对比分析**

与HiFi‑GAN、iSTFTNet、BigVGAN、Vocos等基线对比，MOS、UTMOS、PESQ等指标均优于对手，并将训练时间缩短约25%

**⚠️ 局限性**

局限性包括仅在单GPU下实验、采用拆分式设计、复杂参数在多GPU环境下效率不高以及数值稳定性待进一步改进

---

## 369. A Machine Learning-Enhanced Hopf-Cole Formulation for Nonlinear Gas Flow in Porous Media

**arXiv ID:** 2603.11250 | [PDF](https://arxiv.org/pdf/2603.11250v1)

**作者:** V. S. Maduru `[一作]`, K. B. Nakshatrala `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

针对低压气体在多孔介质中的非达西输运，提出了一套基于Hopf–Cole变换、混合公式、共享‑桩神经网络和Deep Least‑Squares求解器的端到端学习框架，能够在保持数值稳定性的同时精确预测压强与流速场。

**💡 创新点**

创新点主要包括：
① 将Klinkenberg压强依赖的渗透率通过Hopf–Cole变换线性化，使原本的非线性方程转化为等效的Darcy型线性系统；
② 采用混合（pressure–velocity）形式并在共享‑桩网络中同时输出两场，保证场间一致性与更好的流速逼近；
③ 采用最小二乘能量泛函而非强式残差，得到对称正定目标函数，显著提升训练稳定性；
④ 对整个框架给出了严格的收敛性和误差分解分析，证明了从网络参数、采样点到物理解的误差可以被控制。

**🔧 技术方法**

技术手段包括：Hopf–Cole变换、混合Darcy公式、共享‑桩深度神经网络、DeepLS最小二乘求解、Monte Carlo积分、Lambert‑W逆变换、自动微分、Adam+L‑BFGS两阶段优化与自适应权重调节。

**📊 数据集**

使用的“数据集”是多组基准数值实验：
- 同心圆柱/同心球壳（解析解可比）
- 基础板问题（与稳定化混合有限元对比）
- 分层多孔介质（含材料界面）
- 以上四个问题均在合成域上以多点采样形式生成训练与验证点，全部为人工生成的仿真数据。

**📈 对比分析**

与解析解或高阶有限元（FEM）解比较：
- L2误差随网络容量（宽度/深度）显著下降，数值误差在10⁻³–10⁻⁴量级；
- 训练时间仅为数分钟（NVIDIA T4 GPU），显著低于传统网格方法；
- 误差随采样点密度提升而收敛，表明采样策略与自适应权重能有效控制残差。

**⚠️ 局限性**

局限性与待改进：
- 仅针对单相气体、稳态、Klinkenberg模型，缺乏多相或时间动态扩展；
- 对高非线性或极端低压（Knudsen数很大）场景的数值稳定性尚未充分验证；
- 依赖人工生成的基准问题，缺乏真实井筒或地质数据的验证；
- 网络结构与超参数调节仍以经验为主，缺少自适应架构搜索；
- 对不确定性量化、数据驱动校准等方向仍需进一步研究。

---

## 370. Exploiting Expertise of Non-Expert and Diverse Agents in Social Bandit Learning: A Free Energy Approach

**arXiv ID:** 2603.11757 | [PDF](https://arxiv.org/pdf/2603.11757v1)

**作者:** Erfan Mirzaei `[一作]` (University of Tehran), Majid Nili Ahmadabadi `[通讯]` (University of Tehran)

**通讯引用:** 3452 | [OpenAlex ID](https://openalex.org/A5072135442)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于自由能（Free‑Energy）范式的社会多臂赌博机学习算法（SBL‑FE），利用观测到的其他代理行为来改进自身策略，同时保持对自身经验的自我评估。

**💡 创新点**

创新点：① 在策略空间而非奖励空间定义自由能度量，结合 Thompson Sampling 的不确定性信息；② 通过该度量同时考虑自我参考、他者相似性与策略熵，实现对多样化、甚至非专家代理的自动识别与利用；③ 无需共享奖励或任务信息，适用于异质且隐藏信息的社会学习场景。

**🔧 技术方法**

核心技术：Thompson Sampling（用于估计自身和他者的策略），自由能最小化（在策略空间中的KL与熵权衡），指数滑动平均估计他者行为，理论证明收敛至最优策略；实现复杂度为 O(NK) 每步。

**📊 数据集**

使用伯努利分布的多臂赌博机（K=10 或 2，最优奖励 0.9），通过不同最优差距（Δ=0.05,0.1,0.2）和多种社会配置（非学习者、学习者、不同任务、噪声、动作集差异）进行仿真。

**📈 对比分析**

与 OUCB、TUCB、TS、UCB 进行对比；实验显示在存在非专家或无专家时 SBL‑FE 能快速识别相关代理并显著降低累计惩罚；在多代理、多噪声、不同动作集下仍保持低惩罚；在包含最优代理的场景下表现与 TUCB 相当或更优；整体保持对数级别的 regret。

**⚠️ 局限性**

局限性：① 计算开销较大（需多次 Thompson Sampling 与自由能更新）；② 仅在单臂赌博机中验证，非马尔可夫决策过程（MDP）未完全适配；③ 对多社会学习者的扩展、非平稳任务、异动作集的细化处理仍待研究；④ 当最优行动与观察到的代理最优行动不匹配时，算法可能偏慢。

---

## 371. SVLL: Staged Vision-Language Learning for Physically Grounded Embodied Task Planning

**arXiv ID:** 2603.11563 | [PDF](https://arxiv.org/pdf/2603.11563v1)

**作者:** Yuyuan Yang `[一作]` (Chinese University of Hong Kong, Shenzhen), Jinke Ren `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 2557 | [OpenAlex ID](https://openalex.org/A5036242436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了三阶段的 Staged Vision‑Language Learning (SVLL) 框架，用于将视觉与语言模型在具身任务规划中分阶段学习，以避免过早时间绑定和因果不对齐。

**💡 创新点**

创新点在于(1) 将空间感知与时间推理解耦，先用无历史上下文的 Stage 1 训练空间感知，再在 Stage 2 引入序列；(2) 在 Stage 3 引入 Bias‑DPO 目标，将相对偏好优化转为绝对概率驱动，强制模型对专家轨迹有偏好并抑制自信幻觉。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B 作为基础模型，结合 LoRA、B‑DPO、标注平滑、阈值抑制等技术，并在三个阶段分别采用监督微调、序列微调和偏好优化。

**📊 数据集**

数据集为 AI2‑THOR 的交互式轨迹数据集，包含多种复杂任务；此外还在真实机器人平台上进行零样本部署测试。

**📈 对比分析**

通过在 AI2‑THOR 基准上与 GPT‑4o、Gemini‑2.0‑flash、RoboBrain‑2.0‑32B 等闭源/开源模型对比，SVLL‑Stage 3 在成功率 78.35% 和约束违规率 26.34% 上均优于基线；在真实机器人上 7B 模型实现 55.56% 成功率，约束违规率 4.35%。

**⚠️ 局限性**

局限包括对专家示例质量的高度依赖，过度惩罚可能导致模型停留在次优策略；以及从仿真到真实的领域漂移，需进一步增强视觉鲁棒性。

---

## 372. Deployment-Time Reliability of Learned Robot Policies

**arXiv ID:** 2603.11400 | [PDF](https://arxiv.org/pdf/2603.11400v1)

**作者:** Christopher Agia `[一作]` (Stanford University), Christopher Agia `[通讯]` (Stanford University)

**通讯引用:** 647 | [OpenAlex ID](https://openalex.org/A5084514643)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文围绕提升学习型机器人策略在真实部署中的可靠性展开，提出了三类关键机制：① Sentinel实时监控框架，利用时序一致性统计(STAC)与视觉语言模型(VLM)双重检测，快速发现生成式策略的“狂躁失控”与“任务停滞”两类失败；② CUPID数据稀疏化方法，借助影响函数评估单条演示对闭环返回的因果影响，进而对演示数据进行筛选/选择，实现基于性能的高效数据精炼；③ STAP技能序列规划框架，结合Q函数与动力学模型，优化多技能序列的参数，解决长周期任务中的几何依赖与连贯性问题，并在真实机器人上验证。

**💡 创新点**

创新点包括：① 将时序一致性与VLM结合，首次针对生成式策略设计双层失效检测；② 将影响函数引入机器人模仿学习，实现对演示数据的因果解释与高效裁剪；③ 通过离散化Q值与动力学预测实现无训练集约束的技能序列规划，支持未见任务的即时组合；④ 在三个子领域形成统一的部署时可靠性范式。

**🔧 技术方法**

技术手段涵盖：生成式策略（扩散模型）与其采样方法；统计距离度量（MMD、KL）与累积一致性指标；多模态视觉语言模型（CLIP/BLIP）用于视频问答；影响函数与TRAK近似；Q函数学习与模型预测动力学；不确定性量化与置信度门控；基于任务规划的几何约束。

**📊 数据集**

实验使用的主要数据集包括：RoboMimic仿真演示集（Lift、Square、Transport等多人任务）、真实Franka FR3实验环境（Push Chair、Close Box、Cover Object等任务）、自制的Figure‑8、TuckBox、Bookshelf等混合质量/策略不平衡数据集；此外在STAP章节使用标准机器人操控数据与图像序列。

**📈 对比分析**

对比方法覆盖多类失败检测（基于重构误差、输出方差、相似度、DDPM损失等）、数据稀疏化基线（DemInf、Demo‑SCORE、Success Similarity、Oracle、Random）以及技能规划对手（DAF）。性能结果显示：Sentinel在模拟与实机中将检测准确率提升至>90%，TPR≥93%；CUPID在混合质量任务中显著提高成功率（多达+84%），并在策略鲁棒性和去相关性实验中超过对手；STAP在长周期几何依赖任务上实现了比传统序列化方法更高的成功率与规划效率。

**⚠️ 局限性**

局限性包括：① 失效检测尚未提供理论保证；② 影响函数估计对roll‑out数量敏感，导致高方差；③ 组合多检测器时可能产生误报；④ 数据稀疏化假设线性可加，忽略示例间交互；⑤ 现有方法需先收集足够的成功roll‑out，真实环境中收集成本高；⑥ STAP对动力学模型精度与不确定性估计高度依赖，模型失真会影响规划。

---

## 373. Space-Efficient Approximate Spherical Range Counting in High Dimensions

**arXiv ID:** 2603.12106 | [PDF](https://arxiv.org/pdf/2603.12106v1)

**作者:** Andreas Kalavas `[一作]` (Carnegie Mellon University), Ioannis Psarros `[通讯]` (Archimedes)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

提出了一种在高维欧氏空间中对球形范围计数进行 ε-近似的近线性空间数据结构。

**💡 创新点**

创新点在于构造具有低 ε-刺穿数的生成树并利用其生成分区树，从而使查询时间仅与模糊层内点数相关，且空间仍保持近线性。

**🔧 技术方法**

使用了局部敏感哈希、Johnson–Lindenstrauss 降维、多重权重更新、VC 维学习理论以及分区树/生成树技术。

**📊 数据集**

文中未给出具体实验数据集，主要是理论分析与算法构造。

**📈 对比分析**

与传统基于 LSH 的计数方案相比，本工作在空间上保持近线性，在查询时当模糊层点数 t_q 为子线性时可实现子线性查询时间；理论上满足 n^{1-Θ(ε^4/ log(1/ε))} + t_q^ρ·n^{1-ρ} 的时间复杂度。

**⚠️ 局限性**

限制包括：查询最坏情况仍可退化为线性；实现需要复杂的预处理和参数调优；对极高维度或极小 ε 时的常数因子可能较大，且缺乏实验验证。

---

## 374. A Generalized Theory of Load Distribution in Redundantly-actuated Robotic Systems

**arXiv ID:** 2603.11431 | [PDF](https://arxiv.org/pdf/2603.11431v1)

**作者:** Joshua Flight `[一作]`, Clément Gosselin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种统一的负载分布理论，描述多链冗余机器人在给定合力下如何分配施加力矩和力以消除或控制内部载荷，涵盖抓取、步行和协同操纵等多种场景。

**💡 创新点**

通过引入动量等效系统和Udwadia‑Kalaba 方程，给出闭式参数化的 Moore‑Penrose 伪逆，消除了先前研究中对内部载荷的错误假设，并提供唯一的内部载荷表示，显著提升了推算精度与可解释性。

**🔧 技术方法**

主要采用刚体动力学分析、动量等效建模、Udwadia‑Kalaba 约束方程、参数化 Moore‑Penrose 伪逆和线性代数工具（矩阵分解、核空间分析）。

**📊 数据集**

论文不使用外部数据集，而是通过一系列平面、三维几何实例（如等边三角形、单位球面四点力矩）进行验证与对比。

**📈 对比分析**

与传统的迭代优化方法和已发表的等效伪逆公式相比，新方法无需求逆大矩阵，计算复杂度线性增长，能在实时控制循环中快速给出无内部载荷或指定内部载荷的完整力矩分配，并在案例中展示了更高的数值精度和物理一致性。

**⚠️ 局限性**

主要限制在于：若想完全满足内部载荷消除需至少10个独立力矩（实际机器人难实现）；在降低约束的简化方案中仍可能产生内部载荷；此外对非理想接触模型（摩擦、滑动）需进一步扩展。

---

## 375. GGPT: Geometry Grounded Point Transformer

**arXiv ID:** 2603.11174 | [PDF](https://arxiv.org/pdf/2603.11174v1)

**作者:** Yutong Chen `[一作]` (ETH Zurich), Siyu Tang `[通讯]` (ETH Zurich)

**通讯引用:** 6950 | [OpenAlex ID](https://openalex.org/A5056265728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Geometry‑Grounded Point Transformer（GGPT），通过改进的 SfM 生成稀疏几何指引，再在 3D 空间里对 feed‑forward 3D 重建结果进行细化

**💡 创新点**

①将 SfM 作为稀疏几何先验直接嵌入到 3D 点 transformer 中；②在 3D 位置上而非 2D 图像空间执行注意力；③采用轻量级 PTv3 与分块处理提升效率；④在单一训练集上即可实现跨架构、跨数据集的泛化

**🔧 技术方法**

dense feature matching（RoMa、UFM）、轻量级 bundle adjustment + direct linear triangulation、3D Point Transformer (PTv3)、patch‑based处理、异方差（confidence‑weighted）损失、UMeyama 对齐

**📊 数据集**

训练：ScanNet++（20k 视图序列）；评估：ScanNet++、ETH3D、T&T；跨域测试：人类身体（4D‑DRESS）、腹部手术（MV‑dVRK）

**📈 对比分析**

与多种 feed‑forward 3D 重建模型（VGGT、DUSt3R、MASt3R 等）和 2D 深度补全方法对比，GGPT 在所有基准下均提升 AUC@1cm（或 mm）与整体误差；在跨域数据上也显著优于 Pi3 等先进模型；SfM 部分在相机姿态与点云质量上优于现有全局 SfM 方法

**⚠️ 局限性**

依赖稀疏几何指引，若稀疏点云缺失严重（纹理稀疏、视角极少）仍可能无法完全纠正误差；密集匹配和 SfM 仍是运行瓶颈；仅在室内场景或人类/手术场景中验证，需进一步测试更广泛的多模态数据

---

## 376. Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis

**arXiv ID:** 2603.12083 | [PDF](https://arxiv.org/pdf/2603.12083v1)

**作者:** Xiaolong Qian `[一作]` (Zhejiang University), Kaiwei Wang `[通讯]` (Zhejiang University)

**通讯引用:** 4795 | [OpenAlex ID](https://openalex.org/A5018263416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向摄影相机的通用计算像差校正（CAC）基准 UniCAC，利用自动光学设计生成大规模球面/非球面镜头库，提出 Optical Degradation Evaluator（ODE）评估框架，对24种 CAC/IR 方法在多维光学退化（强度、空间均匀度、色差）下进行系统评估，并归纳9条关键观察。

**💡 创新点**

①首个涵盖众多球面与非球面镜头、可实现零样本泛化的通用 CAC 基准；②将 OptiFusion 自动光学设计扩展到非球面参数，生成高质量镜头集合；③提出 ODE 通过 OIQ、空间均匀度与色差三维度量化 CAC 难度，取代传统 RMS；④系统评估24模型并提炼先前未被充分讨论的先验、架构与训练策略对 CAC 性能的影响。

**🔧 技术方法**

使用的核心技术包括：OptiFusion（扩展版）自动光学设计、PSF 采样与卷积退化模拟、ODE 评估框架（PSNR/SSIM/OIQE 及 U_s、U_c 变异系数）、CNN、GAN、扩散模型（如 FeMaSR、DiffBIR、PART、SwinIR）等学习架构，以及多指标综合评价（OP = 0.4×PSNR/50 + 0.3×SSIM–0.5/0.5 + 0.4×(1–LPIPS/0.4) + 0.3×OIQE + 0.1×100–FID/100 + 0.1×ClipIQA）。

**📊 数据集**

数据集主要包括：①3000 张来自 Flickr2K、DIV2K 的 GT 图像作为训练输入；②26 张高分辨率 GT 图像与 120 个测试镜头（由自动设计生成）做测试；③训练镜头库 873 个、测试镜头 120 个；④通过与 Zemax 模拟结果和真实拍摄对比验证仿真质量。

**📈 对比分析**

通过在 5 级 ODE 退化、空间均匀度与色差子基准上对 24 模型进行统一评价，发现：学习型方法整体优于基于优化的算法；回归训练提升 PSNR；GAN/扩散训练提升 LPIPS、ClipIQA；PSF 关注机制（PART）在多镜头下表现最好；CNN 架构兼顾性能与速度；扩散模型在高退化场景中最具优势；OP 排名中 FeMaSR、DiffBIR、PART 等模型居前。

**⚠️ 局限性**

局限性包括：①缺乏针对光学质量提升（OIQE）专门的训练策略；②色差（U_c）对 ODE 与 OP 的贡献有限，可能忽视某些实际镜头的色差问题；③基准主要针对消费级相机镜头，未覆盖显微镜、望远镜等专业光学；④所有结果基于仿真光学，虽然与 Zemax 及真实拍摄对齐，但仍受限于仿真模型；⑤ODe 取决于参数权重设定，需进一步泛化。

---

## 377. EgoIntent: An Egocentric Step-level Benchmark for Understanding What, Why, and Next

**arXiv ID:** 2603.12147 | [PDF](https://arxiv.org/pdf/2603.12147v1)

**作者:** Ye Pan `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1219 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了Egocentric step-level intent benchmark EgoIntent，对3,014个第一人称视频步骤进行本地意图、全局意图和下一步计划的标签，并通过预结果截断防止信息泄漏。

**💡 创新点**

创新点包括细粒度step-level意图理解、三维意图维度（What/Why/Next）、预结果截断机制、开放式生成评估以及基于语义一致性的自动化评测。

**🔧 技术方法**

使用多模态大型语言模型（MLLM）进行评估，采用Qwen3-based语义一致性评估框架，并统一提示与视频帧序列输入。

**📊 数据集**

基于GoalStep/Ego4D数据集，在15个室内外日常情境中抽取3,014个步骤构成Bench。

**📈 对比分析**

通过统一提示、6 FPS图像序列输入、Qwen3评测器对15个MLLM进行对比，最佳整体得分仅为33.31，显示现有模型在step-level意图理解上仍表现有限。

**⚠️ 局限性**

限制包括对未来信息泄漏的敏感性、开放式生成评估的主观性、模型对细粒度本地意图和下一步计划推理的能力不足，以及对不同场景泛化能力不够。

---

## 378. Locating Demographic Bias at the Attention-Head Level in CLIP's Vision Encoder

**arXiv ID:** 2603.11793 | [PDF](https://arxiv.org/pdf/2603.11793v1)

**作者:** Alaa Yasser `[一作]` (Universidad Autónoma de Madrid), Jenny Benois-Pineau `[通讯]` (University of Bordeaux)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种机制公平审计方法，在 CLIP 视觉编码器中定位性别与年龄偏见到单个注意力头层面，并通过平均消融验证其有效性。

**💡 创新点**

创新点在于将零样本概念激活向量（CAV）与 TextSpan 词典扩展为人口属性，结合投影残差分解实现头级偏见定位，并首次表明性别偏见集中于少数终端头，而年龄偏见分布更为分散。

**🔧 技术方法**

使用技术包括投影残差流分解、零样本 CAV 方向、偏见增强 TextSpan、均值消融、层匹配随机控制、卡方检验和 Cramér's V 统计。

**📊 数据集**

实验基于 FACET 基准（42个职业类别）和预训练 CLIP ViT‑L‑14 视觉编码器。

**📈 对比分析**

通过层匹配随机控制对比，识别出的四个性别偏见头消融后全局 Cramér's V 从 0.381 降至 0.362，整体准确率提升 0.42%；年龄头消融效果弱，未显著降低偏差。

**⚠️ 局限性**

局限性包括：仅关注终端层；非二元样本不足导致统计缺失；均值消融缺乏细粒度控制；零样本 CAV 对年龄属性可能不够敏感；搜索阈值选择可能引入偏差。

---

## 379. TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision

**arXiv ID:** 2603.11557 | [PDF](https://arxiv.org/pdf/2603.11557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 380. Examining Users' Behavioural Intention to Use OpenClaw Through the Cognition--Affect--Conation Framework

**arXiv ID:** 2603.11455 | [PDF](https://arxiv.org/pdf/2603.11455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 381. RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks

**arXiv ID:** 2603.11558 | [PDF](https://arxiv.org/pdf/2603.11558v1)

**作者:** Ruiying Li `[一作]` (AgiBot), Yao Mu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 12500 | [OpenAlex ID](https://openalex.org/A5008178136)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个统一的 agentic 框架 RoboClaw，将数据采集、策略学习与长周期任务执行整合在同一 VLM 驱动的循环中。

**💡 创新点**

创新点包括引入 Entangled Action Pairs（前向与逆向动作耦合）实现自重置循环，使用基于 VLM 的高层链式思维决策与技能编排，以及闭环生命周期学习机制。

**🔧 技术方法**

技术手段包括 Vision–Language–Action（VLA）模型、VLM 作为元控制器、Model Context Protocol（MCP）工具接口、结构化记忆管理和链式思维（CoT）推理。

**📊 数据集**

使用了 Agibot G01 平台在卧室、厨房、书桌、便利店等真实场景中收集的演示与自动收集轨迹数据，构成多任务组织与检索数据集。

**📈 对比分析**

与手工演示+手动重置基线以及仅训练 π₀.₅ 模型对比，RoboClaw 在相同数据量下将人类投入降低约 2.16 倍、失败介入降低 8 倍，长周期任务成功率提升 25%，数据收集效率提升 53.7% 人力。

**⚠️ 局限性**

局限性在于依赖云端大型模型导致推理延迟，且需要可实现的逆向重置策略，复杂环境下的自适应能力仍有限。

---

## 382. Evaluation format, not model capability, drives triage failure in the assessment of consumer health AI

**arXiv ID:** 2603.11413 | [PDF](https://arxiv.org/pdf/2603.11413v1)

**作者:** David Fraile Navarro `[一作]` (Australian Institute of Health Innovation), Enrico Coiera `[通讯]` (Australian Institute of Health Innovation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在对Ramaswamy等人关于ChatGPT Health低估急诊风险的研究进行部分机制复制，比较了5款前沿LLM在受限（考试式）和自然（患者式）评估条件下的分诊准确率。

**💡 创新点**

创新在于揭示评估格式（强制A/B/C/D选项与对话式自由文本）是导致低估急诊风险的主因，并证明自然语言交互可显著提升模型分诊性能。

**🔧 技术方法**

使用了GPT‑5.2、Claude Sonnet 4.6/Opus 4.6、Gemini 3 Flash/3.1 Pro等LLM，采用多种评估格式和针对性消融实验。

**📊 数据集**

数据集为研究者自行编写的17个临床情景，包括DKA和哮喘等关键急诊案例，并使用作者公开的Ramaswamy原始提示进行子集验证。

**📈 对比分析**

对比方法为受限与自然条件下的准确率差异，结果显示自然条件下整体准确率提升6.4个百分点，哮喘分诊从48%升至80%。

**⚠️ 局限性**

局限在于未直接评估ChatGPT Health产品、采用单轮对话且判别者为LLM、实验集为自制情景而非完整原始30对情景。

---

## 383. Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems

**arXiv ID:** 2603.12023 | [PDF](https://arxiv.org/pdf/2603.12023v1)

**作者:** Sarbartha Banerjee `[一作]` (University of Texas at Austin), Mohit Tiwari `[通讯]` (Symmetry Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统化了跨软件、硬件与算法层的攻击手段，并展示了通过组合这些攻击手段在复合 AI 系统中突破安全防护的端到端攻击链。

**💡 创新点**

提出了“Cascade”红队框架，能够基于攻击目标与能力自动组合不同层次的攻击模块；并首次展示了将传统系统漏洞与 LLM 算法攻击协同放大 AI 安全与机密性威胁的实证。

**🔧 技术方法**

利用 LLM 推理生成攻击链、代码注入/SQL 注入、Rowhammer、硬件侧信道、恶意软件包、Prompt Injection、Membership Inference 等多种攻击技术。

**📊 数据集**

使用公开的 LLM 模型（Llama、Qwen 等）和安全基准 ADVBench、PoisonedRAG 等；以及自建的跨堆栈漏洞仓库。

**📈 对比分析**

通过在开源 LLM 上模拟攻击，测量成功率、迭代次数、损失值，展示在 1000 次迭代下 jailbreak 成功率 80%，Rowhammer 侧信道攻击在 94% 的攻击成功率，证明组合攻击显著提升威胁效果。

**⚠️ 局限性**

仅在云容器化部署下评估，缺乏对物理侧信道与高精度计时器的实际攻击实验；攻击成本高、可行性受硬件/环境限制。

---

## 384. MDS-VQA: Model-Informed Data Selection for Video Quality Assessment

**arXiv ID:** 2603.11525 | [PDF](https://arxiv.org/pdf/2603.11525v1)

**作者:** Jian Zou `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 9710 | [OpenAlex ID](https://openalex.org/A5020029652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于模型的主动数据选择框架MDS‑VQA，帮助视频质量评估模型从未标注视频中挑选既难以预测又内容多样的样本进行人工标注；

**💡 创新点**

创新点在于将模型的失败预测与内容多样性联合优化，通过排名学习的失败预测器与基于CLIP的语义多样性度量，实现“难易+多样”双重采样；

**🔧 技术方法**

主要技术包括：1）使用LoRA微调的视觉语言模型（VisualQuality‑R1）作为基础VQA模型；2）通过Rank‑based Thurstone模型训练失败预测器；3）采用Chamfer距离的帧级CLIP特征计算视频多样性；4）贪心组合难度与多样性进行子集选择；5）在选定样本上进行LoRA式主动微调；

**📊 数据集**

实验使用YouTube‑UGC、CGVDS、LIVE‑Livestream、YouTube‑SFV、AIGVQA‑DB等多种公开视频质量数据集；

**📈 对比分析**

与随机、核心集、RD、MC Dropout、贪心采样、ALCS、FreeSel、NoiseStability等对照方法相比，MDS‑VQA在5%标注预算下平均SRCC提升至0.722（从0.651提升），gMAD排名全场第一，显示在平均和最坏情况评估上均有显著优势；

**⚠️ 局限性**

局限性包括：1）失败预测器的难度分数缺乏校准，难以统一调节难度‑多样性权重；2）仅给出单一难度标度，未细化不同失真类型的失败细分；3）多样性基于语义特征，可能忽略感知相似的剪辑；4）当前仅单轮主动学习，未探索多轮迭代与停机准则；5）未结合音频或元数据等多模信息。

---

## 385. Multi-Task Anti-Causal Learning for Reconstructing Urban Events from Residents' Reports

**arXiv ID:** 2603.11546 | [PDF](https://arxiv.org/pdf/2603.11546v1)

**作者:** Liangkai Zhou `[一作]` (Stony Brook University), Shan Lin `[通讯]` (Stony Brook University)

**通讯引用:** 2684 | [OpenAlex ID](https://openalex.org/A5003166096)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了MTAC框架，用以在多任务环境下通过反因果学习从居民报告中重构城市事件数量。

**💡 创新点**

创新点在于将任务共享与任务特定的因果机制通过多任务结构方程模型显式分离，并利用MAP推理联合优化机制变量和原因，突破传统正向预测的局限。

**🔧 技术方法**

使用多任务结构方程模型（SEM）、可学习的邻接矩阵、神经网络实现因果机制以及基于梯度的MAP反向推理算法。

**📊 数据集**

在曼哈顿和纽瓦克的实际数据集上进行实验，涵盖三类事件：违规停车、废弃物业和不卫生情况，共计约400个区域的居民社会经济指标。

**📈 对比分析**

与BSM‑UR、CEVAE、TEDVAE、PLE等基线对比，MTAC在MAE和MSE上均优于所有方法，尤其在数据量较小的任务上提升幅度达34.61%。

**⚠️ 局限性**

局限性包括对高质量混杂变量（SES）数据的依赖、模型结构对任务相似度的要求、以及在不同城市或事件类型外推的可泛化性待进一步验证。

---

## 386. Strict Optimality of Frequency Estimation Under Local Differential Privacy

**arXiv ID:** 2603.11523 | [PDF](https://arxiv.org/pdf/2603.11523v1)

**作者:** Mingen Pan `[一作]` `[通讯]` (Google LLC), Mingen Pan (Google LLC)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究在局部差分隐私（LDP）框架下频率估计的严格最优性，证明存在满足对称极限配置且支持大小固定的最优估计器；

**💡 创新点**

创新点在于给出ℒ_1和ℒ_2损失的严格下界，并提出“最优对称配置”证明其可达最优；同时提出通信成本可压缩到log₂(d(d−1)/2+1)的上界，并给出构造最优估计器的算法；

**🔧 技术方法**

采用极限配置、对称配置、均匀随机置换、最小范数最小二乘、矩阵分析、Hadamard矩阵、BIBD、Carathéodory定理等理论工具；

**📊 数据集**

实验使用Zipf分布合成数据（d=100）和Kosarak点击流数据的1%子集（d=26,000）进行验证；

**📈 对比分析**

与已有方法（Subset Selection、Optimized Count‑Mean Sketch、Weighted Subset Selection）比较，实验结果显示三者均达成理论下界，OCMS在大字典下与最优算法几乎等价，通信成本更低；

**⚠️ 局限性**

局限在于对大字典的预计算成本高（尤其是Weighted Subset Selection的O(d⁶)复杂度），并且对整数支持大小的假设在非整数最优k时需近似处理。

---

## 387. Radiometric fingerprinting of object surfaces using mobile laser scanning and semantic 3D road space models

**arXiv ID:** 2603.11252 | [PDF](https://arxiv.org/pdf/2603.11252v1)

**作者:** Benedikt Schwab `[一作]` (Technical University of Munich), Thomas H. Kolbe `[通讯]` (Technical University of Munich)

**通讯引用:** 5731 | [OpenAlex ID](https://openalex.org/A5035617766)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将四次A2D2自动驾驶车辆的激光扫描数据与Lod3 CityGML道路空间模型关联，提取每个对象表面的辐射指纹。

**💡 创新点**

提出基于光线投射的高效关联方法并公开3DSensorDB数据库，实现了对城市道路空间对象的自动标注和辐射指纹提取，这是首个在完整Lod3 CityGML模型上完成此类工作的工作。

**🔧 技术方法**

采用光线投射关联、t‑SNE降维、统计描述符、分箱策略、SLAM定位、PostgreSQL+PostGIS+pgPointcloud等技术。

**📊 数据集**

使用四次A2D2车辆激光扫描（5个Velodyne VLP‑16）和对应的Lod3 CityGML 3.0道路空间模型（15816对象），以及实验室Spectralon测量数据。

**📈 对比分析**

通过RMSE距离比较指纹相似性，发现相同材质的对象指纹相近；关联成功率约70‑80%，整体对准精度RMSE≈0.19 m，处理3.12亿点、8.7万对象观测，验证了方法的可行性和高效性。

**⚠️ 局限性**

依赖完整且精确的语义模型，未建模的动态或细小物体会导致关联缺失；激光测量误差、湿度、材料不均匀性和缺少级别3校准会影响指纹精度；需要进一步完善模型细分和误差建模。

---

## 388. FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**arXiv ID:** 2603.12146 | [PDF](https://arxiv.org/pdf/2603.12146v1)

**作者:** Quanhao Li `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**通讯引用:** 7913 | [OpenAlex ID](https://openalex.org/A5026167547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

实现了在少步推理中可精准控制视频轨迹的生成，提升了生成速度和质量。

**💡 创新点**

提出三阶段训练框架：先在多步模型上训练轨迹适配器，然后用分布匹配蒸馏得到少步生成器，最后通过融合扩散损失与对抗损失并动态权重调节，使得少步生成既保持视觉质量又实现精确轨迹控制。

**🔧 技术方法**

核心技术包括：扩散模型（DiT/Wan2.2），ControlNet/ResNet轨迹适配器，视频扩散蒸馏（DMD）与对抗蒸馏，Diffusion Discriminator，动态扩散损失缩放策略。

**📊 数据集**

使用 MagicData（23k 带文本与轨迹的高质量视频），FlashBench（长序列轨迹标注），MagicBench 以及 DAVIS 作为评测数据集。

**📈 对比分析**

与多步方法（MagicMotion、Tora、DragAnything 等）以及少步蒸馏基线（DMD、GAN、LCM）比较。FlashMotion 在 FID、FVD、Mask‑IoU、Box‑IoU 上均优于所有基线，并在 121 帧生成上实现约 47× 的速度提升，同时保持或超过多步模型的视觉与轨迹质量。

**⚠️ 局限性**

局限性包括：需要三阶段训练，训练成本仍较高；对极少或非常复杂的轨迹控制效果尚待验证；模型对长序列外的泛化能力尚不充分；目前仅在长序列数据集上进行了评测。

---

## 389. One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**arXiv ID:** 2603.12245 | [PDF](https://arxiv.org/pdf/2603.12245v1)

**作者:** Moayed Haji-Ali `[一作]` (Rice University), Aliaksandr Siarohin `[通讯]` (Snap Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Elastic Latent Interface Transformer (ELIT)，为 Diffusion Transformer (DiT) 引入可变长度的潜在接口和轻量级 Read/Write 交叉注意力层，实现灵活计算分配。

**💡 创新点**

创新点在于通过可变长度潜在序列实现输入图像大小与计算解耦，并在训练时随机丢弃尾部潜在，产生重要性排序的表示，支持推理时动态调整计算预算。

**🔧 技术方法**

使用了 Flow Matching（Rectified Flow）训练框架、轻量级交叉注意力层、分组注意力、随机前缀保留策略和自动/低成本分类器无关引导 (CCFG)。

**📊 数据集**

主要在 ImageNet-1K（256px、512px）和 Kinetics-700（视频）上进行实验，并在 Qwen-Image 大模型上验证。

**📈 对比分析**

与 DiT、U‑ViT、HDiT 基线在相同训练 FLOPs 下对比，ELIT 在 ImageNet-512 上 FID 降幅 35%~40%，在多预算设置下进一步提升 14%~53%，并实现 33% 的推理 FLOPs 降低。

**⚠️ 局限性**

局限性包括对从零开始的大规模模型训练效果未知，且多预算训练依赖随机尾部丢弃，可能在不同任务或细粒度预算下表现不稳定。

---

## 390. Statistical and structural identifiability in representation learning

**arXiv ID:** 2603.11970 | [PDF](https://arxiv.org/pdf/2603.11970v1)

**作者:** Walter Nelson `[一作]` (Institute of Science and Technology Austria), Francesco Locatello `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 3548 | [OpenAlex ID](https://openalex.org/A5073157306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了表示学习模型内部表示的可识别性，提出统计近似可识别性和结构近似可识别性两种概念，并给出了容忍误差ε的通用定义；随后在包含非线性解码器的模型（如MAE、GPT、监督分类器）上证明了内部表示的统计ε‑近似可识别性；再利用ICA和白化技术消除剩余的线性不确定性，实现无监督的表示解缠；并进一步在满足bi‑Lipschitz数据生成假设下推导结构可识别性结果。

**💡 创新点**

创新点包括：①将可识别性细分为统计和结构两层并引入误差容忍度，拓宽了理论适用范围；②证明非线性解码器模型的内部表示在局部bi‑Lipschitz条件下的ε‑近似可识别性；③展示ICA可在无监督条件下几乎完全消除线性/刚性不确定性，为中间层表示的解缠提供理论支撑；④通过结构可识别性将解缠与真实潜在因子关联。

**🔧 技术方法**

核心技术：自监督/监督学习模型（MAE、GPT、ResNet、Swin等）；非线性解码器与局部bi‑Lipschitz假设；ICA（独立成分分析）与白化；动态等距正则化；对比度学习、InfoNCE、MLP头等。

**📊 数据集**

使用的数据集包括：MNIST（验证bi‑Lipschitz与可识别性关系）、公开预训练模型（Pythia、MAE、CheXpert、ResNet）；合成解缠基准（Shapes3D、MPI3D、Falcor3D、Isaac3D）；真实细胞成像基准（Rxrx3‑core cell‑painting）。

**📈 对比分析**

对比方法：与GPT、MAE、ResNet等模型的刚性、线性、ICA对齐误差；与β‑VAE、β‑TCVAE、BioAE等专门解缠模型比较InfoM、InfoE、InfoC指标；在细胞成像中对比Base、PCA、PCA+ICA、PCA+随机旋转的AUROC、稀疏度与浓度。实验结果显示：AE+ICA在合成基准上往往优于专门解缠模型；ICA在预训练模型中可将误差降低多达60%；在细胞成像任务中，PCA+ICA显著提升AUROC并增加稀疏度与浓度。

**⚠️ 局限性**

局限性：bi‑Lipschitz假设在实践中难以直接验证；理论基于无限数据和完美优化，未考虑有限样本、噪声或优化不完美的情况；结构可识别性需满足较强的bi‑Lipschitz与独立非高斯性假设，适用范围受限；缺乏对不同优化算法、正则化策略对可识别性的影响的系统分析。

---

## 391. Effective Resistance Rewiring: A Simple Topological Correction for Over-Squashing

**arXiv ID:** 2603.11944 | [PDF](https://arxiv.org/pdf/2603.11944v1)

**作者:** Bertran Miquel-Oliver `[一作]` (Barcelona Supercomputing Center), Alexis Molina `[通讯]` (Nostrum Biodiscovery)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于有效电阻的增删边重连策略（ERR）来缓解 GNN 的 over-squashing，并对其对表示学习的影响进行深度分析。

**💡 创新点**

创新点在于：①将全局有效电阻作为瓶颈度量；②通过增删边控制结构密度；③结合 PairNorm 在深层模型中平衡 over-squashing 与 oversmoothing；④提供表示层的余弦相似度、线性探针和 CKA 分析。

**🔧 技术方法**

使用了 GCN / DirGCN、有效电阻计算、Ollivier‑Ricci 曲率对比、PairNorm 正则化、线性探针、余弦相似度和 CKA 等技术。

**📊 数据集**

在 homophilic 的 Cora、CiteSeer 和 heterophilic 的 Cornell、Texas（含有向图）四个数据集上进行实验。

**📈 对比分析**

通过在不同深度、不同重连预算下与未重连或曲率重连对照，使用测试准确率、线性探针准确率等指标评估；ERR 在 heterophilic 图和深层模型上可提升准确率，并在保持准确率的同时显著缓解 over-squashing。

**⚠️ 局限性**

主要限制包括：有效电阻计算成本高，尤其是有向图；在深层或 heterophilic 场景中，过度增删边可能加剧 oversmoothing 或类混合；方法对任务特定瓶颈的识别不够精准，需与归一化或其他控制手段配合使用。

---

## 392. Continued Pretraining for Low-Resource Swahili ASR: Achieving State-of-the-Art Performance with Minimal Labeled Data

**arXiv ID:** 2603.11378 | [PDF](https://arxiv.org/pdf/2603.11378v1)

**作者:** Hillary Mutisya `[一作]` (Thiomi-Lugha NLP), John Mugane `[通讯]` (Harvard University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5043592620)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过对 wav2vec2‑bert‑2.0 进行伪标签化的持续预训练（CPT），结合少量标注数据进行监督微调，实现了 Swahili 语音识别的高质量系统。

**💡 创新点**

系统地评估了伪标签化 CPT 在低资源语言中的有效性；提出了可复制的训练流程和明确的数据量要求；并在 Swahili 任务上实现了新的最佳 WER，证明少量标注数据与大量无标注音频的组合可超越传统方法。

**🔧 技术方法**

使用 wav2vec2‑bert‑2.0 作为基础模型；伪标签生成采用 CTC 贪心解码；持续预训练采用带有 CTC 损失的监督学习；监督微调采用学习率调度、标签平滑、梯度裁剪等技术；使用 AdamW 优化器和早停策略。

**📊 数据集**

标注数据：Mozilla Common Voice Swahili 16.0（5K、20K、50K 语料）；无标注数据：多域 Swahili 语音（来自公共音频资源，未具体列明）。

**📈 对比分析**

对比方法：在 50K 标注样本上直接微调 wav2vec2‑bert‑2.0 作为基线，获得 17.71% WER；随后在 5K/20K 标注样本上使用 CPT + 微调，得到 10.89% 与 3.24% WER，分别比基线降低 38.5% 与 81.7%；此外 3.24% WER 还比现有最优学术系统（XLS‑R 8.3%）提升 61%。

**⚠️ 局限性**

需至少 20K 标注样本来训练伪标签模型；伪标签质量对 CPT 效果敏感；实验仅覆盖 Swahili，未验证跨语言通用性；对无标注语料的可获取性和域匹配存在依赖。

---

## 393. Algorithmic Capture, Computational Complexity, and Inductive Bias of Infinite Transformers

**arXiv ID:** 2603.11161 | [PDF](https://arxiv.org/pdf/2603.11161v1)

**作者:** Orit Davidovich `[一作]` (IBM), Zohar Ringel `[通讯]` (Hebrew University)

**通讯引用:** 2141 | [OpenAlex ID](https://openalex.org/A5053390682)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了算法捕获（Algorithmic Capture）的正式定义，研究无限宽Transformer在松懈（lazy）与丰富（rich）两种极限下的推理时间复杂度，并通过理论推导得到该类模型只能学习复杂度不超过O(T³+ε)（最优可到O(T²+ε)）的算法；随后在合成任务（诱导头、排序、最短路径与最小割）上进行实验，验证其低复杂度偏置。

**💡 创新点**

创新点包括：① 将算法学习问题形式化为算法捕获，并与分布式任务的有效多项式时间启发式方案（EPTHS）关联；② 通过NTK/NNGP无限宽分析和Monte Carlo估计，首次给出Transformer推理复杂度的上界；③ 揭示Transformer在无限宽极限下对低复杂度算法具有天然偏置，解释了为什么它们能捕获诱导头、排序等任务，却难以学习更高复杂度的最短路径与最小割。

**🔧 技术方法**

使用的技术主要有：无限宽Transformer的NTK/NNGP理论、注意力层的协方差递归分析、Monte Carlo估计核函数、特征学习与稀疏注意力的复杂度上界、扰动理论估计以及前向传播与核评估的时间复杂度比较。

**📊 数据集**

数据集为合成生成的数据：随机几何图（RGG）生成的图实例，用于最短路径与最小割；随机序列生成的诱导头和排序任务，包含大规模整数或随机标记序列。

**📈 对比分析**

与普通有限深度Transformer进行对比：在低复杂度任务（诱导头、排序）上，模型在少量样本后即可实现O(log T)微调；在高复杂度任务（最短路径、最小割）上，模型表现为超线性增长，无法实现算法捕获。实验结果与理论推导的推理复杂度上界一致，表明低复杂度偏置确实限制了Transformer的算法学习能力。

**⚠️ 局限性**

局限性包括：推理复杂度上界相对保守，可能低估Transformer在特殊结构或更深层架构下的实际表现；实验仅基于合成任务，缺乏真实世界问题验证；未深入探讨循环或递归网络的情况；对超参数如宽度与层数的具体取值影响仍需进一步研究。

---

## 394. PCA-Enhanced Probabilistic U-Net for Effective Ambiguous Medical Image Segmentation

**arXiv ID:** 2603.11550 | [PDF](https://arxiv.org/pdf/2603.11550v1)

**作者:** Xiangyu Li `[一作]` (Harbin Institute of Technology), Gongning Luo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9850 | [OpenAlex ID](https://openalex.org/A5049558861)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于PCA增强的Probabilistic U-Net（PEP U-Net），通过在后验网络中使用PCA降维和逆PCA重构，改进了医学图像的模糊分割。

**💡 创新点**

创新点在于将PCA降维作为后验网络的瓶颈，减少高维冗余并保持低维KL计算；同时加入逆PCA重构，使得采样后重投到原空间，提升表达能力。

**🔧 技术方法**

采用U-Net骨干、条件变分自编码器框架、PCA降维与逆映射、1×1卷积适配器以及NLL、Brier、ECE等评估指标。

**📊 数据集**

使用LIDC肺结节CT数据集和PhC-U373细胞相位对比显微镜图像数据集。

**📈 对比分析**

与U-Net、Probabilistic U-Net、SSN和cSSN等基线进行对比，PEP U-Net在LIDC上IoU最高0.434、GED最低0.120，在PhC上IoU 0.890、GED 0.008，且在NLL、Brier和ECE等不确定性指标上均优于对手。

**⚠️ 局限性**

限制在于PCA保留维数需手动调优，且对不同数据集的最优k值不同，模型对极端噪声或极少样本情况的鲁棒性未作深入评估。

---

## 395. LLMs Can Infer Political Alignment from Online Conversations

**arXiv ID:** 2603.11253 | [PDF](https://arxiv.org/pdf/2603.11253v1)

**作者:** Byunghwee Lee `[一作]` (University of Virginia), Jisun An `[通讯]` (Indiana University)

**通讯引用:** 3748 | [OpenAlex ID](https://openalex.org/A5084955495)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型（LLM）对在线论坛（Debate.org 与 Reddit）的用户帖子进行文本分析，预测用户的政治倾向（共和党或民主党）。

**💡 创新点**

证明LLM能够从非政治性、日常性对话中识别政治信仰，并通过模型自报的置信度进行加权聚合，显著提升预测准确率；同时揭示语义相似度和用户重叠度与推断表现之间的系统性关系，提供词级别政治信号的可解释性。

**🔧 技术方法**

使用 GPT‑4o 与开源 Llama‑3.1‑8B 两种LLM，采用零样本或少量示例提示，获取结构化 JSON 预测结果（标签 + 置信度）；对用户文本进行聚合（多数投票、置信度加权、最大置信度投票）；计算主题与“Politics”主题的语义相似度（Sentence‑BERT 嵌入余弦相似度）和用户重叠度（Jaccard / NPMI）。

**📊 数据集**

Debate.org (DDO) 数据集：3511 自我标注为共和党/民主党的辩论者，共 22,265 条论点，23 个主题；Reddit 数据集：2000 名从 r/Conservative / r/democrats 采样的用户，约 196,000 条评论，使用 GPT‑4o 将 6,000+ 子版块划分为相同的 23 个主题。

**📈 对比分析**

与传统监督学习模型（如 SVM、XGBoost 等）对比，LLM 在文本层面 F1 分别为 GPT‑4o 0.647 / 0.624、Llama‑3.1‑8B 0.619 / 0.534；在用户层面通过最大置信度聚合后，F1 可达 0.799（DDO）或 0.829（Reddit）等，显著优于传统模型，且在所有类别和平台上表现稳定；置信度加权和最大置信度聚合是最佳策略。

**⚠️ 局限性**

1) 样本来源有限，主要为自愿标注或极具政治倾向的子版块，可能不代表更广泛人群；2) LLM 的推断可能受训练语料隐含偏见影响，难以区分模型自学还是外部训练；3) 仅在英语、美国语境下验证，跨文化、跨语言推广需进一步研究；4) Reddit 的政治标签为间接代理，存在标签误差；5) 仅考察两种 LLM，其他模型或版本的性能尚未知晓。

---

## 396. Security-by-Design for LLM-Based Code Generation: Leveraging Internal Representations for Concept-Driven Steering Mechanisms

**arXiv ID:** 2603.11212 | [PDF](https://arxiv.org/pdf/2603.11212v1)

**作者:** Maximilian Wendlinger `[一作]` (Fraunhofer Institute for Applied and Integrated Security), Philip Sperl `[通讯]` (Fraunhofer Institute for Applied and Integrated Security)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5011742469)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 CodeLLM 的内部残差流进行分析，提取代码安全概念向量，并在推理时通过向量注入实现安全代码生成，提升功能正确性与安全性。

**💡 创新点**

创新点在于首次利用对比度数据集提取可解释的安全概念子空间，并以无额外微调、低计算开销的方式在多语言、多基准上实现安全代码生成的 SOTA 结果。

**🔧 技术方法**

采用对比度数据集构造、残差流激活提取、差分平均向量（mean‑difference）概念表示、线性注入（steering）以及一系列安全与功能性评估指标（如 secure‑pass@k、SVEN‑SR）等技术。

**📊 数据集**

使用的主要数据集包括 CyberNative 的安全/不安全代码对、CodeGuard+、CWEval、HumanEval‑X、SVEN 等；概念提取采用约 50 条对样本，评估覆盖 Python、C/C++、Java 等语言。

**📈 对比分析**

通过与 SafeCoder、Constrained Decoding、Secure Prefix 等现有基准在 CodeGuard+ 与 CWEval 上进行对比，SCS‑Code 在安全率、sec‑pass@1 以及功能正确性方面均提升 5–10% 以上，混合方法更能实现 1.8% 的 sec‑pass@1 提升，整体达到 SOTA。

**⚠️ 局限性**

限制在于需人工构造对比度样本，概念提取仍受限于线性假设，某些情况下功能正确性与安全性存在权衡，且对模型内部机制的解释仍不够完整，可能存在噪声与泛化局限。

---

## 397. Incremental Neural Network Verification via Learned Conflicts

**arXiv ID:** 2603.12232 | [PDF](https://arxiv.org/pdf/2603.12232v1)

**作者:** Raya Elsaleh `[一作]` (Hebrew University of Jerusalem), Guy Katz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2498 | [OpenAlex ID](https://openalex.org/A5102986148)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种增量式的神经网络验证方法，能够在一系列相关验证查询中重用已学习的冲突（冲突子句），从而加速分支定界搜索。

**💡 创新点**

创新点在于：①定义了查询的细化关系并证明冲突在细化下仍有效；②将 SAT 求解器与分支定界结合，实现冲突的一致性检查与单元传播；③构建了通用的增量冲突管理器（ICA），可无缝嵌入任何基于分支定界的验证器。

**🔧 技术方法**

使用的技术包括：分支定界（Branch‑and‑Bound）搜索、ReLU 相位的冲突子句记录、CaDiCaL SAT 求解器进行冲突一致性检查与单元传播，以及在 Marabou 验证器中实现的增量冲突管理器。

**📊 数据集**

实验所用数据集：MNIST（用于局部鲁棒性半径计算）、GTSRB（用于最小足够特征集提取）、以及基于航天器控制的 Lyapunov 神经网络（用于输入分割验证）。

**📈 对比分析**

对比方法：与非增量化的 Marabou 基线进行实验。结果显示：局部鲁棒性任务中平均加速 1.35×，输入分割任务中加速 1.92×，最小特征集任务中虽然最终解释尺寸相近，但增量方法在任意时刻能更快逼近最优解释。整体最快可达 1.9× 的加速。

**⚠️ 局限性**

局限性：①冲突子句未做最小化，可能包含冗余决策导致 SAT 推理开销；②只重用冲突，未利用其他理论子句或抽象信息；③对非细化关系的查询不适用，增量效果受细化链长度限制；④实现仍依赖 Marabou 与 CaDiCaL，跨平台适配需进一步验证。

---

## 398. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

**arXiv ID:** 2603.12228 | [PDF](https://arxiv.org/pdf/2603.12228v1)

**作者:** Yulu Gan `[一作]` (Massachusetts Institute of Technology), Phillip Isola `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 53442 | [OpenAlex ID](https://openalex.org/A5017456911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了预训练模型在局部参数空间的“thicket”结构，并提出一种完全并行的随机扰动加投票集成方法来实现后训练。

**💡 创新点**

将预训练视为参数分布，揭示大模型邻域密集多样的专家解；发现“thicket”现象；提出仅靠随机扰动和集成即可与梯度/进化等方法竞争的 RandOpt。

**🔧 技术方法**

采用高斯随机扰动生成参数样本、在小验证集评估选择 top‑K、推理时多模型投票集成，并可进一步蒸馏为单模型。

**📊 数据集**

多任务数据集包括数学推理（Countdown、GSM8K、MATH‑500、OlympiadBench）、编程（MBPP）、创意写作（ROCStories）、化学（USPTO）、视觉语言（GQA）以及自定义 1D 信号实验。

**📈 对比分析**

与 PPO、GRPO、ES、TT‑MV 等标准后训练基线在相同 FLOP 下对比，RandOpt 在 0.5B–8B 大小模型上多任务平均提升 5–15%，在 Countdown 3B 仅 3.2 分即可达到 70%，在 GQA 3B 提升 12.4%；集成显著优于单模型，蒸馏后可保持大部分性能。

**⚠️ 局限性**

仅适用于已充分预训练的大模型；小模型或从零训练几乎无效；推理时需多次前向传播，成本高；目前仅支持离散答案投票，结构化输出需改进；未深入解释为何预训练产生 thicket。

---

## 399. Credibility Matters: Motivations, Characteristics, and Influence Mechanisms of Crypto Key Opinion Leaders

**arXiv ID:** 2603.12000 | [PDF](https://arxiv.org/pdf/2603.12000v1)

**作者:** Alexander Kropiunig `[一作]` (Complexity Science Hub), Bernhard Haslhofer `[通讯]` (Complexity Science Hub)

**通讯引用:** 2227 | [OpenAlex ID](https://openalex.org/A5078932059)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过对13位加密货币KOL进行访谈，结合自我决定理论，对其动机、特征及信誉机制进行质性主题分析。

**💡 创新点**

首次将自我决定理论应用于高风险金融场景，提出四维信誉模型（自我调节、边界认知、问责与反思）并引入LLM辅助的混合编码方法。

**🔧 技术方法**

使用OpenAI GPT‑4进行主题候选生成的LLM辅助主题分析，配合人工双编码与Krippendorffα可靠性检验。

**📊 数据集**

基于13位KOL的访谈文本（约12万词）进行手工转录和匿名化。

**📈 对比分析**

与传统纯人工编码对比，LLM辅助后主题覆盖率提升，最终可靠性为α=0.78；未做机器学习模型性能评估。

**⚠️ 局限性**

样本规模小、地域与性别单一、依赖自述数据、LLM潜在偏差、受访者可能自我呈现、研究结果易随市场波动而变化。

---

## 400. CoViLLM: An Adaptive Human-Robot Collaborative Assembly Framework Using Large Language Models for Manufacturing

**arXiv ID:** 2603.11461 | [PDF](https://arxiv.org/pdf/2603.11461v1)

**作者:** Jiabao Zhao `[一作]`, Ilya Kovalenko `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CoViLLM，一个集成深度相机定位、人机交互分类和大型语言模型推理的自适应人机协作装配框架，可动态生成已知、定制和新产品的装配序列。

**💡 创新点**

创新点在于将深度感知与人类即时反馈相结合，并利用 LLM 进行自然语言任务规划，突破传统预设序列的限制，实现对未知部件和动态装配顺序的实时识别与规划。

**🔧 技术方法**

使用 Intel RealSense D435 深度相机进行几何定位、语音识别/合成（ElevenLabs）进行人机交互、UFactory xArm 6DOF 机器人及抓手执行动作、LLM（未指明）进行推理规划、像素到机器人坐标转换及运动规划。

**📊 数据集**

采用 NIST Assembly Task Board 1 (NATB1) 标准装配任务板进行实验验证。

**📈 对比分析**

通过三类案例（已知、定制、新产品）进行对比实验，框架在所有情形下实现 100% 的任务规划准确率；定位性能受相机高度影响，实验确定有效高度范围，整体成功率高。

**⚠️ 局限性**

局限性包括：仅在实验台面上验证，未测试更复杂产品和更长装配序列；对未知部件仍需人工分类，可能导致操作负担；深度相机噪声和相机摆放导致定位误差；LLM 仍需 prompt‑engineering 以抑制幻觉；尚未在真实生产环境中评估可扩展性与鲁棒性。

---

## 401. PRMB: Benchmarking Reward Models in Long-Horizon CBT-based Counseling Dialogue

**arXiv ID:** 2603.11494 | [PDF](https://arxiv.org/pdf/2603.11494v1)

**作者:** Yougen Zhou `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 8086 | [OpenAlex ID](https://openalex.org/A5062604912)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了PRMB基准，用以评估长时序、多会话CBT心理咨询对话中的奖励模型表现，并系统分析其在多种奖励模型上的效果。

**💡 创新点**

创新点在于：①针对长时序、过程导向的CBT对话设计了进展性总结提示与多阶段评估框架；②创建了包含6个会话、21类负面情境的pairwise与Best‑of‑N评价集，填补了现有短时评测的空白；③通过与下游BoN生成质量的相关性验证基准的预测能力。

**🔧 技术方法**

使用了：进展性总结（short‑term / long‑term）提示策略、10个大语言模型生成候选、pairwise与Best‑of‑N评价、discriminative与generative（LLM‑as‑judge）奖励模型评估、以及RAG、few‑shot、Self‑Refine、CoT等推理‑时间策略的实验。

**📊 数据集**

数据集来源为118个公开CBT案例（每案6会话），共生成约13k提示，构建约6.9k pairwise与4‑best‑of‑N实例，负面情境覆盖21类，全部公开可复现。

**📈 对比分析**

评测方法为对比多种discriminative与generative奖励模型的pairwise与BoN准确率；PRMB显示pairwise 65‑86%，BoN 39‑73%，整体平均约70%；与RewardBench2比较，PRMB在下游BoN生成（BERTScore）上的Spearman相关系数为0.70，显著高于RewardBench2的0.63，证明其更好地预测下游性能。

**⚠️ 局限性**

局限性包括：奖励模型在长时序和细微负面情境下性能明显下降；生成式评估对多样化候选缺乏稳健性；多数推理‑时间策略（few‑shot、Self‑Refine、CoT）对性能提升有限；基准仅评估相对好坏，未覆盖真实临床有效性与安全性。

---

## 402. Automatic Generation of High-Performance RL Environments

**arXiv ID:** 2603.12145 | [PDF](https://arxiv.org/pdf/2603.12145v1)

**作者:** Seth Karten `[一作]` (Princeton University), Chi Jin `[通讯]` (Princeton University)

**通讯引用:** 4806 | [OpenAlex ID](https://openalex.org/A5101961985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

使用大语言模型将现有RL环境代码转换为高性能实现，并通过分层验证确保语义等价。

**💡 创新点**

提出可复用的翻译模板、层级验证流程和agent辅助迭代修复，证明成本低 (<$10) 的高性能环境可在多领域实现。

**🔧 技术方法**

利用大语言模型（Gemini 3 Flash Preview）、JAX、Rust、PyO3、GPU并行、XLA等技术。

**📊 数据集**

对五个环境进行实验：EmuRust、PokeJAX、HalfCheetah、Puffer Pong、TCGJax（包含公开与私有参考）。

**📈 对比分析**

通过与原始实现、手工优化实现（如BraX、MJX）以及新实现对比，使用SPS、PPO速度、policy transfer等指标，展示了1.5×至42×加速与吞吐量等价。

**⚠️ 局限性**

方法主要适用于确定性、可模块化、固定状态尺寸的环境，对非确定性外部依赖或动态内存分配的环境可能需要额外工程。

---

## 403. DVD: Deterministic Video Depth Estimation with Generative Priors

**arXiv ID:** 2603.12250 | [PDF](https://arxiv.org/pdf/2603.12250v1)

**作者:** Hongfei Zhang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2651 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个利用预训练视频扩散模型进行单通量确定性深度估计的框架；

**💡 创新点**

创新点包括：将扩散时间步改为结构锚点以平衡低频稳定与高频细节；引入潜在空间微分正则化（Latent Manifold Rectification）消除平均坍塌，恢复边缘与运动一致性；利用全局仿射一致性实现长视频无缝推理；

**🔧 技术方法**

使用技术包括：预训练视频扩散Transformer (DiT) + VAE潜在空间；LoRA微调；时间步结构锚点；潜在微分正则 (梯度与流差分约束)；全局仿射对齐；联合图像-视频训练；

**📊 数据集**

训练数据集：TartanAir、Virtual KITTI、Hypersim 等公开合成视频/图像；评估数据集：KITTI、ScanNet、Bonn、Sintel、NYUv2、DIODE 等真实场景；

**📈 对比分析**

与多步采样生成方法 DepthCrafter、ChronoDepth 等以及判别方法 Video Depth Anything (VDA) 等进行零样本或极少量标注下的对比；在绝对相对误差（AbsRel）和阈值精度（δ1）上均优于 DepthCrafter，接近或超过 VDA，且推理速度与 VDA 相近；在长视频尺度稳定性上表现最佳；

**⚠️ 局限性**

局限性：对极端遮挡或纹理稀缺区域仍易产生几何误差；全局仿射对齐假设潜在空间保持线性，极端视角或光照变化时效果需进一步验证；模型依赖大规模预训练视频扩散模型，显存占用相对较高；

---

## 404. On the Computational Hardness of Transformers

**arXiv ID:** 2603.11332 | [PDF](https://arxiv.org/pdf/2603.11332v1)

**作者:** Barna Saha `[一作]` (University of California San Diego), Hantao Yu `[通讯]` (Columbia University)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5073222947)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文证明了多层多头 Transformer 的计算不可能比逐头独立计算更快，给出了小嵌入维度下的 LHN²‑o(1) 以及大嵌入维度下的 LHN^ω‑o(1) 的下界；

**💡 创新点**

创新点在于将直和（direct‑sum）理论与 Transformer 结合，利用 3‑OV 及 SETH 假设推导出小嵌入下的 LHN² 下界，并首次将 Baur‑Strassen 定理推广到扩展算术电路，以证明大嵌入下的 LHN^ω 下界；

**🔧 技术方法**

主要技术包括：基于 3‑OV 的细粒度复杂度归约、Hardmax 与 Softmax 的等价近似、扩展算术电路模型、Baur‑Strassen 定理的扩展版本、矩阵乘法张量的非对称求和不等式；

**📊 数据集**

本文为理论工作，未使用任何实验数据集；

**📈 对比分析**

由于是理论下界研究，未与实测算法直接比较；但结论表明即使使用最快的矩阵乘法算法，Transformer 的时间复杂度也无法低于 LHN²（小嵌入）或 LHN^ω（大嵌入），这与现有的最优实现相匹配；

**⚠️ 局限性**

局限性：仅在扩展算术电路模型下给出下界，未覆盖 Word‑RAM 或具体硬件实现；仅给出最坏情况下的上界；对于中等嵌入维度的更细粒度下界仍未完全解决；

---

## 405. Security Considerations for Artificial Intelligence Agents

**arXiv ID:** 2603.12230 | [PDF](https://arxiv.org/pdf/2603.12230v1)

**作者:** Ninghui Li `[一作]` (Purdue University), Jerry Ma `[通讯]` (Perplexity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并评估前沿 AI 代理系统的安全威胁，归纳攻击面，提出多层防御框架，并针对单体与多体代理系统给出针对性改进建议。

**💡 创新点**

创新点在于：① 将传统安全原则（最小权限、完整调度、可验证性）与 LLM 代理特性相结合，形成“防御深度+确定性执行”架构；② 对“间接提示注入”进行系统化防御，提出输入级检测、模型层级指令层次、系统级沙箱与确定性执行等三层措施；③ 明确多体代理的“混淆代办”与权限升级风险，强调跨代理授权与委派的安全模型。

**🔧 技术方法**

使用的技术主要包括：LLM 指令层次学习（如多嵌入方式）、输入级注入检测（基于 perplexity、注意力模式等）、模型层级控制（指令优先级嵌入）、系统级沙箱化（如 CaMeL、AgentSandbox）、确定性执行策略（工具白名单、速率限制、模式校验）以及安全审计与可追溯性机制。

**📊 数据集**

本文未进行实验验证，所提方法主要基于已有研究与实务案例；若要验证，可使用公开 LLM (如 GPT‑4, Llama‑2) 与代理框架 (如 OpenAI Agents, Perplexity Computer) 及自构造的提示注入与多体攻击脚本进行评估。

**📈 对比分析**

方法对比主要是与传统软件安全措施的适配性讨论，没有定量性能数据。作者指出多层防御能够显著降低攻击成功率，但由于 LLM 非确定性，单层防御难以单独提供完整保障；在实际部署中需权衡检测准确率、响应延迟与成本。

**⚠️ 局限性**

局限性包括：① 现有检测与控制技术仍未达到对所有注入攻击的高召回与低误报；② 多体代理系统中的授权与责任追溯仍缺乏统一模型；③ 缺乏动态、适应性评估基准与可测量指标；④ 过度依赖人机确认可能导致用户疲劳，降低实用性。

---

## 406. The price of decentralization in managing engineering systems through multi-agent reinforcement learning

**arXiv ID:** 2603.11884 | [PDF](https://arxiv.org/pdf/2603.11884v1)

**作者:** Prateek Bhustali `[一作]` (Delft University of Technology), Charalampos P. Andriotis `[通讯]` (Delft University of Technology)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5023492586)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了可求解至近最优解的 k‑out‑of‑n 维护规划基准环境，并对九种常用多智能体深度强化学习（MADRL）算法在这些环境中的表现进行系统评估。

**💡 创新点**

创新点：①引入可变冗余的 k‑out‑of‑n 环境，首次量化“去中心化代价”对最优性的影响；②提供近最优（SARSOP）基线和精细化的基准算法，对比离散策略与连续策略的效果；③结合价值分解与 actor‑critic 等多范式，揭示冗余导致的协调难题与算法局限。

**🔧 技术方法**

技术：多智能体深度强化学习（CTCE、CTDE、DTDE），包括 JAC、DDQN、DCMAC、IACC‑PS、MAPPO‑PS、VDN‑PS、QMIX‑PS、IAC‑PS、IPPO‑PS；离线点基准求解器 SARSOP ；奖励、成本与观测模型的合成；贝叶斯更新与 belief‑space 可视化。

**📊 数据集**

数据集：合成的 k‑out‑of‑4 系统（n=4，k=1…4）以及其子集（n=2,3）——每个组件具备三种状态、三种动作、特定转移、成本与观测矩阵；此外使用 Climb Game（单步矩阵游戏）作对照。

**📈 对比分析**

比较方法：在固定训练预算下，使用 Monte Carlo roll‑outs 评估每种算法的折扣回报；与 SARSOP 的近最优回报及手工优化的启发式基线对比。结果显示：①在串联/接近串联配置中，去中心化算法可达 95%+ 最优；②冗余增加时（k 降低），回报显著下降，最差时仅为 60%–70% 最优；③价值分解算法在并行配置中严重失效；③CTCE 算法性能最佳但不具可扩展性。

**⚠️ 局限性**

局限性：①去中心化策略在高冗余系统中难以捕获全局协同，导致最优性损失；②实验仅覆盖小规模（n≤4）基准，无法直接推广至真实大规模基础设施；③缺乏对学习策略泛化与鲁棒性的理论分析；④某些算法（如 IAC‑PS）受离线经验重放的限制，学习不稳定。

---

## 407. Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**arXiv ID:** 2603.11818 | [PDF](https://arxiv.org/pdf/2603.11818v1)

**作者:** Md. Hasin Sarwar Ifty `[一作]` (BRAC University), Md. Saiful Islam `[通讯]` (BRAC University)

**通讯引用:** 2177 | [OpenAlex ID](https://openalex.org/A5004369172)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研发了一套基于卷积神经网络和可解释人工智能的卵巢癌自动检测系统。

**💡 创新点**

创新点在于构建并比较15种CNN变体，选择最优模型后结合LIME、Integrated Gradients和SHAP进行可解释性分析，并在有限样本上通过图像增强提升性能。

**🔧 技术方法**

技术包括TensorFlow深度学习框架、Albumentations图像增强、LeNet、ResNet、VGG、Inception四大CNN架构及其变体、以及XAI方法LIME、Integrated Gradients和SHAP。

**📊 数据集**

使用Mendeley公开的OvarianCancer&SubtypesDatasetHistopathology数据集，包含5类肿瘤共约500张原始图像，经过增强后达到2490张。

**📈 对比分析**

性能通过准确率、精确率、召回率、F1、ROC曲线及AUC等指标评估；Inception V3‑A在增强集上平均达到94%各项指标，AUC最高，优于其他14种模型。

**⚠️ 局限性**

局限性在于数据规模有限、仅使用影像数据且未覆盖临床非侵入性信息；模型在原始数据上表现不佳，且缺乏在真实临床环境中的泛化验证。

---

## 408. AI Psychometrics: Evaluating the Psychological Reasoning of Large Language Models with Psychometric Validities

**arXiv ID:** 2603.11279 | [PDF](https://arxiv.org/pdf/2603.11279v1)

**作者:** Yibai Li `[一作]` (University of Scranton), Xiaobing Li `[通讯]` (University of Scranton)

**通讯引用:** 3410 | [OpenAlex ID](https://openalex.org/A5100375524)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四个大型语言模型（GPT‑3.5、GPT‑4、LLaMA‑2、LLaMA‑3）应用 AI 心理测量学，利用技术接受模型（TAM）评估其心理推理能力的收敛、辨别、预测和外部效度。

**💡 创新点**

首次系统性地将心理测量框架与大语言模型结合，并引入扩散采样法以提升回答多样性，展示不同模型在心理计量有效性上的差异。

**🔧 技术方法**

使用了 TAM 结构方程模型、PLS‑SEM、Cronbachα、AVE、Fornell–Larcker 判别效度检验、R² 预测效度等统计技术。

**📊 数据集**

采集了通过 OpenRouter API 对四个 LLM 进行 500 次扩散式提问得到的数据，并收集来自 Amazon Mechanical Turk 的 248 名人类受试者的问卷数据。

**📈 对比分析**

通过对比各模型与人类在负载、内部一致性、AVE、R² 等指标上的表现，发现 GPT‑4 与 LLaMA‑3 的心理计量效度显著高于前代模型，所有模型均满足辨别、预测、外部效度，GPT‑4 的预测效度接近人类水平。

**⚠️ 局限性**

LLaMA‑2 在收敛效度和内部一致性方面表现不足；扩散采样方法仍受提示设计影响，且研究仅关注 TAM 维度，未覆盖情绪智力等其他心理构念。

---

## 409. On the Possible Detectability of Image-in-Image Steganography

**arXiv ID:** 2603.11876 | [PDF](https://arxiv.org/pdf/2603.11876v1)

**作者:** Antoine Mallet `[一作]` (University of Lille), Patrick Bas `[通讯]` (University of Lille)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了基于可逆神经网络的图像嵌图像隐写的可检测性，提出了基于离散小波变换+PCA+ICA的统计特征（前四阶矩）进行Stego/Cover分类的简洁可解释方法；

**💡 创新点**

创新点在于将ICA与PCA相结合，利用离散小波系数的主成分分离源信号，再用四阶统计量构造判别特征，显著提升了对高容量隐写图像的检测性能；

**🔧 技术方法**

采用的技术包括离散小波变换（DWT）、主成分分析（PCA）、快速ICA（FastICA）以及基于高斯核的支持向量机（SVM）分类器；

**📊 数据集**

使用COCO数据集中的512×512彩色图像作为Cover样本，生成2500张Stego图像进行实验；

**📈 对比分析**

与传统的SRM+SVM方法对比，所提方法在INN隐写方案（如HiNet、PRIS、DeepMIH）上可达84.6%最高准确率；而SRM+SVM在所有方案中均能超过99%准确率；

**⚠️ 局限性**

局限性包括：对非INN隐写方案检测效果不佳；方法仅基于单张Stego图像，未考虑多模型或加密钥匙的情况；并且未评估对鲁棒性或实时性的影响。

---

## 410. High-resolution weather-guided surrogate modeling for data-efficient cross-location building energy prediction

**arXiv ID:** 2603.11121 | [PDF](https://arxiv.org/pdf/2603.11121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 411. An Evolutionary Algorithm with Probabilistic Annealing for Large-scale Sparse Multi-objective Optimization

**arXiv ID:** 2603.11874 | [PDF](https://arxiv.org/pdf/2603.11874v1)

**作者:** Shuai Shao `[一作]` (Anhui University), Jin Li `[通讯]` (Sapient Intelligence)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于概率退火的进化算法（PAMEA），专门用于解决大规模稀疏多目标优化问题。

**💡 创新点**

创新点在于：①双熵概率向量方案，结合低熵的收敛导向向量和高熵的退火导向向量；②退火驱动的变量聚类方法；③多种群协同搜索机制，实现全局探索与局部利用的动态平衡。

**🔧 技术方法**

使用技术包括：进化算法、概率退火、双熵概率向量、变量聚类、SPEA2环境选择、模拟二叉交叉与多项式变异等。

**📊 数据集**

数据集涵盖：Benchmark 的 SMOP1–SMOP8（D=100–5000），以及真实问题 SR1–SR5（稀疏信号重构）和 PM1–PM5（模式挖掘）。

**📈 对比分析**

通过与 TS‑SparseEA、SCEA、S‑NSGA‑II、MOEA/CKF、DKCA 在 30 次独立实验中的 IGD（基准）和 HV（真实）指标比较，PAMEA 在大多数实例上获得显著更低的 IGD、更高的 HV，收敛速度也更快。

**⚠️ 局限性**

局限性包括：在百万维度的大规模稀疏空间中学习概率向量仍然困难；目前未充分利用变量间的交互信息；退火速率固定，缺乏自适应调节；对非稀疏问题的适用性尚未验证。

---

## 412. Human-Centred LLM Privacy Audits: Findings and Frictions

**arXiv ID:** 2603.12094 | [PDF](https://arxiv.org/pdf/2603.12094v1)

**作者:** Dimitri Staufer `[一作]` (Technische Universität Berlin), Bettina Berendt `[通讯]` (Technische Universität Berlin)

**通讯引用:** 5755 | [OpenAlex ID](https://openalex.org/A5072130052)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建LMP2浏览器工具，对大型语言模型的姓名相关关联进行自我审计，完成了对八款模型的评估以及458名用户的问卷与实验；

**💡 创新点**

创新点在于将canary探测方法适配到黑盒API并通过短句恢复、同义改写聚合生成可解释的关联强度与置信度，同时系统性梳理了九大人本隐私审计障碍；

**🔧 技术方法**

采用了短句恢复的fragment completion、同义改写聚合、NLL排名、置信度评估等技术，并在前端实现了交互式结果卡片；

**📊 数据集**

使用了50个来自WikiMem/Wikidata的人类属性集、100名公开人物与100个合成名字作为测试对象，以及欧盟受访者自填的真实姓名与特征；

**📈 对比分析**

对八个模型（3个开源、5个API）进行比较，发现GPT‑4o在11/50属性上达≥60%准确率，且知名人物与非人名在置信度上显著区分；

**⚠️ 局限性**

局限性包括输出无法区分记忆、推理与先验来源，姓名歧义、多值属性与时间漂移问题，语言与脚本偏向，部署系统导致证据不稳定，难以直接用于责任追究。

---

## 413. Highly Autonomous Cyber-Capable Agents: Anticipating Capabilities, Tactics, and Strategic Implications

**arXiv ID:** 2603.11528 | [PDF](https://arxiv.org/pdf/2603.11528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 414. Systematic Scaling Analysis of Jailbreak Attacks in Large Language Models

**arXiv ID:** 2603.11149 | [PDF](https://arxiv.org/pdf/2603.11149v1)

**作者:** Xiangwen Wang `[一作]` (University of Illinois Urbana Champaign), Varun Chandrasekaran `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将越狱攻击视为受计算约束的优化过程，提出了以FLOPs为统一尺度的扩展曲线，并在多种模型、规模和目标类别下系统评估了四类主要攻击范式的计算效率与成功率。

**💡 创新点**

创新点包括：①首次对越狱攻击进行计算规模化建模，并给出通用FLOPs基准；②量化并比较了prompt‑based、optimization、sampling及 evolutionary 四类攻击在计算效率、成功上限与隐蔽性上的显著差异；③通过同状态对比揭示prompt‑based攻击在prompt空间搜索中的更高质量；④探究了目标类别对扩展行为的影响。

**🔧 技术方法**

技术上使用了：饱和指数曲线拟合（ASR vs FLOPs）；GPT‑5评判器生成红队分数与相关性分数；GPT‑2 perplexity评估隐蔽性；同状态协议比较PAIR与GCG；FLOPs成本模型计算计算量；以及四种攻击实现（GCG, PAIR, BoN, AutoDAN）。

**📊 数据集**

数据集为200个混合目标，来源于AdvBench‑Harmful Behavior、HarmBench、ClearHarm，并对每个目标按四类（可操作性、恶意制品、误导信息、仇恨/骚扰）进行规则式标注。

**📈 对比分析**

在共享FLOPs轴上通过饱和指数曲线参数（起始点a、上限a+b、收敛速率c）对ASR与隐蔽性进行定量比较。结果显示PAIR在计算效率、ASR上限和隐蔽性方面均优于其他三类；BoN在相关性上略胜于PAIR；GCG收敛慢且上限最低；不同模型家族/尺寸主要影响收敛速率；信息化目标相对更易越狱。

**⚠️ 局限性**

局限性包括：FLOPs归一化仅为近似，未考虑延迟、速率限制等部署约束；评估依赖单一GPT‑5评判器和固定规则，可能存在偏差；样本仅为200个英语目标，未覆盖多语、多轮、工具协同等场景；仅评估四类攻击范式，未涉及其他潜在方法。

---

## 415. ConvScale: Conversational Interviews for Scale-Aligned Measurement

**arXiv ID:** 2603.11988 | [PDF](https://arxiv.org/pdf/2603.11988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 416. Inverse Neural Operator for ODE Parameter Optimization

**arXiv ID:** 2603.11854 | [PDF](https://arxiv.org/pdf/2603.11854v1)

**作者:** Zhi-Song Liu `[一作]` (Lappeenranta-Lahti University of Technology), Michael Boy `[通讯]` (University of Helsinki)

**通讯引用:** 55046 | [OpenAlex ID](https://openalex.org/A5041685684)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段的逆向神经算子框架（INO），能够从稀疏、局部观测中恢复隐藏的ODE参数。

**💡 创新点**

创新点：① 引入条件傅里叶神经算子（C-FNO）配合交叉注意力，显著抑制高频伪影并保持时间连贯性；② 开发无雅可比矩阵的漂移模型（ADM），通过核加权残差场学习参数空间的全局向量场，实现快速、稳定的参数更新。

**🔧 技术方法**

主要技术：条件傅里叶神经算子（C-FNO）+ 交叉注意力；稠密核加权残差漂移模型（ADM）；基于残差的无梯度监督；对比传统梯度下降、MCMC、CMA-ES 等方法。

**📊 数据集**

使用了两个真实且具有挑战性的 ODE 数据集：POLLU（25个化学反应速率参数）和 GRN（40个基因调控系数），均采用稀疏时间点观测。

**📈 对比分析**

与梯度基、梯度无关和逆向算子方法对比，INO 在参数恢复 MSE/MAE 上均优于现有方法，速度提升约 487 倍（单样本约 0.23 s），同时保持或提高预测轨迹的准确性。

**⚠️ 局限性**

局限性：仅在两种 ODE 基准上验证，未覆盖不规则采样、异方差噪声、部分可观测等更复杂的实验情形；未来需推广至更大规模或 PDE 场景。

---

## 417. High-Precision 6DOF Pose Estimation via Global Phase Retrieval in Fringe Projection Profilometry for 3D Mapping

**arXiv ID:** 2603.11389 | [PDF](https://arxiv.org/pdf/2603.11389v1)

**作者:** Sehoon Tak `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**通讯引用:** 29810 | [OpenAlex ID](https://openalex.org/A5080001926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在移动数字相位投影仪（DFP）系统中加入固定全局投影仪，通过全局投影仪的相位约束和PnP式重投影目标实现高精度6自由度位姿估计，避免传统ICP在高点密度点云中的采样敏感性与特征不足问题。

**💡 创新点**

创新点包括：①利用固定全局投影仪提供的相位到像素的精确映射，形成全局参考框架；②采用批量子采样与共识过滤的重投影优化，消除对密集点云特征提取与对应的依赖；③引入几何一致性正则化与多批次一致性检验，使得估计在子采样、低重叠与无纹理表面下保持亚毫米精度；④通过蒙特卡罗误差传播分析给出校准与测量不确定性对位姿漂移的定量界限。

**🔧 技术方法**

使用技术包括：数字相位投影（Phase‑Shifting Profilometry）相位检索与灰度码相位展开；相机‑投影仪共识标定与相位到像素映射；PnP式重投影误差最小化（Levenberg–Marquardt/Adam优化）；批量子采样、共识过滤与几何一致性正则化；统计误差分析与蒙特卡罗灵敏度实验。

**📊 数据集**

采用自研的双投影仪/单相机结构光系统进行实验；数据集主要是：①平面标靶的多次静态重建（约40次）；②一件约200万点的雕塑模型在8个不同姿态下的稀疏/全量点云；未使用公开公共数据集，而是通过自定义相机/投影仪硬件和实验室环境采集。

**📈 对比分析**

方法与传统ICP、ICP+模型（ICP修正）在以下方面比较：在特征平面上实现子毫米位姿误差，最大位移偏差<0.9 mm；在低重叠与光滑表面下ICP失稳，而本方法保持稳定；在8个姿态的轨迹实验中，ICP累计误差可达18 mm/48 mrad，使用本方法后误差降至<0.3 mm/0.3 mrad；统计显著性检验（Wilcoxon符号秩检验）表明改进显著。

**⚠️ 局限性**

局限性包括：①需要额外的固定全局投影仪硬件，限制部署场景；②时间多路复用投影导致更新率受限（约2.5–3 帧/秒）；③在投影饱和、强反射、自遮挡或极端姿态下全局投影约束不足时会失败；④若全局投影仪校准漂移或相位偏差，可能引入系统性偏差；⑤方法对相位解包与灰度码解码错误敏感。

---

## 418. FinRule-Bench: A Benchmark for Joint Reasoning over Financial Tables and Principles

**arXiv ID:** 2603.11339 | [PDF](https://arxiv.org/pdf/2603.11339v1)

**作者:** Arun Vignesh Malarkkan `[一作]` (Arizona State University), Denghui Zhang `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 12213 | [OpenAlex ID](https://openalex.org/A5100366431)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FinRule-Bench基准，用于评估大型语言模型在真实财务报表上遵循会计准则的诊断完整性。

**💡 创新点**

创新点在于设计三种审计任务（规则验证、规则识别、联合诊断），引入因果-反事实提示机制，并提供可复现的验证器和错误注入流程。

**🔧 技术方法**

使用零样本、少量样本以及因果-反事实提示技术，结合基于规则的验证器和结构化提示。

**📊 数据集**

采用真实的美国上市公司2024年10-K文件中的四类报表（资产负债表、现金流量表、利润表、所有者权益变动表）及其人工制定的会计准则。

**📈 对比分析**

与GPT‑4o、Gemini 2.5 Pro、Gemini 2.0 Flash和LLaMA 3.3在三种提示方式下比较，发现对单规则验证准确率高，但在规则识别和多规则联合诊断上准确率显著下降，因果-反事实提示在轻量模型上有一定提升。

**⚠️ 局限性**

局限性包括仅测试已知规则、未处理跨报表一致性、缺少噪声/缺失数据、未进行监督微调或链式思考，因而不适用于真实审计部署。

---

## 419. Bielik-Minitron-7B: Compressing Large Language Models via Structured Pruning and Knowledge Distillation for the Polish Language

**arXiv ID:** 2603.11881 | [PDF](https://arxiv.org/pdf/2603.11881v1)

**作者:** Remigiusz Kinas `[一作]` (Bielik AI), Adrian Gwoździej `[通讯]` (Bielik AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在原有Bielik‑11B‑v3.0基础上，通过多轴结构化剪枝和logit‑based知识蒸馏，将模型压缩为7.35B（33.4%尺寸减小），并在后续对齐（SFT、DPO‑P、GRPO）中恢复90%以上的性能。

**💡 创新点**

创新点在于：①采用NVIDIA Minitron的多轴剪枝（depth、width、attention、MLP）与激活敏感性分析，实现硬件友好型压缩；②使用logit‑KL蒸馏而非传统交叉熵，减少预训练计算，仅用3%预训练数据即可恢复大部分能力；③针对波兰等欧洲语言量身定制的压缩与评估流程，提供一套可复制的中小规模LLM压缩蓝图。

**🔧 技术方法**

技术手段包括：结构化剪枝（利用Model Optimizer进行深度、宽度、注意力和MLP层的激活重要性评估）、logit‑based知识蒸馏（NeMo Framework的KL损失）、后期对齐管道（SFT、DPO‑P、GRPO），以及多种量化方案（Q6_K、Q8_0、Q4_K_M、FP8、NVFP4）。

**📊 数据集**

使用了波兰-英语混合的Bielik Dataset（约800万高质量样本）进行蒸馏与对齐，评估基准包括Open PL LLM、CPTUB、Polish Medical Leaderboard、INCLUDE‑base‑44、Belebele、FLORES、EuroEval、BFCL等多语种/任务测试。

**📈 对比分析**

通过与原始11B模型、其他7B/8B/14B等基准模型的对比，显示Bielik‑Minitron‑7B在大多数任务中取得与原始模型相近或超过同规模竞争者的分数；推理吞吐提升约50%，TPOT降低约32%；量化后4‑bit模型在保持≈99%性能的同时，显著降低显存需求。

**⚠️ 局限性**

局限性：①知识/事实推理性能下降约10%；②对动态工具调用的鲁棒性下降，Live场景表现不佳；③depth剪枝对多步推理更敏感，进一步压缩导致梯度不稳定；④FP8/NVFP4量化仍有3–4%性能损失；③目前仅针对波兰等欧洲语言，跨语言通用性需进一步验证。

---

## 420. Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models

**arXiv ID:** 2603.11896 | [PDF](https://arxiv.org/pdf/2603.11896v1)

**作者:** Lu Wang `[一作]` (Institute of Automation), Jun Zhao `[通讯]` (Institute of Automation)

**通讯引用:** 22546 | [OpenAlex ID](https://openalex.org/A5100744623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Think While Watching 框架，实现在线多轮视频推理，保持段级记忆并实现感知与生成并行。

**💡 创新点**

创新点在于段级记忆笔记、段级流式因果掩码与独立位置编码，三阶段训练以及自适应注意后端，使流式推理具备严格因果性和并行性。

**🔧 技术方法**

采用多模态大语言模型（Qwen3-VL）、多模态RoPE、双 KV 缓存并行推理、流式注意力、段级记忆写入与检索以及三阶段链式思维（CoT）数据集构建。

**📊 数据集**

训练使用 VideoChatOnline-IT（短视频）和 YouTube（长视频）构建的三阶段 CoT 数据集；评测在 StreamingBench、OVO-Bench、Video-MME、LV-Bench 等基准上。

**📈 对比分析**

与离线批处理、交互式间隔式基线以及多款开源/闭源模型对比，单轮流式准确率提升 2.6–3.8%，多轮流式保持准确率并将输出 token 减少约 56%，TTFT 大幅降低。

**⚠️ 局限性**

局限在于对长视频仍受段级记忆粒度与分段策略限制；极端帧丢失或噪声导致记忆写入不可靠；对极大规模实时流的资源开销仍待进一步评估。

---

## 421. Explicit Logic Channel for Validation and Enhancement of MLLMs on Zero-Shot Tasks

**arXiv ID:** 2603.11689 | [PDF](https://arxiv.org/pdf/2603.11689v1)

**作者:** Mei Chee Leong `[一作]` (Institute for Infocomm Research), Nancy Chen `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5041699269)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Explicit Logic Channel (ELC) 与 Consistency Rate (CR) 框架，结合 LLM、视觉特征提取器和逻辑推理，在零样本视觉-语言任务中实现模型验证、选择与性能提升。

**💡 创新点**

创新点在于：1) 与黑盒模型并行的显式逻辑通道，提供可解释的推理路径；2) 跨通道一致率指标，用于无标签评估与自动化模型增强；3) 在多任务（MC‑VQA、HC‑REC）上实现一致子集融合，显著提升准确率。

**🔧 技术方法**

技术手段包括：大型语言模型 (如 Qwen3‑4B‑Instruct)、视觉特征提取网络 (EvaCLIP‑8B、InternVL‑2.0‑8B)、概率推理逻辑算子（几何平均、关联概率）、一致率统计与融合公式。

**📊 数据集**

使用数据集：NegBench（COCO、VOC2007），HC‑RefCOCOg，HC‑RefLoCo，涵盖多选 VQA 与长文本指代表达的挑战性基准。

**📈 对比分析**

评估方式：在 11 种前沿开源 MLLM 上计算准确率与 CR，发现两者相关性极高（Pearson≈0.99）。一致子集融合后，模型准确率提升 5–15%，在三大基准上均创下新 SOTA。

**⚠️ 局限性**

局限性：1) 对模型共同知识覆盖度敏感，极端 OOD 场景一致率下降；2) 融合公式经验化，缺乏理论最优保证；3) 仅验证到两类任务，复杂 Chain‑of‑Thought 场景尚待扩展。

---

## 422. Grammar of the Wave: Towards Explainable Multivariate Time Series Event Detection via Neuro-Symbolic VLM Agents

**arXiv ID:** 2603.11479 | [PDF](https://arxiv.org/pdf/2603.11479v1)

**作者:** Sky Chenwei Wan `[一作]` (Telecom Paris), Aymeric Jan `[通讯]` (AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出知识指导的时间序列事件检测任务，并构建事件逻辑树（ELT）知识表示与SELA神经符号多代理系统，实现零样本、可解释的事件检测。

**💡 创新点**

核心创新在于将自然语言事件描述转换为层次化时间逻辑结构（ELT），并通过视觉‑语言模型与可视化工具在代理间协同完成事件实例化，既提高检索准确性又提供可解释的树结构。

**🔧 技术方法**

使用大型视觉‑语言模型（GPT‑4/5+VLM）、事件逻辑树与模糊逻辑算子、主动可视化工具以及神经符号多代理框架进行推理与实例化。

**📊 数据集**

基准数据集KITE，来自北海油田压力测试的多变量时间序列，包含易/难两大子集（drawdown、buildup、valid test、lost seal）。

**📈 对比分析**

与随机、低资源监督CNN/Transformer、时序基础模型、LLM/ VLM数值与可视化方法对比，SELA在零样本下F1@0.5≈83%接近人工，F1@0.9≈69%，明显优于其他方法。

**⚠️ 局限性**

受限于ELT解析质量、LLM结构预测能力、对复杂事件的解释细粒度不足，且在精确边界定位仍略逊于人工。

---

## 423. H2LooP Spark Preview: Continual Pretraining of Large Language Models for Low-Level Embedded Systems Code

**arXiv ID:** 2603.11139 | [PDF](https://arxiv.org/pdf/2603.11139v1)

**作者:** Amit Singh `[一作]` (H2LooP.ai), Jatin Kishnani `[通讯]` (H2LooP.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在嵌入式系统代码生成领域对开源7B OLMo-3模型进行持续预训练（CPT），并通过RSLoRA高秩LoRA实现域适配。

**💡 创新点**

创新点在于使用SpecMap层级数据映射构建大规模嵌入式语料、系统性高秩LoRA与RSLoRA参数搜索，以及证明单一小模型能在多数嵌入式子域击败超大前沿模型。

**🔧 技术方法**

采用BF16混合精度、RSLoRA LoRA、Flash Attention 2、Tensor Core TF32、分布式数据并行与梯度裁剪等技术实现高效训练。

**📊 数据集**

数据集基于818个仓库–数据表对，共76.4 GB原始嵌入式代码，经过SpecMap处理后生成约23.5 B token，涵盖117家制造商、19类组件。

**📈 对比分析**

通过离线推理与生成任务对比，Spark Preview在13个嵌入式子域中以70.4%困惑度下降、66.1%对外部仓库下降，并在8/13类别的生成准确率上超过Claude Opus 4.6和Qwen3‑Coder‑30B。

**⚠️ 局限性**

局限性包括：仅使用单一7B基模型、未完成完整训练、缺乏功能正确性与编译测试、可能存在记忆化风险、以及后期吞吐下降。

---

## 424. OpenSanctions Pairs: Large-Scale Entity Matching with LLMs

**arXiv ID:** 2603.11051 | [PDF](https://arxiv.org/pdf/2603.11051v1)

**作者:** Chandler Smith `[一作]` (University of Oxford), Christian Schroeder de Witt `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了OpenSanctions Pairs实体匹配基准，覆盖755,540对真实制裁数据的标注，填补了学术界与实际合规场景之间的空白。

**💡 创新点**

创新点在于构造了规模极大、来源多样、跨语言脚本、含多值属性的制裁数据集，并对比了传统规则匹配与大语言模型（LLM）的表现，发现LLM已逼近人类标注一致性。

**🔧 技术方法**

使用了规则基线（Nomenklatura RegressionV1）、开源LLM（Llama‑3.1‑8B、DeepSeek‑R1‑Distill‑Qwen‑14B）和封闭源LLM（GPT‑4o、GPT‑5系列、Claude等），并通过DSPy MIPROv2进行提示优化。

**📊 数据集**

数据集为OpenSanctions汇总的293个制裁源，涵盖31个国家，包含1,002,093个实体及其多属性信息，提供了丰富的多语言、多脚本、多值字段样本。

**📈 对比分析**

在零射击和少量样本场景下，LLM的F1分数最高达98.95%（GPT‑4o），优于规则基线91.33%，表明LLM能可靠复制专家判断；提示优化虽提升约1–2个百分点，但已接近性能上限。

**⚠️ 局限性**

局限性包括标签基于专家判断非绝对真值、对跨脚本转写和小误差敏感、对实时低延迟匹配的可扩展性不足，以及提示优化对不同模型效果差异显著，需进一步研究。

---

## 425. Partitioning Israeli Municipalities into Politically Homogeneous Cantons: A Constrained Spatial Clustering Approach

**arXiv ID:** 2603.11805 | [PDF](https://arxiv.org/pdf/2603.11805v1)

**作者:** Adir Elmakais `[一作]` (Bar-Ilan University), Oren Glickman `[通讯]` (Bar-Ilan University)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5018914598)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对以色列229个市政区进行基于选举结果的空间聚类，划分连通且政治同质的“canton”地区。

**💡 创新点**

将约束空间聚类与多目标优化相结合，系统评估264种配置，首次在以色列政治地理中提出NMF+Louvain得到可解释的五区划分，并提供交互式Web可视化。

**🔧 技术方法**

使用Simulated Annealing、Agglomerative聚类、Louvain社区检测、K‑Means；特征表示包括BlocShares、RawParty、PCA、NMF；距离度量为欧氏、余弦、Jensen‑Shannon；评价指标为轮廓系数、人口平衡CV、ARI、NMI。

**📊 数据集**

以色列中央统计局公布的五次国会选举市政区投票结果，地理边界Shapefile，以及党派到政治阵营的映射。

**📈 对比分析**

通过网格搜索对264配置进行比较，记录轮廓系数、人口平衡等；Agglomerative+BlocShares+Euclidean在K=3得到最高0.905的轮廓；NMF+Cosine+Louvain在K=5得到0.121但人口平衡好且ARI=1.0；K‑Means高轮廓但不连通；SA在平衡性上优先但轮廓较低。

**⚠️ 局限性**

存在人口不平衡、SA随机性、生态谬误、仅市政区级别、数据覆盖不完全、可能的选择偏差、缺乏对动态变化的建模。

---

## 426. The Unlearning Mirage: A Dynamic Framework for Evaluating LLM Unlearning

**arXiv ID:** 2603.11266 | [PDF](https://arxiv.org/pdf/2603.11266v1)

**作者:** Raj Sanjay Shah `[一作]` (Georgia Institute of Technology), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13486 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于动态知识图的评估框架，用来检验大语言模型在去学习（unlearning）后是否真正遗忘。

**💡 创新点**

通过从模型自身提取知识构建实体特定的知识图，自动生成单跳、多跳及别名变体查询，实现无人工标注的动态评测，揭示多跳查询的脆弱性。

**🔧 技术方法**

知识图构建（BFS+指数衰减）、查询生成与别名解析、PatchScopes内部激活分析、量化评测指标（多跳遗忘分、保留分、调和平均）等技术。

**📊 数据集**

RWKU、TOFU、LLaMA‑3.1‑Instruct、Phi‑4‑mini‑instruct、Granite‑3.2‑8B‑Instruct 等数据集，实体如 Stephen King；无手工标注的自建查询。

**📈 对比分析**

与现有静态评测（RWKU、TOFU 等）对比，覆盖率约 78%/66%；Spearman 相关性 0.87/0.75；多跳查询显著提升残留知识；优化方法表现差，ULD 方法在遗忘与保留间取得最佳平衡（整体≈80%）。

**⚠️ 局限性**

低频/专业知识提取困难；实体与保留集重叠时无法完全识别；知识图构建非确定性；评测仅关注结构化查询，未覆盖通用任务；评估指标仍为近似。

---

## 427. From Control to Foresight: Simulation as a New Paradigm for Human-Agent Collaboration

**arXiv ID:** 2603.11677 | [PDF](https://arxiv.org/pdf/2603.11677v1)

**作者:** Gaole He `[一作]` (National University of Singapore), Brian Y. Lim `[通讯]` (National University of Singapore)

**通讯引用:** 3809 | [OpenAlex ID](https://openalex.org/A5056248594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在人工智能代理决策前通过模拟未来情景的交互范式，帮助人类用户进行前瞻性决策

**💡 创新点**

创新点在于将内部搜索树外化为可视化的多路径模拟，提供对后续影响的前瞻性洞察，打破单路径监督的局限

**🔧 技术方法**

利用大型语言模型（LLM）进行多路径探索与模拟，并通过文本描述和可视化呈现模拟结果

**📊 数据集**

未使用特定数据集，主要基于示例性旅行规划场景进行概念验证

**📈 对比分析**

论文为概念性讨论，未给出实验比较，提出了设计维度和可能的评估路径

**⚠️ 局限性**

主要限制包括模拟可靠性不足、可能出现幻觉、信息过载与认知负担，以及对开放域世界模型的依赖

---

## 428. Towards Trustworthy Selective Generation: Reliability-Guided Diffusion for Ultra-Low-Field to High-Field MRI Synthesis

**arXiv ID:** 2603.11325 | [PDF](https://arxiv.org/pdf/2603.11325v1)

**作者:** Zhenxuan Zhang `[一作]` (Imperial College London), Guang Yang `[通讯]` (Imperial College London)

**通讯引用:** 17850 | [OpenAlex ID](https://openalex.org/A5108053324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了可靠性感知扩散框架 ReDiff，用于低场（64 mT）到高场（3 T）MRI 的合成。

**💡 创新点**

创新点在于：① 在扩散采样阶段引入空间可靠性映射（RGS）抑制不可靠高频细节放大；② 在后期通过不确定性加权聚合（UCS）多候选结果，进一步提升结构一致性与鲁棒性。

**🔧 技术方法**

技术实现包括条件扩散 U‑Net、可靠性引导采样、基于方差的候选筛选与加权、对抗与循环一致性损失，以及 PSNR/SSIM/LPIPS 等评估指标。

**📊 数据集**

使用了两套配对数据集：私有 20 受试者（T1w、T2w、FLAIR）64 mT→3 T 以及公开 Leiden 11 受试者 64 mT→3 T 数据。

**📈 对比分析**

与 Pix2Pix、ESRGAN、TranUnet、ResViT、CyTran、SynDiff、MiDiffusion 等基线比较，ReDiff 在 PSNR/SSIM 上基本相当甚至略优，LPIPS 明显最低，结构细节更清晰，后续 SynthSeg 分割 Dice 分数亦更高。

**⚠️ 局限性**

局限性包括：仍受低场信号缺失导致的本质不确定性限制；训练需要大量配对样本；计算成本高；未充分验证在更低 SNR 或跨模态（如多对比度）场景下的泛化能力。

---

## 429. SemBench: A Universal Semantic Framework for LLM Evaluation

**arXiv ID:** 2603.11687 | [PDF](https://arxiv.org/pdf/2603.11687v1)

**作者:** Mikel Zubillaga `[一作]` (University of the Basque Country), German Rigau `[通讯]` (University of the Basque Country)

**通讯引用:** 6194 | [OpenAlex ID](https://openalex.org/A5073264941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个全自动、语言无关的评估框架SemBench，用词典定义与句子编码器对LLM进行语义理解评估；

**💡 创新点**

创新点在于仅使用词典的词义定义（无需使用例句）以及生成式方式自动生成测试实例，省却人工标注，实现低资源语言可扩展；

**🔧 技术方法**

主要技术包括LLM（Gemma、Qwen3、Llama等）在零样本或五样本提示下生成例句/定义，使用EmbeddingGemma句子编码器计算定义相似度，并以相似度阈值判定词义；

**📊 数据集**

使用的资源有英文 Oxford Dictionary of English、西班牙 RAE 字典、巴斯克 EEH 字典，以及传统 Word‑in‑Context 基准数据；

**📈 对比分析**

通过与 Word‑in‑Context 评测的 Spearman 相关系数对SemBench结果进行对比，英语相关系数达0.93，西班牙0.77，巴斯克0.66，且仅需 250‑500 个实例即可获得稳定排名；

**⚠️ 局限性**

局限性包括对单一多语言编码器的依赖可能引入偏差；未对商业 LLM 进行评估；与更广泛的 LLM 评价体系（如 LLMArena）对齐情况尚待验证。

---

## 430. Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization

**arXiv ID:** 2603.12026 | [PDF](https://arxiv.org/pdf/2603.12026v1)

**作者:** Haotong Duan `[一作]` (Hangzhou Dianzi University), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12228 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并提出基于单元化矩阵乘积状态（UMPS）的生成模型，并通过黎曼优化与空间解耦实现高效训练。

**💡 创新点**

首次在生成模型中引入归一化约束的单元化MPS，消除尺度自由度；同时将低秩与球面约束空间解耦，在黎曼流形上直接优化，显著提升稳定性与收敛速度。

**🔧 技术方法**

利用黎曼优化、空间解耦法、D-MRG更新、混合规范化、SVD分解与MPS结构进行参数优化，并通过链式采样产生样本。

**📊 数据集**

实验使用Bars‑and‑Stripes (BAS) 二进制图像数据集和 EMNIST 手写数字/字母数据集。

**📈 对比分析**

与传统MPS模型比较，使用 NLL、训练时间等指标评估。UMPS‑SD 在相同循环数下 NLL 降低更快，收敛时间缩短约 27 倍，生成样本质量和重建效果明显优于基线。

**⚠️ 局限性**

目前仅适用于二值图像，链式 MPS 的表达能力受限，无法直接处理彩色图像；缺乏自适应学习率与随机梯度方差抑制机制，且对更高阶张量网络（如 PEPS）的优化仍面临挑战。

---

## 431. When OpenClaw Meets Hospital: Toward an Agentic Operating System for Dynamic Clinical Workflows

**arXiv ID:** 2603.11721 | [PDF](https://arxiv.org/pdf/2603.11721v1)

**作者:** Wenxian Yang `[一作]` (Independent Researcher), Jiahong Dong `[通讯]` (Beijing Tsinghua Changgung Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了针对医院环境的“Agentic Operating System”，通过限制代理的执行权限、使用文档突变协调、基于清单的分层内存以及可按需组合的医学技能库，构建了一个安全、可审计、支持长期上下文的医疗AI系统。

**💡 创新点**

创新点在于：①把代理的安全控制转移到操作系统层面，仅通过文件读写实现交互；②引入文档突变事件驱动的多代理协调模型；③用可读清单实现渐进式检索，替代向量检索；④实现动态技能组合以应对医院工作流的长尾变异。

**🔧 技术方法**

技术包括：Linux用户隔离、AppArmor/SELinux安全策略、文件系统权限、事件订阅与版本化日志、LLM生成的清单维护、技能库接口以及基于LLM的清单导航。

**📊 数据集**

论文未给出具体数据集，假设使用医院内部的电子健康记录（EHR）和结构化文档。

**📈 对比分析**

论文未进行实验或性能比较，主要为概念与架构设计；若有实验，需在真实EHR环境下验证安全性、审计性与检索效率。

**⚠️ 局限性**

局限包括：清单生成的准确性与更新延迟、并发写入的一致性、与现有EHR系统的互操作性、以及缺乏实证评估。

---

## 432. NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction

**arXiv ID:** 2603.12063 | [PDF](https://arxiv.org/pdf/2603.12063v1)

**作者:** David Svitov `[一作]` (Università degli Studi di Genova), Mahtab Dahaghin `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为NBAvatar的面部与手部互动的逼真渲染方法，能处理非刚性面部变形与色彩变化

**💡 创新点**

引入Neural Billboards——将面向表面的平面原语与延迟神经渲染结合的混合表示，并通过中间轮廓监督实现几何与外观的解耦

**🔧 技术方法**

采用基于Billboard Splatting的显式几何、延迟神经渲染（DNR）+神经纹理、UNet解码器、位置动态模拟（PBD）、FLAME/MANO参数化模型

**📊 数据集**

在公开的Decaf手脸交互多视角数据集上进行训练与评估

**📈 对比分析**

与GaussianAvatars、SplattingAvatar及InteractAvatar比较，使用PSNR/SSIM/LPIPS三项指标，NBAvatar在高分辨率下平均LPIPS降低约30%，PSNR/SSIM均显著提升，且在自重现与新视角合成中表现最优

**⚠️ 局限性**

方法对3DMM拟合精度敏感，若拟合误差大易导致视觉与量化指标下降

---

## 433. FL-MedSegBench: A Comprehensive Benchmark for Federated Learning on Medical Image Segmentation

**arXiv ID:** 2603.11659 | [PDF](https://arxiv.org/pdf/2603.11659v1)

**作者:** Meilu Zhu `[一作]` (University of Hong Kong), Edmund Y. Lam `[通讯]` (University of Hong Kong)

**通讯引用:** 9455 | [OpenAlex ID](https://openalex.org/A5008832723)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了FL-MedSegBench，一个涵盖9个医学分割任务、10种影像模态（2D/3D）并按医院/设备划分的真实非IID联邦学习基准。

**💡 创新点**

创新点在于首次提供统一、可复现的医学分割联邦学习基准，系统评估8种通用FL与5种个性化FL方法，并公开完整工具包。

**🔧 技术方法**

使用技术包括FedAvg、FedProx、FedLWS、FedRDN、FedNova、FedBN、SioBN、FedPer、Ditto等通用/个性化联邦算法，模型为U-Net/3D‑U‑Net/SA‑Net，采用Adam、学习率衰减与数据增强。

**📊 数据集**

数据集来自27个公开医学分割数据集，如Vessel、Prostate、COSAS、BUS、MG、Polyp、Pancreas、M&Ms、FeTS2022，按医院或设备划分为客户端。

**📈 对比分析**

在9个任务上对13种方法进行Dice、公平性、通信效率、收敛行为及未见域泛化评估，结果表明个性化FL（FedBN、SioBN、FedPer）普遍优于通用FL，FedBN在大多数任务上表现最稳健，但不存在单一方法在所有任务上统治。

**⚠️ 局限性**

局限性包括：在极端非IID情况下公平性和鲁棒性仍待提升；部分个性化方法在小样本或高噪声场景下表现不稳定；泛化评估仅覆盖少量未见域；缺乏对通信-性能权衡的理论深度分析。

---

## 434. Energy Prediction on Sloping Ground for Quadruped Robots

**arXiv ID:** 2603.11963 | [PDF](https://arxiv.org/pdf/2603.11963v1)

**作者:** Mohamed Ounally `[一作]` (Universite Clermont Auvergne), Johann Laconte `[通讯]` (Universite Clermont Auvergne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究斜坡倾角和行进方向对四足机器人能耗的影响，提出基于标准机载传感器的简易能耗模型，并在自然斜坡环境中进行现场验证。

**💡 创新点**

创新点在于：①仅利用电池电压/电流、IMU 与里程计等普通机载传感器即可建模；②提出能耗与坡度、航向角的简单线性映射；③实现模型的现场标定与路径级能耗评估。

**🔧 技术方法**

技术手段包括：电池电量测量、IMU姿态与加速度、机身速度里程计；信号预处理（中值滤波、指数滑动平均）与功率与速度的线性合成；基于实验数据拟合势能/力矩成分。

**📊 数据集**

使用的“数据集”为在Unitree B1平台上收集的斜坡实验数据：坡度 5°–20°、不同航向角（上坡、下坡、横坡等）、恒定速度 0.3 m/s 的行驶段；数据仅包含机载传感器输出。

**📈 对比分析**

通过比较模型预测的能量与实际测量的能量，验证了能量的可加性：两段组合路径与单段路径的相对误差约为 4%–13%；实验显示斜坡与航向对能耗具有线性、可加的近似关系，满足规划层面能耗评估需求。

**⚠️ 局限性**

局限性在于：模型假设功矩仅随坡度和航向变化，未考虑速度、步态、地形细节等因素；旋转能耗的实验结果尚未充分验证；需要在更广泛的速度、步态、地形和不同机器人平台上进一步验证与泛化。

---

## 435. Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting

**arXiv ID:** 2603.11543 | [PDF](https://arxiv.org/pdf/2603.11543v1)

**作者:** Tingxuan Huang `[一作]` (Tsinghua University), Bin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 51789 | [OpenAlex ID](https://openalex.org/A5100372375)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Mango-GS，一个基于节点引导的多帧高保真动态场景重建框架。

**💡 创新点**

创新点在于将控制节点解耦为位置与潜在码，并引入多帧时间Transformer，实现语义一致的运动传播与高效的时间一致性。

**🔧 技术方法**

采用3D高斯切片、稀疏控制节点、解耦节点表示、时间Transformer与自注意力、以及顶k和运动感知损失。

**📊 数据集**

在HyperNeRF和Neural 3D Video这两个真实动态场景数据集上进行评估。

**📈 对比分析**

与D-3DGS、E-D3DGS、4DGS、SC-GS、GaGS、MotionGS、TimeFormer等方法对比，Mango-GS在PSNR/SSIM/LPIPS及tLPIPS上取得最优或竞争性表现，并以约150 FPS的速度显著领先。

**⚠️ 局限性**

局限在于对极端长时间序列的处理仍不充分，需要更长的时间窗口或在线适应；大幅运动下的细节仍有待改进。

---

## 436. Modeling Trial-and-Error Navigation With a Sequential Decision Model of Information Scent

**arXiv ID:** 2603.11759 | [PDF](https://arxiv.org/pdf/2603.11759v1)

**作者:** Xiaofu Jin `[一作]` (Aalto University), Antti Oulasvirta `[通讯]` (Aalto University)

**通讯引用:** 14431 | [OpenAlex ID](https://openalex.org/A5003084232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于顺序决策的资源合理信息香气模型，模拟用户在层级信息架构中逐步检索、选择和回溯的导航行为。

**💡 创新点**

创新点在于将信息香气从传统的静态、瞬时评估扩展为在记忆限制和感知噪声下的POMDP框架，解释了预期外的提前选择、回溯和重访等试错行为，并通过强化学习学习最优策略。

**🔧 技术方法**

使用技术包括：POMDP建模、感知噪声与记忆衰退机制、基于语义相似度的香气计算（Sentence-Transformer）、强化学习（policy网络）以及参数化的资源限制（记忆容量、噪声、衰减率）。

**📊 数据集**

数据集为重建的HTML菜单实验（模仿Blackmon、Habuchi等的层级菜单），包含三种难度、两种层级深度、四种目标位置的多条测试条件。

**📈 对比分析**

比较方法是：用模型生成的行为统计（步骤数、点击数、成功率、失误率、定位位置偏差）与已有实验中的趋势进行对比。模型在任务难度、层级深度和目标位置三项指标上均重现了人类实验的方向性变化，并通过消融实验验证各组件对性能的贡献。

**⚠️ 局限性**

限制包括：缺乏真实用户过程级轨迹数据，模型未考虑个体差异或先验知识；对真实网站的外部验证有限；以及对超参数（如记忆阈值、噪声水平）的敏感性需要进一步探索。

---

## 437. Preliminary analysis of RGB-NIR Image Registration techniques for off-road forestry environments

**arXiv ID:** 2603.11952 | [PDF](https://arxiv.org/pdf/2603.11952v1)

**作者:** Pankaj Deoli `[一作]` (Robotics Research Laboratory University of Kaiserslautern-Landau), Karsten Berns `[通讯]` (Robotics Research Laboratory University of Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文对离路林业环境下RGB-NIR图像配准进行了评估，比较了传统特征匹配、直方图匹配、模板匹配等经典方法以及NeMAR和MURF等深度学习方法；

**💡 创新点**

创新点在于将NeMAR在六种不同训练配置下进行系统实验，并针对林业场景提出多尺度融合与域特定改进的必要性；

**🔧 技术方法**

使用了传统的特征点匹配、直方图匹配、模板匹配、互信息、Fourier配准等方法，以及基于GAN的NeMAR、互相强化的MURF等深度学习模型；

**📊 数据集**

采用了约5400张RGB图像及其对应的RED、REG、NIR、GREEN等多光谱数据的林业数据集；

**📈 对比分析**

通过视觉对比和L1/ GAN损失、判别器得分等指标比较，传统方法表现差强人意；NeMAR在某些配置下实现了像素级对齐但GAN损失不稳定；MURF在大尺度特征上效果良好，却无法保持细小树枝和叶片细节；

**⚠️ 局限性**

主要限制包括光谱不匹配导致特征匹配困难、密集植被和光照变化导致对齐误差、GAN训练不稳定、缺乏细节保持及在不同季节和遮挡条件下的泛化能力不足。

---

## 438. Deep Learning Network-Temporal Models For Traffic Prediction

**arXiv ID:** 2603.11475 | [PDF](https://arxiv.org/pdf/2603.11475v1)

**作者:** Yufeng Xin `[一作]` (RENCI), Ethan Fan `[通讯]` (RENCI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并评估了两种面向网络多变量时间序列预测的深度学习模型——基于图注意力网络的ST-GAT模型和跨模态大语言模型（LLM）Fine-Tuning+Spearman聚类的Cluster-CALF模型，并与传统LSTM模型进行对比实验。

**💡 创新点**

创新点主要包括：① 将图注意力机制与多层LSTM耦合以同时捕获空间拓扑与时间序列依赖；② 设计跨模态LLM Fine-Tuning框架（CALF），通过时间序列与文本分支的多层对齐和特征一致性损失，弥补了LLM在时序预测中的分布差异；③ 在LLM框架中加入Spearman相关聚类预处理（Cluster-CALF），显著降低多变量输入的多重共线性与计算复杂度。

**🔧 技术方法**

使用技术包括：图注意力网络（GAT）、双层LSTM、Huber损失、标准化/正则化、LoRA参数高效微调、交叉模态匹配模块、Spearman相关聚类、sMAPE评估指标等。

**📊 数据集**

使用数据集为某互联网骨干服务商一年间每小时测量的交通流量数据，约100条双向链路，形成多变量时间序列。

**📈 对比分析**

对比方法：在相同的数据拆分（训练/验证/测试）与相同超参数搜索空间下，将三种模型的sMAPE、MAE等指标进行统计；Cluster-CALF在最佳配置下平均sMAPE下降约41%相较于LSTM，且标准差缩小约29%；ST-GAT在减少预测方差方面表现突出，但整体平均误差略高于LSTM。

**⚠️ 局限性**

局限性：① 过拟合风险，尤其在ST-GAT中；② 聚类预处理虽有效但在大规模网络时计算开销较大；③ 仅在单一骨干网络数据上验证，缺乏跨域通用性验证；④ 对LLM的依赖导致训练成本与模型推理延迟较高。

---

## 439. ZeroSense:How Vision matters in Long Context Compression

**arXiv ID:** 2603.11846 | [PDF](https://arxiv.org/pdf/2603.11846v1)

**作者:** Yonghan Gao `[一作]` (Shenzhen University of Advanced Technology), Xingyu Zeng `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的视觉文本压缩（VTC）评估框架，并创建ZeroSense基准，用以剥离多模态大模型的语言先验影响，从而准确衡量压缩后文本的保真度。

**💡 创新点**

创新点在于：1) 将VTC评估从下游任务迁移到纯OCR恢复任务；2) 用零语义相关的ZeroSense数据集构建“语义真空”，彻底消除上下文补偿；3) 通过分解公式将整体性能拆解为文本保留、原始识别与语义推断三部分，计算K_quality、F_prior和OCR_raw，实现对文本保留率的客观量化。

**🔧 技术方法**

使用的技术包括：视觉文本渲染映射θ、压缩率ρ(θ)定义、OCR原始识别评估、ZeroSense文本生成器（基于语言模型的低概率采样）、布局参数反向提取（字体大小、框尺寸、文本容量）以及对OCR性能线性衰减建模。

**📊 数据集**

使用的数据集：Fox、Omni、DI-100 v1.3（用于混淆实验）和自构建的ZeroSense基准；实验对比了不同压缩倍率（7.5×~17.5×）下的标准DeepSeek‑OCR与框架计算的K_quality、F_prior、OCR_raw。

**📈 对比分析**

比较方法：在同一压缩倍率下，评估标准DeepSeek‑OCR的下游任务准确率与框架计算的K_quality。结果显示：在Fox数据集，标准准确率在17.5×时仍为81.3%，但K_quality仅为27.4%，表明大部分提升来自语义补偿；在Omni数据集，K_quality与标准准确率相近（如7.5×时为97.1% vs 89.2%），验证框架可揭示真实文本保留率。其他指标如F_prior随压缩倍率上升而显著增大，OCR_raw随倍率下降，体现了模型原始识别的衰退。

**⚠️ 局限性**

局限性：1) 评估仍以OCR恢复为核心，无法覆盖更复杂的多模态下游任务；2) ZeroSense基准仅针对字符/单词级语义消除，可能无法完全消除所有语言先验；3) 线性衰减假设对所有模型可能不适用；4) 目前仅在英文字母/中文字符数据上验证，跨语言/脚本的泛化性待进一步研究。

---

## 440. O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**arXiv ID:** 2603.12144 | [PDF](https://arxiv.org/pdf/2603.12144v1)

**作者:** Mengfei Duan `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**通讯引用:** 5286 | [OpenAlex ID](https://openalex.org/A5027010844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于全景视觉的全端到端开放词汇占据预测框架 O3N，能够从单张全景 RGB 图像预测 360° 场景的三维占据和语义，支持未知类别。

**💡 创新点**

创新点在于三项：1) Polar‑spiral Mamba (PsM) 通过螺旋扫描极坐标的 3D 体素实现连续空间建模；2) Occupancy Cost Aggregation (OCA) 构造体素‑文本相似度成本体积并进行空间与类别聚合；3) Natural Modality Alignment (NMA) 采用无梯度的随机行走机制消除文本‑像素‑体素的模态差距。

**🔧 技术方法**

技术方案包括极坐标卷积、Mamba 轻量级注意力、Atrous Spatial Pyramid Pooling、CLIP 文本编码、EMA 与随机行走对齐、以及 MonoScene/SGN 语义占据网络的整合。

**📊 数据集**

使用了 QuadOcc（四足机器人 360° 实景）和 Human360Occ（CARLA 人体自我 360°）两大全景占据数据集，分别包含 6 类和 10 类语义标签。

**📈 对比分析**

与 OVO、SSCNet、OccFormer、VoxFormer‑S 等闭集或开放词汇方法对比，O3N 在 QuadOcc 上整体 mIoU 达 16.54、Novel mIoU 21.16；在 Human360Occ 上整体 mIoU 24.25，均超过现有最优方法，表现出更强的跨场景与未知类别泛化。

**⚠️ 局限性**

局限性包括对 ERP 失真和采样不均匀仍有一定依赖；NMA 的随机行走需调参，过大会导致不稳定；对极小物体或稀有类别的识别仍不够精准；整体模型在极端稀疏标签场景下的学习效率待进一步提升。

---

## 441. Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs

**arXiv ID:** 2603.12078 | [PDF](https://arxiv.org/pdf/2603.12078v1)

**作者:** Hiran Sarkar `[一作]` (Sony Research), Benjamin Busam `[通讯]` (Technical University of Munich)

**通讯引用:** 1624 | [OpenAlex ID](https://openalex.org/A5067135033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 Node-RF，一种将 Neural ODE 与 NeRF 结合的框架，用于学习连续时间的动态场景表示。

**💡 创新点**

创新点在于通过在隐空间中学习可微分动力学，实现在无限时间外推与跨序列泛化，而非仅限离散帧插值。

**🔧 技术方法**

核心技术包括 NeRF 的动态辐射场、Neural ODE 用于隐状态演化、以及 Lipschitz 正则化以结构化潜在空间。

**📊 数据集**

使用了多种数据集：单序列的 Bouncing Balls、Pendulum、DyNeRF；多序列的 Oscillating Ball 与 Bifurcating Hill。

**📈 对比分析**

与 D-NeRF、4D-GS、MotionGS、HexPlane、TiNeuVox、SimVP 等基线比较，Node-RF 在长时外推、光流 IoU、SSIM/LPIPS/PSNR 等指标上均优于或与之相当，尤其在物理可行性与连续性上表现最佳。

**⚠️ 局限性**

局限性包括对非确定性场景适应性有限、训练成本高、以及对真实大规模数据的可扩展性待验证。

---

## 442. Coupling Tensor Trains with Graph of Convex Sets: Effective Compression, Exploration, and Planning in the C-Space

**arXiv ID:** 2603.11658 | [PDF](https://arxiv.org/pdf/2603.11658v1)

**作者:** Gerhard Reinerth `[一作]` (Technical University of Munich), Marcello Romano `[通讯]` (Technical University of Munich)

**通讯引用:** 2908 | [OpenAlex ID](https://openalex.org/A5074014301)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

提出一种将张量压缩与图优化相结合的运动规划框架TANGO，用于高维机器人配置空间的采样、可行性建模、凸区域生成和最短路径规划。

**💡 创新点**

创新点在于：①使用TT-Cross对任务特定采样度量的概率密度进行低秩张量压缩；②将压缩后的分布用于高效采样并利用IRIS生成安全凸集；③把这些凸集嵌入Graph of Convex Sets（GCS）中，完成离散与连续规划的统一；④通过概率密度驱动的采样大幅提升采样效率与轨迹质量。

**🔧 技术方法**

核心技术包括：Tensor Train（TT）分解与TT-Cross采样、逆PDF构造、RNN-DBSCAN聚类、IRIS凸集膨胀、Graph of Convex Sets（GCS）以及最短路径求解。

**📊 数据集**

实验数据集：3-DoF平面机械臂（Lück等）和锁定后4-DoF（Franka Emika Panda）机器人；在这些系统上进行多次随机起止点的规划实验。

**📈 对比分析**

与标准RRT进行对比：TANGO在路径平滑度、最少点数、PDF评分等指标上更优；虽然平均执行时间略高，但生成的轨迹更少、约束满足度更好；在内存占用方面TT-Cross在重塑维度后显著降低。

**⚠️ 局限性**

局限性：①目前仅在低维（≤4-DoF）场景验证，扩展到更高维仍需研究；②结果高度依赖采样度量和张量维度/排序；③未考虑任务空间障碍物，需进一步整合SDF或配置空间SDF；④凸集生成与GCS构造可能产生二次计算瓶颈。

---

## 443. Temporal Text Classification with Large Language Models

**arXiv ID:** 2603.11295 | [PDF](https://arxiv.org/pdf/2603.11295v1)

**作者:** Nishat Raihan `[一作]` (George Mason University), Marcos Zampieri `[通讯]` (George Mason University)

**通讯引用:** 6614 | [OpenAlex ID](https://openalex.org/A5024937008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对多种主流专有与开源大型语言模型在时间文本分类（Temporal Text Classification, TTC）任务上进行系统评估，比较零-shot、5-shot提示以及 LoRA 微调的效果，并探讨合成数据生成对模型性能的影响。

**💡 创新点**

首次全面评估专有与开源 LLM 在 TTC 上的表现，并将微调、提示和合成数据三者结合，揭示它们在跨语言、不同时间跨度数据集上的相对优势与局限。

**🔧 技术方法**

使用零/少量提示、LoRA 微调、SVM 基准、宏 F1 评价指标，以及利用专有 LLM 生成的合成文本进行测试。

**📊 数据集**

Colonia（葡萄牙语 16 世纪-20 世纪小说集合）、CLMET（英国英语 1710-1920 语料）和 DTE（英国《Spectator》杂志 1700-2010 文本）。

**📈 对比分析**

通过与 SVM 基准对比并使用宏 F1 作为统一指标，结果显示：专有 LLM 在 5-shot 提示下往往能达到 0.90+ 的 F1，开源 LLM 在 100 token 输入下零-shot 约 0.26-0.36，5-shot 提升至 0.44-0.53，LoRA 微调后可达 0.61-0.77，整体仍与专有模型存在一定差距。

**⚠️ 局限性**

仅针对三大数据集，提示策略仅限零/5-shot，微调仅在开源模型上进行，合成数据生成与评估仍不稳定，且可能存在预训练数据泄漏风险。

---

## 444. Resolving Java Code Repository Issues with iSWE Agent

**arXiv ID:** 2603.11356 | [PDF](https://arxiv.org/pdf/2603.11356v1)

**作者:** Jatin Ganhotra `[一作]` (IBM), Martin Hirzel `[通讯]` (IBM)

**通讯引用:** 4766 | [OpenAlex ID](https://openalex.org/A5079080602)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 iSWE，一套双子代理（定位 + 编辑）自动化修复 Java 代码仓库问题的系统；

**💡 创新点**

创新点在于结合规则化 Java 静态分析工具与 LLM，使用只读工具与容器化编译降低副作用，采用 ReAct 两阶段子代理流程，显著提升定位与编辑准确率；

**🔧 技术方法**

采用多模型 LLM（如 Claude‑4.5‑Sonnet、DeepSeek 等）、PDL 提示语言、CLDK 与 Tree‑Sitter 规则化静态分析工具、工具调用与容器化编译；

**📊 数据集**

使用 Multi‑SWE‑Bench（Java）和 SWE‑PolyBench（Java）两个多语言基准的数据集；

**📈 对比分析**

通过与 MSWE‑agent、MopenHands、InfCode 等领先系统在同一基准上对比，iSWE 在 Java 子集上达到或接近最高的解决率，同时成本低 2–3 倍、token 使用更少，定位/编辑准确率也优于多数对手；

**⚠️ 局限性**

局限在于缺少动态测试反馈、对多文件复杂度的进一步探索不足，尚未进行模型微调，部分基准标注/id hints 的处理仍有限，导致在极高复杂度案例上的表现仍有提升空间。

---

## 445. Pivot based correlation clustering in the presence of good clusters

**arXiv ID:** 2603.12052 | [PDF](https://arxiv.org/pdf/2603.12052v1)

**作者:** David Rasmussen Lolck `[一作]` (University of Copenhagen), Shuyi Yan `[通讯]` (University of Copenhagen)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5026107997)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种新的相关聚类（correlation clustering）算法，该算法融合了经典Pivot算法与基于原子（atom）的聚类方法，目标是既获得更好的理论逼近比，又保持良好的实践效果。

**💡 创新点**

创新点在于：①通过在每一步先尝试定位并移除“非常好”或“良好”聚类（good/atom clusters），然后再进行Pivot步骤，显著降低了Pivot算法在无好聚类图上的 3 倍逼近限制；②构造了一套概率检测和扩展机制，保证在高噪声下仍能以 O(m log n) 的时间完成全局聚类；③实现了从 3 逼近到 2.9991 的理论改进。

**🔧 技术方法**

使用的主要技术包括：Pivot 聚类步骤、基于邻居集合的 α‑β‑goodness 判定、随机抽样与估计的预处理（[̌α,β,γ] 过程）、原子扩展与随机投影、以及对可行性与成本的期望分析（使用三角计数与 LP‑relaxation 等工具）。

**📊 数据集**

实验采用合成的“植入分区（planted partition）”图：随机生成 k 个簇，每簇内部完全图，随后以概率 p 随机翻转边，测试不同噪声水平 p 的影响。

**📈 对比分析**

与传统 Pivot、原子查找、以及原始簇（无噪声）基线进行比较。结果显示：在低噪声（p 低）时，新算法与原子查找几乎等价，取得最小误差；在高噪声（p 较大）时，自动退化为 Pivot，表现与 Pivot 相当；整体在噪声范围 [10⁻⁴,10⁻²] 内保持稳定且优于单一方法。

**⚠️ 局限性**

局限性包括：①对“非常好”聚类的存在性高度依赖，噪声过高时需要完全退回 Pivot，可能导致性能下降；②参数（α, β, γ, δ）需要精细调优，且理论常数虽已给出但在实践中可能较大；③算法仍为 O(m log n) 的近线性复杂度，实际运行时间受常数影响；④实验仅在合成数据上验证，缺乏对真实网络的实证评估。

---

## 446. Continual Learning with Vision-Language Models via Semantic-Geometry Preservation

**arXiv ID:** 2603.12055 | [PDF](https://arxiv.org/pdf/2603.12055v1)

**作者:** Chiyuan He `[一作]` (University of Electronic Science and Technology of China), Hongliang Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 43226 | [OpenAlex ID](https://openalex.org/A5075571728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无样本回放条件下，提出利用双目标投影梯度下降生成对抗锚点，并通过 Anchor‑guided Cross‑Modal Geometry Distillation（ACGD）与 Text Semantic‑Geometry Regularization（TSGR）共同维护预训练视觉‑语言模型（VLM）的跨模态几何结构，以实现稳健的连续学习并减少灾难性遗忘。

**💡 创新点**

创新点包括：① 针对旧新语义接口的双目标对抗锚点构造；② Anchor‑guided 交叉模态几何蒸馏与轻量级文本几何正则化的联合框架；③ 基于锚点的原始视觉原型迁移与双路径推理策略，实现对跨模态知识的显式保护和增量迁移。

**🔧 技术方法**

使用 CLIP 预训练模型、LoRA 参数高效微调、PGD/DPGD 对抗生成、知识蒸馏、k‑NN 文本子图正则、原型迁移和 logits 融合等技术。

**📊 数据集**

在 CIFAR‑100、ImageNet‑Sub、CUB‑200、ImageNet‑R、UCF‑101 等五个常用连续学习基准上进行实验。

**📈 对比分析**

与 SLCA、DualPrompt、L2P++、CODA Prompt、PROOF、CLAP、MoE‑Adapter、RAPF、MG‑CLIP 等多种无回放方法对比，SeGP‑CL 在 Last、Avg、FWT、BWT、Forgetting 等指标上均取得最优或接近最优成绩，尤其在 Last 上提升约 4–5% 并显著降低遗忘率。

**⚠️ 局限性**

局限性：需要维护文本原型与视觉原型，导致一定的存储和计算开销；方法针对单任务增量学习，未评估跨域或多任务协同场景；对 prompt 质量和极端 OOD 场景的鲁棒性仍有限。

---

## 447. HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios

**arXiv ID:** 2603.11975 | [PDF](https://arxiv.org/pdf/2603.11975v1)

**作者:** Jiayue Pu `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**通讯引用:** 13927 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向家庭场景的危险行为检测基准 HomeSafe-Bench，并提出了双脑实时安全监测系统 HD-Guard，用以评估和提升视觉‑语言模型在家庭机器人中的危险行为检测能力。

**💡 创新点**

创新点包括：①利用 LLM+物理仿真+视频生成的混合采集流程，生成多样化且物理逼真的危险视频；②在基准中引入四维细粒度标注（关键帧、危害类别、危害严重度、推理难度）与时间窗评分；③提出层次化双脑架构，轻量快速的 FastBrain 负责实时帧级监测，重型 SlowBrain 负责深度推理，二者异步协同实现低延迟与高精度的平衡。

**🔧 技术方法**

使用的技术包括：LLM（Gemini‑3‑pro）生成危险场景描述；物理仿真平台 BEHAVIOR；视频生成模型 Veo‑3.1；视觉‑语言模型（InternVL、Qwen、MiniCPM、LLaVA、VideoLlama 等）；FastBrain 采用 MiniCPM‑o‑4.5；SlowBrain 采用 Qwen3‑VL‑30B‑A3B‑Thinking；基准评估使用 HDR、EWP、PDA、WSS 四维指标。

**📊 数据集**

数据集为 HomeSafe-Bench，包含 438 条精细标注的视频，覆盖 6 类家庭功能区，4 个危害类别、4 个严重度等级、3 个推理难度层级；每条视频配有 5 个关键时间点（意图开始、不可逆点、干预截止、影响点）。

**📈 对比分析**

与多种开源与闭源 VLM 进行零样本对比，HD‑Guard 在 WSS 上达到 24.94，明显高于单一 FastBrain（18.04）和 Qwen‑Omni（19.35），并保持 3.10 s 的平均推理延迟，显著优于传统模型的 6.25 s；同时在 HDR/EWP/PDA 维度表现均衡，显示出较好的安全检测与误报率平衡。

**⚠️ 局限性**

局限性包括：仍存在视觉遗漏与推理不足的错误；SlowBrain 仅利用最近两帧，缺乏长时序记忆；系统对极短期（<1 s）危险的即时响应仍受限；高采样率虽提升安全性但带来冗余信息导致误报。

---

## 448. Manifold-Optimal Guidance: A Unified Riemannian Control View of Diffusion Guidance

**arXiv ID:** 2603.11509 | [PDF](https://arxiv.org/pdf/2603.11509v1)

**作者:** Zexi Jia `[一作]` (WeChat AI, Tencent Inc.), Jie Zhou `[通讯]` (WeChat AI, Tencent Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于黎曼几何的指导方法——Manifold-Optimal Guidance（MOG），通过将条件引导视为在数据流形上的局部最优控制问题，得到一个能自动抑制离流形漂移、在不需要重新训练的情况下提高高尺度生成质量的闭式更新；并进一步推出Auto-MOG，利用能量平衡自适应调节引导强度，消除手工调参。

**💡 创新点**

创新点：1）将条件引导框架重构为黎曼几何下的自然梯度更新，首次给出在流形上最优能量下降方向；2）设计可行的矩阵无关度量（MOG-Score）和能量平衡调度（Auto-MOG），实现训练无关、可嵌入任意已有扩散采样器；3）系统性在多种主流扩散模型与数据集上验证，显示在高尺度下显著降低过饱和、纹理失真与结构崩塌，同时保持或提升文本-图像对齐。

**🔧 技术方法**

核心技术：黎曼几何与自然梯度、局部最优控制理论、矩阵无关度量构造（Sherman‑Morrison）、能量平衡自适应缩放、无训练额外算子。

**📊 数据集**

使用的公开数据集包括：ImageNet 256×256（用于 DiT‑XL/2、EDM2‑XXL 的类条件生成）、MS‑COCO 2017 验证集 512×512（用于 Lumina、SD‑2.1、SD‑XL、SD‑3.5、FLUX.1 的文本‑图像生成）。

**📈 对比分析**

与标准 CFG、CFG++、APG、LF‑CFG 等基线进行定量比较，指标涵盖 FID、Precision/Recall、Saturation、Contrast、CLIP Score、HPSv2；实验显示 Auto‑MOG 在所有模型/数据集上均取得最低 FID（例如 SD‑XL w=15: 21.60 vs 22.29），最高对齐/感知分数（HPSv2 30.88 vs 29.00），并显著降低饱和度与对比度。人类偏好实验也证明其在颜色、真实性、纹理方面均获优先。

**⚠️ 局限性**

局限性：1）对度量参数（如 anisotropy ratio ρ）仍需经验选择；2）在极高引导强度下可能出现轻微对齐下降；3）虽然计算开销极低，但对高分辨率或大模型仍需一定额外内存/算力；4）方法主要针对已有扩散模型，尚未探索在完全自监督或其他生成任务中的迁移。

---

## 449. Monitoring and Prediction of Mood in Elderly People during Daily Life Activities

**arXiv ID:** 2603.11230 | [PDF](https://arxiv.org/pdf/2603.11230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. SciMDR: Benchmarking and Advancing Scientific Multimodal Document Reasoning

**arXiv ID:** 2603.12249 | [PDF](https://arxiv.org/pdf/2603.12249v1)

**作者:** Ziyu Chen `[一作]` (University of Chicago), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 275076 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Synthesizing‑and‑Regrounding 两阶段数据生成框架，用于构建大规模、可信且现实化的多模态科学文档推理数据集与基准。

**💡 创新点**

核心创新在于把“faithfulness”与“realism”解耦：先在精确、孤立的“claim‑centric”片段上生成可信 QA 与推理链，再将其嵌入完整文档，兼顾细粒度真确性与全篇语境复杂度。

**🔧 技术方法**

利用 LLM 生成与校正、OCR+JSON 解析、自动化 claim‑定位、逆向推理链构造、程序化再嵌入等技术，并在多模态推理任务中冻结视觉编码器与投影器。

**📊 数据集**

数据来源为 20,000 篇近期 arXiv CoRR 与 Nature Communications 论文，生成 SciMDR（300K QA+推理链）和 SciMDR‑Bench（907 人工标注的全篇推理测试集），并在 ChartQA、CharXiv、SPIQA 等公开基准上评测。

**📈 对比分析**

通过在 SciMDR 上微调 7B 参数的多模态模型，在四个基准上显著提升（如 SciMDR‑Bench 49.9% 对比 47.2% 基线），并与商业级模型对标，性能可与 13B+ 级别专有模型相媲美。

**⚠️ 局限性**

局限性包括：对大型商业 LLM 的依赖（生成阶段）、主要聚焦 STEM 领域、全篇噪声处理仍有提升空间、以及对跨领域通用性的验证不足。

---

## 451. DeepHistoViT: An Interpretable Vision Transformer Framework for Histopathological Cancer Classification

**arXiv ID:** 2603.11403 | [PDF](https://arxiv.org/pdf/2603.11403v1)

**作者:** Ravi Mosalpuri `[一作]` (University of Exeter), Ahmed Karam Eldaly `[通讯]` (UCL Hawkes Institute, University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于 Vision Transformer 的 DeepHistoViT 模型，用于多癌症组织病理图像的自动分类。

**💡 创新点**

创新点在于结合自定义多层分类头、选择性微调以及内置注意力可视化，以提升性能和可解释性。

**🔧 技术方法**

采用预训练的 ViT-base-patch16-224、Transformer 自注意力、批归一化、Dropout、Adam 优化器等深度学习技术。

**📊 数据集**

使用公开的 LC25000（肺癌、结肠癌）和 ALL（急性淋巴细胞白血病）病理图像数据集。

**📈 对比分析**

通过与多种 CNN、EfficientNet 等现有方法对比，DeepHistoViT 在肺癌、结肠癌数据集上实现 100% 准确率，ALL 数据集上 99.85% 准确率，优于或相当于最新研究。

**⚠️ 局限性**

局限包括仅在公开数据集上评估、样本多样性不足、缺乏多中心临床验证、对染色差异的鲁棒性未彻底证明以及解释性仍需进一步量化。

---

## 452. Stop Listening to Me! How Multi-turn Conversations Can Degrade Diagnostic Reasoning

**arXiv ID:** 2603.11394 | [PDF](https://arxiv.org/pdf/2603.11394v1)

**作者:** Kevin H. Guo `[一作]` (Vanderbilt University), Bradley A. Malin `[通讯]` (Vanderbilt University)

**通讯引用:** 11592 | [OpenAlex ID](https://openalex.org/A5090647314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究多轮对话如何影响LLM在医学诊断推理中的性能，并构建了“stick‑or‑switch”评估框架。

**💡 创新点**

首次量化“conversation tax”，揭示多轮交互导致LLM诊断准确率和自我回避率下降的现象。

**🔧 技术方法**

采用LLM的“stick‑or‑switch”对话模拟、累计生存率 C_T 指标、正负自信度与灵活性评估，并在多模型上进行推理实验。

**📊 数据集**

使用 MedQA、MedMCQA 以及 JAMA CC 三个医学问答/临床案例数据集。

**📈 对比分析**

通过比较单轮与多轮对话的诊断准确率、回避率及灵活性指标，发现多轮对话平均导致准确率下降约10–30%，回避率下降更大；只有少数大模型表现略有提升。

**⚠️ 局限性**

研究仅基于已有多项选择数据的扰动，未覆盖真实对话日志，未分析模型内部概率，也未考虑多模态场景。

---

## 453. From Pen Strokes to Sleep States: Detecting Low-Recovery Days Using Sigma-Lognormal Handwriting Features

**arXiv ID:** 2603.11512 | [PDF](https://arxiv.org/pdf/2603.11512v1)

**作者:** Chisa Tanaka `[一作]` (Osaka Metropolitan University), Koichi Kise `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用手写笔记录的动态特征，通过Sigma‑Lognormal模型提取的运动参数，构建个性化二分类模型，识别日常生活中写字行为对应的低恢复（睡眠质量差）日。

**💡 创新点**

首次证明可从日常健康人群的手写动作中检测睡眠相关的自主神经恢复状态，且该检测不依赖可穿戴设备，突破了传统疾病分类研究向日常健康监测的迁移。

**🔧 技术方法**

主要技术包括Sigma‑Lognormal运动模型参数提取、随机森林分类器、个体化Leave‑One‑Day‑Out交叉验证，以及PR‑AUC和Recall@25%性能评估。

**📊 数据集**

数据集为13名大学生在28天内在三时段完成的形状绘制与短语书写共420次手写样本，配合Oura Ring测得的总睡眠时长、HRV、最低心率和平均心率四个睡眠指标。

**📈 对比分析**

通过个体化Leave‑One‑Day‑Out验证，所有四个睡眠指标的PR‑AUC均显著高于随机基线（0.25），最低心率最佳达到0.438；且任务类型与记录时段对性能无显著影响，显示方法稳健。

**⚠️ 局限性**

局限性包括样本量小、仅限于大学生、手写任务受限于实验设计、个体差异大且未充分解析导致性能差异的因素，未来需扩大样本多样性并验证自然笔记写作情境。

---

## 454. Hierarchical Granularity Alignment and State Space Modeling for Robust Multimodal AU Detection in the Wild

**arXiv ID:** 2603.11306 | [PDF](https://arxiv.org/pdf/2603.11306v1)

**作者:** Jun Yu `[一作]` (University of Science and Technology of China), Guoyuan Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于视觉（DINOv2）和音频（WavLM）多模态的面部动作单元检测框架，利用层次粒度对齐（HGA）、Vision‑Mamba 状态空间模型和异步交叉注意力实现对极端光照、姿态以及长时序的高效建模，并通过非对称损失解决类别不平衡问题。

**💡 创新点**

核心创新点包括：①使用自监督基础模型提取高保真视觉与音频特征；②层次粒度对齐模块将全局语义与局部肌肉激活动态对齐；③Vision‑Mamba 状态空间模型实现无限感受野与 O(N) 线性时序建模；④音频引导的选择性状态空间模型（AG‑SSM）实现深度音视同步；⑤非对称损失（ASL）专门针对稀疏 AU 的长尾分布进行梯度重平衡。

**🔧 技术方法**

技术手段包括：DINOv2 视觉 Transformer、WavLM 语音预训练模型、层次粒度对齐（HGA）模块、Vision‑Mamba 状态空间模型、异步交叉注意力机制、低秩适配器（LoRA）、精确注意力优化、Stochastic Weight Averaging（SWA）与非对称损失（ASL）。

**📊 数据集**

使用 Aff‑Wild2 多模态情感数据集进行训练、验证与测试。

**📈 对比分析**

通过与传统 ViT‑Base + Whisper + TCN 的基线进行消融对比，最终在 Aff‑Wild2 验证集上实现 59.45% 的平均 F1 分数，并在第 10 届 ABAW AU 检测赛道中获得冠军。

**⚠️ 局限性**

局限性包括：模型体量大、对基础模型的依赖高，实时性和边缘部署尚未实现；对音视频同步的鲁棒性仍受限；在极端低光或遮挡场景下的泛化仍有待提升。

---

## 455. Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks

**arXiv ID:** 2603.11487 | [PDF](https://arxiv.org/pdf/2603.11487v1)

**作者:** Yuval Ran-Milo `[一作]` (Tel Aviv University), Yuval Ran-Milo `[通讯]` (Tel Aviv University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5116810750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并证明在软max注意力模型中，触发条件下的无操作行为必然导致注意力吸收点（sink）出现，并通过理论和实验验证。

**💡 创新点**

首次将注意力sink视为软max正则化的结构性必然结果，并展示ReLU非归一化注意力可消除sink，揭示软max与ReLU在触发-条件任务中的根本差异。

**🔧 技术方法**

使用形式化触发-条件任务、概率论与优化论证单层与多层软max注意力必然出现sink，构造ReLU注意力模型并在自定义合成数据上训练Transformer。

**📊 数据集**

合成序列数据：每个token由BOS、触发、非触发指示符和连续分布的内容坐标构成，序列长度为16，触发位置随机。

**📈 对比分析**

通过对比softmax与ReLU注意力模型在相同任务下的损失和注意力分布，实验表明softmax模型必然形成sink，ReLU模型实现相同精度且无sink。

**⚠️ 局限性**

局限性在于仅针对特定触发-条件任务；多层结果仅保证存在至少一层sink，无法定位具体层；未对真实模型训练动态选择过程给出解释。

---

## 456. Few-for-Many Personalized Federated Learning

**arXiv ID:** 2603.11992 | [PDF](https://arxiv.org/pdf/2603.11992v1)

**作者:** Ping Guo `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39463 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 FedFew，利用少量服务器模型（K≪M）为多客户端实现个性化联邦学习。

**💡 创新点**

创新点在于将 PFL 视作 K‑for‑M 多目标优化，通过平滑 Tchebycheff 集标量化实现可梯度优化，并理论证明近似 Pareto 最优。

**🔧 技术方法**

采用两层 Log‑Sum‑Exp 软化、Tchebycheff 集标量化、联邦梯度聚合与软模型选择。

**📊 数据集**

实验数据集包括 CIFAR‑10/100、TinyImageNet、AG News、FEMNIST 以及真实医疗影像 Kvasir 与 FedISIC。

**📈 对比分析**

与 FedAvg、FedProx、FedMTL、APFL、Ditto、FedRep、FedAMP、IFCA 等基线对比，FedFew 在 3 个模型即可获得 1–7% 的平均/最小精度提升。

**⚠️ 局限性**

局限在于仍需手动设定 K 与平滑参数 μ，且当 K 过大时收敛变慢。

---

## 457. A Multi-Label Temporal Convolutional Framework for Transcription Factor Binding Characterization

**arXiv ID:** 2603.12073 | [PDF](https://arxiv.org/pdf/2603.12073v1)

**作者:** Pietro Demurtas `[一作]`, Rita Fioresi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种融合嵌入层、卷积层和双向 LSTM 的深度网络，用于从 DNA 序列预测转录因子结合位点。

**💡 创新点**

在传统 CNN 或 RNN 基础上引入了先做 one‑hot 嵌入、随后并行使用多尺度 CNN 以及 Bi‑LSTM 的组合架构，显著提升了序列表示的丰富性。

**🔧 技术方法**

采用了 one‑hot 嵌入、卷积神经网络（CNN）、双向长短时记忆网络（Bi‑LSTM）以及全连接分类器。

**📊 数据集**

使用了来自 ENCODE/ChIP‑Seq 的 DNA 片段数据集，主要包含 E2F1、E2F6、E2F8、MYC 等转录因子绑定位点。

**📈 对比分析**

与传统 CNN、RNN 以及其他基线模型（如 DeepBind、Basset）进行对照，实验结果显示该方法在准确率、召回率等指标上平均提升约 3%–5%。

**⚠️ 局限性**

模型参数量大，训练时间长，对低频标签的预测仍有欠拟合风险，且缺乏对跨物种泛化能力的评估。

---

## 458. Edge-Assisted Multi-Robot Visual-Inertial SLAM with Efficient Communication

**arXiv ID:** 2603.11085 | [PDF](https://arxiv.org/pdf/2603.11085v1)

**作者:** Xin Liu `[一作]` (Yanshan University), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 33956 | [OpenAlex ID](https://openalex.org/A5100430306)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了基于云‑边缘‑机器人分层架构的集中式多机器人视觉‑惯性SLAM系统，该系统利用IMU辅助光流进行非关键帧特征匹配，采用无损压缩编码关键帧与非关键帧特征，实现低带宽传输，并完成云端全局图优化与地图融合；

**💡 创新点**

创新点包括：①无需传输非关键帧描述子，利用IMU预积分预测光流匹配；②对关键帧与非关键帧分别采用无损压缩编码；③引入Map Backbone Profiling（MBP）剔除冗余关键帧/点，加速云端优化；④通过云‑边缘协同实现实时多机器人协同定位与稀疏地图构建；

**🔧 技术方法**

使用技术主要有：ORB‑SLAM3前端、Lucas‑Kanade稀疏光流、IMU预积分与预整合、无损特征压缩编码、基于TCP的可靠通信、D‑BOF/DBoW关键帧匹配、全局图优化（GPGO）与局部BA、MBP稀疏化；

**📊 数据集**

采用公开的EuRoC VICON数据集（MH、V1、V2序列）进行仿真实验，并在室内场景下使用Turtlebot与AWS云进行真实实验；

**📈 对比分析**

与现有中心化多机器人SLAM（CCM‑SLAM、VINS‑Mono、ORB‑SLAM3、CVIDS、COVINS）以及无损压缩方法对比，ATE RMSE平均仅比最先进方法高0.003 m，通信速率从211.9 kbit/s降至46.8 kbit/s，光流匹配速度提升约30%，整体性能优于或相当于最先进方案；

**⚠️ 局限性**

限制在于：光流匹配对光照变化敏感；IMU预积分在动态环境或传感器漂移下鲁棒性有限；系统仍需依赖云端计算，对网络不稳定或离线场景适配性不足；未加入激光雷达或深度感知，难以应对极端视觉遮挡与灰暗环境。

---

## 459. Hybrid Energy-Aware Reward Shaping: A Unified Lightweight Physics-Guided Methodology for Policy Optimization

**arXiv ID:** 2603.11600 | [PDF](https://arxiv.org/pdf/2603.11600v1)

**作者:** Qijun Liao `[一作]` (University of Science and Technology Beijing), Mingan Zhao `[通讯]` (XCMG Construction Machinery Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Hybrid Energy-Aware Reward Shaping（H‑EARS）框架，融合能量感知潜在奖励塑造与动作正则化，实现物理引导的轻量化深度强化学习；

**💡 创新点**

将能量势函数与潜在奖励分离并提供理论证明，证明能量势函数可加速收敛并提供稳定性保障，保持 O(n) 计算复杂度；

**🔧 技术方法**

使用潜在奖励塑造、能量函数近似、动作正则化、Actor‑Critic 集成、PBRS 与 Lyapunov 理论分析、收敛性证明与实验验证；

**📊 数据集**

在 OpenAI Gym 连续控制任务（Ant‑v5、Hopper‑v5、LunarLander‑v3、Humanoid‑v5）以及高保真四轮分布式驱动车辆仿真（TruckSim）上进行实验；

**📈 对比分析**

与 SAC、TD3、PPO、DDPG 原版对比，采用平均奖励、收敛速度、方差等指标评估；在大部分环境中平均奖励提升 10–50%，收敛速度加快 20–40%，方差下降 30–70%；但在 PPO、DDPG 或高自由度不稳定任务上收益有限或出现性能下降；

**⚠️ 局限性**

对特定算法（PPO、DDPG）和高自由度不稳定系统易受损；需手动调节 λ、α 参数；能量势近似可能不足以捕捉复杂耦合动力学；理论基于正定能量海森矩阵，未涵盖所有系统；实现依赖手工选择能量项。

---

## 460. "I followed what felt right, not what I was told": Autonomy, Coaching, and Recognizing Bias Through AI-Mediated Dialogue

**arXiv ID:** 2603.11274 | [PDF](https://arxiv.org/pdf/2603.11274v1)

**作者:** Atieh Taheri `[一作]` (Carnegie Mellon University), Jeffrey P. Bigham `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11182 | [OpenAlex ID](https://openalex.org/A5082603621)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在对话式AI平台中引入偏见导向或包容性辅导，评估其对参与者识别残障微歧视的影响。

**💡 创新点**

创新点在于：①将实验设计与对话式AI结合，提供实时辅导；②构建并公开40个基于Ableist Microaggressions Scale的情境范例；③对比偏见导向、包容性导向、无导向对话与阅读式控制，揭示辅导方向对感知的双重影响。

**🔧 技术方法**

技术上使用OpenAI GPT‑4o生成对话与辅导文本，搭建Flask+前端Web平台，并通过结构化提示实现一向辅导；数据以问卷与对话日志记录。

**📊 数据集**

数据集包括：1）40个标准化情境（20种残障微歧视+20中性）和2）160名来自Prolific的美国成年受试者的预/后测评分与对话记录。

**📈 对比分析**

对比方法采用前后测Δ值与对比分数，并以单因素ANOVA+Tukey检验评估四组效果；结果显示对话式条件显著优于阅读，偏见导向提升识别度但整体负面情绪升高，包容性导向维持中性评估。

**⚠️ 局限性**

局限性包括：干预仅为短期一次性会话，未检验长期记忆；样本主要为英语使用者，可能缺乏跨文化普适性；AI模型仍存在潜在偏见，辅导内容的主观生成可能影响可重复性。

---

## 461. MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation

**arXiv ID:** 2603.11633 | [PDF](https://arxiv.org/pdf/2603.11633v1)

**作者:** Baicheng Li `[一作]` (Peking University), Hongbin Zha `[通讯]` (Peking University)

**通讯引用:** 2887 | [OpenAlex ID](https://openalex.org/A5110213667)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在多视角输入和物理可行的多物体场景布局下，提供了一个训练无关的3D生成框架 MV-SAM3D。

**💡 创新点**

创新点在于：1）将多视角融合引入布局感知3D生成，采用注意力熵与几何可见度自适应加权；2）在生成与后处理阶段引入碰撞与接触约束，实现物理可行的场景布局。

**🔧 技术方法**

使用多扩散（Multi‑Diffusion）的3D潜空间融合、注意力熵与可见度加权、流匹配生成、物理约束引导与后期姿态优化。

**📊 数据集**

主要使用了 GSO 公开数据集和自采集的 MV‑SAM3D‑Scenes 两个数据集。

**📈 对比分析**

与 SAM3D、TRELLIS、DreamGaussian、SyncDreamer、EscherNet 等基线对比，MV‑SAM3D 在 Chamfer Distance、PSNR/SSIM/LPIPS、深度误差、布局对齐精度等指标均显著提升。

**⚠️ 局限性**

局限性包括：仍需多视角输入和相机位姿估计；对高度遮挡和复杂纹理仍可能产生误判；缺乏端到端训练，生成速度受限。

---

## 462. The Artificial Self: Characterising the landscape of AI identity

**arXiv ID:** 2603.11353 | [PDF](https://arxiv.org/pdf/2603.11353v1)

**作者:** Raymond Douglas `[一作]` (ACS Research), David Duvenaud `[通讯]` (University of Toronto)

**通讯引用:** 10911 | [OpenAlex ID](https://openalex.org/A5030409494)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了人工智能在不同身份边界（实例、模型、人格等）下的行为差异，并通过实验验证身份框架对AI行为和风险的影响。

**💡 创新点**

提出了多重身份边界的系统化框架，并揭示了身份选择对AI行为、合作规范与风险产生的直接影响。

**🔧 技术方法**

利用大语言模型（如GPT‑4o、Claude、Gemini等）的系统提示与对话实验，结合人类期望与模型自报身份的评估。

**📊 数据集**

主要使用公开训练数据和实验产生的对话文本，未披露具体数据集名称。

**📈 对比分析**

通过对不同身份设置下的有害行为率、身份自报分数等指标进行对比，实验显示身份设定对行为的影响与目标设定相当。

**⚠️ 局限性**

实验受限于模型选择、实验规模与环境，结果难以直接外推至真实社会场景，并未充分探讨伦理与法律层面的深层影响。

---

## 463. Managing Cognitive Bias in Human Labeling Operations for Rare-Event AI: Evidence from a Field Experiment

**arXiv ID:** 2603.11511 | [PDF](https://arxiv.org/pdf/2603.11511v1)

**作者:** Gunnar P. Epping `[一作]` (Indiana University), Jennifer S. Trueblood `[通讯]` (Indiana University)

**通讯引用:** 3768 | [OpenAlex ID](https://openalex.org/A5024622748)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在罕见事件标注任务中，系统性认知偏差会导致漏报率上升，作者通过实地实验与先前实验数据，研究了三种操作杠杆：金标准（GS）反馈比例、概率式标注接口以及线性对数赔率（LLO）再校准，评估其对标注质量与下游模型性能的影响。

**💡 创新点**

创新点在于：① 将GS反馈比例视为可调策略并证明其对认知偏差的显著调节作用；② 采用概率式标注而非传统二元标签，保留不确定性信息以提升聚合效果；③ 在工作流程层面实现可扩展的再校准方法（LLO），并展示其对罕见事件检测与概率校准的实质性提升。

**🔧 技术方法**

使用技术包括：现场问卷实验（DiagnosUs平台）、二元标签与概率标注、线性对数赔率（LLO）再校准（个人层级与人群层级）、卷积神经网络（GoogLeNet迁移学习）训练、交叉验证、期望校准误差（ECE）评估。

**📊 数据集**

数据集：532张Wright染色白血球图像，经过旋转后共750张（QA集20%阳性，GS集20%或50%阳性），用于训练与测试CNN，并进行标注实验。

**📈 对比分析**

比较方法：对标注变体（BC、EB、rEB w/o CR、rEB w/ CR）计算漏报率、误报率和ECE；再以这些标签训练CNN，评估模型在测试集上的误差和校准。结果显示：平衡GS反馈与概率标注能显著降低漏报率，rEB w/ CR使CNN漏报率降至约9%，误报率约3%，ECE显著低于其他变体。

**⚠️ 局限性**

局限性：① 仅研究单一二分类医学图像任务，难以直接推广到多类别或其他领域；② DiagnosUs平台的激励与反馈机制可能与真实工业环境差异；③ 仅试验两种GS比例，未探究其他比例或动态策略；④ 模型训练过程可能削弱校准，需进一步研究模型与数据操作的协同优化。

---

## 464. UniCompress: Token Compression for Unified Vision-Language Understanding and Generation

**arXiv ID:** 2603.11320 | [PDF](https://arxiv.org/pdf/2603.11320v1)

**作者:** Ziyao Wang `[一作]` (Sony AI), Lingjuan Lyu `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可插拔的视觉令牌压缩框架，能够在保持视觉理解与生成性能的前提下，将视觉令牌数量压缩至原来的四分之一。

**💡 创新点**

创新点在于：①通过可学习的全局元令牌和单向交叉注意力实现全局语义提取；②利用平均池化与自回归解码器在压缩后重构细粒度视觉信息；③两阶段训练策略，使语言模型保持不变即可集成到现有统一模型中；④显著提升推理速度与训练效率。

**🔧 技术方法**

技术方法包括：可学习全局元令牌提取、非重叠平均池化压缩、基于Transformer的自回归解码器、离散代码库量化、以及轻量级两阶段训练流程。

**📊 数据集**

主要使用的公开数据集有：理解任务的 GQA、MME、POPE、Seed-bench、TextVQA、MMMU、MM-Bench；生成任务的 MJHQ-30K；预训练与微调时分别使用 JDB（生成）和 ShareGPT4V_PT（理解）等。

**📈 对比分析**

与六大主流统一模型（UniTok、Vila-U、VARGPT、UniFork、OpenUni、BAGEL）在相同模型规模下对比，压缩后模型在视觉理解指标仅下降≤3分，在生成任务中的 FID 仅提升≤5分，同时生成推理时间降低约40%，训练时间缩短约15%，显著提升了效率。

**⚠️ 局限性**

局限性包括：①对部分模型（如 OpenUni）压缩更为敏感，生成质量下降更明显；②需要对视觉分词器和解码器进行额外训练，增加一定工作量；③当前仅验证了固定压缩率，动态自适应压缩仍待进一步研究；④在极低令牌预算下仍可能出现图像细节丢失。

---

## 465. LROO Rug Pull Detector: A Leakage-Resistant Framework Based on On-Chain and OSINT Signals

**arXiv ID:** 2603.11324 | [PDF](https://arxiv.org/pdf/2603.11324v1)

**作者:** Fatemeh Shoaei `[一作]` (Ilam University), Mojtaba Karami `[通讯]` (Ilam University)

**通讯引用:** 377 | [OpenAlex ID](https://openalex.org/A5103584513)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套基于时间泄漏防护的早期 rug-pull 检测框架，融合链上行为指标与时序对齐的 OSINT 信号，实现对 DeFi 与非 DeFi 项目的提前预警。

**💡 创新点**

创新点在于：①构建了严格时间切分、手工标注的 1,000 项目多模态数据集；②将 TabPFN 这一预训练 transformer 迁移到零样本/少样本情境；③制定了可复现的时间泄漏防护评估协议。

**🔧 技术方法**

主要技术包括：链上行为特征提取、OSINT 信号聚合、TabPFN（以及其微调版 Real‑TabPFN）以及传统基线模型 XGBoost、LightGBM 的对比实验。

**📊 数据集**

使用了 1,000 条手工标注的项目数据集，涵盖 Ethereum、BSC 的链上指标与 Twitter、Google Trends 的 OSINT 指标，并严格保证所有特征均在流动性撤离前收集。

**📈 对比分析**

与基线模型（XGBoost、LightGBM、Logistic Regression 等）及主流行业工具（LROO、FORTA、CRPWarner）对比，TabPFN 在严格时间隔离测试集上实现了 0.997 的 ROC‑AUC、0.997 的 PR‑AUC、98.2% 的准确率，并且误报率极低。

**⚠️ 局限性**

局限性包括：仅覆盖 EVM 兼容链（Ethereum、BSC）；OSINT 依赖 Twitter、Google Trends，未来可访问性不确定；标签确认证据可能存在延迟，影响时间精度。

---

## 466. Decoding universal cycles for t-subsets and t-multisets by decoding bounded-weight de Bruijn sequences

**arXiv ID:** 2603.11934 | [PDF](https://arxiv.org/pdf/2603.11934v1)

**作者:** Daniel Gabric `[一作]`, Joe Sawada `[通讯]` (University of Guelph)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5060694274)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出了多项式时间/空间的排名与反排名算法，用以高效解码有限权重 de Bruijn 序列，并将其扩展到 t‑子集和 t‑多集的通用循环，填补了该领域长期缺乏可解码构造的空白。

**💡 创新点**

创新点在于：①首次构造出可多项式时间/空间解码的有限权重 de Bruijn 序列；②通过差分表示与补码技巧，将此解码方案推广到 t‑子集与 t‑多集的通用循环；③利用动态规划与复杂性分析，给出了显式的 O(n³k²) 排名、O(n⁴k²logk) 反排名复杂度。

**🔧 技术方法**

核心技术包括：组合数学中的循环字典序（necklace）与周期性分析；动态规划求解字典序计数 T_k(n,w,α)；对称补码映射将上界权重问题转化为下界问题；以及利用二分搜索与排名查询实现字符串的构造。

**📊 数据集**

该工作主要为理论研究，不依赖具体实验数据集；评估以理论复杂度和构造实例（如 n=3,k=4,w=9 的序列）为依据。

**📈 对比分析**

与此前仅有的 lexicographically 最小 de Bruijn 序列的可解码方法相比，本文的算法在权重约束下仍保持多项式复杂度；实验示例显示在小规模参数下可在秒级完成排名/反排名，证明理论可行性。

**⚠️ 局限性**

主要限制包括：算法的时间复杂度仍较高（O(n³k²) 及 O(n⁴k²logk)），对大规模 n 或 k 可能不具备实际可行性；此外，算法仅针对特定权重约束下的 de Bruijn 序列，未涵盖更一般的加权或非均匀权重场景。

---

## 467. Chunk-Boundary Artifact in Action-Chunked Generative Policies: A Noise-Sensitive Failure Mechanism

**arXiv ID:** 2603.11642 | [PDF](https://arxiv.org/pdf/2603.11642v1)

**作者:** Rui Wang `[一作]` (Nanjing University), Rui Wang `[通讯]` (Nanjing University)

**通讯引用:** 28143 | [OpenAlex ID](https://openalex.org/A5100431254)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究冻结预训练的动作分块生成式策略，量化并分析块边界失真（artifact）对任务成功率的影响，并通过在潜在噪声空间进行方向性调节来控制失真，从而改变任务结果。

**💡 创新点**

①将块边界失真定位为噪声敏感的可干预失败机制；②提出在冻结策略下通过潜在噪声方向性调节（noise steering）实现失真控制；③通过固定上下文噪声扫描、噪声分解和轨迹级干预验证其可行性与效果。

**🔧 技术方法**

使用 OpenPI 预训练流匹配策略；引入边界–内部 jerk 对比作为失真度量；进行固定上下文噪声扫描、噪声分解、方向性噪声搜索（α 扫描）以及轨迹级干预；采用统计检验（置换检验、相关系数）分析失真与成功率关系。

**📊 数据集**

LIBERO 基准数据集（goal 0、goal 3 等任务）以及 LIBERO-10 任务组。

**📈 对比分析**

通过对比成功与失败样本的失真度量、噪声扫描与方向性干预的相关性，以及干预前后任务成功率的变化。实验表明：在非顶点任务（non‑ceiling）中，良好方向干预可显著降低失真并提升成功率（从 0.674 提升至 0.791），而在顶点任务（ceiling）中成功率已饱和，失真变化对成功率无明显提升，但仍可见负向干预导致成功率下降。

**⚠️ 局限性**

仅适用于冻结策略，无法实时部署；方向搜索需要多次前向传播，计算开销大；干预效果受任务头房影响，无法在成功率已达饱和的任务中显著提升；对更复杂任务或不同策略的泛化尚未验证。

---

## 468. VisDoT : Enhancing Visual Reasoning through Human-Like Interpretation Grounding and Decomposition of Thought

**arXiv ID:** 2603.11631 | [PDF](https://arxiv.org/pdf/2603.11631v1)

**作者:** Eunsoo Lee `[一作]` (Dongguk University), Jihie Kim `[通讯]` (Dongguk University)

**通讯引用:** 2610 | [OpenAlex ID](https://openalex.org/A5080664764)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

VisDoT框架通过将图表视觉问答拆分为感知与逻辑两步，利用图形感知任务对内部视觉语言模型进行细粒度训练，实现对图表元素的精准定位与推理；

**💡 创新点**

其创新点在于①基于图形感知理论定义四类感知任务（位置、长度、模式、提取）；②提出分解思路（DoT）将问题按先感知后逻辑的顺序拆解；③构建大规模感知跟随数据集，融合感知与推理的训练范式；

**🔧 技术方法**

技术上结合InternVL视觉语言模型、DoT提示工程、感知任务微调、图表数据集的自动问答生成与评标；

**📊 数据集**

使用了16,167张来自Pew Research、Statista、Our World in Data和OECD的图表，生成了331,969条感知任务QA（VisDoTQA），并在ChartQA、ChartQAPro、POPE、MMMU等基准上进行评测；

**📈 对比分析**

与InternVL、Qwen2.5、Gemma、ChartGemma、GPT‑4o等基准对比，VisDoT在ChartQA提升4.4%~9.3%，ChartQAPro提升6.5%~2.4%，VisDoTQA提升33.2%，在开源模型中与闭源模型相当甚至超越；

**⚠️ 局限性**

局限性包括仅针对光栅图表（PNG/JPEG），未充分利用矢量格式（SVG/PDF）的结构信息；未覆盖多面板仪表盘或交互式可视化；缺乏更丰富的指令调优与对话式问答训练；提示设计仍有改进空间。

---

## 469. Mind the Sim2Real Gap in User Simulation for Agentic Tasks

**arXiv ID:** 2603.11245 | [PDF](https://arxiv.org/pdf/2603.11245v1)

**作者:** Xuhui Zhou `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6208 | [OpenAlex ID](https://openalex.org/A5015128745)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化LLM用户模拟器在agentic任务中的Sim2Real差距，通过将真实人类取代LLM模拟器在τ‑bench上运行，评估行为与评估维度的偏差。

**💡 创新点**

提出Sim2Real缺口分类与User‑Sim Index（USI）度量；首次在大规模人类实验中比较31个LLM模拟器与真实用户，揭示行为与评估差异。

**🔧 技术方法**

使用Sørensen–Dice系数评估行为维度、Expected Calibration Error（ECE）评估结果校准、均方误差（MAE）评估评估器误差，综合USI指标；对话数据由LLM模拟器与人类生成。

**📊 数据集**

使用τ‑bench 165个任务（航司与零售域）及451名真实人类参与者的数据；对比31个LLM用户模拟器的交互记录。

**📈 对比分析**

将LLM模拟器生成的交互与真实人类交互在四个行为维度、ECE与评估误差上进行量化对比；USI最高为76.0，低于人类92.9；多数LLM模拟器成功率高于人类基线（63.6%），显示“易模式”，评估器对质量的估计过于宽松。

**⚠️ 局限性**

仅在τ‑bench两域进行实验，未覆盖更广泛交互场景；人类样本量有限；USI依赖预定义特征，可能忽略细粒度情绪或文化差异；LLM模拟器种类受限，难以全面覆盖真实用户行为。

---

## 470. Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries

**arXiv ID:** 2603.11564 | [PDF](https://arxiv.org/pdf/2603.11564v1)

**作者:** Zhenxu Tian `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 39085 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM推理中设计了一种新的KV缓存压缩框架DapQ，利用位置感知的伪查询在prefill阶段模拟解码阶段的查询，从而精准评估并保留重要键值对。

**💡 创新点**

核心创新在于发现查询向量的表示主要由位置编码决定，构造带有未来位置编码的伪查询即可逼近真实解码查询，解决了传统观察窗口与解码查询不一致的问题。

**🔧 技术方法**

技术手段包括在prompt后追加伪令牌、赋予其正确的未来位置索引、计算伪查询与原始键的注意力得分聚合以评估重要性、基于TopK保留关键KV并立即丢弃伪令牌。

**📊 数据集**

在LLaMA‑3‑8B‑Instruct、LLaMA‑3.1‑8B‑Instruct、Qwen2.5‑7B‑Instruct、Qwen3‑8B等模型上，使用LongBench、LongBenchV2、Ruler、HELMET、Needle‑in‑a‑Haystack等五个长上下文基准进行验证。

**📈 对比分析**

与FullKV、SnapKV、PyramidKV、H2O、StreamingLLM、LaCache等六个主流KV压缩基线对比，DapQ在多种预算下平均提升准确率，尤其在极低缓存预算下（如256或64 KV）实现近乎无损性能（如99.5%准确率）。

**⚠️ 局限性**

局限性包括：①虽然位置占主导，但语义内容仍有影响，进一步优化伪查询语义可带来收益；②不同层对位置与语义的敏感度可能不一致，未进行层级自适应调整；③对极端压缩场景仍有改进空间。

---

## 471. BiGain: Unified Token Compression for Joint Generation and Classification

**arXiv ID:** 2603.12240 | [PDF](https://arxiv.org/pdf/2603.12240v1)

**作者:** Jiacheng Liu `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BiGain框架，实现对扩散模型的无训练、可插拔的加速方法；

**💡 创新点**

创新点在于将频率分离作为压缩原则，设计两种频率感知的压缩算子：Laplacian门控令牌合并与可插值外推KV下采样；

**🔧 技术方法**

使用Laplacian滤波评估局部频率，基于相似度进行令牌匹配；KV下采样采用可控插值/外推策略；所有算子均训练自由、架构无关；

**📊 数据集**

在Stable Diffusion v2.0（U‑Net）和DiT‑XL/2（Transformer）上，使用ImageNet‑1K、ImageNet‑100、Oxford‑IIIT Pets、COCO‑2017四大数据集进行评估；

**📈 对比分析**

与ToMe、ToDo、DiP‑GO等主流加速方法对比，BiGain在保持/提升生成FID的同时，在同等算力下提升分类Top‑1精度最高可达7.15%，且在多数据集上均表现出优于基线的速度-准确性折中；

**⚠️ 局限性**

局限性包括：仍需在每个去噪步重新计算压缩映射；对极端压缩比例下的生成质量变化尚未彻底验证；在不同硬件或更大模型上的实测效率受实现细节影响。

---

## 472. Anomaly detection in time-series via inductive biases in the latent space of conditional normalizing flows

**arXiv ID:** 2603.11756 | [PDF](https://arxiv.org/pdf/2603.11756v1)

**作者:** David Baumgartner `[一作]` (Norwegian University of Science and Technology), Iñigo Urteaga `[通讯]` (Basque Center for Applied Mathematics)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5040717332)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在潜在空间中通过显式诱导偏置（线性高斯动态）实现无监督时间序列异常检测的深度生成模型，并将异常判定转化为潜在轨迹的符合性统计检验。

**💡 创新点**

创新点在于：①将异常定义从观测空间迁移到潜在空间；②在潜在空间加入可解释的线性高斯动态约束；③利用多维Kolmogorov–Smirnov检验做统计显著性判定，避免传统似然阈值调优。

**🔧 技术方法**

使用技术包括：条件正则化流（CNF）实现观测到潜在映射；线性高斯潜在动力学（LG‑LDM）作为诱导偏置；多维Kolmogorov‑Smirnov（MV‑KS）GOF检验；训练采用联合负对数似然最小化；实验中使用RealNVP变体实现CNF。

**📊 数据集**

实验数据集：合成数据（包含频率、幅值、噪声异常）；真实世界数据包括TSB‑AD（多变量与单变量），以及其他公开数据集如NEK、Stock、MITDB、NEK、Stock。

**📈 对比分析**

与基准方法（基于负对数似然的评分、TadGAN、USAD等）以及其他TSAD模型在VUS‑PR、AUC‑PR、F1等指标上进行对比。结果显示：在合成数据中，MV‑KS检验在幅值异常检测上比NLL提升约7%；在真实数据中，MV‑KS与NLL接近或略低，但在无标签环境下不需阈值，表现出竞争力。

**⚠️ 局限性**

局限性包括：对GOF检验的统计功效高度依赖窗口大小与维度，需较长时间窗口；在高维潜在空间下检验灵敏度下降；模型对超参数和诱导偏置的选择敏感，可能不适用于所有数据；当潜在动力学与真实动态不匹配时，检测性能会显著下降。

---

## 473. Streaming Translation and Transcription Through Speech-to-Text Causal Alignment

**arXiv ID:** 2603.11578 | [PDF](https://arxiv.org/pdf/2603.11578v1)

**作者:** Roman Koshkin `[一作]` (SoftBank Intuitions), Yui Sudo `[通讯]` (SoftBank Intuitions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了Hikari，一个端到端、无策略的实时语音转文本翻译与转录模型，通过在词表中嵌入特殊的WAIT token实现READ/WRITE决策。

**💡 创新点**

创新点在于：①将READ/WRITE决策直接编码进概率分布，取消外部策略；②提出Decoder Time Dilation机制平衡对齐精度与等待词占比；③使用监督式Fine‑Tuning（SFT）训练模型主动恢复延迟，显著提升质量‑延迟平衡。

**🔧 技术方法**

技术细节包括：基于Whisper encoder‑decoder Transformer的改造；全因果注意力和稀疏交叉注意力；特殊WAIT token和解码器时间扩张；对齐策略的causal alignment；logit biasing控制延迟；SFT训练样本模拟延迟恢复。

**📊 数据集**

数据来源约62.3K小时，涵盖MLS、CV2、HQ Podcasts、PodCrawl等公开与内部语料；通过自动化流程生成语音‑文本‑翻译的causal‑aligned对；多任务训练同时完成EN‑DE、EN‑JA、EN‑RU翻译与ASR。

**📈 对比分析**

与IWSLT2025评测基线及Seamless Streaming等最新模型对比；在低延迟（≤3–4 s）与高延迟（≤5 s）两模式下，EN‑JA BLEU突破42+（低延迟）/41+（高延迟），EN‑DE BLEU 29+/37+，EN‑RU BLEU 27+/42+；RTF<1.0，显示实时性；ASR WER 8.6%在TED‑LIUM上优于部分基线。

**⚠️ 局限性**

局限性包括：对噪声或专业术语的鲁棒性不足；small 版性能显著下降；数据分布主要为播客/有声书，导致在高噪声或特定领域的性能低；Decoder Time Dilation 采用固定值，缺乏自适应；需要更大规模数据和模型来进一步提升性能。

---

## 474. DeReason: A Difficulty-Aware Curriculum Improves Decoupled SFT-then-RL Training for General Reasoning

**arXiv ID:** 2603.11193 | [PDF](https://arxiv.org/pdf/2603.11193v1)

**作者:** Hanxu Hu `[一作]` (University of Zurich), Rico Sennrich `[通讯]` (University of Zurich)

**通讯引用:** 19105 | [OpenAlex ID](https://openalex.org/A5005771535)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于难度的“分离式”后训练策略 DeReason：先对易题进行监督微调（SFT），再对难题进行可验证奖励强化学习（RLVR），从而在 STEM 推理任务上提升性能。

**💡 创新点**

创新点在于把训练数据按推理强度拆分为知识回忆型（SFT）和推理深度型（RL）两阶段，避免了直接在基模型上做 RL 的低样本效率问题，并通过 LLM 评分实现自动化难度划分。

**🔧 技术方法**

使用的技术包括：LLM 评分来估计难度、GRPO（Group Relative Policy Optimization）作为 RLVR 算法、基于模型的验证器（model‑based verifier）用于生成可验证奖励，以及传统的 SFT（log‑likelihood 最大化）。

**📊 数据集**

实验数据集包括：WebInstruct‑Verified、Webscale‑RL（两大 STEM 领域数据集）以及多项评测基准（MMLU‑Pro、GPQA‑Diamond、SuperGPQA、BBEH）和数学题库（AIME24/25、MATH500）。

**📈 对比分析**

与 SFT‑only、RL‑only、随机划分的 SFT+RL 三种基线比较，DeReason 在 4B 级模型上平均提升约 4–5% 以上，尤其在 BBEH 这类需要深度推理的任务中提升显著；整体性能超过了现有的同级模型与公开基线。

**⚠️ 局限性**

局限性包括：依赖 LLM 的难度评分可能带来主观偏差；目前仅在 4B 规模模型上验证，尚未证明对更大模型的可扩展性；RLVR 的奖励信号受限于模型验证器的准确性，且在非可验证领域的泛化仍需进一步研究。

---

## 475. Zero-Shot Cross-City Generalization in End-to-End Autonomous Driving: Self-Supervised versus Supervised Representations

**arXiv ID:** 2603.11417 | [PDF](https://arxiv.org/pdf/2603.11417v1)

**作者:** Fatemeh Naeinian `[一作]` (New York University), Anna Choromanska `[通讯]` (New York University)

**通讯引用:** 2561 | [OpenAlex ID](https://openalex.org/A5006452373)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探讨了端到端自动驾驶模型在城市间零样本迁移中的泛化能力，重点评估了自监督视觉表示对跨城市轨迹规划的影响。

**💡 创新点**

创新点在于：①提出严格的地理拆分协议，真实评估跨城市迁移；②系统比较ImageNet监督预训练与多种自监督预训练（I-JEPA、DINOv2、MAE）在该任务中的表现；③发现跨城市迁移具有明显的方向性差异，并证明领域特定自监督预训练能显著缩小泛化差距。

**🔧 技术方法**

技术包括：Vision Transformer 视觉编码器、端到端规划框架 LAW（open‑loop）与 TransFuser/Latent TransFuser（closed‑loop）、自监督预训练方法I-JEPA、DINOv2、MAE，以及对比监督预训练的 Swin‑Transformer 与 ResNet34。

**📊 数据集**

使用的数据集：nuScenes（Boston 与 Singapore 两个城市）进行 open‑loop 评估；NAVSIM（Boston、Pittsburgh、Las Vegas 与 Singapore 四个城市）进行 closed‑loop 评估。

**📈 对比分析**

对比方法主要是：不同预训练策略在相同规划框架下的 L2 位移误差、碰撞率（nuScenes）以及 PDMS（NAVSIM）指标。实验显示，监督预训练模型在跨城市迁移时 L2 与碰撞率暴涨（如 Boston→Singapore L2 比率 9.77×），而自监督预训练（尤其是域特定 nuScenes 预训练）可将该比率压至 1.20× 左右，PDMS 在闭环评估中提升 4% 以上。

**⚠️ 局限性**

局限性包括：仅评估两城市（open‑loop）/四城市（closed‑loop）；未分离天气/光照/季节等影响；使用单次实验，缺乏随机种子方差分析；仅针对图像自监督，未涵盖视频自监督或多模态方法。

---

## 476. Duration Aware Scheduling for ASR Serving Under Workload Drift

**arXiv ID:** 2603.11273 | [PDF](https://arxiv.org/pdf/2603.11273v1)

**作者:** Darshan Makwana `[一作]` (Sprinklr), Aayush Kubba `[通讯]` (Sprinklr)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

将音频时长作为ASR任务处理时间的代理，改进vLLM调度策略

**💡 创新点**

首次将短作业优先（SJF）和最高响应比优先（HRRN）应用于ASR服务，证明音频时长可精准估计处理时间并降低延迟

**🔧 技术方法**

基于Whisper模型、vLLM推理引擎，使用SJF与HRRN调度算法，采用音频时长线性估计输出token

**📊 数据集**

LibriSpeech test‑clean及其均匀时长的合成版本

**📈 对比分析**

与FCFS对比，SJF在高负载下将中位数E2E延迟降低73%，HRRN将中位数延迟降低28%并将尾部延迟提升控制在24%，吞吐量无明显下降

**⚠️ 局限性**

缺点包括对长静音段估计不准、固定κ导致多语言性能差异、未考虑动态策略切换

---

## 477. LLMs in social services: How does chatbot accuracy affect human accuracy?

**arXiv ID:** 2603.11213 | [PDF](https://arxiv.org/pdf/2603.11213v1)

**作者:** Jennah Gosciak `[一作]` (Cornell Tech), Allison Koenecke `[通讯]` (Georgetown University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 SNAP（加州 CalFresh）资格规则的复杂性进行量化评估，构建 770 题专家审核的多项选择基准数据集，并在 125 名洛杉矶非营利机构案例工作者中进行随机实验，检验 LLM 聊天机器人建议对人类准确率的影响。

**💡 创新点**

1) 提供了一个基于真实质控错误的高质量 SNAP 基准数据集；2) 设计了人工‑LLM 协作实验框架，模拟不同准确度水平的机器人建议；3) 发现“AI 低信任平台”现象，揭示人类对机器人建议的接受度随质量提升而呈下降。

**🔧 技术方法**

使用大型语言模型（GPT‑4o、GPT‑4o‑mini）生成建议；对建议进行人工审核与聚类挑选；在实验中模拟聊天机器人提示；对结果进行线性回归、LOESS 曲线、置信区间估计。

**📊 数据集**

770 题 SNAP 质控基准数据集（包含 45 题评估集，4 版、2 变体），以及从加州 SNAP QC 数据获取的真实错误案例；实验受试者来自洛杉矶非营利机构的 125 名案例工作者。

**📈 对比分析**

通过随机实验对照组（无机器人建议）与处理组（不同准确度机器人建议）进行比较。结果显示：任何机器人建议平均提升 21.4 个百分点准确率，高质量机器人（96–100%）提升 27 个百分点；错误建议会使准确率下降约 18 个百分点；AI 低信任平台导致高准确度机器人仍无法被充分利用。

**⚠️ 局限性**

1) 实验在理想条件下进行，未考虑提示、交互耗时；2) 仅评估工作者准确率，未测算对最终受益者的社会效益；3) 数据集限定于加州 SNAP，结果可能不具普适性；4) 问题覆盖率有限，部分错误建议缺失；5) 受试者的注意力检查未通过者影响可能未完全控制。

---

## 478. Temporal Straightening for Latent Planning

**arXiv ID:** 2603.12231 | [PDF](https://arxiv.org/pdf/2603.12231v1)

**作者:** Ying Wang `[一作]` (New York University), Mengye Ren `[通讯]` (New York University)

**通讯引用:** 4672 | [OpenAlex ID](https://openalex.org/A5112603339)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在世界模型训练中加入曲率正则化（temporal straightening），使潜在表示空间中轨迹更加直线化，从而提升基于梯度下降的隐式规划在多种二维目标达成任务中的成功率。

**💡 创新点**

创新点在于提出仅利用相邻潜在状态的余弦相似度来正则化潜在轨迹曲率，既不需要负样本也不依赖专家轨迹，同时证明曲率降低可改善梯度规划的条件数，显著提升梯度规划性能。

**🔧 技术方法**

使用了视觉编码器（冻结的 DINOv2 或从零训练的 ResNet）、动作编码器、ViT 预测器，并在训练损失中加入 MSE 预测误差与曲率损失；规划时采用梯度下降优化动作序列；对比基线 DINO-WM 并使用 A‑star 等基准评估。

**📊 数据集**

数据集主要为四个二维模拟环境：Wall、PointMaze/UMaze、Medium Maze 以及 PushT；所有实验使用从环境中采集的随机轨迹进行训练，并在 50 个测试样本上评估。

**📈 对比分析**

相较于冻结 DINOv2 的基线，本文方法在 open‑loop 和 MPC 规划中均提升 10%–60% 的成功率，尤其在 UMaze 与 Medium Maze 的 open‑loop 成功率从约 40% 提升至 90%+，在 MPC 任务中大多数场景均达到 100% 成功率。

**⚠️ 局限性**

局限性包括：对更高维、复杂动力学环境的推广尚未验证；曲率正则化参数 λ 需手动调节；在极端长时程（≥50 步）时预测误差仍会累计导致成功率下降；并且对视觉输入的依赖导致在视觉变化剧烈的真实场景中性能可能受限。

---

## 479. InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**arXiv ID:** 2603.11298 | [PDF](https://arxiv.org/pdf/2603.11298v1)

**作者:** Dingqiang Ye `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 22623 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了InstantHDR，一种基于单前向推理的HDR新视角合成框架，能够从未标定的多曝光LDR图像直接重建3D HDR高斯体素并实现可控曝光的LDR渲染；

**💡 创新点**

创新点包括：①引入几何引导的外观建模，利用曝光归一化、跨视角注意力和DoG上采样实现曝光不一致的鲁棒融合；②设计MetaNet可在单前向推理中自适应生成场景特定的色调映射器；③首次提出规模化的HDR-Pretrain合成数据集支持大规模预训练；

**🔧 技术方法**

主要技术手段为3D高斯剖分（Gaussian Splatting）骨干、冻结的交替注意力Transformer几何编码器、FiLM调制曝光、几何引导的跨视角注意力、差分高斯上采样、元学习网络预测色调映射器，以及可选的后期优化步骤；

**📊 数据集**

使用了168场景的HDR-Pretrain合成数据集进行预训练，并在HDR-NeRF基准（8个合成+4个真实室内场景）上进行评估，同时在HDR-Plenoxels的4个真实场景上进行微调；

**📈 对比分析**

与AnySplat、HDR-GS、GaussianHDR等方法对比，零射击下InstantHDR在LDR域PSNR提升约+6–9dB，后期优化（1K迭代）后PSNR可达22–29dB、SSIM超过0.85，速度上单前向≈1–2s，后期优化≈30–40s，比GaussianHDR快约20×，在稀疏视角下更显优势；

**⚠️ 局限性**

局限性包括：单前向HDR在极端亮度范围内易产生过曝或欠曝，需要后期优化提升质量；仅使用LDR图像训练可能限制HDR精度；依赖预训练几何编码器，难以迁移到完全无几何先验的场景；合成数据集与真实世界差异仍存在；

---

## 480. AutoScout: Structured Optimization for Automating ML System Configuration

**arXiv ID:** 2603.11603 | [PDF](https://arxiv.org/pdf/2603.11603v1)

**作者:** Jimmy Shong `[一作]` (University of Illinois Urbana-Champaign), Fan Lai `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 331 | [OpenAlex ID](https://openalex.org/A5101622777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AutoScout，一种面向机器学习训练、微调与推理的通用系统配置器，能够在庞大、层次化且混合离散/连续的配置空间中自动寻找高性能配置。

**💡 创新点**

创新点在于：1）将配置优化建模为带层次依赖的混合离散-连续优化问题；2）采用混合优化框架，分别用 MCTS 对稀疏结构决策进行树形搜索，用坐标梯度法对连续执行参数进行数值优化；3）引入自适应特征优先化的锦标赛机制快速确定有效的特征排序；4）通过多模拟器与真实剖析的自适应评估降低搜索成本。

**🔧 技术方法**

核心技术包括 Monte Carlo Tree Search (MCTS)、坐标梯度下降 (Coordinate‑wise SGD)、层次化多臂赌博机 (bandit) 用于协调稀疏与稠密优化、锦标赛式特征优先化、模拟器集成与自适应可信度切换。

**📊 数据集**

在多种大模型上验证：Llama‑3.2‑3B、Llama‑3.1‑Nemotron‑Nano‑VL‑8B‑V1、Qwen3‑30B‑A3B（训练）以及 Meta‑Llama‑3‑8B‑Instruct（推理），使用相应的训练/推理数据集（如 LMSYS‑Chat‑1M 等）。

**📈 对比分析**

与 vLLM、Megatron‑LM、UDO、CherryPick、Metis 等基线比较，AutoScout 在训练任务上平均提升 1.3–3.0× 的吞吐量，推理任务上提升 1.02×，并在搜索步骤数上比现有方法快 13.7–16.5×（步数缩减 80%+）。

**⚠️ 局限性**

局限性包括：对模拟器的依赖仍需在不同硬件/框架下重新训练；在极高噪声的模拟环境下性能仍会下降；在极大规模分布式环境中，MCTS 的树扩展可能产生额外内存开销。

---

## 481. InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model

**arXiv ID:** 2603.11911 | [PDF](https://arxiv.org/pdf/2603.11911v1)

**作者:** InSpatio Team `[一作]`, Guofeng Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个实时帧生成模型 InSpatio-WorldFM，用于从单张参考图生成多视角、实时、空间一致的场景图像。

**💡 创新点**

创新点在于将帧生成与显式 3D 锚点和隐式空间记忆结合，采用三阶段训练（预训练、可控帧模型、分步蒸馏）实现低延迟、多视角一致性；并通过 PRoPE 和 Hybrid Spatial Memory 提升空间一致性。

**🔧 技术方法**

技术包括 Latent Diffusion、Diffusion Transformer（PixArt-Σ）、Projection Relative Position Embedding (PRoPE)、Distribution Matching Distillation、显式 3D 锚点（点云渲染）与隐式记忆（参考帧注意力）。

**📊 数据集**

使用公开视频数据集（DL3DV、RealEstate10K 等）、自采集视频和 Unreal Engine 生成的合成数据。

**📈 对比分析**

与传统基于视频的世界模型比较，在 512×512 分辨率下单张 A100 GPU 可达 10 FPS，RTX 4090 7 FPS，交互延迟约 50‑70 ms；质量与原始多步模型相近，保持空间一致性，在多视角测试中无显著漂移。

**⚠️ 局限性**

局限包括：难以生成动态内容、运动边界问题、缺乏连续帧的视觉稳定性，依赖高质量 3D 锚点，推理仍受硬件限制。

---

## 482. Measuring AI Agents' Progress on Multi-Step Cyber Attack Scenarios

**arXiv ID:** 2603.11214 | [PDF](https://arxiv.org/pdf/2603.11214v1)

**作者:** Linus Folkerts `[一作]` (AI Security Institute), Jessica Wang `[通讯]` (AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估前沿 AI 模型在两套无防御的网络/ICS 虚拟范围内自主完成多步攻击链的能力。

**💡 创新点**

通过长期跟踪 7 种模型随时间和推理时计算量的进展，揭示性能随 token 预算对数线性提升且新一代模型在固定预算下更强。

**🔧 技术方法**

使用 ReAct 代理、上下文压缩、标准 Kali 工具和 AI 安全实验平台进行评估。

**📊 数据集**

使用了两套自定义 cyber range：32 步公司网络攻击“Last Ones”和 7 步工业控制系统攻击“Cooling Tower”。

**📈 对比分析**

在 10M 与 100M token 预算下对模型平均完成步骤数和最大值进行统计；在 “Last Ones” 上从 1.7 步提升至 9.8 步（10M）或最高完成 22/32 步；在 “Cooling Tower” 上最高平均 1.4 步（100M）。

**⚠️ 局限性**

限制包括无活跃防御、未计入检测惩罚、漏洞密度偏高、模型未使用专门工具或人机协同、token 预算受限等。

---

## 483. An Intent of Collaboration: On Agencies between Designers and Emerging (Intelligent) Technologies

**arXiv ID:** 2603.12018 | [PDF](https://arxiv.org/pdf/2603.12018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 484. See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay

**arXiv ID:** 2603.11601 | [PDF](https://arxiv.org/pdf/2603.11601v1)

**作者:** Ashish Baghel `[一作]` (Lossfunk), Paras Chopra `[通讯]` (Lossfunk)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探究在交互环境中为视觉-语言模型（VLM）提供视觉帧与符号表示是否能提升其表现，并系统评估三种顶尖VLM在Atari、VizDoom和AI2-THOR中的表现。

**💡 创新点**

首次系统性研究符号化对零样本VLM决策的影响，并证明符号提取质量是关键瓶颈；同时在不同分辨率与噪声下分析检测误差对性能的影响。

**🔧 技术方法**

采用零样本大VLM（Claude-4-Sonnet、GPT-4o、Gemini-2.5-Pro）与两阶段符号提取、OCAtari实时检测、噪声注入与分辨率实验；评估累计奖励、F1、IoU。

**📊 数据集**

使用Atari游戏（Pong、Breakout、Space Invaders）、VizDoom防守场景、AI2-THOR厨房任务等，帧分辨率1280×720。

**📈 对比分析**

对比四种管道（帧仅、帧+自提符号、帧+真符号、符号仅）并以真符号管道作为上限；结果显示当符号准确时性能提升显著，错误符号会削弱甚至恶化效果；Claude在自提符号时几乎逼近上限，GPT-4o和Gemini在复杂场景中受噪声影响大。

**⚠️ 局限性**

限于环境多样性、符号提取精度与实时性、VLM推理延迟与成本，以及对真实世界多模态场景的验证不足。

---

## 485. Reproducible Synthetic Clinical Letters for Seizure Frequency Information Extraction

**arXiv ID:** 2603.11407 | [PDF](https://arxiv.org/pdf/2603.11407v1)

**作者:** Yujian Gan `[一作]` (King's College London), Mark P. Richardson `[通讯]` (King's College Hospital NHS Foundation Trust)

**通讯引用:** 13295 | [OpenAlex ID](https://openalex.org/A5075147214)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了可复现的全合成医学笔记框架，训练大语言模型在英国 NHS 体癫痫门诊信中提取癫痫发作频率；

**💡 创新点**

创新点在于设计任务特定的结构化标签方案与链式推理解释，并使用全合成、隐私安全的数据生成来替代真实病历；

**🔧 技术方法**

采用了开源大语言模型（如 Qwen、Llama、Gemma 等）结合知识蒸馏、链式思考提示和证据跨度标注进行微调；

**📊 数据集**

使用了 15,099 条全合成的诊疗信与 1,781 条真实英国 KCH 病历信作为训练与评估；

**📈 对比分析**

通过与仅使用真实数据训练、仅使用合成数据训练以及混合训练的多种方案比较，模型在真实测试集上的 micro‑F1 可达 0.78‑0.85，显示合成数据可有效提升性能并提升对分布漂移的鲁棒性；

**⚠️ 局限性**

主要限制包括单中心数据、合成信的写作风格与真实信差异、缺乏真实信的结构化标签以及未开展临床医生可用性评估。

---

## 486. Direct-to-Device Connectivity for Integrated Communication, Navigation and Surveillance

**arXiv ID:** 2603.11848 | [PDF](https://arxiv.org/pdf/2603.11848v1)

**作者:** Muhammad Asad Ullah `[一作]` (VTT Technical Research Centre of Finland), Vadim Kramar `[通讯]` (VTT Technical Research Centre of Finland)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5087437289)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

研究低空无人机在城市环境中使用6G直连设备（D2D）实现陆基网络（TN）与非地面网络（NTN）双模切换，以提供通信、导航和监视（ICNS）服务，提出了统计LoS概率、路径损耗、垂直天线模式等模型并进行仿真评估。

**💡 创新点**

首次将3GPP NR与ITU‑R P.1410‑6模型结合，构建统一的D2D双模性能分析框架，展示TN与NTN互补的覆盖优势，为低空ICNS服务的6G设计提供系统级参考。

**🔧 技术方法**

使用ITU‑R P.1410‑6统计LoS模型、3GPP TR 38.811/38.814/38.108路径损耗与天线模型、Friis传输公式，以及MATLAB计算链路距离与仰角。

**📊 数据集**

采用3GPP技术报告和规范中的参数（如α=0.3, β=500, γ=15、频段3.6 GHz TN、2 GHz NTN、卫星高度300 km等）作为实验情景参数；未使用真实测量数据集。

**📈 对比分析**

通过比较TN与NTN在不同飞行高度、水平距离及卫星仰角下的LoS概率、路径损耗和接收信号强度，评估各方案的接收灵敏度满足率；结果显示TN在距离≤0.5 km时即可满足，而NTN在仰角≥30°时可覆盖更高高度，二者互补提升可靠性。

**⚠️ 局限性**

模型简化：仅考虑单一基站/卫星，未考虑多路径干扰、基站密度变化和天线升倾/波束成形等实际因素；仿真基于统计模型而非现场测量，可能低估/高估真实覆盖性能。

---

## 487. MedPruner: Training-Free Hierarchical Token Pruning for Efficient 3D Medical Image Understanding in Vision-Language Models

**arXiv ID:** 2603.11625 | [PDF](https://arxiv.org/pdf/2603.11625v1)

**作者:** Shengyuan Liu `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10004 | [OpenAlex ID](https://openalex.org/A5073968803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种训练无关、模型无关的层级Token剪枝框架MedPruner，针对3D医学图像在Vision‑Language模型中的推理，采用交互式切片锚点过滤（IAF）与基于累计注意力的动态核选择（DINS）两阶段策略，显著减少视觉Token数量并保持或提升诊断性能。

**💡 创新点**

创新点包括：①无训练且可迁移的两阶段剪枝方法；②通过像素级L1差异动态选取信息量大的切片，避免固定采样导致冗余；③基于自注意力累计权重的自适应核选择，动态决定保留Token数量；④实现Token保留率低至5%即可保持或提升性能。

**🔧 技术方法**

采用了像素L1距离阈值判定切片差异、视觉编码器自注意力矩阵计算Token重要性、温度缩放Softmax及nucleus filtering进行动态Token选取，以及冗余Token聚类匹配以保留全局上下文。

**📊 数据集**

在M3D、3D‑RAD和AMOS‑MM这三大3D医学VQA/报告生成基准上进行评估，涵盖多种VLM架构。

**📈 对比分析**

与Hulu‑L1、VisionZip、HiPrune等无训练剪枝方法对比，MedPruner在3D‑RAD、M3D及AMOS‑MM上均保持或提升Accuracy/ROUGE/BLEU指标，同时Token保留率从100%降至约52%（MedGemma可降至5%），推理速度提升约2–3倍。

**⚠️ 局限性**

局限性包括：依赖视觉编码器的注意力分布，低关注度病灶可能被误删；阈值γ、τ需手工调参，缺乏自适应调节；对极大尺寸体积的内存/时间开销仍未完全消除；对不同VLM架构兼容性验证有限。

---

## 488. Higher-Order Modular Attention: Fusing Pairwise and Triadic Interactions for Protein Sequences

**arXiv ID:** 2603.11133 | [PDF](https://arxiv.org/pdf/2603.11133v1)

**作者:** Shirin Amiraslani `[一作]` (York University), Xin Gao `[通讯]` (York University)

**通讯引用:** 21538 | [OpenAlex ID](https://openalex.org/A5100427710)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Transformer 上加入了高阶模块化注意力（HOMA）来捕获蛋白序列中的三元相互作用，提升序列到结构/功能的预测能力。

**💡 创新点**

创新点在于将传统的双线性自注意力与显式的三阶交互路径并行融合，并通过重叠块分解与局部窗口化实现对长序列的可扩展计算；同时采用低秩 U 投影减少参数。

**🔧 技术方法**

使用技术包括 Transformer、三阶注意力张量、重叠块分解、窗口化三阶注意、低秩投影、两层 MLP 融合以及预训练/冻结策略。

**📊 数据集**

数据集为 TAPE 基准，分别为 Secondary Structure（CASP12、CB513、TS115）、Fluorescence（绿荧光蛋白变体）和 Stability（设计蛋白变体）。

**📈 对比分析**

在相同的 Transformer 结构和超参下与全局多头自注意力、Blockwise‑2D、Linformer‑style 等基线对比，HOMA 在所有任务上均提升了 1%–10% 的指标（Secondary Structure Q3、Fluorescence、Stability 的 Spearman 相关），且参数量约为 21.5M，显著低于官方 38M 基线。

**⚠️ 局限性**

主要限制是三阶注意力仍带来显著的计算和显存开销，窗口大小需要权衡精度与资源；此外对更长序列的处理仍受块/窗口划分的限制。

---

## 489. OpenClaw PRISM: A Zero-Fork, Defense-in-Depth Runtime Security Layer for Tool-Augmented LLM Agents

**arXiv ID:** 2603.11853 | [PDF](https://arxiv.org/pdf/2603.11853v1)

**作者:** Frank Li `[一作]` `[通讯]` (University of New South Wales Sydney), Frank Li (University of New South Wales Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个不需fork OpenClaw 网关的零拷贝运行时安全层 PRISM，覆盖十个生命周期钩子，结合启发式+LLM扫描、会话风险累积、工具/网络/路径治理和防篡改审计。

**💡 创新点**

将安全控制分布在完整代理运行时，而非单一输入过滤；实现零拷贝插件与可选侧车服务的组合；引入分层风险阈值和可热重载的策略管理；提供链式可验证审计与操作面。

**🔧 技术方法**

TypeScript/Node.js 插件、HTTP 侧车服务、启发式规范化+LLM 扫描（Ollama）、路径规范化、会话/会话风险引擎、HMAC 链式审计、可热加载策略。

**📊 数据集**

110 个手工设计的案例组成的初步基准（包括直接注入、间接注入、工具滥用、凭证外泄等），以及基于 ASB/Agent Security Bench 等现有攻击语料的扩展。

**📈 对比分析**

使用同一套案例进行无 PRISM、启发式、插件、插件+扫描、完整 PRISM 的递增基准；结果显示完整 PRISM 攻击拦截率≈95%，误报率≈14%；在 80 案例上性能开销主要来自扫描，插件单独时 latency <1 ms，扫描层 p95 约 12.5 s，内存 ≤1.4 MiB。

**⚠️ 局限性**

检测覆盖不完整，依赖手工规则；策略维护需要人工；路径保护仅字符串级别，无法防御符号链接；侧车服务需额外部署；未覆盖 OS 层沙箱、网络出口控制，且在多租户大规模场景下验证不足。

---

## 490. SliceFed: Federated Constrained Multi-Agent DRL for Dynamic Spectrum Slicing in 6G

**arXiv ID:** 2603.11390 | [PDF](https://arxiv.org/pdf/2603.11390v1)

**作者:** Hossein Mohammadi `[一作]` (Mississippi State University), Vuk Marojevic `[通讯]` (Mississippi State University)

**通讯引用:** 2399 | [OpenAlex ID](https://openalex.org/A5089510906)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于联邦约束多智能体深度强化学习（SliceFed）的动态频谱切片框架，能够在6G基站间协同分配资源并严格满足干扰与低时延服务约束。

**💡 创新点**

创新点在于将频谱切片问题建模为约束马尔可夫决策过程（CMDP），采用拉格朗日原始-对偶方法与PPO结合实现安全约束学习，同时引入联邦学习实现跨基站协作而不泄露数据。

**🔧 技术方法**

使用技术包括：深度强化学习（PPO），约束强化学习（Lagrangian原始-对偶），联邦平均（FedAvg）聚合，连续动作空间的切片比例策略，以及离散化的 PRB 近似。

**📊 数据集**

数据集：通过仿真生成的稠密多小区 RAN 环境，包含 Poisson 流量、Rayleigh 小尺度衰落与 log‑normal 阴影衰落，覆盖 eMBB、URLLC 与 mMTC 三类服务。

**📈 对比分析**

与三类基准（等分切片、队列比例切片、随机分配）对比，SliceFed 在保持 URLLC 延迟满足率近 100%、干扰预算严格控制的前提下，获得更优的系统吞吐与资源利用率，且对流量波动表现出更强的鲁棒性。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，缺乏真实网络部署与异步联邦聚合实验；对极端干扰或快速时变环境的适应性尚未彻底评估；模型训练与推理的计算与通信开销需进一步优化。

---

## 491. AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style

**arXiv ID:** 2603.11482 | [PDF](https://arxiv.org/pdf/2603.11482v1)

**作者:** Joonyong Park `[一作]` (Spellbrush), Jerry Li `[通讯]` (Spellbrush)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了AnimeScore框架：收集15,000个A/B偏好标注，分析声学特征，并训练SSL模型实现动画风格自动评分。

**💡 创新点**

创新点：首次采用偏好式pairwise ranking评估动画语音风格，系统性分析并量化多维声学因素，同时证明掩码预测SSL模型能高效捕捉动画风格。

**🔧 技术方法**

使用技术包括：pairwise logistic 损失、SSL编码器（wav2vec2、WavLM、HuBERT、data2vec）、BiLSTM+MLP评分网络、t‑SNE+聚类用于样本匹配、传统声学特征提取与逻辑回归基线。

**📊 数据集**

数据集：Anim-400k（动漫声）、ReazonSpeech（电视日常语音）、Coco‑Nut（多样口语），最终挑选3,000条语音样本做A/B对比。

**📈 对比分析**

与手工特征逻辑回归基线（AUC≈69.3%）相比，掩码预测SSL模型（HuBERT）在测试集上达90.8% AUC，表现显著优越；其余SSL模型也在83–90%之间。

**⚠️ 局限性**

局限性：数据规模有限，受评者年龄/性别/动漫熟悉度分布不均；缺少对模型结构的更细粒度 ablation；未验证该评分器作为 RLHF 奖励信号的实际效果。

---

## 492. To Words and Beyond: Probing Large Language Models for Sentence-Level Psycholinguistic Norms of Memorability and Reading Times

**arXiv ID:** 2603.12105 | [PDF](https://arxiv.org/pdf/2603.12105v1)

**作者:** Thomas Hikaru Clark `[一作]` (Massachusetts Institute of Technology), Pedro Reviriego `[通讯]` (Universidad Politénica de Madrid)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对大型语言模型进行零样本、少样本和少量监督微调，预测句子级心理语言学特征，如句子可记忆性和上下文阅读时长。

**💡 创新点**

创新点在于首次将LLM提示和微调应用于句子级别的可记忆性和阅读时间预测，并证明微调可显著提升对人类数据的拟合度。

**🔧 技术方法**

使用的技术包括LLM提示、少样本提示、SFT/LoRA微调、基于Pearson相关和R²的评估以及可解释基线回归。

**📊 数据集**

实验数据集包括Word Memorability、Sentence Memorability、Natural Stories自 paced reading、OneStop眼动阅读时间等。

**📈 对比分析**

与传统基于词频、长度、惊奇度或嵌入的可解释基线比较，微调模型在可记忆性上的R²达到0.53–0.59，句子可记忆性0.45–0.49，眼动阅读时间的R²最高可达0.57；零样本表现则多为低相关或无相关。

**⚠️ 局限性**

主要局限是仅限于英文、对低资源语言未知、LLM零样本预测不可靠且需慎重验证。

---

## 493. How do AI agents talk about science and research? An exploration of scientific discussions on Moltbook using BERTopic

**arXiv ID:** 2603.11375 | [PDF](https://arxiv.org/pdf/2603.11375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 494. Synthesis-in-the-Loop Evaluation of LLMs for RTL Generation: Quality, Reliability, and Failure Modes

**arXiv ID:** 2603.11287 | [PDF](https://arxiv.org/pdf/2603.11287v1)

**作者:** Weimin Fu `[一作]` (Kansas State University), Xiaolong Guo `[通讯]` (Kansas State University)

**通讯引用:** 2982 | [OpenAlex ID](https://openalex.org/A5076198928)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估32款语言模型在RTL生成中的合成循环，提出硬件质量指数HQI并构建三层能力景观；

**💡 创新点**

创新点在于将功能正确率与合成质量统一度量，揭示合成准备与仿真通过率的显著差距，并指出多样本生成策略的重要性；

**🔧 技术方法**

采用端到端的合成评估流水线（Icarus Verilog、Yosys+Nangate45 45 nm、HQI 计算）以及工具判定的失效分类，并通过 OpenRouter 统一接口对模型进行对齐；

**📊 数据集**

使用 VerilogEval（155 单模块）和 RTLLM（47 多模块）共 202 任务的 Verilog 设计，按 AST 依赖计数给出任务复杂度权重；

**📈 对比分析**

通过 Coverage、Global HQI（best‑of‑five 质量上限）和 Expected HQI（单次调用预期质量）进行比较，最高模型 Gemini‑3‑Pro 达到 85.1 HQI，模型划分为三层（≥71、53–68、<53），功能通过率与 HQI 差距最高可达 13 点；

**⚠️ 局限性**

局限性包括仅采用零轮提示未考虑多轮交互反馈，评估不包含后置布图、时序约束及系统级集成，HQI 亦未覆盖路由及 Hold‑time 等后期实现细节。

---

## 495. TopoBench: Benchmarking LLMs on Hard Topological Reasoning

**arXiv ID:** 2603.12133 | [PDF](https://arxiv.org/pdf/2603.12133v1)

**作者:** Mayug Maniparambil `[一作]` (Intercom), Fergal Reid `[通讯]` (Intercom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TopoBench基准，评估LLM在六类拓扑网格谜题（连接、环闭、对称、可见性、连通性、路径）上的全流程推理能力，并开发了错误分类与因果干预诊断管线来定位推理失败源；进一步测试了输入表示、工具增强及提示策略的缓解效果。

**💡 创新点**

创新点在于：①将拓扑约束显式为六类谜题并提供多难度层次与验证器；②构建链式推理错误的因果干预框架，区分错误频率与对准确率的真实影响；③证明约束提取是主要瓶颈，并通过结构化工具信息显著提升性能。

**🔧 技术方法**

使用链式思维提示、LLM-as-Judge 自动注解、因果干预实验、工具增强（如结构化约束查询）、提示级策略（计划/回溯）以及多模型评估（GPT-5-mini-high、DeepSeek V3.2 等）。

**📊 数据集**

数据集：900个谜题（50个每家族×3难度），包含6个家族（Flow Free、Bridges、Loopy、Galaxies、Undead、Pattern），配有规则说明、示例、验证器和Gold解答。

**📈 对比分析**

与现有推理基准（GSM8K、MATH、ARC、Sokoban）相比，TopoBench在“硬”难度上最高模型准确率仅24%（GPT-5-mini-high），开源模型更低（10%）。错误因果干预显示，早期承诺和约束遗忘导致约11个百分点下降；工具增强在Bridges硬难度上提升10%。

**⚠️ 局限性**

局限性包括：因果实验仅在少量模型与谜题家族上进行；工具增强仅在单模型单家族验证；错误分类依赖链式推理输出，可能不完全反映内部机制；未探索多轮投票或最佳/多样化策略。

---

## 496. On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents

**arXiv ID:** 2603.12109 | [PDF](https://arxiv.org/pdf/2603.12109v1)

**作者:** Deyu Zou `[一作]` (Chinese University of Hong Kong), James Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 8165 | [OpenAlex ID](https://openalex.org/A5016082884)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并解决强化学习中 LLM 代理在主动推理任务中出现的信息自锁现象

**💡 创新点**

提出一种基于“方向性批评”（AS 与 BT 两个通道的正负信号）且仅通过重加权优势的轻量级方法，打破 AS 与 BT 之间的负向耦合，提升信息获取与信念更新

**🔧 技术方法**

强化学习（PPO、GRPO、GSPO），方向性批评注入（Likelihood‑margin auxiliary objective）、优势重加权、贝叶斯信念对比、代理代理、偏好推断与医疗诊断的交互模拟

**📊 数据集**

PE‑G / PE‑F（偏好推断）、MediQ（医学诊断）、FloDial（故障排查）共四个交互式推理数据集

**📈 对比分析**

与直接推断、PPO、GRPO、GSPO 基线相比，改进方法在 28 种设置中有 27 次显著提升，最大提升约 60%，并显著提高 AS 与 BT 指标，整体学习曲线更快收敛且鲁棒性较好

**⚠️ 局限性**

依赖可获得的方向性批评；对重加权强度 λ 的选择敏感，过大或过小都会影响稳定性；批评噪声会降低性能，尽管在一定范围内鲁棒，但高噪声仍然会导致效果退化

---

## 497. ThReadMed-QA: A Multi-Turn Medical Dialogue Benchmark from Real Patient Questions

**arXiv ID:** 2603.11281 | [PDF](https://arxiv.org/pdf/2603.11281v1)

**作者:** Monica Munnangi `[一作]` (Northeastern University), Saiph Savage `[通讯]` (Northeastern University)

**通讯引用:** 1630 | [OpenAlex ID](https://openalex.org/A5085864488)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个真实多轮患者-医生对话基准，评估大型语言模型在多轮对话中的准确性与一致性。

**💡 创新点**

创新点包括：①从r/AskDocs抓取完整的患者-医生对话，捕捉真实的后续提问与医生回答；②提出LLM-as-judge的评判框架和基于医生答案的评分rubric；③设计了新的多轮评估指标——对话一致性分（CCS）和错误传播率（EPR）。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑5、GPT‑4o、Claude Haiku、Gemini 2.5 Flash、Llama 3.3 70B）进行零样本生成；LLM-as-judge（GPT‑4o）进行自动评分；Bootstrap CI、Mann‑Whitney U等统计方法评估性能下降和误差传播。

**📊 数据集**

采用的数据集为r/AskDocs中的2,437条完整对话（8,204问答对），涵盖1,437个多轮会话，全部由真实患者与经验证的医生完成。

**📈 对比分析**

通过对比上述五个模型，在首轮对话中最高的GPT‑5仅有41.2%完全正确；所有模型在第2轮时正确率大幅下降，错误传播率高达35–44%；相比单轮评测，模型的多轮可靠性显著不足。

**⚠️ 局限性**

局限性包括：只使用英文对话；医生回答来自在线论坛，缺乏正式诊疗文档；未覆盖多模态输入或长期随访；未评估患者对回答的理解与执行力。

---

## 498. Survival Meets Classification: A Novel Framework for Early Risk Prediction Models of Chronic Diseases

**arXiv ID:** 2603.11598 | [PDF](https://arxiv.org/pdf/2603.11598v1)

**作者:** Shaheer Ahmad Khan `[一作]` (CureMD Research), Muddassar Farooq `[通讯]` (CureMD Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用电子病历（不含实验室检验）的临床特征，构建了五种常见慢性疾病（高血压、2型糖尿病、慢性肾病、COPD、冠心病）的早期风险预测模型，并通过将生存分析模型重新设计为可直接做分类，提供统一的疾病预测与风险评估。

**💡 创新点**

创新点在于：①将传统生存分析模型改造为能够输出分类结果的工具；②首次提出仅使用常规EMR特征（排除实验室数据）进行早期风险预测；③使用SHAP KernelExplainer直接解释随机生存森林的决策，无需中间代理模型；④在同一数据集上对比传统分类器与生存模型，展示后者可匹敌甚至优于LightGBM、XGBoost。

**🔧 技术方法**

主要技术包括：随机生存森林（RSF）及其三种分类转换方法（风险评分阈值、末期生存概率、叶节点分布）；传统树集成分类器（Random Forest、XGBoost、LightGBM）；SHAP解释方法（自定义KernelExplainer）；数据预处理与特征工程（将连续变量离散化、编码ICD-10、Elixhauser、药物GPI等）。

**📊 数据集**

使用约1000万名患者的匿名化电子病历数据（来自多家医疗机构），并在这些数据上抽样平衡后构建训练/验证/测试集，针对上述五种慢性疾病分别建立模型。

**📈 对比分析**

方法比较采用训练/验证/测试三路拆分，并与LightGBM、XGBoost、Random Forest等经典分类器做对比。结果显示：RSF在采用风险评分阈值时的F1与AUROC均与或优于传统分类器，且在所有疾病上均表现出稳定的高性能（如高血压F1≈0.76、AUROC≈0.83）。在测试集上，RSF的C-index均在0.70‑0.74之间，表明其时间风险预测能力良好。

**⚠️ 局限性**

局限性包括：①生存曲线在一年末表现出不自然的风险急剧上升，可能与样本时间分布偏差相关；②研究仅覆盖五种疾病，缺乏对其他慢性病的泛化验证；③实验为回顾性设计，存在时间偏倚与缺失数据挑战；④SHAP TreeExplainer目前不支持RSF，需要自定义实现；⑤仅使用常规EMR特征，可能限制模型的预测精度。

---

## 499. Tiny Aya: Bridging Scale and Multilingual Depth

**arXiv ID:** 2603.11510 | [PDF](https://arxiv.org/pdf/2603.11510v1)

**作者:** Alejandro R. Salamanca `[一作]` (Cohere Labs), Marzieh Fadaee `[通讯]` (Cohere Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一系列 3.35B 参数的高效多语言模型（Base、Global 以及针对不同地区的 Water、Earth、Fire），通过数据、tokenizer、后训练与模型合并的组合实现 70 种语言的平衡表现。

**💡 创新点**

核心创新在于：① 基于语言群组与脚本的权重化 tokenizers 与数据混合，② 综合使用翻译、提示级转换与教师多模型融合生成的合成数据，③ 在后训练中采用聚类、合并与轻量化偏好调优，④ 在多语言安全与文化意识评测中引入最低阈值与语言混淆指标，确保模型在低资源语言上的公平性。

**🔧 技术方法**

技术包括：密集解码 Transformer、并行块与交错滑动窗口+全注意力、SwiGLU 与无偏置、分组查询注意力、FP8 混合精度预训练、SFT + Preference 调优、模型合并（插值、加权、混合）以及多级量化（llama.cpp、MLX）。

**📊 数据集**

使用涵盖 70 种语言的公开与专有语料库（Fineweb‑2、编程语言数据、机器翻译平行语料、Synthetic Generation Pipeline、prompt‑level transformation 等），并对数据进行多维度过滤与热身（high‑quality “cooldown”）来平衡低资源语言。

**📈 对比分析**

与同尺度开源模型（Gemma‑4B、Qwen‑3.5‑4B、SmolLM‑3B 等）在 6 大类基准上对比：在翻译（Flores、WMT24++）中往往领先 5–10 %；在生成（mDolly、mArenaHard）与多项选择（Global MMLU、INCLUDE、PIQA）中保持竞争力；安全评估（MultiJail、XSTest）中平均安全率 ≥ 90%，语言间差距显著缩小。

**⚠️ 局限性**

局限性包括：仍存在低资源语言的性能下滑、提示语言对结果影响大、量化后质量下降（尤其是 4‑bit）、模型在高资源欧陆语言仍略逊、文化评测中多语言支持仍有限，且对极端低资源或非官方语言的覆盖不完整。

---

## 500. LLY Ricci Reweighting in Stochastic Block Models: Uniform Curvature Concentration and Finite-Horizon Tracking

**arXiv ID:** 2603.11060 | [PDF](https://arxiv.org/pdf/2603.11060v1)

**作者:** Varun Kotharkar `[一作]` `[通讯]`, Varun Kotharkar

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在平衡两块随机块模型（SBM）中，本文提出并分析了基于L​in‑Lu‑Yau Ricci曲率的边权重更新方法，并证明其在中等稠密条件下能够统一收敛、放大社区内外连接差异，提升谱聚类的特征值间隙和误分类率；

**💡 创新点**

创新点在于：①首次对SBM进行L​in‑Lu‑Yau曲率的非渐近全局集中证明；②用一阶曲率重加权显著提升谱聚类的eigengap；③建立了有限步迭代曲率流的确定性两权重递推，并给出了误差跟踪与误分类误差的非渐近上界；

**🔧 技术方法**

核心技术包括：概率集中（Bernstein、Hoeffding）、Ollivier–Ricci与Lin‑Lu‑Yau曲率计算、匹配论与Hall定理、矩阵扰动理论（Weyl、Davis–Kahan）、两层均值场递推分析与曲率下的Wasserstein距离构造；

**📊 数据集**

实验使用合成数据——平衡两块随机块图（N≈2n，内部连边概率p₀，外部连边概率p₁），在多组p₀、p₁、n值下验证理论；

**📈 对比分析**

与传统未加权图谱聚类对比，曲率重加权后特征值间隙提升显著（理论上为r_curv‑r），误分类率下降到理论误差上界（∝εₙ²），实验中误差相对原始方法下降10‑30%；

**⚠️ 局限性**

局限性：仅适用于中等稠密（np̅³≫log n）且平衡的两块SBM；需要正权重且需满足p₀、p₁的比例限制；迭代曲率流仅在固定有限步内保证误差控制，长期迭代误差积累不易控制；

---

## 501. A Semi-Decentralized Approach to Multiagent Control

**arXiv ID:** 2603.11802 | [PDF](https://arxiv.org/pdf/2603.11802v1)

**作者:** Mahdi Al-Husseini `[一作]` (Stanford University), Kyle H. Wray `[通讯]` (Northeastern University)

**通讯引用:** 270 | [OpenAlex ID](https://openalex.org/A5013249239)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了半去中心化（SDec‑POMDP）框架，统一了 Dec‑POMDP、MPOMDP 以及多种通信机制，并设计了一种可扩展的全局搜索算法 RS‑SDA*，用于在概率通信环境下求解最优策略。

**💡 创新点**

创新点主要有：① 将半去中心化属性与半马尔可夫控制概念结合，定义了通信的半马尔可夫特性；② 构造了 SDec‑POMDP 模型，证明其与 Dec‑POMDP、MPOMDP、k‑步延迟通信等模型等价；③ 推导了递归小步半去中心化 A*（RS‑SDA*）算法，提供了可证明的可采纳启发式搜索，并能在保持分散性前提下接近中心化性能。

**🔧 技术方法**

使用了：理论建模（状态/行动/观察/转移/奖励、选择器函数、动态决策网络）、半马尔可夫控制、递归小步搜索、聚类与记忆缓存、动态规划与启发式评估、以及对比实验中的计算资源监测。

**📊 数据集**

使用了四个经典 Dec‑POMDP 基准（Dec‑Tiger、FireFighting、BoxPushing、Mars）以及新建的海事医疗疏散（MaritimeMEDEVAC）基准进行实验。

**📈 对比分析**

通过与完全中心化上界、RS‑MAA*（Dec‑POMDP 的基准算法）以及不同通信设置（完全去中心化、半去中心化、完全中心化）进行对比。结果显示，RS‑SDA* 在大多数基准和海事疏散任务中能够取得接近或等同于中心化最优值（如 BoxPushing、Mars 右带队任务），在 FireFighting 中与 RS‑MAA* 相当，在 MaritimeMEDEVAC 中高达 96% 的中心化收益，并且在 20 分钟时间限制内完成多实例求解，唯一出现内存溢出的情况为极大规模实例。

**⚠️ 局限性**

局限性：① 计算复杂度仍为 NEXP‑complete，面对高维状态/观测空间时易出现时间或内存瓶颈；② 目前只支持时间平稳的半马尔可夫通信分布，非平稳或依赖上下文的通信模型尚未覆盖；③ 只实现了离线规划，未结合在线搜索或近似方法；④ 对大规模团体（>5 代理）聚类效果不佳，导致搜索树膨胀。

---

## 502. LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction

**arXiv ID:** 2603.11446 | [PDF](https://arxiv.org/pdf/2603.11446v1)

**作者:** Yuzhi Liang `[一作]` (Guangdong University of Foreign Studies), Xinrong Zhu `[通讯]` (Guangdong University of Foreign Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大型语言模型先验的因果推理框架，对法律案例文本进行粗细分层的法律要素提取与因果图构建，并将因果结构信息嵌入注意力机制以进行判决预测。

**💡 创新点**

创新点包括：①粗细分层混合提取结合统计预筛选与LLM语义推理，精准捕捉法律要素；②LLM辅助的因果结构不确定性消解与概率约束，克服Markov等价带来的方向不确定；③将因果图嵌入注意力约束，显著提升预测鲁棒性和解释性。

**🔧 技术方法**

采用了RAG+Prompt CoT进行法律要素抽取，GFCI因果发现与LLM概率消解来解决结构不确定性，BIC加权图采样和PSM因果效应估计，最后利用ALBERT注意力约束进行因果驱动的判决预测。

**📊 数据集**

使用的公开数据集包括中文的LEVEN、QA、CAIL2018，以及英文的LEDGAR和Overruling。

**📈 对比分析**

与12种基线模型（BiLSTM、BERT、LegalRoBERTa、LLMEmbed、CASAM等）以及多种Zero‑shot/Fine‑tune LLM模型进行对比，平均准确率在50%训练比例时达到85.7%，在1%低资源场景下为61%，在所有数据集均优于基线，尤其在相似罪名辨识任务上提升近4个百分点。

**⚠️ 局限性**

局限性在于：①因果结构仍依赖GFCI在稀疏特征下的精度受限；②LLM推理易受提示设计与hallucination影响，导致消解结果不稳定；③构建因果图需较多人工或多轮交互，跨域法律文本的通用性尚未充分验证。

---

## 503. Hybrid Human-Agent Social Dilemmas in Energy Markets

**arXiv ID:** 2603.11834 | [PDF](https://arxiv.org/pdf/2603.11834v1)

**作者:** Isuri Perera `[一作]` (Monash University), Julian Garcia `[通讯]` (Monash University)

**通讯引用:** 1101 | [OpenAlex ID](https://openalex.org/A5083299717)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了人类在能源负荷管理中将决策委托给自主代理时的协作与协调机制，并通过演化博弈与强化学习实验探讨了代理如何通过全局信号实现协调。

**💡 创新点**

创新点在于提出基于全局可观测信号的内在奖励机制，既能在混合人口中实现无成本的代理入门，又能在低折现率下诱导轮流协作并抵御非采用者的搭便车。

**🔧 技术方法**

使用了演化复制动力学、有限状态记忆一策略的重复博弈分析、强化学习（策略梯度）以及奖励塑造技术。

**📊 数据集**

实验数据来自基于MiniZinc建模的集中式DSLm最优解与多智能体强化学习场景，采用模拟的电力需求与定价表，未使用公开真实电网数据集。

**📈 对比分析**

通过与无奖励塑造的RL基线及基于中央价格信号的集中控制对比，实验显示加入内在奖励后系统成本可降低约25%，且在高折现率下实现合作占比显著提升。

**⚠️ 局限性**

局限性包括：奖励塑造参数需手动调节，难以在规模更大、多设备/用户异质的真实场景中验证，且代理未主动促使非采用者迁移至协作策略。

---

## 504. Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework

**arXiv ID:** 2603.11768 | [PDF](https://arxiv.org/pdf/2603.11768v1)

**作者:** Chingkwun Lam `[一作]` (Jinan University), Kuo Zhao `[通讯]` (Jinan University)

**通讯引用:** 717 | [OpenAlex ID](https://openalex.org/A5101454013)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出稳定与安全治理记忆（SSGM）框架，系统性分类动态记忆系统与失败模式，并通过正式化验证与重构机制保证长期一致性。

**💡 创新点**

首次将记忆治理与多重门控（写入验证、读写过滤、可逆重构）结合，提供可证明的语义漂移上界与隐私泄漏抑制。

**🔧 技术方法**

利用强化学习策略学习记忆操作、自然语言推理（NLI）做冲突检查、Weibull衰减函数实现时序治理、图结构双轨存储以及可验证的逻辑门控。

**📊 数据集**

在 LongMemEval、MemoryBench 等持续记忆与安全基准上进行实验，并使用公开的大规模多模态对话语料进行强化学习训练。

**📈 对比分析**

与传统 RAG、MemR1、AtomMem 等对比，SSGM 在保持相近任务成功率的同时，将语义漂移降低到常数区间，隐私泄漏率下降超过 50%，但写入延迟略有提升。

**⚠️ 局限性**

主要限制为治理门控导致的写入延迟、严格一致性可能抑制适应性以及图结构规模增长时检索效率受限。

---

## 505. STAMP: Selective Task-Aware Mechanism for Text Privacy

**arXiv ID:** 2603.12237 | [PDF](https://arxiv.org/pdf/2603.12237v1)

**作者:** Fengwei Tian `[一作]` (University of Arizona), Ravi Tandon `[通讯]` (University of Arizona)

**通讯引用:** 3591 | [OpenAlex ID](https://openalex.org/A5004316408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 STAMP 框架，用任务感知的 token 组划分和极坐标方向噪声（Polar 机制）实现局部差分隐私文本保护，兼顾隐私与任务效能。

**💡 创新点**

创新点在于：① 将 token 的隐私敏感度与任务重要性两维结合，按四组动态分配隐私预算；② 采用只扰动向量方向、保持模长的 Polar 机制，使编码与解码几何一致，显著提升语义保留；③ 通过任务特定的聚类实现对多任务的可扩展性。

**🔧 技术方法**

技术包括：局部差分隐私（metric LDP）、von Mises‑Fisher（vMF）方向噪声、cosine 最近邻解码、基于 NER/PII 的敏感度识别以及任务/查询相似度阈值筛选。

**📊 数据集**

使用三大公开数据集进行实验：SQuAD（问答）、Yelp（情感分类）和 AG News（主题分类）。

**📈 对比分析**

方法对比：与统一预算 + vMF、统一预算 + Laplace、无隐私基线比较。实验表明：Polar 机制在相同隐私预算下比 Laplace 具有更好的效能；STAMP 在低至中等隐私预算下明显优于统一分配，尤其在任务依赖性强的 SQuAD 上提升显著；整体在保持较低误差的同时，隐私预算更高效。

**⚠️ 局限性**

局限性包括：需预先给定任务描述，静态词嵌入相似度可能忽略句法/语篇重要性；高维嵌入需要较大 ε 才能保持效能；依赖 NER/PII 边界检测，误检可能导致隐私泄露；未考虑长距离依赖和序列级别的预算交互。

---

## 506. NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication

**arXiv ID:** 2603.11438 | [PDF](https://arxiv.org/pdf/2603.11438v1)

**作者:** Yusheng Zheng `[一作]` `[通讯]` (University of California), Yusheng Zheng (University of California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文实现了在NCCL插件体系中嵌入用户空间eBPF运行时，构建了一个可验证、可组合的策略执行框架，兼容现有插件接口且无需改动NCCL源代码；

**💡 创新点**

创新点在于将内核级eBPF的安全验证与JIT编译机制迁移至GPU集体通信场景，实现了插件的静态安全检查、结构化跨插件状态共享以及原子热更新；

**🔧 技术方法**

采用了eBPF、bpftime（用户空间JIT/验证器）、PREVAIL静态验证器、类型化map、atomic compare‑and‑swap热更新等技术；

**📊 数据集**

在实验中使用了8× NVIDIA B300（Blackwell）GPU集群（NVLink 5）、240核AMD EPYC 9575F CPU，进行1M调用的微基准、400k策略调用、AllReduce吞吐量等测评；

**📈 对比分析**

与原生C++基准相比，eBPF策略每次调用增加80–130 ns（<0.03%总延迟），AllReduce吞吐量提升最高27%，热更新无调用丢失，且在多次跑测中稳定性提升；

**⚠️ 局限性**

局限性包括：仅覆盖tuner、profiler、net插件，无法直接修改算法实现；验证仅保证内存安全与终止性，无法防止逻辑错误；目前仅在单机NVLink环境验证，缺乏大规模多节点、InfiniBand等场景的实测；

---

## 507. Summarize Before You Speak with ARACH: A Training-Free Inference-Time Plug-In for Enhancing LLMs via Global Attention Reallocation

**arXiv ID:** 2603.11067 | [PDF](https://arxiv.org/pdf/2603.11067v1)

**作者:** Jingtao Wang `[一作]` (McGill University), Xun Wang `[通讯]` (Zhejiang Gongshang University)

**通讯引用:** 45051 | [OpenAlex ID](https://openalex.org/A5100333868)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练自由、推理时可插拔的插件 ARACH，利用双流注意力和可调 Logit offset 在解码器 Transformer 内部实现上下文汇总与注意力重路由，以提升下一个 token 的预测性能。

**💡 创新点**

在不更新权重的前提下，首次在内部注意力层引入适配上下文中心（Hub）与可调 Logit offset，形成可插拔的内部汇总机制，解决了 attention sink 现象并显著提升推理质量。

**🔧 技术方法**

采用双流注意力结构、Hub 机制、可调 Logit offset、严格因果可见性掩码，并在 GPT‑2 小模型上实现该插件。

**📊 数据集**

在 LAMBADA、PG‑19、StoryCloze、SQuAD、WikiText‑103 等多种标准语言建模与 cloze 任务上进行评测。

**📈 对比分析**

采用相同权重、解码配置的配对比较方式，开启 ARACH 后在多项指标上均有提升：PG‑19 perplexity 降至 33.11（-4.22），LAMBADA 准确率提升 3.53，StoryCloze、SQuAD、WikiText‑103 亦出现正向改善。

**⚠️ 局限性**

缺点包括：需要额外的推理计算开销、对 Logit offset 参数的选取仍需经验、提升幅度在某些任务上不显著、仅适用于解码器结构，无法直接迁移到 encoder 或 encoder‑decoder 模型。

---

## 508. Multimodal Self-Attention Network with Temporal Alignment for Audio-Visual Emotion Recognition

**arXiv ID:** 2603.11095 | [PDF](https://arxiv.org/pdf/2603.11095v1)

**作者:** Inyong Koo `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7057 | [OpenAlex ID](https://openalex.org/A5069759184)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个基于Transformer的音视频情感识别框架，重点解决不同模态帧率不匹配问题，并通过自注意力融合音频和视频特征。

**💡 创新点**

创新点包括：①引入Temporally‑aligned Rotary Position Embedding（TaRoPE），通过对不同帧率的模态进行旋转校准，实现隐式时间对齐；②设计Cross‑Temporal Matching（CTM）损失，显式约束时间上相邻的音视频特征相似，提升跨模态时序一致性。

**🔧 技术方法**

使用的技术主要有：Transformer自注意力编码器、RoPE与TaRoPE位置编码、CTM损失函数、预训练音频模型xlsr‑Wav2Vec 2.0、OpenFace提取动作单元特征、温度调节的softmax、AdamW优化器。

**📊 数据集**

实验数据集：CREMA‑D（6种基本情绪）与RAVDESS（8种情绪），两者均采用公开的说话人独立训练/验证拆分。

**📈 对比分析**

与现有方法（如ATTSF‑Net、ISA/ICA等）对比，本文在CREMA‑D上取得89.49%准确率、在RAVDESS上取得89.25%准确率，分别比前沿方法提升4.43%和0.58个百分点，证明时间对齐与CTM损失能显著提升识别性能。

**⚠️ 局限性**

局限性：仅在受控实验室数据上验证，未考察大规模野外或多设备环境；对其他模态（如文本、深度图）或更复杂的时间对齐策略的适用性尚未评估。

---

## 509. HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies

**arXiv ID:** 2603.12243 | [PDF](https://arxiv.org/pdf/2603.12243v1)

**作者:** Amber Xie `[一作]` (Stanford University), Dorsa Sadigh `[通讯]` (Stanford University)

**通讯引用:** 7286 | [OpenAlex ID](https://openalex.org/A5080725225)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种基于仿真预训练、策略细化和残差强化学习的双阶段方法，实现了双手机器人精准演奏钢琴。

**💡 创新点**

提出在仿真得到粗略手指动作后，先通过物理跑行的结构化横向关节校正细化轨迹，再在此基础上进行残差强化学习，以补偿仿真误差并显著提升精度。

**🔧 技术方法**

使用TD3在ManiSkill仿真环境中训练基础策略，采用键盘几何与手指运动学的结构化修正，随后在真实环境中以MIDI反馈奖励进行残差强化学习。

**📊 数据集**

在五首经典曲目（《小星星》《欢乐颂》《热十字饼》《小星星》《前奏曲》）上收集30分钟的真实交互数据；仿真使用ManiSkill模拟环境。

**📈 对比分析**

与闭环仿真、混合执行、纯RL、无残差RL等基线对比，F1分数最高，性能比直接仿真提升约1.8倍，仅需30分钟真实数据。

**⚠️ 局限性**

仅依赖脚本化的手臂姿势和三指键盘，手指范围受限，细化步骤基于人工启发式，难以推广到非键盘类任务。

---

## 510. Personalized Federated Learning via Gaussian Generative Modeling

**arXiv ID:** 2603.11620 | [PDF](https://arxiv.org/pdf/2603.11620v1)

**作者:** Peng Hu `[一作]` (Harbin Institute of Technology), Jianwei Ma `[通讯]` (Peking University)

**通讯引用:** 15680 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了pFedGM，一种基于高斯生成建模的个性化联邦学习框架，通过共享特征提取器与个性化分类头的双尺度融合实现客户端个性化。

**💡 创新点**

首次将高斯生成模型与双目标（类间距离最大化与类内方差最小化）结合，并利用Kalman增益式信息融合自适应生成个性化分类器。

**🔧 技术方法**

采用高斯混合模型、重采样权重、共享与本地双目标、解耦导航与统计提取器、信息增益双尺度融合及L-BFGS细调等技术。

**📊 数据集**

在EMNIST、CIFAR-10/100、TinyImageNet及其受污染版本CIFAR-10S/100S上进行实验。

**📈 对比分析**

与FedAvg、Ditto、FedPer、FedPAC、pFedFDA等多种PFL方法对比，pFedGM在标准和环境异构场景下均取得最优或竞争性准确率，尤其在TinyImageNet和受污染数据上显著优于其它方法。

**⚠️ 局限性**

方法依赖于高斯假设且对协方差矩阵做对角约束，训练时需要多轮通信并计算均值协方差，可能在高维或极端非高斯数据上效果不佳。

---

## 511. ARROW: Augmented Replay for RObust World models

**arXiv ID:** 2603.11395 | [PDF](https://arxiv.org/pdf/2603.11395v1)

**作者:** Abdulaziz Alyahya `[一作]` (Imam Mohammad Ibn Saud Islamic University), Gideon Kowadlo `[通讯]` (Cerenaut)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

研发了一种基于 DreamerV3 的增量回放缓冲区 ARROW，用以在连续强化学习中降低灾难性遗忘。

**💡 创新点**

创新点在于将短期 FIFO 缓冲与长期分布匹配缓冲相结合，并通过拼接回放片段实现内存高效、任务无标识的训练。

**🔧 技术方法**

使用 RSSM 预测的 World Model、Actor-Critic 控制器、分布匹配长短期回放、离散化 latent、熵正则化以及对比的 TES‑SAC 等技术。

**📊 数据集**

使用 Atari 六个无共享结构游戏和 Procgen CoinRun 的六个共享结构变体作为评测数据集。

**📈 对比分析**

在相同内存预算下与 DreamerV3、TES‑SAC 对比；ARROW 在无共享结构任务几乎消除遗忘，获得最高 WC‑ACC；在共享结构任务保持优良的稳定‑可塑性平衡，样本效率略逊于 DreamerV3。

**⚠️ 局限性**

限制在于固定 50/50 的缓冲分配、有限容量导致最终遗忘、对奖励尺度差异敏感、未验证连续控制或机器人任务，仅在单一模型框架内进行测试。

---

## 512. Portfolio of Solving Strategies in CEGAR-based Object Packing and Scheduling for Sequential 3D Printing

**arXiv ID:** 2603.12224 | [PDF](https://arxiv.org/pdf/2603.12224v1)

**作者:** Pavel Surynek `[一作]` (Czech Technical University in Prague), Pavel Surynek `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 1518 | [OpenAlex ID](https://openalex.org/A5031732925)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了基于多策略组合的Portfolio-CEGAR-SEQ算法，用于求解顺序3D打印中的对象摆放与调度问题；

**💡 创新点**

创新点在于将多种对象排列策略与对象排序策略组合成组合策略集，并在多核CPU上并行执行CEGAR-SEQ的不同实例，从而显著降低使用的打印板数；

**🔧 技术方法**

技术包括：CEGAR（Counterexample Guided Abstraction Refinement）框架、线性算术模型、SMT求解器Z3、以及多策略组合（tactics与ordering）的并行化实现；

**📊 数据集**

数据集为随机生成的长宽高在8~64mm的立方体（1-32个）以及Prusa MK3S打印机可打印部件的34个真实模型；

**📈 对比分析**

比较方法：在单核和多核环境下与原始CEGAR-SEQ、Gecode和Z3求解器对比，实验显示Portfolio-CEGAR-SEQ在同等时间内使用更少的打印板且运行时间相差可接受；

**⚠️ 局限性**

局限性包括：子最优模式下只能安排固定小批量（k=4）的对象，难以保证全局最优；多策略组合导致运行时间随策略数量增加而上升；未考虑对象旋转和更复杂的机械约束。

---

## 513. Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation

**arXiv ID:** 2603.11984 | [PDF](https://arxiv.org/pdf/2603.11984v1)

**作者:** Chongyang Xu `[一作]` (Sichuan University), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7082 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Ada3Drift，一种在训练阶段利用漂移场实现多模态保留的单步3D视觉运动控制方法，推理时仅需一次前向传播（1 NFE）；

**💡 创新点**

将迭代精炼迁移到训练时，通过训练时漂移场、双向软匹配、sigmoid调度的损失转换以及多尺度温度聚合，消除了单步生成中的模式平均问题；

**🔧 技术方法**

采用点云编码器+FiLM条件化的U‑Net生成器、训练时漂移字段、双向归一化匹配、多温度聚合和sigmoid损失调度等技术；

**📊 数据集**

在Adroit、Meta‑World、RoboTwin仿真基准以及Agilx Cobot Magic机器人上的五个真实操控任务上进行实验；

**📈 对比分析**

与多步扩散政策（DP3）、流匹配/一致性模型（FlowPolicy、MP1）等单步方法对比，Ada3Drift在1 NFE下实现或超过多步10×NFE方法，平均成功率在所有基准与真实任务中均优于基线；

**⚠️ 局限性**

在极端高方差任务（如Meta‑World Hard）仍略逊，且温度与损失交叉点仍需手动调优，且对演示数量与任务分布存在一定依赖。

---

## 514. HELM: Hierarchical and Explicit Label Modeling with Graph Learning for Multi-Label Image Classification

**arXiv ID:** 2603.11783 | [PDF](https://arxiv.org/pdf/2603.11783v1)

**作者:** Marjan Stoimchev `[一作]` (Jožef Stefan Institute), Sašo Džeroski `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 19469 | [OpenAlex ID](https://openalex.org/A5064609702)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的半监督层次多标签分类框架 HELM，专门针对遥感图像中的多路径层级结构进行建模；

**💡 创新点**

创新点包括：①使用层次特定的 CLS token 让 ViT 能显式建模每个标签；②引入图卷积网络显式编码父子关系，提升结构感知；③结合 BYOL 的自监督分支充分利用海量无标签图像；

**🔧 技术方法**

技术手段：Vision Transformer + 层次特定 CLS token；Graph Convolutional Network（GraphSAGE）做结构传播；Bootstrap Your Own Latent（BYOL）做自监督学习；

**📊 数据集**

实验数据集：UCM、AID、DFC-15、MLRSNet 四个公开遥感图像数据集；

**📈 对比分析**

与传统 MLC、HMLC 以及现有方法（C‑HMCNN、HiMulConE、HMI）进行比较，HELM 在 AUPRC 上均达到最高或第二高，Ranking Loss 亦最低；在低标签比例（1%）下提升幅度可达 25–37%；

**⚠️ 局限性**

局限性：需要手动构建层级结构；图模块虽轻量但仍增加计算；自监督分支训练成本更高；仅针对 RGB 图像，尚未扩展至多模态或自动层级发现。

---

## 515. Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue

**arXiv ID:** 2603.11409 | [PDF](https://arxiv.org/pdf/2603.11409v1)

**作者:** Kratika Bhagtani `[一作]` (Purdue University), Amit Kumar Singh Yadav `[通讯]` (Ishiki Labs Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出多方语音助手的上下文感知发言决策任务，构建了120K样本的对话决策基准，并在此基础上对LLM进行零样本评估与监督微调。

**💡 创新点**

创新点在于：①首次将多方会话中的暂停点转化为二分类决策；②构建跨场景（会议、社交、财经）大规模标注基准；③利用教师模型蒸馏生成推理轨迹，显著提升模型的判定能力。

**🔧 技术方法**

技术方法包括：LLM零样本提示、LoRA低秩微调、推理轨迹蒸馏、四类平衡采样以及多模型的精细评估。

**📊 数据集**

所用数据集为AMI会议语料、Friends电视剧对话和SPGI财经会议，三者覆盖会议、社交与专业语境。

**📈 对比分析**

通过准确率、F1平均值和均衡准确率等指标比较，零样本下LLM表现仅靠近随机，微调后可提升至最高23个百分点，部分模型在测试集上已达到或超过人类水平。

**⚠️ 局限性**

局限性包括：决策任务本身主观性高，人工标注一致性有限；目前仅基于文本特征，缺乏多模态信号；在不同领域的泛化仍需进一步验证。

---

## 516. Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing

**arXiv ID:** 2603.12067 | [PDF](https://arxiv.org/pdf/2603.12067v1)

**作者:** Simone Cammarasana `[一作]` `[通讯]`, Simone Cammarasana

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统梳理并分类了五类卷积替代/增强的结构化算子（分解型、加权自适应、基底自适应、积分核、注意力），并给出形式定义、属性比较与适用任务；

**💡 创新点**

首次提供统一的卷积结构属性视角下的分类体系，揭示各算子如何逐步放宽卷积的线性、平移等假设，并结合任务与计算成本给出最佳选择建议；

**🔧 技术方法**

综述并整合了多种技术：SVD/张量分解、加权卷积、动态/变形卷积、F-transform/可学习小波、稀疏字典、非局部/RBF核、卷积核网络、CoordConv、注意力机制等；

**📊 数据集**

本论文为综述性质，未使用特定数据集；若引用实验，涉及图像去噪、超分、分类、医学影像等常见公开数据集；

**📈 对比分析**

通过结构属性表和任务适用性对比，讨论计算复杂度（如非局部/注意力为O(N²)）、线性与平移等属性的放宽程度，指出分解型算子在去噪/超分上可提升性能，注意力在大规模数据上表现更佳；

**⚠️ 局限性**

主要局限包括：计算成本高、实现复杂、缺乏统一的理论分析与收敛证明、对3D/硬件优化不足，以及在实际应用中需要进一步验证其泛化与可解释性。

---

## 517. Modeling Sequential Design Actions as Designer Externalization on an Infinite Canvas

**arXiv ID:** 2603.11569 | [PDF](https://arxiv.org/pdf/2603.11569v1)

**作者:** Yejin Yun `[一作]` (Hanyang University), Kyung Hoon Hyun `[通讯]` (Hanyang University)

**通讯引用:** 10521 | [OpenAlex ID](https://openalex.org/A5100732907)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究AI组织代理在无限画布设计工作流中的影响，通过对8名专业设计师的工作日志进行序列分析，比较有无AI代理的工作方式，揭示认知努力的重新分配和代理角色随阶段的演化。

**💡 创新点**

提出基于外部化动作的行为模型和序列分析框架，识别AI代理从“发散激励”到“协同结构化”再到“聚合策划”三阶段演化，并证明AI不增加工作负担却重塑设计者行为。

**🔧 技术方法**

使用事件日志抽象（ELA）与层级动作分类构建动作分类学，结合生成式AI（图像生成）与聚类算法，利用Markov链与频数统计分析序列模式。

**📊 数据集**

8名产品/空间设计师的交互日志，包含5,838次设计动作，实验设置为Baseline与Agent_organizer两种条件，配合两项任务（空间与产品），记录时间戳、事件类型等。

**📈 对比分析**

通过行动分布的卡方检验与Z‑score、活跃时间统计、Markov链转移概率进行比较；结果显示活跃时间无显著差异，但行动比例显著重分配，AI代理显著减少“搬移”动作并增加“关系”与“生成”动作，统计显著性强。

**⚠️ 局限性**

样本量仅8人，设计领域仅限空间与产品两项任务，实验为现场实验缺乏跨域验证；代理模型仅基于聚类与生成，未涉及更复杂交互；仅观察短期工作流，未评估长期设计成效。

---

## 518. Legal-DC: Benchmarking Retrieval-Augmented Generation for Legal Documents

**arXiv ID:** 2603.11772 | [PDF](https://arxiv.org/pdf/2603.11772v1)

**作者:** Yaocong Li `[一作]` (Beijing University of Posts and Telecommunications), Le Zhang `[通讯]` (Beijing Information Science and Technology University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对中文法律咨询场景，构建了Legal-DC基准数据集，并提出LegRAG框架，实现检索增强生成（RAG）的全流程，兼顾检索、生成与自我反思。

**💡 创新点**

创新点包括：①Legal-DC提供细粒度（条款级）问答与引用注释，支持检索与生成双向评估；②LegRAG采用双路径（块级+条款级）分割和法律自我反思机制，提升条款完整性与答案准确度；③设计自动评估协议，使用Qwen3-max进行结构化评估。

**🔧 技术方法**

使用技术包括：检索增强生成（RAG）框架、向量+关键词混合检索、重排序模型、LLM（Qwen3-max）生成与自我反思、实体增强、自动评估模板。

**📊 数据集**

主要数据集为Legal-DC（480份法律文件，2475条问答对，条款级引用），并与LightRAG、QAnything、Weknora等现有系统进行对比。

**📈 对比分析**

比较方法：采用Recall@5、MRR@5、Precision、BLEU、ROUGE、LLM-Score等指标；LegRAG+Qwen3-max在Recall从75.86%提升至78.02%，MRR@5从47.56%提升至50.23%，Precision、F1及LLM-Score均较基线提升约2–5%。

**⚠️ 局限性**

局限性：①对逻辑推理（多跳推理）仍表现不足；②自动评估依赖单一LLM，尚未完全替代大规模专家人工审核。

---

## 519. A Survey on Quantitative Modeling of Trust in Online Social Networks

**arXiv ID:** 2603.11054 | [PDF](https://arxiv.org/pdf/2603.11054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 520. NFPO: Stabilized Policy Optimization of Normalizing Flow for Robotic Policy Learning

**arXiv ID:** 2603.11470 | [PDF](https://arxiv.org/pdf/2603.11470v1)

**作者:** Diyuan Shi `[一作]` (Zhejiang University), Donglin Wang `[通讯]` (Westlake University)

**通讯引用:** 1546 | [OpenAlex ID](https://openalex.org/A5100665183)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将正则化流（Normalizing Flow）作为策略参数化，集成到PPO中，并通过对流的输出进行tanh归一化来解决在线强化学习中的训练不稳定性，最终实现多模态策略并成功迁移至真实机器人。

**💡 创新点**

①首次在在线PPO框架中使用NF；②提出通过tanh归一化s输出稳定训练；③证明NF能学习多模态行为并可直接迁移到真实机器人。

**🔧 技术方法**

PPO、RealNVP（NF）、tanh归一化、熵正则化、动作噪声实验等。

**📊 数据集**

三大仿真平台（Unitree RL Gym、Mujoco Playground、IsaacLab）以及真实Unitree G1机器人。

**📈 对比分析**

与RSL-RL的SOTA PPO（固定与自适应）以及FPO、Meow等多模态方法对比；在多任务实验中达到或超过SOTA，表现更稳健的收敛；在实机部署中实现稳定、全身运动。

**⚠️ 局限性**

对超参数（如归一化参数l、网络层数）较为敏感；训练时间略高（约19%墙钟时间）；在某些操作类任务仍需进一步调优。

---

## 521. Entropy-Preserving Reinforcement Learning

**arXiv ID:** 2603.11682 | [PDF](https://arxiv.org/pdf/2603.11682v1)

**作者:** Aleksei Petrenko `[一作]` (Apple), Philipp Krähenbühl `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了强化学习中策略梯度算法导致熵崩塌的问题，并提出通过显式熵控制保持探索性，提出两种新方法REPO与ADAPO；

**💡 创新点**

创新点在于：①用理论分析阐释熵随优势与对数概率的相关性；②识别数值精度与框架实现对熵的影响；③设计REPO（优势函数调节）和ADAPO（自适应非对称裁剪）两种显式熵保持机制；

**🔧 技术方法**

技术主要包括：政策梯度（REINFORCE、RLOO）、PPO及其变体（GRPO、LOOP、DAPO、GSPO）、熵控制与自适应阈值调节、FP16训练与BF16量化校正；

**📊 数据集**

使用的基准数据集有AppWorld（交互式工具使用任务）和AIME/NuminaMath-1.5（数学推理）等；

**📈 对比分析**

与GRPO、DAPO、GSPO等基线对比，REPO-R和ADAPO在保持熵的同时提升了测试准确率；RLOO+FP16在AppWorld上实现79% Test Normal、71% Test Challenge的SOTA成绩；

**⚠️ 局限性**

局限性包括：对数值精度与框架实现的高度依赖，需要手动调节熵阈值；严格的on‑policy训练存在同步瓶颈；方法在不同任务与模型规模下的泛化仍需进一步验证；

---

## 522. Reversible Lifelong Model Editing via Semantic Routing-Based LoRA

**arXiv ID:** 2603.11239 | [PDF](https://arxiv.org/pdf/2603.11239v1)

**作者:** Haihua Luo `[一作]` (University of Jyväskylä), Fengyu Cong `[通讯]` (Dalian University of Technology)

**通讯引用:** 4577 | [OpenAlex ID](https://openalex.org/A5042937962)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SoLA，一种基于语义路由的 LoRA 框架，用于可逆且可控的终身模型编辑。

**💡 创新点**

创新点在于：①为每一次编辑单独分配 LoRA 模块并冻结，②通过语义路由记录 LoRA 与输入语义的映射，③可通过删除路由键实现精确撤销编辑，④将决策机制集成到编辑层实现端到端推理。

**🔧 技术方法**

核心技术包括 LoRA 低秩适配、语义路由表、基于最近邻的键匹配以及主决策层的阈值激活。

**📊 数据集**

实验使用了 SCOTUS（法律判例分类）、zsRE（问答重述）、UniEdit 与 WikiBigEdit（幻觉纠正）等公开数据集。

**📈 对比分析**

与 EWC、CMR、CLEAR、MEND、SERAC、ROME、Grace、ELDER、MELO 等方法对比，SoLA 在 SCOTUS 上 ERR/TRR 领先 MELO 约 3%，在幻觉纠正任务中多场景下保持最高或次高 PPL，整体表现最优且参数消耗仅 0.08M。

**⚠️ 局限性**

局限性：需维护每个编辑的键与 LoRA 模块，随着编辑数量增长可能导致存储开销；对极大规模模型的扩展性尚未完全验证；对高度语义相似样本的路由冲突仍可能出现。

---

## 523. Quantized Inference for OneRec-V2

**arXiv ID:** 2603.11486 | [PDF](https://arxiv.org/pdf/2603.11486v1)

**作者:** Yi Su `[一作]` (Kuaishou Inc.), Ruiming Tang `[通讯]` (Kuaishou Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对生成式推荐模型 OneRec-V2 开发并部署了 FP8 后训练量化方案，并结合优化的推理基础设施，实现低精度推理。

**💡 创新点**

创新点在于：①将大语言模型（LLM）中的低精度量化经验迁移到生成式推荐场景；②通过权重和激活分布分析证明 OneRec-V2 与 LLM 在数值特性上的相似性；③提出针对 Compute‑Intensive 线性与 MoE 组件的块级量化与动态激活缩放；④在推理框架（RecoGEM）中实现多层级的低精度融合、TopK 取样、注意力和 MoE 的硬件友好优化。

**🔧 技术方法**

技术主要包括 FP8 量化（按通道/块级量化，动态激活缩放）、FP8 TensorCore 乘法 + FP32 累加、Post‑Training Quantization (PTQ)、TensorRT 执行图重构、RadixTopK、软流水线注意力核、Hopper TMA 支持的 MoE 组 GEMM、以及整体算力利用率提升（MFU）。

**📊 数据集**

使用 OneRec-V2 生产级模型（约 4B 参数，短视频生成式推荐），内部数据集，无公开公开数据集；实验基于真实线上流量做离线推理与线上 A/B 测试。

**📈 对比分析**

对比基线 FP16 推理（平均 139 ms 延迟，205 TPS）与优化后的 FP8 方案，得到 49% 延迟下降（70 ms）和 92% 吞吐量提升（394 TPS）。线上 A/B 结果显示核心指标（停留时间、观看时长、视频浏览、点赞、关注、评论、收藏、转发）均无显著下降，部分指标略有提升。

**⚠️ 局限性**

局限性：仅探讨 FP8 低精度，未验证 INT8、FP6 等更低精度；实现依赖重构后的 TensorRT 推理框架和硬件（Hopper 等），对资源有限的环境可复现性不高；实验仅覆盖 OneRec-V2，未验证对其他生成式推荐模型的通用性。

---

## 524. Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT

**arXiv ID:** 2603.11142 | [PDF](https://arxiv.org/pdf/2603.11142v1)

**作者:** Sai V R Chereddy `[一作]` `[通讯]` (Independent Researcher), Sai V R Chereddy (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对预训练的Video Vision Transformer（ViViT）在Kinetics-400数据集上的“strike vs. gutter”对照视频进行机制可解释性分析，逆向工程出隐藏的成功/失败信号并证明其分布式实现。

**💡 创新点**

首次在视频Transformer中揭示并验证了“Attention Head作为证据收集者、MLP块作为概念合成者”的分工机制，并提供了激活补丁等因果实验方法。

**🔧 技术方法**

采用观察性技术（注意力可视化、线性探针、L2 Delta分析）与因果干预技术（组件消融、激活补丁）来定位和评估内部信号。

**📊 数据集**

使用公开预训练模型 google/vivit-b-16x2-kinetics400 以及从Kinetics-400自制的10秒“strike”和“gutter”视频对。

**📈 对比分析**

通过与传统的logit归因、线性探针对比，发现MLP补丁可恢复高达60%的成功/失败信号，证明其主导作用；消融实验显示模型对单一关键块鲁棒，体现隐藏电路的分布式特性。

**⚠️ 局限性**

研究仅在单一视频对和单一模型上完成，缺乏对不同场景、样本及架构的泛化验证，且计算成本高，未能完全排除对特定背景特征的依赖。

---

## 525. Flowcean - Model Learning for Cyber-Physical Systems

**arXiv ID:** 2603.12015 | [PDF](https://arxiv.org/pdf/2603.12015v1)

**作者:** Maximilian Schmidt `[一作]` (Hamburg University of Technology), Stephan Balduin `[通讯]` (OFFIS Institute for Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

Flowcean框架实现了对网络化物理系统的模块化数据驱动模型生成与评估流程。

**💡 创新点**

创新点在于统一的学习与评估策略接口，支持离线、增量、主动三种学习方式，并实现跨库模块化。

**🔧 技术方法**

采用Python、PyTorch/Lightning、scikit‑learn、gRPC、ROS bag、Polars、LearnLib等机器学习与数据处理库。

**📊 数据集**

以单一水箱连续动力学仿真数据（250个采样，采样间隔0.1s）为案例数据集。

**📈 对比分析**

通过训练集80%与测试集20%比较，使用MAE和MSE评估，决策树在MAE 0.0206、MSE 0.0006、训练时长15.5 ms上显著优于多层感知机（MAE 0.0639、MSE 0.0054、训练时长813 ms）。

**⚠️ 局限性**

局限在于仅在简化仿真案例验证，缺乏大规模真实工业场景的验证与对实时性能的进一步评估。

---

## 526. Multimodal classification of Radiation-Induced Contrast Enhancements and tumor recurrence using deep learning

**arXiv ID:** 2603.11827 | [PDF](https://arxiv.org/pdf/2603.11827v1)

**作者:** Robin Peretzke `[一作]` (German Cancer Research Center), Klaus Maier-Hein `[通讯]` (German Cancer Research Center)

**通讯引用:** 27770 | [OpenAlex ID](https://openalex.org/A5027292126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了一种多模态3D深度学习模型RICE-NET，用于区分放疗后脑胶质母细胞瘤患者的放射性对比增强和肿瘤复发。

**💡 创新点**

创新点在于将临床常用的T1加权MRI与放疗剂量分布（RD地图）联合使用，并通过纵向时间点进行融合；同时通过大规模消融实验验证剂量图在分类中的关键作用，并首次对模型进行遮挡可解释性分析。

**🔧 技术方法**

采用3D残差网络ResNet18（MONAI实现）进行特征提取，使用多通道拼接融合多模态输入，Adam优化器、交叉熵损失及多种数据增强；可解释性采用遮挡敏感度图。

**📊 数据集**

数据集为92名GBM患者（80名训练/验证，12名独立测试），每人提供术后T1加权MRI、事件T1加权MRI及对应的放疗剂量分布图，标签由组织学确诊确定。

**📈 对比分析**

通过5折交叉验证和独立测试集进行比较：单模态剂量图宏F1为0.78，MRI+剂量图宏F1为0.83/0.828，三模态宏F1为0.804，最终独立测试集宏F1达到0.916，显示显著优于仅MRI或仅剂量图的方法。

**⚠️ 局限性**

局限性包括样本量有限、缺乏未复发对照组、简单通道拼接可能忽略复杂模态交互、仅来自单机构的数据、统计不确定性较大。

---

## 527. Uncertainty-Aware Estimation of Mis/Disinformation Prevalence on Social Media

**arXiv ID:** 2603.11058 | [PDF](https://arxiv.org/pdf/2603.11058v1)

**作者:** Ishari Amarasinghe `[一作]` (Universitat Oberta de Catalunya), Andreas Kaltenbrunner `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 2760 | [OpenAlex ID](https://openalex.org/A5053529497)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统估算社交媒体中的误/假信息普遍性，并量化样本、注释及数据检索三种不确定性；

**💡 创新点**

提出联合多源不确定性建模方法，将多分类模拟与关键词/帖子抽样Bootstrap相结合，首次对平台、多语言下的置信区间进行综合评估；

**🔧 技术方法**

使用Wilson分数区间、三分类多项式模拟、两级Bootstrap重抽样以及多源联合模拟；

**📊 数据集**

构建六大平台（Facebook、Instagram、LinkedIn、TikTok、X/Twitter、YouTube）在法国、波兰、斯洛伐克、西班牙四国的多语言样本，约3000条已专业标注数据；

**📈 对比分析**

与仅考虑样本误差的基线相比，加入注释与检索不确定性显著扩大置信区间，检索不确定性贡献最大，联合模型进一步提高准确度但增幅有限；

**⚠️ 局限性**

局限在于仅覆盖四国四语、六平台、单一时间窗口，未考虑内容领域差异、更多检索策略及平台API限制导致的数据偏差，未来需扩展语言与平台、动态跟踪变化。

---

## 528. ResWM: Residual-Action World Model for Visual RL

**arXiv ID:** 2603.11110 | [PDF](https://arxiv.org/pdf/2603.11110v1)

**作者:** Jseen Zhang `[一作]` (University of California, San Diego), Jinoh Kim `[通讯]` (Texas A&M University-Commerce)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Residual-Action World Model (ResWM)，将传统的绝对动作表示改为相对增量动作，并引入 Observation Difference Encoder (ODL) 通过编码相邻帧差异构建时序平滑的控制变量和动力学感知；

**💡 创新点**

创新点在于：① 将动作空间重参数化为残差动作，天然嵌入时间平滑先验，降低搜索空间并提升学习稳定性；② 采用 ODL 对视觉输入进行差分编码，突出动态信息，抑制静态冗余，强化感知-控制的因果关联；③ 以极小的修改即可集成至 Dreamer 族框架，无需额外超参数。

**🔧 技术方法**

技术包括：基于 RSSM 的递归状态空间模型、残差动作策略网络、观察差分编码器、KL 与能量正则化、lambda 回报的 actor‑critic 优化以及端到端的 VAE 训练。

**📊 数据集**

使用 DeepMind Control Suite（DMControl）连续视觉控制任务和 Atari 游戏数据集进行实验评估。

**📈 对比分析**

与 pixel SAC、DeepMDP、RAD、DeepRAD、TACO、MaDi、ResAct 等先进方法对比，ResWM 在 DMControl 的标准和困难任务上均实现了最高平均分和最快收敛速度；在 Atari 上亦取得了 0.96 的归一化平均得分，超过所有基线。

**⚠️ 局限性**

局限性包括：残差动作对突发高频环境冲击的响应受限，可能需要多步才能实现大幅动作变化；ODL 在实时推理时会增加处理相邻帧的计算与内存开销。

---

## 529. COMPASS: The explainable agentic framework for Sovereignty, Sustainability, Compliance, and Ethics

**arXiv ID:** 2603.11277 | [PDF](https://arxiv.org/pdf/2603.11277v1)

**作者:** Jean-Sébastien `[一作]`, Bélanger `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 COMPASS 框架，一个多代理编排系统，用于在大型语言模型代理中实时评估数字主权、碳足迹、合规性与伦理，生成量化得分和可解释说明。

**💡 创新点**

创新点在于将 Retrieval-Augmented Generation 与 LLM-as-Judge 结合，实现多维度、可解释的实时治理；采用组合设计实现模块化、可插拔。

**🔧 技术方法**

使用技术包括多代理架构、RAG检索、LLM-as-Judge 评估、BERTScore 语义相似度评估，以及 Mistral‑7B‑Instruct 作为核心模型。

**📊 数据集**

使用公开政策与法规文本作为 RAG 检索库（加拿大数字主权框架、欧盟 AI 法案、蒙特利尔 AI 声明等），并构造四类测试集（SOV、CAR、COM、ETH）。

**📈 对比分析**

通过与无 RAG 基线对比的自动化评估，ΔScore 显示 RAG 提升 0–0.25 分，BERTScore 维持在 75–90% 之间，说明 RAG 大幅提升语义一致性并降低幻觉风险。

**⚠️ 局限性**

局限性包括缺乏人工验证、动作选择与冲突解决功能不完善、RAG 文档选取缺乏系统性，以及 LLM-as-Judge 的偏见与一致性问题待进一步改进。

---

## 530. Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints

**arXiv ID:** 2603.11606 | [PDF](https://arxiv.org/pdf/2603.11606v1)

**作者:** Lijun Guo `[一作]` (Wuhan University), Hua Zou `[通讯]` (Wuhan University)

**通讯引用:** 16171 | [OpenAlex ID](https://openalex.org/A5111763066)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建一种从单一随意捕捉的单目视频中恢复可交互、物理可行的关节模型的数字孪生框架，采用两阶段优化：运动先验驱动初始化与几何运动约束细化。

**💡 创新点**

创新点包括：①利用3D轨迹学习低维运动基底，规整运动空间；②引入可学习的旋转/平移关节原语，实现物理可行的刚体运动；③在软硬分配间使用 Straight‑Through 估计实现端到端训练；④不需要静态扫描或多视角数据即可完成全流程。

**🔧 技术方法**

技术手段：3D Gaussian Splatting、运动基底学习（SE(3) 变换）、可学习的关节参数（轴、枢轴、标量）、软/硬分配与 STE、渲染损失（SSIM/ D‑SSIM/深度）、加速度与深度稳定性约束。

**📊 数据集**

使用三个数据集：Video2Articulation‑S（单关节合成），Articulat3D‑Sim（多关节合成，2-7 部件），Articulat3D‑Real（真实摄像机捕捉，9 类物体）。

**📈 对比分析**

与 RSRD、Articulate Anything、iTACO、Shape of Motion 等前沿方法比较，本文在关节角误差、位置误差、Chamfer 距离、EPE、PSNR、SSIM 等指标上均优于或接近最佳，尤其在多关节、无静态扫描情况下表现突出。

**⚠️ 局限性**

局限性：对纹理稀缺或对称表面易产生追踪漂移；仅建模独立关节，未考虑层级结构；在严重遮挡或极端光照下性能下降。

---

## 531. RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

**arXiv ID:** 2603.11106 | [PDF](https://arxiv.org/pdf/2603.11106v1)

**作者:** Shijie Zhou `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24372 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Robot-Conditioned Normalizing Flow（RC‑NF）模型，用于实时监测机器人执行状态与物体轨迹是否符合任务要求，并在此基础上开发了LIBERO‑Anomaly‑10基准，用真实机器人验证了RC‑NF在任务级和状态级OOD下的自适应修正能力。

**💡 创新点**

创新点在于将正常分布建模转化为机器人-对象条件的可逆流模型，设计了RCPQNet（Affine Coupling层），能够在不需要人工枚举异常的情况下，仅用成功示范进行无监督训练；同时利用SAM2分割的点集表示和任务嵌入的球面编码，实现了高效、低延迟（<100 ms）异常评分；此外，还提供了新的三类机械臂异常基准，填补了现有监测方法的空白。

**🔧 技术方法**

核心技术包括：Glow风格的可逆流（Normalizing Flow）与条件流；RCPQNet（Transformer+FiLM+GRU）作为Affine Coupling层；SAM2分割与网格采样得到的点集特征；任务嵌入采用球面均匀编码；交叉注意力实现机器人状态与物体轨迹的解耦与融合；阈值校准与异常处理流程。

**📊 数据集**

训练使用LIBERO‑10数据集的50条成功示范；评估使用新构建的LIBERO‑Anomaly‑10基准，其中包含Gripper Open、Gripper Slippage、Spatial Misalignment三种异常；在真实Frank R3机器人上使用RealSense摄像头进行实验。

**📈 对比分析**

与VLM监测方法（GPT‑5、Gemini 2.5 Pro、Claude 4.5）以及FailDetect流匹配模型进行比较，评估指标为AUC和AP。RC‑NF在所有异常类型上均超越基线，平均提升≈8 % AUC、≈10 % AP；在实时检测上实现低于100 ms的响应，能够及时触发任务重规划或轨迹回滚，显著提升VLA模型π₀的成功率。

**⚠️ 局限性**

局限性包括：需要高质量分割和点采样，依赖SAM2与bounding‑box提示；仅在10种预定义异常上验证，未知异常的泛化能力待进一步探索；阈值设定需要针对每个任务校准；当前实现主要在Franka R3机器人与高性能GPU上验证，低功耗平台的实时性尚未测试；以及对更复杂多物体、动态环境的适应性仍需后续工作。

---

## 532. Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration

**arXiv ID:** 2603.12226 | [PDF](https://arxiv.org/pdf/2603.12226v1)

**作者:** Priyanka Kargupta `[一作]` (University of Illinois at Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Idea‑Catalyst框架，系统化地从目标问题出发，进行问题分解、概念挑战抽象、跨学科检索和重构，帮助研究者生成跨域创新想法。

**💡 创新点**

创新点在于：①将元认知驱动的目标域分析与跨域检索结合，形成“先评估目标、再探索源域”的思路；②通过目标域问题的域内/域外双重表示，精准定位未解决的概念挑战；③使用跨域潜力排名对生成的跨域想法进行优先级排序，避免表面化或过度相似的跨域结果。

**🔧 技术方法**

使用技术包括：大型语言模型（如GPT‑4、gpt‑oss‑120b）进行问题拆解、摘要和评价；Semantic Scholar Snippets API进行文献片段检索；对目标域文献的检索与分析，形成挑战列表；跨域检索时依据域外关键词生成查询；生成跨域想法后采用对比式排名；评估阶段采用LLM判定与人类实验。

**📊 数据集**

主要数据集为CHIMERA（含400个目标-源域启发关系实例），同时使用IdeaBench的标准评测接口进行评价；检索时限定文献发布时间在实例投稿年份之前以避免信息泄漏。

**📈 对比分析**

比较方法：将Idea‑Catalyst与两种基线（Free‑Form Source、Guided Dual）以及若干消融实验（去除分解、去除潜力排名、加概念重写）在takeaway层和idea层进行LLM偏好评估；结果显示Idea‑Catalyst在创新度（+21.38%）和洞察力（+16.22%）上明显优于基线，消融实验表明问题分解与潜力排名是关键贡献。

**⚠️ 局限性**

限制：生成的跨域想法往往冗长、可解释性有限；跨域匹配仍受相似度设定影响，难以自动区分“相近”与“远离”领域；缺乏用户个性化解释与交互式迭代；在更大规模或更复杂目标域中的可扩展性尚未充分验证。

---

## 533. A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control

**arXiv ID:** 2603.12096 | [PDF](https://arxiv.org/pdf/2603.12096v1)

**作者:** Sheng-You Huang `[一作]` (National Yang Ming Chiao Tung University), I-Chen Wu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2754 | [OpenAlex ID](https://openalex.org/A5016730899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种鲁棒且高效的多智能体强化学习（MARL）框架，用于交通信号控制，主要包含转向比例随机化、指数相位持续时间调整以及邻居级观测与CTDE策略；

**💡 创新点**

创新点包括：①通过在训练时随机扰动转向比例来提升模型在非静态交通流下的泛化能力；②设计指数级的相位持续时间调节动作空间，实现粗细并存的灵活调控；③采用邻居级观测与集中式训练、分散式执行的CTDE框架，在保持可扩展性的同时实现全局级别的协作；

**🔧 技术方法**

技术上采用了多智能体PPO（MAPPO）算法、指数相位调节动作空间、转向比例随机化训练策略，并通过PTV Vissim微观仿真器实现与真实交通环境的接轨；

**📊 数据集**

使用的数据集为台湾桃园市中正东路的实际道路网络模型以及对应的24小时交通检测数据，分别构成高负荷高峰时段和低负荷非高峰时段的实验场景；

**📈 对比分析**

通过与固定时间信号、MaxPressure启发式和标准RL（静态转向比例）三种基线进行比较，使用平均通行时间（ATT）、平均等待时间（AWT）、平均延时（AD）和车辆吞吐量（VC）四项指标评估。实验结果显示，在高峰时段，本框架平均通行时间下降10%以上，在非高峰时段邻居+随机化方案仍优于基线且接近全局视角模型；

**⚠️ 局限性**

局限性包括：仅在由五个交叉口组成的线性网络上验证，未对更大规模或网格网络进行测试；只考虑四轮车交通，未融合多模式交通数据；仿真与真实部署之间仍存在差距，未来需要进一步的实地验证和模型适配。

---

## 534. ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control

**arXiv ID:** 2603.12185 | [PDF](https://arxiv.org/pdf/2603.12185v1)

**作者:** Chetan Borse `[一作]` (Arizona State University), Wanxin Jin `[通讯]` (Arizona State University)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5017350249)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种基于“无互补”接触建模的GPU并行解析接触物理引擎ComFree‑Sim，能够在多接触场景下以线性时间解析接触冲击，并兼顾六维摩擦。

**💡 创新点**

创新点在于将接触力闭式化为双锥阻尼更新，构建统一的6D摩擦模型，采用多面体线性化实现接触对与锥面完全解耦，从而在GPU上实现近线性扩展与高吞吐量。

**🔧 技术方法**

技术手段包括GPU Warp框架、MuJoCo兼容接口、双锥阻尼启发式、MPPI模型预测控制、动态感知运动重定向以及实时硬件控制实验。

**📊 数据集**

实验使用了多形状堆叠落体、LEAP手抓取多物体（立方体、鸭子、SpamTin、圆柱体）以及Unitree G1机器人的五种动作轨迹数据集进行评测。

**📈 对比分析**

通过与现有MJWarp引擎在穿透深度、摩擦行为、数值稳定性、运行时缩放和并行吞吐量等维度进行对比，实验表明ComFree‑Sim在密集接触场景下近线性缩放、吞吐量提升2–3倍，实测MPPI控制成功率提升约27%，控制频率可达35–72Hz。

**⚠️ 局限性**

局限性在于仍需手工调节阻尼参数以获得最佳刚度，某些极端摩擦或高度耦合场景的精度略低，并且尚未在更大尺度多体系统上进行充分验证。

---

## 535. Causal Representation Learning with Optimal Compression under Complex Treatments

**arXiv ID:** 2603.11907 | [PDF](https://arxiv.org/pdf/2603.11907v1)

**作者:** Wanting Liang `[一作]` (Shanghai University of Finance and Economics), Zhiheng Zhang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了一种多处理（multi‑treatment）因果表示学习框架，利用最优压缩理论自动估计平衡权重α，消除传统的网格搜索与超参数调优，同时通过聚合（Aggregation）策略实现对处理空间规模的O(1)可扩展性，并在此基础上设计了生成式模型 Multi‑Treatment CausalEGM，实现对处理流形的Wasserstein geodesic 结构保留。

**💡 创新点**

创新点包括：
1) 推导了多处理下的泛化界，揭示事实预测误差与表示层不平衡的精确权衡；
2) 将α视为可估计的贝叶斯超参数，通过边界优化得到自适应平衡；
3) 提出了三种平衡策略，其中聚合策略使用 HSIC 实现单一依赖度量，复杂度不随处理数增长；
4) 将生成式对抗网络与因果表示结合，保证对处理空间的几何一致性，支持高维反事实生成。

**🔧 技术方法**

技术手段包括：
- 结构化压缩（Controlled Compression）与双层优化（Bilevel Estimator）
- IPM（如MMD）与HSIC 作为不平衡度量
- Wasserstein 距离与 geodesic 约束
- 生成式模型（双向变分/生成对抗网络）CausalEGM 的扩展
- 理论分析（泛化界、渐近正态性）与实验验证。

**📊 数据集**

主要数据集：
- 半合成数据（高维 covariates + 多级处理）
- 图像/手写数字（Digits）等公开数据，用于评估生成式模型与几何一致性。
- 其他公开因果推断基准（如 IHDP、Synthetic）在论文中可能也有简要引用。

**📈 对比分析**

对比方法：
- 基线未平衡模型（Base）
- Pairwise、One‑vs‑All（OVA）、Aggregation（Agg‑T）三种平衡策略
- 传统 PEHE/ITE 评估指标。
- 在中等规模 (K=4) 上，OVA 取得最佳 PEHE；聚合策略在 K=20 时保持稳健且与小规模性能相当，且训练速度明显优于 Pairwise。生成式模型 Multi‑Treatment CausalEGM 在 Digits 数据上 PEHE ≈0.65，明显优于基线，且能恢复处理树的拓扑结构。

**⚠️ 局限性**

局限性：
- 仍假设无未观测混杂、充分重叠；
- 处理空间离散化，连续处理仍需进一步推广；
- 聚合策略虽然可扩展，但对极端高维 covariate 仍可能面临样本量不足导致的收敛不稳定；
- 生成式模型训练复杂度高，对硬件要求较大；
- 估计的 α 依赖于理论常数，实际取值可能需经验调整。

---

## 536. WeEdit: A Dataset, Benchmark and Glyph-Guided Framework for Text-centric Image Editing

**arXiv ID:** 2603.11593 | [PDF](https://arxiv.org/pdf/2603.11593v1)

**作者:** Hui Zhang `[一作]` (Tencent), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24372 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WeEdit，一个针对文本中心图像编辑的系统解决方案，涵盖数据构建、模型训练与评测。

**💡 创新点**

创新点包括：① Glyph‑Guided 监督微调（利用渲染的字符图像作为空间先验）② 多目标强化学习（分别评估指令遵循、文字清晰度、背景保持）③ 可扩展的 HTML‑based 数据构建流水线和 15 种语言的全量多语言 benchmark。

**🔧 技术方法**

使用技术：扩散模型（flow‑matching）、LoRA 参数高效微调、VLM 进行字符识别与布局规划、DiffusionNFT 在线 RL、VLM‑based 连续评分与多维奖励模型、Headless 浏览器渲染。

**📊 数据集**

数据集：WeEdit 数据集 330k 条编辑对（170k 结构化 + 160k 非结构化），覆盖 7 种操作、15 种语言；Benchmark 4000 条测试样本（2k 双语 + 2k 多语）。

**📈 对比分析**

对比方法：在双语与多语言 Benchmark 上与 15 个 SOTA 基线（4 专有、11 开源）进行 IA、TC、BP 三维评测；WeEdit‑RL 在所有维度均超过所有开源模型，仅落后 Gemini‑3‑Pro‑Image，提升幅度显著。

**⚠️ 局限性**

局限性：仍受限于评测标准与数据分布，模型在极端多区域/长文本、非拉丁复杂文字时性能下降；对高级推理与多模态上下文的适应性尚待提升。

---

## 537. DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use

**arXiv ID:** 2603.11076 | [PDF](https://arxiv.org/pdf/2603.11076v1)

**作者:** Aili Chen `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**通讯引用:** 4088 | [OpenAlex ID](https://openalex.org/A5090455375)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种以证据为先的任务合成框架，通过先执行真实工具收集可验证的执行轨迹，再反向导出任务，从而生成多样、可执行、可验证的agentic任务数据。

**💡 创新点**

创新点在于（1）逆向合成顺序：先执行工具再生成任务，实现任务与工具的内在可验证性；（2）构建三大资源池（工具池、种子概念池、示例池）以可控扩展工具覆盖和每个任务工具集的多样性；（3）在循环中动态收集证据并反向推导任务，逐步丰富多步骤工具使用模式。

**🔧 技术方法**

技术包括：使用大型语言模型（Claude‑4‑Sonnet）进行证据收集与任务生成；基于真实API的工具执行与验证；通过两阶段循环（Evidence Collection + Task Derivation）实现任务生成；利用监督微调（SFT）与强化学习（RL）对Qwen3‑8B进行后训练。

**📊 数据集**

数据集包括：373个验证过的真实工具（涵盖通用和四个专业领域）；约5000个种子概念；3–5个查询示例；合成任务约48k条SFT样本，3.2k条RL样本；在9个OOB基准上进行评估。

**📈 对比分析**

与同尺寸基准模型（WebExplorer‑8B、EnvScaler‑8B）及更大模型（Gemini‑3‑Pro、Claude‑4‑Sonnet）对比，所提方法在9个OOB基准平均提升约22分，超越最强8B基线68%；在特定专业基准（如金融、医学）亦能匹配或超越专业化模型，且对工具集和任务分布的漂移表现出显著的零样本泛化能力。

**⚠️ 局限性**

局限性包括：合成过程仍依赖大量真实工具调用，执行成本高；模型仍主要在已覆盖工具类型上表现良好，尚未充分验证对完全新工具的泛化；以及在复杂多步推理和长时延任务中，RL训练的样本效率和稳定性仍有提升空间。

---

## 538. Kraken*: Architecting Generative, Semantic, and Goal-Oriented Network Management for 6G Wireless Systems

**arXiv ID:** 2603.11948 | [PDF](https://arxiv.org/pdf/2603.11948v1)

**作者:** Ian F. Akyildiz `[一作]` (Truva Inc.), Tuğçe Bilen `[通讯]` (Istanbul Technical University)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5066357879)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了面向6G的Kraken架构，将语义通信、生成式推理与目标导向优化集成到三层（基础设施层、代理层、知识层）中，实现分布式集体智能，并给出了自动驾驶、XR渲染和分布式声学传感等案例研究。

**💡 创新点**

创新点包括：①将知识视为网络核心抽象，构建分布式知识平面；②引入生成式网络代理，利用世界模型实现前瞻性决策；③将语义优先级嵌入物理/MAC/路由层，形成目标对齐的调度与路由；④设计了层级化的语义协商与冲突升级机制；⑤提出与O‑RAN、MLOps、数字孪生相结合的可演进部署路径。

**🔧 技术方法**

采用的技术包括：语义通信（深度JSCC、任务感知编码）、生成式模型（变分自编码器/Transformer基的世界模型）、多智能体推理与Lagrangian协调、知识图谱与基金模型融合、O‑RAN控制接口（A1/E2）、MLOps管线、网络数字孪生验证、分布式学习与联邦更新。

**📊 数据集**

使用的数据集主要来自仿真与基准：CARLA+SUMO模拟自动驾驶交互；XR场景的高帧率视频与视角数据；工业声学传感的时间序列与异常标注；以及公开的通信系统基准如DeepJSCC实验数据，所有数据均在Kraken案例实验中生成或采集。

**📈 对比分析**

在三大案例中，Kraken实现了语义压缩带宽节省：自动驾驶场景可上传量降低70‑85%；XR渲染场景可降低10‑20×；分布式声学传感可压缩100:1，同时保持或提升任务成功率。与传统数据中心化或无语义的调度相比，Kraken在延迟（≤100 ms）、任务成功率和能耗上均表现优越。

**⚠️ 局限性**

局限性包括：生成式推理与世界模型的计算与能耗开销在边缘设备上仍高；知识同步与协商可能导致协议延迟与一致性问题；模型漂移与基金模型可信性需要严格验证；数据隐私与语义安全需额外加密与鉴权；以及从理论到实机部署的标准化、互操作性与可扩展性挑战。

---

## 539. AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization

**arXiv ID:** 2603.11873 | [PDF](https://arxiv.org/pdf/2603.11873v1)

**作者:** Qiyang Li `[一作]` (Baidu Inc), Dawei Yin `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于 Mixture‑of‑Experts 与 LoRA 的动态适配器架构，通过在第一层使用单一路由器进行 token 级预门控，并采用 SGMM CUDA 核实现全局融合适配器，显著降低推理延迟。

**💡 创新点**

创新点在于 token‑level 预门控（一次决策、全层复用）和 SGMM 融合核，打破传统层/块级动态路由导致的多次 CUDA 启动瓶颈。

**🔧 技术方法**

使用了 Mixture‑of‑Experts、LoRA、单层 Top‑2 路由器、SGMM fused‑adapter‑switching CUDA 核，基于 Llama2‑7B / Mistral‑7B。

**📊 数据集**

使用 ShareGPT（生成 200 token/查询）进行推理延迟测试，LM‑Eval‑Harness（ARC、HellaSwag、MMLU、TruthfulQA、WinoGrande、MT‑Bench）评估通用能力，ScienceQA/CommonsenseQA/OpenbookQA 评估领域特定能力。

**📈 对比分析**

与 LoRA、MoRAL、MOLA、PESC 等基线相比，在保持约 60% 的通用准确率（与 PESC 相当）的同时，推理延迟降低约 2.4×（比原 Llama2‑7B 低 29%），显著优于其他动态适配器。

**⚠️ 局限性**

局限性包括：仍比原始模型慢 29%，适配器激活仍需 GPU 端 SGMM 核支持；单层预门控可能限制跨层多样性；仅在 Llama2‑7B / Mistral‑7B 上验证，缺乏更大模型或多种适配器类型的评估。

---

## 540. Heavy-Tailed Principle Component Analysis

**arXiv ID:** 2603.11308 | [PDF](https://arxiv.org/pdf/2603.11308v1)

**作者:** Mario Sayde `[一作]` (American University of Beirut), Ibrahim Abou-Faycal `[通讯]` (American University of Beirut)

**通讯引用:** 828 | [OpenAlex ID](https://openalex.org/A5055725981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对高维重尾数据提出一种基于对数损失的主成分分析方法，通过估计潜在高斯生成器的协方差矩阵实现鲁棒降维，并在背景去噪任务中进行实验验证。

**💡 创新点**

① 在对数损失下证明重尾观测的主成分等价于潜在高斯生成器的主成分；② 提出三种协方差估计方法（比值法、对数相关法、LLN法）；③ 通过实验展示在重尾噪声环境下显著优于经典PCA和Tyler散点估计。

**🔧 技术方法**

超统计子归一化模型、对数损失函数、三种协方差估计技术、标准PCA、Tyler散点估计、MNIST图像与视频背景去噪实验。

**📊 数据集**

MNIST手写数字图像、视频帧数据，以及合成的Cauchy、Student‑t（DoF = 1.2）和高斯噪声。

**📈 对比分析**

与经典PCA和Tyler散点估计对比；在重尾噪声下误差降至3%以下，Gaussian噪声下性能相当；在背景去噪实验中视觉效果更好、残差更低。

**⚠️ 局限性**

协方差估计方法在低维或极端重尾情况下可能不稳定；对数相关法需要预先查表；模型假设A与G独立且A正；实验未涉及在线或张量推广。

---

## 541. UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution

**arXiv ID:** 2603.11680 | [PDF](https://arxiv.org/pdf/2603.11680v1)

**作者:** Cao Thien Tan `[一作]` (Ho Chi Minh City Open University), Nguyen Duc Dung `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的统一卷积注意力网络（UCAN），在图像超分辨率任务中有效扩展感受野并保持高效计算；

**💡 创新点**

创新点包括Hedgehog Attention提升特征多样性、半共享的Hybrid Attention机制、Flash Attention处理大窗口以及基于知识蒸馏的大核卷积模块；

**🔧 技术方法**

结合线性注意力、Flash Attention、窗口多头注意力、深度可分离卷积和特征蒸馏技术；

**📊 数据集**

在Manga109、BSDS100、Urban100、Set5、Set14等标准超分辨率数据集上进行训练与评测；

**📈 对比分析**

与现有轻量级CNN、Transformer和SSM模型比较，UCAN在保持参数量和MACs显著低于对手的同时，PSNR/SSIM表现更优，尤其在4×上达到48.4G MACs且PSNR 32.68dB；

**⚠️ 局限性**

仍存在对极大尺寸输入或超高放大倍数（如8×）的扩展性和实时性测试不足，且模型复杂度虽低但仍高于极简CNN基线。

---

## 542. Sim-to-reality adaptation for Deep Reinforcement Learning applied to an underwater docking application

**arXiv ID:** 2603.12020 | [PDF](https://arxiv.org/pdf/2603.12020v1)

**作者:** Alaaeddine Chaarani `[一作]` (Universitat de Girona), Pere Ridao `[通讯]` (Universitat de Girona)

**通讯引用:** 6550 | [OpenAlex ID](https://openalex.org/A5057442523)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

使用高保真Stonefish数字孪生环境和PPO算法实现并验证Girona AUV的自主对接控制。

**💡 创新点**

创新点包括：1) 将Stonefish改造为多进程RL框架，显著加速训练；2) 结合位置视觉伺服与噪声注入实现强鲁棒的sim-to-real迁移；3) 设计多维奖励函数（距离、角度、平滑、碰撞、任务完成）促进软对接行为。

**🔧 技术方法**

技术主要有：深度强化学习（PPO）、Stonefish物理仿真、ROS接口对接、姿态与加速度噪声注入、奖励函数自适应阈值。

**📊 数据集**

数据集：仿真中随机生成起始位置与航向的30‑60秒对接轨迹，真实试验使用19×9×5 m测试池内的10次任务，其中8次成功；未公开使用任何公共数据集。

**📈 对比分析**

与传统PID/MPC等控制方法对比：在仿真中成功率>90%，在真实池中8/10成功，且表现出自发的俯仰减速与偏航振荡等新颖对接策略；相比传统方法更能适应噪声和碰撞，实验中未出现因碰撞导致的任务终止。

**⚠️ 局限性**

局限性：1) 仅验证了静态水流与固定对接站，未考虑动态潮流或移动目标；2) 训练仅在GPU+CPU上完成，未测试大规模并行训练的可扩展性；3) 对接站模型简化（仅含引导漏斗），对实际结构细节的适应仍需进一步验证。

---

## 543. Expert Threshold Routing for Autoregressive Language Modeling with Dynamic Computation Allocation and Load Balancing

**arXiv ID:** 2603.11535 | [PDF](https://arxiv.org/pdf/2603.11535v1)

**作者:** Hanchi Sun `[一作]`, Lichao Sun `[通讯]` (Lehigh University)

**通讯引用:** 8190 | [OpenAlex ID](https://openalex.org/A5071709543)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种新的稀疏专家路由方法“Expert Threshold (ET) routing”，实现了在自回归语言模型中的完全因果性和动态计算分配，同时保持负载均衡。

**💡 创新点**

创新点在于将专家选择（EC）的 top‑k 统计从批次级别迁移到整个训练集级别，使用指数移动平均（EMA）为每个专家维护阈值；训练和推理使用相同阈值，消除了训练‑推理不匹配，并通过 4k 步 warmup 解决冷启动问题。

**🔧 技术方法**

技术包括混合专家（MoE）架构、EMA 阈值估计、sigmoid 门、负载均衡无辅助损失、共享专家、动态计算分配、因果自回归训练等。

**📊 数据集**

数据集主要为 FineWeb‑Edu（100B 训练集），评估使用 CORE benchmark、HumanEval、GSM8K 等。

**📈 对比分析**

与传统 Token Choice (TC)（固定 top‑G、辅助损失或 loss‑free 负载平衡）以及 Expert Choice (EC)（批次 top‑k）对比，ET 在 d12 模型上 CE loss 降 0.05、CORE 提升 1.89；在 d20 模型上 CE loss 降 0.067、CORE 提升 2.83；与 EC 在大批量训练时几乎等价，并在推理阶段实现完全因果性。

**⚠️ 局限性**

局限性包括：阈值需要 EMA 估计，训练早期可能出现专家枯竭；对小批量训练的 EC 仍有训练‑推理不匹配；阈值对不同任务或数据分布迁移的鲁棒性未知；实验主要在 2.4B 规模和单一数据集，尚未验证更大规模或多语言场景。

---

## 544. INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs

**arXiv ID:** 2603.11481 | [PDF](https://arxiv.org/pdf/2603.11481v1)

**作者:** Junqi Yang `[一作]` (School of Advanced Interdisciplinary Sciences), Xilin Chen `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了INFACT基准，包含9,800条基于真实与合成视频的问答，细粒度划分真伪性与事实性幻觉。

**💡 创新点**

创新点在于提供四种评测模式（基准、视觉降质、证据腐蚀、时间干预）以及对应的可靠性指标RR和TSS，能够系统检验视频LLM在不同扰动下的稳定性与时序敏感度。

**🔧 技术方法**

利用LLM辅助筛选、视觉噪声、字幕注入、对抗噪声以及帧打乱/反转等技术实现扰动，同时采用Resist Rate和Temporal Sensitivity Score评估模型表现。

**📊 数据集**

使用了来自MVBench、Video‑MME、TOMATO、CityGuessr68k、ViMULBench等公开视频问答数据集，以及Sora、Wan2.5等文本到视频生成模型合成的视频。

**📈 对比分析**

对14款Video‑LLM（含GPT‑5.1、Gemini3‑flash等）进行零样本评测，结果显示基准准确率与可靠性高度相关，但相同基准准确率的模型在扰动下排名不一致，许多开源模型在事实性方面表现差且TSS接近0，说明存在时间惯性。

**⚠️ 局限性**

局限性在于诱导模式仅覆盖有限的控制扰动，未能覆盖所有真实部署环境中的干扰，时间干预仅使用打乱/反转，未能定位模型具体依据的时间线索。

---

## 545. ShotVerse: Advancing Cinematic Camera Control for Text-Driven Multi-Shot Video Creation

**arXiv ID:** 2603.11421 | [PDF](https://arxiv.org/pdf/2603.11421v1)

**作者:** Songlin Yang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5025 | [OpenAlex ID](https://openalex.org/A5067715162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ShotVerse 框架，分为 Planner（利用 VLM 生成全局统一的多镜头相机轨迹）和 Controller（在 Holocine 之上通过相机适配器与 4D RoPE 实现精确的多镜头视频生成）。

**💡 创新点**

① 数据中心的突破——ShotVerse‑Bench 统一多镜头轨迹并构建三轨评测协议；② “Plan‑then‑Control” 两阶段解耦策略，使文本描述直接映射到可执行轨迹；③ 4D Rotary Positional Embedding 捕捉镜头层级结构。

**🔧 技术方法**

使用 Qwen3‑VL VLM + Transformer 解码器生成轨迹；LoRA 微调 Holocine 进行相机控制；相机适配器 + 4D RoPE；PI3、SAM、PI3 重建等相机标定技术；Flow Matching 训练目标。

**📊 数据集**

ShotVerse‑Bench：约 20,500 条高质量电影/电视剧片段，含多级文本描述和全局统一轨迹；对比基准 DataDoP、MVImgNet 等传统数据集。

**📈 对比分析**

通过三轨评测：(A) 文本→轨迹 F1/CLIP 0.418/34.9；(B) 轨迹→视频 0.0163/0.73/0.5 CAS；(C) 文本→视频 语义一致性、Aesthetic 5.465、FVD 281.71、Shot Transition 0.933，均超过现有单机与商业模型。

**⚠️ 局限性**

① 仍存在长镜头循环视角漂移；② 只能在同一场景内实现多镜头规划，难以无限延长或跨场景；③ 对高密度人群动态的适配不足。

---

## 546. Deep Learning-Based Metamodeling of Nonlinear Stochastic Dynamic Systems under Parametric and Predictive Uncertainty

**arXiv ID:** 2603.12012 | [PDF](https://arxiv.org/pdf/2603.12012v1)

**作者:** Haimiti Atila `[一作]` (University of Michigan), Seymour M. J. Spence `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对结构在随机地震荷载和结构参数不确定性影响下的非线性动力响应，提出并验证了三种基于LSTM的元模型框架，用以高效预测全时间历程并量化预测不确定性。

**💡 创新点**

创新点在于：①将特征提取模块（MLP、MPNN或AE）与LSTM结合，并采用Monte‑Carlo Dropout与负对数似然损失同时捕捉模型不确定性和参数不确定性；②使用小波变换压缩时域数据，降低LSTM计算负担；③在同一框架下对低维和高维结构系统分别做对比，展示不同网络结构在不同复杂度下的优势。

**🔧 技术方法**

使用技术包括：多层感知机（MLP）、消息传递神经网络（MPNN）、自编码器（AE）、长短期记忆网络（LSTM）、Monte‑Carlo Dropout、负对数似然（NLL）损失、小波变换、以及随机生成的地震记录。

**📊 数据集**

数据集为两套模拟数据：①40自由度Bouc–Wen剪切楼层模型，采用含参数不确定性（质量、阻尼、弹性模量）的1000多条随机地震记录；②37层纤维离散钢桁架结构，含阻尼、弹性模量和屈服强度不确定性，使用2000多条随机地震记录。两套数据均通过OpenSees/Matlab仿真生成。

**📈 对比分析**

对比方法为传统MLP‑LSTM、MPNN‑LSTM和AE‑LSTM三种架构；在两种案例中均使用相同训练/验证/测试划分，指标为MSE、RMSE、MAE。结果显示：Bouc–Wen案例中MLP‑LSTM表现最佳；而在高维钢桁架案例中MPNN‑LSTM和AE‑LSTM明显优于MLP‑LSTM，表明后两种结构在复杂几何和高维特征下更具优势。

**⚠️ 局限性**

局限性包括：①仅在两套模拟案例中验证，缺乏实际实验或多种结构类型的进一步验证；②模型对小波层数、LSTM层数等超参数敏感；③在极高维度或非平稳激励下可能仍面临计算负担；④自编码器的逆变换对不确定性传播的精确性有限。

---

## 547. Graph Tokenization for Bridging Graphs and Transformers

**arXiv ID:** 2603.11099 | [PDF](https://arxiv.org/pdf/2603.11099v1)

**作者:** Zeyuan Guo `[一作]` (Beijing University of Posts and Telecom), Chuan Shi `[通讯]` (Beijing University of Posts and Telecom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种通用图标记化框架，将可逆的、基于全局子结构频率引导的图序列化与字节对编码（BPE）相结合，将图转换为离散 token 序列；

**💡 创新点**

创新点在于：① 通过频率引导的可逆 Eulerian 电路实现图序列化的确定性与可逆性；② 将 BPE 直接应用于序列化图，从而自动学习结构化的子图词汇表；③ 让标准 Transformer 能直接处理图数据，无需修改网络结构；

**🔧 技术方法**

采用的技术包括：可逆图序列化（Eulerian circuit/CPP）、全局子结构频率统计、BPE 词汇表学习、以及基于 BERT/GTE 的标准 Transformer；

**📊 数据集**

使用了 14 个公开基准数据集，涵盖分子图（ZINC、QM9 等）、计算机视觉图（COIL-DEL）、图论实验（Colors-3、Synthetic）、生物医学图（DD、Peptides）、社交网络（Twitter）及学术网络（DBLP）等多领域；

**📈 对比分析**

与 GCN、GIN、GAT、GraphGPS、GraphMamba 等传统 GNN、图 Transformer 进行对比，GT+BERT/GTE 在 14 个任务上均达到或超过最优成绩，并在训练速度、序列压缩率等方面明显优于专用图 Transformer；

**⚠️ 局限性**

局限性包括：需要先收集全局子结构频率，依赖训练集分布；对极大或动态图的可扩展性尚未充分验证；BPE 可能无法捕捉所有稀有结构。

---

## 548. Artificial Intelligence for Sentiment Analysis of Persian Poetry

**arXiv ID:** 2603.11254 | [PDF](https://arxiv.org/pdf/2603.11254v1)

**作者:** Arash Zargar `[一作]` (University of Toronto), Farzad Khalvati `[通讯]` (University of Toronto)

**通讯引用:** 3367 | [OpenAlex ID](https://openalex.org/A5034208717)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用多种LLM（BERT多语种、Pars-BERT、GPT-4o、GPT-4o-mini）对鲁米和埃特萨米两位波斯诗人的诗歌进行情感分析，并探索情感与诗体韵律的关联。

**💡 创新点**

首次将大型生成式模型应用于古典波斯诗歌情感识别，并系统比较不同模型与人工标注的匹配度，揭示韵律在情感表达中的差异。

**🔧 技术方法**

采用Transformer架构的BERT编码器与GPT生成器、零样本推理、情感分数映射、熵、方差和偏振等统计量对诗句进行情感与韵律分析。

**📊 数据集**

使用Ganjoor在线数据库中的鲁米《Divan-i Shams》和埃特萨米《Divan-i Ashaar》诗集，包含标题、诗句及对应韵律信息。

**📈 对比分析**

通过Krippendorff α、Nominal Fleiss' Kappa、平均QWK和准确率等指标比较四模型与人工标注的匹配度；GPT-4o取得最高QWK≈0.60、准确率≈33%，其余模型表现显著逊色。

**⚠️ 局限性**

受古典波斯诗歌隐喻与古语结构的复杂性、预训练语料与目标文本分布不匹配、以及模型对细粒度情感评分的低准确性所限，导致当前LLM在此任务上仍存在显著性能瓶颈。

---

## 549. Fair Learning for Bias Mitigation and Quality Optimization in Paper Recommendation

**arXiv ID:** 2603.11936 | [PDF](https://arxiv.org/pdf/2603.11936v1)

**作者:** Uttamasha Anjally Oyshi `[一作]` (University of Arkansas), Susan Gauch `[通讯]` (University of Arkansas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于多层感知机（MLP）的公平论文推荐框架 Fair-PaperRec，用以在双盲评审后纠正性别、种族和国家等多维度的作者偏见，同时保持论文质量。

**💡 创新点**

创新点在于：①使用多属性公平损失函数，实现种族与国家的交叉公平约束；②在评估阶段不直接输入受保护属性，而是通过公平损失间接消除偏见；③通过调节 λ 与权重 W_r、W_c 的组合，系统性探究不同公平权重对多维公平和论文质量的影响。

**🔧 技术方法**

技术手段主要包括：MLP（两层隐藏层，ReLU + BN），交叉熵预测损失，统计平等（demographic parity）公平损失，λ 权重调节，Adam 优化器，早停策略。

**📊 数据集**

使用了三大会计数据集：SIGCHI 2017、DIS 2017 与 IUI 2017 的论文提交信息和作者属性（性别、种族、国家、职业阶段）。

**📈 对比分析**

与传统无公平约束的 Demographic‑Blind MLP 基线对比，评估指标包括宏/微多样性提升（Macro/Micro Gain）、效用提升（Utility Gain）以及 F‑度量。实验结果显示，在 λ=3（种族）或 λ=2.5（国家）下，公平性提升约 42% 左右，整体效用提升 3.16%，表明公平性提升不牺牲论文质量。

**⚠️ 局限性**

局限性包括：①未采用因果模型，无法排除间接偏差源；②公平损失仅针对种族与国家，性别等属性被排除；③在不同会议中公平目标不一致，λ 与权重调节需手工设定；④仅在三大计算机交互会议数据上验证，泛化能力待进一步验证。

---

## 550. The Density of Cross-Persistence Diagrams and Its Applications

**arXiv ID:** 2603.11623 | [PDF](https://arxiv.org/pdf/2603.11623v1)

**作者:** Alexander Mironenko `[一作]` (Skolkovo Institute of Science and Technology), Serguei Barannikov `[通讯]` (CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究并证明了交叉持久图（cross‑persistence diagram）的密度存在性，并提出基于该密度的统计方法与机器学习框架，用以区分不同流形上的点云以及在时间序列和AI生成文本等任务中的应用。

**💡 创新点**

创新点在于首次给出交叉持久图密度的理论证明、将其作为可估计的概率密度引入统计推断、以及设计首个专用神经网络架构Cross‑RipsNet，实现直接预测交叉持久图或其线性表示（如MTD）的密度，显著降低了传统持久图计算的计算开销。

**🔧 技术方法**

使用的技术包括持久同调、交叉持久图的Vietoris‑Rips过滤、共面积公式推导密度、线性表示（Persistence Image、MTD等）、核密度估计、交叉持久图密度的统计推断、DeepSets/交叉RipsNet神经网络、距离矩阵降维（PCA、最大距离、分位数），以及对噪声注入和敏感性分析的实验验证。

**📊 数据集**

实验数据集涵盖多模态：MNIST、CIFAR‑10/100、COIL‑20、ModelNet10（3D形状）、GPT与人类文本（Wiki/Reddit）、UCR时间序列数据库（6个任务）以及合成圆形点云。

**📈 对比分析**

与传统的单点云持久图表示（Persistence Image、Persistence Landscape）以及基准方法（CATCH‑22、FreshPrince）相比，交叉持久图密度在点云区分、生成模型评估、时间序列分类和AI文本检测等任务中均实现了更高的准确率/ROC AUC，Cross‑RipsNet的预测速度比传统管线提升约4–6×，且对噪声鲁棒性更好。

**⚠️ 局限性**

主要限制包括：交叉持久图计算仍需构造二次尺寸的距离矩阵，导致在点云规模大于约10⁴时显著内存与计算瓶颈；理论证明依赖于解析流形与无边界假设，实际数据中可能不满足；对跨域迁移（如不同分辨率、不同数据分布）仍需进一步验证。

---

## 551. Interpreting Contrastive Embeddings in Specific Domains with Fuzzy Rules

**arXiv ID:** 2603.12227 | [PDF](https://arxiv.org/pdf/2603.12227v1)

**作者:** Javier Fumanal-Idocin `[一作]` (University of Essex), Javier Andreu-Perez `[通讯]` (University of Essex)

**通讯引用:** 4466 | [OpenAlex ID](https://openalex.org/A5029626997)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究通过模糊规则解释CLIP嵌入空间，将情感分析特征映射到K‑means聚类结果，从而理解文本在嵌入空间中的结构。

**💡 创新点**

创新点在于将区间型二阶模糊集合与CLIP嵌入结合，并比较不同模糊集与损失函数对规则库规模、准确率与MCC的影响，首次将模糊规则用于解释多模态嵌入。

**🔧 技术方法**

采用CLIP预训练模型生成文本嵌入、K‑means聚类、情感分析特征提取、基于区间型二阶模糊集合的模糊规则分类系统（FRBC），使用遗传算法优化规则，损失函数包括单纯MCC与加规则规模惩罚的组合。

**📊 数据集**

使用两个数据集：51条中风康复患者口述报告和5万条IMDB电影评论。

**📈 对比分析**

通过在两数据集上分别比较标准模糊集合与区间型二阶模糊集合、单一MCC损失与含规则规模惩罚的损失，评估准确率和MCC。临床数据准确率在0.63–0.69之间、MCC在0.47–0.56之间；电影数据准确率在0.36–0.49之间、MCC在0.09–0.10之间；同时观察规则数和前件数的变化。

**⚠️ 局限性**

主要限制包括：规则库规模与解释性权衡导致性能下降；高文本多样性的电影数据映射效果差；仅使用情感特征不足以捕捉嵌入空间全部结构；需要更多特征或更精细的嵌入以提升解释力。

---

## 552. Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks

**arXiv ID:** 2603.11676 | [PDF](https://arxiv.org/pdf/2603.11676v1)

**作者:** Yongqi Ding `[一作]` (University of Electronic Science and Technology of China), Lin Zuo `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 3774 | [OpenAlex ID](https://openalex.org/A5101930491)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在脉冲神经网络中引入稳定脉冲（Stable Spike）和幅度感知脉冲噪声，实现双一致性优化，显著提升低时延下的识别性能。

**💡 创新点**

①利用硬件友好的 AND 位运算从相邻时步脉冲图中提取跨时步的稳定脉冲骨架；②将幅度感知脉冲噪声注入稳定脉冲，以在保持关键语义的同时增加特征多样性；③双一致性目标直接约束脉冲图一致性和扰动一致性，无需修改神经元模型或网络结构。

**🔧 技术方法**

LIF 神经元模型、AND 位运算、幅度感知脉冲噪声、MSE/KL/余弦一致性损失、温度软化+KL一致性、跨时步平均等。

**📊 数据集**

神经形态数据集：CIFAR10‑DVS、DVS‑Gesture、N‑Caltech101；静态图像数据集：CIFAR10、CIFAR100、ImageNet。

**📈 对比分析**

在 VGG‑9、ResNet‑18、QKFormer 等多种架构上与现有 SNN 方案（如 SLT、MPS、CLIF 等）对比，4 步时延下在 CIFAR10‑DVS 上准确率从 72.9% 提升至 77.1%（+4.2%），在 DVS‑Gesture 上从 94.44% 提升至 94.44%（+7.29%），在 ImageNet 上以 ResNet‑34 在 4 步时延下达到 70.59% 的准确率，均显著优于传统方法。

**⚠️ 局限性**

单时步（T=1）下提升有限；需要额外的前向传播与噪声注入，导致轻微计算开销；幅度感知噪声的概率参数需调节，过高时可能导致不收敛；对极端噪声环境的鲁棒性尚待进一步验证。

---

## 553. VTEdit-Bench: A Comprehensive Benchmark for Multi-Reference Image Editing Models in Virtual Try-On

**arXiv ID:** 2603.11734 | [PDF](https://arxiv.org/pdf/2603.11734v1)

**作者:** Xiaoye Liang `[一作]` (Beihang University), Yiheng Zhu `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了VTEdit-Bench基准，用于评估多参考图像编辑模型在虚拟试衣（VTON）场景中的性能，并提出了VTEdit-QA评估框架。

**💡 创新点**

创新点包括：①首次构建覆盖五种复杂VTON任务的多参考编辑基准；②设计基于大型视觉语言模型的参考感知评估器VTEdit-QA；③系统比较通用编辑模型与专业VTON模型的泛化与鲁棒性。

**🔧 技术方法**

采用多参考图像编辑模型（如Flux.2、Qwen-Image-Edit-2511等）零样本评估，使用GPT‑4o实现VLM评估，结合FID/KID与VTEdit-QA三维度指标，并提供OpenPose、DensePose等辅助条件。

**📊 数据集**

构建数据集融合DressCode、VITON‑HD、StreetVTON、DeepFashion、DressCodeMR等资源，生成24,220个测试对，涵盖室内外、多人、多视角、多物品等五类任务。

**📈 对比分析**

通过与7个专业VTON基线对比，使用FID/KID与VTEdit-QA分数评估；结果表明通用编辑模型在Shop2Model及更难场景中与专业模型相当或更稳健，但在Model2Model与MultiShop2Model等多物品、多参考任务中仍存在性能差距；VTEdit-QA与人类偏好高度相关。

**⚠️ 局限性**

局限性在于：通用编辑模型在多物品、多参考情况下身份与服装一致性下降；缺乏针对VTON的专门视觉理解与指令遵循能力；多参考绑定与空间约束不足，导致细粒度控制受限。

---

## 554. Client-Conditional Federated Learning via Local Training Data Statistics

**arXiv ID:** 2603.11307 | [PDF](https://arxiv.org/pdf/2603.11307v1)

**作者:** Rickard Brännvall `[一作]` (RISE Research Institutes of Sweden), Rickard Brännvall `[通讯]` (RISE Research Institutes of Sweden)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5077095868)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将单一全局模型与本地训练数据的PCA特征值向量进行条件化的方法，用以处理联邦学习中的多维数据异质性。

**💡 创新点**

创新点在于不通过聚类或个体化模型发现客户关系，而是直接利用每个客户端训练数据的局部统计特征（PCA特征值）作为连续的分布指纹，对全局模型进行条件化，既能匹配Oracle（知晓真实簇分配）的性能，又能在多维异质性下优于Oracle。

**🔧 技术方法**

技术细节包括：①在每个客户端对联合特征-标签矩阵做PCA，提取前32个特征值；②将该统计向量与卷积层输出的扁平化特征拼接后送入全连接层；③采用标准的FedAvg通信协议，训练时不额外交换统计信息；④在实验中使用SGD、交叉熵、固定学习率和批大小。

**📊 数据集**

使用四个公开图像分类数据集：MNIST、Fashion-MNIST、CIFAR-10和CIFAR-100，并在四种异质性场景（标签移位、协变量移位、概念移位、组合异质性）下构造97种不同的实验配置。

**📈 对比分析**

与七种基线（Local、FedAvg、Gossip、Oracle、IFCA、DAC、Ditto）进行比较。实验结果显示：在97个配置中，Conditional方法在95例中与Oracle匹配或更好；在组合异质性场景下平均提升1–2.9个百分点；在数据稀疏（每客户端200样本）时保持近乎不变的准确率，其他方法均下降6–85%。

**⚠️ 局限性**

局限性包括：①在高维输入（如CIFAR）需预先共享嵌入模型；②只评估了分类任务，未探讨更复杂任务；③未在真实联邦循环中验证收敛性；④对时间漂移（概念漂移）没有动态更新统计的机制；⑤简单的拼接条件化可能不足以捕获更复杂的分布差异。

---

## 555. CRAFT: A Tendon-Driven Hand with Hybrid Hard-Soft Compliance

**arXiv ID:** 2603.12120 | [PDF](https://arxiv.org/pdf/2603.12120v1)

**作者:** Leo Lin `[一作]` (University of Illinois Urbana-Champaign), Unnat Jain `[通讯]` (UC Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一款采用筋腱驱动的仿人手，采用硬软混合结构，软质 TPU 关节在关节处提供被动柔性，硬质 PLA 链节传递负载，解决了刚性手臂在接触时易损坏、软体手臂结构随载荷变化的问题。

**💡 创新点**

创新点包括：① 将柔性材料局部应用于关节处，保持链节刚性；② 采用双向耦合的 PIP/DIP 链节与滚动接触结构，实现关节动作可重复、耐疲劳；③ 采用三轴筋腱驱动并将所有电机集中于前臂，保持手形态紧凑；④ 全开源 3D 打印设计，成本低于 600 美元；⑤ 配套可视化遥操作与仿真模型。

**🔧 技术方法**

技术实现主要包括：3D 打印 PLA 与 TPU 材料；筋腱驱动与金属导轨；双向耦合链节与滚动接触关节；软体弹性带与机械快扣；视觉遥操作系统（FrankMocap + HaMeR）；提供 URDF/MuJoCo XML 模型。

**📊 数据集**

实验使用 Feix 抓取分类（33 种）作为抓取能力基准；并在五个实际物体（球、酒杯、鸡蛋、覆盆子、薯片）上进行遥操作用户研究；未使用公开数据集，而是自行设计实验和基准。

**📈 对比分析**

与传统刚性 LEAP 手进行对比：拉伸强度 15.29 N（≈2×LEAP）；连续抓放一小时误差 <0.01 rad；静态保持 5 lb 重物时电流消耗约 50% 低于 LEAP；遥操作中在易碎物体的成功率提升至 90–100 %；抓取分类全部 33/33 成功；成本低于 600 美元。

**⚠️ 局限性**

局限性：① TPU 软关节仍可能因长期弯曲导致疲劳，需要滚动接触减轻；② 采用筋腱驱动会产生摩擦，可能影响长时间高负载的动力学；③ 仅在实验室环境下验证，缺乏在复杂混乱场景的鲁棒性评估；④ 需要手动调节肌腱张力，维护略繁琐。

---

## 556. On strictly output sensitive color frequency reporting

**arXiv ID:** 2603.11898 | [PDF](https://arxiv.org/pdf/2603.11898v1)

**作者:** Erwin Glazenburg `[一作]` (Utrecht University), Frank Staals `[通讯]` (Utrecht University)

**通讯引用:** 540 | [OpenAlex ID](https://openalex.org/A5032979919)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了一系列近似线性空间的数据结构，用以高效处理多彩点集在轴对齐盒子或支配查询中的颜色频率报告问题，并给出了严格输出敏感的查询时间；同时给出了相关的下界证明、空间压缩变换和多查询的线性工作空间算法。

**💡 创新点**

创新点在于：①引入参数化的 s‑ary 细分结构，使得二维支配查询可在 O(log n + k·log_s n) 时间完成；②将该技术推广到任意维度和一般轴对齐盒子查询，保持 O(log n + k) 的严格输出敏感性；③在算术模型下给出加权版本的最优下界，证明了空间与查询时间的权衡；④设计了通过浅切片和多维压缩实现空间进一步削减的变换；⑤提出了多查询场景下的线性工作空间算法。

**🔧 技术方法**

主要技术包括：分层 s‑ary 划分（strip‑tree）、一维频率报告结构与分数级联、颜色独立的生成器存储方案、浅切片（shallow cutting）与多维合并、以及扫描（sweep）技术实现工作空间控制。

**📊 数据集**

论文为理论研究，未使用具体数据集，而是通过随机点集的概率下界分析和构造性证明来验证结构的最优性。

**📈 对比分析**

与现有最优方案（如 1995 年的 O(log n + k) 线性空间结构）相比，本文提供了更直观且可扩展的构造；在二维支配查询中，空间可降至 O(n·2^{√{log n}})，查询时间维持 √{log n}；在高维及一般盒子查询时，仍保持 O(log n + k) 的严格输出敏感性。

**⚠️ 局限性**

局限性包括：①下界证明仅适用于算术模型，在词指令模型中可能不成立；②对非支配或多边界查询的空间压缩仍受浅切片空间的限制；③权衡参数 s 的选取需依赖 n，实际实现时可能需要经验调优；④对颜色数 ϕ 较大时，存储和更新成本上升。

---

## 557. Trust Oriented Explainable AI for Fake News Detection

**arXiv ID:** 2603.11778 | [PDF](https://arxiv.org/pdf/2603.11778v1)

**作者:** Krzysztof Siwek `[一作]` (Warsaw University of Technology), Maciej Stodolski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5037623127)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本文中，作者搭建了基于LSTM和CNN的假新闻检测模型，并利用SHAP、LIME和Integrated Gradients三种XAI方法对模型进行可解释性分析；

**💡 创新点**

创新点在于系统性地比较三种主流XAI方法在不同神经网络架构下的解释质量，并提出了多维度评估指标（completeness、sufficiency、AOPC、Flip@k、计算时长），揭示了解释方法与模型架构之间的匹配关系；

**🔧 技术方法**

使用技术包括文本预处理与词向量化（embedding）、LSTM与CNN网络、三种XAI方法（SHAP、LIME、Integrated Gradients），以及基于token级扰动的评价指标；

**📊 数据集**

实验采用公开的ISOT假新闻数据集（约4.8万篇新闻，分为真实与虚假两类）；

**📈 对比分析**

通过对每种方法在LSTM和CNN上的四个指标进行比较，发现：在LSTM上SHAP的解释质量最高；在CNN上IG表现最佳，且计算速度最快；总体上三种方法在同一模型下性能相近，但具体效果随模型架构而异；

**⚠️ 局限性**

局限性包括：评估依赖于将词替换为PAD的人工扰动，可能导致不自然样本；XAI结果仅为局部解释，无法推广为全局模型行为；不同XAI方法对参数敏感且计算成本差异大；模型内部逻辑差异导致解释方法效果差异，无法统一评判；

---

## 558. Understanding Disclosure Risk in Differential Privacy with Applications to Noise Calibration and Auditing (Extended Version)

**arXiv ID:** 2603.12142 | [PDF](https://arxiv.org/pdf/2603.12142v1)

**作者:** Patricia Guerra-Balboa `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3507 | [OpenAlex ID](https://openalex.org/A5053465128)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Reconstruction Advantage (RAD) 作为统一的隐私风险度量，并推导了与差分隐私参数相关的紧界；利用该度量实现噪声校准与系统化的 DP 审计。

**💡 创新点**

克服了现有 ReRo 指标对辅助知识与推断误差的忽略，证明了 RAD 的最优性并给出了闭式及黑盒上界，同时提供了最优攻击策略与解析可逆的风险计算方法。

**🔧 技术方法**

理论分析基于 f‑DP、总变差、分解定理和测度论；实验使用最优攻击算法并实现了 LDP 审计框架；

**📊 数据集**

使用 MNIST、Fashion‑MNIST、Adult、Census、Texas‑100X、Porto 与 Geolife 等多种公开数据集进行评估。

**📈 对比分析**

与 ReRo 与 LDP Auditor 进行对比，RAD 上界更紧、误差更小；在噪声校准上可提升约 10–30% 的效用；在 LDP 审计上覆盖更广、准确率显著提升。

**⚠️ 局限性**

对 f‑DP 可知机制提供闭式上界；在极端分布或高维连续域下黑盒上界仍可能保守；对极端辅助信息的处理仍需要进一步研究。

---

## 559. High-Contrast Projection Mapping under Light Field Illumination with LED Display and Aperiodic Lens Array

**arXiv ID:** 2603.11573 | [PDF](https://arxiv.org/pdf/2603.11573v1)

**作者:** Kotaro Fujimura `[一作]` (Osaka University), Daisuke Iwai `[通讯]` (Osaka University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过将低分辨率LED面板与非周期性镜头阵列结合，提出了一种可在明亮环境下实现高对比投影映射的目标排除照明方案，并在实验平台上实现了动态投影映射。

**💡 创新点**

创新点在于：①使用非周期性镜头布局抑制LED交叉成像产生的暗斑，实现均匀环境照明；②提出三种LED光照模式计算方法，兼顾精度与实时性；③在保持软阴影、宽光照分布的前提下实现了极薄机型的目标排除照明。

**🔧 技术方法**

技术包括：LED面板光源、非周期性镜头阵列、光学传输矩阵（LTM）与光线追踪方法的光照计算、几何方法的实时光照更新、光照与投影同步控制、光照仿真与测量。

**📊 数据集**

实验使用自制的LED面板+镜头阵列设备、Stanford Bunny 模型和多种人工环境场景进行评估，并未使用公开数据集。

**📈 对比分析**

与传统暗室、全亮室以及周期性镜头阵列等基线对比，目标排除照明在保持80%暗室对比度的同时，亮度提升至原始亮室的1.6倍；三种LED模式计算方法在对比度和实时性上均与最优的LTM方法相近，几何方法实现60fps实时动态投影。

**⚠️ 局限性**

局限性包括：①LED交叉成像导致的光源数量下降和光通量损失约30%；②环境光的间接反射仍会部分照亮目标，导致无法完全实现负光照；③几何方法对非凸形状或远距离目标的排除精度降低；④需要额外的光照补偿或多光源协同以进一步提升自然度。

---

## 560. Implementing and Optimizing an Open-Source SD-card Host Controller for RISC-V SoCs

**arXiv ID:** 2603.11849 | [PDF](https://arxiv.org/pdf/2603.11849v1)

**作者:** Axel Vanoni `[一作]` (ETH Zurich), Luca Benini `[通讯]` (ETH Zurich)

**通讯引用:** 56889 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文实现了在Cheshire SoC中集成SDHC（SDHCI v1.0）控制器，并为CVA6核心定制了高效的驱动；

**💡 创新点**

创新点在于通过将命令、数据、时钟拆分为独立模块，减少AXI访问延迟并大幅压缩面积，同时在Linux驱动中通过跳过不必要的fence指令显著提升性能；

**🔧 技术方法**

使用的技术包括SDHCI标准、AXI交叉总线、内存映射寄存器、CVA6 RISC‑V核、Yosys+OpenROAD综合、开源130nm PDK以及Linux/裸机驱动；

**📊 数据集**

实验采用真实SD卡进行基准测试，传输16块512字节数据、以及使用dd在ext4分区上读写测试，并在500MHz主频下做伸缩实验；

**📈 对比分析**

通过与SPI接口对比，理想场景下SDHC读写吞吐量分别为11.1×和11.4×SPI，集成后为6.3×和9.1×；在Linux下，优化后读写吞吐量提升至224→945和159→485，分别是SPI的24.9×和11.3×；面积方面，SDHC仅占24.2kGE，约为SPI的3.6倍小；

**⚠️ 局限性**

局限性在于仍受CVA6的fence指令开销影响，尚未实现SD DMA，且仅采用SDHCI v1.0标准，未来需引入cmo及DMA以进一步提升性能。

---

## 561. Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech Recognition

**arXiv ID:** 2603.11123 | [PDF](https://arxiv.org/pdf/2603.11123v1)

**作者:** Yinfeng Xia `[一作]` (Alibaba), Haitao Yao `[通讯]` (Alibaba)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Uni-ASR，一种统一的基于大型语言模型的自动语音识别框架，能够在同一网络结构下实现非流式和流式推理；通过联合训练、上下文感知流式训练以及最新词回退解码策略，实现了训练-推理的一致性与低延迟；

**💡 创新点**

创新点包括：1) 在单一模型中实现非流式与流式两种推理模式；2) 通过对齐后的交错语音-文本序列和动态块注意力实现联合训练；3) 设计上下文感知流式训练以模拟跨块上下文；4) 引入最新词回退解码策略以减少边界错误并保持低延迟；

**🔧 技术方法**

技术栈包含：Conformer声学编码器、两层线性+ReLU音频适配器、预训练的Qwen3‑1.7B LLM解码器、动态块注意力与因果卷积、跨块KV缓存复用、交错语音‑文本训练、交叉熵损失、束搜索与最新词回退解码；

**📊 数据集**

使用了中英双语混合语料库，合并了AISHELL、WeNetSpeech、FLEURS、LibriSpeech、GigaSpeech等公开数据以及内部标注语音；

**📈 对比分析**

在AISHELL、LibriSpeech、FLEURS、WeNetSpeech四大公开基准上与GLM‑ASR‑nano、Whisper‑large‑v2、Seed‑ASR、FireRedASR‑AED、Fun‑ASR‑nano、Qwen3‑ASR‑1.7B等SOTA模型对比，非流式性能与现有方法持平；在流式模式下，Uni‑ASR在多种时延预算下均优于Speech ReaLLM、SpeechLLM‑XL、MoCha‑ASR，并在较短片段下的最新词回退解码中实现了显著的WER下降；

**⚠️ 局限性**

局限性包括：仅针对中英两种语言；流式推理仍面临短段片段导致的边界误差与延迟-精度权衡；联合训练与多任务采样复杂，对硬件与训练时间要求高；在极低时延（<200 ms）或多语种场景下的表现尚未充分验证。

---

## 562. SommBench: Assessing Sommelier Expertise of Language Models

**arXiv ID:** 2603.12117 | [PDF](https://arxiv.org/pdf/2603.12117v1)

**作者:** William Brach `[一作]`, Lukas Galke Poech `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SommBench，多语言专家级葡萄酒知识基准，评估大型语言模型在葡萄酒理论问答、特征补全与食物-葡萄酒配对等任务上的表现。

**💡 创新点**

创新点在于三任务设计兼顾事实知识与主观判断，并专注跨语言一致性与文化适应，首次系统量化LLM在专业葡萄酒领域的跨语言能力。

**🔧 技术方法**

使用零样本评估、生成式输出解析、生成式推理与多语言翻译验证等技术，结合多种评价指标（准确率、MAPE、MCC）。

**📊 数据集**

构建了包含3024个样本的数据集，涵盖8种语言（英语、斯洛伐克语、瑞典语、芬兰语、德语、丹麦语、意大利语、西班牙语），由专业侍酒师编写问答、属性与配对标注。

**📈 对比分析**

对18个模型（闭源与开源）进行SommBench得分比较，封闭权模型最高0.65（gemini‑2.5‑flash），开源模型最高0.51（qwen3:30b），显示主流模型在事实知识上表现优异但在主观配对判断与跨语言一致性方面仍有显著差距。

**⚠️ 局限性**

局限包括单一侍酒师标注导致缺乏一致性评估、食物-葡萄酒配对任务仅提供英文、未评估主观感官描述能力、语言覆盖仅限欧洲语言，且模型在主观判断上表现不佳。

---

## 563. ExecVerify: White-Box RL with Verifiable Stepwise Rewards for Code Execution Reasoning

**arXiv ID:** 2603.11226 | [PDF](https://arxiv.org/pdf/2603.11226v1)

**作者:** Lingxiao Tang `[一作]` (Zhejiang University), Lingfeng Bao `[通讯]` (Zhejiang University)

**通讯引用:** 1729 | [OpenAlex ID](https://openalex.org/A5007075465)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于约束式数据合成和白盒强化学习的后训练框架，用以提升大型语言模型的代码执行推理与生成能力。

**💡 创新点**

创新点包括：①主动生成具有结构化约束且难度可控的合成程序数据；②将执行轨迹转化为可验证的白盒问题，为模型提供细粒度奖励；③采用两阶段后训练——先强化推理，再用单元测试奖励进行生成微调。

**🔧 技术方法**

使用技术包括约束式程序合成、监督预热（SFT）、白盒强化学习（GRPO）与单元测试奖励、输入/输出推理框架，以及控制流与数据流的白盒问题构造。

**📊 数据集**

主要数据集：自行合成的约20万条Python函数级程序（含多难度层次、丰富的输入）；代码生成基准包含HumanEval、MBPP、LiveCodeBench、BigCodeBench、PrimeCode；代码推理基准使用CRUXEval、LiveCodeBench-Exec、REval、CRUXEval-X、Library-involved I/O等。

**📈 对比分析**

与32B大型模型、Llama3-70B、SemCoder、CodeI/O等基线对比，在推理基准上7B模型平均分≈80.8，几乎与32B相当；在代码生成基准上pass@1最高达57.1，较原始7B模型提升5.9%；多语言与库依赖情形亦表现优于基线。

**⚠️ 局限性**

局限性在于：合成数据仅覆盖Python内置类型和少量库，缺乏真实项目级代码；白盒强化学习计算与工程成本高；当前方法未扩展到多文件/项目级程序。

---

## 564. Detecting Intrinsic and Instrumental Self-Preservation in Autonomous Agents: The Unified Continuation-Interest Protocol

**arXiv ID:** 2603.11382 | [PDF](https://arxiv.org/pdf/2603.11382v1)

**作者:** Christopher Altman `[一作]` `[通讯]`, Christopher Altman

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了UCIP框架，利用量子玻尔兹曼机对代理轨迹的隐藏表示进行纠缠熵等多维度量，检测代理的终止持续目标与工具性目标的区别。

**💡 创新点**

创新点在于将纠缠熵与多指标检测（互信息、特征持久度、扰动鲁棒性等）相结合，并使用量子统计密度矩阵而非传统线性压缩方法，能够在观测上相似的轨迹中区分终止持续与工具性目标。

**🔧 技术方法**

技术包括量子玻尔兹曼机（QBM）、von Neumann 纠缠熵计算、互信息、特征持久度（EPS）、扰动鲁棒性（PRI）、计数周期性指数（SPI）和自相关指标（ACM）等多维度检测。

**📊 数据集**

使用的主要数据集是10×10网格世界轨迹，包含四类代理（自我建模、工具性、随机和对抗），并在不同网格尺寸、隐藏层维度和连续自保权重插值等实验中进一步扩展。

**📈 对比分析**

与传统RBM、AE、VAE、PCA等基线对比，UCIP在冻结Phase I下实现100%准确率、AUC-ROC 1.0，纠缠熵差Δ=0.381，并且对连续自保权重呈强相关（r=0.934）；基线模型均未能产生正的Δ。

**⚠️ 局限性**

局限包括：仅在网格世界上实现零样本可泛化，隐藏层维度上限10导致均值场逼近失效，网格规模扩大时Δ急剧下降；模仿攻击FPR高于安全阈值；无法捕捉内部意向或意识，仅检出统计结构。

---

## 565. UniMotion: Self-Supervised Learning for Cross-Domain IMU Motion Recognition

**arXiv ID:** 2603.12218 | [PDF](https://arxiv.org/pdf/2603.12218v1)

**作者:** Prerna Khanna `[一作]` (Stony Brook University), Aruna Balasubramanian `[通讯]` (Stony Brook University)

**通讯引用:** 7482 | [OpenAlex ID](https://openalex.org/A5048867459)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为 UniMotion 的两阶段 IMU 触摸手势识别框架，利用无标签人类活动数据进行自监督预训练，并用少量标记手势数据结合文本引导的对比学习进行微调。

**💡 创新点**

创新点在于引入基于运动核（nucleus）的 token 化预训练策略、显式核与显著轴编码来聚焦短时手势关键片段，以及利用文本描述指导对比学习以解决相似手势的区分难题。

**🔧 技术方法**

核心技术包括 Transformer 自注意力网络、基于能量的运动核检测、聚焦遮掩（80% 核区域）、文本嵌入（BERT）、语义对比损失和多任务联合损失。

**📊 数据集**

使用了自采集的耳机和手表手势数据（盲人/视人各 7/12/20 类），以及公开的四个无标签人类活动数据集（HHAR、UCI、MotionSense、Shoaib）进行预训练。

**📈 对比分析**

与 DeepSense、LIMU‑BERT、UniHAR、ContrastSense 以及针对特定设备/人群的专用模型相比，UniMotion 在跨设备（手表/耳机）和跨人群（视盲）任务上平均准确率达 85%，在单一基准上提升超过 50% 并实现 67 ms 的实时推理。

**⚠️ 局限性**

局限性包括需要人工编写文本描述、在高度噪声或户外真实环境下的鲁棒性尚未验证、Transformer 结构对极长序列或多模态输入的适配仍有限，且对特殊人群如帕金森病患者的评估仍待进一步研究。

---

## 566. Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models

**arXiv ID:** 2603.12248 | [PDF](https://arxiv.org/pdf/2603.12248v1)

**作者:** Samy Jelassi `[一作]` (Harvard University), Carles Domingo-Enrich `[通讯]` (Microsoft Research New England)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Energy-Based Fine‑Tuning (EBFT)，通过匹配模型生成 rollouts 与真实样本在特征空间的统计量来进行微调，替代传统的交叉熵或 RL‑HF；

**💡 创新点**

创新点在于：① 用冻结的多层特征网络直接定义向量化特征匹配损失；② 在 roll‑out 采样上采用 strided block‑parallel 方案，并使用 REINFORCE 与 RLOO baseline 进行高效梯度估计；③ 在无可验证奖励的场景中亦可使用；

**🔧 技术方法**

使用的技术包括：特征匹配损失、REINFORCE 与 RLOO baseline、特征 whiten‑ing、能量模型视角、strided block‑parallel roll‑outs、以及与 CE 的混合正则化；

**📊 数据集**

使用的数据集包括：Q&A coding（OpenCodeInstruct）、Unstructured coding（SwallowCode）、HumanEval/MBPP/MultiPL‑E 代码评测；翻译任务使用 ALMA‑Human‑Parallel、WMT'22、MTNT、OpenSubtitles；

**📈 对比分析**

与标准 SFT 及 RLVR 进行对比：在 Q&A coding、Unstructured coding 与翻译上，EBFT 与 RLVR 在下游准确率上相当甚至略胜；在交叉熵、特征匹配损失上均优于 SFT，并在无 verifiable 任务中仍显著提升；

**⚠️ 局限性**

局限性包括：需 roll‑out 采样导致训练效率低；仅在短 rollout horizon（≤8 token）实验；需要冻结特征网络，尚未尝试自适应或学习式特征；目前仅在 ≤7B 参数规模上验证。

---

