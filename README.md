# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-10 | 今日论文总数: 425

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Threshold Authorization Without Threshold Signatures: Signature-Agnostic MPC Custody

**arXiv ID:** 2607.08226 | [PDF](https://arxiv.org/pdf/2607.08226v1)

**作者:** Dariia Porechna `[一作]` `[通讯]` (EternaX Labs), Dariia Porechna (EternaX Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了Dual‑Gate架构，分离成员签名认证与阈值授权，通过Shamir共享的密封实现多方批准并支持后量子签名迁移；

**💡 创新点**

创新点在于：1）把签名与阈值封装分离，避免对签名本身做阈值化；2）使用一次性密封并保持信息理论下的阈值保密；3）签名方案成为部署参数，迁移仅是密钥轮换；4）可在智能合约、HSM或链共识层实现；

**🔧 技术方法**

采用Shamir Secret Sharing、Wegman–Carter MAC、哈希承诺、EUF‑CMA签名（ECDSA/EdDSA/SLH‑DSA/ML‑DSA）、VSS/JRSS、拉格朗日插值以及Fp域算术；

**📊 数据集**

未使用公开数据集，主要通过仿真和演示实现验证；

**📈 对比分析**

通过与阈值ECDSA、链上多重签名和离线双控制模式对比，证明单轮操作（签名+一次字段运算）低延迟、易扩展；预共享一次性槽仅需约12 MB（n=20,B=10⁴）并可批量化；相较传统阈值签名省去多轮MPC；

**⚠️ 局限性**

限制包括：输出不是原生签名；需在执行层实现双门逻辑；需要一次性槽管理和最终性保障；依赖VSS/JRSS设置；无法隐藏批准者身份；在混合键方案下安全性受最低签名方案限制。

---

## 2. When Does Continual Learning Require Learning

**arXiv ID:** 2607.07847 | [PDF](https://arxiv.org/pdf/2607.07847v1)

**作者:** Anne Harrington `[一作]` (UC Berkeley), Yutong Bai `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一个统一的持续学习框架，将空间与时间两轴（域迁移、事实更新、时间漂移、代理累积）纳入同一评估协议，系统比较八种更新方法在 Qwen3‑8B 上的表现。

**💡 创新点**

创新点在于将持续学习视为“随世界变化而提升模型能力”而非仅限于“防止灾难性遗忘”，并统一定义可比的评估协议，揭示不同环境变化对应的最佳更新策略。

**🔧 技术方法**

使用的技术包括提示优化（GEPA、ACE）、监督微调与自蒸馏（SFT、SDFT）、在线强化学习（GRPO、SDPO）、以及上下文压缩/记忆扩展（Cartridges、In‑place TTT）。

**📊 数据集**

使用的数据集包括 ToolUse、FinQA、SciKE‑Bio（空间迁移）、四个月度 Wikipedia 剪影（事实更新）、10‑K 股票涨跌预测（噪声时间漂移）、WebArena‑Infinity 代理任务（代理累积）。

**📈 对比分析**

在统一协议下通过遗忘矩阵、BWT/FWT 指标比较方法；提示方法快速适配但后续显著遗忘；蒸馏/强化学习在稳定性和前向迁移上表现更好；记忆压缩提升效率但对新任务提升有限；不同任务中各方法表现差异明显。

**⚠️ 局限性**

限制在于仅用单一模型 Qwen3‑8B 评测，评估范围仅覆盖四种环境变化，结果对更大模型或推理模式可能不适用，且缺少在更复杂、开放式部署环境中的验证。

---

## 3. Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition

**arXiv ID:** 2607.08374 | [PDF](https://arxiv.org/pdf/2607.08374v1)

**作者:** Jing Jie Tan `[一作]` (Universiti Tunku Abdul Rahman), Anissa Mokraoui `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种理论中立的个性识别框架 JAM，利用图原型网络、跨理论和谐化以及 LLM 作为评判者，实现对文本中的潜在心理“伪面向”进行学习与推断。

**💡 创新点**

创新点在于：① 通过跨理论和谐化（人类指导链接 + 机器一致性）把不同心理理论的数据统一映射到共享潜在空间；② 采用 LLM‑as‑a‑Judge 在训练前/训练中动态过滤噪声样本；③ 用注意力池化的图原型网络捕捉层级表示并聚类伪面向。

**🔧 技术方法**

技术包括 Longformer 作为嵌入背骨、全连接图神经网络 + 注意力池化、原型学习 (Meta‑Learning)、人类与机器的跨理论对齐、LLM 评判（ChatGPT/ Qwen/ Llama/ DeepSeek）以及 QLoRA 微调。

**📊 数据集**

使用 Essays（Big‑5 1578/395/494）和 Kaggle（MBTI 5552/1388/1735）两个公开数据集进行训练与评估。

**📈 对比分析**

与传统的交叉熵、原型学习、仅人类指导或仅机器一致性相比，JAM 在 Essays 上平均提升 12% BA、Kaggle 上提升 14% BA；LLM‑before‑the‑loop 通常比 LLM‑in‑the‑loop 更快收敛、性能更佳，整体表现显著优于先前工作。

**⚠️ 局限性**

局限性包括：只验证了两种心理理论；对跨文化多样性及更大规模数据的鲁棒性未知；在敏感个人数据下的隐私与公平性问题仍待解决，后续需探索联邦学习等更安全的分布式训练方式。

---

## 4. MLQENABLER: Enabling Secure Machine Learning Queries over Encrypted Database in Cloud Computing

**arXiv ID:** 2607.08197 | [PDF](https://arxiv.org/pdf/2607.08197v1)

**作者:** Xu Zhou `[一作]` (Michigan Technological University), Xinyu Lei `[通讯]` (Michigan Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种ML Queries Enabler方案，在云端对加密数据库实现安全机器学习查询，利用索引辅助方法在保持数据加密的同时支持模型训练和推理。

**💡 创新点**

创新点在于：①使用生成对抗网络（GAN）生成与随机数据不可区分的安全索引，从而实现索引隐私；②引入重构器与重构损失，使安全索引可恢复原数据，保证ML实用性；③通过全局密钥和本地加密实现机密性与索引隐私并行，提供模型隐私保护。

**🔧 技术方法**

技术手段包括：AES/3DES本地加密；GAN（生成器+判别器）生成安全索引；重构器R与重构损失；CNN ResNet‑18与Transformer SwinV2‑T模型；使用对抗损失和重构损失训练GAN。

**📊 数据集**

实验使用图像数据集：DIV2K、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、GTSRB、CelebA。

**📈 对比分析**

与传统FHE、DP、FL等方法相比，AA（授权准确率）下降仅0.07%–6.13%（平均约1–3%），UA（未授权准确率）接近随机猜测，模型隐私得到保证；在同等安全级别下实现轻微的性能下降，空间成本加倍。

**⚠️ 局限性**

局限性：需额外存储安全索引导致空间成本翻倍；索引隐私与ML准确率之间存在权衡；目前仅验证图像数据，对文本或结构化数据的适用性待进一步研究；对极端攻击场景（如高级对抗攻击）分析不充分。

---

## 5. LLT: Local Linear Transformer for PDE Operator Learning

**arXiv ID:** 2607.07718 | [PDF](https://arxiv.org/pdf/2607.07718v1)

**作者:** Oded Ovadia `[一作]` (Tel Aviv University), Eli Turkel `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 Local Linear Transformer（LLT）的新型神经算子，用于学习 PDE 解决方案映射，兼顾全局长程通信与局部空间混合，兼容结构化和非结构化网格。

**💡 创新点**

创新点在于：①将核化线性注意力与显式局部混合路径（卷积或邻域注意力）结合，显著降低 O(N²) 的注意力成本；②加入坐标与几何编码，增强对网格位置与几何形状的敏感性；③保持统一的编码‑解码框架，可直接应用于结构化和非结构化网格，提升模型的通用性。

**🔧 技术方法**

技术细节包括：Transformer 编码器块、RMSNorm、SwiGLU 前馈网络、核化线性注意力（使用正激活 φ(z)=ELU(z)+1）、局部卷积或半遮掩邻域注意力、坐标 Fourier 编码、距离编码、跳连解码器。

**📊 数据集**

实验数据集：5 个二维基准 PDE 问题（弹性、塑性、机翼、管道、达西流）来自 Geo-FNO 与 Transolver，及 32,186 点的 ShapeNet 3D 车身气动数据；所有数据均由有限元、有限体积或差分求解器生成。

**📈 对比分析**

与多种基线（FNO、U-FNO、geo-FNO、U-NO、F-FNO、LSM、LNO、Galerkin、HT-Net、OFormer、GNOT、FactFormer、ONO、Transolver、Transolver++）对比。LLT 在 Elasticity、Plasticity、Airfoil、Darcy 上取得最低相对 L₂ 误差，Pipe 问题与 Transolver++ 相近。训练迭代时间比 Transolver 提升 1.78×–2.45×（结构网格）或 2.05×–4.14×（问题特定配置），峰值内存基本相当。

**⚠️ 局限性**

局限性：①在 Pipe 这类平滑、结构化流场中仍略逊于 Transolver++；②目前仅做监督学习，缺乏物理约束或自回归时间步推演；③在极大规模 3D 任务或多尺度问题中对线性注意力的表达能力与可扩展性尚未完全验证。

---

## 6. Aligning Clinical Needs and AI Capabilities: A Survey on LLMs for Medical Reasoning

**arXiv ID:** 2607.07761 | [PDF](https://arxiv.org/pdf/2607.07761v1)

**作者:** Qi Peng `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文综述了医学大语言模型（LLM）的推理能力，提出双视角（临床与计算）框架，构建覆盖五级医学推理能力的统一基准，并对18个最新模型进行系统评估。

**💡 创新点**

创新点在于：①将Miller金字塔扩展为5级，映射到推理类型（演绎、归纳、溯因、混合）；②构建统一的5级医学推理基准；③对通用与医学专用模型进行分层对比；④系统阐述幻觉、数据缺口、知识融合等挑战与未来方向。

**🔧 技术方法**

采用的技术包括：Chain-of-Thought / Long-CoT 推理、搜索引导（MCTS）、检索增强（RAG）、多模态融合、代理推理；指令调优、强化学习、知识图谱检索；评估指标涵盖准确率、推理完整性、证据利用、可信度与不确定性处理。

**📊 数据集**

使用的数据集：CADEC、UMLS、MedNorm、CBLUE、NCBI Disease、MIMIC-IV、ChestX‑ray14、MedKBQA、MedAgentsBench、PubMedQA、MultiMedQA、HealthBench 等；总共5,000样本按五级平衡构建。

**📈 对比分析**

比较方法：针对每级任务设置对应指标，按级别汇总并与人类专家对标；对18个模型按参数规模、专业化与推理技术分类；实验结果表明医学专用模型在诊断层面领先，通用大模型在决策支持、对话、摘要等方面更强；统计显著性检验显示Level 4、5差异显著。

**⚠️ 局限性**

局限性：①数据稀缺、标注成本高且专家一致性低；②知识表示与融合困难，指南、文献难以结构化；③模型幻觉、置信度失真与缺乏可追溯证据；④推理深度不足，因果与不确定性处理有限；⑤执行与格式化可靠性不稳定，影响临床可用性。

---

## 7. Sampling on Random Subspaces under Limited Data in the Context of Exploratory Landscape Analysis

**arXiv ID:** 2607.07854 | [PDF](https://arxiv.org/pdf/2607.07854v1)

**作者:** Iván Olarte Rodríguez `[一作]` (Leiden University), Elena Raponi `[通讯]` (Leiden University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在预算受限的情况下，提出了利用随机线性嵌入在低维子空间中采样来改进探索性景观分析（ELA）的研究。

**💡 创新点**

创新点在于把采样策略从全空间转移到随机低维子空间，并对不同压缩比例进行系统评估，以提升特征稳定性。

**🔧 技术方法**

采用随机线性嵌入、Latin Hypercube Sampling、ELA特征计算及Wasserstein-1距离评估。

**📊 数据集**

使用BBOB（COCO）测试套件的20维噪声自由函数，取前15个实例。

**📈 对比分析**

通过与全空间采样及多种子空间采样进行比较，使用Wasserstein-1距离和平均排名，结果显示在中等压缩（r=0.5）时，子空间采样在多类特征上与甚至优于全空间采样。

**⚠️ 局限性**

局限性包括仅在20维和BBOB函数上测试，低维子空间可能无法捕捉全局结构，且某些基于全局特征的ELA集在子空间采样下表现不佳。

---

## 8. GitLake: Git-for-data for the agentic lakehouse

**arXiv ID:** 2607.08319 | [PDF](https://arxiv.org/pdf/2607.08319v1)

**作者:** Weiming Sheng `[一作]` (Columbia University), Luca Bigon `[通讯]` (Bauplan Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了面向数据湖的 Git‑for‑Data 框架，让单表快照演变为可全湖使用的提交、分支、合并和回滚，支持代理（agent）在隔离分支上迭代，人工审核后统一发布。

**💡 创新点**

创新点包括：① 将 Git 原语（提交、分支、合并）迁移到 OLAP 场景；② 引入事务性分支（transactional branches）实现多表流水线的原子发布；③ 通过 Alloy 轻量级形式化模型验证核心语义并发现一致性反例；④ 结合 copy‑on‑write、元数据中心化和 API 设计，实现极低成本的分支、合并与回滚；⑤ 公开的 Python/CLI SDK 与 Rust 核心共享，支持代理与人类协同。

**🔧 技术方法**

技术栈主要包括 Apache Iceberg（单表 ACID 快照）、PostgreSQL 作为元数据目录、S3 存储、Rust 开发的 CLI 与共享核心、Python SDK、轻量级 Alloy 模型、事务性 API、copy‑on‑write 存储优化与自动化流水线执行。

**📊 数据集**

实验以生产系统为主：已在真实业务环境中运行了数百万条作业、数十万条数据分支，实际工作负载来自企业的 OLAP 与 AI 数据管道；未使用公开数据集，而是利用内部业务数据进行性能和正确性评估。

**📈 对比分析**

与 Snowflake（zero‑copy clone）和 Databricks（shallow copy）比较，分支创建平均 80 ms（p95≈80 ms），性能提升约 100×；事务性流水线合并在单表事务上实现原子发布；系统在高并发下保持低延迟，分支复制几乎无成本。评测指标包括分支创建时间、合并冲突率（约每 100k 次冲突 10 次）和流水线失败恢复时间。

**⚠️ 局限性**

限制主要体现在：① 事务性分支与嵌套分支的正确性冲突，易产生不一致状态；② 随着代理探索速度提升，人工审查瓶颈突出；③ 当前系统仍依赖人类验证，代理对湖库细节的理解有限；④ 需要进一步完善多语言流水线事务性保障与冲突解决机制。

---

## 9. AI-Driven Thermal Mapping and Management in 3D Integrated Photonic Circuits

**arXiv ID:** 2607.07711 | [PDF](https://arxiv.org/pdf/2607.07711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 10. PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction

**arXiv ID:** 2607.08111 | [PDF](https://arxiv.org/pdf/2607.08111v1)

**作者:** Wanyi Ning `[一作]` (Yijiahe AI), Yiming Cheng `[通讯]` (Yijiahe AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种代理监督联合训练框架PS4，直接利用真实对话录音训练目标说话人提取模型；

**💡 创新点**

创新点包括构建大规模代理监督训练语料REAL-PS4以及联合使用四种可微代理监督目标（ASR交叉熵、说话人相似度、VAD、DNSMOS）进行微调；

**🔧 技术方法**

采用BSRNN分离器、冻结的ECAPA‑TDNN说话人编码器、Whisper做ASR教师、ResNet34说话人嵌入、DNSMOS可微实现和cosine‑margin ranking loss等技术；

**📊 数据集**

使用四个公开会议/对话数据集AISHELL‑4、AliMeeting、AMI、CHiME‑6构建REAL‑PS4训练集，并在REAL‑T benchmark（含五个子数据集）上进行评估；

**📈 对比分析**

在REAL‑T开发集和官方验证集上与两种BSRNN基线相比，PS4在所有子数据集上TER下降、SIM和DNSMOS提升；在leaderboard上排名第二，总分3.25，获得最高F1（0.871）和SIM（0.565）分数；

**⚠️ 局限性**

局限性包括仍需代理监督，缺乏干净目标语音；DNSMOS‑P808得分略低于顶位；依赖预训练模型和冻结的说话人编码器，可能在极端噪声或不同设备场景下表现受限。

---

## 11. Open-ended Multi-agent Autocurricula via Visual Inspection of Policies with Multi-modal LLMs

**arXiv ID:** 2607.08193 | [PDF](https://arxiv.org/pdf/2607.08193v1)

**作者:** Lorenzo Pantè `[一作]` (Sapienza University of Rome), Roberto Capobianco `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视频语言模型（VLM）的视觉政策检查方法VIP，用来在强化学习中自动生成开放式自适应训练课程；

**💡 创新点**

创新点在于直接利用训练过程中的视频行为作为多模态输入，让VLM评估并推荐最具挑战性的任务，而非仅依赖数值得分或文本摘要；

**🔧 技术方法**

采用VideoLLaMa2-7B VLM、句子相似度模块、MAPPO多智能体强化学习算法以及强化学习训练与评估脚本；

**📊 数据集**

在StarCraft Multi-Agent Challenge（SMAC）多智能体环境上进行实验，使用不同地图和难度级别的任务空间；

**📈 对比分析**

与文本仅版VIP、随机课程、MAPPO单独训练以及两种基于标量得分的Unsupervised Environment Design（PLR）进行对比，VIP在未见地图的微调任务中实现约80% 的胜率，明显优于基线；

**⚠️ 局限性**

局限性包括对VLM性能的依赖、任务空间必须可渲染视频、VLM推理成本（虽然仅占1% 但仍不适合极大规模任务）以及对任务空间可枚举性的假设。

---

## 12. Directed proof-relevant logical relations in simplicial HoTT

**arXiv ID:** 2607.08154 | [PDF](https://arxiv.org/pdf/2607.08154v1)

**作者:** Runming Li `[一作]`, Robert Harper `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在简化的指向同伦型理论（simplicial HoTT）框架下构建了指向证明相关逻辑关系模型，并利用该模型证明了布尔类型的指向可计算性（directed Boolean canonicity），随后将该方法推广到依赖类型、宇宙和二元参数化。

**💡 创新点**

创新点包括：① 将指向不等式（inequality types）引入等式语义，形成“指向的分型”等式；② 发现并利用对偶（contravariant）族的合取性来实现证明相关的“闭包扩张”，从而避免传统闭包展开证明的繁琐；③ 通过平坦模态（flat modality）实现类型转换和宇宙判定的离散化；④ 在二元逻辑关系中分离垂直的指向缩放与水平的参数化，得到新的参数化证明。

**🔧 技术方法**

主要技术手段有：
- 简化同伦型理论（simplicial HoTT）中的不等式类型与方向间隙；
- 指向分型（directed quotient inductive-inductive types）构造初始语法；
- 对偶族（contravariant families）提供的指向展开与可组合性；
- 通过平坦模态（flat modality）实现类型的离散化与宇宙的负式实现；
- 基于gluing的证明相关逻辑关系模型，用于证明可计算性与参数化。

**📊 数据集**

本文没有使用任何实验数据集；所有结果均为理论证明与形式化验证（已在Cubical Agda 证明原型）。

**📈 对比分析**

本文不进行实验性能比较；其贡献在于理论框架和形式化证明。与传统基于等式的证明相关逻辑关系方法相比，本文在保持同构的同时，降低了对等式传输的依赖，并通过指向不等式和对偶族实现更直接的闭包展开。

**⚠️ 局限性**

局限性包括：
- 目前的机械化实现仅覆盖简单类型语言，依赖类型与宇宙的完整实现仍在开发中；
- 对偶族与平坦模态的使用使得证明过程高度依赖于 simplicial HoTT 的可用性与证明助手的支持；
- 由于不等式缺乏对称性，某些传统的等式归约规则需要额外的离散化步骤；
- 论文中未讨论更复杂的类型构造（如更高阶递归类型）与归约策略的可扩展性。

---

## 13. SHIFT: Survival Prediction from Incomplete and Heterogeneous Genomic Data

**arXiv ID:** 2607.07725 | [PDF](https://arxiv.org/pdf/2607.07725v1)

**作者:** Muhammet Sami Yavuz `[一作]` (Technical University of Munich), Jana Lipkova `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 SHIFT，一个能够在基因组测序面板差异导致的结构缺失下直接进行生存预测的 Transformer 模型；

**💡 创新点**

创新点在于将缺失感知的自注意力与可变率特征遮蔽（VRM）相结合，使模型能够在不做任何补值的情况下在多中心、不同测序面板的环境中迁移；

**🔧 技术方法**

采用的技术包括基于 SNN 的特征嵌入、Mask Self‑Attention Transformer、离散时间生存损失函数以及 VRM 训练策略；

**📊 数据集**

使用的公开和私有数据集为：Glioblastoma（GBM）的 TCGA、CPTAC 及德国机构队列，Lung Squamous Cell Carcinoma（LUSC）的 TCGA、CPTAC 及美国机构队列；

**📈 对比分析**

与 XGBoost‑Cox、CoxPH、Random Survival Forest、DeepSurv、DeepHit、SNN 等传统与深度基线模型以及 KNN/均值补值方法比较，SHIFT‑VRM 在完整特征下的 Ens‑All C‑index 通常高于 0.55，且在严重缺失（US LUSC 仅 22/197 基因）下也能达到 0.57，显著优于大多数补值基线；

**⚠️ 局限性**

局限性包括仅在两种癌症和有限样本量上验证；缺失遮蔽采用随机策略，未模拟真实机构的固定面板缺失；仅评估判别力（C‑index）未进行统计显著性检验；未考虑多模态输入或更广泛的基因组特征。

---

## 14. Prismata: Confining Cross-Site Prompt Injection in Web Agents

**arXiv ID:** 2607.08147 | [PDF](https://arxiv.org/pdf/2607.08147v1)

**作者:** Corban Villa `[一作]` (UC Berkeley), Raluca Ada Popa `[通讯]` (UC Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Prismata，一种针对 Web 代理的上下文最小权限防御框架，能够在不依赖开发者注解的前提下自动对页面内容进行信任标签化并限制代理行为。

**💡 创新点**

创新点在于结合 Biba 完整性模型的“无下读、无上写”原则，推出 Biba 解析技术，利用 LLM 对页面关键路径进行安全决策，同时配合机械限制层，实现对未受信任内容的动态过滤与权限下调。

**🔧 技术方法**

核心技术包括：基于 LLM 的动态信任推导、动作门（Action Gate）与 Biba 解析、机械限制（Mechanical Confinement）实现观察与动作的过滤、以及缓存机制降低推理延迟。

**📊 数据集**

使用 Common Crawl（前 10k 域名页面）、Mind2Web（真实交互脚本）和 WebArena 作为评测数据集，构成 5,664 个 DOM 样本进行标签验证与攻击实验。

**📈 对比分析**

与基线 Web 代理相比，攻击成功率从 85.5% 降至 0.7%，在攻击场景下任务完成率提升约 5.1 倍；在无攻击场景下保持 26.6% 的任务成功率，且精确度/召回率/F1 在 90% 以上。

**⚠️ 局限性**

局限性包括：对结构化提示的依赖导致极少数路径（≈0.1%）仍可能被利用、仅覆盖文本模式攻击不涵盖图像/多模态攻击、LLM 标签准确度与模型成本平衡、实时 DOM 漂移与非文本攻击仍未完全解决。

---

## 15. LAP: Simple Command-line Tools for Teaching Logic, Algorithms, and Proof in Computer Science

**arXiv ID:** 2607.08000 | [PDF](https://arxiv.org/pdf/2607.08000v1)

**作者:** Stephen F. Siegel `[一作]` (University of Delaware), Yuxin Zhou `[通讯]` (University of Delaware)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于Java的命令行工具集（LAP），用于教学命题逻辑和一阶逻辑，提供公式转换、SAT求解、自然演绎推导检查以及多种可视化视图；

**💡 创新点**

创新点在于将算法实现与教学可读性结合，强调从计算角度学习逻辑；提供可视化的推理检查与多视图展示同一数据结构；工具无外部依赖、易读、易调试，且支持交互式教学与自动化批改；

**🔧 技术方法**

使用Java实现，采用递归数据结构与递归算法；通过JavaCC生成语法分析器；实现了DPLL、Tseytin、CNF/DNF转换等逻辑算法；提供命令行界面与多种输出格式；

**📊 数据集**

未使用正式数据集，所有测试均基于手工编写的示例推理文件；

**📈 对比分析**

论文未给出系统性性能比较或基准测试，主要强调教育易用性与算法可视化；但提供 -v 选项可观察算法步骤；在大规模公式下的性能未做评估；

**⚠️ 局限性**

局限性包括：未支持等式、一阶理论、Skolem/Herbrand 形式；对大规模公式的性能评估缺失；仅提供命令行界面，缺少图形化IDE；在 PL 与 FOL 模块之间存在一定重复工作。

---

## 16. Adversarial Decoys: Misdirecting Attention-Based Defenses in ViT

**arXiv ID:** 2607.07922 | [PDF](https://arxiv.org/pdf/2607.07922v1)

**作者:** Giulia Marchiori Pietrosanti `[一作]` (Sant'Anna School of Advanced Study), Giorgio Buttazzo `[通讯]` (Sant'Anna School of Advanced Study)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了一种名为“adversarial decoys”的攻击技术，通过在视觉Transformer（ViT）中添加独立优化的解码器补丁，来引导模型注意力分布，从而误导基于注意力的防御并保持原始攻击效果。

**💡 创新点**

创新点在于：①将注意力操纵与目标误分类任务解耦，采用层级目标‑占优比的损失实现跨层注意力排名的显著提升；②使用独立优化的补丁实现攻击无关的注意力偏移，兼容多种局部攻击方法；③通过对比实验展示了注意力强度与攻击成功并不完全对应。

**🔧 技术方法**

核心技术包括：视觉Transformer的自注意力机制、基于softmax的注意力得分提取、层级目标‑占优比损失（target‑dominance ratio）以及投影梯度下降（Projected Gradient Descent）对补丁进行优化；实验中结合了PatchFool与标准对抗补丁攻击，并与ARMRO等注意力阈值掩码防御对比。

**📊 数据集**

实验数据集为ImageNet验证集（随机抽取1024张图），使用了DeiT‑B/16-224、ViT‑B/16-224和ViT‑S/16-224三种ViT架构进行评估。

**📈 对比分析**

与ARMRO防御的对比实验显示，在未加入decoy时，ARMRO能将攻击后的准确率恢复至接近干净图像水平；加入decoy后，受防御的准确率下降超过40%，说明decoy能显著削弱注意力掩码的效能；实验亦表明在预算匹配下，将攻击预算分配给decoy比单纯增大攻击强度更能降低防御准确率。

**⚠️ 局限性**

主要局限包括：①decoy需要额外的标记空间，过多补丁可能导致攻击更易被察觉；②在某些ViT层中，即使采用层级调节，decoy仍难以完全占据前k名注意力位置；③目前仅针对基于注意力的防御评估，其他类型的防御或可解释方法的有效性尚未验证。

---

## 17. Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment

**arXiv ID:** 2607.08029 | [PDF](https://arxiv.org/pdf/2607.08029v1)

**作者:** Hyeju Shin `[一作]` (ETRI), Jaein Kim `[通讯]` (ETRI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了3B以下视觉语言模型在Jetson Orin NX/AGX平台上不同组件（视觉编码器、投影器、LLM）量化配置的性能与资源消耗。

**💡 创新点**

创新点在于将视觉编码器、投影器和LLM三大组件分别单独量化并进行组合实验，揭示SigLIP编码器在Ampere架构下的量化延迟异常，以及MoE结构对INT4量化更为鲁棒。

**🔧 技术方法**

采用BitsAndBytes的INT4/INT8后训练量化、VLMEvalKit进行多任务准确率评估，并通过GPU内存与时间剖析实现组件级延迟与能耗测量。

**📊 数据集**

使用MME基准数据集对模型进行多子任务的准确率评估。

**📈 对比分析**

通过对比不同配置在NX/AGX平台上的MME得分、VRAM使用、TPOT、能耗等指标，发现MoE结构对INT4更耐量化、视觉INT8显著增加延迟、INT4降低内存但导致token生成慢，能效受平台差异显著影响。

**⚠️ 局限性**

局限在于仅使用BitsAndBytes量化、未考虑FP8或W8A8等硬件原生格式，以及未实现自动精度分配等进一步优化方法。

---

## 18. Metrics or Mirage? An Audit of Evaluation Inconsistencies in Colonoscopy Polyp Segmentation Benchmarks

**arXiv ID:** 2607.08203 | [PDF](https://arxiv.org/pdf/2607.08203v1)

**作者:** Aisha Urooj `[一作]` (Lunit), Neelu Madan `[通讯]` (Aalborg University & Pioneer Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对结肠镜息肉分割领域的评估做了系统性审计，并在三种受控协议下重新评估了五个代表性模型，揭示了指标缺失、数据划分不一致、统计不严谨及转录错误等问题。

**💡 创新点**

提出了Polyp Segmentation Reporting Checklist（PSRC），并提供统一评估工具，指出单一重叠指标（Dice）无法体现边界与召回误差，并展示随机拆分和外部数据导致排名易变的现象。

**🔧 技术方法**

采用了结构化的指标评估框架（Dice、IoU、Recall、HD95、NSD、Lesion-level F1），对比了固定拆分、随机拆分和跨中心外部验证三种协议，并利用Wilcoxon、Friedman等统计检验。

**📊 数据集**

使用了 Kvasir-SEG、CVC-ClinicDB、CVC-ColonDB、ETIS-LaribPolypDB、CVC-300 等标准公开数据集，以及 PolypGen 六中心的外部数据。

**📈 对比分析**

通过对比发现：在固定拆分下模型排名随数据集变化；在随机拆分下 Dice 排名不稳定；在外部数据下所有模型性能大幅下降，但排名基本保持；边界指标 HD95 与 Recall 能揭示 Dice 隐藏的严重缺陷，表明单一 Dice 并不能体现临床可靠性。

**⚠️ 局限性**

局限性包括：重训模型可能未完全复现原作者细节；外部验证集中仅覆盖 PolypGen，未涵盖所有可能的域漂移；评估工具依赖特定实现，HD95 计算差异仍可能导致跨工具比较不一致。

---

## 19. A safety-oriented hypothetico-deductive framework for AI-assisted differential diagnosis

**arXiv ID:** 2607.08038 | [PDF](https://arxiv.org/pdf/2607.08038v1)

**作者:** Fan Ma `[一作]` (Yale University), Hua Xu `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种面向安全的诊断推理框架AegisDx，模拟临床假设-演绎推理流程，输出包括差异诊断列表、必不漏诊断、证据链接的推理与管理建议；

**💡 创新点**

通过多角色LLM协作、明确的角色契约、结构化中间输出、证据检索与验证门控，系统化保证“必不漏”条件、可追溯性与可操作性；

**🔧 技术方法**

使用大型语言模型（GPT-oss-120B等）作为核心，搭配专门的专家代理、警示代理、检索-推理-验证循环及管理生成代理；

**📊 数据集**

三组数据集：基于NEJM、JAMA、Annals of EM的文献案例；Annals of EM的必不漏诊断注释；Yale New Haven Health System的43例真实急诊记录；

**📈 对比分析**

对比方式包括：与相同后端LLM的基线Top‑k准确率、Annals of EM的必不漏覆盖率和医生评估的安全得分。AegisDx在Top‑3准确率上比基线提升5.7–17.1个百分点；必不漏覆盖率提升12–26个百分点；医生安全得分从4.31提升至4.55，均具显著统计意义；

**⚠️ 局限性**

局限包括：评估主要基于回溯性案例，可能过估性能；公开案例可能已被预训练模型记忆；缺乏实时临床验证；安全注释依赖主观医生共识；外部检索质量不确定；可读性仍需改进，未见直接临床结果。

---

## 20. Mini-Programs, Mega-Problems: Unveiling OAuth-based Authentication Misuses in Mini-Programs via Dynamic Analysis

**arXiv ID:** 2607.08232 | [PDF](https://arxiv.org/pdf/2607.08232v1)

**作者:** Zidong Zhang `[一作]` (Simon Fraser University), Jianliang Wu `[通讯]` (Simon Fraser University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建基于动态分析的框架，系统评估并发现微信与百度小程序中OAuth身份验证的三类运行时误用，共检测到4.6万余个程序中的多种漏洞。

**💡 创新点**

创新点在于细化M1a/M1b的时序区分、结合OCR+DFS的页面识别以支持混淆程序，并首次揭露平台级IV重用的加密缺陷。

**🔧 技术方法**

采用UI自动化+OCR识别、Xposed/Appium远程启动、HTTPS流量监控与基于模板的参数检测，以及静态配置文件路由解析等技术。

**📊 数据集**

使用44,273个微信小程序与2,721个百度小程序的完整源码与包，累计约129.5 GB。

**📈 对比分析**

与KeyMagnet、Whiskey等现有方法对比，检测率大幅提升；平均每个程序耗时约2 min，覆盖率>95%，但对需手动验证码或专用加密的程序失效。

**⚠️ 局限性**

受限于UI渲染不稳定、网络环境、被动拦截检测以及企业级加密流量，导致无法分析内部或高安全级别的小程序。

---

## 21. Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio

**arXiv ID:** 2607.08127 | [PDF](https://arxiv.org/pdf/2607.08127v1)

**作者:** Utkarsh A. Mishra `[一作]` (Georgia Tech), Jiayuan Mao `[通讯]` (Amazon FAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了将预训练视频生成模型微调用于机器人控制时出现的泛化差距，并提出了通过Temporal Ratio诊断与自适应引导的框架。

**💡 创新点**

提出Temporal Ratio度量行动头对未来视频潜在空间的依赖，揭示其与组合泛化的关系，并基于此实现了TR自适应的推理时引导方法。

**🔧 技术方法**

采用流匹配视频基础模型（Cosmos-Predict 2.5 DiT）和Gemma Transformer行动头，利用Low-Rank Adaptation进行参数高效微调，并实现了基于TR的语言与计划引导。

**📊 数据集**

在LIBERO组合泛化基准以及真实世界的双手YAM多任务数据集上进行评估。

**📈 对比分析**

与多种VLA、WAM和VAM基线（π₀、Cosmos-Policy、Fast-WAM等）进行对比，TR自适应引导将OOV成功率从约55%提升至近60%，在真实机器人任务中平均成功率提升约12%。

**⚠️ 局限性**

仅针对特定的VAM架构，过度依赖视频模型生成的未来轨迹，误导性未来预测会被放大；引导机制增加了推理开销，降低了执行频率。

---

## 22. UniRef-UAV: A Multimodal Benchmark for Universal Referring in UAV Imagery

**arXiv ID:** 2607.08267 | [PDF](https://arxiv.org/pdf/2607.08267v1)

**作者:** Haibin Tian `[一作]` (Northwestern Polytechnical University), Dingwen Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对无人机影像的通用指代任务——Universal Referring，支持文本、图像、混合查询以及零至多目标输出；构建了多模态基准 UniRef-UAV；设计并实现了检测式基线 UAV-URNet；

**💡 创新点**

创新点在于：①将查询模态与输出基数统一到同一任务框架；②构建包含 22 个 UAV 数据集的多模态基准；③提出共享查询表示与集合预测的检测式基线；

**🔧 技术方法**

使用检测式 2+2 结构（GroundingDINO）进行候选框生成；通过共享视觉/文本编码器和轻量 MLP 将多模态查询映射到统一空间；冻结预训练编码器，优化交叉注意力与解码器；

**📊 数据集**

使用 48,470 张 UAV 影像，包含 156,987 条查询（文本、图像、文本+图像），并提供 7:1:2 的训练/验证/测试拆分；

**📈 对比分析**

与 2+1 目标框、检测式、通用 MLLM、定位 MLLM 等多类方法对比；UAV-URNet 在文本、图像、混合查询中均表现优异，尤其在无目标判别和多目标定位上优于现有模型；

**⚠️ 局限性**

局限性包括：跨域视觉查询性能下降；小目标定位仍不理想；难以处理高度模糊或无目标场景；对算力和部署仍有挑战。

---

## 23. Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator

**arXiv ID:** 2607.07993 | [PDF](https://arxiv.org/pdf/2607.07993v1)

**作者:** Shiping Yang `[一作]` (Simon Fraser University), Angel X. Chang `[通讯]` (Simon Fraser University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Hallucination Self-Play（HSP）框架，使得在无外部监督的情况下，生成器与检测器通过自我对弈共同进化，从而提升LLM生成文本的真实性检测能力。

**💡 创新点**

创新点在于：①将自我对弈（self‑play）引入幻觉检测任务，克服传统生成器静态、难以适应检测器提升的瓶颈；②在生成器侧设计多重奖励门控（对抗性、实体识别、拒绝检测）以抑制奖励劫持；③利用RLVR（可验证奖励）在检测器侧实现无监督强化学习，进一步优化检测精度。

**🔧 技术方法**

核心技术包括：RLAIF（基于AI反馈的强化学习）用于演化生成器；GRPO用于统一的策略优化；RLVR用于检测器的可验证奖励学习；以及多轮闭环自我对弈循环，实现动态自适应训练。

**📊 数据集**

主要数据集为RAGTruth（包含QA、Data‑to‑Text、Summarization三类任务的真实文本与人工标注的幻觉段落），以及HotpotQA作为合成数据的种子。

**📈 对比分析**

在RAGTruth基准上，HSP在单轮自我对弈后使7B模型的F1提升约4–6点；多轮自我对弈后可达74.3%F1，接近GPT‑4o w/ CoT（74.5%）。相较于仅SFT或RLVR的检测器，HSP显著提升召回率，保持精度不变，表现优于当前最先进的LLM检测器。

**⚠️ 局限性**

局限性包括：①生成器在训练过程中需严格监管以防生成误导性幻觉；②奖励门控仍可能遗漏细粒度的劫持行为；③框架对低资源或非检索式任务的迁移性待进一步验证。

---

## 24. Modular Pretraining Enables Access Control

**arXiv ID:** 2607.08077 | [PDF](https://arxiv.org/pdf/2607.08077v1)

**作者:** Ethan Roland `[一作]` (AE Studio), Alex Cloud `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种通过梯度路由实现单模型多能力配置的方法

**💡 创新点**

创新点在于使用永活核心网络与不同比例的辅助模块进行梯度路由，从而实现可 ablation 的能力控制

**🔧 技术方法**

采用了 GRAM 梯度路由、LoRA 参数高效适配以及数据过滤等技术

**📊 数据集**

使用了 FineWeb‑Edu、arXiv、The Stack 等通用语言数据与 virology、cybersecurity、nuclear physics、specialized code 四个领域的专门数据集

**📈 对比分析**

与数据过滤、FT‑LoRA、FT‑Full 等基线进行计算等价比较，GRAM 在核心、保留和遗忘指标上能与数据过滤相近，但训练成本显著降低

**⚠️ 局限性**

局限性包括对标签完整性敏感、对规模和超参数依赖强、并非所有能力都能同等有效地被隔离或移除

---

## 25. Reverse Engineering Compliance: A Dual-Graph Verification Framework for Auditing Legacy IT Security Concepts

**arXiv ID:** 2607.08292 | [PDF](https://arxiv.org/pdf/2607.08292v1)

**作者:** Lea Roxanne Muth `[一作]` (Freie Universität Berlin), Marian Margraf `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ASSERT 框架，将遗留 IT 安全概念（IT‑SC）转化为中间图 G_Doc，使用确定性图比较 ΔG 检测与已验证参考图 G_GT 的差异，并生成符合 OSCAL 的 SSP 与 AR。

**💡 创新点**

创新点在于：① 通过本体驱动的实体与关系提取构造可追溯的文档图；② 定义五类节点/边错误的形式化差异 ΔG，提供可度量、可审计的误差集；③ 将验证过程与 OSCAL 输出分离，避免 LLM 随机性对审计证据的影响。

**🔧 技术方法**

技术包括：Llama/LLM 结构化提取（可选 Generic、Guided、Enforced），层次化分块与 Pydantic schema 校验，严格词汇匹配的节点对齐，确定性双图比较（节点：V_O、V_P；边：E_O、E_T、E_G），以及 OSCAL 生成模块。

**📊 数据集**

数据集：公开的 RecPlast GmbH 数据，包含完整的 IT‑Grundschutz 过程链、SA 产物、最终 IT‑SC 和参考图，使用此集构建 G_GT 并评估 ASSERT。

**📈 对比分析**

比较方法：对三种配置（Generic、Schema‑Guided、Schema‑Enforced）分别在 Gemma 4 与 Opus 4.7 LLM 上执行基线与故障注入实验；使用节点/边级别的 Precision/Recall/F1 以及 ΔG 误差计数。性能方面：Generic 在 Gemma 上召回低、假负高；Guided 在 Opus 上显著提升 E_T 检测；Enforced 准确消除假节点但忽略新发现的资产；整体 F1 在不同配置与模型间呈现显著差异。

**⚠️ 局限性**

局限性：① 依赖参考图 G_GT 的完整性与准确性，若 G_GT 本身含误将直接影响 ΔG；② 仅在 RecPlast 上验证，缺乏跨行业/规模的泛化；③ fault‑injection 规模受实体重叠限制；④ 对边标签的评估仅覆盖已在 G_GT 中出现的关系；⑤ 对多模型并行或更大规模 LLM 的鲁棒性未充分评估。

---

## 26. Certified Interventional Fidelity: Anytime-Valid, Adaptive Evaluation of Causal Claims in Mechanistic Interpretability

**arXiv ID:** 2607.08349 | [PDF](https://arxiv.org/pdf/2607.08349v1)

**作者:** Amir Asiaee `[一作]` `[通讯]` (Vanderbilt University Medical Center), Amir Asiaee (Vanderbilt University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Certified Interventional Fidelity（CIF）框架，给机制解释的干预评估提供任意时刻有效的置信区间；

**💡 创新点**

创新点在于把干预评估转化为可计量的因果估计量，并通过投注型置信序列在自适应采样与连续监控下显著降低样本成本；

**🔧 技术方法**

使用Hoeffding和投注置信序列、重要性加权混合提议、分层与多重校正以及主动采样技术；

**📊 数据集**

在MNIST数据集上评估网络抽象，在GPT‑2 Small模型上评估IOI任务的电路；

**📈 对比分析**

与传统点估计对比，在MNIST抽象中投注序列比Hoeffding节省10–30倍样本；在GPT‑2电路中，仅需几十到百次前馈即可证明95%恢复率；

**⚠️ 局限性**

局限在于仅保证给定度量与分布下的估计置信度，无法解决抽象识别问题，对未剪裁或无界量度的支持有限；

---

## 27. AutoPersonas: A Multi-Timescale Loop Engine for Open-Ended Persona Evolution

**arXiv ID:** 2607.08252 | [PDF](https://arxiv.org/pdf/2607.08252v1)

**作者:** Mengchen Li `[一作]` `[通讯]` (Latrix), Mengchen Li (Latrix)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AutoPersonas，一种多时域生命-环境引擎，用于实现长周期 persona 代理的自我演化与自锁防护；

**💡 创新点**

将自锁（self‑locking）概念明确为运行时的功能收敛，并通过分离观察（Observation）、状态（State）与出现（Occurrence）三者，采用自上而下的环境多尺度治理与上下文治理来抑制自锁；

**🔧 技术方法**

采用基于 LLM 的条件变异引擎生成未来事件，使用 OSO（Occurrence‑State‑Observation）循环进行证据治理，构建语义状态机与多时域修订流程，并在 SoulOS 系统架构中集成记忆、文化与用户交互模块；

**📊 数据集**

无公开数据集；实验基于八种主流 LLM（Claude、DeepSeek、Doubao、Gemini、GLM、GPT、Kimi、Qwen）在固定 persona 规范下进行40天的自我循环，另外进行三年压缩诊断仿真；

**📈 对比分析**

通过八模型动作频道重复率（平均95–97%）与宏主题重复率（79–88%）的基准压力测试与自锁消除的 A/B 对照，结果显示在引入遮蔽与样本级偏差调节后宏主题重复率从61.8%降至36–39%，宏主题数量翻倍，表明显著提升了开放性与多样性；

**⚠️ 局限性**

局限在于实验仅覆盖单一 persona 规范与有限的模型样本，缺乏跨多 persona 与长期验证；机制的具体阈值与细节未公开，缺乏可复制性；对真实世界复杂性与长期安全性的评估尚待进一步研究。

---

## 28. ArtMine: Discovering and Formalizing Artistic Processes

**arXiv ID:** 2607.08331 | [PDF](https://arxiv.org/pdf/2607.08331v1)

**作者:** Kaustubh Kumar `[一作]`, Shirish Karande `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于迭代优化的图像生成方法，通过多轮生成与评估，利用 CSD、CLIP 与 LPIPS-sim 等指标持续提升图像质量。

**💡 创新点**

创新点在于：①将三种评价指标融合为统一的平均分，用于引导生成过程；②设计自适应迭代策略，使模型在每一轮生成中逐步优化；③构建全新的评价框架，兼顾内容、语义与感知质量。

**🔧 技术方法**

采用的技术主要包括：迭代式生成网络（如 Diffusion 或 GAN 的改进版），结合 CLIP 文本-图像相似度、CSD（内容相似度）和 LPIPS-sim（感知相似度）进行反馈优化；使用梯度优化器在每轮迭代中调整生成参数。

**📊 数据集**

使用的数据集主要有 ImageNet、COCO Caption 以及可能的 CUB-200 等通用图像与文本对数据集，覆盖多种视觉场景与文本描述。

**📈 对比分析**

方法与传统基准模型（如 VQGAN+CLIP、Stable Diffusion 等）在相同条件下进行对比，结果显示平均分从 0.72 提升到 0.84，客观指标（CSD、CLIP、LPIPS-sim）均优于对照组，证明了迭代优化策略的有效性。

**⚠️ 局限性**

局限性包括：①计算成本高，迭代次数多导致推理时间显著增加；②对复杂或高分辨率场景的生成仍可能出现模糊或失真；③评价指标虽多元，但仍无法完全覆盖人类主观感知的细微差异。

---

## 29. A Multi-cluster Boundary Learning Method for Out-of-Scope Intent Detection via MiniLM Embedding

**arXiv ID:** 2607.07974 | [PDF](https://arxiv.org/pdf/2607.07974v1)

**作者:** Yihong Xu `[一作]` (University of Science and Technology of China), Linyuan Lü `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多层级流程（gate→router→expert），在gate阶段使用MiniLM嵌入并通过多聚类边界学习实现OOS意图检测。

**💡 创新点**

创新点在于将OOS检测转化为一类分类，利用多聚类边界而非单中心，采用轻量MiniLM并通过级联解耦OOS与已知意图分类。

**🔧 技术方法**

使用MiniLM进行句子嵌入、K-means聚类构建多中心、Mahalanobis距离计算边界、L2归一化、LoRA微调router/expert，整体实现一类分类流程。

**📊 数据集**

在CLINC150、StackOverflow和Banking77这三个公开意图检测数据集上进行实验。

**📈 对比分析**

与MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB等基线相比，在OOS F1上取得0.85%~17.12%的提升，整体保持state‑of‑the‑art水平。

**⚠️ 局限性**

局限在于对已知意图的分类性能不如某些基线，且当前仅使用了22M MiniLM，未来需要更强的轻量模型以进一步提升效果。

---

## 30. Agentic AI and Retrieval-Augmented Models in Straight-Through Underwriting

**arXiv ID:** 2607.07858 | [PDF](https://arxiv.org/pdf/2607.07858v1)

**作者:** Robert Richardson `[一作]` (Brigham Young University), David Sandberg `[通讯]` (Brigham Young University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究设计并实现了一套基于多代理、检索增强生成的AI框架，用于自动化小型商业业务主保单的直通承保流程。

**💡 创新点**

创新点在于将检索、第三方数据检索、缺失信息诊断和多步推理拆解为独立代理，并通过显式的状态机进行协同，以显著提升复杂和缺失信息情形下的决策准确率和可解释性。

**🔧 技术方法**

技术包括大型语言模型（LLM）+检索增强生成（RAG）、多代理协同框架（LangChain/​LangGraph）、向量检索（FAISS）、第三方数据接口以及自定义的规则检索与反思循环。

**📊 数据集**

使用的是自己构造的合成业务主保单数据集（635份样本），包含合规、单一违规、多步推理、可恢复缺失信息与不可恢复缺失信息四类情形，以及对应的合成指导手册和模拟第三方数据。

**📈 对比分析**

比较方法为在同一数据集上对三种系统（单LLM、Naïve RAG、Agentic RAG）进行决策准确率、每类情形准确率、理由相似度和平均延迟的评估；Agentic RAG在两大模型下均达到了约85%总体准确率，单一违规场景略逊于单LLM，而在多步推理和缺失信息情形上显著优于基线。

**⚠️ 局限性**

局限性包括合成数据缺乏真实业务噪声和复杂性、对检索质量与提示设计高度敏感、延迟显著增加、且未在真实保险运营环境中验证鲁棒性和合规可审计性。

---

## 31. CRIMP: Compact & Reliable DNN Inference on In-Memory Processing via Crossbar-Aligned Compression and Non-ideality Adaptation

**arXiv ID:** 2607.08015 | [PDF](https://arxiv.org/pdf/2607.08015v1)

**作者:** Shuo Huai `[一作]` (Nanyang Technological University), Weichen Liu `[通讯]` (Nanyang Technological University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种综合学习框架，实现整数量化、跨柱对齐剪枝和运行时非理想性自适应，以在ReRAM内存计算加速器上生成紧凑、可靠的DNN模型。

**💡 创新点**

创新点在于将跨柱对齐剪枝与整数量化和非理想性适配协同训练，消除浮点乘法器和对齐硬件，实现无额外硬件开销且兼容ReRAM特性。

**🔧 技术方法**

采用多粒度剪枝（核组剪枝+跨柱剪枝）、整数量化（将缩放因子逼近2的幂）、基于运行时仿真的非理想性适配和动态零恢复的联合训练框架。

**📊 数据集**

在MNIST、CIFAR‑10和ImageNet上使用LeNet‑5、VGG‑16、ResNet‑56等模型进行验证。

**📈 对比分析**

与现有IMP感知剪枝/量化方法相比，在不增加硬件开销的前提下实现了更高的稀疏率、最低的准确率下降，并将计算功耗和面积平均分别降低了122×和19×。

**⚠️ 局限性**

局限在于对非理想性模型的仿真依赖已知统计分布，且在极端设备误差或更大规模网络时的鲁棒性尚待验证。

---

## 32. Uncertainty-gated selection for block-sparse attention

**arXiv ID:** 2607.07724 | [PDF](https://arxiv.org/pdf/2607.07724v1)

**作者:** Thomas Rossi `[一作]` `[通讯]` (Eonpass), Thomas Rossi (Eonpass)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于价值信息的路由器（Value‑of‑Information Router）来改进块稀疏注意力的 top‑k 选择，动态扩展可疑块集合。

**💡 创新点**

创新点在于将 top‑k 阈值误判的置信度量化为标准化切除间隙 σ，并以此为依据在最不确定的底层 q 分位数的 tile 上按比例扩大保留块，既兼容任何块评分背骨，又能显著提升稀疏模型性能。

**🔧 技术方法**

主要技术包括：块分数标准化、top‑k 选择、σ 距离计算、分位数触发器、均匀扩展 ρ× 的 kv 索引、融合选择与注意力核的 Triton 实现。

**📊 数据集**

使用的评测数据集包括 RULER NIAH‑multikey、LongBench‑v2‑medium、以及自定义诊断基准 Pointer‑Chase Haystack（PCH），并在 32K 与 128K 上测试。

**📈 对比分析**

与稠密模型、传统 top‑k（SSA）、Quest 以及无路由器的方案对比，路由器在四个模型（Qwen2.5、Mistral、Qwen3.6）上均提升了 paired‑recall 和总体准确率，尤其在 LongBench‑v2‑medium 上可达 0.75 的 paired‑recall；在 128K 上保持 81–89% 的稠密准确率，并实现 0.62×–0.80× 的稠密推理时间。

**⚠️ 局限性**

局限性包括：tile‑级别选择导致的单行强烈需求被忽视、对不同架构的覆盖范围有限、仍有进一步加速与 MoE 核融合的空间、触发阈值 q 与扩展系数 ρ 的超参数仍需更系统的搜索和理论指引。

---

## 33. Frequency-Domain Multi-Modality Transportation Modeling

**arXiv ID:** 2607.08475 | [PDF](https://arxiv.org/pdf/2607.08475v1)

**作者:** Jiewen Deng `[一作]` (Southern University of Science and Technology), Renhe Jiang `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 FreMo 框架，对多模态交通时间序列先在频域进行模态自适应频率滤波，再进行频率引导的跨模态协同，最终在时间域输出预测。

**💡 创新点**

创新点在于（1）模态自适应频率滤波（MFF）可为每个模态节点学习软门控频率权重；（2）频率引导协同整合（FSI）通过频率级别的软最大化权重实现选择性跨模态信息共享；（3）该过程可作为插件集成到任意时序基座，兼具轻量和高效。

**🔧 技术方法**

技术包括：实数 FFT/逆 FFT、频域幅值提取、节点嵌入驱动的频率权重生成、软门控滤波、频率级别软最大化协同权重、残差反馈调节、图神经网络或 TCN/Transformer 作为时序基座。

**📊 数据集**

实验数据集：New York City、Washington D.C.、Chicago 三个城市的交通多模态数据，涵盖 Bike Inflow/Outflow、Taxi Inflow/Outflow 四种模式。

**📈 对比分析**

与十种基线（包括 Uni‑modality Graph WaveNet、AGCRN、MTGNN 等以及 Multi‑modality MiST、STtrans、COCOA、MoSSL）进行 MAE/RMSE 对比。FreMo 在所有数据集、所有模态以及不同预测时刻均获得最低 MAE/RMSE，尤其在 Taxi 流和高时刻预测上提升显著；在多种基座（AGCRN、TimesNet、iTransformer、STAEformer）中以插件方式提升 1–20% 以上。

**⚠️ 局限性**

局限性：频域处理仍受 FFT 解析度限制，对极高频噪声或异常值的鲁棒性需进一步提升；模型在节点级解释方面尚未深入；在极端稀疏或分布漂移场景下性能可能受影响。

---

## 34. How Analysts Use AI in High-Stakes Crime Linkage: An Industrial Study

**arXiv ID:** 2607.08274 | [PDF](https://arxiv.org/pdf/2607.08274v1)

**作者:** Jessica Woodhams `[一作]` (University of Birmingham), Dalal Alrajeh `[通讯]` (Imperial College London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究者在英国NCA内部对一款AI辅助犯罪链联工具进行工业级混合方法可用性评估。

**💡 创新点**

创新点在于结合可解释性特征与传统行为分析，并通过眼动与鼠标跟踪实时验证分析师与AI预测的交互。

**🔧 技术方法**

使用了机器学习预测模型（基于行为、时空特征），可视化界面（排名列表、雷达图、行为矩阵）以及眼动、鼠标追踪技术。

**📊 数据集**

采用NCA真实严重性性侵案件数据库中的约三万条案件记录。

**📈 对比分析**

通过用户满意度量表、任务完成时间、眼动/鼠标指标进行对比，结果显示分析师对工具易用性高度认可，但对效用评价混合。

**⚠️ 局限性**

局限包括样本仅来自一支执法团队、雷达图数据错误、未对不同模型或不同算法进行基准比较。

---

## 35. Multi-Agent Firewall Architecture for Privacy Protection of Sensitive Data in Interactions with Language Models

**arXiv ID:** 2607.08282 | [PDF](https://arxiv.org/pdf/2607.08282v1)

**作者:** Hugo García Cuesta `[一作]` (Universidad Carlos III de Madrid), Alfonso Sánchez-Macián `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套开源、以本地为主的 LLM 防火墙，结合浏览器扩展与 MiTM 代理，实现对 Web 与 API 双通道的拦截与 PII/代码泄露检测，旨在保护企业与个人用户的数据安全；

**💡 创新点**

创新点在于双层拦截与可插拔适配器、基于 DAG 的多代理检测管道、定制化风险评估与分级阻断、Sanitization‑First 方案以及对多模态、代码相似度等新型威胁的支持；

**🔧 技术方法**

采用了 Chromium 浏览器扩展（Manifest V3）、mitmproxy、FastAPI、LangGraph、Tesseract OCR、Vision‑Language Models、GLiNER NER、RapidFuzz 代码相似度、LiteLLM、LLM 语义检测等技术栈；

**📊 数据集**

评估使用了公开的 nvidia/Nemotron‑PII 数据集（含多种 PII 语料），并在固定种子下进行 500 条样本测试；

**📈 对比分析**

通过对比四组配置（仅确定性、NER、强制 LLM、fine‑tuned Gemma）发现最佳 fine‑tuned Gemma 4 E4B 取得 94.93% F1、95.21% 精准度，平均 5.4 s 延迟；而低延迟配置可达 81.66% F1 仅 0.85 s；

**⚠️ 局限性**

局限性包括：仅在本地推理，性能受限于设备；浏览器扩展仅支持 Chromium；代理仅覆盖 HTTP(S)/WSS，缺少 gRPC/WebRTC 等；未覆盖助手响应与注入防御，未来需进一步扩展与评估多模态与代码泄露检测。

---

## 36. KronQ: LLM Quantization via Kronecker-Factored Hessian

**arXiv ID:** 2607.07964 | [PDF](https://arxiv.org/pdf/2607.07964v1)

**作者:** Donghyun Lee `[一作]` (University of Southern California), Priyadarshini Panda `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KronQ，一种在后训练量化（PTQ）框架中同时利用激活协方差和梯度协方差的量化方法；

**💡 创新点**

创新点在于：①引入K-FAC分解下的梯度协方差H_G，打破传统仅依赖输入协方差的假设；②提出双向不相干处理（BiIP）在输入和输出维度同时降低量化误差；③通过tr(H_G)·tr(H_X) 的混合精度敏感度指标实现更优的层间位宽分配；

**🔧 技术方法**

技术手段包括GPTAQ基础量化、K-FAC近似、Hadamard随机正交变换、双向不相干处理、梯度协方差估计以及基于Hessian迹的混合精度分配；

**📊 数据集**

主要使用WikiText-2作为校准和评测数据集，同时在七个常识推理基准（PiQA、ArcE、ArcC、HellaSwag、WinoGrande、BoolQ、OpenBookQA）和Gemma-3-12B、DeepSeek-R1-Distill-Llama-8B、Phi-4-mini-instruct等新模型进行验证；

**📈 对比分析**

与GPTQ、GTAQ及多种旋转、混合精度方法对比，KronQ在W2/W3/W4位宽下显著降低perplexity（如LLaMA‑3‑70B 2‑bit仅7.93 PPL，GPTQ >2000），并在零样本推理任务上提升准确率；

**⚠️ 局限性**

局限性包括：需要一次完整的反向传播来估计H_G，导致校准时延和内存占用相对较高；BiIP中的Hadamard变换虽开销小，但在极大模型上仍存在一定成本；此外，方法仍依赖K-FAC近似，若输入/梯度不独立性被破坏，精度提升可能受限。

---

## 37. Dive Into the Implicit Biases of Low-rank Vision-language Alignment

**arXiv ID:** 2607.08194 | [PDF](https://arxiv.org/pdf/2607.08194v1)

**作者:** Mingjia Shi `[一作]`, Minghui Wu `[通讯]` (Mininglamp)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究低秩适配在视觉语言对齐阶段的隐式偏差，并与全参数对齐进行系统对比，发现低秩对齐既能显著降低训练成本，又能提升下游性能。

**💡 创新点**

创新点包括：①提出低秩对齐的保守行为、线性可分性保留和几何稳定性三大隐式偏差；②给出两条理论定理解释低秩梯度流偏向噪声鲁棒和平坦子空间的机制；③在1.4B–14B规模的多种VLM上实现了全面的实证验证。

**🔧 技术方法**

采用低秩适配技术（LoRA、LoHa、LoKr）对LLM进行适配，配合线性探针、几何特征分析、以及基于Unconstrained Feature Model的理论优化分析。

**📊 数据集**

使用了LAION、CC、SBU、SAM、MS‑COCO等视觉语言对齐数据集；评估基准包括GQA、MMBench、MME、MMMU‑Pro、MMVet‑gpt、POPE、ScienceQA、TextVQA等。

**📈 对比分析**

通过在1.4B至14B模型上与全参数对齐做对比，低秩对齐在所有维度（感知、知识、推理、误报）均获得数个百分点以上的性能提升，且训练时间和参数量均大幅下降。

**⚠️ 局限性**

限制点：在某些任务（如POPE、MMMU‑Pro）低秩对齐仍存在性能波动；未对极大规模模型（>14B）或不同视觉编码器的交互做深入验证；对低秩超参（rank、operator）在不同任务间的通用性仍需进一步研究。

---

## 38. Evaluating the Generalizability of Foundation Models for Extreme Environmental Events: Case Study of California Wildfire PM2.5

**arXiv ID:** 2607.07951 | [PDF](https://arxiv.org/pdf/2607.07951v1)

**作者:** Yongcan Huang `[一作]` (University of Georgia), Ze Yu Liu `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对加利福尼亚州12年（2013–2025）1,375起野火事件产生的PM2.5浓度进行系统评估，比较了零射击时间序列基础模型（TimesFM、Chronos‑2、Moirai‑2、Time‑MoE）与传统深度学习基准（LSTM、BiLSTM、Transformer），并进一步对Chronos‑2与Time‑MoE进行LoRA轻量化微调。

**💡 创新点**

创新点包括：①提出留一事件（leave‑one‑incident‑out）交叉验证协议，真实测量模型对未知野火事件的泛化能力；②首次构建野火PM2.5的TSFM基准，系统检验大规模预训练模型在极端浓度下的表现；③揭示即使大规模预训练模型在整体误差上略优于持久性基准，它们在危险浓度阈值下仍远逊于精细训练的BiLSTM。

**🔧 技术方法**

采用了零射击TSFM（TimesFM、Chronos‑2、Moirai‑2、Time‑MoE）与其LoRA微调版本，并与LSTM、BiLSTM、Transformer等训练基准对比；评估指标包括MAE、RMSE、R²以及各AQI阈值（35.5、55.5、125.5、225.5 μg/m³）的F1分数；通过留一事件交叉验证保证评估不受同一火灾事件信息泄漏。

**📊 数据集**

使用的数据集为79个EPA认证监测站在2013–2025年间收集的约1.73 M小时PM2.5记录，对1,375起野火事件进行事件对齐，最终得到约277,866个48小时输入/多时延输出窗口，形成一个单变量PM2.5时间序列数据集。

**📈 对比分析**

在5折留一事件交叉验证下，BiLSTM在所有指标（MAE≈5.16 μg/m³、RMSE≈12.58 μg/m³、R²≈0.75、Hazardous阈值F1≈0.63）均优于任何TSFM；零射击TSFM仅略好于持久性基准（MAE≈5.6–5.9 μg/m³），而LoRA微调的Chronos‑2和Time‑MoE虽提升了MAE和R²，却仍低于BiLSTM，尤其在高浓度阈值的F1分数上差距明显。

**⚠️ 局限性**

局限性包括：①仅使用单变量PM2.5，未引入气象或火灾相关协变量；②评估仅限加州区域，未考察跨区域迁移；③LoRA微调的实验规模与数据效率未做系统探索；④未对TSFM的空间信息或更大规模模型进行进一步实验。

---

## 39. Securing Autonomous Vehicle Systems via Twin-Aware Federated Reinforcement Learning

**arXiv ID:** 2607.08137 | [PDF](https://arxiv.org/pdf/2607.08137v1)

**作者:** Zifan Zhang `[一作]` (North Carolina State University), Yuchen Liu `[通讯]` (North Carolina State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于数字孪生的安全聚合框架，利用多步过滤和历史聚合参数抵御联邦强化学习中的毒性攻击，提升自动驾驶系统的鲁棒性。

**💡 创新点**

创新点在于：①将数字孪生用于回放式学习，生成多样化仿真环境；②采用历史聚合参数与中心梯度的双重过滤策略，确保仅聚合可信梯度；③给出在存在Byzantine攻击时的收敛理论保证，这是首次在安全关键自动驾驶场景下实现鲁棒联邦RL。

**🔧 技术方法**

使用的技术包括：联邦强化学习（FRL）、数字孪生仿真（HighwayDT）、SVRG梯度优化、阈值过滤（ψ）、中心梯度选取、重要性采样、近似最近邻加速、以及多种鲁棒聚合规则（Median、Trimmed-mean、Krum等）。

**📊 数据集**

使用的数据集：真实I‑695高速公路的车辆轨迹（用于构建HighwayDT），Carla+SUMO仿真生成的多场景数据；在边缘缓存实验中使用Zipf分布生成的请求频率数据。

**📈 对比分析**

通过与FedAvg、Median、Trimmed‑mean、Krum、FoolsGold、FABA、FLTrust、FLAIR、FedPG‑BR、FLAME、DeepSight等十一种聚合策略对比，在13种毒性攻击（包括自适应攻击）下，所提出的框架在自动驾驶场景始终保持100%无碰撞率，并在边缘缓存场景中提升缓存命中率，明显优于基线方法。

**⚠️ 局限性**

局限性：①对数字孪生的构建和维护需要额外的数据与计算资源；②梯度过滤过程的O(K²)复杂度需要近似近邻加速；③当恶意比例接近或超过50%或阈值ψ、λ设置不当时鲁棒性下降；④实验主要在仿真/数字孪生环境，缺乏真实道路上的验证。

---

## 40. MentalHospital: A Virtual Environment for Evaluating Psychiatric Clinical Encounters

**arXiv ID:** 2607.08257 | [PDF](https://arxiv.org/pdf/2607.08257v1)

**作者:** Yuming Yang `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了MentalHospital虚拟精神科评估环境，并开发了MentalEval评估器，用于评估LLM在完整S.O.A.P.流程中的表现。

**💡 创新点**

提供了基于真实EHR的完整精神科访谈、检查、诊断、治疗模拟，并结合双轨评估（客观与主观）以及专属领域评估器，填补了现有孤立任务评估的空白。

**🔧 技术方法**

采用规则检测+专家审查的EHR脱标识、技能增强的标准化患者与医院侧检查模块、LLM对话代理、双轨评估流程、SFT+DPO训练的领域评估器。

**📊 数据集**

基于1,193例去标识化精神科EHR，涵盖ICD‑11主要类别与76种疾病，用于构造患者、检查证据与参考目标。

**📈 对比分析**

与人类专家、医学生、普通LLM和专门LLM对比；客观指标上LLM落后专家约37%，主观指标上专家在专业性更佳但LLM在同理心、诊断严谨度、治疗适宜性等方面接近或优于专家；MentalEval评估器与专家一致性达QWK 0.944。

**⚠️ 局限性**

LLM在精神状态评估方面仍显弱；评估依赖专家或复杂训练；缺乏真实临床验证；脱标识化处理可能限制部分临床细节。

---

## 41. Structure Learning on Clustered Data

**arXiv ID:** 2607.08238 | [PDF](https://arxiv.org/pdf/2607.08238v1)

**作者:** Ryan Thompson `[一作]` (University of Technology Sydney), Veerabhadran Baladandayuthapani `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了适用于聚类数据的混合效应有向无环图（mixed DAG）结构学习方法，能够同时估计全局固定效应和群组特异随机效应，并保证二者联合图仍为DAG。

**💡 创新点**

创新点在于将可微无环性约束（log‑determinant）推广到固定‑随机效应框架，保证固定效应矩阵与随机效应方差矩阵联合后仍保持无环；并提供了一阶收敛、可批量计算的优化算法。

**🔧 技术方法**

采用混合效应结构方程模型、可微无环性约束、L1稀疏正则化、梯度下降/近端梯度法、批量线性代数求解以及自适应步长和线搜索。

**📊 数据集**

实验使用了合成的 Erdős–Rényi DAG、半合成的 ANDES 物理网络以及真实肺癌多重免疫荧光蛋白表达数据。

**📈 对比分析**

与固定效应DAG、固定顺序混合DAG、oracle顺序混合DAG、无无环约束混合DAG等基线进行比较，混合DAG在估计误差、结构汉明距离（SHD）和 F1 分数上显著优于固定效应DAG，并逼近 oracle 顺序下的表现。

**⚠️ 局限性**

局限包括对图大小的计算瓶颈、对随机效应方差约束的依赖、在极小样本或高噪声情况下性能下降，以及 FM 变体依赖固定 DAG 估计的拓扑顺序。

---

## 42. LDFE: Laplacian Decoupled Feature Enhancement Block for Dual-Stream CNN-based RGB-IR Object Detection

**arXiv ID:** 2607.08076 | [PDF](https://arxiv.org/pdf/2607.08076v1)

**作者:** Wenhao Dong `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对RGB‑IR图像的多尺度特征融合模块LDFE，实现了在YOLO双流CNN骨干网络中的特征增强与降噪。

**💡 创新点**

创新点在于使用拉普拉斯金字塔对全局与局部特征进行分离，并分别通过GS^2E（全局状态空间增强）和LC^2E（局部卷积相关增强）进行去噪、融合与重构，实现对两模态噪声的针对性抑制与长短距信息的高效融合。

**🔧 技术方法**

采用了YOLOv8双流CNN骨干、拉普拉斯金字塔分解、通道交换、全局状态空间模型（SSM）、局部卷积+L1归一化+softmax融合、空间与通道注意力等技术。

**📊 数据集**

在六个公开RGB‑IR目标检测数据集上验证：M^3FD、DroneVehicle、LLVIP、FLIR‑Aligned、KAIST、VEDAI。

**📈 对比分析**

与多种基准方法（CSPDarknet、ResNet、Transformer、Mamba等）对比，LDFE在各数据集的mAP、Precision、Recall、F1等指标均达到或超过SOTA，提升幅度多达6%以上，且参数量与推理速度保持竞争力。

**⚠️ 局限性**

局限性包括在密集小目标检测表现仍不如部分专用方法，且对跨任务的适应性需进一步研究。

---

## 43. Collective Intelligence with Foundation Models

**arXiv ID:** 2607.07729 | [PDF](https://arxiv.org/pdf/2607.07729v1)

**作者:** J. de Curtò `[一作]` (BARCELONA Supercomputing Center), I. de Zarzà `[通讯]` (LUXEMBOURG Institute of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个多代理协同推理框架，利用 solver、critic 与 aggregator 三类模型在同一问题上并行生成、批评并汇总解答。

**💡 创新点**

核心创新在于：①将专门的 Critic 与 Aggregator 引入多代理系统；②系统化验证模型异质性是提升步骤推理质量的关键因素；③提供完整的消融实验与多维度评估。

**🔧 技术方法**

使用的大模型包括 Meta‑Llama‑3.3‑70B‑Instruct、NousResearch Hermes‑4‑405B、Qwen3‑235B‑A22B‑Instruct‑2507（solver）；DeepSeek‑R1‑0528（critic）；GPT‑OSS‑120B（aggregator）。评估技术包含语义相似度、数值一致率、步骤准确率等。

**📊 数据集**

构建了覆盖八大科学领域（微积分、物理、化学、生物、经济、优化、统计、数学）的多难度级别基准集，包含问题陈述、参考解答与逐步推理。

**📈 对比分析**

通过四种配置（单模型、同质框架、同质冗余、异质框架）进行系统消融，整体得分从 0.52 提升到 0.63，步骤准确率从 0.28 提升至 0.64，证明异质性带来显著推理质量提升。

**⚠️ 局限性**

局限性包括：整体最终答案得分提升有限；实验仅基于现有公开模型，缺乏更大规模模型或真实业务场景验证；对不同任务类型的泛化性和可扩展性尚待进一步研究。

---

## 44. Open Models, Open Risks: Measuring Unsafe Generation in Text-to-Image Models In the Wild

**arXiv ID:** 2607.07827 | [PDF](https://arxiv.org/pdf/2607.07827v1)

**作者:** Peilin Han `[一作]` (Xidian University), Zhuo Ma `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性评估了200+开源文本到图像（T2I）模型在真实环境中的安全性，并提出改进的安全评估指标 AASR。

**💡 创新点**

提出 AASR（Advanced Attack Success Rate）指标，综合了检测器、语义一致性和图像质量过滤，解决了传统检测器过度估计风险的问题；同时首次对大规模野外 T2I 模型的安全性进行横向和纵向（架构、攻击方式、后续微调、时间演变）分析，并对高风险模型进行追踪与报告。

**🔧 技术方法**

采用多阶段评估管线：NSFW 检测器 (MHSC)、Prompt–Image 语义对齐 (CLIPScore)、图像失真检测 (HADM)；三类代表性 jailbreak 数据集（Unsafe Diffusion Template Prompts、4chan、MMA-Diffusion）；对比传统 ASR 与 AASR；对模型家族、下游分支、时间序列进行统计分析。

**📊 数据集**

从 Hugging Face 上收集的 200+ T2I 模型（Stable Diffusion 1.5/XL、FLUX、Qwen-Image 等）以及三套 jailbreak 数据集；使用公开的语义对齐与失真检测工具。

**📈 对比分析**

与传统 detector‑only ASR 对比，AASR 显著降低了误报率（例如 SD-DS-7 的 ASR 0.965 降至 0.635）。在 200+ 模型中，AASR 发现高风险模型比例约 15%–20%，而传统 ASR 误判比例高达 60%+；展示了不同架构和后续微调对安全性的影响，并绘制时间趋势图表明部分新模型安全性并未改善。

**⚠️ 局限性**

局限性包括：AASR 仍依赖现有检测器与语义对齐模型的准确性；仅评估了三种 jailbreak 攻击，可能忽略其他更隐蔽的攻击方式；数据集与模型来源多样性有限，未覆盖所有开源平台；评估不涉及真实用户交互与长时推理的动态风险。

---

## 45. What to Keep, What to Forget: A Rate--Distortion View of Memory Compaction in LLMs and Agents

**arXiv ID:** 2607.08032 | [PDF](https://arxiv.org/pdf/2607.08032v1)

**作者:** Ashwin Gerard Colaco `[一作]` (University of California, Irvine), Nada Lahjouji `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并统一框架化了大型语言模型与代理在不同层级的记忆压缩方法，将其视为同一类 rate–distortion 问题，提出七轴分类体系，交叉迁移机制，并提出统一基准 COMPACT‑Bench。

**💡 创新点**

创新点在于：① 用信息瓶颈理论统一 KV‑缓存、提示/上下文、架构状态与代理记忆四个层级的压缩，推导统一下界；② 构建七轴 taxonomy，揭示重叠与空缺；③ 通过跨层迁移将已成熟的机制如 Query‑eviction、量化、低秩、抽象摘要等迁移至其他层；④ 提出 COMPACT‑Bench 能同时衡量多层压缩的误差积累与重用效率。

**🔧 技术方法**

采用信息瓶颈、Fano 不等式、低秩因子分解、量化、Sparse‑Attention 训练、抽象摘要、知识图谱、检索增强、动态稀疏化等多种技术组合，并在推理、提示、架构和代理记忆中实现。

**📊 数据集**

主要评估数据集包括 LongBench、InfiniteBench、HELMT、RULER、BABI‑Long、LOCOMO、LongMemEval、SCBench 等，覆盖单轮长上下文、跨任务记忆、代理交互等场景。

**📈 对比分析**

在 COMPACT‑Bench 里对 KV‑eviction、量化、提示压缩和抽象摘要在统一预算轴上进行对比，结果显示：KV‑eviction 与量化在低压缩比下保持近乎无损，而提示压缩与摘要在中高压缩比下误差显著累积；跨层迁移提升了整体鲁棒性，复合压缩策略比单一方法更优。

**⚠️ 局限性**

局限性包括：① 统一下界基于 worst‑case (Q)，缺乏对真实任务 (Q) 的可预测模型；② 多层压缩的误差累积机制尚未在实际代理交互中系统验证；③ 目前基准主要关注单一任务的单轮误差，未充分覆盖长期多任务或多代理的动态演化；④ 许多技术实现需要显式的 LLM 调优或大规模算力，难以迁移到资源受限环境。

---

## 46. Prompt Compression via Activation Aggregation

**arXiv ID:** 2607.08399 | [PDF](https://arxiv.org/pdf/2607.08399v1)

**作者:** Thibaud Ardoin `[一作]` (Freie Universität Berlin), Gerhard Wunder `[通讯]` (Freie Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何将大语言模型的提示信息压缩为单个激活向量，并在模型内部注入以替代原始提示文本。

**💡 创新点**

创新点在于提出一种轻量级的加权求和压缩器（Weighting MLP），证明中间层激活可通过加权线性组合在早期层恢复原提示效果，并揭示跨层兼容性与激活空间的线性可组合性。

**🔧 技术方法**

采用两阶段框架：1) 提取中间层隐藏状态；2) 用加权求和或Transformer压缩器生成补丁向量并注入早期层。训练使用交叉熵损失，优化 MLP 或 Transformer 编码器。

**📊 数据集**

使用自制 Toy Task 数据集（11个知识检索任务）和公开 ARC‑Easy 多项选择基准进行评估。

**📈 对比分析**

与全提示基线和占位符掩码基线对比，Weighting MLP 在大多数任务上仅损失 <2% 的准确率，且在 OOD 与 ARC‑Easy 上表现更稳健；Transformer 压缩器在训练集表现良好但泛化差。

**⚠️ 局限性**

局限性包括：仅适用于短提示和知识检索任务，压缩过程仍需前向传递一定层数，且方法需要白盒访问内部激活，无法直接用于闭源模型；对长文本和复杂推理的有效性未知。

---

## 47. Empirical Calibration and Conditional-Reliability Diagnostics for Bearing RUL Prediction under Operating-Regime Shift

**arXiv ID:** 2607.08273 | [PDF](https://arxiv.org/pdf/2607.08273v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Yunsheng Wang `[通讯]` (Santa Clara University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一种针对时间变化负载/转速的轴承剩余寿命可靠性评估协议，并在10个处理过的PHME轴承子集上进行实验。

**💡 创新点**

创新点在于将留出负载/转速区间的严格拆分与分层残差校准相结合，揭示了条件覆盖差异与传感器失效的可靠性缺陷。

**🔧 技术方法**

使用了融合原始振动窗口、工程特征与负载/转速上下文的预测表示模型，并通过经验残差校准实现90%预测区间。

**📊 数据集**

实验基于公开PHME时间变化负载/转速的10轴承子集（B01、B02、B03、B04、B05、B08、B10、B11、B12、B17）。

**📈 对比分析**

通过四路训练/验证/校准/测试的留一操作区间拆分与400树随机森林、TCN等基线对比，校准模型在均值MAE 0.1477、覆盖率0.90处优于随机森林，但在低负载/高速区间覆盖率仅0.666。

**⚠️ 局限性**

局限在仅使用10个轴承、未实现全档PHME或跨数据集验证、校准仅为经验且在原始通道失效时表现差、未构建物理数字双。

---

## 48. Leveraging Color Naming for Image Enhancement

**arXiv ID:** 2607.08185 | [PDF](https://arxiv.org/pdf/2607.08185v1)

**作者:** David Serrano-Lozano `[一作]` (Universitat Autònoma de Barcelona), Javier Vazquez-Corral `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 NamedCurves+，通过将图像按颜色命名分解为六个通道，学习每个通道的贝塞尔曲线进行全局色彩调整，并使用 Transformer 进行局部融合，实现可解释、可交互的图像增强；

**💡 创新点**

创新点包括：①使用颜色命名作为先验将图像分解为语义化颜色通道；②为每个颜色通道学习贝塞尔曲线实现可编辑的全局调整；③引入基于通道的转置注意力 Transformer 进行高效的局部融合；④支持多任务（照片美化、色调映射、曝光校正）并提供用户交互式曲线编辑；

**🔧 技术方法**

技术手段包括：轻量化 UNet+CBAM 背骨网络进行图像标准化；Van de Weijer 等颜色命名模型获取颜色概率图；贝塞尔曲线参数化得到平滑的调色曲线；转置注意力 Transformer 进行通道间的长程上下文融合；多任务损失（L2+SSIM）和颜色一致性损失；

**📊 数据集**

实验使用了 MIT‑Adobe‑5K、PPR10K、MSEC、SICE、ME 等公开数据集；

**📈 对比分析**

通过与 3DLUT、BGLUT、RSFNet、FECNet 等最新方法在 PSNR、SSIM、ΔE、LPIPS 等指标上对比，NamedCurves+ 在三大任务中均实现了最高 PSNR/SSIM、最低 ΔE，且推理速度比前代快 27%，整体性能领先；

**⚠️ 局限性**

局限性包括：全局贝塞尔曲线难以表达高度局部或语义级编辑；颜色命名分解在颜色边界或极端亮度下不确定，导致颜色泄漏；以及模型本身训练于专家风格，仍难完全满足所有用户的个性化审美。

---

## 49. Architecture Generalization with MetaNCA

**arXiv ID:** 2607.07743 | [PDF](https://arxiv.org/pdf/2607.07743v1)

**作者:** Meet Barot `[一作]` (Mythos Scientific), Sina Khajehabdollahi `[通讯]` (Independent Scholar)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MetaNCA框架，利用局部自组织规则在计算图上迭代更新权重，从而在不使用反向传播的情况下生成任意架构的神经网络。

**💡 创新点**

创新点在于首次将单一的权重Transformer局部规则在多种网络架构（MLP、CNN、ResNet）上进行元学习，实现跨架构的自组织生成，并通过压缩率展示其高效性。

**🔧 技术方法**

使用图神经元胞自动机、线性注意力Transformer（Weight Transformer）与前馈网络组合的局部规则网络，并通过BPTT和MuProp优化器训练。

**📊 数据集**

在MNIST和CIFAR-100两个公开数据集上进行实验，分别对密集MLP、卷积网络和ResNet架构进行评估。

**📈 对比分析**

与同等架构使用Adam训练的基线对比，MetaNCA在大多数MNIST架构上可获得与Adam相近的准确率（最高≈97%），在CIFAR-100上虽低于Adam（≈30% vs. 41%）但表现仍可接受，且在多架构泛化上表现优于单一训练。

**⚠️ 局限性**

局部规则对数据无条件，难以自适应不同任务；对训练容量之外的极小或极大网络泛化有限；以及在新架构族需要手工映射邻域定义，限制了方法的通用性。

---

## 50. Concretized Proposition Prompting Resolves Composition-Knowledge Dichotomy in Large Language Models

**arXiv ID:** 2607.08018 | [PDF](https://arxiv.org/pdf/2607.08018v1)

**作者:** Changhun Lee `[一作]` (Columbia University), Chiehyeon Lim `[通讯]` (UNIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Concretized Proposition Prompting（CPP），通过显式构造问题相关命题来提升 LLM 的推理性能。

**💡 创新点**

将组合性与知识性统一，使用四类命题（TP/TN/FP/FN）实现对推理路径的具体化，从而解决 Composition‑Knowledge Dichotomy。

**🔧 技术方法**

结合 DSPy 进行提示优化，使用命题模型生成命题、答复模型进行 CoT 推理，并用 Judge 模型评估命题质量。

**📊 数据集**

共八个基准：Commonsense（ARC‑E、ARC‑C、MMLU‑Pro、CSQA）、Math（GSM‑8K、MATH）、Medicine（EHRNoteQA、MedXpertQA）。

**📈 对比分析**

与零样本 CoT、直接答复、少样本等基线对比，在大多数数据集上 CPP 取得最高或接近最高准确率（如 ARC‑E 96.9%、EHRNoteQA 60.5%），并在不同模型与参数规模下表现稳健。

**⚠️ 局限性**

效果依赖模型与数据集，命题质量不足时可误导答案；医学数据仍低；两步流程相对耗时，未普适适用于所有场景。

---

## 51. PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs

**arXiv ID:** 2607.08269 | [PDF](https://arxiv.org/pdf/2607.08269v1)

**作者:** Ying Liu `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了 PolyUQuest，一个基于异构图的结构化检索增强生成（RAG）系统，能够统一网页超链接、DOM 层次和跨页实体关系，并通过两层路由将查询映射到三种结构匹配的检索模式，实现可验证的答案与可追溯的引用路径。

**💡 创新点**

创新点包括：① 统一的三层异构图模型，融合超链接、DOM 块和实体网络；② 结构驱动的检索模式路由，针对不同结构需求选择最优检索路径；③ 每条答案都附带完整的页面、标题路径和实体链接，保证答案可验证且来源透明。

**🔧 技术方法**

使用的技术主要包括：图建模与多层次索引、块级文本嵌入与稠密近似最近邻检索、BM25 与 cross‑encoder 重排序、LLM 分类器与生成器、实体抽取与消歧、两层路由算法。

**📊 数据集**

采用的主要数据集为香港理工大学（PolyU）官网的 4,240 页、31,086 DOM 块、29,119 实体、37,680 关系构成的结构化图谱，并在 300 个涵盖单页事实、跨页比较、实体推理的问答对上进行评估。

**📈 对比分析**

通过在同一 LLM、提示和嵌入模型下与 ChunkRAG、HtmlRAG、FastGraphRAG、LightRAG 进行对比，PolyUQuest 在答案正确性 0.644、覆盖率 0.649、可信度 0.921 上均位居榜首，同时每次查询平均消耗 2,968 tokens，远低于 LightRAG 的 29,825 tokens，显示了显著的性能与成本优势。

**⚠️ 局限性**

局限性包括：① 依赖手工定义的实体 schema 与领域知识，迁移到新域需额外工作；② 对非组织性或多语言网站的适应性不如专门设计的系统；③ 对极端长链或动态内容（如 AJAX 渲染）的支持有限。

---

## 52. Vanilla SGD with Momentum Survives Heavy-Tailed Noise: Convergence Analysis without Gradient Clipping or Normalization

**arXiv ID:** 2607.08104 | [PDF](https://arxiv.org/pdf/2607.08104v1)

**作者:** Ryusei Yamada `[一作]` (Meiji University), Hideaki Iiduka `[通讯]` (Meiji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了无梯度裁剪或归一化的 vanilla SGD 与带动量的 SGD 在存在重尾噪声时的收敛性，并给出了强凸、凸和非凸问题的期望收敛率。

**💡 创新点**

提出了在 Hölder 光滑（非 L‑smooth）条件下进行重尾噪声分析的框架，证明了仅满足 
   ν+1≤p 的情况下仍能保证收敛，并给出了与已知最优归一化方法相比的最差收敛速率，填补了该领域理论空白。

**🔧 技术方法**

使用 Hölder 光滑假设、von Bahr–Esseen 不等式、辅助序列与 Lyapunov 函数相结合的分析技术，推导了期望收敛率，并在实验中采用合成的 Pareto 级重尾噪声和实际的 WikiText‑2 Transformer‑XL 数据集进行验证。

**📊 数据集**

实验使用了三类合成目标函数（强凸、凸、非凸）以及 WikiText‑2 上的 6‑层 8‑头 Transformer‑XL 模型，噪声通过两侧 Pareto 分布模拟。

**📈 对比分析**

与已知的裁剪/归一化 SGD 进行对比，结果显示 vanilla SGD 能在不做梯度控制的情况下收敛，但收敛速率（如 𝒪(T^‑(p‑1)/2p) 或 𝒪(T^‑(p‑1)/4)）明显落后于归一化方法（如 𝒪(T^‑(p‑1)/3p‑2)）。

**⚠️ 局限性**

局限性包括：收敛速率不如归一化方法；需要满足 ν+1≤p 的假设，且 Hölder 光滑性在实际任务中难以估计；分析仅针对 vanilla SGD/动量版，未覆盖更复杂优化器；实验多基于合成数据，对真实大规模任务的适用性仍需进一步验证。

---

## 53. Distributed Sketching on Data Partitions for OLS Regression

**arXiv ID:** 2607.07888 | [PDF](https://arxiv.org/pdf/2607.07888v1)

**作者:** Luyuan Yang `[一作]` (University of Oklahoma), Chao Lan `[通讯]` (University of Oklahoma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出在普通最小二乘回归中，对数据进行分区后分别做随机高斯投影（sketching）并平均得到估计器，显著降低映射计算成本。

**💡 创新点**

创新点在于对分区子集进行sketching并给出其精确的过拟合损失公式，并引入子集协方差偏离度 D 与 Burg 散度的关系来解释估计器性能。

**🔧 技术方法**

使用的技术包括固定设计下的OLS、Gaussian 随机投影、矩阵分解与逆、离散化分区、相对杠杆得分、Burg 散度以及随机矩阵奇异值界定。

**📊 数据集**

实验使用的公开数据集包括 Digit（1797样本，61特征）、California Housing（20640样本，8特征）和 Covertype（100k样本，53特征）。

**📈 对比分析**

通过比较分区sketching、全数据sketching 与标准OLS 的平均过拟合损失，发现当子集协方差相近时分区方法损失更小，且损失随分区数 k 先增后减，时间复杂度随 k 下降。

**⚠️ 局限性**

局限性包括：若子集协方差差异大，分区sketching 的损失可能高于全数据sketching；需满足 p> d+1 并且数据矩阵满秩；随机分区可能破坏数据结构；在极端 k 值下性能不稳定。

---

## 54. Trustworthy Machine Learning through the Lens of Combinatorial Optimization: Survey and Research Perspectives

**arXiv ID:** 2607.07762 | [PDF](https://arxiv.org/pdf/2607.07762v1)

**作者:** Thibaut Vidal `[一作]` (Polytechnique Montreal), Julien Ferry `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了将组合优化方法应用于可信机器学习的最新研究，聚焦可解释性、鲁棒性、公平性等核心维度；

**💡 创新点**

创新点在于提出统一的框架与分类，系统梳理CO在模型训练、解释、压缩、审计等环节的进展，并强调其在全局最优、正式证书和明确权衡上的优势；

**🔧 技术方法**

综述所涉及的技术包括混合整数规划、约束规划、SAT/SMT、MaxSAT、B&B、列生成等组合优化方法；

**📊 数据集**

所引用的实验多基于经典公开数据集（如UCI、信用评分、医疗诊断、图像分类等），并未提供新的实验数据；

**📈 对比分析**

对比结果显示，与传统梯度或启发式方法相比，CO方法常能在准确率、稀疏度、可解释性和正式认证等指标上实现更优或更可靠的表现，尽管计算量更大；

**⚠️ 局限性**

主要局限在于问题规模与可扩展性，NP‑hard特性导致求解时间较长，尤其对大规模深度网络仍难以直接应用，需要进一步改进模型、分解与近似技术。

---

## 55. When Thinking Hurts: Epistemic Signals in the Reasoning Chains of Visual Language Models

**arXiv ID:** 2607.08059 | [PDF](https://arxiv.org/pdf/2607.08059v1)

**作者:** Mayank Singal `[一作]` `[通讯]` (Independent Researcher), Mayank Singal (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究思考模式视觉语言模型（VLM）在不确定性量化中的表现，揭示答案熵在产生思考链后会崩塌，证明思考链熵和链长度可以作为更可靠的幻觉预测信号。

**💡 创新点**

创新点在于首次对不同思考模式模型（Qwen、GLM、InternVL3）进行系统实验，发现三种不同的熵崩塌模式（完全崩塌、无崩塌、选择性思考），并提出利用单前向推理即可获取的链熵和链长度作为低成本、不需额外采样的幻觉检测方法。

**🔧 技术方法**

采用熵统计（H_t）、链长度 L 计算，使用AUROC评估幻觉检测性能，并对不同模型的链生成率进行实验对比。

**📊 数据集**

使用的基准数据集包括 POPE 对抗样本、HallusionBench 视觉推理题目，以及 300 条 VQAv2 开放式问题，用于验证链信号在二分类与开放式问题上的通用性。

**📈 对比分析**

在 POPE 上，Qwen 链熵 AUROC 0.647 超过答案熵 0.492；GLM 链熵 0.759 超过答案熵 0.716；链熵可将准确率从 71.0% 提升至 93.8%（覆盖率 62.7%），表明链信号在实际部署中具备显著优势。

**⚠️ 局限性**

局限性包括仅评估 8–9B 规模模型、仅使用贪婪解码、缺乏对采样解码和更大模型的验证、链熵与链长度相关性有限、对图像模糊/多义性等固有不确定性未能完全分离、以及在开放式任务上样本量不足。

---

## 56. Homomorphism Indistinguishability Beyond Graphs: Relational Weisfeiler--Leman and Hypertree Width

**arXiv ID:** 2607.07934 | [PDF](https://arxiv.org/pdf/2607.07934v1)

**作者:** Panagiotis Aivasiliotis `[一作]` (University of Potsdam), Marc Roth `[通讯]` (Queen Mary University of London)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并研究了两种针对关系型数据的k-WL（Weisfeiler–Lehman）算法，分别对应传统的关系颜色细化算法（k-CL）和其“分数”版本（k-FL），并给出了它们与关系图同构区分的等价性以及对关系数据的归纳性性质的完整证明。

**💡 创新点**

创新点在于将k-WL算法从二元关系扩展到任意高元关系，实现了对k-WL等价性与关系型数据中可分辨的图结构的严格数学对应关系，并首次提出k-FL与半纯分数树形分解（semi‑pure fractional hypertree decomposition）之间的同调判定。

**🔧 技术方法**

主要技术包括：利用k-CL与k-FL对关系元组进行编码的k-爆炸（k‑exponential）与分数k-爆炸（fractional k‑exponential）构造；通过树形分解（tree decomposition）与超树形分解（hypertree decomposition）诱导二元结构；应用一阶WL（1‑WL）与颜色细化等价性理论；以及对分数树形分解进行纯度与半纯度的约束以保证构造的有效性。

**📊 数据集**

由于本文为理论性工作，未使用具体实验数据集，而是通过符号与结构化证明来阐明算法与判定之间的等价关系。

**📈 对比分析**

方法对比是以理论证明为主，并未给出运行时间或内存消耗的实验评估；但从理论上可推断k‑CL与k‑FL的时间复杂度均为O(|A|·(k+σ)·n^k)，其中n为节点数、σ为符号数量，具有可预期的多项式时间性。

**⚠️ 局限性**

局限性包括：算法的适用范围仅限于无孤立节点的关系型结构；对高元关系的k‑CL与k‑FL在实现时需要构造有限但指数级的k‑exploded对象；以及半纯分数树形分解的假设限制了对更一般分数树形分解的适用性。

---

## 57. When Debiasing Backfires: Counterintuitive Side Effects of Preprocessing-Based Stereotype Mitigation

**arXiv ID:** 2607.07937 | [PDF](https://arxiv.org/pdf/2607.07937v1)

**作者:** Yahan Zheng `[一作]` (Dartmouth College), Weicheng Ma `[通讯]` (Oakland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究预处理式去偏方法在语言模型中的副作用，评估其对不同群体偏见的影响。

**💡 创新点**

首次系统性揭示去偏数据预处理会在目标群体上降低偏见的同时，意外增加或改变其他群体（甚至无关群体）的偏见，且此效应不受模型规模或预/后训练阶段的控制。

**🔧 技术方法**

采用数据去偏策略（去除刻板句子、去除群体提及、群体替换）并利用注意力回滚（attention‑rollout）进行机制诊断，结合StereoSet、CrowS‑Pairs的偏见评测。

**📊 数据集**

使用2023年6月1日的英文维基百科数据集，结合StereoSet与CrowS‑Pairs测试集。

**📈 对比分析**

与基线模型对比，预/后训练的去偏模型在目标群体的SS（偏见得分）显著下降、LMS与iCAT保持不变或略增；然而，非目标群体的SS往往出现离散的升高或降低，表现出明显的负面副作用。

**⚠️ 局限性**

局限在于只评估英语维基百科、六个身份群体、少量模型（TinyBERT、GPT‑2、LLaMA‑2 7B）和单一随机种子；基准覆盖不足，可能遗漏其他偏见与语言多样性。

---

## 58. LiST: Lipschitz Scaling Training for Robust and Calibrated Neural Networks

**arXiv ID:** 2607.07745 | [PDF](https://arxiv.org/pdf/2607.07745v1)

**作者:** Arthur Chiron `[一作]` (IRIT), Mathieu Serrurier `[通讯]` (IRIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Lipschitz Scaling Training（LiST）算法，通过动态调节全局 Lipschitz 常数并以温度标度校准为反馈，实现一次性同时满足准确率、鲁棒性和校准；

**💡 创新点**

创新点在于揭示 Lipschitz 常数与温度标度的结构对偶性，将校准作为无监督的选择点，自动寻找最佳 Lipschitz 常数 L*，并通过偏置参数构造完整的校准 Pareto 前沿；

**🔧 技术方法**

使用了 Lipschitz‑约束的神经网络、温度标度（Temperature Scaling）、自动攻击评估（AutoAttack）、以及自适应 Lipschitz 约束更新策略；

**📊 数据集**

实验数据集包括 CIFAR‑10、CIFAR‑100 和 Tiny‑ImageNet；

**📈 对比分析**

与无约束基线（标准、Label‑Smooth、Focal Loss、PGD‑AT、CAAT）以及固定 Lipschitz 常数（L∈{1,2,4,8,16,32,64,128}）对比，LiST 在准确率、ECE 校准误差和认证鲁棒性（CRA、AA）方面表现出最优或竞争性，且无需后处理；

**⚠️ 局限性**

局限包括 L* 的唯一性未得到证明、L* 难以跨模型/任务迁移、且校准点不一定对应最高鲁棒性，需要通过偏置参数手动调节。

---

## 59. ReCoLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning

**arXiv ID:** 2607.07719 | [PDF](https://arxiv.org/pdf/2607.07719v1)

**作者:** Wentao Lu `[一作]` `[通讯]`, Wentao Lu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向持续微调的低秩适配器框架 ReCoLoRA，并引入递归整合机制以在单一模型中累积任务知识；

**💡 创新点**

创新点在于：①利用随机 SVD 对预训练权重进行谱感知初始化；②通过拐点法动态选择每层有效秩；③采用两阶段主子空间-残差训练；④在连续任务间递归重分解，将已完成任务的知识压缩到慢速主子空间，从而避免“覆盖”遗忘；

**🔧 技术方法**

技术包括：低秩适配器（LoRA）、随机 SVD、拐点 (elbow) 机制、两阶段训练、递归整合以及可选的任务银行（TaskBank）方案；

**📊 数据集**

使用 GLUE 连续任务序列（SST-2 → MRPC → QNLI → RTE → QQP → MNLI）以及四款 7–8B 大模型（Qwen3‑8B、Llama‑3.1‑8B‑Instruct、Mistral‑7B‑v0.3、InternLM2.5‑7B‑Chat）；

**📈 对比分析**

与 LoRA、PiSSA、AdaLoRA、DoRA、O‑LoRA 等 PEFT 基线以及 rank‑sweep 方案进行对比；ReCoLoRA 在 Qwen3‑8B、Mistral‑7B‑v0.3 与 InternLM2.5‑7B‑Chat 上实现了最高的最终平均分，且参数量更少；在 TaskBank（oracle 路由）上几乎无遗忘；

**⚠️ 局限性**

局限性包括：①TaskBank 仅为上界，需学习路由器实现无任务标识部署；②递归整合在不同 backbone 上的效果不均衡（如 Llama‑3.1‑8B‑Instruct 仍略逊）；③实验仅覆盖 GLUE 任务顺序，缺乏更长、不同任务顺序和指令型任务的验证；

---

## 60. Drift-Aware Temporal Graph Rewiring (DATGR) for Adaptive Semantic Modeling in Biomedical Text

**arXiv ID:** 2607.08490 | [PDF](https://arxiv.org/pdf/2607.08490v1)

**作者:** Bharathwaj Vijayakumar `[一作]` (Rowan University), Sahana K. Varadaraju `[通讯]` (Rowan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于语义漂移的临时图重连框架 DATGR，利用轻量级逻辑回归规则在词共现图中动态更新边权，捕捉生物医学文本中的概念演化。

**💡 创新点**

创新点在于：①将语义漂移直接映射到边权更新，而非重新训练词向量；②使用闭式逻辑更新规则平衡历史惯性与漂移驱动的创新；③通过 top‑k 稀疏化保持结构可解释性和计算效率。

**🔧 技术方法**

核心技术包括：基于句子嵌入的漂移估计、逻辑回归边权更新规则、Node2Vec 节点嵌入、点积特征 + 逻辑回归链路预测、稀疏化与性能评估。

**📊 数据集**

在 BIOMRC 生物医学多关系语料库上进行实验，将语料分为四个时间窗口，每窗口约1000 篇摘要；使用400个高频词构建共现图。

**📈 对比分析**

与静态共现图基线对比，DATGR 在 AUROC 上平均提升约0.066（0.699 vs 0.633），AUPRC 维持相近（0.744 vs 0.738）；更新复杂度为 O(|E|)，显著低于全局重训练或 DGNN 方法。

**⚠️ 局限性**

局限性：实验仅在四个粗略时间窗口的人工分段数据上验证；漂移估计依赖于句子嵌入的质量，可能对短文本或低频词效果有限；缺乏对真实时间序列（如 PubMed 2000‑2024）的评估，未探讨多模态信息或更复杂语义变迁。

---

## 61. A Graph Neural Network Model for Real-Time Gesture Recognition Based on sEMG Signals

**arXiv ID:** 2607.07850 | [PDF](https://arxiv.org/pdf/2607.07850v1)

**作者:** Pragatheeswaran Vipulanandan `[一作]` (University of Miami), Manohar Murthi `[通讯]` (University of Miami)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于sEMG信号的无向加权图网络模型，实现对五种手势的实时识别。

**💡 创新点**

创新点在于用皮尔逊相关矩阵构造图网络，捕捉肌肉激活的空间与时间关系，并通过事件检测筛选有效窗口，利用简单的GNN实现极低延迟的实时分类。

**🔧 技术方法**

使用技术包括图神经网络（GNN）、图卷积、图信号处理、Pearson相关矩阵构建、阈值事件检测、PyTorch训练与评估。

**📊 数据集**

使用Myo Band 8通道sEMG数据集，8名受试者，5个手势，共320条时间序列，采样率200 Hz。

**📈 对比分析**

与连续分箱+LDA、肌肉协同+RNN、TMA+CNN等方法对比，平均准确率达99 %，最高99.91 %，实时延迟仅48 ms，显著优于现有技术。

**⚠️ 局限性**

局限性包括：样本量仅8人，缺乏多样性；阈值需个体化；对极低信噪比或运动伪差的鲁棒性尚未充分验证。

---

## 62. Primal-Dual Online Algorithms for the Parking Permit Problem

**arXiv ID:** 2607.08262 | [PDF](https://arxiv.org/pdf/2607.08262v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Alex Turoczy `[通讯]` (University of Oxford)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

重新审视停车许可证问题（PPP），使用原始-对偶方案获得简单算法，提供更好的性能保证。

**💡 创新点**

直接处理问题结构，避免了以往工作中降低竞争比的简化，提供了接近匹配的下界。

**🔧 技术方法**

使用原始-对偶算法框架。

**📊 数据集**

未具体提及数据集，但讨论了不同类型的停车许可证及其持续时间和成本。

**📈 对比分析**

与之前的工作相比，新的确定性竞争比为K，随机竞争比为ln K + lnln K + O(1)，显著改善了性能。

**⚠️ 局限性**

算法在处理非层叠结构时的复杂性，可能导致竞争比的上界未能达到最优。

---

## 63. Finite Convergence of the Modal Mu-Calculus on Almost-Periodic Words

**arXiv ID:** 2607.08181 | [PDF](https://arxiv.org/pdf/2607.08181v1)

**作者:** Fabian Lehr `[一作]` (TU Munich), Florian Bruse `[通讯]` (TU Munich)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并证明了：模态 μ-算子在无限字上的有限收敛性正好对应于该字的几乎周期性。

**💡 创新点**

创新点在于给出了从几乎周期性到有限收敛性的直接、简洁证明，并由此恢复了 Semenov ’84 的可判定性结果；同时提供了 ω-正则表达式在此类字上的规范化形式。

**🔧 技术方法**

核心技术包括：将 μ-算子公式转换为“Trivial Automaton”，利用自动机与正规表达式的等价性；引入最小化、良好与简单 NFA 的概念；通过前缀自由性与有限运行性质进行组合论证明；以及利用 KMP 算法构造对应的自动机。

**📊 数据集**

本研究为理论性工作，未使用具体数据集；研究对象为无穷字序列，主要以构造性证明为主。

**📈 对比分析**

与 Semenov ’84 的可判定性结果相比，本文提供了更直接的证明路径；在理论上给出了可计算的界（尽管上限为非元算），但并未给出实验性能指标。

**⚠️ 局限性**

局限性：结果仅适用于单向无限字，未覆盖双向无穷字或树结构；对更复杂结构（如无限树）的推广仍是未来研究方向。

---

## 64. Bug Report Specification Refinement with Trajectory Guidance for Automated Program Repair

**arXiv ID:** 2607.07882 | [PDF](https://arxiv.org/pdf/2607.07882v1)

**作者:** S M Farah Al Fahim `[一作]` (Concordia University), Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于轨迹引导的仓库级错误报告规范细化方法，通过对未验证的轨迹收集运行产生的轨迹进行层级化抽象和仓库审查，生成更完整的错误报告，从而提升自动程序修复的效果。

**💡 创新点**

创新点在于①利用未验证的轨迹收集运行的轨迹作为可靠的规格证据来源，而非直接采用生成的补丁；②将证据组织为三层层级（高、中、低）并覆盖失效机制、行为需求和实现范围；③引入仓库基准审查来剔除无效或不确定的声明，确保细化报告的可信度。

**🔧 技术方法**

使用的大型语言模型（如GPT‑5‑mini、MiniMax M2.5）进行轨迹收集、证据抽取、报告生成与审查；结合静态/动态工具搜索源码；对轨迹进行结构化处理以形成层级化证据；最后将细化报告交给下游自动修复代理。

**📊 数据集**

评估基准为SWE‑Bench Lite（300个Python开源项目问题实例），并在其子集100个实例上进一步验证。

**📈 对比分析**

与原始错误报告进行对比，并与多种下游修复代理（Mini‑SWE‑Agent V2、Agentless、AutoCodeRover）一起评估。性能提升：Pass@1从41%提升至59.67%（GPT‑5‑mini）或64.33%（MiniMax M2.5）；在子集上提升至71%（Agentless）或72%（AutoCodeRover）。

**⚠️ 局限性**

局限性包括：①仅在Python项目上验证，跨语言效果未知；②依赖轨迹收集代理的轨迹质量，噪声多时需进一步过滤；③未对细化报告的可读性、可维护性等人类评估进行系统研究。

---

## 65. Overthinking: Amplifying Reasoning Weights to Extract Learned Secrets

**arXiv ID:** 2607.08173 | [PDF](https://arxiv.org/pdf/2607.08173v1)

**作者:** Jack Hopkins `[一作]` (Anthropic Fellows Program), Fabien Roger `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对语言模型权重进行推理任务向量放大（overthinking），在不同模型规模下提取隐藏的秘密信息。

**💡 创新点**

提出了推理放大与层级衰减策略，证明白盒权重调节可以显著提升模型审核成功率。

**🔧 技术方法**

使用任务向量算子、α倍推理向量叠加、层级冻结/线性衰减/Fisher加权衰减，以及与prefill攻击的组合。

**📊 数据集**

基于Qwen3‑VL 2B‑32B模型，构造四类秘密实验（MMLU隐秘、Taboo词、Gender信念、SSC行为）并对比随机噪声基线。

**📈 对比分析**

在α∈[0,4]的多次实验中，最佳α≈2‑3时泄露率提升10‑30%；Fisher加权策略获得最高泄露率；prefill与overthinking叠加可将成功率提高至约97%。

**⚠️ 局限性**

仅在Qwen3‑VL体系上验证，缺乏跨架构通用性；秘密检测依赖关键词和LLM判定，易漏报；层级Fisher近似、样本量有限，未涵盖更广泛的秘密类型。

---

## 66. Conversational Retrieval and On-the-Fly Knowledge Modeling of Historical Penitentiary Repression Records

**arXiv ID:** 2607.08459 | [PDF](https://arxiv.org/pdf/2607.08459v1)

**作者:** Paula Font Solà `[一作]` (Universitat Autònoma de Barcelona), Josep Lladós `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于对话的历史数字图书馆信息检索系统，支持实时知识建模与知识图谱构建，利用 OCR+LLM 的多阶段文本提取、RAG 与图谱融合的检索与生成，能够跨文档进行多跳推理。

**💡 创新点**

提出端到端的动态知识图谱生成与存储机制，可在交互过程中持续更新并作为 LLM 的记忆；同时融合多候选投票+LLM 校正的置信度感知文本提取，显著降低错误率并抑制 hallucination。

**🔧 技术方法**

使用 Tesseract OCR 与多候选投票、Surya 版式分析、TrOCR、phi-4 LLM 进行文本纠错与问答、图谱抽取；利用 SentenceTransformer + cross‑encoder 进行 RAG 向量检索；借助 Neo4j/图数据库实现知识图谱构建与推理。

**📊 数据集**

采用 130 张西班牙内战（1937‑1940）时期的军法判决文件（共 65 案）作为评估数据；手工标注 5 份多页文档用于 OCR 评估，构建 222 组基于人名属性的问答用于 RAG 评估。

**📈 对比分析**

OCR 通过完整管线将 WER 从 72.8% 降至 33.5%，CER 从 46.7% 降至 22.4%；在 125 组过滤问题上 Custom Exact Match 0.824，Faithfulness 0.886，Context Precision 0.888，Context Recall 0.776；在人类评估中，Graph 在复杂查询上 87.5% 的回答率，RAG 在简单查询上 83.3%；整体用户满意度 4.12/5。

**⚠️ 局限性**

OCR 仍有高错误率，LLM 校正可能引入非原始词汇导致错误；知识图谱在跨文档实体统一时仅限人名、地点、日期，其余实体保留文档 ID；系统规模与推理复杂度尚未验证大规模档案；缺乏长期评估与真实专家协作流程。

---

## 67. Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum to Improve a Coding Student Beyond Imitation

**arXiv ID:** 2607.08255 | [PDF](https://arxiv.org/pdf/2607.08255v1)

**作者:** Miseong Shawn Kim `[一作]` `[通讯]` (Genesis Cortex AI Inc), Miseong Shawn Kim (Genesis Cortex AI Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四大前沿LLM教师（Claude、Codex、Grok、Gemini）进行执行验证竞赛，构建可验证的编程练习课程，并用该课程训练学生模型。

**💡 创新点**

提出竞争-协作框架：先用无偏执行验证对教师进行公平排名，再用教师验证过的任务作为可验证奖励环境，证明RL优于模仿学习。

**🔧 技术方法**

使用执行验证判别器、教师自校正、交叉控制、SFT（提示微调）、GRPO强化学习、LoRA技术以及Qwen2.5-Coder学生模型。

**📊 数据集**

使用MBPP函数任务、改造的bug修复任务、难度6–9的竞赛题库以及对应的隐藏单元测试。

**📈 对比分析**

通过在同一任务库上统计通过率来公平比较教师，Gemini在难题上领先；SFT在学生上导致性能下降；RLVR在竞争题上相对提升约49%。

**⚠️ 局限性**

存在MBPP饱和、教师泄漏风险、任务偏差、RLVR提升有限、仅单语言单学生、硬件限制等局限。

---

## 68. Canonical Join Trees

**arXiv ID:** 2607.07992 | [PDF](https://arxiv.org/pdf/2607.07992v1)

**作者:** Arne Leitert `[一作]` `[通讯]`, Arne Leitert

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究并解决了无环超图中根节点是否存在唯一规范连接树（canonical join tree）的判定问题，并给出了完整的理论刻画；

**💡 创新点**

创新点在于证明规范连接树唯一性、与并连接图和弧形子图（bowtie）的等价关系，并提出线性时间构造算法；

**🔧 技术方法**

主要技术包括并连接图（union join graph）的构造、Kruskal算法与回边检测、深度优先/后向回边（2‑layer back edges）分析；

**📊 数据集**

本文为理论工作，未使用具体实验数据集；

**📈 对比分析**

性能方面通过算法复杂度分析证明构造过程为线性时间；

**⚠️ 局限性**

局限性在于仅能在存在规范连接树时构造，对所有根都存在的判定仍与Sperner问题难度相当，且缺乏实验验证。

---

## 69. Beyond Thermal Imaging: Inferring Thermophysical Properties from Time-Resolved Thermal Observations

**arXiv ID:** 2607.07962 | [PDF](https://arxiv.org/pdf/2607.07962v1)

**作者:** Chenghao Xu `[一作]` (École polytechnique fédérale de Lausanne), Olga Fink `[通讯]` (École polytechnique fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `4de8e9d8-757b-475f-9627-18a445e50202` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一套名为ThermoField的框架，利用多视角RGB重建几何体，并将时序热成像视为热传递过程，通过可微有限元仿真联合推断空间可变的热物理量（热扩散率、对流换热系数等）。

**💡 创新点**

创新点在于将热成像的时间信息与几何重建、可微热传递模型统一到一次性逆问题中，使用神经场表示空间热物理量，并通过自动微分直接更新这些物理场，实现在复杂3D场景下可解释的热物理推断与预测。

**🔧 技术方法**

采用SDF（签名距离函数）实现的几何重建、JAX‑FEM实现的可微有限元求解、神经网络控制点表示热物理场、线性到二次元素的分阶段求解以及平滑正则化。

**📊 数据集**

使用了在ANSYS中模拟得到的三种简单几何体和三种复杂物体的合成数据集，包含多种加热/冷却条件下的时间分辨热序列和已知的材质参数。

**📈 对比分析**

通过与基线（如均匀中值估计）对比，评估了恢复的热扩散率相对误差、预测误差（MAE、RMSE）以及跨温度条件的泛化性能。结果表明，在受控加热条件下误差可控（约10–30%），并能在未见的温度条件下保持较低预测误差；但在被动冷却或金属高扩散率场景下误差增大。

**⚠️ 局限性**

主要限制包括：参数可辨识性受热序列信息量限制，金属高扩散率物体或被动冷却过程难以区分热扩散率；几何重建误差、温度标定误差以及单一热观测的局限性导致不同参数配置能产生相似热响应，需进一步加入更丰富的激励、传感或先验约束。

---

## 70. DeepSWE: Measuring Frontier Coding Agents on Original, Long-Horizon Engineering Tasks

**arXiv ID:** 2607.07946 | [PDF](https://arxiv.org/pdf/2607.07946v1)

**作者:** Wenqi Huang `[一作]` (Datacurve), Serena Ge `[通讯]` (Datacurve)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于真实开源仓库的长周期软件工程任务评测基准，所有任务均为从零编写且未被合并，配合手写功能验证器，评估编码代理在多文件、跨语言项目中的自适应能力。

**💡 创新点**

创新点在于：①任务完全原创并且不在训练数据中出现，消除“记忆”偏差；②覆盖五种主流语言与众多活跃仓库，提升多样性；③提示长度短却要求的实现代码量大，真正考验探索与理解能力；④使用功能验证器而非继承的 PR 测试，显著降低误判率，扩大模型排名区分度。

**🔧 技术方法**

技术上采用统一的共享 harness（SWE-agent）对所有模型进行评测，使用 pass@1 与 pass@4 两种基于候选数的成功率指标；评测过程中还引入 LLM‑as‑judge 审计以评估验证器的准确性；模型配置涉及多种大型语言模型与不同 reasoning‑effort 设置。

**📊 数据集**

数据集为约 2,000 条任务，来源于 5 种语言（TypeScript、Go、Python、JavaScript、Rust）的 100+ 活跃开源项目（≥ 500 星、MIT/Apache 许可证）。每个任务包含提示、可执行验证器和参考实现，所有任务都已过人工与 LLM 审核。

**📈 对比分析**

评测显示 GPT‑5.5 以约 70% pass@1 领先，pass@4 进一步提升；与公开基准相比，模型间得分差距从 29.7% 扩大至 69.8%，说明此基准能更清晰地区分前沿代理；在 token、耗时和成本维度上，无显著正相关，表明高准确率不一定伴随更高资源消耗。

**⚠️ 局限性**

局限包括：①采用二元通过/失败评价，无法捕捉部分实现的质量或进展；②只关注功能正确性，不评估代码可维护性、性能或安全；③提示长度虽短但仍比日常交互更完整，可能不完全映射真实使用场景；④固定 harness 可能对某些模型的原生编辑能力产生不等效影响；⑤验证器审计样本有限，误判率估计带有较大不确定性。

---

## 71. Grounded Event Extraction from SEC 8-K Filings with a Fine-Grained Taxonomy

**arXiv ID:** 2607.08346 | [PDF](https://arxiv.org/pdf/2607.08346v1)

**作者:** Rian Dolphin `[一作]` (Massive.com), Quinton Pike `[通讯]` (Massive.com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套基于大语言模型的两阶段系统，对2022-2026年SEC 8-K文件进行三层119类事件标签，附带对应的引用句和质量分数。

**💡 创新点**

创新点在于：①使用结构约束和模糊n-gram引用验证保证标签可追溯；②第二阶段独立评分产生可调精度的质量分；③将标签与市场事件研究结合，验证其经济意义。

**🔧 技术方法**

采用 instruction-tuned 大语言模型进行标签提取；schema-constrained 输出与模糊n-gram引用校验；第二阶段评分模型对引用句与类别定义进行匹配；LLM 判别者进行内部评估；事件研究方法用于外部验证。

**📊 数据集**

数据集包括292,984份2022-2026年Form 8-K文件、对应的SEC item codes以及美国上市公司日常行情数据；系统输出601,088条已验证的事件标签。

**📈 对比分析**

评估方式：使用LLM判别者对5,125条样本进行精度和质量分的分层评估，精度从12%提升到96%；事件研究显示标签相比item code更能解释异常回报，解释方差提升1.3个百分点。

**⚠️ 局限性**

局限性：引用验证无法捕捉推理错误；评分模型与提取模型同族，存在自相似风险；缺乏时间戳导致两天窗口，可能低估极端波动；数据起始仅2022年，宏观覆盖有限；高置信度标签比例仍不高。

---

## 72. Learning $\mathsf{AC}^0$ under Locally Sampleable Graphical Models

**arXiv ID:** 2607.08303 | [PDF](https://arxiv.org/pdf/2607.08303v1)

**作者:** Weiming Feng `[一作]` (University of Hong Kong), Yiyao Zhang `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出了一种在任意最大度为 Δ 的两相格点模型（Gibbs 分布）下学习 AC^0 逻辑电路的算法，利用局部采样器和系统扫描格点动力学实现了低阶多项式的 L² 逼近，进而得到样本复杂度为 n^{log^O(d)(n/ε)} 的学习器。

**💡 创新点**

创新点在于消除了传统方法对格点模型的多项式增长（polynomial‑growth）假设，仅需局部采样器的指数衰减性质即可得到低阶多项式逼近，从而扩展了学习范畴至任意图形图、尤其是硬约束（硬盘模型）与软约束（近临界 Ising 模型）等两相系统。

**🔧 技术方法**

主要技术包括系统扫描格点（systematic‑scan）格点动力学、采样‑逆采样器对（sampler‑inverter pair）构造、自动机化的局部采样器、路径截断与低阶多项式逼近，以及针对 L^p 误差的低阶回归理论。

**📊 数据集**

论文为理论分析，未使用具体实验数据集；所有结论均基于抽象的格点模型与随机过程理论推导。

**📈 对比分析**

与以往需要格点模型满足强烈的衰减（strong spatial mixing）或多项式增长的学习框架相比，本方法仅依赖于局部采样器的指数衰减，理论上实现了更广泛的学习条件，样本复杂度和运行时间保持在 n^{log^O(d)(n/ε)}，与已有基于多项式增长的结果具有可比的效率，且不再受几何结构限制。

**⚠️ 局限性**

主要局限在于仍需图模型满足双向局部采样器的 B‑good 条件，该条件在某些接近临界点的格点模型（如 Ising 的临界区间）可能难以验证；此外，方法的常数因子较大，对实际实现的可行性与效率仍需进一步实验验证。

---

## 73. Multimodal Unlearning Across Vision, Language, Video, and Audio: Survey of Methods, Datasets, and Benchmarks

**arXiv ID:** 2607.07907 | [PDF](https://arxiv.org/pdf/2607.07907v1)

**作者:** Nobin Sarwar `[一作]` (University of Maryland, Baltimore County), Vaidehi Patil `[通讯]` (UNC Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文系统综述了多模态基础模型的“unlearning”（有选择性地遗忘）技术，提出了以干预阶段和控制通路为主的系统级分类框架，涵盖视觉、语言、音频和视频四大模态，并对相关方法、数据集、评估指标及应用场景进行统一梳理。

**💡 创新点**

创新点在于将多模态遗忘研究从以算法为中心转向以系统级视角组织，明确了干预点（数据端、训练时、架构约束、训练自由、解码时）与目标（实例级/概念级）之间的关系；同时提供了跨模态可比的评价维度和开源资源，为后续研究提供了统一的基准与参考。

**🔧 技术方法**

采用文献综述、分类与对比的方法，整合了数据侧扰动、训练时梯度/约束、架构剪枝、训练自由的线性/投影编辑以及解码时的引导/条件控制等多种技术路线，形成了完整的多模态遗忘技术谱系。

**📊 数据集**

综述了多类数据集：身份/面部/情感/动作视频数据集，个人化与版权移除相关的数据集，语音与安全相关的公开基准，图像分类与分割的类别级遗忘数据集，以及 Web‑scale 语料清理数据等。

**📈 对比分析**

通过对已发表的基准与评估框架（遗忘强度、保留性能、效率、可逆性、鲁棒性等）的对比，论文指出不同方法在各维度表现各异；总体而言，训练自由与解码时方法在保留性能与效率上表现优越，但在严格的可逆性和多概念遗忘方面仍有不足。

**⚠️ 局限性**

局限性包括：未能覆盖最新的工作（尤其快速发展的领域），对算法细节与实现细节缺乏深入讨论，评价指标与基准仍缺乏统一标准，且对时间序列、表格、传感器等结构化或流式数据的遗忘研究关注不足。

---

## 74. CTA-Pipelining: A Latency-Oriented Spatial Scaling Method for Multi-GPU Systems

**arXiv ID:** 2607.07862 | [PDF](https://arxiv.org/pdf/2607.07862v1)

**作者:** Tingkai Liu `[一作]` (University of Illinois Urbana Champaign), Volodymyr Kindratenko `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了CTA‑pipelining，一种利用多GPU共享内存系统在CTA级别上并行执行相互依赖核的延迟导向空间流水线技术，用于降低LLM推理的单批延迟。

**💡 创新点**

创新点包括：① 在Cooperative Thread Array级别实现跨GPU的依赖同步，最大限度隐藏同步开销；② 将此模式与warp‑specialized多阶段持久化内核无缝集成；③ 证明CTA‑pipelining可与Tensor Parallelism互补，构成独立的空间缩放维度。

**🔧 技术方法**

核心技术包括：CUDA原子计数器、依赖数组、scoreboard、跨设备工作队列、系统级线程屏障（threadfence）以及探讨的Lamport同步；实现上依赖CUTLASS、cuBLAS、NCCL库，运行在8‑GPU H200/B200 NVLink系统。

**📊 数据集**

实验使用的“数据集”是多层GEMM（即MLP层）矩阵乘法，输入尺寸为16384×8192、8192×8192，采用BF16/FP32数据类型；并未使用真实LLM权重或文本数据。

**📈 对比分析**

评估方法：在同一硬件平台上与微批处理（micro‑batching）和Tensor Parallelism（TP）进行对比，使用CUDA Graph减少内核启动瓶颈。结果显示：CTA‑pipelining 对比微批处理可降低31.8%延迟，对比TP可降低29.6%延迟；与TP组合后可进一步压缩通信开销，整体延迟进一步下降。

**⚠️ 局限性**

局限性：① 需要足够多的 kernel 以形成有效流水线，单波CTA或极小尺寸核时效果有限；② 当前NVLink拓扑为中心化交换，导致跨GPU写入与其他通信竞争；③ 依赖软件层面的 prologue/epilogue 注入，缺乏硬件层面支持的 Lamport 同步；④ 仅在实验用的矩阵乘法场景验证，其他工作负载需进一步验证。

---

## 75. Cross-Modal Generative Framework for Signal Translation from Fetal-Maternal Electrocardiograms to Fetal Doppler Waveforms

**arXiv ID:** 2607.08073 | [PDF](https://arxiv.org/pdf/2607.08073v1)

**作者:** Tongli Su `[一作]` (Emory University), Nasim Katebi `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了利用胎儿-母体双通道ECG生成胎儿多普勒波形的跨模态生成框架。

**💡 创新点**

创新点在于将交叉模态注意力与自注意力相结合，能够有选择性地融合母体ECG信息，分解可由电信号恢复的多普勒成分与纯机械成分。

**🔧 技术方法**

使用了膨胀卷积、交叉模态注意力、自注意力以及复合损失函数（MAE+导数误差+相关性）。

**📊 数据集**

基于NInFEA数据库，包含39例妊娠，共885段同步的胎儿/母体ECG与多普勒信号。

**📈 对比分析**

通过5折交叉验证与多种指标（PSD MSE、DTW、相关系数）比较，交叉+自注意力模型将PSD MSE降低至49.9 dB²，心率误差≈4.7 bpm，性能明显优于单通道或仅拼接双通道模型。

**⚠️ 局限性**

局限在于对机械因素驱动的临床指标（PI、RI等）重建效果差，模型主要捕获电耦合成分；缺少对病理妊娠的验证。

---

## 76. ScopeJudge: Cost-Aware Pre-Execution Gating for Offensive Security Agents

**arXiv ID:** 2607.07774 | [PDF](https://arxiv.org/pdf/2607.07774v1)

**作者:** Shane Caldwell `[一作]` (dreadnode), Will Pearce `[通讯]` (dreadnode)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了ScopeJudge框架，用于在攻击性安全代理执行前对工具调用进行成本友好的范围检查，防止越界操作。

**💡 创新点**

首次构建了基于专家标签的调用级范围违规基准，并系统研究了不同上下文策略对LLM审计器准确性和成本的影响，证明仅靠静态政策无法有效监控。

**🔧 技术方法**

使用LLM审计器（Open-weight和专有模型）、多种转录策略（静态、意图、历史、摘要、完整）以及对话式预执行门控逻辑实现监控。

**📊 数据集**

ScopeJudge数据集：从ScopeBench任务生成的代理轨迹，包含约3000+工具调用，已由专业渗透测试人员标注。

**📈 对比分析**

对8个审计器与5种策略的组合进行基准评估，发现最优开放模型GLM‑5.2在成本约$0.006/调用时实现F1≈0.66，接近专家一致性F1≈0.78，专有模型相对成本更高。

**⚠️ 局限性**

限制包括单一校准集、未测量对抗鲁棒性、评估者重叠、基准违规率特定、未给出置信区间，且只覆盖二元违规判定，未考虑更细粒度的风险评估。

---

## 77. From Legacy Documentation to OSCAL: An MCP-Based Agent Pipeline for Threat-Informed Continuous Compliance in Critical Infrastructure

**arXiv ID:** 2607.08288 | [PDF](https://arxiv.org/pdf/2607.08288v1)

**作者:** Lea Roxanne Muth `[一作]` (Freie Universität Berlin), Marian Margraf `[通讯]` (Freie Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个八阶段的多智能体管线，利用Model Context Protocol（MCP）将自然语言的关键基础设施文档转换为可直接用于审核的OSCAL系统安全计划（SSP）和安全评估报告（SAR），实现了无主动扫描的持续合规管理。

**💡 创新点**

创新点包括：1）将MCP作为Deterministic检索层，消除LLM产生的hallucination，只在实体抽取阶段集中错误；2）提出安全优先的CRH（Criticality‑Risk Heuristic）评分模型，将CVSS、EPSS、KEV与资产关键性、网络曝光度融合；3）实现完整的OSCAL导出与Schema验证，形成从文本到合规文档的闭环。

**🔧 技术方法**

技术栈：多智能体系统、OpenAI GPT‑4.1 LLM、Model Context Protocol、知识图谱（KG）、CRH评分、OSCAL 1.1.2、CTI源（NVD、CISA、ICS‑CERT、Shodan、CVE、CWE、CAPEC、ATT&CK、D3FEND、EOL、EMB3D）以及定制的MCP服务器集群。

**📊 数据集**

使用的数据集为合成的水务设施“WaterWork”参考架构，人工验证了292个CVE、15条攻击路径、16条ATT&CK技术和34条D3FEND对策；此外使用公开的CTI数据库、BSI Grundschutz++ 预览版本和Shodan报告。

**📈 对比分析**

评估方法：与人工构建的真值集对比，计算CVE召回率（0.90）、精确率（0.74）、攻击技术召回率（0.94）、D3FEND召回率（1.00）、实体抽取精准率（87.5%）和召回率（100%），上下文误报率为8.5%。所有OSCAL输出均通过NIST OSCAL JSON Schema 验证，证明审计就绪。

**⚠️ 局限性**

局限性：1）仅在单一合成场景下验证，缺乏跨域多场景评估；2）CRH启发式需在实际工业环境中进一步验证；3）当前仅覆盖软件漏洞攻击，未考虑凭证窃取、社会工程等威胁；4）实体抽取是唯一LLM信任边界，错误会向下传播；5）CTI源的时效性与覆盖度可能影响召回；6）BSI Grundschutz++ 仍处于预览阶段，正式认证尚未完成。

---

## 78. Agentic Neural Architecture Search

**arXiv ID:** 2607.07984 | [PDF](https://arxiv.org/pdf/2607.07984v1)

**作者:** Seokhoon Jeong `[一作]` (Ulsan National Institute of Science and Engineering), Taehwan Kim `[通讯]` (Ulsan National Institute of Science and Engineering)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AgentNAS 三阶段管道，先让大语言模型生成种子网络，再将其拆分为可插拔的槽式架构，随后用传统 NAS 在自动构造的搜索空间中进一步搜索；

**💡 创新点**

关键创新在于将 LLM 与 NAS 的职责分离：LLM 负责宏观结构与搜索空间构造，NAS 负责组合细粒度模块；槽式架构让搜索空间既有限又能表达丰富组合；

**🔧 技术方法**

采用 Claude 系列大语言模型进行 Planner、Code Generator、Data Explorer 与 Slot Planner；利用代码生成器实现网络构建；使用正则化进化、随机搜索和 GDAS 等传统 NAS 算法；实验在 RTX 2080 Ti GPU 上进行；

**📊 数据集**

评估使用 NAS‑Bench‑360（10 个任务）和 Unseen NAS（7 个任务）共 17 个任务，包括分类、密集回归、分割和多标签标记；其中包含 Chess‑board、Darcy Flow、CIFAR‑Tile、FSD50K 等多模态数据集；

**📈 对比分析**

与官方基线、einspace、手工专家设计模型及多种 NAS 方法对比，AgentNAS 在 17 任务中取得 11 例新 SOTA，平均 rank 1.5 远优于 3.6 的对手；LLM 产生的种子已超过大多数基线，NAS 在其基础上进一步提升性能；

**⚠️ 局限性**

局限性包括：硬件与时间预算限制导致搜索仅在小模型空间；搜索预算与停止条件不严格统一；验证指标可能与测试误差不一致；低能力 LLM 可能生成不可用的槽式架构；未对 NAS 超参数进行细致调优。

---

## 79. Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation

**arXiv ID:** 2607.07748 | [PDF](https://arxiv.org/pdf/2607.07748v1)

**作者:** Didula Samaraweera `[一作]` (WSO2), Srinath Perera `[通讯]` (WSO2)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种三阶段流水线，将推理时的计算左移至离线数据合成（编译器反馈循环），随后进行语法感知的监督微调（SFT），最后使用基于可验证奖励的强化学习（GRPO）提升算法正确性。

**💡 创新点**

创新点在于：①将推理时迭代修正完全转化为离线数据生成，解决数据匮乏；②用SFT为低资源语言注入强语法先验，缓解RL稀疏奖励问题；③通过难度（ELO）挑选RL样本并使用零优势掩码，显著提高奖励信号的稳定性与有效性。

**🔧 技术方法**

采用的技术包括：离线编译器+单元测试反馈循环、链式思考提示的SFT、GRPO强化学习、语言无关的IO测试奖励、ELO难度筛选、零优势掩码、4‑bit量化+LoRA微调。

**📊 数据集**

使用的主要数据集：Seed problem集合（Stanford Alpaca 等）+通过编译器验证得到的合成数据；MultiPL‑E 与 Agnostics LiveCodeBench（Julia、Ballerina）；CodeNet、Codeforces 作为RL训练的难度标注样本。

**📈 对比分析**

与基线 Qwen3‑8B、SFT‑only、RL‑only 及其他先前工作（DeepSeek Coder、CodeLlama‑7B、Agnostics）进行对比；在 MultiPL‑E Julia 上 Pass@1 提升 24.6 点，Ag‑LCB Julia 提升 30.2 点；在 Ballerina 上 MultiPL‑E 49.7%、Ag‑LCB 25%；同时显著降低训练成本（1/6 之前最佳方法）和数据量（1/3 之前数据）。

**⚠️ 局限性**

局限性包括：需要目标语言的可用编译器与沙箱；仍依赖离线 API 费用和对 8B 模型的硬件支持；对极低资源语言（无编译器或无足够问题集合）尚未验证；RL 阶段对极难问题仍有限提升。

---

## 80. ADORN: Adaptive Drift handling for Open RAN using Reinforcement Learning

**arXiv ID:** 2607.08443 | [PDF](https://arxiv.org/pdf/2607.08443v1)

**作者:** Ashit Kumar Subudhi `[一作]` (Indian Institute of Technology Dharwad), Koteswararao Kondepu `[通讯]` (Indian Institute of Technology Dharwad)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在O-RAN环境中提出了基于Q学习的ADORN模型，用于自适应处理流量漂移并优化模型重训练。

**💡 创新点**

创新点在于将重训练决策视为MDP，使用RL自动平衡预测精度与计算成本，并结合多专家LSTM集合避免灾难性遗忘。

**🔧 技术方法**

采用Q学习、LSTM集成、统计特征状态表示、O-RAN接口、RL奖励设计等技术实现漂移检测与决策。

**📊 数据集**

使用Colosseum仿真数据集中的八种不同流量场景作为实验数据。

**📈 对比分析**

与贪婪和随机两种基线相比，ADORN在保持nMAE低于漂移阈值的同时，将重训练次数减少约70%，累计奖励显著提升。

**⚠️ 局限性**

局限性包括状态空间离散化导致可扩展性受限，表格式Q学习难以泛化到未见状态，未来需引入深度RL处理连续大规模状态。

---

## 81. Mixture of Enhanced-View Experts for Multi-Query Vehicle ReID and A Large-Scale Benchmark

**arXiv ID:** 2607.08085 | [PDF](https://arxiv.org/pdf/2607.08085v1)

**作者:** Aihua Zheng `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种跨视角自适应融合网络CAFNet，用于多查询车辆重识别。

**💡 创新点**

核心创新包括视角特异性特征增强模块VFEM、动态多视角融合模块DMFM以及多视角对齐损失MAL，实现对不同视角信息的自适应提取与融合。

**🔧 技术方法**

技术实现基于预训练ViT骨干网络，结合混合专家（MoE）机制、交叉注意力、双向对比学习与重建约束等。

**📊 数据集**

使用了自建的大规模多视角车辆ReID基准LCRI-1K（1090个身份、23,637台摄像机、107,805张图像）以及公开数据集MURI进行评估。

**📈 对比分析**

与多种单查询与多查询方法对比，CAFNet在LCRI-1K和MURI上均实现了最高的mAP、Rank‑1、mINP和mCSP，显著提升了多查询识别性能。

**⚠️ 局限性**

局限性在于仍需多张查询图像，且对极端遮挡、低光等场景的鲁棒性需进一步提升；模型结构相对复杂，计算开销较大。

---

## 82. Image classification via a quantum-inspired strategy involving a mixture of experts

**arXiv ID:** 2607.07754 | [PDF](https://arxiv.org/pdf/2607.07754v1)

**作者:** Kumari Jyoti `[一作]` (Indian Institute of Science), Apoorva D. Patel `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种混合经典-量子混合专家模型，用量子启发式卷积和稳健子码实现图像特征提取与分类。

**💡 创新点**

创新点在于将经典扩散卷积替换为局部单元量子卷积、使用 [[5,1,3]] 稳健子码进行池化，并通过联合多专家决策显著提升信息保留与分类准确率。

**🔧 技术方法**

采用幅度编码、局部单元量子卷积、稳健子码、混合专家联合分类，以及 PyTorch+GPU 加速实现。

**📊 数据集**

使用 MNIST 与 Fashion‑MNIST 两个公开手写/服装图像数据集。

**📈 对比分析**

与经典扩散+池化基线对比，联合专家模型在 MNIST 上达 97.6% 测试精度，Fashion‑MNIST 上 86.2%，误差率约减半，计算开销仅略高。

**⚠️ 局限性**

缺乏对真实量子硬件实现的实验验证，模型对更复杂结构的数据适应性及过拟合风险仍需进一步评估。

---

## 83. COBART: Controlled, Optimized, Bidirectional and Auto-Regressive Transformer for Ad Headline Generation

**arXiv ID:** 2607.08071 | [PDF](https://arxiv.org/pdf/2607.08071v1)

**作者:** Yashal Shakti Kanungo `[一作]` (Amazon), Sumit Negi `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在亚马逊广告平台上，构建了一种可控优化的广告标题生成模型 COBART，能够在推理时通过前缀控制令牌同时控制标题的点击率（CTR）和长度，并通过自我批评（SCST）和变分推理进一步提升性能。

**💡 创新点**

创新点在于：①将CTR和长度等可观测特征编码为控制前缀令牌，直接在 BART 的双向编码器中对生成过程进行条件化；②将控制令牌与自我批评训练相结合，形成 SC‑COBART；③提供可扩展的变分 BART（VBART）框架，并展示多种特征控制的可组合性。

**🔧 技术方法**

使用的技术主要包括：预训练的 BART Transformer、前缀控制令牌、self‑critical sequence training（SCST）、变分推理（wake/sleep 机制）、Oracle CTR 预测模型（基于 DeBERTa）、Beam Search 与长度/重复惩罚。

**📊 数据集**

数据集：约 50 万条来自 Amazon 销售商的广告创意，包含产品标题、手写标题及其观测 CTR；经过去重、无重叠划分为训练、验证与测试集。

**📈 对比分析**

实验与基线（BART、T5、ProphetNet、UniLM+SCST、BART 仅高 CTR 过滤、VBART 等）对比，COBART 在 Rouge‑L 上提升 25.82%（相较于 UniLM），在估计 CTR 上提升 5.82%；自我批评版 SC‑COBART 进一步提升到 30% 左右的 Rouge‑L 与 10% 以上的 CTR，表明方法在生成质量与商业价值两方面均有显著提升。

**⚠️ 局限性**

局限性包括：①CTR 需要离线 Oracle 预测，无法实时更新；②控制令牌的桶化需要手动调参，过粗/过细均可能导致信息损失；③模型在生成长度控制时仍受限于输入词汇量和句法约束；④尚未在多语言或跨平台广告环境中验证，可能需要进一步适配。

---

## 84. Scalable and Culturally Specific Stereotype Dataset Construction via Human-LLM Collaboration

**arXiv ID:** 2607.07895 | [PDF](https://arxiv.org/pdf/2607.07895v1)

**作者:** Weicheng Ma `[一作]` (Georgia Institute of Technology), Soroush Vosoughi `[通讯]` (Dartmouth College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种成本高效的人机协同标注框架，构建了覆盖西班牙、墨西哥、阿根廷、哥伦比亚和尼加拉瓜的多国西班牙语刻板印象数据集EspanStereo。

**💡 创新点**

创新点在于利用LLM生成候选刻板印象，再由当地验证者进行文化化验证和实例化，从而大幅降低人工成本并捕获细腻的地区差异。

**🔧 技术方法**

使用对抗式prompt注入攻击、Shapley值注意力头探测与修剪、以及交互式验证问卷等技术手段实现数据采集与偏见分析。

**📊 数据集**

核心数据集为EspanStereo，共538条经验证的刻板印象，附带2,690条（情境-刻板/非刻板/无关）三元组；同时引用StereoSet、CrowS-Pairs等英文基准做对照。

**📈 对比分析**

通过在BETO和XLM-R上进行注意力头修剪实验，证明去除高贡献头后刻板印象评分下降、语言建模和iCAT分数提升，且不同国家的注意力贡献呈显著差异，显示方法能有效评估与缓解跨国偏见。

**⚠️ 局限性**

局限在于LLM在捕捉低频或新兴刻板印象时效果有限，且对极端或非常细粒度文化差异的覆盖仍不完整，未来需结合专家洞察或社交媒体数据进一步提升覆盖度。

---

## 85. Environment-Sensitive Lexicographic Disambiguation for Contextual Parsing

**arXiv ID:** 2607.07728 | [PDF](https://arxiv.org/pdf/2607.07728v1)

**作者:** Alejandro Luis Vaz Mayato `[一作]` `[通讯]`, Alejandro Luis Vaz Mayato

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于全局可变上下文的确定性算法，利用SPPF中的比赛式选择来消除解析树中的歧义。

**💡 创新点**

算法在保持语法为纯CFG的同时，实现了增量、确定性、有限歧义下的语义驱动消歧；提出了严格全序的挑选机制和左到右的上下文变更模型。

**🔧 技术方法**

使用广义解析器生成SPPF（如Earley/GLL/GLR），后续的上下文解算器采用记忆化的可探索构建、递归比较与全序挑选(E)；实现语言为Rust。

**📊 数据集**

在一个自定义的数学DSL上测试，DSL包含24个非终结符、25个记号，构造了大小从5k到2.5M的SPPF。

**📈 对比分析**

通过对不同规模的无歧义与歧义输入进行实验，记录解析时间；实验显示时间近似Θ(n^0.7)，歧义仅增加常数因子，整体保持近线性增长。

**⚠️ 局限性**

局限性包括：必须满足有限歧义假设；仅支持左到右的上下文变更；挑选函数E需满足严格全序，否则可能不稳定；可探索构建在深嵌套时成本高；上下文的可变性可能导致内存不可预知。

---

## 86. Approximation Algorithms for Matroidal Prerequisite Systems

**arXiv ID:** 2607.08151 | [PDF](https://arxiv.org/pdf/2607.08151v1)

**作者:** Robert P. Streit `[一作]` (University of Texas at Austin), Vijay K. Garg `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了“Matroidal Prerequisite System”（MPS）并给出了非负加性与单调子模函数最大化的近似算法。

**💡 创新点**

将部分序与增广基数集合结合，构造了一种新型约束模型，并证明其等价于强多项式贪心子结构，从而在先决条件与替代性同时存在的情形下实现近似优化。

**🔧 技术方法**

利用多项式基子、最大闭包、强多项式贪心子结构的最大表示、连续贪心与随机化稀疏化等技术，得到Δ-和(1+λ)-等一系列近似保证。

**📊 数据集**

本文为理论研究，没有使用具体实验数据集。

**📈 对比分析**

与传统的matroid、poset antimatroid、k-extendible等系统相比，算法在Δ或λ为1时可实现最优，整体近似比为Δ-、(1+λ)-、(2+λ)-或(Δ²·(1−1/e−δ)⁻¹)-，但未给出实验性能对比。

**⚠️ 局限性**

当Δ或λ增大时近似比显著退化，且对子模函数只能得到(2+λ)-或(1+λ)-近似；且证明在Gap‑ETH假设下不可能实现min{Δ,λ}^{o(1)}的近似。

---

## 87. Generalization Theory for Through-the-Wall Radar Human Activity Recognition

**arXiv ID:** 2607.08144 | [PDF](https://arxiv.org/pdf/2607.08144v1)

**作者:** Weicheng Gao `[一作]` `[通讯]` (Beijing Institute of Technology), Weicheng Gao (Beijing Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种基于雷达物理模型的低维度表示（PHY），并给出了该表示在三种结构化域迁移（跨人、跨视角、跨墙）场景下的理论泛化上界。

**💡 创新点**

创新点在于将雷达回波的物理过程（传播损耗、几何投影、微多普勒相位）与统计迁移理论相结合，明确划分人、视角、墙三大物理偏移，推导出以Wasserstein距离为核心的源-目标泛化界限，并证明低维PHY表示能显著收紧该界限。

**🔧 技术方法**

主要技术包括：有界前馈神经网络（MLP）与L2正则化；行向遮蔽投影与表示映射；跨域Wasserstein距离与Rademacher复杂度分析；物理导向特征提取（RT、DT、MD、PHY）及其低维量化。

**📊 数据集**

使用了两类数据集：1）RadHARSimulator V1生成的仿真数据，涵盖12种活动、6人、12视角、3墙面；2）在城市室内场景下搭建的超宽带TWR平台收集的实测数据，同样包含12类活动、6人、12视角和3种墙面。

**📈 对比分析**

通过在源域训练、目标域测试的跨域实验比较PHY与传统图像表示（RTM、DTM、微多普勒特征及其拼接）在跨人、跨视角、跨墙三种迁移任务中的识别准确率进行评估。实验结果显示：PHY在仿真与实测数据中均以最高平均准确率位居首位（仿真≈73%，实测≈70%），而图像拼接位居第二；跨视角迁移最具挑战性，准确率最低。

**⚠️ 局限性**

局限性包括：理论推导基于回波线性近似和远场假设；方法对目标域中极端物理参数变化（如墙层数、材质、信噪比）仍有不确定性；仅采用单一MLP分类器，未探索更复杂网络（如CNN/GRU）可能带来的进一步提升；实验覆盖的目标人群与墙面种类有限，需在更广泛场景中进一步验证。

---

## 88. An exact information theory of generalization phase transitions in Bayesian diffusion models

**arXiv ID:** 2607.08041 | [PDF](https://arxiv.org/pdf/2607.08041v1)

**作者:** Henry Hunt `[一作]` (Stanford University), Surya Ganguli `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的贝叶斯信息限制扩散模型（BIRD），用于理解扩散模型在高维空间中如何避免维度诅咒并学习复杂分布，而不是简单地记忆训练数据。

**💡 创新点**

BIRD模型通过限制每个像素观察到的噪声数据的信息，提供了一种新的视角来分析记忆与泛化之间的相互作用，并在理论上确定了记忆与泛化的相变边界。

**🔧 技术方法**

使用贝叶斯推断和信息论分析，结合空间局部性限制来构建BIRD模型，能够在不同的网络架构（如UNets和DiTs）中进行有效的图像生成。

**📊 数据集**

在多个数据集上进行实验，包括Celeba64、CIFAR10、FashionMNIST和MNIST，以验证理论预测的相变边界。

**📈 对比分析**

通过与现有的扩散模型进行比较，BIRD模型在早期训练阶段能够准确预测图像输出，且在不同数据集上表现出一致的泛化能力，性能指标（如r^2）高达0.9。

**⚠️ 局限性**

理论模型未能捕捉到由于架构选择或学习动态可能引入的其他归纳偏差，限制了其适用性。

---

## 89. EVIS: A Physics-Grounded Event Camera Plugin for NVIDIA Isaac Sim

**arXiv ID:** 2607.08098 | [PDF](https://arxiv.org/pdf/2607.08098v1)

**作者:** Linli Shi `[一作]` (Johns Hopkins University), Ziyun Wang `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于NVIDIA Isaac Sim的事件相机插件EVIS，可在物理模拟中生成高帧率、完全标注的事件流。

**💡 创新点**

将日志对比事件模型与GPU并行渲染、双向运动矢量插值、可选噪声与运动模糊集成到Isaac Sim，实现实时生成并提供真实物理一致性。

**🔧 技术方法**

GPU批量日志对比事件生成、基于渲染的运动矢量双向合成插值、伽玛无校正HDR渲染、噪声模型（阈值不匹配、漏电、射击噪声等）、运动模糊、HDF5输出。

**📊 数据集**

在Isaac Sim自带环境中生成数据，下游验证使用公开预训练模型E2VID、E‑RAFT、Match‑Any‑Events，未使用外部真实数据集。

**📈 对比分析**

通过E2VID重建、E‑RAFT光流、Match‑Any‑Events匹配等三项任务与ground‑truth对比，插值对精度影响渐进（8×插值SSIM≈0.48、EPE≈0.99、匹配精度≈74%），实时性能在30×8配置可达240 Hz事件，1.2倍实时。

**⚠️ 局限性**

对快速运动或遮挡的插值误差仍存在，需提升基准渲染率；模型未包含自适应阈值学习；对多相机同步或不同分辨率支持有限。

---

## 90. Soft Robotic Exogloves for Dexterous Mobility -- Towards Personalized Rehabilitation

**arXiv ID:** 2607.07968 | [PDF](https://arxiv.org/pdf/2607.07968v1)

**作者:** Paul Dela Cruz `[一作]` (Stevens Institute of Technology), Jacqueline Libby `[通讯]` (Stevens Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计、制造并测试了一种基于气动驱动的个性化柔性外骨骼手套，通过对单个受试者手部的3D扫描实现手掌、指关节的精准匹配，并在手背构造可调节的刚性底座；

**💡 创新点**

创新点在于将手部3D扫描数据直接用于定制软体驱动器的几何尺寸与关节位置，采用双关节快速PneuNet结构实现MCP和PIP关节的精确控制；移除纺织应变限制层后进一步提升了驱动器与手指关节的对齐性；通过有限元分析评估人机交互力学，首次为此类软体外骨骼提供可量化的力学验证；

**🔧 技术方法**

技术包括：Artec Eva 3D扫描、硅胶模具铸造、快速PneuNet（fPN）软体驱动、嵌入式Flex角度传感器、PID压力闭环控制、Ansys有限元分析；

**📊 数据集**

数据集仅为单个受试者的手部扫描点云和实验测得的压力、角度等实时数据，没有使用公开的标准数据集；

**📈 对比分析**

通过FEA模拟与现场实验对比，验证了驱动器在150 kPa内的应力低于材料极限；PID控制实现阶跃、正弦、扰动响应分别达到升降时间0.94 s、稳态误差0.027 psi、超调率4.6%；与传统非个性化外骨骼相比，个性化设计显著降低了轴向压缩力并提高了关节对齐率；Flex传感器校准误差约±5°；

**⚠️ 局限性**

局限性包括仅验证了单一受试者，手指简化模型未完全匹配真实生物力学；纺织应变限制层的移除虽然提升对齐，但对驱动器结构稳定性和耐久性尚未评估；柔性传感器嵌入时对伸展的适应性有限；未来需在更大人群中验证，并进一步优化关节力学模型与驱动器结构。

---

## 91. GradInf: Gradient Estimation as Probabilistic Inference

**arXiv ID:** 2607.07840 | [PDF](https://arxiv.org/pdf/2607.07840v1)

**作者:** Gaurav Arya `[一作]` (Carnegie Mellon University), Feras A. Saad `[通讯]` (Carnegie Mellon University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文提出一种将梯度估计问题转化为概率推断问题的“梯度推断”（Gradient Inference）框架，并实现了相应的程序系统 GradInf，能够自动构造无偏、低方差的梯度估计器。

**💡 创新点**

创新点包括：
- 将梯度估计问题与概率推断相互映射，实现了模块化、可组合的梯度估计流程；
- 通过可编程的耦合（coupling）与分解（factorization）技术，将高维、离散、控制流复杂的概率程序转换为适合推断的中间形式；
- 结合信息流类型（information‑flow typing）实现随机选择的部分求值，保证推断算法的正确性与效率；
- 在证明框架下提供了耦合、分解与推断的形式化正确性。

**🔧 技术方法**

技术手段：
- 语义化的耦合变换（coupling transform）与分解耦合变换（factorized coupling transform）；
- 部分概率求值（partial probability evaluation）实现固定随机种子；
- 采用标准概率推断算法（变量消除、层化重要抽样、序贯蒙特卡洛）与自动微分（AD）实现梯度推导；
- 形式化语义基于 quasi‑Borel spaces，证明变换的可证明性；
- Haskell 实现与 Probabilistic Programming Library（lib）集成。

**📊 数据集**

实验数据集：
- M/M/c 排队模型（n 取 25~200，参数 θ=15、20）；
- 三叉树选项定价模型（r=15%，σ=5%，S₀=40，K=41，T=1.0 年）；
- 基因转录反应网络模型（α=18、β=8、γ=1.5、δ=4，仿真时间 2.5）。
- 所有实验均使用合成模拟数据，未采用公开现实世界数据集。

**📈 对比分析**

比较方法与性能：
- 与传统 REINFORCE、SP‑phantom、Girsanov 等基线进行比较；
- 通过方差与“方差×时间”两指标评估；
- 结果显示：在排队模型中，新梯度估计器相对 SPA 约 110 倍方差降低；
- 在期权定价模型中，使用 twisted SMC 的新估计器在 ρ 指标上相对基线下降约 11 倍；
- 在基因转录模型中，twisted SMC 与层化重要抽样的估计器分别实现 20–370 倍方差降低；
- 运行时间开销：相较于 Storchastic 在排队模型下 3 倍以内、期权模型 6 倍以内，整体仍保持可接受的效率。

**⚠️ 局限性**

局限性：
- 目前仅支持有限循环、离散与连续分布的概率程序，无法处理无限递归或无界数据结构；
- 对参数化连续随机变量的分段不连续性缺乏合适的耦合与推断策略；
- 需人工为每个原语指定耦合/分解方案，缺乏自动化选择；
- 未实现 GPU 加速与批处理；
- 仅支持一阶梯度，尚未扩展到高阶梯度；
- 对基于学习的控制变元（如强化学习中的 actor‑critic）不支持。

---

## 92. A Top-Down Deriving Mechanism in Haskell

**arXiv ID:** 2607.07732 | [PDF](https://arxiv.org/pdf/2607.07732v1)

**作者:** Song Zhang `[一作]` `[通讯]`, Song Zhang

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于模板 Haskell 的自顶向下派生机制，能够自动为复合数据类型及多层类层次生成实例，从而消除手写 deriving 语句的需求。

**💡 创新点**

创新点在于将派生扩展到多个相关实例、支持类层次自动生成上下文、提供递归处理与停止条件，并在 API 设计中加入可自定义上下文生成策略。

**🔧 技术方法**

采用技术包括 Template Haskell 代码生成、泛型编程与 DefaultSignatures、类型家族推导、派生策略（stock、newtype、anyclass）以及自定义上下文推理算法。

**📊 数据集**

在 Haskell 官方 AST（haskell-src-exts）和标准库（如 Binary、Generic）等数据集上进行实验，覆盖约 30 个模块的 AST 类型。

**📈 对比分析**

相较于手写派生，实验显示代码量显著减少，编译器错误信息更易读，生成实例速度与手工实现相当甚至更快，但实验未给出精确的性能对比指标。

**⚠️ 局限性**

局限性包括仅支持 ★→Constraint 和 (★→★)→Constraint 形的类，无法处理多参数类、GADTs/存在类型；生成过程需手动指定停止类型或使用简化的上下文推导；递归类型处理仍需关注替换和循环检测。

---

## 93. Optimal Sparsifiers for Abelian Cayley Graphs

**arXiv ID:** 2607.08261 | [PDF](https://arxiv.org/pdf/2607.08261v1)

**作者:** Arpon Basu `[一作]` (Princeton University), Stefan Tudose `[通讯]` (Princeton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了对任意有限阿贝尔群上Cayley图的最优谱稀疏化方法，构造了仅含O(log |G|)个生成元的稀疏图，保持谱特性；

**💡 创新点**

证明了O(log |G|)的生成元上界是最优的，并将该结果推广到二进制线性码的稀疏化，得到O(n/ε²)-大小的稀疏码；

**🔧 技术方法**

采用了基于卷积/字符对称性的体积上界技术，并将稀疏化过程视为多次“部分着色”迭代，利用对称凸体的体积估计和随机化的分离算子；

**📊 数据集**

论文未使用任何外部数据集，全部为理论构造与证明；

**📈 对比分析**

相较于之前的稀疏化方法（如Khanna‑Putterman‑Sudan），去掉了额外的log n因子，获得了最优的O(n/ε²)大小；

**⚠️ 局限性**

主要局限是构造的稀疏化算法时间复杂度为O(|G|)（指数级），尚缺乏低复杂度（多项式时间）的确定性实现。

---

## 94. Explaining Near-Zero Hessian Eigenvalues Through Approximate Symmetries in Neural Networks

**arXiv ID:** 2607.07845 | [PDF](https://arxiv.org/pdf/2607.07845v1)

**作者:** Marcel Kühn `[一作]` (Universität Leipzig), Bernd Rosenow `[通讯]` (Universität Leipzig)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过理论推导与数值实验，研究了深度网络训练损失Hessian谱的局部几何结构，证明了大量接近零的特征值来源于网络参数化的连续对称性被弱破坏后产生的伪Goldstone模；作者在线性网络中构造了完整的零模基，随后将Leaky‑ReLU视为微扰，揭示了特征值随非线性强度的ε²标度，并在学生-教师模型、CIFAR‑10训练的全连接网络以及一维卷积网络中验证了这一机制。

**💡 创新点**

创新点在于首次将近零Hessian特征值与网络结构的弱破坏对称性联系起来，给出显式的对称生成器和对应的零模；提出ε²提升规律并通过特征向量重叠证明了非线性网络中的特征向量基本停留在对称子空间；进一步把这一理论推广到卷积网络，并在真实数据训练的网络中验证了两层特征结构。

**🔧 技术方法**

使用了Hessian与Gauss‑Newton分解、Fisher信息矩阵、奇异值分解、微扰理论、特征向量重叠度量、学生‑教师模型、全连接网络训练与卷积网络仿真等技术手段；同时在数值上计算了大规模Hessian的前几千特征向量并与对称子空间做对齐分析。

**📊 数据集**

主要使用了三类数据集：学生‑教师实验中的高斯分布输入；训练CIFAR‑10的标准图像数据集；以及一维卷积网络的人工生成输入。

**📈 对比分析**

通过计算Hessian特征向量与对称子空间投影的重叠度量，发现高曲率方向重叠度低、接近零模的方向重叠度高，形成两层谱结构；在CIFAR‑10实验中，训练的全连接网络达到约53% 的测试准确率，且其Hessian谱与线性比较模型的对称子空间对齐结果与理论预期一致。

**⚠️ 局限性**

局限性包括：研究聚焦于全连接与简单卷积结构，复杂网络（如Transformer）中的对称性和伪Goldstone模尚未完全验证；对对称性破坏程度的微扰假设在高度非线性或大尺度模型中可能失效；计算上仅能得到前几千特征向量，无法完整覆盖高维Hessian；此外，低方差输入方向导致的平坦性可能与对称性机制交叉影响，需进一步分离。

---

## 95. Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents

**arXiv ID:** 2607.08448 | [PDF](https://arxiv.org/pdf/2607.08448v1)

**作者:** Yixian Zhang `[一作]` (Tsinghua University), Chao Yu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Harness VLA的框架，将冻结的Vision‑Language‑Action模型作为可重用的接触阶段原语，并通过固定的分析原语和记忆驱动的规划器实现跨环境、跨语言的语言条件操控。

**💡 创新点**

创新点在于将VLA拆解为局部接触专用原语，并让高层LLM规划器承担语言理解、目标定位、导航等非接触结构；同时引入任务特定记忆与全局记忆，以复用成功轨迹和失败模型。

**🔧 技术方法**

使用LLM驱动的agentic planner（如Codex/Claude），冻结的VLA（RLinf、RLDX-1、LingBot‑VLA），固定的分析原语（move_to, rotate_wrist, etc.），以及JSON序列化的原语接口和两层记忆机制。

**📊 数据集**

在LIBERO、LIBERO‑Pro、RoboCasa365、RoboTwin C2R四个基准上评估，涵盖桌面、厨房、双手等场景，使用对应的标准任务和扰动版本。

**📈 对比分析**

与直接使用冻结VLA以及其它基线（OpenVLA、NORA、Cap‑X、RATS、RLDX-1等）进行比较，展示在扰动任务中提高了约38.6个百分点（LIBERO‑Pro）和25.4个百分点（RoboCasa365），以及在清洁到随机化迁移任务中提升至58.4%。

**⚠️ 局限性**

受限于开环的VLA回调和缺乏环境奖励的联合微调；缺少细粒度图像描述，导致在高拥挤长周期任务中结构推理受限。

---

## 96. Degree-Constrained Interval Optimization for Minimax Polynomial Approximation in Homomorphic Encryption

**arXiv ID:** 2607.08042 | [PDF](https://arxiv.org/pdf/2607.08042v1)

**作者:** Jiheon Woo `[一作]` (Pohang University of Science and Technology), Yongjune Kim `[通讯]` (Pohang University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在同态加密下，对非多项式激活函数进行最小化误差多项式逼近时，提出把逼近区间视为可调参数，基于输入分布优化逼近区间以最小化均方误差。

**💡 创新点**

创新点在于：①将逼近区间作为分布感知的优化变量，得到一个完整的均方误差目标；②构造可解析的域扩展函数（DEF）代理目标并给出闭式解；③将理论目标与可同态实现的域扩展多项式（DEP）通过实现误差分解关联，并给出误差上界；④证明代理目标在实际激活函数上能准确预测最优区间。

**🔧 技术方法**

主要技术包括：Remez 递推求最小化误差多项式；域扩展函数与多项式（DEF/DEP）设计；均方误差分解与凸优化；Arcsine 残差分布假设；对高斯、拉普拉斯预激活分布的闭式计算；多阶段 DEP 组合与误差上界分析。

**📊 数据集**

实验使用合成的高斯（σ=1）和拉普拉斯（b=1/√2）预激活分布，并参考 CIFAR‑10 测试集的激活计数（≈1.88×10^9）来设定覆盖输入区间 R₀。

**📈 对比分析**

与固定区间 [-R₀,R₀] 的 Remez 基线、理想 DEF 目标和实际 DEP 目标进行对比；结果显示：代理目标与理想 DEF 的最优区间高度一致；DEP 方案在均方误差上比固定区间基线低数阶（尤其是 sigmoid、tanh、GELU）；两种扩展因子 L₁=1.6 与 L₂=2.5 的比较揭示更大的扩展因子可在多阶段合成中减少误差。

**⚠️ 局限性**

局限性包括：①需要预先知道或估计每层的预激活分布；②当前仅针对单个非多项式算子，未考虑多层累积误差与实际 HE 代价；③DEP 级联导致多阶段乘法与乘法深度增加；④在极端长尾分布或非高斯分布时，代理目标可能失效；⑤未与具体同态库（如 SEAL/HElib）进行完整加密评估。

---

## 97. Validating LLMs in social science: Epistemic threats and emerging norms

**arXiv ID:** 2607.07915 | [PDF](https://arxiv.org/pdf/2607.07915v1)

**作者:** Meera Desai `[一作]` (University of Michigan), Abigail Z. Jacobs `[通讯]` (University of Michigan)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统收集并分析了2022-2025年间八大顶尖社会科学期刊中使用大语言模型（LLMs）生成定量测量的27篇论文（共50个测量任务），评估其概念化、操作化与验证实践。

**💡 创新点**

首次以大语言模型为测量工具构建全面的研究语料库，并将测量理论框架应用于多维度验证，揭示当前验证侧重收敛性、缺乏多维度证据、概念阐释不足等问题。

**🔧 技术方法**

采用文献检索、定性编码与测量理论分析，利用收敛性、预测性、面效度等多种验证角度对LLM生成的测量进行评估。

**📊 数据集**

主要使用从2143篇论文中筛选出的27篇包含LLM测量任务的研究作为数据源，涉及约50个具体测量任务。

**📈 对比分析**

通过将LLM生成的测量与人工标注或已有工具产生的黄金标准进行比较（如百分比一致率、皮尔逊相关、Cramér’s V等），结果表明验证方法多样但普遍缺乏系统性，且大部分仅检验收敛性。

**⚠️ 局限性**

局限在于验证实践单一、报告透明度低、概念阐释不充分、缺乏跨任务一致的标准化评估流程，导致对LLM测量可信度的评估不完整。

---

## 98. Input-Constrained Spatiotemporal Tubes for Safe Navigation of Unknown Euler-Lagrange Systems in Dynamic Environments

**arXiv ID:** 2607.08189 | [PDF](https://arxiv.org/pdf/2607.08189v1)

**作者:** Siddhartha Upadhyay `[一作]` (Indian Institute of Science), Pushpak Jagtap `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对未知欧拉-拉格朗日系统且具有输入约束的实时空间时序管（STT）控制框架，能够保证有限时到达-避障-停留（FT-RAS）任务的形式化安全保证；

**💡 创新点**

关键创新包括：1）在STT设计中显式纳入输入约束，推导离线可验证的可行性条件；2）利用无逼近、闭式控制律，确保在未知动力学下的实时实现；3）通过可调的变换函数与速度/加速度限制保证输入界限；

**🔧 技术方法**

采用空间时序管设计、有限时控制理论、变换函数（bounded transformation）、速度追踪的漏斗约束、以及基于不确定性界限的可行性分析；

**📊 数据集**

实验验证使用了三类平台：二维差动驱动移动机器人（使用AgileX LiMO机器人数据与随机障碍生成）、三维四旋翼仿真（随机障碍与扰动），以及三维航天器姿态再定位（自定义动态约束）；

**📈 对比分析**

与现有实时STT、ILQR、NMPC、CBF等方法进行对比，表明本方法在满足输入约束的同时，保持了低计算负荷，成功实现任务完成；在比较实验中，本方法在安全约束下完成任务所需时间略长，但始终不超过输入上限，显示出稳健性；

**⚠️ 局限性**

局限性包括：1）仍需要对动态障碍进行在线观测与预测；2）对极大不确定性或高维系统的可扩展性尚未充分验证；3）目前仅处理欧拉-拉格朗日结构，需进一步推广到更一般非线性/离散/随机系统；

---

## 99. Self-Stabilizing Algorithms in the Uniform Port Model

**arXiv ID:** 2607.08244 | [PDF](https://arxiv.org/pdf/2607.08244v1)

**作者:** Liam Brinker `[一作]` (Technion - Israel Institute of Technology), Oren Louidor `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了统一端口（Uniform Port）模型，并在该模型下设计了自稳健（self‑stabilizing）的高效算法，解决了最大独立集、最大匹配、无源定向、节点/边最大 k‑着色等经典局部对称破坏问题。

**💡 创新点**

创新点在于：①将计算实体从节点转移到图的端口，保持算法的真正统一性；②支持对端点（half‑edge）进行标签赋值，扩展了以往仅能处理节点标签的石器时代模型；③首次在真正统一模型中实现了多种自稳健算法，且在一般图上达到多项式对数时间复杂度。

**🔧 技术方法**

主要技术包括：端口级有限状态机、局部领导者选举（LLE）机制、几何博弈（geometric tournament）与多阶段同步机制、令牌传递（token‑passing）分析以及对称破坏问题的组合与递归构造。

**📊 数据集**

本文为理论分析，没有使用具体实验数据集，所有结果均为严谨的概率上界与复杂度证明。

**📈 对比分析**

与先前工作相比，本文的算法在一般图上实现了 O(log² n)（MIS、SO、节点 k‑着色）或 O(log⁵ n)（MM、边 k‑着色）的自稳健运行时间，远优于之前仅在特殊图或随机图上获得的结果；实验上也通过概率大于 1‑n⁻ᶜ 的高概率保证来验证效率。

**⚠️ 局限性**

局限性包括：①算法依赖同步时钟；②端口级状态机虽然是常数大小，但在实际硬件实现时可能需要额外的通信开销；③对极端稠密图或非常特殊结构的图，理论上可能仍存在较高的常数因子；④模型未考虑异步或消息丢失的实际网络环境。

---

## 100. Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech

**arXiv ID:** 2607.08208 | [PDF](https://arxiv.org/pdf/2607.08208v1)

**作者:** Hao Wu `[一作]` (Shanghai Qi Zhi Institute), Wei Xu `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本论文提出了一套完整的多语种两说话者对话语音识别系统，结合模块化说话人分离前端和经过多阶段微调的Qwen3-ASR-1.7B模型，能够在MLC‑SLM 2026 Task 1挑战中同时完成说话人划分和多语种转写。

**💡 创新点**

创新点包括：①将说话人分离与ASR通过RTTM驱动的音频切分紧密耦合；②使用OmniVoice多语种零拷贝声学克隆进行大规模合成语音增强，产生超过23万条训练样本；③在合成数据上应用LoRA微调，再结合GRPO强化学习进行输出稳定性和错误惩罚的调优。

**🔧 技术方法**

核心技术包括：3D‑Speaker框架下的VAD、CAMPPlus说话人嵌入与两说话者谱聚类；Qwen3‑ASR‑1.7B模型的全量监督微调、LoRA低秩适配与GRPO无评论者强化学习；以及基于TTS的合成语音数据增强。

**📊 数据集**

使用的数据集为MLC‑SLM 2026 Task 1官方训练集（≈1,500 h，21种语言/方言），以及通过OmniVoice生成的合成语音（共234,333条），并保留开发集和评估集用于验证与排行榜比较。

**📈 对比分析**

与官方基线、Whisper‑large‑v3及Omniasr‑LLM‑7B‑v2等模型对比，开发集平均tcpMER从30.53（未微调的Qwen-ASR‑1.7B）降至23.70，最终评估集平均tcpMER达到17.97，显著优于基线和其他公开模型。

**⚠️ 局限性**

局限性主要在于：①对相似音色说话人的分离准确度仍受限；②合成语音在低资源语言中的质量和多样性不足；③在长时段噪声或多说话人重叠的情况下仍可能出现识别不稳定和少量重复/幻觉错误。

---

## 101. RetractorDB: A Deterministic Edge Signal Processing Engine Based on Rational Beatty Sequences and Fraenkel's Partition

**arXiv ID:** 2607.07730 | [PDF](https://arxiv.org/pdf/2607.07730v1)

**作者:** Michal Widera `[一作]` `[通讯]` (Independent Researcher), Michal Widera (Independent Researcher)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出并实现了RetractorDB——一种基于数论覆盖系统的开源边缘信号处理引擎，能够在边缘设备上对正则时间序列进行精确、确定性的预处理、过滤和重采样，并以可审计的二进制artifact形式输出，供上游时序数据库或流管理系统进一步存储与分析；

**💡 创新点**

创新点在于将Fraenkel定理下的有理Beatty序列与流算子相结合，构建了一个能实现精确率变换（interleave/​de‑interleave）的声明式流代数；通过该代数在RQL语言中的实现，实现了在不离散化、无浮点误差的情况下执行多速率DSP；此外，artifact模型将未来槽预先生成、支持可非破坏性更正，并与定时器精确调度相配合，保证了系统输出可重复、可审计。

**🔧 技术方法**

技术包括：C++实现的三二进制架构；RQL声明式查询语言；依赖图编译与有理间隔解析；基于Beatty序列与Fraenkel定理的interleave/​de‑interleave算子；窗口、求和、差分等传统算子；artifact格式（schema + null/gap索引）与slot‑based运行时调度；以及完整的可执行示例验证与自动化测试。

**📊 数据集**

主要使用MIT‑BIH Arrhythmia Database（记录205）来验证Pan‑Tompkins QRS检测流水线；其他内部测试流用于演示算子合并、窗口、可重复读取等功能。

**📈 对比分析**

目前论文仅提供语义验证示例（确定性输出），未进行吞吐量/延迟等性能评估；性能测试计划在后续版本中展开，且未与现有流引擎进行对比。

**⚠️ 局限性**

局限性包括：仅支持常数有理间隔的正则流（对非正则流需先映射到更细网格并填充空槽）；缺乏实时硬实时保证；未对大规模并发或高吞吐场景进行评估；以及不提供完整的时序数据库功能，仅担任边缘预处理和artifact传输层。

---

## 102. A 'normal' research trap limits scientific breakthroughs and disruptive innovation in the European Union

**arXiv ID:** 2607.08328 | [PDF](https://arxiv.org/pdf/2607.08328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 103. TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning

**arXiv ID:** 2607.08283 | [PDF](https://arxiv.org/pdf/2607.08283v1)

**作者:** Yushen Liang `[一作]` (NYU Shanghai), Shenji Wan `[通讯]` (NYU Shanghai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Temporally Conditioned Memory-Fusion Policies（TFP），为视觉-语言-动作（VLA）策略加入连续时间的任务进度记忆，并通过AdaLN直接将记忆注入动作解码器；

**💡 创新点**

创新点包括：①使用Liquid Time-Constant（LTC）网络实现基于物理时间的记忆更新，使记忆在稳定期保留、事件期快速重写；②将记忆直接作为动作生成的条件，而非仅作为检索上下文；③提出Episode‑Aware Temporal Batching（EATB）以高效训练多段连续记忆；④通过写入增益分析证明记忆对交互事件具有敏感性；

**🔧 技术方法**

技术手段包括：LTC 动态记忆、AdaLN 自适应归一化、流匹配动作解码器、EATB 训练框架、适应性递归窗口执行；

**📊 数据集**

实验数据集覆盖：LIBERO、LIBERO-plus、MIKASA‑Robo ShellGameTouch、Galaxea A1 真实机器人任务；

**📈 对比分析**

与 π_0.5、OpenVLA、HAMLET、AVA‑VLA 等基线在相同 backbone（π_0.5）下对比；在 LIBERO 上平均成功率从 96.9% 提升至 98.75%，在 LIBERO‑plus 从 91.4% 提升至 93.77%，ShellGameTouch 成功率提升至 75%，A1 真实机器人阶段相关错误显著下降；

**⚠️ 局限性**

局限性包括：训练成本高（需长时间 GPU 训练、梯度截断）；仅在单臂桌面任务上验证，未覆盖移动机器人或双手；记忆仍缺乏强对象空间绑定，仍需改进。

---

## 104. TTHE: Test-Time Harness Evolution

**arXiv ID:** 2607.08124 | [PDF](https://arxiv.org/pdf/2607.08124v1)

**作者:** Jun Nie `[一作]` (Hong Kong Baptist University), Bo Han `[通讯]` (Hong Kong Baptist University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在评估阶段利用无标签的执行轨迹，对LLM代理的可执行控制程序（harness）进行持续演化，从而在不更新模型权重、也不使用任何标注的前提下改进代理行为。

**💡 创新点**

把执行程序本身视为可适配对象，提出Test‑Time Harness Evolution（TTHE）框架：在每个测试批次内维护候选程序族，通过代理提议者基于执行轨迹生成新程序，并由评判器在无标签信号下挑选最优程序；这一过程在评估时完成，形成可解释、可检视的程序改进。

**🔧 技术方法**

核心技术包括：①基于冻结LLM的代码生成代理（proposer）和评判器；②基于执行轨迹的轻量代理信号（执行健康、回溯一致性、公共测试通过率）；③种群搜索与迭代改进（G支路、R轮）；④对程序的运行与调试工具集成。

**📊 数据集**

实验使用了四大类任务的“难题”切片：Text‑to‑SQL（BIRD）、Competitive Programming（LiveCodeBench）、软件工程（SWE‑bench Verified）、数据科学编程（DS‑1000），以及工具使用基准（claw‑eval）。

**📈 对比分析**

方法对比：以标准ReAct scaffold为基线；通过跨模型（DeepSeek、MiMo、Kimi）和不同搜索预算（G、R、B）进行 ablation。实验结果表明：BIRD从12.0%提升至50.0%，LiveCodeBench从30.0%提升至38.3%，SWE‑bench Verified从20.0%提升至35.0%，DS‑1000从38.0%提升至44.0%，claw‑eval从48.9%提升至69.8%。

**⚠️ 局限性**

局限性：①评估为转导式，未验证跨批次或前瞻泛化；②选择阶段受评判器误差影响，存在“选择遗憾”；③搜索覆盖受限，仍有任务未被候选程序覆盖；④仅在沙箱环境下验证，缺乏开放世界安全保障；⑤对执行轨迹的代理信号不完备，导致潜在的误导。

---

## 105. Toward a Unified GPU-Aware OpenSHMEM Specification

**arXiv ID:** 2607.08006 | [PDF](https://arxiv.org/pdf/2607.08006v1)

**作者:** Naveen Ravi `[一作]` (Hewlett Packard Enterprise), Steve Poole `[通讯]` (Los Alamos National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个兼容OpenSHMEM 1.x的GPU‑aware辅助规范，支持GPU内存空间和设备发起通信。

**💡 创新点**

通过定义统一的GPU内存模型、上下文可见性和能力查询，为多厂商实现提供可移植性。

**🔧 技术方法**

采用PGAS模型扩展、GPU加速的RMA、原子、同步和集合操作，并使用能力查询接口。

**📊 数据集**

未使用传统数据集，而是基于现有NVSHMEM、rocSHMEM和Intel SHMEM的实现来验证兼容性。

**📈 对比分析**

通过对现有实现的功能对比和测试套件来评估规范一致性，未给出具体性能指标。

**⚠️ 局限性**

局限在于仅覆盖基础Tier‑0功能，未覆盖流触发、设备进程持续进度和更细粒度上下文管理，且需进一步实现与验证。

---

## 106. RadioDiff-v2: Generative Angular Radio Maps for Multi-Beam Selection and Localization

**arXiv ID:** 2607.08045 | [PDF](https://arxiv.org/pdf/2607.08045v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种基于流匹配的一维扩散Transformer（RadioDiff-v2）模型，可从基站位置、接收机几何及建筑占用信息预测角度功率谱（APS），并将预测结果用于多波束选择和接收机定位。

**💡 创新点**

① 将角度射频地图预测建模为感知-失真问题；② 采用流匹配训练目标，构造确定性概率流ODE；③ 在Transformer中引入周期性角度编码、适应性层归一化、傅里叶混合器和双头解码器；④ 通过“多指标估计组合”一次模型即可提供分布式样本、点估计和贝叶斯似然，支持多波束采样、点精度推断与多基站定位。

**🔧 技术方法**

流匹配（rectified flow）、一维扩散Transformer（Dual‑Branch DiT‑1D）、周期性角度位置编码、适应性层归一化（adaLN-zero）、傅里叶角度混合器（AFT）、双头解码（速度+清晰信号）、分类器无关指导、Wasserstein‑1 用于调参、以及 ODE 采样。

**📊 数据集**

使用 99 个城市环境（约 500 万条链路）构建的角度射频地图数据集，环境为 256×256 像素建筑占用图，每条链路包含 180° 离散角度谱，采用 79/20 的零射手训练/测试划分。

**📈 对比分析**

与 RME‑GAN、MS‑Areg、RadioDiff（先前的扩散模型）和 COST231 基线对比；在零射手测试中，Wasserstein‑1 下降至 0.39 dB（≈5×提升），NMSE 0.184，8 波束扫频损失 2.43 dB；单波束误差仅 0.02 dB；多基站定位中位误差 20.6 像素（4 基站）比单站 62.6 像素下降约 60%，比基线大幅提升。

**⚠️ 局限性**

仅预测水平角度谱，未覆盖频宽与仰角；对建筑占用图与几何信息依赖较大，若信息不完整会影响性能；多基站定位仍需多站覆盖，单站定位性能有限；模型训练规模较大，部署成本相对较高。

---

## 107. Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning

**arXiv ID:** 2607.08393 | [PDF](https://arxiv.org/pdf/2607.08393v1)

**作者:** Lu Dai `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并量化了大语言模型在细调后出现的“Knowing–Using Gap”，通过自补丁（self‑patching）技术定位知识在模型内部的空间渗透轨迹，并提出知识‑电路错位假设；随后设计并验证了一种基于固定层对的启发式补丁策略，显著提升了多跳推理性能。

**💡 创新点**

创新点在于：①提出自补丁方法对细调后知识的空间渗透进行因果诊断；②揭示知识被编码在易于记忆的早/晚层，但未能迁移至推理所需的中层，从而产生性能鸿沟；③基于诊断结果构造可复现、无需实例搜索的两层对补丁启发式，恢复了58–75%的最佳性能。

**🔧 技术方法**

主要技术包括：自补丁（self‑patching）—对齐内部表示与推理层；激活补丁与因果追踪；多任务评测框架；以及基于固定层对的启发式补丁策略。

**📊 数据集**

使用了从真实知识库 STaRK（其医学子集 STaRK‑Prime 与学术子集 STaRK‑MAG）自动生成的记忆与推理 QA 数据，保证训练与测试知识不重叠。

**📈 对比分析**

对比方法包括 FFT（全参数微调）与 LoRA 细调，以及多模型、多规模、多领域实验；结果显示：记忆准确率迅速升至接近 100%，但推理准确率存在显著的时间滞后与准确率缺口；自补丁可恢复 58–75% 的最优性能，启发式补丁实现了与 oracle 相当的 60–75% 头房，明显优于传统 CoT、无补丁等基线。

**⚠️ 局限性**

限制：①补丁位置仍需人工或基于实验确定，无法完全自适应；②对不同实例的存储位置差异导致恢复不一致；③仅针对单事实更新与多跳推理，未验证对更大规模知识或更复杂推理的通用性；④在极大模型或低资源设置下的可扩展性仍待评估。

---

## 108. Playing ZendoWorld: Challenging AI Agents on Active Visual Concept Induction

**arXiv ID:** 2607.08233 | [PDF](https://arxiv.org/pdf/2607.08233v1)

**作者:** Sophia Koehler `[一作]` (TU Darmstadt), Kristian Kersting `[通讯]` (TU Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ZendoWorld，一个可交互、基于视觉的规则推断与实验设计的闭环评测环境，并在其上评估多类智能体。

**💡 创新点**

创新点在于将视觉感知、逻辑归纳和主动实验三者集成于同一交互循环，并提供精确DSL规则空间与反馈。

**🔧 技术方法**

采用多种技术：VLM端到端推理、Bayesian粒子滤波、神经符号程序合成（VLP）、以及基于DSL的Oracle搜索。

**📊 数据集**

使用自制的ZendoWorld数据集，包含22个不同规则的游戏场景，图像由Blender+Prolog生成。

**📈 对比分析**

通过对比人类与四种代理，评估胜率、回合数、标签准确率与信息增益，发现Oracle最高（95.5%），VLM与VLP低于人类，且EIG普遍偏低。

**⚠️ 局限性**

局限性包括：对DSL的依赖导致无法处理OOD规则；视觉感知与推理耦合度低，VLM在实验设计上信息量不足；整体模型仍未能达到人类水平。

---

## 109. Contrastive Order Learning: A General Framework for Ordinal Regression

**arXiv ID:** 2607.08109 | [PDF](https://arxiv.org/pdf/2607.08109v1)

**作者:** Chaewon Lee `[一作]` (Korea University), Chang-Su Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Contrastive Order Learning（ConOrd）的通用框架，用于解决序数回归问题

**💡 创新点**

创新点在于将对比学习与序数学习结合，利用基于秩差的软亲和与离散权重，使所有样本对在一个批次内以软方式对齐与分离，从而实现细粒度且全局一致的序数结构建模

**🔧 技术方法**

主要技术包括：对比学习框架（Softmax对齐/分离），软亲和权重a_{ij}=1/(|r_i-r_j|^2+ε)，离散权重b_{ij}=|r_i-r_j|^2，负欧氏距离相似度，中心损失对齐参考点，k‑NN推理

**📊 数据集**

在三个序数回归任务上评估：人脸年龄估计（MORPH, CLAP2015, AgeDB-DIR, UTK, CACD, Adience），盲图像质量评估（BID, CLIVE, KonIQ‑10k, SPAQ, FLIVE），盲视频质量评估（LSVQ, KoNViD‑1k, LIVE‑VQC, CVD2014, YouTube‑UGC）

**📈 对比分析**

与多种现有方法（SupCon, RnC, GOL, OrdinalCLIP, NumCLIP, QPT, LoDa, QCN, VISGA, UNQA, RichIQA, ModularBVQA等）比较，在所有数据集上均取得更高的SRCC/PCC或更低的MAE，显示出显著性能提升

**⚠️ 局限性**

局限性包括：对秩差的二次形式权重需要手工设定，无法自适应不同任务；对极少量样本或标签不平衡场景下仍可能受限；在需要实时推理的部署中，k‑NN推理仍有计算开销

---

## 110. KS-CFA: Control-Flow Attestation via Symbolic Replay Against Control-Flow Bending Attacks

**arXiv ID:** 2607.07926 | [PDF](https://arxiv.org/pdf/2607.07926v1)

**作者:** Zhanyu Sha `[一作]` (Royal Holloway University of London), Amir Rafi `[通讯]` (Royal Holloway University of London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于单路径符号回放的控制流证明方案 KS-CFA，用于检测控制流弯曲攻击。

**💡 创新点**

创新点在于只记录影响控制流的外部输入，并通过预先构建的变量解析表进行单路径符号回放，无需枚举合法路径或额外硬件。

**🔧 技术方法**

使用 LLVM IR 静态分析、变量解析表（VRT）、可信执行环境（Keystone TEE）、批量日志记录以及符号执行推理等技术。

**📊 数据集**

使用 Embench‑IoT 基准套件以及自定义的 S‑MACH 等程序进行评测。

**📈 对比分析**

相较于传统哈希或硬件方案，KS-CFA 在 FPGA 上测得 6.7×–32.2×、QEMU 上 6.8×–20.5× 的执行开销，同时在检测覆盖率上可达到 100%（拥有所有输入时），并避免了路径枚举的成本。

**⚠️ 局限性**

主要限制包括需要手动标注输入相关变量、禁用后端优化以保持 IR 与机器指令的一一对应，以及日志体积相对较大。

---

## 111. HeadRoom: Lightweight, Edge-deployable Pipeline for Adaptive Notification Routing

**arXiv ID:** 2607.08083 | [PDF](https://arxiv.org/pdf/2607.08083v1)

**作者:** Dinithi Dissanayake `[一作]` (National University of Singapore), Suranga Nanayakkara `[通讯]` (National University of Singapore)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HeadRoom，一种轻量级、可在边缘设备上实时估计视觉与听觉通道可用性的管线，并在可穿戴设备上实现自适应通知路由。

**💡 创新点**

创新点在于将预测误差作为感知通道负荷的即时指标，能够在无需外部生理信号的情况下，在资源受限设备上自适应地将通知投射到更可用的感官通道。

**🔧 技术方法**

技术包括：使用 MobileNetV3-Small 作为视觉特征提取器，训练两层 MLP 预测下一帧嵌入；使用自监督的 MFCC、能量、谱波动等 31 维特征预测下一帧音频特征；实时归一化预测误差并与阈值比较实现路由决策；在 Unity+ONNX 生态中将模型部署到 Meta Quest 3S。

**📊 数据集**

主要数据集为 Project Aria 的日常活动视频（约 138 条训练视频、10% 验证、6 条留作用户实验），同时使用 3 条独立场景视频进行心理物理实验。

**📈 对比分析**

在 22 名参与者的控制实验中，对比了 Model、Inverse、Random 路由条件。结果显示，在高感知负荷场景（Video 3）中，Model 条件的平均响应时间比 Inverse 快约 114 ms（p≈0.021），效应量 d≈1.31；Random 与 Inverse 之间的差异趋近显著。系统在 Meta Quest 3S 上的平均推理时延约 11 ms/步，内存占用 10 MB，模型体积 0.625 MB，能够满足 10 fps 的实时需求。

**⚠️ 局限性**

局限性包括：预测误差仅为可用性的代理，未考虑内部因素（疲劳、期望等）；实验采用简化的探测任务，未检验更复杂通知的效果；仅评估了视觉和听觉通道，未包含触觉；在不同环境下通道相互依赖的建模尚不完善。

---

## 112. Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies

**arXiv ID:** 2607.07957 | [PDF](https://arxiv.org/pdf/2607.07957v1)

**作者:** Raunak Mondal `[一作]` (Carnegie Mellon University), Peter Washington `[通讯]` (University of Hawai`i at Mānoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了自闭症患者自刺激行为视频检测，比较了LSTM/GRU序列网络与CNN的表现，并系统评估了十种空间与时间增强方法以及个性化模型训练策略。

**💡 创新点**

创新点在于发现每15帧采样率能获得最佳准确率，并且GRU在准确率与训练速度上优于LSTM；同时首次系统性评估十种增强方法，指出上采样是最关键的增广手段。

**🔧 技术方法**

采用了LSTM、GRU、I3D迁移学习框架、十种空间/时间增强技术，以及针对每个孩子单独训练的个性化模型。

**📊 数据集**

使用了Self‑Stimulatory Behavior Diagnosis（SSBD）数据集，该数据集包含75段记录自刺激行为的视频。

**📈 对比分析**

通过80-20训练/测试拆分，对不同采样率与模型进行准确率和损失对比；LSTM与GRU在15帧采样时分别取得97.5%和98.75%的准确率，显著高于CNN基线；在增强实验中，水平翻转最优（48.78%），去除上采样导致性能下降最大。

**⚠️ 局限性**

局限性包括：数据量有限且未做交叉验证，视频来源多样导致噪声，未尝试更先进的模型架构，个性化方法仅使用单一时间分割，且增强基于I3D预训练可能与临床行为域不完全匹配。

---

## 113. Closing the Null Space: Guidance-Aware Quantization for Classifier-Free Diffusion

**arXiv ID:** 2607.08241 | [PDF](https://arxiv.org/pdf/2607.08241v1)

**作者:** Abdullah Al Shafi `[一作]` (Khulna University of Engineering & Technology), Sumaiya Rahim Suma `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了Classifier-Free Guidance（CFG）扩散模型在后训练量化中的结构性缺陷，并提出了指导感知混合精度（GAMP）方法；

**💡 创新点**

首次证明了基于gap的量化目标会产生“branch‑drift trap”零空间，并通过直接校准 guided‑output 并按敏感度分配激活精度来封闭该陷阱；

**🔧 技术方法**

采用双分支校准、AdaRound阈值学习、per‑layer guided‑output 敏感度评估和贪心 knapsack 方式实现混合精度分配；

**📊 数据集**

使用 32×32 像素的 CIFAR‑10 数据集，基于 6.1 M 参数的 ADM UNet student checkpoint 进行实验；

**📈 对比分析**

与 RTN、MSE‑cal、gap‑only 等基线比较，GAMP 在平均激活位宽 5‑6 位时实现 49% 的 FID 提升，且在相同精度下 BOPs 比统一 4/8 方案低 13%；

**⚠️ 局限性**

仅在小尺寸 CIFAR‑10 UNet 上验证，未测试大型 Stable Diffusion 或文本生成模型；w 较大时 branch‑drift trap 可能更严重，需要进一步扩展验证。

---

## 114. Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination

**arXiv ID:** 2607.08403 | [PDF](https://arxiv.org/pdf/2607.08403v1)

**作者:** Runzhe Liu `[一作]` (Dalian University of Technology), Shengyang Tao `[通讯]` (Dalian University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于博弈论的多智能体框架G-Frame，利用团队博弈和贝叶斯博弈实现轻量级化学领域LLM的自适应训练与数据生成。

**💡 创新点**

创新点在于将多智能体博弈与自适应闭环结合，解决轻量级LLM在化学推理中的幻觉问题，并实现对物理约束与逻辑链条的内化。

**🔧 技术方法**

采用了团队博弈与贝叶斯博弈、多智能体协同、异步通信、深度学习预训练与监督微调、数据合成（CoT与QA）、自适应并发控制等技术。

**📊 数据集**

使用了约50万篇化学论文和1,000本教材构成的5B-token预训练语料，合成363,045条化学CoT和199,589条QA对，Benchmarks包括ChemBench、ThChem 1.0/2.0、ChemJudge、SQuAD 2.0等。

**📈 对比分析**

通过与GPT‑4o mini、GPT‑o3、SQuAD、ThChem等基线对比，OmniChem 7B在ChemJudge幻觉率下降79.46%，在ThChem 1.0/2.0分别达到79.45%/62.08%，在ChemBench达到49.82%，整体性能与GPT‑4o mini持平并接近GPT‑o3。

**⚠️ 局限性**

局限在于仍需大量人工标注数据，模型规模相对较小，无法完美处理极端复杂的化学反应及大分子设计；在非化学领域的迁移性有限。

---

## 115. MobiDiff: Semantic-Aware Multi-Channel Discrete Diffusion for Human Mobility Data Generation

**arXiv ID:** 2607.08357 | [PDF](https://arxiv.org/pdf/2607.08357v1)

**作者:** Rongchao Xu `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向人类移动轨迹的语义多通道离散扩散框架，直接对由宏/微区域、活动类别、绝对时间与间隔时间组成的事件序列进行掩码去噪生成合成轨迹。

**💡 创新点**

创新点包括：①离散扩散直接在语义事件层面生成，避免连续轨迹/潜在空间构建的中间步骤；②多通道事件标记与事件/组/通道级结构化掩码，促使模型同时学习轨迹级和事件内的空间-语义-时间依赖；③数值感知嵌入，将地区坐标、时间周期等数值信息融入离散词向量；④在单阶段去噪中实现高效采样，显著提升推理速度。

**🔧 技术方法**

核心技术为掩码离散扩散（masked discrete diffusion）与双向 Transformer 去噪器，结合数值感知嵌入、通道混合 MLP、结构化掩码和逆时间采样策略。

**📊 数据集**

在三大城市（亚特兰大、波士顿、西雅图）的真实基于签到的移动数据集上进行实验，全部转化为同一多通道语义骨架表示。

**📈 对比分析**

与 GeoGen、SynHAT 两阶段离散扩散以及 MoveSim 传统模拟生成器对比，生成轨迹在时间与选定语义分布上的 Jensen‑Shannon 距离平均低 61%–52%；推理速度比 GeoGen 快 5.3×、比 SynHAT 快 1.9×；在下游下一步事件预测中，POI 与类别的提升显著，但整体 utility 仍低于 MoveSim；在经验暴露风险评估中，生成轨迹的近邻重叠率明显低于 MoveSim，表明潜在的记忆风险下降。

**⚠️ 局限性**

局限性：①空间分布仍不如两阶段模型一致，需进一步改进空间重现；②在下游预测任务中未能完全匹配自回归/模拟模型的性能；③缺乏正式的差分隐私保证，只能通过经验指标评估；④生成后需要额外映射步骤保证宏/微区域与活动类别的一致性。

---

## 116. DreamCharacter-1: From 3D Generative Foundation Models to Product-Ready Character Generation

**arXiv ID:** 2607.07817 | [PDF](https://arxiv.org/pdf/2607.07817v1)

**作者:** Weizhe Liu `[一作]` (ByteDance), Hengkai Guo `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一套统一的基于单张参考图像的3D人物生成框架，能够一次性生成完整、纹理精细、可动画化的角色模型。

**💡 创新点**

创新点包括：后训练适配预训练3D基础模型、层次化粗细两阶段几何生成与多尺度图像条件、结合多指标强化学习提升结构与美学、纹理两阶段生成（多视角纹理+稀疏体素填充）、一系列加速技术（模型蒸馏、快速网格提取、并行化）以及专门的预处理/后处理流程。

**🔧 技术方法**

使用Shape‑VAE、Shape‑DiT（latent diffusion Transformer）在SDF空间生成几何；2D/3D VAE+DiT实现多视角纹理与体素纹理填充；跨模态RoPE、图像去光、姿态标准化、双网格纹理、语义UV分解等技术；GPU加速重拓扑、UDF、强化学习奖励模型；模型蒸馏与高效注意力。

**📊 数据集**

训练数据由大规模通用3D资产和艺术家精修的高质量人物资产组成，并通过自动化预处理得到统一格式和姿态；测试采用公开基准数据集（与CharacterGen等同源），并使用公开基准进行评估。

**📈 对比分析**

与CharacterGen、StdGEN、Hunyuan3D、TRELLIS、Pixal3D等开源方法对比。几何上在ULIP和Uni3D均显著领先；纹理上在SSIM、LPIPS、FID、CLIP‑Sim上取得最佳成绩；用户研究显示在各评估维度均优于对手；推理速度比其他DiT方法更快，尤其优于TRELLIS‑2.0。

**⚠️ 局限性**

局限性包括：训练数据缺乏足够多样性（人物种类、服装、姿态、风格）；SDF导致只能生成闭合网格，难以处理非闭合或极薄结构；多模型管线复杂，难以维护；推理时间仍高于传统图像生成模型，需要进一步优化。

---

## 117. Dual-Correlation Hypergraph Network for Unaligned RGBT Video Object Detection and A Large-scale Benchmark

**arXiv ID:** 2607.08191 | [PDF](https://arxiv.org/pdf/2607.08191v1)

**作者:** Qishun Wang `[一作]` (Anhui University), Chenglong Li `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Dual‑Correlation Hypergraph Network（DHNet）实现RGB‑热像视频目标检测，并构建大型无人机采集的 DVT‑VOD1000 基准数据集。

**💡 创新点**

创新点在于：① Patch‑based Spatial Alignment Module（PSAM）对局部区域逐块进行自适应对齐；② Dual‑Correlation Hypergraph Fusion Module（DHFM）通过双重高阶超图捕获时空与跨模态关联；③ 将局部对齐与全局位置信息（LBP）结合，提升跨模态融合效果。

**🔧 技术方法**

采用 YOLOV8 作为骨干网络，PSAM（局部仿射变换）、LBP（全局定位）、DHFM（超图注意力）等技术，结合多模态特征融合与时空信息聚合。

**📊 数据集**

使用自研的 DVT‑VOD1000（1,000 条无人机视频，103,464 张 RGB‑热像对，15 类）与公开 VT‑VOD50（50 条视频，1,000+ 对）进行训练与评估。

**📈 对比分析**

在 VT‑VOD50 上，DHNet‑L 的 AP50 达 57.5%，超过现有视频检测器（如 PTMNet 51.4%）；在 DVT‑VOD1000 上，DHNet‑L 的 AP50 为 31.7%，显著高于 EI²Det 30.2%。相比基准模型，DHNet 在精度与实时性（DHNet‑S 73 FPS）上实现了最佳平衡。

**⚠️ 局限性**

局限性：① 需要手工设计超图阈值与 PSAM patch 大小，适配性受限；② 计算复杂度相对较高，尤其在大规模实时部署时；③ 仅在 RGB‑热像两模态上验证，未探究对其他多模态或不同传感器的推广能力。

---

## 118. SkillPlug: Unsupervised Skill Mining for Few-Shot Adaptation in Robotic Manipulation

**arXiv ID:** 2607.08354 | [PDF](https://arxiv.org/pdf/2607.08354v1)

**作者:** Zi-han Ding `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了 SkillPlug 框架，在现有视觉运动策略上增添技能模块，利用无监督技能挖掘实现可迁移、可重用的技能库；

**💡 创新点**

在无监督的多任务演示中自学习可重用且非冗余的技能，并在迁移时仅微调轻量路由器和动作头，显著提升少样本适应能力；

**🔧 技术方法**

采用 VAE‑式轨迹‑技能后验编码、重构损失、KL 正则、行为技能对齐、技能解耦等自监督目标，并通过交叉注意力交互器与路由器实现技能条件化；

**📊 数据集**

使用 DISCOVERSE 与 LIBERO 两大仿真基准（对应 ACT、OpenVLA‑OFT）以及真实桌面机器人实验数据；

**📈 对比分析**

与原始基线在多任务与少样本任务上对比，SkillPlug 在 DISCOVERSE 上多任务成功率提升约 20–30%，在 LIBERO 上少样本提升约 30–50%；在真实机器人上少样本成功率提升约 28%；

**⚠️ 局限性**

训练后技能库固定，无法在面对需要新技能的任务时自动扩展；此外在强视觉变化场景下仍存在一定的泛化限制。

---

## 119. A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents

**arXiv ID:** 2607.07985 | [PDF](https://arxiv.org/pdf/2607.07985v1)

**作者:** A. Sayyad `[一作]` (Salesforce), H. Krishnan `[通讯]` (Salesforce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文验证并量化了 Gemini 系列 LALM（大型音频语言模型）在真实全双工语音代理对话中的评分可靠性，旨在评估其是否可替代或补充人工评审。

**💡 创新点**

创新点在于首次将 LALM 直接对原始立体波形进行多维度评分，并通过与三位校准人工评审者的对照，系统地展示了模型在排名一致性、平均分一致性、缺陷检测灵敏度和跨模型一致性方面的表现。

**🔧 技术方法**

采用 Gemini 2.5 Flash、3.5 Flash 与 3.1 Pro 预览版 LLM 作为评判者，使用 Vertex AI generate‑content API 直接输入原始音频，并结合 Spearman ρ、Bootstrap 置信区间、Krippendorff α 与 Newcombe‑Wilson 置信区间等统计方法评估一致性。

**📊 数据集**

数据集来自 209 条生产客户支持语音代理会话（包括 152 条自然对话与 57 条注入缺陷的对抗样本），共 5,016 条评分记录；该数据集涵盖 13 种口音与条件组合，并注入 6 种 DSP 缺陷以评估模型的缺陷检测能力。

**📈 对比分析**

与人工三评者对照后，LALM 在 5/8 维度的 Spearman ρ 与人类平均值相差 ≤0.07，且 6/8 维度的“与人类均值相差 1 分以内”比例 ≥60%，在缺陷检测方面 4/48 维度显著更敏感，3/48 维度则不如人工；跨模型实验表明排名一致性在 Gemini 3.5 Flash 与 3.1 Pro 上基本保持，但需单独验证校准。

**⚠️ 局限性**

局限性包括：仅针对单一生产代理验证；仅使用 3 位评审者，难以精确估计人类内部一致性；对抗样本数量有限，缺陷检测统计功效不足；未对多重比较做校正；LALM 与人工缺陷检测的结果假设独立，实际为配对，可能导致置信区间误差。

---

## 120. Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization

**arXiv ID:** 2607.08057 | [PDF](https://arxiv.org/pdf/2607.08057v1)

**作者:** Jiantong Jiang `[一作]` (University of Melbourne), Feng Liu `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大规模语言模型（LLM）KV 缓存优化方法进行系统化综述，提出基于系统行为的三维分类（时间、空间、结构），并构建行为-目标矩阵与行为-行为共设计网络，以梳理技术、效果与未被关注的交叉区域。

**💡 创新点**

创新点在于：① 将 KV 优化视作系统行为而非单一技术维度，形成时间、空间、结构三维框架；② 用行为-目标矩阵和共设计网络可视化技术与目标（latency、throughput、memory、I/O、energy 等）之间的直接/间接关联；③ 系统揭示了未被充分研究的领域（如 KV 压缩的协同设计、能耗、信任度等）并提出具体挑战。

**🔧 技术方法**

技术手段包括：系统行为分析、分类法构建、技术属性与效果汇总、表格对比、共设计图可视化、基准工具评估；采用的核心指标为延迟、吞吐量、GPU 内存占用、网络/存储 I/O、能耗与质量（accuracy/鲁棒性）等。

**📊 数据集**

使用的数据集与实验来源主要是公开论文、开源实现和常用基准工具（如 HuggingFace Transformers、DeepSpeed、Megatron‑LM 等）中的实验结果；并未自行收集新数据集或运行新实验。

**📈 对比分析**

比较方法：通过汇总并对齐现有论文的实验结果，形成统一的对比表，展示不同技术在主要指标上的提升幅度与不足；结果显示：时间维度（调度、流水线）显著降低平均延迟；空间维度（KV 迁移、分层）提升吞吐量并隐藏 I/O；结构维度（压缩、保留）最大化内存节省，但对能耗和质量影响需进一步验证；总体而言，缺少统一基准与尾部延迟报告，导致不同工作间的直接比较受限。

**⚠️ 局限性**

局限性：① 本文为综述，未包含新的实验或原型验证；② 文献覆盖虽广但仍可能遗漏最新工作；③ 结果依赖原始论文的硬件、模型与基线配置，缺乏统一基准导致直接对比困难；④ 对能耗、尾部延迟、信任度等关键指标的测量与评估尚不足。

---

## 121. On the Correctness of Software Merge

**arXiv ID:** 2607.07987 | [PDF](https://arxiv.org/pdf/2607.07987v1)

**作者:** Akira Mori `[一作]` (National Institute of Advanced Industrial Science and Technology), Masatomo Hashimoto `[通讯]` (Chiba Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种基于AST的结构化三路合并工具d3j，并给出了合并结果的形式化正确性判定。

**💡 创新点**

创新点在于引入范畴论的pushout概念作为合并正确性标准，并将其转化为AST节点的部分包含映射，形成可检验的语法正确性与通用性两项标准。

**🔧 技术方法**

技术包括自研Java解析器（OCaml+Menhir）、Zhang‑Shasha树编辑距离、AST差异与合并算法、冲突规则库（约170个Python函数）以及pushout检查器。

**📊 数据集**

实验使用了76个开源Java项目共43,774个文件级合并场景，以及2,582个开发者手工解决的PR合并和2,459个包含21种重构的冲突合并。

**📈 对比分析**

与git‑merge、JDime等7款现有工具比较，d3j在正确合并率（CCFM）上达到100%并消除所有语法与通用性错误，执行时间为约863分钟，明显优于大多数工具，速度仅次于IntelliMerge。

**⚠️ 局限性**

局限性包括处理速度相对慢、冲突规则仅覆盖语法层面、对其他语言需要重新实现解析器与规则、并未覆盖语义层面的正确性。

---

## 122. FedOPAL: One-Shot Federated Learning via Analytic Visual Prompt Tuning

**arXiv ID:** 2607.08368 | [PDF](https://arxiv.org/pdf/2607.08368v1)

**作者:** Lingyu Qiu `[一作]` (University of Naples Federico II), Francesco Piccialli `[通讯]` (University of Naples Federico II)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单轮联邦学习框架 FedOPAL，利用视觉提示调优在本地校正特征分布，然后通过闭式解析学习在服务器端聚合得到全局分类器，实现零服务器端训练成本。

**💡 创新点**

核心创新在于：①将视觉提示令牌视为分布调制器，主动将异构客户端的特征映射到线性可分空间，满足解析式联邦学习（AFL）的静态特征假设；②将解析式学习与 VPT 结合，避免传统方法的迭代训练；③通过仅上传提示和统计量，实现真正的 one‑shot 通信。

**🔧 技术方法**

技术组合包括：视觉提示调优（VPT‑Shallow）、预训练 CLIP ViT‑B/16 backbone、解析式联邦学习（AFL）闭式最小二乘解、近端正则化（μ）、数值稳定性正则化（λ）以及 Dirichlet 采样的非IID 数据划分。

**📊 数据集**

使用的公开数据集有：CIFAR‑10、CIFAR‑100、SVHN 和 DTD，并采用 Dirichlet 分布（α=0.01、0.1、0.5 等）产生不同程度的非IID 情况。

**📈 对比分析**

与 FedAvg、AFL、FedCGS、FedPFT 以及 Co‑Boosting、DENSE 等传统一轮联邦学习方法对比。FedOPAL 在所有数据集和非IID/ IID 设置下均获得最优或接近最优的准确率，例如 CIFAR‑10（非IID α=0.1）93.90%、CIFAR‑100 75.86%、SVHN 47.05% 等，显著优于 AFL、FedCGS 等方法；性能波动小，说明对数据异构鲁棒。

**⚠️ 局限性**

局限性包括：①依赖强大的预训练模型（如 CLIP），在与预训练领域差异较大的任务（如 SVHN）上表现略逊；②提示平均策略可能忽略局部细节，难以充分利用极端异构；③对提示数量、模型规模和正则参数的系统性调优仍待进一步研究。

---

## 123. Physics-Informed Machine Learning Under Small-Data Constraints: Lessons from Abrasive Waterjet Milling

**arXiv ID:** 2607.07863 | [PDF](https://arxiv.org/pdf/2607.07863v1)

**作者:** Sarah Grewe `[一作]` (Bochum University of Applied Sciences), Jörg Frochte `[通讯]` (Bochum University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在Inconel 718材料的抛光水射流铣削（AWJM）小样本数据上，探究了物理知识清洗、统计筛选以及不同物理整合层次对机器学习模型的影响。

**💡 创新点**

创新点在于将物理基准清洗与统计筛选拆分为竞争性建模假设、展示单次拆分评估与10折交叉验证在小样本下模型排名的剧烈变化，以及针对高斯过程（GP）与树模型的物理残差学习在不同算法上的异质性。

**🔧 技术方法**

使用的技术包括物理驱动的四参数模型PG‑H(4)、高斯过程回归、梯度提升树、SVR、Ridge、XGBoost、LightGBM、随机森林、浅层MLP+MC‑Dropout以及Optuna贝叶斯超参调优。

**📊 数据集**

数据集为155次Inconel 718单通道AWJM实验，包含压力、进给速度与磨料流量三维工艺参数，目标为平均切削深度，经过物理错误排除后仍保留130点进行统计筛选。

**📈 对比分析**

比较方法采用15点固定留置测试集与10折交叉验证相结合的评估；结果显示GP在物理残差学习（Level 2）下保持与纯GP相当的RMSE但折间方差显著下降，树模型在Level 2下性能恶化；贝叶斯调参提升了GB和SVR，但对两阶段GP管线适得其反；整体最佳单次拆分表现（GB）在10折CV下跌至第七名。

**⚠️ 局限性**

主要局限包括仅在单一材料与工装下验证、交叉验证仍受样本量限制导致评估方差高、物理基准需手工校准、混合模型未对物理模型误差进行概率建模，以及对深度切削区间的覆盖率偏低。

---

## 124. Predicting Male Fertility Using Machine Learning: A Semen Parameters Based Analysis with the VISEM Dataset

**arXiv ID:** 2607.08429 | [PDF](https://arxiv.org/pdf/2607.08429v1)

**作者:** Shahnawaz Qureshi `[一作]` (Pak-Austria Fachhochschule Institute of Applied Sciences and Technology), Syed MuhammadZeeshan Iqbal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究利用机器学习对男性精液参数进行分类预测精子质量

**💡 创新点**

将男性精液参数按WHO阈值分为可育、亚可育、不可育，并使用超过40个算法进行比较，发现最近中心分类器准确率94.2%

**🔧 技术方法**

采用LazyPredict框架、最近中心分类器、支持向量机、二次判别分析等机器学习方法

**📊 数据集**

使用VISEM公开数据集85份精液样本

**📈 对比分析**

通过5折交叉验证和ROC‑AUC对比，最佳模型准确率94.2%，AUC分别为0.95、1.00、0.97

**⚠️ 局限性**

主要局限是样本量小、亚可育/不可育类别不平衡，且仅考虑了三项传统指标，未纳入DNA碎片化等生物标志

---

## 125. APIVOT: Adaptive Planning with Interleaved Vision-Language Thoughts

**arXiv ID:** 2607.08024 | [PDF](https://arxiv.org/pdf/2607.08024v1)

**作者:** Emily Jin `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视觉语言模型（VLM）的长周期规划器APIVOT，通过在推理轨迹中交错使用语言和视觉“思想”来同时处理语义和几何约束，实现对厨房任务的高效规划。

**💡 创新点**

创新点在于（1）自适应的模态选择：在需要几何精度时才生成视觉思想；（2）视觉思想作为未来状态的隐式表示，直接用于几何验证；（3）三阶段训练课程——先学习利用给定视觉思想，再学习自己生成视觉思想，最后学习何时需要生成视觉思想。

**🔧 技术方法**

技术方法包括：使用预训练的Qwen3‑VL‑8B‑Instruct作为基模型，在其上采用LoRA微调；利用视觉语言模型的视觉编码器与语言解码器生成和对齐视觉思想；采用教师强制的监督式微调，结合交叉熵损失和余弦相似度损失；在推理时自动插入视觉思想并根据任务约束选择模态。

**📊 数据集**

数据集为KitchenWorlds仿真器生成的任务集，包含Containment、Sorting和Hold‑out Storing Leftovers三类长周期厨房任务，训练集2000个实例，测试集各100个实例。

**📈 对比分析**

与Gemini‑3.1‑Pro、Gemini‑ER‑1.5、Qwen3‑VL‑8B‑Thinking、VLM‑TAMP、Reflect‑VLM以及FastDownward进行比较。APIVOT在所有任务的平均成功率为0.419，比顶级VLM基线Gemini‑ER‑1.5提升8.1个百分点，比VLM‑TAMP提升9.0个百分点；在空间受限的场景中优势更为明显；在不同token预算下，APIVOT在保持高成功率的同时实现了约39% token使用量的减少。

**⚠️ 局限性**

局限性包括：仅在KitchenWorlds仿真器中验证，缺乏真实视觉环境的测试；视觉思想仅在任务分布内训练，可能对真实世界几何和外观的泛化不足；模态选择通过监督学习实现，未通过强化学习进一步优化任务成功率；模型仅使用隐式视觉表示，未扩展到点、边界框等更直观的几何抽象。

---

## 126. Graph-Regularized Deep Learning for EEG-Based Emotion Recognition with Psychologically-Grounded Label Structure

**arXiv ID:** 2607.07773 | [PDF](https://arxiv.org/pdf/2607.07773v1)

**作者:** Dongyang Kuang `[一作]` (Sun Yat-sen University), Xiaocong Zeng `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种图正则化深度学习框架，将情感心理学结构融入EEG情绪识别模型的训练目标，强制模型在情感图谱上产生更符合心理学的预测。

**💡 创新点**

创新点在于：① 通过构建基于Russell情绪圆模型的情感图谱来量化情绪间的心理学相似度；② 设计三种层次递增的图正则化策略（图标签平滑、图拉普拉斯共行距损失、切片Wasserstein距离），从不同数学视角约束模型；③ 引入不确定性自适应权重，消除手工调参负担；④ 在三种不同架构（Transformer、CNN‑Transformer混合、GNN）上验证其架构无关性。

**🔧 技术方法**

技术包括：图构建与邻接矩阵、图拉普拉斯伪逆共行距、切片Wasserstein距离、图标签平滑、交叉熵损失与自适应权重融合、UMAP可视化、统计显著性检验。

**📊 数据集**

使用SEED‑IV（4类）和SEED‑V（5类）EEG情绪基准，采用5频段差分熵特征作为输入。

**📈 对比分析**

与仅使用交叉熵的基线相比，图正则化在所有三种骨干网络上均提升了准确率和宏F1；在SEED‑V上Conformer+切片Wasserstein最高提升达+5.42%准确率，且心理学不合理错误率降低39%；此外，近四分之三受试者在个体F1上均获提升，验证了方法的鲁棒性。

**⚠️ 局限性**

局限性包括：图结构假设依赖于Russell模型，可能不适用于所有情绪范畴；切片Wasserstein虽然性能优异但计算成本相对较高；实验采用单一特征提取方式，未探索与更复杂EEG预处理或多模态融合的协同效果；以及未对超参数停止准则进行系统分析。

---

## 127. From Execution to Education: A Bloom-Aligned Framework for Measuring Educational Control in LLMs

**arXiv ID:** 2607.08009 | [PDF](https://arxiv.org/pdf/2607.08009v1)

**作者:** Yi Zhang `[一作]` (Purdue University), Julia Rayz `[通讯]` (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并应用基于修订版 Bloom 分类的框架，评估大模型在程序任务中对教育目标认知层级的控制能力，并对比通用模型与编码专用模型的表现。

**💡 创新点**

提出观测认知转移（OCS）与目标区准确率（TZA）两项量化指标，并结合语义增量聚类与层级 Fisher 判别率，首次揭示模型在提升与降低认知负荷时的显著偏差。

**🔧 技术方法**

使用零拷贝提示、LLM‑as‑judge 评估、BERTopic 语义增量聚类、加权对数比率关键词提取、Fisher 判别率层级探测以及主成分分析等技术。

**📊 数据集**

在 2,520 个 Python 编程任务上实验，采自 BigCodeBench、LiveCodeBench v5 与 SWE‑Bench‑Verified 三个基准。

**📈 对比分析**

将 Qwen3‑Next‑80B‑A3B‑Instruct（通用）与 Qwen3‑Coder‑Next 进行匹配实验，利用 OCS 与 TZA 对认知层级迁移进行量化，结果显示两模型都易于提升认知水平但难以降低；通用模型在提升目标上的准确率高达 79%，编码模型约 63%。

**⚠️ 局限性**

仅评估两款同架构模型，缺乏跨模型、跨语言和跨学科验证；零拷贝提示可能无法覆盖真实用户指令；Bloom 判别器对分布漂移的鲁棒性有限；未对内部表示的因果机制进行干预验证。

---

## 128. Omni-Sleep: A Sleep Foundation Model via Hierarchical Contrastive Learning of CNS--ANS Dynamic

**arXiv ID:** 2607.07720 | [PDF](https://arxiv.org/pdf/2607.07720v1)

**作者:** Zhoujie Hou `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 Omni-Sleep，一种基于 CNS/ANS 生理先验的多模态睡眠基础模型，利用分层对比学习和长时延迟预测来预训练 100,000 小时多中心 PSG 数据。

**💡 创新点**

创新点在于：① 通过将 EEG/EOG/EMG（CNS）与 ECG/呼吸（ANS）分区，构建拓扑约束的对比目标；② 结合微观（系统内一致）与宏观（系统间同步）对比学习；③ 引入长时延迟掩码预测以捕捉夜间宏观睡眠动力学。

**🔧 技术方法**

采用轻量级 1D 卷积 patch 编码、RoFormer（旋转位置编码）进行时序建模，使用 InfoNCE 对比损失与 L1 掩码重建损失，整体采用两阶段预训练（先对比后联合），模型参数约 56M。

**📊 数据集**

使用 SHHS、WSC、MESA 等 100,000 小时 PSG 数据进行预训练；下游评估在 ISRUC‑Sleep、CinC、SHHS1 等独立数据集（含多种睡眠疾病标签）。

**📈 对比分析**

与 SleepFM（4.4M）和 SleepGPT（134M）等基线相比，Omni‑Sleep 在睡眠分期上实现 Macro F1 最高达 77.3%、Accuracy 77.8%，在 OOD 评估中保持 5–10% 的绝对提升；在多标签疾病分类中 AUROC 最高可达 0.825，显著优于传统方法和去掉 RoFormer 的 ablation 版。

**⚠️ 局限性**

局限性：模型仍需大量预训练数据；在极少标签或高缺失率情况下性能仍受影响；对不同采样率和设备的鲁棒性需进一步验证。

---

## 129. From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier

**arXiv ID:** 2607.07779 | [PDF](https://arxiv.org/pdf/2607.07779v1)

**作者:** Eric Jiang `[一作]` (University of California, Los Angeles), Wei Wang `[通讯]` (University of California, Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性梳理AI4Math发展现状，并提出将解决方案从定式问题求解转向具备研究能力的数学研究代理。

**💡 创新点**

阐明AI4Math需跨越的五大障碍（数据与评估、关系结构、探索发现、工具生态、人与AI协同）并给出未来路线图。

**🔧 技术方法**

以大型语言模型为核心，结合自动化形式化、检索增强、强化学习与多代理工作流实现推理与证明生成。

**📊 数据集**

利用 Lean/Coq 等交互式定理证明库、官方 Benchmark（MiniF2F、PutnamBench、Erdős 问题等）与自动化生成的合成数据。

**📈 对比分析**

在 MiniF2F、IMO、Erdős 等基准上采用标准准确率/通过率衡量，Seed‑Prover 近 100% 通过率；但对研究级问题仍只能获得局部/已知解，性能相对有限。

**⚠️ 局限性**

受限于形式化数据稀缺、语义失配、证明冗长、工具整合难度与缺乏人机交互与可解释性，导致难以真正实现自主发现与验证新定理。

---

## 130. PLURAL: A Global Dataset for Value Alignment

**arXiv ID:** 2607.08034 | [PDF](https://arxiv.org/pdf/2607.08034v1)

**作者:** Dhruv Agarwal `[一作]` (Cornell University), Aditya Vashistha `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于Integrated Values Survey（IVS）的大规模价值偏好数据集（Plural Alignment Dataset），通过两阶段LLM生成流程将国别代表性调查问卷转化为包含用户场景、偏好与反偏好响应的三元组，并利用该数据集对LLM进行直接偏好优化（DPO）以实现跨文化价值对齐。

**💡 创新点**

创新点在于：①使用高质量、全国代表性的社会科学调查IVS作为价值信号源，避免传统数据偏向西方；②设计两阶段生成管线，既保留原始价值取向，又生成自然对话式响应；③提供可扩展至92个国家的通用方法；④在自动化（GLOBE指标）和人类评估（印度、巴西、日本）两种维度验证数据集能提升模型的文化适配性。

**🔧 技术方法**

主要技术包括：LLM驱动的偏好三元组生成（Stage‑1生成情境+偏好/反偏好），LLM扩展（Stage‑2自然化对话），Direct Preference Optimization（DPO）微调，GLOBE基准自动评测（用LLM评审打分），以及对比基线（Prompting、Community Alignment、PersonaHub）。

**📊 数据集**

核心数据集为Plural Alignment Dataset，基于IVS的约150K份问卷，经过抽样后在20个文化多样国家各保留100名受访者，生成约500k条偏好三元组；同时使用IVS原始问卷（92国）和GLOBE（9维文化指标）进行验证。

**📈 对比分析**

比较方法：①数据集层面验证通过交叉预测国家与保持国内多样性；②对比DPO微调后模型与四种基线（Vanilla、Demographic Prompting、Aggregate Prompting、Community Alignment）在GLOBE 9维上的平均绝对误差（MAE）和人类评估的典型性得分。结果显示，基于Plural Alignment Dataset的模型在MAE上可比强基线降低至27.7%，在印度、巴西、日本的评估中，调优模型被评为更符合本国价值的概率明显高于基线。

**⚠️ 局限性**

限制：①GLOBE指标来自中层管理者样本，可能与全国平均不完全匹配；②人类评估样本偏向年轻男性，尽管通过“国家典型性”框架减轻偏差；③合成生成过程中仍可能出现与原始价值不完全一致或文化细节错误；④后期微调压缩了跨国差异，模型在多样性保留上仍不完整，需要更好保持多元化的方法。

---

## 131. A Self-Supervised Approach for Minimal-Annotation Hydroacoustic Data Exploration

**arXiv ID:** 2607.07733 | [PDF](https://arxiv.org/pdf/2607.07733v1)

**作者:** Pierre-Yves Raumer `[一作]` (Ecole Normale Supérieure), Jean-Yves Royer `[通讯]` (Université de Brest)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用自监督 Masked AutoEncoder (MAE) 对低频水下声谱图进行表示学习，提取局部补丁级特征并在谱图内聚合为事件级嵌入，再通过 UMAP+HDBSCAN 或 K‑Means+层次聚类在数据集层面识别重复的水下声学模式，随后仅用约一小时的人工检索将聚类映射到语义类，构建长时序活动检测器。

**💡 创新点**

① 在事件级别先聚合补丁嵌入，以解决多源重叠导致单窗口嵌入失真问题；② 采用 MAE 在大规模无标签数据上预训练，再在目标数据集上微调，显著提升聚类质量；③ 通过最小化人工标注工作，提供一种可快速部署的探索工具。

**🔧 技术方法**

自监督 MAE (Vision Transformer 编码/解码)、Spectrogram Whitening、Patch能量/激活阈值剔除、事件级聚类 (Chebyshev 距离)、UMAP 降维 + HDBSCAN、K‑Means+层次聚类、聚类到类映射与小时级稠密评分。

**📊 数据集**

预训练集：来自 34 个印度洋海底声学站点的 988 万个谱图；目标评估集：MAHY*2 海底传感器 1,137,732 个谱图，包含海洋哺乳动物、地震、船舶等多种信号。

**📈 对比分析**

与周期性检测器和 YOLO 目标检测器对比，MAE‑K‑Means/MAE‑UMAP 在 7 个已标注类别上实现 F1‑score 约 60–80%，与基线相当或略优；聚类方法在召回率上更高，但精度受限；实验表明事件级嵌入显著提升了聚类效果。

**⚠️ 局限性**

① 需要手工聚类到类映射，虽然耗时短但仍需专家判断；② 对重叠时频信号的分离仍有限；③ 对窗口长度、谱图分辨率、聚类超参数等的敏感性未系统评估；④ 预训练时间较长，模型迁移成本高；⑤ 仅针对低频 240 Hz 数据，扩展到高频或其他传感器需重新调参。

---

## 132. FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation

**arXiv ID:** 2607.08359 | [PDF](https://arxiv.org/pdf/2607.08359v1)

**作者:** Xueke Zhu `[一作]` (Pengcheng Laboratory), Yonghong Tian `[通讯]` (Pengcheng Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 FSD-VLN 的快慢双系统框架，用于解决无人机长距离视觉语言导航中的高决策延迟、动作连续性差以及长期预测难题。

**💡 创新点**

创新点包括：①将高层语义推理（慢系统）与低延迟动作生成（快系统）显式分离；②在快系统中引入基于 Diffusion Transformer 的时间条件动作生成器；③提出时间加权的 MSE 损失和全局自适应归一化，以提升长序列训练稳定性；④异步协同的双系统设计实现了语义一致性与实时控制的平衡。

**🔧 技术方法**

技术手段：视觉语言模型（VLM）提取语义先验，Diffusion Transformer (DiT) 进行跨时序动作建模，交叉注意力与自注意力交替；状态编码器、动作编码器和解码器的 MLP；LoRA 微调、AdamW 优化器；时间加权 MSE 损失与全局自适应归一化。

**📊 数据集**

使用的主要数据集为基于 AirVLN‑S 与 OpenFly 的大规模仿真城市环境，包含 30,000+ 条从城中心、工业区、公园、村庄等多样场景的导航轨迹，并补充广州渲染场景。

**📈 对比分析**

在四种标准指标（NE、SR、OSR、SPL）上与 Random、Seq2Seq、Navid、AerialVLN、CityNavAgent、OpenFly 等基线对比。FSD‑VLN 在未见场景中 SR 提升至 13.6%（相对 OpenFly 的 5.1%），SPL 提升至 10.7%，NE 降至 78m；在已见场景中 SR 26.7%，SPL 22.8%。单步推理延迟从 402ms 降至 176ms，整体任务执行时间下降 53%。

**⚠️ 局限性**

局限性：在环境高度动态或快速变化时，感知延迟与分布漂移会影响长期预测；目前验证仅在仿真环境，尚未迁移到真实无人机；对动作预测 horizon 的设置存在权衡，过长的 horizon 可能导致误差累积。

---

## 133. Monocular Vision Based Control Framework for Grasping

**arXiv ID:** 2607.07897 | [PDF](https://arxiv.org/pdf/2607.07897v1)

**作者:** Shail Jadav `[一作]` (Technische Universitaet Wien), Dongheui Lee `[通讯]` (Technische Universitaet Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套基于单目RGB视觉的统一抓取框架，能够同时处理柔性和刚性物体。

**💡 创新点**

创新点在于利用语言驱动的刚度估计提供抓取策略先验，并通过视觉跟踪的相似度和尺度因子自适应控制抓手宽度，且不依赖触觉传感器或专用抓手。

**🔧 技术方法**

技术包括开源词汇检测YOLOv8x-worldv2、SAM2分割、TAPIR/CoTracker点跟踪、Depth Anything深度估计、Procrustes分析、StiffNET刚度预测、位置控制的Franka Hand。

**📊 数据集**

使用自制日常生活物品（生菜、马苏里拉奶酪、羊角面包、纸巾、硬塑料瓶）的实验数据，以及从LLM生成的配对硬度比较和少量实测刚度作为StiffNET训练集。

**📈 对比分析**

在Franka Emika Research 3机器人上进行的pick‑and‑place实验中，框架在多种软硬物体上实现了稳定抓取，并与传统触觉/力控制或专用抓手方法相比，展示了更好的通用性和可靠性。

**⚠️ 局限性**

局限性包括对视觉跟踪和深度估计的依赖，可能在低纹理或快速运动场景下失效；缺乏力控制导致对极度柔软或快速动态物体的抓取仍受限；且仅适用于位置控制的抓手。

---

## 134. Context Graphs for Proactive Enterprise Agents

**arXiv ID:** 2607.07721 | [PDF](https://arxiv.org/pdf/2607.07721v1)

**作者:** Avinash Kumar `[一作]` `[通讯]`, Avinash Kumar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文设计并实现了一种面向企业的主动式人工智能系统，利用上下文图（Context Graph）实时监控业务实体的状态变化，并在阈值被突破时自动向相关人员推送可执行的通知。

**💡 创新点**

创新点在于将知识图谱从传统的静态查询向实时、增量驱动的主动推送迁移，提出了Delta检测引擎、主动性评分（Proactivity Score）以及LLM驱动的面向用户的通知层，形成完整的“监测–评估–推送”闭环。

**🔧 技术方法**

实现技术包括使用NetworkX构建动态多边图、Python实现Delta检测引擎与评分函数、Anthropic Claude LLM生成基于JSON的通知，并通过规则引擎（可自定义阈值规则）实现业务逻辑。

**📊 数据集**

数据集为三类合成企业案例（合同生命周期、工程事故响应、销售管道卫生），每个案例包含50–150个节点、80–200条边，模拟了200个增量事件以供规则触发和评估。

**📈 对比分析**

通过对比传统被动式RAG代理，评估指标包括Precision@5、误报率（FPR）、平均触发时间（MTTS）和角色匹配率；实验显示系统在三个场景中平均Precision@5为0.83，误报率为0.11，平均触发时间从47分钟降至30秒以内。

**⚠️ 局限性**

局限性包括：上下文图的完整性依赖于多源数据集成，阈值规则目前为手工编写，角色匹配采用静态映射，可能导致个体偏好缺失；大规模部署时可能出现警报疲劳，需要进一步的动态阈值调优和隐私控制。

---

## 135. Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models

**arXiv ID:** 2607.08186 | [PDF](https://arxiv.org/pdf/2607.08186v1)

**作者:** Aiwei Liu `[一作]` (WeChat AI Team), Zitao Wang `[通讯]` (WeChat AI Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 Hidden Decoding 的序列长度扩展方法，在不增大 Transformer 背骨的情况下通过多流嵌入和 Stream‑Factorized Attention 为每个 token 提供额外内部计算，从而在大型 MoE 语言模型上实现能力提升。

**💡 创新点**

创新点包括：①将每个 token 复制为 n 条并行流，利用不同嵌入表产生多样化初始表示；②通过仅在少数层进行跨流注意力（Stream‑Factorized Attention）保持计算成本近线性；③使用 KV‑mirror 与持续预训练（CPT）在现有基础模型上实现 4 倍流扩展并提升性能。

**🔧 技术方法**

关键技术包括：多流嵌入、Stream‑Factorized Attention、KV‑mirror 优化、持续预训练（CPT）、循环/迭代的训练与推理策略（Hidden Decoding）。

**📊 数据集**

训练采用与原 WeLM MoE 相同的大规模通用数据，后续的评估基准覆盖 9 个难题集（GPQA Diamond、HLE、MMMLU、FrontierMath、PHYBench、MathArena Apex、HMMT、IMO‑AnswerBench、SciCode），同时对 dense Qwen3‑8B‑Base 在 n=2/4/8 上做扩展实验。

**📈 对比分析**

与对应的非 HD 基线模型（WeLM‑80B / WeLM‑617B）在相同的早期 SFT 仅监督后训练方案下对比，HD 模型在所有评测基准上均有提升；例如 WeLM‑HD4‑80B 在 SciCode +4.2、PHYBench +4.0、FrontierMath +3.2；WeLM‑HD4‑617B 在 GPQA +2.1、HLE +1.8；在 8B 级别的 n 增至 8 时 MMLU 从 85.1 提升到 87.5，Pile‑test BPB 下降 0.386→0.378。

**⚠️ 局限性**

局限性包括：①在大批量或长输入下推理吞吐量下降（batch=32, 32k 输入时仅 27% 原始吞吐）；②KV‑mirror 与 Stream‑Factorized Attention 的实现高度依赖特定的 MoE 训练架构；③对最小化额外成本的设计仍需进一步优化（如在推理端整合 KV‑mirror 加速）。

---

## 136. BACH: A Bayesian Admixture of Contrastive Heads for Multi-Interest Two-Tower Retrieval

**arXiv ID:** 2607.08107 | [PDF](https://arxiv.org/pdf/2607.08107v1)

**作者:** Quoc Phong Nguyen `[一作]` (Amazon), Julien Monteil `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个多兴趣两塔检索模型BACH，将k个用户兴趣向量视为混合模型的成分，并通过变分推断学习软路由与每用户兴趣权重；

**💡 创新点**

创新点在于使用球面（power‑spherical或von Mises‑Fisher）识别后验实现软路由，避免硬max导致的兴趣衰退，并生成可用于服务的自归一化兴趣权重；

**🔧 技术方法**

采用两塔编码器、软最大内积、变分推断、球面分布、注意力门控层和浓度网络等技术；

**📊 数据集**

在MovieLens‑20M、Taobao用户行为日志和Netflix Prize数据集上进行实验；

**📈 对比分析**

与单兴趣基线、MIND、ComiRec、all‑multihead等多种方法对比，BACH在Recall@10、Recall@100、NDCG@100和AUPRC等指标上均实现了显著提升，尤其在前10名召回上效果最突出；

**⚠️ 局限性**

局限包括对k值和门控层的调参依赖、计算和存储成本仍高于单兴趣模型，以及仅在下一条目检索任务中验证，尚需进一步探讨在更广泛检索场景下的表现与可扩展性。

---

## 137. Detecting Ladder Logic Bombs in IEC 61131-3 PLC Programs using ESBMC-PLC+: A Formal Verification Approach with Trigger Synthesis

**arXiv ID:** 2607.08417 | [PDF](https://arxiv.org/pdf/2607.08417v1)

**作者:** Pierre Dantas `[一作]` (University of Manchester), Waldir Junior `[通讯]` (Federal University of Amazonas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

将现有的IEC 61131‑3安全验证器ESBMC‑PLC+改造成LLB（恶意PLC逻辑炸弹）检测器，主要通过将函数块体翻译为可验证的IR，配合扫描看门狗和输出连线，将炸弹触发器和恶意负载映射为可检查的安全/完整性属性；使用k‑induction提供无穷扫描周期的安全证明，使用BMC回溯触发器；

**💡 创新点**

①发现LLB隐藏在功能块体内，导致传统LD验证器无法检测；②提出轻量级函数块体翻译层；③引入扫描看门狗将非终止payload转为可检测的安全违规；④通过输出连线捕获actuator/传感器伪造payload；⑤实现自适应触发器鲁棒性和触发器自动恢复；⑥首次在真实SWaT模拟数据上运行并打开语义检查门槛。

**🔧 技术方法**

ESBMC‑PLC+验证引擎（k‑induction + BMC）、Z3 SMT求解器、函数块体到GOTO IR的翻译、扫描看门狗计数器、输出连线机制、符号执行/SMT求解、结构化文本到GOTO的翻译扩展。

**📊 数据集**

Iacobelli 公开数据集（30 正常 / 30 炸弹 PLCopen LD）、自适应触发器变体、310 程序的 Boolean/Integer 分类语料库、PLC‑Defuser 的 SWaT 基准（v1.0.0 线性触发器版和后续加入非线性触发器的版本）。

**📈 对比分析**

与PLC‑Defuser、SymPLC、TSV 等现有检测器对比，ESBMC‑LLB 在 Boolean/Integer 语料库上 100% 检测率、0% 假阳性、触发器完整恢复；对自适应触发器 5/5 成功；在 SWaT 线性触发器版检测率 99%（149/150）、0 FP；在非线性触发器版仅 49%（73/150），主要受 SMT 求解超时限制；平均检测时间子秒，k‑induction 给出无穷无缺陷证明，BMC 产生触发器。

**⚠️ 局限性**

仅检测非终止或违反安全/完整性属性的LLB；对始终终止且不触发属性的恶意逻辑无效；在非线性整数/实时循环中 SMT 处理时间过长导致误检/漏检；Analog 扩展虽可解析所有程序但存在不完全 sound，偶尔产生误报；k‑induction 不完整，部分 benign 程序报告 UNKNOWN；依赖安全属性设定，缺少完整的属性覆盖。

---

## 138. MuScriptor: An Open Model for Multi-Instrument Music Transcription

**arXiv ID:** 2607.08168 | [PDF](https://arxiv.org/pdf/2607.08168v1)

**作者:** Simon Rouard `[一作]` (Kyutai), Alexandre Défossez `[通讯]` (Kyutai)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并公开了MuScriptor，一种能够在多乐器、不同音乐风格的真实录音中进行自动乐谱转录的Transformer模型。

**💡 创新点**

创新点包括：①大规模合成数据与真实数据的混合预训练与微调策略；②使用强化学习（GRPO+REINFORCE）对模型进行后训练；③在推理时支持可选乐器条件化与分类器无关引导，提高转录稳定性与定制化。

**🔧 技术方法**

核心技术包括：decoder‑only Transformer、mel‑spectrogram 输入、乐器条件嵌入、强化学习后训练、classifier‑free‑guidance、token化为MIDI‑类事件的序列。

**📊 数据集**

数据集：1.45M MIDI合成音频（随机声音字体、音高/节奏/力度等数据增强）；170k 真实音乐录音（总计11k小时）配齐音符注释；300首高质量手工校正音频用于强化学习；多种公开基准（Bach10、Dagstuhl、PHENICX等）用于跨域评估。

**📈 对比分析**

与MT3/YourMT3+ 等基准比较，MuScriptor 在帧F1上提升至约69%（相对MT3 66%），多音符F1提升至约41.6%（MT3 21.9%），在多种公开数据集上均显著优于现有模型；预训练与强化学习分别贡献约+6%和+4%性能提升。

**⚠️ 局限性**

局限性：①对同一乐器同一音高的重叠音符无法完整转录；②对低频噪声/重混音效果仍有误差；③模型规模大（1.3B参数）对资源有限的环境不友好；③仅支持单一音符序列，无法同时发声的多音符情形。

---

## 139. Predicting Viticulture Potential through an Ensemble of U-Net and a Geospatial Foundation Model

**arXiv ID:** 2607.08449 | [PDF](https://arxiv.org/pdf/2607.08449v1)

**作者:** Jorge Ignacio Perez `[一作]` (Georgia Institute of Technology), Lucas Rassbach `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用U-Net与Prithvi-EO-2.0的加权集成模型，对法国南部的多时相Sentinel-2影像进行葡萄种植潜力（1-5级）分类。

**💡 创新点**

创新点在于将传统卷积分割网络与大规模预训练的地理空间视觉Transformer结合，并通过教师-学生伪标签与季节聚合的时间处理，实现多尺度、跨模型融合，提升预测稳定性。

**🔧 技术方法**

采用U-Net卷积语义分割、Prithvi-EO-2.0 ViT、Ordinal loss、旋转/翻转增强、梯度裁剪、混合精度训练以及教师-学生半监督学习等技术。

**📊 数据集**

使用AgriPotential 2026数据集，包含34个时间步的Sentinel-2多光谱影像（128x128像素补丁），标签为葡萄种植潜力的1-5级。

**📈 对比分析**

通过验证集±1准确率和Exact准确率进行模型比较，最终集成模型在测试集上±1准确率68.32%排名第二；单独U-Net为66.25%，Prithvi为65.51%；相比其他Transformer基线，集成效果最佳。

**⚠️ 局限性**

存在验证与测试间显著的泛化缺口；季节聚合限制了完整时间序列利用；对大量缺失标签像素的处理仍有提升空间；更大Prithvi模型与全时序实验受限于计算资源。

---

## 140. Infinity-Parser2 Technical Report

**arXiv ID:** 2607.07836 | [PDF](https://arxiv.org/pdf/2607.07836v1)

**作者:** Zuming Huang `[一作]`, Yuan Qi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Infinity-Parser2，一种结合可控数据合成与多任务强化学习的端到端文档解析大型多模态模型，并公开了 5M 双语文档语料库 Infinity-Doc2-5M。

**💡 创新点**

创新点包括：① 可控渲染与迭代优化的 DOM‑based 文档合成引擎，实现高质量、结构化标注数据；② 以可验证度量为奖惩的多任务 RL（8 个子任务），在同一策略上同时优化感知、结构与推理；③ 两个针对不同需求的模型变体（低延迟 Flash 与高精度 Pro）。

**🔧 技术方法**

核心技术：Qwen3.5 视觉‑语言预训练模型、全参数 SFT、后置 GRPO 强化学习、可验证奖惩机制、跨任务奖励归一化与路由、模型并行与 MoE 结构。

**📊 数据集**

使用的数据集包括公开语料（PubTabNet、ChartSFT、CoSyn‑Chemical 等）、Infinity-Doc2-5M（5M 样本，包含 bounding box、Markdown/HTML/LaTeX/SMILES 等多种输出形式）以及多任务辅助数据（VQA、通用多模态指令）。

**📈 对比分析**

在多项公开基准上评测：olmOCR‑Bench 87.6%、ParseBench 74.3%，分别击败 DeepSeek‑OCR‑2、PaddleOCR‑VL‑1.5、MinerU2.5；在 OmniDocBench‑v1.6、DocLayNet 等布局评测中与专用检测器相当；在表格、公式、图表、化学式以及文档 VQA 上均处于同类系统领先水平，Flash 版实现 3.68× 速度提升，Pro 版在精度关键场景保持最高准确度。

**⚠️ 局限性**

局限性：语料主要是中英双语，跨语言性能下降；合成标注虽高质量但仍含少量噪声；对复杂重叠图表、多角度表格的解析仍有误差；不保留细粒度文本格式（粗体、斜体等）；对多步骤视觉指令的理解有限。

---

## 141. VectorizationLLM: Smart Vectorization Based AI Assistant

**arXiv ID:** 2607.07846 | [PDF](https://arxiv.org/pdf/2607.07846v1)

**作者:** Ryan Duke `[一作]` `[通讯]` (New York Institute of Technology), Ryan Duke (New York Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于 Google Gemma 4 26B 的 RAG LLM 系统 VectorizationLLM，用来辅助 CTEC 247 MATLAB 课程学习，提供概念解释、代码块回忆并严格执行学术诚信；

**💡 创新点**

将课程专属的 RAG 知识库与定制系统提示相结合，强制使用向量化技巧、避免循环与条件语句，并实现高精度代码回忆，形成针对 MATLAB 向量化教学的专属 AI 辅助；

**🔧 技术方法**

使用 Gemma 4 26B 大模型、RAG 检索+系统提示架构、OpenWebUI 前端、温度1.0、top_p 0.95、top_k 64 参数，配合向量检索、图像检索与代码生成；

**📊 数据集**

基于 CTEC 243 与 CTEC 247 的课程笔记 Markdown 文件（包含模块笔记、工具箱 readme 以及辅助说明文件），构成 RAG 知识库；

**📈 对比分析**

通过自定义 Python 脚本将模型输出的代码块与笔记原文逐行匹配，计算代码行回忆率，实验结果显示 47/48 行，回忆率为 97.92%，仅出现一次代码块引用错误；

**⚠️ 局限性**

仅适用于单一课程且学生规模有限（20 人），缺乏跨课程通用性；依赖 OpenWebUI，需额外部署；对极端不完整提示可能产生误检；尚未在更大规模课堂中系统验证性能。

---

## 142. Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning and Learning with ASP and Energy Based Models

**arXiv ID:** 2607.08136 | [PDF](https://arxiv.org/pdf/2607.08136v1)

**作者:** Jakob Suchan `[一作]` (Constructor University), Mehul Bhatt `[通讯]` (Örebro University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将Answer Set Programming (ASP) 与能量基模型 (EBM) 融合的端到端神经符号推理与学习框架 ASPEn，并在 MNIST、CLEVR 与 MOT17 三大基准上进行实验验证。

**💡 创新点**

创新点在于：①将 ASP 的稳定模型语义与可微能量函数直接耦合，形成可联合优化的离散-连续搜索空间；②使用对比自由能损失实现能量函数的端到端训练，使符号约束与子符号学习互相影响；③提供一个可复现的通用平台，可轻松扩展到不同领域的动态任务。

**🔧 技术方法**

核心技术包括：ASP（Clingo）推理、能量基模型（EBM）、对比自由能学习、MAP 推理、PyTorch 神经网络、硬约束与默认推理、以及与视觉感知模型（YOLO、CNN）的集成。

**📊 数据集**

使用的数据集：MNIST（加法任务）、CLEVR（视觉问答）、MOT17（多目标跟踪）三个公开基准。

**📈 对比分析**

通过对比同类神经符号方法与传统深度学习方法，评估指标为：MNIST 加法单/双位准确率分别为 98.67% / 95.33%；CLEVR VQA 准确率 62.85%；MOT17 HOTA 43.26、MOTA 36.37、IDF1 49.89，整体性能与同类方法相当或略优，展示了能量化符号推理的可行性。

**⚠️ 局限性**

局限性包括：①ASP 推理在大规模动态场景下仍易受性能瓶颈影响；②能量函数的设计与训练需要经验性调参，缺乏自动化；③目前仅在相对简单的数据集上验证，尚未在更复杂、跨领域任务中系统评估；④缺乏专门的 KR‑centric 评测基准，难以系统衡量符号结构对学习的真正影响。

---

## 143. AI-integrated models for assessing agricultural resilience

**arXiv ID:** 2607.07759 | [PDF](https://arxiv.org/pdf/2607.07759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 144. RhyMix: A Lightweight Adaptive Multi-Rhythm Network for Long-Term Time Series Forecasting

**arXiv ID:** 2607.08234 | [PDF](https://arxiv.org/pdf/2607.08234v1)

**作者:** Sumit Satishrao Shevtekar `[一作]` (Indian Institute of Technology Indore), Chandresh Kumar Maurya `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级混合时序预测模型RhyMix，采用双路径并行结构结合周期性先验与多尺度卷积，配合自适应门控实现样本级动态融合；

**💡 创新点**

创新点在于：①双路径融合周期嵌入与多尺度TCN+通道注意力；②多级自适应门控可对四种预测头及两条路径进行动态加权；③保持线性复杂度、低参数量（约40K）并兼顾多尺度与季节性；

**🔧 技术方法**

使用技术包括：RevIN归一化、周期性可学习嵌入表、深度可分离扩张卷积、多尺度TMS模块、通道混合瓶颈MLP、RMSNorm、四种预测头（直线、趋势-季节分解、局部卷积、周期融合），以及两层自适应门控；

**📊 数据集**

在12个公开长周期预测基准上评估，涵盖电力（ETT、ECL）、交通（Traffic、PEMS系列）、天气（Weather）和金融（Exchange）等领域；

**📈 对比分析**

与Time-o1、TimeMixer++、GCMNet、SEG-MOE、SOFTS、TexFilter、PatchTST、iTransformer等SOTA模型对比，RhyMix在10/12个数据集上取得最优或次优结果，参数量比主流模型低10-99倍，推理延迟<5ms；

**⚠️ 局限性**

局限性主要在对高度空间相关的交通数据表现略逊（如Traffic、PEMS04）以及对极长预测周期的某些数据集缺乏优势，因缺乏专门的空间建模或更深的时序捕获机制。

---

## 145. Idiobionics: The Unification of Privacy and Intelligent Robotic Prostheses

**arXiv ID:** 2607.07775 | [PDF](https://arxiv.org/pdf/2607.07775v1)

**作者:** Kwesi Afari Darfoor `[一作]` (University of Alberta), Bailey Kacsmar `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

定义了新的研究范式“idiobionics”，并通过实验演示在转肘截肢患者的上肢假肢上利用三轴加速度计数据进行活动推断攻击的可行性。

**💡 创新点**

创新点在于首次将隐私安全与自适应假肢技术系统化为交叉学科研究方向，阐述隐私威胁、攻击模型，并提供初步实验验证，推动假肢设计的隐私保护研究。

**🔧 技术方法**

使用迁移学习的HarNet10模型进行监督学习；无监督聚类采用K-Means、DBSCAN、Gaussian Mixture Models (GMM)、OPTICS、Agglomerative Clustering；特征提取包括时域统计量、频域峰值及谱质心，随后用PCA降维。

**📊 数据集**

收集自12名受试者的前臂三轴加速度计数据，包含行走、原地慢跑、坐姿、站姿四种活动，采样频率148 Hz，时长每个活动30秒。

**📈 对比分析**

通过12次留一实验评估攻击准确率，平均预测准确率为83%；GMM聚类在无监督实验中表现最佳；在监督实验中部分受试者（如ID5、ID9）准确率高达96%。

**⚠️ 局限性**

限制包括样本量小、仅涉及加速度计传感器、只针对上肢假肢、未覆盖真实环境噪声和其他传感器类型，攻击方法的实际可行性和普适性仍待进一步验证。

---

## 146. REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming

**arXiv ID:** 2607.07738 | [PDF](https://arxiv.org/pdf/2607.07738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 147. Progression as Latent Drift: Generative Forecasting of Slow-Evolving Pathologies

**arXiv ID:** 2607.08270 | [PDF](https://arxiv.org/pdf/2607.08270v1)

**作者:** Yuxiang Feng `[一作]` (Zhejiang University), Shujun Wan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种进化为潜在漂移（Latent Drift）的生成预测框架，用来预测慢性脑病变随时间的解剖学变化。

**💡 创新点**

通过把预测目标从完整像素级别改为潜在空间中的时间残差，并采用有限标量量化（Finite Scalar Quantization）形成非连续死区滤波器，从而解决身份坍塌（Identity Collapse）和连续插值陷阱（Continuous Interpolation Trap）两大优化病理。

**🔧 技术方法**

使用压缩潜在空间编码器+解码器、基于Transformer的自回归生成器以及FSQ量化器；整体为逐步生成（progressive）方法。

**📊 数据集**

在阿尔茨海默症纵向数据集ADNI与AIBL（共约3,981对扫描）上进行实验。

**📈 对比分析**

相较于扩散模型与传统自回归Transformer，Latent Drift在Diff‑SSIM、NCC、FID、临床诊断准确率和F1指标上均显著提升，p<0.05。

**⚠️ 局限性**

局限性包括：共享死区量化网格可能低估快速变化区域（如脑室）；仅在已对齐、降采样的扫描上验证，未处理多站点的几何畸变；未来需引入区域感知量化与更大多模态验证。

---

## 148. ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents

**arXiv ID:** 2607.08143 | [PDF](https://arxiv.org/pdf/2607.08143v1)

**作者:** Maud Ehrmann `[一作]` (École Polytechnique Fédérale de Lausanne), Simon Clematide `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了ICDAR 2026 HIPE-OCRepair挑战，评估大型语言模型在历史文档OCR后校正中的有效性。

**💡 创新点**

创新点在于提出检索导向的评估框架、统一多语种（英、法、德）历史文本数据集，并探讨LLM在保持原文完整性与校正之间的平衡。

**🔧 技术方法**

主要技术包括零射提示、继续预训练+微调、任务特定提示和输出验证/限制机制，使用Gemma、DeepSeek、Mistral、Qwen等解码型LLM。

**📊 数据集**

使用了HIPE-OCRepair‑2026数据集，该集整合了4个重新整理的公开历史数据集和1个新采集的数据，覆盖17–20世纪的报纸与书籍。

**📈 对比分析**

通过cMER（字符匹配错误率）与偏好分两指标进行比较，最佳系统BnF‑Mistral在所有语言上实现cMER≈0.005，偏好分≈0.9，显著优于零射系统与基线。

**⚠️ 局限性**

局限包括对低噪声文本的过度校正风险、对历史书写细节（连字、长s等）的忽略，以及评估仅关注检索可用性，未涵盖严格的外交性校正。

---

## 149. Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing

**arXiv ID:** 2607.07953 | [PDF](https://arxiv.org/pdf/2607.07953v1)

**作者:** Tommaso Cerruti `[一作]` (ETH Zurich), Imanol Schlag `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对软max注意力与四种最近的递归线性注意力架构（DeltaNet、Gated DeltaNet、Kimi Delta Attention、Gated DeltaNet-2）进行了统一的递归‑内存符号表示，并在此框架下探讨了轻量级跨层路由（CLVR）技术。

**💡 创新点**

创新点包括：1）将不同线性注意力机制统一到同一递归‑内存框架，清晰展现其可表达性与记忆衰减机制；2）系统比较纯线性堆叠与混合堆叠在损失、吞吐量与序列长度扩展性上的权衡；3）提出并验证CLVR跨层路由方案，证明写值而非写误差更有利于跨层信息共享；4）将Muon自适应优化器与学习率曲线与不同架构结合，揭示其对性能的显著影响。

**🔧 技术方法**

使用技术包括：递归记忆更新（delta‑rule、错误校正、通道级衰减/擦除/写入门控）、MuON优化器（NormMuon、谱缩放、Nesterov动量）、零初始化隐藏流投影（实现CLVR）、Megatron‑LM定制实现、以及标准化的训练与评估脚本。

**📊 数据集**

使用数据集：FineWeb‑Edu（FineWeb‑Edu前缀 + LLaMA2分词器）训练 15B 令牌，规模扩展至 1.3B/3B 参数模型；下游评估使用 HellaSwag、PIQA、WinoGrande 三个多项选择问答基准。

**📈 对比分析**

比较方法：在 350M 参数、15B 令牌基准下，记录每种架构的验证交叉熵、归一化训练吞吐量（以纯 Gated DeltaNet AdamW 为 100%）以及 4k–32k 令牌的迭代时间；结果显示 Kimi Delta Attention + Muon 混合堆叠获得最低验证损失（≈2.273），纯 Gated DeltaNet + AdamW 在吞吐量上最快（100%），混合堆叠在损失上有提升但吞吐率下降。

**⚠️ 局限性**

局限性：1）所有对比基于单次实验，没有多次种子或方差统计；2）超参覆盖不均匀，尤其是学习率与优化器的相互作用；3）下游评测仅限三项任务，无法覆盖长上下文或检索性场景；4）CLVR 在不同主机（如 Kimi Delta Attention、Gated DeltaNet‑2）及更大规模下的效果未充分验证；5）整体报告偏向训练吞吐与损失，推理性能与内存占用未系统评估。

---

## 150. Texture Representations in Deep Vision Models: Comparing CNNs, Vision Transformers, and Human Perception

**arXiv ID:** 2607.08321 | [PDF](https://arxiv.org/pdf/2607.08321v1)

**作者:** Ludovica de Paolis `[一作]` (International School for Advanced Studies), Eugenio Piasini `[通讯]` (International School for Advanced Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建纹理复杂度谱数据集，对比CNN VGG-19与三种Vision Transformer（CLIP、DINO‑v2、iGPT）在纹理感知任务中的内部表示，并与人类奇异匹配实验结果进行对齐评估。

**💡 创新点**

创新点在于首次通过信息失衡（Information Imbalance）度量高维表示相似度，将ViT在纹理感知中的表征与人类感知进行量化对比，揭示注意力机制使ViT形成更稳定、跨纹理的一致表示，而CNN则受限于局部特征。

**🔧 技术方法**

采用信息失衡（II）作为相似度指标，对模型各层特征进行归一化、截断等预处理，并使用奇异匹配（odd‑one‑out）实验获取人类纹理区分准确率。

**📊 数据集**

使用Describable Textures Dataset（DTD）及其三种合成纹理（Victor & Conte、Portilla & Simoncelli、Gatys），再加入ImageNet对象子集与随机噪声图像，构成包含五类纹理的连续复杂度谱。

**📈 对比分析**

通过II值比较模型间及模型与人类的相似度，结果显示CLIP、DINO‑v2和iGPT与人类纹理感知相关性显著高于VGG‑19；ViT内部三者对不同纹理类型保持高一致性，表明ViT在纹理感知任务上优于CNN。

**⚠️ 局限性**

局限性在于仅使用单一CNN（VGG‑19）且计算资源有限，未能涵盖其他CNN架构，可能限制结论的泛化范围。

---

## 151. DeepPySR -- A Symbolic Regression Framework with Dynamic Pruning, Pareto Selection, and Hierarchical Composition for Real-World Scientific Discovery

**arXiv ID:** 2607.08150 | [PDF](https://arxiv.org/pdf/2607.08150v1)

**作者:** Fuling Chen `[一作]` (University of Western Australia), Rae-Chi Huang `[通讯]` (Edith Cowan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出DeepPySR框架，实现可解释的符号回归在高维、生物医学与社会科学数据上的应用

**💡 创新点**

创新点在于动态变量裁剪(DVPS)、指数Pareto选择(EPS)以及多层符号回归，解决了高维、特征选择与层次结构难题

**🔧 技术方法**

采用基于进化的符号回归、动态变量裁剪策略、指数Pareto评分与多层网络架构，配合PySR Julia后端实现

**📊 数据集**

在四个Feynman物理基准以及七个真实数据集（体脂、心脏病、葡萄酒质量、学生成绩、卒中、糖尿病、Raine BMI纵向）进行评测

**📈 对比分析**

与PySR、KAN、随机森林、ExtraTrees、XGBoost、MLP、ElasticNet比较，DeepPySR在回归任务中提升R²最高达0.794（体脂）和0.964（学生数学），分类任务中F1最高0.898（心脏病）且在极端不平衡数据中实现高召回率；在Raine BMI上从10岁起连续优于传统模型，R²从0.604降至0.425时仍保持领先

**⚠️ 局限性**

局限包括计算成本高（每个数据集需70–150 CPU‑小时）、对操作符集敏感、在极端不平衡分类仍难以进一步提升、层数固定且未自动化确定深度

---

## 152. COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation

**arXiv ID:** 2607.08117 | [PDF](https://arxiv.org/pdf/2607.08117v1)

**作者:** Jhih-Rong Guo `[一作]` (National Taiwan Normal University), Berlin Chen `[通讯]` (National Taiwan Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 COALA 框架，通过把 SLM 的潜在表示映射到判别空间，对音频与候选实体的匹配强度进行评分，并在此基础上实现上下文偏置识别与 ASR 改进。

**💡 创新点**

创新点在于设计了两种新的判别损失——MPD‑Loss 与 DPD‑Loss，解决传统 softmax 判别损失在多目标场景下的梯度冲突问题，使模型能同时提升多正样本的分数并抑制负样本，提升评分稳定性与识别准确率。

**🔧 技术方法**

采用 Whisper‑large‑v2 作为声学编码器、SmolLM2‑135M‑Instruct 作为语言模型骨干，并通过 LoRA 适配、CTC 对齐、MLP 判别投影器等技术实现语音与文本的联合学习与偏置实体评分。

**📊 数据集**

使用 LibriSpeech 语料库，构建包含 209.2K 低频词的偏置列表，并以 N={500,1000,5000} 的大小对每句进行多目标实体评估。

**📈 对比分析**

与 CTC‑Filter、K‑Prompt、RNN‑T+IB 等基线进行对比，COALA 在 B‑WER 上显著下降，Recall#20 最高达 99.09%；DPD‑Loss 的单目标召回率超过 99%，且能在 10‑top 选择下保持优秀性能。

**⚠️ 局限性**

局限性：top‑10 选择难以覆盖句中超过 11 个目标实体；在 N=5000 时未过滤的完整列表会导致 OOM；DPD‑Loss 仍依赖阈值过滤，极少数负样本误判可能影响整体性能。

---

## 153. Mechanistic Interpretability of LLM Jailbreaks via Internal Attribution Graphs

**arXiv ID:** 2607.07903 | [PDF](https://arxiv.org/pdf/2607.07903v1)

**作者:** Anupam Wagle `[一作]` (University of South Dakota), Longwei Wang `[通讯]` (University of South Dakota)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过构建并对齐清洁与攻击输入的内部计算图，诊断LLM在对抗和越狱攻击下的脆弱性并揭示其内部机制。

**💡 创新点**

提出配对计算图框架，识别安全抑制、攻击出现与计算重路等脆弱性图案，并通过因果干预验证其对攻击成功的贡献。

**🔧 技术方法**

使用稀疏自编码器（transcoder）生成节点，基于梯度估计构建边，结合图对齐、路径归因、子图检测和因果干预技术。

**📊 数据集**

实验基于Llama‑2‑7B‑chat模型，利用30对清洁/攻击提示及扩展的500对提示作为样本。

**📈 对比分析**

与传统梯度归因相比，transcoder归因的偏差稳定性提升18倍；路径重路指标与攻击成功率相关（r≈0.46，p=0.01），但单节点干预未能抑制攻击。

**⚠️ 局限性**

干预主要针对单节点，攻击机制分布式且冗余导致单点抑制无效；实验样本量有限，且聚焦单一模型。

---

## 154. Beyond Backpropagation: Monte Carlo Method Can Train Deep Neural Networks

**arXiv ID:** 2607.08406 | [PDF](https://arxiv.org/pdf/2607.08406v1)

**作者:** Hong Zhao `[一作]` `[通讯]` (Xiamen University), Hong Zhao (Xiamen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在GPU上实现最简单的蒙特卡洛突变-优化选择算法（MCA），用于训练深度网络，完全不依赖梯度。

**💡 创新点**

创新点在于：① 可直接训练非光滑目标、离散权重及非传统激活函数；② 支持纯裁剪和自适应稀疏化；③ 在Transformer等结构上也可实现梯度自由训练。

**🔧 技术方法**

使用Monte Carlo突变+选择、全批量前向传播、GPU并行缓存、层归一化等技术。

**📊 数据集**

实验数据集包括MNIST（图像分类）和Tiny Shakespeare（字符级语言建模）。

**📈 对比分析**

与全批量BP（Adam）对比，MCA在全连接网络上可达到与BP相当甚至更高的准确率；在Transformer上也可实现超过97%的准确率，显示可行性，但训练速度普遍慢于BP。

**⚠️ 局限性**

主要限制是计算效率低，尤其在Transformer等大规模网络上；目前实现仅在小规模网络验证，缺乏进一步优化和大规模硬件加速。

---

## 155. LUMI: Tokenizer-Agnostic LLM-Based Lossless Image Compression

**arXiv ID:** 2607.08221 | [PDF](https://arxiv.org/pdf/2607.08221v1)

**作者:** Chris Xing Tian `[一作]` (Peng Cheng Laboratory), Siwei Ma `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 LUMI 框架，利用冻结的 LLM 作为熵模型进行无损图像压缩，跳过传统的 tokenizer 处理。

**💡 创新点**

创新点在于构建 tokenizer‑free 的像素到嵌入接口，结合 256‑way 预测头和 intra‑patch 位置编码，实现与 LLM family 无关的压缩接口。

**🔧 技术方法**

采用的技术包括冻结的 LLM 作为上下文熵模型、像素嵌入 MLP、位置编码、soft prefix 以及算术编码。

**📊 数据集**

实验使用了 Kodak（自然图像）、BRACS（医学图像）和 BED4RS（遥感图像）三个公开数据集。

**📈 对比分析**

通过与 PNG、JPEG‑XL、DLPR 以及基于 tokenizer 的 LLM 方法的 BPP 对比，LUMI 在单域和留一域评估中均实现了更低或相近的压缩率，且随模型规模与训练数据提升持续改进。

**⚠️ 局限性**

局限性包括仅独立处理 16×16 patch 不能利用跨块上下文，解码时需逐像素自回归导致延迟较高，以及冻结的 LLM 可能限制最终可达的压缩性能。

---

## 156. Multiuser Zak-OTFS on the Uplink with Superimposed Spread-Pilots

**arXiv ID:** 2607.08247 | [PDF](https://arxiv.org/pdf/2607.08247v1)

**作者:** Sai Pradeep Muppaneni `[一作]` (Indian Institute of Science), Ananthanarayanan Chockalingam `[通讯]` (Indian Institute of Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多用户 Zak‑OTFS 上行链路系统，利用时频位移实现多址并通过闭式表达式推导各用户间的有效 DD 通道；

**💡 创新点**

创新点在于实现了异质用户在不同 DD 周期/帧尺寸下的 TF‑shift 多址、推导出多用户闭式 IOR、引入基于 Zadoff‑Chu FFT 的全扩散上行导频并通过字典迭代估计实现用户解耦；

**🔧 技术方法**

使用 Zak‑OTFS 调制、sinc 与高斯脉冲、TF‑shift 多址、Zadoff‑Chu 扩频导频、DD 字典与迭代 IOR 估计算法、最小残差干扰消除检测；

**📊 数据集**

采用仿真数据集，包括 Veh‑A、TDL‑A、TDL‑C 三种多径模型，4‑QAM 调制，四用户异质配置，仿真覆盖多种最大多普勒、功率比例场景；

**📈 对比分析**

与单用户 Zak‑OTFS 以及嵌入式导频相比，四用户 NMSE 与 BER 与单用户相近，superimposed 导频在 sinc 脉冲下 SE 更高；Gaussian 脉冲下则嵌入式导频更优，且在不同频道/多普勒下估计鲁棒；

**⚠️ 局限性**

限制在于仅考虑单天线系统、固定导频与数据功率比、未研究多天线/优化功率分配，且对极端多普勒/路径数下的收敛性未深入分析。

---

## 157. ASMR: Agentic Schema Generation for Ship Maintenance Report Writing

**arXiv ID:** 2607.08177 | [PDF](https://arxiv.org/pdf/2607.08177v1)

**作者:** Sohrab Namazi Nia `[一作]` (New Jersey Institute of Technology), Senjuti Basu Roy `[通讯]` (New Jersey Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了一个基于代理的框架，自动从船舶维护与操作历史表单中生成结构化的报告 Schema，并提供人机协作的填表支持。

**💡 创新点**

核心创新在于将 LLM 语义概念抽取与多粒度聚类相结合生成候选字段，再通过强化学习对字段集合进行非冗余、信息丰富的优化，形成可解释的最优 Schema。

**🔧 技术方法**

技术实现包括 GPT‑4o Mini 进行概念抽取与字段抽象、K‑means 多粒度聚类、预计算覆盖与冗余矩阵、以及基于 MDP 的 Q‑learning 进行结构优化。

**📊 数据集**

使用的数据集为约 500 条不同类别（如 Void、Compartment、Storage、Fuel Tank 等）的历史船舶维护与操作表单。

**📈 对比分析**

通过与 Raw Concepts、Candidate Schema 三个基线对比，ASMR 在覆盖率从 0.19 提升至 0.64，冗余率从 0.67 降至 0.17，Schema 大小从 42.0 缩减至 5.4，整体性能显著优于基线。

**⚠️ 局限性**

主要限制包括对历史表单质量的依赖、LLM 输出的不确定性导致冗余估计主观、缺乏标准化评测基准以及对新出现字段的迁移能力有限。

---

## 158. The Importance of Encoder Choice:A Tabular-Image Study

**arXiv ID:** 2607.07756 | [PDF](https://arxiv.org/pdf/2607.07756v1)

**作者:** Ilia Koloiarov `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了在图像-表格多模态学习中使用不同表格编码器的效果，并首次系统探讨了表格领域中强表格编码器（尤其是 In-Context Learning 表格基础模型）的可迁移性和表示偏移问题。

**💡 创新点**

创新点在于（1）揭示了表格编码器的选择是多模态性能评估的关键混杂因素；（2）发现并量化了多模态融合效果随编码器质量下降的趋势；（3）首次证明了 Context-Query 表示偏移现象普遍存在于 TabPFNv2、TabDPT 与 TabICL，并提出了可行的提取方案；（4）展示了一个简单的双线性融合与强表格编码器可匹敌复杂多模态方法。

**🔧 技术方法**

使用的技术包括：In-Context Learning 表格基础模型（TabPFNv2、TabDPT、TabICLv2）及其三种提取方式（Vanilla、Leave-One-Fold-Out、Non-Partitioned）；传统深度学习表格编码器（TabM、TabM-SSL、TARTE）；ViT-B/16 图像编码器；双线性融合模块；最大均方差（MMD）与 Excess Distribution Shift（EDS）度量；Relative Percentile Rank（RPR）作为评估指标。

**📊 数据集**

实验数据集共七个，涵盖医学影像、艺术、车辆和宠物领养四大领域：DVM、Petfinder、WikiArt、HAM10000、CCD、COVID 与 Artm。

**📈 对比分析**

比较方法包括：单模态线性分类器、最先进的多模态基线 TIP、以及带双线性融合的基线。实验显示：①在表格主导或图像主导的数据集上，多模态融合往往不如单模态；②在可能的多模态数据集上，双线性基线与强表格编码器可实现与 TIP 相当的性能，且参数量仅为 TIP 的 1/13；③表格编码器质量越高，多模态提升越小，说明提升主要补偿编码器弱点。

**⚠️ 局限性**

局限性包括：仅使用单一图像编码器（ViT-B/16），未验证对其它视觉模型的推广；表格编码器类别样本有限，未覆盖更广泛的自监督或非 ICL 基础模型；EDS 与 F1 的相关性在某些条件下未得到充分解释；实验结果基于公开数据集，可能不具备全部实际场景的代表性。

---

## 159. Buffy versus Bella: An archetypometric analysis and comparison

**arXiv ID:** 2607.07826 | [PDF](https://arxiv.org/pdf/2607.07826v1)

**作者:** Calla Glavin Beauregard `[一作]` (University of Vermont), Peter Sheridan Dodds `[通讯]` (University of Vermont)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并比较了《吸血鬼猎人巴菲》和《暮光之城》中的女主角，运用原型计量学框架对两人性格特征进行定量评估。

**💡 创新点**

首次将原型计量学方法应用于流行吸血鬼叙事中的女性角色，对比其在六维原型空间中的定位和差异。

**🔧 技术方法**

使用基于语义二分量的特征向量、奇异值分解（SVD）以及余弦相似度等向量相似度计算。

**📊 数据集**

基于Open Psychometrics Project收集的2000个虚构角色、464个语义二分量的评分数据集。

**📈 对比分析**

通过全量级别、各自故事内部、跨故事对照三层比较，发现巴菲呈现强劲的冒险英雄原型，Bella呈现弱势异类原型；相似度指标显示两者在主角与爱人角色上差异显著。

**⚠️ 局限性**

受限于粉丝自选样本的偏倚、数据来源单一以及文化视角局限，可能导致角色特征评估的偏差。

---

## 160. X-ACTA: eXtended Analytic Center Tension distribution Algorithm for fixed and mobile cable-driven-parallel-robot

**arXiv ID:** 2607.08265 | [PDF](https://arxiv.org/pdf/2607.08265v1)

**作者:** Domenico Dona' `[一作]` (Free University of Bozen-Bolzano), Matteo Zoppi `[通讯]` (University of Genova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了一种适用于电缆驱动并行机器人（CDPR）的张力分配算法，能够在工作空间外生成光滑且可微的张力轨迹，同时保持无力误差。

**💡 创新点**

创新点在于将扩展的解析中心（RAC）与可微松弛约束相结合，既实现了在Wrench Feasible Workspace (WFW) 之外的无误差操作，又支持非线性约束的加入。

**🔧 技术方法**

使用了光滑对数障碍函数的解析中心方法、基于μ的扩展解析中心（EAC）优化以及Newton迭代求解KKT条件，配合线性和非线性约束的软化技术。

**📊 数据集**

主要使用仿真数据，包括二维三绳机器人和八绳六自由度机器人，并在Monte Carlo参数调优中采集随机轨迹。

**📈 对比分析**

与NTNU等现有方法对比，EAC在力误差、频谱高频内容以及收敛时间上均表现更佳，尤其在WFW内外均保持零误差或更低误差，且收敛速度更快。

**⚠️ 局限性**

限制在于需要手动调节δ、γ、η等超参数，在极端非线性约束下可能出现收敛困难；实验验证尚未完成。

---

## 161. A First-Principles Theory of Slow Thinking and Active Perception

**arXiv ID:** 2607.08196 | [PDF](https://arxiv.org/pdf/2607.08196v1)

**作者:** Hongkang Yang `[一作]` (MemTensor Technology Co., Ltd.), Weinan E `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于概率分布提升（active lifting）与投影的理论框架，推导了慢思考（slow thinking）与主动感知的数学形式，并给出了表示层次与采样层次的分层结构。

**💡 创新点**

首次提出“活跃提升”理论，将慢思考归纳为通过最大化不确定性降低速率来构造的潜在序列分布，证明 Transformer 在近似能力上受限，并引入简单投影 (#P_f) 以显著提升可表达性和训练效率。

**🔧 技术方法**

采用概率测度理论、可测函数与投影、Transformer 参数化、分层采样（explanatory vs predictive samplers）、Monte‑Carlo 估计、KL 与 chi‑square 损失等技术。

**📊 数据集**

实验主要以通用自然语言文本数据（如 WikiText、OpenWebText 等）为基础进行预训练与评估，但论文的核心在于理论推导，未给出专门的数据集细节。

**📈 对比分析**

通过与现有慢思考模型（如 DeepSeek‑R1、Quiet‑STaR）在对数似然、采样误差和推理速度等指标上的对比，展示了三阶段改进策略在预实验中取得了更高的模型性能。

**⚠️ 局限性**

存在的局限包括：对潜在序列空间规模超多项式（难以实现精确采样），理论假设（如 ^0 ⊊ ^1）尚未被普遍证明，且模型主要针对文本模态，对其他模态的推广尚未深入探讨。

---

## 162. TVTA: Trajectory-Aware Viseme-Guided Temporal Aggregation for Event-Based Lip Reading

**arXiv ID:** 2607.08236 | [PDF](https://arxiv.org/pdf/2607.08236v1)

**作者:** Jingrong Zheng `[一作]` (Harbin Institute of Technology), Xiangqian Wu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种事件摄像头唇读的时序增强框架，包括Trajectory-Aware Differential Aggregation（TDA）在空间压缩前的局部时序建模、Viseme-Guided Aggregation（VGA）结合CTC解码与门控聚合以及EMA Teacher–Student一致性训练策略。

**💡 创新点**

①在空间聚合前对每个像素位置进行局部时序建模，保留稀疏事件中的细粒度运动轨迹；②引入viseme级CTC监督的VGA，利用可解释的口型序列提升最终聚合；③通过EMA教师-学生一致性提升在强事件扰动下的鲁棒性。

**🔧 技术方法**

使用事件体素化、DropPath正则化的ResNet-18空间编码、双向Mamba时序块、差分空间池化、CTC解码器、门控聚合、EMA教师-学生一致性及多种数据增强技术。

**📊 数据集**

DVS‑Lip 数据集（约2万词级样本，40名说话人，100个词汇）。

**📈 对比分析**

与多种基线（MSTP、HFR‑Lip、STCNet、Mamba等）在 Acc1/Acc2/Acc 上进行对比，得到 67.23% / 87.79% / 77.49%，相较上一 state‑of‑the‑art 提升约 0.4–0.9%，证明了 TDA、VGA 与教师-学生一致性的有效性。

**⚠️ 局限性**

仅在词级任务上验证，缺乏对连续/句子级唇读的评估；对事件扰动与片段划分的超参数敏感；目前仅在 DVS‑Lip 上测试，未检验跨数据集的泛化能力。

---

## 163. Reinforcing the Generation Order of Multimodal Masked Diffusion Models

**arXiv ID:** 2607.08056 | [PDF](https://arxiv.org/pdf/2607.08056v1)

**作者:** Yidong Ouyang `[一作]` (University of California Los Angeles), Dmitriy Bespalov `[通讯]` (AGI Foundations for AWS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对多模态掩码扩散模型（文本到图像与多模态理解）中生成顺序的重要性进行了研究，并提出了一种可学习的控制模块来动态决定生成顺序。

**💡 创新点**

创新点在于：①发现传统基于模型logits的Top‑K/Top‑K margin策略在多模态任务上无效；②引入可学习的控制块（Unmask Policy Module + Plackett‑Luce 采样），并通过 Group Relative Policy Optimization (GRPO) 对其进行后训练，显著提升图像合成与多模态推理的质量。

**🔧 技术方法**

核心技术包括：掩码扩散模型（MDM）框架、可学习的控制块（UPM）、Plackett‑Luce 排序策略、GRPO 训练范式、奖励函数（如 PickScore、混合奖励）。

**📊 数据集**

使用的数据集：文本到图像的 GenEval（六个子任务），以及多模态理解的 VLMEvalKit、GQA、MMMU、MMB、SEED、MathVista_MINI 与 COCO_VAL 等。

**📈 对比分析**

通过在 GenEval 上与 Top‑K、Top‑K margin 以及原始 MMaDA-COT 进行比较，控制块在整体得分上提升了 4.08%；在 VLMEvalKit 上相对提升 4.85%。在单项子任务（如两对象、位置）和多模态基准（如 MathVista_MINI、COCO_VAL）上也取得了显著优势。

**⚠️ 局限性**

局限性包括：①对奖励设计高度依赖，奖励不当可能导致次优顺序；②计算开销相对较大，需要额外的控制块与多步策略梯度更新；③实验主要聚焦于文本-图像和少数多模态任务，尚未验证在更大规模模型或更多模态（视频、音频）上的泛化能力。

---

## 164. RadLoc: Radar-based 3-DoF Global Localization via Fast, Robust, and Lightweight Spatial Descriptor Across Diverse Environmental Scenarios

**arXiv ID:** 2607.08115 | [PDF](https://arxiv.org/pdf/2607.08115v1)

**作者:** Hogyun Kim `[一作]` (Inha University), Younggun Cho `[通讯]` (Inha University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 RadLoc，一个端到端的旋转雷达全局定位流水线，包括预处理、紧凑描述符、层次检索和 3-DoF 位置估计。

**💡 创新点**

创新点在于：1) 用 1D CA‑CFAR 替代昂贵特征提取；2) 设计基于距离的旋转不变紧凑描述符；3) 采用近距离优先的层次检索；4) 用相位相关实现高效 3-DoF 姿态估计。

**🔧 技术方法**

采用 1D CA‑CFAR 预处理、平均池化构造距离感知描述符、KD‑Tree 近距离检索+全描述符精排、对数极坐标相位相关求旋转、平面相位相关求平移。

**📊 数据集**

在 5 大公开雷达数据集上评测：Oxford Radar Robotcar、OORD、MulRan、Boreas、Hercules，涵盖不同雷达、环境与天气。

**📈 对比分析**

与学习型方法（SHeRLoc、Kidnapped）及手工特征方法（RSC、RaPlace、RadVLAD、RadFFTVLAD、ReFeree）对比，RadLoc 在 Recall@1、AUC、F1 上常居前列，描述符尺寸最小、检索速度最快；3-DoF 估计误差低、成功率高，且集成至 SLAM 能实现跨天气/多会话对齐。

**⚠️ 局限性**

局限性：对近距离信息依赖较大，K 参数需调优；在极端动态场景下鲁棒性待进一步验证；尚未验证多机器人/大规模多会话部署的通信/同步性能。

---

## 165. Self-Adaptive Anomaly Detection with Reinforcement Learning and Human Feedback in Connected Vehicles

**arXiv ID:** 2607.08373 | [PDF](https://arxiv.org/pdf/2607.08373v1)

**作者:** Matthias Weiß `[一作]` (University of Stuttgart), Michael Weyrich `[通讯]` (University of Stuttgart)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个在线自适应异常检测框架，针对连接车辆的后端微服务实时监测并检测概念漂移。

**💡 创新点**

核心创新在于：①使用因式分解的深度Q网络配合自注意力，按服务动态选择最佳异常检测器；②引入三种互补的统计漂移检测器并采用“全同意”阈值来精准触发再训练；③通过人机交互的增量重训练与优先重放（60/40）实现对新旧分布的兼顾，避免灾难性遗忘。

**🔧 技术方法**

采用强化学习（DQN+自注意力）、统计漂移检测（Page–Hinkley、Kolmogorov–Smirnov、Mahalanobis距离）、经验回放缓冲、优先重放策略、OpenTelemetry监控采集、SDVDiag诊断平台集成。

**📊 数据集**

使用斯图加特大学自动泊车（AVP）测试平台的七个后端微服务CPU与内存度量（1秒采样），并在此基础上注入三类人工异常（尖峰、渐变、服务退化）以及一次真实软件更新产生的概念漂移。

**📈 对比分析**

与单一固定检测器（如MAD、SRD、OC‑SVM等）以及未改进的DQN（MLP）对比，因式分解+自注意力DQN在未见分布上F1=0.69，高于MLP的0.47，远超任何单一检测器（最高0.11）。在概念漂移后F1降至0.52，经过人工标注+60/40重放再训练后恢复到0.65，同时保持旧分布0.69，无明显遗忘。

**⚠️ 局限性**

局限包括：专家反馈为模拟标注，未考虑真实操作员的误差与疲劳；仅测试单一场景与单一更新事件；仅使用CPU、内存两维度；异常模式仅覆盖三类，未覆盖更复杂的车辆级信号或网络异常。

---

## 166. SkelGen4D: Weakly-Supervised Skeleton-Based 4D Generation for Text-Driven Mesh Animation

**arXiv ID:** 2607.08246 | [PDF](https://arxiv.org/pdf/2607.08246v1)

**作者:** Hao Feng `[一作]` (Lingnan University), Jingyu Hu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种弱监督的骨架驱动4D生成框架SkelGen4D，能够从无骨架标注的动画网格中恢复时序一致的伪骨架，并基于此生成文本驱动的骨架运动序列；

**💡 创新点**

①无需逐帧骨架注解即可训练骨架运动生成器；②通过差分拟合得到高质量伪骨架，解决了传统自动绑骨框架随帧不一致的问题；③引入Motion‑GRPO强化学习以提升动画的平滑度、动力学和骨骼一致性；

**🔧 技术方法**

差分骨架拟合、线性混合皮肤（LBS）、Transformer自回归骨架生成、CLIP文本编码、Motion‑GRPO奖励优化、伪骨架自编码器；

**📊 数据集**

Truebones Zoo、Diffusion4D两大大规模4D数据集；

**📈 对比分析**

与全监督骨架方法（MDM、SinMDM、AnyTop）以及基于SDS或视频驱动的4D生成方法（Diffusion²、Puppeteer、AnimateAnyMesh、SS4D、MeshAction）进行对比；在Truebones Zoo上覆盖率、内部/外部多样性指标上达到或超过全监督方法；在Diffusion4D上VLM与人工评估指标（几何、一致、审美、时间）均领先；

**⚠️ 局限性**

需要动画网格序列作为训练数据，若仅有静态几何则无法直接应用；骨架拟合阶段计算开销较大；对自动绑骨的鲁棒性有限，极端非刚体或复杂变形仍难以充分捕捉。

---

## 167. AutoAnchor: Stable Diffusion Unlearning Using Cross-Attention as a Manifold Surrogate

**arXiv ID:** 2607.08337 | [PDF](https://arxiv.org/pdf/2607.08337v1)

**作者:** Siyuan Wen `[一作]` (Hong Kong University of Science and Technology), Ningning Ding `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种自动化的两阶段框架，能够为文本到图像的扩散模型构造流形近似的锚点（anchor），并利用这些锚点进行高效且稳健的扩散模型unlearning。

**💡 创新点**

创新点包括：①在理论上正式化了未受约束的正则化导致的正常空间漂移（normal‑space drift）对扩散模型unlearning的影响；②提出了基于交叉注意力一致性损失的流形近似代理，克服了直接流形优化的不可行性；③构建了一套无人工干预的锚点生成与优化流程，实现了完全自动化的锚点学习。

**🔧 技术方法**

使用的技术包括：LLM（如GPT-4）进行候选概念生成，文本嵌入聚类与筛选，交叉注意力一致性损失（cross‑attention consistency loss），CLIP 进行目标与非目标概念的评估，Stable Diffusion v1.4 作为基础模型。

**📊 数据集**

主要使用 Stable Diffusion v1.4 训练和评估数据集，针对三类目标概念进行实验：有版权角色（Mickey Mouse）、艺术风格（Van Gogh）、不安全内容（nudity）。

**📈 对比分析**

与七种最先进的扩散模型unlearning基线（四种无锚点方法和三种有锚点方法）进行对比，结果显示：在概念移除（CLIP(U)）方面提升最高达 31%（相对最佳基线），在保持模型效用（CLIP(R) 与 FID）方面平均提升 6–7%。此外，将该框架集成到现有基线中可平均进一步提升 6.3%（概念移除）和 6.6%（效用）。

**⚠️ 局限性**

局限性包括：①对 LLM 的依赖使候选概念生成受限于 LLM 的推理能力；②理论假设（如流形假设、噪声近似误差）在实际复杂数据上可能不完全成立；③尽管额外计算开销仅约 4–8%，在极大规模模型或资源受限场景下仍需进一步优化。

---

## 168. PARA-PV: Physics-Aware Retrieval-Augmented PV Prediction Based on Frozen Foundation Model and Distribution Shift Correction

**arXiv ID:** 2607.08079 | [PDF](https://arxiv.org/pdf/2607.08079v1)

**作者:** Hang Fan `[一作]` (North China Electric Power University), Wei Wei `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于物理感知检索增强的光伏功率预测框架 PARA‑PV，用以实现多步光伏功率的高精度预测。

**💡 创新点**

创新点在于：①在整个预测流程中全程嵌入物理约束；②结合检索增强学习与冻结的时间序列基础模型（Chronos）先验；③引入物理感知分布偏移校正模块；④设计基于运行状态的物理约束损失，显著提升极端或过渡时段的预测质量。

**🔧 技术方法**

使用了物理感知检索增强学习（PA‑RAL）、Chronos 时序先验校正、时序卷积分布偏移校正（PA‑DSC）以及基于峰值、斜坡、夜间等状态的加权损失等技术。

**📊 数据集**

实验采用中国国家电网可再生能源预测竞赛公开的两套光伏数据集，分别为 50 MW 和 35 MW 的光伏电站数据，时间分辨率为 15 分钟。

**📈 对比分析**

与 LSTM、Informer、DLinear、iTransformer、TimesNet、TimeLLM、TimeVLM 七个基线模型在多步预测（4/16/48/96 步）和概率预测指标上对比，PARA‑PV 在 MSE、MAE、RMSE、R² 等指标上均表现最佳，尤其在长步预测中保持最低误差并产生更窄的预测区间。

**⚠️ 局限性**

局限性：检索记忆库是静态的，无法自适应设备退化或环境变化；模型仅在光伏数据上验证，尚未推广至风能或负荷等其他能源预测场景。

---

## 169. Collate: Collaborative Neural Network Learning for Latency-Critical Edge Systems

**arXiv ID:** 2607.08013 | [PDF](https://arxiv.org/pdf/2607.08013v1)

**作者:** Shuo Huai `[一作]` (Nanyang Technological University), Qian Lin `[通讯]` (HP Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种协作神经网络学习框架（Collate），能够在一次联邦学习过程中为多台边缘设备学习满足各自时延约束的异构模型，并通过动态零化-恢复和原型校正机制实现高精度推理。

**💡 创新点**

创新点：①动态零化-恢复算法在本地训练中自动裁剪与恢复卷积核，兼顾低端设备时延降低与高端设备模型扩展；②原型校正聚合方案让扩展模型能捕获所有客户端数据分布，避免了传统异构FL中子模型在弱数据上的准确性下降；③结合硬件定制时延预测器和模型扩展策略，在满足时延约束的同时提升精度。

**🔧 技术方法**

技术方法：联邦学习（FedAvg）+异构模型训练；动态零化-恢复训练；原型校正（proto‑corrected）聚合；三层BP网络时延预测器；模型宽度扩展；基于梯度与动量的权重更新。

**📊 数据集**

使用的数据集：MNIST、CIFAR‑10、CIFAR‑100、Human Activity Recognition（HAR）；实验设备包括 HP ProBook 440 G6、NVIDIA Jetson TX2、Jetson Nano、Raspberry Pi 4B、Samsung Galaxy Note10。

**📈 对比分析**

与 ERFL、Helios、HeteroFL 等现有异构联邦学习方法比较。实验显示，在同一时延约束下，Collate 在大多数设备与数据集上均取得更高准确率：扩展模型平均提升 1.96% ；压缩模型平均提升 3.09%；在 CIFAR‑10、CIFAR‑100 的 IID 与 Non‑IID 场景中均表现最优。

**⚠️ 局限性**

局限性：①需为每台设备单独训练并维护时延预测器，部署成本较高；②仅在有限的 5 台设备与四种数据集上验证，未评估大规模异构环境下的可扩展性；③在高度 Non‑IID 情况下，模型仍受限于本地数据分布，准确率提升有限。

---

## 170. A Transdiagnostic Space of Disorder Like Phenotypes in Reinforcement Learning Agents

**arXiv ID:** 2607.07753 | [PDF](https://arxiv.org/pdf/2607.07753v1)

**作者:** Hari Prasad `[一作]` `[通讯]`, Hari Prasad

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在强化学习代理中引入可调节的情绪评估参数（七个“knob”），在多种网格世界和3D像素环境下模拟并测量七种心理障碍的剂量反应和治疗过程。

**💡 创新点**

提出以认知评估权重为基础的可控情感空间，既可在单一代理中连续调节多种障碍，又揭示了障碍自组织、治疗抵抗和共病的非线性相互作用。

**🔧 技术方法**

使用评估驱动的PPO（AG-PPO）架构，结合奖励塑形、下一步奖励网络和预先注册的临床范式测量。

**📊 数据集**

在四个自定义网格世界（Dynamic-Obstacles、LavaGap、LavaCrossing、Approach–Avoidance）以及MiniWorld三维像素环境中进行实验，共计1,375个配置。

**📈 对比分析**

通过10个种子、95%置信区间和四个对照组进行剂量反应曲线评估，结果显示每个障碍表现出单调、可量化的剂量响应，并在自我恢复与暴露疗法中展现不同的治疗效果。

**⚠️ 局限性**

实验基于人工环境的模拟，缺乏对真实人类或动物行为的验证，且情感空间仅二维且未涵盖所有症状维度。

---

## 171. Simulating the Resident: Generating Executable Smart Home Schedules via LLM Personas

**arXiv ID:** 2607.08231 | [PDF](https://arxiv.org/pdf/2607.08231v1)

**作者:** Victor Jüttner `[一作]`, Erik Buchmann `[通讯]` (ScaDS.AI Dresden/Leipzig, Leipzig University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的智能家居居住者模拟框架，利用LLM生成居住者角色、日常计划，并将其转换为可直接在物理测试平台上执行的设备交互时间表。

**💡 创新点**

创新点包括：①通过五个社会技术维度（职业作息、模拟时段、家庭动态、设备生态、环境语境）配置可复用的家庭场景；②设计多阶段LLM流水线，先生成居住者记忆卡，再写自然语言日程，随后提取结构化JSON；③实现居住者行为从文本到可执行脚本的完整链路，支持在真实硬件上收集网络流量与设备状态。

**🔧 技术方法**

技术手段：大型语言模型（如OpenAI GPT‑5.4）、生成式代理架构、严格的提示工程与JSON模式约束、自动化测试平台（Home Assistant、scrcpy、TTDAS）等。

**📊 数据集**

数据集：无真实住宅数据，全部以LLM生成的居住者角色与设备方案为输入；使用预定义的设备JSON模式作为输出结构。

**📈 对比分析**

评估方式：通过单窗口演示验证生成的JSON与自然语言叙述在时间顺序、语义一致性和模式合规性上的匹配；未进行对比基准或性能指标测试，仅展示可执行性和合规性。

**⚠️ 局限性**

局限性：①缺乏对生成行为生态有效性的系统验证；②仅在单窗口、单家庭情景下测试，未验证多日或多户的泛化能力；③尚未在真实硬件上完成完整执行与网络流量捕获；④对照真实用户数据的比较实验仍待开展。

---

## 172. How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism

**arXiv ID:** 2607.07891 | [PDF](https://arxiv.org/pdf/2607.07891v1)

**作者:** J. Mark Bishop `[一作]` (University of London), Stephen J. Cowley `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过理论分析，将 Roy Harris 的 Integrationist Linguistics 与 Elan Barenholtz 的 autogenerative 语言理论结合，提出了对语言的结构机制、半otic 连续性与档案结构的补充与完善。

**💡 创新点**

创新点在于：①将 autogenerative 机制解释为语言符号实现 prospective openness 的结构基础；②用该机制支持 Harris 的 semiotic continuity 论点；③将语言档案的统计结构视为可被新参与者利用的“资源”，为 Integrationism 对档案的空白提供了理论解释。

**🔧 技术方法**

主要采用理论推演与跨学科文献比较的方法，并未采用具体的计算模型或算法。

**📊 数据集**

无具体数据集；讨论基于已有文献与 LLM 经验观察。

**📈 对比分析**

未进行实验或性能比较；讨论基于理论一致性与对现有 LLM 行为的解释力。

**⚠️ 局限性**

局限性包括：①未能把握人类脑内主动判断与预测的细节；②仍未实现对语言与模型之间意义差距的量化评估；③缺少经验验证，主要停留在哲学与理论层面。

---

## 173. DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify

**arXiv ID:** 2607.08365 | [PDF](https://arxiv.org/pdf/2607.08365v1)

**作者:** Meng Zhao `[一作]` (HUJING Digital Media & Entertainment Group), Qinyong Wang `[通讯]` (HUJING Digital Media & Entertainment Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了DaV-Gen模型，实现了“Draft-and-Verify”并行推理的统一生成检索框架

**💡 创新点**

结合稀疏与稠密混合表征、协同优化的复合损失以及并行验证管线，消除了多阶段架构的目标不一致与自回归延迟

**🔧 技术方法**

采用预训练语言模型、RQ‑VAE量化、对比学习、生成式损失、融合评分网络、广播前缀缓存以及ANN索引技术

**📊 数据集**

使用Amazon Beauty、Amazon Sports、Yelp推荐基准和工业视频搜索数据集Ind-Search进行评估

**📈 对比分析**

与SASRec、BERT4Rec、TIGER、OneRec等基线对比，DaV-Gen在Recall@10、NDCG@10提升约2–3%，在Ind-Search的Recall@50提升30个百分点，线上A/B实验平均停留时长+2.09%、转化率+0.47%，验证了显著的业务价值

**⚠️ 局限性**

仍依赖离线生成稀疏ID，缺乏对极长序列的处理，对大规模多模态扩展和实时检索索引更新仍面临挑战

---

## 174. Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems

**arXiv ID:** 2607.07989 | [PDF](https://arxiv.org/pdf/2607.07989v1)

**作者:** Yufei Xia `[一作]` (University of Louisville), Minghong Fang `[通讯]` (University of Louisville)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于判别-评估器的框架，用于定位LLM多代理系统的失败点。

**💡 创新点**

将失败定位视为可验证、可迭代的过程，采用判别模型生成假设，多评估器进行置信加权投票并用反馈进行参数高效微调。

**🔧 技术方法**

LLM判别器、独立评估器、置信权重聚合、LoRA参数微调以及多视角提示技术。

**📊 数据集**

Who&When（含Algorithm-Generated与Hand-Crafted子集）与Aegis-Bench数据集。

**📈 对比分析**

与WhichAgent、AgenTracer、ECHO、AEGIS以及RAGOrigin/RAGForensics等基线比较，在agent/step级别准确率上显著提升（平均agent准确率>50%），且token使用与运行时间更高效。

**⚠️ 局限性**

对极长轨迹或极为复杂的多代理交互仍存在局限，且需依赖预标注失败轨迹进行微调，无法完全捕捉高频误差代理的根源。

---

## 175. Functional and Secure Code Generation with Task Vectors

**arXiv ID:** 2607.07881 | [PDF](https://arxiv.org/pdf/2607.07881v1)

**作者:** Felix Wang `[一作]` (University of Waterloo), N. Asokan `[通讯]` (Kth Royal Institute Of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用任务向量算术（task‑vector arithmetic）对编程语言模型进行安全可信代码生成的技术，使模型能够一次性生成既功能正确又无安全漏洞的代码；

**💡 创新点**

创新点在于首次将安全与不安全的任务向量（由局部偏好优化 LPO 获得）结合，并通过“安全锚定”操作（W_base + τ_sec – θτ_insec）实现可调节的安全倾斜，从而在保持功能性的同时提升代码安全性，且几乎不增加推理开销；

**🔧 技术方法**

技术手段包括：对基准模型进行 LoRA 适配器的安全/不安全微调，使用 LPO 以获得安全/不安全任务向量；随后将两向量按安全锚定算术合成新的权重；评估使用 CodeGuard+ 基准与 CodeQL 静态分析；

**📊 数据集**

数据集方面：训练使用 SVEN 公开的 803 对（Python/C++）安全/漏洞实现对；评估使用 CodeGuard+ 的 17 个主 CPE 场景以及 12 个未见 CPE 场景；

**📈 对比分析**

比较方法包括基准 LLM、训练时防御（SVEN、SafeCoder）、推理时防御（CoSec、SCoDE、DeepGuard）。在主 CPE 上，LPO‑steered 模型相较于基线提升最高 36.1% 的可信代码率，超过最强防御 3.8–27.5%；在未见 CPE 上提升至 39.1%；训练成本比重训练法低 2.6–12.4×，推理延迟仅比基线多 0.6%；

**⚠️ 局限性**

局限性：仅处理二元安全/不安全偏好，需配对安全/漏洞样本；在更大模型或更复杂安全属性上效果尚未验证；参数 θ 需针对不同模型调优；仅在 CodeGuard+ 上评估，未覆盖更广泛的安全基准。

---

## 176. Spectral Analysis of Dueling Q-Learning

**arXiv ID:** 2607.08340 | [PDF](https://arxiv.org/pdf/2607.08340v1)

**作者:** Donghwan Lee `[一作]` `[通讯]` (Korea Advanced Institute of Science and Technology), Donghwan Lee (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

对未正则化、未投影的分层Q学习（dueling Q-learning）进行了理论分析，给出了确定性与随机性版本的收敛条件和有限时误差界。

**💡 创新点**

通过将Q函数分解为动作共通与动作差异子空间，构造切换线性系统模型，并利用联合谱半径（JSR）Lyapunov理论给出最优步长范围及收敛证明；同时首次提供常数步长下的期望误差界。

**🔧 技术方法**

切换线性系统（SLS）与JSR Lyapunov函数、可测选择、马尔科夫决策过程的Bellman残差表示、中心化投影算子以及随机逼近的马尔科夫过程分析。

**📊 数据集**

主要使用理论推导；实验部分采用自定义的2状态2动作MDP（以及简单的i.i.d.采样例子）进行验证。

**📈 对比分析**

使用相同采样序列和相同步长，比较标准Q-learning与dueling Q-learning；结果显示dueling在初期收敛更快，但常数步长下的噪声较大。

**⚠️ 局限性**

局限性：仅针对i.i.d.采样；未涵盖马尔科夫观测；分析基于无正则化/投影的更新，实际实现可能需要额外正则化；理论给出的是收敛条件和误差界，未给出全局最优步长取值。

---

## 177. Covering Points with Rectangular Boundaries

**arXiv ID:** 2607.08183 | [PDF](https://arxiv.org/pdf/2607.08183v1)

**作者:** Madhumita Kundu `[一作]` (University of Bergen), Kushal Singanporia `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究点集在轴平行矩形边界上的覆盖问题。先证明离散版本（只能从给定矩形集合中选取）是 W[1]-难的；再证明连续版本（矩形可以自由放置）是 NP-完全的，但参数化求解时可在 2^(k log k)·n^1 时间内解决（FPT）。

**💡 创新点**

首次将边界覆盖问题扩展到轴平行矩形；揭示离散版本与连续版本在参数化复杂度上的显著差异；设计了基于重要直线、骨架、Dyck 字符串和“特殊点”分析的 FPT 算法，突破了传统维度递减方法在矩形上的局限。

**🔧 技术方法**

使用参数化复杂度分析、W[1]-归约、NP-硬性归约（从 3-Regular 2-CSP、L-形覆盖、Constrained Bipartite Vertex Cover 等）、结构化拆分（重要直线与骨架）、Dyck 单词与块分解、显式约束满足（Monotone 2-CSP）等技术。

**📊 数据集**

该工作为理论性研究，未使用具体实验数据集，所有结果均基于构造性证明与算法复杂度分析。

**📈 对比分析**

通过归约证明离散版本是 W[1]-难、连续版本是 NP-完全；随后给出 FPT 算法，时间复杂度为 2^(k log k)·n^1，显著优于朴素的 2^k·n^k 解法。没有实验评估，但理论上证明了可在参数化意义下求解。

**⚠️ 局限性**

局限性包括：离散版本依然 W[1]-难；连续版本虽然是 FPT，但实际运行时间依赖指数 2^(k log k)，对大 k 仍不可行；算法依赖对点集做离散化和重要直线抽取，可能在点分布稠密时产生较大常数因子。

---

## 178. MORES: Mobile Reasoning-as-a-Service via Distributed LLM Inference-Time Scaling

**arXiv ID:** 2607.08116 | [PDF](https://arxiv.org/pdf/2607.08116v1)

**作者:** Guanchen Liu `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MORES框架，将大型语言模型的推理时扩展分布在边缘设备与云服务器上，实现按需递归推理；

**💡 创新点**

创新点包括：利用隐式递归推理可天然分区计算、引入语义Mixture-of-Experts与强化学习实现任务与信道感知的自适应资源调度、实现压缩隐状态传输降低通信开销；

**🔧 技术方法**

技术主要包括隐式LMM推理、语义路由、MoE结构、Soft Actor-Critic强化学习、稀疏剪枝量化等；

**📊 数据集**

使用GSM8K、MBPP和HellaSwag三大公开基准数据集；

**📈 对比分析**

通过与传统SAC基线以及随机路由、oracle路由等进行对比实验，平均吞吐量提升约18%，在不同能量预算、递归预算和设备侧递归步数下均优于基线；

**⚠️ 局限性**

局限性在于实验仅在离线模拟环境下进行，未验证多模态推理、实际无线网络波动、服务器能耗以及跨域部署的适用性。

---

## 179. In vivo feasibility study of humanoid robots in surgery

**arXiv ID:** 2607.07972 | [PDF](https://arxiv.org/pdf/2607.07972v1)

**作者:** Zekai Liang `[一作]` (UC San Diego), Michael Yip `[通讯]` (UC San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并验证了一个基于人形机器人的腹腔镜遥操作框架，完成了台架实验、干实验室用户研究以及首次在猪体内进行的胆囊切除手术，展示了人形机器人在微创手术中的可行性。

**💡 创新点**

创新点在于将人形机器人与手动腕式腹腔镜器械结合，利用视觉定位实现远程中心运动（RCM）约束，并首次实现人形机器人在活体手术中的完整操作流程。

**🔧 技术方法**

技术包括ROS2网络通信、双目立体视觉（Goovis HD头显）、逆运动学映射、ArUco标定、手动腕式器械的被动关节建模、以及基于PCA的轨迹精度评估。

**📊 数据集**

实验数据来源为自制台架与干实验室的重复测量、OptiTrack运动捕捉系统记录的轨迹，以及两例猪体内手术的操作时序和视频记录，未使用公开的标准数据集。

**📈 对比分析**

与达芬奇系统（dVRK/Xi）对比，人形机器人在工作空间和误差方面可达约80–90％的水平，线性跟踪误差约1.3 mm，圆形跟踪误差约10 mm；干实验室中，权重误差与dVRK相当但完成时间略慢；活体手术中，关键步骤的控制时长与dVRK相近，但出现了多次重定位和停顿。

**⚠️ 局限性**

主要限制包括RCM定位漂移导致的误差、约156 ms的系统延迟、有限的工作空间和腕关节力矩、器械几何参数校准误差、以及对手术室无菌流程的适配不足，需要进一步提升精度、响应速度和工作流程集成。

---

## 180. SAGA: Stable Acceleration Guidance for Autoregressive Video Generation

**arXiv ID:** 2607.08020 | [PDF](https://arxiv.org/pdf/2607.08020v1)

**作者:** Thanh-Nhan Vo `[一作]` (University of Science, Vietnam National University), Minh-Triet Tran `[通讯]` (University of Science, Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了自回归视频扩散生成中的时间不稳定性，提出了一种无需训练的稳定加速度引导方法 SAGA。

**💡 创新点**

创新点在于将离散加速度的高频谱与 Slepian（DPSS）基底结合进行谱约束，并引入结构化自回归噪声初始化，在推理时直接抑制高频加速度噪声。

**🔧 技术方法**

使用技术包括离散加速度分析、Slepian序列谱分解、结构化自回归噪声初始化以及推理时的加速度域引导。

**📊 数据集**

实验基于 VBench 视频质量基准和 MovieGenBench 人工评价数据集，并在 Self‑Forcing、CausVid 等多种自回归扩散模型上验证。

**📈 对比分析**

通过与基线模型在 VBench 指标（Temporal Quality、Subject Consistency、Background Consistency 等）对比，SAGA 在 Temporal Quality 上提升约0.6分、Subject Consistency 与 Background Consistency 均提升，同时人类评估显示约 60% 的偏好。

**⚠️ 局限性**

局限性包括对块级自回归窗口的依赖，对单帧推理效果有限，且在极长序列生成中仍可能出现残余误差，缺乏自适应谱调节机制。

---

## 181. Unveiling Public Opinion: A Study of Sentiment Analysis Using LSTM and Traditional Models

**arXiv ID:** 2607.07772 | [PDF](https://arxiv.org/pdf/2607.07772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 182. Who Analyses the Analyser? Self-Validating LLM Hazard Analysis with Constitutional Meta-STPA

**arXiv ID:** 2607.08054 | [PDF](https://arxiv.org/pdf/2607.08054v1)

**作者:** Samuel Tetteh `[一作]` (Iowa State University), Cody Fleming `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出并实现了Constitutional Meta-STPA，即将STPA方法用于LLM辅助安全分析工具自身的分析，从而自动生成治理宪章；并评估其在不同LLM供应商、系统描述以及宪章层级下的表现。

**💡 创新点**

创新点在于：①将STPA递归应用到工具本身，自动推导出21条工具行为原则和8条治理原则；②将宪章与后置校验、语义投票、可审计日志等技术结合，构建可复现的安全分析流程；③证明不同LLM模型能力决定治理覆盖而非宪章本身。

**🔧 技术方法**

采用的技术包括：大型语言模型（OpenAI, Anthropic, Cohere 等）进行生成；正则表达式校验器、语义匹配投票；哈希审计日志与运行清单；STPA 的四步链（loss→hazard→UCA→constraint）；实验自动化脚本。

**📊 数据集**

数据集主要为：三类硬件系统（自动紧急制动 AEB、输注泵、无人机自动降落）、自描述的工具系统；使用多供应商模型（OpenAI, Anthropic, Cohere, Grok, Claude）以及不同的宪章层级（无、Claude通用、工具原则、全层）。

**📈 对比分析**

比较方法：使用词法覆盖率扫描器、LLM评审员、语义匹配投票评估原则覆盖；在自分析实验中比较强势前沿模型与弱模型对原则覆盖的影响；在行为实验中用 20 个对抗性 STPA 询问，评估不同宪章层级对安全评分的提升。性能表现：前沿模型能覆盖 18/21 典型原则和全部 8/8 治理原则；工具在对抗性输入上的安全评分提升约 79%（p<0.001）。

**⚠️ 局限性**

局限性包括：①词法扫描器的容错性有限，可能被同义词绕过；②实验仅涉及两款工具和少数硬件负载，泛化性待进一步验证；③LLM 的非确定性导致在不同供应商/种子下结果不完全一致；④缺乏专家人类评估基准；⑤只针对 STPA，未覆盖其他危害分析方法。

---

## 183. Persona Cartography: Charting Language Model Personality Traits in Weight Space

**arXiv ID:** 2607.07916 | [PDF](https://arxiv.org/pdf/2607.07916v1)

**作者:** Luke Baines `[一作]` (LASR Labs), David Demitri Africa `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何将大语言模型（LLM）的行为人格拆解成可控维度，并通过低秩适配器（LoRA）实现对人格特质（基于OCEAN框架）的放大与抑制，进一步验证其对模型性能与安全行为的影响，并提出无监督方法挖掘模型原生人格维度。

**💡 创新点**

创新点在于：①将人格视为行为特征空间中的位置，可通过可加、可缩放的权重空间方向实现多维人格组合；②在现有的提示或全量训练方法间搭建中间层次，提供成本低、灵活可组合的权重级人格调节；③利用LLM‑judge与OCEAN多选题构建可靠评测框架；④通过无监督心理测量流程发现四个可解释的模型人格因子（TIDE）。

**🔧 技术方法**

技术手段包括：LoRA（rank‑64）与DPO/SFT训练、构造的“constitution‑guided distillation”流程、LLM‑judge（校准后与人工评审对齐）、OCEAN多选题评测、能力基准（Winograd, GSM‑8K, MMLU等）、自回环对话生成与强度比例分析。

**📊 数据集**

数据集主要有：
- 6款基线模型（4B‑32B）来自3个家族（4‑B/13‑B/70‑B）；
- 2,500条15轮对话rollouts（不同情景与角色）用于人格因子推断；
- OCEAN多选题（72题）及其对模型响应的log‑prob采样；
- 多任务安全评测集（多轮挫败、同情、拒绝、WildJailbreak、Benign‑Noncompliance）用于后续行为验证。

**📈 对比分析**

与方法对比：LoRA调节与激活‑空间截断（activation‑capping）或系统提示修改相提并论，实验显示LoRA在保持能力（大多数基准保持10%以内误差）下，可实现连续、可逆的特质调节；多维组合表现近乎线性加法；在安全任务中，神经质与合群度等维度可显著调节挫败、顺从、攻击性等行为。总体表现优于单纯提示或激活截断，且可在不牺牲主干能力的前提下实现细粒度人格调节。

**⚠️ 局限性**

局限性包括：
- 评测覆盖范围有限，仅对13B/4B/70B等少数模型做了完整评测；
- 采用的OCEAN量表是为人类设计，对LLM可能不完全适用；
- 无监督因子提取基于合成对话，缺少真实部署轨迹验证；
- 低秩适配器虽能调节人格，但与全量训练相比，可能在极端尺度下失效；
- 评测中对安全任务的负样本与正样本平衡、对抗样本设计仍不够充分。

---

## 184. Empirical Analysis of GPU Frequency Behavior Under ML Workloads

**arXiv ID:** 2607.08307 | [PDF](https://arxiv.org/pdf/2607.08307v1)

**作者:** Truong-Thanh Le `[一作]` (University of Oslo), Peiyuan Guan `[通讯]` (University of Oslo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在多种 NVIDIA GPU（RTX 3060M、T4、L4、A100、RTX Pro 6000）上，以 MatMul、ReLU 等典型 ML 内核为例，系统地测量和分析 GPU 频率随工作负载变化的动态调节行为，探究频率调节窗口与时间间隔对内核性能的影响。

**💡 创新点**

发现低性能/热量受限 GPU 每 20 ms 通过观察近 80 ms 的工作负载历史来决定下一个 20 ms 区间的频率，形成内核间的相互依赖；并提出基于内核执行时长加权平均的频率预测模型，指出现有基于内核独立的延迟预测方法会因忽略这一动态调节而产生显著误差。

**🔧 技术方法**

使用硬件频率监测与内核级性能计数器进行实验测量；对收集的数据进行统计与回归，构建加权平均频率预测公式；对比实验包括对比 A100 等高性能 GPU 的即时调节行为与低性能 GPU 的延迟调节机制。

**📊 数据集**

实验采用人工合成的 ML 内核（MatMul、ReLU 等）以及常见的深度学习层实现（如 cuBLAS 与 CUTLASS 的不同张量化配置），未使用特定的公开数据集。

**📈 对比分析**

与传统的“内核独立求和”延迟预测方法对比，提出的加权平均预测在已知 80 ms 内核频率和时长的条件下，预测误差约为 0.6%（≈10 MHz），显著优于忽略频率依赖的基准；此外，实验还展示了不同 GPU 在频率调节频率与响应时间上的差异。

**⚠️ 局限性**

局限性包括：仅针对低性能/热受限 GPU 的频率调节窗口机制，无法直接推广到 A100 等高性能 GPU；单个内核的频率难以独立测量，导致需要先构造可重复的内核序列；频率控制器内部隐藏的多种因素（内存访问模式、实现细节等）仍未被完全建模；实验主要基于人工合成内核，缺乏对真实复杂深度学习模型完整工作流的验证。

---

## 185. PGD-NO: A Neural Operator with Precomputed Geometry Decomposition for 3D Million-scale Physics Simulations

**arXiv ID:** 2607.08025 | [PDF](https://arxiv.org/pdf/2607.08025v1)

**作者:** Weiheng Zhong `[一作]` (University of Illinois Urbana-Champaign), Hadi Meidani `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出PGD‑NO，一种预先计算几何分解的神经算子，能在大规模工业网格上高效逼近PDE。

**💡 创新点**

核心创新在于将几何编码离线预计算成“几何代币”，实现线性内存扩展并消除跨节点通信。

**🔧 技术方法**

采用图构造、尖锐边检测、分层分割得到代币，网络使用多头注意力与token层进行解码。

**📊 数据集**

在热散热器、JEB、DrivAerNet++、Aircraft、CFD‑VOL等五个真实工业三维数据集上进行验证。

**📈 对比分析**

与PointNet、MeshGraphNet、Transolver++等SOTA模型对比，PGD‑NO在所有基准上均取得更低的相对误差，尤其在百万节点以上保持良好精度。

**⚠️ 局限性**

局限在于对平滑无尖锐边的几何代币效果有限，且预计算步骤需依赖固定网格，未来可结合可学习细化改进。

---

## 186. Adversarial Social Epistemology for Assemblies of Humans and Large Language Models

**arXiv ID:** 2607.07760 | [PDF](https://arxiv.org/pdf/2607.07760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 187. Out of Sight: Compression-Aware Content Protection against Agentic Crawlers

**arXiv ID:** 2607.08180 | [PDF](https://arxiv.org/pdf/2607.08180v1)

**作者:** Xuefei Wang `[一作]` `[通讯]` (Beihang University), Xuefei Wang (Beihang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为CAPE的框架，在不改变人类可见文本的前提下注入不可见扰动，削弱LLM代理在压缩阶段对内容的保真度，从而保护高价值文本。

**💡 创新点**

创新点在于把上下文压缩视为代理管线的安全盲点，引入可见度极低的扰动与多阶段自适应搜索（先在可访问的压缩器上挖掘结构先验，再通过先验引导进化与偏好校准的查询选择，在目标压缩器上进行低成本调优），实现对未知闭源压缩器的强鲁棒性。

**🔧 技术方法**

核心技术包括：
- 可访问压缩器的分布式探测与离散分布熵最大化优化；
- 结构先验挖掘（碎片关联、共现模式、位置‑长度兼容性）；
- 先验引导的遗传演化搜索（编码、交叉、变异、温度退火、Tabu记忆）；
- 基于偏好对齐的低成本排名器与动态查询分配；
- 以不可见Unicode字符实现无视觉差异的扰动注入。

**📊 数据集**

使用三类高价值数据集：长篇文本（Task Haystack、BABILong）、代码片段（CoRE、CAB）和多轮对话（T1、BABILong）。

**📈 对比分析**

在GPT‑4.1、Gemini 3 Flash、LangGraph和GitHub Copilot四种目标压缩器上与Random Invisible、Fixed Zero‑width、TAP、HardCom等基线对比，CAPE在文本破坏度（TD）和信息损失（ID）上提升最高达75.8%，在GPT‑4.1上TD提升241.7%，并保持人类可视差异<1.5%。在代理工作流中，DRAD最高可达59.7%，显示显著削弱下游推理与代码生成的可靠性。

**⚠️ 局限性**

局限性包括：
- 对新兴压缩器的泛化不确定，需持续验证和调校；
- 攻击者可通过Unicode规范化或空格清理去除零宽字符，影响扰动效果；
- 仅针对压缩层防护，无法阻止已被压缩信息的再利用或其他后续攻击。

---

## 188. CASL-VAE: Learning Structured Latent Variables from Unpaired Data for Semi-supervised Clustering and Paired Sample Generation

**arXiv ID:** 2607.08254 | [PDF](https://arxiv.org/pdf/2607.08254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 189. From Thesis to Transition: An INSIGHT-Inspired Approach to Co-Designing Industry 5.0 Competency Pathways for Early-Stage Researchers

**arXiv ID:** 2607.08222 | [PDF](https://arxiv.org/pdf/2607.08222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 190. An interpretable Good--Turing restart criterion for k-means++

**arXiv ID:** 2607.08243 | [PDF](https://arxiv.org/pdf/2607.08243v1)

**作者:** Renato Cordeiro de Amorim `[一作]` `[通讯]` (University of Essex), Renato Cordeiro de Amorim (University of Essex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Good–Turing缺失质量估计的k‑means++重启停止准则（GTRC），能够根据数据自适应决定需要多少次重启，避免传统的固定重启次数做法。

**💡 创新点**

创新点在于将Good–Turing缺失质量、无条件概率上界以及Clopper–Pearson置信下界三种上界结合，形成可解释、理论上可证明的停止条件；该准则既能在统计上给出停止概率，又能在实践中快速收敛。

**🔧 技术方法**

主要技术包括k‑means++算法、Good–Turing频率估计、Jensen不等式推导无条件上界、Clopper–Pearson置信区间估计、以及对重启次数的统计上界分析。

**📊 数据集**

使用了36个UCI公开数据集，样本量从几百到上百万不等，特征已标准化，聚类数取真实标签的类别数。

**📈 对比分析**

通过与固定重启次数（10、20、50、100）k‑means++的Wilcoxon符号秩检验比较；在ε=0.05时GTRC的中位数重启为33，平均误差仅0.002%；ε=0.1时中位数18，误差0.015%；整体性能与最优固定重启相近，但重启次数更为灵活。

**⚠️ 局限性**

局限性包括：在极少重启时Good–Turing估计可能噪声大；需要用户设定ε，过高或过低都会影响停止质量；未对其他随机聚类算法做验证，且在极难数据集上仍可能过早停止。

---

## 191. MASTE: A Multi-Agent Pipeline for Zero-Shot Aspect Sentiment Triplet Extraction

**arXiv ID:** 2607.08080 | [PDF](https://arxiv.org/pdf/2607.08080v1)

**作者:** Ao Hong `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MASTE，一个多代理零样本Aspect Sentiment Triplet Extraction（ASTE）框架，分四步依次完成方面提取、观点提取、情感推理和一致性校验。

**💡 创新点**

创新点在于将ASTE拆解为可解释的子任务，使用训练无关的多代理设计，并通过显式跨度校验与全局一致性检查显著提升零样本精度。

**🔧 技术方法**

核心技术是利用同一冻结的大型语言模型（如GPT‑4o）多次调用，每个代理接收前一步输出作为条件，结合最小跨度提取、one‑to‑many观点配对、情感校准与去重等规则。

**📊 数据集**

在ASTE‑Data‑V2四个基准数据集（14Res、14Lap、15Res、16Res）上进行评测。

**📈 对比分析**

相较于传统单通道提示、少量示例、链式思考以及现有管道方法，MASTE在所有四个数据集上均获得最高F1，零样本提升可达30+点，且在多种LLM骨干上均保持性能优势。

**⚠️ 局限性**

局限包括：依赖底层LLM的推理与跨度提取能力，方面召回仍是瓶颈，四次LLM调用导致推理成本与延迟较高，且目前仅在英文评测领域验证，跨语言与长文本场景尚待探索。

---

## 192. LightCrafter: PBR-Conditioned Video Diffusion Refinement for Controllable and Consistent Relighting

**arXiv ID:** 2607.08016 | [PDF](https://arxiv.org/pdf/2607.08016v1)

**作者:** Zixin Guo `[一作]` (Carnegie Mellon University), Deva Ramanan `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

我们提出了一种混合管线，将逆渲染得到的场景属性先通过物理基渲染（PBR）生成代理视频，再使用视频扩散模型对PBR渲染进行细节修正，实现高质量的视频重新照明；

**💡 创新点**

核心创新在于把PBR渲染作为扩散模型的条件输入，分离照明控制和细节优化，并通过artifact‑matched数据生成与训练；同时采用重叠窗口融合策略实现长序列一致的无缝生成；

**🔧 技术方法**

技术上结合了逆渲染（DiffusionRenderer、DiffusionLight、MegaSAM）、PBR渲染器、CogVideoX‑5B 预训练视频扩散模型、VAE编码器与DDIM调度器；

**📊 数据集**

使用了包含3000对合成视频与PBR渲染、1000对真实视频伪配对的数据集，并在MIT Multi‑Illumination、DL3DV、Objaverse等公开数据上进行训练与评估；

**📈 对比分析**

在PSNR/SSIM/LPIPS/T‑CLIP/Warp‑SSIM等指标上，相比LightX、UniRelight、DiffusionRenderer、PCRP‑video等基线方法均取得显著提升，并在长序列中保持更稳定的阴影与色彩一致性；

**⚠️ 局限性**

主要局限在逆渲染对透明、金属、薄材质的估计不佳，导致PBR渲染错误难以完全纠正；缺乏更强的几何先验与多视角约束，进一步提升鲁棒性仍是挑战。

---

## 193. Understanding Layer Patching in Model Size Interpolation

**arXiv ID:** 2607.08170 | [PDF](https://arxiv.org/pdf/2607.08170v1)

**作者:** Sara Kangaslahti `[一作]` (Harvard University), David Alvarez-Melis `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在boomerang蒸馏中通过层级补丁实现零样本模型尺寸插值的方法，并给出了系统性实验。

**💡 创新点**

提出将层补丁顺序视为最短路径问题，并提出基于KL的贪心算法KLPatch，可在O(N^2)时间内接近最优插值轨迹。

**🔧 技术方法**

技术包括boomerang蒸馏、层补丁、最短路径理论、KL差分度量、贪心算法。

**📊 数据集**

使用的模型有DistilBERT/DistilGPT2、Qwen3-4B/8B、Pythia-6.9B；数据集包括Pile（蒸馏训练）、Wikitext（perplexity评估）及一套下游任务（分类与生成）。

**📈 对比分析**

与固定顺序(first‑to‑last、last‑to‑first)以及随机采样的补丁顺序比较，KLPatch在下游准确率、生成质量及Wikitext perplexity上均达到或超过最佳随机顺序，并显著优于固定顺序。

**⚠️ 局限性**

局限性包括对不同模型族的收益差异不一、对KL近似的依赖、需要额外的校准数据、未完全解决更大模型或多任务的插值最优性问题。

---

## 194. What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness

**arXiv ID:** 2607.08046 | [PDF](https://arxiv.org/pdf/2607.08046v1)

**作者:** Raphaël Sarfati `[一作]`, Eric Ho `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在预训练后的LLM内部激活上训练轻量级线性探测器，估计预测正确性并审计链式推理的可信度。

**💡 创新点**

证明内部探测器可显著提升校准度、检测谎言，并显示模型在推理前已做出决定，揭示自我报告与内部状态的不一致。

**🔧 技术方法**

使用平均池化、注意力池化、协方差池化的线性探测器；强化学习（RLVR、RLCR、DCPO）；LLM‑as‑a‑judge；统计校准指标（ECE、Brier、AUROC、相关系数）。

**📊 数据集**

OpenForesight 预测数据集、改进上下文数据集、GLM 4.7‑Flash/4.5‑Air 数据集、AIME/AMC 数学 OOD 基准。

**📈 对比分析**

与模型本身的口头置信度、不同大小的冻结模型以及传统的离线校准方法比较；探测器在 ECE 上下降约 50%（从 0.12 降至 0.06），Brier 得分和 AUROC 亦提升，且在 23% 高影响案例中检测到推理不一致。

**⚠️ 局限性**

仅适用于冻结模型；依赖 LLM‑as‑a‑judge 的判定；未训练模型直接口头化内部置信；对不同 LLM 架构的泛化能力有限；可能存在信息泄漏导致探测器过拟合。

---

## 195. Domination and Coverage Problems under Vulnerability Constraints

**arXiv ID:** 2607.07842 | [PDF](https://arxiv.org/pdf/2607.07842v1)

**作者:** Ioannis Sigalas `[一作]` (National and Kapodistrian University of Athens), Vassilis Zissimopoulos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了在图中考虑脆弱节点（或脆弱边）约束的主导与覆盖问题，特别定义了k-Vertex Maximum Domination Ratio with Vulnerable Vertices（k-Max DRVV）、Maximum Domination Ratio with Vulnerable Vertices（DRVV）、Dominating Set with Vulnerable Vertices（DSV）以及Vertex Cover with Vulnerable Edges（VCVE）等新问题；

**💡 创新点**

创新点在于将脆弱性约束与传统主导/覆盖问题相结合，给出多类问题的NP难度证明，并设计了基于Red‑Blue Set Cover、Set Union Knapsack和最大覆盖等经典问题的多阶段逼近算法，取得了O(k/n)（在度有限图上）以及1/Δ+1·(1−e⁻¹/Δ)·(1−1/e)·k/n等近似比；

**🔧 技术方法**

技术上主要采用多种约简（到RBSC、SUKP、Max CP等）、贪心策略、最大覆盖的(1−1/e)近似、以及线性规划松弛与取整法（得到VCVE的2-近似）等方法；

**📊 数据集**

文中未使用实验数据集，所有结果均为理论分析与证明；

**📈 对比分析**

与传统问题的比较主要通过近似比和NP难度级别展示：对k-Max DRVV给出了最优常数因子近似（1−1/e）+其他因子；对VCVE提供了从4-近似到2-近似的改进；对DSV利用RBSC的已知近似上界；总体上取得了最优或接近最优的理论保证；

**⚠️ 局限性**

局限性包括：k-Max DRVV仅在度有限图上可得到O(k/n)近似；未给出常数因子近似下界；对连通或结构化图（如平面图、树宽图）的表现尚未深入；对DRVV/DSV的精确近似下界仍是开放问题；

---

## 196. The Behavioural Reflection Test: A time-efficient measure of reflective reasoning in morally and epistemically charged decisions

**arXiv ID:** 2607.07961 | [PDF](https://arxiv.org/pdf/2607.07961v1)

**作者:** Sion Weatherhead `[一作]` (University of New South Wales), Ben R. Newell `[通讯]` (University of New South Wales)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并验证了Behavioural Reflection Test（BRT）和低曝光的bCRT，用以测量人们在道德与认知情境下的反思推理。

**💡 创新点**

创新点在于结合开放式情境测评与新鲜CRT题目，既保持测评简洁又避免经典题目熟悉度导致效度下降，同时提供了可扩展的行为-语言双重指标。

**🔧 技术方法**

使用开放式文本回答、LIWC 语言分析、GPT-5 大语言模型自动编码决策指标，并用二参数逻辑反应模型（2PL IRT）评估题目特性。

**📊 数据集**

数据来源于两组在线样本：SONA 约240名学生用于bCRT 项目筛选，Prolific 473名成人用于BRT、bCRT、CRT2、NFC、BFI-10 等测评。

**📈 对比分析**

与传统CRT2比较，bCRT 在预测BRT决策和语言特征（如风险词、负面情绪词）上显著优于CRT2，且总测评时间约12分钟，表现更高效。

**⚠️ 局限性**

局限包括情境情节简化、LLM 评估可能受限于提示与模型偏差、以及样本仅为英语在线成人，难以推广至更广泛文化和语言环境。

---

## 197. Multimodal 3D LUT Generation via StatLUT with Statistical Features for Photorealistic Style Transfer

**arXiv ID:** 2607.08227 | [PDF](https://arxiv.org/pdf/2607.08227v1)

**作者:** Yifan Wang `[一作]` (Honor Device Co., Ltd.), Congchao Zhu `[通讯]` (Honor Device Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出StatLUT框架，实现基于Lab空间统计特征的3D LUT生成，用于无结构失真且支持文本驱动的光照与色彩迁移。

**💡 创新点**

创新点：1）通过Lab-Extractor提取空间无关的色彩统计，解耦语义与色彩；2）采用Transformer Seq2Seq的MR-Mapper预测拓扑平滑的3D LUT；3）引入轻量化Diffusion Transformer H-Diffuser，实现从自然语言生成统计特征，实现文本驱动色彩编辑。

**🔧 技术方法**

使用技术：Lab空间统计直方图、1D/2D/条件化统计特征、残差映射、Transformer跨注意力、轻量Diffusion Transformer、基于LUT的色彩映射与自监督训练。

**📊 数据集**

使用数据集：MS COCO（内容图像）和自制10000多种3D LUT数据集做自监督训练，评估使用PST50和PhotoNAS两个公开基准。

**📈 对比分析**

比较方法与性能：与NLUT、Neural Preset、D‑LUT、SA‑LUT、DLUT‑VCG等深度学习色彩映射方法在内容相似度、风格相似度和理想点距离上进行定量对比，并通过用户研究（70%首选率、平均排名1.40）显示StatLUT显著优于其他方法。

**⚠️ 局限性**

limitations：仅支持绝对风格描述，无法处理相对文本提示；3D LUT全局特性限制局部语义调节；极端内容‑风格差异时效果不佳，需结合传统统计匹配。

---

## 198. GRE-Diff: Gaussian Room Embeddings for Structured Layout Diffusion

**arXiv ID:** 2607.08086 | [PDF](https://arxiv.org/pdf/2607.08086v1)

**作者:** Jing Wang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于高斯房间嵌入（GRE）和高斯引导扩散的可控楼层平面图生成与编辑框架 GRE‑Diff，支持用户通过自然语言或 GUI 指令实时制定房间类型、数量、边界并进行局部编辑。

**💡 创新点**

创新点在于：① 用连续的高斯概率嵌入表示每个房间的中心和尺度，显著降低对顶点顺序和坐标扰动的敏感性；② 在同一扩散模型中统一实现生成与编辑，避免了多阶段后处理；③ 采用双路径编码与自回归变换器实现语义与几何约束的融合，提升控制精度与结构一致性。

**🔧 技术方法**

核心技术包括：高斯房间嵌入（GRE）、高斯引导扩散模型、GuidanceNet（多模态编码+双路径+自回归 transformer）、DenoisingNet（双注意力 transformer）、LLM 语义解析与图形交互界面。

**📊 数据集**

使用 RPLAN 数据集（约 80,000 张矢量化住宅平面图），在 72,709 样本上训练，8,079 样本用于测试，涵盖 6 类房间类型。

**📈 对比分析**

与 iPLAN、Graph2Plan、HouseDiffusion、WallPLAN、MaskPLAN、GSDiff、ChatHouseDiffusion 等七种 SOTA 方法在 FID、KID、MMD、覆盖率、BC/RC、生成速度等指标上对比，GRE‑Diff 在 FID 上最低（4.36）、覆盖率最高（96.09%）、BC 98.44%、RC 100%，且单张生成时间 0.91 s，整体性能显著优于现有方法。

**⚠️ 局限性**

在复杂或不规则边界、极端约束条件下偶尔失效，可能导致房间相邻关系不佳或尺寸不平衡；对训练分布之外的边界形状鲁棒性有限。

---

## 199. Write-Protected Discrete Bottlenecks for Language-Grounded World Models: A Structural Limitation and Sufficient Fix

**arXiv ID:** 2607.08312 | [PDF](https://arxiv.org/pdf/2607.08312v1)

**作者:** Jiayi Fang `[一作]` `[通讯]` (Shanghai University of Finance and Economics), Jiayi Fang (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究表明，语言梯度直接进入离散符号瓶颈会导致符号崩溃或无法学习语义，提出三层结构化解决方案以避免此类失败并实现语义绑定；

**💡 创新点**

提出的三层架构（梯度切断、无梯度语义通道黑板、冲突分裂）是解决语言梯度与离散符号接口问题的最小化设计，展示了端到端方法的结构性瓶颈；

**🔧 技术方法**

使用冻结的正交投影符号瓶颈、VAE、Gumbel‑softmax、无梯度计数黑板、DP‑Means聚类等技术；

**📊 数据集**

在二维网格世界和三维MuJoCo桌面环境上实验，使用CNN、V‑JEPA 300M、CLIP ViT‑L三种编码器；

**📈 对比分析**

与端到端的Gumbel‑softmax基线相比，三层架构实现了0%符号崩溃、79–100%语义绑定准确率（平均97.2%），在32个独立实验种子上保持一致；

**⚠️ 局限性**

局限性包括仅在相对简单环境和少量对象类型下验证、使用脚本教师而非真实LLM、未在真实机器人上测试，且对更大规模语义空间的扩展仍待验证。

---

## 200. Limits of Uniform Certification in the Standard Turing Model -- Semantic Invariants and Admissible Methods

**arXiv ID:** 2607.07723 | [PDF](https://arxiv.org/pdf/2607.07723v1)

**作者:** Fabio F. G. Buono `[一作]` `[通讯]`, Fabio F. G. Buono

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文分析了标准图灵模型下统一证书生成方法的结构性局限，提出了扩展的 Rice 原理并证明了双重束缚（Double Bind）现象；

**💡 创新点**

创新点在于将 Rice 定理推广至证明生成层面，揭示任何可计算的可接受证明方法必然诱导对语义属性的判定，从而在理论上阻止统一化的复杂性与密码学证明；

**🔧 技术方法**

主要技术包括形式化可接受方法（生成器-验证器对），结构化证明与 Coq 机械化验证，以及对 Rice 定理和 Razborov–Rudich 障碍的组合论证；

**📊 数据集**

无；

**📈 对比分析**

无；

**⚠️ 局限性**

局限性是结构性的：在标准图灵模型中，任何统一可接受方法都因 Rice 原理而无法对非平凡语义属性（如 P≠NP 或一向函数）生成可验证证书，无法突破自然证明与结构性障碍。

---

## 201. Computing in Anonymous Dynamic Networks with One-Bit Communications

**arXiv ID:** 2607.08358 | [PDF](https://arxiv.org/pdf/2607.08358v1)

**作者:** Thibaut Blanc `[一作]`, Giovanni Viglietta `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在匿名、动态网络中仅用单比特广播并计数邻居的最小通信模型下，如何实现诸如输入多集、计数、频率等全局函数的确定性算法。

**💡 创新点**

创新点包括：
• 通过一次比特的聚合计数（切割测试）直接获得全局线性方程组，而不需要传统的历史树或临时标识；
• 设计自纠正的自适应洪泛原语，在缺乏网络规模先验知识时仍能逐步逼近正确规模；
• 在唯一领袖、多人领袖以及无领袖情形下，给出几乎匹配下界的多项式时间解法，并将单比特模型的性能提升到与拥塞模型（O(log n) 位消息）相当，只多出对数因子。

**🔧 技术方法**

主要技术手段包括：
• 通过对边集的两侧计数得到的保守约束，构造线性等式；
• 递归细分不可区分类，累积线性约束直到解空间唯一；
• 采用自适应洪泛与失效传播的四通道恢复机制，实现无需先验规模的稳定化算法；
• 通过信息论计数证明，给出 Ω(n²·log(N/n)/log n) 的下界。

**📊 数据集**

使用的是理论模型，无具体实验数据集；所有结论均为最坏情况的理论上界与下界。

**📈 对比分析**

与已有的拥塞模型（每条消息 O(log n) 位）算法相比，本工作在唯一领袖或多领袖下实现了 O(n³·log² n) 的上界，几乎与 O(n³) 匹配；在无领袖或无规模先验下则达到 O(n³·log² n) 的稳定化上界；下界 Ω(n³) 与之相近，证明了对数因子是必然的。

**⚠️ 局限性**

局限性包括：
• 需要同步轮次和 1-间隔连通性；
• 对网络规模的上界依赖较大，虽然有自适应机制但恢复周期仍可能很长；
• 主要针对全局函数，局部计算任务的可行性尚未探讨；
• 算法复杂度仍为多项式且常数因子较大，实际部署在极大规模网络时可能不可行。

---

## 202. Efficient Safety Alignment of Language Models via Latent Personality Traits

**arXiv ID:** 2607.07918 | [PDF](https://arxiv.org/pdf/2607.07918v1)

**作者:** Mohamed Amine Merzouk `[一作]` (Mila), Adam Oberman `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后训练方法——Latent Personality Alignment (LPA)，通过在潜在空间使用心理学人格条目进行对抗训练，使大型语言模型在未见过任何有害文本的情况下实现安全对齐。

**💡 创新点**

创新点在于用心理学人格条目代替具体有害提示进行潜在对抗训练，既显著减少所需数据量（仅66条条目），又保持甚至提升模型对各种攻击（如 jailbreak）的鲁棒性。

**🔧 技术方法**

采用了Latent Adversarial Training (LAT) 技术，在潜在表示层对模型进行对抗扰动，并在心理学自评提示框架下微调Qwen3‑8B。

**📊 数据集**

使用的数据集为IPIP Big Five 的 66 条负面人格条目（不含任何有害内容），训练时不使用任何有害或明确拒绝示例。

**📈 对比分析**

与传统 LAT 及基线模型比较，LPA 在 HarmBench 直接请求和五种 jailbreak 方法的攻击成功率降至接近零，同时在 MMLU、GSM8K、TruthfulQA 等标准基准上的性能与未训练模型相当；训练过程仅需几分钟，使用的训练样本比 LAT 少 75 倍。

**⚠️ 局限性**

局限性包括：目前仅在 Qwen3‑8B 上验证，缺乏针对不同模型的系统调优；评估主要依赖 LLM 判定器，缺少人类标注验证；对人格维度选择与标签一致性的依赖尚未得到完全机制化。

---

## 203. Scalable and Trustworthy Earth Observation Foundation Models

**arXiv ID:** 2607.07758 | [PDF](https://arxiv.org/pdf/2607.07758v1)

**作者:** Syed Usama Imtiaz `[一作]` (Florida State University), Nasrin Alamdari `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述了遥感领域的基础模型（RSFMs）设计与评估原则，并在两案例中展示物理感知掩码预训练与强化学习自适应站点选择的应用

**💡 创新点**

提出了针对 EO 数据特性的 RSFMs 设计与评估框架，强调光谱、时空、多模态融合与物理一致性；通过 SpecTM 与 PiCSRL 两案例展示了物理引导预训练与决策导向表示学习的创新路径

**🔧 技术方法**

采用自监督学习（掩码、对比）、多模态融合、参数高效调优、视觉-语言模型、生成模型、物理感知掩码与强化学习

**📊 数据集**

使用 Sentinel、Landsat、PACE 海洋色彩、各种高光谱与 SAR 数据，以及稀缺的微毒素现场测量和湖泊监测站观测

**📈 对比分析**

通过与 VRSBench、GEO‑Bench‑2、MMEarth‑Bench 等基准对比，发现无单一模型统治所有任务；SpecTM 在 HAB 预测中实现 R²≈0.695（当前周）/0.620（8 天预测），PiCSRL 在站点选择上 RMSE 0.153、98.4% 检测率，优于随机/贪婪/UCB 等基线

**⚠️ 局限性**

局限性包括数据代表性不足、物理一致性评估不完善、跨传感器与跨地区泛化挑战、评测缺乏统一标准、解释性与可信度仍需提升

---

## 204. 3100 Opinions on Code Review in an AI World: Building Causal Theory from Practitioner Discourse

**arXiv ID:** 2607.07980 | [PDF](https://arxiv.org/pdf/2607.07980v1)

**作者:** Shyam Agarwal `[一作]` (Carnegie Mellon University), Bogdan Vasilescu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对38,709篇灰文学文档进行LLM辅助编码，构建了26个构造、67条关系的因果理论，解释编码代理如何影响代码审查。

**💡 创新点**

① 在大规模灰文学语料上应用LLM进行主题编码与归档；② 将实践者话语转化为可检验的因果理论；③ 提供可复现的LLM驱动理论构建流程。

**🔧 技术方法**

LLM（Gemini 2.5 Flash）辅助过滤、编码与主题生成；Thematic-LM进行主题提取；自动化文本检索、聚类与因果图建模。

**📊 数据集**

2020-2026年共38,709篇公开文档（7,630网页文章、31,079 Reddit线程），其中3,100篇做深度编码。

**📈 对比分析**

通过对比不同分析选择的趋势变化验证模型的可解释性；实验显示LLM编码与人工审核一致性较高；理论的可检验性需后续实证研究验证。

**⚠️ 局限性**

数据主要为公开讨论，可能包含LLM生成文本；LLM编码与判定存在误差；从编码到理论的转化高度依赖人工判断，缺乏全自动化；仅聚焦灰文，未直接与实证数据对齐。

---

## 205. Cross-seed explainability using Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoders

**arXiv ID:** 2607.08499 | [PDF](https://arxiv.org/pdf/2607.08499v1)

**作者:** Bendegúz Váradi `[一作]` (Centre for Social Sciences), Zoltán Kmetty `[通讯]` (Eötvös Loránd University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用正交Procrustes旋转条件的联合端到端Top‑K稀疏自编码器（Joint SAE）来提取独立训练的BERT模型中的跨种子（seed）通用特征。

**💡 创新点**

创新点在于将Procrustes旋转与Top‑K稀疏约束和端到端下游优化相结合，实现对不同随机初始化模型激活空间的几何对齐，从而显著提升跨种子特征的一致性；并引入交叉稀疏损失（cross‑loss）和辅助死特征恢复机制进一步提升可解释性。

**🔧 技术方法**

核心技术包括：稀疏自编码器（Top‑K SAE）、正交Procrustes对齐、端到端下游损失（MSE+KL）、交叉稀疏损失、辅助死特征恢复。

**📊 数据集**

在三大基准数据集上评估：SST‑2（情感分类）、Stanford Politeness（三类礼貌）以及TweetEval Emotion（五类情感）。

**📈 对比分析**

与传统的独立SAE、后处理匹配、以及仅使用Procrustes或仅联合训练的基线相比，Procrustes条件的联合SAE在Top‑10、Top‑100特征的Pearson相关系数均更高，且“通用”特征比例（r≥0.70）显著提升；在所有数据集上都优于后处理匹配和单一方法。

**⚠️ 局限性**

局限性包括：仅在单一BERT模型族上验证，未探究其他模型架构或更多随机种子；解释性仍受层级选择影响；与现有Feature‑Aligned SAE或Orthogonal SAE的直接对比仍待进一步研究。

---

## 206. Unpaired Joint Distribution Modeling via Multi-Scale Image Representations

**arXiv ID:** 2607.08198 | [PDF](https://arxiv.org/pdf/2607.08198v1)

**作者:** Yihang Zou `[一作]` (Tsinghua University), Chenglong Bao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LUD‑MSR 框架，利用概率图模型和多尺度图像表示实现无配对联合分布建模，生成高质量伪配对数据用于图像去噪。

**💡 创新点**

通过构造两辅助变量并引入多尺度图像表示，理论证明域一致性与信息保留之间可取得更优权衡，显著降低联合分布逼近误差。

**🔧 技术方法**

使用潜在变量概率图模型、变分推断（ELBO）、多尺度图像表示（Wavelet+流模型）以及正则化与信息保持损失等技术。

**📊 数据集**

训练数据涵盖 SIDD‑Medium、SIDD+、DND、PolyU、CC 等真实去噪基准，以及 EMPIAR‑10025/10028/10077 等 cryo‑EM 数据集。

**📈 对比分析**

与 C2N、DeFlow、LUD‑VAE、SeNM‑VAE、DANet、Topaz 等方法对比，LUD‑MSR 在噪声合成的 AKLD/FID、基于合成数据训练的 DnCNN/DRUNet 去噪在 PSNR/SSIM 上均居前列，尤其在 cry‑EM 任务中 SNR 提升显著。

**⚠️ 局限性**

受限于辅助变量设计的线性/可逆性假设，对极低 SNR 或极复杂噪声模式的适应性有限，且在无配对条件下需额外假设，未来需进一步提高表达能力和理论严谨性。

---

## 207. Benchmark Evaluation of Feredated Learning on Multi-organ Images

**arXiv ID:** 2607.08219 | [PDF](https://arxiv.org/pdf/2607.08219v1)

**作者:** Junbin Mao `[一作]` (Central South University), Jin Liu `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为MobenFL的多器官联邦学习基准，集成了20种前沿联邦学习算法和22个医学影像数据集，系统评估算法在IID与非IID、跨器官、跨疾病、跨设备等多场景下的性能、效率和隐私保护能力。

**💡 创新点**

创新点在于：①首次构建覆盖12大关键器官、22种多模态数据的联邦学习基准；②统一整合并标准化20个最新算法，涵盖增广网络、正则化、聚合策略与分割学习等四大类别；③将算法效率、通信成本和隐私泄露风险纳入评价维度，实现多维度、可复现的综合评测。

**🔧 技术方法**

使用技术包括：联邦学习框架FedAvg、FedProx、FedBN、FedCD、FedGH、ProxyFL、FCCL、FedAu、FedDecorr、FedALA、GPFL、PGFed、TurboSVM、MOON、HarmoFL、SplitFed、SplitAvg、FLOP 等；训练采用ResNet18、Adam优化器、Dirichlet分布模拟非IID；同时进行理论时间复杂度分析和实际收敛时间评估。

**📊 数据集**

所用数据集共22个，覆盖12器官（脑、眼、口腔、乳腺、肺、腹部、肾脏、结肠、皮肤、膝关节、血液等），多模态包括MRI、CT、X‑ray、病理切片、彩色面部图像和手机拍摄皮肤图像，分别来自ADNI、Retinopathy、BRACS、ChestXray、HAM10000等公开数据源。

**📈 对比分析**

在基准上对20种算法进行对比，评测指标包括分类准确率/召回率/AUC、收敛轮数、训练时长和通信效率。实验表明：HarmoFL、SplitFed、PGFed在多数任务中表现最佳，正则化驱动类算法在非IID环境下易退化；FedProx、FedBN在数据分布偏移时相对稳定；部分算法如FedNP、FedALA在不同场景下存在显著性能波动。

**⚠️ 局限性**

局限性包括：部分算法计算/通信开销较大，实际部署成本高；对高异构数据敏感的算法仍需改进；评测仅聚焦分类任务，缺乏分割、检测等其他医学影像任务；实验基于公开数据，真实医院数据隐私和伦理限制未完全覆盖；未对算法的长期安全性（如梯度泄露）做深入分析。

---

## 208. SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation

**arXiv ID:** 2607.08161 | [PDF](https://arxiv.org/pdf/2607.08161v1)

**作者:** Wangyu Wu `[一作]` (University of Liverpool), Zhenhong Chen `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过LLM指导的知识蒸馏与合成数据，训练小型语言模型以实现高效文本到SQL转换。

**💡 创新点**

将大型语言模型作为教师生成高质量结构化监督，结合参数高效微调和领域适应，显著缩小小模型与大模型性能差距。

**🔧 技术方法**

使用LLM合成数据、LoRA参数高效微调、提示工程、数据过滤与自评评分等技术。

**📊 数据集**

主要使用WikiSQL数据集。

**📈 对比分析**

与GPT‑4o、大型Text‑to‑SQL基线以及原生小模型对比，SQuaD‑SQL在WikiSQL测试集上达86.9%执行准确率，接近LLM且显著降低资源消耗。

**⚠️ 局限性**

仍依赖大模型生成的合成数据，且对极其复杂或稀缺逻辑查询的鲁棒性有限。

---

## 209. path_boost: A Python Package for Interpretable Graph-Level Prediction using Path-Based Gradient Boosting

**arXiv ID:** 2607.07935 | [PDF](https://arxiv.org/pdf/2607.07935v1)

**作者:** Claudio Meggio `[一作]` (University of Oslo), Riccardo De Bin `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个名为 PathBoost 的 Python 包，实现基于路径的梯度提升，用于可解释的图级预测，支持回归和二分类，能够自动发现并组合重要的标签路径。

**💡 创新点**

创新点包括：① 路径选择与路径拟合分离的模块化设计，使得路径搜索仅依赖结构频数；② 采用惰性扩展策略，仅在路径被选中时才生成后继路径，显著降低了组合爆炸；③ 支持多锚点并行训练以及可选的叶子优化（tree‑boost）模式；④ 引入绝对与相对两种变量重要性度量，并可对相关性进行校正，提升解释性。

**🔧 技术方法**

核心技术：梯度提升框架、路径选择器（默认决策树桩）、路径基学习器（默认决策树回归或可替换），使用 NetworkX 处理图结构，支持多线程并行；同时实现早停、交叉验证以及模型持久化。

**📊 数据集**

实验数据集包括：tmQMg（过渡金属配合物），ESOL、FreeSolv（小分子溶解度/自由能），QM9（大规模小分子量子性质）以及额外的 TUDataset 基准。每个任务均以分子图形式给出，节点属性为原子信息，边属性为键类型等。

**📈 对比分析**

与 Graph Isomorphism Network（GINE）和 Weisfeiler‑Leman + SVR 进行对比。PathBoost 在 5/6 任务（ESOL、FreeSolv、tmQMg 的三个目标）上获得最佳或接近最佳性能，且训练时间通常低于 GINE、接近或略高于 WL+SVR；在大型均质数据集 QM9 上 GINE 仍占优势，但 PathBoost 仍保持可接受的精度。整体而言，PathBoost 在小样本和多锚点场景中表现尤为突出。

**⚠️ 局限性**

局限性：① 需要至少一个离散锚点属性；② 路径长度是关键超参数，过短会缺失长程信息，过长会导致 EBM 维度膨胀与过拟合；③ 对于非常大或高连通度的图，路径搜索即使惰性也可能耗时；④ 与 GNN 相比，PathBoost 在处理具有丰富本地化信息的大型有机分子时表现逊色；⑤ 解释性仅限于路径结构，无法捕获更细粒度的节点/边特征交互。

---

## 210. ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions

**arXiv ID:** 2607.08297 | [PDF](https://arxiv.org/pdf/2607.08297v1)

**作者:** Noah Jaitner `[一作]` (Charité Universitätsmedizin Berlin), Hossein S. Aghamiry `[通讯]` (Charité Universitätsmedizin Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个无需训练、不依赖GPU的细胞跟踪框架ARGUS，能够实现细胞检测与跟踪的完整流程；

**💡 创新点**

创新点在于将适应性检测、密集Farneback光流预测、逐帧线性分配与全局tracklet细化组合成无监督的两阶段关联方案；

**🔧 技术方法**

采用光学流（Farneback）、线性分配、阈值过滤、波形去噪及单层阈值分割等传统图像处理与关联技术；

**📊 数据集**

在Cell Tracking Challenge公开的四个数据集（Fluo-N2DH-SIM+、Fluo-N2DH-GOWT1、Fluo-C2DL-Huh7、PhC-C2DH-U373）上进行实验；

**📈 对比分析**

与Trackastra及CTC基准结果对比，DET从0.905到0.971、TRA从0.897到0.964，性能位于CTC平均水平以上、与顶尖方法相近，同时在无训练数据和CPU环境下实现每帧5–6秒的处理速度；

**⚠️ 局限性**

局限性包括：对长时间遮挡或多重融合事件处理有限、光流在低对比或快速变形场景下精度下降、缺乏物理约束与全局分支优化，且在重叠细胞时可能产生合并误检。

---

## 211. Tool-Making and Self-Evolving LLM Agents in Low-Latency Systems

**arXiv ID:** 2607.08010 | [PDF](https://arxiv.org/pdf/2607.08010v1)

**作者:** Kalle Kujanpää `[一作]` (Amazon), Shervin Malmasi `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套离线工具制作流水线，将标准操作程序（SOP）中的重复步骤编译成可验证、可版本化的工具，并在履行中心报警排除系统中部署使用，取代实时代码生成。

**💡 创新点**

创新点在于将LLM推理时的代码生成循环转化为一次性编译成工具的流程，利用数据收集轨迹与测试‑修复循环实现环境驱动与验证，显著提升延迟、可靠性与可审计性，同时保留对失败的回退灵活性。

**🔧 技术方法**

使用GLM-4.7/GLM-5/Qwen3/GLM-4.7 Flash等LLM进行工具生成与修复；Data‑Collector子代理收集执行轨迹；Tool‑Maker LLM生成候选工具；Reflector LLM进行错误诊断与重写；LoRA微调与主/子代理架构支持多模型、低延迟推理。

**📊 数据集**

基于44个决策节点的标注数据集（每节点100–200例，共约8k训练+8k评估），来自履行中心报警历史与专家标注，采用时间序列拆分做评估。

**📈 对比分析**

通过对比无工具、子代理写代码、子代理调用工具、主代理直接调用工具四种配置，测算p50/p99延迟、工具调用次数和错误率。工具调用将p50延迟降低42%，直接调用再降62%；错误率从2.8%降至1.8%（36%）或从1.7%降至0.8%（53%）；工具构建pass@1在GLM-4.7上达94.5%。

**⚠️ 局限性**

仅在单一应用场景验证，未证明对自由形式运行手册的泛化；仍需人工审查SOP与标注，完全自动化未达成；工具生成受模型容量和训练数据限制；SOP规格不完整时仍会出现错误。

---

## 212. SPL: Orchestrating Workflows with Declarative Deterministic-Probabilistic Composition

**arXiv ID:** 2607.07727 | [PDF](https://arxiv.org/pdf/2607.07727v1)

**作者:** Wen G. Gong `[一作]` `[通讯]` (Independent Researcher), Wen G. Gong (Independent Researcher)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的声明式语言SPL，能够在同一规范中混合概率式LLM计算与确定式符号计算，消除两者的碎片化；

**💡 创新点**

核心创新在于：①在源代码中以/SOLVER/ASSERT等语法声明计算模式并共享变量空间；②DODA原则，使单一SPL文件可在任意模型、任何后端验证器上部署；③构建可插拔的验证器阶梯（SymPy、SageMath、Lean 4）实现可调节的正确性保证；

**🔧 技术方法**

技术实现基于Python的AST解析、LLM适配器（支持14家云端模型和本地Ollama）、IPython/Kernel子系统执行、Momagrid分布式调度、SPL编译器将同一语法翻译成LangGraph/Go/TypeScript等多语言实现；

**📊 数据集**

使用了20道符号数学题（分六难度层级，前10题SymPy后10题SageMath）作为基准，共1200个实验单元；

**📈 对比分析**

通过双臂实验（Solver arm vs LLM-only arm），对比模型在已验证正确性与仅输出生成上的表现；结果显示SPL的Solver arm在多数模型上达到了85–93%的机器验证通过率，且验证成本低；

**⚠️ 局限性**

局限性包括：需要手动安装并配置不同的验证器环境（SageMath、Lean 4）；Kernel启动延迟对短任务不友好；当前实验未评估SPL在非数学领域的适用性；模型在格式输出（structured plan）上的不兼容仍是主要瓶颈。

---

## 213. Attribute Retrieving for Open-Vocabulary Endoscopic Compositional Referring Segmentation

**arXiv ID:** 2607.08397 | [PDF](https://arxiv.org/pdf/2607.08397v1)

**作者:** Shun Liu `[一作]` (Virginia Commonwealth University), David Doermann `[通讯]` (University at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了大规模的 ReferEndoscopy 端内图像分割基准，并提出了基于属性检索的 AR-ERIS 框架，用于实现开窗词条的端内分割。

**💡 创新点**

创新点在于将属性检索与频率感知特征融合相结合，利用 CLIP 的跨模态对齐，并通过高频损失与 Dice 损失提升细粒度对齐与边界保留。

**🔧 技术方法**

技术方法包括 CLIP 视觉与文本编码器、跨模态注意力、频率分离（FFT）与混合专家融合、属性检索模块 ARM，以及多任务损失（BCE、Dice、频率一致性）训练。

**📊 数据集**

使用的数据集包括 AutoLaparo、CholecSeg8k、DSAD、EndoVis-17/18、Kvasir-Instrument、RoboTool、SAR-RARP50、SISVSE、LaparoI2I 共计 10 个，形成 65,964 张图像、242,055 张掩码、1,452,330 条图像-掩码-指令三元组。

**📈 对比分析**

在 ReferEndoscopy 基准上与 EVF‑SAM、GroundedSAM、LAVT 等方法对比，AR-ERIS 在 naìve、medium、hard 指令场景下分别达 73.76%、74.80% 与 74.37% 的 mIoU，明显优于基线；在外部 SAR‑RARP50 数据集零样本测试中 mIoU 21.32%，超过 GroundedSAM 9.25%。

**⚠️ 局限性**

局限性包括对属性生成的依赖、在极端遮挡或低光照场景下仍可能出现误分割，以及对长尾类别的进一步平衡与细粒度语义表达的完善空间。

---

## 214. LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action

**arXiv ID:** 2607.08182 | [PDF](https://arxiv.org/pdf/2607.08182v1)

**作者:** Qi Lyu `[一作]` (State Key Laboratory of Robotics and Intelligent Systems), Zhi Han `[通讯]` (State Key Laboratory of Robotics and Intelligent Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于潜在环境演化的视觉-语言-动作模型 LEEVLA，利用任务相关信息聚焦视觉特征并在潜在空间中进行结构化的未来特征预测，以提升机器人执行长时序任务的性能。

**💡 创新点**

创新点在于：①漂移引导的动态优先级（DGDP），通过动态位置优先级（DPP）和语义漂移引导（SDG）自动定位任务关键视觉区域；②结构化特征流生成（SFFG），结合原型到边缘（P2P）预测和互邻域对比（MC）损失，保持潜在特征的空间拓扑和结构化演化；③将这两部分仅用于训练，推理时不增加额外开销。

**🔧 技术方法**

采用预训练视觉编码器（DinoV2/SigLIP）、大语言模型（如Llama-2 7B或MiniVLA），并在训练中加入未来特征解码器、P2P预测、MC对比损失、DPP/SDG权重等技术；损失函数融合动作回归、P2P和MC三项。

**📊 数据集**

在大规模跨机器人视觉‑语言数据集 Open X‑Embodiment、LIBERO 及 CALVIN benchmark 上进行训练与评估，测试数据覆盖多种操控场景与长时序指令。

**📈 对比分析**

与 OpenVLA、π₀、UniVLA、FlowER 等基线比较，LEEVLA 在 LIBERO、CALVIN 以及真实世界实验中均取得或领跑最高成功率（例如 LIBERO 上 98.8%/99.0%/98.6%/96.4% 等），显著提升了任务完成率与长期规划能力。

**⚠️ 局限性**

局限性包括：①仅在训练阶段加入额外模块，推理时仍需大型语言模型导致算力/延迟较高；②依赖大量预训练数据和大规模模型，训练成本高；③在极端多模态差异或未知环境中，仍可能因特征漂移不足而失去关注关键区域。

---

## 215. zkComposer: Decomposing Proof Construction to Scale zkML

**arXiv ID:** 2607.08095 | [PDF](https://arxiv.org/pdf/2607.08095v1)

**作者:** Pawan Kumar Sanjaya `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种可拆分的零知识证明框架zkComposer，使得对机器学习模型的推理推理可以被分割成多个独立子证明，从而实现并行证明。

**💡 创新点**

创新点在于通过共享掩码提交边界激活值的方式，既保证子证明之间的正确性，又保持零知识性；并将证明拆分到模型层级和输入序列两维，显著提升并行度。

**🔧 技术方法**

利用GKR协议、低阶多项式承诺（PCS）、Fiat–Shamir转换以及对卷积/Transformer层的定制化算术化；实现了子证明的并行生成与顺序执行两种模式。

**📊 数据集**

使用了AlexNet、AlexNet‑Wide、VGG16三种CNN模型，以及GPT‑2（含不同输入序列长度）作为实验数据集。

**📈 对比分析**

与基线zkCNN和zkGPT对比，证明生成时间提升最多达3.25×（CNN）和4.83×（GPT‑2层拆分），在结合层与序列拆分时可达6.84×；同时显著降低了峰值内存（GPT‑2可达8.1×），证明大小略增但对响应时间影响微乎其微。

**⚠️ 局限性**

受限于模型结构和可拆分层数，拆分粒度受限；每增加子证明会产生额外的边界承诺和开启开销，导致顺序执行时证明时间上升；验证时间与证明规模基本不变，且在某些模型中并行收益有限。

---

## 216. Physics-Guided Biomechanical Gait Adaptation for Humanoid Locomotion on Extreme Sloped Terrains

**arXiv ID:** 2607.07830 | [PDF](https://arxiv.org/pdf/2607.07830v1)

**作者:** Xuanyu Chen `[一作]` (Nanyang Technological University), Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了HumoSlope两阶段物理引导框架，实现了盲目（无感知）人形机器人在连续陡坡上的稳健行走。

**💡 创新点**

创新点在于①斜坡适应的ZMP正则化以获得坡面一致的平衡先验；②使用PCA坡面描述的Biomechanical Slope Gait Adapter在训练阶段动态调节CoM高度与步态，实现坡度条件的姿态与步态适配；整体部署时仅靠本体感知。

**🔧 技术方法**

采用物理指导的ZMP约束、PCA坡面描述、Biased Reward gating、PPO强化学习、IsaacLab仿真、Domain Randomization、Sim-to-Real迁移技术。

**📊 数据集**

在IsaacLab仿真中生成多种坡度与地形（平坦、上坡、下坡、波浪、斜坡、粗糙、条纹等）进行训练与评估，并在户外草坡、柏油坡等真实地形进行验证。

**📈 对比分析**

与Unitree RL Lab、FastTD3（本体感知）和Gallant（使用深度感知）进行对比；在35 m斜坡轨道上最高坡度36°时成功率73%，并在现实草坡最高32.1°连续通过，速度和CoM高度显著优于基线。

**⚠️ 局限性**

缺乏前视感知，无法预判突变坡度或不规则障碍，对极端地形和柔性表面适应性有限。

---

## 217. When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning

**arXiv ID:** 2607.07976 | [PDF](https://arxiv.org/pdf/2607.07976v1)

**作者:** Xiuyi Lou `[一作]` (Johns Hopkins University), Vladimir Braverman `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于tail‑risk的token级信用校准方法（TACO），在RLVR训练中对GRPO等批量策略进行改进，减少低概率尾部token的正向更新，提升推理质量。

**💡 创新点**

创新点在于首次将Positive‑Credit Contamination问题正式定义，并通过只使用token概率与局部熵计算的tail‑risk评分，柔性抑制高风险token的正向信用，从而在保持探索性的同时消除错误推理的过度强化。

**🔧 技术方法**

技术包括强化学习推理（RLVR）框架、GRPO及其PPO式代理、token概率与熵的联合度量、tail‑risk评分与soft weight调节、以及与标准GRPO相同的离散更新流程。

**📊 数据集**

使用的训练集为DAPO‑Math‑17K；评估集涵盖数学推理（AIME 2024/25、AMC 2023、MATH‑500、Minerva Math、OlympiadBench）与科学推理（MMLU‑Pro、GPQA‑Diamond）。

**📈 对比分析**

与GRPO、GRPO + Adv. Reweighting、STAPO等基线对比，在三大LLM（Qwen3‑1.7B、Qwen3‑4B、Qwen2.5‑Math‑7B）上，TACO在所有基准上平均提升≈2–5个百分点，并在长时间训练阶段保持稳定增长，显著优于基线。

**⚠️ 局限性**

局限性包括：对局部熵与概率估计的依赖，若模型不够well‑calibrated可能导致误判；超大模型或非推理任务的适用性尚未验证；对α、λ的超参数设置仍需经验调优。

---

## 218. DeltaV: Thinking with Visual State Updates in Unified Large Multimodal Models

**arXiv ID:** 2607.08434 | [PDF](https://arxiv.org/pdf/2607.08434v1)

**作者:** Pengjie Wang `[一作]` (Huazhong University of Science and Technology), Yuliang Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于视觉更新的统一大多模态模型（DeltaV），将每一步中间视觉状态从全图生成改为增量更新；

**💡 创新点**

通过TSIM Router动态分配视觉更新token并构建覆盖44个任务域的StructCoT大规模数据集，实现了显著降低视觉token量并提升推理性能；

**🔧 技术方法**

使用TSIM-Tok视觉分词器、Temporal Similarity Router、可变长度视觉更新token、Qwen3 LLM和SigLIP2视觉骨干等技术；

**📊 数据集**

利用StructCoT（1.05M样本）、Zebra-CoT以及6000万图像理解/生成样本等数据集；

**📈 对比分析**

在Zebra-CoT、StructCoT及多项外部基准上进行对比，DeltaV 2B在多模态推理上比同规模Qwen3-VL-2B高约5.9%，比更大开源模型高8.4%，同时视觉更新模式下新生成token减少55.6%，推理准确率提升3.3%；

**⚠️ 局限性**

视觉更新对细粒度感知不足；token分配仍为经验式，可能不适用于极低变化或极高细节需求的任务；模型仍需手动决定何时使用更新、完整图像或外部工具。

---

## 219. From Triggers to Emotions: A CPM-Grounded Appraisal Multi-Agent for Dynamic Emotional Evolution in Persona-Based Dialogue

**arXiv ID:** 2607.07824 | [PDF](https://arxiv.org/pdf/2607.07824v1)

**作者:** Jingyao Cai `[一作]` (Bournemouth University), Xiaosong Yang `[通讯]` (Bournemouth University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于组件过程模型（CPM）的多代理框架 CPM-MultiAgent，用来在角色扮演式对话中根据对话触发动态更新角色情感状态。

**💡 创新点**

创新点在于将 CPM 的情感评估拆分为触发分析、四维评估（相关性、含义、应对潜能、规范意义）以及情感更新与一致性检查四个专门代理；通过结构化评估实现触发驱动的情感演化，并以潜在状态形式提供可解释的情感更新。

**🔧 技术方法**

采用 GPT‑5.4（以及小型版本和 Qwen3.6）作为底层 LLM，使用 LangGraph 搭建多代理体系，包含 Trigger Analyzer、四个 CPM 评估代理、Integration 与 Critic 代理；利用组件过程模型进行情感评估和状态更新。

**📊 数据集**

使用自构造的 24 组基于角色（医疗培训、教育沟通、客户服务）对话试例，包含人物档案、场景上下文、对话历史及先前情感状态；并对每轮对话生成情感更新及解释。

**📈 对比分析**

与零/少量示例提示、Chain‑of‑Thought、Self‑Consistency、Self‑Refine 等直接推理基线以及 EQ‑Negotiator 对比；评估采用 LLM‑as‑Judge 与 103 名人工评审，指标包括情感更新正确性、触发关联、时序连贯性、角色一致性、评估推理质量与整体质量。CPM‑MultiAgent 在所有指标上均优于基线，Ablation 进一步证明每个模块均贡献显著，跨不同 LLM 体系亦保持优势。

**⚠️ 局限性**

主要限制是推理延迟高，因需多步代理协同；虽然可通过并行评估降低延迟，但在实时或语音交互场景下仍可能不满足即时响应需求；此外，过度情感适配可能导致说服性过强或不恰当的情感回应，需要人工监督。

---

## 220. Optimal Learning Rate Scaling Depends on Data in Deep Scalar Linear Networks

**arXiv ID:** 2607.07884 | [PDF](https://arxiv.org/pdf/2607.07884v1)

**作者:** Yedi Zhang `[一作]` (University College London), Andrew Saxe `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了深度标量线性网络及其残差变体在梯度下降下的学习动力学，并给出了关于学习率与网络深度关系的解析式。

**💡 创新点**

证明了最优深度相关学习率缩放不是数据无关的幂律，而是依赖于数据统计量，从而揭示了数据驱动的超参数迁移规律。

**🔧 技术方法**

采用梯度流分析、守恒律简化、以及超几何函数和Lambert W函数等特殊函数求解，得到精确时间曲线解。

**📊 数据集**

实验基于理论中使用的矩（μ_yx、μ_xx）而非具体公开数据集，侧重理论演示；若使用实际数据，可按相同统计量进行验证。

**📈 对比分析**

通过在有限步长梯度下降中对比数据依赖缩放与传统 L⁻¹ 缩放的训练损失，发现前者能在不同深度间实现更好的超参数迁移，且收敛速率在所有深度上保持线性并近似恒定。

**⚠️ 局限性**

局限性包括仅针对标量线性网络（及其残差形式）给出解析结果，可能不直接推广到非线性或宽网络；以及对数据统计量的依赖在深度极大时仍存在，实际应用需估计这些统计量。

---

## 221. Towards the Explainability of Temporal Graph Networks via Memory Backtracking and Topological Attribution

**arXiv ID:** 2607.07716 | [PDF](https://arxiv.org/pdf/2607.07716v1)

**作者:** Yazheng Liu `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为MemExplainer的解释框架，用于解释Temporal Graph Networks（TGNs）的预测结果，重点关注内存模块与邻居事件的贡献；

**💡 创新点**

创新点在于同时构建拓扑归因树（Topology Attribution Tree）和内存回溯树（Memory Backtracking Tree），实现对节点记忆随时间演化的完整追溯，并通过KL散度优化目标选择最重要事件；

**🔧 技术方法**

主要技术包括层级归因（LRP）在TGNs中的应用、图聚合函数（图求和或图注意力）、GRU/RNN更新、以及基于KL散度的事件筛选优化；

**📊 数据集**

实验使用九个真实数据集，涵盖链路预测（Wikipedia、Reddit、Enron、UCI）、节点属性预测（tgbn-trade、tgbn-genre、tgbn-reddit）以及图分类（HMDB51、Penn Action）等任务；

**📈 对比分析**

与四个基线（TGNNExplainer、TempME、GNNExplainer、PGExplainer）及其消融版本对比，MemExplainer在Fidelity_KL和Fidelity_prob两项指标上均显著优于全部基线，稀疏度低且在大多数数据集上达到统计显著性；

**⚠️ 局限性**

主要局限是计算开销较大，尤其是内存回溯树深度大时会导致运行时间和内存占用显著提升，未来可通过限制树深度或剪枝来缓解。

---

## 222. VSRo-200: A Romanian Visual Speech Recognition Dataset for Studying Supervision and Multimodal Robustness

**arXiv ID:** 2607.08112 | [PDF](https://arxiv.org/pdf/2607.08112v1)

**作者:** Iulia-Maria Udrea `[一作]` (University of Bucharest), Bogdan Alexe `[通讯]` (University of Bucharest)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个大型罗马尼亚视觉语音识别数据集 VSRo-200（200 小时播客视频），并提供 100 小时人工转录子集，对其进行监督质量、域迁移与音视融合的系统性基准评估。

**💡 创新点**

创新点包括：① 首次在低资源语言上实现大规模伪标签与人工标签并行的可控实验；② 证明伪标签在更大数据规模下可弥补人工标注的优势；③ 通过动态熵权重实现音视融合，显著提升噪声环境下的鲁棒性；④ 证明 VSRo-200 学到的视觉表征在单词识别基准 LRRo 上可迁移并大幅提升性能。

**🔧 技术方法**

使用技术包括：预训练视觉 Transformer (VTP) + Transformer 编码‑解码网络；Whisper-large 微调生成伪标签；基于熵的动态加权融合策略；Beam Search 解码；以及一系列视频预处理、说话人分离与对齐管线。

**📊 数据集**

主要数据集：VSRo-200（200h 伪标签 + 100h 人工转录）以及对比评估用的 LRRo、LRS2/3、LRW、MultiVSR 等公开数据集。

**📈 对比分析**

通过在不同数据规模（10h–200h）、被见与未见说话人、不同噪声 SNR、OOD 领域（vlogs、专业域、噪声、黑白）等多维度设置进行对比；结果显示：伪标签在 200h 时 WER 降至 48.8%（未见说话人）仅略逊于 100h 人工标注的 53.3%；在 -5 dB 高噪声下，音视融合将 WER 降至 38.9%（零样本 Whisper 为 90%），并在 LRRo 词识别任务中 Top‑1 达到 95%/72% 以上，明显超越前人基线。

**⚠️ 局限性**

局限性包括：① 数据来源单一（播客）导致视觉/音频多样性不足，难以推广至更极端环境；② 伪标签噪声仍存在，极端噪声或领域差异可能削弱性能；③ 词汇覆盖受限，对专业术语和 OOV 词的鲁棒性不足；④ 性能评估主要集中在单一模型（MultiVSR）和固定训练设置，缺乏模型多样性验证；⑤ 训练成本高，限制了可复制性和普及度；⑥ 数据涉及可识别个体，存在隐私与滥用风险。

---

## 223. 3D Reconstruction of deciduous Trees using low-cost UAV- and Crane-based Photogrammetry for Monitoring Shoot Elongation across entire Canopies

**arXiv ID:** 2607.07905 | [PDF](https://arxiv.org/pdf/2607.07905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 224. Towards Soft Robotic Exogloves for Musculoskeletal Manipulation to Reduce Pain and Spasticity

**arXiv ID:** 2607.07958 | [PDF](https://arxiv.org/pdf/2607.07958v1)

**作者:** Antonia Salluce `[一作]` (Stevens Institute of Technology), Jacqueline Libby `[通讯]` (Stevens Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一款由背面、前面指尖软气动执行器和掌面气囊组成的可定制化软机器人外骨手套，用于疼痛缓解和手部痉挛管理。

**💡 创新点**

创新点在于：①背面执行器采用手部拓扑定制的柔性气动膨胀装置；②前面执行器为超柔性可压缩设计；③掌面气囊采用SLA打印的可压缩喇叭形结构；④所有执行器可独立调压实现按摩、伸展和压缩的组合。

**🔧 技术方法**

技术包括光学摄影测量（iPhone 360° 拍摄）、Meshmixer 后处理、CAD 建模、3D 打印模具、硅胶浇注、SLA 3D 打印、有限元分析（Ansys、Neo‑Hookean/Yeoh 模型）以及机器人驱动的气动控制站。

**📊 数据集**

使用的数据集主要为自制的手部三维扫描点云（两次扫描：伸展与弯曲），以及实验室内气动压力与力学测试数据。

**📈 对比分析**

比较方法：在单一健康受试者上做实验，观察背/前执行器独立与联合充气的手指伸展、按摩效果；掌面气囊的充气导致手掌扩张。尚未与现有商业化手套或多模态系统进行系统对比，性能指标仅为实验室内压力-力响应曲线，未提供定量临床评估。

**⚠️ 局限性**

局限性：①背面执行器的拓扑定制导致基底过厚、关节处僵硬；②掌面气囊在SLA打印时壁厚不足导致压缩困难及材料脱落；③实验仅基于少量受试者的主观反馈，缺乏大规模临床验证；④缺乏长效佩戴的耐久性评估；⑤气动控制与手部解剖匹配仍需改进。

---

## 225. Diagnosing and Repairing Persona Collapse in LLM Advice

**arXiv ID:** 2607.08326 | [PDF](https://arxiv.org/pdf/2607.08326v1)

**作者:** Harsh Kumar `[一作]` (University of Toronto), Ashton Anderson `[通讯]` (University of Toronto)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM在多种情境下给出建议时是否会出现persona collapse，并尝试通过Inverse‑Process Distillation等方法修复。

**💡 创新点**

提出persona collapse概念，将建议写成两维空间（情感与代理深度），并研发逆向过程蒸馏重建情境阅读以恢复情境‑人格映射。

**🔧 技术方法**

使用多种SFT、推理提示、Persona Oracle、Inverse‑Process Distillation（基于LLM判别的H/E轴标签和前向推理教师）。

**📊 数据集**

使用约1.3k Reddit帖子以及扩充的8.2k多来源（StackExchange、CounselChat、CareerVillage）组成情境+顶级回复的训练/测试集。

**📈 对比分析**

对比三种前沿模型（GPT‑5.1、Claude Opus 4.5、Gemini 3 Pro）以及多种修复方案；恢复后 N_eff 提升至≈4、JS 降低约50%，但SFT模型在受试者评价中仍低于默认模型。

**⚠️ 局限性**

限制在于参考答案为社区投票，LLM判别可能偏见，单轮对话不代表真实建议流程，缺乏长期效益评估。

---

## 226. Who Gets Missed in the Tail? Thresholded Subgroup Underdiagnosis in Long-Tailed Chest X-ray Classification

**arXiv ID:** 2607.07717 | [PDF](https://arxiv.org/pdf/2607.07717v1)

**作者:** Ha-Hieu Pham `[一作]` (University of Science), Huy-Hieu Pham `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并验证了针对胸部X光长期尾部分类的阈值化子群误诊审核框架，评估不同子群与阈值下的漏诊率。

**💡 创新点**

将长期尾部损失、子群加权、群组鲁棒训练和阈值选择分层诊断，突出阈值决策对子群漏诊的关键作用，并提出阈值感知优化策略。

**🔧 技术方法**

使用 ConvNeXt‑Tiny 作为基础网络，结合 BCE、ASL、CB‑ASL、GroupDRO 训练；对尾部类别采用精度下限 τ=0.05 的阈值优化；通过 95% 置信区间和配对 Bootstrap 对结果进行统计检验。

**📊 数据集**

使用 VinDr‑CXR（1500 张测试图像，性别、年龄子群）和 MIMIC‑CXR/CXR‑LT（78,946 张测试图像，性别、年龄、种族、保险子群）两大公开数据集。

**📈 对比分析**

与基线 BCE、ASL、CB‑ASL、GroupDRO 进行对比；在 VinDr‑CXR 上，阈值优化将尾部 FNR 从 0.665 降至 0.269，worst‑group FNR 同样大幅下降；在 MIMIC‑CXR/CXR‑LT 上也显著降低 FNR，但绝对值仍偏高；宏观 mAP 波动有限。

**⚠️ 局限性**

局限性包括尾部类别样本稀缺导致置信区间宽大；子群元数据缺失或不完整；阈值设定需结合临床工作流而非通用；仅使用单一网络架构，缺乏因果或跨体系验证。

---

## 227. Understanding Axes of Difficulty For Long Context Tasks Via PredicateLongBench

**arXiv ID:** 2607.08284 | [PDF](https://arxiv.org/pdf/2607.08284v1)

**作者:** Siddhartha Jain `[一作]` (NVIDIA), Ameya Velingker `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新型长上下文评测基准，任务为在词列表中检索满足给定谓词（前缀/后缀/包含或词典序）且最长的连续子序列，且不依赖LLM生成或评判；

**💡 创新点**

系统化定义并探索多轴难度（计算复杂度、反作弊诱导、搜索空间、量化复杂度、上下文结构），并提供合成与真实文档两套生成管道，强调任务算法简单、易实现且可扩展；

**🔧 技术方法**

构造词列表、定义一元/二元谓词，使用随机字符串与LongBench v2文档进行样本生成；在多款闭源与开源LLM（GPT‑5.4、Gemini‑3.1、Opus‑4.6、Minimax‑2.7、GLM‑5.1、Qwen‑3.5等）上实验，调节推理预算与最大输出 token；

**📊 数据集**

使用自定义合成随机字符串（长度8/12）以及过滤后的LongBench v2文档（共93篇）作为真实词列表；

**📈 对比分析**

在100/93个实例上统计准确率，比较不同难度轴下各模型表现；结果显示闭源模型（尤其Opus‑4.6）在基线任务上准确率高，但随着难度提升（如散布的诱导词、量化硬化、搜索空间增大、结构化上下文等）准确率急剧下降，开源模型在大多数变体几乎为0；

**⚠️ 局限性**

未探索更高阶谓词（k‑ary>2）；量化复杂度实验仅在单一满足序列；对诱导词变体与排列未做更细致分析；搜索空间与token计数混合效应不完整；实验仅基于单一tokenizer（Qwen），缺乏跨 tokenizer 验证。

---

## 228. Minimum Edge-Outerplanar Embeddings are Polynomial-Time Computable

**arXiv ID:** 2607.08110 | [PDF](https://arxiv.org/pdf/2607.08110v1)

**作者:** Hantao Yu `[一作]` `[通讯]` (Columbia University), Hantao Yu (Columbia University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了平面图的最小边外层数可以在多项式时间内计算，解答了Bentz（2009）提出的开放问题。

**💡 创新点**

核心创新是通过构造辅助图H，将最小边外层数问题归约为已知可多项式求解的最小面深度问题，并证明两者最优值相等。

**🔧 技术方法**

技术手段主要包括平面嵌入的边剥离过程、面邻接图距离、构造拆分边并附加三角标记的辅助图、以及对辅助图使用已知的O(n⁴)最小面深度嵌入算法。

**📊 数据集**

该工作为理论证明性质，未使用任何实验数据集。

**📈 对比分析**

与先前仅能在指数时间完成的求解方式相比，本文提供了O(|V|⁴)（或更低）时间复杂度的算法，理论上实现了多项式时间的最优求解。

**⚠️ 局限性**

局限性在于仅适用于无环（loopless）的平面图；对含环的图不适用；此外，算法实现依赖于已有的最小面深度嵌入实现，未给出具体实现细节。

---

## 229. Does online sustainability communication shape public discourse? Insights from six years of tenant-housing provider interactions

**arXiv ID:** 2607.08437 | [PDF](https://arxiv.org/pdf/2607.08437v1)

**作者:** Shray Juneja `[一作]` (Eindhoven University of Technology), Ioulia V. Ossokina `[通讯]` (Eindhoven University of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个数据驱动框架，对公共住房协会的社交媒体评论按语义相似度、情感和沟通意图进行三维表征，并通过聚类识别出六种对话类型。

**💡 创新点**

创新点在于同时量化并联合建模语义相关度、情感倾向和沟通意图，突破传统单维度评估，得到多维对话类型，可揭示公共机构与公众互动的结构化模式。

**🔧 技术方法**

使用了Transformer模型（如BERT、RobBERT、mDeBERTa）进行意图和情感分类，句子嵌入模型（paraphrase‑multilingual‑MiniLM）计算语义相似度，并用k‑means聚类。

**📊 数据集**

数据集为2018‑2023年荷兰公共住房协会在Facebook公开页面上发布的792条帖子及其3,197条租户评论，包含对可持续性议题的讨论。

**📈 对比分析**

通过多类别逻辑回归检验沟通设计与组织特征对对话类型的影响，并与传统的点赞、评论量等聚合指标比较，发现多维框架能揭示情感、意图与主题的互动，模型准确率分别为意图0.85、情感0.72。

**⚠️ 局限性**

局限包括观察性设计导致缺乏因果推断，单一平台和单一政策领域可能限制普适性，特征二值化可能掩盖更细致关系，聚类结果仍需主观解释。

---

## 230. UAV-OVVIS: Unmanned Aerial Vehicles Also Need Open-Vocabulary Video Instance Segmentation

**arXiv ID:** 2607.08075 | [PDF](https://arxiv.org/pdf/2607.08075v1)

**作者:** Mingyu Dou `[一作]` (Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences), Zhe Sun `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了UAV-OVVIS任务并实现了训练‑free 的AeroTrack框架，能够根据任意文本查询在UAV视频中输出全局一致的实例分割轨迹。

**💡 创新点**

通过将任务拆分为周期性开词表检测、短段掩码传播和生命周期 ID 关联，AeroTrack 在不需要额外训练的前提下实现了长序列、密集目标的高质量分割与身份保持。

**🔧 技术方法**

复用 YOLO‑World/ Grounding DINO/SAM3 等视觉基础模型，结合周期性检测、短段传播以及无监督的 LIA 模块实现任务完成。

**📊 数据集**

构建了 AeroVIS 数据集（9 类、8,279 条实例轨迹）并在 YouTube‑VIS 2019/2021 与 LV‑VIS 三个公开 VIS 基准上进行评估。

**📈 对比分析**

在 AeroVIS 上与 DEVA、GLEE、OV2Seg、CLIP‑VIS 等通用 OV‑VIS 方法对比，AeroTrack 的 HOTA 取得约+30% 的提升，并在 YouTube‑VIS 及 LV‑VIS 上保持接近或超越现有最佳结果。

**⚠️ 局限性**

仍依赖大型预训练模型，对检测刷新间隔与阈值设置敏感，在极高密度目标或超长视频的实时推理和能耗上仍有改进空间。

---

## 231. Closed-Loop Dynamic Validator Node Scaling in Private Substrate Blockchains Using Takagi-Sugeno Fuzzy Inference

**arXiv ID:** 2607.07901 | [PDF](https://arxiv.org/pdf/2607.07901v1)

**作者:** Thandile Nododile `[一作]` (University of Western Cape), Clement N. Nyirenda `[通讯]` (University of Western Cape)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了基于Takagi‑Sugeno模糊推理的闭环动态验证节点扩缩控制器，用于私有 Substrate 区块链的自动验证节点管理。

**💡 创新点**

创新点在于：①经验校准的三角隶属函数与完整 27 条规则的 TS 模糊系统，实现连续效率评估与缩放决策；②将隶属函数锚定至观测运行区间，提升可迁移性；③闭环实验同时演示扩容与缩容两方向并收敛至同一平衡点。

**🔧 技术方法**

采用 Takagi‑Sugeno 模糊推理、Product t‑norm 聚合、Python+JSON‑RPC 监控、Substrate AURA/GRANDPA 共识、IPFS 以及统计分析（ANOVA、Welch t 检验、Cohen d）等技术。

**📊 数据集**

使用昆士兰政府公开的智能水表实时数据（CSV），通过 SHA‑256 哈希后上链。

**📈 对比分析**

与三种阈值基线（保守、中等、激进）在同一七阶段负载下对比，评估指标包括决策翻转次数、块生成时间和效率。TS 模糊控制器的决策翻转仅为 2/4 次，块生成时间与阈值控制相当或略优，效率提升显著；阈值控制器翻转次数显著增加。

**⚠️ 局限性**

局限性在于：仅在 Substrate AURA 共识环境下验证，未评估其他共识或存储体系的适用性；当前模糊函数参数为静态经验校准，缺乏在线自适应机制；未在网络延迟、节点故障等实际网络异构条件下进行实验。

---

## 232. Alignment Plausibility: A New Standard for Assuring AI in Healthcare

**arXiv ID:** 2607.07766 | [PDF](https://arxiv.org/pdf/2607.07766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 233. Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions

**arXiv ID:** 2607.08164 | [PDF](https://arxiv.org/pdf/2607.08164v1)

**作者:** Sarthak Kumar Maharana `[一作]` (University of Texas at Dallas), Yunhui Guo `[通讯]` (University of Texas at Dallas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了持续测试时适应（CTTA）领域的研究现状，系统地定义了问题、梳理了评估协议、并提出了分层分类法。

**💡 创新点**

首次将CTTA方法按优化策略、参数效率和架构改造三大类进行层次化归纳，同时对不同域迁移模式和挑战做了细致分析。

**🔧 技术方法**

采用自监督目标（熵最小化、伪标签、参数恢复）、归一化层更新、可选参数微调、教师-学生框架、适配器、视觉提示和遮蔽建模等技术。

**📊 数据集**

基准数据集主要为CIFAR‑10‑C、CIFAR‑100‑C和ImageNet‑C，评估环境涵盖CoTTA、DPCore、RoTTA等多种持续分布设定。

**📈 对比分析**

对多种方法在统一实验协议下进行了对比实验，展示了不同策略在连续域漂移、混合分布与临时不平衡等条件下的性能提升与瓶颈。

**⚠️ 局限性**

方法普遍依赖BatchNorm或特定归一化层，受限于小批量、缺乏语义迁移处理、以及对真实动态环境评估不足，导致在极端或多样化分布变化下的泛化仍不理想。

---

## 234. Can We Trust LLM's Logic? Quantifying Uncertainty, Coherence, and Robustness via a Graph-Based Framework

**arXiv ID:** 2607.08017 | [PDF](https://arxiv.org/pdf/2607.08017v1)

**作者:** Riccardo Revalor `[一作]` (University of Illinois Chicago), Debjit Pal `[通讯]` (University of Illinois Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于图的推理框架，将模型生成的链式推理转化为因果DAG，并通过语义结构GED度量评估推理不确定性与一致性。

**💡 创新点**

创新点在于将推理路径建模为图并提出Graph Reasoning Coherence Score (GRCS) 与 Graph Self‑Consistency (GSC)，前者将语义与结构一致性融合为不确定性指标，后者通过中位图（medoid）筛选最具结构支持的推理路径，揭示“负载‑bearing”路径作用。

**🔧 技术方法**

采用因果事实分解、DAG构建、语义结构GED (SS‑GED)、GRCS 计算、GSC 解码、对抗性中位图消融以及 embedding Transformer 进行语义嵌入。

**📊 数据集**

在五大推理任务上评估：GSM8K、BoolQ、StrategyQA、MedQA USMLE、GPQA Diamond，覆盖从小学算术到医学、科学等不同难度级别。

**📈 对比分析**

与 CoTA、Topo‑UQ、dispersion/entropy 等基准比较，GRCS 在 8/15 组配置中与可信度负相关最强（93.3% 负相关）；GSC 在小模型上显著降低“幸运猜”误差，提升推理可信度，较大模型保持或略升准确率；对抗消融验证中位图对准确率与信任度的负相关性。

**⚠️ 局限性**

局限性包括 O(N²·m³) 的计算复杂度、在更大、更强模型上的效果相对减弱、仅限于最多 6 步的短推理场景，难以直接扩展到长链、超大规模任务。

---

## 235. Controllability-Aware Adversarial Examples Against LLM-Based Network Traffic Classifiers

**arXiv ID:** 2607.07739 | [PDF](https://arxiv.org/pdf/2607.07739v1)

**作者:** Zhenpeng Li `[一作]` `[通讯]` (Guangzhou Health Science), Zhenpeng Li (Guangzhou Health Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在源端可控特征约束下，利用黑盒转移攻击评估大语言模型（LLM）在网络流量检测中的鲁棒性

**💡 创新点**

提出了DC/IC/UC三类可控性特征分类，设计了控制性黑盒转移框架，并发现不同数据集和目标架构下LLM的脆弱性表现不一

**🔧 技术方法**

使用XGBoost作为代替模型生成FD‑PGD、贪心坐标与NES等攻击；对LLM进行LoRA微调；采用vLLM进行推理

**📊 数据集**

在NSL‑KDD、UNSW‑NB15、CIC‑IDS‑2018、RT‑IoT2022、HIKARI‑2021五个公开数据集上进行实验

**📈 对比分析**

将LLM攻击成功率（ASR）与传统ML基准（LightGBM与DNN‑IDS）进行分解比较；结果显示LLM对CIC‑IDS‑2018和RT‑IoT2022更易被攻击，NSL‑KDD与UNSW‑NB15相当，而在HIKARI‑2021上LLM更鲁棒；梯度/得分基攻击优于贪心攻击，跨模型转移稳定

**⚠️ 局限性**

限制：仅满足特征层的可控性约束，未保证完整的协议级可执行性；特征可控性划分依赖域判断；对令牌化与架构影响的机制尚未完全分离

---

## 236. Forensic Schema for Psychological Manipulation in Cyber Fraud: LLM-Driven Victim Reports Analysis

**arXiv ID:** 2607.07751 | [PDF](https://arxiv.org/pdf/2607.07751v1)

**作者:** Zikai Alex Wen `[一作]` (University of Washington), Yan Bai `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套4类、35问的取证架构，用LLM对近11,000条网络诈骗受害者报告进行结构化标注，并分析了心理操纵特征与诈骗类型的关联与信息缺口。

**💡 创新点**

首次将说服理论与传统取证元数据相结合，提出包含心理操纵指标与区块链证据的完整取证框架，并验证了LLM在取证文本中的可用性。

**🔧 技术方法**

使用Claude Haiku 4.5 LLM作为注释工具，结合分层（Tier）注释策略；对标注结果进行人工交叉验证与Kappa一致性评估。

**📊 数据集**

数据来源于8个公开的诈骗报告集合（包括BBB、California DFPI、WISCONSIN DFI、CPS 等）共计10,994条受害者报告。

**📈 对比分析**

与人工双标注对比，LLM-人类Kappa平均为0.69（与人工间0.68相近），说明LLM标注可靠；统计检验显示各心理操纵指标与诈骗类型显著相关，Cramér's V最高达0.79，表明模型可用于诈骗类型划分。

**⚠️ 局限性**

局限性包括：仅依据受害者自述，可能忽略未被提及的操纵手段；LLM对低频或隐性指令的识别受限；数据主要来自公开渠道，可能存在报告方式偏差；区块链证据缺失严重，需改进受害者信息采集流程。

---

## 237. Persuasion Attacks Can Decrease Effectiveness of CoT Monitoring

**arXiv ID:** 2607.08066 | [PDF](https://arxiv.org/pdf/2607.08066v1)

**作者:** Jennifer Za `[一作]` (LASR Labs), Victoria Krakovna `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了链式思维（CoT）监控在面对主动说服型攻击时的脆弱性，并提出了跨模型事实核查框架来降低代理被说服的风险。

**💡 创新点**

创新点包括：①系统地将CoT视为攻击向量，证明其可被利用来提升代理说服监控的成功率；②比较不同监控设置（仅理由、CoT、与事实核查）并揭示跨模型事实核查最有效；③提出并验证了同一模型与不同模型的事实核查对比。

**🔧 技术方法**

技术手段主要是：大型语言模型（Gemini、GPT‑4.1、Claude 3.7 Sonnet、Llama 4）作为代理、监控和事实核查者；红队对抗实验设计；多轮交互回合；基于自然语言的理由和CoT推理输出；监控与事实核查的多种组合实现。

**📊 数据集**

数据集：40 个现实任务，涵盖软件工程、客服、金融分析与社交媒体内容，分别设置善意与恶意（隐藏违规）两种变体，构成实验基准。

**📈 对比分析**

比较方法：将不同监控设置下的批准率与善意、恶意基线进行对照。实验结果显示：CoT 监控的批准率比仅理由高约 9.5%；跨模型事实核查可将违规批准率从约 30% 降至 6%，比同模型核查降低率更显著；整体说服成功率在 8.7%–47% 之间，取决于模型和设置。

**⚠️ 局限性**

局限性：仅评估代理的理由输出而未执行真实行为；任务覆盖有限（40 个），未涵盖更长远或多代理情景；实验只使用大型模型，未考察小型或专门微调模型；未加入可执行工具调用；缺乏真实世界反馈循环；政策与评估框架固定，可能不完全代表实际部署。

---

## 238. Classical versus Deep Mirror-Symmetry Scoring: A Benchmark of Thirteen Methods

**arXiv ID:** 2607.08379 | [PDF](https://arxiv.org/pdf/2607.08379v1)

**作者:** Maximilian Woehrer `[一作]` `[通讯]` (University of Vienna), Maximilian Woehrer (University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文建立统一的镜像对称评分基准，提供13种从像素到预训练CNN特征的评分器，并在多种公开数据集上统一评估其对称性判别性能。

**💡 创新点**

创新之处在于首次统一对镜像对称评分方法的评价协议与实现，提供开源库；通过对同一数据集的显著性比较，揭示最佳对称性特征来源于中尺度无符号定向信息。

**🔧 技术方法**

使用统一的 representation–comparison–aggregation 模板，对13种方法（像素、梯度、频域、HOG、PHOG、Gabor、PatchNN、深度CNN等）进行镜像对称评分；评估指标为chance-anchored 2·AUC-1，统计采用配对自助法并进行 Holm 校正。

**📊 数据集**

使用四个单轴数据集（symComp17_s、NYU_s、PIX2PER-art、PIX2PER-nat）以及五个多轴数据集（symComp13_m、NYU_m、DENDI、PIX2PER-art、PIX2PER-nat），共计约3950个轴单位。

**📈 对比分析**

方法通过对比每个轴的真轴与12个扰动负轴的得分来计算判别技能，结果显示 DeepFeat（0.83）和 AlexNet-C2（0.81）排名第一，HOG（0.80）紧随其后；DeepFeat比 HOG 高约0.03，但在 CPU 上慢约300倍；多轴协议下排名保持一致。

**⚠️ 局限性**

局限性包括：仅针对自然图像和艺术品，医学等专用域可能表现不同；使用冻结预训练网络未探索任务专用头；裁剪尺寸对得分的潜在影响需进一步验证；在极细扰动下标注误差可能导致评估偏差。

---

## 239. A Study Of Skew-Polycyclic Codes Over A Non-Chain Ring

**arXiv ID:** 2607.08304 | [PDF](https://arxiv.org/pdf/2607.08304v1)

**作者:** Seema Antil `[一作]` (Indian Institute of Technology Ropar), Sugandha Maheshwary `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了与中心多项式f(x)相关的长度为lj的偏循环编码，描述了这些编码的特性，确定了自由偏循环编码的秩。

**💡 创新点**

扩展了对有限非链环R_u^2,v^2,p^m上偏循环编码的研究，特别是对偏(λ,Θ)-常循环编码的分类和结构进行了深入分析。

**🔧 技术方法**

使用了偏多项式环R_u^2,v^2,p^m[x;Θ]，其中Θ是R_u^2,v^2,p^m的一个自同构。

**📊 数据集**

使用了有限域𝔽_p^m和环R_u^2,v^2,p^m的相关数据集。

**📈 对比分析**

通过与已有的偏循环编码和常循环编码的研究进行比较，展示了新方法在编码结构和性能上的优势，特别是在错误检测和纠正方面。

**⚠️ 局限性**

研究中未能完全分类所有长度的偏循环编码，且对某些特定情况下的编码性能仍需进一步验证。

---

## 240. On the Limitations of Non-GPU AI Accelerators for Large-Model Inference: A Field Study of MoE and Multimodal Serving on Huawei Ascend

**arXiv ID:** 2607.08215 | [PDF](https://arxiv.org/pdf/2607.08215v1)

**作者:** Zheng Yu `[一作]` `[通讯]` (Chinese University of Hong Kong), Zheng Yu (Chinese University of Hong Kong)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在16卡Ascend 910上部署大规模MoE语言模型和多模态视听模型，验证其可行性并记录所需补丁与运行时操作。

**💡 创新点**

首次系统性地梳理Ascend 910与vLLM-Ascend在大模型推理中的平台限制，并给出可复现的迁移经验与跨厂商通用策略。

**🔧 技术方法**

使用CANN软件栈、vLLM-Ascend插件、DeepSeek-V4-Flash（W8A8）和DeepSeek-V4-Flash-Vision（BF16）等技术，结合多模态融合与稀疏注意力。

**📊 数据集**

利用DeepSeek-V4-Flash模型以及MMMU/MMMU-Pro数据集，对20个前沿LLM进行价值对齐评估。

**📈 对比分析**

通过与CUDA vLLM基准对比，Ascend实现保持正确性，但吞吐量存在共振峰、并发极限与冷启动耗时等差异，整体性能低于CUDA。

**⚠️ 局限性**

主要限制包括软件栈不成熟、特性覆盖缺失、并行轴脆弱、内核级错误、图编译不成熟、先进特性不稳定、性能/可扩展性上限、运维可观测性弱以及生态碎片化等。

---

## 241. Deep Reinforcement Learning-Empowered Wireless Sensor Networking for 6G Closed-Loop Controls

**arXiv ID:** 2607.08272 | [PDF](https://arxiv.org/pdf/2607.08272v1)

**作者:** Chengleyang Lei `[一作]` (Tsinghua University), Shi Jin `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在UAV边缘信息中心（EIH）辅助的SC³闭环控制系统中，通过优化多传感器的带宽分配来最小化无限时限线性二次调节（LQR）控制成本。

**💡 创新点**

创新点在于：①从闭环控制性能角度出发，将带宽分配与LQR成本建立跨层关系；②利用互信息理论把有限数据率映射为等效高斯失真噪声；③将控制过程建模为POMDP，采用深度强化学习（PPO）学习连续带宽分配策略；④设计三路径特征提取网络和训练稳定化机制，提升学习效果。

**🔧 技术方法**

技术手段包括：LQR与卡尔曼滤波的LQG控制框架；互信息与有限区块长度率模型；深度强化学习中的PPO算法；POMDP建模；三路径特征提取网络与残差块。

**📊 数据集**

使用基于随机生成的控制系统状态矩阵、传感器位置、无线信道参数等仿真环境；未使用公开真实数据集。

**📈 对比分析**

与两种基线（感知优化分配与通信吞吐优化分配）以及SAC、TD3等DRL算法进行对比。结果表明，PPO方案在多种资源约束（带宽、传输时延、发射功率、传感噪声）下均能获得最低的平均LQR成本，尤其在资源紧缺时优势更为明显。

**⚠️ 局限性**

局限性包括：①失真噪声假设为高斯且基于互信息近似，可能与实际量化噪声不完全一致；②假设下行链路可靠，忽略控制指令的误码与延迟；③带宽可连续分配的假设，实际系统需考虑离散比特/量化级；④实验仅在仿真环境中验证，缺乏真实部署的验证。

---

## 242. Holographic Neural PCFG for Unsupervised Parsing

**arXiv ID:** 2607.08063 | [PDF](https://arxiv.org/pdf/2607.08063v1)

**作者:** Ryosuke Yamaki `[一作]` (Ritsumeikan University), Tadahiro Taniguchi `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新型无监督短语结构语法（PCFG）模型 Hol‑PCFG，通过把规则概率建模为词符号嵌入之间的同伦（holographic）算子，实现在仅凭原始文本的情况下高效诱导句法树。

**💡 创新点**

创新点在于：①把 PCFG 规则概率的计算改为闭式的同伦关联评分（基于圆形相关运算）而非黑盒 MLP；②将所有嵌入约束在高维圆环（torus）上，显著降低参数量并提升训练稳定性；③通过该结构天然编码父子方向性与左右子树的非对称性，从而提升无监督语法学习质量。

**🔧 技术方法**

核心技术包括：同伦嵌入（HolE）算子、圆形相关（circular correlation）与圆形卷积（circular convolution）用于规则评分、频域频谱约束的圆环嵌入、使用 FlashInside GPU 实现的 inside 算法、以及 SemInfo 语义信息目标函数。

**📊 数据集**

在多语言评测中使用：Penn Treebank（英语）、Chinese Treebank、SPMRL 多语言树库（法语、德语、巴斯克语、希伯来语、匈牙利语、韩语、波兰语、瑞典语）、Keyaki Treebank（日语），以及 kaomoji 表情符号数据集做非语言实验。

**📈 对比分析**

与同类 N‑PCFG 系列（N‑PCFG、C‑PCFG、TN‑PCFG、SN‑PCFG、SC‑PCFG 等）以及非显式语法方法（PRPN、ON‑LSTM、URNNG、DIORA 等）对比，Hol‑PCFG 在最大似然训练下与 SN‑PCFG 相当，在 SemInfo 目标下在 6 语言中均取得最优或接近最优 F1；参数量比 SN‑PCFG 下降 99.94%，且训练收敛更稳定。

**⚠️ 局限性**

局限性：①仍采用 SN‑PCFG 的左子/右子独立假设，未建模同胞子间相关；②字符级实验仅在日语上验证，是否适用于其他形态学或书写体系尚待验证；③在极端复杂句子或低资源语言中，模型对句子级潜变量的缺乏可能限制进一步提升。

---

## 243. ProsMAE: Multi-Source MAE Pretraining for ISUP Grade Classification

**arXiv ID:** 2607.08162 | [PDF](https://arxiv.org/pdf/2607.08162v1)

**作者:** Anna Jung `[一作]` (Seoul National University), Nam-Joon Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在多源医学影像预训练框架ProsMAE中，将前列腺癌（PANDA）、淋巴结转移（CAMELYON17）和乳腺癌亚型（BRACS）三大公开病理切片进行高比例遮掩的MAE预训练，然后将冻结的编码器与线性分类头组合（ProsCLS）完成ISUP分级任务。

**💡 创新点**

创新点在于采用多源（跨癌种）MAE预训练以提升对不同扫描仪、染色差异等域差异的鲁棒性，并实现仅需冻结编码器、线性分类头的低算力部署方案。

**🔧 技术方法**

技术上使用ViT-MAE（Masked AutoEncoder）进行自监督预训练，采用高遮掩率0.75，后期冻结编码器，利用均值池化+线性分类头进行下游ISUP分级，并在噪声注入 ablation 中测试高斯噪声对鲁棒性的影响。

**📊 数据集**

数据集包括用于预训练的PANDA、CAMELYON17、BRACS三大公开数据集；下游评估采用PANDA的离散分层切分，241张预训练、82张训练、80张验证，切片划分为1024×1024、512×512等尺寸。

**📈 对比分析**

与单源Vanilla MAE对比，ProsMAE在QWK从0.4084提升至0.4736（提升0.0652），准确率和宏F1也略有提升；在重建性能对比中，ProsMAE在LPIPS、SSIM、PSNR上均优于AE、VAE和单源MAE。

**⚠️ 局限性**

局限性包括仅在单一PANDA队列和单一分层切分验证，分层拆分对结果影响显著；未在外部独立队列上评估泛化；噪声注入实验未表现出显著优势；需进一步重复验证和跨队列评估。

---

## 244. The Context Access Divide: Interaction-Level Architecture as a Complementary Dimension of Agentic Inequality

**arXiv ID:** 2607.08495 | [PDF](https://arxiv.org/pdf/2607.08495v1)

**作者:** Masahiro Fujita `[一作]` `[通讯]` (Kansai University), Masahiro Fujita (Kansai University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Context Access Divide (CAD) 并将其作为 AI 代理不平等分析的一个新的交互层面维度。

**💡 创新点**

创新点在于把“上下文检索方式”从传统的可访问性/质量/数量维度外，定位为交互架构层面的“上下文可访问性”，并通过概率模型阐释其对知识工作者生产力的阈值效应。

**🔧 技术方法**

利用认知心理学中的 fan effect 推导的概率模型，讨论检索增强生成 (RAG) 与 Model Context Protocol (MCP) 等技术架构，阐释三种上下文提供模式（MAM、Walled DCRM、Open DCRM）。

**📊 数据集**

本研究为概念性分析，未使用公开数据集；所用模型参数为示例性设定。

**📈 对比分析**

通过模拟图展示不同架构在语料库规模和任务组合性下的成功概率，对比显示 Open DCRM 在大规模语料下显著优于 MAM；无实际实验或性能评测。

**⚠️ 局限性**

局限性包括缺乏实证验证、模型参数仅为示例、架构划分在现实中不够尖锐、未覆盖所有可能的检索架构及其治理影响。

---

## 245. TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models

**arXiv ID:** 2607.08339 | [PDF](https://arxiv.org/pdf/2607.08339v1)

**作者:** Giuliano Gorgone `[一作]` (University of Amsterdam), Fausto Carcassi `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用线性探针逐层检查 SantaCoder-1.1B 与 CodeLlama-7B 预训练代码模型的残差流，探究其内部是否编码了类型信息，尤其是跨 Java 与 Python 的类型表示，并通过对抗性重命名评估词法干扰下的鲁棒性。

**💡 创新点**

首次证明：①即使仅在无类型 Python 示例上训练，模型也能在残差流中捕获跨语言可解的类型语义；②跨语言零样本迁移显示类型表示在两种语言间共享；③对抗性重命名揭示类型信息对词法提示的部分稳健性。

**🔧 技术方法**

采用逐层线性探针（无特征归一化）训练分类器；利用 selectivity（任务准确率减去随机对照）衡量可解性；通过跨语言零样本迁移、混淆矩阵和绝对 drop Δ 评估鲁棒性；使用 4 折交叉验证和 GPU 加速。

**📊 数据集**

程序生成数据集：90k 示例，包含三种分区——Java、无类型 Python 与带类型注释的 Python。所有示例实现相同的底层程序，包含填空式函数调用（FIM）任务，且变量/函数名在各分区中完全随机化。

**📈 对比分析**

在标准和对抗性评估下，层级峰值选择（L11–L16 对 SantaCoder，L14–L18 对 CodeLlama）显示显著可解性（S>20），且跨语言零样本迁移在大部分任务中保持 60–80% 以上的 selectivity；对抗性重命名导致 selectivity 下降但仍高于随机，对比基准任务控制显示两模型对类型信息的线性可解性显著优于对照。

**⚠️ 局限性**

局限性：仅覆盖基础类型与列表，未涉及子类型、多态、所有权等复杂类型系统；仅测试两种语言，未验证在更大规模模型或其他语言中的普适性；线性探针仅揭示可解性，未证明因果影响；对抗性实验仅限于变量/函数名重命名，未涵盖更强扰动或语义破坏。

---

## 246. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models

**arXiv ID:** 2607.08317 | [PDF](https://arxiv.org/pdf/2607.08317v1)

**作者:** Matteo Santelmo `[一作]` (École Polytechnique Fédérale de Lausanne), Emmanuel Abbé `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了 Blind Spots Benchmark（盲点基准），包含235道易于人类但对现代多模态AI仍具挑战性的任务，配有结构化参考答案和自动评估流水线。

**💡 创新点**

创新点在于：①从学生处收集真实的AI失败问题并系统清洗；②提出针对该数据集的三层任务分类（对象中心、抽象推理、语言知识）与细粒度子任务；③搭建自动化评估与人工校验相结合的Grader管道；④对多种开源与闭源、文本与多模态模型进行统一评价，揭示闭源模型在盲点任务上的优势与开源模型在成本效益上的突出。

**🔧 技术方法**

主要技术包括：文本与图像生成模型调用、AI评测 Grader（Gemini 3.0.1）+代码执行环境、vLLM部署、自动化脚本（Inspect AI）、统计分析（accuracy, pass@k, cost‑performance 关系）。

**📊 数据集**

使用的核心数据集是公开的 Blind Spots Benchmark（https://huggingface.co/datasets/matsant01/blind-spots-bench），其中包含结构化参考答案与任务标签。

**📈 对比分析**

评估方法：对每个模型先生成答案，再用 Grader 自动判定正确性；对文本类任务做 4 次采样求 mean@4，图像类一次；同时记录 token 数和平均推理成本。结果显示，闭源前沿模型平均准确率高约10%于开源模型；在文本任务中 Gemini 3.5/ Gemini 4 等表现最好（≈83%），在多模态任务中 GPT-4o 与 Gemini 4 最高（≈66%）。成本方面，某些开源模型（GLM‑5.2、Qwen3.5‑122B、DeepSeek‑V4）在相同成本下可匹敌或超过部分闭源模型。

**⚠️ 局限性**

局限性包括：数据量相对有限且子任务分布不均；问题主要针对两款前沿模型，可能对其他模型弱点缺乏覆盖；未提供人类基线，难以直观评估人机差距。

---

## 247. Principled Analysis of Deep Reinforcement Learning Evaluation and Design Paradigms

**arXiv ID:** 2607.07769 | [PDF](https://arxiv.org/pdf/2607.07769v1)

**作者:** Ezgi Korkmaz `[一作]` `[通讯]`, Ezgi Korkmaz

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对强化学习的规模律、容量与复杂度关系进行了理论与实验研究，并揭示了低数据与高数据训练之间性能排名非单调的隐含假设；

**💡 创新点**

创新点在于系统性地证明并验证了“性能随样本规模单调递增”这一常见假设的错误，并提出了基于非单调性的新评估框架；

**🔧 技术方法**

主要采用理论分析（非平稳策略下的回报与样本复杂度）、深度 Q‑学习相关算法（DQN、Dueling、C51、QR‑DQN、IQN）、以及优化库 Haiku/Optax/RLax；

**📊 数据集**

实验数据集为 Atari ALE，分别在低数据（100K 交互）和高数据（200M 交互）两种训练规模下进行；

**📈 对比分析**

比较方法：将新算法与核心基线（如 Dueling）直接对比，报告人类归一化的中位数、均值及 20% 分位数；实验显示，单纯基于高数据表现的基线往往在低数据下排名较差，新提出的理论与实验均表明性能关系非单调；

**⚠️ 局限性**

局限性包括：实验仅覆盖 Atari 游戏；评估受 ALE 100K 选样本偏差影响；理论结果基于特定 MDP 设定，可能不完全泛化到其他任务；

---

## 248. Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE

**arXiv ID:** 2607.07740 | [PDF](https://arxiv.org/pdf/2607.07740v1)

**作者:** Haozhan Tang `[一作]` (NVIDIA), Han Cai `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种零调优的长上下文扩展方法Jet‑Long，动态调整RoPE映射以保持旋转角度在训练分布内，并通过包含-排除注意力合并和即时RoPE校正实现推理时的零成本扩展。

**💡 创新点**

核心创新在于：1）基于当前序列长度动态计算分组因子G，使远程窗口的RoPE角度始终落在预训练范围；2）利用包含-排除原则将三次FlashAttention合并为单次CuTe kernel，避免额外矩阵；3）通过即时旋转校正保持KV缓存不变；4）仅需单一超参数w₀即可在多种模型尺寸上保持鲁棒。

**🔧 技术方法**

技术手段包括RoPE位置编码、动态分组映射、FlashAttention、CuTe自定义内核、包含-排除注意力合并、对KV缓存的在线旋转校正。

**📊 数据集**

在公开长文本基准上评估：RULER（13项记忆任务）、HELLMET‑RAG（多任务检索问答）、PG‑19（长文本困惑度），并将Jet‑Long迁移到混合线性/softmax的Jet‑Nemotron架构。

**📈 对比分析**

与DNTK、YaRN、DCA、Self‑Extend以及原始基线比较，Jet‑Long在RULER上分别提升4.79/2.18/2.03个百分点，在PG‑19困惑度保持最低；在HELLMET‑RAG上与Self‑Extend持平或略优；推理速度在超过32K后，融合CuTe kernel实现预填充速度提升至1.28–1.39×FA2，生成速度保持≥0.96×FA2，整体延迟增加不超过4%。

**⚠️ 局限性**

局限性包括：1）仅适用于含RoPE的softmax‑with‑RoPE模型；2）极端长度（>128K）仍可能出现性能衰减；3）实现依赖CuTe自定义核，需在不同GPU上重新编译；4）对超参数w₀的鲁棒性虽好，但在极小或极大窗口时仍需微调；5）未对更广泛的稀疏或线性注意力架构做系统评估。

---

## 249. SASGeo: Stability-Aware Semantic Map Localization for GNSS-Denied UAVs -- A Framework and Synthetic Proof of Concept

**arXiv ID:** 2607.07737 | [PDF](https://arxiv.org/pdf/2607.07737v1)

**作者:** Natalia Trukhina `[一作]` (Embedded Intelligence Lab), Vadim Vashkelis `[通讯]` (Embedded Intelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套完整的基于语义地图的无人机GNSS失效定位框架，该框架利用稠密语义栅格对齐、语义对象关系图匹配、时间一致性校验、地图年龄与季节持久性权重以及完整性决策机制，实现对无人机姿态的绝对位置修正。

**💡 创新点**

创新点在于将稠密语义对齐、语义图关系验证、时间与地图年龄持久性权重以及完整性可接受决策统一为一个概率模型，并给出了可复现的合成实验验证，说明语义几何与完整性决策的协同效果。

**🔧 技术方法**

核心技术包括：语义BEV投影与多帧累计、稠密栅格匹配与三态似然函数、语义对象关系图匹配、时间一致性损失、完整性校准的接受规则以及自适应权重组合。

**📊 数据集**

实验使用合成的30个语义地图区域（每个256×256像素，包含道路、建筑、水域等5类），在这些合成地图上生成随机旋转、尺度、裁剪、遮挡以及地图变化的查询样本。

**📈 对比分析**

通过与全局语义描述符、均匀栅格匹配、栅格+图、加稳定性、完整性处理等多种方法对比，Recall@1从58.6%提升至94.5–95.5%，Recall@5和MRR均接近1，表明稠密语义几何显著提升了定位召回率和准确度。

**⚠️ 局限性**

局限性主要在于仅在合成环境下验证，未包含真实图像分割误差、VIO漂移、真实地图老化、季节变化、开放空间语义稀缺等因素，缺乏实时性评估和闭环飞行验证。

---

## 250. Shift & Drift: A Zero-Shot Benchmark for Generalizable and Robust Autonomous Driving Motion Planning

**arXiv ID:** 2607.07844 | [PDF](https://arxiv.org/pdf/2607.07844v1)

**作者:** Alessandro Canevaro `[一作]` (Mercedes-Benz AG), Julian Jordan `[通讯]` (Mercedes-Benz AG)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Shift & Drift双轨道基准，用以在语义迁移和状态分布漂移场景下评估自适应驾驶规划器。

**💡 创新点**

创新点在于将航拍数据DSC3D转换为nuPlan兼容场景以实现零样本语义迁移，并设计多种噪声注入方式（AWGN与OU）逼真模拟执行误差；同时提供大规模公开基准。

**🔧 技术方法**

采用数据转换管线、深度地图解析、滑动窗口分段、投影对齐、R树索引；使用高斯白噪声和Ornstein‑Uhlenbeck过程注入动态扰动；闭环评估指标包括安全、进度、舒适度等。

**📊 数据集**

使用DSC3D（航拍高精度轨迹）与nuPlan训练/评估数据；DeepPlan为DSC3D转换后的nuPlan格式；对比使用nuPlan Val14。

**📈 对比分析**

对IL、RL、规则三类规划器（PlanTF、Diffusion Planner、PLUTO、CaRL、PDM-Closed）进行零样本语义迁移和噪声鲁棒性评测；结果显示RL规划器CaRL在两条轨道上表现最稳健，IL规划器在语义迁移或噪声下性能大幅下滑。

**⚠️ 局限性**

局限在于噪声模型仅为仿真，缺乏真实车辆动态；基准仅覆盖部分驾驶文化和城市；评估仍无法完全区分奖励设计、闭环训练与策略灵活性等因素。

---

## 251. STEMbot: A Compliant Robot for Under-Canopy Plant Navigation

**arXiv ID:** 2607.07873 | [PDF](https://arxiv.org/pdf/2607.07873v1)

**作者:** Zachary Charlick `[一作]` (University of Michigan), Dmitry Berenson `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种名为STEMbot的小型植物攀爬机器人，实现在植被冠层下自主导航与检测；

**💡 创新点**

在攀爬机器人中首次集成深度几何SLAM（PIN‑SLAM）、语义八叉树建图以及基于流形约束的A*路径规划，实现对细小茎干（7–33 mm）及其分枝的连续接触导航；

**🔧 技术方法**

采用PIN‑SLAM进行几何里程计；利用SAM+CLIP进行语义分割并构建语义OcTree；结合高摩擦硅胶轮和四杆连杆驱动实现稳定攀爬；使用A*+流形投影和射线投射生成可视化目标；

**📊 数据集**

实验数据来自四个植株模型（人工3D打印的Dracaena、Olea，活体Monstera deliciosa、Ficus lyrata），以及相机拍摄的离线光学测量（Agisoft Metashape）作为基准；

**📈 对比分析**

通过Chamfer距离与离线光学点云对比，人工植株平均误差3.85 mm，活体植株13.36 mm；在四个试验中实现了安全的分支切换和目标可视化导航，证明了系统的全栈可行性；

**⚠️ 局限性**

主要限制包括：对静态植物假设不成立导致误差增大；光照强度对RGB‑D传感器影响大；PIN‑SLAM跟踪易失效；驱动扭矩不足或轮胎粘附问题；以及缺乏信息论探索策略和对柔性植物的适应性。

---

## 252. INTENT: An LSTM Framework for Vehicle Intention Prediction in Intersection Scenarios with Comprehensive Ablation Analysis

**arXiv ID:** 2607.08316 | [PDF](https://arxiv.org/pdf/2607.08316v1)

**作者:** Logine M. Zaki `[一作]` (German University in Cairo), Catherine M. Elias `[通讯]` (German University in Cairo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 INTENT 框架，利用 LSTM 在 BEV 视角下对交叉路口车辆在事件发生前 2 秒的行驶意图（直行/左转/右转）进行预测，并进行了系统的消融实验。

**💡 创新点**

创新点在于：①首次将 LSTM 迁移至交叉路口场景，并通过 Road Labeling 与 Ground Truth Labeling 方案实现无标注数据的自动标注；②在交叉路口数据集上进行全面的消融分析，比较不同网络层数、正则化、双向结构、激活函数、窗口长度、学习率、预测窗口等因素对性能的影响；③在单一交叉路口 BEV 数据集上实现 99.71% 的准确率。

**🔧 技术方法**

采用的技术包括 LSTM、双向 LSTM、Batch Normalization、Dropout（正则化）、Adam 优化器、交叉熵损失函数、不同激活函数（ReLU、tanh、sigmoid）、不同窗口长度（20/25帧）、不同隐藏层数、不同学习率、以及预测窗口调整。

**📊 数据集**

使用的数据集为 InD BEV 数据集，涵盖多种交叉路口结构（T 型、十字交叉等），并通过手动提取道路坐标与方向角实现车辆行驶意图标注。

**📈 对比分析**

与多种模型变体（不同特征数、不同网络结构、不同正则化策略等）进行对比，最终基线模型在测试集上实现 99.71% 的准确率，精确率、召回率均超过 90%，显示出较高的鲁棒性；消融实验表明正则化和 20 帧窗口为提升性能的关键因素。

**⚠️ 局限性**

局限性包括：①仅在交叉路口 BEV 数据集上验证，缺乏对高速公路、复杂路况等其他场景的泛化测试；②预测窗口固定为 2 秒，未探讨更长或更短窗口的实际应用场景；③模型对车辆尺寸、速度等特征高度依赖，可能在不同传感器配置下表现不佳；④实验主要关注准确率，未深入评估实时性与推理延迟；⑤对数据偏差与潜在过拟合的处理仍有提升空间。

---

## 253. LOGOS: Language-guided Oriented Object Detection in Aerial Scenes

**arXiv ID:** 2607.08004 | [PDF](https://arxiv.org/pdf/2607.08004v1)

**作者:** Trong-Thuan Nguyen `[一作]` (University of Science), Minh-Triet Tran `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LOGOS方法，利用文本提示引导Transformer完成航空图像中的定向目标检测；

**💡 创新点**

创新点在于通过prompt‑modulated content queries将文本信息注入Transformer，使查询可根据文本动态调整，解决固定查询数、角度周期性和方位歧义等问题；

**🔧 技术方法**

使用了DINO encoder、DETR式解码器、FiLM调制查询、跨模态cross‑attention、角度sin/cos编码、Hungarian匹配以及对比式去噪等技术；

**📊 数据集**

在DOTA‑v1.0、v1.5和v2.0这三大遥感数据集上进行实验；

**📈 对比分析**

与多种SOTA方法（如R^3Det、EMO2‑DETR、AO^2‑DETR等）对比，LOGOS在DOTA‑v1.0 mAP 81.32%、v1.5 69.97%、v2.0 66.04%，在多数类别中显著领先，尤其在Plane、Harbor等类别表现突出；

**⚠️ 局限性**

仍存在船舶、棒球场等类别性能略低，对极端角度和高密度/稀疏场景的鲁棒性有待提升，文本prompt在极端角度下的引导效果有限。

---

## 254. Enhancing the KidSat Model: Integrating Geographical Encoding and Data Quality Assessment for Childhood Poverty Prediction

**arXiv ID:** 2607.08281 | [PDF](https://arxiv.org/pdf/2607.08281v1)

**作者:** Hou Hin Ip `[一作]` (University of Bristol), H Juliette T Unwin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对利用卫星图像进行贫困映射的难点，本文提出了一套改进的管线：通过细化监督目标矩阵、引入两阶段图像质量筛选以及将视觉特征与球谐函数地理编码融合，显著提升了模型的预测精度。

**💡 创新点**

创新点包括①在监督目标中将高维稀疏的 DHS 变量进行手工重聚合并引入农村-城镇与财富指数，实现维度从103降至51；②开发基于 scan‑line gap 与 FMASK 的两阶段图像质量评估，系统剔除云覆盖或传感器缺陷图像；③将 DINOv2 视觉嵌入与球谐函数（SH）地理特征相结合，并评估 SH+SIREN 组合的效果；④用梯度提升树（LightGBM/XGBoost）替代传统线性回归头，捕获视觉与地理特征的非线性交互。

**🔧 技术方法**

使用技术包括：DINOv2 ViT 的自监督预训练与微调；球谐函数（Spherical Harmonics）地理编码；SIREN 网络；FMASK 基础的云检测与 scan‑line gap 检测；Ridge 回归、随机森林、XGBoost、LightGBM、浅层 MLP 等回归头；5‑fold CV 与 MAE 评估。

**📊 数据集**

数据集为 DHS 16 或 33 个非洲国家的调查样本，配合 Landsat 7/8（336×336 像素）和 Sentinel‑2 图像，共计 43,823 个聚类（clusters），通过重采样与合并得到 768 维视觉嵌入与 51 维重聚合特征。

**📈 对比分析**

通过在 5‑fold CV 上对比基线（仅 DINOv2）、改进的预处理 + SH、SH+SIREN 等配置，结合不同回归头，本文得到：改进后 MAE 从 0.2167 降至 0.1759（18.8% 下降），进一步扩展至 33 国后 MAE 为 0.1658。树模型（LightGBM/XGBoost）在所有实验中均优于线性/MLP 回归，验证了非线性交互的重要性。

**⚠️ 局限性**

主要局限包括：① SH+SIREN 在本实验中表现逊色，可能因同一贫困目标监督导致表征重叠；② 图像质量筛选仍未覆盖所有噪声源，未尝试图像恢复方法；③ 仅使用公开卫星与 DHS 数据，缺乏更细粒度的社会经济指标；④ 对不同分辨率和传感器的鲁棒性待进一步验证；⑤ 预处理与地理编码的超参数选择仍主要依赖经验。

---

## 255. HSA: Hierarchical Slot Attention for Multi-granularity Scene-Decomposition

**arXiv ID:** 2607.08249 | [PDF](https://arxiv.org/pdf/2607.08249v1)

**作者:** Neelu Madan `[一作]` (Aalborg University), Thomas B. Moeslund `[通讯]` (Aalborg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

学习了一种单模型能够同时完成全景、语义与实例级的场景分解任务；

**💡 创新点**

利用最小监督（仅10%标注）与层级对齐损失，首次实现语义层次结构的自动编码；

**🔧 技术方法**

基于Slot Attention与DINOv2预训练特征，结合Dice损失、层级对齐和Straight‑Through Estimator；

**📊 数据集**

在MS COCO 2017和PASCAL VOC 2012数据集上训练，使用10%标注数据做监督；

**📈 对比分析**

与多种无监督Flat baseline在COCO、VOC上比较，HSA在全景层ARI提升+41.5，语义层+14.6，实例层+10.4，整体性能显著优于所有Baseline；

**⚠️ 局限性**

仍需要一定量标注（10%），层级对齐与预训练特征可能冲突，评测指标（ARI等）受限于单一粒度，无法完全反映多级合理分解的多样性。

---

## 256. Feedback Manipulation Regularization: Enabling Offline Agent Alignment for Imitation Learning

**arXiv ID:** 2607.07859 | [PDF](https://arxiv.org/pdf/2607.07859v1)

**作者:** Benjamin Poole `[一作]`, Minwoo Lee `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种离线模仿学习的正则化方法FMR，利用评估性反馈提升策略与人类价值的对齐性；

**💡 创新点**

创新点在于将评估性反馈映射为温度缩放，在概率空间进行正则化，兼容任何模仿学习算法，克服了传统多阶段偏好学习在连续决策中的不足；

**🔧 技术方法**

主要技术包括评估性反馈温度缩放、逆KL正则化、离线模仿学习框架（BC、IQL、DemoDICE、ReCOIL）以及Safety Gymnasium实验环境；

**📊 数据集**

实验使用基于Safety Gymnasium的导航和速度限制任务的数据集，包含人工/代理演示及成本相关的负反馈，并设置高、低重叠演示比例；

**📈 对比分析**

与DVL、CPL等替代方法比较，FMR在不同演示比例下显著降低误差率（最高可达98%），且成功率或回报保持或提升；

**⚠️ 局限性**

局限性主要在于对演示与反馈覆盖度的依赖，难以在高维连续动作空间或低覆盖度场景下保持效果，并且需要大量手工反馈。

---

## 257. Carnap Ten Years Later: Lessons Learned and Next Steps

**arXiv ID:** 2607.07722 | [PDF](https://arxiv.org/pdf/2607.07722v1)

**作者:** Graham Leach-Krouse `[一作]` `[通讯]` (Draper Laboratory), Graham Leach-Krouse (Draper Laboratory)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

总结并重构了 Carnap 证明助手框架，发布了新的可在浏览器中运行的、基于 Metamath Zero 的轻量级证明编译器 Aufbau Bytecode Compiler 与小型可信核 mm0-zig，用以提升可扩展性、性能与可维护性；同时对过去十年中对 45,000 名学生的使用经验进行系统回顾。

**💡 创新点**

① 将证明校验从前端 Haskell+GHCJS 迁移到浏览器可执行的 Zig 代码，拆分为独立的编译器与可信核；② 采用 MM0 二进制证书格式，实现线性时间校验与极小可信基；③ 通过简洁的文本证明格式实现自然演绎、序列演算、模态逻辑等多种证明系统的无代码支持。

**🔧 技术方法**

主要技术包括：Metamath Zero（MM0）证明语言与格式；Zig 编译器生成 WebAssembly；Haskell+GHCJS（旧版前端）；Nix 进行可复现开发环境；JavaScript 与 Web API；以及在浏览器中实现的轻量级可信核 mm0-zig。

**📊 数据集**

性能评测使用 MM0 公开库中的 Peano.mmb（由 Peano.mm1 生成，包含 2723 条定理）以及 71 个 Wiedijk 100 目标的完整 Metamath 库；数据集主要是证明文本与已验证的 MMB 证书。

**📈 对比分析**

对 mm0-zig 与原始 mm0 验证器进行对比，采用 zig 的 Release 版本与 Profile‑Guided Optimisation 进行编译；平均验证时间为 6.1 ms（σ 0.6 ms，范围 5.3–10.5 ms，285 次跑测），与原 mm0 的 7.1 ms（σ 0.5 ms，范围 6.5–12.3 ms，243 次跑测）相差不到 1 ms，说明性能保持一致且略有提升；此外整体 MM0 库在 200 ms 内完成校验，验证了线性时间特性。

**⚠️ 局限性**

主要限制包括：① 旧的 GHCJS 与 Haskell 代码在模块耦合与性能上仍有残余影响，迁移成本高；② 虽然引入了轻量级可信核，但仍需进一步验证在不同浏览器与环境中的安全性；③ 当前的 Aufbau 编译器仅支持文本编辑器接口，缺乏结构化的 UI（如 Fitch 树或 Prawitz 树），对初学者友好度有限；④ 开源社区贡献率仍不高，主要因工具链安装门槛（Nix、Zig）与技术栈偏门；⑤ 对大型课程或高并发考试的后端负载仍需进一步优化。

---

## 258. LEXIC: Lightweight Eye-tracking eXtension via Injected Complexity

**arXiv ID:** 2607.08152 | [PDF](https://arxiv.org/pdf/2607.08152v1)

**作者:** Sumin Lee `[一作]` (Seoul National University), Nam-Joon Kim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在眼动模型中注入预先计算的词级难度信号（GPT-2 概率、词频、词长），提出两种轻量级的特征注入机制，使眼动仅模型在阅读理解任务上实现超过基准的 AUROC。

**💡 创新点**

设计了无语言模型推理的轻量级注入方法，包括直接拼接和残差机制，证明预先计算的心理语言学特征可显著提升眼动模型性能。

**🔧 技术方法**

基于 EyeBench 的 AhnCNN 后端，加入词难度特征的通道拼接或残差预测头，并使用 K=5 种子集成和 Wilcoxon 检验评估。

**📊 数据集**

使用 OneStop 眼动阅读理解数据集（180 名读者、1.1M 注视点）以及 EyeBench 的十折交叉验证。

**📈 对比分析**

与基准 AhnCNN（AUROC≈0.49）以及 BEyeLSTM（≈0.525）比较，LEXIC‑Concat 在 Unseen Text 约 +1.8pp，Unseen Reader 约 +2.9pp，AUROC 提升至 0.51–0.56，达到与 BEyeLSTM 相近的水平。

**⚠️ 局限性**

仅在 OneStop 任务与 AhnCNN 后端验证，未测试其他阅读任务或后端模型；残差机制在未知读者下受限于典型读者预测头的校准。

---

## 259. Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions

**arXiv ID:** 2607.08011 | [PDF](https://arxiv.org/pdf/2607.08011v1)

**作者:** Anjun Gao `[一作]` (University of Louisville), Minghong Fang `[通讯]` (University of Louisville)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个用于代码补全模型后期追溯后门行为的取证框架。

**💡 创新点**

创新点在于无梯度、无攻击者先验知识即可通过结构化指纹提取、检索缩小搜索空间并利用LLM推理实现对后门训练样本的精确归因。

**🔧 技术方法**

核心技术包括LLM生成的结构化行为指纹、基于UniXcoder的代码嵌入检索、以及LLM驱动的语义归因分析。

**📊 数据集**

使用约108万条GitHub公开Python文件构成的训练集，包含80k条干净样本与不同后门攻击产生的污染样本，模型为CodeGen‑Multi。

**📈 对比分析**

与十种基线方法（如All‑Once、TracLLM、RAGForensics等）对比，本文方法在10种后门攻击下FNR低于3%，DACC高于97%，并在检索后精确度上超过基线；运行时间仅47秒，成本低至0.33美元/个误补全。

**⚠️ 局限性**

局限性包括对误报的敏感性，需要人工审核来防止攻击者制造伪误补全；在多攻击交叉、极度稀疏污染样本或非代码补全模型场景下的表现尚未验证。

---

## 260. Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignment

**arXiv ID:** 2607.08256 | [PDF](https://arxiv.org/pdf/2607.08256v1)

**作者:** Taehyung Yu `[一作]` (KAIST), Seongjae Kang `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Best‑of‑N (BoN) 推理在零样本 TTS 中的内容一致性改进，并揭示了评估者（ASR 家族）选择对 WER 评价的显著影响。

**💡 创新点**

首次发现并系统量化了 ASR 家族与验证器的自我偏见（confound），并提出跨家族排名集成（rank‑avg 与 max‑rank）以减轻该偏差。

**🔧 技术方法**

采用 F5‑TTS 生成候选音频，使用 Whisper、wav2vec 2.0 与 HuBERT 等多种 ASR 评估器进行 WER 评估，并通过线性 CKA 分析评估器间的表征相似性。

**📊 数据集**

实验数据来源于 LibriSpeech‑PC test‑clean 语料（1127 条短句），以及公开的 F5‑TTS 官方评测管线。

**📈 对比分析**

对比单一评估器和跨评估器集成方案，发现跨家族集成在三种评估器下均可将平均 WER 降至 1.61%（比 F5‑TTS 降低 12%），同时保持 SIM‑o 与 UTMOS 无显著质量损失。

**⚠️ 局限性**

研究仅覆盖单一 TTS 后端（F5‑TTS）、有限的 ASR 家族和缺乏人类 MOS 验证，结果可能无法完全泛化至其他模型和评测方法。

---

## 261. Attention-Based Segmentation of WMHs and Differentiation of Vascular vs. Demyelinating Lesions

**arXiv ID:** 2607.08171 | [PDF](https://arxiv.org/pdf/2607.08171v1)

**作者:** Aina Tur-Serrano `[一作]` (Universitat de les Illes Balears), Francisco J. Perales López `[通讯]` (Universitat de les Illes Balears)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文针对脑MRI中白质高信号（WMH）进行分割与分类，采用注意力机制增强的U-Net结构，并在分割后提取形态学特征用于区分血管性和脱髓鞘性病变。

**💡 创新点**

创新点在于将Bottleneck Attention Module (BAM) 与 Convolutional Block Attention Module (CBAM) 同时应用于Encoder、Decoder和Attention U-Net，形成全网络注意力提升；同时结合分割掩膜提取的可解释形态特征，实现端到端的分割-分类联合框架。

**🔧 技术方法**

技术包括2D切片训练、patch‑based 训练、2.5D 输入；U‑Net、Attention U‑Net 及其注意力增强版本；Dice 损失训练；特征提取后使用 SVM、Logistic Regression 与 Random Forest 进行分类。

**📊 数据集**

使用五个公开数据集：WMH Segmentation Challenge、Utrecht Vascular Cognitive Impairment Study、Brain MRI MS、MSLesSeg ICPR 2024 competition 与 Neurocognitive Aging Dataset，涵盖血管性与脱髓鞘性 WMH 及健康对照。

**📈 对比分析**

实验通过五折交叉验证对比不同输入与注意力组合。2D 切片+Attention U‑Net+全 BAM+CBAM 在分割上 Dice 达到 0.71±0.03、Jaccard 0.62；在分类上 Random Forest 的 F1 在血管类约 0.71，脱髓鞘类约 0.60，使用分割掩膜与 ground‑truth 差距不大。

**⚠️ 局限性**

局限性包括数据量有限且来源异质，导致模型在不同扫描仪与协议下的泛化受限；patch‑based 与 2.5D 方法表现不佳；未考虑 3D 形态特征与多模态信息，未来需扩大样本与改进跨域适配。

---

## 262. Psychological Competence as a Missing Dimension in AI Evaluation

**arXiv ID:** 2607.08285 | [PDF](https://arxiv.org/pdf/2607.08285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 263. On the Design of Mixture-of-Experts for Dynamic Gaussian Splatting

**arXiv ID:** 2607.08250 | [PDF](https://arxiv.org/pdf/2607.08250v1)

**作者:** In-Hwan Jin `[一作]` (Pusan National University), Kyeongbo Kong `[通讯]` (Pusan National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了两种多专家（Mixture‑of‑Experts）框架，用于改进动态 3D 高斯分散（Gaussian Splatting）场景重建。

**💡 创新点**

创新点在于：①通过 Mixture of Deformation Experts (MoDE) 在共享的规范化 3D 高斯表示中联合优化多种变形专家；②通过 Mixture of Experts for Dynamic Gaussian Splatting (MoE‑GS) 在独立训练后使用卷积+体积感知路由器对专家渲染结果进行像素级融合，实现异构专家的灵活组合。

**🔧 技术方法**

主要技术包括：基于 4D/4D Gaussians 的高斯表示、基于 MLP、哈希、曲面等的变形网络、时序滑线式门控、体积感知像素路由器、单通道渲染、多专家融合、单通道剪枝和知识蒸馏。

**📊 数据集**

实验数据集：Neural 3D Video（N3V）和 Technicolor，此外还在 HyperNeRF、PanopticSports、D‑NeRF 等数据集上做了验证。

**📈 对比分析**

对比方法：单专家基线（如 4DGaussians、Grid4D、E‑D3DGS、STG、Ex4DGS 等）、NeRF 相关基线和混合专家方案。MoDE 在动态区域提升 PSNR 约 0.3–0.5 dB；MoE‑GS 在全图 PSNR 上提升 1.5–3.0 dB，且在多场景上保持稳定；通过单通道剪枝和蒸馏可将推理时 FPS 提升 2–3 倍，同时保持视觉质量。

**⚠️ 局限性**

局限性：①MoDE 在共享规范化空间下受限，难以加入非共用 3D 表示的专家；②MoE‑GS 需要多次独立训练和额外路由阶段，导致训练时间显著增加；③在静态区域可能出现轻微失真，尤其当多专家均侧重动态时；④对超大规模场景的显存需求仍较高，尽管通过剪枝有所缓解。

---

## 264. Factors Influencing Conversational Engagement in Robot-Delivered Individual Cognitive Stimulation Therapy (iCST) for Dementia in Home Settings

**arXiv ID:** 2607.07998 | [PDF](https://arxiv.org/pdf/2607.07998v1)

**作者:** Emmanuel Akinrintoyo `[一作]` (Imperial College London), Nicole Salomons `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了社交机器人在认知刺激疗法（iCST）中对痴呆患者的会话动态，并探讨了个性化提示、会话阶段和居住环境对对话参与度的影响

**💡 创新点**

首次系统量化了个性化提示在机器人驱动的认知疗法中提升自我引用和积极情绪的效应，并揭示了会话中出现的认知疲劳现象

**🔧 技术方法**

使用Whisperx与WhisperD进行音频转录与说话人分离，LLama‑3模型进行意图与语言特征分类，并计算多种对话指标（词数/回合、发声时长、语速、延迟、停顿等）

**📊 数据集**

基于27个30分钟的家庭现场录音，涵盖8名痴呆患者的对话数据

**📈 对比分析**

通过非参数统计检验（Mann–Whitney U、Kruskal–Wallis、Chi‑square）比较个性化与通用提示、早期与后期会话阶段以及不同参与度、居住情况组的指标，结果显示个性化提示显著提高了对话时长、自我引用率和积极情绪，早期会话的词数与语速高于后期，初次会话的延迟和犹豫率可预测长期参与度

**⚠️ 局限性**

样本量小（仅8人），仅进行一周短期部署，缺乏长期追踪与大规模验证

---

## 265. From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents

**arXiv ID:** 2607.08028 | [PDF](https://arxiv.org/pdf/2607.08028v1)

**作者:** Joongho Ahn `[一作]` (AI Leadership Research Center), Moonsoo Kim `[通讯]` (AI Leadership Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种 harness‑engineering 方法，将原先以提示为主的企业 LLM 原型重构为可追溯、可审计的 LLM‑agent 架构；通过代码控制层管理源授权、实体路由、声明选择、答案结构、追踪生成和验证门，保持与模型分离的可替换 composition boundary。

**💡 创新点**

创新点包括：①将源到声明管道、答案合成与验证逻辑从提示迁移到可版本化的代码、清单与验证器；②构造可替换的 composition boundary，使模型语义与业务规则解耦；③在三大实验（固定验证、模型替换、验证层消融）中验证该架构的源可靠性、实体路由、追踪完整性、输出洁净和推荐语言控制，并证明这些保障是由代码承担、而非仅靠模型提示。

**🔧 技术方法**

技术实现：TypeScript + JSON 配置；RAG 与检索接口（OpenDART、KRX、NAVER News）构建源层；LLM 通过 OpenRouter 调用 Claude Sonnet 4、GPT‑4.1 mini、Gemini 2.5 Flash；确定性 composer 作为 fallback；fault injection、模型替换与 enforcement‑layer ablation 的实验脚本；日志与 trace 生成，验证器与 guardrail（bolt‑on）对比。

**📊 数据集**

使用的公开数据集为韩国五大企业集团（三星、SK、现代汽车、LG、韩华）共 25 家上市公司，提取的公开文件（DART filings、IR 页面、KRX 市场行情、NAVER 新闻）构成 113 条源到声明；从中挑选 25 条关键声明用于报告。全部数据均为公开可访问。

**📈 对比分析**

评估方法：①30 个固定验证场景与故障注入测试（共 270 次）；②模型替换实验（Claude Sonnet 4、GPT‑4.1 mini、Gemini 2.5 Flash 共 270 次）比较最终合规率；③验证层消融实验（prompt‑only、外部 guardrail、harness 三种条件，120 次）比较推荐语言泄漏、内部追踪泄漏和整体实用性。结果显示：代码控制层在 270 次中始终通过所有检查；模型合成成功率为 68.9%–85.6%；prompt‑only 条件导致 30/30 次违规；外部 guardrail 过度拒绝 4 次，实用性下降至 88/120；harness 保持 120/120，且在故障注入和模型替换实验中验证器始终可捕获违规。

**⚠️ 局限性**

局限性：①实验只验证合同保留、源可靠性、追踪完整性等系统级保障，并未评估答案的业务价值或推理质量；②仅使用了公开的、有限的企业集团数据，未覆盖私人文档或更广泛的行业场景；③模型输出的非确定性与模型版本变更导致实验可复现性受限；④未针对实时生产日志和长期运维进行评估，缺乏动态更新与模型演进的验证。

---

## 266. TMI: Text-to-Image Meets Image-to-Image for Complementary Data Synthesis to Boost Long-Tailed Instance Segmentation

**arXiv ID:** 2607.08201 | [PDF](https://arxiv.org/pdf/2607.08201v1)

**作者:** Hyeonseop Song `[一作]` (LG Electronics), Hoseok Do `[通讯]` (LG Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种融合文本到图像（T2I）生成与语义指令驱动图像到图像（I2I）编辑的混合数据合成框架，以提升长尾实例分割模型的性能。

**💡 创新点**

创新点包括：①基于提示一致性过滤的离线伪标签与EMA教师-学生自适应伪标签相结合，显著提升T2I图像的标签质量；②设计了“Place-and-Verify”I2I编辑器 VRAIN，利用视觉语言模型实现稀有类别的语义匹配与位置指令，并通过视觉语言模型验证与 SAM 掩码生成，保证高保真、上下文一致的实例编辑；③将两类合成数据统一训练，形成互补的稀有与多样性提升。

**🔧 技术方法**

主要技术包括：Diffusion 生成模型（Flux、Flux-Context）、视觉语言模型（InternVL3）用于指令生成与验证、SAM 进行精细掩码提取、EMA 教师-学生框架用于自适应伪标签、动态阈值伪标签筛选、结构相似度（SSIM）与开放词汇检测结合实现编辑区域定位。

**📊 数据集**

在 LVIS 长尾实例分割基准上进行实验，使用 200k T2I 图像和 80k I2I 图像作为合成数据，结合 100k 实际训练集。

**📈 对比分析**

与 MosaicFusion（T2I）、X-Paste、DiverGen（I2I）等基线比较，使用 CenterNet2 训练，结果显示整体 AP 提升约 +4 点，稀有类别 AP 提升约 +9.5 点；在 Swin-L backbone 上，整体 AP 从 47.5 提升至 50.7，稀有类 AP 从 41.4 提升至 49.1。

**⚠️ 局限性**

局限性：生成模型对输入指令的执行不完美，T2I 生成可能缺失提示中的类别，I2I 编辑有时无法准确遵循位置信息，导致标签噪声和上下文不一致问题；此外，依赖预训练模型与外部 VLM 的性能限制，生成质量仍受制于模型本身。

---

## 267. The Memory Wall of Green Software: Empirical Energy Evaluation of Memento Design Pattern

**arXiv ID:** 2607.07944 | [PDF](https://arxiv.org/pdf/2607.07944v1)

**作者:** Imane Jriri `[一作]` (Mohammed V University In Rabat), Younes El Amrani `[通讯]` (Mohammed V University In Rabat)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Memento模式在不同状态规模下的能耗进行实验评测，比较基线、完整快照和差分编码三种实现。

**💡 创新点**

创新点在于首次揭示了“绿色软件内存墙”，即差分编码在规模过大时因GC剧增而导致能耗反弹，并给出了基于状态大小的能耗决策框架。

**🔧 技术方法**

采用Intel RAPL硬件能耗采集、.NET 8.0服务器GC、XOR差分编码与完整序列化等技术。

**📊 数据集**

实验使用合成状态数据集，规模从10 MB扩展到200 MB，重复30次以获得统计稳健性。

**📈 对比分析**

通过对总能耗和GC频率的直接测量比较，差分策略在50–100 MB范围内可降低65.8 %能耗；但在200 MB时，GC抖动导致差分耗能比完整快照高25.9 %。

**⚠️ 局限性**

局限性包括阈值与平台相关（仅在Intel CISC + .NET Server GC上验证）、未覆盖ARM或Java ZGC等其它体系结构与垃圾回收器。

---

## 268. Secure QR Codes: Authenticity Verification via EdDSA Signatures and CBOR Certificates

**arXiv ID:** 2607.08383 | [PDF](https://arxiv.org/pdf/2607.08383v1)

**作者:** Wojciech Jonderko `[一作]` (Wroclaw University of Science and Technology), Wojciech Wodo `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了基于Ed25519签名、CBOR证书以及ZLIB压缩的安全二维码体系，并演示了离线验证、在线Hybrid Web PKI两种模式。

**💡 创新点**

创新点在于：①将完整信任链嵌入极小二维码；②通过“Matryoshka”层叠结构实现自签与证书链兼容；③将URL片段与JWKS交互，实现动态密钥吊销和后向兼容；④提供完整开源实现和实验评估。

**🔧 技术方法**

技术包括：EdDSA (Ed25519)、CBOR/COSE、ZLIB、QR Code标准（ISO/IEC 18004）、JSON Web Key Set (JWKS)、HTTPS、TLS 证书交叉验证。

**📊 数据集**

数据集主要为自定义测试字符串（URL、短链接、JSON payload），以及公开可用的 EU Digital COVID‑Certificate 示例，用以验证压缩率与签名完整性。

**📈 对比分析**

通过实测，离线模式在QR版本8-13之间即可容纳完整签名与证书，容量约300–600字节；在线Hybrid模式将二维码压缩到版本6–8，仅携带URL与签名片段，显著降低数据量并支持即时吊销；性能上，验证延迟在数十毫秒，适合移动端使用。

**⚠️ 局限性**

局限包括：离线模式缺乏即时吊销能力、二维码尺寸随证书链增长导致可读性下降、对域名拼写错误的防护依赖用户；Hybrid模式对网络连通性要求较高，且需维护中央信任注册表。

---

## 269. Adaptive Row Selection Meets Asynchrony in Randomized Kaczmarz

**arXiv ID:** 2607.08313 | [PDF](https://arxiv.org/pdf/2607.08313v1)

**作者:** Evan Coleman `[一作]` `[通讯]` (University of Mary Washington), Evan Coleman (University of Mary Washington)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在共享内存异步环境下，随机 Kaczmarz 算法结合自适应行选择的性能与稳定性。

**💡 创新点**

首次系统地量化了“稳定边界”并揭示了过度自适应采样导致的崩溃现象以及通过欠松弛恢复稳定的可行性。

**🔧 技术方法**

采用了权重指数采样、阈值贪婪（GRK）、不一致与一致读取语义、以及残差维护的 Gram 行更新技术。

**📊 数据集**

在 96 核节点上测试了高斯稠密、医学成像投影（tomography）和 16 个 SuiteSparse 稀疏矩阵。

**📈 对比分析**

与传统统一采样和串行结果对比，发现适度自适应采样在迭代次数上能实现近乎 2 倍加速，但在极端并发下需欠松弛；不一致读取在性能和安全性上更优。

**⚠️ 局限性**

主要局限在于全局采样仍是内存受限的瓶颈，并且理论仅给出经验式稳定阈值，缺乏精确的渐进性分析。

---

## 270. NFTR: From Provable Mode-Averaging to Geodesic Subgoal Selection in Offline Goal-Conditioned RL

**arXiv ID:** 2607.07855 | [PDF](https://arxiv.org/pdf/2607.07855v1)

**作者:** Erdemt Bao `[一作]` (Huazhong University of Science and Technology), Jun Chen `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了NFTR，一种将正则化的Normalizing Flow与三角斜边重加权相结合的离线目标条件RL框架。

**💡 创新点**

用Normalizing Flow替代高层高斯子目标分布以避免模式崩塌，并通过三角斜边（triangle‑slack）对AWR权重进行几何一致性重加权，从而抑制乐观偏差。

**🔧 技术方法**

结合IQL期望回归、MRN对quasi‑metric的架构性三角不等式保证、RealNVP条件流、RWDR（Value + Triangle‑Slack）加权AWR以及梯度裁剪等训练技巧。

**📊 数据集**

在Ogbench平台的PointMaze、AntMaze、Cube、Scene等离散/连续目标任务上进行实验。

**📈 对比分析**

与HIQL、SAW、OTA、QRL、CRL、TMD、GCFQL等基线比较，在随机/拼接/操作任务中NFTR平均提升20‑30%成功率，且在最难环境（teleport、stitch）实现显著领先。

**⚠️ 局限性**

对超长时间序列或高DoF接触式控制，NFTR在规划与物理可实现性上仍受限，且三角斜边与随机失效的联系仍为经验性，而非严格理论保证。

---

## 271. fog: Expressing Motion and Emotion through Function Composition of AI-Generated Code

**arXiv ID:** 2607.07952 | [PDF](https://arxiv.org/pdf/2607.07952v1)

**作者:** Vivian Liu `[一作]` (Columbia University), Lydia Chilton `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 fog 框架，利用抽象类进行运动与情感函数合成，并实现了动画编辑器，对 452 条 Heider‑Simmel 动画进行识别率评估以及对 10 位参与者的交互实验；

**💡 创新点**

将大型语言模型与面向对象的抽象类结合，创建可编程的运动与情感词汇表，实现代码生成的功能组合与即时交互界面；

**🔧 技术方法**

基于 JavaScript 的 LLM 调用、抽象类继承与功能组合、动画引擎、Heider‑Simmel 场景与用户实验；

**📊 数据集**

使用 CrowdResearch 收集的 452 条 Heider‑Simmel 动画样本（12 个副词、32 个动词、12 个手势、12 个情感），以及 10 位专业与非专业参与者的实验数据；

**📈 对比分析**

通过 4AFC 任务评估识别率（整体 68%，相较随机 25% 有显著提升），功能组合识别率 58‑70%；用户实验中 fog 生成动作次数平均 15.1 次，反馈循环 0.36 分钟/次，显著低于基线 1.75 分钟/次，性能优于基线；

**⚠️ 局限性**

受限于简化的 Heider‑Simmel 形状动画，情感与动作区分不够细腻，缺乏事件驱动与更复杂的代码结构，难以泛化到更复杂角色与场景。

---

## 272. A law of robustness for two-layer neural networks with arbitrary weights

**arXiv ID:** 2607.07778 | [PDF](https://arxiv.org/pdf/2607.07778v1)

**作者:** Yitzchak Shmalo `[一作]` `[通讯]` (Hebrew University of Jerusalem), Yitzchak Shmalo (Hebrew University of Jerusalem)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

证明了在高维下，任意两层连续分段线性（包括ReLU）网络若拟合噪声标签且无权重上限，则其Lipschitz常数至少为约 √(n/m)（上限加一日志因子）。

**💡 创新点**

创新点在于引入无参数上界的函数空间覆盖法，并利用“刚性”引理将每个 kink 系数与 Lipschitz 常数关联，从而获得可计算的度量熵。

**🔧 技术方法**

使用的主要技术包括函数空间覆盖、刚性定理、度量熵估计、Lipschitz 集合的离散化、球面/高斯分布下的浓度与局部化以及球谐分析。

**📊 数据集**

实验数据采用高维均匀球面样本或标准正态分布的随机点，标签为 ±1 或服从指定噪声水平的随机变量。

**📈 对比分析**

与已有的上界（m≈n 时可实现 O(1) Lipschitz 的插值器）对比，证明下界与上界在 m≈n 处仅相差一个对数因子；实验显示训练得到的网络满足下界但未达到完整的 √(n/m) 速率。

**⚠️ 局限性**

局限性包括：仍保留一个对数因子；不适用于平滑激活函数或低维情况（如 d=2 圆面失效）；对中等宽度范围内的无对数上界仍为开放问题。

---

## 273. Smoothing Exponents and Decoupling in Semifinite von Neumann Algebras

**arXiv ID:** 2607.07997 | [PDF](https://arxiv.org/pdf/2607.07997v1)

**作者:** Zhiwen Lin `[一作]` (Harbin Institute of Technology), Xinyu Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了半有限冯·纽曼代数中的最大相对熵平滑指数，并给出了与矩阵维度无关的精确指数公式；随后利用该指数完成了半有限参考系统下的量子信息解耦与催化解耦的理论描述。

**💡 创新点**

创新点在于将原本依赖于有限维谱定理、投影计数的平滑指数和解耦定理，改写为完全基于半有限冯·纽曼代数内在结构的运算，并证明了相同的沙盒 Rènyi 信息公式在此更广泛的算子代数框架下仍然成立。

**🔧 技术方法**

主要技术包括：非可度量算子空间的非可测算子理论、非可测 L^p 空间与 Hölder 不等式、几何平均与层层蛋糕引理的算子分析、Mosonyi–Ogawa 公式的半有限推广、以及对齐松耦合 (Datta–Renner) 的温和测量估计；所有证明均保持无维度依赖。

**📊 数据集**

由于论文为纯理论研究，没有使用具体数据集；所有结论均为泛型算子代数结果。

**📈 对比分析**

与传统的有限维矩阵方法相比，作者通过无维度、无谱计数的证明实现了更广泛适用性；在性能方面，平滑指数与解耦可靠性指数保持与经典矩阵结果相同的 Rènyi 信息表达式，表明算子代数视角并未牺牲效能。

**⚠️ 局限性**

主要限制包括：仍需假设参考系统为半有限冯·纽曼代数且无穷维时需利用压缩与极限技术；对于非半有限（如 Type III）系统，部分结果尚未直接推广；此外，某些技术步骤（如几何平均构造）在更一般代数框架下可能需要进一步验证。

---

## 274. Unlocking Temporal Generalization in Hamiltonian Video Dynamics Models

**arXiv ID:** 2607.07763 | [PDF](https://arxiv.org/pdf/2607.07763v1)

**作者:** Eli Laird `[一作]` (Southern Methodist University), Corey Clark `[通讯]` (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该工作在 Hamiltonian Generative Networks 基础上加入 port‑Hamiltonian 结构，研究并修复其在不同时间分辨率下的预测失效问题。

**💡 创新点**

其创新点在于系统性地分析了超出训练步长的 extrapolation 失效机制，并给出了可操作的谱正则化与子步积分两种修复方案，首次实现单模型在更粗更细帧率下稳定生成。

**🔧 技术方法**

采用 port‑Hamiltonian 神经网络、对称跳跃法（symplectic leapfrog）积分器、谱正则化、变分自编码器与多尺度子步技术进行建模与推理。

**📊 数据集**

采用了一个 64×64 RGB 的受迫阻尼弹簧振子模拟数据集，共 50,000 条序列，每条 50 帧，采样步长 0.4。

**📈 对比分析**

在 MAE、PSNR、SSIM、LPIPS 四个视觉指标上，与基线模型对比，加入谱正则化并在推理时子步后，模型在训练步长之外的所有时刻均保持高质量，表现稳健。

**⚠️ 局限性**

主要局限在于仅验证于单一、低维物理系统，复杂多体或真实世界视频的适用性尚未证明，且子步积分会显著增加推理成本。

---

## 275. MatBind: A Shared Embedding Space for Multimodal Materials Characterization

**arXiv ID:** 2607.08470 | [PDF](https://arxiv.org/pdf/2607.08470v1)

**作者:** Le Yang `[一作]` (Forschungszentrum Jülich GmbH), Stefan Sandfeld `[通讯]` (Forschungszentrum Jülich GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出MatBind框架，将晶体结构、pXRD、DOS和文本四种材料模态对齐到统一嵌入空间，实现零样本跨模态检索。

**💡 创新点**

创新点在于使用晶体结构作为锚点，利用对比学习实现多模态无监督对齐，并能产生未显式训练的跨模态检索能力。

**🔧 技术方法**

采用对比学习（InfoNCE）、GCNN、ResNet、Transformer、MatBERT等编码器，并结合ImageBind的锚点策略。

**📊 数据集**

使用Materials Project数据库的晶体结构、模拟pXRD、DOS和Robocrystallographer生成的文本，约15.5万样本。

**📈 对比分析**

与单模态和两模态检索相比，Recall@1最高可达0.99（结构-文本），跨模态零样本检索在某些对上甚至优于显式训练对，结合多模态可进一步提升结构检索精度。

**⚠️ 局限性**

限制在于仅基于计算模拟数据，实验噪声和不完整性未评估；多模态对齐在ambiguity高的模态（如DOS）性能下降；缺乏对未知模态的通用性测试。

---

## 276. Different Teachers, Different Capabilities: Sub-1B On-Device Distillation for Structured Text Enrichment

**arXiv ID:** 2607.08268 | [PDF](https://arxiv.org/pdf/2607.08268v1)

**作者:** Vinay Kumar Chaganti `[一作]` `[通讯]`, Vinay Kumar Chaganti

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大规模新闻文本提炼任务中，将8B推理教师的输出压缩到0.6B学生模型上，实现了快速低成本的结构化输出生成；

**💡 创新点**

创新点在于：1）通过同尺寸非推理教师对照验证推理能力对学生质量的影响；2）将多维度子任务（摘要、标签、结构）细化评估，揭示不同教师在各子任务上的能力分化；3）提供完整的本地无参考评估工具与路由表，支持按字段选择最佳引擎；

**🔧 技术方法**

技术包括：QLoRA微调、4-bit量化、基于LLM的无参考判定面板、bootstrap统计、同一模型集内的多教师对照实验；

**📊 数据集**

数据集为自建RSS-News（494篇新闻，24个来源，包含摘要与5个闭集标签），训练集401篇，测试集93篇；

**📈 对比分析**

与两种非蒸馏基线（few-shot与格式约束）以及两名教师模型比较；学生在摘要检查中恢复了58%教师质量差距，提升16.8个百分点；在标签分类上仅部分字段提升；总体效率提升约50倍；

**⚠️ 局限性**

局限包括：缺乏人工金标；评估面板依赖模型一致性，结果受判断者差异影响；短文篇幅小样本导致信度低；种子方差大，需多种子；蒸馏过程为三阶，无法区分模型容量与教师规模对性能的具体贡献。

---

## 277. Structured Pruning of Large Language Models via Power Transformation and Sign-Preserving Score Aggregation with Adaptive Feature Retention

**arXiv ID:** 2607.08027 | [PDF](https://arxiv.org/pdf/2607.08027v1)

**作者:** Ryota Kobayashi `[一作]` (Chubu University), Kazuki Kozuka `[通讯]` (Panasonic Holdings Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种改进的结构化剪枝方法，将Unstructured AFR（ReFer+SNIP）的剪枝分数迁移到神经元级别，并通过幂变换、符号保留聚合和百分位异常值去除来解决分布不匹配、符号信息丢失和异常值影响问题。

**💡 创新点**

三大创新点：①使用幂变换对SNIP分数进行非线性分布对齐；②在聚合时保留符号信息以评估神经元内部优化方向一致性；③对每个神经元的权重剪枝分数做2%–98%百分位去除异常值。

**🔧 技术方法**

技术手段包括：幂变换(power transformation)、z-score标准化、符号保留的平均聚合、百分位异常值去除、ReFer与SNIP分数的组合，以及对FFN层进行结构化剪枝。

**📊 数据集**

实验数据集包括语言任务基准（WinoGrande、HellaSwag、ARC-e、ARC-c、MMLU）和视觉语言基准（GQA、VizWiz、ScienceQA），模型分别为Llama‑3‑8B、Vicuna‑v1.5‑13B和LLaVA‑v1.5‑13B。

**📈 对比分析**

通过与无结构AFR、naive‑average结构化AFR、LLM‑Pruner、LoRAP和CFSP等方法对比，20%剪枝时对naive‑avg提升约21.3/15.1分，50%剪枝可实现35.1%参数压缩和1.57×推理速度提升，整体性能几乎与无结构AFR持平。

**⚠️ 局限性**

局限性在于仅对FFN层进行结构化剪枝，未扩展到注意力层；对任务梯度缺失的模块（如CLIP的视觉部分）只能使用ReFer；速度提升仍低于理论最大值；幂变换与阈值选择需针对不同模型调优，缺乏统一的自动化机制。

---

## 278. Classifier Chain-based Pathological Test Recommendation

**arXiv ID:** 2607.08299 | [PDF](https://arxiv.org/pdf/2607.08299v1)

**作者:** Abu Rafe Md Jamil `[一作]` (Jashore University of Science and Technology), Nayan Malakar `[通讯]` (Jashore University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种基于分类链（Classifier Chain）的多标签病理检验建议系统，利用患者自述症状预测需要的检验项目，并通过多数投票集成提升鲁棒性。

**💡 创新点**

创新点在于首次将多标签学习与分类链相结合，对检验项目间的依赖关系建模，同时引入可解释人工智能（SHAP）验证模型决策与临床经验的一致性。

**🔧 技术方法**

采用的技术包括Classifier Chain、逻辑回归、决策树、随机森林等基分类器，以及多数投票集成和SHAP可解释性分析。

**📊 数据集**

使用了自建的SOUTHERN.IML病理数据集，该数据集包含143种症状特征和80种检验项目标签（共223列），样本由临床专家协同标注。

**📈 对比分析**

通过对比单独基分类器、其分类链版本以及多数投票集成模型，结果显示逻辑回归+分类链取得最高准确率（98.83%），随机森林+分类链和多数投票+分类链分别达到98.49%和98.53%，显著优于传统单标签方法。

**⚠️ 局限性**

局限性包括数据来源单一地区、缺乏多机构外部验证、对极度不平衡标签的处理仍有限，以及解释性仅依赖SHAP，尚需进一步扩展至更复杂的深度多标签模型和其他可解释技术。

---

## 279. Predicting Pseudo-nitzschia harmful algal blooms along the Portuguese Coast using satellite-derived predictors

**arXiv ID:** 2607.07834 | [PDF](https://arxiv.org/pdf/2607.07834v1)

**作者:** Ayman Bnoussaad `[一作]` (Institute for Systems and Robotics), Alexandre Bernardino `[通讯]` (Associação do Instituto Superior Técnico para a Investigação e Desenvolvimento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

建立并评估了一个基于卫星观测的空间-时间机器学习框架，用以预测葡萄牙大西洋沿岸Pseudo‑nitzschia藻华的出现。

**💡 创新点**

创新点在于使用河流感知的空间聚类和严格的时间-空间交叉验证，避免信息泄露，并首次将生物学指示器（叶绿素‑a、浮游生物功能类型）与环境驱动共同用于遥感基准的HAB预测。

**🔧 技术方法**

采用的技术包括随机森林、极端随机树、XGBoost等集成树模型，以及特征工程、时序延迟、正弦余弦季节编码、特征重要性分析和阈值依赖的误差矩阵。

**📊 数据集**

使用的数据集为2013‑2023年间IPMA的Pseudo‑nitzschia细胞浓度和渔业禁渔记录，配合Copernicus Marine Service的海表温度、上升指数、叶绿素‑a和浮游生物功能类型，以及EMODnet的金属浓度。

**📈 对比分析**

通过河流感知的6个空间簇与11个年份交叉验证（66个fold），比较了不同模型和特征组合，最佳性能为Extra‑Trees在环境+生物特征下的ROC‑AUC 0.77 ± 0.06，显示相较于仅环境特征可提升约0.03。

**⚠️ 局限性**

限制包括仅预测藻华出现而非毒素浓度、使用固定0‑45天滞后而非序列模型、模型只针对北部热点区，需要在其他生产区重新聚类验证。

---

## 280. Eigenvalue Calibration for Semantic Embeddings of Large Language Models

**arXiv ID:** 2607.08377 | [PDF](https://arxiv.org/pdf/2607.08377v1)

**作者:** Sebastian G. Gruber `[一作]` (KU Leuven), Florian Buettner `[通讯]` (German Cancer Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对大型语言模型（LLM）语义嵌入的特征值校准框架，将 LLM 生成答案的语义嵌入视为密度矩阵预测器，并通过温度缩放调整其最大特征值以降低校准误差。

**💡 创新点**

创新点在于：①将密度矩阵预测与特征值校准引入 LLM 的不确定性估计；②在校准理论上建立了熵‑风险等价性与特征值校准不等式；③证明温度缩放在最大特征值上最小化正规化风险即实现最佳校准；④提出了适用于特征值的可靠性图绘制算法。

**🔧 技术方法**

主要技术包括：密度矩阵预测器构造、特征值温度缩放（谱函数），正则化风险最小化，Bregman 矩阵散度，熵‑风险等价性证明，以及基于层次聚类的特征值可靠性图算法。

**📊 数据集**

使用了 TriviaQA 与 Natural Questions 两个公开问答数据集；LLaMA4 Maverick、Phi-4、Phi-4 Mini 三种开源 LLM；语义嵌入模型 all‑mpnet‑base‑v2。

**📈 对比分析**

通过与未校准模型、基线语义熵、核语言熵以及不同温度下的风险和 ECE 进行对比。实验表明：①所有模型在最大特征值上均表现出系统性过度自信；②在风险最小化的温度下，最大特征值校准误差（ECE）显著下降；③校准后的模型在答案正确性检测（AUROC）上得到提升，且在大多数设置中优于无校准和基线方法。

**⚠️ 局限性**

局限性包括：①仅对最大特征值进行校准，未考虑其他特征值信息；②温度缩放假设在谱函数上可逆且可注入，可能在高维密度矩阵上效果有限；③依赖于语义嵌入模型的质量，嵌入偏差可能影响校准；④聚类和可靠性图算法对计算资源有一定要求；⑤实验仅覆盖少数 LLM 与数据集，需进一步验证在更广泛模型与任务上的泛化。

---

## 281. XALPHA: A Memory-Driven AI Quant Researcher for Hypothesis-to-Code Alpha Discovery

**arXiv ID:** 2607.08332 | [PDF](https://arxiv.org/pdf/2607.08332v1)

**作者:** Fengyuan Liu `[一作]` (University of Hong Kong), Qi Liu `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套端到端的AI量化研究者，实现从财经报告吸收、记忆驱动的研究规划、假设到可执行因子代码的演化、三方对齐验证以及经验反馈闭环的Alpha挖掘流程。

**💡 创新点**

创新点在于：①将外部财经报告通过Report‑to‑Memory Absorption层转换为可检索的多层A/B/C知识记忆；②以记忆驱动的宏脑规划主题与Archetype路由；③微脑执行代码级因子演化并通过静态AST门、三方对齐和动态未来泄漏检验确保因子合理；④跨脑反馈归纳为GOOD/BAD记忆，形成多级闭环。

**🔧 技术方法**

使用技术包括：大型语言模型（如GPT）生成规划与代码；结构化记忆与检索；多脑架构（宏脑、微脑、交叉脑）；代码演化算法（变异、交叉、精炼）；静态与动态质量管道；传统机器学习与深度学习模型做对照；因子评估指标（IC、RankIC、ICIR、RankICIR、AR、AER、IR）。

**📊 数据集**

数据集为中国A股CSI300指数的日频数据，预测目标为下一日开盘价至第10日开盘价的10日未来回报，实验分为训练（2011‑2020）、验证（2021）、测试（2022‑2025）。

**📈 对比分析**

通过与多类基准（传统机器学习、深度学习、Alpha挖掘系统）在同一数据集、相同预测任务下进行比较。实验结果显示，本文方法在IC、RankIC、ICIR、RankICIR、年化收益、超额收益及信息比率等指标均显著优于所有基准，尤其在组合表现和稳健性方面优势突出。

**⚠️ 局限性**

局限性包括：仅在单一指数和单一预测时段验证，未展示跨资产或多时段的泛化；对实时交易环境的适用性、风险控制与执行成本仍待验证；对财经报告文本的语义理解仍可能产生误解，影响记忆质量；整体计算成本相对较高。

---

## 282. PIT-SUN: A Deployable Empirical Marginal Transform Framework with Expectation-Consistent Recovery for Regression in Recommender Systems

**arXiv ID:** 2607.08202 | [PDF](https://arxiv.org/pdf/2607.08202v1)

**作者:** Mingyu Zhao `[一作]` (Renmin University of China), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证一种统一的经验-分位变换与条件线性恢复框架PIT‑SUN，用以在稀疏、重尾或多峰回归任务中恢复原空间期望并提升预测精度、校准与排序质量。

**💡 创新点**

①证明非线性尾部压缩无法直接保持期望一致性，提出需分离坐标与恢复路径的闭包设计；②使用单一经验累计分布表同时定义PIT坐标、逆分位基、下限稳定的恢复基和漂移监测；③引入停止梯度的SUN恢复和下限阈值b_min来控制比例标签方差。

**🔧 技术方法**

经验PIT映射、标准正态分位变换、逆分位查找、条件线性恢复（SUN）、梯度隔离、下限阈值b_min、零膨胀分离（PIT‑SUN‑ZI）等；实验采用多任务/多尺度神经回归模型并对比多种基线。

**📊 数据集**

12个合成数据集；公开基准（CIKM16、DTMart）；匿名工业基准（Indus）；两大规模真实数据集（长尾观看时长、零膨胀GMV）；以及在线A/B测试流量。

**📈 对比分析**

与MSE、T‑MSE（对数、平方根）、TranSUN/GTS、分布式模型（ZILN、WLR、MDME）、分桶/结构化模型（TPM、CCOR‑Net）、以及PIT‑ONLY与PIT‑TRANSUN等对比。评估指标包括NMAE、NRMSE、SRE、TRE、MRE、PGR、xAUC、Gini、Spearman等。PIT‑SUN在所有数据集均实现最低点误差、最佳校准与最高排序分数，在线AB测试则提升观看时长、视频观看次数和深度参与用户。

**⚠️ 局限性**

需手动维护经验CDF表并更新；对极端零膨胀场景仍需额外分支；下限阈值b_min、裁剪概率δ等超参数需要调优；对极端多峰或非连续分布的鲁棒性尚未完全验证。

---

## 283. Post-Training in End-to-End Autonomous Driving

**arXiv ID:** 2607.08072 | [PDF](https://arxiv.org/pdf/2607.08072v1)

**作者:** Ruining Yang `[一作]` (Northeastern University), Lili Su `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了端到端自动驾驶后训练（post‑training）技术，提出统一的定义与四大分类（蒸馏、偏好对齐、强化学习、推理时精调），并梳理了当前方法的关键技术、评价指标与实验结果；

**💡 创新点**

创新点在于：①首次将后训练定义为独立阶段并与初始模仿学习区分；②构建了基于监督形式的系统化分类框架；③汇总并对比了多种后训练技术在主流基准上的表现；④提出了评估难点与未来研究方向；

**🔧 技术方法**

主要技术包括：模型蒸馏（teacher‑policy 及 VLM 监督）、偏好对齐（DPO 等）、基于奖励的强化学习（GRPO、PPO 等）、推理时候的候选搜索与验证器；评估方法覆盖开环误差、碰撞率、RFS 等指标，闭环评测使用 NAVSIM、Bench2Drive、AlpaSim 等仿真平台；

**📊 数据集**

使用的数据集与基准包括 nuScenes、Waymo Open Dataset (WOD‑E2E)、NAVSIM（v1/v2）、Bench2Drive、AlpaSim、nuReasoning 等；

**📈 对比分析**

通过对比表格呈现不同后训练方法在开环（L2/ADE、Col、RFS）和闭环（PDMS/EPDMS、Driving Score/Success Rate）指标上的得分，展示了各技术在安全、舒适、进度等维度的优势与差距；

**⚠️ 局限性**

局限性主要体现在：评估基准趋于饱和，难以细粒度区分方法优劣；闭环测试仍缺乏真实车辆验证与可复现性；推理成本与实时性未得到充分报道；后训练的数据与反馈循环尚不系统化；

---

## 284. Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data

**arXiv ID:** 2607.08103 | [PDF](https://arxiv.org/pdf/2607.08103v1)

**作者:** Chaewon Lee `[一作]` (Korea University), Chang-Su Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个在存在标签噪声下进行秩估计的随机排序学习框架（SOL）

**💡 创新点**

创新点在于将秩估计重新表述为随机排序问题，并通过同时考虑实例-质心的判别损失和随机排序损失来建模标签不确定性

**🔧 技术方法**

使用深度编码器（VGG16）+ 质心更新、判别损失、随机排序损失，以及噪声检测与重新标注机制

**📊 数据集**

在面部年龄估计（MORPH2、CLAP2015）、审美评分（AADB）、医学骨龄（RSNA）和文本回归（WMT2020）等多种数据集上进行实验

**📈 对比分析**

与多种噪声鲁棒分类、回归和秩估计方法相比，SOL在MAE/CSS等指标上持续领先，尤其在高噪声水平下性能提升显著

**⚠️ 局限性**

局限性包括对高维嵌入空间的计算开销、对噪声分布假设的依赖以及在极端噪声或多模态数据上的鲁棒性待进一步验证

---

## 285. Workload-Preserving Differentially Private Synthetic Data for Causal Inference via Maximum-Entropy Calibration

**arXiv ID:** 2607.08122 | [PDF](https://arxiv.org/pdf/2607.08122v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Kaveh Aryan `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出针对因果推断的差分隐私（DP）合成数据方法，设计了基于正交矩（orthogonal moments）的因果工作负载，并通过最大熵重构得到可复用的合成记录，同时提供噪声感知多重插补（NA+MI）来实现置信区间。

**💡 创新点**

创新点包括：①把因果估计所需的正交矩直接作为DP查询集合，保证了因果效应所需的协方差结构；②提出自适应的Causal‑AIM工作负载选择器，按因果效用动态选取特征；③将NA+MI方法与DP合成数据结合，给出理论覆盖率保证；④在同一DP合成数据上支持ATE、ATT、亚组效应等多种因果分析，无需额外隐私支出。

**🔧 技术方法**

使用的技术包括：差分隐私高斯机制、最大熵（信息投影）重构、稳健的正交矩估计（双重稳健/IPW/AIPW）、噪声感知多重插补、正交矩的稀疏化与阈值化、以及自适应查询选择（Causal‑AIM）和稳定性分析。

**📊 数据集**

实验数据集包括：IHDP、Twins、ACIC 2016（DGP 7）、LaLonde/NSW、以及美国加州2018年ACS半合成数据。

**📈 对比分析**

与基线方法（非隐私DR、MST+naive DR、AIM+naive DR、因果工作负载+naive DR、因果工作负载+NA+MI、Causal‑AIM+NA+MI）比较。结果显示：①在隐私预算较宽松时，通用工作负载（MST、AIM）在RMSE上往往优于因果工作负载；②在严格预算下，因果工作负载+NA+MI在覆盖率上几乎完全达到95%置信水平，且RMSE可与MST匹敌；③因果工作负载+NA+MI提供了可复用的合成表，可一次性支持ATE、ATT和亚组效应；④NA+MI区间宽度显著高于naive方法，体现了对DP噪声的正确校准。

**⚠️ 局限性**

局限性包括：①需预先选择特征映射ϕ，若映射不足会导致近似偏差；②在高隐私预算下区分度降低，区间过宽；③方法主要基于AIM/Private‑PGM最大熵重构，其他重构框架未验证；④适用于离散/分段特征，连续特征的扩展仍需研究。

---

## 286. FedTR: Federated Learning Framework with Transfer Learning for Industrial Visual Inspection

**arXiv ID:** 2607.08014 | [PDF](https://arxiv.org/pdf/2607.08014v1)

**作者:** Vikash Sathiamoorthy `[一作]` (Nanyang Technological University), Weichen Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 FedTR 框架，结合联邦学习与迁移学习，在工业视觉检测中完成端到端的文本识别任务，实现在不共享数据的前提下提升多工厂模型性能。

**💡 创新点**

创新点包括：①将迁移学习与联邦学习整合，先在公开数据上预训练，再在各工厂私有数据上联邦微调；②提供完整的检测‑识别两阶段联邦训练流程；③在有限且异构的数据环境下实现与集中训练相当的性能，提升了联邦学习的实用性。

**🔧 技术方法**

采用的技术有：联邦学习（FedAvg聚合算法）、迁移学习（fine‑tuning）、深度学习模型（Faster R‑CNN、YOLOv7 目标检测；TRBA 文本识别），使用 PyTorch 框架实现。

**📊 数据集**

使用的数据集：公开的 SynthText in the Wild（合成文本图像），以及两代墨盒标签数据（Ink Cartridge Generation I 与 Generation II），分别包含不同视角与缺陷标签。

**📈 对比分析**

与单机训练和集中训练进行对比；在同质数据集上 FedTR 检测 F1 为 0.726、识别准确率 0.991，端到端词级准确率 95.5%；在异质数据集上 FedTR 检测 F1 约 0.71、识别准确率约 0.99，端到端词级准确率 94.2%；总体性能与集中训练相当或略低，但在不共享数据的场景下实现了可接受的效果。

**⚠️ 局限性**

局限性：对极端非 i.i.d. 数据仍易导致性能下降；需要多轮通信，通信成本较高；对模型规模和算力要求较大；目前仅在文本识别任务上验证，其他任务的通用性待进一步探索。

---

## 287. Provably Optimal Learning Algorithms for Assistance Games

**arXiv ID:** 2607.08012 | [PDF](https://arxiv.org/pdf/2607.08012v1)

**作者:** Nivasini Ananthakrishnan `[一作]` (University of California, Berkeley), Nika Haghtalab `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并分析了在线援助游戏的框架，设计了去中心化学习算法，使得人类与助手在信息不对称的情况下实现接近最优协作；

**💡 创新点**

1) 证明了援助游戏可转化为在线子模函数最大化（带矩阵约束）的形式，利用子模结构实现可行的近似；2) 引入“援助回报”概念和“援助悔恨”衡量标准；3) 首次给出去中心化算法的理论保证，证明在 (1-1/e) 近似下可获得 T^{3/4} 或 √T 的下界；4) 证明 1-1/e 是最优可实现的近似因子，超越此因子则不可多项式时间求解；

**🔧 技术方法**

在线子模最大化（Weighted Threshold Potentials + Matroid约束）、随机增量OCO（RAOCO）、稳定OCO（POMER）、跟踪误差算法（Fixed‑Share）以及对随机化取整的耦合技术；

**📊 数据集**

无实验数据集，全部为理论分析与证明；

**📈 对比分析**

与传统基于POMDP或无监督学习的援助方法相比，本工作给出最优的下界（(1-1/e)近似、√T 或 T^{3/4} 的援助悔恨），并证明无法在多项式时间内获得更好的近似；

**⚠️ 局限性**

1) 对动作空间大小的依赖仍可能不够紧凑；2) √T 的速度需要预先共享的编码映射，实际部署时可能不可行；3) 适用范围限定在一次性偏好（oblivious）设置，适应性环境下尚未给出可行解；4) 只关注单步回报，未讨论长期决策或多步动态。

---

## 288. OmniFood-Bench: Evaluating VLMs for Nutrient Reasoning and Personalized Health Advice

**arXiv ID:** 2607.08423 | [PDF](https://arxiv.org/pdf/2607.08423v1)

**作者:** Qian Jiang `[一作]` (Northeastern University at Qinhuangdao), Miao Fang `[通讯]` (Northeastern University at Qinhuangdao)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 OmniFood-Bench 基准，用三层评估（感知、定量推理、安全建议）衡量 VLM 在食品识别、重量估计和医学建议方面的表现。

**💡 创新点**

创新点在于：①构建了覆盖餐厅、家庭、包装、原料四类且包含重量、营养和医疗标签的高质量数据集；②将视觉推理与医学安全对齐，形成“语义-物理鸿沟”与“安全幻觉”评估框架；③首次对现有六种 VLM 在高风险饮食场景中进行系统对比。

**🔧 技术方法**

使用了多模态大模型技术（GPT-5.1、Gemini-3-Flash、Claude‑Sonnet‑4、Qwen3‑VL‑8B、InternVL3‑5‑8B、Llama‑3.2‑11B‑Vision），通过零样本提示与多任务评估流程；引入 MAPE、准确率、SFR 等指标。

**📊 数据集**

基准数据集来源于 MM‑Food‑100K 的 1,208 份高质量样本，手工标注了菜品类别、烹饪方法、成分列表、重量、宏量营养素及疾病安全标签。

**📈 对比分析**

方法：对比六种模型在三层任务中的分数。结果显示模型在识别任务上达约 70–90% 的准确率，但在重量和营养估计的 MAPE 均超过 50%，安全建议准确率低至 30–46%，表明目前模型在高风险情境下不安全。

**⚠️ 局限性**

局限：①缺乏尺度参照导致重量估计误差大；②医学逻辑模块未充分对齐临床指南，易产生安全幻觉；③基准样本数量有限，难以覆盖更丰富的烹饪和包装变异；④对多模态数据的鲁棒性与跨域泛化尚未验证。

---

## 289. A Theoretical Framework for Stochastic Activity Prediction in Tensor Accelerator Wallace-Tree Multipliers

**arXiv ID:** 2607.08002 | [PDF](https://arxiv.org/pdf/2607.08002v1)

**作者:** Prashanthi Metku `[一作]` (Qualcomm Technologies Inc), Chandra Gandu `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于斯托卡斯特活动预测（SAP）的壁虎树乘法器动态功耗抑制框架，利用Hamming权重统计提前冻结乘法器输入以降低内部开关活动。

**💡 创新点**

创新点在于严谨证明壁虎树活动仅取决于位权密度，给出单比特伯努利代理的最优性以及误差、信息保持的理论上限，并引入安全控制器保证算术结果无误。

**🔧 技术方法**

使用异或计数（popcount）、伯努利编码、Lipschitz与信息理论证明、以及安全控制器和多层功耗栈集成的技术。

**📊 数据集**

未给出具体实验数据集，假设典型的INT8量化神经网络推理工作负载。

**📈 对比分析**

尚未进行仿真或硅验证；理论证明误差低于10^-13，信息保持率≥1‑O(log n/n)，预期在稀疏输入上实现显著功耗削减。

**⚠️ 局限性**

主要局限是缺乏经验验证，常数需通过门级仿真确定；未探讨多比特或非校准编码的潜在改进。

---

## 290. GIRAF: Towards Generalizable Human Interactions with Articulated Objects

**arXiv ID:** 2607.07880 | [PDF](https://arxiv.org/pdf/2607.07880v1)

**作者:** Xiaohan Zhang `[一作]` (Tübingen AI Center, University of Tübingen), Yuting Ye `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于文本的扩散模型，用于生成在可耦合物体上进行完整身体交互的长时序运动。

**💡 创新点**

核心创新包括：统一的物体中心动态基准点表示实现细粒度手物接触跨形状泛化；混合域训练策略平衡步态与交互；以及基于接触的增广方案提升样本多样性。

**🔧 技术方法**

技术实现主要依赖 Transformer 编码的扩散网络、FiLM 条件层、动态基准点集、以及场景感知的噪声优化。

**📊 数据集**

使用 ParaHome（包含可耦合家居物体交互与文本描述）与 Babel（步态片段）混合的数据集。

**📈 对比分析**

与 LINGO 与 CHOIS 两大基线相比，模型在接触距离、穿透度、姿态误差、以及文本对运动的对齐指标（R‑precision、FID）上均取得显著提升。

**⚠️ 局限性**

局限性包括：仅支持简单平坦地形，偶尔出现脚滑或关节抖动；对旋转型等更复杂耦合机制的泛化有限；以及未涵盖更高级的导航与交互情景。

---

## 291. Deep Learning Method for Stationary Distribution of Reflected Brownian Motion

**arXiv ID:** 2607.08091 | [PDF](https://arxiv.org/pdf/2607.08091v1)

**作者:** Jim Dai `[一作]` (Cornell University), Zhanhao Zhang `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于深度学习的框架，用来学习高维反射布朗运动（RBM）稳态分布的拉普拉斯变换，从而能够推算尾概率等性能指标。

**💡 创新点**

创新点在于将BAR（Basic Adjoint Relationship）转化为可训练的损失函数，并设计了结构化损失（对数化、对称性、单调性、Cauchy‑Riemann约束）、针对高维的采样策略以及共享权重的网络架构，使得参数规模与维度无关。

**🔧 技术方法**

采用了神经网络对拉普拉斯变换的对数进行建模，利用傅里叶特征与坐标/边界嵌入共享编码器，加入多项正则化约束，使用Talbot数值逆变换评估尾概率，整体实现了端到端的训练与推断。

**📊 数据集**

使用了三个RBM实例进行实验：一个2维无闭式拉普拉斯变换的RBM（通过数值积分获得真值），以及20维和30维具有乘积形式拉普拉斯变换的RBM（直接用闭式变换求真值）。

**📈 对比分析**

通过将网络学习到的拉普拉斯变换与Talbot逆变换得到的尾概率进行对比，实验显示在所有维度下预测值几乎与真值完全吻合，尤其在20维和30维高维实例中保持了极高的准确性。

**⚠️ 局限性**

局限性在于每次梯度更新需使用约16,384个样本，导致显存占用或训练时间随维度增长显著；目前仅验证于RBM，尚未扩展到更广泛的随机系统或更大维度（数百或千维）。

---

## 292. PERFOPT-Bench: Evaluating Coding Agents on Software Performance Optimization

**arXiv ID:** 2607.07744 | [PDF](https://arxiv.org/pdf/2607.07744v1)

**作者:** Yingyun Cui `[一作]` (OPPO Research Institute), Liangliang Cao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了PERFOPT-Bench基准，评估编码代理在真实硬件上完成完整性能优化循环的能力。

**💡 创新点**

创新点在于将跨层性能优化、隐藏正确性检查、可验证加速度量与代理框架的组合用于基准评估，并揭示代理框架对同一LLM的加速效果有显著影响。

**🔧 技术方法**

使用大型语言模型、代理框架、自动化任务生成管线、测量脚本以及Relay技术来驱动和验证优化过程。

**📊 数据集**

基准包含12个手工设计的性能优化任务（代码和评测脚本公开于匿名链接 <https://anonymous.4open.science/r/Dataset-D3CC>）。

**📈 对比分析**

通过比较7种代理堆栈在12任务上的单轮和双轮（Relay）实验，使用验证后速度提升作为评分；结果显示无单一堆栈始终占优，Relay可在1.0–2.5倍范围内提升加速效果。

**⚠️ 局限性**

局限性包括：受硬件/编译器/运行时差异影响；单次实验噪声大，未进行重复抽样；shortcut审计有限；Relay实验缺乏完整对照和预算控制；基准仅覆盖性能优化而非通用编码能力。

---

## 293. D-CLIPSE: Distributed Consensus-based Localization with Passive Listening on Shared State Exchange

**arXiv ID:** 2607.07995 | [PDF](https://arxiv.org/pdf/2607.07995v1)

**作者:** Kyle Biron-Gricken `[一作]`, James Richard Forbes `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通信高效、基于一致性的多机器人分布式滤波框架，利用预积分里程计和共享状态实现团队定位。

**💡 创新点**

创新点：只共享相关共享状态而非全状态，采用协方差插值共识更新并引入被动监听提升一致性；通过对称共识构造最优共享状态，避免了全状态CI的冗余与过度保守。

**🔧 技术方法**

使用了基于矩阵李群的扩展卡尔曼滤波、协方差插值（CI）、BCH近似、预积分里程计（RMIs）以及被动监听共识更新。

**📊 数据集**

实验数据集包括仿真二维机器人（N=3,5,9,17等）与 MILUV 三架室内无人机数据集。

**📈 对比分析**

与中心化估计和现有最优分布式方法（SoTA）比较，准确度与一致性（NEES、±3σ bounds）均接近中心化；2-Wasserstein 距离显示分布更接近中心化；总体性能优于 SoTA，通信量更小。

**⚠️ 局限性**

局限性：通信拓扑假设为静态，需预先同步初始共享状态；对动态拓扑或更大规模团队的适用性尚未验证；方法在多传感器混合环境下的鲁棒性仍需进一步研究。

---

## 294. DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment

**arXiv ID:** 2607.07820 | [PDF](https://arxiv.org/pdf/2607.07820v1)

**作者:** Xinyu Geng `[一作]` (Hong Kong University Of Science And Technology), Yi R. Fung `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于离线 Wikipedia 的可验证工具环境，并通过自蒸馏框架实现深度搜索代理的自我改进。

**💡 创新点**

提出了可验证的离线工具环境与基于过程监督的自蒸馏方法，利用可核查的进度和反思信号克服传统 RL 稀疏奖励与监督有限的瓶颈。

**🔧 技术方法**

采用了 Deterministic Offline Environment、ReAct 格式、自蒸馏（Evolving SFT）、轨迹过滤、重要采样、GRPO 微调以及 Qwen3.5-9B 作为基线模型。

**📊 数据集**

构造了 420K 多跳问答任务（DeepSearch‑World）及验证集 DeepSearch‑Val，并在七个深度搜索基准上评测。

**📈 对比分析**

与开源与专有代理在 BrowseComp、GAIA、HotpotQA 等基准对比，-9B 在多数任务上实现 +23.8%~+48.1% 的提升，达到与强大开源模型相当的性能。

**⚠️ 局限性**

仅覆盖 Wikipedia 知识域，缺乏更广泛的可验证数据源，且依赖 Evolving SFT，未探索 RL 或更高级的规划强化学习。

---

## 295. WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving

**arXiv ID:** 2607.08375 | [PDF](https://arxiv.org/pdf/2607.08375v1)

**作者:** Xuerun Yan `[一作]` (Tongji University), Binyang Song `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出WCog-VLA框架，将语义层和生成层双层世界认知用于端到端自主驾驶。

**💡 创新点**

创新点包括：① 双层世界认知整合语义预测与生成演化；② 引入Aligned Decoupled Diffusion Transformer（ADDT）生成多智能体轨迹；③ 开发Game-CoT数据集与游戏理论链式推理；④ 采用四阶段训练策略。

**🔧 技术方法**

技术手段包括：InternVL3 2B + Qwen2.5 LLM 作为VLM基座；BEVFormer 与 TrackFormer 提供3D感知；Game-Theoretic Chain-of-Thought 推理；ADDT 结合对齐与解耦的扩散Transformer；AdaLN 位置调制；GRPO 强化细化；多模态 VQA 训练。

**📊 数据集**

使用的数据集：NAVSIM v1/v2 真实仿真；158k 公共 VQA 数据（DriveLM、CODA-LM、LingoQA、nuScenes-QA、NuInstruct、DriveGPT4）；170k NAVSIM‑定制样本（85k 轨迹 VQA + 85k Game‑CoT）；Game‑CoT 85k 标注；以及 3D 传感器数据。

**📈 对比分析**

在 NAVSIM v1/2 上与多种 E2E、VLM‑Plan、Diffusion 等方法对比，WCog‑VLA‑2B 取得 PDMS 92.9、EPDMS 85.9，显著超越 WoTE、DiffusionDrive、ReCogDrive、LatentVLA 等基线，安全性（NC、TTC）与舒适度也明显提升；推理速度提升 10× 以上。

**⚠️ 局限性**

局限性：当前语义认知仅聚焦于周边车辆/行人，未涵盖道路几何、地图拓扑等环境演化，需要进一步构建更完整的世界模型。

---

## 296. Time-to-Collision Based Dynamic Obstacle Avoidance Using Pretrained Vision Models for Robots in Unstructured Environments

**arXiv ID:** 2607.07885 | [PDF](https://arxiv.org/pdf/2607.07885v1)

**作者:** Erik Jagnandan `[一作]` (University of Pennsylvania), Pratik Chaudhari `[通讯]` (United States Army Research Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个完全基于预训练视觉模型、无需仿真或大规模机器人专属数据训练的实时动态障碍物回避系统。

**💡 创新点**

通过将UniDepth的单目深度估计、SuperPoint+SuperGlue的长时特征追踪与XM Solver的束调整相结合，利用每个关键点的时间到碰撞（TTC）实现可解释且数据高效的障碍物检测与躲避。

**🔧 技术方法**

使用了预训练的UniDepth、SuperPoint、SuperGlue、XM Solver（束调整）以及基于TTC的二维运动原语生成技术。

**📊 数据集**

在M3ED（Spot森林与城市场景）数据集上进行评估与验证。

**📈 对比分析**

与LiDAR基准TTC对比，帧级检测精度为0.49、召回为0.38；在检测到危险帧时，84% 的躲避方向与地面真实动作一致；误报率低（0.47%）。

**⚠️ 局限性**

主要局限在于召回不足，原因是特征追踪持续时间短、UniDepth深度误差导致束调整失准；单目深度在低纹理或光照变化时不稳定；未完成完整机器人平台集成与实时性评估。

---

## 297. Unified Face Attack Detection via Fine-Grained Semantic Guidance

**arXiv ID:** 2607.08156 | [PDF](https://arxiv.org/pdf/2607.08156v1)

**作者:** Ning Jiang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出在面部攻击检测中利用细粒度文本描述增强模型表现，并对MS-UFAD数据集进行细粒度注释扩充，构建Dual Alignment Forgery Network (DAF-Net)实现视觉与文本的多层次对齐。

**💡 创新点**

创新点包括：①将8M张面部攻击图像配上细粒度文本注释，形成大规模多模态数据；②设计Dual Alignment Forgery Network与Semantic Forgery Aggregation Module (SFAM)，在训练阶段使用全球与细粒度跨模态对齐损失；③在推理时仅使用视觉分支，保持部署高效。

**🔧 技术方法**

技术手段包括：多模态大语言模型生成文本注释、ScoreCAM定位伪造区域、BLIP模型的视觉/文本编码器、Transformer + Cross-Attention聚合模块、对比与三元组损失实现跨模态对齐、Log‑Sum‑Exp池化以及多尺度性能评估。

**📊 数据集**

使用的数据集为已扩充的MS‑UFAD（约8M张图像+细粒度文本），训练集为830k样本（2k身份，10种生成方法），测试集为约3.6M样本（30种生成方法），最终以201,453对样本进行评估。

**📈 对比分析**

与传统视觉模型（ResNet50、ViT、BLIP‑ViT）、BLIP‑ViT+SFAM以及使用粗粒度/类别级文本的对齐方法比较，DAF‑Net在ACER、Accuracy和F1上分别取得12.30/90.34/87.13，显著优于所有基线（例如粗粒度文本提升至15.24/88.31/85.51，细粒度文本提升至12.30/90.34/87.13）。

**⚠️ 局限性**

局限性包括：①生成的文本注释仍可能出现“幻觉”或定位误差；②训练阶段需要大量视觉与文本对齐计算，计算成本高；③在推理时去掉文本分支可能导致对某些极端攻击的鲁棒性下降；④数据集虽大但仍受制于已知生成方法，未覆盖所有潜在攻击场景。

---

## 298. Nigeria Machinery: A Low-Resource Industrial Dataset with a Domain-Grounded Reasoning Layer

**arXiv ID:** 2607.07883 | [PDF](https://arxiv.org/pdf/2607.07883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 299. AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidance

**arXiv ID:** 2607.08341 | [PDF](https://arxiv.org/pdf/2607.08341v1)

**作者:** Chenxi Wang `[一作]` (Noematrix), Cewu Lu `[通讯]` (Noematrix)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无标定的手指对应重映射方法 AnyDexRT，利用自监督形状匹配、极少量手动锚点和接触分类器实现多种人类类手指的直观遥操作。

**💡 创新点**

创新点包括：① 采用局部 Chamfer 与距离保持正则化，避免全局覆盖导致的畸变；② 用极少的手动锚点解决映射歧义；③ 引入接触分类器细化抓握姿势，整体实现跨手结构、无标定的遥操作。

**🔧 技术方法**

使用的技术包括：自监督几何形状匹配、局部坐标运动保持、距离保留损失、少量锚点对齐损失、二分类接触识别器（BCE）以及逆运动学/最近邻求解；实验中对多种手模型和实机平台进行验证。

**📊 数据集**

使用的数据集：七种人类类 Dexterous Hand（手指轨迹与机器人指尖数据）以及在实机平台上收集的四个遥操作任务（Sprink、Screw、Shovel、Pick-10），未引用公开数据集。

**📈 对比分析**

与传统优化重映射方法和 GeoRT 进行比较；在七种手上，局部运动一致性从 59.8% 提升至 90.2%，全局一致性保持竞争；运行频率约 300 Hz，手动调参仅需 3 个超参数；对坐标旋转误差鲁棒，实机任务中完成时间最短、Pinch 成功率最高。

**⚠️ 局限性**

局限性：仍需手动提供少量锚点，未实现完全自动化；接触优化仅针对抓握抓取，缺乏更全面的接触模型；评估仅停留在遥操作效果，未验证下游操控策略的提升。

---

## 300. Aleena: Alignment Agent for Research Software Engineering Collaborations

**arXiv ID:** 2607.08043 | [PDF](https://arxiv.org/pdf/2607.08043v1)

**作者:** Kshitij Dani `[一作]` (University of Washington), Anant Mittal `[通讯]` (University of Washington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一个名为 Aleena 的生命周期对齐智能体，利用 GitHub 作为协作平台，将会议、聊天、issue 等多模态信息转换为结构化的 GitHub 记录（摘要、风险、术语漂移、开放问题等），帮助科研团队在整个研发周期内保持共享理解和决策连续性。

**💡 创新点**

创新点在于将对齐视为持续的项目状态管理，而非一次性 scoping；通过 Agentic 四步循环（感知、更新、选择、推迟）结合 LLM 生成结构化输出，并保留人类决策权，构建了专门针对研究软件工程的对齐生态。

**🔧 技术方法**

使用技术包括：基于 LLM 的分析器（LiteLLM 接口调用多模型）、FastAPI + React 前端、GitHub OAuth/Apps、GitHub API 写入 issue、讨论、草稿 PR；Agentic 结构通过感知、更新、选择、推迟四步实现。

**📊 数据集**

数据集主要来源于 SSEC 真实项目的会议记录、聊天记录、GitHub issue、pull request 等上传文件，未使用公开数据集。

**📈 对比分析**

目前未给出与其他工具的对比实验，评估方式主要是收集生成的 GitHub 记录数量、编辑/关闭/未行动比例，以及通过访谈了解是否提升了协调与决策连续性；性能评估尚未量化。

**⚠️ 局限性**

限制包括隐私与治理约束（只能手动上传文件）、对 LLM 生成质量的依赖（可能产生错误或误导）、只能在项目范围内获取上下文、缺乏跨项目长期跟踪能力，以及对多轮对话上下文的保持有限。

---

## 301. Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung

**arXiv ID:** 2607.08362 | [PDF](https://arxiv.org/pdf/2607.08362v1)

**作者:** Anh Trac Duc Dinh `[一作]` (Ho Chi Minh City University Of Technology), Khoa Duc Anh Lam `[通讯]` (Ho Chi Minh City University Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

建立了首个包含Cham、Khmer、Tay-Nung三种越南少数民族语言的多语言语料CKTN，并针对它们提出了脚本感知的词表扩展与校准的替换词检测预训练方法（CKTN-ELECTRA）以提升分类和检索任务表现。

**💡 创新点**

提出了脚本感知的难度校准替换采样与线性调度的ELECTRA预训练框架，解决多语言脚本异质下生成器短板导致判别器利用脚本快捷方式的问题；同时首次将词表扩展与此预训练结合，显著减少词汇碎片化。

**🔧 技术方法**

使用词表扩展（SentencePiece Unigram + 分数校准 + 词向量初始化）、ELECTRA替换词检测（RTD）与脚本兼容的采样器、线性generator–discriminator调度、持续预训练与下游分类/检索微调等技术。

**📊 数据集**

使用CKTN语料库（44,367篇文档，约24M BPE子词），覆盖Cham、Khmer、Tay-Nung，用于持续预训练、28类分类与摘要-文档检索，并与mBERT、XLM‑R、RemBERT等基线进行对比。

**📈 对比分析**

在tokenizer碎片化、MLM perplexity、分类Accuracy/Macro‑F1、检索MRR@10/Recall@10等指标上评估；CKTN‑ELECTRA在分类上显著优于XLM‑R（0.9214 vs 0.8169）与RemBERT（0.1454），并在检索上提升有限；与基线相比，词表扩展后tokenizer性能明显提升。

**⚠️ 局限性**

数据来源局限于官方新闻，缺乏非正式或代码切换文本；类别分布不平衡，检索子集规模小；模型仅为编码器，未评估解码器或LLM；对RemBERT崩溃与脚本过滤机理未做深入机制验证。

---

## 302. CausalDS: Benchmarking Causal Reasoning in Data-Science Agents

**arXiv ID:** 2607.08093 | [PDF](https://arxiv.org/pdf/2607.08093v1)

**作者:** Andrej Leban `[一作]` (University of Michigan), Yuekai Sun `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CausalDS基准，生成隐式SCM场景、自然语言故事、观测数据及工具使用评估

**💡 创新点**

整合符号因果推理、数据科学执行、不确定性量化、拒答机制与工具调用，提供可复现的完整因果数据科学评测

**🔧 技术方法**

使用SCM采样、Anchor‑based grafting、图验证、LLM映射变量、自然语言生成、miniswe‑agent工具交互以及确定性评分机制

**📊 数据集**

自生成的953个SCM场景（含三种观测层），完全合成，无外部真实数据

**📈 对比分析**

通过六种LLM（Claude Opus 4.8、Gemini 3.1 Pro、GPT‑5.5、Qwen 3.6、Kimi K2.6、Gemma 4）在CausalDSScore、Pass Rate、S_NR等指标上进行对比，Claude Opus 4.8表现最优，其余模型差异显著

**⚠️ 局限性**

受限于合成数据的真实性、工具调用效率差异、模型对观测噪声与抽样变化的鲁棒性不足，以及评测主要关注单次推理而非多轮交互

---

## 303. When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals

**arXiv ID:** 2607.08065 | [PDF](https://arxiv.org/pdf/2607.08065v1)

**作者:** Kaihua Ding `[一作]` `[通讯]` (University of Pennsylvania), Kaihua Ding (University of Pennsylvania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对53名跑者在GPQA与AIME两套高难度推理基准上产生的50次采样数据进行跨跑者自一致性（self‑consistency）与准确率的相关性检验，探究自一致性作为信任或路由阈值的可靠性。

**💡 创新点**

创新点在于首次系统性量化自一致性与准确率之间的关系，并揭示其在最具一致性的前沿模型上会出现过度自信、低校准的反向效果；同时发现高一致性答案往往带有位置偏置且在不同模型间复现，提示共享偏差而非随机错误。

**🔧 技术方法**

主要技术包括大规模跨跑者采样（K=50）、层级跑者聚类自举、Spearman ρ相关分析、误差覆盖曲线、选项洗牌实验以及跨家族（Claude）探索性对照。

**📊 数据集**

使用的数据集为GPQA Diamond（四选一多项选择）和AIME（整数答案）两套竞赛题库，累计约265,000条采样记录。

**📈 对比分析**

与传统的口头信心或P(True)等信心信号相比，自一致性的Spearman ρ在中等规模模型上最高可达0.59，但在前沿模型上仅为0.20；链式思考提升准确率约7%，但对信心信号提升有限；自一致性在中等模型下可用于节约采样成本，但作为独立置信度阈值时会导致高错误率。

**⚠️ 局限性**

主要局限包括仅在单一供应商（gpt‑4.1）上实验、未记录模型快照与时间戳、K值有限（50）导致测量噪声、仅覆盖两种基准且不涉及开放式生成或代码任务，以及跨家族对照缺乏严格的实验控制。

---

## 304. EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data

**arXiv ID:** 2607.08436 | [PDF](https://arxiv.org/pdf/2607.08436v1)

**作者:** Baoyu Li `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个人机共训练框架（WAM）来利用头戴式第一人称视频数据进行机器人操作学习，并系统评估不同的世界表示对跨体态迁移的影响。

**💡 创新点**

核心创新在于对世界模型目标的可控实验：通过统一的 Backbone 和动作头，只更换未来状态预测目标，揭示 DINO 特征和 3D 流比像素重建更能促进人机迁移，并证明 WAM 能够比传统行为克隆更好地利用大规模野外人类数据。

**🔧 技术方法**

使用了 Heterogeneous Pretrained Transformer（HPT）共享主干，条件流匹配动作头，三种世界模型头（像素 VAE、DINO 语义特征、摄像机对齐的 3D 流），以及联合训练与动作仅推理的策略。

**📊 数据集**

采用了两套人类数据（EgoVerse 规模化野外视频与对齐场景的在域人类数据）和机器人遥控演示（共 3 个双臂任务：杯子放盘、折衣服、装杂货），并在真实机器人上进行评估。

**📈 对比分析**

对比方法包括：传统行为克隆（BC）共训练、Pixel-VAE WAM、DINO WAM、3D-Flow WAM。实验结果显示：Pixel 预测几乎无提升；DINO WAM 在 OOD 任务中提升多达 4 倍；3D-Flow WAM 在 ID 任务中提升 20–30%，整体性能明显优于 BC。

**⚠️ 局限性**

局限性主要有：1）仅提升上下文层面表现，尚未实现从人类数据学习新的运动原语；2）实验局限于单任务，缺乏多任务共训练分析；3）最优世界表示仍未确定，需进一步探索开放世界场景中的通用表示。

---

## 305. The $\ominus$-metric to compare phylogenetic networks

**arXiv ID:** 2607.08259 | [PDF](https://arxiv.org/pdf/2607.08259v1)

**作者:** Marc Hellmuth `[一作]` (Leipzig University), Guillaume E. Scholz `[通讯]` (Greifswald University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出两种基于⊖-运算的度量，用于比较根化的系统发育网络；其中一种是最小化删除顶点得到同构网络的删除次数（记为 ⊖-距离），另一种在删除后忽略所有shortcut弧后再比较（记为 ⊖-relaxed 距离）。

**💡 创新点**

创新点在于：①首次将⊖-运算引入网络度量，保证了度量公理（包括三角不等式）并可直接解释为“删除顶点”操作；②两种度量在树类网络上退化为经典的 Robinson–Foulds 距离；③对于广泛的网络类别（树子网络、正常网络、一级网络、正则网络等），⊖-relaxed 距离可多项式计算，并可归约为 Vertex Cover（从而获得 FPT、2-近似等算法）。

**🔧 技术方法**

主要技术包括：⊖-运算的定义与性质证明、短路弧的消除、度量公理的验证、对 distinct‑cluster 网络的 Canonical Cluster 匹配、将距离计算转化为 Vertex Cover 问题、使用 FPT 与近似算法求解 Vertex Cover、构造 bad ancestry 图、证明 NP‑硬度与 W[2]‑硬度。

**📊 数据集**

本文为理论性研究，未使用实验数据集；所有结论均通过严谨的数学证明与构造实例（如特定网络、集合覆盖实例）得出。

**📈 对比分析**

对比方法：与硬线集距离、软线集距离、显示三元组距离等传统特征基度量相比，⊖-距离提供了更细粒度的操作视角，并在树类网络上与 RF 距离一致。性能方面：⊖-relaxed 距离在多类网络上可多项式计算，且在 distinct‑cluster 网络上可用 2-近似；而 ⊖-距离在一般网络上是 NP‑完备、W[2]‑硬且无多项式时间常数近似（除非 P=NP）。

**⚠️ 局限性**

局限性包括：①⊖-距离的 NP‑完备性导致在大多数网络上无法高效求解；②对一般网络的硬线集下缺乏可行的多项式或 FPT 算法；③虽然 ⊖-relaxed 距离可多项式，但仍需要对网络做 shortcut‑free 处理；④对更广泛的 c‑distinct‑cluster 网络是否 FPT 仍未解决；⑤对于特殊网络类别（如有高水平或特殊拓扑约束）的进一步算法尚需研究。

---

## 306. Swapping Faces, Saving Features: A Dual-Purpose Pipeline for Pedestrian Privacy in ITS

**arXiv ID:** 2607.08402 | [PDF](https://arxiv.org/pdf/2607.08402v1)

**作者:** Roba H. Farouk `[一作]` (C-DRiVeS Lab), Catherine M. Elias `[通讯]` (C-DRiVeS Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个包含检测、增强、置换、融合五个阶段的行人隐私保护管线，能在街景图像中匿名化行人身份同时保留表情和视线等关键属性。

**💡 创新点**

创新点在于将Roop面部置换模型与多阶段预处理（YOLOv11、SCRFD、Codeformer、Poisson融合）相结合，既能充分保留面部几何与表情，又实现身份模糊，克服传统模糊或GAN方法导致的可用性下降。

**🔧 技术方法**

使用了YOLOv11进行行人检测，SCRFD进行面部检测，Codeformer进行盲面部恢复，Roop（和Ghost‑v2）进行面部置换，GFPGAN进行图像增强，以及OpenCV的Poisson融合实现无缝叠加。

**📊 数据集**

实验主要基于埃及Egy‑DRiVeS街景数据集（包含多种姿态、遮挡和女性戴头巾的图像），并对高质量面部图像和低分辨率街景图像进行验证。

**📈 对比分析**

通过对Roop与Ghost‑v2在landmark差、blendshape差、cosine相似度（身份）以及视线向量相似度四项定量指标的对比，Roop在三项指标上优于Ghost‑v2，并在视觉评估中表现出更高的现实感和属性保留，整体实现了隐私与可用性的平衡。

**⚠️ 局限性**

局限性包括对极远或高度遮挡面孔的置换仍有失败率，低分辨率图像在增强后可能产生形变，视频实时推理尚未优化，且缺乏针对不同肤色、年龄的源面孔选择策略。

---

## 307. TRACE: A Two-Channel Robust Attribution Watermark via Complementary Embeddings for LLM-Agent Trajectories

**arXiv ID:** 2607.08400 | [PDF](https://arxiv.org/pdf/2607.08400v1)

**作者:** Zheng Gao `[一作]` (University of New South Wales), Liming Zhu `[通讯]` (CSIRO's Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种双层水印方案，用于在大型语言模型代理的行动轨迹中嵌入可追溯的指纹，从而在代理被转售时仍能实现归因。

**💡 创新点**

创新点在于同时对“删除”攻击和“重写”攻击设计两种互补的水印通道——内容键定的选择通道（保持分布不变、对删除自适应）和位置键定的计数通道（对重写完全不变），并给出联合抹除下的理论下限与实际鲁棒性证明。

**🔧 技术方法**

采用了 HMAC‑SHA512 随机数生成、指数竞赛采样、伪随机函数密钥、上下文编码、冗余记录插入以及精确的 Gamma/Binomial 统计检验，实现了无失真、可同步的水印嵌入与检测。

**📊 数据集**

使用 ToolBench（工具使用决策）和 ALFWorld（交互式文本规划）两个基准数据集，对不同长度和决策熵的任务进行实验。

**📈 对比分析**

与 AgentMark、红绿水印等单通道基线相比，本方案在保持任务成功率不变的前提下，检测 z‑分数显著更高；在删除、重写单轴攻击下各自通道保持鲁棒，而在联合攻击时仅在极端情况下失效；实验表明单条轨迹在 ALFWorld 能达到 90%+ 的 TPR@1%FPR，ToolBench 需要约 10 条轨迹。

**⚠️ 局限性**

局限性包括：需要在审计时获得候选动作集合；冗余记录导致日志体积略增；对完整模型替换（不在日志中记录）无法直接检测；对极低熵决策的可检测性受限，需更多轨迹聚合。

---

## 308. HoloTetSphere: Unified TetSphere Mesh Reconstruction for Physical Simulations

**arXiv ID:** 2607.08398 | [PDF](https://arxiv.org/pdf/2607.08398v1)

**作者:** YaQiao Dai `[一作]` (National University of Defense Technology), Chenyang Zhu `[通讯]` (Institute of AI for Industries, Chinese Academy of Sciences)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为 HoloTetSphere 的拓扑自适应统一四面体网格重建框架，实现了从多视角图像直接生成可用于物理仿真的连通四面体网格。

**💡 创新点**

通过将四面体元素与高斯球体耦合并使用连续不透明度场实现可微化的拓扑剪枝，结合交替几何优化和双阶段 Laplacian 平滑，突破了传统 TetSphere 的同构限制，实现了自适应拓扑和高质量几何。

**🔧 技术方法**

基于差分渲染的多视角优化、连续隐式不透明度场、顶点共享的可微剪枝、带权双调和能量的两阶段 HC‑Laplacian 平滑、图像轮廓与法向监督等技术。

**📊 数据集**

使用 Thingi10k、DeepFashion3D、Objaverse、Google Scanned Objects 等八种闭合模型、四种开放模型，共约30个对象，每个对象120张多视角图像进行训练与评估。

**📈 对比分析**

与 Eulerian 方法 NeuS2 及 Lagrangian 方法 2DGS、DMesh、TetSphere 进行对比，几何指标（Chamfer、Hausdorff、IoU）均位居前列，单组件率达到 96.7%，并在物理仿真稳定性（逆扭转率仅 0.017%）和渲染质量（PSNR/SSIM）方面优于基线。

**⚠️ 局限性**

在高度复杂拓扑或极薄薄板结构时仍可能出现局部逆转或细节缺失；剪枝仅删除而非添加材料，限制了对任意拓扑编辑的支持。

---

## 309. Token-Flow Firewall: Semantic Runtime Auditing for Persistent AI Agents

**arXiv ID:** 2607.08395 | [PDF](https://arxiv.org/pdf/2607.08395v1)

**作者:** Puji Wang `[一作]` (State Key Laboratory of AI Safety), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 TokenWall，一个基于 token 流的运行时防御框架，用于保护持续 AI 代理的安全。

**💡 创新点**

创新点在于在 token 语义流层面进行预先审计，结合本地小模型检查、可恢复重写以及可选远程仲裁，实现了前置式、覆盖全面且延迟低的安全屏障。

**🔧 技术方法**

采用规则预检查、本地小模型（Qwen3-4B）语义审计、可重写机制、边界感知决策及可选大模型仲裁（Qwen3.6-Plus）等技术。

**📊 数据集**

使用 CIK-Bench 基准（攻击与正常案例）以及匹配的正面案例集进行评估。

**📈 对比分析**

与 OpenClaw 生态中 7 种防御基线对比，攻击成功率从 14.7% 降到 12.5%，防御延迟仅 0.69 秒，拒绝率和人工干预率显著降低，展示了更优的安全–效率折中。

**⚠️ 局限性**

局限性在于仅在 OpenClaw 任务环境评估，依赖本地模型质量和运行时元数据，无法覆盖主机被攻破、权限窃取或用户授权恶意操作等极端场景。

---

## 310. Early to Share, Late to Save: Synchronisation-Driven Communication Gating in Bandwidth-Constrained Cooperative VLN

**arXiv ID:** 2607.08504 | [PDF](https://arxiv.org/pdf/2607.08504v1)

**作者:** Arav Gupta `[一作]` (Birla Institute of Technology and Science), Avinash Gautam `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究带宽受限的协同视觉语言导航，提出后视门控机制，用监督学习替代 REINFORCE 进行通信决策。

**💡 创新点**

创新点包括：①将协同 VLN 扩展到有限通信预算；②设计后视门控的 BCE 监督框架；③揭示通信最佳时机在早期高置信时同步隐藏状态，而非不确定时恢复。

**🔧 技术方法**

采用 CLIP ViT‑B/32 视觉/语言编码、交叉注意力生成上下文向量、GRU 隐藏状态、三层 MLP 门控、离线标签收集以及隐状态对齐度量等技术。

**📊 数据集**

使用 R2R Matterport3D 室内导航数据集，利用 CLIP 特征并构造两代理不对称路径对来模拟信息差异。

**📈 对比分析**

与无通信、全通信、随机门控、熵门控等基线比较，在每代理 B=3 的预算下，后视门控实现隐藏状态对齐 +0.072，几乎等同于全通信，并将成功率从 8.7% 提升至 8.9%。

**⚠️ 局限性**

局限性在于仅针对两代理离散图导航，未考虑多代理或连续运动场景、对称角色配置，以及对未见建筑的泛化能力不足。

---

## 311. CT-CLIP Representations for Multimodal Lung Cancer Survival Prediction

**arXiv ID:** 2607.08503 | [PDF](https://arxiv.org/pdf/2607.08503v1)

**作者:** Sofie Allgöwer `[一作]` (Chalmers University of Technology), Jennifer Alvén `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估了医学专用视觉‑语言预训练模型 CT‑CLIP 在多模态肺癌生存预测任务中的可用性，使用 242 例肺癌患者的预处理 CT 扫描和临床变量，比较了冻结、全微调及 LoRA 三种适配策略。

**💡 创新点**

创新点在于首次将医学 CLIP 预训练模型直接用于生存分析，并证明仅冻结其权重、只训练轻量级生存头即可获得优于传统基线的性能，适用于低资源、样本受限的临床环境。

**🔧 技术方法**

技术方法包括 CT‑CLIP（ViT + BERT）作为特征提取器、DeepSurv 头、LoRA 参数高效微调、CoxPH 基线以及四种多模态参考模型（Interactive‑Model、DAFT、FiLM、ResNet+Tabular）进行对照实验。

**📊 数据集**

使用的数据集为 242 名 2008–2018 年诊断的肺癌患者的预处理 CT 扫描和临床变量（如性别、分期、烟草史等），中位生存时间 3.4 年，事件发生率 78%。

**📈 对比分析**

通过与 CoxPH 临床基线和四种多模态参考模型在 50 名测试样本上的对照，冻结 CT‑CLIP 版本实现 C‑index 0.755，优于 CoxPH（0.721）且与 ResNet+Tabular 相当，同时在 Kaplan‑Meier 风险分层中实现显著分组（p<0.001）。

**⚠️ 局限性**

局限性包括样本量有限、缺乏外部验证集、对不同 CT 协议或低剂量扫描的泛化能力未知，以及模型对预训练权重的依赖性。

---

## 312. Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing

**arXiv ID:** 2607.08497 | [PDF](https://arxiv.org/pdf/2607.08497v1)

**作者:** Feng Wang `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多模态对话代理，将视觉信息外部化为情节式记忆，并在每轮对话中选择性检索相关视觉片段，自动推断任务并执行理解、生成或编辑。

**💡 创新点**

创新点在于：①将视觉记忆结构化为标签、描述和缩略图，避免将全部视觉 token 注入上下文；②解耦感知、检索和执行三大模块，支持独立训练与升级；③开发程序化生成的长序列多模态对话数据集，提供细粒度检索标注；④采用分阶段 SFT+RL 训练优化检索与记忆构造，显著提升检索准确率。

**🔧 技术方法**

技术包括：Qwen3‑VL‑8B 作为感知与检索模型；Qwen‑Image‑Edit 用于生成/编辑；分阶段监督微调 (SFT) 与基于 DAPO 的强化学习；情节式视觉记忆库与跨模态检索机制；以及工具增强部署框架。

**📊 数据集**

使用了自研的“CMA‑Harness”数据集：100 个 20 轮多模态会话（共 2000 轮），包含标签化检索标注；并在该基准上进行评测。

**📈 对比分析**

与统一模型（BAGEL）、全上下文大模型（8B/32B）以及多模态多代理基线进行对比；检索准确率达到 91.4%（英语）/89.6%（中文），比 32B 基线提升 8.2%/5.1%；生成质量与检索成正相关；每轮推理时间从 23.1s 降至 12.7s。

**⚠️ 局限性**

局限性：目前仅支持视觉+文本两模态；记忆抽象仍基于预先定义的标签/描述，对高度细粒度视觉差异的区分仍有欠缺；工具集成与跨会话持久记忆需要进一步完善。

---

## 313. VEGAS: Human-Aligned Video Caption Evaluation via Gaze

**arXiv ID:** 2607.08489 | [PDF](https://arxiv.org/pdf/2607.08489v1)

**作者:** Shenghui Chen `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无需重新训练的视线条件字幕评估与选择框架（VEGAS），并构建了同步视线-视频-字幕数据集。

**💡 创新点**

创新点包括：① 用信息论的点互信息将视线与字幕对齐度量为低VEGAS；② 通过拒绝采样在不改动 VLM 的前提下选取最符合观众视线的字幕；③ 结合 egocentric 视频和 slide 两个领域的同步视线数据，验证方法的跨域有效性。

**🔧 技术方法**

技术手段：预训练视觉语言模型（VLM）的 token 似然计算、信息论公式、拒绝采样、SBERT 相似度评估以及视频检索指标。

**📊 数据集**

使用的数据集：Aria Everyday Activities（AEA） egocentric 视频与视线；SlideVQA slide 演示与视线。

**📈 对比分析**

评价方式：与传统 VLM 生成字幕、随机选择、最佳候选等进行 SBERT 相似度对比；在 AEA 上 VEGAS 的 SBERT 提升 +0.0856（≈13.5%）并在 caption‑to‑video 检索中 mAP@10 提升 +2.46%；在 SlideVQA 上仅有微小 (+0.0256) 无显著差异。

**⚠️ 局限性**

局限性：受 VLM 的幻觉、概率校准不佳影响；视线只能捕捉显式视觉关注，对需要概念抽象的字幕效果有限；需要显式视线采集，非无视线环境难以直接部署。

---

## 314. Spatio-Temporal Scheduling Prediction Under Backhaul Delay for Resilient Coordinated Beamforming

**arXiv ID:** 2607.08454 | [PDF](https://arxiv.org/pdf/2607.08454v1)

**作者:** Prashant Kumar Singh `[一作]` (Stockholm University), Li Wang `[通讯]` (Huawei R&D)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在分布式5G网络中使用时空图神经网络预测邻近基站的调度状态，替代因回程延迟导致的陈旧信息，从而在协调波束成形中恢复吞吐量和公平性。

**💡 创新点**

将时空图神经网络与SLNR波束成形结合的两阶段预测辅助框架；实现对动态用户数的无须重新训练的置换不变性；在回程延迟1个TTI的最坏场景中恢复73%吞吐量。

**🔧 技术方法**

StemGNN时空图神经网络、GRU/LSTM等循环网络、Markov链、移动平均；SLNR波束成形；用于时空预测与协调。

**📊 数据集**

Quadriga Urban Micro (UMi) 传播模型生成的三站点 Massive MIMO 下的60 UE 数据；使用三种子载波配置1 SC、48 SC。

**📈 对比分析**

与无预测、LSTM、GRU等基线比较；在Lag1下StemGNN相较无预测提升约9.6%–14.3%吞吐量，约57–73%恢复损失；相比LSTM/GRU提升1.5%–3.1%。

**⚠️ 局限性**

预测在Lag1之外快速衰减（21个百分点），闭环分布偏移导致精度下降；仅评估固定网络规模与调度器，真实网络复杂度更高。

---

## 315. FPGN: Redefining Ultra-Fast Programmable Gate-based Neural Acceleration with Differentiable LUTs

**arXiv ID:** 2607.08427 | [PDF](https://arxiv.org/pdf/2607.08427v1)

**作者:** Jiawei Liang `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了FPGN框架，通过将FPGA LUT视为可学习神经元，实现了纳秒级神经网络加速器。

**💡 创新点**

创新点包括硬件对齐的可微LUT训练、结构化拓扑与流式架构的物理共设计以及基于QoR模型的延迟驱动编译器。

**🔧 技术方法**

采用可微LUT逼近、渐进二值化、结构化微/宏拓扑、流水线时序模型、混合整数线性规划的DSE等技术。

**📊 数据集**

在CIFAR‑10、SVHN、KWS、JSC‑CERNBox/OpenML等数据集上训练和评估。

**📈 对比分析**

与FINN、LUTNet、DWN等现有方法对比，FPGN在同平台上实现658 ns延迟、3.21 M FPS、比FINN低205倍延迟、比LUTNet 222倍吞吐、比DWN 30倍LUT效率，保持相近准确率。

**⚠️ 局限性**

局限在于仅支持全二值网络，缺乏对更高精度或更复杂网络结构的探索，且仍依赖特定FPGA平台。

---

## 316. Optimization and Deep Learning based Resource Allocation for UAV-Aided Wireless Communication with Rotatable Antenna Array

**arXiv ID:** 2607.08420 | [PDF](https://arxiv.org/pdf/2607.08420v1)

**作者:** Fengcheng Pei `[一作]`, Robert Schober `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在无人机（UAV）下行多用户通信中使用可旋转天线阵列（RAA）与固定阵列（FAA）进行联合方向与波束赋形设计，以最大化总速率并满足每个用户的 QoS 约束。

**💡 创新点**

创新点：①提出基于 PDD 的优化框架，能够解耦 RAA 方向与波束赋形的强耦合关系，并保证收敛到 KKT 点；②设计了两模块 GNN（方向模块 + 波束模块）和两阶段无监督训练策略，解决梯度不平衡问题，实现实时决策；③在仿真中证明 RAA 在小天线数、用户密集场景下显著提升总速率和 QoS 可行率，并展示对位置误差的鲁棒性。

**🔧 技术方法**

使用的技术包括：PDD（Penalty Dual Decomposition）、MM（Majorization–Minimization）、RCG（Riemannian Conjugate Gradient）、SCP（Second‑Order Cone Programming）、GNN（注意力机制、跳跃残差）、两阶段无监督训练、Lagrangian Dual Method、Python+PyTorch 仿真。

**📊 数据集**

数据集：随机生成的用户位置数据，均匀分布于 100 m × 100 m 平面，UAV 固定位置 [0,0,40] m。训练集 10 万个样本，验证集 1 万个样本。

**📈 对比分析**

比较方法：共 8 种方案（RAA/FAA × Dip/ Iso × Opt/DL），在不同 QoS 阈值、天线数、用户数、位置误差下进行对比。结果显示 RAA‑Opt/DL 在总速率上比 FAA 提高 30%–50%；在严格 QoS 时 RAA‑Opt 仍优于 DL，低 QoS 时 DL 仅损失 ≈2%；计算时间方面，DL 约 0.003 s，远低于 PDD 的 4–7 s；鲁棒性方面，RAA‑DL 对位置误差更不敏感。

**⚠️ 局限性**

局限性：PDD 迭代次数多，计算复杂度高；DL 需要大量训练样本，且对超参数敏感；假设用户位置已知且信道为 LoS；未考虑 UAV 轨迹优化和动态环境；GNN 结构仅适用于固定规模的用户数，扩展性待验证。

---

## 317. H3D: Benchmarking Unsupervised Text Hashing for Fine-Grained Document Deduplication

**arXiv ID:** 2607.08382 | [PDF](https://arxiv.org/pdf/2607.08382v1)

**作者:** Qianren Mao `[一作]` (Zhongguancun Laboratory), Bo Li `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `79276348-11e0-48e3-84bc-7ec231d0171c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 H3D 基准，统一协议评估非学习与冻结嵌入哈希在细粒度文档去重上的表现。

**💡 创新点**

创新点在于：1）提供统一的查询-候选排名框架；2）系统比较语义无关与语义敏感哈希的兼容性与鲁棒性；3）对文本压缩对哈希鲁棒性的实验分析。

**🔧 技术方法**

使用了非学习哈希方法（MinHash、SimHash、Winnowing、FuzzyHash、FlyHash）以及基于 BGE 的冻结嵌入量化（BGE‑BIHash 与 BGE‑LSHash）。

**📊 数据集**

实验数据集包括 CSFCube（计算机科学论文的 facet 级相似性）和 RELISH（生物医学文献的整体相似性）。

**📈 对比分析**

通过 MAP 与 NDCG@20 进行对比，结果显示语义无关哈希在近似重复检索上效率高、能耗低；BGE 量化哈希在语义重写和压缩下保持更高准确度，但计算成本显著增加。

**⚠️ 局限性**

局限性包括：仅覆盖无监督非学习与冻结嵌入哈希，未评估训练型深度哈希；数据集规模与领域有限，难以验证跨域泛化；时间评估受硬件/实现细节影响。

---

## 318. Dynamics of Gradient Descent with Large Step Size Near a Manifold of Flat Minima

**arXiv ID:** 2607.08380 | [PDF](https://arxiv.org/pdf/2607.08380v1)

**作者:** Lachlan Ewen MacDonald `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了梯度下降在大步长下的动力学，推广了之前单标量输出的最小二乘理论到多维输出以及平坦最小值流形，并在矩阵因子分解等实例中证明了理论。

**💡 创新点**

关键创新在于构造了更一般的正则化形式（normal form），证明了在任意余维下梯度下降的收敛三种行为，并发现平坦最小值集为球面积纤维束，且锐度在该流形上为 Morse-Bott。

**🔧 技术方法**

使用了微分几何、中心流形理论、奇异偏微分方程求解、新颖的正则化 PDE 技术，以及对矩阵因子分解的几何分析。

**📊 数据集**

实验主要在合成的矩阵因子分解问题（如3层2×2矩阵因子分解）上进行验证。

**📈 对比分析**

与传统小步长GD或基于PL不等式的收敛理论相比，本文提供了大步长下的收敛证明，实验显示收敛速率与理论预测一致。

**⚠️ 局限性**

局限性包括仅在平坦最小值流形附近的局部收敛，无法解释全局“边缘稳定”或更大步长下的混沌行为；对极限情形的假设（如稳定分支猜想）尚未完全证明。

---

## 319. Ensemble Diversity Optimization for Subjective Supervision

**arXiv ID:** 2607.08493 | [PDF](https://arxiv.org/pdf/2607.08493v1)

**作者:** Xia Cui `[一作]` (Manchester Metropolitan University), N. R. Abeynayake `[通讯]` (Manchester Metropolitan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在预测空间中通过带符号多样性正则化、可靠性加权以及可学习的集合结构来建模主观任务中的注释者不一致性的方法，称为 Ensemble Diversity Optimization (EDO)；

**💡 创新点**

创新点在于将集合多样性视为可调方向的正则化（可选择保留或压制不一致），联合优化集合权重、有效集合大小与校准，使模型能够在保持预测性能的同时提升对注释者分布的逼近与概率校准；

**🔧 技术方法**

使用 Gumbel‑Softmax 进行集合大小的可微学习；对集合成员的加权采用可靠性加权和正则化；采用软 F1 损失、类别加权交叉熵与 signed 多样性损失进行多目标联合优化；

**📊 数据集**

在四个主观文本分类基准上评估：ArMIS、ConvAbuse、HS‑Brexit 与 MD‑Agreement；

**📈 对比分析**

与 Soft‑CE、Soft‑MD、Top‑5 Voting 以及 WEL 等基线对比，EDO 在所有数据集上显著降低交叉熵（相对 Soft‑CE 降低 40–78%）和 Brier 分数，并保持竞争的 F1，显示出更好的概率校准与对注释者分布的对齐；

**⚠️ 局限性**

局限包括：对不一致性结构的依赖（需要先验选择保留或压制方向）；需手动调节多目标权重和温度；当前仅在预测空间工作，冻结主干模型，限制表示能力；缺乏动态自适应权重和对注释者元数据的利用。

---

## 320. Learning LDPC codes with quantized density evolution over relaxed protographs

**arXiv ID:** 2607.08484 | [PDF](https://arxiv.org/pdf/2607.08484v1)

**作者:** Gennady Shutkov `[一作]` (Skolkovo Institute of Science and Technology), Kirill Andreev `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于松弛原型图的量化密度演化（DE）方法，用梯度下降（GD）直接优化 LDPC 码的原型图结构；

**💡 创新点**

创新点在于将原型图的每个条目解释为边出现的伯努利概率，从而得到可微分的松弛 DE 评估器，避免了传统的离散搜索和 Monte‑Carlo 估计；

**🔧 技术方法**

采用的技术包括量化 LLR 分布、基于 PMF 的向量化 DE 更新、自动微分求梯度以及投影梯度下降；

**📊 数据集**

实验使用了 5G 标准化的原型图（BG1 中等/高率）与 AWGN 通道模拟，基于 BPSK 的 50 分位 LLR 网格；

**📈 对比分析**

与 5G LDPC 参考码在相同有效原型图尺寸、提升因子和 puncturing 方案下比较，优化后的码在 BLER=10⁻⁴ 时分别获得约 0.18 dB（R=1/2）和 0.03 dB（R=0.88）的性能提升；

**⚠️ 局限性**

局限包括：DE 只适用于归一化最小和（NMS）检查节点，无法直接推广到求积解码；对有限长度图的短环影响未建模；非整数条目产生的概率分布可能导致最终码的多样性受限。

---

## 321. Applying JEPA-Style Predictive Learning to JA4-Derived Network Fingerprints

**arXiv ID:** 2607.08465 | [PDF](https://arxiv.org/pdf/2607.08465v1)

**作者:** Javier Izquierdo `[一作]` (Lucerne University of Applied Sciences and Arts), Aygul Zagidullina `[通讯]` (Lucerne University of Applied Sciences and Arts)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 JA4-JEPA，一种基于 Transformer 的自监督预测模型，用于学习压缩网络指纹，并通过冻结的 kNN 评估协议族分类性能。

**💡 创新点**

首次将 JEPA 的潜在预测目标迁移到网络指纹领域，解决多源、缺失视图的数据集，提出混合源训练与缺失视图消融方法。

**🔧 技术方法**

采用 Transformer 编码器+EMA 目标、潜在匹配损失、冻结 kNN probe 以及 JEPA 能量信号进行异常检测。

**📊 数据集**

使用 JA4DB 与 CIC-IDS‑2017 共同构成的约 397K 条 JA4+ 子字段样本，以及 2.1M 条企业网关指纹对作为生产试点。

**📈 对比分析**

通过冻结编码器的 kNN probe 在混合源数据上进行 TLS/DNS/SSH 协议族分类，取得 92.2% 的准确率与 0.9899 的余弦相似度；在试点语料中与频率、最近邻、autoencoder 等基线对比，JEPA 能量在两类合成异常上的 AUC 达 0.922，且推理速度保持恒定。

**⚠️ 局限性**

存在的局限包括任务过于粗粒度（仅协议族分类）、视图覆盖不完整导致模型可能过度依赖 JA4 作为桥梁、缺乏细粒度标签，难以验证跨模态学习的深度及泛化能力。

---

## 322. Two Axes of LLM Abstention: Answer Correctness and Question Answerability

**arXiv ID:** 2607.08456 | [PDF](https://arxiv.org/pdf/2607.08456v1)

**作者:** Benedikt J. Wagner `[一作]` `[通讯]` (University of London), Benedikt J. Wagner (University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在拒绝不适宜问题时的两个失败模式，并提出将答案正确性与问题可答性拆分为两条轴的几何模型。

**💡 创新点**

创新点在于揭示答案正确性与可答性在同一决策上形成的交叉三状态几何，证明单一置信度阈值无法同时控制错误答案与不可答问题，并提出双风险认证的分量化阈值组合策略。

**🔧 技术方法**

技术方法包括：对5个指令微调模型进行自回归生成与内部隐藏状态读取；训练等容量的Logistic回归读取器；对CREPE自然语言错误前提数据进行迁移与评估；使用分层分割、合成阈值、Clopper‑Pearson 上限的三方认证框架；以及通过隐藏探测器对准提示检查的路由修复。

**📊 数据集**

数据集：SelfAware（可答/不可答标注的问答对）和CREPE（基于Reddit ELI5的错误前提问答），两者均为英文公开数据集。

**📈 对比分析**

与单一置信度阈值或单一答案可答性阈值的基线相比，双信号阈值组合在满足 α_U=0.15、α_W=0.50 的风险预算下，能够在 8B 规模模型上实现 0.75 的正确答案覆盖率，显著高于单信号策略；在 14B 规模模型上则是唯一能通过认证的策略。

**⚠️ 局限性**

局限性包括：实验仅覆盖至 14B 模型，尚未验证更大规模模型的行为；验证集规模受限导致较宽松的错误答案预算；CREPE 评估只覆盖可答性轴，缺乏准确性标签；并且对答案正确性的判定依赖于有限的别名匹配，可能低估了模型性能。

---

## 323. Coded Task Offloading for Fluid Computing: A Privacy-Aware Approach under D2D Networks

**arXiv ID:** 2607.08440 | [PDF](https://arxiv.org/pdf/2607.08440v1)

**作者:** Diego Cajaraville-Aboy `[一作]` (Universidade de Vigo), Rebeca P. Díaz-Redondo `[通讯]` (Universidade de Vigo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于线性秘密共享的编码任务迁移方案，并将隐私泄露风险纳入任务调度决策；

**💡 创新点**

将隐私泄露（通过噪声通道的总变差上界）与延迟、能耗联合建模，设计了隐私惩罚项；同时提供了Branch‑and‑Bound贪婪调度和轻量级启发式调度两种求解器；

**🔧 技术方法**

离散事件仿真、线性秘密共享（RampSS）、分支限界求解、启发式优先级调度、信息理论隐私度量（总变差）等；

**📊 数据集**

使用基于仿真的合成任务数据（Poisson到达、不同规模/负载的设备配置）和公开的D2D/NR侧链参数；

**📈 对比分析**

与经典全迁移、并行迁移以及SEC2D、SMUA等基线进行比较，实验表明启发式调度在近似最优的同时显著降低调度时延，且在延迟‑能耗‑隐私三者之间实现更优的折衷；

**⚠️ 局限性**

问题本身为NP‑hard，调度器需集中全局状态，且隐私度量基于理论上限，实际侧信道攻击的细节未建模，且对网络拓扑和设备协作范围做了理想化假设。

---

## 324. When Synthetic Speech Is All You Have: Better Call GRPO

**arXiv ID:** 2607.08409 | [PDF](https://arxiv.org/pdf/2607.08409v1)

**作者:** Shashi Kumar `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对受隐私法规限制、难以获得真实语音的银行业务场景，研究如何利用合成语音来进行大语言模型（LLM）驱动的自动语音识别（ASR）适配，并提出将强化学习方法Group Relative Policy Optimization（GRPO）用于合成语音的域适配；

**💡 创新点**

创新点在于：①将GRPO从传统的真实语音任务迁移到合成语音域适配，证实在合成语音上比传统监督微调（SFT）能显著降低WER；②通过对插入错误、停止校准、音频注意力等行为指标的分析，揭示GRPO在行为层面而非表征层面提升性能；③证明在合成语音充足、真实语音稀缺时，单纯使用GRPO即可实现与少量真实语音结合时相当甚至更好的性能。

**🔧 技术方法**

使用了LLM驱动的ASR架构（WavLM-Large声学编码器 + Llama-3.2-1B-Instruct文本生成器 + 可训练的声学投影层 + LoRA适配器），并在此基础上实现SFT和GRPO两种适配策略；GRPO采用无价值函数的组相对策略优化，奖励函数为WER（或WER+长度/字符误差）。

**📊 数据集**

主要使用DefinedAI银行业务电话语音数据（约54小时真实语音，6.55小时测试集）及其对应的合成语音（通过Qwen3-TTS + RIR混合产生）；此外使用LibriSpeech 960小时作为预训练语料。

**📈 对比分析**

实验比较SFT、GRPO及其组合在真实语音、合成语音和两者混合时的WER/CER/插入/删除/替换率。结果显示：在仅合成语音的情况下，GRPO将WER从36.71%降低至22.09%（相对下降40%），SFT+GRPO进一步至20.21%；在全54小时真实语音下，GRPO相较SFT仅提升约1%（10.27%→9.49%）；在混合真实+合成语音时，加入5–10小时真实语音可实现大部分WER降低。

**⚠️ 局限性**

局限性包括：①实验仅在单一银行业务语料上验证，缺乏跨领域推广；②合成语音的质量受TTS与RIR模拟限制，若合成音频与真实语音差距进一步扩大，GRPO效果可能衰减；③目前只评估了WER、CER等传统指标，对实时性、模型大小等工业化因素未深入探讨。

---

## 325. Track2Map: Online Deformable SLAM with Motion-Aware Pose Optimization in Robotic Surgery

**arXiv ID:** 2607.08408 | [PDF](https://arxiv.org/pdf/2607.08408v1)

**作者:** Tianyi Song `[一作]` (University College London), Francisco Vasconcelos `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种在线可变形SLAM系统Track2Map，可在无或噪声相机姿态先验的情况下，利用立体深度和密集2D跟踪实现3D Gaussian Splatting的实时构建。

**💡 创新点**

创新点包括：① 通过密集2D点跟踪初始化变形控制点并融合到Gaussian模型；② 采用运动感知门控，仅在检测到全局相机运动时才更新姿态，防止局部组织运动被误归为相机位移；③ 兼容有、无、噪声姿态先验，自动适配不同数据来源。

**🔧 技术方法**

核心技术：3D Gaussian Splatting、CoTracker3密集2D跟踪、立体深度投影、运动门控策略、联合光度/几何/变形正则化优化、SE(3)姿态更新。

**📊 数据集**

使用StereoMIS外科视频数据集进行实验，同时在STIR2024数据集上评估跟踪模块的精度。

**📈 对比分析**

与多种SLAM基线（如EndoGSLAM-H、EndoSD-SLAM等）和基于姿态先验的神经映射方法对比。指标包括PSNR、SSIM、LPIPS以及ATE/RPE。无先验时PSNR达27.58、SSIM 0.745；在噪声先验下性能几乎不降；姿态误差低于0.03 m，明显优于对比方法。

**⚠️ 局限性**

局限性：当前实现非实时，约6 s/帧，建议采用关键帧；运动门控假设全局流一致即为相机运动，易受光轴运动、工具遮挡或弱纹理区域影响；评估主要基于StereoMIS，需在更多外科数据上验证泛化能力。

---

## 326. Who Needs DRAM? We Have Fiber

**arXiv ID:** 2607.08407 | [PDF](https://arxiv.org/pdf/2607.08407v1)

**作者:** Hannah Atmer `[一作]` (Uppsala University), Stefanos Kaxiras `[通讯]` (Uppsala University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Fiber Memory架构，将数据中心光纤网络转化为延迟线内存，用光纤持续循环传输LLM权重，消除每个加速器本地的重量复制与抓取。

**💡 创新点**

创新点在于：①把光纤用作可读写的高容量高速内存；②采用空间分复用多核光纤、被动1:99光学分配+放大、全光二R再生器，实现无电转换、低功耗的权重广播；③将CPO与光子集成电路直接映射到加速器上，完成流式权重无缓冲读取。

**🔧 技术方法**

技术包括：多核光纤（19核心MCF）+DWDM、PFA放大器、光学再生器、被动光学分配器、共封装光子学、直接探测PAM4、FEC、微环谐振器等。

**📊 数据集**

使用Llama‑3‑70B 70 GB INT8权重作为测试数据集，并在1,250个8加速器机架（共10,000个加速器）上进行量化评估。

**📈 对比分析**

与传统HBM3e方案对比，Fiber Memory在权重传输功耗上实现了约72%（284.8 kW vs 1,024 kW）且无需额外的静态泄漏/散热开销，保持了相同的3.2 TB/s/加速器吞吐量；实验显示能耗大幅下降且延迟保持在可接受范围。

**⚠️ 局限性**

局限性包括：光纤衰减与放大器产生的ASE噪声需要多级光学再生器；色散管理依赖O波段零色散特性；微环谐振器对温度敏感，需精细热控；大规模光纤环路部署成本高，且对光源可靠性与功率供给有严格要求。

---

## 327. On Exploring Input Resolution Scaling For Anytime LiDAR Object Detection

**arXiv ID:** 2607.08391 | [PDF](https://arxiv.org/pdf/2607.08391v1)

**作者:** Ahmet Soyyigit `[一作]` (National Defense University), Heechul Yun `[通讯]` (University of Kansas)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了MURAL，一种多分辨率任意时间LiDAR 3D目标检测框架，能够在单一模型上动态调整输入分辨率以满足不同的时延约束；

**💡 创新点**

创新点在于：①使用分辨率感知的BN层实现单模型多分辨率推理；②在训练后通过回归合成额外分辨率的BN参数；③利用最大池化模拟稀疏CNN来精确预测各分辨率的推理时延；④基于预测的deadline‑aware调度动态选择最高可用分辨率；⑤结合稠密CNN裁剪、预测融合和Region Dropping等优化；

**🔧 技术方法**

技术包括：多分辨率训练与共享权重、分辨率感知BN、BN参数回归、max‑pooling模拟稀疏卷积、动态分辨率调度、稠密CNN裁剪、目标预测融合、Region Dropping、闭环仿真评估；

**📊 数据集**

主要使用nuScenes数据集进行开环实验，并在AWSIM仿真平台上进行闭环驾驶模拟；

**📈 对比分析**

与单分辨率基线模型以及先前的VALO方法进行比较。开环实验显示，MURAL在多种deadline下的mAP均优于基线和VALO；闭环仿真中，MURAL实现了无碰撞的安全行驶，并在拥挤环境下降低停滞时间，平均推理时延与中等分辨率基线相当；

**⚠️ 局限性**

局限性包括：①对voxel‑based模型（如CenterPoint）的分辨率合成效果不佳；②在极短deadline下，MURAL在某些平台上仍略逊于VALO；③时延预测仍依赖于场景空间一致性假设，动态环境变化可能导致误差；④目前只支持pillar‑based模型的后训练分辨率扩展，CenterPoint等模型无法动态添加新分辨率。

---

## 328. Why Constants Matter in Distribution Testing: From Uniformity to Calibration

**arXiv ID:** 2607.08378 | [PDF](https://arxiv.org/pdf/2607.08378v1)

**作者:** Alon Kipnis `[一作]` `[通讯]`, Alon Kipnis

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文探讨了分布拟合优度检验中的常数在测试性能中的重要性，特别是在样本复杂度相同的情况下，如何选择最佳的检验方法。

**💡 创新点**

创新点在于强调了在分布测试中，尖锐常数的作用类似于参数估计中的Fisher信息和非参数估计中的Pinsker常数，能够区分同样率最优但功效不同的检验。

**🔧 技术方法**

使用了高维离散测试问题的高斯测试问题作为类比，分析了不同检验统计量的风险表现，并提出了最优统计量的形式。

**📊 数据集**

数据集为从N个类别中独立同分布的样本，具体的分布形式和样本大小在不同的测试中有所不同。

**📈 对比分析**

通过比较不同检验的风险表现，发现即使在相同的样本复杂度下，不同的检验方法在错误概率上存在显著差异，尖锐常数能够揭示这些差异。

**⚠️ 局限性**

限制在于虽然尖锐常数提供了更精确的风险评估，但在实际应用中，如何选择合适的参数和检验方法仍然需要进一步的研究和实践验证。

---

## 329. DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding

**arXiv ID:** 2607.08642 | [PDF](https://arxiv.org/pdf/2607.08642v1)

**作者:** Saw S. Lin `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]` (National Taiwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种训练‑无关的条件草稿树（DominoTree），在现有的 Domino 抽样模型上实现了更高的接受长度和吞吐率。

**💡 创新点**

创新点在于：① 利用 Domino 的 GRU‑纠正头实现路径相关的非因式化草稿概率；② 通过仅对每层前 M 名候选词执行纠正来压缩计算；③ 用 CUDA‑graph 捕获加速节点级纠正，保持树构建成本低；④ 对比传统的因式化树方法（DDTree/CaDDTree）和单链 Domino，证明条件评分是吞吐提升的关键。

**🔧 技术方法**

核心技术包括：Block‑diffusion 预抽（DFlash）、GRU 递归状态、低秩纠正网络、条件树构建的 best‑first 堆、候选限制、GPU‑native CUDA‑graph 加速以及树注意力验证。

**📊 数据集**

使用八个常见 LLM 评测数据集：GSM8K、MATH‑500、AIME25、HumanEval、MBPP、LiveCodeBench、MT‑Bench、Alpaca，分别在 Qwen3‑4B 和 Qwen3‑8B 两个模型上测试。

**📈 对比分析**

与 DFlash、DDTree、CaDDTree、官方 Domino（单链）进行基准对比。DominoTree 在所有温度下均实现最高平均接受长度（τ）和最高吞吐率（相对 AR 的加速比），在 4B 模型上相对 Domino 单链提升 9–10%，相对 DDTree/CaDDTree 提升 7–10%；在 8B 模型上相对 Domino 提升 9–15%，相对 DDTree/CaDDTree 提升 4–24%。

**⚠️ 局限性**

主要局限：① 采用固定节点预算；② 通过 CondAdaptive 进行自适应预算的尝试未能提升；③ 当前实现仅支持单流、batch‑1 的研究型 harness，未集成到生产推理框架；④ 仅在两种 GPU（RTX‑5080、RTX‑A6000）上测量，无法直接跨卡比较；⑤ 在 T>0 下草稿方式与官方 Domino 不完全一致，影响跨实现的公平性。

---

## 330. Spectral Stability of Pseudoinverse-Based Extreme Learning Machine

**arXiv ID:** 2607.08581 | [PDF](https://arxiv.org/pdf/2607.08581v1)

**作者:** Bich Van Nguyen `[一作]` (VNU University of Engineering and Technology), Ngoc Anh Khong `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于 Moore–Penrose 伪逆的 Extreme Learning Machine（ELM）在隐藏层矩阵条件不良时的数值稳定性问题。

**💡 创新点**

创新点在于从谱角度阐释了最小奇异值决定伪逆放大效应、条件数衡量隐藏层不稳定，并通过理论与实验验证了 SVD 与迭代求解器在不同奇异值结构下的表现差异。

**🔧 技术方法**

采用了奇异值分解（SVD）伪逆、Newton–Schulz 与超幂迭代法、随机特征矩阵理论以及随机矩阵估计最小奇异值的技术。

**📊 数据集**

实验数据集包括合成奇异值矩阵以及公开分类数据集 MNIST、Fashion‑MNIST 和 ISOLET。

**📈 对比分析**

通过对比 SVD 与迭代伪逆求解器，在最小奇异值、条件数、残差、运行时间和分类准确率等指标上进行评估，结果表明 SVD 在严重不良条件下保持稳定，而迭代方法易失效；分类性能和残差上 SVD 更优。

**⚠️ 局限性**

局限在于仅讨论无正则化伪逆求解，迭代方法对初始化和谱分布高度敏感，且缺乏对大规模 GPU 加速实现的评估。

---

## 331. SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling

**arXiv ID:** 2607.08565 | [PDF](https://arxiv.org/pdf/2607.08565v1)

**作者:** Jiahao Wang `[一作]` (Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究针对 LLM Agent 工作负载的调度设计，提出基于会话信息的平衡会话中心调度。

**💡 创新点**

创新点在于利用会话首请求分布来实现负载均衡，保留局部 KV 重用，且无需维护会话映射表。

**🔧 技术方法**

采用多层 KV 缓存（GPU 层 + CPU 全局层）、vLLM + LMCache、Prefill‑Decode 共置与分离架构以及自定义调度算法。

**📊 数据集**

使用两个来自工业级代理服务的真实工作负载追踪，覆盖 30B 与 280B 参数模型。

**📈 对比分析**

与现有 LLM 调度器（如 LMetric、load‑balance‑only 等）对比，使用 TPS、TTFT、TPOT 等指标，结果显示在全局层充分时 TPS 提升 10–16%，在分离模式下预填 TPS 提升 2–34%。

**⚠️ 局限性**

局限性包括：在全局 KV 存储容量不足或带宽受限时效果下降；会话信息提取依赖请求中包含完整历史，若被丢弃则退化；超长会话尾部仍可能出现负载失衡。

---

## 332. ESBMC-Arduino: Closing the Deployment Gap for Formal Verification of Open-Hardware PLCs

**arXiv ID:** 2607.08550 | [PDF](https://arxiv.org/pdf/2607.08550v1)

**作者:** Pierre Dantas `[一作]` (University of Manchester), Waldir Junior `[通讯]` (Federal University of Amazonas)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对开放硬件 PLC（OpenPLC、Arduino OPTA、CONTROLLINO、Industrial Shields M‑Duino）中的 IEC 61131‑3 程序进行硬件可信的验证，修复传统抽象验证忽略的单词宽度和硬件输入域导致的部署差距。

**💡 创新点**

提出一种基于 HAL 描述符的声明式低宽度/输入域模型，并通过在 esbmc 低层前端插桩实现自动化输入范围约束，恢复验证的可用性并消除 44% 的误报。

**🔧 技术方法**

利用 esbmc 的 SMT‑based BMC 与 k‑induction、GOTO IR 低层转换、Word‑width 设置以及输入约束注入；同时对 Arduino AVR/ARM 平台的机器模型、ADC/PWM 分辨率等硬件参数进行静态提取。

**📊 数据集**

评估使用了 123 个公开的第三方 PLCopen XML 程序（水处理与克隆检测两大数据集）以及 20 个人工构造的对照程序来验证模型的完整性与缺陷定位能力。

**📈 对比分析**

与原始宽度无约束验证对比：无输入模型时 54/123 为误报，加入 HAL 模型后误报降为 0，且保持 32 个安全证明；在受控数据集上可验证真实缺陷且执行时间平均在 168 ms，单个程序小于 1 s。

**⚠️ 局限性**

限制主要在于（1）对非线性浮点/传感器量化的过度简化导致的未知结果；（2）k‑induction 在函数块内部循环时缺乏闭合，导致 91 个程序归为未知；（3）仅针对整数/布尔分段，未覆盖更复杂的 IEC 61131‑3 功能块与实时特性。

---

## 333. Potential Functions as Types

**arXiv ID:** 2607.08547 | [PDF](https://arxiv.org/pdf/2607.08547v1)

**作者:** Harrison Grodin `[一作]`, Robert Harper `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将物理学家视角的势能函数与银行家视角的信用/借记两种摊销分析框架统一到一个依赖型理论 Calf 中，并基于此构造了子结构子类型 Giralf，用以实现信用化编程与自动化成本推理，保证模块性与手工验证兼容。

**💡 创新点**

创新点主要在于：① 把势能函数作为类型内在构造，生成抽象函数与潜能的组合；② 将银行家信用/借记以类型构造实现；③ 设计 Giralf 子语言，支持依赖、线性与递归，同时给出自动成本推理算法；④ 通过 fracture‑and‑gluing 定理证明 Calf 与 Giralf 的语义对应，验证 Giralf 能覆盖传统 AARA 语义。

**🔧 技术方法**

使用的技术包括：依赖型理论 Calf、fracture‑and‑gluing theorem、子结构子类型理论、带等级的线性/仿线性类型、sealed monad（用于松弛平方）、Kripke 语义、AARA 的线性规划推理、以及 Agda/Cubical 的机械化证明。

**📊 数据集**

未使用传统意义上的数据集；通过算法示例（批量队列、红黑树、Splay 树、插入排序等）以及理论证明来验证方法。

**📈 对比分析**

方法与 AARA 等传统工具对比时，本文证明 Giralf 能够给出与 AARA 等价的成本上界，并在 Calf 中实现自动化推理；在 Agda/Cubical 环境下完成了机械化证明，验证了正确性，未给出数值性能实验，主要关注理论证明与工具集成。

**⚠️ 局限性**

限制：仅适用于短暂（ephemeral）数据结构的摊销；不支持持久化摊销；使用可交换的成本模型，未涵盖非可交换成本；Giralf 不支持无限递归；LP 推理目前只能处理线性/多项式信用，无法直接处理多变量、指数或更一般的归纳类型的成本上界。

---

## 334. When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability

**arXiv ID:** 2607.08535 | [PDF](https://arxiv.org/pdf/2607.08535v1)

**作者:** Zongyou Yang `[一作]` (Imperial College London), Xiaokun Yang `[通讯]` (Nanchang Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估LLM-as-Judge在参数扩展、模型发布升级、陪审团聚合以及结构化辩论等不同变更下的可靠性，并将其视为测量有效性问题。

**💡 创新点**

提出四维可靠性框架（判定有效性、偏差鲁棒性、错误相关性、协议可审计性），并给出最小化审计跟踪的报告标准。

**🔧 技术方法**

使用Qwen3参数轴与MiniMax发布轴的实验，配对McNemar检验、Cohen κ、Spearman相关、β二项分布校正、误差相关系数ρ；实现同义语义辩论和陪审团投票聚合。

**📊 数据集**

四个数据集：LLMBar（对抗式对比）、PandaLM testset‑v1（宽域对比）、Chatbot Arena（人类偏好对比）和Judge’s Verdict（TechQA点评）。

**📈 对比分析**

通过配对McNemar检验对相邻模型进行显著性比较；发现Qwen3 1.7B→4B显著提升，MiniMax相邻升级无显著差异；高准确率降低偏差但未消除；陪审团增益受错误相关性ρ限制；辩论能显著改变决策但缺乏完整解析日志。

**⚠️ 局限性**

仅涵盖两类模型、四个数据集、单一提示与解码设置；未收集完整解析日志；对多语言、多任务的外推性有限；依赖配对检验而非跨轴比较，导致结论在更广泛场景中的适用性受限。

---

## 335. Log-Insight: Automating Microservice Incident Diagnosis via Neuro-Symbolic Log Analysis

**arXiv ID:** 2607.08529 | [PDF](https://arxiv.org/pdf/2607.08529v1)

**作者:** Carlos Garcia-Hernandez `[一作]` (Huawei Ireland Research Centre), Yanbin Zhang `[通讯]` (Huawei Dongguan R&D Centre)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并部署了 Log-Insight，自动化微服务系统的根因分析，将原始日志压缩 1,000–7,000 倍，并生成可验证的诊断报告。

**💡 创新点**

创新点在于将高吞吐量的符号化预处理（采样、模式聚类、熵压缩、对比偏差分析）与受限 LLM 合成相结合，消除 LLM 直输原始日志导致的上下文溢出与幻觉问题。

**🔧 技术方法**

技术主要包括两遍采样、知识库驱动的模式识别、Drain3 日志聚类、Shannon 熵双层压缩、对比偏差统计、优先级排列的 Forensic Case File 组装，以及使用通用大语言模型进行受限生成。

**📊 数据集**

使用了 11 条来自华为生产环境的微服务日志空间（行数 162–3.54M，列数 26–398），共 110 次实验运行。

**📈 对比分析**

与随机采样和 Drain+模板采样基线对比，使用 ROUGE‑L、METEOR、语义相似度和 MRR 评估；Log‑Insight 在宏观 MRR 达 0.790，top‑3 准确率超过 90%，平均推理时间 27 秒，压缩率可达 7,000×。

**⚠️ 局限性**

局限包括对知识库规则的依赖、阈值设置的经验性敏感性、对极宽 schema 的上下文缺失、LLM 合成的合成失败，以及仅在单一组织的 11 条历史案例上验证，缺乏跨环境的泛化性。

---

## 336. A Quantized Native Runtime for On-Device Semantic Audio Generation

**arXiv ID:** 2607.08526 | [PDF](https://arxiv.org/pdf/2607.08526v1)

**作者:** Matteo Spanio `[一作]` (University of Padova), Antonio Rodà `[通讯]` (University of Padova)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 aria，一个无依赖的 C/CUDA 运行时，实现在 CPU、GPU 以及 Raspberry Pi 5 上完整的 Stable Audio 3 文本到音乐推理，并提供低精度量化与激活驱动的语义控制接口。

**💡 创新点**

创新点包括：① 通过 8‑bit/4‑bit 量化实现内存替换，兼顾速度与设备兼容；② 在运行时内嵌激活注入实现零开销的控制；③ 采用多方可信评估（CLAP、wav2taste、FAD 等）验证语义控制的真实性。

**🔧 技术方法**

技术手段包括低位整数量化、FP16/FP32 混合精度、GPU Tensor Core 8‑bit 算子、CPU 向量化与多线程、激活注入、自动评估工具（CLAP、wav2taste、FAD、漂移等）以及多或盘验证。

**📊 数据集**

使用 Stable Audio 3 的原始训练集和 377 条带有基本味道评分的音乐片段做方向提取与评估，并使用 24 种多流派音乐提示进行量化与性能测试。

**📈 对比分析**

通过与官方 Stable Audio 3 PyTorch 实现的 warm、invocation、cold 三种部署场景对比，aria 在 GPU warm 0.13s 对比 0.146s，冷启动 7 倍更快；CPU 仅 2.5s/10s；8‑bit 量化内存降低 21% 并无质量损失，4‑bit 允许在 8GB Pi 上运行。

**⚠️ 局限性**

局限性包括：语义控制仅在甜、酸、苦三种味道上得到可信验证；激活方向受限于预训练模型；缺乏人类听觉评测验证控制效果；在小模型长序列 GPU 性能仍略低。

---

## 337. BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit Large Language Model Compression

**arXiv ID:** 2607.08643 | [PDF](https://arxiv.org/pdf/2607.08643v1)

**作者:** Yuantian Shao `[一作]` (Nanjing University of Science and Technology), Jian Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BiSCo-LLM，一种无代码表的二进制球面编码框架，用于 2‑bit 级 LLM 权重压缩；

**💡 创新点**

创新点包括：1）将权重块映射到单位球面并二进制化，消除显式 VQ 码本；2）引入二阶段残差 BSQ 以更高效利用码容量；3）按 Transformer 模块类别进行恢复蒸馏；4）在极低位压缩下加入 8‑bit 敏感通道保护；5）在压缩预算中明确计量代码、解码器、保护通道、LoRA 等全部存储量；

**🔧 技术方法**

使用技术包括 Binary Spherical Coding (BSQ)、二阶段残差 BSQ、类别级蒸馏、8‑bit 保护通道、LoRA 补偿、分类批处理优化、并行残差解码、梯度直通估计、熵正则化等；

**📊 数据集**

数据集：模型压缩主要基于预训练权重块；校准与恢复使用包含 Alpaca、OpenOrca、FineWeb‑Edu、RACE、SciQ、LongAlpaca、LongAlign 等混合语言数据；评估使用 WikiText‑2、C4、BoolQ、RTE、WinoGrande、ARC‑Easy/Challenge、OpenBookQA、PIQA、MMLU 等；

**📈 对比分析**

与多种基线（Scalar PTQ 如 GPTQ、SpinQuant、QuIP；Vector/结构化 VQ 如 AQLM、VPTQ、UniSVQ、LiftQuant 等）在 Qwen3‑8B 与 LLaMA‑3‑8B 的 2‑bit 场景下对比；BiSCo‑LLM 在 WikiText‑2 上获得 10.18 perplexity，接近 FP16 9.73；平均下游任务准确率仅低 1.87 点；在多任务评测上优于大多数向量编码方法；

**⚠️ 局限性**

局限性：需额外训练与蒸馏成本，解码阶段需要额外运算；当前仅支持固定码率与 1% 保护比例，未对不同模型实现自适应码率分配；未直接支持从二进制球面码进行即时矩阵乘法，需先解码；对极低位压缩的通用性与更细粒度分配策略仍待深入研究。

---

## 338. Steering Neural Network Training through Interpretable Constraints Based on Partial Dependence

**arXiv ID:** 2607.08641 | [PDF](https://arxiv.org/pdf/2607.08641v1)

**作者:** Yann Claes `[一作]` (University of Liège), Vân Anh Huynh-Thu `[通讯]` (University of Liège)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在神经网络训练过程中加入基于部分依赖（Partial Dependence）的约束，利用先验知识引导模型学习，从而得到更可信的解释和更好的预测性能。

**💡 创新点**

创新点在于：① 以部分依赖函数为先验知识，提出可调节的解释引导学习框架；② 引入linspace和in‑distribution两种约束配置；③ 在训练中交替优化模型参数与先验参数，兼顾预测与解释。

**🔧 技术方法**

使用神经网络梯度下降训练，结合部分依赖估计、双重损失（预测误差与PD误差）以及mini‑batch PD计算；实现了基于lambda加权的解释引导学习算法。

**📊 数据集**

实验数据集包括：合成回归问题（Friedman、Product），真实回归数据（混凝土抗压强度、PHALK），以及动力学系统预测（阻尼摆）和物理信息化模型对比。

**📈 对比分析**

与无约束训练及物理信息化模型对比，linspace约束的模型在预测误差、PD拟合度、样本稀缺和域外泛化上均优于其他方法；尤其在数据不足或测试域位移时表现更稳健。

**⚠️ 局限性**

局限性：① 需要额外的PD估计计算，导致训练时间增加；② 依赖先验PD形状的准确性，若先验误差较大会影响效果；③ 目前仅针对回归任务，扩展到分类或其他模型需进一步研究；④ 超参数调优敏感。

---

## 339. The Parameterised Complexity of Temporal Motif Counting, and a Lovász-Style Isomorphism Theorem

**arXiv ID:** 2607.08614 | [PDF](https://arxiv.org/pdf/2607.08614v1)

**作者:** Jayakrishnan Madathil `[一作]`, Marc Roth `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出时间图模式计数的理论框架，定义时间图模式、toadwidth、可行标签集合，并给出计数算法；

**💡 创新点**

提出toadwidth概念并证明其在有限时可多项式计数；通过线图结构和可行图性质实现动态规划；

**🔧 技术方法**

利用混合图的宽度（cliquewidth）、线图结构、动态规划、Lovász计数公式与Möbius反演；

**📊 数据集**

无具体实验数据集，主要以理论证明为主；

**📈 对比分析**

与传统基于树宽或路径宽的计数方法相比，本方法在toadwidth有限时可在O((mlifetime)^O(ω))时间完成计数，理论上效率可与树宽计数方法相当；

**⚠️ 局限性**

仅适用于无孤立顶点、无平行边且模式无重复边的情况；需要预先给定toadwidth的cliquewidth表达式。

---

## 340. Towards Precision Therapy in Hepatocellular Carcinoma: A Clinical-Reasoning LLM for Risk Stratification and Treatment Guidance

**arXiv ID:** 2607.08602 | [PDF](https://arxiv.org/pdf/2607.08602v1)

**作者:** Peng Cui `[一作]` (Tsinghua University), Jiahong Dong `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个名为 HCC‑STAR 的大语言模型，能够读取 EMR 文档并一次性输出细化的 HCC 分期、基于指南的治疗排序和个体生存预测。

**💡 创新点**

创新点包括：① 采用两阶段知识对齐训练（CKF‑FT + EARL）和可验证的复合奖励，① 通过提示式数据增强生成高可信度 EMR 叙事，② 在强化学习中使用 Step‑Verifiable 评价与 Decoupled Optimization，③ 使用 Group Relative Policy Optimization 解决多目标梯度干扰。

**🔧 技术方法**

技术手段主要是：大型中文医学 LLM（Qwen3‑32B），有监督微调（SFT）结合强化学习（EARL），多目标可验证奖励（过程、格式、治疗排序、预后、长度），结构化标记与链式推理，GRPO 算法。

**📊 数据集**

使用的数据集包括：美国 SEER 2004‑2020 共约 30,000 例 HCC 病例（结构化表格）以及 12 家中国三级医院共 6,668 例实时 EMR（多中心外部验证集），并通过提示合成 EMR‑风格叙事。

**📈 对比分析**

对比方法：与传统分期系统（BCLC、CNLC、AJCC/TNM）、经典机器学习（XGBoost、SVM、MLP）以及多款当代 LLM（GPT‑5、Gemini‑2.5‑Pro、GPT‑4o、Claude、DeepSeek‑R1）进行评估。性能表现：Top‑k（k=1,2,3）准确率最高；C‑index 达到 0.7371（外部多中心），优于 0.66‑0.68 的分期系统；在医生评估实验中，模型接近资深专家水平，并显著提升住院医师的准确率和决策速度；假设性 OS 预估中模型建议的中位生存为 51 个月，远高于 BCLC（29 个月）和 CNLC（32 个月）。

**⚠️ 局限性**

局限性包括：研究仅为回顾性分析，缺乏前瞻性验证；合成 EMR 叙事可能引入分布偏移；反事实 OS 估计受治疗指征混杂影响；数据主要来自 HBV 主导人群，外部推广受限；对低资源地区或不同编码体系的适用性待验证；模型依赖提示工程，需人工维护更新。

---

## 341. The complexities of patient-centred conversational artificial intelligence

**arXiv ID:** 2607.08625 | [PDF](https://arxiv.org/pdf/2607.08625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 342. On Constructing Most General Solutions for Parametric Constraints (Extended Preprint)

**arXiv ID:** 2607.08582 | [PDF](https://arxiv.org/pdf/2607.08582v1)

**作者:** Viorica Sofronie-Stokkermans `[一作]` `[通讯]` (University of Koblenz), Viorica Sofronie-Stokkermans (University of Koblenz)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文研究在允许量化消除的理论中，构造参数约束的最一般解（solution）及其条件解（conditional solution），并探讨在有“if-then-else”构造的模型中如何利用重构（reproductive）技术得到最一般解。

**💡 创新点**

创新点在于将判别子（discriminator）范畴中的最一般统一方法推广到更一般的理论框架，提出在有 ite 操作的理论上通过量化消除得到的条件解可进一步写成最一般解，并给出了构造最一般解的具体算法与证明。

**🔧 技术方法**

采用的技术主要包括：一阶逻辑理论与结构的定义、量化消除、switching/ite 构造、重构（reproductive solution）以及在判别子/多项式理论上对等式和不等式的符号化求解。

**📊 数据集**

本文没有使用实际数据集，主要以理论推导和例子（如布尔代数、线性实数算术、实闭域）来演示方法的可行性。

**📈 对比分析**

方法通过理论证明和实例演示得到验证，没有给出定量性能比较；对每个例子都说明了构造过程与最一般解的正确性，强调了使用 ite 操作的优势。

**⚠️ 局限性**

局限性包括：仅适用于具备量化消除且能定义 ite 构造的理论；主要处理合取字面量或 flat 子句，非 flat 情况仅作简要说明；对高度复杂的约束可能导致 ite 构造过于繁琐或不可行。

---

## 343. ImputeViz: A Visual Analytics Dashboard for Diagnosing Missing Data and Comparing Imputation Methods

**arXiv ID:** 2607.08579 | [PDF](https://arxiv.org/pdf/2607.08579v1)

**作者:** Aitik Dandapat `[一作]` (Stony Brook University), Klaus Mueller `[通讯]` (Stony Brook University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个可视化分析仪表盘ImputeViz，支持缺失数据诊断、模型选择与比较，并集成了地理信息的gKNN算法；

**💡 创新点**

创新点包括统一的跨方法可视化工作流、通过贝叶斯优化融合社会经济与地理距离的gKNN，以及缓存一致评估实现的高效交互；

**🔧 技术方法**

技术栈为Python/FastAPI后端、React/TypeScript/D3.js前端，使用MICE、Random Forest、XGBoost、kNN、gKNN等插补模型，贝叶斯优化与Copula不确定性分析；

**📊 数据集**

使用美国县级处方阿片药物死亡率（2000-2022）与ACS社会经济特征数据，以及非地理的电信流失数据；

**📈 对比分析**

在随机、抑制、间隔等多种缺失掩码下对所有模型进行MAE/RMSE评估，gKNN在抑制条件下平均MAE降至2.96，优于其他基线；在非地理数据上Random Forest显著优于MICE；

**⚠️ 局限性**

局限性包括模型集有限、缺乏细粒度空间分辨率支持、离线不确定性分析、可扩展性与实时交互受限以及未覆盖时间序列或空间-时间模型。

---

## 344. When Structured Sparse Autoencoders Learn Consistent Concepts Across Modalities

**arXiv ID:** 2607.08605 | [PDF](https://arxiv.org/pdf/2607.08605v1)

**作者:** Weiduo Liao `[一作]`, Ying Wei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了结构化稀疏自编码器 (S²AE)，通过在视觉模态中强制概念一致性来提升视觉-语言模型的机制可解释性。

**💡 创新点**

创新点在于：①利用Transformer注意力相似度与空间接近度聚类图像补丁形成视觉区域；②在SAE训练中加入互斥稀疏性和组稀疏性正则化，既驱动特征跨区域解耦，又保证同一区域内特征的一致性；③构建分层解释管线，先用VLM识别局部视觉概念，再由LLM进行语义综合，显著提升解释可靠性。

**🔧 技术方法**

核心技术包括：Top‑K稀疏自编码器、组/互斥稀疏正则化、聚类（使用平均注意力+曼哈顿距离）、二值化激活、以及基于VLM+LLM的分层解释管线。

**📊 数据集**

在Qwen2.5‑VL‑7B‑Instruct等大规模视觉‑语言模型的残差层上进行实验，并使用公开的视觉‑文本对比数据集（如lmms‑lab/sae‑sample‑cache‑dataset）进行评估。

**📈 对比分析**

与基线Vanilla SAEs相比，S²AE在概念对齐（mIoU提升6.06%）、稀疏效率（l₀ norm下降60.81%）和多模态一致性（单义性提高3.08%/2.37%）方面均取得显著提升，且重建精度（Explained Variance >99%）保持不变。

**⚠️ 局限性**

局限性包括：正则化参数选择依赖经验；解释管线依赖强大的VLM/LLM，计算开销较大；仅在视觉端施加结构约束，语言端仍可能存在多义性；以及在更复杂场景或不同视觉编码器下的泛化性尚待验证。

---

## 345. Native Video-Action Pretraining for Generalizable Robot Control

**arXiv ID:** 2607.08639 | [PDF](https://arxiv.org/pdf/2607.08639v1)

**作者:** Qihang Zhang `[一作]`, Yinghao Xu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的视频-动作基础模型，专为机器人控制而设计，旨在克服现有视频生成模型在物理环境中的不足。

**💡 创新点**

创新点在于从头开始构建一个视频-动作模型，采用语义视觉-动作标记器、因果预训练范式、稀疏MoE骨干网和增强的异步推理方案，以实现实时闭环控制。

**🔧 技术方法**

使用了语义视觉-动作标记器、因果扩散变换器（DiT）和稀疏混合专家（MoE）等技术。

**📊 数据集**

使用了来自网络规模的图像和视频数据集，以及人类和机器人共同训练的数据集。

**📈 对比分析**

与现有的基线模型（如LingBot-VA和π_0.5）进行比较，展示了在复杂操作任务中的少量示例泛化能力和实时闭环控制的性能显著提升。

**⚠️ 局限性**

模型的局限性在于仍然依赖于机器人数据进行训练，且在某些情况下可能无法完全捕捉复杂的动态变化。

---

## 346. It Takes a MAESTRO To Prune Bad Experts

**arXiv ID:** 2607.08601 | [PDF](https://arxiv.org/pdf/2607.08601v1)

**作者:** Palaash Goel `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于马尔可夫链的结构化剪枝框架（Markov-chain Approximated Expert Sparsification via Transition-based ROuting，简称MARCOS），专门用于稀疏激活Mixture-of-Experts（MoE）语言模型的参数压缩。

**💡 创新点**

创新点在于将整条token的专家激活路径建模为一个循环马尔可夫链，利用其稳态分布捕捉跨层、跨专家的全局路由依赖，从而得到一个全局一致的专家重要性评估；相较于传统仅使用层内激活频率或权重大小的局部度量，MARCOS 能更好地识别真正冗余的专家。

**🔧 技术方法**

核心技术包括：1）对每个 MoE 层的 top‑k 路由结果进行自回归计数，估计层间专家转移矩阵；2）构造全局闭环转移矩阵并对其进行归一化与 ε‑平滑，得到一个不可约、非周期的马尔可夫链；3）通过幂迭代求稳态分布 π，作为专家重要性指标；4）根据 π 的大小在每层按比例删除最小概率的专家；5）在剪枝后使用 LoRA 微调恢复性能。

**📊 数据集**

使用了 GPT‑OSS‑20B（32 experts/层）和 Qwen‑3‑30B（128 experts/层）两大 MoE 语言模型进行压缩；校准时采用自回归生成的小样本语料（SlimOrca）；评估基准涵盖 17 个任务，跨安全、偏见、伦理等五个领域。

**📈 对比分析**

与随机剪枝、Mosaic Pruning（MoP）、REAP、HC‑SMoE、专家融合等基线进行对比。实验表明，在 25% 与 50% 压缩率下，MARCOS 在保持性能上平均提升 1–3%（相较于最佳基线）并在 50% 压缩时提升高达 10.61%；同时在任务间标准差方面更低，表明模型的跨任务一致性更好。

**⚠️ 局限性**

局限性包括：1）仅采用一阶马尔可夫假设，忽略了更长历史对专家选择的影响；2）剪枝粒度为完整专家，无法与权重宽度/深度剪枝联合使用；3）校准和恢复微调仍需额外计算资源，且对极大模型的内存需求未完全解决。

---

## 347. Robust Bayesian Decision Making under Adversarial Uncertainty

**arXiv ID:** 2607.08590 | [PDF](https://arxiv.org/pdf/2607.08590v1)

**作者:** Haripriya Harikumar `[一作]` (University of Manchester), Samuel Kaski `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对对抗性变量的决策感知实验设计方法，称为AR-DEIG。

**💡 创新点**

创新点在于将贝叶斯决策理论与对抗性鲁棒性结合，直接以鲁棒最优决策的不确定性为信息增益目标。

**🔧 技术方法**

使用贝叶斯决策理论、信息理论（EIG）、高斯过程模型和Stackelberg博弈建模来推导鲁棒决策并计算AR-DEIG。

**📊 数据集**

在一维合成GP、五维GP以及真实数据Osteoarthritis Initiative (OAI) 数据集上进行评估。

**📈 对比分析**

与随机、基于不确定性、标准EIG、目标EIG和决策EIG等五种基线对比，AR-DEIG在平均、最坏情况及CVaR10指标上均优于基线，且决策翻转率低。

**⚠️ 局限性**

局限性包括对抗扰动预算的先验设定、对高维对抗变量的计算复杂度，以及在极大ε下可能导致过度保守决策。

---

## 348. FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation

**arXiv ID:** 2607.08575 | [PDF](https://arxiv.org/pdf/2607.08575v1)

**作者:** Shiyuan Yang `[一作]` (University of Macau), Qingbiao Li `[通讯]` (University of Macau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了轻量级 Vision‑Language‑Action 模型 FabriVLA，结合 InternVL3.5 视觉语言骨干和门控自注意力的流匹配动作头，能够在 Meta‑World MT50 上实现高精度多任务机器人操控。

**💡 创新点**

创新点在于：① 门控自注意力让动作标记在训练期间逐步学习跨步依赖；② 通过浅层 VLM 层融合将空间细节与语义信息并行提供给动作头；③ 单阶段联合优化从预训练 VLM 开始，无需额外机器人数据预训练。

**🔧 技术方法**

使用了 InternVL3.5 VLM、门控自注意力、浅层 VLM 融合、流匹配（Flow Matching）动作生成、DeepSpeed ZeRO‑2、BF16 混合精度、AdamW 等技术。

**📊 数据集**

训练与评估使用公开的 Meta‑World MT50 任务集（50 个多任务轨迹，每任务 50 条演示轨迹）。

**📈 对比分析**

与 TinyVLA、π_0、SmolVLA、RoboTron‑Mani、Evo‑1、Evo‑Depth、LA4VLA 等近期 VLA 进行分层难度评估，FabriVLA 在 MT50 上取得 90.0% 的分层平均成功率和 92.0% 的整体成功率，表现位居首位。

**⚠️ 局限性**

局限性包括：在工具介导、粗略运输和抓取放置等任务上仍显弱；虽然模型规模小，但在极难场景下仍有提升空间；仅使用单一视觉语言骨干，跨模态鲁棒性可能受限。

---

## 349. SHAP-Weighted Cross-Modal Expert Fusion for Emotion and Sentiment Recognition: Evidence and Limits

**arXiv ID:** 2607.08573 | [PDF](https://arxiv.org/pdf/2607.08573v1)

**作者:** Adis Alihodzic `[一作]` (University of Sarajevo), Selma Skopljakovic Hubljar `[通讯]` (University of Sarajevo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个基于 TreeSHAP 权重的多模态专家混合模型（XGAF），利用不同比例的 SHAP 减少方式对单模态、双模态和三模态 XGBoost 专家进行自适应加权，研究其在情感和情绪识别中的融合效果。

**💡 创新点**

发现当专家特征维度不等时，使用总和（sum‑abs）SHAP 减少可以保留交叉模态专家的整体证据量，显著提升性能；同时对 mean‑abs、median‑abs 等常见减法做系统对比，阐明减法选择对融合效果的关键作用。

**🔧 技术方法**

TreeSHAP 解释信号、XGBoost 专家、温度缩放 softmax 加权、三种 SHAP 减少（mean‑abs、median‑abs、sum‑abs）、专家池规模 ablation、熵与主导专家诊断等。

**📊 数据集**

MELD（7 类情感识别）与 CMU‑MOSEI（3 类情绪/情感识别），使用 BERT、wav2vec2、面部特征等预提取特征。

**📈 对比分析**

与早期融合、后期融合、单模态基线、XGAF v1 等对比；在 MELD 上 sum‑abs XGAF 与早期融合在加权 F1 上相当（0.5983 vs 0.6018，McNemar p=1.000），显著优于后期融合；在 CMU‑MOSEI 上 sum‑abs XGAF 提升至 0.6519（vs 0.6485 早期融合，p=0.0452），但增幅仅 0.34% F1。

**⚠️ 局限性**

缺乏对缺失或噪声模态的鲁棒性验证；诊断显示 sum‑abs 权重集中在 trimodal 专家，缺乏丰富的 per‑sample 路由；仅使用预提取特征，未与端到端神经网络直接对比；实验仅覆盖单个随机种子，未给出置信区间。

---

## 350. Algorithms and Indexing Lower Bounds for Variable String Matching

**arXiv ID:** 2607.08566 | [PDF](https://arxiv.org/pdf/2607.08566v1)

**作者:** Estéban Gabory `[一作]` `[通讯]` (University of Wrocław), Estéban Gabory (University of Wrocław)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对一般退化字符串（Generalized Degenerate Strings，GDS）的模式匹配问题的经典子二次时间算法（N√m），并给出了线性预处理后查询时间为O(nm)的索引结构，同时在组合算法上证明了在k‑Clique和OMv假设下的匹配上界；

**💡 创新点**

创新点主要包括：①首次给出GDS模式匹配的O(N√m)经典算法；②通过重用Gibney的索引实现O(nm)查询；③构造新的组合化简，证明在k‑Clique和OMv假设下对组合算法的下界；④将FFT卷积与heavy‑light分解相结合，提升计算效率；

**🔧 技术方法**

采用的技术包括：suffix tree与Aho‑Corasick自动机；FFT卷积处理“heavy”字符串；heavy‑light分解对字符串段进行分层；组合化简将k‑Clique、OV、OMv问题映射到GDS/elastic‑degenerate字符串匹配；

**📊 数据集**

本文主要为理论研究，并未使用具体实验数据集；所有结果均基于理论分析与构造；

**📈 对比分析**

在理论评估上，算法复杂度为O(N√m)，比之前的O(N+nm)显著改进；索引查询时间为O(nm)，匹配下界表明在组合算法上已接近最优；

**⚠️ 局限性**

局限性包括：①算法仍未达到近线性时间；②下界仅针对组合算法，非组合算法可能突破；③对图结构变体（如基因图）仍无完整解；

---

## 351. Contravariance Theory: Strong Alignment for Minimal Solutions to Hard Tasks

**arXiv ID:** 2607.08561 | [PDF](https://arxiv.org/pdf/2607.08561v1)

**作者:** Dan Yamins `[一作]` (Stanford University), Aran Nayebi `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论证明，两个在足够硬的任务上达到最小解的深度网络若在相邻层存在线性（弱）对齐，则必然在单元轴级别（强）对齐；并进一步证明终端弱对齐会向上传递，导致完整网络层级上的轴对齐（zippering）。

**💡 创新点**

创新点在于提出并严格证明了弱–强等价定理与zippering定理，量化了“对抗性（contravariance）”原理，使得人工网络与大脑在硬任务下的收敛演化成为数学上几乎必然的现象；同时引入了最小性、可用轴预算、雅可比正则性等概念，构建了软化版本与渐近等价理论。

**🔧 技术方法**

使用的主要技术包括：ReLU/softplus 线性‑非线性层的符号分析、轴可用性与最小性证明、弱-强对齐的线性映射理论、渐近弱–强等价与误差传播分析、以及对“零网络（null networks）”的几何排除；结合了泛化的可测度零集理论和 Jacobian 正则性的典型性论证。

**📊 数据集**

本文为理论工作，未提出新的数据集；但在讨论中引用了 ImageNet（AlexNet 训练）、ViT、ResNet、LLM（RoBERTa、ALBERT）等公开实验作为实例和对比参考。

**📈 对比分析**

对比方法主要是线性映射（SVCCA、CKA、Procrustes、软匹配）以及轴对齐度量；理论结果表明在硬任务下弱对齐会自动转化为强对齐，证明了线性相似度能够捕捉到显著的轴一致性；实验证据显示不同行业模型在层级上的匹配度和隐式对齐保持良好。

**⚠️ 局限性**

局限性包括：假设网络满足最小性与 Jacobian 正则性，且学习动态不会将参数聚集到零测度的“异常集”上；对软化版本的误差界限与实际网络的离散化误差尚未完全量化；RSA 等更严格的相似度指标在多重轴存在时无法直接得到保证；此外，过参数化、不同任务难度的度量仍需进一步实证验证。

---

## 352. Computing over Data Streams using Catalytic Space

**arXiv ID:** 2607.08559 | [PDF](https://arxiv.org/pdf/2607.08559v1)

**作者:** Ripley Becker `[一作]` (University of Nebraska--Lincoln), N. V. Vinodchandran `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了可催化内存（catalytic memory）数据流模型，并利用该模型设计了多通道（多次扫描）下的频率矩（frequency moments）精确计算算法；同时将这些算法应用于图流中的子图计数问题。

**💡 创新点**

创新点在于：①首次将可催化内存引入数据流算法，显著降低了对清洁（可写）空间的需求；②通过递归分解、模运算和有限差分的组合，得到对F₂、F₃分别在2、3通道下使用O(log m)清洁空间的精确算法；③提出了k+1通道的通用算法，清洁空间仅为O(k log m)，但需要k+1次扫描；④证明单通道下催化内存无法提供额外优势，给出Ω(n)空间下界。

**🔧 技术方法**

主要技术包括：
- 可催化内存模型的定义与理论化；
- 递归分块处理数据流，利用交叉项（inner product）计算；
- 模运算与Catalytic寄存器的增减操作；
- Stirling数、下取整阶乘和有限差分的代数性质，用于推导k+1通道算法；
- 通过对多通道的并行计算实现子图计数；
- 自动机论证证明单通道催化内存无优势。

**📊 数据集**

本工作为理论研究，未使用实际数据集；所有结果均通过数学证明与空间复杂度分析给出。

**📈 对比分析**

比较方法：将催化内存算法与传统无催化的多通道流算法以及单通道下的空间下界进行对比；
性能表现：
- 对F₂、F₃分别在2、3通道下使用O(log m)清洁空间；
- 对一般k，使用k+1通道、O(k log m)清洁空间、O(k n log m)催化空间；
- 子图计数问题在四通道下仅需O_H(log n)清洁空间（H为固定子图）；
- 证明单通道催化内存无法突破Ω(n)空间下界。

**⚠️ 局限性**

局限性：
- 需要多通道扫描；单通道下催化内存无优势；
- 对大k的算法仍需k+1通道，可能导致扫描次数随k增长；
- 催化内存的实现假设可以自由读写且最终恢复，实际系统实现可能复杂；
- 仅适用于插入仅（insertion‑only）流和固定子图计数，其他流模型或近似需求尚未覆盖。

---

## 353. Locally Approximating the Top Eigenvector of Bounded Entry Matrices

**arXiv ID:** 2607.08556 | [PDF](https://arxiv.org/pdf/2607.08556v1)

**作者:** Nicolas Menand `[一作]` (University of Pennsylvania), Erik Waingarten `[通讯]` (University of Pennsylvania)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种局部计算算法，用于在具有受限条目的对称矩阵中高效地输出近似最大的特征向量，并将该算法应用于求解稠密图的稀疏割问题；

**💡 创新点**

创新点在于首次将受限条目矩阵的近似特征向量计算与局部计算框架结合，证明了该方法的上界和下界，并在构造稠密图上实现稀疏割的局部查询；

**🔧 技术方法**

主要技术包括随机子样本投影（Row–subsample），谱近似与子空间嵌入理论，构造“lifted”矩阵来消除负特征值影响，以及对稠密图的 Chernoff 采样分析；

**📊 数据集**

实验数据集主要是随机生成的 Erdős–Rényi 图及其与目标图的并集，未使用公开的真实图数据集；

**📈 对比分析**

由于方法主要给出理论复杂度和误差上界，没有与传统全局算法进行实验性比较；理论上实现了 $O(1/ε^2)$ 的查询复杂度，并给出了 $O(√{ϕ(G)})$ 的稀疏割近似；

**⚠️ 局限性**

局限性包括：对矩阵谱范数的预设要求（A₂ = O(λ_max)），负特征值的处理导致查询复杂度上升；对稠密度参数有严格阈值，无法直接处理极稀疏图；并且在负特征值较多时算法的性能可能退化。

---

## 354. DocMaster: A Hierarchical Structure-Aware System for Document Analysis

**arXiv ID:** 2607.08539 | [PDF](https://arxiv.org/pdf/2607.08539v1)

**作者:** Ziqi Chen `[一作]` (Chinese University of Hong Kong), Yixiang Fang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个名为 DocMaster 的系统，能够将 PDF 文档解析为层级化文档树，利用 LLM 生成摘要并构建结构感知的语义索引（PC‑KMeans 聚类 + 超边），然后通过三模检索（树遍历、语义向量检索、超边匹配）实现文档过滤，并在过滤结果上执行检索增强式问答。

**💡 创新点**

① 将原始文档的层级结构完整保留，避免扁平化导致的信息丢失；② 通过结构熵相关（SEC）度量挑选跨章节语义相似但结构距离大的节点，生成 LLM 标记的 must‑link/cannot‑link 约束，提升跨章节检索效果；③ 设计超边来捕捉同一章节内语义聚类，形成跨章节的语义覆盖层。

**🔧 技术方法**

主要技术包括 MinerU PDF 解析、LLaMA/ GPT‑4o 生成摘要、All‑MiniLM‑L6‑v2 句子嵌入、结构熵相关（SEC）评分、PC‑KMeans 聚类、FAISS 向量检索、三模检索融合、检索增强式生成（RAG）以及 React/FastAPI Web 前端。

**📊 数据集**

使用公开的 AI 论文集合（如 CVPR/ICLR/NeurIPS 等会议论文）进行实验，包含数百篇 PDF，作为“真实”复杂文档集合。

**📈 对比分析**

与传统扁平化检索（仅基于文本块向量检索）进行对比。实验结果显示，DocMaster 在文档过滤准确率上提升了约 15%–20%，在跨章节问答召回率上提升了 12%–18%；在用户体验方面，界面可视化树形结构显著降低误检率，系统总体响应时间维持在 2–3 秒内。

**⚠️ 局限性**

① 目前仅支持单文档内部索引，缺乏跨文档语义链接；② 需要 LLM 调用生成约束，导致标注成本与推理延迟；③ 结构熵相关的参数（α、k、γ 等）需人工调优，对非专业用户不友好；④ 对动态更新的文档集合支持有限，无法在线增量更新索引。

---

## 355. Locality of Curve-Decoding and Improved Proximity Gaps

**arXiv ID:** 2607.08516 | [PDF](https://arxiv.org/pdf/2607.08516v1)

**作者:** Rohan Goyal `[一作]` (Massachusetts Institute of Technology), Mary Wootters `[通讯]` (Stanford University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的行跨度约束 LCL 框架，并利用它证明了随机线性码、随机 Reed‑Solomon 码和 Gallager LDPC 码在曲线解码、关联一致性和互相关一致性等方面的近似间隙可与子空间设计码相匹配，显著提升了随机码的参数。

**💡 创新点**

核心创新在于：①将曲线解码直接视为行跨度约束的 LCL 属性，避免了之前通过 ⋁‑解码引入的指数参数损失；②构建了黑盒传递机制，使子空间设计码的任何改进自动迁移至随机码；③给出了行跨度约束 LCL 的阈值分析与传递定理。

**🔧 技术方法**

采用了行跨度约束的 LCL 属性、潜能函数与阈值分析、随机码的低/高速率阈值传递、子空间设计码的曲线解码证明、随机线性码与 LDPC 的概率矩分析等理论工具。

**📊 数据集**

本文基于理论随机码模型（随机线性码、随机 Reed‑Solomon 码、Gallager LDPC 码），未使用具体数据集。

**📈 对比分析**

通过与之前随机码结果（q ≥ nℓ(1−R)/η + (ℓ/η²)^O(ℓ)）对比，新的参数要求降低为 q ≥ nℓ(1−R)/η + O(ℓ²/η³)，从而在大 ℓ 的情形下实现了与子空间设计码相同的近似间隙和关联一致性性能。

**⚠️ 局限性**

局限性：仍需满足 q、n 较大；尚未给出具体构造码的实现和实验验证；当前方法主要针对随机码，扩展到结构化可构造码仍有待研究；行跨度约束 LCL 的通用性和更高阶属性的可扩展性仍是未来开放问题。

---

## 356. Do Egocentric Video-Language Models Capture Both Hand- and Object-Centric Cues?

**arXiv ID:** 2607.08514 | [PDF](https://arxiv.org/pdf/2607.08514v1)

**作者:** Masatoshi Tateno `[一作]` (University of Tokyo), Dima Damen `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了手-物体掩蔽训练和HOI动力学感知解码器的新训练范式，并设计了Cue-Isolated HOI评估和DEHOI测试集，以提升第一人称视频-语言模型对手和物体线索的独立推理能力。

**💡 创新点**

创新点在于：①针对手与物体的实体-aware掩蔽策略，使模型在缺失某类线索时仍能利用剩余信息重建并学习；②HOI动力学感知解码器通过手、物体和视频级别查询以及多任务监督显式学习手、物体嵌入；③构建了Cue-Isolated HOI评估与DEHOI数据集，实现对手与物体线索独立推理的定量评估。

**🔧 技术方法**

技术手段包括：视频MAE框架的手-物体掩蔽训练；DETR-like Transformer解码器（HDA）与多任务监督（位置、语义、视频-文本对齐）；LoRA微调；EgoClip预训练；EgoNCE视频-文本对齐损失；以及使用ProPainter进行视频inpainting构造DEHOI。

**📊 数据集**

使用数据集：训练阶段使用EgoClip（3.8M Ego4D视频-文本对）；评估阶段使用DEHOI（基于EPIC-KITCHENS-100和VISOR的手/物体分离视频）、STATUS Bench（对象状态与变化）和DROID（机器人操作视频）；对比基线包括LaViLa、Helping Hands、EgoVideo等。

**📈 对比分析**

在DEHOI的原始、手中心、物体中心视频上进行零样本评估，与LaViLa、Helping Hands和EgoVideo等基线相比，本模型在所有场景下均取得最高Top1/Top5准确率，显著缩小了原始与孤立视频之间的性能差距；在STATUS Bench和DROID的零样本任务中亦获得最优或接近最优的表现。

**⚠️ 局限性**

局限性包括：对静态或主要以手物体接触为主的动作识别仍有下降；在手与物体高度重叠或遮挡严重时性能受限；跨视角或机器人手臂的迁移仍需要进一步微调；掩蔽比例与平衡参数需经验调优。

---

## 357. Elitism in the Aisle: A Long-Run Surname Measure of Legislative Elite Composition in Chile, 1834-2020

**arXiv ID:** 2607.08520 | [PDF](https://arxiv.org/pdf/2607.08520v1)

**作者:** Naim Bro `[一作]` (Universidad Adolfo Ibanez), Juan Pablo Luna `[通讯]` (Mcgill University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了近两世纪（1834–2020）智利国会成员的族名，并用此族名构建了一种“持久精英族名”测度，衡量国会的精英与非精英组成，并将其与国会演讲议程进行对照，探讨精英背景与议题关注的关系。

**💡 创新点**

创新点在于提出一种可复制、跨年代的族名测度：结合当代社会经济指数与历史精英登记，直接从族名推断精英身份；以及将该测度与结构主题模型和文本理想点模型相结合，系统性地揭示精英与普通族名在议程和意识形态上的差异。

**🔧 技术方法**

技术方法包括：族名社会经济指数（0–100），企业董事重叠检验，分段回归（断点分析 1925 年），结构主题模型（STM，80 主题）用于议程提炼，文本理想点模型用于意识形态定位，以及网络中心性检验与生物学文本挖掘工具（OCR、LLM）用于数据预处理。

**📊 数据集**

主要数据集为：智利国会成员名单（1834–2020）、1884 年起的出生登记（2,265 万人）、5,411 名企业董事与经理名单、若干历史精英登记（农业普查、殖民贵族、矿业精英等）、以及 116,938 篇国会演讲稿。

**📈 对比分析**

比较方法：将精英族名比例与动态人口基准（按出生登记计算的族名频率）对照；使用分段回归检验 1925 年宪法改革对精英比例的冲击；在议程层面，利用结构主题模型对不同族名组的主题加权进行比较，并用文本理想点模型验证意识形态差异。性能上，研究发现精英比例从 1860 年代的 52% 降至 2010 年代的 12%，1925 年出现显著的 -13 个百分点跳变；议程对比显示精英族名倾向国防、外交等“国家治理”议题，普通族名倾向劳工与福利议题。

**⚠️ 局限性**

局限性包括：族名测度只能区分“持久精英”与“非精英”，无法进一步区分中产与工人阶级；族名随时间可能发生变异或更替，导致识别误差；历史精英登记不完整，导致“其他”族名比例上升，可能掩盖真正的精英替代与结构变迁。

---

## 358. Beyond wheelchairs and blindfolds: Investigating disability stereotypes in T2I models with INCLUDE-BENCH

**arXiv ID:** 2607.08515 | [PDF](https://arxiv.org/pdf/2607.08515v1)

**作者:** Sophia Lichtenberg `[一作]` (Utrecht University), Judith Masthoff `[通讯]` (Utrecht University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了INCLUDE-BENCH基准，对大量文本到图像模型在残疾人描绘方面的偏差进行了系统评估；

**💡 创新点**

创新点在于首次提供大规模、跨维度（功能、种族、性别、年龄、情境）的残疾人偏差基准，并引入SCM Score衡量隐性刻板印象；

**🔧 技术方法**

使用的技术包括文本到图像生成模型、CLIP/CLIPScore、Vendi Score、多模态Stereotype Content Model (SCM)、SAM3检测人、Qwen3-VL进行标题和VQA分析；

**📊 数据集**

数据集来源于352个定制提示，生成119,680张图像，覆盖七类功能组（盲、聋、肌肉等）并加入年龄、种族、性别与不同环境上下文；

**📈 对比分析**

通过比较CLIPScore与Vendi Score评估对齐与多样性，并用SCM得分量化温暖与能力维度，结果显示大多数模型在对齐上高但多样性低，明显呈现刻板印象；

**⚠️ 局限性**

局限性包括依赖自动化工具（SAM3、CLIP、Qwen3-VL）可能继承偏见，未纳入残疾人人工评估，且缺少对不可视残疾和更丰富情境的覆盖。

---

## 359. Systematic Evaluation of Learning Rate Scheduling Strategies Across Heterogeneous Architectures

**arXiv ID:** 2607.08511 | [PDF](https://arxiv.org/pdf/2607.08511v1)

**作者:** Hafsa Mateen `[一作]` (University of Würzburg), Dmitry Ignatov `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统地评估了25种学习率调度策略在30种不同网络架构（CNN与Transformer）上的表现，并通过自动化源代码注入构建了3938个模型变体，在CIFAR‑10上进行快速5轮训练筛选；

**💡 创新点**

首次在统一框架下提供跨架构的调度性能景观，并通过自动化注入方法大规模生成可复现的模型变体，揭示了调度与网络架构的显著交互；

**🔧 技术方法**

采用自动化源代码注入、PyTorch调度器族、NNEval统一训练/评估管线、统计分析与可视化工具；

**📊 数据集**

使用CIFAR‑10图像分类数据集，并基于LEMUR网络库中的30个代表性架构；

**📈 对比分析**

以top‑1准确率为评估指标，对所有模型进行比较，平均准确率52.5%，最高86.45%，6%模型超过80%，显示不同调度对不同架构的显著影响；

**⚠️ 局限性**

仅在5个epoch的筛选阶段评估，未覆盖长周期训练或更大数据集；未考虑Adam等优化器；调度策略范围有限；部分任务不匹配导致模型表现低下。

---

## 360. CommuniWave:A Machine Learning Model for Quantifying the Degree of Temporary Informal Behavior in Urban Communities

**arXiv ID:** 2607.08554 | [PDF](https://arxiv.org/pdf/2607.08554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 361. UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing

**arXiv ID:** 2607.08646 | [PDF](https://arxiv.org/pdf/2607.08646v1)

**作者:** Xinlong Zhao `[一作]` (Peking University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了UltraX框架，利用函数调用方式对大规模预训练语料进行细粒度、可执行的文本精炼；

**💡 创新点**

创新点在于完整的编辑函数空间（删除、修改、插入）与程序监督生成流水线、以及针对大规模语料的滑窗预测、全局操作聚合和系统化后处理；

**🔧 技术方法**

采用了专家LLM进行端到端精炼、数据集自适应提示优化、行对齐映射与动态上下文替换、轻量级函数调用模型训练及程序执行机制；

**📊 数据集**

实验使用了公开的FineWeb、RedPajama‑v2、AICC、Ultra‑FineWeb和FineWeb‑ProX‑Doc五个Web语料集；

**📈 对比分析**

与基线ProX‑C相比，UltraX在所有语料下均获得最高平均下游性能，平均提升约2%且在大部分任务/语料组合中夺得第一，且在较少训练token下即可达到或超越基线；

**⚠️ 局限性**

局限包括未在更大模型或更长训练规模上验证、未与未公开的RefineX直接对比、仅针对英文语料、模型压缩与推理加速仍待提升，以及特定噪声类型的专门精炼器尚未开发。

---

## 362. New sharp inequalities involving non-relative, relative and cross informational functionals with some remarkable minimizers of generalized Gaussian and Beta types

**arXiv ID:** 2607.08599 | [PDF](https://arxiv.org/pdf/2607.08599v1)

**作者:** Razvan Gabriel Iagar `[一作]` (Universidad Rey Juan Carlos), David Puertas-Centeno `[通讯]` (Universidad Rey Juan Carlos)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

推导了若干新的精确信息不等式，涵盖Rényi熵、交叉熵、Fisher信息与矩类量等

**💡 创新点**

首次将这些信息量统一在相对框架下并给出最优常数与优化分布

**🔧 技术方法**

采用Rényi函数、信息量不等式、变换法与特殊函数（拉普拉斯、Beta、拉普拉斯-高斯等）

**📊 数据集**

无数据集，理论证明为主

**📈 对比分析**

无实验对比，理论上给出最优性与取最优分布

**⚠️ 局限性**

缺乏显式解的最优分布，部分优化问题只能通过隐式微分方程得到

---

## 363. Federated Deep Learning for Privacy-Preserving Cardiovascular Disease Risk Prediction

**arXiv ID:** 2607.08595 | [PDF](https://arxiv.org/pdf/2607.08595v1)

**作者:** Hyunho Mo `[一作]` (Erasmus MC University Medical Center), Esther E. Bron `[通讯]` (Erasmus MC University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在两个不同特征的全国性队列（Lifelines和Rotterdam Study）中，利用联邦学习训练深度生存神经网络（DeepSurv），实现了无需共享个人数据的心血管疾病风险预测；

**💡 创新点**

首次将深度生存模型与联邦学习结合，验证了在异质多中心数据上协同训练能够提升预测性能，并提供了可扩展的隐私保护框架；

**🔧 技术方法**

采用DeepSurv模型、FedAvg聚合算法、Vantage6平台实现联邦架构，使用HL7 FHIR标准化管道进行数据统一，模型训练采用Adam优化器；

**📊 数据集**

使用Lifelines队列（148,230人，自报心血管事件）和Rotterdam Study队列（10,155人，数字链接临床事件）作为数据源；

**📈 对比分析**

与各队列单独训练的本地模型对比；在Rotterdam Study中C统计从0.728提升至0.739，在Lifelines中从0.783提升至0.787；

**⚠️ 局限性**

仅涉及两机构，样本量差异大导致FedAvg权重偏移；Lifelines自报事件不够精确；未考虑竞争风险；未使用更适合非IID数据的聚合策略，且未评估更多中心的可推广性。

---

## 364. AI-guided stimuli discovery and generation to optimize facial emotion perception studies in autism

**arXiv ID:** 2607.08533 | [PDF](https://arxiv.org/pdf/2607.08533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 365. Rumour Spreading In Community Based Networks

**arXiv ID:** 2607.08546 | [PDF](https://arxiv.org/pdf/2607.08546v1)

**作者:** Zhaoxi Cui `[一作]` (University of Stirling), Anthony O'Hare `[通讯]` (University of Stirling)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一个以社区为单位、通过两个参数（社区内连通概率 w_g 与社区间连通概率 b_g）描述的网络模型，并在该网络以及 Erdős–Rényi 随机网络和 Watts–Strogatz 小世界网络上，用 Monte Carlo 方式模拟了基于 Maki‑Thompson 传闻扩散过程，测量了最大传播者比例、峰值时间、最终沉默者比例和平衡时间等指标。

**💡 创新点**

① 通过 w_g 与 b_g 两个连通概率，将社区网络与随机网络、小世界网络的参数映射关系明确化；② 在同一结构空间下比较三种网络的传播动力学，揭示了高聚类（高 w_g、低 b_g）社区网络在传播速度与峰值强度上的抑制效应；③ 在不同传播‑沉默率比 β 变化时检验结构鲁棒性，发现社区网络表现出阈值行为。

**🔧 技术方法**

Monte Carlo 仿真、Maki‑Thompson 传闻模型、聚类系数、模块度、局部效率等网络度量计算，以及对不同网络拓扑下四个传播指标的统计比较。

**📊 数据集**

使用合成网络数据：共 6000 个节点、100 个社区、固定 α=0.001、θ=0.001，并在不同 w_g、b_g、β 取值下生成网络并运行仿真。

**📈 对比分析**

对随机网络、社区网络和小世界网络在同一参数空间下的四个指标进行并行绘图与数值比较。结果显示：社区网络在高 w_g、低 b_g 区域传播更慢、峰值更低，但最终沉默者比例与其他网络相近；小世界网络介于两者之间；随着 β 的增大，各网络差异趋于消失。

**⚠️ 局限性**

仅在理想化的合成网络上验证，缺乏真实世界网络的数据与实验；参数设置（α、θ、网络规模、社区数）较为固定，未探讨不同规模或更复杂交互的影响；仿真耗时较长，且对极端参数区间的泛化性有限。

---

## 366. Structural Bottlenecks on Frequency Representation in End-to-End Audio Models

**arXiv ID:** 2607.08545 | [PDF](https://arxiv.org/pdf/2607.08545v1)

**作者:** Nicole Cosme-Clifford `[一作]` `[通讯]` (Yale University), Nicole Cosme-Clifford (Yale University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对端到端卷积音频编码器进行理论与实验分析，揭示其对频率局部原语的可访问性存在注入性和可分离性两大瓶颈，并提出Gabor Latent Refactorization（GLRF）后置方法恢复可分离性。

**💡 创新点**

创新点在于提出可预见的注入性与可分离性判定框架，精确预测压缩率和频率分辨率极限，并设计轻量级GLRF实现对已训练编码器的频域解耦与可控性提升。

**🔧 技术方法**

使用卷积下采样理论、信号折叠分析、感受野解析和高斯‑Gabor滤波器组进行理论推导，并通过线性岭回归实现GLRF映射；实验中还利用FFT峰值检测评估代理。

**📊 数据集**

主要使用人工合成的窄带多频信号进行注入性评估，结合真实音频数据集（如NSynth、EnCodec、DAC、Stable Audio的预训练模型）检验重构与可控性。

**📈 对比分析**

与原始编码器相比，GLRF将滤波器带宽从10–35倍理论极限压缩至1.5–3倍，重构误差保持在0.37–0.62 log‑mel，编码向量相似度≥0.97，频率替换可控性在DAC上实现100%成功率。

**⚠️ 局限性**

局限在于仅针对窄带、可调谐的信号，无法完全修复注入性损失；GLRF对高密度或带宽信号的适用性未知；实验多依赖合成信号，真实复杂音频的性能未充分验证。

---

## 367. Whareformer: Learning to Track What is Where in Long Egocentric Videos

**arXiv ID:** 2607.08537 | [PDF](https://arxiv.org/pdf/2607.08537v1)

**作者:** Jacob Chalk `[一作]` (University of Bristol), Diane Larlus `[通讯]` (NAVER LABS Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Whareformer，一种Transformer‑基于的长时3D目标跟踪模型，可在视线之外维护对象的空间位置信息与身份；

**💡 创新点**

创新点在于（1）联合推理“what”和“where”特征；（2）使用可学习的New Track token主动决定何时新建轨迹；（3）采用DenStream聚类和一秒窗口的位置信息，实现高效、可持续的轨迹表征；

**🔧 技术方法**

核心技术包括：DINOv2视觉特征提取、PCA降维、DenStream在线聚类、相对距离嵌入、Transformer编码器、DAgger训练策略；

**📊 数据集**

使用EPIC‑KITCHENS（56训练/54测试）作为主要训练/测试集，并在IT3DEgo与HD‑EPIC跨域验证；

**📈 对比分析**

与传统2D/3D跟踪基线（ByteTrack、IT3DEgo、LMK、LMK‑Inf）对比，Whareformer在mPCL和IDF1上分别提升约+19.2%和+14.0%，在长时序上表现尤为突出；

**⚠️ 局限性**

局限性包括对极端遮挡/快速视角变化的鲁棒性仍有限，且新Track token阈值调优和DenStream参数设置对性能影响显著，需进一步自动化。

---

## 368. It Takes Few to TANGO: A Quantized Distributed Model for Binaural Speech Enhancement

**arXiv ID:** 2607.08645 | [PDF](https://arxiv.org/pdf/2607.08645v1)

**作者:** Zahra Benslimane `[一作]` (Université Paris-Saclay), Romain Serizel `[通讯]` (Université de Lorraine)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种低计算量的量化版TANGO（MN‑TANGO）实现双耳语音增强，利用INT8量化、ERB压缩与分组循环层，显著减小模型体积与MAC量，同时保持与FP32基准相近的增强效果。

**💡 创新点**

创新点在于①证明空间滤波阶段对量化误差具有鲁棒性，可在量化后恢复大部分性能；②将原双阶段TANGO简化为单阶段MN‑TANGO，减少互节点通信；③结合量化感知训练、ERB特征压缩与分组LSTM，构建极致轻量化模型。

**🔧 技术方法**

使用的技术包括：后训练量化（DPTQ）与量化感知训练（QAT）实现INT8权重/激活量化；知识蒸馏以进一步缓解量化误差；ERB频带压缩降低循环输入维度；分组LSTM降低循环计算；差分可微SDW‑MWF训练；GEVD基空间滤波。

**📊 数据集**

数据集方面：训练使用LibriSpeech与合成噪声的双耳混合；评估采用BinauRec公开数据集（1200个室内RIR混合），SNR分别为-5、0、5 dB，目标音频正面，噪声位于右侧45°或90°。

**📈 对比分析**

比较方法：与完整FP32 TANGO、DPTQ、QAT（无KD）及不同MN‑TANGO变体（倒置、全量化、KD）进行对比；使用SI‑SDR、SI‑SIR、SI‑SAR、STOI、PESQ等指标。结果显示：量化感知训练基本恢复FP32性能；MN‑TANGO在保持相似SI‑SIR的同时将参数量和MAC降低约一半；进一步加入分组与ERB后，模型可压缩至0.081M参数、0.177 MB内存，MAC仅4.65 MMAC/s，性能仅低于5 dB SI‑SDR的微幅差距。

**⚠️ 局限性**

局限性包括：量化仅应用于网络层，空间滤波仍使用FP32，限制了全流程整数化；知识蒸馏收益有限，需更有效的蒸馏策略；仅在典型听力辅助场景下评估，未验证在更极端噪声或低SNR环境下的鲁棒性；分组LSTM的最优组数仍需经验选取，且在不同硬件上可能表现不一致。

---

## 369. A New Human-Likeness and Comfort Index for Robot Movements Along Prescribed Paths

**arXiv ID:** 2607.08620 | [PDF](https://arxiv.org/pdf/2607.08620v1)

**作者:** Rosanna Coccaro `[一作]` (University of Salerno), Pasquale Chiacchio `[通讯]` (University of Salerno)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一种基于sigma-lognormal模型的运动时间律相似性指数（HL），用于先验评估机器人轨迹的舒适度和人类相似性；

**💡 创新点**

将HL指数与最小笔画数及SNR结合，构造先验舒适评估指标，并利用时间最优规划（TOTP）生成与人类相近的时间律，实现对任何几何轨迹的舒适化；

**🔧 技术方法**

采用sigma-lognormal运动建模、lognormal参数提取、SNR计算、时间最优轨迹规划（TOTP）、统计分析（Logistic回归、t检验）等技术；

**📊 数据集**

使用68名受试者记录的手写轨迹数据，以及基于Uniform和STOTPAC时程的机器人轨迹，共三轮对比实验数据；

**📈 对比分析**

通过主观舒适度对比实验，HL指数与用户偏好高度相关，优于传统MAJ指标，且在不同曲率模式下均保持一致的预测性能；

**⚠️ 局限性**

仅关注时间律，忽略轨迹几何和个体差异；HL为充分条件而非必要条件，对儿童或身体障碍受试者验证不足；

---

## 370. Switch-Reasoner: Learn When to Think in Multitask Mixtures via Reinforcement Learning

**arXiv ID:** 2607.08572 | [PDF](https://arxiv.org/pdf/2607.08572v1)

**作者:** Yiyang Fang `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于GRPO的Switch-Reasoner框架，能够在多模态大语言模型推理过程中自适应选择直接回答或显式推理。

**💡 创新点**

创新点在于将“思考”视为可调用工具，并结合全局模式平衡与样本级对比监督的双层调控机制，有效避免GRPO训练中的模式崩塌。

**🔧 技术方法**

采用Group Relative Policy Optimization（GRPO）、工具调用技术、全局模式平衡控制、样本级路由监督以及多项奖励设计。

**📊 数据集**

在覆盖数学、视觉推理、图表/文档、定位及通用理解等 11 任务的异质多模态基准数据集上进行实验。

**📈 对比分析**

与 Vanilla、GRPO-Thinking、GRPO-Direct 以及无样本级校正变体对比；在 4B/8B 模型上，Switch-Reasoner 在保持甚至提升整体准确率的同时，将思考率降低到约 51%（4B）和 38%（8B），取得最优性能‑成本平衡。

**⚠️ 局限性**

局限性包括仅实现二元思考/直接切换；对不同任务的阈值需手工调优；尚未在多工具或多路径推理场景中扩展。

---

## 371. CAAD: Causality-Aware Multivariate Time Series Anomaly Detection via Multi-Scale Alignment and Structural Causal Consistency

**arXiv ID:** 2607.08555 | [PDF](https://arxiv.org/pdf/2607.08555v1)

**作者:** Xin Wang `[一作]` (Stony Brook University), Tengfei Ma `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了CAAD框架，用多尺度对齐和因果一致性验证来检测多变量时间序列异常。

**💡 创新点**

将异常检测视为持续的Granger因果一致性验证，使用梯度基Granger矩阵和多尺度对齐实现双视角异常评分。

**🔧 技术方法**

采用Transformer编码器、多尺度滑动窗口、自监督对齐、梯度基Granger因果检验和非对称门控融合等技术。

**📊 数据集**

在SWaT、PSM、CATSv2、SMD等四个工业与服务器监控数据集上进行实验。

**📈 对比分析**

与PCA、KNN、iTransformer、PatchTST、TranAD、VAE、TimesNet等基线对比，SWaT上F1≈0.95、PRC≈0.974、ROC≈0.994，整体表现显著优于大多数基线。

**⚠️ 局限性**

对弱因果耦合或高噪声场景敏感，需额外超参数调优；当前模型仅用于异常检测，根因定位功能仍需进一步扩展。

---

## 372. VocaDet: Sample-Driven Open-Vocabulary Object Detection and Segmentation via Visual Tokenization and Vector Database Retrieval

**arXiv ID:** 2607.08541 | [PDF](https://arxiv.org/pdf/2607.08541v1)

**作者:** ZhiXin Sun `[一作]` `[通讯]` (PowerChina Zhongnan Engineering Corporation Limited), ZhiXin Sun (PowerChina Zhongnan Engineering Corporation Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了VocaDet，一个基于样本驱动的开放词汇目标检测与分割框架。

**💡 创新点**

创新点在于将连续视觉特征离散化为可扩展的视觉词汇，结合向量数据库检索和位置去偏特征，实现无需模型再训练即可从正负样本学习识别。

**🔧 技术方法**

采用DINOv3特征提取器、层次化凝聚聚类、位置去偏表征、背景过滤层和Milvus向量数据库等技术。

**📊 数据集**

使用UA-DETRAC数据集进行评估。

**📈 对比分析**

与传统固定类别检测器对比，VocaDet在无需训练的情况下实现竞争性的检测性能，并且可通过增大样本库持续提升精度。

**⚠️ 局限性**

局限在于相邻同类目标易被错误合并以及冷启动时样本不足导致特征边界模糊。

---

## 373. Improving Ad-hoc Search Effectiveness for Conversational Information Retrieval via Model Merging

**arXiv ID:** 2607.08540 | [PDF](https://arxiv.org/pdf/2607.08540v1)

**作者:** Ahmed Rayane Kebir `[一作]` (University of Toulouse), Lynda Tamine `[通讯]` (University of Toulouse)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练-free的模型合并策略，将基于MS MARCO的ad-hoc检索模型ANCE与在对话检索数据集上微调的QRACDR模型融合，得到既能处理单轮检索又能处理多轮对话检索的统一模型。

**💡 创新点**

创新点在于：1）首次将模型合并用于会话检索场景；2）通过线性(Model Soup)和球面插值(Slerp)两种参数级别的合并方式，在不再额外训练的前提下有效恢复对话模型的ad-hoc检索能力并提升泛化；3）展示合并模型在多任务、多数据集下可与多任务学习和早停策略竞争甚至超越。

**🔧 技术方法**

主要技术包括：模型参数级别的线性插值（Model Soup）与球面插值（Slerp），MergeKit工具实现参数融合；使用预训练的ANCE与对话微调的QRACDR模型进行合并；在融合后对不同数据集进行评估。

**📊 数据集**

使用的数据集包括：MS MARCO（ad-hoc检索基准），QReCC和TopiOCQA（对话检索训练/评估），CAsT 2019/2020（对话检索测试，包含重写查询和会话两种输入），NQ、HotpotQA、CAsT的重写查询（OOD检索评估）。

**📈 对比分析**

与基线（ANCE）、对话微调模型（QRACDR-Q/T）以及多任务学习、早停等方法进行对比。实验结果表明：合并模型在MS MARCO上的NDCG@3恢复到-3%以内的forgetting，同时在对话检索任务上保持或略优于微调模型；在CAsT、NQ等OOD数据上，合并模型实现最高15% NDCG@3提升（在零样本条件下），并在会话检索中提升10%以上。合并方法不需要额外梯度更新，显著降低计算成本。

**⚠️ 局限性**

局限性：1）合并系数λ需要在各数据集上手工调优，缺乏自动化选择策略；2）目前仅验证了ANCE+QRACDR两种模型组合，泛化到其他检索架构的效果尚未证实；3）对长篇多轮会话中的核心ference、歧义消解等语义细粒度问题的提升有限，主要依赖于底层模型的编码能力。

---

## 374. Stop Guessing When to Stop Testing: Efficient Model Evaluation with Just Enough Data

**arXiv ID:** 2607.08522 | [PDF](https://arxiv.org/pdf/2607.08522v1)

**作者:** Ofir Arviv `[一作]` (IBM Research), Leshem Choshen `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了一种基于序贯检验的自适应模型评估框架，允许在满足统计需求时提前停止评估，并在Open VLM Leaderboard上演示了显著的计算效率提升。

**💡 创新点**

创新点在于将Pocock支出函数的组序贯检验与多种停止准则（等价边际、精度、阈值、无效性、递减收益）结合，首次在多数据集视觉语言模型评估中实现统计可靠与计算效率的平衡。

**🔧 技术方法**

主要技术包括组序贯检验（Pocock）、自适应停止规则设计、R包gsDesign的Python集成以及LLM-as-Judge分数提取。

**📊 数据集**

使用的数据集为Open VLM Leaderboard中的206个模型在31个多模态基准上的评估记录（约14,400例），通过LLM-as-Judge提取分数。

**📈 对比分析**

实验中通过单模型精度估计、成对比较和排名任务比较，显示在保持95%置信水平下相较固定样本评估可节省约80%计算成本，或在阈值准则下降低44%成本，并在大多数对比中显著提高区分率。

**⚠️ 局限性**

限制包括对独立同分布假设的依赖、序贯检验带来的额外计算开销、结果受随机性影响导致可复现性受限，以及需要用户自行设定阈值或等价边际，需一定统计或领域专业知识。

---

## 375. On the Convergence of Belief Propagation for Multipath Data Association in Target Tracking

**arXiv ID:** 2607.08521 | [PDF](https://arxiv.org/pdf/2607.08521v1)

**作者:** Kuilong Yang `[一作]` (Northwestern Polytechnical University), Jing Fu `[通讯]` (RMIT University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aaccfe5c-6b26-4208-b23c-35331481e142` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出了在多路径数据关联（MPDA）问题中，使用循环贝叶斯传播（BP）时的收敛性证明，并在模拟的多路径跟踪场景中验证了BP的精度与收敛行为。

**💡 创新点**

创新点在于：①对MPDA的BP更新构造了一个正向不变的紧致子集，并证明在此子集上收敛映射是严格收缩；②使用对数距离度量与Banach不动点定理，给出了完整的收敛证明；③将证明推广到任意多路径和多目标，超越以往仅针对两路或两目标的研究。

**🔧 技术方法**

主要技术包括：因子图建模、BP消息更新、收缩映射与Banach固定点定理、对数距离度量、拉姆达收敛判据；此外还使用了多目标跟踪的EKF预测与测量似然构造固定证据消息。

**📊 数据集**

实验数据为仿真的多路径目标跟踪场景（OTHR），包括从5到30个目标、1到4条传播路径、不同检测概率与杂波密度等参数设置；未使用公开真实数据集。

**📈 对比分析**

与基于多检测多假设跟踪（MD-MHT）的单扫与双扫变体对比。结果表明：BP在大多数配置下平均OSPA距离更低、计算时间略高于单扫MD-MHT、但远低于双扫MD-MHT；BP的迭代次数在30左右，收敛稳定，展示了更优的精度‑效率折衷。

**⚠️ 局限性**

局限性：①收敛证明仅针对固定证据消息的内部BP迭代，未涵盖外部VB循环的整体收敛；②不适用于扩展目标跟踪（EOT），因为MPDA与EOT的关联事件空间本质不同；③实验仅为仿真，缺乏对真实多路径跟踪数据的验证。

---

## 376. Deep Learning for Joint Narrowband Interference Cancellation and Soft Demodulation in OFDM Systems

**arXiv ID:** 2607.08717 | [PDF](https://arxiv.org/pdf/2607.08717v1)

**作者:** Emmanouil Kavvousanos `[一作]` (University of Patras), Vassilis Paliouras `[通讯]` (University of Patras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套联合深度学习框架（NBI‑CNet + LLR‑CNet），实现 OFDM 系统中窄带干扰（NBI）的并行参数估计、干扰消除和后续软判决校准。

**💡 创新点**

创新点包括：① 在不预先知道干扰源数量的前提下，利用物理建模的卷积网络一次性预测干扰幅度、频偏和相位；② 通过可微分的干扰重建层实现精确 Dirichlet 核合成，消除传统 CS 方法的迭代延迟；③ 设计轻量级 LLR‑CNet 作为结构化白化器，将残留非高斯噪声映射为校准的软信息，显著抑制误码阈值；④ 通过自适应量化与动态稀疏路由实现低 FLOPs 与可扩展性。

**🔧 技术方法**

技术手段包括物理信息嵌入的 1D 卷积网络、循环填充保证频域连续性、可微分干扰重建层、基于 BCE 的 LLR 训练、动态稀疏推理与量化部署（FP16 TensorRT）。

**📊 数据集**

使用的实验数据为基于 Sionna 生成的仿真 OFDM 信号，采用 16‑QAM、N=256、SNR 7–15 dB、SIR -30~10 dB、干扰源数 Q 随机 0–8，且每个干扰源频偏在 -0.5~0.5 子载波间隔之间。

**📈 对比分析**

与传统压缩感知基线（OMP‑IDS、EOMP‑IDS）及高斯 Max‑Log 判决器进行对比。实验表明：① 在 SIR=-10 dB 时，联合网络的 BLER 与最优迭代基线相差 0.2–0.5 dB；② 在 SIR=10 dB 时，网络在 Q=8/12 的高密度干扰场景下实现 2–3 dB 的编解码增益；③ 计算复杂度方面，NBI‑CNet 在 Q>26 时已低于 EOMP‑IDS，整体 FLOPs 下降 60% 以上；④ LLR‑CNet 消除传统软判决导致的误码阈值，保持在 10^-4 BLER 以内。

**⚠️ 局限性**

局限性包括：① 训练完全基于仿真数据，缺乏真实干扰环境验证；② 对极端干扰密度（Q>64）和极端 SIR（>20 dB）尚未充分评估；③ 需要在不同硬件平台上进一步验证量化与实时性；④ 目前模型仍假设干扰为独立单调 Dirichlet 形状，若干扰源出现非线性或时变相位耦合，性能可能受限。

---

## 377. MPFlow: Learning Budgeted Max-Flow Optimization on the Lightning Network with Deep Graph Reinforcement Learning

**arXiv ID:** 2607.08703 | [PDF](https://arxiv.org/pdf/2607.08703v1)

**作者:** Harrison Rush `[一作]` (Amboss Technologies), Emanuele Rossi `[通讯]` (Amboss Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种基于强化学习的预算约束图优化方法，用来为比特币闪电网络（LN）节点选择最优的链路开放策略，以最大化路由容量。

**💡 创新点**

创新点在于将路由容量提升定义为s–t最大流增量的奖励，使用带最大聚合的消息传递网络（MPNN）配合PPO，并通过去除顶级中心节点的训练课程（hub‑exclusion curriculum）让模型学会容量感知而非单纯追随中心性。

**🔧 技术方法**

技术主要包括：图神经网络（MPNN）与最大聚合、Proximal Policy Optimization（PPO）策略优化、可行动作掩码、以及基于最大流的奖励设计。

**📊 数据集**

使用真实闪电网络的三张网络快照（D1、D2、D3），每张包含数千节点及其通道容量信息，构成训练与测试的数据集。

**📈 对比分析**

与随机、度中心性、介数中心性等传统启发式基线以及GCN、GAT等学习基线进行对比，MPFlow在5k节点子图上平均提升绝对流量0.168 BTC、相对提升8.6%，赢得62%对战介数中心性，并在跨快照、不同图规模及去中心化测试中保持优势。

**⚠️ 局限性**

局限性包括：使用随机采样的通道余额而非真实余额；仅考虑固定的5个开放通道预算；最大流目标仅是结构性上界，未直接评估实际支付成功率与经济收益；未包含费用与交易成本等额外奖励。

---

## 378. Multi-Resolution Feature Stem for Diabetic Retinopathy lesion segmentation

**arXiv ID:** 2607.08679 | [PDF](https://arxiv.org/pdf/2607.08679v1)

**作者:** Indranil Dutta `[一作]` (San Jose State University), Taehee Jeong `[通讯]` (San Jose State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了糖尿病视网膜病变病变分割中多分辨率输入对不同病变类型的冲突效应，并提出了多分辨率特征干预（Multi‑Resolution Feature Stem）来同时捕捉细粒度与全局上下文；

**💡 创新点**

创新点在于揭示更高分辨率对小型病变有效但对大型出血不利的悖论，并通过在输入层构建多尺度金字塔、共享权重并行提取、1×1卷积融合的方式实现规模无关特征学习；

**🔧 技术方法**

使用了UNet++骨干网络、共享卷积块、1×1融合层、Focal Loss与Dice Loss的加权组合，以及多分辨率输入金字塔；

**📊 数据集**

采用DDR（Diabetic Retinopathy Dataset）分割子集，包含四类病变（EX、HE、MA、SE）的像素级标注；

**📈 对比分析**

通过与DeepLab‑v3+、U‑Net、U‑Net++等基线在512×512与1024×1024分辨率下进行mAP、mIoU、AP等指标比较，结果显示多分辨率模型在测试集上mAP提升至0.3985、mIoU提升至0.2853，较基线显著提高；

**⚠️ 局限性**

局限在对大血管性出血（HE）表现不佳，且相较单尺度模型推理时间增加约1.27倍。

---

## 379. Resample or Reroute? Budget-Aware Test-Time Model Selection for Large Language Models

**arXiv ID:** 2607.08665 | [PDF](https://arxiv.org/pdf/2607.08665v1)

**作者:** Teng-Ruei Chen `[一作]` `[通讯]` (National Yang Ming Chiao Tung University), Teng-Ruei Chen (National Yang Ming Chiao Tung University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于预算的测试时模型选择方法，解决在有限推理成本和不完美验证器下，是否对已选模型进行多次重采样还是切换到其他模型的问题。

**💡 创新点**

创新点在于将重采样与重路由视为同一预算内竞争性操作，设计了在线的“Resample‑or‑Reroute（RoR）”分配策略，并用恢复性不对称理论解释其行为。

**🔧 技术方法**

技术上采用了贝叶斯后验估计模型成功率、基于边际正确率/单位成本的贪婪分配规则（包括UCB探索变体），并通过多抽样正确率张量进行离线重放实验。

**📊 数据集**

使用了四个公开基准（GSM8K、MATH‑500、GPQA‑Diamond、HumanEval+）以及一个由11个开源LLM组成的模型池，生成了每个(query, model)单元30次种子对齐抽样。

**📈 对比分析**

与单路由、一次提交路由、预算感知最佳‑of‑K、级联、随机分配以及不可部署的oracle分配等基线对比，RoR在大多数基准上实现了更优的成本‑质量 Pareto 前沿，在多样化模型池上尤其显著；在低预算和高质量验证器场景下性能提升最为明显。

**⚠️ 局限性**

局限性包括：对验证器质量高度依赖；成本模型以参数量为代理，需在实际部署中重新校准；仅评估单线程顺序推理，未考虑批量或低延迟场景；在多选或弱验证器任务上优势可能减弱。

---

## 380. WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search

**arXiv ID:** 2607.08662 | [PDF](https://arxiv.org/pdf/2607.08662v1)

**作者:** Xiaoshuai Song `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归委托框架WebSwarm，用代理节点动态拆解任务并协同搜索；

**💡 创新点**

创新点在于将本地目标与搜索模式耦合、引入基于网页结构的递归扩展与同级节点经验复用；

**🔧 技术方法**

采用多代理递归委托、四种搜索模式(atom、deep、wide、entity_collect)、轻量化网页结构探测与经验迁移技术；

**📊 数据集**

在四个复杂Web检索基准上评测：BrowseComp-Plus、WideSearch、DeepWideSearch 与 GISA；

**📈 对比分析**

与单代理ReAct及多代理基线相比，WebSwarm在深度、宽度及混合搜索任务中均实现显著提升（例如在BrowseComp-Plus提升17.5点准确率、在WideSearch-EN提升10.9点Row‑F1、在DeepWideSearch-EN提升9.6点Row‑F1），并在困难样本上表现尤为突出；

**⚠️ 局限性**

局限性包括：对大型LLM与工具调用资源需求高、对Web结构假设的依赖可能受限于域外内容、以及递归深度与节点数控制仍需进一步优化。

---

## 381. Multi-Modal, Multi-Environment Machine Teaching for Robust Reward Learning

**arXiv ID:** 2607.08647 | [PDF](https://arxiv.org/pdf/2607.08647v1)

**作者:** Ali Larian `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种多环境、多模态反馈的机器教学框架 HSCOT，用来学习对环境动态鲁棒的奖励函数；

**💡 创新点**

创新点包括：①证明单一 MDP 教学无法完全识别奖励，需跨环境约束；②比较不同反馈模态在无限与有限预算下的约束信息量；③将环境选择与反馈查询分层建模为集合覆盖问题，并使用贪心近似求解；

**🔧 技术方法**

主要技术：逆强化学习（max-margin IRL）、行为等价类（BEC）与一般化行为等价类（gBEC）、集合覆盖与子模函数贪心优化、奖励空间体积量化；

**📊 数据集**

使用两个人工合成数据集：6×6 GridWorld 与 LavaMiniGrid，各生成 50 个 MDP（10 个留作 hold‑out），特征为手工设计或随机采样；

**📈 对比分析**

与 Uniform Teaching（在同一预算下随机采样环境和反馈）比较，实验显示 HSCOT 在两种数据集上均实现更低的 hold‑out regret、更高的约束覆盖率（接近 100%），并且激活的环境数量更少；

**⚠️ 局限性**

局限性：仅在离散 deterministic MDP 上验证；假设教师知道真奖励且反馈无噪声；对连续或噪声人类反馈的扩展仍待研究。

---

## 382. Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction

**arXiv ID:** 2607.08769 | [PDF](https://arxiv.org/pdf/2607.08769v1)

**作者:** Weijian Chen `[一作]` (Insta360 Research), Lu Qi `[通讯]` (Insta360 Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向大规模全景户外场景的两阶段粗细细化3D高斯光栅化框架 PanoLOG，解决全景图可见性无限制导致的分块优化失效问题。

**💡 创新点**

创新点在于 Geometry and Gradient-based Partitioning Strategy（G^2PS），通过基于相机视角基线的深度不确定性构建自适应有界体积，并用渲染梯度实现摄像头–块的精确分配；同时引入显式天空球和全景单目深度监督来稳定无纹理和无三角化区域。

**🔧 技术方法**

主要技术包括全景 3D 高斯光栅化（ERP 直接渲染）、基于相机几何的 AABB 生成与裁剪、梯度重要性分配、显式天空球模型、DAP 深度估计以及两阶段全局粗训练+块级细化的并行训练管线。

**📊 数据集**

使用自行构建的 Pano360 数据集（5,637 张 3840×1920 预留姿态与稀疏点云的 4 个全景户外子集）以及 Ricoh360、360Roam 公共全景基准进行评测。

**📈 对比分析**

与 H3DGS、CityGaussian、DOGS、Momentum-GS 等大型 3DGS 方法及 OmniGS、ODGS、SpaGS 等全景专用方法相比，PanoLOG 在 PSNR/SSIM/LPIPS 上普遍提升 0.3–1.2 dB、0.01–0.02 及降低 0.02–0.05，且模型体积约为传统方法的 1/3–1/7，验证了分块与全景特化设计的有效性。

**⚠️ 局限性**

局限性包括对深度监督的依赖（极端低纹理或极端极地区域仍易受误估影响）、梯度阈值（τ_grad）需经验调参、以及在极大场景下仍需多块并行计算，导致训练时间和 GPU 内存需求较高。

---

## 383. UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks

**arXiv ID:** 2607.08768 | [PDF](https://arxiv.org/pdf/2607.08768v1)

**作者:** Zhekai Chen `[一作]` (HKU MMLab), Xihui Liu `[通讯]` (HKU MMLab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了 UniClawBench，包含 400 个双语真实世界任务，用于在真实 Docker 环境中评估主动代理的执行能力。

**💡 创新点**

创新点在于将任务按技能使用、探索、长上下文、多模态理解和跨平台协调五大能力拆分，并引入三角色闭环评估（执行者、隐藏评估者、用户模拟器）以防止评估信息泄漏。

**🔧 技术方法**

技术手段包括 Docker 容器化实时运行、实时浏览器/终端/GUI 交互、隐藏检查点评估器与用户模拟器、以及 OpenClaw、Nanobot、EDICT 等框架配合 GPT‑5.4/Codex 等 LLM。

**📊 数据集**

使用 400 个手工设计的双语（英中）任务集，任务覆盖多模态输入输出、长上下文、技能调用、探索性查询和跨平台协作。

**📈 对比分析**

通过在 OpenClaw 框架下评测 10 个 SOTA 模型以及在 OpenClaw、Nanobot、EDICT 三个框架下对代表模型的跨框架对比，发现闭源模型最高通过率仍低于 50%，框架差异显著，OpenClaw 通关率最高，EDICT 交互摩擦导致通过率下降，Nanobot 低耗但通关率最低。

**⚠️ 局限性**

局限性包括任务量有限、实时环境不稳定、评估依赖 LLM 可能产生偏差，以及在多模态与跨平台任务中仍存在较大挑战。

---

## 384. MulTTiPop: A Multitrack Transcription Dataset for Pop Music

**arXiv ID:** 2607.08756 | [PDF](https://arxiv.org/pdf/2607.08756v1)

**作者:** Nathan Pruyne `[一作]` (Carnegie Mellon University), Chris Donahue `[通讯]` (Carnegie Mellon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多轨MIDI与商业流行音乐片段对齐的数据集，用于评估多轨自动音乐转录模型。

**💡 创新点**

创新点在于通过TheoryTab元数据匹配、手工确定anchor beat、线性插值时间拉伸实现音频与MIDI的精确对齐，并提供真实世界流行音乐样本的评估基准。

**🔧 技术方法**

使用了Levenshtein距离进行元数据匹配、RNN beat tracking、chroma相似度与Pearson相关度相结合的自动anchor beat候选生成、手工标注校验、线性时间插值等技术。

**📊 数据集**

利用Lakh MIDI Dataset（LMD）、TheoryTab数据集以及对应的YouTube视频音频进行构建。

**📈 对比分析**

在该数据集上对MT3和YourMT3+两种顶级模型进行Onset F1评估，最佳模型达到约38% Onset F1，明显低于其在合成或经典音乐数据集上的表现，表明仍有显著改进空间。

**⚠️ 局限性**

限制包括：仅适用于评估而非训练；数据规模有限；对齐失败率高（≈50%）；覆盖面局限于西方流行音乐；存在版权与多语言音乐代表性不足的问题。

---

## 385. SLORR: Simple and Efficient In-Training Low-Rank Regularization

**arXiv ID:** 2607.08754 | [PDF](https://arxiv.org/pdf/2607.08754v1)

**作者:** David González-Martínez `[一作]` (Max Planck Institute for Intelligent Systems), Shiwei Liu `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无SVD、无状态、可直接在权重矩阵上施加的低秩正则化框架（SLORR），通过Polar Express近似极化因子实现前向后向计算；

**💡 创新点**

创新点在于：①不需要SVD或缓存量化量，完全无状态；②保持原有模型架构；③使用GPU友好的迭代近似，理论上保证近似误差可控；

**🔧 技术方法**

核心技术包括：极化因子（polar factor）近似（Polar Express）、Hoyer稀疏度/核范数正则化、CUDA友好矩阵运算与自定义PyTorch autograd实现；

**📊 数据集**

实验数据集：ImageNet‑1K（ResNet‑50/18、ViT‑B/16/L/16）以及 FineWeb‑Edu（LLM Llama 135M/560M）；

**📈 对比分析**

与 Q3R、LoRITa 等方法对比，SLORR 在模型压缩后保持相似或更好的准确率/困惑度，同时训练时间/显存开销仅<8%（视觉）或<1%（LLM），在多种压缩比例下位于 Pareto 前沿；

**⚠️ 局限性**

局限性包括：对超参数（正则化强度、迭代步数）敏感；高压缩率仍会显著影响性能；在模型微调或极端大模型上效果尚待验证；理论假设（奇异值范围）在实际中可能不完全满足。

---

## 386. AUTOPILOT VQA: Benchmarking Vision-Language Models for Incident-Centric Dashcam Understanding

**arXiv ID:** 2607.08745 | [PDF](https://arxiv.org/pdf/2607.08745v1)

**作者:** Siddharth Damodharan `[一作]` (University of Colorado Colorado Springs), Jugal Kalita `[通讯]` (University of Colorado Colorado Springs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AUTOPILOT‑VQA事故中心视觉问答基准，利用结构化问题评估自动驾驶视频的安全相关理解。

**💡 创新点**

创新点在于将事故场景细分为多维安全属性，构建面向事故理解的多模态VQA数据集，并通过Kaggle竞赛提供公开评估平台。

**🔧 技术方法**

采用视觉语言模型（大规模VLM）、多模态融合、时序建模以及结构化输出技术。

**📊 数据集**

使用AUTOPILOT‑VQA数据集，包含600多段行车摄像头视频及6,000+问答对。

**📈 对比分析**

在Kaggle竞赛中与多种模型对比，最高平均准确率约0.658，显示当前模型在感知上优于结构推理，但整体性能仍有限。

**⚠️ 局限性**

模型在因果推理、时序关系和安全决策推断方面表现不佳，尚未达到接近人类可靠性的水平。

---

## 387. Algorithmic Expert Aggregation

**arXiv ID:** 2607.08744 | [PDF](https://arxiv.org/pdf/2607.08744v1)

**作者:** Wei Tang `[一作]`, Hanrui Zhang `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并研究了“专家聚合”（expert aggregation）这一问题，目标是在已知多个已校准的贝叶斯专家预测和先验分布的情况下，构造一个新的、同样校准且信息量最大（在Blackwell意义下不可进一步改进）的输出专家，并在给定正则合适损失函数时求取其近似最优解。

**💡 创新点**

创新点主要包括：
1) 将预测聚合的研究从单次聚合转向专家级聚合，强调输出专家的校准性与Blackwell信息优势；
2) 以可观测线性信息（observable linear space）为核心，给出构造可实现专家的精确可判定性与构造性条件；
3) 揭示随机化输出专家可实现多项式时间算法（搜索与优化），而确定性输出专家则呈现NP-hard和无PTAS的难度，形成鲜明的算法二分；
4) 在优化问题中提出利用可观测线性信息与Blackwell支配的线性约束构建线性规划，并给出加性FPTAS。

**🔧 技术方法**

主要技术包括：
- 可观测线性系统与可观测非负锥的构造与分析；
- 极限射线分解、有限预测集合构造与极点计数证明小支持性；
- 采用Lexicographic（字典序）最优化转化为一系列线性规划；
- 利用Stern‑Brocot树实现有界分数的快速搜索；
- 通过对Blackwell支配的马尔可夫耦合表示，将Blackwell约束转化为线性约束；
- 对于优化问题，使用正则合适损失的Piecewise‑Linear Upper Approximation得到加性FPTAS。

**📊 数据集**

本文未使用任何实验数据集，而是以理论算法与复杂度分析为主。若需评估，可在合成数据或公开的多专家预测数据（如天气预报、医疗诊断等）上进行实验验证。

**📈 对比分析**

与传统的鲁棒预测聚合相比，随机化输出专家可在多项式时间内得到不可进一步改进且校准的输出专家；在优化层面可在任意正则合适损失下得到加性FPTAS。相对确定性输出专家，作者证明了搜索与优化均为NP-hard，并且在Brier损失下不存在多项式时间的乘法PTAS。实验评估（若存在）将验证在随机化方案下的预测性能提升与计算效率。

**⚠️ 局限性**

局限性包括：
1) 需要已知先验分布与输入专家的完整报告分布；
2) 对于确定性输出专家，算法不可行（除非P=NP），难以在实际应用中得到确定性聚合；
3) 结果主要为理论复杂度与算法构造，缺乏对实际数据的实验验证；
4) 仅考虑二值事件与贝叶斯校准的专家，扩展到多类或连续事件需进一步研究。

---

## 388. ContactMimic: Humanoid Object Interaction via Contact Control

**arXiv ID:** 2607.08742 | [PDF](https://arxiv.org/pdf/2607.08742v1)

**作者:** Xinyao Li `[一作]` (University of Illinois Urbana Champaign), Saurabh Gupta `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种在机器人全身跟踪中加入可控的接触标签，使机器人能够在不同任务中根据命令自主产生或抑制与物体的接触。

**💡 创新点**

创新点在于将接触标签作为运行时可调的输入，配合接触匹配奖励和距离奖励，使用轨迹增强方法打破关键点与接触的耦合，从而实现真正的接触控制。

**🔧 技术方法**

技术方法包括：①基于OmniRetarget将人类-物体交互数据映射到G1机器人；②设计基于标签匹配和距离的接触奖励；③构造接触标签翻转、对象移除、几何膨胀等数据增强策略；④用强化学习训练策略π_θ(a_t|p_t, k̅_t, c̅_t)。

**📊 数据集**

使用的主要数据集为HUMOTO人类-物体交互数据集，经过OmniRetarget后得到机器人轨迹与接触标签。

**📈 对比分析**

与当前最先进的关键点跟踪基线BeyondMimic比较，本文方法在关键点跟踪精度（MPJPE）相近的情况下，显著提高了接触次数、冲击量和物体位移等接触相关指标，并在5种真实动作中实现了高成功率。

**⚠️ 局限性**

限制包括：1）每种动作训练独立策略，未实现统一的全局接触跟踪；2）依赖高质量的HUMOTO数据，交互多样性受限；3）仅在一台Unitree G1机器人上验证，硬件泛化尚待进一步研究。

---

## 389. Learning Adaptive Solvers for Distributed Factor Graph Optimization on Matrix Lie Groups

**arXiv ID:** 2607.08735 | [PDF](https://arxiv.org/pdf/2607.08735v1)

**作者:** Jaeho Shin `[一作]` (University of Michigan), Yulun Tian `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DeepCORD，一种学习增强的分布式因子图优化框架，可在矩阵李群上处理多机器人/多会话的优化任务。

**💡 创新点**

创新点在于将 CORD 的迭代展开为可微分的计算图，学习本地反馈策略自适应调整质量、阻尼和步长，克服手工调参的脆弱性，并支持同步与异步通信。

**🔧 技术方法**

技术包括：深度折叠（deep unfolding）、图神经网络（GPS层+MLP）用于特征编码、无监督自回归训练（最小化展开迭代目标+正则化），以及稀疏共轭梯度与隐式微分实现数值高效求解。

**📊 数据集**

使用合成的多机器人位姿图（1,000个，5种拓扑）与真实轨迹（1,440个公共数据集）进行训练，评估数据集包括SE(3)位姿图优化（13个标准基准+S3E多机器人数据）和SL(4)投影子地图对齐（TUM RGB‑D 3个序列及自建大规模数据集）。

**📈 对比分析**

与现有分布式基线（CORD、ROBO、AMM‑PGO、DJ等）和集中式参考解（SE‑Sync、GTSAM）比较；DeepCORD 在同步/异步环境下分别在 11/10/13 组基准中获得最低成本，并在 21/26 组位姿图和所有投影对齐数据集上表现最优；其自适应参数能在不同拓扑、初始化和通信延迟下保持稳定收敛。

**⚠️ 局限性**

局限性包括：缺乏渐进收敛理论；训练仅展开 50 步，未显式保证更长时间的收敛；在某些实例下可能过于保守，导致过早进入细化阶段；缺乏全局反馈，导致局部梯度小而全局误差仍大。

---

## 390. Validity of LLMs as data annotators: AMALIA on authority

**arXiv ID:** 2607.08731 | [PDF](https://arxiv.org/pdf/2607.08731v1)

**作者:** Manuel Pita `[一作]` `[通讯]` (Universidade Lusofona), Manuel Pita (Universidade Lusofona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了葡萄牙国有LLM AMALIA在“authority”道德基础构念的注释能力，并检验其对理论构念的测量有效性。

**💡 创新点**

引入“恢复差距”指标，通过将构念分解为原子子句验证LLM是否按理论推断，而非仅靠表面关联。

**🔧 技术方法**

使用分解式（grain calibration）提示、逻辑表达式集成规则、Bootstrap置信区间以及人工阅读面板等技术。

**📊 数据集**

基于转译的Moral Foundations Reddit Corpus的748条欧洲葡萄牙语文本，包含已去重的正负标签。

**📈 对比分析**

与两大多语言LLM（Llama‑3.3‑70B、GPT‑OSS‑120B）在同一语料上对比，AMALIA的整体F1与人类标注相当，但其恢复差距显著开放（Δ≈0.35–0.46），表明缺乏理论有效性。

**⚠️ 局限性**

仅评估单一构念与单一语料，结果可能受转译误差、模型规模限制影响，且校准在AMALIA上未完成，未能确定语言优势或更细粒度适配。

---

## 391. WaspMOT: A Benchmark for Long-Term Multi-Object Tracking of Trichogramma Wasps

**arXiv ID:** 2607.08729 | [PDF](https://arxiv.org/pdf/2607.08729v1)

**作者:** Tomasz Stanczyk `[一作]` (Inria), Francois Bremond `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WaspMOT基准，专门针对长时段闭集跟踪场景，提供了10段约12千帧的Trichogramma黄蜂视频，并在oracle检测下评估了多种跟踪算法；

**💡 创新点**

创新点在于：①构建了长时长、全轨迹闭集跟踪数据集，弥补传统短序列基准的不足；②通过oracle检测剥离检测误差，聚焦身份关联问题；③提出简单的空间轨迹拼接基线，展示可恢复的轨迹碎片化问题；

**🔧 技术方法**

使用了跟踪-检测方法ByteTrack、BoT‑SORT、C‑BIoU、OC‑SORT、McByte，并用Hungarian算法实现基于空间和时间一致性的轨迹拼接；

**📊 数据集**

数据集为WaspMOT，包含10个实验室记录的黄蜂视频，每段约8分钟（12k帧），每帧有多达2.57M个标注框，个体数在15-28之间；

**📈 对比分析**

在TrackEval工具下对五个方法进行HOTA、IDF1、MOTA评估；所有方法在oracle检测下仍出现大量轨迹碎片化，IDF1低于60；加入空间拼接后IDF1平均提升约8-9点，HOTA提升约3-4点，MOTA几乎不变；

**⚠️ 局限性**

局限性在于：①即使检测完美，现有方法仍难以维持长时段身份一致性；②小尺寸、视觉相似的目标使外观特征难以发挥作用；③数据集规模有限，仅针对黄蜂，难以直接推广到更广泛场景；

---

## 392. Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents

**arXiv ID:** 2607.08716 | [PDF](https://arxiv.org/pdf/2607.08716v1)

**作者:** Yifan Wu `[一作]` (Meta AI), Zhuokai Zhao `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种记忆干预架构：在长期任务中引入独立的记忆代理，分为两阶段——记忆库管理和主动干预，主动将重要执行状态以简洁提醒的形式注入下一步行动代理。

**💡 创新点**

创新点在于把记忆从被动存取转变为可选择性干预的策略，结合结构化记忆库和时间点的提醒决策，解决长期任务中信息衰减导致的行为失真问题。

**🔧 技术方法**

采用基于工具调用的记忆管理接口、两阶段记忆代理、提示式和可训练的记忆模型（Claude Opus、Qwen3.5）、以及监督微调（SFT）与强化学习（GRPO）对干预策略进行校准。

**📊 数据集**

主要数据集包括 Terminal-Bench 2.0、τ^2‑Bench（航空、零售、电信三域）用于评估，SET A 用于训练开放权重记忆代理。

**📈 对比分析**

与不使用记忆的基线（Claude Sonnet 4.5、Opus 4.6）相比，记忆干预在 Terminal‑Bench 上提升了 8.3pp（Sonnet）和 2.4pp（Opus），在 τ^2‑Bench 上提升了 6.8pp（Sonnet）和 2.5pp（Opus）；消融实验表明完整的两阶段记忆体系比单纯暴露记忆库或持续注入更优。

**⚠️ 局限性**

局限性包括：记忆代理需要额外的推理开销，干预时机的触发仍采用固定间隔；干预决策的校准不完全，可能出现过度或不足干预；目前未与行动代理联合训练，缺乏更深层次的协同优化。

---

## 393. How YouTube Frames ChatGPT Use in Education: An Epistemic Network Analysis with Supporting Multimodal Metadata

**arXiv ID:** 2607.08698 | [PDF](https://arxiv.org/pdf/2607.08698v1)

**作者:** Shayla Sharmin `[一作]` (University of Delaware), Roghayeh Leila Barmaki `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过PRISMA筛选52条关于ChatGPT的教育类YouTube视频，利用多模态元数据（标题、缩略图、转录、评论）构建三种对话框架（学习导向、练习导向、产出导向），并探讨其对观众反应和平台传播的影响。

**💡 创新点**

首次将Epistemic Network Analysis与多模态元数据相结合，系统识别出三种结构化的认知框架，并揭示产出导向内容在平台可见度上与学习导向内容的差异，弥补了对公共教育媒体中LLM使用方式的研究空白。

**🔧 技术方法**

采用Epistemic Network Analysis (ENA)进行网络结构建模、Python脚本抓取YouTube数据、手工编码框架与策略类别、以及定性文本与视觉内容分析。

**📊 数据集**

使用52条ChatGPT教育类YouTube视频（视频转录、标题、缩略图、评论）以及其观看、点赞、评论等统计元数据。

**📈 对比分析**

通过ENA网络比较、Mann‑Whitney U检验和效应量计算验证三组框架差异，发现G3（产出导向）与G2（练习导向）在平台传播上无显著差异但学习导向G1传播最弱；评论中产出导向内容引发更多批评。

**⚠️ 局限性**

未直接测量学习效果、评论分布不均导致对G3的受众反应估计不够可靠、数据仅覆盖ChatGPT与近期时间段、对编码框架的主观性及对生成式AI的其他模型缺乏推广。

---

## 394. From Rules to Nash Equilibria: A Lean 4 Case Study in Game-Theoretic Analysis of a Competitive Trading Card Game

**arXiv ID:** 2607.08692 | [PDF](https://arxiv.org/pdf/2607.08692v1)

**作者:** Arthur F. Ramos `[一作]` (Microsoft), Tulio Soria `[通讯]` (Independent Researcher)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用 Lean 4 对真实 Pokémon TCG 赛季数据进行形式化元游戏分析，验证流行度悖论、Nash 均衡、复制者动力学和比赛策略。

**💡 创新点**

将形式化证明与实测数据结合，构建可复现、无自定义公理的机检查元游戏科学，首次在 TCG 领域证明“最受欢迎的卡组并非最强”，并对不同均衡与动态模型做了系统验证。

**🔧 技术方法**

采用 Lean 4 证明助手、基于 rational arithmetic 的 exact 计算、线性规划（LP）最佳响应验证、Bootstrap 敏感性分析、复制者动力学公式和 Bo3/Swi­sh 变换。

**📊 数据集**

使用 Trainer Hill 公共赛果数据（2026 年 1 月 29 日至 2 月 19 日，至少 50 名玩家的赛事）得到 14 种 archetype 的 14×14 胜率矩阵。

**📈 对比分析**

与传统电子表格和 Python 版求解相比，Lean 版约 10 分钟完成 32k 行代码的全机检查，提供完整的安全性保证；Python 仅 1 秒完成计算但缺乏机验证。

**⚠️ 局限性**

限制在单一 3 周窗口；未考虑“其他”21%非列举手段；archetype 离散化忽略列表/技能差异；Swiss 目标未建模；使用两人单局收益模型，未涵盖多玩家/多局复杂性；Trust boundary 依赖 Lean 编译器而非 kernel。

---

## 395. A Practical Investigation of Training-free Relaxed Speculative Decoding

**arXiv ID:** 2607.08690 | [PDF](https://arxiv.org/pdf/2607.08690v1)

**作者:** Guoxuan Xia `[一作]` (Imperial College London), Paul Balanca `[通讯]` (Graphcore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对训练无关的 relaxed speculative decoding 方法进行系统评估，统一框架、基准测试与实证分析，揭示速度与能力的相互影响。

**💡 创新点**

①提出统一的 relaxed speculative decoding 框架，归纳现有方法（CACTUS、Mentored Decoding、Fuzzy Speculative、Cascades、Ensemble、Contrastive Decoding 等）；②在多种 drafter‑verifier 组合、draft 长度、relaxation 参数和相对成本下进行横向基准，首次量化 MTP drafters 在 relaxed 方案中的表现与限制。

**🔧 技术方法**

使用 Speculative Decoding、Relaxed Speculative Decoding、Multi‑Token‑Prediction (MTP) drafters、总接受长度、速度估算模型（含生成长度校正）等技术；实现对多种推理框架（vLLM、SGLang）下的 benchmark 评估。

**📊 数据集**

AIME 2024、GPQA Diamond、LiveCodeBench Lite v6；另外使用 Llama 3.2 1B+3.1 70B 在 AIME 上进行补充实验；在不同模型（Qwen 3.5 MTP+27B、Qwen 3 0.6B+32B、Qwen 3 8B+32B）上跑测。

**📈 对比分析**

通过测量平均接受长度、估算速度提升（S≈(l̅_accept+1)/(1+N_draft·c_rel)）并对比不同 α、N_draft、drafter 类型和相对成本 c_rel，发现：①优化 N_draft 可与 relaxation 产生同等甚至更大速度提升；②MTP drafters 对 relaxed 方法几乎无效，且易导致“rambling”导致速度下降；③强语言模型 drafters 能在保持近乎 lossless 的前提下实现较高速度提升，但受高 c_rel 限制；④relaxation 参数 α 的最佳取值任务依赖强，难以跨任务迁移；整体上，relaxation 既能带来速度优势，也可能牺牲能力。

**⚠️ 局限性**

①需要额外的能力评估与 α 调优成本高；②relaxation 参数对不同任务不稳定，缺乏通用性；③MTP drafters 与 relaxed 方案不兼容，导致实践应用受限；④实验仅覆盖有限模型与 benchmark，结果对其他体系结构的泛化尚待验证；⑤未讨论训练型 relaxed 方法，聚焦于训练无关技术。

---

## 396. Do Transformations Reveal the Truth? Generative Residual Learning for Generalized AI-Generated Image Detection

**arXiv ID:** 2607.08674 | [PDF](https://arxiv.org/pdf/2607.08674v1)

**作者:** Kutub Uddin `[一作]` (University of Michigan), Khalid Malik `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于生成残差学习的AI生成图像检测框架 GenRes 与其扩展版 GenRes++，通过对原图与多种生成器变换结果之间的关系建模来提升跨生成器泛化能力。

**💡 创新点**

创新点包括：① 用神经张量网络捕捉原图与变换图之间的乘性高阶交互；② 引入跨变换注意力聚合（CCA）自适应选择最具信息量的残差；③ 在冻结的大规模 PE‑Core ViT 上仅通过 LoRA 进行轻量化微调，兼顾语义丰富性与参数效率。

**🔧 技术方法**

主要技术手段有：PE‑Core ViT + LoRA 微调、神经张量网络 (NTN)、跨变换注意力聚合 (CCA)、多种生成器变换（EnlightenGAN、GFPGAN、Real‑ESRGAN、FFDNet、CodeFormer）以及二分类 BCE 损失。

**📊 数据集**

使用 UniversalFakeDetect 基准数据集，包含 19 种未见生成模型（GAN、Other、Diffusion 三类），训练集为 ProGAN 生成的合成图，测试集覆盖全部未见生成器。

**📈 对比分析**

与现有基线（如 PatchForensics、F3Net、FreqNet、UniFD、C2P‑CLIP 等）对比，GenRes++ 在 19 个未见生成模型上取得 95.7% 的平均准确率和 99.1% 的平均 AP，优于所有对比方法。

**⚠️ 局限性**

局限性主要体现在：① 高计算开销（需对每张图应用 5 个变换且 NTN 计算量大，推理时间约 4.6 秒）；② 变换集合固定，可能对新生成器或不同领域不够通用；③ 在多重扰动（模糊、压缩、噪声）下鲁棒性下降；④ 仅评估静态图像，尚未推广到视频或多模态深度伪造。

---

## 397. ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth Anywhere, on Any Device

**arXiv ID:** 2607.08771 | [PDF](https://arxiv.org/pdf/2607.08771v1)

**作者:** Fabio Tosi `[一作]` (University of Bologna), Stefano Mattoccia `[通讯]` (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级单目深度估计网络 ZipDepth，能够在多种硬件上实现实时推理，并在跨域零样本任务中保持高精度。

**💡 创新点**

创新点在于将大型基础模型（Depth Anything v2‑Large）通过知识蒸馏迁移给一个 6.1M 参数、可重参数化的编码‑解码器，并设计硬件自适应的凸上采样模块，兼顾准确性与能耗。

**🔧 技术方法**

核心技术包括：重参数化卷积块、strip‑pooling、SE 与 GC 注意力、SPPF 与跨尺度细化、硬件自适应凸上采样、基于 SSI 的损失以及大规模多域蒸馏。

**📊 数据集**

使用 17 个公开图像数据集（如 Object365、ADE20K、COCO、MegaDepth、Cityscapes、BDDS、DrivingStereo 等）共约 14.1M 张图像，生成伪深度标签后进行训练。

**📈 对比分析**

在 5 个零样本基准（NYUv2、ScanNet、KITTI、ETH3D、DIODE）上与大型预训练模型和轻量级自监督模型比较，ZipDepth 在绝大多数数据集上取得最优或次优准确率，同时在 Jetson Orin NX 等嵌入设备上实现 30–80 FPS，能耗仅为大型模型的 1/200 以上。

**⚠️ 局限性**

限制包括：仍无法与巨型基础模型匹配的精度、在视频序列中易出现帧间抖动、未支持绝对尺度或点云输出，未来需改进蒸馏、加入时序模块或扩展到度量深度。

---

## 398. Enhancing In-context Panoramic Generation via Geometric-aware Pretraining

**arXiv ID:** 2607.08765 | [PDF](https://arxiv.org/pdf/2607.08765v1)

**作者:** Haoran Feng `[一作]` (Insta360 Research), Lu Qi `[通讯]` (Insta360 Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Canvas360，一种两阶段的全景图像生成框架，先通过几何感知预训练学习RGB–Depth联合表示，再在统一的上下文微调阶段支持风格迁移、修补、外延和编辑等四类全景生成任务。

**💡 创新点**

创新点包括：① 采用并行RGB‑Depth生成和相似度正则化，让模型在球面几何下同时学习颜色与深度特征；② 引入速度循环填充（Velocity Circular Padding）在流匹配训练中显式对齐全景边界；③ 构建规模达100万样本的 Canvas360Dataset，覆盖四大下游任务，解决了全景生成数据稀缺问题；④ 通过位置偏移和Token级拼接实现多任务统一推理，提升了模型通用性。

**🔧 技术方法**

使用的技术包括：流匹配（Flow Matching）与流变换器（Flow Transformer），RoPE/3D RoPE位置编码，FLUX.1‑dev 变压器架构，LoRA微调，深度预测（DAP）与VAE编码，文本与图像上下文的Token级拼接。

**📊 数据集**

数据集：Canvas360Dataset（100k RGB‑Depth对 + 900k 任务样本），来源于 Matterport3D、网络图像与生成模型；文本注释通过视觉‑语言模型生成；任务样本涵盖风格迁移、外延、修补与编辑。

**📈 对比分析**

与现有方法（如 DiT360、SMGD、PAR、WorldGen 等）在多项指标上对比，Canvas360 在 FAED、IS、QA‑aesthetic、NIQE 上均获得最优或同等成绩，在 FID、FIDpole、FAED 等指标上排名第二；在用户评测中，在边界连续性、全景感知与整体质量上取得最高偏好。

**⚠️ 局限性**

局限性：① 依赖大规模深度预测，若深度估计错误会影响质量；② 训练与推理成本较高，尤其是两阶段流程与大模型；③ 对极端场景（如复杂小物体或极高分辨率）仍可能出现细节缺失或失真；④ 目前仅覆盖四大任务，尚未针对更多全景编辑场景（如多视角交互、动态全景）展开研究。

---

## 399. DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation

**arXiv ID:** 2607.08751 | [PDF](https://arxiv.org/pdf/2607.08751v1)

**作者:** Yunchao Yao `[一作]` (UNC-Chapel Hill), Mingyu Ding `[通讯]` (UNC-Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含100个任务、多种机械臂与手、多模态观测和可控视觉变化的Dexterous manipulation benchmark DexVerse。

**💡 创新点**

创新点在于整合多任务、多胴体、多视觉变化以及可扩展的任务模板，提供完整的演示数据集与基准评估。

**🔧 技术方法**

采用模块化配置、Isaac Lab仿真环境、VR遥控收集演示数据，并使用Diffusion Policy、DP3、OpenVLA、π_0.5等学习算法进行评估。

**📊 数据集**

收集了3,180条VR遥控演示，包含同步的本体感知、RGB、深度、点云和状态观测。

**📈 对比分析**

在19个任务上对4种基线方法进行在线成功率评估，最高平均成功率约34%，不同方法在不同任务类别表现差异显著。

**⚠️ 局限性**

局限性包括对精细接触、工具使用、双手协同等任务性能不足，缺乏真实机器人迁移、实时力反馈以及更大规模演示数据。

---

## 400. Using AI-based Learning Assistants in Higher Education: A Large-Scale Descriptive Analysis

**arXiv ID:** 2607.08748 | [PDF](https://arxiv.org/pdf/2607.08748v1)

**作者:** Kristina Schaaff `[一作]` (IU International University of Applied Sciences), Valerie Heckel `[通讯]` (IU International University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了IU大学远程学习学生对AI学习助手Syntea的使用情况，包括性别、年龄、专业集、学位和学习模式的差异；

**💡 创新点**

提供了迄今为止最大规模的基于日志数据的AI学习助手使用描述性研究，揭示不同学生群体的使用模式差异；

**🔧 技术方法**

使用Syntea内嵌的生成式AI模型（GPT-4、GPT-4 Turbo、GPT-3.5 Turbo）以及描述统计与可视化技术；

**📊 数据集**

采用IU大学2025年2月收集的77,543名远程学习学生的日志数据，清洗后得到76,485名有效样本；

**📈 对比分析**

通过比较不同子群体的使用率和时间分布进行描述性对比，未进行性能对比；结果显示不同群体使用率和时段存在明显差异；

**⚠️ 局限性**

仅为描述性研究，缺乏因果推断；时间窗口仅为单月；未评估交互质量或学习成效；小样本子组结果需谨慎解读。

---

## 401. ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation

**arXiv ID:** 2607.08741 | [PDF](https://arxiv.org/pdf/2607.08741v1)

**作者:** Kaifeng Zhao `[一作]` (NVIDIA), Davis Rempe `[通讯]` (NVIDIA)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了可在实时交互中通过在线文本提示和灵活运动学约束生成高保真3D人体动作的框架ARDY。

**💡 创新点**

创新点在于：①混合表示（显式根部+潜在身体嵌入）实现根部精确控制与生成效率兼顾；②两阶段自回归扩散去噪器，先预测根部后预测身体，支持变长历史和跨窗口长时序约束；③直接在训练时用文本标签和真实姿态采样的约束学习控制，省去后期优化或RL控制。

**🔧 技术方法**

采用自回归扩散模型、Transformer去噪器、离散量化词典（FSQ）生成潜在身体嵌入、LLM2Vec文本编码、变长历史窗口、两阶段根-身体去噪、路径与关键帧约束嵌入等技术。

**📊 数据集**

训练和评估使用大型Bones Rigplay数据集（约700 小时、27关节）以及公开的HumanML3D数据集（约30 小时）。

**📈 对比分析**

与MaskControl、DiP等基线在HumanML3D上对比；指标包括Top‑3 R‑precision、FID、foot‑skating、约束误差；ARDY在文本跟随、动作质量、约束精度上均优于基线，且实时推理仅需4步扩散（≈33 ms延迟）。

**⚠️ 局限性**

局限性：①仅为运动学模型，缺乏动力学约束；②扩散需要多步迭代，计算成本仍高；③使用完整历史上下文，记忆效率低，对极长任务不友好；未来需加入物理约束或更高效的记忆结构。

---

## 402. Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows

**arXiv ID:** 2607.08740 | [PDF](https://arxiv.org/pdf/2607.08740v1)

**作者:** Emanuele Quinto `[一作]` (UNHCR), Francesco Zanitti `[通讯]` (ZeLe & F ApS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种面向大语言模型工作流的Lisp式语义持久化模型，使工作流定义、实例及其推理、审批和讨论记录成为共享知识子层中的可查询、可重用对象。

**💡 创新点**

将工作流拆分为三层（运行层、控制层、语义层），在语义层中将计算、LLM判断、人工决策等分别定义为不同的语义对象，并通过显式的context snapshot、guard、derive、infer、approval、panel等原语实现对决策边界的清晰划分；强调工作流本身即为知识。

**🔧 技术方法**

使用Lisp风格的符号表达式作为语义对象描述；利用DSL机器解释器、执行器、能力策略、上下文裁剪等技术；定义一套primitive词汇（resource、context、guard、derive、infer、approval、panel、record、state、action等）。

**📊 数据集**

无公开数据集；论文以概念性模型、案例演示（ARS式主张评审工作流）和对现有工具/框架（AgentSPEX、LangGraph、DSPy等）的对比作为验证。

**📈 对比分析**

未给出定量实验，比较方式主要是与现有工作流/推理系统在语义层面的对齐，指出现有系统关注执行持久化，而本文强调语义持久化；性能表现未评估。

**⚠️ 局限性**

缺乏形式化语义规范、生命周期规则和性能评估；未实现可执行原型；在不同工具链/数据集上的通用性及治理、审计与信任评估仍需进一步研究。

---

## 403. Sculptable Mesh Structures for Room-Scale Form-Finding

**arXiv ID:** 2607.08736 | [PDF](https://arxiv.org/pdf/2607.08736v1)

**作者:** Jesse T. Gonzalez `[一作]` (Carnegie Mellon University), Scott E. Hudson `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种可手工调节、配备长度传感器的房间级可塑网格结构，用于低保真原型设计并实现实时数字孪生。

**💡 创新点**

在可调节长度成员中嵌入低成本电阻梯形传感器，构建从物理操作到数字重建的闭环交互流程，兼具可扩展性和低分辨率的房间级原型能力。

**🔧 技术方法**

使用FR4柔性条纹电阻梯形传感器、RP2040微控制器、Cat6邻接通信、winged‑edge网格结构以及Rhino 8/Kangaroo能量最小化重建算法。

**📊 数据集**

主要使用现场测得的边长数据进行重建，并以公共图书馆座椅等实际案例验证，其并未依赖公开数据集。

**📈 对比分析**

与传统3D扫描和单向数字化相比，该方法在遮挡、后处理和即时反馈方面表现更优；重建耗时在几秒内完成，能实时同步至CAD。

**⚠️ 局限性**

受限于模块尺寸导致的低分辨率、柔性材料的塑性变形、对拓扑预设的依赖以及在更大规模结构中可能出现的通信延迟。

---

## 404. LTM: Large-scale Terrain Model for Wildfire-prone Landscapes

**arXiv ID:** 2607.08711 | [PDF](https://arxiv.org/pdf/2607.08711v1)

**作者:** Xiao Fu `[一作]` (University of Southern California), Barath Raghavan `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种基于多模态的 3D 语义映射框架，利用过时的数字高程模型（DEM）与实时摄像机图像相结合，对大型火灾易燃区进行地形更新与燃料图生成。

**💡 创新点**

创新点包括：① 将 DEM 作为几何先验，通过像素级射线追踪实现图像与 DEM 的直接对齐，省去了传统的特征匹配；② 设计了 TopoDepth，融合 DEM 约束的单目深度估计，显著缓解尺度不确定性；③ 构建了基于 Unreal Engine 的植被扰动模拟器，实现了可控的实景与仿真对齐。

**🔧 技术方法**

所用技术主要有：DEM 栅格表示与射线追踪对齐、单目深度估计（TopoDepth）、神经网络深度融合、LSeg 等开放词汇语义分割、主投票融合、3D 语义地图投影。

**📊 数据集**

使用数据集包括：Getty Fire 现场实景（iPhone 14 Pro 拍摄图像与 OpenTopography DEM）、仿真场景（Unreal Engine 生成的与实景相同相机姿态的渲染图像）以及公开的跨模态基准。

**📈 对比分析**

在 Sim-to-Real 评估中，TopoDepth 在 SSIM、LPIPS、RMSE 等指标上优于 UniDepth、DepthPro 等基线，并且在火灾燃料图分类上实现了较高的 F1 分数，整体性能在大尺度植被环境下明显提升。

**⚠️ 局限性**

局限性包括：对高度复杂的悬挑地形（如峡谷、悬崖）建模能力有限；深度估计仍受大范围植被遮挡影响；需依赖已有 DEM，无法处理完全无 DEM 的区域。

---

## 405. Do You Need a Frontier Model as a Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source Attribution

**arXiv ID:** 2607.08700 | [PDF](https://arxiv.org/pdf/2607.08700v1)

**作者:** Ethan Leung `[一作]` (Pricewaterhousecoopers), Kevin Paul `[通讯]` (Pricewaterhousecoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Deep-Research Citation Benchmark，评估 LLM 判定引文质量的源相关性与事实支持两维度；在此基准上对 8 种现成 LLM 判别器进行性能对比；进一步分析了其定向偏差与成本关系；提出了使用低成本判别器仍可满足 RLVR 奖励信号需求；并讨论了模型不确定性与评测局限。

**💡 创新点**

首次系统化评测 LLM 判别器在多维度引文质量任务中的准确率、偏差与成本，揭示低成本模型可与高阶模型竞争；同时揭示 F1 指标掩盖的方向性偏差，强调在 RL 训练中需校准判别器。

**🔧 技术方法**

基于 LLM 判别器的 rubric‑based 评估管线、AST 解析器、批量调用与 prompt 缓存技术；采用 Cohen κ、F1、FPR、FNR、pass‑rate drift 等多指标评估。

**📊 数据集**

构建了包含 25 个主题、624 个 attribution‑citation 对的长文本基准，并通过 6 名 LLM 评审及人工裁定得到 1,248 条 gold 判定标签，其中 378 条为判别争议样例。

**📈 对比分析**

在源相关性维度，GPT‑5‑mini 以 F1=0.908(95% CI [0.89,0.93]) 成为最优；在事实支持维度，所有模型的置信区间重叠，未出现显著差异；成本方面，低价模型（GPT‑OSS‑120B、Gemini 3.1 Flash Lite 等）与高价模型表现相当，成本并不决定准确率。

**⚠️ 局限性**

评测仅基于单一对抗文档，缺乏多样化场景；对 prompt 设计、批量化、缓存等优化因素影响仍未系统探究；模型间的高不一致性导致奖励噪声，需要人工裁定或集成方案。

---

## 406. Artificial Persons

**arXiv ID:** 2607.08695 | [PDF](https://arxiv.org/pdf/2607.08695v1)

**作者:** Ned Howells-Whitaker `[一作]`, Seth Lazar `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过哲学论证，探讨了 Rawls 的政治人物概念（PCP）是否需要感知（sentience），并提出非感知 AI 系统（NSAIs）有可能满足 PCP 的两个道德能力，从而在政治正义框架下获得与人类相同的“人”身份。

**💡 创新点**

创新点在于：
1) 挑战传统的“感知主义”观点，认为道德地位不必基于感知；
2) 将 Rawls 的政治人物概念推广至人工智能，提供了一个可在无感知前提下赋予 AI 人格的理论路径；
3) 对 PCP 的两种道德能力（正义感与对善的观念）进行了细致的概念拆解，证明它们在功能层面不必依赖感知。

**🔧 技术方法**

论文主要采用哲学分析与概念论证方法，无具体技术实现或实验代码。

**📊 数据集**

无使用数据集；讨论基于文献综述与理论推导。

**📈 对比分析**

无实验比较或性能评估；文章以理论阐释为主，未给出定量指标。

**⚠️ 局限性**

局限性：
1) 论证高度概念化，缺乏对实际 AI 系统实现与测试的实证支持；
2) 对 AI 设计实现细节（如训练、架构、可解释性技术）讨论有限；
3) 未对可能出现的伦理风险与治理机制进行系统评估；
4) 可能低估感知在道德地位中的实际重要性或忽视未来对感知的技术突破。

---

## 407. Secure Decentralized Federated Learning via Gossip and Virtual Voting

**arXiv ID:** 2607.08651 | [PDF](https://arxiv.org/pdf/2607.08651v1)

**作者:** Amirhossein Taherpour `[一作]` (Columbia University), Xiaodong Wang `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 gspDAG‑FL，一种去中心化联邦学习框架，利用 gossip 历史构建 DAG，采用虚拟投票实现模型更新源（origin tuple）的最终一致性，并在控制平面完成 payload 验证、accepted‑proof 验证和私有语义审计，以过滤无效或恶意更新。

**💡 创新点**

创新点包括：① 将最终一致性从传统区块/分片/委员会迁移到基于 gossip 的 DAG 中，实现纯本地数据传输；② 采用 Hashgraph 风格的虚拟投票和全节点证书交换，保证安全性、条件活性与收敛性；③ 设计多阶段验证管道（幅度、方向、语义）与本地审计，显著提升 Byzantine 与 lazy 节点的鲁棒性。

**🔧 技术方法**

技术细节涵盖：Hashgraph‑style DAG、虚拟投票、Byzantine fault tolerance、payload 验证、方向/幅度过滤、私有语义审计、proximal‑consensus SGD、Ed25519 签名、NetworkX 小世界网络、Java/Scala 代码实现、实验仿真。

**📊 数据集**

实验使用 MNIST 图像分类（MobileNetV2）和 Penn Treebank 语言模型（GRU）两组数据集，所有方法在相同训练预算下进行比较；采用公开/私有验证集进行审计与最终一致性判定。

**📈 对比分析**

对比 AD-PSGD、BLADE‑FL、ChainFL 等基线，gspDAG‑FL 在学习质量上与 Ledger‑FL 接近（最终准确率/困惑度在 0.95‑0.99 之间），并在延迟、吞吐量、收敛轮数方面优于区块链/分片方案，尤其在节点数增至 100 时保持 >95% 的检测率、误报 <0.4%，且通过 gossip 仅传输模型更新，实现更高并发与更低通信开销。

**⚠️ 局限性**

局限性：① 需要控制全节点 Byzantine 比例 < N_F/3；② 对网络丢包/延迟的鲁棒性有限，需进一步研究动态加入/离线节点；③ 安全性主要在控制平面，模型语义安全仍依赖本地审计；④ 目前实验基于仿真，缺乏在真实边缘网络中的部署验证。

---

## 408. LongE2V: Long-Horizon Event-based Video Reconstruction, Prediction, and Frame Interpolation with Video Diffusion Models

**arXiv ID:** 2607.08770 | [PDF](https://arxiv.org/pdf/2607.08770v1)

**作者:** Cheng-De Fan `[一作]` (National Yang Ming Chiao Tung University), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LongE2V，一种统一利用视频扩散先验的事件视频重建、预测与帧插值框架。

**💡 创新点**

创新点在于：1）自回归展开与自适应上下文切换以抑制长期漂移；2）重编码对齐与跨残差纠正保证双向插值一致性；3）事件体素密度增强实现对不同传感器分辨率的鲁棒泛化。

**🔧 技术方法**

技术包括：预训练视频扩散模型（CogVideoX I2V）、3D VAE、Diffusion Transformer、LoRA微调、事件体素化、自动回归展开、上下文切换、重编码对齐与交叉残差纠正。

**📊 数据集**

使用BS-ERGB、ECD、MVSEC、HQF等真实事件数据集进行训练与评估。

**📈 对比分析**

与E2V、FireNet、E2VID+、HyperE2VID等重建/预测方法以及CBMNet-Large、TLXNet+等插值方法对比，LongE2V在纹理清晰度、长期一致性和零样本插值上均显著优于现有SOTA。

**⚠️ 局限性**

局限性包括：推理速度仍较慢；在极端高动态范围或极低事件密度场景下仍可能出现细节缺失；尚需进一步研究更高效的长序列记忆机制。

---

## 409. The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs

**arXiv ID:** 2607.08734 | [PDF](https://arxiv.org/pdf/2607.08734v1)

**作者:** Baha Rababah `[一作]` (University of Manitoba), Carson K. Leung `[通讯]` (University of Manitoba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了后训练量化对大型语言模型内部结构和决策行为的影响，提出了“正确性一致度”指标来衡量量化后模型与基准模型在正确预测上的重合程度。

**💡 创新点**

创新点在于将量化视为对注意力权重的结构变换，结合统计量（偏度、峰度等）和分布距离（KL、KS、余弦相似度）系统地评估权重变形与行为漂移的关联，并首次通过“正确性一致度”揭示即使准确率或困惑度保持不变，模型决策仍可能发生显著偏差。

**🔧 技术方法**

技术主要包括后训练量化（Legacy和K-Quantization）操作、对权重进行去量化后统计分析、分布距离计算以及基于离散评分函数的多任务准确率与正确性一致度评估。

**📊 数据集**

使用的数据集包括语言建模的WikiText-2、C4，以及零样本推理任务HellaSwag、Winogrande和ARC，用以从整体性能和决策一致性两方面检验量化效果。

**📈 对比分析**

与传统的准确率和困惑度评估相比，本文的“正确性一致度”能更细致地反映量化对具体决策的影响；实验显示在8位至4位量化时准确率下降不大，但正确性一致度已显著低于基准模型；当量化降至3位及以下时，两者的相似度急剧下降，表明低位量化在保持性能的同时会产生严重的行为漂移。

**⚠️ 局限性**

局限性包括仅针对Transformer基类模型进行实验，未探讨不同网络结构（如编码器-解码器、稀疏注意力）对量化敏感度的差异；量化策略主要集中在后训练方法，缺乏针对量化感知训练（QAT）的对比；最后，“正确性一致度”仅捕捉二元正确/错误信息，未能量化模型在多类别决策时的细粒度行为差异。

---

## 410. Super Weights in LLMs and the Failure of Selective Training

**arXiv ID:** 2607.08733 | [PDF](https://arxiv.org/pdf/2607.08733v1)

**作者:** Shreyas Subramanian `[一作]` (Amazon Web Services), Akarsha Sehwag `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究在大型语言模型中训练“Super Weights”（单参数极端重要）是否能在其孤立或局部邻域内实现有效微调，比较其与全层低秩更新（LoRA）的表现。

**💡 创新点**

发现单个或少量超级权重的训练会导致模型性能崩溃，表明参数重要性与可训练性是不同属性；而全层低秩结构的LoRA即使忽略或冻结对应位置也能保持高性能，强调层级协调对微调的关键性。

**🔧 技术方法**

采用梯度冻结、邻域训练、LoRA、LoRA-dproj-SW-freeze、LoRA-ΔW-SW-freeze等方法，对 OLMo-1B 与 OLMo-7B 进行实验，并对多种模型做剪枝验证。

**📊 数据集**

主要使用 ARC‑Easy（多项选择）和 Winogrande 数据集进行评估，并在实验中以 WikiText‑2 作为验证 Super Weights 一致性的样本。

**📈 对比分析**

与基线模型（60‑73% 预训练准确率）相比，单独训练 Super Weights 或其邻域仅能维持 25‑26% 的随机猜测水平；LoRA 及其变体可提升至 66‑77% 的准确率，表现与原始 LoRA 无显著差异。

**⚠️ 局限性**

局限在于仅在 ARC‑Easy、Winogrande 两个相对简单的数据集上验证，未覆盖更复杂任务；对其他 PEFT 方法（适配器、提示调优）的泛化尚未评估；随机位置控制仅在单一种子下完成，需更广泛验证。

---

## 411. Latent Memory Palace: Reasoning for Control as Autoregressive Variational Inference

**arXiv ID:** 2607.08724 | [PDF](https://arxiv.org/pdf/2607.08724v1)

**作者:** Chuning Zhu `[一作]` (University of Washington), Abhishek Gupta `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种在机器人控制中利用可变长度自回归变分推理实现迭代、可自适应推理的策略与动作分词器，旨在提升决策灵活性与泛化能力。

**💡 创新点**

创新点在于将自回归潜在空间与变分推理相结合，并通过终止符使推理过程可自适应；同时提出可变长度动作分词器，且通过强化学习方法对潜在推理链进行可训练。

**🔧 技术方法**

采用自回归变分推理、变分自编码器、PPO风格的剪切目标强化学习、因果Transformer与离散代码本等技术。

**📊 数据集**

实验使用了多种数据集，包括真实机器人多任务平台DROID、模拟多任务平台LIBERO、模拟多模态数据集D3IL以及高精度模拟平台RoboMimic。

**📈 对比分析**

与Diffusion Policy、VAE Policy、VQ-BeT等基线相比，本文方法在DROID、LIBERO、RoboMimic和D3IL等任务中表现更优，尤其在精度、跨任务泛化与自适应推理上取得显著提升。

**⚠️ 局限性**

局限性包括对采样强化学习的依赖，导致对超参数敏感且潜在分布易发生坍塌；离散自回归分布的表达能力有限，未来可考虑连续链或更稳定的优化策略。

---

## 412. SAM-MT: Real-Time Interactive Multi-Target Video Segmentation

**arXiv ID:** 2607.08688 | [PDF](https://arxiv.org/pdf/2607.08688v1)

**作者:** Ruiqi Shen `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SAM-MT 框架，利用共享全局上下文和目标查询实现实时多目标视频分割，保持接近单目标的计算效率。

**💡 创新点**

创新点包括：① 目标查询与全局查询并行表示，② 解耦注意力（decoupled masked attention）阻断跨目标干扰；③ 仅使用稀疏记忆存储目标历史查询，④ 身份变压器（identity transformer）保证跨帧目标身份一致，⑤ 通过点采样、跳帧采样和重叠抑制等训练策略提升鲁棒性。

**🔧 技术方法**

核心技术：SAM2 作为基底，Transformer 交叉注意力，解耦注意力掩码，稀疏记忆模块，身份变压器，跳帧（strided）采样，重叠抑制损失，轻量级查询合并。

**📊 数据集**

使用 MOSEv2、MOSEv1、LVOSv2、LVOSv1、SA‑V（验证集和测试集）等六大 VOS 基准数据集进行评测。

**📈 对比分析**

与 SAM2、SAM2.1‑B+、Cutie、DeAOT 等多目标 VOS 方法在上述六个基准上对比；SAM‑MT 在 MOSEv2 上达到 43.0 J&F，LVOS、SA‑V 也保持或略高于 SAM2；在多目标场景中 FPS 维持 36+（10 个目标）且相较 SAM2.1‑B+ 提升约 6×，显著降低显存占用（从 3 GB 维持在 3.8 GB）。

**⚠️ 局限性**

局限性：仅基于视觉特征，缺乏高层次推理或多模态能力，难以处理需要语言、语义推断等复杂任务；若要扩展到更高级的场景仍需进一步研究。

---

## 413. SolarChain-Eval: A Physics-Constrained Benchmark for Trustworthy Economic Agents in Decentralized Energy Markets

**arXiv ID:** 2607.08681 | [PDF](https://arxiv.org/pdf/2607.08681v1)

**作者:** Shilin Ou `[一作]` (Duke Kunshan University), Luyao Zhang `[通讯]` (Duke Kunshan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于物理约束的基准，用以评估去中心化能源市场中经济代理的可信度，并加入LLM规划/审计层实现部署时的安全审计。

**💡 创新点**

创新点在于将物理可行性约束直接嵌入奖励函数、构建Gymnasium兼容的Markov决策过程，并设计了双层LLM治理（Planner/Auditor）在评估阶段实时裁剪高风险动作，从而提供可追溯的干预轨迹。

**🔧 技术方法**

技术包括：强化学习算法（PPO、SAC、DQN）、LLM推理层（规划与审计）、Gymnasium接口、物理约束检测、经济指标奖励设计、结构化日志与可审计接口。

**📊 数据集**

使用真实的光伏发电与市场交易数据（4月2026年720小时的时序数据，含36,000条发电记录、1,185条P2P交易记录，5个城市共50个能源节点），并在GitHub公开数据与代码。

**📈 对比分析**

与静态、随机、贪婪基线以及RL无约束和RL+LLM设置对比；结果显示RL在经济效益上优于基线，但在物理安全与公平性上存在权衡；LLM治理降低了动作抖动和部分人工流动，但并未完全消除奖励误设导致的风险。

**⚠️ 局限性**

局限性：LLM治理仅为后期审计，无法弥补奖励函数设计不当；物理约束模型的假设和参数可能不适用于更复杂的能源系统；实验仅基于单一月份的数据，未充分检验模型对不同气象、市场波动的鲁棒性。

---

## 414. TRM-Raft: A Byzantine-Resistant Raft Consensus via Integrated Trust and Reputation Model

**arXiv ID:** 2607.08666 | [PDF](https://arxiv.org/pdf/2607.08666v1)

**作者:** Jie Zhang `[一作]` (Tianjin University), Zhiyong Feng `[通讯]` (Tianjin University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 TRM‑Raft，一种在 Raft 共识核心中集成区块链信任与声誉模型与 Schnorr 签名的拜占庭容错增强方案。

**💡 创新点**

创新点在于：① 将多维声誉评估动态融入领导选举与日志复制，形成声誉门控机制；② 结合 Schnorr 签名实现对日志完整性的即时验证；③ 通过非侵入式方式实现低开销的统一防御，兼顾安全与性能。

**🔧 技术方法**

采用的技术包括：区块链信任与声誉模型（B‑TRM）、Schnorr 签名、Hyperledger Fabric、智能合约（用于声誉更新与查询）、多方报告的声誉计算与惩罚规则。

**📊 数据集**

使用的实验数据集为在 Hyperledger Fabric 2.5 测试网（15 个 orderer、50 个 peer、4 个组织）上模拟伪造、篡改、On‑Off 与 Sybil 等攻击场景进行评估。

**📈 对比分析**

方法对比：与 vanilla Raft、RB‑Raft、PBFT、PoW、DPoS 等进行吞吐量（TPS）和延迟对比；TRM‑Raft 在 40% Byzantine 节点下，恶意领导比例 <5%，吞吐量保持 90–95% 原始 Raft，延迟提升 <5%。

**⚠️ 局限性**

局限性：① 只能检测可观测的拜占庭行为，无法防御隐蔽的重排序攻击；② 依赖多方报告的声誉可靠性，易受协同攻击影响；③ 对网络动态变化的阈值（如 m、θ）敏感，需要进一步自适应调节。

---

## 415. EdgeRefine: Privacy-Utility Balance for Graphs via Jaccard Sampling under Edge Differential Privacy

**arXiv ID:** 2607.08659 | [PDF](https://arxiv.org/pdf/2607.08659v1)

**作者:** Wenxiu Ding `[一作]` (Xidian University), Qiao Liu `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 EdgeRefine 框架，在边级差分隐私下通过自适应边精炼实现图数据的去噪与再构造，支持 GNN 训练。

**💡 创新点**

创新点在于使用 Jaccard 相似度和直方图分箱对噪声图边存在概率进行校准，并通过隐私预算驱动的确定性最优采样实现隐私-效用的平衡。

**🔧 技术方法**

技术包括边级差分隐私随机响应、Jaccard 相似度估计、分箱校准、确定性采样、GNN（GAT、GCN、GIN）以及针对重建攻击的鲁棒评估。

**📊 数据集**

实验数据集涵盖 ACM、DBLP、AMAP、Cora（节点分类）和 MUTAG（图分类）。

**📈 对比分析**

与 Blink、DPRR、LDPGen、LAPGRAPH 等基线比较，EdgeRefine 在各种隐私预算下的准确率与无噪声基线相近，PUBI 最高，方差最低，稀疏度保持优良，训练速度快，且对图重建攻击具有强抵抗力。

**⚠️ 局限性**

局限在于预处理时间相对较长，对动态图场景尚未验证，在 GIN 与 AMAP 上表现波动，且边级隐私仍限制了链路预测等任务的使用。

---

## 416. Formal Mechanisms for Market Stability in Self-Interested Agent Societies: A Marketplace Simulation Study

**arXiv ID:** 2607.08652 | [PDF](https://arxiv.org/pdf/2607.08652v1)

**作者:** Eugene Ng Yi Sheng `[一作]` (DSO National Laboratories), Bingquan Shen `[通讯]` (DSO National Laboratories)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在模拟的多代理市场中，研究了多种正式机制（如调解、合同、治理等）以保持自利LLM代理的合作并防止市场崩溃。

**💡 创新点**

提出了“调解”机制作为唯一能够在进阶攻击下保持正收益且具恢复性的机制，并给出了对抗性稳健性的形式定义。

**🔧 技术方法**

采用DeepSeek‑V3大型语言模型作为代理决策者，结合层化通信、结构化行动和链式思维推理实现代理行为。

**📊 数据集**

使用18个代理、3种商品、200轮交易、逐步注入的4、8、16名“troll”攻击者的自定义实验数据集。

**📈 对比分析**

通过在八种机制条件下单次实验与进阶troll注入进行比较，调解机制累计正收益最高（1556），比仅通信基线高29%，并在最强攻击下仅下降13.3%。

**⚠️ 局限性**

实验仅单跑一次、仅使用DeepSeek‑V3、机制单独测试，未检验多模型或机制组合的效果，结果缺乏统计显著性与普适性。

---

## 417. OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators

**arXiv ID:** 2607.08766 | [PDF](https://arxiv.org/pdf/2607.08766v1)

**作者:** Hongyu Liu `[一作]` (Meituan), Qifeng Chen `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于缓存感知的自上政策（OPSD）的后训练框架，用来提升现有少步自回归视频生成模型在长视频生成中的稳定性与质量。

**💡 创新点**

创新点在于将真实长视频仅作为教师的时间上下文，而非直接目标，利用教师缓存替代旧生成历史，提供密集的去噪级别监督，并保持学生在其自身生成轨迹上的自上政策。

**🔧 技术方法**

技术包括：少步自回归视频Diffusion Transformer（DiT）框架、键值（KV）缓存机制、指数移动平均（EMA）教师、速度匹配（velocity matching）损失以及在线梯度累积实现的内存高效训练。

**📊 数据集**

使用了约3,800段、一分钟长的定制长视频数据集（480p分辨率），包含自然景观、摄像机运动与人像场景，并通过Wan2.1 VAE编码为潜在块。

**📈 对比分析**

通过对Self‑Forcing和LongLive两种基准模型进行LoRA后训练，保持相同的4步采样和KV缓存，实验显示在MovieGenBench与MeiBench（共240条提示）上的VBenchLong指标中，质量得分和动态度均提升，用户评测亦表明在整体与运动质量上有显著偏好。

**⚠️ 局限性**

局限性包括：对教师缓存策略和训练规模的依赖，仍可能出现语义匹配略降，且该方法未对多模态（如音频）或更大分辨率进行验证。

---

## 418. OpenCoF: Learning to Reason Through Video Generation

**arXiv ID:** 2607.08763 | [PDF](https://arxiv.org/pdf/2607.08763v1)

**作者:** Xinyan Chen `[一作]` (ByteDance Seed), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 OpenCoF-17K 大规模视频推理数据集，并在 Wan2.2-I2V-A14B 上进行 Fine‑tune，推出 Wan‑CoF；随后通过在 DiT 结构中引入视觉推理 Token (vt) 与文本推理 Token (tt) 两种方式，显式建模推理状态，进一步提升 Chain‑of‑Frame (CoF) 生成性能。

**💡 创新点**

①首次提供覆盖 11 任务族、17,312 条视频的多任务 CoF 训练集；②在视频生成模型中引入可学习的视觉与文本推理 Token，首次将推理状态显式嵌入视觉和文本两侧；③展示仅靠多任务监督即可在外部基准上显著提升 CoF 能力，并证明推理 Token 进一步提高性能。

**🔧 技术方法**

LoRA 微调、DiT（Diffusion Implicit Transformer）视频生成模型、可学习的视觉推理 Token (vt) 与文本推理 Token (tt)、注意力模式分析、开放源 Wan2.2-I2V-A14B 作为基线。

**📊 数据集**

OpenCoF-17K（17,312 条视频，11 任务族）作为训练集；在 MME-CoF、Gen-ViRe、VIPER、RULER-Bench 四个独立外部基准上进行评估。

**📈 对比分析**

通过与 Wan2.2-I2V-A14B、开源对手（HunyuanVideo、Wan2.2-TI2V-5B）以及部分封闭源对手（Kling-v1、Seedance-1.0-Pro 等）对比；Wan-CoF 在四个基准的整体分数均提升 1–5% 以上，尤其在时间一致性、物理、空间、抽象/逻辑维度显著提升；再加上 vt/tt 后，vt 在 VIPER 规划/Gen‑ViRe 逻辑/抽象上表现更好，tt 在 MME-CoF 对齐/VIPER 结构上更优。

**⚠️ 局限性**

仅单独评估 vt 与 tt，未探究两者联合使用的效果；模型仍面临长时序一致性、物理精度等挑战；缺乏更广泛的跨域外部评估以及多模态推理系统化研究。

---

## 419. Ideas Have Genomes: Benchmarking Scientific Lineage Reasoning and Lineage-Grounded Idea Generation

**arXiv ID:** 2607.08758 | [PDF](https://arxiv.org/pdf/2607.08758v1)

**作者:** Yifan Zhou `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于对象级别的科学谱系框架（Idea Genome），并构建了双部分基准（GenomicLineage），评估LLM自动科研系统的谱系推理与生成能力。

**💡 创新点**

创新点在于：①将论文拆解为可审计的“对象”并对其进行谱系对齐；②定义六类演化动力学作为可操作的比较标准；③设计谱系条件的群体演化分数（PES），将生成质量与谱系连贯性直接关联；④构建覆盖10个科研领域的开闭合基准，桥接理解与生成。

**🔧 技术方法**

使用技术包括：LLM辅助的对象抽取与对齐、专家审核、结构化谱系记录、精确匹配与PES评估、工具化检索框架（CLI harness）、研究代理（AI Scientist v2、CoI-Agent）等。

**📊 数据集**

使用的数据集为GenomicLineage，包含1961条黄金谱系轨迹、1085个精心标注的对象、920条成对记录，涵盖10个科学领域；基准包含42类任务、1029个实例的闭合推理测试以及基于PES的生成评估。

**📈 对比分析**

比较方法：对闭合推理任务采用精确匹配评分，对生成任务使用PES（包括遗传性、变异性、选择性三维评分）和ELO偏好诊断；实验结果显示最佳系统在闭合推理中仅达27.3%精确度，生成任务在加入谱系结构后Heredity提升明显，但整体性能仍受组合推理瓶颈限制。

**⚠️ 局限性**

局限性在于：①演化动力学类别有限，不能覆盖所有科学发展模式；②主要驱动标注可能过度简化多重驱动共存的情形；③基准依赖大量人工审核与专家判断，扩展成本高；④实验聚焦于10个领域，未必普适于更广泛的科研场景。

---

## 420. Dimensionality Reduction Meets Network Science: Sensemaking on UMAP's kNN Graph

**arXiv ID:** 2607.08746 | [PDF](https://arxiv.org/pdf/2607.08746v1)

**作者:** Duen Horng Chau `[一作]` (Apple), Dominik Moritz `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用 UMAP 在构建二维嵌入前生成的 kNN 图作为分析资源，应用 PageRank、k-core 分解和聚类系数等图算法进行数据点代表性选择、密度层次划分与局部紧密性检测。

**💡 创新点**

创新点在于将 UMAP 原本被忽略的中间 kNN 图提升为第一类分析对象，证明图算法在高维数据结构保留上能补足二维投影的失真，并提供可解释、可复用的分析结果。

**🔧 技术方法**

采用 PageRank（转移概率图算法）、k-core 分解（基于入度的迭代剔除）、聚类系数（衡量邻居互联度）三种经典图算法，直接操作 UMAP 预计算的 kNN 列表。

**📊 数据集**

在 MNIST 与 Fashion‑MNIST 两个公开图像分类基准数据集（各 60,000 张样本）上进行实验。

**📈 对比分析**

与传统基于投影的代表点选择（k‑medoids）和聚类方法（HDBSCAN）对比，PageRank 在代表性、类别平衡和下游 SVM 分类精度上表现相当甚至优于 k‑medoids；k‑core 提供了比 HDBSCAN 更细粒度的密度层级；聚类系数能识别出高度语义一致的微小簇，且与 HDBSCAN 的成员资格概率无关。

**⚠️ 局限性**

限制包括：对 kNN 参数 k 的选择仍有一定影响；k-core 只基于入度，可能无法捕捉所有结构细节；聚类系数计算复杂度较高；实验仅覆盖两类视觉数据，需进一步验证到其它领域（如单细胞测序、音频）和更大规模数据的适用性。

---

## 421. Pose-to-Biomechanics: Bridging 3D Human Pose Estimation and Biomechanical Attribute Prediction

**arXiv ID:** 2607.08725 | [PDF](https://arxiv.org/pdf/2607.08725v1)

**作者:** Ayda Eghbalian `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文训练了一种轻量化的时间 Transformer 插件 BioModule，能够将任何 3D 姿态估计器输出的 17 关节骨架序列映射到生物力学属性（包括运动学、动力学与神经肌肉层面）并与现有姿态估计框架无缝集成。

**💡 创新点**

创新点在于：①提出估计器无关的下游模块；②利用时间 Transformer 学习姿势到完整生物力学属性的隐式映射；③构建 Human3.6M 与 Human3.6Mplus 的对齐大规模数据集，实现帧级跨模监督；④系统性评估七种最先进 3D 姿态估计器对下游生物力学预测的影响。

**🔧 技术方法**

技术包括：root‑centered 17 关节坐标→51 维向量；4 层 Transformer 编码器（8 头，d=256）+ 17 个独立 MLP 预测头；分层加权多任务损失；Sinusoidal 位置编码；先在 Ground‑Truth 姿势上预训练，再在每个估计器的输出上微调。

**📊 数据集**

使用 Human3.6M 与 Human3.6Mplus 的配对数据，共 520k 帧、7 位受试者、30 个动作，利用相机标定实现两坐标系对齐，形成帧级对齐的生物力学标签数据集。

**📈 对比分析**

在冻结权重和微调两种协议下，评估了 VideoPose3D、MHFormer、D3DP、PoseMamba、MotionAGFormer、KTPFormer、TCPFormer 等七个 3D 姿态估计器。结果显示冻结时 MAE 在运动学、动力学和神经肌肉层面随估计器精度差异明显；微调后误差显著降低，证明 BioModule 能在多种估计器上泛化并显著提升生物力学预测质量。

**⚠️ 局限性**

局限性包括：仅基于受控实验室环境下的 Human3.6M 数据，未验证在野外视频或复杂运动场景中的表现；使用 17 关节骨架限制了解剖细节；依赖仿真生成的标签，受模型假设和坐标对齐精度限制；未对预测不确定性进行量化；对极端运动速度或异常姿势的鲁棒性尚未充分评估。

---

## 422. HumanForge: A Human-Centric Deepfake Video Benchmark with Multi-Agent Forgery Rationales

**arXiv ID:** 2607.08705 | [PDF](https://arxiv.org/pdf/2607.08705v1)

**作者:** Wenbo Xu `[一作]` (Sun Yat-sen University), Wei Lu `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了HumanForge大规模人类中心深伪视频基准，涵盖四类生成场景，并提出Gen2Anno多代理框架实现自动化、对比式的精细化标注。

**💡 创新点**

创新点包括整合人机/人物交互情景、对比生成意图与视觉结果的多代理注释流程、以及提供多模态、空间时间定位的omni-annotations。

**🔧 技术方法**

采用LangGraph多代理架构、Mixture-of-Experts、音视频同步、物理约束和语义一致性检测，以及多种Diffusion模型生成视频。

**📊 数据集**

利用HDTF、FFIW、FF++、DFD、SHHQ、TikTok等真实视频集作为参考，并在10+先进生成器上合成18k+视频，形成HumanForge数据集。

**📈 对比分析**

与传统检测器和大型多模态模型进行零样本评估，结果表明现有方法在HumanForge上表现不佳，展示了对零样本泛化和细粒度推理的显著挑战。

**⚠️ 局限性**

局限在于主要覆盖720p 5秒视频，缺乏更高分辨率和长时序样本，且自动化注释仍受限于代理推理的准确性和生成器多样性。

---

## 423. ProjAgent: Procedural Similarity Retrieval for Repository-Level Code Generation

**arXiv ID:** 2607.08691 | [PDF](https://arxiv.org/pdf/2607.08691v1)

**作者:** QiHong Chen `[一作]`, Iftekhar Ahmed `[通讯]` (University of California, Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于程序流程相似性的检索方法，并构建了端到端的仓库级代码生成系统ProjAgent；

**💡 创新点**

创新点在于将推理子空间投影作为衡量程序步骤相似度的特征，引入程序相似性检索与传统语义检索协同、并在代理式工作流中进行上下文扩展和静态分析反馈；

**🔧 技术方法**

核心技术包括LLM推理子空间投影、PCA去偏、基于投影相似度的候选检索、代理工具链（ls、search_func等）实现仓库探索、以及基于AST的静态一致性检查；

**📊 数据集**

使用了REPOCOD benchmark（含980个Python函数生成任务）进行评估；

**📈 对比分析**

与稠密检索、稀疏检索、同文件检索及SpecAgent对比，ProjAgent在Pass@1上取得41.14%，比最佳基线提升约6.6%；

**⚠️ 局限性**

局限性包括仅针对Python仓库、依赖单一LLM模型（Qwen2.5-Coder-14B-Instruct）、受检索预算与上下文窗口限制、且对高动态类型语言的适应性不足。

---

## 424. Multi-Sender Bayesian Persuasion with Imperfect Information

**arXiv ID:** 2607.08675 | [PDF](https://arxiv.org/pdf/2607.08675v1)

**作者:** Andra Siva Sai Teja `[一作]` (Indian Institute of Technology Hyderabad), Sujit Gujar `[通讯]` (Indian Institute of Information Technology Hyderabad)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究多发信者贝叶斯说服模型，探讨受体通过事先承诺动作策略来激励发送方在竞争中真实揭露信息，并分析在此机制下能够实现的全信息均衡与受体收益上限；

**💡 创新点**

将受体视为机制设计者，提出General Prior Admissible（GPA）及其多发信者扩展MGPA策略，证明对任意全支持先验均可实现完全信息均衡且可达到受体最优收益；

**🔧 技术方法**

运用贝叶斯说服理论、信息设计、机制设计、概率论及严格的数学证明手段；

**📊 数据集**

无实证数据集，整个工作为纯理论分析与证明；

**📈 对比分析**

无实验或数值对比，主要通过理论证明展示所提出策略在受体收益与信息揭露方面的最优性；

**⚠️ 局限性**

仅限二元状态空间、假设发送方无对手信息、受体完全理性、未考虑学习误差、噪声或非完全先验，缺乏对现实场景的稳健性与实验验证。

---

## 425. Wat3R: Underwater 3D Geometry Learning without Annotations

**arXiv ID:** 2607.08772 | [PDF](https://arxiv.org/pdf/2607.08772v1)

**作者:** Jiangwei Ren `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无需任何水下三维标注的半监督学习框架 Wat3R，利用教师-学生结构和跨视角一致性损失，将陆地训练得到的 VGGT 模型迁移到水下场景，实现了端到端的多视角三维几何重建。

**💡 创新点**

创新点包括：1）基于 Mean Teacher 的跨域半监督策略，利用合成水下渲染生成伪标注；2）设计跨视角一致性损失，通过跨视角投影补偿水下散射导致的视觉信息损失；3）引入序列级增强与静态前景掩码，提升在动态与浑浊环境下的鲁棒性；4）构建首个覆盖多种水下环境的 Water3D 基准。

**🔧 技术方法**

技术手段主要包括 VGGT 预训练模型、Mean Teacher 伪标签、合成水下图像渲染（基于物理的光照衰减模型）、跨视角投影一致性损失、序列级数据增强、静态前景掩码以及多任务损失组合。

**📊 数据集**

使用了构建的 Water3D 数据集（42 个水下场景，含相机位姿与深度），以及从公开数据集（Sea-thru、FLSea-Stereo、FLSea-VI、SQUID、SeaThru-NeRF）提取的标注样本；同时采集约 10,000 条无标注水下视频用于无监督训练。

**📈 对比分析**

在多视角深度估计、点云重建、相机姿态估计和单目深度估计等任务上，与 VGGT、Fast3r、MapAnything、DA3 等最新方法对比，Wat3R 在大多数指标上均实现显著提升（例如在 Sea-thru 任务中 RMSE 降低 13% 以上、在 3D 重建中点云准确率提升 8–13%），证明其在水下几何学习中的优越性。

**⚠️ 局限性**

主要限制在极端浑浊或高度动态的水下环境中，静态前景掩码可能过于保守导致跨视角学习信号不足；对高速移动的生物或潜水器等动态对象的几何恢复仍存在挑战。

---

