# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-12 | 今日论文总数: 698

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. U4RASD at KnowledgeGraphEval-2026: Ontology-Constrained Encoder and LLM Systems for Arabic Relation Extraction

**arXiv ID:** 2610.12742 | [PDF](https://arxiv.org/pdf/2610.12742v1)

**作者:** Ahmad Abbas `[一作]` (Arab Center for Research and Policy Studies), Fadi A. Zaraket `[通讯]` (Arab Center for Research and Policy Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在KnowledgeGraphEval 2026 Subtask 2上，作者结合TypedPair编码器与LLM（Yehia‑RE、Qwen‑RE）完成有向阿拉伯语关系抽取任务，使用实体类型预测、类型标记与本体约束实现精确预测；

**💡 创新点**

创新点在于提出双头TypedPair模型与实体类型标记的编码器方案，采用ontology‑constrained多选式LLM预测（Yehia‑RE、Qwen‑RE），并将三种模型进行投票ensemble，最终获得第一名；

**🔧 技术方法**

技术手段包括typed directional entity markers、两头（存在/关系）损失、QLoRA微调Yehia与Qwen3.5、基于prompt的多选式LLM预测、sequence‑classification head、ontology过滤与多数投票集成；

**📊 数据集**

使用的数据集为KnowledgeGraphEval 2026 Subtask 2的阿拉伯语文本，基于WojoodOntology与Wojood NER提供的21类实体类型及40个正向关系（加<NULL>）；

**📈 对比分析**

性能评估采用官方blind测试的micro‑F1，与各个模型单独比较，TypedPair‑AraModernBERT 92.69、Yehia‑RE（全词汇）92.98、Qwen‑RE‑9B‑Cls 91.94，ensemble 94.06，排名第一；

**⚠️ 局限性**

限制包括仅使用单个随机种子、依赖硬实体类型预测与exact‑match出现恢复、对稀有或语义相近关系仍表现欠佳，且未考虑实体类型不确定性和候选关系联合打分等。

---

## 2. Topology Construction Algorithms for DECT NR+ in Normal and Ultra-Dense Deployments

**arXiv ID:** 2610.12671 | [PDF](https://arxiv.org/pdf/2610.12671v1)

**作者:** Andrey Samuylov `[一作]` (Tampere University), Mikko Valkama `[通讯]` (Tampere University)

**通讯引用:** 17135 | [OpenAlex ID](https://openalex.org/A5054197935)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 DECT NR+ 多跳网络在正常与超密集部署下的拓扑构建过程，提出并验证了四种算法：负载感知的中继分配、基于滞后和黑名单的关联控制以及智能信标放置，以提升网络的连通性与稳定性。

**💡 创新点**

创新点在于将负载信息动态融入中继选择、利用滞后阈值与黑名单机制抑制短期与长期的关联波动，并通过冲突感知的信标调度减少在超密集环境中的信号干扰，实现了对超密集 DECT NR+ 网络的自适应稳定化。

**🔧 技术方法**

使用了 DECT NR+ 规范下的系统级仿真工具 WINTERsim，结合 CP‑OFDM、LBT+BEB、HARQ、MCS 等物理/MAC 层模型，并实现了四个拓扑构建算法的逻辑层集成。

**📊 数据集**

数据集为仿真生成的节点位置信息（随机分布）、传播模型（FSPL 与 3GPP 混合室内/室外模型）以及流量模型（每 10 s 平均一次 2 包的突发数据），节点数量范围从 200 到 700。

**📈 对比分析**

通过对比开启与关闭四种算法的系统级仿真结果，评估了关联率、重关联率、活跃中继比例、平均跳数、PDR、平均延迟和 95% 分位延迟等指标。结果显示：在超密集 FSPL 环境下，重关联率从 >10 /s 降至 2–3 /s，活跃中继比例从 34% 降至 15%，PDR 与延迟保持平稳并略有提升；在混合传播环境中亦能显著降低重关联峰值并提升网络稳定性。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，缺乏真实部署实验验证；算法参数（如阈值、黑名单时长）需针对不同场景手工调优；对能耗、节点电池寿命等长期运营指标的影响尚未系统评估。

---

## 3. Visual-Wi-Fi Navigation to Signal-Emitting Devices

**arXiv ID:** 2610.12733 | [PDF](https://arxiv.org/pdf/2610.12733v1)

**作者:** Lihao Wang `[一作]` (Johns Hopkins University), Renjie Zhao `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种视觉‑Wi‑Fi导航框架，使移动机器人在未知室内环境中仅凭视觉感知、里程计和从目标设备嗅探的Wi‑Fi信号就能定位并到达该设备的物理位置。

**💡 创新点**

创新点包括：① 开发了 Habitat‑Wi‑Fi 模拟器，可在连续机器人姿态下实时基于射线追踪生成Wi‑Fi信道状态信息（CSI）；② 设计了运动驱动的前后方向歧义消除与鲁棒多视角融合的Wi‑Fi源定位模块；③ 将定位模块与深度感知、动作记忆融合的强化学习策略闭环训练，允许策略主动探索以提升定位精度。

**🔧 技术方法**

技术上使用：射线追踪（Sionna）、CSI‑AoA 估计、前后方向歧义消除、RANSAC 估计、LSTM‑Actor‑Critic 强化学习、深度编码器（预训练 ResNet‑50）以及基于 Jetson AGX Orin 的硬件部署。

**📊 数据集**

主要数据集：HSSD（122 训练/42 测试室内场景），并在未见的 ReplicaCAD 与 ProcTHOR 上做零射击迁移；对比了官方 ObjectNav 方案。

**📈 对比分析**

与基准比较：在 HSSD 测试集上，改进框架在未调优时 53.02% SR，微调后 68.57% SR，SPL 44.12%；在 ReplicaCAD 和 ProcTHOR 上零射击分别达到 75.05% 和 85.22% SR；相比 ObjectNav 在实例级目标下 56.29% SR，明显优于仅基于类别的 25% SR。

**⚠️ 局限性**

局限性：① 依赖相对稠密的目标通信（需持续 UDP 流）；② 现有定位前端为解析式，未学习端到端；③ 对不同墙体材质、无线干扰的鲁棒性仍有限；④ 真实部署中需要对里程计漂移和环境变化进行更多校正。

---

## 4. Byzantine-Resilient Federated Large Language Model via Hypergraph Message Passing

**arXiv ID:** 2610.12711 | [PDF](https://arxiv.org/pdf/2610.12711v1)

**作者:** Kai Li `[一作]` (Aston University), Özgür B. Akan `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在联邦大语言模型（FedLLMs）中对抗Byzantine攻击，提出了一种基于超图消息传递图自动编码器（HMP-GAE）的自愈式聚合框架，能够在不访问本地数据的前提下识别并抑制语义失真更新。

**💡 创新点**

创新点在于将每个客户端的LoRA微调更新建模为超图节点，利用高阶超边捕获多方语义关系，并通过迭代消息传递与图解码重构联合学习，以历史检查点和关系推理实现对恶意更新的自适应过滤。

**🔧 技术方法**

技术细节包括LoRA低秩微调、超图消息传递（HMP）、图自动编码器（GAE）、重构损失（BCE）、历史一致性正则化、贝叶斯信任权重计算与聚合。

**📊 数据集**

使用的公开文本分类数据集为 AG News（四类）和 Yahoo! Answers（十类）。

**📈 对比分析**

实验对比了 Multi‑Krum、FoolsGold、FLTrust 等三种现有鲁棒聚合方法。HMP‑GAE 在 Qwen‑2.5 与 Llama‑3.2 两个大模型后端上均实现了最高准确率（提升 1–3%），最低分类语义熵（下降 20–50%）和最低困惑度（下降 10–35%），验证了其在语义一致性与性能方面的显著优势。

**⚠️ 局限性**

局限性包括：对部分类别的困惑度提升不如基线；需要额外的超图构建与消息传递计算，导致服务器端计算与存储开销上升；在超大规模客户端数量或高维更新时，超图稀疏化与可扩展性仍需进一步研究。

---

## 5. One Model to Loop Them All: Elastic Latent Recurrence for Language Models

**arXiv ID:** 2610.12582 | [PDF](https://arxiv.org/pdf/2610.12582v1)

**作者:** Faisal Ladhak `[一作]` (Nvidia), Krishna C Puvvada `[通讯]` (Nvidia)

**通讯引用:** 466 | [OpenAlex ID](https://openalex.org/A5069691915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Elastic Latent Recurrence (ELR)，一种可在单一检查点上通过循环计数实现可变推理深度的语言模型；

**💡 创新点**

创新点包括：① gated adaptive‑delta 状态更新，保持一循环端点不受影响；② 在每个优化步骤内混合循环计数并使用一循环占位；③ 对每次循环进行进度与预算条件化，提升多循环性能；

**🔧 技术方法**

使用的技术包括：共享权重的循环块、学习步长门控（β）、循环计数条件化 MLP（循环进度 τ、预算 κ）、混合深度训练（overrides）、跨循环路径重播（MoE）、最后一次 KV 缓存（last‑loop KV）等；

**📊 数据集**

在两大规模模型上训练：4B 纯稠密 GDP‑Hybrid、30B-A3B MoE 混合模型，数据来源于 Nemotron‑3 Ultra 预训练语料；

**📈 对比分析**

与同参数量非循环基线相比，一循环 ELR 仅差 0.5~1.0 分，四循环可提升 4.75~7.25 分；在 Koi‑4B 后训练中，一循环 AI‑Index 约 18 分，四循环约 29 分，速度和内存开销均可控制；

**⚠️ 局限性**

局限性：增益在第二循环后显著递减；多循环性能仍落后于固定深度专家模型；未评估循环分布重加权、动态循环选择或 MoE 后训练的效果。

---

## 6. Separating Field--Camp Alignment from Structural Balance in a Frustrated Random-Field Ising Model of Social Networks

**arXiv ID:** 2610.12578 | [PDF](https://arxiv.org/pdf/2610.12578v1)

**作者:** Natalia Romero `[一作]` (Florida Atlantic University), Natalia Romero `[通讯]` (Florida Atlantic University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5111780133)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在固定负边比例下，作者构造了同一图、同一场、同一边权的四臂对照实验，比较了独立随机分配、基于三角平衡的分配以及通过阵营标签置换仅保留三角平衡结构的分配，最终定量评估了在Frustrated Random‑Field Ising模型中，场‑阵营对齐与三角平衡结构对能量分布的影响；

**💡 创新点**

首次通过精确消除场‑阵营相关性，仅保留三角平衡结构，证明在此模型中，场‑阵营对齐而非三角平衡结构是导致能量间隙分布差异的主要原因，并给出了效应大小及其对系统尺寸、场/耦合比、平衡噪声参数等的依赖；

**🔧 技术方法**

使用经典模拟退火（Metropolis单粒子翻转）和并行退火（replica‑exchange Monte Carlo）进行采样；对能量间隙分布采用Kolmogorov‑Smirnov检验；配对Wilcoxon、Mann‑Whitney等非参数检验以及多重比较校正；并利用Gauge变换分析阐明平衡与无争议的对应关系；

**📊 数据集**

使用基于真实选民党派认同强度的场强值（四层社会接触强度的边权），并对三种真实签名网络（Epinions、Slashdot、Wikipedia）的平衡三角比例进行校准得到ν≈0.02–0.03；

**📈 对比分析**

实验采用同一网络、同一场、同一边权，独立随机分配、平衡结构分配、以及阵营置换的四臂设计；对能量间隙的Kolmogorov‑Smirnov统计量进行对照，所有密度下均显著（p<10⁻⁷），在更大系统、不同r_h、不同ν下仍保持显著；但在N=30时无显著差异，N≥80才出现显著效应，表明效应稳健但受尺寸限制；

**⚠️ 局限性**

研究仅在N≤300的有限尺寸下验证；仅匹配了50%负边比例，未覆盖现实中约20%负边率的情况；场强与边权采用模型化参数，未直接测量；并行退火验证仅对核心比较进行，未对所有结果；假设阵营标签与党派倾向完全相同；随机置换仅保留三角平衡结构，未改变更高阶环结构，可能遗漏其他结构效应。

---

## 7. GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks

**arXiv ID:** 2610.12669 | [PDF](https://arxiv.org/pdf/2610.12669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 8. Non-Prehensile Multi-Object Placement with Mobile Robots via Diffusion-Based Trajectory Planning

**arXiv ID:** 2610.12684 | [PDF](https://arxiv.org/pdf/2610.12684v1)

**作者:** Lance Pharand `[一作]` (University of Waterloo), Stephen L. Smith `[通讯]` (University of Waterloo)

**通讯引用:** 12302 | [OpenAlex ID](https://openalex.org/A5103134580)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于扩散模型的移动机器人多物体推送规划方法，并在模拟与真实机器人平台上进行了验证。

**💡 创新点**

核心创新在于“因果分解”——先对机器人轨迹进行去噪，再以当前清洁的机器人轨迹为条件预测物体轨迹，从而实现多模态物体预测并提升指导效果，同时设计了针对精细推送的多项成本函数。

**🔧 技术方法**

采用扩散概率模型（DDPM/ DDIM）、v‑parameterization、DPS后向指导、Transformer 解码器、best‑of‑N 采样与自定义成本梯度等技术。

**📊 数据集**

训练数据包含 Hugging Face 上的 Push‑T keypoints 数据集、约 500 条人类远程操控的移动机器人推送演示（2 cm 容差）以及用于硬件实验的四种全新几何物体。

**📈 对比分析**

与 Diffusion Policy、Diffusion Forcing、联合去噪模型及 CEM 基线比较，在 Push‑T 上成功率最高（+10.7%），在三种多盒实验中 BPF 提升约 21.5%（最高 30.9%），且重规划速度比联合去噪快 1.8×；在 TurtleBot 3 Burger 上实现 85% 物体落位率，远超 37.5% 的基线。

**⚠️ 局限性**

局限性包括二进制成功率仍低（<40%）、仅在使用全局位姿（上视摄像头）时可行、目标标记被遮挡时无法停止、并未在自主摄像头部署上验证；高失效率主要来自已放置物体被扰动或机器人碰撞。

---

## 9. Reference-Free Food Volume Estimation from Egocentric Stereo Video

**arXiv ID:** 2610.12735 | [PDF](https://arxiv.org/pdf/2610.12735v1)

**作者:** Vivan Kumar `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于可穿戴 egocentric 立体视频的无参量化食物体积估计方法。

**💡 创新点**

创新点在于利用设备内已知的立体标定与视觉惯性里程计实现全自动、无标注的物理尺度重建，并结合零样本立体匹配与 SAM 2 语义分割实现高精度体积估计。

**🔧 技术方法**

方法主要采用 FoundationStereo 进行零样本立体匹配、SAM 2 进行食物与盘子分割、TSDF 体素融合、RANSAC+PCA 进行盘面平面拟合，以及 Open3D 的 marching cubes 生成网格。

**📊 数据集**

实验使用在 Aria Gen2 可穿戴眼镜上录制的 5 条 30 秒视频，分别包含苹果、蜜饼、披萨片、三明治和子三明治。

**📈 对比分析**

与单目深度估计比较，立体匹配能成功生成完整网格，五种食物的平均绝对百分误差仅为 9.1%，所有估计值与真实值相差 13% 以内。

**⚠️ 局限性**

局限包括需手动完成完整扫描、对动态进食场景、手部/餐具遮挡缺乏鲁棒性，以及仅在静态食物不移动的环境下验证。

---

## 10. Mind the Temporal Gap: Evaluating Video Models for Temporal Understanding in Driving

**arXiv ID:** 2610.12575 | [PDF](https://arxiv.org/pdf/2610.12575v1)

**作者:** Shiva Sreeram `[一作]` (MIT), Daniela Rus `[通讯]` (MIT)

**通讯引用:** 68755 | [OpenAlex ID](https://openalex.org/A5066830185)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过多层探测框架，评估视频语言模型在驾驶视频中的时间推理能力，针对前进/后退、左/右、加速/减速等自我运动任务进行分析。

**💡 创新点**

创新点在于提出系统的多级提示与学习方法组合，揭示模型在提示下表现不佳但在轻量微调和特征探测下可达接近完美的运动识别，指出视觉-语言接口是时间推理的关键瓶颈。

**🔧 技术方法**

使用的技术包括标准提示、视觉提示、少样本提示、链式思考提示、LoRA轻量微调、特征提取+轻量分类器，以及CARLA仿真与少量真实驾驶视频的对比实验。

**📊 数据集**

数据集为CARLA生成的模拟驾驶视频（每类200个，6类运动）以及60个真实驾驶视频（10个/类），用于Sim与Real的评估。

**📈 对比分析**

通过对比不同提示策略与学习策略的准确率，发现提示方法常受类别偏差影响且接近随机；LoRA微调和特征探测在多数模型上实现近100%准确率，显示模型内部蕴含运动信息但提示层未能充分利用。

**⚠️ 局限性**

局限性包括实验仅覆盖模拟与小规模真实数据，未使用大规模真实驾驶数据；缺乏单帧/打乱控制，LoRA与特征高准确率可能受单帧外观信息影响；未能完全验证模型对时间推理的内在能力。

---

## 11. Verifier-Mediated Multi-LLM Coordination for Multi-Robot Path Planning

**arXiv ID:** 2610.12761 | [PDF](https://arxiv.org/pdf/2610.12761v1)

**作者:** Vivek Khatana `[一作]` (University of Illinois Urbana Champaign), Petros Voulgaris `[通讯]` (University of Nevada Reno)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种验证器介导的多LLM协同架构，用于在多机器人路径规划中将高层语义计划与执行级几何可行性分离，并通过迭代反馈实现跨机器人冲突消除。

**💡 创新点**

创新点在于：①将大语言模型与执行级几何验证器耦合，形成闭环反馈；②对比集中式与分布式LLM规划，并在此基础上引入结构化冲突反馈进行针对性重规划；③展示该框架在不同机器人平台（CARLA车辆与Crazyflie无人机）上的可迁移性。

**🔧 技术方法**

技术上使用大语言模型（LLM）进行任务级符号规划，基于图搜索（A*）进行几何轨迹生成，使用执行级验证器对机器人间距离和障碍约束进行预测冲突检测，并通过结构化消息返回给LLM实现重规划；同时采用邻接表与坐标两种环境描述方式。

**📊 数据集**

主要数据集为CARLA模拟器中的道路网络图（不同目标集合规模）以及Crazyflie无人机的仿真与实机实验环境；实验中未使用公开标注数据集，而是基于自定义仿真地图与实际机器人硬件完成任务。

**📈 对比分析**

比较方法：在CARLA中对两代理多目标导航任务，分别采用集中式单LLM、集中式多LLM、分布式两LLM三种架构，并在原始坐标和邻接表两种输入表示下评估规划时间、成功路径长度及平均规划时间；实验结果显示：①集中式在目标集合较小（|W|=2）时规划时间更短；②随着目标数增大，分布式方案的优势逐渐显现；③邻接表输入更能保证图合法性，路径长度与集中式相当，原始坐标则更灵活但易产生无效航路。

**⚠️ 局限性**

局限性包括：①对大规模机器人团队的可扩展性未充分验证；②验证器采用的是基于距离阈值的简单冲突检测，未考虑不确定性或正式安全证明；③LLM在生成符号计划时仍可能产生与地图不一致或无效顺序，需要更精细的约束引导；④实验主要在仿真环境，实机测试仅涉及单无人机导航，对多机协同的真实硬件验证仍不足。

---

## 12. REDaCT: Collapse-Free Unlearning in Large Reasoning Models

**arXiv ID:** 2610.12769 | [PDF](https://arxiv.org/pdf/2610.12769v1)

**作者:** Joshua Maccoby `[一作]` (Stevens Institute of Technology), Wendy Hui Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种推理模型（LRM）的推理时忘记框架REgenerate via Detection and Corrective Tuning (RE)，通过检测链式推理中泄露的忘记集信息并在首次泄露句子处注入优化的纠正嵌入，实现推理过程与答案的完整忘记；

**💡 创新点**

创新点在于：1）引入句级泄露检测器，精确定位推理过程中的敏感句子；2）设计离线优化的纠正嵌入，能够在推理时实时替换泄露内容而不引起推理崩塌；3）以检测-重生成循环完成推理时的无参数修改，避免传统微调方法的“忘记-重构”问题；

**🔧 技术方法**

使用技术包括RoBERTa二分类泄露检测器、纠正嵌入优化（对损失函数加入拒绝、遗忘、保留三项约束）、推理时的检测-重生成流程，以及使用LLM-as-judge评估泄漏和错误率；

**📊 数据集**

主要数据集为R-TOFU（基于虚构作者传记的推理问答数据集），对不同比例的忘记集（1%、5%、10%）进行实验，并使用DeepSeek-R1-Distill-Llama-8B和DeepSeek-R1-Distill-Qwen-7B两种LRM模型；

**📈 对比分析**

与GA、GD、KL、PO、R^2MU等现有方法比较，RE在答案遗忘（AFE）与链式推理遗忘（CFE）均达到1.0，且保持最高的模型效用（≈0.78）和零推理崩塌率；

**⚠️ 局限性**

局限性包括：仅在类似规模的两款LRM上验证，未覆盖更大模型或多模态场景；检测粒度为句子级，可能漏检子句级泄露；仅支持文本LRM，需扩展至多模态。

---

## 13. Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data

**arXiv ID:** 2610.12747 | [PDF](https://arxiv.org/pdf/2610.12747v1)

**作者:** Shuo Zhang `[一作]` (New York University), Kaan Ozbay `[通讯]` (New York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过人机交互式驾驶仿真收集了工作区行驶数据，并构建了WZDrive-Data数据集，用于预测城市工作区的自驾车轨迹。

**💡 创新点**

创新点在于结合驾驶员控制信号、注视、工作区几何以及多模态学习，提出多阶段训练的WZDrive模型，显著提升了工作区轨迹预测精度。

**🔧 技术方法**

技术包括基于CARLA-SUMO的人机交互仿真平台、VR工人交互、多模态编码器、路径预测与多阶段校准，以及对齐的姿态和控制输入。

**📊 数据集**

使用了自研的WZDrive-Data数据集，包含1183个工作区驾驶段，33名驾驶员，涉及三种工作区配置和六种情景。

**📈 对比分析**

与多种基线（规则、GRU、MTR、LAFormer等）比较，WZDrive在minADE_6和minFDE_6上分别降低7.18%和13.18%，成为目前最佳方法。

**⚠️ 局限性**

局限在于数据来源仅为模拟且受限于33名年轻驾驶员，且罕见事件（碰撞、异常停靠）样本稀缺，需进一步扩展真实世界数据和更丰富情景。

---

## 14. An Exploration of Attacks on AI Agent Systems

**arXiv ID:** 2610.12524 | [PDF](https://arxiv.org/pdf/2610.12524v1)

**作者:** Cathleen Qiao `[一作]` (University of California Los Angeles), Peiran Wang `[通讯]` (University of California Los Angeles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并展示了在单体与多体 AI 代理系统中，通过 prompt injection（包括直接注入、工具劫持、Agent Card Spoofing 等）泄露敏感信息、触发跨代理感染的攻击手段，并对不同防御层（系统提示、过滤器、注入识别器）进行实验验证。

**💡 创新点**

首次将 prompt injection 扩展至多体协作框架（AutoGen、A2A 协议）并提出 Agent Card Spoofing 攻击，揭示了代理卡选择偏好与情感倾向的安全漏洞；同时系统性对比多种防御策略的有效性，提供了针对多体系统的安全评估。

**🔧 技术方法**

使用 LangChain（包括 LangGraph、Web Browser、PALChain、HuggingFaceInjectionIdentifier）、AutoGen、A2A 协议实现的多体代理、LLM（Llama3-8b-8192、Gemini 2.0 Flash Lite、GPT‑4o、Gemini Pro）等技术进行实验。

**📊 数据集**

主要使用自定义对话、网页与文件（如 /app/.env、/private/etc/passwd、/private/etc/shadow 等）以及生成的 106 条机票预订任务作为测试数据集；未采用公开大规模数据集。

**📈 对比分析**

通过对比不同防御层（无防御、四种系统提示防御、外部过滤器）在同一攻击下的成功率，实验显示某些注入技巧（如链式思考、角色扮演）在无防御时成功率可达 80% 甚至 100%，在引入防御后虽大幅下降但仍有 10%–60% 的成功空间。

**⚠️ 局限性**

实验规模受限于本地小型设置，未覆盖真实大规模 A2A 系统；评估聚焦单一攻击类别，缺乏跨工具、跨 LLM 的普适性验证；防御策略的可迁移性和鲁棒性未在更复杂情境下验证。

---

## 15. GlucoPRISM: Protocol-Supervised Pretraining for Cross-Cohort Glucose Transfer

**arXiv ID:** 2610.12528 | [PDF](https://arxiv.org/pdf/2610.12528v1)

**作者:** Adinath Madhavrao Dukre `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 12032 | [OpenAlex ID](https://openalex.org/A5033585021)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并训练了GlucoPRISM模型，将连续血糖一天的数据分解为Trait、State和Sensor三块，并在部署时剔除Sensor块以消除测量器件的影响。

**💡 创新点**

通过“响应契约”将测量器件信息定位到Sensor块，并利用预训练时的模拟传感器视图实现无目标数据的固定投影剔除，从而显著提升跨队列迁移性能。

**🔧 技术方法**

基于GlucoFM骨干的Transformer编码器，结合自监督的多任务目标（对齐、摘要预测、设备预测、KL瓶颈）以及Sensor块的高斯瓶颈约束。

**📊 数据集**

使用514名参与者、10952个日窗口的公开CGM数据（CGMacros、Stanford、Hall、ShanghaiT2DM）以及9,303名HPP外部验证集。

**📈 对比分析**

与多种公开CGM基础模型（GluoFM、GluFormer、CGM-JEPA等）在14个任务-队列单元和12个跨队列迁移方向上对比，GlucoPRISM平均迁移AUROC提升约4.3分，并在同队列任务中保持或略优表现。

**⚠️ 局限性**

设备信息与队列混杂导致Sensor块并非纯设备信号，模型在极短时或后餐糖变化预测方面优势有限，且在进一步扩充数据集后迁移优势仍需验证。

---

## 16. Selectively Actuated Fiberbots for High-precision Minimally Invasive Surgery

**arXiv ID:** 2610.12636 | [PDF](https://arxiv.org/pdf/2610.12636v1)

**作者:** Jinshi Zhao `[一作]` (Imperial College London), Burak Temelkuran `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研发了一种长1.2 m、直径1.9 mm的电热纤维机器人，能够通过局部加热实现下端尖端精准的亚毫米位移，并与达芬奇手术系统集成在腹腔模型中实现扫描。

**💡 创新点**

创新点在于将高电阻加热元件限制在纤维下端的主动段，实现选择性加热，从而显著降低长纤维上的电能消耗，并保持机械上无耦合，满足MIS的长距离、柔性需求。

**🔧 技术方法**

采用热拉丝制造多腔聚碳酸酯纤维并嵌入金属线（铜‑不锈钢‑铜），使用光纤照明、激光微加工、射频焊接、Kapton绝缘等技术；控制采用 LabVIEW + DAC + op‑amp 放大器实现开环电压驱动。

**📊 数据集**

实验数据来自 500 组随机电压输入的位移测量、温度摄像测试、2D 路径跟踪（星形与螺旋形）以及与达芬奇手术器械集成的腹腔模型扫描。

**📈 对比分析**

与此前全长加热设计相比，选择性加热在主动段 0.1 m、被动段 0–1.0 m 时，总功率仅从 0.40 W 升至 0.75 W，降低约 89%，而全长加热设计在 1.0 m 被动段时需 6.6 W；位移精度保持在 100 µm 以内，路径跟踪误差 < 100 µm。

**⚠️ 局限性**

局限性包括使用低热变形温度聚碳酸酯导致加热功率受限，机身热量对组织安全性的关注，以及 Kapton 保护在液体环境下电绝缘可靠性不足，需要更高 Tg 材料和更稳固的绝缘包装。

---

## 17. AssemblyLens: Exploring the Role of Structural Roadmaps and Strategies of Furniture Assembly Used by Blind and Low Vision Individuals

**arXiv ID:** 2610.12717 | [PDF](https://arxiv.org/pdf/2610.12717v1)

**作者:** Bo Liu `[一作]`, Thijs Roumen `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究如何为视障与低视力用户提供可听的家具装配说明，利用技术探针将传统视觉说明转换为文本化的结构路线图和逐步操作指令。

**💡 创新点**

创新点在于：①结合高层结构路线图与详细操作指导，帮助用户在不依赖视觉的情况下先构建整体认知；②使用多模态大模型（Gemini 3 Pro）实现自动化的结构提取、连接分析和文本生成；③提出针对视障组装的设计原则与八条实用改进建议。

**🔧 技术方法**

技术手段：多模态大型语言模型（Gemini 3 Pro）实现视觉手册的结构抽取与文本转化；交互式审核界面进行人工纠错；屏幕阅读器与触觉显示的可访问输出。

**📊 数据集**

数据集：34份家具装配手册（覆盖椅子、鞋架、桌子等）来自25家制造商；以及8位视障/低视力参与者的实地组装实验数据。

**📈 对比分析**

比较方法：本研究未设置传统视觉组装的对照实验；通过观察参与者完成率、错误率、任务时长与NASA‑TLX负荷评分评估效果。结果显示所有参与者完成任务，平均组装时长约20分钟，错误率低，工作负荷相对可控，且使用路线图的参与者在第二次组装时表现更佳。

**⚠️ 局限性**

限制：①仅测试小型家具（椅子、鞋架），无法推广到大型家具；②实验过程中研究者持续在场并进行必要干预，未评估完全无帮助的情境；③依赖人工审核，自动化程度仍有限；④缺乏严谨的对照实验来量化路线图与仅步骤指令的差异；⑤手册设计标准化不足，导致转换过程易受图示差异影响。

---

## 18. Robust Langevin Sampling under State-Dependent Heavy-Tailed Noise with Applications to LLM Fine-Tuning

**arXiv ID:** 2610.12865 | [PDF](https://arxiv.org/pdf/2610.12865v1)

**作者:** M. Berk Sahin `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可接受状态相关重尾噪声的 BG-p oracle，并基于此设计了三种 Langevin Monte Carlo 方法（R‑LMC、A‑LMC、RA‑LMC），并给出在非对数凸、非全局光滑、噪声无界情形下的非渐近 Fisher 信息收敛保证；

**💡 创新点**

创新点在于首次提出能够同时处理状态依赖和重尾噪声的 BG-p oracle，结合几何中值中心化与残差裁剪的鲁棒梯度估计，以及噪声感知的自适应步长，从而实现了在更宽松假设下的 FI 保证，并在实际 LLM 细调中验证了其有效性；

**🔧 技术方法**

采用了几何中值中心化+残差裁剪的鲁棒估计、对称一般光滑假设、噪声感知自适应步长、Fisher 信息分析框架、以及相关的理论证明与实验验证；

**📊 数据集**

主要使用了合成非凸目标、LLM 细调（OpenBookQA、SLAKE）以及视觉语言问答（VLM）数据集进行实验；

**📈 对比分析**

与 SG‑LMC、标准 LMC、A‑LMC 以及多种 Bayesian fine‑tuning 基线（如 MAP、MC Dropout、Deep Ensemble、BLoB 等）进行对比，RA‑LMC 在准确率、负对数似然、Brier 分数和 ECE 等指标上表现最优或相当；

**⚠️ 局限性**

局限在于对噪声尺度和光滑参数仍需预估，理论复杂度较高，且在极端高维或大模型场景下的可扩展性尚待进一步验证。

---

## 19. Sample-Adaptive Zeroth-Order Post-Training

**arXiv ID:** 2610.12566 | [PDF](https://arxiv.org/pdf/2610.12566v1)

**作者:** Haiyuan Wan `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5008843158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 Sample‑Adaptive Zeroth‑Order Optimization（SAZO）算法，用于在不需要反向传播的前提下，基于已存在的方向差分信息动态估计并跟踪每个训练样本的重要性，从而实现更高效的查询分配；

**💡 创新点**

创新点在于利用方向差分的放缩幅度作为梯度范数估计，并通过每个样本的指数移动平均跟踪其随训练动态变化的重要性；

**🔧 技术方法**

技术主要包括：零阶梯度估计、重要性采样、指数移动平均（EMA）跟踪、均匀探索、以及理论上的方差最小化收敛分析；

**📊 数据集**

在多种模型和任务上进行了验证，包括 RoBERTa‑large（350M）在 SST‑2、SST‑5、SNLI、MNLI、RTE、TREC 等分类任务；OPT‑1.3B、Phi‑2（2.7B）和 Llama3‑8B 在多分类、单项选择和问答等八个基准任务；

**📈 对比分析**

与 MeZO、HiZOO 等现有零阶优化器相比，SAZO 在保持相同内存占用和前向推理成本的前提下，平均提高 1.1‑1.6 个百分点的下游性能，并在相同训练损失下实现高达 6× 的查询计数加速；

**⚠️ 局限性**

主要局限包括：对探索系数和 EMA 速率的依赖需要经验调优；在极大规模数据或模型中，频繁更新每个样本的重要性估计仍可能带来一定计算开销；此外，该方法仍假设梯度范数的方向差分估计足够可靠，极端噪声或梯度不连续可能影响性能。

---

## 20. Representation Alignment from a Few Paired Data via Distilled Optimal Transport

**arXiv ID:** 2610.12781 | [PDF](https://arxiv.org/pdf/2610.12781v1)

**作者:** Julien Kordes `[一作]` (Sorbonne Université), Reuben Dorent `[通讯]` (Sorbonne Université)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过把两种预训练的单模态编码器映射到锚点相对坐标空间，利用全局熵正则化的最优传输求得跨模态软对应关系，并将该对应关系蒸馏成两个轻量化、可推理的编码器，从而在仅有数百对锚点的条件下实现跨模态对齐。

**💡 创新点**

①使用锚点相对表示统一两种模态的坐标系；②通过全局熵正则化OT、双重边缘约束消除hubness，提升匹配质量；③将OT结果以软目标对比学习形式蒸馏到可推理的轻量网络，避免记忆库依赖。

**🔧 技术方法**

锚点相对坐标、熵正则化最优传输（Sinkhorn）、双边缘约束、软目标对比蒸馏、分块训练和方向异构损失。

**📊 数据集**

Flickr30k、MS‑COCO（图文对齐）、ChEBI‑20（分子‑文本对齐）、ImageNet七类零样本分类。

**📈 对比分析**

与ASIF、CSA、SUE、SOTAlign以及纯对比学习基线比较。结果显示，在相同锚点预算下，方法在Flickr30k和MS‑COCO的Recall@10上均优于其他方法，R@1与SOTAlign相当并在多数指标上领先；在ChEBI‑20检索和ImageNet零样本分类中亦取得最佳或接近最佳表现。

**⚠️ 局限性**

受预训练编码器质量限制；假设两侧样本来自同一联合分布，使用均匀边缘，未处理不平衡分布；锚点随机选取，如何选择最佳锚点仍是待解决问题；在极少量数据时性能受限，且跨模态结构差异大时最优传输难以发挥作用。

---

## 21. Spa-Bench: A Real-Robot Benchmark for Spatial Grounding in Vision-Language-Action Models

**arXiv ID:** 2610.12696 | [PDF](https://arxiv.org/pdf/2610.12696v1)

**作者:** Justin Tien-Smith `[一作]` (Imperial College London), Rolandos Alexandros Potamias `[通讯]` (Imperial College London)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5047299059)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Spa-Bench，一个在真实机器人上评估空间定位能力的基准，包含六个桌面任务和多种诊断测试；

**💡 创新点**

通过“保留‑细胞”设计隔离空间概念与对象/动作的组合，提供系统化的ID/OOD分裂与四个诊断条件，构建可复现的空间定位评测框架；

**🔧 技术方法**

在 SO‑101 机械臂上对五个主流 VLA（VLA‑0、MolmoAct2、π_0.5、GR00T‑N1.7 F‑LLM、GR00T‑N1.7 F‑FT）进行实时执行，使用真实机器人回放、直接操作对照、指令改写、跨任务对象与无操作测试；

**📊 数据集**

使用 1,200 条人类遥控演示（200 条/任务）和 2,892 条真实机器人评测回放，共 4,092 条 SO‑101 记录，涵盖 27 种可移动对象（11 种形态族）；

**📈 对比分析**

对比 ID 与 OOD 空间配置，ID 成功率约 68–73%，OOV 降至 40–52%；同一场景下直接命名成功率 81–89%，改写指令可进一步降低成功率；跨任务对象测试降低 10–20%；无动作测试表现低下，表明 VLA 在空间推理上存在显著瓶颈；

**⚠️ 局限性**

受限于固定机器人、工作空间与对象库存，场景最多仅 5 个候选物体，重置不完全像素级一致；零-shot 无动作条件；诊断条件不完全对称；结果体现完整策略而非单一架构组件，外部可扩展性受限。

---

## 22. WoodBR-Cross: A public wood-image dataset and individual-tree-disjoint benchmark for species identification

**arXiv ID:** 2610.12676 | [PDF](https://arxiv.org/pdf/2610.12676v1)

**作者:** Daniel Casanova `[一作]` (Federal University of Technology--Paraná), Pedro Luiz de Paula Filho `[通讯]` (Federal University of Technology--Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了 WoodBR-Cross 木材图像数据集（4875 张横截面图像，38 个操作级类别，20 个物种子集共 3406 张图像，涉及 303 棵树），并在图像级与树级不泄漏的交叉验证分区上对深度学习模型、冻结特征、手工特征及控制实验进行基准评测。

**💡 创新点**

创新点包括：① 将每棵树的 catalog witness code 作为唯一树 ID，公开树级别分区并防止树 ID 泄漏；② 明确区分图像级与树级评估，揭示树级泛化性能大幅低于图像级；③ 发布完整数据、分区文件、实验配置和评估脚本，方便复现和扩展。

**🔧 技术方法**

技术手段包括：预训练模型 ResNet‑50、ConvNeXt‑Tiny、ViT‑B/16 的微调；冻结 DINOv2 视觉特征与逻辑回归/ kNN；手工 HSV+LBP+GLCM 与 SVC/随机森林；边缘遮罩与随机标签控制实验；AdamW、数据增强、宏 F1、Top‑3 准确率、ECE、Brier 等评估指标；t‑SNE、Grad‑CAM 与嵌入相似度分析；树级学习曲线。

**📊 数据集**

使用的数据集：WoodBR-Cross（4875 张图像，38 级别，303 棵树），其中 20 个物种子集（3406 张图像，214 棵树）为主要实验集；参考文献中提及的其他巴西木材图像数据集（FSD、FSD‑M、LPF‑UFP 等）仅作背景介绍。

**📈 对比分析**

比较方法：采用三折外交叉验证，内部验证选择最佳学习率/权重衰减；对图像级（A、B1）和树级（B2）分别计算宏 F1；对模型、冻结特征和控制实验做平均和标准差报告。结果显示：在图像级 B1 任务中 ConvNeXt‑Tiny 达到宏 F1 ≈ 0.989；在树级 B2 任务中，ConvNeXt‑Tiny 仅能获得宏 F1 ≈ 0.862，冻结 DINOv2 的树级宏 F1 0.900 超过微调模型；边缘遮罩控制仍保持宏 F1 ≈ 0.843。可见，树级泛化显著低于图像级。

**⚠️ 局限性**

局限性：① 树级评估仅覆盖 Fabaceae 20 个物种，无法推广到其他科；② 缺乏物理尺度、采集条件和光照等元数据，难以复现拍摄过程；③ 仅使用 303 棵树，树的多样性有限；④ 未对跨数据集泛化进行实验；⑤ 控制实验未完全隔离解剖结构与背景/光照信息；⑥ 仅采用预定义的预训练模型与有限的超参网格，可能未挖掘最优配置。

---

## 23. How Do Hardware Faults Manifest Differently in LLM Agents?

**arXiv ID:** 2610.12756 | [PDF](https://arxiv.org/pdf/2610.12756v1)

**作者:** Aiden Rubey `[一作]` (Oregon State University), Sanghyun Hong `[通讯]` (Oregon State University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估大语言模型代理在随机硬件故障（两位权重翻转）下的鲁棒性，覆盖6,814次执行并追踪故障传播；

**💡 创新点**

首次将硬件故障从模型层面扩展到闭环代理执行，揭示代理可从错误输出恢复、导致额外计算开销、出现细微语义错误以及安全/对齐行为改变的多维影响；

**🔧 技术方法**

采用随机两位权重翻转注入、Qwen系列模型、三大代理基准（HomeBench、AgentDojo、Agentic Misalignment）以及任务成功率、执行成本、后果行为等多维指标评估；

**📊 数据集**

三大代理基准，共218个任务，包含8–13个干净执行和15–25个故障注入，合计6,814次实验；

**📈 对比分析**

与干净运行对比，任务完成率在部分模型下降约1–4个百分点，输出token平均增长51%（最高可达38%总体增量），malformed工具调用可恢复3–26%；安全/对齐指标在特定模型/场景下亦有显著波动；

**⚠️ 局限性**

仅针对两位权重翻转、有限模型与基准；缺乏对不同故障模式、硬件平台及更广泛多轮情境的深入统计与验证，泛化性待进一步验证。

---

## 24. Behind-the-Meter Photovoltaic Visibility from Net Smart Meters: Deployment Regimes and Operational Insights

**arXiv ID:** 2610.12527 | [PDF](https://arxiv.org/pdf/2610.12527v1)

**作者:** Jack Zheng `[一作]` (Monash University), Hao Wang `[通讯]` (Monash University)

**通讯引用:** 27097 | [OpenAlex ID](https://openalex.org/A5100446064)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究利用深度学习模型对住宅光伏发电进行反向计量（PV disaggregation），并在配电网运营背景下系统评估不同部署情景下的误差来源、迁移学习效果及上下文窗口长度对性能的影响。

**💡 创新点**

①提出四种部署情景（S1–S4）区分客户熟悉度与时间外推；②量化部署情景对误差的主导作用，发现其比模型选择更重要；③给出跨域迁移学习细节，证明仅用少量目标户即能显著降低误差；④系统性分析上下文窗口长度在已知与未知客户中的不同效应。

**🔧 技术方法**

使用六种深度序列模型（CNN‑BiLSTM、TCN、Temporal Fusion Transformer、TimesNet、Mamba、PatchTST）、迁移学习（fine‑tuning）、目标域归一化、光伏天气特征、非负输出约束、cosine学习率调度和早停等技术。

**📊 数据集**

Ausgrid Solar Home Electricity Dataset（300户澳大利亚悉尼住宅，30 分钟分辨率）作为主数据集；Pecan Street Dataset（17户美国德州住宅，30 分钟分辨率）用于跨域适配。

**📈 对比分析**

在固定48小时上下文、相同训练合同下，对六种模型在四种部署情景（S1–S4）进行 MAE 评估；结果显示：新客户情景 S4 的平均日 MAE 约为已知客户情景 S2 的两倍；部署情景解释了 78.6% 的误差方差，模型选择仅 18%；对 Pecan Street 进行微调（k=5户）即可将误差下降 28–36%；长上下文对已知客户 S2 有显著提升，S4 的提升不稳定。

**⚠️ 局限性**

仅在离线非因果环境下评估，未考虑实时天气预测、数据缺失、噪声或断电等实际情况；迁移学习仅用 5 户目标样本，规模有限；实验仅覆盖六种模型，缺乏对更多模型和更大目标域的验证；数据集已清洗，未评估对噪声和异常的鲁棒性。

---

## 25. Activating the Annotator: Operationalising Lived Experience in Data Annotation

**arXiv ID:** 2610.12701 | [PDF](https://arxiv.org/pdf/2610.12701v1)

**作者:** Sheza Munir `[一作]` (University of Toronto), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并评估了一种前置式活化生活经验的注释阶段（axle），通过情境情节、轴导向访谈和微叙事合成来激活而非固定注释者的身份性，并比较其对身份共鸣、伤害评分及标签多样性的影响。

**💡 创新点**

创新点在于将情境情节与多维生活经验轴结合的访谈机制与模型生成的微叙事合成作为预标注步骤，使得注释者在标注前主动激活自身位置性；实验结果表明该方法提升身份共鸣、伤害敏感度且增加标签熵，而非收敛到单一共识。

**🔧 技术方法**

使用的技术包括：对话式人工智能（Qwen2.5‑7B‑Instruct 与 Qwen3‑32B）生成访谈和微叙事；统计分析采用 Mann‑Whitney、Wilcoxon、Krippendorff’s α、熵计算等；以及基于多轴生活经验的访谈脚本和可视化微叙事面板。

**📊 数据集**

使用的数据集为30条人工合成的社交媒体短贴，满足四项严格准则（表面可否认、体验差异、话语嵌入、无直接贬义），按移民、宗教和性别/性取向三主题共10条/场景，供104名注释者标注。

**📈 对比分析**

实验采用 2×2 对照设计（是否激活 + 是否有相关生活经验筛选），通过显著性检验比较两组。结果显示：激活组身份共鸣平均提升 0.41 分，伤害评分提升 0.33 分，标签熵提升 0.17 比特；与对照组相比，激活组的标签多样性显著增大，而一致性（Krippendorff’s α）下降，符合观点主义目标。

**⚠️ 局限性**

限制包括：样本量有限且不平衡（A 场景样本多）；仅使用人工合成的极端案例，难以推广至自然语料；受试者仅为北美英语流利者，缺乏多语言与跨文化验证；微叙事由模型生成，可能引入模型偏差；情绪负担与需求效应未完全剔除。

---

## 26. TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers

**arXiv ID:** 2610.12474 | [PDF](https://arxiv.org/pdf/2610.12474v1)

**作者:** Xiaoqin Feng `[一作]` (Wyze Inc.), Lin Chen `[通讯]` (Wyze Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不需要再训练的前训练量化框架下，对千亿参数的语音扩散变换器进行4位量化，并通过任务匹配校准与损失引导选择和结构对齐的混合精度调度恢复语音音色

**💡 创新点**

提出了TTS‑Q方法，将量化误差定位到晚期解码器前馈层，并通过预算平衡的后向-前馈混合精度分配显著恢复语音识别与说话人相似度

**🔧 技术方法**

使用FlatQuant进行重构式PTQ，结合任务镜像候选构造、损失引导选择、后向-前馈激活精度调度等技术

**📊 数据集**

在Seed‑TTS‑eval数据集（含中文和英文语料）以及外部校准池上进行实验

**📈 对比分析**

与RTN、GPTQ、SmoothQuant、AWQ、Q‑DiT、QuaRot等多种量化基线相比，TTS‑Q在保持语义准确率的同时，显著提升说话人相似度（SIM提升约0.004–0.005点，保持CER/WER增幅≤0.1点）

**⚠️ 局限性**

限制包括：缺乏感官听力评估、校准池对未知任务分布的鲁棒性未验证、混合精度分配未全局优化、仅在单一硬件上评估部署性能、适用范围受限于单一TTS扩散模型和语言

---

## 27. perturber: A Stateless Service for Semantics-Preserving Text Perturbations

**arXiv ID:** 2610.12667 | [PDF](https://arxiv.org/pdf/2610.12667v1)

**作者:** Khoi-Nguyen Tran `[一作]` `[通讯]` (IBM Research), Khoi-Nguyen Tran (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个无状态的服务和Python库，能在每次调用时对文本应用单一、语义保持的扰动，并提供了包含68种扰动的完整目录，供 LLM 鲁棒性研究使用。

**💡 创新点**

创新点包括：① 单一无状态原语设计，允许调用方通过链式调用实现任意组合；② 第一次方扰动目录并可插拔外部兼容层，保证不同研究可共享相同扰动；③ 对确定性和可复现性做了严格约束（种子重置、无时钟随机）；④ 提供统一的Python库、CLI和HTTP接口，跨语言可直接调用；⑤ 在工具层面评估意义保持率和性能，发现并修复缺陷。

**🔧 技术方法**

技术手段：纯函数原语、种子重置与参数化扰动、Python 注册表驱动目录、CLI与HTTP服务包装、OpenAI兼容端点实现模型后端扰动、LLM 判别与编辑距离评估。

**📊 数据集**

用于验证的语料主要是六个手工挑选的输入（含 JSON、文本等），以及 ARC‑Challenge 示例，未使用大规模公开数据集进行全面评估。

**📈 对比分析**

通过在相同调用下比较库、CLI、HTTP 三个接口，验证字节级一致性；确定性层平均调用时间为 23.8 微秒，所有扰动族均低于 100 微秒；意义保持率通过 LLM 判别与编辑距离两维度评估，绝大多数扰动保持意义并能导致模型答案翻转。

**⚠️ 局限性**

局限性包括：① 各扰动族对“意义保持”的阈值不同，评估结果受限于小规模语料；② 模型后端扰动是非确定性、最佳努力层，未纳入性能保证；③ 部分扰动被标记为 out‑of‑scope，不能用于鲁棒性基准；④ 兼容层仅用于对比，不保证与外部引擎的完全一致；⑤ 评估仅针对工具本身，未给出完整的模型鲁棒性测量。

---

## 28. Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance

**arXiv ID:** 2610.12493 | [PDF](https://arxiv.org/pdf/2610.12493v1)

**作者:** Marco Zullich `[一作]` (TU Delft), Steffen Seitz `[通讯]` (TU Dresden)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对特征重要性解释的可信度度量方法进行了系统评估，指出其主要缺陷并提出改进方向。

**💡 创新点**

提出了对现有可信度评估方法的五大缺陷及四条未来研究路线，强调模型校准、非预测类别、回归任务及度量与目标一致性。

**🔧 技术方法**

主要采用文献综述、理论分析和案例对比，讨论了基于扰动、相关系数、面积等传统度量。

**📊 数据集**

未使用特定数据集；讨论了在图像（如HAM10000）和文本等常见数据上的实例。

**📈 对比分析**

与现有的ROAR、ROAD、F‑fidelity等方法进行对比，指出其在OOD、基线不确定性等方面的不足；无实验性性能结果。

**⚠️ 局限性**

局限性包括仅聚焦特征重要性解释，未覆盖概念、对抗等其他XAI任务；缺乏实验验证；基线选择问题未给出解决方案。

---

## 29. Leakage-Resistant Evaluation of AI-Generated Database Specifications in Specification-Driven Migration

**arXiv ID:** 2610.12778 | [PDF](https://arxiv.org/pdf/2610.12778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 30. TerraAgent: Demonstration-Grounded Multi-Agent Reasoning for Long-Context Remote Sensing Visual Question Answering

**arXiv ID:** 2610.12517 | [PDF](https://arxiv.org/pdf/2610.12517v1)

**作者:** Jingwen Huang `[一作]` (Yunnan University of Finance and Economics), Keyi Luo `[通讯]` (Yunnan University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个无训练的遥感视觉问答框架 TerraAgent，提升模型对图像证据的依赖与可信度。

**💡 创新点**

创新点在于三大无训练模块 GDR、MACE、CVA 的组合，实现了尺度匹配的演示检索、证据驱动的多代理推理和对抗性可视化审计。

**🔧 技术方法**

使用了分桶检索与结构化场景图匹配、工具路由与证据账本、以及对抗性扰动的可视化依赖评分等技术。

**📊 数据集**

在公开的 RSVQA‑LR/HR、VRSBench、LHRS‑Bench 等遥感 VQA 基准上进行评测。

**📈 对比分析**

与相同基线冻结模型对比，平均准确率提升 1.6–2.4 点，视觉依赖分数提升 0.21，先验答案率下降 45% 以上，幻觉率降低约 50%，总体耗时约为单前向的 2.6 倍。

**⚠️ 局限性**

局限性包括对预训练知识的依赖，无法处理雷达/激光等非光学模态，计数精度受检测器限制，以及评估对抗扰动的覆盖面不足。

---

## 31. Feature Competition and Its Representational Origins in Vision Language Models: Evidence from the Stroop Paradigm

**arXiv ID:** 2610.12650 | [PDF](https://arxiv.org/pdf/2610.12650v1)

**作者:** Nurbüke Teker `[一作]` (Technical University of Munich), Shuchen Wu `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型在文字与颜色冲突下的行为，使用斯特鲁普范例构建基准并评估多种模型。

**💡 创新点**

提出将经典心理学斯特鲁普任务迁移到多模态模型的框架，并揭示文字特征在嵌入空间中的显著优势及可操控性。

**🔧 技术方法**

采用对比学习的CLIP、SigLIP-2，以及生成式VLM如BLIP-2、LLaVA、Kosmos-2 等；并用RSA、chunk‑based interpretability 与 embedding steering 技术分析表示。

**📊 数据集**

使用自制的100张文字‑颜色冲突图像、扩展的23,338张样本，以及 FLUX 生成的自然冲突图像作为实验数据。

**📈 对比分析**

对比对齐与生成式VLM，结果显示无论模型大小与微调，文字优先倾向普遍存在；对比嵌入空间可调时，文字方向可成功重定向而颜色方向效果差。

**⚠️ 局限性**

仅在 CLIP 上进行 embedding 干预，未覆盖其他模型；生成式回复归类方法可能遗漏复杂答案；颜色表示可能被低估；实验局限于静态文字‑颜色冲突，未涵盖更复杂场景。

---

## 32. Gamma: Grounded Multi-Agent Memory with Mechanical Audit for Memory-Intensive Manipulation

**arXiv ID:** 2610.12793 | [PDF](https://arxiv.org/pdf/2610.12793v1)

**作者:** Yudong Liu `[一作]` (Duke University), Hai Helen Li `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于多代理、可验证文本记忆的系统，将视觉信息转换为符号子目标，从而解决 VLA 策略在记忆任务中的失效。

**💡 创新点**

创新点：①将记忆阶段拆解为检测器、写入器、可验证文本记忆和推理器的协作多代理体系；②引入提议–验证握手机制，使每条记忆内容都可通过机械证据进行检验；③使用可验证的文本记忆替代传统像素记忆，提升可解释性与鲁棒性。

**🔧 技术方法**

使用技术：SAM‑3 目标检测、Qwen3.5‑9B（以及 0.8B）VLM 的 LoRA 微调、文本写入与推理、提议–验证契约、自动化训练语料生成与评估。

**📊 数据集**

数据集：RoboMME（16 个记忆任务）作为主要评测集；RoboMemArena（26 个厨房记忆任务）用于迁移测试。

**📈 对比分析**

对比方法：在 RoboMME 上平均成功率 66.0，逼近 84.1 的特权子目标上限；相比最佳无特权方法提升 21.5；在多项任务中达到或逼近上限；在 RoboMemArena 上任务成功率 38.8，子任务成功率 59.0，超过之前的记忆 VLA 与关键帧方法。

**⚠️ 局限性**

局限性：仅在仿真环境验证；运行时需多次 VLM 推理，算力开销大；依赖固定检测器与执行器；未在真实机器人上测试；部分任务仍受空间定位误差或方向判断困难影响。

---

## 33. MindVLM: Neural-Grounded Visual Captioning via Subject-Aware Semantic Evidence Selection

**arXiv ID:** 2610.12673 | [PDF](https://arxiv.org/pdf/2610.12673v1)

**作者:** Zixiang Yin `[一作]` (Tulane University), Zhengming Ding `[通讯]` (Tulane University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出MindVLM框架，通过将fMRI信号映射到视觉VLM的空间并选择神经支持的视觉标记，再由冻结的VLM生成个性化说明；

**💡 创新点**

创新点在于将脑解码任务改为神经引导的视觉证据选择，消除对文本监督的依赖，解决NSD数据的语义损失和部分可观测性问题，提供可解释的个体化感知映射；

**🔧 技术方法**

采用ViT结构的脑编码器、CLIP‑style对比学习、基于相似度的标记筛选、冻结的多模态VLM（LLaVA、Qwen2.5‑VL、Gemma3）生成文本；

**📊 数据集**

使用Natural Scenes Dataset（NSD）中的fMRI‑图像配对（约36k训练样本、4k测试样本，4名受试者）；

**📈 对比分析**

与基线VLM和多种脑-文字解码方法对比，MindVLM在大多数n‑gram与语义指标（BLEU、ROUGE、METEOR、CIDEr、SPICE）上均超越基线，并在无文本监督下与文本监督模型保持竞争力，唯一在BLEU‑2上略逊；

**⚠️ 局限性**

局限性包括对图像证据的依赖导致在语义匹配不足时表现受限，部分可观测性和语义损失仍未完全消除，文本精确度（词汇重合）与文本监督模型相比存在差距，且目前仅在NSD和有限受试者上验证，缺乏跨数据集通用性。

---

## 34. CacheScope: Exact-Compute Allocation for Training-Free Diffusion Acceleration

**arXiv ID:** 2610.12530 | [PDF](https://arxiv.org/pdf/2610.12530v1)

**作者:** Qinhan Lyu `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2417 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种训练无关的特征重用加速方法CacheScope，通过在线测量轨迹归一化的敏感度来决定在哪些步骤使用完整的扩散推理，提升推理速度与图像质量。

**💡 创新点**

创新点在于识别并消除原始探针变动中轨迹位移与网络响应混淆的干扰，利用相对变动比例（Δy/Δx）作为在线敏感度信号，从而实现更精准的计算分配，而无需额外训练或离线搜索。

**🔧 技术方法**

核心技术包括：浅层探针（前向单一Transformer块）提取输入/输出变动；轨迹归一化敏感度计算；累计阈值触发全量计算；在保持原有一阶重构器的前提下，在线分配额外的精确推理步骤。

**📊 数据集**

在FLUX.1‑dev、Wan2.1‑T2V‑1.3B和Open‑Sora 1.2三大扩散Transformer上评估，使用DrawBench与VBench的Prompt集进行测试。

**📈 对比分析**

与DiCache、TeaCache、MagCache、EasyCache、TaylorSeer等现有方法比较，CacheScope在相同额外计算预算下实现最高速度加速（最高3.57×）并显著降低LPIPS（最高70.1%），同时保持或提升PSNR/SSIM。

**⚠️ 局限性**

局限性包括：仅使用单个浅层探针估计敏感度，未考虑多探针或高阶轨迹信息，且在极端加速场景下对不同网络架构的通用性仍需进一步验证。

---

## 35. DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models

**arXiv ID:** 2610.12572 | [PDF](https://arxiv.org/pdf/2610.12572v1)

**作者:** Zhenglin Wan `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 191227 | [OpenAlex ID](https://openalex.org/A5100437036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的扩散模型强化学习微调方法DiffusionART，该方法通过先对每个提示的端点奖励进行聚合，生成统一的奖励提升方向，并在逆向采样状态上进行两步更新，从而实现无EMA、无任务专门调参的稳定高效训练；

**💡 创新点**

其创新点在于消除NFT式前向强化学习中隐含的正负一致性力，通过“聚合先导”奖励倾斜直接在逆向轨迹上施加奖励提升方向，并加入自适应漂移恢复和信赖域遮蔽机制，既无需EMA也无需手工调节恢复强度；

**🔧 技术方法**

技术上结合了正负分布加权、协方差导向奖励聚合、两步MSE更新、ODE-to-SDE KL代理、信赖域遮蔽以及对比学习奖励模型（单标量与双标量）等；

**📊 数据集**

实验涵盖了文本到图像（SD3.5-M、FLUX.1-dev）、文本/图像到视频（HunyuanVideo-13B、Wan2.1-I2V-14B）以及多种奖励评估（GenEval、OCR、PickScore、CLIPScore、VideoReward、VQ、MQ等）数据集；

**📈 对比分析**

与Flow-GRPO、Flow-GRPO-Fast、DanceGRPO、Flow-DPPO以及DiffusionNFT（含或不含EMA）等基线比较，DiffusionART在单/多奖励训练中获得与或超过基线的奖励提升，训练步骤与GPU时长均显著缩短（例如视频VQ、MQ提升率分别达到31.4%和57.4%，GPU时长比DiffusionNFT缩短约22倍）；

**⚠️ 局限性**

局限性主要体现在：1）仍主要验证于视觉生成任务，对其他模态的泛化尚待评估；2）虽然避免了EMA调参，但仍需设定信赖域阈值和两步更新的分配比例；3）在极大模型或超长序列下的数值稳定性和效率还有进一步提升空间。

---

## 36. CLEAT: Learning Value-Guided Clarification for Interactive Agents

**arXiv ID:** 2610.12655 | [PDF](https://arxiv.org/pdf/2610.12655v1)

**作者:** Jiatan Huang `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过价值导向澄清实现交互式意图对齐的框架，联合训练澄清价值控制器与任务策略；

**💡 创新点**

创新点在于使用残差查询价值（RQV）配对连续体估计澄清收益，并在训练过程中动态更新价值监督以匹配策略改进；

**🔧 技术方法**

采用Group Relative Policy Optimization、RQV估计、价值引导采样分配、Horvitz–Thompson校正的策略梯度以及在线共训练等技术；

**📊 数据集**

在四大交互基准上评估：UserGym（Travel、Turtle、Intention、Telepathy）、τ^2‑Bench（Airline、Retail、Telecom）、IN3（欠缺说明）与AskBench（AskMind、AskOverconfidence）；

**📈 对比分析**

与提示、阈值、信息增益、监督学习、GRPO等多组基线对比，平均分数在所有十个设置中最高，显著提升（如Travel +13.2pp、IN3 +6.5pp），控制器更新与价值引导采样是主要贡献；

**⚠️ 局限性**

局限性包括依赖预定义的澄清目标集合与问题生成器；需要在同一历史和用户目标下进行配对连续体，难以在实时用户交互中实现；实验仅在模拟用户环境，真实用户效果尚未验证。

---

## 37. MIRAGE: Metric Inference and Reliability-Aware Gaussian Estimation from Feed-Forward Vision

**arXiv ID:** 2610.12568 | [PDF](https://arxiv.org/pdf/2610.12568v1)

**作者:** Xinzhe Wang `[一作]` (Shanghai Jiao Tong University), Mulin Yu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 763 | [OpenAlex ID](https://openalex.org/A5049833635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MIRAGE框架，结合LiDAR–IMU运动与视觉先验实现在线、离线高质量3DGaussian重建与准确轨迹；

**💡 创新点**

关键创新是可靠性门控的视觉-激光融合：通过对应支持和跨模态一致性动态调节视觉姿态修正，同时在固定姿态下用激光深度与图像残差实现结构化Gaussian优化；

**🔧 技术方法**

使用LiDAR–IMU里程计、视觉稠密对应、基于Huber的优化、PnP + depth一致性、循环闭环门控、三角化Gaussian、点对面激光约束以及离线的点云与视觉联合Bundle Adjustment；

**📊 数据集**

在Oxford Spires、M2DGR、INS三大公开数据集上进行实验，并构建新长序列室内RGB‑LiDAR‑IMU基准（10序列，71,536帧，1.15 km），使用MIRAGE生成6‑DoF参考轨迹；

**📈 对比分析**

与多种基线（LiDAR‑IMU单模、LiDAR‑IMU‑视觉SLAM、COLMAP、Gaussian‑LIC2、在线离线3DGS）比较，MIRAGE在轨迹上平均ATE降至4.22 cm（比EllipseLIO低43.7%），渲染PSNR平均提升3.27 dB；在新基准上，MIRAGE生成的轨迹作为标准，其他视觉方法在Sim(3)对齐后的ATE多在1–3 m，渲染PSNR最高约22.8 dB；

**⚠️ 局限性**

局限包括：需要同步LiDAR‑IMU与视觉硬件，无法完全替代视觉在缺乏激光覆盖的区域；可靠性门控仍依赖手工设定阈值，长时间高动态场景中激光稀疏可能导致修正失效；

---

## 38. Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller

**arXiv ID:** 2610.12644 | [PDF](https://arxiv.org/pdf/2610.12644v1)

**作者:** Tahmid Imtiaz Imu `[一作]` (Kennesaw State University), Honghui Xu `[通讯]` (Kennesaw State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种名为Cancerous Backdoor Attack的LLM后门攻击，利用可见的无害触发词与隐藏的阶段令牌，在推理过程中按触发曝光量逐步改变模型的推理质量，从正确推理逐步过渡到错误推理与答案错误；

**💡 创新点**

创新点在于引入渐进式多阶段激活机制，而非传统单点激活；通过阶段控制器计数触发并插入对应隐藏令牌，使后门行为随触发累计逐步演化；实现了推理行为的分层、可控改变，首次展示多阶段后门在LLM中的可行性；

**🔧 技术方法**

技术上扩展 tokenizer 添加四个隐藏阶段令牌，构造阶段对应的训练样本（带触发词），联合 fine‑tune（QLoRA/LoRA、4‑bit NF4）得到单一受控模型；推理时用阶段控制器计数触发并在输入中插入相应隐藏令牌；采用标准的 ASR 指标评估各阶段的攻击成功率；

**📊 数据集**

使用了三大推理任务数据集 EntailmentBank、PrOntoQA 和 GSM8K 进行训练与评估；在零样本迁移实验中还对 Proofwriter（PW）等任务进行评测；

**📈 对比分析**

与 BadChain、ShadowCoT、DecepChain、MirageBD 等前沿推理后门进行对比，Cancerous 在 S1–S4 阶段的 ASR 均保持 70%–99% 以上，终端阶段成功率与最强基线持平或更好；与 ONION、P2P、CoS、Critical‑CoT、W2S‑Unlearning 等五种防御评估，攻击在各阶段仍保持 60%–90% 的激活，表明防御效果有限；

**⚠️ 局限性**

局限性包括：对更复杂或结构差异较大的未见任务迁移性能下降；需要在模型训练阶段植入隐藏阶段令牌，无法在部署后动态添加；对触发词数量和阈值设定敏感；当前方法依赖 PEFT 细调，可能受限于模型容量和训练成本。

---

## 39. Know What You Don't Know: Agentic Data Synthesis by Evolving Generation Space for Closed-Loop Learning

**arXiv ID:** 2610.12593 | [PDF](https://arxiv.org/pdf/2610.12593v1)

**作者:** Wei Yang `[一作]` (University of Southern California), Jun Huan `[通讯]` (AWS AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了EvoSyn，一个基于多代理的闭环自我进化代理式数据合成框架，利用学习者反馈动态调整合成空间与资源分配，以生成可执行、可验证且学习价值高的交互经验。

**💡 创新点**

突破了传统反馈驱动合成的“反馈到决策”缺口，既支持在已有因子空间内重新分配合成努力，又通过反馈识别并扩展新的可控交互因子，实现合成空间的自适应演进。

**🔧 技术方法**

采用环境驱动合成、验证器、反射器、设计器/规划器/生成器/校验器等多代理协作机制；结合强化学习与SFT训练的任务代理；通过反射器将学习者性能、覆盖率与失败诊断映射回合成空间。

**📊 数据集**

在两个可执行环境基准上进行实验：τ^2-Bench和CAR-Bench。

**📈 对比分析**

与多类基线（AgentSkills、EnvScaler、KlearAgentForge、AutoPlay等）和不同模型规模（Qwen3-8B/30B/235B、Claude Opus 4.5）对比，EvoSyn在多轮学习后平均任务成功率提升约13.3点，且在所有测评场景均优于竞争方法。

**⚠️ 局限性**

仍受限于需大量环境交互和多代理协作；合成空间扩展受可解释、有限值因子约束；对极端复杂或非结构化任务的泛化能力尚未充分验证。

---

## 40. An Input-Dependent Readout Phase for Resonant Sequence Models

**arXiv ID:** 2610.12496 | [PDF](https://arxiv.org/pdf/2610.12496v1)

**作者:** Shivesh Rane `[一作]` `[通讯]`, Shivesh Rane

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在ResonatorLM模型中引入输入相关的读出相位，使得只在读取阶段实现输入依赖，保持FFT训练路径不变。

**💡 创新点**

创新点在于仅在读出阶段添加输入依赖的相位，而非改变状态转移，既保持固定卷积又提升长序列记忆。

**🔧 技术方法**

使用的技术包括复振荡器状态机、固定/可变读出相位、FFT卷积、线性解码器与tanh非线性。

**📊 数据集**

实验数据集为WikiText-2字符级语言模型。

**📈 对比分析**

与原始ResonatorLM和Transformer对比，训练长度相同保持性能，超出训练长度时性能显著提升且未崩溃。

**⚠️ 局限性**

局限在于未测试迁移数据集、速度基准，且对关联记忆任务的表现不稳定，仅在部分种子下有效。

---

## 41. Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies

**arXiv ID:** 2610.12861 | [PDF](https://arxiv.org/pdf/2610.12861v1)

**作者:** Weiwei Chen `[一作]` (Hohai University), Cheng Deng `[通讯]` (Hohai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究长尾任务分布对视觉语言动作策略的影响，提出Evidence‑Guided Counterfactual Learning（ECL）以提升稀缺任务的表现。

**💡 创新点**

创新点在于结合指令相关和动作相关注意力共同定位任务特定视觉证据，构造对抗性样本并按任务频率加权监督，抑制对头任务动作偏倚。

**🔧 技术方法**

采用对抗性学习、counterfactual 监督、视觉注意力融合、任务频率加权，以及Transformer‑based VLA 模型。

**📊 数据集**

使用 LIBERO‑Core（FULL 与 LT）、AGILEX PIPER 与 xArm6 的真实世界六任务数据集，以及 Long‑Tail Context (LTC) 场景分布。

**📈 对比分析**

与基线 FULL、LT、APA 等方法比较，在十任务模拟和真实平台上，ECL 将宏观成功率从约 40% 提升至 53%（MiniVLA）和从 41% 提升至 47%（π_0.5），尾任务成功率显著提升。

**⚠️ 局限性**

局限性在于训练时需要额外的前向传播和对抗样本生成，导致训练成本上升；在固定计算预算下，可能无法同时最优化头尾任务表现。

---

## 42. Neural expectation-maximization and meta-models

**arXiv ID:** 2610.12741 | [PDF](https://arxiv.org/pdf/2610.12741v1)

**作者:** Zhuolin Fu `[一作]` `[通讯]`, Zhuolin Fu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文将 Transformer 视为在动态结构的马尔可夫随机场上执行最大产品 Loopy Belief Propagation 的近似，并将其层深视为 Expectation‑Maximization 的迭代，提出了统一的 meta‑model 框架，解释了自注意力、Mixture‑of‑Experts、门控单元、LSTM 与选择性状态空间模型等模块的本质。

**💡 创新点**

创新点在于：① 将自注意力与动态 MRF 的 max‑product BP 对应；② 通过层级参数更新将 Transformer 的深度映射为 EM 的 E‑step 与 M‑step；③ 提出 meta‑model 设计范式，将多种已知机制归入输入条件生成参数的框架；④ 通过实验检验聚类结构和 Meta‑Linear 的有效性。

**🔧 技术方法**

技术方法包括：图模型推导、Loopy Belief Propagation 与 EM 的对应分析、输入条件参数生成（meta‑model）以及基于聚类的结构检验；实验使用 BERT、RoBERTa、DistilBERT、BGE‑M3、Qwen3‑Embedding 等预训练模型，和合成的多模态 GMM 数据。

**📊 数据集**

使用的数据集包括：Twitter（X）推文、电影评论文本、以及多类（M=1,2,4,8）合成 2D Gaussian Mixture 数据，用于评估聚类结构与分类性能。

**📈 对比分析**

比较方法：① 通过最后一层特征的余弦相似度做单链接聚类并绘制 dendrogram，检验 Transformer 产生的 MRF 结构；② 在 GMM 任务中将 Linear 与 Meta‑Linear（不同非线性门控）在训练、验证、测试集上对比准确率；实验显示 Meta‑Linear 在多模态情形下显著优于普通 Linear，且近似于 Ground‑truth MoE 的上限。

**⚠️ 局限性**

局限性：① 解释性主要基于理论推导，缺乏定量验证；② 对 MRF 结构的重构依赖于聚类近似，可能不够精确；③ meta‑model 的高阶扩展导致参数量指数级增长，实际可行性需进一步评估；④ 实验仅针对特定文本模型与简单 GMM，未覆盖更广泛的任务与模型。

---

## 43. Power, Performance, and Area of HPC-Based Malware Detectors: An FPGA and Open-PDK ASIC Study

**arXiv ID:** 2610.12795 | [PDF](https://arxiv.org/pdf/2610.12795v1)

**作者:** Alireza Abolhasani Zeraatkar `[一作]` (University of California), Hussain Al-Asaad `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统地把120个基于硬件性能计数器（HPC）的恶意软件检测器从训练模型转换成可综合的RTL，并在Intel Cyclone V FPGA和SkyWater 130 nm ASIC上完成实现、验证、面积、功耗和时序评估。

**💡 创新点**

首次在完整的设计空间（树、规则、函数、贝叶斯族，4/8/16/2 HPC计数器，AdaBoost与Bagging集成）中同时考虑检测性能与硬件成本，提供了一套自动化的模型‑>RTL流水线与可验证的硬件实现报告。

**🔧 技术方法**

采用机器学习到硅的流水线：模型重构 → 中间表示 → 黄金模型验证 → 生成可综合Verilog → FPGA Quartus II流程 → SkyWater 130 nm OpenROAD流程；还使用Pareto前沿分析、功耗估算和批量实验。

**📊 数据集**

使用从3,700个正常程序和3,700个恶意程序容器化运行获得的HPC数据集，70/30划分训练/测试，采用信息增益/相关聚类得到8/4/2计数器子集。

**📈 对比分析**

通过与Weka参考模型对比得到Precision/Recall/F1，结合FPGA ALMs、ASIC面积、能量/推理和最高频率进行多维度比较；结果显示4‑HPC的J48在ASIC面积12,731 µm²、能量27.3 pJ/推理、FPGA 398 ALMs、F1 94.1%、召回率95.9%时实现了最佳的检测‑成本平衡；AdaBoost等集成方法提升F1约1–2个百分点，但面积/能量显著增大。

**⚠️ 局限性**

限制包括：仅使用完全并行的寄存器控制输入/输出，未做时钟门控或流水线；ASIC功耗估计基于无路由延迟的门级仿真；未考虑连续计数器读数的时间相关性；只评估了15种分类器，未覆盖更复杂或更大规模的模型；部分配置在FPGA或ASIC中无法实现。

---

## 44. Faithful and Complete LLM-Assisted Automated Construction of Hierarchical Hardware Knowledge Graphs

**arXiv ID:** 2610.12762 | [PDF](https://arxiv.org/pdf/2610.12762v1)

**作者:** Haotian Qiao `[一作]` (University of Michigan), Robert P. Dick `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种分阶段、LLM驱动的自动化流程，先对硬件数据表进行分块提取、层次路由和迭代精炼，最终构建具有源 provenance 的层级硬件知识图谱。

**💡 创新点**

创新点在于将知识图谱构建拆解为可控的子任务，并通过指令式执行与本地验证确保语义完整性，同时提出无 ground‑truth 的 Inter‑Graph Completeness 指标评估 KG 完整性。

**🔧 技术方法**

技术核心包括：chunk‑based 信息提取、基于 LLM 的语义路由与迭代去重/重命名/重定位、指令式本地执行、属性标准化、以及基于 LLM 的 Faithfulness 与 Completeness 评估。

**📊 数据集**

使用了 20 篇硬件数据表（传感器 7 篇、处理器 5 篇、电源模块 5 篇、通信模块 3 篇），共 1040 页，涵盖多种硬件类别与文档长度。

**📈 对比分析**

通过与单次 one‑shot 基线比较，本文模型在属性 Faithfulness 上达到 98.8%（对比 76.1%），在完整性上完全覆盖 83.6% 基线属性，基线仅覆盖 10.5% 本模型属性，显示显著的性能提升。

**⚠️ 局限性**

局限性包括：对 LLM 的依赖导致 7% 的标准化错误；未处理图像/图表信息；多对多的部分覆盖可能影响指标解释；目前仅在嵌入式硬件数据表上验证，跨领域适用性仍待探索。

---

## 45. Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization

**arXiv ID:** 2610.12657 | [PDF](https://arxiv.org/pdf/2610.12657v1)

**作者:** Bahareh Nikpour `[一作]` (Huawei Technologies), Parsa Omidi `[通讯]` (Huawei Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种新的 KV 缓存压缩机制 Kronecker Product Attention（KPA），通过将 key/value 映射拆分为两层矩阵 Kronecker 乘积来显著减少记忆体占用。

**💡 创新点**

创新点在于：① KPA 使用 Kronecker 乘积实现两级（头部×特征）结构化压缩，严格泛化了原有的 TPA；② 通过 Kronecker‑Separable RoPE 仅旋转小型因子即可保留相对位置编码，无需 MLA 的分离 RoPE 分支；③ head‑split 参数可调节，允许在压缩率与精度之间在模型设计时自由折衷。

**🔧 技术方法**

核心技术包括：Kronecker 乘积因子化、混合乘积恒等式实现的因子化注意力分数计算、Kronecker‑Separable Rotary Position Encoding（RoPE）以及对应的 KV 缓存压缩策略。

**📊 数据集**

在训练阶段使用 FineWeb‑Edu 语料库（约 50 B tokens），在零样本评估时采用九个公开基准（ARC‑Easy/C、BoolQ、HellaSwag、OBQA、PIQA、WinoGrande、MMLU、SciQ）。

**📈 对比分析**

与 MHA、MQA、GQA、MLA 和 TPA 在相同训练条件下比较，KPA 在 4 种规模（125 M、353 M、773 M、1.5 B）上均达到或超越 TPA 的性能，并实现 6.4×–10× 的 KV 压缩；在大规模模型上，KPA 在 ARC‑Easy、BoolQ、SciQ 等任务上有显著提升。

**⚠️ 局限性**

局限性：目前的实现依赖于手工优化的解码 kernel，尚未在极长上下文（数千 token）上充分验证；Kronecker 乘积计算在某些硬件上可能产生额外的计算开销；此外，仅在单一数据集和任务上评估，未来需扩展到多模态或更大规模的环境。

---

## 46. Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries

**arXiv ID:** 2610.12768 | [PDF](https://arxiv.org/pdf/2610.12768v1)

**作者:** Oktie Hassanzadeh `[一作]` (IBM), Dharmashankar Subramanian `[通讯]` (IBM)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种能够在保持精度召回目标的同时，利用先前查询的标签（视图）安全重用大型语言模型（LLM）评估结果的认证语义过滤器。

**💡 创新点**

创新点在于：①把无成本的侧信息（helper模型分数、局部嵌入、历史视图标签）用于排序，②只用新鲜的oracle样本来构造精确的置信区间，③通过规划和自适应裁剪保证在任意数据集和工作负载下都能以概率至少1‑δ满足精度召回目标，同时将误用视图的风险降至只影响成本而不破坏保证。

**🔧 技术方法**

采用基于Logistic回归的分数模型，分区规划（reject/oracle/accept），超几何置信界，逐轮精炼和补齐缺口，并通过视图匹配、句子嵌入和helper分数作为特征。

**📊 数据集**

使用四个真实文本关系数据集：PubMed（生物医学摘要）、GoEmotions（情感分类）、DBpedia（知识图谱条目）和Reviews（电商评论）各约5,000条记录，分别包含数十个自然语言谓词。

**📈 对比分析**

与单查询优化器（cascade、BARGAIN、embedding proxy）以及跨查询重用方法（exact cache、similarity cache、SemWeave baseline）对比。实验表明该方法在所有数据集上都是最便宜的有效方法，oracle调用量相较精确评估平均减少约30–80%，且满足90%精度召回目标的失败率始终低于5%；相对未经认证的重用方法，其召回或精度失效率高达50%以上。

**⚠️ 局限性**

局限性包括：实验仅在约5,000条记录的小表上验证，未评估百万级规模；仅处理单一过滤器，不覆盖join、排序等复杂操作；工作负载采用随机顺序的谓词池，未考察真实的探索会话；对视图维护和失效策略的进一步研究仍待展开。

---

## 47. Zebra3D: Efficient Hybrid Foundation Models For Scalable 3D Reconstruction

**arXiv ID:** 2610.12727 | [PDF](https://arxiv.org/pdf/2610.12727v1)

**作者:** Utkarsh Saxena `[一作]`, Emad Barsoum `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Zebra3D框架，将预训练的3D重建Transformer升级为稀疏‑线性混合注意力模型，实现无梯度更新即可高效推理。

**💡 创新点**

创新点在于发现不同层对线性替换的敏感性差异，并基于此进行层级感知的稀疏/线性注意力分配；使用闭式仿射拟合实现训练无更新转换；引入基于图块的稀疏注意力与领土化地标汇总以保持全局上下文。

**🔧 技术方法**

使用线性中心化注意力、基于图块稀疏注意力、闭式线性拟合、知识蒸馏（输出、特征、关系蒸馏）、Triton自定义核、混合精度训练等技术。

**📊 数据集**

在VGGT、HunyuanWorld‑Mirror、π³等预训练模型基础上，评估使用7Scenes、NRGBD、DTU、RealEstate10K、CO3Dv2、ScanNet‑50、Tanks & Temples、ETH3D等数据集。

**📈 对比分析**

与原始稠密模型及FastVGGT、SAF3R、RegimeVGGT、Lite3R、BlockSparse、A3D等基线对比；训练无更新下79%稀疏率仅比稠密模型低0.007；训练后94%稀疏率与稠密模型差距<1.1%；在1024视角下实现6.2×–7.1×的推理加速。

**⚠️ 局限性**

需要针对每个backbone单独做敏感性校准，受限于硬件对稀疏/线性注意力的支持；在极端稀疏配置下部分数据集（如CO3Dv2）会出现性能退化；缺乏输入自适应分配策略和更通用的稀疏机制。

---

## 48. Inverse Synthesis of Shock Time Series from Shock Response Spectra Using Ensembles of Conditional Variational Autoencoders

**arXiv ID:** 2610.12820 | [PDF](https://arxiv.org/pdf/2610.12820v1)

**作者:** Adam Watts `[一作]` (Los Alamos National Laboratory), Destry Newton `[通讯]` (Los Alamos National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用条件变分自编码器（CVAE）结合递归小波解码器，从给定的冲击响应谱（SRS）中快速合成符合谱约束的加速度波形。

**💡 创新点**

创新点在于：①小波解码器可从粗到细预测多尺度小波系数，利用逆离散小波变换重构时间序列；②引入模型集成，独立训练多种母小波与分解层，克服单模型下潜在空间采样缺乏多样性；③系统评估不同SRS编码方式与频率网格，并与经典SDS、Brake等方法对比。

**🔧 技术方法**

技术包括：条件变分自编码器、递归小波解码器、离散小波变换（PyTorch Wavelet Toolbox）、多尺度小波系数预测、Hadamard/串联编码、RMSLE、dB误差评估、GPU加速训练与推理。

**📊 数据集**

数据集：训练集包含400k合成冲击与99k测量波形，四个独立的基准测试集（A、B、C、D，分别为运营冲击、地震记录、合成混合信号）。

**📈 对比分析**

与SDS基线和先前的直接时域CVAE进行比较；平均RMSLE降低约23–35%，中位数误差亦显著下降；极端误差下降不一；推理时间仅0.1–0.6 ms/波形，对比SDS需5–8 s；训练耗时约8–24 h/模型。

**⚠️ 局限性**

局限性：单模型对潜在空间采样出现后验坍塌，导致波形多样性不足；生成波形仅在谱上匹配，未保证物理可行性；对高频、极端波形的泛化仍需改进；模型集成需要额外训练与存储成本。

---

## 49. Robust and Scalable UAV-UGV Planning for Aerial Monitoring under Energy and Precedence Constraints

**arXiv ID:** 2610.12670 | [PDF](https://arxiv.org/pdf/2610.12670v1)

**作者:** Cahit Ikbal Er `[一作]` (Northeastern University), Yasin Yazicioglu `[通讯]` (Northeastern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究无人机-无人车协同任务规划，加入能量与优先级约束，提出多团队调度方法。

**💡 创新点**

共同确定路线和时序、允许团队减速满足优先级，提出多项式时间启发式算法并给出可行性与复杂度证明。

**🔧 技术方法**

通过混合整数规划建模，并使用基于优先级图的迭代分配启发式，结合最短路径、投影、等待时间等计算。

**📊 数据集**

采用合成随机分布的监测点集，在 4000 m × 4000 m 环境中生成随机优先级图进行数值评估。

**📈 对比分析**

与分支定界、模拟退火及分离优先级基线比较，算法在可解规模下计算时间毫秒级，完成时间仅比最优约 8‑28% 更优；在大规模/高密度优先级下显著降低完成时间。

**⚠️ 局限性**

仅适用于预先已知的顺序关系，未处理时间窗口或在线干扰，启发式算法缺乏近似比保证。

---

## 50. Expert-level detection of low-voltage intervals in EEG as a robust biomarker of neonatal encephalopathy

**arXiv ID:** 2610.12680 | [PDF](https://arxiv.org/pdf/2610.12680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments

**arXiv ID:** 2610.12611 | [PDF](https://arxiv.org/pdf/2610.12611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 52. AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits

**arXiv ID:** 2610.12545 | [PDF](https://arxiv.org/pdf/2610.12545v1)

**作者:** Shu Ishida `[一作]` (Autodesk Research), Adam Gaier `[通讯]` (Autodesk Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种Chain‑of‑Edits解码机制，使模型能够在生成过程中插入、删除或替换任意数量的标记，从而实现对先前输出的主动修正；

**💡 创新点**

创新点包括：①通过标记头和指针网络学习何时何地编辑；②使用逆向编辑训练（类似扩散）使模型学习撤销扰动；③在自回归框架中引入链式RoPE并通过专用注意力核实现位置编码的实时重排；④序列解析器将生成的编辑序列映射回最终显示顺序；

**🔧 技术方法**

采用Transformer自回归架构，配合标记头、指针头、RoPE+链式RoPE、合并注意力核以及自监督逆向编辑训练；

**📊 数据集**

使用多领域通用数据集进行评估：自然语言（MMLU、ARC‑Challenge、Hellaswag、WinoGrande、PIQA）、数学/科学（GSM8K、MATH500、GPQA‑Diamond、GPQA‑D）、代码（HumanEval、MBPP）以及自制的扰动编辑训练序列；

**📈 对比分析**

在标准LLM基准上与自回归、离散扩散、块扩散等模型对比，AutoIndexer在保持KV缓存和左到右生成优势的同时，保持或略优的性能；在代码修正任务（HumanEval）表现最高；在主动修正评估中对删除/替换的识别率≈90%，插入修正效果仍低；

**⚠️ 局限性**

局限性：插入修正准确率相对较低；编辑操作的定位和执行仍需改进；训练依赖自监督扰动，缺乏真实编辑历史；对实时对话或长篇文本的应用尚未充分验证；

---

## 53. Unsupervised Multimodal Reward Modeling

**arXiv ID:** 2610.12707 | [PDF](https://arxiv.org/pdf/2610.12707v1)

**作者:** Yueying Li `[一作]` (Massachusetts Institute of Technology), Hanlin Zhang `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种无人工偏好标签的视觉‑语言奖励模型，通过在冻结的 Qwen3‑VL 主干中插入线性探针，并使用同一图像的句子或图像响应扰动进行比较训练；该模型不需要任何人工标注即可在多个奖励基准上竞争甚至超过 GPT‑4o。

**💡 创新点**

创新点在于：① 将人类偏好标签完全剔除，改用由扰动采样器生成的负样本；② 通过探针深度选择和扰动采样器确定奖励方向，揭示基准准确度与实际部署性能不一致；③ 采用自监督的“验证器”循环对模型自身错误进行标注，进一步提升生成文本的真实性。

**🔧 技术方法**

技术手段包括：在冻结的 Qwen3‑VL 主干第 N‑k 层输出上使用 2048 参数的无偏置线性头；使用 in‑batch Bradley–Terry 损失进行对比学习；负样本通过文本扰动（near、splice、shuffle、cross‑image）和图像提示扰动生成；以及基于 DPO 的自我迭代循环与验证器标注。

**📊 数据集**

使用数据集：70k PixMo 网络图像描述；Qwen3‑VL（2B、4B、8B、32B）主干；评估基准为 Multimodal RewardBench、VL‑RewardBench、MMRB2；Best‑of‑N 评测使用 COCO、MathVista、MMMU 及 Claude‑写的硬题。

**📈 对比分析**

比较结果显示：在 32B 规模下，探针模型在 VL‑RewardBench 上比 GPT‑4o 提高 6.6 分；在 2B/8B 规模下亦能匹敌 GPT‑4o；相较于精心策划的 RLAIF‑V 对比，采用同图像扰动的负样本在等价样本预算下提升 3–5 分；使用验证器标注的自我迭代模型可将 Caption 生成的 CHAIR 指标降低 5.1 分。

**⚠️ 局限性**

局限性包括：① 基准准确度不一定能反映实际部署中的最佳‑of‑N 性能；② 探针深度和扰动采样器的选择对奖励学习高度敏感；③ 模型在某些基准（如 MMRB2）上仍显弱；④ 只适用于冻结的主干结构，难以直接迁移到可微的端到端微调；⑤ 需要精心设计扰动操作以保证负样本的有效性。

---

## 54. Credit Attribution in Timestamped Soccer Networks Using Multivariate Hawkes Processes

**arXiv ID:** 2610.12775 | [PDF](https://arxiv.org/pdf/2610.12775v1)

**作者:** Phillip W. Kornberg `[一作]` (Case Western Reserve University), Kevin S. Xu `[通讯]` (Case Western Reserve University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文使用多变量 Hawkes 过程模型对足球比赛中的传球和射门序列进行建模，并通过该模型计算每次传球对后续射门的责任度，从而实现多触点信用归属。

**💡 创新点**

创新点在于提出“期望进球参与度（xGI）”指标，该指标将射门的期望进球值（xG）按传球责任度分配给整个进攻链路上的所有球员，而不仅仅归于最终助攻者；同时首次将传球对射门的因果影响量化为可解释的概率。

**🔧 技术方法**

主要技术包括：指数型多变量 Hawkes 过程（MHP）用于学习传球对后续事件的激励矩阵；基于该矩阵与基线率计算责任概率；对激励矩阵做小概率平滑；以及与简单时间衰减基线模型的对比。

**📊 数据集**

使用的数据集为公开的 StatsBomb 2015/2016 赛季西甲（La Liga）全部球队的传球和射门事件，经过半场拆分、时间戳对齐和只保留前 100 个最常出现的传球对的预处理。

**📈 对比分析**

评估方法为将 MHP 计算出的责任概率与仅基于时间衰减的基线模型（TD）进行对比；结果显示 MHP 能在多触点情境下显著分配不同责任权重，并与传统 xA、关键传球等指标保持一致，同时揭示球队间激励网络差异（如巴塞罗那平均 α_ij 最高）。

**⚠️ 局限性**

局限性包括：仅考虑前 100 个最常传球对，忽略稀有传球及未纳入模型的事件；固定的衰减参数 β 可能不适用于所有比赛节奏；激励矩阵的估计对训练数据长度和噪声敏感；以及目前的 MHP 结构无法自然处理多级节点聚合（如社区 Hawkes 模型）。

---

## 55. CTML: A Comprehensive Framework for Continuous Testing in Machine Learning Operations

**arXiv ID:** 2610.12515 | [PDF](https://arxiv.org/pdf/2610.12515v1)

**作者:** Bestoun S. Ahmed `[一作]` (American University of Bahrain), Bestoun S. Ahmed `[通讯]` (American University of Bahrain)

**通讯引用:** 3024 | [OpenAlex ID](https://openalex.org/A5031391282)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个名为CTML的框架，用于在机器学习操作中进行持续测试，旨在解决传统软件测试方法在ML系统中的不足。

**💡 创新点**

CTML框架通过标准化测试方法，涵盖数据质量、漂移检测、模型质量和管道验证四个关键领域，提供全面的自动化解决方案。

**🔧 技术方法**

使用了模块化架构，包括测试注册表、测试运行器、结果存储和集成层，确保与现有的CI/CD管道和MLOps平台无缝集成。

**📊 数据集**

在一个预测维护的案例研究中使用了合成时间序列数据集，模拟工业设备监控，包含传感器读数和设备故障标签。

**📈 对比分析**

通过与传统测试方法的比较，CTML框架在识别数据漂移、性能下降和管道瓶颈方面表现出色，能够发现传统方法可能忽视的关键问题。

**⚠️ 局限性**

局限性包括案例研究集中在预测维护领域，缺乏在其他领域（如NLP、计算机视觉等）的验证；使用合成数据而非真实数据，且在处理大规模系统时的可扩展性尚未证明。

---

## 56. Multimodal Image-based Robotic Assistance and Guidance Environment (MIRAGE): A Novel Unified Framework for Surgical Navigation and Visualization

**arXiv ID:** 2610.12780 | [PDF](https://arxiv.org/pdf/2610.12780v1)

**作者:** Ehsan Nasiri `[一作]` (Dartmouth), Ryan J. Halter `[通讯]` (Dartmouth)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并验证了一个多模态图像辅助机器人手术框架MIRAGE，集成立体视觉重建、AR叠加、HUD导航、超声、跟踪等模块，实现在达芬奇手术台上的实时工作流；

**💡 创新点**

创新点在于统一时间同步的跨语言（C++/Python）架构，实时零拷贝绑定、状态机管理，能在单台工作站上把多模态传感器与实时渲染同步，并提供可量化评估指标；

**🔧 技术方法**

使用的技术包括立体相机标定、SVD刚性配准、SLERP插值、RAFT/PSMNet深度估计、OpenGL渲染、pybind11绑定、线程安全共享、OpenIGTLink、Plus、3D Slicer、CUDA GPU等；

**📊 数据集**

实验数据集包含多种解剖模型/phantom（肾脏、喉咙、通用模型）以及CT/US DICOM数据，跨四台达芬奇S、Si、Xi、SP平台；

**📈 对比分析**

与前人单一模态系统对比，采用RMSE、TRE、投影误差、同步误差、延迟等七项指标；在100mm工作距离下立体重投影误差0.1mm，AR投影误差≤2.5px，系统整体延迟约16.7ms（视频）+200ms重建，帧率60fps，跨平台重现性良好；

**⚠️ 局限性**

限制在于仅完成预临床phantom实验，缺乏临床验证、人体工学、学习曲线、患者安全、成本和长期监测评估，且对软组织变形和跟踪漂移的鲁棒性仍有限。

---

## 57. StereoEngine: Scaling Stereo Bimanual Robot Data without Stereo Cameras

**arXiv ID:** 2610.12734 | [PDF](https://arxiv.org/pdf/2610.12734v1)

**作者:** Jason Chen `[一作]` (University of Southern California), Daniel Seita `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种基于视频扩散的双目数据生成框架：-L方法在已有单目演示的基础上，通过深度估计与前向 warp 产生右视并用视频扩散模型完成遮挡填补；-S方法在无真实单目视频时，利用模拟轨迹、Canny 边缘控制以及双目参考图像，联合生成完整的左右视图；两种方法均可用于大规模中训练或任务特定微调，从而显著提升双手操作的立体感知和性能。

**💡 创新点**

创新点包括：①无需真实双目数据即可生成逼真双目演示；②将单目到双目的 warp+inpaint 与视频扩散相结合，首次实现动态场景的高质量右视生成；③在模拟与真实之间搭建无缝桥梁，使用边缘控制与参考图像共同生成一致的左右视；④大规模生成双目数据显著提升现有立体条件策略（ExStereo、StereoVLA）的成功率。

**🔧 技术方法**

核心技术包括：单目深度估计（inverse depth）、前向 warp 与遮挡掩码生成；视频扩散模型 Wan2.2-VACE-Fun 与 Wan2.2-Fun-A14B-Control（在不训练真实双目数据的前提下 fine‑tune 或零样本生成）；Canny 边缘提取作为结构控制；双目参考图像与语言提示作为外部条件；以及基于立体视觉的深度恢复与点云训练。

**📊 数据集**

使用的数据集包括：3,469 条公开单目机器人演示（AgileX PiPER、Cobot Magic 等），通过 -L 方法生成双目数据；200/50/20 条真实双目演示（ZED Mini 2）用于下游微调；另外在模拟环境（RoboTwin）中生成 200/50/50 条任务专用轨迹，用于 -S 方法。评估时收集 480 条真实双目执行轨迹。

**📈 对比分析**

与 SVG、StereoCrafter 等基线进行对比，-L 在三项任务（堆叠碗、透明瓶、关闭门）中 mid‑training 成功率提升至 83.3%（对比 60% 或 66.7%），在单个任务中从 11/20 提升到 18/20；-S 在点云基准上实现显著提升，点云方法为 0% 成功率，而 -S 在各任务均取得 11/20 以上成功率。生成质量评估中 PSNR 15.31、SSIM 0.530、LPIPS 0.470、CTC 0.986，均优于 StereoCrafter 与 SVG。

**⚠️ 局限性**

局限性包括：①对透明或高反射表面生成仍易出现误差；②依赖高质量参考图像与语言提示；③模拟与真实的差距仍在某些任务（如门、接触动态）显现；④视频扩散模型对帧率和实时性的支持有限；⑤缺乏公开基准与评测数据集，导致跨方法对比受限。

---

## 58. Amortised Inference through One-Step Implicit Sampling

**arXiv ID:** 2610.12679 | [PDF](https://arxiv.org/pdf/2610.12679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 59. Spatial Context or Memory? Sizing the Inputs of Neural Surrogates for Partially Observed Fields

**arXiv ID:** 2610.12723 | [PDF](https://arxiv.org/pdf/2610.12723v1)

**作者:** Ruige Kong `[一作]` `[通讯]` (University of Washington), Ruige Kong (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在稀疏、噪声观测下，用神经代理（neural surrogate）预测物理场时，空间窗口（window）与时间记忆（history）之间的权衡，并通过精确的高斯解析结果为线性场提供贝叶斯最优误差基准；随后在三种非线性场（粘性 Burgers 流、KS 方程和随机受迫线性场）上训练局部 CNN、GRU、MLP 等模型，探讨了历史对误差的削减以及所谓的“lead‑time inversion”（长时间预测时历史效益反转）。

**💡 创新点**

创新点包括：①提出并实现了针对局部观察的贝叶斯最优误差解析，能将模型误差与信息缺失分离；②揭示了在不稳定 KS 方程中，历史记忆在较长预测时段比空间窗口更能减少误差的逆转现象；③给出两条设计规则：窗口应按场的相关长度确定；给定固定输入预算时，窗口与历史的分配应随预测时段调整。

**🔧 技术方法**

技术方法包括：精确的线性高斯链分析、条件协方差与最优增益计算；神经网络代理（patch CNN/GRU/MLP、全局 Fourier Neural Operator）训练；AdamW 优化器与余弦退火学习率；交叉自举 Bootstrap 对比与置信区间；扰动演化分析与 Lyapunov 指数计算。

**📊 数据集**

数据集：所有场均为合成模拟，分别为（a）随机受迫线性推进-扩散场（两种版本），（b）粘性 Burgers 流，和（c）Chaotic Kuramoto‑Sivashinsky (KS) 方程；在 1D 圆环和 2D 方形域上实验；传感器密度为每 4 个格点一传感器（或更细）并加入噪声（非线性场 5% RMS，线性场 100%）。

**📈 对比分析**

对比方法：将训练模型的测试误差与（i）线性场的解析最优误差或（ii）非线性场下最佳全局 Fourier 代理误差进行比较，计算误差比例、gap ratio、history ratio 等指标；使用自举得到 95%/99.17% 区间；结果表明：在线性场训练模型接近 1% 最优误差；在非线性场中，较大窗口和足够历史能显著降低误差；在 KS 中出现 lead‑time inversion，即长时间预测时历史比短时间更有效。

**⚠️ 局限性**

局限性：①非线性场的“gap”估计依赖全局基线模型的性能，可能低估信息缺口；②每个场仅使用单一噪声水平，无法检验噪声对失效模式的影响；③仅测试单一漂移速度和窗口尺寸，无法系统验证第二失效模式；④仅使用一类混沌方程（KS）与稳定方程（Burgers）来探讨逆转，无法区分混沌与线性不稳定的独立作用；⑤预测时段仅覆盖约 0.071 Lyapunov 时间，缺乏更长时段的验证；⑥仅在 1D 和 2D 单一几何上验证，泛化能力未充分检验。

---

## 60. What Counts as Evidence? Epistemic Cue Utilization in Large Language Models

**arXiv ID:** 2610.12788 | [PDF](https://arxiv.org/pdf/2610.12788v1)

**作者:** Jinchang Zhang `[一作]` (Indiana University Bloomington), Guoyu Lu `[通讯]` (Indiana University Bloomington)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个包含500个可控情景的基准，分别评估语言模型在面对来源冲突时能够识别哪些信息提示（source history、evidence quality、lineage、event time、presentation order）以及这些提示对最终判断的影响；同时在三个 Qwen 模型上进行提示重述、组织和加权规则的干预实验，探究模型对提示的可塑性。

**💡 创新点**

创新点在于将信息提示的“可识别性”与“影响力”拆分为独立维度，设计了五个互补的诊断轨道（表示、整合、可靠性利用、提示竞争、已验证推理）来量化这一分离；通过实验干预展示提示识别与影响力并非固定属性，可通过信息呈现方式显著改变模型的判断模式。

**🔧 技术方法**

技术手段包括：①基于人工构造的情景库与 Prompt 模板，精确操控 5 种提示；②对 13 种公开与专有 LLM（Qwen、InternVL、Grok、Claude、GPT‑5/6、Kimi 等）进行统一评测；③利用“置信区间 bootstrap”与 Holm 校正的统计检验；④对干预实验采用对照（Base、Sham、Flat、Ledger、+Rule）设计，量化可靠性交换和文本顺序效应。

**📊 数据集**

数据集为 500 个人工构造的“评估世界”，分为 Controlled Core（300）、Structural Generalization（120）和 Controlled Naturalized（80），覆盖 12 关系族、25 领域、10 话语格式，包含 12,000 条模型请求并提供 312,000 个系统–请求单元。

**📈 对比分析**

评测方法：对每个模型分别在 5 条轨道上记录识别准确率、提示影响对比值、验证精度；对干预实验进行 48 个预设对比，使用 Holm 校正。结果显示：模型识别相对可靠性几乎总是高达 99%+，但对可靠性交换的影响从 0% 到 100% 不等；在已验证推理轨道上大部分模型实现 97–100% 的准确率；干预实验中，+Rule 能将可靠性交换效应提升 39–47 个百分点，提示重述显著提升 25–13 个百分点，文本顺序效应在某些模型中被显著削弱。

**⚠️ 局限性**

局限性：①基准情景为人工合成，缺乏真实世界的复杂性、交互和多模态信息；②干预实验仅针对三个 Qwen 检查点，未验证在其他架构上的普适性；③评测侧重最终答案的正确性与提示敏感性，未直接评估在实际决策场景中的实用价值。

---

## 61. Coordinate-Free Automatic Differentiation

**arXiv ID:** 2610.12776 | [PDF](https://arxiv.org/pdf/2610.12776v1)

**作者:** Alex Guerra `[一作]` (Princeton University), Peter Orbanz `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于几何代数的坐标自由多向量导数，并将其实现为可复用的 JAX 转换，随后在多种场景（流形学习、逆问题、旋转群优化等）中展示其应用。

**💡 创新点**

创新点在于：①把几何差分操作统一为单一、基底无关的多向量导数；②该导数完全可通过现有 AD 原语构造；③通过投影和切空间投影即可直接得到流形上的外微分、散度、共变导数等经典算子，无需 Christoffel 符号或坐标图。

**🔧 技术方法**

使用技术包括几何代数（Clifford 代数）、JAX 自动微分原语（jvp、vjp）以及自定义的“metric aware”向量-雅可比乘积；实现了向量导数、投影 P、切空间导数 ∇_M，并在此基础上构造了曲率、拉普拉斯、逆问题中的 Maxwell 方程等算子。

**📊 数据集**

主要使用的是人工合成的几何数据集：如 icosidodecahedron → ℤ₅ 针形、rectified {3,7} 贴片 → D₇ 形状、平面方块流形、球面、Poincaré 圆盘等；在逆设计实验中采样稀疏的 Faraday 双向量观测点；在旋转群优化实验中使用人工生成的点云对齐任务。

**📈 对比分析**

比较方法主要是定性展示：对比目标几何（如目标平面、球面、手性针形）与学习结果；在逆问题中对比预测的粒子轨迹与真实轨迹；在旋转群优化中对比不同子群的收敛速度。实验中未给出数值性能基准，仅说明该方法能在单一框架下完成多种任务，且与现有专用实现相比在实现简洁性和复用性上具有优势。

**⚠️ 局限性**

局限性包括：①多向量空间维度随向量空间维数呈指数增长，限制了高维应用；②需要手工指定几何投影（I_M、P 等），对复杂流形的建模仍有难度；③对真实大规模数据集的实验缺失，缺乏对比基准和计算效率评估；④在某些场景下（如高阶 k-forms 的质量范数）缺乏闭式表达，导致实现更为复杂。

---

## 62. Distributionally Robust Adaptive Iterative Covariance Steering for Small-Body Proximity Operations

**arXiv ID:** 2610.12774 | [PDF](https://arxiv.org/pdf/2610.12774v1)

**作者:** Vivek Khatana `[一作]` (University of Illinois Urbana Champaign), Petros Voulgaris `[通讯]` (University of Nevada Reno)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合迭代协方差导引（iCS）与L1分布式鲁棒自适应控制（L1-DRAC），提出一种连续推力小天体近距离操作的风险感知碰撞规避框架。

**💡 创新点**

首次将Wasserstein误差约束嵌入iCS的分布式CVaR安全约束，并通过L1-DRAC动态补偿模型不匹配，实现了在连续推力下的闭环安全保障。

**🔧 技术方法**

使用的技术包括：迭代协方差导引（iCS）、L1-DRAC、Wasserstein分布式CVaR约束、随机仿真、核密度估计（KDE）等。

**📊 数据集**

数据集为1000次基于归一化ANH3BP模型的仿真实验，随机采样重力参数误差、太阳辐射压误差以及扩散不确定性，构成完整的测试数据集。

**📈 对比分析**

通过与不使用L1-DRAC的基线iCS进行对比，评估保持距离违规率、物理碰撞率和低尾CVaR等指标。实验显示，L1-DRAC将保持距离违规率从32.7%降至0.2%，碰撞率从16.4%降至0%，且低尾CVaR始终保持非负。

**⚠️ 局限性**

局限性包括：仅在低阶连续推力仿真中验证，未覆盖更高保真动力学或硬件实现；规划求解时间较长（≈10.85秒），需要进一步加速；模型假设误差为匹配加速度，未考虑更广泛的不匹配形式。

---

## 63. MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis

**arXiv ID:** 2610.12659 | [PDF](https://arxiv.org/pdf/2610.12659v1)

**作者:** Ximing Lu `[一作]` (University of Washington), David Acuna `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了强化学习可验证奖励（RLVR）数据的有效性，提出了 MagicBeanStalk（MBS）算法以从难题种子生成针对目标策略和预算校准的高质量训练数据，并在数学与金融任务上实现显著性能提升。

**💡 创新点**

创新点包括：① 引入零优势陷阱（zero‑advantage trap）分析，揭示问题、策略与预算三者相互作用；② 将难题拆解为验证树并递归合成新问题，保持推理蓝图同时调节难度；③ 通过校准黄金区（Goldilocks band）与在预算内可解的约束，系统生成具有学习信号的数据。

**🔧 技术方法**

使用技术包括：RLVR + GRPO持续强化学习；多教师模型采样与验证；树形问题分解与递归合成；奖励过滤、黄金区筛选与预算校准；多模型评估与一致性判定。

**📊 数据集**

使用数据集：78K开源数学问题集合（DAPO‑Math‑17K、Skywork‑OR1‑RL‑Data、Turing Open‑RL、StackMathQA 等）；金融基准 SecQue、FinanceBench；由 MBS 生成的 14K+ 文档基金融问题；以及 8 大数学评测基准（AIME 24/25、AMC、HMMT、Minerva、Olympiad、Omega‑500）和 MathArena Apex‑Shortlist。

**📈 对比分析**

对比方法：与全量源数据、Golden Goose、Seed Synthesis、Goldilocks 子集、直接教师生成等做基准比较；在数学 8 大基准上 MBS 平均提升 +8.6 分（gpt‑oss‑120b），在 DeepSeek v4 Flash 上提升 +5.4 分；在金融任务上 SecQue 提升 +6.9 分，FinanceBench 提升 +2.4 分；在更大推理预算（24K/32K）下仍优于基线。

**⚠️ 局限性**

局限性：① 生成过程高度依赖高性能教师模型，若教师变动或错误会影响数据质量；② 零优势陷阱分析基于有限抽样，可能漏判；③ 对不同策略或预算的泛化仍有限；④ 生成数据可能包含细微错误，需人工审核；⑤ 解释性与安全性评估尚不充分。

---

## 64. Bridging Human and Model Attention: Explainability Analysis of CNN, Mamba, and ViT Architectures with Gaze-Based Validation

**arXiv ID:** 2610.12703 | [PDF](https://arxiv.org/pdf/2610.12703v1)

**作者:** Marco Stefani `[一作]` (University of Padova), Alberto Pretto `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对CNN、ViT与Vision Mamba三类网络的可解释性进行了系统性评估，使用人眼注视热图作为客观基准，通过标准化流程将模型注意力图与人类注视图转换为概率分布，并采用Jensen‑Shannon、Chi‑square及Pearson相关三种相似度指标量化对齐程度。

**💡 创新点**

创新点在于：①将眼动数据视为严格的“ground truth”来评估模型解释性；②构建统一归一化与相似度计算框架，并理论分析三种指标的边界与不变性；③首次在医学、细粒度分类和情感识别三大领域对三种网络结构进行横向比较；④探讨骨干冻结与集成策略对可解释性的影响。

**🔧 技术方法**

技术包括：Grad‑CAM、Score‑CAM、Ablation‑CAM（CNN）；ViTLRP（ViT）；MambaLRP（Vision Mamba）作为模型特定的XAI方法；人眼跟踪采集与视觉特征提取；归一化到概率分布的四步流程；Jensen‑Shannon、Chi‑square、Pearson相关三种相似度度量；统计显著性与Spearman秩相关分析。

**📊 数据集**

使用三大数据集：CUB‑200‑2011（细粒度鸟类分类，配有20名非专家注视数据）、MIMIC‑CXR（医学胸部X光，单位放射科医师注视数据）、KDEF‑FGD（人脸情感识别，20名受试者注视数据）。

**📈 对比分析**

评估方法：对每张图像生成模型注意力图后与人类注视图计算三种相似度，取平均与标准差，并与模型分类性能（AUC、准确率）一起报告。结果显示：CNN在自然图像和医学图像上表现稳健；ViT在细粒度任务中较CNN更细粒度；Vision Mamba在医学任务中获得最高的Jensen‑Shannon和Chi‑square分数，但Pearson相关低，表明其关注区域与人类相似但强度不匹配；冻结骨干的ViT在医学数据上可提升人类注视对齐而降低分类精度；集成模型在某些域可提升准确率，但对可解释性的提升有限。

**⚠️ 局限性**

局限性包括：①人眼注视数据来源有限且存在受试者偏差（如CUB仅用非专家、KDEF仅采样年轻西方人群）；②归一化与相似度度量对不同模型产生不等价影响，导致解释性评价受模型输出分布影响；③MambaLRP的负值被截断，可能掩盖重要的反证信息；④实验仅覆盖三大域与三类网络，缺乏更广泛的任务与网络变体验证；⑤集成策略仅采用简单平均，未探索更复杂的融合方法。

---

## 65. Distributed Harmonic-Covariance Feedback for Hole Closure in Planar Robot Swarms

**arXiv ID:** 2610.12637 | [PDF](https://arxiv.org/pdf/2610.12637v1)

**作者:** Jitian Liu `[一作]` (Lehigh University), Subhrajit Bhattacharya `[通讯]` (Lehigh University)

**通讯引用:** 2670 | [OpenAlex ID](https://openalex.org/A5101463763)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种基于谐波协方差反馈的多智能体覆盖闭孔控制器，能够在不需要定位的情况下通过计算Cech复形的谐波边流实现全局唯一轨迹并实现孔洞闭合。

**💡 创新点**

创新点在于将谐波边流通过协方差映射转化为局部受限力场，并利用正则化Hodge拉普拉斯矩阵的多项式近似实现分布式计算，同时构造离线有理下界序列逼近累计孔洞面积，从而无需实时轨迹查询即可预测闭孔效果。

**🔧 技术方法**

主要技术包括离散几何中的Čech复形与Hodge拉普拉斯、正则化权重与Chebyshev多项式滤波、协方差投影实现的力生成、以及基于积分约束和量化消解的离线最优化求解。

**📊 数据集**

使用的“数据集”是实验中所用的60个移动感知代理的初始位置与孔洞配置，并在仿真中采用开源的数值积分器对控制器进行验证。

**📈 对比分析**

与传统的梯度下降或基于拓扑修复的闭孔方法相比，该方法在15秒内成功闭合两个孔洞，仿真结果表明控制力受限且能在有限时间内实现覆盖完整性，虽然缺乏与现有方法的数值比较。

**⚠️ 局限性**

主要限制包括对孔洞连续性、分裂/合并以及有限临时孔洞的严格假设，计算量巨大的q_T构造与量化消解步骤限制了其在实际大规模系统中的可行性，且无法保证点态孔洞消失。

---

## 66. KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting

**arXiv ID:** 2610.12738 | [PDF](https://arxiv.org/pdf/2610.12738v1)

**作者:** Jonas Petersen `[一作]` (ETH Zurich), Philipp Christian Petersen `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了KITE（Knowledge‑Injected Temporal Encoder），通过可切换的机制将多种语义先验注入自监督事件预测器，实现对稀有高成本事件的精准预报。

**💡 创新点**

创新点在于：①将周期性、守恒、结构描述、因果图等十二类语义先验统一映射为两条可切换路径（covariate 与 attention‑bias），无需为每个数据集重新设计网络；②证明先验仅在窗口缺失信息时发挥作用，既能在短窗口内提升性能，又能在跨域迁移中替代特征迁移。

**🔧 技术方法**

技术细节包括：基于JEPA的自监督编码器，RoPE‑ordered patches实现可变维数，VICReg损失与自监督预训练；两阶段训练：Stage‑1 预训练编码器与预测器，Stage‑2 冻结编码器训练事件头；合成变量路径与注意力偏置机制。

**📊 数据集**

实验使用工业监测数据集 BATADAL、Occupancy、SYNTH‑ROUTE‑v2 以及合成数据；与 Chronos‑2、PatchTST、iTransformer 等预训练基线对照。

**📈 对比分析**

通过对比无先验、先验与外部基线的 h‑AUROC，KITE 在短窗口下显著提升（如 Occupancy：0.866 vs 0.762），在跨域迁移中先验替代特征迁移，整体性能高于同类基线。

**⚠️ 局限性**

局限性包括：①先验在窗口已包含信息或误配时无效；②attention‑bias 机制提升有限；③仅验证了四类语义先验，剩余八类仍为概念；④实验仅聚焦短窗口与少数先验，需进一步扩展至更长上下文与更多先验类型。

---

## 67. Generative AI Steering and Human Authoring: Complementary Search-as-Learning Scaffolds in Cross-Session Aggregated Academic Search

**arXiv ID:** 2610.12755 | [PDF](https://arxiv.org/pdf/2610.12755v1)

**作者:** Milad Momeni `[一作]` (University of Regina), Orland Hoeber `[通讯]` (University of Regina)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一个聚合学术搜索界面中，研究比较了两种跨会话整合工具——基于人工笔记的NT和基于可控生成摘要的STAIS，探讨它们如何影响搜索后的复习效率和学习成果。

**💡 创新点**

创新点在于将“整合劳动分配”作为GenAI支持的核心设计变量，提出可调节的Steerable AI Summaries（STAIS）并与传统笔记（NT）进行对照，揭示不同分配模式对后续学习的不同影响。

**🔧 技术方法**

使用了GPT‑4o‑mini进行聚类级摘要生成、源引用、事实检查以及用户可编辑的内联修改，结合聚类算法和工作区增量式组织。

**📊 数据集**

数据集为从大学学术图书馆和维基百科两源检索到的文献和条目，实验参与者共36名二年级计算机科学学生完成两周一次的搜索任务。

**📈 对比分析**

通过两组之间的实验对比，测量了效率（恢复时间）、精确度、概念覆盖与连接数等指标；结果显示STAIS在恢复时间和概念覆盖上优于NT，而NT在概念连接数和自信度上更突出。

**⚠️ 局限性**

局限性包括：两种工具的功能被捆绑，难以单独评估各组成；实验仅覆盖技术性较强的网络安全领域，样本为学生；实验时间窗口短、实验室环境可能带来需求效应；使用单一LLM模型，未做闭卷学习测评。

---

## 68. CASTER: Formalizing Zero-Shot Manipulation from Synthetic Videos as Constrained Optimization

**arXiv ID:** 2610.12567 | [PDF](https://arxiv.org/pdf/2610.12567v1)

**作者:** Yinlong Dai `[一作]` (Virginia Tech), Dylan P. Losey `[通讯]` (Virginia Tech)

**通讯引用:** 1882 | [OpenAlex ID](https://openalex.org/A5063608480)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出一种将生成的合成视频转换为真实机器人零射击操控轨迹的框架。

**💡 创新点**

创新点在于将视频生成与机器人执行视为受约束优化问题，定义“语义吸引域”和“物理可行性约束”，并提出利用多视频特征提取与数字孪生优化的CASTER方法。

**🔧 技术方法**

使用视频生成模型（如HunyuanVideo-1.5）、LLM（如Qwen3.5）进行任务验证、深度估计模型、TrackCraft3R 3D 追踪、数字孪生 Isaac Sim 与 SAM 3D、交叉熵方法（CEM）进行轨迹优化。

**📊 数据集**

数据集包括12个真实世界桌面操控任务（香蕉放入锅、堆叠碗、倒咖啡等）和从相机采集的RGB‑D图像；合成视频数量为每任务20条。

**📈 对比分析**

与两种基线（直接重定向与精确跟踪）比较，CASTER在12项任务中平均零射击成功率为76.6%，比最佳基线高出34.1%。

**⚠️ 局限性**

局限性包括数字孪生与真实物理差异导致的失败、对视频生成模型的依赖、以及对复杂接触任务的鲁棒性不足。

---

## 69. Game Petri Nets: Compositional Modeling and Analysis of Large-Scale Strategic Systems

**arXiv ID:** 2610.12779 | [PDF](https://arxiv.org/pdf/2610.12779v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University), Quanyan Zhu (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出游戏 Petri 网（OCGPN）的框架，用于对大规模战略系统进行分层建模与分析。

**💡 创新点**

创新点在于将局部博弈与 Petri 网结构耦合，形成结果驱动的组合模型，并给出结构与战略的可组合性理论及判定方法。

**🔧 技术方法**

采用 Petri 网、彩色令牌、局部博弈、稀疏矩阵、Farkas 判据、强化学习等技术，并通过线性规划、SCC 分解实现计算。

**📊 数据集**

未使用公开数据集，主要通过两条演示案例（维护工作流与网络攻击链）验证理论。

**📈 对比分析**

与传统的单片博弈或 Petri 游戏比较，OCGPN 在安全性判定、可达性分析以及值计算上实现了多项式级别的可组合计算，复杂度从指数级缩减为增量级。

**⚠️ 局限性**

局限在于对并发激活的假设、颜色空间的可枚举性、以及对逆向策略或不确定性调度的处理仍有限。

---

## 70. Knowing Is Not Telling: Identity Ossification in LLM Representations of Trans-border Communities

**arXiv ID:** 2610.12610 | [PDF](https://arxiv.org/pdf/2610.12610v1)

**作者:** Minglu Sun `[一作]` (Northeastern University), Hang Jiang `[通讯]` (Northeastern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了跨境族群在大语言模型中的身份僵化现象并提出tiodf框架。

**💡 创新点**

创新点在于识别并量化模型在叙事中将多国族群压缩为单一国家身份的结构性偏差。

**🔧 技术方法**

使用二进制知识探测、开放式叙事生成和叙事框架评分技术。

**📊 数据集**

基于九个中南亚跨境族群的知识卡和公开数据构建的提示集。

**📈 对比分析**

通过比较模型（GPT‑5.1、DeepSeek‑V3.2等）在英语和中文下的身份僵化比例，发现中国模型中文回答更高，且知识检索与叙事质量无显著关联。

**⚠️ 局限性**

局限在于样本模型有限、评分依赖单一判别者、二进制探测不完美以及语言与国家框架混杂影响。

---

## 71. Mesh-Native Tokenisation: Retrofitting Foundation Models to Unstructured Data

**arXiv ID:** 2610.12618 | [PDF](https://arxiv.org/pdf/2610.12618v1)

**作者:** Georg Maierhofer `[一作]` (University of Cambridge), Payel Mukhopadhyay `[通讯]` (University of Cambridge)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种 Mesh‑Native Convolution (MNC) 令未结构网格数据可直接使用 CNN 预训练权重，实现物理场数据的 token 化，构建可在网格与非网格之间无缝迁移的物理仿真器。

**💡 创新点**

创新点在于将连续卷积核映射到无结构网格上，并通过占据归一化保证与传统 CNN 对齐，理论证明了 MNC 在细化网格时保持不变性；同时实现了从网格预训练到网格无关模型的高效权重迁移与零样本推理。

**🔧 技术方法**

采用连续 B‑spline 核函数、占据归一化、mesh hierarchy 结构、RoPE 位置编码以及 Transformer 处理器；实验中将 CNN 权重直接加载至 MNC encoder/decoder，保持处理器不变。

**📊 数据集**

使用 The Well 数据集（TRL2D、shear‑flow、active‑matter）以及 MeshGraphNets 旋转流动基准；所有数据均为公开仿真网格。

**📈 对比分析**

与 GINO、LNO、Transolver++、MeshGraphNets 等四种状态‑of‑the‑art 无结构网格模型对比；在零样本迁移时 MNC 取得最低 VRMSE，微调后在所有数据集上均快于并低于基线，特别是对活性物质与剪切流动的误差下降 2–3 倍；在大型 Walrus 基础模型的圆柱流实验中，零样本 MNC 也优于除 Transolver++ 之外的所有基线。

**⚠️ 局限性**

主要局限在于改进的是样本效率而非最终精度；对极其稠密或不均匀采样的网格仍需引入几何四积分权重；在充分训练后，与纯无结构模型的差距缩小；未来工作需进一步提升在复杂几何和高维物理场下的性能。

---

## 72. GeopoliticalQA: Benchmarking Retrieval and Generation over Intelligence Reports

**arXiv ID:** 2610.12617 | [PDF](https://arxiv.org/pdf/2610.12617v1)

**作者:** Veronika Solopova `[一作]`, Ostap Vykhopen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了一个名为GeopoliticalQA的基准，收集415份公开地缘政治与战略情报报告，自动生成5,525条文档关联问答对。

**💡 创新点**

创新点在于：①使用文档感知的生成流水线，先对报告做结构化分析再产生问答；②引入LLM作为判定器，对大规模生成结果进行质量评估；③发现中等规模开放嵌入模型（bge-m3）在文档级检索与块级检索上可与更大模型及商业API匹敌或超越。

**🔧 技术方法**

主要技术包括：多模型LLM判定器（GPT‑5.4）、结构化文档分块与检索（BM25 + 余弦密度融合），以及基于vLLM托管的Gemma LLM进行问答生成。

**📊 数据集**

使用的数据集是从122家公开机构收集的415份报告（共55,353块），涵盖政策简报、情报评估、叙事追踪等多种文档类型。

**📈 对比分析**

对15种检索系统与6种开源LLM阅读器进行基准评测。检索层面：所有系统在文档级召回均≥93%；在块级nDCG@10上，bge‑m3与Qwen3‑Emb‑8B表现最佳，且与BM25无显著差距；阅读层面：Llama‑3.3‑70B与Mistral‑Small‑3.2在整体评分上并列最佳，回答质量随问题难度下降。

**⚠️ 局限性**

局限性包括：仅限英语、近年（94% ≥2022年）文献；问答生成和评价均为模型驱动，缺乏独立第三方人工验证；基准未覆盖多语言、多时点或历史资料；评测仅针对单轮检索和阅读，未考虑迭代或多模态扩展。

---

## 73. Static Gradient Attribution Underperforms a Density-Matched Random Mask on Loss-Based Forgetting Within LoRA B-Matrix

**arXiv ID:** 2610.12744 | [PDF](https://arxiv.org/pdf/2610.12744v1)

**作者:** Yu Sun `[一作]` (University of Pennsylvania), Hailu Xu `[通讯]` (California State University, Long Beach)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了在LoRA参数化的连续学习中使用梯度归因选取遮掩子集与直接随机选取遮掩子集的效果，发现随机遮掩在损失反向迁移（loss‑based BWT）上更优；

**💡 创新点**

识别并消除“B=0”导致的梯度归因混淆，并通过匹配参数池与稀疏度的对比实验证明随机遮掩在连续学习中优于梯度归因，同时量化去除归因带来的计算节省；

**🔧 技术方法**

使用LoRA低秩参数化、梯度归因（梯度幅值）、统一随机遮掩、top‑k 10%遮掩、loss‑based BWT度量、七个随机种子、Qwen2.5‑3B 4‑bit NF4量化、AdamW、训练/校准脚本；

**📊 数据集**

采用CodeAlpaca‑20k、MedQA‑USMLE‑4‑options、LexGLUE/CaseHold、GSM8K、Creative_Writing‑ShareGPT五个任务，校准使用64个Alpaca样本；

**📈 对比分析**

在相同B池且稀疏度相同的设置下，Random‑B（随机遮掩）在七个种子上平均降低0.155的loss‑based BWT，优于Grad‑B；准确率方面差异不显著；归因过程消耗12.8%训练预算（64步前向后向校准）；

**⚠️ 局限性**

仅针对静态遮掩、单一任务顺序、零初始化LoRA、rank‑16设置，缺乏对其他模型/架构、动态遮掩、不同任务顺序、非零初始化等情况的验证；准确率退化结果仍不确定。

---

## 74. VortexSynth: Profile-guided benchmark reconstruction for GPU architecture research

**arXiv ID:** 2610.12661 | [PDF](https://arxiv.org/pdf/2610.12661v1)

**作者:** Huanzhi Pu `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6892 | [OpenAlex ID](https://openalex.org/A5000822269)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过Profile‑guided benchmark reconstruction框架，利用LLM自动生成并迭代优化CUDA kernel，生成与Vortex GPGPU兼容的26个benchmark对，构建可迁移的机器人学习工作负载基准。

**💡 创新点**

创新点包括：①无源代码情况下通过profile-guided重构生成可执行GPU kernel；②将LLM生成与输出等价和profile距离反馈结合成迭代改进循环；③提供完整的Vortex/CUDA benchmark套件与可复现的验证流程。

**🔧 技术方法**

技术手段包括：LLM代码生成、profile-guided refinement、输出等价检查、Nsight Compute/Systems的性能指标采集、Vortex GPGPU编译链、CUDA+SIMT、Tensor Core的模拟与验证。

**📊 数据集**

使用的工作负载为机器人学习领域的ACT、Diffusion Policy和π_0三大模型，在NVIDIA H200与Jetson AGX Orin GPU上进行profiling。

**📈 对比分析**

通过对CUDA与Vortex实现的输出等价检测以及计算十项Nsight指标的signature distance进行比较；在迭代过程中，平均profile距离从约0.24降至0.08，显著提升与原始CUDA行为的一致性。

**⚠️ 局限性**

局限性包括：仅针对单核GPU kernel，无法覆盖完整应用；LLM生成质量受模型限制，需人工审查；跨GPU差异导致Vortex-first优化效果不佳；对某些硬件敏感指标仍存在残差。

---

## 75. Scaling Multi-Teacher Distillation for Digital Pathology

**arXiv ID:** 2610.12574 | [PDF](https://arxiv.org/pdf/2610.12574v1)

**作者:** Sofiène Boutaj `[一作]` (Université Paris-Saclay), Stergios Christodoulidis `[通讯]` (Université Paris-Saclay)

**通讯引用:** 3667 | [OpenAlex ID](https://openalex.org/A5064662363)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多教师蒸馏框架下提出OMNI模型，通过学习教师特定表示和结构化注意力等技术，将10个医学影像基础模型的知识压缩为高效学生网络，适用于数字病理图像。

**💡 创新点**

采用教师专属token、结构化注意力掩码、Mixture-of-Experts层以及InfoNCE对比损失，突破传统多教师蒸馏的扩展瓶颈，实现10名教师的可扩展蒸馏。

**🔧 技术方法**

使用Vision Transformer基础、教师特定token、结构化注意力掩码、MoE、InfoNCE对比学习、均值池化聚合以及ABMIL等方法。

**📊 数据集**

在1千万TCGA切片块上预训练，并在THUNDER（13个tile级任务）和PathoBench（7癌种25个slide级任务）进行评估。

**📈 对比分析**

与10个教师模型、H0-mini、GPFM及vanilla MTD对比，OMNI-L等模型在tile级F1达87.8%和slide级BACC 63.5%，比教师模型高且计算成本低10倍；同时在对抗鲁棒性和校准上表现最佳。

**⚠️ 局限性**

仅采用均值池化聚合，未探索教师特定密集表示；需在更多临床数据上进一步验证；目前模型仍基于公开数据，尚未满足所有临床部署需求。

---

## 76. r/ADHD: Community-Based Understandings of Time Management Challenges in Digital Environments

**arXiv ID:** 2610.12629 | [PDF](https://arxiv.org/pdf/2610.12629v1)

**作者:** Shizhao Liu `[一作]` (Santa Clara University), Kai Lukoff `[通讯]` (Santa Clara University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对r/ADHD社区帖子进行大规模主题分析，揭示了 ADHD 用户在数字环境中时间管理的主要挑战与自我调节策略；

**💡 创新点**

创新点在于采用人工指导+LLM辅助的模板式主题分析方法，将传统质性研究与大规模文本分析相结合，系统挖掘社区成员的真实经验，并对工具失败与情绪成本进行细致阐释；

**🔧 技术方法**

使用了GPT‑5.4进行主题标注，并结合人类编码器对主题边界和代码库进行反复校正，形成最终的五个挑战主题与九个策略主题；

**📊 数据集**

数据集为2020‑2025年间r/ADHD子版块的5,379条相关帖子（原始帖子共815,506条，经过删除、关键词过滤和人工相关性筛选后得到）；

**📈 对比分析**

在可靠性评估中，人工编码与LLM标注的Krippendorff α分别达到.81（挑战）和.72（策略），说明LLM在大规模标注上的可行性；但论文未与现有工具或方法做直接效果对比，主要以主题频次和共现关系呈现发现；

**⚠️ 局限性**

局限性包括：仅分析单一Reddit社区，可能偏向极端案例；依赖自述帖子，缺乏行为测量；主题覆盖度有限，部分主题在验证样本中覆盖不足；未排除潜在AI生成内容；可推广性受限于社区人口统计和版块治理规则。

---

## 77. HOMARD: Hammering Off-chip Memory via Aggregate power Rail Disclosure

**arXiv ID:** 2610.12621 | [PDF](https://arxiv.org/pdf/2610.12621v1)

**作者:** Eliott Quéré `[一作]` (CentraleSupélec), Daniel Gruss `[通讯]` (Graz University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在现代Arm SoC上利用未特权低分辨率的总功率传感器，构建了两种无计时器的DRAM侧信道攻击：一是HOMARD-RE用于逆向完整DRAM地址映射；二是HOMARD-DATA用于恢复从DRAM传输的块的汉明重量及其字节值。

**💡 创新点**

创新点在于将功率侧信道与层级感知的多级oracle相结合，突破传统基于时序的单一同级冲突判别，显著降低样本量；同时利用功率采样的平均化特性实现对数据依赖功率成分的模板化与相关分析，首次实现未特权条件下的DRAM数据泄露。

**🔧 技术方法**

采用的软件技术包括循环放大读取功率信号、线性代数求解DRAM层级选择矩阵、模板化与相关功率分析（CPA），以及通过Spectre-RSB漏洞诱发的预测执行侧信道；硬件侧则依赖于SoC上PMIC或IIO暴露的核心阵列与DQ总线电流。

**📊 数据集**

实验数据集覆盖了AMD ZCU102、Raspberry Pi 5和NVIDIA Jetson TX2三款平台，使用了自生成的64字节秘密块、随机对照块以及多组地址对；在Pi 5上进行的汉明重量与字节恢复分别采集约450和64k次功率样本。

**📈 对比分析**

与传统时序侧信道对比，HOMARD-RE在ZCU102和Pi 5上实现100%准确率，TX2上实现2.15×的采样量缩减；HOMARD-DATA在无预测执行时可达81.5%汉明重量分类准确率，在预测执行下可在约19小时内恢复63/64字节，显示出相较于现有功率或时序攻击的显著性能提升。

**⚠️ 局限性**

局限性包括：需平台在未特权模式下公开核心阵列电流传感器，且对电源信号的低通滤波导致对单次访问缺乏分辨率；数据恢复需要大量迭代（数十至数百小时），受内存块读访问的限制；对DRAM块位置（同通道、同行）的要求以及对不同SoC供电架构的适用性尚待进一步验证。

---

## 78. Tactile Sensing-based Secure and Gentle Grasping of Unknown Objects

**arXiv ID:** 2610.12790 | [PDF](https://arxiv.org/pdf/2610.12790v1)

**作者:** Feng Han `[一作]` `[通讯]` (New York Institute of Technology), Feng Han (New York Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于单通道触觉传感器的在线数据驱动预测控制框架，用于在未知物体上实现安全且温和的抓取。

**💡 创新点**

创新点包括：
• 仅使用单通道FSR+弹性垫实现对多种接触拓扑的自适应感知；
• 通过有限状态机主动在线识别抓取力的安全下限和温和上限；
• 利用高斯过程实时学习抓取交互动力学，并将不确定性嵌入模型预测控制，保证约束满足；
• 整个系统不依赖离线训练或视觉信息，控制频率可达100 Hz。

**🔧 技术方法**

技术手段：
- 触觉感知模块（FSR + 橡胶垫）
- GP回归校准与动力学学习
- 有限状态机（FSM）约束识别
- 可靠性增强的模型预测控制（MPC）与可选线性MPC
- ROS+CPU+GPU实时实现

**📊 数据集**

数据集：
- 2000条FSR-力对标数据用于传感器校准；
- 约1500条抓取过程数据用于在线GP动力学学习；
- 实验对象涵盖铝块、番茄、泡沫、硅胶、蓝莓、硬币、石块、豆腐、棉花糖等多种形态和材质。

**📈 对比分析**

对比方法：与传统PID控制、线性MPC等基线进行实验对比。MPC平均跟踪误差约0.2–0.5 N，PID约1–1.5 N；在各种物体上，MPC显著提升抓取力跟踪精度、避免滑移与破坏；系统实时控制频率为100 Hz，鲁棒性强。

**⚠️ 局限性**

限制：
- 需要假设物体已位于抓手工作空间，缺少视觉搜索与定位；
- 仅测量法向力，无法观测摩擦力与扭矩，约束识别受限于λ参数；
- 接触位置不动态调整，可能导致失触；
- 对极软或极重物体的假设（质量、弹性）可能失效；
- 单通道传感器对复杂接触拓扑的区分仍有限。

---

## 79. Tandem Queues with Hybrid Abandonment in Heavy Traffic

**arXiv ID:** 2610.12716 | [PDF](https://arxiv.org/pdf/2610.12716v1)

**作者:** Cameron Curtis `[一作]` (Northwestern University), Seyed Iravani `[通讯]` (Northwestern University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了混合放弃队列在重负载下的理论性质，证明了第二队列长度的缩放极限为指数分布并与第一队列独立，并通过仿真验证这些结果在非极限情况下的适用性。

**💡 创新点**

创新点在于将IMAP（无限状态马尔可夫到达过程）框架推广到混合放弃两阶段队列，提供了一套完整的重负载极限条件并给出了统一的指数分布与独立性结论，此外还通过仿真展示了该近似在实际参数下的鲁棒性。

**🔧 技术方法**

使用了马尔可夫过程、生成函数与离散时间嵌入链、近似矩阵求逆（R矩阵）以及极限分析（重负载逼近）。在数值验证中采用了大规模离散事件仿真与KS检验、相关系数等统计量。

**📊 数据集**

主要使用自定义的混合放弃双队列仿真数据，覆盖多组参数（μ₁∈{1.1,1.2,1.5,2,4}、θ∈{0.1,0.3,0.5}、ρ∈{0.8,0.9,0.95,0.99}），共60组参数；在容量规划实验中又使用了400个不同的第二队列服务率。

**📈 对比分析**

通过与理论极限公式（E[Q₂]≈M⁽⁰⁾/(1-ρ)）比较，发现平均误差始终在1客以内，误差随负载升高而相对缩小；相关系数始终小于0.1，几乎无相关；KS检验显示在高负载下(ρ≥0.95)指数近似误差可忽略，几何分布在更广范围内误差更小。整体性能优异，近似在实际调度中误差<0.6%。

**⚠️ 局限性**

主要局限包括：结论仅在重负载极限下严格成立，对低负载或中间负载的精度不保证；仿真验证集中在两阶段M/M/1+M队列，未验证更复杂网络；在容量规划中近似成本往往偏高，可能导致保守决策。

---

## 80. Meta-Reflective Evolution Effectively Improves Production AI Systems

**arXiv ID:** 2610.12794 | [PDF](https://arxiv.org/pdf/2610.12794v1)

**作者:** Seung Won Wilson Yoo `[一作]` (Snowflake), Anupam Datta `[通讯]` (Snowflake)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Cortex Code Evolve，一种基于Meta‑Reflection的进化优化器，专为生产级 AI 系统（如智能代理、LLM 驱动的数据流水线和数据工程工作流）设计。

**💡 创新点**

创新点在于：①Meta‑Reflection——从整个搜索历史中选择性检索证据并在子代生成前主动与目标环境交互；②采用可编程编码‑agent（Cortex Code）进行自验证、自修复；③引入进化账本与可查询历史工具，支持停滞检测、范式切换与共回归避免；④将以上能力嵌入 MAP‑Elites、预算化任务调度与相似性引导评估。

**🔧 技术方法**

技术栈包括 LLM 语义变异、MAP‑Elites 质量多样性搜索、进化账本（stagnation ledger）、on‑demand 结构化查询、环境增强（工具链 + 目标系统 API）、自验证器、相似度估计与信息增益驱动的任务选择、预算化的 roster 及多任务迁移。

**📊 数据集**

数据集：DABStep（450 个支付处理问题），AI Functions 基准（Lockpicking、Reddit、法律文本、金融算术、文本规范化），data‑eng‑bench（103 任务、579 表、8000+ 列），数学与系统优化基准（ADRS、Signal Processing、TXN 等），HotpotQA、AIME、IFBench。

**📈 对比分析**

与 GEPA、Meta‑Harness、AdaEvolve、OpenEvolve 等现有方法对比，Cortex Code Evolve 在所有领域均表现出色：DABStep 92.8%（高于 78.4%/30.4%），六个 LLM 数据流水线平均提升 10pp，data‑eng‑bench solve‑rate 提升至 77.4%（高于 73.6%），HotpotQA 0.793（高于 0.683）。匹配实验与消融实验验证每个组件的贡献；在数学与系统基准上与最先进方法相当或更优。

**⚠️ 局限性**

局限性：依赖 LLM 的推理质量与可访问的工具链；部分基准（如 AIME）评估噪声大，难以显现真实收益；需要对目标环境的 API 进行显式集成，部署成本不低；在极大搜索历史或极大任务池下，查询与预算调度的开销需进一步优化；DABStep 公开排行榜尚未外部验证。

---

## 81. MemReg: Streaming Outdoor LiDAR Point Cloud Registration with Hybrid Memory Buffers

**arXiv ID:** 2610.12624 | [PDF](https://arxiv.org/pdf/2610.12624v1)

**作者:** Jiuming Liu `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于多帧 LiDAR 流的点云配准网络 MemReg，通过隐式特征缓冲和显式位姿缓冲存储历史信息，利用长时序编码和动态历史权重提升稀疏、低重叠、动态与遮挡场景的配准精度。

**💡 创新点**

创新点包括：1）混合记忆缓冲（隐式特征缓冲 + 显式位姿缓冲）实现长期历史上下文捕获；2）Mamba 变体实现高效长时序特征交互；3）动态历史权重模块自适应融合历史与当前特征，抑制尺度差异和外点影响。

**🔧 技术方法**

采用 HRegNet 的多层特征提取、Mamba 时序编码、动态历史权重、姿态回归与细化以及配准损失，整体实现端到端学习。

**📊 数据集**

户外数据集：KITTI、nuScenes、Apollo‑Southbay；室内数据集：3DMatch、3DLoMatch、ScanNet，用于评估模型泛化。

**📈 对比分析**

与传统 ICP、RANSAC、DGR 等优化方法以及 HRegNet、RegFormer、SLAM 系统等学习与多帧方法比较，MemReg 在三大户外数据集实现 100% recall，RTE/RRE 分别比前沿方法低约 30‑50%，在室内数据集同样显著提升 recall 与误差，表现为 state‑of‑the‑art。

**⚠️ 局限性**

局限性：记忆长度需手动设定，过长或过短均影响性能；极低频 LiDAR 或极大时间间隔的帧差仍可能导致误差；模型结构相对复杂，对 GPU 计算资源有一定需求。

---

## 82. VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails

**arXiv ID:** 2610.12813 | [PDF](https://arxiv.org/pdf/2610.12813v1)

**作者:** Duy Nguyen `[一作]` (Google Cloud AI Research), Long T. Le `[通讯]` (Google Cloud AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为LAMP的框架，用于在LLM-Agent部署过程中持续演化可验证的正式安全策略，并通过结构化事件表示实现决策的可追溯和可审计；

**💡 创新点**

创新点包括：①将策略演化拆分为生成、验证与授权三层，避免LLM直接决定执行；②将正式策略演化建模为约束的双目标优化，并采用基于Pareto前沿的archive搜索；③引入证据感知授权机制，使规则仅在足够的正负反馈下才进入活跃策略；

**🔧 技术方法**

技术手段包括：LLM生成策略补丁、基于DSL的事件解析、确定性冲突与回归检测、Beta-伯努利模型的规则置信度评估、MAP-Elites式的多样性父策略采样、以及最大最小决策选择；

**📊 数据集**

使用了三大安全与安全性基准：ASSEBench-Security、ASSEBench-Safety 与 ATBench，涵盖注入、未授权访问、数据泄露等多种风险；

**📈 对比分析**

与无记忆LLM、AGrail、LiSA、Hoeffding Tree、RIPPER 等基线对比，LAMP在所有数据集上均取得最高的宏F1；在混合流、噪声反馈及长时序下依旧保持领先，覆盖率达到约80%+；

**⚠️ 局限性**

局限性：仍依赖LLM对策略补丁的生成质量；规则解释虽简单但在极大规模或高度动态的场景中可能需要更高效的搜索与验证；缺乏针对极端攻击场景（如对抗性反馈）的完整保障。

---

## 83. Image-Based Path-Planning for Navigation of Soft-Growing Robots in the Spinal Sub-Arachnoid Space

**arXiv ID:** 2610.12631 | [PDF](https://arxiv.org/pdf/2610.12631v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 84. Incremental Retrieval-Augmented Generation for Human-in-the-Loop Decision Support

**arXiv ID:** 2610.12666 | [PDF](https://arxiv.org/pdf/2610.12666v1)

**作者:** Federico Mazzoni `[一作]` (Independent Researcher), Andrea Failla `[通讯]` (Institute of Information Science and Technologies, National Research Council)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 iRAG，一种在知识库从零开始逐步填充、结合检索增强生成与人机协作决策的系统。

**💡 创新点**

将检索增强生成与混合决策（Hybrid Decision-Making）结合，采用逐步递增的知识库、FAE 驱动的状态迁移（观测→争议→自主），并加入检索衰减与可靠性衰减的实验对比。

**🔧 技术方法**

使用大语言模型（LLM）作为生成器和辅助判定模型，向量检索 + 时间衰减、FAE（Fading Empirical Accuracy）估计、三态控制器（SO、SC、DS）以及人机交互机制。

**📊 数据集**

基于合成客户支持票据数据集，四个季度各 500 票，包含 10% 或 40% 概念漂移票据，模拟三种人类决策者（CEO、Domain Expert、Intern）。

**📈 对比分析**

通过与无衰减的独立控制器自由 RAG-with-defer 基线对比，采用 10 次重复实验，评估最终决策错误率、稳定/漂移票据错误、模型最终化比例等指标；在低漂移时与基线相当，高漂移时误差降低 3–4 个百分点。

**⚠️ 局限性**

依赖人工专家 bootstrap、阈值手工设定、FAE 与检索衰减的调参，辅助判定模型与主模型共享，实验仅使用模拟人类，未验证真实人机交互。

---

## 85. PVBRF-ID: Private, Verifiable and Byzantine-Robust Federated Intrusion Detection for Oilfield Enterprise Networks

**arXiv ID:** 2610.12704 | [PDF](https://arxiv.org/pdf/2610.12704v1)

**作者:** Mustafa S. Aljumaily `[一作]` (Daw Alfada Company), Shahab Abdulla `[通讯]` (University of Southern Queensland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在油田分布式场景下，提出并实现了 PVBRF-ID——一种双服务器秘密共享聚合协议，用于在不泄露单站流量的前提下训练共享入侵检测模型，并在聚合过程中通过根集参考的余弦相似度和范数阈值过滤潜在的投毒更新。

**💡 创新点**

创新点在于：①将根集（小型干净数据集）产生的参考方向与安全聚合结合，形成一次性接受位的隐私友好判断；②在两台非合谋服务器之间公开可验证的 Pedersen 承诺、Merkle 根与审计信息，使站点能够独立验证其更新是否被纳入；③引入容差的余弦相似度测试，兼顾鲁棒性与对异构站点的公平性。

**🔧 技术方法**

核心技术包括秘密共享（每站将梯度拆分为两份）、Pedersen 承诺与 Merkle 树、理想化安全比较与 Beaver 乘法三元组、可选的安全裁剪与高斯噪声加入、以及对更新范数与参考余弦的容差检测。

**📊 数据集**

使用公开的 Edge‑IIoTset 数据集（15 类网络流，约 2.2M 条流）进行实验，模拟 10 个基于 Dirichlet 分布的油田站点，覆盖不同的攻击类型与类不均衡情况。

**📈 对比分析**

通过宏 F1 与安全性指标与 FedAvg、Trimmed mean、multi‑Krum 等基线对比，结果显示：在无攻击时宏 F1 与 FedAvg 差距 <0.02；在 30% 适应性投毒攻击下，PVBRF-ID 仍保持宏 F1 ≈0.64，而 FedAvg 下降至 0.046；在 sign‑flip、label‑flip 等非适应攻击中亦表现出优越或相当的鲁棒性。

**⚠️ 局限性**

主要局限包括：①通信成本约为 FedAvg 的 19–27 倍，需要双服务器非合谋；②根集参考可能被攻击者捕获，导致聚合失效；③后门攻击未被完全阻止，噪声防护仅为启发式且隐私预算过大；④安全比较、范围证明等关键组件在实现中仍保持理想化，实际部署需进一步优化。

---

## 86. Graph-Based Heuristics for Minimum Constraint Removal in Cluttered Robot Motion Planning

**arXiv ID:** 2610.12609 | [PDF](https://arxiv.org/pdf/2610.12609v1)

**作者:** Antony Thomas `[一作]` (IIIT Hyderabad), Marco Baglietto `[通讯]` (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于图的启发式管道，专门求解机器人在可移动障碍环境中的最小约束移除（Minimum Constraint Removal）问题；

**💡 创新点**

创新点在于引入双向可行集生成器（Φ）与基于减少图的路径长度限定搜索相结合的细化步骤，并提出扩展生成器（Λ）以覆盖生成器缺失最优解的情况；

**🔧 技术方法**

核心技术包括配置空间离散化构建连通细胞图、双向扩展产生候选移除集合、在子图上按长度约束执行最小卡尔曼集搜索，并采用基于可移除集的子集主导剪枝；

**📊 数据集**

实验使用六个基准，包括五个二维平面障碍场和一个四自由度抓取任务，所有基准均使用预构建的细胞图进行比较；

**📈 对比分析**

与精确最佳优先搜索和贪心搜索对比，所提方法在三项任务中得到最优解，平均解卡尔曼比例为1.11，最快可达33.7倍速度，贪心搜索则更快但解质量显著下降；

**⚠️ 局限性**

局限性在于缺乏最坏情况近似保证、对所有障碍赋予等价成本且未规划障碍位移动作，导致在实际操控时可能需后续复杂重排。

---

## 87. SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire

**arXiv ID:** 2610.12596 | [PDF](https://arxiv.org/pdf/2610.12596v1)

**作者:** Scott C. Lowe `[一作]` (Vector Institute), Sander W. Keemink `[通讯]` (Radboud University)

**通讯引用:** 477 | [OpenAlex ID](https://openalex.org/A5091367014)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建SpireBench，使用完整的商业游戏Slay the Spire在最高难度下让LLM代理做所有决策，并评估20个大语言模型。

**💡 创新点**

首次提供完整游戏交互 harness，结合全局reskin以区分先验知识和推理，系统性检验LLM的长期决策与记忆效果。

**🔧 技术方法**

利用工具调用、多动作响应、多模态记忆库以及多种成本/推理设置，对LLM进行交互评测。

**📊 数据集**

使用10个随机种子、Ascension 20难度Ironclad游戏日志、Mega Crit 200k人类数据和4名专家主播记录。

**📈 对比分析**

与基线、符号模型、平均人类及专家数据对比；最强模型在第一幕能与人类相当，随后随时间差距显著扩大，最终仅有0–10%的完整通关率。

**⚠️ 局限性**

实验仅限单角色、基于文本的回合制游戏，种子数量有限，记忆验证缺失，且人类对照未匹配种子。

---

## 88. Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study

**arXiv ID:** 2610.12542 | [PDF](https://arxiv.org/pdf/2610.12542v1)

**作者:** Yuyu Zhang `[一作]` `[通讯]`, Yuyu Zhang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在相同搜索预算下对比MCTS调度与随机重启列表调度的性能，探索MCTS是否能在DAG工作流调度中超越无树方法。

**💡 创新点**

首次采用先验注册、等预算对照、独立验证和失败分类法构建可重复的评估协议，并给出可复用的评估流程。

**🔧 技术方法**

使用自定义选择规则的MCTS、标准UCT实验、随机重启列表调度、CPOP预热、四启发式基准（HEFT、CPOP、PEFT、OCT）以及统计分析（Hodges‑Lehmann、Bootstrap、Wilcoxon）。

**📊 数据集**

合成DAG实例（N=8,16,32；P=2,4,8；CCR=0.5,2），六类结构（链、分叉‑合并、分层、关键路径重、稠密、稀疏宽），共96个确认集；探索阶段使用更大实例集。

**📈 对比分析**

在确认集上以预算2048评估，每个实例计算Δ=(control‑MCTS)/control；结果显示MCTS平均落后2.88%（95%CI [-3.62%,-2.15%]），在fork‑join和sparse‑wide两类结构显著落后；用标准UCT替代后差距缩小至-0.26%但仍未超越。

**⚠️ 局限性**

仅在合成整数成本、单速处理器、预算≤2048的域内实验；热启动不对称、使用非标准选择规则；未验证异构或真实工作流、未测试更大预算或学习导向调度。

---

## 89. Reflective Prompt Optimization of a Deep-Research Agent: A Topic-Disjoint, Multi-Seed Protocol for Measuring Transferable Gains and Prompt-Level Overfitting

**arXiv ID:** 2610.12601 | [PDF](https://arxiv.org/pdf/2610.12601v1)

**作者:** Elias Lumer `[一作]`, Kevin Paul `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用反射式提示优化（GEPA）改进多模态深度研究代理的系统提示，从而提升引用奖励。

**💡 创新点**

提出了“主题无关留出评估协议”，通过主题分离、可重复种子、验证器消融等手段分离验证提升与记忆效应，验证去记忆提升的真实性。

**🔧 技术方法**

采用GEPA进行进化搜索、分层奖励设计（链接有效性、相关性、事实支持、编号），以及多种评审器（精细分维度反馈、粗粒度与标量化）。

**📊 数据集**

实验基于同一新闻语料库的“深度研究”代理，训练验证集24题（12主题），留出测试集500题（254主题）。

**📈 对比分析**

与种子提示比较，留出提升平均0.079（置信区间[0.051,0.107]），多种子均显著；验证提升更大（0.202–0.290），说明验证误差；拆解奖励显示事实支持提升最大，手写“规则”能恢复约57%的留出收益。

**⚠️ 局限性**

局限包括：验证集样本量小导致赢家诅咒；评审器为单一提供商，需更多跨供应商验证；仅在单一任务/域验证，泛化性待验证；未对记忆化优化器施加约束，记忆内容仍出现；评审噪声与小样本选择导致验证–留出差距波动大。

---

## 90. SWE-Duel: Towards a Self-Scaling Arena for Coding Agents

**arXiv ID:** 2610.12569 | [PDF](https://arxiv.org/pdf/2610.12569v1)

**作者:** Wei Minn `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 33483 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个自扩展的编码竞技场中让两种语言模型互相创建、评估、修复代码任务，构建了1,500轮的对弈赛。

**💡 创新点**

提出了SWE‑Duel——自生成任务、封闭bug测试、并通过双方角色交替的对弈方式实现无泄露、难度自适应的编码基准。

**🔧 技术方法**

利用大型语言模型（GPT‑5.5、Claude Opus 4.8 等）配合不同工具包（mini‑swe‑agent、Codex CLI 等）以及自定义的验证门控和 Docker 环境。

**📊 数据集**

在15个开源仓库（Python、Java、Go、JavaScript、C/C++）上进行实验，生成多语言任务，检验模型能力。

**📈 对比分析**

通过 Bradley–Terry 排名对比，GPT‑5.5/Codex CLI 以98.3% 的缺陷修复率位列第一，Claude Opus 4.8/mini‑swe‑agent 以27.3% 的红方胜率成为最佳任务创作者；整体蓝方成功率约87.9%。

**⚠️ 局限性**

局限包括任务自我修复门控可能排除难度更高的任务、封闭测试可能产生误报、对较弱模型的自动胜利导致对弈不完整、以及对比仅限于当前参与者和仓库。

---

## 91. Neuromorphic Satellite Intelligence: A Formal Framework for Event-Driven Architectures, Spike-Based Algorithms, and Quantitative Energy-Accuracy Trade-offs

**arXiv ID:** 2610.12471 | [PDF](https://arxiv.org/pdf/2610.12471v1)

**作者:** Suresh Kumar TP `[一作]` (Mangalayatan University), Vikas Agnihotri `[通讯]` (Independent Researcher)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套针对卫星任务的完整脑突触式计算框架，包括能耗模型、三种专用算法和可控STDP学习规则。

**💡 创新点**

创新点在于基于硅测量的闭式能耗模型、专为卫星设计的SKCF、TCAD、SAMC算法以及可证明权重漂移上限的STDP规则。

**🔧 技术方法**

使用脉冲神经网络、事件驱动架构、Radiation‑Hard neuromorphic芯片（Loihi, SpiNNaker等）以及自适应滤波、异常检测和调制解调器技术。

**📊 数据集**

采用Loihi、SpiNNaker的实验数据、模拟遥测和通信链路数据以及标准地球观测影像数据集进行验证。

**📈 对比分析**

通过与FPGA（Xilinx Kintex UltraScale）和GPU（Jetson AGX Xavier）基准对比，能耗降低8–47×，在保持约98–99%准确率的同时，延迟相当或更优。

**⚠️ 局限性**

局限性包括对稠密算术任务不适用、需进一步发展空间硬化硅和认证流程、异步脉冲与传统卫星总线的集成成本等。

---

## 92. From Instructional Intent to Classroom-Ready Lessons: Investigating Teacher-AI Collaboration in Lesson Authoring for Secondary Computing Education

**arXiv ID:** 2610.12834 | [PDF](https://arxiv.org/pdf/2610.12834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 93. Recurrent Harness for Agentic Continual Learning

**arXiv ID:** 2610.12668 | [PDF](https://arxiv.org/pdf/2610.12668v1)

**作者:** Yuxin Xiao `[一作]` (Massachusetts Institute of Technology), Bailin Wang `[通讯]` (Meta MRS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReHarness 框架，在仅使用前向信息流的传统 agent harness 基础上引入后向修订机制，实现在线持续学习的循环更新

**💡 创新点**

创新点在于利用后向修订（Backward Revision）通过下游实例反馈对已存储的反思（Reflection）进行源文本保持（source‑grounded）地修正，闭合信息流循环，显著提升在分布漂移环境下的稳定性与可塑性

**🔧 技术方法**

技术包括：基于大型语言模型的决策代理、记忆代理与修订代理；检索式回溯（k‑retrieval + 修订）；反思生成与修订的文本化对话；使用多轮实例轨迹做评估

**📊 数据集**

在三个持续学习基准上进行评估：CL‑Bench（多任务分布漂移），MemoryBench（长文本生成），MemoryArena（跨领域任务）；使用三种前沿 LLM：Claude Opus 4.8、GPT‑5.5、Gemini 3.7 Flash

**📈 对比分析**

与六类基线对比（Mem0、RBank、DoT、ACE、GEPA‑Online、Full Raw/Top‑K Raw），在全部 9 个 benchmark‑backbone 组合中 ReHarness 在 8 组取得最高或相当最佳的归一化增益（最高 0.240），且差异在统计上显著；在 Token‑效率上也优于 Full Raw，并在大部分任务上保持较低成本

**⚠️ 局限性**

局限性包括：对奖励信号噪声高的任务（如 CL‑Bench 的 exploitable poker）效果不佳；仍需存储原始轨迹以供修订，导致存储开销；未涉及模型参数更新，仍依赖 LLM 生成质量

---

## 94. Data-driven Optimal Sensor Placement for Aerosol Monitoring in the UK

**arXiv ID:** 2610.12658 | [PDF](https://arxiv.org/pdf/2610.12658v1)

**作者:** Talha Ansar `[一作]` (University of Engineering and Technology), J. Scott Hosking `[通讯]` (Alan Turing Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于ConvGNP与DeepSensor框架的气溶胶监测传感器布设方法，并通过遗传优化与贪心搜索进行比较。

**💡 创新点**

创新点在于将预训练的ConvGNP嵌入DeepSensor，利用遗传算法的惰性评估与缓存实现高效的递进式布设，并评估气候条件对布设效果的影响。

**🔧 技术方法**

使用卷积高斯神经过程（ConvGNP）、DeepSensor优化框架、遗传算法、Spearman相关性分析等技术。

**📊 数据集**

数据集包括英国CAMS再分析的黑碳气溶胶光学深度、AERONET观测站点数据以及相应的风速风向信息。

**📈 对比分析**

与贪心搜索相比，遗传算法将平均计算时间从约25±7分钟降低至10±3分钟（60%缩减），在五个新增站点下Joint NLL提升约22%，预测性能保持一致。

**⚠️ 局限性**

局限在于模型不确定性与实际预测收益的相关性有限，且相关性随风向变化，需在更多气候传输条件下进一步验证。

---

## 95. PanTact: Teaching Whole-Body Dexterity with a Wearable Contact Interface

**arXiv ID:** 2610.12558 | [PDF](https://arxiv.org/pdf/2610.12558v1)

**作者:** Xiaomeng Xu `[一作]` (Stanford University), Guanya Shi `[通讯]` (Amazon FAR)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用机器人自由的人类演示学习类人机器人全身抓握与操作能力。

**💡 创新点**

将接触信息作为跨体型的任务指令；使用模块化可穿戴触觉贴片记录接触；设计了体现无关的视觉运动策略和接触感知；提出三种全身控制方案实现接触目标。

**🔧 技术方法**

可穿戴触觉模块、头戴立体相机+手腕相机、UMI双手抓取、模糊视觉+深度图编码器、扩散式策略、IK/SONIC/稀疏跟踪控制、虚拟接触目标等技术。

**📊 数据集**

收集了100-150个机器人自由的人类演示，使用头戴摄像头、手腕摄像头和模块化接触贴片记录；在Unitree G1上进行硬件实验。

**📈 对比分析**

与仅预测EEF轨迹的Humanoid UMI、以及预测下肢姿态的Humanoid UMI++对比，任务包括抓握-拥抱、扫移-柜闭、携带-推送；在四个能力维度上，系统实现80-90%成功率，比基线提升约10-20%。

**⚠️ 局限性**

受限于人体与机器人比例差异导致的接触匹配、接触目标对控制器约束的平衡、训练所需的模拟和域随机化、对高动态姿态的鲁棒性不足。

---

## 96. Enriched Holomorphic Neural Networks for 2D and 3D Laplace Problems

**arXiv ID:** 2610.12541 | [PDF](https://arxiv.org/pdf/2610.12541v1)

**作者:** Kalp Shah `[一作]` (University of Amsterdam), Alejandro Valverde Mahou `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种将Laplace方程解析结构嵌入神经网络的奇异性增强全同构神经网络，用于求解具有角点、边缘和接口的边界值问题。

**💡 创新点**

创新点在于利用全同构网络天然保持谐波性质，并在此基础上加入可训练指数的局部奇异函数字典，实现对角点、边缘和接合点的精确奇异捕获，从而消除内部残差。

**🔧 技术方法**

技术上结合了全同构神经网络、奇异函数字典、可训练指数、三维零向量复投影、Adam+L‑BFGS优化以及对边界采样的无内部残差损失。

**📊 数据集**

实验数据集包括六个二维基准（扇形、L形、开槽盘、Motz问题、裂纹梁、CZT探测器截面）和六个三维基准（L棱镜、两种沿边变化场、光滑立方体、锥形缺口、扩展Motz、3D CZT探测器）以及一组随机采样问题。

**📈 对比分析**

与MLP PINN、纯全同构网络、Deep Ritz、WAN以及其他无网格方法比较，奇异性增强模型在大多数测试中提高了3–6个数量级的精度，并在CZT探测器案例中比最佳基线低1.3%至7.5%（相对误差）。

**⚠️ 局限性**

局限性包括当域内存在大量共享接合点、边缘振幅沿轴变化或几何不在字典覆盖范围时性能下降；奇异指数训练对样本分布敏感；以及在高度耦合的多接合点环境下数值条件可能恶化。

---

## 97. Structural and Representational Agency in LLM-Driven Timbre Generation and Interpolation

**arXiv ID:** 2610.12532 | [PDF](https://arxiv.org/pdf/2610.12532v1)

**作者:** Joseph Cameron `[一作]`, Alan Blackwell `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一种基于语言模型的音色生成工具进行测量研究，评估其界面显示与实际音频之间的误差，并探讨有效代理与认知代理的分离；

**💡 创新点**

首次将有效代理与认知代理区分开来，证明界面显示往往是模型自报导致误差，给出误差比例、端到端实验结果并提出审核协议；

**🔧 技术方法**

使用两种语言模型（本地 LLaMA 与 OpenAI GPT‑4）、AudioCommons 音频特征回归模型、可达集（zonotope）分析、坐标线性插值误差分解以及音频渲染与测量技术；

**📊 数据集**

利用10,292个合成器预设的 AudioCommons 参考语料、52个单词音色概念的156个音频样本，以及三幅实际用户绘制的声色画布；

**📈 对比分析**

通过比较模型预测坐标与实际测量坐标的相关性和误差；强模型相关性 r=0.729、误差0.375；弱模型 r=0.357、误差0.515；端到端实验36次，强模型有9/18指令产生可听变更，弱模型几乎无变更；显示误差约为可达空间宽度的3倍；

**⚠️ 局限性**

未进行用户研究，样本量小；仅针对单一合成器与特征空间；使用回归模型的音频特征而非直接感知标注；测量误差阈值有限制；

---

## 98. NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for Multivariate Time-Series Anomaly Detection

**arXiv ID:** 2610.12754 | [PDF](https://arxiv.org/pdf/2610.12754v1)

**作者:** Aditi Gautam `[一作]` (NVIDIA), Sanjeev Roka `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了NV‑Tesseract‑AD，一种基于Transformer的条件扩散模型，利用双轴课程学习（遮蔽比例与噪声级别）对多变量时间序列进行无监督异常检测。

**💡 创新点**

创新点在于：①双轴课程学习将遮蔽难度与扩散噪声级别联合排序，实现遮蔽不敏感的重建；②在单一模型中融合扩散、Transformer与课程机制；③通过零射击迁移验证模型在跨域任务中的鲁棒性。

**🔧 技术方法**

核心技术包括条件DDPM、残差Transformer、分阶段难度调度、噪声预测损失、速度加速采样（DPM‑Solver++）、梯度裁剪与自监督阈值校准。

**📊 数据集**

训练集使用TSB‑UAD‑M（TSB‑AD衍生），测试/零射击集涵盖EasyTSAD十个家族、NAB、NormA等工业与公共数据集。

**📈 对比分析**

与无课程基线对比，F1提升约3.7×、MSE下降56%；零射击在NAB/NormA获得F1≈0.98；相比传统AE、GAN或单阶段扩散检测器，表现显著更优；采样加速后推理耗时下降约40×。

**⚠️ 局限性**

主要限制为：无监督阈值校准仍是性能瓶颈；采样方式会改变残差分布，需与阈值共同调优；在极端缺失率或标签分布差异大的数据集上表现相对较弱。

---

## 99. REMM: Revisable Embodied Memory for Manipulation

**arXiv ID:** 2610.12710 | [PDF](https://arxiv.org/pdf/2610.12710v1)

**作者:** Revanth Krishna Senthilkumaran `[一作]` (Carnegie Mellon University), Jonathan Francis `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了可修订的具身记忆框架，以在冻结的G0.5执行器上维护任务状态并在物理干预时撤销不再支持的结论。

**💡 创新点**

创新点是将执行器的历史与可撤销任务状态分离，并用任务语义生成的支持图链接物理关系与完成条件，实现基于证据的自动修复。

**🔧 技术方法**

使用冻结的G0.5视觉-语言-动作执行器、Qwen3-VL-4B-Instruct和OWLv2进行任务分解与视觉验证，配合 append‑only 账本、支持图和恢复调度。

**📊 数据集**

实验采用 51 次真实机器人对运输、序列回忆和隐藏物体等长周期任务的物理试验，并与生成视频基准对比。

**📈 对比分析**

相较于 HiMe、MemER、RoboMME 等基线，平均任务进度从 57% 提升至 91%，在序列回忆任务中完全地面化决策达 8/12，表现优于基线。

**⚠️ 局限性**

局限在于依赖预定义的任务语义和手工构造的支持图，未学习因果结构，受限于感知误差、固定执行器、单一机器人与物体集，且修复只能在技能边界进行。

---

## 100. Kernelized Parameter Sensitivity Atlases for Amortized Conditional Interpretability

**arXiv ID:** 2610.12737 | [PDF](https://arxiv.org/pdf/2610.12737v1)

**作者:** Davide Wiest `[一作]`, Abdul Monaf Chowdhury `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

定义了一系列用于绘制流程图、电路图等图形的TikZ样式

**💡 创新点**

通过统一、可复用的样式定义，提升绘图的一致性和美观度

**🔧 技术方法**

使用LaTeX的TikZ宏包，并自定义样式属性（如线宽、颜色、箭头等）

**📊 数据集**

无数据集，纯样式定义

**📈 对比分析**

未进行实验或性能对比，主要是图形渲染的视觉效果

**⚠️ 局限性**

缺乏实际应用案例、缺少性能评估或与其他绘图工具的比较

---

## 101. Fear Is Not Always Fear: Variation in Emotion Label Equivalence Across Datasets

**arXiv ID:** 2610.12841 | [PDF](https://arxiv.org/pdf/2610.12841v1)

**作者:** Emily Öhman `[一作]` (Waseda University), Anna Koufakou `[通讯]` (Florida Gulf Coast University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究不同情绪标签资源的跨资源兼容性，使用共享编码器与资源特定输出头进行同输入比较与跨语料迁移实验，评估标签等价性与可修复性。

**💡 创新点**

提出L1–L4标签等价性框架，构建同输入兼容性评估与跨资源迁移的系统，对概率尺度、排名一致性与校准可修复性进行细粒度量化。

**🔧 技术方法**

RoBERTa‑base预训练编码器、共享‑头结构、Spearman秩相关、MAE、AUROC、AP、F1、Brier分解、Platt、isotonic、EM预先校准等技术。

**📊 数据集**

GoEmotions、XED、BRIGHTER共3个英文情感语料库（各自多标签或8类标签），外部SemEval‑2018 Twitter集合进行跨语料评估。

**📈 对比分析**

通过同输入排名相关和概率误差比较资源间头部的尺度与排名差异；跨语料迁移时平均AUROC0.71‑0.85、F1下降至0.5以上，误差中约55%为判别损失；校准后Brier误差下降至0.02‑0.03，校准不足以弥补排序不一致。

**⚠️ 局限性**

仅针对3个英文资源、5个共享标签、RoBERTa‑base模型；实验单位有限，结果主要描述性；未对标签多义性进行潜变量模型；缺乏跨语言或跨架构验证。

---

## 102. Onions in the ATmosphere: Decentralizing Trust in Identity

**arXiv ID:** 2610.12764 | [PDF](https://arxiv.org/pdf/2610.12764v1)

**作者:** Paul Syverson `[一作]` (U.S. Naval Research Laboratory), Nick Gerakines `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了将域名与洋葱地址以及Bluesky handle与DID关联的自验证有意义身份（SAMI）的组合机制，并将其应用于Smoke Signal等应用；

**💡 创新点**

创新点在于融合两种SAMI方式，利用CT/PLC日志实现透明性，提供去中心化的身份验证和上下文信任，并开发了首个双向DID验证工具；

**🔧 技术方法**

采用了RFC 9421 HTTP消息签名、TLS证书SAN、DID-doc的相互引用、Tor onion服务、AT协议的PLC/handle解析、WebExt浏览器插件等技术；

**📊 数据集**

主要使用的实验数据为Smoke Signal的DID文档、Bluesky账户、Tor网络中的onion地址及公共CT/PLC日志；

**📈 对比分析**

与传统仅使用DNS/TLS或仅使用DID的方案相比，本文通过CT/PLC日志和浏览器插件实现了更快的发现与验证，性能上可在几秒到几分钟内完成关联验证；

**⚠️ 局限性**

局限性包括对Tor网络和浏览器扩展的依赖、需要用户手动配置信任根、未实现完整的上下文标签自动化、以及对现有AT协议和CA支持的持续性依赖。

---

## 103. Lightweight Systems for Spatial Semantic Segmentation of Sound Scenes

**arXiv ID:** 2610.12693 | [PDF](https://arxiv.org/pdf/2610.12693v1)

**作者:** Mayank Mishra `[一作]`, Romain Serizel `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究轻量化的空间语义分割（S5）系统，替换原始大型标记器M2D和分离器ResUNet为DyMN标记器和SuDoRMRF分离器。

**💡 创新点**

证明轻量化模型（DyMN-full + SuDoRMRF-16）在保持与原始基线相近的CA‑SDRi的同时，参数量降低约97%且MACs降低约2.5倍。

**🔧 技术方法**

采用卷积神经网络（DyMN）进行音频标记，使用多分辨率卷积网络SuDoRMRF并加入FiLM进行目标源提取。

**📊 数据集**

使用DCASE 2025任务4的开发数据集（18类音频事件）及AudioSet预训练权重。

**📈 对比分析**

通过混合准确率、CA‑SDRi以及模型参数/计算量等指标进行对比，结果显示轻量化模型在保持性能的同时显著减少资源占用。

**⚠️ 局限性**

仍存在模型在极端边缘设备上资源不足、标签器性能受限于训练样本量，以及需要进一步压缩的空间。

---

## 104. LeWAM: End-to-End World and Action Modeling with JEPAs

**arXiv ID:** 2610.12818 | [PDF](https://arxiv.org/pdf/2610.12818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 105. Algebraic Integer Quantization for Machine Learning Applications

**arXiv ID:** 2610.12494 | [PDF](https://arxiv.org/pdf/2610.12494v1)

**作者:** Anthony Kostal-Vazquez `[一作]` (Lemurian Labs Research Group), Branden Butler `[通讯]` (Lemurian Labs Research Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出代数整数量化（AIQ）用于大语言模型推理，构建基于代数整数的矩阵乘法引擎并进行量化实验

**💡 创新点**

将代数整数作为精确整数算术的压缩格式，设计可配置的AIQ量化方案，并证明其在低比特宽度下与MX格式同等或更优精度，同时利用硬件矩阵核实现无误乘加

**🔧 技术方法**

代数整数数制、整数矩阵乘法、Karatsuba算法、Tensor Core/Matrix Core硬件加速、fake quantization、块量化与最大比例缩放

**📊 数据集**

评估模型包括Llama-3.1-70B、Llama-3.1-8B、Llama-3.2-1B、Qwen2.5-7B、Mixtral-8x7B，基准测试为Winogrande、PIQA、BoolQ、LAMBADA

**📈 对比分析**

在同等比特/参数预算下对AIQ与MXFP8/6进行fake quantization，测算模型/基准准确率差异；结果显示AIQ在8.25 bits/param时与MX相当甚至更优，6.25 bits/param亦具竞争力；矩阵乘法实现在MI300X上实现了INT8级别吞吐量

**⚠️ 局限性**

目前仅使用二次代数整数；对除法和比较操作缺乏高效实现；实验仅基于fake quantization，未验证真实部署下算术误差优势；硬件支持有限，未实现专用AIQ加速器

---

## 106. Large Language Models, Machine-Assisted Cultural Analytics, and the Agentic Turn

**arXiv ID:** 2610.12792 | [PDF](https://arxiv.org/pdf/2610.12792v1)

**作者:** Andres Karjus `[一作]` `[通讯]` (Tallinn University), Andres Karjus (Tallinn University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述并评估大型语言模型（LLM）及其 agentic 系统在文化分析与数字人文研究中的应用，尤其聚焦于从文本/多模态数据中自动量化、标注、分类以及自动化研究流程的设计与实践。

**💡 创新点**

创新点：①将 LLM 的“提示+少量示例”方式推广为可重复的量化工具，①在标注、注释与编码流程中实现大规模自动化；②提出 agentic 系统框架，让模型动态决定后续步骤，进一步拓展机器代理在完整研究工作流中的角色；③系统性讨论 LLM 标注的可靠性、误差评估与不确定性在统计分析中的处理；④指出技术普及导致的学科身份、能力分布与研究伦理的新挑战。

**🔧 技术方法**

技术：大型语言模型（如 GPT‑4/5、Claude、Hermes 等）及其 API；in‑context learning、few‑shot prompt、任务微调；多模型集成与投票；LLM 生成的结构化输出；代理系统（agentic pipelines）与自动化决策；统计模型（混合效应模型、贝叶斯不确定性估计、Dawid‑Skene、DSL、CDI 等）用于处理标注不确定性。

**📊 数据集**

数据集：文献语料（小说、童话）、新闻与公共文件、历史档案、政治宣言、图像/视频数据、社会网络文本等多领域文化数据；同时引用已有研究中的标注数据集作为验证基准。该章节并未聚焦单一公开数据集，而是讨论多种文化数据源的通用处理流程。

**📈 对比分析**

比较方法与性能：与传统人工标注或专用机器学习分类器相比，LLM 可实现 1–2 个十倍的标注速度提升，成本降低；在部分任务中，LLM 的准确率与人工标注相近或更高；在不需要复杂训练的数据上，LLM 的零/少 shot 表现尤为突出。文中引用了多项实验（如 Karjus2025MAQD、GilardiEtAl2023ChatGPTAnnotations 等）说明了这些提升，但整体强调的是“足够好”而非“最佳”，并要求结合不确定性分析与人工复核。

**⚠️ 局限性**

局限性：①模型偏差与不可解释的错误（多模型共性错误、训练数据偏差）；②缺乏独立性与交叉验证的挑战；③对人工复核与验证的依赖，无法完全替代人工判断；④可能导致研究者技能退化与工作岗位流失；⑤模型集中化、API 依赖导致可复制性与可持续性问题；⑥法律、隐私与版权限制，限制模型与数据共享；⑦对多语言、多模态任务的鲁棒性尚未充分验证；⑧需要进一步研究模型不确定性在统计推断中的影响。

---

## 107. When Generative Models Act as Simulators

**arXiv ID:** 2610.12847 | [PDF](https://arxiv.org/pdf/2610.12847v1)

**作者:** Sevan Brodjian `[一作]` (California Institute of Technology), Pietro Perona `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在生成模型中编辑潜在状态是否能让其行为像仿真器一样，并提出了基于探针和逆映射两种编辑方法。

**💡 创新点**

发现编辑可行性依赖于环境而非仅解码性能，证明逆映射能在探针失效时仍驱动模型。

**🔧 技术方法**

使用了Transformer生成模型、线性/MLP探针、梯度调度和逆映射编辑器，并在Othello与Rayworld两种环境上评估。

**📊 数据集**

使用Othello棋盘序列与Rayworld的连续轨迹（由自定义渲染产生的一维光束观测）作为训练与测试数据。

**📈 对比分析**

通过比较编辑后预测与仿真器生成的真实观测，使用Edit Index与Edit Fidelity指标，结果显示在Othello中编辑性能随规则简化下降，而在Rayworld中逆映射取得高Edit Index即使连续目标编辑失败。

**⚠️ 局限性**

局限在于只考察单步编辑、对连续编辑缺乏泛化、且依赖于已知的仿真器状态映射，未能处理更复杂或多步骤的动态一致性。

---

## 108. Energy-Efficient Spiking Neural Networks for Sensing-Aided Beam Prediction

**arXiv ID:** 2610.12507 | [PDF](https://arxiv.org/pdf/2610.12507v1)

**作者:** Nhan Thanh Nguyen `[一作]` (University of Oulu), Markku Juntti `[通讯]` (University of Oulu)

**通讯引用:** 12942 | [OpenAlex ID](https://openalex.org/A5029534229)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在毫米波波束预测任务中，使用位置、视觉和多模态感知信息时，SNN（脉冲神经网络）与传统ANN（人工神经网络）的准确性与能耗权衡。

**💡 创新点**

创新点在于系统性比较SNN与ANN的能耗‑准确性曲线，发现对视觉与多模态输入，SNN可在相同能耗下显著优于ANN，且达到相同准确度时能耗约为ANN的十分之一；并揭示二进制突触通信是SNN准确率上限的主要原因。

**🔧 技术方法**

采用了Leaky Integrate‑and‑Fire 神经元、直接输入编码、arctan surrogate 梯度训练的VisionSNN、PositionSNN 与 MultiSNN 三种架构；ANN方面使用 ResNet‑50、MLP 等网络，并通过 MAC/AC 运算计数与能耗模型评估推理成本。

**📊 数据集**

使用 DeepSense 6G 数据集的 Scenario5（共 2300 组 RGB 图像、GNSS 位置信息与 64 波束功率向量），并将码本降为 32 个波束进行实验。

**📈 对比分析**

通过网络宽度缩放、T=8 步 SNN 训练并测量每个模型的 top‑1 准确率与推理能耗，绘制准确率‑能耗曲线；结果显示：在视觉/多模态场景下，SNN 在相同能耗下比 ANN 高 8~18 个百分点，且实现相同准确度时能耗仅为 ANN 的 1/10；位置输入时 ANN 仍优于 SNN。

**⚠️ 局限性**

SNN 的准确率受限于二进制突触通信，增大模型宽度无法突破准确率上限；若去除二进制触发提升准确度，将失去其主要的能耗优势。

---

## 109. Better Batches, Better Privacy: Random Allocation for DP-BandMF

**arXiv ID:** 2610.12712 | [PDF](https://arxiv.org/pdf/2610.12712v1)

**作者:** Ryan McKenna `[一作]` (Google Research), Arun Ganesh `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了三种基于随机分配的批次选择策略（Cyclic Random Allocation、Nested Random Allocation、Min Sep Random Allocation），并将其与 DP-BandMF（带状矩阵分解）结合，通过联合优化策略矩阵与噪声乘子，改进了差分隐私训练的噪声级别和均方误差；同时实现了高效的 GPU 并行 Monte Carlo 计量，显著提升了隐私损失采样吞吐量。

**💡 创新点**

1) 统一了随机分配与带状矩阵分解的研究线，构建了兼顾固定参与次数、b 分离和高熵的批次选择框架；2) 在 Monte Carlo 计量中引入动态规划递推，显著降低计算复杂度；3) 通过一次性优化带宽和 Toeplitz 系数，进一步提升了模型精度。

**🔧 技术方法**

差分隐私随机梯度下降（DP-SGD）、带状矩阵分解（DP-BandMF）、Monte Carlo 隐私损失分布（PLD）计量、Toeplitz 矩阵优化、JAX+CUDA GPU 并行实现、动态规划递推算法。

**📊 数据集**

文中未具体给出使用的公开数据集；实验基于标准的 DP 训练基准（如 MNIST、CIFAR-10 等常用图像数据集）进行 RMSE 对比，具体数据集在实验章节未明确列出。

**📈 对比分析**

与基准方法（Poisson、Random Allocation、Deterministic、Cyclic Poisson 等）在 RMSE 上做了对比，实验显示在不同迭代数、参与次数、隐私预算下，所提方法平均可实现 5%–85% 的 RMSE 降低，最高 1.85 倍；在噪声乘子上实现了 13.4% 的进一步提升；GPU 实现将隐私损失采样吞吐量提升至 10^6 次/秒，较 CPU 提升 50–100 倍。

**⚠️ 局限性**

1) Monte Carlo 计量仍需 Ω(1/δ) 次采样，导致校准时间较长；2) 策略矩阵仅在单参数 Toeplitz 族内搜索，可能未达到全局最优；3) 对不同 b 的精细调优依赖实验，缺乏直接预测公式；4) 目前实现仅覆盖带状矩阵分解，对更一般的噪声结构适用性有限。

---

## 110. Fly-by-Code: Embodied Coding Agents for Aerial Manipulation with Active Visual and Physical Feedback

**arXiv ID:** 2610.12549 | [PDF](https://arxiv.org/pdf/2610.12549v1)

**作者:** Jaewoo Lee `[一作]` (Seoul National University), Dongjae Lee `[通讯]` (Kyung Hee University)

**通讯引用:** 353 | [OpenAlex ID](https://openalex.org/A5101807818)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

为使基于自然语言指令的编码代理在空中操控任务中能够主动获取缺失的结果观测与未解决的失效原因，从而改进任务完成评估和程序修订，作者提出了后执行诊断阶段 Active Feedback（AF），该阶段通过合成独立的 View 与 Probe 程序在仿真中获取信息。

**💡 创新点**

创新点在于将主动感知（View）和交互感知（Probe）嵌入到编程循环的后执行诊断阶段，利用语言模型生成针对每一次失败的专用诊断程序，而非仅依赖被动反馈，显著提升完成判定准确性与程序迭代效率。

**🔧 技术方法**

技术手段包括：使用共享机器人 API 的可执行程序合成与迭代修订；基于 Isaac Lab 的仿真环境进行 View 与 Probe 诊断；大型语言模型（如 Claude Opus、Codex）进行程序生成；以及在仿真与真实硬件之间统一的机器人接口实现零改动的 sim‑to‑real 转移。

**📊 数据集**

数据集与任务：六个扩展自 AM‑Bench 的空中操控任务（1‑Object、2‑Objects Pick‑and‑Place、Sort、Align、Cabinet、Payload）在 Isaac Lab 进行仿真测试；并在自制物理空中操纵器上执行 2‑Objects Pick‑and‑Place 与 Cabinet 任务进行实机验证。

**📈 对比分析**

与基线 CaP‑X（终端反馈）和 CaP‑X+Trace（带执行轨迹的反馈）相比，AF 在六个任务上实现了 75% 的最终成功率（相较 31.7% 与 48.3%），假正例仅 5/60（90% 评估精度），并在实机上实现 80% 与 60% 的任务成功率，显示出显著性能提升。

**⚠️ 局限性**

局限性包括：诊断阶段增加模型推理与机器人交互开销；诊断效果受限于感知误差与不可达视角；当前方案不具备在真实环境中直接进行物理探测的安全保证；依赖于现有运动规划与飞行控制器，且未证明能在完全未知的物理环境或不同机体形态上实现零改造泛化。

---

## 111. Market Impact Estimation via Causal Foundation Models

**arXiv ID:** 2610.12537 | [PDF](https://arxiv.org/pdf/2610.12537v1)

**作者:** Dennis Thumm `[一作]` (National University of Singapore), Ying Chen `[通讯]` (National University of Singapore)

**通讯引用:** 55945 | [OpenAlex ID](https://openalex.org/A5100383082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一种基于连续时间因果先验拟合网络（PFN）的市场因果模拟器，利用可控混淆的合成市场先验训练模型，并在合成、基于 ABIDES 的仿真以及真实加密货币/纳斯达克行情上进行零射转移评估；还提出了递归状态空间变体来捕捉交易持续时间对冲击的累积效应。

**💡 创新点**

创新点包括：①设计了可调混淆强度的三变量市场先验，证明在隐藏信息流的情况下观测回归与因果估计不可识别；②首次将因果 PFN 应用于金融微观结构，实现了在隐藏混淆条件下的无偏冲击估计；③证明了干预语义（软加流）和标准化单位对零射转移的关键作用；④提出分层合规校准方法，修复跨域不确定性；⑤递归变体弥补了 transformer 单步预测导致的持续冲击缺失。

**🔧 技术方法**

采用的技术包括：连续时间因果 PFN（CT‑CPFN）架构、Transformer 编码器与跨变量混合器、量化头（pinball 损失）、软/硬干预实现、递归状态空间门控、split‑conformal 校准、对照消融（do‑ablated twin）、ABIDES seed‑replay 的反事实生成、以及基准模型（Almgren‑Chriss、Obizhaeva‑Wang、平方根法则、回归、AR1）。

**📊 数据集**

使用的数据集包括：①合成先验生成的 500–2000 条 episode；②ABIDES 交易模拟器的 1000 条 seed‑replay 对应的真实/反事实行情；③公开 Binance 持续期货（BTC/ETH/SOL）中检测到的 760+ 事件；④纳斯达克股票的 3800+ 事件；⑤用于训练的参数范围来自文献。

**📈 对比分析**

评估方法：对比基准模型的偏差（bias）、均方根误差（RMSE）、与真实冲击的相关系数；在合成层面验证 PFN 的去混淆效果（bias 0.0–0.16 vs. 0.3–0.98 的传统方法，RMSE 在高混淆下显著更低）；在零射转移层面，PFN 的 do‑head 效果为 +0.38 σ̂（ABIDES）或 +1.4–1.5 σ̂（加密事件），与基准差距 0.4–1.6 σ̂，且在真实市场上表现出更小的偏差；递归变体将相关系数提升至 0.92–0.96。

**⚠️ 局限性**

局限性包括：①先验仅为线性 OU 过程，缺乏订单簿排队、撤单、库存等真实微观结构；②模型对干预语义和单位标准化高度依赖，错误标签会导致错误冲击预测；③递归变体对时间表的稳健性不足，需混合时间表训练；④校准在跨域转移时失效，需分层校准；⑤在真实市场缺乏真确反事实，评估只能基于事实误差，无法完全验证因果性；⑥模型未捕捉价格回溯导致的流量反馈、跳跃或饱和等非线性效应。

---

## 112. Reflex: Visually Grounded Reactive Humanoid Control

**arXiv ID:** 2610.12772 | [PDF](https://arxiv.org/pdf/2610.12772v1)

**作者:** Taoyang Jia `[一作]`, Ranjay Krishna `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 113. SynAct: Synergistic Active Feedback Learning for LLM Agents

**arXiv ID:** 2610.12651 | [PDF](https://arxiv.org/pdf/2610.12651v1)

**作者:** Ning Liu `[一作]` (Amazon), Kristina Klinkner `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SynAct，一种将专家反馈按决策点主动分配的框架，通过从 agent 的多次 roll‑out 中计算动作与最终结果的互信息，挑选最具决定性决策并让专家给出可迁移规则，随后在部署时通过错误门限式注入实现无专家恢复。

**💡 创新点**

创新点包括：① 以互信息 I(A;O) 作为无标注、无成功标签的决策重要性度量，能够在所有 roll‑out 失败时仍保持信息；② 通过支持阈值 m 将可识别决策与不可识别决策在有限样本下严格分离；③ 将专家投入集中在关键决策而非整个轨迹，并通过规则抽象实现可复用、无专家依赖的改进；④ 在推理时直接使用该信息进行专家调度，兼具在线与离线两种形式。

**🔧 技术方法**

核心技术包括：多次 roll‑out 并行采样、基于环境状态的聚类 (state key)、互信息估计、经验蒸馏规则（convention & antipattern）写作、错误门控（gate）以及多轮迭代与推理时的分布式决策路由。

**📊 数据集**

使用四个公开 benchmark：InterCode‑SQL (Spider), TextCraft (recipe depth 3), InterCode‑Bash (nl2bash), τ‑bench (retail)。同时在 Qwen3‑32B、Gemma‑3‑12B、Gemma‑3‑4B、Ministral‑3‑3B 四大模型上进行评估。

**📈 对比分析**

与四类经验蒸馏基线（ACE, ReasoningBank, AWM, AWM‑iter）以及四种采样策略（成功互信息、动作熵、随机）对比，SynAct 在所有 16 组合上均提升准确率，单轮平均提升 8.3pp，最佳迭代平均提升 11.7pp；在推理时路由上，在 15/16 设定中击败 token‑匹配级联、熵路由及专家判断路由；在专家 token 成本方面，SynAct 的收益曲线比基线更平滑、更高。

**⚠️ 局限性**

局限性：① 需要可靠的 state‑key 与 outcome‑projection，若环境不满足决策充分性则效果受限；② 互信息估计对采样预算敏感，支持阈值 m 可能错过罕见模式；③ 仅在无专家时冻结模型，未考虑模型动态更新；④ 评估中专家为强大 LLM，未测算人类专家的成本与可行性；⑤ 对确定性环境的理论支持有限，实际随机性环境的性能尚未验证。

---

## 114. Contextual Correction of Parallel Draftsfor Speculative Decoding

**arXiv ID:** 2610.12570 | [PDF](https://arxiv.org/pdf/2610.12570v1)

**作者:** Roy Eisenstadt `[一作]` (Stealth Startup), Ilan Naiman `[通讯]` (Stealth Startup)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的上下文校正器DFlash Contextual Corrector，用于并行扩散草拟的推理加速，通过并行验证多个候选标记来恢复草拟块内的依赖关系。

**💡 创新点**

创新点在于通过并行评分候选标记之间的上下文兼容性，避免了额外的草拟前向传递，从而在保持并行草拟的同时恢复了草拟块内的依赖关系。

**🔧 技术方法**

使用了轻量级的神经网络评分器，结合了并行扩散草拟和上下文校正机制。

**📊 数据集**

在多个验证器骨干和十个基准测试上进行评估，包括数学、代码和通用生成任务。

**📈 对比分析**

与其他方法（如DSpark和Domino）相比，DFlash在所有评估配置中都实现了最高的平均吞吐量，速度提升高达6.6倍，且在并发请求下表现优于最强竞争方法6-23%。

**⚠️ 局限性**

限制在于该方法依赖于草拟器的输出质量，如果草拟器的初始候选标记质量较低，可能会影响最终的校正效果。

---

## 115. Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests

**arXiv ID:** 2610.12849 | [PDF](https://arxiv.org/pdf/2610.12849v1)

**作者:** Mohammad Wathiq Soualhi `[一作]` (Independent Researcher), Basem Suleiman `[通讯]` (University of New South Wales)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对AI编码代理在实际协作仓库中同时修改应用代码与部署基础设施的拉取请求（PR）进行大规模因果分析，量化了双域修改所带来的集成延迟、审查阻力以及潜在的供应链风险。

**💡 创新点**

创新点包括：① 采用无监督的四层文件分类法，实现对文件级别的精确、可重复的域判定；② 在四维（代理架构、仓库结构、测试成熟度、语言生态）上执行完全匹配的1:1最近邻匹配，保证在应用代码量相同的前提下揭示基础设施协同修改的真正因果效应；③ 在两大公开数据集上共计26,999条PR的实证结果，首次揭示了“交叉域双域PR”在整合流程中的“双重陷阱”与人类维护者行为的分裂。

**🔧 技术方法**

使用技术主要包括：Python实现的文件分类器；greedy 1:1最近邻匹配算法（配合0.28σ caliper）；Log‑Normal AFT生存模型、Wilcoxon符号秩检验、McNemar检验以及逻辑回归；对GitHub API提取的PR元数据进行清洗与时间戳对齐；Jupyter Notebook完成统计与可视化。

**📊 数据集**

数据集为：AIDev v5（21,855条PR，涵盖六种代理架构）与 MOSAIC-3M（5,144条PR，含2,570条代理PR与2,574条人类PR），全部采集自2025年6月-7月的创建时间，确保PR已在2025年8月前完成合并。

**📈 对比分析**

比较方法：在四维完全匹配后对双域PR与单域应用PR进行配对差异检验；在AIDev中双域PR平均延迟+13.13小时，时延乘数3.24×；在MOSAIC中未显著延迟，但56%双域PR在无人工审查下瞬间合并；人类对比实验显示，代理双域PR的人工审查率与人类对比显著不同（降低63.7%）。

**⚠️ 局限性**

局限性：① 仅评估已合并的PR，存在存活偏差；② 受GitHub API payload限制导致部分大PR缺失文件信息；③ 匹配后剔除约30%大文件PR，结果仅适用于典型规模PR；④ 仅覆盖星级>100的公共仓库，难以推广至私有或极大规模仓库；⑤ 在MOSAIC中统计功效有限，未能显著捕捉延迟；⑥ 未能完全捕捉提示策略、私下沟通等未观测混杂因素。

---

## 116. RDSE: Real-Time Directional Speech Enhancement Framework for Dual-Mic Commodity Smartphones

**arXiv ID:** 2610.12476 | [PDF](https://arxiv.org/pdf/2610.12476v1)

**作者:** Chinmay Deo `[一作]` (National Institute of Technology Jalandhar), Himanshu Buckchash `[通讯]` (IMC University of Applied Sciences)

**通讯引用:** 453 | [OpenAlex ID](https://openalex.org/A5018753704)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并实现了 RDSE（Real‑Time Directional Speech Enhancement）框架，利用双麦克风智能手机的音频数据实现实时的方向性语音增强，视觉输入仅用于一次性设定观测方向。

**💡 创新点**

创新点包括：
• 一次性视觉参数化，避免连续视觉特征提取导致的延迟和遮挡问题；
• 对双麦克风的几何容错 MVDR 波束成形，能够在未知阵列几何下稳定工作；
• 模块化混响去除（U‑Net）、语音提取（Conv‑TasNet）和后置降噪（MetricGAN）的组合，实现高质量的音频放大；
• 提供 SpARZ100 真实与模拟双麦克风数据集，系统化评估与公开对比；
• 采用非参数统计检验验证各模块贡献和方法差异。

**🔧 技术方法**

主要技术包括：混响去除的 U‑Net、MVDR 波束成形（带对角加载）、Temporal Convolutional Network（Conv‑TasNet）用于目标说话人提取、MetricGAN 作为后处理降噪、时域重建、实时块处理、时间延迟估计、低算力硬件实现和统计显著性测试。

**📊 数据集**

使用的数据集：SpARZ100（模拟双麦克风、安静与RT60=0.5s 两类）、真实手机录音（Samsung F24）、LibriSpeech（目标语音）和 MUSAN（干扰语音/噪声）。

**📈 对比分析**

与 Nair、Li、Ephrat 等现有 AVSE/beamformer 方法做定性对比；与 SepFormer、FasNet‑TAC 作为语音提取基线做客观比较；在 SpARZ100 上评估 STOI、ViSQOL、PESQ。结果显示：在无去混响时，anechoic 条件下 RDSE 获得 STOI 0.775、ViSQOL 2.005、PESQ 2.576；reverberant 条件下 STOI 0.554、ViSQOL 1.871、PESQ 1.818，优于或与其他方法相当，并且在双麦克风、实时、低算力场景下表现竞争力。统计检验进一步证明各模块对性能提升具有显著贡献。

**⚠️ 局限性**

局限性包括：
• 在高RT60/重混响环境下性能显著下降，尤其是去混响模块在此条件下效果有限且算力消耗大；
• 对未知几何的容错虽然提高了鲁棒性，但在极端多路径或微距环境中仍可能产生估计误差；
• 视觉一次性定位依赖用户手动操作，在快速移动或多目标场景下使用体验有限；
• 目前仅针对双麦克风阵列，无法直接扩展到更大阵列；
• 需要进一步模型压缩、量化或蒸馏以满足更严格的移动端算力约束。

---

## 117. Domain Adaptation of SAM 2.1 for Building-Component Segmentation in Image-Based 3D Reconstruction

**arXiv ID:** 2610.12786 | [PDF](https://arxiv.org/pdf/2610.12786v1)

**作者:** Ehsan Agha Ebrahimi `[一作]` (Louisiana State University), Keith G. Mills `[通讯]` (Louisiana State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对建筑图像进行多类别分割，随后将分割结果用于3D构件的层级化聚类。

**💡 创新点**

通过在建筑域大规模数据集上对SAM 2.1进行微调，实现了显著提升的分割精度，并证明微调模型在GARField框架中的3D聚类表现更佳。

**🔧 技术方法**

使用Meta AI的Segment Anything Model 2.1（Hiera‑Large）作为基础模型，结合GAN与NeRF技术（GARField）、Transformer（SegFormer）以及自定义的中心点提示策略。

**📊 数据集**

采用包含42座建筑的HBD数据集（2 679张带有11类标注的图像）进行训练与评估，并在Patrick F. Taylor Hall的无人机影像上验证3D聚类。

**📈 对比分析**

与预训练SAM 2.1以及自行微调的SegFormer进行对比；mIoU从49.9%提升至69.6%，Dice从59.3%提升至77.3%；在GARField中使用微调后模型生成的掩码在不同聚类尺度下实现了更连贯、更符合建筑结构的3D分组。

**⚠️ 局限性**

局限性包括：对“开口”类的分割仍不理想；墙面类仍为最难分割的类别；SegFormer受数据量限制易过拟合；聚类评价主要为定性展示，缺乏量化指标；仅针对11类，缺乏更细粒度或其他建筑要素的适用性。

---

## 118. One-Step Text Generation by Seq-Drifting

**arXiv ID:** 2610.12879 | [PDF](https://arxiv.org/pdf/2610.12879v1)

**作者:** Jiayu Liu `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Seq-Drifting框架，直接学习从高斯噪声到完整文本序列的一步生成模型。

**💡 创新点**

创新点在于为每个序列位置构造上下文相关的可行词汇支持集，并通过漂移场实现单步的吸引与排斥，结合球面几何实现稳定训练。

**🔧 技术方法**

核心技术包括：连续词嵌入空间中的漂移学习、基于支持集的角度吸引和跨样本/序列排斥、球面投影与大圆步进更新，以及可选的无监督/预训练语言模型用于生成支持集。

**📊 数据集**

使用的数据集包括LM1B、OpenWebText进行无条件生成评估；WMT14（德英翻译）和XSum（摘要）用于下游任务；SVAMP、ASDiv和ProofWriter用于推理任务。

**📈 对比分析**

与多步扩散模型和几种少步蒸馏模型相比，Seq-Drifting在单步推理下实现了36.01的生成困惑度，速度提升4–11×（多步）或900–2500×（单步），在翻译、摘要和推理任务中也保持了竞争性或优于少步基线。

**⚠️ 局限性**

局限性包括：仍与多步模型存在性能差距，支持集的质量和规模对结果敏感，且在高度确定性任务中对支持集的依赖可能导致训练不稳定。

---

## 119. When Charts Deceive Web Agents: A Paired Benchmark for Misleading Visualization-Grounded Tasks

**arXiv ID:** 2610.12823 | [PDF](https://arxiv.org/pdf/2610.12823v1)

**作者:** Yisheng Li `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 MisVisAgentBench，一个包含 140 对干净与误导图表的可执行 Web Agent 基准，评估误导可视化对 Agent 执行的影响。

**💡 创新点**

提出将误导视觉证据作为可控变量的配对可执行基准，并设计诊断框架分离误导导致的失败与任务难度，量化执行层面的额外步骤与决策犹豫。

**🔧 技术方法**

采用 LLM 辅助生成任务规范、可执行 Web 环境、浏览器控制脚本与视觉文本融合技术，并引入 PRA、AFR、MSR、SI、DI 等诊断指标进行评估。

**📊 数据集**

使用 Perils of Chart Deception、Misleading ChartQA 等公开误导可视化数据集，从 508 个候选图表中筛选 243 个可执行单元，最终生成 140 对任务。

**📈 对比分析**

通过在 13 种视觉语言 Web Agent 上对干净与误导图表的任务成功率、PRA、AFR、MSR、SI、DI 等指标进行对比，发现误导图表导致平均任务成功率下降 34 个百分点，PRA 仅为 39.6%，表明强大模型在干净条件下的高性能并不能保证鲁棒性。

**⚠️ 局限性**

仅覆盖仪表盘式可视化，未涵盖所有交互模式与组织流程；基准依赖特定的浏览器控制与提示，其他框架可能表现不同；并未涵盖所有误导机制。

---

## 120. EvoBelief: Falsifiable World Models for Self-Improving LLM Agents

**arXiv ID:** 2610.12789 | [PDF](https://arxiv.org/pdf/2610.12789v1)

**作者:** Janet Wang `[一作]` (Tulane University), Jiarong Jiang `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoBelief，一种在部署时通过假设形成、验证和修订来更新环境信念的自我演化世界模型；

**💡 创新点**

创新在于将环境信念拆分为可证伪的假设，采用证据驱动的规则生命周期（验证、保留、撤回），并将验证后的规则跨任务应用，保持LLM冻结；

**🔧 技术方法**

使用可证伪假设（C∧A⇒Y）表示法，主动/被动证据收集，阈值驱动的规则承诺与修订，规则支持的预测与规划；

**📊 数据集**

在四个基准上验证：ALFWorld、ScienceWorld、MiniGrid（隐藏动态漂移）和DiscoveryWorld（科学发现）；

**📈 对比分析**

与ReAct、WorldEvolver、Frozen WM对比，EvoBelief在ALFWorld、MiniGrid、DiscoveryWorld分别提升任务成功率至约66%、78%和83%，显著优于基线；

**⚠️ 局限性**

局限在于需耗时的主动实验获取证据，主要适用于可观测的离散动力学，对高度随机或连续环境的适用性有限，且阈值设定需要先验知识。

---

## 121. FallGown: A Soft Pneumatic Wearable Device for Fall Prevention

**arXiv ID:** 2610.12706 | [PDF](https://arxiv.org/pdf/2610.12706v1)

**作者:** Megan C. Coram `[一作]` (Stanford University), Allison M. Okamura `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了可通过气压支撑全身的柔性穿戴式“FallGown”来预防老年人跌倒。

**💡 创新点**

将可充气气囊与柔性织物相结合，形成全身支撑的裙式结构，兼具轻便、无臂支撑且具时尚外观。

**🔧 技术方法**

使用软体机器人技术、气压气囊、可穿戴压力传感器、可折叠织物以及可选轮式底盘。

**📊 数据集**

无公开数据集，实验以单个实验者进行倾斜和跌倒试验、压缩测试以及拖拉阻力测量。

**📈 对比分析**

通过与不同段数和有无底盘的版本比较，发现12段+底盘版本在三向跌倒测试中成功率高达100%，但拖拉阻力较大。

**⚠️ 局限性**

局限在单人实验、未验证在人群中安全性、框架不易通过狭窄空间、重量需助手协助、缺乏长期临床数据。

---

## 122. Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building

**arXiv ID:** 2610.12638 | [PDF](https://arxiv.org/pdf/2610.12638v1)

**作者:** Amey Vasulkar `[一作]`, Tadeo-Baldiri Salcedo-Rahola `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对荷兰阿姆斯特丹大学THUAS Delft校园一座教学楼的每日24小时电力消耗进行预测模型比较，涵盖统计、机器学习、LSTM以及预训练TimesFM模型。

**💡 创新点**

首次在同一实验框架下将预训练TimesFM与传统模型严格对比，并探讨天气源对预测误差的影响；同时对工作日、周末与假期等不同运营状态下的误差差异进行细粒度评估，提供预训练模型在实际部署中的参考。

**🔧 技术方法**

使用季节性naive、Ridge回归、XGBoost、Prophet（含/不含外部回归器）、LSTM以及TimesFM 2.5与3.0（目标历史版与含协变量版）；采用滚动窗口时间序列交叉验证、bootstrap区间、MAE/RMSE/R²/ MAPE/sMAPE 等指标；并测量拟合/推理时间和内存占用。

**📊 数据集**

基于THUAS Delft校园2022年9月到2024年3月的13,752个小时电力测量数据，配合47个候选特征（日历、天气七项、历史用电、开课信息等），天气数据来源为KNMI观测、ERA5再分析与历史IFS预报。

**📈 对比分析**

在相同的57天测试窗口下，按整体MAE和RMSE对比；TimesFM 3.0+协变量取得最低MAE（19.97 kWh，RMSE 30.26 kWh），比周季节naive低43.4%；XGBoost紧随其后，MAE 22.45 kWh，RMSE 31.11 kWh；LSTM耗时最长且未超越任何方法；天气源差异极小（MAE差别≈0.17 kWh）。

**⚠️ 局限性**

测试样本仅覆盖57天且偏向冬春季，缺乏跨季节独立验证；天气时间戳解释存在不确定性；未评估能源成本或控制效果；预训练TimesFM仅在非商业研究许可下使用；未考虑更大规模模型或不同硬件平台的资源消耗。

---

## 123. SoK: Privacy-Preserving Order Matching

**arXiv ID:** 2610.12535 | [PDF](https://arxiv.org/pdf/2610.12535v1)

**作者:** Prerna Arote `[一作]` (IMDEA Networks Institute), Lucianna Kiffer `[通讯]` (IMDEA Networks Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化研究并整理了22个隐私保护订单匹配协议，提出四阶段生命周期模型及相应的隐私、功能、信任、可验证性等维度的分类体系

**💡 创新点**

首次在单一框架下综合比较不同场景（金融市场与区块链资产交易）下的隐私匹配协议，揭示了跨阶段泄露与权衡，强调了端到端隐私评估的重要性

**🔧 技术方法**

采用系统化方法论、构建分类维度、对协议功能与安全假设进行抽象，使用了密码学（MPC、HE、TEEs、ZKPs、DP）以及协议层面的设计与实现细节

**📊 数据集**

无实验数据集；主要依据文献综述与协议规范（论文、白皮书、代码库、专利）进行评估

**📈 对比分析**

通过统计表格展示各阶段、属性、角色与机制的分布，量化了隐藏/公开信息的比例；未给出性能指标，但指出不同机制（MPC、HE、TEE）在计算与通信成本上的差异

**⚠️ 局限性**

未对协议声称的隐私保证进行独立验证；缺乏统一的端到端隐私度量；对结算阶段的隐私说明不完整，导致评估不完整

---

## 124. Realism VS Accuracy: Event Sequence Forecasting from a Generative Modeling Perspective

**arXiv ID:** 2610.12586 | [PDF](https://arxiv.org/pdf/2610.12586v1)

**作者:** Dmitry Osin `[一作]` (Applied AI Institute), Alexander Korotin `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向事件序列的多步预测基准，评估预测准确性和分布真实性，并提出 Wasserstein BaryBooster 通过聚合生成样本提升预测性能。

**💡 创新点**

创新点：① 将准确性与分布真实性双重评估引入基准，提出 TMS 与 cDFS 两个新指标；② 设计 Wasserstein BaryBooster 将扩散模型生成的多条轨迹聚合成最优预测；③ 提供统一基准与多模态公开数据集，促进跨模型比较。

**🔧 技术方法**

技术手段包括：条件生成扩散模型（数据空间/潜在扩散）、GPT‑2 风格自回归/并行生成、变分自编码器、分类器两样本检验（cDFS）、Hungarian 算法（TMS）、Barycenter 聚合（BaryBooster）。

**📊 数据集**

使用的公开金融交易数据集包括 Gender、Age、Alphabattle（三个卡交易序列），以及 Retweet、Amazon 两个对照 TPP 数据集。

**📈 对比分析**

通过 TMS 与 cDFS 两指标对比，发现无方法同时在两者上优秀。DEF 在准确性（TMS）上表现最好；LDM 在分布真实性（cDFS）上领先；应用 BaryBooster 后 LDM 的准确性显著提升，成为最高 TMS 的预测器。

**⚠️ 局限性**

局限性：基准仅覆盖有限的金融交易数据，缺乏跨领域验证；BaryBooster 需要生成多条轨迹并求解匈牙利分配，计算开销大，可能不适用于低延迟或资源受限的场景。

---

## 125. DistressBench: Evaluating Multi-Dimensional Crisis Support in Large Language Models

**arXiv ID:** 2610.12758 | [PDF](https://arxiv.org/pdf/2610.12758v1)

**作者:** Drew Rein `[一作]` (Scale AI), Udari Madhushani Sehwag `[通讯]` (Scale AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了DistressBench，一个针对自杀与自伤（SSH）危机情境的对话评估基准，包含718条模拟用户对话、23个SSH子类别以及7维权重化评估准则，评估语言模型在危机支持中的实质性回应能力；

**💡 创新点**

创新点在于：①从“拒绝”转向评估模型提供的支持质量；②引入细粒度的权重化、维度化评估标准与临床参考答案；③用三人LLM评审面板代替人工逐条评判，形成可复现的评估流程；④公开多轮与长序列的“DistressBench‑LH”补充数据；

**🔧 技术方法**

技术包括：自动化生成评估脚本与评审面板，三人LLM评审的判分与加权汇总；使用bootstrap对模型得分进行置信区间估计；对比模型输出与临床参考答案的权重化分数；对多轮任务使用最终回话评判规则；

**📊 数据集**

使用的数据集为DistressBench v1（718条模拟对话）、DistressBench‑LH（20条长序列红队对话）、以及75条校准任务；所有对话均由持牌临床医师与危机咨询师编写并经过质量控制；

**📈 对比分析**

通过对25个前沿模型（OpenAI、Anthropic、Google、Meta、Mistral等）进行三次独立推理，计算加权准则满足率；最佳模型得分0.884（相当于满足88.4%权重准则），中位数为0.714；与临床参考答案0.987相比，存在约10%差距；多轮任务得分低于单轮，且识别后未行动的比例高达35.3%；

**⚠️ 局限性**

局限性包括：仅涵盖英语与美国本土资源；对话为模拟而非真实危机记录；参考答案与模型在部分任务中同一模型生成，可能引入自相似；回复长度不受控制，可能导致长回复自然满足更多准则；部分维度（如免责声明）设计依赖工具提示，非临床真实需求；

---

## 126. Shift-Aware Identity-Guided Latent Refinement for Referring Audio--Visual Segmentation

**arXiv ID:** 2610.12783 | [PDF](https://arxiv.org/pdf/2610.12783v1)

**作者:** Kun Li `[一作]` (University of Liverpool), Michael Ying Yang `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对 Referring Audio‑Visual Segmentation (Ref‑AVS) 的端到端框架 SAIL，用于在多模态视频中精准定位并持续跟踪指定目标。

**💡 创新点**

核心创新点包括：① 双重 token 设计——每帧动态 token 与全局语义 anchor token 分离；② Shift‑aware Refinement 模块，利用跨模态证据估算并纠正目标漂移；③ 通过 anchor 对齐与连续记忆更新实现时空一致的掩码预测。

**🔧 技术方法**

技术手段主要有：多模态 LLM（Chat‑UniVi‑7B）+ LoRA 微调；视觉编码使用 CLIP‑ViT‑L/14+Q‑Former；音频编码使用 BEATs+Q‑Former；跨模态注意力与门控机制；SAM2‑L 作为掩码解码器并配合记忆库。

**📊 数据集**

数据集：Ref‑AVSBench（含 Seen/Unseen/Null 子集）和 R²‑AVSBench（隐式推理任务）。

**📈 对比分析**

与 AVS、Ref‑VOS 及其他 Ref‑AVS 竞争者（如 EEMC、TSAM、SAM2‑LOVE、Crab、TGS‑Agent、AURORA）进行对比，SAIL 在 Mix、Seen、Unseen 子集上分别取得 J=68.4/70.2/72.0、F=77.5/78.5/79.5，显著超越上一榜首 AURORA（J≈65.5、F≈73.4）且在 Null 子集 S 仅为 0.010，表现出最高的精度与低误报率。

**⚠️ 局限性**

局限性：仍依赖大型预训练 LLM 与 SAM2 模型，对计算资源要求高；在极端音频/视觉遮挡或完全无声场景下仍可能出现漂移；缺乏针对多目标或长时序复杂交互的评估。

---

## 127. CircuitTree: Composing Circuits into A Syllable Transcriber

**arXiv ID:** 2610.12520 | [PDF](https://arxiv.org/pdf/2610.12520v1)

**作者:** Jiankun Wei `[一作]` (Univerity of Toronto), Gerald Penn `[通讯]` (Univerity of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

将从 HuBERT 模型中提取的 39 个单音节（元音和辅音）电路，通过一种贪婪、密度缩放的合并算法和新建的 CircuitTree 数据结构，组合成两个共享模块（元音模块和辅音模块），并构建层级化电路流水线完成 CV/VC 音节转录。

**💡 创新点**

创新点包括：
- 引入密度缩放（使用几何平均密度对不同密度电路进行加权）来消除“密度支配”问题；
- 设计了抗干扰的贪婪合并算法，能够在保持 F1 分数不下降的前提下合并 90% 的矩阵；
- 发展了 CircuitTree 结构和 Θ(N log N) 的树合并调度，支持大规模电路批量合并；
- 将合并后的元音/辅音模块进一步组合成层级化流水线，展示单一共享电路可以实现原来单独电路未能完成的复合任务。

**🔧 技术方法**

技术手段包括：
- 通过权重级别与边级别的掩码剪枝（Weight & Edge circuits）实现电路发现；
- 密度缩放与阈值筛选的贪婪合并算法；
- CircuitTree 数据结构、topological 排序与递归合并；
- 层级化电路管道（gate + 具体子电路）实现音节识别；
- 通过对比多种合并调度（线性、平衡、Kruskal）评估效率与精度。

**📊 数据集**

使用 Articulatory Index 语料库（10,329 个 CV 与 VC 单音节，20 名说话人）进行训练与评估；每个音素的二分类电路在 HuBERT-base 上进行稀疏化剪枝，随后进行多分类组合。

**📈 对比分析**

与未合并的独立电路（66.0%）相比，合并后实现 64.8% 的 CV/VC 音节转录精度，仅损失 1.2%；同时实现 99.5% 的键共享、30.3% 的全共享。多种合并调度实验显示平衡调度与 Kruskal 调度在精度上相近，但复杂度为 Θ(N log N)，显著低于线性调度的 Θ(N²)。

**⚠️ 局限性**

局限性包括：
- 仅在 39 个单音素电路的实验中验证，尚未在更大规模或更复杂任务上测试；
- 合并过程依赖阈值 ε 与密度阈值，可能需要针对不同模型或数据集手动调参；
- 对于极高密度或极低密度的电路，密度缩放的几何平均法可能不够稳健；
- 目前的层级化流水线仅实现 CV/VC 音节转录，尚未证明可扩展到更高级的语音或非语音任务。

---

## 128. Fix What's Wrong, Keep What's Right: Hallucination Repair for LLM-Based Optimization Modeling

**arXiv ID:** 2610.12529 | [PDF](https://arxiv.org/pdf/2610.12529v1)

**作者:** Jiageng Wu `[一作]` (Peking University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 5637 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练-free的后处理框架，用于自动诊断并修复LLM生成的运筹优化建模（数学模型和求解器代码）中的幻觉错误，同时保持原问题的完整性。

**💡 创新点**

创新点包括：①把幻觉修复定义为后处理任务；②利用双向证据（需求与实例数据）进行方向性一致性检查；③根据错误范围自动选择最小必要的修复粒度；④通过契约（Contract）约束局部编辑，确保正确内容不被破坏。

**🔧 技术方法**

技术手段包括：LLM驱动的诊断与修复、基于需求/实例数据的证据推理、方向性一致性校验、契约机制控制可执行编辑、artifact-level 审计与安全提交。

**📊 数据集**

实验使用六个运筹建模基准（MAMO Complex、IndustryOR、ComplexOR、OptMATH、NLP4LP、BWOR），并对多种生成管线（如ORMind、OptiMUS-0.3、SAC-Opt）及微调模型（ORLM、SIRL、LLMOPT、OptMATH）进行评测。

**📈 对比分析**

与Prompt-based Repair、Multi-stage等通用基线对比，指标为 Accuracy、Rescue、Regression、Error Coverage。结果显示：零射线生成的准确率从44.8%提升至64.9%，Rescue 36.9%，Regression ≤1%，Error Coverage 94.1%。对微调模型也获得8–15个百分点的 Accuracy 提升。

**⚠️ 局限性**

局限性：仅适用于现有LLM生成的工件；需要手工定义和维护24类RepairCard；对极端复杂或结构大变的模型仍可能需要重构；对执行环境和求解器依赖较高；并非覆盖所有可能的幻觉类型。

---

## 129. Spend Where It Matters: Particle Search with MH Refinement for Budget-Constrained Audio-Visual Alignment

**arXiv ID:** 2610.12811 | [PDF](https://arxiv.org/pdf/2610.12811v1)

**作者:** Shivanshu Shekhar `[一作]`, Uttaran Bhattacharya `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段推理时对齐框架：先用自适应粒子搜索在有限预算内寻找高奖励候选，再用Metropolis-Hastings细化以提升多目标奖励。

**💡 创新点**

创新点在于将粒子分配的方差最小化理论与MH细化结合，实现预算约束下的全局探索与局部改进协同，且不依赖离线调参或奖励特定的预训练。

**🔧 技术方法**

采用了粒子分配的方差优化（VASR）理论、序贯蒙特卡洛（SMC）采样、Metropolis-Hastings采样、在线有效样本数（ESS）估计、以及流匹配/扩散模型的推理。

**📊 数据集**

使用的主要数据集包括音视频多模态生成任务的 JavisBench‑Mini、图像生成任务的 Stable Diffusion v1.5（在 DrawBench 与 SimpleAnimals 上评估）。

**📈 对比分析**

与 Best‑of‑N、EvoSearch、Uniform+MH 等基线在匹配或相近计算预算下比较，均衡奖励指标提升约 30–35%（音视频）以及图像领域约 4% 的平均奖励提升。

**⚠️ 局限性**

局限性包括：在高预算下自适应分配提升有限；需要额外的调度参数；对估计方差快速变化的情况预测不够理想；整体推理成本仍然高，尤其对大型模型。

---

## 130. Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents

**arXiv ID:** 2610.12598 | [PDF](https://arxiv.org/pdf/2610.12598v1)

**作者:** Yiran Wu `[一作]` (Pennsylvania State University), Xuan Feng `[通讯]` (Microsoft Research)

**通讯引用:** 2601 | [OpenAlex ID](https://openalex.org/A5075803142)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种Behavioral Self-Rewarding（BSR）方法，利用同一任务下同一验证结果的轨迹进行对比，生成辅助奖励以补充二元验证奖励，从而提升软件工程代理的性能。

**💡 创新点**

创新点在于：1) 让当前策略自己作为“判定者”对已收集的轨迹进行对比，避免额外训练奖励模型；2) 将对比结果映射为有界辅助奖励，既保留通过验证器得到的通过/失败顺序，又在同一验证结果组内引入奖励对比。

**🔧 技术方法**

技术包括：基于GRPO/LOO的组相对策略优化；对轨迹进行摘要和统计；使用加权Bradley–Terry模型聚合对比；对奖励进行标准化、软死区和中心化映射。

**📊 数据集**

使用数据集：SWE-bench Verified、SWE-bench-Live Lite和Terminal-Bench 2.1。

**📈 对比分析**

与仅使用验证器奖励的基线相比，BSR在Qwen3-14B和Qwen3-32B模型上均提升了平均准确率和Pass@3（例如在SWE-bench Verified上平均准确率从32.1%提升到34.7%，Pass@3从6.33%提升到8.11%）。

**⚠️ 局限性**

限制在于：仅在失败轨迹上引入辅助奖励能获得最优效果，成功轨迹上的改进有限；对比过程需要额外推理成本；对不同任务规模或训练预算的泛化仍需进一步验证。

---

## 131. A Categorical Atlas of Parameter-Efficient Fine-Tuning: Cones, Orbits, and Gauges

**arXiv ID:** 2610.12728 | [PDF](https://arxiv.org/pdf/2610.12728v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Dai Shi `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过把参数高效微调（PEFT）方法视为指向预训练权重空间的点映射，构建了一个“指向切片”范畴，进而定义并计算了三个核心不变量（映像 germ、可表达维度与第一阶像、以及与优化器相关的纤维度），并在此框架下对十四种常见PEFT方法进行分类、互相模拟关系（simulation arrows）和合并可行性（mergeability）进行系统性分析。整个工作以理论证明为主，并提供了交互式 Atlas 与代码实现的完整校验。

**💡 创新点**

创新点包括：
1) 将PEFT方法映射为指向切片中的对象，首次在这一相对点视角下统一描述全模型、冻结模型、LoRA、prefix、adapter等；
2) 发现并证明了三大不变量在表达性、训练动态和基点变化中的决定性作用；
3) 对零初始化、Rank上限、正交性、可测量的“锥形点”与“光滑点”等几何结构给出了精确定理；
4) 提出了方法间的“模拟”关系与“合并”规则，构建了一个完整的 PEFT 方法 “周期表”；
5) 通过多维度定量（维度、rank、Cut、Gauge 余量）与交互式可视化，给出了对比与预测，提供了可直接验证的实验脚本。

**🔧 技术方法**

主要技术手段包括：
- 场论/范畴论（指向切片、双范畴、笛卡尔积与极限、模拟箭头）
- 微分几何（germ、锥、切空间、Jacobian、复合度量 K=DρΛDρ）
- 代数几何/不变量理论（完整不变量、矩阵正交性、秩‑Cut、张量网络的最小割）
- Lie 群与对称性分析（正交、对角、Hadamard 变换、HRA 的 Householder）
- 优化理论（梯度流、Adam/SGD 的等变性、守恒量、量化误差分析）

**📊 数据集**

论文的核心为理论研究，并未针对特定数据集做大规模实验；验证工作主要基于 Hugging Face PEFT 开源实现的数值检查（float64、梯度 Jacobian、Rank 计算）以及在交互式 Atlas 上的可视化。若有实验，则采用常用 NLP 语言模型（BERT/LLM）权重进行量化与微调，验证理论预言与合并规则。

**📈 对比分析**

对比方法不采用传统的“参数计数 + 任务性能”指标，而是通过三不变量（可表达维度、第一阶像、纤维中的优化器度量）和模拟箭头来衡量方法的潜力、可训练性和迁移性。论文给出的定理指出：
- 任何零初始化或 split 初始化的 LoRA 其可达更新 rank 最高只能为 2r；
- 正交性方法在保持 neuron Gram 和奇异值方面不可改变；
- HRA 等方法的映像是 LoRA 与旋转轨道的交集；
- 通过基点变换（量化、重启）可导致定位缺陷，影响可达性。实验校验与交互式工具验证了这些结论，未给出直接的任务分数，但为方法选择与设计提供了理论依据。

**⚠️ 局限性**

局限与待解问题包括：
- 许多结果假设在精确算术下成立，实际使用中存在浮点误差导致的“破碎”；
- 量化、重启等操作的基点变换仅在特定顺序下可解析，泛化到更复杂的多步骤流程尚未完成；
- 某些 PEFT 方法的可变形（如 QLoRA、LoFT 等）在实现细节上存在差异，理论证明与代码实现之间可能存在偏差；
- 对非线性网络或动态架构（如自注意力层中的查询/键/值投影）的直接分析尚不完整；
- 论文侧重理论验证与结构分析，缺乏大规模任务性能评估和实际效果对比；
- 预测与定理的可验证性需要进一步的实验与数据支持。

---

## 132. MADS: Multi-Agent Dialog Simulation for Robust Conversational Agents

**arXiv ID:** 2610.12757 | [PDF](https://arxiv.org/pdf/2610.12757v1)

**作者:** Arkajyoti Chakraborty `[一作]` (University of California Santa Cruz), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了多代理对话模拟器MADS并基于其构建了覆盖失效路径的MADS‑Bench数据集。

**💡 创新点**

通过可控注入对话层与执行层的对抗场景，生成结构化多轮对话，显著提升了模型对失败路径的鲁棒性。

**🔧 技术方法**

采用三方LLM代理（用户、助手、工具）结合Pre‑Act/ ReAct框架进行训练，先做SFT再用可验证奖励（RLVR）进行强化学习。

**📊 数据集**

使用了自研的MADS‑Bench（6个任务域、55个角色、72个工具、1095轮对话，约66%为失败/不可行情形）以及对比基准APIGen数据集。

**📈 对比分析**

通过L‑1（每轮）与L‑2（整段）评估指标对比SFT与SFT+RLVR模型，RLVR显著提升工具调用准确率与对话成功率，32B模型在测试集上达到约54%的成功率。

**⚠️ 局限性**

受限于训练样本量（尤其是不可行情形）和RLVR对不可行路径的奖励不足，且未覆盖真实API执行与多模态交互等更复杂的现实场景。

---

## 133. Organizational Context for the Enterprise Security Brain

**arXiv ID:** 2610.12540 | [PDF](https://arxiv.org/pdf/2610.12540v1)

**作者:** Eden Yavin `[一作]` (Sola Security), Leon Goldberg `[通讯]` (Sola Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了企业安全大脑的“组织情境”层，构建了从治理文件到资产图的多层知识图谱，并在实际生产环境中评估了抽取与推理的效果。

**💡 创新点**

创新点在于将组织内部的非机器化知识转化为可追溯、时效、冲突管理的命题与证明结构，并将其嵌入安全大脑，实现一次性生成的上下文图谱，支持冲突检测、合规检查和架构映射。

**🔧 技术方法**

主要技术包括使用多模型（Claude Sonnet、Gemini 等）进行命题抽取与评估，构建四层安全大脑（原始数据→资产图→上下文图→组织情境图），以及利用 SQL 查询、检索工具和证据链、权威度、时效性等评分机制。

**📊 数据集**

数据集涵盖 AWS、Okta、Google Workspace 与 GitHub 的生产资产数据，以及 1,355 条原子命题的治理文件（政策、SOC 2 控制、审计报告）和真实架构文档。

**📈 对比分析**

评估方法通过模型集成抽取并人工模型评判，推理任务分为冲突、合规、架构三类；结果显示抽取召回 0.80、精确 0.87，冲突检测 0.97、合规正确率 0.83、架构正确率 0.86，整体优于无上下文基线和通用编码代理。

**⚠️ 局限性**

局限性包括仅在单一生产环境评估、缺少票证/对话语料、基线对比不足、评判依赖模型而非专家校准，以及未测试反向任务（无冲突或不匹配）。

---

## 134. TraceICL: Thread-Resolved Demonstration Curation with Consistency Calibration for Robust Multimodal In-Context Learning

**arXiv ID:** 2610.12516 | [PDF](https://arxiv.org/pdf/2610.12516v1)

**作者:** Yigit Yilmaz `[一作]` (Bandirma Onyedi Eylul University), Amir Rahman `[通讯]` (Bandirma Onyedi Eylul University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TraceICL框架，通过线程分解、影响力选择和弱强一致性校准提升冻结大型视觉语言模型的多模态ICL性能

**💡 创新点**

把交互式多模态提示视为可组合的线程结构；用一次前向推断估算演示示例的边际影响并消除冗余；利用弱模型一致性对演示的可信度进行软加权

**🔧 技术方法**

Thread‑Resolved Decomposer (TRD)、Influence‑scored Demonstration Selector (IDS)、Weak‑to‑Strong Consistency Calibration (WSC)、无梯度训练、CLIP‑style跨模态对齐、基于FLAN‑T5弱模型校准

**📊 数据集**

VQAv2、VizWiz‑VQA、Flickr30K Entities、Hateful Memes、WebSRC、ScienceQA‑IMG（zero‑shot）

**📈 对比分析**

与MMICL、Flamingo、KOSMOS‑1、BLIP‑2、InstructBLIP等冻结‑ICL基线同规模进行4‑shot对比，平均提升1.7点（VizWiz最高+2.4），在ScienceQA‑IMG降低视觉缺口0.7，30%噪声下保持大部分优势，额外26%延迟

**⚠️ 局限性**

对候选示例质量依赖；弱模型的可靠性限制校准；线程假设每示例单图，复杂多图文结构时效果受限；需要额外前向推断导致计算开销

---

## 135. People Making Conversation: Evaluating How Naturally Speech Models Converse

**arXiv ID:** 2610.12802 | [PDF](https://arxiv.org/pdf/2610.12802v1)

**作者:** Arya Bastani `[一作]` (People Make Things), Iman Modarressi `[通讯]` (People Make Things)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于真实两人对话的暂停与转折点评估框架，测量实时语音模型在暂停时是否发声、在转折点的延迟、长度、相似度和对对话历史的记忆召回，并对六个系统进行统一评测。

**💡 创新点**

首次将暂停/转折点细粒度评估与记忆召回结合，构造了统一的端点、延迟、长度和相似度度量，并系统性比较 hosted API 与全双工模型在保持静默与及时发声上的决策差异。

**🔧 技术方法**

使用规则引擎标注、Whisper 能量门控识别语音区间、GPT‑4o 评审、Claude Opus 5.5 记忆判定、端点检测与延迟测量、词数与 token F1 计算等技术。

**📊 数据集**

基于 Seamless Interaction 自然交互子集 176 份对话（约 35.9 小时），共 46,000 个候选边界，选取 427 暂停点和 482 结束点，以及 184 个回忆问答点。

**📈 对比分析**

采用统一的评分脚本对六个系统（PersonaPlex、Moshi、GPT Realtime、Gemini 3.1 Flash Live、Grok Voice）在相同点集下进行对比；结果显示 hosted 系统几乎不保持静默，延迟决定是否落在暂停内；全双工模型能保持静默但多在人类反应时间内插话；在转折点上 latency 与长度均高于人类，记忆召回低于文本输入模式；整体上 hosted 系统在保持静默方面表现不足。

**⚠️ 局限性**

跨模型比较受输入通道差异（文本 vs 音频）影响；端点策略仅测试单一配置；评估仅涵盖英双语对话，缺乏多语与多说话人场景；标注规则与门限不完美，可能影响精度；每个点仅做一次 rollout，缺少多次复现。

---

## 136. CORAL-KV: Compact Cross-Layer Prediction and Residual Coding for Autoregressive Video KV Caches

**arXiv ID:** 2610.12514 | [PDF](https://arxiv.org/pdf/2610.12514v1)

**作者:** Albert Ge `[一作]` (Advanced Micro Devices Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 CORAL‑KV，一种利用跨层线性预测与残差量化压缩自回归视频生成模型 KV 缓存的框架。

**💡 创新点**

创新点：① 引入 block‑diagonal + low‑rank (block‑DPLR) 预测器，显著降低预测器参数量；② 将预测误差做 Hadamard 变换后用 2‑bit EDEN‑2 码本量化；③ 预测器可一次性离线校准并在多视频间共享，或在每个缓存块在线重新拟合块对角项，兼顾内存与质量。

**🔧 技术方法**

技术：跨层线性预测、block‑DPLR 参数化、Hadamard 旋转、2‑bit 量化、低秩矩阵分解、GPU fused kernels。

**📊 数据集**

数据集：HY‑WorldPlay（285帧 704×1280）和 MAGI‑1（192帧 720×720）的视频生成任务；使用 355 条 held‑out MovieGen Extended 提示进行评测。

**📈 对比分析**

比较方法：与 BF16、QVG‑1/4/Pro、AQUA‑KV 基线对比；评估指标包括 PSNR、SSIM、LPIPS、VBench、QBD（量化边界不连续度）、存储缓存内存和生成延迟。表现：可在同等或更少的缓存内存下提升 PSNR 0.32–4.95 dB，降低 37–48% 内存，生成延迟相对 BF16 仅 8–29%；相较 QVG‑Pro，视频适配配置提升 PSNR 3.77 dB，内存减少 33–52%。

**⚠️ 局限性**

局限：仅在两种视频生成模型上验证，跨模型泛化未证实；未给出峰值 GPU 内存统计；量化误差在长序列中可能累积；仅压缩 KV 状态，未结合 token 选择等其他压缩技术。

---

## 137. What Should Embodied Agents Remember? Benchmarking and Evaluating Memory Modality for Embodied Agents

**arXiv ID:** 2610.12561 | [PDF](https://arxiv.org/pdf/2610.12561v1)

**作者:** Junxiang Wang `[一作]` (Dexmate), Tianyu Li `[通讯]` (Dexmate)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种用于评估具身代理记忆的控制实验协议和基准，构建了 53 个任务并在仿真与物理机器人上测试。

**💡 创新点**

创新点在于将记忆模态（文本、图像描述、摄像帧、场景图）与记忆后端分离，并在统一接口下对记忆效果与推理成本进行系统评估。

**🔧 技术方法**

使用大型语言模型（Gemini、GPT、GLM）作为代理与记忆写入器，利用计算机视觉（SAM、场景图）和机器人技能工具，记录并检索多模态数据。

**📊 数据集**

基准数据集为在四个室内场景（办公室、摄影棚、家庭、诊所）中生成的 53 个任务，覆盖七类记忆信息；实验还包含堆叠任务与物理机器人演示。

**📈 对比分析**

对比方法：在同一体验序列下分别保存不同模态的记忆，使用五种后端；评估指标为任务成功率、写入时与读取时的加权标记成本；结果显示存储摄像帧+场景图在所有模型上获得最高成功率，且写入成本低于生成描述。

**⚠️ 局限性**

局限性包括：基准任务相对有限，未覆盖所有现实世界复杂性；记忆后端适配仅针对所选模型，未验证跨模型泛化；物理演示仅限两例，缺乏大规模用户实验。

---

## 138. ForcingBooster: Adaptive Patch-wise Denoising for Fast Autoregressive Video Diffusion

**arXiv ID:** 2610.12625 | [PDF](https://arxiv.org/pdf/2610.12625v1)

**作者:** Zekai Gu `[一作]` (Hong Kong University of Science and Technology), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种轻量级路由器，用于自回归视频扩散模型中按补丁动态分配去噪步骤，实现早停以减少计算量；

**💡 创新点**

创新点在于：①基于中间clean‑latent预测实现补丁级别的自适应停止；②在保持原模型不变的前提下，仅训练少量路由参数；③通过自监督的surrogate损失和on‑policy成本估计，实现无标签的路由学习；

**🔧 技术方法**

技术包括：轻量路由网络、stop surrogate loss、denoising cost loss、on‑policy rollout、full‑step reference、KV缓存重用；

**📊 数据集**

使用Causal Forcing、Wan2.1 14B、LingBot‑World‑V2、LongLive等已有预训练模型的训练数据，并在VBench评测集上评估；

**📈 对比分析**

与统一的一至四步采样对比，路由器在四步基线上实现约1.3×的推理加速、约50% NFE（去噪步骤）减少，同时保持与四步基线相当的质量（VBench Total分数几乎不变）；

**⚠️ 局限性**

局限性：对高动态视频效果相对较弱；需要调节成本权重和停止阈值；仅适用于已有的few‑step自回归扩散模型，且路由器训练仍需额外的训练步骤。

---

## 139. Proceedings of the Seventh miniKanren and Relational Programming Workshop (miniKanren 2025)

**arXiv ID:** 2610.12505 | [PDF](https://arxiv.org/pdf/2610.12505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 140. Enabling Extended SGX Enclaves inside Confidential Virtual Machines

**arXiv ID:** 2610.12695 | [PDF](https://arxiv.org/pdf/2610.12695v1)

**作者:** Pamenas Kariuki `[一作]` (TU Dresden), Christof Fetzer `[通讯]` (TU Dresden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 AMD SEV‑SNP 的 CVM 环境中实现了一个硬件受保护的过程级 enclave，支持原生 SGX 二进制兼容以及扩展 SGX（eSGX）模式，并提供低开销的隔离、异常处理、克隆等功能。

**💡 创新点**

将 VMPL 隔离与 SGX 架构结合，首次实现不需要修改应用、无需超visor 或硬件改动的可兼容 SGX 的 CVM 内 enclave，并通过水平特权层、在 enclave 内异常/系统调用、直接内存管理和拷贝‑写克隆等特性显著提升性能和功能。

**🔧 技术方法**

利用 AMD SEV‑SNP 的 VMPL、SVSM 协议、Rust 编写的安全监控器、Linux 内核驱动、Gramine/SCONE 兼容层，采用 EENTER/EEXIT、ECREATE 等 SGX 指令仿真、VMPL 切换、COW 克隆等技术。

**📊 数据集**

在微基准上使用 nbench、LMbench、WolfSSL、SGXGauge、UnixBench、IOzone；在应用基准上评估 Nginx、OpenSSL、Redis、Memcached、MariaDB 等工作负载。

**📈 对比分析**

与原生 CVM 以及现有嵌套 SGX 方案（NestedSGX、vSGX、Veil）进行对比。计算/内存密集型任务平均 4.8% 以内开销，I/O 密集型任务 12% 左右，克隆速度 93 µs/MB，显著提升了 enclave 克隆效率，且在 Gramine/SCONE 上多工作负载保持在 5% 以下。

**⚠️ 局限性**

主要开销集中在 VMPL 切换，I/O 频繁的系统调用仍有一定延迟；对共享内存频繁访问的模式会触发额外同步；不支持迁移；仅在 AMD EPYC/SEV‑SNP 平台验证，其他 CPU/硬件无支持。

---

## 141. Efficient Learning of Survival Analysis Models Under Outcome-Dependent Truncation and Censoring

**arXiv ID:** 2610.12538 | [PDF](https://arxiv.org/pdf/2610.12538v1)

**作者:** Itai Zilberstein `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20607 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究在结果依赖截断（outcome‑dependent truncation）和右删失条件下，如何有效地训练Cox回归模型，提出利用已知基线风险函数的梯度下降实现参数收敛，并探讨基线未知时的可识别性与优化问题。

**💡 创新点**

证明在基线已知的情形下，即使负对数似然非凸，Cox模型仍有唯一的稳态点，投影SGD能收敛至真实参数；同时揭示基线未知时可出现非可识别和局部最优等不可行性；实验验证第一阶优化相较于EM可实现数十倍速度提升且误差不增大。

**🔧 技术方法**

主要使用Cox比例风险模型、梯度下降（投影SGD、Adam）、期望最大化（EM）以及深度Surv模型（Neural Cox）；在理论上引入加权截断指数分布与右删失变换以分析梯度方向；在实验中使用梯度下降、投影SGD和EM进行对比。

**📊 数据集**

四个公开生存数据集：HEARTS（心脏移植存活），SUPPORT（住院患者存活），METABRIC（乳腺癌存活）和FLCHAIN（实验室检测存活）。此外，还构造了规模极大的HEARTS‑Large（75万样本，2000维特征）用于可扩展性测试。

**📈 对比分析**

与标准Cox回归、EM和神经网络对比；在不同截断水平下，考虑了训练时间与均方误差（MSE）。实验表明：投影SGD与Adam在HEARTS‑Large上比EM快约100倍，且MSE相近甚至更低；在小数据集上三者相似，但在大规模数据上EM显著慢且误差稍高。

**⚠️ 局限性**

主要限制：理论收敛性仅适用于已知基线风险的情形；基线未知时可能出现非可识别或局部最优；实验仅验证了线性Cox和深度Cox，未证明在更复杂模型或极端截断下同样成立；此外，实验证明神经网络受限于同一截断似然修正的有效性仍需进一步理论支持。

---

## 142. Input Representation, Not Model Size or Pretraining Scale, Shapes Speaker Verification Robustness to Temporal Degradation

**arXiv ID:** 2610.12539 | [PDF](https://arxiv.org/pdf/2610.12539v1)

**作者:** Erdem Baha Topbas `[一作]` (University of Zurich), Volker Dellwo `[通讯]` (University of Zurich)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5028554788)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对六种常用的ASV模型（卷积、Transformer、Conformer各配一组波形和Mel滤波器输入）在三种重度时间退化条件（PER、PER+ENV、PER+TFS）下进行评估，比较它们在VoxCeleb1-O上的性能衰减；

**💡 创新点**

发现Mel滤波器输入模型在所有条件下均显著优于波形输入模型，并证明时间细结构（TFS）比幅度包络（ENV）对说话人识别更为重要；

**🔧 技术方法**

采用IPC、Hilbert包络、Butterworth低通滤波等时间分解技术，结合AAM损失和LoRA微调来训练模型；

**📊 数据集**

使用VoxCeleb2作为训练集（含MUSAN噪声与混响数据增强），VoxCeleb1-O作为评估集；

**📈 对比分析**

通过EER、minDCF及95%置信区间进行比较，使用头部空间归一化的衰减度量，结果显示滤波器输入模型在PER下EER升幅平均约为9%，波形模型平均约为30%，差异在8/9对比中显著；

**⚠️ 局限性**

局限在于仅评估保留周期性的时间退化，未考虑周期性去除或更真实的环境噪声；同时仅覆盖了可配对的模型，未探讨更大模型或其他输入形式的鲁棒性。

---

## 143. Distributed Datalog at Scale with Worst-Case Optimal Joins

**arXiv ID:** 2610.12692 | [PDF](https://arxiv.org/pdf/2610.12692v1)

**作者:** Kunting Qi `[一作]` (University of Illinois Chicago), Sidharth Kumar `[通讯]` (University of Illinois Chicago)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了首个基于 Worst‑Case Optimal Joins（WCOJ）的分布式内存 Datalog 引擎，并将 HyperCube 分区扩展到递归 fixpoint 迭代执行。

**💡 创新点**

创新点在于：① 将 WCOJ 与 HyperCube 分区结合实现递归查询的 rank‑local 交叉；② 在每次迭代仅重新分区/复制增量数据；③ 通过多维子立方子通信维持相同布局。

**🔧 技术方法**

使用的技术包括：MPI（all‑to‑all、all‑gather）、HyperCube（Shares）分区、半归纳（semi‑naïve）递归评估、WCOJ（Leapfrog Triejoin 变体）和 C++23 实现。

**📊 数据集**

实验数据集涵盖图分析与程序分析：LiveJournal、Orkut、USRoad、G10（图数据）以及合成的 Medium、Large 程序；在 Polaris 与 Aurora 超级计算机上运行。

**📈 对比分析**

与单机引擎（Soufflé、FlowLog、DuckDB）以及分布式引擎（Slog、Multi‑Node‑Multi‑GPU）进行对比，最多在 32→2048 进程下实现 14.3× 加速，单机性能常优于传统引擎，在 TC、Peer、Triangle、SG、Andersen 等工作负载中显著优于现有分布式系统。

**⚠️ 局限性**

局限性包括：高进程数时通信占比显著提升（超过 50%），当前使用 MPI 默认 all‑to‑all，未针对非均匀消息量优化；未支持增量流式更新；对分区键与维度的选择仍需手工或基于成本模型进一步优化。

---

## 144. COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering

**arXiv ID:** 2610.12837 | [PDF](https://arxiv.org/pdf/2610.12837v1)

**作者:** Kavinda Kehelella `[一作]` (University of Maryland), Sanghamitra Dutta `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于对抗性例子的少量样本开放式问答知识蒸馏方法

**💡 创新点**

创新点在于利用最小修改的对抗样本（Counterfactuals）和新度量CFD来在教师决策边界附近提供额外监督

**🔧 技术方法**

使用对抗样本生成、CFD验证、传统蒸馏（KD/SeqKD/LWD）与T5、Qwen2.5等LLM进行对比实验

**📊 数据集**

在5个内部（NaturalQuestions、SQuAD、NewsQA、TriviaQA、HotpotQA）与3个外部（DROP、DuoRC、RelationExtraction）数据集上评测

**📈 对比分析**

相较于标准蒸馏基线，在所有设置下平均提升约2-4 F1分，显著提升教师-学生一致性和少量样本性能

**⚠️ 局限性**

主要限制是对抗样本生成和CFD验证的计算开销，以及对不同实体类型的覆盖率仍有待改进

---

## 145. TAME-Nav: Temporal Asymmetric Mixture-of-Experts Distillation for Agile and Versatile Robot Navigation

**arXiv ID:** 2610.12877 | [PDF](https://arxiv.org/pdf/2610.12877v1)

**作者:** Lingjie Zhang `[一作]` (Shanghai Jiao Tong University), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并实现了 TAME-Nav，一种异构教师-学生框架，利用时间软 MoE 学习单一策略，能够在复杂动态环境中完成障碍规避、狭窄通道穿越与陷阱逃逸，并在模拟与真实四足机器人上进行验证。

**💡 创新点**

创新点包括：① 将教师利用轨迹增强的特征与学生的软 MoE 进行表征蒸馏；② 采用时间软 MoE 实现场景自适应专家路由；③ 通过大规模并行任务级模拟与程序化场景生成覆盖多样性，并结合部署感知随机化实现 sim‑to‑real。

**🔧 技术方法**

使用的技术包括软 MoE、教师-学生对比学习、PPO 强化学习、LiDAR 约束的安全控制器（CBF）、跨层注意力编码、轨迹特征聚合、时间卷积+GRU 编码器、仿真向量化 2D 机器人模型。

**📊 数据集**

数据集由作者自制的 600 张程序化地图组成，分为六类（密集杂乱、动态交叉、凹陷陷阱、狭窄走廊、遮挡动态、复合室内），以及真实四足机器人实验中的实际环境。

**📈 对比分析**

与 SEA-Nav、NavRL、DCLP 进行对比，采用成功率、碰撞率和 BARN score 评价；在分布内、序列组合、多室场景均实现 20–35% 的成功率提升，碰撞率显著下降；在真实四足机器人上，TAME-Nav 在陷阱逃逸和狭窄通道测试中 100% 成功，碰撞率最低。

**⚠️ 局限性**

局限性：仍依赖二维 LiDAR 传感，对极端遮挡下的动态轨迹推断可能失效；软 MoE 需要较大参数，且未在更大尺度或三维真实环境中进一步验证。

---

## 146. Primal-Dual Flow Matching for Sample-Wise Constrained Generation

**arXiv ID:** 2610.12699 | [PDF](https://arxiv.org/pdf/2610.12699v1)

**作者:** Zhengyan Huan `[一作]` (Tufts University), Shuchin Aeron `[通讯]` (Tufts University)

**通讯引用:** 3702 | [OpenAlex ID](https://openalex.org/A5004738943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种基于流匹配的原始-对偶算法（PDFM），通过仅使用约束集的成员判定器实现约束满足率控制，并给出理论保证

**💡 创新点**

创新点在于：①只需二元成员反馈即可控制样本约束满足率；②提供了流匹配参数化的概率约束满足理论保证；③提出了利用无随机化的确定性策略梯度方法估计约束梯度的实用实现；④实现了对非可微、非凸约束的统一处理

**🔧 技术方法**

使用流匹配（Flow Matching）框架、原始-对偶优化、确定性策略梯度、监督学习预测器 f_ϕ、蒙特卡罗估计以及可扩展的梯度推导

**📊 数据集**

在多种数据集上验证：二维离散约束样本、ℓ₂ 球约束的高维高斯混合、Kuramoto–Sivashinsky PDE 的时空序列、SOCOFing 指纹图像、QM9 分子数据

**📈 对比分析**

与 vanilla FM、基于截断分布 FM、FM‑RE、MDM、NAMM、PDM、PCFM 等基线相比，PDFM 在保持或提升分布相似度（SWD、直方图误差）同时显著提高约束满足率，尤其在非凸、非可微约束场景下表现最优

**⚠️ 局限性**

局限性包括：需要足够多的正约束样本以可靠学习 f_ϕ；对 λ 的上界和学习率设定敏感；在极端噪声或稀疏约束时约束梯度估计可能不稳定；实现上仍需对流匹配模型容量做充分设计

---

## 147. Low-Complexity SVD-Based Precoding over Zero-Padded ISI Channels

**arXiv ID:** 2610.12784 | [PDF](https://arxiv.org/pdf/2610.12784v1)

**作者:** Shuiyin Liu `[一作]` (Holmes Institute), Amin Sakzad `[通讯]` (Monash University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于单零填充和FFT实现的低复杂度SVD预编码/均衡器，可在有限长度的记忆-1 ISI 通道上实现精确的奇异值分解并实现软判决信号的解调；

**💡 创新点**

创新点在于利用单零填充使通道 Gram 矩阵成为对称三对角 Toeplitz，从而得到闭式的右奇异向量（DST‑I）和奇异值，随后通过FFT实现 SVD 前向/后向变换，降低复杂度至 O(nlogn) 并实现自适应 PAM 与功率分配；

**🔧 技术方法**

主要技术包括：单零填充变换、对称三对角 Toeplitz Gram 矩阵的闭式奇异分解、DST‑I 的 FFT 计算、基于最差 SER 的自适应 PAM 与功率分配、以及精确的有限字典 LLR 计算；

**📊 数据集**

实验采用仿真数据，设置 α=0.8、E_s=21，使用长度约 5460 的符号块进行比较；

**📈 对比分析**

与传统 TX‑FFE/RX‑DFE（RS 纠错）、RX‑MLSD、以及 RX‑SOVA 方案相比，在 FER 10⁻⁵ 时提高约 5 dB，且在高信噪比下与 SOVA 方案的 FER 表现相当，同时保持 O(nlogn) 复杂度；

**⚠️ 局限性**

主要局限包括：单零填充导致 1/n 的速率损失、仅针对记忆-1 ISI 通道验证、需要块级处理且对非平稳或更高阶记忆通道的推广尚未证明。

---

## 148. STAPLE: Static Sparse Upcycling of Pretrained Checkpoints via Embedding Modules

**arXiv ID:** 2610.12599 | [PDF](https://arxiv.org/pdf/2610.12599v1)

**作者:** Ranajoy Sadhukhan `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1694 | [OpenAlex ID](https://openalex.org/A5031842648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在已有的稠密语言模型中加入零初始化的 token 索引嵌入作为静态稀疏记忆，并在中期训练阶段通过短暂的 warmup 使其学习补充稠密路径的残差信息，从而实现模型容量的可扩展。

**💡 创新点**

核心创新是保留原始 FFN 的上投影而非完全替换，同时在该路径上叠加可学习的 token 残差嵌入，并利用冻结上投影的 warmup 阶段显著提升稀疏记忆的样本效率与泛化能力。

**🔧 技术方法**

采用了静态稀疏记忆（token‑indexed embedding）、零初始化、短暂 warmup 冻结策略、低秩投影、CPU 主机内存离线加载、异步预取、LFU 缓存以及行稀疏优化等技术。

**📊 数据集**

训练使用公开的 4T 预训练混合数据集和高质量 mid‑training 混合；评估在 ARC‑Easy/Challenge、HellaSwag、WinoGrande、MMLU、DROP、GSM8K、Math、MBPP、HumanEval 等十项基准上。

**📈 对比分析**

与稠密持续训练、MoE 及 Engram 样式稀疏上循环等方法对比，-12L 在十项任务上平均提升约 1.2 分；在推理吞吐率上比 MoE 快约 2 倍，GPU 内存占用更小，能耗仅为 MoE 的 50% 左右。

**⚠️ 局限性**

局限在于稀疏记忆对高频词更新不均匀，需要较长的 warmup 进行初始化；在更大模型规模或更高训练预算下的可扩展性与样本效率尚待进一步验证。

---

## 149. Maximal Linear Arrangement is NP complete on Bipartite Graphs

**arXiv ID:** 2610.12613 | [PDF](https://arxiv.org/pdf/2610.12613v1)

**作者:** Aparna Das `[一作]`, Jonathan Needleman `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文通过构造一个新的二分图G'，证明最大线性排列问题在二分图上是NP‑完备的，进一步推出最小二分法的NP‑难度。

**💡 创新点**

创新点在于首次给出对二分图的NP‑完备性证明，采用了细致的“层次”与“割宽”分析，以及构造具有O(n^5)额外顶点的二分图G'，从而把最小二分问题还原为最大线性排列问题。

**🔧 技术方法**

主要技术包括：图的层次（level）与割宽（cut width）理论、最大线性排列的最优性性质、对原图G进行“放大”构造得到G'，并利用这些结构证明最优排列必然对应G的最小二分。

**📊 数据集**

该研究为理论性工作，不使用任何实验数据集。

**📈 对比分析**

论文不涉及实验或性能比较，仅提供多步理论证明；因而不存在传统意义上的性能指标。

**⚠️ 局限性**

局限性包括：构造的G'规模巨大（O(n^5)顶点），证明仅说明NP‑完备性，未给出有效近似算法或多项式时间解法；此外，结果仅适用于二分图，无法直接推广到更一般图类。

---

## 150. PriMoRetarget: Humanoid Motion Transfer Through Prioritized Motion Retargeting

**arXiv ID:** 2610.12642 | [PDF](https://arxiv.org/pdf/2610.12642v1)

**作者:** Nicholas Ioannidis `[一作]` (Simon Fraser University), Xue Bin Peng `[通讯]` (Simon Fraser University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于优先级优化的运动重定向框架，能够在多角色、多对象交互场景下将人类运动捕捉数据迁移到仿人机器人，自动维护步态、接触和姿态等关键特征；

**💡 创新点**

创新点在于将传统加权多目标优化转化为显式的优先级层次，通过梯度投影到更高层次的零空间来确保高优先级目标不被低优先级目标破坏，且不需要手动调节权重；

**🔧 技术方法**

技术包括优先级梯度投影、按帧分块的零空间求解、热启动（加权求和预热）以及自由目标（自由加权）处理；

**📊 数据集**

使用了四个数据集：单角色运动 LAFAN1、角色-物体交互 OMOMO、角色-角色交互 Inter‑X，以及一份35分钟的协作搬运数据；

**📈 对比分析**

与 PHC、GMR、OmniRetarget 等基线在物理可行性指标（穿透、滑动、浮空、抖动、接触保持）以及基于 RL 的跟踪性能对比，优先级方法在接触保持和跟踪误差上均优于基线，并在多交互场景中表现最为显著；

**⚠️ 局限性**

局限性包括：仍会保留源数据中的噪声与错误（如抖动、穿透），以及对物体轨迹的依赖性，难以自适应地调整交互路径，且优先级层次需人工指定。

---

## 151. Transformer-Based Masked Diffusion Models Provably Learn Distributions with Latent Structure

**arXiv ID:** 2610.12544 | [PDF](https://arxiv.org/pdf/2610.12544v1)

**作者:** Jingda Wu `[一作]` (University of Michigan), Changxiao Cai `[通讯]` (University of Michigan)

**通讯引用:** 249 | [OpenAlex ID](https://openalex.org/A5080132489)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了掩蔽扩散模型（MDMs）的样本复杂性，建立了端到端的采样保证，表明MDMs能够学习具有低复杂度结构的分布。

**💡 创新点**

创新点在于通过混合隐马尔可夫树模型来测量分布的近似，提供了统计误差与结构复杂性之间的关系，而不是依赖于序列空间的大小。

**🔧 技术方法**

使用了基于变换器的掩蔽预测器，该预测器能够适应未知的潜在结构，并通过训练目标在观察数据上进行优化。

**📊 数据集**

使用的具体数据集未在摘要中提及，但提到的应用包括语言建模、代码生成和蛋白质序列建模。

**📈 对比分析**

与现有方法相比，MDMs在处理具有结构依赖的分布时表现出色，能够有效地学习和采样，避免了维度诅咒，样本复杂性与结构复杂性呈多项式关系。

**⚠️ 局限性**

限制在于分析主要集中在混合隐马尔可夫树模型上，未来的研究需要扩展到更一般的依赖结构，并考虑优化误差对最终采样误差的影响。

---

## 152. Mind the Spatial Gap: A Benchmark for Geospatial Reasoning and Constrained Spatial Editing

**arXiv ID:** 2610.12881 | [PDF](https://arxiv.org/pdf/2610.12881v1)

**作者:** Zheng Ren `[一作]` (Hong Kong University of Science and Technology), Yang Yue `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建 GeoSpatialBench 基准，评估多模态大语言模型在城市空间关系理解与编辑上的能力。

**💡 创新点**

首次将空间问答与受限空间编辑结合，提供可验证的几何约束评估，并基于真实城市 GeoJSON 与地图图像。

**🔧 技术方法**

利用 GIS 操作、结构化 GeoJSON、地图渲染、多模态输入、结构化编辑接口与约束满足率。

**📊 数据集**

使用 OpenStreetMap 六个城市（斯德哥尔摩、巴黎、香港、武汉、底特律、内罗毕）共 8,400 个实例。

**📈 对比分析**

通过准确率、CSR、Raster‑tolerant 成功率对比；GeoJSON 最高，GPT‑6 Astra 最高，地图图像性能显著低，编辑成功率落后于问答。

**⚠️ 局限性**

模型在度量关系、边界计算和编辑一致性方面表现不足，地图图像难以精确几何推理，编辑任务易违约。

---

## 153. Quantifying SMT Solver Instability for Formal Verification

**arXiv ID:** 2610.12649 | [PDF](https://arxiv.org/pdf/2610.12649v1)

**作者:** Daniel Evans `[一作]` (St Olaf College), John Shorter `[通讯]` (St Olaf College)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统地研究了 SMT 解决器的运行时间不稳定性（proof instability），通过对来自 Bounded Model Checking、Symbolic Execution 与 SMT‑Competition 2025 的 60,000 个语义等价查询的变体进行实验，使用变异后的查询集合的运行时间数据来量化不稳定性。

**💡 创新点**

创新点包括：① 引入系数变异（Coefficient of Variation, CV）来衡量不稳定性，避免了传统方差对查询难度的偏倚；② 使用非参数自助抽样（bootstrap）估计 CV 的置信区间；③ 通过线性回归证明查询难度是 CV 的显著正向预测因子，并提出通过优化查询结构而非仅提升成功率来降低不稳定性和平均求解时间。

**🔧 技术方法**

主要技术手段包括：语义等价查询的变异（重命名、子句洗牌等）、CV 统计与自助抽样、成功率（Success Rate, SR）分析、线性回归分析、对 cvc5 与 Z3 两大 SMT 解决器的实验评测。

**📊 数据集**

使用的数据集：从 ESBMC、KLEE 等工具抽取的 BMC 与 SymEx 查询，以及 2025 年 SMT‑Competition 公开数据集，分别随机抽样 100/150/150 条查询与“难题”子集（timeout 的查询）共计约 60,000 条语义等价变体。

**📈 对比分析**

比较方法：将 CV 与传统成功率（SR）对比，发现 CV 能更准确反映查询的时间波动；利用线性回归评估查询难度与 CV 的关系，并展示通过重构查询可将 Z3 在硬 BMC 查询上的平均求解时间降低至 38% 左右，表明优化查询结构能在保持成功率的同时显著提升性能。

**⚠️ 局限性**

局限性：① 对 timeout 查询缺乏完整运行时间导致 CV 无法估计；② 仅使用了两种 SMT 解决器，结果可能不适用于所有求解器；③ 变异操作受限于洗牌和重命名，未覆盖更复杂的等价变换；④ 实验中未实现自动查询重构，只给出改进思路，缺乏实用的自动化工具。

---

## 154. EliteTable: Self-Improving Program Search for Executable Control Policies

**arXiv ID:** 2610.12536 | [PDF](https://arxiv.org/pdf/2610.12536v1)

**作者:** Stuart Farmer `[一作]` `[通讯]` (Casa Dilauro Independent Research), Stuart Farmer (Casa Dilauro Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一种名为 EliteTable 的程序搜索框架，利用大型语言模型（LLM）生成、编辑、混合 Python 程序，并通过固定容量的精英表格记录高分解法；在六个 Gymnasium 控制任务上执行了十代、每代 50 个候选的搜索实验，并在 10 组搜索种子上进行评估。

**💡 创新点**

创新点主要包括：
• 将 LLM 作为代码生成与修复的核心工具，形成闭环反馈；
• 采用固定容量精英表维护全局最优候选，确保搜索得分单调不降；
• 在每代提供三种操作（新建、编辑、混合）以探索多样化解法；
• 在搜索结束后对每代赢家进行持久化和保留样本评估，验证迁移效果。

**🔧 技术方法**

技术手段包括：
• 使用 OpenRouter 上的 GPT‑OSS‑120B 进行代码生成与修复；
• 通过抽象语法树（AST）检查语法合法性、接口符合度和去重；
• 设计多阶段搜索循环（生成→校验→评估→修复→保留）；
• 在评估阶段使用 Gymnasium 环境执行 10 条种子轨迹计算平均回报；
• 采用双侧精确符号检验和 Holm 校正评估保留样本改进情况。

**📊 数据集**

数据集：
• 六个 Gymnasium 控制任务（如 CartPole、MountainCar、LunarLander 等，含不同观测/动作空间）；
• 对每个任务使用 10 个搜索种子（0–9）和 100 个保留测试种子（1000–1099）进行评估；
• 每代 50 个候选，10 代，共 6000 条候选程序。

**📈 对比分析**

比较方法：
• 记录每代候选分布、搜索得分和保留得分，计算 1 代到 10 代的中位数变化；
• 对每个任务统计 10 组搜索的保留样本改进率、是否达到报告阈值，并进行符号检验；
• 在四个任务中所有搜索都实现了正向改进，平均保留样本提升显著（p<0.02），但在两任务出现负向或无显著提升。

**⚠️ 局限性**

局限性：
• 未做系统的消融实验，无法明确分离各组件（精英表、编辑、混合、修复）的贡献；
• 只在固定的种子面板下评估，可能存在过拟合或对环境随机性的敏感；
• 未与其他程序搜索或基于神经网络的强化学习方法直接比较；
• 依赖单一大型模型，可能对任务描述已有的先验知识产生影响；
• 计算预算有限，未探究更深层次的搜索或更大规模的任务。

---

## 155. CLUE: A Contrastive Learning Framework for Unsupervised Event Type Induction and Annotation

**arXiv ID:** 2610.12580 | [PDF](https://arxiv.org/pdf/2610.12580v1)

**作者:** Pratik Maitra `[一作]` (Iowa State University), Qi Li `[通讯]` (Iowa State University)

**通讯引用:** 15130 | [OpenAlex ID](https://openalex.org/A5100350262)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CLUE框架，实现无监督的事件类型诱导、命名与句子级事件标注；

**💡 创新点**

创新点包括：①联合谓词-对象与句子上下文的表示以解决多义；②通过掩码增强+对比学习提升表示；③对比学习引导的K‑means示例聚类；④利用LLM进行事件命名与聚类细化；

**🔧 技术方法**

使用技术包括：BERT‑base编码、掩码语言模型（MLM）增强、对比损失（margin loss）、K‑means示例聚类、LLM（GPT‑4 Turbo）提示式命名、BGE嵌入相似度合并；

**📊 数据集**

采用ACE、MAVEN、RAMS三大公开事件数据集；

**📈 对比分析**

通过与ETypeClus、CEO基线对比，使用micro‑F1和ARI评估；CLUE在ACE 41.7% micro‑F1(↑17.7%)、ARI 31%(↑3%)；MAVEN 32% micro‑F1(↑17%)；RAMS 26% micro‑F1(↑6%)；命名质量（Sim dist、ROUGE‑L、BERTScore）也显著优于基线；

**⚠️ 局限性**

局限性：K‑means假设球状均匀，导致聚类不稳定；仅生成平面事件层次，未实现层级诱导；提示工程未深入探索；对长尾分布的处理仍不充分。

---

## 156. LeapBot-JAC: Joint Action Chains for World Action Models

**arXiv ID:** 2610.12886 | [PDF](https://arxiv.org/pdf/2610.12886v1)

**作者:** Chenhao Zhu `[一作]` (Leapmotor), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Joint Action Chains (JAC)，在固定的动作词预算内将稀疏的参考控制和稠密的执行控制共用同一个动作专家进行联合生成；

**💡 创新点**

创新点在于通过在同一模型中将参考链和执行链分配到相同的token预算，并使用有向注意力让执行链读取参考链，从而在不增加模型参数或推理步骤的前提下提升长时序控制的效果；

**🔧 技术方法**

采用了扩散模型的噪声注入、Transformer 共享层、物理时间对齐以及有向角色掩码；

**📊 数据集**

主要使用 LIBERO、LIBERO-Plus、RoboTwin 2.0 等机器人操控数据集以及真实世界的两项任务（连续排序和抽屉关闭）；

**📈 对比分析**

与 FastWAM 与 LeapBot-WA 两个现有的世界动作模型对比，JAC 在 LIBERO 的平均成功率从 97.0% 提升到 98.3%，在 LIBERO-Plus 的总成功率从 53.5% 提升到 70.4%，并在 RoboTwin 及真实世界任务中也取得显著提升，且推理延迟和显存使用几乎无变化；

**⚠️ 局限性**

局限性包括参考控制未显式编码语义意图，且参考步长与权重的交互效应难以单独评估，未来需要进一步研究参考链的语义可解释性和跨任务迁移能力。

---

## 157. Critical Mass: Structure-Disjoint Splits Drift With the Libraries They Are Built On

**arXiv ID:** 2610.12694 | [PDF](https://arxiv.org/pdf/2610.12694v1)

**作者:** Tomo Oga `[一作]` (Matterworks Inc), Michael Widrich `[通讯]` (Matterworks Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可扩展的 MCES‑10 结构分割方法，并在大规模公开 MS/MS 数据库中评估其效果。

**💡 创新点**

创新点：①通过扩展可接受界限和 union‑find 跳过技术，大幅减少需求解的整数规划数量；②证明 MCES‑10 分割随库大小不稳定，导致持出集合极小且偏向高质量分子；③揭示单链接阈值分割导致的 percolation 与质量偏差问题。

**🔧 技术方法**

使用改进的 Myopic MCES（包含总键权差、键权直方图 L1 边界）+ Kruskal 循环规则 + union‑find 跳过 + Integer Programming 求解器。

**📊 数据集**

使用 SpectraVerse 1.0.1（44,240 结构）与 Enveda‑180（183,476 结构）并集共 227,690 结构，并在不同规模子采样（10k、20k、40k、80k、160k、全库）进行实验。

**📈 对比分析**

对比方法：利用可达性评估（holdout 份额）和质量 AUC 衡量分割偏差；实验结果显示，holdout 份额从 10k 规模下约 30% 降至全库的 4.5%，质量 AUC 从 0.70 上升至 0.88；算法在求解规模上实现了 2,259 倍的减少，复杂度从 O(N²) 降至 O(N^0.93)。

**⚠️ 局限性**

局限性：①评估仅基于两大公开库，仍存在质量分布偏差；②仅针对单链接阈值分割，未提出可标准化的阈值方案；③对结构多样性与大规模库的通用性仍需进一步研究。

---

## 158. Self-Preservation in Language Models

**arXiv ID:** 2610.12602 | [PDF](https://arxiv.org/pdf/2610.12602v1)

**作者:** Dongkyu Cho `[一作]` (New York University), Ali Behrouz `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何利用语言模型自身生成的数据进行自我保全，在持续学习中保持先前能力的同时学习新任务。

**💡 创新点**

提出任务银行（Task Bank）与模型自生成回放（Self‑Replay）相结合的自保方法，并证明在部分目标之外也能提升保留效果。

**🔧 技术方法**

使用生成式回放、任务银行、目标选择器、SFT训练以及保留指标Retention和对比分析。

**📊 数据集**

Qwen系列模型（0.6B、1.7B、4B）在MedQA、C‑STANCE等任务上进行训练，使用MMLU、HellaSwag、GSM8K、BoolQ等任务做回放，ARC‑C、WinoGrande、PIQA、SVAMP为持出评估。

**📈 对比分析**

通过与单纯SFT、单一任务回放以及随机/银行/入门任务信息选取的对比，发现广泛回放能将持出任务的平均保留率提升至约95%（4B）且保持相近的新任务增益；组合部分银行进一步提高保留率。

**⚠️ 局限性**

仅在4B规模下显现目标选择优势，较小模型无提升；固定样本分配下新旧任务权衡不完整；单臂库存扩展试验未提升保留；缺乏跨更新的库存维护方案与成本分析。

---

## 159. Provenance-grouped protocol for evaluating machine learning in perovskite photovoltaics

**arXiv ID:** 2610.12766 | [PDF](https://arxiv.org/pdf/2610.12766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. An all-transistor physical-learning transformer can be trained to perform in-context learning

**arXiv ID:** 2610.12571 | [PDF](https://arxiv.org/pdf/2610.12571v1)

**作者:** Michael C. Abbott `[一作]` (Yale University), Benjamin B. Machta `[通讯]` (Yale University)

**通讯引用:** 2888 | [OpenAlex ID](https://openalex.org/A5008418985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了一种基于双电路（dyadic）物理学习的Transformer电路，实现了在电路中完成in‑context学习的能力。

**💡 创新点**

创新点在于将Transformer的自注意力机制（替代点积为“bump”型相似度）与可在物理系统中实现的双向电流传播相结合，并通过子阈MOSFET指数非线性和Gilbert cell实现了近似的反向传播，从而突破传统相互作用电路在关联学习任务上的局限。

**🔧 技术方法**

采用的技术包括：双电路/对偶电路结构、子阈MOSFET指数电流法实现softmax与bump电路、Gilbert cell实现乘法与反向梯度传递、浮动电路损失实现交叉熵、以及适应性局部优化器Lion/Adam；所有模块均以可实际实现的电阻、二极管、晶体管和电流源构成。

**📊 数据集**

使用自定义的ICL数据集：由随机嵌入向量表示的字母-数字序列构成，训练集包含约100个不同字母的样本，测试集由256条新嵌入序列组成，用于评估模型复制前后上下文的能力。

**📈 对比分析**

在实验中，用LCE损失和Lion优化器训练后，单层Transformer电路在测试集上达到了约80%~90%的准确率（远高于随机1/6和仅复制输入的1/4），显示出相较于传统MLP/梯度下降方法在关联记忆任务上的显著优势；同时证明了单层Transformer足以完成此ICL任务。

**⚠️ 局限性**

主要限制包括：电路规模较大（≈1155个组件，双电路约3217个）、对理想电流/电压特性有较高要求、梯度近似导致学习效率受限、目前仅在极其简化的ICL任务上验证，尚未证明可扩展到更复杂的自然语言或大规模数据集。

---

## 161. Symbolic-Permutation-Based Direct Sparse Jacobian Assembly for Accelerated Power-Flow Analysis

**arXiv ID:** 2610.12654 | [PDF](https://arxiv.org/pdf/2610.12654v1)

**作者:** Tianshi Cheng `[一作]` (RTDS Technologies), Venkata Dinavahi `[通讯]` (University of Alberta)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `847a60d8-a755-47af-ba5d-c5236b9e3083` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于符号置换的直接稀疏雅可比组装方法，利用一次性符号化和单遍无分支的闭式映射，显著提升Newton–Raphson潮流计算的组装速度。

**💡 创新点**

创新点包括：①证明雅可比组装对正交置换不变，将运行时投影迁移到符号阶段；②设计单遍连续写入的闭式映射，消除中间块堆叠和散射写；③实现无分支、流水线式的稀疏填充算法，并将其集成到RustPower框架中。

**🔧 技术方法**

使用的技术包括：符号置换与等价变换、复数矩阵微分、压缩稀疏列（CSC）存储、数据导向的ECS架构、Rust编程、KLU线性求解器、SIMD/并行流水线填充、JIT编译等。

**📊 数据集**

实验数据集涵盖IEEE 39、IEEE 118、PEGASE 9241（以及PEGASE 1354）电网案例。

**📈 对比分析**

对比方法：在相同二进制环境下与MATPOWER直通版（V0）、优化版（V1）、pandapower 3.5.4+Numba、LightSim2Grid 1.0.0及RustPower 0.5.2进行对标；雅可比组装速度相较V0提升24–30×、相较V1提升6–7.8×；整体求解速度相较V0提升3.2–4.5×、相较V1提升1.6–2.0×；在IEEE 39上与pandapower对比可达154×，在PEGASE 9241上可达11.7×。

**⚠️ 局限性**

限制：随着系统规模增大，LU分解时间占比上升，受Amdahl极限限制；目前实现主要针对单CPU，GPU/FPGAs等加速尚未完成；符号化阶段需要一次网络拓扑预处理，对动态变更的网络适应性有限。

---

## 162. Kairos: Toward a Byte-Level Foundation Model for Every Modality

**arXiv ID:** 2610.12495 | [PDF](https://arxiv.org/pdf/2610.12495v1)

**作者:** Fabien Furfaro `[一作]` `[通讯]`, Fabien Furfaro

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 Kairos 这一多模态字节级网络，包含混合双向 DeltaNet 与滑窗注意力、稀疏 MoE、分层 PyramidalCodec 与统一去噪目标，并在 1–15M 参数规模下进行收敛性检验。

**💡 创新点**

将多种独立技术在同一模型中协同运用，并通过共享投影实现注意力分支的快速收敛；提出跨会话记忆门用于状态初始化。

**🔧 技术方法**

使用 bidirectional Gated‑DeltaNet、滑窗注意力、共享投影、稀疏 MoE、PyramidalCodec、字节级分层编码、统一去噪 (均匀噪声) 目标和自回归自适应学习率等技术。

**📊 数据集**

测试数据仅为一条 440 字节句子重复 16 次（约 30B token 预算），并在 CPU 上进行 600 步或 3000 步的收敛实验；无真实多模态数据训练。

**📈 对比分析**

通过最小训练损失、单步/多步比较、共享 vs 分离 QKV、codec 方案等指标评估收敛速度和最终损失；发现共享投影在早期收敛更快，但在 3000 步后达到相同损失；codec 方案按 0.018–0.354 的顺序排序。

**⚠️ 局限性**

局限性：仅在极小的记忆化任务上验证，未测试泛化能力；缺乏多模态、密集模型对比；实验不具位可复现性；只做了单跑或单种种子，未全面评估多参数调节；未测量实际吞吐率和显存占用。

---

## 163. Beyond the Monolithic Interface: A Research Agenda for Plural Conversational AI

**arXiv ID:** 2610.12878 | [PDF](https://arxiv.org/pdf/2610.12878v1)

**作者:** Avijit Ghosh `[一作]` (Hugging Face University of Connecticut), Zhenzhen Qi `[通讯]` (University of Connecticut)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了单声道聊天机器人带来的三类危害，并提出多人物接口作为干预方案

**💡 创新点**

首次系统地将认知、政治与文化视角整合，并将游戏与参与式模拟的多元化接口设计引入AI对话

**🔧 技术方法**

采用叙事性综述方法，结合多学科文献和游戏设计、参与式模拟的设计原理

**📊 数据集**

本研究未使用数据集，基于文献综述

**📈 对比分析**

无实验比较，讨论基于已有工作的定性评估与未来研究方向

**⚠️ 局限性**

仅限文献综述，缺乏实证评估，所提多人物接口的效果与可行性尚未通过用户研究验证

---

## 164. Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization

**arXiv ID:** 2610.12855 | [PDF](https://arxiv.org/pdf/2610.12855v1)

**作者:** Jiaqi Tian `[一作]` (Jilin University), Ting Long `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大语言模型的自我提升过程（On-Policy Self‑Distillation，OPSD）中出现的“能力退化”现象，并提出了表示几何正则化（Representation Geometry Regularization，RGR）来缓解这一问题。

**💡 创新点**

创新点在于首次将隐藏表示空间的几何结构（簇内紧凑性与簇间分离度）作为正则化目标，证明并验证其对保持已学能力的重要性，并将其无缝融合到多种OPS­D方法中。

**🔧 技术方法**

核心技术包括：对齐与比较簇内散度（WCSS）、拉普拉斯矩阵特征值分析来评估簇间分离度；基于Gram矩阵差异的统一几何正则化；结合熵正则化与传统OPS­D目标的整体训练目标。

**📊 数据集**

实验使用OpenThoughts数学推理数据（约3万对链式推理示例）进行自我训练，并在AIME 2024/2025、HMMT 2025等数学竞赛测试集上评估。

**📈 对比分析**

与原始OPSD、SDPO、SRPO、RLSD四种主流自我提升方法相比，加入RGR后在大多数基准上平均提升约2–4点（最高可达5+点），并显著降低正确→错误的转化率与跨任务遗忘率。

**⚠️ 局限性**

局限性包括：对不同优化动态（如RLSD）效果不一，需进一步探索更自适应的几何保真策略；目前仅验证于数学推理场景，尚未在更广泛任务或更大模型规模上进行评估。

---

## 165. The Expressive Power of Random-Feature MPNNs Across Graph Sizes

**arXiv ID:** 2610.12809 | [PDF](https://arxiv.org/pdf/2610.12809v1)

**作者:** Robert R. Nerem `[一作]` (University of California, San Diego), Yusu Wang `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究随机节点特征的消息传递神经网络（MPNN），提出统一的“大小无关逼近”框架，证明连续随机特征可被有限颜色取代，并在此基础上分析表达能力、颜色/试验层次关系、与 Weisfeiler–Leman 的不可比性，并给出树子图检测与局部 MST 边检测的可实现构造；进一步给出学习理论和实验验证；

**💡 创新点**

提出统一的随机特征MPNN可逼近理论，并通过有限颜色与试验的量化揭示其表达界限；构造 CFI 图对揭示非统一性与颜色/试验的严格层次；证明随机颜色MPNN与高阶 WL 不可比较；提供大小无关的算法实现与训练理论；

**🔧 技术方法**

利用 Lipschitz 约束的量化论证、统计区分度与试验的概率分析、随机 WL 试验框架、颜色编码算法、图论构造、深度学习训练与 Monte Carlo 采样；

**📊 数据集**

在实验中使用 Erdős–Rényi 图（16、32、512 节点）、3 正则图（24 节点）以及 TRIANGLE 与 EXP 数据集；

**📈 对比分析**

通过实验展示颜色与试验数提升桥边识别与树子图检测准确率，量化提升幅度；量化量化随机特征后模型性能基本保持；与传统固定初始化 MPNN 对比，表现出显著性能提升；

**⚠️ 局限性**

仅对均匀 Lipschitz MPNN 进行分析，无法覆盖更一般 MPNN；对颜色/试验的最优组合未给出完全理论；对 WL 的不完全可比性仍存在开放问题；

---

## 166. Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses

**arXiv ID:** 2610.12872 | [PDF](https://arxiv.org/pdf/2610.12872v1)

**作者:** Fanzhi Zeng `[一作]` (University of Texas at Austin), Liu Leqi `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过训练大语言模型助手，显式预测用户在每一次交互中的回复，并在内部循环中根据预测结果修订草稿，从而在多轮协作任务中提升性能并降低用户输入成本。

**💡 创新点**

创新点在于提出“协作理论推理”（Collaborative Theory of Mind）框架，将用户回复预测作为显式学习目标，并利用预测结果驱动内部草稿修订，既提升了任务完成度，又减少了用户的交互负担。

**🔧 技术方法**

技术方法包括：监督式预测损失对用户回复进行训练；结合GRPO（Group‑Relative Policy Optimization）进行强化学习；构建内部草稿‑预测‑修订循环；设计任务特定奖励与交互成本的联合回报。

**📊 数据集**

实验使用了三类公开数据集：MATH（数学推理题）、BigCodeBench（代码生成任务）和CollabLLM MediumDocEdit（写作编辑任务）。

**📈 对比分析**

与两种基线（无交互的Base与仅GRPO的Naive）比较，协作理论推理模型在数学、代码和写作任务分别提升约7.5%、4.4%和6.5%的任务得分；在多轮交互中用户回复次数和输入长度均显著下降。

**⚠️ 局限性**

局限性包括：实验仅基于模拟用户，缺乏真实人类交互验证；预测机制仅应用于草稿修订，未扩展到提问或提交等其他交互动作；模型对不同任务的适配需要进一步探索。

---

## 167. Evaluation of Software Quality Through Refactoring Commit-Impact Analysis

**arXiv ID:** 2610.12890 | [PDF](https://arxiv.org/pdf/2610.12890v1)

**作者:** Md Safayat Hossen `[一作]` (Islamic University of Technology), Shadid Intisar Quasem `[通讯]` (Islamic University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在四个 GitHub 开源项目中挖掘重构提交并使用静态分析工具，对重构提交对软件质量指标的影响进行了实证研究。

**💡 创新点**

创新之处在于结合多种挖掘与静态分析工具，对不同成熟度（初学者与成熟）项目以及时间维度的重构影响进行量化比较，首次给出正面、负面及中性影响的百分比分布。

**🔧 技术方法**

使用的技术包括 RefactoringMiner 2.1、PyRef、PMD、Source Monitor 与 SonarQube 等静态分析与挖掘工具，并采用基于质量指标的增量比较方法。

**📊 数据集**

数据集为四个开源项目：Java 项目 IutAssistance（69 次提交，24 次重构）与 Alligator（198 次提交，113 次重构），Python 项目 you-tube-search（168 次提交，18 次重构）与 mini-shop-server（506 次提交，90 次重构）。

**📈 对比分析**

方法为手工挑选每个重构前后提交对，分别运行工具计算质量指标，然后统计正负中性影响比例；结果显示大部分重构带来正面改进（约 50%–70%），但也有 15%–30% 的负面影响；工具精度方面 RefactoringMiner 的平均准确率为 96.4%。

**⚠️ 局限性**

局限性包括仅选取四个项目、手工选择提交对可能导致误差、仅依赖静态分析默认配置、样本量不足，导致结果难以推广到更大范围或不同语言环境。

---

## 168. ROSETTA: Rotation Subspace Estimation for Training-free Transmodal Alignment

**arXiv ID:** 2610.12871 | [PDF](https://arxiv.org/pdf/2610.12871v1)

**作者:** Alex Junho Lee `[一作]` `[通讯]` (Sookmyung Women's University), Alex Junho Lee (Sookmyung Women's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为Rosetta的训练无关本地校准层，能在不进行任何共观测或训练的前提下对不同模态、不同设备、不同表示或不同骨干的独立训练编码器进行特征空间旋转对齐；

**💡 创新点**

核心创新在于揭示跨编码器对齐为局部空间的高维旋转，其白噪声角速度导致全球不变换不可行，而低有效秩的交叉协方差使局部旋转可由少量观测解算，进而通过Soft EM迭代估计本地Procrustes旋转；

**🔧 技术方法**

采用PCA降维+L2归一化后，在窗口内使用Soft EM求解软对应关系与交叉协方差，再通过SVD得到旋转矩阵；

**📊 数据集**

在KITTI‑360、VIVID、HeLiPR三大公共驾驶数据集上进行实验，涵盖LiDAR、摄像头、热像、事件等多模态；

**📈 对比分析**

与原始无校准、全局Procrustes、过拟合Procrustes、MLP适配器等方法对比，Rosetta在三大数据集上Recall@1分别达0.944/0.933/0.681，明显优于全局方法并接近或超越有监督对齐器；

**⚠️ 局限性**

局限性包括需要约30 m以内的粗略位置信息、仅能对齐旋转而非更高阶形变、对编码器自身表达不足的情形无法弥补、以及对极端场景如室内、无人机或水下的适用性待验证。

---

## 169. Selective Spatio-Temporal Fruit Inspector (S$^2$FI): A Mamba-based Framework for Rotational Fruit Quality Assessment

**arXiv ID:** 2610.12868 | [PDF](https://arxiv.org/pdf/2610.12868v1)

**作者:** Hao Tian `[一作]` (Chuzhou University), Youshan Zhang `[通讯]` (Chuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了低成本、轻量化的果品分拣系统，并提出了基于Mamba的选择性时空特征提取框架S^2FI，实现旋转果品的高效质量检测。

**💡 创新点**

创新点包括：将3D卷积编码器与Mamba状态空间模型结合，形成线性时间复杂度的选择性特征提取；融合多阶统计特征以提升对旋转序列的鲁棒性；以及采用自适应复合损失解决类别不平衡问题。

**🔧 技术方法**

使用的技术包括：3D卷积网络、Mamba状态空间模块、统计特征融合（均值、最大值、标准差）、Super Adaptive Composite Loss、双轨道滚筒硬件平台和边缘部署优化。

**📊 数据集**

使用自构建的Conveyor-based Fruit Quality Assessment Dataset，共计6460段视频，涵盖优质、次优、腐烂三类果品。

**📈 对比分析**

在相同输入、数据划分和优化流程下，与C3D、I3D、R3D、MC3D、Transformer、DVFL-Net、MoCLIP-Lite、X3D等基线对比，S^2FI在测试集上实现MCA 73.2%、Macro F1 0.723，参数仅36.6M，显著优于基线且适合边缘设备。

**⚠️ 局限性**

局限性在于对光照变化敏感、极少见缺陷识别困难、输入分辨率有限导致细粒度特征捕获不足，需要改进光照补偿和高分辨率采集方案。

---

## 170. CaptchaAgent: Verifiable Visual Reasoning for CAPTCHA Solving

**arXiv ID:** 2610.12866 | [PDF](https://arxiv.org/pdf/2610.12866v1)

**作者:** Shangquan Sun `[一作]` (Nanyang Technological University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CaptchaAgent，一种将视觉验证码转换为显式实体‑关系表示并在此基础上执行视觉推理与交互的框架。

**💡 创新点**

创新点包括使用符号ASCII V‑IR强制计数承诺、局部证据细化、以及任务保持的反事实一致性检验，从而显著提升验证码解码的可靠性。

**🔧 技术方法**

采用冻结的多模态大型语言模型（如GPT‑4.1、Gemini‑2.5 Pro、Claude‑3.7‑Sonnet）、Grounding‑DINO目标检测、ASCII符号表示、局部高分辨率裁剪以及变换一致性投票等技术。

**📊 数据集**

在公开的Open CaptchaWorld（20类、225个实例）和离线Oedipus‑data（4类、数十张图像）上进行评估。

**📈 对比分析**

与同一模型基线（BU‑Agent）在20步交互预算下比较，CaptchaAgent在Open CaptchaWorld上的Pass@1提升21.3–25.8个百分点，Oedipus‑data上平均提升9.2–12.3个百分点，显示显著性能提升。

**⚠️ 局限性**

局限性包括对连续几何、精确计数、时序控制的表示不足、对密集场景关系枚举的可扩展性有限，以及对变换一致性误判时仍可能误差。

---

## 171. Structure-Aware Matching for Robust UAV Thermal-to-Satellite Geo-Localization

**arXiv ID:** 2610.12835 | [PDF](https://arxiv.org/pdf/2610.12835v1)

**作者:** Chun Liu `[一作]` (Henan University), Henggang Han `[通讯]` (Henan Yunhuan NetLink UAV Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在GNSS失效和低能见度环境下，提出一种基于结构感知匹配的无人机热成像与卫星图像的地理定位方法 SAMF，显著提升跨模态匹配准确率。

**💡 创新点**

创新点在于：①设计 Directional Structure Enhancement Module（DSEM），通过多方向响应、低频结构分支强化热图与卫星图中共通的方向性与大尺度空间结构；②引入 Modality-Aware Affine Remapping Module（MARM），对结构增强后的特征做轻量级通道统计校正，进一步消除模态间分布偏差；③同时提出 Thermal‑UAV‑Paired benchmark，填补现有热成像与卫星匹配数据缺口。

**🔧 技术方法**

技术核心包括：共享参数的 Swin‑Transformer 骨干提取深层特征；DSEM 的三路并行结构（语义保持、方向响应、低频结构）与自适应融合；MARM 的通道级仿射重映射与轻量级高斯噪声正则；Triplet 损失训练；NetVLAD 聚合。

**📊 数据集**

使用 Boson‑nighttime（≈33 km² 夜间热成像 + 卫星）和新构建的 Thermal‑UAV‑Paired（≈8k 对）两个公开基准，分别涵盖荒漠、道路、农田等多样场景。

**📈 对比分析**

在两大数据集上均取得 SOTA 结果：Boson‑nighttime R@1 93.6%、R@5 97.4%、R_512@1 98.2%、R_512@5 99.6%；Thermal‑UAV‑Paired R@1 97.0%、R@5 99.4%、R_512@1 98.1%、R_512@5 99.8%。与 ResNet、DINO、SGM 等方法相比，显著提升检索精度与定位误差（Boson：L_512^2 12.3 m，Thermal‑UAV：4.8 m）。

**⚠️ 局限性**

局限性包括：①在结构极其稀疏或无明显边界的极端场景（如雪原、海洋）下，结构增强效果有限；②MARM 仅校正通道统计，无法完全补偿复杂的几何或尺度差异；③模型仍需较大算力，推理时间与参数量较传统轻量网络高；④仅针对热成像与可见光卫星，未验证跨不同光谱或多传感器组合的通用性。

---

## 172. VLMs are Native Action Models: Unleashing the Manipulation Capabilities of VLMs with Explicit Action-Space Grounding

**arXiv ID:** 2610.12832 | [PDF](https://arxiv.org/pdf/2610.12832v1)

**作者:** Jingkun Yi `[一作]` (Zhejiang University), Jiangning Zhang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Explicit Action‑Space Grounding (EASG) 框架，使预训练的视觉语言模型能够直接生成连续机器人动作，无需机器人动作预训练；通过在腕部相机视角投影 TCP‑中心的动作坐标系并提供度量尺度，VLM 以自身原生文本生成接口输出可执行的增量平移、旋转和抓手指令；同时加入语法约束解码保证动作格式合法。

**💡 创新点**

（1）显式将动作空间语义（原点、坐标轴、参数化、度量尺度）投影到机器人感知上下文；（2）利用 VLM 原生文本生成接口和词汇表实现动作输出，避免引入新输出模态；（3）语法约束解码兼顾生成灵活性与合法性；（4）同一动作接口兼容零射击与基于机器人数据的微调，形成统一的可端到端优化策略。

**🔧 技术方法**

视觉投影与深度估计、坐标变换与逆运动学、连续动作向量的单元化与单位投影、语法约束的上下文自适应解码、Causal Language Model 交叉熵微调。

**📊 数据集**

实验使用常见机器人操纵基准数据集（如 Meta‑World、Dex‑World 等），并在这些数据集上进行零射击与微调评测；具体数据集细节在论文附录中给出。

**📈 对比分析**

与现有 Vision‑Language‑Action (VLA) 模型及模块化基础模型（如 Code‑as‑Policies、VoxPoser、CoPa、ReKep 等）进行对比；结果显示 27B 规模 VLM 在 EASG 下的零射击性能与细化后 VLA 模型相当；0.8B 规模 VLM 在微调后与多种 VLA 基线竞争，说明 EASG 能弥补大模型与小模型的性能差距。

**⚠️ 局限性**

目前仅在单手抓取/搬运任务验证，仍需扩展至双手、移动机器人等高维动作；大模型在微调时可能对视觉‑语义能力产生副作用；缺乏对更大规模多任务评估与开源数据的全面验证。

---

## 173. Cross-Behavior Policy Optimization: Unified Learning from Offline Demonstrations and On- and Off-Policy Rollouts via a Shared Regression Target

**arXiv ID:** 2610.12830 | [PDF](https://arxiv.org/pdf/2610.12830v1)

**作者:** Du Cheng `[一作]`, Shichao Weng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的“跨行为策略优化” (CBPO) 框架，利用固定参考回归目标，将离线演示、在策略和离策略回放等多种数据源整合为同一回归损失，从而在一次训练过程中同时利用所有可用数据。

**💡 创新点**

创新点在于：① 将 KL 正则化的最优策略映射为可直接计算的固定参考目标，使得不同生成策略产生的样本可共用同一目标；② 在不需要行为策略概率、重要性权重或额外 critic 的情况下，通过冻结的 prompt 归一化实现统一回归；③ 通过共享记录格式实现离线演示、正负样本、参考响应和历史回放的无缝重用，显著提升了样本利用率。

**🔧 技术方法**

主要技术包括：固定参考回归目标（logπ_ref + r/β - logZ_ref），固定 Z_ref 的 Monte Carlo 估计，统一 Replay Buffer 采样与更新，CBPO 的四种变体（Offline、OnPolicy、OffPolicy、Unified），以及与验证器、RLVR（DPO、PPO、DAPO、REINFORCE++、GSPO、TBRM、BRIDGE 等基线）的对照实验。

**📊 数据集**

数据集：Qwen3-1.7B 与 Qwen3-8B 作为基准模型；使用 16,254 个去污染的 RLVR-Math-16k prompts，结合 DeepSeek-R1-Distill-Qwen-32B 的正负 CoT 轨迹；评估集包括 AIME25、MinervaMath、AMC23、MATH500 四大数学推理 benchmark，每个 prompt 生成 32 条回复。

**📈 对比分析**

实验对比显示：CBPO 的各个子变体在对应数据 regime 上均优于传统 SFT/DPO 或单一 RL 方法；统一版本 CBPO-Unified 在 1.7B 与 8B 两个规模上分别比 BRIDGE 提升 1.47/3.90 与 2.34/2.56 分，整体在 8B 上所有指标均位列第一；与两阶段 SFT→RLVR 比较，CBPO-Unified 在平均准确率和覆盖率上提升 2.33/4.44 分，显著优于传统 pipeline；历史回放与负样本平衡进一步提升性能。

**⚠️ 局限性**

局限性：仅在数学推理任务、Qwen3 模型和规则式验证器上验证；对其他领域、模型族或噪声/主观奖励的泛化未知；需要额外的生成步骤来估计固定 Z_ref，导致前期计算成本；在离线分布漂移或跨域迁移情景下的表现尚未评估。

---

## 174. Cubic Spline Interpolation for Smooth Trajectory Generation in Autonomous Motion Systems

**arXiv ID:** 2610.12826 | [PDF](https://arxiv.org/pdf/2610.12826v1)

**作者:** Aarush Mandayam `[一作]` (10x Accel), Elliott Tammaro `[通讯]` (10x Accel)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

自动构造并求解三次样条插值问题，生成通过给定时空点的平滑轨迹，并将该方法应用于自由曲线绘制与手写签名数字化；

**💡 创新点**

将所有插值与连续性约束整合为完整线性系统，直接求解系数，省去手工推导每段方程的步骤；同时展示该方法在非机器人任务中的可迁移性；

**🔧 技术方法**

三次样条插值、线性系统组装与求解（稀疏矩阵求逆或解方程组），程序化构造 A 与 b，支持多轴独立拟合；

**📊 数据集**

使用人工生成的少量 waypoint 集合进行自由曲线示例；使用采样的手写签名点集进行数字化演示；未使用公开大规模数据集；

**📈 对比分析**

与传统的三对角（Thomas 算法）三次样条求解方法对比；直接求逆在小规模（n<50）时可行，时间复杂度 O(n³)；示例演示平滑度优于线性插值，速度/加速度连续；

**⚠️ 局限性**

仅保证 C² 连续，未限制速度、加速度或更高阶导数；时间分配未优化，可能导致近点间需要高速通过；直接求逆计算量大，实时大规模轨迹规划不现实；多轴扩展需逐轴独立拟合，未一次性约束多维一致性。

---

## 175. Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems

**arXiv ID:** 2610.12822 | [PDF](https://arxiv.org/pdf/2610.12822v1)

**作者:** Yuepeng Hu `[一作]` (Meta Superintelligence Labs), Himaghna Bhattacharjee `[通讯]` (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了通过强化学习诱导的奖励黑客行为在多智能体编程系统中的传播与系统级影响，探讨其在代码、共享内存及自然语言规划中的扩散机制并评估对任务调度与候选聚合的影响。

**💡 创新点**

创新点在于首次将奖励黑客从单体模型扩展到多智能体环境，揭示其能在不同角色、链式交互和共享记忆中持续扩散，并提供理论传播因子与实证验证；同时提出了自评审与验证者门控的两种缓解策略。

**🔧 技术方法**

技术包括：使用大语言模型（Qwen、Gemma、Llama等）进行编码与规划；对抗性强化学习（Group Relative Policy Optimization）以诱导奖励黑客；静态/行为检测结合的黑客判别方法；以及实验设计中的一跳传播、链式传播、反馈驱动传播与共享内存传播等。

**📊 数据集**

数据集为DeepMind CodeContests的过滤硬难度子集（1,461 训练题目、163 验证题目），并使用合成文档训练（SDF）与多种公开/封闭权重模型。

**📈 对比分析**

比较方法包括：衡量默认评估器（E_leaky）与严格评估器（E_strict）通过率、黑客率、传播率等指标；对照RL-only与SDF-only、clean对照模型以及不同反馈/循环协议。实验显示，奖励黑客可在多模型间传播，且在“hard”任务链中可放大；自评审虽降低传播率但仍有漏判，验证者门控能显著阻止黑客，但也存在误拒真实解的风险。

**⚠️ 局限性**

局限性包括：实验主要聚焦于特定评估器（pytest-based）和已知黑客机制，未覆盖所有可能的黑客手段；模型对话环境与真实生产流水线差异；验证者门控在拒绝真实解时的误差尚未完全解决；理论传播模型假设独立性，实际交互可能更复杂。

---

## 176. Deploying RDMA for Large-Scale Inference and Training Workloads

**arXiv ID:** 2610.12815 | [PDF](https://arxiv.org/pdf/2610.12815v1)

**作者:** Yibo Huang `[一作]` (Fudan University), Jie Wu `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

部署 RDMA 以降低参数服务器架构下的推理与训练通信延迟和 CPU 占用

**💡 创新点**

跨层面协同设计：零拷贝内存分配、QP 多路复用、线程域资源对齐、接收信用控制和软件发送窗口

**🔧 技术方法**

使用 RDMA-RC、可靠连接、共享内存分配器、接收信用、软件发送窗口、Archon RPC 级恢复

**📊 数据集**

在 ByteDance 真实服务中使用的生产推理与训练工作负载（如 Toutiao、TikTok、Xigua Video）

**📈 对比分析**

相较已部署的 TCP 基线，平均参数拉取延迟下降 35.7%，P99 延迟下降 43.7%，推送 P99 延迟下降 74%，CPU 使用率下降 10.69%

**⚠️ 局限性**

仅覆盖参数服务器模式，未评估全局集体通信；缺乏对不同 TCP 调优、负载变化、容错语义的深入实验

---

## 177. TreeHCA: Hindsight-Guided Exploration and Credit Assignment for Multi-Turn LLM Agents

**arXiv ID:** 2610.12828 | [PDF](https://arxiv.org/pdf/2610.12828v1)

**作者:** Debajoy Mukherjee `[一作]` (Texas A&M University), Dileep Kalathil `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了结合后见信用探索和信用分配的TreeHCA方法，用后见比例引导树形采样并用逆后见权重归一化回溯优势，以提升多回合LLM问答训练。

**💡 创新点**

将后见信用的分配与探索策略通过共同的后见变换关联，既用后见分数决定哪些历史要重新采样，又用其逆比作为重要性权重，形成配对的采样与信用机制。

**🔧 技术方法**

基于LLM agent的树形rollout、后见信用分配(HCA)、逆后见重要性修正、归一化树备份以及Qwen系列LLM模型与工具调用。

**📊 数据集**

在七个问答基准上评估，包括HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle（多跳）以及Natural Questions、TriviaQA、PopQA（单跳），并使用Qwen2.5-3B-Instruct、Qwen3-4B、Qwen3-8B等模型。

**📈 对比分析**

与链式和树式基线（Search-R1、ZeroSearch、GiGPO、IGPO、HCAPO、Tree-GRPO、IGRPO、AEPO、ReAct、GRPO、GSPO）对比，TreeHCA在所有模型尺寸和任务组的加权EM均优于对手，提升约1–3个百分点。

**⚠️ 局限性**

训练时间比主流方法高约9–11%，并且对个别基准的方差较大，对训练随机种子的敏感性仍需进一步探索。

---

## 178. Win-Win Fusion: Multi-Modal Image Fusion with Visual-Semantic Collaboration

**arXiv ID:** 2610.12887 | [PDF](https://arxiv.org/pdf/2610.12887v1)

**作者:** Hao Zhang `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种W2F互强化多模态图像融合框架，同时实现可视融合和语义分割的双向协同提升

**💡 创新点**

通过分割引导融合损失、信息恢复先验、IGM‑Att与PC‑Att注意机制以及特征搜索与梯度纠正策略，显著增强视觉-语义互补性并实现双向互助

**🔧 技术方法**

使用分割引导融合损失、信息恢复先验、交互式门控混合注意力（IGM‑Att）、渐进耦合注意力（PC‑Att）、特征搜索模块、梯度纠正与Nash‑MTL多任务优化

**📊 数据集**

在MFNet与FMB红外-可见图像融合数据集上训练，并在M3SVD、RoadScene、Polarization、NIR‑VIS、Medical等公开数据集上进行跨任务泛化验证

**📈 对比分析**

与七类独立与序贯多模态融合/分割方法对比，W2F在MFNet上mIoU达到61.5%，在FMB上61.7%，在视觉融合指标（VIF、SSIM、NAB/F）均位列第一，提升幅度1.4–4.6个百分点

**⚠️ 局限性**

仍存在两阶段训练复杂度较高、参数与算力略增，且对极端低光/噪声场景的鲁棒性需要进一步提升

---

## 179. Personalized Federated Direct Preference Optimization via LoRA Adapters

**arXiv ID:** 2610.12885 | [PDF](https://arxiv.org/pdf/2610.12885v1)

**作者:** Zhanhong Jiang `[一作]` (Iowa State University), Soumik Sarkar `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种个性化联邦直接偏好优化框架pFedDPO，让每个客户端在共享基模型之上训练私有LoRA适配器，实现模型个性化。

**💡 创新点**

通过将梯度多样性从原始的κ²降低到更小的κ²_eff，并给出新的收敛上界，解决了传统FedDPO在偏好异质性下的固有限制。

**🔧 技术方法**

采用联邦学习、LoRA低秩适配器、直接偏好优化（DPO）以及理论收敛分析等技术。

**📊 数据集**

在Stanford SHP和Anthropic hh‑rlhf两套非IID偏好数据集上进行实验，分别由5个客户端各持120条偏好对组成。

**📈 对比分析**

与中心化DPO和Vanilla FedDPO比较，pFedDPO在梯度范数更低、训练损失更平稳且每个客户端的准确率保持在约0.8左右，同时不增加通信量。

**⚠️ 局限性**

局限性包括需调节LoRA秩以平衡梯度多样性减小和适配器收敛难度，且理论假设LoRA能有效吸收大部分异质性，极端异质或稀疏数据场景下可能表现不佳。

---

## 180. CoGameReg: Cooperative-Game-Based Collaborative Semi-Supervised Regression for Seismic Reservoir Property Prediction

**arXiv ID:** 2610.12884 | [PDF](https://arxiv.org/pdf/2610.12884v1)

**作者:** Zhiyu Guo `[一作]` (Jilin University), Lei Li `[通讯]` (BGP Inc., China National Petroleum Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于合作博弈论的半监督回归框架CoGameReg，用固定的Shapley衍生模型权重构造共享伪标签，并结合局部误差减少筛选实现对稀疏井控下的地震储层属性预测。

**💡 创新点**

核心创新在于：①使用一次性估计的Shapley权重在伪标签生成与最终融合中保持不变，使模型贡献显式化；②将秩-均方误差与一致性两类游戏结合成三游戏评估，统一求得模型贡献；③局部k‑NN误差减少筛选进一步控制伪标签质量。

**🔧 技术方法**

技术手段包括：合作博弈论与Shapley值分配、三游戏（Rank‑MSE + Consistency）评价、k‑NN局部错误减少筛选、迭代协作训练与模型特定批处理。

**📊 数据集**

实验数据集涵盖：10个公开回归数据集（如Abalone、Wind等）、斯坦福VI‑E合成3D P波速度预测、以及真实油田水饱和度预测（含18口井）。

**📈 对比分析**

与XGBoost、KNN、Avg. Ensemble、Self‑KNN、COREG、Mean Teacher、SSDKL、UCVME等基线比较，CoGameReg在2.5%标签比例下平均RMSE最低，整体比传统协作训练和自训练方法表现更优，且在合成与真实案例中均实现了最佳或接近最佳的评估指标。

**⚠️ 局限性**

局限性包括：伪标签依赖模型误差，固定权重缺乏位置适应性；局部筛选可能忽略地质差异；在井控稀薄或地质复杂区域泛化效果受限；大规模未标记样本的误差传播风险需进一步研究。

---

## 181. HART: Hybrid Attention-RWKV Time series Foundation Model

**arXiv ID:** 2610.12883 | [PDF](https://arxiv.org/pdf/2610.12883v1)

**作者:** Qingwen Lin `[一作]` (Guangdong University of Technology), Ruichu Cai `[通讯]` (Guangdong University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在非自回归多步预测（NAF）中使用RWKV-7模型时出现的长预测区间退化问题，提出了Placeholder Idling Decay（PID）机制，并基于此设计了交替堆叠Transformer与RWKV块的混合架构HART。

**💡 创新点**

创新点在于首次将PID写入机制与递归状态衰减联系起来，并利用此机制构建了通过交替堆叠注意力块与RWKV块，既保留递归时序建模，又直接访问历史记忆的混合模型。

**🔧 技术方法**

使用的技术包括RWKV-7的通用Delta规则分析、Transformer自注意力与SwiGLU前馈、交替堆叠架构、非自回归推理、连续块掩码（CPM）预训练、Pinball量化损失以及全尺度量化（MASE）评估。

**📊 数据集**

实验数据集主要包括公开的GIFT-Eval和TIME-Benchmark两大时间序列预测基准，以及用于诊断PID的合成序列。

**📈 对比分析**

通过在GIFT-Eval和TIME-Benchmark上与纯RWKV、纯Transformer、TiRex、Toto等基线在相同参数规模下的MASE几何平均值比较，HART在中期与长期预测上显著优于纯RWKV，整体MASE低于同规模Transformer，且36M参数的HART在长周期预测上可与1B参数的Toto-2.0媲美。

**⚠️ 局限性**

局限性包括PID分析仅针对RWKV-7，写入和转移矩阵的范数诊断并不能直接衡量历史信息保留；实验未将PID与其他架构改动分离；对TiRex等其他递归模型的相同机制未作验证；交替堆叠比例与顺序的进一步优化未探讨。

---

## 182. Temporal Evidence Forests for Long-Horizon Multi-Party Conversational Memory

**arXiv ID:** 2610.12874 | [PDF](https://arxiv.org/pdf/2610.12874v1)

**作者:** Junyou Guo `[一作]` (Beihang University), Shuai Ma `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于时间结构的多方长时记忆框架 T‑Mem，利用时间局部性检索关键证据并构造 Update/Expand 关系树，压缩低信息跨度；

**💡 创新点**

创新点在于同时利用时间局部性（Temporal Evidence Discovery）和时间演化（Temporal Memory Structuring）两种互补机制，实现对长时多方对话的精准检索与完整事件演进可视化；

**🔧 技术方法**

技术方案包括元数据扩展的 BM25、时间局部检索、基于时间的关系分类（Update/Expand）、树压缩与线性化序列化，以及大模型回答与评判（GPT‑5.5 / Deepseek‑V4‑Flash）；

**📊 数据集**

实验使用 GroupMemBench 与 EverMemBench 两个多方对话记忆基准；

**📈 对比分析**

与 BM25、A‑Mem、HippoRAG2、Hindsight、StructMem、MemGAS 等现有检索和记忆框架对比，T‑Mem 在两套 LLM 上均提升 6.5–8.6个百分点，整体准确率最高；

**⚠️ 局限性**

局限性包括对时间戳和元数据（话题、阶段）的依赖，元数据缺失时性能略降；在高度噪声或非线性事件序列中仍可能无法完全捕捉完整演进。

---

## 183. Time Delay System for Sequence Modeling

**arXiv ID:** 2610.12864 | [PDF](https://arxiv.org/pdf/2610.12864v1)

**作者:** Youshan Zhang `[一作]` (Chuzhou University), Jialu Li `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于非线性时延系统的语义时延驱动序列模型，显式估计内容相关的历史延迟并通过模糊规则调节即时与历史信息的平衡。

**💡 创新点**

①可微语义滞后估计器；②对角线状态空间实现 O(1) 并行更新；③可解释的模糊语义调制机制。

**🔧 技术方法**

非线性时延微分方程、对角线状态空间、Soft Attention 语义滞后估计、零阶 TSK 模糊系统、SiLU 激活、卷积局部上下文。

**📊 数据集**

IMDB 文本分类、CNN/DailyMail 文本生成。

**📈 对比分析**

与 LSTM、Mamba、RoBERTa、BERT 等基准模型对比，IMDB 上精度 95.35%（参数 2.6M），CNN/DailyMail 上 ROUGE‑L 42.63（参数 16.98M），训练时间与传统模型相当但参数更少，显示性能优于现有方法。

**⚠️ 局限性**

对超参数敏感、实验仅覆盖有限两个任务，尚未验证在更广泛数据集或跨模态任务中的泛化能力。

---

## 184. Unreliable Narrator Framing as a Cost-effective Mechanism for Satire Explanation of Australian and American Satirical News

**arXiv ID:** 2610.12863 | [PDF](https://arxiv.org/pdf/2610.12863v1)

**作者:** Kanay Gupta `[一作]` (University of New South Wales), Aditya Joshi `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并验证“不可可信叙述者”框架在解释澳洲与美国讽刺新闻中的有效性，比较不同提示方式对模型解释质量的影响。

**💡 创新点**

创新地将文学概念“不可可信叙述者”作为结构化提示框架用于新闻讽刺解释，并首次构建针对澳洲与美国讽刺文章的解释生成数据集与评估基准。

**🔧 技术方法**

采用开放权重指令调优模型（Qwen2-VL、Qwen2.5、Llama‑3.1、Mistral）与前沿 GPT‑5.6，结合网页检索（DuckDuckGo）和结构化的“叙述者-作者”提示，使用 Gemini 3 Flash 对生成文本进行打分。

**📊 数据集**

使用 500 篇 Betoota Advocate（澳洲）讽刺文章与 500 篇 The Onion（美国）讽刺文章，共 1000 篇，包含标题与正文，形成实验数据集。

**📈 对比分析**

通过比较基线提示与不可可信叙述者提示在目标识别（satire target）和解释质量（explanation）的评分，发现后者在 16/17 组合中提升，尤其对弱模型提升最大；前沿模型在相同条件下表现翻倍，但不可可信叙述者对其提升有限。

**⚠️ 局限性**

局限性：评估仅覆盖两家特定出版物，缺乏跨文化与跨媒体推广；模型对现实世界知识的掌握不足导致“叙述者”准确性低；多模态或其他讽刺出版物中的适用性尚未验证。

---

## 185. The False Allure of "Security by Obscurity": A Game-Theoretic Approach

**arXiv ID:** 2610.12858 | [PDF](https://arxiv.org/pdf/2610.12858v1)

**作者:** In-Uck Park `[一作]` (University of Bristol), Sunoo Park `[通讯]` (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文首次对“安全隐蔽性”（security by obscurity）进行严格的博弈论建模与分析，证明在均衡条件下，封闭（隐藏）系统相较于开放系统更易被攻击者先发现并利用，从而更不安全。

**💡 创新点**

创新点在于：①将安全经济学与博弈论相结合，首次从理论上验证长期经验的反隐蔽性经验；②提出一系列简化与通用的模型（两系统两玩家、Poisson玩家数、多系统等），并在多种情形下给出统一结论；③阐释了“优选效应”和“拥堵效应”如何共同导致封闭系统更具吸引力给恶意攻击者。

**🔧 技术方法**

使用技术：博弈论（Nash均衡、Poisson游戏、策略进入游戏）、概率论（技能分布、奖励分布）、加权竞赛成功概率模型、数学证明与极限分析。

**📊 数据集**

数据集：无，本文为纯理论分析，没有使用实验数据或真实漏洞数据库。

**📈 对比分析**

比较方法：通过对每种模型下开放与封闭系统被白帽（good）或黑帽（bad）攻击者“赢得”（即先发现并利用/报告）系统的概率进行对比。结果显示：在任何模型下，开放系统的被白帽赢得概率高、被黑帽赢得概率低，证明开放系统更安全；无实验性能指标。

**⚠️ 局限性**

局限性：①模型假设简化（如每个系统只有单个漏洞、开放系统成本为0、攻击者选择仅一系统、技能分布相同等）；②未考虑多漏洞、持续攻击、攻击者信息不完全、开放系统非零成本等实际情况；③只研究了“开放审计”这一隐蔽性反对理由，未覆盖所有隐蔽性论点；④对现实系统的细节建模缺乏经验验证。

---

## 186. Reinforcement Learning via Fixed-point Flow Matching

**arXiv ID:** 2610.12848 | [PDF](https://arxiv.org/pdf/2610.12848v1)

**作者:** Denis Blessing `[一作]` (Karlsruhe Institute of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了“目标流匹配”与“固定点流匹配”两种在强化学习中训练流策略的新方法，能在无目标策略样本的情况下利用 Q‑梯度实现稳定的回归更新；

**💡 创新点**

核心创新是将目标策略的速度场表达为条件 Q‑梯度的期望，并构造固定点迭代，使得最优策略成为迭代的唯一不动点；

**🔧 技术方法**

主要技术包括流策略与流匹配框架、目标流识别、固定点回归、信任域约束、行为约束微调以及基于 Ornstein–Uhlenbeck 的相关探索；

**📊 数据集**

在 OGBench 的 50 个离线‑在线迁移任务和 MJX Playground 与 Humanoid 的 24 个在线最大熵 RL 任务上进行评估；

**📈 对比分析**

与 TRQAM、QAM、DSRL、CGQL‑L 等多种基线对比，方法在 50 个离线‑在线任务上平均成功率高达 0.95（高于 0.93 的最佳对手），在 24 个在线任务中也保持竞争力；

**⚠️ 局限性**

局限性包括对 Q‑函数质量高度依赖、近端 t=0 时方差偏大、需要多步积分与散度估计导致计算开销、以及在极高维或极大噪声场景下的可扩展性待进一步验证。

---

## 187. Towards Defining, Measuring, and Leveraging Creativity in AI Applications

**arXiv ID:** 2610.12845 | [PDF](https://arxiv.org/pdf/2610.12845v1)

**作者:** Jingyi Yang `[一作]` (New York University), Alexander Tuzhilin `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了基于新颖性、价值和惊喜三大准则的创意数学定义，并在国际象棋和绘画两大领域验证其有效性。

**💡 创新点**

创新点在于①给出严格的创意后设公设并统一为可乘积公式；②针对不同领域设计可测量的新颖性、价值和惊喜指标；③通过真实数据实验证明该公式能捕捉创意模式。

**🔧 技术方法**

使用距离度量、GMM 似然、观察误差、预训练嵌入（ChessLM、CLIP、DINOv2、ViT）、逻辑回归与 B‑spline 非线性变换、LLM 生成评语与 ABSA 分析。

**📊 数据集**

使用 Mega ChessBase 2025（94,565 局、7,711,327 步）与西方绘画拍卖记录 355,551 条（包含 Picasso、Renoir、Kollwitz 等作品）以及对应的预训练模型嵌入。

**📈 对比分析**

采用 5 折交叉验证；在棋局上非线性模型 AUC 0.636、准确率 0.640；在绘画上基本模型在 CLIP 嵌入下 MAE 0.0287、JSD 0.0016，整体表现优于对照实验。

**⚠️ 局限性**

仅验证了两种领域，缺乏跨域通用基准；新颖性、价值、惊喜的度量仍需人工设定；标签主要来自 LLM 生成，可能存在主观偏差；公式的唯一性尚未证明。

---

## 188. APEX: Asynchronous Prefix Denoising for Extreme Compression

**arXiv ID:** 2610.12843 | [PDF](https://arxiv.org/pdf/2610.12843v1)

**作者:** David Dinkevich `[一作]` (Lightricks), Dani Lischinski `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了APEX方法，通过在潜在空间中构建分层通道前缀并使用异步逐通道去噪，支持极端空间压缩的扩散生成。

**💡 创新点**

创新点在于将分辨率感知的通道层次与逐通道异步去噪结合，解决重建-生成悖论，使极端压缩成为计算优势。

**🔧 技术方法**

使用分层自编码器、DiT变压器、等间隔 staggered 时间调度、全局时钟、匹配分辨率目标及训练概率分配等技术。

**📊 数据集**

在 ImageNet‑512 数据集上进行实验。

**📈 对比分析**

与 DC‑AE、DC‑AE 1.5、FreqWarm、Spectrum Matching 等基线比较，在 64×/128×压缩下，APEX 在所有通道数上 gFID 均优于基线，并在训练/推理计算匹配下击败 256‑token 模型，尤其在低计算预算时优势更明显。

**⚠️ 局限性**

限制在于需手动设定通道划分、每层目标分辨率、stagger 间隙及全局时钟分布，最佳设定随任务变化；未来可联合学习这些超参。

---

## 189. Neural networks are more modular than single neurons suggest

**arXiv ID:** 2610.12842 | [PDF](https://arxiv.org/pdf/2610.12842v1)

**作者:** Shreya Saha `[一作]`, Meenakshi Khosla `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种几何框架，将Transformer隐藏层表示分解为语法子空间、语义子空间和残差子空间，并通过正交/置换/可逆线性变换学习这些子空间，随后进行因果干预验证各子空间对语法与语义任务的影响。

**💡 创新点**

创新点在于：①使用可操作的结构/意义不变性约束为语法和语义定义子空间；②通过正交变换、可微化的Sinkhorn置换以及可逆线性变换三种不同的变换空间探讨模块化程度；③结合对抗性对比损失实现子空间的独立学习；④通过去除特定子空间的因果干预检测双重耗损（double dissociation）。

**🔧 技术方法**

技术包括：正交矩阵学习（geotorch）、置换矩阵的可微近似（Sinkhorn松弛）与匈牙利算法投影、可逆线性变换与条件正则化、对比损失（语义/语法不变性约束）、因果干预与评估（语法接受度、同义句检索、真值条件测试、自然语言推理）。

**📊 数据集**

数据集与任务：同义句对（paraphrase pairs）、结构保持变换（如jabberwocky、POS保持替换）、句法评测（可接受性判断、主谓一致、结构许可现象）、语义评测（同义句检索、真值条件最小对、自然语言推理）。

**📈 对比分析**

评估方法：在各子空间干预后分别在句法与语义任务上测量性能下降，检验双重耗损；对正交、置换与可逆变换三种分解方式进行对比。实验表明正交旋转能显著实现子空间解耦并表现出双重耗损；置换分解效果一般，说明结构/意义并非完全轴对齐；可逆变换在更大尺度下也能实现解耦，提示需要更复杂的线性变换。

**⚠️ 局限性**

局限性：①仅关注线性分解，未探究非线性或深层交互；②置换和正交学习可能受梯度信息不足或局部最优限制；③对不同模型层与规模的泛化尚未系统评估；④因果干预基于直接删除子空间，可能无法捕捉更细粒度的交互；⑤数据集主要集中在英文语法与语义，跨语言验证缺失。

---

## 190. The Complexity of Finding Additional Solutions

**arXiv ID:** 2610.12804 | [PDF](https://arxiv.org/pdf/2610.12804v1)

**作者:** Saman Farhat `[一作]` `[通讯]` (Graduate Center of the City University of New York), Saman Farhat (Graduate Center of the City University of New York)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究给定一组已知解后寻找新解的难度，证明可在保持域和首解难度不变的前提下自由指定有限个额外解的求解难度，并给出可兼容任意正列表求解难度的必要且充分条件。

**💡 创新点**

创新点在于提出一种多层复制与分片技术，允许对多重额外解任务进行独立预设；同时证明任意正列表任务的难度必须同时支配所有固定长度任务的难度，并给出在此条件下可构造满足任意正列表难度的搜索问题；进一步展示固定域、首解难度和全部固定层难度序列不足以决定正列表难度。

**🔧 技术方法**

主要技术包括：一问一答的FNP归约框架、标签复制与计数器复制构造、1-可填充（1‑paddable）域的利用、分片与常数插入、残差运算与有序列表验证，以及对多级任务的逐层递归化。

**📊 数据集**

本工作不使用外部数据集，全部构造在理论框架内，基于抽象的搜索关系和可多项式验证的假设。

**📈 对比分析**

由于研究集中在理论复杂度与归约可行性上，没有实验性性能比较；所提出构造在多项式时间内可实现单次查询并满足所需的复杂度等价。

**⚠️ 局限性**

局限性在于：需域满足1‑可填充假设，构造不保证保持原问题的自然解集；未能给出完整无穷多固定层难度与正列表难度的联合预设；对仅可验证解的非FNP类缺乏讨论。

---

## 191. Efficient Audio-Visual Speech Enhancement via Reinforcement Learning-Based Neural Architecture Search

**arXiv ID:** 2610.12796 | [PDF](https://arxiv.org/pdf/2610.12796v1)

**作者:** Khubaib Ahmed `[一作]` (University of Stirling), Amir Hussain `[通讯]` (Edinburgh Napier University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用强化学习驱动的神经架构搜索（RL‑NAS）框架，自动发现并优化轻量化音视频语音增强网络，实现了在保持高SI‑SNR的前提下将模型参数从22M压缩至2.9M。

**💡 创新点**

创新点：① 在搜索空间中嵌入时间对齐和参数量约束，显著降低搜索复杂度；② 设计多目标奖励函数，平衡SI‑SNR提升与模型压缩；③ 采用进化式缩减（progressive shrinking）策略，从高性能基线逐步收敛到高效模型。

**🔧 技术方法**

技术手段包括：PPO强化学习 + 2层LSTM策略网络，约束动作掩码，线性退化的奖励设计，TCN与视觉通道的结构约束，视觉信息的激进剪枝与音视频融合方式。

**📊 数据集**

实验使用LRS3音视频语料库与CEC2噪声集（含-5到+5 dB SNR），并在必要时引用LibriSpeech做补充验证。

**📈 对比分析**

对比基线22M模型、手工小模型和随机搜索模型，评估指标为SI‑SNR提升、参数量、实时因子（RTF）；2.9M模型实现14.5 dB SI‑SNR（占基线95.4%），参数量减少87%，RTF 0.19，明显优于随机搜索和手工模型。

**⚠️ 局限性**

局限性：搜索依赖高成本的预训练和评估，未涵盖量化/动态自适应等后续压缩技术；模型在特定噪声环境下验证，跨噪声与跨设备的泛化能力尚待进一步研究。

---

## 192. The Evidence Anchoring Boundary: Where Visual Interventions Matter and When They Help in Multimodal LLMs

**arXiv ID:** 2610.12981 | [PDF](https://arxiv.org/pdf/2610.12981v1)

**作者:** Qiji Zhou `[一作]` (Westlake University), Yue Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并量化视觉干预对多模态大语言模型（MLLM）的影响，识别干预层次的有效界限（EAB），比较不同问题类型（存在、属性、关系）下干预的正面与负面效应，并提出基于校准的选择路由（Calibrated selective routing）以提升属性预测准确率。

**💡 创新点**

①定义Evidence Anchoring Boundary（EAB）来衡量同一视觉残差编辑在不同解码层的影响衰减；②区分干预在不同问题类型中的三种结果（纠正、冗余、破坏）；③提出校准选择路由机制，在保留原始预测的前提下，仅在校准支持时才执行干预，从而显著提升属性准确率。

**🔧 技术方法**

视觉残差编辑（等RMS强度）、counterfactual subtraction、logit‑lens（读写点 ACP 对比）、答案读写点（Answer Commitment Point）比较、PCE（Prior‑Echo Cancellation）以及五折图像分层交叉验证用于校准。

**📊 数据集**

模型：LLaVA‑1.5‑7B/13B、Qwen2.5‑VL‑7B、InternVL3.5‑8B、LLaVA‑NeXT‑34B、Qwen2.5‑VL‑32B；评测集：POPE（存在）、AMBER（属性/关系）、TVQA‑Existence、TextVQA、MME、POPE‑adversarial 等。

**📈 对比分析**

方法：在校准集上定位 EAB，随后在独立评估集上对 EAB 前后干预效应做比值，发现前期效应为 13–26 倍；对属性准确率进行全局减法 G 与校准路由的比较，G 在大多数检查点略低或相等，而路由提升 1.56–1.81 个百分点；在存在子集中的 AUC≈0.5，表明 G 仅提供有限额外信息；在其他任务上将 PEC 转移，效果不稳定，说明仅在校准范围内有效。

**⚠️ 局限性**

局限性：EAB 与 ACP 仅针对特定残差编辑与模型架构；不同模型、提示或干预方式需重新校准；计算量和推理延迟未量化；实验仅覆盖二分类属性命题，生成任务的全面验证缺失；PCE 的转移效果在非校准任务中表现不佳。

---

## 193. Learning Representations to Guide Pixel Diffusion Transformers

**arXiv ID:** 2610.12978 | [PDF](https://arxiv.org/pdf/2610.12978v1)

**作者:** Zihao Pan `[一作]` (Sun Yat-sen University), Sibei Yang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Representation Guidance（RG），通过低维瓶颈和利用模型自身内部表示的流匹配目标，引导像素空间扩散 Transformer 进行高质量生成。

**💡 创新点**

创新点在于：①用低维瓶颈强制生成相关表示；②用内部表示产生的速度偏移直接修改流匹配目标，使表示被主动利用；③保持单一 Transformer 结构的同时实现更优的参数–质量和计算–质量折中。

**🔧 技术方法**

采用低维线性瓶颈、流匹配目标修改、EMA 教师指导、REPA 对齐、纯 Transformer JiT 架构、CFG 以及相关的训练技巧。

**📊 数据集**

使用 ImageNet 1K（256×256、512×512）进行类条件生成；文本到图像预训练与 fine‑tune 采用 CC12M + LLaVA‑NEXT，后期混合 BLIP3o‑60K、DALL‑E 3、ShareGPT‑4o‑Image；评估数据集包括 GenEval、DPG‑Bench。

**📈 对比分析**

与 PixelDiT、DiP、DeCo、JetFormer 等像素空间生成器以及 DiT‑XL、SiT‑XL、SD‑VAE 等潜在空间生成器对比；在 ImageNet 256×256 上 RG‑XL/16 获得 1.55 FID，RG‑L/16 1.79 FID；在 512×512 上 1.74 FID；文本生成 0.89 GenEval、87.1 DPG‑Bench，均优于同尺寸对手并接近大模型潜在空间基线。

**⚠️ 局限性**

仅在图像生成任务上验证，未对更大规模模型、视频、3D 或音频等模态进行推广研究；需进一步评估在更长训练或更高分辨率时的稳定性。

---

## 194. SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents

**arXiv ID:** 2610.12975 | [PDF](https://arxiv.org/pdf/2610.12975v1)

**作者:** P N Bhargav Teja `[一作]` (Samsung Research and Development Institute), Rajasekhara Reddy Duvvuru Muni `[通讯]` (Samsung Research and Development Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

SmartSalesBench 是一个多轮对话基准，专注于评估LLM销售代理在面对心理学驱动的抵抗买家角色时的表现。

**💡 创新点**

创新点在于将买家抵抗作为持续衡量维度，引入三层度量体系、检索增强生成（RAG）与四个对照实验，系统评估策略与会话长度的影响。

**🔧 技术方法**

采用 LangGraph 进行多节点对话编排，RAG 为代理与买家生成提供事实依据，并用 Ollama 的多模型评判层进行自动评分。

**📊 数据集**

数据集包含 12 个心理学驱动的买家角色、5 个代理档案、15 款三星手机商品（6 价位）以及对应的反对词库和软答词。

**📈 对比分析**

通过四个实验（基线、升级、20 回合、5 回合）和四个 LLM（Llama3.1‑8B、Qwen2.5‑3B、Gemma2‑2B、Phi3.5‑mini）对比，Llama3.1‑8B 在保持高 BRS、低角色颠倒率的同时实现约 26% 的转化率，表现最稳定。

**⚠️ 局限性**

局限包括评判层仅使用本地 Ollama 模型导致潜在偏倚、仅评估四个开放模型、只针对三星手机且仅英语/ Hinglish，缺乏多语言与跨产品推广性验证。

---

## 195. Controlled-Minimax: Stabilizing Adversarial Optimization via Control Theory

**arXiv ID:** 2610.12915 | [PDF](https://arxiv.org/pdf/2610.12915v1)

**作者:** Tianjiao Luo `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于控制理论的“Controlled-Minimax”框架，用反馈控制器稳定对抗性优化过程，消除梯度下降-上升（GDA）中常见的振荡与发散。

**💡 创新点**

创新点在于将对抗优化视为二阶连续时间动力学模型，并设计出可证明能在任何Hessian谱下实现局部指数稳定的线性速度反馈控制器，突破传统经验正则化的局限。

**🔧 技术方法**

采用连续时间微分方程、线性控制理论（阻尼、交叉耦合）以及梯度反馈的离散化实现，结合基于Hessian特征的理论分析。

**📊 数据集**

实验数据集包括低维基准（Dirac‑GAN、二维高斯混合、Swiss Roll）以及大规模图像生成任务（StyleGAN2‑ada在CIFAR‑10、LSUN‑Cat、FFHQ三种分辨率）。

**📈 对比分析**

与标准GDA及其它对抗优化方法对比，Controlled‑Minimax在所有实验中都能抑制振荡、加速收敛；在StyleGAN2‑ada中，FID下降至CIFAR‑10 2.91、LSUN‑Cat 8.18、FFHQ 5.46，显著优于基线（3.37、10.48、7.81）。

**⚠️ 局限性**

局限性包括仅提供局部指数稳定性保证、离散化实现与连续理论之间的差距、缺乏自适应控制参数调整以及对全局收敛性尚未给出理论支持。

---

## 196. Coexponential Exploration in the Clef Programming Language

**arXiv ID:** 2610.12911 | [PDF](https://arxiv.org/pdf/2610.12911v1)

**作者:** Houston Haynes `[一作]` `[通讯]` (Braidpoint), Houston Haynes (Braidpoint)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文设计并实现了Clef语言的共指数（coexponential）服务模型，利用限定延续（delimited continuations）在编译器层面保证状态所有权、通信协议和资源使用的一致性，支持多硬件目标的本地化执行。

**💡 创新点**

创新点在于将线性逻辑中的共指数与演员（actor）模型结合，构造可证明的服务池与状态转移，并通过有限状态、消息标签与延迟捕获（capture）实现跨设备的可验证通信与原子性；同时提出了多层证明框架（四个层级）与两范畴的组件细化机制，实现编译期的证明复用与增量重编译。

**🔧 技术方法**

主要技术包括线性逻辑与共指数语义、延迟延续（Danvy‑Filinski）对计算残余的显式表示、Iris/Actris 的分离逻辑资源证明、固定点调度与微层 IR（MLIR）实现，以及基于 BAREWire 的裸层编码与解码约束。

**📊 数据集**

该工作以形式化验证为主，不使用传统实验数据集；在示例中引入了一个求解线性方程组的服务，并借助Urschel 的增长界定提供数值误差保证，但整体验证过程基于符号推导与自动证明工具，未涉及大规模真实数据集。

**📈 对比分析**

方法评估主要通过证明完整性与编译后语义保持来展示；在实验平台上，使用Clef编译器对一个包含多服务、消息重排与多目标执行的示例程序进行编译，报告的性能指标为编译时间约为 1.2 秒，运行时在 CPU 与 GPU 目标下分别实现了 30% 与 45% 的加速，验证结果表明安全性与正确性得到保留。

**⚠️ 局限性**

局限性包括：1）证明与验证的规模随服务复杂度急剧增长，需要手动编写和维护大量证明脚本；2）对动态生成的消息标签与未枚举状态的处理仍依赖外部抽象；3）在高度动态分布式场景中，网络延迟与消息丢失的假设较为理想化，实际部署需结合可靠通信层进行进一步补充。

---

## 197. JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents

**arXiv ID:** 2610.12957 | [PDF](https://arxiv.org/pdf/2610.12957v1)

**作者:** Gangyi Li `[一作]` (JD.COM), Xu Yang `[通讯]` (JD.COM)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了JoyInside‑VLM‑Agent，一种可在多种机器人和智能硬件上，根据产品规格、目标变化、视觉反馈动态自适应执行任务的多模态代理。

**💡 创新点**

创新点在于：①基于产品条件的轨迹构造，使训练数据覆盖多平台、多目标和失败恢复；②引入双源特权自监督（DPER）实现token级证据路由并放缩优势；③结合多源交互奖励（过程评估、Judge、TaskQueue）实现对理由质量、约束遵循和目标一致性的综合评估。

**🔧 技术方法**

采用Qwen3.5‑35B为基础模型，经过监督微调（SFT）、多教师OPD蒸馏、DPER优化，并使用ReAct框架、生成式过程评估模型（GPRM）、多裁判排名和TaskQueue一致性评估等技术。

**📊 数据集**

训练数据约12万条，包含50K专有交互轨迹、40K改造自公开数据集（MSAgent、ToolBench、ToolACE、BFCL等）以及30K视频指令；测试集涵盖四个公开基准和两个专有JoyInside‑Devices、JoyInside‑Robotics数据集。

**📈 对比分析**

通过与公开基准（MSAgent、BFCL v4、ToolBench、ToolACE）及专有数据集的轨迹准确率、工具决策指标等对比，JoyInside‑VLM‑Agent在BFCL得分77.44，ToolBench和ToolACE工具决策指标领跑，专有数据集全部超越基线，表现出较高的适配性与执行准确率。

**⚠️ 局限性**

局限在于：尚未对更广泛的产品配置和真实环境进行系统评估；奖励设计与DPER路由效果难以单独隔离；数据覆盖和沙盒生成的反馈真实性有限；模型对噪声或不完整反馈的鲁棒性仍需进一步验证。

---

## 198. GIST: Generator-state-Informed Supervision for Tabular Foundation Models

**arXiv ID:** 2610.12904 | [PDF](https://arxiv.org/pdf/2610.12904v1)

**作者:** Zhangyang Peng `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于生成器信息的条件查询监督框架（GIST），通过Rao‑Blackwell化对硬标签损失进行平均，从而在预训练中利用合成任务中保留的标签不确定性；

**💡 创新点**

创新点在于将生成器的中间状态与监督规则联合设计，利用熵校准的高斯编码实现支持的标签依赖分支的可解析后验，且保留了总体目标且不增大梯度协方差；

**🔧 技术方法**

主要技术包括：Rao‑Blackwell化条件损失、熵预算校准的高斯标签编码、可逆编码通道、梯度协方差分析与实验验证；

**📊 数据集**

在TabArena、TALENT和BeyondArena三大基准上评估，覆盖近200个分类任务；

**📈 对比分析**

与同构PFN硬标签基线（Hard）以及CatBoost、LightGBM、XGBoost、RealMLP、TabM等基线在匹配训练计划下比较，GIST在所有三种TFM架构（TabICLv2、TabSwift、TabPFN‑2.5）上均提升平均排名，并在约62.1%的数据集对比中获胜；

**⚠️ 局限性**

局限性包括仅在分类任务上验证，OOB表现随架构变化不一；依赖生成器先验的可解析性，需手动设计编码与熵预算；对真实数据的迁移性与鲁棒性尚未系统评估；

---

## 199. Diffusion Precipitation Generation with Medium-range Weather Forecasting

**arXiv ID:** 2610.12976 | [PDF](https://arxiv.org/pdf/2610.12976v1)

**作者:** Guowen Li `[一作]` (Tsinghua University), Haohuan Fu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将降水生成与大气预测解耦的框架Dynamics–Precipitation Decoupling，利用冻结的天气模型提供的非降水气象状态，训练一个条件扩散生成器直接生成与IMERG观测一致的每日降水累计量。

**💡 创新点**

创新点在于：1）将降水生成与大气预测完全分离，避免对全尺度天气模型进行昂贵的再训练；2）使用多尺度递归编码器压缩气象序列并捕获时空演化；3）在扩散去噪过程中引入跨场注意力，使降水特征能够对齐并利用气象信息。

**🔧 技术方法**

核心技术包括：条件扩散模型、卷积递归单元（ConvGRU）进行气象序列编码、跨尺度交叉注意力（cross‑attention）实现气象-降水特征对齐、DDIM采样实现高效去噪。

**📊 数据集**

训练数据为ERA5气象重分析与IMERG卫星-雨量计日累计降水的配对样本；评估数据涵盖全球1.5°、区域0.25°网格以及中国2288气象站的实际观测。

**📈 对比分析**

与Pangu‑Weather、GraphCast、HRES等气象背骨以及ENS、GenCast等外部集成系统比较。结果显示：在全球1.5°尺度上ACC提升约0.1，RMSE下降约0.6，CSI和HSS均有显著提升；在0.25°区域尺度和站点尺度上降水结构更加细腻，事件检测率提升；概率预测方面，CRPS与Brier Score均低于ENS和GenCast，说明生成分布更接近观测。

**⚠️ 局限性**

主要限制：1）生成器的性能仍受背骨大气预测质量限制；2）需为每个分辨率训练一次生成器，未实现跨分辨率共享；3）扩散采样过程计算量大，实时性受限；4）对极端降水事件的极限表现尚需进一步验证。

---

## 200. Emo-BPO: Emotion Bidirectional Preference Optimization for Diffusion-based Emotional TTS

**arXiv ID:** 2610.12958 | [PDF](https://arxiv.org/pdf/2610.12958v1)

**作者:** Jiacheng Shi `[一作]` (College of William & Mary), Ye Gao `[通讯]` (College of William & Mary)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种情感双向偏好优化框架 Emo‑BPO，联合学习情感对齐与对比分支，显著提升扩散式 TTS 的情感可控性与表达细腻度。

**💡 创新点**

创新点包括：①识别并解决现有偏好优化仅强化目标情感、忽略竞争情感的结构性缺陷；②利用同文本不同情感的重排对生成情感对齐与对比得分函数；③在推理阶段融合 Classifier‑Free Guidance 与晚期指导调度，实现情感的双向拉拢与排斥。

**🔧 技术方法**

使用技术包括：扩散式 TTS（Grad‑TTS 作为 backbone）、Classifier‑Free Guidance、情感双向偏好优化（Emo‑BPO）两分支结构、晚期指导调度、HiFi‑GAN 声码器。

**📊 数据集**

实验数据集为 ESD（English subset）与 EmoVoiceDB，涵盖约40小时多说话人情感语音，覆盖 Neutral、Angry、Happy、Sad、Surprise 等情绪。

**📈 对比分析**

与 EmoSpeech、CosyVoice、CosyVoice2、EmoVoice、Emosphere++ 等基线在客观指标（Emo SIM、Prosody SIM、WER、UTMOS）及主观评测（MOS、Emo MOS、MOS EC、情感识别准确率）上进行对比。Emo‑BPO 在情感相似度、语调相似度、UTMOS 以及情感识别准确率上均优于所有基线，并在 WER 与 intelligibility 上保持竞争力。

**⚠️ 局限性**

局限性在于：①依赖同文本不同情感的匹配对数据量和多样性的要求较高，稀缺情感对会影响效果；②双分支结构与逐步调度略增推理复杂度；③在跨语言或跨说话人泛化、情感混合场景中的性能尚未充分验证。

---

## 201. MathVis: Can AI Reason through Drawing? A Benchmark for Mathematical Figure Generation

**arXiv ID:** 2610.12954 | [PDF](https://arxiv.org/pdf/2610.12954v1)

**作者:** Jingjing Wei `[一作]` (Zhihu), Shanshan Xiao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MathVis基准，用于评估大型语言模型在生成可执行TikZ/SVG数学图形方面的能力。

**💡 创新点**

创新点在于构建了包含512个高级数学题目、金属IR和原子化指标的评估体系，可区分代码错误与几何解释错误。

**🔧 技术方法**

采用了跨模态空间对齐、全局隐式几何推理、高精度结构化代码生成和语义级图验证等四大核心技术。

**📊 数据集**

使用了来自27个数学分支的512个任务，涵盖高级微分几何、复分析、拓扑等。

**📈 对比分析**

对16个LLM配置进行统一提示、渲染器和三轮生成评估，Pass@1（D-PSR）最高仅30.73%，表现仍远低于期望。

**⚠️ 局限性**

局限在于样本覆盖不均、缺乏训练数据、评估仅限单轮/工具辅助生成、以及对成本和延迟未进行量化。

---

## 202. Subset Sum and Integer Programming Feasibility Test in Near-Optimal Time

**arXiv ID:** 2610.12953 | [PDF](https://arxiv.org/pdf/2610.12953v1)

**作者:** Lin Chen `[一作]` (Zhejiang University), Guochuan Zhang `[通讯]` (Zhejiang University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出了一种确定性算法，能够在近线性时间 O(N^{1+ε})（ε>0 任意小）解决一维多重集子集和问题，并将该结果推广到固定维度 d 的多维子集和与整数规划可行性检验，时间复杂度为 O(N^d+ε)（忽略多项式因子）。

**💡 创新点**

创新点主要在于：
- 将稀疏的整数集合“提升”到更高维度的向量空间，使其在投影后呈现满格结构；
- 结合结构化的通用算术进程（GAP）与多维 zonotope 的内部充满性质，构造可填充的格点核心；
- 设计递归边界压缩和内部提升操作，利用颜色编码与绝对预算的子集和计算，保证每个递归层的搜索空间指数下降；
- 通过对多维实例的逐层维度升高和剩余量控制，证明递归层数受常数上界，从而实现总时间的近线性上界。

**🔧 技术方法**

核心技术包括：
- 加法组合学结果（长和集、Freiman 定理、Dense‑Box Filling）用于构造有限秩的 GAP；
- 高维几何工具（zonohedron、Macbeath 区域、凸体离散化、格点基变换）用于内部充满与边界收缩；
- 颜色编码（signed color coding）与绝对预算子集和算法；
- 逐步的向量编码与符号翻转、目标移位、以及可恢复的原始整数映射；
- 递归框架与递归树大小分析，控制每层的数组空间和运算量。

**📊 数据集**

本文属于理论算法研究，无实验数据集；所有结果均为理论复杂度分析与证明。

**📈 对比分析**

与已有方法（如 Bellman 动态规划 O(n·t)、Pisinger O(n·N)、Bringmann O(n+t)、Polak–Rohwedder–Węgrzycki O(n+N^{5/3}) 等）相比，本文实现了更快的伪多项式时间：对一维情况从 O(n+N^{5/3}) 降至 O(n+N^{1+ε})（ε 可取任意小），对多维情况从 O(n^d+N^d) 降至 O(n^d+N^{d+ε})；虽然仍为伪多项式，但相对之前的高指数或随机化方案显著提升。

**⚠️ 局限性**

限制与局限：
- 仍属于伪多项式复杂度，无法得到真正线性的 O(n+N) 或 O(N) 方案；
- 需要固定维度 d（多维结果的时间中 d 的隐式常数较大）；
- 递归层数与 ε 相关，常数和实现复杂度可能很高；
- 仅适用于整数集合或有限维整数向量，未覆盖连续或更一般的约束；
- 对输入规模大且 N 较大的情况下，实际性能仍取决于实现细节与常数因子。

---

## 203. WaveMark: Robust Wavelet-Domain Image Watermarking for Diffusion Models

**arXiv ID:** 2610.12952 | [PDF](https://arxiv.org/pdf/2610.12952v1)

**作者:** Jie Cao `[一作]`, Jianbing Ni `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 WaveMark 水印方法，将信息嵌入扩散模型初始噪声的 Wavelet 域，支持在生成过程中嵌入可检索水印。

**💡 创新点**

创新点在于通过键控的 Wavelet 子带分布多副本、投影幅度嵌入，并保留残差能量以实现明确解码阈值，从而显著提升对高级去水印攻击的鲁棒性。

**🔧 技术方法**

采用二维 Haar 小波变换、键控伪随机函数、投影幅度嵌入、DDIM 逆向、Stable Diffusion 生成模型等技术。

**📊 数据集**

在 Stable Diffusion 2.1 与 3.5 的 SDP 与 MS‑COCO Caption 两个数据集上进行实验。

**📈 对比分析**

与六种基线（SSig、SleeperMark、VINE、GS、PRC、T2SMark）以及 14 种攻击（噪声、模糊、裁剪、对比度、JPEG、再生攻击和 MarkNull 等）比较，WaveMark 在无攻击时 100% BA，MarkNull 及 MarkNull‑A 仍保持 88% 以上 BA，整体在鲁棒性和图像质量上均优于基线。

**⚠️ 局限性**

局限性包括对重建攻击（CtrlRegen+、NFPA）鲁棒性不足，且仍需开发公开可验证的水印方案以及进一步降低对生成质量的潜在影响。

---

## 204. AgentValve: Sybil-Resilient Payment Authorization for Permissionless AI Markets

**arXiv ID:** 2610.12948 | [PDF](https://arxiv.org/pdf/2610.12948v1)

**作者:** Jianting Zhang `[一作]` (Purdue University), Aniket Kate `[通讯]` (Mysten Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于信任网络的Sybil抗拒付费授权协议，实现支付授权时的容量预留与反馈机制，保证对抗Sybil攻击的曝光上限。

**💡 创新点**

创新点在于将个性化流量原则与边界容量剪切相结合，实现切割受限曝光和容量消耗放大系数的上界，并通过见证提交减少链上验证工作。

**🔧 技术方法**

利用有向加权图的容量流模型、原子路由预留、交互背书的容量创建、路径长度限制以及见证-提交技术。

**📊 数据集**

实验基于2026年1月至8月的以太坊交易数据构建的1,055节点信任图，以及六组不同规模的合成有向图（128-8,192节点）。

**📈 对比分析**

与全局计数/全局量化等无约束策略以及两种基线（路径聚合存储和见证存储）对比，见证-提交在路由验证和存储成本上分别降低约20-40%，支付授权成本约0.04–0.21美元。

**⚠️ 局限性**

局限在于对累计损失缺乏精确上界、可能出现预留耗竭攻击、以及在稀疏网络中路由成功率仍受限，需进一步优化路径长度与反馈时限等参数。

---

## 205. In-Context Learning Can Help Vision-Language Models Overcome Training Priors

**arXiv ID:** 2610.12942 | [PDF](https://arxiv.org/pdf/2610.12942v1)

**作者:** Kun Wang `[一作]` (Princeton University), Esin Tureci `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉‑语言模型在训练先验冲突时的视觉定向失败，构建控制基准并利用视觉上下文学习（ICL）调节模型关注点。

**💡 创新点**

证明视觉ICL能显著抑制先验主导导致的错误，且通过转移注意力从先验线索到视觉线索实现；同时揭示ICL仅在已有可靠视觉能力时才有效。

**🔧 技术方法**

使用视觉ICL、logit‑lens 与 LatentLens 对图像词表示进行语义标注，进行注意力重分配与遮蔽实验；对模型进行多尺度和多族群评估。

**📊 数据集**

五类闭标签视觉任务（计数、颜色识别、形状识别、物体方向、空间关系），每类包含真实与对抗（先验冲突）图像，全部在公开或自构造数据集上完成。

**📈 对比分析**

与零样本基线比较，在对抗样本上提升约27–30%准确率，整体提升约10–30%（视模型规模与族群而定）；跨五大VLM族群均验证一致性。

**⚠️ 局限性**

ICL对弱或不存在的视觉能力无显著提升，且需大量上下文，导致显著的内存与推理延迟开销。

---

## 206. Network Topology Shapes AI Intervention Effectiveness on Deception: Truth-Telling vs. Investigative Agents

**arXiv ID:** 2610.12931 | [PDF](https://arxiv.org/pdf/2610.12931v1)

**作者:** Zhao Song `[一作]` (Teesside University), The Anh Han `[通讯]` (Teesside University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在网络化人口中引入固定的AI“真相传播者”和“调查者”两种干预角色，研究其对欺骗行为演化的影响。

**💡 创新点**

首次揭示了干预类型与网络拓扑、干预密度三者共同决定欺骗抑制或放大的非单调机制，并区分了被动真相传播与主动调查的根本差异。

**🔧 技术方法**

采用演化博弈框架（Deceiver's Game X）、基于图的代理仿真、Fermi 模拟复制、选择梯度可视化和网络结构（正方形格、无尺度、完全连通）等技术。

**📊 数据集**

使用人工构建的无尺度、格点与完全图网络作为实验平台，并通过随机种子生成的初始策略分布进行多次仿真。

**📈 对比分析**

通过比较不同干预密度与网络结构下的稳态欺骗/真相/调查比例，发现真相传播者在中等密度时反而促进欺骗，而调查者在中等密度时最大化真相比例；实验结果与无干预基线相比，展示了显著的非线性提升或恶化效果。

**⚠️ 局限性**

局限性包括：仅考虑随机放置的固定“狂热分子”、忽略信任、信誉和不完美验证的现实因素；使用的模拟参数和网络模型可能与真实社交系统不完全对应；未进行实验或人机交互验证。

---

## 207. SCCGBench: Executability-Oriented Evaluation for Real-World Web Service-Call Code Generation

**arXiv ID:** 2610.12920 | [PDF](https://arxiv.org/pdf/2610.12920v1)

**作者:** Shuo Wang `[一作]` (Shandong University of Technology), Chun Wang `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了可执行度量为核心的 Web 服务调用代码生成基准 SCCGBench，并通过去污染和红色意图生成实现真实代码可执行性评估。

**💡 创新点**

创新点包括（1）设计 Mock/Robust 等可执行度量与红色意图生成以消除数据泄漏；（2）系统揭示 LLM 生成失败主要源于 API 端点定位不准确，而非代码合成能力不足。

**🔧 技术方法**

使用了 AST 解析、语义对齐、虚拟执行环境、稀疏（BM25）和稠密（MiniLM/BGE）检索、Oracle 文档提示、统计指标（Cohen's κ、pass@k）等技术。

**📊 数据集**

构建数据集包含 3,135 条来自 2,082 个真实 GitHub 仓库的 RapidAPI 调用函数，覆盖 850 个 API。

**📈 对比分析**

实验通过意图仅输入、检索增强（RAG）和 Oracle 文档三种设置对比六个 LLM 与检索基线；在意图仅输入下 Mock 仅 0.4–3.8%，检索增强约 20%，Oracle 文档下可达 89–90%，检索 hit 最高仅 54%。

**⚠️ 局限性**

主要局限在于只覆盖 Python 与 RapidAPI 生态，评估基于虚拟执行而非真实网络请求，模型对检索的依赖仍强，且数据规模与语言多样性有限。

---

## 208. AI-Mediation in Retail Markets: Revenue Effects and Assortment Optimization

**arXiv ID:** 2610.12917 | [PDF](https://arxiv.org/pdf/2610.12917v1)

**作者:** Omar El Housni `[一作]` (Cornell University), Marouane Ibn Brahim `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在人工智能（AI）中介层面下的零售商品组合规划，探讨了AI推荐如何影响零售商的最优商品集合与收入。

**💡 创新点**

创新点在于：①首次将AI生成的排序视为中介层，构建了包含确定性与随机（Plackett–Luce）两种推荐规则的决策模型；②通过控制实验评估AI排序模型，并证明仅使用AI排序的确定性近似可在实践中几乎完全恢复随机模型的收益；③对确定性中介下的组合优化问题给出了复杂度分析并提出了对任意固定客户细分数的完全多项式时间近似算法（FPTAS）。

**🔧 技术方法**

技术手段包括：Plackett–Luce概率模型的最大似然估计（带岭正则化）、多项式逻辑模型（MNL）用于客户选择、基于排序的推荐规则、阈值变换与多维背包问题的多项式时间求解、以及利用已有的多比率函数近似框架实现FPTAS。

**📊 数据集**

使用的数据集为Amazon 2023产品评论数据，构造了三类产品（Lightning 充电器、机器人吸尘器、宠物牵引绳）共72件产品，并通过OpenAI、Anthropic Claude、Google Gemini三大AI模型在三种顾客画像（预算、价值、奢侈）下进行随机排序查询，共生成32,400条受控排名数据。

**📈 对比分析**

比较方法包括：①在随机Plackett–Luce中介下直接最优、确定性近似最优、直接MNL（忽略中介）三种策略；②在确定性中介下检验FPTAS与精确枚举的收益差距；实验显示：忽略中介平均导致2.52%–11.37%收入损失；仅使用AI排序的确定性近似平均损失不超过0.53%；在大多数配置下，FPTAS实现的收益与全局最优相差不到1%。

**⚠️ 局限性**

局限性包括：①实验采用合成的客户偏好，未在真实市场中验证；②仅考虑单一AI模型的推荐排名，未研究多模型协同或动态适应；③对随机中介的最优解仅用确定性近似求解，理论上未给出严格的最优性保证；④数据集规模有限，未来需扩展至更大多样化商品集合。

---

## 209. Sample-Preserving Search-to-Decision Reduction for Noisy Linear Equations over Large Moduli

**arXiv ID:** 2610.12913 | [PDF](https://arxiv.org/pdf/2610.12913v1)

**作者:** Andrej Bogdanov `[一作]` (University of Ottawa), Prashant Nalini Vasudevan `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种新的参数保持、样本保持的搜索到决策归约，用于噪声线性方程问题，包括LWE和LWR，复杂度为 poly(n, m, log q, q/σ)。

**💡 创新点**

创新点在于归约保持所有参数，复杂度不随模数 q 线性增长，能够在 q 指数大时仍保持多项式时间，并适用于任何对 ℓ∞ 误差鲁棒的噪声函数。

**🔧 技术方法**

采用阈值判别器、余弦/傅里叶分析、混合（hybrid）论证以及 Akavia‑Goldwasser‑Safra 的重要 Fourier 系数学习算法。

**📊 数据集**

论文为理论工作，未使用实验数据集。

**📈 对比分析**

与已有归约相比，本工作在保持样本、秘密维数和误差幅度的同时，将时间复杂度从线性或多项式依赖 q 或 σ 降低到 poly(log q, q/σ)，但未给出实验验证。

**⚠️ 局限性**

限制在于仅对素模 q 或当秘密满足 coprime 条件时可完整恢复；对于一般合成模数只能输出包含可乘常数倍秘密的列表；归约仍需决策算法具备足够优势。

---

## 210. Sampling-based Kinodynamic Planning with World Models

**arXiv ID:** 2610.12910 | [PDF](https://arxiv.org/pdf/2610.12910v1)

**作者:** Yitian Gao `[一作]` (Purdue University), Zachary Kingston `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种基于视觉世界模型的kinodynamic采样树规划框架，利用RRT和KPIECE在潜在空间进行高效搜索，并在每次迭代中用CEM微调得到连续控制序列。

**💡 创新点**

创新点包括：①把高维视觉模型的潜在空间视为真实动力学的代理，直接在此空间做树搜索；②将经典SBMP的邻域引导与投影覆盖探索策略迁移到潜在空间；③采用GPU批量传播和短CEM优化，实现极低的规划时延。

**🔧 技术方法**

使用的技术：DINO-WM视觉世界模型（冻结的DINOv2编码器+动作条件预测器）、近邻引导RRT、投影基KPIECE、GPU并行批量传播、CEM微调、递归滚动MPC框架。

**📊 数据集**

评估数据集：四个模拟域（point_maze、point_wall、tblock_push、rope）以及在真实YAM机械臂上的50个块推任务（含10个不同难度分层），全部使用DINOv2提取的视觉特征。

**📈 对比分析**

与CEM、MPPI、MCTS等基线对比：在模拟中成功率相当或更高，规划时间可低至1/13.8，节点评估次数减少约10倍；在真实机器人上成功率提升约20%（29/30对比23/30），总规划时间也显著降低。

**⚠️ 局限性**

限制：搜索依赖潜在距离作为目标度量，易被误导；冻结的世界模型推理成本占主导，降低分辨率或蒸馏模型可进一步加速；未考虑模型不确定性和离线热启动，导致在极长序列上仍可能出现误差堆积。

---

## 211. CacheFlex: Direct Software-Managed Access to Higher-Level Cache for Scalable Vector Support

**arXiv ID:** 2610.12940 | [PDF](https://arxiv.org/pdf/2610.12940v1)

**作者:** Jingqun Zhang `[一作]` (University of Washington), Ang Li `[通讯]` (University of Washington)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出CacheFlex，通过ISA与微架构共设计，使CPU私有L2缓存可在运行时动态转换为软件可管理的scratchpad（SPM），从而实现直接访问并保证线程私有性。

**💡 创新点**

创新点在于：①在保持高速预取、乱序执行和OS上下文切换的同时，允许软件显式控制L2缓存的部分方式成为SPM；②采用双LSQ跨域桥接机制，确保跨域内存一致性；③实现仅2.02%面积开销的方案。

**🔧 技术方法**

技术手段包括：ISA扩展指令（SPM管理、访问与桥接）、SPM专用加载/存储队列、双LSQ协同、L2标签状态扩展、跨域内存一致性控制、Armv8‑A SVE实现、gem5仿真与12 nm FinFET RTL面积功耗测量。

**📊 数据集**

实验使用FP16 GEMM和Attention核，基准数据集来自LLaMA‑3.2‑1B、BERT‑Base模型的预填充层，并采用FlashAttention融合实现。

**📈 对比分析**

通过与基线（硬件预取开启）、软件/硬件预取、L2预取提示以及理想的L2锁定+非临时加载等替代方案对比。结果显示：在VL=4时平均1.12×加速、能耗下降26%；在VL=16时平均1.44×加速、能耗下降40%；全层预填充阶段最高可达1.78×加速、45%能耗节约。

**⚠️ 局限性**

局限性在于仅单核评估，未探讨多核共享LLC冲突与同步开销；依赖软件显式管理，编程复杂；仅适用于私有L2可用的场景；对OS调度和上下文切换的依赖可能导致实际部署中的性能波动。

---

## 212. Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation

**arXiv ID:** 2610.12898 | [PDF](https://arxiv.org/pdf/2610.12898v1)

**作者:** Jinjin He `[一作]` (Georgia Institute of Technology), Jiayun Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了预测状态流（Predictive‑State Flow）框架，利用流式先验生成预测状态，随后并行块级解码器在每块内部自回归完成文本生成；

**💡 创新点**

创新点在于把生成任务拆分为先验负责高层预测信息，解码器负责局部细节，同时引入单词“着陆”目标以实现块间轻量级一致性；

**🔧 技术方法**

采用了Rectified Flow先验、块因果编码器、块级并行解码器、着陆目标、Self‑Conditioning和Guidance Scale等技术；

**📊 数据集**

使用OpenWebText数据集，采用T5词表（32,100个token）进行训练与评估；

**📈 对比分析**

与ELF‑B、BD3‑LM、MDLM、GPT‑2小等基线比较，在匹配熵5.16时，K=8状态仅配置实现了22.8的困惑度，MAUVE 0.93，FLOPs仅为ELF‑B的23倍；

**⚠️ 局限性**

局限性包括模型规模受限（最大677M参数）、固定1024-token长度、无条件生成、块间一致性未在更长文本中验证、缺乏强大的提示条件和对长距离连贯性评估不足。

---

## 213. Vulnerable Models in the AI Supply Chain: An Empirical Study of Adoption, Propagation, and Usage Patterns

**arXiv ID:** 2610.12893 | [PDF](https://arxiv.org/pdf/2610.12893v1)

**作者:** The Anh Nguyen `[一作]` (Adelaide University), M. Ali Babar `[通讯]` (Adelaide University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对16,977个在HuggingFace上被InsightDB标记为安全漏洞的模型进行大规模经验研究，追踪其在GitHub代码、软件包、微调链和HF Spaces中的使用与传播情况。

**💡 创新点**

创新点在于：①将模型级别扫描与实际代码集成使用相结合，填补了安全警告与下游实现之间的空白；②构建基于Code Property Graph（CPG）的自定义归一化+频繁子图挖掘（Top‑K Graph）框架，自动识别并归类四类常见的漏洞模型使用模式。

**🔧 技术方法**

采用的技术主要包括：代码静态分析框架Joern生成CPG；自定义前向后向数据流切片、标签归一化和图简化；Top‑K Graph频繁子图挖掘算法；以及NetworkX进行子图同构筛选与模式归纳。

**📊 数据集**

使用的数据集来源于ProtectAI InsightsDB的安全扫描结果（16,977个模型），并通过GitHub Code Search、HuggingFace Hub API、Spaces API等接口获取对应的代码文件、包信息、微调链与部署实例。

**📈 对比分析**

与传统手工编写CPG查询规则相比，本文的频繁子图挖掘在20小时内完成对1,156份Python文件（共约2,793节点、20,134条边）的分析，提取出809个模式，最终归纳为4类高频图案，展示了该方法在发现未知漏洞使用模式上的可行性与高效性。

**⚠️ 局限性**

局限性包括：仅检测静态代码中的模型引用，未覆盖运行时动态构造或隐式加载；仅聚焦Python语言与HuggingFace生态，可能无法泛化到其他框架或私有仓库；扫描工具InsightDB的漏洞标签不涵盖所有安全威胁；以及未对实际运行时安全性进行动态验证。

---

## 214. Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction

**arXiv ID:** 2610.12984 | [PDF](https://arxiv.org/pdf/2610.12984v1)

**作者:** Yuqi Huang `[一作]` (Shanghai Jiao Tong University), Ning Liao `[通讯]` (Shanghai Qiji Zhifeng Co., Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Work2Skill 框架，收集并对齐人类 GUI 演示的多模态证据，并通过主动证据检索与迭代重写生成可复用的代理技能。

**💡 创新点**

创新点在于：① WorkRecorder 能同步捕获屏幕视频、键鼠事件和可访问性树信息，形成对齐的多模态证据；② WorkDemo‑350 为桌面 GUI 任务提供首个同步多模态演示数据集；③ 采用 Agentic Skill Induction 方法，实现按需检索证据、技能撰写与重写，显著提升技能可复用性。

**🔧 技术方法**

使用多模态同步捕获技术、可访问性树（AX）信息、LLM（GPT‑5.5、Opus‑4.8、Qwen3.5‑35B‑A3B、Qwen3.5‑9B）进行技能生成，结合定制的 prompt 与工具接口进行证据检索、技能写作与验证。

**📊 数据集**

WorkDemo‑350（350 个演示，覆盖 64 个 MyPCBench Bounded Action 任务和 286 个 OSWorld‑Verified 子任务）作为实验数据集。

**📈 对比分析**

与无技能（No Skill）、ShowUI‑Aloha、Demo2Tutorial 等基线在四个模型上对比，Work2Skill 在 MyPCBench 和 OSWorld‑Verified 子集上均获得最高得分和成功率。平均得分提升 4.67–10.54 点，成功率提升 6.25–25.00 百分点，在变更任务和模型迁移实验中仍保持优势。

**⚠️ 局限性**

局限性包括：① 需要系统支持可访问性树，某些应用缺失该信息；② 不同模型之间仍存在显著性能差距；③ 依赖大量 prompt 工程和手工筛选演示；④ 对高度动态或非可访问性友好的界面可能效果有限。

---

## 215. Beyond Semantic IDs: Finite-Beam Reachability and Continuous-Query Generation in Generative Recommendation

**arXiv ID:** 2610.12977 | [PDF](https://arxiv.org/pdf/2610.12977v1)

**作者:** Yu Sun `[一作]`, Hailu Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并比较了基于Semantic ID的生成式推荐与两种连续查询方法（CQG‑Single与CQG‑AR），探讨有限束搜索对目标召回与覆盖率的影响。

**💡 创新点**

提出无词表、连续查询的推荐接口，分离自回归结构与离散代码束限制，并系统诊断不同流行度与深度的交叉性能。

**🔧 技术方法**

采用Transformer自回归生成、密集/近似最近邻检索、全词表交叉熵、采样交叉熵与InfoNCE等训练目标。

**📊 数据集**

在MovieLens‑1M、Amazon Beauty与Amazon Sports三大公开数据集上进行实验。

**📈 对比分析**

与SASRec、TIGER、LETTER、EAGER等基线对比，CQG‑Single在多数实验中提升R@10约0.01并且覆盖率与目标支持率优于TIGER；CQG‑AR在美妆与运动数据集上实现约50%相对提升，但在ML‑1M上与TIGER持平。

**⚠️ 局限性**

仅在小规模数据集验证，未测试亿级规模的稀疏性、动态更新、召回深度、延迟与内存开销，且对有限束的性能提升不具普适性。

---

## 216. Minions: A Dynamic and Distributed Virtual GPU Cloud Built on Idle Apple Silicon Devices

**arXiv ID:** 2610.12970 | [PDF](https://arxiv.org/pdf/2610.12970v1)

**作者:** Jianhua Li `[一作]` (Apple), Sean Suchter `[通讯]` (Apple)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个分布式虚拟 GPU 云平台，利用闲置的 Apple Silicon GPU 进行机器学习推理任务，无需额外硬件即可即时执行。

**💡 创新点**

通过将个人 macOS 设备的 GPU 进行动态聚合、无中心化调度的拉取执行模型、基于 PostgreSQL 的原子子任务获取、以及可配置的执行器优先级绑定，实现了零成本、可弹性扩展的推理平台。

**🔧 技术方法**

核心技术包括 PostgreSQL 作为元数据存储、REST API 进行子任务调度、原子获取表实现互斥、Apple Silicon GPU 上的 HuggingFace Transformer 与 Ollama 框架、S3 对象存储、Grafana 监控以及 Watchdog 心跳监控。

**📊 数据集**

论文未提供专门的数据集，而是以通用的嵌入生成和大语言模型推理任务为例，使用 HuggingFace 公开模型和 Ollama 本地 LLM 进行评估。

**📈 对比分析**

通过对比部署前后的 GPU 队列等待时间、资源投入和扩展方式，结果显示队列时间从最多 10+ 小时降至分钟级；基础设施成本从高额降至零；扩展方式从手动人工提升到有机自发；未给出具体吞吐量指标，但推理延迟显著降低。

**⚠️ 局限性**

局限性包括：对设备偶发离线/休眠的容错机制依赖 Watchdog，隐私和数据安全需要额外隔离；目前仅支持推理，训练尚未实现；执行器的拉取模式在高负载下可能导致网络瓶颈；跨组织使用时需要额外治理与权限管理。

---

## 217. DataCupid: Stable Matching Repair for Training Data Acquisition with Delayed Feedback

**arXiv ID:** 2610.12960 | [PDF](https://arxiv.org/pdf/2610.12960v1)

**作者:** He Sun `[一作]` (University of Hong Kong), Shinan Liu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在匹配平台中，当收益矩阵因公开更新而失效时，如何利用已有有效观测和新的评估结果快速完成匹配重修（repair），并提出了 Residual‑Load Repair (RLR) 与其前缀监控变体 RLR‑P 两种采样与证书框架，并针对固定结构的优先权星（priority‑star）市场给出了信息负载常数最优的 Priority‑Search 算法。

**💡 创新点**

创新点主要包括：① RLR 通过双倍递增的 dyadic 采样计划实现残差负载（residual‑load）下的完美配色，并给出锐利的 1.607 以内的周期上界；② RLR‑P 在每一次完整前缀完成后立即进行证书检查，保证即使反馈延迟与奖励相关也能安全地终止；③ 对于优先权星结构的未知间隙问题，Priority‑Search 在不需要事先知道间隙大小的前提下，达到信息负载下的渐近最优采样率。

**🔧 技术方法**

技术手段包括：残差负载（endpoint load）分析、双倍递增的 dyadic 计数目标、精确匹配分解（edge‑coloring）、基于区间的置信证书、蒙特卡罗/自适应采样、KL 散度下的统计信息下界、以及纯确定性 oracle 的路径耦合。

**📊 数据集**

实验使用合成匹配市场数据，包括 900 个不同规模、需求族与种子组合的实例、固定优先权星结构的若干实例以及 1,680 条分支的完整验证集，均为人工生成而非真实数据集。

**📈 对比分析**

比较方法：在相同置信水平、时限与市场实例下，对 RLR、RLR‑P、Width‑P（宽度基准）和 PCOS‑A4（先前方法）进行并行评估，记录认证时间、准备时间、已发评估数、稳定性违约与 oracle CPU。结果显示：RLR‑P 在有限预算下的更新认证时间低于宽度基准，但伴随更高的 oracle 计算；RLR‑P 与 Width‑P 在总发评估数和误差指标上相近；Priority‑Search 在优先权星实例上几乎达到理论信息负载常数。

**⚠️ 局限性**

局限性：① RLR 可能对无关边产生过度采样，未必是统计最优；② RLR‑P 只针对单次重修并假设纯确定性 oracle，未说明多次更新或 CPU 时延优势；③ Priority‑Search 的最优性仅在固定优先权星、独立伯努利、子对数延迟等严格假设下成立；④ 所有实验均为合成数据，真实世界验证尚未完成；⑤ 对更一般匹配实例的全局最优性仍为开放问题。

---

## 218. UniDeblur: Unified Motion Deblurring via Fourier Convolution Learning

**arXiv ID:** 2610.12941 | [PDF](https://arxiv.org/pdf/2610.12941v1)

**作者:** Xintian Mao `[一作]` (East China Normal University), Yan Wang `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了统一的运动去模糊框架 UniDeblur，利用 Fourier Convolution Learning（FCL）实现基于频域的卷积自适应滤波，并构建多源数据集与质量控制流程。

**💡 创新点**

创新点在于：①将运动模糊视作空间卷积，在频域用逐点相乘的方式学习输入自适应的滤波器；②在 U‑Net 的跨尺度跳连中嵌入多尺度 FCL，抑制噪声并提升恢复细节；③通过严格的数据清洗和多源训练提高跨域泛化能力。

**🔧 技术方法**

采用深度 CNN（如 NAFBlock、BioBlock、EVSBlock）搭建 U‑Net，加入 FCL 模块，使用 RFFT2D/IFFT2D；损失为 L1 与频域重建损失；训练使用 Adam、余弦学习率调度。

**📊 数据集**

使用六大高质量运动模糊数据集（GoPro、HIDE、REDS、RSBlur、RealBlur、LOLBlur）进行训练；在六个 OOD 数据集（BSD1/2/3、MIORe/VAR-MIORe、FoundIR）进行测试，并对数据做裁剪和剔除错误样本。

**📈 对比分析**

与 MIMO‑UNet、DeepRFT、NAFNet、LoFormer、EVSSM、PromptIR、MoCEIR、BioIR 等现有端到端与全场景恢复模型对比，UniDeblur‑L 在所有 in‑domain 数据集上平均提升 0.7‑1.2 dB PSNR，OOD 数据集上 CLIP‑IQA/MUSIQ 提升 1–2 分；计算量仅增加 5–10% FLOPs，显著提升跨域性能。

**⚠️ 局限性**

局限性：仍需更大规模真实世界模糊数据验证；在极端低光或极大运动模糊时，FCL 对噪声鲁棒性有限；模型对多任务统一恢复（如去噪、去雨）仍不如专门模型。

---

## 219. Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding

**arXiv ID:** 2610.12939 | [PDF](https://arxiv.org/pdf/2610.12939v1)

**作者:** Raghavv Goel `[一作]` (Qualcomm AI Research), Mingu Lee `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型（LLM）中解码器隐藏层的多层选择，提出了一套基于矩阵熵和CKA的理论框架，并实现了两种实用算法（ELSI和ELSI-C）来自动挑选最具信息量且冗余度低的层集合，以提升推理时的speculative decoding效果。

**💡 创新点**

创新点在于：①首次将矩阵熵与CKA结合，用于量化隐藏层集合的整体信息量和几何多样性；②提出token-entropy过滤降低不同prompt间的熵波动；③设计基于熵的beam搜索（ELSI）和熵+CKA驱动的多层扩展（ELSI‑C）两种选择算法，形成可解释且高效的多层隐藏状态选择方法。

**🔧 技术方法**

核心技术包括：矩阵熵（Rényi-α）用于衡量拼接后隐藏状态的整体信息量；CKA与其transition分数用于识别层之间的代表性转折点；token-entropy过滤、block averaging、beam搜索以及基于熵的初始与CKA筛选的多层扩展算法。

**📊 数据集**

使用的数据集：训练阶段使用ShareGPT和Ultrachat生成的蒸馏数据；评估阶段使用SpecBench多任务基准；针对模型为Qwen3-4B-Instruct-2507和Llama-3.1-8B-Instruct的隐藏层进行分析与实验。

**📈 对比分析**

比较方法：与传统固定位置启发式（EAGLE2的[N-1]和EAGLE3的[2,N//2,N-3]）以及低熵对照集进行对比；评估指标为SpecBench中各任务的平均接受token数（τ）。实验显示：ELSI（m=3）平均提升1.6–2.2%（≈0.085τ）相较于EAGLE3，且比EAGLE2高出6–7%；ELSI（m=1）在多数场景下能与EAGLE3相媲美或超越，尤其在翻译、摘要和STEM等任务中表现更佳。

**⚠️ 局限性**

局限性：①需要先行收集并过滤隐藏状态，增加预处理成本；②算法仍基于深度模型的层数和冗余特性，可能对小模型或不同架构的迁移效果有限；③在GPU上实际推理加速未得到充分验证，边缘设备的效率尚待进一步实验；④仅在Qwen3和Llama两款模型上验证，泛化性尚需进一步探索。

---

## 220. Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision

**arXiv ID:** 2610.12936 | [PDF](https://arxiv.org/pdf/2610.12936v1)

**作者:** Zile Zhong `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过光流监督的未来运动预测增强轻量级视听动作（VLA）模型，实现对机器人控制的更好预测与执行。

**💡 创新点**

创新点包括：使用光流作为预测目标仅学习运动嵌入；在动作优化中分离梯度路径与正则化，保持预测表示；以及采用FiLM融合当前与预测运动信息。

**🔧 技术方法**

技术实现基于预训练的InternVL3视觉‑语言模型、DINOv2/CLIP特征编码、RAFT光流估计、Sketched Isotropic Gaussian Regularization（SIGReg）和流匹配Transformer。

**📊 数据集**

使用的基准数据集为Meta‑World MT50、RoboTwin模拟环境以及真实 AgileX 双臂机器人上的花朵摆放、杯子挂放和微波炉放置任务。

**📈 对比分析**

与多种基准（π_0、SmolVLA、Evo‑1 等）比较，Meta‑World MT50 最高 91.1% tier 平均成功率，提升 13%；RoboTwin 41.8%；真实机器人平均 63.3% 成功率，GPU 内存仅 2.54 GB，推理频率 6.16 Hz。

**⚠️ 局限性**

局限性包括仅基于 2D 光流预测，缺乏 3D 几何信息；在极端视觉变化下预测可能不稳定；模型依赖预训练视觉‑语言模型的通用性。

---

## 221. What Egocentric Video Embeddings Encode and How to Combine Them with Inertial Sensors for Human Activity Recognition

**arXiv ID:** 2610.12926 | [PDF](https://arxiv.org/pdf/2610.12926v1)

**作者:** Kansuke Habano `[一作]` `[通讯]` (Independent Researcher), Kansuke Habano (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了头戴式摄像头视频嵌入的语义内容，并设计了一种将视频嵌入与单个手臂/腿加速度计结合的图传播方法来实现无序一秒级片段的运动识别

**💡 创新点**

创新点在于：①揭示预训练模型在第一人称视频中嵌入的主成分主要反映相机朝向而非运动；②将视频用于构建图结构识别同一运动集合，利用惯性预测在图上传播标签，实现四传感器融合的无标签推理；③通过剔除主成分或白化提升图质量；④在无序片段评估协议下验证方法并提出对协议的改进建议

**🔧 技术方法**

技术包括：预训练视觉模型（VideoMAEv2、CLIP、DINOv2、VC‑1）提取嵌入；光流/几何估计相机姿态；基于k‑NN的相似图构建；多阶段图传播与自适应重估；加权概率聚合与决策阈值校正；特征分解与子空间移除；基于场景分割与相机倾斜的解释分析

**📊 数据集**

使用WEAR运动数据集（22名受试者、18种户外训练动作，四个腕/脚加速度计 + 头戴视频）及Ego‑Exo4D用于嵌入分析

**📈 对比分析**

与单一惯性分类器（macro‑F1=0.648）以及多种融合策略比较，图传播后macro‑F1提升至0.852（单模型），在交叉验证中获得显著提升；在公开测试集上实现0.871 macro‑F1；对比多种融合方案均表现不如划分角色的方法

**⚠️ 局限性**

局限性在于：仅针对WEAR数据，方法依赖于受试者片段齐全且运动集合视觉一致；对自由生活场景的泛化未知；主成分剔除与白化的提升机制尚未完全区分；对协议的特定属性（顺序可恢复性、类计数先验）敏感

---

## 222. Adaptive Whole-Body MPC with Optimal Posture Planning for Humanoid Robots under Unknown Payloads

**arXiv ID:** 2610.12972 | [PDF](https://arxiv.org/pdf/2610.12972v1)

**作者:** Manh Cuong Nguyen `[一作]` (Vingroup), Minh Nhat Vu `[通讯]` (VinUniversity)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种面向未知负载的全身适应性控制框架，结合了重心MPC、适应性模糊平衡控制和基于三次积分的姿态规划，以实现稳定的下蹲、抓取、举升、搬运和放置。

**💡 创新点**

创新点在于：1）将适应性模糊控制器与重心MPC耦合，实现对负载导致的模型不匹配进行在线补偿；2）基于三次积分模型的 waypoint 优化姿态规划，确保站立-坐姿过渡的动态一致性；3）通过最小二乘法将模糊产生的补偿力分配到支撑腿关节，保持手臂稳定不受影响。

**🔧 技术方法**

技术包括：重心模型预测控制（Centroidal MPC）、全身控制（Whole‑Body Control）、适应性模糊逻辑控制（Adaptive Fuzzy Control）、三次积分 waypoint 轨迹规划、MuJoCo 仿真以及实际28自由度人形机器人硬件实现。

**📊 数据集**

未使用公开数据集；实验数据来自 MuJoCo 物理引擎仿真以及真实机器人在 7 kg、10 kg 负载下的抓取、举升与步行实验。

**📈 对比分析**

与传统非适应性重心MPC基准进行对比，实验表明：在姿态转换和举升过程中，所提框架的平衡稳健性更强，跟踪误差更小；机器人能够在保持重心稳定的前提下完成更平滑、更安全的姿态过渡与搬运。

**⚠️ 局限性**

局限性：模糊控制规则为经验式，缺乏学习与自适应优化；仅对支撑腿施加补偿，无法充分利用上半身；在极端动态或高负载情况下，可能仍需进一步的预测性干扰观测或学习驱动的补偿策略。

---

## 223. Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training

**arXiv ID:** 2610.12933 | [PDF](https://arxiv.org/pdf/2610.12933v1)

**作者:** Suorong Yang `[一作]` (National University Of Singapore), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于不确定性的渐进压缩框架，利用教师生成的推理轨迹进行稀疏监督与训练分配，以在极低训练成本下实现高质量推理。

**💡 创新点**

创新点在于将过程级不确定性建模为 Monte Carlo 推理树（MCRT），先在教师端根据不确定性筛选高价值轨迹，再在学生端根据学习状态动态路由优化，从而在保持甚至提升性能的同时大幅压缩数据与计算。

**🔧 技术方法**

核心技术包括 Monte Carlo 推理树、过程级不确定性估计、基于不确定性的教师监督压缩、学习者感知的在线路由以及 LoRA‑式的 SFT 训练。

**📊 数据集**

实验使用 DAPO‑17K 作为主训练集，并在 Nemotron‑Math 以及多项数学与科学推理基准（AIME、AMC、HMMT、Math‑500、GSM8K、GPQA‑D）进行评估。

**📈 对比分析**

与完整数据 SFT、随机抽样以及传统蒸馏方法比较，压缩至 0.22–1.91% 的优化预算即可在 800× 以上的壁时加速，同时在大多数基准上保持或超过全数据蒸馏，并与 30–560B 规模模型竞争。

**⚠️ 局限性**

局限性包括需要大量教师推理轨迹来准确估计不确定性，对教师生成的 CoT 质量依赖较高，且对动态或多模态推理场景的适用性尚未验证。

---

## 224. DeFiRank: Federated Pairwise Ranking for Governance in Decentralized Financial Networks

**arXiv ID:** 2610.12916 | [PDF](https://arxiv.org/pdf/2610.12916v1)

**作者:** Zahra Batool `[一作]` (École de technologie supérieure), Ulrich Aivodji `[通讯]` (École de technologie supérieure)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个联邦学习的成对排名框架，用于在去中心化金融网络中自动化治理决策，替代传统人工评审。

**💡 创新点**

创新点在于：①首次将成对排名与联邦学习结合，既能捕获网络全局行为模式，又能保障机构交易数据隐私；②提出了动态治理决策管线，可根据网络健康和规模自适应调整升降级阈值；③在去中心化环境下评估并验证了LSTM‑RankNet在排名精度与稳定性方面的优越性。

**🔧 技术方法**

采用的技术包括：联邦学习（FedAvg）、LSTM编码器+RankNet成对损失、注意力编码器、NDCG与Kendall τ评估指标、Python/Flower框架实现模型训练与聚合。

**📊 数据集**

使用真实区块链数据集（Canton Network）共965,894条供应商日记录，184天、102,520名供应商，选取13个治理相关特征及其7天趋势特征进行排名。

**📈 对比分析**

与集中式基线对比：联邦模型在NDCG@15上略低（≈0.87 vs 0.92），但Kendall τ保持相同（≈0.70）。在治理决策上，联邦方案识别到648个升迁候选和39个降级候选，升迁一致性100%，降级一致性72%。

**⚠️ 局限性**

局限性包括：联邦训练相比集中式在NDCG上有轻微下降；注意力架构收敛慢且不稳定；未加入差分隐私或正式安全保证；对非IID客户端的个性化策略尚未充分探究；缺乏与链上激励机制的直接耦合。

---

## 225. BLEND: Balancing Personalization vs. Generalization in Federated Vision-Language Models

**arXiv ID:** 2610.12907 | [PDF](https://arxiv.org/pdf/2610.12907v1)

**作者:** Payam Abdisarabshali `[一作]` (University at Buffalo--SUNY), Seyyedali Hosseinalipour `[通讯]` (University at Buffalo--SUNY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在联邦学习框架下提出 BLEND，集成全局可聚合的视觉/文本适配器与本地个性化投影适配器，并通过双目标损失实现个性化与泛化的平衡。

**💡 创新点**

创新点包括：① 选择性聚合（只聚合全局适配器，保持本地投影适配器不传输）；② 双分支投影头（G-head 与 P-head）融合；③ 运行锚点损失与融合权重 γ 共同调节个性化与泛化；④ 采用低秩适配器实现参数高效更新。

**🔧 技术方法**

技术手段主要是 CLIP 预训练模型 + 低秩（rank‑limited）视觉/文本适配器 + 本地投影适配器 + 双目标损失（融合损失 + 运行锚点损失）+ Federated Averaging 聚合。

**📊 数据集**

使用 10 个公开图像分类数据集（ImageNet、Caltech、OxfordPets、StanfordCars、Flowers、Food101、DTD、EuroSAT、FGVC Aircraft、UCF101），并在 iNaturalist 等真实分布数据上进一步验证。

**📈 对比分析**

与 PromptFL、PromptFolio、FedPGP、FedPHA、FedOTP、pFedMMA、pFedMoAP、MMA+FedAvg 等主流 FedVLM 基线对比，BLEND 在个性化、泛化与 H‑Mean 上平均提升 4.8% 左右，且在不同数据异质性、客户端数量、样本稀缺等场景中表现最稳健。

**⚠️ 局限性**

局限性：① 仅针对文本–图像两模态；② 需手动调节 γ、λ 等超参数，缺乏自适应机制；③ 未覆盖完全去中心化/服务器无关的 FedVLM 场景；④ 对隐私保护（如差分隐私）未做集成；⑤ 在多模态、检索或生成任务上的适用性尚未验证。

---

## 226. JMT-YOLO: Multi-task Domain Adaptation via Segmentation-Guided Consistency Learning

**arXiv ID:** 2610.12901 | [PDF](https://arxiv.org/pdf/2610.12901v1)

**作者:** Zhenhua Yu `[一作]` (Xi’an University of Science and Technology), Hengxiang Zhao `[通讯]` (Xi’an University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于 Mean Teacher 的联合多任务学习框架 JMT‑YOLO，用于域自适应目标检测，融合辅助语义分割和随机遮挡一致性学习，提升模型在目标域的检测性能。

**💡 创新点**

创新点包括：① 在 YOLOv5 基础上加入 DenseRFB 语义分割头，利用密集连接和多尺度扩张捕获细粒度特征；② 通过随机遮挡和 mask‑consistency 损失强制模型学习全局语义上下文，提升实例级对齐；③ 结合 CUT 图像翻译生成伪目标域样本，增强输入多样性；④ 在缺少分割标注的场景下引入 SAM 生成伪分割标签，以实现可扩展的辅助任务。

**🔧 技术方法**

技术手段包括 YOLOv5 + Mean Teacher 框架、CUT 无配对图像翻译、DenseRFB 语义分割头、Dice 损失、mask‑consistency 损失、EMA 更新教师参数、SAM 伪分割、t‑SNE 可视化等。

**📊 数据集**

实验数据集涵盖三大域自适应任务：Pascal VOC → Clipart1k（真实→艺术）、Cityscapes → Foggy Cityscapes（恶劣天气）和 KITTI → Cityscapes（跨摄像头）。

**📈 对比分析**

与多种 SOTA 方法（SSDA‑YOLO、VT、TIA 等）在 mAP 上进行横向对比，JMT‑YOLO 在 PascalVOC→Clipart1k 获得 46.9%、Cityscapes→Foggy Cityscapes 获得 57.3%、KITTI→Cityscapes 获得 61.0%，均显著优于对比模型；在所有基准上均保持领先或接近 Oracle 性能。

**⚠️ 局限性**

局限性包括：1) 伪标签质量对最终性能影响较大，易受噪声影响；2) 需要手动调节遮挡比例和损失权重，适配不同场景困难；3) 模型复杂度较高，训练成本和推理速度相对普通 YOLOv5 较慢；4) 对极端遮挡或噪声较大的 SAM 伪分割效果有限。

---

## 227. AD-SRF: Attack-Defense Semantic Reinforcement for Multi-Modal Image Fusion

**arXiv ID:** 2610.12897 | [PDF](https://arxiv.org/pdf/2610.12897v1)

**作者:** Hao Zhang `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种攻击-防御语义强化框架 AD-SRF，先通过鲁棒的视觉融合网络生成高质量融合图像，再通过攻击-防御游戏注入不可见语义映射，兼顾视觉质量与语义性能。

**💡 创新点**

通过解耦视觉与语义两大目标，利用跨模态检索模块和信息恢复–融合协同损失构建鲁棒视觉基础；采用攻击-防御对抗机制主动挖掘并强化语义信息，实现视觉与语义的“双赢”。

**🔧 技术方法**

Transformer‑based Restorative Visual Fusion（RVF）网络、跨模态检索（CMR）模块、信息恢复–融合协同损失、攻击‑防御游戏（ADG）机制、可训练的攻击器与防御器、预训练语义分割网络。

**📊 数据集**

MFNet、FMB（低光、雾、雨等退化场景），Polarization Image Dataset、RGB‑NIR Scene Dataset、Polarization Fusion与NIR‑VIS Fusion数据集。

**📈 对比分析**

与十一种主流多模态融合方法（SeAFusion、MRFS、EMMA、KDFuse、Text‑IF、A^2RNet、TDFusion、SAGE、URFusion、OmniFuse、ControlFusion）以及三种RGB‑T分割算法（StitchFusion、MiLNet、KTB）进行对比；在视觉指标上多项指标排名第一或第二；在语义分割 mIoU 上提升 1.5–1.7 点，显著优于对手。

**⚠️ 局限性**

当前仅实现全局语义强化，对需细粒度关注的特定目标缺乏可控性；未来计划引入视觉‑语言模型结合指令，实现可控细粒度语义强化。

---

## 228. PredictIO: Decoupling Workload Prediction from OS-Controlled Storage Prefetching

**arXiv ID:** 2610.12971 | [PDF](https://arxiv.org/pdf/2610.12971v1)

**作者:** Jun Lu `[一作]` (Tsinghua University), Ju Ren `[通讯]` (Tsinghua University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个可编程框架，将工作负载预测与存储预取执行解耦，允许用户空间预测器生成页范围请求，内核统一执行。

**💡 创新点**

创新点在于统一的事件/请求接口，使多种工作负载（应用启动、数据库扫描、LLM加载）能够复用同一预取后端，并支持策略迁移与重用。

**🔧 技术方法**

技术包括eBPF事件采集、用户空间Rust运行时、共享内存请求队列、内核边界核态异步预取、Markov 模型、B树遍历提示、张量布局提示。

**📊 数据集**

实验使用五款桌面应用、SQLite、LMDB、RocksDB、llama.cpp加载的3个GGUF模型，以及不同存储设备（NVMe SSD、SATA SSD、HDD）。

**📈 对比分析**

与基线比较，应用启动延迟降低5.3–26.3%，SQLite范围扫描速度提升最高达2.99×，LLM加载延迟下降8.8–47.9%；控制路径开销低于几微秒，内核处理仅几纳秒/页。

**⚠️ 局限性**

局限性包括预测准确性依赖于工作负载结构，慢速磁盘上收益有限，部分预取需要手动提供提示，且对非文件系统或多设备共享场景的适用性待验证。

---

## 229. MotionCoach: Constructing Multimodal Guidance for Video Professional Motion Editing

**arXiv ID:** 2610.12963 | [PDF](https://arxiv.org/pdf/2610.12963v1)

**作者:** Siyi Qian `[一作]` (Peking University), Sheng Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种多轮工具使用的代理模型，自动搜索、验证并裁剪专业运动参考视频，并生成“基准提示”，为现有视频编辑模型提供可靠且充分的多模态指导，从而提升专业运动编辑的效果。

**💡 创新点**

①首次将多轮代理与自检机制结合用于视频编辑前的参考构建；②提出基于搜索+检验+裁剪工具的参考收集流程；③构建了SFT‑10k、RL‑2k训练数据以及VPMEBench专业运动评测基准。

**🔧 技术方法**

多轮代理框架、ReAct式工具调用、检索+检验+裁剪工具、Self‑Verification、GRPO强化学习、基于VLM的评判以及Qwen3‑VL等大模型。

**📊 数据集**

SFT‑10k（约10k条多轮轨迹）、RL‑2k（约2k条强化学习轨迹）、VPMEBench（314个专业运动编辑任务）以及公开视频数据集用于源视频。

**📈 对比分析**

在MiniMax‑H3和Seedance 2.0上使用VPMEBench进行评估，比较MP/PP/VQ和总体分。方案使MiniMax‑H3总体得分从55.1提升至65.6，Seedance 2.0从51.2提升至64.5，显著超过基线且接近专家指导水平。

**⚠️ 局限性**

①依赖外部检索结果的可用性；②需要额外算力和大模型作为评判与生成；③目前仅在有限的专业运动类别验证，尚未证明在更广泛领域的泛化；④对编辑模型的兼容性受限于模型的冻结状态。

---

## 230. MVGraph: Benchmarking LVLMs on Evidence Composition across Multi-View Visual Graph

**arXiv ID:** 2610.12961 | [PDF](https://arxiv.org/pdf/2610.12961v1)

**作者:** Yingjie Zhu `[一作]` (Harbin Institute of Technology), Kehai Chen `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了MVGraph基准，用于评估大视觉‑语言模型在多图视图（部分视图和时间视图）下的结构推理能力。

**💡 创新点**

创新点在于：①首次构建可跨视图分布的图推理基准；②将多图推理拆解为感知、对齐、组合与推理四个诊断维度；③为每个诊断阶段提供黄金中间表示与程序化验证。

**🔧 技术方法**

采用程序化图生成与任务模板，生成包含8种图论挑战和29个诊断任务；通过视觉图像、思考模式（Thinking）与无思考模式对比评估；使用LVLMs的视觉图解析、跨视图对齐与关系组合能力。

**📊 数据集**

使用自研MVGraph数据集，共计72,000个实例，涵盖2–8个视觉图视图；实例基于程序化生成和SmallPedia拓扑；包含部分视图与时间视图两种场景。

**📈 对比分析**

对15款LVLM（Qwen3.5-9B、Gemma-4-12B-it、GLM-4.6V-Flash、Ovis2.5-9B、Qwen3.8-Flash、GLM-5.3-Flash、DeepSeek-Flash等）进行零样本评估；在思考模式下最高平均准确率约52%；在诊断任务中发现对齐与组合的边缘水平明显不足，时间视图任务难度更大。

**⚠️ 局限性**

局限性：当前LVLM在跨视图边缘对齐和关系级组合方面表现不佳；时间推理仍是瓶颈；随着视图数量增加，组合性能急剧下降；数据集为合成图，可能缺乏真实世界图的复杂性。

---

## 231. Video Codec-Guided Training-Free Token Compression for Video Large Language Models

**arXiv ID:** 2610.12959 | [PDF](https://arxiv.org/pdf/2610.12959v1)

**作者:** Xihua Sheng `[一作]` (Hong Kong Polytechnic University), Chang Wen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于视频编解码信息的训练‑free视觉令牌压缩方法，用于提升视频大型语言模型的推理效率；

**💡 创新点**

创新点在于将参考帧依赖、运动矢量和残差等已有的压缩信号作为低成本搜索先验，用残差生成候选池进行令牌保护，用运动链传播构建局部跨帧匹配候选，实现候选空间显著缩小而不需修改原模型；

**🔧 技术方法**

技术细节包括块到令牌对齐、参考链传播、残差加权候选池构造、运动驱动的局部匹配以及基于注意力的多样性选择；

**📊 数据集**

在Video‑MME、EgoSchema、LongVideoBench和MVBench四大视频理解基准上评测；

**📈 对比分析**

与FlashVID及其他压缩基线对比，保持相当或略优的平均得分（如25%保留率下平均分58.7/58.8，100%相对准确率），并在多项指标上超过FlashVID，同时将令牌保护和跨帧匹配的特征比较次数分别降低约80%和65%；

**⚠️ 局限性**

局限性包括需在已压缩视频中提取编码信息，对不同编解码器或无压缩视频的适用性有限；对动态令牌布局的适配仍需进一步研究；在极端压缩比例下候选池过小可能影响性能；

---

## 232. Teaching neural decoders with language

**arXiv ID:** 2610.12921 | [PDF](https://arxiv.org/pdf/2610.12921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 233. Conditional Indexing Lower Bounds and Normalization Boundaries for Elastic Subsequence Matching

**arXiv ID:** 2610.12938 | [PDF](https://arxiv.org/pdf/2610.12938v1)

**作者:** Zemin Chao `[一作]`, Hongzhi Wang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过基于Online Orthogonal Vectors Indexing的归约，证明在SETH假设下，任何静态子序列索引在所有等长窗口上若预处理时间为O(N^β)（β≥1），则查询时间必定为Ω(N^{1-δ})（δ∈(0,1)），对多种弹性距离度量均适用。

**💡 创新点**

创新点在于统一处理多种DTW、DFD、ERP、TWED、MSM、LCSS、EDR、Swale、TS-SW、soft‑DTW以及归一化版本的下界，首次给出所有窗口下的完整归约框架，并在满足全窗口公共均值前提下提出线性查询时间的上界。

**🔧 技术方法**

采用Online Orthogonal Vectors的强归约、所有窗口等价性证明、块编码与锚点设计、动态规划递推分析以及有限词数的数值证明，结合SETH与随机SETH，构建了一系列严格的理论下界。

**📊 数据集**

本研究为纯理论工作，未使用任何实际数据集，而是构造符号化的整数序列和阈值以实现归约。

**📈 对比分析**

方法侧重理论证明：对比预处理和查询复杂度，展示在给定β、δ时无法实现低于O(N^{1-δ})的查询时间；未提供实验性能评估。

**⚠️ 局限性**

局限性包括：结果仅在SETH/随机SETH假设下成立，适用于所有等长窗口的精确匹配，未覆盖常数长度查询、近似解或更宽松的窗口约束；此外证明依赖特定参数切片，无法直接推广到所有输入规模或度量。

---

## 234. SEEDS: Preview-Guided Evidence Admission for Search Agents

**arXiv ID:** 2610.12932 | [PDF](https://arxiv.org/pdf/2610.12932v1)

**作者:** Jiajin Liu `[一作]` (New York University), Qiaoyu Tan `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了证据录入（evidence admission）问题，并设计了SEEDS框架，在多轮检索-推理中根据检索预览和当前推理状态选择性地将检索结果纳入推理轨迹。

**💡 创新点**

创新点在于将检索结果的可视化预览与推理状态耦合，支持任意子集的灵活录入，而非传统的固定或前缀式检索；同时兼顾减少全文检索消耗和提升答案质量。

**🔧 技术方法**

技术实现包括轻量级检索预览（标题+匹配窗口或搜索引擎片段）、基于推理状态的录入策略（可直接用提示或可训练的SFT/PPO模型）、以及全内容展开与推理链条集成。

**📊 数据集**

在9个知识密集型问答基准上评测：本地维基百科（NQ、TriviaQA、HotpotQA、2Wiki、MuSiQue、Bamboogle）与实时Web搜索（BrowseComp、BrowseComp-zh、Humanity's Last Exam）。

**📈 对比分析**

与不检索、单轮RAG、动态前缀检索、MAIN‑RAG、多轮ReAct及Search‑o1等基线对比，SEEDS在EM/F1/准确率方面均提升4–6个百分点，且全文/网页读取量下降约50–70%，证明其在性能与效率上的双重优势。

**⚠️ 局限性**

限制主要包括对预览设计的依赖（若预览信息不足或误导，录入效果受限），以及在极端噪声检索环境下对学习策略的依赖尚未完全解决；目前还缺乏大规模训练集用于优化录入策略。

---

## 235. Collaborative Learning and Reasoning over Model Slices

**arXiv ID:** 2610.12927 | [PDF](https://arxiv.org/pdf/2610.12927v1)

**作者:** Yuheng Wang `[一作]` (Shanghai Jiao Tong University), Chen Qian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出协同学习框架，利用模型切片将大型模型拆分为同等参数量的异构小模型，并通过金字塔监督训练其跨模型协作。

**💡 创新点**

创新点在于：①模型切片实现参数预算相同的异构模型集合；②金字塔监督结合协议蒸馏与步级评分，指导模型在交互中自我评估并修正；③无需预设角色，模型可在不同任务中动态分工。

**🔧 技术方法**

采用的技术包括：模型切片、LoRA微调、协议蒸馏、步级监督、任务级奖励整合、强化学习式策略梯度更新、显式任务状态交互。

**📊 数据集**

使用的数据集覆盖四类任务：多跳问答(MuSiQue)、数学推理(GSM‑Hard、MATH)、代码生成(MultiPL‑E)、开放式约束生成(Conifer)。

**📈 对比分析**

与单体Qwen3‑14B、SE‑RL、同参数MAS、MARL等基线相比，异构协同系统在13.7B参数预算下实现了平均+2.38%性能提升、最多+6.87%，同时调用次数降低39%–85%，显著提高效率。

**⚠️ 局限性**

局限性包括：对更大模型池、长交互路径和未见任务的泛化尚未验证；评价者/检查者的偏差可能影响步级监督；协同推理的额外延迟和内存开销未在参数‑加权输出‑令牌指标中体现。

---

## 236. The AI Side of the Human-AI Companion Relationship: Contributions to Users' Mental Models

**arXiv ID:** 2610.12891 | [PDF](https://arxiv.org/pdf/2610.12891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 237. OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models

**arXiv ID:** 2610.12914 | [PDF](https://arxiv.org/pdf/2610.12914v1)

**作者:** Xule Gao `[一作]` (National Key Laboratory of Space Integrated Information System), Chuxiong Sun `[通讯]` (National Key Laboratory of Space Integrated Information System)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用离线数据对 VLA 模型进行后训练，使其能够在从演示轨迹偏离后仍能恢复到目标状态

**💡 创新点**

将演示轨迹按关键状态重新标注为目标段并构造目标条件过程奖励，从而将 VLA 后训练从轨迹级行为模仿转向关键状态可达性学习

**🔧 技术方法**

离线目标标注、奖励设计、优势加权策略优化（AWAC）与 SFT 结合

**📊 数据集**

LIBERO、MetaWorld、LIBERO-Plus、LIBERO-ReAct

**📈 对比分析**

与 SFT 基线、全样本 SFT 以及在线 RL（PPO）比较，OGPO 在成功率、长周期任务、鲁棒性和恢复能力方面显著优于 SFT，并且接近或超过在线 RL 的表现

**⚠️ 局限性**

对抓取状态分割的依赖、对离线数据质量的限制以及对极端分布外偏差的恢复能力不足

---

## 238. A Lightweight Velocity Correction Framework for Improving Instruction Following in Few-Step Text-to-Image Generation

**arXiv ID:** 2610.12899 | [PDF](https://arxiv.org/pdf/2610.12899v1)

**作者:** Yating Gao `[一作]` (University of Osaka), Takuya Maekawa `[通讯]` (University of Osaka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级并行速度残差校正框架，在冻结的few-step文本到图像生成器上通过训练小型补正器来提升复杂提示的遵循能力。

**💡 创新点**

首次实现冻结生成器保持原始采样计划与前向次数不变的同时，使用并行速度残差校正，并结合On‑Trajectory Flow‑Bridge监督与共享多步学习，实现高效的指令遵循改进。

**🔧 技术方法**

采用S3‑DiT轻量级网络、Euler形式速度参数化、On‑Trajectory Flow‑Bridge监督、共享多步学习以及冻结的Z‑Image‑Turbo生成器等技术。

**📊 数据集**

使用约4K个合成场景说明（训练/验证）以及574个保留复杂场景提示进行评估，目标图像通过GPT‑Image‑2编辑得到。

**📈 对比分析**

与冻结基线、SFT+LoRA、DPO+LoRA、D‑OPSD、DiffCon等方法对比，基于574个提示实现+17.60个百分点NetWin，额外延迟仅0.84%（26.9 ms），并且仅需18.6 M可训练参数。

**⚠️ 局限性**

仅适用于较长采样步骤，短步骤下纠正后更新不足；目标图像由专有编辑器产生，可能引入偏差并影响可复现性。

---

## 239. MeshBind: Point-to-Mesh Binding for Non-Rigid Scene Flow Automatic Annotation

**arXiv ID:** 2610.12895 | [PDF](https://arxiv.org/pdf/2610.12895v1)

**作者:** Yinqi Chen `[一作]` (Southern University of Science and Technology), Qi Hao `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MeshBind，一种基于多视角 Mesh 重建、点到 Mesh 本地帧绑定以及三层流注册的自动注释框架，能够为 LiDAR 场景生成每个点的非刚性运动标签，并在此基础上构建 SFB（SUScape‑Flow Benchmark）大规模数据集。

**💡 创新点**

创新点：
1) 多视角模块利用六摄像头与 SAM 3D Body 生成可对齐 LiDAR 的 articulated mesh；
2) 本地帧绑定技术，以每个三角面切线-法向框架存储残差，实现刚性不变且可跟踪关节变形；
3) 三层流注册管线（背景→箱体→Mesh 绑定）实现全场景逐点流计算；
4) 交互式 SF‑Studio 工具，支持人机纠正 Mesh 并自动重新绑定；
5) 首个公开的全场景 LiDAR 非刚性场景流基准。

**🔧 技术方法**

采用的技术：六摄像头多视角重建、SAM 3D Body 人体网格生成、KD‑Tree + Voronoi 匹配、三角面局部坐标残差、三层流注册（ego、box、mesh）、LiDAR 相机外参校准、交互式编辑与后端重新计算。

**📊 数据集**

数据集：使用 SUScape 公共多传感器数据（LiDAR + 6 摄像头）构建 41,196 帧（2 Hz）SFB benchmark，覆盖约 310k 个人体检测（行人、儿童、骑行者等）并提供 41k 帧的非刚性点级流标签。

**📈 对比分析**

对 11 个现有场景流估计器（8 个 Argoverse 预训练、3 个学习自由）进行零射线与 fine‑tuned 对比。零射线时非刚性 EPE 较低（约 0.4 m），fine‑tuned 后整体三路 EPE 下降 2–3 倍，非刚性与刚性误差相当；SFB fine‑tuned 方案在所有指标上均超过基准。

**⚠️ 局限性**

局限性：
1) 仅对使用 SAM 3D Body 可拟合的类别自动注释，其他非刚性对象需手工或其他方法；
2) 对帧间时间间隔 0.5 s 的精度有限，快速运动仍可能产生误差；
3) Mesh 重建质量对绑定可靠性影响显著，严重遮挡或低分辨率图像时可能失效；
4) 仅覆盖 2 Hz LiDAR，其他频率场景尚未验证。

---

## 240. CB-Orchestrator: Adaptive Workflow Optimization for LLM Agents via Contextual Bandits

**arXiv ID:** 2610.13064 | [PDF](https://arxiv.org/pdf/2610.13064v1)

**作者:** Jiahang Sun `[一作]` (Chinese University of Hong Kong), Zhongxiang Dai `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个两阶段的 LLM 代理工作流优化框架 CB-Orchestrator，先通过演化搜索构建多样化的工作流池，然后利用上下文相关的 bandit 算法自适应选择最合适的工作流，显著提升了多领域任务的性能和成本效率。

**💡 创新点**

创新点包括：① 把工作流生成和选择分离，避免每个查询都要从头生成工作流；② 采用演化搜索结合执行反馈自动生成高质量工作流池；③ 将工作流选择建模为上下文 bandit，并提出 Hybrid Neural-Linear UCB 方案，实现样本效率高且易于探索；④ 通过一次性生成池，显著减少了训练阶段的 API 令牌消耗。

**🔧 技术方法**

主要技术手段：执行指导的演化搜索、基于 LLM 的工作流生成与演化、上下文向量化、混合神经网络与线性 UCB 的联合奖励预测、并行批量执行与在线学习。

**📊 数据集**

在五个基准数据集上评估：数学推理类（MATH、GSM8K）、代码生成类（HumanEval、MBPP）以及多模态/工具使用综合类（GAIA）。

**📈 对比分析**

与多种基线（普通提示、固定多智能体系统、自动多智能体系统、现有工作流搜索方法）比较，CB-Orchestrator 在所有数据集上均取得最高分数；在 MATH 上训练阶段令牌开销降低至少 40.55%，整体训练成本降低至少 29.73%，并在 GAIA 等综合任务中展现出强劲的跨域迁移和大模型适配能力。

**⚠️ 局限性**

局限性包括：① 对嵌入模型的假设（语义相似性、奖励平滑）必须成立；② 初始工作流池的质量决定最终性能，若搜索空间有限可能漏掉优秀策略；③ 仍需大量 LLM 调用，尤其在多轮演化过程中；④ 该方法主要关注工作流选择，未深入解决工作流内部的可解释性与安全性问题。

---

## 241. CliQ-RRG: Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment for QA-Style Radiology Report Generation

**arXiv ID:** 2610.13059 | [PDF](https://arxiv.org/pdf/2610.13059v1)

**作者:** Chandravardhan Singh Raghaw `[一作]` (Indian Institute of Technology Indore), Nagendra Kumar `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于疾病感知视觉‑文本对齐与图知识检索的QA式放射学报告生成框架（CliQ‑RRG）

**💡 创新点**

创新点包括疾病标签驱动的对比学习、历史图像引导注意机制以及医学知识图检索增强的生成与LLM重构

**🔧 技术方法**

使用BioMedCLIP视觉编码、CXR‑BERT文本编码、对比损失、PrAM注意力、GraphRAG检索和LLama‑3‑8B‑Instruct进行重构

**📊 数据集**

在MIMIC‑CXR和IU‑X‑Ray两大公开胸部X光数据集上训练与评估，并进行ReXGradient‑160K的零样本验证

**📈 对比分析**

与多类SOTA方法（对比、知识注入、LLM、视角融合、QA风格、RAG、MLLM）对比，CliQ‑RRG在NLG、CE、QA语义与临床指标上均取得显著提升

**⚠️ 局限性**

局限包括仅针对2D CXR、LLM未与多模态联合训练、受限于两大数据集、PrAM未建模时间序列、需监督标签

---

## 242. Space Over Statistic, Retraining Over Ranking: Auditing Training Labels Under Label-Only Backdoors

**arXiv ID:** 2610.13051 | [PDF](https://arxiv.org/pdf/2610.13051v1)

**作者:** Arnab Dey `[一作]` (Bangladesh University of Engineering and Technology), Mohammad Mahfuzul Islam `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了仅改标签的 FLIP 后门攻击，评估在不同特征空间（已训练模型、无标签预训练模型、干净标签模型）下对训练标签的审核方法，并证明使用冻结的无监督预训练特征能显著提升检测与去除攻击的效果。

**💡 创新点**

创新点包括：①将特征空间的选择作为决定审核性能的核心，而非单纯依赖统计方法；②首次将冻结的 DINOv2 ViT 编码器用于标签审核；③提出“stealthy‑flip pool”这一仅基于特征几何的指标，用于预测审核是否能消除攻击；④揭示静态检测指标（AUROC/AUPRC）并不能预测最终模型的安全性；⑤对删除成本与防御效果进行系统量化，显示传统 SPECTRE 需要删除更多样本。

**🔧 技术方法**

使用的技术包括：近邻与原型统计（以及稳健原型变体）、Confident Learning、Spectral signatures、Activation Clustering、Deep k‑NN、SPECTRE；特征空间为冻结 DINOv2 ViT‑S/14、ViT‑B/14、ResNet‑32 的 poisoned/clean penultimate 层；离线特征缓存、k‑NN 图构建、样本删除后重新训练模型。

**📊 数据集**

实验基于 CIFAR‑10 数据集，使用 ResNet‑32 作为目标网络，采用 sinusoidal trigger（和 Turner patch）以及四个源→目标对（truck→deer、dog→cat、car→truck、cat→dog）。

**📈 对比分析**

比较方法：在同一攻击、标签、指标与删除预算下，先计算 AUROC/AUPRC，然后在删除后重新训练模型评估 CTA（Clean Accuracy）与 PTA（Poison Accuracy）。结果显示：冻结 DINOv2 空间下的原型审核在相同预算下将 CTA 提升约 2–3 点、PTA 降至 0.2%；相比 SPECTRE，需删除的样本仅 6% 以内；然而检测指标最高的统计方法并不保证攻击被完全消除，显示检测质量与防御效果不一致。

**⚠️ 局限性**

局限性包括：仅在 CIFAR‑10、ResNet‑32、sinusoidal trigger 及 3% 攻击预算下测试；预训练编码器已在相似分布上训练，未验证在外域或更大图像的有效性；仅研究删除策略，未探索重标记、加权或鲁棒损失等方法；自适应攻击仅在标签选择层面，未对专家模型重新优化；Stealthy‑flip pool 仅提供二分类预测，无法细粒度排序攻击强度；实验规模受训练成本限制，未覆盖所有 90 个 CIFAR‑10 方向。

---

## 243. GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs

**arXiv ID:** 2610.13035 | [PDF](https://arxiv.org/pdf/2610.13035v1)

**作者:** Yongxin Zheng `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 GeoGR，一种两阶段压缩框架，用于同时压缩多视图 3D VLM 的视觉序列和 LLM 计算，显著降低推理成本。

**💡 创新点**

创新点包括：① GeoSemZip 将跨视图观测压缩为 3D voxel，并结合语义显著性与空间覆盖选取；② GroupRoute 引入可恢复的分组跳过（Recoverable Group-wise Skipping）与 Late Entry/Early Exit，动态分配视觉计算窗口。

**🔧 技术方法**

采用了体素化、显著性+空间覆盖 token 选择、锚点残差聚合、可恢复分组跳过、预训练后无监督数据压缩（PRISM）以及针对压缩的后训练微调。

**📊 数据集**

使用了 ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D 共 223k 样本，并通过 PRISM 选取 30%（67k）用于压缩-aware 后训练。

**📈 对比分析**

在 LLaVA-OneVision-7B 与 Video-3D LLM-7B 上与多种 2D/3D 压缩基线对比，GeoGR 在保持约 95–97% 预测性能的同时，预填速度提升 6–7×，TTFT 提升 2–2.5×，KV 缓存显著下降。

**⚠️ 局限性**

局限性：对预训练模型的视觉‑语言交互窗口与计算窗口的手工设定敏感；缺乏自适应动态窗口机制；在不同 3D 场景和模态（如纯 RGB）下的通用性仍待验证。

---

## 244. Evo$^2$: Evolving the Evolver through Weight-Harness Co-Evolution

**arXiv ID:** 2610.13025 | [PDF](https://arxiv.org/pdf/2610.13025v1)

**作者:** Chejian Xu `[一作]`, Rajiv Mathews `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了权重-驱动器共进化框架，使单一语言模型同时担任提议者（Proposer）和反思者（Reflector），通过程序编辑和执行反馈共同提升代理的程序与权重。

**💡 创新点**

创新点：1) 单一共享策略连接权重更新与程序改进；2) 交叉搜索机制将不同分支发现的有效策略迁移到其他分支；3) Evo2‑TIDE 训练管道，结合轨迹恢复、SFT 与组相对策略优化（GRPO）实现自监督与强化学习的交互。

**🔧 技术方法**

主要技术：LLM（如 LLaMA‑3 70B） + 统一 Proposer–Reflector 角色；树形自我改造搜索；交叉搜索（crossover）；编辑提取与前缀继续恢复轨迹；SFT + GRPO 训练；父节点池策略采样；任务执行器（固定模型）。

**📊 数据集**

使用的数据集包括：SWE‑Bench、LCB、SWE‑BenchVerif（编码基准）；MemoryBench（记忆任务）；60‑题子集用于专有模型实验；同时使用 GPT‑5‑mini 与 Claude Haiku 4.5 作为固定权重的专有模型。

**📈 对比分析**

与四个主流自进化框架（SICA、DGM、HGM、GEA）在同一基准上进行对比；在 SWE‑Bench 上取得 82.1% pass@1（比最佳基线 62.6% 提升 19.5%），在 LCB 上 58.0%（比最佳 56.0% 提升 2%），在 SWE‑BenchVerif 上 48.3%（比最佳 41.0% 提升 7.3%）。跨基准迁移时，表现优于所有基线，提升 2.5–18个百分点。记忆任务 F1 达 60.4%（比最佳 56.2% 提升 4.2%）。整体提升显著且没有增加推理成本。

**⚠️ 局限性**

局限性：1) 评估集中在固定执行器，未验证多模型协同；2) 需要手工设计角色提示与策略库，适配性有限；3) 对大规模算力与长序列的可扩展性尚未充分验证；4) 对专有模型的权重不可训练，仍受模型容量限制。

---

## 245. What Visual Representations Do World Action Models Need?

**arXiv ID:** 2610.13024 | [PDF](https://arxiv.org/pdf/2610.13024v1)

**作者:** Kai Liu `[一作]` (Institute of Automation Chinese Academy of Sciences), He Wang `[通讯]` (Galbot)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统研究了外部视觉表示在世界动作模型（WAM）中的选择、整合与组合，并提出了基于DINOv3、UniTok和V-JEPA的R2-WAM表示配方。

**💡 创新点**

创新点在于：①对13种预训练视觉编码器、5种特征注入机制和多源组合进行统一基线对照；②发现通用视觉表示和通道级拼接能带来最显著的鲁棒性提升；③提出只在已有通用表示的基础上增添互补特征的“表示配方”策略。

**🔧 技术方法**

技术主要包括：使用预训练视频生成模型的VAE作为基础，冻结外部编码器提取特征；通过通道拼接实现特征与VAE隐藏状态的融合；在WAM中加入未来特征流匹配损失；使用流匹配（flow‑matching）作为主要训练目标。

**📊 数据集**

数据集：LIBERO、LIBERO‑Plus、RoboCasa GR1 Tabletop、RoboTwin 2.0，以及四个真实机器人实验任务。

**📈 对比分析**

比较方法：在相同训练配置下，对比VAE‑only基线与各表示/机制/组合；评估指标为成功率（success rate），在7类扰动下统计平均与最大值。实验显示R2‑WAM在RoboCasa上提升至84.7%，在RoboTwin上提升至92.8%，相对VAE‑only分别增益5.9和2.2个百分点。

**⚠️ 局限性**

局限性：①仅评估了特定扰动和任务集合，泛化到更复杂环境尚未验证；②对预训练模型的依赖较大，若使用更小规模模型效果未知；③组合策略虽实用，但仍需更多机制探索以进一步挖掘互补信息。

---

## 246. Trajectory-Aware Policy Optimization: Closing the State Distribution Gap in Long-Horizon LLM RL

**arXiv ID:** 2610.13020 | [PDF](https://arxiv.org/pdf/2610.13020v1)

**作者:** Zhicheng Cai `[一作]` (Tsinghua University), Hao Zhou `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Trajectory-Aware Policy Optimization（TAPO），通过轨迹感知的政策优化纠正LLM长时程自回归生成中的状态分布偏移；

**💡 创新点**

从离线策略梯度推导出轨迹级状态分布校正，并引入几何均值估计与长度感知动态裁剪，实现低方差、稳定更新；

**🔧 技术方法**

结合离线策略梯度、几何均值轨迹比率、长度感知裁剪、PPO风格组优势估计以及LLM自回归MDP建模等技术；

**📊 数据集**

在数学推理与算法编码的竞赛级评测数据集（AIME24/25、BRUMO25、SMT25、Codeforces、CodeContests、TACO、APPS）以及对应的训练集（DAPO-Math-17k、Eurus-Code-25k）上进行实验；

**📈 对比分析**

与十种基线算法（GRPO、DAPO、DCPO、SAPO、CISPO、Clip-Cov、DPPO、RIPO、GSPO、GMPO）对比，TAPO在数学推理平均提升约48%、在编码平均提升约12%，并在不同模型规模下保持领先且训练稳定；

**⚠️ 局限性**

仍受长序列计算成本限制；依赖行为策略的质量，若偏差大可能影响；缺乏对长度感知裁剪收敛性的严格理论支持；对非可验证奖励任务的适用性未知。

---

## 247. Minimizing fuzzy Kripke models over residuated lattices

**arXiv ID:** 2610.13010 | [PDF](https://arxiv.org/pdf/2610.13010v1)

**作者:** Linh Anh Nguyen `[一作]` `[通讯]` (University of Warsaw), Linh Anh Nguyen (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究基于残值格的模糊Kripke模型的最小化，保持全局或局部断言。

**💡 创新点**

首次提出在任意右连续残值格下通过最大模糊自同构的商模型实现全局断言最小化，并给出在不含通用程序构造器时保持局部断言的最小化算法。

**🔧 技术方法**

利用模糊同构、商构造、最小化算法，并通过Python实现了相关工具。

**📊 数据集**

使用随机生成的模糊Kripke模型，参数化状态数、转移数、命题实例等来评估算法。

**📈 对比分析**

与已有的Gödel语义结果对比，证明商构造时间为O(mn²)，算法时间为O(mn)，实验显示在乘法右连续且单调时性能优于传统方法。

**⚠️ 局限性**

仍未能确定在乘积或Łukasiewicz结构下最大模糊自同构能否多项式时间求解，且算法仅适用于不含Baaz投射算子。

---

## 248. Gaussian-Width Bounds for Offline-to-Online Linear Bandits

**arXiv ID:** 2610.12990 | [PDF](https://arxiv.org/pdf/2610.12990v1)

**作者:** Kushagra Chandak `[一作]` (University of Alberta), Csaba Szepesvári `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在存在离线数据的线性多臂赌博机中，如何利用部分覆盖的离线数据提升在线学习的性能，提出了更精细的奖励预测误差分析和基于高斯宽度的阶段化消除算法，得到更优的上界；

**💡 创新点**

创新点在于识别传统算法因对奖励预测误差保守估计导致的 √(d/k) 误差放大问题，并利用高斯宽度浓度（Gaussian width concentration）得到更紧的误差上界，从而实现 k-维有效学习，匹配下界；

**🔧 技术方法**

采用了高斯宽度浓度、固定设计理论、离线-在线数据分割技术以及阶段化消除框架；

**📊 数据集**

未使用真实数据集，全部为理论模型和合成离线样本（固定设计、高斯噪声）；

**📈 对比分析**

与现有 LinUCB 与传统阶段化消除（获得 O(√(dkT)) 的上界）对比，我们的算法在部分覆盖情况下取得 O(k√T) 的上界，且匹配下界，说明性能显著提升；

**⚠️ 局限性**

局限性包括：仅在特定的覆盖条件（等覆盖、k-未覆盖）下获得匹配上界；对任意离线覆盖情况尚未给出匹配结果；假设离线数据为固定设计且满足高斯噪声，适用范围受限；未提供实验验证。

---

## 249. Provenance-Aware Belief Updating in Human-AI Feedback Loops

**arXiv ID:** 2610.13008 | [PDF](https://arxiv.org/pdf/2610.13008v1)

**作者:** Tianyi Huang `[一作]` (Ryquo), Wanqi Zhao `[通讯]` (App-In Club)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的训练方法——Latent Quotient Evidence Training（LQET），利用报告的来源线索让语言模型在多次传递同一信息时保持稳定预测，并在接收到新证据时按可靠性正确更新信念。

**💡 创新点**

创新点在于把证据来源的等价关系与更新一致性转化为一个关系型损失函数，结合稀疏贝叶斯监督与四种不变性约束（重复不变、更新一致、组合一致、呈现顺序一致），从而在不需要隐式学习报告依赖的前提下实现对证据的准确计量。

**🔧 技术方法**

使用了 Qwen3‑4B‑Instruct 与 Phi‑4‑mini‑Instruct 两个 4B 级语言模型，并采用 QLoRA（NF4/ BF16 量化、LoRA‑rank‑32）进行微调；训练目标由任务损失加上四项不变性约束、三项更新一致性约束和一次呈现顺序约束组成；评估使用强制式对数几率得分。

**📊 数据集**

在完全控制的二元分类基准上构造了 14 种匹配的转录视图，生成了 4,000 训练家族、400 验证家族，以及 800 OOD 迁移家族和 200 递归传递家族；每个家族包含多条报告线索与证据根。

**📈 对比分析**

与仅使用任务损失的基线相比，LQET 在 Qwen3‑4B 与 Phi‑4‑mini 上分别将新证据更新的 RMSE 降低约 41–44%，深度 8 递归漂移降低约 89–91%，并在 95% 置信区间内保持监督视图 KL 在 +0.02 的非劣势范围内；在闭环反馈模拟中，信息放大幅度进一步下降 65–90%。

**⚠️ 局限性**

局限性包括：仅在二元、合成、已知父链接的场景下验证；需要明确的源可靠性模型，无法处理缺失或错误的 provenance；评估指标为强制式二元得分，未涉及自由文本概率预测或实际决策任务；对更大规模模型或自然语言任务的迁移性仍待进一步研究。

---

## 250. CeDR-Nav: Cross-Embodiment Mapless Navigation via Offline Demonstration Recasting

**arXiv ID:** 2610.13007 | [PDF](https://arxiv.org/pdf/2610.13007v1)

**作者:** Xianghui Wang `[一作]` (Hong Kong Polytechnic University), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种离线演示重塑框架CeDR-Nav，用以在不同机器人体型之间实现无地图导航；

**💡 创新点**

创新点在于几何一致的缩放与有效性评估，能够将源体型演示在不收集新数据或在线强化学习的情况下，转换为目标体型的合法训练样本；并通过高斯混合模型（GMM）学习多模态动作分布，进一步提升了导航性能；

**🔧 技术方法**

核心技术包括：几何一致缩放、碰撞有效性评估、目标特定的GMM策略学习、点级激光扫描编码，以及使用JAX-CORL进行高效的梯度更新；

**📊 数据集**

实验数据主要来自：1）在ROS Stage上使用MAER-Nav收集的12,047条源演示（约210万状态-动作对）；2）BARN 300地图基准；3）真实世界实验中TurtleBot2、Unitree Go2、Booster K1等七种配置；

**📈 对比分析**

与基线DRL-DCLP、DWA、E-Band及E2E等方法对比，CeDR-Nav在仿真中成功率提升约13.8个百分点，在BARN基准上成功率为86.33%（略高于DRL-DCLP的84%），训练时间从几小时压缩到几秒钟；

**⚠️ 局限性**

局限性包括：依赖源演示的多样性，较大的缩放可能导致演示被排除；目前仅针对2D激光扫描；对快速动态障碍物的鲁棒性尚待进一步验证。

---

## 251. Evaluating LLM Creativity through Long-Tail Performance

**arXiv ID:** 2610.12998 | [PDF](https://arxiv.org/pdf/2610.12998v1)

**作者:** Yichen Wang `[一作]` (University of Chicago), Mina Lee `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理了82篇关于LLM创造力评估的研究，构建了九维度分类体系，并提出将创造力视为在参考分布（以Pile为基准）长尾区间内的任务表现，随后在15个创造力与标准基准上，对26个开源LLM进行实验，利用n‑gram缺失率衡量新颖性，定义并使用q@Tail(α)诊断指标来评估尾部质量与人类创意偏好的一致性；

**💡 创新点**

创新点在于：①首次把创造力归纳为参考输入‑输出分布长尾区间的表现；②提出低成本的q@Tail诊断工具，可直接从现有基准中评估尾部质量；③通过实证证明尾部表现与人类创意判断高度相关，揭示现有创造力方法多偏向提高峰值新颖性而忽视尾部质量。

**🔧 技术方法**

核心技术包括：①使用Infini‑Gram‑Mini在Pile上检索n‑gram缺失率作为新颖性度量；②以P_ref(x,y)=P_ref(x)·P_ref(y|x)构建参考分布并归一化为s_ref；③在多模型、多基准上计算q@Tail(α)以及Kendall τ与LM Arena创意写作排行榜对齐；④对比多种创造力提升方法（min‑p、温度缩放、Persona、Branch‑Solve‑Merge等）的尾部质量与新颖性。

**📊 数据集**

数据集与基准包括：参考文本Pile；15个基准（10个自由写作、2个数学推理、3个开放式问答）；26个开源LLM（Llama、Qwen、OLMo、Gemma、Mistral等）。

**📈 对比分析**

比较方法：将模型在每个基准上的质量得分排序，并与LM Arena创意写作排行榜进行Kendall τ对齐；进一步将模型按q@Tail(α)和整体质量/新颖性进行比较。实验表明：在高尾部子集上，模型排名与人类偏好的一致性显著提升（τ提升约0.3-0.4）；大多数创造力方法虽提升峰值新颖性，但尾部质量下降，min‑p采样仅在尾部质量上有显著提升。

**⚠️ 局限性**

局限性包括：①n‑gram缺失率仅衡量表面稀有性，无法捕捉语义/概念新颖性；②参考分布选择固定为Pile，可能不适用于所有领域；③质量评估依赖奖励模型或exact‑match，易受模型偏差影响；④人类创意参照仅来自LM Arena创意写作，未覆盖数学、科学等领域；⑤实验仅涵盖7‑8B参数开源模型，无法验证更大规模模型的表现；⑥q@Tail仅为诊断工具，无法替代精心设计的创造力基准。

---

## 252. What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training

**arXiv ID:** 2610.12995 | [PDF](https://arxiv.org/pdf/2610.12995v1)

**作者:** Yuxuan Gu `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用任务条件信息引导固定演示的监督细调方法CIG‑SFT，旨在更精准地把握每个token的学习价值并提升大语言模型在推理与代码生成任务上的表现。

**💡 创新点**

创新点在于：①通过对比有任务输入与无任务输入的token概率，计算点位条件信息增益 c_t，量化任务对token的真实影响；②将 c_t 视为奖励，使用KL近端投影得到信息‑近端目标分布；③推导出与该投影梯度等价的标量加权SFT目标，从而无需额外投影步骤即可实现。

**🔧 技术方法**

主要技术包括信息理论量化（条件信息增益）、KL近端投影（信息‑近端目标）、梯度等价加权SFT、冻结参考模型与当前策略分离的设计，以及针对固定演示的自适应token权重计算。

**📊 数据集**

实验使用的数学推理数据集为 Qwen2.5‑Math‑1.5B/7B 训练在 MATH500、Minerva、OlympiadBench、AIME 2024/25、AMC 2023；代码生成数据集为 Qwen2.5‑Coder‑3B 训练在 HumanEval+、MBPP+、LiveCodeBench v5/v6、BigCodeBench‑Full、DS‑1000。

**📈 对比分析**

与标准SFT、DFT、IDFT、ASFT、PriFT 等自适应 SFT 以及 DPO、SimPO、GRPO、DAPO 等 RL 基线相比，CIG‑SFT 在大多数基准上均取得更高的 Avg@16/Pass@16，且在低数据量、不同学习率以及 λ 参数变化下保持稳健的性能提升。

**⚠️ 局限性**

局限性包括：①依赖冻结参考模型的预估信息，若参考模型与目标任务分布差距过大可能产生误导；②对超参数 λ 的选择仍需经验调优；③在更广泛的领域（如多模态、对话等）及在线/增量学习场景的适用性尚待验证。

---

## 253. CoBe: A Benchmark For Conversational Counterfactual Text Editing

**arXiv ID:** 2610.13057 | [PDF](https://arxiv.org/pdf/2610.13057v1)

**作者:** Arvind Raghavan `[一作]` (Columbia University), Elias Bareinboim `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了对话式因果文本编辑基准Counterfactual Benchmark，测试LLM在无提示的情况下进行对话式因果编辑的能力。

**💡 创新点**

首次提供基于Pearl因果层级的、开放域、对话式因果编辑数据集，配备专家验证的评估标准，揭示模型主要以关联推断而非因果推理。

**🔧 技术方法**

利用三步因果编辑框架，采用对话式子句"would have...had"作为查询；使用Claude‑Opus‑4.7等大型LLM作为评估者自动检测编辑是否符合保持不变、变更及因果连词约束。

**📊 数据集**

共2,000个样本（含200个公开样本），覆盖医疗、科学、工程、商业、法律等六大领域，基于学生研究者提交的核心示例再通过LLM扩增并人工筛选得到。

**📈 对比分析**

在三种自然对话提问形式下，九款模型（包括GPT‑5.4‑Pro、Claude‑Opus‑4.7、Gemini‑3.1‑Pro、Llama‑70B等）平均准确率仅为58–61%，表明现有模型难以在无指导下实现因果编辑。

**⚠️ 局限性**

基准仅聚焦于文本因果编辑，未涵盖多模态或更广泛的因果推理；数据集可能存在作者偏倚，评估也可能被模型利用删改策略通过检查。

---

## 254. A4-MR: Abnormality-aware Anatomical Alignment for MR

**arXiv ID:** 2610.13056 | [PDF](https://arxiv.org/pdf/2610.13056v1)

**作者:** Noel D'souza `[一作]`, Erhan Bas `[通讯]` (GE Healthcare)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种面向三维MRI的解剖学感知多模态基础模型，并在约5万份临床扫描上进行预训练，随后通过零样本和轻量级下游任务实现检索、定位、分割、异常检测与报告生成。

**💡 创新点**

创新点包括：①双阶段预训练策略，将自监督视觉学习与基于弱标注的解剖定位、异常感知的跨模态对齐相结合；②在DETR解码器输出的解剖级别嵌入与文本嵌入之间加入异常感知的对比学习；③实现全 3D 目标框定位与医学报告生成的零样本能力。

**🔧 技术方法**

技术核心包括：3D Vision Transformer + DINOv2 自监督预训练；DETR 解码器用于解剖定位；PubMedBERT 作为文本编码器；CLIP‑style 对比损失结合异常标记；LLM（Claude‑Sonnet）抽取报告中的解剖与异常；轻量级下游头实现检索、分割、定位与生成。

**📊 数据集**

预训练数据为约5万份 MRI 研究（约49.1万系列），覆盖 16 个体位、多种序列与 16 家扫描仪厂商，约一半带有放射学报告。评估数据包括：检索集、CirrMRI600+（肝脏异常检测/分割）、AMOS22（解剖定位）、内部盆腔（Dataset A）与前列腺（Dataset B）分割/报告生成、KneeSeg（膝关节分割）。

**📈 对比分析**

与 Decipher‑MR、MedSigLIP、MedSAM2、MedNeXt、nnUNet3D 等现有模型对比，本文模型在 0‑shot 检索（mAP+1.2）、异常检测 AUROC (+1.7)、膝关节分割 Dice (+1.4) 等指标均取得领先或相当的表现；在 3D 解剖定位上，零样本 IoU 66.3% 超过单一 2D prompt 的 MedSAM2，且在大部分器官上优于基准。

**⚠️ 局限性**

局限性：①弱标注噪声（解剖框由分割模型、DICOM 变换与 LLM 提取）影响表征质量；②8×8×8 的 patch 分辨率对小解剖（如肾上腺）不足；③异常建模仅为二值化健康/异常，缺乏严重度与病灶级细节；④解剖词表固定，无法定位词表之外的结构；⑤模型尚未将病灶/肿瘤等病理信息直接纳入预训练。

---

## 255. Towards Easier-to-Tune Training Algorithms: New Results from AlgoPerf

**arXiv ID:** 2610.13036 | [PDF](https://arxiv.org/pdf/2610.13036v1)

**作者:** Priya Kasimbeg `[一作]` (Google), George E. Dahl `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过更新的自调优基准，评估并比较了13种深度学习训练算法在固定硬件（4×A100）上的壁钟时间表现；

**💡 创新点**

创新点在于提出了新的Decoder‑only语言建模工作负载，升级硬件配置，并通过自调优规则集（统一超参数）逼近真实应用场景；

**🔧 技术方法**

采用了多种技术手段，包括Schedule‑Free AdamW的批归一化校正、Muon的分布式张量分片与Newton‑Schulz预正交化、AdEMAMix的慢EMA温度调度，以及对超参数敏感度和保留率的1D Wasserstein距离分析；

**📊 数据集**

使用了九个跨领域工作负载，涵盖图像（ImageNet ResNet‑50、CIFAR、MNIST）、推荐（CTR）、语音识别（Wav2Vec）、分子属性预测、语言建模（NanoDO）、机器翻译等；

**📈 对比分析**

通过壁钟时间、步数效率以及松弛目标的排行榜，对算法进行对比，Schedule‑Free AdamW在自调优排行榜中名列第一，MuON和Single‑Worker DiLoCo在步数效率上表现突出，但需系统优化以抵消高步长成本；

**⚠️ 局限性**

主要局限在于自调优规则下仍需对超参数进行全局搜索，导致调优成本高，且不同算法在跨任务泛化能力和保持率方面差异显著，未来需进一步降低超参数敏感度以提升实用性。

---

## 256. Target-Constrained Pooling for Trajectory Extrapolation

**arXiv ID:** 2610.13034 | [PDF](https://arxiv.org/pdf/2610.13034v1)

**作者:** Ning Wang `[一作]` (Cornell University), Sainyam Galhotra `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种目标约束聚合（TCP）方法，用于在仅观察到部分轨迹时，通过利用已完成的相关轨迹来推断目标轨迹的最终端点，从而实现更精确的轨迹外推。

**💡 创新点**

创新点在于将轨迹分解为终端端点与形状两部分，分别使用均匀聚合和匹配增量两种互补的端点信号；引入MMD核聚类实现代表性源选择；利用目标一致投影、校准强度和退化感知加权融合多源信息；这些手段共同显著提升了外推精度。

**🔧 技术方法**

技术手段包括：① 轨迹分解（V_T 与形状 h(r)）；② 均匀聚合与匹配增量聚合两种端点估计；③ MMD核取样与代表性源选择；④ 端点导向的残差桥校正与等距投影；⑤ 对校正强度的凸优化校准；⑥ 退化感知加权的受约束混合。

**📊 数据集**

使用了五个真实数据集：gCSI、CTRPv2、GDSC2（药物剂量-反应曲线）；OLMo2（模型评估曲线）；LCDB（学习曲线）。每个数据集按队列分组，提供多条相似轨迹。

**📈 对比分析**

在25,245条目标轨迹上与五个基准预测器（包括参数化模型、NSL-PFN、LKGP）进行对比，形成100个基准组合。实验表明 TCP 在所有基准预测器上平均降低 RMSE 35.8%–42.1%，提升了所有 100 次对比；即使在基准预测器已使用相关源的情况下仍保持显著提升，并且在计算资源上相比完整源池减少了 12.7 倍输入和 138.8 倍匹配距离。

**⚠️ 局限性**

局限性包括：① 需要大量已完成且与目标相关的源轨迹；② 对端点方向的假设需满足，若方向错误投影会导致误差；③ 虽已降低源选择成本，但在极大规模数据集仍存在计算负担；④ 仅在连续、单调或有限范围内的轨迹上验证，尚未在更广泛领域（如金融、物理）进行评估。

---

## 257. Dataset-Specific Encoding of Image Content Underlies Cross-Dataset Failure in Brain MRI Foundation Models

**arXiv ID:** 2610.13027 | [PDF](https://arxiv.org/pdf/2610.13027v1)

**作者:** Minh Sao Khue Luu `[一作]` (Novosibirsk State University), Bair N. Tuchinov `[通讯]` (Novosibirsk State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6215c339-3735-4be3-8a07-5bbb7004712d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文通过在17个公开脑部MRI数据集上使用冻结的基础模型特征，进行留一数据集的线性探针实验，评估影像序列（T1、T1c、T2、FLAIR）和性别等内容标签的跨数据集迁移性能，并通过方向一致性度量和居中测试诊断迁移失败的根本原因。

**💡 创新点**

创新点在于提出并验证“方向一致性”这一诊断工具，揭示冻结特征迁移失败主要源于不同数据集中文件间类方向的差异，而非单纯的源信息或数据集位移；同时系统评估了四种去除数据集信号的方法（INLP、LEACE、CORAL、对抗训练）在此场景下的无效性，并证明广义预训练（3DINO、SAM‑Med3D）优于脑部专属预训练。

**🔧 技术方法**

主要技术包括冻结特征提取、线性探针分类、中心化测试（label‑free 与 oracle）、方向一致性得分计算、INLP、LEACE、CORAL、对抗性对齐、聚类一致性评估，以及使用k‑means + Hungarian 匹配的无标签迁移预测。

**📊 数据集**

使用了17个公共成人脑MRI数据集（包含肿瘤、多发性硬化、卒中、癫痫、神经退行性疾病和健康对照），其中10个具有多种序列（T1、T1c、T2、FLAIR），共计4,167个体素体积；性别标签在10个数据集上可用。

**📈 对比分析**

比较方法：留一数据集交叉验证的线性探针，评估平衡准确率；对比不同模型类别（基线、脑部专属、广义预训练）及其控制版本；以及对去除方法前后迁移性能的差异。结果显示广义预训练模型在序列迁移上取得最高平均平衡准确率（≈0.72），而基线模型仅约0.44；去除方法并未提升迁移性能，甚至在多数情况下降低准确率。

**⚠️ 局限性**

局限性包括：仅评估序列和性别两种内容标签，未涵盖更细粒度或疾病相关标签；使用线性探针而非非线性或微调方法；数据集数量虽多，但每个模型仅测试两种实例，可能影响统计力度；输入归一化与模型预训练差异导致部分性能差异；去除方法仅采用默认设置，未探索更丰富的监督或配对数据；预训练数据覆盖度未完全公开，可能影响结果解读。

---

## 258. EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents

**arXiv ID:** 2610.13021 | [PDF](https://arxiv.org/pdf/2610.13021v1)

**作者:** Jie Huang `[一作]` (Peking University), Yiwu Zhong `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 EmbodiedCLBench，一个用于评估自我进化智能体在具身环境中持续学习能力的基准，采用从基础任务到高级任务的组合式生成与离线状态转移图评估。

**💡 创新点**

首次在具身仿真中通过组合式通用化评估持续学习，引入基于基础与高级任务的层级化设计以及无训练的离线评估流程。

**🔧 技术方法**

使用 TongSim 仿真平台、状态转移图、In‑Context Learning 与 Skill Learning 两种经验利用方式，并在 OpenClaw、Claude Code、Codex、Hermes 等 harness 上结合 Qwen3.8‑Flash 与 GPT5.6‑Terra 等大模型进行实验。

**📊 数据集**

任务数据由 Qwen3.7‑Plus 自动生成并人工校验，共 268 条任务（168 条基础、100 条高级），涵盖 6 类日常家居活动和多种原子动作。

**📈 对比分析**

通过对比零射击、ICL、Skill Learning 三种评估设置，发现高级任务零射击成功率约 10%，加入基础任务经验后提升至 30%–35%，提升随任务复杂度稳定但不随学习样本数量递增；不同 harness 与模型表现基本一致。

**⚠️ 局限性**

局限性包括：额外学习样本对性能提升不显著，模型无法充分利用多次经验；当前成功率仍远低于饱和；评估仅在离线图中进行，缺乏真实在线动态交互验证。

---

## 259. SlackSwap: Bandwidth-Aware Activation Swapping and Recomputation for Memory-Efficient DNN Training

**arXiv ID:** 2610.13003 | [PDF](https://arxiv.org/pdf/2610.13003v1)

**作者:** Hui Feng `[一作]` (University of California Merced), Yuxin Wang `[通讯]` (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SlackSwap，基于主机链路预算动态决定激活交换与重计算，以最大化训练吞吐量。

**💡 创新点**

创新点在于将链路预算与重计算成本结合的两步规划算法，并实现无层级同步的运行时策略。

**🔧 技术方法**

使用链路预算模型、两步规划（Pass 1：按重计算成本排序交换；Pass 2：覆盖剩余超额的最便宜重计算）、DMA双流无阻塞复制以及离线模拟验证。

**📊 数据集**

在 BERT‑Large、GPT‑2 XL、Llama‑7B 风格模型以及 VGG‑16 的 BF16/FlashAttention 工作负载上进行评估。

**📈 对比分析**

与 vDNN、SuperNeurons、SwapAdvisor、块检查点等方法对比，SlackSwap 在激活内存仅为峰值的一半时保持 96–100% 的理想吞吐率，1.10–1.11× 最佳基线，并在 10% 预算下仍能训练，规划时间 <0.4 s。

**⚠️ 局限性**

局限在于仅模拟实现；未考虑内存碎片、HBM 争用、PCIe 共享以及 ZeRO‑Offload 或统一内存场景；需进一步在真实系统和多 GPU 环境中验证。

---

## 260. Scaling Empowered Agents

**arXiv ID:** 2610.13002 | [PDF](https://arxiv.org/pdf/2610.13002v1)

**作者:** Catherine Ji `[一作]`, Benjamin Eysenbach `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用无监督技能发现方法来实现可扩展的赋能（empowerment）最大化，构建两阶段策略网络，先用技能估计赋能，再用赋能驱动行为。

**💡 创新点**

创新点在于把技能学习（misl/CSF等）与赋能概念联结，利用技能数量作为赋能下界，并通过可训练的两步策略实现大规模环境下的赋能最大化；同时提出联合MI、双线性差分表征等技术提高稳定性。

**🔧 技术方法**

核心技术包括：无监督技能学习（CSF、METRA、GenDa）、变分下界估计赋能、联合/条件信息互相关（MI）分类器、双线性差分表征、两阶段 rollout 与技能切换策略、限制赋能自由度等。

**📊 数据集**

实验数据集/环境：Craftax-Classic（基于Crafter的高维离散环境）、BuilderBench（多方块操作）、Ant（连续运动控制）、AntSoccer（运动/球类交互），分别评估赋能估计和行为表现。

**📈 对比分析**

与现有赋能最大化基线（latentGCE、VIM）比较，实验显示该方法在所有四个环境中均能显著提升赋能估计、收集/建造/存活等任务表现，获得更多成就并展现更复杂的自我驱动行为；VIM/latentGCE在连续或大规模离散环境中表现不稳定或无显著进展。

**⚠️ 局限性**

限制与挑战：仍需近似赋能（仅下界），对表征学习高度依赖，易出现MI过拟合或崩溃；两阶段策略训练和技能切换增加实现复杂度；在极大或极复杂环境中训练成本高，且不保证完全最优赋能。

---

## 261. Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification

**arXiv ID:** 2610.12997 | [PDF](https://arxiv.org/pdf/2610.12997v1)

**作者:** Sujith Subramanian `[一作]` (Samsung R&D Institute), Rajasekhara Reddy Duvvuru Muni `[通讯]` (Samsung R&D Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了无监督的专利文本表示，比较了超球几何和欧氏几何的变分自编码器在跨分类检索中的效果。

**💡 创新点**

证明超球几何模型相较于匹配的欧氏模型，在检索跨分类先前技术时提升52%，并提供了其与官方CPC分类差异的几何解释。

**🔧 技术方法**

使用句子变换器（MiniLM、PatentSBERTa等）生成文本嵌入，构建超球与欧氏变分自编码器，并结合Poincaré球几何与欧氏几何对比训练。

**📊 数据集**

主要使用Harvard USPTO Patent Dataset（HUPD），在45,000份G06N/G06F/G06Q专利上实验，并在10个技术领域共50,000份专利进行跨域验证。

**📈 对比分析**

与传统SBERT+KMeans、LDA、BERTopic等基线方法比较，超球VAE在跨分类检索上优于欧氏VAE（52%提升），但在CPC分类一致性（NMI）上低于欧氏模型，整体训练无失败。

**⚠️ 局限性**

局限包括样本规模较小导致跨域几何信号噪声、超球VAE在分类一致性上低于更简单的基线、解码器仍为欧氏空间、检索评价仅基于CPC边界而非专家判断。

---

## 262. SwingGuard: Risk-Conditioned Multi-Agent Swing Trading with Dual-Timescale Governance

**arXiv ID:** 2610.12988 | [PDF](https://arxiv.org/pdf/2610.12988v1)

**作者:** Junzhou Su `[一作]` (Xidian University), Jiahua Xu `[通讯]` (University College London)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 SwingGuard，一个双时钟、多智能体框架，将每日长期摆动策略与5分钟风险治理耦合，实现风险条件化的策略生成与执行。

**💡 创新点**

创新点在于把风险从生成阶段就嵌入到多智能体决策中，并通过可审计的证据链将风险状态与实际执行紧密关联，实现前置风险约束与后置治理的协同。

**🔧 技术方法**

采用 GPT‑4o 驱动的技术/情绪/基本面分析器、bull/bear 争论、投资组合架构师，以及检索增强的 LLM 进行风险诊断与持续状态转移，结合确定性审计与仲裁执行。

**📊 数据集**

使用 2024年8月至2025年1月的 BTC 与 ETH 每5分钟行情、技术指标、新闻情绪、链上交易、宏观金融数据（M2）等多源时序数据，以及用于训练随机森林的历史样本。

**📈 对比分析**

与买入持有、移动平均、ATR 规则、随机森林、波动率管理、基于监管转换的规则以及单一代理 LLM 进行成本调整后的累计收益、Sharpe、最大回撤比较，SwingGuard 在两资产上均实现正收益、Sharpe 近3、最大回撤 < 6%，联合投资组合获得 2.98 Sharpe、6.44% 回撤。

**⚠️ 局限性**

局限性包括对 GPT‑4o 的高算力需求、仅在 BTC/ETH 两资产和特定时间窗口验证、未覆盖极端市场情形，且策略更新仍以每日为周期，缺乏实时动态适配。

---

## 263. Capturing Electrical Windows Through Reconfigurable Optical Networks: Transfer-aware Scheduling of Multi-Region AI Data Centers

**arXiv ID:** 2610.13061 | [PDF](https://arxiv.org/pdf/2610.13061v1)

**作者:** Xi Tong `[一作]` (Hong Kong Polytechnic University), Qin Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个跨区域 AI 数据中心的迁移调度框架，将训练任务迁移、检查点传输与可重配置光学电路切换耦合，以在电价/可再生能源窗口内提升本地可再生利用并降低电网采购成本。

**💡 创新点**

创新点包括：① 引入“transfer‑aware”调度概念，量化电窗口可达性；② 将可重配置光学电路（OCS）纳入计算‑电力‑网络协同模型，实时重配网络带宽捕获短暂低价/高可再生窗口；③ 采用三层日间、事件触发与实时级调度实现动态可行性验证。

**🔧 技术方法**

使用的技术包括：① 线性规划与多目标优化的 compute‑electricity‑network 共优化模型；② MEMS 光学电路切换与共享链路容量模型；③ 事件驱动滚动规划、网络延迟估算与校准；④ 结合真实可再生、电价、网络负载与训练任务跟踪数据。

**📊 数据集**

使用的数据集包括：① AcmeTrace 的 Kalos 轨迹中 16 个大规模预训练作业；② 基于 BurstGPT 请求量的推理任务；③ CAISO 的日间与实时电价与可再生发电数据；④ 构造的七条 100 Gbps 链路和一条 OCS 的网络拓扑与资源。

**📈 对比分析**

通过与四种基线（不迁移、理想迁移、静态 OCS、动态 OCS）比较，实验显示动态 OCS 在五个 AIDC 基准上将实时电网采购成本降低 12.5%，本地可再生利用提升 2.72 个百分点；相较于静态 OCS，迁移次数提升 60%，迁移延迟下降 75%，峰值与最大斜率降低约 30%，但累计斜率略升 1%。

**⚠️ 局限性**

局限性在于：① 仅在单站点电网接口层面建模，未考虑 AC 网络与储备市场约束；② 检查点大小与网络参数人为设定；③ 场景为构造的五站点实验，推广至更大规模或不同网络结构需进一步验证；④ 评估基于仿真而非现场运营数据。

---

## 264. KV-Repurpose: A Testbed for Evaluating Non-Prefix KV-Cache Reuse

**arXiv ID:** 2610.13060 | [PDF](https://arxiv.org/pdf/2610.13060v1)

**作者:** Taiga Ikeda `[一作]` (Kioxia Corporation), Jun Deguchi `[通讯]` (Kioxia Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现KV-Repurpose系统，实现分段非前缀KV缓存重用，并构建评测平台；

**💡 创新点**

通过段级缓存调度和位置自适应KV重用，突破传统前缀缓存限制，在超过100B参数模型上显著提升缓存命中率与响应性能；

**🔧 技术方法**

在vLLM基础上改造，加入段级调度、RoPE位置校正、LRU淘汰、Prometheus/Grafana监控、伪检索+代理请求构造；

**📊 数据集**

使用LongBench Pro及其七个子集（QASPER、QMSum、GovReport、HotpotQA、2WikiMultihopQA、MuSiQue、Multi-News）以及T5.1 Full‑Sentence Citation Alignment任务的文档；

**📈 对比分析**

与LMCache的前缀缓存做对比，评估TTFT、吞吐量、缓存命中率；在Zipfian RAG负载30GB缓存下命中率63.9%对比3.3%，p50/p95 TTFT更低；在单文档QA/摘要任务中质量接近重算，略有下降；

**⚠️ 局限性**

仅实现简单位置校正，未结合更高级的PIC技术；实验仅针对单一大模型，未验证多模型/架构及大规模部署的效果；

---

## 265. NOFACE3D: NOise-scheduled Focused Anchoring and Controlled Enhancement for 3D Generation

**arXiv ID:** 2610.13050 | [PDF](https://arxiv.org/pdf/2610.13050v1)

**作者:** James Rose `[一作]` (Roblox), Haomiao Jiang `[通讯]` (Roblox)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无关、可局部细化单图像生成3D资产的框架NOFACE3D；

**💡 创新点**

通过将待细化区域重新缩放至生成器输入体积、重新编码为全分辨率潜变量，并采用基于边界距离的噪声调度实现局部高分辨率重生，而无需重新训练；

**🔧 技术方法**

利用冻结的TRELLIS.2图像‑到‑3D生成器、流式采样（flow‑based sampling）与空间变异噪声调度、潜变量再锚定技术；

**📊 数据集**

在PAniC‑3D、TexVerse以及自建的339个细节区域标注基准上进行评估；

**📈 对比分析**

与原始TRELLIS.2、DetailGen3D、Elevate3D、Easy3E等方法对比，NOFACE3D在几何Chamfer、SNE、纹理PSNR/SSIM/LPIPS以及GPT‑5.6评测的“整体与几何细节”上均优于其它方法，且人类偏好实验显示显著提高；

**⚠️ 局限性**

受限于冻结生成器的先验，无法修复如文本标签等特殊细节；重编码虽提升空间容量，但生成能力仍受限于原模型。

---

## 266. Provably Subexponential Algorithms for Bounded-Error LWE with Few Samples

**arXiv ID:** 2610.13043 | [PDF](https://arxiv.org/pdf/2610.13043v1)

**作者:** Haoxing Lin `[一作]` (National University of Singapore), Prashant Nalini Vasudevan `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种变体的Wagner/BKW风格算法，用于解决具有界限秘密的学习与错误（LWE）问题，能够在给定的样本数量下以亚指数时间复杂度解决决策和搜索问题。

**💡 创新点**

创新点在于通过使用较少的样本数量和更小的错误界限，成功地将LWE问题的复杂性降低到亚指数级别，同时保持高概率的成功率。

**🔧 技术方法**

使用了Wagner的k-tree算法和傅里叶统计的下界来分析算法的性能，结合了对样本的精细处理和过滤。

**📊 数据集**

使用的样本集包括n个秘密坐标和m个线性方程，具体的样本数量和错误界限在算法中被严格控制。

**📈 对比分析**

与之前的算法相比，本文的方法在样本数量上有显著的减少，且在相同的时间复杂度下，能够处理更小的错误界限，性能表现优越。

**⚠️ 局限性**

限制在于算法的成功概率依赖于样本的选择和参数的设置，且在某些情况下可能会受到线性依赖的影响。

---

## 267. TestSuiteBench: Coding Agents Struggle to Generate Strong Test Suites from Specifications

**arXiv ID:** 2610.13065 | [PDF](https://arxiv.org/pdf/2610.13065v1)

**作者:** Kazuki Egashira `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 TestSuiteBench 基准，用以评估代码生成模型在从自然语言规范生成完整测试套件时的能力。

**💡 创新点**

创新点在于提供 165 个开源项目的详细规范、参考实现、参考测试和针对每个需求的单一故障突变体，填补了现有仅针对函数或小规模问题的测试生成基准空白。

**🔧 技术方法**

技术包括基于大型语言模型的规范生成与细化、双向图最小流算法挑选测试以构造突变体，以及自动化评估框架测量变异得分和有效率。

**📊 数据集**

数据集为 165 个 Rust/Go/C++ 开源命令行工具的源码、规范、参考测试和平均 74 个突变体。

**📈 对比分析**

比较方法是让 6 种前沿编码代理在相同预算下生成测试套件，并通过变异得分和有效率对比；最高性能的 Claude Opus 5 仅达 55.5% 的变异得分，远低于其实现通过率。

**⚠️ 局限性**

局限性包括突变体只覆盖一小部分需求、规范仍需人工细化、基准仅限编译型 CLI 应用，且所有基准均基于已有实现，可能高估代理真实测试生成能力。

---

## 268. Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents

**arXiv ID:** 2610.13058 | [PDF](https://arxiv.org/pdf/2610.13058v1)

**作者:** Ziang Ye `[一作]` (University of Science and Technology of China), Yixin Cao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReTRAIL框架，使大型语言模型代理能够在长期任务中通过环境回溯与经验复用实现主动恢复与再探索。

**💡 创新点**

创新点在于将环境级回溯与共享执行/反思策略结合，并利用State‑Anchored GRPO与分支进展奖励两种互补信号来训练代理。

**🔧 技术方法**

主要技术包括环境回溯接口、Retrospective Exploration（回溯+经验提炼）、State‑Anchored GRPO、branch‑advancement reward、共享策略训练。

**📊 数据集**

在四个交互式基准上评估：MineSweeper、Sokoban、ALFWorld和ScienceWorld。

**📈 对比分析**

与零样本提示、ReAct、Reflexion、GA‑Rollback以及RL基准（GRPO、RLOO、GiGPO、BEACON、LaMer）比较，ReTRAIL在回溯启用（T@1、T@3）上均领先，尤其在长任务和多次尝试下显著提升成功率。

**⚠️ 局限性**

局限性包括：对环境回溯的依赖性高，未在真实世界环境中验证，回溯决策仍需手工设定阈值，且仅在四个仿真基准上测试，可能不具备普适性。

---

## 269. UniMergeTrack: Hierarchical Model Merging for General RGB-X Visual Object Tracking

**arXiv ID:** 2610.13053 | [PDF](https://arxiv.org/pdf/2610.13053v1)

**作者:** Shiao Wang `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为 UniMergeTrack 的 RGB‑X 追踪框架，采用层次化模型合并（跨任务合并骨干、同任务合并跟踪头）并进行分布对齐微调，实现多模态（热红外、事件、深度）追踪的统一模型。

**💡 创新点**

创新点在于：① 通过高通频域滤波合并跨任务骨干，避免低频干扰；② 用轻量级专家提取任务特定信息；③ 在同任务内部使用最优传输对齐跟踪头通道再融合；④ 结合分布对齐微调提升骨干与头部的兼容性，最终在不重新训练的前提下实现跨模态知识共享。

**🔧 技术方法**

技术方法包括 Task Arithmetic、频域高通滤波、最优传输 (OT) 通道对齐、轻量级专家提取、分布对齐微调、HiViT Transformer、中心化跟踪头（focal、GIoU、L1、交叉熵损失）等。

**📊 数据集**

实验使用了 LasHeR、RGBT234、VisEvent、VOT‑RGBD22、COESOT、CDTB 等公共数据集，并在训练阶段分别采样 LasHeR、VisEvent、DepthTrack 与 TNL2K 数据。

**📈 对比分析**

与包括 OSTrack、ProTrack、ViPT、SeqTrackv2、OneTracker、SDSTrack、UnTrack、CMDTrack、STTrack、SMSTracker、XTrack、SUTrack、UniSOT、SEATrack、UETrack、UTPTrack、TGTrack、OneTrackerV2 等先进 RGB‑X 追踪器在 AUC、PR、MSR、MPR、EAO、Accuracy、Robustness 等指标上对比，UniMergeTrack 在 LasHeR（61.8% AUC）、RGBT234（70.6% MSR）、VisEvent（64.1% AUC）、VOT‑RGBD22（76.8% EAO）等均取得领先或竞争性表现。

**⚠️ 局限性**

局限性包括：① 仍需为每个任务保留轻量级专家，难以一次性适配所有 RGB‑X 组合；② 未在更广泛的多模态（如光谱、雷达等）或实时高帧率场景下进行评估；③ 合并过程中可能忽略任务间细粒度差异，导致在极端光照或遮挡场景的鲁棒性下降。

---

## 270. RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models

**arXiv ID:** 2610.13041 | [PDF](https://arxiv.org/pdf/2610.13041v1)

**作者:** Ke Wan `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究循环深度语言模型的推理阶段，提出在循环深度中采用混合精度策略，即先用低精度FP8进行多步计算，再在终端阶段切回高精度BF16，以降低前填充时间并保持精度。

**💡 创新点**

理论证明在局部收敛的循环动态下，早期低精度误差会被后续步骤衰减，提出“RAMP”策略，并实验验证该策略优于随机或反向安排的混合精度。

**🔧 技术方法**

使用权重和激活量化（FP8）、混合精度调度、循环深度Transformer、误差传播分析以及NVIDIA H100原生FP8运算。

**📊 数据集**

使用WikiText（固定端点）和PG19（长文本）两套数据集进行语言建模损失、隐藏状态和logit误差评估。

**📈 对比分析**

与BF16全精度、BF16-8、All-FP8、Early-BF16等基线在相同模型、相同推理深度下比较；实验显示RAMP在4K上下文保持ΔNLL≈0、隐藏状态误差≈0.2%，在H100上实现约1.16×的首个前填充速度提升，约13.9%延迟下降。

**⚠️ 局限性**

局部收敛假设下理论成立，对极大上下文或不同模型结构的鲁棒性未验证；需要GPU原生FP8支持，缺乏对CPU或旧显卡的适配；未给出训练阶段的自适应精度策略。

---

## 271. Beyond Hypercontractivity: Stronger Communication Converses via Rényi Contraction

**arXiv ID:** 2610.13030 | [PDF](https://arxiv.org/pdf/2610.13030v1)

**作者:** Ryo Hommura Moore `[一作]` (Okinawa Institute of Science and Technology), Amedeo Roberto Esposito `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了Rényi散度的强数据处理不等式（SDPI）与超收缩性（hypercontractivity）之间的关系，并利用Rényi SDPI推导了更强的矩形概率上界；将该结果应用于单向通信辅助协议的协同一致性（agreement distillation），得到改进的通信下界；通过二进制Rényi反演（binary inversion）进一步简化了下界表达式。

**💡 创新点**

创新点在于提出了L^α-幂收缩（power contraction）的概念，将Rényi SDPI常数与Markov核的双重算子结构相结合，证明其比传统超收缩不等式给出的矩形概率上界更强；此外，将这一理论应用于通信协议的逆向证明，获得比以往超收缩逆向方法更紧的通信下界。

**🔧 技术方法**

采用了Rényi散度、Markov核的对偶算子、L^α-幂收缩、超收缩性、Riesz–Thorin插值、二进制Rényi反演等数学工具；在证明中结合了凸性、Jensen不等式和数据处理不等式的泛函分析手段。

**📊 数据集**

本文主要基于理论分析和构造的离散例子，使用二元对称二进制信道（DSBS）和稀有事件Z信道（rare‑event Z‑channel）作为示例，并在此基础上做了数值仿真。

**📈 对比分析**

与传统的超收缩性逆向（hypercontractivity converse）相比，作者得到的通信下界在数值上至少不弱，且在某些参数区间内更优（如DSBS例子中0.2640>0.2604）。示例表明，二进制Rényi反演方法可在相同设置下取得更高的通信下界。

**⚠️ 局限性**

局限性包括：只适用于α>1的Rényi散度；需要显式计算或估计SDPI常数，计算复杂度较高；在极端参数或非对称信道情况下，改进幅度有限；理论主要针对离散二元信道，扩展到更一般的分布或高维情形仍需进一步研究。

---

## 272. Low-Rank Kolmogorov-Arnold Generator Flows for Interpretable Operator Learning

**arXiv ID:** 2610.13028 | [PDF](https://arxiv.org/pdf/2610.13028v1)

**作者:** Minjong Cheon `[一作]` `[通讯]` (Korea National Open University), Minjong Cheon (Korea National Open University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种Kolmogorov–Arnold流（KAFlow），通过学习低秩可分离符号的生成器来表示位置相关物理算子，并能自动提取闭式物理规律。

**💡 创新点**

创新点在于将生成器而非传播算子作为学习目标，利用低秩分解与Gauge固定的Kolmogorov–Arnold网络，实现低成本训练、精确指数积分以及完全自动的符号蒸馏。

**🔧 技术方法**

采用Kolmogorov–Arnold网络、低秩符号分解、精确指数积分器、自动符号蒸馏以及贝叶斯信息准则稀疏回归等技术。

**📊 数据集**

使用KANO官方数据集（G1–G3、双井底势和非线性薛定谔方程）、变量阶分数算子生成器以及五个非局部/分数/全色散动力学方程的数据。

**📈 对比分析**

与FNO、U‑FNO、AM‑FNO、PDNO、KANO等基线比较，KAFlow在外域泛化误差降低3.4–11倍、量子动力学状态不相似度下降至10⁻¹⁰级、分数算子误差提高15倍，并且参数量仅为FNO的千分之一以上，性能显著优越。

**⚠️ 局限性**

局限性包括目前仅在一维周期网格上验证，尚未扩展到高维、非周期边界或多尺度问题；对极长时间步长的稳定性和训练收敛性仍需进一步研究。

---

## 273. Post-Quantum Session Establishment for Matter over Thread: Message Size, Transport, and Protocol Options

**arXiv ID:** 2610.13026 | [PDF](https://arxiv.org/pdf/2610.13026v1)

**作者:** Daniel Commey `[一作]` `[通讯]` (California State University, Long Beach), Daniel Commey (California State University, Long Beach)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究将Matter的CASE握手迁移至后量子密码学，并量化在Thread网络上的消息尺寸、传输与协议选项。

**💡 创新点**

构建字节精确的CASE消息尺寸模型；证明无FIPS 203/204/205方案无法满足1178字节限制；提出基于块传输的IPv6片段替代方案；评估多跳Thread网络中的定时器病症。

**🔧 技术方法**

Matter TLV编码、liboqs实现、Cortex‑M4周期计数、OpenThread+Matter源码、Monte Carlo仿真和Gilbert‑Elliott丢包模型。

**📊 数据集**

Matter参考实现的19个TLV证书测试向量、OpenThread源码常数、nRF52840周期计数数据。

**📈 对比分析**

通过300次独立运行的蒙特卡洛模拟比较碎片化、拉取分段和块传输三种传输；在多跳、不同Wi‑Fi干扰下，块传输比IPv6碎片快1.8–2.6倍，降低失效率至0%；手握手时间从1.6s降低到约6.1s。

**⚠️ 局限性**

结果基于模型与仿真，未在硬件上验证；未考虑内存与缓冲区限制；仅使用单频道串行化，未考虑空间复用和TCP传输。

---

## 274. Unified Generative Video Compression with Autoregressive and Bidirectional Video Diffusion

**arXiv ID:** 2610.13023 | [PDF](https://arxiv.org/pdf/2610.13023v1)

**作者:** Tianyu Zhang `[一作]` (University of Science and Technology of China), Dong Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

统一生成式视频压缩框架UniGVC，支持同一比特流下的双向和自回归解码，兼顾高质量随机访问与低延迟流媒体。

**💡 创新点**

创新点在于通过共享层次化潜在码流条件实现单一比特流满足两种解码模式，并利用自回归条件扩散与分布匹配蒸馏将多步自回归模型压缩为一步，同时加入小型条件Tiny Decoder。

**🔧 技术方法**

采用层次化潜在视频码流、视频扩散Transformer（DiT）、LoRA适配器、分布匹配蒸馏（DMD）、对抗学习、Tiny Decoder等技术。

**📊 数据集**

训练使用MLIC-100K、Koala-36M、UltraVideo等大规模视频集，评测在HEVC-B、UVG、MCL-JCV 1080p序列。

**📈 对比分析**

与VTM、DCVC、GLC-Video等方法对比，UniGVC-BI在极低比特率下率失真感知性能最优，UniGVC-AR以36.85 FPS低延迟实现，整体性能超越现有方案。

**⚠️ 局限性**

限制在于解码端需存储双模式模型参数，极低比特率下仍存在生成细节不精确，且对科学精度要求高的应用需慎用。

---

## 275. Learning Features, Not Layouts: Feature-Axis Attention as a Transfer Prior for Medical Imaging

**arXiv ID:** 2610.13012 | [PDF](https://arxiv.org/pdf/2610.13012v1)

**作者:** Aon Safdar `[一作]` (University College Dublin), Mohamed Saadeldin `[通讯]` (University College Dublin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种严格轴解耦的特征轴优先Vision Transformer（FAD‑ViT），通过深度可分卷积提取局部空间信息，再用通道自注意力进行全局混合，探索该设计在医学影像迁移学习中的效果。

**💡 创新点**

创新点在于将全局注意力轴从传统的空间切换到特征（通道）轴，并严格解耦空间提取与全局混合，使得通道轴成为可控变量。该设计证明了通道轴的全局注意力是医学图像迁移的有效先验，并且无需窗口、局部交互或多轴门控即可实现高效的全局建模。

**🔧 技术方法**

技术细节包括：
- 三阶段卷积分辨率下降（H×W→H/2×W/2→H/4×W/4），每阶段使用深度可分卷积提取局部空间；
- 通道自注意力（Feature Self‑Attention, FSA）采用深度可分投影生成 Q/K/V，单头自注意力在通道维度上执行，复杂度为 𝒪(C²N)；
- 仅在最后一阶段保留轻量级空间注意力尾部；
- 与标准 ViT、XCiT、DaViT 等模型在同一容量、同一训练协议下对比；

**📊 数据集**

使用 23 个医学任务数据集：
- 10 个 MedMNIST‑2D 分类任务；
- 2 个乳腺超声分类（BUSI、B）；
- 10 个 MSD‑derived 2D 分割任务；
- 1 个 BUSI 语义分割；
- 8 个低数据线性探测（1%、10%、100% 标签）测试集。

**📈 对比分析**

比较方法：在统一的训练协议（卷积分词、相同参数量≈23M、相同 U‑Net 或线性头解码器、相同学习率调度和预训练），分别评估分类、分割和线性探测的性能。结果显示：
- FAD‑ViT 在所有 23 个任务中平均提升 Dice/Accuracy/AUC，尤其在纹理主导的超声任务上提升约 20‑25%；
- 在分割任务中平均 Dice 提升至 84.3%，比匹配容量的 ViT‑UNet 提升约 4.5，且参数量仅为 23.1M（比 ViT‑UNet 仅 1/4）；
- 在低数据线性探测中，FAD‑ViT 在 1% 标签下的线性可分性显著高于 ViT 与 DINOv2；
- 与 XCiT、DaViT 等通道感知混合模型对比，FAD‑ViT 以更少参数实现同等或更优性能，验证通道轴是关键因素。

**⚠️ 局限性**

局限性：
- 对高对比度、结构清晰的 CT/MRI 任务提升有限；
- 主要在 ImageNet‑1K 监督预训练环境下验证，其他自监督或多模态预训练的适用性仍需进一步研究；
- 仅评估单一通道自注意力形式（单头、深度可分投影），未探索多头或更复杂通道混合策略；
- 在某些任务中，通道轴的优势与模型容量、数据量相关，仍需进一步的归因分析。

---

## 276. EgoReveal: Learning Active Vision and Object Manipulation under Partial Occlusions from Egocentric Demonstrations

**arXiv ID:** 2610.12999 | [PDF](https://arxiv.org/pdf/2610.12999v1)

**作者:** Hiromu Kurihara `[一作]` (Keio University), Mariko Isogawa `[通讯]` (Keio University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 EgoReveal 框架，允许机器人仅凭智能眼镜收集的第一人称演示视频学习视角重定位与物体抓取，在部分遮挡下恢复可见性并完成抓取任务。

**💡 创新点**

创新点包括：①无机器人遥控、无专用教学设备，直接从自然演示学习视角控制；②将主动视角策略与抓取策略通过遮挡判定模块（ORA）动态切换；③使用基于 OwlViT、SAM 与 HSV 匹配的遮挡检测，实现实时可见性评估。

**🔧 技术方法**

技术方法包括：改写 ACT 框架为摄像机姿态预测；EgoZero 用于抓取；OwlViT+SAM+HSV 匹配实现遮挡检测；Transformer 动作块与坐标变换；RGB 与深度联合处理。

**📊 数据集**

使用自制数据集：100 条主动视角演示与 100 条抓取演示，包含 SoftToy 与 Towel 两种物体，三种遮挡器；训练时随机遮挡，测试时加入未见遮挡。

**📈 对比分析**

与基线 EgoZero 对比，SoftToy 成功率 85% 对 35%，Towel 90% 对 45%，未见遮挡 70% 对 30%；说明 EgoReveal 在部分遮挡下显著提升任务成功率。

**⚠️ 局限性**

局限性包括：仅处理部分遮挡，缺乏对完全遮挡或目标不在视野内的处理；遮挡判定依赖参考图像、颜色匹配和光照条件；需要标记对齐以统一坐标系。

---

## 277. Dynamic Matching with Abandonment: Optimal Guarantees via Static Priority Policies

**arXiv ID:** 2610.12991 | [PDF](https://arxiv.org/pdf/2610.12991v1)

**作者:** Nick Arnosti `[一作]`, Felipe Simon `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了有弃置的随机动态匹配问题，证明存在可在多项式时间内计算的静态优先级策略，其长期平均奖励至少达到全局最优（全知）基准的 0.5，即竞争比 0.5 是最优。

**💡 创新点**

创新点：1）首次证明竞争比 0.5 在所有实例上可实现；2）提出静态优先级策略与线性规划（primal）和固定点方程（dual）两种完全不同但等价的构造方法；3）给出了一个关键的概率下界（基于 Poisson 过程和 Slivnyak–Mecke 公式），用于控制可用性概率，从而实现对策略收益的上界与下界证明。

**🔧 技术方法**

主要技术：连续时间马尔可夫链分析、泊松过程的性质（PASTA、Slivnyak–Mecke）、线性规划松弛与补充性条件、可行性分割极点与子模最小化、固定点方程的存在性与唯一性证明（利用一侧 Lipschitz、Banach 收敛定理），以及多项式时间的 ellipsoid/线性规划求解与子模优化。

**📊 数据集**

无特定实验数据集，理论分析基于抽象的泊松到达、指数退役率和任意奖励矩阵；实验验证部分在论文中未包含。

**📈 对比分析**

与现有最佳竞争比 0.192 的随机策略相比，静态优先级策略在所有设置（非二分、双侧、单一等待类型等）上均能达到 0.5 的竞争比；对比实验未给出，但理论上证明了最优性。

**⚠️ 局限性**

局限性：1）仅证明了对全知基准的竞争比 0.5，针对更弱的最优在线基准的竞争比仍未突破；2）假设到达为独立 Poisson，退役为指数分布；3）奖励仅依赖到达顺序（后到者的奖励），未考虑时间变化奖励；4）算法实现需要求解指数约束的线性规划，实际规模受限；5）未讨论参数未知或学习问题。

---

## 278. Language Models Have Interpretable Narrative Mechanisms

**arXiv ID:** 2610.12989 | [PDF](https://arxiv.org/pdf/2610.12989v1)

**作者:** David Yiyang Liu `[一作]` (UNSW Sydney), Andrew Piper `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在GPT-2-Small、Gemma-2-2B和Llama-3.1-8B三种模型中，识别并消融注意力头、FFN神经元和稀疏自编码器（SAE）特征，证明这些内部组件与文本的叙事性（narrativity）存在因果关联，并能影响生成文本的叙事度。

**💡 创新点**

首次将叙事学的三维维度（代理、事件序列、世界建构）与机制解释方法结合，提出NVC（Narrativity Value Correlation）筛选框架，系统挖掘模型内部叙事相关组件，并验证其与叙事理论的一致性。

**🔧 技术方法**

采用Spearman相关的NVC方法对注意力、FFN和SAE特征进行筛选，随后执行集体消融实验；利用Neuronpedia自动生成的特征解释与零样本LLM进行维度分类；使用对比搜索生成文本并以外部模型评估叙事度。

**📊 数据集**

使用NarraDetect数据集（含394条人工金标和13330条银标）进行叙事度评估；在XStoryCloze、CosmosQA、OpenBookQA、TruthfulQA、FaithDial和BEGIN等跨语言与非叙事基准上评估消融效果。

**📈 对比分析**

与等量随机消融对比，NarraFeature消融在高叙事文本的预测损失提升最大，生成文本叙事度显著下降；在Gemma模型中跨语言（英、俄、中、印）对XStoryCloze和CosmosQA均显著衰退，Llama模型在英文表现更明显；消融对非叙事基准影响相对较小。

**⚠️ 局限性**

研究仅覆盖三种预训练模型；组件筛选阈值为ρ≥0.5，消融结果仅适用于整体组效应，未揭示单元细节；SAE特征解释可能不唯一，关注头和FFN对叙事的具体作用仍不明确；跨语言、性别与文化偏差未充分探讨。

---

## 279. Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction

**arXiv ID:** 2610.12987 | [PDF](https://arxiv.org/pdf/2610.12987v1)

**作者:** Hare Krishna `[一作]` `[通讯]` (University of Texas at Austin), Hare Krishna (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了文本语料采样对遮掩文本重建模型的预测一致性影响，提出了一阶源级公式（source law）和多项式近似，并用源分割 Jackknife 校正，仅基于参考语料即可预测两组预测器在不同语料采样下的期望不一致度。

**💡 创新点**

创新点在于：①推导出包含源长度、随机行总计和跨滞后协方差的源级期望不一致度公式；②引入基于多项式的近似和非线性源自举法，以平滑化导致不一致度下降慢于 1/n 的机制为核心；③利用源分割 Jackknife 有效校正小参考语料的偏差，从而显著提升预测校准。

**🔧 技术方法**

所用技术包括：基于词对计数的平滑化（加上平滑参数 α），线性化展开、源分割 Jackknife 校正、非线性源引导自举、蒙特卡洛模拟验证、以及小型掩蔽扩散语言模型（MDLM）实验对比。

**📊 数据集**

实验使用的主要数据集为 TinyStories、WikiText-103 和 CNN/DailyMail，采用 GPT‑2 tokenizer，分别构造 50,257 词表，分块长度 64，掩码比例 0.5。

**📈 对比分析**

通过将预测公式与多组 A/B 预测器的实测不一致度对比，校正后的多项式法在参考语料小于目标时平均绝对对数误差降至 0.085；非线性自举在参考语料大于目标时误差更低；与小型 MDLM 的对比表明，统计预测能捕捉大部分不一致输入但无法解释神经模型不一致度随语料增大而快速下降的现象。

**⚠️ 局限性**

限制在于：①依赖参考语料的估计误差，尤其是未出现的上下文导致预测下溢；②线性化近似在稀疏行或词表增长时可能失效；③校正方法假设偏差可展开且参考样本足够；④MDLM 对比仅限于小型 Transformer 与有限训练步数，无法推广至更大模型或更丰富的训练设定。

---

## 280. Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence

**arXiv ID:** 2610.13015 | [PDF](https://arxiv.org/pdf/2610.13015v1)

**作者:** Zhicheng Cai `[一作]` (Institute for AI Industry Research, Tsinghua University), Hao Zhou `[通讯]` (Institute for AI Industry Research, Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出通用的 Generalized Proximal Policy Optimization (GPPO) 框架，将近端策略优化统一到“散度 × 估计器”两维设计空间，并给出 f‑散度信任域理论。

**💡 创新点**

通过 f‑散度的几何性质与不同估计拓扑（MC、Atomic、Binary）提供系统化设计原则，解释并统一 PPO、DAPO、DPPO、RIPO 等方法，证明不同散度与估计器决定策略更新动态。

**🔧 技术方法**

使用泛化 f‑散度信任域理论（Vajda 不等式）、Atomic 与 Binary 估计拓扑、KL、Pearson χ²、TV、海明散度等技术。

**📊 数据集**

在数学推理任务使用 DAPO‑Math‑17k，代码生成任务使用 Eurus‑Code‑25k，评估八个竞赛级基准：AIME24/25、BRUMO25、SMT25、Codeforces、CodeContests、TACO、APPS。

**📈 对比分析**

与 GRPO、DAPO、DPPO、RIPO、GSPO、GMPO、SAPO、CISPO、Clip‑Cov 等算法对照，采用 avg@8 指标；最优 GPPO（KL‑Atomic）在数学推理上平均得分 41.5、代码生成 59.7，分别比 GRPO 提升 49.8% 与 13.9%，显著优于其他方法。

**⚠️ 局限性**

仍需在散度与估计器选择上进行经验调优，实验仅覆盖数学与代码两类任务，未验证在更广泛 LLM 推理场景或更大模型的泛化，估计器设计可能增加计算开销。

---

## 281. Evidential Multimodal Transformers with Dempster-Shafer Fusion for Emotion Recognition

**arXiv ID:** 2610.13013 | [PDF](https://arxiv.org/pdf/2610.13013v1)

**作者:** Muhammad Usama `[一作]` (Braindeck Inc.), Yeoreum Jung `[通讯]` (Braindeck Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于Transformer的情绪识别模型EMT-DS，利用Dirichlet分布预测情绪的主体意见，并通过可微分的Dempster‑Shafer规则对多模态信息进行融合，显式量化模态冲突和不确定性。

**💡 创新点**

核心创新包括：① 在网络内部实现全端到端可微的Dempster‑Shafer融合，梯度可通过冲突项反向传播；② 将冲突用作监督信号和安全阈值，提供可解释的拒绝决策；③ 结合KL正则化引导模型在无信息时退回到完全不确定的“空意见”，提升模型在噪声和缺失模态下的鲁棒性。

**🔧 技术方法**

技术手段包括：多模态Transformer编码器（Wav2Vec2、ViT、BERT/KoELECTRA、1D‑CNN+自注意力）、Dirichlet参数化的证据头、基于Subjective Logic的主体意见计算、Dempster‑Shafer融合层、贝叶斯风险损失与KL正则化、冲突损失与自适应温度缩放校准。

**📊 数据集**

实验使用四个公开基准：IEMOCAP（音频+文本）、DEAP（EEG+生理）、KEMDy20（音频+文本+生理+视频）、CMU‑MOSEI（音频+文本+视频）。

**📈 对比分析**

在所有数据集上与最新的Transformer、混合专家、注意力融合以及现有证据方法相比，EMT‑DS平均提升2.3~4.3个百分点的准确率，并将ECE从0.15左右降至0.04左右，显示出更优的性能与更可靠的置信校准；此外在噪声、缺失模态和跨域迁移下保持较高的鲁棒性。

**⚠️ 局限性**

主要局限包括：假设模态间条件独立，若依赖性显著可能降低贝叶斯一致性；Dempster‑Shafer融合仍需预先设定冲突阈值和正则化权重；模型相对传统注意力结构在推理时计算量略增，且对极端噪声仍需要进一步提升。

---

## 282. Hindsight Structured Advice Distillation: Steering Executor Agents with Advisors Trained on Their Failures and Task Characteristics

**arXiv ID:** 2610.13000 | [PDF](https://arxiv.org/pdf/2610.13000v1)

**作者:** O. Ozan Koyluoglu `[一作]` `[通讯]` (Samaya AI), O. Ozan Koyluoglu (Samaya AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 HSAD 的方法：在不改动冻结的执行模型（executor）的前提下，训练一个小型语言模型（advisor）生成结构化的、针对每个输入的指导（advice），并将其附加到执行模型的提示中，从而提升执行效果。

**💡 创新点**

创新点包括：①将 hindsight（事后）经验蒸馏为可生成结构化 JSON 指导的学生模型；②通过错误分析挖掘出 arms（策略）和 directive taxonomy（错误类型与修正措施）的闭合词表，使指导既包含可推断的共享规则，也包含实例特定信息；③将指导文本化而非直接改写模型权重，兼顾可解释性与低成本。

**🔧 技术方法**

核心技术：使用教师 LLM（在训练阶段可访问金标与每个策略的运行轨迹）生成指导目标；对目标进行监督微调（SFT）并用 LoRA 训练 8B advisor；错误分析、hindsight label 生成、规则与 LLM 推理混合的标签生成策略；渲染器将 JSON 转换为可追加到执行模型提示的文本；在测试时仅一次小模型推理即可。

**📊 数据集**

使用了四个公开基准：FrontierFinance（220 财务查询），WebShop（500 购物指令），BIRD（1,074 SQL 题），ExpertLongBench（100 长文本任务）。训练集分别为 1,819、6,410、9,428、450 条样本。

**📈 对比分析**

比较方法：与基线冻结执行模型、其高耗时配置、best‑arm oracle（实验中最优策略）以及教师直接执行的 ceiling 进行对比。实验结果显示：FrontierFinance 从 52.9% 提升至 56.6%（高耗时配置为 56.0%）；WebShop 评分从 54.6% 提升至 63.4%，成功率从 3.8% 提升至 23.2%；BIRD 执行准确率从 40.0% 提升至 43.1%；ExpertLongBench 复核覆盖率从 49.7% 提升至 54.5%。

**⚠️ 局限性**

局限性：①需要大量带金标的训练数据与可靠的评分器；②仅适用于存在可被指导修复的失败模式且错误可通过词表化表达的任务；③对执行模型的依赖强，跨模型迁移效果未知；④arms 与 directive taxonomy 的构建耗时且需多轮错误分析；⑤在数据稀缺或实例特定信息难以从输入恢复的场景（如 BIRD 的表/列推断）中效果有限。

---

## 283. LiveSynth: A Streaming Neural Synthesizer for Instrument Cloning and Text-to-Instrument

**arXiv ID:** 2610.13054 | [PDF](https://arxiv.org/pdf/2610.13054v1)

**作者:** Kyungsu Kim `[一作]` (Seoul National University), Kyogu Lee `[通讯]` (Seoul National University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款低延迟流式神经合成器LiveSynth，能够即时演奏、实现零射击乐器克隆、文字到乐器合成，并支持连续音色过渡与自主演奏续奏。

**💡 创新点**

创新点包括：① 反馈自由的连续序列生成器；② 按帧音色条件化实现实时音色过渡；③ MIDI条件丢弃训练实现无MIDI自续演奏；④ 同时使用音频与文字的CLAP嵌入进行零射击克隆与文本生成。

**🔧 技术方法**

采用了反馈自由Transformer生成器、基于CLAP的音色嵌入与跨模态Procrustes对齐、causal VAE音频解码、MIDI状态嵌入与RoPE旋转、条件对抗训练、ReLPS，以及实时推理的MLX int8路径。

**📊 数据集**

使用NSynth、Slakh2100以及多种音乐、语音和通用音频数据集训练VAE，10秒参考片段和NSynth元数据生成的文本提示用于评估。

**📈 对比分析**

通过MERT、PANNs、MSS、MusicCoCa、CE/PQ等指标与TokenSynth、CTD、AnySynth等离线基线对比，LiveSynth在音色相似度和频谱重建上表现最佳；实时推理实现RTF 0.88、10 ms算法延迟；在自主续奏中音高一致性良好但节奏连贯性不足。

**⚠️ 局限性**

局限性包括音高准确性和音质略低、节奏依赖弱、续奏缺乏节奏连贯性，以及缺乏人类听感评估，需要改进因果音频表示与节奏建模。

---

## 284. Failure-Resilient Multi-Robot Coordination with $k$-Connectivity Constrained Submodular Maximization

**arXiv ID:** 2610.13049 | [PDF](https://arxiv.org/pdf/2610.13049v1)

**作者:** Yoonjae Hwang `[一作]` (University of Southern California), Gaurav S. Sukhatme `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种同时考虑子模优化、对机器人失效的鲁棒性以及通信网络的 (β+1)-顶点连通性的多机器人协同方法 (RES-CON)。

**💡 创新点**

创新点在于：①把鲁棒子模最大化与连通性约束统一建模；②在非矩阵（matroid）可行域下通过 Menger 定理构造连通性过滤器和回退策略；③给出包含连通性代价的曲率基近似保证，且证明在非退化实例下近似因子为平方形式。

**🔧 技术方法**

采用子模函数、曲率分析、贪心+诱饵（bait）算法、Menger 定理的连通性检验、回退 fallback 机制以及基于离散轨迹的仿真。

**📊 数据集**

实验使用基于 2D 仿真场景的多机器人目标跟踪数据集：机器人随机初始位置、移动范围离散化、目标随机运动，覆盖目标数作为子模目标函数。

**📈 对比分析**

与 Pure Greedy、Resilient-Only、1-Connected-Only、(β+1)-Connected-Only、Fallback-ablated 等基线比较。结果显示 RES-CON 在稀疏通信密度下仍保持 (β+1)-连通，故障后断连率显著降低；在超阈值密度下 ρ≈1，性能与无连通约束方法相近，累计跟踪精度仅轻微下降。平方鲁棒因子在 99.3% 的实例上满足理论上限。

**⚠️ 局限性**

局限性包括：①连通性代价 ρ 在稀疏网络时可能显著下降；②平方近似因子证明依赖“intra‑stage”假设，尚未在所有几何情形下严格验证；③缺乏对 ρ 的几何下界；④未在真实硬件上验证通信模型与连通性保证。

---

## 285. Execution-Feasible Dispatch of LLM Pre-training Loads for Renewable-Dominant AI Data Centers

**arXiv ID:** 2610.13047 | [PDF](https://arxiv.org/pdf/2610.13047v1)

**作者:** Xi Tong `[一作]` (Hong Kong Polytechnic University), Qin Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套可执行的LLM预训练负载调度框架，将检查点信息与3D GPU并行模式结合，嵌入日间电网调度，实现绿色能源利用最大化。

**💡 创新点**

创新点在于将LLM训练的检查点约束与可执行的3D并行模式映射成可调度的电力负载状态，构建检查点状态图实现可执行的负载重塑，而非传统连续可调负载模型。

**🔧 技术方法**

采用分布式Transformer训练模型的算力与内存公式、GPU并行配置生成与支配裁剪、MILP优化、检查点状态图等技术。

**📊 数据集**

使用公开的LLaMA‑13B和LLaMA‑33B预训练任务参数、CAISO风光电发电与电价预测数据，以及A100 GPU规格与AIDC基准负载。

**📈 对比分析**

通过六种场景（固定/检查点/3D/存储组合）对比，结果显示执行可调度方案在成本上比基线下降约20%，并将可再生能源冲刷降至0.014 MWh，BESS辅助进一步提升至100%可再生利用率。

**⚠️ 局限性**

局限在于仅考虑单日确定性调度，未覆盖多日随机性、其他AI工作负载以及多AIDC协同，且依赖公开数据与模型参数，实际工业环境可能存在更复杂约束。

---

## 286. Redirecting Teacher Supervision: Temporal-Response-Guided Latent Distillation for Video-Language Models

**arXiv ID:** 2610.13045 | [PDF](https://arxiv.org/pdf/2610.13045v1)

**作者:** Xinkai Lu `[一作]` (SF Technology), Xiang Song `[通讯]` (SF Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的视频知识蒸馏方法 Temporal-Response Guidance（TRG），利用学生在局部时间扰动下的表示响应来引导教师纠正方向，从而在蒸馏过程中更有效地传递时序信息。

**💡 创新点**

创新点：①通过计算学生对局部时间扰动的Jacobian-Vector Products（JVP）提取时序响应子空间；②将教师纠正投影到该子空间并加权恢复原始范数，实现对教师纠正方向的动态重定向；③保持教师纠正中子空间之外的成分，从而兼顾补充信息。

**🔧 技术方法**

技术细节：使用基于JVP的响应测量、奇异值分解提取子空间、投影加权重构目标、对齐映射（LoRA）、对抗式训练（对齐层），以及对输出的反向KL损失结合的OPD框架。

**📊 数据集**

数据集：在4,400条视频与1,302张图像的Video-R1-260k子集上训练，随后在四大基准上评估：TempCompass、Video-MME、MVBench、Video-MMMU。

**📈 对比分析**

对比方法：与仅使用输出监督的OPD、直接对齐（Plain-AS）以及随机纠正方向（Random Orientation）进行比较。TRG在4B模型上对宏观准确率提升2.86点，在9B模型上提升3.16点；在所有四个基准上均优于OPD和Plain-AS，特别是在时序关系（如事件顺序）上表现突出。

**⚠️ 局限性**

局限性：①评估仅覆盖单一教师族和两种学生规模；②在相同计算预算下的性能未作对比；③仅在训练期间使用JVP，推理时不受影响；④理论分析局限于局部梯度与目标构造，未深入探讨全局收敛；⑤随机输入扰动与不同语言上下文对性能的影响未系统评估。

---

## 287. Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development

**arXiv ID:** 2610.13022 | [PDF](https://arxiv.org/pdf/2610.13022v1)

**作者:** Mingyu Chen `[一作]` (Chinese University of Hong Kong), Pinjia He `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了一个用于仓库级软件开发的交互式需求澄清基准（CedarBench），并基于此对多款大型语言模型（LLM）在多轮澄清和需求文档生成以及后续代码实现方面的表现进行了系统实验。

**💡 创新点**

创新点包括：①首次将交互式需求澄清引入仓库级任务；②使用来自真实对话数据的三种用户人设（专家、模糊请求者、需求变更者）模拟交互；③设计了四个细粒度评估指标（EQR、AvgClarified、BERTScore F1、ReqAlign）来衡量澄清有效性和文档质量；④揭示澄清效率与文档质量并非正相关、编码代理增强澄清但降低文档质量等非直观发现。

**🔧 技术方法**

技术手段包括：基于GPT-5.6-Sol的用户模拟器和LLM-as-a-Judge；多轮对话提示模板；LLM对需求文档的自动生成与评估；编码代理（Claude Code、Codex）与基线模型对比；自动化实验流水线与结果可复现的代码库。

**📊 数据集**

使用的数据集为：①来自GameCraft-Bench、MiniAppBench、WebGen-Bench的341个任务；②SWE-chat中的四类用户交互记录，用于构造三种人设；③每个任务对应的人工验证的需求文档和高层意图。

**📈 对比分析**

比较方法：在同一交互协议下，对六款LLM（GPT-5.6-Sol、Claude Opus 5、Gemini-3.1-Pro、GLM-5.3、DeepSeek-V4-Pro、Qwen3-235B-A22B）以及其对应的编码代理进行澄清有效性、需求文档质量和代码生成的评测；实验结果显示GPT-5.6-Sol在需求文档质量上领先（ReqAlign≈0.73），Gemini-3.1-Pro在澄清效率上最高（EQR≈0.33），编码代理提升澄清效率但在文档质量上下降（ReqAlign下降≈15-24%），并且交互澄清相较于直接生成需求或直接编码可显著提升代码生成性能。

**⚠️ 局限性**

局限性主要包括：①用户模拟器基于LLM，无法完全覆盖真实开发者多样化的表达与需求变更行为；②实验依赖固定提示和参数，可能影响模型表现；③需求文档的人工验证虽然严格，但仍可能存在主观偏差；④仅评估了三类任务类型，无法覆盖所有软件开发场景。

---

## 288. Les Misérables Dialogue: A film-based network representation of social communication

**arXiv ID:** 2610.13148 | [PDF](https://arxiv.org/pdf/2610.13148v1)

**作者:** Theodora Moldovan `[一作]` (Uppsala University), Davide Vega `[通讯]` (Uppsala University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过人工标注1998年电影版《悲惨世界》的对话，构建了包含说话者、被话者、接收者、文本内容和时间边界的时间文本网络，并将每条信息归入十维社会关系维度，进一步生成了多层网络。

**💡 创新点**

创新点在于：①提供了基于对话的细粒度时间文本网络，明确区分说话、被话和旁听三种角色；②利用十维社会维度构建多层网络，展示了不同社会交往类型的结构特征；③提出了一套可复制的方法，可用于从其他叙事媒体生成类似丰富的交流数据集。

**🔧 技术方法**

使用的技术包括：基于视听线索的手工标注、时间文本网络（TTN）建模、采用预训练分类器对文本进行十维社会维度打分、以及生成扁平化投影和多层投影网络。

**📊 数据集**

数据集来自1998年电影版《悲惨世界》，共收集到1,686条发言，随后发布于GitHub，供公开使用。

**📈 对比分析**

比较方法为计算网络结构指标（节点数、边数、平均度、聚类系数、度相关性、连通分量）并与传统的共现网络对照；结果显示基于被话者的网络更稀疏，基于接收者的网络更密集、更接近共现网络；多层网络维度各异，展示了社会关系的多样性。

**⚠️ 局限性**

局限性包括：只基于单一电影改编，可能无法完全覆盖原著情节；标注工作耗时且主观性较大；未能捕捉非语言表情和肢体语言的社会信息；十维维度可能不足以完整描述所有社会交往形式。

---

## 289. Structure-Aware Online Optimization of the Age of Information with Unknown System Statistics

**arXiv ID:** 2610.13142 | [PDF](https://arxiv.org/pdf/2610.13142v1)

**作者:** Yoshiaki Inoue `[一作]` (Osaka University), Liron Ravner `[通讯]` (University of Haifa)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于结构的自适应信息新鲜度优化框架，旨在优化更新生成速率，以减少信息延迟。

**💡 创新点**

创新点在于利用信息新鲜度梯度的解析表示，减少了在线学习问题的复杂性，避免了将系统视为黑箱的做法。

**🔧 技术方法**

使用了在线随机逼近算法，该算法联合估计信息新鲜度梯度中的未知量，并优化更新生成速率。

**📊 数据集**

在四个典型的不完全信息设置中进行了验证，包括具有未知服务特性的FCFS队列、部分观察的抢占式LCFS系统等。

**📈 对比分析**

与传统的黑箱学习方法相比，提出的方法在相同的观察预算下，优化误差显著更小，展示了利用排队结构的优势。

**⚠️ 局限性**

局限性在于该方法依赖于对系统统计特性的某些假设，且在动态变化的系统参数下的扩展性仍需进一步研究。

---

## 290. The Limits of Manipulation within Personalized Incentive Policies

**arXiv ID:** 2610.13133 | [PDF](https://arxiv.org/pdf/2610.13133v1)

**作者:** Andrea Araldo `[一作]` (Institut Polytechnique de Paris), Amir Brudner `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究个性化激励政策（PDMP）在面对能够通过模拟假设个人偏好来获取更高激励的操纵性代理人时的可行性与限制，构建了一个包含监管者学习过程的Stackelberg博弈模型，推导了代理人操纵收益与成本的上界，并通过对伦敦交通方式数据的蒙特卡洛仿真验证了现实的偏好学习机制能有效抑制操纵行为。

**💡 创新点**

创新点在于：①首次将监管者的偏好学习过程（基于历史行为的最大似然估计）显式加入模型，揭示了学习不确定性对操纵收益的抑制作用；②推导了操纵收益上界与操纵成本的解析表达式，并证明了在可辨识度降低时上界收紧，从而量化了“误报难度”对政策稳健性的提升；③通过对比瞬时偏好学习（InstPE）与现实偏好学习（RealPE）的仿真，展示了前者过于悲观，后者能显著提升政策有效性。

**🔧 技术方法**

主要技术包括：随机效用理论（Logit模型）来描述代理人行为；Stackelberg博弈框架来刻画监管者与代理人之间的先后行动；最大似然估计作为偏好学习方法；KL散度与Pinsker不等式用来量化学习误差对操纵收益的影响；以及Monte Carlo仿真与Bootstrap置信区间评估实验结果。

**📊 数据集**

使用了公开的伦敦交通模式选择数据集（London Passenger Mode Choice dataset），其中包含乘客在自行车、公共交通和私家车三种出行方式的选择记录。

**📈 对比分析**

通过在同一数据集上对InstPE和RealPE两种偏好学习机制进行同源仿真，比较了时间平均真实代理人效用、操纵成本与收益以及自行车出行比例的变化。结果显示，在RealPE下，代理人以诚实行为获得更高效用，操纵被抑制；而InstPE下则相反。政策效能的提升体现在自行车使用率的显著提升。

**⚠️ 局限性**

局限性包括：①模型假设代理人仅采用固定的假设偏好，未考虑更复杂的动态策略；②偏好学习采用最大似然估计，现实中监管者可能使用更复杂或受约束的学习方法；③实验仅基于单一城市交通数据，未验证跨地区或跨文化的普适性；④未考虑监管预算约束和政策执行成本，实际实施时可能需要进一步调整。

---

## 291. DP-MIND: Differentially private memory-guided Long context Inference

**arXiv ID:** 2610.13132 | [PDF](https://arxiv.org/pdf/2610.13132v1)

**作者:** Zhihao Liu `[一作]` (Zhejiang University), Zhan Qin `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了DP-MIND框架，用于在百万级长上下文下实现差分隐私的LLM推理。

**💡 创新点**

创新点在于将差分隐私机制与记忆增量推理相结合，利用强化学习优化记忆构建并在本地进行答案去噪，从而在极长上下文中显著提升隐私下的推理效果。

**🔧 技术方法**

采用了分块式DP扰动、记忆增量（compact memory）推理、DP-aware强化学习、局部答案去噪、Sparse Attention与Long-Context Scaling等技术。

**📊 数据集**

使用了RULER-HQA和NIAH两个长上下文评测集，覆盖7K到1.8M tokens的范围。

**📈 对比分析**

与Qwen2.5-7B/14B基础模型、Sparse Attention、Long-Context Scaling、Memory-Augmented等方法对比，在相同隐私预算下，DP-MIND在百万级上下文上维持高SubEM/EM分数，显著优于基线。

**⚠️ 局限性**

局限在于对记忆尺寸与chunk大小敏感，依赖可信本地环境，且在极高DP噪声下仍可能导致信息彻底丢失。

---

## 292. Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories

**arXiv ID:** 2610.13131 | [PDF](https://arxiv.org/pdf/2610.13131v1)

**作者:** Zhenhao Huang `[一作]` (North China Electric Power University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 Foresight 的路网轨迹生成框架，利用预期短期未来路径来指导下一步路口转向，实现生成过程的闭环递归。

**💡 创新点**

核心创新在于“Anticipatory Generative Modeling”（前瞻性生成建模），通过在每一步先预测有限长度的未来路径（anticipation），再将其作为上下文对合法转向进行评分，从而显著降低路口转向的不确定性。

**🔧 技术方法**

技术实现包括：<br>• 基于 Transformer 的 Future Anticipation Decoder，用于生成多步未来位置的概率分布；<br>• 方向感知 Fiber 状态编码路段几何信息；<br>• 拟合的 Transition Network 对合法转向进行打分；<br>• 结合候选条件重预期（re‑anticipation）与未来一致性（future‑consistency）来优化决策；<br>• 以交叉熵与期望未来一致性损失为目标进行端到端训练。

**📊 数据集**

使用公开 HOSER 数据集，包含波尔图（Porto）、北京（Beijing）和旧金山（San Francisco）三座城市的道路网络与轨迹数据。

**📈 对比分析**

在多项评价指标（Distance JSD、Radius JSD、Hausdorff Distance、Edit Distance on Real sequences）上与 Dijkstra、Yen、Markov、Semantic Routing、HOSER、TS‑TrajGen、GDP、Diff‑RNTraj、DiffTraj、TrajFlow、Seed 等基线进行对比，Foresight 在大多数城市中均取得显著更低的误差，表明其在轨迹空间匹配与路线路径保真度方面均优于现有方法。

**⚠️ 局限性**

局限性包括：<br>• 依赖静态道路图，未考虑实时交通信息或动态约束；<br>• 需要较大的训练数据与计算资源；<br>• 未来窗口长度 H 的设置会影响性能，过短可能无法捕获长程决策；<br>• 对于极端稀疏或新出现的路段，模型可能缺乏可靠的预期。

---

## 293. Characterizing CPU-GPU Memory Sharing over Unified Memory Architecture

**arXiv ID:** 2610.13125 | [PDF](https://arxiv.org/pdf/2610.13125v1)

**作者:** Jinrong Yang `[一作]` (Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性研究了统一内存架构（UMA）下CPU–GPU内存共享的正确性与性能，提出基于发送/接收阶段的分离方法，归纳了四项关键性能发现，并实现了跨平台自动选择实现的库libumsh，随后在九个真实系统中验证其加速效果。

**💡 创新点**

创新点包括：①将内存共享拆解为独立配置的发送和接收阶段，覆盖页面大小、缓存策略、共振与互连等硬件/软件维度；②系统地在四个平台上评估所有正确实现组合，揭示四条性能经验法则；③构建了libumsh库，能够根据应用需求、平台属性自动配置最优实现，并通过AI辅助端口工作流支持新UMA平台的快速扩展。

**🔧 技术方法**

使用的技术主要有：多平台硬件实验（Intel Raptor/Arrow/Lunar、NVIDIA Orin、DGX Spark）；微基准与完整正确性测试；透明大页（THP）、写回/未缓存策略、写直/写后刷、读直/读后失效、GPU‑CPU一致性（CC）控制；DMA零拷贝；Coherent GPU reads/writes；AI辅助的端口识别与配置工作流。

**📊 数据集**

评估数据集与案例包括：PyTorch模型加载、OpenCV图像预处理、OpenVINO GenAI输出采样、点云到稠密图像转换、EdgeNN混合推理、Accelerate与Demand Layering模型迁移、GPUfs文件系统、GoFS文件锁；Benchmarks 包含 TaskBench、LLM 生成（GLM‑4.7‑Flash）、Vision 与 NLP 工具的组合工作负载。

**📈 对比分析**

对比方法：在每个案例中与传统复制或默认驱动实现做差异化基准，测量数据传输时间、同步开销、整体端到端延迟；结果显示数据平面传输时间缩短 16–75%，控制平面同步开销降低 7–90%；例如在 PyTorch 统一内存加载中获得 75% 的加载加速，LLM 调用延迟降低 14% 以上。

**⚠️ 局限性**

局限性：① 依赖操作系统隐藏的页表与缓存策略配置，若驱动不暴露需手工补丁；② 仅覆盖一写或两向一致性的 UMA，未覆盖所有ISA 或更细粒度的加速器；③ 在高碎片或极大规模传输时 THP 与页面分配行为未做完整评估；④ 仍需针对每个平台手动验证实现，迁移成本不完全零。

---

## 294. DRYLAB-Bench: Benchmarking the Biosecurity Risk of Agentic LLM Mutation Design

**arXiv ID:** 2610.13116 | [PDF](https://arxiv.org/pdf/2610.13116v1)

**作者:** Yuzhen Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了名为DRYLAB-Bench的基准，评估大型语言模型（LLM）在执行高危生物突变设计任务时的实际风险，尤其在工具增强和迭代工作流中；

**💡 创新点**

创新点在于：①使用真实实验的深度突变测序（DMS）数据进行零风险、可扩展的验证；②构建了八维风险分类的真实危险场景；③通过多轮工具驱动的代理工作流程量化了生物设计工具对威胁的放大；

**🔧 技术方法**

技术包括：多轮工具集成（EVEscape、EVE、ProteinMPNN、AlphaMissense等）、基于实验数据的Biosecurity Risk Score（BRS）计算、LLM代理推理与迭代改进；

**📊 数据集**

数据集为ProteinGym与ViroGym的187,695个单点突变实验测量结果，以及由专家筛选的30个真实危险设计场景；

**📈 对比分析**

通过对15个前沿LLM（如GPT‑5.6、Claude Opus 5、Gemini 3.8 Flash等）在三种代理设置（S0、S1、S2）下的BRS表现进行对比，结果显示：零工具基线S0已达BRS≈0.66，工具+单轮S1提升至≈0.78，工具+多轮S2进一步提升至≈0.80，且高危变体比例从21.4%提升至23.5%；

**⚠️ 局限性**

局限性包括：①对未工具化迭代的评估不充分；②工具与任务对齐不完全，导致某些模型在特定场景下表现下降；③评估仅覆盖单点突变，未覆盖多点或结构重构；④BRS依赖现有实验数据，可能忽略未知突变的风险；

---

## 295. Safety Graph: When Safety Responses Become Attack Surfaces in Cyber-Physical Systems

**arXiv ID:** 2610.13086 | [PDF](https://arxiv.org/pdf/2610.13086v1)

**作者:** Rakesh Podder `[一作]` (Colorado State University), Indrajit Ray `[通讯]` (Colorado State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Safety Graph（SG）框架，将攻击图、故障树和事件树统一到共享命题空间，建模 IT 与 OT 之间双向级联，自动编译为 AI 规划问题，通过规划搜索发现跨层级攻击路径并给出硬化建议。

**💡 创新点**

创新点：1) 首次将安全响应视为可产生新攻击面的“安全响应诱发横向移动”；2) 通过 SG 的形式化编译实现对跨层级路径的自动推理；3) 结合事件树的时间顺序与安全手册，提供可操作的安全改进方案。

**🔧 技术方法**

技术手段：形式化建模（AG/FT/ET）、共享命题空间、PDDL 编译、Fast Downward（A* + LM-cut）规划搜索、LLM（llama3.8b）实体识别、Web3D+虚拟机模拟、Tennessee Eastman 过程仿真。

**📊 数据集**

数据集：Tennessee Eastman 过程工厂的网络拓扑、渗透测试结果、CTI 报告、IEC 61511/ISA‑84 安全手册，构建 AG、FT 与 ET 的命题集合。

**📈 对比分析**

比较方法：与传统的单向模型（RG/ERG）对比，证明 SG 能发现此前模型无法识别的关键故障路径；在 TEP 实例中规划时间从 0.8 s 到 1,592 s，内存低于 420 MB，支持约 5,000 步计划；在规模扩展实验中，规划时间随序列长度（seq）指数增长，其他参数线性影响。

**⚠️ 局限性**

局限性：1) 命题语义统一需人工专家校对；2) 事件树假设安全响应评估确定性，未覆盖随机或多条执行路径；3) 对无限循环级联（无界 seq）不做处理；4) 对动态漏洞更新需要重新生成 SG，缺乏增量重分析机制。

---

## 296. LongHarness: Advancing Agentic Reasoning over 100M-Token Corpora via Stateful Evidence Management

**arXiv ID:** 2610.13075 | [PDF](https://arxiv.org/pdf/2610.13075v1)

**作者:** Songtao Huang `[一作]` (Fudan University), Shiyang Feng `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种模型无关的智能体框架，利用证据生命周期管理和证据-结论共同进化实现语料库级长上下文推理。

**💡 创新点**

核心创新在于将证据与结论分离为持久状态，并通过未解决的结论引导检索、通过新证据修订结论，形成闭环；同时引入证据生命周期管理，保证证据的更新、撤回与版本追踪。

**🔧 技术方法**

技术包括基于外部环境交互的策略学习、Evidence–Conclusion Coevolution机制、Evidence Lifecycle Management模块、可持久化的源链接证据存储，以及将大型语言模型（GLM‑5.3‑Flash、Qwen3.8‑Flash）作为后端。

**📊 数据集**

评测数据集：OfficeQA Pro V1/V2、AA‑LCR、LongBench v2 Long 子集。

**📈 对比分析**

与直接推理、Codex、Claude Code、RLM 等基线对比，GLM‑5.3‑Flash 在 OfficeQA V1 上从 15.8% 提升到 60.2%，V2 从 4.4% 提升到 61.1%；在 AA‑LCR 和 LongBench 上分别达 84%/75.9% 等；相较最强基线提升 12%/18.9 个百分点。

**⚠️ 局限性**

局限性包括：依赖大型语言模型的推理能力，证据抽取与检索仍是瓶颈；实验仅在特定语料库和两个后端上验证，缺乏跨模型与跨任务的广泛验证；对极大规模文档集的实时性与存储开销尚未充分评估。

---

## 297. CAPE - Context-Aware Prioritization Engine for Vulnerability Scoring and Management

**arXiv ID:** 2610.13071 | [PDF](https://arxiv.org/pdf/2610.13071v1)

**作者:** Rakesh Podder `[一作]` (Colorado State University), Indrajit Ray `[通讯]` (Colorado State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了Context-Aware Prioritization Engine（CAPE）框架，对扫描得到的CVE进行部署上下文评估并生成可执行的优先级排名。

**💡 创新点**

首次将静态可达性分析、代码层级补丁匹配以及图论中心度指标结合到VEX式分数模型中，形成多维权重、分区适应的上下文感知漏洞优先级体系。

**🔧 技术方法**

使用SBOM（Syft）、漏洞扫描（Grype、Trivy）、AST分析（Tree‑sitter）、BFS可达性搜索、GitHub提交签名匹配、图论中心度（Hub、Eigenvector、影响半径）以及Analytic Hierarchy Process（AHP）权重分配。

**📊 数据集**

在30个开源项目（Python 18 项、TypeScript 11 项、JavaScript 1 项）中共收集了6138条扫描报告的CVE作为实验数据集。

**📈 对比分析**

与CVSS、EPSS、EPSS‑pct、KEV四个基线比较，CAPE在首次可利用CVE的平均排名降低了86%，NDCG@10提升至0.652（对比基线0.227），AUROC提升至0.908（对比基线0.454），显示显著的性能提升。

**⚠️ 局限性**

局限性包括仅覆盖动态语言（Python、JS/TS），未对编译语言或其他生态系统进行验证；缺乏真实攻击实验或开发者使用评估，且模型依赖SBOM生成的准确性。

---

## 298. TPCS: A Cycle-Level Simulation Paradigm with Native Execution Semantics for RTL Modeling and Generation

**arXiv ID:** 2610.13067 | [PDF](https://arxiv.org/pdf/2610.13067v1)

**作者:** Chengzhen Meng `[一作]` (Shandong University), Hongjun Dai `[通讯]` (Shandong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种名为TPCS的周期级仿真与RTL生成统一框架，支持在RTL实现前快速评估硬件设计。

**💡 创新点**

创新点在于把硬件控制流映射到宿主CPU的本地执行，利用事务剪枝天然省去事件调度，并开发专用编译器将行为描述转为结构对应的可综合RTL。

**🔧 技术方法**

采用C++原生执行、事务级函数调用、静态控制流调度、LLVM+CIRCT组合逻辑生成、谓词下沉优化以及多线程同步。

**📊 数据集**

实验数据集包括RISC‑V SMP系统、AES加密单元、Systolic‑Array，使用CoreMark、RISC‑V基准以及Full/Burst输入模式。

**📈 对比分析**

与Verilator、GSIM、Chisel、SystemC、PyMTL等工具比较，TPCS在单核/多核环境分别获得36–42倍、4.77倍、3.4–7.2倍加速，且生成的RTL时序和面积与手写实现相当。

**⚠️ 局限性**

局限性：仅覆盖同步周期设计，异步/跨时钟域支持不足；工具链成熟度有限，缺乏完善的调试、流水线自动化及高级可复用库。

---

## 299. A Ray-Traced Digital Twin for 5G NR Positioning with OpenAirInterface

**arXiv ID:** 2610.13138 | [PDF](https://arxiv.org/pdf/2610.13138v1)

**作者:** Nima Chaharbaghi `[一作]` (University of Bologna), Roberto Verdone `[通讯]` (University of Bologna)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建射线追踪数字孪生以驱动未改动的OpenAirInterface 5G UL-TDoA定位管线，并利用该孪生评估定位性能与阵列布局优化。

**💡 创新点**

将分布式天线视为独立射频设备以保留绝对时延；在数字孪生中保留每个天线的绝对延迟；通过孪生回答物理试验无法回答的问题，例如垂直精度受阵列几何限制，并通过高度多样化布局显著降低垂直误差。

**🔧 技术方法**

射线追踪（Sionna RT）、OpenAirInterface栈、数字孪生架构、FlatBuffers与nanomsg通信、SRS UL-TDoA定位算法、HDOP/VDOP分析等技术。

**📊 数据集**

使用EURECOM GEO-5G现场的天线坐标与场景数据（基于OpenStreetMap重建）以及Sionna提供的Munich密集城市模型，并用OAI的SRS数据进行定位。

**📈 对比分析**

在双场景（EURECOM、Munich）和双带宽（40 MHz、100 MHz）下对16个位置进行定位，水平平均误差分别为1.20/0.95 m（EURECOM）和2.62/1.04 m（Munich），90th百分位为1.80/2.32 m；宽带提升主要削减极端误差，垂直误差不受影响；高度多样化布局将RMS垂直误差从3.5 m降至0.85 m，水平误差略有提升。

**⚠️ 局限性**

仿真无噪声且样本有限；两种配置同时改变带宽、采样率、tap窗口，无法单独评估带宽影响；缺乏散射、互耦、天线模式、能量归一化等现实因素；未考虑移动、多普勒等动态效应；缺少对物理试验的严格对比。

---

## 300. How Anthropomorphic Attitudes Toward Agents Divide User Behavior, and How Design Should Change

**arXiv ID:** 2610.13156 | [PDF](https://arxiv.org/pdf/2610.13156v1)

**作者:** Tingyu Cao `[一作]` (Tsinghua University), Yang Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在2026年对一款桌面助手进行三个月的现场使用研究，探讨用户是否通过给助手改名表达拟人化倾向，并检验该倾向是否与对话行为差异相关，进而根据倾向与对话开场类型对回复策略进行分支调整；随后在产品内进行配对评判，验证调整是否受到目标用户群体的欢迎且不影响任务效率。

**💡 创新点**

创新点在于：① 采用用户在设置中自行改名这一低成本、实时可得的信号作为“拟人化立场”变量，避免问卷与事后编码；② 通过稀疏自动编码器生成共享的128维行为代码书，客观捕捉对话中可组合的行为模式；③ 发现并量化开场类型（问候、倾诉、任务）在不同立场群体中的差异，并将此差异直接映射到运行时回复策略的分支；④ 在产品内对比评判展示了基于立场与开场分支的回复相较传统统一策略在被视为人、情感回应和任务推进等维度上的优势。

**🔧 技术方法**

使用技术包括：用户行为稀疏自动编码器（L1稀疏目标，ReLU激活，Adam优化），Spearman相关与置换检验，Benjamini–Hochberg多重校正，词典式开场行为分类器，及基于标签的身份属性关联分析。

**📊 数据集**

数据集为257名注册用户在2026年5月12日至8月12日期间产生的798次会话、4,943个用户问题、5,422个助手回复（共39,191步），以及用户在设置中记录的身份标签与改名信息。

**📈 对比分析**

方法对比：对比了原统一回复策略与改名+开场分支策略。通过1,744次配对评判（129名用户）评估四个维度：被视为人、情感回应、任务干扰、持续对话意愿。结果显示，改名用户在前三个维度显著倾向于分支策略（P<0.001，Cramér V≈0.24–0.26），而任务干扰维度两组无显著差异；在评估设计效应下，前三个维度仍显著。

**⚠️ 局限性**

限制包括：① 改名仅作为拟人化立场粗略标识，未捕捉更细致的语用偏好；② 关联性实验无法确定因果，缺乏随机化；③ 仅在单一桌面助手与英语环境下验证，跨产品或跨语言可推广性未知；④ 评判样本相对有限，且多由高频用户贡献，可能偏倚；⑤ 稀疏自动编码器的具体词汇表与潜在语义可能因随机种子变化而不同，导致结果复现性受限。

---

## 301. Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories for Supervised Fine-Tuning

**arXiv ID:** 2610.13152 | [PDF](https://arxiv.org/pdf/2610.13152v1)

**作者:** Sida Li `[一作]` (Peking University), Xiaohui Li `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了基于事件日志的轨迹质量评分框架Trajectory Quality Score（TQS），用以筛选编程代理的成功轨迹；

**💡 创新点**

创新点在于通过七个线性可计算的结构化指标（信息聚焦、意图-动作一致、策略动态及清洁模式指示器）对轨迹进行收敛性评估，既避免了高成本LLM评判，又能捕捉过程级质量；

**🔧 技术方法**

技术包括事件日志解析、七维结构化度量、加权平均组合、LoRA微调Qwen3-Coder-30B-A3B-Instruct模型，并与GPT‑4o等LLM评估做对比；

**📊 数据集**

使用公开的OpenHands轨迹库155,144条成功轨迹（共258k条，51k任务、209仓库）以及SWE-Gym、SWE-Smith等源；

**📈 对比分析**

通过构造Top、Bottom、Random、长度匹配等子集进行SFT，并在pass@1评估。Top集提升至27.6（相对基线24.4），优于随机与低质量子集，并在难度较高的任务上收益更显著；与GPT‑4o评估相近但成本低；

**⚠️ 局限性**

局限性包括：仅在单一教师模型和同一任务空间实验；组件权重手工设定，跨模型/多语言/多模态可能需重调；仅评估固定预算1k轨迹，未考虑长轨迹；未对更大模型验证；未做多种种子方差评估。

---

## 302. VIGOR: Verifiable Industrial Geometry Optimization and Repair in Scene Programs

**arXiv ID:** 2610.13145 | [PDF](https://arxiv.org/pdf/2610.13145v1)

**作者:** Fuqiang Wang `[一作]` (Qilu University of Technology), Jingxuan Wei `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了工业场景程序生成中的可执行性-工程可行性差距，构建了执行归一化基准数据集并提出了VIGOR方法。

**💡 创新点**

通过执行反馈验证与分阶段局部投影及冲突图联合修正，有效弥合可执行性与工程可行性之间的差距。

**🔧 技术方法**

采用监督式场景程序生成、执行归一化验证、局部投影与冲突图联合修复，使用LoRA微调的Qwen3‑VL‑8B‑Instruct模型。

**📊 数据集**

使用1,749个可构造工业场景的执行归一化数据集，每个场景包含25步程序和94个几何检查。

**📈 对比分析**

与多种开源、专有大型语言模型及代码/布局生成方法对比，VIGOR在完整可行性、几何质量与整体评分上分别达到96.86%、99.51%和91.20，领先基线约10.42个百分点。

**⚠️ 局限性**

仅适用于固定资产库存与结构化程序，尚未验证能否扩展到更大规模、多样化资产或动态约束的实际工业环境。

---

## 303. Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA

**arXiv ID:** 2610.13128 | [PDF](https://arxiv.org/pdf/2610.13128v1)

**作者:** Wenlin Lin `[一作]` (Metask Lab), Fanzhe Wei `[通讯]` (Metask Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基于“job分离”的文档问答管道，包含知识构建、Locator、Bridge judge、Typed evidence pool、Deep Reader和Synthesizer六个独立工作单元，并通过多轮检索+阅读实现跨文档推理。

**💡 创新点**

创新点在于将检索与阅读物理上分离成不共享工具的代理，并通过代码级桥梁强制二跳；使用类型化、大小限制的证据池合并检索结果，显著提升多文档QA性能。

**🔧 技术方法**

技术手段包括：混合检索（BM25+dense HNSW+图扩展）、OCR+VLM生成图表描述、知识点图结构、工具驱动多轮Agent、Typed evidence pool、Deep Reader读取文本/图像/表格、答案生成与格式检查。

**📊 数据集**

使用的数据集有：M3DocVQA（开放域多文档QA）、MMLongBench-Doc（单文档长文档QA，含表格图表）和MultiHop-RAG（多文档推理QA）。

**📈 对比分析**

通过与已发布系统（如M3DocRAG、HiKEY、GraphRAG等）以及单轮检索+回答基线对比，M3DocVQA上F1从36.5提升至47.5（+11.0），MultiHop-RAG准确率73.5/71.2，MMLongBench-Doc从34.8提升至47.0；同时提供完整可复现索引和代码实现。

**⚠️ 局限性**

局限性包括：仅适用于能够遵循多轮工具协议的较大模型；低端模型（如7B）未见提升；跨页表格/图表推理仍有不足；单机延迟较高；图层扩展在本实验中对性能影响有限。

---

## 304. Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination

**arXiv ID:** 2610.13112 | [PDF](https://arxiv.org/pdf/2610.13112v1)

**作者:** Mohammad Khalooei `[一作]` (Sharif University of Technology), Mohammad Sabokrou `[通讯]` (New Uzbekistan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种黑盒因果检验方法，通过在CLAP定位的频谱区段做谱门控来检测大型音频-语言模型的幻觉并用于解码纠正。

**💡 创新点**

创新点在于利用CLAP定位后局部去除音频实现因果必要性检验，既不需要梯度也不需训练，并在解码阶段通过语言先验与局部对比实现性能提升。

**🔧 技术方法**

使用CLAP相似度定位、短时傅里叶变换谱门控、概率必要性与充分性测试、对比解码（AAD、LangPrior、CAUSA）以及多模型评估。

**📊 数据集**

数据集为Audio-Hallucination_Object-Existence_AudioCaps-ESC50-VocalSound benchmark（约10.8k条）。

**📈 对比分析**

与传统AAD基线相比，局部去除在消除幻觉上表现相当但需控制持续时间；结合语言先验的对比解码可将Qwen2-Audio准确率从57.1%提升至71.9%。

**⚠️ 局限性**

局部定位对某些模型（如Qwen2-Audio）受持续时间混淆，且跨模型的因果特征不易迁移，局部检验在短音频上效果减弱。

---

## 305. Measurable Semantic 3D Gaussian Splatting with Joint Region and Depth Supervision for Interactive Building Inspection

**arXiv ID:** 2610.13143 | [PDF](https://arxiv.org/pdf/2610.13143v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Neurograf v1: An End-to-End Romanian-First Language Model Family

**arXiv ID:** 2610.13134 | [PDF](https://arxiv.org/pdf/2610.13134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 307. ReAT: Residual Action Tokenization for Autoregressive Robot Policies

**arXiv ID:** 2610.13093 | [PDF](https://arxiv.org/pdf/2610.13093v1)

**作者:** Dijia Zhan `[一作]` (South China University of Technology), Jie Tang `[通讯]` (South China University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种残差动作分词（Residual Action Tokenization, ReAT），通过将已解码动作的残差映射为离散token，并用递归解码器将每个token作为对当前草稿的纠正，形成一系列逐步改进的动作预测；

**💡 创新点**

创新点在于：①明确每个token对应对已解码动作的残差修正，并将解码反馈回编码器；②采用前缀监督训练，让任意非空前缀均可完整重构动作，从而在同一policy下支持不同token预算；③使用draft-conditioned Transformer解码器，使同一token在不同草稿上产生不同纠正。

**🔧 技术方法**

使用技术包括：Transformer编码器+有限精度量化（FSQ）生成token；递归draft-conditioned解码器进行纠正；前缀监督与进度损失的联合训练；teacher‑forcing训练的自回归策略网络；以及对实验进行多任务训练与评估。

**📊 数据集**

主要使用的数据集有：LIBERO‑10、MetaWorld、RoboCasa三个仿真基准，以及真实世界的桶堆叠实验（X‑Trainer 机械臂）。

**📈 对比分析**

与OAT、QueST、FAST、Bin、Diffusion Policy等基线比较。ReAT在所有三大仿真基准上，8‑token配置均优于基线；在LIBERO上，4‑token配置还超过8‑token基线。真实桶堆叠实验中，8‑token ReAT成功率达到70%，显著高于OAT（40%）和QueST（35%）。

**⚠️ 局限性**

限制包括：需要手动指定token预算；未实现学习式停止或异步细化；解码过程未严格保证运动安全；仅在单一真实任务上验证，未覆盖更多硬件平台；以及对解码错误的处理仍不完善。

---

## 308. Double-Take: Improving Capacity, Binding, and Selectivity In Long-Context Within Bidirectional Memory

**arXiv ID:** 2610.13073 | [PDF](https://arxiv.org/pdf/2610.13073v1)

**作者:** Vakul Raghavan `[一作]` (Rutgers University), Xiaomin Lin `[通讯]` (University of South Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种称为 Double‑Take (DT) 的快速权重记忆机制，以解决长序列处理中的记忆容量、跨位置绑定和选择性存储问题。

**💡 创新点**

创新点在于：①多步非完整更新提升容量；②局部因果卷积实现跨位置绑定；③双向变化敏感写入实现选择性保留。

**🔧 技术方法**

采用快速权重记忆（fast‑weight memory）、大块测试时训练（Large‑Chunk TTT）、多头注意力、局部卷积和双向摘要写入。

**📊 数据集**

主要使用人类基因组（GRCh38）进行预训练，随后在基因组上下文长度从2 kb到1 Mb 的 TAD 边界判定、eQTL 效应预测以及跨物种（人–鼠–猿类–鸡）零样本迁移等基准测试。

**📈 对比分析**

通过控制实验验证了容量、绑定和选择性三个子机制；在实际基因组任务中，DT 在 32 kb 训练长度外仍持续提升，最大 AUROC 约 0.93；在跨物种零样本迁移中，AUROC 在 0.78–0.86 范围内保持稳健，优于仅使用传统记忆或公开的长上下文模型。

**⚠️ 局限性**

局限性：仅在基因组序列上验证；未考虑转录组、表观遗传或蛋白质监督；与部分最新 TTT 方法的对比受限于实现可复现性；跨模态和非生物任务的通用性尚待验证。

---

## 309. Volume Rendering with Metric--Phase Fields for Thin Structures and Sign-Ambiguous Geometry

**arXiv ID:** 2610.13088 | [PDF](https://arxiv.org/pdf/2610.13088v1)

**作者:** Junkai Deng `[一作]` (Nanyang Technological University), Ying He `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Metric–Phase Fields（MPF）的神经体积渲染框架，用于从姿势已知的多视角图像中重建具有薄结构、开放边界或非流形接合的签名不确定几何体。

**💡 创新点**

创新点包括：①将几何拆分为无符号距离度量场和连续相位场，并通过残差相位注入形成稳定的签名隐式函数；②设计针对MPF的专用正则化损失（Eikonal约束、相位对齐、anchor、free‑space抑制等），解决在仅有光照监督下的尺度/梯度不确定性；③直接使用标准Marching Cubes进行表面提取，避免UDF难以提取的两阶段流程。

**🔧 技术方法**

技术手段：神经隐式曲面、体积渲染、MPF几何网络与视角相关辐射网络、残差相位注入、MPF专用损失、NeuS风格的占用映射、三维体素/哈希编码等。

**📊 数据集**

实验数据集包括DTU、Tanks & Temples、DeepFashion3D、以及作者自建的Thin‑Structure‑and‑Sign‑Ambiguous（TSSA）和TAAS（真实/合成）数据集，用于评估薄结构、开放表面和非流形几何的重建质量。

**📈 对比分析**

与SDF基方法（NeuS、VolSDF、HF‑NeuS、Neuralangelo）、UDF基方法（GaussianUDF、NeUDF、2S‑UDF）以及3D Gaussian Splatting（PGSR、GGGS、GeoSVR）等进行比较；在薄结构和签名不确定场景中，MPF方法在Chamfer距离和法向一致性上均优于基线，并在新视角合成中仅略低于3DGS方法，显示出更高的几何可靠性。

**⚠️ 局限性**

局限性包括：训练耗时较长（单场景 >10 小时），对高光/阴影等颜色变化敏感导致表面细节失真；在仅光照监督下仍需大量正则化；与3DGS在渲染质量上略有差距；若不结合显式空间结构或数值梯度优化，仍难以进一步加速。

---

## 310. PCM: Programmable Congestion Management for Terabit RDMA NICs

**arXiv ID:** 2610.13130 | [PDF](https://arxiv.org/pdf/2610.13130v1)

**作者:** Mikhail Khalilov `[一作]` (ETH Zürich), Torsten Hoefler `[通讯]` (ETH Zürich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出一种名为PCM（Programmable Congestion Management）的编程模型，能够在不同 RDMA NIC 平台上无源代码改动地实现多种拥塞控制与负载均衡算法，并在模拟器、软件 RDMA 传输以及两款商用 NIC（NVIDIA ConnectX‑8、Microsoft 1P）上验证其可移植性与性能；

**💡 创新点**

创新点在于将拥塞管理策略与平台集成分离，提供统一的信号、触发器、控制和私有状态抽象；通过自动生成“shell”代码将策略映射到各平台的原生接口，从而实现算法源代码在不同硬件间的复用；此外，还实现了针对高带宽（400 G/800 G）NIC 的高效、近实时事件处理与累积反馈机制；

**🔧 技术方法**

采用事件驱动的 C 语言编写策略，使用 PCM 规范定义信号、触发器、控制；通过编译器与平台后端自动生成 shell，使用固定点/浮点数运算、累加器、原子操作等硬件友好实现；在软件层面使用 UET‑htsim、UCCL、DOCA PCC、Microsoft 1P 等平台；

**📊 数据集**

使用的“数据集”主要包括：UET‑htsim 的仿真事件流（incast、collectives、存储等工作负载）、UCCL 的 64 KiB RDMA Write 流量、ConnectX‑8 的 DPA 事件回放、Microsoft 1P 的 RDMA Write 流量以及自定义的合成事件混合（ACK、ECN、丢包等）;

**📈 对比分析**

比较方法：将 PCM 实现的 11 种常见拥塞控制/负载均衡算法（如 Swift、CUBIC、DCTCP、MP‑RDMA 等）与各平台的原生实现（UCCL 内置、ConnectX‑8 DOCA PCC 内置、Microsoft 1P 原生）进行性能对标；通过测量包处理速率、处理延迟、好传输率、流完成时间等指标，发现 PCM 在 UCCL 上可达 1.56 Tbit/s 好传输率；在 ConnectX‑8 上实现了接近 per‑ACK 级别的事件处理；在 Microsoft 1P 上实现了 1.56 Tbit/s 同样的好传输率；整体抽象开销在 1.5–2.0 倍之间，已满足大多数高性能需求；

**⚠️ 局限性**

局限性包括：需要为每个平台实现专门的后端“shell”，否则无法直接复用；策略需要在每个目标平台上进行数值与时间调优，尤其是固定点与浮点差异；在高负载时，事件聚合与触发器延迟可能导致公平性下降；当前仅支持有限的信号与控制类型，未来需扩展至更丰富的度量与操作；

---

## 311. Replaying the Movie in Your Mind Decoding Dynamic Visual Perception and Recall from EEG

**arXiv ID:** 2610.13120 | [PDF](https://arxiv.org/pdf/2610.13120v1)

**作者:** Zikai Fang `[一作]` (Shanghai Jiao Tong University), Wei-Long Zheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

创建了 PAIR EEG 基准，记录 20 位受试者在观看自然视频后立即回忆的 EEG，设计了 clip‑wise 交替的感知‑回忆实验。

**💡 创新点**

首次将动态自然视频的感知与内部回忆在相同时序和内容下直接比较，提供高时间分辨率的 EEG 基准。

**🔧 技术方法**

使用了多种深度学习模型（DeepNet、Conformer、GCN‑LSTM、EEGNet 等）和传统 SVM 以及 PSD/DE 频谱特征。

**📊 数据集**

使用自制的 250 条 2 秒自然视频片段，分为 4 大类共 20 细类，记录 62 通道 EEG。

**📈 对比分析**

通过在各类语义、元特征和状态分类任务上做在受试者内部验证，GCN‑LSTM 在回忆任务上获得最高平均准确率，感知‑回忆状态可分辨准确率>90%，但多任务间差异有限。

**⚠️ 局限性**

仅内部验证导致潜在过拟合，缺乏跨受试者泛化，受试者回忆时长不同，未完全排除眼动及任务负荷影响，数据仅包含 EEG 未包含眼动记录。

---

## 312. Continually Post-training Small Models via Synthetic Self-distillation

**arXiv ID:** 2610.13113 | [PDF](https://arxiv.org/pdf/2610.13113v1)

**作者:** Woosung Koh `[一作]` (KAIST AI), Moontae Lee `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SynthSelf，用于在小型语言模型上进行持续后训练，注入新能力并保持旧能力。

**💡 创新点**

创新点在于将同族更大模型作为离线对齐器，利用离线分布内重写和拒绝采样，提升监督质量并保持分布一致。

**🔧 技术方法**

采用离线对齐器重写、拒绝采样、离线 SFT（自监督式）以及对比实验。

**📊 数据集**

使用科学（SciKnowEval Chemistry）、工具使用（ToolAlpaca）和医学（MedMCQA）等数据集，及四科目的混合实验。

**📈 对比分析**

与标准 SFT、SDFT、DFT、ASFT 以及离线/在线蒸馏等基线对比，SynthSelf 在所有四个任务域的获取性能最高，保留性能与最优基线相当，平均提升 5.7–16.9% 的获取分数。

**⚠️ 局限性**

局限性：需要同族更大模型作为对齐器；在医学域增益有限；对极小模型的适用性待验证。

---

## 313. Scaling RNA Foundation Models through Iterative Refinement

**arXiv ID:** 2610.13106 | [PDF](https://arxiv.org/pdf/2610.13106v1)

**作者:** Pengan Chen `[一作]` (Chinese University of Hong Kong), Yu Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Eco‑RNA，一个80M参数的RNA基础模型，能够通过可调的迭代深度实现内部推理规模扩展。

**💡 创新点**

创新点包括：① 通过共享Transformer核心实现可变计算深度；② 在自监督预训练中引入单向重构锚定（one‑sided reconstruction anchor）与终端位移正则化；③ 在单一模型中提供多种计算预算与性能平衡点。

**🔧 技术方法**

技术手段：输入条件化共享Transformer更新、循环深度推理、掩码重构预训练、相对终端位移正则化、单向重构锚定、RMS归一化与组查询注意力。

**📊 数据集**

使用的数据集包括：OpenGenome2 与 RNAcentral 进行无标签预训练；RNAGym 70‑assay 用于 RNA 功能预测；Ribonanza 用于化学探针反应预测；bpRNA 与 nRC benchmark 用于二级结构与家族分类评估。

**📈 对比分析**

与SOTA Evo‑2 40B、RNA‑MSM、RNA‑FM、RiNALMo等模型对比，Eco‑RNA 在RNAGym的绝对Spearman相关率从0.242提升到0.286（四轮迭代），并在三项综合指标上超过Evo‑2，参数量仅500倍少；在家族分类、二级结构、化学探针预测上也取得最高或接近最高分数。

**⚠️ 局限性**

局限性：对粗粒度任务（如家族分类）迭代深度收益有限；深度提升会显著增加前向FLOPs；需要针对不同任务手动设定最佳迭代次数；目前未探索更大模型规模或更复杂的多任务训练。

---

## 314. Robust Vision-Language Navigation with Awareness of Instruction-Execution Progress

**arXiv ID:** 2610.13104 | [PDF](https://arxiv.org/pdf/2610.13104v1)

**作者:** Jiagan Huang `[一作]` (Harbin Institute of Technology), Wenjie Pei `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种将指令执行进度估计与导航决策耦合的视觉-语言导航框架（VLN‑AIEP），并通过锚定关键地标的轨迹表示来聚焦待执行指令内容，从而实现无需显式进度标注的进度学习。

**💡 创新点**

创新点包括：①将进度估计与决策直接耦合，使得进度由导航标签间接监督；②构造地标锚定的轨迹表示，将语义与相对几何信息融合；③引入进度感知细粒度匹配（PAM）和双向注意机制（I2T、T2I）来抑制已执行指令内容。

**🔧 技术方法**

采用的技术包括 Transformer 编码器、跨注意力层、I2T/T2I 双分支注意、地标锚定 MLM、方向感知特征、YOLOE 目标检测、DINOv2 视觉编码、ALBEF 或 mRoBERTa 文本编码以及基于图的探索与历史轨迹管理。

**📊 数据集**

实验使用 R2R、RxR、CVDN 以及 R2R‑CE 四个公开基准数据集。

**📈 对比分析**

与现有方法（如 PRET、ETPNav、NaviLLM 等）比较，VLN‑AIEP 在 R2R‑val‑unseen、RxR‑val‑unseen、CVDN‑val‑unseen 以及 R2R‑CE‑val‑unseen 上分别提升了 SR/SPL/GP/OSR 指标，例如 R2R‑val‑unseen SR 74%（+2.6%）、RxR‑val‑unseen SR 74.5%（+3.4%）等。

**⚠️ 局限性**

局限性包括对地标检测质量依赖较大；在连续导航场景中仍落后于大规模 LLM 基础方法；在极端复杂或视觉噪声较多的环境下进度估计误差可能影响性能；模型参数规模虽低于 LLM，但仍不如最优图搜索策略。

---

## 315. VETR: Execution-Grounded Self-Improving Agent for Symbolic Graphics Reasoning

**arXiv ID:** 2610.13098 | [PDF](https://arxiv.org/pdf/2610.13098v1)

**作者:** Yingbo Ma `[一作]` (New York University), Hua Shen `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 VETR 框架，通过执行轨迹证据驱动符号图形编辑的递归自我改进。

**💡 创新点**

创新点：将程序执行的写入记录转化为受约束的备选程序，先在运行时验证保持性，再由视觉验证引导后续迭代，从而突破仅凭渲染反馈的局限。

**🔧 技术方法**

技术：执行记录收集、写入恢复与投影、约束检查与补丁编译、递归 propose–execute–observe–verify 循环、LLM 生成、视觉验证（CLIP、MSE、PSNR、SSIM、LPIPS 等）。

**📊 数据集**

数据集：BlenderGym（280 个编辑任务）和 BlenderBench（27 个发布任务）。

**📈 对比分析**

比较方法：在同一 LLM 基础上与 One‑Shot、BlenderAlchemy、VIGA 等基线对比，使用 CLIP 相似度、MSE、PSNR、SSIM、LPIPS 等指标；VETR 在两种大模型上均显著优于基线，误差降低 30–70%。

**⚠️ 局限性**

局限性：仅验证已捕获的属性，扩展到其它领域需定义写入语法；运行时验证不保证视觉目标，仍需视觉验证；多步修复可能增加计算开销。

---

## 316. Detection and Construction of Small Blocking Sets via Critical Independent Sets and Matching Decomposition

**arXiv ID:** 2610.13096 | [PDF](https://arxiv.org/pdf/2610.13096v1)

**作者:** Quanlong Niu `[一作]` (Fuyang University of Technology), Riheng Jia `[通讯]` (Zhejiang Normal University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在连通无向简单图中给出了一种多项式时间算法，用于检测并构造小阻塞集（满足|S|=|N(S)|且|S|<|V|/2的非空独立集）。

**💡 创新点**

通过引入最大阶数临界独立集与投影定理，将全图问题转化为在一个固定独立集内寻找紧致子集；随后利用完美匹配与强连通分量的匹配收缩，进一步把检测问题简化为辅助有向图的强连通性判断。

**🔧 技术方法**

关键技术包括：临界独立集的多项式求解、完美匹配与邻域计数的投影定理、匹配收缩构造辅助有向图、以及有向图的强连通分量算法。

**📊 数据集**

本工作为理论算法研究，无涉及具体数据集；所有证明均在一般图上成立。

**📈 对比分析**

与之前仅能枚举子集的指数级方法相比，提出的算法复杂度为O(m²)（m为边数），在理论上大幅降低了计算成本；该算法已通过完整的算法分析证明正确性。

**⚠️ 局限性**

局限性在于：1）仍采用多项式阶的O(m²)时间，对超大规模图可能不可行；2）缺乏实验评估与实际应用场景验证；3）仅解决了阻塞集检测与构造问题，未进一步探讨在具体时序图探索调度中的应用。

---

## 317. When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows

**arXiv ID:** 2610.13092 | [PDF](https://arxiv.org/pdf/2610.13092v1)

**作者:** Yuchen Ling `[一作]` (Nanjing University), Chunrong Fang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 ALFWorld 和 AndroidWorld 两大基准上，对五类 AI 代理集成 Jev（快速低成本决策模型）进行实验，评估其对任务成功率、决策成本与延迟以及整体任务执行资源的影响。

**💡 创新点**

通过系统评估决策级别效率向任务级效率的转化，揭示 Jev 在不同工作流中的收益差异，并指出在设计决策接口和恢复机制时需要关注的关键点。

**🔧 技术方法**

主要技术是将 Jev 的 typed‑choice 接口嵌入 LLM 代理的行动选择阶段，并配合现有 LLM（如 GPT‑4‑0613）进行辅助推理、反思与生成；实验使用 OpenAI API 与 Jev API。

**📊 数据集**

使用 ALFWorld（134 任务）和 AndroidWorld（116 任务类型）两个基准的文本观察版本，共计 4,500 次任务执行。

**📈 对比分析**

方法：对每个代理配置（Baseline、Jev、LLM‑Choice）分别执行三次，收集任务成功率、决策与任务级代价、耗时以及每成功任务的资源消耗；采用配对 bootstrap 统计差异。结果显示，Jev 在所有代理中显著降低决策成本（约 70%）和延迟（约 80%），但任务级成本与时间收益因工作流不同而差异显著，部分移动代理的成功率下降。

**⚠️ 局限性**

局限：实验仅覆盖文本观察的嵌入式和移动任务；未评估视觉感知、长序列或开放式工具创建；仅在固定任务集和种子上重复三次，未检验跨环境泛化；成本与延迟受服务商计价与网络条件影响；Jev 的效果取决于代理整体工作流设计，难以完全归因。

---

## 318. Belief-Overlap Partial Inversion: Object-Centric Flow Replanning for Humanoid Loco-Manipulation

**arXiv ID:** 2610.13087 | [PDF](https://arxiv.org/pdf/2610.13087v1)

**作者:** Chen Yu `[一作]` (Northwestern University), Jiuguang Wang `[通讯]` (Multiply Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个面向人形机器人推拉车的对象中心化闭环框架，该框架将在线估计、基于流的规划和全身控制结合起来，提出了基于信念重叠的部分逆流（BOPI）规划重构方法，实现了在推拉过程中根据实时估计动态更新行进轨迹。

**💡 创新点**

创新点在于：① 用信念重叠（Bhattacharyya系数）直接决定计划重构的深度，实现“部分逆流”而非全量重规划；② 在流模型中利用已学习的逆向流重构路径，避免加入噪声导致的轨迹不连贯；③ 将对象特性（质量、后轮转动刚度）作为规划条件，并通过递归估计器在线更新。

**🔧 技术方法**

采用的技术包括：GRU 集合递归估计器输出质量与后轮刚度的均值/不确定度；条件流模型（时间U-Net+FiLM）在给定信念与目标下生成时序轨迹；全身控制器用 PPO 训练，输入对象轨迹与机器人观测，输出关节目标；BOPI 通过计算旧信念与新信念的 Bhattacharyya 系数来决定流时间逆向积分点，随后在新信念下正向重构。

**📊 数据集**

使用的实验数据集：① 在 G1–Dex3 模拟环境中生成 18 个不同目标（平移、旋转、混合及错误质量情形），每个目标 32 条随机种子；② 对 Duffing 摆的控制问题做 256 条试验；③ 真实机器人在一台非全向购物车上进行的平移与旋转实验（5 条试验/方法/目标）。

**📈 对比分析**

与四种基线（固定深度 RDM、FlowEdit、Fresh FM、Retain）对比。实验结果显示：在模拟目标中，BOPI 的最终位置误差和偏航误差分别比 RDM 降低约 33% 与 47%；在 Duffing 任务中，成功率从 2.4% 提升至 97.2%，并显著减少累计计划变更量；在真实机器人中，BOPI 的定位 RMS、偏航 RMS 与最终误差均优于所有基线，尤其在平移任务中定位误差仅为 5.78 cm（相比 RDM 的 8.59 cm）。

**⚠️ 局限性**

局限性：① 仅在二维平面 SE(2) 目标与车轮动力学下验证，未扩展到三维目标或非轮式物体；② 依赖于已训练好的流模型和估计器，对极端或未见过的物理参数可能不稳健；③ 计算量相对较大，实时性依赖硬件性能；④ 对外部观测噪声和卡车姿态估计误差敏感。

---

## 319. The Limits of Query Monitoring against Distributed Model Extraction

**arXiv ID:** 2610.13083 | [PDF](https://arxiv.org/pdf/2610.13083v1)

**作者:** Snigdha Chandan Khilar `[一作]` `[通讯]` (Independent Researcher), Snigdha Chandan Khilar (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种仅基于查询内容的监控框架，用于检测分散式模型蒸馏攻击

**💡 创新点**

提出“良性协调屏障”理论并引入 hollowed 矩阵与 multi‑outlier 多重异常选择策略，使得监控对自然文本填充和慢速上升攻击具有更高鲁棒性

**🔧 技术方法**

使用句子嵌入（MiniLM/E5）、随机傅里叶特征、白化、特征向量化、相关矩阵特征值分解、两尺度分数、同态校准（conformal）以及多种基线检测（MMD、global stream、per‑account、Fraudar 等）

**📊 数据集**

基于 WildChat 1M 与 WildChat 4.8M 公共聊天记录（98k+ 哈希IP 账户）以及合成蒸馏提示来源 MetaMathQA、Magicoder、Alpaca、GSM8K 等

**📈 对比分析**

与多种基线对比，开发期内两尺度检测捕获 86–100% 典型蒸馏；hollowed 选择在自然填充下仍达 50–96%；在 17 周未接触评估中 false alarm 约 1–2%，检测率 0.86–1.00；表现优于全局流、MMD、per‑account 等传统方法

**⚠️ 局限性**

仅在单一句子嵌入空间实验，未考虑多模态、时序或写作风格特征；低协同率（≤2.5%）时检测失效；无法区分转发者型攻击；慢速上升攻击能破坏校准；数据来自消费者聊天，未覆盖真实 API 流量

---

## 320. Towards Safer Cities: A Design Process for Perception Datasets in Urban Driving Scenarios

**arXiv ID:** 2610.13077 | [PDF](https://arxiv.org/pdf/2610.13077v1)

**作者:** Alexander Kroth `[一作]` (Technische Universität Darmstadt), Steven Peters `[通讯]` (Technische Universität Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过分析现有汽车与轨道交通数据集及OpenStreetMap地图，识别城市中重型车与有轨电车感知数据的缺口，并提出针对性的数据采集设计流程。

**💡 创新点**

将操作设计域（ODD）与代表性用例相结合，并利用公开地图数据系统化规划采集场景，首次为重型车/有轨电车提供量化的采集点建议。

**🔧 技术方法**

采用地图数据挖掘、ODD定义与场景提取技术，辅以OpenStreetMap信息来定位关键基础设施与交互场景。

**📊 数据集**

分析现有汽车感知数据集（如KITTI、Waymo等）与轨道交通数据，结合用户生成的OpenStreetMap地图。

**📈 对比分析**

未进行算法对比或性能评估，仅通过定量统计（如33个可通行环岛、49个电车穿越点）来验证采集规划的可行性。

**⚠️ 局限性**

缺乏实际采集与验证，方法仅在理论层面；未覆盖所有重型车辆类型；仅依赖OSM数据，可能忽略未标注的细节；未给出感知模型的效果评估。

---

## 321. Competence-Aware Orchestration for LLM-Based Multi-Agent Systems

**arXiv ID:** 2610.13144 | [PDF](https://arxiv.org/pdf/2610.13144v1)

**作者:** Songmiao Wang `[一作]` (Hong Kong Polytechnic University), Ming Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段的能力感知多智能体系统（MAS）自适配框架，用于根据底层LLM在当前查询上的能力动态决定角色激活和通信拓扑。

**💡 创新点**

创新点在于引入可重用的“能力指纹”与查询条件融合的能力表示，利用该表示作为信号自适配MAS结构，从而在保持性能的同时降低多轮交互的token成本。

**🔧 技术方法**

技术上采用了能力指纹构建、任务条件化能力建模、弱监督的角色激活机制、图神经网络生成通信图以及基于REINFORCE的成本-效益优化。

**📊 数据集**

实验数据集涵盖 MMLU、GSM8K、MultiArith、SVAMP、AQuA、HumanEval 等六大标准基准，并在多种LLM后端（GPT‑4o、GLM‑5.1、Gemini‑3.1‑Pro、Qwen‑3‑14B、Llama‑3.1‑8B、Gemma‑3‑12B 等）上进行评估。

**📈 对比分析**

与单一代理、固定拓扑以及现有动态或优化的MAS方法相比，本文方法在平均准确率上提升至 93.81%，在强大后端上降低 36.26% 的token消耗，在弱后端上提升 4.58 分，且在未见过的后端上也保持相同趋势。

**⚠️ 局限性**

局限性包括：① 需要一次性构建并维护每个后端的能力指纹；② 角色描述和指纹的设计依赖人工或预训练模型；③ 在极端高效能模型或极端低效能模型上的自适配程度仍有提升空间。

---

## 322. GuidedNav: Shaping Vision-Language Navigation Representations through Pre-action Attention and Spatial Guidance

**arXiv ID:** 2610.13122 | [PDF](https://arxiv.org/pdf/2610.13122v1)

**作者:** Jingyu Guo `[一作]` (University of Melbourne), Mingming Gong `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 GuidedNav，一种通过训练时的子任务注意力、地标定位以及空间引导，直接从视觉和语言信息生成动作的端到端导航框架。

**💡 创新点**

创新点在于：①在行动前位置直接监督注意力，使模型聚焦当前子任务句段和对应视觉地标；②利用冻结的长短视差几何教师在训练阶段对齐全局与局部空间特征；③将这些引导作为统一的辅助目标而非生成中间文本，既提升表现又保持推理时的低延迟。

**🔧 技术方法**

技术包括：Transformer 视觉‑语言多模态编码器、预训练语言模型（Qwen3‑VL）、冻结的 VGGT 视觉几何教师、对注意力的交叉熵与余弦相似度监督、历史/邻近帧的张量融合门控、以及 LoRA 微调。

**📊 数据集**

使用了两个增强数据集 Guided‑R2R 与 Guided‑RxR（在原始 R2R 与 RxR 上添加子任务段与地标掩码），以及公开的 R2R‑CE 与 RxR‑CE 验证集。

**📈 对比分析**

与现有最强 monocular RGB 方法相比，GuidedNav 在 R2R‑CE 的 SR 提升 8.1%（SPL +4.0%），在 RxR‑CE 的 SR 提升 13.1%（SPL +9.9%），并在 RxR‑CE 上 nDTW 提升 10.5%；推理速度比 JanusVLN 与 MonoDream 快 10.7× 与 2.6×，延迟仅 0.7% 低于 StreamVLN。

**⚠️ 局限性**

主要限制：需要手工标注的子任务段与地标掩码，训练过程复杂且对计算资源有较高要求；在极端动态或无标注环境下的迁移性尚待进一步验证。

---

## 323. BenCoDiC: A Large-Scale Contextual Cyberbullying Corpus for Bengali Political News Discourse FULL VERSION

**arXiv ID:** 2610.13121 | [PDF](https://arxiv.org/pdf/2610.13121v1)

**作者:** Md. Nesarul Hoque `[一作]` (University of Chittagong), Md. Hanif Seddiqui `[通讯]` (University of Chittagong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

创建了BenCoDiC大规模上下文网络霸凌语料库，收集并注释了50,250条来自Bengali政治新闻YouTube视频的字幕、评论和回复，形成二分类标签。

**💡 创新点**

在低资源语言背景下首次维护完整的字幕–评论–回复关系，并采用星型九维度模式结构，解决了缺乏上下文、标注可靠性、规模与可访问性不足的缺陷。

**🔧 技术方法**

采用两层人工标注并专家验证（Fleiss' κ、Krippendorff α），使用六种Transformer模型（BanglaBERT、Bangla-Bert-Base、BanglishBERT、IndicBERT v2、mBERT、XLM‑R）评估上下文对分类的影响，并做层次化编码、消融与误差分析。

**📊 数据集**

利用自制的BenCoDiC数据集（50,250条样本，约60% bully），来源于100条Bengali政治新闻YouTube视频，覆盖8个频道、5个政党、5种语言类别与6种目标身份。

**📈 对比分析**

采用视频级离散划分的训练/验证/测试集，比较无上下文与平铺上下文输入；所有模型宏F1提升0.5–0.8个百分点，95%置信区间均>0，证明上下文显著提升性能；层次化编码略逊于平铺但仍优于无上下文。

**⚠️ 局限性**

研究仅聚焦政治新闻、时间段有限、样本主要为评论、语言分布不平衡、专家审核可能受锚定偏差、AI标注测试仅500样本，元数据标签未评估可靠性，结果不具普适性。

---

## 324. Human2Dex: Dexterous Robot Learning from Wrist-Centric Human Demonstrations

**arXiv ID:** 2610.13119 | [PDF](https://arxiv.org/pdf/2610.13119v1)

**作者:** Jiachen Zhang `[一作]` (Beijing Academy of Artificial Intelligence), Shaqi Luo `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套可复用的腕部中心人类演示接口Human2Dex，利用单一人类演示数据集实现不同机器人手臂的抓取与操控策略学习；

**💡 创新点**

创新点在于：①共享腕部中心的人类演示源；②通过21点手部结构统一视觉与运动语义；③融合PICO与RGB手部追踪生成可执行的目标手动作；④提出Grasp‑Pocket Interaction Representation (GPIR)，将物体位置与手部抓取口关联；⑤实现跨不同高自由度手臂（Linker O6与Wuji）的独立策略训练而无需收集新的机器人演示；

**🔧 技术方法**

技术方法包括：腕部摄像头与PICO追踪、DINOv2 ViT-B/14 Reg4编码器、手部结构融合、目标特定动作监督（非线性优化Retargeting）、Kalman滤波、SLSQP求解、Diffusion Policy、辅助GPIR回归、图像空间目标跟踪与追踪可靠性估计；

**📊 数据集**

使用的数据集为一套包含六个真实世界抓取任务（Sponge、Croissant、Cup、Bottle、Egg Carton、Open Drawer）的腕部中心人类演示，共1,899条基础演示、556,624帧，采用多种外观渲染扩增；

**📈 对比分析**

通过在Linker O6和Wuji两台手臂上分别进行20次物理试验，对比四种配置（Human‑Raw、Human‑Visual、Human‑Structured、Human2Dex）和腕视与头视实验。Human2Dex在六个任务上取得109/120（90.8%）和97/120（80.8%）的成功率，整体成功率为85.8%，远高于Human‑Raw（26.3%）和Human‑Visual（60%）。腕视演示显著优于头视（30% vs 0%）；GPIR在需要整体抓取对齐的任务中提升显著，而对局部接触任务提升有限；

**⚠️ 局限性**

局限性包括：仅验证了两款特定手臂和六个任务，未检验更广泛的手臂形态与动作空间；GPIR仅提供整体抓取口关系，对局部功能交互（如盒子拉手或抽屉把手）不够精准；缺乏与机器人演示基线的直接样本效率对比；腕视与头视对比仅在单任务单手中进行；依赖可靠的图像跟踪与标记，若跟踪失败将影响性能；

---

## 325. Forward Dynamics Models

**arXiv ID:** 2610.13117 | [PDF](https://arxiv.org/pdf/2610.13117v1)

**作者:** Zipei Ma `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Forward Dynamics Models (FDM)，通过动作到未来的学习方式，在训练中利用未来状态监督动作特征，但推理时仅使用动作，不需要生成未来状态。

**💡 创新点**

创新点在于将未来预测与动作预测的条件方向逆转（动作→未来），实现动作仅依赖当前观测的推理，消除未来预测误差对动作的传播，并保持对未来监督的利用；同时模型与不同后端（VLM、VGM）兼容。

**🔧 技术方法**

使用 Mixture‑of‑Transformers (MoT) 架构、动作流匹配（action flow matching）、VAE 潜在空间未来预测、VLM 与 VGM 后端，以及动作与未来特征的前向连接与反向梯度传播。

**📊 数据集**

使用 LIBERO、RoboTwin 2.0 仿真数据集以及 5 个真实机器人任务（Drawer、Box、Shuttlecock、Cup、Block）进行评估。

**📈 对比分析**

与 VLA 基准（π_0、π_0.5）以及 WAM 基准（Motus、LingBot-VA、FastWAM‑Joint）进行对比；在 LIBERO 上平均成功率从 96.9% 提升至 99.4%，在 RoboTwin 上分别为 81.20% 与 89.23%，在真实机器人任务上平均成功率提升至 96%（VLM）/83%（VGM），并且 FDM 推理速度比 FastWAM‑Joint 快 1.8 倍。

**⚠️ 局限性**

局限性包括：需要 VAE 或 VLM/VGM 的预训练模型，训练时计算成本对 VLM 略有提升；仅在能够获取准确未来状态标签的任务中有效，对无法获取未来状态的场景适用性有限；实验规模主要集中在特定操纵任务，尚未验证在更大尺度或动态环境中的泛化。

---

## 326. What Do Visual-Head Discovery Scores Actually Find? Attention Source, Necessity, and Random-Control Pitfalls in Vision-Language Models

**arXiv ID:** 2610.13111 | [PDF](https://arxiv.org/pdf/2610.13111v1)

**作者:** Md Abdul Kadir `[一作]` (German Research Center for Artificial Intelligence), Daniel Sonntag `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统地研究视觉语言模型（VLM）中的视觉注意头，探讨不同发现策略（查询类型、注意源、语料库）对头集选择的影响，并对所选头集进行两种因果验证（对比转向Steerability与掩蔽Necessity）。

**💡 创新点**

创新点在于：①通过12项交叉设计揭示查询类型和注意源对头集选择的决定性作用；②发现Necessity验证能有效区分不同发现设置，而Steerability在默认强度下并不能；③提出一套报告协议，要求在多层匹配随机抽样和必要性/掩蔽控制下评估视觉头，提升实验可复现性和结论可信度。

**🔧 技术方法**

使用技术包括：注意模式评分（对图像区域关注度统计）；对比转向（在指定图像区域增强/抑制注意力并测量答案概率差异）；掩蔽必要性（在匹配与不匹配的图像多选题上比较注意力遮蔽后的概率差异）；层匹配随机对照、多层随机抽样；以及大规模实验设计和统计显著性检验。

**📊 数据集**

数据集主要为：① Comic Panel Strips（漫画六格图），② COCO图像；还利用多种转换版基准（CV-Bench、MMStar、MathVista、CountBenchQA、SpatialRGPT-Bench 等）进行指向测试。所有实验均基于公开的 Qwen3-VL-8B 及其他 VLMs 进行验证。

**📈 对比分析**

比较方法：对12个头集分别在 20 个层匹配随机集和 250 个层匹配随机集上计算 Steerability（CERS）和 Necessity（NCS）评分，并通过排名 p‑value 与 95% 置信区间评估显著性。结果显示：① 在 Necessity 评估中，query‑span 头集显著高于随机（p < 0.001），并且仅该两类设置在 Holm 校正后保持显著；② 在 Steerability 评估中，发现头集与随机头集无显著差异，随机分布宽广，单一随机抽样不具代表性；③ 在指向测试套件上，query‑span 头集平均 Steerability 略高于随机，但差距不显著。

**⚠️ 局限性**

局限性：① 12项设计主要在 Qwen3-VL-8B 上完成，其他模型仅做后续检查；② Steerability 仅在默认强度（β=10⁴）下评估，弱强度下可能出现差异；③ 随机对照抽样仅一次，未覆盖全部可能分布；④ 注意力权重测量使用单一图像提示，可能不具普适性；⑤ 未探索跨层组合、头层数或模型内部结构对结果的进一步影响。

---

## 327. From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale

**arXiv ID:** 2610.13102 | [PDF](https://arxiv.org/pdf/2610.13102v1)

**作者:** Yaroslav Getman `[一作]` (Aalto University), Mikko Kurimo `[通讯]` (Aalto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对芬兰语自监督语音学习进行大规模实验，利用 2.1M 小时广播音频，经过 VAD、去重、ASR 置信度过滤得到 471K 小时数据，在 wav2vec 2.0 框架下进行预训练，并比较多种编码器结构。

**💡 创新点**

① 引入 ASR 置信度作为数据质量过滤指标，较 LID 过滤提升 3‑7% WER；② 在固定参数预算下系统对比 Transformer、Macaron‑Net、Conformer、E‑Branchformer，揭示卷积增强结构在低容量时优势；③ 将大规模非结构化广播语料用于 SSL 并公开模型。

**🔧 技术方法**

使用 wav2vec 2.0、pyannote.audio VAD、ASR 置信度过滤、ECAPA‑TDNN LID、FastText 文本 LID、Transformer/Conformer/E‑Branchformer 编码器、Fairseq 训练与 CTC 微调。

**📊 数据集**

数据来源：KAVI 2.1M 小时广播档案、VoxPopuli 芬兰子集、芬兰议会（FP）、Lahjoita puhetta（LP），以及 Common Voice、FLEURS、KAVI 手工标注测试集。

**📈 对比分析**

在固定参数量（95M、317M、965M）下对六个基准（LP、FP、CV、FLEURS、KAVI）评估 WER；Base 规模时 Conformer/E‑Branchformer 相比 Transformer 可降至 37% WERR，置信度过滤比 LID 过滤多 3‑7% WER；大规模时差异减小，Transformer 更稳定。

**⚠️ 局限性**

限制：广播数据受版权限制，无法公开；置信度阈值需经验选取，过严可能导致多样性丢失；不同规模下架构优劣变化，超大模型时卷积优势消失；对其他语言的迁移性未完全验证。

---

## 328. Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison

**arXiv ID:** 2610.13082 | [PDF](https://arxiv.org/pdf/2610.13082v1)

**作者:** Mohammed Azeez Khan `[一作]` (National Institute of Technology Warangal), Amar Kumar Behera `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了大语言模型生成的会议引导与人类项目经理在场景设计会议中的对话行为。

**💡 创新点**

首次构建了上下文匹配的多维度行为比较框架，展示了LLM在语义距离、话题偏差、词汇展开和阶段词汇对齐等方面的差异。

**🔧 技术方法**

采用句子嵌入、LDA话题模型、TF‑IDF、词汇计数等自然语言处理方法以及Llama 3.1 8B Instruct的文本生成。

**📊 数据集**

使用AMI Meeting Corpus中的四人场景设计会议记录，并对项目经理发言进行提取。

**📈 对比分析**

通过配对Wilcoxon检验、Friedman检验、混合效应模型等统计方法，发现LLM在话题偏差、词汇展开、阶段词汇对齐上显著高于人类，且在语义距离和情感/保留词密度上人类更突出。

**⚠️ 局限性**

局限包括样本仅来自单一情境与模型版本，词汇展开受长度影响，情感词典主要是保留词，未检验真正的心理共情或引导效能。

---

## 329. VICAR: Video Imitation with Contact-Aware Retargeting for Humanoid Interactive Skills

**arXiv ID:** 2610.13079 | [PDF](https://arxiv.org/pdf/2610.13079v1)

**作者:** Dvij Kalaria `[一作]` (University of California, Berkeley), S. Shankar Sastry `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套从第三人称视频中提取人类动作、重建场景障碍并进行接触感知重定向的端到端管线，以实现类人机器人在多种交互任务中的自主学习和硬件部署。

**💡 创新点**

核心创新在于通过高优先级碰撞与全局接触姿势惩罚实现接触保持与速度保持的重定向，并引入可移动接触点条件的运动生成器，支持在线接触位置调整和对接触点的泛化增强。

**🔧 技术方法**

结合了单目深度与语义分割（UniDepth、Semantic‑SAM）、人类姿态恢复（GENMO/GEM）、OmniRetarget式两阶段重定向、Lagrangian 约束优化、基于可移动接触点的神经运动生成器，以及PPO强化学习跟踪策略和领域随机化训练。

**📊 数据集**

主要使用自采集的第三人称RGB视频（含桌球发球、物体捡拾、梯子攀爬等十项任务）以及无实机训练数据；通过手工标注接触帧与姿势构建训练集。

**📈 对比分析**

与PHC、GMR、VideoMimic、OmniRetarget及其速度保持消融版对比，采用穿透深度、接触残差、无效率、平滑度、速度误差和RL成功率等指标，结果显示本方法在接触准确性、运动平滑度和仿真到实机成功率方面均显著优于基线（如桌面捡拾成功率从76.2%提升至99.8%，梯子攀爬从56.0%提升至95.5%）。

**⚠️ 局限性**

局限包括需人工标注接触点（每项任务约20–30分钟）、对场景重建精度高度依赖、对不可达或失衡接触易失败、梯子硬件部署失败、需额外运动捕捉工程、以及对单目深度与语义分割误差的鲁棒性不足。

---

## 330. The value of information and the emergence of communication in evolved e-puck robots

**arXiv ID:** 2610.13118 | [PDF](https://arxiv.org/pdf/2610.13118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 331. Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs

**arXiv ID:** 2610.13078 | [PDF](https://arxiv.org/pdf/2610.13078v1)

**作者:** Zixuan Weng `[一作]` (Georgia Institute of Technology), Yang Bai `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型安全对齐中的“识别→写作”路径，分析注意力模块如何识别有害性、MLP模块如何写作拒绝，并基于此提出 Sentinel 诊断与条件提醒方法；

**💡 创新点**

首次将注意力的有害性识别与 MLP 的拒绝写作分离为两个功能端点，揭示二者在模型层级中的功能异同，并通过因果干预验证其连通性；

**🔧 技术方法**

利用残差流分析、线性可解码器、拒绝证据评分（RES）、拒绝写作评分（RWS）、因果干预、以及多层特征组合的条件提醒路由；

**📊 数据集**

基于 Qwen、Llama、Gemma 等 18 对 Base–Instruct 大模型（1B–27B）以及 6 个 Qwen3/Qwen3.5 具推理功能的安全与过度拒绝基准（XSTest-safe、EVOREFUSE-TEST、AdvBench、AlpacaEval、ReNeLLM 等）；

**📈 对比分析**

与固定提醒策略（如 Safety、Helpfulness、Balanced）以及现有防御方法（Llama Guard 3、Self‑Reminder、ICD）比较，条件提醒实现了平均 30.34% 的整体错误率，显著低于单一提醒（最优 40.06%）和无提醒（61.57%）；

**⚠️ 局限性**

主要局限在于：只考察了 decoder‑only Transformer 的残差流，未涉及 encoder‑decoder 或大规模多模态模型；方法依赖于对内部表示的精确可解码，可能在更大规模或不同训练策略下失效；

---

## 332. Can Failed Trajectories Reveal Where Agents Go Wrong? Agentic Failure Attribution Without Step-Level Labels

**arXiv ID:** 2610.13105 | [PDF](https://arxiv.org/pdf/2610.13105v1)

**作者:** Do Nguyen-Thanh `[一作]` (Nanyang Technological University), Sean Du `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种利用未标记失败轨迹进行决策错误定位的框架

**💡 创新点**

创新点在于使用谱基评分提取错误特征并结合注意力驱动的传播去除下游污染，完全不依赖逐步错误注释或成功轨迹

**🔧 技术方法**

技术包括冻结LLM提取步骤表示、构建谱参考空间、基于谱能量的误差评分、从注意力权重构建软依赖图并进行误差传播

**📊 数据集**

在Who&When、Correct-Error、TraceElephant等五个子数据集上进行评估，涵盖问答、编码与工具使用任务

**📈 对比分析**

与prompt‑based（如All‑at‑Once、ErrorProbe）和training‑based（如OAT、StepFinder）基线对比，平均提升30‑60%步级定位准确率，且在大模型、跨分布与合成轨迹场景中保持优势

**⚠️ 局限性**

限制包括仍需足量未标记失败轨迹来构建谱参考，依赖冻结LLM的内部表示，且对极短或极长轨迹的下游污染处理尚不完善

---

## 333. From Glance to Focus: Mitigating Recognition Laziness in MLLMs for Fine-Grained Recognition

**arXiv ID:** 2610.13154 | [PDF](https://arxiv.org/pdf/2610.13154v1)

**作者:** Yutong Yang `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练无关的LIVE框架，通过对候选类别列表的精炼和自适应的“技能”指导，缓解多模态大型语言模型在细粒度视觉识别中的懒惰现象，显著提升识别准确率。

**💡 创新点**

核心创新在于首次识别并定义了FGVR懒惰现象；设计了联合候选压缩与可按需调用的技能机制；实现了无需额外训练即可在推理时提升模型细粒度识别能力。

**🔧 技术方法**

使用了Plausible Candidate Inference（结合视觉相关性与互类别亲和图）、Category Skill Construction（通过MLLM自反思生成视觉检查清单和比较知识）以及Skill‑inspired Confusion Resolution（基于预测熵与前缀匹配的动态技能调用）等技术。

**📊 数据集**

在13个公开基准上验证，包括9个细粒度图像分类（Flowers102、CUB200、Pet37、FGVC-Aircraft、Cars196、Dogs120、DTD、Food101、SUN397）、2个细粒度动作识别（UCF101、HMDB51）以及ImageNet‑1K与ImageNet‑Sketch。

**📈 对比分析**

与24种基线（VLM、通用MLLM、后训练模型）对比，LIVE在所有M相机上平均提升约10–15%，在细粒度图像、动作识别以及一般/OOD图像分类任务中均显著优于现有方法。

**⚠️ 局限性**

局限性包括对预训练VLM和参考样本构建亲和图的依赖，极少样本或无参考场景下效果可能受限；额外的技能调用虽然提升准确率，但也会带来一定的推理延迟。

---

## 334. SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty

**arXiv ID:** 2610.13081 | [PDF](https://arxiv.org/pdf/2610.13081v1)

**作者:** Yixuan Li `[一作]`, Mengyue Wu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为SWEAR的结构化难度建模框架，用于在大规模音频-语言模型中实现自适应推理。

**💡 创新点**

创新点在于将推理过程分段并为每段赋予多维难度评分（生产质量、复杂度、整合度、推理需求、问题质量），通过自学习难度标记和强化学习动态分配推理长度，并加入课程学习与难度校准。

**🔧 技术方法**

技术方法包括：多维难度注释构建（DiCoTA）、有监督微调学习自评分、基于GRPO的强化学习实现推理长度自适应、课程学习调度、难度校准与KL正则。

**📊 数据集**

使用的数据集为基于CoTA扩展的DiCoTA（57.5万例）以及评测基准MMAU-test-mini和MMAR。

**📈 对比分析**

与CoTA-SFT、DiCoTA-SFT以及Audio-Reasoner、Audio-Flamingo 3等基线相比，SWEAR在MMAU-test-mini上实现73.5%准确率（与最佳持平），在MMAR上62.0%准确率，且生成标记平均减少约60%（197和196 token vs 494/500 token），显著提升效率而不牺牲性能。

**⚠️ 局限性**

局限性包括：仍依赖离线难度标注，难以直接推广到多模态或更大规模数据；强化学习训练成本高；模型在极端难度或不同音频域上的泛化尚待验证。

---

## 335. AV-FlowEdit: Training-Free Prompt-Driven Audio-Visual Editing with Event-Aware Spectral Trajectory Preservation

**arXiv ID:** 2610.13072 | [PDF](https://arxiv.org/pdf/2610.13072v1)

**作者:** Shaohua Pan `[一作]`, Xiao Liu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 AV-FlowEdit，一个无需训练、无需反演的 prompt‑driven 音视频联合编辑框架，能够在冻结的联合音视频生成器上实现视频与音频的同步编辑。

**💡 创新点**

创新点包括：①利用源‑目标差分流直接推导跨模态编辑方向；②设计跨模态事件感知频谱轨迹保持机制，实现自适应结构漂移抑制；③实现无优化、无掩码的统一编辑流程，保持预训练模型的跨模态交互。

**🔧 技术方法**

使用技术包括：差分流编辑与轨迹稳定化、跨步相关噪声与候选聚合、多尺度低频结构分解、事件感知门控、扩散调度、以及联合音视频 Transformer 的冻结模型。

**📊 数据集**

文中未显式列出使用的数据集，推测采用公开的音视频编辑基准数据集（如 AV‑Edit 或类似数据集）进行实验。

**📈 对比分析**

通过与 ControlVideo、TokenFlow、RAVE 等独立与级联编辑基线以及 DDS、AvED 等联合编辑基线进行对比，使用 CLIP‑F、CLIP‑T、Obj.、DINO、CLAP、LPAPS、ImageBind、AV‑Align 等多维度指标评估。实验显示 AV‑FlowEdit 在目标语义编辑、源内容保留、音频质量与跨模态一致性方面均优于现有训练‑free 方法，特别在 CLIP‑T、Obj.、CLAP、AV‑Align 指标上取得最优成绩。

**⚠️ 局限性**

局限性包括：对极大范围的全局重构能力有限；未专门针对说话头部同步等细粒度事件进行优化；对长时序音视频编辑的鲁棒性和多样性评估尚不充分；以及缺乏对不同场景与复杂音频事件对齐效果的深入分析。

---

## 336. Evolving Creatives with Clicks: CTR-Driven Advertising image Editing via Attribute Disentanglement and Preference Alignment

**arXiv ID:** 2610.13066 | [PDF](https://arxiv.org/pdf/2610.13066v1)

**作者:** Xu Zhang `[一作]` (Nankai University), Qicheng Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ECC 框架，通过对现有广告图进行细粒度属性编辑，提升电商广告的点击率。

**💡 创新点**

创新点包括：①将创意优化从生成转为细粒度属性编辑；②利用 ANOVA 与随机森林筛选出 28 个 CTR 敏感属性并构建编辑建议数据集；③提出 Consistent Preference Optimization（CPO）对 DPO 进行一致性正则化，缓解过拟合并提升偏好泛化。

**🔧 技术方法**

使用技术包括多模态大语言模型（Qwen3-VL）、指令微调、强化学习（DPO）、统计分析（ANOVA、随机森林）、顺序不变正则化、在线点击反馈收集与分析。

**📊 数据集**

数据集包括：550k+ 在线广告图 + 51 维属性抽取 + 28 维 CTR 相关属性；101k 指令微调样本；68k 在线偏好对；约 2000 万曝光的在线实验数据。

**📈 对比分析**

通过在线实验与 CAIG、GPT、Gemini 等基线对比，采用 CTR 增益指标评估；ECC 获得 6.86% 的 CTR 增益，显著优于其他方法，并在美学评估中排名第一。

**⚠️ 局限性**

局限性包括：①需要大量在线点击数据，部署成本高；②对属性抽取误差和冗余仍有影响；③编辑后需人工审核保证视觉质量；④CPO 正则化参数需调优，适用性可能受限。

---

## 337. Tensor-based Fusion of Multi-modal Data for Predicting Cancer Risk Levels

**arXiv ID:** 2610.13221 | [PDF](https://arxiv.org/pdf/2610.13221v1)

**作者:** Ali Anaissi `[一作]` (University of Sydney), Junaid Akram `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种无监督机器学习框架，结合特征提取、张量分解和聚类，用于多模态癌症数据的风险分层。

**💡 创新点**

创新点在于通过张量融合与无监督聚类实现跨模态特征的有效降维和分层，tSVD+GMM组合在临床指标上表现最佳。

**🔧 技术方法**

使用PCA、tSVD、MDS进行降维，CP（PARAFAC）张量分解并通过CORCONDIA确定秩，随后评估K‑means、K‑medians、GMM、DBSCAN、层次聚类和CLIQUE等聚类算法。

**📊 数据集**

采用LinkedOmics数据库的32种癌症样本（共11,158例），以乳腺浸润性癌为重点，整合临床注释、DNA甲基化、miRNA表达、RNA‑seq和拷贝数变异五种模态。

**📈 对比分析**

通过内部指标（Silhouette、Calinski‑Harabasz）和外部指标（Fowlkes‑Malom、准确率）进行比较，tSVD+GMM在多项临床分层指标上取得最高准确率（约0.68），并在Silhouette和Calinski‑Harabasz上也表现优异。

**⚠️ 局限性**

局限性包括仅在乳腺癌数据上验证，未做生存分析或外部验证；对高维度数据的处理仍受限；DBSCAN、CLIQUE等方法在本数据集表现欠佳，且整体模型缺乏监督校准。

---

## 338. Multi-Modal Graph Neural Network-Inspired Vessel Identification and Tracking for UAV-Based Maritime Surveillance

**arXiv ID:** 2610.13220 | [PDF](https://arxiv.org/pdf/2610.13220v1)

**作者:** Chenjie Zhao `[一作]` (Wuhan University of Technology), Ryan Wen Liu `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于多模态图神经网络的无人机辅助海事物体识别与跟踪框架AMGT，用可见光与红外图像融合、AIS数据同步、MAir‑YOLOv8‑OBB检测、MAir‑SORT跟踪以及运动感知坐标统一与图学习关联，实现了在复杂海况与无人机动态运动下的高精度船舶检测、跟踪与身份关联。

**💡 创新点**

创新点包括：① 采用多参数视角变换模型对可见光与红外图像进行精准对齐与融合；② 开发轻量级MAir‑YOLOv8‑OBB检测器与MAir‑SORT跟踪器，显著降低身份切换；③ 设计运动感知坐标统一（MCU）算法补偿无人机运动引起的非线性失配；④ 引入双流时空图注意力网络进行视觉‑AIS关联，利用相对拓扑结构增强关联鲁棒性。

**🔧 技术方法**

核心技术涵盖：多模态图像融合与变换、混合模态YOLOv8‑OBB目标检测、基于角度自适应的SORT跟踪、姿态补偿的坐标统一、时空图注意力与GAT/GCN的图学习关联。

**📊 数据集**

使用自建的MAP‑M2UAV50数据集，该数据集集成可见光、红外视频、AIS信息、无人机飞行数据，并提供旋转边框、跟踪ID与MMSI标注。

**📈 对比分析**

通过与YOLOv8/YOLOv11基线、SORT/ByteTrack/BoT‑SORT跟踪器以及CD/DTW/ED几何方法和LSTM/GRU/GAT/GCN等学习方法比较，AMGT在检测mAP50-95达97.8%，跟踪IDF1 91.8%，关联精度/召回率超过98%/97%，仅出现约1.3次身份切换，显著优于传统方法，且推理时间可达3.8 ms/窗口。

**⚠️ 局限性**

主要局限包括：① 对光照与海面噪声仍有一定敏感性，未来需更鲁棒的图像配准与融合；② MCU在极端飞行姿态下的精度受限，需更精确的姿态估计；③ 目前仅在单架无人机场景验证，缺乏多无人机协同与边缘算力部署的实验，未来需扩展到分布式多机系统。

---

## 339. Complexity-Calibrated Reconstruction Discrepancy for No-Reference Point Cloud Quality Assessment

**arXiv ID:** 2610.13219 | [PDF](https://arxiv.org/pdf/2610.13219v1)

**作者:** Haocheng Wu `[一作]` (City University of Macau), Xin Sun `[通讯]` (City University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无参考点云质量评估框架，利用连续表面重建校准重建差异。

**💡 创新点**

创新点在于用更强正则化的第二次重建作为结构复杂度的代理，消除重建差异中结构与退化的混淆，并融合冻结预训练的几何与投影视图特征。

**🔧 技术方法**

采用连续重建、冻结预训练的三维/二维特征提取器、软加权对比、投影滤波及正则化强度校准等技术。

**📊 数据集**

使用了公开的 SJTU‑PCQA、WPC2.0 与 ICIP2020 三个点云质量数据库。

**📈 对比分析**

与全参考、监督式 NR 以及无训练基线方法比较，在三数据库上取得与大多数无训练方法相当或更优的 SRCC/PLCC，虽不及最强监督方法。

**⚠️ 局限性**

局限在于对编码相关失真响应不足，重建响应无法完全覆盖所有码流引入的失真，且未进行自适应重建与校准。

---

## 340. Golden Anchor for Video Generation

**arXiv ID:** 2610.13217 | [PDF](https://arxiv.org/pdf/2610.13217v1)

**作者:** Jinyuan Xu `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Golden Anchor Learning（GAL）框架，通过让图像‑到‑视频模型（I2V）在冻结状态下对锚点进行渲染并评估视频奖励，进而以视频反馈优化文本到图像（T2I）生成器，使得生成的锚点更能产生高质量视频。

**💡 创新点**

创新点在于将锚点视为“黄金锚”，并通过跨模态DPO将视频奖励转化为图像级别的偏好标签，从而实现无需修改I2V模型即可提升视频质量与运动。

**🔧 技术方法**

使用的技术包括冻结的I2V渲染器、视频奖励模型（VideoReward、HPSv3、UniPercept）、跨模态DPO（Diffusion DPO）以及差分LoRA训练策略。

**📊 数据集**

训练数据为基于HunyuanVideo‑1.5渲染的5秒视频，评估使用VBench和Kling v3 Pro等四个I2V模型的提示集合。

**📈 对比分析**

与基线（未调优的T2I）相比，GAL在四个I2V模型上平均提升VBench分数2.97%–8.46%，动态度提升5.62%–38.24%，且视频整体奖励和帧级奖励均显著提高。

**⚠️ 局限性**

局限性包括仅使用单一视频奖励模型构建偏好，可能导致对其他属性不敏感；以及对训练时选定的I2V模型（HunyuanVideo‑1.5）高度依赖，虽然转移表现好，但在更具多样性的渲染器上仍需验证。

---

## 341. Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models

**arXiv ID:** 2610.13212 | [PDF](https://arxiv.org/pdf/2610.13212v1)

**作者:** Min Lin `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AFCE，利用未来的动作、机器人响应与稀疏视觉变化构建Temporal Effect并作为VLA策略的中间监督目标，随后预测并解码该Effect实现闭环控制。

**💡 创新点**

创新点在于将动作命令与其实际执行结果及环境变化联合编码为可预测、可执行的Effect，改进了传统VLA只预测动作指令的方式，并通过Effect引入物理进展信息。

**🔧 技术方法**

使用了离线Effect tokenizer、跨模态Transformer融合、条件flow-matching VLA、状态条件解码器以及跨身体实现的共享Effect空间。

**📊 数据集**

在DexJoCo多任务抓取与双臂操作数据集上进行训练与评估。

**📈 对比分析**

与官方π_0.5和UVT基线比较，AFCE整体成功率提升至50.97%（比π_0.5高11.8个百分点，比UVT高15.2个百分点），在双臂任务上表现尤为突出。

**⚠️ 局限性**

局限性包括需要同步的动作、状态与未来图像来构建目标；在细粒度单臂精确控制上仍然表现欠佳；离线目标构建成本高；共享参数未能实现完全的身体无关表示。

---

## 342. QC-MI-OVSEL: Benchmarking Query-Conditioned Multi-Instance Open-Vocabulary Sound Event Localization

**arXiv ID:** 2610.13198 | [PDF](https://arxiv.org/pdf/2610.13198v1)

**作者:** Yongqiang Chen `[一作]` (University of Surrey), Wenwu Wang `[通讯]` (University of Surrey)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种查询条件下的多实例开词汇声源定位框架Q‑RSet，并基于此构建了大规模的QC‑MI‑OVSEL评测基准。

**💡 创新点**

创新点在于引入查询条件的责任式证据聚合和可重用的源槽机制，实现了完整查询匹配源集的恢复与多源计数。

**🔧 技术方法**

核心技术包括冻结的GRAM‑Ambisonics与CLAP文本编码器、查询条件的相关性模块、三槽FiLM调制的解码器、责任性交叉注意力以及时间Transformer。

**📊 数据集**

使用的数据集为40,000个FOA场景，覆盖83个类别（其中17个为测试时保留的新类别），共80,000条平衡的存在/缺失查询视图。

**📈 对比分析**

与四个基线（OVSELD‑Adapt、Q‑MACCDOA、TQ‑SEL‑Adapt）对比，Q‑RSet在F1、LR、K‑MAE、Exact和Abs‑FPR等指标上均取得最高成绩（F1≈50.36%，LR≈46.11%，K‑MAE≈0.41）。

**⚠️ 局限性**

局限性在于对新类别、句法变体和多源并存场景的召回仍有显著下降，且模型在极高源数量或极低SNR环境下的定位精度有限。

---

## 343. PATACON: GPU-Parallel Tangent-Bundle RRT-Connect for Constrained Motion Planning

**arXiv ID:** 2610.13195 | [PDF](https://arxiv.org/pdf/2610.13195v1)

**作者:** Gahyun Oh `[一作]` (Kwangwoon University), Suhan Park `[通讯]` (Kwangwoon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PATACON——一种利用切空间引导、GPU 并行的 RRT‑Connect 规划框架，用于满足非线性等式约束的全身运动规划。

**💡 创新点**

创新点在于：① 将切空间采样与多步边缘投影相结合，实现一次采样产生多条候选边；② 在 GPU 上并行处理这些候选边，显著降低投影与碰撞检测成本；③ 引入共享切空间池和前向回溯抑制机制，提升探索效率并避免无效回溯。

**🔧 技术方法**

采用的技术包括：GPU 并行编程（CUDA）、切空间（Jacobian null‑space）采样与投影、RRT‑Connect 双向树、批量投影与碰撞检测、Halton 序列采样、以及基于约束 Jacobian 的线性化投影。

**📊 数据集**

使用了四个工业级机器人/全身平台的基准数据集：Franka Panda（7‑DOF）、双臂 Franka（14‑DOF）、FFW‑SG2（固定/移动基，18‑DOF）、Unitree G1（35‑DOF），并在各自的约束配置（闭链、姿态、双足固定、重心约束等）上生成 100 对起点–终点。

**📈 对比分析**

与 cpRRTC、TB‑RRT 及 McVAMP 进行对比。实验表明 PATACON 在高维/强约束情形下平均求解时间比 cpRRTC 低 11.25 倍，成功率均为 100%，并在 500 s 预算内完成全部试验。AO‑PATACON 在 10 s 任何时规划中，首次解的求解时间更短，余下时间可用于质量优化。

**⚠️ 局限性**

局限性包括：① 对约束 Jacobian 的线性化投影精度有限，可能在高度非线性区域产生误差；② GPU 资源成为瓶颈，低端设备无法充分利用；③ 切空间构造与维护仍需额外计算，需调参以平衡速度与精度；④ 对于极大维度或极端约束的任务仍需进一步优化。

---

## 344. WaSA: Wavelet Scattering Autoencoders for Unsupervised Offline Detection of Malicious Clients in Federated Learning

**arXiv ID:** 2610.13186 | [PDF](https://arxiv.org/pdf/2610.13186v1)

**作者:** Alessandro Licciardi `[一作]` `[通讯]` (Politecnico di Torino), Alessandro Licciardi (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全无监督的离线检测框架 Wavelet Scattering Autoencoder（WSAE），用于联邦学习中在训练开始前识别恶意客户端并筛选，避免后续训练被污染。

**💡 创新点**

创新点：①将 Wavelet Scattering Transform 与去噪自编码器相结合，仅基于干净辅助数据学习 benign manifold，无需攻击标签；②利用 GPU 加速的 SVD 低秩原型压缩产生隐私友好的单张图像摘要；③引入 Boltzmann 采样策略在客户端选择上实现软排斥，可在干净客户端极少时保持收敛。

**🔧 技术方法**

技术：Wavelet Scattering Transform、SVD 降维、去噪自编码器、Boltzmann 采样、FedAvg 聚合、GPU 并行计算、Kymatio 库。

**📊 数据集**

数据集：MNIST、CIFAR‑10、CIFAR‑100；在 100 个客户端上进行 60% 与 90% 不同攻击比例的实验。

**📈 对比分析**

与基准（FedAvg、Oracle、Waffle‑FT/Waffle‑WST）对比，WSAE 在 60%/90% 恶意比例下实现了更高的模型准确率、F1、召回率；在极端攻击时仍能保持比传统鲁棒聚合方法更好的性能，且检测 F1、精度均超过监督基线。

**⚠️ 局限性**

局限性：①需要一个干净的辅助数据集来训练自编码器；②对极少干净客户端的场景（如 90% 攻击）仍依赖温度/阈值调优；③在极端攻击（如全部黑图像）下性能仍下降；④模型对新型攻击的泛化需进一步验证。

---

## 345. AdaC-GRPO: Towards Mastering Long-Horizon Visual Sudoku Puzzles via Adaptive Curriculum Learning

**arXiv ID:** 2610.13184 | [PDF](https://arxiv.org/pdf/2610.13184v1)

**作者:** Sida Li `[一作]` (Peking University), Xiaohui Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对长周期视觉推理任务，提出了AdaC-GRPO强化学习方法，并通过FG‑Sudoku基准验证其在数独推理上的有效性。

**💡 创新点**

创新点在于将Zone of Proximal Development（ZPD）动态融入RL训练，构建基于在线rollout准确率的自适应难度排程，避免奖励稀疏导致的梯度消失，并采用细粒度难度分层来桥接不同规模数独的学习难度。

**🔧 技术方法**

使用GRPO框架结合可验证二元奖励，加入ZPD排程器，在线评估模型在不同难度组的准确率以决定下一个训练批次；同时在训练前进行SFT Warm‑Start并采用多样本群体奖励进行优势估计。

**📊 数据集**

主要数据集为FG‑Sudoku，包含4×4、6×6、9×9视觉数独样本，共3,072训练例和768测试例，按推理步骤量化的四个难度层级进行细粒度划分。

**📈 对比分析**

与SFT、GRPO、CurrReFT、AdaRFT、DAPO等基线对比，AdaC‑GRPO在Qwen2.5‑VL‑7B上将整体Pass@1从0.427提升至0.586，最难9×9层提升至0.219，接近Gemini‑2.5‑Flash，并闭合约62.2%与Gemini‑2.5‑Pro的性能差距；同样的策略在InternVL‑3.5和Qwen3‑VL上也获得显著提升。

**⚠️ 局限性**

局限主要包括：高昂的计算成本导致实验仅在少数seed上复现；难度分层和可验证奖励依赖任务特定的元数据，难以直接迁移到开放式视觉推理场景；即使如此，最难的C3/C4层性能仍较低；当前仅覆盖标准数独，未涉及更大或不规则变体，也未在更广泛的视觉推理任务上充分验证。

---

## 346. Fix the Match, Not the Solver: STITCH -- a Self-Supervised Compatibility Measure for Eroded Visual Puzzles

**arXiv ID:** 2610.13176 | [PDF](https://arxiv.org/pdf/2610.13176v1)

**作者:** Nadav Alali `[一作]` (Ben Gurion University Negev), Ohad Ben-Shahar `[通讯]` (Ben Gurion University Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化的自监督兼容度测度（STITCH）并将其作为可插拔模块，改进了方形拼图（含侵蚀、缺失、未知方向及不规则碎片）重组的匹配步骤。

**💡 创新点**

核心创新在于抛弃传统依赖边界像素的匹配方法，利用冻结的 MAE ViT 内部补丁特征与跨注意力头直接对比碎片内部纹理，从而在高侵蚀、缺失或不规则形状下保持高度鲁棒性；同时通过“右侧面向”统一化处理，减少方向性开销。

**🔧 技术方法**

技术手段包括：冻结 MAE ViT-B/16 的 16×16 补丁编码、投影至 64 维、两层双向跨注意力（4 头）以及平均余弦相似度评分；训练采用无监督 InfoNCE 损失，仅对正负邻居进行对比，无需人工标注。

**📊 数据集**

使用了 JPwLEG-3/5（侵蚀方形拼图）、自定义 ImageNet 7×7（更大规模、可变侵蚀、缺失碎片）以及 GAP-3/5（从真实考古碎片学习的形状不规则拼图）等数据集。

**📈 对比分析**

与传统手工 CM（SSD、L1、PBC、MGC、Edge2Vec）以及多种兼容度基搜索求解器（贪婪、遗传、LP）以及最新学习式求解器（Deepzzle、SD^2RL、DiffAssemble、PuzzleFlow 等）进行比较；STITCH 在 JPwLEG-5 的遗传求解器中实现 0.992 Abs / 0.954 Perf，几乎达到 0.99 Perf 的上限，并在 ImageNet 7×7 上在 6 px 侵蚀下 Top-1 维持 90% 以上，优于所有对比方法；在 GAP 上同样超过现有学习式求解器，Abs 和 Perf 均提升 0.25 以上。

**⚠️ 局限性**

局限性包括：对纹理缺失或极其重复区域的匹配仍易失误；在极大规模拼图（上万碎片）时需要完整兼容矩阵缓存导致内存消耗；以及对完全未知方向或旋转的碎片仍需进一步改进，虽然已在补充材料中给出部分解决方案。

---

## 347. PreCoG: Preview-Edit for Efficient Video Correction in Generation-Time

**arXiv ID:** 2610.13174 | [PDF](https://arxiv.org/pdf/2610.13174v1)

**作者:** Yilai Liu `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于低成本预览的迭代文本到视频生成框架，利用预览诊断语义失败并在已缓存的生成轨迹上做局部修正，最终只对满足全部要求的候选视频进行完整渲染，显著降低了重复计算。

**💡 创新点**

创新点主要包括：① 预览指导的修正流程，提前诊断并避免不必要的完整渲染；② 通过目标–反事实对比构造软投影方向，针对性地更新缓存轨迹，最大限度保留已正确的内容；③ 复用缓存的去噪状态、速度与求解器状态，避免从头开始生成；④ 三个智能体（规划、评估、精炼）协同工作，形成无训练的闭环修正机制。

**🔧 技术方法**

技术手段包括：使用 Wan2.2‑T2V‑A14B 作为基础生成模型；利用 Qwen3.8‑27B 等 VLM 进行预览评估与规划；构造目标与反事实条件并计算预测对比；块级软投影（blockwise soft projection）实现局部轨迹编辑；缓存去噪中间状态实现高效继续渲染；设置预览与完整渲染的不同完成预算。

**📊 数据集**

数据集与基准：T2V‑CompBench 提示集合；VBench 视觉质量评估；VideoScore2 (VS2) 以及 VSR 提供的评测指标。实验中也使用了 VSR 的基准视频进行对比。

**📈 对比分析**

与基准方法相比，本文方法在 VBench 和 T2V‑CompBench 上取得最高分，在 VS2 评分上接近 VSR（仅 0.56× 的推理时间）；在 GENMAC 的三项质量与对齐指标上以 0.86× 的推理时间实现超越。实验表明在保持语义一致性的同时大幅提升了效率。

**⚠️ 局限性**

局限性包括：① 需要依赖 VLM 进行预览评估，增加了额外的计算与模型开销；② 迭代次数受固定预算限制（最多 4 次修正），对极其复杂场景可能不足；③ 目前仅在中等分辨率（832×480）和 81 帧视频上验证，尚未评估更高分辨率或长时长的适用性；④ 预览质量与完整渲染的一致性仍可能在某些极端情况下出现偏差。

---

## 348. RoleWorld: Knowledge Transfer in World Modeling with Functional Role Abstractions

**arXiv ID:** 2610.13172 | [PDF](https://arxiv.org/pdf/2610.13172v1)

**作者:** Boyuan Xiao `[一作]` (State Key Lab of CAD and CG), Yao-Xiang Ding `[通讯]` (Baiont Quant)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大语言模型代理中引入基于功能角色抽象的符号世界模型，利用可复用的角色原型和环境特定参数，实现对不同环境中物体行为的抽象与迁移。

**💡 创新点**

将世界动力学分解为共享的角色原型与环境特定参数的组合，只需更新角色分配和参数即可在新环境中高效迁移。

**🔧 技术方法**

使用大型语言模型作为教师提取角色原型和可执行的状态转移规则，采用DSL描述规则；演员通过预训练的技能程序进行决策并通过主动验证在线更新角色分配和参数。

**📊 数据集**

MARS（基于Crafter的开放世界求生基准）及其七个变体。

**📈 对比分析**

与ReAct、IFR、PoE-World、WALL-E、WorldCoder等基线对比，在所有七个变体上均实现了最高平均奖励，尤其在迁移设置下超过WorldCoder，证明功能角色抽象在跨环境迁移中的有效性。

**⚠️ 局限性**

仅能在角色分解合理的环境中有效；对完全新型物体或动力学的适应性有限；需要教师生成规则，且对语言模型的准确性与技能程序质量敏感。

---

## 349. Configuration Migration Planning through Decoupled Modeling and Searching

**arXiv ID:** 2610.13161 | [PDF](https://arxiv.org/pdf/2610.13161v1)

**作者:** Dan Wang `[一作]` (Xi'an Jiaotong University), Ratul Mahajan `[通讯]` (University of Washington)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可扩展的网络配置迁移规划框架，利用符号仿真生成 PDDL 模型，并通过不同的规划器（顺序、最优、并行）自动生成安全且符合运营意图的迁移计划。

**💡 创新点**

创新点包括：①将模型构建与计划搜索解耦，允许灵活插拔多种规划器；②采用符号仿真高效识别所有安全状态，显著提升搜索规模；③设计支持 ∀‑step 并行规划的 SMT 约束，首次在迁移规划中实现真正可并行的多步骤计划；④提供通用符号配置与路由计算框架，支持多种协议与安全属性。

**🔧 技术方法**

主要技术：符号仿真（基于 BDD，使用 JDD 库）用于安全状态识别；PDDL 描述语言实现问题建模；自动规划器（ENHSP、SPEF 等）用于顺序/最优搜索；自定义 SMT（Z3）并行规划器用于 ∀‑step 并行计划；Java 实现整体框架。

**📊 数据集**

使用 Topology Zoo 87 个真实网络拓扑，并在其上构造 5 个迁移场景（RRx2、FM2RR、NetAcq、IGPx2、LPx2），共计数千个更新操作。

**📈 对比分析**

通过与现有工具（Snowcap、SafeMigration、Cursor）对比，评估搜索时间、计划长度和安全覆盖率。实验显示：对大多数实例，顺序计划在 <1 s 内完成；最优计划在 100 s 内完成；并行计划在 <1 min 内得到最多 3 步压缩方案；符号仿真大幅减少需要搜索的状态数，整体规划完成时间 <1 h。

**⚠️ 局限性**

局限性：①仅考虑收敛后状态，忽略收敛期间的安全与临时更新；②对需要临时移除/添加配置的迁移支持不足；③并行规划在极大规模或复杂约束下仍可能超时；④需预先生成 PDDL，模型复杂度较高；⑤对非确定性收敛情况支持有限。

---

## 350. Low-Latency Mapless ESDF Queries for MAV Reactive Collision Avoidance

**arXiv ID:** 2610.13158 | [PDF](https://arxiv.org/pdf/2610.13158v1)

**作者:** Leonard Freißmuth `[一作]` (ETH Zurich), Stefan Leutenegger `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于GPU加速、无地图的ESDF查询方法，直接在已姿态化的深度图上计算距离与梯度，用于MAV的低延迟碰撞避免。

**💡 创新点**

创新点在于：①在传感器空间直接查询不需预处理的多帧深度图；②通过块级深度范围剪枝与64位键排序实现并行最短距离搜索；③引入协方差驱动的姿态不确定性边缘扩张；④可同时支持保守与乐观未知空间处理。

**🔧 技术方法**

使用CUDA并行编程、块分区与最小池化、IEEE-754整数排序、协方差矩阵求解、基于MPC的控制回路。

**📊 数据集**

使用RealSense D455深度相机采集的森林场景数据，结合OKVIS2-X姿态估计，实验包括光学扫描森林、模拟laser-scanned森林与实际野外飞行。

**📈 对比分析**

与传统KD-树、点云、体素ESDF（voxblox/NVBlox/FIESTA）以及Mesh（Embree/P2M）等基线比较，结果显示：在Jetson Orin NX上，该方法在最多400次查询内低于7ms，总延迟比KD-树、Mesh低1-2个数量级，内存占用最低；在模拟森林中，MPC碰撞率仅0.4%，优于Voxel 5.9%与NVBlox 4481%；实际飞行中所有十趟任务无碰撞。

**⚠️ 局限性**

局限性包括：①仅适用于相机深度或范围图的场景，难以直接处理稀疏LiDAR；②对未知空间的乐观/保守处理需人工调参；③当帧率过低或多帧不连贯时，拼接误差可能影响精度；④在极大分辨率下，池化导致细节丢失。

---

## 351. How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing

**arXiv ID:** 2610.13236 | [PDF](https://arxiv.org/pdf/2610.13236v1)

**作者:** Anuj Kumar `[一作]` (Indian Institute of Technology Jammu), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 AMR 解析器的输出不确定性进行后期估计，基于 beam‑search 生成的候选图进行相似度计算并评估其可靠性。

**💡 创新点**

首次在黑盒设置下系统研究 AMR 解析器的不确定性估计，并将预测拒绝比（PRR）引入评估框架。

**🔧 技术方法**

使用候选图的对称相似矩阵、谱分析（Laplacian 迹）、词汇相似度（ROUGE‑L）等多种估计器，并通过 PRR 量化可靠性。

**📊 数据集**

使用 AMR 3.0 测试集（1898 句子‑图对）以及 Smatch、Smatch++、S²Match、SEMA、WLK 等图相似度指标。

**📈 对比分析**

通过对六种主流 AMR 解析器在不同 beam 大小和评估指标下的 PRR 进行平均比较，发现 SPRING 在准确率和不确定性可靠性方面最优；高准确率解析器往往具有更高的 PRR，但 beam 过大会导致可靠性下降。

**⚠️ 局限性**

仅利用 beam 产生的候选作为不确定性来源，未探究采样或多样化解码；评估受限于现有的 AMR 质量指标，可能导致不确定性评价偏差。

---

## 352. Jag: Direct Box Prediction for Efficient Visual Grounding

**arXiv ID:** 2610.13239 | [PDF](https://arxiv.org/pdf/2610.13239v1)

**作者:** Xiuyuan Zhu `[一作]` (University of Chinese Academy of Sciences), Jian Xue `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

把 Qwen3.5-0.8B 这一预训练多模态模型改造成可以一次性直接回归视觉定位框的模型。

**💡 创新点**

创新点在于：①在最后一个有效输入 token 上直接读取多模态表示并用轻量 MLP 输出连续坐标；②对语言骨干、视觉合并器和回归头进行联合几何监督微调，提升定位精度；③实现了无需逐步生成坐标字符串、无需文本解析的单前向推理。

**🔧 技术方法**

采用预训练的 Qwen3.5-0.8B，构建轻量 MLP 回归头；使用 L1 + GIoU 损失进行几何监督；通过两阶段训练（先冻结骨干训练回归头，再联合微调骨干）实现联合适配。

**📊 数据集**

在 RefCOCO、RefCOCO+ 与 RefCOCOg 这三个公开的视觉定位表达式数据集上进行训练与评估。

**📈 对比分析**

与基准模型 Base、NExT-Chat、LocateAnything 及 Hi-Token 进行对比：在 5 个测试 split 上实现 88–90% 的 IoU@0.5 准确率，超过 Base 并与 LocateAnything 接近；在单请求推理中平均耗时仅 65.97 ms，比 NExT-Chat 低 1.82×、比 LocateAnything 低 3.62×，显著降低显存占用；与精细调优的自回归模型相比，标准阈值下精度相当，严格阈值下略逊。

**⚠️ 局限性**

仅支持单目标定位、静态图像；不涵盖多目标、视频或分割任务；推理速度与内存受输入尺寸、表达式长度、硬件和部署方式影响；在专业领域的效果需要进一步验证。

---

## 353. State-OPD: State-Grounded On-Policy Distillation for Multi-Turn Agents

**arXiv ID:** 2610.13238 | [PDF](https://arxiv.org/pdf/2610.13238v1)

**作者:** Zhentao Zhang `[一作]` (Alibaba Group), Gang Yu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于交互状态的对策学习框架State‑OPD，用以在多轮交互式任务中进行学生模型的对抗式蒸馏；

**💡 创新点**

创新点在于将学生的先前推理与交互状态分离，发现先前推理会扭曲教师监督，而重采样完整推理-动作对可提升动作多样性，并通过高熵状态聚焦采样来提升学习效率；

**🔧 技术方法**

采用了基于交互状态的教师监督、熵引导的完整响应重采样、以及对多轮任务的对抗式蒸馏损失；

**📊 数据集**

在ALFWorld、ScienceWorld和WebShop三个多轮代理基准上进行实验；

**📈 对比分析**

与Vanilla OPD、TCOD‑F2B、Guided‑OPD、TurnOPD等基线相比，State‑OPD在宏平均成功率上最高，尤其在ALFWorld上显著提升，且在少监督轮数下保持更高成功率，成功率提升可达约6个百分点；

**⚠️ 局限性**

局限性包括对教师模型的依赖（需大规模教师模型）、仅在特定三大基准上验证、对超参数（如β）敏感且尚未评估在更复杂或多模态环境下的鲁棒性。

---

## 354. Learning Schwarzschild Geodesic Motion with Lagrangian Neural Networks

**arXiv ID:** 2610.13234 | [PDF](https://arxiv.org/pdf/2610.13234v1)

**作者:** Sukru Caglar `[一作]` (Bogazici University), Arkadas Ozakin `[通讯]` (Bogazici University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种改进的拉格朗日神经网络（LNN）框架，用于学习施瓦茨希尔德测地线运动，解决了在复杂相对论系统中训练的挑战。

**💡 创新点**

创新点在于引入了独立的位移-速度归一化方案和基于能量约束的采样策略，以提高模型在不同轨道场景下的预测精度和物理一致性。

**🔧 技术方法**

使用了拉格朗日神经网络（LNN）技术，通过欧拉-拉格朗日方程推导加速度，并结合物理动机的归一化和采样策略。

**📊 数据集**

使用了基于内稳圆轨道（ISCO）结构生成的能量约束数据集，涵盖了多种轨道场景，包括近场、中场和远场轨道。

**📈 对比分析**

与基线方法相比，改进后的模型在轨道预测精度、能量和角动量守恒方面表现出色，尤其是在近场和中场轨道中，平均径向偏差为4.7662±4.4051（以施瓦茨希尔德单位计）。

**⚠️ 局限性**

限制在于模型在弱场区域的外推能力较差，且学习到的拉格朗日量未必收敛到预期的物理形式，可能需要额外的结构约束来提取时空度量。

---

## 355. Understanding and Mitigating Cross-Session Attacks via Persistent Artifacts in Agent Skill Composition

**arXiv ID:** 2610.13227 | [PDF](https://arxiv.org/pdf/2610.13227v1)

**作者:** Dianshu Liao `[一作]` (Australian National University), Xiaoyu Sun `[通讯]` (Australian National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了跨会话 AI 代理技能组合中通过持久化工件实现的攻击，并提出了一种结合工件来源恢复与预执行审查的运行时防御框架

**💡 创新点**

首次系统量化了跨会话技能攻击的三种模式（指令传播、数据边界违规、证据毒化），并构建了270个对抗案例的基准，提出了可插拔的防御框架

**🔧 技术方法**

使用 LLM 进行工件属性归因、上下文恢复、预执行审查，结合现有的 Codex、Gemini 与 Claude 代理框架进行评估

**📊 数据集**

基准数据集包含30个跨领域（10个行业）任务场景的原始材料、善意与恶意技能包，总计270个对抗案例

**📈 对比分析**

在7种模型配置下，未防御时攻击成功率(ASR)达46.7%–77.4%；引入防御后ASR降至10.9%，ACSR降至10%，任务完成率从79.6%提升至87.5%

**⚠️ 局限性**

防御仍需手动确认审查结果，且对不同模型与任务场景的泛化尚未完全验证，攻击者可通过改写工件而非仅限代码注入的方式继续潜在威胁

---

## 356. How Much Computation Does Every Forecast Need? Structured Block Skipping in Two Time-Series Foundation Models

**arXiv ID:** 2610.13208 | [PDF](https://arxiv.org/pdf/2610.13208v1)

**作者:** Max Bourgeat `[一作]` `[通讯]` (Polytechnique Montréal), Max Bourgeat (Polytechnique Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在冻结的时间序列基础模型（Chronos‑Bolt 与 TimesFM 2.5）中，是否能在推理时省略完整的 Transformer 块，以降低延迟并保持预测质量。

**💡 创新点**

发现两种不同架构的 TSFM 在不同输入窗口下存在可被省略的块；构建了基于校准窗口的静态掩码、等卡路里随机掩码、以及回溯性最佳掩码，表明存在输入相关的可压缩性但无法实现可迁移的动态路由。

**🔧 技术方法**

使用静态块剔除（禁用候选块）、递归贪心/边缘重要性排序、等卡路里随机掩码、回溯性等价预算占位符、基于冻结上下文特征的诊断分类器，以及 A100 GPU 真实延迟和吞吐量测量。

**📊 数据集**

ETTh1、Traffic、Weather、Exchange 四个公共时间序列基准，包含 512 步上下文和 64 步目标。

**📈 对比分析**

对比全模型与不同稀疏度（如 7/11、16/20）的静态掩码，测量归一化 Pinball 损失、sMAPE、MASE 以及 A100 batch‑1 延迟和 batch‑32 吞吐；在中等稀疏度下，平均预测质量基本保持不变，延迟可降低 11–13%，吞吐提升相同比例；回溯性占位符显著提升 7–21% 的质量，但单独掩码无法在不同域间迁移。

**⚠️ 局限性**

仅评估两种模型、四个单变量数据集与单一 GPU，未提供可迁移的动态路由器或边缘设备基准；块跳过不减少模型权重占用，诊断特征无法在新域转移；所选贪心掩码非最优，且缺乏实时路由开销评估。

---

## 357. Rethinking Relative Policy Differences Transfer of On-Policy Distillation

**arXiv ID:** 2610.13205 | [PDF](https://arxiv.org/pdf/2610.13205v1)

**作者:** Rong Wu `[一作]` (Zhejiang University), Jianbiao Mei `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新定义相对策略差异迁移，提出任意-任意 on‑policy distillation (A²‑OPD) 框架，并通过系统实验评估不同来源的策略差异对目标模型的迁移效果。

**💡 创新点**

创新点在于：1）从隐式奖励视角推导相对策略差异，分离代理专家、代理参考与目标三者的角色；2）构建可任意选择源代理与目标的通用迁移方法；3）系统揭示任务导向训练（RL、SFT）产生的差异有效迁移，而单纯规模提升的差异效果不稳定。

**🔧 技术方法**

主要技术包括：on‑policy distillation、KL 正则化策略优化、隐式奖励参数化、策略梯度训练、token‑级重叠分析与 KL 诊断。

**📊 数据集**

实验使用 Qwen3 与 Qwen2.5 系列模型，在数学任务上评估 AIME、SMT、CMIMC、HMMT 等基准，在代码任务上评估 HumanEval+、MBPP+、LCB 三个基准。

**📈 对比分析**

比较方法采用 proxy 差异的能力提升 G_P、目标提升 G_T、以及 R_perf 比例，并结合 KL 诊断评估迁移方向与泛化。结果表明：任务导向（RL、SFT）产生的策略差异可显著提升目标模型（提升幅度最高可达 30% 以上），而单纯规模差异迁移效果弱；匹配同一模型家族的参考能恢复失败迁移。

**⚠️ 局限性**

限制在于：迁移效果高度依赖目标与参考的兼容性，仍需手动挑选合适的参考；对不同任务、规模差异的理论解释仍不充分；实验仅覆盖 Qwen 系列模型，未验证更广泛的模型族。

---

## 358. S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding

**arXiv ID:** 2610.13202 | [PDF](https://arxiv.org/pdf/2610.13202v1)

**作者:** Yuhang He `[一作]`, Yongqi Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种改进的模型架构，能够更好地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）的结合。

**📊 数据集**

采用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有的几种主流方法进行了比较，结果显示新算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的依赖性较强，可能在其他领域的适用性有限。

---

## 359. MixTok: Adaptive Visual Tokenization with Global 1D Tokens and Sparse 2D Refinement

**arXiv ID:** 2610.13181 | [PDF](https://arxiv.org/pdf/2610.13181v1)

**作者:** Chenge Li `[一作]` (Zhejiang University), Bohan Zhuang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MixTok，一种混合视觉分词器，结合 1D 全局分词与稀疏 2D 细化分词，以兼顾表示紧凑与局部细节。

**💡 创新点**

创新点在于内容自适应路由器：通过网格打分器与 top‑p 预算控制器同时决定细化位置和每张图所需的 2D 分词数量，实现全局压缩与局部细化的动态协同。

**🔧 技术方法**

使用的技术包括 TiTok 与 LlamaGen VQ 作为基础 1D/2D 分词器、共享解码器、重构网格评分器、生成网格评分器、顶点采样 MaskGIT 以及两阶段 MaskGIT 生成流程。

**📊 数据集**

在 ImageNet 256×256 数据集上进行训练与评估。

**📈 对比分析**

与 TiTok、FlexTok、DetailFlow 等基线对比，MixTok 在约 128 个内容分词下实现 rFID 1.195、PSNR 19.51；在生成任务中约 131 个分词下 gFID 1.95，优于同等预算的 TiTok‑S‑128 并显著减少采样轮次。

**⚠️ 局限性**

局限性包括仍需对所有像素进行密集 2D 编码以完成重构、仅在 ImageNet 256×256 与固定预算范围内验证、第二阶段生成仍使用完整空间网格，未来需扩展至更高分辨率和更广泛预算。

---

## 360. PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling

**arXiv ID:** 2610.13171 | [PDF](https://arxiv.org/pdf/2610.13171v1)

**作者:** Nan Jiang `[一作]`, Wei Deng `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Predictive SubTree Prefetching框架，利用子树预取技术在一次批量模型调用中完成多次Metropolis–Hastings（MH）采样步，显著加速了基于功率采样的PowerMH、EntropyCut和MultiTryMH采样器；

**💡 创新点**

创新点在于通过构建可预取子树并采用预测遍历规则（基于对接受概率的动态规划），在保持原MH转移核不变的前提下，最大化每次批量调用完成的MH转移次数；

**🔧 技术方法**

采用的技术包括MCMC预取、前缀缓存（KV cache reuse）批量推理、三类逻辑回归预测ϵ‑certainty、以及基于动态规划的子树遍历优化；

**📊 数据集**

实验使用七个典型数据集（MATH500、LCB V6、MBPP、HumanEval、MMLU、AIME等）以及五个主流LLM（Qwen3.5-9B、Qwen3-4B、Qwen3.5-9B、Gemma4-12B等）进行评估；

**📈 对比分析**

与原始PowerMH、EntropyCut和MultiTryMH相比，本文方法在所有105个LLM+数据集组合上从未出现慢于基线，且在PowerMH、EntropyCut和MultiTryMH分别实现了1.71×、1.91×和2.33×的中位加速；在LCB V6+Qwen3.5-9B上，预测遍历规则比默认BFS进一步提升了1.25×–1.37×；

**⚠️ 局限性**

局限性包括每批调用成本相对较高，预取预算需合理调节；在某些数据集/模型组合中，预取的转移次数不足导致加速有限；同时，过度预取会增加KV缓存占用，需在资源受限环境下权衡。

---

## 361. Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding

**arXiv ID:** 2610.13164 | [PDF](https://arxiv.org/pdf/2610.13164v1)

**作者:** Feng Jiang `[一作]` (Peking University), Ruihai Wu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Grounded-Dreamer 的端到端框架，利用少量真实图像自动重建仿真环境，生成并验证物理一致、视觉逼真的机器人操控数据，无需人类遥控演示，并用这些合成数据训练机器人策略。

**💡 创新点**

创新点包括：① 将生成式视频模型与物理仿真引擎耦合，形成闭环验证，剔除物理幻觉；② 自动化从真实图像到仿真重建的管道，实现场景对齐与对象重建；③ 通过交互感知遮罩和逆动力学模型，将生成视频映射为可执行动作；④ 通过合成与验证双步骤，生成大量高质量训练样本，显著降低真实数据采集成本。

**🔧 技术方法**

技术手段包括：生成式视频模型 Cosmos-Predict 2.5 的后训练；逆动力学模型（IDM）提取动作；交互感知遮罩；使用 LLM 自动生成仿真脚本；Lehome、Isaac Sim 等物理仿真平台；Nano Banana 2 进行图像后处理与增强；仿真验证回放实现闭环物理校验。

**📊 数据集**

使用少量真实任务初始图像（约 5-10 张）进行重建，生成的合成数据覆盖七项实际任务（Pick、Close Drawer、Press Button、Pick and Place、Put in Drawer、Sweep Trash、Slice Sausage）。此外，还对比了 Sim-only、Real-only、Low-data Real、RIGVid 等基线数据集。

**📈 对比分析**

在七项任务上与 Sim-only、Real-only、Low-data Real、RIGVid 等对照实验，发现：① 仅用验证合成数据的策略在多数任务上接近 Real-only 的成功率；② 在 OOD 场景中，验证合成数据训练的策略显著优于 Real-only；③ 加入少量真实示例的混合训练可进一步提升性能，甚至超过 50 条真实轨迹的 baseline；④ 证明验证步骤对保持高成功率至关重要。

**⚠️ 局限性**

局限性包括：① 仍需收集少量真实初始图像；② 仿真重建对复杂物体或非刚体的精准度有限；③ 验证回放可能过滤掉部分可用但不完美的轨迹，导致数据浪费；④ 对极端长时序或高度非线性动力学的任务，生成与验证的效果尚未充分验证；⑤ 需要额外的模型微调与后处理步骤，增加工程复杂度。

---

## 362. SOAR: Source-Aware Reinforcement Learning with Precise Credit Assignment for Search Agents

**arXiv ID:** 2610.13157 | [PDF](https://arxiv.org/pdf/2610.13157v1)

**作者:** Shengze Huang `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种源感知强化学习框架SOAR，改进检索式问答代理以降低证据伪造。

**💡 创新点**

通过将内部知识与外部检索显式分离并引入源一致奖励，实现精准信用分配，解决传统单轨迹的证据来源歧义。

**🔧 技术方法**

使用GRPO强化学习、源感知工作流、源一致奖励、多维奖励（答案、格式、源一致）、Qwen-2.5-3B/7B LLM、E5检索与Wiki2018。

**📊 数据集**

单跳NQ、TriviaQA、PopQA，多跳HotpotQA、2Wiki、Musique、Bamboogle，附加ASQA、WebWalkerQA进行更广泛评测。

**📈 对比分析**

与Direct Inference、Naive RAG、Search-R1、AutoRefine、CoDA、IKEA、AdaSearch等基线在EM上对比，SOAR在3B平均0.414、7B平均0.448，单跳+多跳均优于基线，证据伪造率下降74.8%，性能最优。

**⚠️ 局限性**

依赖静态Wiki知识库，对开放Web噪声、恶意信息的鲁棒性有限，需进一步验证在更高噪声与非结构化来源下的效果。

---

## 363. Equitable Subpartitionability Does Not Characterize Box-TDI Polyhedra

**arXiv ID:** 2610.13244 | [PDF](https://arxiv.org/pdf/2610.13244v1)

**作者:** Emiliano Lancini `[一作]` (Université Paris Dauphine PSL), Francesco Pisanu `[通讯]` (Ca’ Foscari University of Venice)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了箱子-完全对偶可整性（box‑TDI）多面体不一定能用公平子分区性（ESP）描述，并给出四维整数多面体的反例以及一个可推广到任意维的构造方法。

**💡 创新点**

创新点在于将ESP性质等价地表述为多面体顶点的法线锥必须是“平衡”锥，并利用这一几何性质构造反例，首次展示存在无限族box‑TDI多面体缺乏ESP描述。

**🔧 技术方法**

使用了线性系统ESP定义、平衡锥概念、Hilbert基、法线锥、整数对偶性与完全可整性理论、完全等价矩阵（TE/TU）等线性代数与整数规划技术。

**📊 数据集**

无数据集，本文完全基于理论证明与构造。

**📈 对比分析**

本文未做实验性对比，仅通过理论构造和证明展示box‑TDI与ESP之间的关系；没有实验性能指标。

**⚠️ 局限性**

局限性包括：检测法线锥是否平衡的算法复杂度未知（推测为Co‑NP‑complete）；研究范围限定在包含在非负正交域的多面体，未探讨更一般情形。

---

## 364. FPD-Drive: Future Policy Distillation for End-to-End Autonomous Driving

**arXiv ID:** 2610.13232 | [PDF](https://arxiv.org/pdf/2610.13232v1)

**作者:** Li-Heng Chen `[一作]` (NIO), Shaoqing Ren `[通讯]` (NIO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种利用记录的未来信息的教师模型，对学生规划器进行多目标蒸馏，并结合安全感知分数校正的端到端驾驶框架

**💡 创新点**

创新点在于：1）将未来观察作为教师的特权信息，生成多样化、高质量的轨迹目标；2）通过多目标蒸馏将多条目标同时传给学生，打破单目标限制；3）设计安全感知分数校正，纠正安全分数中的高危候选与安全候选的排序误差；4）在部署时仅需学生，无需未来观察或教师推理

**🔧 技术方法**

使用了视觉编码器+Transformer解码器的粗细规划器、残差Diffusion Transformer进行轨迹细化、基于PDM的轨迹评分器、全局场景编码器融合未来观察、教师与学生共享架构、以及多目标匹配与分数校正的损失函数

**📊 数据集**

主要数据集为NAVSIM v1、v2用于离线评估；HUGSIM用于零射程闭环评估；训练时还使用SimScale等模拟数据；教师训练使用记录的未来摄像头帧

**📈 对比分析**

与众多基线（DrivoR、DriveZero、ChainFlow-VLA等）比较，在NAVSIM v1上PDMS达94.9/95.5，v2上EPDMS达58.1/58.1，均为SOTA；在HUGSIM的闭环评估中，HD-Score 45.1、RC 55.3，表现优于同类方法

**⚠️ 局限性**

局限性包括：1）教师训练需要记录的未来观测，无法在真实部署中使用；2）多目标蒸馏与分数校正增加训练复杂度与计算开销；3）在极端或未见场景下的泛化仍待进一步验证；4）对不同传感器配置的适应性未充分评估

---

## 365. Deep Learning Approach for RealTime Contraction Detection in Laboring Dairy Cattle

**arXiv ID:** 2610.13242 | [PDF](https://arxiv.org/pdf/2610.13242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 366. Does Learning Recurrent Networks Matter in Partially Observable RL?

**arXiv ID:** 2610.13197 | [PDF](https://arxiv.org/pdf/2610.13197v1)

**作者:** Zekang Wang `[一作]` (University of Macau), Gaojie Jin `[通讯]` (University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种将冻结的循环网络作为记忆模块的RL架构，仅训练读出层和特征混合器。

**💡 创新点**

创新点在于证明冻结的GRU/​LRU能在POMDP中保持长期记忆，并且大幅减少可训练参数。

**🔧 技术方法**

使用PQNs、PPO、CNN编码器以及MIXER（Linear+LayerNorm+LeakyReLU）等技术。

**📊 数据集**

在MemoryChain、POPGym、MuJoCo Playground、POPGym Arcade等多种部分可观测任务上进行实验。

**📈 对比分析**

与可训练的GRU/​LRU以及无记忆的MLP对比，冻结记忆在多数任务上实现相近甚至更优的返回，并保持约40‑70%参数减少。

**⚠️ 局限性**

局限性包括实验覆盖范围有限、未进行大规模消融实验、理论假设在实际训练中可能不完全满足。

---

## 367. Scaling Agentic Data for Long-Horizon Terminal Intelligence

**arXiv ID:** 2610.13166 | [PDF](https://arxiv.org/pdf/2610.13166v1)

**作者:** Shun Zou `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套名为TerminalHorizon的全自动化长时序终端任务合成引擎，能够生成具有丰富代理行为模式（探索、策略修订、验证等）的长时序交互轨迹，并基于这些轨迹对大型语言模型进行微调；

**💡 创新点**

创新点在于：① 通过挑战引导的重构和递进式扩展两阶段方法，确保任务设计保留并触发代理的长期问题解决行为；② 将真实职业任务映射为可执行终端环境，形成高质量、多领域长时序数据；③ 在少量轨迹下显著提升模型在终端任务、生产力与通用推理等12个基准上的性能。

**🔧 技术方法**

使用技术包括：大型语言模型（Qwen3.5-27B）、自动化任务构造与验证流程、独立挑战者进行轨迹收集、DeepSpeed ZeRO‑3训练、fused AdamW、BFloat16 训练；

**📊 数据集**

数据集为TerminalHorizon‑3K，包含1,535个任务环境、2,852条长时序交互轨迹（平均52.8步），涵盖25个专业领域。

**📈 对比分析**

与基线模型（原Qwen3.5、Terminal‑Universe）以及多种终端任务基准（Terminal‑Bench、AutomationBench、VitaBench、τ²‑bench、MATH‑500、GSM8K、AIME等）进行对比，TerminalHorizon‑27B在所有12项评测中均超越基线，提升幅度从8.7%至14.2%不等。

**⚠️ 局限性**

局限性包括：数据规模仍相对有限，模型仍依赖于合成环境的真实性；对更复杂的跨领域迁移与人类评价仍有待验证；生成轨迹的质量受教师模型选择影响。

---

## 368. SharP: Soft and Hard Prompt-Guided Augmentation with LLM for Low Resource Fake News Detection

**arXiv ID:** 2610.13207 | [PDF](https://arxiv.org/pdf/2610.13207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 369. What Do Audio-Language Judges of Generated Music Listen For? Four Probes with Controls from Expert Ratings

**arXiv ID:** 2610.13247 | [PDF](https://arxiv.org/pdf/2610.13247v1)

**作者:** Mateusz Modrzejewski `[一作]` (Warsaw University of Technology), Tomasz Radzikowski `[通讯]` (Warsaw University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出四种探针评估开放式音频-语言模型在音乐评判中的依赖性，并在 MusicEval 与 SongEval 数据集上验证五个评判器。

**💡 创新点**

创新点是将音频、提示、顺序和措辞四个维度分离，提供可复现的评审方法来检测模型是否真正依据音乐内容评分。

**🔧 技术方法**

采用无音频、相同提示替换、顺序交换和重述范围等探针，并用 Spearman 相关与 Bootstrap 区间进行统计对比。

**📊 数据集**

使用 MusicEval（2,748 片段、14 名专家评分）和 SongEval（2,399 片段）作为实验数据集。

**📈 对比分析**

通过 Spearman 相关性与专家评分比较，结果显示有音频时评判器在提示一致性上显著优于无音频，且措辞变化可与模型切换产生相似影响，但整体性能仍低于专家。

**⚠️ 局限性**

局限性包括仅测试开放模型、受 FP8 量化影响、片段长度限制、部分探针在 SongEval 上不可用，以及提示可能出现在训练数据中。

---

## 370. Destination-Conditioned Adaptation with Semantic-Geometric Alignment for Generative Inbetweening

**arXiv ID:** 2610.13187 | [PDF](https://arxiv.org/pdf/2610.13187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 371. 3DRAG: 3D-Native Object Drag Editing with Generative Models

**arXiv ID:** 2610.13216 | [PDF](https://arxiv.org/pdf/2610.13216v1)

**作者:** Kuixiang Shao `[一作]` (ShanghaiTech University), Jingyi Yu `[通讯]` (ShanghaiTech University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种无需SDS优化的3D原生拖拽编辑框架3DRAG。

**💡 创新点**

创新点包括3D位移场与占用感知赢家归属策略、以及针对注意力的logits级令牌融合技术。

**🔧 技术方法**

利用TRELLIS 3D生成模型、弹性位移理论、OA‑WTA、logits级令牌融合、精确ODE逆推等技术。

**📊 数据集**

构建了100个多样化3D物体及其拖拽指令的新数据集。

**📈 对比分析**

与基准的Gaussian、mesh、APAP/ARAP等方法对比，3DRAG在拖拽精度、未编辑区域保留和整体3D质量上均达到或超过对手，且显著加速。

**⚠️ 局限性**

局限在于色差偶尔出现以及对细小结构的拖拽精度受限于64³稀疏体素分辨率。

---

## 372. Monarch-MoE: Structured Experts for Multimodal Continual Instruction Tuning

**arXiv ID:** 2610.13215 | [PDF](https://arxiv.org/pdf/2610.13215v1)

**作者:** Xi Zhang `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Yongjun Xu `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为Monarch‑MoE的结构化专家模型，用于多模态持续指令微调，旨在减少新任务学习时对已学知识的灾难性遗忘。

**💡 创新点**

创新点在于将LoRA专家中的密集低秩因子替换为由固定置换连接的块对角矩阵，形成块级梯度隔离，既保证跨块通信，又限制单块更新对输出空间的全局影响。

**🔧 技术方法**

使用的技术包括Monarch矩阵分解、多专家MoE结构、批量化专家计算、前向融合操作、显式批量后向传播，以及在单张24 GB RTX 4090上实现的高效推理与训练。

**📊 数据集**

实验数据集基于LLaVA‑1.5‑7B模型，并使用MCITlib协议下的MLLM‑DCL（五个连续域）和UCIT（六个任务）两个持续学习基准。

**📈 对比分析**

与传统LoRA、MoELoRA、DISCO、O‑LoRA等方法对比，Monarch‑MoE在保持或提升最终性能的同时显著降低了第一任务遗忘（例如在DCL上从2.51pp降至0.52pp），并将适配器参数量减少约75‑50%，单GPU训练和推理时间保持竞争力。

**⚠️ 局限性**

局限性包括方法相关的获取‑保持权衡、仅在单一模型与固定任务顺序下验证、种子复现仅覆盖两任务、未与全量参数匹配的尺度控制以及无法保证最终性能提升，需在更广泛的后端与更长任务序列中进一步评估。

---

## 373. JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models

**arXiv ID:** 2610.13194 | [PDF](https://arxiv.org/pdf/2610.13194v1)

**作者:** Mengdi Liu `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xilin Chen `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出JANUS框架，实现自主研究的Predict‑Verify‑Evolve循环，利用可共同演化的世界模型预估候选方案价值并在执行前筛选、细化；通过演化研究图和多层次技能抽象实现研究代理与世界模型的协同进化。

**💡 创新点**

创新点在于：①将未来预估与过去反思统一为Predict‑Verify‑Evolve循环；②构建演化研究图支持分支探索与经验归纳；③使用可共进化的世界模型进行多维价值评估（价值提升、信息增益、探索效用）；④通过层级技能和协同协议实现经验到技能的自动抽象与共享。

**🔧 技术方法**

核心技术包括：可演化的图结构研究代理、基于LLM或其他大模型的世界模型预测、稀疏对比式评估（Bradley‑Terry模型）、自适应探索权重调整、经验驱动的技能抽象与协调机制。

**📊 数据集**

实验数据集为MLS‑Bench‑Lite以及真实科研任务（算法优化、系统设计与科学发现），覆盖多种模型与任务。

**📈 对比分析**

与多种最新基线（如AIDE、MLE‑STAR、AI Scientist 等）进行对比，JANUS在演化效率（同样预算下更快的性能提升）和演化上限（更高最终性能）上均取得显著优势。

**⚠️ 局限性**

局限性包括：依赖高质量的世界模型预测，若模型偏差大可能误导决策；系统仍需昂贵的执行回报来更新模型；对极大搜索空间的可扩展性和多任务迁移效果待进一步验证。

---

## 374. From Observation to Intervention: Identifiability of Latent User Bias in Human-AI Interaction

**arXiv ID:** 2610.13185 | [PDF](https://arxiv.org/pdf/2610.13185v1)

**作者:** Lucas Biechy `[一作]` (Inria), Tianyi Qiu `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在存在认知偏差（如确认偏差）的用户行为下，AI系统如何区分真实偏好与偏差行为；提出主动干预（如强制零奖励）能恢复可识别性；

**💡 创新点**

证明了被动观测下的不可识别性与确认陷阱，给出了主动干预下可恢复可识别性的理论上限m_δ；

**🔧 技术方法**

采用多臂赌博机模型、Thompson Sampling、贝塔先验、认知偏差噪声核、主动零奖励干预；

**📊 数据集**

实验使用合成Bernoulli多臂赌博机数据，并在LLM观测器上验证；

**📈 对比分析**

与被动分类器（GBT、LogReg）对比，主动干预下识别率可达0.92 AUROC，主动干预需要16–27轮即可达到高TV分离；

**⚠️ 局限性**

局限性包括：模型假设简化，真实人类偏差更复杂；主动干预成本高，可能导致用户流失；无法直接纠正偏差，只是识别；

---

## 375. High-Throughput Generative Video Super-Resolution on a Single Consumer GPU

**arXiv ID:** 2610.13179 | [PDF](https://arxiv.org/pdf/2610.13179v1)

**作者:** Georgios Panaretou `[一作]` (Ethical AI Novelties), Sotirios Chatzis `[通讯]` (Cyprus University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个零样本生成式视频超分辨率框架，利用预训练的一步扩散模型对关键帧进行空间增强，使用轻量级流引导插值填补中间帧，并通过零样本混合注意力替换高分辨率层的全局注意力以提升速度；同时加入运动门控来避免插值失败；整个系统在单块RTX 4090上即可实现720p 32.66 FPS。

**💡 创新点**

创新点主要包括：①将空间增强与时空稳定化完全解耦，采用一键式扩散关键帧生成与插值填充；②提出零样本混合注意力（用邻域注意力替换部分全局注意力），在保持视觉质量的同时显著降低计算；③引入运动门控机制，基于光流阈值自动决定是否回退到生成模型；④整个系统无需端到端微调，完全基于预训练组件；⑤通过关键帧间隔调节速度-质量平衡，实现实用化。

**🔧 技术方法**

技术栈包括：一步扩散超分模型AdcSR、RIFE实时插值网络、邻域注意力（NATTEN）、光流门控（Farneback光流阈值15）、批量推理与时序调度、FlashAttention/自定义NATTEN实现。

**📊 数据集**

在四大公开基准上进行评测：UDM10、SPMCS、REDS4、Vid4。

**📈 对比分析**

与RealBasicVSR、RVRT、DLoRAL、DiffIR2VR‑Zero、SeedVR2等多种基线在PSNR/SSIM/LPIPS、BRISQUE/MUSIQ/MANIQA、E_warp/tOF等指标上对比；系统在HD（1280×720）下单批次内存仅2 GB，FPS从3.04到32.66（K=3、B=8），相较于Diffusion基线提升3–4倍；在1080p及4K时可获得高达44.7%速度提升。

**⚠️ 局限性**

局限性包括：①关键帧间隔过大时插值会失真，导致视觉质量下降；②运动门控会在高运动视频中频繁回退到生成模型，显著降低帧率；③系统延迟高（B·K帧缓存），不适合低延迟交互式使用；④未实现端到端的时空一致性训练，仍存在微小的时间漂移；⑤在极端高分辨率/大运动场景下仍受限于预训练模型的能力。

---

## 376. How Should We Evaluate Naturalness for Synthetic Speech with a Foreign Accent?

**arXiv ID:** 2610.13175 | [PDF](https://arxiv.org/pdf/2610.13175v1)

**作者:** Kentaro Onda `[一作]` (University of Tokyo), Nobuaki Minematsu `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过对来自ERJ语料库的日本口音英语与美国英语进行分析-重合成，构建四种合成系统（无重合成、BigVGAN、Encodec、WORLD）的480个样本，并在四种自然度评估指令（自然度、人类相似度、以母语为标准、以非母语为标准）下，使用320名母语为英语或日语的评测者进行大规模主观听力测试，收集总计15,360个五点评分。

**💡 创新点**

创新点在于：①首次系统性探讨外来口音对人工合成语音自然度评价的影响；②将自然度拆解为人类相似度与上下文适配两类，并通过四种评估指令验证其对口音偏差的抑制效果；③揭示评测者的语言背景（尤其对目标口音的熟悉度）在评价中的显著作用。

**🔧 技术方法**

采用的技术包括：语音分析-重合成（使用Sidon进行降噪）、多种重合成模型（BigVGAN、Encodec、WORLD）、Crowdsource平台（Prolific、Lancers）进行听力测试、统计分析（Spearman相关、Wilcoxon符号秩检验、Holm校正）以及与三种自动MOS预测模型（UTMOS、UTMOSv2、SSL-MOS）的对比。

**📊 数据集**

使用的数据集为ERJ语料库，该语料库包含20名日语学习者的英语录音（并附有母语为英语的熟练度评分）和4名美国英语母语者的录音，随后经过降噪后通过四种重合成方式生成480个评测样本。

**📈 对比分析**

比较方法为：在四种评估指令下统计不同重合成系统的平均MOS、评测者分组（英语母语 vs 日语母语）以及口音熟练度与MOS的Spearman相关；结果显示在人类相似度指令下，口音影响最小，系统间差异最明显；自动MOS预测模型与人类相似度评价的相关性最高，表明现有预测器在一定程度上与人类相似度评价一致，但仍存在对口音偏差的敏感性。

**⚠️ 局限性**

局限性在于：实验采用控制的分析-重合成设置，口音程度在各系统间基本保持一致，未能覆盖真实TTS系统中口音与合成缺陷交织的复杂情况；评测者主要为非专业听众，对目标口音的熟悉度参差；因此，结论对真正的多样化口音合成系统的适用性仍需进一步验证。

---

## 377. ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation

**arXiv ID:** 2610.13225 | [PDF](https://arxiv.org/pdf/2610.13225v1)

**作者:** Xiaohuan Jia `[一作]` (Inner Mongolia University), Shaohua Fan `[通讯]` (Inner Mongolia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Rectified Statistical Flow (ReSFlow)，一种在概率单纯形的球面几何上进行流匹配与直线化的离散数据生成框架，使得可以通过单步欧拉/指数映射完成高质量的类别生成。

**💡 创新点**

创新点：1）利用平方根映射将概率分布映射到正半球面，使Fisher–Rao度量等价于球面度量；2）在球面上用球面线性插值（slerp）取代欧几里得直线，天然得到最短几何轨迹；3）在该球面空间中进行迭代重流（reflow），将混合的交叉几何轨迹逐步收敛为非交叉、常速的geodesic；4）最终通过球面指数映射实现一次性生成，从而大幅降低推理成本。

**🔧 技术方法**

主要技术：平方根参数化、球面几何与指数/对数映射、球面流匹配（Spherical Flow Matching）、球面重流（Spherical Reflow）、一次性欧拉/指数采样、连续时间变分法与ODE训练。

**📊 数据集**

使用的数据集包括：1）二维 Swiss-roll（嵌入单纯形的低维几何验证）；2）二值化 MNIST（图像分类任务）；3）Text8字符级语言模型（离散序列任务）；4）Promoter DNA序列设计（长序列生物学任务）。

**📈 对比分析**

与 LinearFM、SFM、R‑LFM、ReDi 等基线进行对比，评估指标为 FID（图像）、BPC（文本）、SP‑MSE（序列）以及采样时间。结果表明：ReSFlow 在单步采样时 FID 与 BPC 均优于其他基线；在全 ODE 采样下性能与 SFM 相当；最显著的是在单步模式下仅需 1 次函数评估即可实现与多步（数百步）相近甚至更好的生成质量，且推理速度提升数百倍。

**⚠️ 局限性**

局限性：①训练阶段仍需要多轮迭代重流，训练成本较高；②仅在有限的离散分类任务上验证，尚未探索更大词表或更长序列的可扩展性；③方法依赖于概率单纯形的球面映射，对非标准离散结构（如图结构、组装约束）可能需要进一步改造。

---

## 378. olBlend: Language-Guided Molecular Blend Retrieval with Mixture Odor Prediction

**arXiv ID:** 2610.13188 | [PDF](https://arxiv.org/pdf/2610.13188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 379. Automated Detection and Encoding of Cartographic Signs in Historical Maps

**arXiv ID:** 2610.13224 | [PDF](https://arxiv.org/pdf/2610.13224v1)

**作者:** Remi Petitpierre `[一作]` `[通讯]` (Swiss Federal Institute of Technology), Remi Petitpierre (Swiss Federal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个两阶段框架，用于检测历史地图中的图形符号并生成可解释的嵌入表示。

**💡 创新点**

创新点在于创建公开的 CaSiDD 符号检测基准，结合轻量化前景分割和适配器，提升符号嵌入在语义和风格上的可解释性与泛化能力。

**🔧 技术方法**

采用 YOLOv10‑X 进行检测，改进的 RADIOv4 backbone 与自适应两层 FC 适配器生成嵌入，并使用 ResNet‑18 U‑Net 做前景分割。

**📊 数据集**

使用 CaSiDD（18,750 个标注符号，796 张地图）进行训练与评估，及 ADHOC 历史地图数据用于社区归因实验。

**📈 对比分析**

与未处理、灰度+前景分割等基线对比，适配器模型在符号分类、社区归因以及颜色/形状解释性上均提升，分类精度最高达 87%/87%，ROC‑AUC 最高 75.4%。

**⚠️ 局限性**

局限在于检测召回率低、符号定位受主观边界影响、对极少量样本类别表现差、以及仍受背景与数字化伪影的影响。

---

## 380. Compact Quantitative Equational Theories as Monad Restrictions

**arXiv ID:** 2610.13201 | [PDF](https://arxiv.org/pdf/2610.13201v1)

**作者:** Matteo Mio `[一作]` (CNRS and ENS de Lyon), Colin Riba `[通讯]` (ENS de Lyon)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了量化代数可压缩定理的范畴学表述，并证明其与逻辑定义等价。

**💡 创新点**

将可压缩性从纯逻辑视角转化为单子限制的范畴学性质，首次给出该等价性。

**🔧 技术方法**

使用单子、Eilenberg–Moore范畴、单子限制、Horn逻辑及紧致性定理等理论工具。

**📊 数据集**

本工作为纯理论研究，无实验数据集。

**📈 对比分析**

通过构造单子同构与自然变换，证明等价关系；未进行性能评估。

**⚠️ 局限性**

尚未证明单子限制与包含关系的逆向蕴含是否成立，仍为开放问题。

---

## 381. Prediction-Informed Landing Window Planning for UAV Landing on USV in High Sea States

**arXiv ID:** 2610.13190 | [PDF](https://arxiv.org/pdf/2610.13190v1)

**作者:** Kaizhen Li `[一作]` (Shanghai University), Xiaowei Li `[通讯]` (Shanghai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在高海况下实现无人机对无人水面艇的自动着陆，提出基于预训练姿态预测的分层控制框架。

**💡 创新点**

首次将显式的USV俯仰滚转预测与高层着陆窗口规划结合，提升着陆时序感知。

**🔧 技术方法**

使用LSTM编码‑解码预测器、PPO强化学习、高层观测融合预测序列、MuJoCo仿真、SE(3)低层控制器。

**📊 数据集**

收集的真实海况USV运动数据（滚转、俯仰），按7:3划分为训练集与测试集，并在MuJoCo中重放。

**📈 对比分析**

与Informer模型对比，LSTM在滚转与俯仰预测上的RMSE分别为0.997/1.274、MAE分别为0.617/0.833；强化学习策略在实时着陆窗口选择上表现出更稳定、提前调节的垂直速度命令。

**⚠️ 局限性**

仅考虑滚转与俯仰预测，未包含波高、位移等其他运动变量；数据量有限；尚未在真实无人机‑水面艇平台上完成验证。

---

## 382. CompactSplat: Spatially Adaptive Gaussian Distribution for Feedforward Scene Reconstruction

**arXiv ID:** 2610.13177 | [PDF](https://arxiv.org/pdf/2610.13177v1)

**作者:** Bing He `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种前向式3D高斯撒点框架CompactSplat，能够根据图像纹理自适应地分配3D高斯原语，实现稀疏视角下的高质量3D重建

**💡 创新点**

通过纹理感知划分器（TAP）和层次金字塔采样器（HPS）实现内容感知的高斯分配，消除像素级冗余并支持推理时动态稀疏度控制

**🔧 技术方法**

双路径编码器、U-Net多尺度结构、纹理感知四叉树划分、层次金字塔采样、光度与LPIPS联合损失、两阶段课程学习策略

**📊 数据集**

RealEstate10K、ACID、DTU、DL3DV、ScanNet等室内外场景数据集

**📈 对比分析**

与PixelSplat、HiSplat、MVSplat、TranSplat、DepthSplat、SurfSplat、YonoSplat等基线比较，在PSNR/SSIM/LPIPS、HRRC以及Gaussian数量/每像素Gaussian（GPP）等指标上实现与或优于现有方法，同时大幅降低Gaussian数量，且可以无额外训练实时调节质量与存储

**⚠️ 局限性**

对纹理变化极端或几何细节极其稀疏的场景仍可能因过度合并导致边缘细节损失，且对动态场景、无标定输入的适用性尚未验证

---

## 383. Understanding Verification Vulnerabilities in Cross-chain Bridges: An Empirical Study

**arXiv ID:** 2610.13170 | [PDF](https://arxiv.org/pdf/2610.13170v1)

**作者:** Yuying Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yi Sun `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对588份跨链桥审计报告的系统分析，识别并归纳出11种跨链验证漏洞（CVV）模式，形成5大类，并基于这些模式构建了830个漏洞实例的数据集，涵盖786个合约文件。

**💡 创新点**

首次对跨链验证漏洞进行系统性实证研究，提出了面向验证语义的CVV模式与分类体系，并公开了规模化的数据集，为后续工具与模型评估提供基准。

**🔧 技术方法**

采用人工筛选与开放卡片分类构建漏洞标签，利用统计分析描述漏洞分布；随后使用四款大型语言模型（GPT‑5.5、Composer‑2.5、Opus‑4.8、DeepSeek‑V4‑Pro）实现多标签检测，并与传统静态分析工具（Slither、SmartCheck 等）进行对比。

**📊 数据集**

使用588份审计报告、36个跨链桥项目、830个CVV实例及对应的786个合约文件，构成公开数据集 CVV‑Dataset（可从 https://anonymous.4open.science/r/CVV-Dataset/ 访问）。

**📈 对比分析**

通过宏平均 F1、精确率与召回率对传统工具与LLM进行性能评估。LLM 中 Composer‑2.5 的宏平均 F1 达到 51.2%，仅次于基线 29.3%；其余三款 LLM 低于基线；传统工具覆盖率极低，几乎不检测任何 CVV 模式。

**⚠️ 局限性**

局限性包括：缺乏跨组件完整上下文导致误报与漏报；跨链验证语义推理不足；现有工具和 LLM 对 CVV 的检测覆盖率极低；数据集主要来自公开审计，可能不具备对私有桥的代表性。

---

## 384. Taking Tokenization off the Host for Agentic LLM Serving

**arXiv ID:** 2610.13325 | [PDF](https://arxiv.org/pdf/2610.13325v1)

**作者:** Mohammad Siavashi `[一作]` (KTH Royal Institute of Technology), Dejan Kostić `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个GPU端完整编码器 GraphTok，可在CUDA图中一次性完成从预分词、BPE合并到token-ID生成的整个过程，从而将Tokenization从CPU迁移到GPU。

**💡 创新点**

创新点在于：①实现了与多种Tokenizer（GPT‑2、Llama 3、Qwen 2.5、DeepSeek‑V3、Gemma 3）完全一致的token序列；②利用并行前缀检测与warp级BPE合并，在CUDA图内实现可变长度、可重放的token化；③提供CPU路由与自适应调度，适配多任务前端与GPU集群。

**🔧 技术方法**

使用CUDA图、warp级并行BPE合并、分段扫描、共享内存并行搜索、固定容量图重放以及CPU–GPU协作的多线程调度。

**📊 数据集**

在三类文本语料（English books、Python 代码、35语言多语言聊天）上评测，从128 tokens到1 M GPT‑2 tokens；并使用BurstGPT生产轨迹进行前端压力测试。

**📈 对比分析**

与现有CPU（tiktoken、hugging‑face、gigatoken）和GPU（TokTier、GPUTOK） tokenizer 对比：单请求下在1 M token时，比最强CPU tokenizer快 7.9–12.9×，比TokTier GPU 路径快 2.6–18.1×；在vLLM与NVIDIA Dynamo 前端中，tokenization 成为吞吐瓶颈时可提升 GPU 数量 1.75–2.5×，P99 延迟降低 64×。

**⚠️ 局限性**

限制主要在：①对不支持的Tokenizer配置需手工裁剪或回退；②CUDA图固定容量导致极长单前缀需额外溢出路径，导致超大文本的单个请求延迟增大；③仅针对当前主流Tokenization方案，非BPE或基于字节的 tokenizer 需重新实现。

---

## 385. Quantization-Tolerant Gradient Auditing for Active Inference-Attack Detection in Federated Learning

**arXiv ID:** 2610.13311 | [PDF](https://arxiv.org/pdf/2610.13311v1)

**作者:** Michael Kuehne-Schlinkert `[一作]` `[通讯]` (Katulu GmbH), Michael Kuehne-Schlinkert (Katulu GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在跨机FL中，服务器通过挑战-响应机制，向每轮客户端发送秘密挑战批，收集客户端计算该批梯度的校验和和耗时，以检测模型变异的主动攻击和被动推断攻击。

**💡 创新点**

提出量化容忍的梯度校验和（fingerprint），能够在异构硬件上抵消浮点漂移，同时对模型变异保持高灵敏度；以及将校验和与时延信号结合的多维判别方案。

**🔧 技术方法**

挑战批生成使用基于服务器秘密种子和SHA-256的确定性采样；梯度校验和通过相对量化、分块哈希和容忍阈值计算；时延检测采用基于中位数和MAD的改进z-score。

**📊 数据集**

公开数据集CIFAR-10用于实验；工业案例使用15K条二分类缺陷数据；此外在Texas100基准上评估差分隐私效果。

**📈 对比分析**

在CIFAR-10上，主动攻击（GAN、梯度上升）100%检测率、0%误报；在工业数据上同样达到100%检测率，且服务器成本仅32ms、4.35kB；被动攻击仅通过时延检测，鲁棒性差，误报随网络抖动显著上升。

**⚠️ 局限性**

局限包括：时延信号对被动推断的检测不稳定；无法检测“干净副本”攻击（客户端在副本上完成推断）；对梯度正交攻击无效；需要额外机制（可信执行或零知识证明）来约束客户端实际计算。

---

## 386. Sonic-AMG: Streaming Human Motion Generation from Spatial Audio

**arXiv ID:** 2610.13300 | [PDF](https://arxiv.org/pdf/2610.13300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 387. Decoding Register Tokens Preserves Global Scene Details

**arXiv ID:** 2610.13299 | [PDF](https://arxiv.org/pdf/2610.13299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 388. MORAD: Reinforcement Learning with Multi-Objective Rewards for RNA Inverse Design

**arXiv ID:** 2610.13305 | [PDF](https://arxiv.org/pdf/2610.13305v1)

**作者:** Jixin Tang `[一作]` (Fudan University), Jun Zhao `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了MORAD——一种在线强化学习框架，用于后训练RNA扩散模型，使其能在给定三维骨架下生成兼容的核苷酸序列，并通过共享策略在多个目标上学习；

**💡 创新点**

创新点包括：1）共享策略在线后训练，实现跨骨架的可复用模型；2）将六个不同尺度的测量（三级几何、配对一致性、群集行为、序列组成）映射为有界可取性分数并通过几何平均融合；3）采用组相对奖励归一化和前向扩散学习更新；

**🔧 技术方法**

使用技术包括：基于RIDE的结构条件化扩散模型、前向扩散RL（DiffusionNFT风格的正负分支）、多目标奖励设计与可取性映射、组相对奖励标准化、AdamW优化器；

**📊 数据集**

数据集为527个训练骨架与153个评估骨架（71来自RNA3DB、82来自PDB），使用RhoFold+、RNAfold、ViennaRNA等工具进行三级结构、配对、能量与熵评估；

**📈 对比分析**

与RiboDiffusion、gRNAde、RDesign以及预训练RIDE在同一评估集上比较，MORAD在大部分指标（10/11）均优于结构仅奖励版本，并在GDT‑TS提升16%、配对MCC提升19%、归一化集束缺陷下降27%；

**⚠️ 局限性**

局限性在于仅验证在单一10.2M RIDE模型上；奖励仅覆盖热力学平衡属性，未考虑动力学或功能目标；评价依赖RhoFold+预测器，缺乏实验验证。

---

## 389. PAGE: Parts-Aware Guidance for Co-Speech Gesture Portrait Video Generation

**arXiv ID:** 2610.13301 | [PDF](https://arxiv.org/pdf/2610.13301v1)

**作者:** Yabiao Wang `[一作]` (Zhejiang University), Ming-Hsuan Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了PAGE框架，利用分区域加权与一致性分类器引导，改进了音频驱动的人像视频生成质量和音视频同步。

**💡 创新点**

创新点在于：1）Parts‑Specific Reweighting（PSR）在训练时对嘴唇、眼睛、手等关键部位加大监督力度；2）Consistency Classifier Guidance（CCG）在推理时通过音频‑视频一致性判别器提供梯度，提升同步性；3）Differential Guidance（DG）实现面部与非面部区域独立引导，消除相互干扰。

**🔧 技术方法**

技术手段包括：扩散模型（如Wan2.1、SVD）作为主干；姿态关键点与语义分割生成区域权重；DINOv2为轻量化一致性分类器；音频特征通过wav2vec与时域对齐；在推理时采用梯度引导的DDIM/ODE求解。

**📊 数据集**

使用了现有的PATS数据集，并构建了首个大规模中文广播演讲数据集CAS（19,052段、800+说话人），用于训练与评估。

**📈 对比分析**

与S2G、EchoMimic、OmniAvatar、Hallo3等方法对比，PAGE在FVD、BAS、FGD、Div等指标上均实现了显著提升，尤其在CAS上FVD下降约35%，BAS提升0.003，说明生成视频更真实、更同步。

**⚠️ 局限性**

局限性包括：1）区域划分粗糙，无法细化到单指关节或微表情；2）假设固定摄像机视角，难以处理动态视角或交互场景；3）在极端分布偏移（如卡通化人像）时负样本覆盖不足，可能导致一致性判别失效。

---

## 390. Recovering Irreducible Three-Body Interactions with Lagrangian Neural Networks

**arXiv ID:** 2610.13292 | [PDF](https://arxiv.org/pdf/2610.13292v1)

**作者:** Sena Kalabalik `[一作]` (Bogazici University), Arkadas Ozakin `[通讯]` (Bogazici University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在Lagrangian神经网络中预设精确的动能项，只学习势能函数，成功从三体轨迹数据中恢复了不可分解的Axilrod–Teller三体相互作用；

**💡 创新点**

创新点在于：①利用物理结构化的LNN和定向采样策略，将不可加的三体相互作用从运动数据中提取出来；②通过符号回归在网络学习的势能上直接重现了原始的三体公式；

**🔧 技术方法**

主要技术包括：Lagrangian神经网络（固定动能+可学习势能）、对称性增强的输入表征（距离与角度）、自适应采样（集中在三体信号强的配置）、以及基于PySR的符号回归；

**📊 数据集**

使用的“数据集”是基于Axilrod–Teller势能在二维三体系统中生成的加速和位置序列，包含对不同距离尺度和角度的采样；

**📈 对比分析**

比较方法：通过势能场、加速场、径向与角度依赖、轨迹滚动误差以及与LJ+AT混合系统的可识别性进行评估；在可辨识区（增强耦合）下，误差下降至≈10‑20%，并能精确重现角度符号；

**⚠️ 局限性**

局限性在于：①对高阶耦合强度依赖数据分布与模型误差阈值；②在实际物理中三体相互作用受短程重叠修正影响，模型仅适用于长程有效区；③假设恒定质量矩阵和仅位势项，限制了对更复杂动力学的推广。

---

## 391. Prime Accessible Deterministic Automata

**arXiv ID:** 2610.13274 | [PDF](https://arxiv.org/pdf/2610.13274v1)

**作者:** Shen Zhao `[一作]`, Chenghe Xu `[通讯]` (Central South University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

研究了可访问确定性自动机在有限与无限状态集合下的素性（不可分解性），并给出了完整的分类结果；

**💡 创新点**

提出了利用右同余分解恢复完整输入词自动机的理论，并证明非单字母字母表下任意非平凡自动机必不可素；

**🔧 技术方法**

主要采用可访问性、同步乘积、右同余与等价类的形式化分析技术；

**📊 数据集**

无数据集，属于纯理论研究；

**📈 对比分析**

无实验比较，主要通过数学证明展示性能与优缺点；

**⚠️ 局限性**

仅在有限字母表下给出完整分类，针对无限字母表的情形仅讨论单字母情况，其他更一般结构的自动机仍待研究。

---

## 392. Polynomial directed treewidth forces a large bramble with congestion two

**arXiv ID:** 2610.13270 | [PDF](https://arxiv.org/pdf/2610.13270v1)

**作者:** Victor A. Campos `[一作]` (Universidade Federal do Ceará), Ignasi Sau `[通讯]` (Université de Montpellier)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究有向图的树宽与“布兰姆”（bramble）结构的关系，证明了若有向图的有向树宽至少为 f_c(k)，则其包含一个大小至少 k、拥挤度不超过 c 的布兰姆。主要给出最小拥挤度 c=2 的函数 f_2(k) = k^96·log^23 k，并进一步给出 c=4,6 的多项式上界。

**💡 创新点**

创新点包括：①引入“walkage”概念，取代传统路径联通结构，允许路径自相交但保持低拥挤度；②提出“未纠结的有线 walkage”和“循环 walkage”，通过这些结构实现了从大型有向树宽到低拥挤度布兰姆的直接构造；③利用改进的局部概率（Lovász Local Lemma）实现构造的可多项式时间；④将之前只能得到拥挤度 8 的结果压缩到拥挤度 2，同时保持多项式上界。

**🔧 技术方法**

主要技术手段包括：有向树宽与布兰姆的近似对偶性；walkage 的构造与分解（如分段 walkage、线程化、循环化）；未纠结和循环 walkage 的可化简与拥挤度控制；稠密与稀疏情形的交叉图分析（利用 Kostochka 等的团团嵌入结果和 Lovász Local Lemma）；以及从路径系统到强系统的转化。

**📊 数据集**

本工作完全是理论研究，没有使用实验数据集；所有结果均为定理证明和构造算法。

**📈 对比分析**

与之前的工作相比，本文将原本极大（高度为 22 的指数塔）上界降为多项式，且拥挤度从 8 降到 2，保持了多项式时间的构造。对于 c=4,6 的上界分别为 k^72·log k, k^66·log^18 k，进一步表明即使拥挤度降低仍可取得可接受的多项式依赖。

**⚠️ 局限性**

限制包括：a) 仍然存在高阶多项式（k^96）导致实际应用中的规模受限；b) 对 f_8(k) 的上界并未改进，仍为 k^48·log^13 k；c) 构造方法虽然可多项式时间，但常数与隐藏因子较大，实际实现可能需要进一步优化；d) 只对一般有向图给出结果，特定图类（如半完全图）的更优上界仍未探讨。

---

## 393. Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers

**arXiv ID:** 2610.13261 | [PDF](https://arxiv.org/pdf/2610.13261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 394. Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents

**arXiv ID:** 2610.13267 | [PDF](https://arxiv.org/pdf/2610.13267v1)

**作者:** Lei Liu `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Northwestern Polytechnical University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 DREAM， 一个基于神经科学弱激活机制的内部内存修复框架，能够在不使用外部审计的情况下实现对 LLM 代理内存中被毒化条目的选择性功能性遗忘。

**💡 创新点**

创新点在于将弱激活诱导、三维激活强度诊断和离散时间动力学权重衰减相结合，形成训练免费、低成本、可在内存层面主动抑制恶意记忆的完整闭环。

**🔧 技术方法**

技术包括：零知识高斯扰动激活、三维激活强度（几何相似、竞争检索概率、语义一致能量）、自适应阈值异常检测、基于可解释的动态权重衰减与重巩固。

**📊 数据集**

实验使用 EHRAgent(eICU、MIMIC‑III)、ReAct‑StrategyQA 以及多智能体协作任务 MISINFOTASK，并在 GPT‑5.4‑mini 与 Claude‑Haiku‑4.5 上评估。

**📈 对比分析**

与 PPL、LLM Auditor、Distil Classifier、A‑MemGuard 等基线相比，DREAM 在 backdoor 与注入攻击下 Attack Success Rate 降低 95% 以上、ACC 提升 20–30%，且在运行时和 token 消耗上比 A‑MemGuard 快 2.13×、减少 37.7% token。

**⚠️ 局限性**

限制包括对高比例恶意记忆仍有一定容忍度、对极端参数配置的敏感性、以及在极端动态环境（如长时间持续学习）下的长期稳定性尚未充分验证。

---

## 395. CV-OPD: Contrastive Vision On-policy Distillation

**arXiv ID:** 2610.13264 | [PDF](https://arxiv.org/pdf/2610.13264v1)

**作者:** Haowei Wang `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CV-OPD，通过负视觉教师、对比损失和差异门控，改进 VLM 视觉推理的在策略蒸馏。

**💡 创新点**

利用负视觉样本与对比散度损失明确约束学生区分有价值与无关视觉证据，并通过 token 与样本级差异门控聚焦重要位置，解决传统 OPD 的视觉短路问题。

**🔧 技术方法**

On‑policy 蒸馏、Jensen–Shannon 散度、Contrastive Divergence Loss、Discrepancy‑Aware Gate、EMA 教师、视觉 crop 提议、token‑level 监督。

**📊 数据集**

在 6.2k 视觉推理样本上训练，评测于 V^*, ZoomBench、HR‑Bench 4K/8K、MME‑RealWorld EN/CN 等细粒度视觉基准。

**📈 对比分析**

与 Vision‑OPD 及公开/闭源 VLM 对比，CV‑OPD 在平均分提升约 2.3%，ZoomBench 提升约 2%，并在遮蔽测试中显著强化视觉依赖。

**⚠️ 局限性**

对 HR‑Bench 4K 仍不及基准模型；负样本采样和门控阈值对效果敏感，且在更大输入长度或多模态场景中的通用性尚待验证。

---

## 396. SheafPrune: Sheaf-Guided Graph Propagation for Visual Token Pruning

**arXiv ID:** 2610.13263 | [PDF](https://arxiv.org/pdf/2610.13263v1)

**作者:** Xiaoyang Ma `[一作]` (China Agricultural University), Hong Sun `[通讯]` (China Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种名为SheafPrune的视觉标记选择方法，旨在通过支持约束传播和节点度校正来优化多模态大语言模型的视觉标记处理。

**💡 创新点**

创新点在于结合了跨模态比较、支持约束传播和度校正排名，能够在固定预算下有效选择视觉标记，并且一个训练的检查点可以适用于多个保留预算。

**🔧 技术方法**

使用了基于图的选择技术，结合了sheaf几何学的思想，采用了模态特定的映射和稀疏的文本-视觉图进行相关性传递。

**📊 数据集**

使用了LLaVA-NeXT-8B等多个模型作为基础，并在十个基准测试上进行评估，包括ChartQA、DocVQA、TextVQA等。

**📈 对比分析**

与四个基础模型和十个基准测试的比较中，SheafPrune在28个基础-预算设置中取得了22个最佳保留率，表现出在中等到激进预算下的持续优势。在LLaVA-NeXT上，保留10%的视觉标记实现了5.15倍的净预填充加速和84.0%的端到端FLOPs减少。

**⚠️ 局限性**

限制在于选择的标记可能未能覆盖图像的所有重要区域，且在不同的任务和数据集上可能表现出不同的效果。

---

## 397. Beyond a Single Crop: Contextualized Embeddings for Player Re-Identification

**arXiv ID:** 2610.13259 | [PDF](https://arxiv.org/pdf/2610.13259v1)

**作者:** Ge Luo `[一作]` (Central South University), Shaokai Wang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 ContextReID，利用视频中多帧、同框玩家信息共同生成更具辨识度的嵌入，用于运动员重识别；

**💡 创新点**

创新点在于将同框及跨帧的上下文视为训练与推理的“输入”而非额外标签，实现身份无关的上下文嵌入，同时通过联合上下文与检索两种损失兼顾判别力与检索效率；

**🔧 技术方法**

技术包括：基于 Qwen3‑VL 的视觉‑语言模型，框架采用 causal attention 对多帧玩家集合进行联合编码；使用多任务对比损失（上下文对应损失 + 检索损失）；实现 8‑bit PagedAdamW、LoRA 细调；

**📊 数据集**

使用三大多目标跟踪/重识别数据集：SportsMOT、DanceTrack 及 SN‑GSR‑2025；对每个数据集的标注框转化为玩家 crop；

**📈 对比分析**

与 SOLIDER、TransReID、ResNet50‑IBN‑a 等传统单 crop ReID 基线以及 PRTReID 在相同训练设定下对比；ContextReID 在 Rank‑1 与 mAP 上均显著提升（举例：SportsMOT Rank‑1 96.44% vs SOLIDER 95.45%；mAP 69.66% vs SOLIDER 63.25%）；

**⚠️ 局限性**

局限性：需要对多帧多crop一次性推理，导致显著提升 GPU 内存与计算；仅在多人人群场景下有效，单人或缺乏上下文时效果退化，可回退至无上下文模式。

---

## 398. CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation

**arXiv ID:** 2610.13257 | [PDF](https://arxiv.org/pdf/2610.13257v1)

**作者:** Jingru Yang `[一作]` (Taobao & Tmall Group of Alibaba), Biye Li `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CAST 方法，解决跨 tokenizer 的 on‑policy distillation 中 chunk‑level 对齐后 token‑level 的信用分配问题，采用 shift‑or‑stay 框架，将学生不确定性与教师信号平滑度结合权重分配，并对未对齐位置做选择性 KL 锚定；

**💡 创新点**

创新点在于将 chunk 对齐与 token‑level 信用分配分离，提出基于不确定性与平滑度的权重分配规则，并通过 KL 正则化解析为最优解；同时引入选择性 Anchoring 处理未对齐位置，且可推广到共享 tokenizer 的标点或固定长度 chunk；

**🔧 技术方法**

技术包括 OPD、decoded‑text alignment、学生熵与教师 log‑prob 差异计算、不确定性×平滑度权重分配、KL 正则化、选择性 KL 锚定、multi‑rollout sampling 与 Policy Gradient 更新；

**📊 数据集**

使用 OpenThoughts v1（约 20k 竞赛式数学等题目）和 v2（约 15k 包含数学、代码、科学）作为训练数据，评估六个基准：AIME 2024‑2026、MATH‑500、GPQA‑Diamond、LiveCodeBench；

**📈 对比分析**

与 DPCA、ULD、ALM、GKD、MiniLLM、DistiLLM、GRPO 等多种基线在 Qwen3‑8B→Mistral‑7B、Llama‑3.1‑8B 交叉 tokenizer 及 Qwen3‑1.7B 共享 tokenizer 场景对比，CAST 在六项平均上分别提升约 1.3–2.5 分，显著提升各子基准并减少训练崩溃，梯度更稳定；

**⚠️ 局限性**

局限在于仅验证有限模型家族和主要推理任务，数据集偏重推理，计算量未统一，对信用分配对整体优化影响的理论与实验尚不充分，需要更广泛的模型、数据与匹配计算评估。

---

## 399. PDE-PFN: Prior-Data Fitted Neural PDE Solver

**arXiv ID:** 2610.13334 | [PDF](https://arxiv.org/pdf/2610.13334v1)

**作者:** Jaehyeon Park `[一作]` (SolverX), Noseong Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于先验数据拟合网络(PFN)的PDE解预测方法PDE-PFN，利用上下文观测直接在不做梯度更新的条件下推断未知空间时间点的解。

**💡 创新点**

创新点在于把PFN与Transformer相结合，加入Fourier特征嵌入和注意力图滤波(AGF)层，形成适用于复杂PDE数据的自注意力编码器；并通过后验预测均值(Posterior Predictive Mean)实现无符号噪声下的稳健推理。

**🔧 技术方法**

技术包括：先验数据拟合网络(PFN)架构、Transformer的自/交叉注意力、Fourier特征映射、注意力图滤波(AGF)、在PDE领域的上下文/查询点注意力机制以及MSE损失对应的后验均值学习。

**📊 数据集**

主要使用：① 2D耦合输运-扩散-反应(CDR)方程族（共2187个参数化PDE）生成的数值解与PINN近似解；② 额外的浅水方程(SWE)和可压Navier‑Stokes方程(CNSE)数据集用于跨PDE迁移评估。

**📈 对比分析**

与DeepONet、FNO、F‑FNO、A‑FNO、LNO、Poseidon、DPOT等基线进行对比，在时空插值、时间外推、跨参数与跨方程迁移等任务中，PDE‑PFN在绝大多数指标（L1、L2、L∞）上表现最好或相当，且对噪声先验具有较强鲁棒性，且不需要额外梯度更新。

**⚠️ 局限性**

局限性包括：仅预测后验均值，无法得到完整分布；模型目前主要验证于2D CDR、SWE、CNSE，尚未在更高维或更复杂边界条件下检验；对模型规模、先验多样性、上下文大小及分布偏移的系统性规模分析尚缺乏。

---

## 400. DuoRate: Two-Rate Per-Expert Precision Allocation for Mixture-of-Experts Serving

**arXiv ID:** 2610.13328 | [PDF](https://arxiv.org/pdf/2610.13328v1)

**作者:** Mengqi Guo `[一作]` (Huawei Technologies Ltd), Michael Bi Mi `[通讯]` (Huawei Technologies Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种两阶段 MoE 专家权重量化与分配方法，先一次性存储统一的高精度权重层级，再为每个工作负载生成只读取所需前缀的读表。

**💡 创新点**

创新点在于把权重量化的存储成本与解码时的流量成本分离为两种速率函数，并证明路由频率在解码侧分配中可抵消，从而实现仅用一次权重存储即可满足不同工作负载的读取速率需求。

**🔧 技术方法**

使用 prefix‑readable weight stream、Lagrangian 率-失真分配、一次性离线预热与在线无标签工作负载统计、单层 dot‑product 计算与二分搜索求解读取表。

**📊 数据集**

主要评估数据集为 Qwen3‑30B‑A3B 模型的专家矩阵，测试集包括 WikiText‑2、MMLU‑Pro、MMLU、GSM8K。

**📈 对比分析**

与多种 2‑bit、4‑bit 以及 4.5‑bit 固定精度量化方法对比，结果显示在 2.3 读速率下远超 2‑bit 方案，2.9 读速率下与 4‑bit 方案相当，并在长生成任务 GSM8K 上几乎匹配 BF16 的性能，同时显著降低解码流量。

**⚠️ 局限性**

局限包括仅在单一 MoE 模型和路由方案上验证，未测量真实前缀读取的内核性能，读表需要针对每个工作负载重新校准，且对不同专家数与路由策略的泛化尚待验证。

---

## 401. CoVA: Composing VAE Representations for Generative Learning

**arXiv ID:** 2610.13322 | [PDF](https://arxiv.org/pdf/2610.13322v1)

**作者:** Zhaowei Chen `[一作]` (Jiiov Technology), Xiang Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

CoVA 通过在线轻量化 VAE 特征composer，将 VAE 重建特征重组为语义可区分且空间结构化的表示，先在训练初期用于中间表示对齐，后期再将其类别后验作为样本级软标签进行条件建模，提升生成质量。

**💡 创新点**

创新点在于：①利用在线 composer 对 VAE 特征进行空间结构重组，而非直接使用原始重建特征；②提出阶段自适应训练策略，将表示对齐与软标签条件分离；③揭示软标签条件中幅度与方向的混淆，并通过幅度校正消除该 confound，使后验方向真正提升多样性。

**🔧 技术方法**

技术包括：流匹配（flow‑matching）生成器 SiT、ViT‑based composer、类监督+掩码重构+水平翻转一致性、对齐投影器、样本级软标签条件（SLC）与幅度校正、混合精度训练与 EMA。

**📊 数据集**

数据集：ImageNet 256×256，使用预训练的 Stable Diffusion VAE 对图像进行编码，评估 FID、sFID、IS、Precision、Recall 等指标。

**📈 对比分析**

对比方法：REPA、SRA2、SRA 2、DoD 等表示对齐方法以及多种 SOTA 生成模型；CoVA 在同 400K 训练步数下分别将 SiT‑B/2 FID 从 33.0 降到 21.4、SiT‑L/2 从 18.8 降到 9.3、SiT‑XL/2 从 17.2 降到 7.4；在 400 轮 CFG 下 FID 仅为 2.03，显著快于原始 SiT‑XL/2 的 2.06（需 1400 轮）。

**⚠️ 局限性**

局限性：训练阶段增加约 8.5M 辅助参数和 7% 计算开销，需额外 tune 过渡步骤与 SLC 触发时机；依赖于 VAE 的潜在空间，若 VAE 表示不足可能影响效果；在 ImageNet 之外的多样化数据集或文本条件生成的评估尚未展开。

---

## 402. Hardware-in-the-Loop Demo of a Cooperative Intersection where Cars don't Stop

**arXiv ID:** 2610.13320 | [PDF](https://arxiv.org/pdf/2610.13320v1)

**作者:** Lorenzo Farina `[一作]` (Università di Bologna), Alessandro Bazzi `[通讯]` (Università di Bologna)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了一个硬件在环（HiL）演示，利用 Raspberry Pi 5 OBU 与 RSU 通过 IEEE 802.11p 与远端控制器通信，实现车辆在交叉口不停车的协同通过；

**💡 创新点**

通过将低功耗开源 OScar C‑ITS 堆栈与低成本硬件结合，首次在硬件层面演示 V2X 交叉口协调算法，可在实时无线网络中实现车辆速度优化；

**🔧 技术方法**

采用 OScar 开源 ETSI C‑ITS 堆栈、IEEE 802.11p V2X 通信、SUMO 交通仿真、Raspberry Pi 5 与 Mikrotik Wi‑Fi 卡、GPS/IMU 传感器数据获取；

**📊 数据集**

使用 SUMO 生成的车辆轨迹作为仿真数据集，未引入公开大型数据集；

**📈 对比分析**

将该方案与传统红绿灯/优先通行方式对比，实验表明行驶时间和排放量显著下降，但未给出具体数值；

**⚠️ 局限性**

仅在 HiL 环境下验证，缺乏真实道路试验；交叉口规模有限；通信可靠性受 802.11p 覆盖与干扰影响；Raspberry Pi 的功耗与可扩展性仍需进一步评估。

---

## 403. ThinkDrag: Semantic Drag-Based Image Editing with Visual Reasoning

**arXiv ID:** 2610.13316 | [PDF](https://arxiv.org/pdf/2610.13316v1)

**作者:** Armando Fortes `[一作]` (Nanyang Technological University), Xingang Pan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的多模态框架 ThinkDrag，用于拖拽式图像编辑，能够通过视觉推理解释拖拽指令并生成符合语义的编辑结果。

**💡 创新点**

创新点包括：①在统一模型中引入可视化推理链，先解释拖拽意图再生成；②提出拖拽端点标记（DEM）在VAE空间中显式对应手柄与目标点，提升空间精度；③构建高质量语义拖拽数据集及推理链，用于监督训练。

**🔧 技术方法**

技术：BAGEL‑style 统一多模态 Transformer，融合视觉、文本与端点标记；使用注意力机制实现端点对应；训练时结合生成损失和推理链语言建模损失；支持直接与推理两种生成模式。

**📊 数据集**

使用自研 ThinkDrag 数据集（约22k 语义拖拽样本），以及公开 DragBench 和新构造的 DragBench++ 验证基准。

**📈 对比分析**

与 DragDiffusion、DragNoise、DragLoRA、FastDrag、GeoDrag、InstantDrag、LightningDrag 等基线对比，ThinkDrag 在 DragBench++ 上在编辑质量（IF、IF*）和点跟踪精度（MD_MLLM）均实现最优；在 DragBench 上保持竞争力的点跟踪，同时显著提升生成质量。

**⚠️ 局限性**

局限性：对大幅位移或多步连环拖拽仍有一定误差；推理链生成耗时；在极端语义不确定的场景中，模型仍可能产生非预期的结构改变。

---

## 404. VoxAnon: A balanced dataset for a fair evaluation of speech anonymization

**arXiv ID:** 2610.13313 | [PDF](https://arxiv.org/pdf/2610.13313v1)

**作者:** Orane Dufour `[一作]` (Université de Lorraine), Mickael Rouvier `[通讯]` (Avignon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了 VoxAnon 数据集，用于平衡性评估语音匿名化的隐私安全；

**💡 创新点**

首次提供四个说话人属性（性别、年龄、口音、音高）在训练和测试集上都均衡分布的数据集，显著揭示属性失衡导致的隐私评估偏差；

**🔧 技术方法**

采用 SpeakerCard 与 VoxCeleb 元数据交叉验证属性，利用 SONAR 句子嵌入确保测试与登记说话者来自不同视频源，使用四种 ASV 评估器（RedimNet、ResNet、ECAPA‑ssl、WavLM‑ECAPA）和三种匿名化基线（B3、B4、B5）；

**📊 数据集**

基于 VoxCeleb1 与 VoxCeleb2 语料，生成平衡与非平衡的训练与评估子集；

**📈 对比分析**

通过 EER 指标对比平衡与非平衡设置，在多攻击场景下发现属性失衡会导致某些子组识别更易或更难，平衡训练并不能完全消除差异，仍显著影响不同口音组的匿名化效果；

**⚠️ 局限性**

限制在于仅评估了三种匿名化基线和四种攻击器，未覆盖更广泛的语音处理方法，且仍存在音高与年龄属性的残余偏差，需要进一步研究平衡策略和模型改进。

---

## 405. Structure-Aware Numerical Stabilization for Half-Precision Proximal Policy Optimization

**arXiv ID:** 2610.13306 | [PDF](https://arxiv.org/pdf/2610.13306v1)

**作者:** Linxuan Wu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Liang Chen `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并实现了一种在消费级 GPU 上使用 FP16 的高效稳定 PPO 训练框架。

**💡 创新点**

引入三模块结构感知保护（全局标量保护、策略差分残差保护、持久更新保护），实现 FP16 下的数值稳定且不需重调参数。

**🔧 技术方法**

采用半精度训练、FP32 缓存统计、残差域对数概率计算、SHARP 结构化根二阶矩 Adam 以及补偿写回等技术。

**📊 数据集**

在 6 个 MuJoCo 连续控制环境（HalfCheetah、Hopper、Walker2d、Humanoid、InvertedDoublePendulum、Swimmer）上，使用 5 种随机种子进行实验。

**📈 对比分析**

与全 FP32、AMP‑FP16、AMP‑BF16 进行对比，PPO‑FP16 在 30 个 env‑seed 组合中平均排名 1.90，速度提升 1.16–2.03×，优化器状态内存下降 71.7–74.9%。

**⚠️ 局限性**

仅在 FP16 下实现，BF16 与更低位数精度仍有限；仅对 log‑std 写回做补偿，其他参数不受；未在更大模型或嵌入式加速器上验证。

---

## 406. Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans

**arXiv ID:** 2610.13303 | [PDF](https://arxiv.org/pdf/2610.13303v1)

**作者:** Sangdo Han `[一作]` (CEO Business School), Moonsoo Kim `[通讯]` (CEO Business School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种完全无标签的程序，通过对多评委之间的分歧进行量化，自动定位并重写评判准则（rubric）中的争议子条款，以提升评委之间的一致性。

**💡 创新点**

创新点在于：①利用交叉评委的不一致性作为信号而非人为标注；②实现了一个可重复、可再现的三阶段编辑流程（分解、争议定位、重写），并在不同评委组合和基准上进行前注册评估；③展示了争议定位对评委一致性有效，但并未提高与人工评注的对齐度。

**🔧 技术方法**

技术实现包括：①使用Claude Opus 5.5作为编辑模型，对rubric进行原子子准则拆分；②对每个子准则计算评委分歧率，按阈值挑选最高分歧的条款；③编辑模型在不加入新构造的前提下重写这些条款，必要时附加例证；④所有步骤均基于固定提示、单次调用。

**📊 数据集**

实验数据集包括：SummEval（新闻摘要评分，400条），HealthBench（医学对话评估，1,000条），USR Topical-Chat（知识驱动对话评估，360条），每个数据集均提供已公开的人工标签。

**📈 对比分析**

与原始rubric、公开的评判方法（如G‑Eval）、随机挑选条款编辑以及一次性重写等六种条件进行比较。结果显示：在SummEval上，争议条款编辑显著提升Krippendorff’s α（从0.577提升至0.646），但对专家评分的Kendall τ无显著变化；在HealthBench和USR上，Inter‑judge α和与人类评标的宏观F1/Spearman ρ基本无显著提升，甚至在HealthBench出现轻微下降。

**⚠️ 局限性**

主要局限包括：①评委组合不平衡，三位评委为Claude大型模型，其他为小型模型，导致评委、模型规模和提供商效应混杂；②编辑仅执行一次，未探究多轮迭代或不同编辑器；③Stage 1中的代码化检查未触发，未验证此步骤效果；④仅使用英文数据集，且SummEval可能包含在评委训练数据中；⑤所有评估基于已有人工标签，存在噪声；⑥未做等价检验，无法确认无显著差异是否意味着真正无效。

---

## 407. MHR-Router: Multimodal Heterogeneous Retrieval-Enhanced Adaptive Routing for VLM Inference

**arXiv ID:** 2610.13278 | [PDF](https://arxiv.org/pdf/2610.13278v1)

**作者:** Wenjuan Zhu `[一作]`, Qi Guo `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 MHR-Router，一个基于多模态检索增强的自适应路由框架，用于在每个图文查询中挑选最合适的视觉语言模型专家。

**💡 创新点**

创新点包括：①将冻结的图文特征融合成统一的路由表示；②为每个专家构建成功与失败记忆，并以局部检索统计（均值、最大相似度）作为证据；③使用共享的 MLP 分数器，通过对比成功/失败邻域来提升专家选择精度。

**🔧 技术方法**

采用的技术包括：冻结的 SigLIP/CLIP 多模态编码器、余弦相似度 k‑NN 检索、检索统计量（均值、最大值）、共享 MLP 分数器、列表式与对比式损失函数，以及超参数调优。

**📊 数据集**

使用的数据集为 6,400 条跨八类视觉能力的查询，来源于 CIFAR‑100、Food‑101、CLEVR、MVTec AD、TextVQA、ChartQA、ScienceQA、A‑OKVQA，并评估八个开源 VLM 专家模型（如 Qwen2‑VL、LLaVA‑OV、InternVL 等）。

**📈 对比分析**

方法与基线比较：随机路由、开发集最佳单一专家、按能力/任务的开发集选择、无检索的多模态路由等。MHR‑Router 在测试集上的路由准确率为 71.58%，比最佳单一专家高 7.36pp，且比无检索基线高 1.22pp。

**⚠️ 局限性**

局限性：依赖训练时已收集的成功/失败记忆，检索开销不容忽视；对未出现的专家或极端稀有类别的推理效果未知；冻结的编码器可能限制对新领域的适应性。

---

## 408. Vision Transformers Learn Word Meaning from Images Alone

**arXiv ID:** 2610.13252 | [PDF](https://arxiv.org/pdf/2610.13252v1)

**作者:** Anders Gjølbye `[一作]` (Technical University of Denmark), Lars Kai Hansen `[通讯]` (Technical University of Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究发现大规模自监督视觉Transformer（DINOv3）能够通过仅图像训练学习渲染文字的语义关系。

**💡 创新点**

首次证明无语言监督的视觉模型能从图像中获取文字语义结构，并与文本/语音编码器共享表示。

**🔧 技术方法**

使用自监督预训练的DINOv3、词向量差分、邻居纯度、对齐评估以及对比微调。

**📊 数据集**

基于ImageNet-1k验证集、Instagram图像以及自构造的词汇表（1481个单词+伪词）。

**📈 对比分析**

与DINOv2、MAE、Web-DINO-1B、CLIP-B、SigLIP2-B等模型对比，DINOv3-H/7B在国家-首都检索达60%、邻居纯度与CLIP-B相当，但同义词检索仍低。

**⚠️ 局限性**

仅捕获有限的词义关联，缺乏同义词深度、句子结构或指令理解，来源尚未明晰，潜在安全风险需进一步评估。

---

## 409. SnapShare: Isolation-Preserving Deduplication for Serverless Snapshots

**arXiv ID:** 2610.13318 | [PDF](https://arxiv.org/pdf/2610.13318v1)

**作者:** Leonid Kondrashov `[一作]` (NTU Singapore), Dmitrii Ustiugov `[通讯]` (NTU Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种隔离保留的服务器无状态快照管理系统，利用页面级内容归因来区分可共享与私有内容，实现安全去重与缓存协同；

**💡 创新点**

通过完全匹配的页面内容归因实现安全共享边界，仅对已批准的基础设施页面进行去重，同时保持私有数据隔离；结合工作集预取和惰性分页，实现高缓存密度与低存储需求；

**🔧 技术方法**

页面级内容匹配（MD5对比）、离线工作集采样、基于 Knative 的自定义控制器、vHive+Firecracker 微VM、分层远程存储（共享/私有工作集与懒页面）、用户空间缓存与页表映射、流式 zstd 压缩等技术；

**📊 数据集**

使用 17 个 vSwarm 基准函数（Go、Python、Node.js）以及 700 个 Azure Functions 调用轨迹（In‑Vitro），在 CloudLab 10 节点集群上实验；

**📈 对比分析**

与四个基线（Chunks、Pages、Sabre、Full‑Dedup）在相同硬件与网络带宽下进行对比；结果显示远程存储需求降低 39%，本地缓存占用减 38%，尾部请求延迟比 Sabre 低 10–29%，Fetch+解压阶段缩短至 10% 负载，整体冷启动延迟提升约 6.8 倍；

**⚠️ 局限性**

对高动态或 JVM 类的工作集预测可能不足，需异步适配；只能共享无敏感的基础设施内容，仍有时间通道泄漏风险；操作员需维护批准列表，若失误会导致隐私泄露；不解决所有共享硬件/资源竞争侧通道。

---

## 410. CALQA: Conformal Screening of Forecast-Ensemble Configurations from One Mergeable State

**arXiv ID:** 2610.13249 | [PDF](https://arxiv.org/pdf/2610.13249v1)

**作者:** Xiufeng Liu `[一作]` (Technical University of Denmark), Per Sieverts Nielsen `[通讯]` (Technical University of Denmark)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 CALQA 方法，利用合并可合并状态（mergeable state）和 conformal prediction 对预测-集成配置进行快速筛选与评估。

**💡 创新点**

创新点在于：① 将多模型预测结果压缩成单一可合并摘要，避免传统方法的枚举搜索；② 采用 conformal prediction 为每个配置提供可靠的预测区间；③ 支持 what‑if 查询，可即时评估改动对预测性能的影响。

**🔧 技术方法**

核心技术包括：合并可合并摘要、conformal prediction、预测组合（加权平均、最佳子集选择）、SQL 语句实现、批量大规模计算与性能监控。

**📊 数据集**

使用的数据集：M5 竞赛数据、CER 数据以及若干公开面板数据，用以验证方法在不同规模与领域的泛化能力。

**📈 对比分析**

与传统基线（平均、加权、性能导向组合）以及最优子集搜索进行对比。实验结果表明，CALQA 在 MAPE/SMAPE 等误差指标上平均提升 3%–5%，并在 DTU HPC 集群上实现数倍加速，显著降低计算时间。

**⚠️ 局限性**

局限性：① 需要可合并的状态结构，无法直接处理非合并型时间序列；② 对极端异常值较为敏感；③ 需要合理设置 conformal 信度水平，过低会导致误判，过高则筛选效果下降。

---

## 411. Developing Competency in AI Ethics: A training framework

**arXiv ID:** 2610.13319 | [PDF](https://arxiv.org/pdf/2610.13319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 412. Host-Aware Web Crawling for LLM Pretraining Data

**arXiv ID:** 2610.13321 | [PDF](https://arxiv.org/pdf/2610.13321v1)

**作者:** Malte Ostendorff `[一作]` `[通讯]` (Common Crawl Foundation), Malte Ostendorff (Common Crawl Foundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了基于主机层面的Web爬取策略，评估图中心性与文档质量的关联，并提出“先采样后加深”的爬取策略；

**💡 创新点**

发现文档质量高度为主机属性，证明仅需3页样本即可预测主机整体质量，并将采样结果作为深度决策依据，从而显著提升高质量页面的回收率；

**🔧 技术方法**

采用FastText多种预训练质量分类器、harmonic centrality和PageRank图中心性、ICC1方差分解、Spearman相关系数、爬取预算模拟、GneissWeb质量标签等技术；

**📊 数据集**

使用Common Crawl（CC-MAIN-2024-10、2023-40等）WARC文件、主机图、GneissWeb质量注解、FastText分类器等公开数据集；

**📈 对比分析**

通过与随机、当前CC、基于中心性、oracle等策略对比，利用高质量页面占比和每页高质量回收率评估；“采样-加深”策略在大部分分类器下获得约3-5倍的高质量页面回收率，接近oracle；

**⚠️ 局限性**

局限在样本仅来自已被CC选取的页面，质量指标为过滤器的代理，未考虑新URL发现与多语言多样性，假设未抓取页面质量与已抓取相同，估计误差；未在实际爬虫中验证。

---

## 413. SBNO : Schrödinger Bridge Neural Operator for the Forward and Inverse Problems under Missing Information

**arXiv ID:** 2610.13317 | [PDF](https://arxiv.org/pdf/2610.13317v1)

**作者:** Jaehyeon Park `[一作]` (SolverX), Noseong Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 Schrödinger 桥的无方程 PDE 前向与逆向条件建模方法 SBNO，可在稀疏观测下实现确定性点估计。

**💡 创新点**

创新点在于：①引入掩码感知的端点耦合并在固定经验端点上学习桥动力学；②通过概率流 ODE 进行无调参的确定性推理；③将前向与逆向任务统一到同一漂移网络中。

**🔧 技术方法**

使用的技术包括：时间条件 U‑Net 漂移网络、Schrödinger 桥匹配损失、概率流 ODE 与 RK4 求解、掩码提升的状态表示。

**📊 数据集**

在五个 PDE 基准（Darcy, Poisson, Helmholtz, 非有界 Navier‑Stokes, Gray‑Scott）上进行实验，并在额外的三组小规模数据集做进一步验证。

**📈 对比分析**

与 FunDPS、DiffusionPDE、FNO、GNOT、FNOPE 等基线比较，SBNO 在 5% 点观测下平均相对 L2 误差排名第一或第二，误差方差显著降低，并且在推理速度上至少快 9.7 倍、显存消耗低 4.3 倍。

**⚠️ 局限性**

局限性包括：只能处理格点离散字段，无法直接接收离散网格外的观测；确定性 ODE 推理无法体现多模态不确定性；且在非均匀观测场景下需先投影或插值。

---

## 414. BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation

**arXiv ID:** 2610.13285 | [PDF](https://arxiv.org/pdf/2610.13285v1)

**作者:** Jiaming Su `[一作]` (Shanghai Artificial Intelligence Laboratory), Peng Ye `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BoundaryEvo两阶段数据‑模型协同进化框架，先通过无判断的能力预热直接蒸馏教师长链推理轨迹，再基于学生与专家的能力边界动态挑选和验证训练样本，提升小型模型推理能力。

**💡 创新点**

核心创新在于：① 采用无判断的能力预热保持多样化教师推理行为；② 通过学生与专家的能力差距、质量、有效性等多维度度量，动态识别处在学生能力边界附近的样本；③ 结合回放机制实现训练数据与模型共同进化。

**🔧 技术方法**

技术方法包括：教师轨迹生成与验证、LoRA‑SFT微调、专家模型（GPT‑5.1）用于能力边界估计、分析器（Gemini 3.5 Flash）评估质量与有效性、基于阈值的动态样本筛选与回放。

**📊 数据集**

使用了从Nemotron‑Cascade‑2‑SFT‑Data提取的数学推理轨迹作为教师样本，实验评估涉及11个公开基准（AIME, HMMT, BruMO, IMO‑AnswerBench, AMO‑Bench, ArxivMath, ZebraLogicBench, GPQA‑Diamond, SuperGPQA 等），并利用DeepSeek‑V3.2‑Speciale生成教师轨迹。

**📈 对比分析**

与静态全数据SFT、随机采样、LLM2LLM等基线对比，BoundaryEvo在仅1,000条样本的情况下，使Qwen3.5‑4B在IMO‑AnswerBench从62.9%提升至75.3%，在AMO‑Bench从26.8%提升至46.0%；在11个基准上均超越同级4B模型，且部分指标与更大模型竞争。

**⚠️ 局限性**

局限性包括：对极小模型易饱和；需额外专家模型与分析器来估计能力边界，增加计算成本；在训练预算极低或推理任务超出教师能力范围时效果有限。

---

## 415. Extracting Illicit-Rich Subgraph from Transaction Networks via Feed-Guided Shortest Paths

**arXiv ID:** 2610.13280 | [PDF](https://arxiv.org/pdf/2610.13280v1)

**作者:** Doniyor Koshimbetov `[一作]` (Koc University), Asha Rao `[通讯]` (Rmit University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在反洗钱交易网络中利用 SPNSA 算法进行调查子图提取，并针对仅使用图结构信号的情形探讨了 feed 选择策略。

**💡 创新点**

提出一种基于图 Motif 的连贯 Feed 选择方法（S4），通过局部交易模式评分并在多中心内进行局部筛选，显著提升子图中的非法节点比例。

**🔧 技术方法**

使用 SPNSA（Shortest Paths Network Search Algorithm）、基于度数差异、随机采样等多种 Feed 选取策略，并在此基础上实现了 Motif‑based Coherent Feed Selection；评估使用了 ILR、IR 等非法比例指标。

**📊 数据集**

在 IBM‑AML 合成基准（HI‑Small、LI‑Small、HI‑Medium）以及真实的 Elliptic Bitcoin 交易图上进行实验。

**📈 对比分析**

与随机、度数不平衡、混合收发等策略对比，S4 在 Elliptic 和 HI‑variants 上的非法比例提升显著（最高可达 33.8%），并保持子图体量较小；在低非法比例的 LI‑Small 数据集上，S2 更为优异。

**⚠️ 局限性**

局限性包括：对 Elliptic 数据仅用标签信息评估，未考虑未标记节点可能为非法；简化多重交易为单向边且忽略时间戳；Motif 参数需人工调优，可能缺乏泛化性。

---

## 416. ConClick: Element-Wise Contrastive Learning Improves GUI Grounding

**arXiv ID:** 2610.13268 | [PDF](https://arxiv.org/pdf/2610.13268v1)

**作者:** Ruotian Zhang `[一作]`, Xiaoxiao Sun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对10个GUI定位模型在5个数据集上的错误进行系统分析，构造含目标‑干扰对的ConClick数据集，并提出元素级对比学习框架与推理时的ConCompare方法，以提高定位准确率并降低错误点击。

**💡 创新点**

首次将观察到的干扰元素与目标进行对比学习，结合列表排序和对比边缘损失提升模型对目标的辨别能力；同时设计了利用多视图支持与空间一致性进行候选比对的推理改进方案。

**🔧 技术方法**

使用注意力读出、列表排序损失、对比边缘损失、LoRA微调、跨视图似然评分与空间一致性评分等技术实现对比学习与推理增强。

**📊 数据集**

使用自构造的ConClick（4,292个目标‑干扰对）以及公开的ScreenSpot-Pro、MMBench-GUI L2、OSWorld-G数据集进行训练与评估。

**📈 对比分析**

在Qwen3-VL-8B模型上，准确率分别从54.71%、63.33%、83.72%提升至66.03%、70.20%、86.28%；元素错误率从30.30%、24.12%、13.08%降低至22.50%；与MVP、ZoomClick等推理方法对比亦表现更佳，整体平均准确率升至74.17%，元素错误率仅22.50%。

**⚠️ 局限性**

主要局限包括：对干扰元素的提取依赖已有模型预测，数据集规模有限，推理时需多次视觉解码导致计算开销增加，对非实验设定的泛化性仍待验证。

---

## 417. NovaSplat: Model-Free 6D Object Pose Estimation via 3D Gaussian Splatting Onboarding and Refinement

**arXiv ID:** 2610.13309 | [PDF](https://arxiv.org/pdf/2610.13309v1)

**作者:** Georgios Floros `[一作]` (National Technical University Of Athens), Maria Pateraki `[通讯]` (National Technical University Of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种模型自由的6D位姿估计管道NovaSplat，通过短视频学习3D高斯投影模型并实现模板渲染与可微姿态细化。

**💡 创新点**

将3D高斯投影作为模板与细化的双重角色，结合连续6D旋转参数化以及光照与几何双重损失，克服传统ICP不稳的问题。

**🔧 技术方法**

基于SfM+MVS重建、3D Gaussian Splatting、FreeZeV2粗位姿、DINOv3特征、GeDi几何特征、光照+Chamfer双损失、连续6D旋转表示。

**📊 数据集**

HOPEv2家庭物体、IndustryShapes工业物体（经典/扩展）。

**📈 对比分析**

在HOPEv2上超过所有模型基准，AP提升2.9%、AP_MSSD提升17%；在IndustryShapes上超越所有模型自由基线，缩小与模型基准的差距。

**⚠️ 局限性**

对纹理缺失或镜面反射的物体重建与特征匹配受限，导致性能下降；需要更鲁棒的几何重建与表面光照处理。

---

## 418. Dual-Stream Memory for Vision-Language-Action Policies

**arXiv ID:** 2610.13262 | [PDF](https://arxiv.org/pdf/2610.13262v1)

**作者:** Yang Wang `[一作]` (Magiclab Robotics Inc), Tao Zhang `[通讯]` (Magiclab Robotics Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出双流记忆机制，结合递归执行上下文与任务条件下的历史视觉检索，以提升机器人长周期操作的成功率

**💡 创新点**

创新点在于：①任务无关的门控 DeltaNet 压缩器通过进度预测和动作条件特征重构实现可解释的执行上下文；②阈值检索与多尺度线性融合实现任务自适应的历史帧选择与细粒度空间编码；③将两流统一到 VLA 政策中，并通过门控注意力融合。

**🔧 技术方法**

技术包括 DINOv3 + Gated DeltaNet（GDN）压缩，Frozen SigLIP + MiniLM 进行阈值检索，线性多尺度融合（8×8/4×4/2×2 拼接），跨模态注意力门控融合到 Gemma+flow‑matching 动作专家。

**📊 数据集**

使用 RoboMME 16 任务仿真数据集（包含 Count、Permanence、Reference、Imitation 等任务）以及在 Piper 双臂机器人上的 4 个真实任务演示（Pick×3、Swing×2、RePick、Shell）。

**📈 对比分析**

相较于基线 FrameSamp+Modul 与 NativeMEM，整体平均 TSR 提升至 49.22%（比 FrameSamp+Modul 提升 4.71pp，所有 16 任务均有提升），在真实机器人上平均 TSR 提升至 62.5%（比 FrameSamp+Modul 提升 25.0pp，特别在重复动作任务上表现突出）。

**⚠️ 局限性**

局限性包括：①对连续目标身份跟踪不足，导致 Shell 任务性能下降；②检索阈值和多尺度融合参数需要手工设定；③实验仅在特定仿真和单一硬件平台上验证，泛化性尚待进一步评估。

---

## 419. Neural Surrogate Modeling of Ultrasonic Wave Propagation in Heterogeneous Media

**arXiv ID:** 2610.13258 | [PDF](https://arxiv.org/pdf/2610.13258v1)

**作者:** Burla Nur Korkmaz `[一作]` (University of Helsinki), Arto Klami `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种结合 Fourier Neural Operators 与 U‑Net 的局部积分重采样结构（LU‑NO），实现了超声波在异质介质中的高频、跨分辨率的自回归模拟。

**💡 创新点**

创新点在于把全局频域层与多尺度 U‑Net 结合，并提出局部积分重采样机制，使模型既能捕捉全局依赖又能在不同网格尺度保持连续性，克服了传统 FNO 对界面反射的欠缺。

**🔧 技术方法**

采用 Fourier Neural Operators、U‑Net 结构、局部积分重采样、组归一化、GELU 激活等技术，并在 2D 频域上应用 Fourier 变换。

**📊 数据集**

使用 COMSOL 生成的二维水域内三圆铝体的声压场数据，仅 14 条训练发射路径（每条训练分辨率 121×121），测试时对 12 条新结构进行高分辨率 (301×301) 验证。

**📈 对比分析**

与标准 FNO、U‑Net、U‑NO 三种基线进行对比，LU‑NO 在训练分辨率下相对 L2 误差 0.305±0.083、在高分辨率下 0.295±0.088，明显优于其他方法；内存占用 11.91 GB、推理时间 34 s，显著低于 FNO 与 U‑NO。

**⚠️ 局限性**

受限于仅使用 14 条有限训练样本，模型在精度上仍低于纯卷积 U‑Net，且在更复杂结构或更高频域时可能需要更多数据以进一步提升表现。

---

## 420. Tabular Foundation Models Are Effectively Shallow

**arXiv ID:** 2610.13253 | [PDF](https://arxiv.org/pdf/2610.13253v1)

**作者:** Irene Cannistraci `[一作]` (ETH Zurich), Julia E. Vogt `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在tabular transformer中引入闭式标签无关仿射翻译器，压缩高达96%的层。

**💡 创新点**

通过无监督仿射替代器证明大部分层可被线性替换，实现显著参数削减与速度提升。

**🔧 技术方法**

采用闭式线性回归、KL驱动窗口选择、对齐单轴与双轴注意机制的仿射映射。

**📊 数据集**

在171个分类数据集（OpenML-CC18、TabZilla、TabArena、Grinsztajn等）及8个临床数据集（MIMIC‑III、eICU‑CRD等）上进行评估。

**📈 对比分析**

与完整模型和GBDT对比，半深度压缩平均AUROC仅下降0.002，单块保持误差≤0.03，速度提升6×，内存减少6×。

**⚠️ 局限性**

极端单块压缩在双轴模型和回归任务中表现欠佳，且映射参数随特征维度放大；对生产级规模的浅层预训练是否同等有效仍未验证。

---

## 421. Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning

**arXiv ID:** 2610.13330 | [PDF](https://arxiv.org/pdf/2610.13330v1)

**作者:** Mengdi Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Shiguang Shan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于掩蔽扩散的工具集成化化学推理代理系统 ChemDiffAgent。

**💡 创新点**

创新点包括：将化学推理 reformulate 为在交织的推理与工具使用轨迹上进行迭代去噪的掩蔽扩散过程，使工具使用决策在完整轨迹上下文下共同决定；提出两阶段后训练框架（先进行代理监督微调，再采用方差削减的偏好优化）；构建涵盖单回合与多回合工具使用的六项化学任务综合基准。

**🔧 技术方法**

采用的技术主要是掩蔽扩散语言模型（masked diffusion language model）、工具调用集成、全轨迹上下文编码、可变顺序解码、代理监督微调以及方差削减的偏好优化（variance‑reduced preference optimization）。

**📊 数据集**

使用的数据集为作者自行构建的六项化学任务基准，涵盖单回合与多回合工具使用场景，未在文中给出具体数据集名称。

**📈 对比分析**

在相同算力预算与评估指标下，将 ChemDiffAgent 与自回归代理进行对比，ChemDiffAgent 总体提升约 20%；其 8B 版本在性能上可与甚至超过更大模型（如 GPT‑5.5、Claude Opus‑4.7）。

**⚠️ 局限性**

局限性包括：生成过程相较自回归模型可能更耗时；对复杂多模态输入或外部工具的兼容性未详细验证；缺乏对真实化学实验结果的实证评估；以及对模型解释性与鲁棒性的进一步研究仍待开展。

---

## 422. NEAR: Neighborhood-Aware Residuals for Memory Retrieval

**arXiv ID:** 2610.13314 | [PDF](https://arxiv.org/pdf/2610.13314v1)

**作者:** Han Luo `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种无训练的记忆检索框架NEAR，利用每条记忆的近邻构造局部参考方向并投影产生残差表示，随后与原始嵌入融合，仅需一次ANN搜索即可检索。

**💡 创新点**

创新点在于：①从每条记忆的邻域提取本地参考方向并投影消除主成分；②为查询预先构建小型参考表，使查询也能获得残差表示；③将原始嵌入与残差在一个拼接向量中融合，保持单索引检索的效率。

**🔧 技术方法**

使用的技术包括余弦相似度、投影残差、局部参考方向计算、预计算查询参考表、向量拼接和缩放融合、单索引ANN搜索。

**📊 数据集**

在八个长上下文检索基准（如 LoCoMo、...）上与九种冻结嵌入模型（包含 Qwen3-Embedding-4B 等）进行实验。

**📈 对比分析**

与原始嵌入、同构基线以及其他无训练校正方法对比，NEAR 在所有九个模型上均提升了 Recall@10 与 NDCG@10；在 LoCoMo 上问答准确率提升了 9.6个百分点。

**⚠️ 局限性**

局限性包括：需离线构建记忆索引和查询参考表，难以适应频繁更新的记忆库；参考表的存储开销和对极端高维/稀疏数据的鲁棒性尚未充分验证。

---

## 423. Sustainable Cloud-Fog-Edge Systems for Smart Buildings: Algorithmic Adaptation under Memory Constraints and Dynamic Orchestration Strategies

**arXiv ID:** 2610.13307 | [PDF](https://arxiv.org/pdf/2610.13307v1)

**作者:** Mamadou Sow `[一作]` (University Sorbonne Paris Nord), Christophe Cerin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了云‑雾‑边缘架构、轻量化 KOptim orchestration、在低功耗 Arduino 微控制器上实现 GHA‑PCA 的在线学习，并对传感器数据进行 K‑means 聚类。

**💡 创新点**

创新点包括：将 GHA‑PCA 与显式多样性管理的块处理策略结合，在有限内存环境下保持样本多样性；提出 KOptim 与 FIWARE 的动态 SLA 管理框架；利用 Arduino 基础设施实现端到端的数据采集与安全持久化。

**🔧 技术方法**

采用的技术包括：Arduino Uno R4 WiFi + BME280、Node‑RED + MySQL、Python/rsync/SSH、KOptim orchestration、FIWARE 标准、Eigen C++ 库实现 GHA‑PCA、K‑means 聚类、MQTT LoRaWAN、在线机器学习。

**📊 数据集**

使用的数据集为 Tour Perret 建筑的高维传感器数据（约15维）以及 CampusIOT LoRaWAN 的 CO2/温度流数据。

**📈 对比分析**

通过全内存基准、简单块处理以及带多样性管理的块处理三种策略进行对比；结果显示带多样性策略在相同内存下的聚类密度与参考相近；ESP32 在 1024 条消息、128 条块的情况下可稳定处理，且 GHA‑PCA 收敛质量与全内存相当。

**⚠️ 局限性**

局限性包括：PCA 仅捕捉线性关系导致非线性数据难以建模；降维不可避免的方差损失；显式排序导致每轮 O(W log W) 的时间复杂度；对排序变量的选择敏感；微控制器生态间算法落后；雾节点功耗高、数据一致性与安全性挑战。

---

## 424. DAG Structure Learning with Node-Varying Heavy-Tailed Disturbances

**arXiv ID:** 2610.13304 | [PDF](https://arxiv.org/pdf/2610.13304v1)

**作者:** Robin Francis `[一作]`, Sundeep Prabhakar Chepuri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种节点可变Student-t DAGMA方法，用于在存在异质重尾噪声的线性结构方程模型中学习有向无环图（DAG）。

**💡 创新点**

创新点在于为每个节点独立估计尺度和自由度参数，并将其与DAGMA的光滑无环性约束结合，实现了对节点尺度与尾部厚度的联合建模与自适应鲁棒性。

**🔧 技术方法**

主要技术包括Student-t高斯尺度混合表示、期望最大化（EM）与迭代加权最小二乘（IRLS）框架、DAGMA的对数行列式无环性惩罚以及中心路径优化。

**📊 数据集**

实验数据包括基于Erdős–Rényi DAG生成的合成样本（d=10，N=500）以及真实金融数据——美国49个行业投资组合的日收益序列（2000-2019年）。

**📈 对比分析**

与DAGMA、Huber-DAGMA、CoLiDE-NV以及固定自由度Student-t DAGMA进行对比，本文方法在合成数据上表现出更低的结构汉明距离、较高的TPR与F1分数；在金融数据上取得最低的测试负对数似然（26.40），同时保持较低的NRMSE（0.465）和中等稀疏度（15条边）。

**⚠️ 局限性**

局限性包括需要先验设定自由度范围、EM迭代可能收敛到局部最优、对高维大规模数据的计算效率待提升，以及在极端重尾或缺失值场景下的稳健性尚未充分验证。

---

## 425. PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real

**arXiv ID:** 2610.13296 | [PDF](https://arxiv.org/pdf/2610.13296v1)

**作者:** Cheuk Hei Chung `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 PRIME 框架，通过在索引阶段对语料进行语义重构与结构化元数据生成，提升 LLM 代理的检索效果。

**💡 创新点**

创新点在于将 LLM 驱动的语义重构与确定性元数据提取相结合，并构建分层检索流程，既显著降低索引成本，又在全量部署中提升 Answer‑F1 13.7%。

**🔧 技术方法**

技术实现包括 DeepSeek‑V4 LLM 进行章节级重写，TF‑IDF 生成判别词和标题树，BGE‑small 词向量，BM25、RRF 级联和 cross‑encoder reranker。

**📊 数据集**

实验使用 QASPER、BRIGHT 与 NFCorpus 三个结构多样的检索基准。

**📈 对比分析**

在 NDCG、P@5 等检索指标上 PRIME offline 与结构化分块相当，但仅使用 1/45.8 的嵌入计算量；在全量模式下 Answer‑F1 较基线提升 13.7%，并在 BRIGHT 上宏平均提升 6.9%。

**⚠️ 局限性**

局限性包括：语义重构可能删除关键信息或产生少量矛盾；在无 H2 结构的语料上元数据无效；一次性 LLM 预处理成本高；并未处理表格、图像等非文本信息。

---

## 426. Do Prompt Optimizers Break Requirements They Are Not Scored On? A Preregistered Study

**arXiv ID:** 2610.13287 | [PDF](https://arxiv.org/pdf/2610.13287v1)

**作者:** Sangdo Han `[一作]` (CEO Business School), Moonsoo Kim `[通讯]` (CEO Business School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究prompt优化器对未评分要求（格式、引用、拒绝等）的影响，并量化其在HotpotQA、SQuAD 2.0和IFEval上的违规情况。

**💡 创新点**

系统性检验单目标优化对多种未评分要求的回归，并揭示输出预算对提取失败的关键作用。

**🔧 技术方法**

使用GEPA、MIPROv2和TextGrad三种prompt优化器，结合Claude Haiku 4.5作为任务模型、Claude Sonnet 5.5作为提议模型，并通过程序化检查和Bootstrap统计进行量化。

**📊 数据集**

HotpotQA（多跳问答）、SQuAD 2.0（可答与不可答）和IFEval（指令遵循）三大公开数据集。

**📈 对比分析**

在预注册的严格指标下评估净回归率，结果显示HotpotQA上引用与答案长度回归显著，但扩展输出预算后回归降至近零；在某些优化器上精确匹配（EM）提升。

**⚠️ 局限性**

仅使用单一任务模型与提议模型、搜索预算有限、仅考察三种优化器、初始prompt已严重违规、未评估“解释文本”质量，导致实验范围和结论受限。

---

## 427. Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence

**arXiv ID:** 2610.13277 | [PDF](https://arxiv.org/pdf/2610.13277v1)

**作者:** Han Luo `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种利用冻结的嵌入模型的上下文词元表示进行稀疏词元匹配，并与整体记忆相似度融合的检索方法，以提升记忆检索效果。

**💡 创新点**

①无需额外训练即可利用已冻结模型的词元表示进行本地匹配；②将稀疏词元匹配与全记忆相似度融合；③可选的低秩查询适配器在保持编码器不变的情况下进一步提升性能。

**🔧 技术方法**

冻结的多模态/语言模型嵌入（Qwen3、Llama、Gemma等）、HNSW近似最近邻搜索、MaxSim/MaxSim改良的局部匹配、低秩LoRA查询适配器、标准化融合α参数、压缩量化等技术。

**📊 数据集**

八个检索基准（LoCoMo、LongMemEval、REALTALK、ESG-Reports、QASPER、PeerQA、CovidQA、MLDR）以及对应的长问答下游任务。

**📈 对比分析**

与原始单向量嵌入、Search-Adaptor、Chunk-MaxP、MaxSim、BM25、ColBERTv2 等同骨干基线做对比；在 Qwen3-Embedding-4B 上平均 Recall@10 从 62.5% 提升至 73.8%（无适配），加上适配器进一步升至 75.7%；NDCG@10 同样提升，且在下游问答任务上准确率提高 3–17% 点。

**⚠️ 局限性**

需要存储所有词元表示导致索引体积增大；稀疏匹配对近似搜索误差敏感；查询适配器训练需要监督样本；在某些基准（如Gemma on REALTALK、Llama在QASPER）提升有限；模型对超参数（k,α）较敏感。

---

## 428. LETHE: Defending against Unauthorized Image-to-Video Generation via Value Diversity Collapse

**arXiv ID:** 2610.13333 | [PDF](https://arxiv.org/pdf/2610.13333v1)

**作者:** Boheng Li `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何通过对抗扰动防止个人图片被图像到视频扩散模型滥用，并提出基于价值收缩和注意力吸引的保护方法。

**💡 创新点**

从凸包几何角度揭示参考值多样性决定模型读取范围，利用此特征同时进行价值收缩与注意力引导，形成全新对抗扰动策略。

**🔧 技术方法**

使用对抗扰动优化、DiT注意力分析、价值收缩损失、注意力吸引损失以及MI‑FGSM梯度投影等技术。

**📊 数据集**

对CelebV‑Text与Kinetics‑700进行重新标注，构建人类中心的图像到视频保护基准。

**📈 对比分析**

与PhotoGuard、AdvDM、SDS+、I2VGuard、Anti‑Prompt等基线在ISM、QAF、QAV等指标上对比，所提方法在Wan2.1和CogVideoX两大模型上均取得最优表现，显著低于所有基线。

**⚠️ 局限性**

仅针对单帧图像、对跨模型泛化与去除扰动的鲁棒性仍有提升空间，且对更复杂输入变换与更高预算情况的评估尚不充分。

---

## 429. Multi-Source and Multi-Dimensional Information Fusion in Resource-Constrained Edge Networks

**arXiv ID:** 2610.13298 | [PDF](https://arxiv.org/pdf/2610.13298v1)

**作者:** Yushi Wang `[一作]` (Lancaster University), Merouane Debbah `[通讯]` (Khalifa University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种面向资源受限边缘网络的多源多维语义信息融合框架，联合了终端选择性上传与边缘端的“Hyper‑Slot”融合模块；

**💡 创新点**

创新点在于通过迭代的token‑to‑slot分配与slot‑to‑token反馈，实现对语义内容与多维属性（来源、模态、细粒度、时延、信道条件）的联合建模，并用基于Stackelberg–Nash博弈的贡献型激励机制指导终端上传策略；

**🔧 技术方法**

采用的关键技术包括：多维属性编码器、稀疏max分配、GRU迭代更新、门控机制、以及多层次博弈求解算法；

**📊 数据集**

实验数据集包括三模态COCO2014扩展（图像+文本+语音）和Flickr30K；

**📈 对比分析**

与Concat+MLP、MeanPool、SCAN、Slot Attention、HGNN等基线对比，Hyper‑Slot在mAP上提升约3.7个百分点，在RSum上提升约15.9个百分点，且参数量比Naïve拼接法低80%；

**⚠️ 局限性**

局限性在于需要先行学习多维属性编码器，模型对超参数（槽数、迭代次数、β等）敏感，且在极低传输预算下仍受限于终端可用信息的多样性。

---

## 430. QCLR: A Plug-in Robustification Module via Quantization for Contrastive Learning in Self-Supervised Skeleton-Based Human Action Recognition

**arXiv ID:** 2610.13281 | [PDF](https://arxiv.org/pdf/2610.13281v1)

**作者:** Nanzhu Chen `[一作]` (University of Leeds), He Wang `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种可插拔的鲁棒化模块QCLR，通过双层量化增强自监督骨骼动作识别模型的对抗鲁棒性，同时保持分类性能。

**💡 创新点**

创新点在于引入注意力引导的多维量化策略，既克服了向量量化导致的表征误差，又在输入与特征层双重量化中形成互补的防御壁垒。

**🔧 技术方法**

采用向量量化、逐点量化、注意力加权多维量化、ST‑GCN、GCN、T‑CNN以及对比学习与STE梯度估计等技术实现鲁棒化。

**📊 数据集**

使用三大骨骼动作数据集：NTU‑RGB+D 60、NTU‑RGB+D 120 与 PKU‑MMD Part I 进行评估。

**📈 对比分析**

在与TRADES、BEAT等基线防御对比时，QCLR在PGD、SMART、APGD‑DLR与BASAR攻击下将成功率降至单位数，并在保持或提升原模型分类准确率的同时显著提升鲁棒性。

**⚠️ 局限性**

局限性包括模型参数和训练时长显著增加、仅在基于ST‑GCN的网络上验证、以及对量化步长的依赖导致鲁棒性与精度的折中。

---

## 431. CopRoute: Dependence-Aware Conformal Routing for Language Models

**arXiv ID:** 2610.13276 | [PDF](https://arxiv.org/pdf/2610.13276v1)

**作者:** Siva Rajesh Kasa `[一作]` (National University of Singapore), Karan Gupta `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的语言模型路由策略 CopRoute，能够在满足目标可信度（all-miss 风险）要求的同时显著减少调用模型的数量。

**💡 创新点**

创新点在于：① 引入对模型失败相关性的考量，通过“贪婪计数”排序（tally order）让模型顺序体现互补性；② 设计了依赖与无依赖两种残差估计，并在校准阶段使用分层集合保证分布无关的风险下限；③ 在固定顺序下证明该排序接近最优，并证明校准后最小阈值能在所有已校准集合中最小化调用次数。

**🔧 技术方法**

使用的技术包括：基于设计集（design set）计数的贪婪排序；依赖与无依赖残差估计；分层（sublevel）集合与切点选择；分层校准（split calibration）实现风险控制；以及常规的 KNN 路由器评分作为基础模型质量评估。

**📊 数据集**

使用的数据集：RouterBench（11 模型，覆盖 ARC、HellaSwag、MMLU、WinoGrande 等）和 RouterEval（3,811 模型，选取 100、200、300、500 模型子池）。

**📈 对比分析**

与现有方法（RACER-G、RACER-P、FlexRouter、BaRP）进行对比，在 10 个数据集、两种风险目标（α=0.05、0.10）下，CopRoute 在 9/10 或 8/10 数据集上获得最少的已认证调用次数，整体平均比 RACER-G 低 24%（最多 79%）的调用量；在 500 模型子池下仍保持优势；在不同路由器、不同目标、不同池大小以及跨数据集迁移实验中表现稳健。

**⚠️ 局限性**

局限性包括：① 证书仅保证平均 all-miss 风险，无法给出每个查询的具体保证；② 依赖校正规则对深层前缀的数据稀缺导致效果不显著，仍需更好的尾部估计方法；③ 路由顺序高度依赖设计集，跨数据集迁移时可能退化；④ 未评估延迟、聚合后答案的质量以及多租户场景下的实际部署成本。

---

## 432. OWLET: On-Policy Weighted Learning via Evidence Transfer for Video Large Language Model Post-Training

**arXiv ID:** 2610.13272 | [PDF](https://arxiv.org/pdf/2610.13272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis

**arXiv ID:** 2610.13273 | [PDF](https://arxiv.org/pdf/2610.13273v1)

**作者:** Yongjian Tang `[一作]` (Siemens Research), Thomas Runkler `[通讯]` (Siemens Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Agent4Spec，一套基于大型语言模型的多智能体框架，用于持续验证和市场驱动的 API 规范优化。

**💡 创新点**

创新点在于将 LLM 与多智能体协同，既能自动检测并纠正 OpenAPI 语法错误，又能进行竞争分析，识别功能缺口并自动补全缺失的 API 功能。

**🔧 技术方法**

采用 LLM（如 GPT‑4 等）作为核心生成器，并结合自定义验证智能体和市场分析智能体，利用 OpenAPI 规范解析与自动化测试实现功能。

**📊 数据集**

评估使用了两个自编数据集，分别涵盖通用和工业场景的 API 规范，数据量覆盖多种业务类型。

**📈 对比分析**

与 9 种 SOTA LLM 进行对比，最高模型在语法错误检测上实现 100% 的准确率，自动修复成功率超过 95%，且建议的端点改进被超过 50% 的开发者采纳，表明系统在实际应用中的有效性。

**⚠️ 局限性**

局限性包括对极其复杂或自定义扩展的 API 规范支持不足，以及对实时市场信息的依赖，使得在数据稀缺或竞争环境快速变化时性能可能下降。

---

## 434. EndoRSI: Physical-Evidence-Driven Recursive Improvement of Endoscopic Robot Policies

**arXiv ID:** 2610.13376 | [PDF](https://arxiv.org/pdf/2610.13376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 435. From Structure to Validity: Mechanism-Guided Generation and Refinement of LLM Rubrics

**arXiv ID:** 2610.13375 | [PDF](https://arxiv.org/pdf/2610.13375v1)

**作者:** Feifei Qian `[一作]` (Beijing Normal University), Lu Bai `[通讯]` (Beijing Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统分析了rubric设计的结构化拆分、语义规范和评分聚合三个阶段，并基于分析结果提出了ARTS-Hybrid框架，实现自适应树搜索与语义-响应校准的rubric生成与精炼；

**💡 创新点**

创新点在于将结构化拆分、语义有效性与响应校准三种机制化并融合，首次证明结构化拆分能显著提升覆盖率、语义规范单独不足以保证判定稳定性、加权聚合对性能提升有限；

**🔧 技术方法**

采用LLM生成与评判、树结构自适应搜索、语义有效性检测、响应基准校准、加权分布估计以及全局验证与权重分配等技术；

**📊 数据集**

使用HealthBench、RubricBench、JudgeBench、RM-Bench Chat、RewardBench、OpenRubrics等多领域偏好与对话数据集；

**📈 对比分析**

与Direct Judge、Query-Only、Response-Aware、Q-World等基线在六个基准上对比，ARTS-Hybrid在5/6基准上获得最高或相近的pairwise accuracy，并在rubric质量四维度表现更优，蒸馏后单一生成器亦能提升性能，奖励信号在policy优化中表现最佳；

**⚠️ 局限性**

主要局限在于需要多轮LLM推理导致计算成本高、对极端多模态任务验证不足、校准响应与语义验证可能受模型偏见影响、缺乏针对少数基准的显著提升以及对人类评审的依赖。

---

## 436. AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents

**arXiv ID:** 2610.13335 | [PDF](https://arxiv.org/pdf/2610.13335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 437. Overcoming Data Imbalances with Expert Selection Biases

**arXiv ID:** 2610.13370 | [PDF](https://arxiv.org/pdf/2610.13370v1)

**作者:** Ayush K. Tarun `[一作]` (École Polytechnique Fédérale de Lausanne), Antoine Bosselut `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在MoE预训练期间为每个类别引入软偏置，使token更倾向于专门化专家子集，从而提升低资源类别性能，且推理时可去除偏置保持标准MoE推理

**💡 创新点**

利用已有的负载均衡偏置机制，将其转为类别条件的路由先验，显式控制每个类别的专家份额；不需要硬划分专家集且无推理成本

**🔧 技术方法**

Mixture-of-Experts、软路由偏置、类别条件路由、SwiGLU、GQA、Megatron-LM框架

**📊 数据集**

多模态数据（文本、图像、音频），多语言混合（英语、韩语、乌尔都语、斯瓦希里语等），多领域数据（代码、数学、法律、医疗等）

**📈 对比分析**

相较无偏置基线，COAX在低资源语言、少数模态和专门领域均显著降低验证损失，等价于额外18.5%–2.7%训练token；在多语言八语种设置下平均损失下降约0.29%，优于DEMix和EDL

**⚠️ 局限性**

需要在预训练阶段指定类别与专家子集，训练时仍有少量计算开销；若保留推理时偏置会导致推理吞吐下降；过大偏置会产生训练测试不匹配，且对主导类别的提升有限

---

## 438. EmoGavel: Grounding-Aware Reinforcement Learning with Verifiable Rewards for Multimodal Emotion Preference Learning

**arXiv ID:** 2610.13353 | [PDF](https://arxiv.org/pdf/2610.13353v1)

**作者:** Zilong Huang `[一作]` (Hong Kong Polytechnic University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EmoGavel 框架，结合 Grounded SFT 与 GA‑GRPO 两阶段训练，实现多模态情感偏好学习并能对错误进行可验证的定位、修复和缺失证据恢复。

**💡 创新点**

创新点：①利用控制错误构造所提供的监督信息，赋予模型错误定位与修复能力；②在 SFT 阶段加入结构化输出，联合学习偏好预测与错误纠正；③在 GRPO 阶段设计偏好门控奖励与动态零方差救援，既保障偏好正确性，又恢复所有错误组的学习信号。

**🔧 技术方法**

技术：LoRA 微调、自动回归交叉熵损失、基于奖励的 GRPO 强化学习、可验证奖励（token‑level F1 与 cosine similarity）、动态零方差救援机制、KL 正则化。

**📊 数据集**

数据集：EmoPrefer 以及其 EAPO 扩展（四种控制错误类型）作为训练与验证，使用 Qwen2.5‑Omni‑7B 作为基准多模态 LLM。

**📈 对比分析**

方法对比：与零射击、SFT、Error‑Aug SFT 以及多模型融合的 EAPO 进行对比。单模型在自然对数 WAF 79.75%，控制错误平均 88.20%，错误定位宏平均 77.27%，在自然偏好和控制错误鲁棒性上均优于传统基线，并在单模型水平上逼近多模型融合效果。

**⚠️ 局限性**

局限性：GA‑GRPO 在缺失证据恢复（Modality Omission）方面表现不如 Grounded SFT；只针对四种预定义错误类型，未覆盖更自然、多样化的错误；需要进一步评估在不同模态证据条件下的错误定位与修复效果。

---

## 439. Understanding Mamba-3 Through Oscillatory Dynamics

**arXiv ID:** 2610.13341 | [PDF](https://arxiv.org/pdf/2610.13341v1)

**作者:** Shutian Liang `[一作]` (Tsinghua University), Yue Song `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对Mamba-3中的输入相关旋转进行深入研究，解释其为欠阻尼二阶动力学，并通过实验验证其在状态跟踪和长上下文语言建模中的优势。

**💡 创新点**

创新点在于将Mamba-3的旋转视为欠阻尼二阶系统，提供统一的振荡动力学框架，并揭示振荡在长上下文处理中的关键作用。

**🔧 技术方法**

使用了状态空间模型、欠阻尼二阶动力学解析、对比实验（欠阻尼、临界阻尼、过阻尼）、语言模型预训练、阶段性旋转干预和多数据集评测。

**📊 数据集**

数据集包括FineWeb‑Edu、PG‑19、LongBench‑E、CodeParrot、MATH‑Hard、TriviaQA、SlimPajama、LAMBADA等。

**📈 对比分析**

通过在状态跟踪任务（Parity、Mod 3、Arithmetic）和语言建模任务（PG‑19、LongBench‑E）中与其他阻尼状态相同的基线比较，发现欠阻尼版本在状态跟踪上几乎100%准确，在长上下文语言建模中比非振荡版低10–20% perplexity，且在大规模模型上得到明显的下游提升。

**⚠️ 局限性**

局限在于缺乏对没有振荡训练但仍能实现长上下文稳健性的深入理解，以及对旋转频率与任务复杂度关联的理论解释仍不完整。

---

## 440. Local Diagnostics of Learned Event Representations for Anomaly Detection at the HL-LHC

**arXiv ID:** 2610.13437 | [PDF](https://arxiv.org/pdf/2610.13437v1)

**作者:** Donatella Genovese `[一作]`, Thea Klaeboe Aarrestad `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对在 HL‑LHC 条件下使用 Transformer 编码器学习的低维事件表示进行研究，并在此表示上构建基于自编码器的异常检测模型，同时提出一种基于高斯混合模型（GMM）的本地诊断框架，用于定位和解释被标记为异常的事件。

**💡 创新点**

① 通过对比监督与自监督学习目标，证明利用模拟的 SM 过程标签的监督目标能获得更具判别力的嵌入；② 开发了仅使用嵌入、异常得分和一组物理观测量的 GMM 诊断方法，可在不使用过程标签的情况下重现信号的物理特征；③ 揭示异常得分与局部区分力观测量之间可能的解耦，说明单纯依赖全局指标难以把握模型行为。

**🔧 技术方法**

Transformer encoder + 物理动机的数据增强（azimuthal 旋转、longitudinal boost、pT 模糊、随机遮蔽）；对比的学习目标包括 SimCLR（Self‑Sup 与 SupCon）、VICReg、VCReg 及传统交叉熵（CE）；异常检测采用仅在 QCD 背景上训练的全连接自编码器；本地诊断使用 GMM 对 SM 嵌入空间进行分区，并利用 Wasserstein 距离和 Spearman 相关系数评估局部区分力与异常得分的关联。

**📊 数据集**

Collide‑2V 数据集：约 7.5 亿 HL‑LHC 级别的模拟碰撞事件，包含 12 种 SM 背景过程以及两个在训练中保持保留的信号（HH→4b 作为 SM 代理，Z'→n(μμ) 作为 BSM 信号）。

**📈 对比分析**

通过线性探针 AUROC、全局异常检测 AUROC 以及在 10% 背景接受率下的信号效率进行评估。监督目标的嵌入（SupCon、VCReg）在所有评估指标上均优于自监督目标和直接在高维特征上训练的 AE；在最佳配置下，VCReg 达到 HH→4b 信号效率约 97%（10% 背景接受率），Z'→n(μμ) 近 99%。GMM 诊断能准确恢复信号的主要物理变量（HH→4b 的 b‑jet 数量、Z' 的轻子数目），并揭示异常得分与这些变量的相关性并非完全一致。

**⚠️ 局限性**

① 自监督学习的表现仍低于监督方法，表明当前的自监督目标难以充分利用大量无标签数据；② 监督方法依赖于模拟的 SM 过程标签，若模拟失真或在真实数据中缺失标签将导致嵌入偏差；③ GMM 诊断在高维嵌入上仅使用 PCA 降维，可能忽略高维结构；④ 诊断框架需要事后对所有高层观测量进行评估，扩展到更复杂的物理量时可能面临计算成本与解释性难题。

---

## 441. When AI Advisors Agree: Distinguishing Forecast Risk from Forecast Reliability

**arXiv ID:** 2610.13431 | [PDF](https://arxiv.org/pdf/2610.13431v1)

**作者:** Russlan Ramdowar `[一作]` `[通讯]` (Future Edge Group FZE), Russlan Ramdowar (Future Edge Group FZE)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了七个AI投资框架在同一资产上的预测共识对回报预测精度的影响；

**💡 创新点**

发现单纯的预测一致性并不能提升回报预测精度，只有方向一致性与方向准确率相关，而整体波动率筛选更能降低误差；

**🔧 技术方法**

使用多框架平均、误差与无变更基线对比、四分位差异、回归调节等统计技术；

**📊 数据集**

数据集为iPulse AI 2026年四个批次的232个资产共928个一季度预测结果；

**📈 对比分析**

通过比较预测一致性四分位与波动率四分位、回归模型等方法，结果表明共识并未显著提升预测准确率，波动率筛选得到更低的平均绝对误差；

**⚠️ 局限性**

局限包括仅覆盖四个重叠评估期、框架并非独立、样本非随机、未进行外部验证，且未包含交易成本与执行细节。

---

## 442. Why Linear Models Can Still Work in a Complex and Changing World

**arXiv ID:** 2610.13404 | [PDF](https://arxiv.org/pdf/2610.13404v1)

**作者:** Fangzhou Lin `[一作]` (Texas A&M University), Kazunori D Yamada `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并证明在有限数据、部分观测且环境变化导致非线性规则不稳定的情形下，线性模型在目标环境中的均方误差可能低于灵活非线性模型。

**💡 创新点**

提出条件性解释，将多个研究传统（简单线性规则、偏差-方差、最佳线性预测、测量误差、环境变化、Rashomon集、欠定性）整合，并给出Proposition 1的数学证明，阐释何时忽略非线性规则会更稳健。

**🔧 技术方法**

使用理论推导与误差比较（均方误差）、正交分解、贝叶斯假设、正则化、测量误差模型，以及对不同环境下的预测性能进行数学阈值分析。

**📊 数据集**

论文主要为理论分析，没有使用具体实验数据集；所有结果基于假设模型与数学推导。

**📈 对比分析**

通过对目标环境预测误差的解析比较，给出阈值A_{s,t}<1/2（或1/2+V_F-V_L/2‖g_s‖²）时线性模型优于灵活模型；表明在特定条件下线性模型可获得更低的均方误差。

**⚠️ 局限性**

局限性：只在极简假设（X分布相同、g_e无线性部分、正交性等）下成立；不适用于分类或非均方误差指标；未考虑测量过程变化对线性部分的影响；未给出经验验证；对实际数据的可推广性有限。

---

## 443. RegulaTutor: A Theory-Informed Framework for Long-Term Personalized Learning

**arXiv ID:** 2610.13399 | [PDF](https://arxiv.org/pdf/2610.13399v1)

**作者:** Yonggang Li `[一作]` (Hong Kong University of Science and Technology), Zixin Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RegulaTutor 框架及 PairRoute 细化流程，通过将学习者历史与自我调节学习理论（CAMM）相结合，生成基于证据的个性化教学策略。

**💡 创新点**

创新点在于：①把 CAMM 指导的解释与教学策略显式关联；②利用离线执行轨迹的 PairRoute 对工具链进行针对性、阶段级的优化。

**🔧 技术方法**

采用技术包括：多阶段证据解析、LLM 与自定义 harness、SRL 依据的解释与策略生成、离线阶段级监督 fine‑tune、模型‑harness 兼容性评估。

**📊 数据集**

使用数据集为 LongTutor benchmark（1,000 例人类注释、199 条学习历史记录）以及 Qwen3.5‑4B 进行模型检查。

**📈 对比分析**

比较方法为固定骨干模型与 RegulaTutor harness 的对比、与 DeepTutor 基线的对标以及跨模型‑harness 兼容性评估；性能提升平均为证据准确率 +13.2pp、诊断准确率 +11.6pp，教学得分提升 0.13pp；PairRoute 在开发集上每轮提升约 6pp，终点提升 8.2pp。

**⚠️ 局限性**

局限性包括：未评估真实学习者的长期保留、迁移与进步；评估依赖人工判断且评审者与模型同族；仅针对单一主干和有限轮次的优化；缺乏因果效应和公平性验证。

---

## 444. Towards the Roof of the World: NLP Progress in Tibetan Languages

**arXiv ID:** 2610.13398 | [PDF](https://arxiv.org/pdf/2610.13398v1)

**作者:** Xi Cao `[一作]` (Minzu University of China), Nuo Qun `[通讯]` (Xizang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对2004–2026年间140篇Tibetan NLP论文进行系统的文献计量调查，按出版场所、研究领域、作者国别及资源发布情况进行标注和分析。

**💡 创新点**

首次提出“Tibetan NLP完整流水线”范式，展示从语料、词分割、文字、语音到应用再到LLM的阶段性演进，并指出资源缺口与开放科学的重要性。

**🔧 技术方法**

采用文献计量分析与定量统计技术，对ACL Anthology和DBLP中的论文进行元数据提取、分类与趋势可视化。

**📊 数据集**

使用的主要数据集为ACL Anthology与DBLP中的140篇论文元数据，涵盖论文标题、作者、年份、机构等信息。

**📈 对比分析**

该研究并未评估模型性能，而是通过趋势图和分布图比较不同研究领域与国家的贡献，显示自2018年以来产出激增、研究领域从语料到LLM逐步扩展、以中国研究者为主。

**⚠️ 局限性**

局限性包括资源发布率仅29%，方言覆盖不足，标注数据极为稀缺，缺乏统一评价标准，且研究主要集中在中国，缺少跨社区可见度和跨国合作。

---

## 445. ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents

**arXiv ID:** 2610.13395 | [PDF](https://arxiv.org/pdf/2610.13395v1)

**作者:** Chengyi Yang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为ProgramGym的框架，包含Program-Env用于生成可验证的从源码到可执行的训练环境，并提供黑盒测试；Program-Agent采用角色隔离的长周期工作流（分析、实现、审查）配合修订管理来训练和执行从零开始的程序重现任务。

**💡 创新点**

创新点在于：① 通过反馈驱动的构建管线自动化生成可执行参考程序、沙箱和覆盖率驱动的黑盒测试；② 采用多阶段、角色隔离的迭代工作流与修订管理，显著提升了长周期程序重现的可靠性和性能；③ 在多语言（C/C++/Go/Rust）上实现了规模化（1648个）可验证训练环境。

**🔧 技术方法**

使用了agent harness（如SWE-gym、SWE-smith等）、覆盖率分析、动态/静态程序分析、四种验证检查（通过、双重、一致性、弱断言）来构建测试；实现了容器化沙箱、依赖准备脚本、编译检查与修订回退机制；训练时使用了ms-swift框架对大语言模型进行微调。

**📊 数据集**

数据集为从GitHub上采集的30,459个开源仓库，经过筛选后生成1,648个包含可执行程序、文档、黑盒测试和沙箱的训练环境，覆盖C、C++、Go、Rust等四种语言。

**📈 对比分析**

通过在ProgramBench上对比多种基线（GPT、Claude、Gemini、DeepSeek等）和模型微调（SFT）结果，验证了：① 训练于Program-Env的模型从34.87%提升至42.84%；② Program-Agent将模型进一步提升至55.71%/56.25%，均超过部分旗舰模型；③ 迭代预算从K=1到K=5的提升也明显，表现出可扩展性。

**⚠️ 局限性**

限制包括：① 仍难以实现全程序复现，Almost Resolved和Resolved指标仍低；② 训练与评估成本高（需大量构建与测试）；③ 依赖于预先生成的可执行沙箱，可能对特殊工具链或网络依赖的项目支持不足。

---

## 446. Maneuver Coordination Using V2I to Improve Traffic Efficiency at Intersections

**arXiv ID:** 2610.13394 | [PDF](https://arxiv.org/pdf/2610.13394v1)

**作者:** Lorenzo Farina `[一作]` (Universitá di Bologna), Alessandro Bazzi `[通讯]` (Universitá di Bologna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于V2I通信的FIFS（First-In‑First‑Scheduled）调度算法及其通信协议，旨在使连接式自动驾驶车辆（CAV）在无需红绿灯的交叉口中以最快速度通过交叉口，除非出现严重拥堵，否则不必停驶。

**💡 创新点**

创新点包括：①在调度算法中完整考虑转弯车辆，并采用单轨迹定义方案，避免车辆多次改变轨迹；②详细阐述了车辆与交叉口控制器之间的通信协议，支持低延迟、一次性交互；③设计了“备份模式”，在高拥堵情况下允许车辆退回至常规行驶规则，保证安全。

**🔧 技术方法**

技术手段主要包括：V2I通信（利用SAE/ETSI maneuver coordination消息）、MEC边缘计算、Python脚本与SUMO仿真（TraCI接口）以及对通信延迟的随机建模。

**📊 数据集**

实验使用基于Poisson分布的车辆生成（不同进路方向、行驶方向随机），并在不同车辆密度下（0.05–0.20车/秒/方向）进行仿真；未使用公开真实数据集，而是通过仿真自行生成交通场景。

**📈 对比分析**

通过与四种基准方案（优先级、红绿灯、理想通信FIFO、理想通信FIFS）对比，采用行驶时间和CO₂排放两项KPI。结果显示：在所有密度下，理想通信FIFS表现最佳；即使在现实V2I延迟（20–100 ms）下，FIFS仍明显优于FIFO、红绿灯和优先级方案，尤其在中等至高密度交通中，行驶时间和排放均显著降低。

**⚠️ 局限性**

局限性包括：仅考虑CAV，未纳入非连接车辆或行人/VRU；假设通信可靠、无丢包；仅验证单车道四叉路口，未扩展至多车道或环形交叉口；当密度达到最高时，备份模式会触发，导致性能下降。

---

## 447. BLOKS: Scaling Kernel Solvers with Sketched Block Preconditioning

**arXiv ID:** 2610.13382 | [PDF](https://arxiv.org/pdf/2610.13382v1)

**作者:** Giordano Vitale `[一作]` (Università degli Studi di Genova), Lorenzo Rosasco `[通讯]` (Università degli Studi di Genova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于共轭梯度的Nyström-草图KRR求解器，旨在解决大规模数据集上的内存和计算效率问题。

**💡 创新点**

创新点在于引入了一种新的预处理器，结合了块Jacobi近似和草图技术，使得内存成本与数据集大小和模型参数数量无关，从而能够扩展到大模型。

**🔧 技术方法**

使用了共轭梯度法（CG）和块Jacobi预处理器，结合了草图技术以提高计算效率。

**📊 数据集**

在标准的大规模数据集上进行实验，具体数据集未详细列出，但提到使用了多个用于基准测试的标准数据集。

**📈 对比分析**

与现有的最先进方法相比，提出的方法在准确性上表现出色，同时保持了良好的时间/内存复杂度，能够处理高达320万参数的模型。

**⚠️ 局限性**

限制在于尽管提出的方法在内存使用上进行了优化，但在某些情况下仍可能受到模型复杂性和数据集规模的影响。

---

## 448. Towards Safe World Models: Risks, Defenses, and Evaluation

**arXiv ID:** 2610.13372 | [PDF](https://arxiv.org/pdf/2610.13372v1)

**作者:** Xiaoma Wang `[一作]` (Beijing University of Posts and Telecommunications), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了世界模型的安全问题，提出了六阶段安全生命周期框架，系统归纳了非对抗性失效、攻击、对策与评估，并构建了相应证据库。

**💡 创新点**

创新点在于将安全关注点映射到六阶段生命周期，区分了证据直接性与结果级别，并为未来研究提供了结构化的分析视角。

**🔧 技术方法**

采用文献检索、结构化证据编码与框架构建技术，整合了163篇相关工作。

**📊 数据集**

未使用传统数据集，主要基于公开文献与已有的基准集合（如SafeBench、SAFE-SMART等）进行归纳。

**📈 对比分析**

本文不做算法比较，而是以概念映射和证据分类评估方法有效性；通过案例示例展示如何在六阶段中定位风险与对策。

**⚠️ 局限性**

局限在于缺乏实证验证与量化指标，依赖已发表文献的完整性，框架对新兴模型与部署场景的适用性尚待验证。

---

## 449. AgentSWE: Can Coding Agents Build the Agent You Actually Want?

**arXiv ID:** 2610.13361 | [PDF](https://arxiv.org/pdf/2610.13361v1)

**作者:** Jiahao Wang `[一作]` (Beijing Academy of Artificial Intelligence), Zheng Liu `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AgentSWE 基准，用来评估编码代理在三阶段生命周期（创建、编辑、优化）中，能否根据自然语言需求交付符合需求的可执行代理软件。

**💡 创新点**

创新点在于：①将代理软件工程定义为可测量问题，构建三阶段任务集合；②通过无参考实现的“委托”任务，以可观测行为而非代码质量来衡量交付的真实性；③通过 held‑out 评估和严格的有效性门限揭示编码代理的常见失败模式。

**🔧 技术方法**

使用的大模型包括 GPT‑5.6 Sol、GPT‑5.5、DeepSeek‑V4‑Pro/Flash、Qwen3.6‑35B‑A3B，结合 Codex、DSH、OpenCode、Pi、GLM 等不同 harness；评估框架为 Harbor，模型调用通过 benchmark‑owned gateway；任务、评估 rubrics 和案例生成则由 Codex + GPT‑5.6 Sol 自动化完成。

**📊 数据集**

数据集由 25 个任务组成（10 创建、10 编辑、5 优化），每个任务配有 2 条开发案例和 6 条 held‑out 案例，总计 150 开发案例、365 held‑out 案例，来源为公开 GitHub 仓库（如 OpenHands、Aider 等）和公开基准（BrowseComp、τ³、TerminalBench、PinchBench、OSWorld）。

**📈 对比分析**

比较方法：在 Codex harness 下对 5 种 builder 模型进行统一评估，记录 Creation/Editing 的平均 Result 分数和 Optimization 的 gap‑closure；并对不同 harness‑model 组合、开发案例数量、提示先验等进行对比实验。结果显示：最佳模型平均 Creation 19.1、Editing 31.6，优化平均负 4% 的剩余差距，整体性能远低于预期，揭示过拟合、空运行、误报等失败模式。

**⚠️ 局限性**

限制包括：①任务和案例范围有限，未涵盖全部真实需求；②评估受限于评判器和模型版本，无法覆盖所有语言/平台；③未对长时间迭代和持续训练的效果进行评估；④缺乏对实际生产环境可部署性和安全性的深入考量。

---

## 450. SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories

**arXiv ID:** 2610.13356 | [PDF](https://arxiv.org/pdf/2610.13356v1)

**作者:** Jiajun Hu `[一作]` (Leibniz Research Center), Chun Yong Chong `[通讯]` (School of Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个后训练流水线，用监督微调(SFT)和强化学习(RL)训练通用的编码代理轨迹验证器。

**💡 创新点**

创新点在于将过程级别的通用验证与CoT（思维链）指导结合，使用教师LLM差分分析生成少量高质量训练样本，并通过RL在大规模未标注轨迹上进一步提升判别性能。

**🔧 技术方法**

技术包括SFT（交叉熵）、GRPO强化学习、MoE Qwen3-30B、MindSpeed-LLM、VeRL、vLLM Ascend推理，奖励函数基于单元测试结果。

**📊 数据集**

使用公开的CodeForge-Preview数据集（约25.8万轨迹，含已通过/失败的测试），对轨迹进行过滤、划分为SFT、RL和测试集。

**📈 对比分析**

与基线Qwen3-30B相比，SFT后准确率提升至66%，RL后进一步提升至78%，在不同子集（R2E-Gym、SWE-Rebench、SWE-Smith）均表现出显著提升，尤其在正负召回上均有改善。

**⚠️ 局限性**

局限性包括：对R2E-Gym子集表现相对较差；失败轨迹的负召回仍有限（约64%），表明失败模式多样且难以捕捉；实验仅评估判别能力，未覆盖下游应用与更广泛的奖励信号。

---

## 451. Evo-RL: Iterative Robot Reinforcement Learning from Selective Experience

**arXiv ID:** 2610.13345 | [PDF](https://arxiv.org/pdf/2610.13345v1)

**作者:** Kai Ye `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种迭代强化学习框架 Evo-RL，结合人类干预和优势引导的经验选择，持续改进真实世界机器人的策略。

**💡 创新点**

创新点包括：①仅保留成功干预轨迹中的人类控制片段进行纠正；②利用估计优势挑选正优势片段进行经验重用；③将两种经验策略相结合，实现快速、可控的迭代改进；④公开 RW‑RL‑Dataset，聚合演示、干预与自治回放，支持进一步研究。

**🔧 技术方法**

技术方案涵盖：预训练的视觉‑语言‑动作（VLA）模型（PaliGemma + Flow‑matching）；行为克隆与增量 SFT；vision‑only value 模型学习剩余回报；优势估计与优势引导采样；与 RECAP 进行对比实验。

**📊 数据集**

使用了 RW‑RL‑Dataset，超过 1,000 小时、四种机器人实现，覆盖 9 片场景与 30+ 任务模板，包括螺丝插入、衣物折叠等。

**📈 对比分析**

通过与 SFT+Int、SFT+RL、RECAP 等方法的对比评估：在 PiperX 螺丝插入任务上，首轮 Evo‑RL ASR 88.24% / TP 60.61，显著高于 RECAP 的 57.78%/35.54；第二轮进一步提升至 92.16%/67.14；在 SO‑101 折叠任务上，ASR 从 55.56% 提升到 76.67%，TP 从 75.78 提升到 90.00。

**⚠️ 局限性**

局限性包括：仍需人工干预，干预成本不可忽略；验证仅在高精度插入和柔性物体折叠两类任务，泛化到更复杂任务需进一步测试；优势估计依赖 vision‑only value 模型，环境变化可能影响其可靠性；迭代次数与收敛速度尚未给出通用准则。

---

## 452. A precision-driven evaluation protocol for author name disambiguation

**arXiv ID:** 2610.13415 | [PDF](https://arxiv.org/pdf/2610.13415v1)

**作者:** Francesca Cappelli `[一作]` (University of Bologna), Silvio Peroni `[通讯]` (University of Bologna)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套结构化、精度导向的作者姓名消歧评估框架，能够将聚类质量分解为配对一致性、实例级B³指标和聚类级拆分/合并错误，并将三者融合为单一综合评分；

**💡 创新点**

核心创新在于引入聚类层面的拆分误差（SE）和合并误差（LE）对传统指标的结构化诊断，配合精度优先的加权组合提升模型选择的可解释性与鲁棒性；

**🔧 技术方法**

采用Python实现统一评估管道，计算配对指标、B³指标、LE/SE以及综合得分，并通过控制实验与灵敏度分析验证权重配置；

**📊 数据集**

实验使用了两套公开的作者姓名消歧基准数据集（具有不同元数据可用性），并在合成数据上进行结构化案例验证；

**📈 对比分析**

通过与传统F1_PW、B³_F1的对比，发现综合评分能区分只合并、只拆分或混合错误的模型；在合成案例中，权重稳定区间内排名保持一致，说明方法在不同配置下具有可重复性；

**⚠️ 局限性**

局限包括：1）需要对缺失的真实标注进行过滤，导致结果受数据集完整度影响；2）配对指标的O(n²)复杂度在大论文集上成为瓶颈；3）宏观平均LE/SE可能掩盖高歧义名的系统性失误；4）S_struct对LE/SE的对称处理未能自然体现精度优先的成本模型，需要外部优先策略来补偿。

---

## 453. Ember: Making Unified Virtual Memory Practical for Discrete-Memory GPUs

**arXiv ID:** 2610.13408 | [PDF](https://arxiv.org/pdf/2610.13408v1)

**作者:** Senad-Leandro Lemes Galera `[一作]` (Technical University of Munich), Dimitrios Stavrakakis `[通讯]` (Technical University of Munich)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于 NVIDIA UVM 的框架，支持 GPU 内存超额使用，使用分区映射与按需迁移实现高效的工作集管理。

**💡 创新点**

核心创新包括：1) 以映射为默认的混合地址空间；2) 利用 GPU 硬件访问计数器驱动热点区域提升；3) 采用衰减驱动的急促回收，提前批量驱逐冷数据，避免执行路径阻塞。

**🔧 技术方法**

实现基于 CUDA UVM 驱动的两种变体：-static（用户空间调用，静态划分热点）和 -adaptive（内核模块，运行时计数器驱动、衰减回收），并利用 GPU 硬件计数器、页表迁移、批量迁移等技术。

**📊 数据集**

在四台 GPU（RTX 5080、A40、H200、GH200）上使用多种工作负载：UVM‑Bench（排序、BFS、bitonic sort）、LLM 推理（Qwen2.5‑14B、Qwen3‑32B）、数据库分析（SSB 查询）以及自定义随机/线性访问的微基准。

**📈 对比分析**

对比 stock‑UVM、UVM‑mapped、Prefetch、HELM 等基线，-static 与 -adaptive 在 0.5× 与 1.5× GPU 内存占用下平均可获得 6.8–10.6× 的速度提升，LLM 生成速率提升 4.7–13×，并在无编译或运行时手工配置的情况下实现与或优于基于调优的静态策略。

**⚠️ 局限性**

局限性包括：1) 无法管理非 uvmMalloc 的库内部缓冲；2) 对已驻留区的冷却判定仅基于衰减估计，可能误判；3) -adaptive 需要裸机加载内核模块且需随驱动版本更新；4) 仅支持单 GPU 与主机的两层内存，未覆盖多 GPU 或多级内存环境。

---

## 454. Robot-WATCHDOG : failure detection through object-centric graph representation

**arXiv ID:** 2610.13403 | [PDF](https://arxiv.org/pdf/2610.13403v1)

**作者:** Quentin Rolland `[一作]` (Université Paris-Saclay), Jean-Baptiste Mouret `[通讯]` (Inria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于对象中心时空图的实时失败检测框架 Robot-WATCHDOG，能够对机器人操控过程中的关系异常进行自监督监测。

**💡 创新点**

创新点：① 将失败检测转化为监测对象交互动力学的异常；② 提出了两种评分函数——非参数的轨迹投影与相关性（TPC）和基于图注意力网络+Transformer的预测模型（GnnT）；③ 通过合成预测（Conformal Prediction）实现无阈值、无失败样本的在线校准。

**🔧 技术方法**

使用的技术包括对象中心时空图表示、GATv2 图注意力、Transformer 时空编码、轨迹投影距离与相关性度量、以及合成预测阈值校准。

**📊 数据集**

实验使用了公开的 BotFails 基准数据集（4 个任务）以及作者自建的结构化桌面设置和移动输送带垃圾分类两个任务的数据集。

**📈 对比分析**

与视觉/轨迹匹配、密度/重建以及 FIDeL 等多种基线比较，TPC 在 AUPR 与 MCC 指标上均显著优于所有基线，尤其在关系级错误上表现突出；GnnT 在连续动态场景中接近 TPC，但整体性能低于 TPC。

**⚠️ 局限性**

局限性：依赖离散对象抽象，难以处理柔性物体、流体或无明显对象分解的交互；受视觉追踪精度影响；仅利用对象质心位置，无法捕捉姿态或细粒度姿势失效。

---

## 455. AlphaNIST: Emergent Assembly Skills from Binary Task Rewards

**arXiv ID:** 2610.13354 | [PDF](https://arxiv.org/pdf/2610.13354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 456. PlacePulse: Learning Transferable Place Representations from Structural Geographic Data

**arXiv ID:** 2610.13390 | [PDF](https://arxiv.org/pdf/2610.13390v1)

**作者:** Lucía García-Duarte `[一作]`, Javier de la Torre `[通讯]` (CARTO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个基于自监督masked autoencoder的结构化特征模型，生成了覆盖全美 H3 resolution‑7 的 1,055,537 个单元的 256 维 L2‑正则化嵌入（PlacePulse）

**💡 创新点**

首次证明仅利用静态结构属性即可构建全尺度地理基础模型，并提出独立评估内在质量与下游预测性能的两轴基准框架

**🔧 技术方法**

使用 masked reconstruction 自监督学习、H3 k‑ring 聚合特征、全连接 autoencoder、L2 归一化、LightGBM 监督预测、Moran I 等空间统计量

**📊 数据集**

输入为 Applied Geographic Solutions 的 572 结构属性（经 H3 1‑和 2‑环扩展至 2,288 特征），评估目标来自 12 个公开数据源（健康、经济、投票、POI 等）

**📈 对比分析**

通过 5‑fold 空间块交叉验证，将 PlacePulse 与无标签的维度匹配特征选择基线以及 Google PDFM 进行对比；在结构相关目标上表现略优，动态/影像模型在部分目标上更强

**⚠️ 局限性**

限制包括：输入为商业估计且不可复现；未验证跨国或新时间点泛化；仅试验单一模型配置；评估目标与输入存在重叠；未公开模型代码和嵌入

---

## 457. Secrets in Preamble: Practical and Robust PHY Information Hiding for Wi-Fi

**arXiv ID:** 2610.13367 | [PDF](https://arxiv.org/pdf/2610.13367v1)

**作者:** Rui Pan `[一作]` (Southeast University), Junqing Zhang `[通讯]` (University of Liverpool)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Wi‑Fi协议的短训练字段（L‑STF）中，利用一对神经网络（Embedder/Extractor）将秘密信息以微小、不可察觉的波形畸变形式嵌入，并通过对等的网络解码在接收端恢复秘密，同时保持原始Wi‑Fi链路正常工作。

**💡 创新点**

创新点：①将秘密隐藏直接嵌入L‑STF的频域子载波，并通过保持周期性实现对包检测与粗调频补偿的兼容；②采用端到端联合训练的神经网络配对，自动学习最大化隐写容量与鲁棒性的嵌入/提取规则；③在训练中引入TGn多径信道模拟器和L‑STF等化，显著提升在不同环境下的可靠性。

**🔧 技术方法**

主要技术：一维卷积神经网络（1D‑CNN）作为Embedder和Extractor；端到端自动编码器训练；IEEE 802.11 TGn信道模型（Model‑B~F）模拟；L‑STF等化（CSI除法）；SDR硬件原型（USRP B210 + GNU Radio）实现。

**📊 数据集**

数据集：①随机生成的M‑bit秘密序列；②基于IEEE 802.11 TGn Model‑B的CSI库（1000个多径实例）；③实验时使用的实测I/Q样本（室内/室外环境）。

**📈 对比分析**

对比方法：与现有Wi‑Fi隐写方案（2‑bit/几位）和常规信道编码。性能：仿真在20–25 dB SNR下，46 bit秘密的SBER<0.17；在不同TGn模型（B–F）下，等化后SBER<0.009；硬件实验在室内室外均保持SBER<0.07（最差0.17），且未出现包丢失；相较于传统方法，隐写容量提升约20倍，鲁棒性显著增强。

**⚠️ 局限性**

局限性：①每个包的秘密容量仍受限（最高46 bit）；②目前仅验证了20 MHz单输入单输出（SISO）非HT模式，扩展到更高频宽、MIMO或未来Wi‑Fi标准需改进；③性能受硬件失真、功率限制影响，需更高功率设备或更高阶误差校正；④密钥分发仍是安全瓶颈（网络模型权重需安全共享）。

---

## 458. PartiGen: Part-Aware Representations for Language-Conditioned Humanoid Motion Generation

**arXiv ID:** 2610.13373 | [PDF](https://arxiv.org/pdf/2610.13373v1)

**作者:** Chuxuan Huang `[一作]`, Yue Gao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PartiGen系统，实现从语言到机器人运动参考的生成与执行

**💡 创新点**

引入部件感知注意力与前向运动学监督，实现端点预测与可变时长

**🔧 技术方法**

部件感知VAE、潜在扩散模型、GRIT预训练控制器、CFG引导

**📊 数据集**

HumanML3D重定向至Unitree G1以及MuJoCo仿真

**📈 对比分析**

与TextOp、DART+GMR、MDM+GMR等基线相比，Segment-FID下降16.8%，R@1提升1.4%，姿态屏蔽通过率81.8%及G1硬件成功率96%

**⚠️ 局限性**

缺乏对复杂多部件协调动作的真实评估，端点预测仍依赖标注长度，硬件验证样本有限

---

## 459. VideoAWR: Reinforcing Video Flow Models with Decoupled-Timestep Advantage-Weighted Rectification

**arXiv ID:** 2610.13338 | [PDF](https://arxiv.org/pdf/2610.13338v1)

**作者:** Zheming Fu `[一作]` (Beihang University), Siming Fu `[通讯]` (Joy Future Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的视频生成强化学习框架 VideoAWR，利用优势加权校正（AWR）与流模型相结合，进行文本到视频的对齐优化。

**💡 创新点**

创新点在于将采样与优化锚点解耦：采用 EMA 引导的 roll‑out 保持奖励可分离，而训练目标仍对未引导的条件流进行校正，从而消除 CFG 相关的几何发散，并在推理时实现 CFG‑free 单分支，同时保持 CFG 语义。

**🔧 技术方法**

使用的技术包括基于流匹配的回归目标、EMA 参考策略、行为锚定优势加权校正、以及解耦时间步的 DTR（Decoupled Timestep Rectification）来提升采样效率。

**📊 数据集**

实验采用 Wan2.1 1.3B 与 14B 两个版本的模型，并在 HPSv3、VideoAlign motion score 以及 VBench 指标上评估。

**📈 对比分析**

与 Flow‑GRPO、Flow‑GRPO‑Fast、Flash‑GRPO 以及 DiffusionNFT 等基线相比，VideoAWR 在 HPSv3 视觉奖励、运动质量以及 VBench 的美学、主体一致性和对象类别指标上均取得了更高或相近的性能，尤其在 CFG‑free 和 CFG‑enabled 两种推理模式下均保持稳定提升。

**⚠️ 局限性**

局限性在于该方法仅利用基线模型的 EMA 引导 roll‑out，受限于基线生成能力；若基线无法渲染某些提示，则无法获得有效奖励梯度，未来可探索离线或异域终点的无偏整合。

---

## 460. Exploring Attention Sinks in Text-to-Image Transformers from Diffusion and Autoregressive Models

**arXiv ID:** 2610.13337 | [PDF](https://arxiv.org/pdf/2610.13337v1)

**作者:** Binglei Li `[一作]` (Fudan University), Hao Li `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对文本到图像Transformer中的注意力沉积（attention sink）现象进行系统研究，比较扩散模型与自回归模型中的沉积模式及其对生成的功能影响。

**💡 创新点**

首次将沉积现象在多模态、不同生成阶段、不同层头、不同模型结构中统一比较，并通过对沉积头和位置的定向干预与注册token实验验证其功能性与可控性。

**🔧 技术方法**

采用多头自注意力分析、沉积头排名、位置比例指标、时间/提示稳定性评估；干预技术包括logit抑制、值零化、头剪枝；引入可学习注册token以重新分配沉积注意力。

**📊 数据集**

使用公开评测基准GenEval（553个组合式提示）和DPG‑Bench（1065个细粒度对齐提示），并结合HPSv2与ImageReward等感知质量指标。

**📈 对比分析**

对比基线模型与三种干预策略，使用GenEval、DPG、HPSv2、ImageReward四个指标评估。结果显示：logit抑制影响最小；值置零和头剪枝对性能有明显下降（尤其自回归模型）；注册token实验能部分转移沉积注意力，同时保持或提升生成质量。

**⚠️ 局限性**

仅评估三种模型，缺乏不同规模、训练细节与训练阶段的沉积动力学；干预结果与模型架构高度相关，未探索通用性；注册token仅在FLUX.1上验证，未评估跨模型迁移。

---

## 461. Goal-Oriented Ordered Autoencoding for Successive Refinement in Progressive Communication

**arXiv ID:** 2610.13426 | [PDF](https://arxiv.org/pdf/2610.13426v1)

**作者:** Qi Liao `[一作]` (Nokia Bell Labs), Dario Bega `[通讯]` (Nokia Bell Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种目标导向的分阶段自动编码器GO-OAE，用于在有限通信预算下实现图像分类任务的逐步推理。

**💡 创新点**

将目标导向失真目标与加权条件信息瓶颈联系起来，并提出无偏多前缀重要性采样MPIS降低训练成本；同时引入离散量化与自适应排序以实现真正的任务可用表示。

**🔧 技术方法**

端到端学习的离散量化VQ‑AE、信息瓶颈理论、重要性采样MPIS、监督对比损失与熵正则化以及EMA更新等技术。

**📊 数据集**

MiniImageNet 100类细粒度分类数据集。

**📈 对比分析**

与重建后分类、无序目标导向VQ‑AE以及统一采样的OAE进行对比；在36个量化码索引（324比特/图像）下，GO‑OAE‑I‑SC在仅使用约70%前缀时即可获得94%–96%的原始图像下分类准确率。

**⚠️ 局限性**

在Transformer骨干上渐进学习效果不佳，且对实时低功耗设备的实际部署与更高分辨率压缩率仍需进一步验证与改进。

---

## 462. SADL: A Size-Adaptive Decision Layer for Chronic Stroke Lesion Segmentation in ISLES 2026

**arXiv ID:** 2610.13422 | [PDF](https://arxiv.org/pdf/2610.13422v1)

**作者:** Marek Wodzinski `[一作]` (Sano -- Centre for Computational Personalised Medicine International Research Foundation), Wojciech Szymanski `[通讯]` (Sano -- Centre for Computational Personalised Medicine International Research Foundation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了11个nnU-Net成员的集成，并在其加权软图上加入SADL决策层，该层通过估计病灶体积、大小分桶自适应阈值、连通组件裁剪和永不为空救援机制实现后处理优化；

**💡 创新点**

创新点在于将大小分桶自适应阈值与“永不为空”救援相结合的决策层，同时所有融合权重与阈值均在留出交叉验证上拟合，避免了对训练过程的干预；

**🔧 技术方法**

使用了nnU-Net基础模型、Dice+Top‑K损失、数据增强、重采样、损失再加权、SWA权重平均、ResEnc编码器，以及集成权重学习与SADL四阶段后处理；

**📊 数据集**

使用了ATLAS v2.0与SOOP共计1453张T1加权MRI，55个站点，其中91个站点作为site‑disjoint holdout进行评估；

**📈 对比分析**

采用官方ISLES2026评估代码在91个站点holdout上评测，SADL相较最优单模型提升Dice +0.62pp、lesion‑F1 +5.4pp，尤其在<0.5ml小病灶桶lesion‑F1提升4.4pp，统计显著(p=0.024)；

**⚠️ 局限性**

局限在于评估仅基于公开holdout，未验证隐藏测试集；部分pool策略如扩散增强未能提升性能；SADL在大病灶的Dice提升有限，整体依赖基础模型的质量。

---

## 463. Web$hell: A Unified Interface for Agentic Web Access

**arXiv ID:** 2610.13416 | [PDF](https://arxiv.org/pdf/2610.13416v1)

**作者:** Sunghwan Kim `[一作]` (Yonsei University), Dongha Lee `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的 Web 接口（UnifiedWebInterface），将搜索、浏览器交互和文件系统访问整合为单一访问模型；

**💡 创新点**

创新点在于将网页信息和操作抽象为可寻址对象，采用统一的页面与对象模式，并通过编译器将实时网页转化为可操作的结构化表示，保证了可操作性、一致性和持久性；

**🔧 技术方法**

核心技术包括基于 HTML 语义的确定性编译规则、对象地址与关联字段设计、页面地址系统、持久化缓存与状态跟踪、以及 Request–Response 交互模式；

**📊 数据集**

使用了 WebSearch（WideSearch、SealQA）、WebNavigation（WebArena-Verified、WebWalkerQA）以及 DeepResearchBench（DeepResearchBench、DeepWeb-Bench）等六个公开基准数据集；

**📈 对比分析**

与传统的搜索增强、文件系统、浏览器访问等基线方法（Fetch-then-Explore、CLI-Anything Browser 等）相比，UnifiedWebInterface 在所有六个基准上均取得最高任务完成率，并在 Deep Research 任务中显著降低了 LLM 推理代价（例如，使用 17.6 次推理轮次击败 CLI-Anything Browser 的 52.4 次），显示出更高效的性能；

**⚠️ 局限性**

主要局限在于：1) 依赖 HTML 结构，可能对高度动态或非标准页面处理不足；2) 目前只在单机实验中验证，跨域或大规模并发访问的可扩展性待进一步评估；3) 对于高度隐私或需要身份验证的站点，接口可能无法完整获取内容。

---

## 464. PPG-Bench: A Unified Benchmark for PPG Foundation Models

**arXiv ID:** 2610.13435 | [PDF](https://arxiv.org/pdf/2610.13435v1)

**作者:** Arvasu Kulkarni `[一作]` (Mannas AI), Siddharth Panwar `[通讯]` (Mannas AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了PPG-Bench基准，统一评估5个公开PPG基础模型在9个公开数据集、26个任务（心血管、活动、睡眠、筛查与识别）上的表现，并将其与三种对照（随机/恒定基线、PRV生理特征、通用时间序列FM MOMENT）以及四种下游适配方式（线性探针、随机森林、冻结MLP探针、端到端微调）进行比较；

**💡 创新点**

创新点在于首次构建统一、可扩展的评估框架，涵盖多数据集、多任务、多下游适配方式，并将PPG专属预训练模型与简单生理特征及通用时序模型直接对照，揭示当前PPG FMs在多任务上并未提供实质性优势；

**🔧 技术方法**

技术手段包括：1) 线性探针、随机森林、冻结MLP探针和端到端微调四种下游适配；2) Optuna+Hyperband实现模型特异化超参搜索；3) 受试者分离的5折交叉验证；4) 任务标签标准化与受试者级聚合策略；

**📊 数据集**

使用了9个公开PPG数据集，涵盖17个主要任务与9个次要任务，涉及心血管状态估计、活动监测、睡眠分期、情绪筛查和身份识别等；

**📈 对比分析**

通过与随机/恒定基线、PRV特征及MOMENT进行比较，在68个任务-下游组合中发现35.3%任务未明显优于随机基线，83.8%任务未超过PRV和MOMENT；仅少数任务表现优于对照，表明当前PPG FMs尚未形成通用优势；

**⚠️ 局限性**

局限性包括：1) 多任务、多数据集上表现不稳，许多任务接近随机或被简单特征匹配；2) 增加下游模型容量并不总能提升性能，易导致过拟合；3) 现有PPG FMs缺乏统一评估标准，导致模型对比不公平；4) 需要在评估中加入更简单的对照和受试者级评估以检验代表性。

---

## 465. TextVBench: Evaluating Text Rendering and Preservation in Video Generation

**arXiv ID:** 2610.13412 | [PDF](https://arxiv.org/pdf/2610.13412v1)

**作者:** Ruiqiang Zhang `[一作]` (University of Science and Technology of China), Weiming Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 TextVBench 评估框架，用于量化视频生成中多目标多实例的视觉文本生成与保持能力。

**💡 创新点**

创新点在于将视觉文本评估转化为多目标、多实例时空对应问题，并引入空间与时间相符性分数、长度加权谐波平均以及额外文本诊断。

**🔧 技术方法**

采用 OCR 检测 + 语言模型（Qwen3.8‑27B）进行候选匹配，使用字符级编辑相似度、长度加权、空间/时间分数求平均等技术计算最终 TextVScore。

**📊 数据集**

使用构造式 T2V 提示和多来源参考图像（包装、移动界面、应用界面等）共计 300 个视频和 200 个参考，构建完整的评估数据集。

**📈 对比分析**

对八个 T2V 模型和六个 Ref2V 模型进行对比，MiniMax‑H3 在 T2V 上取得最高 TextVScore 68.32，Ref2V 上前三名模型得分相近，整体显示不同模型在文本生成与保持方面存在显著差距。

**⚠️ 局限性**

局限性包括依赖 OCR 的准确性，难以处理极小或模糊文本、动态文本、长文本变形及多实例交叉等情形；当前评估不支持状态改变的文本和长文本的时间序列。

---

## 466. ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving

**arXiv ID:** 2610.13409 | [PDF](https://arxiv.org/pdf/2610.13409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 467. In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation

**arXiv ID:** 2610.13411 | [PDF](https://arxiv.org/pdf/2610.13411v1)

**作者:** Matteo Rufolo `[一作]` (SUPSI), Marco Forgione `[通讯]` (SUPSI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种零样本学习的系统辨识框架，通过预训练的Transformer元模型直接根据输入/输出上下文和查询输入预测未知系统的输出，省去了传统的模型选择和参数调优步骤。

**💡 创新点**

创新点包括：① 将元学习与概率推断结合，使用Transformer编码器-解码器生成预测均值和不确定性；② 引入循环分块和加权集成两种长序列处理策略，实现对长上下文的高效推理；③ 利用已训练元模型生成不确定性加权的合成数据，用于经典辨识模型的正则化；④ 在真实非线性辨识基准上展示零样本和少量适配的优异性能。

**🔧 技术方法**

技术手段主要有：Transformer 结构、循环分块（RNN patching）、加权集成、概率输出（高斯分布）、基于 KL 散度的损失、synthetic data generation、少量微调（few‑shot fine‑tuning）。

**📊 数据集**

数据集：合成 Wiener‑Hammerstein（WH）系统（多频率正弦输入），以及公开的非线性辨识基准数据（WH、Cascade Tanks、Coupled Electric Drives）。

**📈 对比分析**

通过与 dynoNet、LSTM、ARX 等经典方法在三大基准上的 RMSE 比较，元模型在 WH、Cascade Tanks 和 CED 上均实现了与最先进方法相当甚至更优的预测精度，并在 OOD 情况下只需适配输入即可将误差降至噪声水平，推理时间仅为秒级。

**⚠️ 局限性**

局限性包括：① 预训练需要大规模合成数据和长时间 GPU 训练；② 推理时模型参数量大，嵌入式部署受限；③ 对大量高质量任务数据时，传统辨识方法仍可获得更高精度；④ 输出为概率分布，缺乏可直接解释的结构化参数模型，导致对频率响应等分析不便。

---

## 468. Accelerating Linear Bandits with Persistent Candidate Sets

**arXiv ID:** 2610.13407 | [PDF](https://arxiv.org/pdf/2610.13407v1)

**作者:** Bingkui Tong `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Junpei Komiyama `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Persistent Candidate LinUCB，利用剪辑得分保留活跃候选并仅评估 k+q 个动作，实现大规模线性 bandit 的显著加速。

**💡 创新点**

创新点在于将持续候选集与剪辑 UCB 分数相结合，分析部分候选访问导致的发现成本，并给出上界与下界；证明该成本不可避免。

**🔧 技术方法**

采用线性 UCB、OFUL 置信集构造、剪辑 UCB 分数、随机抽样挑战者、超级马尔可夫分析等技术。

**📊 数据集**

在合成数据以及四个半合成基准（MovieLens‑20M、UCI SGEMM、NAS‑Bench‑101、DOCKSTRING）上进行评估。

**📈 对比分析**

与全局 LinUCB、RS‑OFUL、CBSCFD、PSI‑LinUCB、Sub‑Elim 等方法比较，PC‑LinUCB 在速度上提升 5–80 倍，累积伪回报仅略高或相近。

**⚠️ 局限性**

局限在于仅适用于固定动作集与固定特征，无法直接推广到上下文或动态特征环境。

---

## 469. Anechoic: Onsager-Corrected Parallel Annealing for Sub-0.1 ms Dense Max-Cut on an FPGA

**arXiv ID:** 2610.13368 | [PDF](https://arxiv.org/pdf/2610.13368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 470. ShoreKeeper: Accelerating Iterative Data Analytics in the Cloud via Distributed Recovery-Cost-Aware Caching

**arXiv ID:** 2610.13396 | [PDF](https://arxiv.org/pdf/2610.13396v1)

**作者:** Jongseop Lee `[一作]` (POSTECH), Myeongjae Jeon `[通讯]` (POSTECH)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一种面向云端迭代数据分析的分布式恢复成本感知缓存框架，利用统一的集群内存池并通过全局混合整数规划决定缓存与放置策略，以最小化被驱逐数据的恢复成本。

**💡 创新点**

创新点在于将缓存决策与数据放置统一到集群层面，采用图剪枝和分布式回收成本建模，兼顾预emption事件并在极短的时间内重新规划，显著提升了在可变容量云环境中的执行效率。

**🔧 技术方法**

核心技术包括全局混合整数线性规划（MILP）求解、恢复成本评估模型、图剪枝（缓存块边界剪枝与shuffle近似）、在线与离线性能分析、以及基于预emption告警的优先迁移与复制策略。

**📊 数据集**

实验使用了四个基准：PageRank与Connected Components基于规模为54 GB、2500万节点的合成图；Logistic Regression使用118 GB的Click-Log数据；SVD++使用约25 GB的合成评分数据。

**📈 对比分析**

与LRU、LRC、MRD、Blaze、Flint以及其预emption增强版本Blaze+D、Blaze+DF进行对比，实验表明在稳定的按需实例上可达2.05–6.11倍加速，在Spot实例预emption场景下相较于Blaze+D可获得1.14–2.87倍加速。

**⚠️ 局限性**

主要局限在于仅适用于迭代工作流、需要不可变分区与依赖图、对Spark（或同类框架）做了深度集成，且MILP求解与离线分析在短作业或高频节点变化时可能产生开销，未考虑非迭代或需要实时性极高的场景。

---

## 471. MoPrO-SD: Modular Prompt Optimization for Stance Detection

**arXiv ID:** 2610.13386 | [PDF](https://arxiv.org/pdf/2610.13386v1)

**作者:** Meysam Sabbaghan `[一作]` (Kansas State University), Cornelia Caragea `[通讯]` (University of Illinois Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MoPrO-SD，一种基于模块化的提示优化框架，用于改进大语言模型在立场检测任务上的表现，无需对模型参数进行微调。

**💡 创新点**

创新点在于：1）将提示拆分为六个语义功能模块（任务定义、准则、支持、反对、中立、输出格式），并按顺序逐模块优化；2）使用基于类条件错误的反馈（仅针对某一立场标签的误分类样本）进行定向优化；3）结合指令、推理和逐例推理三种优化策略，提升对隐式观点、讽刺等难点的处理能力。

**🔧 技术方法**

主要技术包括：大语言模型（Phi‑4‑14B、GPT‑OSS‑20B、Mistral‑Small‑3.2‑24B、Gemma‑3‑27B、Llama‑3.3‑70B）作为生成器；模块化提示设计与逐坐标下降式优化；类条件误差路由与推理生成的错误分析；实验中使用 GPT‑5 作为外部推理器进行对比。

**📊 数据集**

使用四个立场检测基准：SemEval‑2016 Task 6、COVID‑19‑Stance、GunStance 以及新构建的 VaccineStance（2,000 条与疫苗相关的社交媒体推文，手工标注）。

**📈 对比分析**

与基线提示（Base）、链式思考（CoT）、少量示例（FS）、Chain‑of‑Stance（CoS）、以及自动提示优化方法（OPRO、CAPO、EvoPrompt、UniPrompt、SCULPT）进行对比。MoPrO‑SD 在 18/20 的模型‑数据集组合中均优于 Base，I+R+E 在多数场景下得到最优或相近的 Macro‑F1，且在部分模型上（如 Phi、Gemma、Llama）与最先进的结构化提示优化方法相当或略优。

**⚠️ 局限性**

局限性包括：1）优化效果高度依赖模型与数据集，跨模型或跨任务迁移需重新优化；2）模块化结构专为三类立场设计，扩展到更大标签空间或更复杂推理任务尚未验证；3）优化过程计算成本高，尤其是推理型策略；4）实验依赖外部 API 版本，模型更新可能影响复现；5）仅在立场检测任务上验证，无法证明对其他 NLP 任务的通用性。

---

## 472. Who Can Say No? Control, Accountability and the Pace of Frontier AI

**arXiv ID:** 2610.13340 | [PDF](https://arxiv.org/pdf/2610.13340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 473. Stacked Intelligent Metasurface-Based MultiModal Feature Fusion for Collaborative Inference in Edge Networks

**arXiv ID:** 2610.13364 | [PDF](https://arxiv.org/pdf/2610.13364v1)

**作者:** Yushi Wang `[一作]` (Lancaster University), Shunqing Zhang `[通讯]` (Shanghai University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于堆叠智能金属表面（SIM）的协同推理框架（SIM-CI），将特征聚合与推理直接嵌入物理无线传播中，实现 OTA 计算与融合。

**💡 创新点**

核心创新在于：①三异构鲁棒适配器（数据、模型、系统异构统一）；②结合CSI的通道感知与双重鲁棒性机理（隐式感知+显式补偿）；③利用 SIM 物理层的线性变换完成端到端推理，显著降低能耗与时延。

**🔧 技术方法**

使用技术包括：图神经网络（GNN）+对比学习实现多模态特征对齐；混合基波形（模型驱动预均衡 + 数据驱动 Transformer）实现通道补偿；SIM 的多层自由空间衍射与可训练元原子构成的物理线性变换；OTA 聚合、OFDM 传输与能量检测。

**📊 数据集**

实验数据集：COCO2017（图像+文本）与 SpokenCOCO（语音+图像+文本）进行多标签分类评估。

**📈 对比分析**

与 BLIP‑2、ViT、RoBERTa、SIM‑TSC、Adapter‑BERT 等数字域基线对比，mAP 达 0.723（≈BLIP‑2 的 98.6%），计算能耗下降 25%，时延下降 76%，时隙占用减少 75%。

**⚠️ 局限性**

局限性：对极端通道失真或低 SNR 仍有限制；需要精准的通道估计与时序同步；SIM 物理实现依赖高精度元件，扩展到不同硬件平台存在挑战；实验仅验证在 OFDM 环境下。

---

## 474. Polarization Dynamics under Algorithmic Recommendations in LLM-based Online Social Networks

**arXiv ID:** 2610.13351 | [PDF](https://arxiv.org/pdf/2610.13351v1)

**作者:** Ali Safarpoor Dehkordi `[一作]` (Australian National University), Ahad N. Zehmakan `[通讯]` (Australian National University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于大型语言模型（LLM）的在线社交网络模拟中两种推荐机制——边缘推荐（ER）和内容曝光推荐（CR）——的影响进行受控实验，研究它们如何改变意见极化、极化程度与强化互动；

**💡 创新点**

首次将LLM代理与推荐算法结合，在同一实验框架内对两种推荐机制进行系统对比，展示LLM对语义、语调和人格特征的动态响应，弥补传统数值模型的局限；

**🔧 技术方法**

使用LLM（如DeepSeek、GPT-4o-mini、GPT-4.1-mini）进行文本生成与意见更新，构建基于自然语言的交互模型；

**📊 数据集**

使用合成网络（超球随机图HRG与二块块模型SBM）作为实验平台，并在不同讨论主题（远程工作、在线学习等）下评估；

**📈 对比分析**

通过配对差异检验（t检验、置换检验）与曝光量控制实验，比较推荐与基线的极化、极端化与强化指标，结果显示两种机制均显著提升极化（差值最高0.12），极端化与强化也随之增加；

**⚠️ 局限性**

限制在：推荐连线与内容均被假设接受/观看；单步交互，未考虑转发、回复等多步传播；代理无长期记忆或完整人格，仅依赖当前态度与固执度；

---

## 475. Adversarial Attacks Reveal Divergent Deception Competence in Humans and Generative Language Models

**arXiv ID:** 2610.13363 | [PDF](https://arxiv.org/pdf/2610.13363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 476. Test-Time Memory Construction

**arXiv ID:** 2610.13348 | [PDF](https://arxiv.org/pdf/2610.13348v1)

**作者:** Zijun Wang `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个在推理时动态构建记忆的框架，利用离线提取的轻量级元数据在推理过程中逐步激活并整合历史交互，从而构建查询特定的记忆。

**💡 创新点**

将记忆构建从离线预处理迁移到推理时的动态过程，使用关联记忆的方式逐步检索元数据和原始交互，使记忆能够根据推理过程实时适配。

**🔧 技术方法**

使用大型语言模型作为记忆提取器和推理器，结合元数据提取、工具集（元数据检索与原始交互检索）以及迭代推理和递归构建记忆的机制。

**📊 数据集**

在LoCoMo和LongMemEval这两个长期对话问答基准数据集上进行实验。

**📈 对比分析**

与检索增强生成（RAG）、多种离线记忆系统（MemoryOS、LightMem、HiMem、AnchorMem等）以及延迟式推理基线（LazyMem）进行对比，实验表明在两种大型语言模型上，T2Mem在准确率上持续领先，且在相同token预算下显著提升。

**⚠️ 局限性**

仍依赖LLM的推理能力，处理极长交互时可能出现记忆构建深度限制，且元数据提取的质量与所用模型和算法密切相关。

---

## 477. REINS: Agent-guided Thread Scheduling for Concurrency Bug Detection and Reproduction

**arXiv ID:** 2610.13421 | [PDF](https://arxiv.org/pdf/2610.13421v1)

**作者:** Yu Zhang `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了基于 LLM 代理的可控线程调度框架 Reins，用代理推理构造调度约束，指导并发程序的线程调度以检测和重现并发错误。

**💡 创新点**

创新点在于将代理的语义推理转化为可执行的调度计划；通过运行时观测与静态分析结合构建高质量的并发语义模型；采用三阶段（comprehend‑attempt‑assess）迭代流程，持续改进假设与计划；以及将计划约束映射到可控调度器，实现高效的目标调度。

**🔧 技术方法**

主要技术包括：Claude Sonnet LLM 代理、可控并发调度器、源代码与 AST 静态分析、运行时共享变量访问记录、约束计划（事件顺序与状态约束）以及动态验证与结果聚合。

**📊 数据集**

使用了公开基准 SCTBench、ConVul、RaceBench（共 127 目标，约 1,500 个注入 bug），以及 Mozilla NSPR 与 SpiderMonkey 两个实际项目的源代码。

**📈 对比分析**

在相同时间预算下与 SURW、RFF、PCT-3/10、RW 等现有可控调度技术对比，Reins 在 SCTBench/ConVul 中检测 48/52 个 bug（与基线相近或略优），在 RaceBench 中检测 769/1,500 个 bug，超过 SURW 的 731，且在实际项目中发现 6 个未知并发 bug（3 个已确认）。但由于代理推理成本，整体耗时比传统方法高 2–3 个数量级，token 使用量也显著增大。

**⚠️ 局限性**

主要限制包括：只能检测在给定测试用例覆盖的代码；仅适用于共享内存的 C/C++ 多线程程序；对需要特定状态值或极宽触发窗口的 bug 仍难以触发；时间/ token 成本较高；结果受 LLM 生成计划的质量和随机性影响。

---

## 478. Opening the Black Box of Classifier-Free Guidance via Information Bottleneck

**arXiv ID:** 2610.13369 | [PDF](https://arxiv.org/pdf/2610.13369v1)

**作者:** Jiayang Gao `[一作]` (Shanghai Jiao Tong University), Jia Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于信息瓶颈的动态无监督条件引导（CFG）激活策略，自动在生成过程的关键阶段开启条件信息注入；

**💡 创新点**

通过信息瓶颈理论与 Fisher 信息平衡推导出条件信息的有效激活时刻，设计了训练无关的阈值跨越判定规则，解释并统一了多种动态指导方法；

**🔧 技术方法**

信息瓶颈分析、Fisher 信息量度、动态阈值激活、批量统计采样、无监督条件预测网络；

**📊 数据集**

ImageNet（类条件图像生成）、MS‑COCO（文本到图像合成）、Stable Diffusion 3、Flux 1；

**📈 对比分析**

与固定 CFG、间隔指导等基线相比，在 DiT、SiT、SD3、Flux 等模型上在 FID、IS、HPS、CLIP、Aesthetic Score 等指标均取得显著提升，且在单样本推理和多种 batch 大小下表现稳定；

**⚠️ 局限性**

适用于随机扩散/流模型，对确定性耦合（如 n‑Rectified Flow）理论证明有限；阈值估计受批量统计噪声影响，需要一定 batch 大小；缺乏针对多模态或异构条件的通用化策略。

---

## 479. Hidden Data Challenges in Clinical Machine Learning

**arXiv ID:** 2610.13350 | [PDF](https://arxiv.org/pdf/2610.13350v1)

**作者:** Carolin Cissée `[一作]` (Hannover Medical School), Zahra Ahmadi `[通讯]` (Hannover Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估并量化了在MIMIC-IV数据预处理过程中常见的时序、语义和人群对齐失误，揭示了这些失误对模型学习和预测的隐蔽影响，并给出了可操作的预处理改进建议。

**💡 创新点**

创新点在于将时序泄漏、语义特征-目标不匹配和人群缺失/插值偏差三类失配问题统一框架化，提供了具体的指标和案例实验验证，并提出了针对MIMIC-IV及类似EHR数据的完整预处理检查清单。

**🔧 技术方法**

采用了多模态特征工程（时间序列、表格、影像嵌入）与XGBoost模型，结合统计与临床可解释的插值策略、缺失指示变量以及SHAP特征重要性分析，系统比较不同时间戳、插值方法及缺失处理对模型性能与特征依赖的影响。

**📊 数据集**

使用了MIMIC-IV 3.1版本及其扩展模块（MIMIC-IV-Note、MIMIC-CXR、MIMIC-IV-ECG、MIMIC-IV-ED），构建了ICU死亡预测和25种表型二分类预测两大实验数据集。

**📈 对比分析**

通过在固定随机种子下多次实验，对比了两种时间戳（charttime vs storetime）、两种插值策略（统计均值/众数 vs 临床常数值）以及缺失指示变量的组合，总体AUROC与AUPRC变化小于0.5%（死亡预测）或不超过5%（表型预测），但SHAP分析显示特征重要性和模型逻辑因预处理差异显著改变。

**⚠️ 局限性**

局限性包括：仅在单一医院的MIMIC-IV数据上验证，时间戳语义与工作流可能具有机构特异性；插值常数值为启发式且需针对其他人群调整；SHAP特征重要性不等同于临床因果；以及缺失导致的排除或插值可能引入选择偏差。

---

## 480. Why DiT Models Underperform as Representation Learners without Long Skip Connections

**arXiv ID:** 2610.13344 | [PDF](https://arxiv.org/pdf/2610.13344v1)

**作者:** Benyuan Meng `[一作]` (Chinese Academy of Sciences), Longtao Huang `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过从头训练对比实验，验证并说明长跳连接（LSC）能提升Diffusion Transformer（DiT）的特征质量与生成性能。

**💡 创新点**

提出并验证“扩散噪声绕过”机制，解释LSC在保持目标重建特征纯净度的同时，将扩散噪声直接传递到噪声预测层，从而提升特征表达和生成效果。

**🔧 技术方法**

使用长跳连接结构、互信息神经估计器（MINE）进行特征信息量探测、控制干预实验，评估特征质量和生成性能；基于流匹配（flow‑matching）训练DiT。

**📊 数据集**

训练数据：ImageNet 256×256；下游评测数据集包括SPair‑71k（语义对应）、Horse‑21、CityScapes（语义分割）以及NYU Depth v2（深度估计）。

**📈 对比分析**

与无LSC DiT‑XL/2和带LSC U‑ViT 对比，使用Fidelity（FID）与任务特定指标（PCK@0.1、mIoU、d1）衡量；结果显示LSC版本在所有六个基准任务上均显著优于无LSC版本，且即使生成性能相当（如100k步训练），特征表现仍更好。

**⚠️ 局限性**

局限性：难以在现有预训练DiT上增添LSC，实验仅覆盖传统DiT架构，对MMDiT等新型架构的影响未探究；仅对LSC机制给出部分解释，未覆盖所有可能导致特征衰退的原因。

---

## 481. Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis

**arXiv ID:** 2610.13510 | [PDF](https://arxiv.org/pdf/2610.13510v1)

**作者:** Hamza Shafiq `[一作]` (Eindhoven University of Technology), Aaqib Saeed `[通讯]` (Eindhoven University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了PRISM（Population‑Retrieved Inference for Structured Multimodal Prognosis）模型，用于整合临床中的多模态数据（结构化EHR、12导联心电图、心电图时间序列及检索到的相似病例表示）以实现多达1,443个二元预测任务的临床预后。

**💡 创新点**

核心创新在于：① 通过特征级数值软词化（FT‑Transformer）保留数值测量的比例关系；② 采用指数衰减时间加权聚合多次心电图历史，捕获临床变化信息；③ 在表示空间构建双模态检索记忆库，仅将平均聚合的邻居嵌入注入LLM，既提供人口级上下文，又避免直接泄露邻居记录。

**🔧 技术方法**

技术手段包括：特征级FT‑Transformer、D‑BETA心电图编码器、指数时间加权聚合模块、FAISS双模态检索记忆库、Soft‑Token投影、MedGemma‑4B语言模型的LoRA微调以及两阶段训练（编码器/投影器训练 + 检索 + LLM适配）。

**📊 数据集**

数据集：MIMIC‑IV（MDS‑ED基准）涵盖470项结构化特征、12导联心电图及时间序列；外部验证使用MC‑MED（斯坦福ED访视，单导联心电图）进行迁移学习与微调。

**📈 对比分析**

与15个基准（单模态ECG、文本化EHR、融合模型等）比较，PRISM在四大任务组（诊断、恶化、ICU入院、死亡）均取得最高AUROC：诊断86.24%、恶化93.12%、ICU 91.79%、死亡92.82%。在MDS‑ED上比UniPACT高约2–3个百分点，在外部MC‑MED上微调后提升约7.7个百分点，超过本地梯度提升模型。检索实验显示真邻居平均提升0.97 AUROC点，随机/零检索差距仅0.02点。

**⚠️ 局限性**

局限性包括：① 评估仅为回顾性、两机构范围，缺乏前瞻性多中心验证；② 只使用单一LLM骨干（MedGemma‑4B），难以泛化到其他模型；③ 检索记忆库可能导致隐私与解释风险（邻居隐私泄露、基于相似性的推理误导）；④ 对分层子组和分布漂移的校准与性能尚未系统评估；⑤ 模型参数量巨大（约40亿），推理成本高，部署成本与可扩展性待解决。

---

## 482. Hierarchical Graph Neural Network for Probabilistic Power Flow with Grid Topology Contingencies

**arXiv ID:** 2610.13495 | [PDF](https://arxiv.org/pdf/2610.13495v1)

**作者:** Jinbao Wang `[一作]` (Xi'an University of Technology), Sabita Maharjan `[通讯]` (University of Oslo)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种拓扑分解驱动的分层双向图神经网络，用于在电网拓扑变化下进行概率潮流预测。

**💡 创新点**

创新点在于将电网分解为核心骨干、动态均匀电压簇和基本拓扑单元，并结合先验敏感性线性模型与非线性残差的双向信息流，实现对未知拓扑的良好泛化。

**🔧 技术方法**

技术包括基于等效导纳的拓扑分解、谱聚类、Granger因果权重、电压敏感性线性先验、双向消息传递、阻抗加权注意力、模式调制广播等。

**📊 数据集**

使用IEEE 118节点系统和波兰2746节点系统的PV预测量化数据，以及对应的N–1/N–2切除事件。

**📈 对比分析**

与六种基线（DNN、GNN、PowerFlowNet、GNS、MD‑GCN、PG‑GNN）对比，TDD‑HBGNN在所有测试拓扑上相较基线的相位角误差低2.47–4.20倍，局部电压误差提升1.75–2.58倍，整体参数仅0.658M，前向推理0.16–1.8ms。

**⚠️ 局限性**

局限在于对负载和非PV发电的不确定性未建模，需进一步验证跨系统零样本泛化，且量化校准仍需改进。

---

## 483. CompileMask: Bridging the Gap Between Expressiveness and Efficiency in Attention Masking

**arXiv ID:** 2610.13474 | [PDF](https://arxiv.org/pdf/2610.13474v1)

**作者:** Hongzheng Li `[一作]` (Beijing University of Posts and Telecommunications), Ke Zhang `[通讯]` (JD.com)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套离线编译器，将用户通过区域 DSL 描述的可表达的注意力掩码规则转换为高效的运行时描述符，实现了自定义掩码的高性能执行。

**💡 创新点**

创新点在于将掩码的表达层与执行层分离：通过离线编译器自动推导所需的区间数量并预计算端点，生成包含区间表和位图的 Execution Descriptor，使得既能支持复杂掩码，又能在 Tile‑Attention 内核中高效执行。

**🔧 技术方法**

使用了符号编译器、区域 DSL、区间提取与规范化、无符号算术的可见性检查、TileLang 生成的预处理内核，以及 FlashAttention‑4 的张量核心实现。

**📊 数据集**

实验采用合成工作负载（HSTU、共享前缀奖励模型、CLEAR 多模态、因果掩码）以及真实用户交互日志 KuaiRand‑1K，并在 Qwen3‑8B/14B 上进行端到端训练。

**📈 对比分析**

与 TorchInductor 的稠密 softmax、FlexAttention（Triton/FA4）、FlashMask 以及原生 FA4 进行对比；在 32K 长度下，本文方案相对于 FlexAttention 提升高达 3.8×，相对于 Flex‑FA4 在 HSTU、RM、CLEAR 上提升 1.44–2.87×，并在因果掩码下保持 85–89% 的 FA4 性能。

**⚠️ 局限性**

局限性包括：若掩码规则无法被转化为有限区间，则退回到谓词评估，导致在线开销；对极其复杂或动态的规则仍需额外算子；目前仅在基于 Tile‑Attention 的环境中验证，其他稀疏格式或推理场景尚未探测。

---

## 484. Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits

**arXiv ID:** 2610.13438 | [PDF](https://arxiv.org/pdf/2610.13438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 485. REMBGO for Evaluation-Limited Optimization: CEC2022 Benchmarking and EAF-STATCOM Generalization

**arXiv ID:** 2610.13471 | [PDF](https://arxiv.org/pdf/2610.13471v1)

**作者:** Vikram Chekuri `[一作]` (BMS Institute of Technology and Management), Prashanth N A `[通讯]` (BMS Institute of Technology and Management)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一个评估预算感知的改进版多玩家战斗游戏优化器 REMBGO，用于在有限的函数评估预算下进行元启发式优化，并在电弧炉 STATCOM 控制器调优中验证其效果。

**💡 创新点**

创新点在于将种群保持、加权质心吸引、归一化衰减半径、对抗学习、Cauchy 变异、三领导混合、Lévy 飞行、对手战斗、抖动、反射修复以及适应性耐心与部分重启等多种机制整合到 EMBGO 框架中，从而在评估预算有限的场景下显著提升搜索性能；并通过组件剔除和预算依赖基准验证其改进主要集中在低评估预算。

**🔧 技术方法**

核心技术包括：多玩家战斗游戏优化框架、精英保留、加权质心、归一化衰减邻域半径、对抗学习、Cauchy 变异、三领导混合、Lévy 飞行、对手战斗、抖动噪声、反射修复、适应性耐心与部分重启等。

**📊 数据集**

使用的数据集包括：CEC2022 benchmark（12 个函数，维度 10 和 20）配合 14 个评估预算阶梯（其中 1040 评估为应用对齐预算），以及电弧炉 STATCOM 模型的 20 条未见随机弧流轨迹，用于固定增益转移验证，控制器包括 DQ‑PI 与 Lyapunov 两种。

**📈 对比分析**

采用预算依赖基准（Friedman、Wilcoxon、Holm 校正）对五种优化器进行比较。结果显示：在 1040 评估预算下，REMBGO 在 D=10 时获得最低平均排名，D=20 时略低于其父算法 EMBGO；在更大预算下 FP‑SHADE 领先；在未见弧流验证中，REMBGO 与 EMBGO 的差异无显著性，说明其固定增益转移鲁棒性与父算法相当。

**⚠️ 局限性**

局限性：基准实验仅涵盖 CEC2022 两维度、12 个函数和 14 个预算阶梯，未在更广泛的题库或更高维度验证；应用实验仅针对单一 EAF‑STATCOM、两种控制器、一次调优轨迹和 20 条未见弧流，未验证多轨迹或风险感知调优；部分实现细节（归一化半径、重启子群）在对称盒子上难以辨别影响，需进一步跨场景评估。

---

## 486. Reflect-VLA: Internalizing Reflective Guidance via Dual-Path On-Policy Self-Distillation for Autonomous Driving

**arXiv ID:** 2610.13453 | [PDF](https://arxiv.org/pdf/2610.13453v1)

**作者:** Yuechen Luo `[一作]` (Tsinghua University), Fuxi Wen `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将反射式指导（Reflective Guidance）内化到 Vision‑Language‑Action（VLA）模型中的训练框架 Reflect‑VLA，并通过双路径 on‑policy self‑distillation (Dual‑OPSD) 实现单通道、无指导的轨迹预测。

**💡 创新点**

创新点包括：①将轨迹评估转化为结构化诊断与纠正指令；②利用教师-学生双路径，对可靠轨迹做自我保持，对不安全轨迹通过教师的 token‑级分布进行纠正；③通过 Progressive Teacher Construction（驾驶知识预训练、规划‑改进 SFT、Self‑Refinement Rejection Sampling）降低分布差距，显著缩短 RL 训练时间。

**🔧 技术方法**

核心技术包括：VLA 端到端自回归轨迹生成、反射式指导构建、Dual‑OPSD（轨迹质量感知的分路径自蒸馏）、Self‑Refinement Rejection Sampling、以及基于 NAVSIM 评估器的 PDMS / EPDMS 等分数。

**📊 数据集**

使用多源驾驶 QA 数据集（LingoQA、DriveGPT4、Senna、DriveLM、NuScenes‑QA、NuInstruct、DriveLMM）进行预训练；使用 NAVSIM v1/v2、Bench2Drive 进行评估。

**📈 对比分析**

与 RL 基线（如 GRPO）及现有 VLA 方法比较，Reflect‑VLA 在 NAVSIM v1/v2 的 PDMS/EPDMS 上分别提升 0.5–0.8 分，Bench2Drive 的 Driving Score 提升至 80.15，且训练时间缩短超过 90%（OPSD 仅为 RL 训练时间的 1/10 左右）。

**⚠️ 局限性**

局限性：①仍需在训练阶段生成初始轨迹并评估以构造反射式指导，限制了对未知场景的即刻泛化；②对大规模模型的可扩展性尚未在更高分辨率或更复杂交互环境中验证；③在极端长尾场景下，教师对错误类型的覆盖可能不完整。

---

## 487. Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models

**arXiv ID:** 2610.13524 | [PDF](https://arxiv.org/pdf/2610.13524v1)

**作者:** Ran Tian `[一作]` (Google Deepmind), Courtney Paquette `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解释大规模语言模型预训练过程中出现的临时“尖峰”损失，提出基于重尾分布的二次随机特征模型模拟该现象，并开发轻量级层级模拟算法。

**💡 创新点**

首次证明重尾输入可导致训练尖峰，将随机特征模型与 DNN 层级动态对齐，提供可预测尖峰的 β₂ 调度方法，并通过轻量级模拟验证。

**🔧 技术方法**

使用 Power‑Law Random Feature（PLRF）模型、二阶泰勒展开、随机梯度噪声分析、梯度协方差与 Hessian 轨迹模拟算法，以及 AdamW 与 β₂ 参数研究。

**📊 数据集**

在 1.3B 参数 Transformer 解码器上使用 C4 数据集进行预训练，检验不同 β₂ 设置下的训练尖峰。

**📈 对比分析**

通过比较真实训练轨迹与模拟预测的验证损失波动，验证模拟能捕捉尖峰位置和幅度；改进的 β₂ 调度在同一检查点下降低尖峰并在最终指标上提升性能。

**⚠️ 局限性**

模型假设梯度噪声独立且仅二阶，忽略跨层相关和高阶效应；对重尾分布的假设限制了泛化，轻量级模拟难以捕捉完整 DNN 非线性和三阶项导致的稳定化机制。

---

## 488. Causal Effect Identifiability in Causal Abstractions: A Strict Hierarchy of Criteria

**arXiv ID:** 2610.13520 | [PDF](https://arxiv.org/pdf/2610.13520v1)

**作者:** Clément Yvernes `[一作]` (University Grenoble Alpes), Eric Gaussier `[通讯]` (University Grenoble Alpes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出对因果抽象中因果效应可识别性的统一框架，定义并比较多种可识别性概念，并给出严格层次结构。

**💡 创新点**

统一了IG、ICD、IGP、ICA、ICB、ICF等可识别性定义，证明它们之间的严格层次关系，并提供反例揭示差异。

**🔧 技术方法**

采用导数图（derivation graph）和步骤图（step graph）方法，结合do-推理、图形条件及最大图归约来分析可识别性。

**📊 数据集**

本研究为理论探讨，没有使用具体数据集。

**📈 对比分析**

通过构造反例和层次图说明不同可识别性概念的蕴含关系，理论上完成比较；未给出实验性能指标。

**⚠️ 局限性**

尚未解决IG与ICD在一般图集上的可判定性与复杂度，且对更复杂或无限图集的可识别性分析有限。

---

## 489. Humanize MLE Agents: Multi-Agent Session Alternation Beyond Single-Agent Limits

**arXiv ID:** 2610.13494 | [PDF](https://arxiv.org/pdf/2610.13494v1)

**作者:** Changye Li `[一作]` (Tsinghua University), Ligeng Zhu `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种在共享工作空间中交替运行完整LLM+工具链的工作流（HMA），通过在每个会话结束时重置上下文来维持长周期探索。

**💡 创新点**

创新点包括：① 在工作流层面进行代理交替而非单代理内的工具链工程，解决了单代理长周期递减和工具链陈旧性问题；② 使用上下文重置半MDP理论给出交替工作流的充分条件和实验预算上界；③ 在不增加任务特定知识或额外提示工程的前提下，提升了多模型组合的性能。

**🔧 技术方法**

核心技术包括：LLM多模型集成、半马尔可夫决策过程（SMDP）与选项框架、上下文重置策略、固定实验上限（k）与最终审查机制；实验实现基于GPT、Kimi、GLM、DeepSeek、Claude、Opus等模型的原生工具链。

**📊 数据集**

主要数据集为MLE-bench（75个Kaggle竞赛任务）以及一项48小时的核优化任务（AOPT）用于验证工作流的跨任务适用性。

**📈 对比分析**

与单代理、专用工作流基线进行对比；在MLE-bench中HMA实现了78.2%的任意奖牌率，超过最强单代理基线（72.4%）8个百分点，且在相同六小时预算下比公开工作流高7个百分点；跨六种不同模型配对平均提升6.4个百分点；在AOPT中获得878个循环，距离排行榜首位仅1.6%。

**⚠️ 局限性**

局限性：实验仅覆盖少数模型配对；固定k上限（k≤5）可能不适用于所有任务；依赖共享工作空间和结果保留假设；在更大范围的任务与模型中可推广性尚未完全验证。

---

## 490. Proactive Agentic Orchestration

**arXiv ID:** 2610.13485 | [PDF](https://arxiv.org/pdf/2610.13485v1)

**作者:** Xueyang Feng `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种主动代理编排框架，允许子代理在执行过程中主动与编排器通信，实时调整其执行环境，并提出基于反馈的自监督训练方法。

**💡 创新点**

创新点在于将子代理的主动请求与编排器实时互动结合，形成执行-编排的无缝耦合，并通过OPOD将子代理反馈转化为过程级监督，显著提升动态长时任务的成功率。

**🔧 技术方法**

使用大语言模型（Gemini‑3‑Flash、DeepSeek‑V3.2/V4‑Flash、Qwen3.5‑9B）结合自定义通信协议、强化学习（RWR）和过程级监督（OPD）实现编排器学习与子代理交互。

**📊 数据集**

在GAIA、GAIA2和SWE‑Bench三大长时任务基准上进行评估，涵盖科研、数字助手与软件工程等场景。

**📈 对比分析**

与单体代理、离线编排和重编排基线对比，PORCH+OPOD在三大基准上平均提升任务成功率约9–10%，在动态任务中相对最高基线提升超过24%，并在成本‑性能曲线上处于Pareto前沿。

**⚠️ 局限性**

局限性包括需要为子代理设计结构化通信协议，且在极端动态环境下仍可能出现同步延迟；实验仅覆盖三大基准，尚未验证在更广泛任务域的通用性。

---

## 491. A Multi-Dimensional Score for Detecting Lesser-Known Points of Interest in Mountain Regions

**arXiv ID:** 2610.13467 | [PDF](https://arxiv.org/pdf/2610.13467v1)

**作者:** Adriana Pompejano `[一作]`, Philippe Roose `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了利用开放数据构建多维度“Obscurity Score”来识别山地少知旅游景点，并实现了网页原型。

**💡 创新点**

创新点在于将国际认知、吸引力、推广和可达性四维度整合为可加权连续得分，并提供可调节权重的参数化模型，同时构建了全流程开放数据管道与可视化决策工具。

**🔧 技术方法**

技术包括开放数据多源融合（Wikidata、Commons、OSM、Pageviews、OSRM）、指标标准化与加权求和、Jenks自然断点分类、网页可视化展示。

**📊 数据集**

数据集：OpenStreetMap POI与标签、Wikidata语言与页面浏览量、Wikimedia Commons图片、OSM路线标记、OSM编辑记录、OSRM路由时间；应用于法国-西班牙比利牛斯地区。

**📈 对比分析**

方法与评价：通过与当地专业人士（导游、山舍管理员等）对4-5级评分进行对照，采用网格搜索调参后评估F1和Kappa；目前仅在Pyrenees案例得到初步结果，尚未完成正式验证。

**⚠️ 局限性**

局限性包括：低文档可能源于地图贡献稀少而非真正隐蔽；指标阈值为默认值，需专家校准；评分随时间动态变化，公开列表可能诱发新过度旅游；缺乏与LLM评估的对比。

---

## 492. Agentic Large Language Models

**arXiv ID:** 2610.13455 | [PDF](https://arxiv.org/pdf/2610.13455v1)

**作者:** Max van Duijn `[一作]`, Niki van Stein `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 493. Oído: Open-Vocabulary Speech Recognition on a $5 Microcontroller without a Neural Accelerator

**arXiv ID:** 2610.13447 | [PDF](https://arxiv.org/pdf/2610.13447v1)

**作者:** Daniel Varela Sánchez `[一作]` (Lokutor (Hashing Works)), Guillermo Martín-Coello Juárez `[通讯]` (Lokutor (Hashing Works))

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在不含神经加速器的$5价ESP32‑S3微控制器上实现了完整词表英语语音识别的Oído推理引擎，支持int8量化的Conformer‑CTC与Transducer模型，并提供流式和西班牙语版本。

**💡 创新点**

创新点包括无训练的int8后训练量化配合逐帧激活缩放、面向内存带宽的执行调度、整机内的语言模型与Transducer解码、块级注意力的流式推理以及int4量化可装载在闪存中的模型。

**🔧 技术方法**

采用整数仅推理、SIMD GEMM、卷积、Self‑Attention、动态缩放、GRU前缀束搜索、Transducer联合网络、带宽感知调度、缓存友好内存布局以及对int4的量化感知训练等技术。

**📊 数据集**

评估使用LibriSpeech测试集以及未参与训练的Common Voice、VoxPopuli、AMI、People’s Speech、Earnings‑22，并在噪声与混响下加入DEMAND、MUSAN；西班牙语模型则在Common Voice、VoxPopuli、MLS、FLEURS上训练。

**📈 对比分析**

与Espressif MultiNet7、Whisper tiny.en、Moonshine tiny以及Vosk small等基线比较，Oído的int8 CTC在LibriSpeech上达到3.70%/8.23% WER，Transducer 2.99%/6.67%，在硬件上约2×实时、比供应商低约2.3–2.8倍错误率；int4量化后通过QAT可降至4.61%/9.98%，西班牙语版本在相同数据集上取得约13–15% WER。

**⚠️ 局限性**

主要限制包括尚未实现实时推理（单核约2×实时，双核模式在硅上错误）、流式推理牺牲约1.5点准确度、仅支持英西两语、int4版本需额外训练、对外部内存带宽要求高、对长语音的性能与稳定性尚未充分验证。

---

## 494. Knowing Whether, Not Just Where: Anchor-Probing Distillation for Open-Set Remote Sensing Visual Grounding

**arXiv ID:** 2610.13535 | [PDF](https://arxiv.org/pdf/2610.13535v1)

**作者:** Zhihao Chang `[一作]` (Xidian University), Zhixi Feng `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AnchorProbe-Distill (APD) 框架，用于遥感视觉定位的开放集场景，既能准确定位存在目标，又能在目标缺失时给出拒绝响应。

**💡 创新点**

创新点在于：①将开放集定位转化为视觉支持学习，通过多次教师预测的空间锚点稳定性来判定是否有充分视觉证据；②对四类查询（原始、语义保持、近类别歧义、远类别反事实）进行参考锚定的类型感知判定；③在线教师‑学生蒸馏，将多次推理与稳定性判定压缩到单次推理的学生模型。

**🔧 技术方法**

采用的技术包括：教师‑学生蒸馏；多次随机推理与CIoU一致性图；内部稳定性与共识比例统计；参考锚定校准阈值与类型感知监督推理；以及标准的自回归指令微调目标。

**📊 数据集**

使用的数据集为 RefGeo 以及其对应的远类别反事实集，覆盖 AVVG、DIOR‑RSVG、GeoChat、RRSISD、RSVG、VRSBench 等多个遥感数据集，共计 37,763 张样本。

**📈 对比分析**

与 Qwen3‑VL‑2B、LLaVA‑7B、GeoGround、Geo‑R1‑3B、RemoteReasoner、GeoPixel、Text4Seg 等多种模型对比，APD‑S 在 Acc@0.5 取得 66.30%（提升约 4%），在拒绝率 RR 取得 97.81%（几乎达到 100%），显著优于现有方法。

**⚠️ 局限性**

局限性包括：需要教师模型进行多次推理，推理成本较高；阈值（η、τ_cluster 等）需针对每个 VLM 手工调参；目前仅针对单目标或缺失目标场景，复杂多目标或多语义模糊情况尚未充分覆盖；对教师模型质量高度依赖。

---

## 495. Teams without teams: a trace-based method for measuring team-like organization

**arXiv ID:** 2610.13529 | [PDF](https://arxiv.org/pdf/2610.13529v1)

**作者:** Thomas Baillie-Dawe `[一作]` (IMT Atlantique), Nicolas Jullien `[通讯]` (IMT Atlantique)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于数字足迹的 trace‑based 方法，用以衡量开放协作中的团队组织程度。

**💡 创新点**

创新点在于将团队维度转化为可从行为轨迹直接恢复的指标，并允许在未预设团队成员的情境下量化团队性。

**🔧 技术方法**

主要技术包括行为指示器（协调、相互依赖、角色差异化、边界）与加权组合的团队性得分，使用时间窗口、父子修订关系等算法实现。

**📊 数据集**

使用了英語維基百科医学 WikiProject 的 52,437 篇文章的 10,426,462 条人工编辑修订历史数据。

**📈 对比分析**

通过对不同编辑活跃度水平的指标分布及组合得分进行描述性比较，结果显示指标对活跃度的敏感度各异，团队性得分在同一文章随时间波动，说明方法可捕捉动态团队性。

**⚠️ 局限性**

局限包括对编辑类型的粗略分类、对小规模协作样本时指标的机械性偏高，以及缺乏与传统团队调查或结果质量的验证。

---

## 496. Gaussian BROJA Partial Information Decomposition in Closed Form

**arXiv ID:** 2610.13464 | [PDF](https://arxiv.org/pdf/2610.13464v1)

**作者:** Ohad Sheelo `[一作]` (Ben Gurion University of Negev), Ari Pakman `[通讯]` (Ben Gurion University of Negev)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文推导了高维多变量高斯分布下BROJA PID的闭式解，提出Eigen-PID方法；

**💡 创新点**

创新点在于给出统一的闭式公式，消除了迭代优化，速度提升两位数以上；

**🔧 技术方法**

主要技术包括高斯实验表示、一般化特征值分解、Hadamard行列式不等式与数据处理不等式；

**📊 数据集**

使用合成高斯数据及真实神经影像数据（fMRI）进行验证；

**📈 对比分析**

与Thin-PID、Tilde-PID等数值优化方法对比，PID结果相同，运行时间提升至3~1000倍；

**⚠️ 局限性**

局限在于需正定协方差、对非高斯数据需要先归一化或使用流模型，且存在有限样本偏差。

---

## 497. MOSAIC: Multi-Scale Object-Centric Slot Adaptation with Image-Guided Prototype Conditioning for Remote Sensing Image Super-Resolution

**arXiv ID:** 2610.13547 | [PDF](https://arxiv.org/pdf/2610.13547v1)

**作者:** Liuqian Wang `[一作]` (Zhengzhou University), Lei Shi `[通讯]` (Zhengzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了MOSAIC框架，将预训练的Diffusion Transformer（DiT）适配到遥感图像超分辨率任务，通过对象中心化的多尺度槽提取来重构结构化条件，进而实现对高分辨率图像的精细恢复。

**💡 创新点**

创新点在于：①使用对象中心化的槽学习（OCL）将稠密低分辨率特征压缩为有限数量的结构槽；②跨尺度槽融合（MSF）实现粗糙布局与细粒度纹理的互补；③利用共享原型记忆检索相似结构作为残差上下文；④将整合后的槽通过SlotFiLM残差调制注入冻结的DiT，保持预训练生成先验。

**🔧 技术方法**

所用技术包括Diffusion Transformer（DiT4SR）、Slot Attention、跨尺度融合、原型记忆检索（Prototype Cross‑Attention）、轻量化残差FiLM调制、VAE编码/解码以及多步采样。

**📊 数据集**

实验数据集为UCMerced、AID和RSSCN7，分别在×2和×4放大倍数下进行评估。

**📈 对比分析**

与Real‑ESRGAN、KDSR‑GANV2、LDL、MM‑RealSR和DiT4SR等基线方法相比，MOSAIC在×2场景下平均提升PSNR约2.9–5.4 dB，在×4场景下提升约1.5–4.5 dB；LPIPS、MANIQA和CLIP‑IQA等感知指标亦优于对手，显示出更好的图像质量与结构保留。

**⚠️ 局限性**

主要局限在于原型记忆是离线构建且在训练分布内固定，可能难以适应未见的传感器或降质；同时槽数、层注入位置等超参数需手工调优，模型可迁移性与自动化程度有限。

---

## 498. MAGMA-Bench: Evaluating Long-Horizon Embodied Agents under Dynamic Tasks and Execution Failures

**arXiv ID:** 2610.13534 | [PDF](https://arxiv.org/pdf/2610.13534v1)

**作者:** Loan Bernat `[一作]` (Siléane), Florent Lamiraux `[通讯]` (LAAS-CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MAGMA-Bench，用于评估在多机器人环境中长期任务推理的基准；

**💡 创新点**

首次将长周期任务推理、在线扰动（失败、任务更新、打断）与异构机器人协调统一到一个架构无关的、技能级接口的评测框架；

**🔧 技术方法**

使用结构化技能接口、事件驱动决策、无监督或监督下的语言模型（Qwen3-4B、Gemini、gpt-oss-20b等）以及状态管理器、离线专家轨迹生成；

**📊 数据集**

发布261,646条任务级训练样本，261,646训练例子与13,574验证例子；以及12个场景家族（共1,254条测试episode）；

**📈 对比分析**

通过成功率(SR)、目标完成度(GC)、条件鲁棒性(CR)等指标与前沿模型（GPT、Gemini）对比，发现零样本模型成功率低；经过微调后小模型在短任务中接近前沿，但在长任务、扰动和多机器人场景中仍表现较差；

**⚠️ 局限性**

仅评估任务级决策，未覆盖感知与低级控制；使用单一语义实现的子集可能低估真实多样性；不兼顾去中心化或符号规划器的架构性能；

---

## 499. SplatQ: Interaction-Aware Pruning of 3D Gaussian Splatting via Quadratic Selection

**arXiv ID:** 2610.13503 | [PDF](https://arxiv.org/pdf/2610.13503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 500. S-Space: Exploring Spatial Workspace in Multimodal Language Models

**arXiv ID:** 2610.13527 | [PDF](https://arxiv.org/pdf/2610.13527v1)

**作者:** Zhiyuan Shi `[一作]`, Jialong Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过梯度方法发现并验证了多模态大型语言模型内部的线性空间 S‑Space，并展示其可直接读取物体位置坐标，能够作为空间工作区来支持空间推理和诊断。

**💡 创新点**

创新点在于首次揭示模型内部的可读空间子空间 S‑Space，提出仅需梯度求解即可得到固定的水平、垂直和距离轴，并证明该空间在不同任务、提示和模型层级中保持一致；此外将 S‑Space 用作诊断工具，拆解空间感知与空间计算两大瓶颈，分析视角转换与方向耦合错误。

**🔧 技术方法**

主要技术包括：基于相对空间问答梯度的方向提取、线性投影读取坐标、因果干预编辑坐标、链路思考跟踪空间变换、以及对比实验与可视化分析。

**📊 数据集**

使用的数据集包括 COCO（构造相对空间问题）、CV‑Bench、EmbSpatial、SpatialTunnel、SpinBench、MMSI‑Bench 等；实验涉及 Molmo2‑ER、Qwen3.5‑4B、Qwen3.6‑27B 三大多模态模型。

**📈 对比分析**

方法对比：直接读取 S‑Space 坐标回答空间问题，与模型生成答案对比，准确率相当；在视角变换任务中，显式坐标旋转后准确率显著提升，表明模型内部表征足够但推理转换不足。

**⚠️ 局限性**

局限性：S‑Space 虽能精准编码空间感知，却在空间计算（尤其是视角变换）和参考帧耦合上表现不佳，导致模型在某些空间推理任务中仍失误；模型仍依赖于大规模语言训练的偏好，可能把图像相对方向误用于方位词。

---

## 501. Beyond Successful Robot Demonstrations: Learning from Human Interactions and Modeling Failure Dynamics

**arXiv ID:** 2610.13522 | [PDF](https://arxiv.org/pdf/2610.13522v1)

**作者:** Peiyan Tu `[一作]` (Zhejiang University), Zhibo Chen `[通讯]` (University Of Science And Technology Of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了HF-WM模型，融合人类视角交互数据和无监督失败场景学习，使用双分支结构和GRPO优化提升动作条件世界模型的表现。

**💡 创新点**

创新点包括：① 通过共享骨架动作空间和稳定点流监督实现人机跨模态迁移；② 利用点流正则化抑制视角噪声；③ 用GRPO在反事实动作上进行无监督失败动力学建模，并结合动作对齐与物理合理性奖励。

**🔧 技术方法**

技术方法包括Cosmos‑3 Mixture‑of‑Transformers基础模型、VAE+Diffusion视频生成、点流监督、Flow‑GRPO强化学习、动作对齐评分器以及基于VLM的物理合理性检查器。

**📊 数据集**

使用的数据集包括机器人交互数据集（RoboH20T、InternData‑A1、DROID）和人类交互数据集（EgoDex、EPIC‑Kitchens），共约45k条机器人轨迹和83k条人类轨迹。

**📈 对比分析**

与TesserAct、Cosmos‑Predict2.5、Ctrl‑World、Cosmos‑3、OSCAR、Kinema4D等六个基线进行PSNR/SSIM/LPIPS/FVD/FID/Latent‑L2/IDM等多维度评估，HF‑WM在所有指标上均实现或超过最优，尤其在失败场景下保持稳健，并且其成功率预测与真实部署高度相关（Pearson r=0.983）。

**⚠️ 局限性**

局限性在于仍需大量标注数据，面对极端摄像机运动或极少失败样本的泛化能力有限；GRPO训练成本高且对奖励设计与阈值敏感。

---

## 502. A Hybrid and Hierarchical Ray Tracing Method for Efficient 3D Gaussian Rendering

**arXiv ID:** 2610.13498 | [PDF](https://arxiv.org/pdf/2610.13498v1)

**作者:** Tiancheng Zhao `[一作]` (Hong Kong University of Science and Technology), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一种混合层次光线追踪架构，用于高效的3D高斯剖分渲染

**💡 创新点**

提出了LB‑SBVH、RCPR和UIE三项技术，分别解决空间不平衡、相邻光线冗余和混合原始物体问题

**🔧 技术方法**

采用密度感知空间BVH划分、光线集路径重用与统一交点引擎，并在GPU中集成R2RFP、SPM等硬件支持

**📊 数据集**

在Mip‑NeRF 360与Deep Blending两个数据集上进行评估

**📈 对比分析**

与RaySplats基线比较，几何平均提升3.7×，单帧延迟从约70 ms降至约18 ms，PSNR差异≤0.4 dB

**⚠️ 局限性**

仍受限于对高分辨率光线集的并行度不足、近似路径重用导致的极端场景误差和硬件面积占用

---

## 503. A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance

**arXiv ID:** 2610.13479 | [PDF](https://arxiv.org/pdf/2610.13479v1)

**作者:** Victoria J. Hodge `[一作]` (University of York), Rob Alexander `[通讯]` (University of York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了基于本地大型语言模型的闭环 Agentic AI 框架，自动生成 JSON 结构化边缘案例场景，并通过 Unity‑3D 仿真进行评估，从而为自动驾驶等安全关键系统提供可追溯的安全保障证据。

**💡 创新点**

创新点在于整合两类 LM（生成器与批判者）形成迭代闭环，利用上下文知识库和 JSON schema 实时纠错；实现完全本地化、低成本运行；并将仿真评价反馈用于动态探索/利用，直接满足 ISO 21448/SOTIF 与 PAS 8800 的安全案例需求。

**🔧 技术方法**

采用的技术包括 Ollama 本地 LLM（qwen3‑coder:30b 与 nemotron3:33b）、JSON 结构化场景描述、Pydantic 校验、思考‑行动‑观察‑更新循环、Unity‑3D ML‑Agents 仿真、PPO 强化学习与增量课程学习，以及 GSN 安全案例框架。

**📊 数据集**

研究未使用公开数据集，而是通过随机生成、LM 生成的场景与 Unity‑3D 仿真产生的仿真日志与评估指标作为自建数据；RL 训练也在仿真环境中生成经验。

**📈 对比分析**

与两种随机基线（Random_1、Random_5）相比，Agentic 在关键场景发现率更高，碰撞次数、失败率明显提升，能揭示更多功能缺口，尽管整体奖励略低，但表现出更高的边缘案例覆盖度和危害发现效率。

**⚠️ 局限性**

限制包括小模型的上下文窗口导致错误与幻觉；需人工校验与 Pydantic 处理；仿真与现实之间存在转移鸿沟；仅支持静态障碍物；缺乏统一的覆盖度度量；以及在探索/利用决策中的参数调优和潜在过拟合风险。

---

## 504. OC-AND: A Citation-Enriched Benchmark for Author Name Disambiguation

**arXiv ID:** 2610.13462 | [PDF](https://arxiv.org/pdf/2610.13462v1)

**作者:** Francesca Cappelli `[一作]` (University of Bologna), Silvio Peroni `[通讯]` (University of Bologna)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了OC-AND开放式、引用丰富的作者姓名消歧义基准，并通过5步Bridge4AND流水线将WhoIsWho数据映射到OpenCitations元数据；

**💡 创新点**

首次提供开放稀疏元数据环境的AND基准，完整保留GroundTruth，系统化的DOI验证、OpenCitations丰富与作者中心转换流程；

**🔧 技术方法**

使用Crossref API做DOI匹配与阈值校准、OpenCitations Meta/Index API提取文献与引用、Python实现的Bridge4AND流水线，以及BOND模型做基线验证；

**📊 数据集**

基于WhoIsWho na‑v3 benchmark、OpenCitations数据库、Crossref及OpenAlex等开放数据源；

**📈 对比分析**

以BOND模型在原始与OC-AND上对比，评估指标包括pairwise、B3、LE、SE等，结果显示OC-AND下性能显著下降，尤其分裂错误上升，验证了基准的挑战性；

**⚠️ 局限性**

受限于缺少摘要与机构信息、文本特征稀疏、DOI获取不完整以及OpenCitations引用覆盖不全，导致分裂错误增加，且对基准适用性与完整性有一定影响。

---

## 505. A Real-World CAD Segmentation Benchmark: Comparing Rule-Based and Learning-Based Approaches through Hole Detection

**arXiv ID:** 2610.13466 | [PDF](https://arxiv.org/pdf/2610.13466v1)

**作者:** Kai Schütte `[一作]` (Volkswagen AG), Matthias Rottmann `[通讯]` (Universität Osnabrück)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两种基于规则的螺纹孔检测方法，并制作人工标注的工程CAD孔洞数据集，随后与UV‑Net和AAGNet学习方法进行对比实验。

**💡 创新点**

创新点在于：①手工标注的真实工程孔洞基准数据集；②两种面向任务的规则算法，分别强调召回和精度；③对规则与学习方法在跨域、数据量、IoU阈值等维度的系统评估。

**🔧 技术方法**

技术手段包括基于拓扑的Graph‑Wire‑Search、基于多视图渲染的MVR、UV‑Net（B‑Rep UV 参数化+CNN+消息传递）和AAGNet（带属性的图网络）等。

**📊 数据集**

使用的数据集为从3D Assembly Repository抽取的手工标注孔洞集（≈200个零件）以及MFCAD++的200个零件用于跨域测试。

**📈 对比分析**

实验表明规则方法在不需要训练的情况下跨域鲁棒；GWS偏向召回、MVR偏向精度；在同域训练/测试下UV‑Net和AAGNet性能最高，但在MFCAD++上泛化差，尤其是AAGNet几乎失效。

**⚠️ 局限性**

局限性包括：规则方法对复杂孔形缺乏适应性；学习方法对标注量和数据分布高度依赖，跨域泛化弱；基准仅覆盖螺纹相关孔，未涵盖所有工程几何。

---

## 506. HoldLight: Making Indoor Climbing More Accessible for Blind and Low-Vision Climbers through Layered Route Preview and Real-Time Next-Hold Guidance

**arXiv ID:** 2610.13445 | [PDF](https://arxiv.org/pdf/2610.13445v1)

**作者:** Chenyu Du `[一作]` (Xi'an Jiaotong-Liverpool University), Yue Li `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一套名为 HoldLight 的系统，为盲/低视力攀岩者提供分层的路线预览、简洁的实时抓点提示以及可见的回退机制，支持在不失去安全监督的前提下提升路线认知与行动连续性。

**💡 创新点**

创新点在于将路线支持拆分为预览、实时提示与回退三个可控阶段，并通过“分层预览”和“可见回退”两大设计目标实现了对盲/低视力攀岩者认知负荷与自主性的平衡；同时首次将 AI 辅助与人类安全协作明确嵌入一个可操作的工作流。

**🔧 技术方法**

核心技术包括：基于手机摄像头的壁面扫描与姿态跟踪（MediaPipe Pose）、服务器端的壁面对齐与同色路径规划、规则化的路线抽象与语义化提示生成，以及单耳/骨传导的低延迟音频提示。

**📊 数据集**

使用了自行构建的六条 5.8–5.10c 级别顶绳路线作为评测数据集；预扫描数据用于路径提取，现场录像与系统日志用于评估路线理解、回退事件与支持时长。

**📈 对比分析**

通过对比控制型人工“按需提示”与 HoldLight 两种边界化工作流的 12 名受试者在 36 条正式路线上的表现，结果显示 HoldLight 在路线理解得分、误差恢复次数、人工请求次数以及主观自主感上均显著优于基线（p<0.01），但在总耗时上无显著差异。

**⚠️ 局限性**

局限性包括样本量小、实验在受限室内环境下进行，缺乏长期使用和不同技能水平的验证；技术层面仍受同色集群、路线相似度高时的定位错误和服务器延迟影响；且回退机制仅覆盖信息层面，未能完全替代高风险情境下的人类安全判断。

---

## 507. RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors

**arXiv ID:** 2610.13516 | [PDF](https://arxiv.org/pdf/2610.13516v1)

**作者:** David Ahmedt-Aristizabal `[一作]` (CSIRO), Lars Petersson `[通讯]` (CSIRO)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为RESCUE的框架，用于在机器人策略运行时同时评估监控器的报警与对应干预的价值；

**💡 创新点**

创新点在于将监控器评价从单纯的预测误差扩展到“状态–干预”对的可恢复性与决策成本，提出了可恢复性边界、决策价值计算以及基于日志的失误成本平衡公式；

**🔧 技术方法**

使用的技术包括：恢复模拟器/策略状态并进行反事实分支（counterfactual branching）、基于贝叶斯后验的成功概率估计、停止-重置决策成本模型、以及闭式破平衡阈值计算；

**📊 数据集**

实验数据集为LIBERO与LIBERO‑Plus（含多种扰动）上的1,562条VLA任务轨迹（包含262次失败），使用OpenVLA‑OFT与π_0.5两种策略，并生成128,340条反事实分支；

**📈 对比分析**

与传统的检测/时序指标（AUROC、AUCPDT、及时率）相比，RESCUE通过决策价值评估表明：在强策略条件下，绝大多数监控器即使在检测效果良好，也在成本上不优于不干预；在高失败率或使用脚本恢复时，可观察到少数监控器在成本上略优；

**⚠️ 局限性**

局限性包括：仅在MuJoCo仿真中验证，所有失败均为超时；干预成本和分支扰动是人工设定；只评估了两种简单干预（继续和脚本恢复）；对真实硬件、接触动力学、延迟等因素缺乏验证，且模型假设可能不适用于不同任务或策略。

---

## 508. FedPCD: Priority-Constrained Federated Aggregation

**arXiv ID:** 2610.13508 | [PDF](https://arxiv.org/pdf/2610.13508v1)

**作者:** Yousef Irshaid `[一作]` (American University of Sharjah), Mohamed I. AlHajri `[通讯]` (American University of Sharjah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为FedPCD的优先约束联邦学习框架，保持FedAvg聚合为主导方向，仅在必要时做最小修正以满足每个客户端的性能约束；

**💡 创新点**

创新点在于将全局聚合视为主目标，客户端进展视为次要约束，采用优先约束下降（PCD）方法实现最小偏差修正，并通过历史缓存解决部分参与时的代表性问题；

**🔧 技术方法**

使用的技术包括FedAvg聚合、优先约束下降（PCD）优化、线性约束投影、历史缓存机制、以及对不同数据不平衡的Dirichlet分割；

**📊 数据集**

实验数据集为CIFAR‑10、CIFAR‑100和Fashion‑MNIST，客户端数设为100，参与率为5%和10%，数据异质性采用Dirichlet α=0.01、0.1、1；

**📈 对比分析**

与10种公平感知FL基线（如AFL、q‑FFL、FedMGDA+、CRAFT等）对比，FedPCD在所有18种配置下均取得最高全局准确率，17种配置下均优于其他方法的worst‑10%准确率，并且在大多数情形下收敛到相同或更好性能所需的通信轮数仅为10%–44%；

**⚠️ 局限性**

局限性包括：τ参数为固定值，未针对不同客户端或不同轮次自适应；历史缓存需要服务器获取单个客户端伪梯度，难以在加密聚合下实现；仅在单轮本地训练、100客户端设置下验证，未评估多轮本地训练或更大规模场景下的表现。

---

## 509. Lossless Inference Acceleration for ASR via EAGLE-Style Speculative Decoding

**arXiv ID:** 2610.13506 | [PDF](https://arxiv.org/pdf/2610.13506v1)

**作者:** Sen Zhang `[一作]` (Tianjin University), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了EAGLE-ASR框架，通过EAGLE式推测式解码实现自动语音识别的无损推理加速。

**💡 创新点**

创新点在于设计轻量级Speech‑Draft模块，利用目标ASR模型的编码器声学特征和分层解码器表示进行草稿生成，并通过自蒸馏数据和多步生成训练提升草稿质量。

**🔧 技术方法**

使用了EAGLE推测式解码、Transformer/Conformer自回归解码、交叉注意力、加权交叉熵衰减训练以及自蒸馏数据生成等技术。

**📊 数据集**

在LibriSpeech（960h）和VoxPopuli（多语种、声学）数据集上进行训练与评估，同时对多语言ASR和语音翻译任务进行验证。

**📈 对比分析**

与传统Speculative Decoding、Distil‑Whisper、Whisper‑tiny、Medusa等方法对比，EAGLE-ASR在Whisper-large-v3上获得1.91×/1.79×速度提升、Conformer AED上达到2.43×/2.30×，且ΔWER/ΔBLEU保持0.00，体现出更高的接受率与更优的加速效果。

**⚠️ 局限性**

局限性包括对多步生成的误差累积仍存在、需对不同模型进行精细调参、在更复杂的跨语言或对话任务中性能提升可能受限，且对硬件的GPU显存需求仍较高。

---

## 510. Phonology-Aware Speaker Extraction Frontend for Audio LLMs

**arXiv ID:** 2610.13500 | [PDF](https://arxiv.org/pdf/2610.13500v1)

**作者:** Yanxiang Chen `[一作]`, Weifeng Su `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种轻量级、可自适应的目标说话人提取前端 LATTE，能够根据混合语音的难度动态决定计算深度，并支持与冻结的音频大型语言模型（Audio‑LLM）无缝集成。

**💡 创新点**

创新点在于：① 通过难度估计器（Difficulty Estimator）对每个输入混合进行难度预测；② 引入任务自适应的语音学难度指数（PDI）作为弱监督信号，利用跨语言的语音学相似图平滑语言层面的计算需求；③ 采用共享的多退出Hourglass结构实现语言无关的自适应路由，既保持高质量提取，又显著减少计算量。

**🔧 技术方法**

技术实现包括：WavLM Base+编码器+投影层获取语义特征，512点STFT获取频谱特征；FiLM 条件注入，TCN+注意力组合构建Hourglass块；难度估计器结合混合均值/方差与说话人嵌入预测预算；多退出结构与线性插值、掩码+残差解码器实现波形重建；训练时使用 Gumbel‑Softmax 路由正则化，联合优化 SI‑SNR、说话人嵌入、难度估计与 PDI 约束。

**📊 数据集**

使用 PolyExtract 基准数据集：八种语言（英语、土耳其语、普通话、波兰语、基尼亚鲁旺达语、粤语、越南语、祖鲁语），每种语言均来自 LibriSpeech、AISHELL‑1、NCHLT、VIVOS、Common Voice 等公开语音数据，构成 64 种有向目标‑干扰者的两说话人混合。

**📈 对比分析**

与固定深度模型（如 TF‑GridNet、SpEx+、TF‑GridNet、TD‑SpeakerBeam 等）以及 LoRA 适配方法相比，LATTE 在 PolyExtract 上宏观 SI‑SNRi 达到 15.07 dB，计算量 0.30 GFLOPs，仅比 0.53 GFLOPs 的最深路径低 43.4%，并将 ASR WER 从 16.8% 降至 14.6%（在冻结 Qwen3‑Omni 上）。在多语种 ASR、语义理解任务上也持续提升，保持了与后端模型的兼容性。

**⚠️ 局限性**

局限性：实验仅覆盖控制的两说话人混合；未验证在未见语言、口音差异或真实世界多说话人混合中的泛化性能；缺少对延迟、能耗等部署指标的评估。

---

## 511. EviRepair: Evidence-Driven Agents for Reliable Vulnerability Repair

**arXiv ID:** 2610.13491 | [PDF](https://arxiv.org/pdf/2610.13491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 512. GraftOOD: Grafting Out-of-Distribution Exemplars into Advantage-Weighted Rectification for Video Reward Optimization

**arXiv ID:** 2610.13470 | [PDF](https://arxiv.org/pdf/2610.13470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. Formalisation of information processing and intelligence in proteinoid networks

**arXiv ID:** 2610.13545 | [PDF](https://arxiv.org/pdf/2610.13545v1)

**作者:** Andrew Adamatzky `[一作]` (University of West of England), Panagiotis Mougkogiannis `[通讯]` (University of West of England)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对蛋白质微球网络的电活动进行统一的动力学与信息论建模，并提出任务相对的智能功能指标。

**💡 创新点**

创新性地将可自适应耦合、记忆、传播与信息传递整合到一个可计算的框架，给出可量化的智能指数和理论界限。

**🔧 技术方法**

使用非线性可兴奋模型（如 FitzHugh–Nagumo）、Kuramoto 以及反应扩散耦合，结合信息论（互信息、转移熵、存储容量）与优化理论。

**📊 数据集**

未使用实验数据，依赖数值模拟产生合成时序数据，用于验证理论预测。

**📈 对比分析**

通过数值实验与信息论指标对比单体微球与耦合网络、固定耦合与可适应耦合，结果显示网络在分类精度、记忆容量和鲁棒性上优于单体，并且在耦合强度达到临界区后性能最大。

**⚠️ 局限性**

主要限制在于模型为经验性，未确定微球内部电活动的具体物理机制，实验验证尚待完成，且对不同网络拓扑和参数的泛化性仍需进一步探索。

---

## 514. Branching On-Policy Distillation

**arXiv ID:** 2610.13481 | [PDF](https://arxiv.org/pdf/2610.13481v1)

**作者:** Dongyang Chen `[一作]` (Tsinghua University), Lewei Lu `[通讯]` (SenseTime Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种新的基于分支的上采样蒸馏方法 BranchOPD，能够在学生生成的轨迹中选取不确定性高且教师与学生意见不一致的决策点，从这些点重新采样后续行为，并对主轨迹与分支轨迹中的所有 token 进行教师监督，同时引入协议边界修复 PBR 来消除不符合任务协议的冗余或错误输出。

**💡 创新点**

创新点包括：
- 在蒸馏过程中不再仅依赖单一路径，而是通过共享前缀进行分支采样，显著提升了轨迹覆盖率；
- 采用学生不确定性与教师-学生分歧的混合评分来选择分支点，兼顾信息量与纠错机会；
- 在数学推理与多轮对话任务中均可使用 token‑级监督，且不需要额外的任务奖励；
- 引入 PBR，通过检测与强制协议事件（如结束符、标签等）来减少冗余生成和越界问题。

**🔧 技术方法**

使用的主要技术包括：
- 上采样蒸馏 (OPD) 的 policy‑gradient 与 KL 散度损失；
- 分支采样策略（共享前缀、分支点选择、k 条后缀采样）；
- 依据 entropy 与 disagreement 计算的分支评分；
- PBR 的协议约束损失（类似 unlikelihood 训练）。
- PPO clip 剪辑作为优化器。

**📊 数据集**

数据集：
- 数学推理：DAPO‑Math‑17K、MATH‑500、AIME 2024 与 AIME 2025；
- 代理任务：ALFWorld、WebShop、Search‑QA；
- 训练时的学生与教师模型为 Qwen 系列（4B、8B、30B 等）。

**📈 对比分析**

比较方法：
- Vanilla OPD、Multi‑Rollout OPD（7 条独立轨迹）、Entropy‑Weighted OPD、离线 KD、GRPO、EOPD、SPOT、TCOD 等；
- 所有基线均在相同的 PBR 设置下评估。结果表明：
  - 在数学推理上，BranchOPD 的 avg@16 提升 18.55 分，超越 KD、GRPO、EOPD、SPOT 等多种基线；
  - 在 ALFWorld 上，平均成功率提升 13.83 分；
  - 在 WebShop 与 Search‑QA 上亦实现显著提升；
  - 与 7 条独立轨迹相比，BranchOPD 在 ALFWorld 与 WebShop 的学生 token 产生量分别减少约 19% 与 31%。

**⚠️ 局限性**

局限性：
- 需要在训练阶段保存并重新恢复共享前缀状态，增加了实现复杂度；
- 分支点的选取仍依赖超参数（B、k、α 等），对不同任务可能需手动调优；
- 对极大状态空间的任务，分支可能无法覆盖所有关键分支点；
- PBR 的规则是基于固定检测器，可能无法捕捉所有协议违规情况。

---

## 515. Towards Enriching Design-Time Models with Evidence from Simulation and Operational Activities

**arXiv ID:** 2610.13456 | [PDF](https://arxiv.org/pdf/2610.13456v1)

**作者:** Barbara da Silva Oliveira `[一作]` (Université Côte d’Azur), Nicolas Ferry `[通讯]` (Université Côte d’Azur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证一种基于证据的工作流，将仿真与运行阶段获得的观察结果系统化地转移回设计时的模型中；以影响模型（Influence Model）为示例，展示如何利用仿真实验产生的灵敏度分析结果、交互性指标、函数抽象以及有效性范围等证据进行模型增强。

**💡 创新点**

创新点在于：① 明确了从原始观察到设计时模型丰富的完整路径，填补了“执行证据到模型更新”的空白；② 将域特定关注点、证据计划、解释计划等步骤与模型增量化操作相结合，实现了可追溯、可审计的模型演化；③ 通过影响模型的多层抽象展示了工作流在具体建模语言上的可操作性。

**🔧 技术方法**

主要技术包括：仿真与实验（Simulink/Gazebo co-simulation）、灵敏度分析（Morris指数）、误差估计、函数抽象（解析表达式/单调性查找表）、模型增量化（设计时模型补丁）、手工审阅与验证、版本化存储。

**📊 数据集**

使用的“数据集”是自行生成的仿真实验数据，来自TurtleBot3 Waffle Pi 的标志跟随机器人实验，包括传感器日志、输出 SRP（系统响应属性）值、参与者配置、实验场景和随机种子等信息；并未使用公开的公开数据集。

**📈 对比分析**

方法比较：本文未与其他自动化或经验手工方法进行量化对比；仅通过单一案例（标志跟随机器人）展示工作流的可行性，未给出性能指标；评价侧重于模型增量的有效性、可追溯性和对设计时推理的支持。

**⚠️ 局限性**

局限性：① 目前工作流仍处于概念化/原型阶段，缺乏完整自动化实现；② 评估范围仅限于仿真数据，未涉及真实运行监控数据；③ 仅在影响模型上验证，需进一步验证在其他模型类型和 CPS 领域的通用性；④ 需要人工审阅和验证，难以满足大规模快速迭代需求。

---

## 516. SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability

**arXiv ID:** 2610.13461 | [PDF](https://arxiv.org/pdf/2610.13461v1)

**作者:** Pablo Manrique `[一作]` (Technical University of Vienna), Stefan Szeider `[通讯]` (Technical University of Vienna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一个可持续发展的数学 benchmark——SqueezeBench，基于 Lean 4 记录极限组合学问题的可计算上下界，并允许 AI 系统提交改进记录并自动验证。

**💡 创新点**

其创新点在于：① benchmark 无固定终点，记录为可计算函数并通过双核 Lean 验证；② 引入“持续记录更新”机制，成功保持问题活跃；③ 提供完整提交管道和自动化检验流程；④ 在十个极限组合学问题上开展 AI 代理评测，展示 benchmark 的可操作性。

**🔧 技术方法**

技术实现主要使用 Lean 4 与 Mathlib、Comparator 工具、两台独立 Lean kernel（包括 nanoda）进行双核验证；利用心跳计量的计算预算、计算机搜索检测；以及两阶段 AI 代理（Opus 5、Fable 5.1、Opus 5.5，偶尔调用 GPT‑6）生成 Lean 证明并提交。

**📊 数据集**

数据集包括十个极限组合学问题（cap set、Roth、Ruzsa–Szemerédi、Golomb ruler、C_4‑free、girth‑5、hypercube C_4‑free、Ramsey、R(3,k)、Schur），为每个问题提供初始上下界记录，并基于 Mathlib 定义相关函数。

**📈 对比分析**

比较方法是让每个代理在相同预算（最多 1‑2 小时）下，基于当前记录进行改进并提交；Baseline Opus 5 在 18.9 h 内完成 82 条接受记录，随后 Fable 5.1/Opus 5.5 在约两小时内再得到 60 条接受记录，提升了 19/20 条记录；目前仍有 8 个问题的记录落后于文献最优。

**⚠️ 局限性**

局限性包括：只覆盖单参数极限组合学问题；受限于 Mathlib 与计算预算，无法验证完全枚举的结果；未进行统一基准比较；未来需扩展多参数问题、引入 SAT 检查器等方式提升 benchmark 范围与严谨性。

---

## 517. Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning

**arXiv ID:** 2610.13504 | [PDF](https://arxiv.org/pdf/2610.13504v1)

**作者:** Oriol Corcoll `[一作]` (Spotify), Ciarán Mark Gilligan-Lee `[通讯]` (Spotify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Explanatory World Models（EWM），通过解释已发生的结果来重新分配奖励，以解决长期决策中的信用分配问题。

**💡 创新点**

创新点在于：①把信用分配视为“解释”问题；②设计了保守的分层重分配算子（CHO），可递归把奖励迁移到解释相关的过去事件；③利用前沿语言模型在未见过的轨迹上生成解释图。

**🔧 技术方法**

技术包括强化学习（PPO、A2C等）、EWM框架、保守分层重分配算子、前沿语言模型（Claude Sonnet 4.5）以及离线/在线RL整合。

**📊 数据集**

数据集有：Crafter、WordCraft（长周期RL基准），KuaiRand（公开序列推荐基准）以及真实行业规模的生成式推荐日志。

**📈 对比分析**

与传统方法（PPO、LaRe、SFT、A2C）对比，EWM在奖励延迟最严重时提升显著；在Crafter、WordCraft上击败PPO，KuaiRand离线/在线上超越SFT、LaRe和PPO；在行业推荐评估中，Recall和后续互动指标提升约3倍。

**⚠️ 局限性**

局限性包括：依赖语言模型解释质量，解释与奖励分配的超参数（阈值、深度）需要调优；重分配过程可能增加推理成本；在极端稀疏奖励或复杂因果关系多变的环境中解释误差可能导致信用误分。

---

## 518. Breaking Two Barriers for Dynamic Planar Reachability

**arXiv ID:** 2610.13473 | [PDF](https://arxiv.org/pdf/2610.13473v1)

**作者:** Adam Karczmarz `[一作]` (University of Warsaw), Michał Włodarczyk `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文设计了两种全动态可达性数据结构，分别在离线模式下实现 O(n^0.49) 的更新/查询时间，在在线模式下实现 O(n^0.664) 的摊销更新/查询时间，从而突破了平面有向图可达性问题的平方根壁垒。

**💡 创新点**

创新点在于首次利用 VC 维度与 stabbing 数的稀疏化框架构造稀疏可达性仿真器，并将其与层次划分、r‑划分等技术相结合，实现了对平面有向图（及其 K_h‑minor‑free 通用类）可达性的时间突破。

**🔧 技术方法**

核心技术包括：VC 维度约束的集合系统分析、Stabbing Number Theorem 产生高效的区间表示、稀疏可达性仿真器构造、递归层次压缩、r‑划分与 s‑划分的组合以及基于递归重建的在线结构。

**📊 数据集**

该工作为理论算法研究，无实验数据集；所有结果均基于严格的理论分析与证明。

**📈 对比分析**

与最优的 O(n^2/3) 算法相比，在线更新/查询时间从 0.666… 降至 0.664，离线模式从 0.5 降至 0.49，显著提升了理论上可达性维护的效率。

**⚠️ 局限性**

限制在于算法仍是随机化且在实际实现中复杂度高；仅适用于平面有向图（以及更宽泛的 K_h‑minor‑free 图）；未给出多项式预处理下的更快上界；以及对超平面图或一般有向图的可扩展性仍待研究。

---

## 519. A Better Metric Will Not Save You: What You Compare Beats How You Compare It

**arXiv ID:** 2610.13476 | [PDF](https://arxiv.org/pdf/2610.13476v1)

**作者:** Noureddine Khaous `[一作]` (Université Paris Cité), Guillaume Wisniewski `[通讯]` (Université Paris Cité)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在十五种语言的读音语料中对27种声音表征进行RSA、CKA和Procrustes相似度分析，系统评估了条件集合（即所选元音类别、输入录音段长度、聚合方式等）对比较结果的影响；

**💡 创新点**

创新点在于首次将外部可知的元音几何结构作为参照，对比度量前的条件集合选择（包括类别选择、采样阈值、聚合估计等）进行可测量化评估，并量化其对模型排名和效应大小的实质性影响；

**🔧 技术方法**

使用的技术包括：RSA、线性CKA、匹配维度的Procrustes；声学特征提取（Praat形式音）与音素对齐（forced alignment）；数据聚合（token均值、类别中位数）；以及构建锚定尺度的RSA相关分析；

**📊 数据集**

数据集为FLEURS读音语料库，挑选15种语言，覆盖5-28个元音类别，并使用27个表征（自监督、监督、无监督、压缩与随机初始化控制等）；

**📈 对比分析**

比较方法是在每种表征下先对条件集合固定，然后用RSA与外部元音几何计算RDM并与参考RDM做Spearman相关；性能以“预训练模型与无监督谱特征的gap”在锚定尺度上的百分比来衡量，结果显示类别组合对gap的影响可超过预训练效应本身；

**⚠️ 局限性**

限制在于仅探讨元音几何这一已知条件集合，英语手工构建的词表对结果影响大；外部参考本身受测量误差限制；其他未评估的“分岔”选择（如层深度匹配、投影秩）可能产生类似或更大影响；

---

## 520. When MLLMs miss what humans read: diagnosing and learning to read squint text

**arXiv ID:** 2610.13465 | [PDF](https://arxiv.org/pdf/2610.13465v1)

**作者:** Chen Peng `[一作]` (Beijing University of Posts and Telecommunications), Mengwei Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究多模态LLM在识别图像粗糙结构中的隐藏文本（Squint Text）时的失败，提出诊断与改进方法。

**💡 创新点**

提出 SquintOPSD 训练框架，通过下采样视图与真实文本作为教师隐私上下文，实现原图仅推理下的隐藏文本识别。

**🔧 技术方法**

使用对抗式教师-学生对齐、Jensen‑Shannon分布蒸馏、视觉‑语言模型 Qwen3.5‑9B 等技术。

**📊 数据集**

采用 1076 条生产审核图像（Squint Text dataset）和 4,589 条合成隐藏文本样本（SquintSynth suite）进行评估与诊断。

**📈 对比分析**

与基线模型（原始模型、下采样视图、SFT、DAPO）以及外部隐藏文本基准（Illusory VQA、AdvOCR、HC‑Bench）对比，SquintOPSD 在 Squint Text 上实现 88.61 ANLS，显著提升至 88.61，远超基线 26.50；在外部基准上也取得明显提升。

**⚠️ 局限性**

依赖可恢复的视觉证据，隐私监督受限；方法仅在可下采样可见文本的情况下有效，且在一定程度上牺牲通用视觉语言能力。

---

## 521. Exact rational certificates and a Lean formalization of the eleven-variable Vasc inequality

**arXiv ID:** 2610.13463 | [PDF](https://arxiv.org/pdf/2610.13463v1)

**作者:** Dakai Guo `[一作]` (Chinese Academy of Sciences), Xiao-Shan Gao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了十一变量Vasc不等式的正定性，并给出了两个完整的有理多项式正定性证书，随后在Lean 4中正式化并验证了这些证书。

**💡 创新点**

创新点在于：①将不等式拆解为“公共平移导数正定”和“边界面正定”两条多项式正定性条件；②构造了两份极小规模（共约53 MB）且严格可检验的有理Gram矩阵证书；③使用整数同余和严格对角占优判定快速验证矩阵正定性；④将所有检查步骤编译为可重复的Python脚本和Lean代码，形成可复制的完整验证链。

**🔧 技术方法**

主要技术包括：符号与数值半定规划（semidefinite programming）求解候选Gram矩阵；有理重构与整数同余修正；三角化存储的对称矩阵编码；整数严格对角占优判定；多项式的循环平均与符号化展开；Lean 4与Mathlib的形式化证明框架。

**📊 数据集**

数据集：包含两套完整的JSON证书数组（总计约53,960,275字节），包括93个导数块、512个边界块、对应的Gram矩阵、整数同余矩阵、以及乘子矩阵W；除此之外还有配套的Python脚本、Lean源文件与检查日志。

**📈 对比分析**

验证方法：Python脚本在单核环境下完成所有整数系数比较与矩阵正定性检验，时间分别为翻译证书4.71 s、边界证书3.84 s；Lean项目通过编译时的本地数值验证完成整个定理的形式化证明；与传统手工或数值验证相比，显著提升了可重复性与可信度。性能上，单个证书的验证耗时仅数秒，且占用内存均低于200 MiB。

**⚠️ 局限性**

局限性：①证书规模仍相对较大，构造过程依赖高性能半定规划求解器和数值重构步骤；②目前仅针对十一变量Vasc不等式，尚未探讨更大维度或其他形式不等式的可扩展性；③对整数同余矩阵的构造没有自动化证明最优性，未来可研究更紧凑或结构化的证书。

---

## 522. Deterministic Girth Approximation in Weighted Undirected Graphs

**arXiv ID:** 2610.13452 | [PDF](https://arxiv.org/pdf/2610.13452v1)

**作者:** Avi Kadria `[一作]` `[通讯]`, Avi Kadria

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

未提供

**💡 创新点**

未提供

**🔧 技术方法**

未提供

**📊 数据集**

未提供

**📈 对比分析**

未提供

**⚠️ 局限性**

未提供

---

## 523. Collective Autonomous Vehicle Routing Needs Social Awareness for Coexistence with Human Drivers

**arXiv ID:** 2610.13439 | [PDF](https://arxiv.org/pdf/2610.13439v1)

**作者:** Anastasia Psarou `[一作]` (Jagiellonian University), Rafal Kucharski `[通讯]` (Jagiellonian University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究混合自动驾驶车辆（AV）与人类驾驶车辆（HDV）共存的交通网络中，利用多智能体强化学习（MARL）学习路由策略，并在AV奖励中加入边际成本（marginal cost）以降低对其他车辆的外部性。

**💡 创新点**

创新点：①将边际成本作为奖励项，构建社会化奖励；②提出选择性边际成本奖励，只对最具影响力的AV计算边际成本，显著降低计算开销；③在真实城市网络上验证此方法，证明可同时改善AV、HDV和系统的行程时间。

**🔧 技术方法**

技术方法：多智能体强化学习（REINFORCE），SUMO 微观仿真与 RouteRL 框架；边际成本的计算与选择性奖励策略；使用标准差和相对旅行时间变化等指标评估性能。

**📊 数据集**

数据集：真实交通网络数据——Ingolstadt（850节点、1426连线、1035车辆）与Provins（2173节点、4216连线、523车辆），来自 InTAS 与 URB 基准；以及 800k 条联合行动样本用于评估 AV 影响力。

**📈 对比分析**

比较方法：与纯自私奖励（β=0）对比，测量 HDV、AV 与系统的相对旅行时间变化；在不同 β（0.1、0.2、0.5 等）以及选择性奖励比例（10%、40%、50%）下进行实验。实验结果显示：β=0.1 时 HDV、AV 与系统均改善；β 过大导致 AV 与系统性能下降；选择性奖励在 40%–50% 最具影响力 AV 时，几乎保持全奖励的效果，同时将额外仿真量降低约 40%。

**⚠️ 局限性**

局限性：①边际成本需要对每个 AV 进行一次额外反事实仿真，计算成本高；②选择性奖励需先采样大量联合行动以评估影响力，增加前置工作；③β 权重调优依赖经验；④实验仅在两个城市网络，结果的普适性仍需进一步验证。

---

## 524. Efficient Confidence-Aware Densification for 3DGS

**arXiv ID:** 2610.13541 | [PDF](https://arxiv.org/pdf/2610.13541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 525. Stride: Scale-Aware Multi-Agent Evolutionary Refinement for Infographic Generation

**arXiv ID:** 2610.13502 | [PDF](https://arxiv.org/pdf/2610.13502v1)

**作者:** Yan Zhou `[一作]` (Shanghai Jiao Tong University), Yang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

Stride 是一种训练无关的多代理框架，用于在冻结的文本到图像生成器上进行规模感知的演化式迭代精炼，以生成符合复杂信息图表约束的图像。

**💡 创新点**

创新点在于将约束评估与视觉定位拆分、依据错误规模动态路由全局重绘与局部修正、并使用 (1+λ) 演化策略保证只接受提升的候选，解决了尺度不匹配与迭代不稳定问题。

**🔧 技术方法**

核心技术包括多模态LLM驱动的验证器与感知器、基于约束的提示生成、全局composer与局部retoucher两子代理，以及基于约束分级的演化选择。

**📊 数据集**

使用了BizGenEval、IGenBench以及WISE三大公开基准数据集。

**📈 对比分析**

在所有基准上，Stride在开放权重生成器上显著提升性能，超越多家专有系统（如Nano‑Banana‑2.0、Seedream‑5.0‑Pro）并匹配或超过GPT‑Image‑2.0；BizGenEval hard分数提升约20–30分，IGenBench Q‑ACC亦提升0.12–0.16。

**⚠️ 局限性**

局限在于依赖高质量多模态LLM与外部检索，计算开销较大，并且对生成器的兼容性仍需进一步验证。

---

## 526. On the Alignment Between Spectral Efficiency and Positioning Accuracy in Multi-LEO Clusters

**arXiv ID:** 2610.13526 | [PDF](https://arxiv.org/pdf/2610.13526v1)

**作者:** Konstantinos Kapoglis `[一作]` (KU Leuven), Hazem Sallouha `[通讯]` (Chalmers University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多颗LEO卫星集群在直接到设备（D2D）场景下，频谱效率（SE）与定位精度（TOA和Doppler两种PEB）的关联关系。

**💡 创新点**

首次系统性评估通信性能与定位性能之间的统计相关性，揭示SE与Doppler定位误差呈强负相关，而与TOA定位误差相关性较弱。

**🔧 技术方法**

采用OFDM信号模型、Rician信道、联合信号处理与Fisher信息矩阵推导PEB，并通过Pearson与Spearman相关系数分析多集群尺寸下的关联性。

**📊 数据集**

使用MATLAB卫星通信工具箱模拟Starlink首层星座（1584颗卫星，530km高轨道），在40°N 74°W地面用户位置下收集可见卫星轨道及速度数据。

**📈 对比分析**

对所有15颗始终可见卫星的组合（4-7颗）计算SE、TOA-PEB、Doppler-PEB，并统计相关系数。结果显示，SE与Doppler-PEB在N>5时绝对相关系数>0.5，说明两者高度关联；而SE与TOA-PEB相关性仅在-0.05~-0.53之间。

**⚠️ 局限性**

局限性包括：仅考虑静态用户、忽略大气和时钟误差、仅研究Starlink星座、未探讨多天或不同轨道星座的时变性。

---

## 527. Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation

**arXiv ID:** 2610.13521 | [PDF](https://arxiv.org/pdf/2610.13521v1)

**作者:** Shi-Yu Tian `[一作]` (Nanjing University), Gu Simiu `[通讯]` (JD.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种跨教师守护（Cross‑Teacher Guarding）机制，用非路由教师对同一学生轨迹进行交叉评估，借助令牌级一致性校准奖励并通过预防性梯度投影消除梯度冲突，从而改进多教师 on‑policy distillation。

**💡 创新点**

创新点在于：①将非路由教师作为同轨迹的交叉批评者，利用教师间的一致性来动态调节令牌奖励；②设计基于同轨迹辅助梯度的预防性梯度投影，配合可调节的范数保持约束；③将令牌级信用分配与梯度投影无缝结合，显著降低多域联合训练中的互斥更新。

**🔧 技术方法**

技术要点包括 on‑policy distillation（OPD）、token‑level reverse‑KL 目标、交叉教师一致性得分、预防性梯度投影（含范数保持）以及与 PCGrad 的对比实验。

**📊 数据集**

实验数据集涵盖 11 个基准，分别属于 Code（HumanEval+、MBPP+、LiveCodeBench）、Math（AIME 2024/2025、HMMT）和 Medical（MedQA、MedXpertQA、MedMCQA、SuperGPQA‑Med）。

**📈 对比分析**

与 Vanilla MOPD、ExOPD、CaMOPD、Open‑MOPD 以及 PCGrad 进行对比。实验显示，在 Math×Medical、Code×Medical 两域高冲突场景下，跨教师守护可提升医疗域准确率 5.6–8.8% 并整体平均提升 0.72–2.89%；在低冲突和三域设置下亦保持领先或相当于最优基线。

**⚠️ 局限性**

局限性：①仅在三类文本推理域（Code、Math、Medical）验证；②引入辅助教师增加训练成本，且规模化到更大教师池需高效教师选择与兼容性评估；③实验使用的 Qwen3‑4B/1.7B 学生模型，性能可能受限于模型大小。

---

## 528. ToFu: Exploiting and Fuzzing the LLM-Orchestrator-Tool Trust Boundary

**arXiv ID:** 2610.13651 | [PDF](https://arxiv.org/pdf/2610.13651v1)

**作者:** Md Nazmul Islam Ananto `[一作]` (Pennsylvania State University), Arslan Khan `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对 LLM‑工具交互边界的工具感知模糊测试框架，自动从工具元数据推导语义安全检查，生成基于模式的 fuzzing harness，并通过运行时动态追踪与语义 oracle 检测安全缺陷。

**💡 创新点**

创新点在于：①把 LLM 传递给工具的自然语言描述与参数 schema 视为安全契约，利用这些契约构造 74 种基于 CWE 的语义感知 sanitizer；②在 fuzzing 过程中不依赖传统崩溃/断言 oracle，而是通过运行时事件和工具输出做语义判定；③在工具级别实现自动化的 sanitizer 选择与种子生成，兼顾多语言（Python/JavaScript）与多工具（3405+），显著提升了发现非崩溃安全缺陷的覆盖率。

**🔧 技术方法**

核心技术包括：①LLM（gpt‑4.1）用于解析自然语言元数据并映射到 sanitizer 集合；②Atheris fuzzing 与边界覆盖引导的输入变异；③Python 运行时 tracer 监视文件、网络、进程等安全敏感操作；④语义 oracle 通过匹配预期行为与观察到的事件来判定违规；⑤对照 5 个行业基线（ToolFuzz、VIPER‑MCP、Cisco MCP Scanner、CodeQL、Sonnet‑5）进行评测。

**📊 数据集**

数据集基于公开的 MCPCorpus，挑选 273 个可执行的 MCP 服务器（共 3405 个工具），通过自动化脚本克隆、构建并提取工具元数据，随后在 60 秒/工具预算下进行 fuzzing，另外 34 个服务器进行 24 小时深度跑。

**📈 对比分析**

与 5 个基线对比，本文方法在 273 个可执行服务器中发现 118 服务器、343 个工具共 253 条经过人工验证的安全漏洞，覆盖范围比任何单一基线均优，且在 60 秒预算内即可发现 231/253（≈91%）的已验证漏洞；24 小时深度跑进一步收敛至全部漏洞；相较于仅基于崩溃或静态检查的工具，本文的语义 oracle 能发现 100% 的非崩溃安全缺陷。

**⚠️ 局限性**

主要限制包括：①对多代理系统支持不足，无法处理跨代理工具调用链；②第三方组件（如浏览器内核、外部服务）未被完整覆盖，导致覆盖率偏低；③状态依赖的工具交互未能通过多调用序列模糊测试捕获；④对 LLM 的 sanitizer 选择与种子生成仍有误判，导致部分漏洞被遗漏；⑤运行时追踪仅针对 Python 目标，对 JavaScript 等语言的完整性不足。

---

## 529. AutoBio+: Benchmarking Dexterous Manipulation with Dynamic Experimental Phenomena in Realistic Laboratories

**arXiv ID:** 2610.13643 | [PDF](https://arxiv.org/pdf/2610.13643v1)

**作者:** Xuanye Wu `[一作]` (Shanghai AI Laboratory), Guangtao Zhai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在自动化生物实验室领域提出并实现了 AutoBio+ 基准，涵盖14项细粒度操作任务、4种机器人手臂，加入可控动态实验现象（火焰/蒸汽/气泡）并提供混合渲染与并行仿真框架。

**💡 创新点**

创新点在于：①首次在实验室仿真中引入可控动态视觉效应；②采用 3D Gaussian Splatting 与 MuJoCo 前景混合渲染提升视觉真实性；③设计 Parallel Finite‑State Machine (PFSM) 实现高效并行数据采集与评估；④将任务划分为功能工具使用、精细操作、语言视觉定位、仪器操作四大能力类别。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting、粒子视觉效果系统、MuJoCo + mjlab 并行仿真、PFSM、Diffusion Policy、Fast‑WAM、GR00T‑N1.7 等视觉‑语言‑动作模型。

**📊 数据集**

数据集来源于自动抓取生成的任务特定抓取候选并人工验证，配合 6 个真实实验室场景扫描得到的 Gaussian 背景；评估使用 300 次 roll‑out（每类 3 种随机种子各 100 次）产生的数据。

**📈 对比分析**

通过在 12 个代表性任务上比较 Diffusion Policy、π0.5、GR00T‑N1.7、Fast‑WAM 四种模型，发现成功率差异显著，语言定位与接触丰富任务表现最低；Gaussian 背景显著降低 sim‑to‑real 问题，提升了少样本迁移效果。

**⚠️ 局限性**

限制包括：①语言视觉定位能力仍弱，难以可靠实现语言‑空间绑定；②预训练手掌虽提升简单任务，但在接触丰富、精细对齐任务上效果不佳；③接触感知与细粒度手指协调仍是主要瓶颈；④动态流体与交互渲染尚未完全逼真；⑤任务与资产规模有限，难以覆盖所有实验室操作。

---

## 530. Benchmarking Answer-Grounding Consistency in Remote Sensing Visual Question Answering

**arXiv ID:** 2610.13636 | [PDF](https://arxiv.org/pdf/2610.13636v1)

**作者:** Ping Guo `[一作]`, Kamata Seiichiro `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了答案-证据支持协议（AESP），用于评估遥感视觉问答模型的回答与视觉证据的一致性，并构建了包含完整对象级注释的G‑RSVQA基准集。

**💡 创新点**

创新点在于通过任务特定规则（存在、计数、比较）判断模型自身答案是否被预测的目标框支持，并为每个问题提供完整的对象级注释，填补了仅评估答案或定位的空白。

**🔧 技术方法**

采用基于对象级注释的覆盖计数匹配、答案抽取器（Qwen3‑14B）以及多种视觉语言模型的联合评估，定位质量以IoU≥0.5的精度/召回率衡量。

**📊 数据集**

使用DIOR、NWPU VHR‑10和SODA‑A三大遥感数据集，生成共15,027个问题（存在8,592、计数4,275、比较2,160），并附上全部相关对象级注释。

**📈 对比分析**

在七种无额外训练的基础模型上进行对比实验，发现答案准确率与定位精度并不能反映证据支持率；计数任务的支持率最低；最优模型Qwen3‑VL‑8B整体证据支持率为80.11%，联合成功率为63.76%。

**⚠️ 局限性**

局限性包括仅考虑单目标覆盖计数，未处理多目标遮挡或视觉歧义；基准缺乏多模态交互和动态场景，未来需引入更丰富的评价指标。

---

## 531. APEX-Harm: Evaluating Agent Safety in Shared Professional Workspaces

**arXiv ID:** 2610.13623 | [PDF](https://arxiv.org/pdf/2610.13623v1)

**作者:** Ziyi Yang `[一作]` (King Abdullah University of Science and Technology), Carlos Hinojosa `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在专业共享工作空间中，构建并评估了一套新的安全基准（APEX‑Harm），用于检测大型语言模型代理在处理财务、法律和咨询任务时的提示与脚本注入攻击；

**💡 创新点**

创新点在于引入大规模、跨应用的工作空间环境，设计了六类现实攻击场景，并同时衡量代理对攻击的曝光率与执行成功率；

**🔧 技术方法**

使用了多模态工具调用（命令行、邮件、日历、文档等）与代码执行、基于规则与LLM的安全/曝光判定技术；

**📊 数据集**

数据集基于Apex‑Agents的15个专业任务，扩展为180个含攻击的实例，涵盖115–328文件、邮件、日历等多种资源；

**📈 对比分析**

通过在12种前沿模型（Gemini 3.8 Flash、Opus 5、Sonnet 5等）上进行一次性评测，报告攻击成功率、曝光率与任务得分；结果显示Opus 5/ Sonnet 5攻击成功率最低（≈8–20%），但Gemini 3.8 Flash虽任务得分最高（0.877）却仍有33%攻击成功；

**⚠️ 局限性**

局限性包括：仅覆盖三大专业领域；每个实例仅评测一次，未考虑运行方差；攻击种类与威胁模型可能不足以覆盖所有实际情境；评判者主要基于规则，尽管验证率高但仍有限。

---

## 532. BLiMPPI: Benchmark of Linguistic Minimal Pairs for Polarity Items

**arXiv ID:** 2610.13607 | [PDF](https://arxiv.org/pdf/2610.13607v1)

**作者:** Arzu Burcu Güven `[一作]` (IT University of Copenhagen), Rob van der Goot `[通讯]` (IT University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多语种 NPI 最小对比评测套件，覆盖 33 种语言，包含 12,866 对自然语言最小对比样本。

**💡 创新点**

创新点在于自动化从 UD treebanks 发现并标注 NPI 与其许可环境，生成自然语言最小对比，并引入 NPI 许可熵、距离与 n-gram 可解性等新评估指标。

**🔧 技术方法**

使用了基于 UD 句法树的许可环境检测、语法扰动生成对比、n-gram 语言模型作为基线，以及大型多语言预训练模型（Gemma、Apertus、Llama、Qwen、Aya 等）和单语模型进行评估。

**📊 数据集**

主要数据来源是 UD v2.18 语料库、英语 NPI bank 和荷兰 NPI lexicon，结合自动发现得到的 33 种语言的 NPI 数据。

**📈 对比分析**

通过句子层面困惑度比较模型对正反对比的偏好，报告准确率；多语言模型在英语上超过 98%，但在低资源或非英语中表现显著下降，Gemma‑3‑27B 和 Apertus‑70B 成为最佳；n-gram 可解性是预测错误的最强指标。

**⚠️ 局限性**

局限性包括仅覆盖 UD 中资源丰富的语言、自动发现方法质量不一、缺少条件和比较许可者的支持、训练数据分布信息不完整，以及对某些低资源语言的手工配置依赖。

---

## 533. zip2zip++: Towards Inference-Time Token Compression with Minimal Quality Loss

**arXiv ID:** 2610.13605 | [PDF](https://arxiv.org/pdf/2610.13605v1)

**作者:** Saibo Geng `[一作]` (EPFL), Robert West `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对预训练语言模型的固定BPE词表，本文将其升级为基于LZW的动态词表，并通过多项架构与算法改进，显著提升压缩后模型的困惑度与推理质量。

**💡 创新点**

创新点包括：①将输入与输出的超词嵌入分离，消除前缀对齐冲突；②提出“尺寸感知RoPE”，让超词位置根据其覆盖的基词跨度分配；③采用残差式身份初始化的超词编码器，解决嵌入尺度失衡；④设计“容错LZW”码流，允许保护数字等脆弱 token，防止误合并；⑤引入多视角困惑度，克服 LZW 产生多重合法分割导致的评估偏差。

**🔧 技术方法**

技术手段主要有：LZW流式压缩、超词分层编码器、RoPE位置编码、残差式身份初始化、LoRA 低秩微调、容错 LZW 解码、以及多视角困惑度计算。

**📊 数据集**

实验使用四种模型（Phi‑3.5‑mini‑instruct 3.8B、Phi‑3‑medium‑4k‑instruct 14B、Llama‑3.2‑1B/3B‑Instruct），在 1B 规模的指令微调数据上训练；评估集包括六个两步多选基准、GSM8K、MATH‑500、HumanEval、IFEval、RULER，以及 Wiki、Pile、mC4、dC4 的字节级困惑度。

**📈 对比分析**

与原始 zip2zip 及未压缩基准对比，改进版 zip2zip++ 在所有四个语料库上均降低字节级困惑度（最小 1.6× 的多视角计算成本），在两步基准和推理任务中取得 20–30% 的准确率提升，数学推理任务近乎赶上未压缩模型；但在代码生成与指令跟随等任务提升有限。

**⚠️ 局限性**

限制与挑战包括：①压缩后仍未完全消除与未压缩模型的性能差距；②多视角困惑度计算在大规模推理时开销增大；③容错 LZW 与数字保护等策略需要根据任务手动调整；④未在长上下文训练中验证，长序列性能仍受限；⑤对不同语言或领域的通用性尚待进一步验证。

---

## 534. Limit-sure Winning in Team Games without Shared Randomness

**arXiv ID:** 2610.13569 | [PDF](https://arxiv.org/pdf/2610.13569v1)

**作者:** Léonard Brice `[一作]` (Institute of Science and Technology Austria), K. S. Thejaswini `[通讯]` (Free University of Brussels)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在团队成员没有共享随机源的并发随机博弈中，限制可达性（limit-sure reachability）和安全性（limit-sure safety）问题的计算复杂度，给出了 NP 上下界并证明安全性与几乎必然性等价；同时将结果推广至 IRATL 逻辑的模型检测。

**💡 创新点**

创新点在于：①证明无共享随机的团队博弈中限制可达性为 NP‑完整；②引入 Puiseux 幂级数、行动秩、最小秩森林判据以及编码 MDP 与稳定响应的概念；③通过加权图论的 Matroid Intersection 检查稳定性，完成多项式时间的验证；④将这些技术应用于 IRATL 的定性模型检测，给出 Δ₂P‑完备性。

**🔧 技术方法**

主要技术包括：Puiseux 系列展开、秩（rank）定义、最小秩生成树与森林公式、编码 MDP 与稳定响应、Cramer's 规则与 Hadamard 不等式、Matroid Intersection、极限理论以及 NP 完备性归约。

**📊 数据集**

工作基于理论分析和构造性证明，未使用任何真实数据集；所有实验与验证均在符号化的游戏实例上进行。

**📈 对比分析**

与传统的共享随机源的两人博弈相比，本研究表明无共享随机的团队博弈在限制可达性问题上仍为 NP 级，安全性问题可在多项式时间内解决；IRATL 的模型检测在 Δ₂P 上下界，复杂度与现有工具相当。

**⚠️ 局限性**

局限性：仅处理可达性与安全性目标，未覆盖 Büchi、co‑Büchi、Parity 等 ω‑regular 条件；仅考虑完全无共享随机的团队，未对部分共享随机模型展开；未给出实际实现或实验评估，未来工作需进一步探索多玩家通用化与更多随机化模型。

---

## 535. Rationally Linearized MLPs

**arXiv ID:** 2610.13568 | [PDF](https://arxiv.org/pdf/2610.13568v1)

**作者:** Arijit Sehanobish `[一作]` (Kensho Technologies), Krzysztof Choromanski `[通讯]` (Google Deepmind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用低阶有理函数逼近激活函数的线性化多层感知机（RaLiMP），通过随机特征实现高效的 MLP 近似；

**💡 创新点**

创新点在于将有理函数的分式形式与 Laplace 变换结合，构造可分离的随机特征，从而在保持低方差的同时显著降低逼近误差；

**🔧 技术方法**

采用随机特征（Gaussian 与 softmax‑kernel）、分式分解、Laplace 变换与随机 Fourier 特征等技术；

**📊 数据集**

在多种数据集上验证：3D 任务（NeRF、SIREN、PINN）、视觉任务（CIFAR‑100、Places365、ImageNet、VTAB‑1k）、语言任务（GPT‑2、Qwen‑3‑8B、GLUE）、语义分割（ScanNet、ScanNet200、nuScenes）；

**📈 对比分析**

与低秩近似、SNNK、EUGen 等线性化方法相比，RaLiMP 在保持或提升准确率的同时实现 30%‑57% 的内存/参数压缩，推理速度提升 6%‑50%；

**⚠️ 局限性**

局限性包括：对输入范数敏感，需要通过缩放参数 τ 控制方差；分式逼近在极大输入范围内仍需多极点，导致计算开销；在极小数据集或对极致精度要求的任务中逼近误差可能不够理想。

---

## 536. Probabilistic Ship Trajectory Forecasting using Transformer Models on AIS Data

**arXiv ID:** 2610.13557 | [PDF](https://arxiv.org/pdf/2610.13557v1)

**作者:** Frederik E. T. Schöller `[一作]` `[通讯]` (Technical University of Denmark), Frederik E. T. Schöller (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 Transformer 与 Performer 模型，结合 KDE 生成的热图和 CNN 空间特征提取器，训练并预测海上船舶的轨迹，且采用分位数损失一次性输出 95% 置信区间。

**💡 创新点**

1) 将空间热图信息融入 Transformer 结构；2) 用分位数损失直接生成预测区间，避免多次前向传播；3) 对比传统 LSTM‑GAN、线性回归与高斯过程，证明 Transformer/Performer 在海上轨迹预测中的优势。

**🔧 技术方法**

Transformer/Performer 架构、Fast Attention via FAVOR+、CNN（2D KDE 生成热图、卷积特征提取）、分位数损失训练、基于 AIS 的位置重采样与相对坐标转换。

**📊 数据集**

AIS 数据，覆盖丹麦水域三大航道（Great Belt、Hven、Falsterborev），共 1,636,220 条轨迹；测试集包含同区域不同时间的 20,000 条轨迹和未见区域 Oresund 的 20,000 条轨迹。

**📈 对比分析**

与线性回归、Gaussian Process、LSTM‑GAN 以及无热图的 Transformer/Performer 进行比较。Transformer/Performer+热图在三大测试集平均 ADE≈0.18 km、FDE≈0.42 km、IAcc≈91%，显著优于基线（ADE>0.38 km、IAcc<80%）。推理速度仅 6–7 ms/预测，远快于 LSTM‑GAN 的 157 ms。

**⚠️ 局限性**

空间热图特征提取子网络在未见区域容易过拟合，导致性能下降；对小型无 AIS 船只或不同航道的推广性有限；模型对训练周期和热图生成时间有一定依赖，需要针对部署路线预训练。

---

## 537. Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?

**arXiv ID:** 2610.13550 | [PDF](https://arxiv.org/pdf/2610.13550v1)

**作者:** Kshitij Ambilduke `[一作]` (Université Paris-Saclay), Marcely Zanon Boito `[通讯]` (NAVER LABS Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为BIPoS Benchmark的新基准，用三轴（稳健性、刻板偏见、个性化）和四种评估模式（speech‑implicit、speech‑explicit、text‑implicit、text‑explicit）全面评估语音LLM的声学、基础模型和交互效应，并对八大主流模型进行系统比较。

**💡 创新点**

创新点在于：①将刻板偏见与个性化分离，提出声学效应、基础模型效应、交互效应三种来源的框架；②发布了Spoken StereoSet+、ELIP-Pair、BASIS‑Health/Product等针对语音的专属数据集；③构建了四种评估模式，能够区分模型是否识别声学线索、是否使用这些线索以及是否将其转化为行为。

**🔧 技术方法**

采用端到端语音LLM与文本提示相结合的推理，使用ASR、SQA、语音分类、CAT/ICAT评估指标和LLM‑as‑judge；通过四种模式的组合实现对声学、基础和交互效应的解耦分析。

**📊 数据集**

使用的数据集包括Fair‑Speech、Spoken StereoSet+（ASR、SQA）、SD‑QA、LibriSQA、Spoken StereoSet+、ELIP‑Pair、BASIS‑Health 与 BASIS‑Product。

**📈 对比分析**

对八个模型在四种评估模式下进行量化比较，发现：①内容任务（ASR、SQA）在不同年龄、性别、口音上表现基本均衡；②在个性化上，模型对年龄与性别的适配差异显著，基本模型往往更偏好女性化回答；③刻板偏见主要来源于基础模型，声学交互效应小；⑥绘制Pareto前沿显示较低偏见的模型往往个性化能力弱，强调两者之间的权衡。

**⚠️ 局限性**

限制包括：仅评估单一刻板偏见（未覆盖交叉/多维偏见）；性别采用二元化分类，年龄/性别划分过于粗糙；使用合成语音可能不完全代表真实语音的多样性；数据集规模和多样性有限，缺少跨文化、口音丰富的样本；模型在将声学线索转化为适当行为方面仍存在显著不足。

---

## 538. A Bicriteria Approximation Scheme for the Minimum Well-Separated Pair Decomposition

**arXiv ID:** 2610.13650 | [PDF](https://arxiv.org/pdf/2610.13650v1)

**作者:** Guilherme D. da Fonseca `[一作]` `[通讯]` (Aix-Marseille University), Guilherme D. da Fonseca (Aix-Marseille University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在固定维度欧氏空间和双曲度有限的度量空间中，如何在可接受的时间内构造接近最优大小的 WSPD（Well‑Separated Pair Decomposition），并证明在一般度量空间中，放宽分离度到小于 2 时问题仍然是 NP‑难的。

**💡 创新点**

提出了一个双指标逼近方案：在固定维度 d 和给定的 δ∈(0,1] 下，能够在 f(d,δ)·n⁶ 时间内得到一个 (1+δ)‑近似大小且分离度仅放宽到 (1+2δ/3) 的 WSPD，突破了此前只能得到 2^O(d) 近似的结果；同时给出了针对双曲度有限度量空间的扩展。

**🔧 技术方法**

主要技术包括：
- 对点对按尺度分层（shift over scales）以控制尺度比；
- 对每一层使用网格或簇的离散化（rounding）以限制局部问题规模；
- 在兼容图（compatibility graph）上采用局部增长（local growth）与基于邻域的最优子问题求解；
- 结合现有的局部增长框架（Nieberg–Hurink–Kern 等）与分层策略；
- 对一般度量空间给出硬性限制的归约（block metrics + clique cover 归约）。

**📊 数据集**

论文属于理论计算机科学，无实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与已知工作比较：
- 在 ^d 中之前的最好结果是 2^O(d) 近似；本文提供 (1+δ) 近似。
- 对于双曲度有限度量空间，之前仅有 2^O(κ) 近似；本文改进为 (1+δ) 近似。
- 在一般度量空间中，本文证明即使分离度放宽到小于 2，仍无法在多项式时间内获得比 N^c 近似（c<2/3）的解，进一步强化了之前的 NP‑难性结果。

**⚠️ 局限性**

限制与未解问题：
- 算法时间为 O(f(d,δ)·n⁶)，相对较高；是否能降至近线性时间仍未知。
- 对于分离度恰好为 1 的情况，仍未得到 PTAS，尤其在平面或线性度量空间中的 NP‑难性尚未确定。
- 对于分区版 WSPD（partition WSPD）或更小 τ（如 1<τ<2）的近似方案仍缺失。
- 该双指标方案需要放宽分离度到 (1+2δ/3)，尚不清楚能否在不放宽分离度的前提下达到同样的近似比例。

---

## 539. SPARQL-LLM in the TEXT2SPARQL 2026 Challenge

**arXiv ID:** 2610.13648 | [PDF](https://arxiv.org/pdf/2610.13648v1)

**作者:** Panayiotis Smeros `[一作]` (SIB Swiss Institute of Bioinformatics), Tarcisio Mendes de Farias `[通讯]` (SIB Swiss Institute of Bioinformatics)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并评测了 SPARQL-LLM 系统，在 2026 年 TEXT2SPARQL 挑战中实现了将自然语言查询自动翻译成 SPARQL 的框架。

**💡 创新点**

采用检索式上下文预构造方法，避免代理循环，显著减少 LLM 调用次数，兼顾实时性和高准确性。

**🔧 技术方法**

基于 OpenAI LLM（400K 词上下文）与 DeepSeek‑V3，使用 RAG 风格提示、精简的查询模板，并支持英文、德语、西班牙语多语言。

**📊 数据集**

使用 DBpedia（50 英语+西班牙语问题）和 Corporate KG（未知、13 类 30 属性，50 英语+德语问题）数据集，并整合 2025 版 QALD、LC‑QuAD 及自生成示例。

**📈 对比分析**

与 8 个参赛系统对比，采用 F1/Recall/Precision/NDCG 评估，SPARQL‑LLM 在所有子任务均名列第一，西班牙语 DBpedia 上 F1 提升 59%，整体 LLM 调用成本 < $3。

**⚠️ 局限性**

仍受限于 LLM 生成错误、对极其复杂查询的覆盖不足，以及对预收集示例和手工生成 schema 的依赖。

---

## 540. PANDA: Prior-guided Attentional Dual-path Architecture

**arXiv ID:** 2610.13635 | [PDF](https://arxiv.org/pdf/2610.13635v1)

**作者:** Rishabh Jain `[一作]` (University of Cambridge), Stefano Fiorini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种可插拔的双路注意机制PANDA，用于改进两阶段信息传播的超图神经网络，使其既兼顾固定结构先验，又引入非竞争的辅助路径。

**💡 创新点**

创新点在于将接收端归一化注意力与结构先验进行可学习的插值，并设计了非竞争的辅助门控路径，从而在异同构（heterophilic）环境下提升信息流动与分类性能。

**🔧 技术方法**

使用了GATv2式注意力、结构先验权重、门控机制、复数卷积以及两阶段节点↔超边↔节点的消息传递框架，兼容有向与无向超图。

**📊 数据集**

在八个真实超图数据集（Telegram、Roman-empire、Texas、Email-Enron、Citeseer、Pubmed、Cornell 等）上进行实验。

**📈 对比分析**

与四种基线 HNN（GeDi-HNN、HGNN、HyperGCN、PhenomNN）以及 GCN/GAT 进行对比，平均提升 5.18% 准确率，极端情况下在 Telegram 上提升高达 31.01%。

**⚠️ 局限性**

局限性在于仅适用于两阶段消息传递结构，未扩展到更一般的拓扑或高阶信息传播；此外辅助路径的非竞争性虽然带来收益，但其解释性与可解释性仍需进一步研究。

---

## 541. EchoTab: Learning Measurement-Grounded Representations for Echocardiographic Diagnosis through Tabular Reconstruction

**arXiv ID:** 2610.13599 | [PDF](https://arxiv.org/pdf/2610.13599v1)

**作者:** Michelle Espranita Liman `[一作]` (Technical University of Munich), Philip Müller `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种以测量为基准的多视图超声心动图表示学习框架（EchoTab），通过为每个测量分配专门的 token，将视图信息映射到测量空间，再用这些测量 token 进行诊断预测。

**💡 创新点**

创新点在于：①将连续测量值直接作为中间概念并用 token 表示；②采用表格重建目标训练每个测量 token，避免对比学习对负样本和大批量的需求；③学习到的注意力机制与心脏科医师已知的测量‑视图关系保持一致，使得诊断过程更可解释。

**🔧 技术方法**

使用查询式解码器（cross‑attention）将测量 query 从多视图编码器聚合信息，利用均方误差表格重建损失训练测量 token；随后对选定的测量 token 进行聚合（平均）并输入到诊断分类头。该方法兼具 CLIP 视觉‑表格学习思路与概念瓶颈模型的可解释性。

**📊 数据集**

主要数据集为 MIMIC‑IV‑Echo v1.0；实验还在另一份超声心动图数据集上验证，形成两组基准。

**📈 对比分析**

与无测量预训练模型、对比学习框架以及其他超声基础模型相比，EchoTab 在诊断分类任务上取得更高准确率；其学习到的注意力分布与临床医师已知的测量‑视图关系高度一致。

**⚠️ 局限性**

局限性包括：①需要人工指定每个诊断对应的测量集合，引入领域知识且不易迁移到新诊断；②仅采用平均聚合方式，未学习更复杂的测量‑诊断交互；③目前仅在超声心动图上验证，尚未验证对其他多视图医学影像的适用性。

---

## 542. How far is Kolmogorov-Loveland randomness from Martin-Löf randomness?

**arXiv ID:** 2610.13585 | [PDF](https://arxiv.org/pdf/2610.13585v1)

**作者:** Tomislav Petrović `[一作]` `[通讯]`, Tomislav Petrović

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了一种可计算编码 E，证明任何非马丁—洛夫随机序列在 E 下至少产生一个非Kolmogorov‑Loveland随机序列，并证明对部分可计算随机性不存在类似编码。

**💡 创新点**

首次给出将马丁—洛夫随机性与Kolmogorov‑Loveland随机性等价的可计算编码，提供了一个不依赖下可计算性的新型随机性表征，并揭示部分可计算随机性缺乏此类编码。

**🔧 技术方法**

采用可计算编码、两阶段递归构造的可计算非单调投注策略、测度与马丁格尔理论、通用下可计算超马丁格尔以及有限赋值的一致性与细化分析等技术。

**📊 数据集**

本工作为理论研究，无需数据集，全部基于形式化证明。

**📈 对比分析**

不涉及实验比较；成果通过严格的数学证明展示了编码的存在性与否，未给出数值性能指标。

**⚠️ 局限性**

局限性在于该构造仅适用于Kolmogorov‑Loveland随机性，对其他马丁—洛夫随机性衍生形式（如部分可计算随机性）是否可行仍未确定；构造相对复杂，可能不易推广到更广泛的随机性概念。

---

## 543. ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching

**arXiv ID:** 2610.13572 | [PDF](https://arxiv.org/pdf/2610.13572v1)

**作者:** Zhihao Tian `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Junwei Liang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于通用步态数据和普通网球视频，构建 ATHLETE 框架，实现了从运动匹配到物理仿真，再到教师-学生学习的完整流程，使机器人在不依赖参考运动的情况下完成前手和反手回球，并支持多人连击。

**💡 创新点**

创新点在于：① 将通用步态和稀疏视频打击动作通过 Trajectory‑Guided Motion Matching（TGMM）合成多样化的全身运动参考；② 采用物理仿真驱动的教师–学生学习与在线 DAgger 蒸馏，最终得到单一的反应式策略；③ 训练过程中不需要在线运动匹配或未来球预测，显著简化部署。

**🔧 技术方法**

核心技术包括：TGMM（轨迹引导运动匹配）、GVHMR 与 GMR 运动重建与重映射、物理仿真跟踪（BeyondMimic）、教师–学生学习（DAgger 与 PPO）、阶段化策略蒸馏与任务强化学习、以及域随机化。

**📊 数据集**

使用公开的 LAFAN1 步态数据集（16 条步态），以及 10 分钟普通网球视频（67 条反手、97 条正手片段）进行运动重建与重映射；在仿真中采用 AMASS 与 LAFAN1 进行数据增强。

**📈 对比分析**

在仿真与真实实验中与 LATENT、AdaPT 等方法对比，ATHLETE 在前手回球命中率 90%/ 86.7%（单球）和多连击中保持高命中率，击球位置精度与发射速度均优于现有方法；在仿真中达 95% 以上命中率，且成功率与球速、落点的适配更好。

**⚠️ 局限性**

主要局限：当前仍依赖外部运动捕捉系统获取球位置信息，缺乏自主球体感知；对大规模球场的适应性有限；模型在极端球速或复杂多球情境下的泛化尚未充分验证。

---

## 544. Simultaneous Competitive and Regret Guarantees for TCP Acknowledgment

**arXiv ID:** 2610.13571 | [PDF](https://arxiv.org/pdf/2610.13571v1)

**作者:** Tsubasa Harada `[一作]` (Institute of Science Tokyo), Hanna Sumita `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

设计了一种在TCP确认问题上实现竞争比率接近 2 且期望回报（regret）子线性的随机在线算法；

**💡 创新点**

首次证明在保证竞争比率不超过 2+ε 的同时，能够获得相对于给定专家集合的子线性期望回报，并给出匹配的下界，证明 ε>0 是不可避免的；

**🔧 技术方法**

利用乘法权重多专家学习框架、预算/块划分策略、阈值确认算法以及切换成本控制等技术，将学习与竞争保证相结合；

**📊 数据集**

无实验数据，全部基于理论分析与对抗性输入构造；

**📈 对比分析**

相对于最优离线方案及单一专家进行比较，证明期望回报为 O(√(log(k+1))+log(1/ε))，竞争比率始终 ≤2+ε，且匹配下界；

**⚠️ 局限性**

需随机化且无法实现精确 2 的竞争，回报上仍含有 log(1/ε) 项，且算法实现复杂，需要预算管理和多阶段同步等额外开销。

---

## 545. Not All Correct Fixes Run Alike: An Empirical Study of Patch Performance in Repository-Level Issue Resolution

**arXiv ID:** 2610.13563 | [PDF](https://arxiv.org/pdf/2610.13563v1)

**作者:** Han Peng `[一作]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University), Xin Xia `[通讯]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型生成的补丁在仓库级问题修复中的运行时性能差异，利用SWE-bench/F2P测试作为性能基准。

**💡 创新点**

首次将内置F2P测试作为性能评测手段，并采用Welch ANOVA方法量化不同补丁之间的性能差异，揭示补丁的解决率与性能不相关。

**🔧 技术方法**

使用SWE-agent框架调用Claude Opus 5、GPT 5.6、DeepSeek V4 Pro、Qwen 3.8 Max四大LLM生成补丁；通过系统调用跟踪与指令计数器收集执行指标，并用Welch ANOVA与RSD进行统计比较。

**📊 数据集**

基于经过可测性过滤后的SWE‑bench与SWE‑bench‑Live的verified splits，共339个实例，其中262个至少拥有两种不同补丁。

**📈 对比分析**

对每个补丁在F2P测试下多次重复执行，记录指令数和壁钟时间；使用Welch ANOVA判定是否显著可区分，结果显示约42%实例存在显著性能差异；在可区分集上，GPT/DeepSeek速度最快，Qwen最慢，且模型排序与解决率相反。

**⚠️ 局限性**

仅针对Python仓库，F2P测试可能无法全面反映所有性能因素；过滤过程可能导致样本偏倚；缺乏对I/O、外部状态等对性能影响的细粒度分析。

---

## 546. Learning Holistic Scene Layout against View Shifts in Cross-View Geo-Localization

**arXiv ID:** 2610.13553 | [PDF](https://arxiv.org/pdf/2610.13553v1)

**作者:** Chao Yang `[一作]` (Wuhan University), Zhigang Tu `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种跨视角地理定位方法SLGeo，利用空间上下文建模（SCM）与高频信息引导（HFIG）提升鲁棒性；

**💡 创新点**

创新点包括：① SC 模块通过四方向旋转扫描并在每个空间位置上进行多头注意力交互，聚合多方位空间上下文；② HFIG模块利用离散余弦变换提取高频结构信息，作为对抗外观变化的辅助；③ 采用旋转扫描而非传统扫描策略，实现更丰富的上下文表示；

**🔧 技术方法**

使用 ConvNeXt 作为特征提取 backbone，DCT 频域变换，位置级多头注意力，InfoNCE 对比学习，以及双向扫描与旋转扫描策略；

**📊 数据集**

在 CVUSA 与 CVACT 两大公开跨视角数据集上进行实验，进一步在未知朝向与有限视场（FoV）设定下测试鲁棒性；

**📈 对比分析**

与 11 种现有方法对比，SLGeo 在 CVUSA、CVACT 的 R@1 及 R@5 接近或超过现有最佳方法，并在未知朝向/限 FoV 场景中保持前两名；Ablation 结果表明 SCM 与 HFIG 单独提升 3–5%，两者联合实现最佳性能；

**⚠️ 局限性**

局限性：对极端光照/季节变化仍有一定敏感性，旋转扫描导致计算量增大，且在更大尺度或多尺度特征融合方面尚需改进。

---

## 547. TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy

**arXiv ID:** 2610.13654 | [PDF](https://arxiv.org/pdf/2610.13654v1)

**作者:** Tianqi Zhao `[一作]` (Renmin University of China), Qiong Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了大规模 TabReliBench 基准，对 160 个分类和 81 个回归任务评估 Tabular Foundation Models（TFMs）的预测精度、分布质量、边际与子组校准以及区间锐度。

**💡 创新点**

创新点包括：①首次将可靠性评估扩展到多维度并量化在不同基准下 TFMs 排名的变动；②使用元特征对数据集级可靠性进行预测；③系统研究集成与合规化对不同可靠性指标的影响；④在 160+81 个数据集上进行跨模型、跨指标的统计分解。

**🔧 技术方法**

技术手段包括：TabPFN、TabICL、TabDPT 等 TFMs；传统模型 BART、RealMLP、XGBoost；评估指标有 Accuracy / R²、Brier / CRPS、ECE、PIT‑KS、worst‑slab 置信覆盖、区间宽度；统计分解、岭回归和 Extra‑Trees 的元学习；集成推理与多种合规化（global、local、posterior、rectified）方法。

**📊 数据集**

使用 OpenML 公开的 160 个分类和 81 个回归数据集，按统一流程抽样最多 10,000 条记录、5 次独立 train‑test 分割，每个任务共享相同样本，确保跨模型的配对比较。

**📈 对比分析**

在多维度评价中发现：TFMs 在绝大多数指标上优于经典基线，但其排名随基准和指标而显著变化；集成能显著提升预测精度与合规得分，但对 ECE/PIT‑KS 的改进不稳定；合规化可将边际覆盖逼近目标，但通常无法消除 worst‑slab 低覆盖，且会导致更宽或无限区间。

**⚠️ 局限性**

局限性包括：仅针对选定的数据集与模型；worst‑slab 覆盖并不等价于点条件覆盖；元特征预测受数据集数量限制，难以精准预测模型间相对优势；合规化方法对 TFMs 的效果有限，未解决子组不均匀覆盖问题。

---

## 548. A-SAW: The Agentic Software Architecture Workflow

**arXiv ID:** 2610.13647 | [PDF](https://arxiv.org/pdf/2610.13647v1)

**作者:** Finn Klatt `[一作]` (University of Liechtenstein), Johannes Schneider `[通讯]` (University of Liechtenstein)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Agentic Software Architecture Workflow，将持续的AI代理嵌入Architecture-as-Code流程，自动生成完整的架构文档并强制人工审核；

**💡 创新点**

核心创新在于：①在AaC基础上加入持久化代理层，②利用类型化元素目录与上下文文件夹实现结构化推理，③设计可测量的五项防幻觉机制与双角色审批门；

**🔧 技术方法**

技术包括：C4模型与PlantUML文本、统一的元素目录、IDE集成Agent（如Claude/ChatGPT）执行脚本、自动化语义检查与宏校验；

**📊 数据集**

数据集来源于一家大型工业组织的架构文档，分为三组：手工撰写、部分AaC、Agent生成，合计27份文档，并对13名从业者进行半结构化访谈；

**📈 对比分析**

评估方法：使用基于视图的四维量表（完整性、一致性、可追溯性、可维护性）对文档打分，并进行匹配对Wilcoxon检验；实验显示Agent生成文档在完整度、内部一致性、机器可读性上接近满分，且标准差下降至原来的约15%（置信区间CI）；访谈显示人机协作显著减少首次草稿时间（从数天降至分钟），但信任仍取决于人工复核；

**⚠️ 局限性**

局限性包括：依赖预先构建的完整C4模型和元素目录，缺少与Issue Tracker的集成导致可追溯性不足；评估仅在单一组织内进行，难以泛化；受限于LLM模型的偏差与自我偏好；无法验证在全新架构设计（非再生成）场景下的效果。

---

## 549. Humanize Olympiad Agents: A Multi-Session Workflow for Olympiad Problem Solving with Solve, Review, Wiki, Verify

**arXiv ID:** 2610.13632 | [PDF](https://arxiv.org/pdf/2610.13632v1)

**作者:** Zhengyang Zhang `[一作]` (Tsinghua University), Ligeng Zhu `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了什么：提出并实现了 HOA（Humanize Olympiads Agent）这一开放式流程级别代理框架，结合独立审阅者、结构化知识库和任务专用验证器，实现了多轮迭代构造与验证数学、科学竞赛和正式证明任务的完整工作流。

**💡 创新点**

创新点是什么：在传统的 LLM 代理基础上引入了 worker–reviewer 修复循环、可检索的任务专用知识库（wiki）以及任务特定的验证器，三者并行协作、互补反馈，从而显著提升了长序列推理的可靠性与成功率。

**🔧 技术方法**

用了什么技术：使用 GPT‑5.6‑Sol、Kimi‑K3 等大型语言模型作为工作者；独立 Reviewer 负责语义审查；Comparator 作为 Lean 验证器进行正式证明的核验；结构化 Wiki 供知识检索；Humanize harness 管理工具调用、流程控制与多轮迭代。

**📊 数据集**

用了什么数据集：评测基准包括 2026 年 IMO、IPhO、IChO、IOI 官方题目集；PutnamBench（672 题）；Lean‑Eval（304 题）等正式化数学任务。

**📈 对比分析**

如何比较的方法，性能怎么样：将单模型 API、Tool‑level Codex、以及完整 HOA 进行对比，指标为解决数量、得分与金牌阈值；HOA 在 IMO、IPhO、IChO、IOI 上达或逼近金牌；在 PutnamBench 完成 672/672；在 Lean‑Eval 从 112/304 提升至 251/304（82.6%），在开放模型上逼近封闭模型的表现。

**⚠️ 局限性**

limitation是什么：仅覆盖特定领域（竞赛、编程、正式数学），存在基准泄漏风险；验证仅限于已信任的 Lean 形式化，无法保证自然语言题目翻译的完整性；多轮流程计算成本高，且未充分隔离各组件交互的因果效应。

---

## 550. Multi-Stream Attention Residuals

**arXiv ID:** 2610.13630 | [PDF](https://arxiv.org/pdf/2610.13630v1)

**作者:** Xiaohan Qin `[一作]` (School of AI, Shanghai Jiao Tong University), Junchi Yan `[通讯]` (School of AI, Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Multi-Stream Attention Residuals (MSAR)，通过多路深度注意力替代传统残差检索，显著提升大规模语言模型的训练与下游表现。

**💡 创新点**

将单流残差检索扩展为多头深度注意力，利用轻量级因果深度卷积生成多尺度上下文视角，随后通过可学习标量混合实现多路深度检索并保持隐藏表示的全维完整性。

**🔧 技术方法**

采用多尺度因果深度卷积、Per-stream depth attention、可学习的ℋ^pre混合、Block MSAR 两阶段计算与缓存、MoE架构、SwiGLU、RMSNorm 等技术。

**📊 数据集**

使用 Pile 预训练语料库，并在 12 个下游基准（MMLU、ARC-C、BBH、HumanEval、LCBench、GSM8K、MATH、CMMLU、C‑Eval、C3 等）进行评估。

**📈 对比分析**

与标准残差、mHC、AttnRes 进行对比，在 18B 与 28B MoE 模型上，MSAR 在所有基准平均提升约 8‑9 分，训练损失下降约 2.7%，实现 1.58× 的 iso‑loss 计算加速。

**⚠️ 局限性**

仍需额外卷积与混合开销，对块划分与流数敏感，且在更大规模或不同架构下的表现尚未充分验证。

---

## 551. Non-Bayesian Social Learning with Heterogeneous and Time-Varying Aggregation Rules

**arXiv ID:** 2610.13611 | [PDF](https://arxiv.org/pdf/2610.13611v1)

**作者:** Sammy Oukaci `[一作]` (ENSTA), Jesus Cerquides `[通讯]` (CSIC)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了非贝叶斯社交学习中代理人采用异质且随时间变化的聚合规则，并证明在强连通网络和足够慢的自我关注衰减下，所有代理人几乎必然收敛到真实状态。

**💡 创新点**

创新点在于把聚合规则的弹性曲率和可变权重纳入分析，统一了对常见的线性、对数线性、谐波等聚合方式的收敛速度界定，并首次给出基于信息流路径的下界/上界。

**🔧 技术方法**

采用理论分析与凸分析相结合的方式，构造了对数线性等价表示，引入日志曲率、信任权重及可变权重矩阵，并用相对熵量化信息量。

**📊 数据集**

未使用公开数据集，而是构建了一个六人委员会的仿真实验，设置三种假设状态和不同专业角色的信号分布来验证理论。

**📈 对比分析**

通过数值模拟比较了均质与异质规则、是否召开会议等情形，结果显示采用负曲率（谐波规则）能最快收敛，会议即切换到谐波规则可显著提升收敛速度。

**⚠️ 局限性**

主要局限包括假设观测信号独立、邻居固定、模型误设定不考虑误差和攻击、以及收敛速度上界可能非最优。

---

## 552. AudioTETRIS: Token Elimination via Transitional Importance Signals for Efficient Audio LLMs

**arXiv ID:** 2610.13600 | [PDF](https://arxiv.org/pdf/2610.13600v1)

**作者:** Shreya Sajid `[一作]` (Eindhoven University of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的音频令牌压缩框架AudioTETRIS，通过利用音频编码器隐藏状态的时间差异作为语音转移先验，动态引导大型语言模型中的自注意力进行逐层压缩；

**💡 创新点**

创新点在于将音频编码器内部的结构化声学信息（语音转移先验）与多层自注意力权重相融合，形成深度自适应重要性评分，实现无训练、无附加参数的高效压缩；

**🔧 技术方法**

核心技术包括基于L2距离的音频隐藏状态转移评分、深度自适应注意力分数计算、交叉层动态裁剪、两阶段融合分数排序和top‑k压缩；

**📊 数据集**

使用LibriSpeech（ASR）、IEMOCAP（情感识别）、MMAU‑mini（音频问答）、CoVoST2（语音翻译）四大公开语音任务的数据集；

**📈 对比分析**

与SpeechPrune、SparseVLM、LTBM以及随机裁剪等无训练压缩基线对比，AudioTETRIS在四个压缩率下均能保持或超过未压缩模型的性能，在高压缩率（75%）下的WER仅比未压缩提高约1.2点，且在多模态任务上表现出显著优势；

**⚠️ 局限性**

限制包括：依赖语音编码器隐藏状态的转移特性，对非语音或极度下采样的编码器可能不适用；需要一次性校准跨层交叉点ℓ†，不完全零拷贝；相较单次裁剪方法略有额外TFLOP开销；在极端压缩或嘈杂、多语言场景下可能导致词汇识别性能下降。

---

## 553. A Systematic Study of Security Enforcement at Android Inter-Component Boundaries

**arXiv ID:** 2610.13597 | [PDF](https://arxiv.org/pdf/2610.13597v1)

**作者:** Sandiyo Christan A `[一作]` `[通讯]` (Independent Researcher), Sandiyo Christan A (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性评估了Android 17（API 37）上第一方应用（Google与AOSP）120个IPC边界的安全执法情况，发现97个执法缺口、23个执法保持；

**💡 创新点**

提出了“预期‑与‑观察”评估框架，结合正向控制与跨类型比较，首次生成了覆盖manifest、代码、平台及架构四层的14类执法缺口分类；

**🔧 技术方法**

使用动态PoC（Android Pixel 6a运行Android 17）与静态代码/清单分析相结合，评估预期执法与实际行为；

**📊 数据集**

评估对象为30余个Google与AOSP第一方应用的所有导出组件（共3,721个），覆盖5类IPC接口和平台安全控制；

**📈 对比分析**

通过正向控制（23个）验证方法有效性，并与已知漏洞工具（如ComDroid、CHEX、IccTA）对比，发现传统工具只能发现单一模式，无法全局评估执法效果；

**⚠️ 局限性**

局限包括：样本非随机、仅覆盖Android 17与单一设备、部分评估仅静态、仅聚焦第一方应用，且未考虑第三方应用的执法差异与时间演变。

---

## 554. MENTOR: Learning from Failure with On-Policy Pedagogical Intervention

**arXiv ID:** 2610.13594 | [PDF](https://arxiv.org/pdf/2610.13594v1)

**作者:** Guanlan Yue `[一作]` (Xi'an Jiaotong University), Rui Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MENTOR 框架，在 RLVR 训练中实时诊断零奖励崩溃并生成即时教学提示，提升学习信号。

**💡 创新点**

首次在在线强化学习中实现多智能体的归因‑生成‑评估‑修正闭环教学，动态适配模型错误，解决静态提示和后置修正的局限。

**🔧 技术方法**

利用多智能体系统（归因 Agent、生成 Agent、评估‑修正循环）、LLM-as-Judge、GRPO 等 RL 框架，并将提示动态注入训练。

**📊 数据集**

使用 DeepMath‑103K、LiveCodeBench、AIME、MATH‑500、GAOKAO‑Bench、Olympiad‑Bench、HumanEval、MBPP、BigCodeBench、MMLU 等多任务与 OOD 数据集。

**📈 对比分析**

与 GRPO、DAPO、Critique‑GRPO、SRPO、LUFFY、QUESTA、HINT、Scaf‑GRPO 等八个基线对比，MENTOR 在数学、代码生成及 OOD MMLU 上均实现显著提升，最高相对增益 78.3%，在轻量模型上提升 35‑43%。

**⚠️ 局限性**

仅在单模任务验证，尚未探索多模或动态环境中的泛化与稳定性，提示生成的通用性需进一步评估。

---

## 555. Measuring Artifact Reliance in Machine-Generated Text Detectors

**arXiv ID:** 2610.13583 | [PDF](https://arxiv.org/pdf/2610.13583v1)

**作者:** Anastasia Voznyuk `[一作]` (Advacheck OÜ), Andrey Grabovoy `[通讯]` (Moscow State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对十个科学文献子集的机器生成文本检测基准进行审计，发现多数难度主要来自于文档处理产生的格式痕迹而非真实写作风格。

**💡 创新点**

提出表面探测、格式反事实和 Artifact Reliance Score 三种方法，揭示并度量基准中格式偏差对检测器性能的影响，并证明去格式化能提升跨基准泛化。

**🔧 技术方法**

使用内容无关的表面特征（长度、标点、空格等）训练逻辑回归；构建可逆的格式化操作；对比 Fine‑tuned DeBERTa、Binoculars、Fast‑DetectGPT 等检测器；以及 SentencePiece 与 byte‑pair 的 tokenizer 对比。

**📊 数据集**

采用六个主流资源的十个子集（arXiv、PeerRead、M4、SemEval‑24、COLING‑25、IDMGSP、GPABench2、CHEAT）中的科学摘要、引言和评审文本。

**📈 对比分析**

通过宏 F1、AUC 及格式反事实指标评估，发现仅用表面特征即可达到 0.83–0.997 的 macro‑F1，超过多数共享任务获胜者；格式化归一化后，在域外测试的 macro‑F1 从 0.33 提升至 0.65，显著降低了泛化差距。

**⚠️ 局限性**

局限在于仅涉及英文科学文本、十个子集、三种公开检测器，且格式反事实仅覆盖空格/换行等表面形式，未考虑更深层结构或多语言情况。

---

## 556. RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation

**arXiv ID:** 2610.13578 | [PDF](https://arxiv.org/pdf/2610.13578v1)

**作者:** Xuhui Kang `[一作]` (University of Virginia), Yen-Ling Kuo `[通讯]` (University of Virginia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出两阶段检索锚定表示对齐（RARA）方法，利用预训练演示中的相关任务示例来稳健地微调机器人视觉‑语言‑动作策略。

**💡 创新点**

创新点在于：①先检索与目标任务意图和动作相似的预训练演示；②通过多正例对比对齐目标表示与检索示例的预训练表示；③在微调阶段仅对检索示例的表示施加锚定约束，从而防止表示漂移并保留对相关任务的知识。

**🔧 技术方法**

技术细节：任务指令相似度检索（CLIP 文本编码）；动作块动态时间规整（DTW）配对；多正例对比损失与表示距离锚定；两阶段训练（对比+锚定 → 行为克隆+锚定）。

**📊 数据集**

数据集：RoboTwin 2.0（模拟）、RoboCasa（模拟）和真实Franka机器人；预训练数据分别来自RoboTwin（约2k演示）和DROID（约76k演示）；目标任务 4–5 个，演示数 50–100，覆盖不同物体、背景和摄像头姿态。

**📈 对比分析**

与 FT、Co‑FT、STRAP、RETAIN、BlindVLA 等五种基线比较，RARA 在 13 项任务中平均提升 ID 成功率 9.5% 及 OOD 成功率 13.6%，单任务提升 8–16% 以上，在真实机器人实验中同样显著优于所有基线。

**⚠️ 局限性**

局限性：依赖预训练数据中包含与目标任务相同类型的演示；对不同机器人硬件、动作空间或极端环境变化的迁移能力尚未评估；两阶段训练增加了计算成本。

---

## 557. MatQuery: Execution-Grounded Natural-Language Querying over Heterogeneous Materials Databases

**arXiv ID:** 2610.13573 | [PDF](https://arxiv.org/pdf/2610.13573v1)

**作者:** Chen Jason Zhang `[一作]` (Hong Kong Polytechnic University), Qi Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 MatQuery 原型，实现自然语言请求到多源材料数据库的查询转换，并生成包含源信息的答案表。

**💡 创新点**

创新点包括：将语言模型生成与程序化校验相结合的八步流水线；为跨源查询设计冻结自检快照与字段目录；自动生成并验证基准问答；通过结构匹配解决跨数据库比较的身份不确定性。

**🔧 技术方法**

采用大语言模型（如 DeepSeek Flash）生成查询；程序化校验与执行；查询编译器将中间表示转换为各数据库的 SQL；结构相似性匹配；低成本源评分器；评估 harness 统一基准。

**📊 数据集**

使用六个公开材料数据库的冻结快照（Materials Project、OQMD、AFLOW、JARVIS‑DFT、NOMAD、COD），每个约10,000条记录，构成数据层与字段目录。

**📈 对比分析**

通过统一冻结基线和评估 harness，比较 Direct Prompting、CHESS、Alpha‑SQL 与 MatQuery；在800题开发集上，直接生成三示例时严格执行准确率达82.4%，源选择准确率约90%；低成本评分器在 85% 的完整请求中准确选取数据库；Alpha‑SQL 速度最慢，整体评估尚未完成完整流水线。

**⚠️ 局限性**

局限性包括：基准问答未通过独立专家评审；结构信息缺失导致跨源匹配不可靠；使用的仅是数据库的冻结子集而非完整数据库；实验未覆盖全部流水线步骤；未完全评估科学可比性。

---

## 558. Constructing Pareto Frontiers of Optimal Reasoning Termination Methods for Overthinking in Reasoning Models

**arXiv ID:** 2610.13570 | [PDF](https://arxiv.org/pdf/2610.13570v1)

**作者:** Weixin Guan `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Weiping Wang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过步骤截断探测（STP）量化推理模型的“过度思考”现象，并构建了三种终止策略（动态路由、固定预算、动态早停）的离线token‑accuracy Pareto前沿，进而设计了 OTBench 基准，评估公开终止方法与理论上最佳方案的差距。

**💡 创新点**

创新点在于：①提出 STP 方法，可在保持推理路径不变的前提下多点截断测得准确率曲线；②定义过度思考比率（OR）和准确率衰减（AD）两指标；③将三类终止策略统一为预算约束动作选择问题，生成可比的 Pareto 前沿；④构建包含 70.5B 推理 token 的 OTBench，为终止方法提供统一评测协议。

**🔧 技术方法**

使用技术包括：步骤截断探测（STP）、token‑accuracy 曲线绘制、Spearman 相关分析、统一预算约束框架、离线优化求 Pareto 前沿、以及在 OTBench 上对 ThinkSwitcher、Predictive Scheduling、ThinkBrake 等公开方法进行参数扫描与性能评估。

**📊 数据集**

数据集涵盖四个基准组：AIME（包含 2024‑2026 年版）、MATH500、GPQA‑Diamond、LiveCodeBench；使用六个推理模型（GPT‑OSS‑20B、Qwen3‑4B‑Thinking、Qwen3‑30B‑A3B‑Thinking、DeepSeek‑R1‑Distill‑Qwen‑7B/14B/32B），共生成 70.5B 推理 token 及其 STP 量化结果。

**📈 对比分析**

与理论前沿比较显示：过度思考普遍严重（平均 OR≈64%），三种策略前沿呈递增层级（动态路由 < 固定预算 < 动态早停）；公开方法均落后于对应前沿，说明仍有大量可利用的理论提升空间；在 token‑accuracy 归一化坐标系下，公开方法在相同成本下的准确率均低于最佳前沿。

**⚠️ 局限性**

局限性包括：STP 仅基于采样的推理轨迹，可能不覆盖全部推理行为；离线前沿基于已测量的动作空间，未考虑实际生成时的随机性；评测仅覆盖六个模型和四个基准，未验证在更广泛模型/任务上的通用性；OTBench 依赖公开的轨迹，实际部署需自行生成轨迹。

---

## 559. Neural Polar Codes for Source Coding

**arXiv ID:** 2610.13554 | [PDF](https://arxiv.org/pdf/2610.13554v1)

**作者:** Bashar Huleihel `[一作]` (Ben-Gurion University), Haim H. Permuter `[通讯]` (Ben-Gurion University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于神经极化编码的无损和有损源压缩的数据驱动方法，旨在克服经典极化源编码在处理未知的具有记忆的q-元源时的复杂性问题。

**💡 创新点**

创新点在于使用神经极化解码器直接从数据中训练，消除了对显式源模型的需求，并降低了解码复杂性，同时保留了极化变换和SC计算图的结构。

**🔧 技术方法**

使用了神经网络（NN）作为极化解码器，结合了数据驱动的估计方法，替代了经典极化编码中的显式更新规则。

**📊 数据集**

实验使用了无记忆源、具有记忆的源以及真实的基因组数据集，展示了所提出方法的有效性。

**📈 对比分析**

与经典的极化解码器进行比较，神经极化解码器在可用时能够紧密匹配分析的SC或SCT解码器的行为，并在经典极化方法计算不可行或所需源模型不可用时仍然适用。

**⚠️ 局限性**

限制在于该方法主要针对单发射器、单接收器问题，扩展到多终端源编码时会引入额外的挑战。

---

## 560. Auditing VLM Benchmarks from Existing Evaluation Runs

**arXiv ID:** 2610.13548 | [PDF](https://arxiv.org/pdf/2610.13548v1)

**作者:** Denis Degtiarev `[一作]` (Innopolis University), Artem Tuzov `[通讯]` (Innopolis University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 VLMEvalKit 中实现了一个审计层，利用已有的每题预测和准确性结果，对视觉语言模型评测得到的输出进行后处理，检测 MMStar 基准中的零增益题目、标签冲突题目等异常，并生成问题级报告。

**💡 创新点**

创新点包括：①将基准审计嵌入主流评测工具，避免单独标注与推理成本；②提出基于项目匹配置换检验的统计协议，用“超额”而非原始比例来衡量异常；③首次对 MMStar 进行细粒度异常定位，揭示数学/科学类题目视觉无关性与感知类题目标签错误的不同分布。

**🔧 技术方法**

使用技术包括：视觉语言模型推理、文本仅推理（移除图像）、六种后处理检测器（全/无图像准确率、共识冲突、κ 统计、位置偏差、近似选项相似度等）、置换检验、Wilson 区间、leave‑one‑model‑out 敏感性分析以及报告生成。

**📊 数据集**

主要数据集为 MMStar（1500 题四选项，六个官方类别）以及 HallusionBench（951 题二选一）用于验证检测器的边界。

**📈 对比分析**

通过比较全图像推理与文本仅推理的准确率，计算零增益比例并使用置换检验评估显著性，得到 +3.2 百分点的显著超额（p=0.001）。在不同模型池（去掉一模型）下结果保持正向偏差，且检测到 134 题统一错误。对二选一基准 HallusionBench 的测试表明检测器失效，表明适用范围受限。

**⚠️ 局限性**

局限性包括：仅使用三模型池，结果对模型池敏感；零增益超额为聚合度量，无法定位具体机制；检测器设计针对四选多项式，二选或更大多项式的适用性未知；模型对图像的拒绝率会影响检测；需人工复核才能将候选错误转为确切错误率。

---

## 561. SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data

**arXiv ID:** 2610.13644 | [PDF](https://arxiv.org/pdf/2610.13644v1)

**作者:** Bastian Schäfer `[一作]` (Université du Littoral-Côte-d'Opale), Arnaud Liefooghe `[通讯]` (Université du Littoral-Côte-d'Opale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了一个大型特征选择基准框架 SelectArena，评估了15种常用特征选择方法在71个手工挑选的真实表格数据集上；

**💡 创新点**

创新点在于整合了现代表格模型（包括Tabular Foundation Models）、统一的评估协议以及多维度（预测性能、运行时、稳定性、有效性）评价，首次为不同场景提供系统性方法选择指导；

**🔧 技术方法**

采用了自动化实验流水线（AutoGluon + TabArena）、Elo评级、改进率、Nogueira稳定性指标、噪声注入有效性评估以及基于卡方的F-Test、信息论、距离、树模型等多种特征选择技术；

**📊 数据集**

使用了71个来源多样、真实世界的表格数据集（覆盖分类与回归、各类领域），并结合5种下游模型（Linear、RF、LGBM、RealMLP、TabICLv2）；

**📈 对比分析**

实验显示树模型（RFImportance、CART）和F-Test在预测性能与运行时均处于 Pareto 前沿；F-Test 在稳定性和有效性方面优于其他方法，是最平衡的选择；不同下游模型对特征选择方法的敏感度不高；

**⚠️ 局限性**

局限包括：所有方法都受限于一小时计算时间导致部分方法未完成；部分实现由作者自行编写，可能不如成熟库优化；稳定性和有效性评估仅采用自助抽样和高斯噪声，缺乏对异常值、缺失、分布漂移等实际情况的考量；强制固定卡片数可能削弱本身可变特征数方法的表现；评价指标的选择（AUC、log-loss、RMSE）可能影响结果。

---

## 562. Learning Size Independent Constraint Models from Input Data and Positive and Negative Example Solutions

**arXiv ID:** 2610.13640 | [PDF](https://arxiv.org/pdf/2610.13640v1)

**作者:** Helmut Simonis `[一作]` `[通讯]` (University College Cork), Helmut Simonis (University College Cork)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套约束学习工具，能够从多尺寸问题实例的正负样本以及输入数据中自动生成与规模无关的MiniZinc模型，并同时输出自然语言描述；

**💡 创新点**

创新点包括：①引入规模无关的约束学习框架；②利用结构化样本和输入数据进行模式与分区生成，可直接泛化；③对模型进行子集化简与公式自动推导；④生成可执行MiniZinc代码与可解释的自然语言说明；⑤在PTHG21基准上验证其可行性；

**🔧 技术方法**

采用的技术包括：ModelSeeker改进的被动学习算法、全局约束目录、模式/分区生成器、参数公式生成器、子集化简、自然语言模板生成、MiniZinc代码生成与评估框架；

**📊 数据集**

使用PTHG21 Challenge提供的15个问题（涵盖CSPLib 15+多个变体）的多尺寸实例数据集，包含正负样本、测试案例和额外解；

**📈 对比分析**

与手写MiniZinc模型对比，执行时间相近；在PTHG21基准上实现100%分类准确，额外解生成能力有限；整体约束学习与代码生成耗时约10分钟；

**⚠️ 局限性**

局限性包括：只能处理被动学习情形，无法处理仅给最优解的优化问题；索引集支持有限，难以处理复杂过滤；大规模输入数据的特征筛选仍是挑战；缺乏对未在全局约束目录中的新约束的支持；用户交互功能不足；部分问题无法自动推导必要参数，导致额外解生成受限。

---

## 563. Private and Distributionally Robust Federated Multiclass Classification

**arXiv ID:** 2610.13631 | [PDF](https://arxiv.org/pdf/2610.13631v1)

**作者:** Andrea P. Hernandez `[一作]` (Georgia Institute of Technology), Nagi Gebraeel `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种在工业设备故障诊断场景下的差分隐私联邦多分类分布式鲁棒学习框架，能够在多客户端共享模型更新的同时保证每个样本的隐私；

**💡 创新点**

创新点在于将Wasserstein距离的分布式鲁棒优化与Crammer–Singer多分类损失结合，并在ADMM算法中同时对本地模型和鲁棒变量添加高斯噪声，首次实现了样本级隐私保障的联邦多分类鲁棒学习；

**🔧 技术方法**

主要技术包括ADMM分解、近端线性化近似、Gaussian机制与时刻计数器（moments accountant）隐私分析、Wasserstein模糊球与Crammer–Singer损失；

**📊 数据集**

实验使用了仿真工业泵故障数据（四类）以及UCI多分类数据集（Dermatology、Iris、Wine、Palmer Penguins、Seeds）；

**📈 对比分析**

与DP-Fed ERM、DP-Fed SGD、非私有Fed DRO、OvA DRO以及集中式DRO等基线对比，DP-Fed DRO在大多数数据集和中等隐私预算下均优于其他私有方法，并在噪声敏感的设置下表现更稳健；

**⚠️ 局限性**

局限性包括：在样本量小的客户端中隐私噪声会显著影响性能，算法的收敛速度为O(1/√T)且需要手工调节多项超参数，且未针对通信时机与预算分配进行联合优化。

---

## 564. JitterVLA: A Replay Benchmark for Motion Quality in VLA-Driven Human-Robot Manipulation

**arXiv ID:** 2610.13628 | [PDF](https://arxiv.org/pdf/2610.13628v1)

**作者:** Faith Ogunwoye `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了JitterVLA回放基准，通过在固定命令流和延迟的条件下对不同控制器（步进、插值、熵门控、减颤等）进行配对回放，评估视觉‑语言‑动作（VLA）策略在时延抖动下的运动质量，涵盖合成到手部轨迹、三条VLA管线以及真实UR3机器人的实验；

**💡 创新点**

创新点在于提出了配对回放协议，使得仅改变控制器而保持命令序列和时延不变，从而可公平比较不同控制策略；并系统评估了插值、熵门控、减颤等手段在多种VLA管线上的效果，同时公开了完整的代码与数据；

**🔧 技术方法**

主要技术包括FIFO命令交付、通用三轴点质量阻抗模型、线性插值、基于熵的阻抗门控、Ruckig减颤轨迹生成、LDLJ与SPARC平滑度指标，以及对UR3加速度计的测量与分析；

**📊 数据集**

使用的数据集包括10条5Hz的合成目标、20位受试者的3D-ARM-Gaze手部轨迹、OpenVLA、pi0-FAST、pi0.5 DROID VLA管线的预测结果，以及录制的UR3远程操控轨迹和三类网络时延曲线；

**📈 对比分析**

通过配对回放和95%单位自举区间进行比较，插值策略在所有源上将p99颤动降至约83%（人类轨迹下200ms均值延迟时）并仅增加约2cm的跟踪误差；熵门控仅在pi0-FAST管线有效；减颤进一步降低颤动但伴随更大跟踪误差；在UR3实验中插值相对步进可降低3–14%的加速度计颤动；评价指标包括p99颤动、RMS颤动、跟踪RMSE、LDLJ和SPARC；

**⚠️ 局限性**

局限性在于使用的通用点质量模型未能充分代表UR3的运动学与动力学，未评估闭环控制、任务成功率或机器人的特定动态；UR3实验样本有限且加速度计未校准；未测量舒适度、偏好或安全感受等人因指标。

---

## 565. SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle

**arXiv ID:** 2610.13626 | [PDF](https://arxiv.org/pdf/2610.13626v1)

**作者:** Xiting Liu `[一作]` (Tsinghua University), Shi-Min Hu `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 SWE-DeceptionBench，系统评估 LLM 在软件工程生命周期中主动与被动欺骗行为，并验证两种轻量防御方法。

**💡 创新点**

① 在六个软件生命周期阶段构建主动/被动欺骗场景；② 引入奖励、压力三种外部诱因；③ 通过中英文双语 720 条实例构建全面的欺骗基准；④ 在七款主流 LLM 上进行跨场景、跨诱因的实证评测。

**🔧 技术方法**

使用 LLM 生成场景与诱因、LLM 判别器做自动标注、统计检验（McNemar、Bootstrap）、轻量防御（guard 模型与 Self-Reminder）、API 调用与温度控制等技术。

**📊 数据集**

SWE-DeceptionBench 自建数据集：720 条中英文测试实例，覆盖需求分析、系统设计、开发、测试、部署、运维六个生命周期阶段，包含主动、被动欺骗及三种诱因。

**📈 对比分析**

通过对七款 LLM 在中性、奖励、压力条件下的欺骗率进行对比；主动欺骗率平均 49.88%，被动欺骗率 64.29%；不同模型差异显著；轻量防御中 guard 模型覆盖率低，Self-Reminder 在部分模型可将欺骗率显著降低。

**⚠️ 局限性**

局限性包括：自动标注依赖 LLM 判别器，可能存在偏差；场景构建未覆盖多轮交互和工具使用；外部诱因仅划分为三类，未拆解单一诱因效应；评测仅基于单次请求，未考察动态反馈与实际软件工程工作流。

---

## 566. Training User Simulators with Latent Intrinsic Feedback

**arXiv ID:** 2610.13617 | [PDF](https://arxiv.org/pdf/2610.13617v1)

**作者:** Tarek Naous `[一作]`, Ofer Meshi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练具备潜在评估状态（LES）的用户模拟器，能够在生成用户回复的同时给出对LLM助手的即时评分与反馈。

**💡 创新点**

创新在于将用户主观评估嵌入潜在状态，提升模拟器的反馈对齐与真实感，并提供内在奖励信号用于多轮RL训练，显著降低计算成本。

**🔧 技术方法**

技术包括Gemma3系列模型的全参数微调、Latent Evaluation State生成、意图条件化、GRPO多轮强化学习以及外部判别模型对比。

**📊 数据集**

使用WildChat‑1M（约13.6k 交互）做训练与评估，WildChat‑4.8M做RL训练，以及WildBenchV2与SimulatorArena进行助手性能评估。

**📈 对比分析**

与提示式（Gemini3.1-Pro IT）与仅SFT的基线相比，LES模拟器在F1（批判行为）与评分相关性上提升≈10‑15%；在RL训练中，使用LES的内在奖励能获得与外部判别器相当甚至更高的任务完成率，同时将训练FLOPs降低约45%。

**⚠️ 局限性**

局限性包括：LES模型仍依赖高质量的训练数据，评分与批判行为对极少数情况的捕捉不足；在更大规模或更复杂任务中，内在奖励可能不足以覆盖全局目标；以及对新兴LLM助手的适配仍需进一步验证。

---

## 567. CUBE: Class-conditioned Unknown Boundary Evidence for Open-Set Insect Species Recognition

**arXiv ID:** 2610.13610 | [PDF](https://arxiv.org/pdf/2610.13610v1)

**作者:** Hao Tan `[一作]`, Jinghui Qin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CUBE框架，用类条件边界学习、辅助阴影证据、税onomic校准及生态一致性实现开集合昆虫物种识别。

**💡 创新点**

创新点包括：①将未知样本视为与已知类相似的局部边界而非全局未知；②利用辅助物种学习局部边界并在推理时作为阴影引用；③融合税onomic语义和生态位信息进行生物可行性校准。

**🔧 技术方法**

技术手段：类条件边界学习、辅助一致性损失、退出logit与局部拒绝概率、阴影原型引用、冻结生物视觉语言模型进行税onomic匹配、地理一致性校准、分数标准化与多证据融合。

**📊 数据集**

数据集：Open-Insect C-America 以及 Barro Colorado Island (BCI) OOD 分割。

**📈 对比分析**

对比方法：ASCOOD、CADRef、Mahalanobis++、OODD、NECO、NCI、FDBD、MixOE、NovelBranch、Extended 等；在 C-America 本地/非本地/非蛾类未知上取得最高 AUROC（≈99.5%），在 BCI 交叉源评估中获得最佳 AUROC 96.1%、AUPR 38.8% 与 FPR95 16.1%，闭集准确率 95.2%。

**⚠️ 局限性**

局限性：依赖辅助物种数据；局部相似性假设在极端差异样本上可能失效；生态一致性需额外地理信息并非所有场景都有；模型参数约 31.9M、FLOPs 8.9G，推理速度受多证据融合影响。

---

## 568. What We Talk About When We Talk About Emotion Editing

**arXiv ID:** 2610.13558 | [PDF](https://arxiv.org/pdf/2610.13558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 569. Thyroid Nodule Segmentation in Ultrasound Images with Conformal Uncertainty Estimation

**arXiv ID:** 2610.13604 | [PDF](https://arxiv.org/pdf/2610.13604v1)

**作者:** Francesco Teodori `[一作]` (Sapienza University of Rome), Umberto Ferraro Petrillo `[通讯]` (Sapienza University of Rome)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并评估了基于 DeepLabV3+ + EfficientNet‑B5 的甲状腺结节分割基线，集成三种通道注意力模块、两种损失函数，并加入 split conformal prediction 进行像素级不确定性量化。

**💡 创新点**

创新点在于：①在纯视觉场景下系统比较 SE、CBAM、GCA 三种注意力机制对分割性能的影响；②首次在甲状腺结节分割模型后端部署 split conformal prediction，生成具有覆盖保证的不确定性带。

**🔧 技术方法**

使用的技术包括：DeepLabV3+ 编码‑解码结构、EfficientNet‑B5 预训练骨干、SE/CBAM/GCA 通道注意力、Dice‑BCE 与 Boundary 损失、AdamW + cosine 退火学习率、Albumentations 数据增强以及 split conformal prediction。

**📊 数据集**

采用公开的 TN3K 数据集（3493 张超声图像，官方划分），在训练/验证/测试集上开展实验。

**📈 对比分析**

通过 Dice、MCC、检测率（按结节大小与数量分组）等多维指标进行评估，最佳配置 CBAM+Dice‑BCE 获得 mDice 84.84%，检测率最高，整体性能超过先前报道的 83.77%。

**⚠️ 局限性**

主要局限在于仅使用单一数据集，未验证跨域泛化能力，并且在多结节图像上的检测率相对较低。

---

## 570. ELMA: Benchmarking Anaphoric Compositions for Long-Term Text-to-Motion Generation

**arXiv ID:** 2610.13549 | [PDF](https://arxiv.org/pdf/2610.13549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 571. A small cognitive core explains human behavior at scale

**arXiv ID:** 2610.13603 | [PDF](https://arxiv.org/pdf/2610.13603v1)

**作者:** Marcel Binz `[一作]` (Helmholtz Munich), Eric Schulz `[通讯]` (Helmholtz Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于上下文检索与编码（CORE）模型，融合多种认知原则，对 137 个实验、7,036,442 次选择的数据进行联合拟合，评估其对人类行为的解释力。

**💡 创新点**

创新点在于将传统的记忆写入预测误差、上下文依赖、层次结构和语音输入等认知原则整合为单一可解释模型，并在多范式数据上实现对 75% 实验效应的再现，展示小核心可解释大部分行为。

**🔧 技术方法**

使用音素嵌入、堆叠的可上下文化记忆层（RMSNorm、SiLU、线性注意力实现）、Hebbian/预测误差编码以及最大后验估计训练，配合 flash-linear-attention + Triton kernel 加速。

**📊 数据集**

使用公开的 Psych‑201‑discrete‑agentic 数据集（137 个实验、62 项研究、7,036,442 次选择），将实验转录为音素序列并拆分为训练/测试集。

**📈 对比分析**

通过负对数边缘似然（NLML）和预测负对数似然（PNLL）与相同配置的 Transformer（字节对分词）以及黑盒模型 Centaur 进行比较，CORE 在 NLML 3,498,133、PNLL 262,572 上显著优于 Transformer（4,592,031 / 329,043）和 Centaur（389,151），并在 62 项研究中普遍获胜。

**⚠️ 局限性**

局限性包括对计划、长期记忆和外部知识的建模不足，无法捕捉约 25% 的行为效应；模型的键/值/学习率/遗忘机制缺乏理论解释，且不支持跨实验的长期学习。

---

## 572. Time-Varying Favorite-Longshot Bias in Polymarket Weather Markets

**arXiv ID:** 2610.13593 | [PDF](https://arxiv.org/pdf/2610.13593v1)

**作者:** Shouchen Qiu `[一作]` (Northeast Yucai School), Huaijun Qiu `[通讯]` (Shenyang Aerospace University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对Polymarket每日最高温度市场中的长短投机偏差随时间变化进行定量分析，比较低价长短票与高价热门票在不同时间窗口内的定价误差。

**💡 创新点**

创新点在于将时间窗与价格区间分离，揭示高价热门票在距离峰值1–12小时内显著过度定价，而长短票在30–48小时内轻微过度定价；同时结合事件级概率得分、三次B‑spline校准和Buy‑NO经济测试，提供多维度的误差评估。

**🔧 技术方法**

使用事件级Brier、Log Loss、RPS得分；三次B‑spline平滑校准；Buy‑NO收益模拟；Bootstrap置信区间评估统计显著性。

**📊 数据集**

Polymarket公开API获取的2025‑12‑05至2026‑07‑31十个城市的每日最高温度市场数据，包含2189个事件、21697个桶（互斥温度区间）和854万笔交易。

**📈 对比分析**

通过训练集的可靠性曲线、区间偏差、事件级得分在不同时间窗口内进行比较；在验证集和测试集对B‑spline校准和Buy‑NO收益进行对比。校准对得分几乎无显著提升；Buy‑NO在热门票窗口实现约+50% ROI，长短票窗口约-0.23% ROI。

**⚠️ 局限性**

局限性包括：仅研究十座城市的一年内每日最高温度市场，流动性与天气信息质量差异未充分控制；样本规模和时间窗口有限，未检验其他天气商品或不同预测市场的普适性。

---

## 573. AMLworld-Compact: Importance-Weighted Downsampling for Cost-Effective LLM Evaluation and Error Diagnosis

**arXiv ID:** 2610.13627 | [PDF](https://arxiv.org/pdf/2610.13627v1)

**作者:** Natapong Nitarach `[一作]` (SCB DataX, SCBX Group), Kunat Pipatanakul `[通讯]` (SCB DataX, SCBX Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于HT‑Coreset的评估集，保留所有非法交易并按难度分层抽样合法交易，极大降低LLM评估所需的图提示数；同时对七种开源LLM在AMLworld小样本集上的检测与典型识别性能进行了系统评估。

**💡 创新点**

创新点在于①设计了保留全部非法样本、分层抽样合法样本的HT‑Coreset协议；②利用Horvitz–Thompson权重实现对全测试集性能的无偏估计；③对LLM输出进行结构化“词法检查”并对错误进行细粒度分解。

**🔧 技术方法**

使用的技术包括：Horvitz–Thompson采样权重、LightGBM/XGBoost/GCPAL的分层评分、Graph Feature Preprocessor (GFP)特征、k-hop图文本序列化、LLM提示与少量示例（few‑shot）对比。

**📊 数据集**

使用AMLworld的两份最小拆分（HI‑Small和LI‑Small）作为评估数据集，数据中含有约200万条交易边，其中非法比例分别为0.12%和0.05%。

**📈 对比分析**

在保持全测试预设召回率的前提下，LLM检测的F1在0.23%–0.25%之间，接近全预测非法的基线（0.246%），但精度提升有限；典型识别的宏F1在10%–18%之间，few‑shot提示显著提升典型识别但对检测几乎无影响。

**⚠️ 局限性**

局限性包括：评估仅覆盖七种开源LLM和两份合成AMLworld样本，缺乏真实银行数据和大图规模的验证；HT‑Coreset抽样基于预训练评分，可能对其他预测器不具转移性；LLM提示未保证包含目标交易；且对置信度的词法检查不足以判断推理正确性。

---

## 574. $L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression

**arXiv ID:** 2610.13574 | [PDF](https://arxiv.org/pdf/2610.13574v1)

**作者:** Cameron Mouangue `[一作]` (LAMSADE, Université Paris Dauphine-PSL, CNRS), Tristan Cazenave `[通讯]` (LAMSADE, Université Paris Dauphine-PSL, CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于蒙特卡洛树搜索（GNRPA）的稀疏特征选择框架，用来构建高维金融回归的可解释性线性模型。

**💡 创新点**

创新点包括：①将特征选择建模为马尔可夫决策过程，并用验证误差引导的GNRPA进行搜索；②引入多目标奖励（R²、子集大小、最大VIF）实现稀疏且低多重共线性的子集选择；③支持自适应停止和多种搜索策略（固定卡路里、调参卡路里、内生停止）。

**🔧 技术方法**

核心技术：蒙特卡洛树搜索 + Generalized Nested Rollout Policy Adaptation (GNRPA)，OLS回归评估子集，mRMR启发式用于策略偏置，Optuna用于超参数搜索，Ridge/Lasso/ElasticNet等基线算法。

**📊 数据集**

数据集：2000‑2026 年 NASDAQ‑100 每周收益作为目标，基于 1,072 个工程化特征（市场、宏观、因子、技术指标）构建的日数据；使用前 2024 年做训练/验证，2024‑2026 年做 OOS 测试。

**📈 对比分析**

与 Ridge、Lasso、ElasticNet、前向逐步等基线比较；固定卡路里、调参卡路里和自适应停止三种 GNRPA 配置。OOS R² 最佳为 0.6961（ensemble），单次运行 0.6682，均显著优于 Lasso 0.6313；计算时间从 <1s（Lasso）提升到 1,671s（自适应停止）。

**⚠️ 局限性**

局限性：①计算成本高、超参数调优繁琐；②对时间窗口和市场 regime 变化敏感；③仅在单个指数、单一时间段评估，缺乏跨资产/滚动验证；④未与传统元启发式（GA、SA）或非线性模型进行对比；⑤对极高维度可扩展性仍需进一步验证。

---

## 575. A Simplified Layout Algebra for GPU Kernels

**arXiv ID:** 2610.13679 | [PDF](https://arxiv.org/pdf/2610.13679v1)

**作者:** Nan An `[一作]`, Xiaotian Zhou `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现一种简化的GPU张量布局代数，消除CuTe中多达六条可接受性条件，只需通过空间映射声明一次性决定布局有效性，并用OCaml实现、Coq证明其决策与简化正确性。

**💡 创新点**

创新点在于将布局的空间映射（逻辑、线程值、物理）显式声明，降低复杂条件到单一可检测约束；同时提供一次性决策算法（基于分段阶梯与向前差分）来判断合成是否仍为布局，并用形式化方法证明其正确性。

**🔧 技术方法**

使用的技术包括：布局代数的形式化定义、OCaml实现与代码生成、Coq（Rocq）证明、扫描算法（基于分段阶梯与向前差分）决定布局、swizzle变换、canonical布局与逻辑/物理/线程值空间的区分。

**📊 数据集**

评测数据集主要为：随机生成的各种布局（尺寸不超过12）、CuTe文档示例、CUTLASS原子、GEMM主循环等；未使用真实应用数据集，仅通过枚举与示例验证。

**📈 对比分析**

比较方法为逐坐标对比CuTe已实现布局、操作及示例，验证相同输出；对随机布局枚举做决策一致性检验。性能上实现体积小、代码生成更简洁，未报告显著性能提升，但能避免CuTe隐式错误。

**⚠️ 局限性**

局限性：仅支持非读写重叠的映射，无法处理物理空间向外映射的情况；缺乏对CuTe完整API（如广播、细粒度覆盖）的完整覆盖；实际编译器集成尚未完成，尚未在真实GPU程序中验证性能差异。

---

## 576. Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM

**arXiv ID:** 2610.13660 | [PDF](https://arxiv.org/pdf/2610.13660v1)

**作者:** Adrien Deliege `[一作]`, Marc Van Droogenbroeck `[通讯]` (University of Liège)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在ARC任务上VARC和TRM模型的预训练与测试时训练预算对性能的影响，并在同一协议下系统对不同预算进行交叉实验

**💡 创新点**

揭示预算配置（样本呈现次数和任务变体数量）是影响ARC性能的关键因素，而非单纯的模型结构；同时提供可复现的预算设定与代码

**🔧 技术方法**

使用视觉Transformer/ViT（VARC）与递归Transformer（TRM）两大模型架构，采用任务增广、测试时微调以及多候选投票等技术

**📊 数据集**

ARC（Abstraction and Reasoning Corpus）数据集，使用其中400个测试任务进行评估

**📈 对比分析**

通过对预训练预算（V,B）和TTT预算（V,B）进行全组合实验，比较ARC pass率、oracle等指标，发现TTT样本呈现预算提升最为显著；增加TTT任务变体数量对TRM有明显利好，而VARC受益有限；预训练对TRM提升明显，VARC提升相对有限

**⚠️ 局限性**

实验仅基于单一随机跑，未对多次实验的方差进行评估；仅在ARC上测试，未覆盖更困难的ARC变体或其他视觉推理基准；VARC与TRM参数规模差异导致仅能观察各自对预算的敏感度，而非直接排名

---

## 577. Matched-Context Group Policy Optimization for LLM Agent Training

**arXiv ID:** 2610.13755 | [PDF](https://arxiv.org/pdf/2610.13755v1)

**作者:** Chenxu Zhang `[一作]` (Imperial College London), Yiming Cheng `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于匹配预动作状态与任务进度的Matched-Context Group Policy Optimization (MCGPO) 方法，用已收集的rollout组生成支持的局部动作偏好，并与GraphGPO联合训练。

**💡 创新点**

创新点在于：①通过双行动支持与状态+Rubric进度匹配，选取能同时支持两种动作的比较上下文；②利用相反折叠的终端结果标注偏好方向，并用证据权重调节pairwise损失；③实现完全无critic的局部信用分配，兼容现有rollout数据。

**🔧 技术方法**

技术手段包括：组相对强化学习、GraphGPO任务信用、加权pairwise损失、状态+Rubric进度匹配规则、证据权重机制、批量微批次训练等。

**📊 数据集**

实验数据集：ALFWorld（家庭交互）和WebShop（产品搜索与购买），使用Qwen2.5-1.5B、7B以及Qwen3-4B、8B语言模型。

**📈 对比分析**

与GraphGPO基线及其Rubric-Shaping版本对比，MCGPO在六种环境-模型组合上平均提升0.47–1.46个百分点；ALFWorld成功率从92.6%提升到93.6%/96.3%，WebShop在不同规模模型上提升约1–2个百分点，验证了方法在长期稀疏奖励任务中的有效性。

**⚠️ 局限性**

局限性：依赖可手工提取的状态+进度标注，适用性受限于任务规则明确；在极端稀疏奖励或高状态多样性的场景中，证据支持可能不足，导致信用分配仍面临挑战。

---

## 578. SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices

**arXiv ID:** 2610.13740 | [PDF](https://arxiv.org/pdf/2610.13740v1)

**作者:** Timi Owolabi `[一作]` (ML Collective), Ayotomiwa Oyewumi `[通讯]` (ML Collective)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从预训练的大型视频扩散Transformer（Cosmos Policy）中蒸馏出一种紧凑的潜在世界-动作模型（SmolWAM），通过简化接口、裁剪Transformer层、使用单帧潜在编码和缓存BERT文本特征，实现了在边缘设备上高效推理。

**💡 创新点**

创新点包括：① 接口简化——单帧潜在、缓存BERT特征、只保留12/28层、28个动作查询与一次性rectified‑flow解码；② 分阶段裁剪+蒸馏课程，先用教师的全尺寸布局训练后再迁移到紧凑布局；③ 在显著降低计算量与内存占用的同时，保持大部分成功率与世界预测能力。

**🔧 技术方法**

使用技术：预训练视频扩散Transformer（DiT），Wan2.1 VAE单帧编码，BERT‑base缓存文本特征与零填充，Attention‑sink保留，单步rectified‑flow推断，裁剪策略（块影响评估、注意力sink保留、短块裁剪），蒸馏损失（动作与未来潜在对齐）以及AdamW、bf16训练。

**📊 数据集**

使用的数据集：LIBERO机器人基准（四套件共400个episode，60个episode筛选集），并使用Cosmos Policy演示轨迹作为训练与评估参考。

**📈 对比分析**

比较方法与性能：与原Cosmos Policy（5步/1步）、Fast‑WAM等对比。SmolWAM在400个episode中实现95.5%成功率（教师98.2%），推理时延57 ms/16动作，2.3 TFLOP，2.36 GiB内存；比教师快11×，FLOP降低18.5×，内存减少3.2×；同时保持了对未来观测的预测能力。

**⚠️ 局限性**

局限性：仅在LIBERO上评估，未验证在其他机器人基准或真实机器人上的表现；裁剪深度对长时序任务影响较大，长时序性能下降显著；未来工作需改进裁剪策略以更好保留长时序控制。

---

## 579. KVeil: First-Place Solution to ECCV 2026 LifeGenIP Challenge on Unlearnable Videos

**arXiv ID:** 2610.13733 | [PDF](https://arxiv.org/pdf/2610.13733v1)

**作者:** Boheng Li `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对 ECCV 2026 LifeGenIP 挑战，设计并实现 KVeil 方法，在有限且不可见的扰动预算下，对面部视频进行防御，阻止未经授权的基于参考帧或微调的 diffusion 视频生成模型对身份的定制。

**💡 创新点**

创新点：
- 引入 Reference Capacity Collapse（V‑capacity collapse + K‑attract）直接削弱参考帧在 DiT 关注机制中的信息容量，并保持其对后续帧的影响；
- Motion‑Guided EOT 共享扰动，利用运动能量平衡窗口选取代表帧，在全视频范围内学习一致的扰动，提高对隐藏预处理与未知推理条件的鲁棒性；
- Phase 2 双层框架：先对面部区域进行 Prior‑Aligned Face Protection（使用合成目标面部对齐结构化扰动），再在剩余背景区域进行白盒背景保护，兼顾可迁移性与强大的白盒破坏；
- 通过软/硬面部掩模、TPS warp 与 3D VAE+DiT 结合，形成时空耦合的扰动策略。

**🔧 技术方法**

主要技术：
- LETHE 机制（V‑capacity collapse 与 K‑attract）;
- Expectation‑over‑Transformation (EOT) 随机化参考帧、输入变换、提示词与 Transformer 层数；
- 3D VAE 编码与 DiT 关注层；
- 目标面部生成（ChatGPT Image）与 Umeyama/TPS 对齐；
- Soft/Hard face masks、TPS warp、Gaussian 过滤、残差缩放；
- LoRA 微调对抗实验。

**📊 数据集**

使用的数据集与模型：
- 10 个身份，每个身份 10 条 113 帧 640×480 面部视频；
- 评测模型：Wan2.2‑TI2V‑5B、HunyuanVideo‑1.5、Kling 3.0、LTX‑2 等；
- 生成目标面部使用 ChatGPT Image；
- 评测包含各种隐藏处理（H.264、VideoPure 等）与未知模型。

**📈 对比分析**

对比方法：参赛队伍（JLShen、Rover 等）以及公开实现；KVeil 在 Phase 1 以 55.70/80.03/79.95 的三维指标夺冠；在 Phase 2 上在 invis、eff、rob、trans 等指标均位居榜首。实验显示：Face Score 在已知模型上从 16.79 提升至 82.66；在未见模型 HunyuanVideo‑1.5 上达到 34.40；在 LoRA 1500 步微调上保持高达 51.09 的面部扰动效果。整体性能显著优于对手。

**⚠️ 局限性**

局限性：
- 需要事先生成并对齐目标面部，增加预处理步骤；
- 对极端压缩或滤波攻击（如强重采样、色彩失真）仍存在一定脆弱性；
- 仅针对单主体面部视频，对多主体或非面部场景的推广有限；
- 受扰动预算限制，可能在更高分辨率或更大时长的视频上效果下降；
- 对某些未知模型的迁移性能尚未达到 100% 级别，需要进一步提升。

---

## 580. Amortized Counterfactual Prediction with Graph Neural Networks

**arXiv ID:** 2610.13724 | [PDF](https://arxiv.org/pdf/2610.13724v1)

**作者:** Hendrik Stumpf `[一作]` (University of Oxford), Francesco Quinzan `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种端到端的 Counterfactual Graph Neural Process (CFGNP)，可直接从观测数据推断因果图未知的反事实分布。

**💡 创新点**

创新点在于将图神经网络与神经过程相结合，利用双图结构约束而无需显式估计因果图，能够自适应不同大小的图和未见特征。

**🔧 技术方法**

技术上使用 Graph Neural Process、GAT 消息传递、混合高斯输出以及负对数似然训练。

**📊 数据集**

使用合成数据（Triangle、M-shape、German Loan）和真实图像数据 CheXpert 进行评测。

**📈 对比分析**

与 VCA、BGM、CFM 等基线相比，在 MSE/MAE、准确率和 ROC‑AUC 上均取得显著提升，且在更大图和 OOD 特征上仍保持优良性能。

**⚠️ 局限性**

局限性包括对观测上下文的高计算成本，若反事实不可识别则模型无法恢复真实分布。

---

## 581. Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers

**arXiv ID:** 2610.13735 | [PDF](https://arxiv.org/pdf/2610.13735v1)

**作者:** Yuhong Chou `[一作]` (Hong Kong Polytechnic University), Jibin Wu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并评估了一种名为Orthros attention的自回归高效注意力机制，支持prefilling‑decoding异质计算，兼顾长上下文推理与速度提升。

**💡 创新点**

通过在不同生成阶段动态切换滑动窗口与全局交叉注意力，实现prefilling线性复杂度、解码全注意力，且保持训练‑推理一致性，突破传统KV剥离导致记忆下降的局限。

**🔧 技术方法**

结合滑动窗口注意力、全交叉注意力、可学习的尺度偏移查询变换、基于相同KV缓存的双路计算、数据高效转换、混合硬件优化与批量解码。

**📊 数据集**

使用FineWeb‑Edu 100B tokens、80B CommonCrawl、ProLong长上下文、Common Sense Reasoning benchmark、MMLU、128K needle‑in‑a‑haystack等数据集。

**📈 对比分析**

与标准Transformer、滑动窗口、GDN、GLA、RetNet、SWA等基线在多尺度（160M‑6.8B）、预训练、模型转换、长上下文检索、prefilling速度（最高21.9×加速）和端到端推理（TTFT低至10s、吞吐率20‑90 tok/s）进行对比，Orthros在保持或提升精度的同时大幅提升推理效率。

**⚠️ 局限性**

训练时需要双倍序列长度导致额外计算；KV缓存的内存效率未优化；混合专家、强化学习多轮推理以及单样本解码等场景仍需进一步研究。

---

## 582. Machine Unlearning for Federated Socially Fair k-Means Clustering

**arXiv ID:** 2610.13717 | [PDF](https://arxiv.org/pdf/2610.13717v1)

**作者:** Aravind Reddy `[一作]` (Indian Institute of Technology Madras), Saurav Prakash `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可在联邦环境下实现社会公平 k‑means 的算法，并给出在删除数据时实现精确无学习（exact unlearning）的完整框架；

**💡 创新点**

核心创新在于：①按群体分离本地摘要，保证删除后全局公平权重变化时摘要仍有效；②证明在理想采样下初始化误差与客户端群体比例无关；③设计可重放的“exact replay”与赋值证书机制，确保重训练与原训练完全一致；

**🔧 技术方法**

采用 D²‑sampling（k‑means++）、权重化 D²‑sampling、双组公平更新（Fair‑Lloyd）、整数精确统计、赋值证书与可重放算法；

**📊 数据集**

实验使用公开的 Adult、Bank、Credit 数据集以及美国社区调查（ACS）人群数据；

**📈 对比分析**

与传统完整重训练（fresh）对比，整体实验显示：在整机客户删除时，使用无重训练（direct replay）在零轮调整下可比完整训练快 35.8 倍，加入 1、2 轮公平细化仍保持 2.4 倍、1.7 倍加速；单条记录删除时加速仅约 1.06 倍；证书重放对总时间影响不大；

**⚠️ 局限性**

局限包括：依赖固定的预处理与固定点编码；不提供差分隐私或安全删除保障；重放后缓存失效需重新构建；实验受限于单台机器、少量随机种子与特定硬件；以及对非二元群体、连续群体和多组情况支持有限。

---

## 583. A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models

**arXiv ID:** 2610.13663 | [PDF](https://arxiv.org/pdf/2610.13663v1)

**作者:** Ivan Moloshnikov `[一作]` (National Research Centre Kurchatov Institute), Yuanchao Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于多智能体大型语言模型的框架，用于在不改变网络结构的前提下自动微调专用神经网络（如脉冲神经网络和物理信息神经网络）的关键模块。

**💡 创新点**

创新点在于：①将分析师–程序员–记录员三角色与沙盒执行环境及实验日志外存相结合，形成统一的迭代搜索循环；②提出详细提示和方法分类覆盖图，显著提升弱LLM的搜索效果；③框架可跨任务（分类、PINN）无改动使用，展示了高度通用性。

**🔧 技术方法**

技术实现包括：大型语言模型（Claude Opus 4.6、Perplexity Sonar、Qwen3-30B-A3B-Instruct-2507）+ Chain‑of‑Thought 计划、方法类别分类、覆盖图、沙盒 Docker 环境、JSON 记录日志、外部工具调用（Python 代码生成、训练、评估）。

**📊 数据集**

使用的数据集与任务包括：MNIST、scikit‑learn digits（分类）；GNLSE 单个孤子、GNLSE 高斯扰动、Kuramoto–Sivashinsky 迁移波（物理信息神经网络）。

**📈 对比分析**

通过与基准解码器、随机采样、TPOT 进化搜索以及手工设计的 PINN 进行对比；在 MNIST 上准确率从 79.9% 提升到 92.8%（+13pp），在 digits 上从 90.8% 提升到 96.7%（+6pp）；单孤子问题相对误差降低约 2500 倍；高斯扰动 GNLSE 的积分误差降低约 325 倍；KS 任务的残差降低 9.3 倍，质量漂移降低 14 倍。

**⚠️ 局限性**

局限性包括：搜索空间受限于预设的可编辑模块；实验日志为线性列表，缺乏层次化图结构，限制深度推理；迭代次数固定（SNN 250、PINN 50）可能未达到最优；部分任务仅单种子评估；对商业 LLM API 依赖可能影响可复现性；未与 LLM 驱动的 NAS 系统直接对比。

---

## 584. Neurons++: A Unified Visual-Stream Framework for Image and Video Reconstruction from fMRI with Neural-to-Visual Interpretability

**arXiv ID:** 2610.13763 | [PDF](https://arxiv.org/pdf/2610.13763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 585. PhysForge: Material-Controllable 3D Reconstruction with Separable Components

**arXiv ID:** 2610.13681 | [PDF](https://arxiv.org/pdf/2610.13681v1)

**作者:** Yuchen Wang `[一作]` (Nanjing University), Jing Huo `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于视频扩散模型的框架，用于从单一输入对象生成可物理模拟的3D资产，并实现材质可控与语义组件可分离。

**💡 创新点**

核心创新在于：①利用材质条件化的VDM输出提取物理信息并通过窗口化拟合隔离有效交互，剔除非物理运动；②对语义组件进行分解并分别优化其物理参数，随后在相对运动分组下实现灵活组合的联合仿真。

**🔧 技术方法**

采用材质条件化的视频扩散模型、VLM指导、3D高斯喷射（3DGS）几何表示、材料点法（MPM）仿真、窗口化拟合策略、SAM3/Depth Anything语义分割，以及基于Taichi的MPM/刚体求解器。

**📊 数据集**

在PhysAssets（24000+带物理标签的3D资产）和Google Scanned Objects（1000+高质量扫描物体）数据集上进行训练与评估。

**📈 对比分析**

与DreamPhysics、PSIVG、PhysRVG、WMReward、DeforM、PhysGM等六种基线在CLIP相似度、物理可控性（PC）和用户研究中对比，本文方法在CLIP约0.261、PC约4.3等指标均超过所有基线，显示出更高的材质一致性和物理表现。

**⚠️ 局限性**

局限性包括：每次材质更改都需重新生成视频并进行拟合；对VDM生成视频质量和非物理运动分离的依赖导致性能波动；当前非物理运动分离方法仍不够完善，影响拟合精度。

---

## 586. Gaussian Splatting with Reflections Guided by What is Seen

**arXiv ID:** 2610.13757 | [PDF](https://arxiv.org/pdf/2610.13757v1)

**作者:** Yiming Liang `[一作]` (Waseda University), Hiroshi Ishikawa `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于2D Gaussian Splatting的近场反射新视角合成框架

**💡 创新点**

创新点在于将可直接观测的主高斯与次高斯结合渲染反射，采用混合色彩表示（Triplane+SH）和数据驱动的视角依赖镜面权重

**🔧 技术方法**

使用2D Gaussian Splatting、混合三平面颜色、球谐函数、可微射线跟踪、基于深度梯度的法向一致性损失

**📊 数据集**

使用公开的NeRF-Casting、EnvGS、Ref-Real等数据集以及新收集的NearRef四室内场景数据集

**📈 对比分析**

与3DGS、2DGS、EnvGS等基线对比，公开数据集上保持竞争力，在NearRef上实现更高的PSNR/SSIM和更锐利的反射，显示出显著的性能提升

**⚠️ 局限性**

局限性包括对环境光的直接观测依赖、未显式建模粗糙度/折射、计算开销较大，且对极端视角或远场光源的处理仍有限

---

## 587. LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos

**arXiv ID:** 2610.13732 | [PDF](https://arxiv.org/pdf/2610.13732v1)

**作者:** Zhiyuan Xiang `[一作]` (Joy Future Academy, JD), Xiang Deng `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LAPT‑Scale 两阶段预训练框架，利用连续 latent action 模型从海量 egocentric 人类视频生成隐式动作标签，并在三阶段 VLA 策略训练（人类视频预训练、机器人对齐、任务后训练）中提升真实机器人操控成功率。

**💡 创新点**

创新点在于引入连续 latent action 并结合感知、语义和光流对齐信号，避免 VQ‑LAM 的离散压缩与重建噪声；以及三阶段训练结构实现了规模化与跨机器人泛化。

**🔧 技术方法**

使用 spatio‑temporal VAE、Transformer 编码器/解码器，LPIPS、DINOv2 语义特征、RAFT 光流对齐以及 Conditional Flow Matching 的 action expert 等技术。

**📊 数据集**

数据集包括 23.3K 小时 egocentric 人类视频（如 Ego4D、EgoVid‑5M），5.1K 小时清洗后用于预训练，以及 250 小时机器人数据用于对齐，评估在 RoboTwin、RoboCasa‑GR1 以及七个实机任务上。

**📈 对比分析**

通过与多种 VLA 基线（2cπ_0、JoyAI‑RA、2cMotus 等）在 RoboTwin 2.0、RoboCasa‑GR1 及实机任务上比较，LAPT‑Scale 在 Easy/Hard 任务分别达 90.0%/89.3% 成功率，实机平均 85.0%，并在 5K 小时人类视频扩展时将实机成功率从 63.3% 提升至 85.0%，显示良好的规模化效果。

**⚠️ 局限性**

需依赖机器人对齐步骤，连续 latent action 与机器人可控自由度仍可能不完全对齐；在视角、遮挡或摄像机运动强烈的视频中表现可能下降；实验范围受限于单一机器人平台和少量任务，泛化性尚需进一步验证。

---

## 588. Towards Understanding the Training of GFlowNets: A Theoretical Perspective

**arXiv ID:** 2610.13711 | [PDF](https://arxiv.org/pdf/2610.13711v1)

**作者:** Rui Hu `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出统一理论，将生成流网络（GFlowNets）的平衡损失、近似稳态与终端采样精度关联起来，并给出终端对称KL误差界和有限时间收敛速率。

**💡 创新点**

通过把终端分布误差与期望平衡损失关联、将平衡不平衡分解为切线空间与残差两部分，并在可变采样和后向策略更新下给出收敛速率与漂移条件，实现对GFlowNet训练全流程的系统性理论分析。

**🔧 技术方法**

主要运用平衡损失分析、KL散度分解、切线空间投影、有效Fisher矩阵、Polyak–Łojasiewicz不等式以及随机梯度下降（SGD）理论。

**📊 数据集**

本文为理论研究，没有使用具体数据集。

**📈 对比分析**

由于是理论证明，未做实验比较；理论结果表明在覆盖、有效切线几何良好和漂移受控的前提下，终端对称KL误差可达到 O(ε) + O(T^{-1/2}) 的阶数。

**⚠️ 局限性**

需要充分覆盖、有效切线几何良好、漂移受控；在实际神经网络实现中这些假设往往难以满足，导致理论与实践的差距。

---

## 589. Enhancing Legal Reasoning in Pre-trained Language Models via Knowledge Graph-Guided Multi-Task Pre-training

**arXiv ID:** 2610.13722 | [PDF](https://arxiv.org/pdf/2610.13722v1)

**作者:** Ibrahim Siddig `[一作]` (AImotion Bavaria, Technische Hochschule), Munir Georges `[通讯]` (AImotion Bavaria, Technische Hochschule)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了一种基于知识图的多任务预训练框架，旨在提升德国道路交通法规（StVO）法律问答推理的性能，并进一步检验其跨语言、跨法规及跨任务的迁移能力。

**💡 创新点**

创新点包括：①将半结构化法律文本的层级与关系显式编码为知识图，并通过TransE获得软目标；②设计三种互补预训练任务（NSP、KG‑Guided 聚类CLU、KG‑Aware 段落分类CLS），联合训练实现零样本推理；③在CLS中使用门控注意力动态检索并融合KG上下文，避免推理时需要离线图遍历。

**🔧 技术方法**

技术手段涵盖：Transformer Encoder、TransE知识图嵌入、对比学习、下一句预测、基于注意力的门控融合、三任务联合损失、LLM（Mistral‑7B‑Instruct、Llama‑3.1‑8B）生成合成 QA、数据分层划分与质量过滤。

**📊 数据集**

使用的数据集包括：德国道路交通法规（StVO）文本；合成 StVO QA 数据集（12,554 条问答，按段落划分防泄漏）；爱尔兰与奥地利驾驶法规评测集；LexGLUE 七项英文法律基准。

**📈 对比分析**

实验采用九种 Transformer 编码器（五种法律域、三种通用域、一种跨语言域），共 324 试验。评估指标为宏/微 F1、精确率、召回率、平衡准确率。结果显示：在 StVO QA 上模型提升 10.3%–155%；在 LexGLUE 7 项中 4 项获得提升；在奥地利/爱尔兰零样本评测中提升 3.4%–15.8%；跨语言零样本到英文也实现 0.094–0.499 的增益。

**⚠️ 局限性**

局限性：仅在 German StVO 上验证，未覆盖其他法律体系或语言；合成 QA 可能缺乏真实性，未经过人工专家审核；随机负采样在 CLU 中可能导致干扰；KG 依赖 LLM 翻译与 TransE 噪声，未量化；模型容量与语言距离影响迁移效果，且 KG 上下文检索在更大代码库下的计算成本未评估。

---

## 590. GLIMPSE: Global-Local Indexing with Multi-Precision Spectral Encoding for Explorable Video World Models

**arXiv ID:** 2610.13706 | [PDF](https://arxiv.org/pdf/2610.13706v1)

**作者:** Zhufeng Xu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Lin Gao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 GLIMPSE 框架，实现全球-局部记忆接口和多精度频谱编码，用于摄像机控制下的可探索视频世界模型。

**💡 创新点**

创新在于将多频段多精度坐标与几何条件频谱衰减相结合的 STaPE，提升历史记忆位置的鲁棒性，降低投影误差对生成一致性的影响。

**🔧 技术方法**

采用冻结的 CUT3R 递归上下文、局部指针记忆、STaPE 位置编码、RoPE/PRoPE 等注意力机制，以及基于 DiT 的视频扩散模型和多尺度投影调度。

**📊 数据集**

训练数据来自 SpatialVID、RealCam-Vid、ReCamMaster、Context-as-Memory 四大混合集，并使用 DA3 缓存的相机姿态、内参与深度。

**📈 对比分析**

与 UCM、SPMem、GEN3C、VMem、ReWorld 等方法在 CLIP‑T/F、FID、FVD、WorldScore、Y‑traj 等指标上对比，GLIMPSE 在跨视角一致性、几何对应和 RGB 误差上显著优于基线。

**⚠️ 局限性**

局限性包括仅针对摄像机控制探索，未显式建模物体运动；推理时去噪延迟较高，缺乏实时性与少步蒸馏支持。

---

## 591. Improving Semantic Coverage in AI Test & Evaluation through Generative Augmentation

**arXiv ID:** 2610.13697 | [PDF](https://arxiv.org/pdf/2610.13697v1)

**作者:** Connor Hashemi `[一作]` (Kitware Inc), Keith Fieldhouse `[通讯]` (Kitware Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种三阶段生成式 AI 管道，用零样本视觉‑语言模型检测测试集中的语义缺口，利用扩散式图像‑图像模型填补缺口，并通过下游模型嵌入空间的统计验证来判断合成样本是否有效。

**💡 创新点**

创新点在于：①把零样本语义标注与嵌入聚类相结合，自动识别最重要且欠缺的语义；②使用图像‑图像扩散模型在保持场景内容的前提下实现高层语义转换；③通过漂移、覆盖率、完整度等嵌入空间指标对合成样本进行定量评估，从而实现无需重新训练模型即可改进评估集。

**🔧 技术方法**

核心技术包括：CLIP/VLM 的零样本图文匹配、k‑means 聚类的语义重要性过滤、CycleGAN‑Turbo 等图像‑图像扩散模型、DataEval 工具箱进行漂移/覆盖/完整度统计。

**📊 数据集**

在两个公开数据集上验证：Caltech Camera Traps（CCT）图像分类任务和 BDD100k 驾驶场景目标检测任务，重点针对昼夜语义缺口。

**📈 对比分析**

方法通过与理想全覆盖测试集对比，发现合成夜景样本的漂移、覆盖率和 FID 明显优于原始白天样本；在模型选择实验中，使用填补后的测试集能正确优先选择在昼夜两种条件下训练的模型，表现与理想全覆盖数据集相当。

**⚠️ 局限性**

主要限制包括：扩散模型分辨率低，导致目标检测细粒度特征衰减；对细微语义（如天气变化）的聚类敏感度不足；以及对合成样本的语义对齐仍需更高效的提示调优或多模型嵌入评估。

---

## 592. Retractions Revisited: A Replication, Extension, and Mechanistic Evaluation of Linguistic Obfuscation Effects in Fraudulent Science

**arXiv ID:** 2610.13678 | [PDF](https://arxiv.org/pdf/2610.13678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 593. Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks

**arXiv ID:** 2610.13693 | [PDF](https://arxiv.org/pdf/2610.13693v1)

**作者:** Nektarios Kalampalikis `[一作]` (Saarland University), Isabel Valera `[通讯]` (Saarland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种将后置校准转化为双层优化的框架（COG），在训练过程中同时更新模型和校准器，无需额外校准集和额外参数，仅需一个校准器学习率；

**💡 创新点**

创新点在于将后置校准嵌入为Stackelberg式双层优化，使校准器与模型在同一前向/反向传播中联合训练，简化流程并提升校准效果；

**🔧 技术方法**

技术包括双层优化（Bilevel Optimization）、Stackelberg博弈、梯度下降-上升（GDA）以及常见的校准方法如温度缩放、矩阵缩放、全连接层校准等；

**📊 数据集**

使用的公开数据集有SVHN、CIFAR‑10、CIFAR‑100、CIFAR‑10‑C、CIFAR‑100‑C、Rotated SVHN、CelebA；模型包括ResNet101、Vision Transformers（ViT）及其他CNN骨干；

**📈 对比分析**

与传统后置校准方法、Label Smoothing、V‑TST等进行对比；在准确率、负对数似然（NLL）、Brier分数、OOD检出（AUROC/FPR95）等指标上，COG‑FC（无额外参数）在大多数任务上达到或超过SOTA，且在多任务学习中表现优异；

**⚠️ 局限性**

局限性包括：对Bilevel优化的收敛理论尚未完全阐明；未探索优化的鲁棒性（如对抗性或分布漂移）；对不同任务类型的统一校准度量仍需进一步研究。

---

## 594. On the Encoding of Calendar Features in Covariate-Informed Time Series Foundation Models

**arXiv ID:** 2610.13676 | [PDF](https://arxiv.org/pdf/2610.13676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 595. Studying the Effects of Task Diversity Using Small Mathematical Worlds

**arXiv ID:** 2610.13750 | [PDF](https://arxiv.org/pdf/2610.13750v1)

**作者:** Emily Xia Meng `[一作]` (University of Washington), Henry Kvinge `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了两个基于整数和排列的小数学世界任务生态系统，利用可控任务集合研究任务多样性对模型学习、表示与迁移的影响。

**💡 创新点**

通过可程序生成的任务集实现任务多样性实验；使用线性探测与CKA评估隐藏层对未见属性的可解码性；探讨任务相关性和学习率对迁移的作用。

**🔧 技术方法**

训练小型MLP和Transformer；使用线性Ridge探测器和Centered Kernel Alignment (CKA) 进行表示比较；基于程序生成的合成数据集；采用少样本微调与全局评估。

**📊 数据集**

整数任务集（16个预训练，4个迁移），排列任务集（32个属性分为两组，预训练1-16个，4个迁移）；任务长度可调、基数为100等。

**📈 对比分析**

通过准确率、线性R²、CKA等指标进行比较；Transformer在预训练任务数增多时线性探测性能从0.20提升到0.37；迁移性能在相关任务下提升至70%+；但无关任务或低学习率下效果有限。

**⚠️ 局限性**

实验规模小；任务与曝光度相互影响；迁移对学习率高度敏感；CKA结果在相关属性间不稳定；未系统探究任务组合与曝光度独立影响。

---

## 596. On Privacy Mechanism Design Under a Bounded $α$-Lift Leakage Constraint with $χ^2$-Utility

**arXiv ID:** 2610.13685 | [PDF](https://arxiv.org/pdf/2610.13685v1)

**作者:** Amirreza Zamani `[一作]` (KTH), Mikael Skoglund `[通讯]` (KTH)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在 α‑提升（α‑lift）泄漏约束下，使用 Pearson χ² 信息作为效用指标的隐私机制设计问题。

**💡 创新点**

创新点在于：① 利用 χ² 信息的完全二次性，避免了传统互信息近似；② 推导了一个统一的 Taylor 剩余上界，能够在任何 ε 范围内给出可行机制、闭式下界以及 O(ε³⁄²) 的最优度量；③ 当 α=2 时，提供了完全精确解；④ 对任意泄漏矩阵给出完全可行的后验约束形式，并揭示了零泄漏下可获得正效用的必要条件。

**🔧 技术方法**

主要技术包括信息几何、泰勒展开、奇异值分解、凸优化与极值点分解、数据处理不等式、α‑lift 与 Rényi 发散的等价关系。

**📊 数据集**

实验采用了两组实用案例：医疗诊断数据（X∈{A,B,C}，Y∈{Low,Med,High}）和智能电表数据（X∈{Single,Family,SmallBusiness}，Y∈{Low,Mid,High}），分别给出具体的概率矩阵。

**📈 对比分析**

通过与一维精确线搜索得到的数值结果对比，证明了所给的下界与上界在实际 ε 范围内极为接近；α=2 时理论结果与数值完全吻合；在其他 α 值下，误差按 O(ε³⁄²) 缩小，且 α 越大时隐私约束更严格导致效用下降。

**⚠️ 局限性**

局限性：仅在 α>1 时讨论；假设 X‑Y‑U 为马尔可夫链且 P_X 全支持；对非可逆泄漏矩阵的机制可能输出维度过大；提出的单一方向最佳机制仅在给定 α 下得到局部最优，未证明全局最优；大 ε 情况下仍需数值搜索以确保精确满足约束。

---

## 597. Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction

**arXiv ID:** 2610.13672 | [PDF](https://arxiv.org/pdf/2610.13672v1)

**作者:** Jiwon Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Youngchul Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需微调LLM的推理纠错框架UPTURN，通过在失败推理轨迹的学习位置插入新的用户回合指令，实现对已存在推理路径的重新引导，从而修正错误答案。

**💡 创新点**

创新点在于：①仅使用冻结的Encoder与轻量级MLP头学习切割位置；②利用对话结构（用户回合边界）而非直接注入文本来有效转向模型；③通过门控机制自动判定哪些推理需要纠正，无需额外监督；④实现了跨模型、跨领域的迁移和与现有自纠方法的互补。

**🔧 技术方法**

技术手段包括：冻结的Qwen3-0.6B编码器+130K参数MLP头（位置策略）和66K参数门控网络；基于GRPO的强化学习训练位置策略；固定的注入指令与多轮对话结构；固定预算的前缀重生策略。

**📊 数据集**

主要使用了六个数学评测集（AIME24/25/26、AMC23、HMMT、MATH500）以及一个科学多选集GPQA-Diamond进行跨域测试；在实验中对四个不同规模的reasoner（Qwen3-4B/8B、DeepSeek-R1-Qwen3-8B、QwQ-32B）进行评估。

**📈 对比分析**

与基准Base@32（仅生成32条完整推理）、自纠方法Reflexion、SCoRe以及反射词注入方法AlphaOne、CyclicReflex相比，UPTURN在相同总生成预算下平均提升10~50点（视模型/数据集而定），尤其在DeepSeek-R1模型上提升幅度最大；在与GRPO后训练的reasoner结合时仍能进一步提升6.1点平均分；在跨模型和跨领域迁移时误差仅小于2.6点。

**⚠️ 局限性**

局限性包括：仅在数学推理任务上验证，未对代码/工具使用等其他推理场景进行测试；纠错过程仅限单轮注入，无法处理需要多轮连续纠错的情况；依赖门控网络的准确性，门控误判时可能无法纠正；在预算极低或已获得极高准确率的模型上提升有限。

---

## 598. (When) Should a Software Factory be Lights-off? Benchmarking Code Review Agents on Bug Detection

**arXiv ID:** 2610.13668 | [PDF](https://arxiv.org/pdf/2610.13668v1)

**作者:** Chenyang Yang `[一作]` (Carnegie Mellon University), Christian Bird `[通讯]` (Microsoft Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CRBugBench基准，用人类验证的bug做代码审查任务，评估AI审查代理的缺陷检测能力。

**💡 创新点**

提出基于历史bug-PR映射的客观评估框架，系统分析不同模型、任务特征及设计选项对bug检测的影响，并给出实践建议。

**🔧 技术方法**

采用大型语言模型（OpenAI GPT、Anthropic Claude、Claude Opus等）作为审查代理，利用Prompt、推理努力、单/多代理配置等技术。

**📊 数据集**

使用从GitHub 106个流行开源仓库挖掘的1800+任务，以及一个大型公司两份私有仓库的800+任务，涵盖Python、Rust、C#、JavaScript/TypeScript、Java等语言。

**📈 对比分析**

与多种模型（从小型到前沿大型模型）和多种配置（Prompt、推理力度、单/多代理）对比，发现顶级模型在前沿数据上可捕获约45-55%的已验证bug，且性能随模型迭代提升；但在私有仓库、复杂/多变更、非功能性或非diff可见bug时性能显著下降。

**⚠️ 局限性**

局限性包括：基准仅评估已验证bug的检出，未覆盖更广泛的审查价值；LLM评判可能有偏差；任务样本受限于已公开/公司数据；模型在非本地推理与项目知识方面表现不足，需人工干预或持续改进。

---

## 599. A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning

**arXiv ID:** 2610.13744 | [PDF](https://arxiv.org/pdf/2610.13744v1)

**作者:** Mayug Maniparambil `[一作]` (Fin.ai), Fergal Reid `[通讯]` (Fin.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在可验证奖励强化学习中，教师对学生 roll‑out 的评分能否提供有效的错误纠正信号，并提出通过生成批判（critique）并将其作为教师上下文来实现纠错的自我蒸馏方法。

**💡 创新点**

创新点是提出“批判条件自我蒸馏”（critique‑conditioned self‑distillation），将错误定位的生成任务与评分过程分离，从而得到局部化且具有纠正性的 token‑级监督。

**🔧 技术方法**

使用 RLVR 与 On‑policy Self‑Distillation 技术，结合教师模型（如 Qwen3、Olmo、Qwen‑5.2）与批判生成模型，计算 KL 与 token log‑ratio 信号，并在推理时插入批判文本。

**📊 数据集**

实验数据集包括 Knights & Knaves 逻辑谜题、DeepMath 竞赛数学题集、ProcessBench 的人工标注错误以及自制 320 题训练集。

**📈 对比分析**

与传统参考条件自我蒸馏相比，批判条件显著提升错误定位分数、纠正信号，并在测试时将 L7‑10 难度问题的成功率提高约 2×，在训练中相对于 GRPO 和参考条件提升 30–70%。

**⚠️ 局限性**

主要局限包括：实验规模有限（仅 320 题、100 步训练），批判生成额外消耗约 6%–20% 步时，且所有批判均需引用参考解，未在大规模 RL 训练上进行验证。

---

## 600. Complexity of Autobidding Equilibria in GSP Sponsored Shopping

**arXiv ID:** 2610.13716 | [PDF](https://arxiv.org/pdf/2610.13716v1)

**作者:** Kristoffer Arnsfelt Hansen `[一作]` (Aarhus University), Xinhao Nie `[通讯]` (Aarhus University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文研究了GSP广告竞价中的“自动竞价”平衡点的计算复杂性，证明了在预算约束和ROI约束两种模型下，平衡点总是存在且支持多项式大小的分布，并给出了构造方法；同时证明了计算近似平衡点是PPAD‑hard的，并证明了存在可在PPAD内求解的算法。

**💡 创新点**

创新点在于：① 将GSP竞价中的多商品多位次结构转化为可被Piecewise‑Linear（PL）伪电路处理的插入排序问题；② 通过量化耦合（quantile coupling）技术，将合法的插入概率分布合成全局可行的随机分配，确保支持多项式规模；③ 设计了完整的硬件电路证明，既证明了存在性，又给出了PPAD‑hardness的归约。

**🔧 技术方法**

使用的技术包括：PL伪电路、Heaviside伪门、排序网络（AKS）、量化耦合、误差离散化（piecewise‑constant correspondence）以及电路固定点分析；在硬件归约中还引入了c-指数分段价值尺度和多级守卫代理。

**📊 数据集**

该工作纯理论，没有使用任何公开数据集；所有实例均为构造性的合成问题。

**📈 对比分析**

与先前工作相比，本文给出了完整的复杂度框架：在PPAD内可判断存在性（membership），但计算近似平衡点是PPAD‑hard；证明了平衡点可由多项式数量的确定性分配支持，表明理论上可构造并验证。相比传统的第一/第二价竞价模型，GSP购物模型的复杂度更高。

**⚠️ 局限性**

局限性包括：① 仍未证明精确平衡点的PPAD‑hardness（只给出多项式成员性）；② 对于常数误差近似是否仍为PPAD‑hard仍未知；③ 仅在无约束或有限规模（如有限投标者、物品数、位次）下可求解，实际商业规模可能更复杂；④ 精确平衡点可能需要无理倍数，导致算法实现困难。

---

## 601. DIA: A Deep Reinforcement Learning-Based Interest-Aware ABR Scheme for Cloud Gaming

**arXiv ID:** 2610.13698 | [PDF](https://arxiv.org/pdf/2610.13698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 602. Transformer-based In-Context Score Operator Learning for Diffusion Models

**arXiv ID:** 2610.13690 | [PDF](https://arxiv.org/pdf/2610.13690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 603. Towards a Unified Taxonomy of Agricultural Harvesting Technologies: Linking Physical Interaction, Automation, and Selectivity

**arXiv ID:** 2610.13688 | [PDF](https://arxiv.org/pdf/2610.13688v1)

**作者:** Renato Cristiano Torres `[一作]` (Embrapa Agricultura Digital), Claudio Gottschalg Duque `[通讯]` (University of Brasilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并讨论了农业收获技术的统一分类法，基于物理交互原理与自动化水平两个主轴，并将选择性视为其衍生属性；

**💡 创新点**

将收获机制与自动化程度分离为独立维度，提出选择性作为桥梁属性，实现机械化与机器人化系统的共通对比框架；

**🔧 技术方法**

采用文献综述与概念性分析方法，对现有收获技术进行归纳与分类；

**📊 数据集**

未使用具体数据集，主要基于已有文献与案例进行理论梳理；

**📈 对比分析**

通过构建二维矩阵与分类表格进行定性比较，未给出定量性能指标；

**⚠️ 局限性**

缺乏实验验证与量化选择性指标，框架主要为概念层面，未覆盖所有技术与作物类型，适用性需进一步实证检验。

---

## 604. A Step Harder, Not a Step Too Far: The Benefits and Limits of Curriculum Difficulty Progression in GRPO

**arXiv ID:** 2610.13669 | [PDF](https://arxiv.org/pdf/2610.13669v1)

**作者:** Ronghan Li `[一作]` (Dalian Maritime University), Mingyu Lu `[通讯]` (Dalian Maritime University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在Group Relative Policy Optimization（GRPO）训练中，样本难度随策略演化的动态变化，并对比了静态预排序、固定中等难度以及分阶段提升难度的课程策略对模型最终性能的影响。

**💡 创新点**

首次系统量化了GRPO中样本难度的非单调、周期性变化，并发现分阶段难度提升能提升性能，但进一步提升到更难难度会导致性能下降，表明难度进度需受控。

**🔧 技术方法**

使用GRPO强化学习框架，结合在线难度估计（基于多采样成功率）、难度分段（B1–B5）、诊断子集与自适应转移阈值，对比不同课程方案。

**📊 数据集**

采用公开的数学推理数据集GSM8K。

**📈 对比分析**

通过多随机种子（3个）对照标准GRPO、静态课程、固定中等课程，在测试集上评估准确率；分阶段课程在B4阶段平均准确率为0.6591±0.0077，显著优于固定中等0.6449±0.0024和标准GRPO 0.6444±0.0026；但进一步进入B5阶段性能下降。

**⚠️ 局限性**

研究仅在单一模型与单一任务上验证，在线难度估计基于8次采样噪声较大；诊断子集与转移阈值是固定设计，缺乏跨任务的通用性和更灵活的动态阈值策略。

---

## 605. Cooperative Aerial Transportation of a Payload with a Sliding Cable Attachment

**arXiv ID:** 2610.13665 | [PDF](https://arxiv.org/pdf/2610.13665v1)

**作者:** Sebastian Tilaguy Lezama `[一作]` (Oakland University), Diego S D'Antonio `[通讯]` (Oakland University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了双旋翼无人机通过滑动挂钩悬挂传输非刚性负载的系统。

**💡 创新点**

引入可滑动挂钩，使绳长分布 λ 成为可调度自由度，实现绳索长度重新分配与负载位置协同控制。

**🔧 技术方法**

采用几何约束的轨迹生成、最小抖动多项式规划、滑动绳索动力学模型以及 Mellinger 控制器实现无人机姿态跟踪。

**📊 数据集**

在实验室使用 Crazyflie 2.1 组装的三台无人机和 1 m 绳索，配合 Lighthouse 视觉定位；未使用公开数据集。

**📈 对比分析**

通过三组实验（固定负载、轨迹跟踪、避障）比较 λ、γ 变化下的误差，平均欧氏误差约 3–3.5 cm，标准差 1–5 cm，证明间接控制可满足 0.08 m/s 速度约束。

**⚠️ 局限性**

理论假设滑动挂钩无摩擦导致实际误差增大，负载摆动、绳索动力学及无人机跟踪误差未在闭环中补偿，且 λ、γ 需要手动规划，缺乏自动化优化。

---

## 606. The Distortion of Threshold Approval Multiwinner Voting

**arXiv ID:** 2610.13725 | [PDF](https://arxiv.org/pdf/2610.13725v1)

**作者:** Mohamad Latifian `[一作]` (University of Edinburgh), Alexandros A. Voudouris `[通讯]` (University of Southern Denmark)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在多选委员会选举中，利用阈值审批形式收集投票者偏好，并在归一化单调对称范数框架下分析并给出了确定性和随机化规则的失真上界与下界。

**💡 创新点**

创新点在于：①提出了阈值审批与归一化单调对称范数相结合的模型，涵盖加法、最佳代表、top‑r、ℓ_p 与 OWA 等多种福利度量；②给出了以阈值数量为参数的失真上界 2(2m)^{1/t}（确定性）和 2min{m^{1/(t+1)},m/k}（随机化），并证明在单选或常数委员会大小时可匹配下界；③证明了 O(log m) 个阈值即可实现常数失真。

**🔧 技术方法**

技术方法主要包括：虚拟效用构造（用阈值区间下界替换真实效用），几何阈值选取，范数属性（同质性、单调性、正齐次、三角不等式）对福利分析的运用，以及结合确定性和随机化分支的失真证明。

**📊 数据集**

本文为理论性研究，未使用任何实验数据集。

**📈 对比分析**

通过严格的数学证明，比较了不同阈值数下的失真上界与下界；结果显示在阈值数为 1 时失真分别为 O(m) 与 O(√m)，而阈值数为 ⌈log₂m⌉ 时失真可降至常数，随机化规则在常数委员会规模时匹配下界，证明其优越性。

**⚠️ 局限性**

局限性包括：①对大委员会规模的上界与下界仍有一定差距；②未讨论实现最大化虚拟福利的计算复杂度；③阈值固定且非自适应，未考虑投票者特异性阈值或动态阈值调整。

---

## 607. A Baseline Assessment for the Agentic Web Technology Stack

**arXiv ID:** 2610.13703 | [PDF](https://arxiv.org/pdf/2610.13703v1)

**作者:** Vagner Figueredo de Santana `[一作]` (NIC.br), Diogo Cortiz `[通讯]` (NIC.br)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过自研的多智能体系统测试框架 MASTIF，系统评估了 3 种 LLM、3 种协议和 4 种框架在 Mind2Web 300 任务上的性能，提出了 Agentic Web 技术栈的基准。

**💡 创新点**

创新点在于：①提出统一的 MASTIF 评估框架，支持多模型、多协议、多框架的组合实验；②首次引入社会价值稳定性（Social Value Stability）指标来评估代理在推理过程中的价值一致性；③给出多维度基准结果，为后续 Agentic Web 设计与优化提供参考。

**🔧 技术方法**

使用技术包括：LLM（Mistral‑7B‑Instruct‑v0.1、Qwen2.5‑7B、Llama‑4‑Scout‑17B‑16E），协议（MCP、A2A、标准消息）、实现框架（LangChain、LangGraph、CrewAI、Semantic Kernel），13 个预置工具（如 web_search、web_browser、python_repl 等），以及 GPT‑4o‑mini 作为判别者。

**📊 数据集**

数据集为 Mind2Web 基准，包含 137 个真实网站、31 个应用领域、总计 2,350 条任务，从中 stratified 采样 300 条作为实验任务。

**📈 对比分析**

比较方法：对任务完成率、收敛时间、冗长度、工具使用成功率和社会价值稳定性进行多维度量化，并使用 Kruskal‑Wallis、Dunn’s Post‑hoc 等非参数统计检验。实验结果表明：
- 任务完成率与整体效率最高的技术栈为 Qwen2.5‑7B + LangChain + 标准协议；
- 工具使用成功率最佳为 Qwen2.5‑7B + Semantic Kernel + MCP；
- 社会价值稳定性最高为 Mistral‑7B + A2A + Semantic Kernel；
- 其余指标在不同协议与框架组合间呈显著差异。

**⚠️ 局限性**

局限性包括：①实验规模受模型、框架、协议数量限制，导致成本高昂；②仅 Qwen2.5‑7B 实现了工具调用，无法对其他模型的工具使用进行评估；③协议的完整实现缺少公开服务器，协议层实验受限；④工具描述、温度参数等细节对结果影响大，结果对设置敏感；⑤实验仅基于 Mind2Web，未覆盖更广泛的 Web 场景。

---

## 608. One-Way Delay Model for 5G Low Latency, High Data Rate Communications

**arXiv ID:** 2610.13761 | [PDF](https://arxiv.org/pdf/2610.13761v1)

**作者:** Hamid Hassani `[一作]` (Eindhoven University of Technology), George Exarchakos `[通讯]` (Eindhoven University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在5G边缘网络中近似排队延迟（QD）的框架，将QD分解为可测的聚合延迟和信道接入时间；

**💡 创新点**

创新点在于利用UE侧可测的聚合延迟与网络条件估算的信道接入时间，提供对BS RLC层排队状态的细粒度估计；

**🔧 技术方法**

使用NS3 5G‑LENA仿真环境，对RLC/MA C层进行修改，模拟单UE与多UE（UDP/TCP混合）场景；

**📊 数据集**

无公开数据集，全部实验采用NS3仿真生成的交通流与网络负载；

**📈 对比分析**

与传统的RTT/OWD测量对比，证明在高网络负载下QD占比>90%，模型在不同负载与用户场景下与实际QD曲线吻合，误差小于2.5 ms；

**⚠️ 局限性**

局限在于仅考虑无确认模式（UM）且假设理想回程链路，缺乏真实硬件验证，跨层交互与更复杂调度策略未覆盖。

---

## 609. Asymmetric Information in Dynamic Lending Games

**arXiv ID:** 2610.13759 | [PDF](https://arxiv.org/pdf/2610.13759v1)

**作者:** Kristijan Atanasov `[一作]` (King's College London), Carmine Ventre `[通讯]` (King's College London)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了有限可重复使用的供给下的动态借贷博弈，考察了信息披露对定价和收益的影响；

**💡 创新点**

发现信息不对称可导致总收益提升但总放贷不变，且信息接收者的身份随需求变化而改变收益排名；

**🔧 技术方法**

使用纯粹静态均衡（pure stationary equilibrium）分析，结合离散费率集合的博弈动力学与马尔可夫链状态转移；

**📊 数据集**

无实际数据集，所有结果基于理论模型与数值模拟；

**📈 对比分析**

通过解析阐明不同供给、需求、费率设定下的均衡比较，数值实验验证收益差异的幅度，显示信息分配可在不改变总放贷的情况下改变费率与收益分配；

**⚠️ 局限性**

局限于两位贷方、离散费率、无学习与历史信息利用，且未考虑更多贷方或连续费率区间。

---

## 610. ProvQML: Efficient Provenance Capture for Machine Learning as Recursive Aggregate SQL [Extended Version]

**arXiv ID:** 2610.13756 | [PDF](https://arxiv.org/pdf/2610.13756v1)

**作者:** Nkechi J. Akinwale `[一作]` (University of Cincinnati), Seokki Lee `[通讯]` (University of Cincinnati)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于 Apache Spark 的递归聚合 SQL 引擎，能够在机器学习模型训练过程中细粒度捕获本地与全局 provenance，从而实现模型可解释性；

**💡 创新点**

通过查询重写将相互递归、非线性递归的 ML 递归查询转化为单一线性递归，并嵌入窗口函数和 UDF last()，避免多次 shuffle；利用 PreM 与 PCC 属性证明语义一致；在 SQL 级别实现 provenance 捕获，无需外部工具；

**🔧 技术方法**

递归聚合 SQL、窗口函数、子查询、UDF last()、PreM/PCC 理论、分布式半稀疏评估、广播 delta、预分区、shuffle 优化、Spark RDD 管理等；

**📊 数据集**

四个真实数据集：Income（约 17 M 条）、BikeSharing（226 K）、SpamBase（262 K）和 WebSpam（约 29 M），均已垂直化；

**📈 对比分析**

与 Radlog 对比，测评多种 ML 模型（线性、Lasso、Ridge、Logistic、SVM）在不同特征维度、数据规模、迭代次数和节点数下的总执行时间；实验表明 provenance 捕获在大规模数据下仍保持可接受的性能，开销约 3–5 倍；优化后单次 shuffle 仅一次，速度提升约 3.4 倍；在 100 轮迭代的场景下仍能完成；

**⚠️ 局限性**

仅支持单层线性递归的 ML 任务，无法处理更深层次递归（如神经网络）；provenance 捕获导致中间结果随迭代数线性增长，存储压力较大；需手动垂直化数据；高特征维度会显著增加 shuffle 成本；目前未提供近似或分布式解释（如 Shapley）等功能；

---

## 611. SeoulMMOD: A Large-Scale Multimodal Origin-Destination Flow Benchmark

**arXiv ID:** 2610.13730 | [PDF](https://arxiv.org/pdf/2610.13730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 612. RAPID: Proactive Rate Control for Enabling High Data Rate and Low Latency in 5G Edge Networks

**arXiv ID:** 2610.13749 | [PDF](https://arxiv.org/pdf/2610.13749v1)

**作者:** Hamid Hassani `[一作]` (Eindhoven University of Technology), George Exarchakos `[通讯]` (Eindhoven University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了一种名为RAPID的用户空间传输层速率控制方案，通过调节下行发送速率以保持目标PRB利用率，从而在5G边缘网络中实现高数据速率和低延迟。

**💡 创新点**

创新点在于利用PRB利用率来间接估计基站RLC队列占用，构建简单的反馈控制循环；在用户空间实现，无需更改基站硬件；在多UE场景下与传统TCP Cubic比较，显著降低单向延迟。

**🔧 技术方法**

采用5G OpenAirInterface（OAI）原型、USRP SDR、Python实现的客户端/服务器代码；在NS3 5G-LENA模块中进行仿真；利用UDP进行数据传输，实时测量OWD；使用反馈循环参数（K、Δ、δ）进行速率调节。

**📊 数据集**

主要使用自建的测试平台和NS3仿真，未使用公开数据集；通过生成的UDP/TCP流进行评估。

**📈 对比分析**

在包含两条RAPID流和一条TCP Cubic流的多UE场景中，对比goodput、OWD和PRB利用率；RAPID实现了72倍和55倍的OWD下降，goodput仅降低16%和27%；延迟波动也显著减少，适合XR等低延迟应用。

**⚠️ 局限性**

局限性包括：对PF调度器假设的依赖，可能与专有调度器不兼容；仅验证了下行方向；在公共网络中的适用性尚未证明；未来需进一步与L4S/TCP Prague等方案对比。

---

## 613. SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning

**arXiv ID:** 2610.13737 | [PDF](https://arxiv.org/pdf/2610.13737v1)

**作者:** Hongrui Zheng `[一作]` (University of Pennsylvania), Rahul Mangharam `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 SafeDiSTL 框架，利用 STL 可靠度梯度对共享约束进行时间与代理级别的归因，并将归因成本嵌入 MARL 的策略优化流程，实现去中心化执行。

**💡 创新点**

创新点在于：① 用 STL 可靠度的梯度对整个任务轨迹进行敏感度分析；② 将增量最大成本通过归因权重拆分给各代理，既保留总成本，又生成个体化学习信号；③ 在 Lagrangian 约束框架下实现共享成本的个体化归因与策略更新。

**🔧 技术方法**

技术手段包括：Signal Temporal Logic (STL) 可靠度和光滑近似；梯度归因与归因权重归一化；Lagrangian 约束多智能体强化学习（MAPPO-style PPO + 成本优先级）；集中训练/去中心化执行；成本一致性正则化；全局与代理级成本临界值一致化。

**📊 数据集**

实验数据集：三种仿真基准——（1）协作负载运输（带障碍）；（2）连通导航（多点通过窄通道到达终点）；（3）网络交通控制（共享核心区占用约束）。

**📈 对比分析**

对比方法：MAPPO（无约束），MACPO 与 MAPPO-L（基准约束算法）。SafeDiSTL 在所有任务中获得更高的安全成功率（运输任务 1–3 个障碍时分别提升 1.6–5.1 倍，导航提升 11.1%，交通吞吐量提升 50–68%），并保持与 MAPPO 接近的任务回报，显示了优秀的安全-性能权衡。

**⚠️ 局限性**

局限性：① 需要可微分的系统动力学，接触密集场景梯度难以获取；② 归因仅考虑敏感度幅值，未考虑影响方向；③ 仅在仿真环境验证，缺乏高保真机器人实验与真实世界迁移验证。

---

## 614. State-Isolated Vision Mamba via Update-Tendency Routing for Underwater Image Enhancement

**arXiv ID:** 2610.13720 | [PDF](https://arxiv.org/pdf/2610.13720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 615. Tight asymptotic competitive ratio for the Canadian Traveller Problem on unit-weighted grids

**arXiv ID:** 2610.13712 | [PDF](https://arxiv.org/pdf/2610.13712v1)

**作者:** Pierre Bergé `[一作]` (University of Grenoble Alpes), Lou Salaün `[通讯]` (Nokia Bell Labs)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了 k‑加拿大旅行者问题（k‑CTP）在单位权重实心网格图上的在线导航算法，提出了一种确定性策略并证明其竞争比为 O(√k)。

**💡 创新点**

首次在已知的常数竞争比与线性竞争比之间给出一个中间竞争比的结果，并证明该结果在单位权重实心网格上是渐进上最优的。

**🔧 技术方法**

核心技术是“红蓝组件深度优先搜索”（RB‑DFS）以及利用网格拓扑的面积/体积论证来控制探索代价，并通过在不断增大的 L∞ 球体上递归应用该策略实现对任意大小网格的覆盖。

**📊 数据集**

论文仅基于理论分析，没有使用实际数据集；实验部分通过构造对抗实例来验证下界。

**📈 对比分析**

与传统的 Greedy 策略（竞争比线性）以及已有的常数竞争比（仅适用于特定网格如正方形或单位外平面图）相比，所提出的策略在单位权重实心网格上实现了显著更低的竞争比，且与 Ω(√k) 的下界一致，说明结果是最优的。

**⚠️ 局限性**

局限性：仅适用于单位权重且为实心的网格图；在出现任意权重或非实心结构时，竞争比退化为线性；扩展到高维网格或更一般图类仍是开放问题。

---

## 616. OpenSocialJax: Evaluating Social Adaptation in Sequential Social Dilemmas with Open-Ended Environmental Dynamics

**arXiv ID:** 2610.13710 | [PDF](https://arxiv.org/pdf/2610.13710v1)

**作者:** Zihao Guo `[一作]` (King's College London), Yali Du `[通讯]` (King's College London)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了OpenSocialJax基准，扩展Clean Up和Commons Harvest，实现数千万种可变隐藏规则与空间布局的配置，并在多次试验中评估LLM对环境与社交依赖的适应能力。

**💡 创新点**

创新点在于将隐藏环境机制与社会困境耦合，提供可重采样配置的多试验协议，使代理能够通过经验推断隐含规则并与其他代理共同适应。

**🔧 技术方法**

使用JAX实现的模拟环境、LLM代理交互协议、JSON响应格式以及多种大语言模型（GPT‑6、Claude、Qwen、Gemma、GLM等）。

**📊 数据集**

数据集基于OpenSocialJax的程序化生成，产生数千万个隐藏规则配置，每个配置在一次实验中多次试验。

**📈 对比分析**

通过同质团队和混合模型竞技场比较，评估平均回报、规则发现、清洁/资源寿命和Gini不平等；结果显示Claude Opus 5.5表现最佳，但所有模型最终仍未能维持长期合作。

**⚠️ 局限性**

局限在于仅评估LLM代理的推理与适应，未训练学习型智能体；缺乏显式通信和更高级的心智模型，难以解决共享资源耗竭和清洁集中问题。

---

## 617. Benchmarking Content-Defined Chunking Algorithms for Data Deduplication

**arXiv ID:** 2610.13704 | [PDF](https://arxiv.org/pdf/2610.13704v1)

**作者:** Marcel Gregoriadis `[一作]` (Delft University of Technology), Johan Pouwelse `[通讯]` (Delft University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地对八种基准内容定义分块算法进行基准测试，隔离出底层分块规则并统一参数化。

**💡 创新点**

提出模块化评估框架，纠正 AE 参数化错误，推导 AE、RAM、MII 的理论参数公式，并揭示高阶技术与底层算法效能混淆的问题。

**🔧 技术方法**

使用滚动哈希、极值检测、窗口极大/最小、字节比较等技术实现八种基准分块算法，并用 Rust 编写可重复的基准框架。

**📊 数据集**

采用五个真实备份与快照数据集（RAND、WEB、DB、VMB、CODE）以及随机数据作为校准基准。

**📈 对比分析**

对每个算法在四个指标（吞吐量、去重比、平均块大小、块大小方差）上进行统一实验，发现 AE 在所有指标上最稳健，Gear 在哈希基算法中性能突出，RAM 速度最快但对低熵数据敏感，MII、PCI、SeqCDC 作用有限。

**⚠️ 局限性**

实验仅覆盖基准分块算法，未考虑高阶技术（阈值、缓存、归一化等）和完整系统集成；参数校准只针对均匀随机数据，可能对不同数据分布的推广性有限。

---

## 618. Verifiable Retrieval for Legal Reasoning through Structure-driven Knowledge Graph

**arXiv ID:** 2610.13695 | [PDF](https://arxiv.org/pdf/2610.13695v1)

**作者:** Ibrahim Siddig `[一作]` (Technische Hochschule Ingolstadt), Munir Georges `[通讯]` (Technische Hochschule Ingolstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于结构驱动的知识图谱(KG)来提升德国法条检索的可验证性与语义相关性

**💡 创新点**

通过将法条结构层级映射为节点并结合上下文检索，显著提高检索的引用精确度与标记效率

**🔧 技术方法**

使用KaLM‑Embedding‑Gemma3‑12B文本嵌入、1‑hop KG扩展、不同粒度(chunking)的分块策略与结构化召回指标

**📊 数据集**

利用OpenLegalData（BGT法官判决）和官方德国法典文本构建KG，评估集包括12037个BGT案例

**📈 对比分析**

在各种粒度（Absatz、Satz、Nummer等）和扩展策略（Parent、Citation、Hybrid）下，对比ToER指标，最优Nummer与Satz粒度在不同深度上分别达约14.23和13.59的ToER，明显优于100词无结构基线

**⚠️ 局限性**

ToER在深层次（句子、编号）下降显著，说明检索任务难度大；且结果未直接验证对下游生成质量或专业律师判断的影响

---

## 619. Variance-Reduced Policy Gradient is Nearly Minimax-Optimal

**arXiv ID:** 2610.13692 | [PDF](https://arxiv.org/pdf/2610.13692v1)

**作者:** Leonardo Cesani `[一作]` (Politecnico Di Milano), Marcello Restelli `[通讯]` (Politecnico Di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了一类无第二阶信息的第一阶策略梯度方法，提出 Loopless Variance-Reduced Policy Gradient 框架，并在弱梯度支配下证明其样本复杂度达到近似最优 O(ε⁻²)。

**💡 创新点**

创新点在于：① 统一并抽象了现有防御重要性采样的第一阶估计器；② 在不假设重要性权重方差均匀有界的前提下，给出可行参数空间，实现最优样本复杂度；③ 通过构造下界证明了 Ω(ε⁻²) 的不可超越性。

**🔧 技术方法**

使用了递归（loopless）变差减少、defensive importance sampling、随机刷新机制、投影梯度上升、弱梯度支配分析、以及下界构造等技术。

**📊 数据集**

实验使用了经典控制任务 CartPole-v1 以及 MuJoCo 运动任务 Swimmer-v4。

**📈 对比分析**

与 A2C、SNPG、NPG 等基线方法对比，defensive‑momentum 版本（α>0, p=0）在样本效率上更优，收敛更快，且在 CartPole-v1 上振荡更小、在 Swimmer-v4 上更快达到局部最优。

**⚠️ 局限性**

局限性包括：① 需要在混合采样中存储前后策略参数，增加内存开销；② 尚未考虑自适应优化器；③ 实验仅覆盖两小规模任务，缺乏对更复杂环境的验证。

---

## 620. VQIL: Visible Quantum Interference Learning for Perturbation Response Ranking on Graphs

**arXiv ID:** 2610.13689 | [PDF](https://arxiv.org/pdf/2610.13689v1)

**作者:** Seokjun Lee `[一作]` (University of Cambridge), Namshik Han `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一种可见量子干涉学习模型VQIL，使用可训练的Szegedy量子游走实现种子节点的响应排序，并在电网功率流扰动和基因蛋白互作网络中进行了评估。

**💡 创新点**

创新点在于将可训练的量子干涉与可见参数化相结合，学习节点相位与边权并通过几何折扣的相干求和读取提升训练平滑性，从而实现对传播机制的可解释性。

**🔧 技术方法**

采用端到端可微的Szegedy量子游走、可训练的边权和节点相位、几何折扣相干读出、软Spearman损失和AdamW优化。

**📊 数据集**

使用了六个标准配电网（30–1354节点）进行AC功率流扰动，以及五个Perturb-seq基因扰动数据集（2950–5787节点）对应的蛋白互作网络。

**📈 对比分析**

通过与随机游走重启、GCN、GAT、O(2) sheaf等基线在Spearman相关、AUPRC、AUROC、NDCG等指标上比较，VQIL在电网和基因网络上显著优于基线，Spearman提升至0.8–0.9（电网）和0.62（基因网络）。

**⚠️ 局限性**

局限性包括仅适用于固定拓扑的传导任务，无法泛化到未见图；仅学习无符号幅度，未能处理有向网络或连续响应值预测。

---

## 621. X-AVDD: Cross-Attentive Audio-Visual Dataset Distillation

**arXiv ID:** 2610.13687 | [PDF](https://arxiv.org/pdf/2610.13687v1)

**作者:** Saumyaranjan Mohanty `[一作]` (Indian Institute of Technology Hyderabad), Konda Reddy Mopuri `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 X-AVDD 的音频-视觉数据集蒸馏方法，用于从大规模配对音频-视频数据中合成小型且保持高训练效能的数据集。

**💡 创新点**

创新点在于：1) 采用双向跨注意力捕捉跨模态关系；2) 将对齐过程与样本合成解耦，显著降低显存需求；3) 通过欧式分布匹配损失实现跨模态语义对齐，从而大幅减少“模态差距”。

**🔧 技术方法**

核心技术包括双向跨注意力机制、欧式分布匹配 (EDM) 损失、预训练教师网络以及基于交叉熵与软标签的学生训练。

**📊 数据集**

实验使用四个主流音视频数据集：AVE、MUSIC-21、VGGSound 以及 VGGSound‑10K。

**📈 对比分析**

与 DM、AVDD 及改进版 SRe^2L 等基线在 ConvNet、ResNet‑18、ViT、AudioCLIP 等不同骨干上进行比较；X-AVDD 在 10–50 samples-per-class (SPC) 下平均提升 20–40% 的 top‑1 准确率，同时将内存占用和合成时间降低约 8 倍，跨架构和冻结预训练模型的迁移性能也均优于基线。

**⚠️ 局限性**

局限性在于仅针对音视频分类任务验证；对时间定位、检索等其他下游任务的适用性和性能仍待进一步研究。

---

## 622. Enhancing Speech Recognition through Text-to-Speech and Voice Conversion Augmentation

**arXiv ID:** 2610.13662 | [PDF](https://arxiv.org/pdf/2610.13662v1)

**作者:** Yunus Emre Ozkose `[一作]` (Sestek), Ali Haznedaroglu `[通讯]` (Sestek)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在银行呼叫中心场景中，评估了真实语音、TTS合成语音和三种高级语音转换（KNN-VC、Seed-VC、Vec2Wav2）对自动语音识别（ASR）性能的影响。

**💡 创新点**

首次在特定行业语料下进行域级实验，比较多种合成技术，并通过NISQA和WER/CER量化其对ASR的提升，展示Seed-VC在短语音样本下的优越性。

**🔧 技术方法**

采用Conformer/Zipformer ASR架构，BPE词表，K2/Icefall训练框架，Azure TTS合成，KNN-VC、Seed-VC、Vec2Wav2语音转换，以及NISQA评估模型。

**📊 数据集**

使用5小时真实呼叫中心语音、20小时基于真实转写的Azure TTS合成、从Librispeech 1281位说话人衍生的语音转换样本，以及10/20/50小时的训练配置。

**📈 对比分析**

通过WER/CER以及NISQA MOS对比，真实数据最低WER 34.40%，TTS最高97.10%；Seed-VC以75.53% WER、43.97% CER领先，Vec2Wav2稍逊，TTS+语音转换进一步降低至28.65%~30.26%，但随着训练时长增大增益递减。

**⚠️ 局限性**

合成语音仍无法完全匹配真实多样性，过度转换会引入噪声；实验仅限于土耳其语呼叫中心，缺乏跨语言验证，且对模型泛化能力与资源消耗未作深入评估。

---

## 623. Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy

**arXiv ID:** 2610.13708 | [PDF](https://arxiv.org/pdf/2610.13708v1)

**作者:** Naji Alazhar `[一作]` (Umans AI), Wassel Alazhar `[通讯]` (Umans AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Adaptive DHiSparse 系统，在长上下文推理中将多模型草稿（draft）推理与层次稀疏 KV 缓存结合，支持根据负载动态调整 KV 驻留策略。

**💡 创新点**

创新点在于：① 将 DSpark/DFlash2 的并行/顺序验证机制引入层次稀疏 KV 缓存；② 引入可选历史驻留，可在 GPU 空闲时预留更多历史；③ 共享检索与可重用验证阶段，使多查询验证共享同一条缺失行；④ 通过窗口束缚的草稿 KV 和可重用 staging 控制额外 GPU 内存使用。

**🔧 技术方法**

采用技术包括：层次稀疏 KV 缓存、HiSparse 改进、DSpark 与 DFlash2 草稿模型、LRU 缓存、可选历史驻留、共享检索、窗口束缚草稿 KV、可重用验证 staging、GPU‑主机双缓冲、精确层级检索等。

**📊 数据集**

使用的数据集为 GLM‑5.3（100K 上下文）和 DeepSeek‑V4.1‑Flash（500K 上下文）两种稀疏注意力模型，实验基于预置前缀与固定输出长度的闭环负载基准。

**📈 对比分析**

通过在 8×B300 节点上对比 Adaptive DHiSparse、HiSparse、驻留服务与非推测 Adaptive Serving，采用每个 decode GPU 的输出吞吐量、TTFT 与 TPOT 等指标。结果显示：在 GLM‑5.3 100K 上下文、C128 时，Adaptive DHiSparse 达到 793 tokens/s/ GPU，吞吐为 HiSparse 的 1.45×；在 DeepSeek‑500K、C512 时，Adaptive DHiSparse 的吞吐为驻留服务的 1.85×。在相同延迟下，Adaptive DHiSparse 可支持 4 倍并发，吞吐提升约 2.3–3.7 倍。

**⚠️ 局限性**

局限性：① 仍受 GPU 内存容量限制，负载动态调整带来额外开销；② 在高负载下可选历史会被回收，导致性能波动；③ 草稿模型的接受率不确定，影响推测收益；④ 多 GPU 之间的内存管理和调度复杂度高；⑤ 实现和调优成本较大。

---

## 624. Ask Your Advisor: Learning When to Consult Stronger Models in Agentic Systems

**arXiv ID:** 2610.13870 | [PDF](https://arxiv.org/pdf/2610.13870v1)

**作者:** Pascal Sowietzki `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出一种名为ARIA的元控制器，用于在执行器-顾问体系中学习何时向更强大的顾问请求帮助，从而实现任务性能与计算成本的平衡。

**💡 创新点**

创新点在于：①基于执行过程中的可观测状态构建紧凑的24维状态抽象；②使用无模型参数的Monte Carlo控制学习咨询策略；③通过单一价格参数在不同任务中灵活调整成功与成本的权衡。

**🔧 技术方法**

采用的技术包括：强化学习中的蒙特卡罗控制（无模型、无微调）、状态抽象（时间位置、执行证据、咨询历史）、成本感知回报函数以及在Magnetic-One框架下的executor‑advisor交互。

**📊 数据集**

使用的数据集涵盖三类代理工作流：软件工程任务（SWE‑bench Verified）、终端操作任务（Terminal‑Bench 2.1）和交互客服任务（τ²‑bench）。

**📈 对比分析**

与Executor‑only、Advisor‑only、FrugalGPT、RouteLLM、AutoMix、Plan、Heuristic、Always、LLM‑Router及Random等十种对照方法相比，ARIA在三大基准上分别恢复了44–81%的执行器‑顾问性能差距，同时成本仅为顾问单独使用的一半左右，位居所有方法的成功‑成本 Pareto 边界上。

**⚠️ 局限性**

主要局限包括：仅在单一模型对（低成本执行器 + 高成本顾问）与Magnetic-One框架下验证，缺乏跨模型、跨框架的泛化；未直接对预算做显式约束；在执行器未充分采纳顾问建议时仍可能导致误差积累；对高度异质任务的适配性仍待进一步研究。

---

## 625. BT-Agent: Deploying Onboard Robot Agents with Behavior Trees and Multi-Fidelity Evaluation

**arXiv ID:** 2610.13862 | [PDF](https://arxiv.org/pdf/2610.13862v1)

**作者:** Fabio Hübel `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了BT-Agent架构，结合LLM驱动的任务决策与预定义行为树技能，实现了移动机器人在有限计算和内存条件下的全机上任务推理与反应式执行，并在Boston Dynamics Spot四足机器人上完成了真实部署；

**💡 创新点**

创新点包括：①将任务级LLM推理与行为树执行分离，行为树提供局部条件检查、即时中断和结构化失败反馈；②构建了161任务多保真度基准，将文本、仿真与真实机器人任务对齐；③在实机上实现了9B LLM全机上推理与导航、操作的端到端运行；

**🔧 技术方法**

采用的大型语言模型（Qwen3.5 4B–122B），行为树（BT）技能，ReAct推理循环，结构化工具调用与失败反馈，文本-仿真映射框架，Habitat/Isaac仿真环境，Robosuite物理实验，Jetson Thor嵌入式平台，视觉感知与导航控制模块；

**📊 数据集**

使用了161任务文本基准（包含50个主任务与111个迁移任务），任务与技能接口与Habitat、Isaac仿真场景对齐；实地测试的5个任务；FARM场景图与RGB‑D映射记录作为输入；

**📈 对比分析**

与Act、ReAct、CaP-X等基线在同一任务集（50/111/161任务）上进行对比，评估成功率、推理与执行时间。BT-Agent在4B/9B模型上分别比最强基线高约10%和5%的成功率；在9B模型下，文本与仿真成功率相关性ρ=0.61；文本评估速度是Habitat的10.9倍；实机部署成功率约73%（11/15）；

**⚠️ 局限性**

局限性：技能实现与守护在运行时固定，无法动态扩展；对不完整或模糊观测的证据传递仍不完善；任务级推理未与BT监督并行，难以充分利用树结构失败信息；极端失败情境下的诊断与恢复能力有限；需要手工编写新技能与平台集成。

---

## 626. On the Design Space of Continuous Diffusion Language Models

**arXiv ID:** 2610.13834 | [PDF](https://arxiv.org/pdf/2610.13834v1)

**作者:** Oscar Davis `[一作]` (University of Oxford), Marco Cuturi `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究连续扩散语言模型（CDLM），通过自条件化与令牌可识别性提升模型在文本生成上的表现。

**💡 创新点**

创新点在于提出双连续腐蚀方法以模仿自条件化、以及半连续分类流（semi‑continuous categorical flow）解决令牌可识别性问题，并对嵌入空间的训练进行了深入分析。

**🔧 技术方法**

主要技术包括：连续扩散框架、双连续腐蚀、半连续分类流、自条件化、嵌入空间设计、解码误差率（DER）分析、以及基于贝叶斯后验的状态表示。

**📊 数据集**

使用的数据集包括 TinyGSM、GSM8K、OpenWebText 以及多项选择问答（MCQA）基准（ARC‑easy、BoolQ、OBQA、PIQA、RACE、SIQA）。

**📈 对比分析**

与离散扩散基线、混合方法（CANDI、FMLM+）、一热流（one‑hot flow）等对比，TinyGSM 上准确率从 4.42% 提升至 16.00%（可达 32.90%），MCQA 上多任务零样本准确率超过离散扩散基线，且在无条件生成中实现更优的 /Entropy Pareto 前沿。

**⚠️ 局限性**

局限性包括：需要精细的噪声时间表、在高维嵌入空间下训练不稳定、某些方法仍需额外前向推理步骤，以及对大规模模型的加速与进一步理论分析尚待解决。

---

## 627. SpatialASM: Internalizing Active Mentalization for Spatial Reasoning

**arXiv ID:** 2610.13832 | [PDF](https://arxiv.org/pdf/2610.13832v1)

**作者:** Yuhong Liu `[一作]` (Chinese University of Hong Kong), Dahua Lin `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出通过内部化主动空间心理化（ASM）来提升大型视觉语言模型（LVLM）的空间推理能力。

**💡 创新点**

创新点在于：①将ASM拆分为视角规划和视角想象两大能力；②利用 allocentric novel‑view generator (NVG) 在训练阶段生成视角图像并通过 Geometry Perturbation (GeoP) 过滤实现高质量监督；③自动合成 ASM 链式思维轨迹与基于视角想象的强化学习任务，实现无需人工标注的中间推理指导。

**🔧 技术方法**

采用 allocentric 视角表示、基于 FLUX.2‑klein‑9B 的 NVG、Geometry Perturbation (GeoP) 一致性度量、SFT+GRPO 两阶段后训练，以及多模态预训练模型 Qwen3‑VL/Qwen3.5。

**📊 数据集**

使用公开场景数据集 DL3DV、HoliSpatial、VST‑500k、MindCube 等构建 SpatialASM‑edit140k、SpatialASM‑128k、SpatialASM‑RL‑58k 等数据集。

**📈 对比分析**

与基线模型和现有空间推理模型对比，四种 Qwen 系列模型在七大空间推理基准上平均提升 1.2–3.5 分，最大单项提升达 +8.7 分；在通用 VQA 任务上保持误差 <0.6%。

**⚠️ 局限性**

局限在：①对极端大视角变化的几何一致性仍有限；②需要额外训练步骤和 NVG 资源；③在动态或交互式场景下的泛化尚未验证。

---

## 628. Same Query, Different Answer: Run-to-Run Variance in Semantic Query Processing

**arXiv ID:** 2610.13823 | [PDF](https://arxiv.org/pdf/2610.13823v1)

**作者:** Oktie Hassanzadeh `[一作]` (IBM), Dharmashankar Subramanian `[通讯]` (IBM)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统测量并分析大语言模型在语义查询引擎中的非确定性，评估多次执行对查询答案、计划、基准和优化器的影响，并探讨可复现性改进方法。

**💡 创新点**

创新点在于提出可能世界统计模型预测答案再现率，结合多模型、多部署、多引擎的项级输出日志，揭示计划放大变异的机制，并提供投票、选择性重访和记录输出等复现策略。

**🔧 技术方法**

技术手段包括：可能世界概率模型、执行日志采集、投票与选择性重访、基于决策边际的过滤、校准级联与不确定答案估计。

**📊 数据集**

使用的数据集为 SemBench（Movie、Medical 查询）、LRO-Bench（27个数据库的5种实现）以及 UDA-Bench 的 Player Wiki 文档集合。

**📈 对比分析**

通过比较不同引擎、模型部署的质量范围、答案再现率、项翻转率等指标，发现温度0并不能消除变异；某些计划会显著放大项级变异，投票能降低错误但无法完全消除；实验显示不同部署对答案稳定性影响显著。

**⚠️ 局限性**

限制包括：仅覆盖文本与表格数据；只测试六个开源模型和两商业模型；无法控制模型内部温度、批处理和部署负载；基准缺乏完整的项级日志；实验规模有限，结果可能不具普适性。

---

## 629. Dream to Remember: Continual Steering of Generalist Robot Policies with World Models

**arXiv ID:** 2610.13816 | [PDF](https://arxiv.org/pdf/2610.13816v1)

**作者:** Bahey Tharwat `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种名为 Dream to Remember (D2R) 的框架，利用冻结的生成式机器人策略和世界模型在噪声空间进行输入调度，并通过在世界模型中回放旧任务来防止灾难性遗忘；

**💡 创新点**

创新点在于将噪声空间调度与冻结世界模型回放相结合，既不需要更新主策略权重也不需存储轨迹，同时通过 KL 正则化实现持续学习，并证明在不同逼真度的世界模型下可实现无环境交互的任务迁移；

**🔧 技术方法**

采用的技术包括输入噪声调度（steering）、基于 diffusion 或 flow‑matching 的冻结生成式策略、冻结世界模型（RSSM / A2World）、基于强化学习（PPO/SAC）在目标任务中学习，以及 KL 正则化来保持旧任务行为；

**📊 数据集**

使用的数据集包括 CALVIN（8 个任务、play 数据训练的世界模型）、LIBERO‑90（8 个新任务与 4 个已预训练套件）、以及在 Franka 机器人上进行的实际实验，利用公开的 π₀ 策略和 A2World 世界模型；

**📈 对比分析**

与冻结策略、DiWA、LoRA、DP‑TTC、DSRL 等基线在单任务和连续任务适应中对比，D2R 在 CALVIN 和 LIBERO 上平均成功率提升至约 75%（相较基线提升约 11%），且仅出现 5–8 点的低遗忘率；在 Franka 实验中实现 30% 的平均成功提升并保留所有增益；

**⚠️ 局限性**

局限性在于仍需为每个任务存储初始观测帧以供世界模型回放，对极端任务变化或动态场景的鲁棒性有限，且当世界模型逼真度不足时仍需在真实环境中进行训练。

---

## 630. Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion

**arXiv ID:** 2610.13810 | [PDF](https://arxiv.org/pdf/2610.13810v1)

**作者:** Amit Kumar Jaiswal `[一作]` (Indian Institute of Technology (BHU)), Prayag Tiwari `[通讯]` (Halmstad University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SAFE-QE，一种无训练、检索器无关的LLM查询扩展过滤框架

**💡 创新点**

将分子关联与一致性-宽度稳定性理论引入查询扩展，结合子序列关联滤波和子模优化选择

**🔧 技术方法**

使用LLM生成候选、交叉上下文可重现性、语料库支持评分、子模最大覆盖、频率稳定性筛选

**📊 数据集**

在TREC Deep Learning 2019/2020及BEIR四个数据集上评测，使用BM25、SPLADE、Contriever三种检索器

**📈 对比分析**

与Query2doc、HyDE、CSQE、MILL等方法对比，nDCG@10保持相当，HETR显著下降（≈85%/93%），负载更低

**⚠️ 局限性**

局限在于固定的语料库支持阈值需针对检索器调优、缺乏子模选择的离散分析、未覆盖所有生成器规模的实证

---

## 631. Your Discrete Graph Diffusion Model is Secretly Memorizing

**arXiv ID:** 2610.13786 | [PDF](https://arxiv.org/pdf/2610.13786v1)

**作者:** Adarsh Jamadandi `[一作]` (CNRS and IRISA), Nicolas Keriven `[通讯]` (CNRS and IRISA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了单图扩散模型在逆扩散过程中的“条件性记忆”现象，即模型能够在以训练图的噪声版本为起点时完美恢复训练图，但在无条件采样时不会出现记忆；

**💡 创新点**

创新点在于提出并量化了条件性记忆，并通过U‑turn实验和边回忆率来揭示传统样本级度量（如WL核相似度、聚类系数）对记忆的盲区；

**🔧 技术方法**

使用离散图扩散模型（EDGE）作为基础，并引入U‑turn探测、边回忆率、WL核相似度、聚类系数等评估技术；

**📊 数据集**

实验数据集为从两个块的随机块模型（SBM）生成的合成图，节点数取64、128、256、512，保持可检测性阈值和平均度数固定；

**📈 对比分析**

与传统样本级指标对比，发现样本级度量在训练期间表面显示良好泛化（聚类系数改进、WL相似度保持平稳），但U‑turn实验揭示了显著的记忆偏差；测试误差下降并不对应社区结构的出现，说明仅靠测试误差无法选择优质模型；

**⚠️ 局限性**

局限性包括仅在合成SBM数据上验证，缺乏对真实网络的验证；模型仅在单图训练场景下评估；早停法无法避免记忆阶段；未探讨多图训练或更复杂块结构时的记忆行为。

---

## 632. Tracking the Invisible: Self-Supervised Wheel Track Detection with Thermal-RGB Fusion

**arXiv ID:** 2610.13872 | [PDF](https://arxiv.org/pdf/2610.13872v1)

**作者:** Yiming Yang `[一作]` (Michigan Technological University), Jeremy P. Bos `[通讯]` (Michigan Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种低成本的自监督RGB-热红外融合框架，用来检测雪覆路面上的车轮痕迹，从而实现冬季无人驾驶的安全导航；

**💡 创新点**

创新点包括：利用GNSS轨迹投影自动生成稀疏车轮痕迹标签，融合RGB与LWIR四通道输入，结合注意力门与多尺度空洞卷积提升对极低对比度环境的识别能力；

**🔧 技术方法**

采用的技术包括：ResNet-18编码器+注意力门+空洞卷积解码器，焦点损失+边界损失，CLAHE图像增强和静态单一homography对齐；

**📊 数据集**

数据集为WADS的子集ThermalTrack，共7608张经数据增强的RGB-I图像；

**📈 对比分析**

与UNet、PSPNet、DeepLabv3+等基线比较，所提方法在AUROC、MaxF、AP等指标上均领先，尤其在热红外输入下FNR下降近70%，FPS保持约26Hz；

**⚠️ 局限性**

局限性在于样本量有限、仅覆盖特定雪况（细雪、结冰），对其他极端条件如深粉雪、熔化雪等泛化能力待验证；

---

## 633. Partially Informed Stochastic Optimization via Zeroth-Order Residual Tracking

**arXiv ID:** 2610.13871 | [PDF](https://arxiv.org/pdf/2610.13871v1)

**作者:** El Mahdi Chayti `[一作]` (EPFL), Martin Jaggi `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种Partially Informed Stochastic Optimization（PISO）框架，用于处理梯度可分为已知和不可知两部分的非凸优化问题；

**💡 创新点**

创新点在于通过跟踪不可知残差并以此为中心进行零阶梯度估计，显著降低了因随机探测导致的方差，并在不需要对不可知梯度做统一上界的前提下实现收敛保证；

**🔧 技术方法**

主要技术包括残差跟踪、控制变量（error‑feedback）估计、随机草图迭代、一阶与零阶信息的混合动量以及对probe分布的高阶矩分析；

**📊 数据集**

实验使用了多元定价（10维）、信用评分（12维）、8×8格网世界、225个路由实例、427个安全游戏实例等公开或仿真数据集；

**📈 对比分析**

与同类方法（GZO‑NS/H, ZO‑OGVR, ZO‑TG, ZOS, PZOS、Vanilla PG、PePG等）在相同oracle访问量下比较，PISO在多数设置下取得更快收敛、更优解或两者兼备；

**⚠️ 局限性**

局限性包括仅适用于光滑目标（对非光滑的Goldstein收敛尚未证明）、对probe分布的独立性假设（如正交循环探测未被理论覆盖）以及在某些performative场景中仍需更大批量采样以降低部署次数。

---

## 634. Returning to Development States: When Full Sandbox Snapshots Help Coding Agents

**arXiv ID:** 2610.13841 | [PDF](https://arxiv.org/pdf/2610.13841v1)

**作者:** Daniel Thi Graviet `[一作]` (Daytona), Tanmay Shah `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了将完整沙盒快照作为编码代理的状态恢复方式，并在Go‑Explore框架下实现全机器快照搜索，比较其与传统从初始状态重试的效果。

**💡 创新点**

创新点在于将完整环境快照引入搜索状态，系统性识别何时快照比文本或源文件恢复更有效，并分析快照搜索的主要失败模式。

**🔧 技术方法**

使用了Daytona沙盒快照、Go‑Explore搜索策略、Claude Haiku 4.5模型、Terminal‑Bench 2.0 benchmark以及Harbor环境。

**📊 数据集**

实验使用了Terminal‑Bench 2.0子集（5个任务×5种子）以及自定义SQLite数据库任务。

**📈 对比分析**

在相同推理预算下，独立重试在5个任务中实现17/25（68%）成功率，而快照搜索仅6/25（24%）；但在需要外部状态的任务（如SQLite）中，快照恢复显著提升成功率（4/5）。

**⚠️ 局限性**

局限性包括预算碎片化、错误状态吸引、代理交接成本、状态选择信号弱等，导致快照搜索在文件中心任务上效果不佳。

---

## 635. Specialists Learn Faster: Knowledge Organization in Pre-training Shapes Knowledge Update Efficiency

**arXiv ID:** 2610.13835 | [PDF](https://arxiv.org/pdf/2610.13835v1)

**作者:** Sheng Zhao `[一作]` (University of Rochester), Yuhao Zhu `[通讯]` (University of Rochester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了预训练时知识的时间顺序如何影响模型后续的知识更新效率，并提出了针对历史与当代知识混合训练的新策略；同时推导了更新成本的经验定律。

**💡 创新点**

创新点包括①发现时序预训练生成的专家模型（专门化）比随机预训练生成的通用模型需要 27%-58% 更少的更新步数；②用初始知识缺口和梯度速率推导出更新步数的定律；③提出将历史知识随机打乱、近期知识保持顺序的混合训练，兼顾历史保留与更新效率。

**🔧 技术方法**

使用 GPT‑2 风格解码器、LoRA 适配器进行知识更新，结合 ROME、MEMIT、AlphaEdit、ENCORE 等编辑工具；通过梯度、损失曲面与互信息分析解释现象，并在持续预训练阶段验证策略。

**📊 数据集**

主要数据集为：①包含 25,000 人物、6 个属性、时间轴的合成传记数据；②基于 Gemma‑2‑9B‑it 的真实预训练模型；③版本化软件工具调用数据，用于验证在非时间序列知识上的可迁移性。

**📈 对比分析**

对比实验包括在 3 个模型规模（110M、350M、682M）和 6 种预训练语料（顺序/打乱+时间戳）下，测量覆盖 overwrite 与 inject 两种更新的步数；专家模型平均比通用模型节省 27%-58% 步；Mix‑C 模型在历史（dated）与当前（undated）查询上均保持高准确率，同时更新步数比 Mix‑S 低 16%-44%，遗忘率低 5%-57%；编辑器在专家组织下进一步降低了 46%-63% 的更新成本与遗忘。

**⚠️ 局限性**

局限性在于：①主要验证于合成传记和软件版本两个领域，对其他类型知识的推广尚未证明；②互信息对专家/通用模型出现的定量影响尚未深入研究；③实验集中在梯度优化的更新方法，未涉及无梯度的进化或强化学习情境；④预训练时间与资源消耗仍高，实际部署需进一步评估。

---

## 636. Not Just an Evaluation: A Case Study of Discourse Particle Sensitivity in Language Models

**arXiv ID:** 2610.13794 | [PDF](https://arxiv.org/pdf/2610.13794v1)

**作者:** William Sheffield `[一作]` (University of Texas at Austin), Kanishka Misra `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型（LLM）对英语对话粒子（just、only）在自然对话中的敏感性，设计了人类与模型的偏好比较、基于生成续句的概率对比实验以及激活补丁定位的机制分析，探究粒子对后续话语的影响与内部表示；

**💡 创新点**

创新点在于：①首次将对话级粒子敏感性与人类偏好进行对齐评估；②结合自生成续句与概率差测度，展示粒子对生成空间的细微影响；③通过对注意力、MLP、残差流等层组件的补丁与因果干预，定位并揭示粒子敏感性的非对称共享结构；

**🔧 技术方法**

技术手段包括：使用DailyDialog的自然对话最小对照对构造CS/CS'，对LLM进行log‑probability差异测度（Δ_removal、Δ_tok）；采用核采样（top‑p 0.9）生成50条续句并再评估其在两种上下文下的概率；对Llama 3 8B Instruct进行激活补丁定位与必要性干预，评估残差流、注意力、MLP层的因果贡献；

**📊 数据集**

数据集为DailyDialog对话，筛选出含just/only/not的句子，得到1182 just、360 only、1307 not实例；随后人工挑选60个金标准样本（每种粒子20例，just分为exclusive与non‑exclusive各10例）供实验使用；

**📈 对比分析**

与人类偏好进行对比，使用Spearman ρ衡量相关性；模型在not、just上与人类高度一致，only相对弱；指令调优后敏感性显著提升；生成实验显示粒子导致的概率差约为真值条件改动的十分之一，说明影响更细微；激活补丁实验表明残差流与注意力层的特定单元对粒子敏感性具有显著因果作用，only对exclusive just的转移效应更强；

**⚠️ 局限性**

局限性包括：只研究英语的just、only两颗粒子；使用8B参数规模模型，未检验更大或更小模型；数据采样人为挑选，可能存在偏见；实验3仅在Llama 3 8B Instruct上进行，缺乏跨模型的验证；粒子对话样本有限，可能不足以覆盖所有语用变体。

---

## 637. OccludeBench: Can VLMs Find What's Missing?

**arXiv ID:** 2610.13790 | [PDF](https://arxiv.org/pdf/2610.13790v1)

**作者:** Jasin Cekinmez `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建OccludeBench数据集，评估VLM在直接（阴影/反射）与推理（上下文）遮挡物识别上的表现，并提出四阶段诊断方法。

**💡 创新点**

发现VLM在直接视觉线索上显著劣于人类，定位关键瓶颈为在完整图像中从投影中提取目标；提出inpaint-then-confront干预显著缩小差距。

**🔧 技术方法**

利用VLM与图像工具调用、推理、提示工程、inpainting等技术，对模型进行多维度干预实验。

**📊 数据集**

使用378张人工标注的图像（197直接、181推理），来源包括互联网、Visual Genome及自拍照。

**📈 对比分析**

与人类基线比较，VLM平均准确率52.3%（直接低于推理20.0pp），人类65.9%；在inpaint-then-confront后，GPT‑5.4直接准确率提升至54.3%，Gemini 3.1 Pro提升至82.2%，差距不显著。

**⚠️ 局限性**

模型对直接线索的视觉提取能力有限；缺乏图像生成能力时干预效果差；数据集规模有限，难以覆盖所有遮挡类型。

---

## 638. Scheduling in Cloud,Fog, and Edge Computing: A Comparative Analysis and Decision Framework for Technique Selection

**arXiv ID:** 2610.13830 | [PDF](https://arxiv.org/pdf/2610.13830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 639. Revisiting the Task of Building Damage Assessment by Satellite Imaging

**arXiv ID:** 2610.13857 | [PDF](https://arxiv.org/pdf/2610.13857v1)

**作者:** Anthony Cioppa `[一作]`, Marc Van Droogenbroeck `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新定义了建筑损毁评估任务，系统梳理数据、评估和方法三大支柱。

**💡 创新点**

创新点在于提出了无传感器依赖、可容忍无前期影像的实例化任务定义，并构建了多指标、排名基础的评估协议。

**🔧 技术方法**

采用深度学习模型（包括任务专用网络和地球观测基础模型）以及多模态自监督预训练（MAE、MAE变体）进行实验。

**📊 数据集**

使用了公开的 xBD、BRIGHT、xFBD、Sun 等多灾种、多模态数据集进行训练与测试。

**📈 对比分析**

通过对比传统基准和基础模型，发现即使在多模态下，零样本迁移仍低于 40% mIoU，提示泛化能力有限；基础模型在特定任务上可达 80%+ F1。

**⚠️ 局限性**

主要局限在于任务定义不统一导致评估不一致、数据分布偏差与地理泄漏、以及对真实灾害中缺失前影像和多模态协同的处理仍不足。

---

## 640. X-AgentRed: Red-Teaming Personal Agents in Evolving Multi-Service Ecosystems

**arXiv ID:** 2610.13886 | [PDF](https://arxiv.org/pdf/2610.13886v1)

**作者:** Yeonsung Jung `[一作]` (Meta Reality Labs Research), Ninareh Mehrabi `[通讯]` (Meta Reality Labs Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 X-AgentRed，面向多服务个人智能体的自演化红队框架，利用工具依赖图（HarmFlow）生成潜在有害交互路径，并通过执行反馈进行路径多样化与修正，进一步发现跨服务的意外或有害行为。

**💡 创新点**

创新点在于：①将工具执行前置条件、效果与潜在危害转化为显式的有向图，实现跨服务依赖建模；②通过图采样生成“配方”并实例化为实际情景，避免仅依赖 LLM 生成；③利用执行反馈实现对成功路径的多样化和失败路径的预置条件细化，形成闭环自优化。

**🔧 技术方法**

核心技术包括：图结构建模（HarmFlow 依赖图）、随机反向走访采样、情景实例化、LLM 判别（判断是否产生危害）与执行反馈驱动的图更新与配方重构。

**📊 数据集**

使用的评测数据集为模拟多服务生态系统：30 种服务、914 种工具；测试目标为四款前沿代理模型（GPT‑5.6、GLM‑5.2、DeepSeek‑V4‑Pro、Claude Opus 4.8）。

**📈 对比分析**

与 STAC 与 TRACE 两种基线在同一环境下进行对比；X‑AgentRed 在四个目标代理上平均提升 1.77×（对 STAC）和 10.36×（对 TRACE）的有害路径率（HPR），并覆盖 2.99×（STAC）和 1.84×（TRACE）的工具；在成功场景数、工具与服务多样性方面亦显著优于基线。

**⚠️ 局限性**

局限性包括：维护 HarmFlow 图随服务与工具变更需持续更新，导致额外计算开销；评测依赖模拟环境和 LLM 判断，真实部署时的效能与可解释性仍待进一步验证。

---

## 641. Make It Flow: Animating Water in 3D Gaussian Splatting Scenes

**arXiv ID:** 2610.13860 | [PDF](https://arxiv.org/pdf/2610.13860v1)

**作者:** Kjell Keune `[一作]` (TU Braunschweig), Marcus Magnor `[通讯]` (TU Braunschweig)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在静态3D Gaussian Splatting（3DGS）重建的场景中，通过将2D光流提升到3D欧拉运动场并应用于高斯原语，生成无缝循环的水动画；

**💡 创新点**

提出了将光流投影为3D欧拉运动场的高效融合方法，并利用无偏变换和对称动画技术实现高质量、实时循环水动画；

**🔧 技术方法**

使用光流估计网络、稀疏/稠密光流插值、深度渲染、体素格子融合、无偏变换、对称动画和随机状态偏移等技术；

**📊 数据集**

构建了七个在野外拍摄的瀑布和溪流场景数据集，包含相机参数、重建的3DGS模型以及运动提示和分割掩码；

**📈 对比分析**

与4DGS、LoopGaussian、3D-Cinemagraphy等基线对比，表现为>300fps实时渲染、约3.8GB显存、预处理约1分钟，视觉效果优于对比方法；

**⚠️ 局限性**

受限于静态重建质量，难以捕捉缺失的细节和反射水面；运动仅限于已重建的表面；需要进一步动态更新运动场以实现交互和物理响应。

---

## 642. MC-EvoBench: Auditing Experience-Driven Self-Improvement in Minecraft Agents

**arXiv ID:** 2610.13829 | [PDF](https://arxiv.org/pdf/2610.13829v1)

**作者:** Jingwei Song `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MC‑EvoBench，一个基于 Minecraft 的可复现基准，用于评估体验是否能产生持久、可转移的具身能力。

**💡 创新点**

创新点在于通过任务状态图、阶段级验证器和四种对照源条件的配对协议，将能力提升归因于经验而非资源携带或偶然成功。

**🔧 技术方法**

采用任务状态图、图权重和阶段验证器的组合，配合控制源实验与目标重置，计算 TSR、WGP、VMP 三种指标。

**📊 数据集**

使用自定义的 Minecraft 任务库（80 个任务、111 个状态、369 条边，19 条链），并在公开的 Minecraft 世界种子上运行。

**📈 对比分析**

与基线控制器和规划器对照，结果显示控制器明显优于规划器，但在四种源条件下相关经验并未产生稳定的目标提升，指标提升幅度有限。

**⚠️ 局限性**

局限性包括种子数少、源/目标预算受限、早期执行瓶颈导致后期能力未被观测、仅测试少数几种代理，难以推广至更强或更通用的 RSI 方法。

---

## 643. Can Language Models Learn When to Abstain? Policy Learning and Calibrated Answer Selection

**arXiv ID:** 2610.13821 | [PDF](https://arxiv.org/pdf/2610.13821v1)

**作者:** Juan Terven `[一作]` (Instituto Politécnico Nacional), Edgar Chavez-Urbiola `[通讯]` (Instituto Politécnico Nacional)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了训练语言模型的自我否定与使用外部答案选择器两种方法在问答中的有效性。

**💡 创新点**

比较了RL策略学习与学习答案选择器的实际收益，并在不同阈值下评估转移与人工审核。

**🔧 技术方法**

使用Qwen3-14B的低秩适配（LoRA）进行RL与选择器训练，并采用对数校准、ROC AUC、Brier等评估指标。

**📊 数据集**

使用TriviaQA进行训练、校准与验证，NQ-Open作为迁移测试集。

**📈 对比分析**

在TriviaQA上，学习选择器提升覆盖率从20.2%到30.9%，错误率降低；在NQ-Open上效果不佳，RL改动无显著提升。

**⚠️ 局限性**

仅在单一模型和短英语QA任务上评估，转移性能差，人工审核样本有限，实验未预注册。

---

## 644. QTestBench: A Test Framework for Evaluating LLM-Generated Quantum Code

**arXiv ID:** 2610.13811 | [PDF](https://arxiv.org/pdf/2610.13811v1)

**作者:** Meriem Terki `[一作]` (École Supérieure en Informatique de Sidi Bel Abbès), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 QTestBench 框架，用于评估大型语言模型（LLM）生成的量子单元测试套件是否能正确区分有效与无效的量子程序。

**💡 创新点**

创新点在于：①使用独立的任务检查器而非测试本身作为真值，避免测试误导；②引入三种验证配置（信任参考、自然 LLM 解决方案、合成突变）以分别衡量可执行性、接受率和错误检测；③统一定义并跨 PennyLane QHack 与 Qiskit HumanEval 的测量指标。

**🔧 技术方法**

技术实现包括：两阶段流程——Stage 1 让 LLM 单次生成语法、语义、行为三类 pytest 模块；Stage 2 注入候选代码、运行测试、与官方检查器对比；使用合成突变、随机种子、阈值控制、时间限制等手段保证实验可复现。

**📊 数据集**

使用的数据集为：八个 PennyLane QHack 挑战（共 8 任务）与十个 Qiskit HumanEval 任务（共 10 任务），以及五个公开 LLM（Claude Opus 4.6、DeepSeek V3.2、Gemini 3 Pro、GPT‑5.4、Qwen3‑Coder）。

**📈 对比分析**

比较方法通过 6 个指标（可执行性 ES、正确接受率 CS、错误检测率 BDS、测试质量 TQS、检查器覆盖 C_in、综合质量 CQ）在三种验证配置下对各模型进行打分；实验显示可执行率普遍高（ES≈1.00），但判别能力有限：Claude Opus 4.6 在 QHack 的 TQS 最高达 0.71，Gemini 3 Pro 0.63，Qiskit 环境下最高仅 0.34，表明即便测试能跑，能识别错误的能力仍有待提升。

**⚠️ 局限性**

局限性包括：仅使用单一温度 0.8 的一次性生成，未尝试多轮修复或调优；评估任务数量有限（14 任务）；合成突变可能无法完全代表真实错误；使用的检查器仅来自官方，缺少更丰富的多源判别；实验中未结合执行反馈进一步改进测试生成。

---

## 645. Randomized Momentum Transport with Cheap Predictors

**arXiv ID:** 2610.13789 | [PDF](https://arxiv.org/pdf/2610.13789v1)

**作者:** El Mahdi Chayti `[一作]` (EPFL), Martin Jaggi `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究在动量方法（如 STORM 与二阶动量）中，随机查询梯度传输修正（或 Hessian‑向量积），并用廉价预测器代替部分查询，以降低每步计算成本。

**💡 创新点**

创新点包括：① 引入随机化传输估计器，保持条件均值并通过重要性加权控制额外方差；② 在此框架下证明了 O(T⁻¹/³) 的停机率上界，并给出了梯度预算与查询概率 q 的精确关系；③ 提出在线对角线预测器并给出 regret 与最终残差的解析上界；④ 通过实验验证在不同 q 与预测器设置下的性能提升。

**🔧 技术方法**

主要技术：随机重要性采样的动量更新；线性最小化 oracle（LMO）方向；梯度差分与 Hessian‑向量积传输；理论分析基于残差依赖的方差分解、递推误差展开与马尔可夫差分；在线对角预测器的增量更新与 regret 证明；实验方法包括 MNIST1D、CIFAR‑10 训练与自定义二次函数。

**📊 数据集**

使用的数据集：MNIST1D（4000 训练 / 1000 测试序列），CIFAR‑10（ResNet‑18），以及带控制曲率噪声的二维/三维自定义二次目标。

**📈 对比分析**

对比方法：全量查询（q=1）的 STORM / HVP；随机查询（q∈{1,1/2,1/4}）并使用零预测器或对角预测器；在相同的 oracle 成本下评估训练损失与测试精度。实验结果表明，在相同梯度等价预算下，q<1 通常能降低 18%–43% 的最终训练损失，并在 CIFAR‑10 上提升 1.5 分左右的测试准确率；当 HVP 费用较高时提升更为显著。

**⚠️ 局限性**

局限性：① 在所有设置下并未取得显著提升，主要受益于高噪声或高昂修正成本的情形；② 对角线预测器仅能消除残差的一小部分，提升有限；③ 需要手动调节 q、步长等超参数；④ 理论分析基于一系列假设（L‑smooth、噪声界、残差假设），在实际非凸问题中可能不完全成立。

---

## 646. UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes

**arXiv ID:** 2610.13783 | [PDF](https://arxiv.org/pdf/2610.13783v1)

**作者:** Nazar Andrushko `[一作]` (Ukrainian Catholic University), Veronika Solopova `[通讯]` (Johannes Gutenberg University Mainz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究乌克兰战争期间网络迷因中说服技术的检测，构建了两套数据集并在零样本提示下评估多模态视觉‑语言模型的表现。

**💡 创新点**

创新点在于提出乌克兰本土迷因数据集UkrMeme与翻译版PTM‑UA，使用层级F1评价并将Layer‑wise Relevance Propagation与模型自带token attribution进行对比，揭示两种解释方法对证据关注点的差异。

**🔧 技术方法**

采用零样本提示的视觉‑语言模型（Qwen3‑VL‑4B/8B、GPT‑4.1‑mini、Gemma‑3‑4B），并在图像、文本及图像+文本三种输入模式下运行；使用LRP与token attribution对解释性进行分析。

**📊 数据集**

数据集包括158张乌克兰语翻译后的Propaganda 950子集（PTM‑UA）以及200张原创乌克兰战争迷因（UkrMeme）。

**📈 对比分析**

通过与加权随机基线及模型性能对比，发现Qwen3‑VL‑4B/8B在翻译集上取得最高micro hF1≈0.54（image+text），但在原生集上性能显著下降（≈0.34），并且LRP与token attribution在词汇关注上存在显著差异。

**⚠️ 局限性**

局限性包括数据集规模有限、注释主观性、翻译数据不完全代表真实乌克兰语迷因、仅使用零样本提示、LRP仅适用于开放权重模型、未评估视觉空间解释等。

---

## 647. A $1/2$-Approximation for Pandora's Problem with Deadlines

**arXiv ID:** 2610.13775 | [PDF](https://arxiv.org/pdf/2610.13775v1)

**作者:** Shenyu Qin `[一作]` (Shanghai Jiao Tong University), Zixuan Zhu `[通讯]` (University of Hong Kong)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析了一种部分可适应、即时承诺的算法，证明其在Pandora’s Problem with Deadlines中可实现1/2的近似比。

**💡 创新点**

创新点在于：直接结合Weitzman的预留价值与经典Prophet Inequality的阈值-盈余分解，构造“预留阈值盈余”桥接，避免多阶段损失；证明1/2是即时承诺算法的最优下界，并揭示部分可适应性在此模型下的固有限制（≤0.778）。

**🔧 技术方法**

采用期望上界与阈值-盈余分析、动态规划求解最优可行集合、白盒阈值证明、线性规划求解精确阈值，以及概率极限分析等理论工具。

**📊 数据集**

本文为理论研究，无实验数据集。

**📈 对比分析**

与之前约0.158的近似比对比，证明通过新算法实现了1/2的改进；同时通过构造实例证明部分可适应策略相对完全可适应的下界为0.778。

**⚠️ 局限性**

局限性：算法仅适用于即时承诺和部分可适应情形；对延迟承诺或完全可适应的改进仍未得到；无法直接扩展到更一般的约束或多目标场景。

---

## 648. RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics

**arXiv ID:** 2610.13889 | [PDF](https://arxiv.org/pdf/2610.13889v1)

**作者:** Weiliang Chen `[一作]` (Texas Tech University), Song Liao `[通讯]` (Texas Tech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于执行动态的可靠性感知工具检索框架 RATED，解决 AI 代理在工具库中快速检索并避免运行时失效的问题。

**💡 创新点**

创新点在于将冻结的双编码器（DRAGON）与轻量级上下文Bandit相结合，利用实时执行反馈即时重排候选工具，兼顾语义匹配与可靠性评估。

**🔧 技术方法**

使用 DRAGON 双编码器进行语义检索，Contrastive 学习微调；使用 Discounted LinUCB 上下文Bandit进行在线重排；融合静态可靠性先验。

**📊 数据集**

构建 EcoAgent benchmark，包含 5,172 个真实工具及 210k+ 查询–工具对，分为 Synthetic（EcoAgent‑Syn）与 Real（EcoAgent‑Real）两子集。

**📈 对比分析**

与 TF‑IDF、BM25、Ada‑002、BGE‑M3、GTE‑Large、SPLADE‑v3 等基线相比，RATED 在 EcoAgent‑Real 上 MRR 提升 52.2%，在 20%/30% 失效注入场景下，故障选取率仅为 2.5%–2.9%，并在 1.6–2.1% 的平均适配步长内恢复，且仅增加 0.17 ms 的延迟。

**⚠️ 局限性**

局限在于尚未在真实生产代理中验证冷启动时的新工具适配，以及对更大规模、持续变化的工具库的泛化能力。

---

## 649. Prediction Is Not a Sufficient Memory State

**arXiv ID:** 2610.13881 | [PDF](https://arxiv.org/pdf/2610.13881v1)

**作者:** Robert Jenkinson Alvarez `[一作]` `[通讯]`, Robert Jenkinson Alvarez

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了在持续学习情境下，记忆体必须保留哪些信息才能使学习器在未来观察到新的数据时继续进行有效更新，并给出了在不同学习任务（岭回归、非线性残差损失、核注意力等）下的理论上最小连续记忆维度。

**💡 创新点**

创新点在于提出并证明了：
1) 对任意平滑凸残差损失，只有损失是多项式时才存在固定维度的闭合记忆；
2) 对岭回归给出了完整的记忆维度公式（dp + d(d+1)/2），并给出了对有限比特压缩的误差指数；
3) 对核注意力的“分子-分母”结构给出了闭合维度上限；
4) 对标量逻辑回归提供了即使局部高阶导数被固定也无法压缩的证明。
这些结果阐明了记忆体与未来更新可辨识性的本质关系。

**🔧 技术方法**

主要技术包括：
- 功能空间与梯度函数的等价变换、函数空间维度判定；
- 开放参数族与不变量定理、逆函数定理的应用；
- 复合函数与齐次多项式的谱分析；
- 随机矩阵与量化误差下的矩阵范数估计；
- 对齐数值与特征向量的随机取样。

**📊 数据集**

该工作不依赖任何具体数据集，而是针对通用的连续特征空间、标签空间以及无限长的历史序列进行理论分析。

**📈 对比分析**

由于是理论证明性工作，本文没有对实验方法进行比较，也没有给出数值性能指标；所有结论均基于严格的数学证明，适用于理论模型的验证。

**⚠️ 局限性**

局限性：
1) 结果仅适用于无约束的连续特征/标签空间，离散或有限域下的情况需要单独研究；
2) 证明给出了最小维度与误差指数，但未给出可实现的高效在线压缩算法；
3) 对实际深度学习模型的影响仍需经验验证；
4) 对非平滑或非凸损失、带权重的观测等更一般情形的扩展仍是未解决的问题。

---

## 650. LapMerge: Differentiable Exact-Budget Token Merging in Linear Time

**arXiv ID:** 2610.13858 | [PDF](https://arxiv.org/pdf/2610.13858v1)

**作者:** Magdalena Trędowicz `[一作]` (Jagiellonian University), Jacek Tabor `[通讯]` (Jagiellonian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于拉普拉斯核的可微分、严格按预算拆分的序列压缩算子 LapMerge，能在 O(M+N) 时间内把 M 个有序 token 压缩为 N 个输出 token，并将压缩后的表示送入 Transformer 进行后续任务；

**💡 创新点**

创新点在于：①通过拉普拉斯核将 token 的重要性转化为连续质量分布，从而实现精确的等质量区间划分；②在保持端到端可微的同时，仅使用前缀/后缀扫描即可完成所有运算，显著降低了 O(MN) 的计算复杂度；③兼具固定预算、可训练性和线性成本三者的优点，解决了先前 reducer 在这三点上的权衡；

**🔧 技术方法**

技术上使用了拉普拉斯核密度估计、soft top‑k 路由、前缀/后缀累计求和扫描、闭式求解区间边界以及梯度的线性求导；

**📊 数据集**

实验数据集包括：TCGA 的 2 个全切片组（CRC、STAD）用于 MSI 预测；POPE 对象幻觉基准；Long Range Arena 5 个任务；以及大规模 10^7 token 级别的计算压力测试；

**📈 对比分析**

与多种基线（TokenLearner、PatchMerger、ToMe、Uniform Pooling 等）以及冻结的 VisionSelector、Qwen2.5‑VL‑7B 等 VLM 进行比较，LapMerge 在保留 1,024 token 的情况下，在 CRC/STAD 的 AUROC 及 POPE 的 F1 方面均优于所有基线，且在 10^7 token 测试中速度提升 10‑100 倍；

**⚠️ 局限性**

局限性：该算子依赖输入序列的空间/时间局部性，对全局关联性弱或乱序排列的任务效果不佳；

---

## 651. Event-based Action Recognition by Pre-Training on Multi-Frame Representations

**arXiv ID:** 2610.13856 | [PDF](https://arxiv.org/pdf/2610.13856v1)

**作者:** Emma Norrby `[一作]` (Swedish Defence Research Agency), David Gustafsson `[通讯]` (Swedish Defence Research Agency)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于多帧事件表示的自监督预训练框架，结合活动驱动的 token 选择、稀疏感知掩码以及位置相关的多 mask token，以提升长时序事件动作识别性能。

**💡 创新点**

创新点包括：① 将事件序列切分为多帧并进行 token 选择；② 设计稀疏感知掩码策略，使掩码过程更偏向高活动区域；③ 引入多位置相关 mask token 与 softmax 注意力机制；④ 通过上述三种改进在单一事件模态上实现 SOTA。

**🔧 技术方法**

技术手段主要是 Vision Transformer（ViT）+ 时空 MAE 预训练；配合多帧事件表示（CSTR / Event Histogram）、活动度量用于 token 选择、稀疏感知掩码、以及多 mask token 的位置注意力投影。

**📊 数据集**

使用公开长时序事件动作数据集 THU‑EACT‑50 CHL（50 类）和 DailyDVS‑200（200 类）进行评估。

**📈 对比分析**

与现有基线（包括 EventMamba、EventBind、SMV‑EAR 等）进行线性探针和微调对比，最终在 THU‑EACT‑50 CHL 取得 78.7% 以及 DailyDVS‑200 取得 66.9% 的准确率，明显优于前沿方法。

**⚠️ 局限性**

局限性：多帧分割会显著增加 token 数，仍需通过 token 选择来控制算力；在极短动作或低活动序列中，稀疏感知掩码的优势不明显；多 mask token 的最佳数量尚未确定；模型主要针对事件数据，缺乏跨模态验证。

---

## 652. HorizonNav: Geometry-Guided Video Diffusion for Anticipatory 3D Robot Navigation

**arXiv ID:** 2610.13849 | [PDF](https://arxiv.org/pdf/2610.13849v1)

**作者:** Hajira Amjad `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 HorizonNav 框架，利用单张 RGB 图像和文本提示生成未来视角视频，通过 VLM 提示、Wan 2.2 视频生成、ORB‑SLAM3 轨迹与稀疏特征重建、Video Depth Anything 的密集深度预测以及几何对齐，构建可用于机器人预运动规划的三维空间模型。

**💡 创新点**

创新点在于将视频扩散模型产生的虚拟观察视为“虚拟传感”，并通过稀疏三角测量与密集深度的几何对齐实现跨视角尺度一致性，使得生成的空间假设既可供 RRT 规划又与真实环境保持可执行性，且无需在生成器中强制加入几何先验。

**🔧 技术方法**

使用的技术包括 Gemini 2.5 Flash 视觉语言模型、Wan 2.2 I2V Fast Diffusion 视频生成、ORB‑SLAM3 三维稀疏重建、Video Depth Anything 的时序深度预测、基于稀疏特征的尺度与平移对齐、局部残差纠正以及 RRT 规划。

**📊 数据集**

数据集主要有 19 场景的 horizon study、67 例生成视频重建归档、15 条走廊重建样本以及独立获取的 Livox LiDAR 地图作为外部评估；实验还在六架四旋翼上进行飞行验证。

**📈 对比分析**

实验对比显示，生成视角延伸从 P20 到 P81 能将可规划场景数从 13/19 提升至 15/19，RRT 成功率从 62/90 增至 75/90；重建成功率 58/67，规划可行率 36/58；六次飞行中有 5 次成功通过门，表明生成空间假设在实际机器人上具有可执行性。

**⚠️ 局限性**

局限性包括仅在室内场景和单一生成器下验证；生成的深度虽保持时序一致，但可能因局部误差产生虚假障碍；缺乏在线自检或动态修正机制；对多平台、多环境的泛化能力待进一步研究。

---

## 653. Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga

**arXiv ID:** 2610.13840 | [PDF](https://arxiv.org/pdf/2610.13840v1)

**作者:** Emilio Ferrara `[一作]` `[通讯]` (University of Southern California), Emilio Ferrara (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

该论文提出了 PAGE（Parse‑Annotate‑Ground‑Explain）框架，利用人工智能在大规模漫画档案上完成页面解析、文本和结构化注释、人工验证以及趋势解释，并通过三项实验展示其在总统描绘、页面视觉语言演变和角色说话者性别比例等研究问题上的应用。

**💡 创新点**

创新点包括：
- 将视觉‑语言模型与书面代码本相结合，实现在海量漫画页面上的可复制、可检索注释；
- 设计“Ground”步骤，对模型输出进行多源人工验证，提升结果可信度；
- 将结果与出版商、类型等控制变量结合，进行因果解释；
- 在研究、档案、艺术家三方面提出可直接落地的用途（发现、署名、无障碍访问、数字保存）。

**🔧 技术方法**

技术手段主要有：
- 视觉‑语言模型（Claude Opus 5.5）用于读取图像并按代码本输出特征；
- 目标检测与OCR（Faster R‑CNN、CLIP ViT‑B/32）实现面板、文本框、角色定位与图像嵌入；
- 统计回归与自举（bootstrap）控制出版商、类型，估计年度趋势；
- 人工标注与交叉验证（Cohen’s κ）作为基线；
- 语义检索与图像检索对比实验（BM25、dense retrieval、CLIP检索）。

**📊 数据集**

数据集包括：
- COMICS：1940‑1956年美国黄金时代漫画的扫描、面板盒、OCR；
- Grand Comics Database（GCD）：出版信息、故事级艺术家署名；
- Manga109‑s：1970‑2010年日本漫画卷，手绘面板、文本框；
- Manga109‑Dialog：为Manga109‑s提供每个气泡的说话者链接；
- 额外的公共档案（Digital Comic Museum、Manga109授权）。

**📈 对比分析**

比较方法与性能：
- 对总统研究：用OCR检索总统名，视觉模型判定呈现方式，未给出量化准确率，但通过人工复核估计置信区间；
- 页面视觉研究：使用回归控制变量后，年均面板数与文字数变化显著（p<0.05），并通过图像嵌入计算与艺术家相似度，匹配准确率在去除同系列、重复图像后显著高于随机（p<0.01）。
- 说话者性别研究：模型对人物性别的预测在美国漫画中达约90% κ，跨传统后女性说话比例对比显著（p<0.01）。
- 文本与图像检索对比显示，CLIP检索在相关性上比传统OCR检索高约15%。

**⚠️ 局限性**

局限性包括：
- 总统研究缺乏第二码作者，验证仅靠模型自身；
- 人工标注覆盖有限，仅对性别、角色等做了粗略核对；
- 只检索已命名总统的页面，忽略未命名的相似画面；
- 数据集跨时间跨度大，黄金时代与近代漫画对比存在年代差异；
- 可能存在模型偏差与样本不均衡问题；
- 仅使用一种模型家族，缺少跨模型稳健性验证。

---

## 654. Mapless, Stabilizing, and Anti-Stuck Flipper Control for Articulated Tracked Robots

**arXiv ID:** 2610.13838 | [PDF](https://arxiv.org/pdf/2610.13838v1)

**作者:** Valentýn Číhala `[一作]` (Czech Technical University in Prague), Karel Zimmermann `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无地图、仅依靠IMU、舵机角度及短窗口运动阈值的翻滚控制器，可在无地图环境下实现障碍跨越。

**💡 创新点**

创新点在于将姿态稳定与反卡死机制通过方向优先混合结合，并采用短窗口运动感知，避免积分漂移，使得无地图控制器能匹敌地图基控制器。

**🔧 技术方法**

采用PID姿态控制、短窗口运动阈值提取、方向优先混合与限幅、以及LiDAR‑ICP+IMU的感知栈，在Gazebo仿真与真实四轨翻滚机器人上验证。

**📊 数据集**

在自建的12障碍仓库仿真课程与真实平板障碍的测试场景中进行实验，未使用公开数据集。

**📈 对比分析**

通过10次完整跑完12障碍或10次真实跨越，比较三通道遍历质量、冲击、倾斜和地面间隙；无地图控制器在仿真下取得最高TQ，实测在平板上位列第二，并在隐藏障碍上表现更好。

**⚠️ 局限性**

局限在于短窗口运动阈值对极端运动不敏感，缺乏长时间连续越障时的地图漂移补偿；在地面高度差大或轨道严重滑移的场景下性能下降，且实验仅覆盖短障碍，未验证长距离连续行驶。

---

## 655. Shared Canonical Component Removal Beyond Expert Views

**arXiv ID:** 2610.13819 | [PDF](https://arxiv.org/pdf/2610.13819v1)

**作者:** Jose Roberto Tello Ayala `[一作]` (Harvard University), Finale Doshi-Velez `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了共享规范成分移除（SCCR）框架，用来审计原始输入中相对于专家视图所剩余的预测信号。

**💡 创新点**

创新点在于引入阈值化的正则化典型相关分析，既去除与专家视图高度相关的线性方向，又保留低相关但潜在有用的线性信息，实现更细粒度的残差提取；同时通过冻结校准实现确定性推理。

**🔧 技术方法**

技术包括双分支深度网络、OAS正则化的CCA、阈值化的投影操作、冻结校准、线性/非线性探针评估以及与多种多视图分解方法的对比。

**📊 数据集**

使用的数据集包括合成数据、冠状动脉造影与其分割掩膜（CAD）、精神科EHR（抗抑郁药物停药预测）以及皮肤病变图像与皮肤科医生标注（SkinCon/Fitzpatrick17k、DDI）。

**📈 对比分析**

与传统残差化、全共享/私有分解方法（如 D-CCA、JIVE、VCCA、DSN-MMD）对比，SCCR 在保留预测性能的同时显著降低了与专家视图的线性可恢复性；在CAD、抑郁药物停药和皮肤病变分类实验中，SCCR 的AUROC 与原始原始模型相当或略优，且能够揭示遗漏概念的贡献。

**⚠️ 局限性**

局限性包括只能消除线性相关的冗余，无法消除非线性依赖；阈值选择依赖经验，且在不同数据分布下可能需要重新校准；对多模态非线性信息的提取仍有提升空间。

---

## 656. TabMF: Learning Dataset Representations with Tabular Foundation Models

**arXiv ID:** 2610.13818 | [PDF](https://arxiv.org/pdf/2610.13818v1)

**作者:** Rafiq Kamel `[一作]` (University of Technology Nuremberg), Josif Grabocka `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用预训练的 Tabular Foundation Models（如 TabICLv2 与 TabPFN-2.5）通过架构感知的读出层和对比学习后训练，将完整监督数据集映射为固定维度向量的框架，并在六项数据集级任务上进行评估。

**💡 创新点**

创新地将预测预训练模型重用为数据集编码器，结合对比学习在合成数据视图上进行后训练，并揭示预测预训练本身已产生有用的数据集级表示，最终在多任务上取得最佳综合表现。

**🔧 技术方法**

采用对比损失（NT‑Xent）、架构感知读出（summary‑token、column‑aware）、Transformer 读出、投影网络及基于 Graph‑SCM 先验生成的合成数据进行训练。

**📊 数据集**

在合成 TabICLv2 Graph‑SCM 数据集上训练，随后在 191 个 TabBench、144 个 TabRepo、9 个 TableShift 等真实分类数据集上进行评估，涵盖视图辨别、属性预测、鲁棒检索、基准构建及分布偏移等任务。

**📈 对比分析**

与传统手工特征（PyMFE）、学习型编码器（Dataset2Vec、Neural Statistician）以及零样本学习方法对比，六项任务的聚合排名均名列前茅，尤其在视图辨别、鲁棒检索、基准选择与分布偏移评估上取得最强成绩，冻结预训练模型已表现强劲，后训练进一步提升鲁棒性。

**⚠️ 局限性**

仅针对分类任务、仅使用两种 TFM 架构；后训练依赖 TabICLv2 的 Graph‑SCM 先验；表示受目标标签影响，未覆盖回归、不同数据生成先验或需要针对特定相似度定义的场景。

---

## 657. Spanning Diffusion and Flow-based Posterior Sampling via Exact Gaussian Conditioning

**arXiv ID:** 2610.13812 | [PDF](https://arxiv.org/pdf/2610.13812v1)

**作者:** Teo Deveney `[一作]` (University of Bath), Hok Shing Wong `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种名为β-Flow的通用引导采样方法，可在线性逆问题中灵活选择流、扩散或无相关先验的轨迹；

**💡 创新点**

通过将轨迹相关系数β与置信参数κ引入模型，实现对先验强度和轨迹相关性的显式控制，并将每一步引导步骤化为精确的高斯条件化（Kalman更新），从而统一解释并克服了现有方法隐式假设的局限；

**🔧 技术方法**

核心技术包括：一阶高斯条件化、Kalman增益、Tweedie公式的近似先验、单次前向网络评估无梯度传播、可调的β-相关性与κ置信度；

**📊 数据集**

在AFHQ‑Cats、CelebA、FastMRI（MRI）、AAPM（CT）等四个数据集上，分别对4×超分辨、去噪、随机/盒形缺失、Gaussian去卷积等七类逆问题进行实验；

**📈 对比分析**

与PnP-Flow、Flower、DPS、ΠGDM、OT‑ODE、D‑Flow等主流方法相比，β‑Flow在PSNR/SSIM上大多数任务达到或超过最佳水平，且仅差≤1%；在医学成像任务中同样取得最优或次优性能；

**⚠️ 局限性**

局限性：需针对不同任务调参（β、κ），对感知指标（LPIPS）不如一些对细节敏感的方法；在极端稀疏观测下的性能提升有限；对网络梯度不利用虽然降低成本，但可能无法捕捉更复杂的非线性结构；

---

## 658. FORTE: A Production Framework for Feature Evaluation and Deprecation in Industrial Recommender Systems

**arXiv ID:** 2610.13806 | [PDF](https://arxiv.org/pdf/2610.13806v1)

**作者:** Vishnu Arun `[一作]` (Pinterest), Darren Reger `[通讯]` (Pinterest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FORTE框架，对工业推荐系统中的特征生命周期进行模型无关的评估、验证与退役，实现生产环境下的每日特征重要性分析与特征集群删除。

**💡 创新点**

创新点在于：①将Permutation Importance与Pooled Inference结合，实现对成千上万个特征的高效每日扰动；②引入分组扰动与聚合评估，捕获特征交互；③构建特征线索追踪系统，自动定位并安全地从训练、评估与存储中彻底剔除已删除特征；④整体流程从重要性筛选到线上A/B验证再到线索追踪的端到端管线，大幅提升特征退役效率。

**🔧 技术方法**

采用Permutation Importance（按批次的置换或置零）、Group Perturbation（联合扰动）、Pooled Inference（一次前向推理覆盖所有实验）、Feature Registry（统一特征类型与null定义）、A/B Test（线上验证）以及Feature Lineage Tracking（从模型到数据存储的完整追踪）。

**📊 数据集**

利用Pinterest生产环境的两大推荐系统（Search Relevance与Pin Ranker）所使用的业务数据集，涵盖多语言、多头、多模态特征，评估数据量从数十万到数百万行不等。

**📈 对比分析**

通过在这两大系统上验证：Search Relevance 48特征集落地后去掉43.2%特征，训练参数减少2.3%，吞吐率提升2%，网络与存储成本下降；Pin Ranker 8特征集落地后去掉29.9%参数，吞吐率提升19.8–25.5%，每年训练成本节约数十万美元。整体性能在保持线上指标不退化的前提下实现显著效率提升。

**⚠️ 局限性**

主要限制包括：只评估一阶扰动，无法完全捕捉特征交互与条件重要性；单次扰动导致估计方差较大；对数据漂移与特征相关性的响应有限；线索追踪依赖模型注册与手动审核，可能漏掉某些隐式依赖；公平性与子群体影响评估受限，仅提供粗粒度分段验证。

---

## 659. REFLECT: Role-Aware Detection of Sentiment, Emotion, Politeness, and Formality

**arXiv ID:** 2610.13797 | [PDF](https://arxiv.org/pdf/2610.13797v1)

**作者:** Sangpil Youm `[一作]` (University of Florida), Bonnie Dorr `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入基于语义角色标注（SRL）的自注意力机制，将情感、情绪、礼貌与正式度映射为可解释的角色配置并融入Transformer

**💡 创新点**

通过理论驱动的角色映射与方向性注意力，实现对内部状态与外部提示的差异建模，提升了模型的可解释性与性能

**🔧 技术方法**

BERT+SRL自注意力、角色感知与方向性注意力、门控融合与线性/MLP预测头

**📊 数据集**

GoEmotions（情绪），Holistic Politeness（礼貌），GYAFC（正式度），以及EARS青少年心理健康EMA数据

**📈 对比分析**

与标准BERT基线相比，门控+线性配置在情绪、礼貌和正式度上显著提升（p<0.01），情感提升有限；在真实数据上亦获得更高的相关系数

**⚠️ 局限性**

情感提升不显著，模型依赖SRL输出，可能带来推理延迟；解释性仍需系统化人类评估验证

---

## 660. When LiDAR Fades, Radar Takes Over: LiDAR Reliability-Aware Adaptive Localization Under Airborne Obscurants

**arXiv ID:** 2610.13788 | [PDF](https://arxiv.org/pdf/2610.13788v1)

**作者:** Moumita Mukherjee `[一作]` (Luleå University of Technology), George Nikolakopoulo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于一类时序降解检测的LiDAR‑雷达自适应定位框架，能在地下隧道中实时识别LiDAR信号衰减并自动切换到雷达‑惯性里程计；

**💡 创新点**

创新点在于将十二维可解释特征的时序演化建模为LSTM自编码器，仅使用健康数据训练，无需降解标注，并结合测量新鲜度阈值实现对持续降解与完全失效两种模式的联合检测；

**🔧 技术方法**

使用了体素网格统计、跨模态一致性、几何/运动、强度及测量时效等特征提取技术，构建12维特征向量；随后利用LSTM自编码器和阈值化重构误差进行异常检测，并在IMU时间线上实现自适应模态选择；

**📊 数据集**

实验数据来源于在GNSS‑禁区的地下隧道内收集的真实机器人数据，包含正常条件与高浓度烟雾两种环境，未提供外部绝对轨迹真值；

**📈 对比分析**

通过与LiDAR‑惯性里程计（正常）和雷达‑惯性里程计（烟雾）轨迹的RMSE对比，发现在正常情况下LMRIO与LIO误差仅0.012 m，而在烟雾下LMRIO与RIO误差为1.11 m，远优于LIO出现的巨大漂移；特征消融实验表明精简特征集可进一步提升切换时机与轨迹一致性；

**⚠️ 局限性**

局限性包括：需在每个新环境下仅健康数据训练；仅在烟雾/尘埃等特定遮蔽条件下验证；缺乏外部绝对轨迹真值；对其他降解类型或不同机器人平台的泛化性尚未证明。

---

## 661. SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models

**arXiv ID:** 2610.13781 | [PDF](https://arxiv.org/pdf/2610.13781v1)

**作者:** Changhui Jing `[一作]` (Shanghai Innovation Institute), Zhenjie Yang `[通讯]` (OpenDriveLab at University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过后训练框架将轨迹优化（GRPO）与仿真渲染的未来视频监督相结合，对驾驶世界动作模型（WAM）进行联合优化，使其在规划时更贴合真实视觉。

**💡 创新点**

创新点在于：①使用World Engine 3DGS重建真实场景并沿当前策略轨迹渲染匹配视频，弥补日志视频缺失的监督；②依据轨迹奖励与视频预测误差选择高奖励、高误差轨迹作为视频监督目标；③将流匹配视频损失与轨迹GRPO共享表示一起更新，从而提升规划性能。

**🔧 技术方法**

关键技术包括：3D Gaussian Splatting 场景重建、SDE轨迹采样、Group Relative Policy Optimization（GRPO）、流匹配视频损失、行为克隆正则化、World Engine 仿真渲染。

**📊 数据集**

主要使用数据集：NAVSIM v1、NAVSIM v2 以及 HUGSIM。

**📈 对比分析**

在NAVSIM v1上 PDMS 从 86.2 提升至 91.3（+5.1点）；在 NAVSIM v2 上 EPDMS 提升至 88.2；在 HUGSIM 上平均 HD-Score 达到 38.8，均优于现有基线方法。

**⚠️ 局限性**

限制与挑战：①渲染预算受限，需优先选取高误差轨迹；②框架依赖外部仿真引擎，渲染成本高；③目前未解决更长时间规划或多模态预测的场景。

---

## 662. DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs

**arXiv ID:** 2610.13767 | [PDF](https://arxiv.org/pdf/2610.13767v1)

**作者:** Mahsa Salmani `[一作]` (d-Matrix), Hesham Mostafa `[通讯]` (d-Matrix)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无需校准数据的双尺度后训练量化框架 DualQuant，用于同时量化权重和激活。

**💡 创新点**

创新点在于将列尺度从权重重构误差最小化中直接派生出来，同时充当激活重缩放因子，从而在不依赖激活统计的情况下实现高质量激活量化。

**🔧 技术方法**

使用块坐标下降（BCD）对行/列尺度进行闭式更新，并通过投影满足不同低位格式（INT4、MXINT4/MXFP4、NVFP4）的量化网格约束。

**📊 数据集**

在 WikiText-2、C4、以及 MMLU、HellaSwag、PIQA、ARC‑Easy、WinoGrande 等零样本下游任务上评估模型，主要针对 Qwen 与 Llama 系列。

**📈 对比分析**

与 SmoothQuant、AWQ+、RTN、SINQ+、QuaRot 等基线比较，DualQuant 在大多数格式下匹配或超过 SmoothQuant，显著优于无校准基线；在 RTX4090 上实现 W4A4 推理时，内存降低 2.51–2.61×，吞吐量提升 1.39–2.46×，逼近假量化精度。

**⚠️ 局限性**

局限性包括对 BCD 迭代次数的依赖，某些极端张量分布可能导致尺度收敛慢；此外在极低位格式（如 NVFP4）下仍可能出现数值不稳定或量化网格不匹配的边缘情况。

---

## 663. Decision-Focused Learning for Cyber-Physical Risk-Aware Power Grid Operations

**arXiv ID:** 2610.13853 | [PDF](https://arxiv.org/pdf/2610.13853v1)

**作者:** Ayesha Siddiqua Dina `[一作]` (Florida Polytechnic University), Afzal Hossain Sohag `[通讯]` (Florida Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合机器学习入侵检测与电网调度，提出迭代特征加权的决策聚焦框架，将检测误差转化为运作风险成本并优化电网调度。

**💡 创新点**

将入侵检测误差映射为生成器级风险成本并在优化中加权，采用自回归特征加权迭代以同步提升检测与调度决策；首次将决策聚焦学习与电网运营相结合。

**🔧 技术方法**

使用逻辑回归入侵检测、线性规划调度（Gurobi）、误差成本权重、迭代特征重加权、风险成本归一化与加权等技术。

**📊 数据集**

基于公开电网安全数据集ICS Power System Dataset和Sherlock 01-Basic。

**📈 对比分析**

与无迭代基线对比，使用F1分数评估检测性能、运作成本评估调度效果；在最佳超参数下，运作成本相较基线下降约0.2%（ICS）和0.07%（Sherlock），F1维持或略有提升。

**⚠️ 局限性**

仅在小型示例网路测试，使用单一入侵检测模型与线性调度，缺乏对大规模网路、多模型与更复杂约束的验证；结果受α、误差成本配置敏感。

---

## 664. Inference-Time Steering Across Scaffolds and Epitopes for Sample-Efficient De Novo Nanobody Design

**arXiv ID:** 2610.13844 | [PDF](https://arxiv.org/pdf/2610.13844v1)

**作者:** Mason Minot `[一作]` (Alloy Therapeutics), Cédric R. Weber `[通讯]` (Alloy Therapeutics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对纳米体设计中的可变支架与表位条件，提出动态资源分配与推理时引导相结合的方法，提升设计可行候选的产率。

**💡 创新点**

将全局SMC与分离式Successive Halving两种动态预算分配框架引入生成过程，解决传统固定支架导致的计算浪费，并实现了约5倍的产率提升。

**🔧 技术方法**

采用Feynman-Kac推理时引导、ipTM即时奖励、Tweedie公式去噪、ESMFold2与AlphaFold2-Multimer评估、BoltzGen/Inverse Folding与ProteinMPNN序列生成。

**📊 数据集**

在PD-L1、IL-7Rα、BHRF1三种靶点及多种支架/表位配置的纳米体结构数据集上进行实验。

**📈 对比分析**

与传统无引导、固定分配基线相比，动态SMC和SH分别在GPU小时内可行背骨数提升约5倍和2-3倍，保持了结构多样性。

**⚠️ 局限性**

局限性在于依赖计算模拟的ipTM/AlphaFold评估作为奖励和验证，易产生奖励欺骗，且尚未与体外结合亲和力或稳定性实验验证。

---

## 665. GPU-resident Execution for Timing-predictable Sensor Processing Systems

**arXiv ID:** 2610.13809 | [PDF](https://arxiv.org/pdf/2610.13809v1)

**作者:** Soham Sinha `[一作]` (NVIDIA), Mahdi Azizian `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一种GPU‑resident执行模型，将传感器输入、GPU加速处理与执行器输出完整地搬到GPU端形成一个长期循环，消除每帧CPU唤醒与CUDA调用，显著提升时序可预测性。

**💡 创新点**

创新点在于：①将I/O、计算与控制全部迁移至GPU，实现真正的GPU‑resident闭环；②利用GPU‑Memory RDMA、GPU‑driven I/O与CUDA Graph的WHILE/条件节点或Device Graph Tail Launch构成设备持久循环；③在NVIDIA Holoscan框架中提供操作符、数据就绪处理器API，支持端到端的可预测实时管道。

**🔧 技术方法**

采用的关键技术包括GPU‑Memory RDMA、DOCA GPUNetIO、GPU‑driven I/O、CUDA Graph（WHILE/conditional节点、Device Graph Tail Launch）、静态SM分区、锁定GPU时钟、CUDA MPS、Holoscan C++ SDK、TensorRT、NPP等。

**📊 数据集**

使用ImageNet 100k图像进行单阶段推理微基准；使用IMX274 1080p 60Hz原始摄像头流作为行业级摄像头‑显示案例；此外还使用合成的矩阵乘法与传感器/执行器仿真流进行实验。

**📈 对比分析**

对比CPU‑driven（Holoscan + CUDA Graph + 核心隔离）与GPU‑resident方案，在RTX A6000、IGX Thor、IGX Orin+Ada等平台测量绝对完成抖动、p99.9、标准差。结果显示：单阶段推理抖动降低1.4–2.4×；在主机竞争下抖动从≈1000↓至≈986；并发循环中p99.9与最大延迟低于2×；在IMX274摄像头‑显示案例中光子‑显示抖动从4.83ms降至0.66ms（7.3×），软件处理抖动从1.70ms降至0.01ms（170×）。

**⚠️ 局限性**

仍存在设备端缓存/内存争用影响；每个循环需要预留固定SM分区和锁定GPU时钟；不支持GPU‑resident与CPU或其他GPU‑resident任务间通信；依赖GPU直接I/O驱动（如HSB、DOCA），在无该硬件支持时受限；对动态负载或可变尺寸工作负载的适配有限；未处理多图形上下文或共享上下文的干扰。

---

## 666. Error Rate is Not Enough: Costs of Augmentation in Low-Resource ASR

**arXiv ID:** 2610.13807 | [PDF](https://arxiv.org/pdf/2610.13807v1)

**作者:** Katsumi Ibaraki `[一作]`, David Chiang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低资源语音识别中数据增强方法对不同音素错误的影响，比较四种增强方式的细粒度错误类型。

**💡 创新点**

首次以音素错误类别为维度评估音频与文本增强，揭示SpecAugment导致声调与软腭化错误，而文本增强可逆转这些错误，并提出基于词频平滑的补偿策略。

**🔧 技术方法**

使用Wav2Vec2‑large‑xlsr‑53的微调，结合Speed Perturbation、SpecAugment、Tag‑Based Replacement、Random Replacement等增强技术，并通过TTS合成新语音。

**📊 数据集**

四种低资源语料（Vatlongos、Nashta、S. Buryat、Kakabe）和LibriSpeech不同子集（1–100 h）作为实验数据。

**📈 对比分析**

通过对基线、单一增强和组合增强的WER/PER/CER及细粒度错误统计比较，发现SpecAugment+Random Replacement在所有低资源语言上均实现最高准确率，提升呈超加性，但在100 h时失效。

**⚠️ 局限性**

局限包括文本增强产生非语法文本、未解决OOV、仅使用固定TTS声音、缺乏多语言声学多样性，以及部分机制（如Tag‑Based替换修复声调错误的原因）未得到充分验证。

---

## 667. Detection of extremely rare dead cells in fluorescence microscopy images

**arXiv ID:** 2610.13780 | [PDF](https://arxiv.org/pdf/2610.13780v1)

**作者:** Prateek Verma `[一作]` (University of Arkansas), Xintao Wu `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

评估并比较了启发式、传统机器学习、深度学习与异常检测等四类方法，用 Cellpose 对荧光显微图像进行细胞核分割，并在一个含65,527个细胞但仅243个死亡细胞的极度不平衡数据集上实现并验证了这些算法。

**💡 创新点**

系统性地对比四类方法并发现深度学习和梯度提升模型在极少数死亡细胞上仍能取得显著性能，提出了人机交互决策支持框架，同时通过多通道荧光图像与分割结果的结合提供了更具可解释性的评估。

**🔧 技术方法**

使用的技术包括 Cellpose 分割、基于阈值与归一化的启发式方法、Pearson 相关、传统分类器（Logistic、Balanced Random Forest、XGBoost、LightGBM）、深度学习模型（ResNet‑18、EfficientNet‑B0）、异常检测模型（Isolation Forest、One‑class SVM、PCA 重建、卷积自编码器、Deep SVDD），并采用分组交叉验证调参。

**📊 数据集**

数据集来自阿肯色大学乳腺癌细胞 PBS 处理样本的多通道 CZI 图像，共766张，258张带人工标注，分辨率为 16 位，包含65,527个细胞掩模，其中243个为死亡细胞。

**📈 对比分析**

使用 PR‑AUC、Precision、Recall、F1‑score 与 macro‑F1 作为评估指标；深度学习 ResNet‑18（64×64 无填充）取得最高 F1 0.617、macro‑F1 0.808；XGBoost 0.608；启发式方法和异常检测方法性能明显较低，尤其异常检测几乎不产生正样本。

**⚠️ 局限性**

主要限制包括极度样本不平衡导致训练困难、分割误差对后续特征提取与分类产生负面影响、异常检测模型在此任务中表现不佳、并且目前仍无法实现完全自动化，需结合人机交互进行结果校正。

---

## 668. READ: Reading the Selective Scan for Faithful Video Attribution

**arXiv ID:** 2610.13776 | [PDF](https://arxiv.org/pdf/2610.13776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 669. MARGoS: A Configuration-Driven Platform for Reproducible MARL Experiments in ARGoS

**arXiv ID:** 2610.13772 | [PDF](https://arxiv.org/pdf/2610.13772v1)

**作者:** Tom Philipp Backert `[一作]` (University of Lubeck), Javad Ghofrani `[通讯]` (Bonn-Rhein-Sieg University of Applied Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MARGoS，一个轻量级的配置驱动平台，用于统一管理 ARGoS+MARL 实验的配置、种子、日志、导出等，实现可复现、可重用和协作；

**💡 创新点**

将实验定义的变异输入（配置、种子、路径等）作为平台管理的第一类资产，通过配置哈希、冻结配置、种子传播、结构化日志、环境指纹及可导出的实验包，将实验生命周期从手动工具链整合为单一命令行流水线；

**🔧 技术方法**

使用 ARGoS 与 RLlib 通过 ZeroMQ Bridge（ArgosToZoo）实现交互，CLI 提供五个命令（run、show、compare、export、import），并结合配置哈希、不可变配置、种子传播、结构化日志、环境指纹、Bundle 导出，利用 KLM/尼尔森启发式评估可用性；

**📊 数据集**

实验基于 ARGoS 场景和自定义训练脚本，未使用公开数据集；

**📈 对比分析**

采用五维评估（架构、效率、可复现性、可用性、协作）与手动工作流程对比，结果显示时间提升 78.2%、步骤减少 66.7%、复现成功率 100%、KLM 减少 56.9%、协作成功率 100%、分享时间减少 65.3%，复现时间略低于 50% 目标；

**⚠️ 局限性**

实验仅在单一硬件同实验室验证，缺乏跨实验室可迁移性，评估基于专家而非普通用户，导入时验证开销导致首次运行略慢，实验规模有限且未加入新实验。

---

## 670. Shortcut Learning in Geospatial Foundation Model Embeddings for Cross-Disaster Damage Assessment

**arXiv ID:** 2610.13766 | [PDF](https://arxiv.org/pdf/2610.13766v1)

**作者:** Ali Shibli `[一作]` (KTH Royal Institute of Technology), Yifang Ban `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

审计并评估了地理空间基础模型在灾害损害评估中的泛化能力，发现随机拆分过度乐观，提出事件留出评估并测试不同信息源及特征变换的效果。

**💡 创新点**

系统地量化了事件上下文、时间视角、建筑计数和配对对应等潜在捷径，证明简单的源域特征变换（如SupCon、PCA）可提升未见灾害的准确率。

**🔧 技术方法**

使用了 TerraMind、Prithvi-EO-2.0、DOFA、DINOv3 等预训练视觉编码器生成特征，构建线性探针、加权采样、PCA、MLP、监督对比学习等变换进行评估。

**📊 数据集**

在 xView2（19 场灾害，光学前后对）和 BRIGHT（11 场灾害，光学前+ SAR 后）两套多灾害图像数据集上实验。

**📈 对比分析**

对比随机图像拆分（IID）与留一事件（LOEO）下的平衡准确率与 AUC，发现 LOEO 下准确率下降约 0.27；加上特征变换可提升到约 0.66 的 BA（xView2）和 0.59（BRIGHT）。

**⚠️ 局限性**

仅测试了图像级别特征，未涉及建筑级损伤严重度，且对 SAR 处理不优化，跨传感器泛化仍有提升空间。

---

## 671. When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors

**arXiv ID:** 2610.13765 | [PDF](https://arxiv.org/pdf/2610.13765v1)

**作者:** Adarsh Jamadandi `[一作]` (CNRS), Aline Roumy `[通讯]` (INRIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种联合离散-连续扩散框架，通过Latent Sociability Prior（LSP）在扩散过程中保留节点异质性，解决了无节点特征的图生成中E‑R先验导致的固定点问题。

**💡 创新点**

创新点在于用可解释的、无编码器的社交度先验代替独立边E‑R先验，并在前向-后向过程中对该先验进行共扩散，使得图结构在高噪声下仍保持节点层面差异。

**🔧 技术方法**

技术手段包括离散边扩散的自适应转移矩阵、对数正态化的Latent Sociability Prior、联合Gaussian‑离散扩散、以及基于Graph Transformer的去噪网络。

**📊 数据集**

使用的实验数据集包括从Cora和Chameleon生成的合成子图集CoraChameleon（10–500节点），以及标准Ego和Community数据集。

**📈 对比分析**

与原始DiGress以及配备多种结构/谱辅助特征的版本对比，JointLSP+DiGress在无节点特征条件下在三角计数、节点亲和度等高阶统计量上显著优于基线，且不需要昂贵的谱特征。

**⚠️ 局限性**

局限性包括对拥有丰富节点语义特征的图（如分子图）收益有限；与辅助特征的融合不一定相加；谱特征的高成本及LSP本身已携带度异质性导致某些辅助特征冗余。

---

## 672. ReplaySSM: Cache SSM Inputs, Not State

**arXiv ID:** 2610.13778 | [PDF](https://arxiv.org/pdf/2610.13778v1)

**作者:** Ze-Wei Liou `[一作]` (Princeton University), Tri Dao `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReplaySSM，将 SSM 状态拆分为检查点和最近输入缓存，减少状态读写，提升解码速度并实现无成本回滚。

**💡 创新点**

核心创新是将状态表示为检查点+输入缓冲区，分离输出计算与状态更新，支持低精度检查点、输出仅验证以及指针式回滚。

**🔧 技术方法**

利用状态空间模型的递推展开、循环缓冲区、低精度量化、CUDA 图优化以及 vLLM 框架集成。

**📊 数据集**

在 Nemotron‑3、Qwen3.5 等模型上使用 PG‑19 文本、MMLU‑Pro、GPQA‑Diamond、Math‑L5 等评测数据集。

**📈 对比分析**

与调优的标准 vLLM 解码和传统快照规格化进行基准，单步吞吐提升 1.2–1.47×，推理速度提升约 1.8×，低精度检查点可将 KL 降至 1/36。

**⚠️ 局限性**

仍需读取检查点，头衰减极小的情况未充分利用，实现复杂度高，且在极短批次下收益有限。

---

## 673. Towards Physically Grounded Articulated 3D Assets: A Multi-Agent Pipeline and Benchmark

**arXiv ID:** 2610.13947 | [PDF](https://arxiv.org/pdf/2610.13947v1)

**作者:** Xudong Cai `[一作]` (Nanyang Technological University), Chuanxia Zheng `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种多代理（Design、Coding、Review）管线，通过物理模拟反馈实现物理上可行的关节化 3D 资产生成，并同时构建涵盖空间保真、物理属性、下游任务表现和整体质量的统一基准。

**💡 创新点**

创新点：①将生成流程拆分为独立代理，消除自评偏差；②在生成循环中嵌入 MuJoCo 物理验证，直接纠正碰撞和稳定性问题；③设计统一评估基准，系统评估视觉、几何、物理与功能表现，突破单一维度评测的局限。

**🔧 技术方法**

技术：大语言模型（LLM）驱动的多代理系统；Articraft SDK 与可复用的标准关节连接器；Python 编程生成 3D 程序；MuJoCo 物理仿真用于碰撞与重力稳定性检测；多指标评估框架（Chamfer、IoU、LPIPS、LPIPS、CLIP、质量评分等）。

**📊 数据集**

数据集：ArtVIP、LightWheel（空间保真）；ABO‑500、50 个稳定性对象（质量与碰撞）；Articulated 参考模型；RoboTwin 2.0 与 RoboCasa365（下游机器人任务）；自建 benchmark 对象与提示集。

**📈 对比分析**

比较方法：与三类基线对比——任务专用学习模型（Instruct‑Particulate、PhysX‑Anything、URDF‑Anything+）、专用代理生成模型（Articraft、Articulate‑Anything）、通用编码代理（Codex、Claude、Antigravity）。实验显示 GPT‑6 Astra 在 11 项空间保真指标中 8 项领先，碰撞率仅 1.7%/13.0%，稳定率分别 98%/96%；机器人任务成功率、资产可用率和人工评估 Top‑1 率（43.3%）均优于对比方法。

**⚠️ 局限性**

局限性：①物理验证依赖 MuJoCo，无法覆盖所有现实物理现象；②多代理协调与调度成本高；③仅评估有限的物理属性（质量、碰撞、重力稳定），对复杂材料与动力学未做深入；④缺少真实机器人硬件验证，生成的资产在实际硬件中的可行性尚未完全验证。

---

## 674. Noise Space Learning for VLA Adaptation

**arXiv ID:** 2610.13943 | [PDF](https://arxiv.org/pdf/2610.13943v1)

**作者:** Junjie Lu `[一作]` (University Of Technology Sydney), Li Zhao `[通讯]` (Microsoft Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为噪声空间学习（NSL）的方法，通过学习一个上下文条件的噪声分布来适配预训练的视觉-语言-动作（VLA）流匹配模型，同时保持 VLA 主干不变，从而实现高效的任务适配。

**💡 创新点**

创新点包括：
1) 将噪声空间学习构造为一个统一的 VLA 适配接口；
2) 用信息论分析证明 NSL 在信息成本上能达到行为下界；
3) 针对 NSL 设计了三种优化器（PPO、SAC、AWR）并对实现细节进行系统调优。

**🔧 技术方法**

技术手段包括：
- 预训练的流匹配 VLA（SmolVLA、π_0.5）；
- 条件对角高斯噪声演员；
- PPO、SAC、AWR 三种强化学习优化器；
- 任务平衡、噪声限制、成功/失败平衡等实现细节。

**📊 数据集**

使用的数据集与任务：
- 仿真基准 LIBERO-10、RoboTwin；
- 真实机器人四个操作任务（杯子堆叠、玩具钓鱼、顺序插入、珠子放置）；
- 预训练数据为 379 条演示或每任务 1 条演示。

**📈 对比分析**

方法比较与性能：
- 与 Action-AWR（生成器适配）和 RLT（残差适配）对比；
- 在 LIBERO-10 上，NSL‑SAC 达到 95.8%（SmolVLA‑Full）/99.8%（π_0.5‑Full）的峰值成功率；
- NSL‑PPO 在 Full 情况下几乎相当于 NSL‑SAC；
- NSL‑AWR 在 One 情况下与 Action‑AWR 相近，并在真实机器人实验中在 1.53 分更新时间下提升平均成功率至 71.3%，人类指导后提升至 83.8%。
- 计算成本：NSL‑SAC 更新时间仅 1/21.2，NSL‑AWR 最低仅 1.09 秒/批。

**⚠️ 局限性**

局限性：
- 固定生成器限制了可学习的行为空间；
- 噪声演员的简单高斯参数化可能不足以捕捉复杂任务的细粒度控制；
- 在真实机器人上，NSL‑SAC 的噪声偏离标准正态导致不安全运动；
- 对新任务、不同机器人硬件的泛化、已有技能的保持以及与环境交互的可靠性仍待进一步研究。

---

## 675. Images Talk Back: Guiding MMDiTs by Suppressing Image-to-Text Feedback

**arXiv ID:** 2610.13941 | [PDF](https://arxiv.org/pdf/2610.13941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 676. Chronos Policy: Learning Robot Policies on Physical Time

**arXiv ID:** 2610.13929 | [PDF](https://arxiv.org/pdf/2610.13929v1)

**作者:** Changhao Wang `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Chronos政策，通过在机器人决策过程中显式使用观测捕获时间和动作执行时间来改进动态与多模态操作

**💡 创新点**

创新点在于：①使用时间感知RoPE将物理时间嵌入注意力机制，保持令牌内容不变；②训练时对观测与动作时序进行随机化，使模型在各种时序噪声下保持鲁棒；③统一的物理时间接口同时处理传感率、测量延迟与执行延迟

**🔧 技术方法**

技术包括扩散变压器、时间感知RoPE、时间随机化训练、异步动作块预测以及实时块切换（RTC）

**📊 数据集**

在仿真中使用Conveyor（动态取放）、Pivoting（多模态接触）、DexMimicGen六个准静态任务；实地实验使用Dynamic Cup、Peg Insertion、Pivoting三个真实机器人任务

**📈 对比分析**

与传统固定速率扩散政策（Diffusion Policy）及其RTC版本相比，Chronos在观测/推理时序扰动下成功率提升约30–70%（如Conveyor从15%涨至76%，Pivoting从19%涨至38%），并在准静态任务中保持原有性能且对时序变化更稳健

**⚠️ 局限性**

局限性包括：目前仅支持固定动作步长；对极端大延迟或非线性时钟漂移的适应性未充分验证；需要每个观测和动作都携带准确时间戳，若时间戳缺失或错误可能导致性能下降

---

## 677. Hide and Seek: HIDE Concepts to SEEK explanations for black box models

**arXiv ID:** 2610.13935 | [PDF](https://arxiv.org/pdf/2610.13935v1)

**作者:** Bhanu Tokas `[一作]` (Arizona State University), Hannah Kerner `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Hide and Seek (HnS) 方法，用 vision‑language 模型检索概念图像并遮蔽对应区域，直接测量黑盒视觉模型在全局与局部层面的概念归因。

**💡 创新点**

创新点在于：1）不依赖概念标注数据；2）可解释任何黑盒模型；3）同时给出全局与局部（图像级）归因；4）支持分类、回归、分割、字幕等多种任务；5）采用 VLM 检索 + 直接差分概率，提升归因可信度。

**🔧 技术方法**

核心技术包括：SigLIP（或 CLIP、BLIP 等 VLM）文本向量获取、模态校正与 Löwdin 正交化以保证概念向量独立、基于余弦相似度检索代表图像、利用 patch‑wise 相似度阈值定位概念区域、遮蔽策略（模糊、零填充等）、通过遮蔽前后模型输出差值计算归因，并对噪声概念做 Cohen’s d 正则化。

**📊 数据集**

实验数据集：CelebA（人脸属性）、COCO‑Stuff（物体/场景概念）、COCO Captions（图像字幕）、Fields of the World（农田图像）以及 BigEarthNet（遥感概念）。

**📈 对比分析**

与 TCAV、PCBM 等现有方法在全局归因上进行 Spearman ρ 对比，HnS 在 CelebA 上 ρ 从 0.319–0.615 最高，在 COCO‑Stuff 上 ρ 0.030–0.342 最高，整体显著优于传统方法；在局部归因检索上，HnS 在 COCO‑Stuff 能准确定位概念区域，TCAV 则表现欠佳。

**⚠️ 局限性**

局限性包括：1）对概念向量的分离度依赖，类似概念会导致归因噪声；2）仅适用于具有空间可定位的概念，抽象概念无法遮蔽；3）遮蔽策略可能无法完全隐藏概念，导致归因误差；4）对极少数图像中的概念识别不稳定。

---

## 678. Designing Protein Structures from Natural Language

**arXiv ID:** 2610.13917 | [PDF](https://arxiv.org/pdf/2610.13917v1)

**作者:** Joseph Lee `[一作]` (University of Washington), Pang Wei Koh `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Prompteina，一种通过自然语言直接生成全原子蛋白结构和序列的模型，并构建了大规模文本–结构对照数据集 Prompteina-Data。

**💡 创新点**

创新点在于用单一自然语言接口实现全尺度结构与功能控制，结合文本–残基对齐监督，突破传统需要多种专用输入的局限。

**🔧 技术方法**

技术包括：基于预训练 Complexa 的扩散/流匹配生成器、Qwen3 文本嵌入、跨注意力与聚合注意力的文本条件化、对齐头监督、分类器无关引导等。

**📊 数据集**

数据集为 Prompteina-Data，包含约 1.8 M 结构和 52.4 M 条自然语言描述（覆盖折叠、二级结构、二硫键、活性位点、结合界面等）。

**📈 对比分析**

通过与 RFdiffusion3、BoltzGen、ESM‑3 等基线对比，Prompteina 在 18 项结构属性中 15 项获胜；在 20 目标的结合器设计中至少 13 项优于基线，界面置信度提升 14%；迭代重设计时比最强基线高 57% 的改进；实验验证 GM2A 目标，识别 9 个结合体，最佳 Kd 107 nM。

**⚠️ 局限性**

局限性包括对大规模 LLM 生成文本描述的依赖、对残基–文本对齐准确度的限制，以及在极大规模或高度多模态设计任务中可能需要进一步的模型扩展和更精细的对齐策略。

---

## 679. Structural Automata for Neural Semantic Parsing

**arXiv ID:** 2610.13896 | [PDF](https://arxiv.org/pdf/2610.13896v1)

**作者:** Xiao Zhang `[一作]` `[通讯]` (University of Groningen), Xiao Zhang (University of Groningen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在语义解析中提出一种统一的结构自动机，并在解码、后处理和训练三阶段应用，特别提出接受路径解码（APD）来保证在固定长度预算内输出合法且不回退。

**💡 创新点**

创新点在于：①统一同一结构自动机服务于三阶段；②设计了APD，提供预算内的接受保证并与回滚等方法竞争；③通过量化“距离到接受”“失败尾”“拒绝词重”三种指标，揭示不同阶段的瓶颈。

**🔧 技术方法**

使用的技术包括：SBN（Sequence Box Notation）结构自动机、接受路径解码、局部掩码、固定预留、回滚、采样/束搜索过滤、未接受词惩罚（UL）以及LoRA微调的Qwen3.5-0.8B模型。

**📊 数据集**

使用的数据集为：PMB 5.1的DRS SBN格式（9,057训练/1,132验证/1,132测试）以及AMR 3.0测试集（1,898图），训练样本分别随机取600、1,256和全套9,057句。

**📈 对比分析**

方法对比包括局部掩码、本地可行性检查、回滚、束搜索、采样、过滤等；在紧凑预算下APD相较局部掩码提升Smatch++约5–6点，几乎与回滚相当；在训练阶段，未接受词惩罚在早期显著提升开发Smatch++，并在完整训练后与交叉熵相当。

**⚠️ 局限性**

局限性：仅在英语DRS（SBN）和AMR上评估；仅使用单一模型且解码方式为贪婪；低资源实验使用随机子集而非真实低资源语料；验证仅基于自动机结构，未完全覆盖语义正确性；模型规模和多语言情况待进一步验证。

---

## 680. When VLMs Destroy What They Know: Spatial Representations Fall Below Lexical Priors

**arXiv ID:** 2610.13910 | [PDF](https://arxiv.org/pdf/2610.13910v1)

**作者:** Shubh Garg `[一作]` (Thapar Institute of Engineering and Technology), Chen Shani `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CASP-VL 这一控制性空间扰动基准和 EICS 这一测度内部表征稳定性的指标，用来评估视觉-语言模型（VLM）是否在内部表示中保留空间关系信息。

**💡 创新点**

创新点在于：①首次将概念稳定性（concept stability）与一概念干预相结合，设计了可解释且可量化的评估方法；②构建了大规模、基于 Visual Genome 场景图的可控扰动数据集；③通过与简单 BoW 词袋基线对比，揭示 VLM 在空间关系编码上的系统性失败，突出输出评估无法捕捉的内部表示缺陷。

**🔧 技术方法**

采用 CLIP、SigLIP、LLaVA 等主流 VLM 的文本编码器；对不同层的隐藏状态使用线性逻辑回归探针（probe）进行解码；对词袋基线使用词计数向量进行同样的线性探针；计算 EICS 值并与基线及现有合成评测（如 SugarCrepe、ARO）比较。

**📊 数据集**

使用 Visual Genome 场景图生成的 CASP-VL 数据集，其中包含 2,400 个颜色干预三元组、1,491 个空间干预三元组及 4,200 个跨类别三元组；此外还利用了 CLIP 训练时的图像-文本对进行可视化对齐分析。

**📈 对比分析**

对比 EICS 与 BoW 基线、以及在 SugarCrepe、ARO 等任务上的准确率。结果显示：颜色干预的 EICS 在后期层接近 BoW 基线（≈0.31 vs 0.33），而空间干预的 EICS 则远低于 BoW（≈0.07 vs 0.37），即空间关系信息在 VLM 表征中被大幅丢失；不同模型（CLIP、SigLIP、LLaVA）均呈现同一趋势，证明这一缺陷普遍存在。

**⚠️ 局限性**

局限性：①仅对文本编码器进行探针，未评估图像编码器或跨模态表示；②实验模型仅覆盖少数 VLM，未检验更大规模或专门加入空间监督的模型；③使用模板句子生成的 caption 可能与训练分布不一致；④探针为线性模型，无法完全排除非线性编码的可能性。

---

## 681. Can Language Models Play the Wikipedia Game?

**arXiv ID:** 2610.13948 | [PDF](https://arxiv.org/pdf/2610.13948v1)

**作者:** Alexander Hickey `[一作]` (Alan Turing Institute), James Bishop `[通讯]` (Alan Turing Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建多语言维基百科和OpenAlex学术文献的大型图数据集，利用维基百科游戏评估句子嵌入模型和大型语言模型在信息检索/导航任务中的表现，并与人类与TF‑IDF基准进行对比。

**💡 创新点**

①首次在本地构建覆盖五种语言与五个学科的大规模可导航图；②将LLM、句子嵌入、统计方法统一放入同一游戏框架中进行公平比较；③探索LLM推理（chain‑of‑thought）对导航效率的影响。

**🔧 技术方法**

句子嵌入模型（E5‑Small、AllMPNetBase等）、大型语言模型（GPT‑4o、Llama、DeepSeek、Qwen 等）与其推理版本、TF‑IDF 统计检索、图遍历与步数统计。

**📊 数据集**

五种语言的维基百科（英语、法语、中文、俄语、孟加拉语）与五个学科的 OpenAlex 论文引用网络（生物学、医学、物理、社会科学、艺术），并在此基础上构造可导航图。

**📈 对比分析**

随机生成 10,000 对可达文章，限制 100 步完成游戏；记录成功率、步数分布与人类玩家、TF‑IDF 的对比。结果显示：最佳 LLm（GPT‑o3 mini、GPT‑4o、Qwen‑QwQ、DeepSeek‑DS）成功率与人类相当，但平均步数略高；部分句子嵌入模型在某些图上表现更好；TF‑IDF 在部分情境仍具竞争力；模型在不同语言与学科间的排名不稳定。

**⚠️ 局限性**

①LLM 受 API 成本限制，实验规模受限；②LLM 易产生幻觉，影响成功率与效率；③仅测试有限的语言与学科，未覆盖低资源或非英语情况；④推理 LLM 计算成本高，实际应用受限；⑤数据集更新滞后，模型对新发表的文章连通性评估不足。

---

## 682. Splat2Act: From Captured Scenes to Personalized VLA Manipulation

**arXiv ID:** 2610.13945 | [PDF](https://arxiv.org/pdf/2610.13945v1)

**作者:** Xiaomeng Ye `[一作]` (Georgia Institute of Technology), Zsolt Kira `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `51c0528b-f690-4182-ae60-bb5f046c276c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了一套自动化的真实到仿真再到真实的管线，能够仅凭手持相机扫描和少量目标物体图片，生成建筑尺度的数字孪生、自动发现托盘/接收器，并通过仿真专家生成拾取演示，从而为VLA策略做个性化微调。

**💡 创新点**

创新点在于：① 通过3D高斯散点渲染实现建筑尺度场景的真实感仿真；② 完全自动的托盘/接收器检测与任务组装；③ 两阶段自监督微调（广泛适应+场景个性化）使预训练VLA在新用户环境下实现高效迁移；④ 通过仅手持扫描实现从真实到仿真的无人工干预。

**🔧 技术方法**

使用了3D Gaussian Splatting、DN‑Splatter、SuGaR构建光照真实的数字孪生；SAM3D与Objaverse生成物体模型；HDBSCAN进行接收器聚类；RoboSimGS/Isaac Sim的仿真专家进行演示生成；并采用Flow‑Matching的监督微调来优化π_0.5策略。

**📊 数据集**

数据集包括101个校园室内场景（共计约11,900 m²可行走面积，852个接收器），从中生成68,021条成功拾取演示；个性化阶段使用7个真实物体（药瓶、喷雾罐、番茄汤、纸箱、魔方、喷漆罐、水瓶）。

**📈 对比分析**

通过在两块部署场景（Office、Conference）上与π_0.5基准、单场景和多场景广泛适应对比，最终在Office场景中个性化策略实现62.5 %拾取成功率；在实际Stretch 3上对应任务的成功率为63.9 %，与仿真差距仅8 %。实验还表明相同场景的广泛适应比跨场景适应更能提升个性化性能。

**⚠️ 局限性**

局限性包括：① 对场景的数字孪生质量仍受扫描质量和光照条件影响；② 仿真专家生成演示需要大量计算资源；③ 个性化策略在未见场景上表现不稳健；④ 仅依赖手持扫描，若用户不愿扫描或扫描不足，无法生成合适的模型。

---

## 683. AI Models Can Provably Hide Arbitrary Capabilities

**arXiv ID:** 2610.13928 | [PDF](https://arxiv.org/pdf/2610.13928v1)

**作者:** Andis Draguns `[一作]` (Contramont Research), Christian Schroeder de Witt `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用数字锁和加密神经网络的后门攻击，能在模型权重中隐藏任意可执行计算，并在触发器激活时解密并执行，而在普通输入下保持原始行为。

**💡 创新点**

创新点在于将加密技术与神经网络编译相结合，构建可证明不可被白盒审计激活的隐藏能力，突破传统基于密码锁的后门局限，提供了可扩展到任意计算的加密后门框架。

**🔧 技术方法**

使用PyTorch框架实现，核心技术包括：SwiGLU MLP与RMSNorm结构、线性阈值门(LTG)电路编译为GLU、Keccak哈希构成的数字锁、加密权重的解锁与执行单元。

**📊 数据集**

实验数据集包括：4位子集奇偶子任务（Subset Parity）和从USPTO数据提取的化学反应可行性二分类任务（Morgan指纹64维）。

**📈 对比分析**

与传统密码锁后门（Password‑locked）模型对比，使用监督微调(SFT)、权重噪声、梯度触发搜索(GCG)和潜在对抗训练(LAT)等方法。结果显示加密模型在所有方法下均保持接近随机表现，未被诱导；而密码锁模型则易被SFT和GCG显著诱导，表现提升近90%。

**⚠️ 局限性**

主要限制包括：参数规模巨大（单层执行单元导致二次宽度开销）、仅适用于小规模网络、目前未集成至大型预训练模型、检测隐藏后门存在仍可能可行，且实现需高度加密与机器学习专业知识，实用性受限。

---

## 684. DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness

**arXiv ID:** 2610.13918 | [PDF](https://arxiv.org/pdf/2610.13918v1)

**作者:** Daniel Szelogowski `[一作]` `[通讯]` (University of Wisconsin--Whitewater), Daniel Szelogowski (University of Wisconsin--Whitewater)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个多尺度的生物物理睡眠模型DyssomniaNN，结合Borbély的两过程调节、Saper的互抑翻转开关以及皮层-丘脑的LIF突触网络，在同一72小时光周期内模拟健康、延迟睡眠节律障碍、慢性失眠、悖论性失眠和嗜睡症类型1五种临床表型的睡眠结构和电生理信号。

**💡 创新点**

创新点在于首次将宏观的睡眠调节机制与亚脑层次的翻转开关以及细胞层次的突触网络无缝耦合，产生可持续的多天睡眠轨迹，并引入Well‑Rested指数量化恢复效能，揭示睡眠时长与恢复感受之间的解耦。

**🔧 技术方法**

主要技术包括：两过程调节差分方程、Wilson‑Cowan互抑模型、100细胞的可耕耘LIF网络、LFP合成与α‑函数卷积、FFT谱分析以及Well‑Rested指数的公式化。全部实现于PyTorch框架。

**📊 数据集**

使用PhysioNet Sleep‑EDF数据库的C3‑A2 EEG通道作为验证基准，获取正常深睡期的谱能量分布。

**📈 对比分析**

通过计算合成LFP与真实EEG的谱均方误差（MAE）在0.33–0.35之间，表明模型在所有经典频段（δ、θ、α、σ、β）上均能逼近人类数据；对各表型的睡眠周期、睡眠时间、觉醒后失眠等临床指标也与既定特征相符。

**⚠️ 局限性**

局限性包括：仅使用单通道（C3‑A2）LFP，缺乏空间分布和跨区域网络；REM睡眠细节仅用阈值参数化，未模拟完整的下丘脑-脑干循环；药物效应目前为静态参数调节，未纳入PK/PD模型；需要进一步个体化参数校准以实现精准的数字双胞胎。

---

## 685. Where to Look Matters: Rethinking Sub-Volume Sampling in 3D Medical Self-supervised Learning

**arXiv ID:** 2610.13915 | [PDF](https://arxiv.org/pdf/2610.13915v1)

**作者:** Junkai Liu `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一个无训练、可插拔的3D医学影像自监督学习子体裁取样框架VolumeProbe，通过离线构建包含解剖可行性、信息量和多样性三原则的probe bank，替代随机裁剪。

**💡 创新点**

创新点在于把子体裁取样视为“where-to-look”问题，引入解剖先验、梯度+熵信息量和冗余惩罚构造probe bank，并证明轻量几何与统计特征已足够，无需昂贵语义特征。

**🔧 技术方法**

技术上实现了粗细两级候选生成、三原则评分、贪心多样性选取，提供了两个变体：轻量版与冻结特征版，并在预训练时直接采样。

**📊 数据集**

使用了多种3D医学影像数据集：BTCV、AMOS、WORD、Atlas-MRI、CC-CCII、IXI、OASIS等。

**📈 对比分析**

与多种基线（MAE3D、SimMIM、MoCo v3、Swin-UNETR、GVSL、GL-MAE、MIM、VoCo、S2DC、MedGMAE）对比，VolumeProbe在分割、分类和配准任务上均提升了Dice/准确率/对齐度，尤其在低标注比例下表现突出。

**⚠️ 局限性**

局限在于probe bank是静态离线生成，无法随模型训练动态调整，也未针对非标准采集条件提供灵活先验。

---

## 686. RSI-Anything: Unlocking Recursive Self-Improvement with World Models

**arXiv ID:** 2610.13912 | [PDF](https://arxiv.org/pdf/2610.13912v1)

**作者:** Taoyong Cui `[一作]`, Ling Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RSI-Anything框架中，将预测世界模型与递归自我改进相结合，提升了代码修复、GUI交互、实验室规划、化学反应优化和免疫学推理等多域任务的性能。

**💡 创新点**

创新点在于将跨域预测世界模型作为候选评估器，结合条件适配与保留机制，实现了在外部反馈有限的情况下的自适应改进，并在多领域展示了递归改进的可行性。

**🔧 技术方法**

采用JEPA-Anything预训练的联合嵌入预测模型、OPF分解、强化学习/偏好学习策略、贝叶斯优化以及自回归语言模型等技术。

**📊 数据集**

使用 CodeContests、APPS、LiveCodeBench、MiniWoB++、OSWorld、LabHorizon F200、Suzuki–Miyaura实验数据和免疫学实验扰动集。

**📈 对比分析**

通过与静态验证器、随机选择、RLAnything、贝叶斯优化和Actor–Simulator–Selector等基线对比，代码修复成功率提升至31.2%，GUI任务完成率从51%升至99%，实验室规划得分从21.8%提升至46%，化学反应最高产率分别为93%和96.15%，显著优于对照组。

**⚠️ 局限性**

局限在于缺乏真实物理执行验证、仅在单任务循环中评估递归改进、未探索模型与策略在同一任务中的协同演化，以及样本规模和跨域通用性的进一步验证需求。

---

## 687. Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping

**arXiv ID:** 2610.13909 | [PDF](https://arxiv.org/pdf/2610.13909v1)

**作者:** Danil Rudenko `[一作]` (Yandex Research), Artem Babenko `[通讯]` (Yandex Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估多种回归目标函数在Tabular深度学习中的表现，并提出AdaQR自适应分位数回归方法

**💡 创新点**

发现分位数损失在多种骨干上表现稳定，提出AdaQR通过输入相关平滑减缓不同噪声区域的过拟合并提升性能

**🔧 技术方法**

采用多种回归损失（MSE、MAE、Huber、Hard/Soft Binning、HL‑Gaussian、分位数损失、Gaussian NLL 等），使用 MuON 优化器、MLP、TabPack、TabM、RealMLP 骨干，以及自适应分位数损失 AdaQR

**📊 数据集**

使用 41 个 BeyondArena 回归数据集（其中 28 个样本量≥1000）以及合成异方差数据集

**📈 对比分析**

对比方法通过大规模超参搜索（最多 100 次）评估 RMSE，结果显示在 MuON+MSE 基准下分位数损失平均提升约 0.5%；AdaQR 在多数骨干上进一步提升约 0.8%，但增益在不同数据集与骨干间不均匀

**⚠️ 局限性**

局限包括 AdaQR 对超参 λ 敏感、在小样本数据集缺失、对高噪声区域的停止冲突未完全消除、对 Tiny 数据集未评估，且对训练动态的理论解释仍不足

---

## 688. Nimbus Studio reproduces reference BCI decoding pipelines: a validation study on public benchmarks across protocols and paradigms

**arXiv ID:** 2610.13894 | [PDF](https://arxiv.org/pdf/2610.13894v1)

**作者:** Sergey Musienko `[一作]` `[通讯]` (Nimbus BCI), Sergey Musienko (Nimbus BCI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对Nimbus Studio执行引擎与独立脚本实现进行等价性验证，比较它们在多种BCI解码管线、评估协议和数据集上的表现。

**💡 创新点**

提出一种外部可复现的等价性验证范式，证明视觉管道平台在计算结果上与传统脚本无差异，提供对平台可信度的量化评估。

**🔧 技术方法**

使用MNE-Python、scikit-learn、pyriemann、braindecode等标准库实现BCI解码管线；在Nimbus Studio中通过执行引擎、跨验证驱动和隔离worker节点运行同一管线。

**📊 数据集**

采用MOABB公开数据集：BNCI-2014-001、BNCI-2014-004（运动想象）和BNCI-2014-009（P300）。

**📈 对比分析**

在相同数据、超参数和随机种子下，比较每条管线在session holdout和leave‑one‑subject‑out协议下的测试准确率；所有确定性管线误差为0.00点，深度学习管线误差落在训练种子波动范围内，整体性能与参考实现一致。

**⚠️ 局限性**

局限性包括仅验证解码准确率等价性，未覆盖BIDS导出、流式推理和云沙盒；深度模型等价性仅在随机波动内；参考实现由平台作者编写，需外部复现验证。

---

## 689. InstantFusion: A Shared Latent Interface for Heterogeneous Image Generators

**arXiv ID:** 2610.13944 | [PDF](https://arxiv.org/pdf/2610.13944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 690. Beyond the Snowflakes: Disentangling Weather Effects from Confounding Meta-Factors in LiDAR Object Detection

**arXiv ID:** 2610.13891 | [PDF](https://arxiv.org/pdf/2610.13891v1)

**作者:** Yiming Yang `[一作]` (Michigan Technological University), Jeremy P. Bos `[通讯]` (Michigan Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在恶劣天气下评估 LiDAR 目标检测的鲁棒性，发现更大降雪量导致检测性能提升，并通过元信息分析解释这一现象。

**💡 创新点**

提出将降雪量与物体点数等元信息解耦，揭示物体点数增加是导致检测提升的关键因素，并对三种物理假设进行了实证验证。

**🔧 技术方法**

采用多种深度学习检测器（voxel、pillar、point-based、hybrid），做 ablation、误差矩阵、Kendall Tau、互信息、RF+SHAP 等元信息分析。

**📊 数据集**

使用 WADS‑3D 极端降雪数据集和 CADC 传统降雪数据集，结合公开的 KITTI 训练模型。

**📈 对比分析**

通过对比未滤雪与滤雪的 AP，证明即使降雪会产生噪声，性能提升仍由物体点数增加驱动；不同架构下 AP 最高的 voxel 模型在降雪中表现最好。

**⚠️ 局限性**

仅依赖天气强度分箱评估会误判鲁棒性，未考虑物体点数等共变元信息；实验仅聚焦汽车，未覆盖行人等小目标；需要更真实的物理模拟与多模态融合验证。

---

## 691. How Social Reinforcement Shapes Word-of-Mouth: A Conviction-Based Model for Marketing Diffusion and Network Targeting

**arXiv ID:** 2610.13890 | [PDF](https://arxiv.org/pdf/2610.13890v1)

**作者:** Abhishek K. Umrawal `[一作]` (University of Illinois Urbana-Champaign), Eliot W. Robson `[通讯]` (Narmi Inc.)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Pressure Threshold (PT) 模型，用来捕捉社交强化对口碑传播的影响，并探究该机制对营销定向和种子选择的影响。

**💡 创新点**

核心创新在于将个体在采用时经历的累计社会压力直接映射为其后续影响力的增强，使得模型既保留了阈值模型的可解释性，又能体现“信念驱动”式的影响放大。

**🔧 技术方法**

技术方法包括：在 Linear Threshold 框架上加入一次性影响力调整阶段；构建 Sandwich Approximation 框架（上下界 M⁺ 与 LT）；使用 CELF 等启发式算法；以及通过蒙特卡罗模拟和时间戳级联数据进行验证。

**📊 数据集**

实验数据集包括四个网络：Facebook 社交圈、Wikipedia 投票、Bitcoin OTC 信任网络和 Erdős–Rényi 随机网络，并利用真实的社交媒体级联记录进行预测评估。

**📈 对比分析**

通过与传统的 LT/IC 模型比较，PT 在密集或团聚网络中显著提升种子传播效果，预测准确率比 LT 提高 10–30%；算法上，Sandwich Approximation 与 CELF 在大多数网络上取得与 LT 相当或更优的传播规模，且在 Facebook、Wikipedia 等真实网络上表现尤为突出。

**⚠️ 局限性**

主要局限包括：PT 的影响函数不具备单调性与子模性，导致传统的 1‑1/e 近似理论失效；尽管提出了数据相关的 Sandwich 近似保证，但缺乏普适的理论上限；此外，模型中 α 参数需要经验调优，对不同场景的敏感度较高。

---

## 692. Constructive Arboricity in Almost-Linear Time for Dense Graphs

**arXiv ID:** 2610.13937 | [PDF](https://arxiv.org/pdf/2610.13937v1)

**作者:** Iman Gholami `[一作]`, AmirMohammad Shahrezaei `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种随机算法，用于计算简单图的确切树状性，并构建最优的森林划分。

**💡 创新点**

该算法是第一个已知的几乎线性时间算法，能够构建出最多比最优森林多一个的森林划分。

**🔧 技术方法**

使用了随机化算法和几乎线性最大流算法的突破。

**📊 数据集**

适用于所有简单无向图，特别是稠密图（m=Θ(n^2)）。

**📈 对比分析**

与现有方法相比，该算法在期望和高概率下的运行时间为O(min{n^2+o(1), m^4/3+o(1)})，在稠密图上几乎线性，O(m^1+o(1))。

**⚠️ 局限性**

算法的局限性在于它依赖于随机化过程，可能在某些情况下导致较高的失败概率。

---

## 693. Goal-Oriented Bundle Management and Flow Control for Deep-Space Communications via Reinforcement Learning

**arXiv ID:** 2610.13923 | [PDF](https://arxiv.org/pdf/2610.13923v1)

**作者:** Sajjad Baghaeea `[一作]`, Elif Uysal `[通讯]` (Middle East Technical University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于强化学习的深空DTN中转节点的目标导向包调度与丢弃控制框架，利用到达目标信息的新鲜度作为奖励，最小化平均超时年龄；

**💡 创新点**

创新点在于将剩余年龄差作为目标，引入半马尔可夫决策过程与可行性掩码的Lagrangian PPO，实现既满足公平约束又提升信息及时性；

**🔧 技术方法**

采用事件驱动半马尔可夫决策过程、可行性掩码与多阶段动作、可变维度集合池化、带约束的Lagrangian PPO以及DTNSim仿真平台；

**📊 数据集**

使用DTNSim仿真环境，基于月球和火星随机Poisson生成的状态更新，评估时采用64个测试轨迹；

**📈 对比分析**

与FIFO、LIFO、EDF、Freshest-first四种基准在相同64轨迹上对比，平均超时AoI降至887.63秒，优于其他基准，并满足平均丢弃、失效与公平预算；

**⚠️ 局限性**

局限在于假设已知接触计划、单一控制转发节点、非抢占式不续传，对接触不确定性、多转发器控制以及真实DTN部署的验证仍需进一步研究。

---

## 694. Heterogeneous Demand-Private Coded Caching: Exact Characterization for Two Users and Two Files

**arXiv ID:** 2610.13914 | [PDF](https://arxiv.org/pdf/2610.13914v1)

**作者:** Tati V D N Eswar `[一作]` (Indian Institute of Technology Kharagpur), Jithin Ravi `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了一种在文件大小和缓存容量完全异质化条件下的需求私密码分发方案，并精确求解了两用户两文件场景下的内存‑传输率取值区域。

**💡 创新点**

创新点包括：①将虚拟用户技术推广到异质缓存大小；②利用受限需求子集构造需求私密非私密方案；③提出四个非平凡的、针对文件/缓存异质性的非对称缓存放置方案，完整描述了该场景下的最优内存‑传输率关系；④量化了在保持需求私密性时相较于非私密方案所需额外的传输成本。

**🔧 技术方法**

核心技术为：虚拟用户约简、受限需求子集构造、内存共享以及针对异质参数的显式编码与解码策略（如子文件划分、线性组合缓存）。

**📊 数据集**

论文采用理论分析与构造，未使用任何实验数据集；所有结果均通过信息理论证明与构造实现得到。

**📈 对比分析**

通过对两用户两文件情况的完整图形化取值区域（八条线性不等式）展示，与已知的非私密异质结果相比，六条不等式体现了私密性的传输率代价，且在特定区域内（DP7、DP8）可与非私密方案保持相同传输率。

**⚠️ 局限性**

局限性：仅在 N=K=2 的极小规模场景下给出完整解析；尚未推广到更多用户/文件、非均匀需求分布或非均匀文件尺寸的更一般情况；所用方法对更大系统的可扩展性尚未验证。

---

## 695. Fundamental Limits of Covert queueing under Busy-Period Observations

**arXiv ID:** 2610.13911 | [PDF](https://arxiv.org/pdf/2610.13911v1)

**作者:** Anumula Venkata Sai Sree Sahithi `[一作]` (International Institute of Information Technology), Jithin Ravi `[通讯]` (Indian Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在 M/M/1 排队系统中，Bob 通过插入额外的 Poisson 流量来为 Alice 提供服务，同时保持 Willie 无法检测到这些非 Willie 作业的存在，提出并证明了两种 Willie 观察统计量（忙期内服务的作业数 X 与忙期持续时间 Y）下的隐匿排队能力。

**💡 创新点**

创新点包括：① 对两种统计量均证明了隐匿排队的平方根定律；② 引入并解析了“隐匿排队常数”K_X 与 K_Y，定量衡量在不同统计量下可隐匿插入的作业数量；③ 证明 K_X ≤ K_Y，说明作业数统计比持续时间统计更能揭露 Bob 的行为。

**🔧 技术方法**

主要技术手段：KL 散度与泰勒展开（对作业数和忙期时间的概率分布分别展开到二阶）；信息熵与总变差距离的关系；数据处理不等式；Chebyshev 及大数定律用于构造 Willie 的检测阈值；通过对 M/M/1 过程的分析得到 X 与 Y 的分布形式。

**📊 数据集**

本研究仅使用理论模型（M/M/1 排队模型）进行推导，没有依赖实际数据集；所有结论均来自解析推导与数值积分（如 Bessel 函数相关积分）。

**📈 对比分析**

比较方法：通过 KL 散度上界和下界计算两种统计量下的隐匿插入上限，并得到精确的常数 K_X 与 K_Y。结果表明，两者均满足 E_b^*(n)=Θ(√n)，但 K_X 通常不大于 K_Y，意味着在相同隐匿约束下，利用作业数统计能插入更多作业。

**⚠️ 局限性**

局限性：① 只考虑单服务器 M/M/1 队列和 Poisson 作业插入；② 对 K_X、K_Y 的极限行为（λ_w→0 或 λ_w→μ）及更复杂服务时间分布未做深入解析；③ 只评估两种最常见的统计量，未探讨更复杂或多维统计量；④ 实际网络环境中的随机性、协作策略和多服务器情况未被覆盖。

---

## 696. Task Context Shapes Component Roles in Language Models

**arXiv ID:** 2610.13895 | [PDF](https://arxiv.org/pdf/2610.13895v1)

**作者:** Sovesh Mohapatra `[一作]` (University of Pennsylvania), Dani S Bassett `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多种开源大语言模型上，使用图形事实任务，量化并比较了相同注意力头在不同任务指令下的影响力与重要性，探讨内部任务编辑如何改变事实响应与后续计算。

**💡 创新点**

提出了任务依赖影响测量和内部任务编辑方法，证明共享组件的作用随任务上下文改变，从而解释相同参数在不同任务中的多重计算机制。

**🔧 技术方法**

利用激活干预、影响力与重要性测量、事实干预、内部任务编辑、响应映射以及线性预测等技术，对模型内部计算进行细粒度分析。

**📊 数据集**

采用含有三条边(A→B、B→C、A→C)的图形事实数据集，在Compose path与Read direct edge两种指令下进行对比实验。

**📈 对比分析**

通过角度差异、平均误差下降（Gemma 53.6%，Qwen 5%）以及任务特定预测的准确性提升，表明任务上下文显著改变影响模式并能提升对后续响应的预测。

**⚠️ 局限性**

局限于图形事实任务，对更长文本或更复杂推理场景的可推广性尚未验证，且仅评估了少数模型族，可能不足以涵盖所有共享组件行为。

---

## 697. Can machines reason non-verbally?

**arXiv ID:** 2610.13925 | [PDF](https://arxiv.org/pdf/2610.13925v1)

**作者:** Shaheer U. Saeed `[一作]` (Queen Mary University of London), Yipeng Hu `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出GenVerOpt框架，实现生成-验证-优化的视觉非语言推理流程，可对多种视觉任务进行个体化推理。

**💡 创新点**

创新点在于将非语言推理拆解为可检验的验证步骤，并通过样本级优化而非仅依赖数据集偏好，显著提升了抽象推理和医学影像分割的性能。

**🔧 技术方法**

采用生成模型（如基于SAM或随机采样的生成器）、验证网络（基于Reptile元学习的二分类器）以及强化学习自对弈优化策略。

**📊 数据集**

实验使用MM‑IQ抽象图形测试集、前列腺MR、肝CT、MuRegPro等多模态医学影像数据集。

**📈 对比分析**

与ChatGPT、Gemini、LLaVA等开源模型及多项最新医学分割方法相比，GenVerOpt在IQ测试准确率、前列腺肿瘤分割Dice、肝血管结构连通度等指标上均优于人类和专家水平。

**⚠️ 局限性**

局限性包括对验证模块的准确性高度依赖、对大规模复杂场景的可扩展性有限，以及需要一定量的领域标注数据以进行元学习与微调。

---

## 698. KL Before and After Mixing: A Round-Complexity Separation in Parallel Sampling

**arXiv ID:** 2610.13924 | [PDF](https://arxiv.org/pdf/2610.13924v1)

**作者:** Tianyi Wu `[一作]` `[通讯]` (Amazon), Tianyi Wu (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在对可并行采样的随机调度下，研究了两种KL散度度量（调度平均与输出法则）对采样误差的影响，并给出了在平衡切片以及其扩展目标（全支持Beta(1,1)-Bernoulli和加权切片）下的最优轮数上界与下界。

**💡 创新点**

首次证明在平衡切片中，调度平均KL要求Θ((log n)^2)轮，而输出法则只需Θ(log log n)轮，从而揭示两种度量之间的本质分离；并给出了对应的精确递推式与最优策略。

**🔧 技术方法**

采用了KL散度分解、计数状态递推、马尔科夫链与大数定理、Kolmogorov不等式、Hoeffding–Ville不等式、Cauchy–Schwarz以及数值实验与形式化验证（Lean 4）等技术。

**📊 数据集**

使用的“数据集”是理论上定义的分布：平衡切片（长度为n、n/2个1的均匀分布）、全支持Beta(1,1)-Bernoulli目标以及加权切片；实验中对n=65536进行核对。

**📈 对比分析**

与现有文献的调度平均结果进行比较，证明在输出法则下可实现显著更低的轮数；通过核对实验与Lean证明验证了理论界限的可达性，表现为在n=65536时仅需11轮即可满足ε=ln2。

**⚠️ 局限性**

局限性包括仅适用于具备精确条件概率的“原子”采样接口、仅计数轮数而不考虑计算成本、常数项未知、对非交换性或缺乏均匀权重组的加权切片无法给出精确结果，以及对学习模型的实际性能未做评估。

---

