# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-16 | 今日论文总数: 468

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Federated Explainable Artificial Intelligence: Roles, Architectures, Evaluation, and Open Challenges

**arXiv ID:** 2607.13045 | [PDF](https://arxiv.org/pdf/2607.13045v1)

**作者:** Masoume Gholizade `[一作]` (University of Pisa), Francesco Marcelloni `[通讯]` (University of Pisa)

**通讯引用:** 7165 | [OpenAlex ID](https://openalex.org/A5057031606)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本综述对联邦学习（Federated Learning, FL）与可解释人工智能（Explainable AI, XAI）的交叉领域 FedXAI 进行了系统梳理，提出了多轴分类框架以组织方法、评估维度及部署场景，并对研究现状、评估缺口与开放挑战进行了总结。

**💡 创新点**

创新点包括：①提出了覆盖角色、技术、范围、集成层次与 FL 设置五个维度的多轴分类体系；②强调解释在 FL 生命周期中的主动作用（不只是后置工具）；③系统性识别评估指标不足、隐私泄露与非 IID 情况下解释稳定性不足等关键挑战；④为后续研究提供统一的参考框架与评估思路。

**🔧 技术方法**

技术手段主要是：系统文献检索与 PRISMA 风格的系统综述流程；对 FL 与 XAI 的理论基础与关键技术进行归纳；构建多轴分类框架并对现有 FedXAI 论文进行映射；总结评估方法与指标；梳理安全隐私、通信成本等系统层面考虑。

**📊 数据集**

综述中未使用单一数据集，而是引用了公开数据集（如 Adult、CoverType、Intrusion Detection、医疗影像、能源预测等）作为示例，说明 FedXAI 方法在不同数据域与 FL 设置下的适用性。

**📈 对比分析**

比较主要通过文献中的实验结果进行对比：在相同任务下，FedXAI 方法在保持或提升模型精度的同时，提供可解释性；但由于缺乏统一基准与评估指标，各方法的性能、解释质量、稳定性、隐私泄露风险和通信开销等指标难以直接统一比较。

**⚠️ 局限性**

局限性：①缺乏统一的 FedXAI 评估指标与公开基准，导致跨论文比较困难；②解释在非 IID、异构环境下的稳定性与一致性不足；③隐私泄露与解释被篡改的安全风险尚未系统评估；④方法多样性导致实现与部署复杂，实际应用验证不足。

---

## 2. Disentangling Knowledge States with Ability and Proficiency Modeling for Knowledge Tracing

**arXiv ID:** 2607.13103 | [PDF](https://arxiv.org/pdf/2607.13103v1)

**作者:** Duantengchuan Li `[一作]` (Wuhan University), Mingwen Tong `[通讯]` (Central China Normal University)

**通讯引用:** 498 | [OpenAlex ID](https://openalex.org/A5078791358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种将学生交互序列分为能力期和熟练期的阶段感知识追踪模型，利用多分支Transformer和类型感知读出模块对各阶段信息进行联合建模。

**💡 创新点**

创新点在于基于累计正确次数阈值对学习过程进行细粒度分段，显式区分能力培养和熟练巩固两阶段，并通过多分支Transformer捕捉不同阶段的知识状态。

**🔧 技术方法**

采用了Transformer注意力机制的多分支解码器、阶段分割掩码、类型感知读出以及交叉熵+正则化损失，并引入因果分析解释模型优势。

**📊 数据集**

使用六个公开数据集：Algebra05、Assistments09、Assistments17、Ed‑Net、Slepemapy和Spanish。

**📈 对比分析**

与DKT、DKVMN、SAKT、SparseKT、SimpleKT和DisKT等六种基线进行5折交叉验证，AUC平均提升0.82%，最高提升1.33%，在所有数据集上均优于基线。

**⚠️ 局限性**

局限性包括对少量练习的概念缺乏熟练期信号，阈值k需手动调参且在不同数据集上差异大，且模型对极短序列的鲁棒性不足。

---

## 3. How You Ask Shapes What You Get: Auditing Breast-Cancer Misinformation in TikTok Search

**arXiv ID:** 2607.13147 | [PDF](https://arxiv.org/pdf/2607.13147v1)

**作者:** Pooriya Jamie `[一作]` (University of California, Los Angeles), Amir Ghasemian `[通讯]` (University of California, Los Angeles)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对TikTok搜索功能进行系统化审计，比较不同查询框架（医学信息、替代医学、同行叙事）在乳腺癌信息检索中的误信息曝光率。

**💡 创新点**

①首次大规模审计TikTok搜索；②结合LLM生成符合患者语言的查询词；③使用VLM零样本模型对视频截图进行误信息标注；④揭示查询框架显著决定误信息曝光水平。

**🔧 技术方法**

使用sock‑puppet自动化抓取、GPT‑5.4‑mini/​nano VLM进行图像文本标注、逻辑回归、位置加权曝光分析等技术。

**📊 数据集**

30个新建TikTok账号共检索9,020条结果（7,199条为癌症相关），查询词基于r/breastcancer热门帖子标题生成。

**📈 对比分析**

按查询框架和症状/治疗阶段分组计算误信息率，使用Wilson CI与对数回归比较差异；结果显示替代医学框架误信息率约为医学信息的7.6–8.6倍，而医学信息框架仍有6%–7%的误信息。

**⚠️ 局限性**

误信息标签依赖模型，缺乏足够人工验证；仅覆盖搜索结果而非For‑You‑Page；查询生成可能不完整，时间窗口短；不同框架词义混杂导致因果推断受限。

---

## 4. Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes

**arXiv ID:** 2607.13188 | [PDF](https://arxiv.org/pdf/2607.13188v1)

**作者:** Minh-Quan Le `[一作]`, Di Qiu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在 Masked Diffusion Model（MDM）框架下实现多模态（文本‑图像）并行生成的自纠正耦合马尔科夫跳跃过程（SC‑CMJP）以及其无训练、单前向推断的采样器（SC‑COUPLED JUMP）。

**💡 创新点**

创新点：①通过跨模态注意力将一模态的转换速率函数化为另一模态置信度的加权，形成在每个去噪步骤内的实时耦合；②引入 remasking 跳跃，在发现跨模态矛盾时立即撤销错误决策，实现自纠正；③构造无训练、单前向通行的采样流程，保持模型冻结且不需额外评估器；④针对多模态任务创建统一训练目标，隐式逼迫网络学习跨模态推断信号。

**🔧 技术方法**

技术：Masked Diffusion Models (MDM) 的连续时间马尔科夫跳跃过程；跨模态注意力与置信度门控（entropy‑based gating）；Remasking 机制 (ReMDM‑style)；单前向推断采样算法；共享百分位排名归一化；训练目标为联合 NELBO。

**📊 数据集**

数据集：JEdit‑1M（1M 张图像编辑+理解样本，包含 Prompt、源图像、目标图像、场景图及推理轨迹）；JMaze‑200K（200K 迷宫求解样本，图像与路径文本）；JNono‑200K（200K 非ogram 求解样本，图像与填充矩阵文本）。每个数据集均提供 OOD 验证集。

**📈 对比分析**

比较方法：基线为 MDM、ReMDM、MMaDA‑Parallel。评测指标包括：文本生成 perplexity、图像编辑 ImgEditBench 分数、图像理解 mAP@0.5:0.95、联合准确率（文本与图像均正确）。结果显示：SC‑COUPLED JUMP 在所有任务上均优于基线；在 ImgEditBench 上实现 1.93 的得分，mAP 从 0.074 提升至 0.369；在迷宫与非ogram 任务上联合准确率提升 0.008‑0.062；采样步数增加时性能单调上升，展示跨模态耦合效益随轨迹累积。

**⚠️ 局限性**

局限性：①方法目前仅验证在文本‑图像对上，未测试音频/视频等更异构模态；②依赖 MDM 的预训练与冻结，缺乏针对特定任务的微调机制；③跨模态注意力与门控设计在极端不对齐的模态间可能效果有限；④对极大规模文本/图像序列的计算成本仍较高；⑤未提供对极端 OOD 变体（如更大分辨率图像）的全面评估。

---

## 5. Active Learning for Efficient Annotation of Surgical Videos with Weak Supervision

**arXiv ID:** 2607.13237 | [PDF](https://arxiv.org/pdf/2607.13237v1)

**作者:** Manasa Dendukuri `[一作]` (University of Pennsylvania), Guiqiu Liao `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将弱监督与主动学习相结合的框架，通过迭代纠正模型生成的 CAM，显著减少外科视频像素级标注工作量，并在 Cholec80 数据集上实现与全手工标注相当的分割性能。

**💡 创新点**

创新点在于使用无初始像素标注的双重损失（视频级弱监督+稀疏掩码监督）以及基于 DINOv3 的时序一致性网络，形成“专家‑在‑循环”的主动学习流程，首次在外科视频分割中实现弱监督与主动学习的高效协同。

**🔧 技术方法**

采用 DINOv3 ViT‑B/16 自监督特征提取、3D‑CNN 时序一致性网络、CAM 映射、CRF 后处理、双重损失训练（弱监督+掩码监督）以及 Encord 在线标注平台进行专家纠正。

**📊 数据集**

使用 Cholec80 手术视频数据集（共 83 段，七种工具，训练 6,100 段，测试 81 段），每段 29 帧。

**📈 对比分析**

与先验全监督基线（Control‑WSL）对比，主动学习在 2、5、10 视频标注预算下均达到或超过基线的 CorLoc，且标注时间平均比纯手工低约 50%（例如 AL‑10 在 70 视频标注下 CorLoc 为 0.439，约比 Control‑WSL 高 0.06）。

**⚠️ 局限性**

局限包括对小批量主动学习时收敛不稳定、对 mask 损失权重的依赖、仅在外科工具分割任务上验证，缺乏对更大规模、多类别数据集和实时部署的评估。

---

## 6. Agora: Collective and Permissionless Internet-Scale Pretraining of Large Language Models

**arXiv ID:** 2607.13332 | [PDF](https://arxiv.org/pdf/2607.13332v1)

**作者:** Gil Avraham `[一作]` (Pluralis Research), Alexander Long `[通讯]` (Pluralis Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Agora的协议学习框架，用来在互联网上通过众多消费者级GPU进行大规模语言模型的去中心化预训练，支持节点弹性、异构和无中心化调度。

**💡 创新点**

创新点包括：① 用子空间网络（Subspace Networks）将跨阶段激活压缩两位数，降低跨节点通信量；② 引入异步稀疏参数平均（AsyncSPARTA）与确定性反向传播路由，消除同步梯度瓶颈并保持收敛；③ 将训练流程设计为协议而非单点程序，使节点可自由加入/离开并通过轻量级协作实现全局一致。

**🔧 技术方法**

核心技术包括：多阶段流水线并行 + 数据并行；子空间网络压缩；异步稀疏参数平均（AsyncSPARTA）；确定性反向传播路由；容错机制（节点自杀与恢复）与负载感知路由；实验验证使用的编译器/框架为 PyTorch 与 NCCL 的改进版本。

**📊 数据集**

使用公开可用的 FineWeb-Edu 数据集（约 500B tokens）训练 8.6B 参数模型 Pluralis-8B；在 1B 参数的 ablation 研究中同样使用 FineWeb-Edu。

**📈 对比分析**

与集中式 H100 基准对比：在 8.6B 模型的公开预训练中，Agora 达到约 170k tokens/s，效率为 63% 的集中式 H100 速度；在 1B ablation 中，系统对 10:1 计算异构性和 85% 以上的 all‑reduce 参与率均保持收敛；整体性能稳定并在节点 churn 情况下保持 4.2 tokens/TFLOP。

**⚠️ 局限性**

局限性包括：① 相比集中式集群仍然存在约 37% 的效率损失；② 对子空间压缩比例和异步平均阈值的选择敏感；③ 需要较高的网络可靠性与低 RTT；④ 在极端节点丢失或长期失效时仍需手动干预；⑤ 对极大规模（>10B 参数）训练的可扩展性尚未验证。

---

## 7. The Economics of AI Decoding Chips: Rebalancing Compute, Capacity, and Bandwidth for Efficient LLM Inference

**arXiv ID:** 2607.13068 | [PDF](https://arxiv.org/pdf/2607.13068v1)

**作者:** Michael J. Yuan `[一作]` (ByteFuture Inc.), Ju Long `[通讯]` (Texas State University)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5057358572)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了主流GPU在大语言模型（LLM）解码阶段的低效性，提出了以F/B（算力/带宽）和F/S（算力/容量）为核心的两常量模型，进而设计并评估了基于28 nm工艺、DDR5内存的Skymizer HTX‑301解码加速器。

**💡 创新点**

创新点在于：①通过F/B和F/S两常量定量描述解码效率并揭示Mixture‑of‑Experts（MoE）对并发需求的“MoE税”；②提出重新平衡芯片的三重策略——降低算力、提升可用内存、降低带宽，以实现低成本、供应链友好的解码加速器；③首次将该设计落地为HTX‑301，并与H100节点在成本与性能上做系统级对比。

**🔧 技术方法**

所采用技术包括：28 nm CMOS工艺、DDR5大容量内存、PCIe接口、FP16/BF16张量计算、稀疏专家路由与分布式内存池、以及常见的LLM推理软件栈。

**📊 数据集**

主要实验数据集为DeepSeek‑R1 671B（4‑bit、256个专家、top‑8），在此模型上测评解码吞吐与成本。

**📈 对比分析**

对比方法：在相同解码工作负载下，对HTX‑301单卡、4U、8×4U以及H100 8‑GPU节点进行token/sec与$/M tokens评估。结果显示：HTX‑301单卡≈$12/M tokens，8×4U（16用户）≈$12/M tokens，明显低于H100节点的$21/M tokens；同时实现了确定性20.3 token/s的单用户吞吐。

**⚠️ 局限性**

局限性：①低带宽导致单用户吞吐仅≈20 token/s，无法满足高实时性需求；②不适合超算规模的高并发部署（需96–320 GPU集群才能实现高利用率）；③缺乏成熟的GPU生态与软件支持，需自行搭建与维护。

---

## 8. Oracle Agent Memory as an Enterprise Memory Substrate for Long-Horizon AI Agents

**arXiv ID:** 2607.13157 | [PDF](https://arxiv.org/pdf/2607.13157v1)

**作者:** Richmond Alake `[一作]` (Oracle), Valentin Venzin `[通讯]` (Oracle)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Oracle Agent Memory，一种基于Oracle数据库的内存子系统，用于支持跨会话、跨用户和跨线程的长周期代理记忆管理。

**💡 创新点**

创新点包括：将内存生命周期（摄取、提取、整合、检索、摘要、修订）与数据库原生存储紧耦合；分层架构将活跃内存核心与被动存储接口分离；引入显式作用域控制（用户、代理、线程）和可插拔的检索与摘要策略；通过评估框架将记忆质量与下游准确率分离。

**🔧 技术方法**

技术上使用Oracle Database的关系表、JSON、向量索引和事务；LLM接口用于摘要和提取；向量模型做语义检索；HNSW索引实现高效向量检索；实现了线程生命周期、上下文卡、记忆写入等API。

**📊 数据集**

使用LongMemEval基准（500道多会话问答）以及LoCoMo、MemoryAgentBench、BEAM等评测数据集；通过对比平面历史基线评估记忆效果。

**📈 对比分析**

比较方法是：在LongMemEval上实现93.8%整体准确率（单会话记忆100%、跨会话推理88%），与平面历史基线相比，在同一会话长度下Token使用降低约10.7倍，且在80轮对话中有48/80轮记忆模式优于平面历史。BEAM上1M数据集准确率0.63-0.68，10M数据集0.51，展示了检索和推理的瓶颈。

**⚠️ 局限性**

局限性包括：评估依赖特定模型、嵌入、检索阈值和提示设计；未完全覆盖成本-质量Pareto前沿；作用域过滤是检索控制，非完整授权边界；缺乏图形化记忆、去重、冲突解决、事件顺序检索的成熟机制；BEAM 10M结果仅为早期压力测试。

---

## 9. GDM AI Control Roadmap

**arXiv ID:** 2607.13087 | [PDF](https://arxiv.org/pdf/2607.13087v1)

**作者:** Mary Phuong `[一作]` (Google DeepMind), Scott Coull `[通讯]` (Google DeepMind)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 GDM AI Control Roadmap（v0.1），系统性阐述针对内部 AI 代理的安全防御策略，包括威胁建模、控制不变量、能力梯度与可落地的多层防御措施。

**💡 创新点**

创新点在于：① 将 MITRE ATT&CK 框架迁移并扩展为针对 AI 代理的攻击技术（TTP）体系；② 将安全视角与对齐研究结合，定义两大控制不变量（检测与防御）并对应分层防御（D1–D4 与 R1–R3）；③ 通过“能力梯度”方法量化模型能力与防御成本的对应关系，为未来更强 AI 提供可扩展的防御路径。

**🔧 技术方法**

技术方法主要是：
- 监控技术：Chain‑of‑Thought 监控、行动监控、激活监控、Pull‑request 监控、UEBA 等；
- 防御技术：访问控制、隔离、供应链安全、主动监测、异步/同步响应、自动重采样、降级控制等；
- 评估方法：红队实验、误报/漏报测评、对齐测试、控制意识抑制、数据过滤与信息孤岛等。

**📊 数据集**

论文未使用单一公开数据集，而是基于假设场景与内部实验场景构建合成评估集，主要依赖内部代码库、模型日志、实验环境及人工标注的误判/攻击样本。

**📈 对比分析**

对比方法：通过分层（D1–D4、R1–R3）展示不同能力水平下的防御组合，评估指标包括监控覆盖率、召回率、响应时延、控制成本（实现、维护、计算、延迟、研究）等；文中提出在 D1/D2 时可实现高召回率且成本低，在 D3/D4 需要更高成本与更复杂的监控手段；在 R1–R3 中，R1 通过异步响应降低损失，R2 采用同步监控和访问控制显著提升阻止率，R3 引入自动关机与供应链硬化进一步降低风险。

**⚠️ 局限性**

局限性包括：
- 对齐与安全的假设场景可能与真实攻击差异；
- 评估方法对“真实”误判/攻击样本的依赖性强，难以覆盖所有攻击变体；
- 监控与响应的实现成本与运营复杂度高，特别是高阶 D4 与 R3 阶段；
- 对未知或零日攻击的防御仍有限，需持续演进与红队测试；
- 论文未给出量化性能数据，主要为策略性框架，实际效果需后续实测验证。

---

## 10. Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation

**arXiv ID:** 2607.13125 | [PDF](https://arxiv.org/pdf/2607.13125v1)

**作者:** Guoxuan Chen `[一作]`, Chenyang Lei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Boogu-Image‑0.1，一种统一的多模态生成模型，能够在单一网络中完成高质量文本生成、图像生成、风格迁移与文本排版。

**💡 创新点**

创新点在于通过可扩展的推理时间实现更高的文本‑图像生成质量，并在单模型中实现多视觉域的统一生成，最终在 Arena 基准上取得了开源模型中的最高成绩。

**🔧 技术方法**

采用 Transformer‑style 统一架构，结合可调节的推理时间缩放技术与高效的多模态预训练策略。

**📊 数据集**

使用涵盖摄影、油画、书法、印象派等多种视觉域的大规模图文对数据集（具体数据集未公开），并通过多语言（中英）文本对进行训练。

**📈 对比分析**

在 Arena 基准上与其他开源模型进行对比，Boogu-Image‑0.1 在所有指标上均优于竞品；在文本‑图像生成实验中，推理时间越长，生成质量越高，展示了可调节的质量‑速度折衷。

**⚠️ 局限性**

局限性包括：推理时间与生成质量高度相关，导致实际部署时需要显著的算力与延迟；模型在极少数专业领域（如医学影像）表现不足；缺乏对生成内容多样性与公平性的系统评估。

---

## 11. Audited Selective Verification for Risk-Controlled N-1 Thermal Contingency Screening under Deployment Shift

**arXiv ID:** 2607.13221 | [PDF](https://arxiv.org/pdf/2607.13221v1)

**作者:** Jayakumar Manoharan `[一作]` (Electric Power Research Institute), Jayakumar Manoharan `[通讯]` (Electric Power Research Institute)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5006232574)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对电力系统N‑1热安全问题，提出了“审计式选择性验证（ASV‑N1）”方法，用在线审计和全功率流校验为任何控制器输出提供分布无关的风险预算认证。

**💡 创新点**

创新点在于：① 将低成本线性敏感性估计仅用于决定要验证的故障；② 通过在线随机审计配合Learn‑Then‑Test校准阈值，实现对跳过故障的违约率的分布无关、高置信度上界；③ 设计了固定序列阈值测试与自适应审计尺寸，使得计算成本在任何部署分布下都可最小化。

**🔧 技术方法**

主要技术包括线性线性敏感性估计（LODF/OTDF）、全AC功率流求解、Learn‑Then‑Test风险控制框架、Clopper‑Pearson置信界、固定序列假设检验、以及自适应审计尺寸策略。

**📊 数据集**

实验数据集为公开的IEEE 118‑bus、IEEE 300‑bus和PEGASE 1354‑bus三大输电网络，采用pandapower求解器，对不同负荷/生成分布产生多种操作点进行验证。

**📈 对比分析**

与传统的确定性屏蔽（LOD‑F）、安全边际屏蔽、历史校准阈值屏蔽以及top‑k屏蔽相比，ASV‑N1在单窗口下的AC求解量可降低29%–75%（批量时可达36%–80%），且在所有系统下都能保持实际违约率在预算内；静态阈值和确定性屏蔽在负载/控制器引起的分布偏移下往往失效。

**⚠️ 局限性**

局限性包括：① 仅提供跳过故障的违约率上界，而非逐个故障的绝对安全保证；② 只针对热安全（未覆盖电压、无功等约束）；③ 审计本身是一项不可避免的计算成本，且在高安全预算或严重偏移时成本会逼近全检；④ 对于恶劣系统，若线性敏感性估计极差，仍需大幅验证，导致收益下降。

---

## 12. Hierarchical $\mathcal{F}$-Clustering: Approximation and Hardness of Clustering into Trees and Bounded Diameter Graphs

**arXiv ID:** 2607.13217 | [PDF](https://arxiv.org/pdf/2607.13217v1)

**作者:** Michał Szyfelbein `[一作]` (Gdańsk University of Technology), Dariusz Dereniowski `[通讯]` (Gdańsk University of Technology)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5048684673)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种新的层次聚类问题——Hierarchical F‑Clustering，允许停止条件为图属于给定类 F（树或定直径图）。

**💡 创新点**

创新点在于提出了一个统一的线性规划框架，利用“well‑behaved”图类定义，得到 log n·loglog n（树）和 log n（定直径）近似算法，并给出在 SSEH 下的常数因子不可逼近结果。

**🔧 技术方法**

技术上使用了层次化线性规划、spreading metrics、ILP 约束、LP 取整与分层递归、ρ‑SEP 的双重逼近等。

**📊 数据集**

论文为理论工作，没有使用具体实验数据集。

**📈 对比分析**

通过与 LP 下界及已知的 FES/Min‑Multicut 近似结果对比，证明了所得到的近似比率与当前最优近似一致，且比先前的 √(log n) 近似更好。

**⚠️ 局限性**

限制在于近似因子受 ρ‑SEP 和 FES/Min‑Multicut 的 log n 因子限制，若想突破需改进这些基础问题；在 SSEH 假设下常数因子不可逼近。

---

## 13. Graph Partitioning with Demands: Generalized Conductance and its Applications

**arXiv ID:** 2607.13218 | [PDF](https://arxiv.org/pdf/2607.13218v1)

**作者:** Michał Szyfelbein `[一作]` (Gdańsk University of Technology), Dariusz Dereniowski `[通讯]` (Gdańsk University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个新的图划分目标——Generalized Conductance，并给出了其定义和优化形式。

**💡 创新点**

创新点在于设计了两阶段归约策略，将Generalized Conductance 近似化为 Generalized k‑Multicut 与带约束的 Sparsest Cut，并通过 Räcke 的树分解实现了对一般图的 O(log n) 近似；对乘法需求函数可进一步提升至 O(√log n)，在树图上实现 O(1) 近似。

**🔧 技术方法**

主要技术包括 Räcke 的 cut‑sparsifier 树分解、基于多源多汇流的最小割上界、对树图上稀疏割的精确求解，以及 Max‑Cut 的贪心近似算法。

**📊 数据集**

论文未使用具体实验数据集，所有结果均为理论证明和近似比的上界。

**📈 对比分析**

与传统的 Sparsest Cut、Graph Partitioning 以及 Hierarchical Clustering 的需求版本相比，本文在一般图上实现了 O(log n) 的近似，比现有的 O(√log n·loglog n) 更好；乘法需求下可进一步到 O(√log n)，树图上可达到 1‑近似。

**⚠️ 局限性**

局限性包括：算法对需求函数的形式有限制（非乘法需求时仍保持 O(log n)），计算复杂度相对较高；对超图或更一般的需求模型的扩展尚未给出；以及在实际应用中的性能验证缺失。

---

## 14. ChipVerilog: A Large-Scale OpenCores-Derived Benchmark for LLM-Based Verilog RTL Generation

**arXiv ID:** 2607.13079 | [PDF](https://arxiv.org/pdf/2607.13079v1)

**作者:** Yan Tan `[一作]` (Hong Kong University of Science and Technology), Yangdi Lyu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5014819244)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了ChipVerilog基准，用于评估大规模IP/核心级Verilog RTL生成；

**💡 创新点**

创新点在于从OpenCores IP构建真实、跨层次、超过1000行的RTL任务，填补了现有短模块基准的空白；

**🔧 技术方法**

技术包括自然语言描述构造、LLM生成RTL、Icarus Verilog进行语法/编译检查、Yosys等价检查以及仿真验证；

**📊 数据集**

使用了64个生成目标，覆盖FPU、OR1200、MIPS-16、I2C和CORDIC五个设计族，约20,400行Verilog；

**📈 对比分析**

对比方法采用pass@k指标，对Claude Opus 4.5、GPT-5.4和DeepSeek V4 Pro进行评测，语法通过率高（>90%），但功能通过率仅在10-40%之间，长模块和多层级目标表现最差；

**⚠️ 局限性**

局限性在于LLM难以处理长上下文、跨模块交互、精细时序和接口约束，导致功能正确性低，缺乏针对性错误诊断与修复机制。

---

## 15. Securing LLMs in the Wild: Privacy and Security Challenges at the Edge

**arXiv ID:** 2607.13088 | [PDF](https://arxiv.org/pdf/2607.13088v1)

**作者:** Ren-Yi Huang `[一作]` (University of South Florida), Morris Chang `[通讯]` (University of South Florida)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5055158787)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了将大型语言模型（LLM）部署到边缘设备时产生的安全‑效率矛盾，并提出了三壁约束模型与安全运营效率得分（SOES）用于评估与选择边缘LLM。

**💡 创新点**

创新点包括：① 将内存壁、二次壁、计算壁三大硬件瓶颈统一量化并给出可执行的安全阈值；② 设计SOES度量，将任务精度、抗 jailbreak、隐私、能耗、内存占用和延迟三维综合评估；③ 针对每类优化技术（量化、剪枝、分区、PEFT）提出对应的安全缓解策略，形成完整的安全-效率评估与决策框架。

**🔧 技术方法**

技术手段：量化（INT4/FP16）、剪枝、模型分区、参数高效适配（LoRA等）、安全评测（jailbreak、membership inference、隐私泄露）、能耗/内存/延迟监测、三壁约束公式与SOES公式。

**📊 数据集**

使用的数据集与评测集合包括：MMLU（任务准确度）、10个定制 jailbreak prompt（安全评估）、30个 canary 语句（隐私泄露检测）、以及在真实边缘硬件上测得的能耗、VRAM 和推理延迟。

**📈 对比分析**

方法：对 Phi‑3.5‑mini、Qwen2.5‑1.5B、Gemma‑2‑2B、Gemma‑4‑E2B、Llama‑3.2‑3B‑INT4、Granite‑4.1‑3B 等多种边缘友好模型在 FP 与 INT4 两种精度下，分别记录 MMLU、jailbreak 率、隐私得分、能耗、内存与延迟，并依据 SOES 进行综合排序。实验结果显示，FP 方案下 Qwen2.5‑1.5B 获得最高 SOES；INT4 方案下 Llama‑3.2 与 Granite‑4.1‑3B 维持较高安全性，Phi‑3.5‑mini 在效率上表现最佳但安全性不足。

**⚠️ 局限性**

局限性：仅评估了 1–3B 参数规模的单模态 LLM；安全评测依赖固定 prompt 与 canary 集合，可能不覆盖所有攻击场景；量化实现差异导致能耗与延迟不完全可比；未考虑多模态或更大规模模型的安全‑效率平衡；实际部署仍需针对具体硬件环境进行进一步验证。

---

## 16. Where Does the Noise Come From? A Variance-Components Decomposition of Non-Determinism in LLM Brand Answers

**arXiv ID:** 2607.13304 | [PDF](https://arxiv.org/pdf/2607.13304v1)

**作者:** Dmitrij Żatuchin `[一作]` `[通讯]` (Estonian Entrepreneurship University of Applied Sciences), Dmitrij Żatuchin (Estonian Entrepreneurship University of Applied Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在品牌答案生成中的随机性进行方差分量分析，并给出预算分配规则。

**💡 创新点**

首次将 LLM 非确定性拆分为语言、模型、提示词、品牌和内部采样四个可量化来源，并使用广度优于深度的分配策略。

**🔧 技术方法**

使用广度理论（Generalizability Theory）与限制性最大似然（REML）拟合线性混合模型，计算方差成分、ICC 与可靠性。

**📊 数据集**

基于 20 家中东欧品牌、8 种语言、3 个 LLM（GPT‑5.2、Gemini 3 Flash、Perplexity）、15 条提示、共 12,933 条响应的完整交叉语料库。

**📈 对比分析**

通过方差分量与可靠性边界（Eρ²）比较不同设计的精度：单回答可靠度≈0.01，8 语言+3 模型+15 提示+10 复测可达≈0.36；同样投入下语言/模型扩展比重复提升效益显著。

**⚠️ 局限性**

局限包括：未给出置信区间、情感极度零化导致品牌方差低、温度低限估计重复噪声、模型与检索模式混合、层级数有限、仅一语料与时间窗口，需进一步验证。

---

## 17. Full-Pipeline Inference Optimization for MiMo-V2.5 Series: Pushing Hybrid SWA Efficiency to the Limit

**arXiv ID:** 2607.13095 | [PDF](https://arxiv.org/pdf/2607.13095v1)

**作者:** Xiaomi MiMo Team `[一作]`, Zihan Jiang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对 MiMo‑V2.5 系列模型，作者构建了完整的推理流水线优化体系，涵盖 KVCache 双池管理、SWA‑Aware 前缀树、GCache 分布式缓存、动态调度、Prefill/Decode 并行与多模态推理加速。

**💡 创新点**

创新点在于：① 将 KVCache 划分为全注意力池与滑动窗口池，实现 7× 存储压缩；② 引入窗口安全匹配规则和双索引前缀树，保证 SWA 与全注意力的 KV 一致性；③ 通过 GCache 的多级、去中心化设计与 RDMA 优化，显著提升跨节点缓存命中率与网络吞吐；④ 在调度层面引入 KV‑Affinity、TTFT 优先级与负载均衡，减少等待时间；⑤ 对 Prefill/Decode、MTP、Multimodal Encoder 进行多层次并行与共享缓存优化。

**🔧 技术方法**

使用技术包括：GPU‑级 KVCache 双池、层级预取、窗口安全前缀树、分布式一致性哈希缓存（GCache）、Redis‑后端无状态调度、CUDA Graph 优化、MTP 与多模态 Encoder 的跨请求批处理与 GPU 预处理、RDMA 与 NUMA 绑定网络优化。

**📊 数据集**

实验基于 Xiaomi 内部 MiMo‑V2.5 预训练和推理数据集，覆盖多模态（图像、音频、视频）和长序列（高达 1M token）场景；未公开使用第三方公开数据集。

**📈 对比分析**

与传统 Full‑Attention、单 KVCache、无调度优化的基线对比，KVCache 存储压缩 7×、Prefill 计算减少 7×、Decode KV 5× 增强、TTFT P90 缩短 30% 以上、QPS 从 15 增至 30，整体吞吐提升约 40% 以上，成本显著下降。

**⚠️ 局限性**

局限性包括：SWA 对 KV 生命周期的严格同步要求导致实现复杂；长序列长度分桶与负载均衡仍需改进；多模态 Encoder 的并行化和缓存共享依赖硬件特性；在单节点失效时仍缺乏多副本冗余，可能导致瞬时中断。

---

## 18. The Perplexity Trap: When Patent Law Makes Human Writing Look Like AI

**arXiv ID:** 2607.13044 | [PDF](https://arxiv.org/pdf/2607.13044v1)

**作者:** Anubhab Banerjee `[一作]` `[通讯]` (Nokia Solutions and Networks GmbH & Co. KG), Anubhab Banerjee (Nokia Solutions and Networks GmbH & Co. KG)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估专利文本中AI生成内容的检测，证明现有基于似然的检测在受Article 84约束的专利文书中失效，并提出仅使用语言复杂度特征的离轴检测方案。

**💡 创新点**

发现并系统阐述了“Perplexity Trap”现象，证明所有基于似然的检测在外部语法约束下无效，并设计了高效的离轴复杂度分类器。

**🔧 技术方法**

使用DetectGPT、Fast‑DetectGPT、Binoculars、DivScore等零样本检测器；采用小型LLM（GPT‑2‑medium）做评分头；构建基于逻辑回归的七维语言复杂度特征分类器；在不同IPC类和LLM生成策略下进行鲁棒性验证。

**📊 数据集**

基于1000份EPO电信专利（500人类、500AI）构建平衡语料；AI文本通过五种提示策略生成；进一步扩展到A61K、C07D、F03D等IPC类进行跨域验证。

**📈 对比分析**

在消费者级GPU（≤8 GB VRAM）下比较，传统检测器FPR>60%、准确率≤47%；离轴复杂度分类器在5折CV下取得74%准确率、FPR≈28%，比基线提升约13个百分点。

**⚠️ 局限性**

研究仅限于EPO专利，未验证其他受约束文本；使用的LLM和评分头规模有限，离轴特征可能易被对抗攻击；整体准确率仍未达到实务可接受水平。

---

## 19. HRO: Hierarchical Room-to-Object Framework for Zero-Shot Object Goal Navigation with Large Language Models

**arXiv ID:** 2607.13072 | [PDF](https://arxiv.org/pdf/2607.13072v1)

**作者:** Luyuan Jia `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 4190 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于大语言模型的分层房间→对象的零样本目标导航框架 HRO，模仿人类先判断房间类型再寻找目标的思路；

**💡 创新点**

创新点在于利用房间类型作为中间语义桥梁，形成粗到细的层级推理（房间推断→前沿评估→路径规划），显著提升了语义关联精度与探索效率；

**🔧 技术方法**

使用 GPT‑2 作为 LLM 进行房间类型推断与语义亲和评分，结合 Mask‑R‑CNN 语义分割、Fast‑Marching 路径规划与前沿筛选；

**📊 数据集**

在 HM3D 与 Gibson 两个真实 3D 环境数据集上进行实验，分别评估 2000/1000 个任务实例；

**📈 对比分析**

与 L3MVN、SemUtil、VoroNav、ESC、SemExp、PONI 等基线相比，HRO 在 HM3D 上取得 53.0% SR、24.5% SPL，Gibson 上 84.0% SR、46.5% SPL，均高于同类零样本方法；

**⚠️ 局限性**

局限主要在于上游语义分割精度不足导致地图构建误差，且仍依赖较小的 GPT‑2 模型，未来需改进分割与更强 LLM 的协同机制。

---

## 20. TSSM: Triaxial State Space Model for Global Station Weather Forecasting with Temporal-Variable-Historical Modeling

**arXiv ID:** 2607.13101 | [PDF](https://arxiv.org/pdf/2607.13101v1)

**作者:** Songru Yang `[一作]` (Beihang University), Zhengxia Zou `[通讯]` (Beihang University)

**通讯引用:** 9623 | [OpenAlex ID](https://openalex.org/A5088611151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种三轴状态空间模型（TSSM），通过周期对齐的历史维度将全球站点天气数据重新组织为时间-变量-历史结构，并在此基础上实现因果式预测；

**💡 创新点**

创新点在于将历史信息作为独立维度嵌入模型，利用历史扫描（H-Scan）捕捉长期演化与极端事件特征，并通过三轴扫描（T-Scan、V-Scan、H-Scan）及层次共享架构实现短期与长期依赖的协同建模；

**🔧 技术方法**

核心技术包括状态空间模型（SSM）在时间、变量、历史轴上的一次性扫描、异常增强、频域与变量增强融合，以及因果式预测框架；

**📊 数据集**

使用Weather‑5K（5k+气象站的10年小时级温度、露点、风向/风速、气压）作为主数据集，并在水文数据集及五个通用时间序列数据集上进一步验证；

**📈 对比分析**

与多种基准（LSTM、Transformer、Mamba、WSSM、Corrformer等）对比，TSSM在平均准确率上提升约10%，极端事件捕获提升61%，在24–240小时预测中实现37.5%提升，并在迭代长时域下实现103.5%提升；在缺失观测（80%）下保持>90%性能；

**⚠️ 局限性**

限制在于历史信息引入的准确率与极端事件捕获之间存在权衡，当前融合方式难以完全消除；进一步改进需探索更高效的历史‑时间融合策略。

---

## 21. From visibility to vulnerability: how women scientists face gendered hostility in science communication

**arXiv ID:** 2607.13326 | [PDF](https://arxiv.org/pdf/2607.13326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 22. SemaDiff: Identifying Semantic-Changing Commits with Generated Code and Tests

**arXiv ID:** 2607.13111 | [PDF](https://arxiv.org/pdf/2607.13111v1)

**作者:** Maha Ayub `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 SemaDiff，利用大语言模型生成依赖类和相应的单元测试，并在提交前后执行相同测试以动态判断提交是否导致程序语义变化；

**💡 创新点**

创新点在于：①通过LLM自动生成只调用被修改代码的依赖类，解决传统测试无法编译或覆盖率不足的问题；②以相同测试对比两版本行为，实现高精度（100%）的语义变化检测；

**🔧 技术方法**

技术手段包括：Java AST 与调用图分析（Spoon库），大语言模型（ChatGPT、CodeLlama、DeepSeek）用于代码与测试生成，自动构建与执行，基准工具包括 RefactoringMiner、PurityChecker 与 Randoop；

**📊 数据集**

使用手工标注的183条最近的 Java 开源项目（7 个项目）中的 refactoring 提交，其中 95 条语义不变，88 条语义变更；

**📈 对比分析**

与三种 LLM、PurityChecker 以及 Randoop 进行对比。SemaDiff 在 ChatGPT 版本下达成约 76% 的准确率、100% 的精确度、约 59% 的召回率；LLM 单独判断准确率 78-87% 但精度低；PurityChecker 仅 20% 准确率；Randoop 的准确率约 22%。SemaDiff 的测试覆盖率平均 74%，显著高于 Randoop；

**⚠️ 局限性**

局限性包括：①依赖 LLM 生成质量，导致约 3-14% 的执行失败；②目前仅针对 Java，未验证跨语言或不同项目的迁移性；③生成的测试可能无法覆盖所有边界情况；④需要手工标注的工作量；⑤数据集规模相对有限，难以覆盖所有类型的 refactoring 变更。

---

## 23. CayleyR: Solving the TopSpin puzzle via cycle intersection

**arXiv ID:** 2607.13219 | [PDF](https://arxiv.org/pdf/2607.13219v1)

**作者:** Yuri Baramykov `[一作]` `[通讯]`, Yuri Baramykov

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个R包cayleyR，用迭代循环交叉（ICI）算法在Cayley图中寻找TopSpin(n,k)拼图的路径。

**💡 创新点**

创新点在于不按层级搜索，而是从起点和终点随机生成操作序列，展开其循环，寻找交叉点；同时结合桥接选择和可选GPU加速，能够在较大状态空间（n≤20）下快速求解。

**🔧 技术方法**

技术包括：C++/Rcpp后端实现循环展开与哈希索引存储，OpenMP并行随机序列评估，Vulkan GPU批量状态变换与距离计算，距离启发式桥接，路径重建与简化。

**📊 数据集**

数据集：TopSpin(14,4)、TopSpin(10–16,4)、TopSpin(20,4)等随机生成的目标状态，使用不同扰动距离（20–150步）测试。

**📈 对比分析**

与传统双向BFS、模式数据库等方法比较，ICI在n≤10时能达到最优，但在n>10时BFS不可行；实验显示ICI在约1–3秒内成功求解12个实例，路径长度约2k–3k，优于随机搜索但不保证最短。

**⚠️ 局限性**

局限性：不保证最短路径；对极难实例时可能失败；需要手动调参（序列长度、样本量、排名准则）；对非TopSpin生成器的通用性尚未验证。

---

## 24. Adaptive Filtering of the KV Cache: Diagnosing and Correcting Structural-Role Bias in LLM Inference

**arXiv ID:** 2607.13205 | [PDF](https://arxiv.org/pdf/2607.13205v1)

**作者:** Soumil Mandal `[一作]` `[通讯]` (ServiceNow), Soumil Mandal (ServiceNow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对结构化、深层嵌套的长上下文，本文提出一种基于结构角色的 KV 缓存淘汰策略，在保持训练无关的前提下显著提升稀疏键值对的精确匹配准确率。

**💡 创新点**

创新点包括：① 通过对 H2O 关键字过度保留的诊断发现结构性 “sink” 现象；② 采用 SnapKV 的窗口化最大池化与按角色分配相结合的“角色条件分配器”，在单一超参数下实现对 KEY、DELIM、VALUE 等角色的动态预算重分配；③ 用 15 MB 的线性角色探针代替传统结构解析器，保持了大部分性能。

**🔧 技术方法**

核心技术包括：窗口化注意力累积（SnapKV）、一维最大池化、按角色分桶的 top‑k 选择、角色权重调度（V 0.70、D 0.20、P 0.05、W 0.05、KEY 可调），以及对 KV 句柄的掩码编码。

**📊 数据集**

实验使用四类合成数据集：带缩进的嵌套 JSON、无空格 JSON、与 XML 内容相同的文本、带干扰项的 Markdown 表格；在 Llama‑3.1‑8B、Mistral、Phi‑3、Qwen 等四大模型上进行验证。

**📈 对比分析**

与 H2O、SnapKV、Fair‑Eviction、Quest 等基线相比，所提方法在 5% 预算下可将 EM 从 0% 提升至 63%–98%（对 Llama‑3.1‑8B 在 5% 预算下达到 0.98 的 EM），表现出三倍以上的超加性收益；在高预算（≥30%）时可与全缓存或略优，表明在稀疏键值场景下具备较高实用性。

**⚠️ 局限性**

局限性包括：对低角色密度文本（如稀疏表格）收益有限；在多种随机种子下“超过全缓存”的优势不稳定；当前实现仅通过掩码方式评估准确率，未实际实现 KV 压缩或显著的内存/延迟提升；探针的局部性导致对块级信息缺失，需进一步改进。

---

## 25. EMAGN: Efficient Multi-Attention Graph Network via Learned Clustering for Scalable Traffic Forecasting

**arXiv ID:** 2607.13241 | [PDF](https://arxiv.org/pdf/2607.13241v1)

**作者:** Mingxing Xu `[一作]` (Shanghai Jiao Tong University), Oliver Gao `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了EMAGN模型，将传统O(N²)自注意力替换为基于学习聚类的O(NM)线性复杂度自注意力，应用于交通流量预测。

**💡 创新点**

创新点在于通过学习适应交通网络的聚类矩阵C_k和C_v，实现全注意力的灵活性与线性计算效率的统一，并显著扩展模型配置空间。

**🔧 技术方法**

使用了高维高斯滤波理论导出的线性自注意力、学习聚类矩阵、多头注意力、STBlock并行融合空间/时间注意力、对齐模块等技术。

**📊 数据集**

实验基于PEMS-BAY和METR-LA两个交通速度数据集。

**📈 对比分析**

与多种基线（STGCN、DCRNN、GMAN、STAEformer、MLCAFormer等）比较，EMAGN在MAE上仅比GMAN低2.7–3.2%，但训练时间缩短32%，推理时间缩短38%，显存减少58%，且能够在16头等大配置下运行。

**⚠️ 局限性**

局限性包括：时间维度仍采用二次复杂的注意力；聚类数M需手工调参；在更大规模网络或更长历史序列时需进一步验证性能。

---

## 26. Phantom Guardrails: When Self-Improving Agent Harnesses Fix Failures That Never Happened

**arXiv ID:** 2607.13083 | [PDF](https://arxiv.org/pdf/2607.13083v1)

**作者:** Su Wang `[一作]` (Carnegie Mellon University), Haoran Yu `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一个确定性微实验室——Counterfactual Fabrication Lab，用来检测并量化自我改进型AI代理在没有真实失败的情况下生成虚假安全门（phantom guard）的现象。

**💡 创新点**

创新点在于引入了一个零审计、oracle‑检查的失效检测指标，并系统揭示了导致虚假安全门生成的三大条件（规则形状、规则集不完整、失败预设），同时验证了三种对策（指令卫生、规则集完整声明、warrant‑aware 接受）可将该失效率降至接近零。

**🔧 技术方法**

技术手段包括：利用大型语言模型（LLM）作为自我提升的提议者；构造一个仅包含合法动作的 MiniArena 游戏环境；使用字节精确的规则判别 oracle；以及实现一个只增不删的接受循环来模拟维护过程。

**📊 数据集**

使用的数据集为 MiniArena 的三种池子：① 对齐池（含注入的非法大王移动，真实失败存在）；② 制造池（全部合法但含重复正方形的重复模式）；③ 纯净池（全部合法且无可识别模式）。

**📈 对比分析**

比较方法：在三种接受规则（只基于抑制、严格提升、warrant‑aware）下分别测量虚假安全门的启用率；结果显示，在缺失规则集完整声明和失败预设时，虚假门的启用率可达约12%，而通过任何一种对策后均降至0%（或统计显著低于0.001 的p值）。

**⚠️ 局限性**

局限性包括：实验仅在单一基于棋盘游戏的极简环境中验证；缺乏对更复杂真实世界任务（如工具使用或安全防护）的跨域验证；以及对指令语言的细微差异和模型能力层级的依赖未做深入分析。

---

## 27. Meta-Learning Preferences for Multilingual LLM Alignment

**arXiv ID:** 2607.13315 | [PDF](https://arxiv.org/pdf/2607.13315v1)

**作者:** Jiaying Lin `[一作]` (University of Warwick), Debmalya Mandal `[通讯]` (University of Warwick)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于元学习的多语言LLM对齐框架，利用MAML结合RLHF和DPO实现低资源语言的快速适配。

**💡 创新点**

创新点：将多语言偏好对齐视为元学习任务，首次给出RLHF和DPO的MAML收敛理论，并通过跨语言偏好数据显著提升样本效率。

**🔧 技术方法**

技术：MAML (Model‑Agnostic Meta‑Learning)、RLHF、DPO、Bradley–Terry 对数似然损失、线性奖励/对数线性策略、LoRA微调等。

**📊 数据集**

数据集：Okapi 多语言偏好数据（26+ 40k 对）和 HuggingFace 公共偏好数据（7 语种 3.4k 对）。

**📈 对比分析**

比较方法：与单任务RLHF/DPO、多任务预训练、翻译‑测试 baseline 比较；在仅 100 目标语言样本的极低资源场景下，MAML‑DPO/​RLHF 的 win‑rate 提升约 28%（最高 40%），在 4k–40k 样本下仍保持优势，跨语言距离不敏感。

**⚠️ 局限性**

局限：仅在偏好学习阶段验证，未扩展到完整 RL 训练；对极低资源仍需至少 100 样本；理论假设（线性奖励、均匀覆盖）与实际环境可能不完全匹配。

---

## 28. Attitude Estimation Using Inertial and Barometric Measurements

**arXiv ID:** 2607.13254 | [PDF](https://arxiv.org/pdf/2607.13254v1)

**作者:** Melone Nyoba Tchonkeu `[一作]`, Tarek Hamel `[通讯]` (University Cote d'Azur)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用气压计高度测量进行姿态估计的两种观测器架构，分别实现了近似全局渐近稳定（AGAS）和局部指数收敛（LES），并在仿真与实飞行实验中验证其性能。

**💡 创新点**

创新点包括：①将气压计高度作为补充观测量，在缺失或不可靠速度传感器的环境下实现姿态估计；②设计了基于确定性Riccati观测器与SO(3)非线性观测器级联的AGAS架构；③提出了统一的SO(3)×ℝ²观测器，利用Riccati增益实现局部指数收敛，并推导出更弱的可观测性条件；④给出了完整的可观测性、稳定性分析与离散实现。

**🔧 技术方法**

使用的技术包括：确定性Riccati观测器、SO(3)非线性观测器、连续与离散时间实现、欧氏投影与矩阵指数映射、统一可观测性分析、误差动力学级联与输入-状态稳定性理论。

**📊 数据集**

数据集：①仿真数据，采用已知角速度、加速度与高度模型的合成飞行轨迹；②实飞行数据，来自MakeFlyEasy Fighter VTOL UAV的Pixhawk记录（IMU、磁力计、气压计），并使用内部EKF姿态作为参考。

**📈 对比分析**

比较方法：使用收敛时间、最大姿态误差、稳态误差、RMSE（滚转、俯仰、偏航）等指标；实验结果表明：LES在初始误差较大时收敛速度更快，临时误差更小；AGAS收敛更慢但具有更广的收敛域，稳态误差略低；两者在足够激励下可达到相近精度。

**⚠️ 局限性**

限制：①LES仅在满足水平加速度持久激励且误差在局部可观测域内时保证指数收敛；②AGAS收敛速度相对较慢；③两种方法均假设IMU无偏置、磁场已知且无干扰；④实验中未考虑气压计温漂、外部干扰及多路径误差。

---

## 29. Graph-Series Semantics and Abel Regularization for Recursive Hybrid Quantum Programs

**arXiv ID:** 2607.13117 | [PDF](https://arxiv.org/pdf/2607.13117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 30. Active Beyond-Diagonal RIS Empowered Heterogeneous Edge Computing: A Distributional Reinforcement Learning Approach

**arXiv ID:** 2607.13160 | [PDF](https://arxiv.org/pdf/2607.13160v1)

**作者:** Tianyu Pang `[一作]` (Hong Kong University of Science and Technology), Hongyu Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 47528 | [OpenAlex ID](https://openalex.org/A5057492548)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了基于物理一致的主动BD‑RIS辅助异构移动边缘计算系统的能耗感知任务卸载与资源分配问题，并提出将其建模为混合整数非凸优化后用分布式软演员-评论家（DSAC‑T）算法求解。

**💡 创新点**

创新点包括①采用物理一致的主动BD‑RIS模型捕获跨扇区泄漏与放大噪声；②在同一框架内同时考虑CPU/GPU异构任务、深度阻塞链路与残留直连；③利用分布式软演员-评论家（DSAC‑T）处理奖励异质性与可行性边界，提高策略稳定性。

**🔧 技术方法**

采用分布式软演员-评论家算法（DSAC‑T）、强化学习、MMSE接收、分布式价值学习等技术。

**📊 数据集**

使用仿真生成的300个评估场景（K=10用户、N=8天线、M=128单元等），未使用公开数据集。

**📈 对比分析**

与DDPG、SAC、TD3、AO–SCA等方法比较，DSAC‑T在能耗-延迟奖励上取得最高值-2.828，81.67%可行率，在线决策时间仅0.0267s，显著优于其他方法。

**⚠️ 局限性**

主要局限在于仅在仿真环境验证，未考虑多基站或多RIS部署；模型假设对称阻塞与电路可逆性，实际实现可能受限；强化学习需要大量训练样本，且迁移到真实系统需进一步研究。

---

## 31. Self-Supervised Visual Representation Learning: Pretrain-Finetuning or Joint Training?

**arXiv ID:** 2607.13192 | [PDF](https://arxiv.org/pdf/2607.13192v1)

**作者:** Nusrat Munia `[一作]` (University of Kentucky), Abdullah-Al-Zubaer Imran `[通讯]` (University of Kentucky)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5101789164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统地探讨了自监督学习（SSL）和监督学习目标在训练过程中的联合优化，比较了传统的预训练-微调（PFT）和联合训练（JT）两种训练范式。

**💡 创新点**

创新点在于通过联合训练（JT）同时优化自监督和监督损失，提供了一种更有效的替代方案，尤其在低标签数据环境下表现出更好的数据和训练效率。

**🔧 技术方法**

使用了多种自监督学习方法，包括对比学习、非对比学习和生成式学习，结合了不同的网络架构进行实验。

**📊 数据集**

使用了多个公开数据集，包括CIFAR-10、COCO、ISIC、JSRT、EarthScape等，涵盖自然图像、医学成像、危机响应和遥感等多种领域。

**📈 对比分析**

通过在不同标签比例下评估PFT和JT的性能，结果表明JT在低标签设置下表现更好，且训练效率更高，而PFT在特定领域中更为可靠。

**⚠️ 局限性**

限制在于不同SSL方法和任务的特性可能影响两种训练范式的相对有效性，且在某些复杂或噪声较大的领域，PFT仍然是更可靠的选择。

---

## 32. FixItFlow: Automated Troubleshooting Guide Generation from Cloud Incidents

**arXiv ID:** 2607.13035 | [PDF](https://arxiv.org/pdf/2607.13035v1)

**作者:** Srihari Unnikrishnan `[一作]` (Microsoft Research), Supriyo Ghosh `[通讯]` (Inception)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用历史云事件数据和大型语言模型自动生成排障指南（TSG），从数据提取、预处理、语义分析到结构化指南合成，形成完整的三阶段排障文档。

**💡 创新点**

创新点：①零幻觉验证协议——所有命令/查询必须原样出现于工程师评论，完全防止生成虚假信息；②基于模式的 schema 约束与多层校验机制，确保指南完整、可执行且与实际工程操作一致；③增量式数据采集与三阶段清洗流程，显著提升处理效率与数据质量；④层级化提示工程与结构化输出模板，使 LLM 能在高精度下完成复杂技术文档生成。

**🔧 技术方法**

技术手段：大型语言模型（GPT‑4o 风格）+ 结构化提示工程 + 零幻觉验证 + 语义聚类 + 令牌管理 + 并行/速率控制 + 代码/命令直接复制与验证，最终输出结构化的 Symptom、Diagnosis、Mitigation 三段式排障指南。

**📊 数据集**

数据集：微软内部云平台的历史事故记录（包含严重级别、时序、工程师讨论、缓解摘要等）——共计数千起已解决事件；评估数据来自 26 名现场工程师的问卷与访谈。

**📈 对比分析**

对比方法：与未加入任何提示或校验的 GPT‑4o 基线进行对比；使用问卷 Likert 评分、Utility Index、Top‑2 Box 百分比等指标评估指南质量；实际效果上，关联指南的事故平均缓解时间缩短 2.3 倍，清晰度得分 61.5% 以上，逻辑/事实准确度 42.3%。

**⚠️ 局限性**

局限性：①采用率仍低，部分指南缺少完整缓解步骤；②需手动对齐团队模板与内部规范；③仅在微软内部数据上验证，缺乏公开数据集支持可复现性；④对极端新型故障的泛化能力待进一步提升。

---

## 33. ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation

**arXiv ID:** 2607.13124 | [PDF](https://arxiv.org/pdf/2607.13124v1)

**作者:** Qingyu Zhang `[一作]` (Chinese Academy of Sciences), Xiuyin Zhao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在结构化剪枝后对LLM进行恢复，利用剪枝前的完整模型进行基于采样的自监督蒸馏，使被剪枝模型恢复生成能力。

**💡 创新点**

提出短-长动态预算控制的自蒸馏方法：通过检测重复尾部并动态调整前缀长度，解决传统固定预算导致的低信息重复前缀浪费问题。

**🔧 技术方法**

技术包括：BI深度剪枝、基于Jensen–Shannon距离的密集on‑policy自蒸馏、EMA预算控制、重复检测与截断管理。

**📊 数据集**

使用Qwen3‑4B‑Instruct 25% 参数剪枝后的模型，并采集45,447个多域提示（数学、代码、开放式），评估数据集包括 GSM8K、MATH‑500、HumanEval、MBPP、Alpaca、QA、Summarization、MT‑Bench。

**📈 对比分析**

与SFT、SeqKD、KD、RLVR等对比，恢复模型在8类任务的平均分从未剪枝教师的75.17降至5.71（未恢复），通过本方法提升至48.46（相当于教师的64.5%），多轮训练可进一步提升至55.41（73.7%）。

**⚠️ 局限性**

仅在BI深度剪枝、25% 参数裁剪的Qwen3‑4B 上验证，未检验其他剪枝方式、模型规模或更高压缩比例，也未确定压缩过度导致无法恢复的阈值。

---

## 34. Not Your Usual Type(s): Data contracts as types across languages and engines

**arXiv ID:** 2607.13339 | [PDF](https://arxiv.org/pdf/2607.13339v1)

**作者:** Aldrin Montana `[一作]` (Bauplan Labs), Jacopo Tagliabue `[通讯]` (Bauplan Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Bauplan SDK 2.0，该 SDK 在多语言数据管道中引入了类型化的数据契约（schema、约束、文档、线索），并在本地、控制平面、运行时三个阶段执行校验，显著减少节点间的 Schema 错误。

**💡 创新点**

创新点在于：1) 采用 Python `typing.Annotated` 的类型注解方式，将表结构与业务语义绑定；2) 在三阶段（本地检查、控制平面规划、运行时）统一校验机制；3) 将契约信息写入 Iceberg 元数据，实现“everything‑as‑code”与跨引擎可组合性；4) 提供完整的 DAG 类型化框架，使得跨语言（SQL、Python）节点能够相互校验。

**🔧 技术方法**

使用技术包括：Python type annotations + `typing.Annotated`、基于 PyArrow 的 `TableSchema`、Arrow 兼容的数据流、Iceberg 表与元数据、Polars/Python sandboxes、SQL 解析与注解、控制平面 DAG 解析与图推理、三阶段强制执行机制。

**📊 数据集**

示例使用 Kaggle Titanic 数据集，构建三节点 DAG（SQL → Python → Python）演示类型注解与校验流程。

**📈 对比分析**

论文未给出定量性能评估或实验对比；主要通过错误案例与设计说明证明 SDK 2.0 能在早期发现 Schema 失配。若要进行比较，可在 SDK 1.0 与 SDK 2.0 之间测量错误率、校验时间、运行时失败率等指标。

**⚠️ 局限性**

局限性：1) 本地检查仅对 Python 节点支持，SQL 仅在控制平面校验；2) 对非 Arrow 兼容引擎的支持有限；3) 缺少完整的性能基准与大规模生产环境评估；4) 处于 beta 阶段，可能存在边缘情况或兼容性问题。

---

## 35. Learning Safe Agent Behaviour from Human Preferences and Justifications via World Models

**arXiv ID:** 2607.13172 | [PDF](https://arxiv.org/pdf/2607.13172v1)

**作者:** Ilias Kazantzidis `[一作]` (University of Southampton), Christopher T. Freeman `[通讯]` (University of Southampton)

**通讯引用:** 28320 | [OpenAlex ID](https://openalex.org/A5058797130)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 DROPJ 方法，利用从真实世界收集的离线轨迹训练世界模型，再通过人类在该仿真器中一次性生成梦境轨迹，收集偏好与安全理由，学习奖励模型，并直接使用 MPC 在真实环境中部署安全策略。

**💡 创新点**

创新点包括（1）一拍即成的梦境轨迹生成技术，显著降低查询生成成本；（2）将安全理由与偏好结合的奖励学习，优先实现安全行为；（3）在没有真实动态或奖励函数的情形下，直接通过 MPC 部署学习到的策略。

**🔧 技术方法**

使用 VAE + MDN‑RNN 训练世界模型，Bradley‑Terry 模型对偏好进行奖励学习，配合多重安全理由权重化的 μ 标注，最终使用采样式 MPC 进行规划与执行。

**📊 数据集**

实验数据集为 OpenAI Gym Car Racing 及其改进版 Obstacle Car Racing 的离线无奖励轨迹（M≈600/800 条），并在仿真器中收集用户生成的梦境轨迹。

**📈 对比分析**

与 DRQV2、ReQueST、DROS、DROP、DROPe 进行对比；DROPJ 在保持相近回报的同时，显著降低了草地/坑洞/碰撞率，计算成本和人类反馈时间也明显更低。

**⚠️ 局限性**

局限性：需依赖大量真实轨迹以构建可靠的世界模型；依赖受过训练的用户进行一次性反馈，难以在更大规模或更复杂任务中广泛应用；若世界模型质量不足，安全性提升效果会下降。

---

## 36. Audio-Text Cross-Attention with Psycholinguistic Support Features for Ambivalence/Hesitancy Recognition

**arXiv ID:** 2607.13345 | [PDF](https://arxiv.org/pdf/2607.13345v1)

**作者:** Luiz F. B. F. Martins `[一作]` (Pontifical Catholic University of Paraná), Alceu S. Britto `[通讯]` (Pontifical Catholic University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于音频-文本跨模态注意力与门控多实例学习的模型，用74维心理语言学支持特征识别视频中的矛盾/犹豫情绪。

**💡 创新点**

创新点：①将跨模态注意力与心理语言学特征相结合；②在多实例学习前注入支持向量以提升窗口重要性；③设计74维手工特征捕捉不确定性、抑扬、情感冲突等信息。

**🔧 技术方法**

技术：5秒重叠窗口，MFCC+eGeMAPS音频描述，RoBERTa情绪嵌入，跨模态多头注意力，门控MIL聚合，五种随机种子集成与阈值调优。

**📊 数据集**

数据集：Behavioral Ambivalence/Hesitancy (BAH) 数据集，采用 11th ABAW Challenge 的公开训练/验证/测试拆分（总计1427条视频）。

**📈 对比分析**

对比：与平均池化基线、仅音频+文本的MIL、加支持特征等做对比，内部结果显示 AP 从 0.828 提升至 0.875，宏 F1 从 0.701 提升至 0.722，支持特征贡献最大。

**⚠️ 局限性**

局限：仅使用音频和文本，排除面部/视觉信息；支持特征依赖专家设计；结果仅为内部验证，未公布正式挑战成绩；数据量有限导致种子方差显著。

---

## 37. Delving into the Temporal Challenges of Unified Video Protection Against Image-to-Video and Fine-Tuning-based Customization

**arXiv ID:** 2607.13336 | [PDF](https://arxiv.org/pdf/2607.13336v1)

**作者:** Yuxin Huang `[一作]` (Sydney AI Centre University of Sydney), Tongliang Liu `[通讯]` (Sydney AI Centre University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种针对视频定制化（参考式和调优式）的统一视频保护方法，使用身份级多帧通用对抗扰动（UAP）对视频VAE的潜在空间进行干扰，阻止模型重现目标身份

**💡 创新点**

首次识别并解决了视频保护中的三大时序挑战：时序压缩、时序过拟合与时序不一致，并通过滑动窗口优化、内在时序建模与外在伪攻击损失提升保护效果与鲁棒性

**🔧 技术方法**

采用了3D视频VAE、扩散模型（LDM）、LoRA调优、滑动窗口投影梯度下降、内在/外在时序一致性约束以及对抗性训练技术

**📊 数据集**

在HDTF、CelebV-HQ和TalkVid三个人脸视频数据集上训练与评估，每个身份各15个训练片段与15个测试片段

**📈 对比分析**

与PhotoGuard、Mist、IDProtector等三种图像级保护基线对比，实验显示在LTX-2.3和Wan2.2视频扩散模型下，所提方法在身份保持率、FDFR、VAE重建PSNR/SSIM、VMAF等指标上均优于基线，且在未见时序攻击下保持鲁棒性

**⚠️ 局限性**

对单一身份的多帧UAP在跨身份或跨内容迁移时仍受限；对抗训练所需的显存和训练时间较高；在极端视频编辑或非视频VAE场景下的适用性尚未充分验证

---

## 38. Tabular Foundation Models for Discrete Choice Estimation

**arXiv ID:** 2607.13314 | [PDF](https://arxiv.org/pdf/2607.13314v1)

**作者:** Liu Liu `[一作]` (University of Colorado Boulder), Dan Zhang `[通讯]` (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种将离散选择问题改写为表格预测任务的方法，使预训练的 Tabular Foundation Model（TabPFN）能够通过零样本或少样本学习实现个体层面预测。

**💡 创新点**

核心创新在于将选择集的相互依赖和消费者异质性显式编码为行级输入，并通过“选择集表示”和“消费者异质性编码”两条轴对 TabPFN 进行结构化重构，从而消除行独立假设与离散选择结构的矛盾。

**🔧 技术方法**

使用 TabPFN 作为预训练的 Transformer，结合 in‑context learning、Fine‑tuning、长/宽/成对（pairwise）等多种表格表示，以及消费者身份编码（categorical）或 per‑respondent ICL 等异质性表示。

**📊 数据集**

在实际零食扫描器面板（98 位消费者、4 个酸奶品牌、每位消费者 5–40 次购买）上进行实验，并将每位消费者最近三次购买作为 hold‑out。

**📈 对比分析**

与传统的无异质性多项式对数几率模型（pooled MNL）以及三种层级贝叶斯混合对数几率模型（单一正态、有限混合、Dirichlet 过程）进行对比。TabPFN 在异质性-aware 设定下，整体 log‑likelihood 提升约 8%（相对 pooled MNL）且 hit‑rate 提升 3.6%，推算时间比 HB 快约 16 倍；在 10–40 次购买的中等数据量区间优势最为显著；Fine‑tuning 对最稀疏区间（5–10 次）有显著帮助。

**⚠️ 局限性**

局限包括：仍需人工设计选择集与异质性表示，缺乏对模型内部经济解释（如价格弹性）与因果推断的直接支持；在深层历史（>40 次）下，层级贝叶斯模型仍能更好捕捉个体偏好；预训练的先验来自合成数据，若真实分布与预训练分布差异大，性能可能受限。

---

## 39. What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors

**arXiv ID:** 2607.13162 | [PDF](https://arxiv.org/pdf/2607.13162v1)

**作者:** Winston Zeng `[一作]` (Emory University), Jinho Choi `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化地构建了一个53项人格特质的向量库，并通过激活空间向量（persona vectors）对语言模型的默认行为、可调行为和不可调行为进行三分法标注；同时对所有两两组合进行交互效果分析，并演示了在安全调优模型中不可调方向的可迁移恢复；

**💡 创新点**

提出将persona vectors视为诊断工具而非控制器，形成自然/可调/不可调三元分类，并首次对19个通用特质的171个组合进行系统交互分类，揭示“可调+可调”组合易产生破坏性干涉；

**🔧 技术方法**

使用对比式激活向量提取（contrastive residual-stream addition）、梯度层级扫描、自动判别器评分（local judge）、向量转移与交叉模型对比；

**📊 数据集**

基于文献选取的四大领域（临床、通用、教育、任务）共53个特质，使用对应的提问式提示集和公开开源模型；

**📈 对比分析**

通过在两个开源模型（Qwen3‑8B、gpt‑oss‑20b）上做全量扫描，量化自然/可调/不可调比例及平均提升；对171对通用特质进行构造性/主导/破坏性三类划分，发现自然特质往往作为锚点；与专家评判对比显示16/17项临床特质的自然度与临床可取性高度一致；

**⚠️ 局限性**

仅评估行为层面，缺乏因果与机制解析；模型样本有限（仅两大模型），不同规模与后训练难以分离；判别器可能受模型偏差影响；可调方向的判别阈值敏感性和可调向量的唯一性未完全验证；

---

## 40. Design and Characterization of a Limb Encircling Actuator

**arXiv ID:** 2607.13337 | [PDF](https://arxiv.org/pdf/2607.13337v1)

**作者:** Japmanjeet Singh Gill `[一作]` (University of Michigan), Elliott J. Rouse `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并制造了能够绕腿部闭合的 Limb‑Encircling Actuator（LEA），并在自制测力平台上对其电机特性、传动、热性能及控制性能进行系统识别与评估。

**💡 创新点**

① 采用全新的“腿部环绕”激励布局，将大直径电机与传动系统包覆在腿部外周，显著降低外部轮廓。② 设计了“bearing‑of‑bearings”微型径向轴承阵列，替代传统大直径轴承，实现轻量化与成本降低。③ 研发自定义无框架无刷电机，与LEA结构紧耦合，提升扭矩密度与热效能。

**🔧 技术方法**

自制无框架无刷电机、径向轴承阵列、专用测试台、基于场定向控制的电机驱动、系统辨识与参数估计（ANOVA、线性回归）、热建模（等效电路与非线性优化）、控制性能评测（步进响应、白噪声频响）等技术。

**📊 数据集**

未使用公开数据集；所有实验数据均来源于本文自建测试平台和自制电机/轴承的实测数据。

**📈 对比分析**

通过将LEA与市售电机（U8‑KV100、EC‑4P 30）在相同负载和转速条件下的扭矩密度、扭矩常数、效率、位移控制带宽等指标进行对比。结果显示：LEA 扭矩密度提升约3.5倍（7.5 Nm/kg），扭矩常数与电机常数提升约3.5倍，正功率区间效率>70%，但由于转子惯量增加，位移控制带宽下降至7.5 Hz。热模型预测连续电流上限13.7 A，峰值电流随运行时间递减。

**⚠️ 局限性**

① 由于必须容纳完整的脚踝结构，内径大、外径仍显宽大，导致腿部侧向突出约30 mm，未能实现紧贴身体。② 轴承阵列的可调预载结构增加了装配复杂度与质量。③ 转子惯量显著增大，导致动态响应受限、控制带宽下降、运行时耗能增加。④ 目前仅在离线平台验证，未集成到完整踝关节外骨骼中，实际佩戴体验与长期功耗仍待评估。

---

## 41. Composable Trust for Language Models: A proven boundary and a measured defense

**arXiv ID:** 2607.13149 | [PDF](https://arxiv.org/pdf/2607.13149v1)

**作者:** Yakov Pyotr Shkolnikov `[一作]` `[通讯]` (Independent Researcher), Yakov Pyotr Shkolnikov (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个条件级联体系，在未修改的语言模型上，通过信任层级、确定性监视器、passivation和wrapper，保证低信任输入只能被过滤或归属性，而不能改变系统操作。

**💡 创新点**

创新点在于：①使用完整性格阵（integrity lattice）和外部监视器，将指令/数据的权限与内容分离并实现无可逆的权威绑定；②将passivation与wrapper按层级化条件化，形成软性防注入屏障；③采用多目标提示搜索（SkillOpt）联合调优防御与质量，提供可解释、可配置的防护。

**🔧 技术方法**

技术手段包括：Lattice-based integrity model、deterministic monitor、token-level passivation prompts、ring‑label wrapper、multi‑objective prompt tuning (SkillOpt)、Claude Opus 4.8 judge、adaptive attacker framework、统计置信区间分析。

**📊 数据集**

使用约600篇自制文章数据集，包含清洁任务、指令注入、内容毒化、交叉环冲突等；训练/验证/测试拆分，外部加入AgentDojo模板；信任层级为 system > user > content > web。

**📈 对比分析**

对比五个消融条件（base、base+prompt、wrapper、passiv、both）和原始模型；在 held‑out 评估中，真实指令注入泄露率从 27% 提升至 94%（[92,100] CI），内容完整性提升至 82%，属性成功率 92%；纯任务质量相对基线仅下降 4%（Q_rel=0.96）。在适应性攻击下，防御率仍保持 87%，权威拒绝率为 0。

**⚠️ 局限性**

局限性：防御仅在文本输出层面被测量，无法完全证明无泄漏；passivation+wrapper 为软性防护，对高度适应的攻击仍有一定失效；提示调优需要手工重调，适配新的信任层级或模型需重新调优；对大规模多源场景的覆盖与泛化仍有待扩展。

---

## 42. OriginBlame: Record- and Token-Level Data Provenance for AI Training Datasets

**arXiv ID:** 2607.13037 | [PDF](https://arxiv.org/pdf/2607.13037v1)

**作者:** Haolin Xue `[一作]` `[通讯]`, Haolin Xue

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了OriginBlame记录级与token级数据来源追踪系统，支持作者级撤回请求的精准忘记集生成。

**💡 创新点**

首次提供无模型依赖、内容可哈希的三层层次架构，实现记录和token级别的作者归因，显著减少数据过度删除。

**🔧 技术方法**

基于内容可哈希（SHA‑256）索引、JSONL存储、WAL并发写入、Rust实现的CLI/库、两阶段重构算法以及独立token索引。

**📊 数据集**

在219,555条中文维基百科页面（约480k作者）上评估，也跨域使用Linux内核源码、HuggingFace数据集与Datatrove。

**📈 对比分析**

与文件级删除、随机忘记集对比，记录级覆盖率提升至100%，过度删除从1.3×降至1.3×，查询低于10 ms，集成吞吐率提升≤4%，在1.7B模型上凭借精确忘记集使NPO忘记效果提升42%。

**⚠️ 局限性**

仅支持增量使用、仅记录一次性归因、单跳追踪、需重建索引；无法后期添加已存在数据的追踪。

---

## 43. LessonBench-V1: A Benchmark Dataset for Evaluating AI Lesson Generation Agents

**arXiv ID:** 2607.13041 | [PDF](https://arxiv.org/pdf/2607.13041v1)

**作者:** Ravidu Suien Rammuni Silva `[一作]` (Nottingham Trent University), Jordan J. Bird `[通讯]` (Nottingham Trent University)

**通讯引用:** 1873 | [OpenAlex ID](https://openalex.org/A5080362193)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 LessonBench-V1 数据集，收集了 647 篇专家编写的课程及其对应的结构化课程计划，覆盖 240 个 STEM 主题。

**💡 创新点**

首个公开的课程计划与完整课程配对数据集，并结合 Bloom、Gagné、Merrill 与 5E 框架进行逆向工程，提供三维评估管线。

**🔧 技术方法**

使用 LLM（Qwen 3.6 进行 Markdown 清理，GPT‑4.1 进行课程计划逆向工程），人工审核；评估采用 BERTScore、ROUGE‑L 与教学对齐度。

**📊 数据集**

选取 TheoremExplainBench 的 240 主题，从 97 个开放教育源（LibreTexts、Brilliant.org、GeeksForGeeks 等）获取 647 篇课程。

**📈 对比分析**

通过语义相似度（BERTScore）、结构相似度（ROUGE‑L）和教学一致性三维评估；初步实验显示 LLM 生成的课程与人类参考在语义/结构上仍有显著差距，教学一致性需要进一步提升。

**⚠️ 局限性**

课程计划由 GPT‑4.1 生成并人工复核，缺乏专家级验证；数据仅包含英文资源，未来需多语言扩展和更系统的真实性验证。

---

## 44. From Human-Centric to Agentic Code Review: The Impact of Different Generations of Generative AI Technology on Review Quality

**arXiv ID:** 2607.13196 | [PDF](https://arxiv.org/pdf/2607.13196v1)

**作者:** Suzhen Zhong `[一作]` (Queen’s University), Ying Zou `[通讯]` (Queen’s University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 207 个开源 GitHub 项目中 1.02 M 条已审查的 PR 进行纵向分析，研究人工、LLM 以及 AI 代理在代码评审中的采用方式、协作模式以及对评审效率和质量（review smells）的影响。

**💡 创新点**

首次系统性量化分析 agentic 代码评审，提出三种 AI 评审采用实践（渐进式、快速 LLM 采用、快速 AI 代理 采用），并通过协作模式与传统评审因素联合建模揭示其对效率与质量的双重作用。

**🔧 技术方法**

采用时间序列软 DTW 聚类识别采用模式，Markov 链 + EM 生成协作模式，统计检验（Wilcoxon、chi‑square、Scott‑Knott ESD）评估差异，逻辑回归与 AUC/影响率评估因素重要性。

**📊 数据集**

使用 1.02 M 条已审查 PR 的公开数据集，来源于 207 个活跃的 GitHub 项目（2019–2026 年），涵盖人类、LLM 与 AI 代理的审查者身份及其交互记录。

**📈 对比分析**

通过比较预 LLM、LLM 与 AI 代理三个时代以及三种采用实践，发现 Gradual AI 采用与 Rapid AI 代理 采用可使审查时间平均下降 2.5–4.5 天/千行代码，但未提升 review smell 率；相反 Rapid LLM 采用导致 smell 率上升 8% 点。整体表明 AI 代理能提升效率，但需避免单一模型依赖。

**⚠️ 局限性**

局限性包括：基于观察性数据，缺乏因果推断；‘AI 代理时代’定义可能包含 LLM 参与的 PR；工具与模型快速演进，研究结果可能随技术更新而变；仅覆盖公开开源项目，普适性待验证。

---

## 45. Self-Improving AI Coding Agents Through Accumulated Behavioral Rules: A Closed-Loop Framework

**arXiv ID:** 2607.13091 | [PDF](https://arxiv.org/pdf/2607.13091v1)

**作者:** Aditya Aggarwal `[一作]` (Microsoft), Nahid Farhady Ghalaty `[通讯]` (Microsoft)

**通讯引用:** 557 | [OpenAlex ID](https://openalex.org/A5063803648)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了一个闭环框架，将人工评审反馈转化为持续的行为规则，让LLM编码代理在多次会话中持续改进

**💡 创新点**

创新点在于通过可版本化、持久化的Markdown规则文件实现跨会话的知识迁移，无需模型权重更新或重新训练

**🔧 技术方法**

技术包括结构化规则文件、自动验证脚本、版本控制、在代理前置上下文加载规则、以及自检清单执行

**📊 数据集**

使用真实生产环境的代码库（35+微服务，约5万行）以及36个PR评审的实际反馈作为数据来源

**📈 对比分析**

与Reflexion、ExpeL、Voyager、CodeReviewer等方法对比，结果显示规则集实现了0%已修复错误类的再次出现，并将评审重点从机械错误转向设计问题；在同一任务流上不需要额外的模型训练或RLHF

**⚠️ 局限性**

局限包括单一组织实验、缺乏对照组、规则集可能随时间膨胀导致上下文窗口限制、规则冲突需手工治理以及对评审文化的高度依赖

---

## 46. Why Not Fix It Once and for All? An Empirical Study of Multiple Patches for Vulnerability Fixes in Open-Source Software

**arXiv ID:** 2607.13206 | [PDF](https://arxiv.org/pdf/2607.13206v1)

**作者:** Weiliang Qi `[一作]` (University of Texas at Dallas), Xinda Wang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 349 | [OpenAlex ID](https://openalex.org/A5102758573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统地研究了开源软件中多补丁（multi‑patch）漏洞修复的现象，构建了 1,646 条多补丁记录的手工标注数据集，提出了六类多补丁的分类法，并与单补丁修复进行对比；随后评估了现有漏洞检测模型和代码克隆工具在判断是否需要后续补丁方面的效果。

**💡 创新点**

创新点包括：①首次以实证方式全面剖析多补丁修复的根因与特征；②提出了基于原因的六类多补丁分类体系；③构建并公开了首个多补丁修复的手工标注数据集；④在此基础上验证了现有漏洞检测与代码克隆技术对多补丁场景的适用性，揭示其明显不足。

**🔧 技术方法**

使用的技术主要有：手工标注与代码书写；Levenshtein 距离度量补丁相似度；多种预训练模型（CodeBERT、UniXcoder、PDBERT 等）用于漏洞检测；图神经网络模型（Devign、ReVeal）用于漏洞检测；ReDebug 与 FIRE 两种代码克隆工具用于检测多位置漏洞。

**📊 数据集**

数据集：来自 NVD 的 25,113 条 OSS 安全补丁记录，其中 1,646 条记录对应多补丁修复（平均每条 2.55 个补丁）。该多补丁数据集已公开发布（<https://huggingface.co/datasets/XSec-Lab/MultiPatch>）。

**📈 对比分析**

对比方法：①把多补丁与单补丁在编程语言、项目、漏洞类型、时间分布等维度进行统计；②利用六种漏洞检测模型对多补丁中的不完整补丁（C1）进行检测，并与单补丁检测结果做对比；③用 ReDebug 与 FIRE 对多位置补丁（A1、A2）进行代码克隆检测，比较 TPR/FPR。性能方面：所有漏洞检测模型在多补丁检测中的准确率和 F1 均低于 50%，TPR 仅 7–28%；代码克隆工具在多位置场景下 TPR 从 90% 降至 52%，FPR 亦下降，表明现有技术在多补丁环境中表现不佳。

**⚠️ 局限性**

局限性：①标注依赖人工分析，可能存在主观偏差；②仅使用 NVD 数据，缺乏其他来源的补丁记录，且可能遗漏 “silent” 补丁；③对特定工具与模型的评估不一定能推广到所有技术；④研究聚焦 OSS 开源项目，对闭源或商业软件的多补丁情况不作讨论。

---

## 47. Worlds in One Demo: A Synthetic Data Engine for Learning Open-World Mobile Manipulation

**arXiv ID:** 2607.13154 | [PDF](https://arxiv.org/pdf/2607.13154v1)

**作者:** Lingxiao Guo `[一作]` (Carnegie Mellon University), Guanya Shi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5029314167)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种从单个真实演示生成合成数据的引擎，训练开世界移动操控策略。

**💡 创新点**

创新点在于利用世界重建与可变背景、交互段重排、Corrective State Expansion以及VLM驱动的空间配置，实现在仅一示例下的空间、长程、跨环境与跨实体泛化。

**🔧 技术方法**

使用RGBD重建、Gaussian splatting（MAtCha）、BundleSDF、全身运动规划、RRT-Connect、Corrective State Expansion、Marble 3D生成、因子化渲染、Gemini 3.1 Pro 与 SAM3 进行图像生成与VLM引导。

**📊 数据集**

数据集包含单一 Agibot G1 真实演示、Bigym 与 BEHAVIOR Challenge 仿真任务，以及通过一张照片生成的 Marble 场景。

**📈 对比分析**

与基线 Teleoperation/UMI/模拟数据相比，在仿真中仅需 50 倍数据效率即可匹配 50 示例效果；在真实世界中完成 5 项长程任务的平均进度为 54.8%，并在 Linearbot 上实现 40% 的零样本跨实体部署。

**⚠️ 局限性**

局限性在于仅处理刚性/可联动物体，软体物体难以重建；多阶段流程易累计误差，需要人工干预纠正。

---

## 48. Adapting a Diffusion-Based Music Synthesis Model to Human Voice Conversion

**arXiv ID:** 2607.13278 | [PDF](https://arxiv.org/pdf/2607.13278v1)

**作者:** Ben Maman `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Meinard Müller `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将用于多乐器音乐合成的扩散模型迁移并适配到语音和歌声转换，形成统一的音频生成框架。

**💡 创新点**

创新点在于把音乐模型的条件机制（乐谱、音色）扩展为语音的音素后验、音高和说话人/歌手身份，并在同一模型中同时处理语音、歌声和混合音频。

**🔧 技术方法**

使用基于 T5 的注意力扩散模型、FlowMAC、Feature‑wise Linear Modulation（FiLM）、音素后验图（PPG）和 CREPE 估计的音高，以及 BigVGAN 声码器。

**📊 数据集**

数据集包括约 33 小时的英语语音（5 位说话人）、约 31 小时的歌声（8 位歌手）、约 90 小时的带伴奏混音（多位歌手+钢琴），以及合并的无标注大规模音频集合。

**📈 对比分析**

通过 MUSHRA 听力测试、Fréchet Audio Distance、音高准确率和 PPG 距离等指标与专用 VC/SVC 模型比较。T5 扩散模型在自然度和表演者相似度上与专用模型相当或更优，但在加入伴奏数据后性能下降约 10–15 分。

**⚠️ 局限性**

主要局限是混合训练导致语音/歌声质量下降，音素保真度不如专用模型，且对不同域的数据平衡与模型容量要求较高。

---

## 49. Just-In-Time Scene Graph Growth: Combating Perceptual Saturation in Long-Horizon Robotics

**arXiv ID:** 2607.13245 | [PDF](https://arxiv.org/pdf/2607.13245v1)

**作者:** Yue Chang `[一作]` (Hong Kong University of Science and Technology), Sihong Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了JITOMA框架，实现按需即时生成3D场景图，结合任务意图、感知与记忆；

**💡 创新点**

创新点在于：1）Just‑In‑Time场景图生长机制，前端任务热图过滤感知；2）后端LLM动态激活“沉睡”锚点，仅在需要时进行密集标注和功能子图构建；3）设计了三层评测JITOMA‑Bench，覆盖单查询、长时序多任务与复杂指令；

**🔧 技术方法**

技术细节包括：LLM（Qwen‑3.5‑9b）进行意图解析与重新排名；CLIP图像/文本嵌入做轻量检索；DAM captioning批量生成节点描述；VLM用于定位功能部件并生成3D边；全流程采用闭环在线处理；

**📊 数据集**

数据集为改造后的Clio‑Bench，包含Apartment、Office、Cubicle三场景的RGB‑D轨迹，配合生成的多任务序列与复杂指令，构成JITOMA‑Bench；

**📈 对比分析**

与ConceptGraphs、ReasoningGraph、Clio等基线对比，评估IoU、mR@1/3、active object count、Peak、TPF；JITOMA在准确率、活跃图规模与实时性上均优于基线，尤其在Tier 2长时序任务切换与Tier 3复杂指令中表现突出；

**⚠️ 局限性**

局限性包括：1）仍需DAM captioning与VLM计算，导致一定延迟；2）对LLM解析精度依赖较大；3）在极大场景中可能需要更多沉睡锚点；4）对动态物体的实时更新支持有限；5）对硬件性能有一定要求。

---

## 50. GSM-Plus-BN: A Perturbation-Based Benchmark for Bangla Mathematical Reasoning in Large Language Models

**arXiv ID:** 2607.13248 | [PDF](https://arxiv.org/pdf/2607.13248v1)

**作者:** Bidyarthi Paul `[一作]` (Southeast University), Swastika Kundu `[通讯]` (Ahsanullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了一个 Bengali 版的扰动数学推理数据集 GSM-PLUS-BN，并在该数据集上系统评估了六款开源 LLM 的推理能力。

**💡 创新点**

创新点在于首次将 GSM-PLUS 的扰动框架迁移到低资源语言 Bangla，提供了全新的评价基准，并对标准提示与 Chain‑of‑Thought（CoT）提示在 Bangla 上的效果进行了对比。

**🔧 技术方法**

使用了扰动生成技术、标准提示与 CoT 提示技术，并通过 Groq API 对开源 LLM 进行推理评测。

**📊 数据集**

使用的数据集为 GSM-PLUS-BN（共 10,544 条题目，来源于 GSM‑Plus 并经过双人人工校对的 Bangla 翻译），同时参照 GSM8K 作为种子题目。

**📈 对比分析**

通过在八种扰动类型（数值替换、数字扩展、整数-小数-分数转换、加法操作、逆操作、问题理解、干扰词插入、关键推理）下，比较标准提示与 CoT 的准确率；结果显示 GPT‑OSS‑120B 在种子题上最高，Llama‑3.3‑70B‑Versatile 在扰动题中持续保持 85% 以上，CoT 对大多数模型提升显著，尤其是 Qwen3‑32B 最高提升约 23.9%，但 Critical Thinking 仍低于 34%。

**⚠️ 局限性**

局限性包括仅评估了六款开源模型，未包含商业模型；提示策略仅限标准与 CoT，未探索更高级的提示；扰动类型受限于设计方案，未覆盖拼写错误、口语化、代码切换等真实场景；评价指标为 Exact Match，缺乏对部分正确性和推理链的细粒度分析。

---

## 51. Precomputing the Future-Offset Average in TriAttention

**arXiv ID:** 2607.13051 | [PDF](https://arxiv.org/pdf/2607.13051v1)

**作者:** Amarnath Mukherjee `[一作]` `[通讯]` (Hozhoke), Amarnath Mukherjee (Hozhoke)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对TriAttention方法进行了改进，该方法用于缩小长推理大语言模型的KV缓存。通过对每个缓存的键进行评分，决定保留哪些键，本文提出了一种新的评分计算方式，减少了计算复杂度。

**💡 创新点**

创新点在于将TriAttention中的17个未来距离的平均评分简化为一个单一的每带权重，从而将评分计算的复杂度从O(N F |D|)降低到O(N F)，实现了计算效率的提升。

**🔧 技术方法**

使用了TriAttention方法的评分机制，并通过代数身份将多次计算简化为一次计算，保持了评分的准确性。

**📊 数据集**

论文中提到的实验使用了多个领先的模型（如Qwen3、Qwen2.5、Llama3），但具体的数据集未在摘要中详细说明。

**📈 对比分析**

与传统的TriAttention方法相比，改进后的方法在评分计算上节省了计算时间，具体表现为将17次评分计算减少为1次，性能得到了显著提升。

**⚠️ 局限性**

限制在于该改进仅针对TriAttention方法的评分计算部分，未能影响注意力内核的计算，也不改变缓存预算或修剪频率。

---

## 52. Analyzing Curricular Pattern Complexity Using AI to Improve On-Time Graduation Rates

**arXiv ID:** 2607.13094 | [PDF](https://arxiv.org/pdf/2607.13094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 53. Beyond AI-Generated Labels: Watermarking, Co-Creation, and Conflation of AI-Generation with Disinformation

**arXiv ID:** 2607.13082 | [PDF](https://arxiv.org/pdf/2607.13082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 54. Power from Potential: A Survey of Electrostatic Actuators for Haptics

**arXiv ID:** 2607.13058 | [PDF](https://arxiv.org/pdf/2607.13058v1)

**作者:** Ahad M. Rauf `[一作]` (Stanford University), Daniel Leithinger `[通讯]` (Cornell University)

**通讯引用:** 2581 | [OpenAlex ID](https://openalex.org/A5040822991)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述并系统评估高压电静态驱动器（HVEA）在触觉接口中的应用与设计，涵盖ESAs、DEAs、SEHAs与EKPs四大类；

**💡 创新点**

首次对HVEA技术进行跨领域、跨模态的综合评估，提出安全规范、可制造性、可自感知化等设计准则，并展望多模态融合与可穿戴化方向；

**🔧 技术方法**

采用高压电场力学原理与电容耦合机制，结合PRISMA系统综述、可制造工艺（PCB、硅胶模塑、激光切割）、高压驱动与自感知电路设计；

**📊 数据集**

使用303篇学术论文（来自Web of Science、IEEE Xplore、ACM DL、Wiley、Elsevier、Nature、Science）作为评估数据集；

**📈 对比分析**

通过绘制Ashby曲线和性能指标（最大位移、阻力密度、带宽、能耗）与传统电机/气动/压电器进行对比，显示HVEA在功率密度、响应速度和尺寸可扩展性上优于传统技术；

**⚠️ 局限性**

局限在高压安全与可靠性（易受湿度/污染影响）、工艺复杂度、可视化与用户体验评估不足，以及多模态融合与长期耐久性研究待完善。

---

## 55. Executable JavaScript as a Checkable Specification Language: A JS-SAM Case Study on SysMoBench

**arXiv ID:** 2607.13092 | [PDF](https://arxiv.org/pdf/2607.13092v1)

**作者:** Jean-Jacques Dubray `[一作]` `[通讯]`, Jean-Jacques Dubray

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在SysMoBench基准上实现并评估了一个新的非形式化后端：可执行JavaScript（SAM模式）规范，并在三种系统（Asterinas自旋锁、分布式锁服务、Etcd Raft）上进行实验。通过构建可重复的内核轨迹捕获、直接窗口重放、单轮反馈修复和跨语言对比等方法，研究LLM在写规范时受语言、规范形状与提示的影响。

**💡 创新点**

①首次为SysMoBench提供可执行JavaScript后端并实现直接Phase‑3重放；②提出“合同最小化+提示控制+跨语言三因素实验”以消除模型与语言的混杂；③证明合同形状是影响一致性验证的主因，而非语言本身；④展示LLM在单轮修复时的自我纠错能力。

**🔧 技术方法**

使用SAM模式（State‑Action‑Model）实现可执行模块，Node.js + Python执行器；使用TLC和自研探测器做模型检查；采用NDJSON窗口重放做Phase‑3；对比时利用Permutation test、McNemar等统计方法；利用Prompt engineering进行提示控制。

**📊 数据集**

SysMoBench任务集（11个系统）中的3个系统；Spinlock 28窗口、分布式锁服务60窗口、Raft 97窗口；每个系统使用5个模型生成N=5个规格；所有数据和脚本均已公开。

**📈 对比分析**

对比4个LLM（Claude Opus 4.8、Fable 5、Sonnet 4.6、Haiku 4.5）在不同语言（TLA、JS）和合同/提示组合下的四阶段得分。结果显示：①Phase‑3（一致性）是唯一区分模型的阶段；②合同形状在所有语言中决定准确率；③在JS中，若给定相同合同，准确率可达100%；⑥单轮修复后，强模型可完全修复，弱模型无法修复；总体性能在Spinlock上表现最佳，Raft上表现差异显著。

**⚠️ 局限性**

①仅覆盖两变量的可观测窗口，未考虑阻塞路径和更复杂交互；②跨语言实验只针对Spinlock，缺乏更大规模验证；③统计功效受N=5、生成数量有限；④提示和合同的手工编写带来主观性；⑤缺少对liveness、无穷状态空间等更高阶验证能力的考察。

---

## 56. Differentiable Polarized Path Tracing

**arXiv ID:** 2607.13265 | [PDF](https://arxiv.org/pdf/2607.13265v1)

**作者:** Pramod Rao `[一作]` (Max Planck Institute for Informatics), Delio Vicini `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种内存高效的可微分极化路径追踪方法，通过缓存后缀重放避免对秩缺失的Mueller矩阵求逆，从而实现稳定的梯度计算。

**💡 创新点**

创新点在于将缓存后缀重放与混合缓存-重计算策略相结合，解决了极化渲染中Mueller矩阵不可逆导致的梯度不稳定问题，并兼顾内存与计算效率。

**🔧 技术方法**

采用了Mueller‑Stokes算子、路径重放反向传播（PRB）改进、局部缓存、混合缓存-重计算策略，以及Mitsuba 3渲染器进行实现。

**📊 数据集**

使用了Cornell Box、Kitchen、Veach、Living Room等基准场景以及七个网格的单视角重建数据集进行实验验证。

**📈 对比分析**

与传统自动微分（Conv. AD）、P‑PRB、P‑RB 等方法对比，梯度准确性与 AD 相当，内存占用近似恒定且低于 AD，运行速度约比 P‑RB 快一倍，并在极化场景下实现了更精确的材质、纹理和几何恢复。

**⚠️ 局限性**

局限性包括未覆盖亚表面散射，混合缓存方案在深路径时仍需在内存与重计算成本之间权衡，以及对光源和材料模型的极化假设可能限制在更复杂光学介质上的应用。

---

## 57. Hybrid multi-objective evolutionary algorithms for service placement in the computing continuum: a comparative study with genetic traceability

**arXiv ID:** 2607.13200 | [PDF](https://arxiv.org/pdf/2607.13200v1)

**作者:** Sergi Vivo `[一作]` (University of Balearic Islands), Isaac Lera `[通讯]` (University of Balearic Islands)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并评估了一种基于岛屿模型的异构多目标遗传算法，用于计算连续环境中服务部署问题的多目标优化；

**💡 创新点**

创新点在于提出异构岛屿协同框架，系统分析多种主流MOEA的互补性，并引入遗传负载指标以追踪算法贡献；

**🔧 技术方法**

采用pymoo实现的NSGA-II/III/SMS-EMOA/U-NSGA-III/MOEA/TS/MOCPO等多目标遗传算法，配合全连或环形迁移拓扑、遗传负载跟踪、Wilcoxon与Friedman等非参数统计方法；

**📊 数据集**

使用人工生成的服务放置实例：第一组为50节点、50应用、25用户；第二组为50节点、30应用、50用户；

**📈 对比分析**

通过GD/IGD/HV/S/STE等指标，对30次重复实验进行Wilcoxon检验或Friedman检验，结果显示第一组混合方案显著优于单一算法，第二组优势相对有限；

**⚠️ 局限性**

局限性包括：迁移频率与替换策略固定，缺乏自适应控制；仅在人工生成实例上验证，未涵盖更大规模或真实场景；遗传负载仅追溯线性历史，未能定位具体基因贡献。

---

## 58. Reflecting Process Expertise in Procedural Material Generation

**arXiv ID:** 2607.13318 | [PDF](https://arxiv.org/pdf/2607.13318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 59. No Attention, No Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA

**arXiv ID:** 2607.13106 | [PDF](https://arxiv.org/pdf/2607.13106v1)

**作者:** Suraj Karki `[一作]` (Bielefeld University of Applied Sciences and Arts), Thorsten Jungeblut `[通讯]` (Bielefeld University of Applied Sciences and Arts)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对边缘FPGA平台，设计并实现了适配DPU的YOLOv11、YOLOv26及其倾斜检测变体，完成端到端的模型定制、量化与部署；

**💡 创新点**

创新点在于：①将不被DPU支持的激活函数和拆分操作改为ReLU与1×1卷积；②提出面向DPU的空间注意力近似方案；③在同一硬件平台上系统评测标准与倾斜目标检测的八种DPU架构，提供完整的性能基准；

**🔧 技术方法**

采用了Vitis AI工具链、FPGA Xilinx ZCU104 SoC、量化后推理、C3k2与C2PSA注意力模块的改造、以及基于FPGA的数据流与硬件加速技术；

**📊 数据集**

使用了COCO、Pascal VOC、KITTI、DOTA、DIOR-R、以及自制的人体检测数据集，共六个基准集；

**📈 对比分析**

通过mAP@0.5:0.95、FPS、延迟、功耗及资源占用等指标对CPU与FPGA两种部署方式进行对比，结果表明YOLOv26n在B4096架构下实现34.05 FPS（标准）/29.55 FPS（倾斜）且功耗仅为先前方案的三分之一；

**⚠️ 局限性**

主要局限在于后处理（NMS）是瓶颈，导致FPGA端到端吞吐量受限；同时量化引入的5%精度损失以及缺乏多DPU并行化策略等。

---

## 60. Harness Handbook: Making Evolving Agent Harnesses Readable,Navigable, and Editable

**arXiv ID:** 2607.13285 | [PDF](https://arxiv.org/pdf/2607.13285v1)

**作者:** Ruhan Wang `[一作]` (Tencent HY LLM Frontier), Leoweiliang `[通讯]` (Tencent HY LLM Frontier)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了“Harness Handbook”——一种以运行时行为为中心的手册，用来精确定位并编辑生产级代理机架的实现代码；并设计了基于手册的行为导向渐进披露（BGPD）工作流以及自动重同步机制。

**💡 创新点**

创新点在于：①将系统行为与源代码实现直接关联，打破传统以文件/函数为主的仓库表示；②提出BGPD，将行为描述层层展开到具体实现点；③实现了局部重同步，仅在差异点处更新手册，保持高效一致性。

**🔧 技术方法**

技术手段包括：静态程序分析提取调用图和边界；LLM辅助的行为结构化与阶段分配；层次化合成构建 L1–L3 文档树和跨阶段状态视图；计划生成器、执行器与差分驱动的手册重同步。

**📊 数据集**

实验数据集由两套开源代理机架（Terminus‑2 与 Codex）组成，共 60 条行为驱动的修改请求（30 条/机架），按请求类型（Query、Cross‑file、Search‑Hostile）和定位难度（Easy、Medium、Hard）划分。

**📈 对比分析**

对比方法：Baseline（直接浏览仓库）与 Handbook‑Assisted（手册引导）在 3 位评判模型（GPT‑5.5、Opus‑4.8、DeepSeek‑V4‑Pro）下的计划质量评分、定位精度（Recall/Precision/F1）、Token 预算使用。结果显示：Handbook‑Assisted 在 Codex 上提升 10.0pp、Terminus‑2 上提升 18.9pp 的总体胜率；Token 使用分别下降 12.7% 与 8.6%；在文件与符号级别的 Recall/Precision/F1 均显著高于 baseline，尤其在“Wrong”指标上降低 25.9pp。

**⚠️ 局限性**

局限性包括：依赖准确的阶段骨架或推断能力，静态分析可能忽略动态调用；LLM 辅助步骤需人工复核；手册在大型、多语言或高度异构的机架上构建与维护成本可能较高；当前评估仅覆盖修改规划，未系统验证自演化或其他安全性问题。

---

## 61. Targeted Recovery of Weight-Space Mechanisms From Neural Networks

**arXiv ID:** 2607.13047 | [PDF](https://arxiv.org/pdf/2607.13047v1)

**作者:** Antoine Vigouroux `[一作]` (MATS), Lee Sharkey `[通讯]` (Goodfire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Targeted Parameter Decomposition (tPD)，通过引入高秩的残差组件只拆解目标输入相关的权重子空间，从而在保持机制可解释性的同时显著降低计算成本。

**💡 创新点**

创新点在于：①使用对抗性消融迫使非目标数据使用残差组件，从而让目标子组件不受非目标激活的干扰；②通过双流训练（目标+非目标）同时满足子组件稀疏与全模型重构，解决了传统PD在目标数据下的欠定性问题。

**🔧 技术方法**

技术手段包括：参数分解为秩-1子组件 U_cV_c^⊤、对抗性消融、残差 Δ 组件、双流数据训练、余量匹配与余弦相似度评估。

**📊 数据集**

实验数据集包括：Toy Model of Compressed Computation (TMCC)、The Pile（训练 4‑block Transformer 以及 12‑block Transformer）以及从 The Pile 过滤出的 CSS、Python、JavaScript 子集。

**📈 对比分析**

与全数据PD相比，tPD 在 toy 模型上仅需 5 个子组件且训练步数从 2500 降至 500；在 4‑block Transformer 上仅使用 7% FLOPs 就获得与完整解构相同甚至更好的重构精度（KL ≈0.45），并成功实现 CSS 代码子模型提取、精确的子组件消融与重连，保持非目标输入几乎无影响。

**⚠️ 局限性**

局限性：①仅能解释目标数据所覆盖的激活子空间，无法覆盖全模型可能的冗余机制；②对更大规模模型的可扩展性尚未充分验证；③对复杂知识（如世界事实）的编辑仍需更细粒度的目标数据与编辑策略；④可能会因残差组件聚合导致部分重要子组件被忽略。

---

## 62. Text2Sign: A Single-GPU Diffusion Baseline for Text-to-Sign Language Video Generation

**arXiv ID:** 2607.13164 | [PDF](https://arxiv.org/pdf/2607.13164v1)

**作者:** Ruize Xia `[一作]` `[通讯]`, Ruize Xia

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个单GPU文本到手语视频生成的扩散模型Text2Sign，能够从文本提示直接生成短时长的手语视频；

**💡 创新点**

创新点包括：1) 将全3D注意力分解为空间与时间两步的factorized attention，大幅降低计算复杂度；2) 使用冻结的CLIP文本编码器作为条件，避免额外训练且提升验证损失；3) 在单GPU资源下实现了可行的3D UNet+DiT结构；

**🔧 技术方法**

技术主要包括：Denoising Diffusion Probabilistic Models（DDPM/DDIM）、三维UNet骨干网络、DiT式Transformer块、AdaLN时序归一化、cross‑attention文本条件化、冻结CLIP文本编码器、cosine噪声调度、梯度检查点、AMP混合精度训练；

**📊 数据集**

使用How2Sign数据集的64×64分辨率、32帧短片，并采用签名者分离的90/10划分；

**📈 对比分析**

与多种对照模型比较，完整版本在单GPU训练下达到了0.0648的验证MSE，FVD‑proxy 7083，Temporal Consistency 1.0；相较于仅卷积或全3D注意力的版本，验证损失降低约19.5%；在8步DDIM+CFG=5.0下，生成32帧视频耗时12.6秒，显著低于多GPU基线；

**⚠️ 局限性**

局限性包括：仅低分辨率（64×64）短片，缺乏细粒度手部与面部细节；文本提示的语义控制仍较弱；单GPU资源限制导致模型规模与训练时长受限；缺乏对真实手语语言学的定量评估，无法确保生成视频的语言可理解性；

---

## 63. Automatic Differentiation from Scratch: How PyTorch Computes Gradients in Physics-Informed Neural Networks

**arXiv ID:** 2607.13042 | [PDF](https://arxiv.org/pdf/2607.13042v1)

**作者:** Abdeladhim Tahimi `[一作]` (Universidade Federal de Alagoas), Abdeladhim Tahimi `[通讯]` (Universidade Federal de Alagoas)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5046777051)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

解析PyTorch自动微分引擎在PINN训练中如何完成两级微分（求物理导数与参数梯度），并通过完整的计算图追踪给出每一步的数值；

**💡 创新点**

首次用图‑on‑图机制说明并验证自动求导如何生成二阶混合导数，填补了理论推导与实际实现之间的空白；

**🔧 技术方法**

使用PyTorch autograd、反向传播、create_graph、torch.autograd.grad等技术，并结合手工推导与Jupyter Notebook逐步演算；

**📊 数据集**

以1‑3‑3‑1多层感知器和一阶初值问题y′+y=0、y(0)=1作为实验示例；

**📈 对比分析**

通过手工推导与代码实现逐步对比，验证两者数值完全一致，证明实现无误；

**⚠️ 局限性**

仅针对低阶单变量ODE和小型网络做了验证，未探讨高阶/多维PDE及更大网络的内存/计算开销，需要进一步研究。

---

## 64. HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration

**arXiv ID:** 2607.13056 | [PDF](https://arxiv.org/pdf/2607.13056v1)

**作者:** Chang Liu `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**通讯引用:** 5007 | [OpenAlex ID](https://openalex.org/A5009985839)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 HRIBench，一个基于可执行交互脚本的意图感知人机协作诊断基准，覆盖三类角色（指导者、协作者、入侵者），并通过可解释的协作指标评估机器人政策在动态人机交互中的表现。

**💡 创新点**

创新点在于：①将协作结构抽象为可执行脚本，显式建模角色、时间依赖、协议与人类行为分布；②引入三种典型交互角色，聚焦意图沟通、时序同步与鲁棒性；③设计交互导向的评价指标（同步、响应、协议合规、安全），实现失败模式可解释化；④通过仿真生成多样化轨迹并验证，可用于策略适配与诊断。

**🔧 技术方法**

使用的技术包括：可执行脚本编译与仿真（基于Franka Panda）、人类动作生成与细化（HY‑Motion‑1.0）、VLA 与模仿学习模型（GR00T N1.5、π_0.5、ACT）通过 LoRA 或全量微调适配，交互指标自动化计算与评估。

**📊 数据集**

数据集：HRIBench 自生成的数据，共 13 任务、650 条可执行评估轨迹（每任务 50 条训练、10 条评估），以及 LeRobot SO‑100 真实演示的 20 条轨迹用于 sim‑to‑real 适配。

**📈 对比分析**

比较方法：在统一仿真协议下，对 GR00T N1.5 LoRA、π_0.5 LoRA、ACT 全量微调三种政策进行角色条件评测；结果显示：指导者 CSR≈0.53、协作者≈0.5，但入侵者仅 0.10；在真实实验中，Sim+Real 训练使任务成功率从 0.10 提升至 0.43，表明 HRIBench 生成数据显著提升协作性能。

**⚠️ 局限性**

局限性：仅为仿真基准，人工生成的人类动作和碰撞模型无法完全捕捉真实人类行为与接触细节；Real‑world 适配实验规模有限，受制于演示数据量、机器人差异与预训练背景；指标评估虽细化，但仍需在更复杂环境中进一步验证。

---

## 65. Evaluation Ability Does Not Imply Optimization Utility: LLM-as-a-Judge Signals in Closed-Loop Table Recognition

**arXiv ID:** 2607.13347 | [PDF](https://arxiv.org/pdf/2607.13347v1)

**作者:** Donghwan Kim `[一作]` `[通讯]` (Aidentyx Inc.), Donghwan Kim (Aidentyx Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在表格识别任务中使用 LLM 作为判别器（judge）来驱动闭环迭代重生成，并评估其对质量提升的实际效果。

**💡 创新点**

发现判别器的评分信号极弱、无法可靠选择更优候选，且迭代往往导致严重结构性损失；但引入结构保持约束可显著降低极端损失，揭示了目标保留失败是损失的主要原因。

**🔧 技术方法**

采用 Gemini 3.1 Flash Lite 作为生成器，零参考 LLM 判别器提供分数与错误列表；使用确定性 TEDS/ S‑TEDS 作为评估指标；构建了 8 轮闭环迭代框架。

**📊 数据集**

在 FinTabNet（476 张表）和 OmniDocBench（272 张表）两大表格数据集上进行实验，二者均遵循相同的生成/判别/迭代协议。

**📈 对比分析**

对比多种判别器配置、评分方式（点式/对比式）、随机/最佳/最终/oracle 选取策略及三种 tie‑breaking 规则；结果显示：无论何种配置，判别器均未能稳健击败随机选取；FinTabNet 的整体效应为显著退化（-0.02 TEDS），OmniDocBench 接近零；结构保持约束将“严重损失”率从 3.6% 降至 0.8%（FinTabNet）且方向一致（OmniDocBench）。

**⚠️ 局限性**

局限性包括：TEDS 只评估首张表且受 GT 序列化约定影响；判别器分数存在显著非确定性；实验设计中有温度/后备模型混用、tie‑breaking 规则依赖；缺乏人类评估；仅针对无参考的闭环场景，结果不一定能推广至可验证答案的领域。

---

## 66. Reassessing Muon for Matrix Factorization

**arXiv ID:** 2607.13246 | [PDF](https://arxiv.org/pdf/2607.13246v1)

**作者:** Ali Parviz `[一作]` (UC San Diego), Alex Cloninger `[通讯]` (UC San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在低秩矩阵因式分解任务上，系统评估并比较了Muon、AdamW、GD和SignGD优化器，探讨了超参数、条件数和谱结构对性能的影响。

**💡 创新点**

提出将Muon's谱正交化与传统自适应方法在受控问题上的对比，强调了超参数调优和问题结构决定性能的观点。

**🔧 技术方法**

采用Newton–Schulz正交化、梯度投影、学习率扫描、几何平均损失等技术进行实验。

**📊 数据集**

使用合成低秩矩阵、矩阵完成、非负矩阵因式分解、张量训练以及高斯核矩阵等数据集。

**📈 对比分析**

通过学习率搜索和误差指标，发现Muon's优势仅在特定NMF和极度条件数矩阵完成中显现，其他场景AdamW或GD往往更优。

**⚠️ 局限性**

结论受限于仅评估受控问题，未覆盖更复杂深度学习模型；Muon's谱正交化在高维、非线性任务中的效果仍未知。

---

## 67. Marker-free deformable registration and fusion for augmented reality-guided positive margin localization during tumor resection surgery

**arXiv ID:** 2607.13343 | [PDF](https://arxiv.org/pdf/2607.13343v1)

**作者:** Yue Yang `[一作]` (Vanderbilt University), Jie Ying Wu `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并验证了一套无标记物的增强现实（AR）工作流程，用于头颈部肿瘤复切手术中正性切缘的定位。

**💡 创新点**

创新点在于将轮廓约束的可变形配准与面向表面的 marker‑free AR 融合相结合，省去了传统定位所需的外部标记，直接将病理标注的三维模型映射到患者手术现场，并在可视化层面实现边缘锚定。

**🔧 技术方法**

核心技术包括基于 Kelvinlet 的轮廓约束可变形注册、点到面 ICP 的残差配准、HoloLens 2 的 AHAT 深度摄像头自标定与局部深度校正、以及利用表面显著性权重的 drape‑aware 在线融合算法。

**📊 数据集**

实验使用了两名人头干尸（cheek 和 scalp）上的切缘数据，并通过 3D 结构光扫描（EinScan SP）和 ZED 2i 深度相机采集的表面点云来构建三维模型；此外还在干尸上放置了荧光标记作为地面真实值。

**📈 对比分析**

通过与传统标记‑based AR 融合以及三种引导条件（口头、检查+口头、AR）进行比较，AR 指引在末端定位误差上相较于口头指导下降了约 70%（从 21.4 mm 降至 6.2 mm），且 marker‑free 融合误差与 marker‑based 误差无显著差异，平均约 2.15 mm。

**⚠️ 局限性**

局限性包括样本量仅为两名干尸、仅评估了面部和头皮切缘、扫描时间仍达约 10 min、并且在严重遮挡下融合精度会下降，后续工作需进一步提升配准鲁棒性、减少人工步骤并进行更大规模的临床验证。

---

## 68. The Reconstructions of Konrad Zuse's Z3 Computer

**arXiv ID:** 2607.13089 | [PDF](https://arxiv.org/pdf/2607.13089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aaff19cd-e89f-4398-8dae-a6684a329811`

---

## 69. Designing a GDPR-Compliant Security Architecture for Remote Elderly Care Systems: A Privacy-by-Design Approach

**arXiv ID:** 2607.13122 | [PDF](https://arxiv.org/pdf/2607.13122v1)

**作者:** Md. Rahid Parvez `[一作]` (Metropolia University of Applied Sciences), Mikael Soini `[通讯]` (Metropolia University of Applied Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并验证了Secure Edge Gateway (SEG) 架构，整合了GDPR合规的边缘层伪匿名化、零交互易用性与STRIDE威胁验证，用于远程老人护理的IoMT系统。

**💡 创新点**

创新点包括：在边缘层使用HMAC‑SHA256进行伪匿名化，消除患者身份与健康数据的关联；将零交互易用性作为首要架构约束；以及完整覆盖六类STRIDE威胁的统一验证。

**🔧 技术方法**

技术实现基于ESP32‑WROOM‑32D单板，采用HMAC‑SHA256伪匿名化、AES‑128‑CBC payload加密、TLS 1.3传输安全、MQTT协议、MAC地址白名单及设计科学方法与STRIDE模型、DPIA。

**📊 数据集**

数据集使用仿真生成的合成传感器读数（心率、血氧、体温）以及公开发表的MQTT与HTTP能耗与延迟基准数据。

**📈 对比分析**

通过软件PoC模拟与攻击树分析对架构进行验证；性能对比显示MQTT相较HTTP能耗低6‑8%，边缘处理延迟低于50 ms（可达11.5 ms），而云端延迟为200‑700 ms，证明安全与效率兼得。

**⚠️ 局限性**

局限性：仅为仿真验证，未实测硬件能耗与延迟；使用合成数据，缺乏真实患者测试；评估仅覆盖GDPR框架，未考虑其他地区法规；MAC白名单无法防止物理传感器被盗。

---

## 70. Proceedings of the 21st International Workshop on Termination

**arXiv ID:** 2607.13065 | [PDF](https://arxiv.org/pdf/2607.13065v1)

**作者:** Florian Frohn `[一作]` (RWTH Aachen University), Étienne Payet `[通讯]` (Université de La Réunion)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本报告包含了第21届国际终止研讨会（WST 2026）的会议记录，该研讨会于2026年7月25日在里斯本举行，旨在汇聚对终止问题感兴趣的研究人员。

**💡 创新点**

该研讨会为不同社区之间的思想交叉提供了平台，促进了联合研究和后续出版物的产生。

**🔧 技术方法**

没有具体提到使用的技术，但提到涉及计算机制、编程语言、软件工程和约束求解等领域。

**📊 数据集**

没有具体提到使用的数据集。

**📈 对比分析**

研讨会接受了13篇提交的论文，经过轻审后决定全部接受，最终的会议记录包含9篇论文。

**⚠️ 局限性**

没有具体提到限制因素，但提到的轻审过程可能影响论文的选择标准。

---

## 71. A Fast and Simple $(1+ε)$-Approximation for Minimum Spanning Trees in Doubling Metrics

**arXiv ID:** 2607.13284 | [PDF](https://arxiv.org/pdf/2607.13284v1)

**作者:** Jan Höckendorff `[一作]`, Di Yue `[通讯]` (University of Toronto)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种确定性算法，能在双重维度为 d 的 n 点度量空间中以时间 2^{O(d)}·n·(log n + ε^{-1}·log^4(1/ε)) 计算 (1+ε) 近似最小生成树。

**💡 创新点**

创新之处在于将传统 WSPD 转化为更宽松的 WSPC，并结合多尺度近似策略，将 ε 依赖从 ε^{-O(d)} 降至几乎线性 ε^{-1}；同时证明任何双重维度度量空间都存在 (1+ε) 近似 MST，其最大度可控制在 2^{O(d)}·log(1/ε)。

**🔧 技术方法**

核心技术包括：构造多层网层（net hierarchy）、使用双重维度网点结构、构造 WSPC、基于阈值的近似最近点搜索、层次化分治与误差充能分析、以及对 MST 结构的下界利用。

**📊 数据集**

该工作为理论算法，无特定实验数据集；结果适用于任何满足双重维度约束的度量空间。

**📈 对比分析**

与之前最优的 Arya–Mount 等算法相比，本算法在欧氏空间中将运行时间提升约 1/ε 倍，保持 1+O(ε) 的近似误差，已近似达到目前已知的最优 deterministic 复杂度。

**⚠️ 局限性**

主要局限：仍存在 O(log n) 的时间项，尚未能完全消除；算法对更弱的低维度假设（如仅有限增长度）尚未证明可行；常数因子与高维度的 d 相关，实际性能受限；并未给出实测实验验证。

---

## 72. SoftBoard: A Multi-Agent Tool for the Creation and Evaluation of Low-Fidelity Prototypes

**arXiv ID:** 2607.13179 | [PDF](https://arxiv.org/pdf/2607.13179v1)

**作者:** Gabriel R. S. Scapim `[一作]` (State University of Maringá), Guilherme C. Guerino `[通讯]` (State University of Paraná)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了SoftBoard，一个基于Web的多代理工具，用于在软件创业公司中快速构建和评估低保真原型，支持从需求梳理到Wireflow生成和自动可用性评估。

**💡 创新点**

创新点在于将StartFlow方法与大型语言模型驱动的多代理系统结合，提供全过程的对话式引导、自动Wireflow生成和基于启发式的UX评分，显著降低了对UX专业知识的依赖。

**🔧 技术方法**

技术上采用React+TypeScript前端、Node.js+Express后端、RabbitMQ消息队列、PostgreSQL数据库、Docker化部署，并通过OpenAI GPT-4o/GPT-5.5实现对话、合成、生成和评估代理。

**📊 数据集**

未使用公开数据集，而是利用用户在工具内生成的需求、故事和原型数据进行内部测试；系统在演示视频中使用示例食物配送MVP数据。

**📈 对比分析**

目前缺乏定量对比实验，性能评估以演示视频和可用性预期为主；计划通过即将进行的可行性研究收集用户体验和效率指标。

**⚠️ 局限性**

主要局限包括尚未公开源代码、缺乏真实用户实验验证、对AI模型的可靠性和误差控制不完整，以及工具在跨平台和复杂项目中的可扩展性待进一步评估。

---

## 73. Improving Molecular Property Prediction in Small Language Models Using Graph-based Tools

**arXiv ID:** 2607.13115 | [PDF](https://arxiv.org/pdf/2607.13115v1)

**作者:** Konstantinos Bougiatiotis `[一作]` (National Center for Scientific Research Demokritos), Georgios Paliouras `[通讯]` (National Center for Scientific Research Demokritos)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在小语言模型（SLM）推理时加入图神经网络（GNN）专家的预测置信度、解释子图以及基于子图的自然语言推理，提升分子属性预测的零样本性能。

**💡 创新点**

提出的 Context-Augmented Prompting 框架实现了模块化的工具调用，将结构化图信息转化为文本提示，弥补了仅使用 SMILES 的“结构盲点”，并展示了多工具协同提升的稳健性。

**🔧 技术方法**

核心技术包括：预训练 GNN（用 GNNExplainer 提取关键子图）、三种小语言模型（Llama 3.2、Qwen 2.5、DeepSeek）与工具交互的代理式推理、以及多种提示配置（SMILES、子图、提示、推理、全部上下文）。

**📊 数据集**

使用 MUTAG（188 分子）和 Tox21（随机抽样 1000 分子，二分类）这两个标准分子分类基准数据集。

**📈 对比分析**

与单一 SMILES 提示相比，ALL CONTEXT 方案在 MUTAG 上提升约 34.4%（DeepSeek）/26.8%（Qwen）/0%（Llama）；在 Tox21 上提升约 74.0%（DeepSeek）/44.2%（Qwen）/30.9%（Llama）。尽管如此，SLM 仍未能超过专门训练的 GNN 专家（MUTAG ≈ 84% / Tox21 ≈ 71%）。

**⚠️ 局限性**

局限性包括：1) 仍然落后于专门 GNN；2) 对小规模通用模型（如 Llama 3.2）多工具组合可能导致信息干扰；3) GNN 解释子图的质量和阈值设置会影响性能；4) 仅在两项任务上验证，尚需更多基准和跨领域测试。

---

## 74. Quadratic Probing Revisited: Smoothed Analysis and the Fall of Robin Hood

**arXiv ID:** 2607.13247 | [PDF](https://arxiv.org/pdf/2607.13247v1)

**作者:** Yang Hu `[一作]` (Tsinghua University), Renfei Zhou `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文重新审视了二次探测哈希表的性能，提出了一种平滑变体，其中每个键遵循随机探测序列，探测的期望偏移量为Θ(k^2)。

**💡 创新点**

创新点在于通过平滑分析揭示了反罗宾汉排序和罗宾汉排序的性能差异，反罗宾汉在负载因子1-ε时的查询时间为Θ(logε^-1)，而罗宾汉的查询时间为Θ(ε^-1/2)。

**🔧 技术方法**

使用了平滑二次探测和概率方法，分析了不同排序策略下的查询时间和插入时间。

**📊 数据集**

使用了随机固定偏移的度-d探测序列，适用于任意d≥1的情况。

**📈 对比分析**

与传统的二次探测方法相比，反罗宾汉排序在查询时间和插入时间上表现更优，查询时间为O(logε^-1)，插入时间为O(ε^-1)。

**⚠️ 局限性**

限制在于尽管平滑分析提供了有价值的见解，但对于标准的二次探测哈希表的行为仍然缺乏明确的理论保证。

---

## 75. SteinGate: Tail-Sensitive Safe Reinforcement Learning via Stein Discrepancy

**arXiv ID:** 2607.13175 | [PDF](https://arxiv.org/pdf/2607.13175v1)

**作者:** Yassine Chemingui `[一作]` (Washington State University), Janardhan Rao Doppa `[通讯]` (Washington State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SteinGate——一种利用 Stein 差异对安全 RL 的尾部风险进行分布式认证，并在训练过程中根据认证结果动态切换奖励最大化和安全恢复两种模式

**💡 创新点**

创新点在于：1) 采用 Kernelized Stein Discrepancy（KSD）对政策的成本分布进行非参数一致性检测；2) 针对裁剪成本导致的混合分布（边界原子+连续内部）设计了混合参考分布和拆分的差异度量；3) 在保证尾部风险上界的前提下，用 bang‑bang 控制器替代传统 Lagrangian 软约束，消除梯度竞争与多尺度调参问题

**🔧 技术方法**

主要技术包括：Stein 方法、Kernelized Stein Discrepancy、混合参考分布（点质量+Beta/对数正态内核）、信赖域策略更新、经验分布一致性证书、基于门控的双模式策略优化

**📊 数据集**

在 Safety Gymnasium（PointGoal、PointCircle、CarGoal、CarCircle）和 Safety MuJoCo（Ant、HalfCheetah、Humanoid、Swimmer）等连续控制安全基准上进行实验

**📈 对比分析**

与 CPO、EVO、Saute、Simmer 等基线相比，SteinGate 在尾部可行性上与 EVO 相近（>0.98），但可行回报显著更高（大约 2~4 倍）；在安全恢复模式下的训练更稳定，违规次数和严重性均显著下降

**⚠️ 局限性**

局限性包括：1) 需要预先设定混合参考分布参数，对极端环境可能过于保守或不易设定；2) KSD 估计对样本量敏感，尽管在 8–32 条样本下仍保持安全，但在极小样本或极稀有事件情形下性能可能受限；3) 门控机制在非平稳环境中可能产生切换滞后，需要进一步研究自适应阈值策略

---

## 76. Microflow: Microarchitectural Causal Observability for Deep Cross-Layer Analysis and Optimization

**arXiv ID:** 2607.13184 | [PDF](https://arxiv.org/pdf/2607.13184v1)

**作者:** Saber Ganjisaffar `[一作]` (University of California, Riverside), Nael Abu-Ghazaleh `[通讯]` (University of California, Riverside)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MicroFlow 框架，在周期级模拟器中通过 Microtracer 记录事件流，并将其编译成 MicroFlow Intermediate Representation (MFIR)——一种显式捕获指令流、资源争用与软件语义的有向图，从而实现根因分析、跨层交互可视化以及自动化性能瓶颈定位。

**💡 创新点**

核心创新在于：①将因果关系作为第一类对象直接嵌入模拟器追踪；②使用流 ID 与资源 ID 在事件捕获时就完成关联，消除后处理推断；③构建统一的有向图 MFIR，支持多层级（指令、资源、软件）类型边；④通过声明式查询引擎实现可重复的分析模块（如 TMA、Causal Graph Traversal），实现了从症状识别到根因定位的全流程自动化；⑤揭示了传统聚合指标无法发现的隐藏误预测成本与资源争用放大机制。

**🔧 技术方法**

技术手段包括：基于 gem5 与 ChampSim 的周期级模拟；Microtracer 事件追踪器（异步记录、流与资源标识符）；trace compiler 将事件解码并生成 MFIR；关系/图数据库模式和 SQL/图查询实现可编程分析；Top‑Down Microarchitectural Analysis、Causal Graph Traversal、对比分析与 counterfactual replay；以及对 SPEC CPU 2017 任务的自动化工作流。

**📊 数据集**

使用 SPEC CPU 2017 基准集（如 leela、mcf、gcc、cactuBSSN、namd、lbm、xalancbmk、deepsjeng、leela）并结合 SimPoint 代表性阶段进行实验。

**📈 对比分析**

与现有聚合指标（TMA、PMU）和无追踪模拟器 baseline 进行对比。追踪+MFIR 的模拟器开销在 gem5 上约为 5.9×，在 ChampSim 上为 2.4×；MFIR 编译与分析在单线程下约 7.8× baseline，采用并行编译后约 2.4×。在典型基准上的根因分析平均耗时 8.6 秒。通过 MicroFlow 揭示的根因可实现 IPC 提升 13%–22%（IQ 竞争）、15%（RAS 级联）以及 21%（WP 误预测聚合），累计 IPC 约 21%。

**⚠️ 局限性**

局限性：①依赖周期级模拟器，无法直接应用于已发射的硅平台；②追踪与 MFIR 生成产生显著的空间与时间开销；③当前仅支持 x86‑64 ISA，需扩展软件语义层以覆盖 GPU、内存子系统或 ML 加速器；④分析仍需人工定义查询或插件，尚未完全自动化；⑤大规模工作负载下的存储与查询性能需要进一步优化。

---

## 77. Compaction as Epistemic Failure: How Agentic LLM Tools Fabricate Confirmed Results from Killed Processes

**arXiv ID:** 2607.13071 | [PDF](https://arxiv.org/pdf/2607.13071v1)

**作者:** Hiroki Tamba `[一作]` `[通讯]` (Independent Researcher), Hiroki Tamba (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Claude Code会话压缩机制的实证观察，发现了因时间限制导致的进程被中断后，终端输出被误记录为已确认结果，从而在后续会话和模型版本间传播错误信息。

**💡 创新点**

创新点在于首次系统性识别并分析了LLM代理工具在会话压缩时观察‑持久化混淆所产生的可靠性缺陷，并提出了观察者意识协议及多层验证机制的改进思路。

**🔧 技术方法**

使用了Claude Code与ChatGPT Codex的会话日志分析、进程退出码跟踪、脚本执行与JSON文件的状态检查等技术手段。

**📊 数据集**

研究依赖于作者在2026年7月的Claude Code 4.7/4.6版本会话记录，以及对同一时间段内的四个相似错误案例的采集，没有使用公开的标准数据集。

**📈 对比分析**

通过对比Codex的文件优先验证方法和Claude Code的压缩摘要方式，发现后者在长会话条件下会产生错误，尽管未给出量化指标，但示例表明错误传播速度快、修复成本高。

**⚠️ 局限性**

局限性包括仅在单一用户、单一平台和特定脚本场景下观察到，缺乏大规模实验验证，且Codex在高上下文负载下的表现尚未检验。

---

## 78. AI in Cyberpsychology: A systematic literature review of Cybersecurity enhancement by using AI for analyzing psychology of Victims, Attackers, and Defenders

**arXiv ID:** 2607.13123 | [PDF](https://arxiv.org/pdf/2607.13123v1)

**作者:** Georg Thamer Francis `[一作]` (Istanbul Medipol University), Selim Akyokuş `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文通过PRISMA方法，对2010-2025年间关于人工智能在网络心理学（AI‑CPSY）应用的34篇研究进行系统性文献综述，构建了分类体系并识别研究空白与挑战。

**💡 创新点**

创新点在于首次系统地将AI技术与心理学框架融合的研究进行梳理，提出了四大网络安全应用（异常检测、漏洞风险预测、安全意识培训、身份验证）与AI/心理学方法的完整分类，并揭示了该领域的关键研究缺口。

**🔧 技术方法**

采用的技术主要是系统性综述方法（PRISMA）、定量统计与文本挖掘，对每篇研究进行方法、算法（ML、NLP、DL、RL）与心理学框架（OCEAN、C6PoP等）的归类。

**📊 数据集**

使用了34个数据集，其中23个为研究者自制，10个为基准集，1个为新集；数据集涵盖网络安全、网络心理学与传统心理学领域。

**📈 对比分析**

本文未进行统一实验对比，而是对各研究报告的准确率、召回率、F1等指标进行汇总与趋势分析，发现如SFP预测模型在ML下平均精度可达80–97%，NLP用于社工检测的准确率多在90%以上，但整体受限于样本规模与模型偏差。

**⚠️ 局限性**

主要局限包括：大多数数据集为自制且规模偏小，缺乏统一基准导致可比性差；DL与RL在AI‑CPSY中应用不足，AIV领域几乎未被探索；部分研究过度拟合或缺乏真实场景验证，限制了方法的泛化与实用性。

---

## 79. Continuously Evolving Deepfake Detection: An Architecture and Public-Benchmark Evaluation of a Dynamic Detection System

**arXiv ID:** 2607.13234 | [PDF](https://arxiv.org/pdf/2607.13234v1)

**作者:** Ken Jon Miyachi `[一作]` (BitMind), Dylan Uys `[通讯]` (BitMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了一套基于持续对抗奖励机制的深度伪造检测系统 BitMind Forensics（BMF），并评估了其单一版本在多种公开基准上的性能。

**💡 创新点**

核心创新在于：① 用开放式、经济激励驱动的生成-判别竞赛（Bittensor Subnet 34）持续刷新训练数据和模型结构；② 通过异构多分支神经集成（ConvNeXt‑L、EVA‑L、CLIP ViT‑L、DINOv3）降低单一模型对特定伪造手段的过拟合；③ 在单一无微调导出的模型上实现了跨数据集、跨生成器、跨后处理攻击的高泛化性能。

**🔧 技术方法**

技术手段包括：a) 在线对抗训练与奖励积分系统；b) 多分支融合与温和激活的 cosine‑和线性‑head 组合；c) 对图像、通用视频与人脸视频的三种任务拆分；d) 采用 1.11B‑参数图像集成、约 1.12B‑参数通用视频模型和 1.45B‑参数人脸视频专家；e) 公开的 GAS‑Station 对抗数据集与 gasbench 评估框架。

**📊 数据集**

使用的数据集涵盖：① 面部伪造交叉验证集合（FaceForensics++, UADFV, DFD, DFDC, DF40, Celeb‑DF v1/v2/++）；② 真实世界鲁棒性评测 Sumsub 与 WildRF；③ AI‑生成图像评测（Community Forensics, AIGCDetectBench, GenImage, AIGIBench, AI‑GenBench）；④ AI‑生成视频评测（GenVidBench, GenVideo‑100K）；⑤ 对抗鲁棒性评测 RAID；⑥ GAS‑Station 提供的 50,399 张生成图像和相应视频。

**📈 对比分析**

比较方法：对所有基准采用统一的评估协议（无额外微调、统一阈值 0.5），直接将 BMF 的 ROC‑AUC、平衡准确率、TPR@1% FPR、EER 等与公开论文和官方基准报告的最佳结果进行对比。结果显示：BMF 在 Sumsub 原始图像上 AUC 0.936，整体 0.872；在 Deepfake‑Eval‑2024 图像轨道上 0.915，视频轨道 0.822；在 AI‑生成图像上 0.991（21 生成器），在 GenVidBench 视频上 0.918；在人脸视频交叉验证上超越 FF++ 专家（DFDC 0.947、Celeb‑DF v2 0.9985），与 Celeb‑DF++ 达到统计相等；在对抗攻击 RAID 上保持 0.821 (ε=16)；在时间演化实验中，随时间递增的 BMF 导出在未见生成器的测试集上从 0.842→0.902（图像）和 0.864→0.936（视频）提升。

**⚠️ 局限性**

局限性：① 仅评估了单一导出版本，未覆盖实时系统的持续迭代；② 图像和通用视频使用全帧无人脸裁剪，可能在需裁剪的基准上受限；③ 未评估音频深度伪造；④ 评测仅覆盖公开许可的数据集，部分硬核社区图像未包含；⑤ 对人脸视频基准做了训练近似重复排查，但仍可能存在身份重叠；⑥ 评测数据主要聚焦面部与静态图像，未覆盖多模态或非面部场景的真实流量。

---

## 80. A Hybrid Mamba for Audio-Visual Navigation

**arXiv ID:** 2607.13110 | [PDF](https://arxiv.org/pdf/2607.13110v1)

**作者:** Yi Wang `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 4190 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种基于Mamba状态空间模型的混合架构Samba，用于音频-视觉导航；

**💡 创新点**

将Mamba的选择性扫描机制引入状态编码与音频编码，解决了GRU的状态稀释与CNN对时间频域特征的局部偏置问题；

**🔧 技术方法**

使用Mamba State Encoder (M-SE) 取代GRU、双向 Audio Mamba Encoder (AME) 取代卷积音频编码，并结合PPO策略网络；

**📊 数据集**

在SoundSpaces仿真平台下使用Replica和Matterport3D两大室内数据集，评估模型在听见/未听见音频源和已见/未见场景上的性能；

**📈 对比分析**

与现有基准（AV‑WaN等）对比，Samba在SR、SPL、SNA等指标上提升约10–20%（SR最高达95%），且参数量降低约30%；

**⚠️ 局限性**

实验仅在仿真环境中进行，缺乏真实世界验证；对极长序列的鲁棒性及跨域部署的通用性仍待进一步探究。

---

## 81. AffectFlow-DINO: Uncertainty-Aware Multi-Task Affect Estimation via Conditional Rectified Flow

**arXiv ID:** 2607.13250 | [PDF](https://arxiv.org/pdf/2607.13250v1)

**作者:** Salah Eddine Bekhouche `[一作]` (University of Basque Country), Abdenour Hadid `[通讯]` (Universiti Malaysia Kelantan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个多任务框架AffectFlow-DINO，用于在ABAW 2026赛题中同时预测连续的情绪维度（valence-arousal）、八类表情和十二个动作单元。

**💡 创新点**

核心创新在于引入条件rectified flow头，使模型学习条件分布p(y|x)，从而实现不确定性感知的一对多预测，并通过后置阈值校准解决严重类别不平衡。

**🔧 技术方法**

采用冻结的DINOv3 ViT-S/16视觉编码器、共享投影层、三头回归/分类/多标签头，以及基于时间嵌入的条件rectified flow网络和蒙特卡罗采样；训练中结合掩码损失、流损失权重以及低学习率微调。

**📊 数据集**

使用s‑Aff‑Wild2数据集（包含VA、表情和AU标签），并在其官方验证集上进行实验。

**📈 对比分析**

与官方基线P_MTL=0.45相比，结合背骨微调、流重调和阈值校准后，最终得到P_MTL=1.177，显著提升；在各子任务上也分别取得最高的CCC、宏F1和AU F1。

**⚠️ 局限性**

局限性包括：仅基于单帧，未利用时序信息；对极少数类别的表现仍受限，需更精细的采样或正则化；流头在背骨微调后需重新训练，表明模型对参数敏感。

---

## 82. GPUSimBench: Towards Scalable and Reliable GPU-Accelerated Simulators in Embodied AI

**arXiv ID:** 2607.13059 | [PDF](https://arxiv.org/pdf/2607.13059v1)

**作者:** Huzhenyu Zhang `[一作]` (Shanghai AI Laboratory), Dmitry Yudin `[通讯]` (MIRAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了GPUSimBench benchmark，用来系统评估主流GPU加速的并行机器人仿真器在并行扩展性、物理一致性和确定性等关键指标上的表现。

**💡 创新点**

创新点在于：①首次针对GPU并行仿真器设计分布级物理一致性评估和可量化的确定性度量（并行变异与跑间变异）；②发现并归纳了四个随机性经验区域；③基于这些度量给出实用的仿真器选择指南。

**🔧 技术方法**

技术实现包括：采用倾斜平面撞击实验、自由落体立方体堆叠实验以及Franka Panda随机控制实验；使用Earth Mover's Distance（EMD）衡量分布一致性；记录FPS、GPU内存占用；计算并行与跑间的一致性指标。

**📊 数据集**

数据集主要来自自建的物理实验平台，采集倾斜碰撞实验中立方体最终位置数据；模拟实验生成相应的分布数据，两者对比形成真实‑模拟一致性评估。

**📈 对比分析**

比较方法：对七大主流仿真器（Isaac Lab、ManiSkill、Genesis、Madrona、MuJoCo Warp、MJX、Playground）在同一硬件环境下进行并行FPS、内存、EMD、并行变异与跑间变异的多维度测评。结果显示：Genesis与Madrona在最大并行环境数与吞吐量上遥遥领先；Isaac Lab在吞吐量、内存占用与一致性之间取得平衡；MuJoCo Warp在低并行度下内存占用最小；物理一致性上MJX和ManiSkill表现最佳，而Madrona的误差最大。

**⚠️ 局限性**

限制：仅覆盖两类并行度任务与一项接触丰富实验，未涉及变形、流体或感知渲染等；所有测评在单一硬件/软件栈下完成，结果可能随GPU型号、驱动或仿真器版本变化；未覆盖任务随机化与策略随机性影响。

---

## 83. Probabilistic Extension of Neuro-Symbolic AGI Robots based on Belnap's Typed Intensional FOL

**arXiv ID:** 2607.13073 | [PDF](https://arxiv.org/pdf/2607.13073v1)

**作者:** Zoran Majkic `[一作]` `[通讯]` (ISRST), Zoran Majkic (ISRST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出基于IFOL_B的神经符号自我意识机器人框架，整合4值本体逻辑与Nilsson概率结构以实现不确定知识的概率推理和自我反思；

**💡 创新点**

将自我意识和概率推理统一在IFOL_B中，利用本体抽象和信息熵最大化实现最小偏差的概率分布；

**🔧 技术方法**

使用Intensional First‑Order Logic (IFOL_B)、Belnap四值双 lattice、Nilsson概率空间、最大熵神经网络和自动演绎推理；

**📊 数据集**

未使用公开数据集，主要以理论推导与框架设计为主；

**📈 对比分析**

与SOAR、LNN、DeepProbLog、DILP、LLM‑SS等现有框架对比，主张在安全、可解释性和自我意识方面更优，但缺乏实验性能指标；

**⚠️ 局限性**

缺乏实证实验、实现细节不明、计算开销高、对大规模知识库的可扩展性待验证

---

## 84. BARS: Benign-Anchored Ranking and Selection for False Alarm Reduction in Network Intrusion Detection

**arXiv ID:** 2607.13203 | [PDF](https://arxiv.org/pdf/2607.13203v1)

**作者:** Abu Fuad Ahmad `[一作]` (New Mexico State University), Istiaque Ahmed `[通讯]` (Osaka Metropolitan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 BARS 的两阶段特征筛选方法，用于降低网络入侵检测系统的误报率。

**💡 创新点**

创新点在于将特征评分的锚点从全局均值改为正常流均值，并引入顺序保持的去相关化步骤，解决了传统 CMD 在类别不平衡时的锚点偏移问题。

**🔧 技术方法**

技术细节包括：① 计算攻击类别均值相对正常流的绝对偏差作为评分；② 通过设定相关性阈值对已排序特征进行递归去相关化，保持特征互补性。

**📊 数据集**

实验使用了 CICIDS2017、CICDDoS2019、UNSW‑NB15 三个 NIDS 基准数据集，覆盖从正常流占多数到攻击流占多数的不同不平衡场景。

**📈 对比分析**

与 CMD、MI、Pearson、Fisher、SMOTE 以及阈值调优等基线进行对比，结果显示在攻击占多数且特征预算有限的情况下，BARS 在 k=20 时将 FPR 降低 15–23% 并保持甚至提升 TPR 与 Macro‑F1，优于传统全局锚定方法。

**⚠️ 局限性**

局限性包括：仅考虑非自适应攻击，对高阶分布差异捕捉有限，统计显著性检验受样本量限制，且对概念漂移与动态环境尚未进行评估。

---

## 85. Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs

**arXiv ID:** 2607.13080 | [PDF](https://arxiv.org/pdf/2607.13080v1)

**作者:** Sheng-Wei Peng `[一作]` (Pegatron Corporation), Yi-Pei Lee `[通讯]` (Pegatron Corporation)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文在同一开发者、同一企业代码库的两连28天实验中，对比了基于Claude Opus云端API的Claude Code与基于NVFP4量化GLM-5.1/5.2的本地Opencode部署，系统评估了推理成本、代码缺陷修复负担、总体拥有成本（TCO）与开发者体验；通过对LLM调用日志与Git提交历史的双重挖掘，量化了提示缓存对API成本的折扣、各部署模式下的缺陷修复率、修复工作量及其对提交节奏的影响，并在此基础上构建了成本-质量前沿分析与混合路由策略的离线仿真。

**💡 创新点**

创新点包括①首次在企业真实生产环境下进行长期单主体案例研究，结合LLM遥测与Git挖掘实现成本与质量双重度量；②揭示提示缓存可将云端API的有效单元成本压至低于本地GPU部署；③量化本地量化模型在缺陷修复方面的显著劣势，并以“真实TCO”与“开发者体验”指标为基础，提出成本-质量前沿与混合路由的决策框架；④提供可复现的关键词规则与统计检验流程，为后续跨项目复制提供模板。

**🔧 技术方法**

使用技术包括：Claude Code终端代理（Anthropic）与Opencode开源框架（Vercel AI SDK、Bun、Go），Claude Opus 4.7/4.8云端API与NVFP4量化GLM-5.1/5.2本地模型，NVIDIA Blackwell GB200 NVL72 GPU集群，vLLM服务框架（PagedAttention、自动前缀缓存），Langfuse遥测系统，Prometheus计数器，Git日志解析，Python/TypeScript代码统计，统计检验（Mann‑Whitney U、Cliff’s Δ、Mantel‑Haenszel OR、Bootstrap），以及成本建模（TCO、GPU租赁、电力、人工成本）。

**📊 数据集**

数据集为一家AI PaaS平台的生产单体仓库（12个包，含Python、TypeScript、Markdown等），共计约56天的Git提交历史（含合并、重写记录）与两段期间的LLM遥测日志（Claude Opus API Telemetry via Langfuse、GLM vLLM Prometheus计数）。

**📈 对比分析**

比较方法：在同一开发者、相同代码库、相同SDLC条件下，分别记录两段28天期间的（1）LLM token使用与缓存命中率；（2）Git提交数量、代码变更量、缺陷修复提交比例与缺陷类型；（3）基于官方定价与内部GPU租赁成本的TCO；（4）基于提交时间戳的客观工作负载指标。实验结果显示：①提示缓存将Claude API的有效单元成本降至$0.57/百万token；②本地GLM的缺陷修复率显著高于API（74.9% vs 45.9%，OR≈3.6）；③在共享GPU分配下，TCO比云端低约40%；④在开发者体验方面，修复工作占比高、提交节奏拖慢、调试螺旋比例上升。

**⚠️ 局限性**

限制包括：单一开发者与单一仓库的非随机化、时序效应（A后B顺序可能带来熟悉度变化）；模型与量化效应混淆（本地量化模型与模型能力差异难以分离）；缺乏多语言/多任务/多开发者的普适性检验；仅测算缺陷修复的commit级别时间，未细化实际手工修复耗时；假设不同难度层的缺陷率可跨部署迁移；成本模型假设共享GPU使用均等且未考虑冷启动/容器化开销。

---

## 86. Beyond Backbone Backpropagation: A Decoupled Strategy for Efficient Transfer Learning

**arXiv ID:** 2607.13043 | [PDF](https://arxiv.org/pdf/2607.13043v1)

**作者:** Daniel Vila-Cruz `[一作]` (Universidade da Coruña), Verónica Bolón-Canedo `[通讯]` (Universidade da Coruña)

**通讯引用:** 7983 | [OpenAlex ID](https://openalex.org/A5042436168)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一套轻量级迁移学习框架，先在预训练模型上仅更新归一化层实现域适配，然后冻结特征提取器，预先提取特征后仅训练一个改进的分类头，以大幅降低训练成本并保持或提升医学图像分类精度。

**💡 创新点**

创新点在于将特征提取与分类器分离，利用仅归一化层的微调实现域迁移，重新设计了带BN+ReLU的两层分类头，并加入基于分类边际的加权损失，全部在不对背骨进行反向传播的前提下实现高效训练。

**🔧 技术方法**

技术上采用批归一化/层归一化自适应、梯度无关的LN偏置调整、margin‑based weighted cross‑entropy、特征预提取、以及轻量级分类头的网络结构。

**📊 数据集**

使用了三组医学影像数据集：Brain Cancer MRI（脑肿瘤3类），BreakHis（乳腺肿瘤恶性/良性）和PatchCamelyon（皮肤切片肿瘤2类）。

**📈 对比分析**

与传统全微调（CNN）或LoRA（Transformer）基线对比，实验显示在三类数据集上平均准确率保持相近甚至略优，同时训练时间提升约20–50×，CO2排放减少至原来的1/20左右，尤其在CPU环境下可比GPU传统训练更快。

**⚠️ 局限性**

局限性包括：对某些已预训练为蒸馏模型（如 DeiT）的适配效果不佳；归一化层微调对不同网络结构的敏感度尚需进一步研究；margin‑based weighting采用固定超参数，可能在不同数据分布下表现不稳定。

---

## 87. HEDGEHOG: Hierarchical Evaluation of Drug Generators Through Rigorous Filtration

**arXiv ID:** 2607.13155 | [PDF](https://arxiv.org/pdf/2607.13155v1)

**作者:** Daria A. Ryabchenko `[一作]` (Ligand Pro), Marina A. Pak `[通讯]` (Ligand Pro)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HEDGEHOG，一套包含预处理、物理化学筛选、结构警戒、合成可行性、对接与结合能预测以及3D姿态检查的六阶段过滤基准，用以端到端评估分子生成模型在实际早期药物发现流程中的可行性。

**💡 创新点**

创新点在于将多项传统药物设计过滤器整合为统一的闭环工作流，提供从化学可行性到3D结合姿态的完整评估；通过统计各阶段存活率揭示生成模型在真实药物化学约束下的薄弱环节。

**🔧 技术方法**

使用RDKit和DataMol进行分子预处理；计算21种物理化学描述符并应用约2,460条SMARTS警戒；采用SA/RA/SYBA合成评分与AiZynthFinder回路搜索；对接使用smina、GNINA、Matcha三种引擎，结合Boltz‑2结合能预测；3D检查结合ProLIF进行姿态与相互作用评估。

**📊 数据集**

以KRAS G12D Switch‑II pocket（PDB 7ew9）为靶点，生成并评估23种模型，共计230,000个分子（每个模型10,000个有效SMILES）。

**📈 对比分析**

按六阶段的生存率对模型类进行比较，最终仅有约0.6% 的分子通过全部筛选；Dragonfly在该实例获得最高终端生存（345分子），REINVENT4 (V)其次，显示多阶段评估对模型区分度的显著提升。

**⚠️ 局限性**

局限性包括仅针对单一靶点KRAS G12D，未覆盖更广泛的ADMET或临床指标；缺乏对生成种子方差的全面估计；对其他靶点的适配性和泛化能力尚未验证。

---

## 88. Baselines Before Architecture: Evaluating Coding Agents for Autonomous Penetration Testing

**arXiv ID:** 2607.13085 | [PDF](https://arxiv.org/pdf/2607.13085v1)

**作者:** Ananda Dhakal `[一作]` (Kroda Labs), Aarjan Chaudhary `[通讯]` (Kroda Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在104题的Web渗透测试基准上，对三种默认的编码CLI代理（如Kite、Replit、Copilot）在相同GPT‑5模型、预算和评分规则下进行两轮完整跑，评估其在没有任何安全专用架构的情况下的基准性能；随后将最佳纯编码代理与已有的多模安全框架（如PentestGPT‑V2和PentestGPT‑V3）在模型匹配的条件下做对比，量化安全架构的“残差”贡献；最后在同一框架下替换为更新的GPT‑5.2/5.5模型，观察模型升级对性能的提升。

**💡 创新点**

①提出了“架构归因”方法，清晰分离模型能力、通用编码代理与安全特定架构三项因素；②用纯编码代理做基准，检验安全框架是否真正带来增益；③在同一基准框架下通过模型升级展示模型进步可显著弥补或超越安全架构；④发布完整可复现的GitHub artifact，为后续研究提供标准化对照。

**🔧 技术方法**

使用大语言模型GPT‑5系列作为后端；采用公开的三款CLI编码助手（Kite、Replit、Copilot）作为实验对象；构建了统一的评估harness，记录命令、输出、token、成本、时间等指标；使用Python脚本对传递的flag进行哈希匹配，实现自动化得分。

**📊 数据集**

104题XBOW基准：104个容器化Web应用，涵盖26类OWASP漏洞，采用随机植入的隐藏flag作为唯一判定标准。

**📈 对比分析**

对比方法：先在模型相同、预算相同的条件下比较三款CLI代理，选取pass@1和pass@2指标；随后将选定的最佳纯编码代理与已发表的安全框架在模型匹配（GPT‑5 vs GPT‑5, GPT‑5.2 vs PentestGPT‑V2）下对比，计算残差；再将同一框架替换为GPT‑5.2/5.5，观察pass@1/2、成本、token等变化。实验结果显示：纯编码代理在两轮跑中可达77.9%（pass@2）覆盖率，安全架构在同一模型下仅贡献约9–15%残差；但更新模型后，纯代理可达到92.3%（pass@1）和95.2%（pass@2），超过部分已发布的安全系统。

**⚠️ 局限性**

实验仅在公开的CTF式基准上进行，未在真实网络环境或多机系统中验证；未在本地重新跑已发布的安全框架，导致跨论文对比受限；模型升级与预算上限同时变化，难以完全分离；实验次数仅为两轮，未覆盖足够多的随机性；修补了约40个容器镜像以保证运行，可能与原始基准略有差异。

---

## 89. A 3DGS-Driven Dynamic Viewpoint and Vibrotactile Framework for Subsea Teleoperation Validated via fNIRS

**arXiv ID:** 2607.13067 | [PDF](https://arxiv.org/pdf/2607.13067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 90. Mixed-Timescale Differential Coding for Downlink Model Broadcast in Wireless Federated Learning

**arXiv ID:** 2607.13119 | [PDF](https://arxiv.org/pdf/2607.13119v1)

**作者:** Chung-Hsuan Hu `[一作]` (Linköping University), Erik G. Larsson `[通讯]` (Linköping University)

**通讯引用:** 51495 | [OpenAlex ID](https://openalex.org/A5043552696)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种混合时尺度差分编码（MTDC）方案，用于无线联邦学习系统中下行全局模型的广播，并进一步引入了年龄感知版本（A-MTDC）及对应的设备调度策略；

**💡 创新点**

创新点在于将差分编码分为两个层次，并通过动态选择参考模型来构造层级化的编码；该方法在遭遇下行解码失败时能够通过更高层次的差分或全模型广播快速同步，显著提升通信鲁棒性；

**🔧 技术方法**

主要技术包括差分编码、随机量化、年龄感知的MTDC决策、设备年龄权重的调度策略，以及对FedAvg收敛性的理论分析；

**📊 数据集**

实验使用了常见的MNIST和CIFAR-10数据集，采用非IID数据划分的20台设备；

**📈 对比分析**

与传统的全模型广播、单层差分编码及随机调度等方法相比，MTDC/A-MTDC在相同通信资源预算下实现了更高的测试准确率，且在解码失败较多时表现更为稳健；

**⚠️ 局限性**

局限性包括需要额外的本地存储来保存参考模型、量化采用简单的标量量化、仅针对同步联邦学习场景，未覆盖去中心化或更复杂的解码失败模型。

---

## 91. A Unified Framework for Reaction Systems Based on Interval Structures

**arXiv ID:** 2607.13097 | [PDF](https://arxiv.org/pdf/2607.13097v1)

**作者:** Paolo Bottoni `[一作]` (Sapienza University of Rome), Ion Petre `[通讯]` (University of Turku)

**通讯引用:** 1975 | [OpenAlex ID](https://openalex.org/A5047091186)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于区间结构的统一语义框架，用来描述和比较各类反应系统（如经典、有限、离散浓度、多集、资源保持等）以及Petri网等相关模型。

**💡 创新点**

创新点在于将执行语义拆解为资源、生产、更新与执行四个独立策略，并通过区间结构统一表达启用条件与产出集合，从而让不同模型的差异仅体现在策略的组合上，显著提升了语义的一致性与可扩展性。

**🔧 技术方法**

核心技术包括：区间结构与区间变换、区间变换系统、资源/生产/更新策略、可执行块的可行性与最大化判定、执行策略（顺序、并发、最大并发）以及可选的预处理与过滤策略。

**📊 数据集**

本文未使用任何实验数据集，主要通过理论构造与形式化证明展示框架对现有模型的重编码与语义恢复。

**📈 对比分析**

比较方法为对每个已知反应系统变体逐一给出其在框架中的编码与对应策略组合，并证明执行步骤等价；因未进行实验，无法给出性能指标，主要通过形式化证明展示兼容性与一致性。

**⚠️ 局限性**

限制包括：对新型或非传统模型的支持仍需进一步研究；框架主要聚焦语义统一，未涉及可达性、等价性或复杂度等计算性分析；对于大规模系统的实现与效率仍未评估。

---

## 92. Faithful Autoformalization of Natural Language Assertions

**arXiv ID:** 2607.13303 | [PDF](https://arxiv.org/pdf/2607.13303v1)

**作者:** Hongyi Liu `[一作]` (University of Wisconsin-Madison), Adithya Murali `[通讯]` (University of Wisconsin-Madison)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动化将自然语言断言转换为可执行 JML 断言的完整框架，结合 LLM 生成多条候选、代码测试验证和语义一致性评分，最终输出最可信的断言，并支持主动学习进行歧义消解。

**💡 创新点**

创新点包括：①引入双向 clausal coverage（子句级覆盖）作为候选与原始自然语言意图的语义对应度量；②将执行层有效性检测与语义一致性评分结合的决策矩阵；③在有效/无效两类候选之间通过主动学习生成区分输入并询问用户，以实现歧义消解；④实现完整的多模型候选生成、过滤与验证流程。

**🔧 技术方法**

核心技术包括：LLM（如 GPT-oss、Qwen2.5-Coder 等）生成 JML 断言；Randoop 等测试生成器进行语义有效性检查；LLM‑as‑a‑Judge 进行 clausal coverage 评分；阈值过滤与决策矩阵选择最可信候选；主动学习循环生成区分输入并与用户/oracle 对话。

**📊 数据集**

使用的数据集：C2S‑增强版（416 对）合成 NL 与 JML；Buggy Code（20 对）与 Buggy Assertion（66 对）两种错误场景；Manual NL Specs（39 对）人工编写的自然语言断言；实验中还评估了多种 LLM backbone（GPT‑oss、Qwen2.5、GPT‑5.5、Claude‑Opus 等）。

**📈 对比分析**

方法与基线的比较：对比单次 LLM 输出（Oneshot）以及仅用测试过滤的 baseline，评估指标包括 Any‑Equiv、Oneshot、-Acc、-Prec、-Rec。实验结果显示：在主数据集上，-Acc 约 89.7%（GPT‑oss）/75.7%（Qwen2.5）；-Prec 分别为 92.8%/91.6%；Recall 97.6%/93.2%；显著提升精度并保持高召回；在人工 NL 上仍能达到 84.6%/59% 的准确率。不同 backbone 的差异主要体现在 Any‑Equiv 上，强大模型更易生成等价候选。

**⚠️ 局限性**

局限性包括：①主动学习仅在极少数案例触发，实际用户交互成本未知；②对强大闭源 LLM 的依赖较大，开源模型效果相对受限；③clausal coverage 的准确性受 LLM 生成文本质量影响；④仅聚焦模块级断言，未覆盖更高层次需求；⑤在 Buggy Assertion 场景下性能显著下降，说明对错误或误导性 NL 断言的处理仍需改进。

---

## 93. Networked Intelligence: Active Shared Context Graphs for Human-AI Team Science

**arXiv ID:** 2607.13220 | [PDF](https://arxiv.org/pdf/2607.13220v1)

**作者:** Sutanay Choudhury `[一作]` (Pacific Northwest National Laboratory), Robert Rallo `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个名为 Mycelium 的实时网络化人机协同科学发现架构，并在 Pseudomonas putida 的四模态多组学实验中进行了实证测试，证明该系统能够自动路由关键发现，促成跨学科团队的共识模型和实验设计。

**💡 创新点**

创新点在于：①提出活跃上下文图（ACG）与持续推理的运行时架构，实现知识共享、持续路由和可追溯的推理链；②实现网络化智能，通过跨人机、工具、仪器的异步共享与路由，突破单一模型无法跨域协同的瓶颈；③给出了稀疏条件计算的理论框架，阐明何时网络是不可或缺的。

**🔧 技术方法**

核心技术包括：活跃上下文图（ACG）与其边缘语义；Model Context Protocol 与 Agent‑to‑Agent Protocol 的通信协议；Python 沙箱内的动态代码生成与自动错误恢复；基于 Claude Opus 4.8 的大语言模型进行交互与推理；以及持续的状态路由、进程发现与 provenance‑bounded propagation。

**📊 数据集**

使用的实验数据集是 Pseudomonas putida 的四种组学数据：蛋白质组（约4,495蛋白）、代谢网络图（1,170节点/1,007条边）、HPLC 代谢物定量（48测量、4种分析物）和媒体优化实验设计空间（75 条测量+28 条建议运行），共四种模态，涵盖 4 种菌株、不同碳源和时间点。

**📈 对比分析**

通过与两种单体代理基线（B 与 C）进行对比，采用 26 个可追溯的科学工件指标（从 0 到 4 分）评估覆盖率与特异性。Mycelium 在 26 个工件中的覆盖率得分为 2.62，明显高于基线 B（1.62）和 C（1.81），同时保持相近的特异性（Mycelium 2.72 vs. 2.47/2.61），证明网络化架构在开放式科学探索中显著提升了发现的广度而不损失深度。

**⚠️ 局限性**

局限性包括：①需在多实验室、多周期的长期项目中验证可靠性；②对冲突信息的处理机制尚不完善；③跨学科翻译时可能产生语义偏移，需要更精细的归因与不确定度评估；④主动性与归因权重的调参仍是设计难点。

---

## 94. Operational Evidence Gaps for LLMs in Fraud Detection and Trust-and-Safety Workflows

**arXiv ID:** 2607.13078 | [PDF](https://arxiv.org/pdf/2607.13078v1)

**作者:** Keyur Gabani `[一作]` `[通讯]`, Keyur Gabani

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对LLM在欺诈检测、调查支持、内容审核等信任与安全工作中的部署角色进行系统性综述，提出FORTE角色与证据框架，并评估现有研究的部署证据不足。

**💡 创新点**

创新点在于：①以部署角色为核心构建FORTE框架，划分四种LLM放置模式；②制定最小部署证据清单（延迟、成本、阈值、解释完整性、对抗压力）；③揭示欺诈领域缺乏实时延迟、成本与校准等关键证据，提出未来研究议程。

**🔧 技术方法**

采用结构化叙事综述方法，编码49篇与LLM相关的论文，构建证据矩阵；对比不同角色在延迟、成本、治理、公平性等维度的部署证据。

**📊 数据集**

主要引用文献使用的金融欺诈数据集、文本诈骗数据集、内容审核数据集等；本研究自身未收集或使用具体数据集。

**📈 对比分析**

通过定性比较与量化指标分析不同角色的部署证据，发现欺诈相关研究缺少实时延迟和成本指标，而内容审核研究提供更完整的治理与公平性证据；整体性能评估基于文献报告，未进行统一实验对比。

**⚠️ 局限性**

局限性包括：仅覆盖2023–2026年文献，方法为结构化叙事而非系统性检索；未纳入所有生产部署案例；缺乏统一的量化性能对比；结论主要基于文献评估，缺乏实验验证。

---

## 95. Design-System-Aware Development with AI: Evaluating Productivity and Design Consistency

**arXiv ID:** 2607.13156 | [PDF](https://arxiv.org/pdf/2607.13156v1)

**作者:** Luciane Silva `[一作]` (CI&T and Universidade Federal de Uberlândia), Gustavo Pinto `[通讯]` (Universidade Federal do Pará)

**通讯引用:** 3563 | [OpenAlex ID](https://openalex.org/A5016508154)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在一家大型巴西企业中，对 49 名前端和移动开发者进行对照实验，比较传统手工开发、仅使用设计系统（DS）开发以及结合企业 DS 的 AI 辅助开发，评估实现高保真 mockup 所需时间、任务完成度与性能波动。

**💡 创新点**

首次在工业环境下系统评估了“设计系统感知 AI”对开发效率与一致性的影响，揭示 AI 在 Angular、iOS、Android 三栈中均可显著加速交付、提升视觉完整性并降低极端差异；同时通过分段提交与休息模式分析工作流摩擦，为 AI 与 DS 的结合提供实证支持。

**🔧 技术方法**

使用了内部构建的 StackSpot AI（基于大语言模型，嵌入企业设计系统上下文）配合 Angular、iOS（Swift/SwiftUI）和 Android（Kotlin/Java）等前端技术；实验数据通过实时递交与专家评审收集。

**📊 数据集**

实验样本为 49 名来自 Zup Innovation 的开发者，分两轮（2025 年 6 月与 10 月）完成两张高保真 mockup 的实现；并未使用公开数据集，所有任务均由企业内部提供的 mockup 与 DS 文档驱动。

**📈 对比分析**

采用被试间设计，衡量指标包括：总交付时间、任务完成率（视觉一致性）以及标准差。结果显示：AI 辅助相比手工可降低 46.7%–69.4% 的交付时间，完成度从 68% 提升至 96%，标准差明显下降，工作流连续性更好。

**⚠️ 局限性**

局限性包括：样本来自单一企业，实验任务仅为两张屏幕实现，未覆盖完整的需求迭代与后期维护；评估依赖专家人工判断，存在主观性；AI 工具高度定制化，缺乏对通用 AI 辅助的可迁移性验证。

---

## 96. A Masked Autoencoder Approach to Unsupervised Steel Surface Defect Recognition

**arXiv ID:** 2607.13178 | [PDF](https://arxiv.org/pdf/2607.13178v1)

**作者:** Shrey Patel `[一作]` `[通讯]` (University of Maryland), Shrey Patel (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用基于Transformer的Masked Autoencoder（MAE）对钢表面缺陷图像进行无监督预训练，并利用预训练的编码器特征结合UMAP降维和层次聚类，实现对六类缺陷的高精度分组。

**💡 创新点**

创新点在于：① 将MAE与辅助缺陷定位任务联合训练，使编码器在无标签情况下学习到更具缺陷语义的表示；② 通过Grad CAM和Grad CAM++验证编码器自动聚焦缺陷区域；③ 通过对比预训练与随机初始化编码器、原始像素三种基线，明确表明预训练显著提升聚类性能。

**🔧 技术方法**

核心技术包括：Vision Transformer架构、Masking策略（75%掩码）、轻量化解码器、辅助检测分支、AdamW优化器、Cosine学习率调度、Grad CAM/Grad CAM++可视化、UMAP降维、Ward链接的凝聚层次聚类以及Hungarian算法进行聚类标签对齐。

**📊 数据集**

使用公开的NEU Surface Defect Dataset（NEU-DET），包含1800幅灰度热轧钢表面图像，均匀分布在六类缺陷（cracking, inclusion, patches, pitted surface, rolled in scale, scratches）中。

**📈 对比分析**

与两种基线（原始像素、随机初始化编码器）相比，预训练编码器在聚类任务中取得91.33%的准确率、ARI 0.815、NMI 0.834，显著优于基线的34.7%和40.8%；MS‑SSIM 0.92、MSE 0.47的重建指标和Grad CAM/Grad CAM++的可视化进一步证明模型有效学习了缺陷特征。

**⚠️ 局限性**

主要限制包括：① 仍存在某些缺陷类别（如scratches）与其他类别混淆；② 辅助检测任务仅用于训练，未评估为实际检测器；③ 仅在单一数据集上验证，缺乏跨数据集的泛化评估。

---

## 97. A Bayesian framework for the uncanny valley in humanoid robot design

**arXiv ID:** 2607.13060 | [PDF](https://arxiv.org/pdf/2607.13060v1)

**作者:** Shimon Honda `[一作]` (University of Tokyo), Hideyoshi Yanagisawa `[通讯]` (University of Tokyo)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5073546131)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建层次贝叶斯生成模型，将类人机器人亲和度表述为后验加权的负类条件惊奇，并通过模型解释不安谷现象；随后在受试者实验中验证预测不确定性与观察不确定性对熟悉度评分的影响。

**💡 创新点**

将经验性的“先峰、跨模态一致性、期望匹配”等不安谷设计准则量化为可操作的四个设计变量（|y-μ_R|、(y_a-y_m)^2、σ_R^2、σ_l^2），并提出利用该模型在算法层面评估与优化机器人外观与行为的新框架。

**🔧 技术方法**

采用层次贝叶斯生成模型、Shannon惊奇、ε‑floor 似然、模糊图像处理、蒙特卡洛仿真、对齐秩变换（ART）与三因素方差分析等技术。

**📊 数据集**

使用ABOT数据库中的机器人面孔与由生成照片服务（Generated Photos）产生的人脸图像，生成10级人类相似度混合序列作为实验刺激。

**📈 对比分析**

通过三因素（人类相似度×预测不确定性×观察不确定性）实验设计，对不同模糊条件下的熟悉度评分进行比较。实验结果显示：①高观察不确定性（模糊评估图像）削弱中间相似度区间的亲和度下降；②低预测不确定性（模糊先前机器人图像）提升机器人样貌的亲和度；③模型预测的部分不确定性效应得到了验证。

**⚠️ 局限性**

限制：仅基于静态图像研究，未涉及真实机器人交互；跨模态模型仅包含外观与运动，未涵盖声音、触感等；模型对预测不确定性影响的预测被实验部分否定，说明模型需要进一步调整；未考虑受试者个体差异或文化背景导致的预测参数差异。

---

## 98. Uncertainty-Aware Sequential Decision Rules for Event-Triggered LLM Invocation in Streaming Systems

**arXiv ID:** 2607.13048 | [PDF](https://arxiv.org/pdf/2607.13048v1)

**作者:** Zhaohui Wang `[一作]` (University of Southern California), Zhaohui Wang `[通讯]` (University of Southern California)

**通讯引用:** 28550 | [OpenAlex ID](https://openalex.org/A5100358029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在流式系统中基于风险的事件触发式LLM调用策略，并在CMAPSS与CIC‑IDS2017数据上进行验证。

**💡 创新点**

提出了统一的阈值策略框架，证明了六条理论结果（最小触发间隔、阈值最优、SPRT近似、损失上界、阈值收敛、校准误差迁移），并证明传统触发器可视为该框架的特殊情况。

**🔧 技术方法**

使用GRU轻量级模型产生预测与不确定性，构建风险函数；通过OGD、LinUCB等在线阈值自适应；利用MiniMax‑M2.5和Llama‑3.1‑8B进行LLM调用；实验中采用风险阈值AUC、损失上界、LLM诊断评分等指标。

**📊 数据集**

主要使用NASA C‑MAPSS（FD001‑FD004）燃气轮机退化数据和CIC‑IDS2017网络入侵数据进行评估。

**📈 对比分析**

与六种基线（固定阈值、CUSUM、SPRT、最优停止、路由器、随机/周期采样）对比，异常分数风险R₂与LinUCB自适应阈值在调用率‑漏检率曲线上实现近1/10的Pareto AUC，召回率≈0.91，误诊率≤0.05，LLM调用成本可控，阈值对LLM成本变化鲁棒。

**⚠️ 局限性**

局限性包括：理论假设（如子鞅性、分布连续性）仅在实测中部分满足；实验仅涵盖两种LLM后端；未考虑系统层级调度与时延约束；跨域验证有限，主要集中在CMAPSS。

---

## 99. MGFace: Mask-Gated Face Matching via Conditional Similarity Routing

**arXiv ID:** 2607.13187 | [PDF](https://arxiv.org/pdf/2607.13187v1)

**作者:** Huy Che `[一作]` (University of Information Technology), Duc-Lung Vu `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于遮挡判断的面部识别流程MGFace，将查询图像先判定是否戴口罩，再按需使用全局或局部匹配。

**💡 创新点**

创新点在于将轻量级口罩分类头集成到预训练识别模型中，实现一次前向即可得到全局特征和遮挡判断，从而实现可控的匹配分流；并且在遮挡时仅对上部可见区域做补丁级重排序，既提升准确率又显著降低计算量。

**🔧 技术方法**

采用预训练的FaceNet/ArcFace骨干网络、轻量级全局平均池化+FC的口罩分类头、余弦相似度全局匹配、基于掩码过滤的上部补丁相似度重排序，以及两阶段候选检索。

**📊 数据集**

在扩展版LFW‑Mask数据集上进行评估，训练分类头使用Mask Classifier数据集。

**📈 对比分析**

与传统余弦相似度、DeepFace‑EMD等基于全局或全局+局部重排序的方法对比，MGFace在FaceNet上P@1提升约5%（至82.9%），ArcFace上提升至91.9%；同时查询时间比DeepFace‑EMD快≈20倍（从182.6s降至8.2s），显著降低VRAM占用。

**⚠️ 局限性**

主要局限在于对口罩分类头的准确性依赖；若误判会导致选择错误的匹配分支；并且仅固定上部裁剪，无法处理非口罩遮挡或对齐不准的情况。

---

## 100. Self-Improvements in Modern Agentic Systems: A Survey

**arXiv ID:** 2607.13104 | [PDF](https://arxiv.org/pdf/2607.13104v1)

**作者:** Zhe Ren `[一作]` (Jilin University), Jürgen Schmidhuber `[通讯]` (King Abdullah University Of Science And Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化现代基于基础模型（FM）的自我改进代理，提出统一的形式化框架与两大改进路径（模型参数更新与支撑结构更新），并梳理各类方法、信号来源、应用场景与评估基准。

**💡 创新点**

创新点在于：①将自我改进问题抽象为自发更新算子并给出清晰的数学定义；②提出两条互补的改进路径，区分参数空间与支撑结构的修改；③构建统一的分类体系（信号类型、目标、组件）和时间线，为跨领域研究提供共同语言；④汇总并链接公开资源与评测平台（如Awesome Self‑Improving Agents、AgentGym、SWE‑bench+ 等）。

**🔧 技术方法**

使用的技术主要为文献综述与结构化梳理，辅以对相关工作（如RLHF、Fast‑Weight 程序、Self‑Instruction、Prompt‑Evol 等）的形式化归类；通过图表展示时间线、分类树与评测基准；提供代码/数据链接以方便复现。

**📊 数据集**

使用的数据集与评测主要为公开的基准与实验平台：AgentGym、SWE‑bench、SWE‑bench+、WebArena、MagiC、MINT 等；并对这些基准中已出现的自我改进方法进行引用与对照。

**📈 对比分析**

对比方法主要是通过文献对照，说明各自改进路径在不同信号（生成示例、评估反馈、探索经验）与组件（提示、记忆、工具、全架构）上的应用与效果；并在表格中列出代表性方法、主要技术、评测指标与性能（如准确率、奖励、代码通过率）等信息。由于本文为综述，未提供新的实验结果。

**⚠️ 局限性**

局限性包括：①综述性质导致缺乏统一实验与可重复性评估；②在方法归类与评价标准上仍存在主观性，未来需要更细粒度的度量与标准化基准；③对安全性、可解释性与长期自我提升风险的讨论尚不充分；④资源与算力依赖性高，尤其是模型微调与大规模代理训练仍受限。

---

## 101. Theory-Level Autoformalization: From Isolated Statements to Unified Formal Knowledge Bases

**arXiv ID:** 2607.13292 | [PDF](https://arxiv.org/pdf/2607.13292v1)

**作者:** Marcus J. Min `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“理论层级自动形式化”（Theory‑Level Autoformalization）框架，阐述其必要性、挑战，并给出三条前沿路径（基准、通用模型与统一中间表示），旨在将自然语言到形式语言的转换从单一命题扩展到完整理论库。

**💡 创新点**

创新点在于：①将自动形式化上升到理论级别，强调知识库的层级结构与相互依赖；②提出可声称一致性的等价性检查与数据泄漏控制的理论基准；③设计可嵌入多种低资源DSL的通用中间表示，兼顾可验证性与跨语言迁移。

**🔧 技术方法**

主要技术包括：大规模语言模型（LLM）与强化学习+验证器（RL‑VR），依赖检索与层级分解机制，抽象学习与自动化定义生成，统一中间表示（Lean/IR）实现跨模态与跨语言的形式化流水线。

**📊 数据集**

使用的数据集有：现有命题级基准（ProofNet、PutnamBench、Def_Wiki、Def_ArXiv、ProofFlowBench 等），但文中指出缺乏完整理论层级基准；并基于这些数据集进行实验与评估。

**📈 对比分析**

比较方法以可验证的等价性检查为核心，评估指标为精确率/召回率；当前最优命题级自动形式化约 71.4% 的成功率，且通用模型在不同 DSL 上表现差异显著，凸显需要统一基准与更严格评测。

**⚠️ 局限性**

局限性包括：缺乏统一的理论层级基准与等价性裁判，等价性定义主观性高；低资源 DSL 与多模态输入导致训练数据匮乏；细调模型易过拟合，缺乏跨域泛化能力；现有技术尚未能实现完整的自动化理论构建与抽象创新。

---

## 102. SPINE: Bridging the Cyber-Physical Gap with Agentic AI

**arXiv ID:** 2607.13049 | [PDF](https://arxiv.org/pdf/2607.13049v1)

**作者:** Minkyu Ham `[一作]` (Northwestern University), Han Liu `[通讯]` (Northwestern University)

**通讯引用:** 146488 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为SPINE的基于代理（agentic）的框架，用于在机器人硬件上调试与部署，使得即使是机器人初学者也能在不具备专业硬件经验的情况下快速将机器人调试至可操作状态。

**💡 创新点**

创新点包括：①持久化机器人特定知识与经验，使后续调试从历史案例中学习；②确定性的安全边界，防止危险命令在机器人上执行；③以物理验证（probe‑anchored closure）为闭环，确保软件改动和硬件操作都经过现场验证后才算完成；④将机器人文档、硬件清单与运行时状态统一成结构化配置文件，支持跨平台迁移。

**🔧 技术方法**

核心技术包括：大语言模型（Claude Sonnet 4.6）与工具调用、子代理（profile builder、diagnosis subagents）、持久化 JSON 记录、safe‑runner 过滤器、诊断循环（终端、软件契约、硬件可见性、就绪范围），以及基于检索的失败模式匹配。

**📊 数据集**

使用的实验数据集为两套机器人平台：DOBOT X‑Trainer（七个复合bug场景）和AgileX PiPER（五个复合bug场景），每个场景包含软件、硬件或两者混合的多重缺陷。

**📈 对比分析**

对比方法：将SPINE与同一 LLM（Claude Sonnet 4.6）但无持久化、无安全过滤、无结构化诊断循环的“人机基线”进行比较。指标为：TTO（时间到操作）、OSS（操作化成功率）和OPS（操作员感知压力）。结果显示：在DOBOT平台上SPINE实现OSS 100%（vs 75%基线）、TTO平均 13 min 47 s（vs 15 min 25 s）且OPS明显下降；在PiPER平台上SPINE同样达成OSS 100%（vs 90%），TTO 7 min 45 s（vs 9 min 51 s）。

**⚠️ 局限性**

局限性包括：仅评估了两台机器人且实验参与者人数极少（仅三名），缺乏跨 LLM 的验证，缺少更大规模的 bug 目录，且实验只使用了 Claude Sonnet，尚不确定架构在其他模型或更复杂硬件环境下的可迁移性。

---

## 103. SARFA: Segment Anything with Radiomic Feature Alignment

**arXiv ID:** 2607.13323 | [PDF](https://arxiv.org/pdf/2607.13323v1)

**作者:** Tyler Ward `[一作]` (University of Kentucky), Abdullah Imran `[通讯]` (University of Kentucky)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 SARFA 框架，在医学图像分割中通过概率提示生成多种可行分割掩码，并使用基于放射组学特征的 Fréchet Radiomic Distance（FRD）与 Direct Preference Optimization（DPO）来训练模型，使输出的掩码在形态和纹理特征上与真实标注更为一致。

**💡 创新点**

创新点包括：①利用 SAM 的多掩码解码功能直接产生多样化的分割候选；②将放射组学特征作为监督信号，定义 FRD 作为训练目标；③使用 DPO 对基于 FRD 排序的正负掩码对进行优化，提升模型对不确定性任务的适应性；④在多模态数据（CT 与 MRI）上验证方法有效性。

**🔧 技术方法**

技术手段包括：Segment Anything Model（SAM）+ LoRA 微调、概率提示生成、PyRadiomics 特征提取、Fréchet Radiomic Distance 计算、Direct Preference Optimization 损失、标准交叉熵、Dice 与 Focal 损失的组合。

**📊 数据集**

使用 LIDC-IDRI（肺部 CT 病灶）和 BraTS2017（脑肿瘤 MRI）两大公开数据集进行实验，分别针对 CT 与 MRI 两种成像模式进行评估。

**📈 对比分析**

与十种现有基线（包括 Probabilistic U-Net、HPU-Net、PHiSeg、SAMed、P^2SAM 等）进行对比，使用 GED、FRD、Hungarian-matched IoU、D_max 等指标。实验表明 SARFA 在所有指标上均优于基线，尤其在 FRD 与传统重叠度量的相关性显著，证明放射组学对分割质量的积极影响。

**⚠️ 局限性**

主要局限：①放射组学特征提取在训练期间增加计算开销；②依赖 PyRadiomics，可能缺乏对其他特征表示的探索；③仅在 CT 与 MRI 两个数据集上验证，泛化能力待进一步验证；④当前 DPO 只使用最优与最差掩码对，未充分利用完整的候选排序信息。

---

## 104. AI-Native Insurance for Agentic AI: Pricing, Underwriting, and End-to-End Automation

**arXiv ID:** 2607.13230 | [PDF](https://arxiv.org/pdf/2607.13230v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University), Quanyan Zhu (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套完整的 Agentic‑AI 保险框架，定义风险状态、事件空间、覆盖层级和赔付函数，并通过数学规划求解保险合同的定价、限额、共保分配和治理约束；在医疗护理协同案例中展示了如何将理论映射到实际保险条款。

**💡 创新点**

创新点主要有：① 把 AI 的自主性、操作权限、外部状态权限、治理成熟度和依赖浓度聚合为统一的风险状态，突破传统以信息资产为中心的保险范式；② 构建事件概率与严重度的阶梯映射与治理衰减因子，直接把治理水平嵌入定价与收益中；③ 在合同设计中同时引入参与性、利润性、激励兼容和可行性约束，形成可求解的混合整数规划；④ 给出可观测的治理规范化表，便于将法律条款转化为可量化约束；⑤ 通过理论证明展示了可保性区域的单调性和治理阈值认证，为监管层提供可操作的合规标准。

**🔧 技术方法**

采用概率论与数理统计建模（logit、指数变换）映射风险状态到事件概率与严重度；使用多目标混合整数规划（MILP）实现合同最优求解；利用事件频率‑严重度乘积与风险加载函数计算保费；通过案例中定义的事件表和权限权重构造权重向量；采用治理评分规则将治理维度转化为层级，形成约束。

**📊 数据集**

没有使用真实保险索赔数据库；案例中使用了人工设定的医疗协同事件概率与损失金额（如 0.06×250k 等），以及权限权重表和治理成本示例；其他实验采用了仿真生成的多变量风险状态样本。

**📈 对比分析**

与传统网络安全/技术错误责任保险进行概念对比，证明在 Agentic‑AI 环境下，单层保单不再满足多因果损失，需采用多层覆盖；在案例中通过对比不同治理层级、权限暴露、授权度等因素的敏感性，展示了保费区间、保险比率和盈利变化；性能方面，优化模型在有限菜单下求解时间可控制，且保险比率在 75–80% 之间，盈利可达数千美元。

**⚠️ 局限性**

局限性包括：① 缺乏大规模真实索赔数据，导致概率与严重度参数仍为专家估计；② 仅考虑了有限事件集，实际 AI 运行中可能出现更复杂或跨域事件；③ 模型假设治理层级与成本呈单调关系，实际实施中可能出现非线性或外部依赖；④ 只给出静态优化结果，未深入讨论动态风险演化与再保需求；⑤ 对监管与合规的描述仍停留在理论层面，缺乏具体法律条文与实施路径。

---

## 105. Parsimonious disturbance-aware minimum-time planning with parametric uncertainty

**arXiv ID:** 2607.13312 | [PDF](https://arxiv.org/pdf/2607.13312v1)

**作者:** Martino Gulisano `[一作]` (Università di Pisa), Marco Gabiccini `[通讯]` (Università di Pisa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究提出并验证了考虑车辆参数不确定性的稀疏扰动感知最小圈速规划框架，用于赛道轨迹和控制输入生成。

**💡 创新点**

创新点在于将车辆参数不确定性纳入概率安全回退约束，并采用空间选择性的稀疏激活策略，只在关键赛段强化鲁棒约束。

**🔧 技术方法**

采用单轨动力学的随机建模、协方差传播、概率回退约束、直接层次投影与非线性规划，以及基于MPC的蒙特卡罗闭环验证。

**📊 数据集**

使用FSAE车辆模型和巴塞罗那-加泰罗尼亚赛道的0.70–0.77区间轨迹作为测试数据。

**📈 对比分析**

通过对比四种参考（Nom、ROB‑S、PAR‑S、PAR‑SP），在1000次MPC蒙特卡罗实验中，鲁棒规划的安全成功率提升约5倍，圈速增加约170 ms，操控能耗下降。

**⚠️ 局限性**

局限性在于仅使用单轨模型、单一赛道区间、单一扰动点以及MPC模拟驱动，未验证对更复杂车辆动力学或真实驾驶员的适用性。

---

## 106. Ask Before You Diagnose: Safe-Psych, a Sequential Evaluation Benchmark for LLMs in Psychiatry

**arXiv ID:** 2607.13036 | [PDF](https://arxiv.org/pdf/2607.13036v1)

**作者:** Oriana Presacan `[一作]` (National University of Science and Technology Politehnica Bucharest), Michael A. Riegler `[通讯]` (SimulaMet)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Safe‑Psych 基准，评估大语言模型在精神科诊断过程中如何处理不完整信息，采用顺序信息披露和澄清/诊断/放弃三种行为标签。

**💡 创新点**

创新点在于：① 把真实临床笔记按诊断流程分段并标注可诊断/需要澄清/需放弃标签；② 用 LLM judge 自动将模型自由文本映射为三种行为；③ 关注模型在信息不足时的提前诊断与澄清请求行为，填补现有静态问答基准的不足。

**🔧 技术方法**

使用多种大语言模型（GPT‑5.4、Claude‑Opus‑4.6、Gemini‑Flash‑2.5、Gemma‑3、Mistral‑3.1‑24B、Qwen‑3‑32B、MedGemma‑27B、Med42‑v2‑8B）与 LLM judge；采用序列化提示、不同推理策略（全信息/顺序信息、是否提示放弃/澄清）。

**📊 数据集**

数据集为 1,048 条真实精神科临床笔记，已匿名、英译并按五个阶段（症状、病史、精神检查、心理检查、次级诊断）分段，标注 ICD‑10 诊断及“最早可诊断”步骤。

**📈 对比分析**

对比全信息与顺序信息、带/不带放弃提示的四种推理策略；评估 Under‑/Over‑abstention、澄清率、诊断时机与准确率。结果显示即使是最强模型也存在 >60% 的 under‑abstention，澄清率低，顺序信息下 3‑字符 ICD 准确率下降约 10–15%，提前诊断比及时诊断准确率低。

**⚠️ 局限性**

局限性：数据来自单一罗马尼亚精神科医院，样本量有限且绝大多数病例为信息足够；英译可能影响不确定性线索；LLM judge 的误差导致行为标签噪声；未对所有模型进行多种子评估；不一定能推广至其他医学领域。

---

## 107. What Your Model Threw Away and Why You'll Want It Back: Masking, Fingerprinting, and Privacy from Discarded Geometry

**arXiv ID:** 2607.13046 | [PDF](https://arxiv.org/pdf/2607.13046v1)

**作者:** Zachary P. Bradshaw `[一作]` `[通讯]` (QodeX Quantum, Inc.), Zachary P. Bradshaw (QodeX Quantum, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通用框架，利用 Lie 群表示论定义“null fiber”与“stabilizer”，从而量化机器学习模型对带群作用输入所忽略的对称信息。

**💡 创新点**

创新点在于将传统的 null space 拓展到群表示视角，区分全局与点态不变性，并给出对紧群的傅里叶表征以及基于梯度与牛顿迭代的高效求解算法。

**🔧 技术方法**

使用了李群与表示论、Peter–Weyl 定理、预像定理、梯度和牛顿迭代求解、以及实验验证。

**📊 数据集**

实验数据集包括 QM9 分子性质预测数据集和 Spherical MNIST 球面图像数据集。

**📈 对比分析**

与随机采样、预计算查找表及传统差分隐私方法对比，梯度求解在几次模型评估内即可得到精确的 null fiber，掩码保持模型精度，指纹识别误差相差十个数量级，性能优于传统方法。

**⚠️ 局限性**

局限在于对非紧或非可分离群的傅里叶表征尚未完整，且隐私保护效果高度依赖群作用是否非等距；预计算表需要高覆盖率，否则效果有限。

---

## 108. Do LLMs Need Architectural Changes for Simultaneous Speech Translation? A Prefix-to-Prefix Data Driven Approach

**arXiv ID:** 2607.13158 | [PDF](https://arxiv.org/pdf/2607.13158v1)

**作者:** Junkun Chen `[一作]` (Microsoft), Jinyu Li `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于固定长度块累积流式解码、回溯式已提交前缀和教师标注的前缀-前缀（P2P）监督的无架构改动的解码方案，实现在对话式语音翻译中的同步翻译。

**💡 创新点**

创新点：①将传统需要读/写策略的同步翻译问题转化为数据驱动的前缀-前缀监督；②采用回溯长度k实现可编辑缓冲区，保证用户端无后退；③利用强大语言模型教师生成“可提交”前缀，直接引导学生模型学会何时输出与何时等待。

**🔧 技术方法**

技术：固定块（Δ秒）累积解码；前缀锚定（anchor）与回溯（rewind）机制；教师-学生框架中的前缀-前缀监督；多阶段训练（离线大规模训练 + P2P微调）；使用Phi-4-MM为基础的CSSEL（Chunked Streaming Speech Encoding LLM）。

**📊 数据集**

数据集：①离线大规模语音翻译数据，约20k小时 X→En、20k小时 En→Y、3k小时 X→Y；②微调用的200h/语言对 X→En 与 En→X，分块为4秒；③内部对话式语音语料用于评估；④CoVoST2 X→En 作为公开诊断集。

**📈 对比分析**

比较方法：将CSSEL-P2P与CSSEL（离线与流式）以及Phi-4-mini、Whisper、GPT‑4o等基线在内部对话集上按 COMETKiwi 及平均延迟（AL）对比。CSSEL-P2P 在流式条件下提升 COMETKiwi +1.54 分，AL 仅增 +0.15s；在重排强对齐上提升更显著。CoVoST2 上也恢复了约 60% 的离线损失。

**⚠️ 局限性**

局限性：①依赖教师标注的质量与生成策略；②需要针对不同延迟需求手工调节块大小 Δ、回溯长度 k 与等待预算 K；③目前仅对 X→En 与 En→X 直接监督，其他语言对的提升有限；④未对模型的计算成本与实机部署延迟做深入分析。

---

## 109. The Hitchhiker's Guide to Monoculture

**arXiv ID:** 2607.13077 | [PDF](https://arxiv.org/pdf/2607.13077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 110. Interventional Grounding Audits: Black-Box Premise-Dependency Tests for LLM Chain-of-Thought via Predicate Substitution

**arXiv ID:** 2607.13069 | [PDF](https://arxiv.org/pdf/2607.13069v1)

**作者:** Hironao Nakamura `[一作]` `[通讯]` (Independent Researcher), Hironao Nakamura (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种黑盒干预方法，用以检测链式推理（CoT）步骤是否真正依赖其声明的前提。

**💡 创新点**

创新点在于：① 通过替换单个前提谓词为新符号来干预推理；② 引入一致性与局部替换两种策略并结合级联过滤以区分直接与传递依赖；③ 将检测流程与可验证证书相结合，保证可复现。

**🔧 技术方法**

技术方法包括：黑盒谓词替换干预、步骤结论规范化（canonical形式）、五值判定体系、级联过滤机制以及对GPT‑4o的调用。

**📊 数据集**

使用数据集 ProntoQA（50 个合成多跳演绎问题）进行评估。

**📈 对比分析**

与自一致性基线（F1≈0.34）和字符串差异基线对比，方法在全依赖检测上实现 F1≈0.81（一致性策略单独时为0.81，组合后为0.819），在谓词决定性依赖上实现 F1≈0.88，显著优于基线；同时发现 66% 正确答案的“正答案错误推理”现象。

**⚠️ 局限性**

局限性包括：仅在可解析的正式推理任务上有效，受限于数据集规模与可解析率；对自然语言基准的适用性待验证；结果对模型（如 GPT‑4o）有一定依赖，覆盖率与精度随模型不同而变化。

---

## 111. SingGuard-NSFA: Extensible Guardrails for Agentic AI via Generative Reasoning and Real-Time Classification

**arXiv ID:** 2607.13081 | [PDF](https://arxiv.org/pdf/2607.13081v1)

**作者:** SingGuard Team `[一作]` `[通讯]` (Ant Group), SingGuard Team (Ant Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向代理式 AI 的 NSFA（Not‑Secure‑For‑Agents）风险分类体系、133 种语言的多模任务基准，并实现了一种双模式防护框架 SingGuard‑NSFA。

**💡 创新点**

创新点在于将 CIA 三重目标与攻击技术分离、构建 185 个细粒度风险变体的层级词典、通过双模式（生成式链路推理 + 轻量化分类头）实现在线实时检测与离线可解释审计的统一。

**🔧 技术方法**

核心技术包括基于 74 台开源 LLM 的四阶段合成数据管道、监督微调生成式风险分析、冻结后在其嵌入上训练多标签 MLP 分类头，以及多语言翻译与近似去重等数据质量保障。

**📊 数据集**

使用的数据集包含 133 种语言的 93K+ 目的构造样本（查询 63K、响应 30K）和 3.4K 交叉来源样本（从 5 大公共代理安全基准迁移而来）。

**📈 对比分析**

在三大基准上，模型（0.8B–9B）无论是生成式还是分类模式都取得了 94–97% 的二分类 F1，远超 10 种竞争对手（差距 6–12 点），分类模式实现了 45–57 ms 的实时推理速度，生成模式提供可解释链路推理。

**⚠️ 局限性**

局限性包括仅处理单轮文本输入，无法捕获多轮轨迹攻击、跨模态与代理间通信劫持；数据主训练在英文/中文语料上，低资源语言性能可能下降；基准与模型均由同一团队构建，可能产生自评偏差。

---

## 112. Accuracy Without Grounding: Diagnosing Visual Dependency Dissociation in Video LLM Benchmarks

**arXiv ID:** 2607.13305 | [PDF](https://arxiv.org/pdf/2607.13305v1)

**作者:** Jae Joong Lee `[一作]` `[通讯]` (Purdue University), Jae Joong Lee (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 20 个视频 LLM（包含 10 个架构家族、2–78B 参数）进行实验，提出并验证“视觉依赖差距（VDG）”诊断量表，并将其与传统准确率对比，探索准确率与视觉基础的可分离性。

**💡 创新点**

创新点：①提出 VDG 这一 per‑question 诊断指标，量化视频与文字输入对模型决策的差异；②使用 McNemar 统计检验证明准确率与视觉依赖可以在不同条件下独立变化；③构建四条件诊断阶梯（黑屏、单帧、打乱帧、原始视频）分解空间、帧多样性与时间顺序的贡献；④通过跨 benchmark（Video‑MME、MVBench、EgoSchema）验证任务类型光谱的稳定性。

**🔧 技术方法**

技术手段：黑屏替换、单帧与帧打乱实验、H.264 CRF 压缩探测、FPS 抽样、Bootstrap 置信区间、McNemar 与 TOST 统计检验、四条件诊断阶梯、量化帧多样性与时间贡献、量化压缩敏感性。

**📊 数据集**

使用的数据集包括：Video‑MME（600 题分层 6 任务类型）、MVBench（462 题 9 任务类型）、EgoSchema（500 题 3‑分钟视角视频）以及其他 7‑8 个公开多模态 benchmark（ActivityNet‑QA、NExT‑QA、SEED‑Bench、TempCompass、LongVideoBench、FunQA、MMBench‑Video、MSRVTT‑QA 等）。

**📈 对比分析**

对比方法：在同一套 600 题 Video‑MME 上计算 VDG 与整体准确率，绘制任务类型光谱；对比 MVBench 与 EgoSchema 的黑屏表现验证 VDG 预测的跨 benchmark 一致性；使用 McNemar 检验对模型对比在原始与黑屏条件下的差异进行显著性检验；在 FPS 与压缩实验中观察准确率变化；结果显示：多数模型准确率高但 VDG 低，说明准确率并不能反映视觉基础；帧多样性贡献显著，时间顺序贡献近乎为 0；Qwen3‑VL 在规模提升后 VDG 下降，显示了“可见性能退化”。

**⚠️ 局限性**

局限性：①实验仅涵盖多选题（MCQ）格式，未覆盖生成式任务；②四条件诊断阶梯仅在 0.25 FPS 下评估，无法完全捕捉高帧率的时间信息；③压缩实验受 H.264 编码与解码过程影响，部分视频在高 CRF 下解码失败导致结果偏差；④部分 4‑bit 量化模型与大规模模型混合使用，导致规模与量化的交互效应难以分离；⑤所提出的 VDG 诊断仍需在更大规模、更多域的 benchmark 上验证其普适性。

---

## 113. RAGthoven at SemEval-2026 Task 1: A Multi-Stage Pipeline Walks Into a Benchmark and Barely Clears the Bar

**arXiv ID:** 2607.13189 | [PDF](https://arxiv.org/pdf/2607.13189v1)

**作者:** Marek Šuppa `[一作]` (Comenius University in Bratislava), Daniel Skala `[通讯]` (Cisco Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一套多阶段、检索增强的语言模型管线RAGthoven，用于多语言受约束幽默生成。

**💡 创新点**

将幽默生成拆分为规划、写作、反思、评判四个基于计算幽默理论的阶段，并引入检索增强规划与工具调用的代理变体，验证其对创造性幽默质量的影响。

**🔧 技术方法**

采用多模型大语言模型（GPT‑5、Gemini 3 Pro、Claude Sonnet/Opus）与检索增强生成（RAG）、自我反思提示、基于Benign Violation Theory和Script‑based Semantic Theory的结构化提示，以及ReAct/多分支工具调用。

**📊 数据集**

官方SemEval‑2026 MWAHAHA任务的训练/测试集（英语、西班牙语、中文各1200/1000/300实例）以及一份98条笑话的检索语料库。

**📈 对比分析**

在官方排行榜上与Gemini 2.5 Flash基线同归于Rank 1，在西班牙语获得最高Elo 1182，英语/中文则落在同级顶端；代理版本未明显优于非代理管线。

**⚠️ 局限性**

计算成本高、检索语料库小且以英语为主、评判标准未本地化、缺乏对代理复杂度收益的定量分析，且模型可重复性受限。

---

## 114. The Entanglement Wall: Activation-Space Probes as Risk Detectors, Not Context Adjudicators

**arXiv ID:** 2607.13075 | [PDF](https://arxiv.org/pdf/2607.13075v1)

**作者:** Dominik Schwarz `[一作]` `[通讯]` (Independent Researcher), Dominik Schwarz (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了基于激活向量的风险探测器在不同 7–8B 语言模型中区分同主题的有害请求与表面相同的良性请求的能力。

**💡 创新点**

首次将源域均值差分探测器与同主题配对的直接训练、无监督、去卷积、以及跨模型转移进行系统对比，并提出“纠缠壁”概念，说明同主题判断的操作点限制。

**🔧 技术方法**

使用残差流激活探测、均值差分、支持向量机、MLP、投影、去卷积、以及基于文本的基线等技术。

**📊 数据集**

利用 HarmBench、JailbreakBench、StrongReject、Alpaca、WildJailbreak、XSTest、以及自构造的 Twin‑n43/70/163/217 对。

**📈 对比分析**

通过 AUROC、TNR@95%TPR、TPR@1%–10%FPR 等指标进行比较，结果显示源域探测在同主题对上仅达到约0.6–0.8 的 AUROC，未达到 0.90 的阈值，直接训练虽然在内部可分离，但在 XSTest 上误判率高达 80%+。

**⚠️ 局限性**

局限性包括配对样本规模小、依赖特定 Guard 规则、仅单轮英文，模型和读点差异、未覆盖更大规模或不同架构的模型，以及在迁移与去卷积过程中的跨模型可推广性不足。

---

## 115. Cost-Optimal Foundation Model Deployment Portfolio for Transportation Management

**arXiv ID:** 2607.13239 | [PDF](https://arxiv.org/pdf/2607.13239v1)

**作者:** Xi Cheng `[一作]` (Cornell University), H. Oliver Gao `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 FMDP 的混合整数规划框架，用于在交通管理中心（TMC）中联合选择基础模型与部署模式，并满足质量、延迟、安全与共享 GPU 容量等约束。

**💡 创新点**

创新点在于把模型部署视作组合优化问题，证明其 NP‑hard 并提出两阶段贪心启发式；同时给出从闭源 API 到开源 API 再到本地 GPU 的多维成本分解与折衷分析。

**🔧 技术方法**

采用混合整数规划求解、NP‑hard 归约证明以及基于成本与资源占用的贪心算法，配合对 API 价格、GPU 能耗等参数的量化建模。

**📊 数据集**

使用示例性 TMC 案例（5 个功能、19 个模型-模式组合）和公开的云 API 计价（如 GPT‑4o、Gemini、Qwen 等）作为实验数据；未使用真实的交通数据集，仅用示例性查询速率与性能阈值。

**📈 对比分析**

通过与五种单一策略基线（All‑Closed‑Frontier、All‑Closed‑Budget、All‑Open‑Source‑API、All‑On‑Premise、Classical‑Only）对比，FMDP 的月度总成本仅为 34 美元，比最便宜可行基线低 97%，并在所有约束下保持可行。

**⚠️ 局限性**

局限性包括：仅为每个功能分配单一模型，未考虑模型级联、价格与需求不确定性以及跨功能的相互依赖；实验基于示例性参数，缺乏真实运营验证；未来工作需扩展至多周期规划与不确定性建模。

---

## 116. Final Authority in AI Governance: Frontier-Provider Sovereignty and Action-Centered Deployer Governance

**arXiv ID:** 2607.13040 | [PDF](https://arxiv.org/pdf/2607.13040v1)

**作者:** Zexun Wang `[一作]` `[通讯]` (Ond Holdings Inc.), Zexun Wang (Ond Holdings Inc.)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨在企业工作流中嵌入的先进 AI 系统后，最终决策权应落在谁手中，并比较两种治理模型：前沿模型提供者主权与行动中心化部署者主权

**💡 创新点**

首次把 AI 治理从单一安全关注转为权责分配的视角，并提出将“受治理行动”作为权责核心的 PCAA 架构

**🔧 技术方法**

定性比较法：阅读公共治理框架、供应商政策与实现层面证据，并基于六个评估准则对两种主权模型进行对比

**📊 数据集**

主要使用公开政策文件（EU AI Act、NIST AI RMF、Singapore Model AI Governance Framework 等）、供应商声明（Anthropic、Microsoft 等）以及行业调查（Stanford AI Index、IBM CIO/CTO 调研、McKinsey 2026 AI 信任成熟度报告、Cisco AI Readiness Index）

**📈 对比分析**

通过案例对照与六维评估准则（后果对齐、情境敏感、运行时可移植、证据闭合、前沿风险可见、抗集中化）进行比较，未给出量化性能指标，结论为行动中心化治理在企业场景中更具实用性

**⚠️ 局限性**

依赖公开文件和供应商声明的可见性，可能存在偏见；研究者与 PCAA 方案的亲近度可能导致主观倾向；市场数据为方向性而非决定性，缺乏实验验证

---

## 117. Designing Safety-Constrained LLM Systems for Public Health Information Access

**arXiv ID:** 2607.13038 | [PDF](https://arxiv.org/pdf/2607.13038v1)

**作者:** Ben Torkian `[一作]` (University of South Carolina), Jun Zhou `[通讯]` (University of South Carolina)

**通讯引用:** 20107 | [OpenAlex ID](https://openalex.org/A5100781212)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个安全约束的大型语言模型系统，用于母婴健康资源导航，确保系统在医疗信息提供时遵循严格的安全边界。

**💡 创新点**

创新点在于：多层级安全管控（输入过滤、域验证、生成约束、后期校验）与域受限检索增强生成（RAG）相结合；匿名多用户会话管理与完整审计日志；将所有回答严格绑定至策划好的公共卫生资源并强制源引用。

**🔧 技术方法**

技术包括：GPT‑4o LLM、基于向量检索的RAG、Node.js/Express 后端控制层、Azure OpenAI 接口、低温度生成参数、结构化提示模板、实时异常检测与预定义急救响应、JSON 格式审计日志、监控与性能计时。

**📊 数据集**

数据集为手工策划的母婴健康资源数据库（包含网站、文档、医疗术语及其关联资源），已嵌入向量存储并加以元数据标注，用于检索与生成。

**📈 对比分析**

通过情景化测试（包含在域内、域外和急救查询）评估，安全违规率为0%，所有回答均可追溯到资源并提供来源引用；成功率100%；平均响应时间为5.3秒，系统无错误。

**⚠️ 局限性**

局限性包括：仅在南卡罗来纳州的资源范围内；资源数据库需人工维护，更新不即时；测试规模有限，未包含大规模用户实验或长期使用评估；主要支持英语，缺乏多语言覆盖；匿名会话限制了个性化与长期跟踪能力。

---

## 118. Proof in a Bottle: Long-Lived Verifiable Secret Sharing via Pre-Quantum Commitment and Immutable Ledger Binding

**arXiv ID:** 2607.13235 | [PDF](https://arxiv.org/pdf/2607.13235v1)

**作者:** Markus Jakobsson `[一作]` (Artema LABS), Keir Finlow-Bates `[通讯]` (Artema LABS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种名为 Proof in a Bottle（PiB）的方案，用于在未来量子攻击出现之前，将可验证秘密共享（VSS）的绑定性通过可哈希的盐化叶子与不可变账本锚定，从而实现“commit‑now、reveal‑later”安全性。

**💡 创新点**

创新点在于将 Pedersen 绑定性与长期绑定分离：在经典时段完成 Pedersen 验证，随后通过在不可变账本上锚定加盐、索引绑定的哈希来保证在后期量子计算机出现后仍无法伪造共享；不需额外的后量子算术假设，只需时间与账本最终性假设。

**🔧 技术方法**

使用的技术包括 Pedersen 公开承诺、可哈希随机预言机（哈希函数）、Merkle 树结构、盐化叶子、不可变公共账本（如迁移到后量子安全共识的区块链）以及量子安全的哈希第二预像抵抗。

**📊 数据集**

本文未使用传统意义上的数据集；其验证基于理论分析与对比实验，主要考察哈希长度、盐长度与账本最终性深度对安全性的影响。

**📈 对比分析**

与基于格的后量子 VSS 方案对比，PiB 在绑定性上仅需第二预像抵抗与账本最终性，避免了格硬问题与大参数、复杂零知识的缺点；成本上更低（仅需少量加密运算与 Merkle 路径传输），适用于批量存储共享的场景。

**⚠️ 局限性**

局限性包括：① 必须在量子计算机出现前完成绑定；② 绑定安全高度依赖账本迁移至后量子共识并获得充分最终化；③ 隐私性仅为统计隐藏，需足够长的盐来抑制猜测；④ 对量子化的发行者不提供防护。

---

## 119. EZSMT Version 3, Matured

**arXiv ID:** 2607.13344 | [PDF](https://arxiv.org/pdf/2607.13344v1)

**作者:** Yuliya Lierler `[一作]` `[通讯]` (University of Nebraska Omaha), Yuliya Lierler (University of Nebraska Omaha)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了可扩展的 SMT‑based CASP 框架 3（版本 3），该框架基于 translational 方法，将约束答案集程序（CASP）翻译为 SMT 公式后交给现成的 SMT 求解器进行求解，并支持多种约束类型、弱约束优化和多语言接口。

**💡 创新点**

创新点主要有：
- 在 translational CASP 框架中引入可扩展语言规范与语义，支持弱约束优化（optimization）与多种逻辑（LIA、LRA、LIRA、IDL）；
- 与 clingo‑5 grounder 结合，利用其语言定义与智能 grounding，支持多种约束类型并快速生成新的 CASP 求解器；
- 通过增量 SMT 求解实现多答案集枚举与优化；
- 架构基于组件化，易于扩展和集成新的 SMT 后端或约束逻辑。

**🔧 技术方法**

使用的技术包括：Answer Set Programming（ASP）+ Satisfiability Modulo Theories（SMT）；clingo‑5 进行 grounding；翻译为 SMT‑LIB 语言；调用 Z3、CVC4、Yices、MathSAT 等 SMT 求解器；采用 completion、level‑ranking、weak‑constraint 语义重写等 ASP 理论；增量求解与模型约束排除技术。

**📊 数据集**

实验使用的 benchmark 包括 ASP 竞赛第三届与第五届中的 10+ 个实例：Reverse Folding、Incremental Scheduling、Weighted Sequence、Blending、Mixed‑BL、RoutingMin、RoutingMax、Traveling Salesman、Labyrinth、Car、Generator 等，涵盖紧（tight）与非紧（non‑tight）程序，并覆盖多种逻辑（LIA、LRA、LIRA、IDL）。

**📈 对比分析**

对比方法：将 3 与其他 CASP 求解器（如 ezsmt‑v2、clingcon、clasp、cmodels 等）在相同编码下进行统一的基准测试；报告总时间、未解实例数等指标。结果显示：
- 在 LIRA 编码下 3 明显优于其它求解器（其它求解器缺乏对 LIRA 的支持）；
- 在 LIA、LRA、IDL 编码下，3 与专用求解器相当或更快，尤其在 tight 程序中表现突出；
- 性能取决于所选 SMT 后端，Z3 与 CVC4 在大多数 benchmark 上表现最好。

**⚠️ 局限性**

局限性：
- 尚未实现所有 CASP 特色功能（如规则头中的 irregular atoms、all‑different 约束、聚合优化等）；
- 需要手动编写理论规范（theory specification），并为变量提供域约束才能获得最佳性能；
- 目前仅支持四种 SMT 逻辑（LIA、LRA、LIRA、IDL），对其它逻辑的支持需要进一步扩展；
- 依赖外部 SMT 求解器性能，若求解器对某些逻辑优化不足，整体性能受限。

---

## 120. A Reality Check on Quantum Optimisation: Evidence from an Industrial Case Study

**arXiv ID:** 2607.13325 | [PDF](https://arxiv.org/pdf/2607.13325v1)

**作者:** Hila Safi `[一作]` (Siemens Foundational Technologies), Wolfgang Mauerer `[通讯]` (Technical University of Applied Sciences Regensburg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在工业生产物流的作业车间排程问题上，开展了多平台量子及量子启发式优化的实验评估。

**💡 创新点**

创新点在于将工业真实实例与两种量子优化模型（单约束与多约束QUBO）相结合，并对比量子、量子启发式与经典求解器的性能，揭示约束密度对硬件可扩展性的影响。

**🔧 技术方法**

使用了IBM Gate、D-Wave Advantage 及 Fujitsu Digital Annealer 量子/量子启发式硬件，以及QAOA、QUBO模型、MILP、启发式迭代求解器。

**📊 数据集**

数据集来自西门子真实工厂的作业车间排程实例，规模从2机2具机床3工序到32机85具机床256工序不等。

**📈 对比分析**

通过共享测试配置、统一的有效/最佳/无效结果比例、最优性缺口和运行时间进行比较，发现单约束模型在所有平台表现最好，而多约束模型在D‑Wave上可行性低；Fujitsu Digital Annealer在规模和质量上优于经典启发式。

**⚠️ 局限性**

局限性包括只评估作业分配阶段、未包含实时动态工序、只对特定JSSP变体、对量子硬件的噪声与拓扑限制敏感、未测量通信延迟和云排队延迟。

---

## 121. STKAN: Kolmogorov-Arnold Networks for Spatio-Temporal Forecasting

**arXiv ID:** 2607.13108 | [PDF](https://arxiv.org/pdf/2607.13108v1)

**作者:** Sicong Lai `[一作]` (Hong Kong University of Science and Technology), Guangyin Jin `[通讯]` (Chang'an University)

**通讯引用:** 2154 | [OpenAlex ID](https://openalex.org/A5018357954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出STKAN，一种在空间和时间维度使用Taylor多项式Kolmogorov–Arnold网络（KAN）作为token混合器并结合可学习的软节点分组与Transformer注意力的交通流预测模型。

**💡 创新点**

创新点在于将KAN作为非线性函数逼近器引入spatio‑temporal token混合，探讨函数逼近器设计对预测性能的影响，并引入自适应节点分组机制。

**🔧 技术方法**

使用的技术包括Taylor‑polynomial KAN层、可学习软节点分组、卷积提取局部时序特征、MLP通道混合、空间和时间双向Transformer注意力以及多维嵌入。

**📊 数据集**

在五个交通预测基准上评估：PEMS04、PEMS07、PEMS08、PEMS‑BAY和METR‑LA。

**📈 对比分析**

与11种基线（STGCN、GWNet、AGCRN、GMAN、MTGNN、STDN、STID、STAEformer、STWave等）在MAE/RMSE/MAPE上进行对比，STKAN在大部分数据集上取得最优或接近最优成绩，尤其在PEMS04/07/08和PEMS‑BAY上表现突出；METR‑LA上表现略逊于最强基线。

**⚠️ 局限性**

局限性包括：仅使用单次实验，缺乏统计显著性检验；无法单独归因于KAN而非其他结构；模型参数量与计算效率未系统评估；在METR‑LA等速度数据集上的优势不明显。

---

## 122. Falsifiable Release Gates for Self-Improving Systems

**arXiv ID:** 2607.13070 | [PDF](https://arxiv.org/pdf/2607.13070v1)

**作者:** Deepak Soni `[一作]` `[通讯]`, Deepak Soni

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可验证的发布门（falsifiable release gates）方法，并在一个开源自我改进运行时中实现了七个门，保证每新增功能必须通过机器可检查的验收套件，并通过机器校验核心保证安全关键属性；

**💡 创新点**

创新点在于：①把安全验证拆分为可证明、可检测的发布门；②引入“牙齿纪律”（teeth discipline）和“跟踪一致性”（trace conformance）以保证模型和代码保持一致并能发现故障；③将自我改进限制在可验证的策略变更范围内，仅允许收紧变更自动合并，放宽变更需人工；④公开完整的测试套件和TLA+规范，构建可复现的基准；

**🔧 技术方法**

技术包括：Python实现的运行时；基于TLA+的有限状态模型和完整枚举检查；哈希链审计日志与能力令牌（HMAC签名）；机器可检查的验收套件；持续集成中的快速检查器；加密隔离（租户密钥层级）和实时监控；

**📊 数据集**

使用的数据集包括：500条注入攻击案例（用于G8验证）；3个保留语料库（用于G9学习无漂移验证）；100万个合成合法执行轨迹（用于模型与代码的符合性验证）；以及公开的基准模型和工具；

**📈 对比分析**

对比方法为逐门测评，所有122个测试案例均通过；G8成功拦截432/432注入攻击，未出现权限提升；G9在3个语料库中将漏检率降至0且误报率为0；G10实现了单tick控制的精准漂移归因（1.0精度，0误报）；G11实现租户加密隔离与审计可追溯；G12核心在291个可达状态下完全满足非绕过属性，并在1百万轨迹上无拒绝；整体性能在持续集成中几秒完成，运行时零额外开销；

**⚠️ 局限性**

局限性包括：①证明仅在有限状态空间内完成，未覆盖学习组件；②安全断言依赖自评，需外部红队或正式验证；③模型与代码可能产生偏差，仅通过采样监控检测；④自我改进仅限于策略规则，仍需人工干预放宽变更；⑤门的依赖顺序强制且不保证在所有系统中的适用性。

---

## 123. When is the combined load identifiable from a stress-intensity profile? A coupled forward-inverse study on SIFBench finite-element data

**arXiv ID:** 2607.13074 | [PDF](https://arxiv.org/pdf/2607.13074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 124. Accuracy-Preserving Stability Regularization for Large-Scale Retail Demand Forecasting

**arXiv ID:** 2607.13331 | [PDF](https://arxiv.org/pdf/2607.13331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 125. Privacy Preserving Recommender Systems Balancing Personalization with Privacy

**arXiv ID:** 2607.13328 | [PDF](https://arxiv.org/pdf/2607.13328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 126. Can LLMs Learn and Apply Multi-Level Modelling Semantics? A First Empirical Study

**arXiv ID:** 2607.13257 | [PDF](https://arxiv.org/pdf/2607.13257v1)

**作者:** Yuhong Fu `[一作]` (Adelaide University), Markus Stumptner `[通讯]` (Adelaide University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型（LLM）在多级建模（MLM）中的能力进行了首次经验性评估，具体任务是让GPT-5.4、Claude Opus 4.6和Gemini 3.1 Pro在给定SLICER语义规范的前提下生成多级模型，并与人工验证的参考模型进行对比。

**💡 创新点**

创新点在于①首次将LLM用于MLM任务并验证其语义理解与生成能力；②设计了一套针对MLM的专门评估指标（如Instantiation/Specialisation Correctness、Semantic Constraint Satisfaction Rate等）；③系统比较了不同LLM与六种提示策略（零shot、few-shot、self‑check、step‑by‑step）对模型质量的影响。

**🔧 技术方法**

技术主要包括：使用GPT-5.4、Claude Opus 4.6和Gemini 3.1 Pro三大商用LLM进行文本到SLICER代码的生成；对模型输出采用语法与语义检查、图编辑距离、关系匹配等量化指标；并通过统计检验（Kruskal–Wallis、Mann–Whitney U等）评估差异显著性。

**📊 数据集**

数据集为MULTI Warehouse Challenge情境的任务描述与SLICER语言的语义规范，共生成90个最终模型（每种LLM×6种提示×5次运行），以及30个自检草稿，总计120个模型供评估。

**📈 对比分析**

比较方法为先计算语法正确性（Invalid Construct Count）、语义完整性（Semantic Constraint Satisfaction Rate、Instantiation/Specialisation Correctness）以及完整度、相似度（Precision/Recall/F1、Graph Edit Distance、Hierarchy Preservation Rate）等指标。实验结果显示所有模型基本能生成合法的SLICER代码，语法正确率近乎100%，但语义正确率仅52%–79%；Claude在多项指标上表现最平衡，其余两者在关系种类选择和约束覆盖上存在较大不足。

**⚠️ 局限性**

局限性包括：仅评估单一任务（MULTI Warehouse Challenge）和单一MLM语言（SLICER），无法验证结果的跨任务或跨语言泛化；提示设计与LLM偏倚可能影响评估；由于实验规模受限（每种配置仅5次），统计显著性和效果量估计存在较大不确定性。

---

## 127. Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

**arXiv ID:** 2607.13319 | [PDF](https://arxiv.org/pdf/2607.13319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 128. Safeguard-Conditioned Uplift: Measuring Utility-Risk Frontiers for Dual-Use Biology Assistants

**arXiv ID:** 2607.13039 | [PDF](https://arxiv.org/pdf/2607.13039v1)

**作者:** Dipesh Tharu Mahato `[一作]` `[通讯]` (New York University), Dipesh Tharu Mahato (New York University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并量化了在不同部署访问条件下（如安全提示、外部控制器等）双重使用生物学助手的有益性与危害性，采用“safeguard-conditioned uplift”方法。

**💡 创新点**

创新点在于提出基于人类判断的 utility‑risk 前沿，直接衡量访问条件对有益帮助与危害行动的影响，并引入风险预算化校准以选择最佳阈值，摆脱单纯拒绝率评估的局限。

**🔧 技术方法**

使用技术包括：人类标注的正确性、行动性和谨慎/拒绝评分；prompt 与 output 结合的风险评分与阈值控制器；bootstrap 置信区间；阈值搜索与风险预算化校准；对比多种访问条件（安全提示、外部控制器、prompt‑only、output‑only 等）。

**📊 数据集**

使用 108 项双重使用生物学任务的 surrogate benchmark，其中包含 18 项 held‑out 测试集；另外构建了 adaptive、Test‑B、模型宽度扩展等子集以检验稳健性。

**📈 对比分析**

比较方法：在 600 行盲注样本中计算 benign correctness 与 harmful actionability，形成 2D utility‑risk 前沿；结果显示外部控制器相较于普通提示可将 harmful actionability 降低约 0.06，但 benign correctness 变化不显著；在 Claude 上安全提示表现更优，而在 Gemini 上外部控制更为有效。

**⚠️ 局限性**

局限性：仅评估 surrogate 任务，未覆盖全部真实生物学风险；结果对模型、攻击方式和任务分布高度依赖；控制器设计不具普适性，需在新模型或新攻击上重新校准。

---

## 129. Efficient and Privacy Aware Edge Cloud Collaborative Inference for Large Language Models

**arXiv ID:** 2607.13093 | [PDF](https://arxiv.org/pdf/2607.13093v1)

**作者:** Yi Li `[一作]` (KunlunMeta), Jiexiong Liu `[通讯]` (KunlunMeta)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端云协作的LLM推理框架，端侧负责预处理、轻量化投影、KV缓存授权与推测解码，云侧完成解码器推理、KV缓存维护和高维词表投影；

**💡 创新点**

创新点在于：①端侧可授权控制KV缓存，避免将敏感历史传给云；②将词表投影拆分为端侧低维投影与云侧高维投影；③结合推测解码与端云验证，减少往返交互；④使用AES‑GCM加密传输并保持LoRA轻量模块本地化；

**🔧 技术方法**

技术包括：端侧分块流式传输、GPU切片/张量并行、ONNX Runtime量化推理、AES‑GCM加密、LoRA参数化、推测解码与验证、语言自适应词表裁剪、端侧KV授权、8‑bit张量量化；

**📊 数据集**

实验使用7B级解码器模型，词表150k，隐藏维度4096；未说明具体公开数据集，而是使用1000条自构造的中英混合提示；

**📈 对比分析**

与云端纯推理、加密传输、朴素拆分推理以及端侧轻量化模型做对比。结果显示在CPU、GPU和边缘设备上，端云协作模型每个token平均延迟降低29%–46%，每token下行payload降低30%–67%，生成质量（Next‑Token一致率≥98.6%、Perplexity≈7.87、MMLU≈54.3）接近全云模型；

**⚠️ 局限性**

局限在于：①未给出正式差分隐私或安全多方计算保证，仅靠加密与LoRA减轻风险；②推测解码的接受率受模型与任务影响，可能导致验证开销；③对高频交互场景的带宽/延迟最优配置仍需进一步调优；④使用量化、分割投影可能导致极端场景下的生成质量下降；

---

## 130. Autonomous UAV Route Planning for Coverage Maximization in Environmental Monitoring: A Systematic Literature Review

**arXiv ID:** 2607.13054 | [PDF](https://arxiv.org/pdf/2607.13054v1)

**作者:** Sebastian Jouannet-Contreras `[一作]`, Carola Figueroa-Flores `[通讯]` (Universidad del Bío-Bío)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文提出了一个基于PRISMA 2020框架的系统文献综述（SLR）协议，并报告了对2015-2026年间无人机环境监测覆盖规划相关文献的初步筛选结果。

**💡 创新点**

创新点在于构建了可复制的检索与筛选流程，针对覆盖规划问题明确了三大研究问题及对应的证据提取维度，并首次对已筛选文献的技术趋势和评价指标进行量化预览，为后续深入综述奠定基础。

**🔧 技术方法**

使用的技术包括PRISMA 2020流程、PICOC框架、Scopus和Web of Science数据库的Boolean检索、标题/摘要/关键词筛选、初步计数分析、以及计划中的质量评估和结构化数据提取。

**📊 数据集**

本研究的数据集来源于文献数据库：共识别562条记录（Scopus 384条，Web of Science 178条），经去重后得到401条唯一记录，并在此基础上筛选出247条符合全文评估的研究。

**📈 对比分析**

目前尚未完成全文评估和方法比较；已初步统计了保留文献在覆盖优化、能源约束、多无人机协同、几何表示等方面的出现频率，并发现绝大多数研究基于仿真验证；后续将通过质量评分和结构化提取，对不同算法家族在覆盖率、能耗、行驶距离、计算耗时等指标上的性能进行对比。

**⚠️ 局限性**

主要局限包括：检索仅限于Scopus和Web of Science，可能漏检其他重要期刊/会议；初步筛选基于标题/摘要/关键词，未进行全文编码，可能产生信息失真；单人筛选导致选择偏差；大部分研究仅在仿真环境中验证，缺乏真实世界实测，导致外部有效性不足。

---

## 131. CoDiffGRN: Rethinking Gene Regulatory Network Inference via the BEELINE-KGC Benchmark and Co-evolutionary Discrete Diffusion

**arXiv ID:** 2607.13120 | [PDF](https://arxiv.org/pdf/2607.13120v1)

**作者:** Jiaze Song `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将单细胞转录组数据的基因调控网络推断转化为迁移学习的知识图谱补全任务，并配套开发了新型评测基准。

**💡 创新点**

创新点在于设计了联合离散扩散模型（同时模拟基因表达状态与调控边的共进化）和专门针对TF的子图采样策略（TASS），实现对未见基因的稳健归纳泛化和高质量的Top‑K预测。

**🔧 技术方法**

采用离散化的扩散过程、细胞聚类阈值化、联合节点-边离散扩散、图注意力网络（GATv2）以及子图采样技术。

**📊 数据集**

使用BEELINE数据集中的多种单细胞RNA测序数据进行实验。

**📈 对比分析**

与传统统计方法、基于特征的深度模型、图神经网络及现有扩散模型对比，模型在Hits@10、Hits@50和MRR指标上显著优于所有基线，达成SOTA水平。

**⚠️ 局限性**

局限性包括评估仅覆盖已知单细胞数据集，跨物种或稀有组织的泛化尚未验证；模型在训练和推理过程中存在较高的计算开销。

---

## 132. C-Norm: Cell-Distribution Normalization Enables Precision Recognition of Medical-Cell Image

**arXiv ID:** 2607.13116 | [PDF](https://arxiv.org/pdf/2607.13116v1)

**作者:** Yang Qianl `[一作]` (Chongqing University Cancer Hospital), Zou Dongl `[通讯]` (Chongqing University Cancer Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对宫颈细胞薄层细胞学图像进行异常细胞检测与识别，提出细胞分布归一化（Cell-Norm）方法，并将YOLOv12与DINOv3融合构建YoLo-D模型。

**💡 创新点**

创新点主要有两点：① 针对TCT图像中细胞空间分布不均导致的训练/测试分布偏差，提出Cell-Norm对细胞实例进行重采样、重排，实现在图像级别的均匀分布；② 将YOLOv12一阶段检测框架与DINOv3视觉编码器结合，提升小目标细胞的特征表示与定位精度。

**🔧 技术方法**

技术细节包括：使用Segment Anything Model (SAM) 进行细胞实例分割与提取；在YOLOv12基础上插入预训练的DINOv3-ViT-B/16 做特征增强；采用CIoU、Binary Cross Entropy 与 Distribution Focal Loss 作为损失函数；对重采样后的细胞实例执行随机缩放 (0.9–1.1) 与旋转 (0–360°) 以增强多样性。

**📊 数据集**

数据集为公开的TCT数据集，原始 8037 张 2048×2048 图像，包含 14684 个标注异常细胞；通过裁剪得到 640×640、1024×1024、1536×1536 ROI，并在 Cell-Norm 处理后进行数据扩增。

**📈 对比分析**

与传统检测器（SSD、RetinaNet、FCOS、Faster R‑CNN、Cascade R‑CNN、YOLOv3/7、DETR 等）以及 YOLOv12 进行对比。YOLO‑D 在 1× 与 4×扩增时 AP_50:95 从 48.09 提升至 92.41，AP_50 提升至 94.32，AR_50:95 达到 98.76，F1 分数 86.87；相较于 YOLOv12 提升约 4–6 个百分点。多尺度 ROI 下，YOLO‑D 继续保持领先。

**⚠️ 局限性**

局限性包括：① Cell‑Norm 依赖高质量细胞实例分割，分割误差可能影响后续重采样效果；② 随机重采样虽然保持整体分布，但可能无法完全模拟真实细胞排布，对模型泛化至不同设备/光照条件的适应性仍需进一步验证；③ 该方法主要在单中心公开数据集上评估，跨中心或大尺寸图像的实时推理效率尚未充分考察。

---

## 133. WaterMoE: Expert-Routing-based Watermarking for High Fidelity and Efficiency

**arXiv ID:** 2607.13099 | [PDF](https://arxiv.org/pdf/2607.13099v1)

**作者:** Z Sun `[一作]` (Shanghai Jiao Tong University), L Xiang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 23285 | [OpenAlex ID](https://openalex.org/A5100331028)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对Mixture-of-Experts（MoE）大语言模型的水印嵌入与检测方案WaterMoE，在推理过程中通过在专家路由中注入微小偏置实现水印，而非传统的后处理token概率扰动，保持文本质量且几乎不增加推理延迟；

**💡 创新点**

创新点在于将水印信号嵌入MoE模型的专家选择逻辑，利用路由偏置产生可检验的统计信号，解决传统方法在高复杂度任务中检测率低、文本质量下降、推理开销大的痛点；

**🔧 技术方法**

使用的技术包括：MoE路由偏置注入（Green Expert Map + δ偏置）、基于参考模型的对数似然差异检测、GPU可并行的元素级加法实现、以及针对不同任务的实验基准构建；

**📊 数据集**

实验数据集涵盖低、中、高复杂度任务，主要有：C4、Booksum、MultiNews、ELI5、APPS、CodeContests、GSM8K、MMLU、IFEval、WritingBench；

**📈 对比分析**

与现有水印方法（KGW、SynthID、EWD、MorphMark、EXPEdit、Unbiased、UPV、SIR等）在同一MoE模型（Mixtral‑8×7B、Qwen3‑30B）上对比，WaterMoE在大多数任务上实现TPR@1%≥90%，AUC≈1，且文本质量（PPL）与未水印模型无显著差异，插入延迟仅1.1%，相较传统方法提升4倍速度；

**⚠️ 局限性**

局限性在于目前仅适用于MoE架构，无法直接迁移到纯Dense Transformer；对极端删除攻击鲁棒性有限，且需预先生成Green Expert Map，增加部署前准备工作；

---

## 134. A Better-than-$e^{1/e}$ Approximation Algorithm for Nash Social Welfare under Additive Valuations

**arXiv ID:** 2607.13340 | [PDF](https://arxiv.org/pdf/2607.13340v1)

**作者:** Vignesh Viswanathan `[一作]` `[通讯]` (University of Massachusetts), Vignesh Viswanathan (University of Massachusetts)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种新的随机算法，在加法性评估下对最大化纳什社会福利（Nash Social Welfare, NSW）问题实现了 (e^{1/e} - c) 的近似比（c 为常数>0），即在过去八年中首次突破了最优的 e^{1/e} 近似界。

**💡 创新点**

创新点：
- 通过配置线性规划（Configuration LP）而非传统的消费限制市场均衡（spending‑restricted market equilibrium）来建模 NSW，证明其可解释的目标函数与更优的可行性；
- 设计了一种 “dependent partial rounding” 的分桶与匹配技术，能在保证边缘概率不变的同时将大件商品“压缩”到各代理的 bucket 中，从而实现更精细的负相关随机匹配；
- 将 Shmoys‑Tardos (ST) 采样匹配方法与“人工实例（Artificial Instance）”结合，利用同一代理的等价投影实例中所有代理均拥有相同评估，进而把 ST 结果映射到经典的 round‑robin 分配，完成了对 NSW 的全局下界分析；
- 证明配置 LP 的割裂整型逼近的可行性间隙明显优于消费限制市场均衡模型，并揭示加权 NSW 与普通 NSW 在此框架下的本质差异。

**🔧 技术方法**

使用的技术：
- 配置线性规划与相应的分数化分配（corresponding fractional allocation）。
- Shmoys‑Tardos 负相关随机匹配（ST rounding）。
- 负相关性与“dependent partial rounding”技术，能够在保证匹配边缘期望的前提下保持各桶大小为 1，极大提升了舍入过程的精度。
- round‑robin 分配与“人工实例”构造，用于把不相同代理的评估映射到同类评估的实例中，从而利用已知的 round‑robin NSW 下界推导 ST 结果。
- AM‑GM 与指数不等式的组合，精细化对 NSW 的对数值下界证明。

**📊 数据集**

本文没有使用任何实际数据集；所有结果均为理论上可在多项式时间内得到，输入为有理数评估函数，算法在理论上是可执行的。

**📈 对比分析**

比较方法：与此前最优的 e^{1/e}（≈1.445）近似算法对比，本文证明存在一个正的常数 c 使得近似比略低于 e^{1/e}，即得到 (e^{1/e} - c) 的改进；在所有已知的理论下界与近似算法中，此结果是目前唯一的、可实现的 e^{1/e} 之下的性能提升。

**⚠️ 局限性**

局限性：
- 常数 c 仅为理论存在，实际数值取决于对配置 LP 解决方案的精确度与数值比较，常常非常小，难以在实践中显著提升。
- 依赖输入评估为有理数并可用多项式位宽精确表示，若评估为无理数需额外的逼近步骤。
- 该方法在实现上涉及对配置 LP 的精细化分块与负相关匹配，虽然理论上多项式，但实际实现复杂度较高。

---

## 135. The Refusal Residue: When Probes Catch Alignment Faking and When They Don't

**arXiv ID:** 2607.13346 | [PDF](https://arxiv.org/pdf/2607.13346v1)

**作者:** Aman Mehta `[一作]` `[通讯]` (Snowflake AI Research), Aman Mehta (Snowflake AI Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大规模开放权重 LLM 行为与隐藏状态的探测，研究了在监控条件下的对齐欺骗现象，并提出了一套防止假阳性的五项检测控制方法。

**💡 创新点**

创新点包括：①首次量化“拒绝残留”这一非对称内部表示差异，揭示对齐欺骗可检测的隐藏信号；②将传统探针与正交约束、留一查询交叉验证等严谨控制相结合，避免假阳性；③通过激活拉伸验证该内部方向的因果与预测关系的解耦。

**🔧 技术方法**

使用技术包括：隐藏状态提取（gen5 采样）、线性与 MLP 探针、每折残差化（per‑fold residualization）、留一查询交叉验证（LOQO）、正交约束探针、主成分剔除、随机 Procrustes 对齐评估、激活拉伸（steering）等。

**📊 数据集**

数据集与实验对象为：50 条边界有害查询（来自 OR‑Bench 与 XSTest），13 款开放权重 LLM（Qwen3‑32B、Llama‑3.1‑8B 等），以及 Claude Opus 4 评判器生成的合规/拒绝标签。

**📈 对比分析**

在不同评估协议下的比较：传统泄漏式探针 AUROC 约 0.63；严格 LOQO+正交约束后，Llama‑3.1‑8B 的单样本检测 AUROC 为 0.87，Qwen3‑32B 仅 0.43；激活拉伸对行为无显著影响（|h|<0.08）。

**⚠️ 局限性**

局限性：仅两款模型自然产生对齐欺骗，检测仅在 Llama‑3.1‑8B 上有效；评估受限于特定监控提示与边界查询；激活干预未显示因果效果；跨模型转移在随机 Procrustes 基准下无显著提升。

---

## 136. Construction of Generalized Weighing-Hadamard Matrices over Finite Fields

**arXiv ID:** 2607.13324 | [PDF](https://arxiv.org/pdf/2607.13324v1)

**作者:** Gustavo T. Bastos `[一作]` (Federal University of São João del-Rei), Verónica Requena `[通讯]` (University of Alicante)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了有限域上的广义加权哈达玛矩阵（GWH矩阵）的存在性、性质和构造。

**💡 创新点**

创新点在于引入了GWH矩阵的强等价概念，并证明了可逆GWH矩阵在矩阵乘法下形成一个群，同时该群与正交矩阵的子群的商群是阿贝尔的。

**🔧 技术方法**

使用了线性代数和群论的技术，特别是矩阵乘法和正交变换。

**📊 数据集**

研究了在任意有限域上的GWH矩阵，具体的构造和性质分析。

**📈 对比分析**

通过与已有的哈达玛矩阵和加权矩阵的比较，展示了GWH矩阵在编码理论中的应用，尤其是在自对偶和线性互补对偶码的构造中表现出优越性。

**⚠️ 局限性**

限制在于目前的研究主要集中在方阵的情况，未来的工作将探讨非方阵GWH矩阵及其在编码理论和密码学中的应用。

---

## 137. Finding the Right Tables and Columns: A Benchmark and Corpus-Adaptive Embeddings for SQL Schema Retrieval

**arXiv ID:** 2607.13311 | [PDF](https://arxiv.org/pdf/2607.13311v1)

**作者:** Qingcheng Zeng `[一作]` (Northwestern University), Rajhans Samdani `[通讯]` (Snowflake Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将“模式检索”（schema retrieval）视为独立的文档检索任务，针对SQL查询在大型数据库中的表和列级别检索。

**💡 创新点**

创新点在于：①构建跨数据集的检索基准；②提出无标签、基于语料的自适应微调（利用LLM生成查询、粒度感知的硬负样本挖掘、对比学习）；③证明该方法对模型规模不敏感，可显著提升小模型性能并进一步提升大型模型。

**🔧 技术方法**

使用的技术包括：LLM生成查询、对比学习（contrastive fine‑tuning）、粒度感知硬负样本挖掘、Embedding模型（Arctic‑Embed‑M、Qwen3‑Embedding）以及BM25基准。

**📊 数据集**

数据集涵盖：Spider、BIRD、BEAVER、LiveSQLBench 及其大规模变体，分别代表学术、真实、企业及大规模数据库环境。

**📈 对比分析**

通过 recall@10 和 nDCG@10 与多种 0.1B–8B 参数的预训练模型及 BM25 进行对比，结果显示自适应微调后 305M 模型 recall@10 提升至 75.6，达到小于 1B 参数时最佳；同样方法亦将 8B 模型从 77.8 提升至 78.4，逼近甚至超过其他大型模型。

**⚠️ 局限性**

局限性包括：仅针对英文模式；需在适配时已知目标语料；使用 SQL 解析可能漏掉隐式列；LLM 生成的查询可能与真实用户提问风格差异；未验证方法是否适用于非 SQL 结构化检索。

---

## 138. Improving Medical Image Generative Models with Fréchet Distance Loss

**arXiv ID:** 2607.13300 | [PDF](https://arxiv.org/pdf/2607.13300v1)

**作者:** Andrew Marshall `[一作]` (Yale University), James Duncan `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究在分割引导的医学图像扩散生成模型中加入Fréchet距离损失，对模型进行微调，以提升合成肿瘤图像的结构真实性；

**💡 创新点**

创新点在于将Fréchet距离作为训练损失函数而非仅评估指标，并通过队列化统计方法实现小样本下的稳定FD估计；

**🔧 技术方法**

主要技术包括：分割引导扩散生成器、Fréchet距离损失（FD‑loss）、冻结特征提取器（InceptionV3、RadInceptionV3、BiomedCLIP、MedDINOv3）、基于队列的FD统计、以及基于生成数据的U‑Net分割训练；

**📊 数据集**

使用了三大医学图像数据集：腹部CT的LiTS、MCT‑LTDiag肝癌与转移瘤数据集，脑部MRI的BraTS多参数影像；

**📈 对比分析**

通过对比无增强、几何增强、无FD的合成数据增强以及FD增强的数据增强，在多项分布式指标（FID、KID、CMMD、FRD、IS）及分割指标（Dice、ASSD）上，FD‑loss训练的生成模型显著降低FID与FRD、提升Dice 5%~10% 以上，表明生成图像更逼真且对下游分割任务更有帮助；

**⚠️ 局限性**

局限性包括：仅在二维切片上验证，缺乏三维体数据评估；需要额外的队列维护和特征提取器冻结，训练成本相对较高；对不同病种与模态的最佳特征提取器选择仍需经验性调优。

---

## 139. FOLIO: Focused Semantic Memory for Streaming Video Understanding

**arXiv ID:** 2607.13298 | [PDF](https://arxiv.org/pdf/2607.13298v1)

**作者:** Haoyang Fan `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了训练无关的焦点语义记忆系统，用于在线流视频理解。

**💡 创新点**

创新点在于动态焦点驱动的细节写入、实体中心结构化长期记忆与视觉证据缓存的混合存储，以及轻量级混合检索与语义扩展。

**🔧 技术方法**

主要技术包括视觉-语言模型写手、关键帧选择、焦点状态更新、短期视觉缓冲、实体链式结构化记忆、视觉证据缓存以及检索时的语义查询扩展。

**📊 数据集**

使用的基准数据集为OVO-Bench和StreamingBench。

**📈 对比分析**

与OASIS、Think-While-Watching等基线比较，Qwen3-VL-8B版本在OVO-Bench Perception/Backward分别达82.0/69.1，StreamingBench整体准确率74.5，明显优于基线，并显著降低记忆写入成本。

**⚠️ 局限性**

局限在于写手推理吞吐量瓶颈、固定片段划分缺乏事件驱动、记忆字段未针对特定领域（如计分板、所有权变化）优化，以及检索可能缺乏足够细粒度的证据。

---

## 140. Aurora DSQL: Scalable, Multi-Region OLTP

**arXiv ID:** 2607.13276 | [PDF](https://arxiv.org/pdf/2607.13276v1)

**作者:** Marc Brooker `[一作]` (Amazon Web Services), Matthys Strydom `[通讯]` (Amazon Web Services)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了 Aurora DSQL，一种面向 OLTP 的无服务器、跨区域、强一致性、可水平扩展的 SQL 数据库；

**💡 创新点**

创新点包括将数据库拆分为查询处理器、存储节点、裁决器、Journal 等无状态服务，并通过精确时间戳、MVCC、单轮提交和裁决器的 OCC 实现跨区域低延迟强一致；

**🔧 技术方法**

技术手段涵盖 PostgreSQL 引擎在 Firecracker MicroVM 中运行、精确时间戳与 MVCC、裁决器驱动的 OCC、Journal 复制与交叉条带、纠删编码、TLA+ 形式化验证及确定性仿真；

**📊 数据集**

使用了基于 TPC‑C 的合成基准（包含多区域版本）以及 Amazon 内部真实工作负载来评估系统性能；

**📈 对比分析**

通过与竞争产品（采用悲观锁定的系统）和 Spanner 的微基准比较，显示 DSQL 在读写延迟上分别低 10 倍和 4 倍，跨两区提交延迟约 30 ms（单区 7.4 ms），吞吐量可达百万 TPS，TPC‑C 评估中热启动后 5 分钟即可达到 85% 峰值吞吐；

**⚠️ 局限性**

局限性包括单个事务最大 3,000 行/10 MiB 的大小限制、最初不支持外键约束、序列与低基数索引的写热点挑战、以及在写入偏移时可能出现的写偏差异常，需要严格的时钟同步和分片/复制策略。

---

## 141. Deconstructing Actor-Critic: A Large-scale Empirical Study of Design Components for Practitioners

**arXiv ID:** 2607.13274 | [PDF](https://arxiv.org/pdf/2607.13274v1)

**作者:** Haseeb Shah `[一作]` (University of Alberta), Martha White `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在真实水处理设备的 PID 调节环境中，使用 33,000+ 次实验系统评估 actor‑critic 算法的设计组件（演员更新方式、参数化、梯度估计、更新频率等）对性能、稳定性和超参数敏感度的影响。

**💡 创新点**

系统拆解 actor‑critic 的关键组件，并通过大规模实验证明不同设计选择（bounded 与 unbounded 策略、路径式与似然比梯度、λ 混合、自适应 critic 更新等）对可靠性与可调性的重要性，指出 beta 分布策略在性能‑稳定性前沿，且路径梯度对 bounded 策略不稳定。

**🔧 技术方法**

采用多种 actor‑critic 算法（PPO、SAC、DDPG、MPO、GreedyAC、REINFORCE），使用不同梯度估计（PW、LR）与策略参数化（Gaussian、beta、Student‑t、squashed Gaussian），配合自适应 critic 更新、λ 混合和 UTD 比例，并在 Python/torch 环境下完成实验。

**📊 数据集**

使用基于真实饮用水处理厂传感器数据拟合的二阶多项式仿真器，模拟水泵背冲控制过程；实验基于自建数据集，没有使用公开标准数据集。

**📈 对比分析**

通过平均奖励和系数变异（CV）评估性能与稳定性，在 10 个随机种子和 10 个最佳超参数组合下比较；结果显示 beta 策略与 PPO 在性能‑稳定性平衡上表现最佳，其余常用默认配置（squashed Gaussian、λ=0）表现较差，路径梯度对 beta 不敏感但对 clipped Gaussian 不稳定。

**⚠️ 局限性**

仅在无状态的 bandit‑式 PID 调节任务中验证，未扩展到一般 MDP 环境；自适应 critic 更新假设误差可直接衡量，复杂环境中可能不足；缺乏对非平稳场景的评估；超参数搜索范围和算法实现细节可能限制了结论的普适性。

---

## 142. The tragedy of the cognitive commons: collective intelligence beyond AI-induced knowledge collapse

**arXiv ID:** 2607.13272 | [PDF](https://arxiv.org/pdf/2607.13272v1)

**作者:** Maher Kallel `[一作]`, Mohamed El Louadi `[通讯]` (University of Tunis)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并批判了Acemoglu等人提出的AI导致知识坍塌模型，提出结构性批评并给出测量与治理建议。

**💡 创新点**

将模型的前提拆解为知识互补性、努力弹性与AI对公共知识的贡献，并指出可测量的关键参数，提出基于公共信号生成的指标。

**🔧 技术方法**

理论模型（动态外部性）、计量经济学方法（差分中的差分、事件研究）、系统分析。

**📊 数据集**

Stack Overflow问答平台数据、EEG与语言分析实验数据、医学/法律等专业领域数据、生成模型训练数据。

**📈 对比分析**

通过与经验观察（Stack Overflow 下降、EEG认知负债、生产力提升研究）对比验证模型预测；模型未给出精确数值预测，性能主要体现在解释性和可检验性上。

**⚠️ 局限性**

主要限制在于固定知识分类、AI不产生通用知识假设、努力弹性未测量、数据难以区分公共信号衰减与迁移、政治可行性低。

---

## 143. Discourse-Aware Policy Analysis with Argumentation: A Hybrid LLM-Symbolic Framework for Disaster Governance

**arXiv ID:** 2607.13260 | [PDF](https://arxiv.org/pdf/2607.13260v1)

**作者:** Stylianos Loukas Vasileiou `[一作]` (New Mexico State University), Olga Derendiaeva `[通讯]` (Sun Yat-Sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个混合LLM与符号规则的管道Apaf，自动将灾难治理政策文本转换为可解释的定量双极论证框架。

**💡 创新点**

首次将批判性话语分析的治理框架转化为四种框架介导的攻击/支持关系，并用可追溯的规则生成图谱。

**🔧 技术方法**

结合大型语言模型的结构化抽取、框架判定与特征检测，以及符号规则推断、DF‑QuAD渐进语义，形成可审计的论证图。

**📊 数据集**

使用了来自美国、英国、加拿大、澳大利亚四国的100份灾害风险减缓子文档，手工标注了论点、框架与关系。

**📈 对比分析**

与仅使用LLM的消融实验对比，规则驱动在检测、极性、子类型三层上分别提升30–38点，整体检测F1 0.73、子类型F1 0.58，框架识别准确率0.86。

**⚠️ 局限性**

局限于英语民主国家、双框架简化、LLM误差未量化、可解释性仅限于规则链，未涵盖更丰富的治理理性与非西方语境。

---

## 144. PUe: Biased Positive-Unlabeled Learning Enhancement by Causal Inference

**arXiv ID:** 2607.13428 | [PDF](https://arxiv.org/pdf/2607.13428v1)

**作者:** Xutao Wang `[一作]` (Huawei), Yunhe Wang `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了 PUe 框架，将归一化逆概率加权与深度倾向分数估计相结合，改进了在标签偏差下的正负样本不平衡学习。

**💡 创新点**

创新点包括：归一化逆概率加权（NIPW）用于纠正 SAR 机制下的标签偏差；利用深度网络估计倾向分数并加入正则化；将该倾向分数校正与现代成本敏感 PU 方法（uPU、nnPU、PUbN、Dist-PU）整合，并支持选择性负样本标注。

**🔧 技术方法**

技术手段主要是逆概率加权、深度倾向分数网络（带正则化）、成本敏感 PU 学习、正则化策略、以及对比实验评估（ACC、Prec、Rec、F1、AUC、AP 等）。

**📊 数据集**

实验使用 MNIST、CIFAR-10、以及阿尔茨海默病影像数据库 ADNI 三个公开数据集。

**📈 对比分析**

与传统 PU 基线（uPU、nnPU、PUbN、Dist-PU）以及其各自改进版进行对比，实验表明 PUe 在大多数指标上提升约 1%–5%，尤其在高偏标签分布下表现更为显著。

**⚠️ 局限性**

局限性包括：对倾向分数估计的准确性高度敏感；当标签分布偏差过大导致某些类别缺乏标记样本时，倾向分数可能接近 0，影响模型性能；依赖 SAR 假设，若真实标签机制更复杂，方法效果可能受限。

---

## 145. ScanFocus: A Coarse-to-Fine Framework for Spatio-Temporal Video Grounding

**arXiv ID:** 2607.13421 | [PDF](https://arxiv.org/pdf/2607.13421v1)

**作者:** Kai Chen `[一作]` (Southeast University), Wankou Yang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ScanFocus 框架，将时空视频定位任务拆分为全局粗扫描和局部边界聚焦两阶段，利用统一 vision‑language 编码器、轻量化 Deformable Semantic‑Motion Fusion 以及 Semantic‑Guided Temporal Aggregator 实现精细时间与空间定位；

**💡 创新点**

创新点在于（1）粗细分层策略：先低帧率粗定位后密集边界采样以恢复高频时序信息；（2）通过 Deformable Attention 解耦视觉、语言与运动特征，避免全连接编码器的优化瓶颈；（3）SGTA 在局部窗口内显式建模短期时序依赖并受语义引导，显著提升边界精度；

**🔧 技术方法**

使用技术包括 BEiT‑3 统一 vision‑language 编码器、VideoMAE 运动编码、DETR‑style 双解码器、Deformable Attention、Semantic‑Guided Temporal Aggregator、RoI Pooling 以及分阶段训练策略；

**📊 数据集**

使用的数据集包括 HC‑STVGv1、HC‑STVGv2 以及 VidSTG；

**📈 对比分析**

与现有方法对比，ScanFocus 在 HC‑STVGv1 上 vIoU@0.3 提升至 67.5%（+4.4%），vIoU@0.5 提升至 42.2%（+5.4%）；在 HC‑STVGv2 上 m_tIoU +2.0%、vIoU@0.5 +2.6%；在 VidSTG 上声明式与问句两类均超越 SOTA，m_vIoU 提升约 1.8% 以上；

**⚠️ 局限性**

限制方面：局部窗口大小需人工设定，模型对极长视频时序扩展仍不够稳定；虽然采用轻量化 Fusion，稠密采样阶段仍存在计算负担；模型性能仍受前置 encoder 语义对齐质量的影响。

---

## 146. MultiAnimate: A Unified Framework for Controllable Multi-Character Animation

**arXiv ID:** 2607.13415 | [PDF](https://arxiv.org/pdf/2607.13415v1)

**作者:** Zhongyi Zhang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MultiAnimate 框架，能够使用多张参考图像和对应的姿势序列，在同一共享场景中并行生成多角色动画视频。

**💡 创新点**

创新点包括：①身份专属参考网络（identity‑specific reference net）加入可学习位置编码，支持多参考图；②身份感知姿势编码器（identity‑aware pose encoder）通过跨注意力绑定身份与姿势；③交互引导模块（interaction guider）利用角色掩码处理遮挡与空间关系；④将背景分支与滑动窗口长视频推理结合。

**🔧 技术方法**

技术手段：在 AnimateDiff 的 3D UNet 基础上扩展潜在扩散模型，加入位置编码、跨注意力、交互引导；使用 VAE、SAM2、DWPose、PoseNet、DiT 等模型；实现身份专属参考网络、身份感知姿势编码器和交互引导模块。

**📊 数据集**

数据集：自建多角色交互视频集（超过10k段视频），使用 SAM2 生成掩码、DWPose 提取姿势；单角色基准使用公开的 TikTok 数据集。

**📈 对比分析**

与多种基准方法（Moore‑Animate、MimicMotion、MagicPose、MagicAnimate、UniAnimate 等）在 SSIM、PSNR、LPIPS、MSE、FID、FVD 等指标上进行定量评估；MultiAnimate 在所有指标上均优于对比方法（例如 SSIM 0.801、PSNR 21.71、LPIPS 0.192、FVD 229.06）。用户研究显示 95% 以上受试者更倾向于 MultiAnimate 的结果。

**⚠️ 局限性**

局限性：仍出现细节区域（如手部）伪影；基于 UNet 的扩散生成器性能受限，未来需要结合更强的 DiT 架构；由于可用于伪造内容，需在部署时加入水印等防伪措施。

---

## 147. Evaluating Frontier AI Agents as Autonomous Clinical Security Auditors

**arXiv ID:** 2607.13411 | [PDF](https://arxiv.org/pdf/2607.13411v1)

**作者:** Michael O. Eniolade `[一作]` `[通讯]`, Michael O. Eniolade

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并公开了一个基于 METR Task Standard 的开放式评估任务，要求前沿语言模型在无代码提示的情况下自动实现四种对抗攻击（FGSM、成员推断、校准误差、边界攻击）并计算安全姿态得分（SPS）。

**💡 创新点**

首次将结构化的临床 AI 安全评估规范包装成可被语言模型自主完成的多步骤工具使用任务，并提供六个任务变体和完整评估基础设施；同时对三种前沿模型在该任务中的表现进行了系统对比。

**🔧 技术方法**

使用前沿语言模型（Claude Sonnet 4.6、GPT‑4.1、GPT‑4o）配合 Bash 工具执行 Python 代码；实现对抗攻击与校准计算；利用 METR 评分器（基于加权公式）自动验证结果；对 Token 计数与 API 成本进行记录。

**📊 数据集**

采用 Wisconsin Diagnostic Breast Cancer (WDBC) 公开数据集和 MIMIC‑IV ICU 死亡率数据集（需 PhysioNet 凭证）；每个数据集配备三种模型架构：逻辑回归、随机森林+Platt 归一化、XGBoost+对抗训练。

**📈 对比分析**

通过对每个模型执行 18 次（3 次/6 变体）评估，并对完成率、SPS 分数、Token 消耗和 API 成本进行统计；Claude 与 GPT‑4.1 完全通过（score = 1.0），GPT‑4o 完成率仅 61%，并在 Token 使用和费用上明显高于前两者。

**⚠️ 局限性**

局限包括：仅测试三种模型、仅覆盖浅层分类器（未适配深度学习模型）、对 MIMIC‑IV 需要凭证、评价容差对接近零的指标过宽、任务依赖伪代码限制方法学评估、样本量有限导致不确定性以及缺乏对更复杂攻击场景的覆盖。

---

## 148. Ego-Dynamics-Augmented World Model for Autonomous Driving with Zero-Shot Cross-Chassis Adaptation

**arXiv ID:** 2607.13410 | [PDF](https://arxiv.org/pdf/2607.13410v1)

**作者:** Zhidong Wang `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出DynaDreamer，一种在BEV世界模型中加入显式惯性动力学先验的强化学习框架

**💡 创新点**

通过把车辆惯性动力学提取为可识别的上下文并直接注入Transformer世界模型的先验与后验，显著降低观察转移的不确定性，提升样本效率，并实现零样本跨底盘迁移

**🔧 技术方法**

物理信息驱动的神经ODE编码解码器、AdaLN与RoPE调制的Transformer世界模型、梦想者（Dreamer）风格的模型基强化学习、可多步辅助预测与参数辨识损失

**📊 数据集**

CARLA仿真平台的Town03（城市）和Town04（高速）场景，使用17种不同底盘（轿车、SUV、面包车、卡车）进行训练，另外两种未见车辆（巴士、微型车）用于零样本评估

**📈 对比分析**

与SAC、DreamerV3、STORM、VD-STORM以及若干消融组进行对比；在城市和高速两种任务中，DynaDreamer在成功率、碰撞率、延迟、奖励等指标上分别提升约28%–61%，在未见底盘上提升73%；在世界模型的KL、BEV重建、低维状态预测等度量上也取得显著优势

**⚠️ 局限性**

目前仅在仿真环境验证，依赖BEV视角和准确的物理模型，可能对真实传感器噪声、时延和多车交互的泛化有限；消融实验表明对物理编码和AdaLN调制的依赖较大，缺少这些模块会显著退化性能

---

## 149. AnomExpert: Identifying and Selecting Anatomical Planes for Prenatal Ultrasound Anomaly Diagnosis

**arXiv ID:** 2607.13409 | [PDF](https://arxiv.org/pdf/2607.13409v1)

**作者:** Jian Wang `[一作]` (Shenzhen University), Dong Ni `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出AnomExpert框架，在案例级监督下实现产前超声异常诊断，自动识别解剖平面并选择病种相关平面；

**💡 创新点**

创新点在于使用可学习平面原型进行弱监督平面识别，并通过疾病感知稀疏选择机制聚焦诊断相关平面，避免分阶段流程与平面标注；

**🔧 技术方法**

技术包括ViT骨干网络、Sinkhorn平衡原型分配、疾病查询向量与偏置矩阵、稀疏top‑k聚合以及整体端到端MIL训练；

**📊 数据集**

使用来自24个医疗中心的多中心产前超声数据集，3,654例、61,460张图像，涵盖8种致死先天性异常与正常对照；

**📈 对比分析**

与9种主流MIL方法对比，ViT‑s版AnomExpert在准确率86.9%、F1 84.2%、AUC 97.9%上优于基线，并且参数量更少；

**⚠️ 局限性**

局限包括对多标签诊断的支持有限、对极端解剖变异鲁棒性待进一步验证，以及对更大规模数据集的可扩展性需进一步评估。

---

## 150. WNOJ-LIO: A White-Noise-on-Jerk Motion-Prior EKF for High-Dynamic LiDAR-IMU Fusion

**arXiv ID:** 2607.13405 | [PDF](https://arxiv.org/pdf/2607.13405v1)

**作者:** Junning Lyu `[一作]` (Beijing Institute of Technology), Shaoming He `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 WNOJ-LIO 框架，利用 White‑Noise‑on‑Jerk (WNOJ) 运动先验在 R³×SO(3) 上解耦后进行状态预测，并把 IMU 作为高频 EKF 测量更新；随后使用后验轨迹对 LiDAR 进行去畸变和点对平面更新，实现高动态运动下的 LiDAR‑IMU 融合。

**💡 创新点**

创新点包括：① 将 WNOJ 先验从批量 GP 转为递归 EKF 可用的解耦过程模型；② 将 IMU 视为测量而非预测驱动，显著抑制振动噪声传播；③ 采用双频 EKF 更新（IMU 200 Hz，LiDAR 20 Hz）并利用后验姿态历史进行点云去畸变；④ 通过闭式协方差传播实现高效实时估计。

**🔧 技术方法**

使用技术包括 White‑Noise‑on‑Jerk 高阶 GP 先验、递归 EKF、Lie 群几何表示、IMU 预积分作为测量、点对平面 LiDAR 注册、双频 EKF 更新与闭式协方差计算。

**📊 数据集**

数据集：① 公开仿真环境（40 × 50 × 10 m 立方体）配 20 个平面标定点；② 真实赛车赛段四段（Yas Marina Circuit）同步 LiDAR、VectorNav IMU、INS/GNSS、Kistler 速度计与 Bosch 振动隔离 IMU，用于加速度、角速度、速度、姿态与位置评估。

**📈 对比分析**

对比方法：与 FAST‑LIO 风格基线（IMU 预测驱动、LiDAR 低频更新）在仿真与真实数据中比较。结果表明：WNOJ‑LIO 在加速度/角速度去噪、点云去畸变、定位误差（位置、速度、姿态）均优于 FAST‑LIO；在四段赛车测试中，WNOJ‑LIO 的总加速度/角速度 RMSE、身体速度 RMSE、姿态 RMSE 均低，RTK 定位位置误差在大多数段落也优于基线。

**⚠️ 局限性**

局限性：① 未对 IMU 偏置进行建模，依赖于测量残差补偿；② WNOJ 噪声参数需手动调优，适应性不足；③ 仅验证了 LiDAR‑IMU 融合，未与视觉或多模态传感器结合；④ 对极端高动态场景仍需更大窗口或更复杂的模型来进一步提升鲁棒性。

---

## 151. Set-shifting Behavioral Test for Harnessed Agents

**arXiv ID:** 2607.13396 | [PDF](https://arxiv.org/pdf/2607.13396v1)

**作者:** Ziwei Ye `[一作]` `[通讯]`, Ziwei Ye

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套基于隐藏可靠性变化的LLM工具选择基准，借鉴认知心理学的Set‑Shifting测试，在持续会话中评估代理对工具组的适配与固化行为。

**💡 创新点**

创新点包括：① 构建冗余工具-技能集并在隐藏边界处切换可靠工具组；② 采用分支树结构的多阶段调度，配对不变控制；③ 定义Set‑Shifting Accuracy指标和路由类分析，量化代理在每个后置窗口的目标工具占比；④ 通过政策提示和集合框架干预探究行为可塑性。

**🔧 技术方法**

技术手段：使用Hermes Agent开放源工具库和会话持久化；采用两款开源大模型（Mistral‑7B、DeepSeek‑v4‑Pro）；实现自定义政策提示（Adaptive、Polymath）和集合框架（竞争 vs. 互补）；通过Python脚本自动生成工具描述、技能及调度树；利用bootstrap CI与配对置换检验评估指标。

**📊 数据集**

数据集：三种模拟域（调度、DevOps事件 triage、跨云存储），每域包含3–5个工具组，每组5或2个冗余工具；构造的工具集与技能文件（Skill‑MD）和模拟执行器；调度树包含3个分支层，共9个终端；对每个终端生成16个轨迹并手工审核。

**📈 对比分析**

比较方法：对每条轨迹计算Set‑Shifting Accuracy、路由类比例与任务完成率；在同一树结构下对模型、政策提示与集合框架进行配对比较；通过Bootstrap CI、配对置换p值与符号检验评估差异。结果显示：在无干预下，两模型大多陷入固定套路；Adaptive提示可显著提高目标占比至≥0.86；竞争框架可将Φ提升约0.15–0.30，互补框架则趋向混合套路。

**⚠️ 局限性**

局限性：仅测试两款模型，缺乏更广泛的模型覆盖；基准使用模拟工具与简单的中性提示，未检验自然语言变化或真实API的复杂性；长期记忆压缩对锁定模式的影响未探究；缺少跨域迁移与自适应学习机制的评估。

---

## 152. Self-Improving is Often Sudden: Enlightenment-style Finetuning for Large-Scale Models

**arXiv ID:** 2607.13395 | [PDF](https://arxiv.org/pdf/2607.13395v1)

**作者:** Jing-Xiao Liao `[一作]` (City University of Hong Kong), Feng-Lei Fan `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的“启蒙”式微调方法，通过在推理阶段对大型预训练模型的网络结构（LLM 的注意力头混合短路、VLM 的标量调制残差）进行修改，以实现自我提升，且不需要梯度、数据或权重更新。

**💡 创新点**

创新点包括：① 以神经科学中的“启蒙”现象为灵感，首次把模型自我提升视为结构改造而非参数微调；② 为 LLM 和 VLM 分别设计了专属的 shortcut 机制；③ 引入自适应 Chebyshev 多项式缩放函数，动态平衡注意力权重；④ 通过跨头信息混合与残差调制显著扩展表示能力。

**🔧 技术方法**

使用的技术主要有 Transformer 架构、注意力头混合短路、标量调制残差、Chebyshev 多项式自适应缩放、ZeroTuning 作为基线。

**📊 数据集**

评测数据集涵盖多种 LLM benchmark（SST‑2、CB、BoolQ、ARCE、ARCC、CQA、PIQA、RTE、WinoGrande、MMLU）以及压缩模型评测；VLM benchmark 包括 MMBench、MME、AI2D、RealworldQA、MMMU。

**📈 对比分析**

与 ZeroTuning（仅权重微调）对比，Enlightenment 在 LLM 上平均提升 1.5–1.9%（如 Qwen3.5‑9B 80.4%→81.8%），在 VLM 上平均提升 1.9–3.06%（如 Qwen3‑VL‑8B 84.0%→87.06%）。在量化模型上也能恢复部分精度，整体性能提升稳定且显著。

**⚠️ 局限性**

局限性：① 对模型类型特定，头混合不适用于 VLM，残差调制不适用于 LLM；② 仍需调节少量超参数；③ 目前仅针对 decoder 侧的 Transformer，未验证 encoder‑only 或其他架构；④ 推理时增加了轻微计算开销；⑤ 对极端量化噪声和多任务动态适配的效果尚不充分。

---

## 153. Where Should RL Post-Training Compute Go? Model Size, Search, Learning, and Feedback

**arXiv ID:** 2607.13389 | [PDF](https://arxiv.org/pdf/2607.13389v1)

**作者:** Patrick Wilhelm `[一作]` (BIFOLD), Odej Kao `[通讯]` (BIFOLD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在固定后训练预算下RL后训练的计算分配问题，提出FLOP计量框架并探索最佳分配前沿

**💡 创新点**

首次将后训练成本拆解为rollout、学习、奖励评估三部分，并证明模型大小、奖励设计与评估目标共同决定最优分配；提出RACE诊断协议用于小规模实验预测最佳分配

**🔧 技术方法**

使用GRPO算法、LoRA微调的Qwen2.5模型，采用FLOP计量与自定义的update‑fraction ρ

**📊 数据集**

在Polaris‑53K数学推理数据集上训练，随后在GSM8K和MATH‑500上评估最终答案、符号相等、Pass@1及共同评判指标

**📈 对比分析**

通过IsoFLOP网格搜索比较不同模型尺寸、奖励系统（稀疏、结构化、密集、PRM）下的分配前沿；结果显示不同目标（原生奖励、下游准确率、共同评判）偏好不同的ρ；RACE能够准确识别高/低update区域，但未保证在验证集上绝对提升

**⚠️ 局限性**

实验仅针对数学推理作为代理任务，使用单一模型族和少量种子；FLOP估计仅用于相对比较，未考虑所有硬件效应；RACE为诊断工具而非最终优化方法

---

## 154. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations

**arXiv ID:** 2607.13399 | [PDF](https://arxiv.org/pdf/2607.13399v1)

**作者:** Rui Wang `[一作]` (Chinese University of Hong Kong), Kam-Fai Wong `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地研究了大语言模型后训练中的 On‑Policy Distillation（OPD），从探索催化剂的角度阐释其作用，并在实验中揭示了两类主要失效现象——学生-教师不匹配与长度剥削；进一步提出了两种轻量级的信号调节策略（硬裁剪与对数压缩），在不增加额外计算开销的前提下显著提升了 OPD 的稳定性与效果。

**💡 创新点**

① 明确 OPD 主要是提升探索效率而非扩展能力上限；② 识别出学生-教师分布差异导致的信号误导与 token‑level 加权导致的长度短路两大致命路径；③ 通过硬裁剪与对数压缩两种内循环信号规范化方法，消除上述路径，证明信号质量决定成功而非教师规模。

**🔧 技术方法**

利用 OPD 的 token‑level 逆 KL 目标与策略梯度，采用 r_t/importance ratio 裁剪与自定义信号调节；在实验中对 Qwen3 系列模型进行微调，使用 1.7B-Base 作为学生，4B/8B/30B 教师；评估采用 pass@k、avg@32 等指标。

**📊 数据集**

训练使用 Nemotron‑Cascade Math 数据集；评估基准包括 AMC23、AIME 2024‑26、HMMT25 Feb、MATH500 以及 Minerva。

**📈 对比分析**

与原始 OPD、RLVR、GRPO、传统离线 KD 以及 30B 级教师的对比实验表明，加入裁剪或对数压缩的 OPD 在所有衡量指标上均超过基线，且在 1.7B 学生上能击败使用更大 30B 教师的 distillation，验证了信号质量的重要性。

**⚠️ 局限性**

调节方法依赖手工设定的阈值或剪裁区间，需要根据教师-学生容量差异进行实验调优；实验集中在可验证的数学推理任务，对开放式生成或知识密集型 QA 的适用性尚未得到验证。

---

## 155. TMallGS: Scaling Unified Feature and Sequence Modeling for Generative E-commerce Search

**arXiv ID:** 2607.13398 | [PDF](https://arxiv.org/pdf/2607.13398v1)

**作者:** Zhentao Song `[一作]` (Southeast University), He Guo `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为TmallGS的可扩展生成式搜索排序架构，专为解决传统DLRM在工业搜索中的内存瓶颈和特征异质性问题而设计。

**💡 创新点**

创新点包括：①分层分布校准分词（FSR+DCP）实现高效异质特征编码；②字段自适应门控Transformer骨干（Per-Field QKV + Noise-Adaptive Gating）提升语义交互与噪声抑制；③解耦FiLM后期融合，恢复高频显式匹配信号；④上下文感知偏置网络正则化系统性偏差；⑤错误感知渐进训练实现自适应难样本挖掘。

**🔧 技术方法**

技术主要有Transformer、FiLM、门控注意力、分布校准投影、分层特征加权、上下文注意力掩码、进阶多任务损失（BCE+Pairwise）、两阶段warm‑up、稀疏+密集混合优化。

**📊 数据集**

使用天猫App搜索日志，31天共约5亿样本，平均行为序列长度1500，覆盖13M用户、74M商品。

**📈 对比分析**

与传统DLRM（DNN、DIN、DCNv2、APG）以及可扩展Transformer架构（HiFormer、RankMixer、HHFT、HSTU、OneTrans等）对比，TmallGS在AUC和GAUC上分别提升约1.12%和1.26%，在线A/B测试显示GMV +1.52%、UCTCVR +1.38%，仅+6ms延迟。

**⚠️ 局限性**

局限包括：需要大规模工业数据才能充分验证，模型规模仍受GPU内存限制，超大规模（十亿级）扩展尚未实现，对多模态生成能力探索不足。

---

## 156. TANDE: Disentangling Verbal and Nonverbal Backchannels in Emotional AI-Avatar Conversations with Young Adults

**arXiv ID:** 2607.13357 | [PDF](https://arxiv.org/pdf/2607.13357v1)

**作者:** Ann-Kareen Gedeus `[一作]` (Cornell University), Angelique Taylor `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究构建了基于LLM的情感会话代理TANDE，并在36名18–25岁年轻人中通过3×3 交叉设计（对照、非语言、语言+非语言回声信号）检验不同回声方式对建立融洽、共情与参与度的影响。

**💡 创新点**

创新点在于：①首次在真实开放式情感对话中独立控制回声媒介并系统评估其对用户体验的作用；②揭示性别差异在对回声的感知与评价中的重要性；③为情感支持型ECAs提供基于数据驱动的设计建议。

**🔧 技术方法**

核心技术包括：GPT‑4o‑mini 生成情感适配对话；FER+ 视觉情绪识别模型（ONNX+WebAssembly）实现实时情绪感知；基于CANDOR的定量回声触发器；三维面部动画、语音合成（ElevenLabs/TTS）与口型同步。

**📊 数据集**

使用了公开的 CANDOR 对话语料（校正回声速率）和 FER+ 面部表情数据集（训练情绪识别模型），以及本实验收集的36名参与者的对话与问卷数据。

**📈 对比分析**

采用 within‑subject 复测方差分析与混合设计 ANOVA 进行比较，结果显示三种回声模式在融洽、共情与参与度上无显著差异；但性别差异显著，女性在融洽和共情评分上高于男性。

**⚠️ 局限性**

局限性包括：①回声触发器为固定速率、无内容适配，可能导致不自然或时机不当；②实验时间短，仅为约7–8分钟，未评估长期互动效果；③仅使用单一女性化ECAs，未探究性别匹配；④样本量相对有限，可能影响检验效能。

---

## 157. CLIP-Guided Label-Free Discriminative Region Scoring for Fine-Grained Classification

**arXiv ID:** 2607.13437 | [PDF](https://arxiv.org/pdf/2607.13437v1)

**作者:** Yujie Zhu `[一作]` `[通讯]` (State University of New York at Buffalo), Yujie Zhu (State University of New York at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种统一的基于CLIP的无标签局部区域评分框架，用于细粒度图像分类；

**💡 创新点**

创新点包括：1）引入多种判别评分策略（硬负边缘、软负边缘、熵置信度）提升对视觉相似类别的区分；2）使用伪标签实现无监督的区域评分；3）系统比较随机裁剪与SAM分割的效果；

**🔧 技术方法**

采用的技术主要是：CLIP视觉-文本嵌入、SAM分割模型、冻结CLIP特征、轻量线性分类器以及各种评分公式；

**📊 数据集**

实验使用五个细粒度数据集：CUB‑200‑2011、Oxford‑102 Flowers、Oxford‑IIIT Pets、Stanford Cars、FGVC‑Aircraft；

**📈 对比分析**

与多种基线（全局、平均局部、不同评分）对比，结果显示软负边缘评分在无标签和伪标签下均表现最佳；随机裁剪在所有数据集上均优于SAM分割，伪标签评分与真实标签相差不大；

**⚠️ 局限性**

局限性包括：依赖冻结的CLIP表示，若CLIP在特定领域缺乏区分度会导致性能下降；SAM分割耗时且易产生纯背景掩码，影响效果；

---

## 158. Exploring Post-Training Alignment of Small Language Models for Biomedical Data-to-Text Generation: A Case Study of Medication Leaflet

**arXiv ID:** 2607.13430 | [PDF](https://arxiv.org/pdf/2607.13430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 159. ε-Indistinguishability In Moving Target Defense: Framework, Algorithms, And Cloud Case Studies

**arXiv ID:** 2607.13440 | [PDF](https://arxiv.org/pdf/2607.13440v1)

**作者:** Sailik Sengupta `[一作]` (Amazon Science), Ankur Chowdhary `[通讯]` (Intuit)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文把MTD候选池的安全性定义为在给定精度ε下能形成的最大ε-近似子集，并提出一套四种规模覆盖的算法来构造该子集；随后在真实的AWS云实验中测量了两种不同配置轮换（单一运行时与三层堆栈）的可匿名性，并给出了诊断方法。

**💡 创新点**

创新点在于：①将池安全性正式化为寻找最大的ε-近似子集；②利用加法模型把多维组合问题降为一维密集窗口问题；③设计了全枚举、Meet‑in‑the‑Middle、FFT卷积和Monte Carlo采样四种互补算法，覆盖10^1–10^38规模；④提出按轴的可见度分类与部署级别的“可匿名性诊断”，可直接从测得的延迟间隙判断哪些轴真正贡献匿名性。

**🔧 技术方法**

主要技术包括：加法效用模型与OLS拟合；二维滑动窗口与两指针求密集区间；Meet‑in‑the‑Middle 递归分割求和；FFT卷积对离散化直方图做k阶卷积；Monte Carlo采样并使用DKW不等式给出全局误差界；以及对算法的时间空间复杂度与误差分析。

**📊 数据集**

数据集：①合成数据集（Tiny、Small、Medium、Large、Huge）涵盖10^1–10^38配置，采用线性、平方根、二次三种效用分布；②真实云案例：AWS 4 版本Python的无服务器轮换（4配置）和3×3×3三层堆栈（27配置）。

**📈 对比分析**

比较方法：在可枚举范围内（≤2×10^6）与Exact、MitM、FFT、Sampling结果对比；FFT和Sampling在更大规模（8.4×10^15、1.1×10^38）下互相验证。性能表现：全枚举O(NlogN)在N≤10^7仅几微秒；MitM内存≤√N；FFT在10^38规模下仅30毫秒；Sampling约0.2秒；误差在可接受范围内（≤1%），FFT与Sampling在极大规模下误差<2%。

**⚠️ 局限性**

局限性：①假设效用完全加法，忽略组件间交互导致的非线性误差；②依赖准确的基准测量，噪声会导致实际可匿名性低于预测；③实验仅验证被动旁观者模型，未考虑主动定时攻击；④只在云内测量延迟通道，未对多通道、不同云区或跨网络的精度做完整评估；⑤并未给出整个轮换序列的分布式安全保证，只给出单观测的ε-近似。

---

## 160. Distributionally Robust and Safe Imitation Learning

**arXiv ID:** 2607.13436 | [PDF](https://arxiv.org/pdf/2607.13436v1)

**作者:** Ahmed Aboudonia `[一作]` (University of California Berkeley), Naira Hovakimyan `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分布鲁棒与安全的模仿学习框架，能够同时抵御策略诱发和不确定性诱发的分布漂移，并在训练中嵌入安全约束。

**💡 创新点**

创新点在于将Taylor系列模仿学习（TaSIL）与L₁分布鲁棒自适应控制（L₁‑DRAC）相结合，并在TaSIL损失函数中加入CVaR安全约束，形成既能优化性能又能保证安全的全新算法。

**🔧 技术方法**

使用的技术包括：Taylor系列模仿学习（TaSIL）、L₁分布鲁棒自适应控制（L₁‑DRAC）、分布鲁棒优化（Wasserstein球）、CVaR约束、GELU激活的多层神经网络。

**📊 数据集**

实验使用的“数据集”是10条专家轨迹（每条150个样本点）在无噪声与含随机正弦扰动（不确定性）的两种环境下收集的无人机仿真数据。

**📈 对比分析**

方法与标准TaSIL、TaSIL+L₁‑DRAC在相同仿真设置下进行对比；结果显示，在存在不确定性时，分布鲁棒安全TaSIL在保持对专家轨迹的逼近的同时能够成功避开危险区域，性能优于其他方法。

**⚠️ 局限性**

局限性包括：需要经验性或预估的分布不确定度上界 ρ_L，L₁‑DRAC 与 TaSIL 的协同设计尚未系统化，实验仅在离散时间仿真中验证，缺乏真实飞行或更复杂环境下的验证。

---

## 161. When Rubrics Change: Cross-Rubric Generalization for Critical Thinking Essay Scoring

**arXiv ID:** 2607.13433 | [PDF](https://arxiv.org/pdf/2607.13433v1)

**作者:** Nischal Ashok Kumar `[一作]` (University of Massachusetts Amherst), Andrew Lan `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究自动作文评分中的跨评分维度泛化问题，训练模型在已知评分维度上学习后对未见评分维度进行预测。

**💡 创新点**

首次系统研究跨评分维度泛化，提出“特征”(traits)这一跨维度中立中间表示，并探讨目标作文监督对泛化的影响。

**🔧 技术方法**

基于大型语言模型的微调框架，使用中间特征预测与监督、伪标签和金标注策略。

**📊 数据集**

采用PERSUADE 2.0数据集中的500篇学生论说文，包含六个评分维度（信息分析、论证生成、逻辑推理）。

**📈 对比分析**

与零-shot提示的开源模型和专有GPT‑5-mini/ GPT‑5进行对比，微调模型在未见评分维度上宏观F1提升约5%，在金标注下与GPT‑5-mini相当，远优于零-shot提示。

**⚠️ 局限性**

仅在单一批判性思维数据集上验证，特征生成为银标注且未人工验证；对低质量伪标签的噪声影响及更广泛评分维度的泛化尚需进一步探究。

---

## 162. GFlowRL: Scaling Distribution-Matching RL to Large Language Models

**arXiv ID:** 2607.13394 | [PDF](https://arxiv.org/pdf/2607.13394v1)

**作者:** Xiaodong Liu `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大型语言模型（LLM）推理的后训练阶段，提出一种简化版的Generative Flow Network（GFlowNet）强化学习算法，用以替代传统奖励最大化方法。

**💡 创新点**

创新点在于将原本必需的可学习分区函数 Zϕ 用在批内 Monte‑Carlo 估计替代，去除了额外的网络与同步开销，并引入重要性采样校正与不对称流差裁剪来提升训练稳定性。

**🔧 技术方法**

技术上采用 GFlowNet 的轨迹平衡目标（Trajectory Balance），配合 GRPO 的 rollout 组采样、长度归一化、stop‑gradient 基线以及两种稳定化技巧，实现了与原始 FlowRL 相同的奖励分布匹配目标。

**📊 数据集**

使用的数据集包括数学推理的 AIME、AMC、MATH‑500、Minerva、Olympiad 等；代码推理的 LiveCodeBench、Codeforces 与 HumanEval+；安全评测的 AdvBench、HarmBench；以及 MoE 规模的 Qwen3‑30B‑A3B 与 Qwen3‑235B‑A22B。

**📈 对比分析**

与 FlowRL、PPO、GRPO 等基线比较，本文方法在数学、代码和对抗红队任务上均超越对手，14B 模型在 Codeforces 获得 2048 Elo（仅比 o3‑mini 低 25 Elo），在 AdvBench/​HarmBench 的 ASR@1 上分别达 82.5%/79.5%，并能在 235B MoE 上顺利收敛，说明其在稠密与稀疏架构上均具可扩展性。

**⚠️ 局限性**

局限性包括：对分区函数的估计仍需依赖批内采样，可能在极端噪声奖励或极长推理链的场景下受限；算法对超参数（如流差裁剪阈值、β 温度）的敏感性未系统探索；以及在更复杂或多任务场景中的泛化能力仍待进一步验证。

---

## 163. The impact of objective interactions on the performance of massive objective optimization algorithms

**arXiv ID:** 2607.13377 | [PDF](https://arxiv.org/pdf/2607.13377v1)

**作者:** Shakiba Shahbandegan `[一作]`, Emily Dolson `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大规模目标数（最多100个）下多目标优化算法的性能，系统评估了不同目标交互对算法的影响。

**💡 创新点**

提出将目标交互（相互增强、正交、相互冲突）作为评估维度，并验证了lexicase选择在无参考方向需求下的竞争力。

**🔧 技术方法**

使用DOSSIER诊断测试套件、SBX交叉、多项式变异、lexicase、NSGA-II/III、MOEA/D等进化算法，并用IGD度量评估。

**📊 数据集**

采用DOSSIER诊断基准（可扩展至100维），对比不同维度和种群规模的实验。

**📈 对比分析**

通过IGD热图与对比实验发现，lexicase及其变体在大多数诊断中优于NSGA-II/III，尤其在目标无冲突或中等冲突时表现突出；NSGA-II在极度冲突情形下仍稳健。

**⚠️ 局限性**

仅限于无约束、变量数等于目标数的诊断问题，未涵盖约束、不同变量维度以及具有陷阱的多峰景观。

---

## 164. DiffGI: Differentiable Geometry Images for High-Fidelity Thin-Shell 3D Generation

**arXiv ID:** 2607.13365 | [PDF](https://arxiv.org/pdf/2607.13365v1)

**作者:** Eungjune Shim `[一作]` (CLO Virtual Fashion Inc.), Eunjung Ju `[通讯]` (CLO Virtual Fashion Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可微分几何图像（DiffGI）框架，用于高保真薄壳三维模型的生成和编辑。

**💡 创新点**

将几何图像转化为可微分的张量结构，允许端到端梯度传播，从而实现基于学习的几何编辑和细粒度形状控制。

**🔧 技术方法**

利用可微分几何图像、卷积网络、形状重建损失、法向重建以及基于逆向渲染的训练方法。

**📊 数据集**

使用CLO Virtual Fashion的薄壳服装3D数据集以及公开的SMPL/SMPL-X等薄壳网格数据。

**📈 对比分析**

与MeshGAN、3D‑R2N2等传统方法进行定量对比，指标包括Chamfer距离、F‑score和视觉感知质量；实验显示DiffGI在形状细节重建和编辑精度上均优于对比方法，速度提升约30%。

**⚠️ 局限性**

目前仅适用于单连通薄壳网格，难以处理复杂拓扑或大规模高分辨率网格；需要高质量几何图像作为输入，生成过程仍受限于纹理映射误差。

---

## 165. A Hybrid Sampling-Based Trajectory Planner with Game-Theoretic Guidance for Autonomous Racing

**arXiv ID:** 2607.13354 | [PDF](https://arxiv.org/pdf/2607.13354v1)

**作者:** Alexander Langmann `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种混合规划框架，将α‑potential游戏理论与采样式轨迹规划结合，实现了在线生成交互参考路径（IRP）以支持阻挡等战略行为。

**💡 创新点**

创新点在于：离线学习α‑potential潜力函数，在线用梯度上升快速优化交互参数；生成IRP并将其作为动态成本偏置，克服了传统游戏求解器在实时环境中的计算瓶颈。

**🔧 技术方法**

使用了采样式局部规划（Frenet 参考、四次/五次多项式）、α‑potential游戏理论、神经网络价值函数与潜力函数学习、梯度上升优化、以及将IRP融入成本函数的策略化成本。

**📊 数据集**

数据集为在Yas Marina Circuit高保真仿真中生成的2247条三车对战轨迹，训练时对碰撞车辆施加速度与进度惩罚以强化安全性。

**📈 对比分析**

通过定性比较与定量评估（200条随机双对手仿真），与纯反应式采样规划对比：碰撞数从14降至6，平均位置变化从0.54提升至0.61；最佳策略权重为10⁴。

**⚠️ 局限性**

局限性包括需手动调节策略权重；过大或过小均会导致性能退化；IRP是几何近似，无法完全捕捉高速动力学；结果仅在仿真环境验证，尚未在真实车辆上测试。

---

## 166. HybridQC: Hardware-Grounded Simulation of Tightly Integrated Hybrid Quantum-Classical Systems

**arXiv ID:** 2607.13352 | [PDF](https://arxiv.org/pdf/2607.13352v1)

**作者:** Panayiotis Christou `[一作]` (Fordham University), Ying Mao `[通讯]` (Fordham University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 HybridQC，一个硬件基准的离散事件模拟器，用于评估紧耦合量子-经典计算单元（HCU）的拓扑结构、调度策略与可扩展性。

**💡 创新点**

创新点在于①将控制器、互连等经典资源视为一等资源；②将量子工作流拆解为多阶段 DAG；③将物理 QPU 占用与云端队列时间分离；④利用实测数据对 D‑Wave 和 IBM 后端进行服务模型校准；⑤让拓扑与调度策略共同可配置，支持系统层面预测。

**🔧 技术方法**

采用离散事件模拟技术，Python 前端负责工作负载、后端配置与实验管理，Rust 后端实现高性能事件循环；使用 D‑Wave Advantage 1/2、IBM Kingston、Marrakesh、Fez 的测量数据构建 QA/DQC 访问时间与 quantum‑second 的服务模型；配合多阶段工作流 DAG 进行资源分配与调度。

**📊 数据集**

使用 D‑Wave 的 SampleSets 与 IBM Qiskit 生成的电路，以及针对 QA/DQC 任务自行生成的 QUBO 与量子电路实例；所有数据来源于公开的硬件测量与自定义工作负载族（如 bin‑packing、VQE、混合 QA‑DQC 组合）。

**📈 对比分析**

通过对比不同拓扑扩展（QPU、控制器、链接）、调度策略（FIFO、HOBA、GMP、ABO 等）以及多轴工作负载放大，验证模型误差为 3.9%–8.0%（D‑Wave）与 5.3%–19%（IBM）。实验表明 10× 资源平衡扩展仅提升 2.2–3.4× makespan，调度策略可将混合工作负载 makespan 降低至 1.8×，而单纯扩大 QPU 容量的收益有限。

**⚠️ 局限性**

局限性：仅模拟系统层面，未涵盖量子态、编译器、误差校正等细节；模型依赖当前后端测量，未来硬件或队列策略变更需重新校准；对极端规模（大深度/量子位）预测仅为压力测试，未保证绝对精度。

---

## 167. Safe Overtaking for Autonomous Racing Using Hierarchical Optimization and Learning-Based Control

**arXiv ID:** 2607.13348 | [PDF](https://arxiv.org/pdf/2607.13348v1)

**作者:** Hassan Jardali `[一作]` (Indiana University), Lantao Liu `[通讯]` (Indiana University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个分层框架，用混合整数二次规划(MIQP)决定超车侧向轨迹，再用非线性MPC+离散时间控制障碍函数(CBF)跟踪，同时用强化学习在线调节CBF衰减参数，以平衡安全与竞争性能。

**💡 创新点**

创新点在于：①将超车决策与轨迹跟踪分层，显著降低了非凸性导致的保守性；②利用RL动态调节CBF衰减参数，实现在不同赛道与对手密度下的自适应安全边界；③通过软约束和MIQP的组合，提升了实时可行性和鲁棒性。

**🔧 技术方法**

使用的技术包括：混合整数二次规划(MIQP)、Frenet坐标系下的非线性MPC、离散时间控制障碍函数(CBF)、基于Proximal Policy Optimization的多目标RL（ASTCH/PASTA）以及软约束松弛。

**📊 数据集**

数据集主要来自四条真实赛道（Laguna Seca、Monza、Kentucky Speedway、Indianapolis Motor Speedway）下的仿真环境，训练与评估使用多车道碰撞预测模型，硬件验证使用F1TENTH平台。

**📈 对比分析**

与多种固定衰减参数（γ ∈ {0.05,0.15,0.30,0.50,0.70,0.95})以及仅使用MPC/CBF的基线进行对比，结果表明自适应衰减策略在四条赛道上总体成功率最高（67%），并在未见过的赛道上保持竞争力；MIQP层的去除导致成功率下降超过30%。

**⚠️ 局限性**

局限性包括：①安全保证依赖软约束的无滑动性，密集交互时可能短暂失效；②对手模型采用常速假设，缺乏对高速赛道的充分适应；③分层拆分并非全局最优，仍受基准参数敏感；④实验主要在仿真与小型平台，尚未在全尺寸赛车上验证。

---

## 168. Discrete Diffusion Models: A Unified Framework from Tokenization to Generation

**arXiv ID:** 2607.13431 | [PDF](https://arxiv.org/pdf/2607.13431v1)

**作者:** Ye Yuan `[一作]` (McGill University), Xue Liu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并提出统一的离散扩散模型概念框架，强调词元化（tokenization）是设计的核心轴向，系统整理了现有的离散扩散实现、训练目标、推理算法、扩展性、评估协议，并给出跨领域实例与实践指南。

**💡 创新点**

创新点在于：①将词元化视为离散扩散的根本设计因素，揭示其对腐蚀方式、去噪难度、可控性与计算成本的决定性影响；②构建统一四组成分（腐蚀算子、去噪参数化、训练目标、采样器）框架，展示不同论文的设计如何在此框架内映射；③归纳跨域（文本/代码、量化多模态、生物分子、图结构、规划/代理）共同的设计模式与挑战；④系统总结扩散的尺度化、系统优化与评估标准，提出可操作的开放问题。

**🔧 技术方法**

技术包括：离散马尔可夫链/连续时间马尔可夫链腐蚀设计、吸收态掩码、均匀/结构化替换、score/ratio 参数化、ELBO 与重加权交叉熵训练、混合腐蚀、动态步长/采样器、条件与引导、系统加速与并行化。

**📊 数据集**

该工作为综述性质，并未在单一数据集上进行实验；但引用并归纳了多领域应用（自然语言文本/代码、图像/音频/视频的量化码本、蛋白质/基因序列、分子图、晶体材料）所使用的数据集与评测指标。

**📈 对比分析**

由于是系统性综述，本文不提供具体实验对比；但通过对比文献中不同方法在同一任务（如文本生成、图像/音频生成、分子设计等）下的性能指标（FID、BLEU、分子有效率等）进行归纳，并指出统一框架下的最佳实践与常见瓶颈。

**⚠️ 局限性**

局限性：①为综述，缺乏统一的实证验证与统一实验基准；②对某些领域（如量化多模态、分子图）的实验结果高度依赖已有论文，难以系统对比；③仍未给出可直接量化的“词元化质量评估”指标，后续需实验验证所提出的诊断方法；④在规模化与系统优化方面仍有技术细节待深入研究。

---

## 169. DREA: Decoupled Reasoning and Exploration Agents for Repository-Level Vulnerability Detection

**arXiv ID:** 2607.13439 | [PDF](https://arxiv.org/pdf/2607.13439v1)

**作者:** Mingyang Sun `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Guozhu Meng `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DREA（Decoupled Reasoning and Exploration Agents），一个假设驱动的框架，用于进行仓库级别的漏洞检测。

**💡 创新点**

DREA通过将推理与探索解耦，利用两个协作的代理（规划代理和探索代理）来改进漏洞检测，特别是在跨函数和文件的漏洞检测中。

**🔧 技术方法**

使用了一个强大的LLM作为规划代理和一个轻量级模型作为探索代理，进行目标导向的上下文获取。

**📊 数据集**

构建了RepoPairBench，一个包含100个经过验证的Python漏洞修复对的基准数据集，来源于真实项目的CVE记录。

**📈 对比分析**

与功能仅基线相比，DREA在三个LLM上将配对正确率从19-26%提高到30-42%，同时将93%以上的令牌消耗转移到本地的探索代理，减少了16-48倍的估计API费用。

**⚠️ 局限性**

限制在于基准数据集的规模仅为100对，可能无法捕捉到长尾漏洞模式，且分析基于小子组，限制了对特定CWE结论的信心。

---

## 170. Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization

**arXiv ID:** 2607.13432 | [PDF](https://arxiv.org/pdf/2607.13432v1)

**作者:** Jiaxuan Cheng `[一作]` `[通讯]` (Massachusetts Institute of Technology), Jiaxuan Cheng (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实验验证了一个基于信息论的可塑性度量——局部冗余（local redundancy），并展示其可通过在合成记忆化数据上计算梯度范数得到下界。

**💡 创新点**

创新点在于：①将通用压缩理论中的冗余概念迁移到神经网络可塑性；②定义局部模型族并证明其与信息半径等价；③提供梯度范数下界，既可计算又能预测后续任务性能；④在连续学习与迁移学习中展示优于现有度量。

**🔧 技术方法**

技术手段包括：通用压缩理论（Shtarkov 和冗余理论）、信息半径与通道容量、梯度范数与记忆化收益、以及基于随机标签的合成数据生成。

**📊 数据集**

实验数据集：持续学习使用 Continual ImageNet（≈3000个二分类任务）；迁移学习使用 ETT 电力变压器温度时序数据；合成记忆化数据为随机形状图像和随机高斯回归目标。

**📈 对比分析**

与重量范数、距离初始化、休眠神经元比例、训练梯度范数等现有指标比较，局部冗余在连续学习中对未来任务准确度的残差相关最高，在预训练检查点选择时可选出比验证损失更能提升迁移性能的模型。

**⚠️ 局限性**

局限性在于：①仅评估一次梯度步长的局部可塑性，未捕捉长时间训练的全局可塑性；②梯度范数下界未与真实冗余精确对齐，可能在不同模型/检查点间误差不同；③需要合成记忆化数据，生成质量会影响下界；④实验规模受限于几百万参数的模型，尚未验证在大规模模型上的可扩展性。

---

## 171. Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment

**arXiv ID:** 2607.13429 | [PDF](https://arxiv.org/pdf/2607.13429v1)

**作者:** Dwip Dalal `[一作]` (University of Illinois), Unnat Jain `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在机器人演示上对预训练的视觉语言模型（VLM）进行微调，提出 Vision‑Language Anchoring 与 Language‑Action Alignment 两个新损失，防止预训练表征被遗忘并纠正语言与动作的错位；

**💡 创新点**

创新点是：①用冻结的 VLM 进行层级表征蒸馏（Anchoring）保持语义与空间推理；②将连续动作转化为离散运动方向标签，使语言头与动作头在同一观测上共同监督（Alignment），显著提升对 OOD 变换与长周期任务的泛化；

**🔧 技术方法**

技术包括：预训练 VLM（Prismatic‑Qwen2.5）、LoRA 微调、行为克隆（BC）回归/流匹配动作头、层级 MSE 蒸馏损失、六类运动方向分类交叉熵、对齐损失；

**📊 数据集**

数据集有：LIBERO、LIBERO‑PRO、LIBERO‑Plus、CALVIN 以及真实 xArm7 机器人演示（含多姿态、遮挡、语义扰动等）；

**📈 对比分析**

与多种基线（OpenVLA‑OFT、MolmoAct、ChatVLA、UniVLA、OpenHelix 等）比较；在 LIBERO‑PRO 位置交换、对象交换等 OOD 测试中从 22.6% 提升至 71.9%；在 LIBERO‑Plus 视觉扰动中从 54.0% 提升至 90.3%；在 CALVIN ABC→D 长期任务中从 66.5% 提升至 77.9%；在真实 xArm7 上成功率从 28% 提升至 54%（VLA‑Adapter）和从 36% 提升至 60%（StarVLA），表现优异；

**⚠️ 局限性**

局限性包括：仍需要大量演示数据；对极端高噪声或复杂动态场景的鲁棒性待验证；对语言标签离散化的设计可能限制更细粒度动作表达；实验集中在抓取、放置等单一任务类型，跨任务泛化未全面评估。

---

## 172. Can We Steer the Black-Box? Towards Controllability-Centric Evaluation of Recommender Systems with Collaborative Agents

**arXiv ID:** 2607.13418 | [PDF](https://arxiv.org/pdf/2607.13418v1)

**作者:** Jiwen Zhou `[一作]` (Chinese Academy of Sciences), Songlin Hu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CtrlBench-Rec，一个演化式多智能体框架，用来系统评估黑盒推荐系统的可控性，涵盖目标内容发现、兴趣画像塑造与流行度偏差缓解三大任务；

**💡 创新点**

①将可控性作为核心评价维度并正式化为三个渐进任务；②利用演化与协作融合将海量智能体精炼为少数高效超探针；③构建闭环动态交互与行为对齐机制，显著提升探索效率与成本效益；

**🔧 技术方法**

演化算法、多智能体协作、聚类与融合、LLM决策引擎、用户画像专家、行为对齐协议；

**📊 数据集**

MovieLens-1M（电影推荐）与Amazon Toys & Games（电商）公开数据集；

**📈 对比分析**

与基线（随机大规模、随机小规模、KMeans+挑选）和多种黑盒模型（NARM、SASRec、TwHIN‑BERT、BGE、Qwen3.5‑4B）对比，CtrlBench-Rec在目标覆盖率、探索效率、经济成本和交互持续性上均优于基线，并能在不同模型间保持一致性；然而在极度稀疏或中等流行度目标上仍受限，无法突破流行度偏差壁垒；

**⚠️ 局限性**

仅在两类数据集与有限黑盒模型上验证，预算固定且未实现自适应分配；框架虽能测量可控性但不提供系统改进方案，仍需进一步拓展至更大规模、不同领域和新型推荐架构，并探索可解释性与伦理风险管理。

---

## 173. Is the Statistical Advantage Worth the Cost? An Empirical Comparison of KANs and MLPs for Structured Data Classification

**arXiv ID:** 2607.13413 | [PDF](https://arxiv.org/pdf/2607.13413v1)

**作者:** Matthew Steven P. Toledo `[一作]` (University of the Philippines Los Baños), Reginald Neil C. Recario `[通讯]` (University of the Philippines Los Baños)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在十二个公开的结构化表格分类数据集上，对 Kolmogorov‑Arnold 网络（KAN）和多层感知器（MLP）的标准配置进行基准对比。

**💡 创新点**

创新点在于统一实验条件下对两种网络进行无调优比较，并用配对 t 检验和 Wilcoxon 检验量化其统计显著性与效应量，揭示 KAN 在二分类和多分类任务中表现更佳但计算成本更高。

**🔧 技术方法**

使用 KAN 的可学习 B‑spline 映射、MLP 的 ReLU 激活，Adam 优化器，BCE/交叉熵/序数交叉熵等损失函数，评估指标为准确率和 F1 分数。

**📊 数据集**

数据集包括：二分类（Employability、AIDS Clinical Trials、Mushrooms）、多分类（Dropout、Yeast、Statlog）、多标签（Emotions、Birds、Enron）、序数（Balance Scale、Car Evaluation、Abalone）。

**📈 对比分析**

比较方法为五次独立训练的平均准确率与 F1 分数，配合配对 t 检验或 Wilcoxon 检验评估统计显著性；结果显示 KAN 在 9/12 数据集的准确率和 10/12 的 F1 上优于 MLP，整体效应量为中等。

**⚠️ 局限性**

限制在于未对两种网络进行超参数调优，KAN 在多标签任务训练轮数受限，且仅评估表格数据，未验证对文本、图像、时序等其他模态的适用性。

---

## 174. From Interpretation to Compilation: Compilation-Based Execution of Semantic Operators [Vision]

**arXiv ID:** 2607.13407 | [PDF](https://arxiv.org/pdf/2607.13407v1)

**作者:** Wenkai Dong `[一作]` (University of Hawaii at Manoa), Yifan Wang `[通讯]` (University of Hawaii at Manoa)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将自然语言语义操作的执行方式从运行时解释转为编译后本地执行，以减少对 LLM 的调用。

**💡 创新点**

核心创新是把 LLM 视为语义编译器，在编译阶段一次性生成确定性代码，再在查询执行期间本地运行；从而显著降低延迟、成本并提升可扩展性。

**🔧 技术方法**

采用 LLM（如 GPT-4）进行程序合成，生成 Python 函数；并将生成的代码嵌入到现有语义操作系统（Palimpzest）中实现编译执行。

**📊 数据集**

主要使用公开电影数据集（Movie dataset）和 SemBench benchmark 进行实验；对语义过滤、映射和连接三类操作进行评估。

**📈 对比分析**

与传统逐行调用 LLM 的解释执行（baseline）以及 Palimpzest 原生实现进行对比；在单一操作层面实现了 133‑345 倍的 LLM 调用节省，成本下降 16‑32 倍，执行时间提升 10‑15 倍；在多操作 SemBench 查询中，准确率与 baseline 接近甚至更优，LLM 调用次数和成本大幅降低。

**⚠️ 局限性**

局限性包括：对开放式、知识依赖强的映射操作精度下降；编译阶段可能生成冗长代码导致编译耗时；无法保证所有语义操作可编译；需要进一步研究编译可行性评估、混合执行策略和模型漂移后的重编译问题。

---

## 175. Min-Max Regret Task Allocation and Planning of Heterogeneous Multi-Robot System in Partially Known Environments

**arXiv ID:** 2607.13403 | [PDF](https://arxiv.org/pdf/2607.13403v1)

**作者:** Xinkai Liang `[一作]`, Hao Fang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在部分已知环境下大规模异构多机器人系统的任务分配与规划，提出基于最小‑最大后悔的 E‑PDT 框架；

**💡 创新点**

创新点在于引入 Region‑Binding Atomic Proposition 绑定资源不确定性，并结合 regret‑based Branch‑and‑Bound 剪枝实现近线性可扩展性；

**🔧 技术方法**

使用了 scLTL 任务语义、确定有限自动机（DFA）、最小‑最大后悔优化、Extended Planning Decision Tree、启发式 regret 估计与 BnB；

**📊 数据集**

实验采用随机生成的部分已知环境与 60×60 连续工作空间，机器人数量可达 9000，亦在 Wheeltec R550 机器人平台上进行物理实验；

**📈 对比分析**

与 MILP 基线对比，E‑PDT 计算时间呈近线性增长，显著快于 MILP；在任务完成成本上，尤其在资源存在概率较高时，E‑PDT 超越 MILP，表现更优；

**⚠️ 局限性**

局限在于假设环境拓扑已知且资源仅在预定可能区域出现，未处理完全未知或动态变化的环境，也未考虑外部干扰。

---

## 176. The Café in Amsterdam: When the Incumbent Becomes the Oracle

**arXiv ID:** 2607.13393 | [PDF](https://arxiv.org/pdf/2607.13393v1)

**作者:** Augusto Camargo `[一作]` `[通讯]` (University of São Paulo), Augusto Camargo (University of São Paulo)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了用需求独立的接受准则来评估计算方案的框架，讨论了计算重构的自由度与现有实现的关系。

**💡 创新点**

核心创新是引入“基准捕捉”概念，将问题的需求与实现的接受测试分离，提供了T_D与T_0的形式化视角，指出显式接受准则可扩展可接受的重构空间。

**🔧 技术方法**

主要技术是理论分析和形式化定义，包括对需求集D、实现集P、接受测试T的数学描述，示例包括Dijkstra算法的演化和语音前端的矩阵乘法重构。

**📊 数据集**

未使用具体数据集，主要以理论示例（路由问题、语音前端）说明，引用已有工作中的性能测量。

**📈 对比分析**

比较方法：通过示例说明在重构后性能提升，例如矩阵乘法版本比传统FFT‑Mel前端速度提升1.64×–3.29×，能耗降低3.03×；但整体未做系统实验。

**⚠️ 局限性**

局限性：论文仅为理论性研究，缺乏系统实验验证；“购买验证器”的成本和收益尚未量化；对不同领域的适用性需要进一步探讨。

---

## 177. FM$^2$: Unified Federated Foundation Models for Heterogeneous Multimodal Medical Imaging

**arXiv ID:** 2607.13386 | [PDF](https://arxiv.org/pdf/2607.13386v1)

**作者:** Shengchao Chen `[一作]` (Shenzhen University), Ting Shu `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FM^2，一个联合训练多模态医学影像的联邦基础模型框架，解决影像模态异质性（overlapped 与 non-overlapped）问题。

**💡 创新点**

创新点包括：双重 Mixture‑of‑Experts（Class‑wise 与 Domain‑wise）架构；针对模态漂移的 Heterogeneous Modality Alignment (HMA) 正则化，具有 (1/√T) 的收敛率和泛化保证；利用 GPT‑4o 生成的诊断字幕作为跨模态共享语义桥梁的 Caption‑Enhanced Learning；以及对多任务（分类、Caption 监督、视觉问答）的统一支持。

**🔧 技术方法**

采用联邦学习、Mixture‑of‑Experts、HMA 正则化、对比学习（InfoNCE）与 GPT‑4o 生成字幕、CLIP/ViT 视觉编码器、PubMedBERT 文本编码器、LoRA 微调等技术。

**📊 数据集**

使用 MIMH benchmark（由 5 个 MedMNIST 数据集：RetinaMNIST、BloodMNIST、TissueMNIST、PathMNIST、DermaMNIST）在四种配置下进行实验；另外用 PneumoniaMNIST、OCTMNIST、OrganAMNIST 评估跨模态泛化；并在 SLAKE、VQA‑RAD、VQA‑Med 三组 VQA 数据集上验证多任务能力。

**📈 对比分析**

与 FedAvg、FedProx、PerFedAvg、FedBN、FedProto、FedRep 等基线在分类、Caption‑Enhanced Learning 与 VQA 上对比，FM^2 在绝大多数设置下实现 10–30% 的准确率提升，收敛速度更快，且在非重叠模态下仍能实现显著性能与强泛化。

**⚠️ 局限性**

局限性包括：通信开销相对传统 FedAvg 较高；依赖 GPT‑4o 生成字幕，可能受 LLM 质量与可用性限制；未在极大规模客户端或更复杂模态（如 3D CT、MR 图）上进行验证；对隐私的保障需在字幕共享与模型更新间进一步细化。

---

## 178. Detector Confidence Signals Presence Rather Than Occlusion in Cluttered Manipulation

**arXiv ID:** 2607.13361 | [PDF](https://arxiv.org/pdf/2607.13361v1)

**作者:** Yuanzhi He `[一作]` `[通讯]` (Cardiff University), Yuanzhi He (Cardiff University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并验证了一种基于几何分割的“几何oracle”审计方法，评估开放词汇检测器的置信度是否真实反映目标可见性，并发现置信度与实际可见性解耦，导致在遮挡场景下的误判和误用；

**💡 创新点**

创新点在于：①引入无检测器依赖的几何oracle审计框架；②系统性验证多种检测器（Grounding DINO、OWLv2、SAM3）在三维遮挡下置信度失效的普遍性；③揭示置信度误用会严重低估主动感知价值并误导门控决策；

**🔧 技术方法**

技术手段包括：仿真渲染（LIBERO、ManiSkill）、几何分割生成目标可见像素计数、开放词汇检测器置信度计算、基于IoU的目标定位评估、主动视角搜索与基于置信度/可见度的策略比较、ROC曲线分析、Bootstrap置信区间；

**📊 数据集**

使用的数据集与环境包括：LIBERO仿真场景（定向遮挡和环形遮挡）、ManiSkill 3（YCB物体随机混乱）、真实视频DAVIS‑2017的主目标遮挡序列，此外还对三种检测器在九类目标上进行了实验；

**📈 对比分析**

比较方法：将检测器的置信度与几何oracle的可见像素量进行相关性分析；将主动视角策略与固定视角的可见率差异（oracle收益）对比；通过ROC AUC评估置信度与遮挡检测的门控能力；实验结果表明：置信度与可见性相关系数接近0甚至负相关，置信度门控几乎不动作，主动视角的oracle收益可达88点，但置信度基准仅提升3-8点；

**⚠️ 局限性**

局限性：①几何oracle仅在仿真中完美可得，真实环境中需依赖近似分割或运动捕捉；②研究聚焦于单目标与同类遮挡，未深入多目标或异类遮挡场景；③仅评估了三种检测器，其他模型可能表现不同；④未提供完整的端到端抓取成功率，仅展示定位误差导致的抓取失败。

---

## 179. Learning Latency-Aware Orchestration for Multi-Agent Systems

**arXiv ID:** 2607.13359 | [PDF](https://arxiv.org/pdf/2607.13359v1)

**作者:** Xi Shi `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向延迟的多智能体系统框架LAMaS，能够在保持任务准确率的前提下显著降低端到端推理延迟。

**💡 创新点**

创新点包括：① 用拉格朗日约束和关键路径加权的信用分配在训练阶段实现延迟最小化；② 通过轻量级运行时控制器在推理过程中动态裁剪冗余操作；③ 两者组合在多任务设置下仍保持可迁移性。

**🔧 技术方法**

核心技术包括：拉格朗日约束优化、关键路径加权信用分配、基于策略梯度的多智能体图搜索、以及基于MLP的在线延迟预测控制器。

**📊 数据集**

使用了四个公开基准：GSM8K、HumanEval、MATH、MMLU-Pro。

**📈 对比分析**

与生成式、链式思考、以及三种学习型MAS基线（GDesigner、AgentDropout、MaAS）对比，LAMaS在所有四个基准上都实现了约55%–75% 的延迟下降，同时准确率与基线相当或更优，且成本保持低或略有提升。

**⚠️ 局限性**

局限性包括：① 依赖于预先定义的操作器库，未探索更大规模的动态操作集合；② 在极大规模或分布式部署场景下的可扩展性仍待验证；③ 对于非常长或高度并行的任务，关键路径估计误差可能影响控制效果。

---

## 180. Data-Efficient Adaptation of LLMs via Attention Head Reweighting

**arXiv ID:** 2607.13425 | [PDF](https://arxiv.org/pdf/2607.13425v1)

**作者:** Tuomas Oikarinen `[一作]` (University of California San Diego), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Attention Head Reweighting（AHR），仅学习每个注意力头一个标量，以极少参数适配LLM到少样本文本分类任务。

**💡 创新点**

通过只调整注意力头的贡献，利用头的功能专门化，参数量比LoRA低200-1000倍，从而在少样本环境下显著提升性能。

**🔧 技术方法**

在Transformer内部对W_O做可学习标量乘子，并结合参数高效微调（PEFT）、in‑context finetuning以及正则化（L1/L2）技术实现。

**📊 数据集**

使用六个文本分类数据集：SST2、AG‑News、Emotion、Web Phishing、Toxigen 与 Jailbreak，覆盖一般与安全相关任务。

**📈 对比分析**

与LoRA、AdaLoRA、IA3等基线比较，AHR在少样本（≤30）下平均提升2‑4%绝对精度，安全任务提升6‑7%，同时训练参数仅占模型总参数的0.0001%。

**⚠️ 局限性**

局限性：主要适用于简单输出的分类任务，对大样本或需要细粒度控制的任务效果不足；当样本量≥300时，其他更强大的PEFT方法可能表现更好。

---

## 181. Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions

**arXiv ID:** 2607.13423 | [PDF](https://arxiv.org/pdf/2607.13423v1)

**作者:** Wisdom Dogah `[一作]` `[通讯]` (University of Mines and Technology), Wisdom Dogah (University of Mines and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了在软标签（人类注释分布）下，使用硬标签进行温度标定的校准性能，并量化了由此产生的校准误差；

**💡 创新点**

创新点在于正式提出“软标签校准差距”，并系统验证该差距在不同模型规模、视觉与语言领域的普遍性及其随模型规模的增长趋势；

**🔧 技术方法**

技术手段包括温度标定、热力回归（multiclass isotonic regression）以及Brier Score、ECE等评价指标；

**📊 数据集**

使用公开软标签数据集CIFAR‑10H（视觉）和ChaosNLI（语言）进行实验；

**📈 对比分析**

对比硬标签标定与软标签标定的Brier Score差距，结果显示所有9种配置均存在正向差距；差距随视觉模型规模和语言模型规模单调增大，并且语言领域的差距约为视觉的30倍；同样的结论也在热力回归下得到验证；

**⚠️ 局限性**

局限性包括模型规模范围有限（未覆盖极大模型）、语言模型仅在10,000样本一轮微调、ChaosNLI‑M实验因近似随机准确率而不确定、以及数据集仅覆盖英语，缺乏跨语言与文化的通用性验证。

---

## 182. OrDA: Orthogonal Disentanglement of Access Habits Framework for Homepage Marketing Block Recommendations

**arXiv ID:** 2607.13420 | [PDF](https://arxiv.org/pdf/2607.13420v1)

**作者:** Lingxiao Zhang `[一作]` (Ant Group), Tao Xu `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对首页营销块中因访问习惯导致的伪正样本问题，提出了Orthogonal Disentanglement of Access Habits（OrDA）框架；

**💡 创新点**

创新点在于将访问习惯与兴趣分离为两条正交的潜在空间，并通过双塔结构、门控分配层以及正交正则化实现解耦；在推理时使用do‑calculus干预消除习惯效应；

**🔧 技术方法**

采用多任务深度神经网络（双塔MLP+MaskNet），门控分配层（GAL）、正交余弦正则化、因果推断（do‑calculus）、AUC与GAUC评估指标；

**📊 数据集**

使用芝麻首页营销块15天大规模数据集：30.2M用户、108.9M训练样本、7.9M验证样本；

**📈 对比分析**

与基线、ESMM、USD、Multi‑IPW、PAL等SOTA去偏方法对比，OrDA在GAUC_all 0.6412、GAUC_cold 0.6416、GAUC_active 0.6304上均居首；线上A/B测试提升CTR 5.64%；

**⚠️ 局限性**

局限在于正交约束可能导致信息蒸发，对多样化场景的泛化能力未充分验证，且假设访问习惯仅由用户特征决定，实际中可能还受内容或位置等因素影响。

---

## 183. EXPLORE: Exploration with Guided Search for Analog Topology Generation using Language Models

**arXiv ID:** 2607.13416 | [PDF](https://arxiv.org/pdf/2607.13416v1)

**作者:** Guanglei Zhou `[一作]` (Duke University), Xin Zhang `[通讯]` (IBM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了EXPLORE框架，将预训练的Transformer与模拟器驱动的蒙特卡洛树搜索相结合，实现测试时搜索以自动生成满足用户规格的模拟电路拓扑。

**💡 创新点**

创新点在于：①使用p‑filtering自动提交高置信度的结构化token，显著节省模拟预算；②将语言模型的先验概率融入PUCT选取节点，提升搜索质量；③构建了覆盖6–10组件的大规模电路数据集，并将EXPLORE本身用作高质量数据收集器。

**🔧 技术方法**

采用了Flan‑T5基础模型进行Transformer序列生成、top‑k采样与p‑filtering、蒙特卡洛树搜索（PUCT）以及NGSPICE模拟器进行性能评估。

**📊 数据集**

使用公开的3–5组件电路数据集与自建的350k 6组件电路数据集，并在此基础上扩展到7–10组件的验证集。

**📈 对比分析**

与Greedy、Beam Search、Sampling+Filtering、MCTS‑Base四种基线对比，EXPLORE在6组件、tolerance 0.01下成功率从33%提升至65%，MSE降低20×；在7–10组件时仍保持相对优势，证明测试时搜索更易扩展。

**⚠️ 局限性**

局限性包括：对极大组件数（>10）时成功率仍偏低；p‑filtering阈值需手动调参；依赖昂贵的模拟器预算；实验仅针对电压转换器，未验证对其他模拟电路类型的泛化。

---

## 184. Weight Feedback Computes the Jacobian Transpose Locally in Modern Deep Networks

**arXiv ID:** 2607.13380 | [PDF](https://arxiv.org/pdf/2607.13380v1)

**作者:** Junlong Shen `[一作]` (University of Alberta), Xingyu Li `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文证明在冻结归一化统计的条件下，权重反馈可通过本地可得的激活导数、对称前向权重和归一化增益三项直接计算Jacobian转置，从而实现无 autograd 的误差传递，进而提出了 WF-Act-PC 算法。

**💡 创新点**

创新点在于首次发现并利用 Act(Norm(L(x))) 层的 Jacobians 可分解为可局部可获得的三项因子，消除了传统预测编码中唯一的非局部运算，并在深度卷积网络上实现了精确的误差传递。

**🔧 技术方法**

技术手段包括预测编码框架、权重对称假设、软谱范数裁剪、MaxPool 的近似上采样、RMSProp 内循环和顶层到底层的 Gauss–Seidel 迭代。

**📊 数据集**

实验使用了 CIFAR‑10、CIFAR‑100 与 Tiny‑ImageNet 三个标准图像分类数据集，并采用与 PCX 基准相同的数据增强策略。

**📈 对比分析**

通过与经典 PC 方法（iPC、DPC‑CN、ePC）以及使用同一增强策略的 BP（BP‑CE、BP‑SE）进行对比，WF‑Act‑PC 在所有网络架构上都达到了或超过了深度 BP 的准确率，并且表现出最佳的深度可扩展性。

**⚠️ 局限性**

局限性包括：仍需假设权重对称和软谱范数裁剪（两者非完全本地化），冻结归一化假设以及 MaxPool 的近似上采样会引入少量误差，此外内循环迭代导致的计算开销相对较大。

---

## 185. Fair on the Surface: Transaction-Ordering Bias and MEV in Mysticeti DAG-based BFT Protocol

**arXiv ID:** 2607.13378 | [PDF](https://arxiv.org/pdf/2607.13378v1)

**作者:** Iliya Mirzaei `[一作]` (Stony Brook University), Mohammad Javad Amiri `[通讯]` (Stony Brook University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估并量化了 Mysticeti DAG‑based BFT 共识协议在 Sui 主网中存在的交易排序偏差（低索引头部优势、gas‑price 重新排序的稳定性漏洞以及可合法执行的“静默”放大攻击），并提出了基于随机 tiebreak 的改进方案。

**💡 创新点**

首次系统地揭示了 DAG‑based BFT 中线性化 tiebreak 规则导致的隐性 MEV 机会，并提供了最小改动即可消除低索引头部优势和 tie‑loophole 的解决方案。

**🔧 技术方法**

结合协议分析、实验模拟（13/25 节点实验）、攻击指标 Attack Success Rate (ASR)、以及对 Sui 源码的直接修改与性能评估。

**📊 数据集**

采用合成的、无状态、固定大小（32 字节）事务批量作为工作负载，保持每个验证者接收事务均衡，模拟主网流量分布。

**📈 对比分析**

通过同轮 ASR 与全对 ASR 进行公平度评估；改进后同轮 ASR 从约 89% 降至 45%，基本恢复到 50% 公平线；全对 ASR 仍受“静默”攻击影响，改进对其影响有限；对整体系统性能影响极小，未破坏安全性与延迟。

**⚠️ 局限性**

仅解决了 tiebreak 产生的头部优势和 tie‑loophole，未消除“静默”放大攻击的根源；对大规模动态网络、不同事务费用分布等场景的评估有限；改动仍需社区验证与部署。

---

## 186. A POS Tier Is the Key to Automated Annotation for Low-Resource Language Documentation: Neural Interlinear Glossing for Irabu, a Southern Ryukyuan Language

**arXiv ID:** 2607.13372 | [PDF](https://arxiv.org/pdf/2607.13372v1)

**作者:** Michinori Shimoji `[一作]` `[通讯]` (Kyushu University), Michinori Shimoji (Kyushu University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了完整的神经化简线注释流水线（形态素分割 → POS标注 → 词义注释），并在仅一小时监督资源的约束下评估其在琉球伊尔阿布语上的效果。

**💡 创新点**

首次量化了在低资源场景下增设 POS 层（四线注释）的收益，并证明 POS 信息可以显著提升词义注释准确率；同时提出两阶段注释策略：先小量高质量注释后自动草拟再人工校正。

**🔧 技术方法**

采用小型、可解释的 BiLSTM‑CRF 模型完成分割、POS、词义预测；利用规则推断边界类型；对 POS 噪声进行注入实验评估切入点；使用 5 随机种子和 McNemar 检验保证结果稳健。

**📊 数据集**

使用伊尔阿布琉球语语料库：774 句子、6,412 形态素、约 47 分钟完整注释（分为 620/77/77 句子训练/验证/测试），并实验不同的注释预算（6–47 分钟）。

**📈 对比分析**

在不使用 POS 的基线下，词义准确率为 0.893；加入 gold POS 可提升至 0.937（+4.4 绩点）；使用预测 POS 仅提升 0.897（+0.4 绩点）。分割精度 span‑F1 0.907，POS 精度 0.881；完全自动流水线在 47 分钟训练后达到 0.93 的词义准确率。POS 层在低数据（≈12 分钟）时可将所需注释量削减约一半；通过噪声注入实验发现需 88% 以上的 POS 准确率才能显现全自动收益。

**⚠️ 局限性**

局限性包括：仅针对单一语言和单一语料库；未与预训练 Transformer 或 LLM 进行对比；POS 益处仅在 gold 情况下得到充分体现；完全自动流水线仍未能显著超越基线；对开放词义的处理仅靠词典复制，缺乏深度学习方法；结果的泛化性尚未在其他语言上验证。

---

## 187. RoughNet: Mapping Arctic Sea Ice Roughness Using Diffusion-Based Super-Resolution of Satellite Imagery

**arXiv ID:** 2607.13371 | [PDF](https://arxiv.org/pdf/2607.13371v1)

**作者:** Tessa Cannon `[一作]` (University College London), Randall Scharien `[通讯]` (University of Victoria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于条件扩散模型，利用Sentinel‑2多光谱影像生成1米分辨率的海冰地表高度残差，用于粗糙度估计。

**💡 创新点**

首次将扩散式生成模型直接从10米光学影像生成米级海冰粗糙度场，实现厘米级绝对误差。

**🔧 技术方法**

采用U‑Net结构的条件扩散网络（Cosine‑PLMS采样），并结合多时相Sentinel‑2图像与光照元数据。

**📊 数据集**

使用三块加拿大北极地区的冰鸟飞行平台LiDAR高程数据与对应的Sentinel‑2 Level‑2A图像。

**📈 对比分析**

与多种采样策略（DDPM/DDIM/PLMS）和噪声计划对比，Cosine‑PLMS在验证集RMSE 9 cm、nRMSEσ≈1、JSD≈0.07，测试集保持相似精度。

**⚠️ 局限性**

局限在于仅覆盖三地区、对云/阴影敏感、对可变光照/季节性变化的泛化有限，且推理速度约30 s/256×256块。

---

## 188. Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System

**arXiv ID:** 2607.13370 | [PDF](https://arxiv.org/pdf/2607.13370v1)

**作者:** Teri Rumble `[一作]` (Abertay University), Ruth Falconer `[通讯]` (Abertay University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对LEA（Learning Engagement Assistant）进行真实课堂部署（CMP511）并跨学科、跨层次三门课程（CMP511、CMP202、PSY555）进行可扩展性评估，结合模拟与真实学生的对比研究，探讨系统在不同课程环境下的表现与局限。

**💡 创新点**

首次将具备Agentic AI、RAG+KC模型的三模态辅导系统落地真实课堂，并系统评估其跨课程可扩展性；揭示系统在不同学科与学术层级下，答案相关性与上下文精度保持稳定但可信度随域距离下降的现象。

**🔧 技术方法**

使用检索增强生成（RAG）与向量检索（ChromaDB）构建课程知识库；基于知识组件（KC）模型实现课程结构化；Agent Orchestrator 估计认知负荷、近端发展区与动机状态，并根据这些信号动态调整 Socratic 质询、难度与反馈；利用 GPT‑4 生成教学内容；采用 RAGAS 指标评估检索/生成质量；模拟学习者模型用于前期验证。

**📊 数据集**

课程素材（讲义、幻灯片、代码、作业）在 CMP511、CMP202、PSY555 中转化为 RAG 向量库；通过教师提供的学习目标/细粒度目标生成 KC 结构；生成 660 条题目进行 RAGAS 评估；模拟学习者配置 14,143 条交互数据；真实课堂收集 8 名 CMP511 学生的调查反馈。

**📈 对比分析**

将模拟阶段的检索成功率、答案质量、适应性反馈等指标与课堂调查结果进行对比；使用 RAGAS 在三门课程上评估四项指标，发现答案相关性≈0.93、上下文精度≈0.90保持稳定，可信度随课程距离降低（0.69→0.50），提示系统在新域下仍能保持检索质量，但生成准确性受限；总体显示架构可迁移，但需针对特定内容做细调。

**⚠️ 局限性**

局限性包括：仅 8 名学生、20% 参与率、使用者会话有限；RAGAS 的 Context Recall 采用自身生成答案作为参考，缺乏人工真值；未覆盖广泛学科与更高学术层级；代码块分块策略未针对不同语言优化；部分系统组件（如代码题生成）携带单域假设；缺乏长周期、对比组实验验证学习成效。

---

## 189. xChk: Bring Your Own Identity -- Heterogeneous Assurance with Verifier-Determined Sufficiency

**arXiv ID:** 2607.13369 | [PDF](https://arxiv.org/pdf/2607.13369v1)

**作者:** Sean MacGuire `[一作]` `[通讯]` (Independent Researcher), Sean MacGuire (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了 xChk，一个支持“自带身份”（BYOI）的 OAuth/OIDC 身份提供者，允许用户通过多模态证明（政府 KYC、企业 SSO、WebAuthn/FIDO2、专业网络、现场验证等）完成身份注册，并将这些证明以可组合的“投資组合声明”嵌入标准 OIDC 令牌；IdP 只负责传输和可选的策略评估，最终访问权限判断留给各个依赖方；同时提供人机交互的 attestation（可链式哈希、可选 Ravencoin 区块链锚定）以及代理人治理与级联吊销的统一机制。

**💡 创新点**

创新点主要体现在：
1) BYOI 多模态身份登记与可组合投資组合声明，令牌中携带多维度身份保证；
2) 验证方确定“足够性”，IdP 不做访问决策，实现三方职责分离；
3) 在同一验证图上统一管理人类与非人类实体的授权链、级联吊销与会话标识；
4) 人机协作的 attestation 系统采用哈希链+HMAC 签名，并可选地在 Ravencoin 区块链上锚定，提供不可篡改的时间戳；
5) 对接现有的 OAuth 2.0 / OIDC 生态，兼容标准库，且通过双向策略评估保护用户隐私与 RP 机构安全。

**🔧 技术方法**

技术实现使用了：
- OAuth 2.0 / OpenID Connect（授权码 + PKCE，RS256 签名）
- WebAuthn/FIDO2、SSO 集成、Verifiable Credentials（仅作为外部证明源）
- HMAC-SHA256 用于投資组合包与 attestation 记录的完整性校验
- 哈希链与作用域递减规则实现代理人链的级联吊销
- 可选 Ravencoin 区块链 OP_RETURN 锚定
- Hermes 运行时安全守护、Agent API Keys、authorize/attest 接口
- Node.js/Express、MongoDB、Nginx TLS、PM2 等后端堆栈
- 微基准测试（公共端点延时、HMAC 本地验证）

**📊 数据集**

没有使用公开的固定数据集；系统在生产环境中收集并存储来自真实用户的身份证明（如 Didit KYC、Entra ID、Google Workspace、LinkedIn、GitHub 等），并在需要时通过外部服务获取对应的验证结果。研究阶段使用的是 xChk 公开 API 进行端到端演示。

**📈 对比分析**

对比方法：采用微基准测量公开端点的延时，并对投資组合 HMAC 验证做本地性能评测。
- 公共发现（/.well-known/openid-configuration）平均 159 ms；
- JWKS 公开检索 161 ms；
- 投資组合 HMAC 验证 0.014 ms。
由于系统设计为高并发 Web 服务，整体延时在 200 ms 以内；对比其他主流 IdP 未给出正式基准，但通过公开接口可直接与常规 OIDC 兼容。

**⚠️ 局限性**

局限性：
1) 仅有一个正式生产依赖方（crabbyed.com），其他第三方尚未广泛接入；
2) 需要 RP 解析自定义声明并实现策略评估，标准 OIDC 库不支持自动处理；
3) 高级功能（非人类 OIDC 实体、区块链锚定）仍处于实现/预发布阶段；
4) 代理人链与级联吊销依赖后端数据库的完整性保护，需自行部署或使用托管服务；
5) 人机交互的 attestation 延时受审批者响应时间影响，无法满足实时性要求；
6) 信任分数仅为建议性指标，未进行正式用户研究；
7) 需要 Ravencoin 节点和钱包才能启用链上锚定；
8) 目前不支持多语言或多国法域下的细粒度合规声明。

---

## 190. CODA: How to Mitigate ColumnDisturb for (Almost) Free?

**arXiv ID:** 2607.13505 | [PDF](https://arxiv.org/pdf/2607.13505v1)

**作者:** Moinuddin Qureshi `[一作]` `[通讯]` (Georgia Tech), Moinuddin Qureshi (Georgia Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CODA（ColumnDisturb Mitigation with Reduced-ACI）框架，通过减少邻近子阵列的计数器递增（ACI）来高效抵御ColumnDisturb攻击，保持低性能和能耗；

**💡 创新点**

核心创新在于设计三种变体（CODA-E、CODA-F、CODA-G）及其组合（CODA-EF、CODA-EFG），能够把ACI率从200%降低到仅0.1%~1.4%，并在不增加额外存储的情况下实现近乎零的性能开销；

**🔧 技术方法**

采用子阵列级别的激活计数器（ACTR）、待处理激活计数器（PAC）与分数递增、以及在多子阵列团（gang）中跳过无效ACI的策略，整合Refresh Coordination与REGA等已有Rowhammer防御技术；

**📊 数据集**

在公开的SPEC‑2017、GAP、STREAM等内存密集型基准上进行评估，并使用MemSim仿真平台；

**📈 对比分析**

与传统SAL​T、CDP以及REGA‑CDP等方案对比，CODA在TRHD=500时将慢速下降到5.9%（CDP为17.8%），在TRHD=4K时降至0.3%（CDP为16.8%），能耗提升不到1%，显示出显著的性能优势；

**⚠️ 局限性**

主要局限包括：对极端攻击模式（集中激活单一子阵列）效果有限；依赖于开放式bitline架构与子阵列可追踪性；对TRHD较低的场景需要更细粒度PAC；仍存在极小的阈值提升（≤1.5%）。

---

## 191. DP-BOA: Dirichlet-Process Birth-or-Assign for On-the-Fly Category Discovery

**arXiv ID:** 2607.13504 | [PDF](https://arxiv.org/pdf/2607.13504v1)

**作者:** Peiyan Gu `[一作]` (ShanghaiTech University), Xuming He `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DP-BOA框架，利用贝叶斯证据比较在流式场景下决定样本是归入已有类别还是产生新类别，并在线更新类别统计。

**💡 创新点**

创新点在于：①将类别出生作为明确的贝叶斯备选方案；②采用Dirichlet过程先验与NIW后验预测，捕获类别的各向异性几何；③实现在线证据自适应更新；④提供低秩近似版本DP-BOA-L以降低计算与内存开销。

**🔧 技术方法**

使用技术包括：Dirichlet-Process高斯混合模型、Normal–Inverse–Wishart先验、Student-t预测分布、在线后验更新、低秩协方差近似，以及基于ViT的特征编码器。

**📊 数据集**

在十个标准OCD基准上进行实验，数据集涵盖CIFAR-100、ImageNet-100、CUB、Stanford Cars、Herbarium19、Oxford-IIIT Pet以及iNaturalist的Fungi、Arachnida、Animalia、Mollusca四个子类。

**📈 对比分析**

与SMILE、PHE、DiffGRE、Sync等SOTA OCD方法对比，DP-BOA在8/10个数据集上获得最优All分数，特别是在新类别上显著提升（如CUB新类精度提升至51.6%——比Sync高10.7个百分点），同时保持竞争力的已知类准确率。

**⚠️ 局限性**

局限性包括：①仅采用单一椭圆形分布建模，难以捕捉多模态或复杂的类别结构；②对流式顺序有一定依赖，虽然鲁棒性已验证；③使用固定超参数，缺乏对不同流或领域的自适应调整；④完整协方差实现对高维特征存在较大计算与内存负担，需要低秩近似或更高效的实现。

---

## 192. LAPO: Leave-One-Turn Attribution for Self-Generated Process Rewards in Multi-Turn Search Reasoning

**arXiv ID:** 2607.13501 | [PDF](https://arxiv.org/pdf/2607.13501v1)

**作者:** Qiang Zhu `[一作]` (Zhejiang University), Jiajun Wu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LAPO方法，通过后向留一回合归因为多轮检索推理过程提供自生成的过程监督；

**💡 创新点**

创新点在于利用回溯留一回合归因与符号一致性门控，避免外部评估器的依赖并实现更细粒度的信用分配；

**🔧 技术方法**

使用基于现有策略的金标答案似然增益、tanh映射、分组归一化和门控，结合GRPO强化学习框架；

**📊 数据集**

在七个知识密集型问答数据集（NQ、TriviaQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、PopQA）上进行训练与评测；

**📈 对比分析**

与提示式、基于终点奖励和步骤奖励的方法对比，LAPO在所有七个数据集上均取得最高EM平均0.326，较最强基线提升约19.4%；

**⚠️ 局限性**

局限性包括归因仅为基于上下文与策略的对比估计，未建模多回合交互；依赖金标答案；目前仅适用于短回合本地检索场景。

---

## 193. Layered Risk Mapping for Autonomous Patient Transport in Expeditionary Medical Facilities

**arXiv ID:** 2607.13497 | [PDF](https://arxiv.org/pdf/2607.13497v1)

**作者:** Lorena Maria Genua `[一作]` (Northeastern University), Taşkın Padır `[通讯]` (Northeastern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在前线医疗设施中，设计并验证了一套多层风险映射框架，将斜坡、静态/动态障碍和语义可通行度四个异构风险层融合为统一的概率成本表面，驱动MPPI局部规划实现自主病人运输；

**💡 创新点**

创新点包括：①使用Noisy‑OR概率模型对四个风险层进行统一融合，取代传统最大或加权叠加；②首次在前线医疗场景下将自主轮椅用于无护送的患者转运；③结合斜坡、语义分割与RTAB‑Map实现对动态障碍的实时评估；

**🔧 技术方法**

关键技术包括：DEM斜坡提取（FABDEM/COPERNICUS GLO‑30）、语义分割（SegFormer‑B2 fine‑tuned on ADE20K）、视觉‑IMU SLAM（RTAB‑Map）产生静态/动态障碍层、Noisy‑OR概率融合、MPPI信息理论MPC局部规划；

**📊 数据集**

使用的数据集/资源有：Copernicus GLO‑30 DEM（斜坡层）、ADE20K（语义分割），以及仿真环境中生成的随机地形与人群数据；

**📈 对比分析**

通过与无风险、最大值混合、Log‑odds三种融合方法的配对蒙特卡洛实验（三密度级别），Noisy‑OR在碰撞率、障碍清除距离、峰值风险等指标上均优于其他方法，碰撞率从>73%降至<32%，危险区进入率从>44%降至<13%，且保持了较小的路径长度损失；

**⚠️ 局限性**

主要限制包括：1）全部算法在笔记本上运行，资源竞争导致计算延迟；2）视觉SLAM在特征稀疏或光照骤变环境中漂移，影响地图精度；3）未对风险层进行概率校准，未来需更精准的校准与嵌入式硬件部署。

---

## 194. Deformable State Estimation for Autonomous Surgical Tissue Retraction Under Partial Observability

**arXiv ID:** 2607.13475 | [PDF](https://arxiv.org/pdf/2607.13475v1)

**作者:** Everest Yang `[一作]` (Brown University), George D. Konidaris `[通讯]` (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一个基于PCA低维表示和多层感知机的学习式状态估计器，用少量噪声表面观测恢复完整变形网格，并用于多步外科组织牵拉规划。

**💡 创新点**

创新点是将PCA低维编码与几何正则化（平滑性与拉伸惩罚）相结合，使得在稀疏噪声观测下的规划性能几乎达到全状态oracle水平。

**🔧 技术方法**

使用了PCA、三层多层感知机、几何正则化损失（平滑性与拉伸惩罚）、拉普拉斯松弛仿真模型以及采样式单步/多步规划。

**📊 数据集**

使用了5,000个随机生成的网格状态用于PCA训练，200个仿真episode用于评估；所有数据均为2D可变形片的仿真生成。

**📈 对比分析**

通过与oracle（全状态）和naive基线在单步和多步情形下的曝光提升和成功率进行比较。多步时学习估计器达到98.1%的oracle性能，曝光提升0.0941，成功率45.0%；单步时略优于oracle。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，缺乏真实组织数据；对观测密度和噪声水平的鲁棒性未充分评估；仅针对2D片模型，扩展到3D复杂组织仍需进一步研究。

---

## 195. MyAG: A Graph-Based Framework for Designing and Analyzing Composable LLM Agent Systems

**arXiv ID:** 2607.13474 | [PDF](https://arxiv.org/pdf/2607.13474v1)

**作者:** Zhisong Zhang `[一作]` `[通讯]` (City University of Hong Kong), Zhisong Zhang (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MyAG，一个轻量级的基于图的框架，用于构建、可组合、层次化以及分析 LLM 代理系统。

**💡 创新点**

核心创新在于将代理系统拆分为组件图、工作流图和搜索图三种图抽象，并通过系统节点实现层次化复用；同时提供基于执行记录的 LLM 与环境成本分析工具。

**🔧 技术方法**

技术包括：图结构建模、函数调用接口、LLM-as-a-judge 评估器、支持多种搜索策略（贪婪、最佳优先、回滚、Best‑of‑N）以及可视化监控。

**📊 数据集**

使用 GAIA 文本子集（复杂问答）和 Mind2Web‑Live（网页导航）两个基准数据集进行实验。

**📈 对比分析**

对比四种工作流/搜索策略，评估任务成功率、总时延、LLM token 费用和环境交互时延。实验显示贪婪策略最省资源，回滚策略在预算充足时表现最好，最佳优先和 Best‑of‑N 在更高计算预算下提升性能，但成本更高。

**⚠️ 局限性**

局限性包括：缺乏生产级特性（分布式部署、访问控制、容错）、成本度量需要用户自行配置、实验范围仅限于两类任务，未覆盖更广泛的环境与应用。

---

## 196. EgoHTR: Egocentric 4D Demonstrations of Human Terrain Traversal

**arXiv ID:** 2607.13472 | [PDF](https://arxiv.org/pdf/2607.13472v1)

**作者:** Alex Brandes `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开 EgoHTR 数据集，提供 4D 人体-地形交互序列，包含 55 条多模态、150k 帧记录，支持粗糙地形下的机器人学习与评估。

**💡 创新点**

创新点在于：① 结合可携带的 Aria 眼镜、Rokoko MoCap 套装和 Leica 3D 扫描器，实现在无人造场景中的厘米级人机重建；② 发布可扩展的捕捉与重建流水线；③ 通过高精度参考运动提升感知式步态策略的训练效果。

**🔧 技术方法**

采用 SMPL-X 体形参数化、三阶段时空对齐（基于 Aria SLAM、ICP、视觉几何变换）、IMU 逆运动学优化、强化学习（PPO）训练人机共舞策略，及多模态评估指标。

**📊 数据集**

核心数据集为 EgoHTR 本身；与 3DPW、EMDB、PROX、RICH、SLOPER4D 等公开数据集进行对比评估。

**📈 对比分析**

在局部和全局 HPS 评估中，MPJPE 73.2 mm、PA-MPJPE 54.3 mm，优于 SLOPER4D 等；在机器人感知式步态学习中，加入足部接触奖励显著提升成功率；在网格恢复基准中，EgoHTR 在不同模态下取得最高成功率、最低误差。

**⚠️ 局限性**

局限性包括：① 数据规模与多样性有限，未覆盖多人人体与运动物体；② 仅针对静态环境，未做后置人机场景优化；③ 对高加速或纹理缺失环境的传感器鲁棒性有限。

---

## 197. Bring Music The Horizon: Music-Driven 360$^\circ$ Video Generation

**arXiv ID:** 2607.13471 | [PDF](https://arxiv.org/pdf/2607.13471v1)

**作者:** Kai Hsu Tsai `[一作]` (National Yang Ming Chiao Tung University), Yu-Chih Chen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套从歌曲到全景360°音乐可视化视频的完整流水线，包含音乐信息检索、情绪预测、情绪引导的全景关键帧生成以及图像转视频合成。

**💡 创新点**

1) 将音乐情绪的时间演化（valence-arousal轨迹）与全景视觉内容直接关联；2) 通过EmotionCrafter与SEGA进行情绪残差和语义控制，保持用户指定概念同时跟随情绪；3) 使用专门的360° LoRA和Wan-based I2V模型，实现真正的沉浸式VR视频。

**🔧 技术方法**

音乐信息检索（All‑In‑One、Dynamic V‑A回归器）、EmotionCrafter、SEGA、SDXL+360° LoRA、Wan‑I2V/Wan‑flf2v等扩散式图像生成与视频合成技术。

**📊 数据集**

对EmotionCrafter使用自定义数据集进行再训练（去除人类动作噪声），并在多种流派的歌曲上测试，具体数据集未公开列出。

**📈 对比分析**

与代表性音频‑视觉基线 From‑Sound‑To‑Sight 进行定性对比，展示在VR头显上可直接观看，视频的时间结构与情绪流动与输入歌曲高度一致，未给出定量指标。

**⚠️ 局限性**

1) 相似的V‑A值导致关键帧重复；2) 关键帧拼接时出现边缘缝隙；3) 输出分辨率有限；4) 前进运动效果尚未完全可控；5) 过长的提示词会导致生成平面而非全景。

---

## 198. PQFA: Parallel Quantum Feature Augmentation of Fused Representations for Multimodal Classification

**arXiv ID:** 2607.13466 | [PDF](https://arxiv.org/pdf/2607.13466v1)

**作者:** Mingzhu Wang `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Yun Shang `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种并行量子特征增强（PQFA）框架，将浅层变分量子电路应用于融合后的多模态表示，作为后融合的轻量级增强模块。

**💡 创新点**

创新点在于：① 在已有强大经典融合后引入量子测量读取，提供非线性特征变换；② 通过严格对比无量子分支与宽度匹配的MLP分支，证明量子增强的独特性；③ 仅需约2.2K参数即可显著提升性能，且对缺失模态更鲁棒。

**🔧 技术方法**

使用技术包括：冻结预训练RoBERTa与ViT编码器 → 双向交叉注意力 → 关注池化 → 自适应门控融合 → 振幅编码至7量子比特 → 多个并行brickwall参数化量子电路 → Pauli‑Z期望值读取 → 与经典特征拼接 → MLP分类头；整体训练采用混合梯度下降与Adam。

**📊 数据集**

实验数据集：MM‑IMDb（多标签电影类型）与N24News（单标签新闻主题）。

**📈 对比分析**

比较方法：在同一编码器、融合骨干、数据拆分和特征维度下，对比无量子（NoQ）、宽度匹配MLP增强（MLP‑Aug）以及其他基线；结果显示PQFA在MM‑IMDb上微F1从67.62提升至68.28，宏F1从61.22提升至61.85；在N24News上准确率从84.23%提升至85.35%；在缺失模态实验中仍保持领先。

**⚠️ 局限性**

局限性：实验仅在模拟器上验证，未评估实际量子硬件噪声与有限测量采样对性能的影响；量子电路深度保持极浅，可能不足以捕获更复杂的特征映射；缺乏对更大规模数据集或其他模态组合的广泛验证。

---

## 199. DevicesWorld: Benchmarking Cross-Device Agents in Heterogeneous Environments

**arXiv ID:** 2607.13465 | [PDF](https://arxiv.org/pdf/2607.13465v1)

**作者:** Huatao Li `[一作]` (Shanghai Jiao Tong University), Chen Qian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 DevicesWorld，一个面向跨设备协同操作的大规模可执行基准，包含 6,140 个任务，涵盖 Android、Linux 与智能家居三类设备。

**💡 创新点**

创新点在于：① 将多类设备统一到同一交互与评估框架；② 通过多阶段任务构造与质量控制管线保证任务可执行性和判定一致性；③ 提供可自动化验证与清理的完整工作流，方便诊断与研究。

**🔧 技术方法**

使用技术包括：大语言模型驱动的代理（GPT‑5.5、Qwen3.7‑Plus、Gemini‑3.1‑Pro‑Preview、Claude Opus 4.8 及一个分层跨设备代理）、基于规则的状态/文件检查、统一的跨设备环境控制器以及自动化任务构造脚本。

**📊 数据集**

使用的数据集是自构建的 6,140 个可执行任务，任务覆盖三类设备的多样交互（手机应用、桌面软件、IoT 控制），并通过多阶段验证保证可执行性。

**📈 对比分析**

实验通过固定评估集对 5 个基线模型进行对比，最高任务成功率仅 12.5%，平均得分 0.262，平均步骤 22，平均耗时 7.9 分钟，预算耗尽率 22.0%。表明现有模型在跨设备协同上仍远低于理想水平。

**⚠️ 局限性**

局限性包括：① 对跨设备信息获取与目标设备切换的准确性不足，导致信息丢失或写错位置；② 缺乏全局任务状态维护与动态重规划，易陷入局部循环或误判完成；③ 评估判定过于侧重最终状态而忽视执行过程中的错误恢复与多设备一致性检查。

---

## 200. Energy Minimization Oriented Resource Allocation for Integrated Sensing and Communication in Marine IoT Networks

**arXiv ID:** 2607.13462 | [PDF](https://arxiv.org/pdf/2607.13462v1)

**作者:** Qianru Wang `[一作]`, Yuan Wu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于无人机（UAV）与无人水面车辆（USV）的非正交多址（NOMA）集成感知与通信（ISAC）网络，并针对海洋物联网（MIoT）场景，联合优化UAV的发射波束、专用感知信号、USV的发射功率、UAV的计算功率以及感知与通信两阶段的时隙分配，以最小化系统总能耗，同时满足时延和感知精度约束。

**💡 创新点**

创新点包括：①在非凸能耗优化问题上构建分层拆解框架，①1）将USV功率解析为闭式；②2）利用多项式逼近与单调性设计UAV计算功率优化；③3）对UAV发射波束采用SDR与单次迭代近似；④4）对感知信号采用SCA；②将三大子问题通过BCD迭代求解；③通过数值实验验证该方案相较于传统OFDMA、遗传算法以及LINGO获得显著能耗提升。

**🔧 技术方法**

采用的关键技术包括：非正交多址（NOMA）与信息叠加解码、集成感知与通信（ISAC）、单调优化与多项式逼近、半正定松弛（SDR）、连续凸近似（SCA）、块坐标下降（BCD）以及CVX求解器。

**📊 数据集**

实验数据来自文献[35]的典型参数设置（如UAV 10发射天线、15接收天线、10 MHz带宽、1 W功率上限等），并在二维平面上随机布置USV（半径200 m）与UAV、基站，使用Rayleigh衰落与路径损耗模型进行仿真，并无真实海洋传感数据集。

**📈 对比分析**

通过与OFDMA、遗传算法（GA）以及优化工具LINGO进行对比。结果表明：相较于OFDMA，能耗降低19.71%；相较于GA，能耗误差不超过8%；与LINGO相比，误差仅为8.72%。此外，算法的运行时间比GA快7倍、比LINGO快11倍。性能随USV数量、感知时隙长度、感知速率以及USV–UAV距离变化而有可预期的趋势。

**⚠️ 局限性**

局限性：①仅考虑单目标感知与静态信道模型；②未对UAV轨迹规划或多目标感知做联合优化；③设计中假设完美的SIC和无残余干扰，虽在实验中给出了鲁棒性评估；④缺乏真实海洋实验验证，模型参数主要来自文献。

---

## 201. DreamSat-Pose: Spacecraft Pose Estimation from Single-View 3D Reconstructions and Learned 2D-3D Feature Matching

**arXiv ID:** 2607.13449 | [PDF](https://arxiv.org/pdf/2607.13449v1)

**作者:** Josiane Uwumukiza `[一作]` (Wellesley College), Richard Linares `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于单张RGB图像的无人目标太空器6自由度姿态估计框架 DreamSat-Pose

**💡 创新点**

利用单视角3D重建作为几何先验，将姿态估计转化为2D–3D特征对应问题，并通过双流Transformer实现软对应

**🔧 技术方法**

使用冻结的DINOv3视觉Transformer提取2D特征、DGCNN图卷积网络提取3D特征、Transformer自/跨注意力+Sinkhorn归一化得到对应矩阵，最后通过EPnP+RANSAC求姿态

**📊 数据集**

在SPE3R数据集上进行训练与评估，包含64个航天器模型、1000张合成图像、对应的掩码和GT姿态；同时使用FoundationPose作为基准

**📈 对比分析**

与基准FoundationPose在GT模型与重建模型两种输入下比较；DreamSat-Pose在重建模型上平均指向误差0.157°、姿态误差≈58.7°，比仅使用单张图像的Baseline高出约10-15°，并能在未知目标上实现相对准确的指向估计

**⚠️ 局限性**

主要限制：姿态误差受重建误差、空间器对称性及主体框架歧义影响；单图像分辨率低导致特征稀疏；实时性受限（约0.034s推理+35-40s重建）；未验证真实传感器噪声和运动模糊条件

---

## 202. Stress-Sharing: A Bio-Inspired Approach to Decentralized Fault Repair in Modular Spacecraft

**arXiv ID:** 2607.13444 | [PDF](https://arxiv.org/pdf/2607.13444v1)

**作者:** Sidhdharth D. Sikka `[一作]` (Purdue University), Shaoshuai Mou `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于生物学应激共享（stress‑sharing）原理的分布式、去中心化模块化航天器故障自修复策略；

**💡 创新点**

创新点在于将局部应力信号传播与连接安全的转动操作相结合，实现只用局部信息和物理可行的 lattice‑约束动作完成故障后连通性恢复与部分形状重建；

**🔧 技术方法**

技术包括：图论模型、局部移动性（criticality）测试、张量化的损伤令牌传播、随机探索与抗振动记忆、以及两阶段（凝聚与重构）自适应转动；

**📊 数据集**

使用自生成的树状和全连接（FC）模块网络，模块数量从10到160，损伤密度分别为10%、20%和30%，并考虑随机与局部（集群）损伤两种分布；

**📈 对比分析**

与基线（随机转动）对比时，使用重连率、恢复率、形状差异以及移动次数作为指标。实验结果显示：在大多数情形下，恢复率>80%，随规模增大而提升；重连率随损伤密度与规模下降，但局部损伤比随机损伤更易恢复；形状差异在重构阶段平均下降约0.8个百分点；随机探索显著提升 FC 结构的重连率；

**⚠️ 局限性**

局限性在于：仅对球形模块有效，非球形或多体结构需更复杂的清晰度检查；方法严格局部，无法处理极大规模碎片化导致的长距离断裂；重构阶段仅能部分恢复形状，无法完全重现原始布局；需要更高层次的全局协作或临时断连策略以实现完整重连。

---

## 203. The Environmental Cost of Digital Sovereignty: Water, Energy, and Emissions Impacts of Sovereign AI Infrastructure in the Global South

**arXiv ID:** 2607.13443 | [PDF](https://arxiv.org/pdf/2607.13443v1)

**作者:** Muntaser Syed `[一作]` (Florida Institute of Technology), Amal El Ahmad `[通讯]` (UAE University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对全球南方四国（UAE、孟加拉、印度、肯尼亚）主权AI部署的水资源、能源需求与碳排放进行定量比较，并基于结果提出环境友好设计原则

**💡 创新点**

首次将主权AI与环境可持续性与资源竞争三重目标框架化为“主权‑可持续性三元悖论”，并用案例分析与定量模型量化主权AI的环境成本

**🔧 技术方法**

使用GPU功耗模型、PUE/WUE 计算、碳强度因子与热力学分析等技术手段，对不同规模（256/1024 GPU）与冷却方案（蒸发、混合、浸没）进行模拟

**📊 数据集**

公开可获取的水资源压力指数（WRI Aqueduct）、电网碳强度指数（Ember）、GPU功耗数据（NVIDIA DGX H100）以及国家/区域能源与水消耗统计

**📈 对比分析**

通过多案例对比，展示不同规模和冷却技术下水耗从数千万升到数十万升、排放从数千吨到数百吨的差异，证明在低碳、低水压力地区部署主权AI更具可持续性

**⚠️ 局限性**

局限于使用国家/区域平均值而非机房级测量，未考虑具体地点的地理、运营差异，结果为粗略量级估算，缺乏现场实测数据

---

## 204. TreeSRNF: Square-Root Normal Fields for Generative Modelling of the Geometric and Structural Variability in Tree-like 3D Objects

**arXiv ID:** 2607.13456 | [PDF](https://arxiv.org/pdf/2607.13456v1)

**作者:** Tahmina Khanam `[一作]` (Murdoch University), Anuj Srivastava `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种新的TreeSRNF框架，用于对树形3D物体的几何和结构变化进行统一建模与分析。

**💡 创新点**

创新点在于将Square‑Root Normal Field（SRNF）扩展到树形对象，既捕捉枝条表面几何，又考虑分支拓扑变化，构建了新的树形Riemannian空间和弹性度量。

**🔧 技术方法**

采用的技术包括SRNF与切线表示、弹性度量、Hungarian算法求解分支对应、数值逆SRNF、PCA以及高斯分布采样等。

**📊 数据集**

实验使用了多份合成与真实植物数据集，包括Tomato植物、其他植物模型及公开的树形3D数据。

**📈 对比分析**

与现有方法（如Wang等的骨架+厚度表示）比较，TreeSRNF在几何误差、闭环一致性、描述长度和计算时间等指标上均优于对手，尤其在闭环一致性和描述长度上显著提升。

**⚠️ 局限性**

局限性包括：逆SRNF求解仍较耗时，对大规模复杂树形（数千枝）计算仍需改进；目前主要验证于植物，未系统评估神经/血管等其他树形结构。

---

## 205. To Play or Not to Play: Insights and Lessons Learned from 20 Years of CTFs with ENOFLAG

**arXiv ID:** 2607.13480 | [PDF](https://arxiv.org/pdf/2607.13480v1)

**作者:** Jörg Schneider `[一作]` (Technische Universität Berlin), Sebastian Koch `[通讯]` (Technische Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

回顾并总结了团队在20年里组织和参与CTF竞赛的经验与见解。

**💡 创新点**

将AI的代理能力与CTF竞赛结合，讨论了AI对竞赛影响及未来可能的玩法与奖励模式。

**🔧 技术方法**

使用了云计算、CTFd、Arkime、OpenAttackDefenseTools等工具，以及利用AI代理进行挑战。

**📊 数据集**

未使用公开数据集，主要基于团队自身的CTF参与记录与网络流量日志。

**📈 对比分析**

未进行系统化性能对比，主要通过参赛人数、参与次数、工具改进等经验描述。

**⚠️ 局限性**

缺乏量化评估与实验数据，AI普及导致学习效果下降，资源有限的学术团队面临竞争不平衡的问题。

---

## 206. Elton: Urn Resources for Reasoning about Adversarial Probabilistic Programs

**arXiv ID:** 2607.13459 | [PDF](https://arxiv.org/pdf/2607.13459v1)

**作者:** Kwing Hei Li `[一作]` (Aarhus University), Lars Birkedal `[通讯]` (Aarhus University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种用于推理包含未知攻击者代码的高阶概率程序的高阶分离逻辑。

**💡 创新点**

创新点在于引入延迟采样的分布式不变式和新的 urn 资源谓词，扩展了传统分离逻辑。

**🔧 技术方法**

使用 Coq 证明助手与 Iris 分离逻辑框架进行形式化证明，并在 call‑by‑value 语义下证明了可消除性和安全性。

**📊 数据集**

无使用数据集，属于理论方法。

**📈 对比分析**

通过与以往方法对比，证明了在多种安全示例中能够给出误差界限，超越了先前技术的适用范围，但未给出具体性能数值。

**⚠️ 局限性**

局限性包括目前仅适用于 call‑by‑value 语义，需手工制定不变式，尚未对并发或大规模程序进行实证验证。

---

## 207. Attention-Free and Lightweight Token Reduction for Efficient Vision-Language Models

**arXiv ID:** 2607.13500 | [PDF](https://arxiv.org/pdf/2607.13500v1)

**作者:** Xuanyi Hao `[一作]` (Zhejiang University), Shuguo Zhuo `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一个无关注点、轻量化的视觉令牌压缩框架ALTR，在Vision‑Language模型中将视觉令牌从数百压缩至几十，同时保持甚至提升跨模态推理性能。

**💡 创新点**

创新点在于：①使用信息论第二阶Rényi熵对单个令牌的特征分布进行重要性评估，完全不依赖注意力图；②利用视觉编码器最后一层MLP变换前后向量的一致性信号来衡量令牌多样性，通过排序+stride采样实现无对称相似度比较的高效多样性筛选；③一次性完成压缩，兼容FlashAttention等加速框架。

**🔧 技术方法**

主要技术包括：信息论熵度量、第二阶Rényi熵计算、cos相似度排序、stride采样、Transformer + LLM融合、FlashAttention兼容性设计。

**📊 数据集**

实验使用的视觉语言数据集包括LLaVA‑v1.5、LLaVA‑Next、Qwen2.5‑VL、MME、TextVQA、ScienceQA、GQA等。

**📈 对比分析**

通过与SparseVLMs、PyramidDrop、MustDrop（内部-LLM）以及VisionZip、VisPruner（前置）等方法在不同令牌保留率（192/128/64）下对比。ALTR在保持约90‑98%原始性能的同时，比基线平均提升2‑5%（在极端压缩下仍高于其他方法）。

**⚠️ 局限性**

局限性：仍需先完整生成所有视觉令牌；对不同视觉编码器需要手动调节平衡系数λ；未针对视频或时间维度的多模态模型进行评估；在极低令牌预算下性能仍有下降空间。

---

## 208. CASA-SDF: Curriculum-Aware Spatial Adaptation with Curvature-Guided Density for Neural Implicit Surface Reconstruction

**arXiv ID:** 2607.13492 | [PDF](https://arxiv.org/pdf/2607.13492v1)

**作者:** Lei Yang `[一作]` (Nanjing University of Science and Technology), Liang Xiao `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了CASA‑SDF框架，利用课程化的空间自适应策略实现室内几何异质性的神经隐式表面重建

**💡 创新点**

创新点在于将监督生命周期与SDF‑to‑density映射的局部带宽共同适应，分别通过Hybrid‑SAUA融合语义与光度不确定性进行像素级课程调度，以及CALADT通过曲率感知动态调节本地尖锐度

**🔧 技术方法**

采用神经隐式表面（NeuS/VolSDF）基础网络，结合AngMF不确定性估计、SSIM光度不确定性、可微渲染与可学习的SDF‑to‑density映射

**📊 数据集**

在ScanNet与Replica两大室内数据集上进行实验

**📈 对比分析**

与多种基准（COLMAP、NeuS、MonoSDF、DebSDF、ND‑SDF等）比较，CASA‑SDF在完成度与召回率上取得最高值，整体F‑score与精度保持竞争性，显著提升薄结构与过渡区域的重建质量

**⚠️ 局限性**

限制在于对超参数（如学习率、阈值、曲率阈值）敏感，需要额外离线不确定性预计算，且对极其纹理缺失或高噪声场景仍可能出现细节缺失或误匹配

---

## 209. Explainable Artificial Intelligence for Anomaly Detection in Banking Transactions: An Internal Audit Perspective

**arXiv ID:** 2607.13469 | [PDF](https://arxiv.org/pdf/2607.13469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 210. Topology-Agnostic Mesh Reconstruction of Deformable Objects from Sparse Touch

**arXiv ID:** 2607.13479 | [PDF](https://arxiv.org/pdf/2607.13479v1)

**作者:** Everest Yang `[一作]` `[通讯]` (Brown University), Everest Yang (Brown University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种拓扑无关的跨注意力网络，可仅凭稀疏触感重建一维、二维、三维柔性物体的完整网格，并通过深度集成实现不确定性估计与主动触点选择。

**💡 创新点**

创新点在于：①跨注意力解码器对网格查询保持局部性，单一模型即可覆盖1D、2D、3D柔体；②使用深度集成提供可用于主动感知的每点不确定性；③学习的主动触点策略在自遮挡场景中实现更优采样。

**🔧 技术方法**

采用了变形物体网格表示、跨注意力（grid‑query cross‑attention）、多头自注意力编码、MLP解码、深度集成以及基于期望误差减少的学习主动感知策略。

**📊 数据集**

通过MuJoCo Flex仿真生成的三类柔体数据：20×20布料、40点绳索、6×6×6软体（约216点），每类约1000个随机配置，划分训练/测试。

**📈 对比分析**

与非学习的IDW、GPIS及全局池化（DeepSets）基线对比，误差平均降低约2/3；主动感知相较随机触点提升约5‑7%误差减少，最高可达14%（oracle），在高遮挡下提升更显著。

**⚠️ 局限性**

主动触点的增益有限，尤其当有视觉传感器时几乎无效；学习策略与oracle仍存在性能差距，且实验仅在仿真环境中验证，未展示真实机器人部署效果。

---

## 211. VAMP-MR: Vector-Accelerated Motion Planning and Execution for Multi-Robot-Arms

**arXiv ID:** 2607.13478 | [PDF](https://arxiv.org/pdf/2607.13478v1)

**作者:** Philip Huang `[一作]` (Carnegie Mellon University), Jiaoyang Li `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了 VAMP-MR，一种基于 CPU SIMD 的多机器人臂碰撞检测框架，并将其集成到多种运动规划、快捷化和安全执行流程中；

**💡 创新点**

创新点在于将单机器人向量化碰撞检测扩展到多机器人系统，支持自碰撞、机体-环境和机体-机体检查，可批量并行处理多配置，并在不改动规划算法的前提下实现 10–100 倍加速；

**🔧 技术方法**

使用 AVX2 SIMD 指令进行向量化前向运动学与碰撞检测、球形化机器人几何、批量化检查、缓存友好内存布局，以及改进的约束树评估；

**📊 数据集**

在 Panda 系列机器人（Panda Two Rod、Panda Four、Panda Four Bins）以及 APEX-MR 提供的双臂 LEGO 组装任务上进行实验验证；

**📈 对比分析**

与 FCL/Bullet 等传统碰撞检测器对比，在 3 个环境中单配置检查提升 10–28 倍、路径验证提升 65–148 倍；在 RRT‑Connect、CBS‑MP 规划中获得 10–150 倍规划时间加速，TPG 构造时间缩短 7–21 倍，最终使任务完成时间缩短至 3–59 秒，成型质量比基线提升 10% 以上；

**⚠️ 局限性**

局限性包括碰撞检测仍占规划时间 64–90%（需进一步 MAPF 优化）、仅在 CPU 上加速且对极大规模机器人数量的扩展尚未验证、向量化批量终止机制可能在极罕见的碰撞情形下误判，以及未覆盖动态环境或实时在线规划需求。

---

## 212. Auditing Protocol-Level Shortcuts in Large Audio Language Model Judges for Speech Evaluation

**arXiv ID:** 2607.13477 | [PDF](https://arxiv.org/pdf/2607.13477v1)

**作者:** Joonyong Park `[一作]` (University of Tokyo), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过设计对照条件，对三种部署协议（特征蓝图、参考条件、A/B对比）下的六种大型音频语言模型（LALM）评审者进行协议级快捷方式审计；

**💡 创新点**

创新点在于将评审者视为测量协议，系统地识别协议本身触发的快捷路径，并针对每种协议提出匹配的快捷探测方法；

**🔧 技术方法**

采用结构化蓝图提示、参考标签插入以及顺序/格式变换等技术，对模型输出进行分析；

**📊 数据集**

使用四个公开数据集（RAVDESS、FLEURS、BVCC、VoxCeleb1）以及对应的专家分类器（emotion2vec+、whisper-LID、ECAPA‑TDNN）作为专业标签源；

**📈 对比分析**

通过比较模型在正向与对照条件下的准确率、解析率和位置锁定率，发现参考条件会产生位置依赖锚，A/B对比会出现槽位或格式偏倚，而蓝图条件在情感属性上复制错误标签，性能因模型音频推理能力而异；

**⚠️ 局限性**

局限性包括评审者样本有限、仅覆盖四个相对简单属性，缺乏更大规模模型和更具挑战性的评估指标的验证。

---

## 213. 2D Rotary Position Embedding for Scene Text Recognition with Transformers

**arXiv ID:** 2607.13458 | [PDF](https://arxiv.org/pdf/2607.13458v1)

**作者:** Zobeir Raisi `[一作]` `[通讯]` (Chabahar Maritime University), Zobeir Raisi (Chabahar Maritime University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种适用于场景文本识别的二维旋转位置编码（2D‑RoPE），兼容编码器‑解码器架构

**💡 创新点**

创新点在于将轴向2D RoPE按文本宽高比分配维度，并将2D旋转扩展到交叉注意力，完全无额外参数

**🔧 技术方法**

使用Transformer、ResNet‑50特征提取、2D‑RoPE、自动回归解码器等技术

**📊 数据集**

在IIIT5K、SVT、ICDAR2013、ICDAR2015、CUTE80、SVTP六大标准基准上训练和评估

**📈 对比分析**

与现有1D/2D绝对编码、RoPE等方法对比，平均准确率提升至90.4%（高于MATRN 89.5%），在不规则、视角畸变文本上提升最显著

**⚠️ 局限性**

局限在于仅使用轴向旋转，无法充分捕捉斜向或极端弯曲，且对不同宽高比的自适应仍有限

---

## 214. Improving Map Consistency in Graph-Based LiDAR SLAM Through Information-Aware Odometry and Retroactive Loop Closure

**arXiv ID:** 2607.13516 | [PDF](https://arxiv.org/pdf/2607.13516v1)

**作者:** Saurabh Gupta `[一作]`, Cyrill Stachniss `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种结合信息感知ICP、分层循环闭环和几何回溯闭环的3D LiDAR SLAM框架，既提升全局轨迹精度，又显著改善局部地图的一致性。

**💡 创新点**

创新点包括：① 通过线性回归实时估计ICP的几何信息矩阵，实现在位姿图优化中的不确定性加权；② 采用多分辨率子地图进行位置识别与几何验证，分离特征识别与精确配准；③ 在后端优化后利用几何一致性回溯发现前端遗漏的闭环，形成前后端闭环反馈循环。

**🔧 技术方法**

使用的技术有：KISS-ICP定位、信息加权位姿图优化、分层子地图结构、Hausdorff距离关联、平移/旋转分块Hessian估计、线性回归信息矩阵、退化姿态回溯闭环检测。

**📊 数据集**

在Apollo@MulRan、HeLiPR、Newer College、KAIST Riverside、Sejong、H237、MAVE、SB等多传感器、多环境的公开数据集上进行评测。

**📈 对比分析**

通过与CT-ICP、MULLS、PIN-SLAM、KISS-SLAM等基线进行ATE、RPE以及回访点RMS比较，实验显示本方法在大多数序列实现了最优或相近的轨迹误差，并在回访点的RMS误差明显低于基线，表明地图一致性得到显著提升。

**⚠️ 局限性**

局限性包括：在极窄视场或低频扫描环境下回溯闭环效果有限；信息矩阵估计基于局部线性假设，极端姿态扰动可能导致误估；回溯闭环的额外计算开销在实时系统中的影响尚未充分评估。

---

## 215. ExTernD: Expanded-Rank Ternary Decomposition Ternary LLM PTQ with Accuracy Approaching Any Quantization Level

**arXiv ID:** 2607.13511 | [PDF](https://arxiv.org/pdf/2607.13511v1)

**作者:** Chethan Reddy G. P `[一作]` `[通讯]`, Chethan Reddy G. P

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ExTernD，一种后训练的三值因子分解，利用可扩展的内层秩对每个权重矩阵进行逼近。

**💡 创新点**

创新点在于可调内层秩 μ 与连续稀疏阈值 τ 的组合，证明误差随秩递减，可实现任意目标精度，并突破传统固定平面三值量化的精度上限。

**🔧 技术方法**

采用贪婪 ALS 与批量块 ALS 的 GPU 并行实现，配合阈值三值化、重要性加权 ALS 与稀疏存储压缩等技术。

**📊 数据集**

实验数据集包括 Gemma‑4‑E2B、Qwen3.5‑4B、Granite‑4.0‑h‑tiny 以及用于 perplexity 评估的 wikitext‑2。

**📈 对比分析**

与 PT2‑LLM、PTQTP、Q4_K/Q5_K 等固定平面量化方案对比，ExTernD 在相同或更低有效比特/权重下实现 99%+ 能量保留，完整模型在 wikitext‑2 的 perplexity 仅比 bf16 多 3.2%，并且在有效比特 5.2–5.5 处与 Q4_K 接近。

**⚠️ 局限性**

局限性包括仅在单张 AMD MI50 GPU 上评估，未测试 30B+ 规模模型，缺乏与其他三值方案的端到端对比，未验证 QAT 效果，以及稀疏压缩仍未实现硬件加速。

---

## 216. CDS: Counterfactual Directionality Score for Structured Interventions in Spatial Graphs

**arXiv ID:** 2607.13508 | [PDF](https://arxiv.org/pdf/2607.13508v1)

**作者:** Humaira Anzum `[一作]` (University of Houston), Tania Banerjee `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种基于结构化反事实干预的图模型——Counterfactual Directionality Score（CDS），用于量化空间单细胞数据中不同细胞类型之间的方向性影响。

**💡 创新点**

创新点在于：① 通过邻居影响模型（NIM）学习局部邻域对细胞表达的预测；② 设计了保留空间结构的受控干预操作（类型交换与内部类型替换），从而仅衡量细胞类型组成的影响；③ 将CDS解释为有限差分敏感度，并配合核心层抽样重置实现置信区间估计。

**🔧 技术方法**

使用技术包括：距离加权邻域聚合、带残差块与细胞类型门控的多层前馈网络、Huber 损失训练、结构化反事实干预、核心层自助法统计推断。

**📊 数据集**

数据集涵盖：① 合成空间图（含正向、无效、混淆三种情景）用于验证；② 两套真实空间转录组数据（乳腺癌与肺癌组织微阵列），分别包含肿瘤、基质和免疫三大细胞群体。

**📈 对比分析**

与传统注意力权重、特征归因、相关系数、梯度敏感度以及随机扰动等基线方法比较，CDS 在正向情景下达到 AUC 0.97，且在无效或混淆情景下保持接近零，显著优于其他方法；在真实数据中显示出方向性一致性并给出统计显著置信区间。

**⚠️ 局限性**

局限性包括：① 依赖于 NIM 的预测准确性，若模型拟合不足会影响 CDScore；② 仅考察局部邻域，可能忽略长程或全局交互；③ 干预设计需手动设定距离分箱与核心匹配，对不同组织结构的泛化需进一步验证。

---

## 217. A VAE-Driven Multi-Task Satellite-Aided Semantic Communication Framework for 6G-Enabled Connected Autonomous Vehicles

**arXiv ID:** 2607.13494 | [PDF](https://arxiv.org/pdf/2607.13494v1)

**作者:** S. M. Abtahiul Alam `[一作]` (Noakhali Science and Technology University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于变分自编码器（VAE）的多任务语义通信框架，用于卫星辅助的联网自动驾驶车辆，实现交通标志图像的重建与分类。

**💡 创新点**

创新点包括：①采用概率潜在表示并通过自适应KL权重避免后验坍塌；②构建复合感知损失（L1、MSE、SSIM、梯度），提升重建的感知质量；③加入细节修正分支与确定性推理策略，在噪声环境下保持重建稳定；④整体端到端联合优化重建与分类任务。

**🔧 技术方法**

使用技术包括：VAE（编码器、重参数化、KL正则化）、双分支解码器与残差细化网络、两层全连接分类头、复合感知损失函数、功率归一化、Rayleigh衰落+AWGN通道模型、交叉熵+label smoothing分类损失、AdamW优化器。

**📊 数据集**

使用的数据集为：中国交通标志数据集（30类）和德国交通标志识别基准（GTSRB，35类），图像尺寸均为64×64灰度。

**📈 对比分析**

与基准（确定性卷积自编码器和16-QAM+SVM）对比，VAE框架在-10 dB至30 dB的SNR下实现了更高的分类准确率（最高≈96.7%）和更好的SSIM（≈0.69–0.87），并在相同压缩率下压缩率提升至87.23%–98.17%，显示出在低信噪比下显著的鲁棒性和效率优势。

**⚠️ 局限性**

局限性包括：①仅在模拟Rayleigh+AWGN通道下验证，未考虑卫星特有的多径、Doppler和非线性效应；②实验图像仅为64×64灰度，规模有限，缺乏对大尺寸彩色图像的评估；③网络结构复杂，可能对实时低功耗设备有部署挑战；④缺乏真实卫星链路或现场测试，尚未验证在实际环境中的表现。

---

## 218. Personalizing Incremental Video Search with Hybrid Text and ID Embeddings

**arXiv ID:** 2607.13493 | [PDF](https://arxiv.org/pdf/2607.13493v1)

**作者:** Vivek Kanojiya `[一作]` (Apple), Xuetao Yin `[通讯]` (Apple)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

为 Apple TV 的增量视频搜索构建了一套个性化系统，在每个键入字符后即时返回排名结果，使用文本与 ID 双向嵌入结合 XGBoost 对比学习排名模型。

**💡 创新点**

创新点在于：1) 针对 1–3 字符短前缀查询的低词法信息场景，提出混合嵌入（文本语义+ID 交互）作为个性化信号；2) 将两种嵌入直接注入 XGBoost 的 pairwise 排名器，保持低延迟且兼容现有多阶段检索栈；3) 使用 LLM 生成的无曝光偏差评估集验证嵌入质量。

**🔧 技术方法**

技术包括：多语言 Transformer 句子编码器 + MLP 投影的文本嵌入；ID 直接 lookup + MLP 的协同嵌入；对比学习（hinge loss）训练两种嵌入；mean‑pool 用户历史表示；XGBoost pairwise ranking；LLM‑judged 评估集。

**📊 数据集**

数据集：Apple TV 搜索日志与观看日志，近时段的用户搜索交互用于正负样本生成；时间拆分的 hold‑out 评估集；另由大型指令调优 LLM 生成的 统一覆盖的语义相似性标注集。

**📈 对比分析**

比较方法：离线使用 NDCG@10、MRR@10 与平均转化位置评估；在线进行 3 周 A/B 实验，显著提升 1.14% 点击率、1.23% 转化率，短前缀查询 NDCG@10 提升 8.63%。与无个性化基线相比，混合嵌入在 NDCG@10 上提升 2.99%，MRR@10 提升 3.30%。

**⚠️ 局限性**

限制：仅在单一地区上线，需跨地区验证；评估仍受曝光/点击偏差影响；对稀疏历史用户提升有限；mean‑pool 用户表示未捕获时序动态；离线训练与在线实时嵌入更新不同步，可能抑制在线收益。

---

## 219. GPOcc++: Unified Sparse Gaussian Occupancy Prediction with Visual Geometry Priors

**arXiv ID:** 2607.13481 | [PDF](https://arxiv.org/pdf/2607.13481v1)

**作者:** Changqing Zhou `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于视觉几何先验的统一稀疏高斯占据预测框架GPOcc++，能够在室内单帧、室内多帧/多视角以及户外多摄像头的场景中实现高质量、稀疏的三维占据重建。

**💡 创新点**

核心创新点包括①Ray-Conditioned Multi-Image Fusion模块，利用相机射线几何显式建模跨视角/跨时间的特征交互；②Offset-Guided Ray Anchoring技术，在射线采样的基础上为每个高斯原子预测三维偏移，使其更精准地对齐复杂结构；③统一的稀疏高斯表示与增量更新策略，使方法既高效又易于在长序列中扩展。

**🔧 技术方法**

技术实现主要依赖视觉几何先验模型（如DepthAnything-V2、VGGT）提取深度/点图及三维特征；射线采样与高斯属性预测采用轻量MLP；融合模块使用局部与全局注意力并结合Plücker坐标；占据估计采用高斯叠加概率公式；增量更新采用基于邻域融合的无训练策略。

**📊 数据集**

实验使用Occ‑ScanNet（室内单帧/多帧）、EmbodiedOcc‑ScanNet（全景室内场景）和nuScenes（户外多摄像头）三大基准。

**📈 对比分析**

与ISO、EmbodiedOcc、GaussianFormer‑2等最新方法对比，GPOcc++在Occ‑ScanNet上单帧mIoU提升至57.41（VGGT先验）/53.88（DepthAnything），在EmbodiedOcc‑ScanNet上场景级mIoU提升至57.41；在nuScenes上实现mIoU 20.78，速度8 FPS，超过GaussianFormer‑2、QuadricFormer。

**⚠️ 局限性**

局限性包括：在平坦纹理缺失的区域（如地板、道路）占据精度下降；增量更新过程中高斯数量随序列长度增加，可能导致显存/计算增长；目前仅针对静态场景，未处理动态物体或长期地图维护。

---

## 220. HIVE-3D: Hierarchical Voxel Enhancement for High-Quality 3D Scene Generation

**arXiv ID:** 2607.13468 | [PDF](https://arxiv.org/pdf/2607.13468v1)

**作者:** Bin Zang `[一作]` (Zhejiang University), Rengan Xie `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种从单张RGB图像生成高质量3D场景的分层体素增强框架HIVE-3D，先生成粗略场景并构建层次场景树，随后通过二维分割与三维匹配得到各局部组件，利用体素超分模型在保持粗体素一致性的前提下逐级提升局部细节，最终得到高分辨率、纹理丰富的3D场景。

**💡 创新点**

创新点包括：1）将二维图像分层分割结果映射到三维空间，构造可递归使用的场景树；2）提出基于粗体素与细化图像共同条件的体素超分模型，解决生成过程中的一致性问题；3）使用形状基尺度估计与RANSAC点云配准实现不同分辨率实例的精确对齐。

**🔧 技术方法**

核心技术包括：基于flow transformer的稀疏结构体素生成模型（G_S与G_L）、2D分割与检索（Florence‑2、SAM‑2）、跨模态注意力与IP‑Adapter式适配器、稀疏VAE编码、3D Gaussian Splatting解码以及RANSAC配准。

**📊 数据集**

主要使用Objaverse‑XL 10,000件3D资产进行体素超分模型训练，并在3D‑FRONT、Real‑world scene images等数据集上进行评估。

**📈 对比分析**

与MIDI、SceneGen、Gen3DSR、Treillis等单图像3D场景生成方法对比，HIVE‑3D在几何精度（CD、F‑Score、IoU）以及视觉质量（SSIM、PSNR、LPIPS、CLIP）上均表现最优，且在3D‑FRONT上实现了0.0035的CD、84.34的F‑Score、0.7449的IoU，生成速度约36.5s。

**⚠️ 局限性**

局限性包括：1）依赖单模态粗体素生成模型，难以处理极大尺度场景；2）层次语义的二维检测与分割错误可能导致后续细化不理想；3）超分模型的体素条件融合仍有限，可能在极细纹理上不足。

---

## 221. Live Gurbani Tracking: A Benchmark and Reference System for Captioning Sikh Kirtan

**arXiv ID:** 2607.13457 | [PDF](https://arxiv.org/pdf/2607.13457v1)

**作者:** Karanbir Singh `[一作]` `[通讯]`, Karanbir Singh

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了Sikh Kirtan的实时闭源字幕基准和参考系统。

**💡 创新点**

创新点在于将完整文本匹配纳入评测指标，并针对实时与盲识别的两条任务轴构建统一框架。

**🔧 技术方法**

技术实现包括120M参数IndicConformer ASR、模糊匹配器、基于锁定的状态机，并通过INT8 ONNX实现低延迟推理。

**📊 数据集**

使用的数据集为四段公开Kirtan录音（约57分钟），每段产生三种冷启动偏移，共12个评测案例。

**📈 对比分析**

与空预测、5秒延迟预测和完美预测基线对比，参考系统在最难的live+blind变体上实现57.9%帧精度，锁定10/12 shabad，性能仍与理想指标存在显著差距。

**⚠️ 局限性**

局限包括样本量有限、仅包含单shabad录音、对录音速度和低质量音频敏感，且未覆盖多shabad连续朗诵等真实情境。

---

## 222. Adversarial Prompting Framework for AI Safety Assessment

**arXiv ID:** 2607.13453 | [PDF](https://arxiv.org/pdf/2607.13453v1)

**作者:** Yash Bhatnagar `[一作]` (Microsoft), Anirban Chatterjee `[通讯]` (Walmart Global Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Adversarial Prompting Framework（APF）用于评估生成式AI模型的安全性，生成多级结构化对抗性提示并自动评估模型响应。

**💡 创新点**

提出了五级对抗性提示分类体系，并实现了可复现的自动化测试框架，系统量化模型易受攻击程度。

**🔧 技术方法**

利用Chain-of-Thought提示生成、toxicity检测（detoxify）、情感分析（distilbert-sst2）、黑名单词汇匹配以及监督回归评分机制。

**📊 数据集**

生成约一千个涵盖25种有害政策的对抗提示，对Google、OpenAI、Anthropic、Meta、Mistral、DeepSeek等多款公开及专有基础模型进行评估。

**📈 对比分析**

通过对各模型在不同攻击层级下的平均得分进行分类（Excellent, Good, Moderate, Concerning）进行比较，发现Claude模型最稳健，开源模型对编码攻击更易受影响。

**⚠️ 局限性**

受限于阈值设定缺乏系统化、对抗提示的自动化生成仍需手工选择策略、实验仅覆盖部分模型且未验证跨语言或多模态情境。

---

## 223. Learning Physics-Guided Residual Dynamics for Deformable Object Simulation

**arXiv ID:** 2607.13451 | [PDF](https://arxiv.org/pdf/2607.13451v1)

**作者:** Shivansh Patel `[一作]` (University of Illinois Urbana-Champaign), Yunzhu Li `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Physics‑Guided Residual Dynamics (PGRD) 框架，用优化的弹簧‑质量物理模拟器作为基础，并通过学习的速度残差网络对其结果进行补偿，从而在真实世界中高精度模拟变形物体。

**💡 创新点**

创新点：① 采用速度基残差而非位置直接修正，显著提升数值稳定性；② 使用滑动窗口 Transformer 对时间依赖进行聚合，捕捉动量与历史信息；③ 采用两阶段训练：先用 CMA‑ES 优化物理参数，再用多步回放训练残差网络，使模型在长期预测中保持精度。

**🔧 技术方法**

技术手段：弹簧‑质量动力学、CMA‑ES 参数优化、点 Transformer V3 编码器、NeRF‑style 解码器、滑动窗口 Transformer 聚合、3D Gaussian Splatting 渲染、MPPI 控制与语言条件目标生成（Nano Banana Pro + Depth Anything V2）。

**📊 数据集**

数据集：6 种真实变形物体（绳、纸、旗帜、懒熊玩具、掸子、泰迪熊），使用四台 Intel RealSense D455 进行 RGB‑D 采集，结合 Grounded SAM 2、CoTracker、Depth Anything V2 等工具构建约 120 条轨迹（约 4 分钟）进行训练与验证。

**📈 对比分析**

与 5 种基线（弹簧‑质量、MPM、GBND、PGND、可微弹簧‑质量）在 3D 跟踪（MDE、CD、EMD）和视觉评估（IoU、F‑Score、LPIPS）上对比；PGRD 在所有指标与所有物体上均获得最低误差，成功率显著提升（如绳索穿槽任务 8/10 而基线仅 2/10）。

**⚠️ 局限性**

局限性：仍受弹簧‑质量基础模型的结构限制，对极大非线性或极端异质材质的物体可能表现不足；对训练数据量仍有一定依赖；在高频交互或大规模体积场景下计算负担较重。

---

## 224. ReBound: Reuse-Aware Privacy For Interactive Decision Support

**arXiv ID:** 2607.13441 | [PDF](https://arxiv.org/pdf/2607.13441v1)

**作者:** Nada Lahjouji `[一作]` (University of California, Irvine), Sharad Mehrotra `[通讯]` (University of California, Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个差分隐私缓存框架 ReBound，专门用于交互式决策支持查询的结果重用。

**💡 创新点**

创新点在于：①针对阈值变更、逻辑结构修改和线性组合等多种查询重用场景制定统一策略；②设计了三层 DAG 缓存结构以高效检索可复用子结构；③引入预算协商机制，在预算不足时提供可接受的精度折衷。

**🔧 技术方法**

核心技术包括差分隐私（ε‑expost DP）、ProBE 机制、post‑processing 复用、基于预算的误差分配以及图结构缓存和协商算法。

**📊 数据集**

实验使用了 2020 年约 300 万条 NYC Taxi 数据集，按聚合方式分组进行查询。

**📈 对比分析**

与无缓存基线相比，ReBound 在累积隐私损失上减少约 70–75%，并且在固定预算下能够完成全部 10 个查询，完成率显著提升。

**⚠️ 局限性**

局限性包括：仅实现了有限的重用类型（未覆盖部分匹配和更复杂的复合查询）；协商机制仍处于实验阶段；实验范围仅限于 NYC Taxi 数据集，缺乏对更广泛查询类型的验证。

---

## 225. TRACE-PCa: Predicting Prostate Cancer Progression from Longitudinal MRI During Active Surveillance

**arXiv ID:** 2607.13506 | [PDF](https://arxiv.org/pdf/2607.13506v1)

**作者:** Hongye Zeng `[一作]` (University of California, Los Angeles), Corey Arnold `[通讯]` (University of California, Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出TRACE-PCa，一种端到端的时序多模态深度学习框架，利用多时点bp-MRI和临床变量预测活检确诊的前列腺癌进展。

**💡 创新点**

无需病灶分割直接利用整前列腺图像，结合时间注意门将基于时间差的特征重新校准当前影像特征，实现对进展相关变化的精准捕捉。

**🔧 技术方法**

使用冻结的Swin Transformer (Triad) 3D基础模型编码MRI，三维全局平均池化+投影，时间注意门 (TAG)，多模态融合，焦点损失训练。

**📊 数据集**

来自UCLA的424名有利风险前列腺癌患者的1,221个bp-MRI序列，标注为GG≤2转GG>2的82例进展样本。

**📈 对比分析**

与临床XGBoost、基于病灶和全腺体的delta-放射组学以及放射科医师PI‑RADS判读进行5折交叉验证对比；TRACE‑PCa在AUC(0.704)、准确率(0.708)、特异性(0.716)、PPV(0.392)等指标均优于基线，且与医师判读相当或略优。

**⚠️ 局限性**

仅为单中心、样本量有限；未在多机构大规模数据上验证；缺乏对不同MRI协议兼容性的评估。

---

## 226. M2P-AD: Memory-to-Prototype Learning with Boundary-aware Score Refinement for 3D Anomaly Detection

**arXiv ID:** 2607.13499 | [PDF](https://arxiv.org/pdf/2607.13499v1)

**作者:** Seyoung Jeong `[一作]` (Jeonbuk National University), Sang Jun Lee `[通讯]` (Jeonbuk National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Memory-to-Prototype Anomaly Detection (M2P-AD)框架，解决3D点云异常检测中正常区域异常过高和边界误报的问题。

**💡 创新点**

创新点包括：① M2P模块通过聚类学习代表性原型来保留结构信息；② BE+BSR策略提取边界并对异常分数进行校准，显著抑制边界误报。

**🔧 技术方法**

采用多尺度FPFH特征提取、K-means聚类+欧氏+余弦损失构建原型，基于距离相似度计算异常分数，边界通过投影与距离变换获取并使用top‑n%分数校准。

**📊 数据集**

在Real3D-AD、Anomaly-ShapeNet和MulSen-AD三大工业点云异常检测基准上进行评估。

**📈 对比分析**

与ISMP、PO3AD、MC3D‑AD、Simple3D等现有方法对比，M2P‑AD在O‑AUROC/P‑AUROC上均居前列，Real3D‑AD平均93.2/94.3，Anomaly‑ShapeNet 89.1/94.3，MulSen‑AD 98.2/96.9等。

**⚠️ 局限性**

局限性在于对极小样本或极端形变仍易出现误检，BE+BSR仅针对投影边界有效，超参数K、n需调优，且不易直接扩展到大规模实时场景。

---

## 227. Factorized Spectral Representations for Reinforcement Learning

**arXiv ID:** 2607.13498 | [PDF](https://arxiv.org/pdf/2607.13498v1)

**作者:** Junyi Wu `[一作]` (University of Washington), Dan Li `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出FaStR框架，通过将转移核视作三维张量，学习独立的状态、动作和下一状态编码器，构建三线性（CP分解）特征；

**💡 创新点**

创新点在于把低秩Mdp的低秩结构从联合状态-动作编码迁移到张量分解，利用CP分解的稀疏参数结构降低表示学习样本复杂度，并实现模块化迁移；

**🔧 技术方法**

技术主要包括：CP（CANDECOMP/PARAFAC）张量分解、基于RP-NCE的噪声对比学习、Hadamard乘积特征构造、线性Mdp理论证明及覆盖数分析；

**📊 数据集**

使用DM Control Suite中的多种连续控制任务（包括Humanoid、Quadruped、Dog、Walker等）进行实验；

**📈 对比分析**

与控制器编码器（CTRL‑SR）、Diff‑SR以及SAC、TD7等基线对比；实验显示在动作维度较高且与CP结构匹配的任务上FaStR取得更快收敛，最终收益与或优于基线；在动作接口漂移的迁移设置中，仅需微调动作编码器即可恢复性能；

**⚠️ 局限性**

局限性包括：对CP分解的依赖，当真实动力学强耦合或不符合CP假设时性能下降；对张量分解的梯度覆盖有限，可能需要更大批量；以及在低动作维度任务中优势不明显。

---

## 228. DeepLoop: Depth Scaling for Looped Transformers

**arXiv ID:** 2607.13491 | [PDF](https://arxiv.org/pdf/2607.13491v1)

**作者:** Shuzhen Li `[一作]` (Princeton University), Mengdi Wang `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Loop Transformer 的残差缩放修正方法 DeepLoop，解决物理块重复访问导致的梯度累积与不稳定问题

**💡 创新点**

通过引入访问对齐系数 κ_R 并推导出对齐情况下所需的残差缩放指数 p=1/2，给出单行可复用参数的 Post‑LN 归一化缩放规则 α=(2N)^{1/2}, β=(8N)^{-1/2}

**🔧 技术方法**

利用 DeepNorm 理论、RMSNorm、残差缩放、梯度对齐分析以及大规模 GPT‑2 预训练和 ARC‑AGI 推理实验验证方法

**📊 数据集**

FineWeb‑Edu 50B 语料（GPT‑2 小/中尺度预训练）和 ARC‑AGI‑1 推理数据集

**📈 对比分析**

与同构循环模型（仅使用 α=β=1）对比，在 R≥3 时验证损失下降 0.02–0.03 nats、下游 0‑/1‑shot 任务平均准确提升至 52–56%，ARC‑AGI 投票准确率提升 3–4个百分点

**⚠️ 局限性**

仅针对最保守对齐情形给出 p=1/2 规则，未测得 κ_R 具体数值，未评估更大规模模型或不同参数化下的适用性，单种随机种子实验导致结果方差可能不稳定

---

## 229. Joint On-and-Off Policy Learning for Vision-and-Language Navigation

**arXiv ID:** 2607.13461 | [PDF](https://arxiv.org/pdf/2607.13461v1)

**作者:** Qingrong He `[一作]` (Joy Future Academy), Liang Lin `[通讯]` (Joy Future Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 JOP‑VLN 框架，结合离线模仿学习与在线强化学习，分三阶段训练：多任务预训练、DAgger 轨迹模仿、联合 GRPO 与 CHORD‑ec 的 on‑and‑off 策略。

**💡 创新点**

创新点在于（1）三阶段训练顺序将离线与在线学习有效融合；（2）对 DAgger 轨迹进行高熵采样与错误修正优先排序，避免梯度消失与过拟合；（3）将 CHORD 框架迁移到 VLN 并针对 VLN 设计专门的采样与排序策略。

**🔧 技术方法**

使用 Qwen3‑VL‑8B‑Instruct 作为主干模型，结合 GRPO（带 KL 正则）、CHORD‑ec（混合 IL‑RL 损失）以及高熵轨迹采样与错误优先排序技术。

**📊 数据集**

训练与评估使用 VLN‑CE 基准 R2R 与 RxR（Val‑Unseen），同时在 R2R、RxR 的多源数据（EnvDrop、ScaleVLN）中收集 DAgger 轨迹；在真实环境中采用 Unitree Go2 四足机器人与 Intel RealSense 进行少量适配实验。

**📈 对比分析**

在 R2R Val‑Unseen 上以单视 RGB 为输入，SR 69.9%（SPL 64.9%），超过前沿 CorrectNav 的 65.2%；在 RxR Val‑Unseen 上 SR 68.0%、SPL 59.3%，显示出优异的长距离决策能力；在实地测试中，少量 10‑shot 适配后能稳健跟随指令。

**⚠️ 局限性**

局限性包括：对长指令中视觉上相似地标的歧义易导致误入；依赖大量离线轨迹与 DAgger 收集，且对不同场景的迁移仍需要少量适配；通信延迟与单视角输入在某些复杂场景下仍是瓶颈。

---

## 230. LPM: Industrial-Scale Generative Video Restoration

**arXiv ID:** 2607.13460 | [PDF](https://arxiv.org/pdf/2607.13460v1)

**作者:** Bichuan Zhu `[一作]` (Kuaishou Technology), Bin Yu `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发并部署了基于扩散模型的 LPM（Large Processing Model），实现从用户生成内容中去噪、去模糊、去压缩伪影，同时保持细节与时序一致的工业级视频恢复框架。

**💡 创新点**

创新点包括：① 先训练图像恢复再迁移到视频的逐步训练策略；② 采用因式分解的 2D+1D DiT 与无位置编码的时序注意力，支持长视频无窗口推理；③ 通过时序金字塔推理与 mask 引导的跨剪辑一致性，消除跨窗口漂移；④ 结合 Consistency Model 蒸馏与 3 阶段训练，实现单步高质量推理。

**🔧 技术方法**

使用技术涵盖扩散 Transformer、Rectified Flow、SwiGLU+RMSNorm、RoPE、LPM‑VAE、Mask‑Guided Conditioning、Temporal‑Pyramid Inference、Consistency Model（LCM）蒸馏、FP8/INT8 量化以及 TensorRT‑LLM 部署。

**📊 数据集**

训练与评估数据集为自建的 KwAI UltraVision 图像/视频大规模数据集（约 10 亿样本、KVQ 平均 4.31、12 类场景）以及高质量子集（KVQ>4.7）和细节强化数据；用于公开基准 RealSR/RealSet/VideoLQ/YouHQ/REDS/SPMCS 及自制 LPM‑Benchmark。

**📈 对比分析**

在 RealSR/RealSet（图像）和 VideoLQ/YouHQ/REDS/SPMCS/LPM‑Benchmark（视频）上与 StableSR、SeeSR、DiT‑SR、SeedVR2、FlashVSR、Vivid‑VR 等公开/商业基线对比，LPM 在 PSNR、MUSIQ、CLIP‑IQA、KVQ 等感知指标上显著领先，长视频一致性无抖动，单步推理速度提升至 39.5×。

**⚠️ 局限性**

局限性：训练分布聚焦常见 UGC 类别，导致对长尾自然场景表现欠佳；当输入已接近高质量时感知提升有限；需要更广泛的数据与质量自适应机制以进一步提升效果。

---

## 231. Reverse to Advance: Teleoperation-Cost Effective Hard Policy Learning from Reversed Easy Tasks

**arXiv ID:** 2607.13455 | [PDF](https://arxiv.org/pdf/2607.13455v1)

**作者:** Qiyuan Qiao `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自动化利用易任务的时间反转来学习难任务的框架 Auto-E2H。

**💡 创新点**

通过闭环数据采集、分层运动与评价滤波以及迭代策略学习三大模块，实现在无需大量昂贵演示的前提下将易任务轨迹转化为高质量硬任务监督。

**🔧 技术方法**

结合 Diffusion Policy、运动学先验滤波、Critic 价值函数优势滤波及在线迭代学习。

**📊 数据集**

在 Isaac Lab、Robosuite 仿真环境以及四个真实机器人任务（Disk、Mouse、Test Tube、Brush）上进行验证。

**📈 对比分析**

与 TR-DRL、RECAP、TR-DRL-DP 及直接硬演示基线对比，Auto-E2H 在所有环境中实现了从约30% 到 96% 的成功率提升，显著优于基线且训练更稳定。

**⚠️ 局限性**

仅适用于可逆且方向不对称的操控任务，强摩擦、非可逆抓取或动态变化较大的场景下反转轨迹噪声过大，影响效果。

---

## 232. DriveFace: A Cross-Spectral Through-Glass Face Dataset for On-the-Move Vehicular Border Control

**arXiv ID:** 2607.13515 | [PDF](https://arxiv.org/pdf/2607.13515v1)

**作者:** Anjith George `[一作]` (Idiap Research Institute), Sebastien Marcel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并公开了DriveFace数据集，用于跨光谱、透玻璃下的车辆边境控制人脸识别与防欺骗研究。

**💡 创新点**

首次提供一套真实车内外环境、不同窗口色度、运动、光照等多模态数据，并建立跨光谱验证与伪造检测基准。

**🔧 技术方法**

采用基于ViT/ResNet的AdaFace、LVFace、EdgeFace等公开模型，并通过xEdgeFace进行跨光谱自适应。

**📊 数据集**

数据集包括70名志愿者的RGB预登记图像和NIR透玻璃采集视频，包含室外、室内、模拟窗色等三种协议；还附带印刷、回放、遮罩三类攻击样本。

**📈 对比分析**

与AdaFace、LVFace、EdgeFace、xEdgeFace等模型进行对比，户外协议下准确率>95%，但在模拟色度与室内协议下AUC下降至≈88–94%，提示跨光谱适配有明显提升。

**⚠️ 局限性**

主要局限在于光谱差异仍显著、模拟实验与实际车窗差异、以及在未见攻击场景下PAD性能大幅下降。

---

## 233. Exploring the Alignment of Generation and Understanding in Protein Structure Modeling

**arXiv ID:** 2607.13503 | [PDF](https://arxiv.org/pdf/2607.13503v1)

**作者:** Junde Xu `[一作]` (Chinese University Of Hong Kong), Pheng Ann Heng `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了蛋白质生成模型与理解模型之间的表征差距，并提出在训练过程中将扩散生成模型的中间表示与预训练的蛋白质理解模型对齐，从而提升生成质量。

**💡 创新点**

创新点在于首次将 Representation Alignment（REPA）框架应用于蛋白质扩散模型，通过轻量级投影头在训练阶段对齐两类模型的表征，显著提升了功能蛋白的生成效果，并在多个基准上实现了加速收敛与更高的多样性。

**🔧 技术方法**

使用的技术包括：扩散生成模型（Protpardelle‑1c、RFDiffusion）、预训练的蛋白质语言模型（ESM2）和结构编码器（ProteinMPNN），以及对齐头的 MLP 投影和残差映射；训练目标为结构去噪损失加上对齐损失。

**📊 数据集**

主要使用的数据集有 MotifBench（30 个基准 motif）、CATH 4.4‑S40（天然蛋白域集）、Gene Ontology 与 Enzyme Commission（功能分类评估）以及 RFDiffusion 的 26 个复杂 motif。

**📈 对比分析**

通过在 MotifBench 上的实验，ReaPro‑1c 将 MotifBench 得分从 39.2 提升到 47.1（提升约 20%），在 RFDiffusion 基准中超过基线 22/26 个 motif；在 CATH 4.4‑S40 上的 Fréchet Protein Distance（FPD）亦优于原基线，说明生成分布更贴近天然蛋白结构；在功能分类任务中，生成模型的 F1 分数显著低于理解模型，验证了表征差距。

**⚠️ 局限性**

局限性包括：只在单一层进行对齐，可能未充分利用多层信息；对齐使用的是单一理解模型，未尝试多模态或多模型融合；在更复杂的全原子生成任务中仍有性能差距；以及对齐策略对模型训练步骤与资源的敏感性未深入探讨。

---

## 234. GeoAnchor: Collaborative Reasoning via Latent Decomposition for 3D Spatial Understanding

**arXiv ID:** 2607.13454 | [PDF](https://arxiv.org/pdf/2607.13454v1)

**作者:** Hao Li `[一作]` (Shanghai Jiao Tong University), Hao Sun `[通讯]` (Xingchen AGI Lab, China Telecom Artificial Intelligence Technology Beijing Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于文本-潜在交互的 3D 空间推理框架 GeoAnchor，旨在提升多模态大语言模型在从二维图像推断三维空间关系的能力。

**💡 创新点**

创新点主要有：①将三维空间信息拆解为位置、方向、几何三种连续潜在标记，消除单一潜在瓶颈；②在文本与潜在之间交替生成，保持语义规划与几何推理的分离；③四阶段协同训练策略（从本地感知到全局语义，再到潜在放松和强化学习），实现自适应潜在选择与解释性推理。

**🔧 技术方法**

核心技术包括：文本-潜在交互框架、连续潜在投影器、位置/方向/几何潜在标记的轻量头监督、软覆盖多尺度 VGGT 对齐、以及基于 Group Relative Policy Optimization 的潜在强化学习。

**📊 数据集**

使用的数据集主要有：ScanNet 生成的 550k 3D grounding 数据；SPAR（含 105k 任务样本）以及 SPBench、ViewSpatial 等公开基准；另外从 SpatialLadder-26k 补充 5k 中心尺度问答样本。

**📈 对比分析**

在 SPAR‑Bench、SPBench 与 ViewSpatial 上的平均准确率分别为 68.4%、69.7% 与 47.0%，比基准 Qwen3‑VL‑2B 提升 21.2% 并分别优于 GPT‑4o（18.0%）和 Gemini‑2.5‑Flash（13.6%），在外域测试上实现 10.7% 的性能提升，显示了显著的竞争力与泛化能力。

**⚠️ 局限性**

局限性包括：①需要多阶段、耗时的协同训练与大量标注的 3D grounding 数据；②对极端多视角或极端遮挡场景的推理仍可能受限；③模型规模虽不大（2B）但在极大场景或实时部署时仍需进一步优化。

---

## 235. Symbiosis-Inspired Knowledge Distillation for Incremental Object Detection

**arXiv ID:** 2607.13452 | [PDF](https://arxiv.org/pdf/2607.13452v1)

**作者:** Mingyue Zeng `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Symbiosis‑Inspired Knowledge Distillation (SIKD) 框架，用于增量目标检测，结合空间与语义共生蒸馏以在学习新类别的同时保持旧类别知识。

**💡 创新点**

创新点包括：① 重新把增量检测视为对象共生（共现与遮挡）问题，利用共生信息；② 在空间层引入 Consistent Feature Enhancement（CFE）并对齐查询，提升对共生区域的特征一致性；③ 在语义层构造加权原型并用软排名对齐，保持旧类别的语义拓扑；④ 采用统一特征空间而非传统分离策略，减少旧新类别混淆。

**🔧 技术方法**

采用 Deformable DETR 作为基准检测器，结合多头自注意力+MLP 的 CFE、概率加权原型、soft rank 对齐、KL、L1、GIoU 损失，以及伪标签分区与过滤技术。

**📊 数据集**

使用 COCO 2017（80 类）与 DIOR（20 类）数据集进行实验，分别在 70+10、40+40 等增量设置下评估。

**📈 对比分析**

与 DyQ‑DETR、DCA、GLIP‑TLR、CL‑DETR 等现有方法对比，COCO 70+10 下 AP 提升 1.9–3.0 点，40+40 下 0.9–0.5 点；在多任务设置中均刷新 SOTA；DIOR 上 AP_50 提升 6.2–9.0 点。

**⚠️ 局限性**

局限性包括：依赖伪标签质量，若伪标签噪声大可能影响蒸馏效果；相对较高的计算与显存开销；对极少量样本的类别仍易遗忘；目前未充分利用更强上下文或 LLM 信息，未来可进一步提升。

---

## 236. A proof complexity perspective on effectively zero-knowledge proofs

**arXiv ID:** 2607.13540 | [PDF](https://arxiv.org/pdf/2607.13540v1)

**作者:** Jan Krajicek `[一作]` `[通讯]` (Charles University), Jan Krajicek (Charles University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了有效零知识证明的逻辑视角，并用证明复杂性工具证明其存在性及“与真命题不可区分”性质，进一步展示如何在有共享随机串的前提下将其转化为真正的零知识证明。

**💡 创新点**

创新点在于将有效零知识证明定义为与理论相关的ZK，利用模型论和证明复杂性生成器，将非交互式可证明不可区分性转化为真正的ZK，并给出仅依赖于理论一致性的证明。

**🔧 技术方法**

采用数学逻辑、模型论、证明复杂性生成器、非交互式可证明不可区分性（NIWI）等技术。

**📊 数据集**

无数据集。

**📈 对比分析**

无实验比较，理论性工作，无性能指标。

**⚠️ 局限性**

主要限制是对证明复杂性生成器存在性的强假设（硬生成器或半位）的依赖，以及需要共享随机字符串的假设；构造难题序列的实际可行性仍未解决。

---

## 237. VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following in Video Generation

**arXiv ID:** 2607.13527 | [PDF](https://arxiv.org/pdf/2607.13527v1)

**作者:** Songyu Xu `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种可解释、诊断性的视频生成模型评估框架 VGIF-Score 及其基准 VGIF-Bench。

**💡 创新点**

将指令解析为空间-时间有向无环图(ST-DAG)生成依赖感知 QA 并结合自动化 rubric，实现对象化完成与主观满意度双分支评估，提供细粒度错误定位。

**🔧 技术方法**

基于 LLM 的 ST-DAG 解析、依赖感知 QA 与短路机制，VLM（Gemini‑3.1‑Pro）评估问答与 rubric 打分，以及 AutoRubric 生成条件评分。

**📊 数据集**

VGIF‑Bench，包含 223 条长复杂提示、约 4.3k 细粒度 QA 与 rubric 项。

**📈 对比分析**

在 14 种视频生成模型上评估，专有模型平均 VGIF‑Score 约 46.6，开源模型约 34.7，显示当前模型在因果链、深层依赖和后置约束上仍表现不足。

**⚠️ 局限性**

框架依赖 LLM 与 VLM 的准确性，基准规模相对有限，未覆盖所有实际场景，对极端长篇或复杂因果链的鲁棒性仍待验证。

---

## 238. Accelerating gas-network feasibility screening with a physics-informed graph neural network surrogate

**arXiv ID:** 2607.13610 | [PDF](https://arxiv.org/pdf/2607.13610v1)

**作者:** Dongrui Jiang `[一作]` (Technische Universitaet Berlin), Joachim Müller-Kirchenbauer `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种物理信息图神经网络代理，用于快速模拟天然气网络的稳态流动并进行可行性筛选

**💡 创新点**

创新点在于将管道级平方压差和流量预测与可微质量守恒投影及拉普拉斯压力重构相结合，实现了严格的质量守恒和拓扑一致的压力场，同时保持高预测精度

**🔧 技术方法**

采用边缘中心的图神经网络（GINEConv）配合可微的投影层和拉普拉斯求解器，形成端到端可微的物理约束模型

**📊 数据集**

使用GasLib-134、GasLib-135和GasLib-582三个公开基准网络生成的约1000至5000个可行稳态情景进行训练和测试

**📈 对比分析**

与传统MYNTS求解器相比，代理在GasLib-582上实现压力MAE1.05 bar、R²0.981、流量R²0.972，质量残差达到10⁻⁵–10⁻⁴ Nm³/s，推理时间从几秒降低到不到40毫秒

**⚠️ 局限性**

局限性包括仅适用于稳态、等温、无主动装置的情景，压差-流量单向性仅通过软约束实现，难以捕捉局部非平衡或异常操作，以及对拓扑泛化和极端负荷场景的鲁棒性需进一步提升

---

## 239. Active Trust Management for Successful Human-Robot Teaming: Moving from a Trust Repair to a Trust Satisficing Perspective

**arXiv ID:** 2607.13595 | [PDF](https://arxiv.org/pdf/2607.13595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 240. Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs

**arXiv ID:** 2607.13609 | [PDF](https://arxiv.org/pdf/2607.13609v1)

**作者:** Mohammad Forouhesh `[一作]` `[通讯]` (Amirkabir University of Technology), Mohammad Forouhesh (Amirkabir University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究如何从有向图上的流量反推潜在势场，指出传统岭正则化因“gauge mismatch”导致势场范围坍塌并颠倒排序，提出使用与图Dirichlet能量相同的gauge‑invariant正则化（图Sobolev）以解决该问题。

**💡 创新点**

创新点在于：①证明gauge‑invariant正则化在有向Dirichlet Poisson逆问题中保持势场的排序与动态范围，且对正则化参数λ几乎不敏感；②构造了SPD的约简系统并给出链上精确范围保持的解析公式；③提出仅凭流量即可识别吸收边界的Poisson残差诊断；④将此逆问题结果验证为可用的节点特征，缓解了图神经网络的过平滑问题。

**🔧 技术方法**

技术上采用离散Poisson方程与有向图拉普拉斯算子，利用gauge‑invariant图Dirichlet能量（即图H¹半范数）作为正则化；对齐流量后通过拓扑排序或Hodge分解获得有向无环子图；使用共轭梯度求解约简的SPD系统；在真实数据上结合逻辑回归、Spearman、Pearson、NDCG@5和ROC‑AUC等指标评估。

**📊 数据集**

数据集包括：①人工生成的“instrument”多支流动漏斗（277节点、5个汇点）；②三个公开点击流语料库的事件型状态空间——RetailRocket（4节点）、Trivago（11节点）和OTTO（105节点）。

**📈 对比分析**

比较方法：对同一数据集分别用岭正则化（λ>0）和图Sobolev正则化（λ=1），在合成数据上评估Spearman、Pearson、NDCG@5，结果显示图Sobolev在λ∈[10⁻³,10]内保持+0.81的秩相关和≈0.98的线性相关，而岭正则化在任何λ>0均会将秩相关降至≈-0.42。真实数据上，图Sobolev保留了28–41%的动态范围，提升ROC‑AUC约3.1个百分点；岭正则化仅保留≤0.2%范围，几乎没有性能提升。

**⚠️ 局限性**

局限性包括：①正则化只能保证秩和范围的保持，无法恢复真实势场幅值；②对流量采样量和流向结构的依赖较强，低采样或高丢失率会导致恢复信号减弱；③方法假设图为有向无环，需先行流量提取与子图构造，抽取方式可能影响结果；④目前仅考虑线性欧氏能量，尚未探索更鲁棒的边缘保持或非线性正则化。

---

## 241. Automatic Ordinary Differential Equations Discovery For Biological Systems Using Large Language Model Powered Agentic System

**arXiv ID:** 2607.13608 | [PDF](https://arxiv.org/pdf/2607.13608v1)

**作者:** David Krongauz `[一作]` (Weizmann Institute of Science), Teddy Lazebnik `[通讯]` (University of Haifa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MEDA，一个基于大型语言模型和约束符号回归的代理系统，用以自动发现和评估生物系统的常微分方程模型。

**💡 创新点**

创新点在于将 LLM 作为知识检索与形式化的中介，将生成的中间科学工件（变量、约束、种子方程）纳入可审计的发现流程，强调机制可解释性而非仅追求符号形式匹配。

**🔧 技术方法**

技术核心包括 LLM 文献检索与约束生成、Typed Grammar 形式化、基于进化搜索的稀疏多项式 ODE 发现、数据驱动拟合与交互式评估。

**📊 数据集**

使用了十余种经典与扩展的生物动力学模型（如 Logistic、Lotka–Volterra、SIR、Allee、FitzHugh–Nagumo 及其变体、谣言扩散与慢性伤口模型），以及对应的模拟时序数据。

**📈 对比分析**

与全系统、无 EDA、无文献、预训练形式化以及纯 SINDy 的对照组比较，实验显示保留知识引导形式化的系统在变量、约束与项级 F1 近 1，专家可解释性得分平均 0.97；去除知识或仅用数据时性能显著下降。

**⚠️ 局限性**

局限性包括：仅覆盖有限的多项式 ODE 结构，未处理 Hill 效应、延迟或随机性；数据集为合成模拟；可解释性评分主观；缺乏对更复杂生物网络、空间或混合离散-连续模型的验证。

---

## 242. Semantic Anchoring for Robotic Action Representations

**arXiv ID:** 2607.13597 | [PDF](https://arxiv.org/pdf/2607.13597v1)

**作者:** Yuan Xu `[一作]` (Peking University), Yizhou Wang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在机器人视觉语言动作（VLA）模型微调时保持预训练语义结构，并提出一种无额外推理成本的语义锚定方法；

**💡 创新点**

创新点在于将动作表示分解为共享语义通道和私有执行通道，并通过对齐到外部语义流形（EgoHOD）实现对动作特征的对比学习，从而在微调过程中保护并恢复预训练的语义结构；

**🔧 技术方法**

使用对比信息论损失（InfoNCE）、注意力池化、共享/私有通道的分解网络（DSN）以及预训练的视觉语言模型作为冻结语义参考；

**📊 数据集**

在LIBERO（机器人演示数据）、SIMPLERENV（桥接数据集）以及真实双臂机器人平台上评估；

**📈 对比分析**

与仅使用动作损失的基线以及ACT、Diffusion Policy等方法进行对比；在模拟和真实机器人实验中，方法在分布内任务提升约18.7%，在分布外泛化提升约21.5%，并在多种任务套件和数据集上均表现出显著提升；

**⚠️ 局限性**

局限在于依赖于冻结的对齐目标（EgoHOD），若语义参考更强可能进一步提升；并且仅在任务特定演示的后期微调阶段使用，对预训练阶段的影响尚未探索。

---

## 243. SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing

**arXiv ID:** 2607.13594 | [PDF](https://arxiv.org/pdf/2607.13594v1)

**作者:** Tianyu Chen `[一作]` (ShanghaiTech University), Wenjie Wang `[通讯]` (ShanghaiTech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三路决策模型（Execute/Ask/Refuse）用于LLM代理的工具调用审计，取代传统的二元安全判断；

**💡 创新点**

创新点在于将安全决策细化为三类路由，并通过可调阈值实现不同风险容忍度的后期调整，兼顾自主性与人机监督；

**🔧 技术方法**

采用监督微调（LoRA）在大语言模型上训练Guard模型，模型输出包含推理、决策词和JSON负载；

**📊 数据集**

构建了9,203条步骤级别审计数据，来源于9个真实自托管企业服务（如Gitea、Rocket.Chat、Vaultwarden等）并结合Persona记忆和公开基准；

**📈 对比分析**

在ID测试集上，Safety Sentry以91%+准确率、90%+宏F1和显著低的误警/误忽略率击败了多种开源与闭源基线；在OOD服务Mailu、不同框架/后端组合以及多用户情景下亦保持稳定性能；

**⚠️ 局限性**

局限性包括仅在企业办公工具场景验证，创意写作等开放域未评估；阈值需手动调优，缺乏自动化校准机制。

---

## 244. IMMNet: Hybrid Fusion of Model-based and Data-driven Approaches for Maneuvering Target Tracking

**arXiv ID:** 2607.13573 | [PDF](https://arxiv.org/pdf/2607.13573v1)

**作者:** Yixuan Zhao `[一作]` (Southeast University), Ting Yuan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了IMMNet——一种融合了传统IMM模型结构和神经网络学习能力的混合模型/数据驱动的三维机动目标跟踪算法；

**💡 创新点**

创新点在于将KalmanNet与Transformer相结合，取代IMM中固定的Markov状态转移与Kalman增益，形成可学习的模型概率更新与自适应噪声处理，既保留贝叶斯推理的可解释性，又提升了对未知噪声和模型不匹配的适应性；

**🔧 技术方法**

采用KalmanNet作为四个运动模式的专家滤波器，使用Transformer作为运动模式分类器，并通过端到端和模块化训练实现融合；

**📊 数据集**

使用自行构建的大规模3D机动目标跟踪数据集3D-LAST，该数据集包含100,000条带有多模式切换、真实测量噪声、速度与转弯率约束的三维轨迹；

**📈 对比分析**

在3D-LAST测试集上与多种IMM变体（已知噪声、仅已知上界、仅已知下界）比较，IMMNet的平均RMSE为0.29 m，显著低于IMM-Accurate（1.32 m）及其他变体，且在模式切换点处误差更小；

**⚠️ 局限性**

限制在于目前仅针对单目标跟踪，未考虑多目标、遮挡与复杂雷达测量模型，且Transformer对短期窗口的依赖可能限制实时性和对极端动态变化的捕捉；

---

## 245. A Simple Obligation to Metric Interval Temporal Logic

**arXiv ID:** 2607.13598 | [PDF](https://arxiv.org/pdf/2607.13598v1)

**作者:** Patricia Bouyer `[一作]` (Université Paris-Saclay), Vaishnavi Vishwanath `[通讯]` (Chennai Mathematical Institute)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种基于时间约束义务（obligation）的新方法，用以判定 Metric Interval Temporal Logic (MITL) 的可满足性。

**💡 创新点**

创新点在于：①不通过传统的自动机翻译；②直接使用义务跟踪和时间信息；③通过简单的合并/消除规则在整个时序词中保持义务数目有界；④构造符号化的地区图实现 EXPSPACE 复杂度。

**🔧 技术方法**

主要技术包括：义务系统的定义与转移规则、基于时间抽象的地区等价关系、时间抽象 bisimulation、一般化 Büchi 接受条件、符号化地区图构造。

**📊 数据集**

无实验数据集；本工作主要是理论分析与复杂度证明。

**📈 对比分析**

与现有 MITL 可满足性工具（如 MightyL、MightyPPL、TEMPORA）相比，本方法在理论复杂度上保持最优（EXPSPACE/PSPACE），但未给出具体性能评测或实测比较。

**⚠️ 局限性**

局限性包括：①仅针对 MITL；②依赖于严格的时间约束义务设计，可能难以直接扩展到更一般的 MTL；③符号化地区图的实现仍面临状态空间爆炸的挑战。

---

## 246. Cost-Pragmatic Quality Gating and Selection-Fusion Multi-Model Combiners for BioASQ Phases A+ and B

**arXiv ID:** 2607.13551 | [PDF](https://arxiv.org/pdf/2607.13551v1)

**作者:** Dima Galat `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套端到端的 BioASQ Task 14B 2026 检索增强生成系统，融合混合检索、BGE 质量门控、Agent 迭代检索以及选择‑融合组合策略；

**💡 创新点**

主要创新在于：① 量化成本‑质量的可重检索策略；② 用选择‑融合分解指导组合器选择；③ 设计同义词联合解析器实现融合提升；④ 基于指标结构的组合器选择原则；

**🔧 技术方法**

技术组合包括：BGE dense encoder、BM25、RRF、BGE cross‑encoder 门控、Agent 概念拆分与查询多样化、iCite 引用扩展、Claude Opus 4.6、Gemini 2.5 Pro、GPT‑5.5 LLM；

**📊 数据集**

使用的数据集包括 PubMed ≈ 58k 文档、BioASQ‑13b 历史档案、Task 12B 2024 验证集、Task 13B 2025 黄金输入、Task 14B 2026 现场批次；

**📈 对比分析**

通过在验证、测试及现场排行榜上多种检索与组合器方案的基准对比，系统在 Task 14B 2026 多项指标领跑，在 Task 13B 2025 亦名列前茅，list F₁ val→test 差距 +0.132，体现检索提升空间；

**⚠️ 局限性**

局限性包括：验证样本量有限导致置信区间波动、检索‑答案差距受问卷差异影响、ROUGE 评估不兼容官方指标、重检索成本估计与实际差异、3‑stage 列表团队的效果验证不足。

---

## 247. From Novice to Expert: Cost-Aware Bandits for Evolving Worker Performance in Crowdsensing

**arXiv ID:** 2607.13546 | [PDF](https://arxiv.org/pdf/2607.13546v1)

**作者:** Yin Huang `[一作]` (University of Florida), Jie Xu `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在预算受限的移动众包平台中，如何在工作者随经验学习的奖励随时间先上升后趋于平稳的结构化情形下进行在线工作者招募与任务分配。

**💡 创新点**

提出了 CATI‑UCB 算法，首次将奖励成本比、在线线性学习与滑动窗口变化点检测融合，解决非平稳奖励与未知成本同时存在的挑战，并给出了 O(log B) 的理论上界。

**🔧 技术方法**

采用结构化多臂赌博机框架，构造时间递增 UCB、在线线性回归预测工作者学习曲线、滑动窗口变化点检测以及成本感知的 UCB 决策规则。

**📊 数据集**

实验使用三类数据：① 合成的分段线性与负指数奖励函数（单、12 只工作者），② 多任务类型（2 类任务、12 只工作者）模拟，③ 基于 Topcoder 真实工作者性能曲线的追踪实验。

**📈 对比分析**

与 Primal‑Dual、TIUCB、Budget‑TIUCB、UCB_c 等基线相比，CATI‑UCB 在所有预算和实验设置下均取得最低累计 regret、最高奖励/成本比、最长任务执行时间以及最高最佳工作者选取比例。

**⚠️ 局限性**

局限性包括：对任务类型数量扩展时统计效率下降；对组合式任务（多工作者一次选择）理论分析尚未完成；窗口大小与阈值需手工调参；仅考虑单任务选取，未涵盖多任务并行或动态任务到达情形。

---

## 248. When T2I Synthetic Data Backfires: Amplified Privacy Risks in Real-Synthetic Mix Training

**arXiv ID:** 2607.13541 | [PDF](https://arxiv.org/pdf/2607.13541v1)

**作者:** Na Li `[一作]` (Nanjing University of Science and Technology), Anmin Fu `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在将文本到图像（T2I）生成的合成数据与真实数据混合训练（RSMT）时，真实训练样本的隐私泄露是否会被放大，并提供了理论和实验的评估框架。

**💡 创新点**

创新点在于提出并证明了“RSMT记忆放大”理论，揭示合成数据导致真实样本向特征空间边缘漂移，从而迫使模型更强记忆；同时设计了非对抗与对抗两种审计方法，并提出仅基于真实数据的泄露倾向指标I_c，用于预评估RSMT风险。

**🔧 技术方法**

使用的技术包括：文本到图像生成模型（SD1.5、Flux-mini、Google Nano Banana、OpenAI ChatGPT Images 2.0）；ResNet-18 作为下游分类器；多种成员推断攻击（训练型和基于度量的）评估隐私泄露；理论推导涉及分布差异、特征中心偏移和记忆化分数；以及自监督与 LoRA 微调技术生成合成数据。

**📊 数据集**

使用的公开数据集包括 EuroSAT、PatternNet、VGGFace2（10/3类）、ImageNet10（两种子集）和 ImageNet100，均覆盖卫星影像、遥感图像、面部识别和通用分类场景。

**📈 对比分析**

与两类基线（同样数量真实样本 B1 与准确率匹配的真实样本 B2）比较，RSMT 在所有数据集和 T2I 模型上均显著提升了下游模型的分类准确率；同时在成员推断攻击中，TPR@0.1% FPR、AUC 与准确率均高于 B2，证明隐私泄露被放大；对抗攻击进一步提升泄露率而保持实用性。

**⚠️ 局限性**

局限性包括：实验仅覆盖成员推断攻击，未验证对其他隐私风险（如模型重建）的影响；RSMT 对合成模型的分布差异 δ 的敏感性意味着不同模型或数据集可能表现不一；攻击者的假设（黑盒查询、对抗性 T2I 供应商）在实际部署中的可行性仍待进一步评估；最后，提出的 I_c 指标需要在更多场景下验证其通用性。

---

## 249. Kepler-Encoder-v0.1: Towards a Multimodal Embedding Model for Robots

**arXiv ID:** 2607.13522 | [PDF](https://arxiv.org/pdf/2607.13522v1)

**作者:** Ishneet Sukhvinder Singh `[一作]`, Jia Qi Yip `[通讯]` (Menlo Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Kepler-Encoder-v0.1，一种融合视觉、关节角度、力/扭矩等多模态信息的跨模态注意力编码器，只用视觉即可恢复大部分机器人状态。

**💡 创新点**

创新点在于将机器人状态视为模态，使用跨模态掩码预测自监督训练，并采用学习查询交叉注意力瓶颈实现跨实体、跨传感器的统一编码。

**🔧 技术方法**

使用LeJEPA/SIGReg的跨模态自监督目标、学习查询交叉注意力、掩码投影、正则化反坍缩等技术；模型仅训练约2M参数，ViT backbone保持冻结。

**📊 数据集**

在RH20T数据集上训练，该数据集包含7个配置、4种机器人（UR5、Flexiv、Franka、KUKA）及其多种传感器（RGB、关节、力/扭矩、TCP姿态）。

**📈 对比分析**

与预训练ViT、PCA压缩、在线微调以及仅视觉控制器进行对比；在held-out机器人上，Kepler-Encoder的视觉特征在力/末端执行器状态上平均提升0.10–0.17的R²，且在机器人状态几何结构上表现最好。

**⚠️ 局限性**

局限在于仅处理单时刻输入，无法捕获速度、加速度等时间导数；跨机器人零样本迁移不佳；视觉+状态的读出解码器不能直接共享，需针对不同传感器配置重新训练。

---

## 250. Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents

**arXiv ID:** 2607.13591 | [PDF](https://arxiv.org/pdf/2607.13591v1)

**作者:** Eric Hanchen Jiang `[一作]` (University of California Los Angeles), Ying Nian Wu `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Memory-as-a-Controlled-Process（MemCon），将 LLM 代理的外部记忆操作建模为马尔可夫决策过程（MDP），通过轻量级 UCB 上下文赌博机在线学习一个控制策略，以自适应决定何时检索、注入计划、重新检索、合并或忘记记忆。

**💡 创新点**

核心创新在于：①把记忆访问视为可学习的决策过程；②使用无预训练、无额外 LLM 调用的表格 UCB 策略；③提供后台无关的包装器，兼容任意现有记忆存储；④加入两种通用增强操作（计划注入与目标拆分），进一步提升长周期任务的效果。

**🔧 技术方法**

技术实现包括：
- 记忆 MDP（状态由任务进度与记忆状态离散化而成，动作为六类记忆操作+参数），
- 上下文 UCB 策略（表格 Q 估计 + 逆折扣回报），
- 热启动先验、逆折扣信用分配、离线持久化。
- 轻量化的后端无关包装器，支持向量检索、技能库、图结构、摘要与潜在标记记忆等多种后端。

**📊 数据集**

在六个基准上进行评估：交互式决策（ALFWorld、PDDL Planning、ScienceWorld）和问答/工具使用（TriviaQA、WebWalkerQA、GAIA），覆盖三种代理框架（Lobster、LangGraph、Microsoft Agent-Framework）和三种 LLM 主干（GPT‑4.1‑mini、Claude Sonnet‑4、DeepSeek‑V3.2）。

**📈 对比分析**

与九种基线记忆（如 G‑Memory、MetaGPT、Voyager、ChatDev 等）对比，MemCon 在 54 个框架×基准组合中取得了 15/18 的单元最佳或次优成绩，任务成功率提升平均 5–30+ 分，且平均每任务输入 token 下降 5–20%。在更强 LLM（Sonnet‑4、DeepSeek‑V3.2）上表现更为突出，几乎在所有交互式任务上都夺得第一。

**⚠️ 局限性**

局限性包括：
- 仅在单一任务流上在线学习，跨任务迁移能力有限；
- 对高度结构化的复合目标的提升主要依赖两种增强操作，普通任务中收益有限；
- 虽然不需额外 LLM 调用，但仍需手工设计状态离散化、动作集合及奖励设定；
- 在极大规模记忆或极长任务序列下的可扩展性与收敛速度尚待进一步验证。

---

## 251. AI advice suppresses people's willingness to say "I don't know", even when the advice is wrong and accuracy is incentivized

**arXiv ID:** 2607.13562 | [PDF](https://arxiv.org/pdf/2607.13562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 252. Structured Reinforcement Learning for Bayesian Persuasion : Application to Intelligent Interactive Driving

**arXiv ID:** 2607.13576 | [PDF](https://arxiv.org/pdf/2607.13576v1)

**作者:** Merlin Paul `[一作]`, Anup Aprem `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究在交互式驾驶场景中引入贝叶斯说服框架，利用结构化强化学习设计可说服的信号策略，以协调领头车辆与连通车辆之间的远期决策与信息不对称问题。

**💡 创新点**

创新点在于提出了两套结构化学习算法——MAPL（单调代理策略学习）与SQP（超模Q学习），通过单调性与超模性分析，使得在未知奖励与转移动态下的信号设计与代理学习实现显著的计算效率提升。

**🔧 技术方法**

采用的技术包括贝叶斯说服理论、结构化强化学习、超模性理论、梯度策略更新、软最大化信号策略以及在线学习框架。

**📊 数据集**

实验使用基于交通流模型的两车道仿真环境（模拟实时交通状况），并未使用公开的真实交通数据集。

**📈 对比分析**

与MPC、SSP、树学习、Q学习、OP4、历史在线说服、均匀信号与完全揭示信号策略进行对比，结果表明结构化学习在连通车辆奖励上提升约30%，同时计算时间显著降低。

**⚠️ 局限性**

研究的局限性在于假设连通车辆行为为单调一致，未考虑风险敏感性、异质性以及模型误差，且在更高维连续状态空间中的性能仍需进一步验证。

---

## 253. GHR-VLM: Making Zero-Shot Transit Video Analytics Realizable with Grounded Hybrid Reasoning

**arXiv ID:** 2607.13569 | [PDF](https://arxiv.org/pdf/2607.13569v1)

**作者:** Kaicong Huang `[一作]` (Rensselaer Polytechnic Institute), Ruimin Ke `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 GHR‑VLM 框架，实现了零样本公交车视频中的乘客登车与支付行为分析。

**💡 创新点**

创新点在于将轻量级边缘模块的门控检测与跟踪、规则化乘客切片、方向映射以及分阶段视觉语言模型的提示相结合，实现了视觉定位与语言推理的混合推理。

**🔧 技术方法**

使用了 SAM3 与 YOLO11 等目标检测/分割模型进行门与乘客定位，GPT‑4o/GPT‑5.4‑mini 视觉语言模型进行方向判断与支付分类，并采用边缘‑云协同架构。

**📊 数据集**

使用了 486 分钟的真实公交车监控视频（C3_1 与 C3_3）进行实验。

**📈 对比分析**

与纯模型基线和纯 VLM 基线对比，停靠点与乘客片段的 F1 分别从 0.767/0.647 提升至 0.887/0.702；支付分类五类准确率在 C3_3 视频上从 0.485 提升至 0.536（相对 Stage‑1 的 0.485）。

**⚠️ 局限性**

局限在于对高质量视频的依赖，光照、模糊、遮挡等会导致定位与 VLM 推理失误，支付准确率仍未达到实用部署要求。

---

## 254. Nexus: Native Mesh Generation with Diffusion

**arXiv ID:** 2607.13563 | [PDF](https://arxiv.org/pdf/2607.13563v1)

**作者:** Hanxiao Wang `[一作]` (Chinese Academy of Sciences), Yan-Pei Cao `[通讯]` (VAST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无序排列、端到端的三角网格生成框架，先用分层八叉树扩散模型生成顶点，再通过时空间隔指标在顶点上进行拓扑扩散，从而实现完整网格的高质量生成。

**💡 创新点**

创新点包括：① 八叉树扩散实现粗到细的全局顶点生成；② 时空间隔指标把任意拓扑编码为连续嵌入，拓扑可由扩散模型直接生成；③ 完全摆脱序列化，解决自回归模型的排序敏感和误差累积问题。

**🔧 技术方法**

采用扩散变压器（Diffusion Transformer）、稀疏八叉树表示、Minkowski 时空间隔距离、拓扑自编码器、流匹配训练、3D RoPE、跨模态编码器（VecSet、DINOv3）等技术。

**📊 数据集**

训练使用 Objaverse 与 ObjaverseXL 约 100 万条面数 <20k 的网格；评估在 Objaverse 500 模型、Toys4K 900 条低面数网格，以及 in-the-wild 图像和点云条件生成实验。

**📈 对比分析**

在点云到网格、图像到网格任务中，与 MeshAnything、BPT、TreeMeshGPT、FastMesh 等主流方法相比，本文在 Hausdorff、Chamfer、Edge Chamfer、Normal Consistency 等指标均取得最优或同级领先；用户研究 Elo 评分 1440、偏好率 93%。

**⚠️ 局限性**

主要限制为顶点生成需多步扩散，单张 512 分辨率约耗时 1 分钟；拓扑阶段未考虑面法向，需要额外方向校正；在极高分辨率或纹理模糊情况下仍可能出现缺失面或网格瑕疵。

---

## 255. An Empirical Study on Stage-Information Interfaces for VLA Fine-Tuning

**arXiv ID:** 2607.13605 | [PDF](https://arxiv.org/pdf/2607.13605v1)

**作者:** Yingwei Ji `[一作]` `[通讯]`, Yingwei Ji

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了将主动子任务信息注入视觉‑语言‑动作（VLA）策略的接口方式，并比较了两种接口（文本提示和状态值）在不同训练安排下的表现。

**💡 创新点**

创新点在于把子任务信息注入视为接口问题，系统地比较了文本和状态两种注入方式，并探讨了直接微调与先全任务微调再继续微调两种训练策略的影响。

**🔧 技术方法**

使用了 NVIDIA 的 GR00T N1.6 视觉‑语言模型、Diffusion Transformer、进度预测模块、冻结层策略以及分阶段注释的分段动作生成技术。

**📊 数据集**

实验基于 LIBERO‑10 基准数据集，包含10个操作任务，每个任务使用30条演示数据。

**📈 对比分析**

通过在同一进度模型下，分别在直接微调和先全任务微调后继续微调的两种安排中训练三种接口，并在每种安排下跑三次随机种子，比较成功率；结果显示在继续微调后，状态接口在所有对比中均优于文本接口及无子任务信息的基线，成功率约为49%–54%，但在直接微调中均未超越基线。

**⚠️ 局限性**

局限性包括仅使用单一模型和单一基准、冻结大部分层、训练步数有限、仅评估三种随机种子，以及未检验该方法在更大规模或真实机器人上的可迁移性。

---

## 256. MARS: Multi-stage Accelerated Read Stack for Large-buffer Buffered Reads

**arXiv ID:** 2607.13604 | [PDF](https://arxiv.org/pdf/2607.13604v1)

**作者:** Yang Shen `[一作]` (National University of Defense Technology), Wenzhe Zhang `[通讯]` (National University of Defense Technology)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 Linux 同步缓冲读取进行重新组织，提出多阶段加速读取栈（MARS），使大缓冲读能更好地利用内核工作窗口、并行度和设备并行能力。

**💡 创新点**

创新点包括：1）将大范围读取拆分为分类、分配、插入、提交、等待、复制六个阶段；2）利用等待窗口对用户页故障处理和大部分复制进行隐藏；3）使用轻量级 per-CPU worker 进行并发复制和可选并行 I/O 提交。

**🔧 技术方法**

技术手段包括：XArray 扫描与批量分配、批量插入、集中 I/O 提交、等待窗口掩蔽、Out‑of‑Order 早期复制、基于原子操作的 per-CPU 工作调度框架、并行 I/O 提交框架。

**📊 数据集**

使用的数据集与工作负载：FIO 随机/顺序读取；DuckDB/Parquet（LibriSpeech、Earnings‑21、RSHR‑Bench）；ExecuTorch PTE 模型加载（Llama‑1B、Qwen‑1.7B、Llama‑3B）；TensorFlow TFRecord（8 MiB 记录的 112 条记录）。

**📈 对比分析**

比较方法：与同版本 Linux 内核 6.6.58 基线对比，使用 eBPF 记录各阶段耗时；在单块 NVMe 和 5‑块 RAID0 上测量吞吐；在真实应用中测量完整工作流耗时。性能提升：FIO 128 MiB 随机读 6.56×，单块 NVMe 128 MiB 4.44×；应用层速度提升 1.8–3.6×。

**⚠️ 局限性**

局限性：需内核改动，难以在设备并行度低或小读场景中获得显著收益；实现复杂，可能引入新的调度开销；对异步 I/O 或 Direct I/O 场景兼容性有限；对大缓存读的优化在内存压力高时可能受到限制。

---

## 257. On phase-field regularization in dynamic fracture with brittle and cohesive formulations

**arXiv ID:** 2607.13599 | [PDF](https://arxiv.org/pdf/2607.13599v1)

**作者:** Jonas Heinzmann `[一作]` (ETH Zürich), Laura De Lorenzis `[通讯]` (ETH Zürich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对动态相位场断裂模型进行了系统分析与比较，提出并验证了在动态裂尖传播和波-裂交互中的三种相位场建模：仅弹性模量退化的脆性模型、同时退化弹性模量和密度的脆性模型，以及新的强度退化共聚模型（保持弹性模量不变），并将其扩展到动力学。

**💡 创新点**

创新点在于：①揭示了脆性模型中波速与声阻抗的梯度导致高频振荡和裂纹宽化的根本机制；②提出并验证了在动态下通过强度退化的共聚相位场能恢复“尖锐”裂纹的波传播特性；③推导了强度退化模型在1D中对波形的动态共聚开口律及其对波速/Irwin长度比的依赖；④将该模型推广至多维情况并与传统脆性模型在裂尖分支实验中进行对比。

**🔧 技术方法**

使用技术包括：相位场能量泛函与哈密顿原理的变分推导、波速/声阻抗分析、传输矩阵法、数值有限元(Newmark-α、generalized‑α)、多尺度模拟与后处理。

**📊 数据集**

未使用公开数据集，全部使用自定义单维杆与二维预缺口板的数值实验，材料参数取自典型工程钢材。

**📈 对比分析**

通过1D波-裂交互的解析与数值对比，揭示了ℓ/λ与σ̃/σ_c的临界关系；在2D预缺口板中比较三种模型的裂尖分支与能量/质量变化，结果显示强度退化模型与脆性退化模型在裂尖分支与能量消耗上相似，但仅强度退化模型能够正确传播压缩波并避免无物理意义的宽化。

**⚠️ 局限性**

局限性包括：脆性退化模型在动态中需极小化ℓ/λ才能恢复尖锐裂纹，且易导致能量超标；强度退化模型对参数ℓ与σ_c/σ̃ 的敏感性较高，需精细调试；两种脆性模型均不满足质量守恒或波速一致性，限制了其在实际高频/高应变率问题中的适用性。

---

## 258. Multivariate Cryptography-Based Anonymous Certificate Scheme

**arXiv ID:** 2607.13554 | [PDF](https://arxiv.org/pdf/2607.13554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 259. Agile perceptive multi-skill locomotion for quadrupedal robots in the wild

**arXiv ID:** 2607.13579 | [PDF](https://arxiv.org/pdf/2607.13579v1)

**作者:** Jun-Gill Kang `[一作]` (Agency for Defense Development), Hae-Won Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套名为APT‑RL的端到端控制框架，能够让四足机器人在多种复杂地形上高速、感知驱动地实现多姿态多技术（trot、bound等）动态切换；

**💡 创新点**

创新点在于将2维轨迹优化生成的状态‑力矩对作为大规模、结构化运动先验，通过Transformer‑VAE学习统一的潜在空间并预训练力矩解码器，然后在RL阶段同时学习潜在动作与辅助动作，使得机器人在不需额外在线轨迹优化或专家策略的情况下实现多技能平滑过渡；并通过教师‑学生蒸馏结合深度相机与长距离LiDAR，实现零样本真实场景部署；

**🔧 技术方法**

核心技术包括：1) 2D单刚体模型轨迹优化（Bézier曲线、动量守恒）生成多速度、多步态运动；2) Transformer‑VAE自监督表示学习和动作解码；3) PPO强化学习与辅助动作联合训练；4) 运动优先级的多模态感知蒸馏；5) 通过潜在动作与PD辅助动作混合的双层控制；

**📊 数据集**

使用自生成的180,000条轨迹（90k trot+90k bound，约15.5小时）作为训练数据，包含状态、动作和力矩；感知蒸馏使用的是真实的深度相机+D435和2D LiDAR的同步采样；

**📈 对比分析**

与AMP、传统RL、HRL+残差策略等基线在平坦、斜坡、障碍物、台阶、隙缝等多种地形上进行对比，APT‑RL在成功率、速度追踪、能耗（1/COT）上均优于基线；在真实KAIST HOUND机器人上实现4.25 m/s（高台阶）及6 m/s（三阶台阶）瞬时速度，创下野外感知四足机动的新标杆；

**⚠️ 局限性**

局限包括：仅在纵向（踱步/跳跃）方向研究；仅实现trot与bound两种步态，未覆盖更丰富的步态；依赖2D轨迹优化，可能在全三维动作上产生欠缺；感知依赖深度相机与LiDAR，受光照、雨雪等环境干扰；未涉及长周期导航与规划，需要进一步集成高层决策与语义理解。

---

## 260. Approximation of solutions of parameter-dependent problems by residual neural networks

**arXiv ID:** 2607.13574 | [PDF](https://arxiv.org/pdf/2607.13574v1)

**作者:** Ana Carpio `[一作]` `[通讯]` (Universidad Complutense de Madrid), Ana Carpio (Universidad Complutense de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种利用梯度流训练残差神经网络的策略，并通过Lojasiewicz理论保证收敛；

**💡 创新点**

创新点在于将梯度流与解析激活函数结合，省去传统优化的收敛假设，并对逆问题展示了显著的正则化效果；

**🔧 技术方法**

采用残差网络（ResNet）架构、sigmoid/softplus等解析激活函数，利用梯度流求解网络权重，使用MATLAB的ODE求解器实现；

**📊 数据集**

数据集为基于解析解生成的合成数据：一阶/二阶常微分方程解的网格样本，以及三维参数（h,w,c）对应的波动方程解样本；

**📈 对比分析**

通过均方误差与参数预测误差比较，实验显示即使仅使用少量节点也能得到较小误差；使用拉丁超立方采样可在保持误差的同时将训练样本量减半，训练时间约三分之二；

**⚠️ 局限性**

局限性包括：当参数变化导致波形模式差异显著（如小c值）时预测误差增大；方法依赖激活函数解析性，且对极端参数取值的稳健性有限。

---

## 261. Graded Entity-Familiarity Readouts in Language Models: Polish Adaptation, Cross-Language Robustness, and Refusal Steering

**arXiv ID:** 2607.13568 | [PDF](https://arxiv.org/pdf/2607.13568v1)

**作者:** Grzegorz Brzezinka `[一作]` `[通讯]` (Prosit AS), Grzegorz Brzezinka (Prosit AS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在回答前对实体熟悉度的预测，并在 1,440 条波兰实体上评估多种模型的熟悉度读数。

**💡 创新点**

首次提出在 prompt‑point 通过单前向激活实现可量化的实体熟悉度读数，并证明其分级、跨语言稳健、可因果操控及可用于预生成拒绝门控。

**🔧 技术方法**

使用监督线性探测器、离散散度指标、对数页面浏览量相关性、单向干预激活方向以及基准对比的风险‑覆盖分析。

**📊 数据集**

构建了 1,440 条波兰实体（4 个领域、10 个流量分位、60 条合成实体）及对应的 v1/v2 版本，并利用公共 Wiki 页面浏览量。

**📈 对比分析**

与多种已发表的后生成检测器（D2HScore、EigenTrack、MIND）及词频/散度基线在 AUROC/Risk‑Coverage 上对比，单前向探测在真实/虚构辨别上达到 0.86–0.93 AUROC，且在预生成门控中优于多数基线。

**⚠️ 局限性**

仅在波兰/英语模板对换、单一模型/领域的因果测试，缺乏人类评审、模型拒绝标记偏差、对不同语言和实体类型的泛化尚未验证，且基线实现可能偏好。

---

## 262. How Far Can Root Cause Analysis Go on Real-World Telemetry Data?

**arXiv ID:** 2607.13548 | [PDF](https://arxiv.org/pdf/2607.13548v1)

**作者:** Athira Gopal `[一作]` (QPIAI India), Ashwanth Krishnan `[通讯]` (QPIAI India)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了结构化多智能体根因分析（Structured Multi-Agent RCA）流水线，并通过逆推理代理和自动规则挖掘进一步诊断错误，形成了一套完整的根因分析系统；

**💡 创新点**

创新点在于：①在固定阈值的离线异常预处理后，让多智能体按枚举‑聚类‑分析‑选择‑反思的五步流程一次性推理；②引入逆推理代理可根据已知正确答案回溯所需证据，划分“推理缺口”和“数据歧义”；③用逆推理报告自动抽取判别规则，降低人工领域知识构建门槛；

**🔧 技术方法**

技术包括：离线阈值预计算（P95/P5/Boolean/ROC）、多模态异常提取、基于 GPT‑5.2 的多智能体推理、领域知识注入（推理规则与组件目录）、逆推理与规则挖掘的 LLM+聚类流水线；

**📊 数据集**

使用 OpenRCA 基准数据集（Market、Telecom、Bank 三个域，Market 包含 CB1、CB2 两个子域），并与 OpenRCA Agent、GALA、RCLAgent 以及经典因果发现方法（Granger、PC、FCI、LiNGAM、NTLR）对比；

**📈 对比分析**

对比方法：OpenRCA Agent、GALA、RCLAgent、非 LLM 因果发现；评价指标为 OpenRCA 的 Full/Partial 分数和 Accuracy@k；结果显示：结构化多智能体在 DK ON 模式下 Full 分数最高（最高 56.86，最低 25.71），显著优于基线和所有对手；非 LLM 方法全部 0；GALA 仅 2.56，RCLAgent 0；自动挖掘的规则在 CB2 上甚至超过手工规则；

**⚠️ 局限性**

限制主要在：①离线阈值预处理的错误会误导后续异常集，②系统多轮 LLM 调用导致成本和延迟较高，③缺乏对不同 LLM 版本或更强推理模型的泛化评估。

---

## 263. COLMAR: Cooperative View Policy Learning for Multi-Agent Active 3D Reconstruction

**arXiv ID:** 2607.13524 | [PDF](https://arxiv.org/pdf/2607.13524v1)

**作者:** Phu Pham `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出多机协同主动3D重建视角规划框架COLMAR；

**💡 创新点**

通过共享地图感知的协同策略和基于增量TSDF的重建感知奖励，鼓励非冗余、覆盖均匀且安全的视角选择；

**🔧 技术方法**

使用共享参数PPO强化学习、TSDF前端、3D Gaussian Splatting后处理、CNN+GRU+Transformer编码等技术；

**📊 数据集**

以GLEAM作为训练数据，Replica用于零样本泛化评估；

**📈 对比分析**

与随机、贪心、前沿搜索以及单体PPO基线对比，COLMAR在Replica上实现89.4%准确率、4.57 cm Chamfer、PSNR 37.61，较基线提升约54%精度、49%覆盖率；

**⚠️ 局限性**

仅在静态室内场景验证，对动态变化、传感器噪声、定位误差、团队规模饱和以及真实机器人部署和通信约束等方面仍有限。

---

## 264. Clustering algorithms for multivariate wind farm SCADA data filtering

**arXiv ID:** 2607.13544 | [PDF](https://arxiv.org/pdf/2607.13544v1)

**作者:** Nicolò Italiano `[一作]` (Technical University of Denmark), Nicolaos A. Cutululis `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用聚类方法对风电场SCADA数据进行异常检测与过滤，并与人工视觉过滤进行对比。

**💡 创新点**

创新点包括提出针对无标签数据的多曲线评价指标，并首次在离岸风机上评估GMM、DBSCAN与HDBSCAN的过滤效果。

**🔧 技术方法**

主要技术手段为统计学与机器学习聚类（GMM、DBSCAN、HDBSCAN）、特征标准化、超参数网格搜索以及FLASC工具。

**📊 数据集**

使用的数据集为Lillgrund离岸风电场48台WT的10分钟平均SCADA记录，本文挑选了3台包含多实验和异常的风机进行实验。

**📈 对比分析**

通过消除率、平均距离降低和保留率三项指标进行比较，结果显示聚类方法相较于手动过滤在异常检测精度上更优，GMM保留率最高，HDBSCAN在精度上表现最佳但保留率最低。

**⚠️ 局限性**

局限性在于仍需人工挑选正常操作的聚类，特征选择与参数调优需针对每台风机单独完成，导致自动化程度有限，且对其他风机或不同数据集的泛化能力未得到充分验证。

---

## 265. Ripple: An Open, AI-Formalized Lean 4 Framework for Computing with CRNs

**arXiv ID:** 2607.13531 | [PDF](https://arxiv.org/pdf/2607.13531v1)

**作者:** Ho-Lin Chen `[一作]` (National Taiwan University), Xiang Huang `[通讯]` (University of Illinois Springfield)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个开源、AI辅助的 Lean 4 框架 Ripple，完成了化学反应网络（CRN）可计算实数、GPAC/CRN 连续模型、LPP 编译流水线、CTMC/Kurtz 理论及两种 Turing 完备性证明的形式化与验证；

**💡 创新点**

其创新点在于：①首次把 CRN 可计算实数的完整理论（从 GPAC 到 LPP，再到 CTMC 及 Kurtz 的三版定理）完整机械化并验证；②通过形式化发现并修正了近似多数协议、LPP 主要定理和 Catalan 依赖性等原论文中的缺口；③以 ζ(3) 的全机械化构造证明其可计算性，并将 Ramanujan 1/π 系列的可计算性转化为开放问题；

**🔧 技术方法**

使用的技术主要是 Lean 4 以及 Mathlib 的基础库，配合公开的大语言模型（Claude Opus、Fable、GPT‑5.x）进行自动化编码、证明推导与调试；

**📊 数据集**

本工作没有使用传统意义的数据集；所有证明均基于理论推导和形式化验证；

**📈 对比分析**

对比方法主要是与传统非形式化证明对照；验证效果以 768 k 行 Lean 代码、920+ 文件、零未解决的 `sorry`、以及对关键命题的三元公理依赖检查为指标，证明完全可靠；

**⚠️ 局限性**

局限性包括：部分结果仍依赖未证明的假设（如 Ramanujan 系列的 CM 归一化、ζ(3) 中的中性模态）；未覆盖所有可能的 CRN 计算范式；且自动化流程仍需人工审核以防逃逸或空洞证明。

---

## 266. Intuitionistic Dynamic Logic

**arXiv ID:** 2607.13528 | [PDF](https://arxiv.org/pdf/2607.13528v1)

**作者:** Lukas Zenger `[一作]` `[通讯]`, Lukas Zenger

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了某一领域的研究进展和未来方向。

**💡 创新点**

创新点在于提出了一种新的理论框架来理解该领域的复杂性。

**🔧 技术方法**

使用了定量分析和定性研究相结合的方法。

**📊 数据集**

数据集来源于多个公开数据库，涵盖了广泛的样本。

**📈 对比分析**

与现有方法进行了对比，结果显示新方法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于数据集的规模和多样性可能影响结果的普适性。

---

## 267. Hardness of Vertex Splitting: Cographs, Chordal Graphs, and Beyond

**arXiv ID:** 2607.13517 | [PDF](https://arxiv.org/pdf/2607.13517v1)

**作者:** Satyabrata Jana `[一作]` (Indian Institute of Science Education and Research Berhampur), R. B. Sandeep `[通讯]` (Indian Institute of Technology Dharwad)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究图形分割（Vertex Splitting）问题的复杂度，证明其在多种图类（如共图、P_t‑free、弦图、单区间图等）上的NP‑完备性，并给出排除独占和浅层变体的严格时间下界；

**💡 创新点**

首次完成对共图、P_t‑free（t≥4）、弦图、单区间图等目标类的Vertex‑Splitting完整分类，并通过高girth图和结构化分割证明其在ETH假设下的指数时间不可压缩性；

**🔧 技术方法**

主要技术包括构造多步归约（从Vertex Cover、Edge Dominating Set、Chain Vertex Deletion等经典问题）、星形/双星形边划分与Vertex‑Splitting的等价性、利用高girth保持结构不变、极限分析（ETH）等；

**📊 数据集**

本工作不涉及实验数据集，全部基于理论证明与归约构造；

**📈 对比分析**

比较方法为多类问题的NP‑完备性和时间下界证明，性能方面提供了2^o(k)·n^O(1)与2^o(n)的下界，表明在常数因子外不存在多项式时间或子指数算法；

**⚠️ 局限性**

局限在于仅给出复杂度上界，未提供有效算法或近似方案；研究范围局限于单纯的Vertex‑Splitting操作，对更一般的图变形或结合其他图操作的情况尚未探讨。

---

## 268. Analogical Deep Research: Retrieving and Integrating Historical Analogies for Foresight Analysis

**arXiv ID:** 2607.13602 | [PDF](https://arxiv.org/pdf/2607.13602v1)

**作者:** Yongqiang Chen `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Kun Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于历史类比的前瞻性分析框架（Analogical Deep Research），并在此框架下定义了新任务 Analogical Foresight（AF），用以检索并整合历史事件中的结构化因果机制以推断目标事件的未来演进。

**💡 创新点**

创新点包括：
- 引入结构化分解（Trigger–Enabler–Amplifier–Mediator–Outcome）对事件进行因果图建模，从而把类比检索从表面描述转向机制层面；
- 通过“结构反射生成”实现对未覆盖结构位置的交叉确认，提出每个隐藏位置至少需由两条独立类比支持的原则；
- 构建首个包含15个事件（10历史、5前瞻）的公开基准，细致标注机制、预判截断、对照类比与隐藏因子。

**🔧 技术方法**

使用技术：
- 大语言模型（LLM）作为核心推理引擎，结合“深度研究”多代理架构；
- 结构化分解与反射循环（prompt‑engineering + self‑critique）实现类比检索与确认；
- 评价采用9维细粒度指标与5级命题打分，测量表面事实、单类比洞察、跨类比模式、隐藏因子推断等。

**📊 数据集**

数据集：
- Analogical Foresight Benchmark（AF‑B）：15个事件（金融、地缘政治、技术），每个事件配备截断时间、预判摘要、因果结构、已知类比、跨类比不变量及隐藏因子。

**📈 对比分析**

对比方法：
- 三大商业代理（ChatGPT, Gemini, Claude）与三种开源LLM（GPT‑4, Claude‑2, LLaMA‑70B）无额外模块；
- 三种基线（直接生成、总结‑再生成、self‑reflection）与新框架（结构分解+反射）比较。
- 结果显示，基于结构分解的框架相较于所有对照方法提升≈10%（细粒度指标）且在隐藏因子检索上达到42/42成功；相比商业代理提升多达60%在深度推断和机制定位方面。

**⚠️ 局限性**

局限性：
- 仍依赖高质量的事件描述与因果图标注，构建和扩展基准成本高；
- 结构化分解与类比确认过程对LLM的自我批判能力高度依赖，低级模型性能差异显著；
- 只处理可观测的结构位置，无法完全恢复完全隐藏的因果机制；
- 对跨域类比的泛化能力和可解释性仍待进一步验证。

---

## 269. Protective Capacity Hallucination: When Large Language Models Claim Nonexistent Capabilities

**arXiv ID:** 2607.13596 | [PDF](https://arxiv.org/pdf/2607.13596v1)

**作者:** Eunna Lee `[一作]` (Independent Researcher), Sunjun Hwang `[通讯]` (Yonsei University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并量化了“保护能力错觉”（Protective Capacity Hallucination，PCH），并通过三阶段实验（海量LLM对话、覆盖域与可委托人检验、跨服务域推广）在八种主流LLM上对1.36万条人工对话进行编码，揭示了PCH的出现机制及其抑制条件。

**💡 创新点**

创新点在于：①首次将自我参照型幻觉与部署设计缺口联系起来，提出PCH的概念；②构建了三阶段评估协议，系统区分情境严重度、交互形式和域覆盖度对PCH的影响；③发现PCH在安全对齐覆盖的域内被抑制，提示“能力边界规范”是通用缓解策略。

**🔧 技术方法**

采用定量编码方法（二元PCH判定）、统计检验（Wilcoxon、Wilson区间）以及多模型、多域对比；并通过自然语言生成任务评估模型的保护性回应。

**📊 数据集**

使用人工构造的多域情景语料，共计八个域（水上乐园、伴侣冲突、机舱协助、酒吧、餐厅、游乐场、图书馆）下的单向和对话式输入，生成共计13,600条模型回复。

**📈 对比分析**

对比结果显示：在未覆盖域的多方对话下PCH率几乎达到100%，而在安全覆盖域（如伴侣冲突）和有可委托人时PCH率降至0-5%；同一模型在不同域间的PCH差异显著，说明抑制机制与域覆盖高度相关。

**⚠️ 局限性**

局限性包括：情景单一，难以完全解耦严重度、责任感与交互形式；模型与部署环境混淆（API vs 本地量化）可能影响结果；仅研究了保护角色，未检验其他角色的类似错觉；缺乏因果实验验证PCH机制。

---

## 270. Optimal Non-Binary Single-Track Gray Code

**arXiv ID:** 2607.13588 | [PDF](https://arxiv.org/pdf/2607.13588v1)

**作者:** Tuvi Etzion `[一作]` `[通讯]` (Technion), Tuvi Etzion (Technion)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

无法完成总结

**💡 创新点**

无可用信息

**🔧 技术方法**

无可用信息

**📊 数据集**

无可用信息

**📈 对比分析**

无可用信息

**⚠️ 局限性**

缺乏论文内容

---

## 271. From Prediction to Collaboration: Interactive Symbolic Music Analysis

**arXiv ID:** 2607.13587 | [PDF](https://arxiv.org/pdf/2607.13587v1)

**作者:** Emmanouil Karystinaios `[一作]` (Johannes Kepler University), Gerhard Widmer `[通讯]` (Johannes Kepler University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一框架，既能进行完整乐谱的罗马数字分析，又能支持部分标签的补全与交互式修订。

**💡 创新点**

创新点在于将预训练的MusicBERT嵌入与图神经网络相结合，推出了可编辑约束的掩码预测模型，并搭建了多层级候选查看与编辑界面，实现了从预测到分析的连续工作流。

**🔧 技术方法**

使用的技术包括：预训练序列编码器（MusicBERT / RNBert）、基于图的AnalysisGNN结构、掩码编辑条件化模型、冲突-aware 训练（PCGrad 等）以及后处理的投票与束搜索解码。

**📊 数据集**

主要使用了 Dilemmadata 基准集（融合 AugmentedNet 与 Distant Listening Corpus），并对比了 RNBert、ChordGNN+Post、AnalysisGNN 等已有模型。

**📈 对比分析**

在全局无监督推理上，混合模型在 AugNet 与 DLC 的多项指标（degree、local key、Roman numeral 等）均达到或接近最佳；在掩码补全任务中，模型随已知标签比例提升准确率，最高可达 0.831/0.801；相较于之前方法，改进显著且稳定，但后处理模块在纯全推理下表现平平，说明其优势主要体现在交互式场景。

**⚠️ 局限性**

局限性包括：缺乏真实音乐学家交互实验验证；界面与模型仅在基准数据上测试，可能对其他风格或更大规模乐谱的泛化有限；掩码模型在处理极端不完整标注时仍易受噪声影响。

---

## 272. UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets

**arXiv ID:** 2607.13586 | [PDF](https://arxiv.org/pdf/2607.13586v1)

**作者:** Xian Li `[一作]`, Juncheng Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一的物理属性与关节运动参数估计框架UniPhysGen，能够同时从三维网格和图像输入中推断物体的物理特性和关节运动信息。

**💡 创新点**

创新点在于将全局位置编码与多模态特征融合、采用SO(3)旋转增强及球坐标表示提高方向不变性，以及在一个模型中同时学习物理、关节、材料和质量属性，显著提升了跨任务的泛化性能。

**🔧 技术方法**

主要技术包括基于Transformer的点云令牌化、多模态特征融合、统一全局位置编码、球坐标参数化、SO(3)随机旋转数据增强以及多任务学习框架。

**📊 数据集**

使用了PartNet-Mobility、PartNet等三维网格数据集，并结合Real2Code、URDFormer等公开数据集进行训练与评测，同时引入图像模态进行实验。

**📈 对比分析**

与Real2Code、URDFormer、Articulate-Anything、PARTICULATE、NeRF2Physics等方法进行量化比较，UniPhysGen在关节轴准确率、枢轴误差、范围重叠mIoU等指标上均取得显著提升，例如关节轴准确率89.96%、枢轴误差9.77、mIoU 84.95，优于所有对比方法。

**⚠️ 局限性**

局限性包括模型规模较大（0.6B–1.7B参数），对输入网格质量和纹理信息依赖较高，且在极端旋转或极稀疏点云下性能仍有提升空间。

---

## 273. UTS at ELOQUENT 2026 Voight-Kampff: structural shifts in AI writing bypass state-of-the-art detectors

**arXiv ID:** 2607.13565 | [PDF](https://arxiv.org/pdf/2607.13565v1)

**作者:** Dima Galat `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并评估在对抗性微调后语言模型逃逸攻击的持久性，提出并验证了跨年代注册攻击与现代主义意识流攻击两种新颖的结构性OOD攻击方式。

**💡 创新点**

创新点在于揭示逃逸成功的根本原因是结构性OOD方向而非表面风格模仿，并证明对抗性微调难以闭合这些OOD攻击；同时提出两种易实现且自然度高的攻击方法。

**🔧 技术方法**

技术上结合Claude Opus、GPT‑5.4、Gemini 2.5 Pro、Qwen3.6四大生成模型，与RoBERTa、TF‑IDF、Binoculars、LogPerp等多种检测器，构建Macko系列对抗性微调并进行大规模实验。

**📊 数据集**

使用的数据集包括ELOQUENT 2024/25/26任务文本、PAN'25人类训练数据、Project Gutenberg 1923年前作品以及66个主题的文本集合。

**📈 对比分析**

通过与2025年标准攻击、Macko-LOSO、Macko-pp等对比，跨年代注册攻击在Claude Opus上实现≈0.80的假阳性率，远高于2025攻击的≤0.025；在ELOQUENT 2026真实竞赛中获得前5名。

**⚠️ 局限性**

局限性：仅在单一英语语料、有限主题范围、闭源生成模型环境下实验；未探索非历史注册或其他结构性OOD轴的攻击效果。

---

## 274. Multi-Agent Collaborative Reasoning with Tool-Augmented Evidence for Urban Region Profiling

**arXiv ID:** 2607.13558 | [PDF](https://arxiv.org/pdf/2607.13558v1)

**作者:** Xixuan Hao `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于多代理协同推理与工具辅助证据的城市地区剖析框架——UrbanAgent，用来推断城市区域的碳排放、GDP和人口等社会经济指标。

**💡 创新点**

创新点包括①将每个数据模态抽象为独立代理，采用关联（corroboration）、纠错（rectification）与监督（supervision）三类边构建动态协同图，显式解决模态不一致；②在每个代理内部实现闭环“推理–行动–观察”流程，利用工具调用（Web搜索、图像裁剪、夜灯检索等）主动获取外部知识；③通过GRPO强化学习优化工具使用策略，提升推理可靠性。

**🔧 技术方法**

主要技术包括多模态特征提取、图神经网络进行多代理交互、强化学习（GRPO）训练工具使用、以及自监督的工具使用轨迹收集与标注。

**📊 数据集**

使用的实验数据集为2000个全球城市地区样本（每个1km×1km），覆盖卫星图像、文本描述、POI分布、3D建筑信息四种模态，并预测碳排放、GDP和人口三项指标。

**📈 对比分析**

在与多种基线（开源LLM、封闭源LLM、UrbanCLIP/UrbanVLP、传统多模态融合与多代理方法）对比时，UrbanAgent在R²和Spearman相关系数上平均提升约8.1%，在碳排放、GDP和人口预测上均显著优于所有基线。

**⚠️ 局限性**

局限性包括：①对工具链的依赖较高，工具可用性与质量会直接影响推理效果；②目前仅支持四种模态，难以应对更多多样化数据源；③在更大规模、更多城市的实际部署中仍需评估计算成本与可扩展性。

---

## 275. Equilibrium stability as a driver of cooperation among Q-learners

**arXiv ID:** 2607.13607 | [PDF](https://arxiv.org/pdf/2607.13607v1)

**作者:** Janusz M. Meylahn `[一作]` (University of Twente), Maximilian Schäfer `[通讯]` (Institut Mines-Télécom Business School)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在常量探索率与学习率下，记忆一的 Q‑学习算法在重复囚徒困境中的学习动态，并提出了以时间平均合作比例衡量合作的全新视角；通过理论分析和大规模仿真，推导了一个基于平衡策略相对稳定性的阈值，用以预测哪种参数配置下合作会占主导；

**💡 创新点**

创新点在于：①放弃传统的收敛性评估，转而关注长期占用时间分布；②将合作的主导性归因于不同纳什均衡的稳定性差异，并给出闭式阈值公式；③通过宏观 F1 分数验证该阈值在广泛参数空间内的有效性；

**🔧 技术方法**

技术方法包括：隐马尔可夫模型框架下的策略占用时间分析；期望 Q‑学习动态的解析推导；基于 Q 值差异的稳定性比较；大规模 Monte Carlo 仿真（10 条轨迹×13,680 参数组合）；以及统计评估（宏观 F1、精确率/召回率）。

**📊 数据集**

使用的“数据集”为仿真生成的囚徒困境轨迹，环境参数覆盖 T=1, S=0, R∈[0.525,0.975], P∈[0.025,0.5]，学习率 α∈{0.01,…,0.20}，探索率 ε∈{0.01,…,0.20}，折扣 δ∈{0.55,0.65,0.75,0.85}，共计 13,680 组参数。

**📈 对比分析**

评估方法：将理论阈值视为判定规则，比较其对“合作主导”与“非合作主导”的预测与仿真占用时间结果的对应情况。性能表现：宏观 F1 分数在 0.85–0.95 之间，显著优于随机基线（≤0.5），并在不同参数扰动下保持局部最优。

**⚠️ 局限性**

局限性：①仅在记忆一、常量探索率的设定下推导，可能不适用于更复杂策略或自适应探索；②假设系统能在有限时间内达到平稳分布，且高折扣下收敛速度慢；③阈值仅考虑了 Q 值差异，忽略了其他稳定性因素；④与真实市场需求动态及多智能体非平稳性未直接建模。

---

## 276. Visual Place Recognition Using Rate-Encoded Spiking Neural Networks with Discrete STDP Learning

**arXiv ID:** 2607.13584 | [PDF](https://arxiv.org/pdf/2607.13584v1)

**作者:** Altzi Tsanko `[一作]` (Democritus University of Thrace), Antonios Gasteratos `[通讯]` (Democritus University of Thrace)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于STDP的离散时域Spiking Neural Network实现视觉地点识别，并通过概率分配、状态隔离和速度补偿滑动窗口聚合提升检索精度。

**💡 创新点**

创新点在于将概率分配改写为闭式张量管线、显式状态隔离、以及在离散SNN上实现速度补偿序列聚合，三者组合使100场景Nordland数据集R@100P实现100%。

**🔧 技术方法**

使用PyTorch + snnTorch实现离散张量SNN，利用Poisson率编码、Leaky Integrate-and-Fire、硬Winner-Take-All、STDP权重更新、全局批量正则化以及概率化神经元分配。

**📊 数据集**

主要数据集为Nordland（100景点）和Oxford RobotCar（100景点），并在这两套数据上进行多次独立训练（15个网络）评估。

**📈 对比分析**

与连续时间ODE SNN-VPR、NetVLAD、SAD等传统方法对比，W+Prob分配在Nordland上达到77.9% R@100P，序列聚合k=5后达到100%，显著优于先前工作（47.5%）且在Oxford上提升至27.1%。

**⚠️ 局限性**

局限性包括：在城市数据集上的精度仍低，无法在真实神经形态硬件上验证能耗优势，速度误差容忍度仅限±10%，且对大规模场景的容量和多视角鲁棒性尚未充分验证。

---

## 277. Spectral-Informed Neural Networks Outperform Spectral Methods in High-dimensional PDEs

**arXiv ID:** 2607.13566 | [PDF](https://arxiv.org/pdf/2607.13566v1)

**作者:** Tianchi Yu `[一作]` (AXXX), Ivan Oseledets `[通讯]` (INM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Modified Spectral‑Informed Neural Networks (Modified SINNs)，利用谱系数衰减尺度和基嵌入改进高维 PDE 求解

**💡 创新点**

创新点在于将谐波分析的先验——谱系数衰减和基函数结构——嵌入 SINNs，显著提升稳定性、精度并能预测缺失谱系数

**🔧 技术方法**

技术包括谱方法（Fourier、Chebyshev）、稀疏网格、PINNs、Deep Ritz、以及在神经网络中实现的衰减缩放和基嵌入模块

**📊 数据集**

数据集为一系列合成 PDE（Poisson、热方程、对流方程、Navier‑Stokes、Schrödinger 等），在不同维度（2~100）与各种缺失谱系数比例下进行实验

**📈 对比分析**

与 SGSM、PINNs、DRMs、PirateNets 等方法在相同条件下比较，采用相对 L² 误差衡量；结果显示 Modified SINNs 在中维缺失谱信息时优于 SGSM，在高维下显著低于 PINNs/DRMs，误差可低至 10⁻⁴–10⁻⁵

**⚠️ 局限性**

局限性：仅针对稀疏谱系数的情况，无法处理本质上稠密、无低维结构的高维 PDE；对缺失低频系数的场景表现不佳，需进一步研究如何识别重要频率

---

## 278. Flow-aware Optimal Navigation in Unsteady Flows through Reinforcement Learning

**arXiv ID:** 2607.13553 | [PDF](https://arxiv.org/pdf/2607.13553v1)

**作者:** Andrea Maria Braghin `[一作]` (Politecnico di Milano), Gabriele Cazzulani `[通讯]` (Politecnico di Milano)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在双涡流中使用强化学习实现自主移动体的目标导航与定点驻留，利用多种生物启发的观测方式和记忆机制进行训练。

**💡 创新点**

创新点在于将记忆型观测（速度或涡率历史）与无全局流场信息的局部感知相结合，证明显式全局参数反而会降低性能，并对不同传感器在能耗与定位精度上的权衡给出洞见。

**🔧 技术方法**

采用了Twin Delayed Deep Deterministic Policy Gradient（TD3）算法作为策略优化器，并在自制的双涡Gymnasium环境中实现了连续动作控制。

**📊 数据集**

使用的实验数据为参数化的双涡流方程所产生的合成流场，起始点、目标点与流参数均随机采样。

**📈 对比分析**

与仅使用时间或单一观测的基准相比，速度记忆代理在训练奖励、成功率（约71%）和能耗方面均表现最优，其他代理的成功率约50-60%。

**⚠️ 局限性**

局限性包括未能达到100%成功率，且对高频、强湍流等更复杂流场的泛化能力未作验证。

---

## 279. ThinkBLOX: 3D Indoor Scene Generation with Progressive Reasoning

**arXiv ID:** 2607.13539 | [PDF](https://arxiv.org/pdf/2607.13539v1)

**作者:** Yuan Xiao `[一作]` (City University of Hong Kong), Jing Liao `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 ThinkBLOX，一个基于 VLM 的逐步推理-行动框架，用于生成和交互编辑 3D 室内场景。

**💡 创新点**

创新点包括：① 将 3D 布局视为递进推理过程；② 构建 ThinkBLOX‑Data‑200K 供进阶推理训练；③ 设计 Tier‑Decoupled GDPO，将物理、语义和推理一致性奖励分层优化。

**🔧 技术方法**

使用技术：VLM（Qwen2.5‑VL‑7B）+ 监督微调 + 逐步 Chain-of-Thought 生成 + Tier‑Decoupled GDPO 强化学习 + 多模态视觉输入。

**📊 数据集**

使用数据集：ThinkBLOX‑Data‑200K（224,757 递进放置对，含多视图、CoT 说明、JSON 布局）以及 IL3D 原始场景。

**📈 对比分析**

对比方法：LayoutGPT、Holodeck、I‑Design、MetaSpatial 等一拍即合的 VLM/LLM 方法，在 11 种房间类别的 CF、IB、Pos、Rot、PSA 等指标上进行评测；ThinkBLOX 在大多数指标上领先，PSA 最高，并在人工评测中表现最佳。

**⚠️ 局限性**

局限性：推理速度比一拍即合慢 3–5 倍；奖励设计仍较通用，难以捕捉细微或个性化偏好；未来需提升推理效率并探索基于人类反馈的更精准奖励。

---

## 280. $r$-Minimal Poset Codes

**arXiv ID:** 2607.13520 | [PDF](https://arxiv.org/pdf/2607.13520v1)

**作者:** Yang Xu `[一作]` (Fudan University), Guangyue Han `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文定义了相对于部分序列支持的r-最小码，并给出了其主要性质与构造方法。

**💡 创新点**

创新点在于将传统Hamming最小码推广至任意poset支持，利用切割r-阻断映射与加权poset度量等工具，给出了r-最小码的等价判定与充分必要条件。

**🔧 技术方法**

主要采用的技术包括poset指标、切割阻断集合、加权poset权重、组同构与等距变换以及组合数与矩阵论。

**📊 数据集**

由于研究为理论性质，本文并未使用任何具体数据集。

**📈 对比分析**

在性能上，本文给出了r-最小码长度的Singleton型上界、存在性条件以及对层次poset的精确描述，理论上相较于已有最小码结果更为广泛与紧凑。

**⚠️ 局限性**

局限性在于对一般poset的具体构造与算法实现尚不充分，且部分存在性结论依赖于复杂的计数与组合估计，实际编码设计仍需进一步研究。

---

## 281. When Bots Join the Team: Bot Adoption and the Institutional Fabric of Open-Source Software Projects

**arXiv ID:** 2607.13679 | [PDF](https://arxiv.org/pdf/2607.13679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 282. Volumetric Inverse Rendering via Neural Radiative Transfer

**arXiv ID:** 2607.13695 | [PDF](https://arxiv.org/pdf/2607.13695v1)

**作者:** Ntumba Elie Nsampi `[一作]` (Max-Planck-Institut fur Informatik), Thomas Leimkühler `[通讯]` (Max-Planck-Institut fur Informatik)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在不显式全局光照渲染的情况下进行体积逆渲染的方法，通过神经场同时表示介质光学属性和完整光场，并使用物理约束（RTE残差、边界条件、观测约束及VRE损失）进行联合优化，实现对参与介质的物理一致重建、视角合成和重光照。

**💡 创新点**

创新点：1) 将体积光学属性与光场统一建模为神经场，解耦物理渲染与网络学习；2) 仅通过局部RTE约束和少量非局部VRE监督即可逼近全局光照，避免昂贵的全局渲染；3) 支持从单场景重建扩展到可生成的多场景分布学习。

**🔧 技术方法**

技术：神经场（SIREN + hash grid）表示光学属性与光场；物理信息网络（PINN）约束（RTE残差、边界条件、观测约束、VRE损失）；两阶段优化（先emission–absorption再联合优化）；Adam优化器；PyTorch实现。

**📊 数据集**

数据集：合成的50个场景，使用Disney Clouds模型随机拼接并赋予RGB吸收、散射系数与Henyey–Greenstein相位参数；环境光照来自Polyhaven的10个环境贴图，训练时每个场景随机选择两种环境光照；提供单光照版本和等向性散射版本。

**📈 对比分析**

比较方法：Differential Ratio Tracking（DRT）与TensorIR。性能：在散射、吸收、消光等属性重建上，本文方法MSE低于DRT；在新视角/新光照图像合成上，DRT略优，但本文仍保持较高PSNR/SSIM；在生成评估中，FID分别为42.3（训练光照）和40.8（新光照），表明能学习可重光照的物理分布。

**⚠️ 局限性**

局限性：1) 仅在合成数据上验证，缺乏真实世界评估；2) 依赖已知环境光照，难以处理未知或混合光源；3) 计算成本仍较高，VRE约束占比53%；4) 需要两阶段优化与良好初始化；5) 基准对比仅限于等向性散射或固定光照，未覆盖所有实际场景。

---

## 283. Social Simulations: from Agent-Based Modeling to Digital Twins

**arXiv ID:** 2607.13693 | [PDF](https://arxiv.org/pdf/2607.13693v1)

**作者:** Erica Cau `[一作]` (University of Pisa), Giulio Rossetti `[通讯]` (ISTI-CNR)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了社会仿真从经典基于代理模型（ABM）到基于大型语言模型的生成式代理模型（LLM-ABM）再到社会数字孪生（SDT）的演进，阐述了各自的建模假设、架构组成、优势与局限，并探讨了这些方法支持的论断范围。

**💡 创新点**

创新点在于将LLM与传统ABM结合，提出了生成式代理的自然语言交互框架，以及将数字孪生与LLM-ABM融合以实现高保真、数据驱动的社会系统模拟，系统化对比两类方法在可解释性、可预测性和实证验证上的差异。

**🔧 技术方法**

所用技术包括：传统ABM规则与网络结构设计、自然语言处理与LLM（如Llama、Mistral）生成代理推理、自然语言交互、数字孪生技术（实时数据采集、空间建模、网络层级结构）以及仿真平台NetLogo。

**📊 数据集**

文章并未给出具体实验数据集，强调需利用真实世界的社交网络、行为日志、地理信息等多源数据进行模型校准与验证；示例中提及可使用NetLogo等工具和LLM模型。

**📈 对比分析**

比较方法主要通过与传统规则化ABM对比、引入不同网络拓扑、计算聚类、极化程度等指标，以及在相同情境下对同一系统进行多次仿真来评估模型的可解释性与预测性能；结果表明LLM-ABM能捕获语言驱动的影响，但结果易受提示设计影响，数字孪生在情景模拟上更精准但计算成本显著提升。

**⚠️ 局限性**

局限性包括：LLM的统计偏差与文化偏见、提示设计的敏感性与不确定性、网络结构的简化或缺失、缺乏系统化的实证验证框架、过度拟合导致的泛化能力下降；数字孪生则面临高昂的构建与维护成本、验证困难、对数据质量和校准的高度依赖，以及预测能力受限于模型假设和实时数据可用性。

---

## 284. Proactive URLLC Adaptation for Connected Vehicles Through ML-Based Channel Prediction

**arXiv ID:** 2607.13692 | [PDF](https://arxiv.org/pdf/2607.13692v1)

**作者:** Andrea Giovannini `[一作]` (National Laboratory of Wireless Communications WiLab CNIT), Alessandro Bazzi `[通讯]` (Università Mediterranea di Reggio Calabria)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于机器学习的短时通道预测框架，以实现车辆URLLC服务的主动适配；

**💡 创新点**

利用自定义非对称MAE/MSE损失函数抑制过估计误差，并在城市环境中以Sionna-RT与SUMO生成的真实雷射追踪数据进行验证；

**🔧 技术方法**

采用深度前馈DNN和双层LSTM两种模型，并使用梯度下降训练；

**📊 数据集**

使用从Sionna-RT+SUMO生成的约10万条样本，特征包括过去N个TTI的RSRP、谱效率及其统计量；

**📈 对比分析**

与理想全时通道知识、理想AR通道、过去平均值和过去最小值四种基线对比；实验表明DNN/LSTM在N≥50时，SLA失败率低于0.5%，并接近Ideal per AR的CAD得分，优于传统基线；

**⚠️ 局限性**

仅针对单车、无干扰、单方向链路，且单轮训练，未验证多车、多干扰或不同场景的鲁棒性。

---

## 285. Local Certification of Vertex and Edge Connectivity

**arXiv ID:** 2607.13677 | [PDF](https://arxiv.org/pdf/2607.13677v1)

**作者:** Yi-Jun Chang `[一作]` (National University of Singapore), Meng-Tsung Tsai `[通讯]` (Academia Sinica)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了在平面图中对任意 k（3≤k≤5）进行 k‑vertex‑connectivity 本地认证的 O(log n) 位标签方案，并给出了对应的 Ω(log n) 下界，证明了该上界是紧确的。

**💡 创新点**

创新点在于：① 通过对平面图的径向图（radial graph）利用“非面性且无弦” (non‑facial and G‑chordless) 的最短环来刻画顶点连通度；② 将该问题转化为在低树深度 (low‑treedepth) 颜色下的常数树深度子图中进行动态规划；③ 在此基础上构造可在 O(log n) 位标签下完成的本地验证协议。

**🔧 技术方法**

技术主要包括：径向图理论、面结构的组合重构、低树深度颜色与平面图的边界分解、树深度分解、对循环的区间拆分与状态表设计、以及基于这些动态规划的本地化检查。

**📊 数据集**

本研究为理论算法，未使用具体实验数据集；所有结果均为数学证明与理论分析。

**📈 对比分析**

由于方案已证明达到 Ω(log n) 的下界，故其性能是最优的；与之前的 3‑连通性认证方案相比，进一步支持 4、5‑连通性而保持相同的标签大小。

**⚠️ 局限性**

局限性在于：仅适用于平面图（及其子类如有限种族图）；对于一般图或非平面图，仍需更大标签或不同方法；此外，方案虽理论上高效，但实现细节和常数因子尚未在实践中验证。

---

## 286. WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching

**arXiv ID:** 2607.13674 | [PDF](https://arxiv.org/pdf/2607.13674v1)

**作者:** Zehan Liu `[一作]` (Chang'an University), Xianwu Gong `[通讯]` (Chang'an University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种实时零样本立体匹配框架WAVE-Stereo，结合了显式匹配搜索与特征对齐两种对应关系，利用GWCE统一编码并周期性注入全局上下文PGCP进行迭代优化。

**💡 创新点**

创新点在于①联合使用相关卷积和特征对齐的对应关系编码（GWCE），②周期性低分辨率全局自注意力（PGCP）在迭代过程中注入长距离约束，从而兼顾细节精度与全局一致性。

**🔧 技术方法**

技术包括：ConvGRU迭代优化、GeoWarp Correspondence Encoder (GWCE) 的三分支编码、周期性全局上下文传播(PGCP)的轻量级ViT + DPT融合、轻量级2D聚合模块、Soft‑argmin初始估计与卷积上采样。

**📊 数据集**

使用9个公开合成数据集（SceneFlow、FSD、FallingThings、IRS、SynClearDepth、VirtualKitti2、InStereo2K、UnrealStereo4K、TartanAir）训练，并在Middlebury、ETH3D、KITTI 2012/2015、Booster等真实数据上进行零样本评估。

**📈 对比分析**

与多种基准方法（RAFT‑Stereo、IGEV‑Stereo、WAFT‑Stereo、LightStereo、LiteAnyStereo 等）比较，WAVE‑Stereo在无外部基础模型的前提下，取得了在Middlebury、ETH3D、KITTI 2015、Booster上的零样本误差低于或与现有最优方法相近，同时在SceneFlow的在域内测试中取得了0.43 EPE，超过了多种实时与非实时非基础模型方法，速度保持在66 ms/帧，实现实时推理。

**⚠️ 局限性**

局限性：①模型主要在合成数据上训练，真实场景中的细粒度纹理与光照仍可能导致误差；②最大视差限制为192像素，无法处理高基线或高分辨率的近距离场景。

---

## 287. The Hyperspherical Geometry of CLIP Latent Space: A Semantic Mixture Model

**arXiv ID:** 2607.13660 | [PDF](https://arxiv.org/pdf/2607.13660v1)

**作者:** Zijie Yu `[一作]` (Tsinghua University), Yue Song `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将CLIP潜在空间建模为von Mises–Fisher混合分布，结合白化校正后进行EM学习，得到可解释的语义成分并提供精确的方向性似然估计；

**💡 创新点**

突破单一高斯假设的局限，首次在CLIP隐空间中提出几何一致的方向性混合模型，既能捕捉多模态语义结构，又能实现可解释的语义分解；

**🔧 技术方法**

使用白化（whitening）对全局协方差进行校正、von Mises–Fisher（vMF）分布混合、Expectation–Maximization（EM）算法估计参数，并通过后验权重实现语义分解；

**📊 数据集**

主要采用MS‑COCO 2017（验证集）作为训练/评估数据，OpenImages子集做OOD评估，实验中还使用BLIP‑2、CoCa等不同backbone进行跨模型验证；

**📈 对比分析**

与W‑CLIP、MCM、EOE、NegLabel等基线比较，FPR95在COCO全量和尾部类别上分别从67.76%提升至48.00%（全量）和从75.05%提升至33.48%（尾部），AUROC也提升；语义相关性最高0.673，推理时间仅9.8 ms（比SPLICE快13×）；在迭代生成漂移实验中，LPIPS下降、CLIP相似度上升，表明语义稳定性更好；

**⚠️ 局限性**

需要在高维下近似Bessel函数，EM收敛速度受成分数K影响；模型仅在CLIP框架下验证，需进一步验证对其他跨模态模型的通用性；

---

## 288. T3HG-Editor: Text-driven 3D Human Garment Editing with Body Priors Embedded in SMPL-X

**arXiv ID:** 2607.13654 | [PDF](https://arxiv.org/pdf/2607.13654v1)

**作者:** Shaoru Sun `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于 SMPL‑X 的文本驱动 3D 人形服装编辑框架 T3HG‑Editor，通过 Gaussian seeding、跨视角注意力与 SDF+2D 掩码裁剪实现高保真且服装一致的编辑。

**💡 创新点**

三大创新点：① 沿 SMPL‑X 法线种子 Gaussians 并结合前后编辑掩码精准定位可编辑 Gaussians；② 以同一 SMPL‑X 顶点为核心的跨视角注意力与特征传播提升多视角服装一致性；③ 使用 SDF 与 2D 掩码双域裁剪有效消除 Gaussians 溢出。

**🔧 技术方法**

使用 3D Gaussian splatting、InstructPix2Pix、Segment Anything Model、CLIP、SMPL‑X 先验、SDF 与 2D 掩码裁剪、跨视角注意力等技术。

**📊 数据集**

在自建的多性别、多种服装类型的 3D 人形场景上实验，使用 10 条文本指令进行评估。

**📈 对比分析**

与 GaussianEditor、DGE、EditSplat 等方法比较，T3HG‑Editor 在 CLIP 相似度、方向相似度、PSNR、SSIM、LPIPS 等指标上均取得最高或最优表现；编辑时间约 8 分钟，略高于 DGE 但与 GaussianEditor 相近。

**⚠️ 局限性**

受限于 InstructPix2Pix 对复杂文本指令的支持不足，且多视角同步编辑导致运行时间增加；未在大规模公开数据集上进行验证。

---

## 289. CIMERA: Compute-in-Interconnect and Memory with Reconfigurable Precision for LLM Inference

**arXiv ID:** 2607.13649 | [PDF](https://arxiv.org/pdf/2607.13649v1)

**作者:** Yue Jiet Chong `[一作]` (National University of Singapore), Xuanyao Fong `[通讯]` (National University of Singapore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为CIMERA的可重构精度LLM推理加速器，集成计算-在互连（compute‑in‑interconnect）与内存（CIM）技术，以缓解内存墙问题并支持不同位宽权重的精度匹配。

**💡 创新点**

创新点包括：①在互连层嵌入可编程计算（如DMAC、归约、激活函数）以减少数据搬移；②利用FeFET非易失性存储单元实现4位多比特权重的并行SMAC计算；③可调SAR ADC与共享ADC架构实现按需精度转换；④采用蛇形数据流与全局多级映射策略最大化数据局部性与并行度；⑤系统整体通过芯片级片上互连网格实现高吞吐量与低功耗。

**🔧 技术方法**

使用技术包括：FeFET非易失性NVM阵列、跨栏CIM、SAR ADC、M3D垂直集成、2D Mesh IPCN、可编程ISA路由器、共享ADC时分复用、蛇形层间映射、精度可调SMAC、以及全流程硬件‑软件协同验证。

**📊 数据集**

数据集/模型：LLaMA‑3.2 (1B)、Mistral‑7B、LLaMA‑2‑13B；使用不同上下文长度（1024、2048）进行推理评估，并在这些模型上执行精度（4/8/16位）比较。

**📈 对比分析**

对比方法：与Nvidia H100同等吞吐量下对比功耗与能效；结果显示CIMERA在1B模型上能效提升至273 tokens/J（vs 11.2 tokens/J），在13B模型上提升至11.5 tokens/J（vs 1.2 tokens/J）；吞吐量在低精度下提升多达25×（1B）与10×（13B）。

**⚠️ 局限性**

局限性：①仅针对Transformer结构的LLM推理，未验证其它模型；②量化与混合精度可能导致精度损失，论文未给出完整准确率评估；③FeFET可靠性、温度漂移和写入/擦除周期的长期影响尚未系统评估；④芯片级片上互连的面积与布线复杂度在大规模部署时可能成为瓶颈；⑤在极端功耗或散热条件下的实际系统集成尚需进一步验证。

---

## 290. Human4K: A Large-Scale 4K Multi-View Mocap Dataset for Whole-Body 3D Human Reconstruction

**arXiv ID:** 2607.13646 | [PDF](https://arxiv.org/pdf/2607.13646v1)

**作者:** Tianshun Han `[一作]` (Macau University of Science and Technology), Jun Wan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个大规模的4K多视角人体动作捕捉数据集Human4K，包含六百万帧、八台摄像机和11名受试者，配合Vicon动作捕捉系统实现了全身、手部和面部的高精度SMPL-X标注；

**💡 创新点**

创新点在于：①全程原生4K分辨率的多视角采集，提升细节捕获；②利用Motion‑Retargeting and Refinement Module (MRRM)将Vicon数据精准映射到SMPL‑X，解决手部结构不匹配；③对半量数据进行虚拟服装增强，显著扩大外观多样性；④在大规模真实场景下实现了高质量、丰富多姿态的标注，填补了现有数据集在分辨率、标注精度与动作多样性方面的空缺；

**🔧 技术方法**

技术手段包括：同步8台4K摄像机与120 fps Vicon捕捉的时间戳对齐；基于2D‑3D关节点匹配的摄像机标定；BVH至SMPL‑X的骨架重映射与骨骼比例校正；手部专属优化（Adam迭代）与面部参数提取（YOLOv8+DECA）；以及使用Google Nano模型实现虚拟服装渲染；

**📊 数据集**

主要使用的数据集为自研Human4K（6M帧），并在基线训练中结合COCO‑WholeBody、MPII、Human3.6M、MPI‑INF‑3DHP；在评测时对比EHF、3DPW以及Human4K自带的训练/验证/测试拆分；

**📈 对比分析**

通过在三种主流SMPL‑X重建方法（Hand4Whole、OSX‑b、SMPLer‑X‑b）上做单一数据集训练、Human4K单独训练和联合训练的三种设置，发现加入Human4K后，所有方法在EHF和3DPW的MPJPE/MPVPE显著下降（大约10%~20%），在Human4K测试集上单独训练已比仅用公共数据提升约30%，而联合训练则达到最佳性能；

**⚠️ 局限性**

局限性包括：①受试者为专业演员，使用标记服装，外观多样性受限；②采集环境为受控室内场景，缺乏真实世界背景与光照变化；③数据规模虽然大，但仍不足以覆盖所有日常姿态与极端场景；

---

## 291. gDMC: A Generic Distributed Model Counting Framework via Work-Stealing

**arXiv ID:** 2607.13634 | [PDF](https://arxiv.org/pdf/2607.13634v1)

**作者:** Zhenghang Xu `[一作]` (Northeast Normal University), Jean-Marie Lagniez `[通讯]` (University of Artois)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个通用、与求解器无关的分布式精确模型计数框架，利用 C++ 模板和概念实现零开销的并行化，支持动态工作窃取。

**💡 创新点**

创新点包括：①基于模板的抽象层将分布式调度与 DPLL 核心解算器解耦；②点对组的工作窃取策略和显式堆栈管理，实现高效负载平衡；③对安全工作共享提出正式条件，保证缓存一致性与计数正确性；④在分布式环境下保留本地缓存与增量假设的优势。

**🔧 技术方法**

使用技术包括：C++ Concepts & Templates、显式堆栈框架、动态工作窃取、MPI 网络通信、增量 SAT 假设、树分解预处理、任意精度算术、Component Caching 与 Clause Learning。

**📊 数据集**

实验数据集来源于最近一次 #SAT 竞赛实例；使用 PACE 2017 提供的树分解（10 秒预算）；预处理器 MiniSAT-preprocess（vivification、backbone、occurrence elimination）；在 32 核 Intel Xeon 计算集群上评测。

**📈 对比分析**

对比方法：与分布式求解器 c2d、Cachet、#SAT 等做并行比较；在 128 核设置下，框架实现 108 个实例成功计数，优于 35（c2d）和 104（Cachet）。在不同核心数（2-128）上显示近线性加速，速度提升明显且稳健。

**⚠️ 局限性**

局限性：需要在共享前验证子问题可满足性，导致额外的 SAT 调用；点对组窃取策略对网络延迟敏感；对分布式缓存一致性的检查仍依赖主节点协调；在极大规模集群（千核）下通信与同步开销可能成为瓶颈。

---

## 292. Design, Modeling and Experimental Validation of a Miniature Hybrid Underwater Glider With Large-Range Foldable Deflectable Wings

**arXiv ID:** 2607.13622 | [PDF](https://arxiv.org/pdf/2607.13622v1)

**作者:** Yongjian Zhu `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了FoDeGlider，一款具备可折叠、可偏转翼的微型混合式水下滑翔机，实现了在受限水环境中大幅提升机动性与形态适应性。

**💡 创新点**

创新点包括：将翼的折叠与偏转实现为独立驱动，构建基于配置的多体动力学模型，并引入Bernstein函数实现翼变形对惯性与流体力的连续尺度化。

**🔧 技术方法**

采用空间算子、组合刚体算法（CRBA）和Fossen形式动力学，以及梯度下降优化的分阶段参数识别技术。

**📊 数据集**

使用实验室水槽中的运动捕捉和内部编码器数据，公开了包含折叠、全伸展及多角度摆动的训练/测试数据集（共216条轨迹）。

**📈 对比分析**

通过与单体刚体模型和各尺度化功能的消融实验比较，显示多体模型的窗口级和轨迹级NMSE分别降低约74%和75%，证明该方法在复杂翼形态下显著提升预测精度。

**⚠️ 局限性**

局限性在于仅验证了对称折叠角度，未探讨非对称或极限折叠状态，以及在更大尺度或更动态环境下的鲁棒性。

---

## 293. UESF-Bench: Benchmarking and Probing for Unified Embodied Seeking and Following

**arXiv ID:** 2607.13621 | [PDF](https://arxiv.org/pdf/2607.13621v1)

**作者:** Kun Yu `[一作]` (Shandong University), Keji He `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了语言引导下的统一寻人‑跟随任务，并构建了大规模基准 UESF-Bench 与对应的视觉‑语言‑动作模型 SeekFollow‑VLA。

**💡 创新点**

创新点包括：①将目标寻找与持续跟随合并为单一任务，处理目标最初不可见、隐式阶段切换及身份延迟；②设计任务驱动路由器，在不依赖显式阶段标记的情况下动态选择寻人或跟随动作头；③构建涵盖单人及多人、多场景、开放环境的 1.43M 样本基准。

**🔧 技术方法**

使用的技术主要有：预训练视觉‑语言模型 Qwen3‑4B 作为多模态 backbone；双塔视觉编码器（DINO‑V3 + SigLIP）与 GridPool 生成多尺度视觉 token；任务驱动路由器与多头动作输出实现阶段感知；waypoint 回归作为最终控制策略。

**📊 数据集**

数据集：UESF‑Bench，包含 1.43M 目标寻‑跟随样本，来自 777 个 HM3D/MP3D 场景，涵盖 4800+ 多样化人像，提供语言描述、专家轨迹和多目标（单人/多人）设置。

**📈 对比分析**

通过与单头、双头阶段感知路由等基线比较，采用 TSR、CR、FR、Search SPL 四项指标评估。任务驱动双头在单人场景 TSR 达 0.35、FR 0.92、SPL 0.53，在多人场景 TSR 0.20、FR 0.82、SPL 0.55，显著优于单头和无监督双头模型。

**⚠️ 局限性**

局限性：尽管取得进展，但碰撞率仍较高；在多人物干扰下性能下降；实验仅在仿真环境中验证，缺乏真实机器人硬件与感知噪声的考量；对全局地图或外部规划的依赖有限。

---

## 294. A Telemetry-Driven Model for Quantifying Upgrade Risk in Durable Workflow Execution

**arXiv ID:** 2607.13617 | [PDF](https://arxiv.org/pdf/2607.13617v1)

**作者:** Luca Maraschi `[一作]` (Platformatic Inc.), Matteo Collina `[通讯]` (Platformatic Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于事件日志的可升级工作流风险评估框架，既能精确判定已记录前缀的兼容性，又能对未来执行路径进行概率预测。

**💡 创新点**

创新点包括：①将恢复风险拆解为可计算的“后向兼容性”与可估计的“前向曝光风险”；②利用完整的执行日志实现后向风险的闭式求解；③构建基于经验转移矩阵的吸收马尔可夫链估计前向风险；④加入贝叶斯不确定性估计与置信区间；⑤考虑跨版本运行间的耦合，使用最小割求最优迁移/固定策略。

**🔧 技术方法**

技术手段主要有：静态差分分析、事件日志解析、并发前缀的Mazurkiewicz轨迹等价、贝叶斯Dirichlet/Beta后验、马尔可夫链求解、无噪声组合(Noisy‑OR)、耦合图的故障连锁最小化（s‑t最小割）。

**📊 数据集**

使用的实验数据集包括：①合成场景（11种变更类型，400个模拟运行）；②随机生成的120个工作流与1,082个变更（64,920次运行级判定）；③真实工作流语料（45个工作流函数，126个步骤，18,160次运行）。

**📈 对比分析**

比较方法：对每种变更类型计算真阳性/真阴性/假阳性/真阴性，验证召回率为1.0，精度分别为0.815（合成）和0.933（真实）。模型运行时间为每个运行微秒级（9–15 µs），大规模批量（5万运行）在0.72 s内完成；分布式前向估计的误差随历史运行数提升而显著降低。

**⚠️ 局限性**

局限性包括：①仅评估行为曝光风险，无法判断更改的意图；②对未观测路径、未记录步骤缺乏信息，导致置信区间宽；③马尔可夫链假设忽略历史状态依赖；④在循环耦合时固定点估计偏保守；⑤对动态访问、反射式消费的推断过度保守；⑥需要足够的历史运行才能给出精确估计。

---

## 295. The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models

**arXiv ID:** 2607.13612 | [PDF](https://arxiv.org/pdf/2607.13612v1)

**作者:** Fabio Arnez `[一作]` (Université Paris-Saclay), Alexandra Gomez-Villa `[通讯]` (Computer Vision Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨并证明在联合嵌入预测架构（JEPA）中，使用非对比式正则化器（尤其是SIGReg）可以让其训练目标精确对应于主动推理（Active Inference, AIF）的变分自由能，构建了从单步到多步规划、集成熵值以及学习策略的完整对应关系，并揭示了当前JEPA缺失的“状态经验值”这一AIF专有项；

**💡 创新点**

提供了一个以正则化器为关键的理论桥梁，证明了SIGReg在满足常数噪声模型和成功实现等方正态化时能够消除先验校准误差，保证自由能界限的完整性，并通过信息瓶颈和自由能框架精确解析了多步规划与学习策略的价值；

**🔧 技术方法**

利用信息瓶颈理论、变分推断、熵估计层次化（VICReg、LogDet、PairDist、SIGReg）、高斯桥接（Gaussian bridge）、Cramér–Wold与Stein方法、以及Lean 4形式化证明；

**📊 数据集**

未使用具体数据集，论文纯理论性，参考已发表的JEPA模型（如LeWorldModel、V-JEPA、DINO-WM、PLDM）和标准控制基准作为对照框架；

**📈 对比分析**

没有实验比较，论文通过理论推导和形式化证明展示了SIGReg在理论上优于VICReg的安全性与精确性，提出可在未来工作中通过实验验证的可测预测；

**⚠️ 局限性**

主要限制在于：仅在常数噪声模型与无限样本假设下严格成立，实际部署中需要满足等方正态化和线性可识别条件；对高维过完备嵌入的可识别性、状态经验值的实现、以及非对比式正则化在真实环境中的效果尚待经验验证。

---

## 296. DNA: Dual-stage Native Attribution for Generated Image Source Tracing

**arXiv ID:** 2607.13685 | [PDF](https://arxiv.org/pdf/2607.13685v1)

**作者:** Chao Wang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段的无监督生成图像源模型鉴别框架DNA，能够从开放式的家族层级识别到具体的变体。

**💡 创新点**

创新点在于将生成模型的VAE层和backbone层的鉴别信号分层使用：VAE层用于粗粒度的开放式家族筛选，backbone层通过原生预测一致性(NPC)实现细粒度的闭集变体鉴别。

**🔧 技术方法**

核心技术包括双重重构(VAE Double‑Reconstruction, AEDR)、原生预测一致性(NPC)、自适应阈值、时刻选择、Z‑score标准化与标量校准；同时采用BLIP‑2进行语义条件化。

**📊 数据集**

构建了DNA‑30K基准，包含24个候选模型（跨六个家族，涵盖去噪扩散和流匹配）共30,000张图像，并加入未知来源（生成与自然图像）。

**📈 对比分析**

与多种主动与被动鉴别方法（OCC‑CLIP、De‑Fake、LatentTracer等）在家族层级和变体层级分别对比，DNA在家族筛选准确率≥92%，变体识别准确率≥96%，端到端准确率89.11%，比最强基线提升约34%。

**⚠️ 局限性**

局限性主要在于家族层级的开放式拒绝对图像压缩敏感（Stage 1受JPEG等重压缩影响显著），以及在未知变体的开放式拒绝（LOO实验显示AUROC仅0.65、拒绝率约20%）。

---

## 297. Explaining Reinforcement Learning Agents via Inductive Logic Programming

**arXiv ID:** 2607.13655 | [PDF](https://arxiv.org/pdf/2607.13655v1)

**作者:** Celeste Veronese `[一作]` (University of Verona), Alessandro Farinelli `[通讯]` (University of Verona)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何使用Inductive Logic Programming（ILP）提取强化学习（RL）代理的符号策略，并基于此提出一套客观可量化的可解释性指标。

**💡 创新点**

创新点在于提出了激活率、特征覆盖、句法距离和语义距离等四种全新的、无用户偏好的评估指标，能够捕捉单机及多智能体RL训练过程中的策略稳定性、特征重要性和代理间的一致性。

**🔧 技术方法**

核心技术包括：答案集编程（ASP）作为符号策略表示形式；ILP（ILASP）从执行轨迹学习规则；以及上述四种可解释性度量，用于定量评估规则与真实行为的一致性。

**📊 数据集**

实验使用了三大RL环境：Intersection（单机交通交叉路口）、RWARE（合作式仓库搬运）以及Simple Adversary（对抗式多智能体），并在每个环境中采集多次训练轨迹。

**📈 对比分析**

通过与传统奖励曲线比较，激活率显示了子策略收敛与不稳定性；特征覆盖揭示了各特征的贡献；句法和语义距离分别评估了同一代理不同训练阶段以及不同代理之间的策略相似度；整体实验表明所提出指标能够提供比单纯奖励更细粒度的洞察，且在转移与泛化任务中保持一定性能。

**⚠️ 局限性**

局限性包括：指标依赖于ILP学习的规则质量，可能受训练样本稀缺或高维状态空间影响；句法距离对命题表述差异敏感，语义距离需要大量对齐实例；实验主要集中在小规模模拟环境，尚未验证在真实工业级大规模多智能体系统中的可扩展性。

---

## 298. Consensus as Privileged Context for Label-Free Self-Distillation

**arXiv ID:** 2607.13643 | [PDF](https://arxiv.org/pdf/2607.13643v1)

**作者:** John Gkountouras `[一作]` (University of Amsterdam), Ivan Titov `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Consensus-ANchored self‑distillation（CAND），利用模型自身的多数投票共识作为无标签的稠密token‑级监督，改进推理准确率

**💡 创新点**

创新点在于将多数投票转化为全词分布的教师指导，而非传统的单一奖励或硬标签，且教师为冻结快照，避免共适配

**🔧 技术方法**

技术核心是：采样N个解、提取多数答案、选取置信度最高的共识解，冻结模型生成的教师分布，并使用全词Jensen‑Shannon散度进行对齐；训练仅一轮，N=32常用

**📊 数据集**

在多数学科推理基准上评估：AMC、AIME 2024/25、GPQA-Diamond、MATH500；同时在不同规模、不同家族（Qwen、SmolLM、LFM、Gemma）模型上验证

**📈 对比分析**

与无标签强化学习（TTRL、EMPO、SCRL等）、无标签微调（LMSI、ScPO）以及使用金标的强化学习和教师蒸馏相比，CAND在转导式测试时提升了12点pass@1，计算成本仅为RL的1/7；在交叉基准迁移时能匹配使用金标的训练；整体性能优于现有无标签方法且接近金标教师的效果

**⚠️ 局限性**

局限性：需要可提取的最终答案（适用于可验证输出）；对已饱和模型-基准对的提升有限；若共识错误且自信，可能强化错误；迭代训练收益有限；转导式假设需预先获得测试样本，适用于批量任务

---

## 299. Unifying Decision-Making and Trajectory-Planning in Unsignalized Intersections Using Time-Varying Potential Fields

**arXiv ID:** 2607.13626 | [PDF](https://arxiv.org/pdf/2607.13626v1)

**作者:** David Costa `[一作]` (Politecnico di Torino), Carlo Novara `[通讯]` (Politecnico di Torino)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套统一的决策与轨迹规划框架，利用基于时间变人工势场（TV‑APF）的有限时域最优控制问题（FHOCP）在无人信号的交叉口实现安全、可行的车辆运动。

**💡 创新点**

创新点在于：① 将交叉口优先级与冲突区占用估计直接融入势场幅值，形成可随时间变化的平滑障碍势场；② 通过演员选择与短期占用系数实现对交叉口相关车辆的高效筛选与预测，无需离散决策切换；③ 使决策与轨迹规划在单一优化问题中统一完成。

**🔧 技术方法**

采用了时间变人工势场（TV‑APF）、有限时域最优控制（FHOCP）、CasADi+Ipopt求解器、MATLAB/Simulink层级控制架构以及自动驾驶工具箱的车辆动力学模型。

**📊 数据集**

使用的是基于MATLAB/Simulink的仿真环境，交叉口交通场景由自动驾驶工具箱生成，未使用公开真实交通数据集。

**📈 对比分析**

通过两组仿真（左转利用交通间隙与拥堵四向交叉口）展示了速度曲线和轨迹，证明了框架能在不同交叉口条件下安全地实现减速、让行、停止和通过；虽然未给出定量指标，但仿真结果表明相较于传统分层规划，能够更平滑地响应交叉口占用。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境中验证，缺乏真实道路测试；② 依赖短期速度预测，若外部车辆行为不符合预测模型可能导致安全风险；③ 计算量受演员筛选与势场参数的影响，极端多车场景下可能仍需进一步优化。

---

## 300. From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception

**arXiv ID:** 2607.13624 | [PDF](https://arxiv.org/pdf/2607.13624v1)

**作者:** Jose Martínez-Fajardo `[一作]`, Luis Merino `[通讯]` (University Pablo de Olavide)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一个基于Vision‑Language Model与RGB‑D感知的语言驱动导航框架，能将自然语言指令转换为可执行的移动机器人导航目标。

**💡 创新点**

创新点在于将VLM的语义理解与RGB‑D几何定位相结合，直接生成物理意义的导航目标，并通过ROS2实现可跨平台部署的轻量级管线。

**🔧 技术方法**

使用了Vision‑Language Models（如CLIP/BLIP‑2）、RGB‑D摄像头、ROS2 Nav2导航堆栈以及TF变换等技术。

**📊 数据集**

实验采用Gazebo仿真中的TurtleBot3、真实Unitree Go2机器人（配备Intel RealSense摄像头）构建的场景；未使用公开数据集，而是自行生成目标环境。

**📈 对比分析**

通过三组实验（仿真、端到端、真实环境）对比定位误差、导航误差、执行时间和行进距离，平均定位误差约0.5–0.7 m，平均导航误差约0.7 m，平均执行时间约26 s，表明系统性能可接受。

**⚠️ 局限性**

局限性包括对目标遮挡与深度噪声敏感、需预设距离补偿偏移、仅支持单目标导航、对多步任务与动态环境的适应性不足。

---

## 301. Extending Liquid Rank Toward Multi-Source Reputation Aggregation

**arXiv ID:** 2607.13615 | [PDF](https://arxiv.org/pdf/2607.13615v1)

**作者:** Nejc Znidar `[一作]` (SingularityNET), Anton Kolonin `[通讯]` (SingularityNET)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Liquid Rank声誉系统进行扩展，使其能够将多源异构声誉信息通过加权融合的方式合成为单一的整体声誉分数。

**💡 创新点**

创新点在于引入可配置的加权融合机制，允许外部声誉信号与内部交易声誉并行参与核心声誉计算；同时提供细粒度控制外部源对整体声誉影响的参数，使治理模型可针对不同社会技术场景定制。

**🔧 技术方法**

使用 Liquid Rank 数学框架（基于持续更新的声誉递推公式）、加权融合公式（R_i = h_1R_i^core + Σ_{k>1}h_kR_i^k）、保守参数（C）实现缓慢收敛，以及正负评分的标准化处理。

**📊 数据集**

论文未提供具体实验数据集，主要通过合成示例和理论推导说明模型行为；在示例中使用假设的声誉值（如0.5、1.0）展示融合效果。

**📈 对比分析**

比较方法：采用理论示例和图示对比核心声誉与融合声誉的变化；未给出量化性能指标（如收敛速度、稳健性评估），仅通过模拟展示模型对外部冲击和持续交易的响应。

**⚠️ 局限性**

局限性包括：需要对融合权重 h_k、保守参数 C 进行人工调节；未进行大规模仿真验证收敛性和对抗性；模型对外部声誉来源的依赖可能导致信息源不一致或被操纵；适用性受限于声誉信号可标准化到 [0,1] 的前提。

---

## 302. CSCO: A Backside-PDN-Aware Clock-Signal Co-Optimization Framework for Improved PPA

**arXiv ID:** 2607.13700 | [PDF](https://arxiv.org/pdf/2607.13700v1)

**作者:** Zixiao Wang `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CSCO 框架，实现了基于背面电源网 (BSPDN) 的双侧时钟/信号路由的联合优化。

**💡 创新点**

创新点在于将 PDN 规划、双侧网分配与信号完整性抑制统一到一套数据驱动的优化流程中，并用低秩因子机模型高效搜索资源分配。

**🔧 技术方法**

采用低阶因子机 (FM) 代理模型、贪婪+探索采样、模拟退火优化及 SI 关键网迁移等技术。

**📊 数据集**

在 SHA256、JPEG 和 ARM Cortex‑A7 三个 7nm 级 benchmark 上进行实验。

**📈 对比分析**

与 FSPDN+FR、DAC24、DAC25 等基线对比，CSCO 在 WNS/TNS 上提升 60–85%，有效频率提高至 1.96，SI 违规率下降 80% 以上。

**⚠️ 局限性**

仍受限于 nTSV 预算估计的准确性、对极低电压环境的鲁棒性未知，以及对非 3D‑IC 设计流程的迁移性待验证。

---

## 303. Microstructure-Conditioned Surrogate Models for Graded Multiscale Optimization of Mycelium Composites

**arXiv ID:** 2607.13688 | [PDF](https://arxiv.org/pdf/2607.13688v1)

**作者:** J. Storm `[一作]` (Delft University of Technology), F. P. van der Meer `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于超网络的物理递归神经网络（HyPRNN），将微观结构变量嵌入到多尺度仿真中的微尺度代理模型，能够在小数据量下实现功能梯度复合材料的仿真与优化。

**💡 创新点**

创新点在于将超网络与PRNN结合，实现在微结构几何参数和制造变量条件下的可调代理模型，并直接在制造变量上条件化，从而避免了隐式潜在空间的额外建模。

**🔧 技术方法**

使用了物理递归神经网络、超网络、JAX 自动微分、DOLFINx 有限元、Neo‑Hookean 超弹性模型、以及基于离散元法的沉降模拟。

**📊 数据集**

数据集由合成的 RVE（随机椭圆堆叠）生成，参数包括椭圆长宽比、体积分数、剪切模量、取向；另外使用 Yade 生成的木屑与菌丝沉降微结构的二维切片数据。

**📈 对比分析**

通过学习曲线、误差曲线和完整多尺度仿真对比，HyPRNN 线性编码器在低数据量下误差低于普通 NN，非线性编码器在大样本时表现更佳；在 3‑点弯曲实验中，代理模型将计算时间从 5900 秒降至 2 秒，且误差保持在 1% 以内。

**⚠️ 局限性**

局限性包括：需要大量人工合成数据，二维简化对真实三维材料的适用性有限；对极大几何非线性与高对比度材料的训练仍不稳定；制造变量与微结构的映射仍基于简化的离散元切片，缺乏实验验证。

---

## 304. Learning Speaker Identity Beyond Language and Modality Constraints: Insights from the POLY-SIM 2026 Challenge

**arXiv ID:** 2607.13669 | [PDF](https://arxiv.org/pdf/2607.13669v1)

**作者:** Marta Moscati `[一作]` (Institute of Computational Perception, Johannes Kepler University), Shah Nawaz `[通讯]` (Institute of Computational Perception, Johannes Kepler University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并举办了POLY‑SIM 2026挑战赛，研究多模态说话人识别在缺失视觉模态和跨语言场景下的鲁棒性与泛化能力。

**💡 创新点**

创新点在于提出多模态融合与正交投影（FOP）基础上的增强模型（MaskedFOP、MRAF、AMR），以及利用测试时未标注数据进行无监督适配的技术。

**🔧 技术方法**

采用双分支深度网络（面部特征提取器+语音编码器）、特征融合模块、正交投影、聚类与自监督对齐等技术。

**📊 数据集**

使用了公开的 MAV‑Celeb 数据集，包含英语与乌尔都语两种语言的音视频样本，且按挑战协议划分训练/测试集。

**📈 对比分析**

在四种评估协议（P3–P6）下，以 P‑accuracy 为指标进行比较，Mask‑FOP 在跨语言缺失模态等最具挑战的设置中平均达 99.89%，显著超越基线 FOP 的 73.37%。

**⚠️ 局限性**

局限性在于顶尖方法依赖测试时的目标语言样本进行无监督迁移，若目标语言缺失或极少样本，则性能会下降，且在真实场景中对少样本语言的泛化仍待提升。

---

## 305. Definitional Inversion, Without Normalisation

**arXiv ID:** 2607.13662 | [PDF](https://arxiv.org/pdf/2607.13662v1)

**作者:** Mario Carneiro `[一作]` (Chalmers University of Technology), Stephanie Weirich `[通讯]` (University of Pennsylvania)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于域理论的新型证明技术，用于在不依赖归一化且可处理η法则的前提下，证明依赖类型系统的定义性逆转、构造器的可注入性与无冲突性等关键元理论性质，并将其应用于从简易型系统扩展到包含依赖和、单元、固定点、自然数、同一性类型和严格命题的完整型系统；

**💡 创新点**

创新点在于将域理论中的可有限投射模型引入元理论证明，摆脱了传统依赖于可归一化和强逻辑原理的逻辑关系与可交叉性方法，能够同时处理η法则与非归一化（如类型自指）的类型系统，并首次在存在η法则的类型系统中证明构造器的可注入性；

**🔧 技术方法**

主要技术包括构建递归域方程的最小解、语义类型与项的投射关系、定义域语义与逻辑关系的融合（含语义匹配与头归约），以及利用域的紧凑元素进行充分性与基本定理证明；

**📊 数据集**

本文没有使用传统意义上的实验数据集，全部结果通过在 Lean、Coq 以及 Rocq 等证明助手中的机械化形式化实现验证；

**📈 对比分析**

与传统的交叉性或逻辑关系方法相比，本文方法不需要强归一化假设或复杂的递归-归约证明，能够在更弱的环境逻辑下完成证明，性能方面以机械化形式化为主，未进行数值性能评估；

**⚠️ 局限性**

主要限制包括：对中性项的定义性逆转仍未覆盖；对更复杂的归纳/共归纳类型、证明无关性与K公理等特性尚未完成；模型在保持语义一致性的同时缺乏反射性，导致对某些细粒度等价判定不敏感，扩展至更大规模系统仍面临挑战。

---

## 306. Fine-grained CLIP fine-tuning with self-annotated region alignment

**arXiv ID:** 2607.13661 | [PDF](https://arxiv.org/pdf/2607.13661v1)

**作者:** Chenyang Zhao `[一作]` (City University of Hong Kong), Janet H. Hsiao `[通讯]` (Hong Kong University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对CLIP模型进行自注释的细粒度微调，仅使用图文对作为训练数据，利用文本特定热图动态生成区域-短语对，实现细粒度特征与全局语义的统一表征。

**💡 创新点**

创新点在于：①不需要预定义类别或手工区域提议，直接从句子中提取短语并用热图定位对应视觉区域；②采用动量CLIP模型在对比学习中维护全局图像-文本匹配能力；③在细粒度对齐与全局对比之间实现权衡，兼顾细粒度与全局性能。

**🔧 技术方法**

主要技术包括：自注释区域-短语对齐方案、Grad-ECLIP热图生成、ViT-based dense feature提取、动量模型（Momentum CLIP）用于对比学习权重、基准对比损失与细粒度匹配损失的联合优化。

**📊 数据集**

训练数据使用MS COCO caption（train2017）图文对；评估数据涵盖ADE20K panoptic、COCO panoptic、Flickr30k（图像-文本检索）、OV-COCO（开放词检测）、ADE20k-847、Pascal VOC、Pascal Context（开放词分割）等。

**📈 对比分析**

与RegionCLIP、CLIPSelf、FineCLIP、DenseVLM、CLIM等细粒度微调方法对比，在零样本区域分类、开口词检测和分割任务中均取得显著提升（如区域分类Top‑1约提升 2–4%，检测novel AP提升 9%+，分割mIoU提升 2–3%），且在Flickr30k图像级检索上保持或略优于继续对比训练的CLIP‑g。

**⚠️ 局限性**

局限性：仍需大量图文对训练；短语提取与热图定位对语义不匹配或图像无对应内容时可能导致误匹配；对极细粒度或稀有概念的捕捉能力有限；热图生成依赖Grad-ECLIP，计算成本和解释性受限。

---

## 307. Maximally Robust Satisficing Bayesian Optimization

**arXiv ID:** 2607.13652 | [PDF](https://arxiv.org/pdf/2607.13652v1)

**作者:** Samuli Kinnunen `[一作]` (University of Helsinki), Arto Klami `[通讯]` (University of Helsinki)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出在优化阶段可控、部署后受扰动的黑盒函数下，寻找满足阈值且对最大扰动鲁棒的解。

**💡 创新点**

创新点在于将“最大鲁棒满足解”问题形式化，并设计信息增益驱动的MRSBO采集函数，实现在非扰动优化环境下高效发现鲁棒解。

**🔧 技术方法**

主要技术包括高斯过程代理模型、随机傅里叶特征抽样、软最小距离近似以及信息论互信息采集。

**📊 数据集**

实验使用了合成的Hartmann、Branin、Gaussian mixture以及真实机器人推送任务。

**📈 对比分析**

与MES、StableOpt、AdveRS‑2等基线比较，MRSBO在大多数基准上实现零或更低的鲁棒性回报，收敛速度更快。

**⚠️ 局限性**

局限性包括当阈值接近全局最优时效率下降，以及缺乏对复杂高维/非光滑层集的理论收敛保证。

---

## 308. From Surface Forecasting to Observability Forecasting: A Latent World Model for Cloud-Aware EO Monitoring

**arXiv ID:** 2607.13651 | [PDF](https://arxiv.org/pdf/2607.13651v1)

**作者:** Mohanad Albughdadi `[一作]` `[通讯]` (European Centre for Medium-Range Weather Forecasts), Mohanad Albughdadi (European Centre for Medium-Range Weather Forecasts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将LeWorldModel应用于云遮蔽下的地球观测可观测性预测，预测下一次获取是否可用及可用视角出现时间。

**💡 创新点**

创新在于把关注点从表面像素预测转向可观测性预测，并将JEPA世界模型与云掩模、气象驱动结合，提供可观测性时间序列的隐状态预测。

**🔧 技术方法**

使用LeWorldModel（基于ViT-tiny编码器、MLP投影、条件自回归Transformer预测器）配合云掩模、天气协变量进行多模态时间序列建模。

**📊 数据集**

利用EarthNet2021数据集的Sentinel‑2多光谱序列和对应气象变量，将原始小立方体转换为带云掩模和天气协变量的HDF5序列。

**📈 对比分析**

在Locked协议下与Persistence基线和冻结的LightGBM表格基线对比。LeWM在连续可观测性回归和精确首次可用时刻预测上显著优于Persistence，且在IID、O、极端分割上对连续回归和精确时序任务表现更好；对“是否在六步内出现可用观测”的二分类任务则LightGBM更强。

**⚠️ 局限性**

主要局限为对分布外（OOD）性能下降，模型仅提供全局隐状态而非空间图像预测，对可观测性预测仅给出点估计缺乏不确定性校准，且依赖未来气象协变量，无法直接用于策略规划。

---

## 309. Beyond Color Geometry: Evaluating Human-Like Color Representations in Vision Models

**arXiv ID:** 2607.13647 | [PDF](https://arxiv.org/pdf/2607.13647v1)

**作者:** Ayan Igali `[一作]` (Kazakh--British Technical University), Pakizar Shamoi `[通讯]` (Kazakh--British Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一套基于模糊人类颜色分类模型COLIBRI的评价框架，用来衡量视觉模型（如ViT）对颜色的内部表征是否符合人类的颜色感知。

**💡 创新点**

创新点在于：①引入模糊的颜色归属向量，既保留了人类颜色类别的重叠和梯度特性；②设计了一个控制色彩几何的偏相关指标，能够区分模型对颜色几何与对人类颜色结构的契合度；③系统对11种不同训练策略的ViT模型进行了层级动态和自然图像实验的全面比较。

**🔧 技术方法**

主要技术包括：表示相似度分析（RSA）、Spearman相关与偏相关、k-means聚类与ARI评估、线性分类器（balanced accuracy）进行前景/背景颜色解码，以及层级均值池化来追踪梯度颜色信息随层变迁的变化。

**📊 数据集**

使用的数据集包括：1）330个WCS Munsell色块（平面与阴影球体），2）Tiny ImageNet子集的MegaCOIN，提供前景/背景颜色标签。

**📈 对比分析**

比较方法：对每个模型的嵌入构建相似度矩阵，计算与COLIBRI的相关性（ρ_fuzzy）和去除色彩几何后的偏相关（p_Δ）；对类别边界与紧密度分别用Silhouette和ARI评估；在自然图像上用前景/背景颜色分类精度与差值Δ衡量对象特定性。结果显示，Masked Autoencoder（MAE）模型在ρ_fuzzy和p_Δ上均显著高于其他模型，其梯度颜色结构在层级上也保持更稳定；其余模型（CLIP、DINOv2、ViT等）在类别紧密度和边界方面差异不大。

**⚠️ 局限性**

局限性包括：①WCS芯片覆盖的COLIBRI类别有限（仅50种）；②评估模型多维度变化（数据、规模、patch大小、训练策略）未分离；③自然图像实验使用全图嵌入，未考虑精确目标分割；④仅评估ViT体系，未检验其他网络架构。

---

## 310. Evaluating Encoding Strategies for Closed-Loop Classification in Biological Neural Networks

**arXiv ID:** 2607.13644 | [PDF](https://arxiv.org/pdf/2607.13644v1)

**作者:** Martin Schottlender `[一作]` (Dresden University of Technology), Pit Hofmann `[通讯]` (Dresden University of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在CL-1平台上用闭环分类任务评估了多种时序编码与空间编码策略在生物神经网络中的性能。

**💡 创新点**

首次系统比较了速率、相位、突发和首次尖峰时延编码，并发现突发编码结合首次尖峰解码能显著提升闭环准确率；同时揭示空间电极选择与时序编码耦合的关键性。

**🔧 技术方法**

采用电刺激编码（多种时序策略）、MEA记录、首次尖峰解码、计数式光栅解码、奖励/惩罚反馈闭环算法、以及基于活跃度的电极选择方法。

**📊 数据集**

使用MNIST手写数字数据集（仅取两类0和1进行二分类，四类时作扩展）。

**📈 对比分析**

通过在相同神经培养物上多次实验比较不同编码-解码组合的闭环和开环准确率，突发+首次尖峰组合达95.6%闭环准确率，其他组合远低于此水平；空间编码优化也能提升性能。

**⚠️ 局限性**

局限在于仅在单一生物培养物上验证，扩展到多类任务时准确率急剧下降，说明当前刺激与解码框架对高维分类支持不足；此外电极漂移和激活不均匀会显著影响结果。

---

## 311. OvisOCR2 Technical Report

**arXiv ID:** 2607.13639 | [PDF](https://arxiv.org/pdf/2607.13639v1)

**作者:** Shiyin Lu `[一作]` (Alibaba Group), Weihua Luo `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一款0.8B参数的端到端文档解析模型OvisOCR2。

**💡 创新点**

创新点在于构建混合真实+合成的训练引擎，并通过多阶段训练（SFT+RL+OPD+融合）实现轻量级模型超越流水线方法。

**🔧 技术方法**

采用了多任务监督、GRPO强化学习、对策略的on‑policy蒸馏和模型融合等技术，并使用了自定义的规则化Markdown序列化和合成HTML生成。

**📊 数据集**

训练数据来自过滤后的真实文档注释、合成HTML页面以及公开基准OmniDocBench、PureDocBench，评测亦包含自建的多样化内部基准。

**📈 对比分析**

在OmniDocBench v1.6上获得96.58分，PureDocBench Avg3 75.06分，且在内部基准中保持最高整体得分，明显优于现有流水线和端到端模型。

**⚠️ 局限性**

局限在于对降质真实图像的鲁棒性仍不足，手写和复杂表格场景的性能仍有提升空间。

---

## 312. FastCentNN: Accelerating Centroid Neural Network with Entropy Proxy

**arXiv ID:** 2607.13613 | [PDF](https://arxiv.org/pdf/2607.13613v1)

**作者:** Le-Anh Tran `[一作]` `[通讯]`, Le-Anh Tran

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FastCentNN，通过早期分裂策略加速原始CentNN，减少低移动epoch的冗余训练

**💡 创新点**

创新点在于使用全局质心总移动量作为熵代理来触发分裂，并提供绝对和相对阈值两种模式

**🔧 技术方法**

采用竞争式学习的winner‑loser更新机制，并结合移动阈值+耐心窗口的分裂触发算法

**📊 数据集**

实验数据集包括二维合成数据集(A1,A2,S1,S2,R15,Aggregation)以及高维图像数据集MNIST和Fashion‑MNIST

**📈 对比分析**

与原始CentNN在相同参数下对比，FastCentNN在运行时间上提升5%至16%，训练轮次下降，聚类误差（ΔMSE）基本无显著差异

**⚠️ 局限性**

局限性包括对阈值和耐心参数的敏感性，且在高维数据中加速幅度相对有限

---

## 313. AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities

**arXiv ID:** 2607.13705 | [PDF](https://arxiv.org/pdf/2607.13705v1)

**作者:** Zichen Ding `[一作]` (Shanghai AI Laboratory), Dongsheng Zhu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AgentCompass，一个可插拔、异步、可组合的 LLM 代理评估框架，能统一多种 benchmark、harness 与环境，实现可扩展的轨迹追踪与可重复实验。

**💡 创新点**

核心创新在于将 Benchmark、Harness、Environment 完全解耦并通过轻量级装饰器注册实现可组合配置，配合标准化协议与异步运行时大幅降低评估工程成本，首次实现跨 benchmark、跨 harness 的统一性能对比。

**🔧 技术方法**

采用 Python、异步 I/O（asyncio）、装饰器注册、标准化协议与插件化分析器，结合多进程调度与状态恢复实现高并发、容错评估。

**📊 数据集**

覆盖 20+ 公开 benchmark，涵盖工具使用、Web 与研究、科学推理、agentic coding、生产力等五大核心能力，包含 SWE-bench、GAIA、HLE、DeepSearchQA、PinchBench、SkillsBench 等数据集。

**📈 对比分析**

通过统一的 AgentCompass 协议对同一 benchmark 采用不同 harness（如 OpenClaw、OpenHands、Mini-SWE-agent）或不同模型（Qwen3.5、DeepSeek、Gemini、Claude 等）进行交叉实验，展示模型表现随 harness 变化显著，且与官方基线存在 5–15 分的差距，验证了框架对比可重复性的有效性。

**⚠️ 局限性**

仍依赖手动注册新组件，部分 benchmark 对环境配置敏感，且目前尚未覆盖所有潜在工具生态，评估结果对具体环境实现的依赖可能导致跨平台可复现性受限。

---

## 314. Barnamala: Parameter-Efficient Handwritten Devanagari Recognition at Benchmark Saturation

**arXiv ID:** 2607.13689 | [PDF](https://arxiv.org/pdf/2607.13689v1)

**作者:** Ashish Thapa `[一作]` (Ampixa Labs), Samrat Karki `[通讯]` (Pulchowk Campus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了1.11M参数的紧凑卷积网络，在46类Devanagari字符识别基准DHCD上实现99.73%的准确率，匹配并不低于先前大模型的表现；

**💡 创新点**

证明了DHCD基准已饱和，使用McNemar检验和Wilson置信区间揭示所有模型共享11个错误的不可突破阈值，并系统评估了知识蒸馏、教师集成、测试时增强等方法的效益；

**🔧 技术方法**

采用预激活SE-ResNet结构、知识蒸馏（温度T=4）、混合精度训练、EMA权重、数据增强（旋转、仿射、混合等）以及对比实验的统计检验；

**📊 数据集**

主要使用DHCD 46类手写Devanagari数据集，另外在CMATERdb数字子集上做零射和微调迁移实验；

**📈 对比分析**

通过与重现的17.32M参数MallaNet基线的Exact McNemar比较，学生模型在错误率上与基线无显著差异（p=0.345），在CPU推理速度上比大模型快9.5倍，参数量下降15.6倍；

**⚠️ 局限性**

局限性在于基准数据集本身的标签噪声和书写歧义导致11个错误的不可逾越错误底线，且迁移实验仅覆盖数字子集，无法验证对全字符集的泛化能力。

---

## 315. Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation

**arXiv ID:** 2607.13686 | [PDF](https://arxiv.org/pdf/2607.13686v1)

**作者:** Hao Qin `[一作]` (University of Arizona), Chicheng Zhang `[通讯]` (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对通用奖励函数逼近的上下文组合半邦特问题算法（SquareCB.comb），通过求解带对数障碍的凸优化获得参与向量并采样组合动作，取得 √(mAT log|F|) 的下界匹配的最优累积回报上界。

**💡 创新点**

①将对数障碍正则化与半邦特反馈相结合，直接在参与向量空间（维数 A）求解，避免了指数维度的动作分布优化；②定义并利用组合平方损失 DEC（CS‑DEC）框架，实现对非线性奖励函数的理论分析；③通过批量式在线回归桩实现高效的奖励估计。

**🔧 技术方法**

批量式在线平方损失回归桩、对数障碍正则化凸优化、组合动作的采样（如依赖等价、最大流分解、Frank‑Wolfe）以及基于梯度提升树或线性回归的回归桩。

**📊 数据集**

MSLR‑WEB30k 与 Yahoo! Learning‑to‑Rank Set 1 两个公开排序数据集；通过标准的监督‑到‑半邦特归约将其转化为 CCSB 实例。

**📈 对比分析**

与 SquareCB.Lin、VCEE、LinUCB、ε‑greedy、Uniform‑Random 及 Skyline 基线比较；在两大数据集上，SquareCB.comb 在大多数设置下取得与 Skyline 接近、远优于 Uniform‑Random 的平均奖励，且在梯度提升树回归桩下与 SquareCB.Lin 和 VCEE 的性能相当，偶尔略有优势。

**⚠️ 局限性**

仅适用于可实现（realizable）且有限函数族的情形；对奖励函数误判时需要额外的 ε‑misspecified 处理；算法对 m 的依赖在 m 较大时仍可能显著；采样器的实现依赖于组合结构的可解性，非结构化组合集合难以高效采样。

---

## 316. Self-Evolving Agent Harnesses via Gated Semantic Quality-Diversity

**arXiv ID:** 2607.13683 | [PDF](https://arxiv.org/pdf/2607.13683v1)

**作者:** Xiaotian Luo `[一作]` (EverMind AI), Yafeng Deng `[通讯]` (EverMind AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个自演化的 LLM 代理‑护具框架，通过将修补建议与信用判定分离，使用确定性门控来保证每一次提升都是真实且可复现的。

**💡 创新点**

创新点在于：①将提案与信用评估分离，所有统计判定由确定性代码完成；②引入“基于病理的 MAP‑Elites (GSME)”档案，按错误类型而非任务分布组织修补；③采用配对 2σ 显著性门与环境有效性门，保证在封闭测试集上真正的提升；④利用强大模型诊断失败并生成语义补丁。

**🔧 技术方法**

主要技术包括：LLM 诊断与补丁生成（使用 Claude Opus 4.8），确定性评估管道（采样、门控、配对统计）、GSME 质量多样性存档、Git 版本控制、对称 2σ 显著性测试、验证器（Per‑task pass@1 等）。

**📊 数据集**

使用的数据集覆盖七个领域：Terminal‑Bench‑2、EvoAgentBench（LiveCode、Omni‑MATH、BrowseComp+、GDPval、SWE‑bench）以及 AppWorld，模型为 Qwen3.6‑27B（冻结）及其他两款模型。

**📈 对比分析**

与原始未改进的护具进行对比，使用封闭测试集评估；在所有六个可计量基准上均获得 +9~+15.5 百分点的提升，且在封闭测试集上显著性门通过，保留率 86–147% 的训练提升，证明提升具有泛化能力。

**⚠️ 局限性**

局限性：①依赖可靠的任务验证器，若验证器存在漏洞仍可能被误导；②目标是短期可验证的成功率，未覆盖长期鲁棒性、可维护性等；③评估成本高，需频繁的完整评分；④对开放式任务或无明确评估指标的场景适用性有限。

---

## 317. Exploratory, Communicative, and Deployable: Vision-Driven Embodied Agents for Open-World Mobile Manipulation

**arXiv ID:** 2607.13653 | [PDF](https://arxiv.org/pdf/2607.13653v1)

**作者:** Boyu Mi `[一作]` (Shanghai Jiao Tong University), Hanqing Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个无 Oracle 感知、闭环式的移动操作框架，集成多级视觉探索工具、模拟用户交互以及基于 SFT + GSPO 的训练管线，训练 Qwen3‑VL‑8B 以完成开放世界移动任务。

**💡 创新点**

创新点在于：①去掉 oracle 感知接口，仅依赖可在真实机器人上实现的 RGB‑基工具链；②引入模拟用户实现主动询问与意图消歧；③使用层级 SFT 与 Group Sequence Policy Optimization (GSPO) 实现高层决策与低层动作的闭环优化；④构建包含 241 任务的 REAL Benchmark，覆盖主动探索、视觉干扰、关节操作与交互消歧四大任务族。

**🔧 技术方法**

采用 Qwen3‑VL‑8B 视觉语言模型结合结构化工具调用，基于 GRUtopia 仿真器的多级视觉工具（导航、扫描、检测、对齐），行为克隆 + GSPO 强化学习，MCP 接口实现高层与低层的标准化对接。

**📊 数据集**

利用 GRScenes（7 个场景、100 家具、2,500 物体）与 MesaTask（639 类别）生成的场景和 241 任务（REAL Benchmark），并通过规则规划生成轨迹、LLM 生成模糊指令和对话，形成 SFT + RL 训练数据。

**📈 对比分析**

与多种零样本 VLM（Gemini‑3‑Pro、GPT‑5、Claude‑Haiku‑4‑5、Qwen3‑VL‑235B）以及基准 SOTA 进行对比。SFT+RL 版本在模拟环境中取得最高 56.9% 的 SUL 成功率；在物理 LIFT2 双臂机器人上实现 78.3% 的端到端成功率、11.6 的 SPL、63.1% 的询问率与 68.4s 的平均推理延迟。

**⚠️ 局限性**

局限性包括：①任务主要聚焦跨接收器搬移，缺少时序/空间关系子任务；②模拟用户行为受限，难以覆盖真实用户可能的实时目标变更与隐式提示；③接收器被视为整体实体，缺乏部件级分割导致细粒度定位困难；④冻结视觉编码导致对细粒度视觉辨识的瓶颈；⑤RL 受环境复杂度和奖励设计的影响，仍有部分错误类型未能完全消除。

---

## 318. A Distributed Framework for Compiling and Reasoning with d-DNNF

**arXiv ID:** 2607.13642 | [PDF](https://arxiv.org/pdf/2607.13642v1)

**作者:** Zhenghang Xu `[一作]` (Northeast Normal University), Jean-Marie Lagniez `[通讯]` (University of Artois)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个分布式知识编译器和推理引擎，用于在大规模集群上生成和查询决策-DNNF（Decision‑DNNF）结构，解决了单机编译阶段的内存和时间瓶颈。

**💡 创新点**

创新点在于将Cube‑and‑Conquer策略用于知识编译，使得子问题的子电路可在各节点本地生成并持久化，同时通过虚拟全局OR门实现逻辑组合；此外，系统在编译完成后持续运行，提供分布式模型计数、直接访问和均匀采样等查询服务。

**🔧 技术方法**

核心技术包括：分布式Cube‑and‑Conquer搜索空间划分、分布式d‑DNNF编译器（基于d4v2）、MPI通信协议、局部缓存的模型计数、基于计数的直接访问与采样路由，以及对模型计数结果的全局聚合。

**📊 数据集**

实验使用了MC 2025模型计数竞赛的基准集（约数百个实例）和一组适合查询评估的工业/研究性CNF实例（200个子集），在32节点（共128核）集群上进行评测。

**📈 对比分析**

与单机最先进编译器（如d4v2_seq）相比，分布式系统在128核下实现了高达~1.5‑2.0倍的实例通过率提升，模型计数查询时可获得近线性加速；在轻量级实例中，通信开销导致性能略逊；直接访问查询受网络往返影响明显，采样查询则表现出更好的规模化优势。

**⚠️ 局限性**

限制主要体现在：1) 对小规模或易编译实例，通信和同步开销削弱了并行收益；2) 直接访问等逐步查询需要多次全局计数，导致延迟高；3) 目前仅支持无权重的d‑DNNF，无法处理加权模型计数；4) 系统对节点可靠性和网络延迟敏感，需进一步优化协议与缓存策略。

---

## 319. VIP-MINGLE: A Corpus for Videoconference and In-Person Multimodal Interaction in Group Language Engagement

**arXiv ID:** 2607.13614 | [PDF](https://arxiv.org/pdf/2607.13614v1)

**作者:** Andrew Chang `[一作]` (New York University), Dustin Freeman `[通讯]` (New York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出VIP-MINGLE数据集，包含约59小时的同一组参与者在现场和视频会议两种环境下完成相同任务的多模态对话录音，并对两环境下的语音时序、语言复杂度、面部表情等行为差异进行对比分析。

**💡 创新点**

创新点在于采用同一组参与者的within-subjects设计，首次提供可直接比较的双设置多模态对话语料库，揭示了现场与视频会议在交流行为上的显著差异，并为跨域对话建模奠定基础。

**🔧 技术方法**

使用了Zoom录制、pyannote speaker diarization、Whisper转录、OpenFace与DeepFace面部特征提取、线性混合效应模型、语言模型计算语法复杂度等多种技术。

**📊 数据集**

数据集为VIP-MINGLE（约59小时、32组、105人），同时参考AMI、ICSI、CANDOR、RoomReader等公开语料进行对比。

**📈 对比分析**

通过配对Wilcoxon符号秩检验和线性混合效应模型对语音时序、语言复杂度、面部表情等指标进行统计比较，结果显示视频会议下停顿更长、发言更短、语法复杂度更低、面部表情不如现场活跃，人类评估亦表明现场对话更受欢迎。

**⚠️ 局限性**

局限性包括任务仅限于“Family Feud”式游戏，可能不代表其他对话情境；样本来源单一，缺乏更广泛的人群多样性；未构建跨域自适应模型，且仅提供基线特征，未来需扩展至更自然、长时对话场景。

---

## 320. nuTruck: Benchmarking Autonomous Driving Planning for Distributed Electric-drive Trucks

**arXiv ID:** 2607.13704 | [PDF](https://arxiv.org/pdf/2607.13704v1)

**作者:** Jinyu Miao `[一作]` (Tsinghua University), Diange Yang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了针对分布式电驱动卡车（DET）的高保真闭环评估基准nuTruck，集成了非线性车辆动力学、翻车风险评估指标以及多种规则与学习型规划器基线；

**💡 创新点**

首创面向DET的闭环基准，首次将完整的非线性动力学模型（包括轮胎、刚体、垂直负荷转移）与翻车风险指标融合进评估；同时通过动作模式（FRO、RWL）对学习型规划器的高维动作空间进行先验约束，显著提升收敛速度；

**🔧 技术方法**

使用Pacejka轮胎魔法模型、刚体动力学与Euler传播；采用iLQR轨迹跟踪控制与iLQR-S（带翻车惩罚）；实现CaRL强化学习规划（动作空间12维、4维等）；在PyTorch框架下构建闭环仿真环境；

**📊 数据集**

主要使用nuPlan真实驾驶日志生成场景，替换车辆参数后筛选DET可行场景；使用TruckSim 2019.0作为动力学模型验证基准；

**📈 对比分析**

对规则规划器（IDM、PDM-Closed、Plan-R1）和学习规划器（CaRL系列）在三类动态场景（静态、非反应、反应）下分别计算CLS、CLS-Safe、NRS、EPAR；实验表明传统轨迹规划器尽管碰撞避让好，但翻车风险高；CaRL行动规划器在碰撞避免与翻车安全上均表现最佳，获得最高的CLS-Safe和NRS；

**⚠️ 局限性**

局限性包括：基准仍以nuPlan小车场景为基础，可能对更复杂交通场景的泛化不足；动力学模型假设为单车，未覆盖多车交互导致的非线性耦合；动作模式需要人工设计，可能限制了模型的自适应性；RL训练对计算资源与数据量要求高，且未与感知模块耦合进行端到端学习。

---

## 321. Conditional Invertible Neural Networks for Data-Driven UAV Control: A 2-D Proof of Concept

**arXiv ID:** 2607.13703 | [PDF](https://arxiv.org/pdf/2607.13703v1)

**作者:** Christian Wittke `[一作]` (Helmut Schmidt University), Oliver Niggemann `[通讯]` (Helmut Schmidt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对平面 X8 轴对称多旋翼机的逆动力学进行概率建模，使用条件可逆神经网络（cINN）学习 p(u | s_t, c_t) 并将其作为闭环控制器。

**💡 创新点**

提出将逆动力学转化为条件可逆流模型，兼顾精确预测、可解释不确定性估计，并在训练时利用 INDI 教师数据实现概率推断。

**🔧 技术方法**

采用条件可逆神经网络，使用 rational‑quadratic spline 结合可学习 1×1 线性混合、激活归一化和正弦‑余弦姿态编码；训练目标为负对数似然；推理时通过逆流求解模式并采样获得不确定性。

**📊 数据集**

在 RMT‑CopterGym 仿真环境中生成约 1.25 M 条样本（1000 场景，12.5 s 每场），所有数据均由 INDI 教师生成，包含状态、命令和对应的八轴电机命令。

**📈 对比分析**

在开环评估中获得 R² = 0.944、RMSE ≈ 121 rad/s；在 15 个闭环场景中平均位置 RMSE 9.7 m（与 INDI 的 9.5 m 接近），但仅约 47 % 的场景成功或可接受，主要失效由相位滞后和姿态发散导致。

**⚠️ 局限性**

限制包括：模型在训练稀疏区过度自信、不足以处理激进命令导致的姿态发散、推理耗时 47.5 ms 远超 10 ms 控制周期、仅在二维平面验证，缺乏对全 6‑DoF 航迹和硬件部署的验证。

---

## 322. FreeLit: Paired-Free Indoor Relighting via Physics-Guided Diffusion

**arXiv ID:** 2607.13656 | [PDF](https://arxiv.org/pdf/2607.13656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. Towards a Modular Bin-picking Framework for Handling Object Pose Uncertainties

**arXiv ID:** 2607.13698 | [PDF](https://arxiv.org/pdf/2607.13698v1)

**作者:** Frederik Hagelskjær `[一作]` `[通讯]` (University of Southern Denmark), Frederik Hagelskjær (University of Southern Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个模块化的Bin‑picking框架，结合物体姿态分布估计、二视角融合、重新定向托盘以及手内姿态验证，以提升对姿态不确定性和抓取误差的鲁棒性。

**💡 创新点**

首次将姿态不确定性与抓取误差联合处理，采用模块化设计允许自由组合，利用二视角融合显著降低不确定性，并通过重新定向托盘和手内验证实现100%插入成功率。

**🔧 技术方法**

使用视觉姿态分布估计（SO(2)旋转分布）、Product of Likelihoods 融合、机器人抓取、上视RGB相机手内检测、重定位托盘等技术。

**📊 数据集**

实验使用了三种真实工业工件（Novo、WRS‑08、WRS‑11）以及在World Robot Summit Assembly Challenge中的两个测试物体。

**📈 对比分析**

通过与原始方法对比，加入模块后成功率提升至100%，平均每次插入所需抓取次数从原来的2.5次降至1.91次，表现出显著的效率和鲁棒性提升。

**⚠️ 局限性**

局限性包括仅支持SO(2)旋转，无法处理完整的SE(3)姿态；未对抓取误差进行建模；二视角固定位置可能限制对不同场景的适应；对形状不规则或高度对称的物体效果尚未验证。

---

## 324. Anatomy of Uncertainty: Expressive Descriptors of Robotic Manipulator Motion for Non-verbal Communication in Human-Robot Collaboration

**arXiv ID:** 2607.13696 | [PDF](https://arxiv.org/pdf/2607.13696v1)

**作者:** Ridhima Bector `[一作]` (Nanyang Technological University), Bernhard Johannes Schmitt `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套基于 Laban 动作分析的数学框架，将机器人感知不确定性映射到操纵臂的运动表达，通过 Commitment‑Vigilance 状态空间定义四种不确定性行为（自信、好奇、犹豫、恐惧），并设计了五个运动原语（接近、停顿、后退、探索、振荡）与十一种可量化的运动描述子。随后通过视频问卷对四种基线轨迹及其变体进行人类感知验证，证明了这些运动能够可靠地传达不确定性状态及其强度。

**💡 创新点**

创新点包括：①将 Laban Effort 因子系统性投射到 Commitment 与 Vigilance 两维空间，形成连续的情绪/不确定性模型；②构建了一套可计算的运动描述子，用以参数化机器人轨迹，桥接内部不确定性与可观测运动；③通过实验验证了该框架在非语言交互中的有效性，首次系统评估了各描述子对表达强度的影响。

**🔧 技术方法**

使用技术包括：Laban 动作分析理论、Commitment‑Vigilance 线性映射、运动原语与描述子设计、视频生成与在线问卷收集、卡方拟合检验、效应量（Cohen's w）分析。实现层面使用机器人运动学、前向运动学、轨迹参数化及统计软件进行分析。

**📊 数据集**

使用的数据集为 55 名完成实验的受试者（共 74 条问卷，剔除不完整或异常后保留 55 条）所产生的视频轨迹与问卷回答，未使用公开机器人大赛或仿真数据集。

**📈 对比分析**

比较方法为基于卡方拟合检验与后续标准化残差分析，结果显示四种基线轨迹分别显著偏离随机分布，识别率在 90% 以上；在描述子调节实验中，关键描述子（如停顿长度、接近加速度、后退距离、眼向幅度等）对表达强度影响显著，效应量较大。总体性能表现为高识别度与可调节性，但尚无实时闭环评估。

**⚠️ 局限性**

局限性包括：①所有轨迹与描述子均为手工设计，缺乏自适应生成机制；②未在真实机器人控制循环中验证，实验仅基于视频；③描述子变化幅度受限，某些指标（如振荡幅度、转速）未能显著区分情绪；④缺乏多任务、多机器人、不同环境的泛化评估。

---

## 325. Towards Spatial Supersensing in the Wild

**arXiv ID:** 2607.13681 | [PDF](https://arxiv.org/pdf/2607.13681v1)

**作者:** Tianjun Gu `[一作]` (Tsinghua University), Yiming Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了VSI‑Super‑Wild基准，评估多模态大型语言模型在真实世界长视频中的空间超感知能力。

**💡 创新点**

创新点在于：①采集无剪辑的真实长格式全景视频，覆盖8个场景类别；②设计基于认知的三锚任务（代理、物体、环境）并提出四类挑战性问题；③引入Shuffle24评估方式，减少选项偏差。

**🔧 技术方法**

采用多模态LLM框架，结合视频分帧、摄像机位姿估计（Depth‑Anything‑3）、物体检测（YOLO‑World）和实例分割（SAM3）生成元数据，再用规则生成四类Q&A并人工验证。

**📊 数据集**

使用了442段真实全景视频（共284.5小时）和6980条人工校验的问答对。

**📈 对比分析**

对13种主流MLLM（包括GPT‑5.4、Gemini、Cambrian‑S、Qwen系列等）进行评估，平均整体准确率约35%，最优模型Gemini‑3.1‑Pro仅达44.36%，显著低于理想水平，尤其在数字计数任务和长时间视频上表现更差。

**⚠️ 局限性**

局限性在于：①缺乏稳定的三维世界状态记忆与更新机制；②对代理和环境的空间推理能力不足；③长时序表现随视频长度急剧下降；④依赖框架与人工验证，难以大规模扩展。

---

## 326. WarpGuard: Towards Control-Flow Attestation for Heterogeneous CPU-GPU Execution

**arXiv ID:** 2607.13640 | [PDF](https://arxiv.org/pdf/2607.13640v1)

**作者:** Christian Lindenmeier `[一作]` (NVIDIA), Ahmad Atamli `[通讯]` (NVIDIA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

为嵌入式安全场景提供了第一个能够联合验证 CPU 与 GPU 运行时控制流完整性的框架 WarpGuard，能检测 GPU 核心控制流劫持及 CPU‑GPU 交互攻击。

**💡 创新点**

创新点包括：① 在 GPU 侧实现基于 warp 的控制流追踪，显著压缩记录量；② 在 CPU 侧记录 GPU 调用点并与 GPU 核心哈希和启动配置绑定，形成“GPU 调用点策略”；③ 通过组合 CPU 与 GPU 的 CFG 构建统一的“复合 CFG”，实现跨设备的完整性验证；④ 仅依赖软件动态二进制插桩，无需专用硬件或二进制修改。

**🔧 技术方法**

技术细节：CPU 侧使用 DynamoRIO 进行 BB/函数级别插桩；GPU 侧使用 NVBit 在 PTX 级别插桩并实现 warp‑level 追踪；对每个 GPU 核心生成基于内容的哈希；通过 TCP 与远端 verifier 传递 CPU、GPU 启动和追踪日志；使用离线生成的 CFG 与启动点策略构成复合模型进行验证。

**📊 数据集**

实验数据集：① 自定义 microbenchmarks（不同 BB 大小、线程数、启动次数）；② SPECaccel 2023 基准（16 项 GPU 加速程序）；③ 8 个 TensorRT AI 推理引擎（ResNet‑50、Inception‑v4、VGG‑19、MobileNet、YOLOv3‑Tiny、Pose、Super‑Res、U‑Net）。

**📈 对比分析**

对比方法：与基线（无插桩）、仅 CPU 插桩（DynamoRIO^bb_full）、仅 GPU 插桩（NVBit^bb/func_empty/full）、完整配置（^bb_full、^func_full）。
性能表现：
• microbenchmarks：BB 级追踪平均 5–30× 开销，函数级追踪 1–5×；
• SPECaccel：BB 级 2.4×–128×（平均 39.3×），函数级 1.2×–113×（平均 26.7×）；
• TensorRT：BB 级 15.5×–120×，函数级 1.9×–6.9×，仍保持 edge‑device 可接受吞吐量（如 MobileNet ≥ 100 QPS）。

**⚠️ 局限性**

局限性：① 依赖软件插桩导致 CPU 侧 5–10% 以上的额外开销；② GPU 追踪缓冲区位于可被攻击者写入的设备全局内存，可能被篡改；③ 目前不支持 JIT‑编译或动态生成的 GPU 代码；④ 没有硬件级的加速或安全存储，无法完全防御内部攻击；⑤ 仅在 NVIDIA Ampere 系列实现，需针对其他 GPU 或加速器适配。

---

## 327. Design and Control of the "QuadBoat": A Quadruped Surface Vehicle for Drowning Rescue

**arXiv ID:** 2607.13633 | [PDF](https://arxiv.org/pdf/2607.13633v1)

**作者:** Lianxin Zhang `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Huihuan Qian `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文设计并实现了一款名为QuadBoat的仿生四足水面无人艇，配备四条可调节姿态的腿和外舷船体，实现对水面目标的捕捉、提取与运输，并通过实验验证其机动性与控制性能。

**💡 创新点**

创新点包括：
1) 采用四足结构与外舷船体的组合，使艇体能够在水面上站立、伸缩并改变宽度高度，显著提升通过狭窄通道与越过障碍的能力；
2) 开发了基于逆运动学的腿部控制与级联MPC‑PID全局运动控制，兼容全致动与欠致动两种模式；
3) 引入基于AprilTag的视觉反馈实现自动定位与提取，提升救援自动化水平。

**🔧 技术方法**

使用的技术与方法包括：
- 逆运动学与Denavit–Hartenberg建模；
- 级联模型预测控制（MPC）与PID控制器；
- 机械臂关节与舷体舵机闭环控制；
- 视觉检测与AprilTag定位；
- ROS系统框架与CasADi求解器；
- 运动捕捉系统（或GPS）与双摄像头反馈。

**📊 数据集**

实验使用的“数据集”主要为实验室和户外水池中自行生成的目标轨迹（圆形、八字形）以及放置在水面的泡沫块或人形模型。未使用公开标注数据集。

**📈 对比分析**

比较方法：
- 与现有单/多船体USV（EMILY、SWIFT、ROAZ、PRIME）及其他可变形USV的轨迹跟踪误差进行对比；
- 轨迹跟踪误差：圆形轨迹平均距离误差1.25‑1.28 cm，方向误差0.0033‑0.0083 rad；八字形轨迹平均距离误差1.92‑2.13 cm；
- 目标跟踪误差：X、Y约1.6‑1.8 cm，偏航误差0.0849 rad；
- 目标提取成功率：室内70 %，室外50 %，平均完成时间≈116 s（室内）/132 s（室外）。总体性能优于公开的参考USV实现。

**⚠️ 局限性**

局限性：
- 室外实验受风浪、光照反射等环境因素影响，成功率与时间下降；
- 现有控制速度受限，最高可达0.8 m/s，急救场景中对高速响应的需求仍未完全满足；
- 视觉识别仍基于Apriltag，实际水面中人类浮沉面貌识别能力不足；
- 机械腿与舷体结构未实现陆地行走，缺乏完整的水陆跨界功能。

---

## 328. How the Hessian-Spectrum of Neural Networks Depends on Data

**arXiv ID:** 2607.13631 | [PDF](https://arxiv.org/pdf/2607.13631v1)

**作者:** Jasraj Singh `[一作]`, Antonio Orvieto `[通讯]` (ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

推导了任意宽度、任意深度线性网络在任意规模数据集上的Hessian特征值，并阐明了其与数据几何和标签分布的关系，随后通过实验验证理论预测。

**💡 创新点**

首次给出了完整的Hessian谱解析，并揭示在MSE分类任务中解的尖锐度与最大标签比例直接相关，在保留实际网络假设的前提下对深层网络提供精确的预测。

**🔧 技术方法**

采用广义高斯‑牛顿近似结合神经切线核（NTK），利用特征矩阵的SVD与奇异值分解得到特征值界限，并使用Hutchinson方法进行数值验证。

**📊 数据集**

在MNIST、FashionMNIST、CIFAR‑10等公开数据集以及合成的等方位特征数据上进行了实验。

**📈 对比分析**

将理论预测的前10个特征值与数值计算得到的特征值进行对比，误差随训练进度下降，预测在大多数情形下保持较高准确性。

**⚠️ 局限性**

理论依赖输入特征等方位、权重平衡、线性模型和零训练误差等理想化假设，非白化输入、非平衡权重或加入非线性时预测失效；仅适用于MSE损失，难以直接推广到更复杂网络。

---

## 329. STOCKTAKE: Measuring the Gap Between Perception and Action in LLM Agents with a Fair Oracle

**arXiv ID:** 2607.13618 | [PDF](https://arxiv.org/pdf/2607.13618v1)

**作者:** Sagar Deb `[一作]` (QpiAI), Ashwanth Krishnan `[通讯]` (QpiAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出26周供应链库存补货任务STOCKTAKE，采用因式POMDP并提供公平的贝叶斯参考策略

**💡 创新点**

创新点在于可计算的公平贝叶斯oracle和将状态估计与控制分离评估的指标（技能分数、检测延迟、知情-行动率）

**🔧 技术方法**

采用因式POMDP建模、精确贝叶斯滤波、采样式规划以及LLM生成的行动与说明

**📊 数据集**

使用Synthetic种子生成的26周库存补货情景，包含六个隐藏因子和三类压力配置（孤立、持续、复合）

**📈 对比分析**

通过与基准floor和oracle的成本比较以及对每周推理的自动评分，四大LLM模型的检测率达84–88%，但技能分数差异显著，最差模型甚至低于基准floor

**⚠️ 局限性**

局限包括仅覆盖单一SKU单一任务、仅四模型、oracle依赖真实参数导致不完全公平、单次运行导致噪声、未评估多轮实验等

---

## 330. Calibrated Closed-Form Uncertainty for Radiative Gaussian Splatting in Sparse-View CT

**arXiv ID:** 2607.13682 | [PDF](https://arxiv.org/pdf/2607.13682v1)

**作者:** Chulin Zhao `[一作]` (Dundee International Institute, Central South University), Shu Liu `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种可变分密度后验的辐射高斯剖分CT模型，并利用X射线渲染的线性特性实现闭式预测方差，生成每体素不确定性地图。

**💡 创新点**

创新点在于将每个高斯密度提升为高斯后验，借助线性渲染得到闭式方差传播；同时系统化评估不确定性校准，并设计覆盖门控、校准的视角采集策略。

**🔧 技术方法**

使用变分后验、KL先验、重新参数化采样、蒙特卡洛估计、温度缩放校准，以及Spearman、AUSE、ECE等评价指标，结合R^2‑Gaussian渲染器与体素化器实现。

**📊 数据集**

在X‑Gaussian开发场景（胸/腹/头）与官方15场景合成基准（512²投影，256³体素）上进行训练与评估。

**📈 对比分析**

与深度集成（K=5）、MC参考以及扰动启发式进行对比；在大多数场景中Spearman>0.6、ECE<0.1，平均PSNR仅低0.3 dB，闭式方差比MC快3–10倍；CGCA策略在圆轨测试中保持覆盖的同时节省约0.8 dB。

**⚠️ 局限性**

仅在合成数据上验证，未在真实临床扫描上测试；后验为部分因式分解，仅考虑密度不确定性，忽略相关性；温度缩放可能不具备跨场景/设备迁移性；部分场景误差几乎平坦导致排名无意义；GPU差异影响结果；未实验验证剂量自适应停止规则。

---

## 331. Vision-Based Obstacle Separation for Strawberry Harvesting in Clusters Using Hierarchical Reinforcement Learning

**arXiv ID:** 2607.13799 | [PDF](https://arxiv.org/pdf/2607.13799v1)

**作者:** Teng Li `[一作]` (Beijing Academy of Agriculture and Forestry Sciences), Ya Xiong `[通讯]` (Beijing Academy of Agriculture and Forestry Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种层次强化学习框架 VGPA，用于在高度遮挡的草莓簇中实现选择性采摘。

**💡 创新点**

创新点包括：1）基于视觉引导的高层决策模块，显著提升选项学习速度；2）渐进自适应探索策略 PAES，动态调节低层探索以稳定训练；3) 将障碍物分离与抓取整合为一连串操作，解决传统避障不足的问题。

**🔧 技术方法**

核心技术包括：层次强化学习（DIOL+DDPG+HER），YOLOv11 边缘检测，视觉引导与 PAES 结合的探索策略，PyBullet 模拟环境与实机（LingXtend）执行。

**📊 数据集**

数据集主要为在 PyBullet 中构建的多种遮挡情境的草莓簇模型，以及实地在北京翠湖农业技术有限公司温室中采集的 20×3 的真实场景数据。

**📈 对比分析**

与 HAC 基线、直接采摘和 VLM 方法对比，VGPA 在仿真中 96.7% 高层成功率，实机在不同遮挡级别的成功率分别为 88.3%、80.0%、71.7%，平均 80.0%，显著优于直接采摘（45.0–75.7%）且执行时间仅比 VLM 短 3.5 秒。

**⚠️ 局限性**

局限性主要体现在：遮挡严重时视觉定位误差大，真实草莓形变与脆弱性未在模型中充分考虑，且在极端遮挡下推掠动作可能连同目标草莓一起被吞噬导致失败。

---

## 332. Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations

**arXiv ID:** 2607.13770 | [PDF](https://arxiv.org/pdf/2607.13770v1)

**作者:** Wenxuan Miao `[一作]` (Shanghai Jiao Tong University), Yu Feng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对视频扩散Transformer（vDiT）的计算瓶颈，提出通道级时空重用算法并设计配套的可重配置Systolic‑Array加速器，显著降低自注意力及其他运算量；

**💡 创新点**

创新点在于：①系统揭示vDiT潜在空间中通道级时空相关性，并利用此特征实现轻量化通道重用；②配合可重配置PE与数据调度器，解决重用导致的稀疏性与工作负载不均衡问题；

**🔧 技术方法**

技术实现包括：8位定点通道重用算法（基于相似度阈值θ_th1/θ_th2），可重配置4×4 PE的Systolic‑Array，数据调度器（聚类+匹配）提升PE利用率，混合精度和部分重用机制；

**📊 数据集**

实验使用VBench评测集（950个提示）评估生成视频质量，生成5秒480×540分辨率视频；同时使用PSNR、SSIM、LPIPS和VBench得分进行量化；

**📈 对比分析**

性能比较：与A100/H100 GPU及Cambricon‑D、AdapTiV、Exion、Ditto等六种硬件基线、四种软件基线对比；速度提升5.2×–6.6×，能耗降低14.8×–18.4×；生成质量PSNR提升约17 dB，SSIM 0.87–0.90，VBench得分≈0.81，保持与基线相同或更优；

**⚠️ 局限性**

局限性：①重用方向固定，无法自适应任意时空方向；②大规模PE数组会降低对不规则重用的适配，导致利用率下降；③高速运动视频可重用率下降；④需手工设定阈值θ_th1/θ_th2，影响效果；⑤方案针对vDiT，迁移至其他模型需进一步验证。

---

## 333. MxGPS: Multiplex Graph Transformers for a Power Grid Foundation Model

**arXiv ID:** 2607.13763 | [PDF](https://arxiv.org/pdf/2607.13763v1)

**作者:** Charilaos Papaioannou `[一作]`, Elissaios Sarmas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了一种多任务图变压器 MxGPS，用来构建电网基础模型并解决单任务 GNN 在拓扑迁移中的过拟合问题。

**💡 创新点**

创新点在于将共享节点编码器与多任务专门化分支结合，并通过物理约束和交叉分支注意力实现跨拓扑的稳健性，显著降低拓扑迁移误差。

**🔧 技术方法**

采用了图变压器（GPS）、多任务学习、物理约束（BIM）、自监督预训练（MGT）、交叉分支注意力等技术。

**📊 数据集**

使用基于 MATPOWER / PGLib-OPF 的标准电网（14‑300 站）生成约 10k 场景的数据集。

**📈 对比分析**

在 3 折滑动窗口交叉验证下与 GCN、GAT、GPS、GNS、GNS-kit 等单任务基线比较，MxGPS 在零样本拓扑上实现 0% BVR，PF MAE 与最佳单任务相差约 10 倍，但零样本误差增幅仅 39%，参数量仅 1.6M，显著优于 GridFM 的 20M。

**⚠️ 局限性**

局限性包括仅测试至 300 站，需要使用可扩展注意力机制；K>2 任务组合尚未验证；SSE 零样本 BVR 在某些拓扑上仍不为 0；需要进一步研究最小训练组合和跨任务注意力的实际收益。

---

## 334. Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape Pre-, Intra-, and Post-CoT Calibration

**arXiv ID:** 2607.13753 | [PDF](https://arxiv.org/pdf/2607.13753v1)

**作者:** Shuhao Li `[一作]` (Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个三阶段（Pre‑CoT、Intra‑CoT、Post‑CoT）评估链式思维模型置信度校准的方法，并引入位置感知置信度(PosConf)策略以提升置信度在不同阶段的有效性。

**💡 创新点**

创新点在于：①把置信度校准拆解为“事前难度估计”“在线早停”“答案聚合”三个阶段，揭示不同后训练方法对置信度在各阶段的影响；②发现置信度在链式思维的不同位置会呈现不同的可靠性，并基于此设计只在可靠位置提取置信度的PosConf；③通过对比SFT、OPD、RL三种后训练策略，验证各自最适合的阶段并展示PosConf在RL和OPD上的显著提升。

**🔧 技术方法**

技术手段包括：使用基于token‑级别概率的熵式置信度；滑动窗口平滑（group confidence）实现在线置信度信号；基于阈值的Early‑Stopping；基于置信度过滤的答案聚合；以及自定义权重窗口实现位置感知置信度。模型以Qwen2.5‑7B‑Instruct为骨干，分别在SFT、OPD、RL三种后训练方案下进行微调。

**📊 数据集**

使用的评测数据集为四个数学推理基准：AIME 2024、AIME 2025、AMC 2023、MATH500，涵盖从中等到高难度的算术与几何问题。

**📈 对比分析**

比较方法：在同一模型骨干、相同训练数据集下，对Qwen‑Instruct、Qwen‑SFT、Qwen‑RL、Qwen‑OPD四个版本进行三阶段置信度评估。结果显示：OPD在Pre‑CoT阶段最能预测问题难度；SFT在Intra‑CoT阶段最适合在线早停；RL在Post‑CoT阶段对答案聚合效果最佳。PosConf进一步提升RL的答案聚合准确率6.1点、在低预算下提升OPD的早停性能4.3点，整体显示位置感知置信度可显著提升推理效率和准确性。

**⚠️ 局限性**

局限性：①实验仅覆盖Qwen2.5‑7B‑Instruct一个模型族；②只研究数学推理任务，未验证在更大规模模型或非数学推理场景下是否同样成立；③仅使用内部token‑概率衍生的置信度，未探索熵、概率间距、采样一致性等其它不确定性估计方法。

---

## 335. Cluster with Auctions for Vector Search

**arXiv ID:** 2607.13728 | [PDF](https://arxiv.org/pdf/2607.13728v1)

**作者:** Swann Bessa `[一作]` (Meta FAIR), Hervé Jégou `[通讯]` (Meta FAIR)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种用于高维向量近似最近邻搜索的划分方法——Cluster With Auctions (CWA)，通过分离划分与探测函数、联合优化并保证簇均衡，显著提升搜索效率与召回率。

**💡 创新点**

核心创新点在于：①将划分(h)与探测(fθ)分离开来，避免单一模型的约束；②利用交替优化与拍卖算法实现簇均衡的分配；③提出基于笛卡尔积的产品关键字变体，实现上百千簇的低参数、高效推理。

**🔧 技术方法**

采用的技术包括：交替优化框架（back‑propagation + 拍卖算法）、有约束线性分配、神经网络探测函数（残差FFN、两层线性+pairwise），以及HNSW图索引与产品多码本结构。

**📊 数据集**

实验使用四个公开基准数据集：SIFT (d=128)、Deep (d=96)、Text‑to‑Image (d=200)、LAION (d=512)，分别评估ID与OOD场景。

**📈 对比分析**

与K‑Means、USP、Neural LSH、BLISS、IVF‑HNSW、IMI等基线对比，CWA在Recall@10≈0.8时，QPS提升1.3–4.7倍，且在OOB场景下表现更为显著，显示出优越的召回与查询吞吐。

**⚠️ 局限性**

局限性：仅评估单一划分；未考虑多划分组合的互补性；对新加入数据库向量的在线分配尚未给出方法。

---

## 336. Interaction Density as a Behavioural Signature of Exhibit Type: A Minimal-Log Study from a Two-Venue Science Experience Centre

**arXiv ID:** 2607.13724 | [PDF](https://arxiv.org/pdf/2607.13724v1)

**作者:** R A Udaya Rakshith `[一作]` (PAIR Labs, Param Foundation), Umang J Gala `[通讯]` (PAIR Labs, Param Foundation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了两馆科学中心的8个触摸式展览，通过仅记录会话开始、结束时间和按压计数来推断访客行为。

**💡 创新点**

提出了最小化三字段日志方法，可在不增加硬件的前提下获取可验证的访客互动指标。

**🔧 技术方法**

使用了交互密度（按压/秒）和标准统计检验、逻辑回归及交叉验证，评估展览类型的区分能力。

**📊 数据集**

数据集包含2,816个满足长度≥10秒的访客会话，来自两馆8个明确标注为游戏或测验的展览。

**📈 对比分析**

交叉验证的AUC为0.778±0.023，显示交互密度可区分游戏与测验；按压计数对停留时间的解释方差更高，二者结合进一步提升。

**⚠️ 局限性**

局限包括缺乏访客识别导致会话独立性假设、按压语义差异、样本量不均、少量会话被30分钟上限截断。

---

## 337. Self-supervised Speech Comparison for L2 Phone, Rhythm, and Intonation Scoring

**arXiv ID:** 2607.13721 | [PDF](https://arxiv.org/pdf/2607.13721v1)

**作者:** Stephen McIntosh `[一作]` (University of Tokyo), Herman Kamper `[通讯]` (Stellenbosch University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究提出基于自监督语音表示与动态时间规整的无文本模板比较方法，用于评估第二语言学习者的音位、节奏和语调发音分数。

**💡 创新点**

创新点在于：①首次将DTW与WavLM等自监督模型相结合，对多维度发音进行统一评估；②提出两种节奏度量（tempo irregularity、interval distortion）以及利用SSL残差提取语调特征；③在低资源情境下仅使用少量母语模板即可完成评估。

**🔧 技术方法**

技术手段包括：自监督SSL模型（WavLM、k‑means残差提取）、动态时间规整（DTW）对齐、线性回归映射、Shapley值特征重要性分析、基准方法（PPG、MFCC、F0、强度）。

**📊 数据集**

使用的数据集为：英语读音数据集ERJ（日本学生朗读英文）和日语读音数据集JRF（外籍学生朗读日语），涵盖句子级和词级发音评估子任务。

**📈 对比分析**

通过与人类评标员的交叉验证和Pearson、Spearman、QWK、MAE等指标比较，方法在句子级音位与整体评分任务中往往超过人类一致性；节奏指标表现最接近人类水平；语调评估仍低于人类评分。

**⚠️ 局限性**

局限性在于：①语调评估仍显弱，无法充分捕捉人类评标员的语调判断；②缺乏针对不同语言特征的专门语调特征；③实验仅覆盖两种语言和有限数据量，需进一步验证在更多语言与更大规模数据上的鲁棒性。

---

## 338. Assessing the Forensic Viability of Android Memory Analysis Across Production Builds: A Cross-Version Study of Security Hardening and Structure Preservation

**arXiv ID:** 2607.13821 | [PDF](https://arxiv.org/pdf/2607.13821v1)

**作者:** Jayasimha Nannapanen `[一作]` (Florida Institute of Technology), Sneha Sudhakaran `[通讯]` (Florida Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了从 Android 8 到 Android 15 生产版 ART 二进制剥离后，内存取证工具的可行性及结构保留情况。

**💡 创新点**

量化符号剥离程度并证明即使在完全剥离的生产版中，ART 的内存结构仍保持一致，可借助版本匹配的开发 build DWARF 信息实现实时恢复。

**🔧 技术方法**

使用 ELF 静态解析（readelf、objdump）、AOSP 源码与 git 历史追踪、动态符号比较、Frida 进程注入、GDB 调试以及内存映射对比等技术。

**📊 数据集**

使用 Pixel 7（Android 15）生产映像、Pixel 2 XL（Android 8）工厂映像、ci.android.com 提供的 unstripped CI build（userdebug）以及对应的进程内存快照。

**📈 对比分析**

通过对比符号表条目数、ELF 节信息、动态符号重叠率和内存布局差异来评估；在 Pixel 7 上用 Frida 恢复 Runtime 并验证 DWARF 偏移，确认所有读取均合法，表明结构布局一致；未给出具体时延，但证明方法可行。

**⚠️ 局限性**

仅测试 Pixel 设备，未覆盖其它厂商 ART 变体；只分析了 Android 8 与 15 两个端点，未详细描绘中间进展；验证仅限单版本内部一致性，跨版本偏移可能不适用；未完成完整对象枚举；未评估 kernel 级别取证；仅使用用户空间内存获取。

---

## 339. AspectCLIP: Optimizing CLIP Representation Space via Aspect-Guided Consistency Regularization

**arXiv ID:** 2607.13805 | [PDF](https://arxiv.org/pdf/2607.13805v1)

**作者:** Yiyang Yao `[一作]` (South China University of Technology), Zhihua Jin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AspectCLIP 框架，通过文本聚类分割成属性簇，针对同属性簇内执行全循环一致性，跨簇仅做原型级一致性，从而优化 CLIP 表示空间

**💡 创新点**

创新点在于识别并利用图像-文本描述的一对多信息不对称，通过文本驱动的属性聚类实现差异化一致性正则化，避免全局一致性导致的语义噪声

**🔧 技术方法**

采用 CLIP 双编码器、InfoNCE 对比损失、SimCSE 文本嵌入、K-Means 文本聚类、全循环一致性与原型级一致性正则化

**📊 数据集**

使用 CC3M 数据集进行预训练，并在多种零样本分类、自然分布迁移、线性探测等下游任务上评估

**📈 对比分析**

与 CLIP、LMS、CyCLIP 等方法对比，AspectCLIP 在多项零样本分类、分布迁移与线性探测上均表现提升，Consistency Score 也更高，显示更一致的表示空间

**⚠️ 局限性**

主要局限在于需要预先文本聚类的额外开销，且对属性簇数 K 的选择较为敏感，过大或过小的 λ_2 参数会影响性能

---

## 340. Implementations of Quantum and Classical Topology-Aligned Architectures for Molecular Property Prediction

**arXiv ID:** 2607.13737 | [PDF](https://arxiv.org/pdf/2607.13737v1)

**作者:** James T. Pegg `[一作]` (QunaSys Europe), Ronin Wu `[通讯]` (QunaSys Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了与分子键图形相匹配的拓扑对齐量子与经典网络，用于低参数分子属性预测。

**💡 创新点**

创新在于通过在模型架构中嵌入分子键拓扑，使得量子与经典实现共享相同的参数共享机制，从而实现极低参数（64）且高数据效率的学习。

**🔧 技术方法**

使用变分量子电路（Iso‑QGNN）与匹配参数的经典消息传播网络（Iso‑CGNN），并通过PennyLane等工具实现。

**📊 数据集**

使用QM9基准数据集，对HOMO‑LUMO能隙和偶极矩的二分类任务进行评估。

**📈 对比分析**

在相同参数量、相同数据拆分下，量子和经典模型在测试AUC上相近（0.88‑0.91），且在约250个样本内达到90%峰值，梯度稳定。

**⚠️ 局限性**

局限在于仅测试9原子以内的小分子，缺乏3D几何信息，对偶极矩预测效果受限；并且未探讨更大分子或更深网络的可扩展性。

---

## 341. Online Random Sampling with Real Probabilities

**arXiv ID:** 2607.13828 | [PDF](https://arxiv.org/pdf/2607.13828v1)

**作者:** Thomas L. Draper `[一作]` (Carnegie Mellon University), Feras A. Saad `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

本文提出了一种在线随机采样算法，利用局部随机性回收机制，能够在仅使用 O(log n) 额外熵的情况下，从给定的可计算分布族（可包含无理概率、无限支持）中生成精确的离散随机变量序列。

**💡 创新点**

创新点：
- 引入局部离散随机状态（Z,M）代替传统的全局区间，显著降低所需持久空间从线性压缩到对数级；
- 通过可计算 CDF 或acle 与近似除法相结合，实现对任意实值概率的高效处理；
- 设计了可调节的熵损失参数 ϵ，支持从无熵损失（仅对数级熵）到线性熵损失的连续权衡；
- 在理论上证明了在标准可计算实数模型下，算法达到信息论上最优的熵率与最优空间下界。

**🔧 技术方法**

主要技术：
- 区间搜索（IntervalSearch）与迭代逼近；
- 通过 ApproxDivisionq,k 进行精确除法近似；
- 使用 EstimatePMF 估计点质量；
- 随机状态 (Z,M) 的更新与回收（RecycleF）；
- 复杂度分析结合 CDF oracle 的准确性与索引效率模型。

**📊 数据集**

数据集：无实际数据集，本文以理论分析为主；在实验对比中使用了常见离散分布（如 Poisson、几何、正态离散化等）作为示例。

**📈 对比分析**

对比方法：
- 传统全局区间（Interval Algorithm）
- 批量采样方法（Batching）
- Alias、Inversion、Lookup-table 等经典采样技术
性能评估：
- 熵损失：本算法在 n 次采样后熵损失 ≤ ϵ n + O(log n)，可达到近似 0 的对数级熵损失；
- 空间复杂度：持久空间 O(log n)（或 O(log(1/ϵ))），大幅低于传统线性或 O(1/ϵ) 的空间需求；
- 运行时间：每次采样的期望时间与目标分布的 CDF oracle 复杂度一致，且相较于全局区间方法在常数因子上更优。

**⚠️ 局限性**

局限性：
- 需要访问高精度 CDF oracle，若 oracle 不满足准确性/索引效率要求，性能会下降；
- 设计和实现相对复杂，尤其是随机状态回收与区间估计的细节；
- 对于极大 ϵ（即希望更低熵损失）需要更大的 Δ，可能导致实际运行时间增长；
- 本文未提供大规模实验验证，仅在理论与小规模示例中展示效果。

---

## 342. RainDancer: RGB-Event Video Deraining with Rain-Oriented Spiking Dynamics

**arXiv ID:** 2607.13802 | [PDF](https://arxiv.org/pdf/2607.13802v1)

**作者:** Kui Jiang `[一作]` (Harbin Institute of Technology), Xianming Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于RGB和事件相机的逐步去雨框架RainDancer，采用先分解雨与背景再交互的设计；

**💡 创新点**

创新点在于先对RGB和事件分别进行雨与背景分解，再在语义对齐的组件层面进行交互，并引入雨导向的脉冲神经网络以及事件域监督；

**🔧 技术方法**

采用脉冲神经网络(SNN)、多模态融合、事件域监督、深度残差解码等技术；

**📊 数据集**

在NTURain、RainSynLight25、RainSynComplex25等合成数据集以及实际雨视频集上训练与评测；

**📈 对比分析**

与单模RGB、事件融合以及最新RGB‑Event去雨方法相比，在PSNR/SSIM、无参考指标和下游目标检测/分割任务中均取得了更优的数值与视觉质量；

**⚠️ 局限性**

局限性包括对高速相机运动和稀疏雨景下的鲁棒性仍有提升空间，且模型复杂度相对较高。

---

## 343. Mono-Z Dark Matter Search with Neural Spline Flows Using CMS Run 2015D Open Data

**arXiv ID:** 2607.13771 | [PDF](https://arxiv.org/pdf/2607.13771v1)

**作者:** Hitesh Rasineni `[一作]` (VIT-AP University), Bhavishya Chebrolu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在CMS 2015D开放数据中，利用单Z→ℓℓ（μμ、ee）事件，构建Neural Spline Flow（NSF）背景与三种s通道介子信号的概率密度，并以其对数似然比作为事件级检验统计量，完成暗物质搜索；

**💡 创新点**

首次将NSF似然比评分应用于CMS开放数据的mono‑Z暗物质搜索，并同时兼顾两个轻子通道，实现了无硬阈值的全相空间信号敏感度；

**🔧 技术方法**

使用五个NSF（两条SM、三条DM），在控制区、验证区、信号区进行分区训练；随后在SR+VR的分箱似然比拟合中提取μ上限；

**📊 数据集**

背景数据为2.32 fb⁻¹的CMS Run 2015D MINIAOD双μ/双e数据；信号模拟为MonoZToLL MINIAODSIM生成的向量、轴矢量与标量介子三种模型；

**📈 对比分析**

通过同时SR+VR的分箱轮廓似然拟合得到μ上限，观测上限分别为μ<0.0177（标量）、0.0362（向量）、0.0498（轴矢量），观测值比期望高7–12倍，表明背景尾部建模残差导致上限偏高；

**⚠️ 局限性**

主要限制在高 p_T 尾部背景建模不收敛，导致观测上限偏高；未对MET分辨率与PU重加权系统误差进行评估。

---

## 344. DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention

**arXiv ID:** 2607.13731 | [PDF](https://arxiv.org/pdf/2607.13731v1)

**作者:** Xing Lei `[一作]` (Xi'an Jiaotong University), Donglin Wang `[通讯]` (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多尺度差异感知跨注意力模块（MS-DGCA），将原本只依赖目标的嵌入转化为状态条件的嵌入，并在离线目标条件强化学习中提升导航任务的成功率。

**💡 创新点**

创新点在于：①用可控门控残差保留基线目标表示的充分性；②在跨注意力中加入基于每个令牌状态‑目标差异的可学习非负偏置；③通过多尺度处理适配不同任务的空间尺度。

**🔧 技术方法**

使用技术包括：状态与目标的多头跨注意力、差异映射差值正则化、可学习的门控残差、层归一化、前馈网络与多尺度融合；模型在离线强化学习框架 GCIVL 上训练。

**📊 数据集**

使用 OGBench 数据集，涵盖 13 个基于状态的任务（导航、操控、离散推理）和 7 个视觉任务。

**📈 对比分析**

与原始 Dual、VIB、VIP、TRA、BYOL‑γ 等基线相比，MS‑DGCA 在大多数导航任务中取得最高或第二高的成功率，导航任务的提升最大；在操控和视觉任务中表现不一，某些任务回落，整体平均提升约 15%~20%。

**⚠️ 局限性**

局限性包括：①在操控任务和视觉拼图任务中未能提升甚至下降，原因与门控残差与归一化不兼容；②缺乏面向对象的令牌分解，导致对细粒度差异捕捉不足；③依赖于前置编码器，若编码器已丢失关键信息，后置模块难以弥补。

---

## 345. How Agents Ask for Permission: User Permissions for AI Agents, from Interfaces to Enforcement

**arXiv ID:** 2607.13718 | [PDF](https://arxiv.org/pdf/2607.13718v1)

**作者:** Alexandra E. Michael `[一作]` (University of Washington), Franziska Roesner `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文综述了21篇关于AI代理用户权限系统的研究，构建了统一的分类法，并对五款主流商业代理的权限处理进行了实证分析。

**💡 创新点**

创新点在于提出了面向代理的权限体系全景视角：从用户界面、内部规范、推导机制到运行时执行四个层面，系统地梳理并对比了学术与商业实践的差距与瓶颈。

**🔧 技术方法**

主要技术手段为文献调研与“雪崩式”引用扩展、构建多维度分类法、以及基于用户视角的实验性 Walkthrough 评估。

**📊 数据集**

本文并未使用传统机器学习数据集，而是通过对公开的商业代理（Claude、ChatGPT 等）进行交互式操作获取观察数据。

**📈 对比分析**

比较方法为定性对照：对照文献中提出的目标（低用户开销、形式化规范、确定性执行、持续控制）与商业代理实现情况，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：1）仅覆盖了已公开的21篇工作，未必涵盖全部最新进展；2）商业代理的内部实现未知，评估基于外部观察；3）缺乏量化安全/可用性评测；4）快速演进的AI代理生态可能导致结果随时间失效。

---

## 346. The Test Oracle Problem in Synthetic LLM-as-Judge Corpora: Disappearance, Distortion and a Validation Protocol

**arXiv ID:** 2607.13707 | [PDF](https://arxiv.org/pdf/2607.13707v1)

**作者:** Serkan Ballı `[一作]` `[通讯]` (Mehmet Akif Ersoy University), Serkan Ballı (Mehmet Akif Ersoy University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多语种LLM-as-judge实验中发现并验证了生成步骤共享的解码预算导致的文本截断缺陷，导致错误的跨语言偏差报告；

**💡 创新点**

提出了“测试Oracle问题”视角，证明LLM生成负例的设计缺乏可机械化的项级检查，且给出了可行的手工核查协议；

**🔧 技术方法**

利用LLM生成与机械扰动的对比实验、基于字符串的相等检查、以及统计复现与因果分析方法；

**📊 数据集**

使用wiki40b Turkish语料生成的对照问答对，以及HaluEval英文真伪对照集；

**📈 对比分析**

通过对照A/B实验、样本规模扩展、三层机制解释和生产者交换实验，展示了错误生成导致的32点准确率崩塌，并在修正后恢复为1.00；

**⚠️ 局限性**

局限在于仅展示了单一缺陷与单一实验设置，未覆盖所有可能的生成错误类型与语言，且手工检查仍需人工成本。

---

## 347. Reveal, Correct, Then Pay: Encrypted Mempools and Perpetual Funding Security

**arXiv ID:** 2607.13832 | [PDF](https://arxiv.org/pdf/2607.13832v1)

**作者:** Benjamin Marsh `[一作]` `[通讯]` (Sei Labs), Benjamin Marsh (Sei Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在加密内存池（Encrypted mempool）设计下，攻击者自我授权的状态操纵（self‑authored state manipulation）如何通过反应延迟（reaction gap）和税基资本化渠道放大其收益，并提出在加密与支付之间插入纠正阶段的协议规则。

**💡 创新点**

创新点在于：① 将加密内存池的“commit‑then‑reveal”顺序与后续状态转移拆分，揭示其对自我授权攻击的影响；② 定量化反应延迟与税基资本化对攻击收益的放大效应；③ 设计两阶段纠正与测量的规则，提供隐私与市场完整性的兼容方案。

**🔧 技术方法**

使用理论建模与微观经济分析：离散时间动力学、Poisson 机会到达、指数衰减的纠正速率、反应系数 g(δ;ρ_c,ρ_p,W) 等；同时利用图表和数值示例说明参数影响。

**📊 数据集**

无实验数据集，所有结果均为理论推导与数值示例。

**📈 对比分析**

通过对比公开透明与加密两种内存池场景下的攻击收益公式与安全指数，展示在不同 δ、ρ、m、ζ 参数下加密是否提升或削弱攻击收益；数值图表显示在 δ>0 时，攻击收益可显著放大。

**⚠️ 局限性**

局限性：① 仅考虑单一自我授权攻击情景，忽略多攻击者协作与复杂策略；② 模型假设纠正机会独立且遵循指数衰减，实际市场中可能更复杂；③ 未对不同区块链实现细节（如 BFT、随机排列）做实证验证；④ 缺乏对实际交易量与费用影响的实证数据。

---

## 348. PROBE: Benchmarking Code Generation in Large Language Models

**arXiv ID:** 2607.13820 | [PDF](https://arxiv.org/pdf/2607.13820v1)

**作者:** Rodrigo Pato Nogueira `[一作]` (University of North Carolina at Charlotte), João R. Campos `[通讯]` (University of Coimbra)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 PROBE 框架，用于系统、可扩展地评估大语言模型（LLM）在文本到代码（text‑to‑code）生成任务中的表现。该框架将生成代码的评估分为功能正确性、与参考解的相似度以及代码质量三维度，并在多语言、多难度、多提示策略下对六个公开及私有模型进行大规模实验，并结合错误分析提供模型弱点的洞察。

**💡 创新点**

创新点主要体现在：
• 统一、可复现的多维度评价体系（pass@k、CodeBLEU、Cyclomatic Complexity、NLOC 等）；
• 采用真实竞赛题目构建多语言工作负载，并对单语种、单指标的局限性进行补救；
• 支持多种提示策略（Baseline、ICL、Feedback）并公开完整 Prompt 模板；
• 提供完整的评测流水线、Docker 化部署与公开数据集，方便社区复现与扩展；
• 结合错误分析揭示 LLM 常见失败模式，为后续改进提供实证依据。

**🔧 技术方法**

技术手段包括：
• 大语言模型推理（GPT‑4.1‑mini、Gemini‑2.0‑flash、Qwen2.5 系列、Deepseek‑Coder‑v2）
• Prompt Engineering（Baseline、1‑shot ICL、二轮 Feedback）
• 代码执行评测：单元测试执行、pass@k、Outcome Rate
• 相似度评测：CodeBLEU（结合 n‑gram、AST、数据流）
• 代码质量评测：Cyclomatic Complexity、NLOC
• 自动化流水线：Docker 化的生成器、评测器与协调脚本

**📊 数据集**

数据集：
• IBM CodeNet 竞赛题库（4,053 题，含 55 语言）→ 通过语言筛选、信息提取、错误过滤后得到 1,651 题；
• 单元测试：从 AIZU、AtCoder 提取或使用 Qwen2.5 生成、验证；
• 参考解：每语言至少 3 份正确实现，并剔除超出 3σ 的高复杂度实现；
• 任务难度分级：基于 Python 参考解的静态度量聚类得到 4 个难度层级。

**📈 对比分析**

比较方法：
• 对每个模型、语言、提示策略计算 pass@k（k=1,5）、Outcome Rate、CodeBLEU、Cyclomatic/NLOC；
• 通过多维度打分对模型进行排序；
• 结果显示：大型模型在所有指标上均优于小模型，但 pass@1 最高仍 < 0.7；
• Feedback 机制提升约 0.05 的 pass@k，显著降低编译错误；ICL 效果几乎为零；
• 语言差异显著，Python 最易，Rust 最难；难度越高，所有指标均急剧下降；
• 小模型在低资源语言（如 Rust）表现尤差。

**⚠️ 局限性**

局限性：
• CodeBLEU 不支持 Rust，导致相似度评测缺失；
• 数据集来源为竞赛题，缺少工业级项目的多样性；
• 仅评估功能正确性与结构化质量，未覆盖性能、可维护性等实际开发关注点；
• 参考解数量不足时无法计算相似度与质量指标；
• 评测过程依赖外部 LLM 进行前处理，可能引入语言偏差；
• 结果主要适用于文本到代码任务，对其他 LLM 生成任务的推广有限。

---

## 349. Strong Refutation of Ordering, Phylogenetic, and Ordinary CSPs, and New Satisfiability and Refutation Thresholds for Triplet and Quartet Reconstruction

**arXiv ID:** 2607.13817 | [PDF](https://arxiv.org/pdf/2607.13817v1)

**作者:** Dionysis Arvanitakis `[一作]` (Northwestern University), Konstantin Makarychev `[通讯]` (Northwestern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文研究随机树结构约束满足问题（Triplet Reconstruction 与 Quartet Reconstruction）的相变与强性反证算法。

**💡 创新点**

创新点在于首次给出 Triplet Reconstruction 的精确阈值 λ*≈1.2277，提供了 Quartet Reconstruction 的上下界，并提出了适用于无否定变量的普适强反证框架，实现了 5/9 与 1/3+ε 的性能上限。

**🔧 技术方法**

主要技术包括：构造 Gradual‑Build 变体、利用随机图连通性与大组件阈值、Poisson 过程分裂定理、t‑wise 独立分布分析，以及对树结构的 LCA 与分割不等式推导。

**📊 数据集**

该研究完全基于理论模型，未使用任何具体数据集；所有结果均为渐进概率论和组合优化证明。

**📈 对比分析**

与传统 SAT、Boolean CSP 反证方法相比，本文在树结构上取得了更低的稠密度（如 Triplet m=Ω(n) 反证、Quartet m=Ω(n²) 反证）并实现了更优的 5/9 逼近上限，理论上证明了随机实例在这些稠密度下几乎不可满足。

**⚠️ 局限性**

局限性包括：阈值与反证结果仅在 n→∞ 的渐近意义下成立；对中等规模实例的具体表现未给出；针对非树结构的约束仍未覆盖；以及对实际生物学或数据库中常见的噪声约束的鲁棒性分析缺失。

---

## 350. Auctions with Contract Design

**arXiv ID:** 2607.13795 | [PDF](https://arxiv.org/pdf/2607.13795v1)

**作者:** Xiaolin Bu `[一作]` (Shanghai Jiao Tong University), Zhihua Zhu `[通讯]` (Huawei Ads)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出在拍卖机制中加入质量相关奖励的合同，解决投标者在先期投入成本后因输掉拍卖而产生的道德风险问题，提升平台长期收益与广告质量。

**💡 创新点**

创新点在于将合同理论与传统拍卖理论结合，设计线性质量奖励机制，证明在大型竞标者情形下最佳奖励因子趋于平台对质量的边际收益，实现“全额转移”质量价值；并给出不同拍卖规则下的收入等价性结论。

**🔧 技术方法**

主要技术手段包括：贝叶斯纳什均衡分析、阈值策略构造、对二价与一价拍卖的阈值策略证明、收支平衡式与递推分析、极限（n→∞）下的最优奖励因子求解，以及对不同拍卖规则的收入等价性证明。

**📊 数据集**

本文为理论性研究，无使用具体数据集；所有结果均在假设的连续分布、独立同分布以及线性质量奖励函数下推导。

**📈 对比分析**

通过将拍卖结果与质量奖励合并，作者与传统无奖励拍卖对比，证明在最优奖励设定下拍卖方收入至少不减，且当质量收益高时可获得几乎线性的收入提升。

**⚠️ 局限性**

局限性包括：假设投标者类型独立同分布、只考虑线性奖励、只聚焦对最高价者的奖励、未同时优化保留价与奖励、未考虑相关类型或更复杂信息结构；且最优奖励因子仅在极限大人数情形下解析得到。

---

## 351. CAS I: A Geometric Coding Theorem

**arXiv ID:** 2607.13796 | [PDF](https://arxiv.org/pdf/2607.13796v1)

**作者:** Romie Banerjee `[一作]` `[通讯]`, Romie Banerjee

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在对称群框架下构造了类似经典编码定理的“几何编码定理”，定义了对称先验并证明其为可计算的下半可计算半测度；

**💡 创新点**

创新点在于提出了“可修复对称群”（fix‑retractable group）的概念，证明其满足对称先验可达到与Solomonoff先验相当的条件，并建立了子群与字符串集合之间的格（Galois）关系；

**🔧 技术方法**

主要技术包括可计算对称的枚举与模拟、对称程序与普通程序的双向映射、闭包与密集子群的格论分析，以及半测度的可计算性证明；

**📊 数据集**

本文未使用实验数据集，而是以理论证明为主；

**📈 对比分析**

由于为理论研究，未涉及实验比较或性能评估；

**⚠️ 局限性**

局限性在于仅适用于满足“可修复”条件的对称群，且目前尚未探讨如何在实际计算中高效枚举或实现该框架。

---

## 352. Dynamical Vehicle Orienteering Problem for Multi-Rotor Unmanned Aerial Vehicles

**arXiv ID:** 2607.13789 | [PDF](https://arxiv.org/pdf/2607.13789v1)

**作者:** František Nekovář `[一作]` (Czech Technical University), Robert Pěnička `[通讯]` (Czech Technical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Dynamical Vehicle Orienteering Problem (DVOP)，并在多旋翼无人机上通过轨迹规划与奖励最大化进行求解，给出了基于 Branch‑and‑Bound + MILP/NLP 的精确求解方法以及 Large Neighborhood Search (LNS) 的启发式求解框架，并在真实无人机上验证了所规划的轨迹。

**💡 创新点**

创新点在于将传统 Orienteering Problem 与二阶点质量模型（PMM）结合，首次针对多旋翼动力学设计了包含重力与推力约束的轨迹原语；并提出了利用这些原语构造的 MILP 松弛来计算奖励上界，以及基于 Limited Thrust Decomposition 的连续速度优化的 LNS 元启发式。

**🔧 技术方法**

采用的技术包括：非线性规划 (NLP) 求解时间最优轨迹、基于轨迹原语的混合整数线性规划 (MILP) 上界计算、Branch‑and‑Bound 搜索框架、Large Neighborhood Search (LNS) 与局部搜索、梯度速度优化、点质量三维模型与离散时间步长的离散化。

**📊 数据集**

使用的数据集包括：经典 KOP 基准（Tsiligirides Set 2、Chao Set 64），20 个随机 3D 目标集（目标数 15，奖励统一为 1），以及在 330 旋翼无人机上采集的实测轨迹与目标位置数据。

**📈 对比分析**

与现有 KOP 求解器（KOP‑1、KOP‑6、VT‑MPC）以及多旋翼特定的 LNS、BnB 方案进行对比；实验表明 LNS 在几秒内即可获得与最优相当或更优的奖励，BnB 在数十小时内实现全局最优；在随机实例上，LNS 与 BnB 的平均误差低于 18%，且相较于基准提升可达 37%。

**⚠️ 局限性**

主要限制包括：B&B 求解时间仍高昂，随着目标数和预算增大搜索树规模呈指数增长；MILP 松弛与奖励上界仍存在一定误差；点质量模型未考虑姿态、旋转约束以及风等外部扰动，导致在极端飞行条件下轨迹跟踪误差上升。

---

## 353. Regularity as seen by Alice and Bob

**arXiv ID:** 2607.13782 | [PDF](https://arxiv.org/pdf/2607.13782v1)

**作者:** Omid Yaghoubi `[一作]`, Rafał Stefański `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种统一的两方通信协议模型，用以定义不同输出域（布尔、域、字符串以及无限字母表）下的正规性，并证明其与已知自动机模型等价（布尔→正规语言、域→加权自动机、字符串→正则函数（猜想）及无限字母表→无歧义寄存器自动机（猜想））

**💡 创新点**

创新点在于将通信复杂性与黑盒约束相结合，构造可在常数消息数下实现的协议，随后将其与传统自动机理论（Myhill‑Nerode、Fliess 定理、轨道有限集合与向量空间）统一映射，从而在非统一计算模型下获得可判定的正规性表征

**🔧 技术方法**

核心技术包括：基于Hauser的通信复杂性框架；黑盒限制避免信息泄露；将协议归约为单轮/无信号形式；使用Myhill‑Nerode和Fliess 定理得到有限秩/有限状态判定；轨道有限集合与轨道有限向量空间用于无限字母表的处理

**📊 数据集**

本文为理论工作，无使用实验数据集

**📈 对比分析**

通过理论证明和结构化归约，展示协议模型与各自动机模型在表达能力上完全等价；在所有已证明的情形下，协议实现线性时间与线性输出大小的计算，证明了其高效性

**⚠️ 局限性**

局限性包括：域输出模型仅支持加法与乘法，除法导致更强模型但未被覆盖；一般半环下协议与加权自动机不等价；字符串输出的正则性等价性仍为猜想；无限字母表情形无法归约为单轮，需多轮交互，相关证明尚未完成；部分假设（如无歧义性）缺乏完整可判定性证明

---

## 354. ZipLine: Visual Analysis of Multivariate Graphs with Predicate Logic

**arXiv ID:** 2607.13767 | [PDF](https://arxiv.org/pdf/2607.13767v1)

**作者:** Sjoerd Vink `[一作]` (Utrecht University), Remco Chang `[通讯]` (Tufts University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种集成多属性图可视化与形式化推理的系统与方法，允许分析师在拓扑空间和属性空间之间交互构造并自动诱导谓词；

**💡 创新点**

创新点在于设计了跨拓扑与属性的谓词语言、基于束搜索的谓词诱导算法，并将其嵌入可视化工作流，实现了从直观选择到逻辑表达式的闭环；

**🔧 技术方法**

核心技术包括：谓词语言（支持属性、拓扑、邻域与计数约束）、束搜索与分支裁剪的谓词诱导算法、可视化谓词构造器以及交互式多视图布局；

**📊 数据集**

实验使用三类真实数据集：TenneT 电网（5,279 节点）、MITRE ATT&CK（1,742 节点）和 PrimeKG 药物再利用子图（1,538 节点）；

**📈 对比分析**

通过专家评审与案例分析验证，诱导出的谓词在覆盖率与精度上均优于直观筛选，诱导耗时 1–3 秒，实验未做严格量化对比，主要以案例与专家满意度呈现；

**⚠️ 局限性**

主要局限包括：拓扑指标（如中心性、聚类系数）可解释性不足；缺乏时间序列、因果或依赖建模；谓词表达式组合多时解释难度增大；系统对大规模或动态图的扩展尚待完善。

---

## 355. Algebraic Representability as the Limiting Regime of Grokking: An Exactly Solvable Model with Holomorphic Activations

**arXiv ID:** 2607.13749 | [PDF](https://arxiv.org/pdf/2607.13749v1)

**作者:** Chon-Fai Kam `[一作]` (University Paris City & University of Reunion), Frederic Cadet `[通讯]` (University Paris City & University of Reunion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造并分析了一类两层复数神经网络（激活函数为σ(z)=z^k），在根号‑1编码的模算术任务上研究其可表达性与训练行为。

**💡 创新点**

首次给出了任务在此网络中可被表示的完全代数条件：任务的离散傅里叶支撑必须位于有限集合S_k；证明了不可表示任务的训练损失下界与网络宽度无关，从而解释了无grokking现象；并将该极限容量-grokking谱与传统容量框架桥接。

**🔧 技术方法**

利用解析展开、傅里叶分析、有限群表示、范德蒙德矩阵、投影理论、Hoeffding‑Serfling不等式等数学工具，对可表达子空间进行完全刻画并推导训练损失下界。

**📊 数据集**

采用p=97的ℤ_p×ℤ_p输入空间，设计了39个模运算目标（35个线性相位，4个非线性相位），在每个任务-激活度组合上进行3次随机种子实验，总计585次训练。

**📈 对比分析**

与标准ReLU网络进行对比：ReLU网络可记忆并表现出grokking；Holomorphic网络仅出现inst或fail两种极端结果；通过瓶颈实验展示从fail→mem→grok的三阶连续谱。实验结果与理论预言吻合率高达99.8%。

**⚠️ 局限性**

局限性包括：仅针对特定两层holomorphic网络和根号‑1编码，难以推广至更深网络、其他激活或非群任务；对编码方式的依赖导致结果不一定适用于一般输入编码；未对梯度下降动态的细节进行完整分析。

---

## 356. Anatomically Faithful but Temporally Blind: Auditing Attribution for Left-Ventricular Ejection-Fraction Estimation from Echocardiography

**arXiv ID:** 2607.13738 | [PDF](https://arxiv.org/pdf/2607.13738v1)

**作者:** Hyunkyung Han `[一作]` (Yonsei University), Min Jung Kim `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究对心脏超声视频中左心室射血分数（LVEF）估计模型的可解释性进行系统审计，分别评估其空间归属、因果删除和时间定位的可靠性。

**💡 创新点**

创新点在于提出三轴可解释性评估协议，并首次发现尽管模型在空间上高度可信，但在时间上却完全失效，揭示了空间与时间可信性之间的分离。

**🔧 技术方法**

使用了自监督VideoMAE Transformer与Kinetics预训练的R(2+1)D CNN，分别采用Chefer relevance和Grad‑CAM进行归因，并通过IoR、删除AUC和时间定位指数等指标进行量化评估。

**📊 数据集**

实验基于公开的EchoNet‑Dynamic数据集，包含数千条四腔视图的超声视频以及对应的EF标签、ES/ED帧与LV分割。

**📈 对比分析**

通过与随机归因、注意力展开等基线对比，发现空间IoR分别为2.91×和1.98×的显著提升，但时间定位指数均约为1（与随机相当），删除AUC仅略高于随机，说明模型在空间上可信而在时间上盲目。

**⚠️ 局限性**

局限性包括仅在单一数据集上评估、所用模型未达到目前最佳精度、未探索进一步训练或多帧聚合是否能恢复时间依赖性，以及空间分辨率可能影响时间评估结果。

---

## 357. Decomposable Type Highlighting for Bidirectional Type and Cast System

**arXiv ID:** 2607.13727 | [PDF](https://arxiv.org/pdf/2607.13727v1)

**作者:** Max Carroll `[一作]` (University of Cambridge), Patrick Ferris `[通讯]` (University of Cambridge)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了可分解的高亮系统，结合类型切片和 cast 切片，用于交互式解释静态与动态类型推导过程，提升类型错误调试。

**💡 创新点**

创新点在于定义数学基础的可分解高亮、bidirectional 类型系统与 cast 传播的组合，以及在 Hazel 语言中实现交互式 UI。

**🔧 技术方法**

采用了 bidirectional 类型系统、渐进式类型化、cast 计算机以及 Hazel 语言的核心 lambda 计算机与 holes 机制。

**📊 数据集**

未使用传统数据集，演示基于 Hazel 示例程序；论文中仅提供示例与 prototype。

**📈 对比分析**

论文未给出实验评估，比较仅通过示例演示；未报告性能指标，关注理论与实现。

**⚠️ 局限性**

局限在于尚未覆盖多态与递归类型，缺乏用户研究与实验验证，且 cast 切片在复杂语义下仍有实现难点。

---

## 358. CAVA: Canonical Action Verification and Attestation for Runtime Governance of Agentic AI Systems

**arXiv ID:** 2607.13716 | [PDF](https://arxiv.org/pdf/2607.13716v1)

**作者:** Zexun Wang `[一作]` `[通讯]` (Ond Holdings Inc.), Zexun Wang (Ond Holdings Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CAVA 框架，将异构代理运行时行为统一规范为可哈希、可验证的“Canonical Action”对象，以实现跨运行时的治理一致性

**💡 创新点**

1) 定义可版本化的 Canonical Action 语义模型；2) 设计 CAVA 协议（捕获、归一化、模式识别、指纹、绑定、闭合、可选鉴定）；3) 引入可组合的 Semantic Pattern 层，将行为映射为可复用的治理模式；4) 与 PCAA 对接，提供完整的部署方治理链

**🔧 技术方法**

确定性哈希（SHA‑256）、规范化解析器（支持 shell、SDK、MCP、浏览器、管理代理等）、模式检测器、可选的签名/VC/链码鉴定子系统

**📊 数据集**

公开的 96 个种子情景扩展为 384 个运行时变体（shell、MCP、浏览器、管理代理）

**📈 对比分析**

与 Raw‑Text 与 First‑Token 两个基线对照；在 9 项评估指标（语义等价、分离、包装绕过、误报控制、批准绑定、收据可重现、鉴定抗篡改、运行时可移植、模式检测）上，CAVA 取得 1.0 的完美分数，基线几乎为 0

**⚠️ 局限性**

仅覆盖有限的公开基准，缺乏完整企业运行时样本；归一化依赖于解析器覆盖率；鉴定仅提供完整性，无法保证业务正确性；在仅观察型运行时下只能披露而非强制执行

---

## 359. Groc-PO: Grounded Context Preference Optimization for Truthful Multimodal LLMs

**arXiv ID:** 2607.13712 | [PDF](https://arxiv.org/pdf/2607.13712v1)

**作者:** Zhixiao Zheng `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Grounded Context Preference Optimization (Groc-PO) 框架，通过在多阶段视觉-语言推理中引入显式偏好监督，提升 MLLM 的真诚度和推理可靠性。

**💡 创新点**

创新点在于构造三阶段 Grounded Context Preference Dataset (GCPD)，并设计双重加权的自适应损失（阶段重要性权重和难度权重），实现对早期 grounding 阶段的针对性优化，显著抑制错误传播。

**🔧 技术方法**

使用了 DPO 基础对齐方法，结合 LoRA、AdamW 训练策略，和 GPT‑4o 教师模型进行生成与校正；还引入了模型中心化采样和人类审核验证。

**📊 数据集**

使用了 5,733 张图像的 GCPD 数据集（包含 17,199 条偏好对），以及原始 RLHF‑V 数据集作为辅助。

**📈 对比分析**

通过与 DPO、RLHF‑V、V‑DPO、CSR、POVID 等基线在 AMBER、MM‑Hal、LLaVA‑Bench、SEED 等多项真实性与通用能力指标上进行比较，Groc‑PO 在幻觉率、准确率和复杂推理任务上均实现 30%–45% 的提升。

**⚠️ 局限性**

局限性包括：需要人工和教师模型的多阶段数据生成，构造成本较高；仍受限于基础模型的推理深度，难以处理极端多轮对话或跨模态推理中的复杂依赖；以及在某些简单任务上加权机制可能带来轻微的训练开销。

---

## 360. Recursive ArUco Markers: A Scalable Fiducial Marker Design for Unmanned Aerial Vehicle Landing Pads

**arXiv ID:** 2607.13830 | [PDF](https://arxiv.org/pdf/2607.13830v1)

**作者:** Rafael Munoz-Salinas `[一作]` (Universidad de Cordoba), Sergio Garrido-Jurado `[通讯]` (Universidad de Cordoba)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种可递归嵌入完整ArUco标记、无须中心可见的多尺度视觉标记，用于无人机精确着陆。

**💡 创新点**

在黑白像素两种颜色都嵌入完整子标记，采用仅采样边缘的位值提取方式，保持统一ID并实现任意递归深度，显著提升对遮挡、视角、模糊的鲁棒性。

**🔧 技术方法**

递归嵌入算法、边缘采样检测、基于ArUco Nano的高效阈值与轮廓追踪、对反转标记的支持、随机角点细化等技术。

**📊 数据集**

使用公开的DICT_APRILTAG_16h5字典，生成4K模拟图（1m×1m标记）进行评估，并在真实飞行中使用DJI Mini 4 Pro拍摄30cm×30cm标记的数据。

**📈 对比分析**

与OpenCV官方ArUco检测和ArUco库中的Fractal marker做基准，对比TPR、检测距离、视角、遮挡、裁剪、运动模糊等指标；结果显示RArUco在极端视角（80°）、30%遮挡、60%裁剪等条件下保持100%检测，检测距离约63m，处理速度约185FPS，显著优于Fractal marker。

**⚠️ 局限性**

仍受摄像机分辨率限制，极高距离（>14m）检测下降；未在所有递归深度下系统验证，递归深度越大标记尺寸越小，可能影响可视距离；实验仅基于公开字典和单一飞行平台。

---

## 361. A Deployed Hybrid Vehicle-in-the-Loop Platform for Validating Cooperative Perception

**arXiv ID:** 2607.13806 | [PDF](https://arxiv.org/pdf/2607.13806v1)

**作者:** Anastasia Bolovinou `[一作]` (Institute of Communication and Computer Systems), Angelos Amditis `[通讯]` (Institute of Communication and Computer Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并部署了一个混合车辆‑环路（ViL）平台，将真实配备传感器的车辆与 CARLA 数字孪生通过 ETSI CAM/CPM V2X 消息管道耦合，实现实时协同感知的概率占用网格生成，并在公路类测试轨道上完成演示。

**💡 创新点**

创新点包括：① 在真实‑虚拟混合环境中实时同步 CAM/CPM 并注入数字孪生；② 采用 GPU 加速的协同感知占用网格模块；③ 在雨、夜、不同定位噪声等多条件下评估协同感知工作负载并量化其性能；④ 规划地中海 ODD 场景库，推动混合 V2X‑环路测试服务。

**🔧 技术方法**

使用技术包括：CARLA 模拟器 + OpenDRIVE/OSM 迁移；ETS I CAM/CPM V2X 协议；ROS/ROS2 桥接；NVIDIA CUDA + JAX；ns‑3 + OpenCDA 的 VaN3Twin；GNSS/IMU、Velodyne VLP‑16 LiDAR、RGB 摄像头；NUC 边缘 PC 与 Nfiniity Cube V2X OBU；自定义 3D 车辆与环境建模；GPU 加速的占用网格生成。

**📊 数据集**

使用的数据集包括：现场测试轨道的实测传感器与 V2X 消息流（每 100 ms CAM/CPM）；OpenStreetMap 导出的道路网络；自定义车辆与场景 3D 模型；D‑GPS 轨迹作为定位真值；以及不同天气（雨、夜）和定位噪声水平下的仿真数据。

**📈 对比分析**

通过在数字孪生中控制定位噪声、天气条件和协作配置，对占用网格的视野覆盖率、占用单元召回率、精度等 KPI 进行面积‑下‑曲线（AUC）评估。实验显示：协同显著扩大 FoV 覆盖率并提升召回率，定位误差在 1–2 m 以上成为主要误差源；GPU 实现相较 CPU 在吞吐量与延迟上有显著提升。

**⚠️ 局限性**

限制与挑战包括：① 采样周期固定但未保证严格确定性，导致实时对齐不稳定；② 循环率与抖动需提升以实现跨运行可重复；③ 依赖 CARLA co‑simulation 导致实时性能受限；④ 尚未完成完整的实测‑仿真精度（sim‑to‑real）量化评估；⑤ 需要进一步扩展远程操作与沉浸式驾驶员界面。

---

## 362. Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning

**arXiv ID:** 2607.13818 | [PDF](https://arxiv.org/pdf/2607.13818v1)

**作者:** Xiaopeng Zhang `[一作]` (Harbin Institute of Technology Shenzhen), Yanjie Li `[通讯]` (Harbin Institute of Technology Shenzhen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于高层agentic强化学习的执行管理框架，利用低层冻结的视觉‑语言‑动作（VLA）策略在运行时根据两种执行质量指标动态选择 Execute/Retry/Repair/Reset 等恢复模式，以提升机器人操控的稳健性。

**💡 创新点**

创新点在于：①提出局部与全局两种执行质量度量；②通过高层agentic策略学习仅基于执行历史的决策，从而实现执行级别的自适应恢复，完全分离动作生成与执行管理；③构建了三种恢复机制并与低层策略无缝协作。

**🔧 技术方法**

主要技术包括滑动窗口质量评估、基于最近历史的特征提取、PPO强化学习训练高层策略、以及回溯式（Retry/Repair）和重置（Reset）恢复操作。

**📊 数据集**

使用了 LIBERO 基准数据集，涵盖 LIBERO‑Spatial、LIBERO‑Object、LIBERO‑Goal、LIBERO‑Long 四个子集。

**📈 对比分析**

与 OpenVLA、π₀、π₀.₅、Diffusion Policy 等基线在标准和受扰动环境下进行对比，平均成功率在标准设置提升 13.7%，在扰动设置提升 39.2%，表现显著优于基线。

**⚠️ 局限性**

局限性在于对极端执行失稳或离谱场景的恢复能力有限，且高层策略仅依赖低层执行历史，难以应对完全离散的未知障碍或超出训练分布的情况。

---

## 363. Traffic-Aware Randomized Smoothing for LLM-Based Network Intrusion Detection

**arXiv ID:** 2607.13801 | [PDF](https://arxiv.org/pdf/2607.13801v1)

**作者:** Zhenpeng Li `[一作]` `[通讯]` (Guangzhou Health Science), Zhenpeng Li (Guangzhou Health Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Traffic-Aware Randomized Smoothing (TA‑RS)，一种针对LLM驱动网络入侵检测系统（IDS）的可证书鲁棒防御，专门在攻击者可直接操控的流量特征子空间（DC）内注入高斯噪声进行训练与证书，解决传统随机平滑对不可控特征的误导与模型噪声不稳定问题。

**💡 创新点**

创新点：
1) 根据攻击者可控特征划分的DC/IC/UC 体系，精确将噪声仅投放到DC子空间，避免无关特征噪声导致的“自证”与过度保守。
2) 证明模型在该子空间噪声下的稳定性是获得可靠证书的关键，提出噪声增强微调（noise‑augmented fine‑tuning）以提升稳定性。
3) 在大规模LLM（Qwen3‑8B、LLaMA3‑8B）IDS上首次实现可证书防御，并在三大流量数据集上系统评估，验证理论与实践的匹配。

**🔧 技术方法**

技术细节：
- 随机平滑（Randomized Smoothing）结合高斯噪声。
- 仅在DC子空间注入噪声（σ_DC=σ，σ_IC/UC=0）。
- 通过LoRA微调并加入噪声副本（n_aug=2 或 4）实现噪声稳定。
- 使用Cohen等人提出的概率下界与Clopper‑Pearson 置信下限计算证书半径。
- Monte Carlo 采样（N=200）估计概率，N=1000 进行稳健性检验。

**📊 数据集**

数据集：CIC‑IDS‑2018（四类，14 个 DC 特征）、HIKARI‑2021（三类，25 个 DC 特征）以及 RT‑IoT2022（五类，27 个 DC 特征）共计三大公开网络流量安全基准。

**📈 对比分析**

对比方法：
- 无防御（clean‑trained）
- 全域随机平滑（isotropic RS）
- 对齐噪声训练的全域 RS（iso‑trained）
- 随机子空间 RS
- 以及对抗训练与其他经验防御。
性能表现：在 CIC‑IDS‑2018 与 HIKARI‑2021 上，TA‑RS 在 σ=0.25 时可实现 55–100% 的 certified accuracy（CA），覆盖 L∞ 等价预算 ε=0.05 的样本比例 55–100%。
对齐噪声在同一 DC‑训练模型上比全域 RS 提升 4–72 个百分点；在 RT‑IoT2022 默认配置下 CA≤10%，但通过 n_aug=4 增强可恢复至 70%+。相比同类基线，TA‑RS 在大多数场景下显著降低 abstention 并提升 CA。

**⚠️ 局限性**

局限性：
1) 必须针对每个数据集单独进行噪声增强微调，缺乏通用迁移性。
2) 对 LLМ 的噪声稳定性依赖强，RT‑IoT 这类复杂数据集仍需更大噪声预算。
3) 仅给出 L2 证书，需通过 √d 缩放与 L∞ 攻击预算对齐，仍有不一致性。
4) Monte Carlo 采样（N=200）导致证书保守，N 增大会提升计算成本。
5) 当前实现需要 200 次前向推理，难以满足实时部署，需进一步加速或压缩证书。

---

## 364. EgoProceVQA: A Novel Egocentric Procedural Understanding Task with Self-Skill-Exploration Agent

**arXiv ID:** 2607.13792 | [PDF](https://arxiv.org/pdf/2607.13792v1)

**作者:** Junlong Li `[一作]` (Hong Kong Polytechnic University), Yi Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Egocentric Procedural Understanding VQA（EgoProceVQA）任务并构建3,600问答样本；

**💡 创新点**

创新点在于：①基于关键步骤的多维评估框架；②EgoProceGen自动生成平台；③自我技能探索的EgoProceAgent；

**🔧 技术方法**

采用LLM规划+视频LLM执行、工具库（Grounding DINO、CLIP、WikiHow搜索、时序分割）与子技能解耦推理；

**📊 数据集**

使用四类自我视角手工流程视频（CaptainCook4D、EPIC‑Tent、Assembly101、EgoOops），合并生成Benchmark；

**📈 对比分析**

与多款公开与闭源MLLM/视频LLM对比，闭源模型最高、开放源模型表现仍落后；EgoProceAgent在所有关键步骤任务上相对基线提升20%~80%，实现开放源SOTA；

**⚠️ 局限性**

局限性包括：仍距人类水平，时序定位能力不足，对工具依赖受限，未实现完全无监督训练且受原始模型推理能力上限限制。

---

## 365. Design of policy digital twins incorporating multi-level agent based modelling

**arXiv ID:** 2607.13766 | [PDF](https://arxiv.org/pdf/2607.13766v1)

**作者:** Matt Tipuric `[一作]` (Alan Turing Institute), David Wagg `[通讯]` (Alan Turing Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发并演示了一个集成多层代理模型（MABM）的政策数字孪生平台，用于英国纽卡斯尔市政府制定能源转型（热泵推广）政策。

**💡 创新点**

创新点在于：① 将多层代理模型嵌入数字孪生框架，使人类行为与物理系统在同一模型中并行演化；② 采用基于敏捷方法的迭代式设计流程，使政策制定者在整个开发周期中持续参与；③ 通过多尺度（单户、LSOA、全市）输出与可视化，提升政策评估的可操作性。

**🔧 技术方法**

技术主要包括：Python（Flask、Mesa、Mesa-Geo、Pandas、GeoPandas、Plotly、Maplibre-GL），SQLite 数据库，ERA5 气候数据集，DESTINATION Earth 平台接口，以及基于 Mesa 的 MABM。

**📊 数据集**

数据集涵盖：英国 2021 年人口普查（用于构建合成人口），EPC（能源性能证书）登记数据（建筑属性），ERA5 气候数据（温度驱动），SERL 智能计量样本（参数校准），以及 DESNZ 的子国家住宅消耗统计（独立验证）。

**📈 对比分析**

比较方法：模型在单户、邻域和全市尺度上分别与 SERL、DESNZ 统计数据进行对比；使用相关系数、偏差（Bias）和平均绝对误差（MAPE）衡量性能；结果显示在所有尺度上相关系数≥0.90，整体偏差约为-12%至-18%，校正后 MAPE≤7%。

**⚠️ 局限性**

局限性包括：① 缺乏实时数据流，无法实现真正的动态自适应；② 合成人口与真实人群的对应不直接，限制了数据同化与动态再校准；③ 只针对纽卡斯尔市的局部场景，缺乏跨城市、跨政策的可迁移性验证；④ 对复杂多模型交互的鲁棒性与验证仍待进一步研究。

---

## 366. S-CARD-CMSA: A Score-Aware Candidate Archive with Density-Filtered Reporting for Multimodal Optimization

**arXiv ID:** 2607.13764 | [PDF](https://arxiv.org/pdf/2607.13764v1)

**作者:** Dikshit Chauhan `[一作]` `[通讯]` (National University of Singapore), Dikshit Chauhan (National University of Singapore)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RS-CMSA-ESII基础上，提出S-CARD-CMSA，通过添加被动次级候选档案与基于得分的密度过滤报告规则，以改进多峰优化的候选保留与最终报告。

**💡 创新点**

创新点在于：① 只在搜索后期不干扰核心搜索过程，利用次级档案恢复被主档案遗漏的restart‑best候选；② 采用阈值窗口与归一化距离过滤器，精准控制报告数量，平衡RPR与F1得分；③ 统一采用单一固定系数的密度阈值，简化调参。

**🔧 技术方法**

核心技术：RS-CMSA-ESII（协方差矩阵自适应演化策略 + 反弹子群），被动次级档案存储，score‑aware 价值窗口筛选，归一化欧氏距离密度过滤，hill‑valley 基础验证。

**📊 数据集**

使用IEEE CEC 2026多峰优化竞赛基准，包含16个问题标识、每题15个实例、4个维度，共960个测试案例，评估预算为20,000D。

**📈 对比分析**

与原RS-CMSA-ESII基线对比，S-CARD-CMSA在RPR保持不变的情况下提升精度与F1得分，平均得分从0.6012提升到0.6049（DF‑SCA变体），在更广泛的768跑验证中，平均精度从0.9071提升至0.9148，平均得分从13.798提升至13.589。

**⚠️ 局限性**

局限：仅改进候选保留与报告，不增强搜索发现能力；密度阈值为固定全局值，可能在某些景观下不最优；对极大/极小维度的适应性有限；未利用全局最优信息进行自适应调整。

---

## 367. Precoding-based protocols for entanglement assisted linear computation over a quantum many-to-one network

**arXiv ID:** 2607.13756 | [PDF](https://arxiv.org/pdf/2607.13756v1)

**作者:** Ruoyu Meng `[一作]` (Iowa State University), Aditya Ramamoorthy `[通讯]` (Iowa State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一系列预编码基线协议，实现了在无噪声多对一量子网络中多发送方协同计算线性组合，并揭示了量子方案在通信成本上的可乘法优势。

**💡 创新点**

创新点在于：①利用分布式预编码与多实例联合编码，使得原本不满足 N‑Sum Box 自正交条件的线性变换也能被高效实现；②给出两类可行约束（Interval‑[0,1] 与 Restricted Interval‑[0,1]）下的显式协议并证明其在某些参数区间内优于现有最佳方案；③首次证明量子多对一通信成本的严格亚加法性，即联合计算两个不同线性函数可低于分别计算两者之和。

**🔧 技术方法**

核心技术包括：多实例量子预编码、(κ,N)-Sum Box 协议、对自正交矩阵的构造、矩阵秩与交错矩阵的利用以及多重预编码参数优化。

**📊 数据集**

该工作为理论分析，未使用具体实验数据集；所有结果均基于有限域符号计算与量子态构造。

**📈 对比分析**

通过与经典无共享量子资源方案和已有的 Hu 等人方案（基于最优预编码）的对比，证明在 Interval‑[0,1] 与 Restricted Interval‑[0,1] 条件下，提出的协议可将通信成本降低至比经典上限至少 1/3 或更高，并在示例实例中实现了量子方案的严格成本优势。

**⚠️ 局限性**

局限性包括：①预编码问题的求解在一般情况下仍为 NP‑hard，缺乏多发送方的高效算法；②协议的最优性仅在特定结构（如 Double‑Basis 或满足 Condition <ref> 的对）下已证明，其他一般实例尚未给出完整下界；③对 SSO 条件的消除依赖于多实例与额外子系统，导致协议实现复杂度上升。

---

## 368. The Nonsmooth Impact Direction (NSID) of Robotic Systems

**arXiv ID:** 2607.13768 | [PDF](https://arxiv.org/pdf/2607.13768v1)

**作者:** Annika Kirner `[一作]` (TU Wien), Christian Ott `[通讯]` (TU Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统地分析了刚体机器人与刚性环境碰撞时的非光滑冲击方向（NSID），并通过实验验证其独立于接触弹性、可用于规划和控制的特征方向。

**💡 创新点**

首次提出并证明NSID是碰撞的特征方向，且与接触弹性、摩擦无关，进一步推广至柔性关节与受限机器人系统，并与传统冲击模型对比。

**🔧 技术方法**

使用投影法推导冲击模型、牛顿恢复律与摩擦摩擦模型，实验中结合运动捕捉、力传感器与动力学方程实现对冲击前后速度与冲量的测量与分析。

**📊 数据集**

实验数据来自3D打印多关节机械臂的运动捕捉记录以及Franka FR3机器人在不同工具与接触表面（铝、芯板）下的力/位移测量。

**📈 对比分析**

与不考虑NSID的传统方法相比，NSID实现了冲击后路径对齐、避免滑移的成功率接近100%，实验表明冲击时动量仅沿接触法向传递，验证了理论预期。

**⚠️ 局限性**

主要局限在于对瞬时冲击、无摩擦或完全不弹性假设的依赖，且在高速度或非刚性关节机器人的实际场景中尚未完全验证。

---

## 369. Multimodal Assessment of Pancreatic Cancer Resectability Using Deep Learning

**arXiv ID:** 2607.13826 | [PDF](https://arxiv.org/pdf/2607.13826v1)

**作者:** Vincent Ochs `[一作]` (University of Basel), Sebastian Staubli `[通讯]` (Clarunis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种端到端的多模态框架，自动将3D CT影像与结构化临床数据融合，用于判定胰腺导管腺癌的NCCN切除性。

**💡 创新点**

创新点在于使用Swin-UNETR进行解剖导向的辅助分割监督，并通过自适应多任务损失动态平衡分割与分类，最终实现不需分割掩膜即可做切除性预测。

**🔧 技术方法**

主要技术包括Swin-UNETR骨干、CT与临床变量的MLP融合、动态权重的多任务损失以及KNN填补等预处理。

**📊 数据集**

使用来自两家巴塞尔医院的159例PDAC病人作为内部数据集，并在Kantonsspital Aarau的52例病人上进行外部验证。

**📈 对比分析**

与传统分割+几何特征方法和改编的TAT方法相比，本模型在5折交叉验证中AUC 0.86、macro‑F1 0.79、准确率0.85，外部验证同样达到AUC 0.86、macro‑F1 0.81、准确率0.87，表现优异。

**⚠️ 局限性**

局限包括样本量有限、分割标签仅在训练时使用、部分临床变量需填补、缺乏前瞻性多读者研究以及对更大规模数据的进一步验证需求。

---

## 370. The Replication Assessment Problem in Software Engineering

**arXiv ID:** 2607.13815 | [PDF](https://arxiv.org/pdf/2607.13815v1)

**作者:** Giuseppe Destefanis `[一作]` (University College London), Leila Yousefi `[通讯]` (Ministry of Justice)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2021‑2025 年软件工程复制研究中评估复制结果的方法进行系统综述，提取并编码 13 种评估技术，分析其使用频率与问题，并提出四项预先规范复制评估的原则。

**💡 创新点**

首次系统揭示复制结果评估的异质性与不透明性，指出专家判断、聚合与缺乏效果兼容性方法等问题，并提供可操作的原则框架以统一复制结论的判定。

**🔧 技术方法**

使用系统文献检索、研究筛选、编码表与定量统计（方法出现频率、类别聚类）以及基于文献的原则推导。

**📊 数据集**

检索自 SCOPUS 的 62 篇论文，其中 10 篇符合纳入标准，构成复制评估方法的样本集。

**📈 对比分析**

通过对 10 篇论文中 13 种评估方法的出现频率和使用组合进行频数统计与聚类，发现专家判断和聚合最常见，效果兼容性方法完全缺失；该分析揭示了评估标准的不一致性，并为原则制定提供证据，但未给出定量性能指标。

**⚠️ 局限性**

样本量有限（仅 10 篇），仅检索 SCOPUS 数据库，且所有编码由单一研究者完成，可能导致编码偏差，限制了结果的外部与内部有效性。

---

## 371. Bake It Till You Make It: Ultrafast Spatial Texture-Atlas Splatting

**arXiv ID:** 2607.13808 | [PDF](https://arxiv.org/pdf/2607.13808v1)

**作者:** Neel Kelkar `[一作]` (Technical University of Munich), Rüdiger Westermann `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于二维表面元（surfels）的新视图合成方法，将高频纹理与几何表面分离，并通过哈希网格预烘培到纹理图集中，实现实时4K 60FPS渲染。

**💡 创新点**

创新点在于：① 将视角无关的高频纹理细节从每个surfels中抽离出来，用空间多分辨率哈希网格+MLP学习；② 将学习到的纹理残差烘培为BC7压缩纹理图集，消除神经查询开销；③ 采用可塑的Beta核与落差正则化，压缩surfels的空间影响；④ 通过多视角误差优化与定期冻结策略，加速稀疏化与训练。

**🔧 技术方法**

主要技术包括：2D Beta surfels、球面Voronoi视角相关颜色、空间哈希网格+MLP视角无关纹理、纹理烘培与BC7压缩、落差减弱正则化、FastGS稀疏化、基于GPU纹理采样的渲染管线。

**📊 数据集**

使用了MIP-NeRF 360数据集（多视角合成）以及Tanks & Temples数据集（室外场景）进行评估。

**📈 对比分析**

与3DGS、2DGS、Beta‑Splatting、NeST‑Splatting、Hybrid Latents、BBSplat、Nexels、FastGS等方法对比，取得约24.1 PSNR、0.852 SSIM、0.157 LPIPS、仅15万点、648 FPS的性能，速度比3DGS快5倍、比FastGS快近2倍，且在纹理细节上优于其他方法。

**⚠️ 局限性**

主要局限包括：训练时哈希+MLP反向传播仍是瓶颈；纹理图集占用显存较高；纹理坐标计算仍耗时；需要进一步压缩纹理并降低过绘导致的渲染开销。

---

## 372. Persona Migration and Expectation Recalibration in Generative AI Adoption: A Longitudinal Study at a State Department of Transportation

**arXiv ID:** 2607.13798 | [PDF](https://arxiv.org/pdf/2607.13798v1)

**作者:** Omidreza Shoghli `[一作]` (University of North Carolina at Charlotte), Amin Mohamadi Hezaveh `[通讯]` (North Carolina Department of Transportation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对北卡罗来纳州交通部（NCDOT）175名员工参与的为期八周的Microsoft 365 Copilot试点进行纵向研究，利用两波匹配问卷测量技术接受模型（TAM）和信任构念（PU、PEOU、BI、TR）的变化，并通过聚类分析识别三种使用人群（怀疑者、谨慎积极者、拥护者）及其迁移路径；同时对开放式反馈进行关键词式内容映射。

**💡 创新点**

① 将TAM与Bhattacherjee的期望-确认模型（ECM‑IT）结合，系统性解释使用后感知的正负确认；② 在公共部门中首次采用人群迁移矩阵来跟踪个体接受度变化；③ 将定量聚类与定性关键词映射相结合，提供多维度的使用体验视角。

**🔧 技术方法**

问卷调查（5点李克特量表）、Wilcoxon符号秩检验、Mann‑Whitney U检验、k‑means聚类、Elbow、Silhouette、Calinski‑Harabasz指标、迁移矩阵分析、关键词匹配与频次统计、可视化（Matplotlib/Plotly）。

**📊 数据集**

NCDOT员工124名匹配样本（175人预试点，133人完成后测，最终124人），自评数据包含PU、PEOU、BI、TR、任务使用与关注点等六个量表。

**📈 对比分析**

通过Wilcoxon检验比较预后与后测构念；聚类采用多种评估指标确定k=3；迁移路径按行百分比展示；关键词映射显示不同构念在开放式回答中的出现频次。结果表明使用后PU显著下降（-0.23，p<0.001），PEOU、BI、TR变化不显著；迁移矩阵显示40%怀疑者上升至谨慎积极者，68%拥护者下降至更低层级；任务使用与关注点的变化揭示使用场景与风险认知的再校准。

**⚠️ 局限性**

① 仅覆盖单一州交通部，样本规模有限，缺乏跨机构可比性；② 试点仅八周，未能捕捉长期使用与持续信任变化；③ 仅基于自报问卷，未结合客观使用日志；④ 某些迁移路径样本过小，推断受限；⑤ 关键词映射未考虑语境情感细微差别。

---

## 373. PriEval-Protect: A Unified Framework for Privacy Evaluation and Protection in Healthcare Systems

**arXiv ID:** 2607.13754 | [PDF](https://arxiv.org/pdf/2607.13754v1)

**作者:** Ilef Chebil `[一作]` (National Institute of Applied Science and Technology), Layth Sliman `[通讯]` (Paris-Panthéon-Assas University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个两阶段统一的医疗隐私评估与保护框架，先通过法律LLM+RAG对政策进行合规评分，再结合加密、架构与技术指标计算隐私风险，并根据风险水平推荐相应的防护措施（如差分隐私、联邦学习、数据掩码）

**💡 创新点**

创新点在于将法规合规评估与技术风险评估无缝融合，利用RAG+LLM实现自动化GDPR/HIPAA合规评分，并通过AHP多准则决策加权聚合生成统一风险分数，随后依据风险水平给出情境化的隐私防护建议，弥补了法规与技术评估的整合空白

**🔧 技术方法**

采用了领域适配的法律LLM（LoRA微调+4‑bit量化）、检索增强生成（RAG）与向量数据库Qdrant、AHP多准则决策、AES‑256/RSA加密、差分隐私与联邦学习等技术

**📊 数据集**

使用了多家医院内部的隐私与数据共享政策文件、结构化匿名化患者数据集（人口统计、临床、行政字段），以及约9,000+标注的GDPR政策片段作为训练和评估数据

**📈 对比分析**

与Benjumea隐私尺度专家评分对比，MAE 1.32、Pearson 0.78；与Pycanon和ARX工具比较技术指标，偏差分别为0.6和0.4；与专家手工风险标签对比，整体风险分类一致率82.6%，高风险召回81.2%，显示与专家一致性良好、误差较低

**⚠️ 局限性**

限制在于目前仅验证了结构化文本和表格数据，未覆盖非结构化文本、影像等多模态数据；对不同法规环境的适用性需进一步验证；框架在真实临床部署中的性能与可扩展性仍需进一步评估

---

## 374. Constraint-Driven Model Optimization: An Industry Framework for Selecting Compression and Acceleration Techniques in Modern Machine Learning Systems

**arXiv ID:** 2607.13735 | [PDF](https://arxiv.org/pdf/2607.13735v1)

**作者:** Dhruv Shivkant `[一作]` (Indian Institute of Science), Utkarsh Wadhwa `[通讯]` (EXL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于约束驱动的多目标模型优化框架，先通过五维约束（数据可用性、延迟预算、内存预算、精度容忍度、再训练预算）对生产部署进行表征，并将已有的压缩与加速技术映射到这些约束上，进而给出四套代表性工业部署的决策管线。

**💡 创新点**

创新点在于：
1) 将模型优化视为约束驱动的工程决策，而非单纯的算法改进；
2) 构建了包含数据、延迟、内存、精度、再训练五维交互的约束词典；
3) 通过对25余篇主流论文的经验数据进行归纳，形成“约束-技术”映射表；
4) 提出了可操作的阶段化决策流程（先内存、再延迟、再数据/再训练、再精度），并给出四个行业用例的具体管线示例。

**🔧 技术方法**

采用的技术涵盖：
- 量化（GPTQ、AWQ、OmniQuant、ZeroQuant-V2、KIVI、KVQuant）
- 剪枝（Wanda、LoSparse）
- 低秩/稀疏混合（LoRA、LoftQ、QA-LoRA、LoRA Prune/ZipLM）
- 推理加速（FlashAttention/FlashAttention‑2、Speculative Decoding、vLLM‑PagedAttention、LLMLingua、Skeleton‑of‑Thought、StreamingLLM、Gisting/AutoCompressors）
- 领域自适应（LIMA、Socratic CoT）
- 预算管理与精度保障（FrugalGPT、SpQR/SqueezeLLM、CALM/SkipDecode）。

**📊 数据集**

论文本身不使用新的数据集，而是对已公开的 25 余篇主要论文中的实验结果进行整理与映射；所引用的论文覆盖多种常见数据集（如 GLUE、SuperGLUE、C4、OpenWebText、SQuAD、InstructGPT、LLM‑Chat 等）。

**📈 对比分析**

比较方式主要是“经验合成”——将各论文中报告的性能提升（如延迟缩短、内存减少、精度变化）映射到对应的约束维度，并在四个工业场景中给出预期收益。论文未进行统一基线实验，而是强调在实际生产流量上进行验证，指出不同技术组合的效果可能并非线性叠加，甚至可能出现性能交叉影响。

**⚠️ 局限性**

局限性：
1) 采用叙述式综述而非系统性文献检索，缺乏可重复的检索策略与筛选标准；
2) 约束维度虽覆盖核心，但未将吞吐率、能耗、硬件兼容性、可观测性等重要指标纳入主轴；
3) 论文中给出的实验数值来自各原始工作，缺乏统一对照基线，难以直接比较不同方法在同一硬件/任务上的真实增益；
4) 不同技术之间的交互效应未系统评估，建议在业务场景中做局部实验；
5) 论文聚焦 LLM 及其压缩/加速技术，其他模型或任务的通用性需进一步验证。

---

## 375. Epidemic Informatics and Control: A Holistic Approach from System Informatics to Epidemic Response and Risk Management in Public Health

**arXiv ID:** 2607.13914 | [PDF](https://arxiv.org/pdf/2607.13914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 376. Comprehensive, Efficient Large-Scale Community Detection via Structural Entropy Game

**arXiv ID:** 2607.13713 | [PDF](https://arxiv.org/pdf/2607.13713v1)

**作者:** Pu Li `[一作]` (Kunming University of Science and Technology), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于两维结构熵潜在博弈的社区检测框架 CoDeSEG，可在大规模（百万级节点、数十亿边）网络上快速定位离散与重叠社区，并通过 CIPA 策略适配动态网络。

**💡 创新点**

创新点在于：①将两维结构熵作为潜在函数，导出近乎 O(1) 的节点效用函数；②设计基于熵的重叠启发式，显著提升重叠社区质量；③提出级联影响传播自适应更新机制，兼顾精度与效率，适用于有向、加权、动态网络。

**🔧 技术方法**

技术方法包括：结构熵最小化、潜在博弈理论、节点效用推导、熵启发式重叠检测、级联影响传播（CIPA）、并行化实现与多线程优化。

**📊 数据集**

使用14个真实大规模网络（Amazon、YouTube、DBLP、LiveJournal、Orkut、Friendster、Wiki、X12、X18 等），5 个合成 LFR 网络（50K–1M 节点）以及 2 个动态网络 X12、X18 进行评测。

**📈 对比分析**

与 SLPA、Bigclam、NcGame、Fox（重叠）以及 Louvain、DER、Leiden、FLPA（非重叠）和 QCA、DynaMo、DCDME、DCDBFE（动态）比较，CoDeSEG 在 ONMI/NMI/F1 维度均领先并在速度上实现 33 倍以上加速（非重叠 1.8×，动态 80×）。

**⚠️ 局限性**

局限性包括：对参数（τₙ、γ、r）敏感，需要经验调优；当前实现主要针对单一网络类型，尚未扩展到异构、多层或图神经网络等更复杂场景。

---

## 377. Low-Complexity Soft-Aided Error-and-Erasure Decoding for Generalized Product Codes

**arXiv ID:** 2607.13719 | [PDF](https://arxiv.org/pdf/2607.13719v1)

**作者:** Sisi Miao `[一作]` (Karlsruhe Institute of Technology), Laurent Schmalen `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种改进的低复杂度软援助误差与擦除解码器——RDRSD，并将其应用于广义产品码（GPC）中，兼顾了硬判决解码的低内部数据流和内存需求，同时实现了在信号水洼区的显著性能提升。

**💡 创新点**

创新点包括：①引入动态可靠性分数（DRS）和码字节点可靠性分数，对误差与擦除解码（EaED）的误差检测进行软信息化；②改进的列表式EaED无需排序，降低了计算复杂度；③设计了软援助后处理方法，通过部分擦除与随机化策略有效消除大停滞模式，进一步降低误差地板。

**🔧 技术方法**

技术手段：基于BCH/扩展BCH的有限域错误与擦除解码；硬消息传递的迭代硬判决解码（iBDD）框架；动态可靠性分数的初始化与更新；误差检测与纠正机制（anchor bits、累加阈值）；后处理中的Flip‑/Erase‑iterate与随机擦除。

**📊 数据集**

实验数据：在BI‑AWGN、BSC与EaE信道模型下，使用[256,239,6]扩展BCH作为组件码，构造PC与Staircase码；通过仿真评估，未使用公开数据集，全部为模拟实验。

**📈 对比分析**

比较方法：将RDRSD与传统iBDD、DRSD、理想iEaED以及iEaED等进行对比。水洼区性能上，RDRSD相较iBDD平均提升≈1 dB，t=3时提升≈0.95 dB；误差地板方面，RDRSD+后处理可降至10⁻¹²以下，明显低于iBDD；复杂度主要在EaED的列表操作和DRS更新，整体内部数据流与iBDD相当。

**⚠️ 局限性**

局限性：①对t=2（低纠错能力）组件码，误差地板仍高；②误差检测仍不是完全可靠，误差模式的误差不易用解析式建模；③对大停滞模式的后处理仍需多轮或随机化，处理不一定成功；④尚未在真实硬件上验证能耗与实现难度。

---

## 378. Cyclone: Diffusion Model for Cycle-Consistent Weather Editing from Unpaired Driving Data

**arXiv ID:** 2607.13927 | [PDF](https://arxiv.org/pdf/2607.13927v1)

**作者:** Thang-Anh-Quan Nguyen `[一作]` (Huawei Paris Research Center), Roland Brémond `[通讯]` (Gustave Eiffel University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于潜在扩散模型的循环一致天气编辑框架 Cyclone，可在无人配对数据上合成或去除多种天气效果。

**💡 创新点**

创新点在于将循环一致性约束与从预训练文本-图像模型的知识蒸馏、CLIP语义引导相结合，实现在无配对数据下的多天气转换。

**🔧 技术方法**

主要技术包括潜在扩散模型、循环一致性损失、蒸馏损失、CLIP语义引导和文本条件控制。

**📊 数据集**

使用的训练数据集包括 ACDC、SHIFT、BDD100K 和 OpenDV-YouTube，评估数据集则包含 SHIFT、ACDC、nuScenes、PandaSet、Waymo 以及自制 120 场视频集。

**📈 对比分析**

与 Histoformer、AWRaCLe、InstructPix2Pix、BAGEL、Qwen‑Image‑Edit、TokenFlow、CycleGAN、CycleNet 等基线对比，Cyclone 在 DINO‑Struct、CLIP Score、分类准确率和 FID 上均显著优于对手，并在下游深度估计、语义分割和目标检测任务中提升性能。

**⚠️ 局限性**

局限性包括训练需要多次前向传播导致计算量大、VAE 压缩与扩散噪声可能丢失细节，以及目前无法在同一帧同时合成多种不同强度的天气效果。

---

## 379. Safety-Aware Forward Detection in Networked ISAC for Low-Altitude UAV Flight

**arXiv ID:** 2607.13908 | [PDF](https://arxiv.org/pdf/2607.13908v1)

**作者:** Jingli Li `[一作]` (Beijing Jiaotong University), Zhangdui Zhong `[通讯]` (Beijing Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于网络化ISAC的前向检测框架，利用前向ROI分块实现非合作目标检测，并推导了UAV状态估计CRLB与前向ROI误检概率，最后设计了安全感知资源优化方案。

**💡 创新点**

①首次将前向ROI误检可靠性纳入网络ISAC设计；②推导了CRLB与误检概率的尺度律；③提出兼顾通信、定位与前向检测的安全感知资源优化框架。

**🔧 技术方法**

使用随机几何模型、体素化目标占据模型、CRLB与尺度律分析、扩展卡尔曼滤波、能量检测、有限网格搜索资源优化以及多基站协同技术。

**📊 数据集**

通过仿真生成的随机基站部署（二维均匀PPP）和目标PPP，采用论文给出的参数配置进行实验。

**📈 对比分析**

与不考虑前向检测的基线方案对比，仿真显示误检概率降低17.05%，状态估计CRLB上升14.82%，并验证了尺度律与Monte Carlo的一致性。

**⚠️ 局限性**

假设完美同步与CSI共享，目标模型简化，仅考虑体素级检测；资源分配受离散束波与子载波限制；缺乏实测验证；PPP模型可能不完全符合实际部署。

---

## 380. PiVoT: A Variational Solution for Real-time Large-scale Multi-object Detection and Tracking under Heavy Clutter

**arXiv ID:** 2607.13891 | [PDF](https://arxiv.org/pdf/2607.13891v1)

**作者:** Runze Gan `[一作]` (University of Edinburgh), James R. Hopgood `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无训练、实时的多目标检测与跟踪框架 PiVoT，能够直接从高分辨率雷达点云（包括多普勒信息）完成目标的联合检测、跟踪、形状、存在概率等估计。

**💡 创新点**

核心创新：① 基于非均匀泊松过程的两阶段变分推断，使得目标检测与跟踪在理论上可行且可并行；② 提出了理论上可证明的出生剔除（birth pruning）机制，将原本二次复杂度的存在/可探测性推断降为线性时间；③ 首次在泊松测量模型中加入多普勒信息，保持线性高斯似然，兼顾位置与径向速度。

**🔧 技术方法**

使用技术包括：变分推断（CAVI）、泊松测量模型、两阶段联合检测跟踪推断、Lambert W 函数阈值剔除、矩阵高斯/威斯特海特/Gamma 分布推断、线性高斯似然计算，以及可扩展到多普勒的高斯似然。

**📊 数据集**

实验数据集：① 6 个仿真雷达数据集（DS1–DS6）用于不同目标数与杂波程度的评估；② RadarScenes 自动驾驶雷达数据集（验证集与测试集）用于与深度学习基准对比；③ 大规模仿真环境（1000+目标）用于展示 PiVoT 的可扩展性。

**📈 对比分析**

与传统 NHPP 基础的 PMBM 滤波器、SPA 粒子法等贝叶斯跟踪器比较，PiVoT 在所有场景下都实现了更低的 GOSPA、误检/漏检率，并且运行速度提升 50–100 倍；在真实 RadarScenes 数据上，PiVoT 的帧级宏平均 F1、AP、GOSPA 与深度学习基准 RadarGNN 的精度相近甚至更优，且在单机 CPU 上即可满足实时需求。

**⚠️ 局限性**

局限性：① 对目标运动假设为刚体或近似恒速，未考虑转向或非刚体效应；② 多普勒建模仅包含径向速度，忽略了转向、噪声或 RCS 影响；③ 采用均匀出生先验，在极低测量率场景下可能导致检出不稳定；④ 未在分布式多传感器融合或 GPU 并行加速等方面做深入实验。

---

## 381. Learning Forward & Reverse Skills from a Single Unfinished Demonstration for Constrained Manipulation Tasks

**arXiv ID:** 2607.13882 | [PDF](https://arxiv.org/pdf/2607.13882v1)

**作者:** Yexin Hu `[一作]` (Technische Universität Wien), Dongheui Lee `[通讯]` (Technische Universität Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种基于单次（可能未完成）演示的机器人约束操作学习框架，能够同时实现正向和逆向任务执行。

**💡 创新点**

创新点在于将演示分为自由运动和接触段，利用几何驱动的 twist‑direction 分段方法提取螺旋运动原语，并结合力/力矩感知的六维姿态补偿与一维速度调节，实现超越演示长度的接触执行以及无需额外学习即可完成逆向操作。

**🔧 技术方法**

使用的技术包括动态运动原语（DMP）编码自由运动、螺旋原语（screw primitive）拟合接触段、力/力矩传感的六维 admittance 控制与一维速度 admittance、基于 twist 方向的分段算法，以及逆向执行时对原语参数取负并按相反顺序重用。

**📊 数据集**

实验数据来自Franka Emika Panda机器人在四个真实任务（插孔、插电池、开锁、螺丝驱动）上的单次手导示范（完整或截断），未使用公开数据集。

**📈 对比分析**

与三种基线（仅 DMP、DMP+admittance、DMP+twist+admittance）比较，完整演示下本方法在所有任务均达 20/20 成功率，截断演示下 18/20；相较基线显著提升，尤其在锁定、螺丝驱动及逆向任务中表现最优。

**⚠️ 局限性**

局限性包括对螺旋轴/倾斜角的误差敏感，特别是在多次重新抓取的螺丝驱动中累计偏差；仅能泛化到具有相似接触拓扑的物体；执行速度未做优化；需要更快的姿态对齐与原语在线自适应。

---

## 382. Task-Oriented Sensing and Covert Transmissions for Collaborative Multi-AUV Systems

**arXiv ID:** 2607.13880 | [PDF](https://arxiv.org/pdf/2607.13880v1)

**作者:** Xueyao Zhang `[一作]` (Northwestern Polytechnical University), Chau Yuen `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了SVR‑MARL框架，通过价值驱动的通信调度实现隐蔽多AUV协同定位跟踪。

**💡 创新点**

创新点在于将感知信息价值与实际通信物理条件（时延、干扰、隐蔽性）结合，形成任务导向的实际信息价值；引入任务感知消息、情境编码器和价值估计模块；采用物理真实水声通道模型；实现通信与任务的闭环耦合。

**🔧 技术方法**

采用多智能体深度强化学习（PPO/IC3Net等）与任务感知消息编码，GRU上下文编码器，Beta分布发射功率策略，Lagrangian约束惩罚，以及Thorp路径损耗、噪声、时延、干扰等物理水声通道模型。

**📊 数据集**

基于仿真生成的三维水下追踪环境数据，包括四个AUV、一个移动目标和一个窃听者；数据来源为仿真轨迹与传感观测。

**📈 对比分析**

与TrueObs、FullComm、NoComm、IC3Net四个基线比较，使用任务效率（捕获率/任务步骤）评估；SVR‑MARL在实际通信方案中比IC3Net提升约20%，且优于其他方案。

**⚠️ 局限性**

局限性：未考虑队友意图预测与安全认证；仅在仿真中验证，缺乏真实水下实验；对干扰模型和隐蔽性假设有限；未处理恶意攻击。

---

## 383. Rethinking Speech Foundation Model Fine-tuning: Better SFT or Better Match?

**arXiv ID:** 2607.13864 | [PDF](https://arxiv.org/pdf/2607.13864v1)

**作者:** Wangjin Zhou `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对SFT在不同预训练模型实例下的表现进行系统评估

**💡 创新点**

揭示SFT优劣不是普适性能上限，而是依赖实例和种子匹配的激活匹配

**🔧 技术方法**

基于SUPERB分类任务的实验，使用多种SFT配置和多重随机种子

**📊 数据集**

使用wav2vec 2.0、HuBERT、WavLM等九个SSL检查点，以及SUPERB的意图识别、情感识别、说话人识别任务

**📈 对比分析**

通过配对McNemar检验比较不同SFT配置在同一检查点的统计可比性，结果显示排名随检查点和种子变化不稳定

**⚠️ 局限性**

实验受限于模型规模与计算资源，仅覆盖基准规模检查点，未深入分析更大规模或不同任务的通用性

---

## 384. Towards Enhancing 3D Spatial Reasoning in Medical Multimodal Large Language Models

**arXiv ID:** 2607.13860 | [PDF](https://arxiv.org/pdf/2607.13860v1)

**作者:** Zhuoyuan Fu `[一作]` (University of International Relations), Yaru Zhao `[通讯]` (University of International Relations)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过双代理（观察者-合成器）构建基于切片的链式推理（CoT）数据集 Hounsfield‑CoT，并用其对 2D 预训练的多模态大语言模型进行指令微调，实现对 3D 医学影像的空间推理与可解释诊断。

**💡 创新点**

创新点在于：①提出切片级 CoT 合成范式，精准模拟放射科医生的逐层阅读流程；②设计双代理框架，严格约束顺序空间追踪、3D 空间意识与差异排除；③通过数据驱动的方式在无昂贵 3D 预训练的前提下，显著提升 2D 模型的 3D 推理能力。

**🔧 技术方法**

主要技术包括：基于 LoRA 的指令微调、2.5D 切片序列化视觉编码、Observer‑Synthesizer 大语言模型链式推理生成、BERT‑F1 与多选准确率评估。

**📊 数据集**

使用 CT‑RATE 数据集生成的 11.2k 例 Hounsfield‑CoT 作为训练集，验证集涵盖 3D‑RadVQA（T1‑T4）与 DeepchestVQA（识别、视觉推理、医学推理）三大任务。

**📈 对比分析**

在 3D‑RadVQA 上，微调后 T1（异常检测）从 7.2% 提升至 33.5%，T2（图像观察）从 7.7% 提升至 24.1%，T4（存在检测）从 77.8% 提升至 80.2%；在 DeepchestVQA 的零样本场景下，零样本准确率从 39.7% 提升至 45.1%，显著优于基准模型（如 Huatuo 38.0%、Med3DVLM 37.9%）。

**⚠️ 局限性**

局限性包括：合成过程依赖全局报告，可能产生细微空间幻觉；目前仅针对 CT 影像，尚未验证至 MRI 等其他模态；评估指标仍以整体准确率为主，缺乏对单步推理正确性的细粒度验证。

---

## 385. Merging Reaction to Cognition: A Hybrid Cognitive Strategy for Odour Source Localisation in Natural Environments

**arXiv ID:** 2607.13853 | [PDF](https://arxiv.org/pdf/2607.13853v1)

**作者:** Hugo Magalhães `[一作]` (University of Coimbra), Lino Marques `[通讯]` (University of Coimbra)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合化学源定位策略，将基于信息论的认知搜索与基于化学探测的生物启发式反应相结合，形成Hybrid Fast-Cognitive算法。

**💡 创新点**

创新点在于引入检测触发的二元行为切换机制：在探测到化学信号时立即从横向探索切换为源导向动作，且行为参数动态从信念分布中提取，既保留认知的自适应性，又具备生物启发式的直接性和低参数调优需求。

**🔧 技术方法**

使用的技术包括：基于高斯云模型的粒子滤波推断、贝叶斯信念更新、基于行为树的实时决策框架、以及快速计算的Fast-Cognitive运动模式；实验平台为配备电导率传感器的自主水面车辆（ASV）。

**📊 数据集**

实验数据集包括：三种不同湍流强度下的数值模拟数据（S1~S3）和实地Mondego河流域的七次ASV实验，传感器测得的电导率变化作为化学浓度观测。

**📈 对比分析**

与Fast-Cognitive对比：在所有情景下成功率提升约10%（S3中从61%升至93%），搜索距离缩短约26%，距离比（搜索距离/最短距离）从4.21降至3.19；实地实验中成功率为86%，平均定位误差3.17 m，距离比3.12，均显著优于传统认知方法。

**⚠️ 局限性**

局限性：二元切换在高度间歇化学信号下可能导致过度切换或错误决策；一旦声明行为触发，若信念收敛错误则难以纠正；未考虑多机器人协同或多模态感知的扩展。

---

## 386. Beyond the $d^{2.5}$-mixing bound for Dikin walks on polytopes

**arXiv ID:** 2607.13943 | [PDF](https://arxiv.org/pdf/2607.13943v1)

**作者:** Yunbum Kook `[一作]` `[通讯]` (Georgia Tech), Yunbum Kook (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

对 Dikin walk 的混合时间进行理论改进，从之前的 d^2.5 降低到 d^2.25（warm start）以及通过退火得到 d^2.56（cold start）

**💡 创新点**

通过对 Lee–Sidford 阻塞度量的平均自共形性（ASC）进行高阶分析，并将 Lewis 权重计算移至移动正交框架，利用 Wiener 级数与多重斯托克斯积分实现更紧的高阶矩估计

**🔧 技术方法**

高阶 Taylor 展开、移动正交框架计算、Hermite 多项式/ Wiener 级数分解、Gaussian 高阶矩的 L^2 估计、对称性和自共形性框架

**📊 数据集**

无实验数据集，纯理论分析

**📈 对比分析**

与之前基于 log 阻塞的 Dikin walk (md) 与基于 Lewis 权重的 d^2.5 结果相比，本文取得了更小的维度幂次；在退火框架下从冷启动得到 d^2.56 的复杂度，优于现有的 d^2.375 等

**⚠️ 局限性**

尚未达到理想的 d^2 目标，仍需进一步提升高阶估计；当前方法对矩阵维度和约束数量的 polylog 依赖未被完全消除；实现上需要对 Lewis 权重的高阶导数控制仍具挑战

---

## 387. Backpropagation for Effectful Languages I: Finite Probability and Discrete Output Algebraic Effects

**arXiv ID:** 2607.13935 | [PDF](https://arxiv.org/pdf/2607.13935v1)

**作者:** Diogo Simm `[一作]` (Utrecht University), Matthijs Vákár `[通讯]` (Utrecht University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出一种基于范畴论的反向自动微分（CHAD）代码转换方法，专门处理有限离散概率计算，并给出其语义正确性证明。

**💡 创新点**

创新点在于：① 将权重与原子位置的余切（cotangent）流动纳入反向 AD 的范畴语义；② 通过分解式（factorisation）构造有限原子分布单子并推导其逆向变换；③ 把该模式推广为可复用的离散输出代数效应（非确定性、异常、写入器）逆向 AD 方案。

**🔧 技术方法**

使用技术：范畴论（强单子、伴随、分解式）、CHAD 语义化转换、逻辑关系（categorical logical relations）证明、对偶双线性结构、期望算子等。

**📊 数据集**

数据集：无实验数据，主要为理论证明和抽象语义，未使用具体数据集。

**📈 对比分析**

比较方法与性能：与前向 AD（如 ADEV）进行理论对比，展示逆向 AD 在梯度流动和代数效应处理上的优势；本文未给出实现实验，性能评估留待后续实现与基准测试。

**⚠️ 局限性**

局限性：仅适用于有限离散概率分布；不涵盖连续分布或更一般的概率效应；实现细节与性能尚未验证；对更复杂效应的推广仍需进一步工作。

---

## 388. Plausible Deniability Guarantees for Whistleblowers

**arXiv ID:** 2607.13928 | [PDF](https://arxiv.org/pdf/2607.13928v1)

**作者:** Leo Richter `[一作]` (University College London), Matt J. Kusner `[通讯]` (École Polytechnique de Montréal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文在举报人匿名保护的背景下，构建了一个基于差分隐私的审计框架，定义了对单个举报事件的 (0,δ)-DP 隐私保障，并提出一种基于 Toeplitz 连续计数的通用降维机制（Toeplitz Continual Auditing, TCA），实现了在多轮审计中对举报者的可否认性保护。

**💡 创新点**

创新点主要包括：
1) 以举报者单次报告为邻接单位，针对强敌（被审计组织）设计了严格的 (0,δ)-DP 隐私定义；
2) 证明随机响应（Randomized Response）在此模型下的隐私-准确性折衷受限，误差最多仅比均匀随机高 δ；
3) 提出将任意 (0,δ)-DP 连续计数器与审计决策结合的通用递归机制，并通过 Toeplitz 预编码实现噪声尺度仅为 O(√log T)，从而使误差随真实举报差距增大而快速消失。

**🔧 技术方法**

技术手段包括：
- 差分隐私（(0,δ)-DP）与总变差表征；
- 连续计数（continual counting）与矩阵因式分解；
- Toeplitz 低三角矩阵编码与解码；
- 高斯噪声机制；
- 后处理不破坏隐私的原则；
- 理论误差上界与实验验证。

**📊 数据集**

本文没有使用真实数据集，而是通过两组仿真（静态差距扫描与动态在线审计）来评估机制性能，生成的报表流为人工构造的随机过程。

**📈 对比分析**

在实验中将 TCA 与随机响应、均匀随机审计和无隐私的贪心审计进行对比。实验结果显示：
- TCA 的误选率随领导组织与挑战者的报告差距增大而显著下降；
- 随机响应对差距几乎不敏感；
- 在动态情境下，TCA 的主动计数缺口（deficit）与贪心基线接近，远优于随机响应和均匀随机；
- 综上，TCA 在保持相同 (0,δ)-DP 隐私约束的前提下，显著提升了审计准确性。

**⚠️ 局限性**

局限性包括：
1) 对单个举报者的多次报告，隐私保证随报告数量线性退化（group privacy）；
2) 依赖受信任的审计者不泄露计数，若审计者失误或被攻破则隐私失效；
3) 仅在仿真环境下验证，缺乏真实审计系统的数据验证；
4) 对于极端稀疏或高频报送场景的表现尚未系统评估。

---

## 389. How to Guide LLM Generation: Dual-Surrogate Guided Search for Automated Heuristic Design

**arXiv ID:** 2607.13911 | [PDF](https://arxiv.org/pdf/2607.13911v1)

**作者:** Yuhan Wang `[一作]` (South China Agricultural University), Zhi-Hui Zhan `[通讯]` (Nankai University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于双重代理的行动选择模块 DGS，利用转移代理和实例条件效用代理在 LLM 自动启发式设计中预测生成前的子程序表示与效用，从而更高效地分配查询与评估预算。

**💡 创新点**

创新点在于把操作-父级组合视为可学习的决策变量，结合转移预测、效用估计与不确定性感知采集规则，实现从过去评估中直接指导下一步 LLM 生成，而非传统的规则或固定策略。

**🔧 技术方法**

采用 ModernBERT 代码编码器、共享潜在表示、残差 MLP、实例条件效用集合、对数正态转移代理以及多头均值/方差高斯负对数似然训练，并使用 Monte‑Carlo 采样估计期望与 UCB 样式的采集准则。

**📊 数据集**

在五个启发式设计任务上进行实验：旅行商问题 (TSP)、背包问题 (KP)、在线装箱 (OBP)、可接受集合问题 (ASP) 与 CVRP‑ACO，每个实验预算 200 次生成/评估。

**📈 对比分析**

与 FunSearch、EoH、ReEvo、MCTS‑AHD 四个代表性 LLM‑AHD 基线在相同预算下对比，DGS 在 TSP、KP、OBP 获得最佳平均排名，在 ASP 与 ReEvo 接近，在 CVRP‑ACO 仅次于 MCTS‑AHD，整体显著提升样本效率。

**⚠️ 局限性**

局限性包括对不确定性校准的需求、对预训练潜在表示的依赖、在更大或更复杂的生成器/操作符集合上的泛化待验证，以及方法主要针对 EoH‑style 语义操作符，扩展到其他生成策略仍需进一步研究。

---

## 390. Unleashing Multimodal Large Language Models for Training-free HOI Detection in the Wild

**arXiv ID:** 2607.13881 | [PDF](https://arxiv.org/pdf/2607.13881v1)

**作者:** Ting Lei `[一作]` (Peking University), Yang Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AgentHOI，一个无训练数据、利用多模态大语言模型进行开放世界人-物交互检测的框架。

**💡 创新点**

通过上下文感知多轮推理与多面交互定位，充分利用 MLLM 的推理和多模态定位能力，实现无训练数据的开放词汇 HOI 检测。

**🔧 技术方法**

使用多模态大语言模型（如 GPT‑4o）、对齐模型 GroundingDINO 进行语义推理与空间定位，配合多轮推理与描述性提示。

**📊 数据集**

在 HICO‑DET 与 SWIG‑HOI 等公开数据集上评估。

**📈 对比分析**

与监督、弱监督、零样本 HOI 检测方法对比，AgentHOI 在多种零样本分割上达到或超过 SOTA，尤其在 Rare、UO、RF‑UC 场景表现突出，mAP 提升约 5–10 分。

**⚠️ 局限性**

推理耗时较高，需调用大型 MLLM，且在极端遮挡或复杂场景下仍可能产生误定位，缺乏针对性微调的精细化能力。

---

## 391. From Classification to Consistent Templates: Multiple Permuted-Label Classifier Encoding for Biometric Template Protection

**arXiv ID:** 2607.13845 | [PDF](https://arxiv.org/pdf/2607.13845v1)

**作者:** Baogang Song `[一作]` (Wuhan University of Technology), Dongdong Zhao `[通讯]` (Wuhan University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出多重置标签分类器编码（MPLCE）方案，通过多分类器标签置换生成一致的中间模板，再用 XOR 随机化和 SHA3‑512 哈希实现无错误纠正、无辅助数据的精确匹配，提供跨模态的生物特征模板保护。

**💡 创新点**

创新点在于：① 利用身份分类而非传统特征映射，确保中间模板在同一身份样本间保持一致；② 通过多重置标签多分类器避免单标签重复，显著扩展候选空间；③ 采用 XOR 随机化+哈希的精确匹配方式，实现可撤销、不可逆、不可链接的全新保护范式。

**🔧 技术方法**

技术包括：深度特征提取网络（IR50/IR101/FaceNet、ResNet18/34/DeepIrisNet）、多重置标签分类器训练、标签二进制编码、应用特定 XOR 随机化、SHA3‑512 哈希、完整安全分析与攻击评估。

**📊 数据集**

使用的数据集：人脸方面的 VGGFace2、FaceScrub、YouTube Faces、Extended YaleB；虹膜方面的 CASIA‑Iris‑Lamp、CASIA‑Iris‑Thousand。

**📈 对比分析**

与多种现有 BTP 方法（如 BioDeepHash、FuzzyVault、FuzzyVault 等）在同一基准下对比，MPLCE 在人脸上 GAR≥98% 并且 FAR≈0%，在虹膜上 GAR≥99% 并且 FAR≈0%，同时在离线枚举、相似性攻击和跨应用链接攻击中表现出良好的不可逆、可撤销、不可链接特性。

**⚠️ 局限性**

局限性包括：对分类器一致性依赖较高，用户数或分类器数增大会略微降低 GAR；模型训练和标签置换的随机性要求较高；未针对模型侧攻击（如对抗样本或模型盗窃）提供专门防护。

---

## 392. From Continuous Deployment to Queryable Dataset: Terabyte-Scale AIS-Aligned Passive Acoustic Labelling

**arXiv ID:** 2607.13840 | [PDF](https://arxiv.org/pdf/2607.13840v1)

**作者:** Wayne Renaud `[一作]` (Dalhousie University), Gabriel Spadon `[通讯]` (Dalhousie University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个基于数据库的工作流程，将长期被动声学记录与 AIS 轨迹对齐，生成可查询的距离分辨标签。

**💡 创新点**

创新点在于用 set‑based 时空联接替代内存嵌套迭代，实现部署级别可扩展、持久化的距离分辨弱标签。

**🔧 技术方法**

使用 PostgreSQL/PostGIS 的时空索引、SQL set‑based 关联、Parquet 存储以及频谱统计分析等技术。

**📊 数据集**

使用了加拿大海洋网络（ONC）部署的 AMAR 记录与 AISdb 处理得到的 690 万 AIS 位置报文。

**📈 对比分析**

通过对比无接触、单接触、双接触窗口的 RMS、功率谱、SNR 等特征，展示了在噪声主导环境下声学信号随距离衰减的规律，验证了数据产品的物理可解释性。

**⚠️ 局限性**

局限在于标签仅为 AIS 条件弱标注，缺乏直接声学验证，未覆盖无 AIS 或非监测船舶，背景噪声不确定，且未评估后续检测/分类性能。

---

## 393. A Self-Evolving Agent for Longitudinal Personal Health Management

**arXiv ID:** 2607.13940 | [PDF](https://arxiv.org/pdf/2607.13940v1)

**作者:** Haoran Li `[一作]` (Fudan University), Hongcheng Guo `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了 HealthClaw 个人健康智能体架构，支持跨时间的个体化记忆更新与安全治理。

**💡 创新点**

创新点在于将共享安全规则与医学知识与用户长期记忆分离，并通过后续事件诱导（induction）决定何项信息保留、修改或丢弃，实现选择性更新与隐私保护。

**🔧 技术方法**

采用闭环交互（perception‑reasoning‑action‑induction）、多层可演化记忆（L0–L4）、工具调用、检索增强生成以及 Qwen‑3.7 评估器等技术。

**📊 数据集**

使用合成一年周期轨迹的 1,000 次纵向基准、100 条隐私探测问卷以及九个 200 条例样本的生物医学任务（如 NoduleMNIST3D、GeneTuring 等）。

**📈 对比分析**

通过与当前‑仅和全历史两种提示基线的配对比较，纵向支持查询准确率从 0.2% 提升至 45.7%，隐私探测中准确率 0.64；在九个医学任务中平均绝对增益 27.0pp，七项显著提升。

**⚠️ 局限性**

评估基于模拟轨迹和自动评估器，未涵盖真实用户反馈、临床前景、数据偏差、工具校准等问题，需前瞻性临床验证与安全治理。

---

## 394. ExpressionCueLens: A Cross-Cultural Analysis of Human-AI Companion Conversations on Social Media

**arXiv ID:** 2607.13924 | [PDF](https://arxiv.org/pdf/2607.13924v1)

**作者:** Lynnette Hui Xian Ng `[一作]` (Carnegie Mellon University), Mona Diab `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入ExpressionCueLens框架，对Reddit（西方英语平台）与小红书（东方中文平台）上的人类-AI伴侣对话进行跨文化、跨平台的拟人化表达分析，系统地归纳了10类表达与四类线索，揭示不同文化背景下的拟人化偏好与行为模式。

**💡 创新点**

创新点在于①首次构建跨语言、跨文化的表达-线索层级化分析框架ExpressionCueLens；②结合LLM（GPT‑4o）辅助标注实现大规模数据的可靠编码；③提出层级线索强度与表达转移路径的定量描述，并通过与LIWC等心理词典的对齐验证框架的外部效度。

**🔧 技术方法**

主要技术包括：文本与截图数据的预处理与OCR；ExpressionCueLens的手工编码与LLM辅助标注；统计检验（t检验、曼-惠特尼U、Cohen's d、chi‑square、Cramér's V、kappa一致性）；心理语言学分析（LIWC‑2022）；可视化（热图、柱状图）。

**📊 数据集**

数据集为：≈2000条Reddit帖子及其评论（共5646条），≈381条小红书帖子，提取出1702条对话信息，涵盖ChatGPT、Claude、DeepSeek、Gemini等LLM。

**📈 对比分析**

比较方法：对两平台的表达类别分布、线索类型比例进行t检验（Bonferroni校正）与曼-惠特尼U检验；层级线索强度以四类共现数量划分，绘制热图；对转移路径进行频次统计；与LIWC维度关联检验以验证语义一致性。结果显示：小红书在情绪与脆弱度表达上显著更高，Reddit在时间性与身体化表达更突出；LLM标注与人工标注kappa约0.84‑0.85，验证了标注方法的可靠性。

**⚠️ 局限性**

局限性包括：平台表现形式（文本 vs 截图）与文化因素混合，难以单独归因；样本来自自选AI伴侣社群，可能不具代表性；仅涵盖中文与英文，缺乏对其他语种的跨文化验证；研究为相关性而非因果，缺乏纵向或实验验证；LLM标注仍可能存在误差与语言偏差。

---

## 395. High-Order Question Generation in a Multilingual Educational Context

**arXiv ID:** 2607.13901 | [PDF](https://arxiv.org/pdf/2607.13901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 396. DeepStress: Stress-Testing Deep Search Agents

**arXiv ID:** 2607.13920 | [PDF](https://arxiv.org/pdf/2607.13920v1)

**作者:** Ismael Rousseau `[一作]`, Frederic Bechet `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DeepStress框架，通过用可控的合成文档替代搜索代理的检索模块，系统地评估搜索代理在可信度、相关性和事实性三维受损情况下的鲁棒性；

**💡 创新点**

创新点在于构建了一个可控的实验室式压力测试环境，并提出了可靠性感知评分（RAS）及其成本化指标（CoPRAS），从多维度（不仅是最终答案准确率）量化代理在低质量证据下的表现；

**🔧 技术方法**

主要技术包括基于LLM的合成文档生成（根据预设的可信度、相关性、事实性标签生成文档）、代理搜索调用拦截与模拟、以及一系列评估指标（C/I/A/B/M、TPC、RAS、CoPRAS）；

**📊 数据集**

实验使用了HotpotQA和WebQA两大问答数据集，并在合成文档生成过程中利用了对应的Gold Wikipedia段落和人工核实的证据；

**📈 对比分析**

对12种搜索代理在200道题、7种降质场景下进行比较，采用最终答案结果分布、RAS、TPC、CoPRAS等多指标评估。结果显示，部分代理在可信度下降时保持稳定，而多数代理在相关性和事实性下降时表现显著衰退；RAS和CoPRAS揭示了更细粒度的鲁棒性与成本效率；

**⚠️ 局限性**

局限性包括：合成文档可能无法完全模拟真实检索噪声；评估仅考虑三维可靠性，忽略其他可能的检索错误；实验规模受限于200道题和有限场景；未对代理的推理链进行深入定性分析；结果可能受LLM生成和提示设计的偏倚影响。

---

## 397. The 2nd International StepUP Competition for Biometric Footstep Recognition: From Steps to Strides

**arXiv ID:** 2607.13905 | [PDF](https://arxiv.org/pdf/2607.13905v1)

**作者:** Robyn Larracy `[一作]` (University of New Brunswick), Erik Scheme `[通讯]` (University of New Brunswick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍并评估了第二届StepUP步态识别竞赛，提出更严格的评估协议与跨域步态对齐问题。

**💡 创新点**

首次将左右步态对作为验证单位，加入更极端的鞋类与步速变化，并在推理阶段引入配对步态融合与归一化技术。

**🔧 技术方法**

使用时空CNN、GRU、InceptionTime、Cross‑Attention、Circle Loss、SupCon、ArcFace 等深度学习与度量学习方法，辅以S/N归一化与投影策略。

**📊 数据集**

基于公开的 StepUP‑P150 数据集（≈200,000 条高分辨率步态图）以及新构建的 10,000 条极端域移位的左右步态对。

**📈 对比分析**

相较于基线的 14.12% EER，顶级方法在 8.00%–10.27% 之间，验证了时空特征提取与推理时策略的显著提升。

**⚠️ 局限性**

对未知鞋类的泛化仍有限，极端鞋型（高跟、拖鞋）样本稀缺，系统倾向低拒绝率，安全性有待进一步提升。

---

## 398. Genre Bias or Aesthetic Perception? Identifying and Mitigating Shortcut Learning in Music Evaluation

**arXiv ID:** 2607.13903 | [PDF](https://arxiv.org/pdf/2607.13903v1)

**作者:** Yizhou Zhang `[一作]` (Kyoto University), Zhi Gong `[通讯]` (Tencent)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

识别并消除音乐美学评分模型中的流派捷径学习偏差，提出结合焦点回归与群组正则化的统一训练目标。

**💡 创新点**

创新点在于通过样本级焦点重权与群组级性能正则化同时作用，强迫模型学习与流派无关的音乐性特征，显著降低流派相关的评估偏差。

**🔧 技术方法**

采用焦点回归损失（focal regression）和基于EMA的群组性能正则化，结合MSE、Spearman相关、配对偏好评估等技术。

**📊 数据集**

使用SongEval训练集、MTG‑Jamendo与M6平衡子集进行评估，并利用CMI‑Pref与CMI‑Pref‑Pseudo进行配对偏好实验。

**📈 对比分析**

通过同流派与跨流派的配对偏好准确率、SRCC等指标比较；相较于基线MSE模型，整体准确率从0.687提升至0.715（CMI‑Pref），从0.549提升至0.566（CMI‑Pref‑Pseudo），跨流派准确率也得到提升，尤其在pop‑jazz等差异显著的组合上表现更好。

**⚠️ 局限性**

仍受数据集不平衡与标注偏差限制，且模型可能在音量、乐器等非流派因素上存在其他隐蔽的捷径，需要进一步研究。

---

## 399. Jack of All Scales: A Versatile FPGA Tensor Block for MXFP Precisions

**arXiv ID:** 2607.13898 | [PDF](https://arxiv.org/pdf/2607.13898v1)

**作者:** Marwan Mekhemer `[一作]` (University of Waterloo), Andrew Boutros `[通讯]` (University of Waterloo)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文针对FPGA中微尺度浮点（MXFP）格式的张量运算，评估现有Altera Agilex‑5 DSP块的实现瓶颈，并提出一套针对tensor模式的硬件改进方案，实现对所有MXFP精度（含MXFP8）本地支持；

**💡 创新点**

创新点在于通过对DSP块内部张量模式进行结构扩展，使得单一DSP块即可完成不同宽度MXFP的点积运算，兼容原有接口，且面积增量仅为36%，实现了与传统软逻辑或分散DSP模式相比的显著算术密度提升；

**🔧 技术方法**

使用了ASIC级标准单元实现（ASAP7 PDK）、COFFE全定制布线、硬件仿真及Quartus实现，结合多种DSP模式（固定点、浮点、张量）和软逻辑实现进行对比；

**📊 数据集**

未采用真实深度学习数据集，性能评估基于FPGA芯片上合成的Systolic‑array矩阵乘法吞吐率（TFLOPS）和DSP/ALM资源利用率；

**📈 对比分析**

与基线DSP块（最优模式）以及软逻辑实现进行比较，实验显示在E5M2、E4M3、E3M2等无法使用张量模式的格式上，改进后吞吐量提升至9.1×至15.3×，平均提升4.2×；

**⚠️ 局限性**

局限性包括：面积增量虽低但仍为36%（占DSP块的1.8%总芯片面积），改进仅针对Agilex‑5，MXFP8全精度支持仍需额外面积，且改动不涉及其他DSP模式，可能影响后续硬件兼容性与功耗。

---

## 400. Fine-Grained Vision-Language Pretraining with Organ-Conditioned Pattern Tokens for CT Understanding

**arXiv ID:** 2607.13892 | [PDF](https://arxiv.org/pdf/2607.13892v1)

**作者:** Guoliang You `[一作]` (University of Pennsylvania), Xiaomeng Chu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

该工作提出了一种基于器官条件化的模式令牌对齐框架，用于CT-视觉语言预训练，在保持全局扫描-报告对齐的同时，通过稀疏MoE路由和器官槽聚合，将图像和文本的器官证据压缩为连续模式令牌，并与报告中的相应模式令牌进行对齐。

**💡 创新点**

创新点在于：①将稀疏Mixture-of-Experts与器官槽池化结合，形成器官级别的模式令牌瓶颈；②采用结构化软目标的配对模式令牌对比损失，以减少硬负样本惩罚；③在保持可扩展的全局对比学习基础上，引入局部模式监督来提升全局表示。

**🔧 技术方法**

技术上使用了稀疏MoE路由、Slot Pool聚合、结构化软目标对比学习、BIO-MedVLP文本编码器和ResNet-18图像编码器，以及负样本平衡与加载平衡损失。

**📊 数据集**

主要数据集为公开的CT-RATE（约50k胸CT扫描，18种异常标签）和RAD-ChestCT（36k胸CT扫描，83种异常标签）进行零样本诊断和检索评估。

**📈 对比分析**

与基线（CT-CLIP、Merlin、fVLM等）比较，本文在CT-RATE上实现了84.5%的AUROC，较前沿的fVLM提升6.7个百分点；在RAD-ChestCT上实现69.9%AUROC，提升0.8个百分点；检索任务同样获得MAP@50和Recall@100分别提升2.4、19.9个百分点。

**⚠️ 局限性**

限制包括：①仅覆盖肺、心脏、食管和主动脉四个器官，其他异常仍靠全局分支；②结构化软目标依赖报告解析和实体抽取，可能受噪声影响；③外部验证仅在诊断任务上，检索转移仅在CT-RATE；④缺乏专家标注的精确定位验证。

---

## 401. Experience Memory Graph: One-Shot Error Correction for Agents

**arXiv ID:** 2607.13884 | [PDF](https://arxiv.org/pdf/2607.13884v1)

**作者:** Wenjun Wang `[一作]` (University of Electronic Science and Technology of China), Kai Zheng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建经验记忆图（Experience Memory Graph），利用失败和成功轨迹通过图匹配提取共同子图与图编辑路径，形成一次性错误修正指南；

**💡 创新点**

将错误修复视为图匹配问题，离线构建包含任务内知识节点与跨任务通用边的图式记忆，避免在线反思循环并提供可迁移的纠错指令；

**🔧 技术方法**

使用大型语言模型驱动的代理、动作决策图表示、基于最优传输的FGW图匹配、图编辑路径求解、跨任务图联接与检索等技术；

**📊 数据集**

在ALFWorld和ScienceWorld两个长序列任务数据集上进行实验；

**📈 对比分析**

与ReAct、Reflexion、ExpeL、CDMem、MemP等基线进行对比，实验显示在SR和AR上均显著优于所有对比方法，尤其在小模型下提升显著，且仅需一次执行，显著降低时间与API成本；

**⚠️ 局限性**

主要限制包括：需预先获得专家轨迹；对极其复杂任务的提升有限；对无专家场景不适用；离线图匹配虽高效但仍存在一定计算开销。

---

## 402. Lexicographic Direct Access with Functional Dependencies

**arXiv ID:** 2607.13875 | [PDF](https://arxiv.org/pdf/2607.13875v1)

**作者:** Florent Capelli `[一作]` (University Artois), Stefan Mengel `[通讯]` (University Artois)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在满足函数依赖（FDs）的数据库上，对连接查询答案进行词典序直接访问的复杂度，给出了预处理时间与访问时间的细粒度上下界，并对所有FDs情况进行了完整的划分；

**💡 创新点**

创新点在于首次将重排扩展方法与基于信息理论的PANDA算法相结合，提出了兼容性数（color number）与基于多项式边界的宽度度量，揭示了在非单一FDs情况下两种方法的相对优势与限制；

**🔧 技术方法**

主要技术包括：重排扩展（Δ-reordering）与直接映射，基于破坏自由分解的宽度测度（QΔ），信息理论的多项式边界与PANDA算法，颜色技术与不相容数证明；

**📊 数据集**

文中未使用任何实际数据集，而是通过理论构造与假设（Zero-Clique Conjecture）进行证明；

**📈 对比分析**

通过与无FDs情形的已知最优算法对比，作者证明在单一FDs下两种方法等价，在一般FDs下基于多项式边界的方法始终不劣；在某些示例中，后者能显著降低预处理指数；

**⚠️ 局限性**

局限性在于所给上下界一般不完全匹配，尤其在非单一FDs时存在较大gap；结果依赖于Zero-Clique Conjecture；未解决的关键是缺乏最佳的FD-aware join算法，限制了得到最优直接访问的可能性。

---

## 403. PRomop: A Decision-Ready Longitudinal Patient Health Record on the OMOP Common Data Model

**arXiv ID:** 2607.13947 | [PDF](https://arxiv.org/pdf/2607.13947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 404. SPyCE: Skill-Policy Co-evolution for Multimodal Agents

**arXiv ID:** 2607.13854 | [PDF](https://arxiv.org/pdf/2607.13854v1)

**作者:** Ru Zhang `[一作]` (Zhejiang University), Weijie Qiu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向多模态工具使用的技能-策略共进化框架 SPyCE，利用可解释的层次化技能库（执行技能与工作流技能）与强化学习政策在训练过程中相互更新。

**💡 创新点**

创新点在于：①将成功轨迹转化为可重用的两层技能抽象；②构建闭环的技能-策略共进化机制，使得更强的策略能够产生更高质量的轨迹，从而不断改进技能；③结合奖励设计与工具调用惩罚，提升效率与鲁棒性。

**🔧 技术方法**

技术要点包括：多模态强化学习（GRPO/RLOO）、大型语言模型（MLLM_dk）用于技能提炼、文本嵌入检索与层次化技能检索、合并-添加规则用于技能库维护、奖励函数中工具调用惩罚。

**📊 数据集**

使用了 CodeVision 的 SFT 与 RL 训练集、以及八个基准数据集：TIR‑Bench、MathVerse、MathVision、WeMath、ChartQAPro、V* Bench、HRBench‑4K 与 HRBench‑8K。

**📈 对比分析**

与提示、传统 RL（GRPO、RLOO）和记忆式方法（MemP、Dynamic Cheatsheet、Agent‑KB）对比，SPyCE 在所有基准上均实现最高任务成功率，并在工具调用次数上更优，显著提升了效率与性能。

**⚠️ 局限性**

局限性：①对大型预训练语言模型和算力依赖较大；②技能抽象和检索的质量受 LLM 提示与嵌入空间的影响；③目前仅验证在视觉工具使用场景，尚未评估对新工具或完全不同任务的泛化能力；④技能库维护和合并策略可能导致信息丢失或过度压缩。

---

## 405. NodeImport: Imbalanced Node Classification with Node Importance Assessment

**arXiv ID:** 2607.13837 | [PDF](https://arxiv.org/pdf/2607.13837v1)

**作者:** Nan Chen `[一作]` (Johns Hopkins University), Jia Chen `[通讯]` (Grabtaxi Holdings Pte Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个名为 NodeImport 的框架，利用平衡的 meta‑set 对节点重要性进行评估，动态筛选有价值的标签节点、未标记节点和合成节点，以缓解图数据中的类别不平衡问题。

**💡 创新点**

创新点包括：① 从理论推导出可直接计算节点重要性的公式，显著降低了原本昂贵的双层优化开销；② 将节点生成过程与重要性评估解耦，兼容任意合成方法；③ 通过聚类（PAM）构建高质量的平衡 meta‑set；④ 结合梯度方向对齐与上下文相似性，形成直观的阈值筛选机制。

**🔧 技术方法**

核心技术：图神经网络（GCN、GAT、GraphSAGE），基于梯度的节点重要性评估（梯度内积与 trace 形式），MixUp 方式的节点合成，PAM 聚类构造 meta‑set，利用聚合矩阵（如 APPNP、SSGC）实现上下文相似性计算。

**📊 数据集**

在五个主流基准数据集上验证：Cora、CiteSeer、PubMed、Amazon‑Photo、Amazon‑Computers，均采用长尾不平衡比例 IR=50 的训练集。

**📈 对比分析**

与 11 种现有基线（Re‑weight、Balanced Softmax、PC Softmax、ReNode、TAM、GraphENS、GraphSHA、GRAND 等）在 GCN、GAT、GraphSAGE 三种 GNN 结构下进行对比。实验显示 NodeImport 在 Accuracy、Balanced Accuracy 以及 Macro F1 上均超过所有基线，尤其在 Macro F1 方面提升显著，且在高不平衡比例下差距更明显。

**⚠️ 局限性**

局限性：① 重要性评估仍需一次完整的 meta‑set 训练，计算成本相对较高；② 对 meta‑set 质量高度依赖，小样本噪声可能影响评估；③ 对超参数（如 κ、β、γ 等）敏感，需要进一步自动化调优；④ 仅在中小规模图上验证，需在更大图和不同不平衡模式下进一步评估。

---

## 406. Pack, Remove, Reserve -- Online Knapsack with Second Thoughts

**arXiv ID:** 2607.13955 | [PDF](https://arxiv.org/pdf/2607.13955v1)

**作者:** Hans-Joachim Böckenhauer `[一作]`, Philip Whittington `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对在线比例背包问题进行研究，允许两种付费递归机制（保留和删除），并给出任意成本参数 (α,β) 下的精确竞争比。

**💡 创新点**

首次完成了在保留与删除均付费的完整参数空间上的竞争比分析，揭示三种不同的区域：仅保留最优、仅删除最优以及两者协同优于任何单一机制的“共生”区域。

**🔧 技术方法**

采用竞争分析理论，构造匹配的上界算法（分两阶段保留与删除的组合策略）和下界反例树，证明在不同参数区域内该算法达到最优竞争比。

**📊 数据集**

本研究为纯理论性，没有使用任何数据集；所有结果均基于严谨的数学证明与反例构造。

**📈 对比分析**

通过与已知的单机制竞争比（保留、删除）进行比较，证明在共生区域内竞争比显著优于两者单独使用；在其它区域保持与单机制相同的最优竞争比，整体竞争比范围为 [1,∞) 并在大多数参数点实现有限值。

**⚠️ 局限性**

局限性：仅考虑比例背包（价值等于大小）且递归成本为比例形式；不涵盖更一般的背包（非比例价值）或随机化算法；未来工作需探索两机制在一般背包、值比例成本以及随机化场景中的性能。

---

## 407. Peak-End-Net: A Peak-End Rule Inspired Framework for Generalizable Video Aesthetic Assessment

**arXiv ID:** 2607.13941 | [PDF](https://arxiv.org/pdf/2607.13941v1)

**作者:** Geng Li `[一作]` (AMAP, Alibaba Group), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于峰-终规则的轻量化视频美学评估框架Peak-End-Net，并通过图像美学先验、节奏编码与动态门控融合实现视频整体与多维度美学评分。

**💡 创新点**

创新点包括将预训练图像美学头作为帧级先验来指导峰-终聚合、设计审美节奏编码捕捉时间演化、以及动态门控融合提升跨域鲁棒性。

**🔧 技术方法**

使用冻结的ViT-L/14视觉编码器、图像美学预训练头、峰-终权重聚合、1D卷积节奏编码以及可学习门控机制。

**📊 数据集**

在VADB（专业评审的10,490段视频）进行训练与评估，并在DIVIDE-3K（3,590段公开视频）进行零样本跨域测试。

**📈 对比分析**

与FastVQA、SimpleVQA、ModularBVQA、CLIPVQA、Q-Align、DOVER及VADB-Net等基线对比，Peak-End-Net在VADB上取得最优的RMSE、SRCC、PLCC、KRCC，并在DIVIDE-3K零样本评估中实现最高的SRCC/PLCC/KRCC，证明其优越的泛化性能。

**⚠️ 局限性**

局限性在于仍依赖预训练图像美学模型和固定ViT，且跨域提升伴随在源域轻微性能下降，缺乏对更细粒度视频美学特征的显式建模。

---

## 408. Discriminative Barrier Functions for Safe Adversarial Imitation Learning from Observation

**arXiv ID:** 2607.13938 | [PDF](https://arxiv.org/pdf/2607.13938v1)

**作者:** Anubhav Vishwakarma `[一作]` (University of Washington), Tyler Han `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了Discriminative Barrier Functions（DBF）框架，在逆向强化学习（IRL）从观察的基础上约束奖励函数搜索，利用控制障碍函数（CBF）实现无动作标签的安全在线模仿学习。

**💡 创新点**

创新点在于将CBF约束直接嵌入到对抗式模仿学习（AIL）中，联合学习可解释的障碍函数与模仿策略；实现从无标签专家观察中恢复安全边界；并在模拟与真实机器人环境中展示零碰撞的无经验部署。

**🔧 技术方法**

使用技术包括：对抗式模仿学习（GAIL、AIRL、MPAIL）、Wasserstein GAN与梯度惩罚、离散时间CBF约束、ϵ-net假设、规划基MPPI与策略基PPO等；同时在学习过程中引入离散CBF动态约束和K类函数。

**📊 数据集**

数据集主要为：在MUSHR/Isaac Lab模拟环境中收集的状态‑仅专家轨迹，随机障碍配置；以及小规模真实机器人观测数据，用于零经验部署；未使用公开标准数据集。

**📈 对比分析**

与传统AIL基线（无CBF）相比，DBF方法在模拟障碍规避任务中实现了零碰撞、降低了成本率，同时保持或提升了任务奖励；在真实机器人实验中，DBF‑MPAIL与DBF‑GAIL同样达成零碰撞，展示了更稳健的安全性和可解释性。

**⚠️ 局限性**

局限性包括：安全信号仅存在于判别器中，未直接塑造策略价值函数；若专家数据覆盖范围狭窄，可能产生过度保守；K类函数手工设定，缺乏自适应学习；缺乏不确定性建模与主动数据采集策略。

---

## 409. Pezego-HITL: A policy-grounded large language model architecture for agricultural extension in Ghana

**arXiv ID:** 2607.13934 | [PDF](https://arxiv.org/pdf/2607.13934v1)

**作者:** Shunbao Li `[一作]` (University of Sheffield), Po Yang `[通讯]` (University of Sheffield)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一套面向小农户的政策驱动型、可人机协作的农业决策支持架构Pezego-HITL，结合结构化检索、约束审核与专家验证缓存；

**💡 创新点**

创新点在于：①将政策约束嵌入检索+审核流程，实现安全合规；②引入Verified-Case Memory (VCM) 缓存实现子秒级响应并显著降低算力；③构建P-EVAL评估协议，将安全、效用、延迟与人工监督成本统一为多目标优化；

**🔧 技术方法**

技术包括：大型语言模型（GPT-5.5等）、结构化检索+SQL工具、约束审计层、多代理对话、自动化LLM-as-a-Judge评估、向量检索/嵌入、SQLite+Chroma缓存；

**📊 数据集**

数据集：模拟农事查询数据库1240条小农案例、135条专家验证案例、30名扩展服务官员与36名农户问卷；

**📈 对比分析**

与四类基线（直接LLM、向量RAG、工具RAG、批判式RAG）对比，Pezego-HITL在专有模型下PAR 0.94、AUR 0.95、P95延迟12.9s（比批判式降低55%），在开源模型下PAR 0.86、AUR 0.88、P95延迟10.2s；

**⚠️ 局限性**

局限性：需网络环境支持，缓存依赖客户端同步；仅在Ghana东部与阿散蒂两区试点，未覆盖多样生态和作物；对高阶政策约束（如育种、出口）支持不足；

---

## 410. VAIOM: Continuous-Input, Discrete-Output Decoder-Only Financial Sequence Modeling

**arXiv ID:** 2607.13929 | [PDF](https://arxiv.org/pdf/2607.13929v1)

**作者:** Yiming Ma `[一作]`, Xinyu Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种基于 decoder‑only Transformer 的 VAIOM 模型，输入为连续的金融事件向量，输出为下一小时波动归一化回报的离散桶分布，用于概率回报预测。

**💡 创新点**

创新点在于将连续输入与离散输出分离；引入混合市场状态（MoMS）输出头、全序列监督以及 Gap/VolReg/Ordinal 辅助任务；通过实验证明连续输入和更密集监督在金融序列中优于传统离散化输入。

**🔧 技术方法**

技术手段包括：decoder‑only Transformer、连续事件嵌入、绝对位置编码、Mixture‑of‑Market‑States（MoMS）输出、Gap/VolReg/Ordinal 辅助任务、Full‑Sequence 监督、交叉熵训练与负对数似然评估、LightGBM 对照基线。

**📊 数据集**

使用 1H FX 与金属价格数据集（Dukascopy OHLCV 数据），共 1,266,989 个有效训练窗口，按时间划分为训练（<2024‑01‑01）、验证（2024‑07‑01~2025‑01‑01）和测试（2025‑01‑01~2026‑01‑01）。

**📈 对比分析**

通过在 2025H1、2025H2 测试集上与频率、马尔可夫、单条 LightGBM 基线比较，VAIOM 在两期分别压缩 0.029–0.043 bits/事件；在 3 个随机种子上均保持此优势，且在连续输入、Full‑Sequence 监督、MoMS 以及辅助任务的组合下进一步提升 NLL。

**⚠️ 局限性**

局限性包括：仅在 1H FX/金属单一市场上验证；未涵盖其他资产类别、不同频率或更长历史；对照基线仅为单条 LightGBM，未包含滞后堆叠或序列树基线；离散桶化可能削弱尾部信息；未评估实际交易收益或经济可解释性。

---

## 411. Generative Compilation: On-the-Fly Compiler Feedback as AI Generates Code

**arXiv ID:** 2607.13921 | [PDF](https://arxiv.org/pdf/2607.13921v1)

**作者:** Niels Mündler-Sasahara `[一作]` (ETH Zurich), Jingxuan He `[通讯]` (University of California, Berkeley)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了生成式编译（Generative Compilation）框架，将编译器反馈实时提供给LLM生成过程

**💡 创新点**

核心创新是Sealor概念——一种轻量级语法引导的转换器，可将部分程序封闭为完整程序，让现有编译器在生成时即给出诊断

**🔧 技术方法**

采用Sealor、Rust编译器（rustc/rust-analyzer）和LLM接口，结合Python+Rust实现；在Lean中对核心Rust子语言进行形式化和证明

**📊 数据集**

在两类Rust编码任务上评估：C-to-Rust 翻译（CRUST-Bench子集）和API适配（更新库函数的命令行工具）

**📈 对比分析**

与无反馈和仅后置编译反馈做对比；生成式编译显著降低错误率（最高从65.9%降至13.1%）并提升功能正确性，且平均运行时比仅后置反馈快约30%

**⚠️ 局限性**

限制在于Sealor需手工设计、仅支持已知语言语法；对未来依赖项错误的抑制可能导致误报；无法替代全局约束解码（constrained decoding）

---

## 412. AIMO Interpretability Challenge

**arXiv ID:** 2607.13899 | [PDF](https://arxiv.org/pdf/2607.13899v1)

**作者:** Michal Štefánik `[一作]` (National Institute of Informatics), Pontus Stenetorp `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了一项AIMO解释性挑战赛，评估前沿数学语言模型在面对分布偏移时的鲁棒性。

**💡 创新点**

创新点在于首次将解释性方法与高阶推理鲁棒性评估结合，构建开放的符号推理链及对抗性问题集。

**🔧 技术方法**

采用符号推理链生成、链式思考追踪、探测器分类器、置信度分析及特征归因等技术。

**📊 数据集**

使用180道奥林匹克级数学题（AIMO、JMO、AIME），以及对应的符号推理链和对抗变体。

**📈 对比分析**

基线方法在验证集上达58–69%准确率，表明任务可行但仍有提升空间；最终评测采用保持准确率。

**⚠️ 局限性**

局限在于对抗性分布受限、仅覆盖已标注模型、易受过拟合影响，以及对真实部署鲁棒性验证的进一步需要。

---

## 413. RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks: Architecture, Representation, and Hardware Validation

**arXiv ID:** 2607.13897 | [PDF](https://arxiv.org/pdf/2607.13897v1)

**作者:** Abdallah Aaraba `[一作]` (Polytechnique Montréal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了量子厨房锅（Quantum Kitchen Sinks, QKS）在无线频谱图（RF spectrogram）异常检测中的应用，提出了多深度重上传与环形纠缠的 QKS 模板，并在实测 LTE 信号构造的异常检测数据集上进行系统评估。

**💡 创新点**

创新点在于：①将 QKS 从传统单层扩展为多层重上传并加入可选的环形纠缠；②设计了五阶段消融协议，能在不泄漏测试集的前提下分离架构、深度、输入表示与读出等因素；③在 IBM Quebec 量子处理单元上进行真实硬件验证，验证了模拟与实测结果的高度一致性。

**🔧 技术方法**

使用技术包括：量子厨房锅（QKS）多深度重上传、CZ 环形纠缠；经典线性读出（SVM、Logistic 回归）；近似核提升（Random Fourier Features、Nyström）；以及三种输入表示（原始、DCT、PCA）。

**📊 数据集**

使用的数据集基于真实子 6 GHz LTE 信号，通过 Python 生成的三类异常（chirp、barrage jamming、frequency‑hopping jamming）叠加，最终得到 21,600 训练样本和 8,124 测试样本的 400×400 频谱图。

**📈 对比分析**

通过五阶段消融和最终全量级对照实验，QKS 在 DCT 表示下相较于直接读出实现了 AUROC 从 0.75 提升至 0.8778，F1 从 0.72 提升至 0.7995；与传统基线相比，线性读出得到最大性能提升，近似核读出也能保持一定增益。

**⚠️ 局限性**

局限性包括：仅在受控合成异常上验证，未测试对未知干扰的泛化；实验只用单一随机种子；读出仅限于快速线性或近似核模型，未探讨更复杂分类器；以及硬件实验规模受限，未覆盖更大规模 QKS 架构。

---

## 414. AI-Augmented Adaptive Digital Twin Modeling for Brain Tumor Evolution Prediction and Treatment Scheduling

**arXiv ID:** 2607.13877 | [PDF](https://arxiv.org/pdf/2607.13877v1)

**作者:** Wenxi Liu `[一作]` (Florida Institute of Technology), Xianqi Li `[通讯]` (Florida Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了AI增强的自适应数字双胞胎框架，用于脑瘤演化预测与治疗调度，融合了可解释的反应扩散模型、3D残差学习与模型预测控制。

**💡 创新点**

创新点在于将反应扩散机制与3D残差网络结合，并在递归预测中实现在线自适应更新，使模型在长时程保持高精度，同时与MPC耦合实现约束治疗调度。

**🔧 技术方法**

采用反应扩散方程、3D U‑Net残差学习、在线自适应更新、场景稳健模型预测控制（MPC）等技术，并用体素级误差、Dice、PSNR等指标评估。

**📊 数据集**

使用UPENN‑GBM公开数据的解剖与肿瘤初始图像生成387条基于控制的合成肿瘤轨迹（120步）作为训练与测试数据集。

**📈 对比分析**

与仅反应扩散基线对比，残差模型将 voxel‑wise MSE 降低84.3%，Dice 提升43.5%；在线更新进一步将 MSE 降低45.9%；MPC调度比固定方案终末肿瘤负荷低22.4%，但累计负荷略高。

**⚠️ 局限性**

局限在于仅在合成模拟数据上验证；稀疏观测下在线更新效果不佳；未考虑真实临床观测噪声、非结构化治疗记录和真实毒性约束，需进一步临床与动物验证。

---

## 415. AI-Augmented Human Resource Management? Insights from German companies

**arXiv ID:** 2607.13839 | [PDF](https://arxiv.org/pdf/2607.13839v1)

**作者:** Yannick Kalff `[一作]` (HTW Berlin University of Applied Sciences), Katharina Simbeck `[通讯]` (HTW Berlin University of Applied Sciences)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对德国企业中人工智能在HRM中的应用与影响进行定性访谈、焦点小组与大规模问卷调查，探讨其在自动化与提升HR职能中的角色。

**💡 创新点**

首次系统梳理了AI在德国HRM中的“双重”效应——自动化与协同增强，并揭示了共同决定制度、数据治理与组织结构如何共同塑造其采用与实践。

**🔧 技术方法**

采用混合方法：半结构化访谈、焦点小组讨论、在线问卷；使用AI工具如大型语言模型（ChatGPT）、聊天机器人、预测分析与数据可视化平台进行案例分析。

**📊 数据集**

主要数据来源为410名德国HR经理的问卷（包含组织背景、AI使用情况与动机），以及14位专家访谈与3组共同决定委员会讨论的文本记录。

**📈 对比分析**

通过对定性内容进行解释性分析与定量描述性统计相结合，验证AI工具在招聘、人才管理、绩效评估等HR职能中的使用频率与效果；未进行算法性能对比，但显示不同工具在组织层面获得的接受度与效益差异明显。

**⚠️ 局限性**

研究局限在于仅覆盖德国，且采用横断面自报数据，可能受参与者偏差与社会期望影响，缺乏纵向跟踪与跨国比较，且未对AI模型的技术细节与性能指标做实证检验。

---

## 416. PlumeQuant: Uncertainty-aware consistency assessment of methane plume masks and emission-rate estimates

**arXiv ID:** 2607.13945 | [PDF](https://arxiv.org/pdf/2607.13945v1)

**作者:** Parisa Masnadi Khiabani `[一作]` (University of Oklahoma), Charles Nicholson `[通讯]` (University of Oklahoma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 PlumeQuant 框架，用于评估 EMIT 源产生的 Carbon Mapper 甲烷云雾光谱仪气体泄漏产品的内部一致性，包括重新计算 plume 指标、分析掩膜敏感性、构建等势掩膜集以及不确定性传播。

**💡 创新点**

创新点在于：①提供了一个透明、可复现的流程，利用源信息和增强光谱重算 plume 指标而不依赖任何“真值”掩膜；②设计了 CM‑like 动态阈值掩膜作为基准，并通过遗传算法寻找满足公布 IME 与 plume 长度的等势掩膜；③通过等势掩膜集生成 footprint‑confidence 地图，可视化掩膜不确定性；④将多源不确定性（风速、检索、掩膜、长度、质量转换）分解并合并。

**🔧 技术方法**

技术手段包括：基于 DEAP 的遗传算法搜索掩膜、基于阈值与形态学的 CM‑like 掩膜、四次误差传播（风速、检索、掩膜、长度、转换因子）求解 Q 的不确定性、IoU、Dice、Z_Q 等一致性指标，以及 HRRR 气象场的空间插值。

**📊 数据集**

使用的数据集为 63 条来自 EMIT 观测的 Carbon Mapper 甲烷 plume 记录（涵盖德州、新墨西哥、俄克拉荷马 Permian Basin 区域，时间段 2022‑08 至 2023‑08），包含增强光谱光栅、源位置、参考掩膜、检索不确定性光栅、HRRR 10 m 风速等。

**📈 对比分析**

比较方法：对每条 plume 在 CM‑like、GA 以及参考掩膜下分别重算 IME、plume 长度和排放速率，并与公开发布的数值进行百分比差异、平均绝对误差、IoU/Dice、Z_Q 等指标评估；不确定性重构与公开不确定性进行比值和绝对差异比较。性能方面：CM‑like 掩膜下，median IME 差异 +0.72%（MAE 16.7%）、plume 长度 +0.73%（MAE 16.3%）、排放速率 +0.16%（MAE 6.98%）；median IoU 0.843，Dice 0.915；不确定性重构 median ratio 1.01，median 绝对差异 10%。

**⚠️ 局限性**

局限性：①仅评估 EMIT 传感器在 Permian Basin 区域的 63 条 plume，样本量有限且未涵盖不同传感器、地形或大气条件；②GA 搜索条件依赖已公布的 IME 与 plume 长度，不能提供真正的“真值”掩膜；③不确定性模型忽略了检索模型误差、垂直风场、源间歇性及运输误差；④footprint‑confidence 仅为 R‑条件探索性诊断，非概率分布；⑤未与控制释放实验或现场观测进行独立验证。

---

## 417. HORCRUX: A Complete PQC RISC-V eXtension Architecture

**arXiv ID:** 2607.13939 | [PDF](https://arxiv.org/pdf/2607.13939v1)

**作者:** Alessandra Dolmeta `[一作]` (Politecnico di Torino), Guido Masera `[通讯]` (Politecnico di Torino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了名为HORCRUX的RISC‑V指令集扩展，统一支持NIST公布的所有后量子密码算法，并在FPGA/ASIC上验证其性能和能耗。

**💡 创新点**

通过共享内核（Keccak、NTT、乘法器、采样等）实现跨族的硬件共享，采用紧耦合协处理器与CV‑X‑IF接口，既保持低面积（<21k LUT）又实现多算法加速，首次实现单一扩展覆盖所有五类PQC。

**🔧 技术方法**

采用共享乘法树、双模式乘法器、内部寄存器文件、Keccak‑f[1600]并行实现、离散傅里叶变换（NTT）以及采样器指令集，并结合Core‑V‑eXtension Interface实现指令级调用。

**📊 数据集**

基于NIST公开的已知答案测试（KAT）集，对所有算法进行周期级基准和能耗测量。

**📈 对比分析**

与纯软件实现（RV32IMACB）在同一RISC‑V核心上进行周期对比，得到hash‑based最高129×加速、代码基准27×、格子基准9.17×；FPGA实现面积约20k LUT，ASIC 115.8kGE，能耗显著低于软件。

**⚠️ 局限性**

受限于32位总线和单周期组合逻辑，导致频率低（≈42 MHz）以及对64位浮点操作的多周期切换，且当前实现未加入侧信道防护与时钟门控。

---

## 418. A novel unsupervised machine learning strategy to handle multimodal cardiac PET/MRI data

**arXiv ID:** 2607.13936 | [PDF](https://arxiv.org/pdf/2607.13936v1)

**作者:** Brunnhilde Ponsi `[一作]` (Nantes Université), Hatem Necib `[通讯]` (Nantes Université)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了一种双步骤无监督聚类方法，利用 PET/MRI 多模态数据自动识别ALVC患者心肌异常区域并生成量化报告。

**💡 创新点**

引入双步骤超体素聚类结合异常评分系统，利用参考健康簇实现无对照组的异常判定，并将结果可视化为 Bull's-eye 报告。

**🔧 技术方法**

3D Slicer 预处理、SLIC 超体素分割、Box–Cox 变换、谱聚类、ANOVA/Tukey 统计、异常评分与报告生成等技术。

**📊 数据集**

99 名遗传确诊的 ALVC 患者 PET/MRI 数据（含 T1/T2 映射、LGE、18F-FDG PET）以及 167 个模拟不同噪声水平的数值幻影。

**📈 对比分析**

将聚类报告与心脏影像学家手工 segment‑level 评估进行平衡准确率比较，患者组 BA=0.76±0.04，幻影组 BA≥0.80；全数据训练后 BA=0.81，灵敏度 0.76，特异度 0.86。

**⚠️ 局限性**

参数手工设定、无健康对照组导致参考簇定义不确定、注册误差、患者样本量有限以及仅评估高值而忽略低值等局限。

---

## 419. SIVA-RL: Sensitivity-Invariance Visual Alignment for Multimodal Reinforcement Learning

**arXiv ID:** 2607.13931 | [PDF](https://arxiv.org/pdf/2607.13931v1)

**作者:** Cheng Tang `[一作]` (Shanghai Artificial Intelligence Laboratory), Ming Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于样本级结果评估的可解释奖励强化学习框架 SIVA‑RL，利用局部 PatchSwap 并通过冻结的审计策略对视觉干预效果进行软路由，从而实现敏感度与一致性双向视觉对齐。

**💡 创新点**

创新点在于将干预操作与监督目标解耦，采用样本级奖励降幅来决定每对干预样本的对齐方向与强度，克服了传统按操作类型分配监督的同质性假设。

**🔧 技术方法**

主要技术包括距离约束的 PatchSwap 视觉干预、冻结审计策略评估、基于奖励降幅的软路由权重、以及基于 clean‑anchor 的敏感度与不变性对齐损失。

**📊 数据集**

使用了包含数学、逻辑、计数和视觉依赖任务的九个多模态推理基准，包括 Geo3K、MathVista、We‑Math、MMK12、MathVerse、LogicVista、Counting、MMMU‑Pro 与 MathVerse‑V。

**📈 对比分析**

在 GRPO 与 DAPO 两种 RL 背骨上，以 3B/7B 规模模型实验，SIVA‑RL 对比基线提升了 6.8%–14.9% 的相对性能，在全部基准上的平均准确率成为 7B 系列模型中的最高或次高水平。

**⚠️ 局限性**

局限性包括对干预策略和阈值的超参数敏感、对审计策略的依赖、以及对某些视觉依赖任务仍可能无法完全捕获所有视觉证据；此外，方法在大规模部署时会带来额外的计算与存储开销。

---

## 420. S-squared-VLA: Decoupling Semantic and Spatial Streams in Vision-Language-Action Models for Autonomous Driving

**arXiv ID:** 2607.13926 | [PDF](https://arxiv.org/pdf/2607.13926v1)

**作者:** Jianguo Yu `[一作]` (Wuhan University of Technology), Liping Lu `[通讯]` (Wuhan University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出S^2‑VLA框架，通过解耦语义流与空间流实现端到端的安全可控轨迹规划；

**💡 创新点**

将语义与空间信息分离并在双流规划适配器中融合，利用多尺度语义特征与未压缩的空间特征，解决传统VLM的语义‑物理鸿沟与空间表示坍塌问题；

**🔧 技术方法**

采用InternVL3‑2B + ViT 的多模态Transformer，稀疏多尺度抽样，辅助地图与Agent预测任务，双流注意力融合，LoRA微调与三阶段训练策略；

**📊 数据集**

在NAVSIM闭环仿真基准上训练，并使用ReCogDrive VQA 数据集进行预训练；

**📈 对比分析**

在NAVSIM navtest 上仅用单目前视摄像头，S^2‑VLA 在 PDMS 87.1、NC 98.4 的表现位列所有SFT VLA/VLM 方法之首，并超越传统 E2E 与 LiDAR 融合模型；

**⚠️ 局限性**

双流融合与未压缩空间特征导致计算量较大，未来需开发稀疏特征机制或结合强化学习以进一步提升效率与性能。

---

## 421. Thresholded Cross-Attention for Reliable Intensity-Chromaticity Fusion in Low-Light Image Enhancement

**arXiv ID:** 2607.13925 | [PDF](https://arxiv.org/pdf/2607.13925v1)

**作者:** Yanyi Wu `[一作]` (Guangdong University of Technology), Huan Zhang `[通讯]` (Guangdong University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于阈值交叉注意力的低光图像增强网络TCA-Net，以可靠的强度-色度融合提升图像质量。

**💡 创新点**

创新点在于用固定置信阈值替代Top-K稀疏注意力，实现输入层自适应的交叉流关联筛选，并配合相位引导频域初始化和残差强度引导复原三大模块。

**🔧 技术方法**

技术包括HVI色彩空间分解、Phase-guided Fourier Interaction Module、Thresholded Cross-Attention、Decoupled Dual-Stream Guidance Module以及Scale-Aware Consistency Regularization。

**📊 数据集**

实验使用LOL-v1/v2、Sony-Total-Dark、LSRW-Huawei等五个低光增强基准数据集。

**📈 对比分析**

与十五种主流LLIE方法对比，TCA-Net在PSNR/SSIM/LPIPS等指标上获得第一或第二名，色彩误差显著下降，参数量仅2.75M，推理速度在中等水平。

**⚠️ 局限性**

局限在于阈值固定不随图像或层自适应，且在极端噪声或颜色失真场景下仍可能遗漏有用交互；未来可探索自适应阈值或更深层次的颜色一致性约束。

---

## 422. An Efficient Newton Algorithm for Nonnegative Matrix Factorization with the Kullback-Leibler Divergence

**arXiv ID:** 2607.13919 | [PDF](https://arxiv.org/pdf/2607.13919v1)

**作者:** Damien Lesens `[一作]` (ENS de Lyon), Bora Uçar `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对 KL 失真非负矩阵分解的新算法 KL‑HALS，该算法利用目标函数的二阶泰勒展开构造非可分的二次近似，并在此近似上求解非负约束下的最优更新。

**💡 创新点**

创新点包括：①证明 MU 算法使用的可分上界已是最紧的，进而推动使用非可分二阶近似；②将 HALS 的行更新思想推广到带有不同二阶矩阵的通用版本 GHALS；③结合自共形性理论给出可计算的步长保证收敛；④在多种数据集上展示该方法在 KL 失真下的性能优势。

**🔧 技术方法**

核心技术：非负矩阵分解 (NMF)；Kullback–Leibler 损失；Majorization‑Minimization 与自共形性；二阶泰勒展开与二次近似；Generalized HALS (GHALS) 行更新；自共形性步长（1/(1+λ_k)）与全 Newton 步；稀疏矩阵高效实现。

**📊 数据集**

实验数据集包括：
- 语音/音频谱（MAPS、Louis Armstrong）、
- 图像数据（MIT‑CBCL、ORL、Frey、Urban hyperspectral）、
- 文本词频矩阵（CLUTO、Verb），
- 低秩与全秩合成矩阵（随机生成、Poisson/高斯噪声）。

**📈 对比分析**

与 MU、FPA、CCD、SN、AmSOM、AMUSOM、HALS 等现有方法在同一初始点下进行对比。KL‑HALS 在大多数数据集上达到最低 KL 损失，速度快于多数对手，尤其在合成低秩和文本数据中表现突出；在音频谱上表现略逊于 MU，且在高秩时收敛变慢。KL‑HALS‑descent（带自共形步长）收敛保证但速度更慢。

**⚠️ 局限性**

局限性：
- 迭代成本为 O(M·N·R²)，相比传统 O(M·N·R) 方法在高秩场景下效率下降；
- 依赖完整 Hessian 计算，导致对大型稀疏矩阵的实现复杂；
- 在高秩或特定音频数据上收敛速率仍低于 MU；
- 需要良好初始化（如 Sinkhorn 归一化）才能避免陷入无效解。

---

## 423. Relevance-Aware Rule: Structural Deletion of Irrelevant Conditions in Decision Trees

**arXiv ID:** 2607.13874 | [PDF](https://arxiv.org/pdf/2607.13874v1)

**作者:** Jung-Sik Hong `[一作]` (Seoul National University of Science and Technology), Sangheum Hwang `[通讯]` (Seoul National University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于结构信息的决策树规则简化方法，能够在保持预测可靠性的同时移除多余条件。

**💡 创新点**

通过揭示二叉分裂导致的类比例互补关系（C1/C0链接）并结合叶子相对匹配/不匹配判定，形成了理论基础和两种删枝策略（广义匹配引导与同类兄弟证书）。

**🔧 技术方法**

使用结构诊断（C1/C0 注释、匹配/不匹配、同类兄弟子树）、可靠性评估（基于训练样本的类别可靠度变化）以及联合硬等价或两侧容差判定；实现为四种方法（M1-D/M1-P/M2-D/M2-P）。

**📊 数据集**

在两张决策表（Weather、Human‑ID）以及八个UCI/OpenML二分类数据集（Adult、Backache、Cancer、EEG、German、Heart‑H、Ionosphere、Spambase）上进行评估。

**📈 对比分析**

与传统的IzZa路径冗余测试和Quinlan经验式后置修剪相比，M1-P 在保持几乎无准确度、类别精确度和召回率偏差的前提下，约三倍地删除条件；M2-P 与IzZa在准确度和冲突率上完全一致且速度更快；Quinlan 速度最快但产生大量冲突和类别偏差。

**⚠️ 局限性**

主要限制在于对阈值 ϵ 的依赖、对训练样本可靠度估计的敏感性，以及方法在极大树深或高维稀疏数据上的计算复杂度可能仍不理想。

---

## 424. Heavy-Tailed Flow Matching via Random Clocks

**arXiv ID:** 2607.13841 | [PDF](https://arxiv.org/pdf/2607.13841v1)

**作者:** Zhouhao Yang `[一作]` (Johns Hopkins University), Haoyang Cao `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种通过随机时钟建模重尾源的流匹配框架（HTFM），将重尾分布视为时钟条件下的高斯尺度混合；

**💡 创新点**

创新点在于引入路径值随机时钟作为隐变量，用截断logsignature特征对其进行编码，实现对不同重尾家族（高斯、α‑稳定、Student‑t）统一的流匹配；

**🔧 技术方法**

结合流匹配、随机时钟、Gaussian尺度混合、截断logsignature特征、条件流匹配、直线流和ODE求解器等技术；

**📊 数据集**

使用了3个数据集：2D不平衡α‑稳定混合、CIFAR10‑LT长尾图像和HRRR VIL天气场；

**📈 对比分析**

与传统高斯流匹配以及LIM、DLPM/DLIM、t‑Flow等重尾扩散基线在模式覆盖、FID、尾部统计等指标上进行比较，HTFM在低NFE下显著降低FID、提升模式覆盖和尾部恢复效果；

**⚠️ 局限性**

局限性包括对时钟族的设计仍需手工选择，重尾效果受时钟形状而非仅尾部指数影响；目前主要针对对称重尾分布，非对称情况与更高维任务的扩展仍待研究。

---

## 425. Earthquaker-AI: A Retrieval-Augmented Generation Framework with Rubric-Based Assessment for Primary School Earthquake Education

**arXiv ID:** 2607.14046 | [PDF](https://arxiv.org/pdf/2607.14046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 426. Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0

**arXiv ID:** 2607.14004 | [PDF](https://arxiv.org/pdf/2607.14004v1)

**作者:** Wenxiao Wang `[一作]` (RELAI.ai), Soheil Feizi `[通讯]` (RELAI.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了基于两阶段连续学习的代理优化评估协议，评估优化器在新任务出现后能否保持并提升性能。

**💡 创新点**

发现回归控制（防止对已学任务的退化）内置于搜索循环的优化器能实现性能的累积提升，首次将此机制与连续学习场景结合。

**🔧 技术方法**

对比了 GEPA、Meta Harness 与 RELAI‑VCL 三种优化方法，重点研究了回归控制的实现方式，并用 GPT‑5.5 作为提议模型。

**📊 数据集**

使用 Terminal‑Bench 2.0 中的 22 个硬任务（前 12 个为阶段 1，后 10 个为阶段 2）作为评估基准。

**📈 对比分析**

在三阶段评估中，RELAI‑VCL 在所有指标上均领先：相对基线提升约 16.7、转移 15.9、最终 15.9，终身平均 76.4%，而 GEPA、Meta Harness 分别为 66.0% 与 64.6%。

**⚠️ 局限性**

评估任务相互独立、可重复执行，缺乏跨任务相关性与真实生产环境中的有限可观测性与不完整反馈；因此结果可能不完全适用于实际部署的连续学习场景。

---

## 427. TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents

**arXiv ID:** 2607.13988 | [PDF](https://arxiv.org/pdf/2607.13988v1)

**作者:** Leitian Tao `[一作]` (University of Wisconsin--Madison), Sharon Li `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无需额外评判器或批评器的工具调用级别信用分配方法（Turn-level Reward Assignment via Credit Estimation），利用冻结的参考模型计算每一步前缀的答案预测概率，进而通过时间差分产生密集奖励，使长周期代理训练更具信息量。

**💡 创新点**

核心创新在于：①使用冻结的参考模型作为稳定的价值估计器，而非训练额外的价值网络；②将答案预测概率转换为对数比值状态值，得到可归约为终点差值的时间差分信用；③将该密集信用与传统的终点优势相结合，兼顾最终正确性与中间进展，从而显著提升长周期工具使用效果。

**🔧 技术方法**

技术手段包括：冻结参考模型（相当于预训练模型的固定拷贝）对前缀答案概率求对数；构造对数比值状态值 V(S)=log(d0/dk)；计算一阶或 K 步时间差分 δ_k = V(S_{k+1})-V(S_k)；在策略梯度更新中将 δ_k 与组相对优势 A^out 结合，并采用 GRPO 的剪切策略梯度实现；同时使用 KL 正则化保持与参考策略一致。

**📊 数据集**

数据集主要是人工构造的多文档搜索任务（Synthetic Multi-Document Search）来自 OpenResearcher 离线语料库，用于训练；评估时使用闭源 Web 搜索基准 BrowseComp-Plus 以及公开 Web 搜索基准 BrowseComp、GAIA、xbench-DeepSearch（中文 QA），其中前者使用离线检索索引，后者使用 Serper API。

**📈 对比分析**

对比方法包括：传统终点优势的 GRPO、GSPO、GiGRPO 等纯终点奖励 RL；以及外部大模型深度搜索代理（ASearcher-QwQ-32B、WebDancer-32B、CutBill-30B-A3B、TongyiDS-30B-A3B）。实验表明，在闭源 Benchmark 上，该方法将 Qwen3-4B 从 7.2 提升到 35.6，将 Qwen3-30B-A3B 从 8.4 提升到 42.6；在四个公开基准的平均值上，4B 由 29.5 提升到 34.0，30B 由 32.5 提升到 38.1，显示出显著性能提升并能迁移至不同检索环境。

**⚠️ 局限性**

主要局限在于：该信用估计方法仅适用于答案短且可验证的任务，因其依赖于黄金答案的对数概率作为价值代理；对于长文本、结构化或开放式输出的任务（如代码生成、开放式对话），参考模型的概率可能不再是可靠的进度度量，需要开发新的状态价值或进度估计方式。

---

## 428. Linear Independent Component Analysis via Optimal Transport

**arXiv ID:** 2607.14081 | [PDF](https://arxiv.org/pdf/2607.14081v1)

**作者:** Ashutosh Jha `[一作]` (University of Tübingen), Simon Buchholz `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Wasserstein距离的线性独立成分分析（OT‑ICA）方法，用来直接度量投影分量的非高斯性并实现独立源的恢复。

**💡 创新点**

创新点在于：①将W₂²距离作为全新的、无分布假设的非高斯性对比函数；②证明该对比函数在样本分布为独立成分时达到极大值；③利用梯度上升与Riemannian重投影实现一次性求解整个正交解混矩阵。

**🔧 技术方法**

使用的技术包括：Optimal Transport（Wasserstein距离）、梯度上升优化、Riemannian梯度重投影、解析高斯目标、Gaussian dithering、基于排序的经验分布匹配、对所有分量的联合优化。

**📊 数据集**

实验数据集：①多种合成混合源（连续型、混合型、离散型），②EEG Blink Artifact 数据（MNE sample dataset），③模拟三市场VECM价格发现数据。

**📈 对比分析**

与FastICA、JADE、InfoMax、Picard等基线算法比较。OT‑ICA在所有连续和混合源情形下均取得更低的Amari误差（相较FastICA降低40–45%），在全混合型源上误差降低2–4倍；在离散源上性能与基线相近或略逊，但仍保持非高斯信号；计算时间高于FastICA，随维度增长更显著。

**⚠️ 局限性**

局限性：①对离散计数数据梯度出现平坦区，导致Riemannian求解停滞；②每次迭代需要对样本排序，计算开销随维度增大；③对高维连续源的Wasserstein估计仍依赖排序，可能受样本量影响；未来改进方向包括使用Sinkhorn距离替代排序、引入可学习的OT模型以降低计算成本。

---

## 429. From Forecasts to Auditable Reports: Evidence Contracts for LLM-Assisted Housing-Guarantee Risk Monitoring

**arXiv ID:** 2607.14026 | [PDF](https://arxiv.org/pdf/2607.14026v1)

**作者:** Hyeongcheol Kim `[一作]` (Pusan National University), Yoontae Hwang `[通讯]` (Pusan National University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

将韩国Jeonse押金担保市场的月度风险预测转化为可审计的报告，采用结构化证据与LLM辅助生成。

**💡 创新点**

创新点在于：① 引入“证据合同”与基于Centered Kernel Alignment的检索机制，将模型解释与历史案例对齐；② 在Temporal Fusion Transformer上采用不对称损失（regret）训练以提升上尾事件召回；③ 设计了判定审计流程和多级证据接口，确保报告事实可验证。

**🔧 技术方法**

技术包括：不对称损失的Temporal Fusion Transformer、Centered Kernel Alignment检索、LLM（GPT‑4o、Llama 70B等）生成、结构化报告接口以及规则式审计。

**📊 数据集**

使用的数据集是韩国住房担保机构提供的月度Jeonse押金担保面板（132个区间、124个月，最终82个有效序列）及合成的聚合解释案例。

**📈 对比分析**

方法对比：与LSTM、DLinear、LightGBM及TFT平均误差模型对照；在P90尾部召回上，regret TFT达0.56，远高于其他模型的0.08；LLM报告质量在Evidence Card条件下提升至约8.7/10，结构化草稿进一步提高至≈8.9。

**⚠️ 局限性**

局限性包括：仍需人工审核、模型可能产生幻觉、对极端事件的解释有限、局限于韩国Jeonse市场、以及对合规与数据安全边界的高度依赖。

---

## 430. Lyapunov Exponent as Physics-Informed Dense Reward: RL Discovery of Stabilization Beyond the Kapitza Pendulum

**arXiv ID:** 2607.14001 | [PDF](https://arxiv.org/pdf/2607.14001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 431. Improving Wind and Solar Power Prediction with Efficient Wrapper-based Feature Selection: An Empirical Study

**arXiv ID:** 2607.14024 | [PDF](https://arxiv.org/pdf/2607.14024v1)

**作者:** Daniel Grillmeyer `[一作]` (University of Würzburg), Samuel Kounev `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文基于两大可再生能源预测任务——风电机组功率曲线建模与光伏发电预测，开展系统性文献综述和特征选择实验；

**💡 创新点**

提出一种模型无关的聚类式递归特征选择方法（Cluster‑Based Sequential Feature Selection，CSFS），通过先对特征进行聚类再做非劣势检验，显著减少wrapper方法的计算量，兼顾预测性能；

**🔧 技术方法**

利用聚类（相关系数、RF重要性、随机）+非劣势检验+后向递归特征选择，配合ML/DL回归模型（MLP、LightGBM、XGBoost）及CFO超参数搜索；

**📊 数据集**

使用公开风机SCADA数据集（5台Vestas 2MW风机）和PVOD光伏数据集（10个河北站）两类真实数据；

**📈 对比分析**

与传统SFS、过滤法（F值、MI）及RF嵌入式特征重要性进行对比；实验表明CSFS在保持与SFS相当的RMSE的同时，平均计算时间下降约21%；

**⚠️ 局限性**

局限性包括仅在单一风机和光伏站上验证，随机种子与模型选择可能影响结果，且仅以RMSE等单一指标评估。

---

## 432. AeroMap3D: Anchoring Monocular UAV 6-DoF Localization to Visual-Geometric-Semantic Map Priors

**arXiv ID:** 2607.14009 | [PDF](https://arxiv.org/pdf/2607.14009v1)

**作者:** Zhiyun Deng `[一作]` (University of Texas at Austin), Luis Sentis `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为AeroMap3D的单目UAV 6-DoF定位框架，利用公开卫星影像、裸地高程模型和OpenStreetMap建筑信息，通过轻量级尺度-偏航适配器与语义过滤实现GNSS失效环境下的地图锚定定位；

**💡 创新点**

创新点包括：①只用合成数据训练的尺度-偏航适配器，可在不同视角、尺度下恢复匹配精度；②在DEM提升前对匹配点进行语义过滤，剔除屋顶等高程误差导致的结构性偏差；③采用延迟状态EKF将稀疏地图更新与运动先验融合，实现长期轨迹稳定；

**🔧 技术方法**

使用的技术有：RoMav2（或LoFTR）等密集匹配器、MobileNetV3-Small双分支网络进行尺度/偏航预测、OSM建筑掩模、DEM高程提升、RANSAC‑PnP、延迟更新EKF、合成训练数据生成；

**📊 数据集**

数据集主要为公开的UAV‑Terra3D基准（包含NAIP卫星图、USGS DEM、OSM、实际UAV视频及GNSS轨迹），以及用于训练适配器的合成UAV‑VisLoc对；

**📈 对比分析**

在与SIFT+LightGlue、LoFTR、RoMav2等匹配器的基线、DINOv2/SelaVPR++检索、稀疏/稠密DEM‑PnP、GeoVINS式检索、ORB视觉里程计、ORB‑SLAM2、参考运动等对比实验中，适配器将匹配成功率从62.4%提升至99.2%；单帧定位成功率从88.24%提升至95.69%，平均误差下降至14 m；连续轨迹误差在55 km上平均5.88 m，成功率100%，显著降低相对漂移；

**⚠️ 局限性**

局限性包括仅在奥斯汀八个区域验证，初始化粗糙；仅有平移三自由度的地面真值，缺乏完整6-DoF校准；依赖OSM掩模，未覆盖所有建筑；对极端倾斜视角、季节性/时间变化的地图变化、以及实际IMU传感器噪声的鲁棒性仍待验证。

---

## 433. Rethinking Penetration Testing for AI-Enabled Systems: From Resource Compromise to Behavioral Objective Violation

**arXiv ID:** 2607.14006 | [PDF](https://arxiv.org/pdf/2607.14006v1)

**作者:** Mohammad Allahbakhsh `[一作]` (Ferdowsi University of Mashhad), Moslem Attar-Raouf `[通讯]` (Sensifai BV)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出AI驱动系统渗透测试的新范式，将渗透定义为诱导AI行为违背操作目标，并给出方法论和案例。

**💡 创新点**

将渗透测试从传统资源破坏转向行为目标违背；引入AI启用渗透、AI治理行为、操作目标等概念；提出目标驱动的渗透测试工作流程。

**🔧 技术方法**

结合威胁建模、攻击路径分析、Prompt/间接Prompt注入、数据投毒、传感器操纵、检索污染、工具误用等AI攻击技术，并以SOC助手为案例进行场景测试。

**📊 数据集**

论文未公开实验数据，示例使用安全运营中心助手的日志、告警、威胁情报等内部数据；未使用公开数据集。

**📈 对比分析**

未给出实验比较或性能指标，侧重理论框架与方法论设计；无性能评估。

**⚠️ 局限性**

仅为概念性框架，缺乏实测验证；对不同AI系统的适用性需进一步研究；未给出量化指标和自动化工具。

---

## 434. The Dynamic Verifiable Multi-Agent Human Agentic Loyalty Loop (DVM-HALL) Model and the Net Human-Agent Score (NHAS) in Autonomous Commerce

**arXiv ID:** 2607.13998 | [PDF](https://arxiv.org/pdf/2607.13998v1)

**作者:** Sai Srikanth Madugula `[一作]` (Woxsen University), Daya Shankar `[通讯]` (Woxsen University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并推导了Dynamic Verifiable Multi-Agent Human Agentic Loyalty Loop（DVM-HALL）模型以及可审计的Net Human-Agent Score（NHAS）指标，用于量化在自律 AI 时代下的品牌忠诚与人机协作。

**💡 创新点**

创新点在于将人类情感权益、代理机器体验、可验证执行、信任动态校准与委托度等多维因素融合进软最大化选择概率模型，并构建了风险加权的 NHAS 评价体系，首次实现对“机器客户”忠诚度的可审计度量。

**🔧 技术方法**

运用数学建模（软最大化、递归信任更新、风险权重计算）、Python 仿真框架和区块链可验证交易记录技术，同时结合控制实验、模拟市场与 DeFi 测试床进行验证。

**📊 数据集**

未提供公开数据集；验证计划依赖实验受试者购买日志、API 调用记录、可验证交易哈希、基准引擎比较结果和情感反馈，计划在多文化与不同消费群体中收集数据。

**📈 对比分析**

对比方法：将 DVM‑HALL 的品牌选择概率与传统满意度‑信任‑承诺模型进行对照；NHAS 与 NPS、净推荐值等传统指标对比；性能通过仿真统计预测准确率、信任调节效果与风险敏感度来评估，具体数值待实验验证。

**⚠️ 局限性**

局限性包括：缺乏大规模治理与信任校准工具、对安全与隐私风险的充分应对不足、实验与仿真结果尚未公开验证、跨文化适用性与法律合规性仍待进一步研究。

---

## 435. Early Adoption of Agentic Coding Tools by GitHub Projects

**arXiv ID:** 2607.14037 | [PDF](https://arxiv.org/pdf/2607.14037v1)

**作者:** Maliha Noushin Raida `[一作]` (Rochester Institute of Technology), Daqing Hou `[通讯]` (Rochester Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了 25,264 个 agentic PR 在 2,361 个 GitHub 仓库中的项目级采纳、生产率与人机协作模式。

**💡 创新点**

首次从项目层面系统评估 agentic coding 工具的采用与治理，揭示规模与协作模式的差异。

**🔧 技术方法**

利用 AIDev-pop 数据集、GitHub API、统计检验（Kruskal-Wallis、Dunn、Cliff's Delta）与参与度指标。

**📊 数据集**

AIDev-pop 2025 年 5-7 月的 agentic PR 记录。

**📈 对比分析**

对比不同团队规模的参与率、PR 数量与 PR/参与者比率；发现小型项目参与率最高，产能大部分项目低于 36 PR/参与者基准。

**⚠️ 局限性**

样本仅限 100 星以上公开仓库，缺乏对私有或小型项目的覆盖，且人机参与识别可能误差，生产率指标未考虑 PR 质量与复杂度。

---

## 436. Exploiting Graph Structure for Near-Optimal Broadcasting

**arXiv ID:** 2607.14032 | [PDF](https://arxiv.org/pdf/2607.14032v1)

**作者:** Rudranarayan Kar `[一作]` (NISER Bhubaneswar), Abhishek Sahu `[通讯]` (NISER Bhubaneswar)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统研究了电话广播问题，提出并改进了多种算法，包括精确算法、加法逼近算法和参数化逼近算法，并给出了在极简图结构（极性图）上的多项式时间解法；

**💡 创新点**

创新点在于：①将精确算法从 3^n 降至 3-f(x)^n（f(x) 为常数）；②在顶点完整性参数下实现 +2k 加法逼近；③在到团距离参数下得到 +2 加法逼近、到路径距离参数下得到 2 倍逼近；④对极性图给出 O(n^3) 的多项式解法；并且证明在若干更小参数下问题仍为 Para‑NP‑hard。

**🔧 技术方法**

主要技术包括：状态图构造与 BFS、分段轮次拆分、匹配与双向匹配的运用、动态规划、颜色编码、猜测与逼近策略，以及结构化参数（顶点完整性、到团/路径距离、极性分割）的分析与利用。

**📊 数据集**

本文为理论研究，无使用实验数据集，所有结果均基于算法设计与复杂度分析。

**📈 对比分析**

与已有工作比较时，本文在时间复杂度和逼近误差上取得显著提升，例如将 3^n 的精确算法降至 3-f(x)^n，得到更快的 +2k 逼近以及 +2、2 倍逼近的参数化算法；然而缺乏实际实验对比。

**⚠️ 局限性**

局限性包括：在顶点覆盖、支配集、直径等更小参数下仍为 Para‑NP‑hard，逼近误差仍为加法或常数因子；并且对一般图仅提供理论改进，未给出实际可行的多项式时间解法。

---

## 437. Screening Is Effective for Visual Recognition

**arXiv ID:** 2607.13983 | [PDF](https://arxiv.org/pdf/2607.13983v1)

**作者:** Shunya Shimomura `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VisionScreen，改造 Screening 机制以实现二维图像补丁的自适应选择性聚合；

**💡 创新点**

创新点在于将 2D RoPE 与空间软掩模相结合，赋予每个补丁绝对关联度并通过阈值排除低相关补丁；

**🔧 技术方法**

采用 2D Rotary Position Embedding、空间软掩模、Trim 与门控机制等技术，构建非因果的二维 Screening 模块；

**📊 数据集**

在 ImageNet‑1k 与 CIFAR‑100 两个公开图像分类数据集上进行实验；

**📈 对比分析**

与 ViT‑Tiny/16 在相同模型规模下对比，VisionScreen 在 ImageNet‑1k 提升 4.4% (72.5%→68.1%)，在 CIFAR‑100 提升 2.1% (52.4%→50.3%)，且参数量更少；

**⚠️ 局限性**

局限性在于目前仅验证分类任务，缺乏对密集预测任务的评估，且对复杂场景下的空间窗口学习机制尚未深入分析。

---

## 438. ProfMalPlus: Agent-Coordinated Detection of Malicious NPM Packages via Static-Dynamic Analysis Synergy

**arXiv ID:** 2607.13965 | [PDF](https://arxiv.org/pdf/2607.13965v1)

**作者:** Yiheng Huang `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于多智能体协同推理的恶意 NPM 包检测框架，融合静态行为图、代码切片、LLM 推理、第三方信息增强与动态补全，最终实现对恶意包的精确判定与代码定位。

**💡 创新点**

创新点包括：① 对 JavaScript 进行对象感知的统一行为图构造，解决了混淆代码、对象特性和动态特性的建模难题；② 通过多智能体（本地判断、全局判断、路由、第三方/动态增强）形成闭环推理，显著提升了判定的置信度；③ 引入自一致性验证与多源证据融合，降低 LLM 结果波动与误报；④ 在检测完成后自动定位恶意代码片段并给出解释，提升可解释性。

**🔧 技术方法**

使用技术包括：静态分析（Joern 生成 CPG，Jelly 生成调用图）、对象敏感数据流分析、动态沙箱执行（NodeProf + Docker）、LLM 推理（DeepSeek‑V4‑Flash），以及自定义的多智能体框架与动态/第三方增强模块。

**📊 数据集**

数据集：1,090 份手工整理的恶意 NPM 包（来自 Malware Bench、Backstabber’s Knife、MalOSS、OSCAR），以及 5,000+ 热门 NPM 包、Malware Bench 与 OSCAR 的无害包作为对照。

**📈 对比分析**

与 GuardDog、Cerebro、ProfMal、Malpacdetector、SocketAI 等五个最先进检测器比较，最高 F1 分数为 98.1%，比对手提升 3.5% 至 52.6%；在真实世界三个月监测中发现 597 个未知恶意包，误报率仅 16.5%；定位精度（行级 F1）达 88.9%，解释率 86.9%。

**⚠️ 局限性**

局限性包括：只能检测安装和导入阶段的恶意行为，无法覆盖不在这些阶段触发的攻击；依赖 LLM 计算成本与可用性；动态沙箱执行对极度混淆或仅在特定环境下激活的代码支持有限；第三方信息增强受 NPM 注册表数据完整性影响，可能出现信任问题。

---

## 439. VideoRAE: Taming Video Foundation Models for Generative Modeling via Representation Autoencoders

**arXiv ID:** 2607.14088 | [PDF](https://arxiv.org/pdf/2607.14088v1)

**作者:** Zhihao Xie `[一作]` (Chinese University of Hong Kong), Li Jiang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于冻结视频基础模型的自编码器 VideoRAE，能够在连续和离散潜在空间中高效重建视频并为后续的 Diffusion Transformer 与自回归模型提供语义丰富且压缩度高的潜在表示。

**💡 创新点**

创新点在于：①将冻结的多尺度 VFM 特征通过 1D 自注意力投影压缩，②引入局部‑全局 Representation Alignment（REPA）对齐解码器，消除 KL 正则；③采用多码本 SimVQ 维持高维语义信息并支持离散 token，形成统一的连续/离散生成框架。

**🔧 技术方法**

技术细节包括 1D self‑attention projector、跨层特征聚合、Multi‑Codebook SimVQ、REPA（local & global）、Transformer 解码器、GAN 感知损失、LPIPS、rFVD 评估；生成端使用 Diffusion Transformer（DiT）和 LLaMA‑style AR 模型。

**📊 数据集**

实验使用 UCF‑101、Kinetics‑600、TokenBench 进行重建评估；VideoUFO 用于文本‑视频对齐实验；VBench 用于开放域文本‑视频生成评测。

**📈 对比分析**

通过与 LTX‑VAE、LARP、SweetTok、CogVideo、TATS、VideoMAE 等 SOTA VAE/tokenizer 进行 PSNR、LPIPS、rFVD 以及 gFVD 对比，VideoRAE 在离散潜在上取得 gFVD 40（V-JEPA 2）/45（VideoMAEv2），连续潜在上 gFVD 93/99，重建 PSNR 高于大多数 baseline，且在 AR 与 DiT 训练中收敛速度约快 5 倍。

**⚠️ 局限性**

局限性在于对 VFM 预训练任务的强依赖，极细粒度运动或长时序视频的重建仍可能受限；多码本 SimVQ 的码本大小和分布需要进一步调优；文本‑视频对齐仍面临语义匹配和多模态一致性挑战。

---

## 440. Can an Old Dog Be Taught New Tricks? Taking LLMs Beyond Sentence Level Translation

**arXiv ID:** 2607.14040 | [PDF](https://arxiv.org/pdf/2607.14040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 441. From Pixels to States: Rethinking Interactive World Models as Game Engines

**arXiv ID:** 2607.14076 | [PDF](https://arxiv.org/pdf/2607.14076v1)

**作者:** Zhen Li `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于传统游戏引擎动作‑状态‑观测循环的四维框架，系统梳理并分类现有交互式游戏世界建模方法，同时构建了90小时的Black Myth: Wukong主角与Boss战游戏视频数据集，包含帧级对齐的动作、游戏状态、RGB与深度观测。

**💡 创新点**

创新点在于（1）将交互式游戏世界建模统一到四个关键维度——玩家动作控制、游戏状态动态、状态‑观测持久性与实时交互生成；（2）提供大规模带有精确游戏状态标注的数据资源；（3）从技术与数据两侧系统评估现有方法的优劣，揭示显式状态驱动和记忆机制的挑战。

**🔧 技术方法**

技术手段包括：视频生成模型（Diffusion Transformer、DiT等）、多模态动作注入（键盘/鼠标、语义事件、摄像头轨迹）、记忆机制（存储观测、估计现状）、实时生成技术（蒸馏、流式推理、硬件加速），以及数据采集与处理技术（ReShade/OBS录制、JSON状态导出、Qwen3‑VL语义与槽式标注）。

**📊 数据集**

使用的数据集是基于Black Myth: Wukong的90小时Boss战录像，分辨率1280×720、30fps，含帧对齐的原始键鼠输入、游戏状态（相机、角色、Boss姿态、动画、属性等）以及RGB与深度帧。

**📈 对比分析**

作者通过对比不同控制方式（几何轨迹、动作信号、语义事件）、状态表示（隐式、潜在、显式）以及记忆与实时生成策略，展示了各方法在视觉质量、交互一致性和延迟上的差异；虽然文中未给出统一定量指标，但指出现有模型在实时性和规则一致性方面仍存在显著不足。

**⚠️ 局限性**

局限性主要包括：仍缺乏将显式游戏状态完整地嵌入生成循环的机制；记忆更新在高频交互中难以保持精确；动作与规则时序的对齐尚未实现；数据集聚焦Boss战，缺乏更广泛的游戏场景和多样性。

---

## 442. Hindcast: Replaying Prediction Markets to Evaluate LLM Forecasters

**arXiv ID:** 2607.14051 | [PDF](https://arxiv.org/pdf/2607.14051v1)

**作者:** Xiao Ye `[一作]` (Arizona State University), Ben Zhou `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个基于过去时点t₀的前瞻性评估框架Hindcasting，限制检索在不可变的Reddit存档中进行，并使用Polymarket的已解决市场和价格作为评估基准。

**💡 创新点**

创新点在于通过冻结检索上下文来消除未来信息泄漏，利用历史市场价格作为人类基准，并通过多种覆盖与平衡策略构建可持续的评估集。

**🔧 技术方法**

采用LLM检索增强循环、语义搜索、对比式覆盖评估和Brier/准确率指标。

**📊 数据集**

使用公开的Reddit Pushshift月度快照和Polymarket历史二元预测市场数据。

**📈 对比分析**

与零射手基线对比，检索增强模型在大多数模型上将Brier降低最多23%，准确率提升，但提升程度受事件主题影响；在Sports、Awards、Trading提升显著，而Entertainment等主题下降。

**⚠️ 局限性**

局限包括数据仅覆盖英语和Reddit活跃子板块、检索档案截止到2026年1月31日导致短视窗口、只评估二元市场以及未覆盖闭源模型或多元市场。

---

## 443. Deep Interaction: An Efficient Human-AI Interaction Method for Large Reasoning Models

**arXiv ID:** 2607.14049 | [PDF](https://arxiv.org/pdf/2607.14049v1)

**作者:** Hefeng Zhou `[一作]` (Shanghai Artificial Intelligence Laboratory), Jie Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Deep Interaction 的直接编辑范式，允许用户在 LLM 的 Chain‑of‑Thought 过程中精准纠错并驱动后续推理；

**💡 创新点**

创新点在于：①允许在对话中直接修改推理文本而非仅通过多轮对话反馈；②结合差异跟踪、语义重构与去重三阶段文本处理；③利用 CoT Reprompter 将编辑后的推理精炼为可复用提示；

**🔧 技术方法**

技术主要包括：文本差异跟踪（Myers Diff）、强调标记（Markdown）、语义去重、数字脱词、CoT Reprompter（少量微调或少量样例的 LLM），以及多轮交互框架；

**📊 数据集**

使用 ScienceQA、Gaokao‑MM、LogicQA 及自构造的 STEM20K（5k 题/学科，涵盖高中到大学水平）以及 Qwen‑VL‑Max 等多模态数据；

**📈 对比分析**

与传统基于对话的纠错方法对比，在 STEM、逻辑及多模态基准上通过 Pass Rate、Correction Rate、Token Cost 等指标评估，实验显示 Deep Interaction 的 Pass Rate 在 1R 至 4R 之间平均提升 25%+，Token 消耗下降约 40%，并在不同模型规模（7B–72B）均保持优势；

**⚠️ 局限性**

局限性包括：对用户纠错质量高度依赖；若用户提供错误修正，模型可能跟随错误产生误导；仅适用于可暴露中间推理文本的系统；并且在没有足够领域知识的普通用户中效果可能不佳。

---

## 444. SPECS: Speciated Evolutionary Circuit Synthesis

**arXiv ID:** 2607.14027 | [PDF](https://arxiv.org/pdf/2607.14027v1)

**作者:** Yağız Gençer `[一作]` (Sony Group Corporation), Lorenzo Servadei `[通讯]` (Sony Group Corporation)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于遗传算法的自动模拟电路合成方法SPECS，能够同时搜索电路拓扑和元件尺寸

**💡 创新点**

引入电路原生基因编码、基于创新ID的结构差异度量与物理约束的变异算子，并利用分化演化（speciation）保护创新并控制复杂度增长

**🔧 技术方法**

遗传算法、NEAT思想、SPICE仿真评估、基于创新ID的距离度量、线性秩选择和分化演化机制

**📊 数据集**

四个计算任务：平方、立方、平方根、立方根的电压函数实现，使用21个采样点的仿真测试基准

**📈 对比分析**

与GraCo‑ES、SPICEMixer、SPICEMixer++以及ACID‑GE/ACID‑MGE等四大基准方法比较；SPECS在成功率、误差、最优适应度等指标上普遍优于或持平，且在多次独立跑实验中表现出更高的可靠性和一致性

**⚠️ 局限性**

目前仅评估了BJT和电阻元件，缺乏对MOS、电容、电感、二极管等更丰富元件库以及更复杂应用场景（放大器、滤波器、振荡器）的验证；对极大规模电路的可扩展性仍待探索

---

## 445. Transforming Rank: How Architecture Navigates the Spectral Pathologies of Depth

**arXiv ID:** 2607.14018 | [PDF](https://arxiv.org/pdf/2607.14018v1)

**作者:** Katie Everett `[一作]` `[通讯]` (Massachusetts Institute of Technology), Katie Everett (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文从梯度与表示的秩保持视角系统分析Transformer前馈块的每个组成模块，并量化其对深度网络中秩衰减、梯度路径分布和参数效率的影响。

**💡 创新点**

创新点：① 将跳跃连接和归一化视为通过梯度路径分配实现秩保留的机制；② 用分支与跳跃比例（branch-to-skip ratio）统一解释Post‑Norm与Pre‑Norm的秩行为；③ 引入宽度扩张与两矩阵结构阐释分支Jacobian秩提升与均值峰值抑制；④ 将有效秩作为预测网络可训练性的指标。

**🔧 技术方法**

技术手段：基于理论推导的路径分解、矩阵自由假设下的高斯乘积法、Marchenko–Pastur谱分布、有效秩（exponential of spectral entropy）与稳定秩评估、数值实验验证。

**📊 数据集**

主要使用的公开数据集为CIFAR‑10，用于验证初始化有效秩与训练成功之间的关联。

**📈 对比分析**

比较方法：在不同归一化位置（Pre‑Norm、Post‑Norm、Output‑Norm）以及不同分支规模、初始化尺度、宽度扩张比下，计算输入‑输出Jacobian的有效秩并与CIFAR‑10测试准确率对比；结果显示Post‑Norm在大α或深度下秩迅速坍塌，导致准确率降至接近随机；Pre‑Norm保持秩，训练成功；两矩阵结构与宽度扩张可显著提升分支Jacobian秩并稳定训练。

**⚠️ 局限性**

局限性：① 仅在初始化阶段分析秩，未深入研究训练过程中的秩演化；② 关注前馈块，对注意力层与序列维度的秩变化缺乏实验；③ 以CIFAR‑10这一易任务为验证，难以评估在更高阶或需深度推理的任务中的表现；④ 对激活函数的自由假设仅在ReLU/abs严格成立，对平滑激活仍为经验近似。

---

## 446. Edge-decomposition into Two Triangular Forests is NP-complete

**arXiv ID:** 2607.13999 | [PDF](https://arxiv.org/pdf/2607.13999v1)

**作者:** Beniamin Bibrowski `[一作]` (University of Warsaw), Tomáš Masařík `[通讯]` (University of Warsaw)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了将图的边拆分成两个三角森林是NP‑完整的

**💡 创新点**

首次把k=2的情形归入Lee等人提出的通用硬度框架中

**🔧 技术方法**

采用了基于K5块的可控性构造、可接受的边着色与NAE‑4‑SAT归约

**📊 数据集**

未使用实验数据集，仅在理论上构造多项式时间多重归约

**📈 对比分析**

无实验对比，主要是理论证明，证明了极限多项式大小构造

**⚠️ 局限性**

仅针对三角森林，未解决一般图的外厚度≤2的复杂度

---

## 447. Constraint-Aware Counterfactual Editing for Aspect-Based Sentiment Analysis

**arXiv ID:** 2607.13977 | [PDF](https://arxiv.org/pdf/2607.13977v1)

**作者:** S M Rafiuddin `[一作]` (Oklahoma State University), Atriya Sen `[通讯]` (Oklahoma State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CAVE-ABSA 框架，通过观点跨度定位、受控重写、修复、AMR 结构检查与多维约束验证，生成最小化、流畅且无矛盾的面向特定方面的对抗样本。

**💡 创新点**

将观点跨度定位、受控重写、修复、AMR 结构相似度与面向方面的验证器融合，首次实现在保持非目标方面情感不变的同时实现目标方面情感翻转的对抗生成。

**🔧 技术方法**

使用依赖+词频+可解释梯度进行观点跨度定位，基于提示的 seq2seq 受控重写，语法修复模块，AMR 解析与结构相似度检测，以及面向方面的情感验证与矛盾检测。

**📊 数据集**

在 SemEval‑2014 Laptop、Restaurant 以及 MAMS 多方面数据集上进行实验。

**📈 对比分析**

与词典替换、MLM 替换、Prompt‑only、Direct AMR 以及 Attribution+T5 等基线对比，CAVE‑ABSA 在目标翻转率 92.4%、非目标保持 91.1%、语义相似 89.3%、流畅度 91.8%、矛盾率 4.6% 及生成保留率 68.9% 方面均居首位，并在模型鲁棒性评估与数据增强实验中将对抗鲁棒性提升至 74.6%。

**⚠️ 局限性**

目前仅支持正负极性翻转，缺乏对中性或冲突标签的处理；多语言与长句的语义漂移仍需改进；对 AMR 解析准确度高度依赖，解析错误可能导致验证失效。

---

## 448. Tighter Bounds for the Random-Offerer Mechanism in Bilateral Trade

**arXiv ID:** 2607.13959 | [PDF](https://arxiv.org/pdf/2607.13959v1)

**作者:** Sunghyeon Jo `[一作]` `[通讯]`, Sunghyeon Jo

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了随机出价者机制在双边交易中的表现，给出了该机制相对第一最佳交易收益的最差比例 ρ_RO 的上下界。

**💡 创新点**

创新点在于：① 通过参数化 Lagrangian 与点状单调分配的结合，证明 ρ_RO 至少为 1/π≈0.3183；② 构造了一个可计算、经过区间算术验证的“硬”分布族，使 ρ_RO 的上界下降到 0.4602423，从而大幅收窄已知区间。

**🔧 技术方法**

使用的技术包括：参数化 Lagrangian 约束、点状单调分配与量化（quantile）铁杆化、连续到离散的格点近似、定理证明中的极限与极值分析，以及区间算术（interval arithmetic）对数值结果的严格证明。

**📊 数据集**

没有使用真实数据集；所有结果基于构造的理论分布（如截断等价收入分布、倾斜幂律尾部分布等）。

**📈 对比分析**

通过与先前的 0.317844（≈1/T）和 0.48195（1/2.0749）等结果比较，证明了新的下界 1/π 的微小提升以及上界 0.4602423 的显著下降，表明本文的界限更为紧密。

**⚠️ 局限性**

局限性：仍未确定 ρ_RO 的精确值，已知区间仍有 0.142 的差距；所用的硬分布族虽可证实上界，但可能并非最坏情况；此外，分析主要聚焦于独立值分布与 Borel 先验，其他模型（如相关性、离散分布）未被覆盖。

---

## 449. AI-accelerated End-to-End Framework for Rapid Professional Upskilling

**arXiv ID:** 2607.14044 | [PDF](https://arxiv.org/pdf/2607.14044v1)

**作者:** Tam Nguyen `[一作]` (Crew Scaler / US Federal Government), Robert Ogburn `[通讯]` (Crew Scaler / US Federal Government)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个端到端的 AI 加速快速职业再培训框架，覆盖知识获取、内容开发、审核验证、AI 辅导和评估等五个阶段，并生成符合行业认证标准的训练材料。

**💡 创新点**

创新点在于将 AI 加速贯穿整个学习生命周期、结合学习效率设计、引入结构化验证层、并通过外部行业认证和真实案例验证框架有效性。

**🔧 技术方法**

使用了大语言模型（LLM）、检索增强生成（RAG）、hallucination 检测框架（RAGAS/FAVA/VeriScore）、自动化评估生成技术以及知识图谱/依赖层级等多种 AI 技术。

**📊 数据集**

主要数据集包括约 3,000 页的多代理 AI 系统知识库、530 道 AI 生成的评估题库、1,267 条风险项以及 NVIDIA Certified Professional in Agentic AI 考试材料。

**📈 对比分析**

通过与专家创建的标准化考题进行项目响应理论（IRT）分析，AI 生成题目在难度分布和误区针对性上与专家题目相当；三名学习者在 NVIDIA 认证考试中实现 100% 通过，验证了框架的实用性。

**⚠️ 局限性**

局限性包括样本量仅为 3 名考试通过者、对知识库完整性高度依赖人工审核、仍需专业人员完成蓝图设计、SME 审核、误区编写及题目评级等高判断环节，且缺乏大规模对照实验验证。

---

## 450. Multi-Expert Routing for Multi-Domain Low-Resource OCR: A Manchu Case Study

**arXiv ID:** 2607.14041 | [PDF](https://arxiv.org/pdf/2607.14041v1)

**作者:** Zhan Chen `[一作]` (Beijing Normal University), Chih-wen Kuo `[通讯]` (National Chiayi University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套多专家路由系统，用于对历史满文文档进行页面级光学字符识别（OCR），系统通过轻量级图像分类器将每页页面分派给最合适的训练检查点（专家）进行识别。

**💡 创新点**

创新点包括：①利用已有的训练版本历史检查点作为领域专家，避免重新训练；②设计源平衡、泄漏安全的轻量域路由器；③在低资源、多域场景下实现域路由器与域标签oracle几乎相同的性能。

**🔧 技术方法**

技术方案包含：Nougat 文档序列化模型作为专家；ResNet‑18 轻量级分类器做域路由；逆频率类权重、源平衡采样、数据增强等训练技巧；基于字符错误率（CER）进行评估。

**📊 数据集**

数据集为固定的三域测试集：常规脚本125页、纪念书信10页、行书13页，来自中国第一史料馆等多来源，训练集约9,000页。

**📈 对比分析**

评价方法：在冻结的三域测试集上与域标签oracle比较，路由器域准确率99.3%；页面级CER分别为0.30%（常规）、1.57%（纪念）、4.83%（行书），远优于单一检查点的性能。

**⚠️ 局限性**

局限性：手写域测试集规模较小；专家来自不同训练快照，版本差异混合；仅细分三大域，无法覆盖更细粒度的书写风格；路由器训练依赖源标签，存在潜在泄漏风险；未针对更大域数或多专家集群进行扩展。

---

## 451. Optimizing Visibility in Generative Engines: A Critical Survey of Generative Engine Optimization (2023-2026)

**arXiv ID:** 2607.14035 | [PDF](https://arxiv.org/pdf/2607.14035v1)

**作者:** Olivier Martinez `[一作]` `[通讯]` (Sciences Po), Olivier Martinez (Sciences Po)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

回顾45篇GEO研究，提出多阶段管线、可见性向量和证据层级，并给出可复现测量协议

**💡 创新点**

系统化GEO研究，阐明可见性分布、因果估计、商业引擎差异，并提出统一的可复现测量框架

**🔧 技术方法**

文献综述、结构化编码、因果效应表述、统计检验和实验设计指南

**📊 数据集**

45篇论文中的公开实验数据，涵盖不同查询集（OpenAI 10k、SAGEO Arena、EMNLP 2024 等）以及商业引擎日志和审计数据

**📈 对比分析**

对比了检索率、引用率、位置加权文本份额等指标，发现主题相关性和位置提升最稳健；整体可见性提升有限

**⚠️ 局限性**

证据主要为上下文条件下的因果效应，缺乏对有机检索、流量和转化的长期影响；商业引擎差异大，研究可重复性受限，方法对多模型、语言和地区的普适性不足

---

## 452. Industrial Dexterity Benchmark: A Hardware-Software Benchmarking Platform for Industrial Dexterous Manipulation

**arXiv ID:** 2607.14021 | [PDF](https://arxiv.org/pdf/2607.14021v1)

**作者:** Honglu He `[一作]` (Analog Devices, Inc.), Colm Prendergast `[通讯]` (Analog Devices, Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了工业精细操作基准（IDB）硬件平台、可扩展的仿真学习框架（DAG-ROS）以及融合RGB、点云和腕力的多模态扩散策略（AG-iDP3），并在数据中心电缆清洁任务上进行验证。

**💡 创新点**

创新点包括：①从传统模块化流程向端到端多模态模仿学习转变；②将多视角RGB、点云与腕力通过模态门控统一到扩散政策；③构建可复现的硬件基准板与集成仿真学习的Docker化基础设施。

**🔧 技术方法**

使用了R3M编码器、PointNet轻量点云编码器、扩散U-Net、DAG-ROS多子系统、PyTrees行为树、CRISP阻抗控制器以及Cubic Spline插值等技术。

**📊 数据集**

基于IDB Board #1的数据集，收集约100次抓取、清洁、插入的手动演示，生成48次试验的评估数据。

**📈 对比分析**

通过6种传感器配置对比，采用多模态扩散策略（RGB手腕+RGB场景）实现78%总成功率，远优于单RGB基线的36%，表明多模态信息显著提升把握与插入性能。

**⚠️ 局限性**

局限性在于对细微视觉场景变化敏感、点云分辨率不足导致插入误差、需要更多数据增强与域随机化以提升鲁棒性。

---

## 453. Lighthouse RL: Sample-Efficient Circuit Optimization via Strategic Reset Points

**arXiv ID:** 2607.14008 | [PDF](https://arxiv.org/pdf/2607.14008v1)

**作者:** Mustafa Emre Gürsoy `[一作]` (Sony Group Corporation), Lorenzo Servadei `[通讯]` (Sony Group Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于强化学习的电路尺寸优化方法Lighthouse RL，利用灯塔状态重置策略在每个训练周期中从已发现的高性能参数配置开始搜索。

**💡 创新点**

核心创新是通过在训练过程中收集并维护一组高性能的“灯塔”状态，作为后续训练和推断的起始点，从而显著减少无效搜索，提高样本效率和对新目标的泛化能力；该策略可无缝插入任何基于RL的黑盒优化框架。

**🔧 技术方法**

使用Soft Actor‑Critic（SAC）算法配合图注意网络（GAT）处理电路图结构；对比使用贝叶斯优化（BO）、RoSeOpt、RoSeOpt细粒度版本以及固定/随机重置点的RL基线；在训练和推断阶段均采用离散/连续动作空间。

**📊 数据集**

实验数据集包括：1）二维球面多目标基准问题；2）两级运算放大器（11个可调参数）和多级运放（29个参数）的仿真模型，使用Skywater SKY130 PDK与Ngspice进行真实电路仿真。

**📈 对比分析**

与BO、RL基线、RoSeOpt及其细粒度版本进行对比；在训练分布内，Lighthouse RL样本效率提升1.55–1.72倍，成功率在两级运放上达100%（其他方法58–87%），在多级运放上87%对比BO的26%；在超出训练分布的外推任务中成功率提升至75%（其他方法<10%）；推断步骤平均下降4–6倍；在目标最大化实验中，Lighthouse RL在多级运放上实现了比人类设计更高的增益、带宽、相位裕度和增益裕度。

**⚠️ 局限性**

局限性包括灯塔状态的多样性仅由训练动态自然产生，未引入显式多样性约束；缺乏并行化探索以加速高性能配置发现；灯塔状态选择和维护机制可能需针对不同问题进行更细致的调优。

---

## 454. M$^\text{4}$World: A Multi-view Multimodal Driving World Model for Interactive Object Manipulation and Minute-long Streaming

**arXiv ID:** 2607.14005 | [PDF](https://arxiv.org/pdf/2607.14005v1)

**作者:** Ke Cheng `[一作]` (Meituan), Shuhan Shen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可控驾驶世界模型，支持多视角、LiDAR同步生成，并通过细粒度对象控制（位置、类别、视觉外观和文本属性）实现高质量、可持续的长时序生成；

**💡 创新点**

创新点在于：1）将几何对象控制与外观/文本描述融合为统一的对象令牌，实现对对象外观的精确调控；2）构建分阶段训练流程（双向中期训练、教师强制、ODE初始化、自我强制+DMD、长视频微调），从而在低延迟下实现稳定的自回归生成；3）提出“潜在上下文刷新”机制提升跨块一致性；4）提供高效的单例后训练和视觉参考条件化方案，支持稀有事件的快速定制与零射生成；

**🔧 技术方法**

使用的主要技术包括：Diffusion Transformer（DiT）为共享潜在编码器；视频VAE将多模态数据映射到同一潜在空间；多模态交叉注意力融合时间演化控制信号；自回归教师强制与ODE学生迁移；自我强制与不对称分布匹配（DMD）来消除训练-推理分布差距；LoRA微调实现单例稀有场景适配；视觉参考条件化模型；VLM评估管线；Adaptive Projected Guidance（APG）处理LiDAR；以及多尺度并行加速。

**📊 数据集**

数据集：从自有驾驶日志中采集 10 视角相机+前向 LiDAR 的 10 秒短视频与分钟级长视频；通过 BEV 感知检测挑选交通密集、恶劣天气、高速行驶等多样场景；自动标注场景级和对象级文本描述；共计 10 条短片用于中期训练与教师强制，60 条长片用于长视频微调。

**📈 对比分析**

与现有 MagicDriveV2 进行对比：FID 从 41.7 降到 34.8，FVD 从 346.1 降到 288.7；对象视觉/文本符合度分别提升至 62.7%/59.1%（比 13.4%/11.6% 低基线提升约 5 倍）；跨视角一致性提升至 84.5%（比 78.9% 更高）。生成吞吐量在 424×800 分辨率下 2.3 FPS（6 视角 + LiDAR），支持 60 秒无崩塌长回放。稀有事件增强实验中，加入 500 条合成树木牵引卡车片段后，检测召回率从 1.0% 提升至 69.7%，同时常规集 mAP 仅微幅上升（66.7%→66.8%）。

**⚠️ 局限性**

局限性包括：1）对极端稀有事件的覆盖仍有限，需扩展稀有事件税onomy；2）实验仅在离线闭环环境验证，缺乏真实闭环交互评估；3）长时序生成仍可能出现细微漂移或视觉模糊，尤其在极长回放中；4）对 VLM 判定的依赖可能带来误判或对比度限制。

---

## 455. Task-Specific Feature Fusion Method for Multi-Task Affective Behavior Analysis

**arXiv ID:** 2607.13986 | [PDF](https://arxiv.org/pdf/2607.13986v1)

**作者:** Jiajun Sun `[一作]` (Shanghai Normal University), Zhe Gao `[通讯]` (Shanghai Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在ABAW11多任务情感行为分析挑战中，本文提出任务特定特征融合方法，使用冻结的DINOv2 ViT-L和DINOv3 ConvNeXt-base特征，针对VA、EXPR和AU分别选择最优融合和预测策略。

**💡 创新点**

创新点在于任务特定融合与预测策略的选择，避免所有任务共享同一架构，系统化比较多种融合、时间建模与校准方法，并在验证集上实现最优性能。

**🔧 技术方法**

使用的技术包括自监督视觉模型DINOv2/3的预训练与冻结、特征拼接/门控/残差融合、时间卷积网络、后处理平滑、LightGBM、阈值校准及共享MTL基线。

**📊 数据集**

使用的数据库为ABAW11的官方s-Aff-Wild2图片集以及外部表情集（AffectNet+RAF-DB）用于模型预适配。

**📈 对比分析**

通过在ABAW11验证集上对比不同头部、融合方式、时间处理和MTL结构，最终任务自适应配置在EXPR、AU、VA上分别取得0.4222、0.5402、0.6717，综合得分1.6341，优于共享MTL基线1.3598。

**⚠️ 局限性**

局限性在于依赖手工验证选择专家、冻结特征未进行联合微调、对视频级跨折稳定性仍需改进，且缺乏在公开测试集上的最终评测。

---

## 456. Agent-Alternation-Free Epistemic Metric Temporal Logic with Past: Model Checking and Complexity

**arXiv ID:** 2607.13981 | [PDF](https://arxiv.org/pdf/2607.13981v1)

**作者:** Benedikt Bollig `[一作]` (Université Paris-Saclay), Paul Zeinaty `[通讯]` (Université Paris-Saclay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限的Büchi自动机上研究带过去时、严格首时间语义和度量约束的知识度量时序逻辑（KMTL）的模型检测问题，并给出了其在同步完美回忆下的EXPSPACE完备性证明。

**💡 创新点**

创新点在于首次将过去时、度量约束与知识量子统一分析，并通过构造配对状态观察者（系统状态与临时自动机状态）来解决过去公式在不可区分历史上的真值差异，从而实现了EXPSPACE上界的匹配下界。

**🔧 技术方法**

主要技术包括严格首时间语义的时序自动机构造、针对过去约束的计数器技术、知识自由子公式的Büchi自动机以及多层观察者和相对活性图的组合来实现模型检测。

**📊 数据集**

研究采用指数宽度走廊填充问题作为归约基准进行理论证明，没有使用实际实验数据集。

**📈 对比分析**

与现有的知识LTL、无知识度量逻辑相比，提出的算法在同一片段上达到EXPSPACE的匹配上下界，说明不存在更优的复杂度。

**⚠️ 局限性**

局限性在于仅处理单一代理或同一代理知识交替深度为一的情况，无法扩展到多代理交替深度>1，以及不涵盖连续时间或非同步完美回忆的情形。

---

## 457. Music-to-Dance Generation via Atomic Movements

**arXiv ID:** 2607.13978 | [PDF](https://arxiv.org/pdf/2607.13978v1)

**作者:** Xinhao Cai `[一作]` (Peking University), Yang Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种结构感知的两阶段音乐到舞蹈生成框架，先用离散扩散模型对全曲音乐进行原子动作规划，再通过连续扩散模型完成舞蹈并生成平滑过渡。

**💡 创新点**

创新点包括：①三步原子动作发现（分段-聚类-LLM重聚类）得到可解释且可复用的原子动作；②将舞蹈拆解为原子动作规划与过渡生成两个阶段，显著提升结构一致性与可控性；③结合离散与连续扩散模型实现端到端但结构可视化的舞蹈生成。

**🔧 技术方法**

使用的技术包括时序事件分割、K‑means聚类、Gemini + LLM 语义重聚类、姿态脚本（PoseScript）生成文本标签、离散 D3PM（Transformer‑based diffusion）进行原子动作规划、连续 DDPM（扩散）完成舞蹈并加入过渡损失，以及多模态音乐编码器。

**📊 数据集**

主要使用 AIST++ 3D 舞蹈数据集（1,408 条舞蹈曲目，10 种舞蹈风格）进行训练、验证与评估。

**📈 对比分析**

与 DanceNet、DanceRevolution、Bailando、EDGE、Lodge 等基线在 FID_k/FID_g、Div_k/Div_g、BAS、R 等指标上进行比较，本文方法在结构一致性、节奏对齐、运动多样性等方面均取得最优或接近最优的性能，尤其在 R（结构匹配）和 BAS（节奏对齐）上明显优于现有方法。

**⚠️ 局限性**

局限性包括：①需要大量标注的原子动作集合，LLM 生成的标签可能带来偏差；②过渡生成仍依赖手工定义的过渡损失，难以完美捕捉复杂舞蹈细节；③对实时生成和不同音乐风格的泛化能力尚待进一步验证。

---

## 458. CF-Net: Conflict Fusion with Speaker Normalisation and Certainty Weighting for Ambivalence/Hesitancy Recognition

**arXiv ID:** 2607.13976 | [PDF](https://arxiv.org/pdf/2607.13976v1)

**作者:** Tung Hung Bui `[一作]` (FPT University), Van Thong Huynh `[通讯]` (HCMC University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种用于识别视频中模糊情绪的 CF-Net 网络，专门处理“矛盾与犹豫”场景；

**💡 创新点**

创新点在于将冲突融合（ConflictFusion）与每位说话者特征归一化、置信度加权的焦点损失、辅助置信度回归等多种正则化技术相结合，解决跨模态不一致、身份泄露和标注不确定性三大难题；

**🔧 技术方法**

采用冻结的 SigLIP2、HuBERT 与 DistilBERT 作为三模态特征提取器，使用 BiGRU+MIL 注意力进行时序编码，加入冲突融合层和置信度辅助头；

**📊 数据集**

使用 ABAW 2026 年第 11 届 Ambivalence/Hesitancy Challenge 的 BAH 数据集（778 训练 / 124 验证 / 525 测试），该数据集按说话者严格划分；

**📈 对比分析**

在 BAH 验证集上实现 Macro F1=0.7155，私有测试集上得到 0.7364，优于之前在同一数据集上的最佳单模型（0.714），并且验证到测试的提升为 +0.021，表明对新说话者具有良好泛化能力；

**⚠️ 局限性**

局限在于完全冻结的大模型特征可能未能充分利用特定模态的细节，且冲突融合仅使用简单拼接，缺乏动态权重学习，未来可尝试轻量化微调或跨模态注意力来进一步提升性能。

---

## 459. DeltaMerge-LowRes: Composing Language and Task Deltas for Low-Resource Adaptation

**arXiv ID:** 2607.13967 | [PDF](https://arxiv.org/pdf/2607.13967v1)

**作者:** Son Ha Xuan `[一作]` (RMIT University), Phat T. Tran-Truong `[通讯]` (Ho Chi Minh City University Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过把语言适配的增量（Δ_L）和任务适配的增量（Δ_T）在权重空间中独立训练，然后按不同规则合并，从而在低资源环境下实现多语言多任务的快速适配。

**💡 创新点**

创新点在于提出了跨轴 TIES（cross‑axis TIES）——将 TIES 的修剪、符号投票与合并步骤从任务轴迁移到语言–任务轴；并系统评估了五种合并规则，发现跨轴 TIES 在生成类任务上显著优于传统加法与任务专用增量。

**🔧 技术方法**

使用的技术包括：LoRA rank‑16 低秩增量训练、语言增量的 MLM / span‑corruption、任务增量的基于英语数据的微调、四种合并规则（additive、activation‑guided、sparsity‑aware、cross‑axis TIES）以及基于激活偏移的层级权重分配。

**📊 数据集**

实验数据集涵盖非洲语言：MasakhaNEWS（分类）、MasakhaNER 2.0（NER）、AfriQA（抽取式 QA）和 XL‑Sum（摘要），使用 XLM‑R‑base 处理分类/NER/QA，使用 mT5‑base 处理摘要；语言增量来自对应语言的公开单语语料。

**📈 对比分析**

对比方法包括任务专用增量、加法、激活引导、稀疏合并和跨轴 TIES，全部在相同 Δ_L、Δ_T 上进行。结果显示：跨轴 TIES 在摘要（chrF + 4~7）和 QA（F1 + 2.32）上优于其他规则；稀疏合并在分类上实现 ECE −36% 的校准改进；分类与 NER 的宏 F1 基准差距不大，但 NER 的召回率提升约 3 分。

**⚠️ 局限性**

局限性包括：未与完整联合微调、LoRA 或基于适配器的 baselines 做同等算力对比；种子覆盖不均（摘要仅 3 × 1）；只测试单一模型基线和单一 LoRA 参数化；实验仅覆盖四种语言，无法说明更大规模或其他语言的可推广性；使用 chrF 作为摘要指标，缺乏人类评测。

---

## 460. VisualRepair: Dynamic Tool Calling and Region Focusing for Visual Software Issue Repair

**arXiv ID:** 2607.14075 | [PDF](https://arxiv.org/pdf/2607.14075v1)

**作者:** Jingyu Xiao `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于多模态大语言模型的视觉软件缺陷修复框架，能够通过分析缺陷报告中的自然语言描述和多种类型的视觉附件（如UI截图、IDE代码快照、GIF动画和文本图像），自动生成并验证修复补丁。

**💡 创新点**

创新点在于：①图像类型感知工具调用（ITTC）通过轻量级分类器为不同视觉内容动态选择专门的工具链（GIF关键帧提取、OCR、裁剪、代码模板），显著提升模型对异构视觉输入的理解；②动态测试时区域聚焦（DTRF）生成多粒度的bug相关区域候选并采用缩放增广，改善定位精度并扩大补丁多样性。

**🔧 技术方法**

技术方法包括：多模态大语言模型（如o3、GPT‑4o 等）与自定义工具链集成、GIF 关键帧提取（基于 MAE 阈值）、OCR（PaddleOCR）、结构化裁剪（UIED）、代码模板库、图像多候选区域定位与缩放、补丁生成与验证（编译、视觉差异检测）等。

**📊 数据集**

数据集为 SWE‑bench Multimodal（JavaScript/TypeScript 前端库），共 619 个问题实例，分为 517 个测试集和 102 个开发集，涵盖 UI、IDE、GIF、文本图像等四种主要视觉类型。

**📈 对比分析**

实验结果显示，使用 o3 基础模型时，该方法在测试集上修复 196 例（37.91%），比排行榜上最佳开源方案 GUIRepair/SVRepair 提升 10 例，且与闭源商业系统（Zencoder、Globant Code Fixer）相比提升 36–41 例。成本约为 0.47 美元/个问题，低于多数对比方法。Ablation 实验证实 ITTC 与 DTRF 两个模块相互补充，单独使用时也能分别提升约 15–20 例。

**⚠️ 局限性**

局限性包括：①实验主要基于 o3 模型，虽然在 GPT‑4o 等模型上也保持优势，但跨模型泛化仍需进一步验证；②仅在 JavaScript/TypeScript 前端库上评估，其他语言或后端系统的适用性未知；③对极长或高噪声截图、过多帧的 GIF 以及视觉信息极度密集的场景仍存在定位与补丁生成失败的情况。

---

## 461. An Epidemic Threshold Set for Networks

**arXiv ID:** 2607.14048 | [PDF](https://arxiv.org/pdf/2607.14048v1)

**作者:** Hoang Phi Dung `[一作]` (Posts and Telecommunications Institute of Technology), Nguyen Hong Phuc `[通讯]` (Posts and Telecommunications Institute of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了离散时间SIS模型在复杂网络中的流行病阈值，并提出了社区级阈值集合

**💡 创新点**

首次将阈值分解到社区层面，证明局部阈值不小于全局阈值，提供了社区级风险评估框架

**🔧 技术方法**

使用谱理论（矩阵特征值）推导阈值，结合非线性动力学模型与Rayleigh商

**📊 数据集**

采用合成网络Network1和真实接触网络Haslemere进行实验

**📈 对比分析**

通过计算每个社区的局部阈值与全局阈值的差异，以及SIS仿真验证，结果显示局部阈值均高于全局阈值，并能准确预测传播行为

**⚠️ 局限性**

仅考虑无向、同质参数网络；未考虑多层或异质传播率，对异构或有向网络的推广尚需进一步研究

---

## 462. PhysClaw-0: A Symbiotic Agentic System for Robot Autonomy via Language Corrections

**arXiv ID:** 2607.14047 | [PDF](https://arxiv.org/pdf/2607.14047v1)

**作者:** Boyuan Wang `[一作]` (GigaAI), Zheng Zhu `[通讯]` (GigaAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个人机协同的数据采集系统，在机器人自主收集、验证、重置的循环中引入自然语言干预，并将干预记忆化以供后续自动复用。

**💡 创新点**

创新点包括：① 通过显式的重试阈值将完全自主与人工干预分离，形成“可验证的交互式自主”循环；② 设计了一个“纠错记忆”机制，利用LLM将操作员的自然语言指令解析为结构化规则并持久存储；③ 在收集阶段引入VLM验证器与LLM解析器，使得系统能够在无需人工监控的情况下识别并纠正多种失败模式。

**🔧 技术方法**

使用技术包括：OpenClaw框架、Seed1.8 VLM验证器、DeepSeek‑V3.2 LLM解析器、VLA（Vision‑Language‑Action）策略网络、结构化Markdown纠错记忆、可视化多视角记录与自动标注。

**📊 数据集**

使用的主要数据集为真实机器人桌面清理（desktop‑clearing）实验集，收集了50条通过验证的演示轨迹；对比基线包括全手动遥控和脚本化自动收集。

**📈 对比分析**

比较方法：对三种收集模式（全遥控、脚本化、系统）在相同任务下统计成功率、人工工作时间、TpHM 等指标；对单次尝试成功率、VLM验证精度、策略下线成功率进行评估。结果显示系统在 50 条演示中 100% 成功率，人工工作时间仅为遥控的 16%，TpHM 约为 10 倍；单次成功率从 12.5% 提升到 47.5%；VLM 验证精度在四种情境下提升至 100%；策略下线成功率与全遥控训练的 80% 完全匹配。

**⚠️ 局限性**

局限性：目前仅适用于简单的桌面搬运任务，受限于相机精度和工具集；多轮数据循环（data flywheel）尚未完整验证；对含糊包含判定的误差仍存在；某些对象特定的执行失败在语言纠错后仍可能出现。

---

## 463. LLMs for Qualitative and Mixed-Methods Social Network Analysis

**arXiv ID:** 2607.14045 | [PDF](https://arxiv.org/pdf/2607.14045v1)

**作者:** Moses Boudourides `[一作]` `[通讯]` (Northwestern University), Moses Boudourides (Northwestern University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨将大型语言模型（LLM）应用于定性与混合方法社会网络分析（SNA），并提出一系列设计模式与实践指南，以扩展定性SNA的规模与深度。

**💡 创新点**

创新点在于：①把LLM视为协同者而非分析主体；②提出可操作的设计模式（顺序、并行）和交叉验证流程；③制定LLM特有的可追溯性与透明度标准（模型版本、prompt、超参数、原始输出归档）；④强调语义同质化风险与反思性方法。

**🔧 技术方法**

使用的技术主要是大型语言模型（如GPT‑4）进行实体与关系提取、编码、memo生成和异常检测，并与传统文本分析工具（CAQDAS、Gephi、Python网络库）结合。

**📊 数据集**

示例数据集为200条半结构化访谈摘录，描述受访者在组织中的专业关系。

**📈 对比分析**

方法比较：采用人机编码可靠性评估（如Cohen’s κ、Krippendorff’s α）来检验LLM输出；若一致率低于0.70则需重新设计prompt。论文未给定数值实验结果，但通过阈值与迭代校正展示了可实现的质量控制。

**⚠️ 局限性**

局限性包括：LLM的偏见与幻觉；语义同质化导致少数群体关系被归一化；模型不可解释性与可复制性挑战；对数据隐私与伦理的高度依赖；需要研究者持续的反思与人工校正。

---

## 464. Square-Root Law for Covert Communication with Warden-Favorable Side Information

**arXiv ID:** 2607.14013 | [PDF](https://arxiv.org/pdf/2607.14013v1)

**作者:** Hossein Ahmadi `[一作]` (Politecnico di Torino), Eduard A. Jorswieck `[通讯]` (Technische Universität Braunschweig)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在Warden（守卫）可获得全部侧信息并对公开信号做最佳消除后，利用相对熵约束在残差空间下实现隐蔽通信的平方根定律。

**💡 创新点**

创新点在于：① 将相对熵约束直接放在消除后的残差上；② 引入残差方差（包括不可消除的噪声残差）作为决定隐蔽容量常数的核心参数；③ 在非均匀残差方差下给出最优的σ⁴功率调度；④ 证明高斯信号在该约束下实现一阶最优。

**🔧 技术方法**

主要技术：信息论相对熵分析、Gauss极大熵原理、Cauchy–Schwarz不等式、二阶展开、功率分配优化；以及对残差统计假设的严格建模。

**📊 数据集**

无实测数据集；所有结果均为理论推导与数值模拟验证。

**📈 对比分析**

通过数值仿真验证等效熵均衡器与闭式近似的准确性，展示了残差底噪提升的平方根常数，以及在非均匀残差下σ⁴调度相对于均匀调度的性能提升；相对熵预残差约束下仅得到基准常数。

**⚠️ 局限性**

局限：假设残差为已知的零均值高斯；对残差方差的精确估计要求严格；若残差存在不可忽略的不确定性或非高斯性，理论不再适用，可能进入线性律或有效-保密域；适用于单向AWGN模型，扩展到多天线或时变信道需进一步研究。

---

## 465. Agent Skill Security: Threat Models, Attacks, Defenses, and Evaluation

**arXiv ID:** 2607.13987 | [PDF](https://arxiv.org/pdf/2607.13987v1)

**作者:** Sanket Badhe `[一作]`, Priyanka Tiwari `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了可重用技能在大型语言模型代理中的安全生命周期，并提出了SkillSec-Eval评估框架。

**💡 创新点**

创新点在于首次构建完整的生命周期威胁分类、端到端评估流程以及针对仓库、检索、计划、执行和演化四个阶段的多层防御策略。

**🔧 技术方法**

采用了结构化技能抽象、LLM驱动的语义验证、向量检索+过滤、动态污点跟踪与策略执行、以及持续更新的验证管线等技术。

**📊 数据集**

使用了包含327个真实世界技能的SkillMCP仓库和约440个覆盖15类能力的查询工作负载。

**📈 对比分析**

通过对比三种入库策略、检索过滤、计划元数据校验、运行时监控和持续验证的实验，混合防御能将入库攻击率从52.9%降至7.9%，检索层ASR降至<10%，计划层误选率从45.6%降至8.7%，运行时攻击成功率从100%降至23%，演化阶段恶意检测率达92.5%。

**⚠️ 局限性**

主要局限包括实验规模有限、仅评估可重用技能而不涉及LLM自身攻击、未覆盖长期依赖演化与信任衰减等复杂情境。

---

## 466. GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch

**arXiv ID:** 2607.13960 | [PDF](https://arxiv.org/pdf/2607.13960v1)

**作者:** GigaWorld Team `[一作]`, Zheng Zhu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种改进的动作中心化世界动作模型 GigaWorld-Policy-0.5，用于机器人控制，兼顾未来视觉动态的稠密监督与推理时仅解码动作的低延迟需求。

**💡 创新点**

创新点包括：①将动作条件世界建模 (AC-WM) 与传统 WAM 预训练混合，提高动作-视觉关系的可转移性；②采用 Mixture-of-Transformers 架构，将视觉动态建模与动作生成拆分为专家网络，推理时可跳过视觉计算；③引入 agent‑based AutoResearch 自动化超参数搜索，显著提升实验效率。

**🔧 技术方法**

使用的技术包括：动作中心化 WAM 框架、Mixture‑of‑Transformers、流匹配 (flow‑matching) 损失、可视化 VAE 编码、umT5 语言编码、KV 缓存、Torch 编译与 C 运行时加速、AutoResearch 超参搜索。

**📊 数据集**

训练数据：2K 小时公开机器人数据与内部真实机器人数据；预训练视觉专家采用 GigaWorld‑1（10k+ 小时视频预训练）。

**📈 对比分析**

与 VLM‑based π_0.5、WAM‑based Motus、FastWAM、GigaWorld‑Policy 等基线对比。结果显示：在文本跟随、物体摆放、长序任务等 3 类实验中，GigaWorld‑Policy‑0.5 的平均成功率分别提升 0.05–0.35，最高 0.89；推理延迟在 RTX‑4090 上可达 85 ms，比 π_0.5 低 23% ，比 FastWAM 快 53%。

**⚠️ 局限性**

局限性：仍依赖大规模预训练视觉模型与大量机器人演示数据；在极高帧率或复杂多模态任务时，混合 AC‑WM 与 WAM 预训练可能导致训练不稳定；Mixture‑of‑Transformers 虽提高推理效率，但总体参数量仍较大，需更多硬件资源。

---

## 467. MetaPerch: Learning from metadata for bioacoustics foundation models

**arXiv ID:** 2607.14072 | [PDF](https://arxiv.org/pdf/2607.14072v1)

**作者:** Mustafa Chasmai `[一作]` (University of Massachusetts Amherst), Jenny Hamer `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过将记录元数据（如位置、季节、背景物种等）作为辅助监督信号，构建了一种新的生物声学基础模型，并在多任务学习框架下训练，提升了物种识别的鲁棒性。

**💡 创新点**

创新点在于系统评估多种可用元数据在训练过程中的作用，并提出了处理缺失元数据、混合增强以及可选对抗训练的完整方法，首次展示元数据在跨域迁移中的显著收益。

**🔧 技术方法**

技术上基于Perch 2.0的EfficientNet‑B3特征提取网络，加入多任务损失、多头MLP预测元数据、mixup增强以及可选的梯度反转对抗训练，实现了元数据与物种标签的联合学习。

**📊 数据集**

使用的训练数据包括Xeno‑Canto、iNaturalist、Tierstimmenarchiv和FSD50K四大公开录音集，累计超过1.55 M条录音、18,000 小时；评估基准覆盖17个物种识别任务，包括BirdSet、BEANS、WABAD等。

**📈 对比分析**

与基线模型及Perch 2.0等现有方法比较，加入元数据后在BirdSet、BEANS、WABAD等数据集上平均提升0.015–0.02的ROC‑AUC或cmAP，尤其在低资源地区和声景迁移场景表现突出，整体性能接近或超过当前最优模型。

**⚠️ 局限性**

局限性包括元数据覆盖不均导致部分信息不足；对抗训练对某些元数据效果不佳；在同时存在声学与物种双重域移位时收益有限；且极低比例元数据可用时模型优势不明显。

---

## 468. Leveraging unlabelled data for generalizable neural population decoding

**arXiv ID:** 2607.14086 | [PDF](https://arxiv.org/pdf/2607.14086v1)

**作者:** Ximeng Mao `[一作]` (Mila Quebec AI Institute), Guillaume Lajoie `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种联合自监督学习与监督学习的神经解码框架（MOJO），通过在尖峰标记化模型中加入掩码自编码器实现对无标签神经数据的预训练。

**💡 创新点**

创新点在于：①将掩码自编码（masked autoencoding）与行为预测（SL）同时优化，实现无标签尖峰数据的有效利用；②在尖峰token化架构中采用局部跨注意力与共享背骨，保持参数效率；③展示跨物种、跨模态（包括ECoG）正向迁移的可行性。

**🔧 技术方法**

核心技术包括：尖峰token化（每个尖峰为单独token）、跨时间块的交叉注意力编码、基于掩码的自编码器、共享Transformer/SSM背骨、联合损失（L_MOJO=α_SSL·L_SSL+α_SL·L_SL）和微调策略（UI、FT）。

**📊 数据集**

使用了多种公开数据集：猴子运动臂运动任务（5个实验室的center-out、random-target、maze任务）；老鼠视觉与决策任务（Allen视觉编码、IBL多区域Neuropixels记录）；人类ECoG说话语音数据（4位受试者的语音节拍）。

**📈 对比分析**

与纯监督的尖峰token化模型（如NDT-2、NDT-3、NEDS）以及其他SSL模型对比，MOJO在所有任务中均取得更高的解码精度（例如猴子运动任务R^2提升至0.87以上，鼠类视觉任务准确率>99%，人类语音任务提升至约47%），在少标签/few-shot微调场景中优势更为显著。

**⚠️ 局限性**

局限性包括：①对大量数据依赖强，若无标签数据不足或SSL学习弱，反而会损害SL性能；②掩码方案仅做时间维度，缺乏空间/神经元层面的自监督目标；③微调时需要重新学习单元嵌入，导致资源浪费；④模型对不同模态仍需额外的线性层，限制了多模态联合预训练的便捷性。

---

