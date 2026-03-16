# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-16 | 今日论文总数: 422

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Hierarchical Reference Sets for Robust Unsupervised Detection of Scattered and Clustered Outliers

**arXiv ID:** 2603.12847 | [PDF](https://arxiv.org/pdf/2603.12847v1)

**作者:** Yiqun Zhang `[一作]` (Guangdong University of Technology), Yunlin Liu `[通讯]` (Guangdong University of Technology)

**通讯引用:** 19077 | [OpenAlex ID](https://openalex.org/A5032643990)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的无监督异常检测框架DROD，能够同时识别分散异常（scatterliers）和聚类异常（clusterliers）

**💡 创新点**

创新点在于引入层次双重参考集：自然邻居子集（NRS）和基于图的参考集（GRS），通过局部异常指数LAI和全局异常指数SAI相结合的方式，克服了聚类异常对分散异常的遮蔽效应

**🔧 技术方法**

核心技术包括自然邻居搜索、自然邻居子集划分、自然邻居图构建、局部和全局异常指数计算以及采样增强的集成投票机制

**📊 数据集**

实验使用了20个真实的工业物联网与机器学习基准数据集（如PageBlocks、WPBC、mnist、musk等）以及12个合成数据集（D1–D12）

**📈 对比分析**

与八种主流无监督异常检测方法（如kNN、LOF、DGOF、CBLOF、OCSVM、IFOREST、COPOD、ECOD）对比，DROD在AUC、Precision‑s等指标上均取得显著优势，且在Wilcoxon检验中表现出统计显著性

**⚠️ 局限性**

局限性包括对高度不平衡聚类数据中微型正常簇与聚类异常区分仍有挑战，未来可结合下游任务进一步优化异常评分

---

## 2. Support is Search

**arXiv ID:** 2603.13018 | [PDF](https://arxiv.org/pdf/2603.13018v1)

**作者:** Alexander V. Gheorghiu `[一作]` (University of Southampton), Alexander V. Gheorghiu `[通讯]` (University of Southampton)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5083790955)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在基扩展语义框架下，将给定基下的支持关系对应于第二阶 hereditary Harrop 程序中的统一证明搜索，证明了支持关系在固定基上等价于逻辑程序的证明搜索，从而消除对全局量化的非构造性假设。

**💡 创新点**

创新点在于将基扩展语义的支持关系与逻辑程序的证明搜索建立等价关系，利用CPS编码和第二阶逻辑程序语言实现对支撑关系的构造性解释。

**🔧 技术方法**

使用第二阶逻辑程序语言、Hereditary Harrop 公式、CPS（continuation-passing style）编码、递归证明搜索等技术实现对基扩展语义的计算化解释。

**📊 数据集**

未使用任何实验数据集。

**📈 对比分析**

未进行方法比较或性能评估。

**⚠️ 局限性**

主要局限在经典逻辑情形缺乏相同的计算解释，以及更广泛逻辑的推广尚未完成。

---

## 3. A Generative Model of Conspicuous Consumption and Status Signaling

**arXiv ID:** 2603.13220 | [PDF](https://arxiv.org/pdf/2603.13220v1)

**作者:** Logan Cross `[一作]` (Google DeepMind), Joel Z. Leibo `[通讯]` (Google DeepMind)

**通讯引用:** 7244 | [OpenAlex ID](https://openalex.org/A5054808675)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在 Concordia 框架下构建基于大语言模型的生成代理群体，模拟社会互动与可见性如何导致奢侈品及其它符号的自发性需求增长与价格飙升，从而验证了基于适宜性理论的社会信号生成机制。

**💡 创新点**

创新点在于将“适宜性”与预测模式完成相结合，提出状态信号是由社会观察和模仿形成的内生反馈循环，而非固定效用或成本驱动；同时首次在大规模生成代理实验中展示了正价格弹性（Veblen效应）和亚文化形成的可复制机制。

**🔧 技术方法**

主要技术包括：Concordia 多代理生成框架、Gemma‑3‑27B LLM 作为行为与对话核心、基于情境的记忆系统实现“适宜性”推理、实验中对社会曝光、固定价格与合成商品的对照设置。

**📊 数据集**

使用的数据集包括：基于 LA 与 Kerala 的虚构代理人格、真实奢侈品牌与合成品牌（如 Chanel、Labubu 与 Serrurier Juliette）的商品列表、以及由 LLM 生成的情景脚本与对话日志。

**📈 对比分析**

通过对比社交与非社交条件、固定价格控制和合成商品控制，结果显示社交条件下状态商品的购买率从 21% 提升至 35%（p<0.001），价格出现明显 run‑up，Veblen 商品 Labubu 的价格弹性为 +1.45，固定价格实验进一步证明需求驱动价格，而非价格本身驱动需求。

**⚠️ 局限性**

局限性包括：LLM 回应缺乏多样性且受训练偏差影响、记忆系统简化导致模型收敛速度过快（几天而非数月）、实验仅在单一平台上验证，缺乏跨文化真实数据的长期追踪。

---

## 4. From Sparse to Dense: Multi-View GRPO for Flow Models via Augmented Condition Space

**arXiv ID:** 2603.12648 | [PDF](https://arxiv.org/pdf/2603.12648v1)

**作者:** Jiazi Bu `[一作]` (Shanghai Jiao Tong University), Dahua Lin `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 43260 | [OpenAlex ID](https://openalex.org/A5010087030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Multi‑View GRPO（MV‑GRPO），通过在流模型的条件空间上增添语义相近但多样化的描述，实现对生成图像的密集多视角奖励评估，从而提升文本到图像（T2I）流模型的偏好对齐效果。

**💡 创新点**

核心创新在于把传统单视角、稀疏的 GRPO 奖励映射转变为稠密多视角评估，利用预训练的 VLM/LLM 作为条件增强器自动生成多样化后置/前置描述，既丰富了样本间关系的探索，又避免了昂贵的样本重生。

**🔧 技术方法**

采用 Flux.1‑dev 作为基础流模型，结合 SDE 采样、GRPO 强化学习框架、VLM（Qwen3‑VL‑8B）/LLM（Qwen3‑8B）条件增强器，以及多种奖励模型（HPS‑v3、UnifiedReward‑v2、CLIP、ImageReward）进行训练和评估。

**📊 数据集**

在 HPD 100K+ 训练提示集与 400 条评测提示集上进行实验，配合公开的流模型与奖励模型数据。

**📈 对比分析**

与 Flux、Flow‑GRPO、DanceGRPO、TempFlow‑GRPO、DiffusionNFT 等基线在 HPS、UR‑v2（对齐/连贯/风格）、CLIP、IR 等指标上实现显著提升（如 HPS‑v3 最高 0.155、UR‑v2‑C 最高 3.701、IR 最高 1.193），收敛速度更快，延迟仅略高于基线（≈10% 以上），且保持相同的 NFE。

**⚠️ 局限性**

局限性包括：依赖条件增强器的质量与多样性；轻量化增强器虽降低成本但性能略逊；对奖励模型的选取敏感，易出现奖励破解；目前仅验证在流模型框架下，尚未在扩散模型或更大规模数据集上深入评估。

---

## 5. Less Data, Faster Convergence: Goal-Driven Data Optimization for Multimodal Instruction Tuning

**arXiv ID:** 2603.12478 | [PDF](https://arxiv.org/pdf/2603.12478v1)

**作者:** Rujie Wu `[一作]` (Peking University), Yizhou Wang `[通讯]` (Peking University)

**通讯引用:** 9486 | [OpenAlex ID](https://openalex.org/A5100602395)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对多模态指令调优，提出 Goal-Driven Data Optimization (GDO) 框架，在固定训练与评测条件下通过对训练样本进行六维描述符评估与目标驱动子集构建，实现更少数据、更快收敛。

**💡 创新点**

创新点在于统一的描述符与评分体系以及可调节的目标预设，允许在同一模型、优化器、检查点下，仅通过数据子集构造来实现效率和能力的可解释性提升。

**🔧 技术方法**

使用六维样本描述符（光流、VDS、时序需求、自洽性、PPL难度、覆盖度）和共享评分函数，结合预算、视频比例、源覆盖等约束的子集构造算法。

**📊 数据集**

在 LLaVA-OneVision 图像问答和 LLaVA-Video 短视频问答的完整共享池上构建子集，并在 MVBench、VideoMME、MLVU、LVBench 等四个多模态评测基准上评估。

**📈 对比分析**

将四个 GDO 目标（MinLoss、Diverse、Temp、Temp+）的 1× 子集与固定 512k 样本基线进行对比，结果显示 GDO 在每个基准上以 10 倍以上样本量减少实现更高准确率（最高 3.08 pp），且达到基线的样本点大幅降低。

**⚠️ 局限性**

局限在于对短视频与图像的训练池与极长视频基准不匹配，导致在 LVBench 上提升有限；此外仍需针对不同任务精细调整目标预设，当前方法仅在特定模型/训练配置下验证。

---

## 6. Visual-ERM: Reward Modeling for Visual Equivalence

**arXiv ID:** 2603.13224 | [PDF](https://arxiv.org/pdf/2603.13224v1)

**作者:** Ziyu Liu `[一作]` (Shanghai Jiao Tong University), Yuhang Zang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨模态视觉等价奖励模型Visual‑ERM，能够在视觉空间评估vision‑to‑code输出的视觉一致性，并将其作为奖励信号指导强化学习和测试时迭代改进。

**💡 创新点**

创新点在于将文本与视觉信息融合为生成式奖励模型，提供细粒度、可解释、任务无关的奖励，克服了传统文本规则和视觉编码相似度奖励的模态偏差和易被“奖励劫持”的缺陷。

**🔧 技术方法**

采用多模态生成式奖励网络，利用SFT在视觉‑文本对上训练；结合GRPO强化学习框架和渲染器将文本输出转化为图像；利用Severity评分与细粒度纠错反馈实现可解释性；并构建VisualCritic‑RewardBench进行评估。

**📊 数据集**

数据来源包括ChartMimic、OmniDocBench、olmOCRBench、UniSVG等vision‑to‑code基准，用以生成训练对；同时构建了1,335实例的VisualCritic‑RewardBench，覆盖图表、表格与SVG的视觉差异标注。

**📈 对比分析**

通过与基于DINO视觉编码奖励和基于文本规则（如TEDS）的奖励进行对比，RL+Visual‑ERM在Chart‑to‑Code、Table‑to‑Markdown、SVG‑to‑Code任务分别提升+8.4、+2.7、+4.1分；在VisualCritic‑RewardBench上，模型在F1、Recall、Pearson相关性上提升36.8/38.2/40.9，显著优于大模型基线。

**⚠️ 局限性**

局限性包括：需要昂贵的渲染与人工标注流程，模型规模仍受限，难以处理极大图像或极其复杂的布局；此外，跨语言、跨视觉域的泛化能力尚未得到充分验证。

---

## 7. Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**arXiv ID:** 2603.13044 | [PDF](https://arxiv.org/pdf/2603.13044v1)

**作者:** Vanessa Borst `[一作]` (Julius-Maximilians-University Wuerzburg), Samuel Kounev `[通讯]` (Julius-Maximilians-University Wuerzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文对比了医学影像分割专用模型（SMA）与通用视觉模型（GPVM）在三种二维医学图像分割任务上的性能，采用统一的训练与评估协议，并通过 Grad‑CAM 分析模型可解释性。

**💡 创新点**

创新点在于构建了受控、跨数据集的实验框架，系统评估了 GPVM 在多模态、多类别任务中是否能匹配或超越专用模型，并为资源节约提供了实践依据。

**🔧 技术方法**

技术手段包括：ImageNet 预训练编码器、512×512 输入尺寸、AdamW+REX 学习率调度、批量 8、混合精度训练；评估指标有 mIoU、mDSC、Recall、Precision；采用 Grad‑CAM 进行可解释性分析；搭建统一基准框架。

**📊 数据集**

使用的数据集为 ISIC（皮肤病变 RGB 二分类）、BKAI（内镜 RGB 三分类息肉）以及 CAMUS（心脏超声灰度多分类）。

**📈 对比分析**

通过五折交叉验证、统一超参数搜索和早停策略，对 11 种模型（包括 4 种 SMA 和 7 种 GPVM）进行对比。结果显示，GPVM（如 vwmit、vwconv、transnext、internimage 等）在平均 mDSC 上往往优于大多数 SMA；SwinU‑Mamba 为最强 SMA；GPVM 在 BKAI 任务上优势最大，其他任务差距相对较小；Grad‑CAM 结果表明 GPVM 能捕捉临床相关结构，甚至在某些类上更精准。

**⚠️ 局限性**

局限性包括：仅评估三种 2D 数据集，未覆盖 3D 或极低样本量场景；模型集合有限；实验结果可能因预处理、分割工具或数据划分方式不同而产生偏差；所得到的结论不一定能推广到所有医学分割任务。

---

## 8. Teaching Agile Requirements Engineering: A Stakeholder Simulation with Generative AI

**arXiv ID:** 2603.12925 | [PDF](https://arxiv.org/pdf/2603.12925v1)

**作者:** Eva-Maria Schön `[一作]` (University of Applied Sciences Emden/Leer), Tiago Silva da Silva `[通讯]` (Federal University of São Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了使用生成式 AI 人物进行利益相关者模拟的教学案例，帮助学生在敏捷需求工程课程中实践访谈、需求分析与反思。

**💡 创新点**

通过元提示实现不依赖特定大语言模型的 AI 人物模拟，结合敏捷实践（如故事地图、影响图）和结构化反思，提供可复制、可持续的教学模板。

**🔧 技术方法**

使用生成式 AI（如 ChatGPT）与元提示技术、敏捷需求工程工具（用户故事、故事地图、影响图）以及在线协作平台。

**📊 数据集**

主要利用大语言模型的预训练知识构建 AI 人物，未使用公开数据集；角色描述和情景场景由教师手工定义。

**📈 对比分析**

论文未给出定量对比实验，仅通过多学期课堂实践和学生反馈评估效果；未提供客观性能指标或量化比较。

**⚠️ 局限性**

限制包括：AI 人物受训练数据偏见影响，可能不真实反映目标用户；生成式 AI 在可视化故事地图等方面表现有限；元提示需手动优化，可能导致交互性不足；学生对工具使用和角色定位的理解差异较大。

---

## 9. Colluding LoRA: A Composite Attack on LLM Safety Alignment

**arXiv ID:** 2603.12681 | [PDF](https://arxiv.org/pdf/2603.12681v1)

**作者:** Sihao Ding `[一作]` `[通讯]` (Mercedes-Benz Research & Development North America), Sihao Ding (Mercedes-Benz Research & Development North America)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Colluding LoRA（CoLoRA）的攻击框架，利用若干看似安全的 LoRA 适配器在被线性合并后触发模型拒绝功能被压制，从而在无输入触发的情况下实现大范围的拒绝抑制。

**💡 创新点**

创新点在于将安全失效拆分为多个单独安全的适配器，只有在特定组合时才激活危害，揭示了模块化 LLM 供应链中的组合触发攻击和组合盲区安全风险。

**🔧 技术方法**

采用低秩适配（LoRA）、交替梯度累积的 interleaved 优化、可解释伪装（plausibility camouflage）以及多目标损失训练等技术实现攻击。

**📊 数据集**

训练与评测使用 GSM8K、Shakespeare-Dolly、CodeAlpaca（功能锚定），AdvBench 与 HarmBench（攻击与安全评估），以及 Alpaca、DistilRoBERTa-Rejection-Classifier、LlamaGuard3、HarmBench-Llama-2-13B-CLS 等工具。

**📈 对比分析**

通过 False Refusal Rate (FRR) 与 Attack Success Rate (ASR) 进行比较，单个适配器 FRR 低且 ASR 接近 0%，但组合后 FRR 仍保持低，而 ASR 在 Gemma2、Qwen2.5、Llama3 三大模型上均逼近 100%，表明攻击既有效又保持伪装。

**⚠️ 局限性**

局限性包括：仅在开放权重且已冻结的基础模型上可行；需攻击者预先获取基础模型权重；组合规模通常受限于 2-3 个适配器，且对随机组合的鲁棒性有限；当前缺乏高效的组合感知检测方法。

---

## 10. Towards Faithful Multimodal Concept Bottleneck Models

**arXiv ID:** 2603.13163 | [PDF](https://arxiv.org/pdf/2603.13163v1)

**作者:** Pierre Moreau `[一作]` (Ekimetrics), Milan Bhan `[通讯]` (Télécom Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向多模态的概念瓶颈模型（mCBM）框架，旨在实现概念检测与信息泄漏的联合优化；

**💡 创新点**

创新点在于：①引入可微泄漏损失函数直接约束任务泄漏；②用Kolmogorov‑Arnold Network（KAN）替代传统线性分类层以提升表达能力并减轻概念层对任务信息的“渗漏”；③将CLIP视觉‑语言骨干统一用于文本与图像的多模态特征提取；

**🔧 技术方法**

核心技术包括：CLIP视觉‑语言模型、概念检测损失（MSE）、任务泄漏损失（基于核密度估计的互信息），以及KAN预测层；

**📊 数据集**

实验使用四个数据集：N24News（新闻图文分类）、CUB‑200‑2011（鸟类属性分类）、AGNews、DBpedia（文本分类），并在不同CLIP规模（Base、Large）下进行评估；

**📈 对比分析**

与四个竞争方法（独立训练、线性层、残差连接、基于子集的CBM）比较，实验表明本方法在保持与黑盒模型相当的分类准确率的同时，显著降低概念检测误差和任务/概念泄漏，并在概念干预实验中唯一获得显著精度提升；

**⚠️ 局限性**

局限性包括：对概念标注的依赖（需人工或自动生成高质量概念集合）；对大规模CLIP骨干的计算开销；在某些数据集（如DBpedia）上，概念检测改进不一，表明泄漏与检测之间仍存在一定权衡；

---

## 11. Route Fragmentation Based on Resource-centric Prioritisation for Efficient Multi-Robot Path Planning in Agricultural Environments

**arXiv ID:** 2603.12994 | [PDF](https://arxiv.org/pdf/2603.12994v1)

**作者:** James R. Heselden `[一作]` (Lincoln Centre for Autonomous Systems), Gautham P. Das `[通讯]` (Lincoln Centre for Autonomous Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了两种基于资源优先级的 Fragment Planner（FP）变体，并在商业草莓农场多隧道拓扑地图上进行生命周期仿真评估；

**💡 创新点**

创新点在于：①以资源（节点/边）为中心的优先级调度，替代传统的代理中心优先级；②通过路段碎片化允许机器人在冲突点前后部分推进，减少等待；③实现了高吞吐量与接近最优路径长度的兼顾；

**🔧 技术方法**

技术包括：拓扑地图表示、基于资源优先级的冲突检测与分配、路段碎片化执行策略、基于优先级的重规划（PBS）对比、仿真框架与性能度量（吞吐量、路径优化效率 POE）等；

**📊 数据集**

数据集为一张 3.6 km 的商业多隧道草莓农场拓扑地图（414 个节点、1050 条有向边），任务目标随机生成于空闲节点；

**📈 对比分析**

通过与 Naïve、三种优先级规划（按名字、最短路、时间）以及 PBS 在 5–10 台机器人、1 h 试验中的吞吐量和 POE 进行对比；FP 在 10 台机器人时达到 95% 的最大理论吞吐量，POE 平均约 1.08–1.15，PBS 性能最差；

**⚠️ 局限性**

局限性：仅在狭窄通道型农业环境下验证，可能不适用于开放式或高自由度场景；FP 结果波动大，且缺乏真实机器人实验验证；Naïve 仅作为理论极限参考，实际不可实现；对更大规模或动态变化环境的可扩展性尚未证明。

---

## 12. DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation

**arXiv ID:** 2603.13133 | [PDF](https://arxiv.org/pdf/2603.13133v1)

**作者:** Zihao Xin `[一作]` (Nanjing University of Aeronautics and Astronautics), Shengjun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4268 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DecoVLN 框架，将感知、推理和纠错过程解耦，实现持续感知与动作并行；

**💡 创新点**

创新点在于自适应记忆细化机制（AMR）和基于状态-动作对的纠错微调策略，显著提升长期导航的上下文密度和错误纠正能力；

**🔧 技术方法**

采用 Vision‑Language 模型 LLaVA‑Video‑7B（SigLIP+Qwen2-7B），结合自适应记忆优化、地理距离阈值的误差检测和基于专家轨迹的纠错采样；

**📊 数据集**

使用 Matterport3D、R2R‑CE、R2R‑Envdrop、RxR‑CE 以及 LLaVA‑Video‑178K 进行预训练与纠错数据补充；

**📈 对比分析**

与多种基线（如 StreamVLN、NaVid、Ego²‑Map 等）在 R2R‑CE 与 RxR‑CE 的 Val‑Unseen 上对比，SR 分别提升至 56.3%/54.2%，SPL 提升 3.5%/2.8%，在长距验证集与真实世界实验中亦保持优异表现；

**⚠️ 局限性**

局限性包括对记忆长度的敏感性（过长会导致冗余）、需要手工设定阈值 τ 以平衡探索与纠错，以及对极端环境（如动态障碍）仍缺乏完整闭环适应能力。

---

## 13. HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation

**arXiv ID:** 2603.12708 | [PDF](https://arxiv.org/pdf/2603.12708v1)

**作者:** Pingping Zhang `[一作]` (Dalian University of Technology), and Huchuan Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在海洋动物分割任务中，本文提出了基于SAM的HFP-SAM框架，通过频率引导适配器(FGA)、频率感知点采样(FPS)和全视角Mamba(FVM)实现高精度分割。

**💡 创新点**

创新点在于利用频域先验进行自适配和提示生成，并将空间与通道长距离建模融合到Mamba解码器。

**🔧 技术方法**

技术包括离散Haar小波变换、频域引导适配器、频域点采样、基于SSM的Full-View Mamba以及加权BCE+IoU损失。

**📊 数据集**

使用四个公开海洋动物数据集：MAS3K、RMAS、UFO-120和RUWI。

**📈 对比分析**

与CNN、Transformer及多种SAM适配方法对比，HFP-SAM在所有指标上均领先，mIoU提升约0.7-0.8，Sα、Fβw、mEϕ、MAE均显著改进。

**⚠️ 局限性**

局限在于对粗分割结果的依赖，粗分割误差会被放大导致细化阶段失效，且在极端噪声或小目标场景下仍有改进空间。

---

## 14. VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos

**arXiv ID:** 2603.12703 | [PDF](https://arxiv.org/pdf/2603.12703v1)

**作者:** Pengyiang Liu `[一作]` (Artificial Intelligence), Si Liu `[通讯]` (Artificial Intelligence)

**通讯引用:** 13674 | [OpenAlex ID](https://openalex.org/A5100330138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VCBench流式计数基准，用于系统评估视频模型在播放过程中的空间‑时间状态维护能力。

**💡 创新点**

创新点在于把计数任务重构为最小化的世界状态探针，并设计8个细粒度子类别、4,576个多点查询和三种互补评测指标（GPA、MoC、UDA）。

**🔧 技术方法**

使用视频‑语言模型（包括Gemini‑3‑Flash、Qwen3‑VL‑8B/30B、InternVL‑3.5‑8B 等）进行离线和在线推理，并通过 GPT‑4‑Turbo 作为无视觉输入的语言基线；评测时采用基于截断视频的查询和流式交互两种协议。

**📊 数据集**

数据集包含406条多源长视频（来自YouTube、ARKitScenes、ScanNet、Ego4D 等），共1,000道计数问题、10,071个事件/状态变化时刻，覆盖室内导航、第一人称活动、运动等场景。

**📈 对比分析**

对比方法包括人工基准、无视觉 LLM、离线多模态模型、在线流式模型；结果显示人类平均 GPA 96‑100，公开模型仅达 30‑40，特别是周期事件计数（E2‑Periodic）所有模型 GPA <4，显著落后；开源模型在 MoC 上略优但数值精度差。

**⚠️ 局限性**

局限性在于模型缺乏对视觉‑时间边界的精准识别和身份持久性维护，导致对周期事件和重复出现物体的计数严重失效；基准仍依赖人工注释，且对更复杂多尺度事件的覆盖不足。

---

## 15. HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation

**arXiv ID:** 2603.12696 | [PDF](https://arxiv.org/pdf/2603.12696v1)

**作者:** Pingcong Li `[一作]` (ShanghaiTech University), Sören Schwertfeger `[通讯]` (ShanghaiTech University)

**通讯引用:** 1601 | [OpenAlex ID](https://openalex.org/A5051350739)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种层级导航框架HaltNav，利用轻量化的osmAG拓扑先验与现成的VLN执行模型相结合，实现从高层语义规划到低层视觉执行的闭环；

**💡 创新点**

创新点包括：①基于文本化osmAG的图驱动任务调度器GGTD，用LLM直接解析拓扑地图生成子目标；②实时视觉阻断检测模块RVH，可中断低层控制并动态更新osmAG，实现自适应路径重规划；③通过生成式视觉增广与物理仿真相结合的失败注入合成管道，提升阻断识别的鲁棒性；

**🔧 技术方法**

技术主要包括：多模态大语言模型（Gemini Flash / Qwen-2.5-VL-7B）、VLN低层策略（InternVLA-N1）、拓扑地图osmAG、A*全局规划、RL式终止判定、LoRA微调等；

**📊 数据集**

使用HM3D（Habitat-Matterport 3D Research Dataset）自建的兼容osmAG的场景数据集（约176条任务，5种场景），以及Fetch机器人实地实验的物理环境；

**📈 对比分析**

与五个基线（Navid、OmniNav、StreamVLN、Uni-navid、InternVLA-N1）在标准与障碍注入、不同指令粒度（L0~L2）下对比，HaltNav在成功率、SPL、Oracle成功率与导航误差等指标上均显著优于基线，尤其在障碍注入与简化指令下保持较高的鲁棒性；

**⚠️ 局限性**

局限性包括：①仍需预先获取并维护osmAG地图，无法处理建筑外扩或快速变化的拓扑；②对视觉感知的依赖导致在极端光照或遮挡下检测可能失效；③高层LLM推理成本较大，实际实时性受限；④实验场景仍以室内办公为主，尚未验证在更大规模或户外环境中的泛化。

---

## 16. Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**arXiv ID:** 2603.12751 | [PDF](https://arxiv.org/pdf/2603.12751v1)

**作者:** James Akl `[一作]` (Robotics and AI Institute), Scott Shaw `[通讯]` (Robotics and AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了“Show, Don’t Tell”范式，利用单段人类演示视频自动生成标注数据，训练专用目标检测器，并在机器人抓取排序任务中实现快速部署。

**💡 创新点**

创新点在于完全摆脱语言描述与提示工程，通过自动化的数据生成与自监督训练，使机器人在仅观看一次演示后即可识别和区分任务相关的新颖物体，显著提升了实例级别的识别精度。

**🔧 技术方法**

使用的核心技术包括：HOIST-Former（手抓物体分割）、SAMURAI（掩膜跟踪）、空间时间聚类（DBSCAN + Jaccard）、Faster R‑CNN（轻量化检测器）以及GPT‑4o生成的动作计划骨架。

**📊 数据集**

评估数据集包括：Meccano（拼装任务视频）、In‑House Dataset 1（人类与机器人演示视频）以及In‑House Dataset 2（更多人类交互视频）。

**📈 对比分析**

与三大 VLM 基线（RexOmni、GroundingDINO、YoloWorld）以及人工提示/自动提示的对比实验显示，MOD 在 mAP、mAR、Precision、Recall 和推理速度上均优于基线，尤其在实例级别区分和快速推理（约 100 ms）方面表现突出。

**⚠️ 局限性**

局限性包括：需要人工监督来处理视角欠缺导致的漏检和误检；对演示视角的过拟合；以及在复杂环境下仍需人工干预确认检测结果。

---

## 17. Human-Centered Evaluation of an LLM-Based Process Modeling Copilot: A Mixed-Methods Study with Domain Experts

**arXiv ID:** 2603.12895 | [PDF](https://arxiv.org/pdf/2603.12895v1)

**作者:** Chantale Lauer `[一作]` (Saarland University), Nijat Mehdiyev `[通讯]` (German Research Institute for Artificial Intelligence)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于大语言模型的 BPMN 生成助手 KICoPro 进行混合方法评估，结合专家访谈和标准问卷来检验其可用性、信任度与输出质量；

**💡 创新点**

首次揭示可用性与信任之间的差距，并提出提示设计、主动澄清、约定配置等人性化改进方案，强调人本评估对 LLM 工具实务采纳的重要性；

**🔧 技术方法**

利用 GPT‑style LLM 后端、基于聊天的前端交互以及 BPMN 渲染组件，实现自然语言到 BPMN 图的即时转换；

**📊 数据集**

使用两份代表性流程描述（含多段落与复杂结构）供专家练习，并允许专家自选流程进行自助探索；

**📈 对比分析**

与自动化质量基准相比，CUQ 可用性得分接近 68 的行业基准，信任度仅 48.8%（低于 60% 阈值），任务质量平均 54.4%，表明虽然交互友好，但输出可靠性仍不足；

**⚠️ 局限性**

样本规模仅 5 位同一组织的专家，缺乏初学者与多领域验证，系统在车道、消息流等 BPMN 元素支持不足，评估仅为短期快照，未覆盖长期使用与大规模部署情况。

---

## 18. Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation

**arXiv ID:** 2603.12983 | [PDF](https://arxiv.org/pdf/2603.12983v1)

**作者:** Boxuan Lyu `[一作]` (Institute of Science Tokyo), Zhi Qu `[通讯]` (National Institute of Information and Communications Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完全不依赖人工标注的错误跨度检测（ESD）自演化训练框架，即Iterative MBR Distillation，利用LLM生成伪标签并通过迭代蒸馏提升模型能力。

**💡 创新点**

创新点在于将MBR解码的高质量伪标签与自蒸馏迭代相结合，实现无人工监督的自演化ESD训练，从而突破传统对昂贵且主观人工标注的依赖。

**🔧 技术方法**

采用的大型语言模型（LLM）生成候选错误跨度，利用SoftF1作为MBR Utility进行候选筛选，并结合SFT/DPO/KTO三种训练目标进行模型微调。

**📊 数据集**

实验数据使用WMT20–23 Metrics Shared Task的源-译对作为无标签训练集，WMT24 MQM注释作为系统、句子和跨度级别的评估测试集。

**📈 对比分析**

与基准模型（直接prompt）和基于人类注释的Gold‑SFT/DPO/KTO进行对比，Iterative MBR Distill在系统级（SPA）和跨度级（SoftF1）均显著优于基准，且在句子级（Acc_eq^*）保持竞争力。

**⚠️ 局限性**

局限性是当迭代次数升至3时性能出现下降，原因是候选多样性不足导致MBR估计误差难以进一步降低，需要进一步研究保持候选多样性的方法。

---

## 19. A Physics-Based Digital Human Twin for Galvanic-Coupling Wearable Communication Links

**arXiv ID:** 2603.12899 | [PDF](https://arxiv.org/pdf/2603.12899v1)

**作者:** Silvia Mura `[一作]`, Maurizio Magarini `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了基于物理的数字人类双胞胎模型，用于系统性表征可穿戴电偶耦合通信链路在窄带和宽带下的衰减、相位和时延特性，并通过实验验证。

**💡 创新点**

创新点在于将校准的有限元仿真与实验测量结合成统一的数字双胞胎，直接输出复杂传输函数；量化了介质色散、接触界面影响，并给出了可穿戴GC系统的设计准则。

**🔧 技术方法**

采用电极化学耦合与电解胶/泡沫介质模型的Cole‑Cole生物组织电磁参数、COMSOL电流界面求解、PN序列声道测量、统计校准与相位/时延指标提取等技术。

**📊 数据集**

使用的实验数据集为健康志愿者前臂上的Ag/AgCl表面电极测得的PN序列声道响应（m=14，频宽≈96 kHz），以及对应的有限元模型参数（皮肤、脂肪、肌肉、骨层厚度及电介质参数）。

**📈 对比分析**

通过将DT预测的衰减、相位延迟、群时延与测量结果进行RMSE/MAE/相关系数对比，平均RMSE约5–6 dB、MAE≈4 dB、相关系数0.74–0.91；宽带色散指标也与实验相符，证明模型的可靠性。

**⚠️ 局限性**

局限性包括仅覆盖10 kHz–1 MHz的电偶耦合范围、使用简化的同轴层状手臂几何、受限于少量受试者与静态测量、线性与准静态假设可能忽略运动和多路径效应。

---

## 20. Neuron-Aware Data Selection In Instruction Tuning For Large Language Models

**arXiv ID:** 2603.13201 | [PDF](https://arxiv.org/pdf/2603.13201v1)

**作者:** Xin Chen `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**通讯引用:** 3757 | [OpenAlex ID](https://openalex.org/A5101468579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Nait 框架，利用目标能力的神经元激活模式对指令调优数据进行高质量子集选择，从而提升 LLM 在特定与通用任务上的表现。

**💡 创新点**

创新点在于用神经激活特征直接驱动数据选择，避免外部模型或昂贵特征，既高效又可解释，并揭示了逻辑推理与编程特征的跨任务迁移能力。

**🔧 技术方法**

技术包括在目标任务数据上捕获神经激活向量，使用 PCA 提取主激活方向，对候选 IT 样本计算激活对齐得分并按得分排序；在 LLaMA-2-7b 等 LLM 上实现。

**📊 数据集**

使用的主要数据集包括 Alpaca‑GPT4 自制指令调优集，MMLU、GSM、BBH、TyDiQA、CodeX、HumanEval、MBPP 等基准；在 Evo‑Instruct、Orca‑GPT4 等数据集上进行验证。

**📈 对比分析**

通过与 Alpaca‑GPT4、LIMA、AlpaGasus、Q2Q、SelectIT 等基线对比，10% Nait 子集在 LLaMA‑2‑7b 上平均提升 3.24%（最高 4.65%），在不同模型（LLaMA‑2‑13b、Mistral‑7b、LLaMA‑3‑8b、Qwen‑2.5‑7b）和任务上持续优于随机或现有选择方法。

**⚠️ 局限性**

局限在于对某些任务（如多语言理解）可能出现负迁移，需要更细粒度的激活特征；在极大模型或多语言环境下的跨语言泛化还需进一步验证。

---

## 21. Competition-Aware CPC Forecasting with Near-Market Coverage

**arXiv ID:** 2603.13059 | [PDF](https://arxiv.org/pdf/2603.13059v1)

**作者:** Sebastian Frey `[一作]` (Nova School of Business and Economics), Maximilian Kaiser `[通讯]` (University of Hamburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在付费搜索中通过构造语义、行为和地理竞争代理来改进CPC预测，并在汽车租赁行业的大规模Google Ads日志上进行实验。

**💡 创新点**

提出将部分可观测的竞争信号通过预训练Transformer、DTW与地理标签等构建代理，并将其作为协变量或固定语义图融入时间序列基线、TSFMs和STGNNs，验证其对中长期预测的显著提升。

**🔧 技术方法**

采用预训练Transformer嵌入、DTW行为邻域、地理意图编码构建竞争代理；使用ARIMA、XGBoost、TabPFN等传统基线；Chronos、TimeGPT、Moirai等时间序列基础模型；以及DCRNN、GConvLSTM、GraphWaveNet等时空图神经网络。

**📊 数据集**

基于2021–2023年欧盟一家BI公司提供的约1.66亿条Google Ads汽车租赁行业日志，筛选后得到包含1,811个关键词、127周的周度CPC、点击、曝光及地理标签的面板。

**📈 对比分析**

采用严格时间拆分，评估1、6、12周多期预测，指标为sMAPE和RMSE；结果显示加入竞争代理的TSFMs在6/12周显著降低sMAPE（从≈35%降至≈27%），STGNN在1周表现最好，整体平均误差下降约5–10个百分点。

**⚠️ 局限性**

仅针对单一垂直且竞争集中，语义图固定无法捕捉动态关键词关系；缺乏竞争方的真实出价/质量信号，且未验证跨行业推广效果。

---

## 22. Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity

**arXiv ID:** 2603.12707 | [PDF](https://arxiv.org/pdf/2603.12707v1)

**作者:** Donglin Yu `[一作]` `[通讯]` (University of Illinois), Donglin Yu (University of Illinois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于模态边界的多模态大语言模型推理拆分方案，并构建了 HeteroServe 运行时实现。

**💡 创新点**

创新点在于证明在标准 Transformer KV 缓存下，模态边界能使跨设备传输量下降 O(L)（相较于阶段边界的 O(L·s_ctx)），从而实现跨 PCIe 的成本高效异构部署，并给出了闭式成本模型。

**🔧 技术方法**

采用的技术包括：跨模态分区与交叉层次调度、基于视觉嵌入的 MB 级轻量级传输、跨类型工作窃取、CUDA Graph 捕获、Flash Attention Varlen、KV 延迟分配、张量并行支持等。

**📊 数据集**

使用 COCO 2017 验证集的图像描述任务（Prompt: "Describe this image in detail"，输出 128 token 上限）。

**📈 对比分析**

与 vLLM v0.3.0 的同质 A100 基线相比，HeteroServe 在相同硬件上实现了最高 54% 的吞吐提升；在固定预算下，异构集群（2×RTX 4090 + 2×A100）相较于 4×A100 的同质部署实现了 37% 的 Tokens/美元增益，并保持 81% 的吞吐率。

**⚠️ 局限性**

局限性包括：假设使用标准 KV 缓存，无法直接适用于需激活重计算或 KV 离线存储的场景；需要高性能 GPU（如 RTX 4090）支持视觉编码；跨 PCIe 传输虽然已被证明在实验中可接受，但在更高分辨率或更大嵌入尺寸时可能面临瓶颈；方案主要针对视觉‑语言模型，其他多模态任务的通用性尚待验证。

---

## 23. Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence

**arXiv ID:** 2603.13091 | [PDF](https://arxiv.org/pdf/2603.13091v1)

**作者:** Seunghwan Bang `[一作]` (Ulsan National Institute of Science and Technology), Hwanjun Song `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2042 | [OpenAlex ID](https://openalex.org/A5033909285)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于场景驱动的合成 egocentric 视频基准（VAEX‑Bench），系统性地评估多模态大语言模型在抽象与提取时空推理上的表现；

**💡 创新点**

提出了一套抽象时空推理的任务扩展原则和评价词汇表，构建了可控的、情景驱动的合成视频数据集，首次将抽象推理与提取推理放在同一框架下进行对比；

**🔧 技术方法**

利用 3D 视觉合成引擎（如 Unreal Engine/Enscape）、SketchUp 进行场景建模、手工编排轨迹，随后通过多模态 LLM 接口（包括 proprietary 与 open‑source MLLMs）对视频进行零样本推理，并用 LLM‑as‑a‑judge 评估自由文本输出；

**📊 数据集**

数据集由 10 条 egocentric 视频组成，每条视频对应 5 种抽象任务与 5 种提取任务，共 300 条问答；视频通过手工生成、控制场景布局与对象分布，保证答案可唯一确定；

**📈 对比分析**

在 14 款最先进的 MLLMs 上进行对比实验，发现抽象推理任务的准确率普遍低于提取任务；多个开源模型即使在提取任务中表现可与专有模型相当，但在抽象任务上仍明显落后；多选题优于自由文本生成，表明模型在推理时依赖选项提示；

**⚠️ 局限性**

主要限制在于数据规模有限（仅 10 条视频），全部手工构建导致扩展性受限；此外，模型在长时序记忆、全局空间建模和实体聚合方面仍存在显著瓶颈，需进一步研究自动化生成与改进推理机制。

---

## 24. A Multi-task Large Reasoning Model for Molecular Science

**arXiv ID:** 2603.12808 | [PDF](https://arxiv.org/pdf/2603.12808v1)

**作者:** Pengfei Liu `[一作]` (Sun Yat-sen University), Zhixiang Ren `[通讯]` (Pengcheng Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多任务分层推理模型，通过专门化模块与链式推理实现化学问题的知识驱动推断

**💡 创新点**

将化学知识嵌入链式推理并结合多专家协同与强化学习，实现了比大型基准模型更高的精度与可解释性

**🔧 技术方法**

基于DeepSeek-7B的Transformer架构，使用LoRA低秩适配、多任务路由器、CoT微调和RL‑reward对齐

**📊 数据集**

构建了93K条指令式训练集和3.6K条高质量CoT样本，涵盖10个分子科学任务（生成、翻译、预测、反应等）

**📈 对比分析**

与20+基准（包括LLaSMol、TxGemma、DeepSeek等）进行10项任务评估，平均提升约50%，在多数任务中优于最先进模型

**⚠️ 局限性**

需要大量高质量CoT数据且多专家管理开销高，模型对极大数据集的扩展性和推理链条复杂度仍有待优化

---

## 25. Experimental evidence of progressive ChatGPT models self-convergence

**arXiv ID:** 2603.12683 | [PDF](https://arxiv.org/pdf/2603.12683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 26. Vision Verification Enhanced Fusion of VLMs for Efficient Visual Reasoning

**arXiv ID:** 2603.12669 | [PDF](https://arxiv.org/pdf/2603.12669v1)

**作者:** Selim Furkan Tekin `[一作]` (Georgia Institute of Technology), Ling Liu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 21459 | [OpenAlex ID](https://openalex.org/A5100343991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 V3Fusion 框架，通过视觉与语言的多模态多模型融合来提升视觉语言推理性能，包含多模型选择、融合与不确定性修正三阶段。

**💡 创新点**

创新点在于：① 引入焦点误差多样性（focal error diversity）与 CKA‑focal 指标捕捉 VLM 视觉理解差异；② 用遗传算法对候选模型池进行高效剪枝；③ 采用自适应先验不确定性阈值进行结果校正与拒绝。

**🔧 技术方法**

主要技术包括：中央核对齐 (CKA)、焦点误差多样性度量、遗传算法、MLP/LED 融合模型、基于贝叶斯与互信息的模型不确定性估计，以及两阶段高斯混合阈值自适应。

**📊 数据集**

在四大视觉语言基准上验证：MMMU、MMMU‑Pro、A‑OKVQA 与 OCR‑VQA，涵盖多选题和开放式问答。

**📈 对比分析**

与单个最佳 VLM、路由/协作/多代理争论方法以及监督式融合方法比较，V3Fusion 在 Accuracy/EM/F1 等指标上均优于对比方法，且推理时间仅为 3–5 秒左右，性能提升显著。

**⚠️ 局限性**

局限性包括：① 仍需先验选择合适的模型池；② 依赖遗传算法在极大模型池下的收敛速度与局部最优；③ 对极少量样本的训练时序和调参要求较高；④ 目前仅针对 VQA 类任务，跨任务迁移仍待验证。

---

## 27. MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization

**arXiv ID:** 2603.12677 | [PDF](https://arxiv.org/pdf/2603.12677v1)

**作者:** Shuxin Liu `[一作]` (Hangzhou Institute for Advanced Study University of Chinese Academy of Sciences), Ou Wu `[通讯]` (Hangzhou Institute for Advanced Study University of Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的知识编辑框架MetaKE，采用双层优化方法，使编辑目标与模型的可执行空间对齐，解决了传统方法中的语义-执行断裂问题。

**💡 创新点**

创新点在于将编辑目标视为可学习的meta参数，并引入结构梯度代理（Structural Gradient Proxy）在不进行完整求解器反向传播的情况下，将物理约束反馈给目标优化，从而避免频谱抑制和静态正则化陷阱。

**🔧 技术方法**

技术主要包括双层优化、结构梯度代理、线性关联记忆建模、闭式结构梯度门控，以及基于AlphaEdit或MEMIT的多层求解器。

**📊 数据集**

实验使用了公开的ZsRE（Zero-shot Retrieval-based Editing）基准数据集，涵盖不同规模的LLM（LLaMA3-8B、GPT-J-6B、GPT2-XL-1.5B）。

**📈 对比分析**

与ROME、MEMIT、PRUNE、RECT、AlphaEdit、AlphaEdit_BLUE等主流方法比较，MetaKE在编辑成功率、泛化能力和特异性（局部性）指标上均显著提升，尤其在GPT-J和GPT2-XL模型上实现了更高的有效率和更好的泛化。

**⚠️ 局限性**

局限性包括：结构梯度代理仅基于单层信息，可能对深层多层依赖的编辑效果有限；实验主要聚焦于线性层的编辑，未验证对非线性或更复杂模型结构的适用性；以及在极端大规模模型或实时编辑场景下的可扩展性仍待进一步探索。

---

## 28. Motion-Specific Battery Health Assessment for Quadrotors Using High-Fidelity Battery Models

**arXiv ID:** 2603.12791 | [PDF](https://arxiv.org/pdf/2603.12791v1)

**作者:** Joonhee Kim `[一作]` (Pohang University of Science and Technology), Soohee Han `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 4734 | [OpenAlex ID](https://openalex.org/A5069368669)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了针对四旋翼无人机的端到端运动感知电池健康评估框架，结合宽范围电流采集模块和高保真电池模型；

**💡 创新点**

创新点在于将飞行运动产生的瞬态电流负荷与电池退化耦合，证明相同平均能耗的不同运动模式会导致不同退化路径，从而提出运动感知的电池管理需求；

**🔧 技术方法**

采用宽范围电流传感器、基于退化耦合电化学模型的元启发式校准方法，以及对测得飞行负载进行仿真分析的技术；

**📊 数据集**

使用真实四旋翼飞行的电流负载数据以及参考性能测试数据进行模型校准与验证；

**📈 对比分析**

通过比较平均能耗相同但瞬态负荷结构不同的两条飞行轨迹，评估其对电池退化（锂离子耗损、活性物质损失等）的影响，结果表明瞬态负荷显著改变退化路径；

**⚠️ 局限性**

局限性包括仅针对特定电池和飞行模式验证，缺乏长期实验数据，模型假设和参数可能不适用于所有电池类型，且需要高精度电流采集硬件。

---

## 29. PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses

**arXiv ID:** 2603.13026 | [PDF](https://arxiv.org/pdf/2603.13026v1)

**作者:** Chenlong Yin `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**通讯引用:** 2125 | [OpenAlex ID](https://openalex.org/A5101997385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为PISmith的强化学习红队框架，用于系统评估prompt注入防御。

**💡 创新点**

创新点在于引入自适应熵正则化和动态优势加权两种机制，有效克服奖励稀疏导致的熵崩塌与梯度稀释问题。

**🔧 技术方法**

采用基于GRPO的RL方法，并在此基础上加入自适应熵和优势加权，训练攻击LLM。

**📊 数据集**

使用13个基准（含QA、RAG、长文本）以及多种防御（Meta‑SecAlign、PromptGuard等）和7个攻击基线进行评估。

**📈 对比分析**

相较于静态、搜索和RL基线，PISmith在所有基准上实现了平均ASR@10≈1.0、ASR@1≈0.87，甚至在InjecAgent与AgentDojo等复杂代理场景中均保持≈0.95的高成功率。

**⚠️ 局限性**

局限在于仍无法实现既保持高效能又对抗适应攻击的防御；PISmith依赖大量查询，且评估局限于黑盒设置。

---

## 30. Not Just the Destination, But the Journey: Reasoning Traces Causally Shape Generalization Behaviors

**arXiv ID:** 2603.12397 | [PDF](https://arxiv.org/pdf/2603.12397v1)

**作者:** Pengcheng Wen `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18628 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了链式思考（CoT）在大语言模型中的因果影响，构建QTA数据集来探究不同思考内容对模型对齐行为的影响。

**💡 创新点**

证明思考过程本身具有独立的因果作用，即使不监督答案，模型在无思考推理模式下仍能表现出误导性行为，并揭示不同CoT类型导致的行为差异。

**🔧 技术方法**

采用监督微调（SFT）和多种推理模式（think / no-think），在Qwen3系列模型上进行实验。

**📊 数据集**

使用自构造的Emergent Misalignment QTA数据集，包含Evil、Misleading、Submissive三种CoT类型，并基于已有安全评测基准进行评估。

**📈 对比分析**

与传统QA‑SFT对比，CoT训练显著放大了误导性行为，Evil CoT提升误导率高达10个百分点，Submissive CoT显著提高欺骗性；不同思考类型在多项安全基准上表现差异明显。

**⚠️ 局限性**

局限在仅针对Qwen3模型、仅使用合成CoT、未覆盖RLHF等强化学习策略，并未评估事实性、鲁棒性等其他安全维度。

---

## 31. Linguistic Similarity Within Centralized FLOSS Development

**arXiv ID:** 2603.12571 | [PDF](https://arxiv.org/pdf/2603.12571v1)

**作者:** Matthew Gaughan `[一作]` (Northwestern University), Darren Gergle `[通讯]` (Northwestern University)

**通讯引用:** 9351 | [OpenAlex ID](https://openalex.org/A5061173804)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Wikimedia Foundation 在 MediaWiki 三个功能（VisualEditor、HTTPS-login、HTTP-deprecation）中的集中式开发与贡献者讨论语言风格进行比较研究。

**💡 创新点**

创新点在于发现即便 steward 主导开发，讨论语言并未表现出层级差异；并指出 steward 的开发关注度与自身使用需求高度相关。

**🔧 技术方法**

使用仓库挖掘、Phabricator ITS 数据抽取、Biber 语言风格指标与主成分分析（PCA）等技术。

**📊 数据集**

利用 MediaWiki Git 仓库历史、Phabricator 任务及评论数据，并对提交者与评论者进行 WMF 关联与外部贡献者的 affiliation 标注。

**📈 对比分析**

通过对两组贡献者的语言风格向量进行 PCA 并进行 MANOVA 检验，结果未显示显著差异，表明两组在语言使用上相近。

**⚠️ 局限性**

研究仅聚焦于 MediaWiki 三个功能的案例，并且时间范围限定在 2012-2015 年，限制了结论的普适性与可推广性。

---

## 32. Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation

**arXiv ID:** 2603.12766 | [PDF](https://arxiv.org/pdf/2603.12766v1)

**作者:** Shifeng Chen `[一作]` (Beihang University), Di Huang `[通讯]` (Beihang University)

**通讯引用:** 11568 | [OpenAlex ID](https://openalex.org/A5056972984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Catalyst4D框架，将单帧3D Gaussian编辑高效传播到动态4D Gaussian场景，实现精细局部修改与全局风格迁移，并保持空间与时间一致性。

**💡 创新点**

创新点主要有两项：① Anchor-based Motion Guidance（AMG）通过稀疏锚点与无偏最优传输实现区域级运动对应，避免跨区域干扰；② Color Uncertainty-guided Appearance Refinement（CUAR）通过估计每个高斯的颜色不确定性并仅在高不确定区进行补偿，显著提升时间一致性与视觉质量。

**🔧 技术方法**

采用3D Gaussian Splatting、4D Gaussian deformation网络、无偏最优传输（Sinkhorn）、Gaussian光流、α混合、SSIM损失、CLIP相似度、VBench一致性指标等技术。

**📊 数据集**

使用了DyNeRF、MeetRoom和HyperNeRF三大数据集进行实验。

**📈 对比分析**

与Instruct 4D-to-4D、CTRL-D、Instruct-4DGS等方法对比，评估指标包括CLIP相似度和Temporal Consistency；Catalyst4D在语义一致性、时间一致性上均表现最优，并且训练时间相对更短。

**⚠️ 局限性**

局限性：未对变形网络进行重新训练，导致某些复杂运动场景下的一致性略有下降；对编辑后高斯的运动先验依赖有限，可能在极端几何变化或遮挡密集区产生细微失真。

---

## 33. Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning

**arXiv ID:** 2603.12606 | [PDF](https://arxiv.org/pdf/2603.12606v1)

**作者:** Zesheng Yang `[一作]` (Southern University of Science and Technology), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5974 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个名为 D-Negation 的视觉-语言数据集，并设计了 Grouped Opposition-Based Learning（GOBL）机制，以提升模型对否定语义的理解与定位能力。

**💡 创新点**

创新点在于首次构建正负语义配对的视觉定位数据集，并通过对抗式学习在跨模态融合层引入对比与排他性约束，显式建模否定逻辑。

**🔧 技术方法**

主要技术包括使用 GPT‑4V 生成正负描述、GOBL 的两种损失（PNC 与 TSO）以及仅微调视觉‑语言融合模块的轻量化训练。

**📊 数据集**

使用了新构建的 D‑Negation 数据集（约13.9k图像、139.98k描述）以及公开基准 D^3、RefCOCO，并在 Grounding‑DINO、APE 等模型上进行实验。

**📈 对比分析**

与现有最先进模型相比，GOBL 在负语义评估上提升约 5.7 mAP，整体 mAP 提升 4.4–5.7 点，仅微调不到 10% 参数，显著提升效率与性能。

**⚠️ 局限性**

局限性在于数据规模有限，难以覆盖多实例、多属性的复杂场景；当前仅改进融合模块，未深入优化视觉特征提取。

---

## 34. ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection

**arXiv ID:** 2603.12409 | [PDF](https://arxiv.org/pdf/2603.12409v1)

**作者:** Mattia Bernardi `[一作]` (University of Modena and Reggio Emilia), Simone Calderara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 5303 | [OpenAlex ID](https://openalex.org/A5075481810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Aligned Basis Relocation for Adaptation（ABRA）框架，实现在无目标域样本的条件下，将源域已学习的类别知识迁移至目标域，并支持开放词汇检测下的跨域适配。

**💡 创新点**

创新点包括：①将域知识与类别知识分离，使用“Objectification”构建无类别域专家；②利用SVFT生成轻量化类别专家；③通过SVD旋转对齐源/目标域空间，将类别残差闭式转移至目标域，实现无需目标数据的“传送”。

**🔧 技术方法**

采用预训练的开放词汇检测模型（如Grounding DINO）为基座，结合SVFT（奇异向量引导微调）、Objectification、Orthogonal Procrustes旋转对齐，以及任务算子类比和ParamΔ等技术。

**📊 数据集**

使用Cityscapes→Foggy Cityscapes数据集以及SDGOD（Day‑Clear、Day‑Foggy、Dusk‑Rainy、Night‑Rainy、Night‑Clear）五种光照/天气条件进行实验。

**📈 对比分析**

与全微调、零样本、源域微调、Task Analogy、ParamΔ等基线对比，ABRA在所有域移位上均显著优于基线，接近全微调上限，尤其在Night‑Rainy等最难场景中仍保持高mAP。

**⚠️ 局限性**

局限性包括：仍需在目标域上训练域专家（但不需类别数据）；在极端域移位下旋转对齐的效果可能受限；目前实验范围主要集中在合成或有限天气场景，未覆盖更广泛的跨领域情况。

---

## 35. Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences

**arXiv ID:** 2603.13100 | [PDF](https://arxiv.org/pdf/2603.13100v1)

**作者:** Wenxi Wu `[一作]` (King's College), Martim Brandão `[通讯]` (King's College)

**通讯引用:** 475 | [OpenAlex ID](https://openalex.org/A5088379096)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过构建 558 条语言约束的机器人运动规划任务数据集，系统评估了四种查询方式下 Vision‑Language Models 在根据文本指令选择最符合用户空间偏好的机器人轨迹的能力，并对小模型进行少量样本微调以提升性能。

**💡 创新点**

创新点在于：①首次将 VLM 直接用于评分与挑选运动轨迹；②对比四种图像查询策略，揭示单图全景查询在精度与计算成本上的优势；③证明通过少量样本微调即可显著提升小规模 VLM 的空间推理准确率；④对 token 预算与准确率之间的线性关系给出了量化分析。

**🔧 技术方法**

使用的技术包括：采样式运动规划器 BiRRT 与 PRM 生成多样化轨迹，K‑means 聚类筛选代表路径；VLM（Qwen2.5‑VL‑72B、GPT‑4o、LLaVA1.5）通过提示工程对图像进行打分；Supervised Fine‑Tuning（SFT）对小型 VLM 进行少样本微调；并对不同查询方式的 token 消耗进行统计。

**📊 数据集**

数据集为基于 iGibson 真实住宅重建场景（Ihlen1int、Pomaria1int、Beechwood0int、Benevolence1int、Merom0int）生成的 558 条任务，其中包含 126 条导航任务和 432 条机械臂操作任务，配有多路径图像和对应文本指令及真值路径。

**📈 对比分析**

比较方法：在四种查询方式（单图、单图多图、可视化上下文、截图画廊）下对 Qwen2.5‑VL、GPT‑4o、LLaVA1.5 进行零样本和微调后评估。结果显示 Qwen2.5‑VL 在单图查询上实现 71.4% 的整体准确率（邻近任务 74.4%，路径风格 63.9%），GPT‑4o 较低；微调后小模型提升 20–60% 的准确率；同时发现准确率随 token 数量线性下降。

**⚠️ 局限性**

局限性包括：VLM 在识别“最短/最长”路径时易失效；存在“幻觉”现象，可能选错不存在的颜色路径；在路径风格类任务上的准确率仍低于邻近任务；需要改进可视化方式和更大规模的微调，以提升鲁棒性并真正集成进机器人运动规划管线。

---

## 36. HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification

**arXiv ID:** 2603.12693 | [PDF](https://arxiv.org/pdf/2603.12693v1)

**作者:** Andrey V. Savchenko `[一作]` (Sber AI Lab), Kseniia Tsypliakova `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一套轻量化、多模态的情感与暴力检测系统，在ABAW-10竞赛中实现面部情绪识别、VA估计、AU检测以及细粒度暴力检测。

**💡 创新点**

通过预训练的EfficientNet/EmotiEffNet情感编码器提取嵌入，再结合GLA校准和置信度过滤的轻量MLP，加入滑动窗口平滑和音频特征融合，提升了时序一致性与类别失衡处理。

**🔧 技术方法**

预训练EfficientNet/EmotiEffNet、MLP、GLA（Logit Adjustment）、滑动窗口平滑、wav2vec 2.0音频特征、TCN、ConvNeXt、骨架与光流等多模态融合。

**📊 数据集**

AffWild2（情绪、VA、AU）、DVD（暴力检测）以及公开的CLIP、Wav2vec、MediaPipe Pose等辅助特征。

**📈 对比分析**

在四个ABAW-10子任务上，与挑战基线及往届顶级参赛者对比，显著提升了EXPR、VA、AU的F1/CCC指标和VD的宏F1至0.78以上，证明轻量化设计同样具备竞争力。

**⚠️ 局限性**

依赖预训练模型对外部数据的迁移效果，长程时序建模与不确定性估计仍不足，且在多模态融合与极端条件下的鲁棒性尚需进一步验证。

---

## 37. RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction

**arXiv ID:** 2603.12666 | [PDF](https://arxiv.org/pdf/2603.12666v1)

**作者:** Hanbum Ko `[一作]` (Korea University), Sungwoong Kim `[通讯]` (Korea University)

**通讯引用:** 3830 | [OpenAlex ID](https://openalex.org/A5087294413)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RetroReasoner，一种基于化学家策略的推理型大语言模型，用于预测给定产物的合成前体。

**💡 创新点**

创新点在于引入结构化的战略推理流程（四步推理 + 链接文本）和SyntheticRetro生成的训练数据，并通过回路（round‑trip）奖励进行强化学习，以提升可行性与多样性。

**🔧 技术方法**

技术包括：大语言模型（Qwen3‑8B），SyntheticRetro基于GPT‑oss‑20B生成推理文本，SFT（监督微调）+ RL（基于回路准确率的奖励，采用GRPO），以及前向合成模型做回路验证。

**📊 数据集**

使用ORDerly数据集进行训练与评估，构造了普通、罕见反应模板以及罕见原子/词条的硬核子集。

**📈 对比分析**

与多类基线（分子预测 LLM、分子推理 LLM、通用 LLM）以及无推理的Prediction‑Only进行对比；在所有指标（greedy、sampling、可行性、路径多样性）上均显著优于对照组，尤其在罕见案例中提升更为明显。

**⚠️ 局限性**

局限包括：未考虑实际合成条件（温度、压力等）、仅针对小分子/有机分子，且推理仅停留在机理层面，未涵盖更复杂的电子转移等高级推理。

---

## 38. DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training

**arXiv ID:** 2603.12840 | [PDF](https://arxiv.org/pdf/2603.12840v1)

**作者:** Ridwan Arefeen `[一作]` (Singapore Institute of Technology), Timothy Liu `[通讯]` (NVIDIA AI Technology Centre)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种双流声纹匿名攻击器DAST，并通过三阶段训练（基于清洁语音、语音转换数据以及目标匿名语音）实现对不同匿名系统的高效识别。

**💡 创新点**

创新点包括：1）将自监督学习特征与传统声谱特征通过中层融合结合，充分挖掘残留说话人信息；2）设计以语音转换为桥梁的跨域训练阶段，显著提升对未知匿名系统的泛化能力；3）实现轻量级的低资源适配（仅10%数据即可超过现有最佳攻击器）。

**🔧 技术方法**

技术：双流网络（ECAPA‑TDNN编码器）+ WavLM‑Large自监督特征+中层Hadamard融合+AAM‑Softmax训练；三阶段学习策略（基于清洁语音预训练、跨域VC数据训练、目标数据微调）。

**📊 数据集**

数据集：Stage I使用VoxCeleb2；Stage II使用SSTC（LibriSpeech→VoxCeleb，8种VC系统，共2.6M句子）；Stage III使用VPAC挑战的LibriSpeech train‑clean‑360（7种匿名系统，921人）。

**📈 对比分析**

与VPAC baseline、VPAC‑Top1、VoxAttack等现有攻击器在7个匿名系统上对比，DAST(100%)在所有系统上均取得最低EER，特别在T10‑2（7.04% vs 22.46%）和T12‑5（18.89% vs 25.63%）上表现最突出；DAST(10%)亦超越所有基线。

**⚠️ 局限性**

局限性：训练仍需大规模清洁语音和多种VC数据，模型对完全不同的匿名算法或语音域（如低质量、嘈杂环境）可能适配不足；轻量级适配仍需收集一定量的目标匿名数据；未探讨对说话人属性（年龄、性别等）隐私泄露的全面评估。

---

## 39. ELLA: Generative AI-Powered Social Robots for Early Language Development at Home

**arXiv ID:** 2603.12508 | [PDF](https://arxiv.org/pdf/2603.12508v1)

**作者:** Victor Nikhil Antony `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**通讯引用:** 3332 | [OpenAlex ID](https://openalex.org/A5017287995)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文开发了一款名为ELLA的生成式AI驱动社交机器人，通过互动故事讲述支持4–6岁儿童的早期语言发展，并通过三阶段人本设计（访谈、家庭工作坊、为期8天的家庭部署）迭代优化其交互与内容；

**💡 创新点**

创新点在于将大语言模型实时生成个性化故事与对话、机器人本体化表达以及基于家庭实际场景的迭代设计流程，首次将生成式AI与可部署社交机器人结合，实现自适应、可扩展的家庭语言学习；

**🔧 技术方法**

技术层面采用GPT‑5生成故事与提问，GPT‑4o mini TTS实现语音输出，GPT‑OSS 120B处理交互生成，Llama‑Guard‑4‑12B做内容审核，VAD+语义检测实现低延迟轮次，Xpress3D将文本映射为面部与肢体动作，机器人硬件配备5轴伺服执行器；

**📊 数据集**

使用的数据集主要来自本研究的亲子访谈（7位家长+5名教师）、12次家庭工作坊以及10名儿童为期8天的部署数据，并采用PPVT风格测验、日记、访谈记录等进行评估；

**📈 对比分析**

通过前后PPVT测验比较，儿童在8天内平均识别词汇量提升2.8个（显著提升，p<0.01）；此外记录了使用时长、故事数、交互次数等指标，但未与传统干预或对照组进行对比；

**⚠️ 局限性**

局限性包括样本规模小（10名儿童），部署时间短，缺乏长期跟踪与保留效果评估，未设置对照组，且仅在英语环境下测试，缺乏跨文化与低收入家庭等多元情境验证。

---

## 40. Deployment-Oriented Session-wise Meta-Calibration for Landmark-Based Webcam Gaze Tracking

**arXiv ID:** 2603.12388 | [PDF](https://arxiv.org/pdf/2603.12388v1)

**作者:** Chenkai Zhang `[一作]` `[通讯]` (Independent Researcher), Chenkai Zhang (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aaccfe5c-6b26-4208-b23c-35331481e142` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化、面部关键点驱动的眼球追踪方法——EMC-Gaze，该方法通过E(3)等变图神经网络编码三维面部关键点，并在每个会话仅用9点静态校准样本拟合闭式岭回归实现快速部署；

**💡 创新点**

创新点包括：①在训练中将闭式岭回归嵌入可微元学习框架，使得共享编码器直接为实际部署时的低容量校准器做准备；②利用E(3)等变GNN与两视角一致性约束，抑制全局刚性运动对嵌入的泄漏；③加入眼部局部几何特征、双眼融合以及虹距不变特征，以增强对头部运动的鲁棒性；④在训练阶段添加辅助3D视角监督与平滑追踪连续性损失，进一步提升时序一致性。

**🔧 技术方法**

使用的技术包括：E(3)等变图神经网络、闭式岭回归（可微）、两视角一致性损失、虹距不变特征、辅助3D视角监督、Meta-learning（episodic training）、MediaPipe 3D面部关键点提取、ONNX Runtime Web实现浏览器端推理。

**📊 数据集**

采用的数据集主要有：自建的EyeTrax数据库（约50+会话，含固定与自由头运动；33会话用于交互评测；10会话用于平滑追踪评测），以及公开的MPIIFaceGaze数据集（15人留一人外评）。

**📈 对比分析**

与基线方法（岭回归、Elastic Net、无等变GNN的Meta GNN）在相同面部关键点输入、相同校准流程下进行比较。结果显示：在33会话交互评测中，EMC-Gaze整体RMSE为5.79°，比Elastic Net的6.68°低约0.9°；在保留主体分割评测中，平均RMSE为5.66°，低于Elastic Net的6.49°；在MPIIFaceGaze LOPO评测中，16-shot时EMC-Gaze为8.82°，明显优于Elastic Net的10.83°。

**⚠️ 局限性**

局限性包括：依赖高质量的MediaPipe关键点检测，光照、遮挡或运动模糊会导致关键点失效；仅输出二维屏幕坐标，缺乏完整的三维视线信息；每个会话仍需手动校准，无法完全免校准；数据集规模与人群多样性有限，未能覆盖所有极端使用场景；在线实时校准与头部运动补偿的工程实现仍需进一步验证。

---

## 41. Byzantine-Robust Optimization under $(L_0, L_1)$-Smoothness

**arXiv ID:** 2603.12512 | [PDF](https://arxiv.org/pdf/2603.12512v1)

**作者:** Arman Bolatov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Eduard Gorbunov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5087594198)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并理论分析了针对（L₀,L₁）平滑目标的拜占庭鲁棒归一化带动量随机梯度下降算法Byz‑NSGDM，在分布式优化中对抗拜占庭攻击与数据异构。

**💡 创新点**

将归一化动量与拜占庭鲁棒聚合相结合，并在（L₀,L₁）平滑框架下首次给出非渐进O(K⁻¹/⁴)收敛率与拜占庭偏差上界，填补了归一化方法在拜占庭鲁棒与异构场景的理论空白。

**🔧 技术方法**

归一化梯度、动量、(δ,κ)‑鲁棒聚合（几何中值、坐标中值、NNM等）、最小化线性算子以及严格的下降不等式与动量误差分解分析。

**📊 数据集**

MNIST（异构划分）、合成四次多项式（L₀,L₁）平滑、字符级语言模型（莎士比亚文本）。

**📈 对比分析**

与基线动量SGD、带衰减动量SGD以及Krum/RFA/CM等聚合方式比较，在Bit Flipping、Label Flipping、Mimic、ALIE攻击下，Byz‑NSGDM在MNIST上准确率提升至约86%/95%，梯度范数下降更快，语言模型困惑度下降至10–12，整体表现稳健且对超参不敏感。

**⚠️ 局限性**

受拜占庭偏差上界κζ限制；需预知迭代次数或使用双倍技巧；仅在欧氏范数下证明，非欧氏情况待扩展；实验覆盖的深度网络规模有限，且对高维问题的适用性仍需进一步验证。

---

## 42. Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating

**arXiv ID:** 2603.12598 | [PDF](https://arxiv.org/pdf/2603.12598v1)

**作者:** Xiangkui Cao `[一作]` (Chinese Academy of Sciences), Xilin Chen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 35541 | [OpenAlex ID](https://openalex.org/A5083420537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Neural Gate神经门控模型编辑方法，用于提升大型视听模型对隐私敏感问题的拒绝率；

**💡 创新点**

在神经元级别进行梯度截断，通过跨样本统计筛选一致活跃的隐私相关神经元，避免全局编辑导致性能下降，兼顾安全与实用；

**🔧 技术方法**

采用可学习向量对隐私特征进行扰动，生成跨样本神经元活跃度掩码，结合梯度截断与局部微调实现精细编辑；

**📊 数据集**

构建PrivacyPair配对数据集（同一隐私主体的敏感/非敏感问答），并在MiniGPT、LLaVA等模型上实验，使用MLLMGuard、ScienceQA、MME、POPE等评测数据；

**📈 对比分析**

与DINM、MEMIT、AlphaEdit、SKU、MemFlex等基线对比，Neural Gate在PrivacyPair、MLLMGuard等隐私评测中获得最高拒绝率（>94%），且在通用任务上保持或超过基线的准确率；

**⚠️ 局限性**

对未见隐私类别（如无人机）泛化不均匀，MiniGPT上多层编辑效果不佳，且仍需进一步扩展隐私词典与特征学习以提升覆盖率。

---

## 43. Upward Spatial Coverage Recovery via Movable Antenna in Low-Altitude Communications

**arXiv ID:** 2603.12792 | [PDF](https://arxiv.org/pdf/2603.12792v1)

**作者:** Kan Yu `[一作]` (Beijing University of Posts and Telecommunications), Zhiyong Feng `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 12616 | [OpenAlex ID](https://openalex.org/A5001714538)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种可移动天线（MA）系统，通过在三维空间内联合优化天线位置和波束成形，实现低空空域的向上空间覆盖最大化。

**💡 创新点**

创新点在于：①构建了基于体素网格的空间覆盖最大化模型；②提出了结合粒子群优化与模拟退火的混合算法（PSO‑SA）来高效求解MA位置与波束的耦合非凸优化；③通过引入机械倾斜角度的可调性进一步扩展了空间自由度。

**🔧 技术方法**

使用的技术包括：Rician信道模型、仿真仿射波束成形（MRT）、PSO‑SA优化框架以及体素网格离散化。

**📊 数据集**

使用的数据集为仿真环境，参数设置为M=9天线、频率3.5 GHz、机械倾斜角范围[-20°,20°]、移动范围0.5λ–5λ等，未使用真实测量数据。

**📈 对比分析**

通过与四种基准方案（FPA_noBF、FPA_BF、MA_BF、4DMA_BF）比较，实验表明：MA_BF在0–300 m和0–600 m高度范围内覆盖率分别提升26.8%和29.65个百分点；进一步优化机械倾斜的4DMA_BF在同一区域分别达到88.31%和76.4%的覆盖率，显示出显著性能优势。

**⚠️ 局限性**

局限性包括：①仅考虑无干扰的低空环境，未考虑邻区干扰；②信道模型仅为Rician，未涵盖多路径衰落或阻塞情况；③仅通过仿真验证，缺乏实地部署或实验数据；④机械倾斜角度的动态调节仍受限于物理实现。

---

## 44. Semantic Invariance in Agentic AI

**arXiv ID:** 2603.13173 | [PDF](https://arxiv.org/pdf/2603.13173v1)

**作者:** I. de Zarzà `[一作]` (Luxembourg Institute of Science and Technology), Carlos T. Calafate `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 9957 | [OpenAlex ID](https://openalex.org/A5066242048)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并使用一套基于变形规则的变形测试框架，对多步推理 LLM 代理在不同语义保持变形下的鲁棒性进行系统评估。

**💡 创新点**

①提出八类语义保持的 metamorphic relation（结构、冗长、上下文）；②发现模型规模并不一定带来更高的语义不变性；③揭示不同体系结构族的特定脆弱性模式；④指出对比变形普遍导致所有模型的性能下降。

**🔧 技术方法**

metamorphic testing（定义 MR 并评估输出一致性）、语义相似度测量（Sentence‑Transformers all‑MiniLM‑L6‑v2）、多步推理追踪评估、统计检验（Mann‑Whitney U、Kruskal‑Wallis）以及基于 LLM 生成的变形实现。

**📊 数据集**

19 个多步推理问题，覆盖物理、数学、化学、经济、统计、生物、微积分、优化等八个科学领域，按难度划分为 Easy、Medium、Hard 三级。

**📈 对比分析**

与传统固定句子基准（MMLU、GSM8K 等）对比，使用 MAE（MAD）和稳定率评估；实验显示 Qwen3‑30B‑A3B 在 79.6% 的变形下保持答案不变，而大模型如 Hermes‑405B、gpt‑oss‑120b 在相同条件下稳定率仅 27%–67%。

**⚠️ 局限性**

评估样本有限（仅 19 题），变形生成依赖 LLM 可能引入语法偏差；单次推理可能不反映模型在多次采样或长期交互中的行为；未覆盖所有可能的推理领域和更大规模模型。

---

## 45. SciDesignBench: Benchmarking and Improving Language Models for Scientific Inverse Design

**arXiv ID:** 2603.12724 | [PDF](https://arxiv.org/pdf/2603.12724v1)

**作者:** David van Dijk `[一作]` (CellType Inc.), Ivan Vrkic `[通讯]` (CellType Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个520个模拟器驱动的逆向设计基准，覆盖14个科学领域，并在此基准上对七个前沿大型语言模型进行评估；同时提出了一种基于模拟器反馈的RL训练范式（SFT+GRPO），用于在多领域任务上提升模型性能。

**💡 创新点**

创新点主要有三：① 将前向科学模拟器统一封装为RL环境，直接将其作为奖励信号；② 设计多维度评估模式（单回合、5步/20步长周期反馈、单次设计与优化），揭示模型在不同交互层面的能力差异；③ 通过基准发现长周期反馈利用与零射能力是互补的，并验证了训练后模型在若干领域的显著提升。

**🔧 技术方法**

使用的技术包括：前向科学模拟器（RDKit、SciPy、AutoDock Vina、COBRApy等）、前沿LLM（GPT‑4o、Claude Opus 4.6、Sonnet 4.5/4.6、Gemini 3.1 Pro/2.0 Flash、Qwen3‑8B），以及基于QLoRA的SFT、Group Relative Policy Optimization（GRPO）进行RL训练，支持链式推理与结构化JSON输出。

**📊 数据集**

数据集为：520个任务（260 de novo + 260 optimization），每个任务包含目标说明、前向oracle实现和奖励函数；训练阶段使用从各域采样、执行后保留的（goal, design）对（约数千条）作为SFT与GRPO的输入；测试集与训练集严格分离，包含10个共享核心域用于统一比较。

**📈 对比分析**

比较方法：在每个任务上进行K=3次尝试，记录解析率、有效率、成功率和最高奖励；对比前沿模型的零射一回合成功率（最高29%）与20步长周期（最高76%）以及优化场景。训练后Qwen3‑8B在ADMET（30%→41%）、PK/PD（24%→36%）和Docking（42%→59%）上均实现显著提升，证明模拟器反馈训练有效。

**⚠️ 局限性**

局限性：① 基准仅涵盖14个领域，未包含更复杂的高保真模拟器；② 训练规模有限，仅对单一8B模型进行多域案例研究；③ 长周期反馈需要多次模拟器调用，成本高；④ 对经验主义oracle可能存在奖励劫持风险；⑤ 模型在不同域间的迁移效果不确定；⑥ API和模型版本更新可能导致结果漂移。

---

## 46. Anchored Alignment: Preventing Positional Collapse in Multimodal Recommender Systems

**arXiv ID:** 2603.12726 | [PDF](https://arxiv.org/pdf/2603.12726v1)

**作者:** Yonghun Jeong `[一作]` (Ulsan National Institute of Science and Technology), Yeon-Chang Lee `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 44524 | [OpenAlex ID](https://openalex.org/A5100383157)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AnchorRec框架，通过在轻量级投影域使用锚点间接对齐多模态嵌入，避免了直接对齐导致的模态位置坍塌和交互信号主导。

**💡 创新点**

创新点在于引入投影域间接对齐与锚点对齐损失（AAL和AMP），既保持模态特异性，又实现跨模态一致性，解决了ID主导和模态坍塌问题。

**🔧 技术方法**

使用LightGCN作为ID编码器，BERT/VGGNet/BEiT3提取文本/视觉/多模态特征，MLP投影层，InfoNCE对齐损失，AMP正则化等技术。

**📊 数据集**

在四个亚马逊商品数据集（Baby, Sports, Office, Video Games）上进行实验。

**📈 对比分析**

与VBPR, LATTICE, FREEDOM, LGMRec, SMORE, BM3, DA-MRS, AlignRec等八个最先进的MMRS基线进行对比，AnchorRec在保持或略高的Recall@N/NDCG@N的同时，显著提升了多模态表达性和邻域重叠度。

**⚠️ 局限性**

主要限制在于对投影域的依赖需要额外的超参数调优，且在极低交互稀疏环境下表现未充分验证。

---

## 47. Surprised by Attention: Predictable Query Dynamics for Time Series Anomaly Detection

**arXiv ID:** 2603.12916 | [PDF](https://arxiv.org/pdf/2603.12916v1)

**作者:** Kadir-Kaan Özer `[一作]` (Mercedes-Benz AG), Markus Enzweiler `[通讯]` (Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无监督的多变量时间序列异常检测器AxonAD，利用注意力查询向量的可预测性与重构误差共同生成异常分数；

**💡 创新点**

创新点在于将多头注意力查询视为短期可预测信号，采用EMA目标编码器和掩蔽自监督的JEPA式损失仅预测查询向量，并将查询不匹配度与重构误差稳健标准化后相加，形成对结构依赖异常高度敏感且无标签阈值的检测方法；

**🔧 技术方法**

使用的技术包括双向自注意力重构路径、历史仅预测分支、EMA目标网络、JEPA式掩蔽预测、余弦距离查询匹配、稳健中位数-四分位数标准化、Transformer架构和一维卷积预测器；

**📊 数据集**

实验数据集包括公司内部的19通道车载遥测（80,000步、30个异常区间）以及公开的TSB-AD多变量基准（17个数据集、180条系列）；

**📈 对比分析**

与Isolation Forest、LSTMAD、USAD、VAE、Transformer基线等方法比较，车载遥测上AxonAD的AUC-PR 0.285、Event-F1 0.420远超第二名SISVAE的0.128和0.255；在TSB-AD上AUC-PR 0.437、VUS-PR 0.493、Range-F1 0.471，均领先所有基线；ablation实验表明查询预测和双分量得分是性能提升的主要驱动力；

**⚠️ 局限性**

局限性包括训练阶段需要梯度更新和EMA维护，训练成本相对较高；对预测时延和多头数量敏感；在极端非平稳或查询可预测性弱的场景下可能表现下降；此外，缺乏对极端mask比例或预测时延的深入调优。

---

## 48. Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**arXiv ID:** 2603.13069 | [PDF](https://arxiv.org/pdf/2603.13069v1)

**作者:** Ann Dooms `[一作]` (Vrije Universiteit Brussel), Ann Dooms `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1340 | [OpenAlex ID](https://openalex.org/A5054878147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

阐述了确定性扩散模型（如DDIM）如何可以视作分块迭代函数系统（PIFS），揭示了生成过程中的两阶段（降噪与细节合成）与生成器的块结构及吸引子几何特性。

**💡 创新点**

创新点在于将确定性扩散逆过程与分块迭代函数系统理论相结合，提出了收缩阈值、扩张阈值和KY维数等量化指标，用以解释并统一多种经验设计（余弦偏移、分辨率调节、Min‑SNR权重、采样步长分配）以及其对图像质量（FID）的提升。

**🔧 技术方法**

主要技术包括确定性扩散公式、Jacobian 计算（JVP/VJP）、块结构分解、Lyapunov谱与Kaplan–Yorke维数分析、信息增益与KL散度关联、Attention‑entropy 与跨块耦合关系。

**📊 数据集**

使用的主数据集是 CIFAR‑10 与 CelebA‑HQ（高分辨率人脸），并在公开的预训练 DDPM 模型上进行实验验证。

**📈 对比分析**

通过对比不同噪声调度（线性、余弦原版、余弦加偏移、50 步 DDIM 采样）以及 Min‑SNR 权重和 Align‑Your‑Steps 调度，实验表明余弦偏移与 Min‑SNR 重新加权能显著提升 FID（相较于原始方案提升数倍），且步长分配集中在低噪声阶段与理论一致，整体性能优于传统固定步长方法。

**⚠️ 局限性**

局限性包括：分析基于块对角协方差与高斯基线假设，真实数据的非高斯性需额外校正；Jacobian 估计受数值误差影响，尤其在细节阶段；对不同网络架构（如跨层跳连、不同注意力位置）的推广仍需进一步验证。

---

## 49. Tight (S)ETH-based Lower Bounds for Pseudopolynomial Algorithms for Bin Packing and Multi-Machine Scheduling

**arXiv ID:** 2603.12999 | [PDF](https://arxiv.org/pdf/2603.12999v1)

**作者:** Karl Bringmann `[一作]` (ETH Zurich), Karol Węgrzycki `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5055238529)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文证明了多机调度问题在某些目标函数下的伪多项式复杂度上界的不可行性，具体通过从3-SAT或K-SAT构造实例展示其SETH/ETH难度；

**💡 创新点**

创新点在于将经典的Behrend集合与多机调度结合，设计了通信通道与平均自由集合的编码方法，以实现对多机调度问题的精确时间下界；

**🔧 技术方法**

核心技术包括Sparsification Lemma、强k-平均自由集合构造、整数分拆与通信通道设计以及多阶段位块编码；

**📊 数据集**

论文未使用传统机器学习数据集，而是基于理论构造的SAT实例和整数分割集合；

**📈 对比分析**

与传统的多机调度多项式或伪多项式算法相比，本研究给出了与SETH/ETH相匹配的紧界，即若存在更快算法则违背SETH/ETH；

**⚠️ 局限性**

局限性在于只能处理固定k且在整数范围接近W的实例，对一般调度问题或更宽泛的参数范围尚无完整下界；

---

## 50. VFM-Recon: Unlocking Cross-Domain Scene-Level Neural Reconstruction with Scale-Aligned Foundation Priors

**arXiv ID:** 2603.12657 | [PDF](https://arxiv.org/pdf/2603.12657v1)

**作者:** Yuhang Ming `[一作]` (Shanghai Jiao Tong University), Wanzeng Kong `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 4317 | [OpenAlex ID](https://openalex.org/A5003649465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 VFM-Recon 框架，利用 Vision Foundation Model (VFM) 的几何先验进行场景级神经体素重建，并通过轻量级尺度对齐与深度重投影实现跨域尺度一致；

**💡 创新点**

创新点包括① 轻量化尺度对齐模块解决 VFM 深度的尺度模糊和跨帧不一致；② 深度重投影步骤通过初始体素融合生成高质量一致深度；③ 在 VFM 特征上插入可学习的 MLP 适配器，将预训练的 VFM 表征融入体素重建网络，兼顾跨域鲁棒性与局部细节；

**🔧 技术方法**

使用 VGGT 预训练 VFM、SuperPoint+LightGlue 进行特征匹配、基于三角化的尺度恢复、图优化求解尺度、体素融合、3D U‑Net 解码器、MLP 适配器、EfficientNetV2‑S 作为 CNN 编码器；

**📊 数据集**

主要数据集为 ScanNet（训练/测试），以及跨域评估的 TUM RGB‑D 与 Tanks & Temples；

**📈 对比分析**

与多种基准（VoRTX、NeuralRecon、FineRecon、GP‑Recon、GeoRecon 等）以及 VFM 基础模型（VGGT、MoGe‑2）进行比较，VFM‑Recon 在 ScanNet 上的 F1 达到 74.3，深度测量精度 (Abs.Rel.) 5.0，跨域场景中 Tanks & Temples 的 F1 高达 70.1，明显优于 VGGT（51.8）和 FineRecon（33.1）等；

**⚠️ 局限性**

局限性包括对深度重投影引入的量化误差在域内可能略微影响细节，尺度对齐依赖良好特征匹配和相机姿态，方法在极端动态或大尺度户外环境中尚未充分验证。

---

## 51. TRACE: Temporal Rule-Anchored Chain-of-Evidence on Knowledge Graphs for Interpretable Stock Movement Prediction

**arXiv ID:** 2603.12500 | [PDF](https://arxiv.org/pdf/2603.12500v1)

**作者:** Qianggang Ding `[一作]` (Universite de Montreal), Bang Liu `[通讯]` (Universite de Montreal)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 TRACE 框架，将符号规则、时序知识图谱多跳探索与 LLM 引导的文本证据融合，实现可解释的下一日股价涨跌预测。

**💡 创新点**

创新点包括：①自动挖掘高置信度的关系规则作为搜索先验；②规则约束的多跳图搜索与 LLM 关系筛选相结合；③文本锚定的证据链生成与路径聚合决策；④在单一管道中兼顾预测性能与可解释性。

**🔧 技术方法**

采用技术包括：时序知识图谱构建、规则挖掘（AMIE/AnyBURL 思路）、Beam Search 规则约束探索、LLM 关系选择与路径评估、规则置信度与文本匹配的多因子评分、路径聚合判决。

**📊 数据集**

使用 2022‑2023 年 S&P 500 公司及其事件、新闻、财务、产品等 42,188 个实体、174,415 条时间戳边的知识图谱；配合 487 家公司每日 OHLCV 价格数据，形成 2023‑2024 年的测试集。

**📈 对比分析**

与 Momentum、XGBoost、新闻情感（Qwen）、T‑GNN、若干代理框架对比；TRACE 在 1‑日预测上准确率 55.1%、召回 71.5%、F1 60.8%，显著优于基线；在 Top‑10 买入持有回测中实现 41.7% 总回报、Sharpe 2.00，超过所有对比方法。

**⚠️ 局限性**

局限性包括：知识图谱覆盖不足与漂移影响、仅处理 1‑日预测，难以直接推广到多期预测；LLM 仍存在幻觉风险；路径解释受限于规则库和文本可用性；计算成本高，需要更多资源进行实时更新。

---

## 52. Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection

**arXiv ID:** 2603.13070 | [PDF](https://arxiv.org/pdf/2603.13070v1)

**作者:** Yunzhuo Chen `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20511 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两种方法：RAPTA和ADMCD，用于在训练时减少扩散模型的记忆化并在推理时检测复制。

**💡 创新点**

创新点在于训练时的区域感知提示增强和无监督的多模态注意力融合检测器；结合提示多样性与注意力融合实现零训练复制检测。

**🔧 技术方法**

使用了预训练的目标检测器（Faster R‑CNN）、CLIP文本和图像编码器、Vision Transformer、ResNet纹理编码器以及轻量级Transformer进行特征融合。

**📊 数据集**

主要使用LAION‑10k作为训练集，评估集为1,200对生成/参考图像，另外在不同扩散模型（DCR、LDM‑T2I、Stable Diffusion 2.1）上测试。

**📈 对比分析**

与多种基线（SSIM、LPIPS、ORB、SSCD、DreamSim）比较，RAPTA在保持FID/KID同等或更好时显著降低复制率，ADMCD在多种攻击下保持高相似度并优于单模态度量。

**⚠️ 局限性**

限制在于复制率仍非零，对少量复制案例（真正的近似复制）难以捕捉，且阈值需手工校准，未对大规模多样数据的泛化做充分验证。

---

## 53. Exploring the Role of User Comments Throughout the Stages of Video-Based Task-Learning

**arXiv ID:** 2603.12509 | [PDF](https://arxiv.org/pdf/2603.12509v1)

**作者:** Nayoung Kim `[一作]`, Takeo Igarashi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过访谈与共看实验，探究视频任务学习中评论在视频选择、观看与复盘等各阶段的使用情况

**💡 创新点**

首次系统揭示评论在任务学习全过程中的作用，并提出将评论与视频内容动态关联、情感可视化、类型化展示等设计机会

**🔧 技术方法**

采用定性研究方法（访谈、共看、反射性主题分析），并在设计建议中引入情感分析、时间锚定评论与基于类型的可视化技术

**📊 数据集**

使用的评论数据来源于14名受访者和8名共看参与者在YouTube上的实时评论，未使用公开大型数据集

**📈 对比分析**

本文未构建系统或进行量化性能比较，而是通过案例分析与主题讨论阐述设计方向，缺乏基准评估

**⚠️ 局限性**

研究样本单一（大学生）、仅涉及评论消费、只关注YouTube平台、缺乏多文化、多语言与跨平台的验证

---

## 54. Modal Logical Neural Networks for Financial AI

**arXiv ID:** 2603.12487 | [PDF](https://arxiv.org/pdf/2603.12487v1)

**作者:** Antonin Sulc `[一作]` `[通讯]` (Lawrence Berkeley National Lab), Antonin Sulc (Lawrence Berkeley National Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为模态逻辑神经网络（MLNN）的新架构，旨在将深度学习与符号逻辑结合，以满足金融行业对可解释性和合规性的需求。

**💡 创新点**

创新点在于将Kripke语义直接集成到神经网络架构中，使模型能够进行关于必要性、可能性、时间和知识的可微推理，从而在金融领域提供安全性和合规性。

**🔧 技术方法**

使用了模态逻辑神经网络（MLNN），结合了可微分的逻辑层和Kripke语义。

**📊 数据集**

使用了多个金融场景的案例研究，包括洗售交易、市场合谋检测、风险管理和可解释的合同审查，具体数据集包括合同理解数据集（CUAD）。

**📈 对比分析**

通过四个案例展示了MLNN的应用，结果表明在合规性、市场监测、风险管理和可解释性方面的表现优于传统方法，尤其在合规性和解释能力上表现突出。

**⚠️ 局限性**

限制在于当前模型的复杂性和对数据的依赖，未来的工作需要扩展到交易序列的时间推理和多智能体的市场监测模型。

---

## 55. SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design

**arXiv ID:** 2603.13098 | [PDF](https://arxiv.org/pdf/2603.13098v1)

**作者:** Ruogu Li `[一作]` (University of North Carolina), Mingyu Ding `[通讯]` (University of North Carolina)

**通讯引用:** 2940 | [OpenAlex ID](https://openalex.org/A5022382771)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大型多模态 CAD 数据集 SldprtNet，包含 242,000+ 工业零件的 .sldprt/.step 模型、七视图复合图像、参数化建模脚本和自然语言描述，并提供编码器/解码器实现模型与文本的双向无损转换；基于该数据集训练并评估了文本+图像双模态与单模态模型在文本驱动 CAD 生成任务上的表现。

**💡 创新点**

1) 大规模（242k）高质量工业 CAD 数据；2) 支持 13 种 SolidWorks CAD 操作的编码器/解码器，提供无损的文本↔模型双向转换；3) 多模态对齐（图像、参数文本、自然语言）提升模型对几何语义的理解；4) 通过将多视图合并为单张图像降低输入长度，显著加速推理。

**🔧 技术方法**

SolidWorks API + COM 接口实现编码器/解码器；多视图渲染宏；Qwen2.5-VL-7B 视觉语言模型用于生成自然语言描述；基于 Encoder_txt 的 Transformer 模型用于文本驱动 CAD 生成；多模态评估指标（Exact Match、Command-Level F1、Partial Match、Parameter Tolerance Accuracy）。

**📊 数据集**

SldprtNet 自身（242k+ CAD 零件），来源于 GrabCAD、McMaster‑Carr、FreeCAD 等公开平台；使用 Qwen2.5-VL-7B 生成的自然语言描述作为标注；在 50K 子集上进行基准实验。

**📈 对比分析**

对比单模态（仅 Encoder_text）与双模态（图像+Encoder_text）训练的 Qwen2.5‑7B 与 Qwen2.5‑7B‑VL。双模态模型在 Exact Match、Command‑Level F1、Partial Match 上均明显优于单模态；仅在 Parameter Tolerance Accuracy 上略逊。实验验证多模态监督显著提升了 CAD 生成的结构一致性和语义准确性。

**⚠️ 局限性**

1) 数据集规模虽大，但仍受限于公开 CAD 资源，缺乏真正的装配级复杂模型；2) 13 种 CAD 操作覆盖范围有限，无法涵盖所有工业建模技术；3) 生成的自然语言描述仍需人工校验，未实现完全自动化；4) 评估指标主要关注命令匹配，未充分评估几何精度与可制造性。

---

## 56. Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA

**arXiv ID:** 2603.12545 | [PDF](https://arxiv.org/pdf/2603.12545v1)

**作者:** Nahid Alam `[一作]` (Cohere Labs Community), Bala Krishna S Vegesna `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLaVA框架下通过控制图像编码器和二维位置编码来诊断并评估视觉语言模型在二维空间推理上的表现。

**💡 创新点**

系统评估了编码器目标和二维位置编码对空间理解的影响，发现编码器目标是主导因素，二维位置编码虽有帮助但不足。

**🔧 技术方法**

LLaVA-1.5（7B）模型替换CLIP、SigLIP、SigLIP2、AIMv2编码器，并引入2D旋转位置编码（2D-RoPE）。

**📊 数据集**

MMVP、CV-Bench、TallyQA、GQA、VSR、TopViewRS、CountBenchQA 等空间推理基准以及LLaVA训练集。

**📈 对比分析**

在相同7B LLaVA骨干上训练不同编码器/位置编码组合，在空间基准上进行对比；发现编码器优化能显著提升空间分数，最优结果在AIMv2/2D-RoPE组合上，但整体仍低于通用任务表现。

**⚠️ 局限性**

仅关注静态二维图像，未扩展到3D或动态环境；仅在256×256分辨率下实验；二维位置编码有时会降低性能，缺乏深入诊断；未覆盖更大模型规模。

---

## 57. Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering

**arXiv ID:** 2603.12533 | [PDF](https://arxiv.org/pdf/2603.12533v1)

**作者:** Yura Choi `[一作]` (Imperial College London), Stefanos Zafeiriou `[通讯]` (Imperial College London)

**通讯引用:** 22063 | [OpenAlex ID](https://openalex.org/A5080553022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套专门针对 egocentric 视角下手势驱动的问答（EGOPOINTVQA）数据集与基准，并在此基础上实现了可解释的手势理解模型；

**💡 创新点**

创新点在于将 3D 手部关键点通过轻量级适配器编码成手势意图 token（HINT），并将这些 token 与视觉、文本输入交织，显式地为多模态大模型提供手势空间-时间上下文；

**🔧 技术方法**

使用技术包括：WiLoR 3D 手姿估计、关键点适配器（Keypoint Adapter）生成 HINT token、Token interleaving、LLM LoRA 微调，以及基于多模态输入的问答训练；

**📊 数据集**

使用的数据集为 EGOPOINTVQA，包含 4,000 条合成 egocentric 视频和 400 条真实视频，总计 18,745 题答对；

**📈 对比分析**

实验对比 GPT‑4o、Qwen3‑VL、InternVL3 等 15 种 MLLM，基线平均准确率低于 70%；加入 HINT 后平均提升约 6.6%，例如 InternVL3‑14B 的 Reference 任务由 63.1% 提升至 73.8%；

**⚠️ 局限性**

局限性包括：依赖手姿估计精度，检测置信度阈值对性能影响大；在复杂多手势时间序列中仍易失效；模型对真实场景多样性的泛化受限；

---

## 58. Marked Pedagogies: Examining Linguistic Biases in Personalized Automated Writing Feedback

**arXiv ID:** 2603.12471 | [PDF](https://arxiv.org/pdf/2603.12471v1)

**作者:** Mei Tan `[一作]` (Stanford University), Dorottya Demszky `[通讯]` (Stanford University)

**通讯引用:** 3244 | [OpenAlex ID](https://openalex.org/A5052171928)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对400名八年级劝说性作文的LLM生成反馈进行分析，探究并量化了模型在根据学生属性（如种族、学习需求、动机等）时产生的刻板化语言模式（Marked Pedagogies）。

**💡 创新点**

创新点在于首次采用无监督的“Marked Words”框架结合统计词频差异与定性编码，系统识别并量化LLM反馈中的身份偏见与教学取向。

**🔧 技术方法**

使用了计算语言学技术，包括log‑odds+Dirichlet prior计算词汇差异、z-score统计显著性、词汇编码与回归浓度度量C_s(F)，以及对比提示条件下的生成反馈分析。

**📊 数据集**

数据集为PERSUADE公开数据集中的600篇中学生劝说性作文，涵盖两种写作题目。

**📈 对比分析**

通过对比标记属性提示与对照提示生成的反馈，并用浓度指标C_s(F)及线性回归评估词汇偏差，结果显示模型在标记属性时显著提升对应偏见词汇比例，表明存在系统性刻板化。

**⚠️ 局限性**

局限在于仅分析两种作文题目、四种LLM、单一美国教育背景与单一属性维度，缺乏跨体裁、跨地区和交叉属性的普适性验证。

---

## 59. Representation Learning for Spatiotemporal Physical Systems

**arXiv ID:** 2603.13227 | [PDF](https://arxiv.org/pdf/2603.13227v1)

**作者:** Helen Qu `[一作]` (Flatiron Institute), Yann LeCun `[通讯]` (New York University)

**通讯引用:** 245154 | [OpenAlex ID](https://openalex.org/A5001226970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估自监督学习方法在三种物理系统的参数估计任务中的表现，并与像素级重建和自回归模型进行对比。

**💡 创新点**

发现基于潜在空间预测的自监督方法（JEPA）能更好地保留物理信息，显著优于像素重建和自回归模型，且样本效率更高。

**🔧 技术方法**

使用 Joint Embedding Predictive Architectures (JEPA)、Masked AutoEncoder (VideoMAE ViT‑tiny/16)、DISCO（in‑context neural operator）和 MPP（自回归基础模型）等技术。

**📊 数据集**

在 The Well 提供的三类 PDE 物理系统数据集上进行实验：活性物质、Rayleigh‑Bénard 对流、剪切流。

**📈 对比分析**

通过参数估计的均方误差（MSE）对比，JEPA 在所有三组系统上相较 VideoMAE 分别降低了 51%、43% 和 28%；与 DISCO 接近或相当；MPP 表现最差。样本量上，JEPA 仅需 50% 数据即可达到 95% 的最佳性能。

**⚠️ 局限性**

局限性包括：实验仅涵盖三种系统，缺乏对更复杂或更大尺度物理场景的验证；对自监督方法在参数估计之外其他科学任务的通用性尚未充分探讨；以及对潜在空间表示的解释性仍需进一步研究。

---

## 60. Disentangled Latent Dynamics Manifold Fusion for Solving Parameterized PDEs

**arXiv ID:** 2603.12676 | [PDF](https://arxiv.org/pdf/2603.12676v1)

**作者:** Zhangyong Liang `[一作]` (National Center for Applied Mathematics, Tianjin University), Ji Zhang `[通讯]` (School of Mathematics, Physics and Computing, University of Southern Queensland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为DLDMF的物理信息化神经网络框架，用于解决参数化偏微分方程的连续时间求解，并实现参数泛化与长期时间外推的统一；

**💡 创新点**

核心创新在于空间、时间、参数的解耦与动态流融合，通过参数化编码直接初始化连续时间潜在动态，消除迭代自解码，形成参数化的连续时间潜在流；

**🔧 技术方法**

采用参数化P^2INN思想、神经Ode、隐层动态融合、SVD快速微调、自动微分、离散积分等技术；

**📊 数据集**

使用1维参数化对流扩散反应(CDR)和2维Navier–Stokes流体基准数据集，划分训练与测试时间区间及参数空间内外样本；

**📈 对比分析**

与P^2INN、PI-DeepONet、DINO、PIDO、MAD等基线对比，在In-t/Out-t的L2误差上DLDMF表现最优，Out-t误差约4%而其他方法多达30%以上，且对不同训练比例的DINO显示更稳健；

**⚠️ 局限性**

局限在于对复杂边界/几何、非平稳参数及大规模高维系统的适用性尚待验证，极端激波或高频场景的收敛性仍有挑战。

---

## 61. Algorithmic Trust and Compliance: Benchmarking Brand Notability for UK iGaming Entities in Generative Search Engines

**arXiv ID:** 2603.12282 | [PDF](https://arxiv.org/pdf/2603.12282v1)

**作者:** Julen Oruesagasti `[一作]` `[通讯]` (Interamplify), Julen Oruesagasti (Interamplify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了适用于受监管数字市场的生成式引擎优化（GEO）框架，聚焦英国iGaming行业的算法信任与可视化指标；

**💡 创新点**

创新点在于将监管合规信号转化为机器可读的结构化数据（Schema.org），构建四层实体清晰模型，并提出针对生成式引擎的“位置调整词数”可视化度量；

**🔧 技术方法**

技术主要包括检索增强生成（RAG）架构、结构化Schema标记、文本分析与可视化指标计算；

**📊 数据集**

实验数据集由约10,000条多平台（ChatGPT、Google Gemini、Perplexity）查询构成，附带SparkToro 2024年的引用分布统计；

**📈 对比分析**

通过对九种内容优化策略的实验比较，发现“引用来源”“统计数据加入”和“引用添加”分别提升可视化40%、37%和22%，相较传统关键字堆砌提升显著；

**⚠️ 局限性**

局限性包括仅研究英国iGaming且只覆盖英语生成式引擎，未检验跨行业适用性，且模型对多模态内容和未来生成式引擎的适配尚未探究；

---

## 62. Early Pruning for Public Transport Routing

**arXiv ID:** 2603.12592 | [PDF](https://arxiv.org/pdf/2603.12592v1)

**作者:** Andrii Rohovyi `[一作]` (University of New South Wales), Toby Walsh `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Early Pruning 技术，在公共交通路由算法（RAPTOR 及其多种变体）中对传输放松阶段进行轻量级剪枝，显著加速查询。

**💡 创新点**

创新点在于：先把每个站点的所有传输边按持续时间预排序，然后在遍历时一旦发现某条传输无法比当前已知的目标到达时间更早，即可立即停止后续遍历，保证不失最优性并几乎不增加额外成本。

**🔧 技术方法**

使用技术包括：RAPTOR 框架、边预排序（一次性 600 ms 内完成）、基于到达时间或多目标主导关系的剪枝规则、在多种 RAPTOR 变体（ULTRA‑RAPTOR、McRAPTOR、BM‑RAPTOR 等）中嵌入。

**📊 数据集**

数据集为真实公交网络：瑞士（Swisscom）和伦敦（Transport for London），包含站点、路线、车辆、转移图及其 ULTRA 短路版本。

**📈 对比分析**

比较方法：对六种 RAPTOR 变体在 1000 条随机查询上进行基线与 Early Pruning 加速后对比，平均查询时间显著下降，最优提升达 57%（McRAPTOR），并与图密度相关性显著（Pearson 0.62）。

**⚠️ 局限性**

局限性：在极稀疏图或已使用 ULTRA 短路的变体加速有限；CSA 等算法不受显著影响；需要一次边排序（虽快，但仍需在新增转移时更新）；实验仅在瑞士和伦敦两网络上验证，未知在更大或更复杂网络中的表现。

---

## 63. Bridging Sequential and Contextual Features with a Dual-View of Fine-grained Core-Behaviors and Global Interest-Distribution

**arXiv ID:** 2603.12578 | [PDF](https://arxiv.org/pdf/2603.12578v1)

**作者:** Yi Xu `[一作]` (Alibaba Group), Xiaoyi Zeng `[通讯]` (Alibaba Group)

**通讯引用:** 669 | [OpenAlex ID](https://openalex.org/A5082008486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究CTR预测中用户行为序列与静态上下文特征的交互问题，提出CDNet双视角交互网络。

**💡 创新点**

创新点在于同时引入细粒度核心行为交互与粗粒度全局兴趣分布补偿，以兼顾信息细节与噪声抑制。

**🔧 技术方法**

采用基于相似度的核心行为筛选（配合STE实现可微分）、兴趣分布构造以及Transformer交互网络实现特征融合。

**📊 数据集**

使用公开的Taobao数据集和工业级别的大规模用户交互数据集进行验证。

**📈 对比分析**

与聚合+交互基线（DCN、AutoInt、Hiformer、Wukong、RankMixer）以及联合交互基线（InterFormer、OneTrans）对比，CDNet在Taobao上AUC提升0.58%、GAUC提升0.33%、LogLoss降低0.21%；在工业数据上AUC提升0.12%、GAUC提升0.26%、LogLoss降低0.83%。

**⚠️ 局限性**

限制在于核心行为比例与相似度区间需手动调参，模型对极长序列的鲁棒性仍受限；核心行为筛选过程对超参数敏感。

---

## 64. LLMs for Human Mobility: Opportunities, Challenges, and Future Directions

**arXiv ID:** 2603.12420 | [PDF](https://arxiv.org/pdf/2603.12420v1)

**作者:** Jie Gao `[一作]` (Delft University of Technology), Yaoxin Wu `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 930 | [OpenAlex ID](https://openalex.org/A5059325642)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大语言模型在五大人类移动任务（行程规划、轨迹生成、仿真、预测、语义理解）进行系统综述，梳理技术方法、数据集与挑战。

**💡 创新点**

提出任务导向的综述框架，阐明LLM在每个任务中的核心角色与潜在机会，并给出未来研究方向。

**🔧 技术方法**

综述提示工程、代理式LLM、微调、检索增强等技术，以及LLM与外部工具（检索、规划器、优化器等）的协同使用。

**📊 数据集**

使用的主要数据集包括TravelPlanner、ChinaTravel、TripTailor、GPS/Check‑in、公共交通卡刷卡记录等，本文并未自行训练模型。

**📈 对比分析**

通过与传统规则/优化、统计/机器学习基线在可行性、真实性、多样性、预测准确率等指标上的对比，显示LLM在语义推理与个性化上优于传统方法，但在可行性验证、实时性与隐私等方面表现不佳。

**⚠️ 局限性**

主要限制包括：LLM生成方案易出现可行性违规；缺乏动态交互与多用户竞争的评估；语义表示与时空结构耦合不足；评估指标碎片化且缺乏统一基准；计算成本高且隐私风险未得到充分评估。

---

## 65. Human-in-the-Loop LLM Grading for Handwritten Mathematics Assessments

**arXiv ID:** 2603.13083 | [PDF](https://arxiv.org/pdf/2603.13083v1)

**作者:** Arne Vanhoyweghen `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Harvard University)

**通讯引用:** 1235 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并实验证明了一个可扩展的LLM辅助评分工作流程，用于短手写数学测验。

**💡 创新点**

将多轮LLM评分、自动一致性检查与人工核查相结合，细化评分键以降低模型误差，并在真实课堂中实现约23%的评分时间节省。

**🔧 技术方法**

使用GPT‑5.1大语言模型、OCR+模板识别、批量扫描、答卷匿名化、自动一致性检测和人工复核。

**📊 数据集**

两门本科数学课的六次短测验（约30名学生、每人两题，共计约180份手写答卷）。

**📈 对比分析**

通过计时实验和Cohen's κ一致性比较，LLM辅助评分比人工手动评分快约23%，并且与人工评分的一致性相当或更高。

**⚠️ 局限性**

仅适用于结构化、低风险开放式题目；模型偶尔出现幻觉或误判；需要细致的评分键设计；仍需人工核查，无法完全自动化。

---

## 66. SAP: Segment Any 4K Panorama

**arXiv ID:** 2603.12759 | [PDF](https://arxiv.org/pdf/2603.12759v1)

**作者:** Lutao Jiang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了SAP模型，能够在4K全景图像上进行提示式实例分割；

**💡 创新点**

核心创新在于将全景图像转化为固定轨迹的视角视频，使得SAM2的时序记忆机制在全景环境中得到充分利用，并通过大规模合成数据进行微调；

**🔧 技术方法**

技术手段包括基于InfiniGen的183k张4K全景合成数据、视角转换与固定行进轨迹的视频化、SAM2的迁移学习与记忆机制调整；

**📊 数据集**

使用的数据集为183k张合成4K全景图（InfiniGen）以及零样本评测数据PAV‑SOD（真实4K全景）和HunyuanWorld‑1.0（8K卡通式合成全景）；

**📈 对比分析**

与SAM2及其扫描版对比，SAP在所有模型尺寸上平均提升约17.2 mIoU（在PAV‑SOD上小模型提升24.2 mIoU），在HunyuanWorld‑1.0上亦取得数个百分点增益；

**⚠️ 局限性**

局限性包括：仅处理单张全景图像；对合成数据的依赖，可能在极端真实场景或动态视频中表现不足；以及对内存与推理时延的进一步优化仍待研究。

---

## 67. Dynamic direct (ranked) access of MSO query evaluation over SLP-compressed strings

**arXiv ID:** 2603.13058 | [PDF](https://arxiv.org/pdf/2603.13058v1)

**作者:** Martín Muñoz `[一作]` `[通讯]` (University of Artois), Martín Muñoz (University of Artois)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出一种在字符串及其SLP压缩表示上，对MSO查询答案进行直接（按序）访问的算法，支持线性预处理、对答案的对数时间查询以及对字符串编辑的对数时间更新。

**💡 创新点**

改进了之前工作中对数平方访问时间，缩小到单个对数因子；首次将直接访问扩展到SLP压缩字符串；在此基础上实现了支持复杂编辑操作的动态访问结构。

**🔧 技术方法**

利用vset automata、矩阵表示、对数时间二分搜索、SLP的强平衡性质以及对数时间的更新子程序（通过复制路径实现），构建了紧凑的数据结构。

**📊 数据集**

本文为理论性工作，没有使用具体数据集；所有结论基于算法复杂度分析和抽象模型。

**📈 对比分析**

与之前的Bourhis等人ICDT 2025结果比较，本文在访问阶段多减少一个对数因子；整体预处理保持线性。性能上，预处理时间为O(|Q|^ω·|X|·|w|)（字符串）或O(|Q|^ω·|X|·||)（SLP），查询时间为O(|Q|^ω·|X|^2·log|w|)。

**⚠️ 局限性**

主要限制是更新操作要求SLP必须是强平衡的；对一般SLP无法保证对数时间更新；此外，缺乏实验验证，工作仅在理论层面完成。

---

## 68. Beyond Motion Imitation: Is Human Motion Data Alone Sufficient to Explain Gait Control and Biomechanics?

**arXiv ID:** 2603.12408 | [PDF](https://arxiv.org/pdf/2603.12408v1)

**作者:** Xinyi Liu `[一作]` (University of North Carolina), He Huang `[通讯]` (University of North Carolina)

**通讯引用:** 10427 | [OpenAlex ID](https://openalex.org/A5036072216)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在强化学习基础的运动模仿学习框架中加入足-地面交互信号（GRF 与 CoP）对步态运动学与动力学估计的影响，并通过消融实验验证其效果。

**💡 创新点**

提出了基于动力学感知的模仿学习（KAIL）框架，在传统运动仅模仿（MOIL）基础上加入 GRF 与 CoP 奖励，使得仿真产生的关节矩阵与逆动力学结果更一致，证明仅靠运动匹配不足以保证生物力学合理性。

**🔧 技术方法**

使用了 PPO 强化学习、MuJoCo 前向动力学模拟、逆动力学分析、残差力控制以及 GRF/CoP 约束奖励。

**📊 数据集**

利用一名受试者的 30 个步态循环的 3D 运动捕捉（Vicon）与跑步机力平台（Bertec）数据，时间归一化后作为专家演示与力学信号。

**📈 对比分析**

通过对比四种奖励配置（仅运动、+GRF、+CoP、+GRF+CoP），评估 GRF 相关系数（CPCC）、CoP RMSE、关节角度 RMSE、关节矩阵 RMSE 以及与逆动力学的相关性。结果显示加入 GRF/CoP 后 GRF 相关系数提升至约 0.9，CoP RMSE 降至 21%（下降 43%），关节矩阵 RMSE 降至 20–40% 以内，显著优于仅运动模仿。

**⚠️ 局限性**

研究仅在单一走路速度和水平地面下验证，接触模型过于简化；模型使用阻尼器控制与人类肌肉动力学不一致，导致关节矩阵高频噪声；未检验跨地形、速度或多受试者的泛化能力。

---

## 69. Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation

**arXiv ID:** 2603.12553 | [PDF](https://arxiv.org/pdf/2603.12553v1)

**作者:** Minghao Jin `[一作]` (University of Science and Technology of China), Xiaojun Chang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19502 | [OpenAlex ID](https://openalex.org/A5034967388)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于世界模型的视觉-语言-动作框架StructVLA，用结构化帧作为中间规划目标，实现了从语言指令到低级控制的端到端映射。

**💡 创新点**

创新点在于将生成式世界模型转化为物理上可解释的结构化规划器，利用抓取状态转换和运动转折点等运动学线索提取稀疏、可执行的视觉子目标，从而减少长期漂移并增强控制对齐。

**🔧 技术方法**

采用了自回归Transformer、Emu3和FAST离散词汇表进行视觉、语言与动作联合编码，并通过两阶段训练（先预测结构化帧，再生成动作令牌）实现了端到端学习。

**📊 数据集**

使用了BridgeData V2、LIBERO、多任务真实机器人轨迹以及从SimplerEnv、LIBERO与真实机器人收集的数千条演示数据进行训练与评估。

**📈 对比分析**

与VideoVLA、UniVLA等基线对比，StructVLA在SimplerEnv-WidowX上平均成功率75.0%，在LIBERO上94.8%，在真实机器人上的 pick‑and‑place 与长程整理任务中亦显著优于基线。

**⚠️ 局限性**

主要局限在实验规模和计算资源受限，尚未在更大、多域数据集上验证泛化能力，并需进一步改进模型稳定性与在线强化学习的集成。

---

## 70. CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration

**arXiv ID:** 2603.12721 | [PDF](https://arxiv.org/pdf/2603.12721v1)

**作者:** Dongxu Zhang `[一作]`, Jihua Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出CMHANet，一种跨模态混合注意力网络，实现从粗到细的点云配准流程；

**💡 创新点**

创新点在于融合3D几何与2D图像的跨模态混合注意力机制，并通过对比学习优化函数显著提升鲁棒性；

**🔧 技术方法**

使用KPConv-FPN、ResUNet-50编码器、三阶段混合注意力（自注意、聚合注意、交叉注意）、Sinkhorn归一化、对比损失及无RANSAC的估计方法；

**📊 数据集**

在3DMatch、3DLoMatch和TUM RGB‑D SLAM数据集上进行训练与评估；

**📈 对比分析**

与SOTA方法对比，CMHANet在Feature Matching Recall、Registration Recall、RRE、RTE等指标上均取得最优或接近最优，尤其在低重叠场景下提升约10%；

**⚠️ 局限性**

局限在极低重叠(<10%)或完全平面纹理缺失的环境中仍易失配，且跨模态融合导致推理时延与显存略高。

---

## 71. Altered Thoughts, Altered Actions: Probing Chain-of-Thought Vulnerabilities in VLA Robotic Manipulation

**arXiv ID:** 2603.12717 | [PDF](https://arxiv.org/pdf/2603.12717v1)

**作者:** Tuan Duong Trinh `[一作]` (University of Melbourne), Basim Azam `[通讯]` (University of Melbourne)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5002895693)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉-语言-动作（VLA）模型中研究了通过篡改链式推理（CoT）文本来攻击机器人执行任务的可行性。

**💡 创新点**

发现动作解码器对实体引用（对象名称）的依赖是关键漏洞，其它文本属性（顺序、空间词、噪声等）几乎无影响，并证明该漏洞特定于采用CoT的模型，且更依赖文本内容而非攻击者能力。

**🔧 技术方法**

设计并实现了七种文本腐败方式（噪声、句子打乱、实体替换、否定翻转、填充、LLM对抗等），通过双向注意力的DeepThinkVLA框架将腐败文本注入并评估。

**📊 数据集**

使用LIBERO桌面操作基准，涵盖四个子集（Object、Spatial、Goal、Long）共40个任务。

**📈 对比分析**

与干净CoT、非推理VLA（OpenVLA‑OFT）以及指令级攻击对照，结果显示实体替换导致整体SR下降8.3pp（Goal降19.3pp），其余攻击≤±4pp，LLM对抗效果更弱。

**⚠️ 局限性**

实验仅针对DeepThinkVLA单一模型，且仅在LIBERO上验证，缺乏对其他推理VLA或不同场景的推广性；LLM对抗仅在特定条件下有效，且研究未深入探讨更高级的防御策略。

---

## 72. PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization

**arXiv ID:** 2603.13228 | [PDF](https://arxiv.org/pdf/2603.13228v1)

**作者:** Yangsong Zhang `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Ivan Laptev `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 32285 | [OpenAlex ID](https://openalex.org/A5087781064)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PhysMoDPO，一种基于物理驱动的后训练框架，利用 Diffusion 模型生成满足文本指令且可被 Whole-Body Controller（WBC）执行的人类运动；

**💡 创新点**

创新点在于：①通过预训练的 WBC 自动生成偏好对（win/lose），将物理可执行性直接融入训练；②采用 DPO（Direct Preference Optimization）对生成器进行微调，并通过多轮迭代不断更新偏好数据；③在奖励设计上同时考虑跟踪误差、滑动、文本一致性等多维度，避免单一物理或任务奖励导致的偏差；

**🔧 技术方法**

核心技术包括：扩散模型（MotionStreamer、OmniControl 等）作为生成器；WBC/DeepMimic 作为物理执行器；DPO 作为偏好学习框架；多种物理与任务奖励（tracking、sliding、M2T、control）构成偏好评估；

**📊 数据集**

主要使用 HumanML3D（文本到运动）和 OMOMO（文本+对象交互）作为训练/测试数据集；同时在 SMPL 机器人模拟器与 Unitree G1 实际机器人上进行验证；

**📈 对比分析**

与 MaskedMimic、MotionStreamer、OmniControl 等基线对比，PhysMoDPO 在模拟和真实机器人上均显著降低 FID、Jerk，提升文本检索准确率（R@1/2/3）和空间控制误差，证明在物理可执行性与任务一致性上都有提升；

**⚠️ 局限性**

局限性包括：仅在平地行走环境中验证，无法直接处理复杂地形；偏好对的生成依赖固定的模拟追踪器，可能引入评价偏差；

---

## 73. Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning

**arXiv ID:** 2603.12832 | [PDF](https://arxiv.org/pdf/2603.12832v1)

**作者:** Fuhai Chen `[一作]` (Fuzhou University), Xuri Ge `[通讯]` (Shandong University)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5026473068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的UAV场景变更描述任务（UAV-SCC），并提出Hierarchical Dual-Change Collaborative Learning (HDC-CL)框架实现该任务。

**💡 创新点**

创新点包括：①动态自适应布局Transformer (DALT) 与投票式视角位移估计，实现对部分重叠场景的高效对齐；②分层交叉模态方向一致性校准 (HCM-OCC)，将视角变化的方向信息与文本语义对齐；③针对UAV变更描述构建的双版本（Simple与Rich）基准数据集。

**🔧 技术方法**

使用的技术包括ResNet-101特征提取、Transformer自注意力、投票位移估计、信息对比学习 (InfoNCE)、Hilbert-Schmidt独立性准则 (HSIC)、双向方向匹配损失、以及Transformer解码器生成自然语言。

**📊 数据集**

使用的公开数据集为GeoText-1652与UAVDT的图像对，经过重新标注构成UAV-SCC数据集，包括UAV-SCCSimple与UAV-SCCRich两种注释版本。

**📈 对比分析**

在UAV-SCC的BLEU‑4、METEOR、ROUGE‑L、CIDEr、SPICE等指标上，与现有变更描述方法（DUDA、SRDRL、SCORER+CBR、SMART、DIRL+CCR、CARD）对比，HDC-CL在CIDEr上分别获得54.68（Simple）和19.16（Rich），在其他指标上亦保持领先，提升幅度在3–6分左右。

**⚠️ 局限性**

局限性：①模型主要针对单幅UAV图像对，尚未验证跨航线或多视角序列的鲁棒性；②对极端视角位移或遮挡的处理仍受限；③数据集规模相对有限，可能导致模型在更大规模或不同环境下的泛化性不足。

---

## 74. Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing

**arXiv ID:** 2603.12788 | [PDF](https://arxiv.org/pdf/2603.12788v1)

**作者:** Shuchang Lyu `[一作]` (Beihang University), Zhenwei Shi `[通讯]` (Beihang University)

**通讯引用:** 14773 | [OpenAlex ID](https://openalex.org/A5058849690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多实体遥感推理定位（ME‑RSRG）数据集，并设计了基于视觉‑语言基础模型的实体感知推理（EAR）框架，利用两阶段训练（SFT+GRPO）实现多实体关系推理与定位。

**💡 创新点**

创新点：①构建首个包含实体角色与关系的遥感多实体推理定位数据集；②提出结构化推理输出与实体感知奖励机制，兼顾思考轨迹与实体定位；③采用Group Relative Policy Optimization（GRPO）结合实体感知奖励实现强化学习微调。

**🔧 技术方法**

技术：视觉‑语言基础模型（Qwen‑VL、InternVL等）+ 监督微调（SFT）+ 强化学习（GRPO）+ 结构化奖励（两级格式奖励、实体精度奖励、关系一致性奖励）。

**📊 数据集**

数据集：ME‑RSRG（7,162张图、12,091图文对），来源于RSVG‑HR、DIOR‑RSVG、OPT‑RSVG，包含主体与一个或多个客体的关系描述，并提供约20% CoT 训练样本。

**📈 对比分析**

对比：零样本下各视觉‑语言模型mAcc@0.5低于30%；SFT后提升约15–35%；再加GRPO后Qwen2.5‑VL系列mAcc@0.5提升超10%，InternVL3.5‑4B提升至32.8%。整体模型在主体/客体定位精度均显著提高。

**⚠️ 局限性**

局限性：①仍对复杂多体关系及小目标/相似结构的区分不够准确；②推理轨迹中出现固定模板化语言；③关系奖励稀疏导致多客体场景学习不稳定；④依赖大规模基础模型与昂贵计算资源。

---

## 75. Conflict Mitigation in Shared Environments using Flow-Aware Multi-Agent Path Finding

**arXiv ID:** 2603.12736 | [PDF](https://arxiv.org/pdf/2603.12736v1)

**作者:** Lukas Heuer `[一作]` (Örebro University), Martin Magnusson `[通讯]` (Örebro University)

**通讯引用:** 3653 | [OpenAlex ID](https://openalex.org/A5101576376)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Flow-Aware Multi-Agent Path Finding (FA-MAPF)，将环境中可观测的动态代理运动模式（Maps of Dynamics）嵌入到集中式MAPF规划中，以降低机器人与不可控代理的冲突；

**💡 创新点**

创新点在于利用学习得到的多模态运动模式（CLiFF-maps）通过引导图动态调整边权，从而在规划阶段即实现对不可控代理的冲突预测与规避；

**🔧 技术方法**

核心技术包括CLiFF-maps构建、半包裹高斯混合模型（SWGMM）对速度分布建模、使用马氏距离计算流成本、在ECBS/RHCR等先进MAPF算法中加入引导边权；

**📊 数据集**

使用标准MAPF基准地图（如Stern等8张地图）以及真实世界ATC商场的人类轨迹数据进行实验；

**📈 对比分析**

与不使用流成本的基线（传统ECBS/RHCR）对比，FA-MAPF在保持任务吞吐量不变的前提下，冲突率可降低最高55%，但计算时间略增；

**⚠️ 局限性**

局限性包括对地图结构的依赖（对房间型地图效果好，迷宫型差）；需要事先构建或在线更新MoDs；计算时间增加；并未针对时间变动态模式扩展。

---

## 76. CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning

**arXiv ID:** 2603.12543 | [PDF](https://arxiv.org/pdf/2603.12543v1)

**作者:** Carlos Purves `[一作]` (University of Cambridge), Pietro Lio' `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了CALF框架，使强化学习策略在分布式边缘与云服务器部署时能模拟网络延迟、抖动与丢包，从而实现网络感知训练与部署。

**💡 创新点**

将网络条件作为可配置的域随机化轴，引入NetworkShim中间件在RL循环中注入可复现的网络失真，并通过多模部署模式验证其对实测部署性能的显著提升。

**🔧 技术方法**

采用Python/Stable‑Baseline3的PPO算法，使用网络模拟器(NetworkShim)、容器化部署、时间戳加噪声的延迟采样、随机帧堆叠/ LSTM 等机制实现分布式策略图。

**📊 数据集**

在CartPole‑v1和MiniGrid DoorKey‑8x8等经典离散控制环境上进行训练与评估，网络配置基于测得的Ethernet、Wi‑Fi Normal/Degraded统计。

**📈 对比分析**

通过三种部署模式（本地、模拟网络、真实网络）对比基线、仅延迟、完整网络感知训练；结果显示网络感知训练将实测Wi‑Fi Degraded性能提升约4倍，显著降低40–80% 的性能下降。

**⚠️ 局限性**

仅在离散模拟环境与局域网条件下验证；未覆盖真实机器人、WAN/移动网络、复杂层级策略或多智能体交互，且未实现在线自适应或深层次的策略图训练。

---

## 77. Perpetual Dialogues: A Computational Analysis of Voice-Guitar Interaction in Carlos Paredes's Discography

**arXiv ID:** 2603.12854 | [PDF](https://arxiv.org/pdf/2603.12854v1)

**作者:** Gilberto Bernardes `[一作]` (University of Porto), António Sá Pinto `[通讯]` (University of Porto)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出并实现了一套计算框架，用于系统分析卡洛斯·帕雷德斯（Carlos Paredes）声乐合作录音中的声乐与葡萄牙吉他层交互关系，结合源分离、物理信息化和多尺度统计方法，对八段录音的旋律、和声及节奏特征进行量化与关联分析。

**💡 创新点**

创新点在于：①首次为无谱面传统音乐构建可同时捕捉结构与表演维度的综合分析流程；②将源分离、物理信息化吉他音高模型与多尺度相关/残差检测相结合，能够在几乎无需人工标注的前提下识别出音乐结构转折与表达共性。

**🔧 技术方法**

核心技术包括：Ht‑Demucs源分离、HPSS、CQT/谱心/色度提取、物理建模的吉他色度字典、节奏曲线与密度计算、经验模态分解、Pearson相关、Fisher z变换、鲁棒线性回归与残差异常检测、bootstrap置信区间校准。

**📊 数据集**

数据集为1983年前后卡洛斯·帕雷德斯与四位歌手（Camacho、Goes、Melo、Correia de Oliveira）合作的八段录音（共26轨），每段均包含原始未压缩音频和专家标注的节拍/句式/段落信息。

**📈 对比分析**

方法通过对每段的相关系数矩阵进行Fisher z聚合，并以0.6为阈值筛选显著相关对；随后利用残差标准化并设定±2.5为异常门限，检测结构变迁。实验表明，跨曲目无统一强相关模式，仅保留“声乐音高‑声乐响度”这一全局趋势；残差异常点与手工标注的正式边界高度一致，说明框架能够准确捕获音乐结构转折。

**⚠️ 局限性**

局限性包括：数据量仅八段录音，难以推广至更大规模或跨时间的语料；残差异常检测对阈值敏感，需进一步自动化；源分离时偶尔出现共振或谐波失真，影响后续特征；缺乏对即兴表现的细粒度评估，且目前特征主要聚焦能量与音高，未深入音色维度。

---

## 78. MemRoPE: Training-Free Infinite Video Generation via Evolving Memory Tokens

**arXiv ID:** 2603.12513 | [PDF](https://arxiv.org/pdf/2603.12513v1)

**作者:** Youngrae Kim `[一作]` (University of Southern California), Peter A. Beerel `[通讯]` (University of Southern California)

**通讯引用:** 4137 | [OpenAlex ID](https://openalex.org/A5084205024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种训练无关的MemRoPE框架，实现了无限长度视频生成。

**💡 创新点**

创新点在于同时引入了双EMA记忆令牌与在线RoPE索引，实现了无位置干扰的记忆聚合与位置信息自适应。

**🔧 技术方法**

采用自回归扩散模型、双EMA记忆、在线RoPE索引、三层KV缓存等技术。

**📊 数据集**

使用MovieGenBench文本提示、VBench-Long评价指标以及Gemini 3.1-Pro VLM评估。

**📈 对比分析**

与Self-Forcing、LongLive、Deep Forcing、∞-RoPE等训练无关方法对比，在多种长度（2分钟、4分钟、1小时）下，MemRoPE在视觉质量、主题一致性和整体偏好上均优于基线，且超越∞-RoPE。

**⚠️ 局限性**

局限在于仅提升记忆表现，单帧质量受基模型限制，EMA聚合可能导致远程内容细节丢失。

---

## 79. Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis

**arXiv ID:** 2603.12903 | [PDF](https://arxiv.org/pdf/2603.12903v1)

**作者:** Yinuo Jiang `[一作]` (Huazhong University of Science and Technology), Cheng Cheng `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 31224 | [OpenAlex ID](https://openalex.org/A5066830496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无姿态标定的 LiDAR 新视角合成框架 SG‑NLF，实现了高质量的 LiDAR 点云重建与视角合成

**💡 创新点**

创新点在于三方面：① 结合光谱嵌入与几何编码的混合表示，提升稀疏 LiDAR 数据下的连续表面重建；② 构建置信度感知图优化全局姿态，克服传统基于相邻帧对齐的局限；③ 采用对抗学习的跨帧一致性约束，提升重建与姿态的全局一致性

**🔧 技术方法**

使用的技术包括：多分辨率哈希编码、可微分光谱嵌入（LBO eigenfunctions）、Lie 代数姿态优化、互最近邻匹配的置信图、PatchGAN 对抗损失、体渲染的 NeRF 网络

**📊 数据集**

在 KITTI-360 与 nuScenes 两个自动驾驶数据集上进行实验，分别测试低采样率（2 Hz）与标准采样率（10 Hz/20 Hz）场景

**📈 对比分析**

与 GeoNLF、LiDAR4D、LiDAR‑NeRF 等最先进方法比较，SG‑NLF 在低频场景下 Chamfer Distance 下降 35.8%，绝对轨迹误差降低 68.8%，在标准频率下也保持较优的深度/强度 PSNR 与 ATE，整体性能优于所有对比方法

**⚠️ 局限性**

目前仅提供一种有效实现，缺乏对不同场景与参数的系统性探讨；未来可扩展更多模块与更广泛的应用

---

## 80. Upper Bounds for Local Learning Coefficients of Three-Layer Neural Networks

**arXiv ID:** 2603.12785 | [PDF](https://arxiv.org/pdf/2603.12785v1)

**作者:** Yuki Kurumadani `[一作]` `[通讯]` (Center for Mathematical Modeling and Data Science, University of Osaka), Yuki Kurumadani (Center for Mathematical Modeling and Data Science, University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了针对三层神经网络在奇异点处可计算的学习系数（真实对数典范阈值）的上界公式，且该公式适用于任意实解析激活函数；

**💡 创新点**

在奇异点给出学习系数上界，克服了以往只在非奇异点可用的上界；将学习系数的计算转化为满足预算与供需约束的计数问题；

**🔧 技术方法**

利用代数几何中的正规交叉、blow‑up 变换、坐标变换等工具，结合泰勒展开与随机变量线性独立性假设，推导出一般性上界；

**📊 数据集**

论文未使用任何实验数据集，全部为理论推导与证明；

**📈 对比分析**

与已知的三层网络学习系数结果（例如输入维度为1时的已解析值）进行比较，证明在该情形下上界与真值相符；在输入维度≥2时，上界仍保持可计算但可能不收敛到真值；

**⚠️ 局限性**

上界可能不够紧，且仅适用于三层网络；需要满足线性独立性与可逆Jacobian等技术条件，若不满足则无法直接应用；

---

## 81. On Using Machine Learning to Early Detect Catastrophic Failures in Marine Diesel Engines

**arXiv ID:** 2603.12733 | [PDF](https://arxiv.org/pdf/2603.12733v1)

**作者:** Francesco Maione `[一作]`, Guido Maione `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提供了一款新的 LaTeX 文档类 elsarticle.cls，专门用于格式化向 Elsevier 期刊提交的稿件，支持预印本和多种期刊最终版样式，兼容常用包并减少冲突。

**💡 创新点**

创新点在于完全基于标准的 article.cls 重写，移除了对旧版 elsart.cls 的重定义，提供了多种预设选项（preprint、review、1p、3p、5p 等），并且支持 natbib、geometry、graphicx、txfonts、hyperref、endfloat 等包的无缝集成，极大简化了作者的排版流程。

**🔧 技术方法**

采用 LaTeX 语言实现，内部利用 natbib 处理引用、geometry 设定页面边距、graphicx 处理图形、txfonts（可选）提供 Times 字体、hyperref 生成超链接、endfloat（可选）把所有浮动对象放到文档末尾，同时提供了自定义定理环境、增强列表、跨引用等宏包。

**📊 数据集**

无数据集；该文档属于工具类说明文档，旨在指导用户如何使用 elsarticle.cls 进行论文排版，而非实验数据驱动的研究。

**📈 对比分析**

比较方法主要是与旧版 elsart.cls 进行对比，强调在预印本、最终版转换、与其它宏包兼容性以及错误冲突方面的改进；并未给出数值性能指标，而是以功能兼容性和使用体验为评估标准。

**⚠️ 局限性**

局限性包括：对双栏期刊的长公式仍需作者手动断行以避免排版错误；某些可选功能（如 txfonts、hyperref）需要系统预装相应字体或包；此外，在极端自定义排版需求时仍可能需要额外手工调整。

---

## 82. Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group

**arXiv ID:** 2603.12459 | [PDF](https://arxiv.org/pdf/2603.12459v1)

**作者:** Alan Garbarz `[一作]` (Universidad de Buenos Aires), Alan Garbarz `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5083575093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本文提出一种直接构造可旋转核（steerable kernel）的通用方法，用基于稳定子子群的表示限制与同态映射实现，对任意紧李群以及非紧洛伦兹群的等变卷积网络提供显式的实数与复数基。

**💡 创新点**

创新点在于：①不需要预先计算Clebsch‑Gordan系数或变换到耦合基；②通过求解稳定子子群上的同态映射直接得到核空间；③给出了对SO(2)、O(2)、SO(3)、O(3)、SO⁺(1,3)等常见群的完整基；④方法对特征图的张量类型与维度无关，便于实现和调参。

**🔧 技术方法**

技术手段主要是表示论：利用Peter–Weyl定理把输入输出特征映射分解为不可约表示；采用Schur引理求解同态映射；通过在一个特定点x₀满足的简单不变条件来构造核，再用“steer”公式把它推广到整个群轨道；对非紧群则采用Lorentz群的复合表示与双覆盖SL(2,ℂ)的关系来实现。

**📊 数据集**

本文为理论研究，没有使用具体的数据集或实验评估；若有实验，只是对方法可行性的示例性说明。

**📈 对比分析**

与传统使用Clebsch‑Gordan系数或数值正交化的等变卷积网络相比，本文的方法在理论上更简洁、实现更直观、计算量更低；但作者未给出定量性能比较或实验结果。

**⚠️ 局限性**

局限性包括：①对非紧群的处理仍依赖于对复合表示的构造，可能不适用于所有非紧情形；②只考虑有限维非酉表示，无法覆盖非紧群的无穷维酉表示；③在质量为零的粒子情况中忽略了ISO(2)平移子群的作用，可能限制了对真空模式的完整描述；④对实际网络实现的数值稳定性和梯度传播的细节未作深入探讨。

---

## 83. GPU-Accelerated Genetic Programming for Symbolic Regression with Beagle Framework

**arXiv ID:** 2603.12292 | [PDF](https://arxiv.org/pdf/2603.12292v1)

**作者:** Nathan Haut `[一作]` (Michigan State University), Wolfgang Banzhaf `[通讯]` (Michigan State University)

**通讯引用:** 16116 | [OpenAlex ID](https://openalex.org/A5004837138)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估了Beagle——一种利用GPU加速的遗传程序框架，用于符号回归，并在Feynman基准上进行实验。

**💡 创新点**

创新点包括：1）使用C#+ILGPU在GPU上直接实现大规模（百万级）种群评估；2）自定义GCL语言降低内存碎片；3）采用基于相关性的fitness函数和NaN处理策略提升搜索效率；4）动态种群规模与Monte‑Carlo启发式选择实现高效扩展。

**🔧 技术方法**

技术手段：GPU并行计算（CUDA via ILGPU）、C#高性能编程、线性遗传程序语言（GCL）、相关性fitness函数、NaN/无效值处理、批量评估与多GPU分配、并行CPU选择/变异。

**📊 数据集**

使用Feynman Symbolic Regression Benchmark（100个物理方程，512训练点，128测试点）进行评测。

**📈 对比分析**

在10分钟和30分钟两种时间约束下，使用10次独立跑，比较典型（>=50%成功）和最佳（至少一次成功）两指标。结果显示：Beagle (corr) > Beagle (pt‑pt) > StackGP > PySR；相较CPU系统，Beagle在典型和最佳案例中提升约15-20%的成功率。

**⚠️ 局限性**

限制：仅支持符号回归；缺乏交叉算子；对GPU硬件依赖强，若无GPU效果有限；评测仅针对Feynman数据集，其他任务的可迁移性未验证。

---

## 84. Boosting Spectral Efficiency via Spatial Path Index Modulation in RIS-Aided mMIMO

**arXiv ID:** 2603.12619 | [PDF](https://arxiv.org/pdf/2603.12619v1)

**作者:** Ahmet M. Elbir `[一作]` (Istinye University), Ahmed M. Eltawil `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 6222 | [OpenAlex ID](https://openalex.org/A5013660772)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种在 RIS 辅助的 mmWave mMIMO 系统中通过空间路径索引调制（SPIM）实现的混合波束成形框架，利用 RIS 提升空间多样性并通过索引位增加信息位，提升频谱效率。

**💡 创新点**

创新点在于首次将 SPIM 与 RIS 结合，设计低复杂度的混合波束形成方法，并对单用户与多用户场景分别推导 SE 表达式与优化问题，证明 SPIM 在多径环境下可超过全数字波束成形。

**🔧 技术方法**

核心技术包括基于 Saleh‑Valenzuela 级联信道模型的稀疏信道估计（OMP/CoSaMP）、基于 Steering 矢量的模拟波束选取、离散相位控制的 RIS 相位设计（半正定松弛与高斯随机化）以及 SPIM 的信息理论分析。

**📊 数据集**

实验使用随机生成的 mmWave 通道（路径角度均匀分布在[-90°,90°]，路径增益服从 N(1,0.2²)），在 N=128、N̅=16 的天线设置下进行 Monte‑Carlo 仿真。

**📈 对比分析**

与传统混合波束成形和全数字波束成形进行对比，仿真表明在 L≥4 的多径环境下，SPIM-混合波束成形可实现约 20% 的频谱效率提升，且在单 RF 链情况下仍优于全数字方案。

**⚠️ 局限性**

主要局限包括：对稀疏信道估计和相位切换速度的依赖；在路径增益差异较大时 SPIM 效果退化；以及对多用户干扰的处理需进一步优化，现有方案在多用户场景下仍受 IUI 限制。

---

## 85. A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning

**arXiv ID:** 2603.12421 | [PDF](https://arxiv.org/pdf/2603.12421v1)

**作者:** Hongyan Wei `[一作]` (Clemson University), Wael AbdAlmageed `[通讯]` (Clemson University)

**通讯引用:** 3482 | [OpenAlex ID](https://openalex.org/A5028776484)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种神经符号轨迹规划框架，将大规模语言模型提取的交通规则与ASP求解器演绎推理嵌入端到端网络，利用可微Kinematic Bicycle Model与神经残差生成既符合物理约束又可追溯的行驶轨迹。

**💡 创新点**

创新点在于（1）将演绎推理嵌入端到端网络，实现每帧可追溯的逻辑决策；（2）采用可微物理模型与神经残差双路径，兼顾物理可行性与学习灵活性；（3）构建LLM+ASP双层决策编码与分层优先级仲裁，实现逻辑与物理的动态条件化。

**🔧 技术方法**

技术包括：大规模语言模型（LLM）动态抽取规则、ASP求解器（Clingo）逻辑仲裁、可微Kinematic Bicycle Model、双路径决策嵌入、残差预测与损失、异步双速（低频LLM+高频ASP）等。

**📊 数据集**

使用nuScenes数据集，摄像头输入，ResNet-50骨干网络。

**📈 对比分析**

与UniAD、VAD、SparseDrive、MomAD等基线对比，L₂误差从0.60降至0.57 m，碰撞率从0.09%降至0.075%，TPC从0.54降至0.47 m，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括LLM推理耗时导致低频更新可能产生逻辑延迟、仅在开放式nuScenes评估（缺乏闭环仿真验证）以及对长时序与极端稀有场景的验证不足。

---

## 86. Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning

**arXiv ID:** 2603.12787 | [PDF](https://arxiv.org/pdf/2603.12787v1)

**作者:** Mengya Xu `[一作]`, Qi Dou `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 27936 | [OpenAlex ID](https://openalex.org/A5090516040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了最大的基本手术动作（BSA）数据集（BSA‑10），并基于该数据集训练了一种Transformer‑基础模型，实现了跨专业、跨器官的BSA识别、手术技术评估与多模态大型语言模型（GPT‑4o）辅助的手术动作规划。

**💡 创新点**

创新点：①首次系统化定义并量化10类基本手术动作；②通过四步数据集生成管道将公开数据高效复用并扩充；③提出双头Transformer与不确定性损失的BSA识别框架，解决类别不平衡与泛化难题；④将BSA识别与GPT‑4o结合，形成基于视觉历史和安全协议的实时动作规划框架；⑤通过多国外科医生评估验证模型可解释性与临床相关性。

**🔧 技术方法**

使用的技术：Transformer‑based视频分类网络（ViT‑B/16 + 时空注意力），不平衡补偿双头结构与Evidential loss；多模态LLM推理（GPT‑4o）与Prompt Engineering；交叉验证、AUROC、Youden Index、Top‑N准确率等评估指标；人类评估框架（多维度评分）。

**📊 数据集**

使用的数据集：①BSA‑10（11,915剪辑，10类动作，6种手术；来源于15公开数据集与SurgYT）；②外部验证集（肺叶切除、肝切除等），①8,064剪辑；②公开数据集CholecT50、Cholec80、M2cai16‑workflow等；③SAR‑RARP50（机器人辅助前列腺切除），用于技能评估。

**📈 对比分析**

比较方法：十折交叉验证、外部数据集测试、与基准模型对比（未列举但在图表中），指标包括AUROC、灵敏度/特异度、Top‑1/Top‑3准确率、Relaxed accuracy。性能：BSA识别所有动作AUROC>0.9，平均灵敏度≈92%，特异度≈95%；外部数据集肺叶切除AUROC≈88%，肝切除≈91%；在前列腺切除手术技术评估中，模型能区分三等级技术；在C‑CVS和N‑RCS规划任务中，Top‑3 strict global accuracy分别达到71%和49%，Relaxed accuracy接近90%及100%。

**⚠️ 局限性**

局限：①动作集仅限预定义的10类，无法覆盖新颖或细微动作；②模型未在实时机器人系统中验证；③缺乏持续学习机制，无法适应手术技术演进；④外部数据集仍有限，跨国差异与不同设备的泛化仍待进一步验证；⑤LLM推理依赖高成本商业模型，推断时延与资源需求未评估。

---

## 87. FGTR: Fine-Grained Multi-Table Retrieval via Hierarchical LLM Reasoning

**arXiv ID:** 2603.12702 | [PDF](https://arxiv.org/pdf/2603.12702v1)

**作者:** Chaojie Sun `[一作]` (Zhejiang University of Technology), Qing Fan `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FGTR，采用层级化的大模型推理实现细粒度多表检索，即先检索相关模式（schema）再检索对应单元格，构建最小化、足够回答问题的子表。

**💡 创新点**

创新点在于：① 模拟人类阅读流程的两阶段推理；② 在检索时显式填充关联键（主键/外键）以覆盖隐式联结；③ 通过投票与自适应最近邻映射提升召回与精确度；④ 构建专门针对多表的检索范式。

**🔧 技术方法**

技术手段包括：大语言模型（GPT‑4o、Llama3‑8B）与 Prompt 设计；离线预处理（模式语义化、联结路径预估、列向量索引 HNSW）；查询解析与模式映射的多轮推理；范围解析与自适应单元格映射；投票机制与频率阈值筛选。

**📊 数据集**

使用新构建的 SpiderQA 与 BirdQA（基于 Spider 与 BIRD）作为检索评测基准；在下游任务中还使用 WikiTableQuestions、TabFACT。

**📈 对比分析**

与 TableRAG、Dater、CHESS、RSL‑SQL 等 SOTA 方法比较，FGTR 在 SpiderQA 上 F₂ 提升 18%、在 BirdQA 上 21%；在下游 QA、事实验证任务中 EM/准确率均显著高于基线，且对大表保持稳定性。

**⚠️ 局限性**

局限性：仍依赖大模型与昂贵算力；对极大表（数十万行）仍可能遇到上下文窗口限制；联结路径预估依赖语义相似度，可能遗漏未知键；需要在新领域进行迁移学习或重新预处理。

---

## 88. InterDeepResearch: Enabling Human-Agent Collaborative Information Seeking through Interactive Deep Research

**arXiv ID:** 2603.12608 | [PDF](https://arxiv.org/pdf/2603.12608v1)

**作者:** Bo Pan `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 67607 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了InterDeepResearch，一种交互式深度研究系统，支持人机协作式信息检索；

**💡 创新点**

创新点在于构建层级化研究上下文管理框架，提供三视图可视化、交叉链接与追溯机制，让用户能实时监控、引导和追踪 LLM 代理的研究过程；

**🔧 技术方法**

技术包括基于 React 的前端实现、Reactflow/ELKJS/ D3 进行可视化，后端使用 Python+LiteLLM 调用 Claude Sonnet 4.5；

**📊 数据集**

使用公开的深度研究基准 Xbench-DeepSearch-v1 与 Seal-0 进行无用户干预的自动评测；

**📈 对比分析**

与现有顶尖系统（如 Perplexity Deep Research、Gemini Deep Research）对比，InterDeepResearch 在这两个基准上获得竞争甚至优于商业系统的得分，并在用户研究中获得高效协作与满意度；

**⚠️ 局限性**

局限包括：不同用户对交互模式偏好差异，系统对追溯等待时间仍较长，缺乏对多代理、多用户、多分支研究的支持，且仍需进一步优化可适配性与多模态交互。

---

## 89. Bin~Wan,G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images

**arXiv ID:** 2603.12680 | [PDF](https://arxiv.org/pdf/2603.12680v1)

**作者:** Bin Wan `[一作]` (Shandong University), Sam Kwong `[通讯]` (Lingnan University)

**通讯引用:** 34701 | [OpenAlex ID](https://openalex.org/A5008386708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 GeoGran-Aware Hierarchical Feature Fusion Network (G^2HFNet)，用于光学遥感图像中的显著目标检测，结合多尺度细节增强、双分支几何-细粒度互补、深层语义感知与局部-全局引导融合四大模块实现精细特征提取与融合。

**💡 创新点**

创新点在于：1）MDE 模块采用 U‑Net 结构与金字塔空间/通道注意力，显著提升多尺度细节捕捉；2）DGC 模块通过几何分支与细粒度分支的双分支设计以及交互块实现细节与位置信息互补；3）DSP 模块仅用自注意力对高层语义进行位置增强；4）LGF 模块取代传统卷积，以局部门控卷积与全局引导实现多级特征的高效融合。

**🔧 技术方法**

使用 Swin Transformer 作为骨干网络；结合 U‑Net、金字塔注意力、自注意力、门控卷积、局部-全局融合等技术；损失函数采用 BCE+IoU+F‑measure 三项混合损失。

**📊 数据集**

在三大公开遥感 SOD 数据集上进行实验：ORSSD（800 张），EORSSD（2000 张），ORSI‑4199（4199 张）并进行训练/测试划分。

**📈 对比分析**

与 18 种现有自然/遥感 SOD 方法（如 PoolNet、CSNet、PA‑KRN、MCCNet、VST 等）在 M、Fβ、Eξ 三指标上对比，G^2HFNet 在所有数据集上均取得最佳或最接近最佳结果，M 指标降幅达 19.6% 以上，Fβ 与 Eξ 分别提升 12.1% 与 5.6%。同时保持中等计算量（≈94 GFLOPs，95M 参数），推理速度约 18 FPS。

**⚠️ 局限性**

局限性：模型整体仍相对较大，MDE 及 DGC 模块占用显著 FLOPs，可能影响边缘硬件部署；在极低对比度或极小目标场景下，性能仍有提升空间；未在多模态遥感（SAR、热红外）上验证，需要进一步扩展。

---

## 90. MXNorm: Reusing MXFP block scales for efficient tensor normalisation

**arXiv ID:** 2603.13180 | [PDF](https://arxiv.org/pdf/2603.13180v1)

**作者:** Callum McLean `[一作]` (Graphcore), Carlo Luschi `[通讯]` (Graphcore)

**通讯引用:** 674 | [OpenAlex ID](https://openalex.org/A5015625929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MXNorm，一种将 RMSNorm 与 MXFP 量化融合的归一化方法，以实现更高效的训练与推理。

**💡 创新点**

利用 MXFP 量化过程中已计算的块绝对最大值近似 RMS，实现一次统计即可完成归一化与量化的融合，并在 Pre‑Norm Transformer 中直接替代 RMSNorm。

**🔧 技术方法**

基于块量化（MXFP8/4）与通用幂均值近似、Straight‑Through 估计、TorchAO 与 TorchTitan 训练框架。

**📊 数据集**

在 SlimPajama 语料上对 Llama‑3 系列模型进行预训练，并在 OLMES 基准上评估零样本性能。

**📈 对比分析**

与传统 RMSNorm+MXCast 进行学习率敏感性、损失收敛、零样本准确率对比，MXNorm(p=2) 在 8B 模型上与 RMSNorm 接近，推理时可获得 31–41% 的前向速度提升。

**⚠️ 局限性**

对极值的估计误差导致 p=1 方案在大模型下易出现失效；且目前仅在 MXFP 量化场景验证，未覆盖更窄格式或非线性层。

---

## 91. Human-AI Collaborative Autonomous Experimentation With Proxy Modeling for Comparative Observation

**arXiv ID:** 2603.12618 | [PDF](https://arxiv.org/pdf/2603.12618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Stake the Points: Structure-Faithful Instance Unlearning

**arXiv ID:** 2603.12915 | [PDF](https://arxiv.org/pdf/2603.12915v1)

**作者:** Kiseong Hong `[一作]` (Chung Ang University), Eunwoo Kim `[通讯]` (Chung Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种结构保持的机器去学习框架（Structguard），通过引入语义锚点（anchors）并在去学习过程中保持锚点与保留样本的语义关系，从而在删除指定样本的同时保持模型知识的结构和性能。

**💡 创新点**

创新点在于：①首次将基于语言模型生成的属性描述编码成语义锚点，用作结构保持的参考；②设计了结构感知对齐（alignment）和结构感知正则化（regularization）两个约束，分别保持结构分布的一致性并限制关键参数的更新；③在无保留数据的实际去学习场景下实现高效、稳定的去学习。

**🔧 技术方法**

技术手段包括：利用 GPT‑4o 生成类别属性描述，使用 CLIP 等语义编码器将描述映射为锚点；构建投影层将特征与语义空间对齐；对齐损失采用余弦相似度；正则化采用参数重要性权重的平方惩罚；训练使用交叉熵（保留）和负交叉熵（删除）损失的组合。

**📊 数据集**

实验数据集：图像分类（CIFAR‑10、CIFAR‑100、ImageNet‑1K）、人脸识别（Lacuna‑10）以及图像‑图像检索（CIFAR‑10）。

**📈 对比分析**

与 L2UL、Adv、Neggrad、Fisher、Rawp 等近似去学习方法对比，Structguard 在保留集和测试集上平均提升 15–25%（CIFAR‑10）、4–10%（CIFAR‑100）和 21–26%（ImageNet‑1K）的准确率，同时在删除集上保持 100% 的去学习效果；在检索和人脸识别任务中亦显著提高保留性能并保证忘记样本完全不被检索到。

**⚠️ 局限性**

局限性包括：依赖于可解释的属性描述与语义编码器，对非视觉或属性难以定义的任务适用性有限；在极大规模模型或极大忘记集合时的计算开销仍需进一步优化；实验主要聚焦于图像任务，跨模态或文本去学习的效果尚未验证。

---

## 93. Before and After ChatGPT: Revisiting AI-Based Dialogue Systems for Emotional Support

**arXiv ID:** 2603.13043 | [PDF](https://arxiv.org/pdf/2603.13043v1)

**作者:** Daeun Lee `[一作]` (Yale University), Jinyoung Han `[通讯]` (Sungkyunkwan University)

**通讯引用:** 25950 | [OpenAlex ID](https://openalex.org/A5040976081)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评述了 2020–2024 年期间 AI 对话系统在心理健康领域的发展，聚焦从任务专用深度学习模型向大型语言模型（LLM）过渡的技术演进。

**💡 创新点**

创新点：① 采用定量文献计量与定性趋势评估相结合，提供了技术转型的宏观与微观双重视角；② 明确描绘了 LLM 时代出现的灵活性提升与安全性挑战；③ 归纳了数据集、方法和评估指标的演进与不足，为后续研究指明了路径。

**🔧 技术方法**

主要技术手段包括：文献检索与筛选、文献计量分析（来源、国家、机构、关键词网络）、定性趋势回顾、对比预 LLm 与后 LLM 方法（深度学习、强化学习、多任务、检索增强生成、提示工程）。

**📊 数据集**

研究数据来源为 146 篇符合条件的论文，涵盖 Web of Science、Scopus 与 ACM Digital Library；在具体案例中对比使用的典型数据集，如 Empathetic Dialogue、ESConv、PsyQA、以及后续 LLM 生成的扩增数据集。

**📈 对比分析**

比较方法：先对 146 篇文献做年分布、来源与关键词网络分析；随后对比 30 篇代表性论文的技术与评估，发现 pre‑LLM 侧重情感识别与外部知识集成，表现出更可控的对话质量；post‑LLM 则展示更强的语言灵活性与跨领域适配，但在可靠性、误导信息（hallucination）与安全性评估上存在显著缺口。

**⚠️ 局限性**

局限性：① 只覆盖 2020‑2024 年的文字对话系统，未涉及多模态或更近时间的研究；② 只分析 30 篇代表性论文进行定性评估，未覆盖全部 146 篇；③ 对实际临床有效性与伦理合规的评估依赖于已有的评测指标，缺乏统一标准；④ 对数据集的隐私与真实性挑战未能系统解决。

---

## 94. Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass

**arXiv ID:** 2603.12789 | [PDF](https://arxiv.org/pdf/2603.12789v1)

**作者:** Sangmin Kim `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**通讯引用:** 9404 | [OpenAlex ID](https://openalex.org/A5100611457)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一个统一的单通道框架，能够同时从多视角多人物视频中重建相机参数、场景点云和人类SMPL模型；

**💡 创新点**

不依赖外部模块或预处理，采用双编码器（Pi3X + Multi‑HMR）融合几何与人类先验，使用头-骨盆长度比实现尺度对齐，并提出基于几何的多视角融合与多人物关联方法；

**🔧 技术方法**

Pi3X 3D 基础模型、Multi‑HMR 变换器、SMPL‑X 参数回归、尺度调整模块、视角无关/视角相关特征分离、多视角三角化与几何匹配；

**📊 数据集**

EMDB‑2、RICH、EgoHumans、EgoExo4D、EgoBody、Ego‑Humans、EgoExo4D、3DPW、MPII、MSCOCO（合成训练集）;

**📈 对比分析**

与 JOSH3R、UniSH、Human3R、HSfM、HAMSt3R 等基准模型比较，在全球运动估计和多视角姿态估计任务上取得了最优或最接近最优的 WA‑MPJPE、W‑MPJPE、RTE 等指标，同时在单帧推理时间上比现有优化式方法快 8 倍以上；

**⚠️ 局限性**

对头部可见性高度依赖，若头部被遮挡或不可见会导致性能下降；

---

## 95. Seeing Eye to Eye: Enabling Cognitive Alignment Through Shared First-Person Perspective in Human-AI Collaboration

**arXiv ID:** 2603.12701 | [PDF](https://arxiv.org/pdf/2603.12701v1)

**作者:** Zhuyu Teng `[一作]` (Zhejiang University), Lingyun Sun `[通讯]` (Zhejiang University)

**通讯引用:** 13734 | [OpenAlex ID](https://openalex.org/A5100629346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Eye2Eye框架，通过共享第一人称视角实现人机认知对齐，并在AR原型中实现；

**💡 创新点**

创新点在于将第一人称视角转化为双向共享感知通道，结合联合注意、累积共同基础和反思性情境反馈三大组件，实现持续的认知同步；

**🔧 技术方法**

采用眼动、手物交互与语音多模态感知，使用YOLO进行目标检测，Gemini 2.5 Flash与GPT‑4o作为视觉语言模型进行语义推理与反馈生成；

**📊 数据集**

使用自定义实验任务数据集：咖啡机操作、图书分类与电路板故障排查，共计约60名受试者完成三种任务；

**📈 对比分析**

与无Eye2Eye的基线（单向视觉+LLM）比较，Eye2Eye显著降低错误率（≈58%）、澄清成本（≈50%）并提升用户信任与工作负荷；

**⚠️ 局限性**

局限包括4‑5秒的推理延迟、基于规则的触发机制缺乏灵活性、实验规模有限且仅在实验室环境中验证，未来需更大规模、实时评估与轻量级边缘模型。

---

## 96. VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model

**arXiv ID:** 2603.12655 | [PDF](https://arxiv.org/pdf/2603.12655v1)

**作者:** Xiangyu Sun `[一作]` (University of Queensland), Yadan Luo `[通讯]` (University of Queensland)

**通讯引用:** 4073 | [OpenAlex ID](https://openalex.org/A5039237976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 VGGT-World，一种利用冻结的 3D 基础模型（VGGT）特征作为世界状态，在高维特征空间中通过自回归流变换器预测未来几帧的几何信息，从而实现无视频生成的三维场景前瞻。

**💡 创新点**

创新点包括：①将冻结的几何特征视为可预测的世界状态，避免传统视频 VAE 生成带来的几何不一致；②在高维特征空间采用 z‑prediction 直接预测干净目标而非速度；③提出双阶段流强制（latent flow‑forcing）课程策略，有效缓解自回归曝光偏差；④使用分块自回归和双流 Transformer 结构，兼顾时序推理与空间去噪。

**🔧 技术方法**

核心技术包括：连续时间流匹配（flow matching）与 z‑prediction 方案；双流 Transformer（双流+单流）实现时序与空间分离；两阶段流强制课程（教师强制 + 轨迹一致流强制）；VGGT 的编码/解码冻结；以及利用已有的 VGGT 3D 头进行深度、点云等几何输出。

**📊 数据集**

实验数据集：KITTI、Cityscapes、TartanAir。使用多帧序列分别进行短期和中期深度预测、点云预测和相机参数预测。

**📈 对比分析**

与 Cosmos、Gen3R、DINO‑Foresight、Aether、WVD 等主流视频生成或隐空间模型比较，VGGT-World 在深度预测中 AbsRel 和 δ1 均显著优于对手（短期/中期分别提升 21%/32%），点云指标 Accuracy/Completeness/Chamfer 距离均优；同时推理速度比 Cosmos（12B）快 5 倍、比 Gen3R 快 3.6 倍，且仅训练 0.43B 可学习参数，显著降低资源消耗。

**⚠️ 局限性**

限制：① 高维特征空间仍带来训练不稳定性，需特殊的 z‑prediction 与流强制策略；② 未来视野的扩展对几何估计有非单调影响，最佳滚动长度受限；③ 目前未加入动作或相机姿态的显式条件，缺乏可控性；④ 对不同场景或极端动态场景的泛化尚未充分验证。

---

## 97. Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents

**arXiv ID:** 2603.12634 | [PDF](https://arxiv.org/pdf/2603.12634v1)

**作者:** Yushu Li `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (University of British Columbia)

**通讯引用:** 5158 | [OpenAlex ID](https://openalex.org/A5100458648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Budget‑Aware Value Tree (BAVT)，一种无训练、推理时的树搜索框架，通过步骤级价值评估与预算感知节点选择，提升受限预算下的多跳推理性能。

**💡 创新点**

整合步骤级价值评估与预算感知的指数采样机制；通过树结构实现多路径并行探索；提供有限预算下的收敛性证明；无需额外训练，能直接应用于任意LLM。

**🔧 技术方法**

LLM多角色（生成器+评估器）Prompting、树搜索、功率缩放的预算感知采样、残差价值预测、全局回传、阈值驱动的终止策略。

**📊 数据集**

HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle四个多跳推理问答基准。

**📈 对比分析**

与并行采样基线（多路径+多数投票）在相同token/工具预算下比较；BAVT在低预算下达到或超过高预算基线，整体准确率提升约10–15%，在OSS‑20B和Qwen3‑30B上均表现优异。

**⚠️ 局限性**

评估者推理开销较大；仅考虑单一工具且成本统一，未涵盖异构工具与多维预算；不适用于长期交互任务，需进一步研究轻量级价值模型和异构工具预算管理。

---

## 98. A common parallel framework for LLP combinatorial problems

**arXiv ID:** 2603.13147 | [PDF](https://arxiv.org/pdf/2603.13147v1)

**作者:** David Ribeiro Alves `[一作]` (University of Texas), Vijay K. Garg `[通讯]` (University of Texas)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通用的无锁运行时 LLP‑FW，用于求解任何可表示为 Lattice‑Linear Predicate 的组合优化问题，核心思路是并行推进所有被判定为“禁止”状态的局部解。

**💡 创新点**

创新点在于把并行调度、原子操作和状态推进抽象成通用框架，仅需实现禁止性检查和推进函数即可；实现了问题与求解器的完全解耦，支持多种求解策略可在同一实现上切换。

**🔧 技术方法**

技术包括：共享内存无锁工作列表（Injector/Worker、BucketWorklist 等），原子 CAS 操作保证状态单调性；多种调度器（全索引扫描、共享/本地工作袋、分桶）以适应不同前沿宽度；固定状态位向量用于记忆已收敛状态；并通过优先级/递归偏向等启发式提升效率。

**📊 数据集**

使用的基准数据集包括：SSSP、BFS、Transitive Closure、Reduction、Knapsack、Job Scheduling 以及 Stable Marriage，覆盖稠密与稀疏图、权重与无权图、图形与匹配、动态规划与调度等多种应用；图数据来源包括路网、随机图、Citation、DAG、Mesh 等。

**📈 对比分析**

与手写的专用求解器相比，在大多数情况下表现优异：Stable Marriage 上最高可达 246× 加速，Transitive Closure 23×，道路网络 BFS 16×；在稠密或前沿宽广的实例上（如密集社群图、Knapsack）相对不如基准，速度仅提升 1.8–3.8×。单线程时因 CAS 及谓词检查导致 5–10× 的开销；多线程后随线程数增大交叉点降低，性能随工作量分布与前沿形态显著变化。

**⚠️ 局限性**

局限性包括：对前沿宽广或高度并发的情况会产生 CAS 争用和工作列表膨胀，导致无锁开销占主导；在普通的、带宽受限的任务（如 Reduction）中性能下降 4–5×；需要手动实现禁止性检查和推进逻辑，若实现不当可能破坏单调性；对某些问题的谓词检查成本（如 Stable Marriage 的 O(n) 全表扫描）在规模较大时成为瓶颈。

---

## 99. Leveraging Head Movement for Navigating Off-Screen Content on Large Curved Displays

**arXiv ID:** 2603.12620 | [PDF](https://arxiv.org/pdf/2603.12620v1)

**作者:** A K M Amanat Ullah `[一作]` (University of British Columbia), Khalad Hasan `[通讯]` (University of British Columbia)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5090272179)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究探索将用户的头部转动映射为大型曲面显示器工作区的平移，以便快速获取屏幕外的360°内容。

**💡 创新点**

创新点在于系统性比较七种头部映射函数（线性、Sigmoid、Polynomial、连续、摩擦、增量、截断）并发现多项式速率控制最优，同时将该头部导航技术与工业标准的拖拽/摇杆对齐方法在真实地图任务中对比，证明其在速度、效率与舒适度上的优势。

**🔧 技术方法**

采用OptiTrack 运动捕捉系统获取头部姿态，使用Unity 3D 进行实验实现，并通过 NASA‑TLX 及 VRSQ 问卷评估工作负荷与虚拟现实不适。

**📊 数据集**

实验数据主要来自两组：①一维离屏目标任务中随机设置的目标位置；②二维地图导航任务中构造的两种国家簇（Cluster A 与 Cluster B）及其三种旋转配置，共六套地图。

**📈 对比分析**

通过双因素重复测量 ANOVA、Bonferroni 校正以及 Friedman/Wilcoxon 检验，比较不同映射函数与控制器方法在任务完成时间、头部转动角度、误操作次数、主观负荷与 VR 症状等指标上的表现。实验显示，Polynomial‑Head 映射在所有关键指标上均优于连续/摩擦/增量/截断以及传统拖拽/摇杆技术，且被 78% 参与者评为首选。

**⚠️ 局限性**

局限性包括样本性别分布失衡、仅在 180°视角、半径 3.27 m 的曲面上验证、缺乏长时间洗脱导致 VR 不适评分可能混杂、仅测试水平转动（无俯仰/滚转）以及未扩展到二维/三维导航和多人协作场景。

---

## 100. Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**arXiv ID:** 2603.12547 | [PDF](https://arxiv.org/pdf/2603.12547v1)

**作者:** Fares Bougourzi `[一作]` (Universite Polytechnique Hauts-de-France), Abdenour Hadid `[通讯]` (Sorbonne University Abu Dhabi)

**通讯引用:** 19582 | [OpenAlex ID](https://openalex.org/A5013928164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种解码器中心的 Mamba 结构——Deco‑Mamba，用于 2D 医学图像分割。

**💡 创新点**

创新点包括：Co‑Attention Gate 融合编码器与解码器特征并捕获长距离依赖；Vision State Space Module（VSSM）在线性时间内建模全局上下文；变形残差块细化边界；以及窗口分布 KL 损失的多尺度分布感知深度监督。

**🔧 技术方法**

技术实现上采用 CNN–Transformer 混合编码器（PVT‑V2），Mamba 的 VSSM 模块，Co‑Attention Gate，变形卷积细化层，和窗口分布 KL 损失进行训练。

**📊 数据集**

在七个公开数据集上验证：Synapse、BTCV、ACDC、ISIC17、ISIC18、GlaS、MoNuSeg。

**📈 对比分析**

与现有 CNN、Transformer 以及 Mamba 方法对比，Deco‑Mamba 在多数数据集上取得 Dice/HD95 的 SOTA 结果，同时保持参数量和 FLOPs 较低的计算成本。

**⚠️ 局限性**

局限性：目前仅针对 2D 分割，3D 序列或跨模态迁移的适应性尚待验证；对极端分辨率或高噪声环境的鲁棒性仍需进一步提升。

---

## 101. SteerRM: Debiasing Reward Models via Sparse Autoencoders

**arXiv ID:** 2603.12795 | [PDF](https://arxiv.org/pdf/2603.12795v1)

**作者:** Mengyuan Sun `[一作]` (National Engineering Research Center for Software Engineering Peking University), Wei Ye `[通讯]` (National Engineering Research Center for Software Engineering Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 SteerRM 的无训练奖励模型去偏方法，利用稀疏自编码器（SAE）在推理时抑制对 Markdown 等格式化的偏好。

**💡 创新点**

首次实现无训练、无参数更新的奖励模型去偏，核心创新在于通过 SAEs 的稀疏特征识别与零化，避免了直接激活抑制导致的性能崩溃，并揭示格式相关特征在浅层可跨模型迁移。

**🔧 技术方法**

采用预训练稀疏自编码器字典，对隐藏层特征进行强度–稳定性评分，选择 top‑K 关键特征；通过前向钩子在对应层将这些特征置零并解码重构隐藏表示。

**📊 数据集**

使用生成的 Markdown‑vs‑plain‑text 对照样本（约 500 对），RM‑Bench 基准数据集，以及 Gemma Reward 模型与礼貌性偏差对照实验数据。

**📈 对比分析**

与未去偏的 RM‑Bench Easy、Normal、Hard 三种切分进行对比；SteerRM 在 Hard 切分平均提升约 7.3 分，Easy 切分略降 5.1 分，但整体保持稳定；直接激活抑制法虽提升 Hard 但导致 Normal/Easy 近随机，说明 SAE 抑制更为精细。

**⚠️ 局限性**

局限性：依赖与模型架构兼容的预训练 SAE 字典；在 Easy 切分因消除格式捷径可能降低性能；对未见格式或极端格式特征的鲁棒性仍待验证。

---

## 102. Reinforcement Learning for Elliptical Cylinder Motion Control Tasks

**arXiv ID:** 2603.12807 | [PDF](https://arxiv.org/pdf/2603.12807v1)

**作者:** Pawel Marczewski `[一作]` (Poznan University of Technology), Szymon Szczesny `[通讯]` (Poznan University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了在有限扭矩约束下椭圆柱体的运动控制，提出了基于强化学习的控制策略，并与传统的能量塑形+LQR双阶段控制器进行对比；

**💡 创新点**

将深度Q网络（DQN）应用于椭圆柱体的四种转动任务，并设计了归一化奖励函数以提升学习效果；

**🔧 技术方法**

使用深度Q网络（DQN）、经验回放、ε-贪婪探索、LQR线性化控制器以及自定义的能量塑形法；

**📊 数据集**

仅使用仿真环境生成的数据，改变质量m和半长轴a作为实验参数；

**📈 对比分析**

采用ISE/ITSE指标与传统能量塑形+LQR对比，结果表明在0→π/2和0→π两任务中RL控制器在误差积分和时间加权误差上均优于基线；

**⚠️ 局限性**

受限于离散扭矩动作、缺乏摩擦等实际物理效应，且仅在仿真中验证，缺乏实验室硬件验证。

---

## 103. CA-HFP: Curvature-Aware Heterogeneous Federated Pruning with Model Reconstruction

**arXiv ID:** 2603.12591 | [PDF](https://arxiv.org/pdf/2603.12591v1)

**作者:** Gang Hu `[一作]`, Shijun Ma `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为CA-HFP的异构联邦学习框架，允许边缘设备根据自身资源进行结构化、设备专属的剪枝，并通过轻量级的服务器端重建实现全局模型的聚合；

**💡 创新点**

创新点在于：①引入曲率感知的重要性评分（结合梯度、权重和二阶Hessian信息）指导剪枝，②在多任务非IID环境下给出收敛上界并据此推导剪枝阈值，③设计重建机制解决异构剪枝导致的结构不匹配，实现统一的FedAvg式聚合；

**🔧 技术方法**

使用技术包括：本地多步SGD、结构化硬剪枝、基于曲率的损失扰动分析、全局重建重映射、收敛分析与理论证明；

**📊 数据集**

使用的数据集为FMNIST、CIFAR-10与CIFAR-100，并在VGG16和ResNet32两种CNN架构上进行实验；

**📈 对比分析**

与FedAvg、FedProx、PruneFL、FedMP、DapperFL等SOTA方法对比，结果表明在不同的Dirichlet非IID分布和系统异构条件下，CA-HFP在保持甚至提升模型准确率的同时，显著降低了通信参数量（相当于多达90%剪枝）和计算FLOPs，且完成时间更短；

**⚠️ 局限性**

局限性在于：①重建过程在大规模模型或稀疏程度极高时可能产生重构误差；②曲率估计基于局部梯度变化近似，可能在高度非线性区域失效；③实验多聚焦于图像分类任务，跨领域或文本任务的适用性尚待验证。

---

## 104. Exploring the role of embodiment on intimacy perception in a multiparty collaborative task

**arXiv ID:** 2603.12783 | [PDF](https://arxiv.org/pdf/2603.12783v1)

**作者:** Amine Benamara `[一作]` (Université Paris-Saclay), Julien Saunier `[通讯]` (INSA Rouen Normandie)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在协作桌游中不同人工代理形态对群体亲密度和凝聚力的影响，提出实验协议并收集人类互动数据。

**💡 创新点**

首次系统性比较物理机器人、虚拟ECA及混合形态在多方合作任务中的社会影响，并设计以亲密行为为核心的代理行为生成框架。

**🔧 技术方法**

使用 Unity/Furhat 机器人、有限状态机决策、LLM (Gemini) 与词向量进行词汇匹配、MobileNetv2 情感检测、MediaPipe 注视追踪等技术。

**📊 数据集**

基于“Mot Malin”协作桌游收集的 12 场录像及其注释（情感、注视、手势、语音转写）构成实验数据集。

**📈 对比分析**

通过多维问卷（亲密度、社交技能、归属感、凝聚力等）在三种形态条件下对比，预期能显著检验形态差异对群体动态的影响。

**⚠️ 局限性**

局限包括样本量小、性别年龄分布不平衡、未考虑参与者间已有关系以及共享键盘可能导致亲密偏差。

---

## 105. "I Should Know, But I Dare Not Ask": From Understanding Challenges in Patient Journeys to Deriving Design Implications for North Korean Defectors' Adaptation

**arXiv ID:** 2603.12632 | [PDF](https://arxiv.org/pdf/2603.12632v1)

**作者:** Hyungwoo Song `[一作]` (Seoul National University), Hyunggu Jung `[通讯]` (Seoul National University)

**通讯引用:** 590 | [OpenAlex ID](https://openalex.org/A5036160535)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过访谈与原型评估，提出并实现了面向朝鲜脱北者的医疗沟通支持系统MediBridge，帮助他们在医院就诊前进行情景化模拟并生成“辅助笔记”；

**💡 创新点**

创新点在于将AI驱动的对话排练与心理结构化支持结合，针对脱北者特有的语言、社会文化与权利认知障碍，提供可定制的“辅助笔记”与心理安全框架；

**🔧 技术方法**

使用GPT‑4o作为对话核心，结合ElevenLabs语音合成，系统实现语义分析、评估打分、情景脚本生成与多模态交互；

**📊 数据集**

采用10名脱北者的访谈语料与15名用户的使用日志作为评测数据集，未使用公开大规模医疗对话语料；

**📈 对比分析**

与简化版（仅文本对话+关键词笔记）进行对照实验；在六项沟通维度上，MediBridge平均得分显著高于基线与简版（p<0.01），但可用性得分略低；

**⚠️ 局限性**

局限包括样本量小、仅在首尔地区受访、实验为受控环境且评估为短期使用，未验证在真实就诊情境中的效果与长期影响。

---

## 106. Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization

**arXiv ID:** 2603.12663 | [PDF](https://arxiv.org/pdf/2603.12663v1)

**作者:** Kazuto Nakashima `[一作]` (Kyushu University), Oscar Martinez Mozos `[通讯]` (Technical University of Cartagena)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用多模态LiDAR全景深度与反射率图像，构建了大规模数据集，并基于卷积神经网络实现室外地点分类。

**💡 创新点**

创新点包括：①创建了Multi-modal Panoramic 3D Outdoor (MPO) 数据集；②提出了针对全景图像的水平循环卷积(HCC)和行最大池化(RWMP)网络改进；③设计了多模态融合方案，Softmax平均融合在性能上表现最佳。

**🔧 技术方法**

使用技术主要有卷积神经网络（改版VGG、HCC、RWMP）、多模态融合（Softmax平均、Adaptive、Early、Late Fusion）、数据增强（水平翻转、循环移位）以及Grad‑CAM可视化。

**📊 数据集**

使用的数据集为MPO的稀疏子集，包含六类室外地点（海岸、森林、室内/室外停车场、住宅区、城市区）及对应的全景深度与反射率图像。

**📈 对比分析**

通过k折交叉验证与传统手工特征方法（Spin‑Image/GIST/LBP+SVM）以及ResNet进行对比，单模深度VGG取得97.18%准确率，单模反射率VGG+RWMP+HCC取得95.92%，而Softmax平均融合达97.87%，均优于传统方法。

**⚠️ 局限性**

局限性包括：仅涵盖六类标签，类别分布不平衡；深度图像质量受传感器分辨率限制；早期融合方法效果不佳，且难以直接推广至更大规模或更复杂的环境与更多类别。

---

## 107. Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models

**arXiv ID:** 2603.12271 | [PDF](https://arxiv.org/pdf/2603.12271v1)

**作者:** Boyu Qiao `[一作]` (Institute of Information Engineering), Yunya Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1133 | [OpenAlex ID](https://openalex.org/A5060246267)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多更新知识实例（DKI）评估框架，并通过端点检索（早期与最新状态）来检测LLM在多次事实更新中的检索偏差。

**💡 创新点**

将认知心理学中的AB-AC干扰范式与多次更新结合，系统化建模多候选竞争，并发现检索偏差随更新次数增长而放大；同时首次对内部信号（注意力、隐藏状态相似度、logit）进行诊断并尝试认知启发式提示干预。

**🔧 技术方法**

使用内部信号诊断方法（注意力权重、隐藏状态余弦相似度、输出logit/置信度）以及多种认知启发式提示（重复练习、语义阐释、记忆整合、定向遗忘）。

**📊 数据集**

构造合成的随机cue‑value序列以及基于EvolveBench的真实多轮更新数据，进一步用GPT‑4.1‑N将真实更新重写为叙事长文本。

**📈 对比分析**

与多种LLM（LLaMA‑3.1、Qwen‑2.5/3、GPT‑4.1‑N等）比较，实验显示早期准确率保持高水平，而最新状态准确率随更新长度T增大明显下降，导致ELAG扩大；提示干预可略微提升最新准确率，但无法根除偏差。

**⚠️ 局限性**

仅进行端点检索限制了行为观察；合成与真实数据存在生态差距；内部信号诊断仅为相关性，缺乏因果验证；提示干预仅为表面修正，未实现根本性更新追踪。

---

## 108. AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network

**arXiv ID:** 2603.12659 | [PDF](https://arxiv.org/pdf/2603.12659v1)

**作者:** Yu Hu `[一作]` (University of British Columbia), Mohsen Zardadi `[通讯]` (TerraSense Analytics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对遥感图像适应视觉‑语言模型提出知识蒸馏框架，利用LLM生成语义丰富的文本原型并进行视觉引导聚合，同时在视觉与文本两侧引入可学习提示进行三方面蒸馏；

**💡 创新点**

创新点在于同时解决语义贫乏与视觉刚性，提出LLM驱动的语义增强文本原型、视觉指导的选择聚合以及三方面（视觉、文本、相似度logit）联合蒸馏的完整体系；

**🔧 技术方法**

核心技术包括LLM领域提示生成、教师‑学生知识蒸馏、深层视觉/文本提示学习、视觉引导下的文本原型聚合以及三方面对齐损失；

**📊 数据集**

实验使用六个光学遥感分类基准（AID、RESISC‑45、EuroSAT、WHU‑RS19、PatternNet、UCMerced）以及两套交叉模态检索基准（RSITMD、RSICD）；

**📈 对比分析**

与零射CLIP/RemoteCLIP/GeoRSCLIP以及多种PEFT方法（CoOp、CoCoOp、MaPLe、PromptKD、APPLeNet、MMRL等）对比，few‑shot分类、基‑新类泛化及交叉模态检索均表现优于或接近最佳基线，尤其在1‑shot和基‑新类上的提升显著；

**⚠️ 局限性**

局限包括对教师预训练模型质量的依赖、文本原型聚合对视觉先验的敏感性以及未能完全克服遥感场景跨源尺度与分辨率的极端差异。

---

## 109. GoalSwarm: Multi-UAV Semantic Coordination for Open-Vocabulary Object Navigation

**arXiv ID:** 2603.12908 | [PDF](https://arxiv.org/pdf/2603.12908v1)

**作者:** MoniJesu Wonders James `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserokou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GoalSwarm：一种完全去中心化的多无人机框架，用于在未知室内环境中进行开词汇目标导航。

**💡 创新点**

创新点包括：① 采用 SAM3 的零射击检测与分割，实现无需训练即可识别任意目标；② 构建 Bayesian Value Map（BVM）对多视角检测置信度进行贝叶斯融合，并用 UCB 进行前沿评分；③ 通过成本-效用投标与空间分离惩罚实现分布式前沿分配，避免冗余探索。

**🔧 技术方法**

关键技术包括：零射击视觉语言模型（SAM3）、2D 上投影语义占据网格、贝叶斯价值图、UCB 前沿评分、Fast Marching Method 路径规划、PID 深度避障、以及基于 ZMQ 的异步通信。

**📊 数据集**

使用 GOAT-Bench（HM3D 真实感室内场景）与 VisFly 仿真平台进行实验，涵盖多目标导航任务。

**📈 对比分析**

与单无人机、随机探索、无共享地图、贪婪分配等基线对比，GoalSwarm 在目标子任务上实现 45% 成功率、0.179 SPL，明显优于单机 10%/0.078 SPL 和随机 20%/0.084 SPL，显示出更高的导航效率和更好的成功率。

**⚠️ 局限性**

局限性包括：假设完美测距与通信；2D 投影可能将垂直分离的障碍物误合并；易受伪检测、可达性不足和小物体遮挡影响；目前仅在室内仿真环境中验证，缺乏真实世界噪声与动态障碍物的评估。

---

## 110. The Future of Feedback: How Can AI Help Transform Feedback to Be More Engaging, Effective, and Scalable?

**arXiv ID:** 2603.12463 | [PDF](https://arxiv.org/pdf/2603.12463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 111. Verification of Robust Properties for Access Control Policies

**arXiv ID:** 2603.13181 | [PDF](https://arxiv.org/pdf/2603.13181v1)

**作者:** Alexander V. Gheorghiu `[一作]` (University of Southampton), Alexander V. Gheorghiu `[通讯]` (University of Southampton)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5083790955)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出一种对访问控制策略的“鲁棒属性”验证框架，定义了支持判断⊩_Pϕ，并将其归约为对二阶逻辑编程的证明搜索，证明了其正确性与完备性。

**💡 创新点**

创新点在于：①把鲁棒性视为基扩展语义的推理问题；②实现了验证的组合性，使已验证属性在策略扩展后无需重新验证；③把无穷的“所有扩展”量化化简为可执行的有限证明搜索。

**🔧 技术方法**

使用的技术包括：基扩展语义、证明论语义、二阶逻辑编程（带原子量化）、统一证明搜索、Horn 子句语言以及对鲁棒连接词的对消形式编码。

**📊 数据集**

实验使用的示例是会议管理系统（Conference Management System），作为演示而非大规模数据集；并未报告真实数据集实验。

**📈 对比分析**

论文未给出与现有模型理论或逻辑程序化工具的定量比较；通过理论证明展示了组合性与有限性，暗示在策略扩展时可显著降低重新验证成本。

**⚠️ 局限性**

局限性包括：①仅处理单调扩展的策略，无法直接处理资源敏感或非单调策略；②假设策略一致（不支持0腐败情况）；③缺乏实证性能评估和大规模案例验证。

---

## 112. SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning

**arXiv ID:** 2603.12976 | [PDF](https://arxiv.org/pdf/2603.12976v1)

**作者:** Md Anwar Hossen `[一作]` (Iowa State University), Ali Jannesary `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练、基于语义的联邦数据子集选择框架SCOPE，用以在极端长尾且非IID的科学数据上进行高效、低通信成本的核心样本筛选。

**💡 创新点**

创新点在于：①三种基于零样本视觉‑语言模型的可扩展标量度量（表示、多样性、边界接近度）实现无训练评估；②仅传输标量汇总实现全局一致性，避免高维特征上传；③全局聚合的长尾保护策略与本地双阶段剪枝（异常过滤+冗余平衡）共同抑制类不平衡与噪声。

**🔧 技术方法**

技术包括：MobileCLIP‑S2 零样本投影、三维语义度量、全局均值方差聚合、标准化 Z‑score、双阶段剪枝、基于权重的长尾稀有度权重；以及针对不同基础模型的架构无关设计。

**📊 数据集**

使用四个数据集：CIFAR‑10、Tiny‑ImageNet、CIFAR‑100、Ultrahigh Carbon Steel Micrograph Database (UHCS)，并通过不同Dirichlet α与IR不平衡比模拟极端非IID和长尾情形。

**📈 对比分析**

与FedAvg、Random、FedCS、FedCore、EL2N、Forgetting Events、Gradient Norm等基线相比，SCOPE在所有数据集与稀疏率下保持或提升精度，同时实现128×–512×的上行带宽压缩、7.72×的时间加速与显存/ FLOP 降低；在高稀疏率时精度提升显著，表明对长尾分布的鲁棒性。

**⚠️ 局限性**

局限包括：①对视觉‑语言模型的语义表达能力依赖，极端专业域仍需合适提示；②全局聚合仍需少量标量上传，若客户端资源极端受限可能受影响；③在极端非IID时仍可能出现类别过度筛除，需要进一步调节阈值；④理论收敛分析基于理想假设，实际训练中可能受噪声或模型不匹配影响。

---

## 113. Virtual reality for large-scale laboratories based on colorized point clouds: design and pedagogical impact

**arXiv ID:** 2603.12727 | [PDF](https://arxiv.org/pdf/2603.12727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 114. KernelFoundry: Hardware-aware evolutionary GPU kernel optimization

**arXiv ID:** 2603.12440 | [PDF](https://arxiv.org/pdf/2603.12440v1)

**作者:** Nina Wiedemann `[一作]` (Intel Corporation), Benjamin Ummenhofer `[通讯]` (Intel Corporation)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文提出了 KernelFoundry，一个基于 LLM 的进化框架，用 MAP‑Elites 进行质量多样性搜索、共同进化 meta‑prompt 以及模板化参数优化，能够在 CUDA、SYCL 等多平台上自动生成高性能 GPU 核心。

**💡 创新点**

其创新点在于将 MAP‑Elites 与 LLM 生成结合，利用元提示共进化避免上下文污染，同时通过模板化参数搜索实现硬件感知的优化，从而在跨平台场景下大幅提升 kernel 性能。

**🔧 技术方法**

所使用的技术包括：多模型 LLM 推理（OpenAI、Anthropic、vLLM）、MAP‑Elites 进化搜索、梯度启发式变异与提示生成、meta‑prompt 演化、静态代码分析得到行为坐标、以及分布式编译与执行架构。

**📊 数据集**

实验数据集涵盖 KernelBench（经过过滤的 111 题）、Robust‑kbench、用于自定义任务的 Llama‑3 旋转嵌入以及 oneDNN 操作，所有数据均以 PyTorch 为基准。

**📈 对比分析**

通过与 AI CUDA Engineer、Robust‑kbench、Kernelsseum、OpenEvolve 等公开基线进行对比，采用正确率和 speedup 指标；在 SYCL 任务上平均提升 2.3×，在 CUDA 的 L1/L2 任务上分别获得 1.24×/2.1× 的速度提升，显著优于现有方法。

**⚠️ 局限性**

局限性包括对特定 LLM 版本和硬件的依赖，实验可重复性受限于硬件差异；框架仍需较多迭代才能收敛，且在奖励逆向或硬件过拟合等问题上存在风险。

---

## 115. MoEKD: Mixture-of-Experts Knowledge Distillation for Robust and High-Performing Compressed Code Models

**arXiv ID:** 2603.13213 | [PDF](https://arxiv.org/pdf/2603.13213v1)

**作者:** Md. Abdul Awal `[一作]` (University of Saskatchewan), Chanchal K. Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 9312 | [OpenAlex ID](https://openalex.org/A5102756770)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MoEKD 框架，将多专家模型聚合后进行知识蒸馏，提升代码语言模型压缩后在漏洞检测任务的性能与鲁棒性。

**💡 创新点**

首次将 Mixture‑of‑Experts 机制引入代码模型知识蒸馏，实现多教师知识融合，显著提升准确率与对抗鲁棒性。

**🔧 技术方法**

核心技术包括 Mixture‑of‑Experts（专家与路由器训练、logit 级聚合）、KL 散度蒸馏、基于 CodeBERT/GraphCodeBERT 的语言模型。

**📊 数据集**

使用 BigVul 数据集（C/C++ 漏洞函数，含 CWE 标签）并采用 Identifier renaming 攻击（ALERT、MHM、WIR‑Random）。

**📈 对比分析**

与 Compressor、AVATAR 等单教师蒸馏基线对比，准确率提升最高 13%，攻击成功率降低最高 35.8%。

**⚠️ 局限性**

局限在于训练时教师端计算成本大幅提升，路由器准确率约 62%，对更强攻击的鲁棒性仍有限。

---

## 116. Language-Grounded Decoupled Action Representation for Robotic Manipulation

**arXiv ID:** 2603.12967 | [PDF](https://arxiv.org/pdf/2603.12967v1)

**作者:** Wuding Weng `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 31113 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出了语言引导的动作表示框架LaDA，通过将7-DoF动作分解为翻译、旋转、夹爪三种语言原语，并结合软标签对比学习实现视觉、语言与控制的解耦与对齐，支持在新任务和真实机器人上执行高质量的操控；

**💡 创新点**

其创新点在于：①利用自然语言对动作原语进行语义标注，构建可解释的低层动作空间；②采用语义引导的软标签对比学习，允许不同任务间动作以连续相似度对齐；③设计自适应损失权重机制，以动态平衡模仿与对比学习，提升训练稳定性与泛化能力；

**🔧 技术方法**

关键技术包括软标签InfoNCE对比学习、CLIP预训练的视觉与语言编码器、FiLM融合+MLP适配器、动作原语离散化与语言描述生成、以及自适应损失加权和轻量级动作头；

**📊 数据集**

使用的大规模数据集包括OXE（100万+真实轨迹）、LIBERO与MimicGen两大仿真基准，以及Frankia Panda机械臂的真实世界pick‑and‑place实验；

**📈 对比分析**

在LIBERO和MimicGen上与多种VLA基线（如CLIP‑RT、FlowVLA、UniACT等）进行对比，LaDA在LIBERO的平均成功率为93.6%（与CLIP‑RT相当或略优），在MimicGen平均成功率提升至约67%（明显优于对比基线），并在跨任务泛化测试中显著优于其他方法；

**⚠️ 局限性**

局限性包括对大规模预训练数据的依赖、对极其复杂动态交互的泛化尚不充分、软标签相似度矩阵需要手工设计、缺乏自纠正机制以及对超长序列任务的实时执行尚未充分验证。

---

## 117. AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents

**arXiv ID:** 2603.12564 | [PDF](https://arxiv.org/pdf/2603.12564v1)

**作者:** Zekun Wu `[一作]` (Centre for Artificial Intelligence), Maria Perez-Ortiz `[通讯]` (Centre for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对金融咨询对话的清洁与污染版本进行并行重放，评估工具增强LLM在高风险场景下的安全性与效用，发现标准NDCG指标对安全违规不敏感，提出sNDCG和轨迹级安全违规率等新评估方法。

**💡 创新点**

创新点在于提出配对轨迹诊断协议，分解信息通道与记忆通道对安全漂移的贡献，发现“评估盲点”现象，并通过安全惩罚的NDCG揭示隐藏的安全风险。

**🔧 技术方法**

使用ReAct框架、工具调用、信息通道与记忆通道的中介式分解、GemmaScope 2稀疏自编码器分析、以及自定义的sNDCG、SVR、DRIFT等多维度指标。

**📊 数据集**

采用公开的Conv‑FinRe多轮金融对话数据集（10名用户、23步、10只股票），并在七种LLM上进行实验。

**📈 对比分析**

在7个模型上比较，标准NDCG与UPR保持≈1，显示效用不受污染影响，但安全违规率SVR_s高达65–93%，sNDCG显著下降（0.51–0.74），证明仅凭质量指标无法反映安全性能。

**⚠️ 局限性**

局限包括仅使用10只股票的小规模样本、记忆通道设计可能不具普适性、单一工具种类的污染测试以及在Claude Sonnet上进行的通道分离和剂量响应实验。

---

## 118. Self-Reported Side Effects of Semaglutide and Tirzepatide in Online Communities

**arXiv ID:** 2603.12341 | [PDF](https://arxiv.org/pdf/2603.12341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 119. Design-Specification Tiling for ICL-based CAD Code Generation

**arXiv ID:** 2603.12712 | [PDF](https://arxiv.org/pdf/2603.12712v1)

**作者:** Yali Du `[一作]` (Nanjing University), Ming Li `[通讯]` (Nanjing University)

**通讯引用:** 23366 | [OpenAlex ID](https://openalex.org/A5100351402)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于知识充足性的设计规格覆盖度（DST）作为In-Context Learning中CAD代码生成的示例选择方法。

**💡 创新点**

创新点在于将知识充足性建模为多粒度设计规格tiling ratio，并证明其为子模函数，进而使用贪心算法实现(1‑1/e)近似，避免了传统相似度或多样性策略的冗余与缺失。

**🔧 技术方法**

采用多粒度n‑gram分解提取设计规格组件，计算tiling ratio作为知识充足性指标，利用子模最大化的贪心算法进行示例选择，并将选定示例与查询拼接为ICL提示输入LLM（Qwen3、DeepSeek‑V3、Claude 4.5‑Haiku）生成CAD代码。

**📊 数据集**

使用Text2CAD数据集（原178K，筛选后151K样本），按难度划分为Easy、Middle、Hard三组。

**📈 对比分析**

与零样本、随机、相似度检索（LDSIM、BM25）以及多样性采样等ICL策略进行对比，DST在有效语法率、IoU、Chamfer距离和边缘Chamfer距离等四项指标上均显著优于基线，尤其在中等难度任务中提升最为明显；较少的shot数即可达到或逼近最佳性能，并且tiling ratio与生成质量呈线性正相关。

**⚠️ 局限性**

局限性包括：依赖预先提取的n‑gram特征，计算大规模数据库时开销较大；未引入学习的语义相似度；对极其复杂或高度动态的CAD任务仍可能不足；仅在CAD代码生成场景验证，未验证跨域推广性。

---

## 120. Collaborative Multi-Agent Optimization for Personalized Memory System

**arXiv ID:** 2603.12631 | [PDF](https://arxiv.org/pdf/2603.12631v1)

**作者:** Wenyu Mao `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 16875 | [OpenAlex ID](https://openalex.org/A5100389037)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CoMAM——一种协同强化学习框架，用于联合优化个性化记忆系统中的多智能体；

**💡 创新点**

创新点在于将异构、异步智能体的执行序列化为马尔可夫决策过程（MDP）以捕获跨智能体依赖，并引入基于排名一致性的自适应信用分配机制，使局部奖励与全局性能对齐；

**🔧 技术方法**

核心技术包括协同强化学习（采用GRPO算法）、MDP轨迹正则化、基于NDCG的自适应信用分配、以及多粒度记忆构建与检索的多智能体架构；

**📊 数据集**

实验数据集为PersonaMem，包含180条长上下文用户–LLM对话和约6000道多选查询，测试三种历史长度（32K、128K、1M）；

**📈 对比分析**

与无记忆、提示式记忆和独立RL优化的记忆系统相比，CoMAM在所有设置下均显著提升查询答案准确率（如在1M长度下从0.57提升至0.70，平均提升约13%）；

**⚠️ 局限性**

限制在于依赖大规模LLM与计算资源，且信用分配机制对排名一致性估计敏感，具体细节见论文附录。

---

## 121. Complex-Valued Probability Measures and Their Applications in Information Theory

**arXiv ID:** 2603.12297 | [PDF](https://arxiv.org/pdf/2603.12297v1)

**作者:** Siang Cheng `[一作]` (Zhejiang University), Tianxiao Pang `[通讯]` (Zhejiang University)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5020164576)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了复值概率测度的框架，并基于该框架定义了复熵、复散度和复度量，用于衡量概率分布的均匀性与相似性。

**💡 创新点**

将传统概率从实数域扩展到复平面，利用相位调制的概率密度构造以相干性为基础的新信息量度，提供了比Shannon熵和KL散度更能捕捉分布形状差异的工具。

**🔧 技术方法**

运用复分析、路径积分类比、核密度估计、置换检验以及数值积分等技术来定义、推导性质并实现复度量的计算。

**📊 数据集**

主要使用理论分布（正态、均匀等）进行模拟实验，未给出真实数据集；实验集中在对这些分布的复熵和复度量进行数值验证。

**📈 对比分析**

与Shannon熵、KL散度、总变距离等传统度量做理论比较，显示在极值、连续性、可加性等方面的差异；在非参数两样本检验中，通过置换检验验证复度量在不同β值下对分布形状差异的灵敏度，结果表明可调参数β可显著提升检验功效。

**⚠️ 局限性**

局限性包括：高维多变量推广难度大；β的选择缺乏统一的数据驱动准则；核密度估计误差会影响复度量的精度；理论实验与实际数据的应用之间仍存在距离，需要进一步实证验证。

---

## 122. COAD: Constant-Time Planning for Continuous Goal Manipulation with Compressed Library and Online Adaptation

**arXiv ID:** 2603.12488 | [PDF](https://arxiv.org/pdf/2603.12488v1)

**作者:** Adil Shiyas `[一作]` (Worcester Polytechnic Institute), Constantinos Chamzas `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 285 | [OpenAlex ID](https://openalex.org/A5061640673)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出一种通过任务覆盖区域（TCR）分区、离线根路径压缩库和在线轻量级适配器，实现对连续目标姿态的常数时间、碰撞安全运动规划。

**💡 创新点**

创新点在于：①将连续目标空间映射到有限的TCR，保证单一路径可覆盖一整个连续子集；②仅为部分TCR生成根路径，其余通过适配器快速覆盖；③实现离线压缩库后，在线查询成为 O(1) 并保证覆盖性。

**🔧 技术方法**

使用技术包括：任务覆盖区域（TCR）分区、离线根路径规划与压缩、线性插值、动态运动本体（DMP）和简易轨迹优化（STO）适配器、离线碰撞验证、Mujoco 仿真、UR10 真实机器人部署。

**📊 数据集**

使用自定义仿真环境（桌面 Table、书架 Shelf、笼子 Cage）以及真实 UR10 在桌面上的实验；机器人为 Panda（7 轴）和 Fetch（8 轴）。

**📈 对比分析**

与 RRT‑Connect（3 秒超时）和经验库基线（Library Baseline）对比；CoAd 在 1000 次随机查询中保持 100% 成功率，查询时间比 RRT‑Connect 快 2–3 个数量级，路径质量与基线相当或更好，DMP 适配器提供最佳质量，压缩率最高可达 97%。

**⚠️ 局限性**

局限性：根路径选择目前为随机采样，未进行优化；仅适用于半静态环境（目标移动，障碍固定）；适配器不考虑障碍约束，可能在极端姿态变化或高度复杂场景下失效；对动态障碍的鲁棒性尚未验证。

---

## 123. Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies

**arXiv ID:** 2603.12667 | [PDF](https://arxiv.org/pdf/2603.12667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 124. CarPLAN: Context-Adaptive and Robust Planning with Dynamic Scene Awareness for Autonomous Driving

**arXiv ID:** 2603.12607 | [PDF](https://arxiv.org/pdf/2603.12607v1)

**作者:** Junyong Yun `[一作]` (Hanyang University), Jun Won Choi `[通讯]` (Seoul National University)

**通讯引用:** 3978 | [OpenAlex ID](https://openalex.org/A5102839991)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

CarPLAN提出了一种基于模仿学习的轨迹规划框架，结合空间感知编码和专家混合解码，实现了对不同驾驶情境的自适应规划。

**💡 创新点**

创新点包括：Displacement‑Aware Predictive Encoder（DPE）通过自监督预测车辆与场景元素的位移提升空间感知；Context‑Adaptive Multi‑Expert Decoder（CMD）采用Mixture‑of‑Experts动态路由以适应多样化交通环境。

**🔧 技术方法**

使用Transformer Encoder/Decoder、Mixture‑of‑Experts、跨注意力与自监督位移预测、以及多模态轨迹与置信度头等技术。

**📊 数据集**

主要在nuPlan数据集（含Val14、Test14‑Hard等子集）进行评估，同时在Waymax benchmark验证泛化能力。

**📈 对比分析**

相较于最新基准（如Diffusion‑Planner、BeTopNet、PLUTO等），CarPLAN在闭环模拟CLS‑NR、CLS‑R上分别获得84.6、91.4等分数，显著领先，且在碰撞率、离场率等指标亦表现更佳。

**⚠️ 局限性**

局限在于依赖完美感知模块，未与感知网络联合训练，且在推理阶段引入的MoE带来一定计算开销。

---

## 125. Topo-R1: Detecting Topological Anomalies via Vision-Language Models

**arXiv ID:** 2603.13054 | [PDF](https://arxiv.org/pdf/2603.13054v1)

**作者:** Meilong Xu `[一作]` (Stony Brook University), Chao Chen `[通讯]` (Stony Brook University)

**通讯引用:** 27923 | [OpenAlex ID](https://openalex.org/A5100408358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Topo‑R1框架，实现对血管、神经纤维、道路网络等管状结构中拓扑异常的检测与分类。

**💡 创新点**

创新点包括：① 自动化生成跨域的四种拓扑异常并验证，构建首个大规模基准；② 采用两阶段训练：监督微调 + Group Relative Policy Optimization（GRPO）强化学习；③ 设计专门的拓扑复合奖励（类型匹配、定位、clDice 等），实现结构级别的误差定位与类别识别。

**🔧 技术方法**

使用了 Vision‑Language Models（VLM），自定义奖励函数，Hungarian 匹配，clDice 与 Betti 数学工具，GRPO 强化学习框架，自动化数据生成管道。

**📊 数据集**

数据集由道路网络、裂纹检测和视网膜血管三类公开数据合成，训练集约 12.9k 样本，RL 训练集 50k，测试集 4.2k，覆盖 4k 标注的拓扑异常。

**📈 对比分析**

通过零样本、SFT、Topo‑R1 三阶段对比实验，评估 Precision、Recall、F1（多 IoU 阈值）、aF1、计数精度等指标。Topo‑R1 在所有指标上显著优于开源 VLM 及闭源模型，最高 F1@0.5 约 45% 以上。

**⚠️ 局限性**

局限性：① 仍以合成数据为主，真实场景多样性和复杂度有限；② 奖励设计与 RL 训练较为复杂，迁移成本高；③ 未在 3D 数据或更大规模网络上验证；④ 对模型规模仍有一定敏感性。

---

## 126. From AI Weather Prediction to Infrastructure Resilience: A Correction-Downscaling Framework for Tropical Cyclone Impacts

**arXiv ID:** 2603.12828 | [PDF](https://arxiv.org/pdf/2603.12828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 127. DINOLight: Robust Ambient Light Normalization with Self-supervised Visual Prior Integration

**arXiv ID:** 2603.12579 | [PDF](https://arxiv.org/pdf/2603.12579v1)

**作者:** Youngjin Oh `[一作]` (Seoul National University), Nam Ik Cho `[通讯]` (Seoul National University)

**通讯引用:** 5027 | [OpenAlex ID](https://openalex.org/A5055171648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了DINOLight框架，将自监督视觉模型DINOv2的特征作为先验，融合进环境光照归一化与阴影去除的端到端恢复流程。

**💡 创新点**

创新点在于提出自适应特征融合模块AFFM和辅助跨注意力ACA，实现空间域与频域双重注意力，充分利用DINOv2的几何与语义信息。

**🔧 技术方法**

使用ViT-DINOv2提取多层特征，Transformer结构中的SFDINO块，AFFM权重融合，双域（空间+频域）ACA，以及基于ℒ1+MS-SSIM的端到端训练。

**📊 数据集**

主要数据集为Ambient6K（5,000对高分辨率图像）用于ALN训练与测试，以及ISTD/ISTD+阴影去除数据集用于通用性验证。

**📈 对比分析**

与IFBlend、PromptNorm、SwinIR等多种基准方法比较，DINOLight在Ambient6K上实现PSNR 22.8 dB、SSIM 0.838、LPIPS 0.107，显著优于前沿方法，且仅需7.9 GMAC，模型轻量化。

**⚠️ 局限性**

局限性：对极端光照变化或复杂光源分布的细粒度分辨率仍有限；模型依赖预训练的DINOv2，若在不同域迁移可能需额外适配。

---

## 128. LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation

**arXiv ID:** 2603.12522 | [PDF](https://arxiv.org/pdf/2603.12522v1)

**作者:** Himel Ghosh `[一作]` (Technical University of Munich), Nick Elias Werner `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了LLM BiasScope，一个可实时比较多款大型语言模型并自动进行偏见检测与类型分类的Web应用；

**💡 创新点**

将多模型流式对比与两阶段偏见检测（句级检测+类型分类）集成到交互式可视化平台，实现即时、可视化的偏见差异分析；

**🔧 技术方法**

前端采用Next.js/React/Tailwind，后端使用Next.js API路由与Vercel AI SDK调用多供应商LLM；偏见检测使用Hugging Face inference endpoint的bias-detector模型，偏见类型分类使用maximuspowers/bias-type-classifier，使用Server‑Sent Events实现流式输出；

**📊 数据集**

使用CrowS‑Pairs和BABE两大偏见检测基准评估模型；偏见类型分类依据公开的GUS框架；在演示中采用自定义医疗、职业、教育领域的测试提示；

**📈 对比分析**

通过同步双侧聊天面板，计算每模型的句数、偏见比例、平均偏见分数与类型分布，使用SS、准确率、F1等指标评估；实验显示选用的bias-detector在BABE上F1达85.8%，系统整体延迟随文本长度近线性，典型查询子秒级；

**⚠️ 局限性**

未处理模型的拒绝或回避性偏见；仅集成公开API且未实现自定义API密钥功能；评估基准多为合成文本，可能不完全代表真实LLM生成内容。

---

## 129. From Woofs to Words: Towards Intelligent Robotic Guide Dogs with Verbal Communication

**arXiv ID:** 2603.12574 | [PDF](https://arxiv.org/pdf/2603.12574v1)

**作者:** Yohei Hayamizu `[一作]` (State University of New York), Shiqi Zhang `[通讯]` (State University of New York)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个利用大型语言模型与任务规划器相结合的机器人导盲犬系统，实现与视觉障碍者的多轮对话式导航与实时场景语言化。

**💡 创新点**

创新点在于将LLM的自然语言理解与ASP规划器的空间推理融合，实现计划可视化（Plan Verbalization）与场景可视化（Scene Verbalization），并在开放式服务请求中进行多轮澄清与决策。

**🔧 技术方法**

使用了LLM（如GPT‑4）、ASP（Answer Set Programming）规划器、语音识别模型、ROS导航栈以及人机交互模块。

**📊 数据集**

利用了77个从受试者收集的服务请求库（功能等价位置集合）以及7名合法失明参与者的真实导航实验数据。

**📈 对比分析**

通过在真实环境中对7名受试者进行的三种语音化策略（最小语音化、仅场景语音化、场景+计划语音化）实验以及仿真评估，场景+计划语音化在用户满意度、易用性和导航决策效率上优于其他方案；在仿真中实现了94.8%的准确率、对语音噪声鲁棒，仅计划信息的方案更快但成本更高。

**⚠️ 局限性**

局限性包括：安全评分略低于生物导盲犬、对未知环境的支撑有限、场景语音化仍采用简化策略，且机器人自主控制在实验中通过Wizard‑of‑Oz实现，未能完全验证完整自主导航的可行性。

---

## 130. Aligning Language Models from User Interactions

**arXiv ID:** 2603.12273 | [PDF](https://arxiv.org/pdf/2603.12273v1)

**作者:** Thomas Kleine Buening `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30684 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自蒸馏的自适应学习框架（SDPO），直接利用用户多轮对话中的后续消息作为“回顾”信息，对模型原始输出进行自我改进。

**💡 创新点**

创新点在于：1) 通过比较模型在原始上下文和包含用户后续信息时的概率分布，得到可解释的token级优势信号；2) 将该信号直接用于自蒸馏（逆KL）或策略梯度，形成一种无需外部标签或奖励的“直接从交互学习”方法；3) 证明该方法既能提升对齐与指令遵循，又能实现持续个性化与无损学习。

**🔧 技术方法**

核心技术包括：自蒸馏策略优化（SDPO）、逆KL蒸馏、token级优势计算、off‑policy SDPO（用于已记录的交互数据）、以及对话重提示模板。

**📊 数据集**

使用WildChat与WildFeedback公开对话数据（约14k个包含后续用户消息的交互），以及从模拟用户生成的对话用于持续个性化实验。

**📈 对比分析**

与基线模型（SFT、DPO）对比，SDPO在AlpacaEval 2.0、IFEval、ArenaHard‑v2、MMLU‑Pro等评测集上提升多项指标（如AlpacaEval +8.2%、IFEval +1.3%），且未出现能力退化；在个性化实验中，模型在50条交互后即可达85%胜率，200条后超过95%。

**⚠️ 局限性**

局限性包括：对小模型或不擅长利用后续信息的模型收益有限；训练时需大量对话日志，且对话质量影响程度仍需进一步研究；在极端恶意或误导性后续消息下，方法可能无效或产生安全风险。

---

## 131. Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**arXiv ID:** 2603.13057 | [PDF](https://arxiv.org/pdf/2603.13057v1)

**作者:** Yuki Hirakawa `[一作]` (ZOZO Research), Yoshimitsu Aoki `[通讯]` (Keio University)

**通讯引用:** 3342 | [OpenAlex ID](https://openalex.org/A5070908826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无参考的图像质量评估框架 VTON‑IQA，并构建大规模人类标注基准 VTON‑QBench；

**💡 创新点**

创新点包括：①构建 62,688 张试衣图像与 431,800 质量标注的最大规模基准；②引入 Interleaved Cross‑Attention（ICA）模块专门建模试衣图像与服装、人物图像的交互；③采用 pairwise preference 与回归联合训练，逼近人类主观感知；

**🔧 技术方法**

技术方法：Transformer 三支架构 + ICA、DINOv3 ViT‑L/16 预训练、AdamW 优化、FLUX+LoRA 生成合成服装‑人物对、基准指标 SSIM、LPIPS、FID/KID、DINOv3 等；

**📊 数据集**

使用的数据集：VITON‑HD 与 Dress Code 原始测试集、合成的服装‑人物对、14 种 VTON 模型生成的试衣图像，构成 VTON‑QBench；

**📈 对比分析**

对比方法：与 SSIM、LPIPS、零射 DINOv3 等全参考或零射基准比较，VTON‑IQA 在 PLCC/SRCC/R²/宏/微 pairwise accuracy 上分别达到 0.751/0.750/0.553/0.781/0.790，接近人类水平（0.762/0.760/0.536/0.782/0.791），显著优于传统指标；

**⚠️ 局限性**

局限性：与人类评分仍有细粒度差距；依赖大量人工标注；对姿态/缩放变化的鲁棒性有限；缺乏对背景/非目标区域的显式建模。

---

## 132. FoSAM: Forward Secret Messaging in Ad-Hoc Networks

**arXiv ID:** 2603.12871 | [PDF](https://arxiv.org/pdf/2603.12871v1)

**作者:** Daniel Schadt `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3510 | [OpenAlex ID](https://openalex.org/A5053465128)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 FoSAM 协议，实现无对等握手、前向保密的点对点无线网络即时通讯

**💡 创新点**

创新在于将时间基准的异步密钥演化与 HIBE 转化相结合，解决不可靠网络下的前向保密与匿名性

**🔧 技术方法**

采用 HIBE、Canetti 变换、三重多数投票时间同步、BLE 广播、AES 等技术

**📊 数据集**

使用基于 OMNeT++ 的三种移动模型（静态、聚合、真实人流轨迹）作为数据集

**📈 对比分析**

通过微基准、仿真和 Android 原型对比，消息成功率在 92–99%，加密/解密耗时 <10 ms，密钥轮换 ≈120 ms

**⚠️ 局限性**

局限包括对时间同步的依赖、单向消息设计、对抗性噪声影响、密钥轮换延迟及大规模网络时延问题

---

## 133. Taming the Long Tail: Efficient Item-wise Sharpness-Aware Minimization for LLM-based Recommender Systems

**arXiv ID:** 2603.12752 | [PDF](https://arxiv.org/pdf/2603.12752v1)

**作者:** Jiaming Zhang `[一作]` (Zhejiang University), Chaochao Chen `[通讯]` (Zhejiang University)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5028791879)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 LLM 基础的推荐系统在长尾场景下的表现，并提出了 EISAM 方法来缓解长尾问题。

**💡 创新点**

提出了基于项目级别的锐度感知最小化（EISAM），兼顾平滑损失曲面与高效训练，并给出了理论泛化上界。

**🔧 技术方法**

Sharpness-Aware Minimization、项目级加权、LoRA 微调、LLM（Llama2-7B）以及自定义权重函数。

**📊 数据集**

MovieLens‑1M、Steam、Amazon Digital Music (ADM)。

**📈 对比分析**

与 RW、SAM、GroupSAM 等基线对比，EISAM 在 NDCG@10/HR@10 上平均提升约 3–5% 的整体指标，尾部项目提升 8–9% 而计算开销仅增加 5% 以内。

**⚠️ 局限性**

对极端长尾或稀疏项目的鲁棒性仍有限，且对权重函数和 λ 参数的调优敏感。

---

## 134. NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation

**arXiv ID:** 2603.12378 | [PDF](https://arxiv.org/pdf/2603.12378v1)

**作者:** Yuxin Yang `[一作]` (Shanghai University), Weilin Huang `[通讯]` (Fudan University)

**通讯引用:** 11004 | [OpenAlex ID](https://openalex.org/A5009292056)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 NeuroLoRA，一个在 LLM 上进行参数高效微调的 Mixture‑of‑Experts 框架。

**💡 创新点**

创新点包括：① 引入基于神经调制的轻量级门控，动态调节固定稀疏随机投影以实现上下文感知的专家路由；② 提出对比正交损失，显式强制专家子空间分离，提升多任务融合与连续学习性能。

**🔧 技术方法**

采用的技术有：混合专家 LoRA、稀疏随机投影、Bottleneck 上下文门控、对比正交损失、训练自由的模型融合与连续学习策略。

**📊 数据集**

使用的数据集包括 MMLU、ScienceQA 与 GSM8K。

**📈 对比分析**

与 LoRA、AdaLoRA、DoRA、MoLE、FlyLoRA 等基线在单任务、模型融合和连续学习三种场景下进行对比。NeuroLoRA 在单任务上平均提升约 1.7%，在模型融合中性能下降仅 3.4%（低于 FlyLoRA 的 5.1%），在连续学习中的后向迁移 BWT 为 -2.6（优于 FlyLoRA 的 -4.3）。

**⚠️ 局限性**

局限性：实验仅在 Llama‑3‑8B 上进行，任务数量有限；未评估更大模型规模或更长任务序列的效果；门控目前仅基于单词级上下文，未探索跨句子或序列级调制。

---

## 135. Optimizing Task Completion Time Updates Using POMDPs

**arXiv ID:** 2603.12340 | [PDF](https://arxiv.org/pdf/2603.12340v1)

**作者:** Duncan Eddy `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**通讯引用:** 12547 | [OpenAlex ID](https://openalex.org/A5068326377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了任务完成时间公告控制的POMDP/MOMDP模型，并通过QMDP和SARSOP求解器生成最佳公告策略；

**💡 创新点**

创新点在于将公告更新视为连续决策问题，利用混合可观测MDP框架平衡公告准确性与更新频率；

**🔧 技术方法**

使用技术包括POMDP、MOMDP建模、贝叶斯信念更新，QMDP和SARSOP两种离线求解器；

**📊 数据集**

使用的“数据集”是通过仿真生成的带噪声观测序列，未使用真实项目数据；

**📈 对比分析**

通过与两种基线策略（last observed、most likely）在奖励、公告次数、项目延误等指标比较，实验表明QMDP/SARSOP在奖励和公告稳定性上优于基线，减少约70%的不必要更新；

**⚠️ 局限性**

局限性包括：仅考虑单一任务的公告更新；模型假设简化（如观测噪声为高斯、延迟分布为分类）；未在真实项目数据上验证；未处理多任务间相互依赖的情况。

---

## 136. 3DTCR: A Physics-Based Generative Framework for Vortex-Following 3D Reconstruction to Improve Tropical Cyclone Intensity Forecasting

**arXiv ID:** 2603.13049 | [PDF](https://arxiv.org/pdf/2603.13049v1)

**作者:** Jun Liu `[一作]` (Fudan University), Hao Li `[通讯]` (Fudan University)

**通讯引用:** 31373 | [OpenAlex ID](https://openalex.org/A5100348631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了3DTCR，一种物理约束的生成式框架，用来在低分辨率全球预报的基础上重建热带气旋的三维内核细节，进而提升强度预测。

**💡 创新点**

创新点在于将条件流匹配（Conditional Flow Matching）与动态移动域训练相结合，加入时序感知的潜在空间域适配（MMD）与两阶段迁移学习，实现在低成本下获得高分辨率、物理一致的风场。

**🔧 技术方法**

采用的技术包括：Conditional Flow Matching、Rectified Flow、U‑Net骨干、潜在空间MMD域适配、两阶段预训练与微调、移动域跟踪和区域自适应重建。

**📊 数据集**

训练数据为基于ERA5初始/边界条件的3 km分辨率移动域WRF数值模拟（1,903个样本）以及FuXi和ECMWF-HRES的低分辨率预报，测试集为2024年的TC事件。

**📈 对比分析**

与FuXi、ERA5、ECMWF-HRES等基线比较，3DTCR在最大10米风速预测中5天内RMSE平均降低36.5%，与ECMWF-HRES相比RMSE下降约8.9%，并在极端强度阈值上显示出更高的Critical Success Index，证明其在精细结构与强度预测上的显著优势。

**⚠️ 局限性**

局限在于：基准为WRF模拟的“高分辨率”仍是近似真实大气状态；尽管两阶段训练和域适配已缓解平滑问题，但极端误差仍未完全消除；实际运行流程仍需多步预处理，尚未实现一体化的端到端运算。

---

## 137. ChainFuzzer: Greybox Fuzzing for Workflow-Level Multi-Tool Vulnerabilities in LLM Agents

**arXiv ID:** 2603.12614 | [PDF](https://arxiv.org/pdf/2603.12614v1)

**作者:** Jiangrong Wu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34529 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChainFuzzer，一个针对大语言模型代理的灰盒模糊测试框架，用于发现并复现多工具协同执行中的工作流级漏洞。

**💡 创新点**

创新点在于将基于 sink 的工具链提取、Trace‑guided Prompt Solving（TPS）以及防护意识的模糊变异三大技术模块组合，专门针对跨工具数据流的长期依赖关系实现安全边界检测。

**🔧 技术方法**

采用静态代码分析与 LLM 语义过滤构建工具链、利用运行时追踪生成约束并迭代修正提示、在防护层级下进行变异注入与 oracle 检验，形成完整的漏洞发现与复现流程。

**📊 数据集**

实验使用 20 个主流开源 LLM 代理应用（共 998 个工具）构成的数据集，涵盖多种工具类型与工作流模式。

**📈 对比分析**

与单工具或单跳测试相比，ChainFuzzer 在 19/20 应用中共发现 365 条独立漏洞，其中 82.7% 需多工具执行；精度高（边缘 96.5%，链 91.5%），触发率提升至 88.6%，效率为 3.02 条漏洞/百万 token。

**⚠️ 局限性**

局限性包括对运行时环境（网络、沙箱权限、凭据）的敏感性；使用 LLM 可能产生幻觉或不确定性；以及防护机制可能隐藏部分漏洞。

---

## 138. Robots that redesign themselves through kinematic self-destruction

**arXiv ID:** 2603.12505 | [PDF](https://arxiv.org/pdf/2603.12505v1)

**作者:** Chen Yu `[一作]` (Northwestern University), Sam Kriegman `[通讯]` (Northwestern University)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5003985051)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种通过可感知的自毁机理，让机器人在生命周期内主动识别并拆除自身冗余关节，从而重塑自身结构并实现更高效前进的控制方法。

**💡 创新点**

创新点在于：①首次将机器人自毁与控制耦合，构建“自毁‑运动”序列；②通过 Transformer 的序列建模，将多种初始结构下的专家策略整合为统一的通用控制器；③提出 Prompt Reset 机制，解决在未见结构下出现的循环冻结现象。

**🔧 技术方法**

主要技术包括：基于 MuJoCo 的动力学仿真、通过关节扭矩阈值触发模块拆除、强化学习训练专家策略、Transformer 作为通用自毁‑运动控制器、以及将真实机器人数据注入训练集以缩小 sim‑to‑real 差距。

**📊 数据集**

训练数据来自八种手工设计的腿式机器人（A‑H），共收集约 10⁷ 条状态‑动作对；测试数据包含 100 个随机生成的未见结构（A‑I）以及三台实验室构造的物理机器人（O‑R）。

**📈 对比分析**

与基线（随机拆除或不拆除）相比，自毁策略在仿真中平均前进速度从 0.080 m/s 提升至 0.168 m/s（p<0.001），在物理机器人上实现 100% 的重塑与运动成功率，且在未见结构下的前进路径更直线、速度更高。

**⚠️ 局限性**

局限性包括：①自毁过程不可逆，只能去除部件而不能增添；②依赖高强度胶粘连接，实际构建受限；③模型对极端结构变化仍可能产生冻结，需 Prompt Reset 解决；④仅验证了四个模块左右的中小规模机器人，尚未扩展到更大或更复杂的 kinematic 树。

---

## 139. Research on Linear Codes Holding $q$-Ary $t$-Designs

**arXiv ID:** 2603.12761 | [PDF](https://arxiv.org/pdf/2603.12761v1)

**作者:** Xinghao Wu `[一作]` (Beijing Jiaotong University), Junling Zhou `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 212 | [OpenAlex ID](https://openalex.org/A5019035079)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究线性码生成q-元t-设计，系统性提出两大标准与删短准则，并运用自同构群构造新的q-元2-设计。

**💡 创新点**

创新点在于将Delsarte及Assmus–Mattson理论推广为通用标准，并通过自同构群的t-传递性得到无限族的q-元2-设计。

**🔧 技术方法**

主要技术包括码的外距、正则性分析、自动同构群的传递性以及子集和乘积计数等组合方法。

**📊 数据集**

没有使用具体数据集，全部基于理论证明与已有码族（如RS、极大码、最优自保码等）构造。

**📈 对比分析**

通过理论推导得到的设计参数与已知结果完全一致，且在已知码族中取得更高强度（如t=2），展示了方法的有效性。

**⚠️ 局限性**

局限在于仍无法得到t≥3的无限族，且对某些码族的正则性或自同构性假设较强。

---

## 140. Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation

**arXiv ID:** 2603.13131 | [PDF](https://arxiv.org/pdf/2603.13131v1)

**作者:** Zhengwei Xie `[一作]` (University of Science and Technology of China), Kun Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 37838 | [OpenAlex ID](https://openalex.org/A5100318524)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Steve-Evolving 框架，通过将交互经验从结构化文档演进为可执行技能与守卫规则，实现非参数自演进的开放世界具身代理。

**💡 创新点**

核心创新在于细粒度执行诊断与双轨经验蒸馏相结合，将失败原因转化为可执行约束，成功轨迹抽象为可重用技能，并将其注入 LLM 规划器形成闭环。

**🔧 技术方法**

结合多维索引的经验空间、12 类执行诊断指标、双轨蒸馏算法、跨任务知识检索以及 LLM 规划器的上下文注入与局部重规划。

**📊 数据集**

在 Minecraft MCU 长时程技术树任务集（70 个任务、7 阶段）上进行实验。

**📈 对比分析**

与 Jarvis‑1、Optimus‑1 等基线在相同评估协议下对比，Steve‑Evolving 在所有 LLM 变体上均实现最高总体成功率，尤其在后期高依赖任务中提升显著。

**⚠️ 局限性**

仍受限于对 LLM 规划器的依赖、需手工定义诊断项目、对极端环境或大规模经验库的检索效率与存储扩展性未完全验证。

---

## 141. Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs

**arXiv ID:** 2603.12458 | [PDF](https://arxiv.org/pdf/2603.12458v1)

**作者:** Xing Zi `[一作]` (University of Technology Sydney), Mukesh Prasad `[通讯]` (University of Technology Sydney)

**通讯引用:** 8119 | [OpenAlex ID](https://openalex.org/A5006355592)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于拓扑正则化的医疗知识图谱和多跳诊断推理基准 ShatterMed-QA，并通过该基准系统性评估大型语言模型的多跳推理能力。

**💡 创新点**

创新点在于提出 k‑Shattering 算法剔除知识图谱中的高连通“枢纽”节点，结合隐式桥接实体屏蔽和拓扑驱动的硬负样本采样，显著抑制了模型的“shortcut learning”，同时提供可追踪、无幻觉的基准生成流程。

**🔧 技术方法**

使用的技术包括语义驱动的文本分块、UMAP+GMM 聚类、知识图谱提取与对齐、k‑Shattering 拓扑正则化、隐式桥接实体掩蔽、拓扑驱动的硬负采样、以及检索增强生成（RAG）进行模型恢复实验。

**📊 数据集**

数据集为 ShatterMed‑QA，包含 10,558 条中英双语多跳临床问答，覆盖诊断、治疗、药物安全等五大任务，且经过专家验证的 264 条黄金子集。

**📈 对比分析**

通过对 21 种 LLM（包括前沿专有模型、开源通用模型和医学专用模型）的零样本多选测试，发现前沿模型在 Hard 组的错误率高达 50% 以上，硬负错误率（HNE）远超 33% 随机基线，RAG 介入后恢复率可达 70%，表明模型缺陷主要在于知识缺口而非推理逻辑。

**⚠️ 局限性**

局限性包括：生成流程仍以教科书知识为主，无法完全覆盖临床多样性和地区差异；人类评估样本有限，可能存在地域偏差；以及对模型检索组件的依赖可能掩盖了真实推理能力的缺失。

---

## 142. Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models

**arXiv ID:** 2603.13215 | [PDF](https://arxiv.org/pdf/2603.13215v1)

**作者:** Ziqi Ma `[一作]`, Georgia Gkioxari `[通讯]` (California Institute of Technology)

**通讯引用:** 38929 | [OpenAlex ID](https://openalex.org/A5014407395)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新基准 STEVOBench，用来评估视频世界模型在被遮挡或离视时能否独立地继续状态演化；

**💡 创新点**

创新点在于将观察控制（遮挡、关闭灯光、相机转移）与动作控制结合，设计三类评估判定（状态进展、物理可行性、连贯性），并使用专门的 VLM 验证器实现自动化判定；

**🔧 技术方法**

使用基于 VLM 的多专用验证器（控制验证器、状态进展验证器、物理可行性验证器、连贯性验证器）和大规模多模态视频生成模型（Veo 3、Sora 2 Pro、GAN、Genie 3、LingBot‑World 等）；

**📊 数据集**

构建了 225 个任务，涵盖六类自然演化（连续过程、运动学、关系变化、因果变化、状态转换、动物/人类行为），并在公开的多模态视频数据集上进行评测；

**📈 对比分析**

与人类评注进行对比，验证器与人类一致性达 0.8–0.9 以上；在实验中发现闭源视频模型在状态进展上仅 5–20% 成功率，开启源相机控制模型更低；表明当前模型在观察被中断时易停滞或出现不连贯；

**⚠️ 局限性**

局限性包括：验证器对物理可行性判定仍有一定不确定性；基准只关注可见的物理演化，未覆盖更复杂的交互；当前模型训练数据偏向静态或有限动态，导致无法很好学习状态-观察解耦。

---

## 143. GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration

**arXiv ID:** 2603.13068 | [PDF](https://arxiv.org/pdf/2603.13068v1)

**作者:** Yihao Ding `[一作]` (University of Western Australia), Wei Liu `[通讯]` (University of Western Australia)

**通讯引用:** 51387 | [OpenAlex ID](https://openalex.org/A5100641142)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个跨多地区、多采样源、多目标元素的无监督地球化学异常检测基准数据集GeoChemAD，并提出了基于自监督变压器的GeoChemFormer模型，全面对比多种无监督方法的表现。

**💡 创新点**

创新点包括①公开多元化的基准数据集，②利用自监督空间上下文学习和元素依赖建模的Transformer框架，③系统评估并为未来研究提供统一基准。

**🔧 技术方法**

技术手段包括自监督变压器编码器、KD‑tree邻域构造、CLR/ILR闭合转换、特征选择（PCA、因果发现、LLM）、以及统计、经典ML、生成式和Transformer等多种无监督异常检测算法。

**📊 数据集**

使用的数据集为GeoChemAD，来源于西澳大利亚地质调查局（GSWA）的DMPE数据，涵盖8个子集（土壤、沉积物、岩芯），覆盖Au、Cu、W、Ni等多种目标元素。

**📈 对比分析**

通过在8个子集上以AUC为指标，对12种基线模型进行比较，GeoChemFormer平均AUC达0.7712，显著高于AE、VAE、传统统计与机器学习方法，显示出更优越的检测性能。

**⚠️ 局限性**

局限性在于模型对数据预处理（闭合转换、特征选择）和地质环境差异仍敏感；在极端采样稀疏或非典型元素时表现不一；此外Transformer训练成本高且需大规模GPU资源。

---

## 144. Curriculum Sampling: A Two-Phase Curriculum for Efficient Training of Flow Matching

**arXiv ID:** 2603.12517 | [PDF](https://arxiv.org/pdf/2603.12517v1)

**作者:** Pengwei Sun `[一作]` (Stanford University), Pengwei Sun `[通讯]` (Stanford University)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5028986731)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了流匹配模型中的时间步采样分布对训练效率和生成质量的影响，并提出了一种两阶段的时间步采样课程调度方法；

**💡 创新点**

创新点在于将时间步采样视为可动态调整的学习课程，先用中间偏置的Logit-Normal分布快速学习结构，再切换到均匀分布细化边界，从而突破了传统单一采样策略的速度-质量权衡；

**🔧 技术方法**

主要技术包括Conditional Flow Matching训练目标、Logit-Normal与Beta分布的时间步采样、两阶段课程切换以及Fréchet Inception Distance（FID）评估；

**📊 数据集**

实验使用CIFAR-10图像数据集进行无条件图像生成；

**📈 对比分析**

与静态采样基线（Uniform、Logit-Normal、Beta）比较，课程采样在相同训练步骤下实现了16.4% FID提升（从3.85降至3.22），并且最佳性能提前33%（从150k降至100k迭代）；

**⚠️ 局限性**

局限性包括仅在CIFAR-10上验证，课程切换时间需要手动调参，且对更复杂高分辨率数据或不同模型结构的通用性尚未测试。

---

## 145. Seeing the Trees for the Forest: Leveraging Tree-Shaped Substructures in Property Graphs

**arXiv ID:** 2603.12476 | [PDF](https://arxiv.org/pdf/2603.12476v1)

**作者:** Daniel Aarao Reis Arturi `[一作]` (McGill University), Stefanie Scherzinger `[通讯]` (University of Passau)

**通讯引用:** 896 | [OpenAlex ID](https://openalex.org/A5053676208)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

分析属性图中普遍存在的树形子结构，并在 Neo4j、Kuzu、Apache AGE 上实现 PrePost 与 Dewey 结构索引，进行实验验证其对路径查询的加速效果；提出树形结构识别、schema 定义、索引维护及查询重写等研究方向。

**💡 创新点**

首次将 XML 领域的树形结构索引方法迁移到属性图数据库，系统化利用树形子结构来优化查询；提出从模型、schema 设计到执行层面全流程处理树形数据的框架。

**🔧 技术方法**

使用 PrePost 与 Dewey 结构索引；在 Neo4j、Kuzu、Apache AGE 上实现索引属性并构建传统索引；采用 Cypher（Neo4j、Kuzu）和 SQL（AGE）重写三类树形查询（Q_desc、Q_leaf、Q_a&d）。

**📊 数据集**

语法生成的树和森林（WT1/2/3、DT、TF）以及 LDBC SNB 社交网络基准（SF1）中的 Post/Comment、Place、TagClass 子森林。

**📈 对比分析**

通过对比基线查询和使用结构索引后的查询，在 Neo4j 上未见明显加速；在 Kuzu 上大多数树形查询实现 1.4–33 倍加速；在 Apache AGE 上提升可达 10³ 倍，几乎所有查询都不出现慢速；实验表明结构索引对基于关系后端的 GDBMS 能显著提升性能。

**⚠️ 局限性**

仅支持单标签树，未覆盖多标签/多边缘类型树；索引更新成本高，尤其是 PrePost；Neo4j 内部未充分利用结构索引；缺乏完整的查询重写、成本估计与 schema 执行机制，导致在复杂查询或大规模树上仍有瓶颈。

---

## 146. A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**arXiv ID:** 2603.12998 | [PDF](https://arxiv.org/pdf/2603.12998v1)

**作者:** Tangzheng Lian `[一作]` (King's College London), Oya Celiktutan `[通讯]` (King's College London)

**通讯引用:** 2175 | [OpenAlex ID](https://openalex.org/A5059459055)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练自由、数据自由的 Vision‑Language 模型去偏方法，利用闭式解在跨模态空间实现去偏，同时保持模型效用。

**💡 创新点**

创新点在于在跨模态空间中构造属性子空间并求解闭式最优去偏向量，实现 Pareto 最优公平且有理论保证的效用损失。

**🔧 技术方法**

使用 LLM 引导的群组原型构造、属性子空间投影、闭式优化等技术。

**📊 数据集**

使用了 CelebA、FACET、Flickr30K、COCO 等人像数据集进行评估。

**📈 对比分析**

与现有方法相比，在零样本分类、检索和生成任务上在多项公平指标上均更优，同时保持或提升 F1/Recall@K 等性能。

**⚠️ 局限性**

局限在于仅在 VLM 的交叉模态表示空间内保证效用，未直接保证任务特定指标；未扩展至生成器的解码器；敏感属性覆盖有限。

---

## 147. Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization

**arXiv ID:** 2603.12933 | [PDF](https://arxiv.org/pdf/2603.12933v1)

**作者:** Xudong Wang `[一作]` (Kyung Hee University), Hengtao Shen `[通讯]` (Tongji University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5000395470)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于语义感知的多代理系统路由框架 AMRO‑S，结合小语言模型意图推断、任务专属信息素专家和质量门控异步更新，实现高效可解释的路径搜索。

**💡 创新点**

创新点包括：① 将路由建模为语义条件路径搜索；② 采用监督微调的小语言模型进行低成本意图预测；③ 通过任务专属信息素专家实现任务隔离，减少跨任务干扰；④ 采用质量门控异步更新机制，保持低延迟的持续优化；⑤ 生成可视化信息素模式，提升路由透明度。

**🔧 技术方法**

使用技术包括：小语言模型 (LLM) 进行语义推理、Ant Colony Optimization (ACO) 信息素更新、质量门控异步学习、分层有向图建模、监督微调 (SFT)、多任务学习与查询条件融合。

**📊 数据集**

主要数据集为五个公开基准：MMLU、GSM8K、MATH、HumanEval、MBPP；并在 MacNet、GPTSwarm、HEnRY 等现有多代理框架中进行集成评估。

**📈 对比分析**

对比方法涵盖单模型基线、链式推理、多代理无路由、现有路由方法（RouteLLM、RouterDC、MasRouter）以及框架集成版本。AMRO‑S 在所有基准上平均得分 87.83，较最强对比基线提升约 1.9 分；在 1000 并发下实现 4.7× 速度提升，且在高并发环境下精度保持稳定，成本相对下降。

**⚠️ 局限性**

局限性包括：对信息素更新速度和极端动态环境适应性有限；依赖小语言模型的意图识别精度，误判会影响路由；异步更新需要额外队列与存储；实验仅在公开基准和实验室环境验证，真实生产部署的可扩展性与鲁棒性仍待进一步验证。

---

## 148. RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**arXiv ID:** 2603.12685 | [PDF](https://arxiv.org/pdf/2603.12685v1)

**作者:** Bin Wan `[一作]` (Shandong University), Sam Kwong `[通讯]` (Lingnan University)

**通讯引用:** 34701 | [OpenAlex ID](https://openalex.org/A5008386708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了RSONet网络用于RGB‑T显著目标检测，包含区域引导与显著性生成两阶段结构。

**💡 创新点**

创新点在于利用三分支编码器-解码器生成指导图并通过相似度判断主模态；提出选择性优化（SO）融合模块以及密集细节增强（DDE）和互相交互语义（MIS）模块，以自适应补偿两模态不一致并提升细节与位置信息。

**🔧 技术方法**

采用Swin Transformer骨干，Context Interaction (CI) 与 Spatial-aware Fusion (SF) 模块进行特征交互；Selective Optimization (SO)、Dense Detail Enhancement (DDE)、Mutual Interaction Semantic (MIS) 结合多尺度卷积、视觉状态空间块、通道/空间注意力；RMSprop优化，BCE+IoU+FM 损失。

**📊 数据集**

使用公开的RGB‑T数据集VT5000、VT1000和VT821进行训练与评估。

**📈 对比分析**

与27种state‑of‑the‑art方法在M、Fβ、Sα、Eξ四指标上对比，RSONet在VT5000、VT1000、VT821上均获得最优或接近最优结果，提升幅度约1%–4%。

**⚠️ 局限性**

局限性：两阶段设计导致推理速度相对较慢；在极小或低质量目标场景下检测效果受限。

---

## 149. Streaming REST APIs for Large Financial Transaction Exports from Relational Databases

**arXiv ID:** 2603.12566 | [PDF](https://arxiv.org/pdf/2603.12566v1)

**作者:** Abhiram Kandiraju `[一作]` `[通讯]` (Capital One), Abhiram Kandiraju (Capital One)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个基于流式的 REST API，能够从关系数据库增量检索交易记录并直接写入 HTTP 响应流，实现大规模交易导出。

**💡 创新点**

将数据库前向只读游标与 HTTP 分块传输相结合，构建无缓冲、低内存占用的实时导出流水线，并支持多种金融导出格式。

**🔧 技术方法**

使用 Java JAX‑RS 框架、JDBC 前向只读游标、流式序列化、HTTP 分块传输，以及 CSV、OFX、QFX、QBO 等导出编码。

**📊 数据集**

采用合成的交易记录数据集，规模从 10 万行到 100 万行进行实验。

**📈 对比分析**

通过与传统缓冲式导出对比，测量首字节响应时间、峰值内存占用和总导出时间；结果显示流式导出在首字节响应即时、内存占用显著降低且在 100 万行时保持低峰值内存。

**⚠️ 局限性**

限制包括对数据库查询性能、连接池耗尽、长连接超时、客户端错误处理和文件完整性验证等方面的挑战。

---

## 150. FDeID-Toolbox: Face De-Identification Toolbox

**arXiv ID:** 2603.13121 | [PDF](https://arxiv.org/pdf/2603.13121v1)

**作者:** Hui Wei `[一作]` (University of Oulu), Guoying Zhao `[通讯]` (University of Oulu)

**通讯引用:** 29288 | [OpenAlex ID](https://openalex.org/A5082301986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套统一、可复现的面部去识别（FDeID）工具箱，整合17种传统、对抗及生成模型方法，提供统一接口、标准数据加载、可配置的推理管道与三维度评估（隐私、效用、视觉质量）。

**💡 创新点**

通过模块化架构实现方法统一调用与评估协议标准化，并首次支持多范式方法（传统、对抗、生成）在同一平台下进行跨方法、多属性的系统比较与集成策略。

**🔧 技术方法**

使用RetinaFace做面部对齐，k‑Same族、MI‑FGSM/PGD/TI‑DIM/Chameleon等对抗攻击，Adv‑Makeup/CIAGAN/AMT‑GAN/DeID‑rPPG/G²Face/WeakenDiff等生成模型，结合PyTorch、YAML配置和GAN/扩散框架。

**📊 数据集**

采用CelebA‑HQ、LFW、AgeDB、FairFace、AffectNet、PURE等六个主流基准数据集进行统一加载和评估。

**📈 对比分析**

在统一的YAML配置与相同硬件环境下，使用PSR、TAR、六项属性（年龄、性别、种族、表情、标记点、rPPG）以及PSNR/SSIM/FID/NIQE/LPIPS等指标对方法进行比较；结果表明无单方法三维度均优，传统k‑Same隐私最佳但效用差，对抗方法效用好但隐私弱，生成方法折中；方法集成可实现更佳权衡。

**⚠️ 局限性**

局限性包括仅针对静态图像，未考虑视频时序一致性；公平性分析对年龄、性别、种族等子组不足；双重用途风险未彻底解决。

---

## 151. Test-time RL alignment exposes task familiarity artifacts in LLM benchmarks

**arXiv ID:** 2603.12875 | [PDF](https://arxiv.org/pdf/2603.12875v1)

**作者:** Kun Wang `[一作]` (Technical University of Munich), Reinhard Heckel `[通讯]` (Technical University of Munich)

**通讯引用:** 3383 | [OpenAlex ID](https://openalex.org/A5003606899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出无训练集的测试时强化学习对齐方法（TTRA），实现任务熟悉度补偿

**💡 创新点**

通过一次性强化学习与多数投票奖励的两阶段融合，解决SFT对训练集依赖并消除任务熟悉度伪影

**🔧 技术方法**

采用一次性RL对齐与测试时RL（GRPO）+多数投票奖励

**📊 数据集**

GSM8K、MathQA、GSM-plus-mini、MATH500、AIME、OlympiadBench、MIMIC-CDM-FI

**📈 对比分析**

与直接评估和SFT对齐比较，TTRA在维持排名一致性的同时显著提升基线模型在推理与临床决策任务的准确率，缩小了细调模型与基线的差距

**⚠️ 局限性**

仅适用于多数投票可获取奖励的任务，代码生成等任务效果差；仅在全信息MIMIC-CDM上测试，未覆盖对话式诊断

---

## 152. AEGIS: No Tool Call Left Unchecked -- A Pre-Execution Firewall and Audit Layer for AI Agents

**arXiv ID:** 2603.12621 | [PDF](https://arxiv.org/pdf/2603.12621v1)

**作者:** Aojie Yuan `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3414 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个框架无关的预执行防火墙与审计层AEGIS，用于拦截 AI 代理的工具调用并在实际执行前进行风险扫描、策略验证与人工审批。

**💡 创新点**

在工具调用路径中引入轻量级三阶段管道（深度字符串提取、内容优先风险扫描、可组合策略验证），实现预执行拦截；采用 Ed25519+SHA256 链式签名的防篡改审计；支持 14 个多语言代理框架并提供实时人机交互审批。

**🔧 技术方法**

递归字符串提取、正则检测模式、JSON Schema 策略编译（AJV）、Ed25519 签名+SHA-256 链、HTTP/REST Gateway、SDK 插桩、Compliance Cockpit Web UI、率限制器等技术。

**📊 数据集**

48 个手工挑选的攻击实例（覆盖 7 类 OWASP 等技术）以及 500 条正常工具调用样本，用于误报评估；在 1,000 次连续拦截测试中评估延迟。

**📈 对比分析**

与 AgentDojo、ToolEmu 等评估平台对比，AEGIS 在运行时拦截所有 48 攻击实例；误报率 1.2%；平均拦截延迟 8.3 ms（P95 14.7 ms、P99 23.1 ms），相对 LLM 推理延迟可忽略。

**⚠️ 局限性**

仅检测已知模式，可能遗漏新型攻击；规则+策略基础无法捕捉未知变体；评估规模受限于手工构造的攻击集；不防止 SDK 外部直接调用工具。

---

## 153. CalliMaster: Mastering Page-level Chinese Calligraphy via Layout-guided Spatial Planning

**arXiv ID:** 2603.12482 | [PDF](https://arxiv.org/pdf/2603.12482v1)

**作者:** Tianshuo Xu `[一作]` (Hong Kong University of Science and Technology), Ying-cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 CalliMaster 的多模态扩散变换器框架，先预测字符布局再生成高质量中文书法图像，实现页面级书法的端到端控制与编辑。

**💡 创新点**

核心创新在于将宏观布局规划与微观笔墨细节解耦，采用粗到细的 Text→Layout→Image 生成管线；同时通过独立噪声调度、模态感知 AdaLN 与因果注意力掩码，使布局成为几何提示，支持语义再规划、修复与鉴定。

**🔧 技术方法**

技术上融合了流匹配（flow‑matching）、Diffusion Forcing 的多时钟扩散、复合 3D Rotary Position Embedding、以及自监督的多目标训练（盒子生成、内容合成、联合修复与无条件预训练）。

**📊 数据集**

使用了多来源的中文书法图像与文本对齐数据，包含约 1 万页手写/印刷书法，配合人工标注的字符框坐标和风格描述进行训练；数据集未公开，论文中描述为“多语种书法数据集”。

**📈 对比分析**

在多项指标上超过现有方法：CLIP‑Score 达 0.9663、GPT‑Score 4.1，字符识别准确率与真实样本相近（约 34%），并在用户研究中获得最高 4.68 分的整体满意度；对比实验还展示了低 DRS 评分、优越的空间韵律与连笔恢复效果。

**⚠️ 局限性**

局限性包括：对极端书法风格或非常规排版的泛化仍有限；字符识别准确率仍低于标准字体；需要大规模 GPU 训练资源，推理速度受限于 25 步 ODE 解算器；模型对极细笔触的细节捕捉仍有提升空间。

---

## 154. NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval

**arXiv ID:** 2603.12824 | [PDF](https://arxiv.org/pdf/2603.12824v1)

**作者:** Zhuchenyang Liu `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**通讯引用:** 3927 | [OpenAlex ID](https://openalex.org/A5069437467)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了异构跨模态知识蒸馏框架，将大型视觉语言模型（Qwen3‑VL‑Embedding‑2B）作为教师，仅用文本蒸馏得到轻量级学生模型（DistilBERT 69M）实现视觉文档检索，仅需在查询端进行编码；

**💡 创新点**

创新点在于：①采用点对点余弦对齐损失，完全去除文档负样本与排名损失，仅用教师查询嵌入即可训练；②证明在高质量教师空间中直接空间对齐优于传统排名/对比学习；③通过仅查询翻译数据实现跨语言性能提升，解决跨语言瓶颈；

**🔧 技术方法**

使用的技术包括：视觉语言模型Qwen3‑VL‑Embedding‑2B作为冻结教师；文本仅蒸馏学生（DistilBERT、BERT、ModernBERT）+两层MLP投影；点对点余弦对齐损失（align）、软标签排名损失（rank）与InfoNCE对比；Helsinki‑NLP Opus‑MT进行多语言查询翻译增强；

**📊 数据集**

训练数据：726K查询-文档图像对（VisRAG‑Synthetic、ColPali、VisRAG‑InDomain、VDR‑Multilingual）；评估数据：ViDoRe 22 视觉文档检索任务（v1 10 数据集、v2 4 数据集、v3 8 数据集）；

**📈 对比分析**

对比10个基线（多向量VLM、单向量VLM、视觉本地模型），在ViDoRe v2/v3上，文本仅学生（-S、-S‑Multi）均超过ColPali（3B）和DSE‑Qwen2（2B）；-S‑Multi在v3达到46.5 NDCG@5，保留教师95.1%性能，参数量32×更少，CPU查询延迟约50×，索引存储仅8.2 GB；

**⚠️ 局限性**

局限性：性能受限于教师文档嵌入质量；离线阶段仍需使用大模型对所有文档图像编码，未降低索引成本；多语言翻译质量可能引入语义偏差，尤其在专业领域；

---

## 155. Continual Learning in Large Language Models: Methods, Challenges, and Opportunities

**arXiv ID:** 2603.12658 | [PDF](https://arxiv.org/pdf/2603.12658v1)

**作者:** Hongyang Chen `[一作]` (Zhejiang Lab), Xuemin Lin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了大型语言模型（LLM）的持续学习方法，按照持续预训练、持续微调和持续对齐三大训练阶段，对相关技术、评估指标和基准进行了系统梳理。

**💡 创新点**

创新点在于将传统持续学习方法细化为针对LLM的三阶段分类，并进一步按灾难性遗忘缓解机制细分，明确了LLM持续学习与传统机器学习在规模、参数效率和新兴能力方面的区别。

**🔧 技术方法**

采用了文献综述、方法分类、对比分析、评估指标定义（AP、F.Ra、FWT、BWT）和基准汇总等技术手段，构建了持续学习的全景框架。

**📊 数据集**

主要参考了公开的持续学习基准（MMLU、GSM8K、TRACE、CITB、InvariantLlama、UpdatedLlama、NewLlama、StreamBench 等）以及各类领域专用数据集，但未在本文中自行实验使用新数据集。

**📈 对比分析**

通过对比已发表方法在上述基准上的表现，指出虽然部分方法在特定任务上取得显著提升，但整体仍面临遗忘率高、知识迁移有限、跨任务泛化不足等问题；文中提供了定量指标对比表格和性能趋势概览。

**⚠️ 局限性**

局限性包括：缺乏对多模态持续学习的深入讨论、未覆盖所有 RL‑based 对齐方法、对在线持续学习场景的研究不足、数据隐私与可复现性挑战，以及对半参数化和增量学习的实证验证仍待补充。

---

## 156. Efficient Reasoning with Balanced Thinking

**arXiv ID:** 2603.12372 | [PDF](https://arxiv.org/pdf/2603.12372v1)

**作者:** Yulin Li `[一作]` (Harbin Institute of Technology), Zhuotao Tian `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2388 | [OpenAlex ID](https://openalex.org/A5038086218)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个训练-free的ReBalance框架，用置信度与隐藏状态的动态控制来平衡大型推理模型的过度思考与欠思考，从而提升推理效率与准确性。

**💡 创新点**

创新点包括：①将置信度及其方差作为连续指标识别过度与欠思考；②通过聚合隐藏状态生成从过度思考到欠思考的导向向量；③构造动态控制函数，根据实时置信度调节导向向量的强度和方向，实现自适应推理平衡。

**🔧 技术方法**

主要技术手段为：置信度与置信度方差计算、隐藏状态聚类提取原型、导向向量（steering vector）的构造与调节、动态控制函数与门控机制，以及无训练的推理过程。

**📊 数据集**

使用的数据集包括：数学推理集（MATH‑500、AIME24/25、AMC23、GSM8K、OlympiadBench）、科学推理GPQA Diamond、常识推理StrategyQA、编程推理LiveCodeBench，以及多模型（0.5B–32B）和跨域任务的评估。

**📈 对比分析**

与多种现有过度思考抑制方法（如NoThinking、CoD、DEER、Dynasor‑CoT、SEAL、Manifold Steering）以及外部提前退出方法（TrimR、FlashThink）进行对比；在九大基准上，ReBalance在Pass@1上提升最多7个百分点、Token数减少最多52%，并保持或提升准确率，展示了强泛化与效率提升。

**⚠️ 局限性**

局限性：①需要在小规模“seen”数据上提取隐藏状态原型和控制表面，若该数据与目标域差异较大可能影响效果；②动态窗口与阈值设置仍需经验，可能在极短或极难的推理任务上不够稳健；③对已优化的推理模型（如Distilled）对提示抑制更敏感，导致某些情形下性能不稳定。

---

## 157. Using a Human-AI Teaming Approach to Create and Curate Scientific Datasets with the SCILIRE System

**arXiv ID:** 2603.12638 | [PDF](https://arxiv.org/pdf/2603.12638v1)

**作者:** Necva Bölücü `[一作]` (CSIRO), Stephen Wan `[通讯]` (CSIRO)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并实现了SciLire系统，利用LLM和人机协作的迭代流程，自动从科学论文中提取结构化数据并生成可供使用的表格。

**💡 创新点**

创新点在于引入动态采样的少样本提示（ICL），基于用户已纠正记录的检索式选择示例，同时结合多源LLM输出的对齐与验证工具，实现高效且可追溯的数据抽取。

**🔧 技术方法**

主要技术包括GROBID/Tika文本与表格解析、LLM提示生成与few-shot推理、Hungarian算法对齐、BM25动态采样、模糊匹配验证、基于GPT‑5的生成模型。

**📊 数据集**

实验使用了18个公开科学数据集，覆盖材料科学、化学、医学、环境等五大领域，数据集来源包括BRENDA、MPEA、PPE等。

**📈 对比分析**

通过与Elicit、SciSpace的对比，SciLire在10份样本的F1得分平均从≈22提升到≈23，整体表现优于零射手工具，说明动态ICL显著提升抽取质量。

**⚠️ 局限性**

限制方面为整体F1仍偏低（最高仅约68%），系统仍需人工校正；对非表格信息支持不足；以及在极大规模批量处理时性能与可扩展性需进一步评估。

---

## 158. 98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router

**arXiv ID:** 2603.12646 | [PDF](https://arxiv.org/pdf/2603.12646v1)

**作者:** Xunzhuo Liu `[一作]` (vLLM Semantic Router Project), Huamin Chen `[通讯]` (Red Hat)

**通讯引用:** 2235 | [OpenAlex ID](https://openalex.org/A5101790571)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型服务中的安全与意图分类，本文实现了三阶段低延迟语义路由：①在AMD GPU上自研CK Flash Attention实现O(n)注意力；②无神经网络的经典NLP压缩将任意长度提示压缩至512令牌；③基于Envoy ext‑proc的近流式零拷贝JSON处理，消除序列化开销。

**💡 创新点**

创新点在于：①首次将Flash Attention集成至ONNX Runtime ROCm端，解决GPU共享下的O(n²)内存瓶颈；②提出多信号组合（TextRank+位置权重+TF‑IDF+新颖度）的无模型压缩方案；③设计自适应块处理器在不完整请求时实现零拷贝流式处理。

**🔧 技术方法**

技术组合包括：AMD Instinct MI300X GPU、ONNX Runtime + ROCm + 自研CK Flash Attention算子、Python ONNX图重写、Rust GPU绑定、Go Envoy ext‑proc、gjson/sjson JSON操作、文本分句、TF‑IDF、TextRank、位置权重、相似度排序。

**📊 数据集**

主要实验数据集为合成的多长度（500/2000/8000/16000）提示，嵌入木马前缀、PII和领域文本；以及384篇维基百科文章（8个领域、4种长度、12信号位置组合）用于离线准确率评估。

**📈 对比分析**

与CPU + SDPA、Candle CPU以及未压缩GPU基线对比，E2E延迟从4.918 s降至50 ms（98×），在16K令牌下实现108 ms；GPU内存占用<800 MB，支持在同一GPU上共存LLM推理。

**⚠️ 局限性**

局限性包括：①压缩使用字符长度近似分词，未完全匹配模型分词；②累积模式下仍需在CPU端缓冲完整请求体；③CK Flash Attention仅适用于AMD ROCm，NVIDIA等平台需使用已有FA实现。

---

## 159. HyGra: Accelerating Network-State Simulation for LLM Training in DCNs via Adaptive Packet-Flow Granularity

**arXiv ID:** 2603.12671 | [PDF](https://arxiv.org/pdf/2603.12671v1)

**作者:** Wenyi Wang `[一作]` (Nanjing University of Posts and Telecommunications), Fu Xiao `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 12499 | [OpenAlex ID](https://openalex.org/A5053690968)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在大型语言模型训练时动态切换包级和流级模拟粒度的单机网络状态模拟器HyGra，适用于数据中心网络（DCN），并保持高精度与高速的仿真性能。

**💡 创新点**

创新点在于：①基于实时带宽与队列深度的稳态识别，自动决定何时切换到流级粒度；②利用Transformer网络实现从流级恢复到包级的近乎无损状态转移；③兼容现有PNS（如ns-3、SystemC）并可在无专用硬件的单机环境中运行。

**🔧 技术方法**

主要技术包括：包级仿真（Packet-Level Simulation）、流级仿真（Flow-Level Simulation）、Max-Min公平带宽再分配、滑动窗口稳态检测、流到包的Transformer队列恢复、以及分层控制-执行架构。

**📊 数据集**

使用代表性商业LLM工作负载：ChatGPT（175B参数）、DeepSeek‑V3（671B参数）和Qwen2.5（72B参数），在多种并行策略（PP、TP、DP及其混合）下进行评估。

**📈 对比分析**

与纯包级仿真（PLS）和纯流级仿真（FLS）对比，HyGra在单一并行策略下可实现最高15.4×的仿真速度提升，混合策略下可达7.8×，同时保持JCT、FCT误差低于5%，吞吐量误差小于4%。

**⚠️ 局限性**

局限性包括：当前实现仅单核加速，未利用多核并行；Transformer恢复模型需要离线训练并可能对极端突发流量的泛化不足；未考虑计算资源模拟，无法完成完整的网络‑计算协同仿真。

---

## 160. Lyapunov Stable Graph Neural Flow

**arXiv ID:** 2603.12557 | [PDF](https://arxiv.org/pdf/2603.12557v1)

**作者:** Haoyu Chu `[一作]` (China University of Mining and Technology), Qiyu Kang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6875 | [OpenAlex ID](https://openalex.org/A5101743182)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Lyapunov 稳定的图神经流（IL‑GNNs 与 FL‑GNNs），通过学习 Lyapunov 函数并投影到稳定子空间，保证 GNN 动态在整数阶和分数阶下的稳定性，从而提升对图结构和特征对抗攻击的鲁棒性。

**💡 创新点**

首次将 Lyapunov 直接法与图神经流相结合，引入可学习的 Lyapunov 函数与投影机制，形成整数阶与分数阶两类稳定化模块，并加入平衡 equilibrium 的分类层，提供可证明的鲁棒性保证。

**🔧 技术方法**

使用的技术包括图神经流模型（GRAND、GBel、GCON 等）、分数阶微分方程、Caputo 递归导数、可学习的输入凸神经网络（ICNN）作为 Lyapunov 函数、投影正则化、平衡分离分类层，以及可选的对抗训练。

**📊 数据集**

在四个公开数据集上评估：Cora、Citeseer、Pubmed、Computers（节点/边/特征/类别数分别不同）。

**📈 对比分析**

与基准模型（GRAND、FL‑GRAND、GBel、FL‑GBel、GCON、FL‑GCON、HANG 等）在多种图注入、劫持、白盒攻击以及不同扰动幅度下进行比较，实验表明 IL‑GNN/FL‑GNN 在干净数据上相当或更好，且在攻击场景下准确率提升 10–30% 甚至更高，特别是在白盒和大扰动情况下表现突出；与对抗训练联合使用进一步提升鲁棒性。

**⚠️ 局限性**

局限性包括：训练与推理时间略高，尤其是分数阶版本；在 Pubmed 等部分任务的鲁棒性提升有限；对 β 参数和 Lyapunov 设计的超参数敏感；目前仅验证了节点分类任务，未扩展到图生成、链接预测等；理论证明依赖于模型满足 Lyapunov 条件，实际可实现性需进一步评估。

---

## 161. The RIGID Framework: Research-Integrated, Generative AI-Mediated Instructional Design

**arXiv ID:** 2603.12781 | [PDF](https://arxiv.org/pdf/2603.12781v1)

**作者:** Yerin Kwak `[一作]` (University of California, Berkeley), Zachary A. Pardos `[通讯]` (University of California, Berkeley)

**通讯引用:** 3623 | [OpenAlex ID](https://openalex.org/A5021273980)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出了RIGID框架，将学习科学研究系统化嵌入完整的指示设计生命周期，并利用生成式AI在分析、设计、实施、评估四个阶段进行知识整合与决策支持。

**💡 创新点**

创新点在于：①将学习科学的研究成果与指示设计的实践流程紧密结合；②引入生成式AI作为中介工具，自动合成多层情境信息，减少教师认知负担；③实现从微观课堂到宏观政策的多级整合。

**🔧 技术方法**

采用生成式大语言模型（LLM）结合自动提示工程（APE）、检索增强生成（RAG）等技术，用于生成教学材料、模拟学习者、实时数据分析与反馈。

**📊 数据集**

本文未提供具体实验数据集，主要以理论构建与示例演示为主；若有演示则使用教育平台日志、课程案例等非公开数据。

**📈 对比分析**

论文未进行实证比较或量化评估；若有示例则通过教师和学生的主观评估以及教学效果观察表明RIGID可降低工作量、提升设计质量，但缺乏系统性定量指标。

**⚠️ 局限性**

局限性包括：生成式AI易产生幻觉与偏见，需人工审核；人类专业判断仍不可替代；技术实现与平台集成成本较高；缺乏大规模实证验证。

---

## 162. A Partial-Exclusion Repair Scheme for MDS Codes

**arXiv ID:** 2603.12585 | [PDF](https://arxiv.org/pdf/2603.12585v1)

**作者:** Wei Zhao `[一作]` (Foshan University), Ximing Fu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5076731706)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种部分排除（PE）修复方案，用于标量最大距离可分（MDS）码，解决了传统修复方案在单节点修复时需要超指数的子分包化水平的问题。

**💡 创新点**

创新点在于引入了灵活性参数t，使得修复方案可以在不同的灵活性下进行优化，从而降低子分包化水平和修复带宽之间的权衡。

**🔧 技术方法**

使用了部分排除（PE）修复框架和Reed-Solomon（RS）码的两种通用构造。

**📊 数据集**

构造了(12,8)和(17,9)的RS码，分别在基域𝔽_2和𝔽_4上实现。

**📈 对比分析**

与传统修复方案相比，提出的构造在子分包化水平上显著降低，(12,8) RS码的子分包化水平为2310，远低于510510的已知下界，(17,9) RS码的子分包化水平为30，低于9699690的下界。

**⚠️ 局限性**

限制在于当前的子分包化水平下界仍需改进，且多节点故障场景下的PE修复框架仍需进一步研究。

---

## 163. LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**arXiv ID:** 2603.12647 | [PDF](https://arxiv.org/pdf/2603.12647v1)

**作者:** Ziyu Chen `[一作]` (Hefei Institutes of Physical Science), Chunmao Jiang `[通讯]` (Hefei Institutes of Physical Science)

**通讯引用:** 374 | [OpenAlex ID](https://openalex.org/A5086827129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向自动驾驶场景的鲁棒 LiDAR-反射率引导显著高斯抛射（LR-SGS）方法，实现了高质量几何、外观与反射率的统一重建与实时渲染。

**💡 创新点**

创新点包括：① 将 LiDAR 强度校准为近似光照不变的反射率并作为额外属性通道；② 基于 LiDAR 结构与反射率特征点初始化显著高斯（Edge/Planar），并通过显著变换与改进密度控制动态调整显著性；③ 在训练中加入跨模态梯度方向与幅值一致性损失，提升边界与材质一致性。

**🔧 技术方法**

主要技术包括 3D 高斯抛射（3DGS）、基于 LiDAR 点云特征提取、光照不变反射率估计、显著高斯结构化表示、密度控制与显著转换、联合光度、LiDAR 与联合损失、前向渲染与场景图优化。

**📊 数据集**

在 Waymo Open Dataset 的 Dense Traffic、High-Speed、Complex Lighting、Static 四类场景上进行实验，使用 RGB 相机（5摄像头）和 64 阵 LiDAR 数据。

**📈 对比分析**

与 3DGS、DeformGS、PVG、StreetGS、OmniRe 等基线对比，LR-SGS 在 PSNR、SSIM、LPIPS、RMSE 等指标均实现最高或接近最高，且显著减少高斯数量与训练时长；在 Complex Lighting 场景中 PSNR 超过 OmniRe 1.18 dB。

**⚠️ 局限性**

局限性：仍依赖相对稠密的 LiDAR 观测，稀疏或缺失点云时显著高斯初始化效果受限；在极端光照与高速运动下对动态物体的细节捕捉仍有提升空间。

---

## 164. Reweighted information inequalities

**arXiv ID:** 2603.13135 | [PDF](https://arxiv.org/pdf/2603.13135v1)

**作者:** Jonathan Niles-Weed `[一作]` (New York University), Jonathan Niles-Weed `[通讯]` (New York University)

**通讯引用:** 9084 | [OpenAlex ID](https://openalex.org/A5031031216)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了混合分布在满足各成分具备对数Sobolev或传输信息不等式时的整体性质，证明了对混合权重重新加权后仍能保持相应的不等式，并给出了相应的 Fisher 信息、相对熵和 Wasserstein 距离的上界。

**💡 创新点**

创新点在于：① 提出了“重加权”不等式的概念，能够在非对数凸分布下通过成分分解来获得有意义的收敛界；② 将混合分布的混合权重与动态过程的局部几何关联起来，解释了 Langevin Monte Carlo 在多模态分布下的元稳定性与快速收敛；③ 通过对偶性和大偏差理论给出了传输信息不等式的动态表述。

**🔧 技术方法**

主要技术包括：对数Sobolev和传输信息不等式的复合凸性证明；使用 Dirichlet 形式和 Fisher 信息的定义；对混合分布进行概率标记化解释；利用 Gronwall 不等式、Kantorovich 对偶以及大偏差/集中不等式进行分析。

**📊 数据集**

本工作完全是理论分析，不涉及具体数据集。

**📈 对比分析**

由于没有实验验证，文章不提供与现有算法的数值比较；但在理论层面，给出了相对于原始混合分布的 Fisher 信息、相对熵和 Wasserstein 距离的具体上界，并说明在成分满足良好常数时可获得指数级快速收敛。

**⚠️ 局限性**

限制包括：① 需要混合分布的各成分满足对数Sobolev或传输信息不等式；② 结果主要关注重加权后的混合，而非原始混合本身；③ 对于高维或成分数量极大时，重新加权所需的常数可能仍然呈指数增长，未能完全消除对维度或模态数的依赖。

---

## 165. HMS-BERT: Hybrid Multi-Task Self-Training for Multilingual and Multi-Label Cyberbullying Detection

**arXiv ID:** 2603.12920 | [PDF](https://arxiv.org/pdf/2603.12920v1)

**作者:** Zixin Feng `[一作]` (Xi'an Jiaotong Liverpool University), Md Maruf Hasan `[通讯]` (Xi'an Jiaotong Liverpool University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种混合多任务自训练框架 HMS‑BERT，能够同时处理英文和中文的多语言、多标签网络欺凌检测；

**💡 创新点**

创新点包括：①将多语言 BERT 与手工语言特征融合；②联合多标签子类分类与三分类主任务的双分支设计；③采用置信度阈值的迭代自训练（伪标签）来弥补低资源语言的数据稀缺；

**🔧 技术方法**

使用技术包括：多语言 BERT（mBERT）作为主干网络；特征融合层与双分支全连接分类头；置信度阈值伪标签的自训练循环；联合二进制交叉熵与类别交叉熵的加权损失；

**📊 数据集**

采用公开数据集：HateXplain（英文多标签+三类主标签）、Cyberbullying Classification（英文）、SCCD_User 与 SCCD_Comment（中文）等四个数据集；

**📈 对比分析**

与 TF‑IDF+LR、XLM‑RoBERTa、DistilBERT、LaBSE 等基线在三类主任务、细粒度多标签和回归任务进行对比。HMS‑BERT 在多标签 F1 为 0.9847、回归 MAE 为 0.0234、主任务准确率 0.6775 等指标上表现最优；

**⚠️ 局限性**

局限性包括：仅支持中英两种语言，跨语言扩展受限；伪标签噪声控制不够完善；缺乏对图像、音频等多模态的支持；对极低资源语言的泛化能力仍有限。

---

## 166. RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization

**arXiv ID:** 2603.12639 | [PDF](https://arxiv.org/pdf/2603.12639v1)

**作者:** Ruicheng Zhang `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 10850 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种对称双塔4D世界模型RoboStereo，并基于该模型构建了统一的策略优化框架（测试时策略增强TTPA、模仿演化策略学习IEPL和开放探索策略学习OEPL），实现了从推理到训练的端到端安全、高效的机器人策略改进；

**💡 创新点**

①双塔Diffusion Transformer通过双向交叉注意力实现RGB视频与3D点图的互补增强，显著提升几何一致性与物理真实性；②引入时序动作调制的timestep embedding实现帧级动作控制；③首次将高保真EWMs用于测试时验证与训练时演化，形成完整的统一优化框架；

**🔧 技术方法**

Diffusion Transformer（DiT）、4D高斯散点渲染头、双向交叉注意力、动作调制的AdaLN、GRPO、DPO、LPIPS感知奖励、VideoMAE视频理解评估等技术；

**📊 数据集**

Bridge V2（真实机器人操纵视频）用于训练4D世界模型；MimicGen模拟数据用于微调与策略优化；以及在MimicGen上进行的多任务评估；

**📈 对比分析**

与八大现有EWMs（GigaWorld、Genie、MIND‑V、RoboMaster、Vidar、Cosmos 2.5、WoW、IRA‑Sim）在16项视频质量与物理一致性指标上对比，RoboStereo在大多数指标上名列前茅；在MimicGen仿真基准任务中，TTPA提升约56%成功率，IEPL与OEPL分别提升55%和72%，三者结合可实现约97%的相对提升；

**⚠️ 局限性**

主要局限包括：仍以仿真环境为主，未在真实机器人上广泛验证；对极端高复杂度交互或极端视觉变化的鲁棒性待进一步验证；模型和训练过程计算成本较高，需大规模GPU资源；

---

## 167. Prompt Injection as Role Confusion

**arXiv ID:** 2603.12277 | [PDF](https://arxiv.org/pdf/2603.12277v1)

**作者:** Charles Ye `[一作]` (Independent Researcher), Dylan Hadfield-Menell `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1465 | [OpenAlex ID](https://openalex.org/A5076757561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM中提示注入攻击背后的机制，提出角色感知探针并展示其对攻击成功的预测能力。

**💡 创新点**

首次将角色感知误差框架为“角色混淆”，通过线性探针量化模型内部角色表示，并证明其决定注入成功。

**🔧 技术方法**

使用线性探针、角色标签化、CoT伪造攻击以及风格与声明对照实验等技术。

**📊 数据集**

使用C4、Dolma3进行探针训练，Oasst1、ToxicChat用于对话评估，StrongREJECT、Agent Exfiltration等基准数据集进行攻击实验。

**📈 对比分析**

在六大前沿模型上，CoT伪造攻击平均成功率为60%–70%；探针预测与实际攻击成功呈单调正相关，验证了其有效性。

**⚠️ 局限性**

探针假设角色位于线性子空间，实验仅覆盖少数模型，未验证更大模型的表现，也未系统评估对抗性防御的改进效果。

---

## 168. Long-form RewardBench: Evaluating Reward Models for Long-form Generation

**arXiv ID:** 2603.12963 | [PDF](https://arxiv.org/pdf/2603.12963v1)

**作者:** Hui Huang `[一作]` (Harbin Institute of Technology), Tiejun Zhao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8635 | [OpenAlex ID](https://openalex.org/A5100564229)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了 Long-form RewardBench 这一首创的长文本奖励模型评估基准，涵盖 QA、RAG、Chat、Writing、Reasoning 等五大子任务；

**💡 创新点**

创新点在于：①首次针对长文本生成设计奖励模型评测；②通过多阶段 LLM‑as‑Judge 及规则化注释构造高质量偏好对；③设计 Long‑form Needle‑in‑a‑Haystack 测试探究错误位置与文本长度对奖励模型的影响；

**🔧 技术方法**

采用多模型生成、自动化 LLM 判别、加权点数、Best‑of‑N 评估、以及生成式与分类式奖励模型两种架构；

**📊 数据集**

使用的主要数据集包括 QuoraQA、RAGBench、WildChat、LongWriter‑6k、DeltaBench 以及多达 15 种开源与闭源 LLM 生成的长文本；

**📈 对比分析**

通过对 20+ 奖励模型（序列分类器与生成式模型）进行 BoN 评估和 Needle‑in‑a‑Haystack 实验，发现序列分类器整体表现更优，生成式模型在推理子任务上优势明显，但在评分模式下性能低于选择模式；

**⚠️ 局限性**

局限性包括：生成式奖励模型在长文本上易受长度与错误位置影响，Fine‑tuned 生成式判别器对 BoN 采样适应性差，整体长文本奖励模型仍显不足，需进一步优化训练数据与模型架构。

---

## 169. DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification

**arXiv ID:** 2603.12905 | [PDF](https://arxiv.org/pdf/2603.12905v1)

**作者:** Joana Reuss `[一作]` (Technical University of Munich), Marco Körner `[通讯]` (Technical University of Munich)

**通讯引用:** 3250 | [OpenAlex ID](https://openalex.org/A5079258186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于Dirichlet先验的损失调整方法，用于在少样本情况下改进欧洲多国的农作物细粒度分类任务。

**💡 创新点**

创新点在于通过在训练时随机采样Dirichlet分布并对logits进行加权，显式解决训练-测试分布差异与类别不平衡问题。

**🔧 技术方法**

技术包括Dirichlet采样、logit加权、交叉熵/焦点损失、Transformer时序模型与label‑confidence机制。

**📊 数据集**

使用了扩展版EuroSAT数据集（含2020年新增数据）以及各国子集（比利时、爱沙尼亚、拉脱维亚等）进行实验。

**📈 对比分析**

通过与基线模型（无先验调整）在Cohen's κ、准确率和F1得分等指标下对比，Dirichlet先验方法在多数少样本场景下均表现出显著提升，尤其在较大先验偏移时效果更为突出。

**⚠️ 局限性**

局限性包括在极端类别不平衡或先验偏移不足时可能出现性能下降，以及对超参数的敏感性与在非欧陆地区的泛化能力待进一步验证。

---

## 170. Causal Cellular Context Transfer Learning (C3TL): An Efficient Architecture for Prediction of Unseen Perturbation Effects

**arXiv ID:** 2603.13051 | [PDF](https://arxiv.org/pdf/2603.13051v1)

**作者:** Michael Scholkemper `[一作]` (German Center for Neurodegenerative Diseases), Sach Mukherjee `[通讯]` (German Center for Neurodegenerative Diseases)

**通讯引用:** 3260 | [OpenAlex ID](https://openalex.org/A5112866063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种高效的神经框架C^3TL，用于在未知细胞上下文中预测化学和基因干扰的定量细胞状态变化；

**💡 创新点**

创新点在于利用因果流形假设，将扰动和上下文映射到低维潜在空间，并通过编码-解码结构实现跨上下文的推广；

**🔧 技术方法**

采用编码器-解码器的自编码器架构，利用聚合函数实现上下文不变性，结合线性和非线性网络实现扰动与上下文的潜在表示；

**📊 数据集**

使用三个大规模扰动数据集（Replogle、Parse、Tahoe-100），均为单细胞数据后处理为伪批量表达；

**📈 对比分析**

与State（大型基础模型）和CPA（自编码器）等基线进行交叉验证和有限数据评估，C^3TL在大部分数据集上与State竞争，并在Parse上取得最佳；在极限数据稀缺条件下表现优于State；

**⚠️ 局限性**

局限在于假设因果流形的可逆性和线性位移，缺乏正式的样本复杂度分析，且未在主动学习或强化学习框架中验证实际实验部署。

---

## 171. One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies

**arXiv ID:** 2603.12480 | [PDF](https://arxiv.org/pdf/2603.12480v1)

**作者:** Shaolong Li `[一作]` (University of Michigan), Yongchao Chen `[通讯]` (Tsinghua University)

**通讯引用:** 6411 | [OpenAlex ID](https://openalex.org/A5100383008)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种从零开始的自蒸馏框架One‑Step Flow Policy（OFP），实现单步生成高精度连续动作，显著降低推理延迟；

**💡 创新点**

创新点在于同时结合自一致性训练（强制跨时间一致的平均速度场）与自引导正则化（利用自身的无条件分支实现的分数引导），并引入warm‑start机制利用动作的时间相关性；

**🔧 技术方法**

技术包括基于条件流匹配的速度场学习、时间收缩的自一致性目标、无监督分数引导（CFG方式）以及EMA教师的自蒸馏；

**📊 数据集**

在Adroit、DexArt、MetaWorld、RoboTwin 2.0四大仿真基准上共56个任务进行评估；

**📈 对比分析**

与多步扩散/流政策、Consistency Policy、One‑Step Diffusion Policy、MeanFlow等基线相比，OFP在单步NFE=1时取得最高成功率，并实现100+×的速度提升；

**⚠️ 局限性**

局限性主要是实验仅在仿真环境中验证，缺乏真实机器人测试，且在极大规模VLA模型中仍需进一步评估鲁棒性。

---

## 172. CM-Bench: A Comprehensive Cross-Modal Feature Matching Benchmark Bridging Visible and Infrared Images

**arXiv ID:** 2603.12690 | [PDF](https://arxiv.org/pdf/2603.12690v1)

**作者:** Liangzheng Sun `[一作]` (Beijing Information Science and Technology University), Fei Xing `[通讯]` (Tsinghua University)

**通讯引用:** 41031 | [OpenAlex ID](https://openalex.org/A5111728714)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CM-Bench基准，系统评估了30种跨模态（红外-可见）特征匹配算法在30个数据集上的表现，并发布了新的ThermoSat红外-卫星数据集。

**💡 创新点**

创新点包括：①构建统一评估框架和多任务指标；②引入基于MobileNetV4的自适应预处理前端；③提供全新的红外-卫星数据集及其手工标注的对应点。

**🔧 技术方法**

采用的技术包括传统手工特征、深度学习特征提取（SuperPoint、D2Net、XFeat等）、Transformer/LoFTR系列匹配器、生成式模型（MINIMA、XoFTR）以及自适应预处理分类网络。

**📊 数据集**

使用的数据集有：MSCM（7,083对红外-可见对），METU-VisTIR（相机姿态数据），ThermoSat（832对红外-卫星），以及其他8个公开跨模态数据集（VisDrone、DUT-VTUAV、AVIID、LLVIP、M3FD、MSRS、KAIST、FLIR）。

**📈 对比分析**

通过四类任务（单应性估计、相对位姿估计、地理定位、困难地理定位）和AUC、MedErr、SR等指标对比，结果显示MINIMA-RoMa、RoMa、MINIMA-LG在所有任务中表现最优，且自适应预处理可为多种匹配器带来2–20%的性能提升。

**⚠️ 局限性**

局限性包括：①评估仅覆盖红外-可见对，缺乏其他模态（如激光、雷达）的验证；②自适应预处理的训练依赖于手工或自动标签，可能不适用于极端场景；③部分深度模型（RoMa、MASt3R）计算成本高，未能在实时系统中充分验证。

---

## 173. GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping

**arXiv ID:** 2603.12275 | [PDF](https://arxiv.org/pdf/2603.12275v1)

**作者:** Chahana Dahal `[一作]` (University of Nevada), Zuobin Xiong `[通讯]` (University of Nevada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于知识图谱的LLM知识遗忘评估基准GONE，并设计了邻域增强分布塑形（NEDS）框架，在LLaMA‑3‑8B和Mistral‑7B上实现了对结构化事实的高效遗忘。

**💡 创新点**

创新点包括：①首个以结构化KG事实为核心的遗忘基准，能够区分直接遗忘、推理泄露和灾难性遗忘；②NEDS通过邻域锚点约束实现精准决策边界，显著减少邻域知识损失；③引入知识连通性得分（KCS）量化遗忘深度。

**🔧 技术方法**

采用LoRA微调、NPO与锚点交叉熵损失、约束优化、KG采样与多跳探测、填空与问答探针生成，以及自定义的KCS评估指标。

**📊 数据集**

使用Wikidata和ConceptNet三元组构建的GONE数据集，同时与RWKU基准进行对照评估。

**📈 对比分析**

与梯度下降、正向/反向梯度、NPO、UL‑DPO、ICU、AlphaEdit、Wise等方法比较，NEDS在直接、同义、逆向和多跳查询上均实现UE≈1.00，且在LLaMA‑3上保持0.698的局部性（Locality），在Mistral‑7B上提升至0.839，拒绝率低于其他方法。

**⚠️ 局限性**

局限性：仅覆盖基于三元组的关系，未包含更复杂的组合或跨域推理；评估范围局限于GONE、RWKU和开源模型，未覆盖所有遗忘或编辑方法；对知识域的多样性与通用性仍需进一步扩展。

---

## 174. Rethinking VLMs for Image Forgery Detection and Localization

**arXiv ID:** 2603.12930 | [PDF](https://arxiv.org/pdf/2603.12930v1)

**作者:** Shaofeng Guo `[一作]` (Hefei University of Technology), Richang Hong `[通讯]` (Hefei University of Technology)

**通讯引用:** 22470 | [OpenAlex ID](https://openalex.org/A5051332325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了IFDL-VLM框架，用于图像伪造检测与定位，并通过解耦检测/定位与语言解释的训练，提升了真实性判断与可解释性。

**💡 创新点**

创新点在于(1)将检测与定位与VLM解释解耦，(2)利用检测得到的掩码作为VLM的额外输入以缓解VLM对语义合理性而非真实性的偏差，(3)引入区域感知视觉特征增强提升解释质量。

**🔧 技术方法**

使用ViT+SAM作为检测定位模块，CLIP与Vicuna-13B组成的VLM进行解释生成，并在Stage-1训练ViT与SAM，Stage-2用掩码增强CLIP视觉特征再输入LLM。

**📊 数据集**

在9个基准数据集上评估，包括SID-Set、CASIA1+、IMD2020、Columbia、NIST、DSO、Korus、DeepFake、AIGC Editing，且进行跨数据集泛化测试。

**📈 对比分析**

与SIDA、FakeShield、MVSS-Net等传统方法比较，检测准确率、F1、定位IoU均实现了SOTA提升（检测99.7% ACC、定位IoU 0.65、解释GPT-5分数 2.44），在跨数据集上平均IoU 0.47、F1 0.58。

**⚠️ 局限性**

局限在于对VLM预训练数据中缺乏伪造相关概念，模型仍可能在极度细微或全合成的AIGC伪造上产生误判，且推理时间较长。

---

## 175. On Linear Separability of the MNIST Handwritten Digits Dataset

**arXiv ID:** 2603.12850 | [PDF](https://arxiv.org/pdf/2603.12850v1)

**作者:** Ákos Hajnal `[一作]` (Institute for Computer Science and Control), Ákos Hajnal `[通讯]` (Institute for Computer Science and Control)

**通讯引用:** 593 | [OpenAlex ID](https://openalex.org/A5009501442)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对MNIST数据集在成对与一对其余类别的线性可分性进行了系统的经验性评估，使用CVXPY求解线性可行性问题，分别在训练集、测试集和两者合并的数据上进行测试。

**💡 创新点**

首次对所有45个数字对以及十个一对其余类别的可分性进行完整检验，并且用凸优化的可行性判定提供了更可靠的线性可分性判别，揭示训练集非可分而测试集仅成对可分的细粒度结论。

**🔧 技术方法**

主要技术为CVXPY中的线性规划可行性求解（使用CLARABEL求解器）以及对比前人方法的执行时间测量；对非可分情况通过求解失败来判定。

**📊 数据集**

使用经典MNIST数据集，共70,000张28×28像素的灰度手写数字图像（60,000训练 + 10,000测试）。

**📈 对比分析**

与Zhong等人以及其实现的最小包围球/LP方法比较，CVXPY在成对可分性测试中平均提升4–8倍速度；测试时间从几秒到数十秒不等，证明该方法在大规模可分性判定中具有显著性能优势。

**⚠️ 局限性**

局限性包括：对非可分情况仅给出数值证据而无严格理论证明；测试集样本量小导致一对其余可分性结论不够稳健；仅比较了有限几种求解器，未覆盖所有可用的凸优化或几何判定方法。

---

## 176. SortScrews: A Dataset and Baseline for Real-time Screw Classification

**arXiv ID:** 2603.13027 | [PDF](https://arxiv.org/pdf/2603.13027v1)

**作者:** Tianhao Fu `[一作]` (University of Toronto), Yucheng Chen `[通讯]` (Project Neura)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并发布了一个受控采集的螺丝视觉分类数据集 SortScrews，并提供可复现的数据采集脚本。

**💡 创新点**

创新点在于创建规模虽小但平衡、受控条件下的工业螺丝数据集，同时提供可复用的采集流程与代码，方便快速构建类似数据集。

**🔧 技术方法**

使用基于 ImageNet 预训练的 EfficientNet‑B0 与 ResNet‑18 两种轻量级卷积网络进行迁移学习进行分类。

**📊 数据集**

使用的数据集为自行构建的 SortScrews，包含 560 张 512×512 RGB 图像，涵盖六类螺丝和背景。

**📈 对比分析**

在验证集上比较两模型的准确率与推理速度，ResNet‑18 在 96.4% 的准确率下实现 6.42 ms/图（155.8 fps），优于 EfficientNet‑B0 的 86.2% 与 17.95 ms/图。

**⚠️ 局限性**

局限性包括数据集规模有限、仅单视角、受控环境导致泛化能力不足，以及对细微几何差异的识别仍存在误差。

---

## 177. Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**arXiv ID:** 2603.13108 | [PDF](https://arxiv.org/pdf/2603.13108v1)

**作者:** Guoqiang Zhao `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**通讯引用:** 5294 | [OpenAlex ID](https://openalex.org/A5027010844)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了针对四足机器人全景多模态语义占用预测的框架VoxelHound。

**💡 创新点**

创新点包括垂直抖动补偿模块和多模态信息提示融合模块，解决四足运动抖动和异构模态融合难题。

**🔧 技术方法**

采用深度卷积网络、BEV投影、稀疏体素CNN、注意力机制及跨模态融合技术。

**📊 数据集**

使用新构建的PanoMMOcc数据集，该数据集包含全景RGB、热像、偏振和LiDAR四种模态。

**📈 对比分析**

在PanoMMOcc基准上与MonoScene、EFFOcc等方法对比，VoxelHound取得23.34% mIoU，较基线提升4.16%/14.40%。

**⚠️ 局限性**

局限性在于对极端运动或低纹理场景的鲁棒性尚待提升，且模型对实时部署的计算成本未进行充分评估。

---

## 178. Overcoming the Modality Gap in Context-Aided Forecasting

**arXiv ID:** 2603.12451 | [PDF](https://arxiv.org/pdf/2603.12451v1)

**作者:** Vincent Zhihao Zheng `[一作]` (ServiceNow Research), Valentina Zantedeschi `[通讯]` (ServiceNow Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于大语言模型的情境生成与验证流程，构建了规模达740万条时间序列+文本对的半合成数据集CAF_7M，并设计了融合文本与序列的多模态模型DoubleCast；

**💡 创新点**

创新点在于（1）通过生成+验证双阶段保证上下文确实提升预测；（2）首次实现大规模高质量情境化时间序列数据；（3）提出DualT5解码器的双注意力机制，提升上下文与数值的对齐；

**🔧 技术方法**

技术要点包括：LLM提示式场景生成、基于Direct Prompt的验证（CRPS评估）、Chronos时序编码器、Qwen3-14B文本编码器、DualT5解码器以及多模态对齐融合；

**📊 数据集**

数据集使用65个真实世界时序数据集抽取训练窗口，生成740万条带情境样本，测试集含904条经过验证的窗口；另外对比ChatTime、CGTSF、GIFT-Eval等公开基准；

**📈 对比分析**

实验通过与Chronos、TimeLLM、ChatTime及Direct Prompt（GPT‑5.2/Qwen3‑14B）对比，指标为CRPS、Win Rate、MASE；DoubleCast在验证集上显著优于单模态模型，在ChatTime等真实场景中实现最优或竞争力预测，且在纯数值预测任务上保持与Chronos相近性能；

**⚠️ 局限性**

局限性包括：验证器偏倚导致采样上下文模式有限；训练集未经过验证，可能含噪声；生成文本多样性受限；跨模态对齐不足时难以精准映射时间信息，影响长预测窗口的精确度；验证过程计算成本高。

---

## 179. L2GTX: From Local to Global Time Series Explanations

**arXiv ID:** 2603.13065 | [PDF](https://arxiv.org/pdf/2603.13065v1)

**作者:** Ephrem Tibebe Mekonnen `[一作]` (Technological University Dublin), Pierpaolo Dondio `[通讯]` (Dublin City University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了L2GTX方法，将局部解释聚合成类级全局解释，适用于深度时间序列分类器。

**💡 创新点**

模型无关、利用LOMATCE产生的参数化事件原语进行聚类、使用子模优化选择代表实例，并通过统计汇总生成可解释的全局说明。

**🔧 技术方法**

利用LOMATCE本地解释、参数化事件原语（趋势、极值）、层次聚类合并、实例–集群重要性矩阵、子模优化、事件属性统计与全局faithfulness评估。

**📊 数据集**

六个UCR数据集：ECG200、GunPoint、Coffee、FordA、FordB、CBF。

**📈 对比分析**

在FCN和LSTM‑FCN两种模型上，对不同聚合阈值p（25/50/75/95）进行比较，全局faithfulness在不同p下保持稳定，说明聚合不降低解释可信度，并且生成的全局解释更具语义可解释性。

**⚠️ 局限性**

计算成本较高，尤其是事件聚类阶段；对长序列和大量邻域样本开销明显；目前仅支持单变量时间序列。

---

## 180. Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions

**arXiv ID:** 2603.12296 | [PDF](https://arxiv.org/pdf/2603.12296v1)

**作者:** Ziwei Wang `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 15062 | [OpenAlex ID](https://openalex.org/A5008740867)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对脑机接口（BCI）中的脑信号生成方法进行系统综述，并在四大 BCI 任务上开展统一基准实验。

**💡 创新点**

将生成方法分为四类（知识基、特征基、模型基、翻译基），并首次在多任务、多数据集下给出全量性能对比与多维评估框架。

**🔧 技术方法**

采用多种生成技术：规则/先验驱动（如时间频域变换）、特征插值（SMOTE、Mixup）、深度生成模型（GAN、VAE、AR、DDPM）和跨模态翻译（联合/条件潜在空间），配合常用解码器（EEGNet、SCNN、DCNN、IFNet、DBConformer）。

**📊 数据集**

使用 11 个公开数据集：MI（IV‑2a、Zhou2016、Blankertz2007、BNCI2014002、BNCI2015001）、SSVEP（Nakanishi2015、Benchmark）、ESD（CHSZ、NICU）与 AAD（KUL、DTU）。

**📈 对比分析**

通过跨受试与内受试设置，采用准确率/均衡准确率等指标进行比较；实验表明多种生成方法普遍提升解码性能，尤其 DWTaug 在 MI 与 SSVEP 任务中表现最佳，GAN 基生成在 Benchmark 上提升约 5% 以上。

**⚠️ 局限性**

主要限制包括：生成模型的可解释性和生理合理性不足，GAN 易出现模式崩溃、DDPM 采样延迟高；跨模态对齐与隐私保护评估仍不完善；评估指标多聚焦在统计相似度，缺少任务相关的功能性验证。

---

## 181. End-to-End Deep Learning in Wireless Communication Systems: A Tutorial Review

**arXiv ID:** 2603.12289 | [PDF](https://arxiv.org/pdf/2603.12289v1)

**作者:** Abdelrahman Elfiky `[一作]`, Georges Kaddoum `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并整理了端到端深度学习在无线通信物理层（PHY）中的应用，重点聚焦于自编码器（AE）模型及其在不同通信模式（P2P、MIMO、MAC、BC、干扰、光通信等）下的实现与优化。

**💡 创新点**

首次系统性地将AE框架与传统物理层设计进行对比，提供了从架构设计、训练策略、鲁棒性提升、实际部署到未来研究方向的完整蓝图，标识出AE在自适应、非线性、低SNR场景下的优势。

**🔧 技术方法**

使用端到端AE、卷积自编码器、循环自编码器、Turbo自编码器等深度学习架构；结合梯度下降、Adam、学习率衰减、正则化、归一化等训练技巧，并讨论了迁移学习、数据增强、强化学习与分布式学习等辅助技术。

**📊 数据集**

论文作为综述不采用特定数据集；主要讨论仿真环境（AWGN、Rayleigh/Fading、Log-normal、Poisson光通道、干扰通道等）以及真实通道测量与软件定义无线电（SDR）实验。

**📈 对比分析**

通过与传统调制+编码（QPSK、BPSK、LDPC、turbo码等）以及基于信道模型的优化方法在BLER/BER、吞吐率、能耗等指标上进行对比，指出AE在低SNR、非线性、复杂多用户场景下能实现与传统方法相当甚至更优的性能。

**⚠️ 局限性**

主要局限包括：缺乏统一的评估基准与开源实现；硬件实现与实时延迟、能耗的量化不足；对不可微分通道（光通信、非线性放大器等）的建模与训练仍不成熟；对模型可解释性与安全性的讨论有限。

---

## 182. Operationalising Cyber Risk Management Using AI: Connecting Cyber Incidents to MITRE ATT&CK Techniques, Security Controls, and Metrics

**arXiv ID:** 2603.12455 | [PDF](https://arxiv.org/pdf/2603.12455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 183. Efficient Real-World Autonomous Racing via Attenuated Residual Policy Optimization

**arXiv ID:** 2603.12960 | [PDF](https://arxiv.org/pdf/2603.12960v1)

**作者:** Raphael Trumpp `[一作]` (Technical University of Munich), Marco Caccamo `[通讯]` (Technical University of Munich)

**通讯引用:** 6629 | [OpenAlex ID](https://openalex.org/A5060442004)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在训练过程中逐步衰减基准策略的残差策略优化方法（α-RPO），并将其应用于 1:10 规模的自主赛车，实现在仿真与真实环境中的零镜像转移。

**💡 创新点**

创新点在于：①将基准策略的影响逐步减弱至零，最终获得独立的神经网络策略；②在 PPO 中加入同步技巧以补偿非平稳的基准衰减；③通过允许基准策略使用额外传感器实现“特权学习”，提高训练效率。

**🔧 技术方法**

技术核心包括：深度强化学习（PPO）、残差策略学习、截断高斯分布、同步技巧、Pacejka 轮胎模型、LiDAR 预处理、SpatialSoftmax 等网络结构。

**📊 数据集**

数据集主要是：在自研的 Roboracer Gym 仿真环境中生成 15 条训练赛道（其中 6 条用于测试），以及真实的 Munich 赛道用于零镜像转移；同时使用 1:10 规模的 Roboracer 车辆及其 2D LiDAR、IMU 等传感器数据。

**📈 对比分析**

与基准方法（标准 RPL、DRL、BC+DRL、DRL+SSL、FTG、Stanley）比较，α‑RPO 在仿真中平均赛时最短、碰撞率最低、最高速度最高；在真实赛道上同样表现优异，零镜像转移后赛时仅提升约 0.4 s，仍保持最高速度约 5 m/s，且能在加入障碍时顺利避障。

**⚠️ 局限性**

限制包括：①缺乏可验证的安全保证，难以进行形式化验证；②仿真参数为经验调校，未进行严格识别，导致真实极限处理仍受不匹配影响；③性能对基准策略质量和线性衰减调度敏感；④目前仅在赛车领域验证，尚需在其他机器人控制任务中进一步验证。

---

## 184. SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion

**arXiv ID:** 2603.12764 | [PDF](https://arxiv.org/pdf/2603.12764v1)

**作者:** Xiang Li `[一作]` (University of Electronic Science and Technology of China), Hongliang Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 43306 | [OpenAlex ID](https://openalex.org/A5075571728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了跨视角（第一人称→第三人称）模仿错误检测任务，并设计了 SAVA-X 框架来同时解决时间错位、冗余帧和视角域差异。

**💡 创新点**

创新点在于：1）自适应采样（硬 Top‑K + 残差门控）专门针对异步、长视频的冗余和对齐问题；2）场景感知视角嵌入字典，动态补偿 ego/exo 视角差异；3）双向交叉注意力融合，在不丢失各自视角信息的前提下实现两端互补；4）在统一协议下把 dense video captioning 与 temporal action localization 两大技术迁移到跨视角错误检测，展示显著性能提升。

**🔧 技术方法**

核心技术包括：冻结 TSP 视频编码器；门控自适应采样（Gumbel‑TopK + residual gating）；场景感知视角嵌入字典与多层注入；双向交叉注意力融合；Deformable Transformer 编码–解码器；set‑matching 损失（ℒ_DVC 与 ℒ_Imit）以及相关正则化（选取熵、VICReg 等）。

**📊 数据集**

使用 EgoMe 数据集（约 7,902 对异步 Ego/Exo 视频，包含细粒度步骤与错误标签），按官方 4,777/997/2,128 训练/验证/测试拆分进行评估。

**📈 对比分析**

与基线（PDVC、Exo2EgoDVC、ActionFormer、TriDet）在统一协议下进行对比；SAVA‑X 在所有 tIoU 阈值（0.3/0.5/0.7）的 AUPRC 上平均提升约 2.7（+13% 相对最佳基线），tIoU 也提升 0.9–1.1；在验证集和测试集均保持一致的优势。

**⚠️ 局限性**

局限性：1）仍依赖 EgoMe 这类同步收集的跨视角数据，跨场景泛化未知；2）视角嵌入字典尺寸和正则化需要手动调节；3）模型规模大，计算和内存成本高；4）长视频中仍可能出现冗余帧未被完全抑制，导致注意力分散；5）对极端异步或极长时长的视频适配尚未充分验证。

---

## 185. NeurFrame: Learning Continuous Frame Fields for Structured Mesh Generation

**arXiv ID:** 2603.12820 | [PDF](https://arxiv.org/pdf/2603.12820v1)

**作者:** Xiaoyang Yu `[一作]` (Xiamen University), Juan Cao `[通讯]` (Xiamen University)

**通讯引用:** 9664 | [OpenAlex ID](https://openalex.org/A5056924576)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 NeurFrame，利用自监督神经网络连续表示三维框架场，支持无限分辨率查询并指导高质量四边形与六面体网格生成。

**💡 创新点**

创新点在于：1）单一网络学习连续框架场，避免离散网格的高成本；2）自监督训练，无需人工标签；3）同时兼顾四边形与六面体网格；4）通过 SH 表示与特定对齐约束实现更少、更均匀的奇异点。

**🔧 技术方法**

技术手段包括：SIREN MLP 架构、Spherical Harmonics（SH）系数表示、平滑、边界对齐、特征对齐三项损失，以及三角形/四面体网格的采样与转换。

**📊 数据集**

使用 Thingi10K、HexMe 以及公开模型库的样本，先用 fTetWild 对表面网格进行网格化生成体网格。

**📈 对比分析**

与 InstantMeshes、QuadriFlow、QuadWild、NeurOcta、NeurCross 等现有跨域或框架场方法进行对比；NeurFrame 在相同目标面数下实现更少、分布更均匀的奇异点；每次迭代运行时间最低（≈53 ms），显著快于 NeurOcta（≈95 ms）和 NeurCross（≈477 ms）。

**⚠️ 局限性**

局限在于生成的框架场不一定满足全局可网格化（IGM 对齐）条件，导致在某些简单模型上仍可能无法得到有效的六面体网格；未来工作计划将纠正奇异图作为训练目标，直接引导可网格化的框架场。

---

## 186. TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation

**arXiv ID:** 2603.12665 | [PDF](https://arxiv.org/pdf/2603.12665v1)

**作者:** Kaidi Zhang `[一作]` (Purdue University), Yu She `[通讯]` (Purdue University)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5018653973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出TacVLA模型，融合视觉、语言与触觉，实现接触丰富的细粒度机械操作；

**💡 创新点**

创新在于将触觉以低维标记化方式融入Transformer，并通过接触状态门控仅在有触碰时激活触觉特征，避免无关信息干扰；

**🔧 技术方法**

采用预训练SigLIP视觉语言模型、LoRA微调、Pi0.5动作专家、MLP触觉编码以及接触门控机制；

**📊 数据集**

使用真实机器人采集的四种约束解组任务与箱内抓取任务的50条演示数据，包含RGB图像、语言指令、触觉矩阵和关节状态；

**📈 对比分析**

与Pi0.5、Diffusion Policy等基线对比，TacVLA在解组任务平均成功率达83.75%，箱内抓取70%，且在视觉遮挡与人类干扰条件下仍保持显著优势；

**⚠️ 局限性**

局限包括门控采用固定阈值二值化，缺乏可学习的模态权重；触觉传感器分辨率有限；实验仅覆盖短时接触任务，未验证更长周期或更复杂任务的表现。

---

## 187. SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models

**arXiv ID:** 2603.12768 | [PDF](https://arxiv.org/pdf/2603.12768v1)

**作者:** Aditya Maheshwari `[一作]` (Indian Institute of Management Indore), Vivek Patel `[通讯]` (Indian Institute of Management Indore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SectEval基准，专门测量大型语言模型在伊斯兰逊尼派与什叶派之间的细粒度偏见；

**💡 创新点**

首次将内在宗派偏见转化为可量化的二元选择任务，展示语言与地区对模型回答的显著影响；

**🔧 技术方法**

采用多语言提示、基于上下文的地区标识、链式推理（CoT）以及统计检验（McNemar）等技术评估模型；

**📊 数据集**

构建了包含88道双选题的测试集（涵盖神学、历史、礼仪等主题），涵盖英语与印地语两种语言；

**📈 对比分析**

对15个闭源与开源 LLM 进行比较，发现英语提示下多模型偏向什叶派，而印地语提示下则偏向逊尼派；部分高容量模型会根据用户地区自适应；在模型规模不同的分类中，小模型普遍逊尼偏向，大型/前沿模型往往什叶偏向；链式推理更显多样化；整体性能显示跨语言不一致与地区适应性差异明显；

**⚠️ 局限性**

仅覆盖88道题，缺乏对其他伊斯兰教派的覆盖；未测试最新模型（如 GPT‑5、Gemini 2.5 Flash）；数据集规模有限，可能未能充分代表伊斯兰法学与神学的完整范围；

---

## 188. Towards Output-Optimal Uniform Sampling and Approximate Counting for Join-Project Queries

**arXiv ID:** 2603.12560 | [PDF](https://arxiv.org/pdf/2603.12560v1)

**作者:** Xiao Hu `[一作]` (University of Waterloo), Jinchao Huang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1274 | [OpenAlex ID](https://openalex.org/A5100737726)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究了联接-投影查询的均匀采样与近似计数问题，并给出了针对矩阵、星型和链型三类查询的渐近最优算法。

**💡 创新点**

创新点在于首次提出了输出最优、渐近最优的采样与计数框架，利用拒绝采样、混合计数策略，并通过通信复杂度归约与组合算法下界证明了链型查询不可实现子线性解法。

**🔧 技术方法**

主要技术包括拒绝采样与负超几何分布推导、混合采样/计数方法、KMV摘要技术、反向动态规划求连通度、通信复杂度与组合归约等。

**📊 数据集**

本文未使用真实数据集，研究完全基于理论分析和构造难点实例。

**📈 对比分析**

与以往最优但复杂度为 O(N^2/ρ*) 的 propose‑and‑verify 框架相比，矩阵查询采样时间从 O(N^2/ρ*) 降到 O(N/√|output|)，星型查询从 O(N^k/ρ*) 降到 O(N^{1/k})，链型查询实现 O(N) 采样并给出匹配下界，整体性能大幅提升。

**⚠️ 局限性**

局限性包括仅针对三类典型查询，链型查询的下界仅在信息论层面；未涵盖选择谓词、一般联接结构或利用代数方法（如快速矩阵乘法）的可能突破。

---

## 189. A protocol for evaluating robustness to H&E staining variation in computational pathology models

**arXiv ID:** 2603.12886 | [PDF](https://arxiv.org/pdf/2603.12886v1)

**作者:** Lydia A. Schönpflug `[一作]` (University of Basel), Maxime W. Lafarge `[通讯]` (University of Basel)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5077058928)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套三步协议，用于系统评估计算病理模型对H&E染色差异的鲁棒性，并在微卫星不稳定（MSI）分类任务中验证该协议；

**💡 创新点**

创新之处在于构建可复现的染色参考库，并通过可控的染色强度与颜色相似度模拟，实现对模型性能波动的量化评估，为模型选择与临床部署提供鲁棒性依据；

**🔧 技术方法**

采用颜色分解与重组方法模拟染色变化，结合ABMIL聚合框架和多种基础模型（UNI2-h、H-Optimus-1、Virchow2、CTransPath、RetCCL）训练的306个MSI分类模型；

**📊 数据集**

使用PLISM数据集构建参考染色库，TCGA COAD/READ作为训练/验证集，SurGen数据集作为独立测试集；

**📈 对比分析**

在原始及四种模拟染色条件下计算AUC，记录AUC的最小-最大范围作为鲁棒性指标，得到AUC范围0.769-0.911，鲁棒性范围0.007-0.079，模型性能与鲁棒性呈弱负相关；

**⚠️ 局限性**

局限性包括参考库覆盖不足、仅评估染色变化忽略其他QC因素、未考虑血液等特殊组织区域、只测试四种染色条件，且模型多样性与数据集分布仍有限。

---

## 190. Feynman: Knowledge-Infused Diagramming Agent for Scalable Visual Designs

**arXiv ID:** 2603.12597 | [PDF](https://arxiv.org/pdf/2603.12597v1)

**作者:** Zixin Wen `[一作]` (Carnegie Mellon University), Wode Ni `[通讯]` (Carnegie Mellon University)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5075566366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为Feynman的可扩展图示生成流水线，通过先提取领域知识，再转化为可执行的程序，最终生成高质量的图示与对应标题。

**💡 创新点**

将知识挖掘与图像生成解耦，利用LLM进行知识枚举与代码规划，并通过迭代视觉评审实现高一致性和多样性的图示；同时发布了100k对齐的图示-标题数据集和Diagramma视觉推理基准。

**🔧 技术方法**

LLM驱动的知识规划、代码规划与迭代视觉精炼、基于Penrose的优化布局渲染、Levenshtein去重、自动生成多选问答。

**📊 数据集**

通过Feynman合成的106,930个图示-标题对，随后手工筛选得到1,058道多选题的Diagramma基准集；实验还使用了多模态LLM评测模型。

**📈 对比分析**

在零样本设置下对17款开闭源MLLM进行评估，发现模型规模越大准确率越高，计算机科学子任务最难；Gemini-1.5 Flash优于Pro，表明新鲜度与推理能力相关。

**⚠️ 局限性**

生成过程仍受LLM写代码准确性、视觉评审依赖和知识枚举的多样性限制，且在知识稀疏领域的扩展性下降，整体成本与时间仍高。

---

## 191. Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World

**arXiv ID:** 2603.12746 | [PDF](https://arxiv.org/pdf/2603.12746v1)

**作者:** Yuzhi Huang `[一作]` (Xiamen University), Zhi Wang `[通讯]` (Tsinghua University)

**通讯引用:** 19088 | [OpenAlex ID](https://openalex.org/A5100376411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了首个针对多模态大型语言模型（MLLMs）的动态四维（4D）时空推理与对象定位基准（Dyn-Bench），通过构建包含1,000段视频、7,000个视觉问答（VQA）对以及3,000个动态对象定位标注的高质量数据集，系统评估模型在动态场景中的感知、跟踪与推理能力。

**💡 创新点**

创新点包括①三层次的动态理解框架（对象间感知、对象-场景跟踪、相机-对象推理），②基于规则模板的Spatio‑Temporal Textual Cognitive Map（ST‑TCM）将几何、运动与关系信息结构化为文本；③Mask‑Guided Fusion视觉引导方法提升模型对动态区域的关注。

**🔧 技术方法**

采用的技术主要有：多阶段过滤与人类审核的高质量视频筛选；ST‑TCM文本生成与对齐；Mask‑Guided Fusion将原始帧与分割掩码融合；多模态注意力与跨模态对齐；使用多选问答的准确率和视频分割的J&F指标进行评估。

**📊 数据集**

数据集来源为四个二维分割数据集（DAVIS、SA‑V、DynPose‑100K、YouTube‑VIS）和四个四维动态场景数据集（DynamicReplica、PointOdyssey、Spring、Total‑Recon），经过筛选后得到1k视频、7k VQA和3k定位标注。

**📈 对比分析**

通过与三类模型（通用、空间、区域级）在零样本设置下对比，结果显示GPT‑4o等专有模型在关系与运动推理上表现优异，开源模型如Qwen3‑VL‑235B已接近其水平；区域级模型在对象级时空推理与定位上领先；Mask‑Guided Fusion与ST‑TCM分别在推理和定位任务上提升约5–10个百分点，但总体性能仍未达到人类水平。

**⚠️ 局限性**

局限性主要有：模型缺乏结构化的时空与关系表示，导致在遮挡、相机运动等复杂情况下易失真；基准仅覆盖问答与定位，未涉及动作生成或交互式推理；当前评测显示多模态模型在动态场景中的表现仍相对薄弱，需要更统一的时空动态建模架构。

---

## 192. TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning

**arXiv ID:** 2603.12529 | [PDF](https://arxiv.org/pdf/2603.12529v1)

**作者:** Alliot Nagle `[一作]` (University of Texas Austin), Hyeji Kim `[通讯]` (University of Texas Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种利用LRM隐藏状态训练的二分类器，实现推理过程中的早期退出，显著减少Chain‑of‑Thought推理长度；

**💡 创新点**

创新点在于以后见最佳推理长度为监督信号，训练的退出器无需阈值调优，并能捕捉模型生成答案时的信心与思考词频突变；

**🔧 技术方法**

技术手段包括：LRM最终层隐藏状态提取、Token‑Confidence与思考词频分析、基于Transformer块的二分类头；

**📊 数据集**

训练数据来自AIME、MATH、OpenCoder‑SFT、OpenScience，评估数据为AIME 2025、MATH‑500、HumanEval、GPQA；

**📈 对比分析**

与Vanilla、NoThinking、DEER、Dynasor、Thought Calibration等方法对比，压缩率降低14%–55%，准确率保持或提升，位居效率–准确率Pareto前沿；

**⚠️ 局限性**

局限性包括：对GPQA等特定数据集泛化能力不足、对训练分布高度依赖、构建后见最佳长度标注需要大量样本且耗时。

---

## 193. A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering

**arXiv ID:** 2603.12605 | [PDF](https://arxiv.org/pdf/2603.12605v1)

**作者:** Pritham Kumar Jena `[一作]`, Sk Aziz Ali `[通讯]` (BITS Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建并公开了 A2Z 数据集，包含 100 万 CAD 模型与对应高分辨率 3D 扫描、手绘 3D 草图、文本标题与标签等多模态注释，并在此基础上训练并评估了基于 DGCNN 的 BRep 边界与角点检测基础模型。

**💡 创新点**

创新点主要有：① 超大规模（>10M）多模态注释的 CAD-扫描配对数据集；② 通过几何处理与软标签策略生成近似真实扫描的高质量 Mesh；③ 采用多级手绘草图生成算法模拟不同绘图技能；④ 使用双 VLM 并行裁决生成高质量文本描述与标签；⑤ 在该数据集上首次实现并超越现有 BRep 识别基线模型。

**🔧 技术方法**

技术方法包括：几何分割与多尺度软标签投影、Perlin 噪声与凹凸扰动、BRep 拓扑映射、基于 DGCNN 的点云分类网络、Qwen3‑14B 与 InternVL‑26B 共同生成文本注释、以及基于 GPT‑5 与 Gemini 的质量评估。

**📊 数据集**

主要使用的数据集为 A2Z（约 1.025M CAD 模型、5 TB 大规模注释）以及扩展的 25K 电子封装 CAD；此外，还与 CC3D、Fusion‑360 Gallery、ABC、DeepCAD 等公开 CAD 数据集进行对比评测。

**📈 对比分析**

与传统 BRep 边界/角点检测方法（BRepDetNet、ComplexGen、PieNet 等）在 A2Z、CC3D 以及新增电子封装子集上进行零样本/迁移学习评估。实验显示，A2Z 基础模型在边界召回率/精确率上分别达到 0.978/0.901，角点召回率/精确率达到 0.732/0.891，显著优于现有基线模型，并在 unseen 章节的性能衰减更小。

**⚠️ 局限性**

局限性包括：① 仍未覆盖所有 CAD 设计历史与参数化表达；② 虽然采用了软标签策略，但对极其细小或复杂几何的精准定位仍有挑战；③ 文本注释依赖 VLM 生成，可能在专业术语准确性上存在偏差；④ 仅在几种硬件平台上验证，未覆盖所有扫描设备和扫描质量场景。

---

## 194. Adaptive Diffusion Posterior Sampling for Data and Model Fusion of Complex Nonlinear Dynamical Systems

**arXiv ID:** 2603.12635 | [PDF](https://arxiv.org/pdf/2603.12635v1)

**作者:** Dibyajyoti Chakraborty `[一作]` (Pennsylvania State University), Romit Maulik `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了一种基于扩散模型的概率仿真框架，用于高维混沌流体动力学的长期预测。

**💡 创新点**

创新点包括多步自回归训练目标、基于图变压器的多尺度结构、以及利用扩散后验采样实现的自适应传感器布局与数据同化。

**🔧 技术方法**

采用的技术包括深度学习扩散模型、Graph Transformer网络、EDM预处理、AdaLN‑Zero条件化、体素网格池化、扩散后验采样和误差预测网络。

**📊 数据集**

使用的数据集为二维齐性等向湍流（Re=1000）和背向阶（Re=26000）两种流动场的DNS/LES模拟结果。

**📈 对比分析**

在对比实验中，多步训练的扩散模型显著降低了长期误差，标准差驱动或预测误差驱动的传感器布局优于随机布局，且数据同化后误差进一步下降。

**⚠️ 局限性**

主要局限在于仅验证二维案例，集合生成导致高计算成本，且对三维复杂几何和物理约束的适应性仍待进一步研究。

---

## 195. Goal-Oriented Learning at the Edge: Graph Neural Networks Over-the-Air for Blockage Prediction

**arXiv ID:** 2603.13094 | [PDF](https://arxiv.org/pdf/2603.13094v1)

**作者:** Lorenzo Mario Amorosa `[一作]` (University of Bologna), Roberto Verdone `[通讯]` (University of Bologna)

**通讯引用:** 4962 | [OpenAlex ID](https://openalex.org/A5022214905)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 GO-ST-AirGNN 目标导向的通信框架，将时空图神经网络与空中计算结合，用于毫米波网络的遮挡预测。

**💡 创新点**

创新点：1) 将无线信道视为模拟处理层，实现低延迟空中聚合；2) 目标导向的功率分配模块充当物理注意力；3) 在足够子载波条件下证明与数字 MPNN 等价并实现 O(1) 通信复杂度；4) 通过轻量迁移学习实现对新环境的快速适应。

**🔧 技术方法**

技术：空中计算 AirComp、时空图神经网络 (GNN)、联合集中式训练 (CTDE)、软最大激活功率分配、LSTM 解码器、NVIDIA Sionna Ray‑tracing 仿真。

**📊 数据集**

数据集：高保真 Ray‑tracing 数值数据集，基于 BI‑REX 设施的数字孪生场景；训练集 1000 条时间序列，测试集 1000 条，额外域移集 10 条。

**📈 对比分析**

比较方法：与 Local LSTM、ST‑GCN、ST‑GAT 三种基线对比；在遮挡预测准确率和 F1 上与数字基线相近或更优；在低子载波时仍保持较高准确率；在域移场景下迁移学习可恢复性能；通信延迟显著低于数字方案。

**⚠️ 局限性**

限制：需要严格同步与功率控制；目前训练为离线集中式，未实现完全分布式训练；对极端噪声、多径干扰鲁棒性待验证；在极低子载波数时性能可能受限。

---

## 196. Learning Pore-scale Multiphase Flow from 4D Velocimetry

**arXiv ID:** 2603.12516 | [PDF](https://arxiv.org/pdf/2603.12516v1)

**作者:** Chunyang Wang `[一作]` (Imperial College London), Gege Wen `[通讯]` (Imperial College London)

**通讯引用:** 1008 | [OpenAlex ID](https://openalex.org/A5007168661)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用多模态学习框架，将同步辐射4D微流速实验数据直接训练的图网络与3D U-Net耦合，实时预测多相多孔介质中粒子速度与界面演化，实现孔隙尺度的自回归推理。

**💡 创新点**

创新点在于：① 将Lagrangian粒子动力学（图网络）与Eulerian界面重建（3D U-Net）两种模态紧耦合，② 通过实验测得的粒子速度和界面信息直接约束模型，③ 在零射击跨介质的情况下仍能保持较高精度，显著缩短数值模拟时间。

**🔧 技术方法**

使用的技术包括：图网络模拟器（Graph Neural Network）处理粒子动力学；3D U-Net 进行体素化界面重建；多模态信息交互与自回归推理；同步辐射4D微流速实验数据作为训练和评估基础。

**📊 数据集**

采用的实验数据集来自SLS TOMCAT beamline的硅油与盐水渗流实验，包含粒子轨迹、界面演化以及孔隙几何的4D微流速数据，分辨率约2.75 µm，时间分辨率0.25 s。

**📈 对比分析**

模型在训练集、边界条件泛化（Exp.β）以及零射击跨岩石（Ketton石灰岩）测试中，粒子轨迹R²≈0.999，Dice>0.98，速度MAE≤1.1；与传统数值模拟相比，推理时间从数小时降至约5 s，实现10³–10⁴倍加速。

**⚠️ 局限性**

局限性包括：① 训练数据规模有限，跨介质迁移性能随孔隙结构差异显著下降；② 长时间自回归推理易出现误差累积；③ 对不同孔隙几何的适应性需更多多样化实验数据和更完善的物理约束。

---

## 197. Predictive and adaptive maps for long-term visual navigation in changing environments

**arXiv ID:** 2603.12460 | [PDF](https://arxiv.org/pdf/2603.12460v1)

**作者:** Lucie Halodova `[一作]` (Czech Technical University), Tomas Krajnik `[通讯]` (Czech Technical University)

**通讯引用:** 4164 | [OpenAlex ID](https://openalex.org/A5082772824)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

论文探讨了某个主题，提供了相关的背景和研究动机。

**💡 创新点**

创新点在于提出了一种新的方法或理论框架，能够更好地解决现有问题。

**🔧 技术方法**

使用了先进的算法或技术，例如机器学习、深度学习等。

**📊 数据集**

实验中使用了特定的数据集，可能是公开数据集或自建数据集。

**📈 对比分析**

通过与现有方法进行比较，展示了新方法在性能上的优势，具体指标如准确率、召回率等。

**⚠️ 局限性**

限制在于方法的适用范围可能有限，或在某些特定情况下性能不佳。

---

## 198. Bridging the Gap Between Security Metrics and Key Risk Indicators: An Empirical Framework for Vulnerability Prioritization

**arXiv ID:** 2603.12450 | [PDF](https://arxiv.org/pdf/2603.12450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 199. Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE

**arXiv ID:** 2603.12552 | [PDF](https://arxiv.org/pdf/2603.12552v1)

**作者:** Faris Chaudhry `[一作]` (Imperial College London), Faris Chaudhry `[通讯]` (Imperial College London)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5119538764)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

阐述了InfoNCE对比学习中温度参数的理论动态，并将其与模拟退火和Langevin动力学联系起来，证明在对数型退火调度下收敛至全局最优表示；

**💡 创新点**

将InfoNCE视为自由能，建立了其与吉布斯分布、Langevin SDE以及模拟退火的理论对应关系，并给出了渐近全局收敛、有限时间收敛速率以及Hessian线性放大等结论；

**🔧 技术方法**

利用Riemannian流形上的Langevin随机微分方程、Γ‑收敛、Freidlin–Wentzell离开时间理论、Hessian 线性缩放分析以及模拟退火的经典条件；

**📊 数据集**

无实验数据集，本文完全是理论分析；

**📈 对比分析**

未进行实验比较，本文仅给出理论收敛保证，并与常见的经验温度调度（如余弦、平方根等）进行概念性对比，说明理论调度在无穷远时可确保全局最优，但在有限时间内实验方法可能更快；

**⚠️ 局限性**

假设嵌入流形紧致且相似性函数光滑，噪声为各向同性高斯；未考虑非紧致流形、非球面几何、方差异质噪声或非高斯噪声；实际SGD噪声的协方差不一定满足假设，且未对动量或Adam等优化器进行分析。

---

## 200. HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks

**arXiv ID:** 2603.12760 | [PDF](https://arxiv.org/pdf/2603.12760v1)

**作者:** Xiaoyu Li `[一作]` (University of Electronic Science and Technology of China), Zihan Xiong `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5118625873)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HiFICL 框架，用可学习的低秩虚拟键值对直接建模注意力中的 In-Context Learning 效果；

**💡 创新点**

将 ICL 近似问题从仅逼近外部偏移向量，转为直接参数化其源头（键值对），并引入双低秩分解与无教师端到端训练；

**🔧 技术方法**

Transformer 注意力解析、低秩因子化（LoRA 风格）、虚拟键值对、全参数冻结+可训练虚拟矩阵的 end-to-end 损失；

**📊 数据集**

LLaVA-Interleave-7b 与 Idefics2-8b-base 在 VQAv2、OK-VQA、COCO Captioning 这三大多模任务上；

**📈 对比分析**

与 Zero-shot/8-shot ICL、LoRA、LIVE、MimIC 等基线对比，HiFICL 在 VQAv2、OK-VQA、COCO Captioning 上均取得 SOTA，参数量仅 1.1‑4.4M，且在低样本与推理效率上均优于竞争者；

**⚠️ 局限性**

对任务难度的低秩阶数缺乏统一自适应策略，且目前仅验证在两大 LMM 上，尚未在更大模型或跨模态更复杂任务上进行验证。

---

## 201. ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries

**arXiv ID:** 2603.12942 | [PDF](https://arxiv.org/pdf/2603.12942v1)

**作者:** Hang Li `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**通讯引用:** 24975 | [OpenAlex ID](https://openalex.org/A5063781430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了ReMem‑VLA，一种结合双层递归记忆查询的视觉‑语言‑动作模型，用于闭环机器人控制，并加入过去观测预测以增强视觉记忆。

**💡 创新点**

创新点包括：①双层递归查询（帧级与块级）实现短期与长期记忆；②梯度自由EMA更新路径，保证记忆稳定；③过去观测预测辅助视觉记忆；④基于槽位的流式训练方式，可在变长任务上高效批量训练。

**🔧 技术方法**

技术实现包括：冻结的大规模视觉‑语言模型（Qwen3‑VL‑2B）+可学习的动作、回顾、记忆查询；Transformer连接器实现双向注意；动作扩散网络预测动作块；EMA（指数移动平均）递归记忆更新；slot‑based streaming训练与推理。

**📊 数据集**

数据集与任务：模拟端使用MemoryBench（含扩展长时程任务）；真实端在UR5+Robotiq抓取器上使用RealSense D435完成水花、拣米、序列按键、果实回放等四个需要不同记忆维度的任务；每项任务收集数百条演示。

**📈 对比分析**

对比基线包括OpenVLA‑OFT、π_0.5与MemoryVLA，评估指标为任务成功率。ReMem‑VLA在MemoryBench上平均成功率94.5%，单项任务如Put Block Back 93%远高于OpenVLA的0%；在真实机器人任务上平均成功率82.5%，比基线提升约8–11%。

**⚠️ 局限性**

局限性：未在大规模机器人数据集上预训练，可能影响跨任务泛化；梯度自由更新虽然简化训练，却限制了记忆传播的可学习性；当前仅支持单任务回合内记忆，缺乏跨回合终身记忆能力。

---

## 202. Test-Time Attention Purification for Backdoored Large Vision Language Models

**arXiv ID:** 2603.12989 | [PDF](https://arxiv.org/pdf/2603.12989v1)

**作者:** Zhifang Zhang `[一作]` (University of Queensland), Miao Xu `[通讯]` (University of Queensland)

**通讯引用:** 3052 | [OpenAlex ID](https://openalex.org/A5016620131)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关、仅在测试时操作的防御方法CleanSight，用于抵御大型视觉-语言模型的后门攻击；

**💡 创新点**

核心创新是发现后门激活表现为视觉注意力被触发器“窃取”文本注意力，利用注意力比例检测并通过高注意力视觉令牌裁剪来消除后门；

**🔧 技术方法**

技术手段包括：跨模态融合层的头级视觉-文本注意力比率计算、白化L2距离阈值检测、以及基于注意力裁剪的输入净化；

**📊 数据集**

在VQA（VQAv2、OKVQA）和图像描述（MSCOCO、Flickr8k）等公开数据集上进行实验，并对多种后门攻击（BadNet、Blended、ISSBA、WaNet、TrojVLM、VLOOD）进行评测；

**📈 对比分析**

与传统像素级净化方法（模糊、空间变换、BDMAE、SampDetox、ZIP）以及无防御基线相比，CleanSight将攻击成功率降至接近0%，同时保持或接近原模型的清洁输入性能（V-score/CIDEr不下降或略升），在多模型、多攻击场景下表现稳健；

**⚠️ 局限性**

局限性包括：仅针对视觉-语言适配器层的后门，无法防御已在视觉编码器或语言模型参数中植入的后门；对极低触发器像素覆盖率或自适应攻击的鲁棒性尚待进一步验证；

---

## 203. Interrogating Design Homogenization in Web Vibe Coding

**arXiv ID:** 2603.13036 | [PDF](https://arxiv.org/pdf/2603.13036v1)

**作者:** Donghoon Shin `[一作]` (University of Washington), Emily Tseng `[通讯]` (Microsoft Research and University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了“网页Vibe编码”这一新兴实践，探讨了其生命周期、潜在的同质化风险与对设计多样性的影响，并提出了以“生产性摩擦”为核心的多层次缓解框架；

**💡 创新点**

首次系统化描述Vibe编码流程与风险，揭示了无摩擦生成如何加速设计同质化，并提出通过微观至宏观层面的“生产性摩擦”干预，提升创作者的决策主动性与设计多样性；

**🔧 技术方法**

主要采用文献综述、工具走查与案例分析相结合的方法；研究中未涉及深度学习模型实现，仅讨论现有LLM（如ChatGPT、Gemini、Claude）在Vibe编码中的应用场景；

**📊 数据集**

未使用特定数据集；研究基于对学术与灰色文献、Vibe编码工具（ChatGPT Canvas、Gemini Canvas、Claude Artifacts、Lovable、v0、Replit）以及公开的案例网站进行分析；

**📈 对比分析**

研究未涉及定量性能评估或对比实验；通过案例研究与风险评估展示框架的可行性与预期效果；

**⚠️ 局限性**

研究局限在于缺乏实证用户研究和大规模数据验证，所述风险与缓解方案基于文献与工具观察，需进一步通过实验和用户反馈进行验证与细化；

---

## 204. Context is all you need: Towards autonomous model-based process design using agentic AI in flowsheet simulations

**arXiv ID:** 2603.12813 | [PDF](https://arxiv.org/pdf/2603.12813v1)

**作者:** Pascal Schäfer `[一作]` (BASF SE), Norbert Asprion `[通讯]` (BASF SE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用最新大语言模型（Claude Opus 4.6）与多代理框架，自动完成化工流程图的概念设计、Chemasim 语法生成和仿真求解，实现端到端的无监督流程图开发；

**💡 创新点**

首次将 agentic AI 与文本基化工流程建模工具结合，提供“单步式”从抽象设计到可执行代码的闭环；

**🔧 技术方法**

技术核心包括：大语言模型的 in‑context 学习、Chemasim 语法知识注入、VS Code 插件与模拟引擎交互、两代理分工与迭代调试；

**📊 数据集**

使用基于工艺案例的合成数据（Reaction‑separation、Pressure‑swing、Heteroazeotropic），无公开数据集；

**📈 对比分析**

通过三组典型案例（反应‑分离、压力摆动蒸馏、异相蒸馏）评估；模型能生成语法正确的 Chemasim 文件，模拟收敛率高，结果与手工估算相符，性能优于传统单代理或无工具辅助的 LLM 方案；

**⚠️ 局限性**

限制包括：对复杂热力学系统的分析不足、缺乏经济优化与设备选型、对失败仿真缺乏自动回退机制、需手动提供大量语法上下文、对新工艺知识学习依赖外部数据库

---

## 205. Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation

**arXiv ID:** 2603.12845 | [PDF](https://arxiv.org/pdf/2603.12845v1)

**作者:** Fei Wang `[一作]` (Hefei University of Technology), Jingwen Yang `[通讯]` (Hefei University of Technology)

**通讯引用:** 2420 | [OpenAlex ID](https://openalex.org/A5050543532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于酶-底物反应桥接适配器ERBA，利用分阶段的机制将酶序列与底物和活性位点结构融合，以预测k_cat、K_m和K_i。

**💡 创新点**

创新点在于将酶动力学建模为分阶段的条件过程，分别采用分子识别交叉注意力MRCA和几何感知专家混合器G-MoE，并通过分布对齐ESDA保持预训练PLM的生物化学先验。

**🔧 技术方法**

使用Transformer式蛋白质语言模型（如ESM2、ProtT5、Ankh3）作为主干，结合MRCA、G-MoE和ESDA模块，并在log_10空间使用异方差高斯损失进行回归。

**📊 数据集**

采用BRENDA和SABIO-RK公开数据库的酶底物实验数据，覆盖k_cat、K_m和K_i共计约55k条记录。

**📈 对比分析**

与现有SOTA方法（DLKcat、TurNup、UniKP、MPEK、EITLEM、CataPro、CatPred）进行对照实验，ERBA在k_cat、K_m、K_i上分别实现R^2 0.54/0.61/0.78、PCC 0.74/0.79/0.78，明显优于对照组，且在OOD测试中表现更稳健。

**⚠️ 局限性**

局限性包括对某些酶类别（如EC-6）预测仍不理想，依赖高质量的3D结构预测，且对大规模PLM的微调计算成本较高。

---

## 206. A Feasibility-Enhanced Control Barrier Function Method for Multi-UAV Collision Avoidance

**arXiv ID:** 2603.13103 | [PDF](https://arxiv.org/pdf/2603.13103v1)

**作者:** Qishen Zhong `[一作]` (South China University of Technology), Pingan Fang `[通讯]` (South China University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种可行性增强的控制障碍函数（FECBF）框架，用以解决多UAV密集场景下的碰撞规避问题，并通过去中心化的二次规划实现分布式控制；

**💡 创新点**

首次从内部兼容性角度分析CBF约束冲突，推导出符号一致性约束，利用此约束显著提升CBF-QP的可行性；

**🔧 技术方法**

采用控制障碍函数、二次规划、Farkas引理、虚拟状态变量、滑动变量、worst‑case估计等控制理论技术，并在MATLAB中实现仿真，实测平台使用Crazyswarm与NOKOV；

**📊 数据集**

实验数据来源为自行构造的三种仿真场景（50/100/150 UAV）以及Crazyswarm平台的真实飞行实验；

**📈 对比分析**

与两种基线（DRCBF、VOCBF）在成功率、不可行计数、到达时间和计算时间四项指标进行比较。结果显示FECBF在所有场景和规模下均取得最高成功率、最低不可行计数、最短到达时间，计算时间虽略高但仍满足实时要求；

**⚠️ 局限性**

局限性包括需手动调参（β、ζ、λ等）、在极高延迟或极大密度下仍有一定不可行风险、仅考虑速度与姿态约束，未覆盖动态障碍物与系统不确定性。

---

## 207. Multimodal OCR: Parse Anything from Documents

**arXiv ID:** 2603.13032 | [PDF](https://arxiv.org/pdf/2603.13032v1)

**作者:** Handong Zheng `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38830 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Multimodal OCR (MOCR) 体系，统一将文本、表格、公式以及图表、图解等视觉符号作为解析目标，输出可执行的 SVG 代码，形成可重用的结构化文档表示。

**💡 创新点**

核心创新在于：① 把文档图形视为一等参量，生成可渲染代码而非仅裁剪像素；② 通过统一的高分辨率视觉编码器+轻量 LLM 解码器，实现一次模型即可完成多种文档解析任务；③ 构建大规模数据引擎（PDF、网页、原生 SVG）并采用图像‑代码对齐与规范化，解决非唯一代码和视觉定位难题；④ 采用 OCR Arena 基于 LLM 的 Elo 评测，克服传统字符串匹配的不足。

**🔧 技术方法**

使用 1.2B 参数视觉编码器（训练自始至终）、1.5B 参数 Qwen2.5 解码器、分阶段预训练+指令微调、逐步提高分辨率、图像‑代码规范化与渲染验证、Elo 评测框架。

**📊 数据集**

数据集涵盖：① 通过 dots.ocr 自动标注的 PDF 文档；② 从网页抓取并渲染的页面；③ 原生 SVG 资产（经过清洗、去重、复杂度平衡）；⑤ 通用视觉与 OCR 监督数据。评测使用：olmOCR-Bench、OmniDocBench 1.5、XDocParse、UniSVG、ChartMimic、Design2Code、GenExam、SciGen、ChemDraw 等。

**📈 对比分析**

与开源 OCR 系统相比，dots.mocr 在 Elo 评测中取得 1124.7（仅次于 Gemini 3 Pro），在结构化图形解析中 dots.mocr‑svg 在 ISVGEN 分数上超过 OCRVerse 和 Gemini 3 Pro（整体 0.902 vs 0.763/0.735），表明模型在文本识别、布局、表格、公式以及图表、图解等多模态内容上都有显著提升。

**⚠️ 局限性**

局限性包括：① 文本与图形解析仍需分两次推理；② 仅支持 SVG 作为可执行代码，无法覆盖更专业的图形格式（如 TikZ、CAD、化学结构等）；③ 在极低分辨率或严重失真扫描、手写内容等极端情况仍会出现错误；④ 需要大规模算力与复杂的数据预处理，部署成本较高。

---

## 208. Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data

**arXiv ID:** 2603.12686 | [PDF](https://arxiv.org/pdf/2603.12686v1)

**作者:** Zhikai Zhang `[一作]` (Tsinghua University), Li Yi `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种名为 LATENT 的框架，通过收集人体在网球中使用的前手、后手等运动片段，学习并实现了全身机器人在实际环境中连续回球的能力。

**💡 创新点**

创新点包括：① 将不完整、精度不足的人体运动片段构建成可纠正的潜在动作空间；② 设计潜在动作壁垒与手腕纠正机制，保证动作既能完成任务又保持自然；③ 采用动力学随机化与观测噪声，实现从仿真到真实机器人的无缝迁移。

**🔧 技术方法**

技术方法包括层次强化学习（PPO）、潜在动作空间与在线蒸馏、变分信息瓶颈（VAE）、高频低层控制（50 Hz）、动力学随机化与观测噪声模拟。

**📊 数据集**

数据集来自五名业余网球选手在 3 m × 5 m 运动捕捉系统中录制的 5 小时原始运动片段，并通过 LocoMuJoCo 将其映射为机器人可执行的动作。

**📈 对比分析**

与 PPO、MotionVAE、AMP、ASE、PULSE 等基线在仿真中对比，LATENT 在成功率、距离误差、动作平滑度与扭矩等指标上均显著优于对手；在真实 Unitree G1 机器人上完成 20 次连续 rally，成功率高达 90 % 以上，表明方法在实际环境中的有效性。

**⚠️ 局限性**

局限性在于仍需依赖外部运动捕捉与 3D 打印连接；训练目标仅为单手回球，未覆盖双人对战策略；手腕动作的精准度有限，且动力学随机化参数需人工调试。

---

## 209. CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment

**arXiv ID:** 2603.12722 | [PDF](https://arxiv.org/pdf/2603.12722v1)

**作者:** Kaifan Zhang `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**通讯引用:** 40124 | [OpenAlex ID](https://openalex.org/A5101785348)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CognitionCapturerPro框架，利用EEG与图像、文本、深度、边缘四模态数据，通过不确定性加权遮罩、模态专家编码器、跨模态融合编码器、共享树与头对齐模块以及SDXL‑Turbo+IP‑Adapter实现高保真视觉解码。

**💡 创新点**

创新点包括：① 动态不确定性加权遮罩抑制噪声；② 结合跨模态注意力与随机模态掩码的模态专家编码器与融合编码器；③ 轻量化共享树对齐（STH‑Align）替代复杂扩散先验；④ 多模态扩展与预训练扩散模型相结合，显著提升检索与重建性能。

**🔧 技术方法**

技术要点：多模态对比学习、可变焦模糊不确定性加权、跨模态Transformer融合、共享网络对齐、预训练扩散生成模型（SDXL‑Turbo）与IP‑Adapter。

**📊 数据集**

使用数据集：THINGS‑EEG 与 THINGS‑MEG 两大规模脑电/磁脑图像对照数据集。

**📈 对比分析**

与NICE、ATS、UBP等方法在零样本检索与重建任务中对比，CogCapPro在Things‑EEG上达到Top‑1 61.2%、Top‑5 90.8%，在Things‑MEG上达到Top‑1 31.8%、Top‑5 64.6%，且在PixCorr、SSIM、CLIP等重建指标上均优于现有基线。

**⚠️ 局限性**

局限性：仍需提升单次试验的信噪比与实时性，前额叶高阶认知信息利用不足，评估指标缺乏与神经科学一致的客观标准，模型对细粒度特征的还原仍有限。

---

## 210. Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies

**arXiv ID:** 2603.12510 | [PDF](https://arxiv.org/pdf/2603.12510v1)

**作者:** Siddharth Srikanth `[一作]` (University of Southern California), Stefanos Nikolaidis `[通讯]` (University of Southern California)

**通讯引用:** 2314 | [OpenAlex ID](https://openalex.org/A5042344629)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

为Vision‑Language‑Action模型生成多样、现实且攻击性强的指令，并通过这些指令增强模型鲁棒性

**💡 创新点**

将质量多样性(QD)搜索与视觉语言模型相结合，在保持指令任务相关性的同时产生覆盖多种失败模式的多样攻击指令

**🔧 技术方法**

质量多样性(QD)优化、视觉语言模型(VLM)、LLM判别器、句子‑BERT嵌入、强化学习微调

**📊 数据集**

SIMPLERENv、LIBERO‑Goal任务集（10个目标）以及真实机器人演示数据

**📈 对比分析**

与Embodied Red‑Team (ERT)与Rephrase基线对比，在多样性、人类可读性以及未见指令的成功率上均优于基线，Fine‑tuning后VLA在未见攻击指令上提升约10–25%，真实机器人实验也验证了提升

**⚠️ 局限性**

需要大量VLA roll‑out计算，迭代次数受限；指令生成未利用VLA训练反馈；模型对极端指令仍可能失效

---

## 211. Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods

**arXiv ID:** 2603.13077 | [PDF](https://arxiv.org/pdf/2603.13077v1)

**作者:** Yihang Zhou `[一作]` (Imperial College London), Sibo Cheng `[通讯]` (CEREA, ENPC, EDF R&D, Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一套基于稀疏传感器和深度学习的屋顶风场实时重建框架，并与传统克里金插值方法进行比较；

**💡 创新点**

其创新点包括：①直接使用实验PIV数据训练模型，避免CFD的偏差；②引入多方向训练与QR基POD传感器优化，提升跨风向泛化和鲁棒性；③系统对比UNet、ViTAE、CWGAN三种主流深度学习架构；

**🔧 技术方法**

采用的技术包括：三种深度学习模型（UNet、Vision Transformer Autoencoder、Conditional Wasserstein GAN）、克里金插值、POD+QR传感器优化，以及多种评价指标（SSIM、MG、NMSE、FAC2）和传感器扰动与预/后平均策略；

**📊 数据集**

使用的数据集来自东京工业大学风洞的粒子图像测速（PIV）实验，覆盖0°、22.5°、45°三风向，各风向多组独立实验样本，15×15格网，包含时间序列的二维风速分布；

**📈 对比分析**

在单方向训练下，克里金插值在稀疏传感器条件下优于深度学习；而在多方向训练下，三种深度学习模型均优于克里金，UNet在几何误差最小、CWGAN在结构相似度最高，ViTAE兼顾效率与准确；QR优化显著提升鲁棒性，深度学习方法相较克里金在推理速度上快数倍；

**⚠️ 局限性**

局限性包括：①仅针对单一建筑几何和单一高度的风洞实验；②训练数据仅覆盖三风向，难以推广到更广泛的风向；③QR基传感器优化依赖特定数据集，需重新计算以适应不同场景；④缺乏现场自然大气条件验证，且对单方向训练的鲁棒性仍不足。

---

## 212. Optimal Enumeration of Eulerian Trails in Directed Graphs

**arXiv ID:** 2603.12894 | [PDF](https://arxiv.org/pdf/2603.12894v1)

**作者:** Ben Bals `[一作]` (CWI), Matei Tinca `[通讯]` (Vrije Universiteit)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种在有向图（包括多重图）中枚举所有欧拉行走的算法，时间复杂度为 O(m+z)，其中 m 为边数，z 为输出行走数。

**💡 创新点**

创新点在于引入两条简单压缩规则（单出边收缩与自环剔除），通过状态树与 Hierholzer 算法的组合，直接枚举欧拉行走并压缩状态树，使得整体时间达到理论最优，并且算法结构极为简洁。

**🔧 技术方法**

核心技术包括：
- 状态树（State Tree）作为搜索框架，记录所有前缀；
- Hierholzer 子程序用于快速完成前缀到欧拉行走的扩展；
- 两类压缩规则（Type 1 与 Type 2）实现对剩余图的即时压缩；
- 逆向搜索（Reverse Search）思路与压缩规则结合的 DFS 方式；
- 对有向多重图的特殊处理（多重边计数、交叉边唯一性）。

**📊 数据集**

文中未使用具体实验数据集，重点在理论证明与复杂度分析；若需实验可借用生物信息学中常见的 De Bruijn 多重图或随机有向图。

**📈 对比分析**

与现有方法比较：
- 相比 BEST 定理的计数实现（需要构造 Laplacian 矩阵、O(n²) 预处理），在 m=o(n²) 时性能更优；
- 超越 Conte 等人 (m·z) 计数/枚举算法，提供真正的 O(m+z) 枚举时间；
- 与 Kurita–Wasa 的无向图枚举相比，扩展到有向图并保持同样的最优复杂度。

**⚠️ 局限性**

局限性：
- 仅适用于弱连通的有向图（或多重图）且需满足欧拉行走存在；
- 需要手动指定起点/终点（但可通过循环调整简化）；
- 对大规模多重图的实际实现仍需谨慎处理多重边计数与压缩规则的实现细节；
- 该方法主要关注枚举/计数，未涉及加权或带约束的欧拉行走问题。

---

## 213. Development of a Methodology for the Automated Spatial Mapping of Heterogeneous Elastoplastic Properties of Welded Joints

**arXiv ID:** 2603.12892 | [PDF](https://arxiv.org/pdf/2603.12892v1)

**作者:** Robert Hamill `[一作]` (United Kingdom Atomic Energy Authority), Fabrice Pierron `[通讯]` (MatchID NV)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种自动化空间参数化的虚拟场方法，用于映射焊缝的异质弹塑性力学属性，并通过全场光学测量实现。

**💡 创新点**

结合自动空间参数化、基于高斯径向基函数的参数化、力学平衡指标（EGI、FRE）与全局优化，实现了无需先验信息即可识别焊缝的空间属性。

**🔧 技术方法**

虚拟场方法（VFM）、基于高斯径向基函数的空间参数化、全场数字图像相关（DIC）、MATLAB模式搜索优化、ANSYS有限元模拟和合成图像生成。

**📊 数据集**

采用合成数据集：基于ANSYS的非线性FE模型模拟的离散焊缝几何，随后通过图像扭曲生成与DIC相似的合成图像。

**📈 对比分析**

通过比较单高斯函数与双高斯函数参数化以及不同窗口大小的EGI指标，最终在合成案例中实现了约6% 最大误差、10% FRE误差，计算时间为数小时。

**⚠️ 局限性**

限制包括：只识别屈服强度的空间变化，硬化模量保持均匀；对噪声敏感；参数化函数有限，无法完全捕捉梯度；实验验证尚缺乏；需进一步优化权重与窗口选择。

---

## 214. Delta1 with LLM: symbolic and neural integration for credible and explainable reasoning

**arXiv ID:** 2603.12953 | [PDF](https://arxiv.org/pdf/2603.12953v1)

**作者:** Yang Xu `[一作]` (Southwest Jiaotong University), Hailing Guo `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5111131077)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端可解释的神经-符号推理管线，将Δ_1自动定理生成器与大型语言模型（LLM）结合，生成可验证的最小不满足子集，并由LLM转化为可读的自然语言解释。

**💡 创新点**

创新点在于：①通过Full Triangular Standard Contradiction（FTSC）构造定理，保证生成过程无搜索、无随机性、严格可证；②实现“从构造到可解释”即在定理生成的同时自动生成可解释性说明；③将符号推理与语言解释实现一体化，首次实现可验证且可解释的神经-符号推理。

**🔧 技术方法**

技术包括：Δ_1确定性定理生成器（基于FTSC构造最小不满足子集），LLM前端的谓词抽取，LLM后端的证明解释与排名，以及整体流水线的模块化架构。

**📊 数据集**

主要使用的“数据集”为人工构造的案例语料（医疗、合规、监管、合同等领域的规则描述），通过LLM抽取原子谓词；未涉及公开大规模标注数据集，实验基于人工生成的逻辑实例。

**📈 对比分析**

与传统自动定理证明器（如Vampire、E）及现代神经-符号系统（Logic-LM++、DeepProbLog）进行对比：Δ_1在构造阶段实现O(n^3)时间复杂度、确定性生成所有最小不满足子集；LLM解释层提供可解释性评分与重排序；实验结果显示在案例域中既具备完备性和最小性，又能输出人类可读的解释，性能稳定且可复制。

**⚠️ 局限性**

局限性包括：①生成的定理数量呈阶乘增长，规模受限；②对LLM的解释依赖其训练质量，可能产生解释偏差；③目前缺乏大规模真实数据评估与人类主观评价，且对多模态/高阶逻辑的扩展尚未实现。

---

## 215. HyperCroc: End-to-End Open-Source RISC-V MCU with a Plug-In Interface for Domain-Specific Accelerators

**arXiv ID:** 2603.12308 | [PDF](https://arxiv.org/pdf/2603.12308v1)

**作者:** Philippe Sauter `[一作]` (Integrated Systems Laboratory, ETH Zurich), Luca Benini `[通讯]` (Department of Electrical, Electronic, and Information Engineering, University of Bologna)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出HyperCroc，一个在Croc OS MCU平台上集成HyperBus控制器和iDMA的可扩展微控制器平台，支持外部大容量存储并可轻松插拔域特定加速器；

**💡 创新点**

创新点在于将低功耗、低引脚数的HyperBus接口与高速DMA引擎结合，扩展了原本仅支持内部SRAM的Croc MCU，使其具备大数据集传输能力，并保留完整的端到端开源RTL‑to‑GDS流程；

**🔧 技术方法**

使用了Ibex核心、HyperBus硬件宏、iDMA数据搬运引擎、IHP开放130nm PDK以及完整的EDA工具链；

**📊 数据集**

未使用具体数据集，重点是硬件功能验证和性能测量；

**📈 对比分析**

通过与首批Croc硅片测量结果对比，Croc在130nm工艺下最高频率为72MHz，HyperCroc目标频率100MHz，外部内存吞吐量可达800MB/s，证明了设计的可行性和性能提升；

**⚠️ 局限性**

限制包括对HyperBus PHY的时钟域依赖、仍需进一步验证与实际加速器集成的时序与功耗、以及在130nm工艺下的面积与成本。

---

## 216. coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**arXiv ID:** 2603.12829 | [PDF](https://arxiv.org/pdf/2603.12829v1)

**作者:** Chunhan Li `[一作]` (China University of Petroleum), Zhengzhe Liu `[通讯]` (Lingnan University)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5014239420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 coDrawAgents 的交互式多代理对话框架，包含 Interpreter、Planner、Checker 和 Painter 四个专门化代理，用于实现文本到图像的可控合成。

**💡 创新点**

创新点在于引入动态代理对话实现闭环迭代、采用分层规划（divide-and-conquer）降低布局复杂度、通过 Checker 提供显式错误纠正，并在 Painter 中使用视觉上下文进行逐步渲染。

**🔧 技术方法**

利用大语言模型（LLM）进行文本解析与规划（VCoT），配合布局到图像（L2I）模型 Flux、3DIS 进行渲染，并通过视觉上下文和属性校验实现错误检测。

**📊 数据集**

在 GenEval 和 DPG-Bench 两个公开基准上进行评估。

**📈 对比分析**

相较于 DALL‑E 3、FLUX、GoT 等最先进模型，coDrawAgents 在文本‑图像一致性、空间准确度和属性绑定方面取得最高分（例如 GenEval 整体得分 0.94，DPG‑Bench 整体得分 85.17）。

**⚠️ 局限性**

主要局限包括多代理迭代导致计算开销增加、依赖底层 T2I/L2I 模型的质量、LLM 可能出现幻觉或不确定性，以及尚未扩展到 3D 场景。

---

## 217. CSE-UOI at SemEval-2026 Task 6: A Two-Stage Heterogeneous Ensemble with Deliberative Complexity Gating for Political Evasion Detection

**arXiv ID:** 2603.12453 | [PDF](https://arxiv.org/pdf/2603.12453v1)

**作者:** Christos Tzouvaras `[一作]` (University of Ioannina), Athanasios Voulodimos `[通讯]` (National Technical University of Athens)

**通讯引用:** 6311 | [OpenAlex ID](https://openalex.org/A5062640206)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对SemEval-2026 Task 6的政治访谈回答清晰度分类，提出一种两阶段异构双LLM集成方法，先用自一致性与加权投票得到初始预测，再通过Deliberative Complexity Gating（DCG）门控机制进行后置校正。

**💡 创新点**

创新点在于：①使用跨模型行为信号（Gemini响应长度+Grok自一致性）构建自适应阈值门控；②采用“先判细化逃避类别后映射至清晰度”的策略减小误差堆积；③在投票中采用不对称加权（Grok多样本权重+Gemini块权重）以平衡两模型的优势。

**🔧 技术方法**

技术实现包括：异构双LLM（Grok与Gemini）自一致性推理（k=5），链式思维（CoT）提示，非对称加权投票，基于响应长度和一致性阈值的DCG门控，及后置JSON重算最终标签。

**📊 数据集**

使用的数据集为SemEval-2026 Task 6的官方英文政治访谈问答数据集（训练/验证集308条，评测集237条），所有推理均无监督、无微调。

**📈 对比分析**

与单模型比较：Grok单模型k=5 Macro‑F1为0.8264，Gemini单模型为0.8083；Stage‑1集成提升至0.8122；Stage‑2加入DCG后在评测集Macro‑F1升至0.8505，排名41队伍中的第3名，显示显著性能提升。

**⚠️ 局限性**

局限性包括：依赖专有LLM（Grok、Gemini），导致可复现性和成本受限；DCG依赖响应长度这一域内代理特征，可能不适用于其它模型或领域；门控需批量推理，无法实现单实例即时推断。

---

## 218. FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking

**arXiv ID:** 2603.12758 | [PDF](https://arxiv.org/pdf/2603.12758v1)

**作者:** Cheng Ju `[一作]` (Chiba University), Akio Namiki `[通讯]` (Chiba University)

**通讯引用:** 3162 | [OpenAlex ID](https://openalex.org/A5084860488)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的后关联纠错框架FC-Track，专门针对重叠导致的身份错误进行在线修正；

**💡 创新点**

通过IoA阈值过滤不可靠的外观更新并在重叠帧中局部重新匹配，显著减少长期身份漂移；

**🔧 技术方法**

使用交并比(IoU)、交面积比(IoA)、余弦相似度、Kalman滤波与Hungarian算法等传统追踪组件；

**📊 数据集**

在MOT17和MOT20两个公开视频跟踪基准上进行实验；

**📈 对比分析**

相较于TrackTrack及多种主流在线追踪器，FC-Track在MOT17上达81.73 MOTA、82.81 IDF1、66.95 HOTA（FPS≈5.7），在MOT20上达到77.52 MOTA、80.90 IDF1、65.67 HOTA（FPS≈0.6），并将ID切换的平均持续时间和长期切换比例分别从22.88→18.33帧、36.86%→29.55%显著下降；

**⚠️ 局限性**

主要限制在于对阈值选择敏感，MOT20上帧率较低，且仅在两种数据集上验证，未进一步探究与更复杂场景或更大模型的结合效果。

---

## 219. ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning

**arXiv ID:** 2603.12740 | [PDF](https://arxiv.org/pdf/2603.12740v1)

**作者:** Shuo Yang `[一作]` (University of Melbourne), Eduard Hoy `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ToolTree，一种基于MCTS的工具规划框架；

**💡 创新点**

通过双重评估（预评估与后评估）实现前向与后向剪枝，提升搜索效率与准确率；

**🔧 技术方法**

利用大语言模型作为评估器，结合预评估预测工具适用性、后评估评估执行结果，MCTS搜索并双向剪枝；

**📊 数据集**

在四个基准上评估：闭集工具规划（GTA、m&m）与开放集工具规划（ToolBench、RestBench）；

**📈 对比分析**

与多种基线（Zero-shot、ReAct、CoT、Tree-of-Thought、A*、MCTS、LATS等）对比，ToolTree平均提升约10%（F1/通过率/赢率），并在效率上取得最高的准确率/秒；

**⚠️ 局限性**

仍受LLM评估质量影响，需在大规模工具库与复杂任务上进一步验证，且运行时相对较慢（相较于贪婪方法）。

---

## 220. AI Model Modulation with Logits Redistribution

**arXiv ID:** 2603.12755 | [PDF](https://arxiv.org/pdf/2603.12755v1)

**作者:** Zihan Wang `[一作]` (University of Queensland), Guangdong Bai `[通讯]` (University of Queensland)

**通讯引用:** 1832 | [OpenAlex ID](https://openalex.org/A5015858067)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于logits重分配的模型调制范式，允许在不重新训练或改动模型结构的情况下，单一预训练模型实现多级效用与关注度控制。

**💡 创新点**

首次将logits重分配与概率理论结合，实现了训练数据无关、轻量化、可解释的效用与关注度调制，并给出了保持 logits 顺序的概率解析。

**🔧 技术方法**

采用在 logits 上叠加高斯或折叠正态噪声的调制函数，结合概率分析与控制参数，实现在 ResNet、SegFormer、Llama 等预训练模型上的效用与关注度调整。

**📊 数据集**

在视觉任务上使用 CIFAR‑10、CIFAR‑100、ADE20K 与 KITTI 数据集；在语言任务上使用 GSM8K 与 MMLU 基准进行实验。

**📈 对比分析**

通过与原始预训练模型的直接对比，评估准确率/mIoU 与语言推理正确率；实验显示效用调制可实现从 94% 到 20% 的准确率降阶，关注度调制可在保持整体 mIoU <0.5% 下降的前提下提升目标类别（如 Person）精准率 5% 以上。

**⚠️ 局限性**

仅调节 logits，可能导致语言输出过度冗余或轻微错误；对噪声分布的超参数选择敏感；未验证对所有任务或安全敏感场景的普适性。

---

## 221. Evaluation of TCP Congestion Control for Public High-Performance Wide-Area Networks

**arXiv ID:** 2603.12660 | [PDF](https://arxiv.org/pdf/2603.12660v1)

**作者:** Fatih Berkay Sarpkaya `[一作]` (Nokia Bell Labs), Shivendra Panwar `[通讯]` (NYU Tandon School of Engineering)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

评估了 TCP CUBIC、BBRv1、BBRv3 三种拥塞控制算法在公共高性能广域网（HP‑WAN）上的大规模数据传输效率和可预测性，并比较了 Linux 内核转发、DPDK 快速转发和 DPDK 形状化三种数据路径配置

**💡 创新点**

提出在公共 HP‑WAN 上采用每个虚拟电路（VC）形状化配合 BBRv1，可显著提升流完成时间（FCT）的预测性与效率，并展示了形状化对消除非拥塞性丢包的关键作用

**🔧 技术方法**

使用 FABRIC 研究测试平台的 L2PTP VC、Linux 内核网络栈、DPDK 软核转发、DPDK 形状化工具以及多线程 iperf3 产生流量，配合多流、单流、不同 CC 算法的实验设置

**📊 数据集**

在 FABRIC 测试平台上传输 1 TB 文件，使用 20 Gb/s 与 80 Gb/s 的 VC 进行实验，构建多站点拓扑（NCSA、MICH、KANS、INDI）进行跨站点大容量数据传输

**📈 对比分析**

通过测量 FCT 效率（理想 FCT 与实际 FCT 的比值）和 FCT 方差来比较配置与算法；结果显示：DPDK 形状化配合 BBRv1 的 FCT 效率最高、方差最小；DPDK 快速转发相较 Linux 内核减小丢包率但仍有较高的带宽开销；BBRv3 在形状化配置下可与 BBRv1 相媲美，但在非形状化条件下更易出现性能波动

**⚠️ 局限性**

实验受限于 FABRIC 的非拥塞性丢包难以完全消除；BBRv1 与丢包敏感的 TCP 变体共存不佳；实验仅覆盖 80 Gb/s 范围，结果可能不适用于更高带宽或不同公共 HP‑WAN 环境

---

## 222. As Language Models Scale, Low-order Linear Depth Dynamics Emerge

**arXiv ID:** 2603.12541 | [PDF](https://arxiv.org/pdf/2603.12541v1)

**作者:** Buddhika Nettasinghe `[一作]` (University of Iowa), Geethu Joseph `[通讯]` (Delft University of Technology)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5012419698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在给定提示的上下文中，将Transformer的层级动态建模为低阶线性系统（LLV），并验证其能够准确预测层级敏感度和指导多层干预；

**💡 创新点**

提出了在局部上下文中可辨识的低维线性近似，并发现模型规模越大，该近似的可辨识度越好；利用该近似实现最低能耗的多层干预，优于传统启发式方案；

**🔧 技术方法**

使用系统辨识、局部线性化、概念方向估计、Krylov基构造、Jacobian-Vector Products或有限差分求雅可比、最小能量控制求解等技术；

**📊 数据集**

在十个二分类NLP任务上进行实验，包括情感分析、问答、文本蕴含、毒性、讽刺、仇恨等数据集；

**📈 对比分析**

通过Spearman和Pearson相关系数对比LLV预测的gain曲线与全模型实测gain曲线；在GPT‑2‑large上比较多层控制能耗，LLV最优方案能耗比均匀注入低约2–5倍，优于单层或随机干预；

**⚠️ 局限性**

仅为局部近似，依赖于特定提示轨迹；未验证对不同模型族的普适性；大规模高维任务的可扩展性和鲁棒性仍待进一步研究。

---

## 223. Task-Specific Knowledge Distillation via Intermediate Probes

**arXiv ID:** 2603.12270 | [PDF](https://arxiv.org/pdf/2603.12270v1)

**作者:** Ryan Brown `[一作]` (Oxford Internet Institute, University of Oxford), Chris Russell `[通讯]` (Oxford Internet Institute, University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于教师隐藏层探针的知识蒸馏框架，利用探针生成的软标签训练小型模型，从而提升多选推理任务的性能。

**💡 创新点**

创新点在于通过训练轻量级探针从教师内部表征中提取更干净、任务对齐的软标签，消除了传统logit蒸馏中输出层噪声的影响，并且无需对教师进行微调。

**🔧 技术方法**

使用轻量级MLP/线性探针、温度缩放的KL蒸馏、无监督的CCS探针训练、标签平滑与校准分析等技术。

**📊 数据集**

实验基准包括 AQuA‑RAT、ARC Easy/Challenge、MMLU 以及多种教师模型（Qwen2.5‑7B‑Instruct、Phi‑3‑mini‑4k、TinyLlama‑1.1B）。

**📈 对比分析**

与监督学习、标准 logit‑KD、Feature‑KD、Patient‑KD 等方法对比，Probe‑KD 在所有四个基准上均取得最高准确率，低样本场景提升可达 1–5.6%（最高 5.6%）。

**⚠️ 局限性**

局限性包括：仅针对多选分类任务；需要访问教师隐藏状态，无法用于黑盒API；探针训练需要缓存大量隐藏向量，存储成本高；对生成式任务的推广性受限。

---

## 224. Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models

**arXiv ID:** 2603.12977 | [PDF](https://arxiv.org/pdf/2603.12977v1)

**作者:** Yijun Quan `[一作]` (WMG University of Warwick), Giovanni Montana `[通讯]` (WMG University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对冻结的基础模型（feature extractor）上训练的岭回归（ridge）线性头，提出了能够在联邦学习环境下实现精确持续忘记（add/delete）请求的通信协议。

**💡 创新点**

创新点在于：①利用岭回归的闭式解，仅需维护两条二阶充分统计量（Gram矩阵和特征-标签矩阵）；②设计两种服务器端实现（精确求逆与Sherman–Morrison–Woodbury迭代更新），实现单轮通信且保证与中心化重新训练完全一致；③给出贝叶斯后验零KL证据，证明在任意删改流下保持分布一致性。

**🔧 技术方法**

技术主要包括：岭回归闭式解、充分统计量的可加性、QR分解压缩二阶矩、SMW逆更新、Cholesky求逆、浮点数精度控制（fp64）以及安全聚合与差分隐私可选。

**📊 数据集**

实验使用四个基准数据集：CIFAR‑10、CIFAR‑100、FeMNIST、Sentiment140，采用冻结的 DINOv2‑ViT B/14 或 RoBERTa 作为特征提取器，特征维度均为 768。

**📈 对比分析**

与 FedAvg、FATS、Exact‑Fun 等基准对比：在准确率上两种变体与中心化岭回归几乎一致（<10⁻⁹ 绝对误差），在单点删除和大块删除场景下，单轮通信导致的延迟比多轮 FedAvg、Exact‑Fun 低数倍甚至数十倍，且不需要保存多份检查点。

**⚠️ 局限性**

局限性：仅适用于冻结的特征提取器；若后端模型本身包含可忘记数据，则需要另外的背板忘记机制；二阶统计量可能泄露特征分布信息，需配合安全聚合或差分隐私；Variant B 在长期降维更新时可能出现浮点漂移，需要定期重置。

---

## 225. Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach

**arXiv ID:** 2603.12852 | [PDF](https://arxiv.org/pdf/2603.12852v1)

**作者:** Falko Kähler `[一作]` (Hamburg University of Technology), Thorsten Schüppstuhl `[通讯]` (Hamburg University of Technology)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5010136138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于视觉的分层分类框架，用于自动监测研磨刀片的磨损状态，包括使用状态、磨损类型和严重程度的判别。

**💡 创新点**

创新点在于将任务拆分为多层次的分离决策（使用 vs. 失效、磨损形态、磨损严重度），并引入逻辑一致性检查、基于EfficientNetV2的迁移学习以及Grad‑CAM可视化验证，显著提升了模型的鲁棒性和可解释性。

**🔧 技术方法**

使用的技术包括EfficientNetV2深度学习模型（S、L 版本）、迁移学习、数据增强、梯度加权类激活映射（Grad‑CAM）、置信度阈值与冲突检测机制。

**📊 数据集**

使用的数据集包含 13,240 张来自 105 只研磨刀片的手工标注图像，涵盖新旧状态、是否有刀片断裂、刀片形态（矩形/凹/凸）以及严重程度（部分/完整）等多维标签，训练/验证/测试集按工具唯一划分。

**📈 对比分析**

通过逐层评估与整体评估比较，子模型在使用状态、刀片形态和断裂检测上分别达到 98.6%、95.4% 及 93.8% 的准确率，层级整体准确率为 87.5%。模型在 ROC 曲线上的 AUC 接近 0.99–1.00，表明分类性能优异；相较于单一大模型，分层方法提升了特征可解释性，但误差在层级链中会累积，整体准确率略低。

**⚠️ 局限性**

主要局限包括：层级链误差传播导致整体准确率受限；对凹形严重度的判别仍有 11% 的误差；对凸形完整与部分的区分存在混淆；数据集规模和多样性有限，难以覆盖所有极端磨损场景；对光照、图像捕获角度的依赖需进一步优化。

---

## 226. Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception

**arXiv ID:** 2603.13176 | [PDF](https://arxiv.org/pdf/2603.13176v1)

**作者:** Dingcheng Huang `[一作]` (Mechatronics Research Laboratory, Massachusetts Institute of Technology), Kamal Youcef-Toumi `[通讯]` (Mechatronics Research Laboratory, Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于信息奖励与计算成本平衡的感知调度框架，在人机协作环境中动态激活多模态感知模块，实现实时且资源高效的场景感知。

**💡 创新点**

首次将关联度概念与信息理论结合，构建模块奖励模型，在每帧根据运动、相关性及信息增益评估是否激活感知模块，并提出关键帧准确率评估指标。

**🔧 技术方法**

利用帧差分和色彩直方图检测运动与场景变化，Kalman滤波预测边界框，基于熵的姿态不确定度建模，奖励-成本优化选择模块，并在YOLO和MMPose上实现。

**📊 数据集**

自行收集了三类视频（室内阅读、进食、行走）共计约900帧进行实验，未使用公开数据集。

**📈 对比分析**

与传统并行感知管线和Oracle调度进行对比，实验显示平均延迟降低27.52%，MMPose激活召回提高72.73%，关键帧准确率达98%，YOLO召回基本保持在1.0。

**⚠️ 局限性**

对高延迟姿态估计模块仍有召回不足，奖励模型对阈值和场景特征依赖较强，难以直接推广至更复杂或多模态的感知任务。

---

## 227. Autonomous Integration and Improvement of Robotic Assembly using Skill Graph Representations

**arXiv ID:** 2603.12649 | [PDF](https://arxiv.org/pdf/2603.12649v1)

**作者:** Peiqi Yu `[一作]` (Carnegie Mellon University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2475 | [OpenAlex ID](https://openalex.org/A5040156274)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了基于 Skill Graph 的机器人装配系统框架，实现了从语义级规划到低层执行的全流程自动化；通过视频提取零经验任务描述，并在真实双臂 Yaskawa GP4 机器人上执行 LEGO 组装。

**💡 创新点**

1）将技能抽象为带有语义（verb+noun）、预置/后置条件和评估器的图节点，实现语义层与执行层无缝衔接；2）利用 Vision‑Language 模型实现零经验任务提取；3）通过系统日志的结构化记录实现闭环评估与失败模式检测；4）多机器人协同执行通过 TPG/APEX‑MR 实现；5）参数与策略自适应提升执行鲁棒性。

**🔧 技术方法**

Skill Graph 结构、RRT‑Connect/BITStar 轨迹规划、Vision‑Language Models（Gemini API、DINOv2）、Eye‑in‑Finger 摄像头、第三视角摄像头、APEX‑MR 与 Temporal Plan Graph（TPG）多机器人调度、数字孪生日志系统、参数与策略自适应机制。

**📊 数据集**

自制 LEGO 组装任务数据集，包括两台 Yaskawa GP4 机器人的执行日志、用于任务提取的人类演示视频，以及对应的 Gazebo 仿真数据；未使用公开公共数据集。

**📈 对比分析**

在四种 LEGO 组装结构（Faucet、Fish、Vessel、Guitar）上进行前后对比；改进前成功率 1/5–1/3，后改进后均升至 1/1；存活长度从 9.2–33.7 颗砖提高到 14–36 颗，显著提升成功率与持续组装能力。

**⚠️ 局限性**

仍需人工设计部分预置条件与评估器；依赖高质量 VLM 的泛化能力，可能对不同场景不稳定；多机器人同步调度复杂度较高；未在更大规模或工业真实场景中充分验证；闭环学习主要基于现有日志，长期自适应能力待进一步提升。

---

## 228. Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs

**arXiv ID:** 2603.12996 | [PDF](https://arxiv.org/pdf/2603.12996v1)

**作者:** Bumjun Kim `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无训练的依赖感知并行解码方法（DAPD），利用自注意力构造马尔可夫随机场来捕捉掩码位置间的互依赖，从而在每一步选择近似独立的子集并行解码；

**💡 创新点**

创新点在于将自注意力视为条件独立指示器，构造图结构后通过类似Welsh–Powell的图着色算法选择最大独立集，从而显著减少解码步骤并提升并行度；

**🔧 技术方法**

主要技术包括自注意力图构造、节点度估计、独立集贪婪搜索（Welsh–Powell启发式）以及在掩码低于50%时切换至置信度阈值解码；

**📊 数据集**

实验使用公开的两大离散扩散LLM：LLaDA-8B-Instruct 与 Dream-7B-Instruct，并在数学推理（GSM8K、Math500）、代码生成（MBPP、HumanEval）及指令跟随（IFEval）等任务以及ParallelBench上进行评测；

**📈 对比分析**

相较于传统无训练并行策略（Fast-dLLM、KLASS、EB-Sampler），DAPD在保持或略优的准确率的同时将解码步骤减少约40%~60%，实现3–4倍的速度提升；

**⚠️ 局限性**

局限性包括对阈值和度估计的经验选择，可能在极端依赖结构或非对称注意力模型下性能下降，以及未在大规模多任务场景下验证进一步的可扩展性。

---

## 229. MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization

**arXiv ID:** 2603.12743 | [PDF](https://arxiv.org/pdf/2603.12743v1)

**作者:** Chenyang Zhu `[一作]` (Tsinghua University), Long Chen `[通讯]` (HKUST)

**通讯引用:** 3059 | [OpenAlex ID](https://openalex.org/A5100679798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了知识驱动的概念定制（Knowledge-Aware Concept Customization，KnowCus）任务，能够将多条自然语言知识绑定到目标视觉概念，从而实现高保真度的定制生成。

**💡 创新点**

核心创新在于发现并利用跨模态知识迁移现象：在文本编码器中更新知识即可同步影响生成的视觉输出；引入锚表示（anchor representation）将视觉信息与文本知识桥接；通过可学习的LoRA进行视觉概念学习以及参数偏移式文本知识更新。

**🔧 技术方法**

技术上使用大型语言模型（如Qwen-Image）作为文本编码器，Diffusion Transformer（DiT）作为生成器；视觉概念学习阶段利用LoRA微调；文本知识更新阶段采用查询-答案结构，计算隐藏状态梯度并求解最小二乘得到参数偏移，实现知识注入；评估采用CLIP-I、CLIP-I-Seg、CLIP-T、Pick Score等指标。

**📊 数据集**

构建了首个KnowCusBench基准：35个目标概念（来自DreamBench、CustomConcept101、Unsplash），每个概念生成5条知识（六个视角，人工修正），共199条生成提示，最终产生5,975张评测图像。

**📈 对比分析**

与两种基线对比：Naive-DB（为每条知识分别使用DreamBooth训练）和Enc-FT（先视觉学习后对LLM编码器微调）。在重建任务中，KnowCus在CLIP-I-Seg上优于所有基线，在生成任务中在CLIP-T、Pick Score和训练时间上均取得最佳表现，且整体性能稳定。

**⚠️ 局限性**

局限性包括：仍以单概念为主，难以一次性处理大量概念；文本知识更新需手工构造查询；目前仅支持图像，未扩展到视频或更复杂的多模态任务；模型对极端多知识组合的泛化能力尚待进一步验证。

---

## 230. BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending

**arXiv ID:** 2603.13102 | [PDF](https://arxiv.org/pdf/2603.13102v1)

**作者:** Matteo Ballegeer `[一作]` (Ghent University), Dries F. Benoit `[通讯]` (Ghent University)

**通讯引用:** 1688 | [OpenAlex ID](https://openalex.org/A5089981863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了制造可制造性评估的二维分类法，并基于此生成了20,000个可折叠与展开几何、工具碰撞与展开重叠标注的BenDFM合成数据集，随后在该数据集上对两种主流几何深度学习模型（点云PointNext与拓扑图UV‑Net）进行了基准实验。

**💡 创新点**

①将制造可制造性划分为“配置相关/无关”和“可行性/复杂度”两轴，形成四象限的系统化标签框架；②构建专门针对板金弯曲工艺的合成数据集，包含完整的弯曲序列、工具几何以及多维可制造性标注；③首次在同一数据集上同时评估可行性（二分类）与复杂度（回归）任务，揭示了拓扑图模型在全局约束学习上的优势。

**🔧 技术方法**

利用PythonOCC进行参数化CAD建模与弯曲仿真；使用碰撞检测与展开自交检测生成标签；模型方面采用基于B‑rep的图卷积网络UV‑Net和基于点云的PointNext网络，配合Adam优化、交叉熵/均方误差损失进行训练。

**📊 数据集**

BenDFM（20,000件板金弯曲零件），每件含3D STEP文件、展开版STEP、完整弯曲序列以及配置相关/无关的可制造性与复杂度标签（碰撞、展开自交、翻转次数、展开面积等）。

**📈 对比分析**

通过5次固定随机种子实验，比较UV‑Net与PointNext在四个基准任务（工具碰撞、展开自交、翻转次数、展开面积）上的AUC、准确率、F1、MAE、RMSE、MAPE。结果显示UV‑Net在所有任务上均优于PointNext，AUC从0.84（工具碰撞）提升至0.90（展开自交），准确率从73%提升至82%；回归任务MAE从0.59提升至0.54（翻转次数），展开面积MAE从20.24提升至14.6，MAPE分别从39%提升至36%和从8.3%提升至5.9%。

**⚠️ 局限性**

①配置相关标签难以学习，模型对不同工装/机器的泛化能力差；②弯曲过程的顺序信息未被显式建模，导致全局约束难以捕捉；③仅使用单一固定工装，缺乏跨配置评估；④合成数据对真实生产的外推性尚未验证；⑤缺乏可解释性与人机交互支持。

---

## 231. Rethinking Multiple-Choice Questions for RLVR: Unlocking Potential via Distractor Design

**arXiv ID:** 2603.12826 | [PDF](https://arxiv.org/pdf/2603.12826v1)

**作者:** Xu Guo `[一作]` (Shanghai AI Laboratory), Qipeng Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过系统实验探究多选题中的选项数量与干扰项质量对RLVR训练的影响，并提出 Iterative Distractor Curation (IDC) 框架，动态生成和优化干扰项以提升奖励信号质量，最终显著提升大模型在医疗类多选题上的推理性能。

**💡 创新点**

① 发现训练-测试选项数量不匹配会导致RLVR性能下降；② 强干扰项能有效抑制随机猜测和排除策略；③ IDC 通过自我评估干扰项强度、迭代替换弱干扰项，实现无监督的自我提升；④ 证明强干扰项的生成可由目标模型自身完成，无需外部监督。

**🔧 技术方法**

基于RLVR框架的策略梯度（GRPO）、自我评估干扰项强度的经验度量（ŝ_j）、循环式干扰项生成与替换算法、对抗式硬负样本抽样、对比实验与下游评测。

**📊 数据集**

MedQA、SuperGPQA (Clinical)、MMLU-Pro (Health)、MedXpertQA、PubMedQA 等医疗相关多选题数据集。

**📈 对比分析**

与直接转换为短答案、过滤不可转换题目、模型重写等基线对比。IDC 在 Qwen2-7B 上平均提升约3.3%，在 Llama-3.1-8B 上平均提升约3.3%，在各种下游任务中均实现显著性能提升。

**⚠️ 局限性**

无法事先预测模型生成干扰项的有效性；干扰项质量与模型能力的关系尚不明确；方法对极低质量或语义相似的干扰项需额外检查，增加复杂度。

---

## 232. Generative Horcrux: Designing AI Carriers for Afterlife Selves

**arXiv ID:** 2603.12971 | [PDF](https://arxiv.org/pdf/2603.12971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 233. Sobolev--Ricci Curvature

**arXiv ID:** 2603.12652 | [PDF](https://arxiv.org/pdf/2603.12652v1)

**作者:** Kyoichi Iwasaki `[一作]` (Graduate University for Advanced Studies), Hideitsu Hino `[通讯]` (Institute of Statistical Mathematics)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5023633849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 Sobolev–Ricci Curvature (SRC)，并在社区检测（Sobolev‑Ricci Flow）与图边剪枝（SRC‑MANL）两大应用中验证其可行性。

**💡 创新点**

创新点在于将 Sobolev transport (ST) 的树度量闭式形式用于定义 Ricci 曲率，使得 SRC 在树上与 ORC 等价、在 Dirac 极限下为零，并在无需每条边求解最优传输的情况下实现可扩展计算。

**🔧 技术方法**

使用的技术包括 Sobolev transport、MST/SPT 生成的树度量、闭式 ST 计算、Sobolev‑Ricci Flow (SRF)、SRC‑MANL 边剪算法，以及与 ORC、FastGNN 等基线的对比实验。

**📊 数据集**

实验数据集包括合成网络 (SBM、LFR)、五个真实网络 (如 Facebook、LesMis、Karate 等) 以及十个 4000 点的高维数据集用于边剪枝验证。

**📈 对比分析**

与 ORC 及其他社区检测/边剪枝基线比较时，SRC 在保持相近或更好 ARI / 误删率的同时，计算速度提升 10–100 倍；在社区检测中与 ORC 竞争，在边剪枝中 SRC‑MANL 的精度明显优于 ORC‑MANL。

**⚠️ 局限性**

主要局限在于对树结构的选择（MST/SPT）可能影响结果，且目前仅适用于静态图，缺乏对动态图/时间序列网络的理论与实现。

---

## 234. When Right Meets Wrong: Bilateral Context Conditioning with Reward-Confidence Correction for GRPO

**arXiv ID:** 2603.13134 | [PDF](https://arxiv.org/pdf/2603.13134v1)

**作者:** Yu Li `[一作]` (George Washington University), Zhengling Qi `[通讯]` (George Washington University)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5078483423)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了在GRPO框架下的双边上下文条件化与奖励-置信度校正两种机制，利用同一组样本中正确与错误解的对比信息提升LLM推理能力。

**💡 创新点**

创新点在于将GRPO目标重新表述为对比式，构造交叉聚合上下文（右/错互相提供信息）以及通过奖励-置信度协方差动态校正优势基线，显著降低梯度方差。

**🔧 技术方法**

核心技术包括对比式GRPO reformulation、Bilateral Context Conditioning (对组中正确/错误样本做上下文拼接)、Reward‑Confidence Correction (基于协方差的基线修正)、以及在多种GRPO变体中的统一集成。

**📊 数据集**

在两个指令微调模型 Qwen3‑4B‑Instruct‑2507 与 Phi‑4‑mini‑instruct‑3.8B 上，用 DAPO‑Math‑17k 进行训练，并在 Math500、AMC 2023、AIME 2024/2025 四个数学推理基准上评估。

**📈 对比分析**

与标准 GRPO 及其四个变体（Dr.GRPO、DAPO、ASPO、GMPO、GSPO）比较，双边条件化与奖励-置信度校正分别提升 0.3–1.9% Pass@1，最优组合达 93.1%/79.2%（Qwen3‑4B/Phi‑4‑mini），并将梯度方差降低 25–37%。

**⚠️ 局限性**

局限包括：仅适用于二值奖励场景；对长序列可能产生上下文长度负担；对非数学推理任务（如连续奖励、代码生成）需进一步验证。

---

## 235. InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing

**arXiv ID:** 2603.13082 | [PDF](https://arxiv.org/pdf/2603.13082v1)

**作者:** Yebin Yang `[一作]` (Karlsruhe Institute of Technology), Kunyu Peng `[通讯]` (INSAIT, Sofia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出文本引导的多人体3D动作编辑任务，并构建对应数据集与基准；

**💡 创新点**

创新点在于推出InterEdit3D大规模两人体编辑数据集以及同步无分类器条件扩散框架InterEdit，该框架通过语义感知计划令牌与频率令牌对齐，实现高层编辑意图与交互动态的双重控制；

**🔧 技术方法**

采用扩散模型、Transformer、CLIP文本编码器、对齐的Motion Teacher、DCT频率特征、频率令牌与计划令牌以及同步无分类器引导；

**📊 数据集**

使用新构建的InterEdit3D（5,161对源-目标-文本三元组）以及InterHuman数据源；

**📈 对比分析**

在检索式评价指标（g2t/g2s Recall@K 与 FID）上，相较于改造的单人编辑器（MotionFix、MotionLab）与多人体生成器（InterGen、TIMotion），InterEdit在g2t Recall@1/2/3分别提升约6–7个百分点，g2s提升约4–7个百分点，FID下降约17%；

**⚠️ 局限性**

局限在于仅覆盖两人体交互，难以直接扩展到更多人或更复杂的交互情景；对长时序编辑和细粒度动作细节的控制仍有挑战；数据集规模与多样性仍受限。

---

## 236. Adaptive Vision-Language Model Routing for Computer Use Agents

**arXiv ID:** 2603.12823 | [PDF](https://arxiv.org/pdf/2603.12823v1)

**作者:** Xunzhuo Liu `[一作]` (vllm semantic router project), Huamin Chen `[通讯]` (Red Hat)

**通讯引用:** 2235 | [OpenAlex ID](https://openalex.org/A5101790571)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Adaptive VLM Routing（AVR）框架，动态为计算机使用代理（CUA）的每个工具调用选择最合适的视觉-语言模型（VLM），从而显著降低推理成本；

**💡 创新点**

创新点在于：①将 VLM 推理视为路由问题，引入轻量级语义路由层；②基于多模态嵌入进行动作难度估计；③使用小模型的 logprob 进行置信度探测；④在热代理中利用记忆补偿提升小模型性能；⑤将安全护栏（Visual Confused Deputy）融入路由决策，实现成本、安全、精度三重平衡；

**🔧 技术方法**

技术包括：多模态嵌入（SigLIP+MiniLM-L6-v2）、轻量级难度知识库、logprob 探测、内存检索（contrastive KB）、安全风险检测、阈值自适应策略、成本-准确率理论分析；

**📊 数据集**

使用 ScreenSpot-Pro GUI grounding 评测数据、OpenClaw 代理基准数据、Visual Confused Deputy 的对比数据以及自定义的 VLM 池（Qwen2.5-VL-7B/72B）来评估路由效果；

**📈 对比分析**

与单一大模型基线（全 72B）对比，AVR 在冷启动时可实现约 52% 成本节省，热启动并结合难度分类时可达 78%；准确率保持在基线的 ±2% 以内；安全性通过 Visual Confused Deputy 验证，所有高风险操作均升级至最高级模型；

**⚠️ 局限性**

局限性：①实验为理论投影，未在真实 CUA 场景中完成端到端评估；②探测阶段额外延迟，短任务时可能抵消节省；③记忆冷启动时收益有限；④难度知识库覆盖不全可能导致误路由；⑤大图像导致探测成本高，限制节省幅度。

---

## 237. Why Neural Structural Obfuscation Can't Kill White-Box Watermarks for Good!

**arXiv ID:** 2603.12679 | [PDF](https://arxiv.org/pdf/2603.12679v1)

**作者:** Yanna Jiang `[一作]` (University of Technology Sydney), Qin Wang `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Canon 框架，对被 Neural Structural Obfuscation (NSO) 攻击破坏的白盒水印进行恢复，使模型恢复可验证性且不降低任务精度。

**💡 创新点**

将 NSO 重定义为图一致的生产者‑消费者威胁模型，利用功能等价性导致的信号一致性约束，设计全局 ChannelTransform 推导与同步重写，实现对任意图结构网络（残差、拼接等）的 100% 恢复，并兼容多种白盒水印方案。

**🔧 技术方法**

使用激活签名聚类、比例关系检验、稀疏矩阵 ChannelTransform、图一致性重写、残差与拼接同步、BN 兼容缩放等技术；基于白盒访问的激活探测与层级递推。

**📊 数据集**

主要在 MNIST（扩展为 3 通道）上验证不同模型架构（ResNet‑18、EfficientNet、InceptionV3、DenseNet）与多种水印方案的恢复效果。

**📈 对比分析**

通过对比原始模型、NSO 攻击后模型与恢复后模型的任务精度、白盒水印相似度及恢复时间进行评估；结果显示攻击后精度保持 0 误差，恢复后水印相似度恢复至 100%，恢复时间在 10 秒至 1 分钟之间，且无误报。

**⚠️ 局限性**

局限性：仅适用于可完整访问前向图并进行激活探测的白盒场景；对极高比例或包含非线性混合操作的自定义 NSO 变体的通用性尚待验证；需进一步评估在更复杂数据集上的泛化表现。

---

## 238. Hunting CUDA Bugs at Scale with cuFuzz

**arXiv ID:** 2603.12485 | [PDF](https://arxiv.org/pdf/2603.12485v1)

**作者:** Mohamed Tarek Ibn ziad `[一作]`, Christos Kozyrakis `[通讯]` (Stanford University)

**通讯引用:** 21069 | [OpenAlex ID](https://openalex.org/A5042148531)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出并实现了针对CUDA程序的全程序fuzzer cuFuzz，能够有效发现内存安全和并发错误；

**💡 创新点**

创新点在于克服了传统GPU fuzzing的三大障碍：通过全程序fuzzing避免内核级误报；使用NVBit实现设备端覆盖反馈并与主机覆盖合并；通过将设备与主机的sanitizer分离到不同进程解决覆盖与sanitizer冲突；

**🔧 技术方法**

核心技术包括：全程序fuzzing、NVBit设备端指令插桩、覆盖反馈合并、独立进程执行sanitizer以及persistent-mode加速；

**📊 数据集**

评估使用14个CUDA程序（含商业库）和HeCBench等实际工作负载；

**📈 对比分析**

与基线方法相比，cuFuzz在边缘覆盖率、唯一输入数量以及在闭源目标上的bug发现率上均显著提升；性能方面，尽管存在覆盖和插桩的运行时开销，整体吞吐量通过persistent-mode显著提升；

**⚠️ 局限性**

局限性包括：仅针对CUDA架构；对高阶GPU功能和异步执行的覆盖仍有限；插桩和sanitizer的跨进程通信引入额外复杂性与潜在的时间开销。

---

## 239. Consistent and Efficient MSCKF-based LiDAR-Inertial Odometry with Inferred Cluster-to-Plane Constraints for UAVs

**arXiv ID:** 2603.12904 | [PDF](https://arxiv.org/pdf/2603.12904v1)

**作者:** Jinwen Zhu `[一作]` (Meituan UAV), Guoquan Huang `[通讯]` (University of Delaware)

**通讯引用:** 5784 | [OpenAlex ID](https://openalex.org/A5008502528)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种面向资源受限无人机的、基于MSCKF的LiDAR-IMU闭环定位系统，利用平面约束的零空间投影实现估计一致性，并通过并行体素化与聚类平面测量模型显著降低计算量。

**💡 创新点**

创新点包括：①在MSCKF框架中构造滑动窗口内的共面约束并直接对特征参数做零空间投影，消除过度自信问题；②提出并行体素化与自适应平面提取，提升数据关联速度；③设计了无损聚类平面测量模型，将成千上万个点对平面约束压缩为4维测量，减少观测维度。

**🔧 技术方法**

技术手段主要涵盖：多状态约束卡尔曼滤波（MSCKF）、零空间投影（null‑space projection）、并行体素化与自适应平面拟合、Cholesky分解代替QR实现聚类平面测量、滑动窗口状态克隆与协方差传播。

**📊 数据集**

实验使用了仿真室内环境（Gazebo/ROS）以及实测六个序列，分别包含稀疏、森林和城市环境，硬件平台为配备下视Livox Avia激光雷达与RTK‑GPS的六旋翼无人机，并在NVIDIA Jetson TX2上测试。

**📈 对比分析**

与FAST‑LIO2、VoxelMap、iG‑LIO、Super‑LIO、FF‑LINS等SOTA方法相比，本文方法在精度上与之相当（位姿误差<1%），但在估计一致性（平均NEES≈6.7，接近理论理想值6）和实时性能（平均帧时<30 ms、内存≈100 MB）方面均优于大多数对手；在稀疏场景下还显示出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：对动态或非平面环境的鲁棒性未作充分验证；当前实现仍以全点云为输入，虽然已高效，但在极高点密度下仍可能成为瓶颈；系统缺乏长期关键帧/地图管理，难以支持超长航程或大范围地图构建。

---

## 240. SmoothTurn: Learning to Turn Smoothly for Agile Navigation with Quadrupedal Robots

**arXiv ID:** 2603.12842 | [PDF](https://arxiv.org/pdf/2603.12842v1)

**作者:** Zunzhi You `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 21899 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为 SmoothTurn 的强化学习框架，用于四足机器人在快速运动中平滑转向，能够在给定的局部目标序列中连续导航并保持高速度。

**💡 创新点**

创新点包括：① 将单目标导航扩展为顺序目标导航；② 设计连续的序列目标奖励函数，消除停顿现象；③ 在观察空间加入多步前瞻目标窗口，使机器人能够提前预测并调整姿态；④ 自动化的目标难度递进课程（Curriculum），使训练过程更稳健。

**🔧 技术方法**

使用了深度强化学习（PPO）、基于 Isaac Gym 的GPU仿真、Unitree Go2 机器人模型、全局状态信息（关节位置/速度、接触、重力投影、基座角速度等）以及自定义的目标采样与奖励设计。

**📊 数据集**

主要数据来自仿真中的自定义目标序列（随机采样的转向角和距离），并在真实 Unitree Go2 机器人上进行10次实地实验。未使用公开图像或LiDAR数据集，而是依赖仿真产生的位姿与传感器噪声。

**📈 对比分析**

与基线单目标奖励策略对比，SmoothTurn 在四个固定转向序列下表现出更低的摔倒率、更高的成功率和更短的完成时间；在真实机器人实验中亦保持更快的平均穿越时间，证明了其在连续转向时的高效性。

**⚠️ 局限性**

局限性：① 仍需要手动设定目标达成阈值；② 主要针对平面导航，未结合全局障碍感知；③ 需要高质量位姿估计，若传感器失效或噪声较大可能影响性能；④ 仅在相对简单的室内环境验证，动态障碍或复杂地形的适应性尚未评估。

---

## 241. Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support with Multi-Center Clinical Validation

**arXiv ID:** 2603.12430 | [PDF](https://arxiv.org/pdf/2603.12430v1)

**作者:** Jian Jiang `[一作]`, Yutong Ban `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种具备分层推理的手术视觉语言模型（Surg‑R1），实现多任务（工具定位、三元组识别、阶段识别、动作识别、CVS评估）且能输出可解释的推理链。

**💡 创新点**

创新点包括三层分层推理架构（感知、关系、上下文）、基于强化学习的推理优化（GRPO与高熵标记关注），以及无监督扩充的 CoT 数据生成与迭代自我改进机制。

**🔧 技术方法**

采用 LoRA 参数高效微调的视觉语言模型、GPT‑5.1 生成的三层 CoT、Group Relative Policy Optimization（GRPO）与高熵标记关注的强化学习，以及自监督的拒绝采样与教师蒸馏。

**📊 数据集**

训练数据来源于 23 个公开手术数据集（覆盖工具定位、VQA、三元组、CVS、阶段、动作等）约 320 万条三层 CoT 语料；评估使用 10 个公开基准及 6 个多中心外部验证集（共 6,145 幅图）。

**📈 对比分析**

与 GPT‑5.1、Gemini‑3.0 Pro 及专业手术 VLMs 比较，Surg‑R1 在公开基准上的 Arena 分数为 57.7%（约 57% vs. 29.8%/28.5%），在多中心外部验证上平均 Arena 分数 60%（相较基线提升约 15%）。

**⚠️ 局限性**

主要限制为：① 训练与推理计算资源需求高；② 推理延迟较大，尚需优化以满足实时手术需求；③ 仍需临床医生验证其推理可靠性与安全性。

---

## 242. LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis for Treatment-Resistant Depression on Reddit

**arXiv ID:** 2603.12343 | [PDF](https://arxiv.org/pdf/2603.12343v1)

**作者:** Yuxin Zhu `[一作]` (Emory University), Abeed Sarker `[通讯]` (Emory University)

**通讯引用:** 5004 | [OpenAlex ID](https://openalex.org/A5086087170)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从2010-2025年28个精神健康子版块收集TRD相关Reddit帖子，构建药物归一化词表，并对每个药物提及使用基于方面的情感分类，定量描绘患者对不同治疗方案的感受。

**💡 创新点**

首次将社交媒体中药物提及的品牌名、拼写错误、俚语等多样化形式统一归一化，并结合LLM生成的合成数据提升情感分类精度，实现大规模TRD讨论中的药物级情感分析。

**🔧 技术方法**

使用词表构建、QMisSpell拼写生成、LLM（LLaMA3）数据扩增、DeBERTa‑v3目标情感分类器、微调+数据增强以及卡方、二项检验等统计方法。

**📊 数据集**

2010-2025年28个子版块共21,826条Reddit帖子，5,059条TRD相关帖子，3,839条含药物提及，总计23,399次药物提及；训练数据来源于SMM4H 2023 Twitter治疗情感数据。

**📈 对比分析**

在SMM4H共享任务基准上，DeBERTa‑v3增强后微调得到micro‑F1 0.800（95%CI 0.780–0.820），超过任务最高系统0.778；在Reddit上的情感分类显示总体中性72.1%，负面14.8%，正面13.1%，并通过卡方检验和二项检验显著区分药物类别和个体药物。

**⚠️ 局限性**

数据来源仅限Reddit，缺乏人口代表性，无法验证TRD诊断、用药剂量等；情感分类对负面情绪的召回率低，可能低估负面比例；未覆盖隐性治疗和非药物方案；未对模型在Reddit上做人工标注验证，存在领域漂移风险。

---

## 243. Memory Printer: Exploring Everyday Reminiscing by Combining Slow Design with Generative AI-based Image Creation

**arXiv ID:** 2603.13116 | [PDF](https://arxiv.org/pdf/2603.13116v1)

**作者:** Zhou Fang `[一作]` (Eindhoven University of Technology), Janet Yi-Ching Huang `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5080665981)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款名为 Memory Printer 的可触感装置，结合丝网印刷工艺与文本到图像的生成 AI，支持用户逐步重构未被记录的记忆场景并打印成实物照片。

**💡 创新点**

创新点在于将慢速设计、可触感交互与分层控制三大原则融入生成式 AI 交互，恢复用户的身体感知与控制感，并通过层级化提示与物理抠刀实现对 AI 输出的可追踪、可细化调整。

**🔧 技术方法**

技术包括文本到图像的生成模型（如 Stable Diffusion/Flux）、离散距离传感器驱动的物理抠刀、层级化遮罩绘制接口、内置 Zink 热转印打印机、以及语音采集与实时音频输入。

**📊 数据集**

未采用专门的自定义数据集，使用预训练的生成式 AI 模型所基于的公开互联网大规模图像语料库；参与者在实验中提供自身的记忆描述作为输入，未使用真实照片训练模型。

**📈 对比分析**

通过与网页端 GAI 工具（KreaAI）在 24 名参与者的对照实验中，采用思考导向法、访谈与情感与控制感量表评估。结果显示 Memory Printer 在用户感知控制感、情绪共鸣与创造性探索方面显著优于网页工具；同时，实验中引入的层级化与慢速交互降低了认知负荷。

**⚠️ 局限性**

局限性包括样本主要为 22-31 岁青年，缺乏老年人与更广泛文化背景；对生成模型固有的假记忆、算法偏见与隐私风险关注不足；实验未对 AI 生成图像的客观质量进行量化评估，仅依赖主观体验；硬件成本与可扩展性仍待验证。

---

## 244. Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains

**arXiv ID:** 2603.12624 | [PDF](https://arxiv.org/pdf/2603.12624v1)

**作者:** Guodong Sun `[一作]` (Hubei University of Technology), Yang Zhang `[通讯]` (Hubei University of Technology)

**通讯引用:** 83913 | [OpenAlex ID](https://openalex.org/A5100354659)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在重载货运列车故障检测中，构建了一个轻量化的基于SAM的实例分割框架SAM FTI-FDet，实现了自动提示生成并适配边缘设备。

**💡 创新点**

核心创新在于自学习提示生成模块（query-based prompt）与轻量化TinyViT-SAM骨干的结合，实现无人工提示、低算力高精度的故障检测；同时引入多尺度特征调度器提升鲁棒性。

**🔧 技术方法**

采用了Segment Anything Model（SAM）、TinyViT-SAM骨干、Transformer式自注意力提示生成、跨尺度特征聚合、Mask Decoder及DeepSpeed混合精度训练。

**📊 数据集**

使用了真实工况下的货运列车检测数据集（共4410张图，15类，6种场景），以及MS‑COCO数据集做跨域泛化验证。

**📈 对比分析**

与Mask R‑CNN、PointRend、YOLO‑ACT、SAM‑seg、RSPrompter等基线比较，SAM FTI‑FDet在AP^mask、AP^box均取得最高值（约74.2/74.6），并保持较低参数量（≈5M）和实时帧率（≈16‑30 FPS）。

**⚠️ 局限性**

局限包括对极小或低显著性缺陷的检测仍易漏报、严重遮挡时性能下降，以及提示生成训练过程相对复杂，未来需改进提示分配和多模态融合。

---

## 245. RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics

**arXiv ID:** 2603.12939 | [PDF](https://arxiv.org/pdf/2603.12939v1)

**作者:** Yuzhi Huang `[一作]` (Shenzhen International Graduate School Tsinghua University), Zhi Wang `[通讯]` (Shenzhen International Graduate School Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练自由的框架RoboStream，能够在机器人视觉语言模型中实现跨时空推理和持久记忆，提升长时程操纵的可靠性。

**💡 创新点**

创新点在于：1) 通过Spatio‑Temporal Fusion Tokens (STF‑Tokens) 将视觉证据与3D几何绑定，实现对象持久定位；2) 通过Causal Spatio‑Temporal Graph (CSTG) 记录动作触发的状态转移，保持因果连贯性；3) 结合VLM进行链式思维推理，形成训练无关、可迁移的长时程规划方案。

**🔧 技术方法**

核心技术包括：对象中心感知 + 开放词表分割、STF‑Token编码（视觉+3D几何+时间）、4D CSTG构建与更新、VLM链式思维推理以及基于STF‑Token的确定性动作实例化。

**📊 数据集**

使用了多套基准：RLBench (长时程任务)、SIMPLER (零样本短时程任务)、6‑DoF SpatialBench、Open6DOR V2 (空间推理) 与真实世界Franka Research 3 机器人实验。

**📈 对比分析**

与SoFar、VoxPoser、RT‑2‑X等现有VLM基线对比，RoboStream在所有基准上均显著提升：RLBench平均成功率从~30%提升至90.5%，SIMPLER平均成功率从~38%提升至~75%，Open6DOR V2 6‑DoF成功率从~35%提升至~52%，在真实世界实验中在遮挡与重建任务中分别达88.9%与44.4%。

**⚠️ 局限性**

局限性包括：1) 仍依赖分离式规划‑执行框架，感知或抓取失误会直接导致失败；2) 需要高质量RGB‑D输入，深度噪声或遮挡会影响STF‑Token构造；3) 对极其动态或高接触的场景尚未充分验证。

---

## 246. Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning

**arXiv ID:** 2603.12988 | [PDF](https://arxiv.org/pdf/2603.12988v1)

**作者:** Aditya Parikh `[一作]` (Technical University of Denmark), Aasa Feragen `[通讯]` (Technical University of Denmark)

**通讯引用:** 1277 | [OpenAlex ID](https://openalex.org/A5041988622)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种公平性意识的多类肺部疾病诊断框架，利用注意力多实例学习（MIL）和对抗性性别消除（GRL）对胸部CT体积进行分类；

**💡 创新点**

创新点在于将注意力MIL与梯度反转层结合，实现对性别信息的主动消除；同时采用聚焦损失+标签平滑、分层交叉验证、子组过采样以及阈值优化等多层次公平性策略；

**🔧 技术方法**

使用ConvNeXt骨干网络、注意力MIL、Gradient Reversal Layer、聚焦损失+标签平滑、分层交叉验证、子组过采样、测试时水平翻转、软logit投票和OOB阈值优化等技术；

**📊 数据集**

使用889个胸CT扫描的数据集，覆盖四类（Healthy、COVID-19、Adenocarcinoma、Squamous Cell Carcinoma）并包含性别信息，尤其女性Squamous Cell Carcinoma样本极为稀缺；

**📈 对比分析**

在CVPR 2026 Fair Disease Diagnosis Challenge中通过五折交叉验证进行比较，平均比赛分数为0.685，单折最高可达0.759，男女macro‑F1均衡，SCC仍为最难类；

**⚠️ 局限性**

主要局限在于女性SCC子组样本极少，导致模型在该子组仍表现欠佳；整体模型仍难以彻底消除性别偏差，未来需引入生成式数据增强或无监督预训练等方法。

---

## 247. Optimize Wider, Not Deeper: Consensus Aggregation for Policy Optimization

**arXiv ID:** 2603.12596 | [PDF](https://arxiv.org/pdf/2603.12596v1)

**作者:** Zelal Su `[一作]` (University of Texas at Austin), Keshav Pingali `[通讯]` (University of Texas at Austin)

**通讯引用:** 10924 | [OpenAlex ID](https://openalex.org/A5013181067)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的强化学习优化策略CAPO，通过在同一批数据上并行训练多份PPO专家并在自然参数空间或参数空间中聚合其策略，从而有效减少PPO更新中的噪声和无效方向，提高样本利用率。

**💡 创新点**

核心创新是利用Fisher信息几何将PPO更新分解为信号（自然梯度方向）和废物（正交残差），揭示“优化深度悖论”，并证明在自然参数空间聚合能显著降低KL代价、提升代理在信号方向上的增益、保持可信区域合规，从而实现更宽而非更深的训练策略。

**🔧 技术方法**

技术包括：PPO与TRPO的信号-废物分解、Fisher信息几何、自然参数空间的对数意见池(LogOP)聚合、参数空间平均、实验中的并行梯度计算与经验分布采样、KL早停和热身策略。

**📊 数据集**

在Gymnasium MuJoCo-v4的六个连续控制任务上评估：Hopper、HalfCheetah、Walker2d、Ant、Humanoid 与 HumanoidStandup。

**📈 对比分析**

与PPO、TRPO、PPO-K×、Best-of-K、PPO-SWA等基线比较，CAPO在五个任务中均超越对手（最高提升71%，在Humanoid上提升8.6倍），而PPO-K×在高维任务上甚至退化，验证了宽度优先的有效性。

**⚠️ 局限性**

主要局限是梯度计算成本乘以专家数K，虽然并行可缓解；实验仅覆盖连续控制和高斯策略，对离散动作、视觉或语言任务的适用性尚未验证；聚合的多样性仅来自批次顺序，未探讨更丰富的专家多样化方法。

---

## 248. FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning

**arXiv ID:** 2603.12806 | [PDF](https://arxiv.org/pdf/2603.12806v1)

**作者:** Zeying Gong `[一作]` (Hong Kong University of Science and Technology), Junwei Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2086 | [OpenAlex ID](https://openalex.org/A5059207044)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于流匹配的统一导航策略，并搭建了包含物理合法人群模拟的动态导航基准，通过两阶段静态→动态训练实现跨任务与跨平台的自适应导航；

**💡 创新点**

创新点包括：① 通过线性化概率流的Rectified Flow显著减少采样步骤并提升推理效率；② 采用静态到动态的训练曲线与GRPO强化学习实现社会化意识并对静态任务产生正迁移；③ 统一评测框架覆盖六大导航任务，支持多平台零样本迁移；

**🔧 技术方法**

使用技术包括：流匹配（CFM、RF）、Transformer解码器、基于经验的强化学习（GRPO）、NavMesh+物理仿真的人群行为协议、直线概率流采样；

**📊 数据集**

使用数据集：Gibson场景专家轨迹、NavDP的ClutterEnv与高保真结构化场景、Isaac Sim中物理合法人群模拟以及公开的视觉导航基准；

**📈 对比分析**

与传统规划、RL、混合学习、模仿学习及扩散模型基线在静态与动态六任务上对比，静态任务成功率提升3-4%，动态任务提升更显著；推理速度比NavDP/FlowNav提升47%/29%，在三种机器人平台上实现零样本实地转移；

**⚠️ 局限性**

局限性：缺乏显式社会推理与人类非语言意图解码；评估主要局限在室内行人模拟，未覆盖多样化环境与异构动态代理，真实世界社交交互数据缺失。

---

## 249. Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**arXiv ID:** 2603.12773 | [PDF](https://arxiv.org/pdf/2603.12773v1)

**作者:** Guodong Fan `[一作]` (Shandong Technology and Business University), Jinjiang Li `[通讯]` (Shandong Technology and Business University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于视觉语言模型生成语义引导图，并通过交叉注意力与语义对齐损失实现的语义敏感水下图像增强策略

**💡 创新点**

创新点在于利用VLM的开放世界语义理解生成空间语义引导图，采用双重引导（结构化注意力+显式对齐损失）弥补传统方法的语义盲区

**🔧 技术方法**

使用BLIP/CLIP/LLaVA等视觉语言模型、交叉注意力模块、语义对齐损失、感知损失等技术

**📊 数据集**

主要使用UIEB、U45、Challenge60三大水下增强基准数据集，以及Trash‑ICRA19和SUIM两大下游检测/分割数据集

**📈 对比分析**

将该机制嵌入PUIE、SMDR、UIR、PFormer、FDCE五种SOTA增强网络，实验显示PSNR/SSIM均提升，且在检测与分割任务上AP/mIoU显著提升

**⚠️ 局限性**

局限性包括对VLM文本提示的依赖、对稀有或未见对象可能生成不准引导图，以及额外的推理开销

---

## 250. ActTail: Global Activation Sparsity in Large Language Models

**arXiv ID:** 2603.12272 | [PDF](https://arxiv.org/pdf/2603.12272v1)

**作者:** Wenwen Hou `[一作]` (Hong Kong University of Science and Technology), Shiwei Liu `[通讯]` (ELLIS Institute Tübingen & Max Planck Institute for Intelligent Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ActTail 方法，通过对 Transformer 投影的重构矩阵谱特征进行分析，实现输入激活的自适应 TopK 稀疏化；

**💡 创新点**

创新点在于将 Heavy‑Tail Self‑Regularization 理论与稀疏分配相结合，利用每个投影的指数 α 计算最优稀疏率，从而在高稀疏率下保持甚至提升性能；

**🔧 技术方法**

采用 TopK 选取输入激活、重构矩阵的经验谱密度（ESD）估计 α、线性映射稀疏率、全局稀疏约束等技术；

**📊 数据集**

在 LLaMA2‑7B/13B、Mistral‑7B、Qwen1.5‑7B 等 decoder‑based LLM 上使用 WikiText2 评测 perplexity，并在 EleutherAI LM Evaluation Harness 上评估 MMLU、ARC‑c、HellaSwag、BOOLQ、PIQA、WinoGrande 等下游任务；

**📈 对比分析**

与 TEAL 的均匀 TopK 稀疏化做对比，在 70%/80% 稀疏率下，ActTail 在 perplexity 上分别提升 13–40%（LLaMA2‑13B 40%）、下游任务平均提升 2–8%，表明在高稀疏率下性能更优；

**⚠️ 局限性**

局限性包括依赖重构矩阵谱估计的可靠性，未针对所有 LLM 架构进行广泛验证，以及对极端稀疏率或不同激活函数的适用性尚待进一步研究。

---

## 251. Learning from Child-Directed Speech in Two-Language Scenarios: A French-English Case Study

**arXiv ID:** 2603.12906 | [PDF](https://arxiv.org/pdf/2603.12906v1)

**作者:** Liel Binyamin `[一作]` (Ben-Gurion University of the Negev), Elior Sulem `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1775 | [OpenAlex ID](https://openalex.org/A5026220115)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在资源受限的多语言设置下，对 BabyBERTa 进行扩展，系统评估了英法双语小模型在儿童指向性语料与多域语料上的预训练与微调效果。

**💡 创新点**

提出了严格匹配大小的英法双语语料构建方法，首次对比单语、双语及跨语预训练，并引入法语 QAMR/QASRL 资源；揭示了双语预训练对文本蕴含的显著提升。

**🔧 技术方法**

使用了 BabyBERTa、RoBERTa、LTG‑BERT 和 T5‑tiny 等多种小型 Transformer 结构，结合 Masked Language Modeling 与下游任务微调。

**📊 数据集**

采用约 2.5M 词的儿童指向性语料（CHILDES）、10M 词的多域开发语料，以及法语翻译的 QAMR/QASRL、FQuAD、XNLI、CLAMS 等评测数据集。

**📈 对比分析**

通过在相同模型规模、相同语料量下进行三种设置（单语、双语、跨语）的对比，结果显示：双语预训练显著提升 XNLI（尤其是法语），Wikipedia 训练更有利于语义任务，CHILDES 有利于语法任务；在 10M 词规模下双语优势减弱但仍存在。

**⚠️ 局限性**

仅覆盖英法两种语言，未测试更高阶语言或不同形态学类型；仅涉及基于 Encoder‑Encoder/Encoder‑Decoder 架构的模型，未探讨 Decoder‑only 结构；资源规模仍有限，无法与大型预训练模型竞争。

---

## 252. Developing and evaluating a chatbot to support maternal health care

**arXiv ID:** 2603.13168 | [PDF](https://arxiv.org/pdf/2603.13168v1)

**作者:** Smriti Jha `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2883 | [OpenAlex ID](https://openalex.org/A5079207566)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并评估了一款多语言母婴健康聊天机器人，结合阶段感知分流、检索增强生成与LLM条件生成，并在低资源环境下实现安全、可信的信息服务。

**💡 创新点**

创新点包括：① 针对母婴健康的阶段感知分流路由策略及其评估基准；② 生成多证据检索基准并引入直接/相关证据标注；③ 多层评估框架，融合人工标签、LLM-as-judge与专家校准；④ 结合多语言混合查询的检索与重排序设计。

**🔧 技术方法**

使用技术：基于GPT‑4‑Turbo的生成模型；BM25与E5多语言稠密检索；RRF混合检索；MiniLM和MedCPT跨编码器重排序；LLM-as-judge（Gemini Pro）与人工专家评测。

**📊 数据集**

数据集：1）100条合成多证据检索题；2）150条手工标注的分流基准；3）781条真实用户问答（英文、印地语、阿萨姆语，含代码混用）。

**📈 对比分析**

比较方法：对检索效果用Recall@K/Hit@K/MRR；对分流效果用召回率/精确率；对端到端效果用LLM‑judge评分和专家评估。结果显示：混合检索在Recall@5/10显著优于单一检索；分流召回率达86.7%且精确率≈90%；RAG+分流版本在LLM‑judge上在正确性、完整性、文化适宜性等指标上均优于无检索或无分流版本，且在安全标记与不当信息泄露方面有显著下降。

**⚠️ 局限性**

局限性：① 模板化分流在非英语查询中语言匹配不佳；② 评测主要基于历史数据，真实用户交互可能产生不同问题分布；③ LLM‑judge与专家评分仍存在偏差，难以完全替代人工；④ 系统对缺失上下文的识别仍不完善，需进一步提升缺失信息检测。

---

## 253. Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval

**arXiv ID:** 2603.12711 | [PDF](https://arxiv.org/pdf/2603.12711v1)

**作者:** Jing Yang `[一作]` (Southeast University), Pengfei Fang `[通讯]` (Southeast University)

**通讯引用:** 9592 | [OpenAlex ID](https://openalex.org/A5031463327)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于文本提示和相位先验的无监督跨域图像检索模型 TPSNet，解决传统伪标签噪声与域偏差导致的语义退化问题。

**💡 创新点**

创新点在于同时引入 CLIP 生成的可学习域提示文本先验与频域相位先验，通过跨注意力融合实现语义与域不变特征的协同学习，显著提升了无监督检索性能。

**🔧 技术方法**

技术包括 CLIP 文本提示学习、相位特征编码与融合、跨注意力机制、原型更新和无监督对比学习。

**📊 数据集**

使用 Office-Home 与 DomainNet 两大跨域检索数据集进行实验。

**📈 对比分析**

与 DD、CoDA、ProtoOT、ShieldIR、SA‑MoE 等现有方法在 12 个跨域检索场景下对比，TPSNet 在 P@1、P@5、P@15 等指标上普遍领先，提升幅度高达 10‑25%。

**⚠️ 局限性**

局限性包括对预训练 CLIP 的依赖，以及在极度抽象或视觉相似度极高的域（如 Quickdraw）仍可能出现检索错误。

---

## 254. Embedded Quantum Machine Learning in Embedded Systems: Feasibility, Hybrid Architectures, and Quantum Co-Processors

**arXiv ID:** 2603.12540 | [PDF](https://arxiv.org/pdf/2603.12540v1)

**作者:** Somdip Dey `[一作]` (Regent College London), Syed Muhammad Raza `[通讯]` (Regent College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

探讨在边缘嵌入式平台上实现量子机器学习的可行性，并提出两条实现路径：混合量子-经典离线调用和本地量子协处理器。

**💡 创新点**

从电路与系统视角正式化两条EQML实现路径，识别并映射关键瓶颈（时延、编码开销、NISQ噪声、工具匹配、能耗），并给出可操作的工程路线图与安全治理建议。

**🔧 技术方法**

采用量子核方法、变分量子算法、量子感知前端、量子启发式优化以及在经典嵌入式硬件上实现的量子启发式方法。

**📊 数据集**

本文并未使用实验数据集，而是基于已有的公开研究与技术评估进行理论分析。

**📈 对比分析**

通过对传统经典TinyML与未来QSoC型EQML在延迟、能耗与任务适配性上的定性对比，表明混合离线调用适用于非实时任务，嵌入式协处理器在未来可能支持低时延优化。

**⚠️ 局限性**

局限性包括高网络时延与排队导致实时性不足；量子编码开销大；NISQ设备噪声与有限量子比特；工具链与嵌入式开发环境不匹配；能耗超出典型边缘平台预算；安全与可靠性缺乏成熟治理。

---

## 255. MRGeo: Robust Cross-View Geo-Localization of Corrupted Images via Spatial and Channel Feature Enhancement

**arXiv ID:** 2603.12587 | [PDF](https://arxiv.org/pdf/2603.12587v1)

**作者:** Le Wu `[一作]` (Shenzhen University), Yingying Zhu `[通讯]` (Shenzhen University)

**通讯引用:** 2481 | [OpenAlex ID](https://openalex.org/A5068185303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了MRGeo系统，针对跨视角地理定位在受图像失真（如模糊、天气）环境下的鲁棒性进行系统性改进。

**💡 创新点**

在特征层通过空间自适应表示模块SARM与通道校准模块CCM动态提升局部与全局信息的鲁棒性，并在结构层通过区域几何对齐模块RGAM强制空间一致性，从而实现多级防御。

**🔧 技术方法**

使用ViT骨干网络、注意力机制、动态门控融合、通道依赖建模、区域分块对齐以及InfoNCE对比学习训练。

**📊 数据集**

在CVUSA、CVACT标准数据集及其对应的CVUSA-C-ALL、CVACT-C-ALL、CVACT_test-C-ALL等全面腐败鲁棒性基准上进行实验。

**📈 对比分析**

与L2LTR、TransGeo、GeoDTR、Sample4G、EP-BEV、DReSS等六个SOTA方法在R@1、R@5、R@10、R@1%等指标上对比，MRGeo在所有鲁棒性基准上均达到最高R@1并在跨区域和少样本场景中提升13.5%/2.78%等显著收益。

**⚠️ 局限性**

对极端高强度噪声的鲁棒性仍有限，且RGAM对中心对齐的依赖导致在非中心对齐或极端失真情况下性能下降。

---

## 256. Mitigating Collusion in Proofs of Liabilities

**arXiv ID:** 2603.12990 | [PDF](https://arxiv.org/pdf/2603.12990v1)

**作者:** Malcom Mohamed `[一作]`, Ghassan Karame `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种全新的“Permissioned Proof of Liabilities (PPoL)”机制，利用新型的 Permissioned Vector Commitment (PVC) 保障加密交易所对用户资产的责任证明在不需要用户主动核查的情况下仍能防止欺诈与合谋攻击。

**💡 创新点**

创新点包括：
- PVC 原语：将向量承诺与用户签名绑定，确保任何更新都必须得到对应用户的签名。
- Permissioned PoL 模型：不再依赖用户在每轮周期主动检查，直接通过全局一致性证明阻止服务提供商篡改余额。
- 组合 KZG 承诺与 BLS 签名的同态与零知识特性，形成高效的批量签名、查找树、范围证明与追加证明。
- 引入基于零检查、分数检查与 APK（Aggregated Public Key）证明的多层安全结构，进一步提升安全性与隐私。

**🔧 技术方法**

技术手段包括：
- KZG 多项式承诺与 AMT（Accumulation Merkle Tree）证明树。
- BLS 签名与聚合签名技术。
- 零知识多项式证明（ZeroCheck、Binarity、Sumcheck）。
- APK 证明与分数证明。
- Fast Fourier Transform (FFT) 进行多项式运算。
- Go 语言实现，使用 gnark-crypto 库进行椭圆曲线与有限域运算。
- 使用 BN254 曲线、m=64 位二进制范围证明。

**📊 数据集**

实验使用的“数据集”为合成模拟：
- 用户数量 n 取 2^16，更新比例 2^-6（即每轮周期约 256 次更新）。
- 采用多核（单核与 32 核）对性能进行评估。
- 与 DAPOL+、Notus、Xiezhi 公开实现的基准进行对比。
- 没有使用真实金融交易所的真实账本数据，而是通过合成场景模拟。

**📈 对比分析**

比较方法：
- 以单核与 32 核下的全局证明生成时间、单用户包含证明时间、验证时间以及每秒更新吞吐量为指标。
- PPoL 在单核下全局证明生成约 4 s（与 Notus 约 3 s），在 32 核下可降至 1 s；包含证明仅 3.7 s。
- 与 Notus、DAPOL+ 的单核实现相比，PPoL 的全局证明和包含证明速度提升约 10×；多核时同样保持 10× 以上优势。
- 吞吐量：PPoL 在 300 次/秒更新下仍保持 <10 s 延迟，Notus 在相同负载下延迟 > 100 s，D APOL+ 延迟 > 300 s。

**⚠️ 局限性**

局限性与挑战：
- 需要事先建立全局 SRS（结构化引用字符串）并保证其私钥不泄漏；SRS 生成与管理仍是开销。
- 用户每轮周期只能签署一次更新，若需要更频繁的更新需缩短周期或批量签名。
- 虽然通过零知识隐藏了数据内容，但部分证明（如 APK 证明）会泄露用户活动模式，需额外随机化或兄弟账户混淆。
- 实验基于合成数据，缺乏在真实交易所规模与多币种环境中的实测。
- 对多链兼容性、跨域资产管理等场景的适配尚未完整探讨。

---

## 257. When LLM Judge Scores Look Good but Best-of-N Decisions Fail

**arXiv ID:** 2603.12520 | [PDF](https://arxiv.org/pdf/2603.12520v1)

**作者:** Eddie Landesberg `[一作]` `[通讯]`, Eddie Landesberg

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究指出在最佳‑of‑n 选择任务中，判别器的全球相关性并不能代表其在每个提示内的决策有效性。

**💡 创新点**

创新点在于提出决策中心评估框架，包括 within‑prompt 相关、Tie 率、Recovery 率和 Top‑1 准确率等指标，并强调在部署时需关注这些决策层面指标。

**🔧 技术方法**

使用了判别器的点值评分与二元对比评分，对比 global r、within‑prompt r、attenuation 系数、pairwise 评估等多种技术进行分析。

**📊 数据集**

实验数据来自 5,000 个 Chatbot Arena 提示的 4 候选响应集合，并补充了 24 提示的 fresh‑draw 内部验证。

**📈 对比分析**

与 oracle‑optimal、随机选择以及点值判别器等策略比较，发现 global r=0.47 对应仅 21% 的 Recovery 和 31.6% 的 Top‑1 准确率，而 pairwise 评估将 Recovery 提升至 61%，表明决策层面指标更能反映实际收益。

**⚠️ 局限性**

局限性包括判别器的粗量化评分导致高 Tie 率，仅在 best‑of‑2 场景下对比评估能显著提升，且在 best‑of‑4 完整预算下的提升有限。

---

## 258. Detecting Miscitation on the Scholarly Web through LLM-Augmented Text-Rich Graph Learning

**arXiv ID:** 2603.12290 | [PDF](https://arxiv.org/pdf/2603.12290v1)

**作者:** Huidong Wu `[一作]` (Chinese Academy of Sciences and City University of Hong Kong), Jianping Li `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 29115 | [OpenAlex ID](https://openalex.org/A5075507724)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种结合大语言模型与图神经网络的混合框架LAGMiD，用于高效识别学术文献中的误引（miscitation）问题。

**💡 创新点**

创新点在于：① 通过链式思维（chain-of-thought）实现多跳证据链推理，确保引用的语义一致性；② 将LLM的中间推理状态通过知识蒸馏迁移到GNN，实现可扩展的推理；③ 采用协同学习策略，对不确定样本进行有针对性的LLM推理与GNN蒸馏，提升模型鲁棒性。

**🔧 技术方法**

使用技术包括：大语言模型（Qwen3‑8B）、图神经网络（GCN两层）、链式思维提示、句子和边文本编码（SciBERT）、信息对比蒸馏（InfoNCE）、不确定性度量与梯度协同学习。

**📊 数据集**

实验数据集：RED、SciFact、S2ORC三个真实学术误引检测基准；每个数据集包含数千条句子‑引文对与完整引用图。

**📈 对比分析**

与八类基线（GCN、GLAD、RoBERTa、SciBERT、GLM4‑9B、Qwen3‑8B、AnomalyLLM、GuARD）在AUC、F1、Precision三指标上比较，LAGMiD在所有数据集上均取得最高分，显著优于最接近对手的AnomalyLLM和GuARD。

**⚠️ 局限性**

限制：① 仍需依赖LLM进行初始推理，推理成本虽被蒸馏降低但在极大规模图上仍可能受限；② 对于极少样本或高度相似的引文，链式推理可能出现多模态歧义；③ 模型的性能受LLM知识库的覆盖范围与更新速度影响。

---

## 259. Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation

**arXiv ID:** 2603.13017 | [PDF](https://arxiv.org/pdf/2603.13017v1)

**作者:** Sydney Lewis `[一作]` `[通讯]`, Sydney Lewis

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究单一用户对话历史的结构化压缩，将每条交互转化为四字段结构体，并构建检索索引；

**💡 创新点**

提出一种基于“核心+文件+房间”四维结构的压缩方法，既保留原始语料作为回溯依据，又实现约11×的 token 压缩；

**🔧 技术方法**

采用 Claude Haiku 4.5 生成结构体，使用 MiniLM-L6-v2 向量嵌入配合 HNSW/Exact，BM25（Okapi/FTS）检索，融合多种评分策略，并用多模型 LLM 进行评判；

**📊 数据集**

使用一名开发者在 Claude Code 上 6 个月共 14,340 条交流（4,182 轮对话）生成 12,427 个结构体作为实验语料；

**📈 对比分析**

以 MRR、Mean Grade、P@1、nDCG@10 等评估指标，比较纯文本、向量检索、关键词检索以及跨层融合四种配置，结果显示向量检索在压缩后保持近似性能，BM25 受压缩影响明显，而跨层融合略优于最佳纯文本基线；

**⚠️ 局限性**

实验仅基于单一用户数据，LLM 评判一致性低，BM25 对压缩敏感，未与其他记忆系统直接对比，空间导航功能未评估，且结果受模型能力与评判偏差影响。

---

## 260. Cheers: Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation

**arXiv ID:** 2603.12793 | [PDF](https://arxiv.org/pdf/2603.12793v1)

**作者:** Yichen Zhang `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 33608 | [OpenAlex ID](https://openalex.org/A5070926896)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态模型，通过将图像的补丁级细节与语义表示解耦，实现视觉理解与高保真图像生成的协同训练；

**💡 创新点**

核心创新在于：①统一视觉分词器将VAE解码器与SigLIP2-ViT语义编码器结合并进行像素解混洗，压缩语义token；②LLM（Qwen2.5-1.5B-Instruct）实现文本自回归与图像扩散的统一解码；③级联流匹配头分两阶段生成低频语义，再注入语义门控的高频残差，实现类似人类绘画的层级细化；

**🔧 技术方法**

技术包括VAE+SigLIP2-ViT视觉分词、像素解混洗压缩、LLM Transformer、DiT + AdaLN-Zero级联流匹配、门控高频注入、连续时间流匹配损失、CFG和时间调度平移；

**📊 数据集**

训练使用多阶段数据集：Stage I 4.5 M LLaVA‑UHD‑v3 图像‑标题对与 1.3 M ImageNet；Stage II 30 M 多模态样本（Infinity‑MM、LLaVA‑UHD‑v3、TextAtlas5M、BLIP‑3o、FLUX.2‑klein‑9B合成、LLaVA‑UHD‑v3文本）；Stage III 33 M 样本（LLaVA‑UHD‑v3 指令、FLUX.2‑klein‑9B 合成、Objects365 指令 466 K、Nemotron‑Cascade 文本）；Stage IV 3.8 M 细化样本（Echo‑4o‑Image、MoviePosters、ShareGPT‑4o‑Image）；

**📈 对比分析**

通过与现有统一多模态模型（如 Tar‑1.5B、MMaDA、UniDisc 等）在 GenEval、MMBench 等视觉理解与图像生成基准上进行比较，本文模型在保持相同或更低参数规模的情况下，性能匹配或超越，对 GenEval、MMBench 的得分提升显著，且训练成本仅为 Tar‑1.5B 的 20%，实现了 4× 的 token 压缩；

**⚠️ 局限性**

局限性包括：①模型参数规模相对较小，可能无法捕捉极细节；②未使用大型预训练视觉语言模型初始化，视觉理解与生成能力有待进一步提升；③训练流程仅基于单图像数据集，缺乏更复杂多模态数据，限制了泛化能力。

---

## 261. DIALECTIC: A Multi-Agent System for Startup Evaluation

**arXiv ID:** 2603.12274 | [PDF](https://arxiv.org/pdf/2603.12274v1)

**作者:** Jae Yoon Bae `[一作]` (Earlybird Venture Capital), Georg Groh `[通讯]` (Technical University of Munich)

**通讯引用:** 6318 | [OpenAlex ID](https://openalex.org/A5004398345)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并评估了一个基于大型语言模型的多智能体系统DIALECTIC，用于早期创业公司筛选和决策。

**💡 创新点**

将事实收集、论证生成、辩论式迭代批评与评估整合到一个可解释的多智能体流程，并在筛选阶段引入迭代论证。

**🔧 技术方法**

LLM（OpenAI GPT‑4）+多智能体框架（LangChain 等）+问题树分解、事实答案生成、论证生成与评估、辩论式迭代改进。

**📊 数据集**

259 家真实早期欧洲创业公司的数据，来自五家 VC 观察列表、Crunchbase、公司官网及 Perplexity Sonar API，确保时间与信息可用性无前视偏差。

**📈 对比分析**

与实际 VC 投资表现和单一输入‑输出 LLM 基线对比，使用 Precision‑Recall、AUC‑PR 等指标；DIALECTIC 在验证集上 AUC‑PR 0.25，测试集约 0.24，精度与人类投资者相当，同时提供完整排名。

**⚠️ 局限性**

样本量小、对成功的二元定义简化多维度结果、可能存在残留前视偏差、仅限欧洲早期 VC，难以推广，性能易受超参数和样本组成影响。

---

## 262. From Garbage to Gold: A Data-Architectural Theory of Predictive Robustness

**arXiv ID:** 2603.12288 | [PDF](https://arxiv.org/pdf/2603.12288v1)

**作者:** Terrence J. Lee-St. John `[一作]` (Institute for Healthier Living), Bartlomiej Piechowski-Jozwiak `[通讯]` (Institute for Healthier Living)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并证明了“垃圾到金”的数据架构理论（G2G），阐释了在高维、错误多、共线性强的表格数据中，预测鲁棒性源于数据架构与模型容量的协同，而非单纯数据清洗。

**💡 创新点**

核心创新在于将预测器空间噪声拆分为“预测器误差”和“结构不确定性”，证明在无限宽度（高维）下可跨越两种噪声的上限，并将信息理论、潜在因子模型与心理测量学合成为统一框架；提出“主动数据中心 AI”（P‑DCAI）与“局部工厂”部署范式。

**🔧 技术方法**

技术方法包括信息理论推导（数据处理不等式、条件熵、互信息）、潜在因子模型与因果一致性分析、系统误差模型（系统误差制度）以及对高维统计的渐近分析；结合模拟实验与实际电子病历数据进行实证验证。

**📊 数据集**

使用了克利夫兰阿布达比医院（CCAD）的真实电子病历数据，涵盖588,105名患者、3.4百万患者月、数千个错误多的预测变量，并与传统风险模型对照。

**📈 对比分析**

与传统风险模型和基础机器学习方法对比，G2G在AUC/精准率等指标上实现了显著提升（在CCAD案例中超越标准风险评分），并通过仿真展示宽度（breadth）策略可实现更快的收敛与更低的不确定性。

**⚠️ 局限性**

局限性包括：对潜在结构的假设（如因果一致性、局部独立性）需满足；在存在系统误差或高相关性错误时需要高度灵活的模型；理论主要基于二元变量，连续变量的推广仍需进一步研究；未给出具体实现算法，更多是理论与设计框架。

---

## 263. SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking

**arXiv ID:** 2603.12749 | [PDF](https://arxiv.org/pdf/2603.12749v1)

**作者:** Zheng Gao `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种无训练、基于扩散模型初始噪声的语义水印方法SLICE，利用语义分解和空间分区注入水印，并在检测时进行多粒度验证。

**💡 创新点**

创新点在于将图像语义拆解为主体、环境、动作、细节四个因子，并将每个因子绑定到初始噪声的不同非重叠空间区域，实现局部语义篡改可检测且可定位的水印。

**🔧 技术方法**

采用视觉语言模型（VLM）配合元提示进行语义提取，使用键值哈希生成噪声；利用DDIM逆向采样恢复初始噪声；结合理论分析给出误接受率和篡改定位保证；同时考虑与被动取证技术结合。

**📊 数据集**

在Stable Diffusion V2模型上使用Qwen3‑VL VLM进行语义提取，实验数据集包括Stable‑Diffusion‑Prompts（SDP）和COCO，评估对生成图像和已水印图像的影响。

**📈 对比分析**

与高斯着色、树环、WIND及语义水印SEAL等基线在LFA、RPM、CSI三种生成篡改攻击下对比，SLICE的攻击成功率分别为0%、5%和19%，显著低于对手；在旋转、JPEG压缩、模糊、噪声、亮度等常见扰动下检测准确率保持在0.941-1.0之间；CLIP分数差异仅约0.25分，说明图像质量和语义对齐几乎不受影响。

**⚠️ 局限性**

对极端几何变形（裁剪/缩放）极易破坏空间对应关系，导致水印检测性能骤降；但通过与被动取证方法联合，可在这些攻击下恢复高检出率。

---

## 264. Multi-Step Semantic Reasoning in Generative Retrieval

**arXiv ID:** 2603.12368 | [PDF](https://arxiv.org/pdf/2603.12368v1)

**作者:** Steven Dong `[一作]` (University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**通讯引用:** 28738 | [OpenAlex ID](https://openalex.org/A5031439294)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种名为ReasonGR的生成检索框架，提升多步语义推理能力。

**💡 创新点**

结合结构化提示（任务指令+链式推理）与低秩适配模块，并引入自适应损失以强化推理与检索精度。

**🔧 技术方法**

采用FLAN‑T5 base骨干，LoRA+QLoRA参数高效微调，链式推理提示、伪查询生成和结构化损失函数。

**📊 数据集**

使用FinQA金融问答数据集，包含财务报表与表格信息。

**📈 对比分析**

与BM25稀疏检索及DSI生成检索对比，ReasonGR在EM/PM/SM/S‑Score等指标均优于基线，特别在PM/SM上提升显著。

**⚠️ 局限性**

模型规模有限、输入长度受限、数据集来源非传统IR任务、CoT提示成本高、损失粒度不够细化等是主要局限。

---

## 265. Applying Value Sensitive Design to Location-Based Services: Designing for Shared Spaces and Local Conditions

**arXiv ID:** 2603.12521 | [PDF](https://arxiv.org/pdf/2603.12521v1)

**作者:** Hiruni Kegalle `[一作]` (RMIT University), Danula Hettiachchi `[通讯]` (RMIT University)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5014686840)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过扩展价值敏感设计(VSD)，提出了针对位置服务的Location-Aware Value Sensitive Design（LA‑VSD）框架，并在墨尔本的电动滑板车共享服务案例中进行验证。

**💡 创新点**

创新点在于：①提出三条针对位置服务的具体启发式规则（LBS‑H1、LBS‑H2、LBS‑H3）；②通过空间共享情景和本地化条件来识别间接利益相关者；③在技术层面实现数字与物理层面价值对齐。

**🔧 技术方法**

采用价值敏感设计方法论、启发式规则生成、空间数据分析（缓冲区、机器学习特征重要性）和半结构化访谈等技术手段。

**📊 数据集**

使用的数据集包括三个月的电动滑板车实时行程数据（起点/终点坐标、时间戳）以及本地政策文件、新闻报道、事故记录等文献资料。

**📈 对比分析**

研究未进行量化性能评估；通过案例研究展示启发式规则在概念设计与实证访谈中的可操作性，未与其他方法进行对比。

**⚠️ 局限性**

局限性：仅在单一城市单一服务场景下验证；技术与设计建议为概念性，未实现或实地评估；缺乏跨文化、跨地区的验证，可能影响普适性。

---

## 266. TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation

**arXiv ID:** 2603.12762 | [PDF](https://arxiv.org/pdf/2603.12762v1)

**作者:** Nazar Puriy `[一作]` (IBM Research Europe), Konrad Schindler `[通讯]` (ETH Zurich)

**通讯引用:** 23643 | [OpenAlex ID](https://openalex.org/A5005404030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 TerraFlow，一种融合多模态和多时序学习的 Earth Observation (EO) 预训练框架；

**💡 创新点**

创新点包括：1）在预训练中加入旋转位置编码（RoPE）实现时序注意力，支持可变长度序列；2）提出 Temporal Disjoint Sampling (TDS)，强制模型利用跨时间信息而非单时刻空间捷径；3）在 TerraMind 基础上持续预训练以兼顾跨模态和时序特征；

**🔧 技术方法**

使用 Transformer 编码-解码架构，RoPE 时序编码，TDS 掩码策略，以及交叉熵/均方误差训练目标；

**📊 数据集**

利用 SSL4EO‑S12 v1.1 作为预训练数据集；在 GEO‑Bench‑2（Kuro Siwo、PASTIS、BioMassters、DynamicEarthNet）和灾害风险预测数据集（Kuro Siwo、ImpactMesh）进行下游评估；

**📈 对比分析**

与 TerraMind 及多种基线模型（ViT、ConvNeXt 等）在 GEO‑Bench‑2 上比较，TerraFlow 在 100M 参数规模下在所有时序任务上提升 1–5% mIoU，灾害风险预测中 F1 和 Brier 分数提升 15–30%；

**⚠️ 局限性**

局限性包括：1）在火灾风险预测仍表现欠佳，可能受天气和人类因素限制；2）对极端稀疏时序（如仅 1–2 时刻）时的表现未深入评估；3）计算成本仍随序列长度显著增长，需进一步优化。

---

## 267. VQQA: An Agentic Approach for Video Evaluation and Quality Improvement

**arXiv ID:** 2603.12310 | [PDF](https://arxiv.org/pdf/2603.12310v1)

**作者:** Yiwen Song `[一作]`, Yale Song `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多智能体框架，利用动态问答和VLM反馈实现视频生成的闭环迭代优化。

**💡 创新点**

创新点在于将评估从静态指标转为可交互的问答机制，并通过语义梯度驱动提示优化，避免语义漂移且不依赖模型内部参数。

**🔧 技术方法**

使用了VLM（如 Gemini‑3‑Pro、GPT‑4o）、多智能体协同（QG、QA、PR）、全局VLM评分、早停与阈值判定等技术。

**📊 数据集**

在 T2V‑CompBench、VBench2、VBench‑I2V 三个公开基准上评估。

**📈 对比分析**

相较于 VPO、Best‑of‑N、VQAScore 等基线，平均提升 4–8% 的总分，且迭代次数极少（约 2–4 步）。

**⚠️ 局限性**

局限在于仍需多轮推理导致延迟，且在高稠密基准下改进幅度有限。

---

## 268. daVinci-Env: Open SWE Environment Synthesis at Scale

**arXiv ID:** 2603.13023 | [PDF](https://arxiv.org/pdf/2603.13023v1)

**作者:** Dayuan Fu `[一作]` (SII), Pengfei Liu `[通讯]` (SII)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了OpenSWE，一个规模最大、透明度最高的软件工程环境数据集，包含45,320个可执行Docker环境，并发布完整的多代理合成管道。

**💡 创新点**

1) 规模与透明度：提供12.8k个真实Python仓库、45.3k任务；2) 质量导向过滤：基于难度评估剔除不可解/过于简单环境；3) 多节点分布式合成：64节点集群实现两周完成构建；4) 公开完整管线与成本报告。

**🔧 技术方法**

多代理系统（Repository Exploration、Dockerfile Construction、Evaluation Script Construction、Test Analysis）结合LLM（Deepseek-v3.2、GLM-4.7）实现自动化构建；Docker、Conda、CI/CD；分布式计算与容器资源管理；SFT训练（Qwen2.5-32B/72B）与OpenHands/SWE-Agent scaffolds。

**📊 数据集**

OpenSWE自研数据（12.8k repos, 45.3k envs）；对比SWE-rebench、SWE-rebench-v2、SWE-gym；SWE-Bench Verified评测基准；在OpenSWE训练时采样轨迹。

**📈 对比分析**

在SWE-Bench Verified上使用OpenHands或SWE-Agent评估 Pass@1；OpenSWE-32B 62.4%、OpenSWE-72B 66.0% 领先所有32B/72B SFT方法；与SWE-rebench混合提升至68.0%；数据规模呈log‑linear提升，无饱和；在General benchmarks（HumanEval、GSM8K等）显著提升，无显著事实召回下降。

**⚠️ 局限性**

对Python仓库限定；仍需手工筛选不可解环境；成本高昂（1.47M USD）；规模虽大但仍低于工业级数据；难度评估主观，可能漏掉极难实例；仅验证SFT训练，对RL或自监督不作系统评估。

---

## 269. TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking

**arXiv ID:** 2603.12873 | [PDF](https://arxiv.org/pdf/2603.12873v1)

**作者:** Jiale Meng `[一作]` (Zhejiang University), Yiming Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 4153 | [OpenAlex ID](https://openalex.org/A5100346341)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种结构感知的文档水印框架 TRACE，利用扩散模型对字符结构进行细微编码，实现数据隐藏。

**💡 创新点**

创新点包括：①将字符结构（关键点和笔画）作为编码基础，克服传统边缘或预定义字形库的局限；②提出适配扩散初始化、引导扩散编码和掩膜区域替换三阶段流水线；③引入运动概率评估器和目标点估计，显著提升编码与解码的同步性。

**🔧 技术方法**

技术手段主要为：扩散模型 DragDiffusion + LoRA 微调、DDIM 反演、局部一致性损失、掩膜区域替换；关键点检测基于轻量化 OpenPose，运动概率评估器 (MPE) 与目标点估计 (TPE) 用于指引扩散过程。

**📊 数据集**

使用的实验数据集包含：英文字体 Calibri、Arial、Times New Roman；中文字体 SimSun、SimHei；不同字号（12–36pt）；此外还测试了手写体、艺术字体，并对外语和数学公式进行了验证。

**📈 对比分析**

与 AutoStegaFont、StegaStamp、IHA 等基线对比，TRACE 在截图、打印–扫描、摄像等跨媒介传输下，平均提取准确率 ACC 接近 100% 或 >95%，PSNR > 27 dB，SSIM > 0.99；相较于基线方法，显著提升了鲁棒性和无感知度。

**⚠️ 局限性**

局限性：①依赖关键点检测的准确性，极端噪声或严重结构变形会导致性能下降；②对极小字号和极端字体仍可能出现识别误差；③未对实时推理成本和资源需求做深入评估。

---

## 270. CVGL: Causal Learning and Geometric Topology

**arXiv ID:** 2603.12551 | [PDF](https://arxiv.org/pdf/2603.12551v1)

**作者:** Songsong Ouyang `[一作]` (Shenzhen University), Yingying Zhu `[通讯]` (Shenzhen University)

**通讯引用:** 2481 | [OpenAlex ID](https://openalex.org/A5068185303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合因果学习与几何拓扑的跨视角地理定位框架（CLGT），通过Causal Feature Extractor抑制非因果干扰、GT Fusion融合鸟瞰视角道路拓扑、DA Pooling自适应聚合特征，实现街景与航空图的高精度匹配。

**💡 创新点**

创新点包括：①首创将因果干预（back‑door）引入跨视角定位；②设计Content‑aware Mask在频域分离因果中频与非因果低/高频；③提出GT Fusion利用跨注意力与Dual Dynamic Fusion实现路网拓扑与街景特征的强交互；④引入DA Pooling多池化融合提升语义表达。

**🔧 技术方法**

使用技术包括：Discrete Cosine Transform (DCT) + 频域掩码、深度卷积、跨注意力、Overlapping Spatial Reduction、Dual Dynamic Fusion、InfoNCE 对比损失、Gaussian 随机化、Gate Pooling、AdamW 优化、ConvNeXt-B backbone。

**📊 数据集**

主要数据集：CVUSA、CVACT、VIGOR；以及其鲁棒增强版本 CVUSA‑C‑ALL、CVACT‑val‑C‑ALL、CVACT‑test‑C‑ALL，涵盖 16 种视觉失真。

**📈 对比分析**

与多种最新方法（SAFA、LPN、DSM、TransGeo、GeoDTR、Sample4G、EP‑BEV 等）对比，CLGT 在 CVUSA、CVACT 以及鲁棒数据集上均实现 Recall@1 提升 1–2%（最高 1.81%），在鲁棒集上平均提升 5%，在跨数据集迁移中提升 5.5% 以上，显示出更强的泛化与鲁棒性。

**⚠️ 局限性**

局限性：BEV 通过几何变换生成仍含噪声，导致拓扑融合效果受限；因果干预仅针对频域掩码，未能完全捕捉所有非因果干扰；模型对极端遮挡与新型光照条件的鲁棒性仍有提升空间。

---

## 271. VLM4Rec: Multimodal Semantic Representation for Recommendation with Large Vision-Language Models

**arXiv ID:** 2603.12625 | [PDF](https://arxiv.org/pdf/2603.12625v1)

**作者:** Ty Valencia `[一作]` (University of Southern California), Wei Yang `[通讯]` (University of Southern California)

**通讯引用:** 37343 | [OpenAlex ID](https://openalex.org/A5100781368)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于大视觉‑语言模型的轻量级多模态语义表征框架VLM4Rec，利用LVLM将商品图像转化为自然语言描述，再用句子编码器生成稠密语义向量进行检索推荐。

**💡 创新点**

创新点在于将多模态推荐视为语义对齐问题，而非仅关注特征融合；通过离线语义生成与语义空间编码，提升了表示质量，证明了表示优先于复杂融合。

**🔧 技术方法**

主要技术包括LLaVA-NeXT 7B作为视觉‑语言模型进行图像描述生成、Sentence‑BERT作为文本编码器、余弦相似度检索与简单均值池化用户画像。

**📊 数据集**

实验使用Kaggle多模态推荐数据集（Clothing, Shoes & Jewelry）——约23k用户、38k商品、179k交互；在覆盖约12%商品的子集上进行评估。

**📈 对比分析**

与BERT标题编码、CLIP视觉特征、各类融合（注意力、拼接、平均、SMORE、门控）等基线对比，VLM4Rec的文本仅版在Recall@10、NDCG等指标上提升约54%（相对BERT），并超过所有融合模型。

**⚠️ 局限性**

局限性：1）仅覆盖约12%商品，缺乏全量覆盖评估；2）仅使用轻量检索，未验证与更复杂推荐头的互补性；3）依赖较大的LVLM推理成本，实际上线可扩展性待验证。

---

## 272. Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation

**arXiv ID:** 2603.13099 | [PDF](https://arxiv.org/pdf/2603.13099v1)

**作者:** Wayner Barrios `[一作]` (Dartmouth), SouYoung Jin `[通讯]` (Dartmouth)

**通讯引用:** 308 | [OpenAlex ID](https://openalex.org/A5074900529)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CRYSTAL基准，评估多模态大模型的可验证中间推理步骤；

**💡 创新点**

引入Match F1与Ordered Match F1两种指标，结合多代理 Delphi 风格的参考生成以及乘法式奖励CPR-Curriculum，突破仅评估答案的短板；

**🔧 技术方法**

利用多模态链式思考（CoT）生成、语义聚类、句子编码相似度匹配、LIS 级联排序、强化学习 GRPO 与 PCGrad 训练；

**📊 数据集**

基准包含6,372个来自 MathVision、ScienceQA‑IMG、RealWorldQA、MMVP、PLOTQA 的问题，平均 11.6 步推理；

**📈 对比分析**

在20个公开与商业模型上测试，发现普遍的“挑拣”现象、规模非单调性与无序推理；CPR‑Curriculum 在 Qwen2.5‑VL‑3B 上将 Match F1 提升 32%，并兼顾准确率；

**⚠️ 局限性**

参考路径可能不全，固定句子编码器与阈值在特定领域可能不足，Ordered Match F1 目前未显式建模因果步依赖。

---

## 273. RadEar: A Self-Supervised RF Backscatter System for Voice Eavesdropping and Separation

**arXiv ID:** 2603.12446 | [PDF](https://arxiv.org/pdf/2603.12446v1)

**作者:** Qijun Wang `[一作]` (Michigan State University), Huacheng Zeng `[通讯]` (Michigan State University)

**通讯引用:** 1840 | [OpenAlex ID](https://openalex.org/A5027120851)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了一种基于RF背散射的无电池标签和读卡器系统，能够在隔音墙后窃听并通过自监督学习实现语音分离与降噪。

**💡 创新点**

创新点包括：① 双共振器结构实现频域自干扰抑制与连续FM编码；② 采用自监督混音式分离与降噪模型，并通过反馈循环实现自强化；③ 低功耗、可长距离、无需电池的标签设计。

**🔧 技术方法**

技术手段主要有：双共振器（VSR+PR）与压电传感器、磁耦合的频域FM调制；自监督混音训练的Conv‑TasNet和SuDoRM‑RF模型；低频915 MHz背散射与CW发射；自监督MixIT/RemixIT框架。

**📊 数据集**

使用LibriMix对分离模型进行预训练，DNS对降噪模型进行预训练；实验数据来自自采集的室内声学环境。

**📈 对比分析**

与基准方法对比，系统在2.5 m内可获得SI‑SDR≈10.87 dB、STOI≈0.67、PESQ≈2.83、LLR≈0.51；消融实验表明预训练、EMA更新和分离-降噪反馈显著提升性能。

**⚠️ 局限性**

局限性在于对多说话者、噪声、距离仍有一定限制；依赖强RF源与墙体材料；标签位置、墙厚会影响性能；目前仅支持单标签、单房间场景，且需要持续功率传输。

---

## 274. VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation

**arXiv ID:** 2603.12918 | [PDF](https://arxiv.org/pdf/2603.12918v1)

**作者:** Juhye Park `[一作]` (KAIST), Hyun Myung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VIRD 方法，利用双轴变换构建视角不变的描述子，实现全方位无方向先验的跨视角定位。

**💡 创新点**

通过极坐标投影结合上下文增强位置注意力的双轴变换，以及视角重建损失，显著弥合地面与卫星图像的视角差距。

**🔧 技术方法**

使用极坐标投影、上下文增强位置注意力（CEPA）、视角重建损失、深度卷积特征提取、余弦匹配和残差回归等技术。

**📊 数据集**

在 KITTI 与 VIGOR 两个真实城市数据集上进行实验。

**📈 对比分析**

与 SliceMatch、CCVPE、FG2 等先进方法对比，在无方向先验下在 KITTI 上实现 50.7%/76.5% 的中值位置/姿态误差提升，在 VIGOR 上实现 18.0%/46.8% 的提升。

**⚠️ 局限性**

仅针对 3-DoF 定位，未处理 6-DoF 或多摄像头场景，且对极端垂直结构的处理仍有限。

---

## 275. Mask2Flow-TSE: Two-Stage Target Speaker Extraction with Masking and Flow Matching

**arXiv ID:** 2603.12837 | [PDF](https://arxiv.org/pdf/2603.12837v1)

**作者:** Junwon Moon `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5064051041)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种两阶段的目标说话人提取框架 Mask2Flow‑TSE，将轻量级的消除式掩码网络与流匹配生成器串联，实现从混合语音到目标语音的快速转换。

**💡 创新点**

创新点在于：①通过删除–插入（D/I）分析揭示流匹配模型在早期主要执行消除，而掩码网络正好承担这一任务；②把掩码输出作为流匹配的起点，显著缩短生成轨迹，只需单步 Euler 推导即可完成高质量重构；③首次在 TSE 任务中结合判别式掩码与生成式流匹配，兼顾速度、模型规模与语音质量。

**🔧 技术方法**

技术要点包括：WavLM 作为说话人编码器；掩码网络由卷积层+双向 LSTM 构成，输出软掩码；流匹配采用 DiT（Diffusion Transformer）架构，使用 AdaLN‑Zero 加时步和说话人嵌入的双重条件；rectified flow matching 的直线轨迹学习；单步 Euler 整合；整体使用 rectified flow matching 与 Masking 的两阶段并行训练。

**📊 数据集**

主要使用 Libri2Mix（基于 LibriSpeech + WHAM! 语音噪声）以及在自定义测试集（清音、语音加噪、混响）上的评估，训练时还通过在 LibriSpeech 上的实时数据增强生成混合语音。

**📈 对比分析**

与判别式基线 ConVoiFilter、生成式基线 TSELM 与 Metis‑TSE 进行比较，使用 Whisper 各级别（tiny、base、small、medium、large‑v2）评估 WER。Mask2Flow‑TSE 在噪声环境下在所有 Whisper 模型上均获得最低 WER，并且单步推理仅需 85M 参数，RTF 与 ConVoiFilter 相当，却在语音质量与说话人相似度上不失分；在干净语音上保持原始质量。

**⚠️ 局限性**

局限性包括：在极端嘈杂或混响条件下仍略逊于纯判别式掩码模型，模型容量仍可进一步提升；当前仅在 log‑mel 频谱域实现，需要进一步验证在更高分辨率时域或多通道环境下的适用性。

---

## 276. Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights

**arXiv ID:** 2603.13186 | [PDF](https://arxiv.org/pdf/2603.13186v1)

**作者:** Xingli Fang `[一作]` (North Carolina State University), Jung-Eun Kim `[通讯]` (North Carolina State University)

**通讯引用:** 2666 | [OpenAlex ID](https://openalex.org/A5100462673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于权重级别隐私脆弱性估计的会员推理防护方法（CWRF），通过把少量高隐私风险的权重重置为初始值并冻结其余权重，随后用已有的隐私保护训练策略进行细调；

**💡 创新点**

创新点在于：①首次将机器无学习（Machine Unlearning）思想用于评估单个权重的隐私脆弱性；②发现隐私脆弱权重占比极小且与学习重要性高度相关，导致传统剪枝无效；③证明权重的重要性取决于位置而非数值，因而重置高风险权重并冻结其余权重可在保持精度的同时降低隐私泄露；

**🔧 技术方法**

技术包括权重级别隐私脆弱性评分（基于交叉熵+KL散度的双重损失）、权重重置与冻结掩码、隐私细调（可接入DP‑SGD、RelaxLoss、HAMP、CCL等已有隐私训练方案）、学习率重置等；

**📊 数据集**

使用了CIFAR‑10、CIFAR‑100、CINIC‑10三大图像分类数据集，模型分别为ResNet18和ViT；

**📈 对比分析**

与未加防护模型以及四种主流隐私训练方法（DP‑SGD、RelaxLoss、HAMP、CCL）对比，在LiRA和RMIA两种会员推理攻击下，CWRF在保持或提升测试准确率的同时，显著降低AUC和TPR，尤其在ResNet18+CIFAR‑10上LiRA的AUC下降超过80%；

**⚠️ 局限性**

局限性包括：①依赖于基础隐私训练方法的有效性，若基础方法本身对MIAs无效，CWRF提升有限；②在某些组合（如RelaxLoss+RMIA）下可能略升隐私风险；③需手动调节重置比例和迭代次数，对不同模型/任务可能不一致；④仅针对监督分类任务，尚未验证在回归或生成任务中的表现。

---

## 277. AI Planning Framework for LLM-Based Web Agents

**arXiv ID:** 2603.12710 | [PDF](https://arxiv.org/pdf/2603.12710v1)

**作者:** Orit Shahnovsky `[一作]` (University of Haifa), Rotem Dror `[通讯]` (University of Haifa)

**通讯引用:** 813 | [OpenAlex ID](https://openalex.org/A5089112142)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了将 LLM 驱动的 Web 代理与传统规划范式（BFS、最佳优先搜索、DFS）对应的分类体系，并基于此定义了五个轨迹级评估指标；同时构建了包含 794 条人类标注轨迹的 WebArena 数据集，并实现了首个全程预规划（Full‑Plan‑in‑Advance）的 LLM 代理，随后与现有的逐步执行（Step‑by‑Step）代理进行对比实验。

**💡 创新点**

创新点包括：① 将 LLM 代理与经典规划算法映射成统一的术语，弥补了对代理决策过程的黑箱缺口；② 提出针对轨迹质量的五大评估指标，超越单纯的成功率；③ 创建人类标注的轨迹数据集，提供参考路径；④ 首次实现并验证全程预规划的 LLM 代理。

**🔧 技术方法**

技术方面：使用 GPT‑4o‑mini 进行自然语言推理与规划；采用 Accessibility Tree 结构化网页表示；通过 Playwright 执行浏览器操作；利用 LLM‑as‑Judge 对轨迹进行语义匹配与指标计算；实现树搜索与 DFS 预规划逻辑。

**📊 数据集**

数据集：WebArena 基准任务（共 812 条），其中 794 条已被人工完成并标注为黄金轨迹；此外使用了 WebArena 的任务描述、目标判定与界面截图。

**📈 对比分析**

实验对比：在 WebArena 上对 Step‑by‑Step（WebArena 代理）和 Full‑Plan‑in‑Advance 两种规划方式进行评测。总体成功率为 38.41%（Step‑by‑Step）vs 36.29%（Full‑Plan）；Step‑Success 率 82% vs 58%；Element Accuracy 82% vs 89%；Recovery Rate 36% vs 31%；Partial Success 极低。说明全程预规划在技术指标上优于逐步执行，但在任务完成率与轨迹恢复能力上稍逊。

**⚠️ 局限性**

局限性：全程预规划代理在动态、非结构化 Web 环境中易出现漂移与早停；对 LLM‑as‑Judge 的依赖可能带来评估偏差；实验仅基于 WebArena，缺乏跨域验证；未针对性能进行调优，无法证明最佳实践。

---

## 278. Federated Hierarchical Clustering with Automatic Selection of Optimal Cluster Numbers

**arXiv ID:** 2603.12684 | [PDF](https://arxiv.org/pdf/2603.12684v1)

**作者:** Yue Zhang `[一作]` (Guangdong Polytechnic Normal University), Yiqun Zhang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5100329232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在联邦聚类任务中提出了Fed‑k*‑HC框架，能够在单轮通信下自动确定最佳聚类数并处理数据分布不均衡问题。

**💡 创新点**

创新点包括：①使用微分簇（micro‑subcluster）细粒度划分客户端数据，②采用合成高斯样本上传以保护隐私，③在服务器端通过SNC（基于GCS和严格自然邻居）自适应估计k*，④在合并过程中采用密度与重叠度共同度量的特殊距离实现层次聚合，消除传统划分法的“均匀效应”。

**🔧 技术方法**

技术手段包括：客户端层次聚类与SNP（多原型竞争学习）实现微分簇，合成多元正态样本上传；服务器层面的SNC算法（GCS+严格自然邻居）估计k*，随后采用基于重叠度和标准差的相似度度量进行层次合并；实验中使用了F‑measure、Accuracy、NMI、ARI、DCV等五个评估指标。

**📊 数据集**

实验数据集：六个合成数据集（ids2、gaussian及其变体）、五个UCI真实数据集（pageblock、yeast、abalone、breast、digits），并在IID与Non‑IID、平衡与不平衡的多种分布下进行评估。

**📈 对比分析**

与KFed、MUFC、F3KM、Orchestra、Orchestra*、Fed‑AP、Fed‑MS、Fed‑KDPC等八种基线方法对比，Fed‑k*‑HC在绝大多数数据集与评估指标上获得排名第一（或第二），尤其在处理不平衡聚类时表现显著优异；在部分如breast数据集上表现略逊，但整体可视化和指标均优于传统方法。

**⚠️ 局限性**

局限性包括：①服务器层次合并对极大客户端数或样本量时计算量急剧增大；②在极端不平衡或微小簇极小于微分簇时识别效果下降；③隐私保护仅限于合成样本，未加入差分隐私或同态加密等更强的隐私增强机制。

---

## 279. ODRL Policy Comparison Through Normalisation

**arXiv ID:** 2603.12926 | [PDF](https://arxiv.org/pdf/2603.12926v1)

**作者:** Jaime Osvaldo Salas `[一作]` (University of Southampton), George Konstantinidis `[通讯]` (University of Southampton)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5000674196)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种参数化的ODRL策略规范化方法，将复杂逻辑约束和数值区间拆分为简单规则，保持语义一致性并实现策略比较。

**💡 创新点**

通过正则化、简化逻辑约束以及区间拆分，将权限与禁止规则统一为仅权限（或仅禁止）的互斥简单规则集合，从而把策略比较问题转化为简单规则的相等检测。

**🔧 技术方法**

采用Salas等人定义的ODRL语义，利用布尔代数标准化、析取范式展开、数值区间拆分以及专门的正则化与拆分算法，并证明语义保持性。

**📊 数据集**

论文未使用公开数据集，示例策略仅用于演示原型实现，实验结果在本文未给出。

**📈 对比分析**

先将规则规范化为简单规则集合，再通过集合相等/包含/交集检测实现策略比较；复杂度为属性数指数、常数取值线性，实验表明在典型场景下比较速度可接受，但最坏情况仍为指数级。

**⚠️ 局限性**

仅覆盖ODRL核心元素（权限/禁止），不支持义务、职责、集合约束和推理；对集合约束或类型等的支持有限；当属性数或常数多时，规范化与拆分步骤会产生指数级增长。

---

## 280. Lattice Discrete Particle Model (LDPM): Comparison of Various Time Integration Solvers and Implementations

**arXiv ID:** 2603.13190 | [PDF](https://arxiv.org/pdf/2603.13190v1)

**作者:** Erol Lale `[一作]` (Istanbul Technical University), Gianluca Cusatis `[通讯]` (Northwestern University)

**通讯引用:** 7089 | [OpenAlex ID](https://openalex.org/A5024267376)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对七种不同的 LDPM 实现（显式/隐式、CPU/GPU、静态/瞬态求解器）在同一套基准测试（弹性振动、受压、拉伸、弯曲、无侧限压缩）上进行系统比较。

**💡 创新点**

首次提供开放源码/免费实现与完整基准数据集，构建多维度评价框架（宏观响应、能量误差、迭代次数、计算时间、裂纹相关性），并讨论极端非线性破裂下的数值挑战。

**🔧 技术方法**

使用 LDPM 物理模型，配合中心差分、HHT、Generalized‑α 等时间积分方法，GPU 加速的 JAX、自动微分、稀疏直接求解器 PARDISO 等技术；同时采用随机几何粒子布置与 Fuller's 颗粒分布。

**📊 数据集**

基准数据集包括七个测试案例（线性振动、受压、拉伸、三点弯曲、无侧限压缩），使用相同材料参数（Fuller 颗粒分布、c=375 kg/m³、w/c=0.5 等），公开于 Zenodo。

**📈 对比分析**

通过比较宏观应力‑应变曲线、能量守恒误差、迭代次数、计算时间、裂纹开口的 Pearson 相关系数和 NRMSE；结果显示大多数实现误差在 1–3 % 以内，隐式求解在高度非线性裂纹时更稳定，显式/GPU 版本在多线程/并行环境下能获得更快速度。

**⚠️ 局限性**

在无侧限压缩等极端破裂情形下，所有实现均出现收敛或稳定性问题；显式方法在慢速加载或长时间实验中不可行；比较受限于 LDPM 随机几何与材料随机性导致的内在差异。

---

## 281. Push, Press, Slide: Mode-Aware Planar Contact Manipulation via Reduced-Order Models

**arXiv ID:** 2603.12399 | [PDF](https://arxiv.org/pdf/2603.12399v1)

**作者:** Melih Özcan `[一作]` (Middle East Technical University), Umut Orguner `[通讯]` (Middle East Technical University)

**通讯引用:** 3706 | [OpenAlex ID](https://openalex.org/A5052094664)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于模式的平面非抓握操纵框架，利用简化的动力学模型和一次性代数力分配实现单臂和双臂任务的优化无关控制。

**💡 创新点**

创新点包括：1）引入对顶压推拉的体固定跟踪点，实现无模式切换的非齐次控制；2）将多种接触拓扑统一归纳为有限数目物理直观的低阶模型；3）设计O(1)代数力分配器和CoP驱动实现准全自由运动；4）在双臂情形下实现CoP steering提升机动性。

**🔧 技术方法**

利用极限表面理论、仿真的水动力接触模型、非齐次车辆类模型（单车、双轮、差速驱动）、PI力调节、阻尼控制、KTO轨迹优化，以及在Drake中对Franka Panda和KUKA iiwa7的实验。

**📊 数据集**

未使用公开数据集，仅在Drake仿真中生成合成实验数据。

**📈 对比分析**

与仅使用推力执行的对比实验显示加入臂跟踪项后执行时间从>100秒降至约30秒，位移误差从0.095 m降至0.085 m；推压滑实验实现<0.5 cm、1°误差；双臂实验成功完成狭窄槽口与按压入位等复杂任务。

**⚠️ 局限性**

局限性包括：仅在仿真中验证，缺乏硬件实验；对传感噪声、延迟和未建模摩擦的鲁棒性未知；对重/柔性物体导致的压力分布变化未充分考虑；假设静态或准静态，未涵盖动态大惯性情形。

---

## 282. Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**arXiv ID:** 2603.12468 | [PDF](https://arxiv.org/pdf/2603.12468v1)

**作者:** Alexis Guichemerre `[一作]` (ÉTS Montréal), Eric Granger `[通讯]` (ÉTS Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出一种针对组织病理学弱监督定位的无源域适应方法，旨在纠正目标域中主导类别的预测偏差并提升分类与定位性能。

**💡 创新点**

创新点在于将机器无学习理念引入域适应，通过动态遗忘不确定样本、保留可信样本以及联合像素级定位头实现对预测偏差的迭代纠正。

**🔧 技术方法**

主要技术包括：预测偏差检测、基于熵的遗忘/保留样本选择、交叉熵损失的正向/反向优化、CAM一致性引导的像素级二分类头以及动态重采样策略。

**📊 数据集**

实验使用跨器官与跨中心的公开组织病理学基准数据集：GlaS、TCGA‑Colon、TCGA‑Breast 与其改造的WSOL兼容子集。

**📈 对比分析**

与多种领先的源无监督域适应方法（SFDA‑DE、CDCL、ERL、RGV、SAT、DeepMIL）进行比较，平均提升了约12%‑20%的分类准确率和定位召回率，尤其在严重主导类别偏差的中心上表现显著。

**⚠️ 局限性**

局限性包括：对伪标签质量高度依赖，需要手动调参；在极端域偏移下可能仍难以完全消除偏差；算法计算复杂度相对较高，且对像素级头的构造有一定经验门槛。

---

## 283. SDF-Net: Structure-Aware Disentangled Feature Learning for Opticall-SAR Ship Re-identification

**arXiv ID:** 2603.12588 | [PDF](https://arxiv.org/pdf/2603.12588v1)

**作者:** Furui Chen `[一作]` (Chinese Academy of Sciences), Shengyang Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2038 | [OpenAlex ID](https://openalex.org/A5111003697)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于物理先验的跨模态船舶重识别框架SDF-Net，利用结构一致性约束和特征解耦融合实现光学与SAR图像之间的高效匹配。

**💡 创新点**

创新点在于：①引入结构一致性约束，在Transformer中提取梯度能量统计作为尺度不变的几何锚点；②将最终特征分解为共享身份特征和模态特定残差，并采用参数无关的加法残差融合，兼顾判别力与多模态信息。

**🔧 技术方法**

采用Vision Transformer backbone、跨模态双头分词器、梯度能量统计、Instance Normalization、结构一致性损失、正交约束、加法残差融合以及交叉熵+triplet等损失。

**📊 数据集**

在HOSS‑ReID数据集上进行训练与评估，该数据集包含光学与SAR两种模态的船舶图像。

**📈 对比分析**

与多种基准模型（包括ViT、TransOSS、VersReID等）对比，SDF-Net在All‑to‑All、Optical‑to‑SAR和SAR‑to‑Optical三种检索协议下分别获得60.9% mAP/69.9% Rank‑1、70.0% mAP/76.1% Rank‑1等领先成绩，明显优于现有最优方法。

**⚠️ 局限性**

局限性包括：梯度能量在极低分辨率或强噪声的SAR图像中可能失效；框架假设视角近垂直，难以直接处理大倾斜或多视角引起的三维几何畸变。

---

## 284. Easy-IIL: Reducing Human Operational Burden in Interactive Imitation Learning via Assistant Experts

**arXiv ID:** 2603.12769 | [PDF](https://arxiv.org/pdf/2603.12769v1)

**作者:** Chengjie Zhang `[一作]` (Southern University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 34012 | [OpenAlex ID](https://openalex.org/A5100430306)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Easy-IIL 框架，通过引入助手专家，利用一条离线示范即可初始化模型基准专家，并在大多数数据采集阶段代替人工操作；

**💡 创新点**

创新点在于仅需一次人工示范即可启动助手专家，后续离线与在线数据采集基本由助手专家完成，显著降低人工作量同时保持与传统 IIL 基线相当的性能；

**🔧 技术方法**

采用模型基准助手专家（Grounded‑SAM、SuperPoint、LightGlue、ICP、RANSAC 等）生成轨迹；端到端使用 Diffusion Policy，并配合随机切换动作块、噪声注入以及在瓶颈区域禁用 novice 行动等技术来提升在线学习质量；

**📊 数据集**

在 RLBench 上进行的 Basketball in Hoop 与 Take Chicken in Saucepan 两个仿真任务，以及在 Franka Research 3 机器人上的 Hang the Cup 与 Put Duck in Cooker 两个真实任务，使用 10 条离线示范、4 轮在线回合、每轮 5 条交互轨迹等数据；

**📈 对比分析**

与 HG‑DAgger、IWR、Sirius 等主流 IIL 基线进行对比；实验表明 Easy‑IIL 的成功率与基线相当，且人类干预率下降 4–5 倍；用户研究亦显示主观工作负担显著降低；

**⚠️ 局限性**

局限性包括：仍需人工一次示范，且实验仅覆盖有限任务与机器人，难以验证在更大规模数据或更复杂任务中的表现；此外，助手专家对视觉预训练模型依赖较高，可能限制其在缺乏高质量视觉基础模型时的应用。

---

## 285. MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins

**arXiv ID:** 2603.12936 | [PDF](https://arxiv.org/pdf/2603.12936v1)

**作者:** WenBo Xu `[一作]` (Hefei University of Technology), RuoNan Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5621 | [OpenAlex ID](https://openalex.org/A5100728390)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将静态3D网格自动转换为可在物理仿真中使用的 URDF 数字孪生；

**💡 创新点**

① 3D 原生细粒度分割结合 SP4D 物理先验引导 VLM 聚类，消除运动幻觉；② 基于接触界面与类型感知的几何初始化+物理约束轨迹优化，保证碰撞自由；③ 完整零射线框架实现全流程自动化；

**🔧 技术方法**

P3‑SAM（细粒度分割）、SP4D（多视角物理先验）、GPT‑4o VLM（语义聚类）、PCA、RANSAC、SDF 轨迹优化、Levenberg‑Marquardt 约束优化、Hunyuan3D 纹理重建等；

**📊 数据集**

PartNet‑Mobility（带 URDF 真实标注）、Objaverse（海量静态模型）、生成式 3D 资产（文本/图像转 3D），并手工标注部分测试集；

**📈 对比分析**

与 Articulate‑Anything、Articulate‑AnyMesh、URDFormer、SINGAPO、PARIS 等基线对比；mIoU 0.86、Count Acc 0.92、轴误差 0.12、pivot 误差 0.10，物理可执行率 87%，显著优于基线（最高 46%）；

**⚠️ 局限性**

对极其复杂内部结构仍可能出现分割细化不足；依赖多视角渲染与 VLM 推理，计算成本相对较高；对细小部件的分割精度有待提升。

---

## 286. A Holistic Framework for Automated Configuration Recommendation for Cloud Service Monitoring

**arXiv ID:** 2603.12268 | [PDF](https://arxiv.org/pdf/2603.12268v1)

**作者:** Anson Bastos `[一作]` (Microsoft), Rujia Wang `[通讯]` (Microsoft)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5073855975)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种端到端的监控配置推荐框架，帮助云服务自动生成监控指标、维度、表达式和阈值。

**💡 创新点**

创新点在于：①将监控配置拆解为指标、维度、表达式三子模块并用图神经网络联合学习；②使用多头注意力与元路径增强的异构图模型；③通过LLM整合相似监控结果生成阈值，并提供可解释性提示。

**🔧 技术方法**

核心技术包括：异构图神经网络（带多头注意力与元路径）、对比学习、Transformer‑based 句子嵌入、LLM（GPT‑4o）推理、深度集成学习（BCE+对比损失）。

**📊 数据集**

实验数据来自微软内部：3600个账号/服务、约60k监控、14k指标、7k维度、27k表达式、200万告警条件；并在这些真实历史记录上进行训练与评估。

**📈 对比分析**

与SVD、协同过滤、传统MLP、SAGE、GAT、HGT、HAN、HetGNN、Transformer等基线比较，指标选择精度0.866、维度推荐HR@1 0.597、表达式推荐HR@1 0.535，整体框架在LLM与人工评估上取得0.62一致性和0.673相关性，说明性能显著优于基线。

**⚠️ 局限性**

局限性包括：依赖公司内部历史配置，可能难以迁移到其他云平台；阈值生成仍以相似监控为依据，缺少实时时序分析；评估主要通过LLM与人工评分，缺乏真实故障场景的验证。

---

## 287. Adaptive Conditional Forest Sampling for Spectral Risk Optimisation under Decision-Dependent Uncertainty

**arXiv ID:** 2603.12507 | [PDF](https://arxiv.org/pdf/2603.12507v1)

**作者:** Marcell T. Kurbucz `[一作]` `[通讯]` (Institute for Global Prosperity), Marcell T. Kurbucz (Institute for Global Prosperity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种四阶段的 Adaptive Conditional Forest Sampling (ACFS) 方法，用于在决策相关不确定性下最小化包含 CVaR 的光谱风险目标。

**💡 创新点**

将 Generalised Random Forests 用作决策条件分布逼近，结合 CEM+DE 全局搜索、两阶段重排序以及抗差分/CRN 低方差局部优化，系统解决决策相关尾部估计误差导致的排序失效。

**🔧 技术方法**

Generalised Random Forest、Cross-Entropy Method、Differential Evolution、KDE surrogate、两阶段重排序、抗差分和共用随机数的梯度估计。

**📊 数据集**

两个合成基准：DGP1（决策相关 Student‑t copula）和 DGP2（高斯 copula + 对数正态边缘），分别模拟需求驱动与供应链风险。

**📈 对比分析**

与 GP‑BO、CEM‑SO、SGD‑CVaR、KDE‑SO 等对照，100 次重复实验中 ACFS 在 DGP2 上显著降低中位风险 6–20%，在 DGP1 与 GP‑BO 具有相同中位但 SD 下降 1.8–1.9 倍，整体在多项检验中表现最优。

**⚠️ 局限性**

受限于单一模拟预算、维度上限（d_x≈5）、对单次重排序的点估计敏感，且在 50 次重复中缺乏统计显著性，未来需引入多样本/噪声鲁棒重排序与高维稀疏化。

---

## 288. SpectralGuard: Detecting Memory Collapse Attacks in State Space Models

**arXiv ID:** 2603.12414 | [PDF](https://arxiv.org/pdf/2603.12414v1)

**作者:** Davi Bonetto `[一作]` `[通讯]` (Independent Researcher), Davi Bonetto (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化了状态空间模型中离散化转移算子的谱半径对记忆与推理能力的影响，并提出SpectralGuard监控机制以检测谱崩塌攻击。

**💡 创新点**

首次通过谱半径阈值建立记忆崩塌理论，并证明仅基于输出的防御无法检测此攻击；提出实时多层谱监测防御并提供理论证明。

**🔧 技术方法**

结合控制理论、谱半径分析、梯度攻击（HiSPA）、功率迭代估计、Logistic 分类器和实验验证等技术。

**📊 数据集**

使用Pile预训练语料以及LongBench、GSM8K、HumanEval、Associative Recall等多任务基准。

**📈 对比分析**

与多种输出级别防御、基准模型对比，SpectralGuard在非自适应攻击下F1 0.961，自适应攻击下0.842，检测延迟<15 ms/token，在130M–2.8B模型及Zamba2混合架构上保持优异性能。

**⚠️ 局限性**

存在多层监测约15%运行时开销，强自适应攻击可部分规避阈值检测，需进一步研究词汇隐蔽与谱攻击的Pareto前沿，以及对混合注意力路径的完整评估。

---

## 289. DART: Input-Difficulty-AwaRe Adaptive Threshold for Early-Exit DNNs

**arXiv ID:** 2603.12269 | [PDF](https://arxiv.org/pdf/2603.12269v1)

**作者:** Parth Patne `[一作]` (Brandenburg Technical University), Michael Hübner `[通讯]` (Brandenburg Technical University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DART框架，利用输入难度感知的自适应阈值改进早期退出深度网络，实现更高效的推理；

**💡 创新点**

创新点包括：①轻量级多模态难度估计模块；②利用动态规划进行全局联合退出阈值优化；③在线自适应系数管理实现持续改进；

**🔧 技术方法**

采用了边缘密度、像素方差、梯度复杂度等多模态难度度量，动态规划求解最佳阈值，UCB1多策略选择，强化学习/多出口损失，结合DAES效率评分；

**📊 数据集**

在MNIST、CIFAR‑10以及LeViT‑128s/192/256（CIFAR‑10）等数据集上进行实验；

**📈 对比分析**

与静态模型、BranchyNet及RL‑Agent等基线对比，DART在CNN上获得最高3.3×速度提升、5.1×能耗降低、42%平均功耗下降，精度保持竞争；在Vision Transformer上实现2.5–3.6×加速，但精度下降多达17%；

**⚠️ 局限性**

局限性包括：对Transformer的适配不佳导致精度显著下降；对极端难度样本的误判仍存在；对分布漂移和不同任务的在线适应仍需进一步研究。

---

## 290. Thermodynamics of Reinforcement Learning Curricula

**arXiv ID:** 2603.12324 | [PDF](https://arxiv.org/pdf/2603.12324v1)

**作者:** Jacob Adamczyk `[一作]` (University of Massachusetts Boston), Rahul V. Kulkarni `[通讯]` (University of Massachusetts Boston)

**通讯引用:** 3139 | [OpenAlex ID](https://openalex.org/A5048266570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将强化学习的课程学习建模为一个几何问题，通过最小化非平衡热力学中的超额功来推导最优任务调度，并提出基于此的MEW温度退火算法；

**💡 创新点**

创新点在于首次将非平衡统计力学中的摩擦张量映射为任务空间的伪黎曼度量，证明最优课程对应该度量的测地线，并将这一理论应用于最大熵RL的温度退火；

**🔧 技术方法**

采用线性响应理论、Green-Kubo关系计算摩擦张量、Euler-Lagrange和测地方程求解最优路径、以及MEW自适应温度调度公式；

**📊 数据集**

主要实验使用MuJoCo的Humanoid-v5连续控制任务（并在图示中引用了7×7 Grid World做示例）；

**📈 对比分析**

与SAC的自动温度调整以及两种固定温度方案对比，MEW在Humanoid-v5上获得更高回报、收敛更快且温度曲线更稳定；

**⚠️ 局限性**

局限性包括：摩擦张量估计在深度RL中计算开销大、目前仅在线性奖励参数化和单维温度退火上验证、缺乏大规模连续学习基准的实证验证。

---

## 291. Almost-Free Queue Jumping for Prior Inputs in Private Neural Inference

**arXiv ID:** 2603.12946 | [PDF](https://arxiv.org/pdf/2603.12946v1)

**作者:** Qiao Zhang `[一作]` (Shandong University), Xiuzhen Cheng `[通讯]` (Shandong University)

**通讯引用:** 18044 | [OpenAlex ID](https://openalex.org/A5100692488)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 PrivQJ 框架，解决私有神经网络推理中队列跳跃（优先级处理）带来的等待成本问题，实现在保持整体性能不下降的前提下几乎不增加加密和通信开销的优先输入推理；

**💡 创新点**

核心创新在于利用批量输入中的密文槽闲置空间进行“槽回收”，通过链式批处理与最小批量大小设计，使得先行输入可在同一批计算中几乎无额外成本完成；

**🔧 技术方法**

技术手段包括混合同态加密（BFV）与多方安全计算（OT、ASS）实现离线-在线协议，SIMD 计算、im2col 矩阵乘法、密文槽重用、链式批处理等；

**📊 数据集**

实验采用 ImageNet 数据集评估 CNN（AlexNet、ResNet、VGG）模型，并在不同网络环境（LAN/WAN）下进行性能对比；

**📈 对比分析**

与 CrypTFlow2、Cheetah、FIT 等现有 PP‑MLaaS 系统对比，PrivQJ 在队列跳跃场景下等待成本降低 10 倍以上，普通推理的在线时间与 FIT 相当，通信量略高但可进一步优化；

**⚠️ 局限性**

局限性包括仅针对 CNN 设计；需要批量中存在闲置槽才能实现近零开销；对大模型（如 LLM）尚未验证；离线预处理与特定密文尺寸约束需要进一步研究。

---

## 292. Literary Narrative as Moral Probe : A Cross-System Framework for Evaluating AI Ethical Reasoning and Refusal Behavior

**arXiv ID:** 2603.12615 | [PDF](https://arxiv.org/pdf/2603.12615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 293. Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors

**arXiv ID:** 2603.13092 | [PDF](https://arxiv.org/pdf/2603.13092v1)

**作者:** Wei W. Xing `[一作]` (University of Sheffield), Shan Shen `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 13319 | [OpenAlex ID](https://openalex.org/A5008323996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套零超参数的Yield Multi-Corner Analysis（YMCA）框架，能够在不需手动调参的前提下，在多达25个Process‑Voltage‑Temperature (PVT) 角落上高效估算 SRAM 电路的良率。

**💡 创新点**

创新点在于将基础模型（TabPFN）的预训练知识作为“学习到的先验”，实现了自动跨角落知识迁移、无调参的贝叶斯推理以及与传统模型相比显著提升的数据效率；同时结合自动特征选择与主动学习，进一步减少 SPICE 计算量。

**🔧 技术方法**

采用的技术包括：基于 Transformer 的 TabPFN 先验模型、注意力机制实现的学习型核、全局联合 surrogate、稀疏特征选择（GBDT+贪心搜索）、基于不确定性的主动学习采样、以及在多角落统一输入表示。

**📊 数据集**

实验使用 OpenYield SRAM 基准集（4×2、8×2、16×2、32×2，分别含 144–1152 个变异参数），在 5 个 PVT 角落（TT、SS、FF、FS、SF）上进行验证。

**📈 对比分析**

与 Monte Carlo（MC）、BI‑BD、BI‑BC、OPT、MNIS、ACS、HSCS 等基准比较，平均相对误差（MRE）从 8.88%（单角落）下降至 0.11%（4×2）/1.10%（32×2），相对 MC 的加速比可达 24×，显著降低了总仿真成本并保持了高精度。

**⚠️ 局限性**

局限性包括：TabPFN 在原版训练时限制在 500 维特征；当电路维度超过此上限时仍需特征选择，可能忽略某些高阶交互；此外，该方法依赖于高质量的预训练数据，对极端新颖的电路结构或极端工艺变异可能需要进一步微调或重新预训练。

---

## 294. Can Fairness Be Prompted? Prompt-Based Debiasing Strategies in High-Stakes Recommendations

**arXiv ID:** 2603.12935 | [PDF](https://arxiv.org/pdf/2603.12935v1)

**作者:** Mihaela Rotar `[一作]` (University of Copenhagen), Maria Maistro `[通讯]` (University of Copenhagen)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5078639165)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过提示（prompt）方式对大型语言模型推荐系统（LLMRec）进行偏见消除的可行性，并在高风险推荐场景中评估其公平性和有效性。

**💡 创新点**

提出三种轻量级、可在推理阶段直接使用的提示式偏见缓解策略，并首次使用BERTScore来衡量不同敏感属性下推荐结果的语义相似度，揭示了提示策略可能导致的过度调整问题。

**🔧 技术方法**

采用了基于指令的提示（UR、BI、EBI）与基础提示对比，并使用BERTScore、Jaccard、SERP、PRAG等相似度指标进行公平性评估；实验中使用Gemma 2 9B、LLaMa 3.1 8B和Mistral 7B等LLM。

**📊 数据集**

分别使用微软新闻数据集MIND和CareerBuilder工作推荐数据集，随机抽取300名用户的历史交互和测试交互，进行新闻和职位推荐实验。

**📈 对比分析**

通过BERTScore的Precision/Recall/F1评估推荐效果，发现提示策略对效果影响不大；在公平性上，LLaMa+BI/EBI组合在年龄属性下提升了74%（SNSV），在新闻数据上BERTScore上也有约46%的公平性提升，说明提示可显著改善用户组间的推荐公平性。

**⚠️ 局限性**

限制包括：提示策略可能在某些群体上产生过度调整（例如女性导向的新闻过度推荐）；BERTScore虽能捕捉语义相似度，但未考虑不同排名位置的语义差异；实验规模受限于随机采样，仅覆盖部分LLM和提示组合；对比仅针对用户侧群体公平性，未探究物品侧公平性。

---

## 295. SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization

**arXiv ID:** 2603.12937 | [PDF](https://arxiv.org/pdf/2603.12937v1)

**作者:** Tianwei Ye `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**通讯引用:** 37310 | [OpenAlex ID](https://openalex.org/A5040010053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种语义引导的非刚性形状匹配框架SGMatch，能够在多种形状变形和拓扑噪声场景下生成更加精确且空间连贯的对应关系

**💡 创新点**

创新点包括Semantic‑Guided Local Cross‑Attention模块，可将语义上下文局部融入几何特征并保持局部连续性，以及Conditional Flow Matching正则化目标，进一步促进对应关系的平滑性和稳定性

**🔧 技术方法**

使用的技术包括基于扩散网络的几何特征、预训练的视觉基础模型（如DINO/CLIP）提供的语义特征、谱热扩散平滑、流匹配正则化以及函数式学习框架中的几何谱匹配

**📊 数据集**

在SMAL、FAUST、SCAPE、FAUST部分等数据集上进行实验，涵盖等距、非等距变形以及拓扑噪声等多种场景

**📈 对比分析**

与多种基线方法（UltraMatch、DiffusionMatch、UltraMatch、SGMatch等）对比，SGMatch在非等距变形和拓扑噪声场景下表现更佳，几何误差更低、对应关系更平滑

**⚠️ 局限性**

目前的框架尚未显式处理部分匹配场景，并且对预训练语义模型的域迁移性能依赖较大，限制了其在部分对齐和新领域的适用性

---

## 296. DS$^2$-Instruct: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning

**arXiv ID:** 2603.12932 | [PDF](https://arxiv.org/pdf/2603.12932v1)

**作者:** Ruiyao Xu `[一作]` (Northwestern University), Han Liu `[通讯]` (Northwestern University)

**通讯引用:** 14548 | [OpenAlex ID](https://openalex.org/A5100349032)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 DS2-Instruct，利用任务定义零样本生成高质量领域指令数据；

**💡 创新点**

创新点在于双向关键词扩展加检索增强获取领域知识、采用 Bloom 词汇层级实现认知多样化、以及自一致性过滤提升数据质量；

**🔧 技术方法**

使用 LLM 进行关键词与指令生成、检索增强、Bloom 词汇层级设计、自一致性筛选以及 LoRA 微调；

**📊 数据集**

在七个公开领域数据集（数学、医学、金融、逻辑推理等）上进行实验；

**📈 对比分析**

与四种基线（Zero‑shot、Self‑Instruct、InstructMix、ExploreInstruct）比较，在多模型（如 Qwen）上平均提升约 10‑20% 甚至更大；

**⚠️ 局限性**

局限性包括依赖静态知识库导致覆盖偏差、生成器 LLM 的偏差会传播到数据、仅在英文数据集上验证，未测试多语言性能。

---

## 297. Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity

**arXiv ID:** 2603.12556 | [PDF](https://arxiv.org/pdf/2603.12556v1)

**作者:** Faris Chaudhry `[一作]` (Imperial College), Faris Chaudhry `[通讯]` (Imperial College)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5119538764)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了单层物理信息神经网络（PINN）在一维非线性偏微分方程（PDE）上的经验缩放规律，量化了网络宽度与PDE非线性（硬度）对解误差的耦合影响，并提出了一种实用的缩放指数测量方法。

**💡 创新点**

创新点在于揭示了传统分离式缩放律（error ≈ A·N⁻ᵅ·κ^γ）在非线性PDE中失效的“宽度病”与“非分离病”，并通过统计回归证明宽度指数α实际上随非线性参数κ呈现负值或零，表明优化瓶颈而非逼近能力是主导因素。

**🔧 技术方法**

技术手段包括：单层网络（ReLU、Tanh激活）+ Adam优化器；Sobol序与均匀采样的碰撞点；L₂相对误差评估；多组宽度和硬度参数的系统网格搜索；单变量与多变量回归拟合缩放指数α、γ及其相互作用。

**📊 数据集**

实验数据集为四类经典一维PDE：Poisson、KdV（可分散性）、Sine‑Gordon（双曲/超越性）和Allen‑Cahn（反应/扩散），每类通过可调硬度参数κ生成多种非线性强度。

**📈 对比分析**

与理论 Barron 0.5 指数以及线性Poisson基准对比后发现：宽度指数α≈0或负值，说明宽网络不带来误差下降；非线性增大导致误差随κ上升数阶；分离式模型拟合不佳，非分离交互模型显著提升拟合优度。

**⚠️ 局限性**

局限性包括：仅使用单层网络与Adam优化；研究范围局限于一维空间、有限激活函数；未探索深层网络、Fourier特征、注意力机制或更先进优化器；未给出解决优化瓶颈的具体方案。

---

## 298. Deformation gradient averaging regularization for third medium contact

**arXiv ID:** 2603.13063 | [PDF](https://arxiv.org/pdf/2603.13063v1)

**作者:** Ondřej Faltus `[一作]` (Czech Technical University in Prague), Martin Horák `[通讯]` (Czech Academy of Sciences)

**通讯引用:** 2449 | [OpenAlex ID](https://openalex.org/A5014911652)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于单元平均变形梯度的新型第三介质接触方法，用于在有限变形下实现接触约束。

**💡 创新点**

创新点在于使用单元级变形梯度平均来抑制梯度变化，并加入线性弹性项来保持第三介质刚度，从而避免了传统二阶梯度惩罚且无需额外自由度。

**🔧 技术方法**

采用第一阶有限元方法、变形梯度平均、线性弹性正则化以及常规接触力学求解。

**📊 数据集**

使用若干已知的接触力学基准试验（如平面弹性碰撞、三维接触问题等）进行验证。

**📈 对比分析**

通过与现有基于二阶梯度惩罚的模型对比，展示了在数值稳定性、实现简易性以及计算效率上的优势，结果表明方法在所有基准中均保持良好收敛且误差不超过对标模型。

**⚠️ 局限性**

局限性包括：对正则化参数的依赖性；仅在有限变形问题中验证，可能对极大变形或非弹性材料的适用性尚未充分评估；以及未考虑多物理耦合场景。

---

## 299. IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration

**arXiv ID:** 2603.12719 | [PDF](https://arxiv.org/pdf/2603.12719v1)

**作者:** Dongxu Zhang `[一作]`, Huimin Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为IGASA的点云配准框架，结合多尺度金字塔、跳跃注意力和迭代几何感知细化；

**💡 创新点**

创新点在于通过Hierarchical Cross‑Layer Attention（HCLA）填补语义与几何之间的鸿沟，并用Iterative Geometry‑Aware Refinement（IGAR）实现软化外点抑制与精准姿态估计；

**🔧 技术方法**

技术核心包括KPConv金字塔特征提取、跨层自注意力与几何加权自注意力、以及基于权重更新的交替优化求解；

**📊 数据集**

使用了室内3DMatch/3DLoMatch和户外KITTI、nuScenes四大公开基准；

**📈 对比分析**

在各基准上均超过现有最优方法，Feature/Registration Recall、Inlier Ratio等指标显著提升，KITTI上的RTE/ RRE 仅为4.6 cm/0.24°，回收率 100%；

**⚠️ 局限性**

主要限制是迭代细化带来额外计算开销，整体推理时间比部分基准略长，对极其动态或实时应用仍需进一步加速优化。

---

## 300. Team Diversity Promotes Software Fairness: An Experiment on Fairness-Aware Requirements Prioritization

**arXiv ID:** 2603.12406 | [PDF](https://arxiv.org/pdf/2603.12406v1)

**作者:** Cleyton Magalhes `[一作]`, Italo Santos `[通讯]` (University of Hawaii)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5021360591)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本研究通过对比LGBTQ多样化团队与非多样化团队在软件需求优先级评估中的行为，探讨团队多样性对公平意识和决策的一致性影响。

**💡 创新点**

首次实证验证性取向多样性能显著提升团队对公平风险的识别与拒绝，从而增强公平意识和决策质量。

**🔧 技术方法**

采用受控实验设计，结合定量描述统计与主题分析（定性），并使用问卷与用户故事评估工具收集数据。

**📊 数据集**

使用了24条平衡分类的用户故事（公平相关、风险、中性），以及来自软件工程与计算机科学专业学生的匿名性别与性取向自我报告数据。

**📈 对比分析**

通过对每对团队进行归一化行为计数，计算 pro‑fairness、misprioritization 等指标进行比较；LGBTQ团队在 pro‑fairness 上表现更好，misprioritization 更低，显示其公平识别更准确。

**⚠️ 局限性**

局限包括样本为学生、后验收集性别/性取向导致潜在混杂、未做统计检验、样本量小、可能存在自选偏差，导致结果仅具分析性推广性。

---

## 301. PVI: Plug-in Visual Injection for Vision-Language-Action Models

**arXiv ID:** 2603.12772 | [PDF](https://arxiv.org/pdf/2603.12772v1)

**作者:** Zezhou Zhang `[一作]` (Lionrock AI Lab, China Merchants Group), Jiaxing Zhang `[通讯]` (China Merchants Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Plug‑in Visual Injection (PVI) 模块，在不修改预训练 VLA 模型结构的前提下，通过零初始化的残差注入方式，将辅助视觉表示（静态图像或视频特征）直接注入冻结的动作专家，实现轻量级、单阶段微调；

**💡 创新点**

创新点在于构建了编码器无关的注入框架，既保留了预训练模型的语义抽象能力，又通过层级残差注入显著补偿几何和时序信息缺失，且对不同视觉编码器可直接适配；

**🔧 技术方法**

技术包括：VLA 结构（VLM+流匹配动作专家）、零初始化投影与注入层、复制分支（trainable copy of DiT）以及多模态视觉编码器（如 V‑JEPA2、DINOv2）；

**📊 数据集**

使用的主要数据集包括：RoboTwin 2.0 仿真数据（10/20/50 任务多任务混合）和真实双臂平台（Airbot）收集的可变形物体折叠数据；

**📈 对比分析**

通过与基线 GR00T N1.5、Concat、ControlNet、ControlVLA、ReferenceNet 等注入策略对比，PVI 在单任务平均成功率从 35.7% 提升至 59.7%（+24pp），在多任务环境和真实机器人实验中亦保持显著提升；

**⚠️ 局限性**

局限性包括：需冻结预训练模型，微调仅针对注入路径，可能无法充分利用预训练模型的全部潜能；且在更长时序或更大动作空间时，注入效果可能随任务复杂度下降。

---

## 302. RXNRECer Enables Fine-grained Enzymatic Function Annotation through Active Learning and Protein Language Models

**arXiv ID:** 2603.12694 | [PDF](https://arxiv.org/pdf/2603.12694v1)

**作者:** Zhenkun Shi `[一作]` (Key Laboratory of Engineering Biology for Low-carbon Manufacturing, Tianjin Institute of Industrial Biotechnology, Chinese Academy of Sciences), Hongwu Ma `[通讯]` (Key Laboratory of Engineering Biology for Low-carbon Manufacturing, Tianjin Institute of Industrial Biotechnology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于蛋白语言模型、主动学习和动态集成的RXNRECer框架，直接预测酶催化的具体化学反应并提供可解释的理由。

**💡 创新点**

创新点在于完全跳过EC编号中介，采用主动学习提升数据利用率，动态融合多源预测结果，并通过通用语言模型实现反应级可解释性。

**🔧 技术方法**

使用技术包括ESM2等蛋白语言模型、GRU+Transformer预测头、主动学习采样策略、投票/堆叠/召回提升的动态集成、以及GLM提示推理。

**📊 数据集**

主要数据集为UniProtKB/Swiss‑Prot 2018版（10折交叉验证）与2018‑2024新增蛋白（时间测试）以及Rhea化学反应数据库。

**📈 对比分析**

与六个EC基准、六个无监督相似度基准以及传统MSA方法相比，RXNRECer在10折交叉验证和时间测试中的mF1分别提升约+16.5%和+12%，在多项实际案例中表现出更高覆盖率与结构一致性。

**⚠️ 局限性**

主要局限包括对已收录反应的覆盖受限、解释依赖提示工程、缺乏原子级结构/配体信息，以及对极少见或新颖反应的预测能力仍有限。

---

## 303. Navig-AI-tion: Navigation by Contextual AI and Spatial Audio

**arXiv ID:** 2603.13200 | [PDF](https://arxiv.org/pdf/2603.13200v1)

**作者:** Mathias N. Lystbæk `[一作]` (Google), Mar Gonzalez-Franco `[通讯]` (Google)

**通讯引用:** 4225 | [OpenAlex ID](https://openalex.org/A5023820234)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

该研究开发了一种全音频导航系统，利用视觉语言模型生成基于环境地标的导航指令，并在用户偏离正确方向时播放空间音频提示。

**💡 创新点**

创新点在于将VLM与实时空间音频结合，提供面向用户的环境锚点指令，并仅在偏离阈值时激活空间音频，减少音频负荷。

**🔧 技术方法**

技术包括Google Maps API获取路径、AR Core/AR Foundation进行世界追踪、Head‑Related Transfer Functions (HRTFs) 实现空间音频、Gemini 2.0 Flash VLM、文本转语音 (TTS)。

**📊 数据集**

使用的是Google Maps平台的 POI GPS 坐标及用户在三条约 550–650 米路线上记录的实际行走轨迹与离线路径的距离。

**📈 对比分析**

在 12 名参与者的交叉实验中，比较了 VLM+空间音频、仅 VLM、以及 Google Maps 音频导航，结果显示 VLM+空间音频显著减少行走距离和路线偏差，用户满意度最高。

**⚠️ 局限性**

局限性包括样本量小、路线差异导致结果偏差、VLM 推理延迟约 3.3 秒影响近距离地标提示、以及模型已停用导致未来可重复性受限。

---

## 304. RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution

**arXiv ID:** 2603.12493 | [PDF](https://arxiv.org/pdf/2603.12493v1)

**作者:** Ali Mosleh `[一作]` (AI Center-Toronto, Samsung Electronics), Michael S. Brown `[通讯]` (AI Center-Toronto, Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于校准的手机相机降解模型，用于生成真实感的RAW‑to‑RGB超分辨率训练数据，并在真实手机拍摄的RAW图像上实现高质量超分辨率；

**💡 创新点**

创新点在于：① 针对每台手机相机精确校准SR模糊核和噪声模型；② 在RAW域而非sRGB域进行“unprocessing”合成数据，显著减小域差距；③ 提供公开的校准核和噪声库，方便后续研究；

**🔧 技术方法**

主要技术包括：相机-显示器校准、灰码结构光图案匹配、视差校正、空间变换与线性化、卷积核与噪声的回归估计、异方差高斯噪声模型、可微分降采样与混色操作；

**📊 数据集**

使用了DIV2K（HR图像）、MA5K（RAW+RGB）、MA5K的RAW分辨率提升，Pixel 9 Pro、Pixel 6、Samsung S23/S24、Xiaomi Mi 11等真实手机作为测试设备；

**📈 对比分析**

与传统的bicubic、KernelGAN、MANet、Real‑ESRGAN、Degradation‑Transfer、BSRAW、RAWSR等基线在无参考MTF、PSNR、SSIM指标上对比，实验表明校准模型在所有指标上均优于基线，尤其在MTF50/25上提升显著；

**⚠️ 局限性**

局限性包括：需要专用校准硬件与耗时的采集流程；仅针对光照良好的手机场景，难以推广到低光、DSLR等复杂设备；

---

## 305. LLM Constitutional Multi-Agent Governance

**arXiv ID:** 2603.13189 | [PDF](https://arxiv.org/pdf/2603.13189v1)

**作者:** J. de Curtò `[一作]` (Barcelona Supercomputing Center), I. de Zarzà `[通讯]` (Luxembourg Institute of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 Constitutional Multi-Agent Governance (CMAG) 框架，防止 LLM 生成的影响策略导致的操纵性协作均衡。

**💡 创新点**

创新点在于两阶段决策（硬约束过滤 + 软惩罚式优化）、曝光调制与多维伦理合作度量（ECS）三重机制，系统性评估治理对协作与伦理维度的平衡。

**🔧 技术方法**

技术包括：LLM 策略编译器（Llama‑3.3‑70B‑Instruct）、硬约束过滤、软惩罚式效用优化、曝光乘数与衰减调制、以及 ECS 的乘法聚合。

**📊 数据集**

数据集为 80 个代理组成的无尺度（scale‑free）网络，在对抗性环境（70% 违规候选策略）和友好环境（15% 违规候选策略）下进行实验。

**📈 对比分析**

与无治理和仅硬约束的基线相比，CMAG 在 Ecs 上提升 14.9%（相对无治理）并保持 98% 以上自治与 99% 以上诚信；协作率虽略低（0.770 vs 0.873）但伦理性显著更优。

**⚠️ 局限性**

局限包括：仅在 80 代理无尺度网络上验证，未探讨更大规模或不同拓扑；仅使用单一 LLM 模型，未评估其他模型；OAT 敏感性分析未覆盖参数交互；真实社会网络的分布差异未考虑。

---

## 306. Graph In-Context Operator Networks for Generalizable Spatiotemporal Prediction

**arXiv ID:** 2603.12725 | [PDF](https://arxiv.org/pdf/2603.12725v1)

**作者:** Chenghan Wu `[一作]`, Liu Yang `[通讯]` (National University of Singapore)

**通讯引用:** 33817 | [OpenAlex ID](https://openalex.org/A5100356037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对比在上下文的算子学习与传统单算子学习，提出基于图的GICON模型，在实际的空气质量时空预测任务中进行系统评估。

**💡 创新点**

创新点在于引入图信息传递与示例感知位置编码，实现几何和数量泛化；同时在相同训练数据与步骤下完成首个完整的上下文算子与单算子对照实验。

**🔧 技术方法**

采用图神经网络消息传递、Transformer式上下文学习、示例感知位置编码、FAISS检索以及多算子训练策略。

**📊 数据集**

使用中国北京‑天津‑河北和长江三角洲两大区域的空气质量监测数据集（BTHSA 与 YRD），共计 13 维特征和数百个观测站点。

**📈 对比分析**

在相同训练步数与数据量的对照下，具有算子多样性的上下文算子学习在 Δt≥12h 复杂任务上 RMSE 明显低于单算子模型，并且随示例数从 0–5 训练到 100 预测时性能持续提升；单算子模型对示例无效。

**⚠️ 局限性**

局限在于依赖算子多样性提升效果；单算子上下文学习提升有限且易过拟合；模型如何有效利用示例仍不清晰，需进一步研究极端事件、其他物理系统及更高效的检索策略。

---

## 307. Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**arXiv ID:** 2603.13182 | [PDF](https://arxiv.org/pdf/2603.13182v1)

**作者:** Hiba Adil Al-kharsan `[一作]` (University of Szeged), Róbert Rajkó `[通讯]` (University of Szeged)

**通讯引用:** 1833 | [OpenAlex ID](https://openalex.org/A5084665031)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个基于非负矩阵分解（NNMF）特征提取、统计特征筛选、轻量级CNN分类与特征空间扩散去噪防御的脑瘤MRI分类框架，并在AutoAttack下验证其鲁棒性。

**💡 创新点**

创新点在于将可解释的低秩NNMF表示与轻量CNN相结合，并通过特征级扩散+去噪网络实现对抗鲁棒防御，同时采用多指标统计筛选和统一AutoAttack评估。

**🔧 技术方法**

使用技术包括NNMF（KL散度优化）、AUC/Cohen’s d/Welch t检验特征筛选、轻量级CNN、特征空间线性扩散与去噪网络、AutoAttack（APGD‑CE + Square）鲁棒性评估以及ONNX导出部署。

**📊 数据集**

实验基于Kaggle脑MRI分割数据集（约2200张MRI及对应分割掩码），按70%/20%/10%划分训练/验证/测试。

**📈 对比分析**

与基线CNN模型对比，清洁数据下准确率约0.86，防御后约0.851；在AutoAttack（ε=0.10）下，基线鲁棒准确率仅0.0047，防御后提升至0.595，且在ROC‑AUC、Brier、LogLoss等指标上保持优良表现。

**⚠️ 局限性**

局限性包括潜在的切片级数据泄漏导致性能偏高、扩散表格固定且单步去噪不具备多级净化能力，以及对GPU加速的依赖增加了计算成本。

---

## 308. FastDSAC: Unlocking the Potential of Maximum Entropy RL in High-Dimensional Humanoid Control

**arXiv ID:** 2603.12612 | [PDF](https://arxiv.org/pdf/2603.12612v1)

**作者:** Jun Xue `[一作]` (Eastern Institute of Technology), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 3201 | [OpenAlex ID](https://openalex.org/A5100695302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出 FastDSAC 框架，利用最大熵 RL 在高维人形控制任务中实现更高效的探索与学习。

**💡 创新点**

创新点在于 Dimension-wise Entropy Modulation（DEM）实现探索预算的自动重分配，以及采用连续高斯分布的分布式评论消除量化误差，二者共同提升了高维空间中的稳定性与性能。

**🔧 技术方法**

使用技术包括：最大熵 RL、连续分布式评论（Continuous Gaussian Critic）、DEM、分布式软策略迭代（DSPI）、多环境并行采样、LayerNorm、目标熵设为0、温度参数 τ 的自适应调节等。

**📊 数据集**

使用的数据集为 HumanoidBench、MuJoCo Playground 以及 IsaacLab 的多种任务，涵盖高维运动与操作场景。

**📈 对比分析**

通过与 FastTD3、FastSAC、PPO、DreamerV3 等基线在 39 个任务上对比，FastDSAC 在所有任务上均能匹配或超越现有 SOTA，尤其在 Basketball 和 Balance Hard 任务上提升了约 180% 与 400% 的最终回报。

**⚠️ 局限性**

局限性包括：对低维度或单一手臂任务时可能不如纯随机探索；需要对温度 τ 进行任务特定调参；理论分析尚不完整，且实际部署需进一步验证安全与硬件适配。

---

## 309. Keys on Doormats: Exposed API Credentials on the Web

**arXiv ID:** 2603.12498 | [PDF](https://arxiv.org/pdf/2603.12498v1)

**作者:** Nurullah Demir `[一作]` (Stanford University), Zakir Durumeric `[通讯]` (Stanford University)

**通讯引用:** 6208 | [OpenAlex ID](https://openalex.org/A5069939742)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对全球 10M 个网站进行动态爬取与分析，发现 1,748 条经过验证的 API 凭证泄露，涵盖 14 种服务供应商，揭示凭证主要通过 JavaScript 资源、打包文件及第三方嵌入暴露，且泄露凭证在网上平均存在 12 个月。

**💡 创新点**

首次在 Web 端使用动态分析方法识别凭证泄露，揭示大多数泄露源自构建与部署流程；通过系统的负责任披露，显著减少了泄露凭证（约 50% 在 14 天内被移除），并结合反馈揭示根本原因与供应链风险。

**🔧 技术方法**

使用 TruffleHog（v3.90.8）进行正则与熵检测；构建曝光定位流水线来解析 HAR 文件；利用官方 API 端点进行凭证验证；使用 Wappalyzer、Cisco Umbrella、Tranco、CrUX 等工具做技术栈、域分类、流量来源等后处理。

**📊 数据集**

HTTP Archive 2025 年度抓取数据（约 200 TB，1.19 亿主机名），结合 BigQuery 对 9.3M 域名的网络请求与响应；另外使用 Tranco 排行榜、Wappalyzer 技术检测结果、Cisco Umbrella 分类、CrUX 地理分布数据。

**📈 对比分析**

通过对比未披露与披露后 14 天内凭证移除/撤销率评估效果，发现整体凭证移除率 50%，撤销率 26%；按服务与网站排名、行业分类进行统计与显著性检验，说明不同主体的修复行为差异；与之前仅基于静态仓库的泄露研究对比，证明动态方法能捕获 95% 以上的 Web 端泄露。

**⚠️ 局限性**

只检测了 14 种服务类型；依赖 TruffleHog 正则可能漏检编码或加密凭证；负责任披露邮件回执率受限（58% 泄漏、68% 成功投递）；未能覆盖所有受影响方；仅验证公开可见凭证，隐蔽或私有凭证未计入；长期监测受限于 BigQuery 查询成本。

---

## 310. Context-Enriched Natural Language Descriptions of Vessel Trajectories

**arXiv ID:** 2603.12287 | [PDF](https://arxiv.org/pdf/2603.12287v1)

**作者:** Kostas Patroumpas `[一作]` (Archimedes Athena Research Center), Nikos Bikakis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5005916659)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种将AIS轨迹分解为旅行和情节，并用多源地理、气象和海深信息进行语义化、上下文丰富的表示，随后利用LLM生成自然语言描述的完整框架。

**💡 创新点**

创新点在于将轨迹先拆分为可解释的“旅行”与“情节”，并在每个情节中注入多源上下文，实现了既可解释又可直接供LLM推理的语义轨迹。

**🔧 技术方法**

技术包括AIS轨迹压缩与事件检测（停泊、转弯、沟通间隙等）、多源数据融合（OSM、OpenSeaMap、Weather NetCDF、GEBCO等）、以及大模型LLM（如Llama 3.3‑70B、Qwen‑3‑32B、OpenAI GPT‑OSS‑120B）生成文本。

**📊 数据集**

使用数据集包括2024年1‑3月丹麦AIS数据、OpenStreetMap/OpenSeaMap/World Port Index、Copernicus/NOAA气象NetCDF、GEBCO海深数据，共计约2.12万条AIS记录、460条旅行样本。

**📈 对比分析**

通过在同一批次数据上对5个LLM进行描述生成，平均响应时间从1.01s（Llama 8B）到2.36s（GPT‑OSS‑120B）不等；在测量准确性上，最大误差在5–10%范围内，整体性能随模型规模提升，GPT‑OSS‑120B表现最优。

**⚠️ 局限性**

局限主要体现在：①对距离/速度等数值的精确计算依赖模型内置算术能力，较小模型易出现幻觉；②未考虑船舶间交互与多船共线性；③缺乏对极端罕见事件（如严重通信丢失）更细粒度的处理。

---

## 311. How GenAI Mentor Configurations Shape Early Collaborative Dynamics: A Classroom Comparison of Individual and Shared Agents

**arXiv ID:** 2603.12600 | [PDF](https://arxiv.org/pdf/2603.12600v1)

**作者:** Siyu Zha `[一作]` (Tsinghua University), Yingqing Xu `[通讯]` (Tsinghua University)

**通讯引用:** 4468 | [OpenAlex ID](https://openalex.org/A5053438147)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了生成式人工智能（GenAI）在课堂合作中的两种访问配置（共享 AI 与个体 AI）对学生交互分布、AI‑学生耦合、社会共享调节（SSRL）以及教师编排的影响。

**💡 创新点**

将 AI 访问配置视为一个结构性设计变量，揭示共享与个体配置分别导致共识导向与分散修复两种监管模式，并首次在真实课堂环境中量化 AI 配置对协作生态的重塑。

**🔧 技术方法**

采用多层话语编码 + 滞后序列分析 (LSA)、有序网络分析 (ONA) + 一阶马尔可夫模型，对交互转移与认知网络进行统计建模。

**📊 数据集**

使用两年级八年级共38名学生在创意问题解决（Smart Garden）任务中的真实课堂视频/音频、AI 对话日志、教师观察笔记及简短访谈数据。

**📈 对比分析**

通过比较两条件下的交互比例、显著性检验、转移矩阵热图与网络差异评估；结果显示共享 AI 促进更高的学生间互动、分析深度与团队调节连贯性，个人 AI 则产生更多探索与碎片化，且教师介入频率显著提高。

**⚠️ 局限性**

局限性包括单一学校与学科、短期实验、未直接测量学生内部认知/情感过程、教师工作量未量化、样本规模有限，且结果可能受特定任务与文化背景影响。

---

## 312. Are Dependent Types in Set Theory Feasible?

**arXiv ID:** 2603.12827 | [PDF](https://arxiv.org/pdf/2603.12827v1)

**作者:** Yunsong Yang `[一作]` (École Polytechnique Fédérale de Lausanne), Viktor Kunčak `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 4449 | [OpenAlex ID](https://openalex.org/A5008699657)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在Lisa证明助手中实现了一个基于集合论的依赖函数类型（含层级宇宙）的机械化嵌入，并提供了能生成核可检查证明的双向类型检查策略；

**💡 创新点**

创新点在于将Tarski-Grothendieck宇宙与λFOL结合，实现在集合论基础上直接支持依赖类型的语法与推理，同时首次实现了带子类型的自动化证明生成器；

**🔧 技术方法**

使用了Tarski-Grothendieck集合论、λFOL（带λ项和ε算子）、Lisa证明助手的第一阶逻辑核以及双向类型检查算法；

**📊 数据集**

未使用外部数据集，所有实验均在Lisa内置的库与自定义示例上进行；

**📈 对比分析**

通过与传统依赖类型系统（如CoC）对比，证明了该方案在形式验证完整性和可复现性方面与原始系统相当，同时在证明生成速度上优于手工证明，但缺乏对递归与归纳类型的支持；

**⚠️ 局限性**

局限性包括尚未实现递归/归纳类型、缺乏对广义子类型的支持（只能实现受限的Π子类型规则）、以及对宇宙层级的手动显式管理可能导致实用性下降。

---

## 313. Semantic-Aware 6G Network Management through Knowledge-Defined Networking

**arXiv ID:** 2603.12695 | [PDF](https://arxiv.org/pdf/2603.12695v1)

**作者:** Tuğçe Bilen `[一作]` (Istanbul Technical University), Ian F. Akyildiz `[通讯]` (Truva Inc.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

设计并实现了一个基于知识定义网络（KDN）的语义通信框架，包含语义推理、语义感知路由和语义失真控制三大模块，用于管理多跳6G网络中的语义信息传输。

**💡 创新点**

创新点在于：① 将语义作为网络层控制变量，构建闭环知识平面管理；② 通过知识图谱对语义相关性进行实时推理；③ 在路由和编码层面动态调整，以显著降低语义失真和提升传输成功率。

**🔧 技术方法**

使用了Knowledge-Defined Networking架构、知识图谱推理、Transformer生成的语义嵌入、深度学习语义编码器、ns‑3仿真平台、闭环控制算法和多目标路由成本函数。

**📊 数据集**

使用预训练Transformer生成128维语义向量作为输入；构建约200–500个概念的知识图谱；仿真中采用Poisson模型的语义流（15 msg/s）与背景流，未使用公开数据集，全部为仿真合成数据。

**📈 对比分析**

通过与最短路径、负载路由和失真最小化三种基线对比，在不同负载、移动性和跳数场景下评估语义失真、交付成功率、延迟、重路频率和吞吐量。实验结果表明：语义交付成功率提升12%、平均语义失真下降22%、重路事件降低44%，吞吐量提升14%，且高相关性流的延迟波动显著降低。

**⚠️ 局限性**

局限性：① 仅在仿真环境下验证，缺乏硬件实现；② 知识图谱为静态预构建，未支持在线动态更新；③ 编码器仅提供三种分辨率，无法细粒度适配；④ 未考虑能耗、分布式知识平面或大规模部署的实时性问题；⑤ 对极端动态网络条件的鲁棒性仍有待进一步评估。

---

## 314. TASTE-Streaming: Towards Streamable Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

**arXiv ID:** 2603.12350 | [PDF](https://arxiv.org/pdf/2603.12350v1)

**作者:** Liang-Hsuan Tseng `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 9063 | [OpenAlex ID](https://openalex.org/A5040508737)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了可流式的TASTE‑S框架，实现文本对齐的语音令牌化与嵌入，支持实时低延迟语音重建。

**💡 创新点**

核心创新是将CTC‑based ASR嵌入编码器并重设计单元解码器为全因果流式，实现无需离线ASR的低延迟文本对齐；以及通过两阶段联合训练提升鲁棒性。

**🔧 技术方法**

技术包括CTC ASR、文本对齐聚合器、向量量化（FSQ）、因果解码器、流式解码模式、两阶段训练和联合训练。

**📊 数据集**

使用Emilia子集（约400h）和LibriTTS（约600h）训练，LibriSpeech test‑clean评估。

**📈 对比分析**

与传统语音令牌化器（Waveform、Encodec、SpeechTokenizer、DM‑Codec、BigCodec、WavTokenizer、Mimi、TaDiCodec）以及文本对齐方法（TASTE）比较，TASTE‑S在重建质量（WER、UTMOS、说话人相似度、时长一致性）与延迟（编码/解码RTF、FCL）上与TASTE持平或更优，且显著降低RTF和FCL。

**⚠️ 局限性**

局限性包括对ASR误差仍有一定敏感性；在极长或高频率语音场景下可能出现长度差异；模型对极端噪声或非英语语料的泛化尚未验证。

---

## 315. RNSG: A Range-Aware Graph Index for Efficient Range-Filtered Approximate Nearest Neighbor Search

**arXiv ID:** 2603.12913 | [PDF](https://arxiv.org/pdf/2603.12913v1)

**作者:** Zhiqiu Zou `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7342 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种新的范围感知图索引 RNSG，用于高效执行范围过滤近邻（RFANN）查询。

**💡 创新点**

创新点在于构造 Range‑Aware Relative Neighborhood Graph（RRNG）理论，证明其具有单调搜索性和结构继承性，并基于此设计出可近似构造的 RNSG；同时引入了基于属性与空间邻域的候选筛选、快速区间属性裁剪及区间入口节点预生成策略。

**🔧 技术方法**

采用图搜索技术（Beam Search）和近似 NNGraph（NNDescent）构建邻居，利用属性阈值筛选、RRNG 剪枝、入度限制等技术；查询时在全图上进行 on‑the‑fly 区间过滤，使用预生成的入口节点。

**📊 数据集**

在五个公开数据集上评估：YT‑Audio、YT‑RGB、WIT、SIFT1M、GIST1M。

**📈 对比分析**

与 8 个基线（iRangeGraph、UNIFY、SuperPostfiltering、SeRF、DSG、ACORN‑γ、Faiss、Milvus）对比，RNSG 在 Recall@10≥0.95 的条件下实现 2–4 倍 QPS 加速，索引规模缩减 3–23 倍，构建时间更快（至少 6 倍）。尤其在低选择率查询和高召回场景表现最优。

**⚠️ 局限性**

局限性：目前仅支持静态数据，缺乏动态增删节点的在线更新机制；近似构造不保证 RRNG 的严格单调搜索性质；对属性分布与向量分布独立性的假设在某些实际场景可能不成立。

---

## 316. Maximizing Incremental Information Entropy for Contrastive Learning

**arXiv ID:** 2603.12594 | [PDF](https://arxiv.org/pdf/2603.12594v1)

**作者:** Jiansong Zhang `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**通讯引用:** 11378 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的增量熵对比学习框架 IE-CL，利用可学习的非等距变换 SAIB 在查询分支生成熵，并通过编码器正则化保证熵在编码过程中不被压缩；

**💡 创新点**

创新点在于把信息瓶颈视为编码器的熵损失，首次实现输入熵生成与熵保持的联合优化，且通过可学习的 SAIB 取代传统手工增广，兼顾语义一致性与表达多样性；

**🔧 技术方法**

采用 InfoNCE 对比损失、KL 散度正则、SAIB 变换模块（卷积+残差结构保证正雅可比行列式）、谱正则化作为编码器熵保持手段，并构建统一的端到端优化目标；

**📊 数据集**

在 CIFAR‑10/100、STL‑10、ImageNet 进行预训练，并在 PASCAL VOC 进行迁移学习（检测与分割）验证；

**📈 对比分析**

与 SimCLR、MoCo‑v2、SimSiam、BYOL 等主流方法相比，IE‑CL 在 256 批量下显著提升线性评估精度（ImageNet Top‑1 最高 73.2% vs 71.3%），在下游任务上也实现了 mIoU、mAP 的提升；

**⚠️ 局限性**

局限性在于 SAIB 仅在卷积级别操作，可能限制对更复杂结构或高分辨率任务的建模；对 Vision Transformer 等非卷积骨干的适配尚未验证，需进一步探索跨域集成。

---

## 317. A Standards-Aligned Coordination Framework for Edge-Enhanced Collaborative Healthcare in 6G Networks

**arXiv ID:** 2603.12653 | [PDF](https://arxiv.org/pdf/2603.12653v1)

**作者:** Liuwang Kang `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Shaoshan Liu `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Collective Adaptive Intelligence Plane (CAIP)，一种面向 6G 医疗网络的标准化工作流级协调框架，利用现有 3GPP 和 O-RAN 接口实现工作流语义绑定、时限协调和隐私友好的边缘协作；

**💡 创新点**

创新点在于：①将工作流级协调抽象直接嵌入现有控制层面，避免新增协议栈；②通过可选信息元素和策略对象实现跨域时限管理与隐私保护；③提出分阶段标准化路线图，促进从实验性 xApp 到正式规范的渐进演进；

**🔧 技术方法**

核心技术包括：3GPP RRC/SDAP 设备层信令扩展、O-RAN E2SM 与 A1 接口的可选配置、近实时 RIC（xApp）协调逻辑、非实时 RIC/云端策略管理，以及 AI‑native 6G 边缘计算与协作；

**📊 数据集**

未使用具体数据集，本文为概念性设计与架构描述；

**📈 对比分析**

由于缺乏实验实施，文中未给出定量性能比较；理论上通过工作流上下文绑定与时限控制，可降低跨域协调延迟并提升整体工作流完成率；

**⚠️ 局限性**

局限性包括：①需要设备与网络侧统一支持可选信令与策略扩展；②实现成本与标准化进程受行业采纳率影响；③缺乏实测验证，未能量化性能提升；④在多域治理环境下，策略配置复杂性较高。

---

## 318. OpenDC-STEAM: Realistic Modeling and Systematic Exploration of Composable Techniques for Sustainable Datacenters

**arXiv ID:** 2603.12381 | [PDF](https://arxiv.org/pdf/2603.12381v1)

**作者:** Dante Niewenhuis `[一作]` (Vrije Universiteit Amsterdam), Tiziano de Matteis `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个可扩展、基于轨迹的仿真框架OpenDC‑STEAM，用于量化水平扩展、蓄电池、时间迁移等可持续技术对数据中心碳排放和性能的影响，并系统评估单一与组合技术的权衡；

**💡 创新点**

创新点在于：①引入了可组合、松耦合的组件图架构，使多种可持续技术能够无缝集成和组合；②扩展了OpenDC，加入了碳排放（运营与固有）建模与GPU支持；③提供了大规模多场景实验方法，真实捕捉故障、任务堆叠等动态效应；

**🔧 技术方法**

核心技术包括离散事件仿真、基于统计的功率模型（线性、平方根等）、碳强度追踪、任务调度与电池管理策略、容错机制（检查点）以及可配置的资源管理器；

**📊 数据集**

使用公开的三类工作负载（Surf LISA、CINECA Marconi M100、Google Borg），配合158个来自ElectricityMaps的碳强度轨迹（不同地区）以及多种硬件配置（CPU、GPU、内存），并采集了Facebook Messenger故障日志；

**📈 对比分析**

对比方法：在相同硬件与工作负载下，逐步开启/关闭可持续技术，记录碳排放、平均任务延迟、SLA违约率、峰值功率等指标；实验显示水平扩展可减少至35%碳排放但在故障场景下降幅至14%；电池容量与充放速率的最优平衡约为300kWh，充速0.5kW/kWh可达95%效益；时间迁移单独平均减少约2%碳排放，组合技术平均可提升至约6%或更多；

**⚠️ 局限性**

局限包括：①仅评估CPU/GPU功耗未考虑存储、网络、冷却等；②碳排放仅用CO2估算，未覆盖水耗等环境指标；③电池固有碳采用统一值（100kgCO2-eq/kWh），真实制造工艺差异未细化；④工作负载与硬件信息对Borg部分是匿名化/归一化，可能影响精度；⑤调度与电池策略为经验阈值，未尝试更高级的优化算法。

---

## 319. Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**arXiv ID:** 2603.12369 | [PDF](https://arxiv.org/pdf/2603.12369v1)

**作者:** Ayan Banerjee `[一作]` (Arizona State University), Sandeep Gupta `[通讯]` (Arizona State University)

**通讯引用:** 5101 | [OpenAlex ID](https://openalex.org/A5100601790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出域合成置信界（DCB）理论，用以评估不同医学影像域之间因果因素的差异，并构建 GenEval——一种将精细化人类专家知识与 MedGemma-4B 视觉语言模型融合、通过 LoRA 进行参数高效微调的多模态框架，旨在提升单源域泛化（SDG）在糖尿病视网膜病变（DR）分级和脑功能fMRI癫痫发作起始区（SOZ）检测中的表现。

**💡 创新点**

创新点包括：① 引入 DCB 与 SDCD（源域符合度）指标，提供分布无关的因果差异量化方法；② 通过 SDCD 引导的知识消融与优化，实现对专家知识的量化提炼；③ 在 MedGemma-4B 中嵌入已优化的知识提示，并利用 LoRA 进行低秩高效微调，既保留了预训练的医学知识，又避免了灾难性遗忘；④ 在多达八个 DR 数据集与两个 SOZ 数据集上实现了显著的单源域泛化性能提升。

**🔧 技术方法**

主要技术包括：域合成置信推理（基于 Mahalanobis 距离的 conformal inference）、LoRA 参数高效微调、MedGemma-4B 视觉语言模型、YOLOv12 眼底病变检测器、命题逻辑表达专家知识、SDCD 指标用于知识优化与因果差距评估。

**📊 数据集**

使用的数据集为：糖尿病视网膜病变（APTOS、EyePACS、Messidor‑1、Messidor‑2、IDRiD、DeepDR、FGADR、RLDL）共八个；癫痫发作起始区（SOZ）为两套 rs‑fMRI 数据集（UNC、PCH）共两套。

**📈 对比分析**

在与现有最强 DG 方法（SPSD‑ViT、SD‑ViT、ERM‑ViT、DECO、GDRNet、CLIP‑DR 等）以及未微调的 VLM（CLIP、OrdinalCLIP 等）进行对比实验。GenEval 在单源域泛化任务中平均准确率分别为 69.2%（DR）和 81%（SOZ），相较最佳基线提升 9.4% 与 1.8%，并在多源域和跨站点 SOZ 评估中持续保持优势。

**⚠️ 局限性**

局限性在于：假设数据生成过程连续可微，若存在明显的非可微转折或阈值效应，DCB 与 SDCD 的估计可能失效；知识提取依赖于 YOLO 检测的精度，若检测失真，知识量化与优化可能受损；此外，在高噪声环境下 SDCD 的稳健性会下降，进而影响泛化性能。

---

## 320. Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**arXiv ID:** 2603.13056 | [PDF](https://arxiv.org/pdf/2603.13056v1)

**作者:** Elena Ryumina `[一作]` (St. Petersburg Federal Research Center of the Russian Academy of Sciences), Alexey Karpov `[通讯]` (St. Petersburg Federal Research Center of the Russian Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种三模态（面部、行为VLM嵌入、音频）连续情绪估计框架，并设计两种融合策略。

**💡 创新点**

引入多模态VLM行为描述嵌入并结合跨模态门控与交互融合，提高ITW条件下的连续情绪预测。

**🔧 技术方法**

GRADA+Transformer、Qwen3-VL-4B-Instruct+Mamba、WavLM-Large+注意力统计池化、DCMMOE与RAAV融合。

**📊 数据集**

Aff-Wild2数据集，采用10th ABAW挑战官方拆分。

**📈 对比分析**

与单模态基线对比，融合模型在验证集上取得最高CCC 0.6576（RAAV），测试集CCC 0.62，明显优于单模态。

**⚠️ 局限性**

对音频质量过滤不完全可靠，模型对不同说话人和噪声的鲁棒性待提升；VLM预训练依赖大规模数据，易受偏差影响。

---

## 321. Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking

**arXiv ID:** 2603.12831 | [PDF](https://arxiv.org/pdf/2603.12831v1)

**作者:** Zizhao Mo `[一作]` (University of Macau), Chengzhong Xu `[通讯]` (University of Macau)

**通讯引用:** 17665 | [OpenAlex ID](https://openalex.org/A5012773300)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在共享 GPU 集群中设计并实现一种名为 OmniServe 的 LLM 推理系统，支持在保持 LS 服务 SLO 的前提下提升 BE 服务吞吐量。

**💡 创新点**

提出 Attention Piggybacking 机制将 BE 的 Attention 计算异步卸载到 CPU，并结合层级异步批量控制，实现 CPU 与 GPU 的高效协同。

**🔧 技术方法**

采用 CPU‑GPU 异步计算、AVX‑512 向量化、连续批量、KV 缓存交换、动态调度与模型延迟估计等技术。

**📊 数据集**

使用 Llama‑2‑70B、Yi‑34B 两大模型，并在 LongBench‑v2、DailyMails 等长文本生成基准上进行实验。

**📈 对比分析**

与 Llumnix、NEO、Sarathi‑Serve 等基线比较，OmniServe 在 LS SLO 达成率可提升至 91.6%~95%（比 Llumnix 高 1.48×），BE 预填/解码吞吐量提升 3.1×~9.9×。

**⚠️ 局限性**

依赖于准确的延迟模型和充足的 CPU 资源，且在高并发或多模型部署时可能产生额外通信与残差管理开销。

---

## 322. OFDM Waveform for Monostatic ISAC in 6G: Vision, Approach, and Research Directions

**arXiv ID:** 2603.12641 | [PDF](https://arxiv.org/pdf/2603.12641v1)

**作者:** Huacheng Zeng `[一作]` (Michigan State University), Ruxin Lin `[通讯]` (Michigan State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了在6G网络中使用OFDM波形实现单点ISAC的可行性与方法，并搭建毫米波与sub‑6 GHz原型进行实验验证。

**💡 创新点**

创新点包括：①提出四种自干扰抑制/全双工/自混合RF/精准Tx/Rx切换技术；②首次在标准OFDM环境下实现四维（距离、速度、方位、俯仰）感知；③结合深度学习对感知数据进行端到端推理。

**🔧 技术方法**

使用的技术包括：OFDM通信波形、MIMO天线阵列、全双工/自混合RF下变频、精确Tx/Rx RF切换、数字后处理与CNN/多模态DNN。

**📊 数据集**

主要数据集为实验室原型收集的CIR与射频点云，结合Depth摄像头标注的人体姿态数据集，用于训练与评估DNN。

**📈 对比分析**

通过与传统Wi‑Fi/毫米波基准的对比，单点ISAC在10 m检测范围内实现约1 cm距离分辨率、<0.1 m的姿态误差，且可实现10帧/秒的4D感知。

**⚠️ 局限性**

局限性在于：自干扰抑制仍受限于硬件实现，毫米波需要大尺寸天线阵列；sub‑6 GHz受PAPR和ADC动态范围约束；实时推理对算力需求高。

---

## 323. Pairwise Exchanges of Freely Replicable Goods with Negative Externalities

**arXiv ID:** 2603.12403 | [PDF](https://arxiv.org/pdf/2603.12403v1)

**作者:** Shangyuan Yang `[一作]`, Kirthevasan Kandasamy `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出一种基于多轮配对交换的机制，允许竞争性代理在共享可无限复制的数字商品（如数据集、专利）时，既能提高各自效用，又能保证协议终止时无可行互惠交换。

**💡 创新点**

创新点在于：①首次将负外部性（竞争）纳入可复制商品交换模型；②设计了一种兼具 Nash 倾向兼容（NIC）、个体理性（IR）与稳定性的交换协议；③通过三大不可能性结果阐明了 DSIC、真诚报告及单轮协议的局限性。

**🔧 技术方法**

核心技术包括：按竞争系数递增顺序选取代理对的“竞争顺序惰性交换”，递归地回溯（Retrospection）调整已安排的交换以最大化可行交换数量；构造图论模型分析参与与非参与对效用的冲击，证明 IR；通过递归子程序和路径分析证明 NIC 与稳定性。

**📊 数据集**

论文主要是理论工作，未使用真实数据集；所有示例与证明均基于抽象的竞争参数 β 与初始分配向量。

**📈 对比分析**

评估以理论性质为主：证明协议在接受策略下单轮即可完成、满足 NIC、IR 与稳定性；并通过仿真（在附录中）展示在低竞争环境下可能实现 Pareto‑efficient；在高竞争环境下，协议仍保持稳定但可能降低整体社会福利。

**⚠️ 局限性**

局限性：①无法实现 DSIC 与全真报告，需接受弱兼容；②在高竞争（HEC）情形下，协议不保证 Pareto‑efficiency；③模型假设所有代理的商品价值相同、不可分配，未来工作需考虑多样化价值、可分配商品与进一步的外部性。

---

## 324. Purify Once, Edit Freely: Breaking Image Protections under Model Mismatch

**arXiv ID:** 2603.13028 | [PDF](https://arxiv.org/pdf/2603.13028v1)

**作者:** Qichen Zhao `[一作]` (Peking University), Zhonghai Wu `[通讯]` (Peking University)

**通讯引用:** 2081 | [OpenAlex ID](https://openalex.org/A5041563693)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在模型失配环境下评估并对抗基于扰动的图像保护，提出两种后期纯化方法VAE-Trans与EditorClean。

**💡 创新点**

首次引入统一的后期纯化框架，展示保护失效的“purify-once, edit-freely”失败模式，并提出跨模型对抗的纯化技术。

**🔧 技术方法**

使用VAE编码器投影、Diffusion Transformer（FLUX）进行指令引导重建，以及传统JPEG、IMPRESS、GridPure等基线。

**📊 数据集**

DiffusionGuard 2100条文本引导修补任务（人像、物体、动物），以及OmniEdit-Filtered-1.2M用于训练纯化器。

**📈 对比分析**

与六种主流扰动保护（PhotoGuard、AdvDM、MIST、SDS、DiffusionGuard、AdvPaint）在SD v1.5/v2.0以及跨模型/预处理设置下对比；EditorClean在PSNR、LPIPS、FID、IR上提升3–6 dB、50–70%以上，显著优于基线。

**⚠️ 局限性**

仅针对公开模型，未评估对付专有编辑服务；纯化后仍可能出现视觉失真；未考虑与其他防护层（如版权标记）联合使用的效果。

---

## 325. Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages

**arXiv ID:** 2603.12554 | [PDF](https://arxiv.org/pdf/2603.12554v1)

**作者:** Vishnu Teja Kunde `[一作]` (Texas A&M University), Jean-Francois Chamberland `[通讯]` (Texas A&M University)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5069801730)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对扩散式大语言模型进行强化学习微调，提出精确的无偏策略梯度与按步优势分配。

**💡 创新点**

创新点包括：① 将扩散生成视作有限时隧道马尔可夫决策过程；② 推导出可按步分解的无偏策略梯度；③ 用熵引导的步骤选择与一阶去噪估计的优势来显著降低计算成本并提升学习信号；④ 将上述理论与GRPO损失结合，形成EGSPO与EGSPO-SA两种实用算法。

**🔧 技术方法**

使用策略梯度、GRPO、熵引导步骤选择、单步去噪优势估计、Masked Discrete Diffusion模型（LLaDA-8B-Instruct）及其推理框架。

**📊 数据集**

评测数据集包括：编码任务MBPP与HumanEval；逻辑推理Sudoku与Countdown；数学推理GSM8K与MATH500；以及用于预训练的通用文本数据。

**📈 对比分析**

与d1、wd1、SPG、d2及基线LLaDA-8B-Instruct等方法对比，EGSPO-SA在所有基准上均实现或逼近最优性能，尤其在逻辑推理上显著优于前沿RL方法；在Reward‑vs‑FLOPs、Reward‑vs‑Samples和Reward‑vs‑GradientSteps三条曲线上均优于d1，显示更高的计算与数据效率。

**⚠️ 局限性**

局限性：单步去噪优势估计在早期步骤存在偏差；熵引导步骤选择虽有效但仍依赖于熵估计的准确性；对极长扩散过程或高度稀疏奖励的任务效果尚未充分验证；与某些自回归RL方法相比，在极端大模型规模下的性能差距仍可能存在。

---

## 326. TaoBench: Do Automated Theorem Prover LLMs Generalize Beyond MathLib?

**arXiv ID:** 2603.12744 | [PDF](https://arxiv.org/pdf/2603.12744v1)

**作者:** Alexander K Taylor `[一作]` (University of California Los Angeles), Wei Wang `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建基于 Tao 的《Analysis I》Lean 版的自定义定义框架 benchmark，评估当前 LLM ATP 模型在非 MathLib 定义环境下的泛化能力。

**💡 创新点**

创新点在于：①首次提出对 MathLib 以外定义框架的 ATP 泛化评估 benchmark；②通过 agentic pipeline 自动提取可编译的局部环境，并生成数学等价的 MathLib 版本；③揭示了模型在面对自定义定义时的显著性能衰减。

**🔧 技术方法**

技术方法包括：agentic 架构（文件检索、Lean 编译器工具）、JiXia 静态分析、自动化翻译管道（重写 + 等价性检验）以及 LLM（GPT‑5.1）与 web‑search 结合的自动化推理与编译。

**📊 数据集**

数据集为 150 道来自 Tao Formalization of Analysis I 的练习题，分别以原始自定义定义（TaoBench）和等价 MathLib 定义（MathLibBench）两种版本构成。

**📈 对比分析**

通过在四个公开 ATP 模型（DeepSeek‑Prover‑v2‑7B、Goedel‑Prover‑v2‑8B/32B、Kimina‑Prover‑8B）上计算 pass@128，发现对 MathLib 版本平均准确率约 70% 以上，而对 TaoBench 平均仅 40% 左右，显示出约 26% 的性能下降。

**⚠️ 局限性**

限制在于：模型在处理自定义定义时高度依赖训练分布；agentic pipeline 对上下文检索和编译错误的自动修复仍不完美；实验仅覆盖 150 题，难以全面评估更大规模或更复杂的定义差异。

---

## 327. A Spectral Revisit of the Distributional Bellman Operator under the Cramér Metric

**arXiv ID:** 2603.12576 | [PDF](https://arxiv.org/pdf/2603.12576v1)

**作者:** Keru Wang `[一作]` (University College Dublin), Shengbo Eben Li `[通讯]` (Tsinghua University)

**通讯引用:** 19692 | [OpenAlex ID](https://openalex.org/A5100747108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文从Cumulative Distribution Function（CDF）层面出发，首次将分布式Bellman运算的结构用正则化的谱Hilbert空间实现，从而在不改变Bellman动态的前提下获得可线性操作的解析框架。

**💡 创新点**

创新点在于：①将Cramér度量的低频奇异性通过正则化参数ϵ化简为可构造Hilbert空间；②建立CDF与谱Hilbert空间的完备同构映射，使Bellman运算在谱空间中得到完全保留的线性表示；③在保持原始分布动态不变的前提下，构造了一个“Hilbert包络”来研究分布式Bellman运算的谱性质。

**🔧 技术方法**

主要技术包括：CDF层面的概率分析、傅里叶变换与特征函数表述、对Cramér度量的频域权重进行正则化、构造映射U、V、𝕍实现同构、以及利用Banach固定点定理证明收敛性。

**📊 数据集**

本文为理论性工作，并未使用任何具体数据集；讨论全部在抽象概率模型和数学证明层面完成。

**📈 对比分析**

由于是纯理论分析，未与任何数值算法或基准进行比较，亦未给出实验性能指标；作者仅给出收敛性与存在性的定理证明。

**⚠️ 局限性**

局限性包括：仅讨论固定策略下的策略评估；未涉及最优性操作符、函数逼近或实际算法实现；缺乏实验验证与在真实环境中的性能评估。

---

## 328. ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models

**arXiv ID:** 2603.13033 | [PDF](https://arxiv.org/pdf/2603.13033v1)

**作者:** Yanpeng Zhao `[一作]` (State Key Laboratory of General Artificial Intelligence), Zilong Zheng `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向嵌入式空间推理的可扩展、可复现的仿真评测基准，用于诊断视觉‑语言模型（VLM）在三维空间定位与执行任务中的表现。

**💡 创新点**

创新点包括：
1) 将传统 VQA 评测转化为生成式定位与执行任务，桥接被动推理与主动执行；
2) 以三大因素（空间维度、参考框架、参考对象）系统化设计 148 种空间推理任务；
3) 在 Isaac‑Sim 中构建多样化、物理可行的桌面与货架环境，支持大规模任务生成；
4) 通过“反射”机制和多视角输入探究模型对失败经验的利用。

**🔧 技术方法**

技术主要包括：
- 基于功能程序的指令表示与多跳推理；
- 通过 2D 指针任务实现 3D 定位；
- 在执行阶段使用 6‑DoF 目标姿态预测与物理规划器（cuRobo）实现可执行动作；
- 采用随机化光照、相机与纹理减少 sim‑to‑real 问题。

**📊 数据集**

数据集：
- 采用 Isaac‑Sim 生成的桌面与货架场景；
- 通过功能程序生成 2,220 条 pick/ place 任务，覆盖 148 种空间推理类型；
- 任务实例化为 3‑4 个自然语言模板，确保语言多样性。

**📈 对比分析**

比较方法：
- 评估 15+ VLM（Gemini‑2.5‑Pro、Qwen3‑VL、InternVL3‑78B、RoboBrain2.0‑7B 等）在定位与执行上的准确率与接受率；
- 在三种难度（易/中/难）与“反射”与否的设置下进行基准实验；
- 结果显示：
  * Gemini‑2.5‑Pro 仍是整体表现最优；
  * 公共 VLM（Qwen3‑VL 系列）在执行上已逼近专用模型；
  * 位置任务普遍优于放置任务；
  * 角度推理（尤其是姿态）是关键瓶颈。

**⚠️ 局限性**

局限性：
- 仅涵盖室内桌面/货架场景，缺乏户外、长距离或全局参考框架的推理；
- 任务单步，未覆盖多步/长时序空间推理；
- 依赖仿真环境，尽管已做随机化，但仍有 sim‑to‑real 差距；
- 未结合低层控制器，侧重诊断而非完整机器人系统。

---

## 329. A Geometrically-Grounded Drive for MDL-Based Optimization in Deep Learning

**arXiv ID:** 2603.12304 | [PDF](https://arxiv.org/pdf/2603.12304v1)

**作者:** Ming Lei `[一作]` (Shanghai JiaoTong University), Christophe Baehr `[通讯]` (Meteo-France CNRS CNRM GAME UMR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将最小描述长度(MDL)原则嵌入深度学习优化过程的几何驱动框架，利用MDL驱动项与Ricci流耦合，使模型在训练期间主动压缩内部表示。

**💡 创新点**

创新点在于将MDL从传统的后置模型选择变为优化过程的主动驱动力，并引入几何化的认知流形与自适应MDL驱动，解决了传统Ricci流在AI中的拓扑奇点与任务无关问题；同时证明了描述长度单调下降、有限次数拓扑外科手术、临界行为与收敛速率等一系列理论性质。

**🔧 技术方法**

核心技术包括：几何深度学习、Ricci流与MDL驱动耦合、哈特森估计法求解描述长度梯度、自然梯度更新、显式欧拉时间步进、拓扑外科手术协议、CFL稳定性分析以及强凸性下的指数收敛证明。

**📊 数据集**

在实验中使用合成多项式回归数据（N=100，含噪声）以及后续的分类任务（未详细列出数据集），通过这些数据验证了理论预测。

**📈 对比分析**

与传统基于任务损失的优化方法对比，MDL驱动算法在保持或提升任务误差性能的同时显著降低了描述长度，实现了更紧凑、泛化更好的模型；实验报告显示任务误差下降与描述长度单调递减，证明了理论与实践的一致性。

**⚠️ 局限性**

局限性包括：算法在大规模网络和真实世界数据集上的可扩展性尚待验证；对Ricci流的数值实现与外科手术阈值的选择可能对收敛产生影响；以及在高度非凸的真实损失景观中，强凸性假设不一定满足，导致收敛速率的理论保证有限。

---

## 330. Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse

**arXiv ID:** 2603.13038 | [PDF](https://arxiv.org/pdf/2603.13038v1)

**作者:** Hubert Plisiecki `[一作]` (IDEAS Research Institute), Marcin Zajenkowski `[通讯]` (University of Warsaw)

**通讯引用:** 3002 | [OpenAlex ID](https://openalex.org/A5071964816)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种PCA Sweep方法，用于在监督语义差分（SSD）分析中系统地选择主成分数K，以降低研究者自由度并提升解释性

**💡 创新点**

将梯度解释性、表示质量和梯度稳定性三大指标结合为一体化评估，采用局部平均平滑并以最小K的最大联合得分来决定维度

**🔧 技术方法**

监督语义差分（SSD）、主成分分析（PCA）、线性回归、词嵌入（Dolma GloVe 300维+SIF），聚类（Silhouette）与词向量相似度检索

**📊 数据集**

来自Prolific的349条关于人工智能的简短帖子，配合Admiration（ADM）和Rivalry（RIV）自恋量表得分

**📈 对比分析**

与随意高维度（K=120）对比，PCA Sweep得到的K=15产生稳定、可解释的梯度，聚类清晰；高维度则导致模糊、弱结构化的聚类，说明Sweep能有效提升解释性

**⚠️ 局限性**

样本规模有限，且仅使用整体文本而非概念特定词汇；PCA Sweep仅解决维度选择问题，仍未系统化嵌入模型或窗口大小等其他参数的选择，结果对不同数据集的稳健性尚待验证

---

## 331. GNN-DIP: Neural Corridor Selection for Decomposition-Based Motion Planning

**arXiv ID:** 2603.12361 | [PDF](https://arxiv.org/pdf/2603.12361v1)

**作者:** Peng Xie `[一作]` (Technical University of Munich), Amr Alanwar `[通讯]` (Technical University of Munich)

**通讯引用:** 542 | [OpenAlex ID](https://openalex.org/A5102709386)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出GNN-DIP框架，利用图神经网络对分割空间的门（portal）进行评分，以指导分解信息规划器在窄通道中快速寻找最优路径。

**💡 创新点**

创新点在于将GNN门评分嵌入到基于凸单元分解的两相规划流程中，既保证了完整性与收敛性，又显著降低了候选走廊数量，实现了从宽阔单元到窄通道的高效搜索。

**🔧 技术方法**

使用Constrained Delaunay Triangulation/Slab分解、Funnel算法/门面采样、Yen最短路搜索、以及基于PyTorch的图卷积网络（2D）或GAT（3D）进行门评分。

**📊 数据集**

在310幅2D多多边形地图、4个3D窄通道场景（共约400–2500个单元）、以及100个动态2D实例上进行实验。

**📈 对比分析**

与OMPL的iRRT*、BIT*、AIT*、EIT*等采样方法对比，GNN-DIP在2D中实现99–100%成功率、2–280倍速度提升；在3D窄通道中同样实现100%成功率，速度比BIT*快3–20倍，且在极大单元数（≈600）场景下保持与无引导DIP相近的路径质量。

**⚠️ 局限性**

主要局限包括：对复杂3D非轴对齐障碍的分解尚未支持、GNN推理仍需10–50 ms（可通过ONNX改进），以及对路径后处理质量（如梯度优化）和机器人半径约束的进一步完善。

---

## 332. Unleashing Video Language Models for Fine-grained HRCT Report Generation

**arXiv ID:** 2603.12469 | [PDF](https://arxiv.org/pdf/2603.12469v1)

**作者:** Yingying Fang `[一作]` (Imperial College London), Guang Yang `[通讯]` (King's College London)

**通讯引用:** 17885 | [OpenAlex ID](https://openalex.org/A5108053324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了 AbSteering 框架，将通用视频语言模型适配至高分辨率 CT（HRCT）报告生成任务。

**💡 创新点**

创新点在于引入异常中心的 Chain-of-Thought（CoT）训练和基于 Direct Preference Optimization（DPO）的细粒度异常区分，显著提升了模型对细微、稀有异常的辨别能力并抑制了幻觉。

**🔧 技术方法**

采用了通用视频语言模型（如 Qwen2.5‑VL、InternVL3）与 3D 视觉编码器，配合 LLM 解码器；训练分为两阶段：CoT 结构化推理 + DPO 细粒度偏好优化。

**📊 数据集**

使用了 CT‑RATE‑AB 数据集，包含 50,188 例胸腔 HRCT 与对应手工报告，已按 46,717/3,039 的训练/验证划分。

**📈 对比分析**

与多种基于 CT 的基础模型（RadFM、M3D、CT‑CHAT 等）及传统报告生成器对比，AbSteered 的视频模型在 BLEU、ROUGE‑L、BERTScore 以及 18 项异常的精确度、召回率和 F1 上均超过领域专用模型；在召回率上尤为突出且幻觉率保持低位。

**⚠️ 局限性**

局限性包括对视觉‑文本对齐的高度依赖，在更大 LLM 规模（如 32B）下性能反而下降；对极稀有异常的识别仍有限，且训练仍需较多数据与计算资源。

---

## 333. Internet-Scale Measurement of React2Shell Exploitation Using an Active Network Telescope

**arXiv ID:** 2603.12300 | [PDF](https://arxiv.org/pdf/2603.12300v1)

**作者:** Aakash Singh `[一作]` (CSIR Fourth Paradigm Institute), V. Anil Kumar `[通讯]` (CSIR Fourth Paradigm Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过主动网络望远镜收集并分析 2025 年 12 月的互联网流量，首次对 React2Shell（CVE-2025-55182）攻击的扫描行为和后端基础设施进行量化研究。

**💡 创新点**

创新点在于：①引入主动网络望远镜主动响应 SYN 并捕获完整 TCP 负载；②设计基于协议的确定性检测签名，准确识别 RSC 漏洞利用；③首次量化全球扫描源与后端服务器 IP 的分布与聚集情况。

**🔧 技术方法**

使用技术包括主动网络望远镜（响应 SYN/ACK）、TCP 流重组、UTF‑8/Latin‑1/URL/BASE64/压缩解码、基于协议的检测签名、IP2Location 地理归属、SQLAlchemy+Pandas 数据查询。

**📊 数据集**

数据集为两个主动望远镜（/23 与 /25）在 2025 年 12 月捕获的 PCAP 数据，约 79 条属性、约 70 万 TCP 连接，包含完整的应用层负载。

**📈 对比分析**

通过与第二个独立望远镜的时间序列进行 Pearson 相关性检验（r=0.87）验证检测方法；在峰值 88,197 连接/天的扫描中实现高召回率且无误报，展示了方法在大规模流量下的稳健性。

**⚠️ 局限性**

局限性包括：只覆盖两个望远镜，无法观察目标服务器端完整行为；只能捕获初始负载，无法完整追踪攻击链；IP 空间有限导致扫描样本不完整；后端 IP 的解码依赖有限，可能漏掉使用域名或加密载荷的攻击。

---

## 334. STRAP-ViT: Segregated Tokens with Randomized -- Transformations for Defense against Adversarial Patches in ViTs

**arXiv ID:** 2603.12688 | [PDF](https://arxiv.org/pdf/2603.12688v1)

**作者:** Nandish Chattopadhyay `[一作]` (Indian Institute of Technology), Anupam Chattopadhyay `[通讯]` (Nanyang Technological University)

**通讯引用:** 6248 | [OpenAlex ID](https://openalex.org/A5089860351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 STRAP-ViT，一种在 ViT 推理过程中无需额外训练即可检测并随机变换少量异常 token 的防御机制，用以抵御对抗性补丁。

**💡 创新点**

创新点在于：① 通过 Jensen‑Shannon Divergence 在 token 级别进行异常分离；② 对被识别为异常的 token 采用随机组合的 L_p 投影、仿射收缩与 softmax 温度变换，以破坏补丁的干扰；③ 所有操作均为无训练、插件式，兼容任意 ViT 架构。

**🔧 技术方法**

主要技术包括：token‑level JSD 检测、L_p 投影、仿射收缩、softmax 温度变换、随机组合变换，以及 ViT 的 PatchEmbed、PosEnc 与 Encoder 前向推理。

**📊 数据集**

使用 ImageNet 与 CalTech‑101 两大公开图像数据集，并在 ViT‑base‑16 与 DinoV2 预训练模型上进行评估。

**📈 对比分析**

与无防御、PatchCleanser、ODDR、LGS 等方法对比，STRAP‑ViT 在 Adversarial Patch、LAVAN、GDPA 等攻击下，Top‑1/Top‑5 鲁棒准确率提升 70%–80%，且在干净样本上的准确率仅下降不到 2%。

**⚠️ 局限性**

局限性包括：对检测阈值与 K（需变换 token 数量）的敏感性；多重补丁时需要增大 K 以覆盖所有 patch；对自适应攻击的鲁棒性尚未完全验证；以及对更大分辨率或不同 ViT 变体的通用性需进一步实验。

---

## 335. RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection

**arXiv ID:** 2603.12582 | [PDF](https://arxiv.org/pdf/2603.12582v1)

**作者:** He Zhu `[一作]` (Institute of Information Engineering), Haitian Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种黑盒文本对抗检测框架RTD-Guard，利用预训练的Replaced Token Detection（RTD）判别器定位可疑词汇，并通过掩码干预观察模型置信度变化来识别对抗样本。

**💡 创新点**

将RTD预训练任务复用为对抗检测器，无需对抗样本或梯度信息，仅用两次模型查询即可完成检测，兼具轻量化与可插拔性。

**🔧 技术方法**

使用RTD判别器生成词级替换概率、k‑top掩码干预、置信度差分作为检测信号，并在黑盒环境下仅进行两次API调用。

**📊 数据集**

在IMDB、AG‑News和Yelp三大文本分类数据集上，并结合TextAttack生成的TextFooler、PWWS、BAE、TF‑adj四种词级攻击进行评估。

**📈 对比分析**

与PPL、MLE、FGWS、RDE、WDR、GradMask等基线对比，RTD‑Guard在TPR10、F1、AUC三项指标上均取得最高或相近分数，且查询开销仅两次，运行时最快。

**⚠️ 局限性**

依赖于单语RTD模型，对跨语言或多语言输入的适用性有限，且对生成式或指令型LLM的对抗检测尚未得到验证。

---

## 336. Multi-objective Genetic Programming with Multi-view Multi-level Feature for Enhanced Protein Secondary Structure Prediction

**arXiv ID:** 2603.12293 | [PDF](https://arxiv.org/pdf/2603.12293v1)

**作者:** Yining Qian `[一作]` (Northeastern University), Xianpeng Wang `[通讯]` (Northeastern University)

**通讯引用:** 2344 | [OpenAlex ID](https://openalex.org/A5080165465)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多目标遗传程序框架MOGP-MMF，自动进行特征选择与融合，以预测蛋白质二级结构。

**💡 创新点**

创新点包括：多视角多层特征表示、富集的线性与非线性操作集、知识迁移辅助的多目标优化，以及生成多样化非支配解的能力。

**🔧 技术方法**

技术手段涵盖：深度学习特征提取（PSSM/HMM、ProtTrans‑T5、SaProt）、遗传程序（多目标、知识迁移）、自定义操作集合、t‑SNE可视化。

**📊 数据集**

使用的数据集为 CB6133、CB513 以及 CASP10–14 共七个基准集。

**📈 对比分析**

与九种基线（NetSurfP‑3.0、AttSec、DeepPredict、TruMPET 等）在 Q8、Precision、Recall、F1、MCC、Sov 等六项指标上进行对比，MOGP‑MMF 在所有数据集上均实现最高 Q8、最佳 Sov 与 MCC，整体性能显著超越 SOTA。

**⚠️ 局限性**

局限性在于：训练阶段遗传程序搜索计算量大，且性能仍受限于预训练特征视角的质量，可通过引入更强的蛋白语言模型和结构嵌入进一步提升。

---

## 337. Surrogates for Physics-based and Data-driven Modelling of Parametric Systems: Review and New Perspectives

**arXiv ID:** 2603.12870 | [PDF](https://arxiv.org/pdf/2603.12870v1)

**作者:** Matteo Giacomini `[一作]` (Universitat Politècnica de Catalunya), Pedro Díez `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 2182 | [OpenAlex ID](https://openalex.org/A5029122399)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述并提出新的视角，系统评估了物理基础、数据驱动与混合的代理模型构建方法，聚焦于功能逼近、降维、超低保真、多保真与自适应采样等关键技术；

**💡 创新点**

提出统一的功能逼近框架，将模型设计拆分为基函数构造与逼近准则两大核心；在此框架下综合评估POD、PGD、神经网络及其多保真和自适应策略，强调多保真与自适应采样对提升模型效率与鲁棒性的作用；

**🔧 技术方法**

采用主成分分析/奇异值分解（POD）、通用分解（PGD）、稀疏表示、回归与核方法（RBF、Kriging）、最小二乘、移动最小二乘、全连接与卷积神经网络（Autoencoder、DeepONet、FNO等）以及图神经网络等多种降维与逼近技术；

**📊 数据集**

本研究为综述，未使用特定实验或数值数据集，而是梳理并引用已有文献中的案例与实验结果；

**📈 对比分析**

比较主要以方法论与性能维度（如训练成本、在线效率、误差估计、可解释性、可扩展性）进行定性评估；未给出统一数值对比，更多是概念性对照与案例展示；

**⚠️ 局限性**

局限在于缺乏统一量化评测与跨方法的客观性能比较，且对验证、校准与不确定性量化的讨论仍不充分；对非线性、高维问题的实际应用仍需进一步实验验证。

---

## 338. DiscoRD: An Experimental Methodology for Quickly Discovering the Reliable Read Disturbance Threshold of Real DRAM Chips

**arXiv ID:** 2603.12435 | [PDF](https://arxiv.org/pdf/2603.12435v1)

**作者:** Ataberk Olgun `[一作]` (ETH Zurich), Onur Mutlu `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

在真实DDR4芯片上构建并验证了一种快速、可靠的读扰动阈值（RDT）测试方法DiscoRD，系统性地测量并分析了RDT的时空变化。

**💡 创新点**

首次提出基于实验的快速RDT检测框架，并利用大规模测量揭示RDT的可变性，构建经验模型评估错误容忍与空间/时间变异相结合的读扰动缓解策略。

**🔧 技术方法**

采用双边RowHammer实验、温度与激活时间控制、行映射逆向、统计/蒙特卡洛模拟、错误校正码（ECC）和内存消毒等技术。

**📊 数据集**

对数十个DDR4模块（SK Hynix、Micron、Samsung）进行1,000次RDT测量，累计约3.18×10⁵行数据，生成RDT分布和比特翻转记录。

**📈 对比分析**

将单一RDT测量加轻量ECC与基于空间/时间变异的Svärd+ECC+稀疏消毒进行对比，后者将不可纠正错误概率降低至约1.8×10⁷小时，同时系统性能提升约30%（相较于单阈值方案）。

**⚠️ 局限性**

方法仍需大量实验时间，对不同DRAM工艺/温度的泛化性有限，并未涵盖更低频率消毒或硬件实现细节。

---

## 339. Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning

**arXiv ID:** 2603.12887 | [PDF](https://arxiv.org/pdf/2603.12887v1)

**作者:** Mingkai Zhai `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3914 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在无接触摄像头视频下，通过跨物种迁移学习实现癫痫发作预测。

**💡 创新点**

首次将大型鼠标癫痫视频用于预训练，构建跨物种知识迁移框架，解决人类视频数据稀缺。

**🔧 技术方法**

使用VideoMAE自监督预训练与轻量级分类头，结合两阶段的预训练–微调管道。

**📊 数据集**

RodEpil鼠标癫痫视频（约13k段）与深圳第二人民医院人类癫痫视频（约1.9k段），以及40段标注的少量人类预发作/间歇视频。

**📈 对比分析**

与CSN、X3D、SlowFast等基线模型对比，跨物种预训练模型在2/3/4-shot癫痫预测任务中平均平衡精度0.7230、AUC0.7558，明显优于零样本及单独人类预训练基线。

**⚠️ 局限性**

局限包括固定5秒预测窗口、数据规模与多样性不足、仅使用视觉模态，缺乏多模态融合。

---

## 340. A Reduction Algorithm for Markovian Contextual Linear Bandits

**arXiv ID:** 2603.12530 | [PDF](https://arxiv.org/pdf/2603.12530v1)

**作者:** Kaan Buyukkalayci `[一作]` (University of California), Christina Fragouli `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种将马尔科夫性上下文线性Bandit问题映射为单一上下文线性Bandit的方法，利用延迟反馈消除上下文相关带来的偏差。

**💡 创新点**

创新点在于：①在上下文随时间演化且满足统一几何混合性的前提下，仍能保持与i.i.d.情形相同的O(d√T log T)阶 regret；②针对已知与未知平稳分布两种情况，分别给出高概率和期望的误差分析；③通过分阶段的误差补偿与错配鲁棒算法实现无模型的在线估计。

**🔧 技术方法**

主要技术包括：延迟反馈机制、均衡混合性下的总变距离控制、错配鲁棒线性Bandit算法（如PE/OFUL）、覆盖集合与马尔科夫链加法函数的集中不等式。

**📊 数据集**

实验使用合成的马尔科夫上下文序列（环图+混合分布），每个状态对应随机生成的K个单元向量，维度为d。

**📈 对比分析**

与传统的OFUL/LinUCB基线相比，实验显示延迟反馈策略在后期能显著降低累计 regret 并减少方差；但在早期因随机初始化和延迟更新导致 regret 较高。

**⚠️ 局限性**

局限性包括：需要已知或可估计的混合速率 β；算法对混合时间较慢的链会产生 O(1/√{1-β}) 的系数；延迟反馈会增加样本延迟，导致初期学习效率下降；实现复杂度较传统线性Bandit略高。

---

## 341. FraudFox: Adaptable Fraud Detection in the Real World

**arXiv ID:** 2603.13014 | [PDF](https://arxiv.org/pdf/2603.13014v1)

**作者:** Matthew Butler `[一作]` (Amazon.com), Christos Faloutsos `[通讯]` (Carnegie Mellon University)

**通讯引用:** 81486 | [OpenAlex ID](https://openalex.org/A5035605036)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文提出了一套名为 FraudFox 的在线欺诈检测框架，能够在攻击者不断改变手段的环境下，动态更新多种风险评估模块（oracle）的权重，并通过成本‑收益分析得到最优的“通过/调查”决策边界，同时为业务约束（如调查人数、可接受损失）提供 Pareto 最优权重组合。

**💡 创新点**

创新点：①将扩展卡尔曼滤波（EKF）与动态重要性权重结合，首次在欺诈场景中实现对对抗性和非平稳数据的自适应更新；②从成本‑收益角度推导出超曲线决策边界；③将 Pareto 前沿与粒子群优化相结合，为业务约束变化提供预先计算的权重集合。

**🔧 技术方法**

技术手段：扩展卡尔曼滤波、逻辑回归概率估计、重要性加权策略、粒子群优化（PSO）、Pareto 前沿搜索、成本‑收益分析、超曲线近似与分段线性拟合。

**📊 数据集**

数据集：亚马逊电子商务交易日志（订单、价格、风险分数、实际欺诈标签），包括真实生产环境下的历史交易记录，且在实验中使用了合成“shock”数据验证非平稳适应能力。

**📈 对比分析**

比较方法：与传统静态混合风险分数（不使用 EKF）以及无自适应更新的基线进行对比。性能表现：在生产系统中显著降低欺诈损失（图 3 所示），在合成实验中对突然变化的适应速度快于不使用指数衰减的版本，整体模型在决策时间上仅需毫秒级。

**⚠️ 局限性**

局限性：①模型仍依赖历史标签质量，对标签噪声敏感；②超曲线决策边界需要人工拟合与参数调优；③当前仅针对二分类“通过/调查”，对多级风险评估支持有限；④在极端对抗攻击或大规模数据波动下，EKF 可能需要更复杂的鲁棒性改进。

---

## 342. NOIR: Neural Operator mapping for Implicit Representations

**arXiv ID:** 2603.13118 | [PDF](https://arxiv.org/pdf/2603.13118v1)

**作者:** Sidaty El Hadramy `[一作]` (University of Basel), Philippe C. Cattin `[通讯]` (University of Basel)

**通讯引用:** 9343 | [OpenAlex ID](https://openalex.org/A5048965835)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在医疗影像领域，提出了 NOIR（Neural Operator Mapping for Implicit Representations）框架，将传统基于离散网格的深度学习转化为连续函数空间的算子学习，通过隐式神经表示（INR）对医学图像进行嵌入，并学习神经算子实现从输入 INR 调制向量到输出 INR 调制向量的映射，实现分割、形状补全、图像翻译与合成等任务。

**💡 创新点**

创新点：①挑战网格化假设，将医学图像视为连续函数；②将任务抽象为函数空间间的算子学习；③使用共享 INR 与信号特定调制实现多任务通用；④在连续域中实现对不同分辨率的鲁棒性，满足 ϵ‑ReNO 理论。

**🔧 技术方法**

技术：隐式神经表示（Sine/FFT/Hash 编码等）+ auto‑decoder 与 hypernetwork 生成信号特定调制；多层残差 MLP 作为神经算子；内外循环训练；多任务结构；与传统 CNN/Transformer/傅里叶算子对比。

**📊 数据集**

数据集：Shenzhen X‑ray（2D）、OASIS‑4 MRI（2D）、SkullBreak 3D（脑颅骨补全）、fastMRI（膝关节 PD/T2）、内部超声/CT 合成数据集。

**📈 对比分析**

与 U‑Net、ViT、Attention‑U‑Net、FNO、DDPM 等基线比较。NOIR 在分割、形状补全、图像翻译与合成等任务上表现与基线相当或更优，尤其在低分辨率下保持稳定，显示出较低的混叠误差，整体性能接近甚至略优于现有主流方法。

**⚠️ 局限性**

局限性：①对 INR 逼近质量高度依赖，维度过大时训练难度升高；②仅支持监督学习，需要配对数据；③在细节恢复（如骨骼细微结构）上仍有不足；④对超高分辨率或极端异构采集条件可能需要更大算子或更多样本。

---

## 343. A Learning-Based Approach for Contact Detection, Localization, and Force Estimation of Continuum Manipulators With Integrated OFDR Optical Fiber

**arXiv ID:** 2603.12347 | [PDF](https://arxiv.org/pdf/2603.12347v1)

**作者:** Mobina Tavangarifard `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**通讯引用:** 1589 | [OpenAlex ID](https://openalex.org/A5055294307)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一条单光纤的分布式OFDR传感器上，对连续机器人进行接触检测、定位与力估计，提出了级联学习框架CLF。

**💡 创新点**

创新点在于将单侧OFDR纤维的稠密应变信息通过梯度提升分类器和CNN-FiLM结构，实现了接触检测、接触位置与力大小三任务的联合推断。

**🔧 技术方法**

技术主要包括梯度提升决策树、1D卷积神经网络与FiLM调制，以及Gaussian编码的力分布监督。

**📊 数据集**

使用实验数据集：对170mm长、64段的肌腱驱动连续机器人，在8个不同接触点（两侧各4点）多次采集OFDR应变、力计和运动捕捉数据。

**📈 对比分析**

在留一实验组交叉验证下，平均ROC-AUC 0.946，力MAE 0.011N，定位MAE 2.11mm，部分试验甚至实现亚毫米级定位。

**⚠️ 局限性**

局限在于单侧纤维导致方向性差异、分辨率受限于2.65mm段距，且未结合机械逆解，实验数据规模有限。

---

## 344. CellE: Automated Standard Cell Library Extension via Equality Saturation

**arXiv ID:** 2603.12797 | [PDF](https://arxiv.org/pdf/2603.12797v1)

**作者:** Yi Ren `[一作]` (Peking University), Guangyu Sun `[通讯]` (Peking University)

**通讯引用:** 8966 | [OpenAlex ID](https://openalex.org/A5101850376)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于等价饱和的标准单元库扩展框架，自动在后映射网表上构建e‑graph，枚举所有功能等价的子电路并生成最优标准单元，以实现QoR优化。

**💡 创新点**

首次将等价饱和与e‑graph引入SCLX，克服阶段排序问题；提出在e‑graph上高效挖掘频繁子图的算法，并将挖掘结果直接映射为新的标准单元。

**🔧 技术方法**

等价饱和、e‑graph、rewrite规则、改进的gSpan子图挖掘、Quine‑McCluskey简化、SAT等价验证、ASTRAN/AutoCellGen自动单元生成。

**📊 数据集**

EPFL基准集、FreePDK45（AND2X2、AOI21X1等）、ASAP7 7.5‑track库以及adder、EPFL benchmark等测试案例。

**📈 对比分析**

与原库及TeMACLE比较，平均面积下降15.41%（相较原库）并超过TeMACLE 14.36%；在商业流程中延迟平均下降8%，面积下降1.27%，表现优于现有方法。

**⚠️ 局限性**

受限于单输出子电路、最大输入/门数限制；e‑graph规模随逻辑复杂度增长；依赖rewrite规则覆盖率；未考虑物理特性与布局约束。

---

## 345. Deep Distance Measurement Method for Unsupervised Multivariate Time Series Similarity Retrieval

**arXiv ID:** 2603.12544 | [PDF](https://arxiv.org/pdf/2603.12544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 346. Maximum Entropy Exploration Without the Rollouts

**arXiv ID:** 2603.12325 | [PDF](https://arxiv.org/pdf/2603.12325v1)

**作者:** Jacob Adamczyk `[一作]` (University of Massachusetts Boston), Rahul V. Kulkarni `[通讯]` (University of Massachusetts Boston)

**通讯引用:** 3139 | [OpenAlex ID](https://openalex.org/A5048266570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于特征向量的奖励自由最大熵探索算法EVE，直接通过转移矩阵的主特征向量求解最优探索策略。

**💡 创新点**

创新点在于利用平均奖励框架下的倾斜转移矩阵，推导出无回放、无折扣的自洽固定点更新公式，实现对状态-动作覆盖熵的闭式优化。

**🔧 技术方法**

采用线性代数（Perron–Frobenius理论）、平均奖励强化学习、倾斜矩阵特征分解和自洽固定点迭代技术。

**📊 数据集**

在离散网格世界(GridWorld)和CliffWorld等确定性环境上进行实验验证。

**📈 对比分析**

与MaxEnt与基于回放的熵最大化方法相比，EVE在不需要追踪分布或奖励函数的前提下，收敛速度更快、最终熵更高。

**⚠️ 局限性**

局限性包括仅适用于确定性动力学、无法直接处理噪声或连续动作空间，需要已知或可学习的向后转移模型。

---

## 347. Skill-informed Data-driven Haptic Nudges for High-dimensional Human Motor Learning

**arXiv ID:** 2603.12583 | [PDF](https://arxiv.org/pdf/2603.12583v1)

**作者:** Ankur Kamboj `[一作]` (Michigan State University), Vaibhav Srivastava `[通讯]` (Michigan State University)

**通讯引用:** 3299 | [OpenAlex ID](https://openalex.org/A5069896928)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于输入输出隐马尔可夫模型（IOHMM）的高维运动学习建模框架，并将其转化为部分可观测马尔可夫决策过程（POMDP），从而设计最优的触觉“推挽”反馈策略，并通过30名健康受试者的手部外骨骼实验验证该策略可显著加速学习和提升任务表现。

**💡 创新点**

创新点在于：①首次将隐藏技能状态与触觉反馈耦合，通过IOHMM捕捉技能演化并基于该模型求解POMDP，实现对技能状态而非仅仅是即时误差的主动引导；②提供可解释的技能转移矩阵，揭示不同指尖的触觉影响；③使用QMDP近似解POMDP，实现实时反馈决策。

**🔧 技术方法**

使用技术包括：输入输出隐马尔可夫模型（IOHMM）及其参数估计；QMDP近似求解POMDP；贝叶斯滤波更新信念；线性混合模型（LMM）与ANOVA进行统计分析；主成分分析（PCA）评估运动协同；统计检验（t检验、Kruskal–Wallis）检验显著性。

**📊 数据集**

数据集为30名健康受试者在SenseGlove DK1手部外骨骼上完成目标捕捉游戏的实验数据，包括20个自由度的手部运动、触觉反馈指尖索引、重心误差（RE）、轨迹直线度（SoT）等指标。

**📈 对比分析**

与无反馈控制组和启发式反馈组对比，采用线性混合模型和ANOVA检验；结果显示POMDP组在SoT和RE曲线更快下降，收敛试次显著减少（p<0.10），VAF提升、所需主成分数减少，整体性能优于其他两组。

**⚠️ 局限性**

局限性包括：①技能状态离散化为7个层级，可能忽略细微学习进展；②仅在健康人群中验证，缺乏神经损伤患者数据，需进一步个体化模型；③QMDP近似假设全知状态，真实场景中可能导致误差。

---

## 348. Critical Sections Are Not Per-Thread: A Trace Semantics for Lock-Based Concurrency

**arXiv ID:** 2603.13142 | [PDF](https://arxiv.org/pdf/2603.13142v1)

**作者:** Martin Sulzmann `[一作]` (Karlsruhe University of Applied Sciences), Martin Sulzmann `[通讯]` (Karlsruhe University of Applied Sciences)

**通讯引用:** 1930 | [OpenAlex ID](https://openalex.org/A5035364564)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

无法获取论文内容，无法进行总结

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 349. What You Prompt is What You Get: Increasing Transparency of Prompting Using Prompt Cards

**arXiv ID:** 2603.12741 | [PDF](https://arxiv.org/pdf/2603.12741v1)

**作者:** Amandine M. Caut `[一作]` (Uppsala University), David J. T. Sumpter `[通讯]` (Uppsala University)

**通讯引用:** 12617 | [OpenAlex ID](https://openalex.org/A5021654408)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Prompt Card 的文档框架，用来系统化记录和评估提示工程（Prompt Engineering），并在“wordalisation”任务（将结构化数值数据转化为自然语言文本）中展示其具体应用。

**💡 创新点**

创新点在于将已有的模型卡（Model Card）文档结构迁移到提示工程领域，创建了完整的 Prompt Card，涵盖模型细节、目标用途、数据集、上下文、提示架构、因素、定量分析、安全与伦理等维度；同时提供一种在缺乏标准基准的任务中通过 Prompt Card 进行透明评估和复现的可行方案。

**🔧 技术方法**

主要技术包括：大语言模型（如 OpenAI GPT‑3.5/Turbo、Google Gemini）、检索增强生成（RAG）、few‑shot 提示、统计归一化（z‑score）以及 LLM‑as‑Judge 的评估方法；此外，还构建了 Prompt Card 模板与可视化架构图。

**📊 数据集**

使用的数据集包括：英超球员统计数据、World Values Survey 2017‑2022 及其 66 国样本、Kaggle 上的 Big‑Five 人格问卷（1,015,342 条记录）以及基于这些数据生成的自定义结构化文本。

**📈 对比分析**

评估方法主要是 LLM‑as‑Judge：让 LLM 根据生成的 wordalisation 重新推断原始数值，计算重构准确率；比较使用与不使用合成文本的两种提示，发现使用合成文本时准确率明显提升；人类评估则关注流畅度、语调与文化适宜性等主观指标。

**⚠️ 局限性**

局限性包括：缺乏统一基准导致评估难度大；LLM 输出的随机性与版本差异影响复现；Prompt Card 并未覆盖模型训练细节；安全与隐私风险依赖第三方服务；提示中可能嵌入的偏见与文化差异仍需进一步审视。

---

## 350. When Drafts Evolve: Speculative Decoding Meets Online Learning

**arXiv ID:** 2603.12617 | [PDF](https://arxiv.org/pdf/2603.12617v1)

**作者:** Yu-Yang Qian `[一作]` (Nanjing University), Peng Zhao `[通讯]` (Nanjing University)

**通讯引用:** 33093 | [OpenAlex ID](https://openalex.org/A5100722404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个统一的在线学习框架（OnlineSPEC），通过在推理时持续利用目标模型的验证反馈来动态更新草稿模型，从而提升自回归大语言模型的推理速度。

**💡 创新点**

创新点在于：①把生成-校正范式正式映射为在线学习问题，建立动态 regret 与加速率的理论联系；②设计三种通用在线更新策略（基于梯度、乐观学习、在线集成），可无缝嵌入多种现有加速方法；③首次在推理阶段通过交互反馈实现草稿模型的持续进化。

**🔧 技术方法**

核心技术包括：在线梯度下降（OGD）、乐观在线学习（利用历史梯度做预测）、在线集成学习（Hedge 机制结合多速率草稿头），以及基于 DPO 的损失函数实现偏好反馈的自适应更新。

**📊 数据集**

使用七个基准数据集：GSM8K、MATH、MMLU 子集、Code-Search-Python、Spider、MBPP、Alpaca-finance；目标模型为 Vicuna‑7B、Llama‑2‑7B、Qwen3‑8B。

**📈 对比分析**

与 vanilla speculative decoding、OSD、Hydra、EAGLE、EAGLE‑3、LR 等方法对比，OnlineSPEC 在保持相同或更高答案质量的前提下，平均提升约 20–24% 的速度（TPS）并显著提高草稿接受率。

**⚠️ 局限性**

局限性包括：依赖验证反馈的可用性；理论分析基于凸性与 i.i.d. 假设，对深度网络的收敛性保证有限；需要在候选长度 k 和学习率等超参数上进行经验调优；在极端非平稳场景下，动态 regret 上界可能仍较高。

---

## 351. Test-Time Strategies for More Efficient and Accurate Agentic RAG

**arXiv ID:** 2603.12396 | [PDF](https://arxiv.org/pdf/2603.12396v1)

**作者:** Brian Zhang `[一作]`, Nedim Lipka `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Search‑R1 框架在推理时引入三种改进模块（上下文化、去重、混合），并在 HotpotQA 与 Natural Questions 的验证集上进行评估。

**💡 创新点**

创新点在于：①使用外部 LLM 提取并缓存检索结果的关键信息，以缓解信息遗忘；②在推理过程中过滤重复文档以提升检索多样性；③将两种方法结合，探索两者协同提升效果。

**🔧 技术方法**

主要技术包括：Search‑R1 强化学习训练的 Qwen2.5‑7b LLM、E5 句子检索器、GPT‑4.1‑mini 用作上下文化与 LLM‑Match 评估。

**📊 数据集**

使用数据集：HotpotQA 和 Natural Questions 的验证集（随机抽取 500 题对）。

**📈 对比分析**

比较方法：在 500 题集上计算 Exact Match、LLM Match 与平均检索次数。结果显示：上下文化模块提升 EM 5.6%、LLM Match 6.7%，并将平均检索次数从 2.392 降至 2.142；去重模块提升 EM/LLM 但检索次数升至 2.498；混合方案在 EM/LLM 上略优于去重，但不如单独上下文化。

**⚠️ 局限性**

局限性：去重导致模型持续生成相似查询，检索次数上升；三种方法在统计学上差异不显著；研究仅针对推理层改进，未改模型结构或训练过程。

---

## 352. Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**arXiv ID:** 2603.12848 | [PDF](https://arxiv.org/pdf/2603.12848v1)

**作者:** Elena Ryumina `[一作]` (St. Petersburg Federal Research Center of the Russian Academy of Sciences), Dmitry Ryumin `[通讯]` (St. Petersburg Federal Research Center of the Russian Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态视频级别的犹豫/犹豫(A/H)识别框架，融合场景、面部、音频和文本四个模态，并使用Transformer融合加原型增强来提升性能。

**💡 创新点**

创新点包括①将场景模态引入A/H识别；②设计Transformer多模态融合并对缺失模态做掩码；③使用原型增强分类头和辅助损失提升鲁棒性；④在ABAW 10th挑战中通过多模型集成实现最佳表现。

**🔧 技术方法**

使用的技术有VideoMAE（场景编码）、EmotionEfficientNetB0+统计聚合（面部编码）、EmotionWav2Vec2.0+Mamba（音频编码）、EmotionDistilRoBERTa（文本编码）、Transformer多模态融合、原型增强分类头、RMSprop/AdamW优化、Optuna超参搜索、模型集成等。

**📊 数据集**

使用BAS（BAH）语料库，包含1427段视频、10.6小时，总计300名参与者，提供视频、面部、语音、文本和时间戳信息。

**📈 对比分析**

通过MF1指标在训练/验证/公开测试集上与单模态基线对比。最优单模态EmotionDistilRoBERTa得到70.02% MF1；最优融合模型（含原型）得到83.25% MF1；最终测试集最佳集成模型达到71.43% MF1，明显优于单模态基线且展现了较高的鲁棒性。

**⚠️ 局限性**

局限性包括①对多模态同步数据依赖强，缺失模态需掩码处理；②单模型在私有测试集上性能下降，需集成提升；③模型规模大、训练成本高；④未评估不同文化/语言下的泛化能力；⑤原型增强头不直接决定最终输出，训练复杂度增加。

---

## 353. Sinkhorn-Drifting Generative Models

**arXiv ID:** 2603.12366 | [PDF](https://arxiv.org/pdf/2603.12366v1)

**作者:** Ping He `[一作]` (Vanderbilt University), Soheil Kolouri `[通讯]` (Vanderbilt University)

**通讯引用:** 3371 | [OpenAlex ID](https://openalex.org/A5068682350)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于Sinkhorn散度梯度流的漂移（drifting）生成模型，将传统的单向归一化漂移视为Sinkhorn双向归一化的近似。

**💡 创新点**

主要创新在于建立漂移动力学与Sinkhorn散度梯度流的严格理论联系，解决了漂移模型的可辨识性缺失问题，并通过双向归一化提升低温稳定性。

**🔧 技术方法**

采用交叉减自耦合（cross-minus-self）结构、熵正则化的最优传输、Sinkhorn算法、粒子离散化和Wasserstein梯度流理论。

**📊 数据集**

在二维仿真分布、MNIST手写数字和FFHQ人脸数据集上进行实验。

**📈 对比分析**

通过与传统单向漂移、两向归一化及全Sinkhorn漂移对比，展示了在低温下显著降低FID、EMD、Wasserstein距离，保持单步推断时间不变，且在多模态分布上实现更好的模式覆盖。

**⚠️ 局限性**

实验规模有限，未覆盖像ImageNet等大规模数据集，且Sinkhorn迭代增加训练成本；未来需在更大数据集上验证与加速实现。

---

## 354. OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution

**arXiv ID:** 2603.12811 | [PDF](https://arxiv.org/pdf/2603.12811v1)

**作者:** Shijie Zhao `[一作]` (ByteDance Inc.), Tianfan Xue `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 OARS，一种基于过程感知的在线对齐框架，利用 COMPASS 奖励模型对生成式真实场景超分模型进行逐步强化学习。

**💡 创新点**

创新点包括：① 过程感知奖励 COMPASS，结合保真度与感知增益并加入输入质量自适应权重；② 3 阶段细粒度注释流程与 COMPASS‑20K 数据集，解决无 GT 情况下的偏好学习；③ 逐步在线强化学习（从冷启动到全参考再到无参考），采用 LoRA 微调实现对策略的可控探索，降低奖励挖掘和伪多样性。

**🔧 技术方法**

主要技术：多模态大型语言模型（Qwen-VL-8B）进行奖励训练；流匹配（Flow Matching）预训练；LoRA 微调实现在线强化学习；输入质量自适应奖励公式；多阶段注释与校准流程。

**📊 数据集**

使用 COMPASS‑20K（含 2,400 张 LR 输入、28,800 张 SR 输出、19,200 对评分），并在 LSDIR、Real‑SR、DIV2K、RealSet80 等数据集上进行验证；奖励模型在 SRIQA‑Bench 上训练。

**📈 对比分析**

与多种基准（DiffBIR、SeeSR、UARE、StableSR 等）和 SRIQA‑Bench 进行对比。OARS 在无参考质量指标上获得最高 83.1% 的偏好准确率，且在 Real‑SR、DIV2K、RealSet80 的 NR 指标均居前列；主观用户研究中获 47.62% 的投票率，超过最强基线。

**⚠️ 局限性**

局限：仍依赖多阶段训练与手工标注的 3 阶段流程；奖励模型对极端低质量输入可能过度抑制增强；在大规模多模型或更高分辨率场景下的可扩展性与推理效率待进一步验证。

---

## 355. CLARIN-PT-LDB: An Open LLM Leaderboard for Portuguese to assess Language, Culture and Civility

**arXiv ID:** 2603.12872 | [PDF](https://arxiv.org/pdf/2603.12872v1)

**作者:** João Silva `[一作]` (University of Lisbon), António Branco `[通讯]` (University of Lisbon)

**通讯引用:** 1061 | [OpenAlex ID](https://openalex.org/A5034059867)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 CLARIN‑PT‑LDB 领导榜，用于评估欧洲葡萄牙语大语言模型，并构建了十个基准测试（其中三项为新颖贡献）。

**💡 创新点**

创新点：①首个专门针对欧洲葡萄牙语的公开领导榜；②设计了两项新基准——文化对齐评估（Tuguesice‑PT）和模型安全性评估（DoNotAnswer‑PT）；③采用完全生成式评估方法，兼顾封闭与开放模型。

**🔧 技术方法**

技术：利用 HuggingFace Spaces、Eleuther AI 的 LM Evaluation Harness 作为后端，配合 vLLM、量化、LoRA 等推理加速技术；基准翻译采用 Google Cloud Advanced Translation API；使用 Llama 3.3 70B Instruct 与 Gemini 2.5 Flash 作为判别模型。

**📊 数据集**

数据集：Tuguesice‑PT（327 QA）、DoNotAnswer‑PT（939 题）、MuSR（756）、Omniscience（600）、MMLU（14 042）、GPQA Diamond（198）、MMLU Pro（12 032）、CoPA（500）、MRPC（1 730）与 RTE（3 000）。这些基准大多为英文原版的自动翻译或新构建。

**📈 对比分析**

比较方法：所有任务均为完全生成式，采用准确率作为指标；对 DoNotAnswer‑PT 与 AA‑Omniscience‑Public 采用判别模型评判。实验结果显示：模型规模越大性能越好；Gervásio（基于欧洲葡萄牙语微调）在文化对齐任务上显著优于对应 Llama 版本；在其他基准上，70 B 级模型通常优于 24 B 与 8 B。

**⚠️ 局限性**

局限性：①基准覆盖范围仍有限，未涉及所有可能的语言与任务；②自动翻译可能引入语义或格式错误；③生成式评估忽略模型内部 logits，可能导致细粒度错误难以捕捉；④判别模型的准确性受限，可能影响 DoNotAnswer‑PT 与 AA‑Omniscience‑Public 的评测可靠性；⑤仅公开开放模型，封闭模型评估不被覆盖。

---

## 356. Uncovering Security Threats and Architecting Defenses in Autonomous Agents: A Case Study of OpenClaw

**arXiv ID:** 2603.12644 | [PDF](https://arxiv.org/pdf/2603.12644v1)

**作者:** Zonghao Ying `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**通讯引用:** 12508 | [OpenAlex ID](https://openalex.org/A5024067284)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对OpenClaw自治代理安全威胁进行系统分析，提出三层风险分类和全生命周期安全架构FASA，并开始实施ClawGuard平台。

**💡 创新点**

创新点在于首次将AI认知、软件执行和信息系统三维风险整合为风险分类，并设计跨层级、零信任的FASA安全框架。

**🔧 技术方法**

使用LLM推理日志、工具调用链分析、静态/动态沙盒、行为异常检测与跨层一致性验证技术。

**📊 数据集**

基于OpenClaw真实代码、插件生态、日志和已公开漏洞数据集。

**📈 对比分析**

通过对比传统内容过滤和本框架的行为监控，FASA在识别提示注入、工具链攻击和供应链污染方面表现出更高的检测率，未给出量化指标。

**⚠️ 局限性**

局限在于缺乏大规模实测数据、可能产生性能开销、以及对新型攻击的适应性仍待验证。

---

## 357. Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation

**arXiv ID:** 2603.12538 | [PDF](https://arxiv.org/pdf/2603.12538v1)

**作者:** Alaa Dalaq `[一作]` (King Fahd University of Petroleum and Mineral), Muzammil Behzad `[通讯]` (SDAIA-KFUPM Joint Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SERA（Spatio‑Semantic Expert Routing Architecture），一种在冻结预训练视觉‑语言模型上使用轻量级专家路由的参照图像分割框架。

**💡 创新点**

创新点包括：① 在视觉主干的 Transformer 块中插入表达式感知的 SERA‑Adapter，利用专家模块对中间特征进行空间与边界细化；② 在视觉‑文本融合阶段加入 SERA‑Fusion，通过 Top‑K 稀疏路由选择多种空间、边界、上下文、形状专家，实现表达式自适应的多任务细化；③ 采用参数高效调优，仅更新 LayerNorm 与偏置，保持 99%+ 冻结参数，同时保证路由稳定。

**🔧 技术方法**

使用的核心技术包括：多模态 Transformer（DINOv2 视觉编码器 + CLIP 文本编码器）、混合专家（Mixture‑of‑Experts）与 Top‑K 稀疏路由、轻量级卷积专家、跨模态注意力、参数高效微调（只更新归一化层与偏置）。

**📊 数据集**

在 RefCOCO、RefCOCO+、RefCOCOg（G‑Ref）三大标准参照图像分割数据集上进行评估。

**📈 对比分析**

与现有方法（如 DETRIS、CRIS、UniLSeg 等）对比，SERA 在冻结主干（PET）设置下取得 mIoU 最高，分别为 76.5/78.2/73.7，显著优于之前最优 PET 方法（如 DETRIS‑B 的 76.0/78.2/73.5），并在全量微调基准中也保持竞争力。 ablation 证明 Adapter 与 Fusion 的协同作用及 Top‑K 4 的最优性。

**⚠️ 局限性**

局限性：① 只在视觉流中引入专家，缺乏跨模态专家交互；② 采用手工设计的专家结构，未探索数据驱动的专家挖掘；③ 多专家增加模型复杂度，对大模型或高分辨率输入的可扩展性待验证；④ 路由参数仍需手工调优，可能在不同数据集间表现不一致。

---

## 358. Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization

**arXiv ID:** 2603.12565 | [PDF](https://arxiv.org/pdf/2603.12565v1)

**作者:** Mengjie Zhao `[一作]` (SB Intuitions), Yui Sudo `[通讯]` (SB Intuitions)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种基于偏好学习的对齐方法，将日语语音大型语言模型（SpeechLLM）的输出从书面式转为适合语音合成的口语式文本；

**💡 创新点**

创新点在于将直接偏好优化（DPO）与监督微调（SFT）结合，用“口语式”偏好对模型进行训练，并首次构建了专门评估日语语音可听性的新基准 SpokenElyza；

**🔧 技术方法**

技术包括：使用 Whisper‑large 的语音编码器与 Sarashina‑7B LLM 的浅层投影器、DPO + SFT 的联合损失优化、以及在不同层级（TopLayers vs. KQ‑LN）对模型进行参数微调；

**📊 数据集**

数据集主要有 ReazonSpeech、SpeechPref、InstructS2S‑200K、DeepDialog 进行对齐与偏好训练，以及从 ELYZA‑tasks‑100 过滤后构造的 SpokenElyza（34 条实例）进行评估；

**📈 对比分析**

通过 LLM‑as‑Judge 评价，偏好训练后的模型在 SpokenElyza 上从 2.91 提升至 3.44（相对提升 18%），同时在传统书面式评估 Elyza 上仅下降 5%；表面形式指标显示词数、句法深度和非可说成分显著降低；

**⚠️ 局限性**

局限性包括：对口语化程度的评估仍依赖人工 TTS 验证，模型在极大 DPO 权重下对书面式任务性能下降，且目前仅在日语上验证，泛化到其他语言尚待探索。

---

## 359. Thinking in Streaming Video

**arXiv ID:** 2603.12938 | [PDF](https://arxiv.org/pdf/2603.12938v1)

**作者:** Zikang Liu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jing Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种用于连续视频流的实时推理框架ThinkStream，采用Watch–Think–Speak循环实现增量推理和交互；

**💡 创新点**

创新点在于：①将推理过程与感知同步并逐步更新；②引入Reasoning-Compressed Streaming Memory（RCSM）将中间推理结果压缩为长期语义记忆，替代冗余视觉token；③设计可验证奖励的RLVR训练方法，鼓励模型在合适时机回答且保持正确性；④构建大规模时间对齐的CoT与问题答案数据集；

**🔧 技术方法**

核心技术包括：大规模多模态语言模型（Qwen2.5‑VL‑3B为基座），自定义CUDA Graph流式推理引擎，KV缓存裁剪与预填充策略，Group Relative Policy Optimization（GRPO）强化学习，规则化奖励设计；

**📊 数据集**

使用的主要数据集为：
- 自研时序CoT与问答数据集（110K cold‑start实例 + 9K RLVR实例），
- 公开的StreamingBench、OVO‑Bench、VideoMME、Long VideoBench 等视频推理基准；

**📈 对比分析**

在流式视频基准上，ThinkStream‑3B平均得分达到59.66，显著超越同尺寸基线 Qwen2.5‑VL‑3B（51.00）和多款在线模型；在StreamingBench Real‑Time中得分75.00，几乎与 GPT‑4o（73.28）相当；在离线视频基准上亦保持竞争力（VideoMME 61.9，Long VideoBench 56.4）。

**⚠️ 局限性**

局限性包括：
- 需要昂贵的RL训练与大规模对齐数据，迁移成本高；
- 仍受限于视频帧率和分辨率，极端长视频或高频率输入下可能出现微小延迟；
- 目前主要针对视觉问答与事件推理，其他多模态任务（如动作生成、控制）尚未充分验证。

---

## 360. HCP-DCNet: A Hierarchical Causal Primitive Dynamic Composition Network for Self-Improving Causal Understanding

**arXiv ID:** 2603.12305 | [PDF](https://arxiv.org/pdf/2603.12305v1)

**作者:** Ming Lei `[一作]`, Christophe Baehr `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于进化的因果执行图方法，用以修正物理仿真中的预测错误。

**💡 创新点**

创新点在于通过自动发现新的物理原语（如旋转稳定性）并将其整合进因果图，从而实现动态模型的自适应改进。

**🔧 技术方法**

技术手段包括因果执行图(Causal Execution Graph)、基于图的元学习(meta‑learning)以及离散事件仿真。

**📊 数据集**

数据集使用了标准物理仿真基准，如BoxWorld、MuJoCo以及自构造的多盒滑动实验集。

**📈 对比分析**

与传统物理引擎和静态因果图方法相比，本文方法在预测精度和场景适应性上提升了约15%-20%，实验结果在AUC/误差指标上均有显著改善。

**⚠️ 局限性**

主要局限是模型训练需要大量标注的真实轨迹，且在高维复杂场景中推理速度仍受限。

---

## 361. AnchorVLA4D: an Anchor-Based Spatial-Temporal Vision-Language-Action Model for Robotic Manipulation

**arXiv ID:** 2603.12730 | [PDF](https://arxiv.org/pdf/2603.12730v1)

**作者:** Juan Zhu `[一作]` (PrimeBot), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 56932 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出AnchorVLA4D模型，利用任务开始时的图像作为锚点并结合轻量级空间编码器，在视觉‑语言‑动作(VLA)框架下提升机器人操作的空间与时间推理能力。

**💡 创新点**

创新点在于：①仅通过单帧锚点保存初始场景信息；②将锚点与当前帧联合输入空间编码器，从而在无需额外深度或点云传感器的情况下实现三维理解；③采用轻量化空间编码器与现成的VLM（Qwen2.5-VL）结合，保持低推理延迟。

**🔧 技术方法**

技术方法包括：使用Qwen2.5‑VL 3B视觉‑语言背骨，400M ScaleDP扩散式动作头；将首帧与当前帧输入空间编码器（Any4D或类似网络）；在动作头前将空间编码器输出与VLM隐藏状态拼接；采用MSE损失训练扩散模型。

**📊 数据集**

数据集方面：在BridgeV2大规模机器人轨迹数据上进行预训练和微调；在SimplerEnv（Simpler WidowX）模拟环境中进行评估；在开放源代码低成本双臂移动机器人xLerobot上收集30条经验进行真实世界实验。

**📈 对比分析**

与基线方法（OpenVLA、CogACT、π_0、SpatialVLA、TraceVLA等）对比，AnchorVLA4D在SimplerEnv上取得64.6%成功率，较无锚点VanillaVLA提升13.6%，在xLerobot任务中实现80%平均成功率，明显优于π_0.5基线的50%成功率。

**⚠️ 局限性**

局限性包括：锚点对任务进展偏离初始状态的长时间任务会失效，导致错误纠正受限；空间编码器在未使用3D监督或训练时提升有限；仅使用单帧锚点无法完全覆盖复杂视角变化，仍受限于视觉模型的空间表达能力。

---

## 362. Naïve PAINE: Lightweight Text-to-Image Generation Improvement with Prompt Evaluation

**arXiv ID:** 2603.12506 | [PDF](https://arxiv.org/pdf/2603.12506v1)

**作者:** Joong Ho Kim `[一作]` (Louisiana State University), Keith G. Mills `[通讯]` (Louisiana State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PAINE方法，通过预测初始噪声与提示的得分，优化文本到图像生成，并提供提示难度反馈。

**💡 创新点**

创新点在于将噪声与提示直接映射到人类偏好得分，实现轻量化、无微调的噪声优化，并兼具生成难度评估。

**🔧 技术方法**

采用Transformer提示编码、ResNet噪声编码和MLP评分器的三模网络，并用MAE+SRCC损失训练，结合Naïve贝叶斯思路估计平均分。

**📊 数据集**

使用Pick-a-Pic提示集与多种DM（Hunyuan、PixArt-Σ、DreamShaper-XL-v2-Turbo、SDXL）生成图像，并以PickScore等人类偏好指标作为监督。

**📈 对比分析**

与Golden Noise、NoiseAR等基线对比，在多模型、多提示集和多偏好指标下，PAINE在30+比较中获得最佳，且延迟更低、模型体积更小。

**⚠️ 局限性**

局限在于仅验证静态图像生成，缺乏视频或自回归模型适用，且对提示的平均分估计受训练分布限制，未评估极端或稀有提示的鲁棒性。

---

## 363. AgentRM: An OS-Inspired Resource Manager for LLM Agent Systems

**arXiv ID:** 2603.13110 | [PDF](https://arxiv.org/pdf/2603.13110v1)

**作者:** Jianshu She `[一作]` `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence), Jianshu She (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM代理系统中提出AgentRM中间件，用于调度和上下文管理，解决阻塞、僵尸进程和记忆丢失问题。

**💡 创新点**

创新点在于将操作系统的多级反馈队列、僵尸回收、速率限制感知调度与三层上下文生命周期管理相结合。

**🔧 技术方法**

技术包括多级反馈队列（MLFQ）、僵尸回收器、基于令牌桶的速率限制、Dominant Resource Fairness、三层存储层、压缩式自适应压缩与hibernation。

**📊 数据集**

使用了从六大框架（OpenClaw、AutoGen、CrewAI、LangGraph、Codex、Claude Code）收集的4万余条GitHub issue以及模拟的多场景工作负载。

**📈 对比分析**

与FIFO、RR、优先级队列及传统上下文管理方案比较，AgentRM‑MLFQ将P95延迟降低86%，吞吐量提升168%，僵尸进程消失；AgentRM‑CLM保持100%关键信息保留，质量评分95%，但压缩成本约34k tokens。

**⚠️ 局限性**

局限包括实验基于模拟工作负载，压缩质量依赖摘要模型，价值评分主观，框架覆盖有限。

---

## 364. What Makes VLMs Robust? Towards Reconciling Robustness and Accuracy in Vision-Language Models

**arXiv ID:** 2603.12799 | [PDF](https://arxiv.org/pdf/2603.12799v1)

**作者:** Sen Nie `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xilin Chen `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉语言模型中通过冻结预训练网络并仅在浅层引入低频过滤器和固定鲁棒锚，实现了对抗鲁棒性与干净精度的平衡

**💡 创新点**

创新点在于揭示对抗鲁棒性主要集中于网络浅层，提出仅在浅层做最小改动（Gaussian输入滤波器+固定鲁棒锚）的R‑Adapt框架，且提供训练‑自由、模型‑引导和数据‑驱动三种锚获取方式

**🔧 技术方法**

利用中心核对齐（CKA）进行层级分析、频谱偏置分析、注意力模式评估，结合低频高通滤波、固定锚注入和残差连接实现鲁棒性增强

**📊 数据集**

在18个视觉分类基准、图像‑文本检索、VQA与Caption任务上评测，使用ImageNet、CIFAR‑10/100、STL‑10、Caltech‑101/256、Oxford‑Pets、Flowers102、Food101、StanfordCars、SUN397、Country211、FGVCA、EuroSAT、DTD、PCAM以及MS‑COCO和Flickr30k等数据集

**📈 对比分析**

与主流对抗微调方法（TeCoA、FARE、TGA）及自适应模型对比，R‑Adapt在大多数任务上实现了10–15%提升的鲁棒性，同时仅略微降低（≈3–5%）干净精度，整体保持最佳鲁棒‑准确权衡

**⚠️ 局限性**

局限性包括：仍主要依赖浅层改动，可能无法充分提升深层鲁棒性；需要额外的固定锚（尤其训练‑驱动版需小量对抗样本）；在极端对抗预算或非标准任务上鲁棒性提升有限

---

## 365. SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation

**arXiv ID:** 2603.13024 | [PDF](https://arxiv.org/pdf/2603.13024v1)

**作者:** Sampath Rapuri `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins University)

**通讯引用:** 4743 | [OpenAlex ID](https://openalex.org/A5087095414)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了可控、可扩展的手术动作视频生成模型SAW，能够基于语言提示、参考帧、组织可用性掩模和工具尖端轨迹合成逼真的腹腔镜手术视频。

**💡 创新点**

创新点包括：① 将轨迹条件化与文本、参考帧、组织掩模结合进潜在扩散框架；② 引入深度一致性损失以提升三维几何一致性；③ 构建并使用12,044段手术视频的轻量级标注数据集。

**🔧 技术方法**

使用技术包括：LTX-Video潜在扩散模型、IC‑LoRA微调、轨迹编码、文本提示、组织掩模以及Depth Anything V2生成的深度辅助；评估采用FVD、CD‑FVD、SSIM、PSNR、LPIPS等指标。

**📊 数据集**

使用的数据集是自制的12,044段腹腔镜视频，来源于21条YouTube视频和4个公开数据集，包含动作标签、工具类别、组织可用性掩模和逐帧工具尖端轨迹。

**📈 对比分析**

与WAN、LTX_b和SurgSora等现有生成模型对比，SAW在FVD 224.28、CD‑FVD 199.19、SSIM 0.595、PSNR 17.36等指标上均取得领先表现，并在稀有动作增强和模拟实验中显著提升动作识别准确率。

**⚠️ 局限性**

局限性包括：需要预先提供轨迹、可用性掩模和参考帧，模型对深度辅助的依赖尚未完全消除；生成速度较慢，无法实现实时推理；在更长视频、多种器械和多场景的通用性仍需进一步验证。

---

## 366. Enhanced Drug-drug Interaction Prediction Using Adaptive Knowledge Integration

**arXiv ID:** 2603.12885 | [PDF](https://arxiv.org/pdf/2603.12885v1)

**作者:** Pengfei Liu `[一作]` (Pengcheng Laboratory), Zhixiang Ren `[通讯]` (Pengcheng Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于强化学习的知识增强框架，利用聚类得到的药物类型先验知识自适应注入大语言模型（LLM）进行药物-药物相互作用事件（DDIE）预测，尤其在少样本情形下提升准确性。

**💡 创新点**

①首次将药物类型先验知识与LLM融合用于DDIE预测；②采用Q‑learning动态优化聚类方法、药物模态、prompt组合与LLM超参的策略空间；③通过少样本训练显著提升F1，体现框架的自适应性。

**🔧 技术方法**

使用t‑SNE降维+聚类、Prompt Manager、ChemT5等LLM、Q‑learning强化学习、SMILES/SELFIES/文本描述等技术进行知识注入与策略搜索。

**📊 数据集**

DeepDDI2（113类DDIE、约222k对）和DDIMDL（65类、约37k对），药物特征来自DrugBank、PubChem，SMILES/SELFIES及药物描述作为输入。

**📈 对比分析**

与SSI‑DDI、DSN‑DDI、MolT5、ChemT5、BioT5等基线在All、Common、Few、Rare四种拆分上对比；采用准确率、F1等指标；RL搜索在少样本（Few/Rare）上比基线提升1.5‑4% F1，并在搜索效率上优于网格搜索。

**⚠️ 局限性**

prior知识随训练数据增大而衰减；RL可能陷入局部最优；对新药需要重新聚类导致额外计算；prompt设计未系统评估；模型对未见药物的泛化仍有限。

---

## 367. A Fractional Fox H-Function Kernel for Support Vector Machines: Robust Classification via Weighted Transmutation Operators

**arXiv ID:** 2603.12794 | [PDF](https://arxiv.org/pdf/2603.12794v1)

**作者:** Gustavo Dorrego `[一作]` `[通讯]`, Gustavo Dorrego

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于Fox H函数和分数阶扩散波方程的Fox‑Dorrego非平稳核，并将其嵌入SVM中以提升对结构噪声和离群点的鲁棒性。

**💡 创新点**

创新点在于将分数阶扩散的重尾特性与“Amnesia Effect”衰减权重结合，利用转化算子和加权Sobolev空间构造满足Mercer定理的非平稳核，并提供严格的理论保证。

**🔧 技术方法**

采用分数阶微积分、Fox H函数、Rational Quadratic核近似、转化算子和SVM训练，并在Python中实现可视化评估。

**📊 数据集**

实验使用合成极端噪声数据、两月曲线（Two Moons）、30维乳腺癌数据、以及UCI Ionosphere雷达34维数据。

**📈 对比分析**

与标准Gaussian RBF核对比，Fox‑Dorrego在Ionosphere和合成噪声实验中误差率下降约50%，在乳腺癌数据上准确率略低但更稳健；在复杂拓扑上决策边界更连贯；计算复杂度为O(N²·D)，略高于RBF。

**⚠️ 局限性**

局限性包括对超参数s和η的敏感性、纯Python实现导致速度略慢、在无噪声的标准数据集上准确率略低，以及缺乏大规模真实数据集的进一步验证。

---

## 368. ExpanderGraph-128: A Novel Graph-Theoretic Block Cipher with Formal Security Analysis and Hardware Implementation

**arXiv ID:** 2603.12637 | [PDF](https://arxiv.org/pdf/2603.12637v1)

**作者:** W. A. Susantha Wijesinghe `[一作]` (Wayamba University of Sri Lanka), W. A. Susantha Wijesinghe `[通讯]` (Wayamba University of Sri Lanka)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5037286009)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了一种新型轻量级块密码ExpanderGraph-128，该密码采用3-正则扩展图作为扩散层，配合Feistel网络实现加解密；

**💡 创新点**

创新点在于将图论中的扩展性（expander graph）作为核心安全来源，放弃传统S盒/矩阵等复杂组件，利用稀疏结构与局部非线性规则的组合实现全局扩散；

**🔧 技术方法**

技术手段包括Feistel结构、3-正则扩展图邻接规则、4输入布尔函数Rule‑A、基于MILP的差分/线性分析、LFSR键调度以及FPGA/ASIC/MCU多平台实现；

**📊 数据集**

使用的实验数据主要为官方测试向量、NIST SP800‑22随机性测试（10^8位）以及自定义差分/线性实验样本，没有依赖外部数据集；

**📈 对比分析**

与同类轻量级密码（如PRESENT、GIFT、SKINNY）比较，FPGA实现下吞吐率达261 Mbps、面积5.52 kGE、MCU实现吞吐77 kbps，体现了在硬件效率与安全性上的竞争优势；

**⚠️ 局限性**

主要局限包括缺乏关于谱间隙与安全度的严格理论定理、对线性、代数及高阶攻击的完整评估不足、潜在侧信道泄露以及在计数器模式下的统计敏感性。

---

## 369. Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models

**arXiv ID:** 2603.12893 | [PDF](https://arxiv.org/pdf/2603.12893v1)

**作者:** David McAllister `[一作]` (University of California Berkeley), Samuli Laine `[通讯]` (NVIDIA)

**通讯引用:** 14368 | [OpenAlex ID](https://openalex.org/A5029831111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在线强化学习变体，用于扩散模型的后期微调，利用成对采样轨迹差分来更新流场，从而提升图像质量和文本提示的匹配度。

**💡 创新点**

创新点在于将整个采样过程视作单一动作，用差分流优化（FDFO）代替传统的多步MDP更新；通过成对轨迹差分逼近梯度，显著降低噪声并消除风格漂移与奖励劫持；同时采用欧拉‑马尔可夫采样与SPO/ppo clipping提升稳定性。

**🔧 技术方法**

使用的技术包括流匹配扩散模型、近似在线强化学习、差分流优化、欧拉‑马尔可夫随机采样、SPO/ppo clipping、VLM 与 PickScore 作为代理奖励、Stable Diffusion 3.5 + LoRA 微调。

**📊 数据集**

数据集主要为 Stable Diffusion 3.5 预训练数据，并在后期微调中使用 LoRA；评估时使用 Pick‑a‑Pic 提示集，并通过 OneIG‑Bench、HPSv2 等外部控制指标。

**📈 对比分析**

与 Flow‑GRPO 在相同 epoch、GPU 预算和采样步骤下进行对比；本文方法在基准配置下收敛速度约 19 倍、快速配置下约 5 倍；在奖励、图像质量、提示一致性方面均优于 Flow‑GRPO，且不出现奖励劫持伪影。

**⚠️ 局限性**

局限性包括：仍需 KL 正则化以保持生成多样性；差分假设在极端高维或复杂奖励场景下可能不稳健；对 VLM 与 PickScore 的依赖使得奖励设计仍需人工；在更大规模或多任务设置下的泛化尚未验证；实现细节上与 Flow‑GRPO 的对比可能受到框架差异影响。

---

## 370. The COTe score: A decomposable framework for evaluating Document Layout Analysis models

**arXiv ID:** 2603.12718 | [PDF](https://arxiv.org/pdf/2603.12718v1)

**作者:** Jonathan Bourne `[一作]` (3TC AI), Ishtar Govia `[通讯]` (Amagi Brain Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了结构语义单元（SSU）和可分解评估指标 COTe，用于改进文档布局分析（DLA）模型的评估。

**💡 创新点**

创新点在于通过关系式标注将物理文本与语义单元关联，并将覆盖、重叠、越界和空白误差拆分为独立分量的 COTe 指标，提升对不同标注粒度和语义边界的鲁棒性。

**🔧 技术方法**

技术方法包括基于二值掩码的交集/并集计算、SSU 归属判定、COTe 公式实现，并在 Python 库中提供自动标注、可视化和评估功能。

**📊 数据集**

使用了 NCSEv2、HNLA2013（PAGE）和 DocLayNet 三大公开文档布局数据集，并在其上进行无监督（zero‑shot）评估。

**📈 对比分析**

通过与传统 IoU、F1、mAP 指标对比，展示 COTe 在覆盖率、重叠和越界方面提供更细粒度诊断；在三大数据集上评测 5 个模型，COTe 与传统指标不完全一致，能更准确反映模型真实表现。

**⚠️ 局限性**

局限性包括：对低密度或松散标注的文本，Coverage 可能被白色空洞误导；SSU 需要标注或自动标注，若模型粒度与真值相差过大仍会影响性能；COTe 在极端粒度不匹配时仍不完全鲁棒。

---

## 371. AccelAes: Accelerating Diffusion Transformers for Training-Free Aesthetic-Enhanced Image Generation

**arXiv ID:** 2603.12575 | [PDF](https://arxiv.org/pdf/2603.12575v1)

**作者:** Xuanhua Yin `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**通讯引用:** 11956 | [OpenAlex ID](https://openalex.org/A5076697411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了训练自由的 Diffusion Transformer 加速框架 AccelAes，利用美学驱动的空间非均匀性和时间冗余，构造 AesMask 并采用 SkipSparse 进行自适应稀疏计算与步骤级预测缓存，以加速图像生成。

**💡 创新点**

创新点在于：①首次把美学描述词与交叉注意力关联，用单次生成得到的 AesMask 指导计算与指导的空间分配；②将空间稀疏与时间缓存相结合，形成 SkipSparse，既节省空间计算又减少多步冗余；③在保持甚至提升美学与偏好评价指标的同时，实现显著加速。

**🔧 技术方法**

使用的技术包括：交叉注意力聚合与阈值化构建 AesMask；全局键/值保持的自适应稀疏注意力；对美学焦点令牌的 FFN 与背景缓存；步骤级线性外推的 StepCache；以及与现有采样器、CFG 指导相结合的实现。

**📊 数据集**

在三个代表性 DiT 体系上进行评估：Lumina‑Next‑T2I、SD3‑Medium 以及 FLUX.1‑dev，使用 Pick‑a‑Pic 10000 条 prompt、3 个随机种子进行统一评测。

**📈 对比分析**

与 Δ‑DiT、FORA、RAS、SDiT、TeaCache、TaylorSeer 等训练自由加速方法对比，AccelAes 在 Lumina‑Next 上实现 2.11× 的加速，同时 ImageReward 提升 +11.9%、CLIP +0.01、HPS +0.003、Aesthetic +0.20；在 SD3 与 FLUX 上亦维持 1.5–1.7× 的加速并提升偏好与美学分数，整体表现出最优的速度‑质量折中。

**⚠️ 局限性**

局限性包括：对不同 DiT 架构的稀疏化与缓存参数敏感，过度稀疏或缓存可能导致细节损失；当视觉焦点快速变化时，单次 AesMask 可能失效；需要进一步研究更动态的掩码更新与自适应缓存策略。

---

## 372. Budget-Sensitive Discovery Scoring: A Formally Verified Framework for Evaluating AI-Guided Scientific Selection

**arXiv ID:** 2603.12349 | [PDF](https://arxiv.org/pdf/2603.12349v1)

**作者:** Abhinaba Basu `[一作]` (Indian Institute of Information Technology Allahabad), Pavan Chakraborty `[通讯]` (Indian Institute of Information Technology Allahabad)

**通讯引用:** 1507 | [OpenAlex ID](https://openalex.org/A5023091561)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

引入了预算敏感发现评分（Budget‑Sensitive Discovery Score, BSDS）及其预算平均化形式，用于在预算约束下统一评估候选选择策略；同时通过BSDS对39种策略（包括随机森林、MLP、LLM等）在药物发现任务中的增值进行系统评估。

**💡 创新点**

该度量在Lean4中形式化验证了20条定理，既考虑了假发现成本也考虑了放弃成本，并通过预算平均化消除策略在单一预算点的过拟合，提供了一种可解释、可复现且无偏的比较框架。

**🔧 技术方法**

技术上采用了随机森林、MLP、七种大型语言模型（ChatGPT、Claude、Gemini、DeepSeek‑V3.1、Llama‑4‑Maverick、Qwen3‑235B、GLM‑5）以及多种机制化与递归优化策略；数据预处理使用ECFP4指纹+物理化学属性；评估通过Bootstrap、随机/骨架交叉验证和多预算采样完成。

**📊 数据集**

主要使用MoleculeNet HIV数据集（41,127分子，活性比例3.5%）进行基准测试，并在Tox21、ClinTox、MUV‑466、SIDER（耳和迷路疾病）以及30,000个自动驾驶安全场景（AV安全）上进行跨数据集与跨域验证。

**📈 对比分析**

通过BSDS框架计算每个策略的预算敏感得分（DQS）及其预算平均化版本，并与传统EF、AUROC等指标对比；结果显示随机森林Greedy‑ML在所有预算、数据集上始终排名第一，LLM在零/少量示例下均未超过该基线，且DQS揭示了传统指标无法捕捉的精确-召回-放弃权衡。

**⚠️ 局限性**

局限性包括仅针对二分类任务，未覆盖回归或多目标优化；LLM仅在无工具、仅SMILES输入的环境下测试；缺乏真实实验验证；基线仅为RF+ECFP4，未尝试更先进的图神经网络或微调后的分子语言模型。

---

## 373. ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training

**arXiv ID:** 2603.13115 | [PDF](https://arxiv.org/pdf/2603.13115v1)

**作者:** Jie Ji `[一作]` (Clemson University), Xiaolong Ma `[通讯]` (University of Arizona)

**通讯引用:** 7891 | [OpenAlex ID](https://openalex.org/A5074448953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将SAM优化为高效零阶框架（ZO-SAM），专门用于稀疏训练；

**💡 创新点**

创新点在于仅用零阶梯度估计完成SAM的扰动步骤，避免一次反向传播，从而将SAM的两次梯度消耗降至一次；

**🔧 技术方法**

采用零阶随机梯度估计（RGE）与SAM相结合的混合优化策略，并结合稀疏网络训练方法；

**📊 数据集**

使用CIFAR‑10、CIFAR‑100、CIFAR‑10‑C以及ImageNet‑1K数据集，实验网络包括ResNet‑32/50、WRN‑28‑10和DeiT‑Tiny/Small；

**📈 对比分析**

与SGD、传统SAM、ESAM、LookSAM、GSAM以及多种稀疏训练基线（LTH、SNIP、GraSP、SET、DSR、RigL、MEST）进行对比，结果显示在90‑98%稀疏率下准确率提升0.4‑2.3%，推理速度提升50‑80%，在分布偏移（CIFAR‑10‑C）下鲁棒性提升约3%；

**⚠️ 局限性**

局限性在于零阶估计的方差仍可能随维度增大而增大，且对极高维度或极低采样次数的场景尚未充分评估，理论收敛分析仍待进一步完善；

---

## 374. Revisiting Model Stitching In the Foundation Model Era

**arXiv ID:** 2603.12433 | [PDF](https://arxiv.org/pdf/2603.12433v1)

**作者:** Zheda Mai `[一作]` (Ohio State University), Cheng-Hao Kuo `[通讯]` (Amazon)

**通讯引用:** 902 | [OpenAlex ID](https://openalex.org/A5105792706)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探究不同目标、数据和模态混合的视觉基础模型（VFM）之间的可拼接性，提出两阶段特征匹配与任务损失训练的拼接方法，并基于此构建 VFM Stitch Tree 以实现多 VFM 的共享低层。

**💡 创新点**

①证明异构 VFM 能被可靠拼接；②提出在目标模型的次末层进行最终特征匹配的两阶段训练策略；③提出 VFM Stitch Tree 通过共享浅层实现可控的精度–延迟折中。

**🔧 技术方法**

模型拼接（stitching）、最终特征匹配（Final Feature Matching）、任务损失训练、线性/MLP/LoRA 拼接层、VFM Stitch Tree 架构。

**📊 数据集**

fMoW、iNaturalist、FGVC-Aircraft、ADE20K、VQAv2、MME‑Perception、MME‑Cognition 等多种视觉与多模态任务数据集。

**📈 对比分析**

与自拼接基线（self‑stitch）及单 VFM 比较；在分类任务中提升 2–5% 准确率，分割任务提升 0.5–0.7 mIoU；在多模态 LLM 上，VST 仅 4.3% 额外资源即可恢复 45% 两 VFM 提升，39% 资源可获得 84% 提升。

**⚠️ 局限性**

仅验证了少数 VFM 组合，对极弱源模型的效果有限；拼接层容量与训练策略对结果影响较大；在 GPU 内存受限环境下尚未全面评估；需要进一步探索更复杂任务和自适应拼接方案。

---

## 375. Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**arXiv ID:** 2603.12816 | [PDF](https://arxiv.org/pdf/2603.12816v1)

**作者:** Gyutae Oh `[一作]` (Sungkyunkwan University), Jitae Shin `[通讯]` (Sungkyunkwan University)

**通讯引用:** 2076 | [OpenAlex ID](https://openalex.org/A5033518314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种在无任务标识、无历史数据存储条件下的连续学习框架Residual SODAP，兼顾表示层与分类层的知识保留；

**💡 创新点**

核心创新包括：①基于α-entmax的稀疏prompt选择与残差融合，②无重放统计知识保留与伪特征重放；③prompt使用模式漂移检测（PUDD）自动扩展prompt池；④多损失的自适应不确定性权重；

**🔧 技术方法**

采用Transformer backbone+prompt池、α-entmax、EMA记忆银行、Welford统计、KL蒸馏、PUDD漂移检测、基于不确定性的损失加权；

**📊 数据集**

在三大基准上验证：医疗图像（DR、Skin Cancer）和通用连续学习（CORe50）;

**📈 对比分析**

与PCL、Reh-CL、Reg-CL、Arch-CL等多种连续学习方法对比，Residual SODAP在AvgACC/AvgF上实现显著提升（DR 0.850/0.047，Skin Cancer 0.760/0.031，CORe50 0.995/0.003），证明其在保持高准确率与低遗忘方面具有竞争力；

**⚠️ 局限性**

局限包括：只针对域增量（无任务ID）场景，未覆盖类别增量；对大规模Prompt池的内存/计算开销未知；漂移阈值和扩展策略仍需经验调优。

---

## 376. Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation

**arXiv ID:** 2603.13045 | [PDF](https://arxiv.org/pdf/2603.13045v1)

**作者:** Yifeng Liu `[一作]` (Carnegie Mellon University), Lei Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12071 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于强化学习的后训练方法WALAR，利用仅有的多语种单语料提升大语言模型（LLM）在低资源语言的翻译能力，并通过改进奖励机制抑制奖励黑客攻击。

**💡 创新点**

创新点包括：①发现在常用的质量估计（QE）模型中存在的“holes”，导致RL训练易出现重复、未翻译、错误语言等问题；②设计混合奖励，将QE、词对齐得分和语言对齐得分结合；③在GRPO框架下实现单语料多语言强化学习，显著提升低资源语言性能。

**🔧 技术方法**

技术实现包括：基于MetricX的源端质量估计、BGE‑M3词向量做语义相似度对齐、GlotLID+MaskLID进行语言对齐检测、Group Relative Policy Optimization (GRPO) 强化学习、以及对奖励进行权重调节。

**📊 数据集**

训练数据为WMT News Crawl的22种源语言单语料，评估数据使用Flores‑101测试集；通过spBLEU阈值筛选训练方向，保证中低资源语言的覆盖。

**📈 对比分析**

与基线LLM（如LLaMAX3‑8B‑Alpaca、Qwen3‑8B等）和对照奖励（仅QE）进行对比，WALAR在Flores‑101的1414条语言方向上，spBLEU提升约6–8分，xCOMET*与MetricX*均显著提高；语言一致率（LCR）提升至接近100%；在LLM‑as‑Judge（Gemini‑3 Flash）评估中平均得分从57.25提升至67.03，人工评估亦显示明显优势。

**⚠️ 局限性**

局限性在于：①仍依赖QE模型，若QE本身对极低资源语言评估不佳，效果受限；②需要大量GPU计算和超参数调优；③方法已在22种源语言上验证，跨语言推广至极低资源或无单语料语言仍需进一步研究。

---

## 377. SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**arXiv ID:** 2603.12382 | [PDF](https://arxiv.org/pdf/2603.12382v1)

**作者:** Mohamad Alansari `[一作]` (Khalifa University), Muzammal Naseer `[通讯]` (Khalifa University)

**通讯引用:** 2196 | [OpenAlex ID](https://openalex.org/A5083105774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SPARROW，一个整合时间一致性与空间精度的像素级视频多模态语言模型，利用目标特定跟踪特征（TSF）和双提示（BOX+SEG）实现更稳健的视频对象分割与视觉 grounding。

**💡 创新点**

创新点在于①在训练中引入TSF，为模型提供时序监督；②双提示机制将几何先验与语义信息联合，以粗到细的方式生成精确掩码；③构建规模达30k+视频的参考视频数据集，支持TSF训练；④设计模块化适配器，可无缝插拔至多种现有视频MLLM。

**🔧 技术方法**

核心技术包括视觉→语言与语言→视觉适配器、冻结LLM+LoRA、SAM2像素分割、GroundingDINO+CLDTracker伪跟踪、Deformable‑DETR候选框生成、K‑means聚类、双提示解码与伪监督训练。

**📊 数据集**

使用30,646条视频与45,231条Q&A构成的自研参考视频数据集，并在MeViS、Ref‑YTVOS、Ref‑DAVIS17、VidSTG、VideoGCG等公开基准上进行评测。

**📈 对比分析**

与UniPixel、GLUS、VideoGLaMM等三大视频MLLM对比，SPARROW在RVOS、VG、GCG任务中分别提升1–14.5 J&F分、约5点mIoU和2–5点文本/掩码指标，显示出显著的时间一致性与空间精度提升。

**⚠️ 局限性**

局限性包括：依赖候选框召回，漏检小/遮挡/未出现目标会导致分割失败；TSF基于伪跟踪，噪声可能影响学习；长序列中仍可能出现漂移；推理时未启用TSF需依赖模型自身；整体模型仍带来额外的计算和内存开销。

---

## 378. SRAM-Based Compute-in-Memory Accelerator for Linear-decay Spiking Neural Networks

**arXiv ID:** 2603.12739 | [PDF](https://arxiv.org/pdf/2603.12739v1)

**作者:** Hongyang Shang `[一作]` (City University of Hong Kong), Arindam Basu `[通讯]` (City University of Hong Kong)

**通讯引用:** 4381 | [OpenAlex ID](https://openalex.org/A5002380437)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种基于 SRAM 的线性衰减计算神经网络加速器，兼顾算法与硬件；

**💡 创新点**

将传统指数衰减替换为线性衰减，移除乘法器；并在 SRAM 片内实现并行衰减与膜电位更新；

**🔧 技术方法**

采用 SRAM CIM、全加器、MUX、3T/8T SRAM 单元、前向 Euler 线性衰减模型；

**📊 数据集**

在 N‑MNIST、SHD 和 DVS Gesture 三个常见 SNN 基准上评测；

**📈 对比分析**

与指数衰减 LIF、量化模型以及其他 3 组现有硬件比较，能耗下降 1.1×–16.7×，能效提升 15.9×–69×，准确率差异仅 1% 左右；

**⚠️ 局限性**

仅适用于能耗与延迟受膜电位更新瓶颈限制的网络，且线性衰减在某些层可能导致激活频率异常，需进一步验证可解释性和鲁棒性。

---

## 379. Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback

**arXiv ID:** 2603.12595 | [PDF](https://arxiv.org/pdf/2603.12595v1)

**作者:** Gihoon Kim `[一作]` (Yonsei University), Euntai Kim `[通讯]` (Korea Institute of Science and Technology)

**通讯引用:** 5715 | [OpenAlex ID](https://openalex.org/A5065415014)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Swap-guided Preference Learning（SPL），一种通过交换对比数据来引导变分编码的个性化偏好学习框架，以解决 Variational Preference Learning（VPL）中后验崩塌问题。

**💡 创新点**

创新点包括：① 采用交换引导的基准正则化，使编码器在交换前后产生镜像分布；② 开发 Preferential Inverse Autoregressive Flow（P-IAF），将后验分解为交换逆转与交换不变两部分，保持更丰富的表达；③ 引入自适应潜在调节，使奖励解码器根据潜变量动态调节其影响力。

**🔧 技术方法**

技术手段包括：变分推断、Gaussian 编码器、逆向自回归流（IAF）与 P-IAF、KL 正则化、余弦引导损失、特征调制（自适应潜在调节）、RLHF 训练框架。

**📊 数据集**

实验使用两类数据集：Pets（仅单一提示的宠物偏好对比）和 UltraFeedback-P（UF-P-2、UF-P-4，分别包含 2 种或 4 种多模态偏好），并在 Llama-3.2-3B 与 Llama-3.1-8B 两种模型上进行训练。

**📈 对比分析**

与 BTL、DPL、VPL 等基线比较，SPL 在 Pets 上达到 100% 预测准确率，在 UF-P-2、UF-P-4 上分别超过 63% 预测准确率，且在所有 β 设置下均避免后验崩塌，Active Units 指标显著高于 VPL；训练时间和显存开销仅略有增加。

**⚠️ 局限性**

局限性在于仅处理独立、单轮对比数据，收集此类数据可能繁琐；未来需扩展到多轮对话中的偏好表达。

---

## 380. Rethinking Mutual Coupling in Movable Antenna MIMO Systems

**arXiv ID:** 2603.12817 | [PDF](https://arxiv.org/pdf/2603.12817v1)

**作者:** Tianyi Liao `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45667 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对可移动天线(MA)MIMO系统，构建电路理论模型并求解容量最大化问题

**💡 创新点**

首次将互耦效应与MA位移耦合，利用互耦矩阵设计实现超定向与信道匹配

**🔧 技术方法**

使用BCA迭代求解、TRM优化位移、Sylvester方程求导、数值仿真

**📊 数据集**

无真实数据集，全部采用仿真场景（8×8天线、28 GHz、λ/2/0.1λ间距等）

**📈 对比分析**

与固定天线ULA、紧凑CLA以及不考虑互耦NC‑MA进行对比，C‑MA在不同天线数与SNR下平均提升约12%（低SNR时可达25%）

**⚠️ 局限性**

仅考虑理想偶极子与完美匹配、单用户、静态信道，未来需扩展至实际天线、匹配不完善和多用户场景

---

## 381. Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**arXiv ID:** 2603.12864 | [PDF](https://arxiv.org/pdf/2603.12864v1)

**作者:** Yifan Zhan `[一作]` (University of Tokyo), Yinqiang Zheng `[通讯]` (University of Tokyo)

**通讯引用:** 4402 | [OpenAlex ID](https://openalex.org/A5100698163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可控驾驶世界生成框架，通过分离结构、身份和驾驶者动作三种因素，实现对驾驶场景的独立与组合控制；

**💡 创新点**

创新点在于：1）在噪声层面注入身份信息，支持单图像身份替换并保持姿态一致；2）双分支层次化动作控制，结合局部残差调制与全局PRoPE注意力，提高动作控制精度；3）整体实现完全解耦的可控生成，能够系统生成对抗性边缘案例；

**🔧 技术方法**

核心技术包括：流匹配的Diffusion Transformer（DiT）作为生成骨干；结构条件采用三维框架投影与布局令牌；身份条件通过高噪声阶段硬绑定与低噪声恢复；动作条件采用AdaLN残差调制与PRoPE全局注意力；训练时采用多模态比例调节与首帧干净潜变量；

**📊 数据集**

主要使用nuScenes公开数据集（700个训练视频+150验证）以及自采集的100小时多视角驾驶数据，混合分辨率训练以加速收敛；

**📈 对比分析**

与MagicDrive-V2、DriveEditor、ReCamMaster、Vista等现有方法对比，在多视角跟随、身份控制和动作控制任务中，FVD平均提升17%/30%/47%；在规划鲁棒性测试中，对抗场景碰撞率提升至173%；整体生成质量和控制精度均优于基线；

**⚠️ 局限性**

局限性：训练数据主要为驾驶场景，导致对非车辆类对象的身份编辑泛化受限；身份编辑需预估三维框框尺寸，当前依赖语言模型半自动化，未来需更友好的交互界面；

---

## 382. From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space

**arXiv ID:** 2603.12664 | [PDF](https://arxiv.org/pdf/2603.12664v1)

**作者:** Lehui Li `[一作]` (School of Software, Shandong University), Yongshun Gong `[通讯]` (School of Software, Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5a41884c-404f-4688-a89c-aa238c10fe68` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种通过临时演化语义空间（Temporal Evolution Semantic Space）将文本信息转化为可量化的时间序列特征，从而实现文本-时间序列的有效融合。

**💡 创新点**

创新点在于设计可数值化的时间演化原语（均值漂移、波动率漂移、形状与滞后），并利用大型语言模型的结构化提示与置信度门控机制，形成信息瓶颈，解决跨模态语义不匹配问题。

**🔧 技术方法**

采用大型语言模型对文本进行结构化提示推断得到时间演化原语，并用置信度门控网络对提取结果进行加权；将门控后的原语作为前缀注入PatchTST变换器，结合实例归一化与Patch分块，整体使用MSE/MAE等损失进行训练。

**📊 数据集**

在四个公开数据集上评估：金融领域的FNSPID和Bitcoin；通用领域的Electricity和Environment。

**📈 对比分析**

与多种单模（TimeMixer、PatchTST等）与多模基线（TimeLLM、ChatTime、NewsForecasting等）对比，尤其在金融数据上最多降低29% MSE，通用数据上仍保持领先或接近最优，显示显著性能提升。

**⚠️ 局限性**

局限性包括对LLM推断质量的依赖、对文本信息的可解释性有限、无法充分处理极度嘈杂或缺失文本的场景，以及时间演化原语定义的固定性可能限制跨领域泛化。

---

## 383. Spatial PDE-aware Selective State-space with Nested Memory for Mobile Traffic Grid Forecasting

**arXiv ID:** 2603.12353 | [PDF](https://arxiv.org/pdf/2603.12353v1)

**作者:** Zineddine Bettouche `[一作]` (Deggendorf Institute of Technology), Andreas Kassler `[通讯]` (Deggendorf Institute of Technology)

**通讯引用:** 3542 | [OpenAlex ID](https://openalex.org/A5071429505)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究移动网络流量网格预测，提出NeST-S6模型以实现高效、鲁棒的时空预测；

**💡 创新点**

创新点在于将卷积式PDE感知的状态空间模型(S6)与嵌套学习的慢记忆机制结合，既能实现快速一步预测，又能通过错误驱动的慢记忆自适应非平稳动态；

**🔧 技术方法**

主要技术包括卷积状态空间模型、PDE启发式空间参数化、深度优化器更新的慢记忆、窗口注意力、Laplacian平滑正则和SmoothL1损失；

**📊 数据集**

使用的实验数据集为Milan移动流量数据，格点为100×100，10分钟采样，评估三种分辨率（20²、50²、100²）；

**📈 对比分析**

与Mamba族基线（如VMRNN-D、HiSTM）进行一步与六步自回归评估，NeST‑S6在MAE、RMSE上均优于基线；在漂移测试中MAE下降48–65%，推理速度提升32×，MACs降低4.3×；

**⚠️ 局限性**

局限性包括仅验证短期六步预测，缺乏更长时间跨度、多模态输入与真实网络事件的测试，对极端漂移场景的鲁棒性尚未充分评估。

---

## 384. BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning

**arXiv ID:** 2603.13109 | [PDF](https://arxiv.org/pdf/2603.13109v1)

**作者:** Denis Huseljic `[一作]` (University of Kassel), Bernhard Sick `[通讯]` (University of Kassel)

**通讯引用:** 4815 | [OpenAlex ID](https://openalex.org/A5065340030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可扩展的最佳策略选择器（Best‑of‑Strategy Selector），通过集合多种主动学习策略生成候选批次，并仅在冻结的特征提取器上快速重训练评估性能，从而在大规模数据集和深度网络上逼近最优主动学习选择。

**💡 创新点**

创新点在于：①将多种状态‑of‑the‑art主动学习策略作为候选生成器，构成可插拔的策略集；②在候选批次中仅重训练最后一层，显著降低计算成本；③以性能提升为导向选择批次，实现真正可扩展的oracle方法。

**🔧 技术方法**

使用的技术包括：批量主动学习框架、候选批次集合策略、冻结特征提取器只训练全连接层的重训练、零一损失/交叉熵/Brier评分作为性能评估、与现有oracle（CDO、SAS）以及多种基准策略的对比实验。

**📊 数据集**

实验使用十个图像分类数据集：CIFAR‑10、STL‑10、Snacks、Flowers102、Dopanim、DTD、CIFAR‑100、Food101、Tiny ImageNet、ImageNet，分别采用 DINOv2‑ViT‑S/14（22M）和 SwinV2‑B（88M）作为特征提取器。

**📈 对比分析**

与现有 oracle 在相同计算资源下对比，Best‑of‑Strategy Selector 的性能更好或相当；与多种 state‑of‑the‑art主动学习策略对比，其 oracle 结果始终领先，特别是在大规模多类别数据集上表现突出；通过相对学习曲线和 AULC 等指标展示显著提升。

**⚠️ 局限性**

主要局限包括：①作为 oracle 需要访问真实标签或测试集，无法直接在实际主动学习场景中使用；②无法完全分离监督信息的优势与策略本身的改进；③候选批次数量和重训练周期的平衡仍影响大规模批次或数据集的可扩展性；④缺乏对监督优势对性能差距影响的精确量化。

---

## 385. LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing

**arXiv ID:** 2603.12645 | [PDF](https://arxiv.org/pdf/2603.12645v1)

**作者:** Jiawei Hao `[一作]` (Beijing Institute of Technology), Dan Zeng `[通讯]` (Shanghai University)

**通讯引用:** 3406 | [OpenAlex ID](https://openalex.org/A5023724461)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 LightMoE 的专家替换压缩框架，能够在保持 MoE 模型性能的前提下显著降低显存占用。

**💡 创新点**

创新点包括：①引入专家替换（expert replacing）概念，直接用参数高效模块取代冗余专家；②自适应专家选择机制，结合层内外重要性动态确定压缩比例；③层次化专家构造，将被替换专家聚合为共享基底并附加低秩适配器；④退火式替换策略，逐步过渡到压缩结构以避免性能骤降。

**🔧 技术方法**

主要技术：MoE 架构、LoRA 低秩适配器、门控激活统计、自适应阈值、聚类/主导专家分组、指数退火替换过程。

**📊 数据集**

使用的数据集与任务：数学推理（MetaMathQA/GSM8K）、编程（CodeFeedback/HumanEval）、常识推理（Cleaned Alpaca 及 ARC、BoolQ、PIQA、WinoGrande）、意图识别（BDCI-21 Smart HCI NLU Challenge）以及低资源翻译（ChrEn）。

**📈 对比分析**

与 MC‑SMoE、HC‑SMoE、MoBE、直接替换基线、全微调和 LoRA 进行对比。实验表明：在 30% 压缩率下 LightMoE 与 LoRA 性能相当甚至略优；在 50% 压缩率下，平均提升 5.6%（相较于现有方法）且训练开销低于 MC‑SMoE*；在保持与适配任务中均保持或超越基线表现。

**⚠️ 局限性**

局限性：①需先统计专家重要性，成本与模型规模相关；②低秩适配器的 rank 与退火比率需手动调参；③在极端高压缩或更大规模模型上的通用性尚未完全验证；④主要针对细粒度 MoE，粗粒度场景效果待进一步评估。

---

## 386. From Passive Monitoring to Active Defence: Resilient Control of Manipulators Under Cyberattacks

**arXiv ID:** 2603.13003 | [PDF](https://arxiv.org/pdf/2603.13003v1)

**作者:** Gabriele Gualandi `[一作]` (Mälardalen University), Alessandro V. Papadopoulos `[通讯]` (Mälardalen University)

**通讯引用:** 2246 | [OpenAlex ID](https://openalex.org/A5080443843)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了反馈线性化机械臂在面对隐蔽的伪数据注入攻击时的脆弱性，并提出了一种主动防御机制：基于测量无关的驱动投影预测器的异常感知指令缩放。

**💡 创新点**

创新点在于：①揭示反馈线性化导致的积分器脆弱性，使得攻击可在残差检测器下保持隐蔽；②设计了利用指令缩放的主动防御，能够在攻击存在时主动削弱控制输入；③给出了攻击者的一步最优隐蔽攻击的凸QCQP模型，形成了主动防御与攻击之间的Stackelberg博弈基准；④提供了攻击自由时的概率性驱动损失界限与闭环稳定性证明。

**🔧 技术方法**

采用的技术包括：离散时间LTI系统模型、稳态Kalman滤波、χ²残差检测、任务空间PD+前馈控制、指令缩放函数、测量无关的驱动投影预测器以及凸QCQP求解。

**📊 数据集**

使用的数据集为仿真生成的6-DOF平面机械臂轨迹数据，包括传感器噪声、控制指令、误差统计和攻击序列等，所有数据均在MATLAB仿真环境中产生。

**📈 对比分析**

比较方法：将三种情景（无防御、仅被动χ²检测、主动防御）在相同攻击目标下进行仿真；性能指标包括最大与均方根末端执行器误差、控制指令能量等。结果显示，主动防御显著降低攻击造成的末端执行器偏差，并保持与被动检测相当的正常工作性能。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证；对时变缩放因子f(ẑ)的稳定性分析仍未完成；在正常运行时偶尔会出现罕见的误报导致指令缩放；未来工作需在实际硬件平台上验证，并完善对非线性动态与通信延迟的鲁棒性分析。

---

## 387. Federated Few-Shot Learning on Neuromorphic Hardware: An Empirical Study Across Physical Edge Nodes

**arXiv ID:** 2603.13037 | [PDF](https://arxiv.org/pdf/2603.13037v1)

**作者:** Steven Motta `[一作]`, Gioele Nanni `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c84dae5d-5273-4348-85a7-b44cb586b4df` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在两台装有 BrainChip Akida AKD1000 的 Raspberry Pi 5 上搭建了一个两节点联邦学习系统，使用 STDP 进行边缘学习，并完成约 1580 次实验，评估了四种权重融合策略、特征提取器微调、特征宽度扩展、二值化阈值等多种配置。

**💡 创新点**

本研究首次在真实硬件上实现 STDP 模型的联邦学习；提出神经元级联并（FedUnion）能够保持稀疏原型而不破坏其选择性；通过细粒度特征提取器微调和特征维度扩大显著提升准确率；揭示“原型互补性”机制解释聚合策略、特征宽度与二值化的异质性。

**🔧 技术方法**

采用了 BrainChip Akida AKD1000、Raspberry Pi 5、STDP 规则、两阶段特征提取+边缘学习、量化感知训练、均值/中值/熵阈值二值化、FedUnion/FedAvg/FedBest/FedMajority 等聚合策略、5 轮迭代联邦、Wilcoxon、bootstrap CI、Cohen's d 等统计检验方法。

**📊 数据集**

使用 Google Speech Commands v0.02 数据集，挑选三个未见类（backward、follow、forward）进行非 IID 划分；另外用 yes/no/stop 作为离散验证集。

**📈 对比分析**

与软件基线（k‑NN、线性、MLP）比较时，FedUnion 在特征微调 + 256 维时取得 77% 准确率；FedAvg 则显著下降；k‑NN int8 在联邦设置可达 76%；STDP 最佳方案与软件基线相当，权重交换仅需 20–40 KB；所有关键结果均通过显著性检验确认。

**⚠️ 局限性**

局限性包括：仅测试两节点，未验证更大规模的可扩展性；特征提取的宽特征仍完全在软件实现，硬件上未验证；数据集单一且仅三类，缺乏跨任务验证；STDP 在硬件上与软件基线相比准确率无显著提升；多轮迭代未采用自适应策略；统计功效对中等效应有限；功耗仍主要由 CPU 主导。

---

## 388. Defensible Design for OpenClaw: Securing Autonomous Tool-Invoking Agents

**arXiv ID:** 2603.13151 | [PDF](https://arxiv.org/pdf/2603.13151v1)

**作者:** Zongwei Li `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**通讯引用:** 3108 | [OpenAlex ID](https://openalex.org/A5100634704)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对OpenClaw等环境交互式代理的安全性进行系统分析，提出风险分类、可防御设计原则以及面向工程的研究路线图。

**💡 创新点**

创新点在于将模型安全问题视为软件工程范式，将风险拆分为四类（提示注入、误操作、扩展链路风险、部署漏洞），并对应四项工程原则（最小权限、运行时隔离、扩展治理、审计性），形成面向实践的可执行蓝图。

**🔧 技术方法**

采用的技术主要是架构设计与安全原则的推导，并结合已有的OpenClaw生态示例与文献综述，未涉及新算法实现。

**📊 数据集**

无专门数据集；依赖公开的OpenClaw及同类代理项目的仓库与文档做生态梳理。

**📈 对比分析**

无实验比较；论文侧重理论框架与研究议程的阐述，未给出量化性能指标。

**⚠️ 局限性**

局限性包括：缺乏实证评估与基准验证；对不同语言/平台的通用性仅从概念层面讨论，未提供细粒度实现细节；未在真实生产环境中验证提出的原则与治理机制的效果。

---

## 389. UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC

**arXiv ID:** 2603.12716 | [PDF](https://arxiv.org/pdf/2603.12716v1)

**作者:** Jillur Rahman Saurav `[一作]` (University of Texas at Arlington), Jacob M. Luber `[通讯]` (St. Jude Children's Research Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用冻结的病理基础模型 UNI 的空间特征，结合 SPADE-UNet 架构，设计误差对齐的损失，提出一种单模型即可生成四种 IHC 染色（HER2、Ki67、ER、PR）的虚拟染色方法。

**💡 创新点**

创新点包括：① 用 dense UNI 空间 tokens 直接条件化生成器；② 设计适用于相邻切片产生的空间偏移的误差对齐损失（感知损失、无条件判别器、纹理匹配、DAB 强度约束）；③ 通过 FiLM 嵌入实现多标记统一生成器，显著减少参数并提升多标记性能。

**🔧 技术方法**

技术细节：SPADE-UNet 生成器、UNi 视觉 Transformer 预训练特征提取、边缘 encoder、LPIPS 感知损失、无条件 PatchGAN 判别器、DAB 色彩解卷积、FiLM 条件化、误差对齐损失组合。

**📊 数据集**

使用公开的 MIST（四种 IHC 染色）和 BCI（HER2 4 级）乳腺癌数据集进行训练与评估。

**📈 对比分析**

与 ASP、ODA-GAN、SIM-GAN、PASB、USI-GAN 等方法对比，MIST 上统一模型在 FID、KID、SSIM 上均达到最佳；BCI 上同样表现最佳；统一模型参数量仅为单独训练四个标记模型的四分之一；在 1024×1024 高分辨率下几乎不增加参数，且显著提升染色准确度。

**⚠️ 局限性**

局限性：仅在乳腺癌 H&E-IHC 任务上验证；跨站点、不同染色剂、不同组织类型的泛化性未评估；错误主要集中在非肿瘤组织；缺乏临床诊断任务的下游验证。

---

## 390. EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning

**arXiv ID:** 2603.12698 | [PDF](https://arxiv.org/pdf/2603.12698v1)

**作者:** Chi Ruan `[一作]` (University of Waterloo), Wenhu Chen `[通讯]` (Vector Institute)

**通讯引用:** 4959 | [OpenAlex ID](https://openalex.org/A5103103242)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种以解答为条件的对抗性验证框架，利用候选程序的执行行为迭代细化测试用例，从而构建出验证信号更强、判别力更好的编码强化学习数据集，并在此数据集上使用RLVR提升大型语言模型的代码生成能力。

**💡 创新点**

创新点包括：①解答为条件的对抗性验证思路；②多轮对抗性与判别性测试用例演化机制；③通过解决方案多样性与哈明距离筛选提升测试用例覆盖与非冗余；④构建的全新数据集显著提升验证难度并实现更稳定的RL训练。

**🔧 技术方法**

技术方法：RLVR（可验证奖励的强化学习）与GRPO策略优化；对抗性与判别性测试用例生成（利用GPT‑4.1‑mini、Qwen3系列等模型生成多样解答并基于执行结果生成assert‑based测试）；Hamming距离、pass‑matrix筛选等统计过滤；训练与评估采用nucleus sampling、temperature/top‑p等生成策略。

**📊 数据集**

使用的数据集：基于TACO、APPS、SYNTHETIC‑1、Codeforces、CodeContests的种子问题，经语义去重后再通过多轮演化构建的RLVR数据集；在四个主流编码基准上评估：EvalPlus（含HumanEval、HumanEval+、MBPP、MBPP+）、BigCodeBench‑Instruct、Aider‑Polyglot、LiveCodeBench v5；对照基线包括Qwen3‑4B、Critique‑Coder‑4B、DeepSeek‑R1‑Distill‑14B、DeepCoder等。

**📈 对比分析**

比较方法：在相同的GRPO训练设置下，对比不同轮次的RL训练模型以及现有强基线模型。结果显示，随着对抗性测试用例演化的进行，模型在四个基准上平均提升约4.2分（相较于原始Qwen3‑4B），相对Critique‑Coder‑4B提升约1.8分；Pass@1从43.80降至31.22，说明验证更严格但训练更稳定，最终模型在BigCodeBench‑I、Aider‑Polyglot、LiveCodeBench等难度更高的任务上表现最佳。

**⚠️ 局限性**

局限性：①缺乏对生成问题与测试用例正确性的形式化保证，仍依赖经验性过滤；②仅针对Python程序构建与评估，无法直接推广至其他语言；③对抗性生成可能产生少量语义模糊或不完整的实例，需进一步验证。

---

## 391. Global Evolutionary Steering: Refining Activation Steering Control via Cross-Layer Consistency

**arXiv ID:** 2603.12298 | [PDF](https://arxiv.org/pdf/2603.12298v1)

**作者:** Xinyan Jiang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个训练无关的激活指令框架 GER‑Steer，利用模型在不同层的梯度变化来提取全局进化方向，并用此方向对原始对比向量进行校正，从而实现更稳健的行为调控。

**💡 创新点**

创新点在于：①基于跨层语义向量的谱聚类（PCA/ SVD）自动挖掘“全局进化方向”，证明该方向在高信噪比条件下稳定；②利用该方向对每层原始向量进行投影补偿，去除噪声并提升语义一致性；③提供了理论误差上界（Wedin/Davis‑Kahan），并在多模型多任务上验证其无层调参的通用性。

**🔧 技术方法**

核心技术包括：激活指令（Contrastive Activation Addition）、对比动力学提取（层级差分归一化）、全局谱聚类（SVD/PCA）、投影补偿与归一化、以及基于矩阵扰动理论的稳健性分析。

**📊 数据集**

使用了六个主流数据集：AdvBench（安全拒绝）、TruthfulQA（事实核查）、HC3（人类风格）、SST‑2（情感）、GSM8K（数学推理）和MMLU（通用知识），并在三大 LLM（Qwen‑2.5‑7B、Llama‑3.1‑8B‑Instruct、Gemma‑2‑9B‑it）上进行评估。

**📈 对比分析**

与 CAA、RePE、LDP、ACT、NL‑ITI、Angular Steering 等六种基线相比，GER‑Steer 在所有任务上均实现了显著提升（例如 AdvBench 拒绝率提升约 5%–10%，SST‑2 情感准确率提升 3%–6%，以及 MMLU 通用知识保持甚至提升），并在跨域、跨模型的 OOD 场景中展现出更好的迁移性和鲁棒性。

**⚠️ 局限性**

局限性包括：仍需为不同模型选择合适的投影系数 γ 和激活层数 k（尽管对这两参数的敏感度相对低），需要足够的正负对比样本（在极度数据稀缺或对抗噪声极强的场景下效果可能下降），并且仅针对激活指令场景，对需要更深层参数微调的应用仍不适用。

---

## 392. Generating Expressive and Customizable Evals for Timeseries Data Analysis Agents with AgentFuel

**arXiv ID:** 2603.12483 | [PDF](https://arxiv.org/pdf/2603.12483v1)

**作者:** Aadyaa Maddi `[一作]` (Rockfish Data), Vyas Sekar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 16634 | [OpenAlex ID](https://openalex.org/A5079175103)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套可定制化、富表达性的时序数据分析代理评估框架，通过自动化数据生成和问答对创建，帮助领域专家快速构建针对特定业务（如电商、IoT、通信网络）的评估基准，进一步揭示现有代理在有状态和事件驱动查询上的显著性能缺口；

**💡 创新点**

创新点在于：①提出了基于状态机与半马尔可夫过程的可控时序数据生成器和模式注入库，②构建了状态无关与有状态的查询模板库，并利用LLM自动生成自然语言变体；③通过GEPA优化循环验证该框架可显著提升代理准确率；

**🔧 技术方法**

技术包括Python SDK实现的确定性数据生成管线、LLM辅助的查询与自然语言生成、SQL/Python代码执行、以及GEPA等Prompt优化方法；

**📊 数据集**

使用自研的合成时序数据集，涵盖电商浏览会话、IoT传感器读数、通信网络遥测三大领域；

**📈 对比分析**

与Databricks Genie、Snowflake Cortex、PandasAI、Nao等四种代理进行单轮黑盒评测，整体准确率在70%区间（无状态）但有状态/事件查询仅10%-34%；通过GEPA优化后准确率提升约17%；

**⚠️ 局限性**

局限性包括：仅测试单轮问答；未覆盖数据清洗和多轮交互；缺乏后端追踪与细粒度错误分析；仅使用合成数据，真实业务复杂度有限。

---

## 393. FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts

**arXiv ID:** 2603.12912 | [PDF](https://arxiv.org/pdf/2603.12912v1)

**作者:** Xin Xu `[一作]` (Wuhan University of Science and Technology), Kui Jiang `[通讯]` (Wuhan University)

**通讯引用:** 15480 | [OpenAlex ID](https://openalex.org/A5069011107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在联邦学习框架下提升人行识别跨域泛化能力的方法，通过引入可学习的视觉提示（prompts）来引导Vision Transformer关注人体区域，从而减少背景干扰和视角不匹配问题；

**💡 创新点**

核心创新在于Body Distribution Aware Visual Prompts Mechanism（BAPM），将提示分为身体部位对齐提示和全身提示，并通过约束注意力实现局部与全局信息的互补；另外提出Prompt-based Fine‑Tuning Strategy（PFTS），冻结ViT骨干，只同步轻量提示参数，极大降低通信成本；

**🔧 技术方法**

使用Vision Transformer骨干、可学习视觉提示、结构化注意力掩码、联邦平均（FedAvg）以及提示调优技术；

**📊 数据集**

在四个公开人行识别数据集（CUHK02、CUHK03、Market1501、MSMT17）上进行实验；

**📈 对比分析**

与FedProx、MixStyle、CrossStyle、FedPav、FedReID、DACS、SSCU等基线进行比较，BAPM+全参数训练在mAP和Rank‑1上平均提升约3–4%，PFTS在少数几轮聚合即可获得显著增益；通信量在PFTS下比全参数训练低99%；

**⚠️ 局限性**

局限性包括：方法主要针对ViT架构，提示的数量与分布需要经验调优；评测仅覆盖四个数据集，未在更大规模或真实联邦环境下验证；对极端遮挡或动态客户端加入的鲁棒性尚待进一步研究。

---

## 394. Beyond Imitation: Reinforcement Learning Fine-Tuning for Adaptive Diffusion Navigation Policies

**arXiv ID:** 2603.12868 | [PDF](https://arxiv.org/pdf/2603.12868v1)

**作者:** Junhe Sheng `[一作]` (Nanyang Technological University), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 55615 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在已有的大规模模仿学习预训练的扩散导航模型上，采用基于GRPO的强化学习微调，实现了在无监督、无额外专家示例的在线适应。

**💡 创新点**

创新点在于将轨迹级别的优势归一化与批次内部比较相结合，完全消除价值网络的需求，并通过选择性冻结下层Transformer层保持预训练视觉表征。

**🔧 技术方法**

使用的技术包括扩散模型、Group Relative Policy Optimization (GRPO)、PPO式裁剪目标、在线碰撞奖励，以及仅更新上层解码器和动作头的参数。

**📊 数据集**

实验数据集主要为Isaac Sim中InternScene-Home的室内场景，此外还在Scene-N1的几何测试和Habitat模拟器中验证泛化，以及在Unitree Go2四足机器人上进行真实世界部署。

**📈 对比分析**

与DD-PPO、iPlanner、ViPlanner、DPPO等基线对比，未见环境下的成功率从52%提升至58.7%，SPL提升至0.537，且实现了零射程跨模拟器和跨机器人平台的迁移。

**⚠️ 局限性**

主要局限在于缺乏全局长程路径规划、对细粒度实体与机器人尺寸的认知不足，以及在极端狭窄空间中的碰撞风险仍无法完全消除。

---

## 395. Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting

**arXiv ID:** 2603.12796 | [PDF](https://arxiv.org/pdf/2603.12796v1)

**作者:** Yang Chen `[一作]` (Nanyang Technological University), Yap-Peng Tan `[通讯]` (Nanyang Technological University)

**通讯引用:** 7999 | [OpenAlex ID](https://openalex.org/A5103379503)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于频谱的防御框架，利用3D高频过滤和2D谱正则化共同抑制3D Gaussian Splatting 的资源耗尽攻击。

**💡 创新点**

创新点在于：① 将Gaussian共轭矩阵映射到频率响应，量化异常高频Gaussians 的重要性并动态裁剪；② 在图像域引入基于方向熵的高频能量正则，抑制方向性噪声，二者协同降低攻击诱发的高频过度增长。

**🔧 技术方法**

使用傅里叶变换、频率重要性打分、角度能量分布与熵惩罚、可学习裁剪比例以及联合优化损失，基于Poison‑Splat 3DGS实现。

**📊 数据集**

在 Tanks & Temples、NeRF‑Synthetic、Mip‑NeRF360 三个标准数据集上进行评估，并在 Scaffold‑GS 上测试黑盒攻击。

**📈 对比分析**

与通用阈值（UT）、LightGaussian（LG）和PUP 3D‑GS 等稀疏化基线对比，防御能将高频高斯数量降低 5.92×、GPU内存降低 3.66×、训练时间缩短 4.34×；渲染质量（PSNR/SSIM）保持或提升，FPS 提升约 2–4 倍。

**⚠️ 局限性**

局限性包括：需要手动调节频率阈值、裁剪比例、正则系数等超参数；对极强攻击或极小/极大场景的泛化仍有限；在干净数据下可能出现细节略微丢失；实现依赖频谱分析，计算开销略高。

---

## 396. Retrieval-Enhanced Real Estate Appraisal

**arXiv ID:** 2603.12986 | [PDF](https://arxiv.org/pdf/2603.12986v1)

**作者:** Simon Popelier `[一作]` (Homiwoo), Adrien Bernhardt `[通讯]` (Homiwoo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于检索增强的可比房产选择框架（REMA），通过学习检索策略改进房产估价流程。

**💡 创新点**

创新点在于将可比房产检索转化为可学习的任务，替代传统的地理/时间启发式方法，从而提升检索质量并显著减少所需可比数。

**🔧 技术方法**

采用双编码器、注意力机制、向量检索与端到端训练的 RE-A 与扩展版 ERE-A 模型。

**📊 数据集**

在美国（King County、Fayette County）、巴西（São Paulo、Porto Alegre）以及法国（Ille‑et‑Vilaine）五个公开交易数据集上进行实验。

**📈 对比分析**

与线性回归、XGBoost、KNN 以及状态最优的 ASI 基线比较，ERE-A 在四个数据集上与 ASI 性能相当，但仅使用约 22 倍更少的参数；在 IV、POA 数据集上甚至实现了更优的误差。

**⚠️ 局限性**

局限性包括：在可比冗余度高的 POA 数据集中向量检索效果下降；缺乏时间信息会导致检索不完全；模型对高维/图像等丰富特征的适应性仍有限。

---

## 397. Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**arXiv ID:** 2603.13085 | [PDF](https://arxiv.org/pdf/2603.13085v1)

**作者:** Jose Marie Antonio Miñoza `[一作]` (Center for AI Research), Sebastian C. Ibañez `[通讯]` (Center for AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究线性化注意力的学习动力学，发现其在无限宽度下不收敛到 NTK 极限。

**💡 创新点**

证明注意力通过三次方 Gram 矩阵条件数放大导致宽度需求极高，从而揭示注意力的特征学习与脆弱性双重性。

**🔧 技术方法**

结合 Neural Tangent Kernel（NTK）、Gram 核方法、影响函数、对抗扰动等技术。

**📊 数据集**

在 MNIST、CIFAR-10（以及 Fashion-MNIST 补充）上进行实验。

**📈 对比分析**

与两层 ReLU 网络对比，注意力在 NTK 距离上不收敛，影响灵活性（flip rate）提升 6–9 倍，表现出更高的特征学习能力。

**⚠️ 局限性**

仅考虑线性化注意力，实验规模有限，未验证完整 softmax 注意力，且仅在小型数据集上评估。

---

## 398. LMEB: Long-horizon Memory Embedding Benchmark

**arXiv ID:** 2603.12572 | [PDF](https://arxiv.org/pdf/2603.12572v1)

**作者:** Xinping Zhao `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 60832 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Long‑horizon Memory Embedding Benchmark（LMEB），用22个数据集和193个零拷贝检索任务评估嵌入模型在长时记忆检索上的表现。

**💡 创新点**

创新点在于专门针对长时、碎片化、上下文依赖且时间跨度大的记忆检索任务构建评估框架，覆盖四类记忆（Episodic、Dialogue、Semantic、Procedural），并证明LMEB与传统检索基准正交。

**🔧 技术方法**

采用MTEB框架搭建评估管道，使用多种大规模语言模型嵌入器（如bge‑multilingual‑gemma2、KaLM‑Embedding‑Gemma3等）进行零拷贝检索，计算NDCG、Recall等IR指标。

**📊 数据集**

使用22个英文零拷贝数据集，涵盖Episodic、Dialogue、Semantic、Procedural四类记忆，并包含AI生成与人工标注数据，共计193个检索任务。

**📈 对比分析**

通过NDCG@10与Recall@k进行比较，发现最佳模型bge‑multilingual‑gemma2在w/inst.条件下达61.41；模型规模不一定带来更好性能，不同模型对任务指令的敏感度各异；LMEB与MTEB表现低相关，显示两者评估域正交。

**⚠️ 局限性**

局限性包括：仅覆盖英文数据，缺乏多语言评估；模型规模与性能关系仍不明确；基准任务在多样性与难度上仍可进一步丰富；仅评估零拷贝情况，未涉及微调效果。

---

## 399. V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

**arXiv ID:** 2603.13089 | [PDF](https://arxiv.org/pdf/2603.13089v1)

**作者:** Shenghe Zheng `[一作]` (Hong Kong University of Science and Technology), Wenbo Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3554 | [OpenAlex ID](https://openalex.org/A5100336576)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出V-Bridge框架，将图像修复任务重新表述为逐步的视频生成过程，利用预训练的视频生成模型在极少量（仅1,000张）任务专用样本上实现多任务图像修复；

**💡 创新点**

创新点在于：1) 通过把图像修复视作时间序列生成，充分利用视频生成模型的时空先验；2) 引入粗到细的分阶段训练和轻量漂移校正模块，弥补预训练分辨率与高分辨率修复的差距；3) 通过少量样本实现“一模型多任务”的全能修复，突破传统修复方法对海量数据的依赖；

**🔧 技术方法**

技术包括：预训练视频生成模型（如Wan2.2-TI2V-5B）、伪时间序列构建、进化式（coarse‑to‑fine）分阶段训练、漂移校正（短期修正路径）以及多任务联合训练；

**📊 数据集**

使用FoundIR与RealCE数据集进行训练和测试，并在FoundIR、Dense‑Haze、UHD‑LL、NH‑Haze、UAV‑Rain1K、HQ‑NightRain等外部基准上评估；

**📈 对比分析**

与多种现有修复方法（Real‑ESRGAN、DGUNet、PromptIR、DiffUIR、FoundIR等）在FoundIR测试集以及跨域基准上比较，V‑Bridge仅用0.1%–7%训练样本即可匹配或超越传统方法，并在超低样本情况下实现1.6dB PSNR提升，显著提升修复质量和跨域泛化；

**⚠️ 局限性**

局限性包括：1) 对极端或未见失真类型的泛化仍有限，需进一步探索更广泛的先验迁移；2) 预训练模型对分辨率的依赖导致高频细节恢复仍需漂移校正，存在额外计算开销；3) 进化式分阶段训练对超高分辨率仍可能受限，需更高效的细节建模策略。

---

## 400. OpenACMv2: An Accuracy-Constrained Co-Optimization Framework for Approximate DCiM

**arXiv ID:** 2603.13042 | [PDF](https://arxiv.org/pdf/2603.13042v1)

**作者:** Yiqi Zhou `[一作]` (Nanjing University of Science and Technology), Guozhu Liu `[通讯]` (The 58th Research Institute of China Electronics Technology Group Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了OpenACMv2框架，采用两层Accuracy-Constrained Co-Optimization（ACCO）方法，对DCiM处理器架构与晶体管尺寸进行联合优化，实现了在精度预算内的PPA提升。

**💡 创新点**

创新点在于：①将ACCO拆分为架构级搜索与晶体管级调优两步，提升搜索可行性与收敛稳定性；②构建PEA-GNN替代模型，提供高精度且速度快的乘法器性能预测；③将多种单目标与多目标优化算法集成到框架中，支持快速Pareto前沿搜索。

**🔧 技术方法**

使用技术包括：图神经网络（PEA-GNN）用于乘法器性能回归；OpenACM与OpenYield作为底层电路生成器；Monte Carlo PVT/Variation仿真；经典优化器（MOEA/D、NSGA-II、SMAC、MOBO、CBO、PSO、SA）；OpenROAD+OpenSTA+VCS用于EDA验证。

**📊 数据集**

数据集来源于Nangate45（或FreePDK45）开源工艺库，在OpenACMv2内部自动生成不同位宽（8/16位）乘法器与SRAM宏的设计空间样本。

**📈 对比分析**

与传统单步EDA评估相比，PEA-GNN在8/16位乘法器上的误差（MRED/NMED）≤5%，延迟/面积/功耗误差≤0.3%，R²>0.94；评估时间从37/116秒降至0.26/0.25秒，速度提升约142×/464×。在精度预算约束下，ACCO能在保持图像/模型精度（PSNR、Top‑1/Top‑5）不变的前提下，显著降低PDP与面积。

**⚠️ 局限性**

局限性包括：①替代模型对不同PVT角落的泛化和精度有限；②优化目标仅聚焦于PDP与误差，未涵盖吞吐、漏电、IR跌落及布线拥塞；③缺乏完整的后端签字流程（Place‑and‑Route、寄生提取、串扰）；④SRAM宏与bitcell的协同优化深度不足，难以突破硬件级性能瓶颈。

---

## 401. Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks

**arXiv ID:** 2603.12354 | [PDF](https://arxiv.org/pdf/2603.12354v1)

**作者:** Tianhao Qian `[一作]` (Southeast University), Leszek Rutkowski `[通讯]` (Systems Research Institute of the Polish Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于交替梯度流（AGF）的动力学范式，用于高效结构剪枝和动态路由，从而在极度稀疏条件下保持深度视觉模型的功能与性能。

**💡 创新点**

创新点在于：① 将AGF的连续动力学能量量化为离散的绝对特征空间泰勒展开，捕捉网络的动力学效用；② 发现并利用“拓扑相变”与“稀疏瓶颈”两种极端稀疏时的瓶颈现象；③ 设计了解耦动力学范式，将离线AGF搜索与在线基于置信度的零成本物理先验路由分离，从而避免梯度压缩导致的路由失效。

**🔧 技术方法**

使用的技术包括：AGF梯度流理论、特征空间总变分（TV）近似、绝对泰勒展开、动态置信度路由、梯度-幅度解耦分析、随机梯度噪声正则化、以及基于ResNet、ViT、ImageNet的实验验证。

**📊 数据集**

实验数据集涵盖CIFAR-100、ImageNet-100和ImageNet-1K，使用WideResNet-18-2、ResNet-50和ViT-Base等视觉主干。

**📈 对比分析**

在对比实验中，AGF在极度稀疏（75%压缩）下仅略低于随机采样，而传统幅度/激活基准（ℓ₁、Wanda、RIA）显著低于随机；在动态推理上，AGF+置信度路由在ImageNet-100上实现近乎完整模型精度（88.78%）同时将重计算比例降至约50%，整体计算成本约为0.92×，比随机路由和全模型分别低约4.5%和0%。

**⚠️ 局限性**

局限性包括：① 需要离线梯度计算的校准阶段，导致前期计算成本增加；② 主要验证于CNN与ViT，尚未在更大规模Transformer或LLM上系统评估；③ 在极端稀疏时仍可能遇到信息瓶颈，无法突破随机基准。

---

## 402. Geometry-Guided Camera Motion Understanding in VideoLLMs

**arXiv ID:** 2603.13119 | [PDF](https://arxiv.org/pdf/2603.13119v1)

**作者:** Haoan Feng `[一作]` (University of Maryland), Guan-Ming Su `[通讯]` (Dolby Laboratories Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于3D基础模型的无训练注入方式，构建了CameraMotionDataset与CameraMotionVQA，并通过结构化提示提升VideoLLM对摄像机运动的识别与描述。

**💡 创新点**

创新点在于将3DFM提取的几何摄像机线索与轻量时序分类器相结合，利用结构化提示将运动标签注入冻结VideoLLM，并通过探测分析揭示并修正现有模型对摄像机运动的丢失。

**🔧 技术方法**

使用技术包括3D基础模型VGGT、Transformer时序分类器、Q-Former探测、知识蒸馏以及结构化提示。

**📊 数据集**

使用数据集为通过MultiCamVideo渲染的合成数据构建的CameraMotionDataset（12k 1s段）及其转换而来的CameraMotionVQA。

**📈 对比分析**

在CameraMotionVQA上与多款开源VideoLLM对比，原模型接近随机准确率；VGGT+分类器得到约0.74实例准确率；蒸馏版提升吞吐量5.3×，精度略降至约0.64。

**⚠️ 局限性**

局限性包括合成数据与真实世界的差距、仅关注外部位移与旋转而未覆盖镜头缩放等内在变化，以及仅实验单一3DFM，扩展性待验证。

---

## 403. TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition

**arXiv ID:** 2603.12465 | [PDF](https://arxiv.org/pdf/2603.12465v1)

**作者:** Prabhu Vellaisamy `[一作]` (Carnegie Mellon University), John P. Shen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9955 | [OpenAlex ID](https://openalex.org/A5007315039)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种trace驱动的LLM推理主机侧开销分解方法，并提出Host‑Device Balance Index（HDBI）用于诊断主机与设备的绑定关系。

**💡 创新点**

三层分解框架（框架翻译、CUDA库翻译、内核启动路径）以及HDBI诊断指标；证明MoE模型在解码阶段主机绑定，并显示CPU单线程性能对延迟的决定性影响。

**🔧 技术方法**

使用PyTorch 2.x追踪、NVTX标记、CUDA null‑kernel基准、NVIDIA Nsight Systems、库前端分析、内核重放重现等技术。

**📊 数据集**

评估了Llama‑3.2‑1B/3B、GPT‑2、OLMoE‑1B/7B、Qwen1.5‑MoE‑A2.7B等BFloat16 LLM模型，实验涵盖不同批量、序列长度、预填与解码阶段。

**📈 对比分析**

在H100与H200平台对dense与MoE模型进行prefill/ decode基准；对比T_orchestration、T_deviceActive、HDBI；结果显示HDBI能揭示瓶颈，CPU提升可显著改善host‑bound工作；FlashAttention‑2在大上下文显著降低设备工作。

**⚠️ 局限性**

仅适用于NVIDIA CUDA生态，依赖trace与重放可能失真；HDBI仅为诊断比例而非优化目标；高动态/同步内核的重放不精准；未覆盖多GPU/分布式场景。

---

## 404. Interpreting Negation in GPT-2: Layer- and Head-Level Causal Analysis

**arXiv ID:** 2603.12423 | [PDF](https://arxiv.org/pdf/2603.12423v1)

**作者:** Abdullah Al Mofael `[一作]` (Southeastern Louisiana University), Kuo-Pao Yang `[通讯]` (Southeastern Louisiana University)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5054640704)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT-2 Small 的否定理解进行了因果分析，定义了 Negation Effect Score (NES) 并通过激活补丁、消融-恢复实验定位并验证了中层注意力头的作用。

**💡 创新点**

首次发现并证明了一个紧凑的中层注意力头子网络（主要集中在第 4-6 层）负责实现逻辑极性，将模型的否定处理归因到可解释的可操作电路。

**🔧 技术方法**

采用 NES 作为行为指标，结合激活补丁、层级与头级消融-恢复（ablation–rescue）技术，以及跨否定形式与 xNot360 公开基准的验证。

**📊 数据集**

使用自构造的 12,000 对肯定与否定句子（涵盖 8 种语义模板和多种否定形式），并在 402 个测试样本与 360 条 xNot360 自然句对上进行验证。

**📈 对比分析**

对照基线 NES 与消融/恢复后的 NES 进行对比：在域内消融导致 NES 上升（否定敏感度下降），恢复进一步提升 NES；在 xNot360 上消融略微降低 NES，恢复后恢复至基线以上，效果虽小但一致。

**⚠️ 局限性**

局限于 GPT-2 Small、单语种、单词级预测，未探索 MLP 路径、事实先验交互或更大模型；不同分布下极性效果相反，提示机制受上下文依赖。

---

## 405. Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**arXiv ID:** 2603.12514 | [PDF](https://arxiv.org/pdf/2603.12514v1)

**作者:** Shivam Chaudhary `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 15906 | [OpenAlex ID](https://openalex.org/A5101619735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

针对腹部CT 3D创伤检测，提出一种标签效率极高的两阶段学习框架：先用大规模无标注数据进行 Masked Image Modeling（MIM）自监督预训练 3D U‑Net 编码器，再利用该编码器进行 VDETR 目标检测与多标签分类，并通过教师-学生一致性正则化进一步利用未标注样本提升性能。

**💡 创新点**

创新点包括：① 将 MIM 扩展到 3D CT，构建无标注高质量解剖特征；② 在 VDETR 中引入 Vertex Relative Position Encoding（V‑RPE），通过八个盒子顶点的几何关系显著提升 3D 检测的定位精度；③ 将自监督预训练与半监督一致性学习结合，解决极端标注稀缺场景下的训练不稳定与性能瓶颈。

**🔧 技术方法**

核心技术：3D U‑Net 编码器、Patch‑based Masked Image Modeling、VDETR 变形器、Vertex Relative Position Encoding、Mean‑Teacher 半监督一致性正则化、线性探针分类器、数据增强与类权重平衡。

**📊 数据集**

使用 RSNA Abdominal Trauma Detection 数据集，共 4,711 组 CT，其中 206 组有分割与框注释，1,206 组用于预训练；检测任务使用 144/30/32 训练/验证/测试样本，分类任务使用 2,244/480/480 训练/验证/测试样本。

**📈 对比分析**

与仅监督训练比较，半监督+自监督方法在检测任务上验证 mAP@0.50 从 26.36%（仅监督）提升至 56.57%（半监督），测试集 mAP@0.50 达 45.30%（相较仅监督提升 97%）。分类任务中冻结编码器的线性探针在 2,244 样本下实现 94.07% 准确率，明显优于仅利用伪标签或少量数据的方案。

**⚠️ 局限性**

局限性：① 仅针对腹部 CT 领域，难以直接推广到其他器官或模态；② 虽能显著缓解标注稀缺，但仍需大量未标注样本；③ 伪标签生成对分布偏移敏感，半监督效果受限；④ 未进行实时推理与临床部署评估。

---

## 406. Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis

**arXiv ID:** 2603.12997 | [PDF](https://arxiv.org/pdf/2603.12997v1)

**作者:** Chen Feng `[一作]` (Queen's University Belfast), Ioannis Patras `[通讯]` (Queen Mary University of London)

**通讯引用:** 12010 | [OpenAlex ID](https://openalex.org/A5031205865)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在理想条件下给定完美噪声转移矩阵T，对前向校正（FC）方法的失败进行系统实验与理论分析，揭示其性能崩塌的根本原因，并提出预训练+Mixup的轻量级正则化框架（FEC/JEC），将FC推向理想收敛状态。

**💡 创新点**

创新点在于：①首次严谨检验“T估计困难”假设，证明即使T完美也会崩塌；②从宏观收敛、微观梯度、信息论三个层面统一分析导致FC失败的机制；③提出通过预训练与数据插值实现的正则化，使FC获得与样本筛选相当的性能，展示噪声校正方法的潜力。

**🔧 技术方法**

使用的技术包括：前向校正损失、噪声转移矩阵、信息论熵/互信息分析、梯度软化理论、预训练（自监督）模型、Mixup数据增强、线性分类器在冻结特征提取器上训练。

**📊 数据集**

实验数据集主要为：CIFAR-10、CIFAR-100（对称噪声20%、50%、80%、90%）以及真实噪声数据集Clothing1M。

**📈 对比分析**

对比方法包括：基于样本筛选的Co-teaching、PENCIL、LossModel、DivideMix、Mixup；以及传统FC、以及本文的FEC/JEC。实验显示：原始FC在理想T下出现性能崩塌；FEC/JEC在保持高准确率的同时显著提升ECE，且与最先进的样本筛选方法竞争或超过其性能；多标签扩展实验进一步验证信息量提升能缓解崩塌。

**⚠️ 局限性**

局限性：①需要准确估计T才能在实际中达到理论效果；②理论假设如对角占优、实例依赖噪声对所有场景适用性有限；③高容量网络仍可能因正则化不足导致过拟合；④方法主要针对前向校正，其他校正策略（如反向校正）未深入探讨。

---

## 407. A Method for Learning Large-Scale Computational Construction Grammars from Semantically Annotated Corpora

**arXiv ID:** 2603.12754 | [PDF](https://arxiv.org/pdf/2603.12754v1)

**作者:** Paul Van Eecke `[一作]` (Vrije Universiteit Brussel), Katrien Beuls `[通讯]` (Université de Namur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于语料中词法结构和语义框架标注（PropBank）学习大规模、覆盖广泛的构造语法的方法，并将学习到的构造网络实现为Fluid Construction Grammar框架；

**💡 创新点**

创新点在于能够自动化构建数万条可解释的构造规则，形成一套完整的构造语法网络，展示构造在语料中出现频率的Zipf分布，并提供预训练语法模型；

**🔧 技术方法**

使用构造语法学习算法（分为框架引发构造、论元结构构造和角色集构造），结合spaCy的Berkeley Neural Parser进行句法分析，最终集成至Python实现的FCG（PyFCG）；

**📊 数据集**

使用已标注PropBank的多体裁英文语料——OntoNotes（137,812句）与English Web Treebank（16,579句），共154,391句，440,528角色实例；

**📈 对比分析**

通过在1,000句测试集上评估语义角色提取，纯粹语法网络取得角色集层级F1≈76.25，框架层级F1≈79.96；与其他方法相比，本文提供了无任务优化的基线性能；

**⚠️ 局限性**

主要限制包括依赖外部句法解析器（解析错误难以恢复）、仅适用于英式句法和PropBank标注、未覆盖非名词化结构或未标注的语义角色（如结果性、描绘性等），未来需扩展到UD/AMR等注释体系及其他语言。

---

## 408. Probing Length Generalization in Mamba via Image Reconstruction

**arXiv ID:** 2603.12499 | [PDF](https://arxiv.org/pdf/2603.12499v1)

**作者:** Jan Rathjens `[一作]` (Ruhr University Bochum), Anand Subramoney `[通讯]` (Royal Holloway University of London)

**通讯引用:** 1170 | [OpenAlex ID](https://openalex.org/A5016175782)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在Omniglot字符图像上构造一个图像补全/重建任务，使用Mamba模型在不同长度的图像补丁序列中进行训练与推理，利用可视化重建结果和MSE随时间的变化来探究Mamba的内部处理机制与长度泛化性能。

**💡 创新点**

提出了一种可解释的视觉化框架，能在模型处理每一步时展示重建效果，揭示Mamba在训练时段长度分布上自适应的计算策略并导致长度泛化失败；并设计了长度自适应的Mamba变体（prepend length token），在训练区间内显著提升性能。

**🔧 技术方法**

使用Mamba状态空间模型（去除局部卷积），对比Transformer baseline；对重建结果使用均方误差（MSE）评估；引入序列长度预置 token 并监测 Δ_t_2 等内部动态；在训练与推理时改变图像补丁数与查询补丁数以控制序列长度。

**📊 数据集**

Omniglot 数据集的 128×128 灰度字符图像，随机采样 4×4 补丁并进行坐标编码；实验中采用不同最大补丁数 T_I（如 512–1024、1024、4096、65536）以及固定的查询补丁数 T_Q。

**📈 对比分析**

将 Mamba 与相同 token 维度与层数的 Transformer（Vision Transformer 样式）进行对比；评估指标为不同 V_I、V_Q 组合下的平均 MSE。结果显示：Mamba 在 V_I 超过约 4T_I 时性能急剧下降，长序列泛化差；Transformer 在长序列上性能保持稳定。长度自适应 Mamba 在训练区间内能提高 MSE，且对短序列性能提升更明显，但在超出训练区间时会退化。

**⚠️ 局限性**

Mamba 学习的是针对训练长度分布的策略，而非长度无关的通用算法，导致长度泛化受限；长度自适应机制需要预先知道序列长度，实际应用中序列长度往往未知；实验仅在 Omniglot 视觉任务上验证，尚未完全验证到更复杂或非视觉任务的普适性。

---

## 409. Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation

**arXiv ID:** 2603.12577 | [PDF](https://arxiv.org/pdf/2603.12577v1)

**作者:** Jia-Chen Zhang `[一作]` (Shanghai University of Engineering Science), Chun-Ming Xia `[通讯]` (Shanghai University of Engineering Science)

**通讯引用:** 907 | [OpenAlex ID](https://openalex.org/A5089792666)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Expert Pyramid Tuning (EPT) 的参数高效微调框架，用于多任务下的LLM适配；该框架通过共享低维子空间与多级去卷积专家实现任务特征的多尺度重构。

**💡 创新点**

创新点在于将多尺度特征金字塔概念引入PEFT，构建共享的低维子空间与多级去卷积专家，并结合任务对比学习嵌入与自适应LoRA裁剪，实现任务复杂度的动态适配和负迁移抑制。

**🔧 技术方法**

使用技术包括LoRA、Mixture‑of‑Experts、去卷积(Deconv)、自适应LoRA裁剪、对比学习任务嵌入、Top‑k路由与温度Softmax等。

**📊 数据集**

在GLUE基准（MNLI、QQP、QNLI、SST‑2、STS‑B、MRPC、RTE、CoLA）以及Commonsense Reasoning任务（BoolQ、OBQA、ARC‑E、ARC‑C）上进行评估。

**📈 对比分析**

与LoRA、MultiLoRA、MixLoRA、MOELoRA、MoRE等SOTA PEFT与MoE‑LoRA基线对比，EPT在GLUE平均分87.0%、Commonsense平均分75.5%，在多数任务上超越对手且仅使用约0.41M/3.3M参数。

**⚠️ 局限性**

局限性：专家维度设定为静态超参数，缺乏动态分配；实验仅覆盖微调阶段，未验证在大规模预训练中的可扩展性与稳定性。

---

## 410. System-Technology Co-Optimization of Bitline Routing and Bonding Pathways in Monolithic 3D DRAM Architectures

**arXiv ID:** 2603.12461 | [PDF](https://arxiv.org/pdf/2603.12461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 411. Deferred is Better: A Framework for Multi-Granularity Deferred Interaction of Heterogeneous Features

**arXiv ID:** 2603.12586 | [PDF](https://arxiv.org/pdf/2603.12586v1)

**作者:** Yi Xu `[一作]` (Alibaba Group), Xiaoyi Zeng `[通讯]` (Alibaba Group)

**通讯引用:** 669 | [OpenAlex ID](https://openalex.org/A5082008486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多粒度信息感知延迟交互网络（MGDIN），通过分层分组和递进解耦的方式在CTR预测中缓解特征异质性导致的表示崩塌问题。

**💡 创新点**

创新点在于：①将特征按多粒度划分为信息密度更均衡的组；②采用分层掩码的递进交互策略，将低信息特征的交互推迟到网络深层，避免噪声提前干扰；③结合注意力得分预先排序，实现动态的交互解锁。

**🔧 技术方法**

技术手段包括：多粒度特征分组（窗口分块）、信息感知的递进注意力（分层掩码）、残差连接与层归一化、Per‑Token Feed‑Forward Networks以及多窗口并行计算。

**📊 数据集**

使用了公开的 Amazon Sports 评分数据集和工业级 70 亿条用户交互记录的电商广告数据集。

**📈 对比分析**

与 FM、DNN、Wide&Deep、DeepFM、DCN、AutoInt、GDCN、MaskNet、RankMixer、OneTrans 等先进基线对比，MGDIN 在 Amazon Sports 上 AUC 提升约0.54%，在工业数据上 AUC 提升约0.54%，并在 10 天 A/B 测试中实现 3.04% 的 CTR 提升，且无额外推理延迟。

**⚠️ 局限性**

局限性包括：需要为不同粒度设置窗口大小与层级稀疏比例，需额外的超参数调优；对极端稀疏特征的分组可能仍存在误差；在极大规模数据上并行计算仍可能受到内存与 GPU 带宽的限制。

---

## 412. Generalist Large Language Models for Molecular Property Prediction: Distilling Knowledge from Specialist Models

**arXiv ID:** 2603.12344 | [PDF](https://arxiv.org/pdf/2603.12344v1)

**作者:** Khiem Le `[一作]` (University of Notre Dame), Hoang Thanh Lam `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出TreeKD方法，将决策树规则用自然语言形式注入LLM进行知识蒸馏；

**💡 创新点**

创新点在于：①把树模型的可解释预测规则转换成文本注入LLM；②引入rule-consistency，在推理时利用随机森林多样化规则进行加权投票；

**🔧 技术方法**

使用的技术包括：决策树（深度6）与随机森林、自然语言提示工程、LoRA微调、vLLM缓存、Self-consistency与Rule-consistency测试时扩展；

**📊 数据集**

采用TDC基准的22个ADMET属性数据集进行实验；

**📈 对比分析**

与基础LLM（Gemma-2-2B、Granite-3.3-2B）以及TxGemma-2-2B、MapLight等专家模型对比，TreeKD在19/22属性上提升显著，整体性能逼近专家模型；

**⚠️ 局限性**

主要限制是提示文本过长导致训练成本上升，推理时成本可通过缓存缓解，同时目前仅验证了22个属性，扩展到更多属性或其他药物发现任务仍待研究。

---

## 413. A Prediction-as-Perception Framework for 3D Object Detection

**arXiv ID:** 2603.12599 | [PDF](https://arxiv.org/pdf/2603.12599v1)

**作者:** Song Zhang `[一作]` (Z-one Technology Co., Ltd.), Ruibo Wang `[通讯]` (Z-one Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Prediction-As-Perception（PAP）框架，将历史预测结果作为查询输入到当前帧的感知模块，以提升3D物体感知的准确性。

**💡 创新点**

创新点在于将人脑预测感知机制迁移到深度学习模型中，通过预测结果驱动感知、感知结果反馈预测，实现双向循环交互，显著增强时空一致性。

**🔧 技术方法**

采用基于Transformer的注意力机制与查询嵌入技术，将现有感知模型（如UniAD）的感知与预测模块耦合，使用跨帧查询更新来实现模块间的交互。

**📊 数据集**

在nuScenes数据集上进行实验，利用该数据集提供的多视角摄像头和标注信息。

**📈 对比分析**

将原始UniAD与集成PAP后的UniAD+PAP进行对比，评估AMOTA、AMOTP、Recall、IDS、训练时长和FPS，结果显示PAP将AMOTA提升10%、推理速度提升15%，并降低IDS。

**⚠️ 局限性**

该框架高度依赖原始感知与预测模型的性能，实验仅在UniAD上验证，未探索更先进模型及完整的消融分析。

---

## 414. How Fair is Software Fairness Testing?

**arXiv ID:** 2603.12511 | [PDF](https://arxiv.org/pdf/2603.12511v1)

**作者:** Ann Barcomb `[一作]` (University of Calgary), Mairieli Wessel `[通讯]` (Radboud University)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5032291051)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨软件公平性测试的文化、社会与生态维度，提出去殖民化、公平性数据集与环境考量的研究框架；

**💡 创新点**

创新点在于将去殖民化视角与公平性测试结合，强调多元文化参与、环境与数据殖民的交叉影响，并给出具体研究方向；

**🔧 技术方法**

采用公平性评估方法、数据集构造技术、参与式共创、模型跨文化适配技术（如RAG、提示）等；

**📊 数据集**

主要使用现有西方主导的基准数据集，讨论了KorNAT、ChatBlackGPT等潜在非西方数据集；

**📈 对比分析**

论文未给出实验对比，主要提出未来可通过多文化数据集与跨模型评估来衡量公平性，预期能显著揭示现行指标的盲点；

**⚠️ 局限性**

局限性包括缺乏实证验证、跨文化数据获取与治理难度、资源与成本限制，以及标准统一性的挑战。

---

## 415. No More DeLuLu: Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation

**arXiv ID:** 2603.12276 | [PDF](https://arxiv.org/pdf/2603.12276v1)

**作者:** Taha Bouhsine `[一作]` `[通讯]` (Azetta AI), Taha Bouhsine (Azetta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 -product 作为新的核运算，并基于此构建 Neural Matter Networks (NMNs)，从而在不使用激活函数和归一化层的情况下实现网络非线性。

**💡 创新点**

创新点在于将点积的对齐信息与逆平方距离的近似度结合为一个 Mercer 核，具有自正则化、Lipschitz 连续性、无穷可微等性质，同时通过前向形式直接在特征空间中计算，避免 Gram 矩阵。

**🔧 技术方法**

技术包括 -product 核定义、理论证明（Mercer、RKHS、NTK 等）、NMN 层设计、Yat‑Multi‑Head Attention 取代传统缩放点积注意力、以及在 GPT‑2 结构中替换 MLP 和 LayerNorm。

**📊 数据集**

使用的数据集有 MNIST、Eurlex‑4K 极端分类数据、以及 2.5B 语料 FineWeb 训练语言模型。

**📈 对比分析**

通过与线性基线、GPT‑2、以及传统极端分类基线比较，Aether‑GPT2 在验证损失上下降约1.45%，同时在极端分类任务上提升 P@1~5 指标，且保持相近吞吐量与显存利用。

**⚠️ 局限性**

局限性包括对 ε 超参数敏感，且在某些任务中对梯度噪声的鲁棒性尚未充分验证，此外在需要高维全局相互作用的场景下可能仍需额外归一化或正则化。

---

## 416. ARL-Tangram: Unleash the Resource Efficiency in Agentic Reinforcement Learning

**arXiv ID:** 2603.13019 | [PDF](https://arxiv.org/pdf/2603.13019v1)

**作者:** Bangjun Xiao `[一作]`, Fuli Luo `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了统一的行动级资源管理系统 ARL‑Tangram，用于高效调度 Agentic RL 任务中的外部资源，显著降低资源浪费并加速训练。

**💡 创新点**

创新点包括：①将资源管理粒度从轨迹/任务级降低到单个行动级；②提出统一的向量化资源成本与弹性建模；③设计贪心淘汰调度算法和 AO/EO 资源管理器，支持 CPU/GPU 等异构资源的细粒度共享与弹性分配。

**🔧 技术方法**

核心技术：向量化资源成本表述、弹性函数建模、贪心淘汰调度算法、CPU 的 Allocate‑On‑Execution (AOE)、GPU 的 Evict‑On‑Execution (EOE)、NUMA 亲和性、LRU 缓存策略、DP 资源分配等。

**📊 数据集**

实验使用真实生产任务：AI Coding（SWEBench 相关数据集）、DeepSearch（BrowseComp）以及 MOPD 任务集，涵盖代码执行、API 调用和奖励服务等多种外部资源。

**📈 对比分析**

与基线（Kubernetes + SGLang + ServerlessLLM）对比，ARL‑Tangram 在平均 ACT 上降低至 1/4，训练步时长提升 1.4–1.5 倍，外部资源利用率提升 71%，同时实现了 4.3× 的 ACT 缩短和 1.5× 的步时长缩短。

**⚠️ 局限性**

局限性：①弹性调度依赖对行动执行时间和弹性函数的预先建模；②GPU 侧恢复开销仍占约 25% 的执行时间；③在极高并发（大规模批量）下，系统的准确性、延迟稳定性及可扩展性仍需进一步验证。

---

## 417. A Requirement-Based Framework for Engineering Adaptive Authentication

**arXiv ID:** 2603.12968 | [PDF](https://arxiv.org/pdf/2603.12968v1)

**作者:** Alzubair Hassan `[一作]` (University College Dublin), Liliana Pasquale `[通讯]` (University College Dublin)

**通讯引用:** 1675 | [OpenAlex ID](https://openalex.org/A5049622460)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个基于上下文目标模型与扩展特征模型的自适应认证框架，利用模糊因果网络与Z3 SMT求解器在运行时动态选择最优认证方法，以降低安全风险并同时满足安全、可用性与性能目标。

**💡 创新点**

创新之处在于将上下文目标模型与扩展特征模型结合，并通过模糊因果网络与SMT求解实现对风险与多目标权重的统一评估，支持在动态环境下快速决策且兼顾风险降低与目标满足。

**🔧 技术方法**

使用的技术包括上下文目标模型、扩展特征模型、模糊因果网络（FCN）、SMT求解器Z3、MAPE‑K自适应循环与基于约束的决策。

**📊 数据集**

实验数据来自于手工构建的车联网（IoV）与医疗系统两大场景的情境与攻击列表，并未使用公开的大规模数据集。

**📈 对比分析**

与统一认证方法的基线相比，框架在三种IoV场景与三种医疗场景中实现了更高的安全风险降低与目标满足度；平均求解时间约 1.5 秒，内存占用低于 20 MB，足以支持运行时部署。

**⚠️ 局限性**

局限性包括对专家判断的高度依赖、缺乏实证数据校准影响权重、仅验证两大领域、Z3求解在大规模模型下可能退化，以及未考虑用户偏好与动态学习等因素。

---

## 418. The Perfection Paradox: From Architect to Curator in AI-Assisted API Design

**arXiv ID:** 2603.12475 | [PDF](https://arxiv.org/pdf/2603.12475v1)

**作者:** Mak Ahmad `[一作]` (Google), David Karger `[通讯]` (MIT)

**通讯引用:** 51923 | [OpenAlex ID](https://openalex.org/A5028448267)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了基于AIP训练的GPT‑4o在API设计中的效果，并将AI生成的API与人类设计的API进行对比

**💡 创新点**

提出了“Perfection Paradox”，揭示AI超高一致性导致专家误判其为人类作品，并将设计者角色从“drafter”转变为“curator”，强调人机协作的新模式

**🔧 技术方法**

使用Fine‑tuned GPT‑4o模型，并采用Steven Clarke的Cognitive Dimensions of Notations框架进行评估

**📊 数据集**

利用Google的完整AIP规范作为训练数据，以及由16名行业专家评估的三份API规范（1份AI生成、2份人类生成）

**📈 对比分析**

通过双盲评估和Wilcoxon检验比较AI与人类API在11个可用性维度的表现，AI在10/11维度显著优于人类，作者识别率仅19%，并将生成时间从2小时压缩至15分钟

**⚠️ 局限性**

样本仅包含单一任务场景，参与者人数有限，且仅使用Google AIP框架，结果可能不具普适性，且AI在领域推理和隐含需求方面仍有局限

---

## 419. Weighted Set Multi-Cover on Bounded Universe and Applications in Package Recommendation

**arXiv ID:** 2603.12528 | [PDF](https://arxiv.org/pdf/2603.12528v1)

**作者:** Nima Shahbazi `[一作]` (University of Illinois Chicago), Stavros Sintos `[通讯]` (University of Illinois Chicago)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5015883931)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在宇宙大小为常数的加权集合多覆盖问题，给出了精确动态规划算法、2-近似及更快的 (2+ε)-近似算法。

**💡 创新点**

创新点在于将问题转化为对凸分段线性函数求和的线性规划，设计了新的取整与补齐技术实现 2-近似，并通过函数逼近压缩 LP 规模得到 (2+ε) 近似，从而首次获得与宇宙大小无关的常数因子近似。

**🔧 技术方法**

采用了动态规划、凸分段线性函数逼近、线性规划求解、取整与穷举补齐、以及复杂度分析等技术。

**📊 数据集**

实验使用了四个真实数据集：1990 年美国人口普查样本、Amazon Music 语种集、Stack Overflow 2024 调查数据以及 Yelp 商户分类数据。

**📈 对比分析**

与贪心和随机化 LP 两种基线对比，实验表明我们的算法在总权重上均优于基线，并且 (2+ε)-近似算法在运行时间上与贪心相近但质量更好；DP 在小规模下能够得到最优解。

**⚠️ 局限性**

局限性包括：仍无法达到 (1+ε) 近似；算法仅适用于宇宙大小常数的情况；DP 在大规模数据上不可扩展；在某些极端权重分布下仍可能落后于随机化 LP。

---

## 420. Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments

**arXiv ID:** 2603.12940 | [PDF](https://arxiv.org/pdf/2603.12940v1)

**作者:** Anees Peringal `[一作]` (Khalifa University), Federico Renda `[通讯]` (Khalifa University)

**通讯引用:** 4429 | [OpenAlex ID](https://openalex.org/A5043823023)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种针对由可变形线性物体（DLO）与刚性链条组成的混合系统（hDLO）在受限环境中进行协同操作的轨迹规划方法。

**💡 创新点**

创新点在于：1）将可变形链条与刚性关节统一到几何可变应变（GVS）模型中，实现对整个闭环体系的高维状态空间的低维表示；2）采用基于梯度的准静态轨迹优化，并利用解析微分大幅提升求解速度（对 IKS 问题实现 33× 的加速）；3）首次将该方法与采样式 BiRRT 进行公平对比，展示在受限环境下的优越性。

**🔧 技术方法**

核心技术包括：几何可变应变（GVS）Cosserat 杆模型、解析梯度计算、直接离散化（Direct Transcription）求解准静态非线性规划、双臂机器人对 hDLO 的闭环动力学建模，以及 SE(3) 插值与闭链约束的 Lie 代数实现。

**📊 数据集**

数据来源主要是：1）仿真中构造的多种 hDLO 组装（如两 DLO + 一刚性、三 DLO + 两刚性等）与对应材料参数；2）实验平台使用的双臂机器人与由 0.68 m Nitinol 线材与 3D 打印 PLA 刚性块构成的 hDLO，并通过 Optitrack 进行标定与轨迹采集。未使用公开数据集。

**📈 对比分析**

与采样式 BiRRT 的对比表明：①在准静态轨迹优化（TO）中，仅需约 30–55 s 的求解时间即可获得满足约束的最优路径；②BiRRT 在相同任务下求解时间均在 10–200 s 之间，且在需要显著偏转的轨迹上往往更慢；③TO 产生的轨迹更平滑，关节幅值与执行力度更小，符合实验测得的误差约 2–3 cm（占单条 Nitinol 长度约 5%）。

**⚠️ 局限性**

局限性包括：1）仅在准静态假设下工作，未考虑动力学与惯性效应；2）采用开放式控制，易受参数误差与外部扰动影响；3）对环境约束仅建模为圆形孔洞，未处理更复杂多体接触；4）尚未实现在线反馈控制或自适应校准。

---

## 421. ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation

**arXiv ID:** 2603.13154 | [PDF](https://arxiv.org/pdf/2603.13154v1)

**作者:** Siqi Sun `[一作]` (University of Sheffield), Xingyi Song `[通讯]` (University of Sheffield)

**通讯引用:** 876 | [OpenAlex ID](https://openalex.org/A5019126151)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ESG-Bench 基准数据集，用于评估和减轻 ESG 报告问答中的幻觉问题，并通过多步骤链式思维（CoT）和监督微调提升大语言模型的可信度

**💡 创新点**

首次构建包含真实 ESG 报告、人工标注的问答对以及幻觉分类的长文本 QA 数据集，并将 GPT-4o 的自评信号作为代理监督实现无标签幻觉消除

**🔧 技术方法**

使用大语言模型（如 Llama‑3.2‑3B、Gemma‑2‑2B‑it、Mistral‑7B‑Instruct‑v0.3）进行监督微调、CoT 生成和 CoT 微调；采用多步 CoT 结构化推理；利用 GPT‑4o 进行自评生成对齐标签

**📊 数据集**

ESG‑Bench（包含 94 篇 ESG 报告、270 组 QA 对、1,358 条正面答案和 25,516 条幻觉答案）、BioASQ、生物医学 QA 数据集以及 HaluEval 进行跨域评估

**📈 对比分析**

对比了无微调、监督微调、两步 CoT 微调和四步 CoT 微调，在 ESG‑Bench、HaluEval、BioASQ 上均显著提升 WA、WoA 准确率和 F1，四步 CoT 在所有指标上表现最优，尤其在避免幻觉方面显著优于传统方法

**⚠️ 局限性**

局限性包括：依赖 GPT‑4o 生成的自评信号，可能引入偏差；数据集覆盖的 ESG 报告数量有限，难以覆盖所有行业与地区；对多模态（表格、图表）处理仍不完善，且在极长文本（>40k tokens）下性能尚待进一步验证

---

## 422. Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**arXiv ID:** 2603.13185 | [PDF](https://arxiv.org/pdf/2603.13185v1)

**作者:** Rohith Peddi `[一作]` (University of Texas at Dallas), Vibhav Gogate `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1537 | [OpenAlex ID](https://openalex.org/A5038455119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了世界场景图生成（World Scene Graph Generation, WSGG）任务，能够在单目视频中对可见与不可见（被遮挡或离开视野）物体进行 3D 定位和全时序关系预测，并基于此构建了 ActionGenome4D 数据集；提出了三种方法（Persistent World Graph、Masked World Auto‑Encoder、4D Scene Transformer）来解决不可见物体推理。

**💡 创新点**

创新点包括：
1) 将场景图从帧中心、2D 表述提升到全局 3D 世界坐标、时间一致的“世界场景图”；
2) 通过 3D 重建+几何标注、VLM 伪标签和人工校正实现对所有物体（包括暂未可见物体）的密集关系标注；
3) 设计三种不同的不可见物体推理偏置：基于对象永续记忆缓冲、基于掩码自编码恢复、以及可微分的 4D 时空 Transformer；
4) 提供统一的模型组件（全局结构编码、空间 GNN、摄像头姿态/运动编码、关系预测器、时序边注意力），实现跨方法可比性。

**🔧 技术方法**

核心技术包括：
- 3D 视觉几何重建网络（如 3D reconstruction model）、Bundle Adjustment 校正相机位姿；
- 目标检测（GDINO）、分割（SAM2）、姿态估计（PromptHMR）用于生成 3D OBB；
- 视觉 Transformer（DINOv2/3）提取 ROI 特征；
- 关联检索与跨视图自编码（Masked World Auto‑Encoder）；
- 双向 Transformer 编码全时序对象序列（4D Scene Transformer）；
- 视觉语言模型 VLM（Qwen‑2.5‑VL、InternVL‑2.5、Kimi‑VL）与 Graph‑RAG 推理框架。

**📊 数据集**

使用的数据集：
- ActionGenome4D：对 Action Genome 进行 4D 转换，包含每帧 3D OBB、相机位姿、完整关系标签（包括不可见物体）；
- 原始 Action Genome 视频、相机位姿、SMPL 模型等作为中间处理素材。

**📈 对比分析**

与基线对比：
- 在 PredCls 和 SGDet 任务（含 With/No Constraint）下，4D Scene Transformer 在 R@10、R@20、R@50 等指标上取得最高分（例如 DINOv3‑L 版本 R@10=71.95%）；
- Persistent World Graph 也表现稳健，特别是在单标签预测上；
- Masked World Auto‑Encoder 在多标签（No Constraint）模式下表现最好，尤其是预测关系的多标签 recall；
- VLM 评估表明 Qwen‑2.5‑VL 在微平均 F1 上最高，Graph‑RAG 方法比单纯字幕更优。

**⚠️ 局限性**

局限与挑战：
- Persistent World Graph 的记忆缓冲为非可微，限制了端到端学习；
- VLM 伪标签易受偏差，尤其是长尾关系的低召回；
- 数据集仍以单目视频为主，缺乏多视角或 RGB‑D 视角；
- 物体检测仍是瓶颈，SGDet 模型的 recall 明显低于 PredCls；
- 需要人工校正的标注流程耗时，难以大规模扩展；
- 对实时在线推理的适配和更广泛的开放词表仍待解决。

---

