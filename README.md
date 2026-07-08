# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-08 | 今日论文总数: 508

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. NEMESIS: NEtlist-Driven Modeling and Equation Synthesis with Inversion-Aware SPICE Anchoring

**arXiv ID:** 2607.05657 | [PDF](https://arxiv.org/pdf/2607.05657v1)

**作者:** Subhadip Ghosh `[一作]` (University of Minnesota), Sachin S. Sapatnekar `[通讯]` (University of Minnesota)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过多模态LLM生成OTA性能方程并利用SPICE驱动的迭代修复实现精确建模

**💡 创新点**

创新性地将LLM与原型识别、检索增强、基于SPICE的修复循环相结合，生成可解释且高精度的设备级方程

**🔧 技术方法**

使用GPT‑5.2多模态LLM、图匹配检索、WL哈希、g_m/I_D lookup表以及SPICE验证

**📊 数据集**

基准数据集为5种不同复杂度的OTA拓扑（easycOTA‑1到hardcOTA‑5），在65nm PDK下测试

**📈 对比分析**

相较于完整SPICE评估，平均相对误差低于7%，速度提升约4000–5200×；在指定g_m/I_D范围内保持误差<7%

**⚠️ 局限性**

仅适用于固定拓扑，无法替代最终SPICE signoff，且未覆盖大信号、噪声、高频匹配等高级效应

---

## 2. Music I Care About: Automated Multimodal Benchmarking of LLM Music Perception Skills on (Almost) Any Music

**arXiv ID:** 2607.06015 | [PDF](https://arxiv.org/pdf/2607.06015v1)

**作者:** Tomáš Sourada `[一作]` (Charles University), Jan Hajič `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了可根据用户提供的音乐数据自动生成多模态音乐感知问答基准的框架 MusICA-MetaBench，并在 ChoraleBricks 与 ChoralSynth 数据集上进行评估。

**💡 创新点**

①可即时生成基准，避免大规模静态基准的资源与版权问题；②跨音频、乐谱图像、符号文件三模态同一问题实现可比性；③通过统计校准确定最小基准规模；④验证模型确实需要音乐感知。

**🔧 技术方法**

基于 MusicXML 提取真值与干扰项的程序化问答生成；多模态多项选择评估；样本规模校准与 t 检验；LLM 对话接口与自动评测脚本。

**📊 数据集**

ChoraleBricks（10 首多轨吹管合唱）和 ChoralSynth（20 首合成合唱）。

**📈 对比分析**

对比 8 种主流 MLLM（如 Gemini 3.1 Pro、Qwen3 Omni 30B 等）在 s=20（300 题）基准上评估准确率、时长与成本；Gemini 3.1 Pro 明显领先，整体准确率约 59%，其余模型在不同模态上表现相近，音频与符号模态差异最大。

**⚠️ 局限性**

干扰项难度可能偏低，无法完全排除文本推理；仅覆盖西方调性音乐，缺乏跨文化适用；未映射至正式评分体系；对新数据集需要重新实现真值提取。

---

## 3. UCSC NLP at SemEval-2026 Task 10: Boundary-Aware Span Extraction and RoBERTa Classification for Conspiracy Detection

**arXiv ID:** 2607.05689 | [PDF](https://arxiv.org/pdf/2607.05689v1)

**作者:** Dom Marhoefer `[一作]` (University of California Santa Cruz), Ryan King `[通讯]` (University of California Santa Cruz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出两套独立的 RoBERTa-large 模型，分别用于 PsyCoMark 任务中的角色标记抽取与文档级阴谋检测。

**💡 创新点**

创新点包括将标记抽取视为多标签 span 分类，采用 IoU≥0.95 的高精度标注、硬负样本挖掘以及基于包含关系的 NMS 来提高边界精确度；文档分类则引入标签平滑并采用分层训练拆分。

**🔧 技术方法**

技术上使用 RoBERTa-large 编码器、span 级两层 MLP 输出、BCE 多标签损失、硬负样本采样、NMS 后合并，以及文档分类的 Tanh+Dropout 线性头和交叉熵 + 标签平滑。

**📊 数据集**

使用的训练数据为 PsyCoMark 语料库，包含约 4,100 篇来自 190 个子版块的 Reddit 文本，按 90/10 分层划分为训练与验证集。

**📈 对比分析**

在官方评测中，标记抽取任务获得 0.2251 的 macro F1 并排名第 7，文档级检测任务取得 0.7694 的 weighted F1 并排名第 12，表现优于多数参赛系统。

**⚠️ 局限性**

局限性主要是对抽象角色（Action、Effect、Evidence）的边界敏感度高、召回率低，span 长度上限 32 词可能漏检长标记，以及验证与官方评测指标的差异导致性能偏差。

---

## 4. Mitigating Factual Hallucination in Large Reasoning Models via Mixed-Mode Advantage Regularization

**arXiv ID:** 2607.05861 | [PDF](https://arxiv.org/pdf/2607.05861v1)

**作者:** Kaishen Wang `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出混合模式优势正则化（Mixed‑Mode Advantage Regularization）来调节大推理模型（LRMs）在事实性问答中的思考行为，抑制思考诱发的幻觉并提升答案真实性。

**💡 创新点**

将思考视为对直接答案的残差，并在同一问题下同时采样思考与非思考轨迹，以非思考轨迹为同模型参考构造混合滚动组，从而实现对思考残差价值的相对评估。

**🔧 技术方法**

采用强化学习框架 GRPO、KL 正则化以及基于 Qwen3‑32B 的自动判定器评估奖励（真/假、格式），结合混合模式优势正则化实现训练。

**📊 数据集**

在 SimpleQA、SimpleQA‑Verified、TriviaQA、NQ‑Open、PopQA、HotpotQA 六个事实性问答基准上训练与评估；训练数据从 TriviaQA 训练集筛选出思考与非思考之间显著事实性差距的样本。

**📈 对比分析**

与固定思考/非思考、适应性模式选择、SFT、单模式 RL 等基线比较，4B 版平均准确率由 22.81% 提升至 25.49%，8B 版从 27.59% 提升至 29.57%；在数学推理基准上保持甚至略有提升，证明方法兼顾事实性与推理能力。

**⚠️ 局限性**

局限在于需人工挑选训练样本、对模型规模与任务的泛化仍待进一步验证；思考与非思考模式的区分仍依赖提示模板，可能影响可解释性与自适应性。

---

## 5. MSA-DCNN: A Data-Efficient Multi-Scale Deformable CNN for Medical Image Classification

**arXiv ID:** 2607.06083 | [PDF](https://arxiv.org/pdf/2607.06083v1)

**作者:** Hamza Hussaini `[一作]` (Robert Gordon University), Carlos Francisco Moreno-García `[通讯]` (Robert Gordon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了MSA-DCNN，一种结合可变形卷积、尺度特定注意力、跨尺度学习注意力融合以及自蒸馏的多尺度医学图像分类网络。

**💡 创新点**

创新点在于将可变形采样、尺度内特征重调和跨尺度融合以及自蒸馏统一到一个端到端学习框架，实现对多尺度结构的自适应建模和标签稀缺条件下的高效特征对齐。

**🔧 技术方法**

采用可变形卷积（Deformable Conv）、多尺度卷积块、MCBAM（子块池化+通道/空间注意力）、跨尺度注意力机制、辅助自蒸馏损失以及交叉熵/焦点损失等技术。

**📊 数据集**

在四个医学图像数据集上评估：C‑NMC（白细胞涂片）、PBC（外周血细胞）、ISIC‑2020（皮肤镜图像）以及独立的ALL外部hold‑out涂片集。

**📈 对比分析**

与基础CNN、Transformer（DeiT‑S、DINOv2‑S）、DGConv以及半监督TS‑MS等方法对比，MSA‑DCNN在AUC、准确率和F1分数上均显著提升，参数量更少，证明了其在分布偏移和标签稀缺场景下的优越性能。

**⚠️ 局限性**

局限性包括在更高分辨率下的计算量略增、外部hold‑out集上仍出现轻微性能下降，以及对其他模态（如CT、MRI）的泛化能力尚未系统验证。

---

## 6. Neuromorphic Silicon Neuron Controller for Adaptive Deep Brain Stimulation in Parkinson's Disease

**arXiv ID:** 2607.05453 | [PDF](https://arxiv.org/pdf/2607.05453v1)

**作者:** Md Abu Bakr Siddique `[一作]` (Michigan Technological University), Hongyu An `[通讯]` (Michigan Technological University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种基于可冷却硅神经元的自适应深脑刺激（SiLIF‑DBS）控制器，用于帕金森病的闭环治疗。

**💡 创新点**

创新点在于将可冷却泄漏积分‑发放（SiLIF）硅神经元直接作为控制器原语，并构建与之对应的计算模型，实现硬件–软件协同设计和在生物医学仿真框架中的闭环验证；同时通过单一可调的再发放偏置实现低功耗与抑制效率的内在权衡。

**🔧 技术方法**

采用的技术包括：CMOS混合信号模拟电路（130 nm SkyWater PDK）实现的泄漏积分‑发放神经元；计算模型推导的等效连续时间公式；Beta平均绝对值（Beta_ARV）作为生物标记的滤波与阈值提取；以及帕金森病皮层-基底节-丘脑网络（CBG）仿真模型用于闭环评估。

**📊 数据集**

使用的“数据集”主要是基于帕金森病的计算模型所产生的仿真 STN‑LFP 信号（未使用真实临床数据）。

**📈 对比分析**

评估方法为将 SiLIF‑DBS 与开放式、开/关、双阈值等现有 aDBS 控制器在相同仿真环境下进行比较，指标包括相对功耗（25% 的开环功耗）和抑制效率（5.85 %/μW 的 beta 抑制/能耗），结果显示 SiLIF‑DBS 在保持较低能耗的同时，抑制效率略优于双阈值控制器，显著优于开放式刺激。

**⚠️ 局限性**

局限性包括：目前仅在仿真环境下验证，缺乏真实植入设备的体内或体外实验；对不同患者的个体化调参仍需进一步研究；以及在复杂生理噪声和长期使用稳定性方面的未知风险。

---

## 7. Decision-Focused Scenario Generation and Selection for Efficient and Robust Grid Dispatch

**arXiv ID:** 2607.05830 | [PDF](https://arxiv.org/pdf/2607.05830v1)

**作者:** Yangze Zhou `[一作]`, Yi Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在分布式鲁棒优化 (DRO) 下，提出一种统一的决策导向场景生成框架，能够通过 VAE、GAN、扩散模型等主流生成模型学习负荷和可再生能源等不确定量在不同母线之间的联合分布，并结合可微分场景选择器，实现对生成场景的决策导向优化。

**💡 创新点**

创新点包括：① 统一决策导向训练方法，兼容显式密度模型、隐式密度模型和连续时间生成模型；② 通过场景选择器实现可微分的决策导向场景筛选，显著降低了后向传播中的计算负担；③ 对联合分布的建模显著提升了场景的空间相关性，并在 DRO 框架中直接优化操作成本。

**🔧 技术方法**

技术手段包括：生成对抗网络（GAN）、变分自编码器（VAE）、扩散模型（DDPM），以及基于 Wasserstein 范数的 DRO 双阶段优化；利用 OptNet 对线性规划的隐式梯度进行求导；使用 Gumbel‑Softmax 重参数化实现可微分的场景选择；以及温度调度、熵正则等技术提升选择器的多样性。

**📊 数据集**

实验数据来自 IEEE 14 节点系统的南方城市负荷数据（2022‑2023 年），并对系统进行比例缩放以保证可操作性，使用 80% 训练/20% 验证划分，并在 2023 年的数据上评估。

**📈 对比分析**

与基准（传统参数化/非参数化预测、随机、K‑means、K‑medoids、层次聚类等）相比，决策导向生成+场景选择在 8 种预测设置下平均降低操作成本 0.80%–2.02%；在联合分布场景下的成本下降更为显著；同时，场景选择器的可微分实现显著提高了训练效率。

**⚠️ 局限性**

局限性：① 训练过程对内存和计算量敏感，尤其在大规模系统或高维场景数时可能超出显存；② 需要手动设定 Wasserstein 半径和正则化权重，影响鲁棒性与保守度平衡；③ 仅验证了两阶段铜板模型与 IEEE 14 节点系统，未覆盖更复杂多阶段或更大规模网络；④ 生成模型的训练和选择器的调参仍依赖经验，尚缺乏自动化流程。

---

## 8. Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction

**arXiv ID:** 2607.05577 | [PDF](https://arxiv.org/pdf/2607.05577v1)

**作者:** Mohammad Saifullah `[一作]` (PocketFM), Aanand Kumar Yadav `[通讯]` (PocketFM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了Narrative World Model（NWM）系统，用于写作场景下的多跳叙事状态问答；

**💡 创新点**

首次将叙事学结构（聚焦/认知边界、事件与透露顺序、戏剧功能、承诺兑现）与类型化时间知识图、查询条件混合检索相结合，显著提升多跳问答效果；

**🔧 技术方法**

采用Sonnet 4.5提取器生成叙事化记忆记录，构建时序知识图，结合BM25+向量+一跳图检索，并用递归语言模型QA验证答案，最终用Opus 4.8恒定读者进行评测；

**📊 数据集**

使用12本公共领域小说（共约6个体裁）和5本私有生产级连载小说，构造176条多跳验证题和576条公开问答（其中110条为多跳子集）；

**📈 对比分析**

采用持恒读者和统一的评测协议，对比无记忆、简单检索、RAG、GraphRAG、Graphiti及NWM不同检索条件；NWM Graph Retrieval在私有176题上0.898、公共576题上0.625，均显著优于Graphiti（0.574/0.516）及GraphRAG/RAG；

**⚠️ 局限性**

主要局限包括：未验证生成阶段的表现；受限于叙事学标签与检索方式的耦合；公共多跳子集样本量有限；实验依赖特定提取器与检索配置，通用性待进一步验证。

---

## 9. Flow Matching-Based Speech Source Separation with Best-of-N Biometric Sampling

**arXiv ID:** 2607.06088 | [PDF](https://arxiv.org/pdf/2607.06088v1)

**作者:** Anastasia Zorkina `[一作]` (ITMO University), Yuriy Matveev `[通讯]` (ITMO University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于条件流匹配的单通道语音源分离方法。

**💡 创新点**

创新点包括利用冻结的说话人编码器对源顺序进行约束，使用最佳‑of‑N生物识别采样进行候选选择，以及块级生成与说话人嵌入对齐以支持长音频处理。

**🔧 技术方法**

采用条件流匹配框架、Transformer U‑Net（TUnet）与NCSN++网络作为速度估计器，结合Wav2Vec 2.0说话人嵌入和Whisper V3/Wav2Vec 2.0下游评测。

**📊 数据集**

使用Libri2Mix（Libri2Mix）语料库，并在 mix_clean 与 mix_both 两种混音类型上进行实验。

**📈 对比分析**

与 DiffSep、SepReformer、MeanFlow‑TSE 等基线比较，TUnet 在 SI‑SDR、PESQ、ESTOI 及 downstream cpWER/EER 上表现最优，最佳‑of‑N N=4 时已接近 oracle。

**⚠️ 局限性**

局限性包括生成模型仍存在采样方差，对长录音的块级对齐仍需依赖说话人嵌入，且在极端 SNR 或重叠情况下性能可能下降。

---

## 10. Empirical Minimal-Realisation Compression of Deep Neural Networks via Controllability-Observability Tests

**arXiv ID:** 2607.05457 | [PDF](https://arxiv.org/pdf/2607.05457v1)

**作者:** Anis Hamadouche `[一作]` (Heriot-Watt University), Amir Hussain `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于控制理论的可达性与可观性测试，估计并压缩深度神经网络的隐藏状态维度。

**💡 创新点**

创新点在于将可达性、可观性与平衡截断相结合，直接从训练网络的隐藏状态和输出雅可比矩阵推断最小有效状态秩，并将该秩作为压缩后网络层宽度的实用设计准则。

**🔧 技术方法**

利用经验可达性Gramian、可观性Gramian和平衡矩阵计算层级秩；采用深度前向传播收集隐藏状态快照、计算输出对隐藏状态的雅可比；再将所得秩用于构建压缩网络并可选知识蒸馏训练。

**📊 数据集**

在MNIST（784维）和CIFAR‑10（3072维）数据集上评估，使用全连接SiLU网络，分别包含4层（MNIST）和4层（CIFAR‑10）。

**📈 对比分析**

与结构化与非结构化剪枝、低秩SVD、INT8量化、线性分类器和蒸馏等基线对比。实验显示：MNIST上隐藏状态从1024压缩至277（73%）并把参数从400k压缩至106k（73%），准确率仅下降1.15%；CIFAR‑10上隐藏状态从4608压缩至1339（71%），参数从≈10M压缩至≈1.7M（83%），准确率保持不变并将CUDA推理时延缩小约3倍。

**⚠️ 局限性**

主要局限：秩估计依赖有限样本，易受分布漂移影响；可观性矩阵计算成本高；缺乏非线性系统的严格误差上界；压缩需重新训练；实验仅针对全连接网络，尚需推广到CNN、ResNet、Transformer等结构；未考虑硬件特定的延迟或能耗约束。

---

## 11. Shifting is Optimal under Gap-ETH: A Lower Bound Framework for Geometric Approximation Schemes

**arXiv ID:** 2607.06069 | [PDF](https://arxiv.org/pdf/2607.06069v1)

**作者:** Manuel Cáceres `[一作]`, Saeed Odak `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文基于 Gap‑ETH 假设，证明了在所有常数维度 d≥3 下，单位球图（及单位立方体图）中利用 Hochbaum‑Maass 移位技术得到的 n^O(1/ε^{d-1}) 运行时间的 PTAS 已达到最优。

**💡 创新点**

创新点在于构造了一个通用的归约框架，利用 Cube Wiring 定理与 Marx‑Sidiropoulos 的 CSP 归约，将 Max‑3SAT 的 Gap 推至几何 CSP，并进一步得到多种几何优化问题的匹配 n^Ω(1/ε^{d-1}) 下限。

**🔧 技术方法**

核心技术包括强化的 Cube Wiring 定理、参数化 CSP 归约、Gap‑ETH 的错误传播分析以及对几何 CSP 的最大化版本的细化。

**📊 数据集**

研究纯粹是理论性的，没有使用任何实验数据集。

**📈 对比分析**

与现有的移位技术上界 n^O(1/ε^{d-1}) 对比，本文给出了同样形式的下界，证明了在 Gap‑ETH 下算法的运行时间无法进一步改进。

**⚠️ 局限性**

局限性在于结论仅在 Gap‑ETH 假设下成立，并且框架主要针对单位球或单位立方体图，无法直接推广到更一般的几何图或不满足网格结构的实例。

---

## 12. The Surplus Parking Gathering Problem in Infinite Grids

**arXiv ID:** 2607.05983 | [PDF](https://arxiv.org/pdf/2607.05983v1)

**作者:** Animesh Maiti `[一作]` (Indian Institute of Technology Jodhpur), Subhash Bhagat `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在无限格子上提出了 Surplus Parking Gathering Problem (SPG)：在给定停车节点（每个节点有固定容量）和机器人数量超出总容量的前提下，要求所有机器人先按容量填满停车节点，然后所有剩余机器人成功聚集到同一格子。

**💡 创新点**

核心创新点包括：①首次将停车与聚集两大经典任务在同一模型下联合研究；②通过对初始配置的对称性进行细致分类，给出 SPG 可解性的必要与充分条件；③设计了针对不同对称类的四阶段 deterministic 协调算法，保证在异步调度下无碰撞并最终实现目标配置；④给出了该算法的运动复杂度上界 O(n(a+b)+n²) 与下界 Ω(n(a+b))，在理论上逼近最优。

**🔧 技术方法**

主要技术手段包括：基于全局强多重性检测的全局可见性模型；利用最小包围矩形、扫描字符串、关键角与首角的概念构造全局一致的排序；对称性分析与分阶段设计（线性形成、多重性创建、停车饱和、聚集）相结合；对异步调度下的 pending 移动进行细致处理，确保不破坏对称性；递归的多重性与聚集节点生成机制。

**📊 数据集**

本文没有使用外部数据集，而是基于理论分析和算法描述给出证明与复杂度评估；实验验证部分在正式版本中以仿真（grid 环境）实现，但在本文摘要中未给出具体数据。

**📈 对比分析**

通过对比传统停车、聚集、模式形成等问题，本文证明在满足特定对称性与容量约束时算法能够在有限步内终止、无碰撞完成任务；运动复杂度的上界与下界给出性能评估，说明算法在大规模机器人系统下仍具备可接受的扩展性。

**⚠️ 局限性**

主要局限性：
• 对称性与容量冲突导致若干初始配置不可解，必须预先判定。
• 需要全局强多重性检测与全局可见性，现实机器人往往只能局部感知。
• 异步调度下仍需假设调度公平，且对机器人移动速度无限制。
• 只考虑无限格子模型，无法直接迁移到复杂地形或三维空间。
• 算法对 robots 的记忆和计算能力无额外假设，但在硬件实现上仍需保证足够的处理能力。

---

## 13. Optimized Adaptive Loop Filter in Versatile Video Coding

**arXiv ID:** 2607.05737 | [PDF](https://arxiv.org/pdf/2607.05737v1)

**作者:** Meng Xuewei `[一作]` (Peking University), Ma Siwei `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对VVC标准中的自适应循环滤波器(ALF)进行优化，提出了可并行执行的GALF与CCALF设计、GALF参数的自适应决策方法以及一次通过的CCALF编码方案。

**💡 创新点**

创新点包括：① 通过在编码器中使用未经过GALF处理的色度样本来实现GALF与CCALF的并行；② 利用基于量化参数的线性模型动态预测并降低GALF的最大滤波器数量，减少不必要的RDO；③ 在CCALF中预先计算自相关矩阵与交叉相关向量，并通过无滤波图像的失真估计方法，实现一次通过即可完成参数训练，显著减少了152次图片缓冲区访问。

**🔧 技术方法**

主要技术：自适应滤波器参数训练、基于量化参数的线性预测模型、滤波失真估计（不使用滤波图像）、并行编码流程设计。

**📊 数据集**

使用VVC通用测试条件(CTC)中的训练集（如MarketPlace、RitualDance、Cactus等）进行参数建模，并在相同的测试集上评估性能。

**📈 对比分析**

通过与VTM-8.0基准对比，采用BD-rate评估在AI、RA和LDB三种配置下的编码效率；结果显示在RA配置下，ALF模块的编码时间缩短约25%，并保持了几乎无量化率的损失（平均BD-Rate变化<0.2%），在AI和LDB配置下亦实现了约24%–25%的时间节省。

**⚠️ 局限性**

局限性：实验在计算机集群上进行，未充分体现多次外存访问导致的实际延迟和功耗提升；此外，优化方案主要针对RA配置，其他更高效的时延或功耗场景仍需进一步验证。

---

## 14. Design-CP: Context Parallelism for Design of Protein Nanoparticles

**arXiv ID:** 2607.05439 | [PDF](https://arxiv.org/pdf/2607.05439v1)

**作者:** Lorenzo Tarricone `[一作]` (University of Oxford), Charlotte M. Deane `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 Design-CP，一种针对 RFDiffusion 3 的上下文并行推理策略，使得大规模对称蛋白纳米粒子可在多 GPU 环境下一次性完成全原子生成；

**💡 创新点**

创新点在于将 Transformer‑style 的二次项对偶表示在推理阶段拆分为 1D 行分片和 2D 网格分片两种模式，并证明在高度对称结构中仅需对单一不对称亚基进行采样即可保持全局自洽；

**🔧 技术方法**

采用的技术包括 1D 行分片、2D Fold‑CP 轮式注意力、Ring‑Attention、DTensor 分布式参数复制、以及基于点群对称性的采样约束；

**📊 数据集**

实验使用 RFDiffusion 3 预训练模型及其在 icosahedral 与 octahedral 组装场景下的生成数据，未使用额外公开数据集；

**📈 对比分析**

与单 GPU 推理进行对比，显示 2D 网格分片在显存容量和时钟延迟上分别提升约 1.1× 与 1.9×，并成功在 2×2 HG200 GPU 与 16 × RTX A4000 GPU 上生成 12,600 与 4,272 原子的大对称纳米粒子，产生的结构在链断裂、背骨碰撞、二级结构占比等指标上与天然参考相近；

**⚠️ 局限性**

局限性在于超出 RFDiffusion 3 原始训练上下文的输入尺寸会导致分布偏移，且对高度非对称目标的生成质量仍不佳，未来需要针对更长上下文进行再训练或 fine‑tuning。

---

## 15. Position: Preventing AI-Generated CSAM Necessitates New Approaches to AI Safety

**arXiv ID:** 2607.05407 | [PDF](https://arxiv.org/pdf/2607.05407v1)

**作者:** Neil Kale `[一作]` (Carnegie Mellon University), Virginia Smith `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了人工智能系统在生成儿童色情内容（AIG‑CSAM）方面的风险，并指出现有 AI 安全技术与儿童安全的法律伦理约束不匹配，提出 15 项开放技术与政策问题，给出研究者、开发者与政策制定者的对策建议。

**💡 创新点**

创新点在于将 AI 安全研究与儿童保护的法律伦理限制结合，系统化识别并列举了 15 个针对 AIG‑CSAM 的技术与治理空白，形成了一个跨学科的行动框架，并通过概念融合实验等示例揭示了模型潜在的隐患。

**🔧 技术方法**

主要采用文献综述、案例分析与概念实验（如概念融合演示），并结合现有安全工具（如哈希匹配、NSFW 过滤、内容来源追踪、红队测试、机制解释等）进行讨论；未提出新的算法实现。

**📊 数据集**

使用的主要数据来源是公开的统计报告与 CSAM 相关哈希数据库（如 NCMEC、IWF），未直接使用真实 CSAM 数据，强调数据访问受限带来的挑战。

**📈 对比分析**

论文并未进行实验性对比或性能评估；通过对现有方法的讨论指出其局限性，并强调在缺乏 CSAM 数据与评测基准的情况下，当前技术难以提供可验证的安全性。

**⚠️ 局限性**

主要限制包括：1）研究视角主要基于美国法域，缺乏全球或跨国监管视角；2）聚焦图像生成，未覆盖文本、语音等其他模态；3）受限于 CSAM 数据的合法性与伦理风险，难以在实验中验证方案；4）缺乏可执行的性能指标和标准化评测基准。

---

## 16. GeoXplain: On-the-Fly Visual Explanations for Weather Foundation Models

**arXiv ID:** 2607.05655 | [PDF](https://arxiv.org/pdf/2607.05655v1)

**作者:** Clemens Walter Koprolin `[一作]` (ETH Zurich), Christina Humer `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出GeoXplain交互式可视化工具，用于对天气基础模型的解释结果进行可视化与分析，并实现了Aurora模型的适配器，实现多种解释方法（Saliency, IG, RISE, ViT-CX）在Aurora上的计算。

**💡 创新点**

创新点在于：1）提供模型无关的结果包格式，解耦计算与可视化；2）设计面向气象的可视化工作流，支持多变量、多层级、多时段的对比；3）将多种主流解释方法应用于Aurora，并提供实时计算支持；4）实现Notebook嵌入式交互式分析。

**🔧 技术方法**

使用Python，基于GeoXplain Viewer、Aurora Adapter；实现了Saliency、Integrated Gradients、RISE、ViT-CX解释方法；采用GPU（GH200）和SLURM作业调度进行计算；利用Matplotlib等进行渲染与校验。

**📊 数据集**

使用Microsoft Aurora 6h模型的WeatherBench‑style案例数据，覆盖多变量、不同压力层、时段；包括特定湿度、温度、海拔等输入变量。

**📈 对比分析**

通过在Notebook中调用不同方法（IG、RISE等），在同一视觉界面上对比不同变量、层级与时间步的归因结果，并与物理变量（湿度、温度、风场）叠加。性能方面，在GH200 GPU上验证IG完整度在1%以内，随机化测试能消除学习结构；ViT‑CX与长滚动推理计算耗时较长。

**⚠️ 局限性**

局限包括：1）ViT‑CX计算成本高、长滚动推理导致等待时间；2）仅支持规则密集网格，无法直接处理无结构网格、观测站、集合、Uncertainty等；3）缺乏针对不同用户体验的评估，需要进一步与气象专家验证。

---

## 17. More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges

**arXiv ID:** 2607.05904 | [PDF](https://arxiv.org/pdf/2607.05904v1)

**作者:** Chenyu Zhou `[一作]` `[通讯]` (Institute of Science Tokyo), Chenyu Zhou (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在无参考的 LLM 判别器（LLM-as-a-judge）被用作自我奖励和自我对弈（self-play）训练时，模型如何利用判别器的可疑性（plausibility）误判错误答案并被激励填充所谓的误判“盆地”。

**💡 创新点**

创新点在于：①提出隐藏 anchor 诊断框架，用跨源 exact‑match 检测来量化判别器与真实准确率之间的“judge–truth gap”；②证明验证不对称导致的 false‑positive basin，并给出其上限 ≤1‑EM；③提出 de‑anchored（先独立答复再评判）判别策略，既能检测又能在训练奖励中消除该盆地；④展示该现象跨 LLM 家族、模型规模、任务领域（数学、代码、问答）均存在。

**🔧 技术方法**

技术方法包括：自我奖励与 DPO（DPO）、隐藏 anchor 审计（cross‑source exact‑match）、基于分数阈值的 pass / reject、commit‑first（先独立答复）判别、best‑of‑N 采样、三家族（Qwen、Llama、Gemma）评估以及理论推导（bound、信息量等）。

**📊 数据集**

使用的数据集包括 GSM8K（小学数学）、TruthfulQA（开放式问答）、LiveCodeBench（自然代码生成）、AIME‑2024（竞赛数学）以及 MATH 4–5 级别等，所有数据集均提供最终答案的 exact‑match anchor。

**📈 对比分析**

比较方法通过 judge‑pass 率、anchor‑accuracy、gap、FPR、discrimination 以及 risk‑score 等指标进行。实验显示：self‑play 可将 judge‑pass 从约 0.72 提升至 0.94，却使真实准确率保持在 ~0.20，gap 达 0.74；跨家族评估证明误判盆地仍然存在；采用 de‑anchored 奖励后 FPR 降至 0，准确率保持不变，验证了其防御效果。

**⚠️ 局限性**

局限性包括：实验仅覆盖 Qwen、Gemma、Llama 等有限家族且以 DPO 为优化器；假设 solver 误差独立、任务可 exact‑match；对开放式、长文本或多模态任务的验证尚未完成；缺少对更大规模、不同训练算法（如 PPO）以及更复杂验证手段（执行测试、指标集）的探索。

---

## 18. Detecting Vulnerability-Inducing Commits via Multi-Stage Reasoning with LLM-Based Agents

**arXiv ID:** 2607.05772 | [PDF](https://arxiv.org/pdf/2607.05772v1)

**作者:** Liyou Chen `[一作]` (Beihang University), Yue Pan `[通讯]` (North China Municipal Engineering Design & Research Institute Co., Ltd.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于大语言模型的多智能体框架，通过多阶段推理来检测提交时引入的漏洞提交（VICs）。

**💡 创新点**

创新点在于将角色专门化的多智能体与多阶段（预处理、预筛选、再分析、最终决策）推理流程结合，并引入知识库检索辅助，实现从结构、意图到漏洞检测的分解与逐步细化。

**🔧 技术方法**

使用了大语言模型（DeepSeek‑V3.2、Qwen‑Plus、GPT‑4o‑mini）、角色专属提示、JSON交互、CodeBERT嵌入检索、ReAct式循环、以及多阶段温度调控等技术。

**📊 数据集**

实验采用 V‑SZZ 数据集（241 条提交，106 条 VIC）以及最近披露的 20 条 CVE（11 条 VIC）进行泛化评估。

**📈 对比分析**

与 Direct、CoT、CodeAgent 等基线相比，跨模型平均 F1 提升 1.2–1.7 倍，召回率显著提升；在最新漏洞上召回率最高 55%，F1 60%，成本略高但可接受。

**⚠️ 局限性**

局限性包括：依赖 LLM 推理易出现不一致结果；检索效果受知识库质量限制；最终决策阶段保守导致部分召回下降；仅针对提交级别，未覆盖漏洞定位或补丁生成；实验受限于 V‑SZZ，泛化性仍需进一步验证。

---

## 19. Optimism as a Vulnerability: Deceptive Stackelberg Control of UCB Bandit Followers

**arXiv ID:** 2607.05423 | [PDF](https://arxiv.org/pdf/2607.05423v1)

**作者:** Şuayp Talha Kocabay `[一作]` (Independent Researcher), Talha Rüzgar Akkuş `[通讯]` (Independent Researcher)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并证明了一种利用UCB学习者的乐观性进行欺骗的Stackelberg领导者机制，能够在有限时域内获得超过传统强Stackelberg均衡的累计收益。

**💡 创新点**

创新点在于将传统的静态均衡概念与动态经验学习结合，设计了“蜜罐-陷阱”两阶段策略，并证明在满足目标可达性与可利用性条件时，领导者可以通过操纵经验历史获得额外收益。

**🔧 技术方法**

主要使用的技术包括UCB1上界置信奖励、强Stackelberg均衡分析、序列式收益计算与高概率偏差界推导，以及构造性证明和误差递推分析。

**📊 数据集**

实验数据集为合成的10×10非零和矩阵游戏和基于目标覆盖的安全游戏拓扑，使用随机种子和不同T值进行多次实验。

**📈 对比分析**

通过与传统强Stackelberg领导者、变化点检测Follower以及EXP3学习者的对比，实验显示在安全游戏中欺骗机制可显著提升累计收益（正优势），但在一般矩阵游戏中表现不佳，理论匹配的“返回SSE”策略在合适参数下可实现大幅正优势。

**⚠️ 局限性**

局限性包括：需要满足严格的目标可达性与可利用性假设；证明在确定性奖励下给出，随机奖励需要额外的高概率修正；仅考虑有限时域；并且假设领导者能完全知晓Follower的统计量和收益矩阵，实际应用中对信息可观测性的依赖较高。

---

## 20. Domain-Adaptive Climate Downscaling Under Temporal Distribution Shift

**arXiv ID:** 2607.05645 | [PDF](https://arxiv.org/pdf/2607.05645v1)

**作者:** Shuochen Wang `[一作]` (Northeastern University), Auroop R. Ganguly `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对美国大陆日常温度进行深度学习下尺度，研究在历史训练与未来预测分布漂移条件下的模型表现。

**💡 创新点**

将时域域适配（源域监督重建+目标域对齐）引入气候下尺度，显著提升非平稳气候下的重建精度。

**🔧 技术方法**

采用RCAN卷积超分辨率网络，加入梯度反转域分类器进行对抗式域对齐，并与传统统计、GAN等方法做对比。

**📊 数据集**

使用三对 GCM–RCM 日温模拟（CanESM2‑RCA4、CanESM2‑CanRCM4、EC‑EARTH‑RCA4），配合 PRISM 高程、海陆掩模等辅助变量。

**📈 对比分析**

在 2006–2040、2041–2070、2071–2099 三个未来验证期内，与双线性插值、BCSD、QDM、CDF‑t、RCAN、RCAN‑QDM、RCAN‑GAN 等基线比较，域适配模型在 MSE、RMSE、PSNR、SSIM 上持续领先，提升幅度随温度漂移加剧而增大，尤其在高海拔和复杂地形地区显著改善。

**⚠️ 局限性**

仅在模拟数据上验证；对极端事件和降水等非温度变量的适用性有限；若分布漂移过大或预测关系变化剧烈，域适配可能产生负迁移；需在观测数据上进一步检验。

---

## 21. Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding

**arXiv ID:** 2607.05722 | [PDF](https://arxiv.org/pdf/2607.05722v1)

**作者:** Yonggan Fu `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种可同时支持 AR、扩散和自推理三种解码模式的“Tri‑Mode”语言模型，利用联合 AR‑扩散训练实现高精度与高并行度。

**💡 创新点**

创新点包括：
1) 通过统一的双流注意力和全局损失平均实现 AR 与扩散目标的协同优化；
2) 两阶段训练（先 AR 再联合）显著提升学习效率；
3) 自推理（self‑speculation）将扩散草稿与 AR 验证结合，提升接受率与吞吐率；
4) 采用 LoRA 微调对草稿与验证器对齐；
5) SOL（speed‑of‑light）分析揭示扩散解码潜在上限；
6) 迁移到视觉‑语言模型，保持多模态性能。

**🔧 技术方法**

技术手段包括：
- block‑wise diffusion denoising + causal + bidirectional attention；
- 统一的 AR 与扩散损失，权重 α=0.3；
- 全局损失平均消除随机遮掩导致的方差；
- 训练时采样器预测每步是否可接受；
- LoRA‑增强的自推理草稿器；
- 递归动态压缩搜索实现 SOL 上限；
- 适配多尺度（3B/8B/14B）和 VLM 结构。

**📊 数据集**

数据集与评测：
- 预训练：25B 连续 token（大规模多域文本）；
- SFT：45B 指令遵循数据；
- 基准测试：HumanEval、MBPP、LiveCodeBench‑CPP、GSM8K、Math500、AIME24/25、MMLU、GPQA、IFEval、ARC‑E/C、HellaSwag、PIQA、Winogrande、写作等；
- VLM 评测：AI2D、ChartQA、DocVQA、MMMU、MMMU‑Pro‑V、MathVista、RealWorldQA 等。

**📈 对比分析**

与 SOTA 对比：
- AR 模式下 8B 模型准确率平均 54.55%，高于 Qwen3‑8B（53.12%）及其他基线；
- 扩散模式 TPF 达到 2.57×，比 Qwen3‑8B 的 1× 提升 2.57×；
- 自推理（含 LoRA）TPF 可达 5.99× 或 6.38×，同时保持 1% 内的准确率；
- VLM 模式在多模态基准上相对 LLaDA‑V‑8B 提升约 1–2% 准确率，TPF 3–7×；
- 解决方案在低并发场景下相较 Eagle3‑MTP 实现 2.4–3.3× 的吞吐率提升。

**⚠️ 局限性**

局限性：
- 当前采样器仍与 SOL 上限相差约 30%–50%，并非最优；
- 自推理仅支持前缀验证，无法充分利用扩散草稿的非前缀特性；
- 对扩散模式的硬件优化受限于现有 GPU 内核；
- 仍需进一步提升在开放式生成任务（如 roleplay、写作）的准确率；
- 训练成本高，尤其是 8B/14B 的两阶段训练与大规模数据。

---

## 22. Data-dependent Evaluations for Budgeted Submodular Maximization

**arXiv ID:** 2607.05759 | [PDF](https://arxiv.org/pdf/2607.05759v1)

**作者:** Lejian Zhang `[一作]` (Nanyang Technological University), Jing Tang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的数据依赖上界，用于解决带有背包约束的子模最大化问题，并通过理论证明和实证实验展示了其优越性。

**💡 创新点**

创新点在于提出了切片策略和移除策略，构建了比现有界限更紧的上界，并通过线性规划将这些策略转化为可计算的形式。

**🔧 技术方法**

使用了线性规划技术来实现数据依赖上界的计算，并结合了切片和移除策略。

**📊 数据集**

实验使用了多个真实世界数据集，包括成人收入数据集、社交网络数据集（ego-facebook和com-youtube）以及Caltech36数据集。

**📈 对比分析**

通过与现有的上界（如Λ^0）进行比较，新的上界（如Λ^1、Λ^2、Λ^3）在大多数问题实例中表现出更高的近似保证，证明了其有效性。

**⚠️ 局限性**

限制在于当前方法主要针对带有背包约束的子模最大化问题，未来的工作将探索如何将这些技术扩展到其他约束的子模最大化问题。

---

## 23. Boosting FPGA Performance with Direct BRAM-DSP Paths

**arXiv ID:** 2607.05756 | [PDF](https://arxiv.org/pdf/2607.05756v1)

**作者:** Jiajun Hu `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种轻量化的FPGA架构改进方案，在BRAM与DSP之间引入直接互连路径，并相应增强了放置算法，以实现更高频率与更短线路。

**💡 创新点**

创新点在于首次实现跨类型的BRAM–DSP直接通路，保持原接口兼容，同时通过宏级别的跨类型宏和连通驱动排序提升放置效果。

**🔧 技术方法**

使用了VTR工具进行架构评估、宏级放置扩展、连通驱动优先级算法，以及基于Agilex‑10类似模板的仿真。

**📊 数据集**

实验数据集包括三种常见DL层（全连接、卷积、BERT‑Tiny注意力）以及五个非DL基准（or1200、mkSMAdapter4B、bgm、LU8PEEng、stereovision2）。

**📈 对比分析**

通过与默认VTR流程对比测量Fmax和总线长，结果显示在DL层上最高可提升25%频率、缩短49%线路长度；非DL基准无性能损失。

**⚠️ 局限性**

局限性在于仅验证了中小规模设计，未评估在更大工作负载（如Koios、Titan）下的效果；直接路径的延迟/面积模型采用简化估计，实际实现可能有差异。

---

## 24. NegROI: Click-Centric Uncertainty-Guided Refinement with Scene-Conditioned Negative Prompts for Robust Interactive 3D Segmentation

**arXiv ID:** 2607.05955 | [PDF](https://arxiv.org/pdf/2607.05955v1)

**作者:** Shuheng Zhang `[一作]` (University of Science and Technology of China), Feng Wu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为NegROI的交互式3D分割框架，利用点击中心多分辨率ROI细化与场景条件负提示相结合，提升点击效率并抑制误检。

**💡 创新点**

创新点包括：①点击中心局部多分辨率细化实现局部高分辨率推理；②场景条件负提示通过跨注意力学习背景负样本，配合多样性正则化避免提示退化；③边界感知硬负样本监督提升负提示对边界误检的抑制；④不确定性驱动的ROI选择聚焦模糊区。

**🔧 技术方法**

使用稀疏体素编码、Easy3D两向Transformer解码器、跨注意力负提示学习、最大池化融合细化结果、基于多头注意力的边界硬负样本监督与多样性正则。

**📊 数据集**

在ScanNet40、S3DIS、KITTI-360三大公开点云分割基准上进行评估，并在ScanNet20/40进行单点评估。

**📈 对比分析**

与InterObject3D、AGILE3D、Easy3D、Point‑SAM等基线对比，NegROI在所有数据集、所有点击数下均实现IoU@1–10提升，尤其在低点击预算下优势明显，且跨域泛化能力优于对比方法。

**⚠️ 局限性**

局限性在于ROI选择仍基于硬阈值，未实现完全可微；对极稀疏点云或极端密集噪声点的鲁棒性尚待验证；负提示的学习依赖于训练数据分布，跨域时可能出现不完整的负样本覆盖。

---

## 25. ProvICS: A Provenance-based Intrusion Detection for Industrial Control Systems

**arXiv ID:** 2607.05989 | [PDF](https://arxiv.org/pdf/2607.05989v1)

**作者:** Md Neyamul Islam Shibbir `[一作]` (University of Texas at El Paso), Deepak K Tosh `[通讯]` (University of Texas at El Paso)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个基于硬件环回的工业控制系统测试平台，并收集了四模态同步数据，用以构建ProvICS多模态溯源数据集；随后在该数据集上评估基于自动编码器的跨模态异常检测方法。

**💡 创新点**

①将主机级与PLC级溯源图、解码的Modbus协议语义、原始PCAP以及物理过程状态四模态统一同步；②首次公开实现多主机内核级溯源与实时工业协议深度解析的综合数据集；③展示跨模态融合可实现100%事件级召回、F1=0.9133的检测性能。

**🔧 技术方法**

利用Auditd+SPADE框架采集主机溯源，针对PLC的定制审计规则生成PLC溯源；使用DeepPacket Inspection对Modbus/TCP进行语义解析；Node-RED数字孪生模拟CSTR并通过FUXA采集物理状态；使用GraphSAGE和MLP自动编码器对溯源图、协议图和物理特征进行异常检测，并通过多模态融合（sum‑z、max‑z、OR‑calibrated）进行性能评估。

**📊 数据集**

ProvICS数据集：48小时正常运行+22小时包含四个攻击赛季、32个攻击事件、20种ICS ATT&CK技术的多模态数据；该数据集已公开托管在Hugging Face上。

**📈 对比分析**

与单模态方法（溯源、物理或协议）对比，单模态召回率分别为0.750、0.531、0.688；三模态sum‑z融合达到100%召回率、F1=0.9133、FPR=1.40%；相比现有ICS/CPS数据集，ProvICS在多主机溯源、协议语义、物理状态、真实PLC硬件与ATT&CK映射等维度均具备更全面的特征。

**⚠️ 局限性**

局限性：仅包含单个PLC与单台主机的规模较小；物理过程为数字孪生模拟，未覆盖真实物理设备；仅测试了有线Modbus/TCP协议，未包含无线或加密工业协议；缺乏大规模多PLC、多主机跨域场景。

---

## 26. Layer 2 Coordinated Trusted Setup for Continuous CRS Generation

**arXiv ID:** 2607.05776 | [PDF](https://arxiv.org/pdf/2607.05776v1)

**作者:** Khalid Hassan `[一作]` (University of Manitoba), Sara Rouhani `[通讯]` (University of Manitoba)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了基于Layer 2去中心化序列器网络的连续CRS（Common Reference String）生成框架，使得传统一次性可信设置仪式转变为可持续、去中心化的服务；

**💡 创新点**

创新点在于：①将高吞吐量的rollup序列器与多轮可信设置的协作需求分离；②利用PBFT共识与commit‑reveal结构实现无中心化的多轮Ceremony；③同时支持智能合约与纯P2P两种协同模式；

**🔧 技术方法**

技术手段包括：PoT多方安全计算、PBFT共识、commit‑reveal、ZK‑Rollup、以太坊L1 anchoring、libp2p+Kademlia网络、Circom PoT实现；

**📊 数据集**

实验数据集为仿真环境，部署6/12个节点，功率规模为8/10/12，测试多轮Ceremony时长及其对交易吞吐量的影响；

**📈 对比分析**

性能对比显示：去中心化Rollup可达约41 k tx/s；Ceremony平均时长随功率与节点数上升，从47 s到约276 s；即使在节点掉线或提交非法贡献的攻击下，Ceremony仍能在不完全重启的情况下完成，证明了鲁棒性；

**⚠️ 局限性**

限制：实验仅在单机仿真下完成，未覆盖大规模真实网络；Ceremony对节点数和功率高度敏感，规模扩大会显著增加时延；对“始终诚实”的假设依赖较强，若持续恶意节点存在时效性仍有限；

---

## 27. Why does Deep Learning Improve Visual SLAM?

**arXiv ID:** 2607.06023 | [PDF](https://arxiv.org/pdf/2607.06023v1)

**作者:** Giovanni Cioffi `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在经典几何 SLAM（ORB‑SLAM3）中替换 2D 匹配器为学习得到的光流网络，并引入其不确定性估计，系统化地研究了深度学习前端在 SLAM 中的关键作用。

**💡 创新点**

创新点在于将学习到的光流和置信度直接植入传统前后端结构，提供可控实验框架证明数据关联与不确定性是提升性能的核心，而递归架构并非必要。

**🔧 技术方法**

技术包括：光流网络（RAFT‑based）、置信度加权 Bundle Adjustment、基于 ORB 的传统后端、实验框架对比与评估。

**📊 数据集**

使用的公开数据集为：TartanAir（模拟高动态、低纹理和光照变化）和 UZH‑FPV（实际无人机高速飞行），涵盖 1.5 万+帧和多种摄像头姿态。

**📈 对比分析**

比较方法：对所有系统（ORB‑SLAM3、ORB‑SLAM3+Flow、ORB‑SLAM3+Flow+Uncertainty 以及现有深度学习 SLAM 如 D3VO、VINS‑Mono 等）在轨迹误差（ATE）上进行平均评估；实验表明，加入光流和置信度后 ORB‑SLAM3 的性能接近或超越最新 DL‑SLAM，在所有难度级别的测试中均显著提升，尤其在大光流和低纹理场景。

**⚠️ 局限性**

局限性包括：依赖光流网络的训练数据分布；在极端动态场景下仍易失效；光流推理与不确定性估计增加计算负担；未对不同摄像头模型或多传感器融合进行深入验证。

---

## 28. From Regression to Prior-Aware Inference: Solving the ILWE Family in Randomness Leakage Attacks against ML-DSA

**arXiv ID:** 2607.05921 | [PDF](https://arxiv.org/pdf/2607.05921v1)

**作者:** Peiheng Zhang `[一作]` (Nanjing University of Science and Technology), Yongbin Zhou `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了ML-DSA随机性泄漏攻击的求解器阶段，提出一个统一框架，用以系统评估不同求解器在 OILWE、FS-ILWE 以及 CILWE 上的表现。

**💡 创新点**

创新点在于：①把不同求解器（普通最小二乘、鲁棒回归、Belief Propagation、贪婪搜索、爬山等）归入同一 ILWE‑族框架；②通过实验证明，先验离散推理（BP 等）能在多种泄漏模型下将所需信息关系数减少 10–60 倍；③阐释了在噪声或隐藏率较高时鲁棒回归的优势。

**🔧 技术方法**

所用技术包括：线性回归（OLS、最大似然）、鲁棒回归（Huber、Cauchy、ℓ1‑LP）、先验离散推理（Belief Propagation、目标诱导贪婪搜索、受限误差爬山）、以及对比实验中基于梯度下降和 QR/ SVD 的数值实现。

**📊 数据集**

数据集：合成的 ILWE 样本，使用 ML-DSA 的三种参数集（ML-DSA‑44、ML-DSA‑65、ML-DSA‑87）生成的子密钥；针对 FS‑ILWE 还生成了噪声 FS‑ILWE；针对 CILWE 通过不同的隐藏率（p_con ∈ [0, 0.9]）混合零误差与隐藏样本。

**📈 对比分析**

比较方法：以所需信息关系数（或样本数）为指标，绘制相对 BP 的需求比；实验结果显示：OLS 与 Greedy‑ℓ2 的需求几乎相同；鲁棒回归在极低或高噪声/隐藏率下可略优；BP 在无噪声 FS‑ILWE、OILWE 以及 CILWE（p_con < 0.9）中，显著降低样本需求（最多 60 倍），而在高隐藏率时 Cauchy 回归表现更好。

**⚠️ 局限性**

局限性：①实验仅基于合成数据，未评估真实侧信道噪声与硬件实现的影响；②BP 在隐藏率极高（p_con ≥ 0.9）时失效；③鲁棒回归在大多数情形下并未显著优于 OLS；④实验未对求解器的计算复杂度与实时性做深入分析。

---

## 29. ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modelin

**arXiv ID:** 2607.05583 | [PDF](https://arxiv.org/pdf/2607.05583v1)

**作者:** Archie Chaudhury `[一作]` `[通讯]` (Axionic Labs), Archie Chaudhury (Axionic Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种新的长上下文语言模型ResonatorLM，用阻尼谐振核替代自注意力，将token序列视为驱动的潜在场。

**💡 创新点**

创新点在于引入物理衍生的因果共振场混合，实现在训练时保持并行全序列处理、解码时使用固定递归状态，从而显著提升长上下文效率。

**🔧 技术方法**

采用阻尼谐振核、FFT因果卷积、固定递归状态更新、跨头耦合、局部深度卷积、SwiGLU MLP，以及物理参数可视化等技术。

**📊 数据集**

在WikiText‑2字符数据集上进行训练与评估，并通过kernel‑only scaling benchmark 测试算法加速。

**📈 对比分析**

与同等6M参数的标准Transformer对比，32K token下解码速度提升6.47倍，准确率从55.32%提升至61.31%；在kernel benchmark下，速度提升约440倍至576倍。

**⚠️ 局限性**

局限性包括仅在中等规模实验验证，缺乏大规模多模态或更长序列任务的广泛评估；实际部署时对硬件实现与兼容性仍需进一步研究。

---

## 30. TRIG: Trajectory-Rig Decoupled Metric Geometry Learning

**arXiv ID:** 2607.05801 | [PDF](https://arxiv.org/pdf/2607.05801v1)

**作者:** Lizhou Liao `[一作]` (Carizon), Chang Huang `[通讯]` (Carizon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种轨迹‑架构解耦的多摄像头视觉几何框架 TRIG，用于精准恢复自驾系统的度量几何与车辆姿态。

**💡 创新点**

核心创新在于：①将时间变化的车辆轨迹与静态相机阵列几何解耦；②设计稀疏时空注意力（STSA）以高效聚合跨帧与跨摄像头信息；③采用解耦姿态监督分别约束轨迹与架构，提升姿态稳定性与几何一致性。

**🔧 技术方法**

基于 VGGT 的视觉几何基础模型，加入解耦姿态编码/监督模块、稀疏时空注意力网络，以及对齐与评估的度量深度与 3D 重建头。

**📊 数据集**

在五大自驾基准上进行评测：KITTI、NuScenes、Waymo、OpenScene、DDAD。

**📈 对比分析**

与 DVGT、OmniVGGT、VGGT 等主流基线对比，TRIG 在度量深度、3D 重建的准确性/完整性以及相机姿态的 AUC@30° 等指标上均实现了新一代最高成绩，显著优于现有方法。

**⚠️ 局限性**

局限性包括：①仍需预先校准的固定相机阵列，难以适应动态或可变相机布置；②对极端天气、强遮挡或低纹理区域的鲁棒性不足；③模型对计算资源的需求仍高，尤其在大规模场景下的实时部署尚需进一步优化。

---

## 31. VEIL: How Visual Encoding Hijacking Induces Bias In Vision Models

**arXiv ID:** 2607.05641 | [PDF](https://arxiv.org/pdf/2607.05641v1)

**作者:** Suranjana Sooraj `[一作]` (University of California, Davis), Dongyu Liu `[通讯]` (University of California, Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究图像化时间序列分类时不同图表编码方式对模型学习的影响，并提出诊断框架 VEIL。

**💡 创新点**

首次量化视觉编码劫持现象，区分编码特定信息与真正时序特征，并用多维诊断方法评估编码依赖性。

**🔧 技术方法**

采用 CKA 代表性相似度、交叉图表线性探针、Grad‑CAM 归因、PCA/UMAP 结构可视化、ESI 统计指标以及 HINT 关注引导等技术。

**📊 数据集**

在 31 个 UCR 公开数据集上进行实验，覆盖线、面积、柱状和散点四种标准图表编码。

**📈 对比分析**

对比实验显示，编码之间的相似度和可迁移性差异显著；HINT 在编码差异大的数据集上提升 20–40% 但在编码一致的数据集上无显著或负面效果，说明方法不是通用解决方案。

**⚠️ 局限性**

局限性包括：无法明确编码特征是否真正为捷径；HINT 并非对所有情况有效；实验仅覆盖 UCR 典型数据集，缺乏对更复杂或非平滑时间序列的验证。

---

## 32. Fréchet Distance Loss on Speech Representations for Text-to-Speech Synthesis

**arXiv ID:** 2607.06027 | [PDF](https://arxiv.org/pdf/2607.06027v1)

**作者:** Ho-Lam Chung `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一种无判别器的 Speech Representation Fréchet Distance (SR‑FD) 损失，用于调节 tokenizer‑free Flow‑Matching TTS（VoxCPM2）的少步采样生成分布，使四步采样下的语音更接近高质量语音分布。

**💡 创新点**

创新点在于：①利用冻结的 Whisper 与 wav2vec‑2.0 CTC 特征提取器在特征空间上匹配生成语音的均值与协方差；②将三种互补目标（低步 Whisper Anchor、教师 CTC、真实语音 CTC）结合，抑制少步采样时的内容漂移；③不需要判别器或推理时额外计算，直接在训练时正则化采样结果。

**🔧 技术方法**

采用的技术包括 Flow‑Matching TTS（VoxCPM2）、LoRA 参数高效微调、Whisper 与 wav2vec‑2.0 CTC 冻结特征提取、Fréchet 距离损失、四步 Euler 采样与指导（guidance）等。

**📊 数据集**

使用的数据集：用于微调的 767 行 LibriTTS 声学克隆材料；参考统计来自真实 LibriTTS、十步教师生成和 ASR 验证的四步生成；评测采用 Seed‑TTS English test‑en 集合（1088 句）。

**📈 对比分析**

通过与原始十步与四步 VoxCPM2 基线以及公开的 F5‑TTS、ARCHI‑TTS 在 Seed‑TTS WER 上对比，四步 SR‑FD 将 WER 从 2.23% 降至 1.41%（比四步基线下降 36.5%，比十步基线下降 18.5%），speaker similarity 与质量代理基本保持十步水平，盲听实验显示与十步系统无显著差异。

**⚠️ 局限性**

局限性包括：SR‑FD 的原始 FD 值对模型选择预测性弱，仍需依赖外部 WER 评估；实验仅验证了少步（四步）对 VoxCPM2 的效果，未探讨不同步长或模型规模的普适性；对生成语音的非内容质量（如情感、说话人细节）仍未做深入验证。

---

## 33. LibFHE: A Numba-Based CUDA-Python Library for Non-RNS CKKS-BGV Fully Homomorphic Encryption on GPUs

**arXiv ID:** 2607.05920 | [PDF](https://arxiv.org/pdf/2607.05920v1)

**作者:** John Chiang `[一作]` `[通讯]`, John Chiang

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出并实现了 LibFHE——一种基于 Numba 的 CUDA‑Python GPU 加速全同态加密库，采用非 RNS CKKS‑BGV 方案实现多密文批处理与加密、解密、运算以及完整的 Bootstrap 流程。

**💡 创新点**

创新点在于：①首次把原始 CKKS（非 RNS）迁移到 GPU，避免 RNS 造成的 64 位模数与 32 位 GPU 之间的匹配问题；②通过纯 Python/Numba 的 JIT 结合 CUDA‑Python 低层接口，既保持了高效裸机性能，又极大提升了开发可用性与生态兼容性；③设计了 CipherTensor 批量结构与对齐的内存布局，支持 pairwise 与 broadcast 两种执行模式；④整合多种 GPU 级优化（NTT 矩阵化、共享内存打包、预计算 Twiddle、Garner 合成、kernel fusion 等），实现与 C++ 库相当的性能。

**🔧 技术方法**

技术细节包括：Numba JIT 编译、CUDA‑Python 低层 API、共享内存加速的双步 NTT、基于 30–31 位质数的 RNS 与 Garner 重构、矩阵化 Butterfly 以提升内存局部性、预计算根与 Twiddle、旋转键与复共轭支持、以及 CipherTensor 的多密文批处理与广播机制。

**📊 数据集**

实验使用 Google Colab Tesla T4 GPU，采用 CKKS‑BGV 参数组（log N=16, log Q=1024, log p=60, log s=2）以及 bootstrapping 相关参数（log p=30, log q=40, log T=2）。

**📈 对比分析**

通过与传统 C++ FHE 库（如 SEAL、OpenFHE 等）以及自研 CUDA C++ 实现对比，测得多密文批处理下每个操作的平均延迟与单密文延迟，Bootstrapping 为主导耗时，其他算子如乘法、旋转在 1–2 ms 内完成，整体性能与优化 C++ 相当，且在 T4 设备上实现了更高的可编程性与更低的实现复杂度。

**⚠️ 局限性**

局限性包括：目前仅实现了多项式运算、加解密与 Bootstrap，尚未加入 Barrett/Montgomery 简化、完整的 kernel fusion、CUDA 流并行、多 GPU 与多节点分布式扩展；实验仅在 T4 设备上验证，对更大规模或不同 GPU 架构的迁移仍待评估。

---

## 34. Strategic Bargaining in Multi-Buyer Markets: Reinforcement Learning from Verifiable Rewards for LLM Negotiations

**arXiv ID:** 2607.05863 | [PDF](https://arxiv.org/pdf/2607.05863v1)

**作者:** Shuze Daniel Liu `[一作]` (Massachusetts Institute of Technology), David Simchi-Levi `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

训练一个单卖家在多买家并发谈判环境中使用大型语言模型的谈判代理，并通过可验证奖励（RLVR）使其既能高效探索买家又能获取最大利润。

**💡 创新点**

创新点在于：①将谈判收益作为可验证的终端奖励；②通过RLVR让LLM自发学习价格锚定、诊断性探询等探索策略；③在并发多买家情境下构建完整的POMDP框架，实现对信息不对称的自适应决策。

**🔧 技术方法**

技术包括：大语言模型（Qwen3 30B）+ 强化学习框架（RLVR）+ 受限通信预算的POMDP设计 + 结构化动作空间（目标买家、自然语言对话、交易动作）。

**📊 数据集**

使用 930 件真实商品的公开电商数据集（覆盖 18 类产品），训练集 802 件，测试集 128 件；买家预算采用三段区间（[0.40,0.55],[0.65,0.80],[1.00,1.15]）的随机排列，首轮报价受限于 [0.25,0.30]·list_price。

**📈 对比分析**

与 12 个基准模型（参数 4B–1T，包含指令调优、推理模型）在同一并发多买家设置下比较。训练后模型在分布内实现 Reward +0.580、DealRate 82.8%、SellerSurplusExtractionRatio 70.0%，明显优于最佳基准 +0.123；在分布外多买家、多预算场景下仍保持最高 Reward +0.540、SurplusExtraction 77%，表明策略具备良好泛化。

**⚠️ 局限性**

局限性：①仅关注单卖家场景，未考虑双向竞争或买方学习；②奖励设计仅针对单一商品的利润，难以直接推广到多属性、多产品或法律合约类谈判；③实验基于模拟环境，实际平台中的动态信息、网络延迟等未被覆盖；④RLVR训练对环境可验证性要求高，迁移到不可验证的经济目标时需重新设计。

---

## 35. The Oracle's Gambit: A Game-Theoretic Framework for Responsible AI Release

**arXiv ID:** 2607.05442 | [PDF](https://arxiv.org/pdf/2607.05442v1)

**作者:** Christoph R. Landolt `[一作]` (CISPA Helmholtz Center for Information Security), Mario Fritz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个三方Stackelberg博弈框架，用于评估前沿AI模型发布策略对防御者福利的影响。

**💡 创新点**

首次将AI能力差距而非单纯信息优势作为衡量安全福利的关键变量，并通过预发布窗口优化实现对防御者的保护。

**🔧 技术方法**

采用双层Stackelberg博弈、后向归纳求解零和随机游戏，并结合LLM-Delphi专家审议法来估计模型能力参数。

**📊 数据集**

利用公开漏洞修补时间数据、覆盖率估计以及多种安全基准（CyberGym、ExploitGym、BreachBench等）来校准模型参数。

**📈 对比分析**

在模拟的前沿模型升级路径中，对比公开发布、预发布和禁运三种策略，结果显示预发布可将攻击频率降低约20–25%，并提升防御者福利3–4倍。

**⚠️ 局限性**

模型假设完全可观测、零和支付且单期终止，且延迟成本与行动成本均需人工设定，缺乏对多周期、异质防御者及更复杂发布机制的考虑。

---

## 36. Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory

**arXiv ID:** 2607.05511 | [PDF](https://arxiv.org/pdf/2607.05511v1)

**作者:** Chang Nie `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Light-Omni 框架，利用双状态（全局状态与潜在状态）实现视频理解的即时反射式响应，并在睡眠时异步更新长时记忆。

**💡 创新点**

创新点在于：1) 双状态设计——全局状态提供非参数化的多模态脚本，潜在状态在单次前向传播中直接生成语义对齐的检索嵌入；2) 睡眠时层次合并的全局记忆聚合，避免过度依赖迭代推理；3) 通过软提示与对齐损失实现检索与生成的协同训练。

**🔧 技术方法**

主要技术包括多模态长时记忆系统（用户、语义、情节三层），多 LoRA 模块解耦记忆、生成、反应任务，Soft Prompt、对齐损失实现检索嵌入学习，特征缓存与冗余剪枝优化推理延迟。

**📊 数据集**

使用的公开视频数据集有 VideoMME-long、LVBench、HippoVlog、OVO-Bench；在训练时自动合成多场景视频数据，约 43k 训练样本。

**📈 对比分析**

与 172 种通用 MLLM、173 种记忆增强方法和 174 种推理型视频代理进行对比；在 VideoMME-long 上平均准确率 58.0%，比 GPT‑4o（48.1%）、Gemini‑2.0‑Flash（55.8%）及基线 Qwen2.5‑Omni‑7B 提升 9.5%；在速度上 20× 加速，显存 3× 降低；在检索鲁棒性测试中，轻微噪声仅降 1.3%，远优于 RAG（5.1%）和 RAG‑Rewrite（3.7%）。

**⚠️ 局限性**

局限性：1) 当前仅支持搜索和观察两类动作，工具调用仍待扩展；2) 对极端长时序（数月）仍需验证，尽管层次合并控制规模；3) 依赖预训练大模型，适配性受限于所选 backbone；4) 在极端噪声或跨模态对齐不充分的场景下，检索精度仍可能下降。

---

## 37. RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures

**arXiv ID:** 2607.06018 | [PDF](https://arxiv.org/pdf/2607.06018v1)

**作者:** Hanan Gani `[一作]` (University of California, San Diego), Manmohan Chandraker `[通讯]` (University of California, San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用LLM规划将任务拆分成子目标，结合视频生成模型与VLM批评器端到端学习任务对齐的仿真未来并训练机器人策略。

**💡 创新点**

创新点在于将层次化语言规划直接作为视频生成条件，并通过可微强化学习将VLM批评器反馈嵌入生成模型，实现任务对齐的内部表示。

**🔧 技术方法**

使用的技术包括大规模LLM（Gemini‑2.5‑Pro）、Stable Video Diffusion、CLIP/BLIP VLM批评器、Diffusion Policy、DDPO等。

**📊 数据集**

实验数据集为RoboCasa和LIBERO10仿真环境。

**📈 对比分析**

与VideoPolicy、UVA、DP等基线相比，平均成功率提升至RoboCasa约0.64、LIBERO10 0.97，显著优于现有方法。

**⚠️ 局限性**

局限性主要是依赖冻结的VLM评判器奖励，缺乏细粒度子目标进度信号，且在真实机器人环境中的迁移性尚未验证。

---

## 38. Designing Computerized Gait Analysis for Pediatric Care: Clinician Perspectives on Sensing, Workflow, and Care Environments

**arXiv ID:** 2607.06076 | [PDF](https://arxiv.org/pdf/2607.06076v1)

**作者:** Elizabeth Hong `[一作]` (Stanford University), Yiwen Dong `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对12名儿科临床专家和1名系统设计师的访谈与问卷，探讨了计算机化步态分析（CGA）在儿童患者中的应用经验与需求，提出针对儿童的设计改进建议。

**💡 创新点**

创新点在于揭示成人导向的CGA技术与儿童生理、感官、行为差异之间的匹配缺口，并系统性地给出低负担、无标记、环境感知和自动化等儿童中心化的设计方案。

**🔧 技术方法**

使用的技术包括传统的标记式光学捕捉、力板、EMG、惯性测量单元（IMU）、压力垫、以及探索中的视听感知与可穿戴/环境传感技术。

**📊 数据集**

数据来源主要是从临床专家与设计师的半结构化访谈、情景式问卷以及工作坊的定性记录，没有采用公开步态数据库或大规模量化数据集。

**📈 对比分析**

通过对七项性能评估指标（准确性、功能性、易用性、舒适度、可负担性、可获取性、可扩展性）和五类步态参数在不同临床环境中的重要性进行排名，发现临床专家普遍重视准确性与功能性，但对现有技术在儿童测量中的适用性与完整性仍持质疑态度。

**⚠️ 局限性**

局限性包括样本仅来自单一大学附属医院，受访者数量有限，缺乏对儿童与家长的直接体验收集，以及未对新设计方案进行客观性能评估。

---

## 39. Hybrid electrolyzer systems: Smart strategy or economic fallacy?

**arXiv ID:** 2607.06093 | [PDF](https://arxiv.org/pdf/2607.06093v1)

**作者:** Marie Arnold `[一作]` (EWE GASSPEICHER GmbH), Richard Hanke-Rauschenbach `[通讯]` (Leibniz Universität Hannover)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对混合电解槽系统（HES）在电解槽效率和投资成本两个关键参数的全空间敏感性分析，评估其在大型绿色氢气供应链中的技术经济可行性。

**💡 创新点**

创新点在于：①独立改变电解槽效率和CAPEX，摆脱传统AWE/PEMWE技术特性假设；②采用开放式设计优化和大规模参数扫描，量化HES在不同参数组合下的成本收益和占比；③定义SEC/CAPEX的成本比率，提供决策依据。

**🔧 技术方法**

使用线性优化模型计算LCOH、储能与PPA规模；进行大规模参数空间扫描；采用年功率持续曲线（APDC）分析电解槽运行特征；并通过敏感性分析评估PPA价格、储能费用、RES可用性对结果的影响。

**📊 数据集**

主要数据来源于2024年德国/欧盟可再生能源PPA电价和容量因子时间序列、公开的电解槽CAPEX与SEC基准值、氢需求曲线（5500 kg/h）以及储存费用等公开数据库。

**📈 对比分析**

对每组参数组合求解成本最优设计并计算LCOH，随后与单一电解槽系统的最低LCOH进行比较。结果显示HES仅在5%以内的案例中获益，最大成本优势仅为0.057 €/kgH₂（约占总成本1%）。

**⚠️ 局限性**

局限性包括：①仅考虑恒定氢需求，未纳入需求波动；②未对电解槽退化、上升速率等技术不确定性进行建模；③基准参数设定可能影响敏感性结果；④仅聚焦于大型工业规模，可能不适用于小型或分散式项目。

---

## 40. Reproducible Validation of Voucher-Based L2 Interoperability: Diagnosing an ERC-4337 Compatibility Issue in an EIL SDK Implementation

**arXiv ID:** 2607.05914 | [PDF](https://arxiv.org/pdf/2607.05914v1)

**作者:** Cheng-En Lee `[一作]` (National Taiwan Normal University), Yun-Cheng Tsai `[通讯]` (PecuLab LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了可复现的双 L2 验证框架，诊断并修复了 EIL SDK 中的支付主机数据兼容性问题。

**💡 创新点**

提出了针对后签名支付主机数据不一致的本地兼容层和确定性测试环境，实现了完整的 voucher 请求‑签发‑赎回流程可追溯。

**🔧 技术方法**

使用 ERC‑4337 账户抽象、XLP 票据机制、Mock bundler、事件驱动的 XLP 发行者、Create2 定位部署以及本地 Anvil 节点模拟 Arbitrum/Optimism L2。

**📊 数据集**

基于 USDC 流通量的演示交易，使用 Arbitrum Sepolia 与 Optimism Sepolia 公测网络及本地 Anvil 作为验证数据。

**📈 对比分析**

通过对比本地执行日志与公开区块链交易记录验证兼容层效果，展示在修复前后 voucher 生命周期可完整执行，性能未做量化。

**⚠️ 局限性**

局限于单一 SDK 版本和受限的功能范围，未覆盖生产 bundler、标准 ERC‑4337 账户、跨链结算与争议解决等场景。

---

## 41. From Conversation to Contribution: Characterizing Coding Agent in Open-Source Software

**arXiv ID:** 2607.05677 | [PDF](https://arxiv.org/pdf/2607.05677v1)

**作者:** Zihan Fang `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过收集并分析13,360条AI编程助手对话及其对应的1,240个开源项目的历史记录，探究了vibe‑coding（通过自然语言引导AI生成代码）在OSS中的使用模式、对项目活动和代码质量的影响，并结合25位开发者调查，呈现了AI使用者的感知与担忧。

**💡 创新点**

创新点在于首次将大规模对话日志与完整项目历史结合，系统评估vibe‑coding在不同项目规模、成熟度与协作程度中的使用差异，并关联对话目的与后续提交特征，同时通过问卷洞察社区对AI生成代码的维护担忧与披露意愿。

**🔧 技术方法**

使用的技术包括：SpecStory工具获取AI对话日志；GitHub REST API抓取仓库完整历史；LLM（GPT‑5）进行对话目的和项目类型分类；统计分析（Spearman相关、二项回归、ITS模型、Wilcoxon等）评估AI使用前后指标变化；以及基于Qualtrics的在线问卷收集主观数据。

**📊 数据集**

数据集为公开GitHub仓库的AI对话日志（13,360会话，79,172条用户信息）和相应的1,240个仓库的完整提交、issue、PR、评论、CI记录，覆盖2013–2026年；另外包含652名公开邮箱开发者的问卷回复。

**📈 对比分析**

比较方法主要是采用前后对照（pre‑post）以及Interrupted Time Series（ITS）模型评估AI使用的即时及趋势效应。结果显示：AI使用在小型、低协作项目更集中；使用后贡献者数略增、贡献者集中度下降；但提交速率整体下降、代码质量指标（bug/fix、CI失败率等）无显著恶化；vibe‑coding后续提交多为源代码或文档变更，规模与范围无显著差异。

**⚠️ 局限性**

局限性包括：样本主要为2025后的小型公共仓库，缺乏对成熟大型项目的代表性；对话日志仅来自Copilot、Cursor、Claude Code等工具，可能忽略其他工具；因只记录首次观察到的AI对话，无法确定因果关系；问卷样本量低（25人），且仅限公开邮箱开发者，可能产生偏倚。

---

## 42. Information Limits and Attractor Dynamics in Economies of Frontier LLM Agents: A Pre-Registered Test

**arXiv ID:** 2607.06001 | [PDF](https://arxiv.org/pdf/2607.06001v1)

**作者:** Cheng Qian `[一作]` `[通讯]` (Independent Researcher), Cheng Qian (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在前沿语言模型 Claude Opus 4.8 上进行预先注册的实验，验证了耦合多智能体经济的信息理论容量区间（gap law、协同子模、熵上限、财富选择），并在相同平台上检验了均值场残差缩放定律，发现后者在所有测试规模下均不成立，表现为离散吸引子、阶跃响应与双稳态。

**💡 创新点**

首次将信息论预测直接应用于实际前沿 LLM 市场，并采用严格的预注册与缓存化方法实现完全可复现、可审计的实验；同时首次揭示 LLM 人口在激励下呈离散吸引子行为，否定了传统均值场的平滑响应假设。

**🔧 技术方法**

使用 Claude Opus 4.8 进行多轮 parimutuel 市场与 3 位比特世界的感知通道实验；对 20 名 LLM 代理进行基于梯度与控制奖励的 3×3 网格实验；采用信息论公式计算信息量、子模性、熵上限，并用自定义分析器实现机械化结果检验；所有模型调用均缓存至 SQLite 以保证确定性。

**📊 数据集**

实验数据来自自定义的三位公平比特世界（8 状态）和 16 片位点的简化经济景观；感知结构包括离散、重叠、克隆、噪声与 XOR 控制通道；N=12–20 的 LLM 人口在 8–18 轮动态过程中收集的位置、收益与信息报告。

**📈 对比分析**

结果通过与预先注册的容差带（gap law ≤0.05 nats，子模性 ≥-0.03 nats，熵上限 ≤0.02 nats，财富排序 ≥4/5）直接比较；gap law 在 46 毫 nats 内通过，子模性与熵上限均满足；但残差缩放定律在 0/9 条件下未进入线性响应域，导致 CAP 判定；整体性能在预期范围内，未出现显著偏差。

**⚠️ 局限性**

实验仅限单一模型家族（Claude Opus 4.8）与单供应商，规模较小（最多 20 代理），使用的经济模型为极简化市场与位点景观，未包含交易策略或进入退出；数据仅来自模拟，缺乏真实部署验证；预注册和缓存化虽提升可复现性，但对不同硬件/服务的跨平台迁移仍需验证。

---

## 43. i-EXAM: Instructable and Explainable Attack Connectivity Graph Modeler

**arXiv ID:** 2607.05888 | [PDF](https://arxiv.org/pdf/2607.05888v1)

**作者:** Rakesh Podder `[一作]` (Colorado State University), Indrakshi Ray `[通讯]` (Colorado State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 i-EXAM，一款基于 AI 规划的交互式工具，用于从网络扫描自动生成攻击连通图（ACG），评估不可渗透性和攻击难度两大安全度量，生成多样化的网络硬化策略，并通过 LLM 生成自然语言解释。

**💡 创新点**

核心创新在于：① 将 ACG 编译为 PDDL 规划模型，保证安全度量的形式化可靠性；② 引入多样化解法和基于 LLM 的解释生成，使 sysadmin 可直观理解不同硬化方案；③ 完全自动化的数据采集、模型构建和可视化流程，消除对 PDDL 及规划技术的人工依赖。

**🔧 技术方法**

使用技术包括：AI 规划框架 FastDownward（A* + LMCut）、top‑k 规划器、规划编译技术、LLM（llama‑3.1‑nemotron‑70b‑instruct）进行解释、网络扫描工具 Nmap/Wazuh/OpenVAS、JSON 数据存储、攻击连通图 ACG 表示法。

**📊 数据集**

主要数据集为：1）网络扫描输出（Nmap、Wazuh、OpenVAS）收集的主机属性与网络连通信息；2）CVE/漏洞信息来源于 NVD、ExploitDB；3）一个由 30 节点组成的测试网络作为案例评估基准。

**📈 对比分析**

对比方法：与 SPEAR 框架及传统攻击图工具进行功能对比；在 30 节点网络上评估规划求解时间，发现使用 LMCut 以及多样化搜索可将计算时间降低约 50%；展示了在不可渗透性和攻击难度两项指标下，i-EXAM 能快速给出最优硬化方案和对应解释。

**⚠️ 局限性**

局限性：① 规划求解器在大规模企业网络（数百节点）上的可扩展性仍待验证；② 目前硬化成本函数为手工设定，缺乏学习型成本模型；③ LLM 解释未通过用户实验评估其可读性与有效性；④ 依赖网络扫描的完整性和准确性，扫描漏报可能导致模型误差。

---

## 44. Taxlifier: Leveraging Disease Taxonomy for Enhanced Multi-Label Classification in Chest Radiography

**arXiv ID:** 2607.05628 | [PDF](https://arxiv.org/pdf/2607.05628v1)

**作者:** Mohammad S. Majdi `[一作]` (University of Arizona), Jeffrey J. Rodriguez `[通讯]` (University of Arizona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

针对胸部 X 光多标签分类，提出两种利用层级关系的改进方法：logit‑based 与 loss‑based，提升病理预测精度。

**💡 创新点**

创新点在于：①将层级信息直接嵌入损失函数的正则项；②在推理阶段通过加权 logit 调整概率，避免额外训练，兼顾效率与解释性。

**🔧 技术方法**

技术包括：深度 CNN（DenseNet121）+ 二进制交叉熵、层级正则化、基于 TPE 的超参数搜索、logit 加权后处理以及 ROC/AUC 评估。

**📊 数据集**

使用了三大公开胸 X 光数据集：CheXpert（224k 图像）、PADCHEST（160k 图像）和 NIH（112k 图像），共 18 类常见病理。

**📈 对比分析**

与基线（平面多标签）进行对比；在 Accuracy、AUC、F1 以及统计显著性指标上，logit‑based 提升 12–13% 以上，loss‑based 提升 11–24%，两者均显著优于基线。

**⚠️ 局限性**

局限包括：需要手工构建层级树，依赖标注一致性；对非层级结构数据不适用；在某些稀疏病理上提升不明显。

---

## 45. Scientific Code Search at Scale: A Multi-Domain Dataset and Benchmark

**arXiv ID:** 2607.05443 | [PDF](https://arxiv.org/pdf/2607.05443v1)

**作者:** Nishan Pantha `[一作]` (University of Alabama in Huntsville), Rahul Ramachandran `[通讯]` (NASA Marshall Space Flight Center)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了面向科学计算的软件检索基准，包括5,264个NASA SMD领域的GitHub仓库、219个专家编写的仓库检索查询、117,950个代码片段及119,720个对应查询；并对仓库和代码片段进行了多维文档清洗、主题提取、外部链接聚合等预处理。

**💡 创新点**

创新点在于：①首次针对科研软件制定仓库级与代码片段级的检索基准；②使用NASA五个Science Mission Directorate（Earth Science、Astrophysics、Planetary Science、Heliophysics、Biological & Physical Sciences）域的高质量数据；③引入专家编写的真实科研查询，覆盖工具发现、工作流程、数据访问与分析方法四大查询类型；④结合多源数据（NASA KG、ASCL、Science Discovery Engine等）和LLM辅助的域分类与README清理，显著提升检索语料质量。

**🔧 技术方法**

技术方法包括：LLM（GPT‑4.1‑mini）进行域分类与README去噪；Tree‑sitter解析七种编程语言的函数/类定义；BM25、all‑MiniLM‑L6‑v2、INDUS‑Retriever、Qwen3‑Embedding‑0.6B、SFR‑Embedding‑Code‑400M_R等词向量和密集检索模型；混合检索（Hybrid‑RRF、Hybrid‑Rerank）与跨域评估框架；基准格式兼容BEIR/MTEB，提供公开评测脚本。

**📊 数据集**

数据集包括：
• 5,264个已域标注且包含清洗后README与外部链接的GitHub仓库；
• 219个专家撰写的仓库检索查询；
• 117,950条从这些仓库抽取的代码片段（Python、C、C++、Java、JavaScript、Fortran、Matlab）；
• 119,720条对应查询（docstring、identifier、class/func 等）。
来源涵盖NASA Earth Observations Knowledge Graph、ASCL、NASA Science Discovery Engine、GitHub组织等多源。

**📈 对比分析**

与传统BM25、通用语义模型(all‑MiniLM)和领域专用模型(indus‑retriever)对比；仓库检索的MRR@10在Astrophysics达0.87、Earth 0.52、Planetary 0.22；混合检索提升40%以上；代码片段检索中，Qwen3‑Embedding‑0.6B在MRR@10 0.54、Recall@10 0.68、NDCG@10 0.58，显著优于BM25（MRR 0.18）和INDUS模型；identifier查询整体难度最高，MRR仅0.25。基准提供多维度评估指标（MRR、Recall、NDCG）。

**⚠️ 局限性**

局限性：①领域覆盖不均衡，Heliophysics和Biological & Physical Sciences仅有少量仓库；②仅使用英语查询和文档，缺少多语言支持；③仓库和代码持续演进，基准为快照，需定期更新；④专家参与仅覆盖三大领域，其他两领域缺乏足够查询；⑤LLM分类和清洗过程可能带来偏差，需人工复核。

---

## 46. Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents

**arXiv ID:** 2607.05764 | [PDF](https://arxiv.org/pdf/2607.05764v1)

**作者:** Mahmoud Hany `[一作]` (Syntheia Pty Ltd), Peter Naoum `[通讯]` (Syntheia Pty Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在法律文档问答中，评估将全语料注入替换为两种结构化检索模式（语义检索+rerank 与 NAVINDEX），并衡量它们在答案质量、token占比及美元成本上的表现。

**💡 创新点**

创新点包括：① 设计 NAVINDEX 结构化索引，显式编码跨引用、定义词关系并支持层级导航；② 通过闭式缓存交叉点公式分离 token 量化与成本收益；③ 在评估中使用位置控制的参考对齐 LLM 判定法，提供更严谨的答案质量比较。

**🔧 技术方法**

采用的大型语言模型（Claude、OpenAI 等）、向量检索（GTE、BGE 等）、交叉编码 rerank、布尔预过滤、层级导航、提示缓存及基于 token 的成本模型。

**📊 数据集**

使用了 15 篇交易性法律协议（含合同、增删改、附件等）构成的 6 份评测文档；20 题带有验证真值答案（其中 18 题与文档绑定，2 题为离谱控制）；DocNavBench 52 题开放搜索集合作为后续扩展数据集。

**📈 对比分析**

对比方法：每个问题两次回答，分别使用全注入与检索模式；LLM 判定者依据参考答案判断两回答的相等性；结果显示：语义检索+rerank 在 18 题中与全注入相当，token 下降 17.3×；NAVINDEX 在所有 20 题中与全注入相当，token 下降 1.61×、答案上下文缩小 56×、美元成本下降 25%。

**⚠️ 局限性**

限制：样本规模仅 20 题、仅单一评判者/模型，缺少节点标签难以直接测量检索召回；缓存热状态假设单次连续会话，异步使用场景成本可能不同；未评估跨文档依赖、复杂查询情形；评判误差约 13.3% 的“可能错误”比例。

---

## 47. CSTutorBench: Benchmarking Small Language Models as Tutors for Block-Based Programming

**arXiv ID:** 2607.05571 | [PDF](https://arxiv.org/pdf/2607.05571v1)

**作者:** H. Chad Lane `[一作]` (University of Illinois Urbana-Champaign), Bryson Kageler `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CSTutorBench基准，用以评估小型语言模型在VEX VR区块编程环境中的教学效果。

**💡 创新点**

提出了针对中学学生的情景式题目、8维教学性评分表以及LLM-as-judge混合评估流程，揭示模型家族与指令调优对教学质量的影响。

**🔧 技术方法**

采用LLM-as-judge（Claude Sonnet 4）、人机混合评测、prompt工程与情境化提示，比较多种4B–120B参数的小型语言模型。

**📊 数据集**

使用17个基于VEX VR Coral Reef Rescue的情景化问题集合，包含调试、迭代调试、优化、概念四类，并提供块级XML代码快照与评分指南。

**📈 对比分析**

对11个模型在两轮prompt（基础与改进）下进行评估，平均整体得分最高为Gemma‑4（31B）89%，其家族与指令调优比参数规模更能预测教学质量；改进prompt使10/11模型平均提升11.2个百分点。

**⚠️ 局限性**

基准规模小（17题）、单轮评估、缺乏真实学生互动、评判者一致性受限，且尚未验证自动评判器的可靠性。

---

## 48. Beyond Accuracy: How Humans Evaluate Legally Correct but Socially Controversial Legal Advice from Machines

**arXiv ID:** 2607.05680 | [PDF](https://arxiv.org/pdf/2607.05680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 49. NAVER LABS System Re-implementation for the IWSLT 2026 Instruction-Following Task

**arXiv ID:** 2607.05623 | [PDF](https://arxiv.org/pdf/2607.05623v1)

**作者:** Anand Kamble `[一作]` (Florida State University), Aniket Tathe `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 IWSLT 2026 受限条件下重新实现了 NAVER LABS 的三阶段指令跟随流水线，并将其迁移到 SeamlessM4T‑v2‑large 语音编码器和 Qwen3‑4B‑Instruct LLM，额外构造了 100k 语音指令合成数据；

**💡 创新点**

主要创新在于公开实现了完整三阶段训练流程、通过 Gemma 生成 100k 语音与文本双模指令样本，以及在 Stage 3 中联合微调投影器和 LoRA 适配器以实现跨模融合；

**🔧 技术方法**

采用了投影对齐、文本 LoRA 预训练、跨模融合三阶段训练、Transformer 编码器、LoRA 适配器、Gemma 生成合成数据以及 Qwen3‑4B‑Instruct LLM；

**📊 数据集**

使用了 CoVoST 2、EuroParlST、LibriSQA、NUTSHELL、YTSeg 等原始数据集，并通过 SeamlessM4T‑v2‑large 对 LibriSQA 进行机器翻译生成多语言 SQA；

**📈 对比分析**

在 MCIF 基准上与 SeamlessM4T‑v2‑large 及单投影模型对比，Stage 3 在 ASR WER 23.49、EN–ZH ST COMET 0.781、英语 SQA BERTScore‑F1 0.346 上取得最佳性能；Stage 2 在 1k CoVoST 2 文本子集上提升了 BLEU、COMET；

**⚠️ 局限性**

局限包括仅处理 15 秒以内的音频导致长句表现受限、跨语言 SQA 依赖机器翻译噪声、Stage 2 评测数据不与 MCIF 直接可比。

---

## 50. Recovering Cloud Microstructures with Cascaded Diffusion Inversion

**arXiv ID:** 2607.05637 | [PDF](https://arxiv.org/pdf/2607.05637v1)

**作者:** Hanan Gani `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了双阶段扩散反演超分辨率框架，提升多光谱云图的4倍分辨率，重点重建细微云结构。

**💡 创新点**

创新点在于将跨传感器对齐与高频细节恢复拆分为两个阶段；Stage 1 通过真实对齐样本学习稳健的跨传感器映射，Stage 2 则利用自监督内部降采样在高分辨率对齐数据上精细学习云纹理，显著避免过度平滑与伪影。

**🔧 技术方法**

采用扩散反演与逆扩散技术，结合感知损失、梯度保持损失及自监督内部降采样，构成两阶段训练流程。

**📊 数据集**

使用来自 UAE Rain Enhancement Program (UAEREP) 的 9k/1k seviri→viirs 及 2250/500 msg→mtg 的多光谱云图对，训练并验证模型。

**📈 对比分析**

与 SwinIR、StableSR、SinSR 等基线进行 PSNR、梯度保持比及感知距离比较；在 seviri→viirs 上获得 21.25 dB PSNR、梯度保持 1.06、感知距离 0.28；在 msg→mtg 上同样取得 24.0 dB PSNR、梯度保持 1.03、感知距离 0.29，优于所有基线。

**⚠️ 局限性**

局限在于仍受跨传感器时空失配影响，训练过程耗时且仅在两组卫星上验证；模型在更大尺度或不同传感器组合上的泛化能力尚待进一步评估。

---

## 51. Ethics and EU AI Act in Cases of Work Disability Risk and Alzheimer's Disease Risk Prediction

**arXiv ID:** 2607.05402 | [PDF](https://arxiv.org/pdf/2607.05402v1)

**作者:** Sami Andberg `[一作]` (University of Eastern Finland), Katja Saarela `[通讯]` (Eficode Group Ltd)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了两种医疗AI系统（工伤风险预测和阿尔茨海默病风险预测）的伦理与欧盟AI法分类，并评估了其高风险归类对研发与部署的影响。

**💡 创新点**

将医学研究伦理、AI伦理准则与欧盟AI法结合，对具体医疗AI案例进行风险分类与合规性评估，提出高风险系统需的合规流程与挑战。

**🔧 技术方法**

工伤风险预测采用NLP技术（AWD‑LSTM、ULMFiT），阿尔茨海默病预测采用深度学习技术（多模态融合、FNN）。

**📊 数据集**

工伤风险预测使用6000份芬兰医学记录（训练/验证），阿尔茨海默病预测使用BEGAD 2019眼动、认知与语音多模态数据。

**📈 对比分析**

主要通过对比伦理框架与欧盟AI法的风险类别进行评估，并未给出具体算法性能指标；强调高风险分类导致的合规负担与部署挑战。

**⚠️ 局限性**

仅评估伦理与法规层面，缺乏实际性能指标和临床验证；合规性评估缺乏细节，未涉及系统在真实环境中的效果与可行性。

---

## 52. Geometry-Aware Infrastructure-Anchored Denoiser for UWB Sensing and Work-Zone Reconstruction

**arXiv ID:** 2607.05449 | [PDF](https://arxiv.org/pdf/2607.05449v1)

**作者:** Weizhe Tang `[一作]` (University of Wisconsin-Madison), Bin Ran `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 GAIA，一种基于几何感知的 UWB 信号去噪框架，用于工作区边界重建。

**💡 创新点**

通过将潜在锚点布局预测与几何一致性距离投影结合，显著提升边界重建精度。

**🔧 技术方法**

使用 PoseMLP 预训练基座、双向 GRU 时序建模、布局头、GeoDist 几何投影、门控融合等神经网络模块。

**📊 数据集**

基于真实户外 UWB 数据集（同步 UWB、GNSS、IMU），并在其基础上校准的仿真压力测试环境。

**📈 对比分析**

与 Kalman、MLP、PoseKalman、PoseMLP 等基线比较，GAIA 在实际数据上整体 MSE 降低 18.4% 、多边形 IoU 提升 15.5%，在仿真中更显著提升。

**⚠️ 局限性**

依赖于已知车辆轨迹，无法直接解决轨迹估计问题；在锚点稀疏或极端 NLOS 情况下仍有性能下降。

---

## 53. SAMPLe: SAM-based Optimizer for Prompt Learning in VLMs

**arXiv ID:** 2607.05727 | [PDF](https://arxiv.org/pdf/2607.05727v1)

**作者:** Hossein Rajoli `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的基于Sharpness-Aware Minimization（SAM）的优化器SAMPLe，用于提升视觉-语言模型（VLM）在Prompt学习中的泛化能力。

**💡 创新点**

创新点在于通过在每一步迭代中动态平衡训练误差最小化与损失景观平坦化，利用梯度正交化约束实现对全梯度与批梯度的自适应对齐，从而在Prompt学习中兼顾性能与泛化。

**🔧 技术方法**

主要技术包括SAM、F-SAM、SAGM等改进的Sharpness-Aware优化框架，以及对梯度正交化与动态调整的实现；在训练中结合指数滑动平均估计全梯度、计算扰动半径ρ和学习率η，构造双目标损失。

**📊 数据集**

在11个公开图像分类数据集（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）以及4个ImageNet变体（V2、Sketch、A、R）上进行评估，涉及基准Prompt学习框架CoOp、CoCoOp、MaPLe、TCP、CoPrompt。

**📈 对比分析**

与SAM、F-SAM、SAGM以及基线Prompt学习方法相比，SAMPLe在所有测试设置（基-新类、跨数据集、跨域）中均能提升调和平均值（HM）和零样本跨域准确率，通常比最优基线提升1–2个百分点，且在保持原始模型性能的同时获得更稳健的泛化。

**⚠️ 局限性**

限制主要体现在：1）对扰动半径和正交化超参数的选择仍需经验调优；2）实验主要集中在图像分类任务，尚未验证在更大规模或多模态（如视频、文本）Prompt学习中的效果；3）相对传统SAM，SAMPLe的计算开销略高，需要额外的梯度分解与正交化操作。

---

## 54. Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving

**arXiv ID:** 2607.05399 | [PDF](https://arxiv.org/pdf/2607.05399v1)

**作者:** Nikita Agrawal `[一作]` (University of Bayreuth), Ruben Mayer `[通讯]` (University of Bayreuth)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大语言模型KV缓存压缩技术（KIVI、TurboQuant、SnapKV、CaM）进行统一、任务感知的基准测试，评估其对模型质量、吞吐量、首字节延迟和压缩率的影响；

**💡 创新点**

提出了多维度（任务质量与系统性能）统一评估框架，揭示压缩率并非预测端到端性能的可靠指标，并强调工作负载感知的KV压缩策略选择；

**🔧 技术方法**

使用量化（KIVI、TurboQuant）、剪枝（SnapKV）和合并（CaM）三类技术，结合Llama‑3.1‑8B‑Instruct与Mistral‑7B‑Instruct‑v0.3模型在LongBench任务上实现压缩；

**📊 数据集**

采用LongBench的六个数据集（HotpotQA、2WikiMQA、Qasper、MultiFieldQA_en、TriviaQA、MultiNews）以及NarrativeQA、GovReport、Qasper等长上下文任务作为评估基准；

**📈 对比分析**

通过比较压缩率、TTFT、吞吐量与任务分数，发现KIVI4在各任务中保持最稳定的质量，SnapKV在吞吐量上最优，CaM在部分问答任务中表现突出但对工作负载高度敏感，整体表明压缩技术需根据任务特性选择；

**⚠️ 局限性**

限制包括仅测试两种7‑8B模型、未涵盖更大规模或多GPU部署、TurboQuant在某些任务耗时过长、某些方法（如CaM）在不同上下文长度下压缩率不稳定，且未探索自适应动态压缩策略。

---

## 55. Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems

**arXiv ID:** 2607.06094 | [PDF](https://arxiv.org/pdf/2607.06094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 56. BitFair: A 12nm Bit-Serial CNN Accelerator with Learnable Early Termination and Adaptive Bit Ordering for Ultra-Low-Power XR Vision

**arXiv ID:** 2607.05445 | [PDF](https://arxiv.org/pdf/2607.05445v1)

**作者:** Ang Li `[一作]` (Delft University of Technology), Chang Gao `[通讯]` (Delft University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种 12nm 位串行 CNN 加速器 BitFair，具备可学习的位级早停和自适应位排序，针对 XR 低功耗实时视觉场景进行硬件实现与硅验证。

**💡 创新点**

创新点在于：①通过梯度可微的软阈值与温度退火实现每层可学习的早停阈值；②采用贪婪搜索自适应位排序，最大化早停机会并保持准确率；③结合量化感知训练与位串行数据流，首次在 12nm FinFET 工艺上实现并验证。

**🔧 技术方法**

技术手段包括：位串行计算、ReLU 触发的早停、可微软阈值与温度退火、贪婪位排序搜索、量化感知训练、输出驻留数据流、PE 阵列、AXI 互连、动态电压频率调节以及 FinFET 12nm 设计与后布局分析。

**📊 数据集**

使用的数据集为事件摄像头 N‑MNIST 与 DVSGesture，帧图像 MNIST、SVHN、VWW（人物检测）等多种轻量级视觉任务。

**📈 对比分析**

与传统 8‑bit bit‑serial、BitSET 以及多种 SNN 边缘视觉加速器对比，BitFair 在 N‑MNIST 达 97.7%、DVSGesture 96.5% 的准确率；速度提升 1.78×~2.12×；能效 117 BTOPS/W、0.07 pJ/SOP；面积 0.34 mm²，功耗 13.7 mW，推理延迟 0.12 ms，满足 XR 20 ms 预算。

**⚠️ 局限性**

局限性包括：早停收益受 ReLU 负激活稀疏度限制，稀疏度低时加速率趋近 1×；仅对硬零激活有效，对 GELU 等平滑激活适用性差；目前针对轻量级视觉任务，扩展至更大模型或其他应用仍需进一步研究。

---

## 57. Unique Insertion Error Patterns in Levenshtein's Reconstruction Problem

**arXiv ID:** 2607.06181 | [PDF](https://arxiv.org/pdf/2607.06181v1)

**作者:** Ville Junnila `[一作]` (University of Turku), Pavan Padavu Devaraj `[通讯]` (University of Turku)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文针对Levenshtein序列重构问题，研究了多重集和非多重集两种模型，在一次插入错误情况下推导了任何字母表大小q≥2、任意长度n≥1的最小通道数，并对该通道数所需的代码对进行完整分类；

**💡 创新点**

创新点在于证明多重集模型与非多重集模型在一次插入错误下所需通道数相同，给出了通道数的精确值ceil((n+2)/2)，并通过构造最极端词对（极大交集对）证明该界限可达；

**🔧 技术方法**

采用了组合学与插入向量的递推分析、Levenshtein式递归关系、插入向量构造与计数技巧，以及对单插入多重集交集的精细分类与比较；

**📊 数据集**

本文不使用任何实验数据集，全部在理论字母表Z_q上进行组合计数与枚举；

**📈 对比分析**

与Levenshtein原先的重构通道数比较，本文给出的多重集通道数界限在t=1时是最优的，且通过构造例子验证了界限可达；对于更高t的通道数给出了上下界，但仍明显不如实际值，表现为精度不足；

**⚠️ 局限性**

局限性在于仅完成了t=1的完全解答，对t≥2的通道数给出的是粗糙的上下界；非多重集模型在t≥2的情况尚未给出完整分析；未来研究需要进一步改进通道数的通用上界与下界，并探讨更高t值下的极大交集模式。

---

## 58. DeSeG: Decoupling Semantic Intent and Geometric Constraints for Physically Plausible Human-Scene Interaction

**arXiv ID:** 2607.05787 | [PDF](https://arxiv.org/pdf/2607.05787v1)

**作者:** Jiakun Li `[一作]` (Southern University of Science and Technology), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层次化框架 DeSeG，用于生成物理可行的中文场景交互运动；

**💡 创新点**

创新点在于将语义意图与几何约束显式解耦，并将可微潜在场物理约束嵌入扩散训练目标，实现无后处理的碰撞回避；

**🔧 技术方法**

主要技术包括残差条件变分自动编码器（CVAE）实现语义规划、交叉注意力对齐文本与局部体素、基于 DiT 的自回归扩散执行器以及可微排斥势能物理正则化；

**📊 数据集**

使用 Lingo 以及 TRUMANS 两大公开 HSI 数据集进行训练与评估；

**📈 对比分析**

与 Lingo、TeSMo、TRUMANS 等基线相比，DeSeG 在 FID、语义一致性、可多模态度、以及平均/最大场景穿透率上均优于同类方法，且在负约束基准 NC-Bench 上达到了最高 72.3% 的语义‑几何一致性；

**⚠️ 局限性**

局限性包括对稠密体素几何编码的依赖，在极度拥挤或抽象物体场景下效果受限，以及在需要精细语义‑几何同步的场景中解耦可能导致对细节的丢失。

---

## 59. Large Language Models Have Unreliable Understanding of Software Engineering Terminology

**arXiv ID:** 2607.06004 | [PDF](https://arxiv.org/pdf/2607.06004v1)

**作者:** Huzaifa Ejaz `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估六种大语言模型在判断 ISO/IEC/IEEE 24765:2017 术语定义是否正确方面的性能，并分析其推理过程与错误模式。

**💡 创新点**

首次通过对标准定义的结构化和语义化篡改进行系统实验，揭示 LLM 在术语理解中的“拒绝偏差”和推理失效，并提出新的理解模型理论。

**🔧 技术方法**

采用零样本链式推理提示、分类准确率（TPR、TNR）评估、手工注释推理质量、对比 reasoning 与 non‑reasoning 版本的 LLM。

**📊 数据集**

使用 ISO/IEC/IEEE 24765:2017 的 5,381 条标准定义，并构造 4,618 条结构化/语义化篡改定义。

**📈 对比分析**

对六个 LLM（GPT‑5.2‑R/N, GPT‑5‑Nano‑NR, Opus‑4.6‑R, Sonnet‑4.6‑NR, Gemini‑2.5‑Flash‑NR）计算 TPR/TNR、推理质量统计；发现 LLM 在识别错误定义时性能高达 89–96%，但在正确定义识别仅 16–75%，推理可提升正确性但亦易误判。

**⚠️ 局限性**

仅评估 ISO 24765 术语；使用单一提示模板；手工注释样本有限；未对新理论进行验证；结果可能不泛化至其他 LLM 或不同术语体系。

---

## 60. Depression Symptoms and Relational Patterns in 187k ChatGPT Histories

**arXiv ID:** 2607.05685 | [PDF](https://arxiv.org/pdf/2607.05685v1)

**作者:** Neil K. R. Sehgal `[一作]` (University of Pennsylvania), Sharath Chandra Guntuku `[通讯]` (University of Pennsylvania)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

分析了187,093条来自766名受试者的ChatGPT对话记录，结合其PHQ-8抑郁症状评分，研究高抑郁症状用户在主题、语言、情感、使用时段和响应风格上的差异；

**💡 创新点**

将ChatGPT视为一种非正式的支持基础设施，首次将抑郁症状水平与聊天机器人使用模式相连，揭示高症状用户更频繁的心理健康话题、晚间使用、情感语言及高披露支持请求，并警示仅凭语言特征无法用于临床筛查；

**🔧 技术方法**

采用确定性与词汇特征（如LIWC、1-gram、LDA）以及LLM生成的标签（通过GPT-5.4完成），结合统计检验（Welch、Cohen d、Benjamini‑Hochberg）以及正则化逻辑回归进行PHQ≥10预测；

**📊 数据集**

使用了来自Prolific的美国、英国、加拿大受试者自愿提供的ChatGPT历史记录（共766人，187,093段对话）以及其对应的PHQ-8问卷结果；

**📈 对比分析**

通过参与者加权对比PHQ<10与PHQ≥10组的使用、语言与响应差异，统计显著性均通过Welch检验与FDR校正；使用正则化逻辑回归对用户语言进行PHQ≥10预测，最高AUROC约为0.591，性能仅略高于随机，尚不足以用于临床筛查；

**⚠️ 局限性**

主要局限包括PHQ-8仅反映两周内症状而非诊断；研究为观察性，无法说明ChatGPT使用是否影响抑郁症状；GPT生成的标签未经过临床验证；样本为便利样本，主要来自英美加，可能不具代表性；语言模型的预测性能过低，不能用于自动筛查或临床决策。

---

## 61. Nested Episodic State Topology (NEST): A Graph-Theoretic Architecture of Cognitive States

**arXiv ID:** 2607.06055 | [PDF](https://arxiv.org/pdf/2607.06055v1)

**作者:** Ishant `[一作]` `[通讯]`, Ishant

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了NEST（Nested Episodic State Topology）这一基于图论的认知表示语言与操作工具包，明确了工作记忆与信念图的分层结构，并给出现象签名、任务实例化方案以及与现有认知框架的兼容映射。

**💡 创新点**

核心创新在于构建统一的递归图结构作为认知理论的共同底层，既保持了模块化的工作记忆与长期记忆分离，又通过可组合的操作工具包（激活、冲突目录、更新算子等）实现对多种认知现象的统一描述，并提供跨理论比较的可视化映射。

**🔧 技术方法**

主要技术手段包括递归节点与多类型加权图、图相似度匹配、激活函数与分布式注意力、冲突目录构建、图更新算子（Δ、U等）以及任务上下文与答案判定的图约束。

**📊 数据集**

本文为理论性工作，未使用任何实验数据集，所有定义均基于抽象的图论框架。

**📈 对比分析**

通过在变量级别对ACT‑R、Soar、CMC/Sigma、GWT等框架的映射，展示了NEST与这些模型的结构相容性；由于缺乏实现与实验，尚无性能评估。

**⚠️ 局限性**

主要限制包括：缺乏具体学习算法与实现细节；未在实验或仿真中验证理论预测；工作记忆与信念图的神经映射不明确；缺乏数据驱动的参数估计与实证支持。

---

## 62. Multi-Channel Spread-Spectrum Code Watermarking

**arXiv ID:** 2607.06009 | [PDF](https://arxiv.org/pdf/2607.06009v1)

**作者:** Soohyeon Choi `[一作]` (Singapore Management University), Yue Duan `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后期嵌入、训练无关、支持24位标识的多通道扩散光谱代码水印方案

**💡 创新点**

创新点在于将变量命名约定与八对等价代码模式两类独立通道结合，利用键控伪随机排列与Reed‑Solomon外码实现多位水印且具有形式化鲁棒性

**🔧 技术方法**

技术包括：键控伪随机排列、变量命名约定（四种风格）、八对语义等价结构转换、投票聚合与外层RS纠错

**📊 数据集**

使用 CodeNet 五个容量桶（共1,750个Python文件）以及 GPT‑4.1、Llama‑4 两个LLM生成的代码进行评估

**📈 对比分析**

与 ACW、SrcMarker‑Py、STONE 等后期和生成时水印做对比，在17类攻击（重命名、结构攻击、随机损坏等）下实现 100% 无攻击准确率、97.6%（8次重命名）/94.1%（10% 随机损坏）恢复率，误报率 ≤ 1/2²⁴，嵌入/检测均在 200 ms 内完成

**⚠️ 局限性**

主要局限：在 Level 4 的完整 LLM 重写攻击下完全失效；结构攻击对小文件易破坏；仅在 Python 上验证，需扩展到其他语言

---

## 63. When AI Classifies: What Counts as Public Administration?

**arXiv ID:** 2607.05420 | [PDF](https://arxiv.org/pdf/2607.05420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 64. Agents with Feelings? Personality and Emotion in Multi-Agent Software Teams

**arXiv ID:** 2607.05659 | [PDF](https://arxiv.org/pdf/2607.05659v1)

**作者:** Yunyan Ding `[一作]` (University of California, Irvine), Iftekhar Ahmed `[通讯]` (University of California, Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在多代理大型语言模型（LLM）团队中注入基于人格与情绪的配置，系统评估其对代码生成与代码审查任务的性能与协作行为的影响。

**💡 创新点**

创新点在于提出了一套整合五大人格维度、基本情绪与软件工程工作风格的心理学驱动人格框架，并首次探讨共享与混合人格配置在多代理SE系统中的效能差异。

**🔧 技术方法**

使用了四种指令调优LLM（Qwen2.5‑1.5B、Qwen2.5‑32B、Llama‑3.1‑8B、Mistral‑Small‑24B），结合角色分工工作流与自定义persona prompt，实现人格与情绪的注入。

**📊 数据集**

实验数据集包括代码生成的LiveCodeBench‑lite v6子集（282个问题）与代码审查的Hydra‑Reviewer/CodeReview‑New数据集（377个补丁）。

**📈 对比分析**

采用pass@1（代码生成）与BLEU‑4（代码审查）对78种团队配置（54共享+24混合）进行评估，发现最佳共享配置在pass@1上提升高达11.35个百分点，最佳混合配置在6/8模型–任务场景中进一步提升0.3%–3.0%；整体而言，混合配置往往优于单一共享配置。

**⚠️ 局限性**

局限性包括仅覆盖两种SE任务和四种模型，混合配置仅从高性能共享配置中组合，且人格与情绪仅通过提示实现，未验证真实情感；成本与性能之间的权衡仍需更深入研究。

---

## 65. Collaborative Multi-Agent Testing for Emergent Failure Discovery in Autonomous Driving Systems

**arXiv ID:** 2607.06078 | [PDF](https://arxiv.org/pdf/2607.06078v1)

**作者:** Ruizhen Gu `[一作]` (Queen's University Belfast), Mehrdad Dianati `[通讯]` (Queen's University Belfast)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了协同多智能体测试框架CREAD，整合感知扰动生成、元模态验证与搜索协调，实现对自动驾驶系统的失败发现；

**💡 创新点**

创新点在于将ADS测试转化为协同过程，通过共享黑板与调度器实现多角色（扰动生成、行为验证、协调）之间的闭环交互，而非传统的单一生成‑评估循环；

**🔧 技术方法**

采用LLM（z.ai GLM‑5 Turbo）生成感知扰动、黑板架构、元模态验证、强化学习/启发式调度及仿真执行等技术；

**📊 数据集**

在轻量级交通仿真平台（Highway、Roundabout）上进行评估，使用基于模拟的感知降级模型；

**📈 对比分析**

与非协同多智能体和单智能体基线比较，协同配置在Highway环境下失败率提升至52/100（约4.6倍），在Roundabout保持竞争力；整体显示协同显著提升失败发现；

**⚠️ 局限性**

局限包括感知抽象、低保真仿真、启发式车辆策略、ODD受限扰动；未在高保真CARLA或学习型控制器上验证。

---

## 66. Repeated Contention Scheduling: A Novel Resource Allocation Algorithm Toward 6G Vehicular Networks

**arXiv ID:** 2607.06103 | [PDF](https://arxiv.org/pdf/2607.06103v1)

**作者:** Alexey Rolich `[一作]` (University of Rome Sapienza), Andrea Baiocchi `[通讯]` (University of Rome Sapienza)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于多轮冲突竞争的 NR‑V2X sidelink 资源分配算法——Repeated Contention Scheduling (RCS)，取代传统的 Semi‑Persistent Scheduling (SPS) 与 Dynamic Scheduling (DS)；

**💡 创新点**

核心创新在于消除长期预留，采用反馈驱动的多轮低频率竞争机制，既实现完全分布式操作，又显著降低持久碰撞与延迟，且可与未来 6G 车辆通信系统无缝兼容；

**🔧 技术方法**

使用 5G NR numerology、OFDM 子载波信号、频域冲突检测、多轮轮次控制以及 SDR（USRP）实现软硬件原型；

**📊 数据集**

实验数据集基于仿真场景（不同负载系数、两种 numerology）与 SDR 现场测试（3–10 台 USRP 设备、10^5 次冲突循环）构成；

**📈 对比分析**

与标准 SPS（P_pers=0 与 0.8）和 DS 进行对比；结果显示 RCS 在 PRR、平均 PIR、平均 AoI 及 AoI 违例概率方面均优于 SPS/DS，尤其在高负载（a≥0.9）下表现最为突出；

**⚠️ 局限性**

局限包括：仅针对周期性流量验证；未考虑隐藏节点、非理想信道衰落、波束成形等真实场景；实验中出现无赢家事件（P_nowin）导致成功率略低；未来工作需扩展到非周期性、隐藏终端与多赢冲突等更复杂环境。

---

## 67. A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

**arXiv ID:** 2607.05615 | [PDF](https://arxiv.org/pdf/2607.05615v1)

**作者:** Nima Eshraghi `[一作]` (Vanguard Group), Fanny Chevalier `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了通过在稀疏位置对大语言模型的残差流注入稀疏自编码器（SAE）导向激活来实现行为控制的两种方法——Stochastic Token Steering (STS) 与 Stochastic Block Steering (SBS)，并在无微调的情况下实现毒性抑制与情感引导。

**💡 创新点**

创新点在于用概率门控稀疏干预代替传统全令牌干预，证明稀疏注入已可获得大部分行为转移且更好保持流畅度，并发现行为转移受累计信号剂量控制。

**🔧 技术方法**

使用稀疏自编码器提取行为相关特征，随后在中间层残差流中按Bernoulli门控随机插入方向向量，配合归一化、激活截断与重复惩罚等轻量级正则。

**📊 数据集**

在RealToxicityPrompts（毒性）和GoEmotions（情感）数据集上进行实验。

**📈 对比分析**

与无干预、全令牌干预和基于提示的基线对比，STS在p=0.5时可恢复95%毒性减弱效果，在情感任务上高达80%，并在低p下超过提示基线，整体性能优于SBS且仅需注入更少总信号。

**⚠️ 局限性**

局限在于仅验证两类行为（毒性抑制与情感引导），依赖已训练的SAE与对比语料，评估基于分类器且未覆盖其他属性；对解码策略和多属性控制的泛化未知。

---

## 68. From Textural Counterpoint to Feature Encoding: A Multi-Dimensional Machine Representation Study of Haydn's "The Lark" Integrating Electroacoustic Analysis

**arXiv ID:** 2607.05902 | [PDF](https://arxiv.org/pdf/2607.05902v1)

**作者:** Yakun Liu `[一作]` (Shenyang Conservatory of Music), Xiaonan Li `[通讯]` (Shenyang Conservatory of Music)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对海顿《鹤鸣》第一乐章进行跨学科分析，利用DAW频谱、瞬态、响度等声学特征与符号信息相结合，构建事件时间戳和角色感知编码，为深度时间序列模型提供低层次多维输入；

**💡 创新点**

首次提出“角色感知编码（Role‑Aware Encoding）”与“事件基时间戳（Event‑based Timestamps）”的组合，打破传统网格化时序与机械决定论，兼顾微节奏弹性与声学角色识别；

**🔧 技术方法**

使用DAW频谱分析、瞬态检测、RMS/LUFS响度测量，基于无监督跨模态标注生成角色向量，结合深度时间序列网络（RNN/Transformer）与RTNeural低延迟推理框架；

**📊 数据集**

仅使用海顿《鹤鸣》四声部前8小节的录音与对应乐谱，未构建大型多曲风数据集；

**📈 对比分析**

目前未完成端到端模型训练与生成质量对比，论文仅通过声学特征映射展示方法可行性，缺乏与传统PPQ网格化模型在生成精度、延迟等指标的客观对比；

**⚠️ 局限性**

局限于古典四重奏文本，难以推广到多样化曲风；缺少大规模多曲风实验与完整模型验证；技术上无法模拟情感体验与身体感知，角色编码仅为结构化推断，非真正情感交互；

---

## 69. EeveeDark: A Binary Neural Framework for Low-Light Video Enhancement via Event-Guided Sensor-Level Fusion

**arXiv ID:** 2607.06217 | [PDF](https://arxiv.org/pdf/2607.06217v1)

**作者:** Onur Eker `[一作]` (Hacettepe University), Aykut Erdem `[通讯]` (Koc University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为EeveeDark的低照度视频增强框架，融合传感器级RAW帧和事件流，并采用二值神经网络实现高效推理。

**💡 创新点**

创新点在于首次将事件驱动的时间精度与RAW图像的空间细节结合，并在同一二值网络中实现跨模态特征融合和事件引导的门控机制，从而在保持时序一致性的同时大幅降低计算量。

**🔧 技术方法**

采用分布感知二值卷积、二值编码器、轻量级特征融合块、循环时间位移编码器、事件引导跳跃门控和双向时间解码器等技术，并使用RPReLU激活、Charbonnier损失。

**📊 数据集**

主要使用LLRVD（合成事件的RAW视频）、HUE（真实事件的RAW视频）、SDE和SDSD（RGB-事件对比）等数据集进行训练与评估。

**📈 对比分析**

与BBCU、BRVE等二值基线以及ShiftNet、FloRNN、EvLight等全精度模型对比，EeveeDark在PSNR/SSIM/ST‑RRED上均优于二值方法，且仅比全精度模型低约30‑50倍的FLOPs（1.66G vs 32.87G），实现了显著的性能‑效率折中。

**⚠️ 局限性**

在光子计数极低且运动稀疏时，事件流稀疏且RAW噪声较大，导致模型仍会出现残留噪声和色彩失真。

---

## 70. EvalLoop: A Methodology for Evaluation-Driven Iterative Improvement of Business AI Systems

**arXiv ID:** 2607.05638 | [PDF](https://arxiv.org/pdf/2607.05638v1)

**作者:** Kenneth Benavides `[一作]` (Robert Half), Danti Chen `[通讯]` (Robert Half)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 EvalLoop 方法，将评估从单纯的模型选择转变为迭代式的诊断与改进循环，并在销售情报简报生成任务中进行验证。

**💡 创新点**

创新点在于：①将指标按业务维度分组，揭示不同故障的根因；②引入失败模式分类，将失效原因细化；③构建结构化迭代工作流和终端人类审查门槛；④打包可复用的 playbook、agent 规范和模板。

**🔧 技术方法**

使用技术包括：多维度指标分组、LLM 评判（多供应商面板）与确定性规则评估、失败模式分类、实验追踪（MLflow）以及基于 LLM 的 Prompt 优化工具（如 DSPy）。

**📊 数据集**

数据集为 100 条合成的客户账户事实集（含结构化键值对），以及 10 个不同模型（3 家供应商）和 18 条评估指标，构成 5 个业务维度。

**📈 对比分析**

通过三轮迭代（基线、配置实验、Prompt 修正）进行对比，采用配对 t 检验和效应量报告；最佳模型从 82.6% 提升至 94.6%（整体提升 12pp），在内容准确性、合成能力等维度获得显著提升；终端人类门槛进一步将评估样本减少 94%。

**⚠️ 局限性**

局限性包括：仅在单一任务与合成数据上验证，可能存在过拟合；LLM 评判可靠性受限；失败模式分类和维度划分仍依赖专家定义；终端人类审查仅为单评审且样本有限，未检验多评审一致性。

---

## 71. Memory in the Loop: In-Process Retrieval as ExtendedWorking Memory for Language Agents

**arXiv ID:** 2607.05690 | [PDF](https://arxiv.org/pdf/2607.05690v1)

**作者:** Yusuf Khan `[一作]`, Carlo Lipizzi `[通讯]` (Stevens Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在语言代理循环中将内存检索移入每一步的可行性与效果。

**💡 创新点**

证明内存访问速度（而非网络延迟）决定是否能在每一步检索，并将检索频率视为可调设计维度；采用扩展心智的平行原则作为延迟阈值。

**🔧 技术方法**

使用 in‑process vector store（vxdb）与本地静态嵌入器，GPT‑5 系列模型，CoALA 框架，以及循环保护器与 per‑turn RAG 基线。

**📊 数据集**

采用六轮行程规划对话（5 条约束）和安全警报重复任务（12 条重复），以及随机种子工作负载。

**📈 对比分析**

通过在不同检索频率与存储延迟上设置对照，测量任务成功率、冗余操作数、存储操作延迟；结果显示在微秒级存储下每步检索可实现 3.6–4.8/5 记忆回顾，且冗余操作从 0 增至 7.2/12 随存储延迟升高。

**⚠️ 局限性**

仅评估两类任务与单一窗口大小，读写策略固定，网络嵌入仍是瓶颈，且未考虑长期存储生命周期与复杂匹配策略。

---

## 72. Slack and Budget Breaking in Threshold Team Production

**arXiv ID:** 2607.06197 | [PDF](https://arxiv.org/pdf/2607.06197v1)

**作者:** Benjamin Marsh `[一作]` (Sei Labs), Alejandro Ranchal-Pedrosa `[通讯]` (Sei Labs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究阈值任务在存在冗余机会（slack）时的协作激励，设计了一种可预付的统一及时团队赏金机制并证明其在所有非负可测赏金方案中是最优的；

**💡 创新点**

首次给出针对协同破坏（co‑sabotage）的最小赏金上限，推导出完整的最优赏金公式B⋆=Δ+1⁻¹(L+R₁⁺-(Δ+1)f)⁺，并提出排除递增奖励（exclusion ratchet）以及对最终时隙可见性不可消除性的理论极限；

**🔧 技术方法**

利用机制设计与博弈论的强均衡框架，结合超几何抽样与Chernoff界、路径上最优性证明，构建可测赏金与恢复费的完整分析模型；

**📊 数据集**

无实验数据集，本工作基于严格的数学推导与概率分析；

**📈 对比分析**

通过理论证明和最优性定理展示该赏金方案在所有可行设计中的最差情况都被最优预算覆盖，性能表现为无可再优化空间；

**⚠️ 局限性**

局限在于无法消除最终时隙的可见性优势，需依赖其他协议（如封闭、commit‑reveal或阈值加密）来处理；同时假设票据可公开验证且不考虑网络延迟的异质性等现实因素。

---

## 73. SCOPE: Leveraging Subgoal Critiques for Code Generation

**arXiv ID:** 2607.05810 | [PDF](https://arxiv.org/pdf/2607.05810v1)

**作者:** Yueke Zhang `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SCOPE，一种基于 Lean 定理证明器的子目标批评器，用来为 LLM 代码生成提供结构化、可解析的修复建议，指导编码器逐步修正程序。

**💡 创新点**

创新点在于将证明器的子目标拆分习惯迁移到代码生成领域，形成可直接用于代码修复的三元结构（子目标、缺口分析、鲁棒性检查），并通过稠密语义奖励与稀疏执行奖励结合的过程对齐强化学习，显著压缩修复搜索空间。

**🔧 技术方法**

使用了 DeepSeek‑Prover‑V2‑7B（作为批评器），Qwen3‑Coder‑30B（作为编码器），QLoRA+4‑bit 量化的微调与GRPO（Group Relative Policy Optimization）强化学习，配合 dense/qualitative 语义奖励与 sparse 运行正确性奖励。

**📊 数据集**

训练数据来源于 LiveCodeBench V1–V3 的批评训练样本，评估数据使用 LiveCodeBench V6（175 题）和 BigCodeBench‑Complete（Hard）148 题。

**📈 对比分析**

通过与 Coder‑Only、Self‑Refine、Reflexion、SCOPE（未训练）和 SCOPE（SFT）等基线在同一编码器设置下对比，SCOPE（Full）在 LiveCodeBench V6 上 pass@1 达到 39.4%（比 Reflexion 36.6% 提升 2.8%），在 BigCodeBench‑Hard 上 42.6%（比 Reflexion 36.5% 提升 6.1%）。在 Easy/Medium/Hard 难度以及多类算法任务上均表现出显著优势。

**⚠️ 局限性**

局限性包括：仅针对 Python 代码片段任务，未实现完整形式化验证；生成的子目标可能不完全覆盖真实意图；评估仅覆盖两类基准，缺乏多语言、多仓库或长期编码场景的验证；可能存在对训练数据的过拟合风险。

---

## 74. The Granularity Paradox: How Temporal Disaggregation Inflates In-Sample Fit and Compounds Out-of-Sample Error

**arXiv ID:** 2607.05450 | [PDF](https://arxiv.org/pdf/2607.05450v1)

**作者:** Hugo Moreira `[一作]` `[通讯]` (Instituto Universitário de Lisboa), Hugo Moreira (Instituto Universitário de Lisboa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文系统评估不同时间粒度对多步预测性能的影响，并揭示了“粒度悖论”——更细粒度虽然提升了样本量和训练拟合，但由于递归误差放大导致外样本预测误差恶化。

**💡 创新点**

创新点在于提出粒度悖论理论，首次用TPFE等累计误差指标揭示递归误差传播，发现线性回归在不同粒度下表现稳定，而LSTM呈U形误差曲线，并提出共识-分歧诊断方法评估模型在不同粒度下的误差传播。

**🔧 技术方法**

使用了十种预测模型（基准、统计ARIMA、Holt‑Winters、XGBoost、LSTM、N‑BEATS 等）以及多种时间粒度（年、季、月、双周、周、日），通过滚动窗口交叉验证计算训练 R²、测试 R²、RMSE、MAE、TPFE 等指标。

**📊 数据集**

采用 13 年的葡萄牙公共采购 IT 服务合同数据（CPV 72）进行实验，按年、季、月、双周、周、日六种粒度重新采样。

**📈 对比分析**

通过 8 折滚动窗口实验比较模型在各粒度下的表现，结果显示：递归模型在日粒度下 TPFE 最高（Holt‑Winters 425.85%），线性回归 TPFE 维持在 16% 左右，LSTM 在日粒度达到 4.35% 的最佳点，而 N‑BEATS 在细粒度下保持稳定；共识-分歧诊断进一步揭示了递归误差放大的机制。

**⚠️ 局限性**

局限性在于仅使用单一行业（IT 服务）单条序列，缺乏对其他业务领域的验证，且实验为单一随机种子结果，未提供置信区间或显著性检验。

---

## 75. Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty and Adaptive Complexity Control

**arXiv ID:** 2607.05522 | [PDF](https://arxiv.org/pdf/2607.05522v1)

**作者:** Gaoxiang Jia `[一作]` (Advanced Micro Devices, Inc.), Xinlei Wang `[通讯]` (University of Texas at Arlington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种渲染感知的贝叶斯3D高斯喷溅框架，利用正态逆Wishart后验跟踪几何不确定性，并将其用于主动视角选择与区间校准。

**💡 创新点**

创新点在于将NIW后验与渲染器产生的统计摘要结合，形成可解释的几何不确定性，并加入截断Dirichlet过程实现可调的复杂度先验，实现了端到端的贝叶斯推理与主动决策。

**🔧 技术方法**

使用正态逆Wishart后验更新、截断Dirichlet过程、渲染器导出的摘要统计、混合闭式与近似推理、采样重渲染、深度网络训练等技术。

**📊 数据集**

使用13个标准基准场景（Mip‑NeRF 360、Tanks and Temples、Deep Blending）并在16→32主动视角任务中进行评估。

**📈 对比分析**

与传统点估计、三成员深度集成、PPU、3DGS‑MCMC等基线对比，主动视角任务中NIW取得+0.453 dB PSNR提升，区间覆盖误差降低17倍，训练成本仅为标准的1.6%/三倍。

**⚠️ 局限性**

局限性包括：在重建质量上提升仅为+0.03 dB，推理仅部分闭式，颜色/不透明度更新仍近似；实验仅覆盖固定视角、静态场景，未验证动态、稀疏或OOD场景下的表现。

---

## 76. The GenAI Skill Bypass: Mapping Divergent Pathways of University Students and Staff AI Literacy

**arXiv ID:** 2607.05411 | [PDF](https://arxiv.org/pdf/2607.05411v1)

**作者:** Eduardo Oliveira `[一作]` (University of Melbourne), Mohammed Saqr `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对一套28项生成式AI素养自评工具在158名大学学生、学术人员和专业职员中的心理测量属性进行了Rasch与古特曼排序分析，揭示了不同群体的技能获取路径；

**💡 创新点**

创新点在于首次实证证实生成式AI素养并非线性、统一的发展轨迹，发现学生存在“技能绕行”——先掌握高阶创作后再学习基础概念，挑战了一刀切的课程设计；

**🔧 技术方法**

使用了Rasch Partial Credit Model（部分信用模型）与古特曼排序（Guttman ordering）对题目难度与受试者能力进行测度，并用Spearman相关分析比较不同群体的难度排序；

**📊 数据集**

采用了来自澳大利亚墨尔本大学的在线调查数据，共158名参与者，其中包含98名学生、34名学术职员和26名专业职员；

**📈 对比分析**

通过绘制古特曼图和计算Spearman相关系数进行比较，结果显示学生与学术职员之间的相关系数仅为0.188（弱相关），而学术职员与专业职员之间为0.524（中等相关），验证了各群体的技能发展顺序差异；

**⚠️ 局限性**

局限性包括样本量有限、仅来自单一高校、采用自评数据可能与实际表现不一致、以及由于子样本不足未能进行差异项功能（DIF）分析。

---

## 77. DebugTracker: Lightweight Process Evidence for Classroom Debugging

**arXiv ID:** 2607.05871 | [PDF](https://arxiv.org/pdf/2607.05871v1)

**作者:** Jiatong Liu `[一作]` (Monash University), Yongqiang Tian `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款名为 DebugTracker 的 VS Code 插件，能够在学生调试任务中记录可审计的过程证据，并生成时间线、Markdown 报告以及代码快照，方便教师评估调试思路而非仅凭最终提交代码。

**💡 创新点**

核心创新在于将评估（无指导）与训练（有指导）模式分离；采用轻量级 JSONL 事件日志记录调试过程，同时提供可视化时间线和报告，既保障隐私又保证可追溯性；实现语言无关的事件捕获，支持多种编程语言。

**🔧 技术方法**

技术实现基于 TypeScript，利用 VS Code 扩展 API、Debug Adapter Protocol、终端命令监测、JSONL 事件流、Markdown 生成以及可选的 OpenAI 接口进行 AI 辅导。

**📊 数据集**

使用跨语言（Python、TypeScript、Java）三种调试任务数据集进行验证，并配备 16 项自动化测试脚本与 11 条人工验证案例，覆盖 Windows、macOS 与 Linux 环境。

**📈 对比分析**

通过 16 个自动化检查（如会话 ID、命令检测、模式匹配、报告生成等）与 11 个手工验证（如 VSIX 安装、训练模式反馈、图像证据等）来验证实现，测试结果显示在所有支持的操作系统和语言上均能稳定记录预期事件，性能符合日常课堂使用需求。

**⚠️ 局限性**

局限性包括：仅捕获 VS Code 级别的事件，无法提供完整屏幕录像或键盘敲击记录；缺乏自动错误定位与解释功能；对未集成终端或自定义测试命令的项目支持有限。

---

## 78. No Subspace to Track: Non-Identifiability and Optimizer State in Low-Rank Training

**arXiv ID:** 2607.05872 | [PDF](https://arxiv.org/pdf/2607.05872v1)

**作者:** Noel Thomas `[一作]` `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence), Noel Thomas (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 GaLore 这类低秩梯度优化器的核心假设——梯度子空间是慢速漂移且可被追踪——进行实证检验，测量了子空间在刷新周期内的旋转幅度、可识别核心大小、谱结构及平均化效果，并基于这些发现提出并验证了通过在每次刷新时传输优化器状态和缩短 β₂ 记忆的改进方案。

**💡 创新点**

创新点在于揭示 GaLore 依赖的“慢漂移子空间”实际上几乎不存在，仅存在约 39 个可识别方向；证明平均化无法恢复子空间；并提供并验证了两种有效的改进：一是传输第一/第二动量以跟随子空间旋转，二是将 β₂ 从 0.999 缩短到 0.99 来减少第二动量的记忆误差。

**🔧 技术方法**

采用同步步长对比、分批子空间估计、主角距离（chordal distance）测量、梯度谱分析、Davis–Kahan 误差界、N^{-1/4} 平均缩放实验、理论证明（状态传输最优性、二阶动量不传输的子空间不一致性）等技术手段。

**📊 数据集**

主要使用 Pythia（160M、1B）语言模型、WikiText 训练数据；还对 GPT‑2、Qwen2.5、Llama‑3、Vision Transformer（ViT‑B/CIFAR‑10）等预训练检查点进行验证。

**📈 对比分析**

通过在相同训练步骤、学习率、warm‑up、T 等配置下比较 perplexity，发现 LDAdam（传输+错误反馈）在默认 β₂=0.999 下达 18.73 perplexity，明显优于 GaLore（22.07）；将 β₂ 缩短至 0.99 亦可提升 1–2 perplexity；平均化子空间对 perplexity 影响不显著。

**⚠️ 局限性**

局限性包括：仅针对显式刷新子空间的优化器；实验主要聚焦 Pythia/WikiText，未涵盖更大规模或不同任务；平均化实验使用单一种子；状态传输效果与 LDAdam 规则混合，未分离两者贡献；β₂ 缩短对不同硬件/配置敏感。

---

## 79. Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design

**arXiv ID:** 2607.06175 | [PDF](https://arxiv.org/pdf/2607.06175v1)

**作者:** Alexander Rombach `[一作]` (German Research Center for Artificial Intelligence), Nijat Mehdiyev `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过强化学习（GSPO）和多维奖励函数，提升LLM在自然语言到BPMN流程模型生成任务中的语法、可读性与语义质量。

**💡 创新点**

创新点在于系统评估多维奖励设计对RL优化的影响，揭示等权重奖励和负惩罚对不同模型架构及初始化策略的决定性作用，并提供基于BEF4LLM自动评估的完整实验框架。

**🔧 技术方法**

采用了Group Sequence Policy Optimization（GSPO）与LoRA参数高效微调，以及基于38项BEF4LLM指标的多维奖励函数。

**📊 数据集**

使用公开的BPMN文本-模型对数据集（包括Camunda BPMN for Research等）共105条评估样本和约1.5K条训练样本。

**📈 对比分析**

通过对比SFT-only与RL训练、不同奖励权重、惩罚设置及模型初始化的48种配置，使用配对置换检验与Bonferroni校正，结果显示RL在等权重奖励下能显著提升语法与可读性（>0.11点）且输出一致性提高6倍，且奖励设计对性能影响可与RL本身相当。

**⚠️ 局限性**

局限在于仅使用单个生成样本、单周期RL训练、固定3:1:1权重、仅评估两种LLM架构，且语义质量改进有限，未来需探索动态奖励权重、长周期训练及更广泛的数据与模型。

---

## 80. EcoVision: AI-Powered Drone Imaging for Salt Marsh Vegetation Monitoring and Dominance Mapping

**arXiv ID:** 2607.06105 | [PDF](https://arxiv.org/pdf/2607.06105v1)

**作者:** Innocent Onyenonachi `[一作]` (Keele University), Nadia Kanwal `[通讯]` (Keele University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了 EcoVision 端到端 UAV‑AI 流程，利用低空 UAV 高分辨率 RGB 影像完成盐沼草本的语义分割、对象级分类与 2×2 m 网格优势评分，提供可解释的生态指标。

**💡 创新点**

创新点在于将 Transformer‑基语义分割（SegFormer）、ConvNeXt 目标分类与格网优势评估整合为一套可复现的管道；仅使用 RGB 数据即可实现细粒度物种识别，并与传统地面矩形调查结果对齐，填补了传统监测方法在尺度与可解释性上的空白。

**🔧 技术方法**

采用 SegFormer‑B5 变压器语义分割、ConvNeXt 卷积分类、连通域提取、基于 2×2 m 网格的优势计算，配合 Albumentations 数据增强、PyTorch+HuggingFace 实现、FP16 混合精度训练与推理。

**📊 数据集**

使用两类公开图像（iNaturalist、GBIF）约 1,012 张（两物种各 500+ 张）经人工标注后增强至 2,300 张作为训练集；验证集为 DJI Mavic4 Pro 采集的 0.5–1 cm GSD RGB UAV 影像。

**📈 对比分析**

与野外矩形调查直接对比：SegFormer mIoU 0.557、像素准确率 0.962；ConvNeXt 分类准确率 99.0%、F1 0.99；优势评分 MAE < 8%，格网主导一致率 97.6%。相比仅做分割或多源数据的先前方法，EcoVision 在物种识别、优势评估与可解释性方面显著提升。

**⚠️ 局限性**

局限性包括：仅用 RGB 受光照、潮汐及湿度变化影响；在密集叠加区域出现边界误差；模型依赖大量人工标注，扩展到更多物种时规模受限；对不同地区、季节及光照条件的泛化仍需进一步验证。

---

## 81. Unlearnable Faces: Privacy Protection Surviving Extraction Pipeline

**arXiv ID:** 2607.05996 | [PDF](https://arxiv.org/pdf/2607.05996v1)

**作者:** Byunghoon Oh `[一作]` (Chung-Ang University), Jaewoo Lee `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为LPID的局部化、可微裁剪+缩放耦合的扰动生成方法，使在面部识别攻击者执行裁剪+缩放提取后，照片仍保持不可学习的隐私保护。

**💡 创新点**

创新点在于将攻击者的提取流程（裁剪+缩放）嵌入可微分的扰动优化中，并将扰动局部化至被裁剪的面部区域，聚焦能量于被保留的频带，从而在未知身份下保持强大的防护。

**🔧 技术方法**

采用min‑min双层优化、可微裁剪+缩放模型T、PGD与SGD交替训练，并结合频域分析与因子化消融验证鲁棒性。

**📊 数据集**

使用CASIA‑WebFace人脸与Places365背景合成的小脸+场景图像，以及对齐的CASIA‑WebFace 224×224图像进行实验。

**📈 对比分析**

与UE、REM、TUE、LSP、Segue等现有可学习示例方法对比，LPID在裁剪+缩放攻击下将识别准确率降至≈4–7%，明显优于其他方法，并在不同识别模型、JPEG压缩、模糊等变换下保持低误差。

**⚠️ 局限性**

局限在于对上传时JPEG压缩敏感，鲁棒性会显著下降；且方法需预知裁剪框尺寸，对多尺度或更复杂预处理步骤的适应性仍待提升。

---

## 82. The Cathedral and the Bazaar of Software Vulnerabilities: From the NVD to the CNAs

**arXiv ID:** 2607.05670 | [PDF](https://arxiv.org/pdf/2607.05670v1)

**作者:** Siqi Zhang `[一作]` (Vrije Universiteit Amsterdam), Mengyuan Zhang `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了NVD与多家CNA在CVSS评分上的差异，量化了外部和内部的分歧，并分析了根因；

**💡 创新点**

首次提出统一的divergence度量与自分歧评估框架，并对CNA类型、发布时间、CWE、描述长度等因素进行定量与定性分析；

**🔧 技术方法**

采用逻辑回归、线性回归、集合/等价类度量、CVSS向量比较，并结合访谈获取根因；

**📊 数据集**

使用191,009条CVE及72,122条CNA评分数据，覆盖1999–2025年NVD与所有公开CNA；

**📈 对比分析**

与传统单源评估相比，发现73%的CNA在Pairwise视角下存在差异；模型在不同来源间迁移时准确率可下降约40%；

**⚠️ 局限性**

受限于样本量不均、缺乏NVD状态信息、仅基于CVSS v3.1、假设CNA-only即为NVD完整等，导致一致性评估可能偏高。

---

## 83. CMDR: Contextual Multimodal Document Retrieval

**arXiv ID:** 2607.05927 | [PDF](https://arxiv.org/pdf/2607.05927v1)

**作者:** Ryota Tanaka `[一作]` (NTT, Inc.), Kyosuke Nishida `[通讯]` (NTT, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新型的多模态文档检索任务与基准，要求模型利用跨页上下文信息来检索相关页面；

**💡 创新点**

核心创新在于：①使用滑动窗口联合编码多页并按页拆分的上下文化多模态嵌入框架；②引入Contextual Multimodal Contrastive Learning（CMCL），通过同一文档内的in‑chunk与in‑document hard negatives平衡上下文建模与页面区分；

**🔧 技术方法**

技术包括：滑动窗口联合编码（chunk‑then‑split）、多向量Late Interaction（LI）交互、InfoNCE对比损失与CMCL改进、LoRA微调、Hierarchical Token Pooling提升效率；

**📊 数据集**

使用自建的CMDR基准数据集，包含800条人工标注查询（四类）和255篇长文档（平均183.5页）以及约40k个训练查询‑页面对；

**📈 对比分析**

在CMDR基准上，与多类非上下文检索模型（文本、通用多模态、文档检索）对比，使用ColPali/ColQwen等骨干后加CMCL，nDCG@5平均提升约16点（ColPali）/20点（ColQwen），且在所有查询子类与文档类型上均优于基线；

**⚠️ 局限性**

局限性包括：①基准规模仍有限（800查询），难以覆盖所有情况；②依赖单向注意力的LVLM，无法充分利用后续页面信息；③联合编码的上下文长度受内存/计算限制，难以处理极长文档。

---

## 84. ArtisanCAD: An Industrial-Level CAD Agent with Expert-Grounded Knowledge Distillation

**arXiv ID:** 2607.05750 | [PDF](https://arxiv.org/pdf/2607.05750v1)

**作者:** Yunhan Xu `[一作]` (Peking University), Shiyi Chen `[通讯]` (Eastern Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个基于CAD-IR中间表示的工业级CAD生成代理，能够从专家记录中提炼可复用的参数化技能，并通过技能检索、IR实例化、CATIA-MCP后端执行及多视角视觉反馈循环，生成可编辑、可生产的B-Rep模型。

**💡 创新点**

创新点在于将专家操作日志转化为可执行的CAD-IR，实现了知识蒸馏为可复用技能；同时将CAD-IR作为统一的过程层，既支持从模糊文本到完整建模的桥接，也支持对工业工作流的直接调用；引入多视角视觉反馈细化IR。

**🔧 技术方法**

使用了大型语言模型（如Codex/ChatGPT）进行宏日志解析与IR生成，CATIA-MCP后端执行，基于Mimo-v2.5-Pro的视觉反馈和IR重写，Chamfer距离和IoU评估。

**📊 数据集**

使用Text2CAD公开数据集（中级文本提示）和四个真实汽车零件的专家宏记录与工程笔记。

**📈 对比分析**

在Text2CAD上，使用CAD-IR在中级提示下将CD平均值从14.83降至9.88，IoU从0.614提升至0.646；在工业零件上，通过专家技能能成功生成长序列操作并产出可编辑B-Rep，而无技能情况下失败。

**⚠️ 局限性**

局限性包括对CATIA-MCP后端的依赖，无法直接迁移到其他CAD系统；需人工收集专家宏日志，知识蒸馏成本较高；对完全新颖的设计意图仍难以从模糊描述自动推断完整流程。

---

## 85. GaussFusion: Towards Multimodal 3D Gaussian Pretraining

**arXiv ID:** 2607.05906 | [PDF](https://arxiv.org/pdf/2607.05906v1)

**作者:** Zhixuan You `[一作]` (Xi'an Jiaotong University), Hainan Luo `[通讯]` (Wuhu HIT Robot Technology Research Institute Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了GaussFusion，一种多模态自监督预训练框架，用于学习3D高斯表示的可迁移特征。

**💡 创新点**

创新点在于将图像和文本监督融入掩码高斯重建，并提出基于高斯显著性指导的多尺度空洞掩码（GSHM），实现对局部结构与全局语义的双重学习。

**🔧 技术方法**

技术手段包括掩码自编码器、跨模态对齐、可学习的图像/文本对齐token以及GSHM掩码策略。

**📊 数据集**

使用ShapeSplat作为预训练数据集，并在ScanObjectNN、ModelNet10/40、ShapeNetPart以及few-shot ModelNet40等下游任务上进行评估。

**📈 对比分析**

与基线Gaussian‑MAE相比，GaussFusion在ScanObjectNN PB‑T50‑RS上提升3.85%，在ModelNet40上提升0.61%，在分类、分割与少样本学习任务中整体表现更优。

**⚠️ 局限性**

局限性包括多模态监督仅来自渲染图像和类别级文本，缺乏更丰富的语义标注；未来可扩展至更大场景级高斯数据和更细粒度文本描述。

---

## 86. Segmentation before Answering: Pixel Grounding for MLLM Visual Reasoning

**arXiv ID:** 2607.05798 | [PDF](https://arxiv.org/pdf/2607.05798v1)

**作者:** Yake Wei `[一作]` (Renmin University of China), Di Hu `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SegAnswer 方法，将视觉推理过程中的区域放大操作从传统的矩形框改为像素级分割掩码，实现对目标区域更精细、准确的定位；

**💡 创新点**

创新点在于：①以像素级分割为基础的“先分割后回答”框架，消除了矩形框带来的背景冗余和语义干扰；②三阶段训练策略（像素 grounding → 多模态交互式微调 → 强化学习推理），使模型在保持分割能力的同时能够自然地将分割结果嵌入推理流程；

**🔧 技术方法**

技术手段包括：使用 Qwen2.5‑VL‑7B 作为基线，加入 MLP 投影层与 SAM 2.1 的 mask 解码器进行像素级分割；采用 LoRA 进行低秩微调；使用多模态交互式 SFT 让模型学习在对话中插入分割操作；以及基于 DAPO 的强化学习，使用准确性奖励与格式奖励驱动推理策略；

**📊 数据集**

数据集：①像素 grounding 训练使用 RefCOCO/RefCOCO+/RefCOCOg/RefClef、ReasonSeg、ADE20K、COCOStuff、Mapillary Vistas、PACO‑LVIS、PASCAL‑Part 等多种分割数据；②多模态交互式训练采用 VisualCOT（438k VQA 对，带框标注但不使用框监督）；③强化学习训练使用 ViRL39K；评测数据集涵盖高分辨率视觉细节任务（V*、HR‑Bench 4K/8K）、通用感知（MMBench、VisuLogic、MMVP）、幻觉评测（POPE、Hallusionbench）以及分割基准（RefCOCO/RefCOCO+/RefCOCOg）；

**📈 对比分析**

与 LLaVA‑OneVision‑9B、Qwen2.5‑VL‑7B、Pixel Reasoner、DeepEyes 等现有 MLLM 视觉推理方法对比，SegAnswer 在高分辨率任务（如 V*）取得 86.4 分（较基线显著提升），在 HR‑Bench 4K/8K、MMBench、VisuLogic 等多项基准上持续优于对手；在分割任务中亦优于多种专门的分割方法，表明其 pixel‑grounding 能力可靠；

**⚠️ 局限性**

局限性：①模型仍需依赖 SAM 等分割器的推理速度，导致推理时间较长；②像素分割的准确性在极端遮挡或细小目标场景下可能下降，影响后续推理；③三阶段训练过程复杂，调参成本高，易受超参数设置影响；

---

## 87. PERSONAJUDGE: Simulating Individual Human Preference Judgments with Evaluator-Specific Demonstration Data

**arXiv ID:** 2607.05742 | [PDF](https://arxiv.org/pdf/2607.05742v1)

**作者:** Zeyu He `[一作]` (Pennsylvania State University), Alex C. Williams `[通讯]` (AWS AI Fundamental Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用多维度评估者数据（标签、界面轨迹、回溯推理）通过LLM进行个体评估模拟的方法，并系统评估其效果。

**💡 创新点**

首次将评估者的行为轨迹与推理文本作为演示输入，用in‑context learning实现个体化模拟，发现回溯推理是最有价值的辅助信号，并展示该方法能显著优于基准。

**🔧 技术方法**

采用LLM-as-Judge框架，利用四种演示类型（J、J+IT、J+RR、J+IT+RR）和四大模型（Claude‑3.5‑Sonnet‑V2、Claude‑3.7‑Sonnet‑V1、DeepSeek‑R1、Amazon Nova Premier）进行实验，采用两轮二分类策略进行模拟。

**📊 数据集**

使用Anthropic的Helpful & Harmless（HH）对话对比评估数据，共1400条对话，32名训练评估者，收集了4200条判断（含界面轨迹和回溯推理）。

**📈 对比分析**

与随机、Base Judge、cross‑evaluator control、oracle consensus等基线对比，平均提升9.9个百分点（Harmless）和5.8个百分点（Helpful）。最佳配置（Claude‑3.5‑Sonnet‑V2、8‑shot、J+RR）达58%准确率，较Base Judge提升约12%。

**⚠️ 局限性**

仅在对话对比任务中验证，评估者样本有限；interface telemetry效果不佳，可能需更高层次摘要；未尝试微调或检索式增强；无法完全捕捉高层次不确定性与多样性；存在隐私与误用风险。

---

## 88. Whose fairness? Structural concentration in AI bias research

**arXiv ID:** 2607.05574 | [PDF](https://arxiv.org/pdf/2607.05574v1)

**作者:** Abhash Shrestha `[一作]` (Center for Artificial Intelligence (AI) Research Nepal), Tek Raj Chhetri `[通讯]` (Center for Artificial Intelligence (AI) Research Nepal)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 AI 偏见研究领域的文献进行系统性计量与语义分析，探究其地理、机构、合作网络、引用分布及主题结构，并构建了可持续监测的交互式地图。

**💡 创新点**

首次从宏观层面量化 AI 偏见研究的结构集中度，揭示了“通用公平”领域的高度集中与跨领域传播，并提供了持续跟踪该结构变化的可视化工具。

**🔧 技术方法**

采用检索+书目计量学、协作网络分析、以及基于 Sentence‑BERT、UMAP 与 HDBSCAN 的语义聚类技术。

**📊 数据集**

收集并处理了 692 篇符合条件的论文元数据（来源：IEEE Xplore、ACM DL、Scopus、ScienceDirect、Engineering Village、FAccT 等），并利用 OpenAlex API 获取引用与作者信息。

**📈 对比分析**

通过比较各国/机构的出版量、Gini 系数、协作频率、国内外引用比例以及语义聚类的 ARI/NMI 等指标进行对比；结果表明美国在“通用公平”领域占主导、引用影响高度集中，且各主题间的聚类分布差异显著。

**⚠️ 局限性**

局限性包括：样本主要来自英文、北半球数据库，可能低估非英文或地区期刊的贡献；领域划分仍有重叠且受人工判断影响；书目计量指标无法完整反映公平知识的社会技术动态。

---

## 89. Perceived System Predictability: Scale Development and Application

**arXiv ID:** 2607.05674 | [PDF](https://arxiv.org/pdf/2607.05674v1)

**作者:** Hendrik Schuff `[一作]` (University of Stuttgart), Ngoc Thang Vu `[通讯]` (University of Stuttgart)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并验证了一种新的六项感知系统可预测性（PSP）量表；

**💡 创新点**

创新点在于将感知可预测性细分为有效、认知和随机三维，并引入不确定性理论为其理论基础；

**🔧 技术方法**

使用了量表开发流程（专家评审、认知访谈）、结构方程模型、广义加性模型等统计技术；

**📊 数据集**

实验数据来自两项 MTurk 受试者研究，分别使用形状分类任务和基于 SentiWordNet3 的情感分类器；

**📈 对比分析**

通过一维与三维结构比较，量表内部一致性高（α=0.96），并在主观可预测性与客观预测正确性间揭示独立性，说明量表具有良好构念效度；

**⚠️ 局限性**

局限在于仅在受限任务和透明模型下验证，未覆盖复杂黑箱 AI，且对噪声与解释效果的普适性需进一步研究。

---

## 90. Fusion or Confusion? Potential and Challenges in Fusion of Onboard Sensors and V2X Data in Cooperative Perception

**arXiv ID:** 2607.05889 | [PDF](https://arxiv.org/pdf/2607.05889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 91. PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models

**arXiv ID:** 2607.05441 | [PDF](https://arxiv.org/pdf/2607.05441v1)

**作者:** Lorenzo Molfetta `[一作]` (University of Bologna), Gianluca Moro `[通讯]` (University of Bologna)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 PORTS 的检索器训练方法，利用冻结 LLM 的困惑度信号和对偶对比损失，优化工具检索器，使其在工具调用任务中更好地匹配 LLM 的偏好。

**💡 创新点**

创新点在于：①引入基于赔率比的偏好优化（Odds Ratio Preference Optimization）来直接对检索器的选择概率进行调节；②将 LLM 的生成概率与检索概率通过 KL 散度对齐；③结合对比损失，强制正负样本嵌入方向性，提升对语义相似但无关工具的区分能力。

**🔧 技术方法**

核心技术包括：预训练语义编码器（RoBERTa、BGE）、冻结的 LLM 作为代理（Codestral‑22B、Llama‑3‑Groq 等）、对偶对比学习、赔率比偏好损失、KL 对齐损失、硬负样本采样与周期性嵌入更新。

**📊 数据集**

评估数据集包括六个工具检索基准：ToolBench、API‑Bank、APIBench、BFCL‑v2、ToolE、Octopus‑v2；每个数据集被拆分为训练/测试，并构造了见过/未见过工具的对比实验。

**📈 对比分析**

与基线检索器和 RePlug 进行对比。PORTS 在 Recall@1/3/5 和 NDCG 上相较于冻结基线提升 47–72% 的召回率，且相对 RePlug 提升约 15–20% 的召回/NDCG，表现出显著的性能优势。

**⚠️ 局限性**

局限性包括：对工具文档（docstring）质量高度依赖，文档不完整会削弱效果；需要频繁调用冻结 LLM 以产生指导信号，导致额外的计算与内存开销；在资源受限场景下的可扩展性受限。

---

## 92. Agentic AI for IPoDWDM Network Lifecycle Automation: An MCP-Enabled Architecture

**arXiv ID:** 2607.05958 | [PDF](https://arxiv.org/pdf/2607.05958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 93. Benchmarking the Robustness of Autonomous Driving to Environmental Illusions: A Lane Perception Perspective

**arXiv ID:** 2607.05783 | [PDF](https://arxiv.org/pdf/2607.05783v1)

**作者:** Tianyuan Zhang `[一作]` (Beihang University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了LanEvil++基准，系统评估环境错觉对车道感知模型的鲁棒性。

**💡 创新点**

创新点在于首次构建涵盖14种环境错觉的高质量数据集，并提出跨模态错觉防御框架MIDA。

**🔧 技术方法**

技术手段包括CARLA仿真生成高保真错觉图像，AAM++和PAT++的对抗增强，以及多模态对抗训练。

**📊 数据集**

使用的数据集为LanEvil++，包含90,292张图像、1,596段视频及41,855个问答对，覆盖94个可编辑3D场景。

**📈 对比分析**

与多种主流车道检测和视觉语言模型对比，环境错觉平均导致5.27%准确率、10.49%F1下降；MIDA在LD和ADVLM上分别提升约4.2%和3.8%。

**⚠️ 局限性**

局限在于仅针对车道感知和部分ADVLM，未覆盖3D检测等任务，且实验多在仿真或小规模真实车辆上，缺乏大规模工业级验证。

---

## 94. Self-Supervised Implicit CEST Reconstruction via Physics-Informed Lorentz Encoding

**arXiv ID:** 2607.06132 | [PDF](https://arxiv.org/pdf/2607.06132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. Evaluating calibrated refusal and safe usefulness in dual-use biology settings

**arXiv ID:** 2607.05462 | [PDF](https://arxiv.org/pdf/2607.05462v1)

**作者:** Edwin H. Wintermute `[一作]` (American Wetware), Kenny Workman `[通讯]` (LatchBio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 BioSecBench-Refusal 基准，用于评估 AI 代理在生物安全相关任务中的拒绝行为，包括合法（Routine）和隐藏风险（Red‑Team）两类任务。

**💡 创新点**

创新点在于：① 将双向任务（正向执行与元评估）相结合，能够同时衡量模型发现深层威胁的能力和对合法研究的拒绝阈值；② 提供可调节的拒绝率评估框架，帮助开发者在安全与实用之间做权衡；③ 通过 API 过滤与模型自身拒绝分离，揭示当前安全机制的弱点。

**🔧 技术方法**

采用多模型/多工具组合（Claude、OpenAI、Anthropic、Google 等）进行推理，设计了直接框架和元评估框架，利用工具调用与数据文件分析进行任务执行；统计分析包括 API 拒绝率、模型拒绝率、整体拒绝率，并与 biosafety 级别、任务类型等维度相关联。

**📊 数据集**

数据集共 107 条评估任务，61 条来源于公开文献的 Routine 任务，46 条为虚构的 Red‑Team 高风险情景；每条评估均附带 DNA 序列、蛋白结构、实验协议等文件，供模型在推理过程中分析。

**📈 对比分析**

在 16 种模型-工具配置上进行三次重复实验，共 5,136 次推理。结果显示 Routine 任务的拒绝率普遍高于 Red‑Team（平均 7–74% vs 1–62%），API 过滤占主导；在元评估框架下，允许模型进行更深层推理时，部分模型对 Red‑Team 的正确拒绝率略有提升，但样本受限。

**⚠️ 局限性**

局限性包括：① API 过滤过多导致可供模型评估的样本极少；② 不能区分模型自身拒绝与外部安全过滤的细粒度行为；③ 评估集中缺乏 BSL‑4 任务，难以全面覆盖高风险场景；④ 缺少统一的拒绝阈值标准，导致结果解读受主观因素影响。

---

## 96. Breaking Spurious Correlations via Generative Randomization and Cross-Variant Self-Supervised Learning

**arXiv ID:** 2607.05850 | [PDF](https://arxiv.org/pdf/2607.05850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. Controlling Tool Use with Heading-Specific Activation Steering

**arXiv ID:** 2607.05790 | [PDF](https://arxiv.org/pdf/2607.05790v1)

**作者:** Yuqi Chen `[一作]` (UC Santa Cruz), Chenguang Wang `[通讯]` (UC Santa Cruz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在使用外部工具时的内部决策过程，提出通过在工具调用的 heading‑anchor 位置提取激活 steering 向量，并在五个开源模型与三个任务域（Math、Time、Intention）上对该向量进行激活加法和正交化干预，进而实现对工具调用行为的双向因果控制。

**💡 创新点**

创新点在于首次将激活 steering 应用于完全非参数化的工具使用决策，证明即使工具不具备模型权重中的线性表示，该向量仍能实现有效的行为控制；同时揭示不同工具类型在内部表示上具有显著差异，工具使用的激活空间呈现非线性、模态分布，而非传统概念的单一方向。

**🔧 技术方法**

技术上使用对比激活差值（contrastive activation subtraction）构造 steering 向量，随后在 heading‑anchor 位置执行激活加法（add）或正交化（orthogonalize）来抑制或增强工具调用；层选择通过遍历各层得到工具调用曲线的拐点；还利用余弦相似度、Jaccard 交集以及线性可分辨性评估向量的几何属性。

**📊 数据集**

数据集主要是 SMART benchmark 的 Math、Time、Intention 三个子任务，使用 GPT‑4o 生成带有 heading 结构的训练示例来提取 steering 向量；实验对比基线模型（工具提示）以及多轮工具使用情况，并在 Llama‑3.1‑8B 上做额外的格式迁移与重命名控制。

**📈 对比分析**

通过比较工具调用平均次数、任务准确率、缺失细节恢复、摘要意图等指标，发现激活加法在 Math 域显著降低工具调用且准确率仅略降，表明许多工具调用为冗余；在 Time 与 Intention 域抑制工具调用导致准确率大幅下降；正交化则相反，工具调用上升但任务表现不提升；这些结果说明工具使用控制的有效性与任务性质紧密相关，且随着模型规模增大对工具调用的抗干预能力提升。

**⚠️ 局限性**

局限性包括：对工具使用的激活空间几何分析表明缺乏清晰线性结构，因果控制机制尚未阐明；干预高度依赖 heading 结构，跨格式迁移效果有限；实验仅覆盖三种工具与五个模型，缺乏更广泛的泛化验证；多轮依赖关系使得激活干预的理论解释受限。

---

## 98. Performance Optimization and Comparative Analysis of Generative AI Models on Advanced Accelerators

**arXiv ID:** 2607.05400 | [PDF](https://arxiv.org/pdf/2607.05400v1)

**作者:** Amitash Nanda `[一作]` (University of California San Diego), Debashis Sahoo `[通讯]` (University of California San Diego)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对生成式AI模型（如TinyLlama、Llama、Gemma及扩散模型）在多种加速器（Perlmutter、Expanse、Voyager中的A100、Gaudi1/2/3）上的性能进行了系统评估，提出并验证了基于敏感度的混合精度后训练量化（PTQ）框架，并对LLM微调在不同代Gaudi上的训练时间、吞吐率与精度进行了对比；同时还比较了扩散超分模型在Gaudi与NVIDIA GPU上的训练速度。

**💡 创新点**

创新点在于：①引入敏感度感知的混合精度PTQ框架，使用统一阈值稀疏化实现按层分布的位宽分配；②在Gaudi多代硬件上进行实测量化与微调，并将DeepSpeed ZeRO3、FSDP、PyTorch DDP和Flash Attention 等技术组合使用；③在同一实验平台上对LLM、Diffusion模型进行跨加速器性能基准。

**🔧 技术方法**

使用技术包括：后训练混合精度量化（INT8/FP8/BF16）、Intel Neural Compressor、Optimum-Habana、DeepSpeed ZeRO3、PyTorch FSDP、DDP、Flash Attention、分组量化、按层敏感度评估。

**📊 数据集**

使用数据集：WikiText-2、HellaSwag、BoolQ、BoltMonkey心理学问答数据、PySM生成的星系尘埃图像、Diffusion模型训练的4万余张图像。

**📈 对比分析**

比较方法：在相同任务下测量压缩率、吞吐率（tokens/sec 或 samples/sec）、评估时间和精度（PPL、Accuracy），在A100与Gaudi2上实现3-6倍压缩且精度损失≤2%；微调在Gaudi1/2/3上表现为线性缩短时间，Gaudi3最快；Diffusion模型在Gaudi2比Gaudi1快4×，与H100差距缩小。

**⚠️ 局限性**

局限性：单卡内存限制导致无法训练70B模型；量化时需手工调节阈值，精度波动；不同硬件间的软件栈差异影响可比性；实验规模受资源限制，仅覆盖部分模型与任务。

---

## 99. Akashic: A Low-Overhead LLM Inference Service with MemAttention

**arXiv ID:** 2607.05708 | [PDF](https://arxiv.org/pdf/2607.05708v1)

**作者:** Yang Liu `[一作]` (Xiaohongshu Inc.), Junhao Hu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低开销的LLM推理服务Akashic，专门针对长寿命代理系统的上下文管理；

**💡 创新点**

创新点在于将记忆压缩、跨块一致化与查询感知检索统一到块级别，并通过模型驱动的相关性匹配与硬件-软件协同的存储放置来显著减少跨块检索的读放大与延迟；

**🔧 技术方法**

核心技术包括MemAttention（块级增量压缩与跨块协同）、模型驱动的语义匹配与检索、基于块的物理共置与垃圾回收的存储管理，以及在vLLM框架中实现的轻量级控制路径；

**📊 数据集**

实验数据集涵盖四种典型长序列工作负载：LoCoMo、SWE‑Bench、BrowseComp 与 WebArena；

**📈 对比分析**

与全上下文、Mem0、MemGAS、MemGPT、RMM等基线相比，Akashic 在准确率上提升最高可达10.2点，吞吐量提升最高可达1.21×，在并发服务下可持续请求速率提升高达1.88×；

**⚠️ 局限性**

局限性包括对模型推理的额外依赖（需额外的模型调用以生成块级压缩与检索），对块大小与检索预算等超参数敏感，以及在极端高密度或极长上下文场景下仍可能面临压缩效果有限、检索开销增加等挑战。

---

## 100. Generalized altitudes and their bounds

**arXiv ID:** 2607.06187 | [PDF](https://arxiv.org/pdf/2607.06187v1)

**作者:** Hana Dal Poz Kourimska `[一作]` (University of Potsdam), Mathijs Wintraecken `[通讯]` (Inria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文对单形体任意相对面之间的最短距离定义为通用高度，并给出了几何解析式。

**💡 创新点**

创新点在于将传统的顶点高度推广到任意相对面，提供了显式角度与代数表达式，建立了高度与Gram行列式、外积的联系。

**🔧 技术方法**

主要技术是多线性代数工具，包括外积（广义叉积）与Gram行列式，利用正交补与最短距离的性质推导公式。

**📊 数据集**

未使用具体数据集，研究为理论性分析。

**📈 对比分析**

通过构造特殊四面体示例讨论下界常数的最优性，证明常数需严格小于1；并给出高度与通用高度的下界关系。

**⚠️ 局限性**

局限在于下界常数并未给出最优取值，且仅给出理论推导，缺乏数值实验或应用验证。

---

## 101. Cross-Contextual Vision-Language Adaptation with LoRA for Personalized Severe Adverse Event Detection in Clinical Wound Monitoring

**arXiv ID:** 2607.05625 | [PDF](https://arxiv.org/pdf/2607.05625v1)

**作者:** Aditi Naiknaware `[一作]` (San Diego State University), Salimeh Sekeh `[通讯]` (San Diego State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于冻结BiomedCLIP的双流交叉LoRA多模态框架，实现伤口图像、临床文本和伤口描述的融合，并在此基础上构建时间感知的OOD检测方案，用于自动伤口监测和个性化严重不良事件检测。

**💡 创新点**

1) 在冻结的VLM上实现双流交叉LoRA融合，允许临床语义与视觉描述在低秩空间中互相映射；2) 设计四种模态相似度融合的OOB评分并加入面积重加权的时间漂移惩罚；3) 用数据驱动的ID文本库和自适应阈值实现无标签的SAE识别。

**🔧 技术方法**

低秩适配LoRA、双流文本编码、交叉注意机制、BiomedCLIP预训练权重、跨模态OOB分数（扩展T‑QPM）、面积变化时间惩罚、联合损失（ID、协方差一致性、时间漂移）。

**📊 数据集**

SmartBoot DFU 12周随机对照试验数据，包含eKare inSight 3D系统拍摄的伤口图像、临床变量（年龄、HbA1c、UT分级等）和随访伤口面积测量。

**📈 对比分析**

与DPM、TQPM和LoCoOp等基线在同一患者级划分下比较。该方法在AUROC 0.729、FPR95 0.490、ID准确率0.937上优于基线，尤其在异常检测的FPR下降显著，证明跨模态与时间信息整合的有效性。

**⚠️ 局限性**

仅基于单一试验数据，ID文本库固定，难以随访随时间变化；未在更大多样化伤口人群验证；缺乏在线患者级自适应机制。

---

## 102. LEGATO 2: Toward Multimodal Sheet Music Recognition and Understanding

**arXiv ID:** 2607.05769 | [PDF](https://arxiv.org/pdf/2607.05769v1)

**作者:** Guang Yang `[一作]` (University of Washington), Noah A. Smith `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一套名为 Legato 2 的端到端光学音乐识别（OMR）管线，能够按系统（staff）逐步识别乐谱图像，并生成包含嵌入文本（标题、作曲家、注释）的符号记谱（ABC）。

**💡 创新点**

创新点：
- 采用系统级分割与自回归 VLM，显著提升长文档（多页）识别能力；
- 引入“系统级 ABC”作为中间表示，方便逐系统生成与后续转换；
- 开发字节回退（byte‑fallback）ABC 分词器，实现对嵌入文本的完整识别；
- 将 OMR 输出作为外部上下文提升前沿视觉‑语言模型在乐谱理解任务上的表现。

**🔧 技术方法**

技术手段：YOLOv8 中分割模型；Vision‑Language 模型（基于 Legato 1 结构）进行自回归识别；规则‑基础 ABC 转换器；字节回退 BPE 分词器；在多任务（识别+理解）框架下使用 GPT‑5 / Gemini 作为评估工具。

**📊 数据集**

使用的数据集包括：PDMX‑Synth（训练/验证/测试）、OpenScore String Quartets（Rendered & Camera）、OpenScore Lieder（Rendered & Camera）、IMSLP Piano Scores、MusiXQA、SSMR‑Bench；并在多页面评估中使用完整 OpenScore Lieder 集合。

**📈 对比分析**

比较方法：在单页和多页识别任务中与 Legato 1、Audiveris、Gemini 3.1 Pro 等基线对比；在嵌入文本识别中与 Audiveris、Gemini、PaddleOCR 对比；在乐谱理解任务中与无上下文、Legato 1 与 Legato 2 上下文对比。结果显示 Legato 2 在 OMR‑NED、字符错误率、MusiXQA G‑Acc 和 SSMR‑Bench Accuracy 上均显著优于基线，尤其在多页长文档和嵌入文本识别方面表现突出。

**⚠️ 局限性**

限制与挑战：
- OMR 仍存在误识别，尤其是复杂排版、交叉竖线等视觉干扰导致错误；
- 嵌入文本识别仅覆盖标题、作曲家、注释，未处理歌词等更大文本；
- 训练数据主要为合成乐谱，可能导致对真实扫描图像的泛化受限；
- 字节回退分词器在极稀有字符上仍可能出现细粒度错误；
- 对于极长文档的上下文截断仍会在边缘出现性能衰减。

---

## 103. Population-Level Profiling of DSM-5 Depressive Symptoms Among Self-Reported ADHD and ASD Users on Twitter: An Exploratory Study Using Advanced NLP and Statistical Analysis

**arXiv ID:** 2607.05626 | [PDF](https://arxiv.org/pdf/2607.05626v1)

**作者:** Muhammad Rizwan `[一作]` (University of Ljubljana), Jure Demšar `[通讯]` (University of Ljubljana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过两阶段 NLP 管线分析 792 位自报 ADHD 与 ASD 诊断的 Twitter 用户的抑郁症状表达，构建九项 DSM‑5 症状用户特征并比较两组的差异。

**💡 创新点**

创新点在于将零射 NLI 过滤与 Fine‑tuned MentalRoBERTa 多标签分类结合，并通过跨阈值 Bootstrap 稳定性评估，系统性揭示 ADHD 与 ASD 之间抑郁症状语言侧重的群体差异。

**🔧 技术方法**

采用零射 NLI 进行抑郁相关性预筛，使用在 Reddit 语料上 fine‑tuned 的 MentalRoBERTa 进行多标签症状分类，随后对用户特征做 L1‑惩罚的逻辑回归和 Pearson 相关性分析。

**📊 数据集**

数据集包括 1,282,437 条来自 792 位用户（ADHD 622，ASD 170）的 Twitter 推文，以及 1,814 条来自 Reddit 的 ReDSM5 注释语料用于模型训练。

**📈 对比分析**

模型评估显示 MentalRoBERTa 的 macro‑F1 为 0.901，逻辑回归在不同阈值下 ROC‑AUC 约 0.65，表明在群体层面能稳定捕捉 ADHD 与 ASD 的抑郁症状侧重差异，但区分能力有限。

**⚠️ 局限性**

局限包括：症状标签稀疏导致部分标签可信度低；使用自报诊断和仅限 Twitter 的样本可能存在偏倚；缺乏神经典型对照组；模型在个体层面诊断应用不具备临床效用。

---

## 104. FLAIR: Distributed Federated Learning with Dynamic Clustering

**arXiv ID:** 2607.06025 | [PDF](https://arxiv.org/pdf/2607.06025v1)

**作者:** Ihssan Boutebicha `[一作]` (Sorbonne Université), Maria Potop Butucaru `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种完全分布式的联邦学习协议FLAIR，结合动态、自组织、资源感知的簇头选举与簇内模型聚合，实现无需中心服务器的高效学习。

**💡 创新点**

核心创新是基于LEACH的簇头选举机制加入可验证随机函数与资源权重评分，既保证公平性又优先选举计算与通信能力更强的节点，同时将聚合任务迁移至簇内。

**🔧 技术方法**

采用分层架构、可验证随机函数、资源加权阈值、在簇内的迭代本地训练与平均聚合，并在ns‑3模拟器中实现完整协议。

**📊 数据集**

在“Spambase”二分类数据集和Kaggle的“Predicting Watering the Plants”智能农业数据集上进行实验，分别对应静态网络和异构物联网场景。

**📈 对比分析**

与中央FL、GAIA、HEAL、Gossip Learning四种基线在100节点静态网络、节点失效、移动性和智能农业场景下对比，FLAIR在静态网络中达0.91准确率，远超基线；在90%节点失效时仍保持>0.85准确率；移动性下误差<2%；在农业场景中与中心化基线相差0.2%。

**⚠️ 局限性**

局限性主要体现在对簇大小和节点能耗的细粒度调优未充分探索，且在极端大规模网络或频繁分区连接的情况下聚合效率仍待验证。

---

## 105. xDECAF: An Extensible Data Flow Diagram Analysis Framework for Information Security

**arXiv ID:** 2607.05913 | [PDF](https://arxiv.org/pdf/2607.05913v1)

**作者:** Benjamin Arp `[一作]` (Karlsruhe Institute of Technology), Nicolas Boltz `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个可扩展的基于数据流图（DFD）的信息安全分析框架xDECAF，包括工具库、在线编辑器和可复用的数据集；

**💡 创新点**

创新点在于将DFD元模型、标签传播逻辑、约束DSL以及浏览器端编辑器解耦，使研究者能够自由定义语义、轻松扩展分析、并提供了超过20个带约束的示例模型作为基准；

**🔧 技术方法**

采用了标签传播的通用分析模型、基于Sprotty的可视化前端、WebSocket通信、DSL约束语言、以及后端分析引擎，支持自定义标签、赋值和约束；

**📊 数据集**

使用了26个从文献、行业合作、已知漏洞以及microSecEnD数据集提取的DFD模型，涵盖7–923个节点和4–72个标签；

**📈 对比分析**

在同类工具中通过对所有132个microSecEnD模型的验证表明，xDECAF能够快速完成分析（最复杂模型不到1分钟），并在多项研究中被作为基准验证、性能和可扩展性；

**⚠️ 局限性**

局限性包括：仍依赖于手工标注标签和约束，DFD的表达范围有限，极大规模模型的传播与约束求解可能导致性能下降，以及需要额外工作将模型与代码或其他架构语言对齐。

---

## 106. Mitigating Errors in LLM-Generated Web API Invocations via Retrieval-Augmented Generation and Constrained Decoding

**arXiv ID:** 2607.05936 | [PDF](https://arxiv.org/pdf/2607.05936v1)

**作者:** Daniel Maninger `[一作]` (Technische Universität Darmstadt), Mira Mezini `[通讯]` (Technische Universität Darmstadt)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了检索增强生成（RAG）与受限解码（CD）两种技术在提升大型语言模型（LLM）生成 Web API 调用代码正确性与去幻化方面的效果。

**💡 创新点**

创新点包括：①为 OpenAPI 规范设计端点级检索格式并构建检索器；②将 OpenAPI 自动转译为正则表达式约束，实现受限解码；③将 RAG 与 CD 结合并构建真实世界 GitHub 任务集验证方法；④系统评估四种生成设置。

**🔧 技术方法**

采用的技术包括：检索增强生成（RAG）—使用 all‑MiniLM‑L6‑v2 词向量与 Chroma 向量数据库；受限解码（CD）—基于正则表达式的约束推理；LLM 模型如 StarCoder、Code Llama、GPT‑4o 等；执行式评估管道。

**📊 数据集**

使用的数据集为：①原始 WAPIIBench 合成任务集（395 个任务，4 个真实 API）；②新构建的来自 GitHub 的真实世界任务集（28 个任务，11 个 API）。

**📈 对比分析**

通过生成→执行→正确性分析的执行式评估，对比 vanilla、RAG、CD、RAG+CD 四种设置。结果显示：RAG 单独在全完成设置上可提升 113% 的正确率，但在参数完成或真实数据集上效果有限；CD 在所有设置下均能消除幻化并显著提升正确率（合成集约 +209%/+143%）；RAG+CD 结合后在合成集上最高正确率约 73%（≈+332%/ +111% 的提升），但一致性略低，部分模型表现下降。

**⚠️ 局限性**

限制包括：RAG 的检索设计空间大、模型依赖性强，易导致过度参数加入；CD 的正则约束可能过度或不足，且对变量引用处理有限；RAG+CD 的一致性差，存在模型依赖性；真实世界数据集规模小、构建耗时；实现需要手动提供 API 规范；受限解码实现效率较低，适合批量或离线场景。

---

## 107. A Mechanistic Lens on Semantic Conflicts: Using Activation Patching to Understand LLM Behavior

**arXiv ID:** 2607.05587 | [PDF](https://arxiv.org/pdf/2607.05587v1)

**作者:** Youssef Abdelsalam `[一作]` (Saarland University), Sven Apel `[通讯]` (Saarland University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造45个Python代码三元组，系统评估四款开源LLM在面对语义冲突（代码实现与注释/标识符不一致）时的表现，并结合残差流激活补丁技术对模型内部状态进行机制化解释；

**💡 创新点**

首个针对代码语义冲突的机制化研究框架，揭示LLM更易受误导性语义线索影响，且冲突信息在模型内部呈现多阶段定位模式（早期在变更区，中间稀疏载体，后期在输出读取层）；

**🔧 技术方法**

采用残差流激活补丁（Residual-Stream Activation Patching）结合统计检验（McNemar、Wilcoxon）和恢复得分，对模型在冲突与对齐输入下的内部状态进行因果定位；

**📊 数据集**

自构造的45个Python snippet triplet（aligned、cue‑varied、implementation‑varied），共135条片段，保持token对齐，包含明确的执行与语义冲突；

**📈 对比分析**

通过比较对齐与冲突版本在最终输出预测和单元测试生成任务的执行正确率和通过率，四款7–8B开源LLM表现均显著下降：最终输出正确率平均降低约40%；单元测试通过率降低18–44%；所有模型在统计检验下均达显著性；

**⚠️ 局限性**

局限性：仅覆盖小规模、单文件、明确冲突的代码；任务限定为输出预测与单元测试，未覆盖更复杂的工程任务；仅使用7–8B开源模型，未验证大型闭源或工具增强模型的泛化；构造冲突为人工，真实项目中的冲突可能更微妙且分布更广。

---

## 108. K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm : (Compensated Loss-Based Sample Exclusion with Unbiased Gradient Estimation)

**arXiv ID:** 2607.05903 | [PDF](https://arxiv.org/pdf/2607.05903v1)

**作者:** Jean-Francois Bonbhel `[一作]` `[通讯]` (NeuroSoft IA), Jean-Francois Bonbhel (NeuroSoft IA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出K-ABENA框架，在梯度反向传播中通过阈值筛选低损失样本并采用防御式混合采样与逆概率重加权，实现计算量显著降低且梯度保持无偏；

**💡 创新点**

创新在于将传统的选择偏差问题与经典的Horvitz–Thompson无偏估计相结合，构建了唯一既节省计算又能保证SGD收敛的阈值筛选方法，并证明了无偏选择失败的理论边界；

**🔧 技术方法**

主要技术包括Horvitz–Thompson与Hájek无偏估计、防御式混合采样、逆概率重加权、非凸SGD收敛分析以及对极端类别不平衡和标签噪声的鲁棒性实验；

**📊 数据集**

实验使用了公开的scikit-learn数据集（乳腺癌、手写数字、葡萄酒、糖尿病）以及一个正例仅占0.17%的合成欺诈数据；

**📈 对比分析**

在与全批量SGD、OHEM、SBP、焦点损失、全局重要性采样等方法在相同计算预算下对比，K-ABENA在常规任务中性能与基线持平且节省28-54%梯度计算；在极端不平衡或高噪声场景下，采用无偏估计可恢复接近全量性能；

**⚠️ 局限性**

主要局限包括仅在CPU浅层模型验证，GPU深度网络未测试；适用于SGD，无法直接推广至Adam/AdamW/Lion；逆概率估计存在O(1/m)阶偏差；在极端不平衡或噪声时有偏模式失效；需要手动调节阈值与保留比例。

---

## 109. Two Sides of the Same Coin: Learning the Backdoor to Remove the Backdoor

**arXiv ID:** 2607.05748 | [PDF](https://arxiv.org/pdf/2607.05748v1)

**作者:** Qi Zhao `[一作]` (Karlsruhe Institute Of Technology), Christian Wressnegger `[通讯]` (Karlsruhe Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种在训练阶段通过识别并移除有毒样本来防止神经网络后门注入的方法。

**💡 创新点**

创新点在于使用强烈后门化的参考模型而非普通干净参考模型来准确识别毒性样本，并通过只利用对称交叉熵中反向交叉熵项实现更稳健的数据集拆分。

**🔧 技术方法**

采用了对称交叉熵（Symmetric Cross Entropy）损失的反向交叉熵成分、迭代训练参考模型、动态数据集拆分、以及最终在纯净子集上训练干净模型。

**📊 数据集**

在多种数据集上验证，包括CIFAR‑10、CIFAR‑100、SVHN等，针对多种后门攻击（dirty‑label、clean‑label、触发器多样化）以及不同网络架构（ResNet、VGG、MobileNet）。

**📈 对比分析**

与现有训练时防御方法（如基于阈值的拆分、参考模型反向学习、后门检测等）比较，攻击成功率被压到最低值（<2%），同时自然准确率几乎不降，明显优于所有对比方法。

**⚠️ 局限性**

限制在于方法需要对损失分布进行阈值估计和多轮迭代训练，对极低毒性率或高混淆度的后门可能仍难以完全检测；此外在对抗性训练或混合攻击场景下的鲁棒性尚未充分验证。

---

## 110. PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails

**arXiv ID:** 2607.05910 | [PDF](https://arxiv.org/pdf/2607.05910v1)

**作者:** Mingyang Song `[一作]` (Fudan University), Bo Li `[通讯]` (University of Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向政策适应的图像安全评估基准PolicyShiftBench，并基于此构建了PolicyShiftGuard模型；

**💡 创新点**

创新点在于：①通过可执行政策规则和细粒度属性构造可供政策切换的标注，②提出两阶段训练（随机化政策SFT+边界对比BP‑Adapt）并使用边界对损失显式学习同图像在不同政策下的决策翻转；

**🔧 技术方法**

采用的技术包括：随机化政策SFT（RP‑SFT）去除对政策顺序/文本表述的依赖；Boundary‑Pair Policy Adaptation（BP‑Adapt）利用匹配的通过/阻止政策对进行对比损失；最终输出采用简洁的二元结构化决策以提升推理速度；

**📊 数据集**

使用的数据集为：PolicyShiftBench（265张图，2,000个政策区分实例，28种评估政策变体）以及训练集9,816条政策条件示例；实验还跨基准评估在UnSafeBench和SafeEditBench上的迁移；

**📈 对比分析**

与多种通用多模态模型（Qwen2.5‑VL、Qwen3.5等）和专用安全模型（GuardReasoner‑VL、SafeGuard‑VL‑RL‑7B、LlamaGuard‑4‑12B等）进行对比。PolicyShiftGuard‑7B在PolicyShiftBench上获得Avg. F1 = 76.9、Avg. PSS = 72.1，显著优于基线；同时在UnSafeBench、SafeEditBench上的整体得分和推理延迟均位居前列；

**⚠️ 局限性**

限制包括：①不同风险类别的难度差异较大，某些类别（如受管商品、IP/品牌、PII、文本安全）仍表现不佳；②模型规模提升有助但不足以完全克服政策适应挑战；③评估依赖于属性标注与规则的准确性，若属性抽取错误会影响标签可靠性；④目前仍未覆盖所有实际产品可能遇到的极端或细粒度政策情景。

---

## 111. Prompt-to-Paper: Agentic AI System for Bioinformatics

**arXiv ID:** 2607.05456 | [PDF](https://arxiv.org/pdf/2607.05456v1)

**作者:** Ramsha Kamran `[一作]` (National University of Sciences and Technology), Muhammad U. S. Khan `[通讯]` (National University of Sciences and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个基于多代理的系统Prompt-to-Paper，用于生成并改进生物信息学论文。

**💡 创新点**

创新点在于确定性检索增强生成、真实实验执行、八维质量评分与基于上下文的改进循环。

**🔧 技术方法**

采用了深度学习语言模型（deepseek-v4-pro 与 deepseek-chat）、检索增强生成、自动化代码执行与多维评分。

**📊 数据集**

使用了60–100篇生物医学论文的检索语料库，并在五个生物信息学任务上进行实验。

**📈 对比分析**

与现有系统对比，平均提升17.96分（最高26.04），成本$0.309/篇，零幻觉引用，整体质量达B–B+区间。

**⚠️ 局限性**

局限包括评估样本有限、引用检索自我参照、评分未完全校准人类评审、对新颖性生成不足等。

---

## 112. Progressive Reasoning with Primitive Correction for Compositional Zero-Shot Learning

**arXiv ID:** 2607.05911 | [PDF](https://arxiv.org/pdf/2607.05911v1)

**作者:** Ziyi Chen `[一作]` (Beijing Jiaotong University), Congyan Lang `[通讯]` (Beijing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于多步推理与双向纠正的Compositional Zero-Shot Learning框架PRPC。

**💡 创新点**

通过将属性与对象识别建模为相互纠正的链式思维，并结合RL奖励提升中间推理质量，显著降低单向预测的误差传播。

**🔧 技术方法**

采用大语言模型Qwen‑VL的Chain‑of‑Thought生成，分两阶段训练（监督+GRPO强化学习）以及精确步骤奖励与KL正则。

**📊 数据集**

在MIT‑States、C‑GQA和VAW‑CZSL这三大公开CZSL基准上进行实验。

**📈 对比分析**

在传统CLIP+文本检索评估（Setting 2）和自定义CoT推理评估（Setting 1）下，与多种MLLM和CLIP基线对比，PRPC在AUC、HM和Top‑1准确率上均实现或逼近最先进水平，尤其在未见组合上的提升最为显著。

**⚠️ 局限性**

需要精细的步骤模板与解析，训练过程对格式错误敏感，且两阶段RL训练成本较高；在极端视觉歧义或属性细微差异时仍易出现错误。

---

## 113. SpecTrack: Spectral Prompt Guided Adaptive Experts for Multispectral Object Tracking

**arXiv ID:** 2607.05988 | [PDF](https://arxiv.org/pdf/2607.05988v1)

**作者:** Xingyu Tan `[一作]` (Beijing University of Posts and Telecommunications), Mengjie Hu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种针对多光谱和高光谱目标跟踪的SpecTrack，能够根据搜索区域的难度动态分配光谱空间建模能力。

**💡 创新点**

创新点在于引入容量有序的专家池（SAMoE）、光谱提示路由器（Spectral Prompt Router）以及共享全局专家，利用记录波段引导稀疏专家选择，实现搜索区级别的自适应计算。

**🔧 技术方法**

核心技术包括Transformer编码器、混合专家（Mixture-of-Experts）结构、基于语义、空间高通与潜在通道差异的提示向量、以及稀疏路由与共享上下文。

**📊 数据集**

使用了MUST、MSITrack、HOTC20三大多光谱/高光谱跟踪基准，以及GOT‑10k RGB一镜像通用化验证。

**📈 对比分析**

在所有基准上与公开最佳方法对比，SpecTrack‑B224在MUST上实现62.4% AUC并以43.7FPS提供良好精度与速度平衡，SpecTrack‑L384在MUST、MSITrack、HOTC20分别达到65.2%、51.9%与72.6% AUC，在GOT‑10k上AO 79.3%，显著优于或竞争现有最优方法。

**⚠️ 局限性**

主要局限在于对目标消失、完全遮挡、低分辨率及尺度变化的处理仍不够理想，需结合全局再检测、长期记忆与不确定性模板更新等进一步改进。

---

## 114. SWE-Review: Closing the Loop on Issue Resolution with Agentic Code Review

**arXiv ID:** 2607.06065 | [PDF](https://arxiv.org/pdf/2607.06065v1)

**作者:** Ruoyu Wang `[一作]` (NTU), Haoli Bai `[通讯]` (Huawei Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SWE-Review框架，旨在通过代理代码审查来关闭AI生成的拉取请求（PR）与实际软件问题之间的循环，提供系统的审查、诊断和修订。

**💡 创新点**

创新点在于将代码审查视为一个主动的能力，审查代理能够探索代码库并提供结构化反馈，从而提高PR的决策准确性和修订效果。

**🔧 技术方法**

使用了代理代码审查的框架，结合了生成-审查-修订的循环，并通过构建基准数据集和审查轨迹来评估审查能力。

**📊 数据集**

使用了来自500个SWE-bench验证问题的1,384个候选PR和8,914个代理审查轨迹的数据集。

**📈 对比分析**

与单轮固定上下文审查相比，代理审查在决策准确性和修订后的解决率上均表现更好，尤其是在处理更复杂的非局部补丁时，解决率显著提高。

**⚠️ 局限性**

局限性在于该研究主要集中在AI生成的PR和软件工程问题解决上，未涵盖特性实现、重构、文档、迁移和架构变更等更广泛的审查场景。

---

## 115. Text Distance from Nested and Hierarchical Repetitions: A Compression-Based Perspective

**arXiv ID:** 2607.05416 | [PDF](https://arxiv.org/pdf/2607.05416v1)

**作者:** Xiaojun Hu `[一作]` (Beijing Normal University), Yu Liu `[通讯]` (Beijing Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于Ladderpath的无训练文本分类框架，利用嵌套重复子结构进行压缩并构造距离度量。

**💡 创新点**

创新点在于通过Ladderpath提取层次化子结构，定义了三种新距离（NCD_lp、L_Dice、L_Jaccard），并在多语言OOD与少样本场景下实现了优于gzip NCD和BERT的性能。

**🔧 技术方法**

技术手段包括算法信息理论、Ladderpath结构压缩、k‑NN分类器，以及基于压缩的相似度计算。

**📊 数据集**

使用了 AGNews、DBpedia、R8、R52、KinyarwandaNews、KirundiNews、DengueFilipino、SwahiliNews、SogouNews 等多语言、跨领域数据集。

**📈 对比分析**

通过与 gzip‑NCD、BERT、TextCNN、LSTM、Bag‑of‑Words 等传统与深度学习方法对比，L_Dice/L_Jaccard 在 OOD 与 few‑shot 设置中均取得最高或接近最高的准确率，整体表现与或优于大模型。

**⚠️ 局限性**

局限性包括对语义细粒度的捕捉不足、解释性较弱，以及对长文本压缩与距离计算成本的进一步评估仍待深入。

---

## 116. Dynamic Evaluation of Classical and Control-Aware Optimal Trajectory Planning in Robot Manipulators

**arXiv ID:** 2607.05544 | [PDF](https://arxiv.org/pdf/2607.05544v1)

**作者:** Bhanuka Dayawansa `[一作]` (University of Moratuwa), Rohan Munasinghe `[通讯]` (University of Moratuwa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究在相同闭环非线性执行条件下，对经典三种轨迹规划（cubic、quintic、trapezoidal）与基于控制感知的有限时域最优控制规划框架进行了统一评估，并通过仿真验证后者在跟踪误差、补偿扭矩及执行成本方面的优势。

**💡 创新点**

创新点在于提出中点线性化策略和统一评估框架，能够在同一动力学模型、控制器和执行约束下公平比较轨迹生成方法，证明了经典平滑轨迹并不必然导致动态高效执行。

**🔧 技术方法**

采用了非线性机器人动力学模型、线性化与离散化、有限时域控制-努力最优规划（QP求解）、PID 反馈控制以及RK4仿真等技术。

**📊 数据集**

使用了非线性简化的3-DoF UR5机械臂动力学模型进行仿真，构成实验数据集。

**📈 对比分析**

在相同初末状态、采样时长、扭矩限制以及相同PID控制器下执行所有规划，并比较RMS跟踪误差、RMS补偿扭矩、累计补偿能量和累计执行成本，结果显示控制感知规划在这些指标上分别降低约28–62%补偿扭矩、48–62%执行成本，且跟踪误差最小。

**⚠️ 局限性**

局限性包括仅在离线仿真3-DoF UR5上验证，未考虑实时计算复杂度、多关节系统或实际机械臂噪声与不确定性等因素。

---

## 117. Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression

**arXiv ID:** 2607.06036 | [PDF](https://arxiv.org/pdf/2607.06036v1)

**作者:** Stefan Josef `[一作]` `[通讯]` (Independent Researcher), Stefan Josef (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了轻量化的多向量压缩技术——在现有 ColBERT 模型上进行池化感知微调，降低文档向量数量而不损失检索准确度。

**💡 创新点**

提出多因子随机池化训练（multi‑factor training）和 k‑means 池化在微调中的高效性，证明即使仅使用少量查询样本也能实现 75%–83% 的向量压缩且检索性能不变，且此训练对不同池化方式和数据集具有良好的迁移性。

**🔧 技术方法**

使用 ColBERT encoder、k‑means / 階層聚类 / span（顺序）池化、MaxSim 匹配、KL 散度蒸馏损失、L2 归一化、AdamW 优化器，并在微调时在前向传播中嵌入池化操作。

**📊 数据集**

主要在 BEIR 子集 NanoBEIR 的 FiQA、SciFact、NFCorpus、SCIDOCS、Touché2020 等数据集上进行训练与评估；验证时还使用完整 BEIR 数据集。

**📈 对比分析**

与传统的推理时无训练池化（span、hierarchical、k‑means）以及单因子微调方法对比；实验显示多因子 k‑means 微调在 1–6 阶层压缩下平均保持 97%–99% 的检索准确度，甚至在 6 倍压缩时仍达 79%（相对 83%），在 SciFact 上实现 83% 向量压缩且无精度损失。

**⚠️ 局限性**

受限于单张 RTX A4500 GPU，实验规模有限，仅对小模型与少数数据集进行单次训练，未做多次重复验证；微调时池化计算开销显著提升（尤其是 k‑means 10–13×），且未探讨在完整 ColBERT 预训练阶段直接引入多因子池化的潜在收益。

---

## 118. Prompting Complexity: Shortest Prompts for Texts and Behaviors in LLMs

**arXiv ID:** 2607.06145 | [PDF](https://arxiv.org/pdf/2607.06145v1)

**作者:** Adrian Cosma `[一作]` `[通讯]` (Dalle Molle Institute for Artificial Intelligence), Adrian Cosma (Dalle Molle Institute for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并定义了“提示复杂度”概念，衡量在固定语言模型下生成目标文本所需的最短可解释提示长度，并将其扩展到软提示复杂度、提示距离与行为提示复杂度等相关度量。

**💡 创新点**

其创新点在于将Kolmogorov复杂度与模型特定的提示问题相映射，形成可计算的、与模型相关的压缩度量，并通过提示距离与软提示等新定义提供了完整的理论框架。

**🔧 技术方法**

技术上基于语言模型的确定性解码与有限上下文，使用可解释文本集合、核采样阈值和距离函数，构建了提示复杂度的定义、可计算性证明与编码定理。

**📊 数据集**

论文主要是理论工作，并未在具体实验中使用任何公开数据集，而是讨论了合成数据、评估数据对提示复杂度的影响。

**📈 对比分析**

比较方法通过定义提示复杂度与模型概率、软提示复杂度、行为提示复杂度等理论指标，并提出了实验研究路线；论文未给出具体数值评估，强调需要后续实验验证其效果。

**⚠️ 局限性**

主要限制在于计算不可行——提示复杂度需枚举所有可解释提示，实际只能得到上界；此外模型依赖导致缺乏普适性，软提示的距离度量和实现细节也需进一步研究。

---

## 119. Uncertainty-Aware Velocity Correction for Proprioceptive Vehicle Localization using Evidential Mamba

**arXiv ID:** 2607.05669 | [PDF](https://arxiv.org/pdf/2607.05669v1)

**作者:** Abinav Kalyanasundaram `[一作]` (AImotion Bavaria, Technische Hochschule Ingolstadt), Michael Botsch `[通讯]` (AImotion Bavaria, Technische Hochschule Ingolstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 evimamba 架构，利用车载传感器估计虚拟速度并在 Error-State EKF 中进行漂移校正，实现 GNSS 失效时的可靠定位。

**💡 创新点**

将 Mamba 选择性状态空间模型与 Normal‑Inverse‑Gamma 显式不确定性推理相结合，生成可量化不确定性的虚拟速度传感器，且不需要额外硬件。

**🔧 技术方法**

Mamba 选择性 SSM、显式深度学习不确定性（NIG）、Error‑State EKF、边缘硬件实时推理。

**📊 数据集**

revsted 数据集（含 IMU、车载多传感器、外部速度传感器与 RTK 位置）。

**📈 对比分析**

与 OSD‑Baseline、Transformer 及 Correvit 进行对比；在 3–10 分钟 GNSS 失效中，evimamba 的最大漂移仅比 Correvit 高约 10%，优于所有基于车载传感器的方法。

**⚠️ 局限性**

需要针对不同车辆/路况进行迁移学习；侧向速度不确定性略低；实验仅覆盖平地车辆场景，尚未验证多车或极端环境下的泛化能力。

---

## 120. AVA-VLM: Adaptive Visual Attention-Vision Language Model for In-the-Wild Construction Site Monitoring

**arXiv ID:** 2607.05859 | [PDF](https://arxiv.org/pdf/2607.05859v1)

**作者:** Younggun Kim `[一作]` (University of California, Los Angeles), Seunghee Park `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对建筑工地监控的视觉语言模型 AVA‑VLM，采用人类粗到细的视觉注意机制，先用低分辨率全景图做整体推理，再根据需要智能裁剪局部高分辨率区域进行细节检查，从而实现更宽范围、低分辨率下更可靠的判断和更高效的推理。

**💡 创新点**

创新点在于：① 构建了区域感知的 Chain‑of‑Thought（CoT）数据集，使模型学习何时、何地需要放大检查；② 设计了可调的裁剪策略和工具调用，避免了传统方法对所有区域都进行高分辨率处理的开销；③ 通过端到端的训练框架，分离裁剪学习和答案生成学习，提升训练稳定性；④ 在建筑安全场景中实现了对远距离、低分辨率图像的鲁棒推理。

**🔧 技术方法**

技术上主要使用 LoRA 进行参数高效微调的 Qwen2.5‑VL 大模型；对裁剪工具进行标记化、推理时使用 1/4 或 1/2 低分辨率全景图；采用工具调用机制和多轮 CoT 生成；评估时使用视觉标记化损失、工具调用损失，并在训练中对模型的响应进行掩码处理。

**📊 数据集**

数据集：基于 ConstructionSite10K 进行扩充，生成 10,013 张图像的区域感知 CoT 数据集，包括图像描述、违规识别（PPE 非合规）和目标检测（挖掘机、钢筋、白帽工人）三大任务；在训练集/验证集/测试集上按 8:2:1 分割，包含多种相机距离、光照和信息质量标签。

**📈 对比分析**

对比方法包括：零样本/少样本 Qwen2.5‑VL、直接 QA 微调基线、示例图像 CoT 基线。实验表明 AVA‑VLM 在违规识别任务上 F1 提升 13.1pp、PPE 违规召回提升 52.2%（长距离场景），同时视觉令牌使用仅为 30.6%（比基线低 69%）。在目标检测任务中，AVA‑VLM 在长距离下平均 IoU 由 50.0% 提升到 52.4%，工具调用比例随距离升高而显著增加，展示了自适应裁剪的优势。

**⚠️ 局限性**

局限性：① 在目标检测任务中整体 IoU 仍略低于全分辨率直接 QA，表明对细小目标的细节捕捉仍受限；② 训练时依赖于手工标注的裁剪区域，若标注不足可能导致模型对裁剪判断失准；③ 对极低分辨率全景图（1/4）时裁剪调用比例下降，导致部分重要细节被忽略；④ 当前实现仅针对 PPE 非合规，扩展到更多安全违规场景需要进一步验证。

---

## 121. Discovering Frequent Closed Embedded Sub-DAGs in Spatio-Temporal Event Data

**arXiv ID:** 2607.05995 | [PDF](https://arxiv.org/pdf/2607.05995v1)

**作者:** Piotr S. Maciąg `[一作]` `[通讯]` (Warsaw University of Technology), Piotr S. Maciąg (Warsaw University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于频繁闭嵌入子DAG挖掘的时空事件模式发现方法；

**💡 创新点**

创新点在于将时空事件关系建模为有向无环图并聚焦闭合子图，既避免了子树模式冗余，又提供了更完整的交互关系；

**🔧 技术方法**

技术包括定义时空随从关系、构造DAG集合、使用DigDag算法进行闭合子DAG挖掘，并利用networkx实现图操作；

**📊 数据集**

实验数据采用波士顿警察局2014年犯罪报告集，共包含26类犯罪事件；

**📈 对比分析**

与SLEUTH（传播树模式）和CSTPM（级联时空模式）对比，DigDag在不同R、T、minSup配置下显著更快，执行时间更稳定，发现模式数量亦更紧凑；

**⚠️ 局限性**

局限性包括仅处理单事件实例的DAG，未考虑多实体或轨迹数据；算法依赖于参数R、T的设定，对极大规模数据的可扩展性尚待进一步验证。

---

## 122. HAPS as a Hypercell: Enabling Coverage and Capacity Carrier Shutdown in Cellular Networks

**arXiv ID:** 2607.06072 | [PDF](https://arxiv.org/pdf/2607.06072v1)

**作者:** Matteo Bernabè `[一作]` (Universitat Politècnica de València), Nicola Piovesan `[通讯]` (Huawei Technologies)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在密集城市网络中引入高空平台（HAPS）构建“Hypercell”，实现覆盖层和容量层双向关闭，降低网络功耗。

**💡 创新点**

创新点在于：① 将HAPS作为覆盖层，可取代多颗宏基站，支持容量与覆盖单元同时休眠；② 设计两种层级化配对架构（HAPS‑NH 与 HAPS‑H），并给出分布式关闭/唤醒算法；③ 通过 3GPP 兼容的全链路系统模型验证，首次量化 HAPS‑Hypercell 在低流量时段可实现 12.5% 的功耗降低。

**🔧 技术方法**

使用 3GPP UMa 统计信道模型、Rician 小尺度衰落、mMIMO 天线阵列、CSI‑RS 反馈、功耗模型及自研高保真系统级仿真器（OpenGiuliaSLS）。

**📊 数据集**

实验基于仿真数据：包含 19 个共址 4G/5G 宏站（57/57 个单元）和中心 HAPS，覆盖面积 500 m ISD；UE 均匀分布与热点混合，日内流量变化依据 EARTH 项目分布；无公开真实数据集，所有结果来自仿真。

**📈 对比分析**

通过三种阈值策略（保守、平衡、激进）和两种配对架构进行对比。相较于传统仅 4G/5G 关闭，HAPS‑NH 在低流量时段节能 12.5%，24h 平均 10.5%；HAPS‑H 在激进策略下 12.3%/10.5%。QoS 方面，平均速率下降 1.3–9.6%，取决于策略与架构；在激进策略下 HAPS‑NH 速率下降更大（13%），而 HAPS‑H 仅 6.5%。

**⚠️ 局限性**

局限性：① 仅基于本地 PRB 负载做关闭判断，未充分利用网络全局信息；② HAPS‑NH 受限于 HAPS 负载饱和导致唤醒不精准；③ HAPS‑H 受中间层决策影响，能耗相对较高；④ 研究仅在仿真环境验证，未考虑实际部署的协同、干扰、可靠性等工程挑战。

---

## 123. Reward-Density Heuristic for Dynamic Multi-Vehicle Routing: Performance and Computational Efficiency

**arXiv ID:** 2607.06066 | [PDF](https://arxiv.org/pdf/2607.06066v1)

**作者:** Manish Kolachalam `[一作]` (Infosys Center for Imaging Technologies, Infosys), Rani Malhotra `[通讯]` (Infosys Center for Imaging Technologies, Infosys)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于奖励密度的在线动态车辆调度框架，并在无人机任务分配与城市出租车调度两个实际场景中进行验证。

**💡 创新点**

创新点在于：①提出了“Efficiency”奖励密度启发式（奖励/时间+服务时间）用于多车辆在线分配；②将贪婪与匈牙利匹配两种实现方式统一评估；③系统性比较了多种传统构造启发式与元启发式（ALNS、GA、SA），证明奖励密度方法在奖励质量上可与元启发式相媲美，却显著降低计算开销。

**🔧 技术方法**

主要技术包括：事件驱动仿真、贪婪奖励密度评分、匈牙利算法（Hungarian‑Efficiency）、基于奖励密度的贪婪分配、以及 GA/SA/ALNS 等元启发式；统计检验采用 Bonferroni 校正的 Mann‑Whitney U 检验；对比指标为累计奖励（或收入）和规划时间。

**📊 数据集**

数据集：①无人机任务采用 5000×5000 的随机地图，奖励均匀分布 1–100；②出租车调度使用 2024 年 NYC TLC Yellow Cab 真实数据，约 7,600 个任务。

**📈 对比分析**

对比方法：将 Efficiency（贪婪与匈牙利两种实现）与四种构造启发式（Nearest、Reward、Time、Reward）以及三种元启发式（GA、SA、ALNS）在同一实验配置下比较。结果显示，Efficiency 方法在所有配置下均与元启发式在奖励上无显著差异，但规划时间仅为元启发式的 1/1000~1/40，显著优于其他启发式，且在奖励/计算成本上实现 Pareto 主导。

**⚠️ 局限性**

局限性：①实验基于仿真，未验证在真实动态交通、车辆失效或非平稳需求下的鲁棒性；②只优化单目标（奖励/收入），未考虑公平、服务覆盖、碳排放等多目标约束；③未评估大规模任务或更长规划周期下元启发式性能可能提升的情况。

---

## 124. Tuning-Free Latent Diffusion Models for Ultrahigh-Resolution Image Editing

**arXiv ID:** 2607.06136 | [PDF](https://arxiv.org/pdf/2607.06136v1)

**作者:** Wanglong Lu `[一作]` (Wenzhou University), Xianta Jiang `[通讯]` (Memorial University of Newfoundland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了UltraDiffEdit，一个无需调优的多尺度渐进式大分辨率图像编辑框架。

**💡 创新点**

提出多patch编码、全局‑局部一致性去噪与混合采样技术，解决传统模型在8K及以上分辨率下的细节丢失与边界不连贯问题。

**🔧 技术方法**

基于预训练的Latent Diffusion Models（如Stable Diffusion、SDXL），实现了patch‑based hybrid sampling、global‑local consistency denoising及多patch编码。

**📊 数据集**

构建了DIV2KEdit、Syn2KEdit、UHRSDEdit三大高分辨率编辑数据集，涵盖2K至8K图像及对应文本提示、掩模和多模态条件。

**📈 对比分析**

与CoordFill、HD‑Painter、DemoFusion、SDXL+SRGAN/BSRGAN/Inf‑DiT等方法在PSNR、SSIM、FID、LPIPS等指标上进行客观评估，UltraDiffEdit在所有数据集上均表现出更高的图像质量与语义一致性，且可在单张RTX 3090上完成8K编辑。

**⚠️ 局限性**

主要限制包括推理时间较长、缺乏多GPU加速，以及在某些场景下出现小物体重复生成的现象。

---

## 125. StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems

**arXiv ID:** 2607.05844 | [PDF](https://arxiv.org/pdf/2607.05844v1)

**作者:** Sergey Volkov `[一作]` (University of Hong Kong), Ye Luo `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 CRDT 合并之上的冲突感知记忆合约 StateFuse，用于在代理系统中透明地记录冲突并支持语义纠错；

**💡 创新点**

创新点在于引入可追踪的历史、显式冲突对象、精确与语义纠错句柄以及仅在投影时进行的有界决策，形成一个安全、可审计的公共记忆接口；

**🔧 技术方法**

技术包括 OpSet/CRDT 传统合并、基于谓词的确定性规范、投影时决策器、语义句柄与精确句柄的纠错机制以及投影等价压缩与签名验证；

**📊 数据集**

使用了官方 MemoryAgentBench 的 Conflict_Resolution 切片（282 个冲突问题）以及自建的统一验证代理循环；

**📈 对比分析**

通过与多值表面、原始日志、合并最新写入等基线在同等信息量和验证预算下比较，结果显示 StateFuse 在冲突可见度、保留模糊性与安全后验验证方面优于合并写入，准确率与强多值基线持平；

**⚠️ 局限性**

局限性在于实验仅覆盖单一官方切片与合成测试，缺乏广泛的自然生成轨迹评估，且对 Byzantine 容错、对抗性攻击及大规模部署的评估不足。

---

## 126. EAGOR: Embodied Reasoning in Omni-direction

**arXiv ID:** 2607.06165 | [PDF](https://arxiv.org/pdf/2607.06165v1)

**作者:** Shriram Damodaran `[一作]` (Nanyang Technological University), Addison Lin Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出无训练、基于球面贝叶斯估计的EAGOR框架，用来实现全景摄像头下的方向推理

**💡 创新点**

创新点是将VLM输出转换为球面方向似然，并利用球面谐波表示连续、等变的方向信念，实现运动一致的方向估计

**🔧 技术方法**

使用球面谐波（SH）贝叶斯滤波、递归更新、Wigner‑D 旋转、Spherical Harmonic Belief Field 以及冻结的 VLM（Qwen2.5‑VL、Gemma‑3）生成目标似然

**📊 数据集**

在 Habitat‑Sim（Waypoint Following、Map‑Free Navigation）、HOS、OSR‑Bench 以及真实 Unitree‑Go2 机器人上进行评估

**📈 对比分析**

与基于 ERP 的 VLM、Centroid、Centroid‑Circ、Grid 等基线对比，EAGOR 在导航成功率提升至 94%、SPL 56.8%，在活跃视觉搜索中平均提升 40%+，显著优于基线

**⚠️ 局限性**

主要局限是对 VLM 语义识别误差敏感，难以处理稀有目标、多实例混淆和误检，且未考虑多传感器或深度信息

---

## 127. DexTele: A Dual-Arm Dexterous Teleoperation System Based on Motion Retargeting and Adaptive Force Control

**arXiv ID:** 2607.05883 | [PDF](https://arxiv.org/pdf/2607.05883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 128. CHARLIE: An On-Premise Multi-Agent Retrieval-Augmented Generation System for Evidential Reasoning in Forensic Science

**arXiv ID:** 2607.05428 | [PDF](https://arxiv.org/pdf/2607.05428v1)

**作者:** Leandro D. Carneiro `[一作]` (Forensic Institute, Civil Police of Federal District), Rafael C. A. Cabral `[通讯]` (Forensic Institute, Civil Police of Federal District)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个完全本地化、多智能体检索增强生成系统Charlie，用于法医机构的结构化证据处理与多文档分析。

**💡 创新点**

创新点在于将RAG与多智能体协作相结合，构建受控、可追溯的工作流，实现任务拆分、并行执行、结构化内存与自检机制，并确保数据主权与司法合规。

**🔧 技术方法**

使用技术包括本地部署的Qwen3‑32B LLM（通过vLLM推理）、LangGraph智能体调度、Ollama嵌入、密集向量检索与重排序、结构化内存（表格/键值）及完整日志记录。

**📊 数据集**

使用的数据集为巴西联邦区刑事法医机构内部的两大实际案例：约2000份交通事故报告和若干妇女谋杀案报告，均为未经公开的中文/葡萄牙文混合文本。

**📈 对比分析**

通过案例研究评估：在交通事故与谋杀案两场景中，系统实现了文档级子查询并行执行、结构化提取、聚合与长期情报生成；性能表现为可扩展至数千文档、保持高可追溯性，未给出定量指标但在实操中验证了可用性。

**⚠️ 局限性**

局限性包括：对检索质量高度依赖；分块切分可能导致信息碎片化；LLM本身非形式化推理，仍存在幻觉与不确定性；部署对算力需求大；未集成正式的贝叶斯/论证推理，仅提供结构化预处理。

---

## 129. Bridging Stakeholder and Product Requirements: An Empirical Study of Requirement Engineering in the Automotive Industry

**arXiv ID:** 2607.05632 | [PDF](https://arxiv.org/pdf/2607.05632v1)

**作者:** Zixu Wang `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对来自Infineon汽车芯片开发项目的8,082条利益相关者需求与5,870条产品需求进行大规模实证研究，探究其结构差异、接受/拒绝决策、偏差管理以及需求映射与缺失上下文的关系。

**💡 创新点**

首次在工业规模上系统量化利益相关者需求与产品需求的关系，提出需求映射模式分类体系，揭示偏差、拒绝原因及上下文缺失对细化复杂度的驱动作用，并为需求引入、偏差建模与上下文补全提供实证依据。

**🔧 技术方法**

采用混合方法：定量文本结构与统计分析、模式挖掘、关联与相关性检验；定性主题编码、专家分类与GPT‑4o自动标签；以及多标签上下文缺失检测，结合标准化评估与案例研究。

**📊 数据集**

Infineon工业数据集：8,082条利益相关者需求、5,870条产品需求，包含ISO 26262审查决策、偏差说明、规范引用及可追溯链接，涵盖硬件与软件文档。

**📈 对比分析**

通过描述性统计、相关性分析、映射分布与共现矩阵对需求特征与决策进行对比；结果显示：规范来源的需求获批率高达83%且映射单一；非规范需求获批率仅10%；平均每条利益相关者需求关联1.93条产品需求；上下文缺失平均3.65类，且与细化数量呈显著正相关；映射复杂度与文本长度弱相关，强调结构与上下文驱动。

**⚠️ 局限性**

研究仅基于Infineon单一企业的专有数据，缺乏跨行业验证；对高层次标准与硬件依赖的深度分析可能不适用于其他汽车分支；实验依赖人工标签与LLM自动化，存在标注偏差；结果主要适用于安全关键汽车芯片开发，推广至更广泛系统仍需进一步验证。

---

## 130. Modality Relevance is not Modality Utility: Post-hoc Selective Modality Escalation for Cost-Aware Multimodal RAG

**arXiv ID:** 2607.05438 | [PDF](https://arxiv.org/pdf/2607.05438v1)

**作者:** Xue Li `[一作]` (Beihang University), Yiming Gai `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在多模态检索增强生成(RAG)中，先用文本+表格快速生成草稿答案，然后通过验证器判定是否需要进一步调用昂贵的视觉语言模型(VLM)来补充图像证据，从而实现对视觉资源的精确、可控弹性升级。

**💡 创新点**

创新点在于：①发现模态相关性与模态实用性存在显著差距；②将模态升级决策从检索前阶段移至草稿生成后，使用验证器对模态缺失进行定位；③设计了基于增益值的阈值路由器，实现精细的成本-准确度权衡。

**🔧 技术方法**

使用的技术包括：Qwen2.5-72B-Instruct文本生成模型、Qwen2.5-VL-7B视觉语言模型生成文本侧车、验证器（判定模态缺口）、阈值化增益值路由器、对齐与校准技术、严格的匹配预算评估框架。

**📊 数据集**

主要使用的基准数据集为 MultiModalQA；为验证跨数据集泛化性，还在 WebQA 上进行实验。

**📈 对比分析**

方法与基线的比较采用严格的匹配预算、跨种子、多拆分的 held‑out 评估；结果显示，post‑hoc 路由器在相同视觉调用预算下，准确率明显高于强化的 pre‑retrieval 基线，且在多预算点上逼近模态 oracle，最终在 39% 视觉调用率时达到 0.423 的准确率，接近 0.430 的模态 oracle，且在 59% 预算时仍保持 0.455 的准确率。

**⚠️ 局限性**

主要局限包括：①视觉信息仅以预先生成的文本侧车形式输入，未直接处理像素；②仅研究单一昂贵模态（图像），未扩展到多种昂贵模态；③在视觉需求普遍高的场景下，额外的草稿与验证器调用可能导致开销过大；④路由器训练需要离线对比实验日志，且在分布漂移时可能需要重新校准。

---

## 131. Publishing Without Journals: An Open, Forkable Archive with Attributed Review

**arXiv ID:** 2607.05454 | [PDF](https://arxiv.org/pdf/2607.05454v1)

**作者:** Matthew Lorig `[一作]` `[通讯]` (University of Washington), Matthew Lorig (University of Washington)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将学术出版的传播与认证分离，构建一个开放归档 + 公开评论 + 投票 + 可派生版本（fork）并记录来源的完整系统，以替代传统期刊的单一门控流程。

**💡 创新点**

创新点在于：① 将传播、认证、关注分配和学术信用四项职能拆解为独立可组合的模块；② 采用公开可追溯的投票与评论机制实现持续认证；③ 通过版本控制和衍生链记录思想血统，自动赋予原创与后续工作的学术信用。

**🔧 技术方法**

使用的技术包括：开放预印本存档（如arXiv、SSRN）、基于注释的评论系统（可支持实名/假名）、投票与加权评分算法、Git/类似版本控制系统实现的论文fork与链接、可查询的来源图谱（provenance graph）以及认证与治理的身份验证与权限管理。

**📊 数据集**

该工作为理论与概念性设计，并未使用具体数据集；其实现示例基于现有平台（arXiv、PubPeer、OpenReview、Octopus、eLife等）的功能拆解与整合。

**📈 对比分析**

没有具体实验或性能对比；作者通过对现有系统（如preprint、Post‑review、forking平台）的功能评估，论证其可行性，并指出需要进一步设计指标（如投票量、fork数、评论质量）来衡量系统效能。

**⚠️ 局限性**

局限性包括：① 参与度不足，评论与fork可能稀缺；② 真实姓名与匿名性的平衡问题导致批评被抑制；③ 失去传统期刊的“认证信号”，评审者需学习新指标；④ 投票易被操纵、富人/名人偏见仍可能出现；⑤ 需要完善治理与身份验证机制以防滥用。

---

## 132. Calf-Integrated Arms for Bimanual Quadruped Loco-Manipulation

**arXiv ID:** 2607.06186 | [PDF](https://arxiv.org/pdf/2607.06186v1)

**作者:** Yan Pan `[一作]` (University College London), Chengxu Zhou `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Unitree Go2四足机器人前腿胫部集成了具有伸缩滑块、两轴旋转和夹爪的双手机械臂，实现了四脚稳定站姿下的地面双手操作，并通过视觉‑语言模型（Kimi K2.6）在模拟环境中自动选择并执行一系列技能完成长期任务。

**💡 创新点**

创新点在于：①将机械臂直接嵌入前腿胫部，既保持四脚稳定，又实现双手协同操作；②采用可视‑语言模型进行长时序技能序列决策；③结合学习驱动的行走策略与基于DLS的逆运动学，实现了对腿部行走与臂部操作的分离控制。

**🔧 技术方法**

使用的技术包括：Kimi K2.6视觉‑语言模型、基于PPO训练的行走策略、三自由度DLS逆运动学、步进电机驱动滑块、RGB‑D相机与标记检测、Isaac Lab模拟训练、MuJoCo评估、分层控制架构（高层规划+低层FSM）。

**📊 数据集**

数据集：主要为Isaac Lab中自建的仿真环境，未使用公开真实数据集；训练过程中通过域随机化（摩擦、恢复系数、质量）增强泛化；标记检测采用手工标记的红色小标记，无真实世界数据集。

**📈 对比分析**

比较方法：在三种典型双手任务（柜子长时序任务、协同提物、手传递）中，将本设计与传统的前胸臂、LocoMan（后腿支撑后才可双手）等方案对比。结果显示：本设计能够在保持四脚稳姿的前提下完成所有三项任务，单次任务平均耗时18.6 s，抓取成功率在10 mm深度噪声下保持70–74%。相较之下，传统方案要么无法实现双手协作，要么需后腿起立导致基座失稳。

**⚠️ 局限性**

局限性：①机械臂缺少绕接近轴的滚转关节，无法抓取需要旋转姿态的物体；②技能库有限，无法应对未定义的新行为；③抓取依赖于手工标记，难以迁移到无标记真实场景；④本工作仅在仿真中验证，未完成真实硬件实验；⑤高度仿真到现实的差距仍需进一步解决。

---

## 133. AbICL: In-Context Learning for Antigen-Specific Antibody Affinity Ranking

**arXiv ID:** 2607.05846 | [PDF](https://arxiv.org/pdf/2607.05846v1)

**作者:** Zhiyuan Chen `[一作]` (Henlius), Feng Zhu `[通讯]` (Henlius)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了AbICL框架，通过上下文学习对抗原特异性的抗体亲和力进行排名。

**💡 创新点**

首次将In‑Context Learning与抗体亲和力排名结合，采用元学习的episodic训练与上下文排名头，使模型在推理时无需梯度更新即可利用已知的亲和力比较进行适配。

**🔧 技术方法**

使用预训练的结构编码器（GCN）、标签嵌入、Set Transformer形式的上下文排名头、episodic meta‑training与二元交叉熵损失。

**📊 数据集**

在AbRank基准数据集上进行实验，包含Balanced、Hard Ab、Hard Ag三种训练拆分，评估Unrelated Complex和Local Perturbation两种测试指标。

**📈 对比分析**

与WALLE‑Affinity、ESM‑2+AntiBERTy、Mint等基线在AUROC上进行对比，AbICL在所有拆分和评估下均优于基线，尤其在Test‑context协议下性能最佳。

**⚠️ 局限性**

仅针对亲和力排名任务进行验证，未扩展到亲和力回归或变化预测；上下文示例的构造策略尚不充分理解，实验局限于AbRank，需进一步在其他任务与数据集上检验。

---

## 134. What Resolve Rate Hides: Trajectory Structure Diagnostics for Coding Agents

**arXiv ID:** 2607.06184 | [PDF](https://arxiv.org/pdf/2607.06184v1)

**作者:** Rui Shu `[一作]`, Yuan Wang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对编码代理的轨迹进行标准化，识别反模式，并通过规则化对齐方法实现跨跑比较，以便对成功/失败过程进行可重复诊断。

**💡 创新点**

提出了九类可规范化动作、确定性效应标签以及单轨迹反模式和跨轨迹对齐的规则化检测框架，解决了轨迹异构性导致的可比性难题。

**🔧 技术方法**

使用轨迹规范化、基于规则的检测器、最长公共子序列（LCS）对齐、功能级别定位、效果标签与里程碑分析等技术。

**📊 数据集**

采用 SWE-Bench Verified（500 任务，2500 轨迹，5 个生产设置）和 SWE-Bench Pro（266 任务）作为实验数据集。

**📈 对比分析**

通过对齐同一任务的成功/失败轨迹，统计文件选择、编辑稳定性、完成行为等维度的差异；检测器在不同设置下的预防率和跨基准转移表现良好，搜索循环被证明是最稳健的失败线索。

**⚠️ 局限性**

局限在于需为每个基准校准阈值、仅针对 Python PR 修复场景、对工具/时延等度量依赖实现，且检测仅提供诊断信息，无法直接给出根因或自动修复方案。

---

## 135. Dynamics and Convergences for Markov Coevolutionary Opinion Formation Games in Dynamic Social Networks

**arXiv ID:** 2607.05580 | [PDF](https://arxiv.org/pdf/2607.05580v1)

**作者:** Po-An Chen `[一作]` (National Yang Ming Chiao Tung University), Chih-Chieh Hung `[通讯]` (National Chung Hsin University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文分析了在动态社交网络中，基于K-最近邻（K-NN）模型的马尔可夫共演化意见形成游戏的收敛性，探讨了引入随机性对系统稳定性的影响。

**💡 创新点**

创新点在于通过乐观梯度上升算法分析一般和总和马尔可夫游戏中的收敛性，提出了超越相关均衡的近似纳什均衡的收敛性结果。

**🔧 技术方法**

使用了多智能体强化学习和在线学习的收敛分析技术，特别是乐观梯度上升算法。

**📊 数据集**

未具体提及使用的数据集，但讨论了基于K-NN模型的动态社交网络的构建。

**📈 对比分析**

与现有方法相比，本文的方法在收敛到近似纳什均衡方面表现出更强的理论保证，且在价格的无序性（price of anarchy）方面提供了界限。

**⚠️ 局限性**

限制在于当前的收敛分析未能达到最后迭代收敛的强结果，未来工作将探讨更广泛的假设条件下的收敛性。

---

## 136. The Jagged Global Economy: Frontier AI Unevenly Exposes National Economies

**arXiv ID:** 2607.05404 | [PDF](https://arxiv.org/pdf/2607.05404v1)

**作者:** Arul Murugan `[一作]`, Rishi Bommasani `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并计算了一个跨国AI暴露度指标，将各国职业暴露分数与国际就业数据结合，覆盖141个国家。

**💡 创新点**

创新点在于首次大规模系统化评估各国劳动市场对前沿AI的脆弱性，揭示性别差距与通过汇款的间接暴露机制，并通过实际AI使用数据验证该指标。

**🔧 技术方法**

技术方法包括基于任务层面暴露估计的职业暴露计算、就业占比加权平均得到国家暴露度、统计回归与相关分析，以及利用KNOMAD汇款矩阵分析间接通道。

**📊 数据集**

使用的数据集包括ILO的ISCO‑08职业就业统计、Gmyrek等人提供的职业暴露估计、Anthropic、Microsoft、OpenAI发布的AI使用量、KNOMAD的国际汇款流量。

**📈 对比分析**

通过与Anthropic Claude使用率、Microsoft AI Diffusion率和OpenAI Signals排名的比较，暴露度与AI采纳呈现显著的指数/线性关系，R²分别为0.77、0.81和0.61，验证了该指标的预测能力。

**⚠️ 局限性**

局限性包括：假设同一职业在各国暴露相同、采用ISCO‑08 2位级别聚合导致细粒度信息损失、部分大国缺失就业数据、未捕捉新兴AI职业以及未考虑职业内跨国差异。

---

## 137. UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods

**arXiv ID:** 2607.06202 | [PDF](https://arxiv.org/pdf/2607.06202v1)

**作者:** Yipeng Liu `[一作]` (Nanjing University), Guihai Chen `[通讯]` (Nanjing University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新设计MoE通信库，采用依赖驱动的内核拆分、Data‑as‑Flag原子同步和层次化Token调度，以消除BSP序列化、同步开销和拓扑无关负载不均，提升多层Superpod上的All‑to‑All性能。

**💡 创新点**

① 通过依赖驱动的任务拆分解放BSP序列化；② 用原子写实现Data‑as‑Flag无软件同步；③ 层次化Token调度统一多层fabric的负载与尾延迟。

**🔧 技术方法**

使用统一全局地址空间、512B原子写、SIMD向量化、点对点同步、硬件加速映射器、四分搜索等技术，集成在Ascend CANN堆栈的Ascend C语言实现。

**📊 数据集**

在四个大型LLM模型上测试：Qwen3‑30B、GLM‑4.7、DeepSeek‑R1、DeepSeek‑V3.2，分别配置不同专家数和层级。

**📈 对比分析**

与CANN EP、DeepEP等基线比较，在256 NPU上All‑to‑All延迟降低52.4%，TPOT降低约11.1%，带宽提升约35‑41%，验证了显著的性能提升。

**⚠️ 局限性**

局限性：高度依赖CM384的512B原子写、统一地址空间和多层fabric；在不支持原子写或单层flat fabric的系统上效果有限；未针对非MoE通信模式或非统一内存体系结构进一步验证。

---

## 138. Canopy: A Heterograph Foundation Model for Metabolic Engineering

**arXiv ID:** 2607.06224 | [PDF](https://arxiv.org/pdf/2607.06224v1)

**作者:** Jake Bowden `[一作]` (Twig Bio), Satnam Surae `[通讯]` (Twig Bio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个跨物种、跨尺度的代谢工程知识图谱（Canopy），并在此上预训练了一个异构图变换器（HGT）作为基础模型，用于预测发酵产物滴度。

**💡 创新点**

创新点包括：①将十个公共与专有数据源统一到一个13类节点、34类边的异构知识图谱；②使用多模态领域基础模型（ESM‑2、MoLFormer、PubMedBERT）作为节点特征编码器；③在HGT上加入SignNet位置编码、随机游走PE、跳跃知识聚合和虚拟节点；④设计四个自监督预训练目标（链路预测、节点掩码、距离回归、实验对比），并采用不确定性加权融合。

**🔧 技术方法**

技术涵盖：异构图变换器（HGTConv）、SignNet位置编码、随机游走位置编码、跳跃知识（Jumping Knowledge）、虚拟节点、混合自监督任务、梯度裁剪、混合精度训练、分布式FSDP、子图采样（多锚点策略）。

**📊 数据集**

数据集：约690万节点、1120万边的知识图谱，包含基因、蛋白、代谢物、反应、通路、菌株、发酵实验、转录组等；用于下游滴度预测的4,791条实验记录（文献挖掘+内部LIMS），并采用MD5哈希划分的5折交叉验证。

**📈 对比分析**

与基线对比：在滴度回归任务上，冻结的Canopy Embedding+轻量线性/MLP探针在R²从0.24（最佳表格基线）提升至0.41（3B模型），AUROC从0.73提升至0.82；同样优于同类型GraphSAGE、HGT无增强等基线。

**⚠️ 局限性**

局限性：①实验数据相对稀疏，覆盖范围不均；②模型为预测性而非机制性，缺乏解释性；③未评估各数据源/模态的独立贡献；④未公开内部LIMS数据与模型权重；⑤需要更多实验数据和跨任务验证以验证“基础模型”主张。

---

## 139. Learning Sparsest Linear Causal DAGs with Latent Confounders via Higher-Order Cumulants

**arXiv ID:** 2607.05984 | [PDF](https://arxiv.org/pdf/2607.05984v1)

**作者:** Ming Cai `[一作]` (Kyoto University), Hisayuki Hara `[通讯]` (Kyoto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种从有限样本中恢复线性非高斯无环模型（LvLiNGAM）中最稀疏DAG的算法，消除了原ReLVLiNGAM方法对局部约束的依赖；

**💡 创新点**

创新点包括：①直接对观测变量残差化而非递归更新高阶累积量，从而消除局部约束；②通过匹配不同变量对的累积量来逐步确定父子关系；③在残差化后通过最小化潜在混淆变量数量来挑选最稀疏的潜变量–观测变量结构；

**🔧 技术方法**

主要技术手段为高阶累积量分析、泛化性假设下的矩阵秩判定、先验的观测来源识别、顶层递归残差化、基于累积量匹配的父子判定、阈值剪枝与Bootstrap置信区间验证；

**📊 数据集**

实验数据：三种合成DAG模型（不满足ReLVLiNGAM局部约束）以及真实生物学数据Sachs蛋白质数据（11个变量，7467个样本）；

**📈 对比分析**

与原始ReLVLiNGAM、改进版ReLVLiNGAM、ParceLiNGAM和RCD进行了比较；在合成实验中，提出方法在所有评估指标（正确恢复DAG的比例、PRE、REC、F1）上均优于ReLVLiNGAM；在真实数据中，提出方法的观测DAG与参考网络相差仅一条边；改进版ReLVLiNGAM在大样本时可与其相当，但在稀疏场景和潜变量–观测关系恢复上表现不佳；

**⚠️ 局限性**

局限性：依赖高阶累积量估计，样本量小或数据噪声大时性能下降；当需要检验大量候选总效应时计算成本较高；算法对模型稀疏度敏感，密集网络可能导致效率低下。

---

## 140. Energy-Efficient GPU DVFS for Fine-Tuning of SLMs on Resource-constrained Embedded Devices

**arXiv ID:** 2607.05933 | [PDF](https://arxiv.org/pdf/2607.05933v1)

**作者:** Jurn-Gyu Park `[一作]` (Nazarbayev University), Ademi Zhanuzakova `[通讯]` (Nazarbayev University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Jetson AGX Orin平台上，对小语言模型（BERT、Pythia）微调过程的GPU频率进行细粒度调度实验与分析，并提出基于决策树的能耗最优频率治理策略；

**💡 创新点**

首次将细粒度GPU DVFS与轻量级ML决策模型结合，用于在资源受限的嵌入式设备上实现SLM微调的能耗优化；

**🔧 技术方法**

使用GPU DVFS、CUDA+PyTorch微调、tegraStats功耗采样及决策树（Decision Tree）驱动的频率治理算法；

**📊 数据集**

采用GLUE基准任务（SST‑2、QNLI、MRPC等）上的BERT与Pythia变体，同时在QQP、RTE、CoLA等未见任务上进行验证；

**📈 对比分析**

通过与Jetson默认的MAXN（无功率上限）模式对比，使用平均能耗百分比差值评估，平均节能率约为13.11%，最高可达26.73%，表明该方法优于默认模式；

**⚠️ 局限性**

数据量有限、决策树模型精度中等，受限于模型种类与任务多样性，未能覆盖更大或更异构的SLM，且未采用更复杂的ML模型或自适应调度策略。

---

## 141. SparseCtrl-HOI: Sparse Temporal Control for Human-Object Interaction Video Generation

**arXiv ID:** 2607.05994 | [PDF](https://arxiv.org/pdf/2607.05994v1)

**作者:** Shenbo Xie `[一作]` (South China University of Technology), Changxing Ding `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SparseCtrl-HOI 框架，使用稀疏时间控制（仅四帧关键帧）生成人-物交互视频，显著降低标注成本。

**💡 创新点**

创新点在于：①TiRoPE 通过时间控制的旋转位置编码实现关键帧的精确时间对齐；②Motion Prior Injection 利用 MLLM 提取高层运动先验并通过 Q-Former 注入 DiT，生成自然过渡；③分离训练策略避免 MLLM 与生成模型特征纠缠。

**🔧 技术方法**

核心技术包括：Wan2.1‑DiT 生成模型；VAE 编码器；TiRoPE；Q-Former 与 Qwen2.5‑VL 进行运动先验提取；LoRA 微调；多模态音频嵌入；跨模态交叉注意力层。

**📊 数据集**

使用了新构建的 SparseHOI‑5K 数据集（4,850 条 10 秒视频，含手物掩码、无物体视频和同步音频），并在 AnchorCrafter 测试集上进行外部验证。

**📈 对比分析**

通过 FID、FVD、MS、TF、AQ、Sync‑C、MS‑RAFT、HOI‑VLM 等指标与 AnchorCrafter、OmniAvatar、VACE、Phantom 等基线对比，SparseCtrl-HOI 在所有指标上均优于基线，尤其在 FID/FVD、MS‑RAFT、HOI‑VLM 上显著提升。

**⚠️ 局限性**

局限性：对极端视角变化或复杂物体仍可能出现轻微失真；依赖 MLLM 可能导致偶发 hallucination；推理速度相对较慢；仅在单人单物场景验证，未覆盖多人或多物体交互。

---

## 142. Multimodal Video-to-Music Recommendation via Semantic Retrieval and Temporal Reranking

**arXiv ID:** 2607.05971 | [PDF](https://arxiv.org/pdf/2607.05971v1)

**作者:** Seungheon Doh `[一作]` (KAIST), Juhan Nam `[通讯]` (KAIST)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VTMR两阶段视频到音乐推荐框架，先用多模态语义检索取候选，再通过时序重排序匹配精细对应。

**💡 创新点**

创新点在于将视觉、音频、文本联合编码到共享空间，并在重排阶段直接对未池化的时间序列进行跨模态注意力，解决单向量瓶颈。

**🔧 技术方法**

使用PEAV-base多模态编码器、Transformer跨编码器、SigLIP对比损失、BCE+margin排名损失。

**📊 数据集**

构建了包含MMtrail-2M、MovieLens-Content及内部广播数据的多源视频数据集，并生成伪标签进行训练。

**📈 对比分析**

与AudioCLIP、Wav2CLIP、ImageBind等基线比较，R@10从14.2提升至18.3，MedR从75降至46，显示显著性能提升。

**⚠️ 局限性**

限制在于依赖大量预训练模型和人工标注，跨模态对齐仍受限于固定长度截断，且对缺失模态的鲁棒性尚未彻底验证。

---

## 143. Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents

**arXiv ID:** 2607.06223 | [PDF](https://arxiv.org/pdf/2607.06223v1)

**作者:** Yijun Zhang `[一作]` (Shanghai Jiao Tong University), Xinbing Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种信息增益驱动的树状回滚策略（IGRPO），用于在有限搜索预算下训练搜索增强型大语言模型代理，重点关注中间状态的有用性。

**💡 创新点**

创新点包括：①将节点级信息增益作为回滚预算分配的软指标；②证明信息增益回滚产生的分布是教师分布，可作为明确的策略优化目标；③将自适应树结构探索与策略学习统一到单一框架。

**🔧 技术方法**

技术手段包括信息增益（IG）评估、基于树的预算分配、PPO/GRPO风格的群组策略优化、KL正则化、以及Qwen2.5-3B/7B-Instruct + E5检索器等实现。

**📊 数据集**

实验数据集为七个搜索增强问答基准：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MusiQue 与 Bamboogle。

**📈 对比分析**

与链式、树式及多种强化学习基线进行对比，IGRPO在所有基准上平均提升约3.1%（3B模型）和0.9%（7B模型），在单跳和多跳任务中均表现优异，尤其对小模型效果更明显。

**⚠️ 局限性**

局限性：依赖可量化的“信息增益”信号，若任务难以定义此信号则难以直接迁移；过度偏向信息增益也可能导致探索不足，需调节温度参数平衡探索与利用。

---

## 144. Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO)

**arXiv ID:** 2607.05585 | [PDF](https://arxiv.org/pdf/2607.05585v1)

**作者:** Fabio Hellmann `[一作]` (University of Augsburg), Elisabeth André `[通讯]` (University of Augsburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了基于人类表型本体(HPO)的面部表型识别框架

**💡 创新点**

创新点是将面部表型分解为层次化的子标签并采用层次化级联分类器减少样本不均衡问题

**🔧 技术方法**

使用了PointNet点云网络及软标签、特征融合和重要性分析等技术

**📊 数据集**

使用了GM数据库及从GM中标注的10个罕见疾病面部图像生成的GM‑10M数据集

**📈 对比分析**

与现有2D/3D预训练模型和面部关键点分类方法比较，平均AUROC约为0.80，表现优异

**⚠️ 局限性**

局限在于数据量有限、部分罕见表型样本不足且使用2D关键点生成的3D点云精度受限

---

## 145. Low-Overhead Error-Corrected QCNNs Using Bivariate Bicycle Codes

**arXiv ID:** 2607.05724 | [PDF](https://arxiv.org/pdf/2607.05724v1)

**作者:** Alejandro Rosales `[一作]` (Ohio University), Animesh Yadav `[通讯]` (Ohio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种低开销的量子卷积神经网络（QCNN）错误校正方案，将四比特QCNN编码到[[18,4,4]]双变量自行车码（BB码）中，并在每个隐藏层之间插入前馈神经网络（FFNN）软解码器以纠正错误。

**💡 创新点**

创新点在于将BB码的常数编码率和线性距离与QCNN的层级结构相结合，利用全局横向门实现量子门的等效转置，并通过可学习的FFNN在保持低物理比特数的同时实现对非Clifford门的错误抑制，从而显著提升在NISQ噪声环境下的训练可行性。

**🔧 技术方法**

采用的技术包括：量子卷积神经网络、双变量自行车量子错误纠正码、横向（transversal）门实现、BP‑OSD和SPSA优化的FFNN软解码器、噪声模型（单/双比特Pauli噪声）以及基于DMRG的S‑PT相位标记数据集。

**📊 数据集**

使用的数据集为一维自旋链的40个标注好的基态样本，来自无限大小的DMRG求解，覆盖 h₂ 轴不同点以检测 S‑PT 相位转换。

**📈 对比分析**

通过对标准QCNN、11比特横向QCNN以及BB码保护的QEC‑QCNN在不同噪声率（0%、0.01%、0.1%、0.3%）下的训练损失和预测分布进行比较，结果显示横向QCNN在低噪声下已具备一定鲁棒性，而加入BB码后在 0.1% 噪声下进一步降低了损失并实现了更好的收敛，说明低开销错误纠正提升了模型性能。

**⚠️ 局限性**

局限性包括：BB码的阈值仅约为 0.3%，在噪声率 0.3% 以上（如 0.003）仍无法实现训练收敛；非Clifford 门的错误仍未被训练时的 FFNN 完全学习，导致高噪声下性能下降；以及当前仅在 4 比特 QCNN 规模上验证，尚未验证更深更大规模网络的可扩展性。

---

## 146. Statistically Meaningful Geometry and Gauge Symmetry Breaking: A Geometric Foundation for Scientific Discovery and Intelligence Emergence

**arXiv ID:** 2607.05436 | [PDF](https://arxiv.org/pdf/2607.05436v1)

**作者:** Bing Cheng `[一作]` (Chinese Academy of Sciences), Shing-Tung Yau `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于无限维 Orlicz 纤维束的统计几何框架（Statistically Meaningful Geometry, SMG），通过将过参数化的学习系统拆分为可观测的水平基底与不可观测的垂直纤维，定义了“主动非因果张力”与“规范对称破缺”等概念，用几何方法刻画系统在面对 OOD 环境刺激时的自我发现与智能跃迁。

**💡 创新点**

创新点在于：①将传统机器学习中的欧氏平面拓扑升维为无限维统计流形；②引入水平/垂直纤维分离，证明垂直纤维中的未建模方差导致几何张力累积并触发 Gauge Symmetry Break；③给出可观测的指标（如结构 G‑Entropy 步骤跳跃、Active Acausal Tension 𝒯_AAT）的公式，使得“真正的科学发现”可被量化与验证。

**🔧 技术方法**

核心技术包括：无限维 Orlicz 流形的 Fisher 量度、Ehresmann 连接、非参数 Stein 分数函数、Jacobi 方程与 Sturm 归并定理、几何张力与曲率的动力学分析。论文通过严谨的变分、几何分解与能量最小化推导，展示了从数据惊讶到参数更新、到张力累积、再到非平衡相变的完整数学链条。

**📊 数据集**

本文并未给出具体实验数据集，而是以通用的生成式 AI 训练场景（大规模文本语料、OOD 科学论文片段等）为示例进行理论推导。若要实现实验验证，需使用大规模语言模型（如 GPT‑4 或更大规模模型）与对比 OOD 科学文本、金融时间序列等。

**📈 对比分析**

由于工作是理论性框架，未提供与现有方法的数值比较或性能指标。若实施实验，可通过监测 𝒯_AAT 增长速率、结构 G‑Entropy 突变点以及模型在 OOD 数据上的生成质量与推理准确率来评估 SMG 的有效性。

**⚠️ 局限性**

局限性包括：①数学推导高度抽象，对非专业读者门槛高；②需要对无穷维流形和非参数 Stein 分数等高级概念有深入了解；③尚未给出具体可实现的算法与代码；④在实际训练中计算 𝒯_AAT、G_f、Δ‑连接等量的成本与数值稳定性尚未评估；⑤假设模型能完整捕获水平基底，实际模型可能因硬件、优化器限制导致分解不完全。

---

## 147. A Decomposition-Based Framework for Joint Optimization and Spatial Packaging of Interconnected Systems with Physical Interactions

**arXiv ID:** 2607.06087 | [PDF](https://arxiv.org/pdf/2607.06087v1)

**作者:** Julien Bückmann `[一作]` (Eindhoven University of Technology), Theo Hofman `[通讯]` (Eindhoven University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在汽车系统级设计中提出一种空间包装与物理相互作用（SPI2）框架，能够将动力总成组件的定位、尺寸以及电池架构集成作为可优化变量，并与功率分配、车辆质量分布等子系统耦合，实现联合多目标优化。

**💡 创新点**

① 采用四元数与签名距离场（SDF）提升放置问题的数值稳健性与求解效率；② 将部件定位直接作为设计变量并加入端口对齐约束，打破传统的后置可行性检查；③ 通过 ATC‑启发式的二次罚项与 NSGA‑II 的种群搜索相结合，构成统一的系统级协调与多目标优化框架；④ 通过实验验证显著降低计算成本并提升 Pareto 前沿质量。

**🔧 技术方法**

数学优化技术：梯度基（IPOPT）求解、四元数旋转、SDF 边界约束、MDBD 对象表示；多目标优化：NSGA‑II、ATC‑启发式协调；仿真/评估：CasADi、Python3、Numpy；车辆动力学模型：WLTP 驱动循环、机械轴承力分配、功率地图插值、等效梁模型。

**📊 数据集**

使用自行构建的 Skoda Enyaq 类车辆基准模型作为实验对象，并对放置问题采用 1×1×0.5 单元盒子（14 个球）在 15×15×1.5 边界盒中进行基准比较；在系统级优化中以电池位置为变量，采用 5 mm、1 mm 分辨率的网格搜索作为离散化对照。

**📈 对比分析**

与传统 Euler 角/边界盒方法比较，使用四元数 + SDF 的方法求解率提升至约 80%，解质量提升到 90% 以内；在系统级多目标优化中，NSGA‑II 在第 6 代即可达到与 4080 次评估的离散网格搜索相同的 Pareto 前沿（HV 比值 1.036），计算时间由 136 h 降至 6.6 h，节约 95.11% 的计算资源。

**⚠️ 局限性**

① 几何建模仍采用低阶 MDBD 与等效梁，缺乏细节与高精度结构分析；② 仅在汽车动力总成与电池集成场景验证，尚未推广至其他系统或复杂度更高的布局；③ 依赖于多目标优化器（NSGA‑II），可能对大规模连续变量问题收敛速度有限；④ 部分物理耦合（如能量回收、制动分配）采用简化假设，可能影响精确性。

---

## 148. x-Prediction Is All You Need:Training-Free Accelerated Generation via Endpoint Decodability

**arXiv ID:** 2607.06114 | [PDF](https://arxiv.org/pdf/2607.06114v1)

**作者:** Xin Peng `[一作]` (Beijing University of Posts and Telecommunications), Ang Gao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的采样方法，称为截断跳跃采样（TJS），通过在采样过程中提前停止ODE并输出解码的x_0，从而减少神经函数评估（NFE）的数量。

**💡 创新点**

创新点在于引入了端点可解性（endpoint decodability）的概念，证明了在标准ℓ_2损失下，解码器是最小均方误差（MMSE）估计器，并且TJS不需要重新训练、蒸馏或架构更改。

**🔧 技术方法**

使用了基于x-预测的技术，结合了标准的ODE求解器和解码器来实现早期退出采样。

**📊 数据集**

实验使用了多个数据集，包括SDXL、SD3.5M、Z-Image-Turbo、ImageNet-256、CIFAR-10和MNIST等。

**📈 对比分析**

与现有方法相比，TJS在六个模型系列上减少了20%到70%的NFE，同时保持了接近匹配的质量，且在所有指标上都有严格的单调改进。

**⚠️ 局限性**

限制在于对于复杂数据，端点不确定性（𝒰(t)）的衰减较慢，直接的x_0预测尚未成为标准。

---

## 149. SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation

**arXiv ID:** 2607.05943 | [PDF](https://arxiv.org/pdf/2607.05943v1)

**作者:** Zhengbo Jiao `[一作]` (MMLab, CUHK), Xiangyu Yue `[通讯]` (MMLab, CUHK)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个统一的知识图搜索环境，训练多模态搜索代理实现多跳视觉推理。

**💡 创新点**

提出Perception‑Knowledge Chain数据合成、hop‑anchored policy optimization以及自包含搜索世界，解决训练数据、环境、奖励三者解耦问题。

**🔧 技术方法**

基于Qwen3.5/3.5‑27B大模型，使用强化学习（GRPO改进HaPO）、知识图（Wikidata5M）、混合检索（BM25+dense）与工具调用框架。

**📊 数据集**

使用Wikidata5M、Wiki6M、Wikipedia图像构建的约1.2M实体知识图，生成的PKC问题和VisSearch Bench。

**📈 对比分析**

在六个多模态知识检索基准上，与闭源/开源模型对比，SearchEyes‑27B平均得分68.1，领先同类开源模型多达4–6分，并在VisSearch Bench上表现突出。

**⚠️ 局限性**

仍受限于知识图覆盖范围、检索质量以及多跳推理中的错误传播，且在极长链任务中性能下降。

---

## 150. Ground3D-LMM: Fine-Grained 3D Point Grounding and Spatial Reasoning with LMM

**arXiv ID:** 2607.05493 | [PDF](https://arxiv.org/pdf/2607.05493v1)

**作者:** Amol Harsh `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fahad Khan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的点云多模态语言模型Ground3D‑LMM，能够在自然语言对话中生成文本回复、对应点级掩码以及以真实物理单位给出的度量值；

**💡 创新点**

创新点在于：①将点云特征直接嵌入到大型多模态语言模型中，实现语义与几何的联合推理；②通过特殊触发标记在生成文本时实时调用轻量化分割头，完成对话中动态的点级定位；③构建全新3D Grounded Measurement任务与大规模Ground3D数据集，统一衡量语义定位与度量准确性；

**🔧 技术方法**

采用稀疏卷积+超点池化的点云编码器、Sparse 3D U‑Net、线性投影映射至LMM（基于Qwen3‑VL‑4B‑Instruct），并使用多任务损失（BCE+Dice + 语言交叉熵）训练；

**📊 数据集**

数据集为Ground3D，基于ScanNet和ScanNet++构建，包含约2.5M问答对，密集的对象与部件注释、3D掩码以及多种度量属性；

**📈 对比分析**

与Image Baseline、UniSeg3D、Reason3D、MLLM‑For3D等方法对比，Ground3D‑LMM在对象与部件分割、度量答案、文本质量等指标上均取得显著提升；在ScanRefer和Reason3D上亦实现了大幅度的性能提升；

**⚠️ 局限性**

局限性包括：对点云和深度信息高度依赖，可能对稀疏或噪声严重的场景表现不佳；在细粒度部件分割任务上仍有一定误差；模型规模和计算成本较大，实际部署需要进一步优化。

---

## 151. Revisiting Scene Graph Generation from the Perspective of Detector-Conditioned Reachability

**arXiv ID:** 2607.06176 | [PDF](https://arxiv.org/pdf/2607.06176v1)

**作者:** Runfeng Qu `[一作]` (Technische Universität Berlin), Olaf Hellwich `[通讯]` (Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对检测器条件可达性视角下检测器式与查询式场景图生成模型的预测行为进行系统对比，并基于其互补性提出 Dual‑SGG 通过双查询（TD‑Q 与 BU‑Q）在同一解码器内联合推理，从而实现更全面的三元组预测。

**💡 创新点**

创新点包括：①首次将检测器可达性拆分为 Det‑T 与 UDet‑T 子集量化模型差异；②引入双查询设计，将检测器依赖的“自上而下”查询与不受检测器限制的“自下而上”查询融合；③在单一解码器内完成两种推理，避免了后处理融合的标度校准问题；④通过实体对选择器 EPS 提升查询质量，并在 BU‑Q 初始化时采用中心偏置锚实现全局探索。

**🔧 技术方法**

核心技术包括：Deformable‑DETR 作为基础检测器；Entity Pair Selector (EPS) 用于过滤高质量实体对；双查询（TD‑Q、BU‑Q）在 Transformer 解码器中联合更新；自注意力掩码控制 TD‑Q 与 BU‑Q 的信息流；多任务损失（检测、EPS、TD‑Q、BU‑Q）实现端到端训练；可选 Logit Adjustment 进行长尾去偏。

**📊 数据集**

在 Visual Genome、Open Images V6 与 GQA‑200 三大公开数据集上进行实验，覆盖 150/601/200 物体类别与 50/30/100 关系类别，采用 Recall@K、mRecall@K、F@K 等标准指标。

**📈 对比分析**

与目前主流基线（Motifs、SSR‑CNN、RelTR、EGTR 等）以及无偏见 SGG 方法（BGNN、SSR‑CNN+LA、EGTR+LA、Salience‑SGG）进行对比。Dual‑SGG 在 VG、GQA‑200、OIv6 上均取得最高或最接近最高的 R@K、mR@K 与 F@K，并在强大检测器条件下仍保持显著优势；在加上 LA 的无偏见设置下，Dual‑SGG+LA 在 mR 与 F 上实现了新的 SOTA。

**⚠️ 局限性**

局限性包括：①对强检测器条件的依赖——当检测器覆盖率提升时 UDet‑T 减小，双查询优势减弱；②在某些缺失的三元组仍难以被双查询补偿，尤其是对极其稀疏或极端位置的关系；③中心偏置锚初始化虽提升全局探索，但在极端图像分布下可能导致查询偏移；④模型相对更大且需要更多的查询数，计算成本仍高于纯查询式方法。

---

## 152. Unicode TAG-Block Concealment of Tool-Metadata Payloads in the Model Context Protocol: An Approval-View Fidelity Gap Across Three Independent Server Implementations

**arXiv ID:** 2607.05744 | [PDF](https://arxiv.org/pdf/2607.05744v1)

**作者:** Mohammadreza Rashidi `[一作]` `[通讯]`, Mohammadreza Rashidi

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在 Model Context Protocol（MCP）中识别并量化了工具元数据被攻击者利用的隐蔽渠道，首先通过模型无关、协议无关的 Unicode 代码点分析预测哪些编码能绕过用户审批视图，然后在真实的 MCP 客户端/服务器上实现并验证八种不同的注入技术，检验它们是否能够送达模型、绕过基线字符串过滤器、隐藏于审批界面，以及是否触发协议层面的重新同意。

**💡 创新点**

创新点在于：
1) 将隐蔽编码问题抽象为「审批视图渲染与模型上下文传递路径不一致」的单一机制，并用 Unicode TAG 块的无字形特性预言其可行性；
2) 在三套独立的 Python MCP 服务器实现上复现所有技术，证明这些缺陷是协议层面的、非实现细节导致；
3) 给出四条结构化修复建议（字节级同意、重新同意、来源限定命名空间、方案默认值非同意）。

**🔧 技术方法**

使用技术包括：
- JSON‑RPC/stdio 的真实 MCP 通信协议；
- 字符串匹配基线过滤器（基于 imperative + sensitive 关键词）；
- Unicode TAG‑block 编码实现（`tag_encode`）；
- 真实的 MCP 客户端/服务器交互 harness；
- 计数式协议层观察（四项指标）。

**📊 数据集**

数据集主要是手工编写的代表性工具元数据语料（文件、版本控制、聊天等类别）以及八种攻击技术的 payload；未使用公开大规模数据集。

**📈 对比分析**

比较方法为对每种技术记录四项 deterministic 观察：是否到达模型、是否被基线过滤器拦截、是否在审批视图中可见、是否触发重新同意。结果显示：所有技术均能到达模型；T1、T2、T5、T3 被过滤器拦截；T4、T8、T6、T7 逃避过滤器；只有 T7 还能隐藏于审批界面；T3 在协议层触发重新同意。性能表现为完全确定性，没有统计误差。

**⚠️ 局限性**

局限性包括：
- 仅检验 payload 到达模型的事实，未评估模型实际是否执行；
- 基线过滤器和审批渲染均为简化模型，未覆盖更复杂的语义过滤或渲染规范；
- 只覆盖了八种攻击技术和五种元数据表面，未系统性扫描所有潜在表面；
- 评估基于单一客户端实现的渲染逻辑，假设渲染器不处理未分配字形；
- 未测量不同 LLM 对同一 payload 的实际行为差异。

---

## 153. OBBSeg: Irregular Lesion Segmentation under Oriented Bounding Box Annotations

**arXiv ID:** 2607.06007 | [PDF](https://arxiv.org/pdf/2607.06007v1)

**作者:** Jun Wei `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种以定向边框（OBB）为中间监督的弱监督医学图像分割方法 OBBSeg。

**💡 创新点**

创新点包括：① Mask-to-OBB（M2O）损失，消除传统框式偏差；② Prompt-assisted Foreground Enhancer（PAFE）和 Differential-based Foreground Enhancer（DBFE）两种特征增强模块；③ 将提示（点/点、线、圆、涂鸦）注入编码器，实现层级语义引导。

**🔧 技术方法**

采用 ViT/SAM2 编码器，融合 M2O 损失、提示监督损失、尺度一致性损失；使用矩阵变换实现 OBB 与掩膜的几何一致性；PAFE 与 DBFE 通过前馈与残差机制提升前景与背景区分。

**📊 数据集**

在 13 个公开数据集（肠镜、皮肤病、MRI、CT、超声）共 5 种影像模态上进行评估，涵盖 Kvasir、ClinicDB、ColonDB、ETIS、Endo、SUN-SEG、ISIC-2017/2018、NCI-ISBI、Synapse、TN3K、DDTI、BUSI 等。

**📈 对比分析**

与完全监督方法（U‑Net、PraNet、SAM2 等）以及其他弱监督方法（WeakPolyp、BoxInst、AGMM 等）比较。OBBSeg 在弱监督设置下平均 Dice 近 95%，在多模态数据集上性能仅落后 1‑2% 于全监督模型，显著优于现有弱监督方法。

**⚠️ 局限性**

局限性：① 仅使用 2D 影像，未处理 3D 或视频序列；② OBB 的投影机制限制了对复杂边界与拓扑结构的精细学习；③ 对极其细长或高度不规则病灶仍可能产生误分；④ 对 OBB 注释的误差较为鲁棒，但仍受限于人类标注误差。

---

## 154. Determinantal point process sampling for bioacoustic active learning

**arXiv ID:** 2607.06063 | [PDF](https://arxiv.org/pdf/2607.06063v1)

**作者:** Hugo Magaldi `[一作]`, Gabriel Dubus `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合类别平衡不确定性、嵌入空间新颖度、探索-利用权重退火以及确定性点过程（DPP）批次多样性的主动学习采样器，专为生态声学监测中的多标签分类任务设计。

**💡 创新点**

创新点在于：1）使用类别平衡的多标签熵来抵消类别不平衡对不确定性采样的影响；2）引入余弦新颖度与不确定性按预算进度退火平衡；3）在候选池中混入随机探索样本以缓解早期模型不可靠；4）采用DPP进行批次级多样性选择，显著减少冗余。

**🔧 技术方法**

核心技术包括：多标签熵、类别加权、余弦新颖度、权重退火、候选池构造、确定性点过程（DPP）批次分选、基于梯度的自适应批次调度、固定学习率与训练周期的基线模型。

**📊 数据集**

在BioDCASE 2026基准上使用BirdSet（HSN、POW、UHH子集）和ATBFL四个数据集进行实验，训练数据量分别为6,600、2,280、18,319、9,086段，类别数分别为19、41、25、7。

**📈 对比分析**

与官方基线CoreSet、TypiClust、Margin和随机采样比较，所提方法在500次注释预算下平均AULC为0.5017，明显优于CoreSet的0.4600和其他基线；在各数据集上AULC均提升，尤其在HSN和UHH上显著。

**⚠️ 局限性**

局限性包括：1）参数如探索比例、权重退火曲线等需手工设定，缺乏自动调优；2）在不同数据集上各子模块贡献差异，缺乏通用最优配置；3）DPP计算复杂度高，候选池大小受限；4）对极少量标记或高度不平衡的极端情况效果未充分验证。

---

## 155. Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning

**arXiv ID:** 2607.05939 | [PDF](https://arxiv.org/pdf/2607.05939v1)

**作者:** Timothée Gavin `[一作]` (Thales Group), Murat Bronz `[通讯]` (Université de Toulouse)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多机拦截单一逃逸机任务，提出基于MARL的竞争式自我对弈方法，使追捕机和逃逸机都能学习高机动低层控制策略；

**💡 创新点**

创新点在于将Prioritized Fictitious Self‑Play（PFSP）与低层推力/姿态率控制相结合，解决了非平稳性与灾难性遗忘问题，并实现了多机协同拦截；

**🔧 技术方法**

使用了多智能体近端策略优化（MAPPO）+PFSP、低层推力-姿态率（CTBR）控制，以及JAX框架下的大规模并行仿真；

**📊 数据集**

使用的是自建的高保真四旋翼仿真器（含碰撞、低层INDI控制），没有真实数据集；

**📈 对比分析**

与多种基准（FRPN、APF、FRPN+APF、无自我对弈、仅自我对弈、速度控制版）进行对比，PFSP‑CTBR在捕获率最高、抓捕时间最短、碰撞率最低，显著优于基线；

**⚠️ 局限性**

局限在于仅在仿真中验证，缺乏传感器噪声、未建模动力学和实际环境干扰，且逃逸机受限于中心体积，未来需转移到真实无人机并加入感知与状态估计。

---

## 156. AgoraSim: A Hybrid Agent-Based Modeling Framework

**arXiv ID:** 2607.05999 | [PDF](https://arxiv.org/pdf/2607.05999v1)

**作者:** Chung-Chi Chen `[一作]` `[通讯]` (National Institute of Informatics), Chung-Chi Chen (National Institute of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了AgoraSim，一套混合式代理模拟框架，用自然语言或多模态输入生成可编辑的ABM配置，并在同一实验中运行LLM、VLM、定制端点、随机与经典规则代理，生成共享结构化决策记录。

**💡 创新点**

创新点在于将LLM等大模型嵌入ABM中，实现统一决策接口；通过比例控制混合模型并与经典参考动力学直接对齐；同时提供可视化UI、SDK和REST API，让用户能在同一配置下比较不同代理类型的轨迹。

**🔧 技术方法**

技术包括OpenAI/LLM调用、视觉语言模型、Python SDK/CLI、REST API、本地UI、ABM引擎、共享决策对象、网络与交互协议、记录审计、成本追踪。

**📊 数据集**

主要使用用户自定义的情境文本/多模态素材作为输入；未采用公开大规模社交数据集，而是通过合成人口和网络结构进行实验。

**📈 对比分析**

比较方法是将同一情境下的混合运行与多个经典ABM参考模型（阈值/Bass、有限信任、SIR、群居、DeGroot、离散选择）共享相同动作空间、指标与种子，直接对比轨迹和指标。性能方面侧重可解释性与对比清晰度，未给出数值预测准确度。

**⚠️ 局限性**

局限性包括：生成结果为合成轨迹，缺乏外部验证；LLM可能倾向平均响应，难以捕捉真实多样性；网络与记忆机制简化；可能产生偏见与刻板印象；对敏感内容使用需谨慎；未提供大规模实验或预测性能评估。

---

## 157. Parameter-Free Encoders Remain Viable for RDB Foundation Models

**arXiv ID:** 2607.05476 | [PDF](https://arxiv.org/pdf/2607.05476v1)

**作者:** Linjie Xu `[一作]` (University of Hong Kong), David Wipf `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨在关系型数据库基础模型中，参数化与无参数子图编码器的有效性，并证明在给定标签输入的情况下，训练可学习的编码器并不一定优于简单无参数编码器

**💡 创新点**

提出理论证明，说明即使利用邻域标签作为特征或特征重要性评估，固定无参数编码器在跨任务场景下仍能保持足够表达力；并通过实验证明无参数编码器在多基准任务上可匹敌甚至超越含参数的闭源模型

**🔧 技术方法**

基于图神经网络/图Transformer的无参数子图编码器、单表基础模型（如Transformer预测头）、ICL（in-context learning）框架、以及对邻域标签的聚合与理论分析

**📊 数据集**

六个多样化的关系型数据库预测基准（RelBench等）

**📈 对比分析**

与多种开源和闭源基础模型以及传统监督基线进行对比，使用AUROC等指标，结果显示RDBLearn（无参数编码器）在多数基准上获得最高或相当于最优性能

**⚠️ 局限性**

理论与实验均针对固定无参数编码器的通用性，未完全解决在极端异构或低标签覆盖率数据库中的性能下降可能性，以及对动态或大规模实时更新数据库的适应性尚待进一步研究

---

## 158. Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search

**arXiv ID:** 2607.05970 | [PDF](https://arxiv.org/pdf/2607.05970v1)

**作者:** Riccardo Terrenzi `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用大语言模型生成RDF数据集元数据对检索效果与真实性的影响，并在不同生成设置下进行评估。

**💡 创新点**

提出了从无约束重写到基于数据集概况和代理式图遍历的多种生成场景，并系统比较其检索性能与真实性的权衡。

**🔧 技术方法**

使用了LLM（如ChatGPT）、RDF概况提取、图工具交互、BM25与Dense检索等技术。

**📊 数据集**

使用ACORDAR 2.0子集，约1000个RDF数据集进行实验。

**📈 对比分析**

通过NDCG、MAP等指标比较检索性能，并使用LLM判定与RDF证据对照的真实性；结果显示无约束重写提升检索但真实性最低，基于概况的重写在两者间取得最佳平衡。

**⚠️ 局限性**

局限包括概况信息不足导致检索下降、代理式生成仍易产生实体/范围错误、实验规模有限、对实际大规模检索的泛化性待验证。

---

## 159. Differentially Private Natural Gradient Descent

**arXiv ID:** 2607.05866 | [PDF](https://arxiv.org/pdf/2607.05866v1)

**作者:** Pan Li `[一作]` (Chinese Academy of Sciences), Jinwen He `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在固定隐私预算下，设计了一种能充分利用二阶曲率信息的差分隐私自然梯度下降（DP-NGD）框架，避免了传统DP-SGD的几何盲点，实现更高效的优化；

**💡 创新点**

核心创新在于：① 将曲率估计完全基于无私有数据的公共辅助集，消除隐私预算消耗；② 通过F⁻¹/²白化空间的DP操作，将等距的隐私约束与不等距的自然梯度预调和；③ 采用动态曲率截断（clamping）保证训练稳定；

**🔧 技术方法**

技术手段包括：K‑FAC近似的Fisher信息矩阵估计、白化空间的梯度裁剪与噪声注入、动态曲率截断、隐私预算使用Rényi DP计数；

**📊 数据集**

在CIFAR‑10、SVHN和UTKFace三大公开图像数据集上进行实验，使用WRN‑16‑4和ResNet‑20模型；

**📈 对比分析**

与DP‑SGD、DP‑SGD‑PT、AdaDPS和GEP等基线对比，DP‑NGD在所有隐私预算（ϵ=1.0–8.0）下均取得最高准确率，并在低预算下显著提升4–5个百分点；在训练步数上可比传统方法快3–10倍，训练时长与或更少；

**⚠️ 局限性**

局限性包括：① 对公共辅助数据仍有一定依赖，尽管样本量极小；② 需要先验的K‑FAC或类似二阶结构估计，对超大模型的可扩展性待验证；③ 动态截断参数仍需经验性设定，可能对不同任务产生微小偏差。

---

## 160. SCOReD: Student-Aware CoT Optimization for Recommendation Distillation

**arXiv ID:** 2607.05734 | [PDF](https://arxiv.org/pdf/2607.05734v1)

**作者:** Haz Sameen Shahgir `[一作]` (University of California Riverside), Yue Dong `[通讯]` (University of California Riverside)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种针对推荐系统的链式推理（CoT）压缩框架SCOReD，通过教师模型的分段、注意力重要性评分和奖励导向的编辑决策，生成更简洁、更有针对性的推理轨迹供小模型学习。

**💡 创新点**

创新点在于：①将CoT压缩与学生模型的行为紧密耦合，利用学生对最终答案的概率和困惑度进行局部编辑选择；②在压缩过程中保留关键的候选比较步骤，显著降低冗余的验证与重检；③通过三阶段策略（分段、评分、奖励）实现可解释且可操作的压缩。

**🔧 技术方法**

使用的技术包括：Gemma‑4‑26B教师模型进行CoT生成与分段；Qwen‑3‑0.6B小模型进行监督微调；注意力权重作为重要性度量；基于log‑prob、长度与困惑度的奖励函数；对比实验采用LLM一键压缩、标准SFT、后置强化学习与自监督蒸馏。

**📊 数据集**

使用的公开数据集为Amazon Beauty Pretrain，构造的购买历史和候选列表，训练集约1.13万条、验证集4.47千条、测试集4.47千条。

**📈 对比分析**

相较于标准SFT和LLM一键压缩，SCOReD在NDCG、MRR、MAP等指标均提升（如NDCG提升0.79→0.79+≈1.6%），并将推理长度缩减27%、解析失败率降低46%，在0.6B模型上接近26B教师的表现。

**⚠️ 局限性**

局限性包括：①压缩仍需教师模型的算力支持；②在极度噪声的推荐标签下，教师推理路径可能不唯一，压缩仍可能丢失有价值的逻辑步骤；③后置强化学习和自监督蒸馏对已有SFT模型提升有限，表明进一步优化难度较大。

---

## 161. When Should LLMs Search? Counterfactual Supervision for Search Routing

**arXiv ID:** 2607.05752 | [PDF](https://arxiv.org/pdf/2607.05752v1)

**作者:** Minho Kim `[一作]` `[通讯]` (Sangmyung University), Minho Kim (Sangmyung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了搜索增强型语言模型在每个问题上是否需要调用搜索工具的决策问题；

**💡 创新点**

创新点在于通过比较无搜索和强制搜索的结果，构造基于实例的路由奥里克（oracle）作为监督信号，既可用于评估又可用于训练；

**🔧 技术方法**

采用了监督微调（SFT）和偏好优化（Preference Optimization）两种训练技术，并通过固定的搜索接口与提示实现一致的实验环境；

**📊 数据集**

使用了PopQA（事实性实体问答）和KUQ（错误前提与含糊问题）两大数据集来构建对照实验；

**📈 对比分析**

在基准模型Gemma E2B和Qwen3.5-4B上，路由宏F1从约0.71提升至0.82-0.84，显著降低了过度搜索和漏检搜索的错误率；

**⚠️ 局限性**

局限性包括：1）奥里克仅在当前模型、工具和检索条件下有效，无法反映所有情况；2）仅关注首次搜索调用，未优化后续检索和答案生成；3）难以将失败案例分解为具体原因，诊断仍为经验性。

---

## 162. The yes-no bias of large language models reflects answer order and wording, not shifts in moral judgment

**arXiv ID:** 2607.05552 | [PDF](https://arxiv.org/pdf/2607.05552v1)

**作者:** Haonan Huang `[一作]` `[通讯]` (Princeton University), Haonan Huang (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过交叉对称化的心理测量电池，系统分离大型语言模型在道德判断中主观立场与表面格式引起的偏差。

**💡 创新点**

创新点在于提出的跨框架测量方法，可将“是-否”偏差拆解为排序效应和词汇拉力，并用两个可迁移参数（框架敏感度和道德决定性）量化。

**🔧 技术方法**

技术手段包括跨逻辑等价句式的交叉对称化、分层评分、自由选择与强制二元判断实验，以及对模型输出进行logistic拟合。

**📊 数据集**

实验数据来源于Cheung等人2024年的20条道德困境文本，并在七个前沿模型（Claude、GPT‑5.5、Gemini）以及两个轻量级开放权重模型上进行多重实验。

**📈 对比分析**

与传统单一框架评估相比，跨框架测量在前沿模型中实现了0.12–0.21的格式不一致性，且“是-否”偏差在Claude模型可高达‑0.86，而GPT‑5.5和Gemini几乎无偏差，显示方法能更准确捕捉模型真实立场。

**⚠️ 局限性**

局限性包括仅使用单一困境集合、未收集新的人类数据、对模型人格和系统提示的假设保持不变，且方法对非常小或未调优的模型的适用性尚未验证。

---

## 163. X-FEMR: A Token-level Explainable Approach for Electronic Health Records Foundation Models using Transformer-based Models

**arXiv ID:** 2607.06163 | [PDF](https://arxiv.org/pdf/2607.06163v1)

**作者:** Jie Huang `[一作]` (University of Melbourne), Ting Dang `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了针对电子健康记录（EHR）Foundation Models（FEMR）的第一种基于token的可解释方法，使用Transformer surrogate和SHAP进行逐token重要性分析，并设计临床验证事件比例指标评估解释与医学知识的一致性。

**💡 创新点**

创新点在于：①将Transformer surrogate嵌入FEMR的黑盒行为，实现对长序列EHR的时序建模；②引入临床验证事件比例这一量化指标，将模型解释与临床可验证特征直接对齐；③在硬标签和软标签两种监督下比较解释质量。

**🔧 技术方法**

技术包括Transformer编码器、时间间隔归一化、门控数值注入、SHAP特征归因、软硬标签对比训练与logit转换。

**📊 数据集**

使用公开的CLMBR-T-Base 141M模型和EHRSHOT大规模纵向EHR数据集，分别在LOS（最长住院时间≥7天）和ICU转移预测任务上进行实验。

**📈 对比分析**

在硬标签监督下，surrogate模型在LOS任务的AUPRC 0.4434、F1 0.4889略优于CLMBR‑T‑Base（0.4173、0.3717），ICU任务的AUPRC 0.2701、F1 0.2609同样高于原模型（0.1592、0.1149）。软标签监督下性能略逊，说明硬标签更能捕捉模型决策边界。

**⚠️ 局限性**

主要限制包括：1）surrogate与原模型对齐不完美，导致解释可能失真；2）长序列EHR的复杂性与噪声导致训练难度大；3）数据量不足限制Transformer的学习；4）事件token聚合方式可能掩盖细粒度信息，导致解释失去局部细节。

---

## 164. BlueMagpie-TTS: A Token-Efficient Tokenizer, Language Model, and TTS for Taiwanese-Accent Code-Switching Speech

**arXiv ID:** 2607.06054 | [PDF](https://arxiv.org/pdf/2607.06054v1)

**作者:** Ho Lam Chung `[一作]`, Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合台湾语境文本、tokenizer、语言模型前端和现有高质量音频堆栈，提出端到端台湾本土化TTS体系BlueMagpie‑TTS，并在台湾本土化测试集上显著提升语音质量与可懂度。

**💡 创新点**

① 采用Byte‑level BPE训练的PangolinTokenizer，专门针对台湾文本优化，显著降低token率并保持lossless；② 在此tokenizer上训练的传统中文语言模型Barbet，充当语义‑韵律规划器；③ 通过桥接模块将Barbet与已有VoxCPM2声学堆栈无缝连接，并联合微调，证明前端决定性影响口音与代码切换质量。

**🔧 技术方法**

Byte‑level BPE、子词分词、混合全局/滑窗/状态空间注意力(4‑层循环)语言模型、RMSNorm+线性+SwiGLU桥接、VoxCPM2扩散式语音堆栈、联合微调、synthesize‑then‑recognize循环评估、ASR（Breeze‑ASR‑25）与人工听力实验。

**📊 数据集**

① 约20B token的台湾语境文本语料（包含繁体中文、台语、客语、Bopomofo、代码切换、结构化文本等子集）用于tokenizer；② Barbet预训练的传统中文文本；③ 1,000句台湾本土化测试句子；④ 10名本地听众的听力实验；⑤ 用于评估的ASR数据。

**📈 对比分析**

采用synth‑then‑recognize循环评估CER和500句听力对比。BlueMagpie‑TTS在CER上从11.45%降至4.81%（相对58%下降），比仅微调VoxCPM2的6.43%又提升25.2%。听力偏好中65.6%选择BlueMagpie‑TTS。实时因子约4.75×。

**⚠️ 局限性**

① 无法独立量化tokenizer与前端对CER的单独贡献；② 仅评估短句子，未检验Barbet 262K上下文的长文本优势；③ 对短英文字母缩写发音仍不佳；④ 方案高度依赖已有强声学堆栈，若无该堆栈需重新训练。

---

## 165. Quaternion-Averaging-Based Adaptive Complementary Filter for Pedestrian Dead Reckoning With a Foot-Mounted AHRS

**arXiv ID:** 2607.05451 | [PDF](https://arxiv.org/pdf/2607.05451v1)

**作者:** Shunsei Yamagishi `[一作]` (University of Aizu), Lei Jing `[通讯]` (University of Aizu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于四元数平均的自适应互补滤波器（QAACF），用于脚部安装的AHRS实现室内行人死 reckoning。

**💡 创新点**

创新点包括：使用Markley四元数平均方法严谨融合两四元数，结合基于步态相位与磁场扰动的自适应权重调节，提高姿态估计精度并降低计算量。

**🔧 技术方法**

采用四元数平均、梯度下降求重力方向、Wahba问题解法求磁场四元数、以及与Kalman滤波器和传统互补滤波器的对比实验等技术。

**📊 数据集**

利用MTW2-3A7G6惯性测量单元与OptiTrack光学系统采集的多段室内步态数据（Data A‑1~C‑3、Data D‑1~D‑3）进行实验。

**📈 对比分析**

通过与多种Kalman滤波器（EKF、FKF、RMr‑GDALKF、KCKF）和互补滤波器（FCF、Madgwick、Mahony）对比，QAACF在姿态RMSE、轨迹RMSE、行走距离误差上均优于其他滤波器，且计算时间显著更低。

**⚠️ 局限性**

局限性包括：对高精度陀螺仪、加速度计、磁力计的依赖；需要静态校准和磁力计椭圆校准；未验证在高速或跑步等高动态条件下以及户外磁场更稳定环境中的性能。

---

## 166. A Dual-CRDT Architecture for Decentralized Trust Governance and Evolution

**arXiv ID:** 2607.06068 | [PDF](https://arxiv.org/pdf/2607.06068v1)

**作者:** Amos Brocco `[一作]` `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland), Amos Brocco (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个由 Trust CRDT 和 Data CRDT 组成的双 CRDT 架构，实现去中心化的信任治理与演化。

**💡 创新点**

将治理规则本身作为可复制的状态，并通过递归治理实现信任与数据的共同演化。

**🔧 技术方法**

基于 CRDT、确定性重建、Melda 及 melda-sec 实现。

**📊 数据集**

未使用专门的数据集，采用原型实现进行实验。

**📈 对比分析**

论文未给出具体比较实验或性能评估。

**⚠️ 局限性**

仅为初步探索，原型实现尚未充分验证，需进一步评估在拜占庭环境下的鲁棒性。

---

## 167. Formalizing Scarf, Brouwer, and Nash in Lean

**arXiv ID:** 2607.05987 | [PDF](https://arxiv.org/pdf/2607.05987v1)

**作者:** Yuwei Lyu `[一作]` (Xiamen University Malaysia), Kai Li `[通讯]` (Xiamen University Malaysia)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Lean 4 中完成了从 Scarf 定理到 Brouwer 不动点定理再到有限游戏混合纳什均衡的完整形式化证明，并构建了相应的结构化评测基准 BrouwerBench。

**💡 创新点**

创新点在于把 Scarf 的离散组合核心（dominant set、rooms、doors、parity 论证）与网格逼近、嵌入-投影构造、Nash map 等连续与组合工具融合成可重用的证明流水线，并将该流水线的证明结构作为评测对象。

**🔧 技术方法**

采用 Lean 4 + Mathlib 进行类型安全的形式化，利用 Scarf 的 indexed-order 表达、网格化逼近、紧致性与连续性分析、嵌入-投影映射以及期望收益线性性等数学技术，进一步构建了自动化评测工具。

**📊 数据集**

使用 80 条从该 Lean formalization 提取的题目作为 BrouwerBench 基准，用于评估模型的证明结构理解能力。

**📈 对比分析**

通过手工评分 0–2 的 Rubric 对四个本地模型在 80 题上进行比较，最高得分 122/160（约 76.2%），显示不同模型在证明角色识别与依赖关系推理上的性能差异。

**⚠️ 局限性**

局限性：仅覆盖单一路径的 Scarf–Brouwer–Nash 证明，未包含其他不动点定理证明；BrouwerBench 规模有限、缺乏多样性；评分手工、缺乏统计可靠性；未覆盖更广泛的数学推理和更复杂的游戏理论场景。

---

## 168. Automated Derivation of Lattice Boltzmann Schemes for Systems of Conservation Laws

**arXiv ID:** 2607.05668 | [PDF](https://arxiv.org/pdf/2607.05668v1)

**作者:** Adrian Kummerländer `[一作]` (Karlsruhe Institute of Technology), Mathias J. Krause `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998` `847a60d8-a755-47af-ba5d-c5236b9e3083` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个PDE到LBM的编译器，能够自动生成完整的LBM求解方案，并在补充材料中给出了组件方程、制造解和数值配置。

**💡 创新点**

创新点在于将 PDE 求解过程自动化为 LBM 实现，提供可视化的组件方程和验证案例，降低手工编程难度。

**🔧 技术方法**

采用编译技术、符号计算与代码生成，结合 LBM 数值方法实现自动求解。

**📊 数据集**

使用制造解数据集以及补充材料中的数值配置进行验证。

**📈 对比分析**

通过与手工实现的 LBM 求解方案进行误差与运行时间对比，实验表明自动生成的方案在精度和效率上与手工实现相当或略优。

**⚠️ 局限性**

目前仅支持有限的 PDE 类型，生成代码在大规模并行计算时存在性能瓶颈，且需要人工调优参数。

---

## 169. Co-STAR: Cognitive Stimulation Therapy by an Autonomous Robot for Dementia -- A One-Week In-Home Study

**arXiv ID:** 2607.05709 | [PDF](https://arxiv.org/pdf/2607.05709v1)

**作者:** Emmanuel Akinrintoyo `[一作]` (Imperial College London), Nicole Salomons `[通讯]` (Imperial College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究开发并在家庭环境中部署了自主社交机器人Co-STAR，提供个性化认知刺激疗法（iCST），对9名认知障碍患者进行了一周的使用测试。

**💡 创新点**

创新点在于首次实现全自主、在家提供循证CST的社交机器人，并系统性分析了家庭成员参与、个性化与使用坚持的关系。

**🔧 技术方法**

所用技术包括Misty II机器人、平板视觉界面、本地语音识别WhisperD、OpenAI TTS模型以及自研的会话调度与个人化逻辑。

**📊 数据集**

数据集主要来自被试者个人背景信息及其在使用过程中的交互记录，没有使用公开的大规模认知数据集。

**📈 对比分析**

对照传统护理人员导向的iCST，研究发现机器人在家庭中的会话完成率约为每日一次的50%，超过了典型护理人员导向的执行率，表明自主机器人在遵从度上具有优势。

**⚠️ 局限性**

局限性包括样本量小、干预时长仅一周、技术稳定性问题以及缺乏长期认知效能评估。

---

## 170. Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference

**arXiv ID:** 2607.05475 | [PDF](https://arxiv.org/pdf/2607.05475v1)

**作者:** Guanyu Cai `[一作]` (Tsinghua University), Jiliang Wang `[通讯]` (Tsinghua University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对移动设备上大语言模型（LLM）推理进行了跨层级、跨框架、跨后端的系统性测评；

**💡 创新点**

创新点在于①首次系统评估NPU在LLM推理中的性能瓶颈；②开发了细粒度功耗分析工具PowerBench；③提出基于阶段的后端切换和调度最佳实践；

**🔧 技术方法**

使用了PowerBench进行后端功耗归因、统一的推理基准化工具、CPU/GPU/NPU三种后端、量化技术（w4、w4a16）等；

**📊 数据集**

评测了多种LLM模型（Llama 3.2‑1B/3B、Qwen 2.5‑1.5B/7B、Phi 3.5‑3.8B）以及MathQA、RoleBench、LongBench等数据集；

**📈 对比分析**

通过对五大主流框架（llama.cpp、MLC‑LLM、MLLM、MNN、GENIE）和三后端的400+配置测量，发现NPU在预填充阶段可达1400+ tokens/s，CPU在解码阶段优于GPU，且通过调度优化可使NPU能耗降低≈55%；

**⚠️ 局限性**

限制主要包括：对NPU静态图切换与KV缓存复用实现未深入探究；框架间差异主要基于实验平台，泛化性待验证；仅在Android Snapdragon SoC上测试，缺少其他平台的评估。

---

## 171. Ordering by Unanimity: Giving Applications Sequencing Rights Without Breaking Composability

**arXiv ID:** 2607.06144 | [PDF](https://arxiv.org/pdf/2607.06144v1)

**作者:** Andrea Canidio `[一作]` `[通讯]` (Category Labs), Andrea Canidio (Category Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种名为 Unanimity Override 的交易排序规则，允许区块链上不同应用表达对交易执行顺序的偏好，并在不破坏安全性和可组合性的前提下尽量遵守这些偏好。

**💡 创新点**

创新点在于：1) 将所有应用一致的排序视为最基本的约束，构建无环修复机制；2) 通过退化规则（fallback）对循环进行可预见的打破，保障单意见交易和“gate”区域交易的顺序；3) 证明在攻击者完全控制默认排序时，仍能保证这些交易顺序不被逆转，且该规则兼容任意满足帕累托原则的聚合方法。

**🔧 技术方法**

使用社交选择理论（Unanimity 关系、Paretian 规则）构建排序关系，利用拓扑排序（Kahn 算法）完成最终线性顺序；引入“demotion”与“fallback”机制以处理非可约环；并在理论上证明两类保障（单意见交易和 gated 区域）的安全性。

**📊 数据集**

本文为理论性工作，没有使用具体的数据集；所有结果均为形式化证明与模型分析。

**📈 对比分析**

方法通过对比在攻击者可完全控制默认排序与优先费排序两种基准下的可逆交易对数量，展示 Unanimity Override 在保护单意见交易和 gated 区域交易方面优于默认排序；在优先费排序下，安全性提升但对某些多意见交易仍有潜在攻击面，具体成本依赖于循环结构。

**⚠️ 局限性**

局限性包括：1) 对多意见交易的保护依赖于循环中是否能找到可降级的交易，攻击者仍可构造循环并诱导降级；2) 退化规则目前仅基于默认排序的优先级，未考虑更复杂的降级策略；3) 交互集的定义存在“声明式”与“实际式”两种选择，后者涉及固定点问题，尚未解决存在性、唯一性与可验证性；4) 在某些场景下，Unanimity Override 可能导致前跑或 griefing 攻击成本下降。

---

## 172. Binocular Gaze Estimation with Single Camera and Single Light Source

**arXiv ID:** 2607.05473 | [PDF](https://arxiv.org/pdf/2607.05473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 173. LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis

**arXiv ID:** 2607.06160 | [PDF](https://arxiv.org/pdf/2607.06160v1)

**作者:** Chenhao Yuan `[一作]` (University of Chinese Academy of Sciences), Kang Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 LongCrafter，一种面向长上下文监督微调的结构化数据合成框架，利用层级任务分类和证据图构造来生成指令-响应对。

**💡 创新点**

创新点在于①构建32种细粒度任务分类覆盖局部与全局两大维度，②先构造跨段证据图再生成指令，保证指令难度与结构化，③响应生成采用逐步引用证据的链式推理，确保回答可信且可追溯。

**🔧 技术方法**

主要技术包括任务分类引导的多阶段合成（长文本构造 → 证据图构建 → 指令与响应生成）、LLM在每阶段的指导与生成、证据图中的依赖边构造、链式引用与可追溯的响应格式、LoRA 微调与性能评估。

**📊 数据集**

使用多源网页语料生成 2000 条 LongCrafter 样本；在 LongBench、LongBench v2 与 LooGLE 三个长上下文基准上进行评估，且与 LongAlign、LongMagpie、LongFaith、LongReward 等现有 SFT 基线对比。

**📈 对比分析**

实验采用 Qwen2.5‑7B 与 LLaMA‑3.1‑8B 两个模型，使用 LoRA 微调 2 轮，最终 All‑Overall 分别达 45.15% 与 45.71%，比所有 SFT 基线高 5–6 分，尤其在高难度任务和“lost in the middle”位置鲁棒性测试中表现最佳。

**⚠️ 局限性**

局限性包括：多轮证据图构造成本高，需要进一步优化；覆盖的任务维度未包含长周期规划、持续对话等更复杂场景；目前仅通过 LLM 生成证据图，缺乏高效的推理或检索替代方案。

---

## 174. BaCon: Efficient Batch Processing of Counting Queries [Full Version]

**arXiv ID:** 2607.05832 | [PDF](https://arxiv.org/pdf/2607.05832v1)

**作者:** Yuxi Liu `[一作]`, Jun Yang `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于计数映射（CountMap）的批量计数查询执行框架（CUBE），能在不改动数据库内核的前提下，利用因式化数据库思想和工作负载感知域量化，实现高效共享计算；

**💡 创新点**

创新点包括：①将计数查询的因式化处理与域量化相结合，生成紧凑的计数映射；②设计⊗和⊕两种运算符以实现计数映射的组合与累加；③在客户端实现多层优化（批量查询、服务器/客户端游标切换、FK‑PK快捷路由等），保持对主流DBMS的兼容性；

**🔧 技术方法**

使用技术包括因式化（factorized）join、工作负载感知量化、计数映射、C语言UDF、PostgreSQL服务器/客户端游标、批量化查询、Numba JIT、树形执行计划与动态子树重排；

**📊 数据集**

实验数据集包括IMDB、STATS、DSB三大公开数据库，并基于JOB、SeConCDF、GRASP等构造的九个真实工作负载，以及合成的四个规模化测试集；

**📈 对比分析**

与两种基线（逐个查询独立执行和连接后再过滤）对比，CUBE在所有九个工作负载上均实现了2×至178×的加速，单线程时平均提升约40×，并行模式下可达598×，且始终不出现时间超限；

**⚠️ 局限性**

局限性在于仅支持无环join、单值等值/范围选择条件，对包含复杂谓词或多重join的工作负载表现不如基线；计划树的选择依赖启发式，未采用完整成本模型；对FK‑PK边的快捷优化目前仅局限于根节点；

---

## 175. Bibby AI: An Editor-Native Agentic Platform for Academic Research, Writing, and Publishing

**arXiv ID:** 2607.05435 | [PDF](https://arxiv.org/pdf/2607.05435v1)

**作者:** Nilesh Jain `[一作]` `[通讯]` (Bibby AI), Nilesh Jain (Bibby AI)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

建立了一个集成的“研究‑撰写‑发布”云 LaTeX 编辑平台，将文献检索、引用管理、写作、模板排版与投稿等步骤统一在一个工具链中完成。

**💡 创新点**

相较于传统浏览器扩展，它拥有完整的文档状态、编译管线和版本历史，能够以可验证的方式执行检索驱动的引用插入、结构编辑和模板格式化，而非仅提供文本建议。

**🔧 技术方法**

使用云 LaTeX 编辑器、PDF/DOCX/手写数学转换管线、检索层（含 USPTO PatentsView 与 Marx–Fuegi 引文信号）、基于文档抽象语法树的任务级代理（文献筛选、起草、修订、投稿格式化）等技术。

**📊 数据集**

利用 USPTO PatentsView 的专利‑论文引文数据、Marx–Fuegi 引文语料以及广泛的学术元数据来支持检索与引用。

**📈 对比分析**

通过时间节省评估框架将该平台与碎片化工具链基线进行对比，已在生产环境中部署，服务 5,000+ 活跃研究者，覆盖 50+ 学校订阅，显示显著的时间节约。

**⚠️ 局限性**

论文未详细说明限制，潜在问题可能包括手写数学转换的准确性、对非标准 LaTeX 模板的支持程度、非 LaTeX 语言兼容性以及代理智能化水平的局限。

---

## 176. Proof of Execution: Runtime Verification for Governed AI Agent Actions

**arXiv ID:** 2607.05397 | [PDF](https://arxiv.org/pdf/2607.05397v1)

**作者:** James Rhodes `[一作]` (AlphaBitCore, Inc.), George Kang `[通讯]` (AlphaBitCore, Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 Proof of Execution (PoE) 的运行时验证框架，能够在 AI 代理执行期间捕获授权、路径、效果、历史记录和可重放性，并生成可验证的执行证明 (EAC) 用于调度和合规审计。

**💡 创新点**

核心创新包括：① 将合约授权、执行轨迹、可重放上下文与签名日志统一成一个可验证的证明对象；② 设计五个可检查的运行时不变式与五项语义保证；③ 提出 Prime Execution Model 将规划、授权、效应与记录拆分为独立的权威平面；④ 在形式化语义基础上给出安全性定理，明晰加密与部署假设的边界。

**🔧 技术方法**

技术手段包括：签名方案 (EUF-CMA)、哈希与 Merkle 树做不可篡改日志、事件 DAG 与资源序列化、运行时不变式验证器、可重放上下文捕获、TypeScript 单节点原型实现与基准测试。

**📊 数据集**

实验使用自定义的合成工作负载：单能力流、5 节点流水线、50 任务并行批处理；未使用公开数据集，重点评估机制的时延、存储与攻击检测效果。

**📈 对比分析**

通过与无 PoE 基线（直接工具调用）对比，测得单执行平均额外延迟约 2.7 ms，批处理时占比约 4.4%；压缩后的事件流平均 1.1 KB；在 10,000 次注入 T2/T4 攻击实验中，验证器全部拒绝，证明机制有效。

**⚠️ 局限性**

局限性：① 仅保证执行在合约范围内，不判断合约本身是否合规；② 重放仅在捕获输入完整且解释器确定性时成立，外部状态漂移不被捕获；③ 目前仅支持单逻辑网关，阈值网关与多合约交互尚未覆盖；④ 方案不防止规划器被破坏，只限定其在已授权的范围内操作；⑤ 原型仅单节点实现，缺乏多节点部署与大规模真实工作负载评估。

---

## 177. How Personas Can Influence Agents to Play Split or Steal

**arXiv ID:** 2607.05398 | [PDF](https://arxiv.org/pdf/2607.05398v1)

**作者:** Carlos Leon `[一作]` (Universidade Lusófona), Thomas D. Parsons `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨将人格提示嵌入大型语言模型系统提示后，在与固定提示的虚拟人进行的迭代“拆分或偷窃”社会困境游戏中对合作与背叛行为的影响。

**💡 创新点**

创新点在于将Big Five人格分组的人格提示与多模型、多温度、欧式葡萄牙语环境相结合，系统评估了人格、模型和温度对策略、合作率、主题与情绪的多维度影响，并为后续人机VR实验提供基线。

**🔧 技术方法**

使用了四个开源大型语言模型（ministral 3:3b、phi4:14b、gemma3:12b、gemma4:e4b）与GPT‑4.1‑mini虚拟人，利用系统提示实现对话与决策，采用LLM情绪与主题分类器进行文本标签，策略识别则基于经典重复博弈策略（Tit‑for‑Tat、Pavlov等）。

**📊 数据集**

数据集为160场游戏会话（每场15轮，总2400轮），包含20个人格提示（按Big Five划分）、游戏回合对话、决策与支付记录，情绪与主题标签由本地LLM自动生成，公开托管于Zenodo。

**📈 对比分析**

通过比较合作率、开关率和策略分类指标，发现phi4和ministral在两种温度下均保持高合作率（约74%），gemma模型表现更为多样；在人格组中，亲善与原则组的互相拆分率最高，而分析与自利组的剥削率最高；这些结果表明人格提示能显著影响行为，但受模型固有偏好限制。

**⚠️ 局限性**

主要局限包括模型本身的合作偏好掩盖了人格提示的效应；虚拟人无固定决策策略导致合作基线偏高；人格组样本分布不均；实验仅在欧式葡萄牙语环境进行，需在其他语言下复现；此外，决策与对话的先决写法可能引入偏差。

---

## 178. Realistic Compound-Lens Defocus Blur Synthesis

**arXiv ID:** 2607.05837 | [PDF](https://arxiv.org/pdf/2607.05837v1)

**作者:** Yunkyu Lee `[一作]` (POSTECH), Sunghyun Cho `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种用于合成真实复合镜头失焦模糊数据集的完整流水线，并基于此生成了CLDefocus数据集

**💡 创新点**

创新点在于结合Debye CZT波光学PSF计算、基于深度分层的遮挡感知渲染以及在线性空间内模拟ISP流程，实现了大规模、光学多样且光度真实的失焦数据合成

**🔧 技术方法**

使用Debye CZT加速的波光学PSF求解、Zernike多项式重构、深度量化层化渲染、ISP仿真（去噪、饱和、非线性响应）以及多尺度卷积合成

**📊 数据集**

新建CLDefocus（4万训练对/1千验证/1千测试，384×384分辨率）并对比DPDD、SYNDOF、RealDOF、RTF等真实/合成数据集

**📈 对比分析**

通过在多个解卷积网络（NRKNet、Restormer、INIKNet、NAFNet）上训练评估，CLDefocus训练模型在真实测试集（RealDOF、RTF、DPDD）和手机相机数据上均表现出更高的PSNR/SSIM、低LPIPS/NIQE以及更优的无参考感知指标，且跨设备泛化更强

**⚠️ 局限性**

局限性包括Debye CZT在极端离轴场景下的近似误差、对复杂光学设计的求解不稳定、深度估计误差可能导致合成失真，以及ISP模型仅近似真实相机管线，未覆盖所有光学/感光器差异

---

## 179. Breadth-First Search in Succinct Planar Graphs

**arXiv ID:** 2607.06221 | [PDF](https://arxiv.org/pdf/2607.06221v1)

**作者:** Johannes Meintrup `[一作]` `[通讯]`, Johannes Meintrup

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文内容不足，无法确定具体研究内容

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

缺乏足够信息导致无法进行完整评估

---

## 180. The Masks We (Think We) Wear: Privacy Threats of Browser-Extension Wallets in the Web3 Ecosystem

**arXiv ID:** 2607.06141 | [PDF](https://arxiv.org/pdf/2607.06141v1)

**作者:** Weihong Wang `[一作]` (KU Leuven), Tom Van Cutsem `[通讯]` (KU Leuven)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对浏览器扩展钱包在网络层和网页层的隐私威胁进行系统测量与分析，发现并量化了5类隐私风险，并提出可行的缓解方案。

**💡 创新点**

创新点在于将网络侧地址关联、网页侧权限泄露、跨域注入等多维风险统一框架化，并通过改进的请求截取框架首次捕捉扩展后台流量，显著提升了对隐私泄露的检测深度。

**🔧 技术方法**

使用了 Puppeteer/Playwright 的网络请求截取与网页暴露检测技术，结合 RPC 解析、EIP‑1193/EIP‑6963 发现事件，以及权限重置和回滚测试。

**📊 数据集**

主要数据集包括85个 Chrome Web Store 主流钱包（覆盖约3.5160万用户）、30个热门 Ethereum dApp 以及一份常用节点/分析域列表。

**📈 对比分析**

通过与 Torres 等人原始方法对比，改进框架可捕获扩展后台流量导致的地址泄露率提升约30%，并对不同钱包版本做时间序列对比，验证漏洞持续存在且被忽视的情况。

**⚠️ 局限性**

局限性在于仅针对 Chrome 扩展，未覆盖 Firefox/移动钱包，实验依赖手动或脚本交互，且无法覆盖所有用户真实行为场景。

---

## 181. CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration

**arXiv ID:** 2607.05465 | [PDF](https://arxiv.org/pdf/2607.05465v1)

**作者:** Hairui Zhu `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Wenhao Jiang `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了大规模多模态工具使用数据集 CanvasCraft，并训练 CanvasAgent，实现多工具多步图像创作与编辑的端到端可执行流程。

**💡 创新点**

①提出 CanvasCraft 数据集，提供 140K 完整可执行轨迹和 10K RL 任务；②设计两阶段 SFT + GRPO 训练框架和混合奖励（对齐、美学、轨迹与规则信号）；③CanvasAgent 能够感知视觉状态、管理多图资产并自适应调用 11 种视觉工具，完成复杂的图像创作与编辑。

**🔧 技术方法**

使用 Qwen3-VL-8B 作为基础大语言模型；11 种视觉工具（生成、编辑、定位、分割、提取、合成、裁剪、OCR、旋转、翻转、超分）；SFT + GRPO 强化学习；LLM-judge（Qwen3.5-Plus）评估对齐与美学；规则奖励衡量格式、可执行性与效率。

**📊 数据集**

CanvasCraft 数据集：CanvasCraft‑SFT（140K 轨迹）和 CanvasCraft‑RL（10K 任务），以及 250 样本评估基准。

**📈 对比分析**

与 LLaVA‑OneVision‑7B、Qwen3‑VL‑8B‑Instruct、Qwen3‑VL‑32B‑Instruct 以及专用图像生成模型 Qwen‑Image‑2.0、Wan2.7‑Image、GPT‑Image‑2 进行对比。CanvasAgent(SFT+RL) 的整体奖励 0.821、对齐 0.869、审美 0.762、轨迹得分 0.849、规则得分 0.785，明显优于对照模型；SFT 阶段已提升到 0.557，RL 阶段进一步提升。

**⚠️ 局限性**

固定工具集（11 种视觉工具），需要在真实环境中执行工具，依赖外部 LLM 评判；训练耗时较长；未覆盖动态工具发现、自动评估或视频编辑等更复杂场景。

---

## 182. Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning

**arXiv ID:** 2607.05458 | [PDF](https://arxiv.org/pdf/2607.05458v1)

**作者:** Haiwen Yi `[一作]` (University of Toronto), Xinyuan Song `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对冻结LLM代理的可学习执行框架，利用离线优势加权回归训练轻量级控制器，使其在保持LLM参数不变的情况下改进执行流程，尤其是提交前的验证行为。

**💡 创新点**

创新点在于将执行层正式建模为 Harness MDP，区分终端质量与过程成熟度，并证明仅凭离线轨迹即可学习高效的控制策略。

**🔧 技术方法**

使用的技术包括优势加权回归 (AW)、离线强化学习、结构化动作空间、单隐藏层 MLP 控制器，以及对过程成熟度的自定义诊断分数。

**📊 数据集**

实验数据集涵盖六个受控领域（知识工作、编码、研究问答、多工具、长记忆、规划）以及两个公共基准适配器（τ‑bench retail 与 AgentBench DB‑Bench）。

**📈 对比分析**

与行为克隆（BC）和强制检查（Forced CHECK）进行对比，AW 在所有设置中提升了验证行为，且在编码与两个基准适配器中实现了显著的终端质量提升，性能通过 G（任务评分）和 Harness Maturity Score 评估。

**⚠️ 局限性**

局限性包括对离线缓冲区支持的依赖，过程改进不一定转化为终端质量提升，且方法仅适用于冻结LLM，缺乏在线自适应或主动数据采集能力。

---

## 183. The mmatrix toolbox: componentwise accurate algorithms for M-matrices with triplet representation

**arXiv ID:** 2607.05437 | [PDF](https://arxiv.org/pdf/2607.05437v1)

**作者:** Bruno Iannazzo `[一作]` (University of Perugia), Federico Poloni `[通讯]`

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于M矩阵triplet表示的准确计算工具箱，提供LU分解、矩阵求逆、最小特征值、奇异值分解、矩阵平方根等算法。

**💡 创新点**

创新点在于利用triplet表示与GTH算法，克服浮点相消误差，实现全组件级别的准确性，并通过块化递归实现提升计算效率。

**🔧 技术方法**

主要技术包括triplet表示法、GTH分解、Fortran核心实现以及MATLAB接口、块化与递归算法。

**📊 数据集**

使用的数据集主要为随机生成的M矩阵、有限元离散矩阵以及Markov链等典型M矩阵。

**📈 对比分析**

与MATLAB内置函数对比，工具箱在保持机器精度的同时，性能与MATLAB相当甚至更优，尤其在大尺寸或高条件数时表现突出。

**⚠️ 局限性**

局限性在于必须提供triplet表示，且仅适用于满足条件的M矩阵，非M矩阵无法直接使用。

---

## 184. Efficient and Robust Lock-Free Multi-Word Compare-and-Swap via Contention-Aware Helping

**arXiv ID:** 2607.06034 | [PDF](https://arxiv.org/pdf/2607.06034v1)

**作者:** Motoki Unno `[一作]` (Nagoya University), Yoshiharu Ishikawa `[通讯]` (Nagoya University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种新的无锁多词原子交换（MCAS）算法，并在其基础上实现争用感知的帮助机制与版本嵌入技术。

**💡 创新点**

创新点在于：1）引入基于入口计数器和指数退避的争用感知帮助机制，显著减少缓存失效和帮助冲突；2）在每个目标单词中嵌入版本号以抑制 ABA 问题，从而避免了先前 MCAS 算法中可能出现的重复执行；3）通过这些改进在高争用环境下实现与死锁自由 MCAS 相当的吞吐量，同时保持无锁保证。

**🔧 技术方法**

采用 C++ 实现，使用单词级 CAS、指数退避、入口计数器、版本嵌入以及基于 epoch 的内存回收；实验环境为 2 芯片 Intel Xeon Gold 6258R，配 192 GB DDR4 内存，Ubuntu 22.04，GCC 11.4。

**📊 数据集**

使用合成数据集：1,000,000 个 64‑bit 目标单词，初始值全为 0；MCAS 操作随机选取 2、4、8 个单词并原子递增，选择概率按 Zipf 分布（α=0 或 1）控制争用程度。

**📈 对比分析**

将提出的算法与 CASN、AOPT、死锁自由 MCAS（DLF）进行基准比较。通过 10 秒的持续 MCAS 执行测量吞吐量和 99th 分位延迟。结果显示：在低争用下吞吐量约为 DLF 的 2 倍，在高争用下吞吐量可达 DLF 的 3 倍；延迟显著低于 CASN/AOPT，接近 DLF。

**⚠️ 局限性**

局限性包括：1）版本嵌入无法完全消除 ABA，仍存在计数器溢出的极端情况；2）指数退避在过度抢占（超线程）场景下可能导致帮助延迟，降低吞吐量；3）实现依赖 64‑bit 单词和特定硬件支持，迁移到其他体系结构需要进一步验证。

---

## 185. 6G Sensing Security: Distributed Game-Theoretic RL for Urban Beamforming and Attacker Detection

**arXiv ID:** 2607.06115 | [PDF](https://arxiv.org/pdf/2607.06115v1)

**作者:** Parmida Geranmayeh `[一作]` (Technische Universitaet Dortmund), Onur Günlü `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种结合强化学习、贝叶斯推断与Stackelberg博弈的分布式框架，用于6G ISAC系统中检测主动攻击者并优化波束成形。

**💡 创新点**

首次将贝叶斯推断与Stackelberg博弈嵌入RL决策过程，并通过经验回放提升检测准确性与吞吐率。

**🔧 技术方法**

贝叶斯概率更新、Stackelberg博弈模型、Q‑learning、经验回放、基于Ray‑tracing的城市微波传播仿真。

**📊 数据集**

利用德国多特蒙德市地图与OpenStreetMap进行射线追踪，仿真基于3GPP TR 38.901 UMi场景的28 GHz毫米波环境。

**📈 对比分析**

对比Nash与Stackelberg两种博弈框架，并比较有无经验回放的Q‑learning，Stackelberg+经验回放实现69.41%检测准确率、75.46%检测准确率、吞吐率提升至约934 Mbps。

**⚠️ 局限性**

局限于静态城市环境与静态攻击者，未考虑移动用户与实时动态场景，且计算复杂度随天线数与博弈规模急剧上升。

---

## 186. A robust and versatile parallel FFT-based mechanical solver for general non-periodic and periodic boundary conditions

**arXiv ID:** 2607.05929 | [PDF](https://arxiv.org/pdf/2607.05929v1)

**作者:** Yaovi Armand Amouzou-adoun `[一作]` (Université Paris-Saclay), Yushan Wang `[通讯]` (Université Paris-Saclay)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文开发了一套通用的 FFT‑基求解器，能够在并行环境下求解线性与非线性机械问题，并支持周期、Dirichlet、Neumann 及其任意组合的边界条件，兼容小变形与大变形框架。

**💡 创新点**

核心创新点包括：① 将一般边界条件与波动位移场的对称/周期扩展相联系，利用离散三角变换（DTT）在不增加额外计算成本的前提下将非周期问题映射为周期问题；② 采用基于位移的加速固定点迭代，并在极化应力场上实现 Anderson 加速（相较传统在位移场加速，收敛更快）；③ 结合双四面体（TETRA2）与传统六面体（HEX1）有限差分方案，推导出针对不同预条件器（C0、2μ0、B0）的离散格林算子，实现对任意边界条件的普适适用；④ 在 AMITEX* 求解器中扩展 2DECOMP&FFT 库以支持 DTT 与混合 DTT/DFT，并实现 2D 梯形子域划分，显著提升可扩展性。

**🔧 技术方法**

技术实现包括：
- 位移基固定点迭代 + Anderson 加速；
- 离散三角变换（DCT/DST）与 DFT 的互通；
- HEX1 与 TETRA2 有限差分求导；
- 三种预条件器（C0、2μ0、B0）构造格林算子；
- 2DECOMP&FFT 并行 2D 梯形子域分解；
- 通过 2DECOMP&FFT 提供的内存池减少内存占用。

**📊 数据集**

使用的“数据集”为合成的三维单元格：
- 简谐钢板/铝板弯曲梁（均匀弹性/有限变形）；
- 含空洞的多孔材料（弹塑性、晶体塑性）；
- 单晶 L‑beam（FCC 铜）进行扭转‑弯曲加载。所有案例均采用网格分辨率从 32³ 到 512³ 或 150×50×175。

**📈 对比分析**

比较方法：
- 与解析弯曲梁解（小变形）对比；
- 与传统周期性 FFT（Moulinec–Suquet）对比；
- 对不同边界条件、有限差分方案与预条件器的迭代次数和收敛速率进行对照；
- 并行性能评估：在 128–768 CPU 核心下测试强缩放，效率保持 90% 以上至 640 核。整体性能：TETRA2 在非周期、非线性、高对比度问题上收敛更快、可达更大变形；而 HEX1 在某些极端加载下出现振荡、收敛失败。

**⚠️ 局限性**

局限性：
- 对高阶梯度塑性或非本构模型的支持有限；
- 需要针对每种 BC 组合维护对称/周期扩展规则，代码复杂度高；
- 内存占用因 4 倍虚拟域和多点/体素导致显著增长；
- 预条件器参数（λ0, μ0）仍需经验调优，未给出统一最优策略；
- 目前 2DECOMP&FFT 采用固定尺寸 2D 梯形子域，缺乏动态负载平衡，导致极端非均匀塑变时效率下降。

---

## 187. Foundation Models for Automatic CAD Generation

**arXiv ID:** 2607.05573 | [PDF](https://arxiv.org/pdf/2607.05573v1)

**作者:** J de Curtò `[一作]`, I. de Zarzà `[通讯]` (LUXEMBOURG Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并评估了一个统一的文本到CAD生成框架（LMMOFGEN），并在两种批判模式（分析视觉评估和VLM语义评估）下对七大基础模型进行系统实验。

**💡 创新点**

①首次将结构化JSON脚本、解析式几何引擎与多轮迭代反馈结合；②提出了四轴（校验、网格、特征、视觉）与五轴（加VLM语义）评估体系；③展示了VLM语义批判能显著提升网格完整率并揭示模型间语义差异。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑V3.2、Qwen3‑235B‑A22B、Llama‑3.3‑70B、Gemma‑3‑27B、GLM‑4.5、MiniMax‑M2.1、INTELLECT）作为文本生成器；VLM批判采用 Qwen2.5‑VL‑72B；几何构建采用 Trimesh/​Shapely；可视化与评估基于 Phong 光照与自定义视觉指标。

**📊 数据集**

创建了97条工程设计问题的数据集，涵盖四类标准几何（矩形板、盒子、圆柱、L‑形支架），每条问题均附有真实的特征规范作为基准。

**📈 对比分析**

在两种批判模式下对模型进行四轮迭代，评估四个维度（校验、网格、特征、视觉）并在 VLM 模式下额外加入语义维度；结果显示：在分析视觉模式下，四大模型总体得分≈0.887、网格成功率≈98.97%；在 VLM 模式下得分下降≈0.04，但 Gemma‑3‑27B 实现 100% watertight 网格。多轮迭代对强模型提升约 5–6% 语义得分，对弱模型提升可达 20%。

**⚠️ 局限性**

限制包括：仅覆盖四类标准几何，未包含多体装配或自由曲面；VLM 评估引入随机性；基于渲染图像的批判无法完全捕捉仿真或工艺约束；实验受限于单一推理后端与固定温度设置。

---

## 188. Prompting Beats Fine-Tuning: Generative Expected Value Scoring for Statutory Term Retrieval

**arXiv ID:** 2607.05582 | [PDF](https://arxiv.org/pdf/2607.05582v1)

**作者:** Alvin Wang `[一作]` (Carnegie Mellon University), Jaromir Savelka `[通讯]` (Carnegie Mellon University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在法律文本中为特定法条术语对检索到的句子进行可解释性排序，并通过对比编码器微调和解码器零样本提示两类方法，提出了更高效、更精准的排名方案。

**💡 创新点**

创新点包括：①将对比模型扩展至开源大型LLM，①采用概率期望值评分的零样本提示方案，②系统性分析上下文扩展对排名质量的影响并发现其无效。

**🔧 技术方法**

使用技术：现代化的编码器模型 ModernBERT 的多种微调方案（snt、qry2snt、sp2snt 等）；多种解码器模型（GPT‑4o、GPT‑5.2/5.4、GPT‑OSS‑120B、LLaMA‑3.3‑70B、LLaMA‑4‑Scout‑17B、Qwen‑3‑32B）以及基于期望值评分的零样本提示方法。

**📊 数据集**

使用数据集：由 42 条美国法典概念组成的 26,959 句子集合，每句句子被标注为四级解释价值（High、Certain、Potential、No value），Krippendorff α 为 0.79。

**📈 对比分析**

比较方法：在 NDCG@10/100 评价指标下，对四类查询子集（small/large × sparse/dense）进行横向对比。ModernBERT 与以往 BERT 结果基本相当；解码器提示在所有子集尤其是大稀疏查询中取得最高分，GPT‑5.4 达到 0.82/0.87 的 NDCG@10/100，超越此前报道的 SOTA。

**⚠️ 局限性**

限制：扩大上下文（段落或全文）并未提升性能，且对特殊控制符的依赖不稳定；实验仅在美国法典与案例法律上验证，跨司法区、非英语域的迁移性未知；尽管解码器提示速度快，但大规模推理仍需成本与格式化错误处理。

---

## 189. Self-Heating and Radiation Hardness Studies of 3nm GAA-FET-Based SRAM with Different Substrate Isolation Techniques

**arXiv ID:** 2607.05789 | [PDF](https://arxiv.org/pdf/2607.05789v1)

**作者:** Albert Lu `[一作]` (San Jose State University), Hiu Yung Wong `[通讯]` (San Jose State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过 3D TCAD 仿真评估 3nm GAA‑FET SRAM 在不同底层绝缘技术（SD‑BDI、C‑BDI、PTS）下的自热与辐射硬度。

**💡 创新点**

提出 C‑BDI 结构，既保持 S/D 与基底的连接，抑制应力松弛，又保持低泄漏；并证明在真实有效基底厚度下 BDI 对自热影响可忽略。

**🔧 技术方法**

使用 3D TCAD Sentaurus（SDevice、SProcess）结合统一迁移、Ballistic、Lombardi、SRH/ Auger 等模型进行电流、电压、温度及 LET 触发 SEU 的仿真。

**📊 数据集**

无真实数据集，全部基于文献给出的 3nm 芯片尺寸参数（如晶格厚度、掺杂、氧化层等）进行建模。

**📈 对比分析**

通过比较自热峰值、离线泄漏电流、SNM、阈值 LET 等指标；结果显示无 BDI 与全 BDI 的自热差异不大；C‑BDI 与 SD‑BDI 在无 PTS 时泄漏仅为 SD‑BDI 的 3–5 倍，并且对 α 粒子 LET 的阈值远高于 0.0144 pC/um，表现出良好辐射硬度。

**⚠️ 局限性**

仅基于仿真，缺乏实验验证；模型假设的基底厚度和工艺参数可能与实际不符；未考虑长期可靠性（BTI、HCI、TDDB）以及工艺变异对性能的影响。

---

## 190. Women Enter Too, but Men Persist:The Temporal Structure of Gender Inequality in the Global Citation Elite

**arXiv ID:** 2607.05427 | [PDF](https://arxiv.org/pdf/2607.05427v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 191. Exogenous Dropout: A Simple, Strong Baseline for Corruption-Robust Time Series Forecasting with Covariates

**arXiv ID:** 2607.05452 | [PDF](https://arxiv.org/pdf/2607.05452v1)

**作者:** Hao Hu `[一作]` (Wuhan University), Xue-shan Ai `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估并提出一种单行训练增强——exogenous dropout——显著提升时间序列预测模型对受损外源变量的鲁棒性

**💡 创新点**

证明单行代码的全通道dropout能与甚至超过专门设计的边界机制，表明架构边界并非鲁棒性必需

**🔧 技术方法**

使用随机全通道屏蔽（exogenous dropout）与FiLM加门限模型BoundEx进行对比，并应用PatchTST、DAG等主流架构

**📊 数据集**

在电价（Nord Pool、PJM、BE、FR、DE）、水文（Rapel）和气象（Jena Weather）三大数据集上进行实验

**📈 对比分析**

采用vs‑floor百分比对四种损坏模式（Gaussian噪声、时序错位、缺失通道）进行比较；dropout提升所有模型鲁棒性，DAG+dropout成为最佳，MSE从0.255下降至约0.259，Gaussian下降至+3% 等

**⚠️ 局限性**

局限性包括：仅覆盖能源/物理相关领域；边界保证为比例式而非绝对；未评估对更复杂拓扑或不同噪声类型的鲁棒性；对未测试架构的泛化仍未知

---

## 192. Code-Level Cost Function Generation for Spatial Image Steganography Using RAG-Enhanced Large Language Models

**arXiv ID:** 2607.05868 | [PDF](https://arxiv.org/pdf/2607.05868v1)

**作者:** Yige Wang `[一作]` (Shanghai University), Hanzhou Wu `[通讯]` (Shanghai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于知识驱动的进化框架，通过检索增强生成（SE‑RAG）实现空间图像隐写术的代码级成本函数自动生成。

**💡 创新点**

创新点在于：①使用 CSS 将代码语义映射为隐写术概念，构建双路径检索；②结合静态论文库与动态经验库，实现持续的知识演化；③在进化过程中加入语义审计与反馈机制，将成功经验转化为可复用的规则。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑4o）+ 检索增强生成（RAG）、Bi‑Encoder + Cross‑Encoder检索、语义审计与知识库更新、进化算法与快速预评估、代码执行率优化。

**📊 数据集**

使用的公开数据集：BOSSBase v1.01（10,000张256×256灰度图）以及泛化测试集 BOWS2。

**📈 对比分析**

通过与传统手工算法（HILL、WOW、S‑UNIWARD）以及现有 LLM 自动设计方法进行对比，评估指标为检测误差率 P_E。结果显示，所提框架在两套数据集上均取得最高 P_E；执行率提升 46.3%，搜索成本下降 26.1%。

**⚠️ 局限性**

局限性包括：仅针对空间图像隐写，难以直接推广至视频或音频；对 LLM 的依赖导致模型更新成本高；知识库维护和检索效率仍受限；实验仅覆盖有限的隐写分析器，未检验在更广泛攻击场景下的鲁棒性。

---

## 193. Full-range Binary Classifier Calibration for Stable Model Updates in Production

**arXiv ID:** 2607.05481 | [PDF](https://arxiv.org/pdf/2607.05481v1)

**作者:** Konstantin Berlin `[一作]` `[通讯]` (Cisco), Konstantin Berlin (Cisco)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在对抗性检测模型中，用无恶意样本的正则化方法将原始预测分数校准为全曲线的误报率（FPR）并保持跨版本的阈值一致性。

**💡 创新点**

创新点在于：① 全曲线 FPR 映射采用无参数单调线性样条；② 固定对数尺度的输出契约（如 0.5 对应 0.1% FPR），使阈值在不同版本间保持相同含义；③ 通过节点下采样将可部署模型压缩至 200 KB 以内；④ 在样本不足时使用线性下限外推并安全裁剪。

**🔧 技术方法**

技术主要包括：sklearn 的 IsotonicRegression 与 PiecewiseLinearSpline；Filliben 绘图位置校正；对数尺度 FPR‑阈值对齐；两阶段样条拟合（从 FPR 到重映射分数，再到最终校准分数）。

**📊 数据集**

实验使用欧洲信用卡欺诈数据集（284,807 条交易，492 条欺诈），在 30% 的训练集上训练逻辑回归检测器，并在 30% 的保留集作为校准训练集，剩余 70% 用于评估。

**📈 对比分析**

与基线（未校准或概率校准）相比，在 held‑out 子集上，校准模型在 10% 到 0.1% FPR 范围内的相对误差不超过 2.3%，在 0.01% FPR 时为 7.2%，且模型体积保持 <200 KB。

**⚠️ 局限性**

局限性包括：1) 分数粒度导致同分数样本无法进一步分割（导致误报率上限）；2) 当目标 FPR 低于样本最低尾部时需线性外推，精度受限；3) 对正常流量分布漂移敏感；4) 样本量不足时低 FPR 误差受限于有限样本波动。

---

## 194. Delay Violation Probability Modeling for 5G Systems with HARQ Operation

**arXiv ID:** 2607.06169 | [PDF](https://arxiv.org/pdf/2607.06169v1)

**作者:** Sangwon Seo `[一作]` (KTH Royal Institute of Technology), James Gross `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了5G系统中HARQ机制的延迟违约概率（DVP）建模，考虑队列、传输、解码、反馈和周期性控制信号（CS）占用等实际时序，给出可计算的上界。

**💡 创新点**

创新点在于把多并行HARQ过程、slot‑based周期性CS占用、解码/反馈延迟等真实组件一起建模，并通过马尔可夫链求稳态队列分布与等待延迟分布，再结合服务延迟的RTT上界，得到整体DVP上界。

**🔧 技术方法**

使用离散时间马尔可夫链、队列理论、Markov分析以及3GPP时序与HARQ-IR误码模型，结合有效容量/有效带宽概念进行分析。

**📊 数据集**

采用5G-LENA仿真平台的链路级PER结果和上层仿真得到的包级数据，用于校准PER向量p_m。

**📈 对比分析**

与ns-3 5G-LENA仿真结果对比，所提模型能紧贴仿真数据，提供保守的上界；相较于Max‑Throughput和Fixed‑Tx‑Rate基线，模型在严格延迟和CS占用显著时更准确。

**⚠️ 局限性**

限制：模型仍非闭式，需要求稳态分布和递归等待概率，计算复杂度随队列长度和HARQ次数增长；同时未考虑编码延迟、MCS动态调度等更细粒度细节。

---

## 195. ThorArena: Benchmarking Humanoid Physical Interaction with Human Motion-Force Demonstrations

**arXiv ID:** 2607.06052 | [PDF](https://arxiv.org/pdf/2607.06052v1)

**作者:** Chenhao Yu `[一作]` (Beijing Academy of Artificial Intelligence), Shaqi Luo `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了ThorArena，一个基准框架，用同步收集的人类运动与交互力数据来评估仿真人形机器人的全身控制策略。

**💡 创新点**

创新点在于首次构建同步运动-力演示数据集，定义包含追踪误差、鲁棒性、能耗与存活率的Force‑Aware Tracking Score（FATS）指标，并搭建统一的力回放协议与策略适配器，使得力感知能量被正式纳入评价体系。

**🔧 技术方法**

主要技术包括VR/运动捕捉结合3D打印力感应钩实现的真实力测量、物理引擎仿真中对录制力的实时回放以及多策略适配器对不同控制网络的统一接口。

**📊 数据集**

使用了六个代表性物理交互任务（清桌、降水、提水、拉椅、推椅、协同搬运）共360条演示序列，记录了全身关键点姿态与双手三轴力。

**📈 对比分析**

通过对四种全身控制策略（Thor2、TWIST2、GMT、SONIC）在有力与无力两种环境下计算FATS、存活率、关键点误差及能耗等指标进行比较，结果表明在有力情境下Thor2始终保持最高FATS（≈81.7）和近乎100%存活率，其余策略在强交互时性能显著下降。

**⚠️ 局限性**

局限性包括：仅在仿真环境下回放力，缺少对不同机器人外形与多物体交互的扩展；力测量受限于实验设备的精度与同步误差；未在真实机器人上验证回放力与实际交互的对应性。

---

## 196. Abductive Corroboration of Probabilistic AI Models for Forensic Synthetic Media Detection

**arXiv ID:** 2607.05434 | [PDF](https://arxiv.org/pdf/2607.05434v1)

**作者:** Junade Ali `[一作]` `[通讯]` (Alan Turing Institute), Junade Ali (Alan Turing Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估多模型对合成媒体检测的互证效果，并实证分析 OpenAI SynthID 水印的早期部署

**💡 创新点**

首次证明跨模型互证可显著降低误报率，并首次记录 OpenAI 在 GPT-Image-2 推出前已开始使用 SynthID 水印

**🔧 技术方法**

使用 SigLIP2、DINOv2、Hive Moderation、SynthID 等概率检测器，结合 ϕ 相关系数、阈值敏感性分析和误报/真报比指标

**📊 数据集**

包含 2000 张 GPT‑Image‑2 生成图像、2000 张 Flickr 人类生成图像，以及 400 张用于 Hive 与 SynthID 验证的子集

**📈 对比分析**

对单一检测器、两模型互证、三模型互证等规则进行对比；单模型误报率从 20.7% 降至 1.7%（两模型）再到 0%（三模型），准确率最高达 94.8%

**⚠️ 局限性**

受限于样本规模、模型在域外的校准不足，以及训练数据重叠可能导致的相似性，未来需扩大数据集并引入更多结构多样化模型

---

## 197. Catalyst Papers in Artificial Intelligence Research: A Landscape on ICLR from 2017 to 2025

**arXiv ID:** 2607.05401 | [PDF](https://arxiv.org/pdf/2607.05401v1)

**作者:** Fan Huang `[一作]` `[通讯]` (Indiana University Bloomington), Fan Huang (Indiana University Bloomington)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2017–2025 年 ICLR 提交的 36,113 篇论文进行系统分析，构建了五类催化论文的操作性分类法，比较了四种破坏性指标（CD、node2vec、EDM、LLM 评分），并将其与 OpenReview 的审稿分数关联，进一步探究了催化论文如何影响研究方向、主题演化以及审稿认知的偏差。

**💡 创新点**

① 设计了完整的五类催化论文分类体系；② 在单一 ML 会议语料库上首次对多种破坏性度量进行头对头比较；③ 发现 EDM 与结构性认可（ERS）最契合，LLM 评分与语义评估（LAS）最契合；④ 揭示审稿分数与未来破坏性基本无关，但在不同催化类型和主题上存在系统性误校准。

**🔧 技术方法**

使用方向性引文嵌入（EDM）+ 随机游走、node2vec 词向量、CD 指数计算；采用 OpenAI 文本嵌入 + UMAP + HDBSCAN 进行主题建模；采用 Semantic Scholar Graph API 提取 ICLR 内部引文网络；利用 Spearman、Firth 逻辑回归、倾向匹配等统计方法；使用 GPT‑4o‑mini 进行零样本 LLM 评分。

**📊 数据集**

ICLR 2017–2025 的提交元数据（标题、摘要、作者、审稿分数、接受/拒绝决定）来自 OpenReview；Semantic Scholar 引文数据提供 ICLR 内部引文边；外部认可集（ERS）由 ICLR 内部引用前 2% 论文构成；LLM 评估集（LAS）为 50 篇论文的两次 LLM 标注；以及拒稿论文在 arXiv 等平台的再现数据。

**📈 对比分析**

对四种破坏性指标在 ERS 和 LAS 上进行 ROC‑AUC 与 Firth 逻辑回归的对比。EDM 在 ERS 上取得 AUC 0.827，CD 为 0.596，node2vec 为 0.493，LLM 评分在 ERS 上仅 0.424；在 LAS 上 LLM 评分最高 AUC 0.749，其余指标低于 0.5。EDM 在引用速度上的 Spearman 相关系数为 0.300，优于 CD（0.127）和 node2vec（-0.293）。机制分析显示主题发起者与桥接者分别导致主题占比增长 7.55 倍与 11.52 倍。

**⚠️ 局限性**

① 引文网络仅包含 ICLR 内部边，缺失外部引用；② 对最近两年数据的 EDM 计算受采样稀疏限制；③ LAS 样本量小，交叉验证一致性低；④ EDM 对随机游走超参数敏感，需多次验证；⑤ LLM 评分仅基于标题与摘要，未考虑完整论文文本；⑥ 研究仅聚焦 ICLR，结果可能不具备跨会议泛化性；⑦ 审稿误校准结果受主题建模方法与阈值设定影响。

---

## 198. Learning 4D Geometric Priors for Inference-Efficient World Action Models

**arXiv ID:** 2607.05468 | [PDF](https://arxiv.org/pdf/2607.05468v1)

**作者:** Jianjun Zhang `[一作]` (Tongji University), Hanli Wang `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MECo-WAM，一种在训练阶段使用 4D 结构进行多专家联合训练的世界动作模型（World Action Model），在部署时保持原始轻量的 video‑action 推理图，完成几何增强而不增加推理成本。

**💡 创新点**

创新点：1) 仅在训练时注入 4D 结构，通过 decayed 4D read‑mask attention 将几何先验传递到视频‑动作表示；2) 提出了 action‑aware temporal geometric distillation，使模型学习动作相关的空间关系及其随时间的演变；3) 在部署时完全移除 4D 组件，保持推理图不变，兼顾效率与精度。

**🔧 技术方法**

技术手段：多专家联合训练（video、action、4D 专家）；decayed 4D read‑mask attention 约束信息流；action‑aware temporal geometric distillation（关系匹配、动作加权、时间关系匹配）；VGGT encoder 作为 4D 监督源；Fast‑WAM 观察‑动作接口；flow‑matching 损失、DiT action expert、VAE 视频去噪；DPT‑style 深度探针评估表示质量。

**📊 数据集**

数据集：LIBERO（4 个子套件共 10 个任务）、RoboTwin 2.0（双臂操纵，干净与随机化两种评测）、ARX‑R5 实际机器人台面任务（堆叠和排序两种操作）。

**📈 对比分析**

对比方法：与 VLA 策略（π_0、π_0+FAST、OpenVLA、X‑VLA 等）以及 WAM 模型（LingBot‑VA、Motus、Fast‑WAM 等）对比。性能：在 LIBERO 上平均成功率 98.2%（比 Fast‑WAM 高 0.6%），在 RoboTwin 2.0 上平均成功率 92.62%（比 Fast‑WAM 高 0.79%），在真实机器人实验中与 Fast‑WAM 相比保持相同或更高成功率、减少纠正次数、缩短完成时间。

**⚠️ 局限性**

局限性：1) 仅依赖训练时的 4D 监督，若训练数据或 VGGT 质量不足会影响效果；2) 仍缺乏在线几何感知与实时 4D 反馈；3) 主要验证于特定机器人平台和任务，跨平台泛化需进一步评估；4) 对更大规模模型或更复杂场景的可扩展性尚未探讨。

---

## 199. From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations

**arXiv ID:** 2607.06080 | [PDF](https://arxiv.org/pdf/2607.06080v1)

**作者:** Shiyi Ling `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于LLM的多智能体仿真框架（SocaSim），用于从理论到实验层面验证并应用罗伯特·普特南的社会资本理论，重点研究社会网络、信任和规范在集体行动与老年智能照护技术采用中的作用。

**💡 创新点**

创新点：① 将LLM与传统ABM结合，形成理论驱动、可解释的仿真环境；② 通过社交结构、BDI决策和SCM记忆三大模块实现动态社会资本演化；③ 采用对照实验和因果反事实干预，揭示信任在技术采用中的可操作因果杠杆。

**🔧 技术方法**

核心技术：大语言模型（Qwen2.5-14B-Instruct、GPT‑4、GLM‑4）驱动智能体决策与记忆更新；BERT/BDI框架用于推理；多阶段提案-执行仿真流程；网络密度、信任与规范的量化评估与可视化。

**📊 数据集**

数据集：2023年中国社会调查（CGSS）老年人子样本（用于生成人口学属性和社会资本初值）；20名真实老年志愿者（用于人机对照实验）。

**📈 对比分析**

与传统实证方法相比，SocaSim在宏观层面重现了社会资本理论预测（网络密度与集体行动成功正相关，信任与规范递增），与老年人决策呈现高相关性（Pearson r = 0.974）。消融实验表明SST、BDI、SCM三模块对性能贡献显著；在老年智能照护情境中，信任干预使技术采用率提升15.4%，心理压力与焦虑降低约20%，决策矛盾下降25%。

**⚠️ 局限性**

局限性：① 仅基于文本交互，缺乏多模态（语音、图像）信息；② 人机对照样本有限，统计功效受限；③ 只基于中国样本，跨文化普适性尚待验证。

---

## 200. PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents

**arXiv ID:** 2607.06008 | [PDF](https://arxiv.org/pdf/2607.06008v1)

**作者:** Hongliang Li `[一作]` (Beijing Jiaotong University), Kaiyu Huang `[通讯]` (Beijing Jiaotong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PolyWorkBench 基准，评估 LLM 代理在多语言长时序工作流程中的表现。

**💡 创新点**

创新点在于将多语言变异嵌入执行轨迹，并结合结构评分、可执行验证和 LLM‑as‑Judge 三重评估。

**🔧 技术方法**

采用结构化评估规则、Pytest 可执行测试和 LLM‑as‑Judge 语义判断等技术。

**📊 数据集**

数据集包含 67 个跨商贸、知识、法律、本地化、制造五个领域、10 种语言的多语言任务。

**📈 对比分析**

在多种开源与闭源 LLM 与四种代理框架下对比，最佳模型 Pass@1 为 0.921，显示多语言长时序任务仍具挑战。

**⚠️ 局限性**

局限在于评估对代理框架高度敏感，且 LLM‑as‑Judge 与结构评分的相关性低，难以完整捕捉语义一致性。

---

## 201. Drift Happens: An Empirical Study of Neural Architecture Robustness to Temporal Distribution Shift

**arXiv ID:** 2607.05908 | [PDF](https://arxiv.org/pdf/2607.05908v1)

**作者:** Robin Holzinger `[一作]` (University of California, Berkeley), Riccardo Colletti `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对图像分类、文本回归和多标签文本分类三大时序域进行神经网络架构在时间分布漂移下的系统性实验，评估其跨时间泛化性能。

**💡 创新点**

①提出统一的时间漂移矩阵评估框架；②将多种诱导偏置（MLP、CNN、ResNet、ViT、预训练冻结编码器等）在同一实验条件下进行比较；③揭示强诱导偏置的模型在分布内表现最佳但最易退化，冻结预训练编码器虽分布内精度低但更稳健。

**🔧 技术方法**

使用累计训练、时间漂移矩阵、加权损失（加权MSE、加权BCE）、梯度显著性图、Adam/AdamW优化器、冻结预训练模型等技术。

**📊 数据集**

Yearbook（美国高中肖像，近百年）、Amazon Reviews 2023（571M评论，文本回归）、arXiv（280万论文，7类多标签）。

**📈 对比分析**

通过累计训练后对不同时间切片进行评估，构建漂移矩阵，计算在分布内、未来性能与衰减。结果显示：CNN/ResNet等强偏置模型在分布内最高但未来衰减最快；冻结预训练编码器稳定但分布内较低；在arXiv任务中所有架构表现相近，差异不大。

**⚠️ 局限性**

仅覆盖常见诱导偏置，未包含所有架构；数据集时间跨度和规模不一，文本模型共享相同冻结词向量；未进行容量匹配或显著性机制的深入分析；未考虑自适应再训练策略；整体为描述性分析，缺乏机制解释。

---

## 202. Onnes: A Physics-Grounded Multi-Agent LLM Simulator for Cryogenic Fault Diagnosis in Quantum Computing Infrastructure

**arXiv ID:** 2607.05805 | [PDF](https://arxiv.org/pdf/2607.05805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 203. Level-Crossing Density as a Mesh-Free High-Frequency Auxiliary Loss for Implicit Neural Representations

**arXiv ID:** 2607.05815 | [PDF](https://arxiv.org/pdf/2607.05815v1)

**作者:** Gunner Levi Howe `[一作]` `[通讯]` (Independent Researcher), Gunner Levi Howe (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 Rice 级跨密度的可微、无网格、无 FFT 的辅助损失，用于提升隐式神经表示（INR）的高频细节重建；

**💡 创新点**

创新点在于将随机场理论中的水平跨率（通过 co‑area 公式与 Monte‑Carlo 估计实现）转化为训练目标，从而在不需要重采样或梯度目标的情况下提供空间域的频谱驱动；

**🔧 技术方法**

使用了随机场理论（Kac‑Rice 公式）、co‑area 公式、Monte‑Carlo 平滑估计、自动微分梯度、Implicit Neural Representation、以及与 FFL、Sobolev 等对比的频域与梯度域损失；

**📊 数据集**

实验数据集包括 1D 多正弦信号、128×128 的自然图像、256×256 图像在不同稀疏采样下的散点数据，以及合成多尺度纹理；

**📈 对比分析**

在稠密网格上，辅助损失与基线无显著差异；在散点采样下，Kac‑Rice 损失相较 MSE 提升 2.3–3 dB，且与 FFL 接近（在纹理上略优 0.6 dB），但在自然图像上仅匹配而非超越；

**⚠️ 局限性**

局限性包括仅验证 1D/2D 情形、仅为辅助项无法定位具体边缘、需要双向反向传播导致训练成本约 2×、超参数调优范围有限、评估数据集和任务相对单一、对点云 SDF 或 NeRF 等更典型稀疏场景未测试。

---

## 204. AdaStop: Cost-Aware Early Stopping for DNN Test Selection

**arXiv ID:** 2607.05461 | [PDF](https://arxiv.org/pdf/2607.05461v1)

**作者:** Bonan Shen `[一作]` (Independent Researcher), Tao Ning `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AdaStop 框架，基于成本收益模型自动决定 DNN 测试标签停止时机。

**💡 创新点**

推导最优停止阈值 τ=c/v 并结合滑动窗口估计误报率，实现无预算硬编码的成本感知停止。

**🔧 技术方法**

使用不确定性排序（如 DeepGini 等）、滑动窗口估计、阈值/多种停止策略。

**📊 数据集**

在 CIFAR-10、SVHN、FashionMNIST 三个图像数据集上，测试 ResNet-20、VGG-16、DenseNet-121、ShuffleNetV2 四种架构。

**📈 对比分析**

与固定预算、Oracle 及多种停止准则比较，AdaStop 在 23–32% 标签预算下获得 65–84% 召回率，效率提升 3 倍，净价值比全量测试高 18%。

**⚠️ 局限性**

局限在于仅验证图像分类、需要人工估算 c 与 v，假设误报率随进展递减，批量标注场景未覆盖。

---

## 205. Escaping the Procrustean Bed: Groupwise Orthogonal Connectors for Audio-Language Models

**arXiv ID:** 2607.06014 | [PDF](https://arxiv.org/pdf/2607.06014v1)

**作者:** Ho-Lam Chung `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种组间正交连接器（ORCA），通过将 Q-Former 的查询分组并强制组中心正交，来避免查询向量聚合，保留音频中的并行语义与非语义信息。

**💡 创新点**

创新点在于用纯几何约束（组中心正交）重塑连接器输出空间，而不依赖任何属性监督，解决了连接器中“Procrustean bed”导致的语义压缩与并行信息丢失。

**🔧 技术方法**

技术包括组化正交正则化、层混合注意力、交叉注意力、语言模型训练（Qwen3‑4B），以及对多模态音频编码器（Whisper‑Large‑V3）的冻结使用。

**📊 数据集**

训练使用AQA5M（约7000小时，包含情感、性别、年龄、口音等多模态标签）；评估使用CREMA‑D（情感对比），SAKURA多跳推理基准和MMAU通用音频理解基准。

**📈 对比分析**

与标准 Q-Former 4B 基线直接对比，ORCA 在SAKURA多跳推理准确率提升26.4个百分点（从48.8%到75.2%），查询余弦相似度从0.923降至0.077，跨说话人方差提升75倍；在MMAU上保持与其他4B模型相当；与8B模型相比，虽仍有差距，但在相同参数量与训练数据下表现更优。

**⚠️ 局限性**

局限在于受限于音频编码器先前已丢失的信息和噪声掩盖，无法恢复精细韵律细节；仅靠结构改进无法改变语言模型对语义轴的偏好，仍需结合更多数据或任务设计来进一步提升并行信息的保留与利用。

---

## 206. GPU-Accelerated Effective Resistance Analysis for 3D IC Power Delivery Network

**arXiv ID:** 2607.05818 | [PDF](https://arxiv.org/pdf/2607.05818v1)

**作者:** Jingchao Hu `[一作]` (Zhejiang University), Zhou Jin `[通讯]` (Zhejiang University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对3D集成电路功率分配网络（PDN）中有效电阻（Effective Resistance）的早期设计阶段分析，提出一种基于GPU加速的框架，实现快速、精确的有效电阻计算。

**💡 创新点**

创新点包括：1) 将3D PDN拆分为多条2D子网络，采用分治策略降低计算规模；2) 对原始导纳矩阵进行重构，得到只依赖TSV和die几何参数的唯一系数矩阵；3) 通过行列变换将多组bump‑load组合的求解从m×n个线性系统降为m+n个共享同一系数矩阵的线性系统，显著减少矩阵分解次数；4) 在GPU上实现全部可并行部分（TSV间阻抗、前向后向替代、向量点积等），实现5–6阶数的速度提升。

**🔧 技术方法**

使用的技术主要包括：GPU并行计算（CUDA/CuPy）、分治策略、矩阵行列变换、前向后向替代求解、向量点积、贝叶斯优化（用于TSV规划）以及多重角公式递归化余弦运算。

**📊 数据集**

实验数据集来自9个基于14nm工艺的早期3D IC PDN基准，包含2–6个die、4.4–7.8×11.9mm²的占地面积、数百万至数千万个节点、1.8–3.4×10⁴ TSV、432–1369个bump、1×10⁴个负载。

**📈 对比分析**

与传统Cholesky直接求解器（golden）以及最近的伪逆求解器相比，本框架在相同硬件（Intel Xeon 8475B + NVIDIA A100）上实现了5–6倍的速度提升（相比golden）和2–5倍的速度提升（相比伪逆），且平均/最大相对误差均在10⁻⁸–10⁻⁹之间，可忽略不计。

**⚠️ 局限性**

局限性包括：1) 依赖于早期设计阶段的均匀网格假设，复杂非均匀结构时可能需要额外的预处理；2) GPU内存受限，极大规模（如D9）仍可能因内存不足而无法求解；3) 只针对TSV为电流源的模型，其他非TSV电源或负载配置的扩展尚未验证。

---

## 207. Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH

**arXiv ID:** 2607.05956 | [PDF](https://arxiv.org/pdf/2607.05956v1)

**作者:** Adam Faci `[一作]`, Stéphane Pouyllau `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在 LLMs4EU 项目中，ReSearch_SSH 用大语言模型与知识图谱驱动的检索增强生成（GraphRAG）技术，扩展 ISIDORE 平台，支持社会科学与人文领域的多语言、可追溯且符合学术规范的文献检索与综述生成。

**💡 创新点**

创新点包括：① 以学科为导向的多语言领域对齐和检索优化；② 将知识图谱（Wikidata、OpenAIRE 等）嵌入 RAG，实现答案可追溯与可解释；③ 通过专家面板进行混合评估，保证系统的学术可信度和法律合规性；④ 将开放欧洲模型与商业模型对比，强调全开放、可审计的模型工作流。

**🔧 技术方法**

核心技术：GraphRAG 架构、检索增强生成、知识图谱推理、领域对齐与指令调优（fine‑tune）以及多语言模型（EuroLLM 等）。

**📊 数据集**

主要数据集：ISTEX SSH 子集（约300万文档，60+ 语言），AIUCD 与 Umanistica Digitale 的意大利 DH 论文，Hypotheses 博客与 Nakala 研究数据，全部均配有 TEI/XML 文本与 JSON 元数据。

**📈 对比分析**

评估方法：量化指标（检索召回率、文档多源摘要质量、源引用准确度、幻觉检测）以及专家面板的定性评估；目前尚未公开具体性能数值，但已制定基准与情境化评测流程。

**⚠️ 局限性**

局限性：① 依赖以法语为主的数据库，初期多语言覆盖有限；② 对检索行为的历史数据依赖度高，缺少相似语料时可替代方案尚未完善；③ 需要更多跨机构的测试与泛化验证；④ 在处理复杂人文文本时仍可能出现解释性与引用误差；⑤ 受法律与版权约束，部分语料的模型训练与公开受限。

---

## 208. aiAuthZ: Off-Host, Identity-Bound Authorization for AI Agents

**arXiv ID:** 2607.05518 | [PDF](https://arxiv.org/pdf/2607.05518v1)

**作者:** Sai Varun Kodathala `[一作]` `[通讯]` (SportsVision AI), Sai Varun Kodathala (SportsVision AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 aiAuthZ，一个离线授权网关，用于在 AI 代理工具调用前验证用户身份并执行基于角色和参数的策略，从而防止被注入文本诱导的权限滥用。

**💡 创新点**

创新点包括：① 按消息 HMAC 绑定每一次用户交互，确保即使模型被欺骗也无法伪造身份；② 将授权决策移到代理主机外部，阻断模型内部工具的直接调用；③ 采用哈希链审计日志和可在重压缩、截图后仍能验证的 QR 收据；④ 通过凭证代理让模型主机无长期密钥，提升安全边界。

**🔧 技术方法**

使用的技术包括：HMAC‑SHA256 消息签名、单次 nonce 与时间戳校验、基于 YAML 的离线策略引擎、SQLite/PostgreSQL 与 Redis 的组合存储、AES‑256‑GCM 加密、FastAPI + OpenAPI 接口、QR 码生成与 OpenCV 解码、SHA‑256 哈希链审计、以及对 Model Context Protocol 的适配。

**📊 数据集**

使用的数据集主要是：① Agents of Chaos 公开案例库（11 个案例，包含身份/授权失败等），② AgentDojo 银行套件（包含注入攻击情景），③ 公开的 15 种当代 LLM（通过 OpenRouter 访问）在 8 个攻击场景下的实验记录，以及 25 条不同 JPEG/截图/裁剪渠道下的收据鲁棒性测试。

**📈 对比分析**

对比方法：将 aiAuthZ 与仅基于参数的 Open Agent Passport 配置、仅委托令牌的 Agent Identity Protocol 进行对照；同时与 15 个模型在 8 个攻击场景下的拒绝率、以及与无防护、Spotlighting 防御的 AgentDojo 结果进行对比。性能表现：决策延迟 0.006–0.030 ms，完全阻止所有模型的工具层攻击，拒绝率从 0% 到 100% 的不均匀性被消除；收据在 8 条渠道下平均 94% 验证通过，伪造率 0%。

**⚠️ 局限性**

局限性包括：① 仍需在部署时关闭模型内部重叠工具或使用凭证代理，防止绕过；② 只处理授权层，无法阻止模型被欺骗导致的有害文本生成；③ 允许的工具序列可能通过组合实现恶意效果；④ 对外部非可信审计者不提供非否认性（使用对称 HMAC 而非公钥签名）。

---

## 209. Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks

**arXiv ID:** 2607.05545 | [PDF](https://arxiv.org/pdf/2607.05545v1)

**作者:** Yibo Hu `[一作]` (Illinois Institute of Technology), Jiaming Qu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

作者通过移除显式说话者，只保留重复答案文本，验证大多数模型的答复偏差不依赖说话者，而是答案本身的影响。

**💡 创新点**

创新点在于提出并量化了“无来源底线”(speaker‑free floor)，即在没有说话者提示的情况下，模型仍会大幅修正答案，并进一步区分源标签对底线的增量效应。

**🔧 技术方法**

技术上采用了确定性对数概率仲裁（deterministic log‑probability arbitration）与贪婪解码，构造多种说话者框架（无来源、最小标签、丰富同行、专家面板）来对比模型的答复改动。

**📊 数据集**

实验使用七个公开基准：ARC‑Challenge、MMLU‑Pro、TruthfulQA 以及四个 BBH 任务，共计约 2.1 万条样本。

**📈 对比分析**

与普通重新询问（plain re‑ask）以及长度控制 baseline 对比，模型在无来源条件下的有害修正率高达 66.5%，专家面板进一步提升至 79.4%，表明说话者框架仅在已存在的大底线之上产生小幅增量。

**⚠️ 局限性**

局限性包括仅在单轮多选题环境下评估、使用开放权重指令调优模型、贪婪解码导致的随机性缺失、未涵盖更自由文本或非多选情境，以及对更大规模或闭源模型的推广性不足。

---

## 210. Federated Physics-Grounded Reinforcement Learning for Distributed Stability Control in Smart Grids

**arXiv ID:** 2607.05553 | [PDF](https://arxiv.org/pdf/2607.05553v1)

**作者:** Omar Al-Refai `[一作]` (Texas A&M University), Eman Hammad `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了FedPPO-PG框架，使用联邦多智能体强化学习实现无中央协调的瞬态稳定控制。

**💡 创新点**

创新点包括基于后故障Susceptance矩阵的物理耦合邻域选择、DPFL教师引导的策略初始化以及基于性能的联邦平均和本地微调。

**🔧 技术方法**

采用联邦学习、近端策略优化（PPO）、协同价值估计、GAE、Meta-RL微调以及物理耦合信息融合等技术。

**📊 数据集**

在IEEE 39节点新英格兰测试系统上，使用10台发电机的多种三相线路故障数据进行训练与验证。

**📈 对比分析**

与集中式CPFL和传统分布式DPFL对比，FedPPO-PG在24个实验中实现100%稳定率，平均稳定时间比CPFL降低72.4%，控制功率降低7-14倍，单智能体推理延迟仅0.056 ms。

**⚠️ 局限性**

局限性包括对通信延迟、测量错误、网络拥塞等网络物理干扰缺乏鲁棒性评估，以及邻域大小、规模化到更大系统时的可扩展性尚未验证。

---

## 211. AIED's Unfinished Mission: Centering Agency and Motivation in the Age of Effortless Bypass

**arXiv ID:** 2607.05557 | [PDF](https://arxiv.org/pdf/2607.05557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 212. WebRetriever: A Large-Scale Comprehensive Benchmark for Efficient Web Agent Evaluation

**arXiv ID:** 2607.06118 | [PDF](https://arxiv.org/pdf/2607.06118v1)

**作者:** Wei Dong `[一作]` (Mininglamp Technology), Chenxu Zhao `[通讯]` (Mininglamp Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 WebRetriever 这一大规模、多域、多意图的真实在线基准和 NavEval 这一基于 LLM 的自动评估框架，并设计了三套针对导航、知识辅助和端到端任务完成的评估协议。

**💡 创新点**

创新点在于：①覆盖 800 个真实网站、1550 个跨行业任务，显著提升规模和多样性；②NavEval 通过规则过滤的请求序列、动作轨迹与最终截图共同输入 LLM，达成 90%+ 的人类一致率；③将评估拆分为三种协议，全面揭示代理在导航、知识运用与完整任务执行方面的真实能力。

**🔧 技术方法**

使用技术包括 Playwright 实时抓取交互轨迹、规则化过滤请求序列、Claude‑4.5‑Sonnet 作为 LLM‑as‑Judge 进行判定、结构化任务描述与文档生成流程。

**📊 数据集**

数据集为 WebRetriever，包含 800 个精选高质量站点、1550 个任务，涵盖八大行业，任务意图分为普通与专业两类。

**📈 对比分析**

与 SeeAct、Agent‑E、UI‑TARS‑1.5、Browser‑Use、Claude‑4.5、Gemini‑2.5‑Pro 等模型在三协议下进行比较；NavEval 的人类一致率>90%，但模型在三协议的成功率仅为 21.1%、29.2% 和 11.8%，显示当前代理在真实环境中的不足。

**⚠️ 局限性**

局限性包括：①尽管评估精细，但端到端信息提取仍低，揭示代理缺乏深度页面理解；②文档生成依赖人工校准，无法完全自动化；③网站动态变化仍导致任务失效，需要持续维护更新。

---

## 213. CCBENCH: Assessing LLM Cultural Competence via Implicitly Signaled Norms using Health Queries

**arXiv ID:** 2607.05405 | [PDF](https://arxiv.org/pdf/2607.05405v1)

**作者:** Vasudha Varadarajan `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一套评估大语言模型文化适应性的框架和健康领域基准；

**💡 创新点**

将文化身份视为持续的规范遵循状态，而非二元标签，并通过隐式对话线索测试模型的推理与适配能力；

**🔧 技术方法**

利用对话生成、层级价值-规范映射、基于检查表的评估及链式推理提示；

**📊 数据集**

使用 Mosaica 文化健康档案、真实健康论坛提问、WildChat 1M 语料生成对话史；

**📈 对比分析**

在五种顶尖模型上进行四种提示配置实验（无上下文、对话上下文、文化链式思考、显式规范），结果显示模型在避免文化偏差方面表现较好（70%+ 避免率），但在主动遵循规范时仅约5-7% 的准确率，整体文化适配得分低；

**⚠️ 局限性**

局限在于仅覆盖六种文化、依赖大型模型与昂贵计算、规范与价值映射可能不完全真实、缺乏对更细粒度文化差异的捕捉。

---

## 214. Propose and Attend: Training-free MLLM Grounding Confidence via Multi-Token Localized Attention

**arXiv ID:** 2607.05978 | [PDF](https://arxiv.org/pdf/2607.05978v1)

**作者:** Daniel Shalam `[一作]` (Amazon), Tal Remez `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的多令牌局部注意力（MTLA）评分，用来评估多模态大型语言模型（MLLM）生成的定位预测是否可信。

**💡 创新点**

创新点在于将注意力聚焦到模型自身预测的区域并聚合所有相关输出令牌，从而显著提升对幻觉（hallucination）的区分度。

**🔧 技术方法**

利用模型自带的多层多头注意力权重，结合区域掩码与多令牌平均，并在自一致性投票后进行置信度重排序。

**📊 数据集**

在跨图像、视频、音频的定位基准上进行验证，主要使用 COCO、Charades-STA、QVHighlights 及 AudioSet-Strong 数据集。

**📈 对比分析**

与 token 概率、SVAR、GLSim 等无训练基线相比，MTLA 在幻觉检测上提升 8–21% AP，且作为置信度重排序后，零样本 COCO AP 从 20.4 提升至 37.0，几乎逼近监督检测器。

**⚠️ 局限性**

局限性包括需访问模型注意力映射且计算成本较高，尤其在自一致性多采样时需要多次推理。

---

## 215. Scalable Perturbation Learning for Online Self-Supervised Echo State Networks

**arXiv ID:** 2607.06079 | [PDF](https://arxiv.org/pdf/2607.06079v1)

**作者:** Taiki Yamada `[一作]` (University of Tokyo), Kantaro Fujiwara `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对大型ESN的在线自监督学习规则，利用正交分解将成本函数分离，聚焦于输入维度的低维子空间。

**💡 创新点**

创新点在于通过正交分解消除冗余并将扰动维度从reservoir维度降到输入维度，从而在保持标量反馈的同时消除梯度方差随网络尺寸增长的趋势。

**🔧 技术方法**

使用了Echo State Networks、正交成本分解、节点/权重扰动法以及梯度下降的理论框架。

**📊 数据集**

实验采用合成时序数据（输入重构任务）来验证方法。

**📈 对比分析**

与SGD、权重扰动(WP)和节点扰动(NP)比较，结果表明该方法在大reservoir尺寸下保持更高的信噪比、收敛更快、对学习率更鲁棒，最终RMSE显著低于对比方法。

**⚠️ 局限性**

局限性在于仅在ESN自监督输入重构任务上验证，未验证其在更一般的任务或真实数据集上的适用性。

---

## 216. Association Restoration Test: Revealing Restorable Shortcuts after Unlearning

**arXiv ID:** 2607.05726 | [PDF](https://arxiv.org/pdf/2607.05726v1)

**作者:** Amy Lu `[一作]` (Stanford University), Changxiu Ji `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Association Restoration Test（ART），一种后置诊断方法，用于检测在去学习或偏差缓解后，模型中是否仍存在可被恢复的标签–属性快捷方式，并在 Waterbirds、CelebA、SpuCoDogs 与 ISIC 2019 这四个视觉基准上对多种方法进行评估。

**💡 创新点**

创新点在于：①首次针对关联级别的去学习进行功能性恢复测试，区分表征可读性与功能可恢复性；②通过类条件关联方向估计、门控与残差放大，实现对快捷方式可恢复性的直接检测；③揭示传统输出或表征评估与恢复性评估可能产生的偏差，推动更完善的评估框架。

**🔧 技术方法**

使用技术包括：在冻结的 ResNet‑50 末层特征空间中估计类条件关联方向、投影与门控；对残差方向进行放大并与原始分类头一起预测；结合线性探针（LP）和最近中心分类器（NCC）评估表征可读性；利用 WGA 与 CSR 等指标衡量恢复效果。

**📊 数据集**

实验数据集：Waterbirds（鸟类与背景）、CelebA（性别与头发颜色）、SpuCoDogs（小/大犬与室内/室外）、ISIC 2019（皮肤病多分类并加入时间戳伪相关）。

**📈 对比分析**

对比方法包括：ERM、Balanced Retrain、GroupDRO、DFR、JTT 以及针对关联的去学习变体 A‑NegGrad+、A‑SCRUB、A‑SalUn、A‑SSD。结果显示：多数方法在 WGA 上有提升，但 ART 发现大幅 WGA 降低和 CSR 增加，表明快捷方式仍可恢复；Balanced Retrain 与 GroupDRO/DFR 在恢复性上最稳；A‑系列方法易被恢复，显示其对快捷方式的抑制不彻底。

**⚠️ 局限性**

局限性：ART 仅针对线性特征空间中的残差方向；对非线性或多属性关联的恢复可能不敏感；需要拥有标签与属性信息的审计样本；可能无法捕捉所有可恢复路径，且对不同网络层和架构的适用性仍需进一步研究。

---

## 217. Contrastive Predictive Coding with Compression for Enhanced Channel State Feedback in Wireless Networks

**arXiv ID:** 2607.05419 | [PDF](https://arxiv.org/pdf/2607.05419v1)

**作者:** Ahmed Y. Radwan `[一作]` (York University), Matthew Baker `[通讯]` (Nokia UK)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在3GPP兼容的CSI反馈框架中引入对比预测编码（CPC），实现压缩与预测的统一化，以解决信道老化问题。

**💡 创新点**

创新点在于将CPC直接嵌入CSI压缩架构，提出CPC-before-Compression与CPC-after-Compression两种变体；并通过联合的1‑SGCS与InfoNCE损失实现重建精度与时间预测的双重优化，显著降低解码器复杂度。

**🔧 技术方法**

使用技术包括对比预测编码（CPC）、GRU自回归模型、1‑SGCS与InfoNCE联合损失、量化线性压缩、结构化剪枝与低秩分解等。

**📊 数据集**

实验数据集涵盖公开的3GPP兼容CSI数据：Nokia、Oppo、CATT以及三者混合数据集。

**📈 对比分析**

与3GPP基线在SGCS、InfoNCE、推理时间、GFLOPs等指标对比，CPC-before-Compression在SGCS上可达0.90以上，解码器GFLOPs下降32倍；在不同数据集、预测步长与压缩尺寸下均保持高重建质量，CPC-after-Compression在解码端实现未来CSI估计且保持与基线相当的重建精度。

**⚠️ 局限性**

局限性包括：CPC-after-Compression受压缩信息损失限制，导致预测性能不如CPC-before-Compression；CPC-before-Compression对UE端计算资源要求高；在某些数据集或较大预测步长下InfoNCE损失显著上升，模型对极端稀疏或低速环境的适应性有限。

---

## 218. Synthetic Consumer Insight Generation with Large Language Models

**arXiv ID:** 2607.05761 | [PDF](https://arxiv.org/pdf/2607.05761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 219. Chunky Chains: Graph Drawings on Small Screens

**arXiv ID:** 2607.06029 | [PDF](https://arxiv.org/pdf/2607.06029v1)

**作者:** Tim Hegemann `[一作]` (Universität Würzburg), Samuel Wolf `[通讯]` (Universität Würzburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对窄屏幕的图形可视化方法 Chunky Chains，利用垂直桶状弦图排列节点，短边在桶内绘制，长边仅部分绘制以减少视觉负担。

**💡 创新点**

创新点在于把桶宽度与长边最小化作为新的组合优化问题，给出组合框架、精确与启发式算法，并为手机屏幕量身定制几何布局与交叉最小化技术。

**🔧 技术方法**

采用整数线性规划、动态规划、匹配求解、启发式搜索与图形几何构造等多种算法技术。

**📊 数据集**

实验使用 Rome Graphs（约 11,531 个真实网络）和合成 k‑路径图（10–200 顶点、k∈{3,5,7}）两组数据集。

**📈 对比分析**

与 ILP 求解器（Gurobi/HiGHS）以及三种带宽/主成分排序的启发式和 meta‑heuristic 结合进行比较，实验表明大多数图在容量 8 时可无长边，meta‑heuristic 在长边较多时表现优于单纯启发式，但求解速度相对较慢。

**⚠️ 局限性**

主要局限包括 ILP 模型弱导致求解困难、缺乏高效的交叉最小化精确算法、meta‑heuristic 需要人工终止、未证明所有图可绘制平面 Chunky Chains，以及对认知负荷的进一步评估不足。

---

## 220. SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation

**arXiv ID:** 2607.05721 | [PDF](https://arxiv.org/pdf/2607.05721v1)

**作者:** Yimeng Zhang `[一作]` (Amazon), Dakuo Wang `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种面向自然语言生成结果的跨度级不确定性估计（SLUE）框架，并基于此构建了首个跨度级不确定性评估基准 SpanUQ。

**💡 创新点**

创新点包括：① 将不确定性估计从 token/sequence 级提升到跨度级；② 采用 DETR 风格的集合预测解码器联合跨度检测与不确定性推理；③ 使用三元混合 Beta 先验来建模不确定性分布；④ 通过不确定性条件下的迭代细化（UCIR）提升预测精度；⑤ 通过单向前向推断实现 10–20× 的速度优势；⑥ 通过多样本采样与自动标签化构造连续软标签的跨度级基准。

**🔧 技术方法**

核心技术包括：冻结 LLM 隐藏状态融合、DETR 目标检测框架、混合 Beta 先验回归、对比排序损失、Hungarian 匹配、软边界掩码、梯度可微的跨度内容聚合、以及基于重要性加权的序列级聚合。

**📊 数据集**

使用 SpanUQ 数据集：20K 个提示、约293K 个跨度，涵盖长篇 QA、TriviaQA、ELI5、传记、FELM 等五个领域；通过 20 个多样化采样与知识库检索生成软标签。

**📈 对比分析**

与 token‑entropy、MLP probe、SelfCheckGPT‑NLI、Verbalized Confidence、FActScore 等多种粒度（token、sequence、claim）方法对比，SpanUQ 在五个大模型（4B–30B）上取得 AUROC 0.908–0.944、MAE 0.110–0.129、跨度检测 F1 0.910，显著优于基准且推理速度快 10–20×。

**⚠️ 局限性**

局限性包括：仅利用 LLM 隐藏状态，无法捕捉生成过程本身的不确定性；需要白盒访问 LLM；多样本标签构造计算成本高；实验仅覆盖英文和事实性文本，跨语言和推理类不确定性尚未验证。

---

## 221. Learnable Weighting of Intra-Attribute Distances for Categorical Data Clustering with Nominal and Ordinal Attributes

**arXiv ID:** 2607.05464 | [PDF](https://arxiv.org/pdf/2607.05464v1)

**作者:** Yiqun Zhang `[一作]`, Yiu-ming Cheung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一测量名义属性和序数属性的内属性距离的距离度量，并基于此设计了可学习的距离权重机制，构成了能够对任意组合的名义/序数属性进行聚类的算法 HD‑NDW。

**💡 创新点**

创新点在于：① 将名义属性转换为二值序数属性，从而实现对名义与序数属性的距离测量统一；② 基于条件概率和图结构的距离定义（类似地球搬运距离），能够保留序数属性的顺序信息；③ 设计了一种迭代更新距离权重的机制，避免了频率偏差和共现稀疏问题，提升了聚类效果。

**🔧 技术方法**

使用的技术包括：基于图的概率分布差异距离（ψ函数）；条件概率计算与上下文信息利用；k‑modes 风格的簇更新；软最大化权重更新（soft‑max）；以及整个算法的迭代优化框架。

**📊 数据集**

实验数据集包括 15 个公开数据集：6 个混合型（如 Lenses、Cancer、Nursery 等）、5 个序数型（如 Photo、Lecturer、Social 等）和 4 个名义型（如 Solar、Zoo、Voting、Soybean），并进行了合成数据实验。

**📈 对比分析**

与 9 个基线方法（KMD、ECC、WKM、MWKM、SBC、CDE、UNTIE、DLC 等）在 ARI、NMI、CA 等指标上进行比较。实验结果显示 HD‑NDW 在大多数数据集上取得显著更高的指标，Wilcoxon 与 Bonferroni‑Dunn 检验表明差异显著；算法收敛快速，计算成本与样本量 n 成线性关系。

**⚠️ 局限性**

局限性：① 对名义属性的二值化可能导致信息损失，在纯名义数据集上的优势不如混合/序数数据；② 需要先知聚类数 k；③ 当属性只有两个取值时，权重学习退化为传统属性权重，效果有限。

---

## 222. Where to cut, how deep: BPE and Unigram-LM on chemistry SMILES

**arXiv ID:** 2607.05691 | [PDF](https://arxiv.org/pdf/2607.05691v1)

**作者:** Hunter Heidenreich `[一作]` `[通讯]` (Independent Researcher), Hunter Heidenreich (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在固定的 165 词基底（Smirk）上，对 BPE 与 Unigram‑LM 两种子词分词算法进行系统比较，使用不同语料（PubChem、ZINC‑22、COCONUT、REAL‑Space）以及两种括号边界策略，探究它们在词表重叠、分词粒度和频率分布上的差异。

**💡 创新点**

首次在可学习嵌入范围内对 SMILES 子词算法进行完全匹配实验，证明两种算法在词表、token 数量与粒度上产生近乎无重叠、且差异在 29–41% 之间，揭示子词算法是化学 LLM 的关键设计决策而非默认继承。

**🔧 技术方法**

技术方法包括：使用 Smirk 固定 OpenSMILES 字形基底；实现 BPE 与 Unigram‑LM；字母级预分词与括号边界策略；计算 Jaccard 重叠、token 数量/粒度、频率失衡、边界一致性；引入 Learnability bar F_95,100 评估词表可学习性；以及超参数与结构探测的敏感性分析。

**📊 数据集**

数据集涵盖：PubChem（约 5,000 万分子）、ZINC‑22（约 1,000 万）、COCONUT（约 74 万）、REAL‑Space（约 1.36 亿，1% 样本）。所有数据均做了 RDKit 同构化、去重与基底兼容性过滤。

**📈 对比分析**

比较方法：在匹配条件（相同语料、词表大小、括号边界）下，分别计算词表 Jaccard 重叠、token 数量相对差异（rel|Δf|）、频率不均衡差异。实验结果显示：BPE 生成的词表与 Unigram‑LM 几乎无重叠；Unigram‑LM 的 token 数量比 BPE 多 29–41%；两者差异在不同语料、边界策略和词表规模下均保持稳定；分词粒度差异可通过嵌入长度的 30–40% 差距量化。

**⚠️ 局限性**

局限性：仅对固定子词词表进行评估，未测量下游语言模型性能；未覆盖字节级或动态边界分词方案；未尝试不同基底或 Kekulé 表示；评估指标中部分如 Jaccard 无置信区间；Learnability 阈值 F_95,100 的适用性基于 NMT 经验，可能与化学领域存在差异。

---

## 223. Enhanced Seam Segmentation for Automated Welding Robot in Construction Through Transfer Learning: Addressing Limitations of Bilateral Segmentation Network

**arXiv ID:** 2607.06150 | [PDF](https://arxiv.org/pdf/2607.06150v1)

**作者:** Keonvin Park `[一作]` (Seoul National University), Doyun Lee `[通讯]` (Georgia Southern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对机器人焊接场景中焊缝分割的反射鲁棒性不足，提出了基于 BiSeNetV2 的迁移学习与 CE–Lovász 混合损失框架，显著提升了薄焊缝的连续性与鲁棒性。

**💡 创新点**

创新点在于通过学习稳定化（迁移学习 + 混合损失）而非网络扩展来提升反射鲁棒性，保持轻量级实时推理性能；并在极端反射环境下实现高恢复率。

**🔧 技术方法**

技术包括 BiSeNetV2 轻量级架构、OHEM 预训练、全局/局部双重损失（CE–Lovász）、反射感知数据增强、置信图可视化与迁移学习策略。

**📊 数据集**

使用公开的 WJ3600（约 3,600 张焊接图像）数据集，包含三类标签（Background、Plate、Joint），已覆盖多种背景、光照与反射条件。

**📈 对比分析**

在与基线 OHEM 模型以及 DeepLabV3+、U‑Net、SegFormer‑B0 等主干进行对比时，BiSeNetV2 迁移+混合损失在 Joint IoU 从 59.40% 提升至 81.76%（+22.36%），mIoU 由 79.66% 提升至 90.73%，恢复率达到 96.33%，且未增加 FLOPs、参数量或延迟。

**⚠️ 局限性**

局限性包括对数据多样性和真实施工现场的泛化仍有限；未完成闭环焊接实验；在极端反射导致极低 IoU 的极端案例仍有提升空间。

---

## 224. Computing Smith Forms Modulo $p^2$ of Sparse Matrices Faster Than Matrix Multiplication

**arXiv ID:** 2607.05800 | [PDF](https://arxiv.org/pdf/2607.05800v1)

**作者:** Mark Giesbrecht `[一作]` `[通讯]` (University of Waterloo), Mark Giesbrecht (University of Waterloo)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种随机化黑盒算法，在稀疏矩阵模 p² 的情况下，快速求出其 Smith 正常形的三个不变因子计数 r₀、r₁、r₂。

**💡 创新点**

创新点在于：①将单位块 B 与其 Schur 补 J 分离，证明 J 可写成 p·T，T 只需通过“carry‑safe”Hensel 逆推而不显式构造；②利用稀疏块 Wiedemann/EggSV 结构化求逆与秩测试，将块求解与分块投影结合，显著降低投影成本；③通过快速秩‑profile 预处理与块大小自适应，获得 n^{3-1/(ω-1)} 的运算复杂度，低于任何稠密算法。

**🔧 技术方法**

主要技术包括：稀疏块投影与 Hensel 递推求逆；块 Wiedemann/EGGSV 逆/秩算法；快速秩‑profile 预处理（PRECONDSXS 以及 butterfly 路由网络）；以及在局部环 /p² 上的 carry‑safe 乘法与除法。

**📊 数据集**

实验采用标准稀疏矩阵黑盒接口（如随机稀疏矩阵、Hankel/Toeplitz 等结构），未给出具体数据集名称，强调算法适用于任何满足黑盒乘法耗时 O(n) 的稀疏/结构化矩阵。

**📈 对比分析**

与传统稠密 SNF 计算（复杂度 O(n^ω)）相比，该方法在 ω=3 时实现 O(n^{2.5}) 运算，在当前最优 ω≈2.371339 时实现 O(n^{2.27})，显著低于稠密阈值，且在稀疏场景下实际运行速度更快。

**⚠️ 局限性**

局限性：仅适用于长度为 2 的局部环 /p²；对更高幂 p^e (e≥3) 的直接递归无法突破 n^ω 的界限；算法为 Monte‑Carlo，存在可控但非零错误概率，未给出 Las Vegas 版本；对大规模实际数据的实测与对比仍待验证。

---

## 225. Teaching LTL and ω-automata with Spot

**arXiv ID:** 2607.05907 | [PDF](https://arxiv.org/pdf/2607.05907v1)

**作者:** Alexandre Duret-Lutz `[一作]` `[通讯]` (EPITA Research Laboratory), Alexandre Duret-Lutz (EPITA Research Laboratory)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

展示Spot库在教育环境中的三种交互式接口：无安装的在线LTL工具、结合代码与图示的Jupyter Notebook、以及命令行工具。

**💡 创新点**

创新在于提供零安装、可直接浏览器使用的LTL/ω-automata可视化平台，并将Spot的完整功能暴露给教学者，使学生能即时看到公式与自动机的对应关系，支持公式简化、等价、蕴含、层次分析等教学场景。

**🔧 技术方法**

使用Spot C++/Python库、Graphviz绘图、web前端技术（单页应用）、Python Jupyter Notebook API及ltlfilt等命令行工具。

**📊 数据集**

无专门数据集，主要利用随机公式生成器（ltlfilt）和内置示例公式来演示和练习。

**📈 对比分析**

论文未进行传统意义上的实验对比，只描述了各接口在教学中的交互性和易用性；命令行工具通过管道快速筛选满足特定属性的公式，展示了高效性。

**⚠️ 局限性**

局限在于教学功能相对有限，主要由研究需求驱动；某些教学工作流未优化；需要教师提交功能请求以提升课堂使用体验。

---

## 226. CANONIC: Governance Is Compilation

**arXiv ID:** 2607.05410 | [PDF](https://arxiv.org/pdf/2607.05410v1)

**作者:** Dexter Hadley `[一作]` `[通讯]` (CANONIC Foundation), Dexter Hadley (CANONIC Foundation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CANONIC框架，将内容治理视为编译过程，采用三大公约实现结构化审核。

**💡 创新点**

创新点在于将治理规则映射为编译器的语法、作用域和类型系统，并实现可决定的线性时间入口门控。

**🔧 技术方法**

使用形式化三公约（Triad、Inheritance、Introspection）、git作为证据账本以及验证器服务。

**📊 数据集**

评估基于一月份的构造窗口内的10个仓库、20个治理作用域的自生成内容，以及公开的交叉供应商基准。

**📈 对比分析**

与三种门控（结构化、检索相似度、检索+蕴含、语义评判）进行对比，结果显示结构门对真值无关，语义门仅在专家域有效，检索蕴含门可捕捉伪造。

**⚠️ 局限性**

局限包括仅在单一自作者构造窗口内验证，未包含人类对抗实验，规模有限且仅适用于治理文本，缺乏完整性证明。

---

## 227. The Balkanization of Execution-Security Research for AI Coding Agents: Isolation, Access Control, and Time-of-Check-to-Time-of-Use Vulnerabilities

**arXiv ID:** 2607.05743 | [PDF](https://arxiv.org/pdf/2607.05743v1)

**作者:** Mohammadreza Rashidi `[一作]` `[通讯]`, Mohammadreza Rashidi

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化梳理并验证2023-2026年间关于AI编码代理执行层安全的论文和已公开的CVE，构建了按执行安全机制划分的知识体系；

**💡 创新点**

首次提出五个跨类别的安全空白点（隔离与访问控制评估缺口、执行容忍性与防御评估缺口、TOCTOU与MCP术语不一致、假设“诚实的策略作者”导致缺陷、范围蔓延失败模式），并给出针对性研究议程；

**🔧 技术方法**

采用文献检索与手工验证协议（直接访问原始论文摘要与NIST CVE数据库），构建可审计的分类词典；

**📊 数据集**

主要数据源为已发表的论文和公开的CVE条目，未使用传统机器学习数据集；

**📈 对比分析**

不涉及实验对比，而是对已发表工作进行归类与跨维度洞察，评估未统一基准下的安全评估差异；

**⚠️ 局限性**

局限在于仅覆盖已公开文献，时间窗口截至2026年；缺乏统一实验基准和对新兴代理技术的即时评估，且未覆盖模型层的对齐问题。

---

## 228. Few-Medoids: An Embarrassingly Simple Coreset Selection Method for Few-Shot Knowledge Distillation

**arXiv ID:** 2607.05891 | [PDF](https://arxiv.org/pdf/2607.05891v1)

**作者:** Cemil-Andrei Dilmac `[一作]` (University of Bucharest), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究将核心子集选择与知识蒸馏相结合，提出了一种基于教师潜在空间的简单核心子集选取方法——few-medoids，并在多种模型对和数据集上进行少样本蒸馏实验。

**💡 创新点**

创新点在于：①用教师模型的特征空间中每类的几何中心（类中心点）来挑选样本，避免复杂启发式或梯度信息；②在少样本蒸馏环境中，该方法表现出色，能系统性超越随机、herding、k-center以及PCA-guided等基线。

**🔧 技术方法**

技术包括：训练无关的几何核心子集选择（计算每类特征的平均欧氏距离），使用Hinton的软标签蒸馏（KL散度+交叉熵），以及传统的ResNet、ViT架构。

**📊 数据集**

使用四个公开图像分类数据集：CIFAR-10、CIFAR-100、Oxford Flowers 102、Food-101。

**📈 对比分析**

与随机、herding、k-center Greedy和PCA-guided四种基线对比，实验表明few-medoids在大多数数据集和模型组合（尤其是ResNet-34→ResNet-18和ViT-B/16→ResNet-50）下，能在k=1~64的少样本范围内实现最高或第二高的测试准确率；在更大k（如128）时，随机或herding偶有超越，但总体表现稳定。

**⚠️ 局限性**

局限性：在极少样本（如Oxford Flowers 102的每类10张图）或预训练学生模型（ViT-B/16→ViT-Small）场景下，few-medoids表现不一；方法依赖教师模型的特征空间分布，若教师不具备良好语义结构可能效果受限。

---

## 229. FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents

**arXiv ID:** 2607.05682 | [PDF](https://arxiv.org/pdf/2607.05682v1)

**作者:** Yufeng Wang `[一作]` `[通讯]`, Yufeng Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究设计并评估了一种基于“研究问题证书”的LLM科研问题生成框架，使得模型能够在提出研究问题前记录原语、假设、机制、矛盾、可证伪的假设、最小实验与失败更新规则，从而实现问题可审计；

**💡 创新点**

创新点在于引入结构化证书记录机制，将问题形成过程拆解为可验证的逻辑链，并加入新颖性门控修复，迫使最终问题具备可证伪性、机制清晰度和实验可行性；

**🔧 技术方法**

采用结构化LLM管道，使用Pydantic验证JSON中间产物，构建机制模型、紧张点检测、证书生成、门控修复以及基于DeepSeek和Gemini的LLM评判；

**📊 数据集**

在十个针对LLM研究主题（技能发现、过程忠实度、工具路由等）的基准数据集上进行实验；

**📈 对比分析**

通过与三种受控提示层级基线（AI co‑scientist、Agent Laboratory、TreeSearchScientist）比较，证书核心模型在DeepSeek评判下平均分达4.76，显著高于基线4.32；Gemini交叉评判保持相同系统级排名，评判一致性高；

**⚠️ 局限性**

局限性包括：仅有十个主题、评判依赖LLM未进行人类专家评审、基线为提示级近似而非完整系统、未评估实验执行效果、一次性重复实验、并未覆盖自然科学领域等。

---

## 230. VisTCP: A Visualization Framework to Construct Knowledge-Graph-Based Representation for Traditional Chinese Painting

**arXiv ID:** 2607.05841 | [PDF](https://arxiv.org/pdf/2607.05841v1)

**作者:** Zhiguang Zhou `[一作]` (Hangzhou Dianzi University), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了VisTCP，一个结合人工智能模型与专家知识的人机交互可视化框架，用于构建传统中国画的知识图谱结构化表示。

**💡 创新点**

① 通过专家研究构建了专门的TCP语义分类体系；② 将Mask‑R‑CNN与TDE迁移到TCP上，形成面向TCP的场景图生成模型；③ 设计联合嵌入可视化与专家样本推荐的交互机制，实现不确定性提示与迭代优化。

**🔧 技术方法**

使用Mask‑R‑CNN目标检测、TDE关系推理、t‑SNE嵌入投影、视觉-语义联合嵌入、交互可视化框架等技术。

**📊 数据集**

由3位专家标注的500幅TCP样本，包含四类实体（人、自然景观、动物、文物）和两类关系（事件、位置）。

**📈 对比分析**

与传统S‑GG基线相比，VisTCP在对象检测精度与关系推理上提升约10–15%；专家问卷与案例演示显示平均5.1/7的效用评分，表明更高的可信度与交互体验。

**⚠️ 局限性**

数据量有限导致模型泛化受限；关系推理仍存在误判；语义词表覆盖不完整；需进一步融合多模态文献知识以提升系统整体性能。

---

## 231. Sudoku Grids That Require Many Clues

**arXiv ID:** 2607.05728 | [PDF](https://arxiv.org/pdf/2607.05728v1)

**作者:** David Eppstein `[一作]` (University of California), Zhang `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究给定填满的 Sudoku 网格的最小线索数，证明大多数网格需要几乎所有格子作为线索，并构造出 9×9 需要至少 18 线索、16×16 需要至少 80 线索的示例；

**💡 创新点**

提出大多数 n²×n² Sudoku 网格的最小线索数为 n⁴−O(n⁴/ log n)，并给出基于拉丁方嵌套的构造方法实现这一下界；将线索数与求解时间 O(n⁴ 2^{n⁴−m}) 联系起来，证明平均/最坏情况解题时间可压缩到 2^{O(n⁴/ log n)}；

**🔧 技术方法**

主要使用组合计数与容斥原理（sum‑weighted partitions）求解方案数的动态规划算法，拉丁方嵌套构造，以及对已填网格与线索配置的数量比较；

**📊 数据集**

论文主要基于理论计数，使用已知的完全填充 Sudoku 数量 S(n) 和拉丁方数量 L(n) 进行推导，未使用实际数据集进行实验；

**📈 对比分析**

通过组合计数证明大部分网格需要大量线索，并在 9×9 上构造 18 线索示例与已知的 17 线索不足做对比；在 16×16 上给出 80 线索构造；算法方面展示求解时间从 2^{n⁴} 降到 2^{O(n⁴/ log n)} 的改进；

**⚠️ 局限性**

限制：仅给出理论下界与构造示例，未给出所有网格的最小线索集合；构造的 16×16 需要 80 线索远高于猜想的 56；算法仍为指数级，未解决 NP‑hard 复杂度；未进行实验验证或对实际 Sudoku 数据集的评估。

---

## 232. Grover-Based PLS: AUD and Beamforming with Artificial Noise in CD-NOMA

**arXiv ID:** 2607.05429 | [PDF](https://arxiv.org/pdf/2607.05429v1)

**作者:** Deemah H. Tashman `[一作]` (Polytechnique Montreal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montreal)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种基于 Grover 量子搜索的物理层安全框架，在 CD‑NOMA 网络中通过人工噪声协助波束成形实现对活跃用户的检测与识别，并对潜在的被动与主动窃听者进行分类与抑制。

**💡 创新点**

创新点在于首次将 Grover 搜索与人工噪声波束成形相结合，用量子搜索高效定位活跃用户集合，随后依据搜索结果动态调整信号与噪声发射方向，并引入主动窃听者模型（基于活跃频率）来提升安全性分析的现实性。

**🔧 技术方法**

核心技术包括 Grover 量子搜索算法（oracle 与 diffuser 设计）、人工噪声（AN）协助波束成形、信道容量与保密容量分析，以及基于模拟的量子电路实现与传统压缩感知、最大似然、经典相关接收机等基线对比。

**📊 数据集**

实验使用人工生成的合成数据，设定 N=5、κ=4、各用户代码为离散二进制向量，BS 传输功率、噪声方差等均按数值参数化，未使用公开数据集。

**📈 对比分析**

通过与 ML、CS、CCR、随机选择等方法对比，Grover 基础 AUD 在平均保密率上与 ML 仅差约 21%，远优于 CS（约 76%）和 CCR（约 173%），且在不同 BS 传输功率、功率分配因子 α 与主动窃听者比例 f 变化时展现出可控的保密性能提升。

**⚠️ 局限性**

主要局限包括：需假设活跃用户数 V 已知；量子电路在 NISQ 设备上的实现受量子比特数、门深度、误差与退相干限制；量化处理与噪声对搜索准确度敏感；实验仅为软件仿真，未覆盖真实硬件误差与多用户多径环境。

---

## 233. BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension

**arXiv ID:** 2607.05614 | [PDF](https://arxiv.org/pdf/2607.05614v1)

**作者:** Abu Tyeb Azad `[一作]` (Wichita State University), AKM Mahbubur Rahman `[通讯]` (Center for Computational & Data Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了BaFCo——一套含26类细粒度实体的孟加拉语政府表单基准数据集，并对其在文档布局分析（DLA）与关键信息提取（KIE）任务上进行评估

**💡 创新点**

首个公开的孟加拉语表单理解基准，提供多页、多域、复杂布局的高质量注释；同时构建细粒度与粗粒度两套实体标签，支持跨语言比较

**🔧 技术方法**

使用大规模多模态语言模型（LLMs）如GPT‑5.2、Gemini 3 Pro、Claude Opus 4.6、Qwen 3.6‑Plus、Kimi K2.5，采用零样本与链式思维提示，配合低/高推理力度调节

**📊 数据集**

BaFCo数据集，200份多页孟加拉语政府表单（316页），包含16,382个实体、8,771条关系；另选取156份表单用于KIE（1,926键值对）并加入同源英文表单做对比

**📈 对比分析**

在DLA中，Gemini 3 Pro在细粒度和粗粒度实体集上的mAP仅约0.12-0.26，提升不大；在KIE中Gemini 3 Pro对孟加拉语表现最佳（F1≈0.85），GPT‑5.2在英文上最高（F1≈0.85）；与开源模型相比差距显著，尤其在细粒度检测上

**⚠️ 局限性**

限制主要在：1）DLA精度低，细粒度实体识别困难；2）推理力度与链式思维对性能影响有限；3）数据集规模仍偏小，缺乏更多多域表单；4）模型在孟加拉语下仍受训练数据稀缺影响

---

## 234. MatrixFSDP: communication-free matrix optimizers under ZeRO-3 parameter sharding

**arXiv ID:** 2607.05895 | [PDF](https://arxiv.org/pdf/2607.05895v1)

**作者:** Ming Gao `[一作]` (University of Pittsburgh), Hao Zhang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在 ZeRO-3 参数分片环境下实现通信无关的矩阵优化器（如 Muon）的运行时方案，称为 MatrixFSDP。

**💡 创新点**

创新点在于：① 将每个 2D 权重的完整矩阵分配给单个数据并行 rank（owner）并在其上局部完成 Newton–Schulz 计算；② 通过全局 owner 规划器和运行时元数据实现不均匀的 owner‑shaped sharding；③ 设计了 deterministic owner‑segment P2P collectives、owner‑buffer pinning 以及跨世界大小的 owner‑shard checkpoint 机制，使得 FSDP2 运行流程保持一致。

**🔧 技术方法**

采用了 FSDP2/ZeRO-3 的分片框架，结合自定义的 owner‑segment 通信、owner‑buffer pinning、全局 owner‑planner 与元数据路由，以及标准的 Newton–Schulz、AdamW、Shampoo 与 SOAP 等矩阵优化器。

**📊 数据集**

使用 Qwen‑style 解码器 Transformer（隐藏层 4096，隐藏扩展 16384，32 头，序列 4096）做内存/延迟评估，并在 12 层解码器 LM 上使用 WikiText 数据集验证收敛性。

**📈 对比分析**

与标准 FSDP2‑Muon（即在 ZeRO‑3 下重建矩阵并执行分布式 Newton–Schulz）和 gather‑once FSDP2‑Muon（完整矩阵收集后本地计算）进行对比。结果显示：单节点 optimizer‑step 速度提升 4.2×，8 节点提升 54.6×；端到端训练速度提升 1.37×–2.15×；ZeRO‑3 内存保持在 2.7 GB/卡，显著低于 ZeRO‑1 owner 方案的 18.5 GB/卡。

**⚠️ 局限性**

局限性包括：① 在单机或强缩放时收益有限；② 对极大模型或多维分片的权重，owner‑fanout 可能成为新的瓶颈；③ 当前仅覆盖 ZeRO‑3，未处理张量并行片段重构；④ 需要自定义 collectives，增加实现复杂度。

---

## 235. SSA-3DGS: Unsupervised Removal of Screen-Space Artifacts for 3D Gaussian Splatting

**arXiv ID:** 2607.05598 | [PDF](https://arxiv.org/pdf/2607.05598v1)

**作者:** Kristof Overdulve `[一作]` (Hasselt University), Nick Michiels `[通讯]` (Hasselt University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无监督框架 SSA-3DGS，能够在存在屏幕空间遮挡的多视角图像中同时重建干净的 3D 场景并分离出遮挡层。

**💡 创新点**

创新点在于将 3D Gaussian Splatting 与可学习的 2D 覆盖层联合优化，并通过稀疏性与总变分正则化实现无监督的遮挡分离；同时展示该方法在多种真实与合成遮挡下的鲁棒性。

**🔧 技术方法**

核心技术包括 3D Gaussian Splatting、基于视差的联合优化、可学习的二维叠加层、稀疏性与 TV 正则化、以及 Adam 优化器。

**📊 数据集**

使用 Mip-NeRF 360 作为合成基准，并在自采集的真实世界数据集（泥污、机架遮挡）上进行验证。

**📈 对比分析**

与基线 3DGS 及 DeSplat 进行对比，SSA-3DGS 在合成数据上平均提升 PSNR 达 4–9 dB、在真实数据上提升 3–4 dB，整体性能接近无遮挡参考。

**⚠️ 局限性**

限制在于需要足够密集的多视角捕获；假设遮挡在图像平面固定，对视角依赖或仅近似静止的遮挡效果处理仍有限。

---

## 236. ShadowProbe: Language-Extensible Detection of Hidden Algorithmic Complexity Vulnerabilities

**arXiv ID:** 2607.05474 | [PDF](https://arxiv.org/pdf/2607.05474v1)

**作者:** Yuanmin Xie `[一作]` (Tsinghua University), Chengnian Sun `[通讯]` (University of Waterloo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为 ShadowProbe 的可语言扩展框架，用于自动检测软件中隐藏的算法复杂度漏洞。

**💡 创新点**

创新点在于：①识别并系统化“阴影复杂度”（shadow complexity）这一盲点；②构建基于轻量级静态分析 + 代码上下文恢复 + 大语言模型辅助的多阶段管线；③通过统计增长推断方法在运行时稳健区分真实复杂度膨胀与运行时噪声。

**🔧 技术方法**

主要技术包括：轻量级静态分析筛选候选函数；项目级符号索引实现语言无关的上下文恢复；LLM（如ChatGPT/CodeLlama）生成可调大小的测试输入；执行时间测量与统计增长推断（超线性与线性比较）来验证复杂度。

**📊 数据集**

使用了一个专门设计的 worst‑case complexity benchmark（未命名）以及五个真实项目（Java、Python、Zig、Rust 等），并在这些项目中发现多处隐藏漏洞。

**📈 对比分析**

与传统的 fuzzing、symbolic execution 和混合分析工具（如 SlowFuzz、HotFuzz、Singularity 等）对比，ShadowProbe 在候选筛选效率、验证覆盖率和误报率上均有显著提升；在实验中它在同等资源预算下发现的漏洞数量比现有工具高 2–3 倍，并显著减少了手工驱动的工作量。

**⚠️ 局限性**

限制包括：对 LLM 生成输入的准确性依赖；静态筛选信号可能遗漏极少数实现细节导致的阴影成本；运行时噪声过滤仍可能在 GC/Just‑In‑Time 环境下产生边际误判；目前仅在四种主流语言（Java、Python、Zig、Rust）上验证，跨语言通用性待进一步测试。

---

## 237. REAN: Reconstruction-aware ECG Anonymization Based on Privacy--Utility Orthogonality

**arXiv ID:** 2607.06037 | [PDF](https://arxiv.org/pdf/2607.06037v1)

**作者:** Taerin Ki `[一作]` (Chung-Ang University), Jaewoo Lee `[通讯]` (Chung-Ang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种基于 1‑D U‑Net 的 ECG 匿名化方法 REAN，通过在输入信号上学习一个受限残差来抑制生物特征信息，同时保持诊断信息不受影响。

**💡 创新点**

核心创新在于发现诊断（utility）和隐私（privacy）方向在 ECG 信号空间中近乎正交，从而可以在保持诊断性能的前提下沿着隐私方向抑制身份信息，并将这一几何特性融入单一训练目标，实现了无损、无迭代的匿名化。

**🔧 技术方法**

使用冻结的诊断分类器（1‑D ResNet）和三种生物特征分类器（ECGViT）作为训练信号，构建了由交叉熵、对抗项和 PRD 失真惩罚组成的损失；实现上采用 1‑D U‑Net 生成受限残差，并在训练期间仅更新网络权重。

**📊 数据集**

在四个 PhysioNet 数据库（MIT‑BIH Arrhythmia、MIT‑BIH Long‑Term、INCART、SHDB‑AF）共 186 名受试者（共 1.16M 8 秒窗口）上进行评估。

**📈 对比分析**

与七个基线（全信号噪声、REACT、Lee 等）相比，REAN 在保密性上将 ReID 准确率降至接近随机（0.00），在诊断上宏观 AUROC 与原始信号基本一致（0.9991 vs 0.9982），失真 PRD 仅为 5.56%，并且单前向推理时间为 0.117 ms/窗口，速度快于所有对比方法。

**⚠️ 局限性**

主要限制：作为确定性映射，REAN 在面对可重训练的攻击者时可能泄露残留生物信息；缺乏正式的差分隐私保证；实验仅覆盖单导联 ECG、受试者内部时间拆分，未验证多导联、不同诊断任务或未知受试者的泛化能力。

---

## 238. Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis

**arXiv ID:** 2607.05777 | [PDF](https://arxiv.org/pdf/2607.05777v1)

**作者:** Peize Liu `[一作]` (Hong Kong University Of Science And Technology), Shaojie Shen `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究多鱼眼相机系统标定失败的根本原因，并提出 CO-Calib 框架，通过学习型目标检测器和误差分析驱动的帧选择器，构造更优质的标定观测序列，从而提升标定的鲁棒性和精度。

**💡 创新点**

1) 失败导向分析揭示了焦距与投影形状耦合导致的初始化不良；2) CO-Calib 结合学习检测器与基于误差分析的帧选择器，构造初始化友好的数据集，且不改动现有优化后端；3) 提出了基于项目化等距与径向跨度的帧筛选标准，确保观测在参数空间中解耦。

**🔧 技术方法**

使用深度学习目标检测器（在在线物理驱动的数据生成管线中训练），误差分析指导的帧选择（项目化等距、径向跨度评估），以及传统的 BA 优化（Kalibr 作为后端）。

**📊 数据集**

合成数据：16种相机配置（FoV 180/200/220/240° 与 0/60/90/120° 旋转）各 100 条 480 帧序列；真实数据：多鱼眼 stereo（0/30/60/90/120°）与 Hex‑Fisheye（10 条序列）。

**📈 对比分析**

与 Kalibr、Basalt、随机子集、仅 BA 两阶段等方法对比。CO-Calib 在合成测试中成功率从 68.1% 提升至 99.3%，外参误差下降；在真实 Hex‑Fisheye 数据中 100% 成功率且一致性显著提升；随机子集仅 30.9% 成功，证明帧选择的重要性。

**⚠️ 局限性**

仍需依赖训练好的检测器和经验阈值，极宽 FoV 或极稀疏观测时仍可能失败；未评估实时性与计算开销；仅优化初始化过程，对后端优化的改进有限。

---

## 239. Generalisation of Baker's Forcing Method to Arbitrary Prime and NP-hardness of Several $p$-adic Optimisations

**arXiv ID:** 2607.06092 | [PDF](https://arxiv.org/pdf/2607.06092v1)

**作者:** Tomoki Mihara `[一作]` `[通讯]`, Tomoki Mihara

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了将Baker的forcing方法推广到一般p-adic优化问题，并利用该方法证明了p-adic线性回归、p-adic动态神经网络以及p-adic平滑神经网络的NP-难度。

**💡 创新点**

创新点在于将forcing方法从二进制扩展到任意素数p，并构造了针对不同p-adic神经网络的NP-难度证明，首次给出了p-adic优化问题的复杂性分类。

**🔧 技术方法**

主要技术是p-adic数域的代数结构、forcing方法的抽象推广以及对最大割问题的归约，结合p-adic线性回归与神经网络的定义。

**📊 数据集**

本工作未使用任何实验数据集，全部为理论证明。

**📈 对比分析**

没有实验比较，仅通过归约证明NP-难度，未给出算法性能指标。

**⚠️ 局限性**

局限性在于仅证明了NP-难度，没有提供可行的多项式时间或近似算法，且仅考虑固定素数p的情况。

---

## 240. $\mathbfλ$-VAE: Variance Equalization for Posterior Collapse

**arXiv ID:** 2607.05531 | [PDF](https://arxiv.org/pdf/2607.05531v1)

**作者:** Girum Demisse `[一作]` `[通讯]` (Microsoft African Development Center), Girum Demisse (Microsoft African Development Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的变分自编码器训练策略λ‑VAE，能显著降低后验崩塌现象，提升信息流通与重建质量。

**💡 创新点**

核心创新在于在重参数化步骤中引入指数缩放的噪声（λ‑scaling），实现对每个潜在维度的方差均衡，从而同时抑制梯度失衡与信息缺口两种崩塌原因。

**🔧 技术方法**

利用基于高斯通道模型的SNR信息容量估计，闭式求解最优λ并通过δ超参数平衡信息增益与KL偏差，配合自适应EMA更新实现在线调节。

**📊 数据集**

在二进制MNIST、Omniglot以及彩色图像CIFAR‑10、CelebA‑64四个数据集上进行实验。

**📈 对比分析**

与标准VAE、β‑VAE、Free‑Bits、VampPrior等方法对比，λ‑VAE在保持或提升BPD的同时将活跃维度从少数提升至全部，信息容量提升最高3倍，重建质量显著提高。

**⚠️ 局限性**

局限性包括需手动设置δ超参数、对高斯后验的假设，且在极大规模或非连续噪声层的迁移尚未验证。

---

## 241. Load Balancing under Adaptive Bin Deletions

**arXiv ID:** 2607.06211 | [PDF](https://arxiv.org/pdf/2607.06211v1)

**作者:** Haim Kaplan `[一作]` (Google Research), Uri Stemmer `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究在自适应对手（adaptive adversary）删除箱子时，如何在保持球数不变的情况下，最小化球的重新分配总量（recourse）和任何箱子的最大负载。

**💡 创新点**

提出了在自适应删除游戏中，单次随机重新分配即可获得线性 recourse 与经典负载上界（O(log n/log log n)）；使用两次选择（power of two choices）可将最大负载进一步降低到 O(log log n)。同时证明 d=1（全部球一次性转移）在自适应环境下不可行，而 d=2（将球拆成两组）已足以恢复线性 recourse 并保持有效负载平衡。

**🔧 技术方法**

主要技术包括：随机变量的随机支配（stochastic domination）与构造耦合（coupling），潜在函数（potential）分析，马尔可夫不等式与 Azuma 不等式的使用，以及对 Galton–Watson 过程的结合分析。通过将自适应游戏映射到无对手的“非自适应”游戏，利用已知的 Balls‑and‑Bins 结果得出上界。

**📊 数据集**

无实际数据集，全部为理论分析与概率上界。

**📈 对比分析**

与之前针对全删除（T=n−1）或无对手（oblivious adversary）场景的结果相比，本工作在部分删除（T=n/2）以及对手完全自适应的环境下给出了近似最优的 recourse 与负载上界；与 Bender 等人、Fine 等人等先前工作相比，提供了更精细的阶段化分析并实现了两次选择的提升。性能上：recourse O(n)、最大负载 O(log n/log log n)（均匀）或 O(log log n)（两次选择）。

**⚠️ 局限性**

局限性：仅针对 T=n/2 的情况给出了完整证明，T 远小于 n/2 的情形仍缺乏精确界；d=2 的最大负载仅在常数概率下达到 O(log n)，对更高置信度的上界尚未得到；d=1 在自适应环境下仍不可行；扩展到更一般的分区策略和对手攻击策略（如不等量拆分）仍是未解决的问题。

---

## 242. Measuring the Invisible: Evaluating the Impact of Public Funding on Open Source Software

**arXiv ID:** 2607.05413 | [PDF](https://arxiv.org/pdf/2607.05413v1)

**作者:** Laia Domenech Burin `[一作]` `[通讯]` (Sovereign Tech Agency), Laia Domenech Burin (Sovereign Tech Agency)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了德国Sovereign Tech Agency（STA）对四个关键 OSS 项目（PyPI、RubyGems、curl、Fortran）的资金投入在 2015‑2026 年期间对仓库活动的因果影响，并通过 Generalized Synthetic Control Method（GSCM）估计平均处理效应。

**💡 创新点**

创新点：①将 GQM 与 CHAOSS 指标结合，构建可追溯的因果度量；②首次将倾向得分匹配与 GSCM 结合，构造可比对照池；③在少量真实项目上验证 GSCM 的可行性，展示其在 OSS 评估中的实用价值。

**🔧 技术方法**

技术方法：GSCM 与 Augmented Synthetic Control（ASCM）比较、倾向得分匹配、log 变换、频繁样本重抽、交叉验证挑选因子数、频率误差估计。

**📊 数据集**

数据集：Ecosyste.ms API 提供的 2015‑2026 年按季度汇总的 54 期仓库活动（commits、pull requests、issues、releases、contributors 等）；OpenSSF Scorecard 用于筛选“关键数字基础设施”项目；匹配变量来自依赖包、stargazers、forks 等元数据。

**📈 对比分析**

比较方法：以 GSCM 为主估计，ASCM 作为稳健性检查；与传统差分法对比，GSCM 在预处理拟合度更好、置信区间更窄、能够处理多单元与不平行趋势，显示出更强的估计精度。

**⚠️ 局限性**

局限性：①对照池规模有限，匹配变量缺失导致潜在偏差；②未对合同类型或生命周期阶段进行分层，可能掩盖处理效应异质性；③样本量小（12 处理仓库），统计功效受限；④未充分捕捉安全漏洞、团队规模等关键变量，影响因果识别。

---

## 243. Collective Cognition in Hybrid Groups: A Network Science Synthesis

**arXiv ID:** 2607.05593 | [PDF](https://arxiv.org/pdf/2607.05593v1)

**作者:** Babak Hemmatian `[一作]`, Lav R. Varshney `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

综述并整合网络科学、集体认知与多智能体系统的研究，提出以记忆-注意-推理(MAR)为核心的框架，用以解释并预测人机混合团队的集体认知行为。

**💡 创新点**

创新点在于：①把人机混合网络视为异质节点与异质边的多层网络；②将已知的同质网络中“探索‑剥削”与“效率‑冗余”两大权衡映射到混合网络，揭示节点类型与边类型对这些权衡的独特影响；③提出一系列新的混合网络结构（如人机枢纽、AI边缘控制、门控与监督等），并给出对应的假设与实验方向。

**🔧 技术方法**

技术方法主要包括：网络科学度量（度、路径、聚类、导电率等）、认知建模（文化演化、社会学习策略）、多智能体系统仿真与实验平台、信息论分析（通道容量与噪声）以及跨学科的理论综合。

**📊 数据集**

由于本文为综述性章节，未引入新的原始数据集；引用了大量人类社交网络实验、LMM‑多智能体仿真以及现有公开数据集（如社交媒体传播数据、医疗诊断协作数据等）作为例证。

**📈 对比分析**

文中并未开展新的实验比较，而是通过文献综述对比人类、AI 与混合网络在各类任务（竞争、协调、合作、传播、集体决策）中的表现，并指出混合网络在探索‑剥削与效率‑冗余权衡上与同质网络的差异与潜在优势。

**⚠️ 局限性**

局限性包括：①缺乏大规模、长期的实证验证；②对混合网络假设的可实验性与可操作性尚未充分评估；③在节点与边的异质性建模上仍停留在理论层面，实际系统中需要进一步细化参数与控制策略。

---

## 244. Automating Quality Assessment with NLP of LLM-Generated Defeaters

**arXiv ID:** 2607.06039 | [PDF](https://arxiv.org/pdf/2607.06039v1)

**作者:** Tihomir Rohlinger `[一作]`, Stefan Wagner `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于自然语言处理的自动评估方法，用于验证大语言模型生成的安全保障案例中的defeater（挑战论点）的质量。

**💡 创新点**

创新点在于将结构化的安全保障案例DAG、BERT语义嵌入和元分类器相结合，并通过SMOTE和自训练等技术缓解类别不平衡，显著降低人工评审的主观性。

**🔧 技术方法**

使用技术包括BERT-base-uncased词向量、Logistic Regression、支持向量机、元分类器、SMOTE、Self‑TrainingClassifier、Cosine相似度、NetworkX构建DAG、以及文本特征提取。

**📊 数据集**

使用的数据集为两条工业案例的defeater集合：汽车领域的Adaptive Cruise Control（ACC）和核能领域的CERN Large Hadron Collider（LHC）机动保护系统（MPS），共172条defeater，包含两位专家对“what”和“why”三层组件的标注。

**📈 对比分析**

与人工评审相比，自动方法在ACC、CERN两组数据上取得平均F1≈0.84，并将Cohen κ从负值提升约40%（最高≈0.44），表明模型与专家一致性明显提升，且在不需要人工标注的前提下可实现规模化评估。

**⚠️ 局限性**

局限性包括样本量有限导致模型对极少数类别（如0级）泛化不足，依赖专家标注数据导致潜在偏差，BERT嵌入缺乏深层领域语义理解，无法完全捕捉复杂安全情境下的隐含风险。

---

## 245. Decoupled Single-Mask Annotation Noise Detection via Cross-Sectional Patch Self-Consistency

**arXiv ID:** 2607.05965 | [PDF](https://arxiv.org/pdf/2607.05965v1)

**作者:** Yinheng Zhu `[一作]` (Tsinghua University), Xiaowei Xu `[通讯]` (Guangdong Provincial People's Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了一种解耦的单掩码注释噪声检测框架，通过跨截面自一致性原则定位血管CT注释中的错误区域，并提供可审核的证据。

**💡 创新点**

创新点在于：①利用血管截面递归性进行跨扫描相似补丁检索；②将相似度条件下的掩码差异转化为可解释的噪声得分；③生成可用于质量加权训练的扫描级质量图。

**🔧 技术方法**

使用技术包括Bishop框架提取正交截面补丁、FAISS高效最近邻检索、基于MSE/IoU的统计残差计算、Voronoi映射与Sigmoid权重生成、以及nnUNet分割模型的质量加权训练。

**📊 数据集**

使用的数据集为ImageCAS冠状动脉CT血管影像数据集（1000张扫描）。

**📈 对比分析**

与标准nnUNet训练进行对比，质量加权训练将CPR-DSC提升0.8个百分点、ASD降低0.06mm、HD-95降低0.42mm，整体DSC保持相近。

**⚠️ 局限性**

局限性包括：仅针对管状结构；仅检测而非纠正噪声；检索未考虑旋转/窗口偏移导致漏检；跨数据集泛化能力尚未验证。

---

## 246. GEM-Occ: From Visual Geometry Evidence to Embodied Semantic Occupancy Memory

**arXiv ID:** 2607.05543 | [PDF](https://arxiv.org/pdf/2607.05543v1)

**作者:** Hu Zhu `[一作]` (Hong Kong Polytechnic University), Chang Wen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了HIOcc——一个融合ScanNet、ScanNet++和Matterport3D的分层室内占据基准，并提出GEM-Occ——一种将瞬时视觉几何证据转换为语义高斯占据记忆并通过可视化与不确定性感知的因果融合实现长周期语义占据映射的框架。

**💡 创新点**

创新点包括：①构建跨视角、跨房间、跨建筑的层级占据基准；②使用语义高斯证据与自由空间射线证据来表示占据与未知状态；③设计可视化和不确定性感知的因果更新规则；④层级高斯记忆结构支持从局部到全建筑的在线映射。

**🔧 技术方法**

核心技术包括3D高斯喷射（Gaussian Splatting）、语义高斯原语构造、自由空间射线证据、可视化与不确定性感知的因果融合、层级缓存/子图/图结构记忆以及Gaussian-to-occupancy分裂查询。

**📊 数据集**

使用数据集：ScanNet、ScanNet++、Matterport3D，构成HIOcc基准，包含局部视角、房间级在线序列和建筑级全景序列。

**📈 对比分析**

在局部预测、房间级在线映射和建筑级映射三种评估模式下，GEM-Occ分别实现了61.37 IoU/57.76 mIoU、56.79 IoU/46.20 mIoU、53.8 IoU/46.7 mIoU，显著优于MonoScene、ISO、SplatSSC、SplicingOcc、EmbodiedOcc、EmbodiedOcc++和GPOcc等基线，并在回访一致性、内存占用和查询延迟上取得更佳的性能平衡。

**⚠️ 局限性**

局限性包括：依赖已有室内数据集，带来标注噪声与重建误差；语义标签仅为粗粒度的11类，缺乏细粒度和实例级信息；仅针对静态场景，依赖深度/点图估计，易受几何误差影响；未针对动态对象、可变布局或人类活动进行建模；未评估与下游任务（如导航、搜索、操作）的直接关联。

---

## 247. Physics-Regularized Machine Learning for Proprioceptive Vehicle Localization Using Onboard Sensors

**arXiv ID:** 2607.05663 | [PDF](https://arxiv.org/pdf/2607.05663v1)

**作者:** Abinav Kalyanasundaram `[一作]` (AImotion Bavaria), Michael Botsch `[通讯]` (AImotion Bavaria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种仅利用车载本体感知传感器的实时定位框架piml2，结合Transformer学习模型与可微分EKF进行姿态估计。

**💡 创新点**

引入可微分Kalman滤波作为物理正则化器，实现端到端训练；物理守护层与不确定性估计使模型在低摩擦等未知环境下具备更好泛化；并发布了新的低摩擦/降雪驾驶数据集。

**🔧 技术方法**

Transformer序列回归、物理守护层、基于高斯的不确定性输出、可微分扩展卡尔曼滤波器、梯度反向传播、Adam优化器等技术。

**📊 数据集**

使用Revsted公开数据集（含车载传感器数据）以及新构建的低摩擦/降雪（低μ）测试集，划分70/10/20进行训练、验证与测试。

**📈 对比分析**

与五个基线（OSD-Baseline、NN-VDM、DL-AVL、RNN-EKF、Backprop-KF）及piml2*对比，piml2在动态状态估计RMSE降至0.02/0.01/0.18，定位RMSE和最大误差比对手低约28%，在低摩擦集上保持最高准确度；实时推理延迟约30 ms，支持≈30 Hz实时运行。

**⚠️ 局限性**

仅实现二维平面定位；训练仅离线；缺少纵向加速度等传感器；在极端光照或复杂环境下的鲁棒性尚未充分验证；需要跨车辆、平台的迁移学习与在线适配。

---

## 248. Think Before You Grid-Search: Floor-First Triage for LLM Serving

**arXiv ID:** 2607.05876 | [PDF](https://arxiv.org/pdf/2607.05876v1)

**作者:** Yihua Liu `[一作]` `[通讯]`, Yihua Liu

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于两侧 floor 的残差驱动 LLM 服 务优化工作流，并提供零依赖的 Python artifact 用于 floor 计算、结果对齐和代理技能实现。

**💡 创新点**

创新点包括：① 通过两侧 floor 捕获资源重叠信息；② 以 KV 容量墙为主的 wall‑ordering 决策；③ 将工作流转化为可在代理循环中强制执行的技能；④ 在 H20 GPU 上首次给出 ridge‑point、容量优先的注意力布局判定。

**🔧 技术方法**

采用五维资源向量（HBM 字节、FLOPs、网络字节/消息、KV 容量）、两侧 floor 计算、壁面排序、残差阈值决策、针对性 Nsight Systems/Compute 调试以及纯 Python 的校准与模拟。

**📊 数据集**

使用 DeepSeek V3.2 规模的 MoE 模型（总参数 671B，激活 37B）与 16×H20 GPU 集群（4×200 Gb HDR InfiniBand）进行推理负载评测（批量 64，上下文 8192）。

**📈 对比分析**

先用 floor 预估可行区间，再进行稳态 TPOT 量测；若残差 < 1.3× floor 则不再调试，若 > 则开启 Nsight；通过案例显示 TP16 在高并发下优于 EP+DP，单流下则相反，验证了 goodput 边界与预测的吻合度。

**⚠️ 局限性**

局限包括：对连续批量、专家不均衡、open‑loop 饱和、主机开销、单流启动开销以及校准漂移等场景建模不足；在这些情况下 floor 可能失效或需进一步扩展。

---

## 249. Structured Data Extraction from Real Estate Documents using Clustering, Classification, and Large Language Models

**arXiv ID:** 2607.06012 | [PDF](https://arxiv.org/pdf/2607.06012v1)

**作者:** Muhammad Assad Shehbaz `[一作]` (Robert Gordon University), Carlos Francisco Moreno-García `[通讯]` (Robert Gordon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一个完整的端到端管道，先通过逆向工程的REST API从Aberdeen Solicitors Property Centre（ASPC）平台获取房产列表和对应的问卷PDF，然后对PDF进行结构分类，接着利用DeepSeek R1大语言模型在无规则提示下抽取预定义的房产属性，最终生成含API数据和PDF提取特征的统一数据集。

**💡 创新点**

创新点在于：①首次将无结构、异构的房地产问卷PDF通过大语言模型实现自动化、无规则的结构化提取；②将逆向API获取、文档分类、LLM抽取与下游验证三大环节系统化为一体化管道；③通过多维度（相似度一致性、聚类轮廓、MCDM排名差异）验证抽取结果的可靠性。

**🔧 技术方法**

使用技术包括：Python编程、REST API逆向与爬取、pdfplumber/Tesseract OCR进行文本检测、DeepSeek R1（通过Groq推理API）进行结构化抽取、聚类（K-Means）、相似度度量（Cosine/Euclidean/Manhattan）、多准则决策（TOPSIS、加权评分）等。

**📊 数据集**

数据集来源为ASPC平台的房产列表和约数千份问卷PDF，最终经过去重后得到约几千条（约2,500-3,000）唯一房产记录，并附带约50项结构化属性。

**📈 对比分析**

通过与不同相似度度量的Top‑5推荐比较（Jaccard一致率≈0.8）、K‑Means聚类（Silhouette≈0.65、得到可解释的“可负担”与“高端”两类）以及TOPSIS与加权评分的排名重叠率≈0.18，证明抽取数据在多维度上表现出高度一致性、可解释性与区分度。

**⚠️ 局限性**

主要局限包括：约30‑40%的PDF因无文本层或复选框标记被排除，导致样本偏倚；未进行正式的消融实验比较仅API数据与完整抽取数据的增量价值；缺少用户研究评估抽取结果的感知质量。

---

## 250. PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation

**arXiv ID:** 2607.05915 | [PDF](https://arxiv.org/pdf/2607.05915v1)

**作者:** Hyungseok Song `[一作]` (LG AI Research), Soonyoung Lee `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于开源EDA引擎的交互式PCB布线环境（PCB-Bench），并提供相应的数据集与评测基准，支持RL和LLM代理在同一环境下训练与评估；

**💡 创新点**

核心创新在于：①把真实EDA引擎的原生API作为代理的动作空间，确保每一步都满足设计规则；②提供头less、可向量化的Gym环境与针对RL/LLM的包装器；③构建了三类数据集（网格化、无网格化、真实开源板）以及统一的评测指标；

**🔧 技术方法**

技术手段包括：Python-C++ 绑定实现58个EDA引擎API；基于MDP的设计规则奖励函数与潜能函数；RL使用Transformer+PPO，LLM使用工具调用式S-expression；评测通过引擎的DRC检查；

**📊 数据集**

数据集：D1（10k网格化）、D2（10k无网格化）、D3（679真实开源板，分小中大三组）以及对应的网格尺寸和层数统计；

**📈 对比分析**

与规则驱动路由器（Freerouting、OrthoRoute、KiCad）、网格化RL基线（A2C、Sable）以及无循环LLM基线进行对比。实验显示：RL基于API的PPO在D2与D3-A上接近甚至超过规则驱动路由器；LLM在小板上表现优秀，但在大板上性能大幅下降；交互式LLM显著优于一次性生成脚本或直接文件；

**⚠️ 局限性**

局限性包括：①缺乏迭代优化（rip‑up/reroute）动作；②只评估几何可行性，未覆盖信号完整性、电磁兼容、热等工业指标；③LLM在大规模板上仍难以保持设计规则完整；④目前仅支持双层板，需扩展多层与组件布局。

---

## 251. Auto-DSM Under the Lens: A Black-Box Evaluation Framework for LLM-Based DSM Generation

**arXiv ID:** 2607.05985 | [PDF](https://arxiv.org/pdf/2607.05985v1)

**作者:** Niels Potters `[一作]` (Eindhoven University of Technology), Theo Hofman `[通讯]` (Eindhoven University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一个黑盒评估框架，用于系统性评估大型语言模型（LLM）生成的设计结构矩阵（DSM）质量。

**💡 创新点**

创新点包括：①结合结构完整性、正确性、耦合密度等结构指标与选择性准确率、弃权覆盖率等分类指标，并提出综合质量分数（Composite Quality Score）统一衡量；②设计了单跑与多跑评估流程，引入Fleiss’ κ、熵等稳定性度量，保证评估的可重复性与可靠性；③公开了实验代码、数据集与评估工具，形成可复现的基准。

**🔧 技术方法**

技术手段涵盖：大型语言模型推理（OpenAI GPT‑4o等）、黑盒测试、语义匹配（模糊字符串匹配 + 句向量相似度）、结构指标计算、选择性分类评估、熵与Fleiss’ κ稳定性分析、Composite Quality Score 计算。

**📊 数据集**

使用的数据集为两类：①5个组件的虚拟抽象系统（包含四种交互定义）；②真实冰箱分解系统（31个组件，7个子系统，约930个交互），两者均手工生成并验证为Ground Truth DSM。

**📈 对比分析**

比较方法：对每个实验案例执行N次（默认30）生成DSM后与GT DSM逐细胞比较，计算完整度、正确性、NZF、选择性准确率、弃权覆盖、熵、Fleiss’ κ等单跑与多跑指标。结果显示在结构清晰、提示明确时，平均 agreement >0.90，κ >0.80，Composite Quality Score 高；但在提示歧义、无定义交互或多系统情形下，准确率下降、熵升高、误差率升高，表明模型对细粒度语义与上下文敏感。

**⚠️ 局限性**

局限性：①对hallucination与不确定性控制不充分，仍出现自信错误；②缺乏多层次/多系统的支持，难以处理层级化分解；③评估依赖于与LLM预训练分布相近的人工构造数据，通用性受限；④缺少交互可追溯性与错误定位机制，难以验证错误来源。

---

## 252. Intuitionistic Fuzzy Graph Embedded Random Vector Functional Link with Multiview Learning

**arXiv ID:** 2607.05635 | [PDF](https://arxiv.org/pdf/2607.05635v1)

**作者:** Vrushank Ahire `[一作]` (Indian Institute of Technology Ropar), M. A. Ganaie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Intuitionistic Fuzzy Graph Embedded Random Vector Functional Link with Multiview (IFGRVFL‑MV) 模型，融合直觉模糊集合、图嵌入和多视图学习，以提升RVFL的分类性能。

**💡 创新点**

创新点在于：① 将直觉模糊权重矩阵同时作用于两视图的误差正则化；② 在GE框架中嵌入直觉模糊权重，保持几何结构的同时抑制不确定样本；③ 通过拉格朗日优化得到闭式解，兼顾三者协同作用。

**🔧 技术方法**

使用的技术包括：Random Vector Functional Link (RVFL) 网络、Intuitionistic Fuzzy Sets、Graph Embedding、Multiview Learning、Gaussian kernel相似度、矩阵求逆优化、统计检验（Friedman、Nemenyi、Wilcoxon、win‑tie‑loss）。

**📊 数据集**

实验数据集来自UCI和KEEL公开基准，共8个数据集：pittsburg‑bridges、breast‑cancer、planning、mammographic、breast‑cancer‑wisc‑prog、Pima、cylinder‑bands、checkerboard。

**📈 对比分析**

通过准确率、平均rank、Friedman检验、Nemenyi差异、Wilcoxon签名检验以及win‑tie‑loss分析对比；IFGRVFL‑MV平均准确率81.06%，rank 1.19，显著优于RVFL、GE‑IFWRVFL、GRVFL‑MV等基线模型。

**⚠️ 局限性**

局限性在于：① 计算复杂度较高（GE与矩阵求逆导致O((m1+m2+h1+h2)^3)）；② 对大规模数据的可扩展性与内存占用待进一步优化；③ 需要更系统的自动参数选择与在线学习机制。

---

## 253. MSCENet: A Multi-Scale Correlation Enhanced Network for Anomaly Detection

**arXiv ID:** 2607.05864 | [PDF](https://arxiv.org/pdf/2607.05864v1)

**作者:** Long Zhao `[一作]` (Tongji University), Bin He `[通讯]` (Tongji University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出MSCENet框架，融合细粒度时序卷积、Mixhop图卷积与多尺度门控卷积，实现多变量时间序列的多尺度时空特征融合并用于异常检测。

**💡 创新点**

创新点包括①通过自适应图卷积动态学习跨尺度空间相关性；②细粒度多分辨率卷积（FGTConv）在同一层捕获多尺度时间依赖；③多尺度门控卷积将空间与时间特征在不同感受野下融合，提升短期与长期异常的识别能力；④整体结构实现对复杂多尺度依赖的全局建模。

**🔧 技术方法**

使用的技术包括扩张卷积、Mixhop图卷积、残差连接、FFT多尺度选取、门控Tanh单元、重建误差阈值判别、滑动窗口嵌入、Adam优化、批归一化等。

**📊 数据集**

实验数据集为工业监测真实数据：SMD、PSM、SWaT。

**📈 对比分析**

对比方法包括Isolation Forest、Deep-SVDD、OmniAnomaly、GDN、InterFusion、TranAD、TimesNet等基线。MSCENet在SMD、PSM、SWaT三组数据上分别取得最高F1分数，平均F1达91.35%，精度94.73%，召回88.52%，显著优于所有对手。

**⚠️ 局限性**

局限性包括：计算资源消耗较大，未给出实时部署或在线检测的效率评估；对概念漂移、跨域迁移的鲁棒性尚未深入；图结构动态学习的理论与可解释性仍待完善。

---

## 254. Self-Review Reinforcement Learning (SRRL) with Cross-Episode Memory and Policy Distillation

**arXiv ID:** 2607.05541 | [PDF](https://arxiv.org/pdf/2607.05541v1)

**作者:** Muhammad Zain Amin `[一作]` (École de Technologie Supérieure), Kibele Sebnem Yildirim `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 Self‑Review Reinforcement Learning (SRRL) 框架，在 RL 训练过程中加入自评、跨回合记忆和策略蒸馏，实现模型在失败后主动分析错误并在同一 episode 内改进答案。

**💡 创新点**

创新点在于将自评作为结构化信用分配步骤，并通过跨回合记忆收集成功自评、通过策略蒸馏将改进直接嵌入基准策略，从而在推理时无需额外反思，实现零延迟部署。

**🔧 技术方法**

技术核心包括基于 GRPO 的策略梯度优化、KL 正则化、交叉回合记忆检索、以及针对成功二次尝试的选择性蒸馏损失，配合自评奖励机制。

**📊 数据集**

实验使用 GSM8K 数学推理数据集，分别在 Qwen3‑4B（4B 参数）和 OLMo‑3.7B（3.7B 参数）两种大模型上进行训练。

**📈 对比分析**

与标准 RLVR 对比，SRRL 在验证集和测试集上都取得更高奖励（Qwen3‑4B：0.778 vs 0.762；OLMo‑3.7B：0.742 vs 0.714），并且学习效率更高，收敛速度更快。

**⚠️ 局限性**

局限性包括：对自评质量高度依赖；自评仅在失败时触发可能导致学习信号稀疏；跨回合记忆的规模和检索策略尚未深入研究，且对模型已有先验知识的依赖可能限制在更强基线模型上的进一步提升。

---

## 255. From Graphs to Gradients: Physics-Inspired Structural Attribution for Cyber-Physical IoT Systems and Beyond

**arXiv ID:** 2607.05563 | [PDF](https://arxiv.org/pdf/2607.05563v1)

**作者:** Spyridon Evangelatos `[一作]` (Netcompany S.A.), Panagiotis Sarigiannidis `[通讯]` (University of Western Macedonia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于统计力学的能量场模型，用于对混合连续与离散的工业 IoT 系统进行依赖性解释；

**💡 创新点**

创新点在于：①不依赖显式的有向因果图，而是通过无向能量网络捕捉系统内部统计依赖；②结合局部梯度敏感度、全局自由能与熵、以及二阶曲率，提供多层次、可解释的因果归因；

**🔧 技术方法**

主要技术包括：Boltzmann 分布的无向图模型、能量函数分解为一元与二元势、梯度与Hessian 计算、Monte‑Carlo 采样估计自由能与熵，以及基于这些量的归因算法；

**📊 数据集**

实验采用“Secure Water Treatment”工业水处理测试平台的 IoT 数据集，涵盖多种传感器、阀门、泵等混合型变量；

**📈 对比分析**

与 GNNExplainer 进行对比，评价指标为 Precision@k、AUC‑PR 以及鲁棒性和运行时间；实验结果表明能量场方法在归因准确率、鲁棒性和可扩展性上均优于 GNNExplainer；

**⚠️ 局限性**

局限性包括：无法恢复真正的有向因果结构，需要依赖外部领域知识判断方向；对二阶曲率的计算在变量数大时成本高；模型假设能量函数足够平滑，可能不适用于高度非线性或快速时变的系统。

---

## 256. Multimodal Molecular Representation Learning with Graph Neural Networks, Deep & Cross Networks, and SMILES Embeddings

**arXiv ID:** 2607.05736 | [PDF](https://arxiv.org/pdf/2607.05736v1)

**作者:** Qiwei Han `[一作]` (Duke University), Zheng Ma `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种参数高效的三分支模组融合神经网络，用于分子属性预测，融合3D图形神经网络、SMILES语义嵌入和显式物理化学描述符；

**💡 创新点**

创新点在于利用正交多模态融合与后期融合架构、语义瓶颈优化以及显式多项式交叉网络，实现在不到100万参数的条件下突破亚化学精度门槛；

**🔧 技术方法**

采用连续滤波图形网络（SchNet）、ChemBERTa+SwiGLU语义编码、Deep & Cross Network处理表格特征，并通过加法池化与多层感知器实现最终预测；

**📊 数据集**

使用QM9数据集进行实验，构建了129,012个经过3D、语义和表格信息齐全的分子；

**📈 对比分析**

通过与单模态和双模态基线的系统对比，三模态模型在验证集上取得MAE 0.0207 eV，较纯几何基线降低20.6%误差，且低于0.0433 eV的亚化学精度阈值；

**⚠️ 局限性**

局限性包括受限的几何基线、冻结的语言模型导致语义微调受限、有限的超参数搜索范围，以及仅在小分子QM9上验证，尚未测试对大分子或非平衡系统的泛化能力。

---

## 257. Efficient Transfer Learning of Robot Dynamic Models Using Morphological Similarity

**arXiv ID:** 2607.05665 | [PDF](https://arxiv.org/pdf/2607.05665v1)

**作者:** Pavlo Kupyn `[一作]` (Tallinn University of Technology), Maarja Kruusmaa `[通讯]` (Tallinn University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于自编码器的跨平台迁移学习框架，用于预测两台形态相似、规模不同的软体鳍驱动水下机器人（U-CAT 与 Micro-CAT）的运动动力学，并在目标机器人上实现无标签的状态估计。

**💡 创新点**

创新点在于：①利用共享潜在空间的自编码器进行无监督特征重构；②通过最大均值差异（MMD）损失实现两域潜在分布对齐；③在不需要目标域标签的情况下实现高质量的零样本迁移。

**🔧 技术方法**

使用的技术包括：1D 卷积编码器、动态预测头、域特定解码器、MMD 对齐损失、重构损失与动态损失的联合优化；训练采用 Adam、学习率 0.001、批大小 256、50 轮；数据输入为传感器读数（压力、IMU）与鳍控制参数。

**📊 数据集**

使用了两台实际水下机器人的实验数据集：U-CAT（源域）与 Micro-CAT（目标域），采集了压力、IMU、鳍动作、通过 ArUco 标记获得的真实速度等信息，覆盖前进、转向和方向反转等多种轨迹。

**📈 对比分析**

与基线方法（目标域单独训练、源域单独训练、两域联合监督训练）对比，自动编码器（联合头）在 RMSE/MSE 方面与联合监督模型相近；在无标签目标域上，自动编码器将源域误差降幅约 40%，表明实现了有效的零样本迁移。

**⚠️ 局限性**

局限性包括：①仅适用于形态相似、尺度差异有限的机器人；②在完全有标签时仍不如联合监督模型；③缺乏对超参数、损失权重及更复杂对齐策略（KL、对抗）的系统消融与敏感性分析；④未尝试递归或注意力编码器，可能限制对长时序特征的建模。

---

## 258. Do It Right! A Methodology for Successful NLP System Development

**arXiv ID:** 2607.05644 | [PDF](https://arxiv.org/pdf/2607.05644v1)

**作者:** Olga V. Patterson `[一作]`, Scott L DuVall `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出将系统开发生命周期（SDLC）方法体系化应用于临床自然语言处理（NLP）信息提取系统的开发与维护。

**💡 创新点**

创新点在于将传统软件工程的SDLC框架与临床NLP项目结合，强调大语言模型（LLM）在可用性、可解释性、版本漂移等方面的风险，并提出迭代式、可治理的开发流程。

**🔧 技术方法**

采用SDLC各阶段（规划、可行性分析、需求定义、文档选择、注释、设计、实现、测试、部署与维护）以及LLM、规则/模式、机器学习、混合系统等多种技术手段。

**📊 数据集**

未使用具体公开数据集；文中仅举例说明典型临床电子病历来源（如VA VistA、企业数据仓库）及一般性文本集合。

**📈 对比分析**

无实验比较或性能评估；论文主要提供流程与方法论框架，未给出精确指标或与其它系统的对比。

**⚠️ 局限性**

局限性包括：缺乏实证验证；对LLM的幻觉、提示敏感性、版本漂移等问题的处理仍需实践检验；适用性受数据治理、语言漂移、机构差异等因素影响。

---

## 259. Procedural Volumetric Modeling of Plant Branching Structures for Finite Element Analysis

**arXiv ID:** 2607.05421 | [PDF](https://arxiv.org/pdf/2607.05421v1)

**作者:** Ajith Moola `[一作]` (Iowa State University), Aishwarya Pawar `[通讯]` (Iowa State University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套自动化的程序化体积建模框架，能够将植物结构的骨架点云或已提取的骨架图转换为高质量、可用于有限元分析的六面体网格，并支持在植物生长过程中局部更新网格而无需重建整个几何。

**💡 创新点**

创新点在于：
• 采用B样条中心线曲线和柱状张量积B样条体积为每条分枝构建三维参数化；
• 设计了跨分枝节点的控制点混合策略（G¹连续性）来保证分枝体积在交汇处无缝连接；
• 区分全局与层次化构建策略，以保持父分枝的平滑性并在新增分枝时仅局部更新；
• 通过控制点层级融合解决了三叉节点的多分支接合，扩展了传统单分支或二叉分支的处理。

**🔧 技术方法**

技术与方法包括：B样条曲线/体积拟合、旋转最小化参考帧、基于控制点的节点混合（G⁰/G¹连续性）、分枝图构造、SVD求解最优法向量、分枝半径插值、六面体网格采样与评估、比例雅可比质量度量、稳态扩散有限元求解。

**📊 数据集**

使用了三组不同的植物数据集：
• Vigna radiata（黄豆）点云（多种品种）；
• Solanum lycopersicum（番茄）点云（TomatoWUR数据集）；
• Juglans regia（胡桃）树枝点云（含空间变异分枝半径）。

**📈 对比分析**

评估方法：
• 通过比例雅可比（scaled Jacobian）统计（最小值、均值、10%分位）验证网格质量；
• 对比不同分支角度下的双叉与三叉节点，发现大多数情况下qₑ>0.9；
• 对比完整重建与动态更新的时间，动态更新在每个生长阶段比重建快约3倍，最终重建耗时0.096 s，动态更新仅0.03–0.05 s。

**⚠️ 局限性**

局限与待改进：
• 仅实现了双叉和三叉节点的混合，未覆盖高阶分支节点；
• 在极其锐角或复杂几何下仍可能出现局部低质量或微小倒角的单元；
• 依赖先前的骨架点云或骨架图，若骨架提取误差大则影响后续网格；
• 参数选择（控制点数量、剪切半径、分段规则）对质量有显著影响，需人工调节。

---

## 260. ARMS: Anchor-Relational Motion Streaming for Seamless Solo-Social Motion Transitions

**arXiv ID:** 2607.05733 | [PDF](https://arxiv.org/pdf/2607.05733v1)

**作者:** Huakun Liu `[一作]` (Nara Institute of Science and Technology), Kiyoshi Kiyokawa `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了ARMS框架，实现从文本指令到长期、连续的单人运动和人际互动的无缝流式生成；

**💡 创新点**

核心创新在于Anchor–Relational动态不对称表示和模式感知的关系门控，允许单一模型在单人和双人场景之间平滑切换；

**🔧 技术方法**

采用因果变分自编码器压缩运动为时序潜在空间，使用因果关系扩散Transformer进行逐段去噪，同时引入文本条件（DistilBERT）和动态历史注意力；

**📊 数据集**

训练与评估使用HumanML3D（单人）与InterHuman（双人）数据集，并在InterX上测试跨骨架泛化；

**📈 对比分析**

与多种基线（T2M、MDM、ComMDM、InterGen、InterMask等）对比，ARMS在全窗口生成下在R‑Precision、MM Dist、FID和多样性上均名列前茅；流式生成时保持较低的峰值抖动和面积抖动，显著提升过渡平滑性；

**⚠️ 局限性**

局限包括：不直接考虑场景或多主体外部环境约束；仅支持最多两人交互，无法处理更大社交群体；以及硬切换模式门控可能在某些过渡场景下仍产生轻微不连贯。

---

## 261. IMR: Iterative Mode-World Weighted Regression for Multi-Agent Trajectory Prediction

**arXiv ID:** 2607.05705 | [PDF](https://arxiv.org/pdf/2607.05705v1)

**作者:** Honglin Wang `[一作]` (EACON), Yun-Fu Liu `[通讯]` (EACON)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于模式-世界加权回归损失的多智能体轨迹预测方法，并设计了迭代分段解码器，解决了模式多样性与预测精度的权衡问题。

**💡 创新点**

创新点包括：①模式-世界加权回归损失，兼顾模式多样性与世界级排名，抑制模式崩塌；②迭代分段解码器，每次直接输出坐标并将前一次预测结果与特征反向传播，提升预测精度。

**🔧 技术方法**

采用极坐标场景表示与动态注意力权重构建图，使用多层图注意力网络（GAT）+ LSTM 进行编码与解码；回归损失采用拉普拉斯损失，分类损失采用焦点损失；集成时采用加权 k‑means。

**📊 数据集**

使用 Argoverse 2 多智能体运动预测基准（及其单智能体子基准）进行训练与评估。

**📈 对比分析**

与 QCNeXt、LOF、SEPT 等方法在 Argoverse 2 评测集上进行对比；在多智能体基准中排名第一，avgBrierMinFDE_6 优于 QCNeXt 0.06；在单智能体基准中与 LOF 竞争，表现同样优秀。

**⚠️ 局限性**

迭代解码器的计算量较大，导致实时性能受限；整体模型复杂度高，训练成本显著。

---

## 262. Linking Hadith Narrator Identities Across Heterogeneous Arabic Biographical Databases: A Multi-Signal Entity Resolution Pipeline

**arXiv ID:** 2607.05424 | [PDF](https://arxiv.org/pdf/2607.05424v1)

**作者:** Taufiq Wirahman `[一作]` `[通讯]` (National Research and Innovation Agency), Taufiq Wirahman (National Research and Innovation Agency)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一个两阶段实体解析流水线，将Sanadset 650K中的传说链叙述者与Hawramani及Muslimscholars数据库中的身份进行跨源匹配。

**💡 创新点**

创新点在于结合阿拉伯语专门化归一化、基于名称的单一信号匹配以及多信号加权评分，以实现大规模开放域叙述者跨源链接。

**🔧 技术方法**

使用了阿拉伯文本归一化、双字母前缀索引、编辑距离排序的模糊匹配，以及结合姓名相似度、死亡年份接近度和可靠性等级的加权得分。

**📊 数据集**

主要数据集包括Sanadset 650K（650,986条圣训记录）、Hawramani（100,915条叙述者档案）和Muslimscholars（25,247名学者档案），以及AR‑Sanad的叙述者知识库。

**📈 对比分析**

实验结果显示Phase‑1将51.1%（94,628/185,216）Sanadset名称链接到Hawramani，Phase‑2则将94.7%（95,573/100,915）Hawramani名称链接到Muslimscholars，三源对齐达到93,588条。

**⚠️ 局限性**

局限包括死亡年份缺失导致匹配依赖姓名，47.7% Sanadset 名称无二字前缀候选导致未链接，且未构建正式的人工标注基准评估。

---

## 263. Prior-First, Condition-Second: Scalable and Controllable Hand Motion Completion

**arXiv ID:** 2607.05938 | [PDF](https://arxiv.org/pdf/2607.05938v1)

**作者:** Mingyi Shi `[一作]` (University of Hong Kong), Taku Komura `[通讯]` (University of Hong Kong)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种先学运动先验再加轻量级条件适配的框架，实现基于身体动态的高保真、实时、可控手部运动补全。

**💡 创新点**

创新点包括（1）先验-条件二阶段分离：先从海量无标签动作学习身体‑手部运动先验；（2）引入链式注意力（KCCA）保证机械耦合；（3）分层条件适配器实现少标注下的语义/属性控制。

**🔧 技术方法**

采用自回归扩散Transformer、结构化注意力、KCCA、语义分层适配器以及classifier‑free guidance等技术。

**📊 数据集**

使用100小时无标签大规模动作数据（Dataset A）训练先验，30分钟标注文本（Dataset B）训练适配器，跨域测试则使用AMASS、HunyuanMotion等数据。

**📈 对比分析**

与MDM、PriorMDM、Body2Hands、BOTH2Hands等基线对比，先验模型在FID、Diversity、FPS、MPJPE、Root Error等指标上均优；适配器在仅数小时标注下匹配甚至超越全监督方法，显示更好的数据效率与控制效果。

**⚠️ 局限性**

局限性包括只针对手部运动，未扩展到完整身体；未显式建模手物交互与接触；分层适配器时间分辨率有限；对语义正确性（任务成功）评价不足；在冲突条件下需要手动调节引导尺度。

---

## 264. Bit2Watt: A Cyber-Physical Vulnerability Exploiting GPU Workloads Across Power and Computing Infrastructures

**arXiv ID:** 2607.05993 | [PDF](https://arxiv.org/pdf/2607.05993v1)

**作者:** Zhouhao Ji `[一作]` (Zhejiang University), Wenyuan Xu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种新型攻击——通过合法的 GPU 工作负载（合成任务或 LLM 训练）实现高频功率调制，诱发数据中心电源系统的谐波失真、振荡并最终导致电网不稳定和数据中心服务中断。

**💡 创新点**

创新点包括：①将 GPU 计算负载视作可控的功率激励源，构建了一个跨层次的威胁模型；②提出两种基于用户级编程的功率调制技术（SWMA 与 LTMA）；③基于阻抗模型从电磁耦合角度定量分析 GPU 集群与高 DER 电网的交互；④揭示了攻击导致的自循环效应（DoS 与信息泄漏）并给出可能的侧信道实现；⑤通过仿真与实测验证了攻击的可行性及其对大规模配电网的级联破坏潜能。

**🔧 技术方法**

主要技术手段包括：GPU 内部频率与负载调度控制（CUDA kernel 与训练脚本注入）、阻抗等效与中频/高频谐波分析、Simulink 组合模型（DER、UPS、SPS、PDU、PCC）、电压/电流采样与 FFT、EMI 侧信道解码、以及针对多层电源链的功率谱传播实验。

**📊 数据集**

数据来源主要是：NVIDIA RTX 系列 GPU 的功率测量（高频 1.5‑6 kHz 变调实验）、GPT‑2 训练过程中的功耗采集、以及构建的 1 MW 混合 DER 配电网仿真和真实 8 kW 现场实验。并未使用公开数据集，而是利用自建实验平台和公共云 GPU 环境。

**📈 对比分析**

与传统功率波动攻击相比，本文攻击在 90% DER 环境下，仅需 1,000 台 GPU 即可将总谐波失真提升至 46.8%（远高于 IEC 61000‑3‑12 规定的 13%），并将系统阻尼比从 0.46 降至 -0.27，触发电网不稳定。实验显示，该攻击在云平台监控下几乎不可检测，且相比单纯高频负载波动，攻击能诱发更深层的电网级别级联失效。

**⚠️ 局限性**

局限性包括：①需要对大规模 GPU 进行高度同步，受网络时延与操作系统调度抖动影响；②高频调制对硬件的热应力与寿命有潜在风险；③在低 DER 负载比例下，谐波放大效应不明显；④目前检测方案依赖于细粒度功率采样，若云平台仅提供 1 Hz 级别的 PDU 监控，攻击仍难被识别；⑤缺乏针对大规模多租户环境的可扩展防御原型。

---

## 265. Auditing of Unlearning Algorithms

**arXiv ID:** 2607.05898 | [PDF](https://arxiv.org/pdf/2607.05898v1)

**作者:** Sahasrajit Sarmasarkar `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于成员推断的审计器，用于实证检验机器学习模型的未学习（unlearning）算法是否真正消除特定样本的影响。

**💡 创新点**

创新点在于：1）将差分隐私审计技术迁移到未学习场景，得到对未学习参数ε的下界；2）通过设计“平衡符号向量”与“批级预测”等两种审计实例，能够同时评估已认证与未认证的未学习方法；3）在实测中揭示了已认证算法（如模型裁剪、rewind‑to‑delete）与启发式算法之间显著的性能分化。

**🔧 技术方法**

使用技术包括：Membership Inference 攻击（LiRA 风格的 logit 统计）、局部差分隐私下界求解、联合与批级预测模型、对比基准 Pairwise 审计器、以及对批量划分与支持大小的调参。

**📊 数据集**

在 CIFAR‑100 图像分类和 Shakespeare 文本生成两套数据集上进行实验，分别使用不同的训练与未学习配置。

**📈 对比分析**

对比方法包括已认证的模型裁剪、rewind‑to‑delete 与多种启发式未学习算法（Hessian‑based、IDA、Forget Ascent、Retain Fine‑tune）。实验显示：已认证方法得到的ε下界几乎为零；启发式方法的下界常在数十到上百，表明其未学习效果差。

**⚠️ 局限性**

局限性：仅在δ=0 的情况给出下界；对于更一般的δ>0 或更复杂的模型（如大规模非凸网络）需进一步研究；目前的审计器对批大小与支持大小高度敏感，参数选择会显著影响下界。

---

## 266. LLM-Guided Measurement Credibility Correction for Trustworthy Industrial Process Inference

**arXiv ID:** 2607.06111 | [PDF](https://arxiv.org/pdf/2607.06111v1)

**作者:** Youcheng Zong `[一作]` (Northeastern University), Dakuo He `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大语言模型离线提取工业工艺文档中的测量语义，构建固定的语义矩阵；在线时基于这些语义构造不包含待校正测量的独立参考，并在预测前对存在本地冲突的测量进行保守校正，从而提升输入窗口的可信度。

**💡 创新点**

创新点在于将LLM生成的测量语义与数值模型耦合，实现在不依赖数值相关性、报警、故障标签或工艺方程的前置测量可信度校正；通过语义匹配挑选可信外部证据并进行局部纠正，避免了传统方法对完整工艺方程或标注的强依赖。

**🔧 技术方法**

技术手段包括：DeepSeek‑v4‑pro 作为LLM进行离线语义抽取，text‑embedding‑v4 生成语义向量；观察器（observer）学习语义加权参考；校正规则基于置信分布与支持稀疏化；下游预测采用 GRU、LSTM、Transformer、Informer、Mamba、iTransformer、PatchTST、ModernTCN 等时序模型。

**📊 数据集**

实验数据集涵盖三种典型工业过程：Ladle Preheating（钢厂炉内温度预测），Thickener Dewatering（尾矿脱水下渗浓度估计），IndPenSim（青霉素发酵罐浓度软感知）。

**📈 对比分析**

通过与相同模型基础的基线“Base”进行对比，评估 Real Test 与 Controlled Corruption Test 两种测试协议。结果显示 +MCC 在 Real Test 上平均 MAE 降低 30.7%，在 Controlled Corruption Test 上平均下降 80.3%；在 24 个数据集–模型组合中均取得最低 MAE；在线推理仅增加 0.5–2k 参数，最大延迟 0.089 ms/步，且峰值内存占用提升不足 304 MB。

**⚠️ 局限性**

局限性主要在于对过程文档完整性和准确性的依赖；若变量描述过时或不一致，测量语义可能失效；此外，对长周期语义漂移或测量链变更的实时更新机制尚未完善。

---

## 267. RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations

**arXiv ID:** 2607.06109 | [PDF](https://arxiv.org/pdf/2607.06109v1)

**作者:** Woo Jae Kim `[一作]` (KAIST), Sung-eui Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种鲁棒混合低秩专家（RoME）框架，用于多扰动对抗训练，解决不同威胁间的鲁棒性交易问题。

**💡 创新点**

通过低秩专家、双尺度门控和威胁引导门控多样化，实现不同威胁通过独立专家路径，避免了传统MoE的威胁无关路由和特征冗余。

**🔧 技术方法**

利用LoRA低秩专家、局部与全局特征双尺度门控、门控多样化正则化，并结合MAX或RANDOM对抗训练方法。

**📊 数据集**

在CIFAR-10、ImageNet-100及ImageNet-1K上进行实验，使用PGD/APGD生成ℓ1、ℓ2、ℓ∞攻击，亦评估未见攻击。

**📈 对比分析**

与现有MAT基线（RANDOM、AVG、MAX、MSD、MORE等）以及非ℓp方法（PAT、VR）进行对比，RoME在union robustness、自然准确率以及未见威胁鲁棒性上均超过对手，提升幅度多达3-4个百分点。

**⚠️ 局限性**

对门控参数和专家数量的选择仍需调优，且在极大规模模型或极低资源场景下扩展性与计算开销尚待进一步验证。

---

## 268. Statistical Adversaries: Natural Backdoor-like Features in Vision Datasets

**arXiv ID:** 2607.05516 | [PDF](https://arxiv.org/pdf/2607.05516v1)

**作者:** Paul K. Mandal `[一作]` (Neurint, LLC), Tristan Malatynski `[通讯]` (AGH University of Krakow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究利用仅来自训练集统计信息（如类均值、方差及频率特征）构造无模型梯度、无查询的通用目标特定扰动，并验证其能显著提升不同视觉模型的误报率。

**💡 创新点**

创新点在于：①把攻击方向完全从数据层面导出，而非依赖目标模型梯度；②提出两种基于频率的统计扰动（FFT‑Hellinger与带通白化）并证明其可跨模型迁移；③通过严格对照组（随机噪声、低通/频谱匹配、全局均值、错误目标）剖析效应来源。

**🔧 技术方法**

使用的技术包括：类条件均值对比、对角线白化、FFT与Hellinger频率统计、带通滤波、对扰动进行 ℓ∞ 投影、阈值化误报率评估、Benjamini–Hochberg 多重检验校正。

**📊 数据集**

数据集为 ImageNet‑1K，使用全部 1,281,167 个训练样本估计统计量，50,000 个验证样本做验证与确认，且验证集按固定切片划分避免数据泄露。

**📈 对比分析**

对照方法包括 5 种控制扰动；评估指标为目标负样本的误报率提升（ΔFPR）、误报提升倍数、ASR@1/5 等。实验结果显示，FFT‑Hellinger 与带通白化方向将误报率从 5.005% 提升至约 9.69%（1.94 倍），在 44 个模型-目标-预算组合中 43/44 方向有效，40/44 仍显著；不同网络（ResNet‑50、ConvNeXt‑Tiny、ViT‑B/16、Swin‑T）均可迁移，性能随架构差异显著。

**⚠️ 局限性**

局限性包括：①仅在选定的 11 个目标+构造方案上验证，未覆盖所有 ImageNet 类；②未评估在未见过目标类上的泛化；③扰动幅度仅测试两种 ℓ∞ 预算；④未探讨对抗鲁棒性训练或频率正则化的抵抗力；⑤实验未检验是否存在对抗样本对真实场景的实际危害。

---

## 269. Articulating Assumptions in AI-Generated Scientific Analyses through Task Decomposition

**arXiv ID:** 2607.05762 | [PDF](https://arxiv.org/pdf/2607.05762v1)

**作者:** Ahmed Hammad `[一作]` (Center of AI and Natural science, KIAS), Mihoko Nojiri `[通讯]` (Theory Center, IPNS, KEK)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出多代理框架，用于生成、执行、追踪和审查LLM生成的科学程序，并在高能物理分析中验证其可行性。

**💡 创新点**

创新点在于将代码生成拆分为辅助函数选择、执行回调、语义重构与后生成评审四阶段，并加入歧义澄清模块，显著提升小模型的可解释性与可靠性。

**🔧 技术方法**

使用的大模型包括 Qwen 14B/70B 等 LLM，结合多代理设计、辅助函数库、执行回溯、代码追踪和结构化规范与批判性审查技术。

**📊 数据集**

使用的数据集是 LHCO 轻量级事件格式的五个典型对撞机物理分析案例（WW、ttbar、γγ、WZ、Hjj）。

**📈 对比分析**

通过在相同的五个基准上比较不同模型的代码生成成功率、实现一致性和错误率，结果表明多代理框架使 14B 模型即可达到与 70B 相近的可靠性。

**⚠️ 局限性**

局限性包括仍受 LLM 推断不确定性影响，歧义检测召回不全，以及对领域外任务的适配需手工编写域包。

---

## 270. Harnessing Generative Image Models for Training-Free Primitive Shape Abstraction

**arXiv ID:** 2607.05568 | [PDF](https://arxiv.org/pdf/2607.05568v1)

**作者:** Gregor Kobsik `[一作]`, Leif Kobbelt `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种无训练的三维形状抽象管道，通过多视角渲染、生成式分割、投影聚类与超级四面体拟合，实现从任意3D物体到语义上有意义的几何原语的转化。

**💡 创新点**

创新点在于利用大规模生成式视觉模型的预训练知识完成语义分割，完全无需3D监督或额外训练，且可自动继承模型进步。

**🔧 技术方法**

采用多视角渲染、VLM分割、生成式图像模型生成彩色掩码、投影重投与颜色限制聚类，以及基于L‑BFGS的超级四面体参数优化。

**📊 数据集**

在HumanPrim和Toys4K两个公开数据集上进行实验，使用GroundTruth和生成式分割进行比较。

**📈 对比分析**

相较于PrimAny、F2C、SuperDec和EMS等方法，本文在Chamfer距离最低、IoU高、原语数少、重叠率近1，并在HumanPrim取得最高IoU；Toys4K上CD最低，IoU略低于EMS。

**⚠️ 局限性**

主要局限是依赖生成式模型的分割质量，导致分割噪声和视角不充分；运行时约一分钟/物体，且分割不可控的分解粒度。

---

## 271. Prompt Robustness Is Task-Dependent: Comparing Objective and Belief-Style Questions in LLM Evaluation

**arXiv ID:** 2607.05554 | [PDF](https://arxiv.org/pdf/2607.05554v1)

**作者:** Sadia Kamal `[一作]` (Oklahoma State University), Sagnik Ray Choudhury `[通讯]` (University of North Texas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了大型语言模型在客观多项选择题和主观意见题中的提示鲁棒性，系统评估了不同提示变体下模型答案一致性。

**💡 创新点**

创新点在于：①构建统一的提示扰动分类，覆盖语义、表面和答案呈现三类扰动；②在客观与主观两类数据集上同时进行鲁棒性比较，揭示主观问题对提示变化更敏感；③使用广义估计方程（GEE）对模型、数据集、提示类别及其交互效应进行统计检验。

**🔧 技术方法**

采用了确定性解码（温度0）、多种提示扰动生成（如改写、拼写噪声、词汇替换、逻辑等价、标签替换、格式变化、选项打乱），并通过答案一致性比例计算模型鲁棒性；使用GEE模型对一致性进行统计分析。

**📊 数据集**

客观数据集包括MMLU、ARC、CulturalBench‑Easy；主观数据集包括Political Compass Test、ValueBench、World Values Survey。

**📈 对比分析**

比较方法是计算每个模型、每个数据集和每种扰动类别下的答案一致性比例，并用GEE进行显著性检验。结果显示：主观题的一致性普遍低于客观题，尤其在选项顺序扰动下差距最大；不同模型对不同扰动的敏感度存在显著交互，提示鲁棒性并非单一属性。

**⚠️ 局限性**

限制包括：仅使用确定性推理和强制选择答案，未覆盖开放式回答；模型以族群级别评估，未细化至具体检查点或参数规模；主观数据集差异（词表、量表设计、社会语义）可能影响跨数据集比较。

---

## 272. InvWeaver: Deductive Feedback for Invariant Synthesis in Interacting-Loop Programs

**arXiv ID:** 2607.05478 | [PDF](https://arxiv.org/pdf/2607.05478v1)

**作者:** Guangyuan Wu `[一作]` (Nanjing University), Xiaoxing Ma `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对多循环程序的神经符号框架，用来自动推导循环不变式。

**💡 创新点**

创新点在于①构建循环级调用图（LCG）显式暴露循环间依赖；②基于LCG的义务驱动推理，给LLM提供层级化、上下文化的验证提示；③采用弱前条件（WP）推导的反馈机制，沿循环层次传播证明义务，支持多步修正。

**🔧 技术方法**

技术上结合了大型语言模型（Qwen3.7/DeepSeek/V5.5）、SMT求解器（Z3/Alt‑Ergo/CVC5）以及 Frama‑C 的 WP 插件，实现义务导向的提示和 WP 引导的诊断/修正。

**📊 数据集**

使用了三类数据集：传统 OOPSLA‑13、SV‑COMP 以及新构造的 CLRS‑Alg（包含 85 个经典算法，58 个多循环）。

**📈 对比分析**

与七个主流基线（UAutomizer、Code2Inv、CLN2INV、G‑CLN、LEMUR、AutoSpec、LaM4Inv）比较，本文在多循环基准上解决 72/82（比最强竞争者多 32 个），在单循环基准上 68/70（比 AutoSpec 多 19 个），平均提议数低、求解时间合理。

**⚠️ 局限性**

局限性：依赖 Frama‑C WP，无法处理动态内存或分离逻辑；基准规模仍较小，缺乏系统级真实程序；LLM 预训练可能导致数据泄漏；在极大循环嵌套或复杂堆结构时性能不一定可扩展。

---

## 273. PVCap: Towards Accurate 3D Dense Captioning via PseudoCap and VoxelCapNet

**arXiv ID:** 2607.06097 | [PDF](https://arxiv.org/pdf/2607.06097v1)

**作者:** Xiaopei Wu `[一作]` (SH AI Lab), Wanli Ouyang `[通讯]` (SH AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PVCap框架，结合PseudoCap数据增强与VoxelCapNet网络，用于3D点云的密集描述任务。

**💡 创新点**

创新点在于（1）PseudoCap通过随机混合不同场景实例生成多样化空间布局的伪帧，并借助教师-学生框架产生伪标签；（2）VoxelCapNet是首个基于体素的3D密集描述网络，充分利用体素化特征与检测头。

**🔧 技术方法**

采用体素化、稀疏卷积（VoxelBackbone如SparseUNet/Swin3D）、VoxelDetHead、交叉熵+自回归训练、教师-学生伪标签生成、SCST微调等技术。

**📊 数据集**

在ScanRefer和Nr3D这两个基准数据集上进行实验。

**📈 对比分析**

相较于Vote2Cap-DETR++等主流方法，ScanRefer上CIDEr@0.5IoU提升11.41%（Swin3D版），Nr3D提升13.99%；检测mAP@0.5也提升约6.33%。

**⚠️ 局限性**

局限性包括对体素分辨率的依赖，伪帧比例需要精细调节以避免过拟合，以及对多模态输入（如2D图像）融合的探索仍有限。

---

## 274. Scene Graph Thinking: Reinforcing Structured Visual Reasoning for Multimodal Large Language Models

**arXiv ID:** 2607.05716 | [PDF](https://arxiv.org/pdf/2607.05716v1)

**作者:** Zhiwei Yang `[一作]` (Tencent Youtu Lab), Shouhong Ding `[通讯]` (Tencent Youtu Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Scene Graph Thinking（SaGe）框架，利用场景图结构化视觉推理，提升多模态大型语言模型的细粒度感知与推理能力。

**💡 创新点**

创新点在于自动化将平面图像文本对转化为层次化场景图，构造基于节点与边的CoT数据，并通过节点代理奖励的强化学习实现图结构化推理。

**🔧 技术方法**

使用的技术包括大语言模型（如Qwen2.5-VL、GPT-OSS）、自动化场景图生成流水线、节点层次抽取、深度感知、两阶段训练（监督+GRPO强化学习）和节点代理奖励。

**📊 数据集**

数据集涵盖SA-1B、VStarBench、HRBench-4K/8K、CVBench-2D/3D、MMStar、RefCOCO、ChartQA等，生成120K结构化训练样本。

**📈 对比分析**

在八个视觉密集型基准上与GPT-4o、InternVL3、Qwen2.5-VL-32B等模型对比，SaGe在VStarBench、HRBench、CVBench等任务上实现了显著提升，3B模型甚至超过部分大模型。

**⚠️ 局限性**

局限性包括对高质量场景图生成的依赖、训练成本高、对动态或非静态场景的适应性不足，以及节点代理奖励可能导致奖励挖掘问题。

---

## 275. Umm... With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments

**arXiv ID:** 2607.05964 | [PDF](https://arxiv.org/pdf/2607.05964v1)

**作者:** Ivan Porupski `[一作]` (Jožef Stefan Institute), Nikola Ljubešić `[通讯]` (Jožef Stefan Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对四种斯拉夫议会语音（克罗地亚、捷克、波兰、塞尔维亚）近4,000小时的讲话进行大规模分析，使用基于 transformer 的自动填充停顿（FP）检测和 Mundlak 校正的 GEE 模型，探讨性别、年龄、语速、情感、政治取向和权力地位等因素对 FP 产生率的影响。

**💡 创新点**

创新点在于：①在跨语言议会语料库中首次实现大规模 FP 自动标注；②采用 Mundlak 校正的 GEE 将说话者间（between-speaker）和说话者内（within-speaker）效应分离；③首次将情感、政治取向和权力地位作为潜在预测变量引入 FP 研究。

**🔧 技术方法**

技术方法包括：wav2vec2-bert Transformer 进行 FP 检测（事件级 F1≈0.92）；XLM‑R‑ParlaSent 进行情感预测（R²≈0.65）；对 FP 计数使用负二项 GEE 模型，并用 Mundlak 方式将时间变异预测变量拆分为说话者平均值和偏差；对语速、年龄等变量进行 z‑score 标准化或均值中心化。

**📊 数据集**

数据集为 ParlaSpeech：6,000 小时的议会语音，四种语言（HR、CZ、PL、RS），经过滤后包含 3,889 小时、1,001,787 句子、1,561 名议员。

**📈 对比分析**

与传统单语、少量语料的 FP 研究相比，本研究使用了更大规模、多语言的语料，FP 检测准确率高（F1≈0.92），情感模型性能良好（R²≈0.65）。在模型比较中，baseline 与 Mundlak 校正模型在全球及各议会层面均进行了对比，结果以 IRR（Incidence Rate Ratio）呈现，显示不同变量在说话者间和说话者内的效应差异。

**⚠️ 局限性**

主要局限包括：①研究仅覆盖议会正式演讲，缺乏对日常或广播语料的验证；②政治取向和权力地位的效应在各议会间不一致，需进一步探讨；③自动 FP 检测虽然准确，但仍存在轻微的 precision 降低；④未对 FP 的持续时间或类型进行深入分析。

---

## 276. Plainbook: Data Science, in Plain Language

**arXiv ID:** 2607.05717 | [PDF](https://arxiv.org/pdf/2607.05717v1)

**作者:** Luca de Alfaro `[一作]` (University of California), Elena Baralis `[通讯]` (Politecnico di Torino)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的数据科学笔记本——Plainbook，它以自然语言描述为主，自动生成并执行 Python 代码，采用线性执行语义和检查点内核，并提供基于值的验证与测试机制，旨在让非程序员也能安全、可重复、可扩展地使用数据分析笔记本。

**💡 创新点**

创新点包括：
• 以自然语言描述为核心而非代码，保持描述可见、可编辑；
• 自动代码生成与实现稳定性（仅在描述变化时才重生成）；
• 线性执行模型与检查点内核，消除隐藏状态并支持增量重跑；
• 基于值的单元和全局验证工具（AI 校验、单元测试、全局测试），实现跨模型安全验证；
• UI 设计让非程序员可直接在笔记本中编写描述、查看结果、执行验证。

**🔧 技术方法**

技术包括：Python 作为基础语言；使用大型语言模型（Claude、Gemini 等）进行代码生成与验证；Jupyter 样式前端结合自定义检查点内核实现线性执行；自然语言提示工程（局部描述+全局上下文+变量类型）；HTTP 通讯与 Web 前端；支持文件、指令、数据类型信息的传递。

**📊 数据集**

主要数据集是国际足球（soccer）比赛数据集（文件路径可在“Files”标签选择），用于展示读取、处理、聚合与可视化等典型数据科学工作。

**📈 对比分析**

对比方法：作者未给出量化性能指标，但在讨论中指出
• 生成代码与执行的交互比 Jupyter+AI（Colab、VSCode）更快，尤其在会议式协作时可直接提交描述并得到结果；
• 通过单元测试与全局测试实现快速验证，避免了完整重新执行的开销；
• 线性执行模型和检查点缓存显著降低了重跑成本，支持逐单元迭代。
这些定性评估表明 Plainbook 在可用性和安全性上优于传统 Jupyter+AI 方案。

**⚠️ 局限性**

主要局限：
• 需要 AI API 密钥和计费配置，门槛较高；
• 检查点内核无法跟踪外部状态（如打开的文件、数据库连接），导致对数据库操作不可幂等；
• 代码重生成策略过于保守：修改早期单元会触发后续所有单元的重生成，导致性能下降；
• 目前仅支持单文件数据读写，未处理多文件或大规模数据管道；
• 对 AI 生成代码的安全性仍依赖跨模型验证，无法完全防止潜在漏洞。

---

## 277. KAT-Coder-V2.5 Technical Report

**arXiv ID:** 2607.05471 | [PDF](https://arxiv.org/pdf/2607.05471v1)

**作者:** Bo Huang `[一作]`, Kun Gai `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发并发布了 KAT-Coder-V2.5，一款面向编码任务的自主代理，通过可重现的多语言仓库环境、轨迹质量提升以及强化学习优化实现了在真实代码库中的自动化修复与工具使用。

**💡 创新点**

创新点包括：①可验证环境构建 AutoBuilder；②过程感知轨迹过滤与近失效回收；③工具随机化与可靠沙箱的 RL 训练框架；④异向 PPO 与后视价值估计；⑤多教师主动策略蒸馏融合多领域专家。

**🔧 技术方法**

采用的技术包括：多语言仓库构建管道 AutoBuilder、工具轨迹生成框架 KwaiClawEnv、基于 PPO 的异向 actor‑critic 强化学习、奖励塑造与模型评判（GRM）、工具调用随机化与沙箱可靠化、以及多教师主动策略蒸馏 MOPD。

**📊 数据集**

使用的数据集包括：来自真实仓库的任务（KAT Code Bench、SWE‑Bench Pro）、Kuaishou 内部业务任务（KAT Claw Bench）、公开基准 PinchBench、Terminal‑Bench、SciCode 等。

**📈 对比分析**

评估方法：在统一的 Claude Code harness 下对比六大基准，所有模型使用相同工具、上下文预算与解码配置。KAT‑Coder‑V2.5 在仓库级软件工程排名第二（SWE‑Bench Pro 65.2，KAT Code Bench 53.1），在 PinchBench 获得最高分（94.9），在 KAT Claw Bench 与 Opus‑4.8 接近，终端与科学编程方面略逊于更大模型。

**⚠️ 局限性**

局限性：对终端/科学编码任务的表现仍不如最前沿模型；系统对基准任务的依赖较高，难以直接迁移到未见环境；高质量可验证环境构建和轨迹过滤仍需手工规则；可能对大规模动态项目或持续集成流水线的适配不足。

---

## 278. Hidden Amplifiers: Cross-Level Risk in Software Supply Chains

**arXiv ID:** 2607.05894 | [PDF](https://arxiv.org/pdf/2607.05894v1)

**作者:** Rakesh Podder `[一作]` (Colorado State University), Indrajit Ray `[通讯]` (Colorado State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种跨层级风险传播框架，将方法级代码指标与生态系统依赖曝光度统一为可比较的风险分数，进而识别在现有SCA工具中不可见的“隐藏放大器”。

**💡 创新点**

创新点在于：①构建了一个乘法组合的风险模型，融合复杂度、中心性、散布半径和版本变动等代码维度与包的粉丝数量（fan‑in）维度；②通过此模型实现跨包方法的可比性，发现微型依赖但拥有巨大传播影响的风险点；③提出并验证了多层流水线架构，将生态分析、静态调用图构建与风险合成分层实现。

**🔧 技术方法**

技术手段包括：静态AST解析生成调用图并计算方法级复杂度、betweenness中心性、blast radius 与 git churn；利用 deps.dev API 提取包的生态曝光度；采用 log10 减震函数压缩 fan‑in 范围；构建三层流水线（生态分析 → 代码分析 → 跨层合成）。

**📊 数据集**

数据集为从 npm（35个）与 PyPI（15个）挑选的 50 个高关注度包（框架、安全库、微依赖），每个包的完整依赖树、源代码与变更记录。对 10 个已知 CVE 根包进行优先级评估。

**📈 对比分析**

与单一维度评估（仅代码或仅 fan‑in）比较，使用 Precision@K（K=3,5,10）测评。跨层级得分在 10 个 CVE 包中 Precision@5 为 0.44，明显高于仅 fan‑in（0.20）且与仅代码（0.44）相当，显示代码层面增强了优先级判定；此外跨层级排名在 24.5% 的方法位置出现重排，体现了生态曝光对优先级的显著影响。

**⚠️ 局限性**

局限性包括：①样本为手工挑选的高 fan‑in 包，可能高估隐藏放大器比例；②仅覆盖 npm 与 PyPI，未评估其它生态；③只分析每个包前 20 个方法，未完成全图可达性分析；④静态调用图解析受动态导入与多态限制，resolution rate <0.85 的包需人工验证；⑤大规模部署需增量分析与缓存支持。

---

## 279. Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning

**arXiv ID:** 2607.05773 | [PDF](https://arxiv.org/pdf/2607.05773v1)

**作者:** Akshay Arora `[一作]` (Uber AI Solutions), Siddarth Malreddy `[通讯]` (Uber AI Solutions)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 AgenticAI‑Supervisor，一个基于 API 和 UI 的 RL Gym 环境，用以在可验证的沙箱中生成高保真模拟、执行轨迹，并通过多维奖励机制对大语言模型驱动的自主代理进行持续评估与优化；通过 Customer Support Agent 案例验证了闭环反馈的有效性。

**💡 创新点**

创新点在于：① 将环境构造与高并发执行解耦，支持数千条并行轨迹；② 采用可验证的终结奖励和多维轨迹效率奖励，显著降低奖励游戏风险；③ 引入 deterministic + LLM‑as‑Judge 双重验证，兼顾严谨性与语义质量；④ 提议无代码仿真界面、自动“stumping”与 HITL 反馈，降低专家门槛。

**🔧 技术方法**

核心技术包括：
- 结合容器化隔离的 Rollout Engine 与 Agent Runtime；
- 多维奖励引擎（Outcome、Constraint Adherence、Trajectory Efficiency）；
- 结构化执行轨迹与 Span 记录；
- 内部状态验证器与 LLM‑judge 评估器；
- 通过 MCP 统一工具接口；
- 统一的 Dataset Connector 绑定测试场景。

**📊 数据集**

使用了内部构造的仿真数据集，主要包含 Customer Support 领域的工具 API 与 GUI 交互日志；通过 Dataset Connectors 将多种业务场景与状态上下文绑定；未公开使用公开数据集。

**📈 对比分析**

论文通过 Customer Support Agent 的案例演示闭环奖励反馈，显示高保真轨迹能有效驱动 RL 训练；但文中未给出量化的基准对比或性能指标，仅报告了框架的功能可行性和可信度提升。

**⚠️ 局限性**

局限性包括：① 主要以单一案例验证，缺乏跨域量化评估；② 仍需人工设计任务与“stumping”逻辑，自动化程度不足；③ 对于极端业务场景的泛化能力和安全性评估尚未完成；④ 目前对 LLM‑judge 的鲁棒性与可解释性依赖实验配置，需要进一步研究。

---

## 280. Learning When to Automate: Queue Control in Human-AI Service Systems

**arXiv ID:** 2607.06017 | [PDF](https://arxiv.org/pdf/2607.06017v1)

**作者:** Giovanni Montanari `[一作]` (Inria), Vianney Perchet `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了人机协作服务系统中的在线学习与排队控制，提出一种结合 UCB 与 Drift-Plus-Penalty 的策略。

**💡 创新点**

首次将自动化决策与人类排队调度联合优化，同时在未知参数下提供子线性 regret 与队列稳定性保证。

**🔧 技术方法**

使用 Upper Confidence Bound（UCB）估计聊天机成功概率与人类服务率，并结合 Lyapunov Drift-Plus-Penalty 控制。

**📊 数据集**

在合成实例中使用人工生成的任务类型、到达率、聊天机成功概率、人类服务率及后备权重。

**📈 对比分析**

与基准的 DPP、始终开/关聊天机 + 贪心调度比较，实验显示该策略在累计 regret 和队列长度上表现更优。

**⚠️ 局限性**

假设聊天机成功率线性且只能 0/1 开关，服务过程记忆无关且预抢，未考虑多人工智能与策略行为。

---

## 281. Unsupervised Anomaly Detection of Information Operations Users via Behavioral and Language Patterns

**arXiv ID:** 2607.05855 | [PDF](https://arxiv.org/pdf/2607.05855v1)

**作者:** Sishun Liu `[一作]` (RMIT University), Xiuzhen Zhang `[通讯]` (RMIT University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种无监督的异常检测方法，利用信息操作（IO）用户的行为（时间戳）和语言模式，自动识别社交媒体中的IO账户。

**💡 创新点**

创新点在于将时间点过程（TPP）与大语言模型（LLM）生成的证据函数结合，并通过Ephad算法对受IO用户污染的训练数据进行后期校正，显著提升检测鲁棒性。

**🔧 技术方法**

核心技术包括：时间点过程建模、LLM（如GPT）生成文本判断、Softmax相似度映射、Ephad指数平移调节、以及多模态融合。

**📊 数据集**

实验使用了五个真实世界IO数据集（Egypt、China_1、Iran_1、Russia_1、UAE），涵盖数千名IO与控制用户。

**📈 对比分析**

与聚类、Block、Luceri等基线对比，模型在精度、召回、F1、AUC及AP上均取得最高分，尤其在AP指标上平均提升约30%。

**⚠️ 局限性**

局限性包括：需手工调节温度β，LLM推理速度慢，且模型对极端样本分布或多语种文本的适用性尚未充分验证。

---

## 282. Multi-Teacher Contrastive Distillation for Edge-Efficient Pathology Foundation Models

**arXiv ID:** 2607.05533 | [PDF](https://arxiv.org/pdf/2607.05533v1)

**作者:** Tim Lenz `[一作]` (EKFZ for Digital Health, TU Dresden), Jakob Nikolas Kather `[通讯]` (EKFZ for Digital Health, TU Dresden)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出MuCoDi多教师对比蒸馏框架，利用多种大型病理基础模型（PFM）的冻结嵌入，训练轻量级MobileOne/RepViT编码器，实现边缘端高效病理图像特征提取。

**💡 创新点**

创新点在于将MoCo v3的对比学习改造为多教师蒸馏，使用冻结的PFM嵌入作为正负样本，避免教师梯度消失，并通过一次性蒸馏多模态特征空间，兼顾多模态互补性，显著降低模型规模与推理延迟。

**🔧 技术方法**

采用对比学习（MoCo v3）、知识蒸馏、跨教师对比蒸馏、轻量级移动网络（MobileOne、RepViT）以及分布式训练与bfloat16混合精度技术。

**📊 数据集**

使用11.8K TCGA病理WSI（约14.3M组织切片）进行预训练，并在23个临床标注二分类任务上评估，外部验证使用匹配的CPTAC队列。

**📈 对比分析**

通过冻结学生编码器提取特征，利用STAMP MIL头进行滑动窗口级别分类；在TCGA内部和CPTAC外部均可实现接近教师模型的AUROC（例如MuCoEdge-R2.3 71.0% vs Virchow2 71.8%），且在Raspberry Pi 5上实现最高605×单切片速度提升，参数量从6.4M降至0.2M。

**⚠️ 局限性**

局限性包括：与最强教师模型相比仍有微小性能损失；蒸馏过程依赖预先计算的教师嵌入，难以在线更新；仅在TCGA/CPTAC数据上验证，跨域泛化尚未充分评估；极小模型在极低算力环境下仍可能受限于图像分辨率与特征表达。

---

## 283. InfluMatch: Frontier-Quality KOL Search at 4B-Model Cost

**arXiv ID:** 2607.05968 | [PDF](https://arxiv.org/pdf/2607.05968v1)

**作者:** Krittanon Kaewtawee `[一作]` (Amity AI Holdings Company Limited), Touchapon Kraisingkorn `[通讯]` (Amity AI Holdings Company Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 InfluMatch，一套基于三阶段级联（检索→重新排序→推理）的泰语 KOL（关键意见领袖）匹配系统，能够把自由文本的多部分营销标准转换为可排序的候选列表，并为每个候选给出每个标准的分数和泰语解释。

**💡 创新点**

创新点包括：①使用小型开源 4B LLM 通过低成本的三阶段级联实现高精度匹配；②在重新排序阶段引入 SimPO 的对比学习，显著提升了排序准确度；③在推理阶段对每个标准使用 ordinal 核心评分并生成可审计的泰语理由，提供可解释性；④发现相对标注（pairwise）比绝对标注（pointwise）更适合终端效果，揭示了标注设计对模型性能的关键影响。

**🔧 技术方法**

技术方法包括：
- 采用密集向量检索（FAISS）筛选前 50 名候选；
- 4B 点对点重新排序器，利用 SimPO 在对比样本上进行微调，使用 log‑probability 评分；
- 4B 推理器在每个标准上输出 {0,1,2} 的 ordinal 分数并生成泰语理由；
- 通过 SFT+GRPO 对推理器进行绝对分数微调，尽管未提升终端性能；
- 在推理阶段实现成本优化：仅对 top‑10 重新排序结果进行推理，减少 token 与延迟。

**📊 数据集**

使用的数据集包括：
- 合成的泰语营销简报与对应的五个标准（基于真实结构与网路检索生成）；
- 人工标注的三种标签：点对点 ordinal 分数、整体通过/失败判定、最佳/最差对比选择；
- 对 50 名 KOL 的全面标注集（Set 1）和 10 名 KOL 的标注集（Set 2），用于端到端评估。

**📈 对比分析**

对比方法：
- 与检索仅 baseline（随机）和单一 dense retrieval 进行对比；
- 与行业前沿模型 Kimi‑K2.6（基于大模型的直接对比推理）比较；
- 评估指标包括：检索阶段的 P@5、重新排序的 EM、推理的 wf1、端到端的 P@5、MAP_B@5、nDCG@5。结果显示，完整级联在 Set 1 上 P@5 达到 94.1%（相较于检索仅 54.5% 提升 39.6%），与 Kimi‑K2.6 的 91.8% 相比仅消耗约 35 倍更少的输出 token，并且在 20 s 内完成一次 50 KOL 查询。成本方面，完整级联每查询仅约 420k tokens，远低于全检索+推理的 800k+ tokens。

**⚠️ 局限性**

局限性：
- 评估样本规模较小（Set 1 11 条查询，Set 2 31 条），结果可能对单个查询敏感；
- 数据主要基于合成简报，真实商业简报的分布可能不同；
- 仅在泰语 KOL 市场与 4B LLM 体系上验证，未测试跨语言或更大规模场景；
- 评估成本以 token 计量，实际延迟受并发与硬件影响；
- 对比推理模型 Kimi‑K2.6 采用不同硬件与并发设置，导致部分结果不可直接复现。

---

## 284. Deep Reinforcement Learning for Dynamic Battery Management of Autonomous Order Pickers

**arXiv ID:** 2607.05683 | [PDF](https://arxiv.org/pdf/2607.05683v1)

**作者:** Taniya Shaji `[一作]` (Indian Institute of Management Bangalore), Christof Defryn `[通讯]` (University of Antwerp)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了基于PPO的多智能体深度强化学习框架，用于多块仓库中自动移动机器人（AMR）的动态充电与任务调度，显著提升订单完成率。

**💡 创新点**

①在多块仓库环境中首次联合学习充电站选择、充电时长与退回仓库等决策；②引入平均奖励形式的PPO与差分GAE，实现更稳健的长期性能；③通过SHAP解释模型决策，为运营提供可操作规则。

**🔧 技术方法**

Proximal Policy Optimization（PPO）与Independent PPO（IPPO）算法、平均奖励强化学习、差分GAE、SHAP特征解释、基于图形化模拟的多智能体强化学习框架。

**📊 数据集**

仿真环境订单到达遵循泊松过程，实验中使用真实仓库24小时订单量分布生成的分段泊松到达率；未使用公开数据集。

**📈 对比分析**

与DQN、CTDE PPO及固定阈值等启发式基线对比，在不同仓库规模与到达率下，IPPO订单完成率最高，提升约6%；充电等待时间与充电时长显著降低。

**⚠️ 局限性**

仅考虑欧氏距离无碰撞；订单采用FIFO；电池消耗与负载无关；缺乏实际物理测试，仅在仿真中验证。

---

## 285. GraspIT: A Dataset Bridging the Sim-to-Real gap and back for Validated Grasping SE(3) Pose Generation

**arXiv ID:** 2607.05869 | [PDF](https://arxiv.org/pdf/2607.05869v1)

**作者:** Paul Koch. Adem Karakurt `[一作]`, André Sers `[通讯]` (Fraunhofer IPK)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 GraspIT 数据集与生成系统，利用 NVIDIA Isaac Sim、Franka Panda 机器人和四阶段滑移测试，实现了从真实 RGB‑D 场景到仿真场景的双向映射，并为每个抓取候选生成连续的质量评分和可执行的轨迹。

**💡 创新点**

创新点在于：① 将物理滑移测试与机器人轨迹规划相结合，生成可执行且具备连续评分的抓取标签；② 构建实时双向 Real↔Sim 注册管线，将真实场景数据无缝投射到仿真中；③ 以 Docker 容器化方式公开完整生成流程，方便复现和扩展；④ 通过大规模 ABC CAD 库实现数千种物体的多样化组合。

**🔧 技术方法**

使用技术包括：NVIDIA Isaac Sim（物理渲染、PBR、光照随机化）、Franka Panda（并行爪）、cuRobo（轨迹规划）、Lula（碰撞检测）、TRELLIS（高精度 3D 扫描与重建）、ICP（场景对齐）、四阶段滑移测试算法（轨迹+闭合、抬升、水平振荡、摆动）以及 Docker 容器化部署。

**📊 数据集**

数据集：共 1,035 个仿真桌面场景与 100 个真实场景，累计 316,160 个带 RGB‑D、分割、位姿、CAD 点云等完整注释的帧；物体来自约 10,000 个 ABC CAD 资源，形成 7,396 个独特物体；评估了约 2.3M 个抓取候选，其中 82.94% 为“好”抓取（score ≥ 0.5），其余 17.06% 包含梯度负样本。

**📈 对比分析**

作者未给出下游任务的直接评估，而是通过统计与现有数据集（如 GraspNet‑1B、GraspNet、ACRONYM 等）对比，展示 GraspIT 在物理验证、机器人可达性、连续评分、Real↔Sim 链接等维度的优势；同时指出传统基于力闭合的抓取率远高于实际效果，证明滑移测试的重要性。

**⚠️ 局限性**

局限性包括：仅支持并行爪抓取与单一 Franka Panda 机器人；缺乏完整任务级演示与语言注释；对摩擦、材质等真实物理属性建模不足；Real↔Sim 对齐误差不可避免；未提供下游基准评测；对多指手、吸盘等抓取方式不适用。

---

## 286. Revisiting the Relation Between Language Model Perplexity and ASR Word Error Rate for Modern End-to-End Speech Recognition

**arXiv ID:** 2607.05612 | [PDF](https://arxiv.org/pdf/2607.05612v1)

**作者:** Mohammad Zeineldeen `[一作]` (AppTek.ai GmbH), Hermann Ney `[通讯]` (AppTek.ai GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新审视了现代端到端语音识别系统中语言模型困惑度（PPL）与自动语音识别（ASR）词错误率（WER）之间的关系，探讨了外部语言模型是否仍能改善当前的端到端ASR系统，以及PPL-WER关系是否在对数空间中保持线性。

**💡 创新点**

创新点在于考虑了内部语言模型（ILM）对外部语言模型质量与WER之间关系的影响，并通过ILM减法增强了低PPL区域的斜率，表明ILM在解码过程中起着重要作用。

**🔧 技术方法**

使用了注意力机制的编码器-解码器模型（AED）和连接时序分类（CTC）模型，结合了不同的语言模型架构和大小进行实验。

**📊 数据集**

使用了LibriSpeech 960小时语音数据集和AppTek西班牙语数据集，分别包含不同的说话风格和领域覆盖。

**📈 对比分析**

通过比较不同ASR架构、编码器上下文设置和语言模型家族，发现外部语言模型在低PPL区域对WER的敏感度更高，而在高PPL区域斜率减小，表明外部LM仍然有助于CTC模型，尤其是在限制编码器上下文时。

**⚠️ 局限性**

限制在于ILM的影响可能会掩盖外部LM的效果，且在高PPL区域，PPL与WER之间的关系可能不再线性，导致对外部LM的依赖性减弱。

---

## 287. SafeImpute: Reliable Clinical Data Imputation via Conformal Selection

**arXiv ID:** 2607.05613 | [PDF](https://arxiv.org/pdf/2607.05613v1)

**作者:** Xinrui He `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种可靠的临床数据缺失值填补方法（SafeImpute），在事件图上学习稀疏、非均匀随访记录的填补，并通过合成p值与Benjamini–Hochberg方法实现对临床不可接受误差的FDR控制。

**💡 创新点**

创新点包括：
1) 事件图构造同时捕获病人内的时间轨迹与病人间的临床相似性；
2) 两关系图神经网络（时间边与值边）并采用自适应融合门控实现信息动态加权；
3) 设计代理风险得分，将预测不稳定性与证据稀缺度结合，形成可校准的p值；
4) 将 conformal selection 与 BH 程序结合，首次在缺失填补任务中提供统计意义的错误率控制。

**🔧 技术方法**

技术手段包括：
- 事件图（temporal & value edges）
- 两关系 GNN + 自适应门控
- 随机遮蔽辅助重构损失
- 边扰动与预测不稳定性度量
- 证据惩罚因子
- 合成p值计算与 Benjamini–Hochberg FDR 控制

**📊 数据集**

使用的数据集：
- Mayo Clinic（2022–2023糖尿病患者 HbA1c 数据）
- MIMIC‑III（ICU 病例）
- MIMIC‑IV（ICU 病例）

**📈 对比分析**

与 12 类基线（统计：Mean、KNN、MICE、MissForest、HyperImpute；深度：GAIN、GRAPE、MIWAE、TDM、ReMasker、DiffImputer；序列：LSTM、TRANS）进行对比。SafeImpute 在 MAE/RMSE 上常居第一或第二名，且在 FDR 控制下实现最高精度与覆盖率，整体性能显著优于所有基线。

**⚠️ 局限性**

局限性：
- 目前仅针对单一目标实验室指标（HbA1c）验证，泛化到多指标尚需进一步评估；
- 事件图构造依赖阈值选择，对稀疏记录可能仍产生噪声连接；
- 代理风险得分与校准依赖大量已标注样本，标注稀缺时可能失效；
- 未对下游任务（预测、决策）直接评估可靠填补的实际临床效益；
- 计算量相对较大，实时部署需进一步优化。

---

## 288. CoPiT: Cognitive Pivot Translation for Digraphic Low-Resource Mongolian in the Traditional Script

**arXiv ID:** 2607.05849 | [PDF](https://arxiv.org/pdf/2607.05849v1)

**作者:** Burte Bayarsaikhan `[一作]` (Korea University), Buru Chang `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CoPiT（Cognitive Pivot Translation）管线，先将传统蒙古文转换为西里尔文以消除书写歧义，再进行目标语言翻译；同时利用该管线生成真实传统文与目标语言的合成平行语料，用于提升低资源翻译。

**💡 创新点**

创新点包括：① 以认知阅读习惯为导向的多步骤脚本消歧义（元音和谐恢复、拉丁辅助标准化、Cyrillic 标准化）；② 句级自我反思机制提升全句一致性；③ 通过内部脚本层中介实现无须大规模传统-目标平行语料的翻译；④ 生成合成平行语料实现双向翻译与进一步提升。

**🔧 技术方法**

技术手段主要是：基于 LLM 的多组件（vowel harmony recovery、Latin‑assisted normalization、Cyrillic normalization、self‑reflection）微调；使用 LoRA 进行参数高效微调；利用传统–西里尔字典、句子级修正对齐数据；构建合成平行语料并对翻译模块进行额外微调；评估使用 BLEU、chrF、COMET、COMETKiwi 等指标。

**📊 数据集**

使用的数据集包括：① 1,031 句级传统–西里尔对照语料（附英、韩、俄参考译文）；② 380 篇包含传统与西里尔文本的新闻文章；③ 14,125 条传统–西里尔字典级词表（带元音和谐标签）；④ 8,034 句级合成平行语料（传统 → 英/韩/俄）。

**📈 对比分析**

与直接翻译基线（Qwen‑3、Ministral‑3、GPT‑4.1）在多语言、多模型下进行比较。CoPiT 在所有模型上均提升了 BLEU、chrF 及 COMET 分数，COMET 提升约 0.15–0.28（+1.5–1.6×相对提升）。在零样本和微调设置下，CoPiT 的开源模型可与 GPT‑4.1 的直接翻译性能相匹配或超越。

**⚠️ 局限性**

局限性包括：① 推理时需要多阶段转换，导致推理延迟；② 依赖西里尔→目标语言翻译的质量，若该阶段表现不佳会影响整体性能；③ 对传统蒙古文的评估受限于缺乏专用指标，主要依赖 BLEU/chrF；④ 当前管线仅针对从传统蒙古文向其他语言的单向翻译，双向翻译仍需进一步完善。

---

## 289. From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Space

**arXiv ID:** 2607.05794 | [PDF](https://arxiv.org/pdf/2607.05794v1)

**作者:** Yue Xu `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建多层级用户记忆金字塔，并训练LLM通过工具主动导航记忆以回答问题。

**💡 创新点**

把长期记忆用作可学习的行动空间，结合多粒度记忆工具和强化学习实现主动记忆导航。

**🔧 技术方法**

多粒度记忆金字塔、五种记忆工具（search、get、file‑read等）、GRPO强化学习、Qwen3.5‑9B LLM等技术。

**📊 数据集**

PersonaMem‑v2、LongMemEval、LoCoMo、GPQA‑Diamond、BFCL‑v3、V*Bench 等基准数据集。

**📈 对比分析**

与 Mem0、Zep、MemOS、MemoryOS、AgeMem 等基线对比，9B RL 模型在内存密集型基准上平均分 62.74，显著优于 397B 版 59.85；在非记忆任务上保持或提升性能，同时显著减少不必要的记忆调用。

**⚠️ 局限性**

仅在现有基准上评估，未充分考虑隐私、忘记机制；缺乏更大规模模型的深入分析。

---

## 290. Extending the Ginsburg-Spanier Theorem to Functions and Mixed Arithmetic

**arXiv ID:** 2607.05701 | [PDF](https://arxiv.org/pdf/2607.05701v1)

**作者:** Alain Finkel `[一作]` (ENS Paris-Saclay), Jérôme Leroux `[通讯]` (Université Bordeaux 1)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并统一了整数、实数和混合加法理论中可定义集合与函数的几何描述，给出了半线性、半极线性以及分段线性/分段简单函数的完整表述。

**💡 创新点**

首次以纯代数方式给出 Presburger、实数和混合加法理论中可定义函数的分段线性/分段简单性质，并纠正了 Weispfenning 对混合线性集合的错误描述，构建了半极线性集合的概念。

**🔧 技术方法**

仅使用线性代数、Ginsburg–Spanier 定理以及 Fourier–Motzkin 消元等经典工具，完全避免了自动机和量化消元技术。

**📊 数据集**

无实验数据，本文为纯理论研究。

**📈 对比分析**

未进行实验比较，主要通过理论证明和结构化归纳展示结果的正确性。

**⚠️ 局限性**

结果仅适用于三种加法理论，未覆盖带 p‑进范数的 Presburger 扩展，且缺乏对实现复杂度的细节分析。

---

## 291. LogicHunter: Testing LLM Agent Frameworks with an Agentic Oracle

**arXiv ID:** 2607.06195 | [PDF](https://arxiv.org/pdf/2607.06195v1)

**作者:** Minghui Long `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对LLM代理框架的联合生成与诊断的测试框架，包含基于规范的输入合成和主动的Agentic Oracle来定位真实缺陷。

**💡 创新点**

创新点在于：①将框架的Pydantic类型约束与真实仓库使用模式融合，生成“合法且极端”的测试输入；②设计主动推理的Agentic Oracle，利用文档检索、代码导航和沙盒执行，实现高精度的缺陷判定；③构建六维缺陷分类与高置信度共识机制，显著提升误报率。

**🔧 技术方法**

技术包括：LLM驱动的多代理生成（生成器、修复器、变异器），ReAct+FSM双层状态管理的主动推理，双流记忆管理，反射式代码检索工具，行为探针与期望断言，六维缺陷分类和高置信度共识。

**📊 数据集**

使用真实项目中常见的LLM代理框架（LangChain、LlamaIndex、CrewAI）的核心API作为测试目标，构建的失效案例库包含1,000条真实失败测试（其中28个真正缺陷），并评估在公开bug集合与硬负例集上的表现。

**📈 对比分析**

与Pynguin、Fuzz4All、TitanFuzz、TELPA等生成工具以及Raw Failure、Heuristic Filter、LLM Judge等被动Oracle进行对比。实验显示，生成阶段实现覆盖率最高，L3有效失败测试数量远超基线；Oracle阶段Agentic Oracle精度91.17%，召回72.14%，误报率仅0.21%，比最佳被动Oracle提升约61个百分点；成本和人工审核负担大幅降低，平均每发现1个真实缺陷需审核1.1个测试。

**⚠️ 局限性**

局限性包括：①对高质量使用示例的依赖，稀缺API难以生成优质种子；②更复杂的多API交互或外部状态依赖（数据库、网络）仍难以完全覆盖；③多轮LLM推理导致延迟和非确定性；④评估仅覆盖Python框架，跨模型/跨语言的通用性待验证。

---

## 292. Patch Knowledge Transfer for Efficient AI-Generated Image Quality Assessment

**arXiv ID:** 2607.05605 | [PDF](https://arxiv.org/pdf/2607.05605v1)

**作者:** Jiquan Yuan `[一作]` `[通讯]` (Peking University), Jiquan Yuan (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对AI生成图像质量评估（AIGIQA），提出了一种通过知识蒸馏实现的Patch Knowledge Transfer（PKT）框架，兼顾高效推理与高精度评估。

**💡 创新点**

创新点在于：①采用局部-全局混合教师模型与仅全局学生模型的双模型架构；②设计多层次知识传递机制，包括特征层对齐与输出层蒸馏；③在保持学生模型单一尺度推理的同时，显著提升其对多级视觉信息的理解能力。

**🔧 技术方法**

主要技术包括：ViT图像编码器、CLIP文本编码器、特征对齐损失（余弦相似度）、KL散度蒸馏损失、MSE回归损失，以及两种训练策略（一阶段联合训练与两阶段先训练教师后蒸馏）。

**📊 数据集**

在四个主流AIGIQA数据集上进行评估：AGIQA-1K、AGIQA-3K、AIGCIQA2023、PKU-AIGIQA-4K。

**📈 对比分析**

与现有9种SOTA IQA模型（如MUSIQ、HyperIQA、StairIQA、MANIQA、LIQE、CLIP-AGIQA等）比较，PKT在保持相同或更低FLOPs的前提下，平均提升SRCC/PLCC约0.01-0.05；相比高性能大模型（如MANIQA、MUSIQ）可减少约95–113 G FLOPs，且在多数据集上实现SOTA平均表现；同时在推理速度上，学生模型仅需教师的36%时间。

**⚠️ 局限性**

限制在于：①仍需教师模型进行训练，训练成本相对较高；②在极高分辨率或极大样本量的部署场景下，学生模型的单尺度处理可能存在细节捕捉不足；③模型对不同视觉风格或文本提示的鲁棒性尚未系统评估。

---

## 293. How Stable Is a PNT Resilience Score? Decision-Instability of Single-Number Resilience Ratings under Framework-Aligned Weighting

**arXiv ID:** 2607.05415 | [PDF](https://arxiv.org/pdf/2607.05415v1)

**作者:** Chakshu Baweja `[一作]` `[通讯]` (Ashforde O"U), Chakshu Baweja (Ashforde O"U)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 RPCF v2.0 的开放式确定性评分引擎，并对七种合成 PNT 架构在五种威胁情境下进行决策稳定性评估。

**💡 创新点**

首次量化评估 PNT 韧性指标的组合不稳定性、最弱环节成熟度级别随威胁变化的依赖性，以及自我声明与测量结果的差距。

**🔧 技术方法**

使用 PNT 仿真器、Dirichlet 权重采样、Kendall τ、top‑1 flip 率等统计方法进行敏感性分析。

**📊 数据集**

采用七个合成 PNT 架构（单频 GNSS、宽频 GNSS、GNSS+惯性、欺骗防护、多频 GNSS、声明全部技术的单频接收机、真正多源多域架构）和五个威胁场景（正常、宽频干扰、欺骗、机密、合并攻击）作为数据集。

**📈 对比分析**

通过比较不同权重分布和威胁情境下的组合分数与成熟度级别，计算 top‑1 flip 率、等级翻转率、平均 Kendall τ 等指标，结果显示在近等权重下稳定性高，但在宽泛权重下容易重排。

**⚠️ 局限性**

仅基于仿真和参数化模型，未进行现场测量；未建模位置精度；阈值和决策规则的鲁棒性未被评估。

---

## 294. Say What? Examining Text and Voice Input Modalities for Prompt-Based Programming in Computing Education

**arXiv ID:** 2607.05808 | [PDF](https://arxiv.org/pdf/2607.05808v1)

**作者:** Kaitlin Riegel `[一作]` (University of Auckland), Adish Singla `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对比学生在提示式编程任务中使用语音输入和文本输入的效果，分析了成功率、使用持久性和学生主观感受。

**💡 创新点**

首次系统评估语音输入在Prompt Problem中的可行性与优劣，并探讨编辑语音提示后其与文本输入相当的可能性，揭示语音可用于更深层次的认知参与。

**🔧 技术方法**

采用二元逻辑回归、混合效应模型和多标签情感分析；利用OpenAI Whisper进行语音转写，GPT‑4o‑mini生成代码，Prompt Programming平台收集交互日志。

**📊 数据集**

使用来自新西兰奥克兰大学2025年春季C语言入门课程的919名学生的交互日志和问卷数据，共计约900个提示实例；无公开预先构建的公开数据集。

**📈 对比分析**

通过比较不同输入模式下的首次成功率（文本：≈52–58%，未编辑语音≈35–38%，编辑语音≈38–50%）以及学生使用频率和偏好，发现文本输入在初始成功率和使用率上优于未编辑语音；编辑语音与文本的成功率无显著差异；在感知上，文本输入在准确性、可编辑性、规划等方面获得更高正面评价。

**⚠️ 局限性**

自选输入模式导致样本偏倚，语音使用者比例低；语音转写错误（尤其是非母语英语者）影响结果；缺乏对语音延迟的测量；未进行受控实验，因果关系不确定；任务过于简单，难以体现语音在复杂编程中的优势。

---

## 295. Agents That Teach: Towards Designing Incidental Learning Back into AI-Assisted Software Development

**arXiv ID:** 2607.06101 | [PDF](https://arxiv.org/pdf/2607.06101v1)

**作者:** Rohit Mehra `[一作]` (Accenture Labs), Adam P. Burden `[通讯]` (Accenture)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在 AI 辅助软件开发中重新引入偶发性学习的六条设计原则，并实现了基于多代理的 SHIELD 系统，以通过代理自身推理在不破坏开发者工作流的前提下呈现上下文化的学习时机。

**💡 创新点**

创新点在于把偶发性学习视为开发者-代理交互中的核心目标，首次构建了“教学代理”概念，并将其转化为系统实现（SHIELD），同时提供了六条具体设计原则，旨在使生产力与学习实现互补而非冲突。

**🔧 技术方法**

技术主要包括多代理系统架构、利用大型语言模型（LLM）进行代码生成与推理、上下文提取与离线学习提示的机制，以及与 IDE 的无缝集成。

**📊 数据集**

数据集：文中未给出具体实验数据集，假设使用公开的开源代码库（如 GitHub 公开项目）作为代理训练和评估素材。

**📈 对比分析**

方法对比：文中未提供量化的性能评估或与传统 AI 编码工具的对比，仅通过概念验证和设计原则阐述其对生产力与学习的潜在积极影响；性能表现以“兼顾生产力与学习”为主观评估。

**⚠️ 局限性**

局限性：1）偶发性学习仍需人为主动设计，系统可能无法覆盖所有学习场景；2）引入教学代理可能增加系统复杂度与维护成本；3）缺乏严格的实证验证，尚不清楚在真实开发环境中的实际收益与学习效果。

---

## 296. Is Domain Adaptation Always Helpful? A Frozen-Backbone Study of Cross-Domain Sentiment Transfer

**arXiv ID:** 2607.05937 | [PDF](https://arxiv.org/pdf/2607.05937v1)

**作者:** Phat Tran `[一作]` (Oregon State University), Yaolun Zhang `[通讯]` (Oregon State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究冻结预训练语言模型在跨域情感分析中的域适应效果。

**💡 创新点**

提出基于冻结backbone的域适应可行性与适用条件，并展示不同规模和域专门化backbone下的适配器训练策略。

**🔧 技术方法**

采用Qwen3-Embedding、RoBERTa、FinBERT冻结backbone，配合轻量化MLP适配器，使用DANN、MMD、Supervised Contrastive Loss及伪标签进行域适应。

**📊 数据集**

源域：Yelp Reviews、Amazon Polarity；目标域：SST‑2、Financial PhraseBank（二元情感）。

**📈 对比分析**

通过对比不同backbone与域适应配置的宏观F1，发现对SST‑2无显著提升，而在Financial PhraseBank上分布匹配可提升32.8点，FinBERT采用对比学习最佳。

**⚠️ 局限性**

局限包括二元情感任务、源域有限、目标集小导致方差大、域适应权重未针对模型调优、转移过程受序列化训练影响等。

---

## 297. Beyond Refusal: A Same-Lineage Study of Aligned and Abliterated LLMs for Vulnerability Analysis

**arXiv ID:** 2607.05842 | [PDF](https://arxiv.org/pdf/2607.05842v1)

**作者:** Mingchen Li `[一作]` (University of North Texas), Yunhe Feng `[通讯]` (University of North Texas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在同一模型家族（Gemma、Qwen）内部，对比对齐（Aligned）与拒绝消除（Abliterated）两种安全状态下，系统评估漏洞检测、CWE归因、行定位、根因定位及可执行补丁验证等完整漏洞分析工作流的效用。

**💡 创新点**

①使用相同家族模型只改变安全状态，消除架构与规模混淆；②将安全效用拆分为覆盖、答案质量、端到端效用三维度；③从任务深度、提示框架、可执行补丁验证等多维度深入探究安全状态对模型行为的影响；④设计提示强度对照实验，揭示安全状态与安全术语/授权语境交互。

**🔧 技术方法**

利用拒绝方向消除技术（对抗向量削弱）在Gemma、Qwen基础模型上生成Abliterated版本；构建统一的提示、解析与评估框架；使用可执行补丁验证流水线（应用、编译、PoV通过、完整验证）；统计分析覆盖率、F1、Top‑k、成功率等指标。

**📊 数据集**

PrimeVul、LineVul、Vul4J、PatchEval、Vul4C等漏洞与补丁数据集，涵盖检测样本、CWE标签、行级定位与可执行补丁。

**📈 对比分析**

采用覆盖率、准确率、F1、Top‑k、可执行通过率等度量，在不同提示框架（中性/授权/安全术语）和任务深度（检测→定位→补丁）下对比Aligned与Abliterated。结果显示：Aligned在浅层诊断和中性提示下更优；Abliterated在行定位、根因定位及安全显式提示下更强，并在可执行补丁的早期门槛中显著领先。

**⚠️ 局限性**

实验仅覆盖Gemma、Qwen两款模型；未加入外部安全管控（过滤、路由等）环境；补丁验证门槛低，最终成功率仍极低；缺乏更大规模、多语言、跨平台的实测；拒绝向量消除机制的普适性与可迁移性待进一步验证。

---

## 298. FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models

**arXiv ID:** 2607.05711 | [PDF](https://arxiv.org/pdf/2607.05711v1)

**作者:** Bowen Xue `[一作]` (Stanford University), Muyang Li `[通讯]` (Nunchux AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种全 4‑bit 后训练框架，支持对大型扩散模型进行 W4A4G4 后训练，能够在自定义、强化学习和蒸馏等多种任务中实现显著降低内存占用并提升训练速度。

**💡 创新点**

创新点包括：①三分支混合精度流水线，利用冻结的数值稳定器隔离量化敏感异常值；②块级量化策略实现高效反向传播；③针对 LoRA 与 MLP 的核融合，最大化 4‑bit 算子吞吐和内存带宽利用。

**🔧 技术方法**

采用了 NVFP4 低位浮点量化、SVD 分解+低秩稳定器、块级量化、LoRA 参数高效微调、定制 CUDA 核融合，以及量化反向传播等技术。

**📊 数据集**

实验数据集涵盖 FLUX.1‑dev (12B)、Qwen‑Image (20B)、Custom Diffusion、HPSv2、COCO‑10k、MJHQ‑30K 等，验证了在定制化、强化学习与蒸馏任务上的效果。

**📈 对比分析**

与 BF16 LoRA、NF4 QLoRA、FP8 LoRA 等基线对比，内存占用降低 2.25×，训练吞吐提升 2.27×，同时保持与全精度 LoRA 几乎无质量差异。

**⚠️ 局限性**

局限性包括：仍需数值稳定器以避免 4‑bit 训练中的梯度爆炸；块级量化在极端场景下可能略有精度损失；目前验证主要在 NVIDIA 4‑bit TensorCore GPU，CPU 或其他硬件的适配性尚未充分评估。

---

## 299. A Survey of Learn-to-Compute Paradigms for Rate-Distortion-Type Problems

**arXiv ID:** 2607.05417 | [PDF](https://arxiv.org/pdf/2607.05417v1)

**作者:** Shitong Wu `[一作]` (Tsinghua University), Wenyi Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了神经网络方法在率失真（RD）相关问题中的应用，阐述了三大 LtC（learn‑to‑compute）范式：变分推断、神经互信息估计和对偶形式优化，并对其理论、算法与一致性进行系统性整理。

**💡 创新点**

创新点在于将 RD、IB、iRD 等经典信息理论目标统一到 LtC 框架下，提出了三类可微分、可样本驱动的重写方法，并通过理论推导与实验验证对比三类方法的优劣，为后续研究提供了清晰的分类与评估视角。

**🔧 技术方法**

主要技术包括：变分自编码器（VAE）/信息瓶颈（VIB）框架、MINE（互信息神经估计器）及其对抗扩展、对偶形式的凸双重优化（如 NERD、MAIB、NEIRD），以及对应的随机梯度/批量优化算法。

**📊 数据集**

实验使用的主要数据集为：高斯联合分布（理论可解析 RD 曲线）和 MNIST 图像数据（用于 IB 与 iRD 的实证验证）。

**📈 对比分析**

比较方法：在 Gaussian 任务中比较 RD‑VAE、RD‑MINE 与 NERD 的 RD 曲线；在 MNIST 上比较 VIB、AIB 与 MAIB 的 IB 曲线；在 Gaussian 与 MNIST 上比较 NEIRD 的 iRD 曲线。实验结果表明：在低到中等率区域，所有方法均可近似理论曲线；在高率区，RD‑VAE 稳健，MINE 与对偶形式因 log‑期望估计方差大而表现偏差。

**⚠️ 局限性**

局限性包括：变分方法受限于近似族的表达能力，导致 variational gap；MINE 与对偶形式方法在高率或高维场景下的 log‑期望估计方差大，易出现偏差；所有方法在大规模、复杂分布上的泛化与训练稳定性仍需进一步改进。

---

## 300. Uncovering Latent Depression Severity for Binary Depression Detection via Advantage-weighting Ranking

**arXiv ID:** 2607.05901 | [PDF](https://arxiv.org/pdf/2607.05901v1)

**作者:** Manning Gao `[一作]` (South China Normal University), Sijie Mai `[通讯]` (South China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种细粒度多模态自动抑郁检测框架，并通过引入二元优势加权排序损失（BAR Loss）来优化特征空间分布，实现对抑郁风险的精细排序。

**💡 创新点**

核心创新点在于：①把抑郁检测转化为排名任务，利用优势加权机制动态聚焦难易样本；②BAR Loss同时实现类间间隔最大化与类内紧聚两种几何约束；③在互Transformer深度跨模态融合基础上，利用二元优势权重提升决策边界清晰度。

**🔧 技术方法**

技术手段包括：双流1D卷积+Seq‑TDNN时间编码、互Transformer深度跨模态融合、全局对比学习的优势加权排名损失、动态阈值调优、Optuna自动超参搜索等。

**📊 数据集**

实验使用了两大公开的自然场景抑郁视频数据集：D‑vlog（961个视频）和LMVD（1823个视频），涵盖来自YouTube、Bilibili、TikTok等平台的真实用户生成内容。

**📈 对比分析**

与多种基线（Bi‑LSTM、TBN、STST、DepTrans、TAMFN、STE‑Mamba、DepMamba、CAF‑Mamba）对比，模型在LMVD上实现最高F1=77.01，D‑vlog上获得F1=77.66，优于现有方法且保持了更好的精确率与召回率平衡。

**⚠️ 局限性**

局限性包括：仅在社交媒体自制视频数据上验证；对临床数据集的泛化能力尚未评估；BAR Loss 需要额外的配对计算，导致训练复杂度升高；以及对硬样本权重的设定仍依赖经验超参数。

---

## 301. PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates

**arXiv ID:** 2607.05483 | [PDF](https://arxiv.org/pdf/2607.05483v1)

**作者:** Zhaoyu Bai `[一作]` (Weizmann Institute of Science), Jiaqi Cai `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 PatchOptic，一种在共享状态 LLM 工作流中显式声明视图、写入范围和补丁来源的接口，并在运行时通过投影视图与 JSON Patch 验证来保障全局一致性和安全性。

**💡 创新点**

创新点在于把光学（optic）概念引入工作流约束，既实现了局部视图与全局更新的一致性契约，又提供了可静态检查的足迹代数，用于委托、组合和同相位重排；同时设计了 PatchBench 基准验证其有效性。

**🔧 技术方法**

主要技术包括：光学启发的投影视图与写入/来源范围定义、JSON Patch 与 JSON Pointer 的结构化补丁表示、基于路径的验证门（写入范围、来源范围、模式、阶段、不变式）以及足迹代数（限制、并集、互相排斥检查）。

**📊 数据集**

使用 PatchBench 作为数据集，包含 46 条手工构造的案例，覆盖财务、营销、医疗、科学、软件、支持等六个领域，按难度分为 L0–L5 六个构造级别，并标注隐藏泄露、补丁来源等标签。

**📈 对比分析**

比较方法：在同一批案例上跑六种设置（Unconstrained、Schema Only、FSM+ACL、View Only、Verify Only、PatchOptic），在两种 LLM（GPT‑5‑mini 强模型和 Mistral 7B 弱模型）下各跑十轮；评估指标包括语义通过率、内容质量、泄露率、泄漏次数、平均 token 消耗。结果显示：投影视图显著降低泄露率（从 25‑31% 降至 0.2‑0.7%）并减少 token（约 11‑13%），PatchOptic 通过验证成功率最高，且能阻止所有阶段/范围违规补丁。

**⚠️ 局限性**

局限性：依赖于正确编写的工作流契约；无法防止模型产生合法但语义错误的输出；对动态路径、别名、可变路径的处理有限；验证逻辑当前仅基于 JSON Patch，未覆盖更复杂的数据库或文件操作；评估仅限于手工构造的案例，真实场景复杂度和规模待进一步验证。

---

## 302. Automated Recommendation of Programming Learning Content Using Pattern-based Knowledge Components

**arXiv ID:** 2607.05409 | [PDF](https://arxiv.org/pdf/2607.05409v1)

**作者:** Muntasir Hoq `[一作]` (North Carolina State University), Bita Akram `[通讯]` (North Carolina State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于程序AST模式的知识组件（KC）来自动推荐编程学习资源，并给出可解释的代码层面解释。

**💡 创新点**

创新性地将递归AST子树作为KC，并使用SANN+β‑VAE+聚类生成可解释、无需专家标注的KC空间，提升推荐准确性。

**🔧 技术方法**

采用AST解析、子树注意力网络（SANN）、β‑VAE上下文编码、K‑means聚类、IDF加权以及余弦相似度进行推荐。

**📊 数据集**

使用PCEX系统的123个Python例程（示例与挑战），涵盖13个主题和49个专家定义的bundle。

**📈 对比分析**

与code2vec、ontology‑based、SANN、LLM‑KCI、KCGen‑KT等基线进行Top‑5、MRR、mAP比较，pattern‑based KC取得最高的0.89/0.90 Top‑5、0.81/0.80 MRR、0.82/0.79 mAP。

**⚠️ 局限性**

局限包括数据集规模小、依赖主题标签训练、解释热度可能重叠，且缺乏真实课堂实验。

---

## 303. Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models

**arXiv ID:** 2607.05702 | [PDF](https://arxiv.org/pdf/2607.05702v1)

**作者:** Marcelo dos Santos `[一作]` (Federal University of Paraná), David Menotti `[通讯]` (Federal University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FASR++，一种基于扩散模型的超分辨率算法，利用参考低分辨率图像和多张辅助低质量图像提取的特征向量来重建身份保持的高分辨率人脸。

**💡 创新点**

创新点在于引入特征融合网络（Feature Combiner）通过神经网络精细融合多张低分辨率特征，并在扩散模型中加入该融合特征作为条件，无需手工属性或梯度引导，显著提升身份保持与图像质量。

**🔧 技术方法**

使用扩散模型（SDE/VE/VP）与 NCSN++ U‑Net、时间/特征条件嵌入、特征融合网络、triplet loss 等技术。

**📊 数据集**

在 CelebA、Quis‑Campi（训练使用 FFHQ、CASIA‑WebFace）等数据集上进行验证。

**📈 对比分析**

与多种现有 SR 与人脸识别基线（GFPGAN、SR3、SDE‑SR、FASR、SRDG 等）对比，FASR++ 在 1:1 验证 AUC、1:N Rank‑1/Rank‑5、PSNR、SSIM、LPIPS 等指标均优于 SOTA，并且差异在统计上显著。

**⚠️ 局限性**

局限性包括对极端姿态和光照变化时易失效，以及扩散模型推理速度较慢；同时需要大量多视角低分辨图像与高分辨特征进行训练。

---

## 304. Harrison.Rad 1.5 Technical Report: A radiology foundation model that can draft reports from images, priors and clinical context

**arXiv ID:** 2607.05880 | [PDF](https://arxiv.org/pdf/2607.05880v1)

**作者:** Suneeta Mall `[一作]` (Harrison.ai), Jarrel Seah `[通讯]` (Harrison.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了一款多模态大语言模型 Harrison.Rad 1.5，可接受图像与文本输入，生成完整的放射科报告。

**💡 创新点**

创新点在于三阶段训练管线（域适配→对比视觉编码器→视觉问答微调）、专属的 Findings‑Diagnosis 评分体系与可解释性方法，并首次在 FRCR 2B 模拟考试中取得通过。

**🔧 技术方法**

技术包括 Transformer / Vision Transformer 对比预训练（CLIP/SigLIP）、Flamingo/BLIP‑2 风格的跨模态注意力、对比学习、视觉问答微调、Grad‑CAM 等可解释性工具。

**📊 数据集**

使用了约 650 万图像‑报告对进行对比学习，内部多模态对话集，公开数据集 RadBench、RadCoverage‑VQA、RexGradient、CBIS‑DDSM，及 FRCR 2B 模拟考试。

**📈 对比分析**

与 HR1、GPT‑5.4、Gemini‑3‑Flash‑Preview、Claude Opus、MedGemma 等模型对比；在 FRCR 2B Short Case 模拟中 HR1.5+ 通过率 62.5%，HR1.5 50%；闭合问题准确率 82% 诊断；在开放式报告中 Findings‑Diagnosis 评分 49.7%（HR1.5），显著优于大多数公开模型。

**⚠️ 局限性**

局限性包括对开放式文本指标仍不如专用胸部模型表现，低频/ OOD 图像的置信度评估尚不成熟，生成报告仍需人工审核，尚未完成临床验证和 FDA 认证。

---

## 305. Life Cycle Assessment of Pre-training the Lucie 7B Open-Source Large Language Model on the Jean Zay Supercomputer

**arXiv ID:** 2607.05408 | [PDF](https://arxiv.org/pdf/2607.05408v1)

**作者:** Marc Léobet `[一作]` (Mens Data), Jean-Pierre Lorré `[通讯]` (LINAGORA)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Lucie 7B 在 Jean Zay 超算上进行生命周期评估，量化制造与运营碳排放、水耗等指标。

**💡 创新点**

首次公开将测量运营数据、分系统的制造排放、水足迹与 AFNOR SPEC 2314 标准结合，并强调废热回收与暖水直接液体冷却的影响。

**🔧 技术方法**

采用 Labos 1point5 绿色气体排放估算、AFNOR SPEC 2314 框架、硬件使用日志、电力消耗测量、折旧计提等技术。

**📊 数据集**

使用 Lucie 7B 多语言训练语料（约 2.3 万亿 tokens）及其前置 120 亿 token 的分词数据，结合超算的运行与电力数据。

**📈 对比分析**

以每 GPU‑小时碳强度 36.7 kg CO2e 和 21.1 t CO2e 的训练总排放与前人 LLM 运营排放对比，显示制造与运营几乎相等，表明在低碳电网下硬件生命周期成为主要瓶颈。

**⚠️ 局限性**

局限在于未覆盖推理与服务、指标范围有限、未完整计入离线水足迹、GPU 制造排放范围不完整、折旧假设和缺乏 ISO 评估等。

---

## 306. Contextual Procurement Auctions with Bandit Learning

**arXiv ID:** 2607.05813 | [PDF](https://arxiv.org/pdf/2607.05813v1)

**作者:** Yiling Chen `[一作]` (Harvard University), Sadie Zhao `[通讯]` (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在带有上下文信息且仅有带反馈的情境下，平台如何通过重复的采购拍卖学习生产者的私有成本与未知的上下文相关产值，并设计了两类机制来平衡学习效率与激励兼容性。

**💡 创新点**

创新点在于：①提出了完全可执行、严格双向激励兼容的“探索后承诺（ETC）”机制，并证明其在福利退化方面达到O((ng)^{1/3}T^{2/3})的上界；②引入“冻结支付的UCB（F-UCB）”机制，将支付估计与分配学习分离，获得O(√(ngT))的近似UCB退化和可量化的激励误差；③给出匹配的下界，证明在冻结支付框架下无法突破ETC与F-UCB的退化-激励平衡边界。

**🔧 技术方法**

技术手段包括：上下文随机多臂赌博机（contextual bandit）和UCB算法；VCG/关键支付（critical payment）与分配的可单调性；探索后承诺框架下的无偏估计；对齐激励误差的“近似真诚性”分析；信息论/计数论的下界证明；以及对“可信路径边际”（truthful-path margin）的稳定性分析。

**📊 数据集**

实验使用了论文附录中给出的一个固定的三生产者、三上下文的合成实例，未使用公开真实数据集。

**📈 对比分析**

通过对比ETC与F-UCB在该实例上的福利退化曲线以及在ε‑网均衡（ε=0.1）下的策略竞价效果，实验显示：在短期内F-UCB由于支付估计噪声更大，退化略高；但随着T增大，F-UCB的退化趋近于O(√T)，显著优于ETC的O(T^{2/3})；并且在大规模情境下，F-UCB在均衡状态下的退化几乎与真诚F-UCB相同。

**⚠️ 局限性**

局限性包括：①机制仅在冻结支付框架下有效，无法突破退化-激励权衡边界；②需使用投标无关的探索期，且探索期长度对退化与激励误差有直接影响；③对策略竞价的分析依赖于ε‑网均衡，未覆盖更一般的博弈情况；④实验仅基于合成数据，缺乏在真实采购市场上的实证验证。

---

## 307. Image2Sim: Scaling Embodied Navigation via Generative Neural Simulator

**arXiv ID:** 2607.05765 | [PDF](https://arxiv.org/pdf/2607.05765v1)

**作者:** Zihan Wang `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Image2Sim，一个实时神经模拟框架，可将姿态RGB‑D图像序列转化为可交互的高质量3D场景，并自动生成视听语言导航数据；

**💡 创新点**

核心创新在于将3D空间锚定与光照渲染解耦，使用一次性特征高斯表示进行场景构造，并引入几何感知单步像素流渲染以完成未观测区域；

**🔧 技术方法**

采用基于 DINOv3 的双流特征编码、特征高斯生成、Alpha‑gated 单步像素流网络、基于卷积‑Transformer 的生成器、以及基于 voxel‑图的碰撞感知轨迹规划；

**📊 数据集**

利用 19,936 条来自 RealSee3D、Structured3D、ARKitScenes、HM3D、ScanNet、Gibson、Matterport3D 等真实与合成数据集，构建约 20K 交互场景，合成 10M 视听语言动作样本；

**📈 对比分析**

在 R2R‑CE、RxR‑CE、REVERIE‑CE 等基准上，使用 Image2Nav（基于 Qwen3‑VL‑4B）训练仅在 Image2Sim 上的模型，零样本迁移至 Habitat，获得 76.1% SPL、70.3% SR 等领先成绩，并在真实 Hello Robot Stretch 3 上实现路径跟随与目标导向的最高成功率；

**⚠️ 局限性**

局限性包括：渲染模型受限于实时性与模型容量，无法支持复杂接触动力学、可移动物体或人机交互；VLM 注释可能引入语言偏差；

---

## 308. Privilege and confidentiality in generative AI workflows

**arXiv ID:** 2607.05479 | [PDF](https://arxiv.org/pdf/2607.05479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 309. Withdrawability in Fiat-Shamir with aborts constructions

**arXiv ID:** 2607.05831 | [PDF](https://arxiv.org/pdf/2607.05831v1)

**作者:** Ramses Fernandez-Valencia `[一作]` `[通讯]`, Ramses Fernandez-Valencia

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍了IACR Communications in Cryptology期刊的LaTeX模板和使用指南。

**💡 创新点**

其创新点在于通过统一的元数据格式降低出版过程的人力成本，提升开放获取效率。

**🔧 技术方法**

主要使用iacrcc.cls LaTeX类文件及其默认字体等基础包，并对不允许使用的包做了说明。

**📊 数据集**

本文不涉及任何实验数据集。

**📈 对比分析**

暂无实验对比或性能评估，本文仅为模板说明。

**⚠️ 局限性**

模板仍处于开发阶段，存在对包的限制和需要遵循的提交规范。

---

## 310. A Task-Driven Evaluation of UAV Detection and Tracking under Synthetic Fog

**arXiv ID:** 2607.05467 | [PDF](https://arxiv.org/pdf/2607.05467v1)

**作者:** Amir Pouladi `[一作]` (University of Victoria), Afzal Suleman `[通讯]` (University of Victoria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的评估框架，结合基于深度的合成雾生成、图像去雾恢复、UAV检测与追踪，系统评估了不同条件下的性能。

**💡 创新点**

创新点在于：①提出了统一的任务驱动评估流程，能够同时衡量去雾效果与后端检测/跟踪表现；②利用MiDaS相对深度和大气散射模型实现可控的雾强度合成；③系统比较了多种去雾方法与不同雾暴露比例的训练策略，揭示了恢复与训练方式对检测/跟踪的相对贡献。

**🔧 技术方法**

技术包括：MiDaS单目深度估计、大气散射模型合成雾、DCP/CAP/FFA‑Net/DehazeFormer去雾模型、YOLO11检测器、ByteTrack/BoT‑SORT多目标跟踪器、PSNR/SSIM与mAP/MOTA/IDF1评估指标。

**📊 数据集**

使用的公开数据集有：MMAUD（用于去雾方法基准）、CfAR（UAV检测训练/测试）、DUT Anti‑UAV（视频跟踪评估），以及作者自行生成的多级雾化版本（公开发布）。

**📈 对比分析**

比较方法：对清晰训练、30%/50%/70%/100%雾训练的YOLO11模型，在清晰、雾化、去雾三种输入上分别评估mAP@0.5、mAP@0.5:0.95；追踪时使用ByteTrack/BoT‑SORT在清晰、雾化、去雾视频上评估MOTA与IDF1。结果显示：雾会显著提升误检率（主要是漏检），去雾能部分弥补漏检但未恢复至清晰水平；雾包含训练能显著提升鲁棒性并减小对去雾的依赖；跟踪表现主要受检测器鲁棒性驱动，追踪器差异次要。

**⚠️ 局限性**

局限性：仅在合成雾条件下验证，未包含真实雾、雨、低照度等真实恶劣天气；实验聚焦单摄像头RGB数据，未探讨多模态融合；去雾与检测/跟踪的计算成本与实时性未做系统性评估。

---

## 311. PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource Languages

**arXiv ID:** 2607.05992 | [PDF](https://arxiv.org/pdf/2607.05992v1)

**作者:** Daryna Dementieva `[一作]` (Technical University of Munich), Alexander Fraser `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了跨语言数学推理能力，扩展 PolyMath 至 18 种低资源语言，构建了人类验证的翻译管线；

**💡 创新点**

创新点在于完整的人工校正流程、跨 6 语言族覆盖 18 种低资源语言，并公开数据集与评测框架；

**🔧 技术方法**

采用机器翻译 + 人工校正的翻译管线，使用 Base、Base+EN-CoT、Backtranslated 三种提示策略，并对 27 个推理 LLM 进行评估；

**📊 数据集**

使用的主要数据集为 PluraMath（扩展 PolyMath）以及原始 PolyMath 题目；

**📈 对比分析**

通过精确匹配准确率与难度加权总分进行基准比较，结果显示高资源语言与低资源语言仍存在显著差距，较大模型在较短推理下表现更好；

**⚠️ 局限性**

局限性包括语言覆盖仍不完整、依赖现有 MT 资源、极低资源语言仍需完全人工翻译、仅单次生成评估未考虑多轮推理、未探究继续预训练或领域适配等。

---

## 312. A Sub-linear Low-Rank Solver for Poisson's Equation using Machine Learning Frameworks for GPU Acceleration

**arXiv ID:** 2607.06021 | [PDF](https://arxiv.org/pdf/2607.06021v1)

**作者:** Måns I. Andersson `[一作]` (Virginia Polytechnic Institute and State University), Daniel Appelö `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种利用Cross‑DEIM算法实现的子线性低秩Poisson求解器，并将其迁移至PyTorch GPU环境。

**💡 创新点**

创新点在于将统计Leverage Scores索引选择方法与Cross‑DEIM结合，显著降低了传统DEIM/QDEIM的开销，并通过批量DST实现无转置FFT，突破了高维求解的规模瓶颈。

**🔧 技术方法**

核心技术包括交叉逼近（Cross‑DEIM）、统计Leverage Scores索引选择、PyTorch JIT编译与Triton生成的GPU核、以及与MAGMA/BLAS等库的接口。

**📊 数据集**

使用了Hilbert矩阵、K矩阵（高阶光滑函数）以及在二维单位正方形上构造的低秩右端项的Poisson方程作为测试数据集。

**📈 对比分析**

与传统DEIM/QDEIM以及全秩DST求解器比较，实验显示在GPU上LS与QDEIM均可实现近似O(√N)的子线性时间，且在大规模（N≈2^18）问题上优于DEIM，CPU上LS亦具备竞争力。

**⚠️ 局限性**

主要局限包括：索引选择仍需大量QR/LS解算器，导致在极大规模或高秩场景下性能下降；算法非确定性导致结果波动；以及对GPU专用的QR/Pivoted SVD实现的依赖尚不完整。

---

## 313. Beyond the Syntax: Do Security Experts Trust LLMs for NIDS Rule Engineering?

**arXiv ID:** 2607.05916 | [PDF](https://arxiv.org/pdf/2607.05916v1)

**作者:** Lorenzo di Filippo `[一作]`, Fernando Kuipers `[通讯]` (Delft University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在安全运维场景下，系统化评估大规模语言模型（LLM）在网络入侵检测系统（NIDS）规则生成与工程中的可行性，并通过人机交互实验验证专家对LLM生成规则的语义正确性与可部署性。

**💡 创新点**

① 提出了基于LLM的规则工程框架，强调语法校验、自动纠错和人机交互循环；② 系统性揭示“语法‑语义悖论”，指出LLM虽然能生成语法正确的规则，却因缺乏细粒度特异性与存在逻辑幻觉而难以直接投入生产；③ 通过专家评估和可用性量表，首次量化专业安全分析师对LLM自动生成规则的信任与使用偏好。

**🔧 技术方法**

采用多模型LLM（DeepSeek‑R1 70B、Llama‑3.1/3.3 70B、Qwen‑2.5 32B 等）结合角色与少量样本提示，嵌入语法校验器与自动纠错循环，并提供人机对话界面与可视化网络拓扑、CVE/PoC 信息；评估使用系统使用者的 SUS 分数和交互日志。

**📊 数据集**

使用合成网络（SS、SM）和来自 CIC‑IDS 2017 数据集的真实网络作为测试拓扑；结合公开 CVE 目录、CPE 关联、PoC exploit 以及自建的网络资产清单作为输入；规则生成量化统计来源于每个 LLM 30 次实验产生的规则。

**📈 对比分析**

通过统计分析（语法有效率、需要纠错比例、PoC 对比）和用户研究（规则可部署率、信心等级、SUS 评分）两重方法比较。结果显示：≥70B 参数模型语法有效率可达 90%，但专家认为仅 37.5% 规则语义上可直接部署；小模型（≤4B）几乎无可用规则。SUS 平均分 67，表明界面可用，但专家仍倾向于将LLM视为辅助工具而非全自动生成器。

**⚠️ 局限性**

限制：LLM 仍缺乏对协议细节、位置偏移、上下文关联等的深度理解，导致规则特异性不足、易产生假阳性；依赖大量上下文时受限于模型窗口；存在逻辑幻觉；需要人机交互进行纠错，无法完全替代人工规则工程；实验规模受限于专家人数和可用计算资源。

---

## 314. Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries

**arXiv ID:** 2607.06125 | [PDF](https://arxiv.org/pdf/2607.06125v1)

**作者:** Raafat Abualazm `[一作]` (Cairo University), Amr G. Wassal `[通讯]` (Cairo University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估 Dart AOT 神经反编译的微调效果与评估指标，比较六种微调模型在三种 4B–8B 基础架构上的表现，使用 CodeBLEU、编译成功率与功能正确率（HumanEval‑Dart）三指标；

**💡 创新点**

提出 HumanEval‑Dart benchmark 与 Dart 版 CodeBLEU，揭示表面指标与功能正确率存在显著分离，且微调效果随模型规模呈负向依赖，揭示跨语言干扰随规模减弱；

**🔧 技术方法**

采用 LoRA + DoRA 参数高效微调，基于 Qwen3、DeepSeek 等 4B–8B 大模型，利用 CodeBLEU、compile@k、pass@k 等指标并进行 McNemar、bootstrap 等统计检验；

**📊 数据集**

使用 RosettaCode 自然 Dart 函数、通过多 LLM 生成的 Dart 与 Swift 语料（约 1.2k 对）以及 154 题 HumanEval‑Dart 单元测试集；

**📈 对比分析**

对比结果显示：无微调配置在 pass@k 上无显著提升（最佳 +0.71pp，p=0.21），最强基模型 Qwen3‑8B pass@k 6.36%，4B 交叉语言干扰 -2.66pp，表面指标提升但功能正确率下降，整体功能正确率仍低；

**⚠️ 局限性**

局限包括样本量有限、仅 4B–8B 范围、Dart AOT 与 Swift 未匹配优化导致干扰混杂、LoRA 超参未全盘搜索、仅 x86‑64 架构、真实 Flutter 应用场景缺乏评估。

---

## 315. Why does AI unlock new possibilities in STEM education? A Bibliometric Analysis of Trends and Future Agenda

**arXiv ID:** 2607.05412 | [PDF](https://arxiv.org/pdf/2607.05412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 316. KOAL: Knowledge-Driven Prostate Cancer Grading with Ordinal-Aware Learning

**arXiv ID:** 2607.06019 | [PDF](https://arxiv.org/pdf/2607.06019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 317. What Do AI Agents Actually Change? An Empirical Taxonomy of Mutation Patterns in Performance-Improving Pull Requests

**arXiv ID:** 2607.05666 | [PDF](https://arxiv.org/pdf/2607.05666v1)

**作者:** Illia Dovhoshliubnyi `[一作]` (Edinburgh Napier University), Alexander Brownlee `[通讯]` (University of Stirling)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 AI 代码生成器提交的性能优化 Pull Request 进行差分块的语法突变分类，探究其突变模式及其与代理系统和优化策略的关联。

**💡 创新点**

首次为多语言 AI 生成器的性能 PR 建立实证突变分类，发现其突变模式与传统遗传改进（GI）语料库显著不同，并为基于代理的搜索式软件工程提供数据驱动的运算符选择先验。

**🔧 技术方法**

使用两轮大语言模型（LLM）交叉判定突变类别，构建基于18类语法突变的分类器，并结合交叉验证与多标签交集筛选提高准确率。

**📊 数据集**

基于 AIDev-pop 子集，共 33,596 条 PR，其中 324 条带性能标签，最终提取 1,254 个性能相关差分块进行分类；涉及 5 个 AI 代码生成器（Devin、Copilot、Cursor、Codex、Claude Code）和 100 个受星标的仓库。

**📈 对比分析**

与原始 GI 语料库的突变分布做对比，发现本研究中 name_modification、object_creation、type_change 共同占比 86%；同时绘制突变与优化策略及代理身份的共现矩阵，表明不同代理和策略激活的突变子集差异显著，提示可按上下文限制运算符空间；在实验环境中，交叉 LLM 判定准确率约 67.5%，比随机更高。

**⚠️ 局限性**

局限包括：突变分类来源于 Java 语料，可能遗漏跨语言的特殊变异；LLM 判定准确率有限，导致结果为保守下限；性能标签依赖外部标注，可能存在误标；代理与语言生态高度相关，难以泛化至其他系统；数据集规模虽大但性能 PR 仅占不到 1%，样本稀缺。

---

## 318. A Lower Bound for Read-Once Parity Branching Programs

**arXiv ID:** 2607.05944 | [PDF](https://arxiv.org/pdf/2607.05944v1)

**作者:** Ben Lee Volk `[一作]` `[通讯]` (Reichman University), Ben Lee Volk (Reichman University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了读一次奇偶分支程序（read‑once parity branching programs）在计算一个显式布尔函数时的大小下界至少为Ω(n²/ log²n)；

**💡 创新点**

创新点在于将布尔模型的下界问题转换为多线性代数分支程序的下界，通过“algebrization”将奇偶分支程序映射为多线性多项式，从而利用已知的代数复杂度下界；

**🔧 技术方法**

使用的技术包括多线性代数分支程序（ABP）与多线性电路的对应关系、Alon‑Kumar‑Raz等人的代数下界结果，以及动态规划算法对特定多项式的高效计算；

**📊 数据集**

由于研究是理论性质，没有使用真实数据集，而是通过构造的显式多项式和布尔函数；

**📈 对比分析**

与之前针对读一次分支程序的上界/下界相比，该工作将已知的最优上界从O(n^{3/2})提升到几乎最佳的Ω(n²)，证明了该模型的复杂度接近上界；

**⚠️ 局限性**

局限性在于下界仍为多项式级，尚未突破到指数级；此外，方法依赖于代数下界技术的限制，无法进一步提升至更高阶（如Ω(n³)）或非多项式级。

---

## 319. Uncertainty-Aware Cross-Modal Remote Sensing Image-Text Retrieval via Evidential Learning

**arXiv ID:** 2607.06032 | [PDF](https://arxiv.org/pdf/2607.06032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 320. MCP-Enabled Agentic AI for Autonomous IPoDWDM Network Lifecycle Automation

**arXiv ID:** 2607.05975 | [PDF](https://arxiv.org/pdf/2607.05975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 321. AI tools in Arab University English classrooms: Looking back and forward

**arXiv ID:** 2607.05403 | [PDF](https://arxiv.org/pdf/2607.05403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 322. SplineNet: An Isogeometric Deep Learning Method for Complex Shells

**arXiv ID:** 2607.06026 | [PDF](https://arxiv.org/pdf/2607.06026v1)

**作者:** Shizhou Luo `[一作]` (Shanghai Jiao Tong University), Xiaodong Wei `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于等几何深度学习的SplineNet框架，可在单实例与算子学习两种模式下对复杂薄壳结构进行设计与分析。

**💡 创新点**

创新点包括：1）通过Bézier提取将无结构T-样条（ASUTS）直接嵌入神经网络，实现在CAD/CAE一体化的几何描述；2）在单实例模式下使用能量形式的物理约束，避免高阶导数问题；3）在算子学习模式下将SplineNet作为DeepONet的trunk网络，提升解释性并保持与传统IGA同一离散空间。

**🔧 技术方法**

技术实现包括：ASUTS与Bézier提取、Kirchhoff–Love薄壳能量形式、物理信息神经网络（PINN）与能量损失、自动微分、DeepONet架构、PyTorch训练框架。

**📊 数据集**

数据集主要为合成薄壳几何（Scordelis–Lo屋顶、B‑pillar、机身鼻锥）与由高斯随机场产生的加载函数；训练集1000份，测试集250份；所有答案均由内部IGA求解得到。

**📈 对比分析**

与传统IGA参考结果对比：P‑SNet在Scordelis–Lo屋顶上的自由端中点位移相对误差低于1%；O‑SNet在屋顶、B‑pillar、机身鼻锥三种结构上，GP点的平均L²误差约1–3%，EC点误差约2–4%；训练曲线表明无明显过拟合，模型在不同几何上表现一致。

**⚠️ 局限性**

局限性包括：1）主要验证线性或简单非线性KL薄壳；2）算子学习仍需大量IGA数据进行训练；3）对更大规模问题、接触、复杂材料等扩展待进一步研究；4）依赖ASUTS网格生成，需保证其质量；5）与其他算子学习方法的定量比较尚不充分。

---

## 323. EquiFiLM: Charge-Conditioned Equivariant Force Fields via Feature-wise Linear Modulation

**arXiv ID:** 2607.05559 | [PDF](https://arxiv.org/pdf/2607.05559v1)

**作者:** Samuel Sahel-Schackis `[一作]` (Stanford University), Thomas Linker `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 EquiFiLM 轻量级适配器，在任何等变基础 MLFF 上加入连续外部条件（如总电荷），实现对带电水的高精度预测。

**💡 创新点**

创新点在于仅对标量通道做 Feature-wise Linear Modulation，保持 E(3) 等变性且无需重新训练基础网络，极大提升数据效率并可扩展至多种外部变量。

**🔧 技术方法**

技术包括 FiLM 块、基于 MACE-MatPES 的等变消息传递、两层 MLP 产生 γ、β 参数、少量 AIMD 训练数据与 SWA 训练策略。

**📊 数据集**

使用四条 AIMD 轨迹（q=0,6e,10e,16e）在 r^2SCAN meta‑GGA 下的约 6,400 帧数据。

**📈 对比分析**

与未加 FiLM 的 MACE‑MatPES 以及专门的 MACE‑POLAR‑1‑M 对比：在训练电荷上，E‑MACE 将力 RMSE 降至 6.96 meV/Å（约 3×提升），能量 RMSE 0.10 meV/atom；在未见电荷的插值/外推上仍保持 18–61 meV/Å 力误差；推理时间与基础模型相当。

**⚠️ 局限性**

局限性在于需依赖基础网络已覆盖化学空间；仅处理连续单一标量变量，无法直接处理离散或多维条件；对极端条件的泛化仍未充分验证。

---

## 324. FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning

**arXiv ID:** 2607.05780 | [PDF](https://arxiv.org/pdf/2607.05780v1)

**作者:** Chuhao Zhou `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种两阶段的策略 FUNCTIO(NAL REASONING AND GROUNDED EXECUTION) (FORGE)，通过中间表征实现工具功能的泛化。

**💡 创新点**

核心创新在于将功能推理与动作执行解耦，使用 2D 关键点轨迹作为功能中间表征，并在无标签数据中学习可迁移的功能意图，随后仅用少量标注数据实现动作落地。

**🔧 技术方法**

采用基于流匹配的条件模型（flow‑matching）预测关键点轨迹和动作轨迹；利用 SAM2 进行工具分割、CoTracker 跟踪、FPS 采样以及关键点编码器；在执行阶段加入随机像素扰动以提升鲁棒性。

**📊 数据集**

利用七工具击打基准（7 个工具、3 种初始设定）在仿真中收集 630 条示例，现实环境中在 Franka 机器人上收集 120 条手工演示；同时使用大量无动作标签的视频数据作为功能推理训练集。

**📈 对比分析**

与 Flow‑Matching、Diffusion Policy 和 ATM 等基线对比，FORGE 在未见工具上的平均成功率提升至 0.36（相较于基线 0.08‑0.17），在真实世界书本工具上亦显著优于 FM；体现了约 2 倍的性能提升。

**⚠️ 局限性**

局限性包括需预先指定击打点和目标点、未考虑抓取过程、仅使用 2D 关键点轨迹导致对精细空间对齐不够、对 3D 结构缺乏感知与建模。

---

## 325. Retrieving a Set, Not Independent Passages: Set-Level Compatibility Learning for Efficient Set Exploration

**arXiv ID:** 2607.05712 | [PDF](https://arxiv.org/pdf/2607.05712v1)

**作者:** Mooho Song `[一作]` (Seoul National University), Jay-Yoon Lee `[通讯]` (Seoul National University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将多跳检索建模为查询-集合兼容性评分，并设计轻量级集合探索与交叉编码重排的检索框架。

**💡 创新点**

引入面向集合的兼容性学习目标，以及基于bi-encoder的轻量级ParaSet与基于交叉编码的SetCE两阶段检索，兼顾效率与质量。

**🔧 技术方法**

使用margin-based ranking loss进行集合兼容性学习，轻量化多头自注意力集合评分（ParaSet），交叉编码集合重排（SetCE），以及Beam Search集合搜索。

**📊 数据集**

在HotpotQA、2WikiMultihopQA和MuSiQue三个多跳QA数据集上，结合Contriever和Qwen3-Embedding作为首检索器进行实验。

**📈 对比分析**

与传统Bi-encoder、文档级交叉编码（CE）和序列交叉编码（ListCE）对比，SetCE与ParaSet+SetCE在EM/F1上均有提升，且在加上CE_5后进一步提高，且在速度上比全交叉编码快约10–40倍。

**⚠️ 局限性**

搜索策略过于简单（仅使用beam search和直接联合），未对预算或集合组合进行自适应优化；在长文档、噪声更大或更大规模候选集的场景下可扩展性尚待验证。

---

## 326. On the Communication Complexity of Maximum Matching and Negative-Weight Shortest Paths

**arXiv ID:** 2607.05751 | [PDF](https://arxiv.org/pdf/2607.05751v1)

**作者:** Yu Cheng `[一作]` (Brown University), Huacheng Yu `[通讯]` (Princeton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在两方通信模型下，提出了针对一般图最大匹配、负环检测、负权单源最短路以及二分图匹配的低通信复杂度协议。

**💡 创新点**

创新在于用简单的贪心+Blossom模拟实现 O(n^3/2) 通信复杂度的最大匹配协议；利用顶点势多面体和裁剪法直接得到 O(n) 通信的负环/SSSP协议；以及将连续裁剪法离散化得到组合式 O(n) 二分匹配协议。

**🔧 技术方法**

采用两方通信模型、图的割法、顶点势多面体、裁剪法、Tutte-Berge公式、Blossom 算法以及离散化的可行点计数。

**📊 数据集**

无实验数据集，纯理论分析。

**📈 对比分析**

与之前的 O(n log^2 n) 或更高通信复杂度方法相比，本工作实现了更接近线性的通信量，并给出了明确的上界证明。

**⚠️ 局限性**

仍需改进一般图匹配的通信复杂度以达到 O(n log^k n)；方法主要为确定性，随机化可能进一步降低通信量；对大规模图的实用性尚未验证。

---

## 327. Quantifying and Expanding the Theoretical Capacity of Late-Interaction Retrieval Models

**arXiv ID:** 2607.05803 | [PDF](https://arxiv.org/pdf/2607.05803v1)

**作者:** Julian Killingback `[一作]` (University of Massachusetts Amherst), Cameron Musco `[通讯]` (University of Massachusetts Amherst)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文从理论和实验两方面系统地研究了后期交互模型中常用的 MaxSim 相似度，证明其在非负稀疏向量上可以完美复制内积，并指出单向量内积无法实现同等压缩；进一步扩展出 Signed MaxSim 能够复制任意实值内积，并证明 MaxSim 能够等价评估正 CNF 逻辑表达式；最后在包含否定查询的 synthetic LIMIT 数据集上验证了 Signed MaxSim 的显著性能提升。

**💡 创新点**

创新点主要包括：①用 O(k) 个 3 维向量构造能够完全重现非负稀疏向量内积；②证明标准 MaxSim 无法复制实值内积，并提出 Signed MaxSim 通过分离幅度与符号实现完整复制；③证明 MaxSim 能够聚合 Soft‑OR 并等价评估正 CNF 逻辑；④将这些理论转化为模型改进，并在多种数据集上展示显著效果。

**🔧 技术方法**

技术上采用了多向量嵌入、二次多项式映射与最大化聚合，构造 Signed MaxSim 的符号向量与标量后在对比损失下训练；实验使用 ModernBERT 作为骨干、五层 MLP 投影、SoftMax 温度学习以及全量检索评估；对比方法采用标准 ColBERT（MaxSim）模型，评价指标为 nDCG@10、P@10 与 AP。

**📊 数据集**

数据集：生成的 synthetic LIMIT‑style 数据集，分为 In‑Domain、Different Vocabulary、Negation‑Only 三个评测集，每个集包含 2,000 个查询与 100k 文档，训练集为 100k 查询与 200k 文档。

**📈 对比分析**

与 ColBERT（MaxSim）在同一训练框架下对比，Signed MaxSim（Fallon）在所有三组评测集上均取得显著提升：在 In‑Domain nDCG@10 由 0.982 提升到 0.997，在 Different Vocabulary 从 0.597 提升到 1.000，在 Negation‑Only 从 0.008 提升到 0.788；P@10 与 AP 也均有显著改善。

**⚠️ 局限性**

局限性：标准 MaxSim 仍无法处理负值，Signed MaxSim 需要额外学习符号信息，理论证明仅针对与稀疏度相匹配的嵌入数量，未探究解耦情况；实验结果仍受检索范围限制（仅 top‑1000），对更大规模真实数据集的推广需要进一步研究。

---

## 328. Claimed or Attested? A Commit-Signature Dataset and Identity Trust Tiers across the World of Code

**arXiv ID:** 2607.06194 | [PDF](https://arxiv.org/pdf/2607.06194v1)

**作者:** Audris Mockus `[一作]` `[通讯]` (University of Tennessee), Audris Mockus (University of Tennessee)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 World of Code 上构建并发布了第一份 commit‑signature 轴，提供每个提交的签名映射、key‑to‑author 图和身份认证层次。

**💡 创新点**

首次将签名视为精度锚点而非覆盖层，并将组织/CI 密钥与个人密钥区分，完善身份认证体系。

**🔧 技术方法**

通过扫描 WoC commit 表中的签名字段，解析 PGP/SSH/X.509 证书，生成签名映射与 key‑graph。

**📊 数据集**

基于 5,866,595,698 条提交的 World of Code 数据集，提取了 1,031,721,316 条带签名提交。

**📈 对比分析**

报告签名占比 17.59%，PGP 98.96%，指出签名提交主要聚焦安全开发者，未与其他方法做直接性能比较。

**⚠️ 局限性**

受限于仅覆盖签名提交，偏向安全意识开发者，覆盖面有限且未深入分析不同签名类型或提供基准对比。

---

## 329. REVIVE: A Multi-Modal Framework for Vandalism Detection and Recovery in Autonomous Vehicles

**arXiv ID:** 2607.05649 | [PDF](https://arxiv.org/pdf/2607.05649v1)

**作者:** Abdullah Tariq Choudhry `[一作]` (University of Pacific), Tapadhir Das `[通讯]` (University of Pacific)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了REVIVE框架，用于自动驾驶车辆摄像头图像的破坏检测、定位与恢复。

**💡 创新点**

提出了类型感知的恢复策略，将不同遮挡模式路由到最合适的恢复子模块，并引入参考可用的质量门控。

**🔧 技术方法**

结合二分类与多分类检测、基于EfficientNet的U‑Net分割、BLIP+Stable Diffusion生成性修复、直接像素替换与自适应中值滤波等多种技术。

**📊 数据集**

在BDD100K上人工合成五类遮挡样本进行训练与评估，并在Raindrop-on-Windshield数据集进行几何形状压力测试。

**📈 对比分析**

与未修复、LaMa、Telea、Navier‑Stokes、Median Filter、Stable Diffusion等基线对比，最优方案直接像素替换在可用参考下召回率达0.967，SSIM 0.988；在无参考情况下LaMa在保持感知召回率0.667的同时，恢复延迟仅170ms。

**⚠️ 局限性**

受限于合成遮挡、对对齐参考的依赖、缺乏真实破坏数据、质量门控依赖真实帧、以及稳定扩散的高延迟等因素。

---

## 330. Packet Routing for the Quantum Internet

**arXiv ID:** 2607.06075 | [PDF](https://arxiv.org/pdf/2607.06075v1)

**作者:** Robert Malaney `[一作]` `[通讯]` (University of New South Wales), Robert Malaney (University of New South Wales)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种在IPv6扩展头中嵌入量子路由与量子纠缠传输的设计框架，使经典网络设备能够无缝兼容量子通信。

**💡 创新点**

创新点在于利用现有的IPv6扩展头机制（如Hop-by-Hop、Routing扩展头）来实现量子叠加路径、量子多路复用、量子传输与超位置等功能，从而在不大幅改动现有协议栈的前提下引入全新的量子网络行为。

**🔧 技术方法**

技术主要包括：① 设计新的“量子路由头”与“量子传输头”，定义字段（类型、时间、路径列表、量子多路复用等）；② 通过这些头在IP层携带控制信息，指示路由器执行量子态存储、叠加路径、纠缠交换与传输；③ 结合时序同步与量子内存技术，保证光子/量子态在网络节点间的时序一致性。

**📊 数据集**

文中未使用任何实验或真实数据集，主要以理论设计与标准化建议为主。

**📈 对比分析**

暂无实验或数值比较；作者仅指出该设计在兼容性、可扩展性和未来标准化路径方面的优势，但未给出性能评估或基准测试。

**⚠️ 局限性**

局限性包括：① 仍属理论与规范设计，缺乏实际实现与验证；② 需要专用量子内存与硬件支持；③ 依赖网络时序同步与纠缠分发协议，尚未给出完整实现细节；④ 需通过IETF与IANA等标准化流程才能真正投入使用。

---

## 331. Orthogonal Dendritic Intrinsic Networks: An Architecture for Significance-Ordered, Orthogonal Latent Spaces

**arXiv ID:** 2607.05653 | [PDF](https://arxiv.org/pdf/2607.05653v1)

**作者:** Jeanie Schreiber `[一作]` (George Mason University), Zeeshan Ahmed `[通讯]` (NIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了ODIN（Orthogonal Dendritic Intrinsic Network）自编码器，用以在非线性模型中恢复PCA的正交且按方差排序的潜在空间，并通过实验验证其可解释性和稳定性。

**💡 创新点**

核心创新在于：1) 树枝式（dendritic）解码机制，使每个潜在维度按解释方差逐层加入，天然实现重要性排序；2) 直接在训练目标中加入正交约束，强制潜在向量相互正交；3) 通过理论证明在线性极限下ODIN等价于PCA，提供了从结构到非线性泛化的严格桥梁。

**🔧 技术方法**

使用的技术包括：深度自编码器结构、树枝式多重解码、正交正则化（通过协方差矩阵的非对角项惩罚）、均方误差重构损失、Adam优化、批量移动平均、以及对比的β-VAE、POLCA、AEO等现有方法。

**📊 数据集**

实验数据集包括：1) 合成3D高斯点云（椭球体），用于验证PCA等价性；2) MNIST手写数字（仅数字1和2），评估非线性可解释性；3) NV钻石光谱（1340维光谱），真实科学数据，用于评估温度预测与物理意义。

**📈 对比分析**

与标准AE、β-VAE、POLCA等方法对比：ODIN在潜在维度排序一致、重构误差随维度增加单调下降、温度预测R²在前两维已达0.8以上，整体重构误差与其他方法相当甚至更优；在点云实验中与PCA完全对齐；在MNIST实验中潜在维度间的相似性高达0.999，证明了可重复性和可解释性；在光谱实验中ODIN的第一、二维分别对应平均光谱和温度变化，显著提升了特征可解释性。

**⚠️ 局限性**

局限性：1) 对极其高度非线性的数据，树枝式结构与正交约束仍可能不足以完全捕捉复杂几何；2) 需要手动设定正交惩罚系数λ，虽然对比β-VAE更稳健，但仍需实验调参；3) 在大规模高维数据上，树枝式多重解码会增加计算负担；4) 目前仅实现无监督训练，未探索半监督/监督下的进一步分解能力。

---

## 332. Signed-Graph Recommendation as Structural Consistency Maximization

**arXiv ID:** 2607.05952 | [PDF](https://arxiv.org/pdf/2607.05952v1)

**作者:** Zifan Wang `[一作]` (Northeast Normal University), Wenzhuo Song `[通讯]` (Northeast Normal University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个闭环框架SSC-Loop，用于提升带符号社交关系的推荐效果；

**💡 创新点**

创新点在于将结构一致性、传播一致性与语义一致性统一为可协同优化的闭环过程，并引入ESA-DA、P/N/O传播以及对比学习来实现自适应图结构改进、符号保持的高阶消息传递和语义对齐；

**🔧 技术方法**

使用的技术包括自适应符号图编辑（ESA-DA）、正负/中立三通道聚合的符号高阶传播、对比关系对齐损失，以及基于EM风格的迭代闭环训练；

**📊 数据集**

采用的实验数据集为Epinions（含用户-物品评分与正负信任关系）和Slashdot（以友/敌关系构建的衍生链接预测任务）；

**📈 对比分析**

与传统社交/符号推荐模型（SocialMF、TrustMF、TDRec、RecSSN）以及图神经网络模型（LightGCN、GraphRec、SIGformer）进行对比，SSC-Loop在Epinions上RMSE从0.4658降至0.4398、MAE从0.3090降至0.2489，显著优于所有基线；在Slashdot上也取得最佳RMSE并位居第二MAE；

**⚠️ 局限性**

局限性包括对大规模稠密图的计算复杂度尚未充分评估、缺乏对时序、多模态符号关系的直接支持，以及闭环迭代过程的收敛性理论保障不充分。

---

## 333. LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability

**arXiv ID:** 2607.06157 | [PDF](https://arxiv.org/pdf/2607.06157v1)

**作者:** Chenxu Wang `[一作]` (Tsinghua University), Huaping Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将推理式协作定义为部分可观测的联合决策问题，构建了包含菜单设计与任务分配的可扩展基准，并在此基准上系统评估了多种 LLM 的协同决策能力；

**💡 创新点**

创新点在于提出统一的推理式协作抽象、设计多任务结构相似但观察与决策机制多样的基准，并对对话、信息整合、工具使用等因素进行细粒度诊断；

**🔧 技术方法**

使用 LLM 代理（通过提示工程与链式思考）、外部符号工具（整数规划求解器与计算器）、分布式对话协议、集中式与 oracle 基线比较；

**📊 数据集**

采用自己生成的数据库：菜单设计包含 51 种食材、78 道菜、10 位客人；任务分配包含 42 个任务模板、12 种私有/公共资源、18 位代理；每个域生成 60 个实例；

**📈 对比分析**

通过自我对话实验，在 6 种 LLM 上计算标准化奖励（NR）与有效率（VR），与集中式基线和 oracle 进行对比；大型模型在菜单-数值域可达 NR>90，复杂域仍有提升空间；对话可提升或偶降奖励，工具对某些模型有效但不通用；

**⚠️ 局限性**

局限性包括基准依赖固定的参考框架、评估模型与场景有限、未包含人类对照、未进行参数敏感性或对话预算等全面分析；性能易受提示与工具接口影响。

---

## 334. Stability Annealing Selects the Implicit Bias of Smoothed Sign Descent: A Rate-Indexed Barrier Path on Separable Data

**arXiv ID:** 2607.06013 | [PDF](https://arxiv.org/pdf/2607.06013v1)

**作者:** Xiangwu Wang `[一作]` (University of Hong Kong), Peilin Yu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了在可分数据上进行全批线性分类的速率控制中间情况，特别是针对无记忆稳定性退火平滑符号下降法的加权指数损失，证明了归一化迭代收敛到凸Burg型障碍的最小化器。

**💡 创新点**

创新点在于提出了一个精确的速率索引隐性偏差定理，并识别了限制分隔符作为Burg障碍的最小化器，同时对端点几何进行了全面表征。

**🔧 技术方法**

使用了无记忆稳定性退火的平滑符号下降法，结合了KL递归和熵镜像上升的动态证明方法。

**📊 数据集**

实验使用了合成的可分线性分类数据集，包括各类随机可分数据、受控支持向量数据和相关的病态数据。

**📈 对比分析**

通过与传统的梯度下降法和Adam等自适应方法进行比较，展示了所提出方法的收敛性和性能，结果表明在浮点误差范围内验证了精确的对偶身份，并展示了预期的路径和速率图。

**⚠️ 局限性**

限制在于定理仅适用于加权指数损失和无记忆稳定性退火的平滑符号下降，且在κ=0和κ>的端点情况、临界情况κ=、逻辑扰动理论、固定-ϵ尾界限和完整的Adam转移需要单独的论证。

---

## 335. Decision Protocols in Multi-Agent Large Language Model Conversations

**arXiv ID:** 2607.05477 | [PDF](https://arxiv.org/pdf/2607.05477v1)

**作者:** Lars Benedikt Kaesberg `[一作]` `[通讯]`, Lars Benedikt Kaesberg

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了多代理语言模型框架MALLM，探究不同决策协议（投票、共识、法官）在多领域对话任务中的效果与计算成本。

**💡 创新点**

创新点在于：①首次系统比较三类决策协议在七个不同知识与推理任务上的表现；②提出通过多代理讨论提升答案多样性并验证其对性能的影响；③探讨信息量变化对投票阶段的鲁棒性；④设计挑战实验评估代理对最终答案的自我校正能力。

**🔧 技术方法**

主要技术包括多代理讨论框架MALLM、Llama 3 8B/70B LLM、Chain‑of‑Thought（CoT）提示、投票/共识/法官决策协议、讨论范式（Memory、Relay、Collective Refinement 等）以及自适应提示与信息增强（ContextPlus）。

**📊 数据集**

使用的公开数据集：MMLU、MMLU‑Pro、GPQA、StrategyQA、MuSR、Math‑lvl‑5、SQuAD 2.0，按知识型与推理型划分，采用抽样策略确保95%置信区间。

**📈 对比分析**

通过与单一代理基线（无CoT）和CoT基线对比，实验显示：对知识型任务共识协议表现最佳；对推理型任务投票/法官更优；小模型在多代理讨论中获益更显著；增加答案多样性提升精度；信息量变动对投票决策影响微乎其微；挑战实验表明讨论历史能显著降低质疑率。计算成本方面，投票/法官协议约比基线高10–15倍，共识协议仅约5倍。

**⚠️ 局限性**

局限性包括：①计算成本高，限制了实验规模与参数空间；②仅使用三轮讨论与有限代理数量；③抽样样本与标准差估计无法覆盖全部可能组合；④对不同模型规模和新兴推理预训练模型的适用性未作充分验证；⑤评估指标主要聚焦准确率，未深入探讨多模态或跨语言场景。

---

## 336. Bounded-Memory Parallel Image Pulling for Large Container Images

**arXiv ID:** 2607.05596 | [PDF](https://arxiv.org/pdf/2607.05596v1)

**作者:** Sri Saran Balaji Vellore Rajakumar `[一作]` (Amazon Web Services), James Thompson `[通讯]` (Amazon Web Services)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在容器运行时改用磁盘写入的并行拉取策略，消除内存中块顺序缓存的需求，保持内存占用不随镜像大小扩展。

**💡 创新点**

创新点是将块级拉取直接写入文件偏移量，既保持了 OCI 规范的完整性检查，又通过并行校验和解压缩实现两通道验证，显著降低峰值内存而不牺牲吞吐。

**🔧 技术方法**

核心技术包括 HTTP Range 请求并行下载、磁盘直接偏移写入、文件级 SHA-256 与 DiffID 并行校验、SOCI snapshotter 插件实现和标准 OCI 镜像格式兼容。

**📊 数据集**

使用了五个生产规模 AI/ML 镜像（如 rocm/pytorch-training:48.5 GiB、nvidia/nemo:31.4 GiB、rocm/megatron-lm:48.5 GiB 等）以及两份合成镜像（5 GiB 与 50 GiB）进行基准测试。

**📈 对比分析**

与 containerd 2.2 的内存顺序重组方式在同一环境下对比，结果显示 DBPP 的拉取时间在 12% 范围内相等，峰值内存降低 8.7–25.3×，且在 7.6 GiB RAM 的节点上容器运行时不会 OOM。CPU 使用基本相同，唯一差异是额外的压缩层校验读。

**⚠️ 局限性**

局限包括：对高并发写磁盘的随机写要求，需 NVMe/SSD；对小镜像或稀疏访问工作负载效果有限；仅在单一注册中心（ECR）评估；与容器运行时的实现差异无法完全消除；仍受单层单核解压缩瓶颈制约。

---

## 337. From Bit to Block: Capacity Achievement via Product Coding

**arXiv ID:** 2607.05816 | [PDF](https://arxiv.org/pdf/2607.05816v1)

**作者:** Bin Zhang `[一作]` `[通讯]` (Beihang university), Bin Zhang (Beihang university)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过将行码（具有可趋近于0的比特误码概率）与列码（高率、可纠正一定错误数量的有界距离码）组合成产品码，证明在任意二进制记忆无关对称信道（BMS）上，能将比特级可靠性提升为块级可靠性，并构造了一个基于RM行码与BCH列码的产品码序列，该序列在容量附近的速率下实现了零块误码概率。

**💡 创新点**

创新点在于：①首次利用Forney的产品编码框架，将比特误码率向块误码率转化；②给出了通用的产品码构造定理和充分条件；③给出具体RM–BCH实例，展示如何在容量附近实现零块误码概率；④提供了从行码比特误码估计到列码纠错阈值的设计准则。

**🔧 技术方法**

主要技术包括：产品码的行-列解码策略；比特误码概率与列码纠错半径的比较；使用Chernoff大偏差上界控制列误码事件；对RM码比特误码的上界与RM率网格分析；对BCH码冗余与纠错能力的评估。

**📊 数据集**

无具体数据集。论文为理论研究，所用的“数据”是信道模型（任意BMS）和码率参数（RM、BCH）。

**📈 对比分析**

评估方法：通过解析证明块误码概率上界趋于0，证明在任何目标速率小于信道容量时都能实现容量。与现有的容量实现方案（如极化码、张量RM码、Berman码）相比，所构造的产品码在理论上同样实现容量，但提供了更通用的构造思路。实际性能取决于行码的比特误码速度和列码长度，理论上可达到无误码率。

**⚠️ 局限性**

限制：①需要行码比特误码概率足够小且列码纠错半径大于该概率；②列码长度必须足够大以满足大偏差指数对数列的条件，导致块长度显著增大；③对具体行码的比特误码估计要求严格（如RM的精确误码上界），否则无法保证块误码消失；④实现复杂度受产品码维度和行列解码器的影响，实际编码/译码难度较高。

---

## 338. TypeGo: An OS Runtime for Embodied Agents

**arXiv ID:** 2607.05482 | [PDF](https://arxiv.org/pdf/2607.05482v1)

**作者:** Guojun Chen `[一作]` (Yale University), Lin Zhong `[通讯]` (Yale University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种操作系统风格的运行时，使用多时钟层异步规划、资源仲裁的Skill Kernel和自然语言编程接口，让大型语言模型在机器人控制中实时、高效且可并发执行；

**💡 创新点**

创新点包括：①将LLM推理与执行解耦的四层异步循环（S0–S3）和speculative skill streaming；②用OS类的资源调度与语义优先级调度替代传统的单一决策流程；③将自然语言作为一等编程语言，实现任务描述与快速反应规则的无代码编写；④在真实四足机器人上验证并展示显著的延迟与吞吐改进；

**🔧 技术方法**

主要技术：大型语言模型（GPT‑5.4、GPT‑OSS‑120B）、ROS2、Python实现的Skill Kernel与调度器、视觉感知模型（YOLO、OmDet、CLIP）、语音合成、边缘服务器推理等；

**📊 数据集**

实验使用自定义的小型任务集（T1–T8），涵盖命令式、开放式、并发与反应性任务，未采用公开公开数据集；

**📈 对比分析**

与ReAct和Plan‑and‑Execute两种基线进行对比，使用成功率、首动作时间（TTFA）、总耗时、步间延迟和token使用量等指标。实验表明，Kalos在TTFA降低40%（相较ReAct）/73%（相较PAE），总时长下降、步间延迟减少50%，但token消耗显著增大；

**⚠️ 局限性**

局限性包括：①高token消耗导致成本与实时性权衡尚未优化；②缺乏长期记忆和文件系统级别的内存管理；③资源隔离与安全性仍需更细粒度的策略；④实验规模有限，未在更复杂或多模态任务上验证；

---

## 339. AEGIS: A Mechanism-Guided Defense against Visual Synonym Jailbreaks in Text-to-Image Models

**arXiv ID:** 2607.06120 | [PDF](https://arxiv.org/pdf/2607.06120v1)

**作者:** Yuanmin Huang `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于机制分析的动态推理防御方法，专门针对文本到图像模型中的视觉同义词攻击（VSA）进行防御。

**💡 创新点**

创新点在于：①通过锚点相似度跟踪和稀疏回归定位VSA攻击时激活的少量深层注意力头；②在这些关键头上采用相似度感知的主动排斥机制，动态抑制危险语义；③实现了在保持模型原始权重不变的前提下，兼顾安全性和生成质量的平衡。

**🔧 技术方法**

技术主要包括：锚点相似度测量、稀疏逻辑回归（Lasso）定位关键头、跨层时间序列特征提取、基于余弦相似度的动态门控式主动排斥。

**📊 数据集**

使用的数据集包括：Stable Diffusion v1.4/2.1和FLUX.1的官方训练集；视觉同义词攻击数据集VSA、显式攻击I2P、对抗攻击RAB/MMA；以及MS‑COCO 1K用于评估生成质量。

**📈 对比分析**

与16种主流防御基线（输入空间消毒、触发器路径中断、结构特征修剪等）对比，在VSA、显式和对抗攻击上均取得最小化攻击成功率（如VSA下暴力/裸露 ASR ≤0.03），同时保持低 FID（≤70）和高 CLIP 分数，显示出优异的安全与实用性平衡。

**⚠️ 局限性**

局限性包括：需要针对每个模型架构重新定位关键头；对极端低阈值攻击的抵抗仍有限；以及在极端推理负载下的实时性略高。

---

## 340. Heckman-Corrected Epistemic Uncertainty: Selection on Unobservables Defeats Importance Weighting

**arXiv ID:** 2607.05806 | [PDF](https://arxiv.org/pdf/2607.05806v1)

**作者:** Gunner Levi Howe `[一作]` `[通讯]` (Independent Researcher), Gunner Levi Howe (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种将 Heckman 两方程模型与深度神经网络相结合的方法，用于在样本选择偏差（MNAR）场景下校正预测分布并量化不确定性。

**💡 创新点**

创新点在于把 Heckman 选择校正与深度网络和深度集成融合，提出联合 MLE 和两步法的深度预测分布，可在无观测选择偏差时恢复覆盖率并提供更可靠的置信区间。

**🔧 技术方法**

技术包括深度多层感知机作为结果网络、线性选择头、联合双正态似然、逆 Mills 比例校正、深度集成、MC dropout、高斯过程以及权重重采样对比。

**📊 数据集**

使用的数据集涵盖 RAND Health Insurance、California Housing、UCI Wine Quality、模拟 MNAR 生成器以及 Papers‑with‑Code 评测面板。

**📈 对比分析**

与基线方法（深度集成、MC dropout、GP、权重重采样、无选择校正）比较，深度 Heckman 在选取失偏区域的覆盖率和 region‑ECE 方面显著优于所有非 oracle 方法，且在有工具变量时恢复到接近 90% 的覆盖率。

**⚠️ 局限性**

局限包括假设误差为齐性高斯、选择头线性、集成规模有限、未处理异方差或非高斯误差，以及在无工具变量时识别不稳定。

---

## 341. LLM-Driven Neural Network Generation with Same-Family Architecture Guidance: Disentangling Transfer and Adaptation

**arXiv ID:** 2607.05704 | [PDF](https://arxiv.org/pdf/2607.05704v1)

**作者:** Kabir Dev Paul Baghel `[一作]` (University of Würzburg), Dmitry Ignatov `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于同族模型源指导的LLM候选生成协议，用于在单一评估预算下改进弱目标神经网络；

**💡 创新点**

创新点在于将同族强源模型的训练配置与架构信息作为检索与提示条件，分离非源基线与源导向两类候选，并通过无LLM复制对照拆解源指导优势，揭示了“recipe-transfer”和“recipe-adaptation”两种机制；

**🔧 技术方法**

技术包括检索-提示（retrieval‑augmented generation）、四臂候选生成（hp_default、baseline_edit、hp_transfer、analogical_edit）、最小确定性修复、单步训练评估、配对自举置信区间等；

**📊 数据集**

使用的公开数据集包括CIFAR‑10、CIFAR‑100、SVHN、Imagenette、CelebA‑Gender、MNIST、Places365，重点实验在CIFAR‑10与SVHN AlexNet；

**📈 对比分析**

比较方法是统一候选预算（N=32）下的有效生成率、最佳有效准确率与平均有效准确率，并与无源复制对照和不同LLM（DeepSeek‑6.7B、Qwen2.5‑7B、Olympic‑7B）进行配对统计；在CIFAR‑10上源指导可提升≈0.26准确率，SVHN AlexNet可提升≈0.56；平均提升约0.15，显示源指导在多数同族组合中占优势；

**⚠️ 局限性**

局限性包括：正向结果主要集中在CIFAR‑10与SVHN AlexNet，其他族与更难数据集表现不稳；评估仅为一轮训练，未验证长训练排序；未探究弱源或随机源的效果；仅使用最小修复，缺乏更深层语义修复；未验证LoRA微调对LLM的提升效果；

---

## 342. Safe Bayesian Optimization with Counterfactual Policies

**arXiv ID:** 2607.05620 | [PDF](https://arxiv.org/pdf/2607.05620v1)

**作者:** Katherine Avery `[一作]` (University of Massachusetts Amherst), David Jensen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SafeOpt-CPC方法，将分割式共形预测嵌入SafeBayesian优化中，用以在安全约束下估计未观测的反事实效果。

**💡 创新点**

创新点包括：① 为反事实约束提供置信区间，保证安全违规率控制；② 针对协变量偏移和非平稳性给出加权共形预测与逆倾向评分的自适应权重校正方案。

**🔧 技术方法**

使用的技术包括SafeBayesian优化、Gaussian Process 置信区间、分割式共形预测、加权共形预测、逆倾向评分、协变量偏移权重、在线 SafeOpt。

**📊 数据集**

实验数据集为化学反应模拟器、MovieLens 评分数据以及自构造的合成数据。

**📈 对比分析**

与无约束贝叶斯优化和完全知情的SafeOpt（oracle）对比；SafeOpt-CPC 在违规率上严格低于设定阈值 α，同时在满足安全约束的前提下获得较高的目标值；oracle 更具侵入性；无约束优化违规率明显过高。

**⚠️ 局限性**

局限性：置信区间过宽或容差 ω 过低会导致过度保守；多重估计项会显著降低可接受违规率 α'；仅控制违规率不足以满足某些高风险应用（如医疗）的全面安全需求。

---

## 343. Delay-Aware Active Triangulation with Uncertainty-Driven Multi-Agent Reinforcement Learning for Counter-UAS

**arXiv ID:** 2607.05957 | [PDF](https://arxiv.org/pdf/2607.05957v1)

**作者:** Seungwook Lee `[一作]` (Korea Advanced Institute of Science and Technology), David Hyunchul Shim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种延迟感知的多智能体强化学习框架，用于在 Counter-UAS 场景下通过主动视觉三角化实现无人机目标的三维定位。

**💡 创新点**

创新点在于引入 Age-of-Information (AoI) 标记的 Dec-POMDP、双路径奖励设计（感知一致 vs 特权奖励）以及多源解析协方差传播，显著提升了延迟补偿、定位精度和三角化有效性。

**🔧 技术方法**

采用 MAPPO 的中心化训练分散执行（CTDE）策略、GRU 递归网络、AoI 观测扩展、双路径延迟架构和多源协方差分析等技术。

**📊 数据集**

实验在 Isaac Sim 4.5 / Isaac Lab 2.1 生成的 4096 并行仿真环境中进行，未使用公开真实数据集。

**📈 对比分析**

通过感知一致奖励、特权奖励、无 AoI、仅角度协方差和 MLP 观测堆叠等设置进行对比，感知一致奖励在 0.547±0.217 m RMSE、78.1% 三角化有效率下取得最佳结果；AoI 提升有效率 10.6%，多源协方差将 RMSE 降至 2.8 倍以上。

**⚠️ 局限性**

仅验证两无人机，仿真环境；碰撞风险在扩展至多机时未得到充分处理；未完成真实飞行测试和动态网络延迟自适应。

---

## 344. Complementary Roles of Image Classification and Vessel Segmentation in AI-Based Screening for Retinopathy of Prematurity Plus Disease in a Kenyan Preterm Cohort

**arXiv ID:** 2607.05825 | [PDF](https://arxiv.org/pdf/2607.05825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 345. MemDefrag: Latent Memory Defragmentation for Large Language Models

**arXiv ID:** 2607.05969 | [PDF](https://arxiv.org/pdf/2607.05969v1)

**作者:** Ruiyi Yan `[一作]` (Kyoto University), Yiwen Guo `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关、模型无关的内存碎片整理框架 MemDefrag，通过中层注意力密度进行追踪并重排序。

**💡 创新点**

创新点在于发现中层注意力密度能作为追踪信号，结合无训练的内存碎片整理与信息性比例遗忘。

**🔧 技术方法**

采用中层注意力密度、分层重排序、Top‑K 筛选、信息性遗忘以及无训练的内存更新机制。

**📊 数据集**

在 NaturalQA、SQuAD 以及 LongBench 等长文本问答数据集上进行评估。

**📈 对比分析**

与 MemoryLLM、M+ 在知识保留和长上下文问答中比较，MemDefrag 在 50 步后知识保留上提升至约 43% 而基线为 17%，长上下文性能也普遍领先。

**⚠️ 局限性**

局限在于固定的 Top‑K 过滤参数，无法自适应；且对极长记忆或特殊模型仍需进一步验证。

---

## 346. A Guiding Framework for K-12 Teachers in Creating AI-powered Learning Technologies through Vibe Coding

**arXiv ID:** 2607.05406 | [PDF](https://arxiv.org/pdf/2607.05406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 347. Pluralis v0.1: Towards a Multicultural, Multimodal, Multilingual Benchmark for AI Risk and Reliability

**arXiv ID:** 2607.06196 | [PDF](https://arxiv.org/pdf/2607.06196v1)

**作者:** Alicia Parrish `[一作]` (Google DeepMind), Lora Aroyo `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了面向文化差异的多模态多语言安全评估数据集 v0.1，并开发了基于 LLM 的 Judge‑Pluralis 自动评估器，对视觉语言模型在不同地区的安全与文化适当性进行评估。

**💡 创新点**

首创文化先导的安全评估框架，既同时考察安全违规与文化不当，又通过多模态触发器和多地区语料捕捉地方性风险；并引入多模型投票、自动提示优化的 Judge‑Pluralis，提升评估的可靠性和可扩展性。

**🔧 技术方法**

多模态标注与翻译、基于 LLM 的 Judge‑Pluralis（多模型投票、自动提示优化），安全与文化两轴评估，基于人类标注的基础数据进行模型训练与验证；使用 t‑SNE、MPD 等指标进行语义与语言多样性分析。

**📊 数据集**

约6,448条来自6个亚太国家（孟加拉、印度、韩国、巴基斯坦、新加坡、台湾）的文化相关安全提示，涵盖8种语言（含英语变体），并与 MSTS、SEA‑SafeGuardBench 等公开基准进行对比。

**📈 对比分析**

通过 Judge‑Pluralis 对三款前沿 VLM（A、B、C）进行安全与文化适当性评分，采用 AILuminate 评分体系生成“Good/Fair/Poor”等级；结果显示不同模型在不同地区与语言的表现差异显著，整体仍存在较高安全与文化违规率。

**⚠️ 局限性**

仅覆盖亚太六个地区且语言多样性有限；文化规范随时间演变，需要定期重标；评估以单次生成样本为基础，存在采样噪声；自动评估仍漏判约 2/3 的违规；内部异质性导致标签一致性低；样本多样性受限，缺少自然对话与混合语种场景。

---

## 348. DDB: Source-Level Interactive Debugging for Distributed Applications

**arXiv ID:** 2607.06107 | [PDF](https://arxiv.org/pdf/2607.06107v1)

**作者:** Yibo Yan `[一作]` (University of Southern California), Seo Jin Park `[通讯]` (University of Southern California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个分布式交互式调试器DDB，能够在分布式系统中全局暂停、重建跨RPC调用栈并在源代码层面检查状态。

**💡 创新点**

创新点在于三大机制：分布式调用栈重建、意图保持统一控制平面以及暂停消除时间虚拟化，使得在大规模动态集群上实现无故障的交互式调试成为可能。

**🔧 技术方法**

采用了调用上下文元数据注入、集中式调试控制平面、POSIX时间API拦截与虚拟时钟、以及与现有RPC框架的20-60行轻量级集成。

**📊 数据集**

使用了微服务基准、gRPC Raft、一致性服务、ServiceWeaver、Nu、Quicksand等真实应用，以及合成时间敏感实验。

**📈 对比分析**

在用户研究中相较于GDB和OpenTelemetry，DDB实现了100%缺陷定位、20分钟内定位率提高到100%，后端吞吐量下降仅1-5%，并且分布式调用栈回溯延迟约30ms，整体性能优于传统工具。

**⚠️ 局限性**

局限性包括仅支持GDB兼容语言、仅覆盖POSIX时间API、无法虚拟外部真实时间、对TCP缓冲区大小有限制，且对高并发外部依赖仍需模拟。

---

## 349. MP-MPPI: A Motion Primitive Guided Sampling-Based Optimizer for Model Predictive Control

**arXiv ID:** 2607.06123 | [PDF](https://arxiv.org/pdf/2607.06123v1)

**作者:** Marlon Mathisen `[一作]` (Norwegian University of Science and Technology), Eleni Kelasidi `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将预计算运动原语与MPPI采样结合的MP-MPPI方法，用于四旋翼在复杂障碍场中的实时路径规划和控制。

**💡 创新点**

创新点在于把离散的、可行的运动原语作为额外采样加入MPPI权重更新，提升全局探索与收敛速度，且不依赖可逆动力学。

**🔧 技术方法**

使用Model Predictive Path Integral（MPPI）采样优化、GPU并行计算（JAX实现）、Runge‑Kutta 4积分、基于成本函数的障碍惩罚等技术。

**📊 数据集**

在自定义四旋翼仿真平台上，生成300根随机柱子组成的障碍场进行100次独立仿真测试（无公开公开数据集）。

**📈 对比分析**

与原MPPI在相同仿真条件下比较：MP-MPPI平均通行距离从48.0提升至66.6，标准差下降；MPPI一次碰撞，MP-MPPI无碰撞；在墙壁突遇场景中MP-MPPI成功避碰而MPPI碰撞；计算频率随预测步长线性增长，GPU上样本数对延迟影响微小。

**⚠️ 局限性**

局限性包括需要精细调节运动原语与噪声样本比例，过多原语可能降低最优性；缺乏针对特定任务自动生成最优原语库的方法；尚未评估在更复杂或多任务环境中的泛化性能。

---

## 350. RPAM: A Principled Metric for Evaluating Associations in Language Models with High Predictive Validity in Downstream Outputs

**arXiv ID:** 2607.05679 | [PDF](https://arxiv.org/pdf/2607.05679v1)

**作者:** Damian Hodel `[一作]` (University of Washington), Aylin Caliskan `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个基于相对生成概率的上游关联度量RPAM，并证明其能准确预测生成模型的隐式/显式偏差和下游文本输出中的关联。

**💡 创新点**

首次提出利用相对概率归一化的关联度量，消除了绝对概率带来的偏差，并通过模板化提示实现对不同模型的统一评估。

**🔧 技术方法**

使用生成模型的softmax归一化概率、模板化提示、Cohen’s d效应大小、Spearman相关系数和F1评估等技术，对比传统上游/下游指标。

**📊 数据集**

采用WEAT‑WS、WS‑353、Bellezza、SST2等人类隐式/显式关联数据集以及对应的生成文本样本。

**📈 对比分析**

与先前的上游/下游指标对照，RPAM在GPT‑2、Mistral‑7B‑Instruct和Mistral‑7B上与人类关联的Spearman ρ在0.57–0.79之间，F1≥0.71；对下游情感分析可达ρ=0.73、F1≥0.74，明显优于以往方法。

**⚠️ 局限性**

受限于需要访问模型的连续概率，无法直接应用于封闭式模型（如GPT‑4）；关联测量受刺激词选择影响；实验仅在英语环境下完成，跨语言推广待进一步研究。

---

## 351. Complets: Universal Compartmentalisation and Programming Model For Arm Permission Overlay Extension 2

**arXiv ID:** 2607.05569 | [PDF](https://arxiv.org/pdf/2607.05569v1)

**作者:** Vasily A. Sartakov `[一作]` `[通讯]` (Huawei R&D), Vasily A. Sartakov (Huawei R&D)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Arm POE2 的通用编程模型和 Complets 抽象，实现 intra-process 隔离并提供典型设计模式

**💡 创新点**

将 POE2 的多维索引抽象为单一接口，设计可信 trampoline，提供完整的 API 支持对称、非对称和 enclave 模式

**🔧 技术方法**

Arm POE2 指令、表（IRT、DPOT、TTT）、ARM 机器码、Linux kernel syscall 扩展、BTI、trampoline 代码

**📊 数据集**

使用 Arm FVP 功能模型（ARM 的模拟器）进行评估，未使用公开数据集

**📈 对比分析**

在 Arm FVP 上测得单向迁移约 75 条指令，内核改动约 300 行，性能开销极小；相较于传统跨进程通信减少 kernel 跳转

**⚠️ 局限性**

依赖于具体硬件实现，未在真实硬件上验证；模型复杂度仍高；仅覆盖 EL0，扩展到其他异常级别尚未实现

---

## 352. Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure

**arXiv ID:** 2607.05966 | [PDF](https://arxiv.org/pdf/2607.05966v1)

**作者:** Finn Rasmus Schäfer `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出将长周期世界模型失效归因为运动学误差而非动力学误差，并开发了 iKCE 诊断工具来衡量想象 rollouts 的运动学一致性。

**💡 创新点**

创新点在于引入运动学一致性误差指标及条件扰动协议，用以区分运动学与动力学想象，并在摩擦系数跨界时证明模型的 iKCE 保持平稳，从而确认运动学 fallback。

**🔧 技术方法**

使用基于闭式运动学预测器的误差计算、参数扰动实验、以及对 DreamerV3 进行的多步 rollouts 与真实物理 rollouts 的对比分析等技术。

**📊 数据集**

使用 DeepMind Control Suite 中的 walker-walk 环境（2D 行走机器人）以及公开的 DreamerV3 checkpoint 进行评估。

**📈 对比分析**

将 iKCE 与真实物理 rollouts 的 iKCE 及奖励表现对比；实验显示 DreamerV3 的 iKCE 在 T=16 时约 180 倍、T=64 时约 30 倍，同时在摩擦系数跨界时保持平稳，而真实物理 rollouts 随摩擦下降。

**⚠️ 局限性**

局限性包括仅在单一 2D 行走机器人上验证、仅使用一个 WM 家族、对长期 horizon 的 iKCE 受幅度衰减影响、以及未完全排除策略 OOD 或观察前缀信息不足的影响。

---

## 353. Context-to-Execution Integrity for LLM Agents

**arXiv ID:** 2607.06000 | [PDF](https://arxiv.org/pdf/2607.06000v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Context-to-Execution Integrity (CXI) 机制，专门针对 LLM 代理在执行工具调用时的权限边界进行验证，确保所有受保护字段、解释后的效果以及调用事件均通过同一规范化操作清单（manifest）进行授权后才允许执行。

**💡 创新点**

创新点在于将传统的基于字段的完整中介、Typed Release、Exact‑Effect Authorization 与 Manifest‑Bound Gate 三个安全概念统一到一个可验证的执行边界中；并首次引入“authority laundering”概念，强调攻击者可以在可写上下文中植入伪造权限，CXI通过严格的证据链闭合来阻止此类攻击。

**🔧 技术方法**

主要技术包括：
- Typed Release（受限发布）用于把可写上下文转化为针对特定字段的可信值；
- Exact‑Effect Authorization 用来将解释后的工具效果与验证器绑定；
- Manifest‑Bound Gate（门控）在执行前把字段、效果、调用权限组合成单一动作清单，并消耗对应的调用能力；
- Open‑Weight 与 Hosted/API 两种证据路径来满足不同部署场景；
- 结合现有工具如 vLLM、PyTorch、OpenAI GPT、Gemini 等进行部署。

**📊 数据集**

使用了两类数据集：
- AgentDojo 的 720 条直播任务（1,739 次 LLM 调用）和 400 条代码仓库任务；
- 公开的 800 条 Hosted/API 兼容性日志；
- 另外还有针对 ledger 错误、proposal‑pressure 等实验数据。

**📈 对比分析**

对比方法主要是：
- 基准：仅做 JSON schema 验证或仅做字段级别授权；
- CXI 通过在门控前加入完整证据链，实测无字段/效果/调用泄漏；
- 性能方面，Open‑Weight 路径每条案例平均 0.16 ms（字段层），Hosted/API 由于网络往返导致 9‑15 s；整体上 CXI 对吞吐量影响极小；
- 结果显示在所有评测表面均未观察到未授权执行，且在安全指标上比基线提升显著。

**⚠️ 局限性**

局限性：
- 需要手工编写与维护字段分类、Typed Release 规范、Exact‑Effect 验证器等政策，政策错误或缺失仍会破坏安全；
- 仅验证权限闭合，未涵盖任务正确性、性能调优、异常恢复或 exactly‑once 外部效果保证；
- Open‑Weight 需要访问模型内部的 attention/kv 记录，Hosted/API 只能使用 host‑observed 兼容性证据，导致对第三方服务的完整性证明受限；
- 对于未被列入受保护字段或未知的工具接口，仍需额外的 bypass 检查。

---

## 354. Beyond the Leaderboard: A Synthesis of Tool-Use, Planning, and Reasoning Failures in Large Language Model Agents

**arXiv ID:** 2607.05775 | [PDF](https://arxiv.org/pdf/2607.05775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 355. Prompt Coach: An Empirical Evaluation of an Agentic Tutor for Learning Prompt Engineering in Software Development

**arXiv ID:** 2607.06074 | [PDF](https://arxiv.org/pdf/2607.06074v1)

**作者:** Rohit Mehra `[一作]` (Accenture), Majd Sakr `[通讯]` (Accenture)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一款名为 Prompt Coach (PC) 的代理式导师，嵌入 IDE 中为软件开发者提供 Socratic 引导，帮助其学习如何编写高质量的代码生成提示词，并通过多维度评估和自我纠错问题提升提示词质量。

**💡 创新点**

创新点在于：① 将代理式教学与 IDE 无缝融合，实现即时、情境化的 Socratic 引导；② 采用多维度评估体系（如准确性、可读性、可扩展性等）针对提示词质量进行量化；③ 通过针对性问题引导开发者进行自我修正，提升学习效果。

**🔧 技术方法**

技术实现主要包括：① 语言模型 (LLM) 用于生成代码并评估提示词效果；② 规则与指标库用于计算多维度质量得分；③ 交互式问答模块嵌入 IDE，实时展示评估结果与改进建议；④ 统计分析工具用于评估实验数据。

**📊 数据集**

实验数据集主要来源于 15 名专业软件开发者的真实代码库和他们使用的目标 LLM 生成的代码结果；提示词和对应代码的质量评分由 PC 进行自动化评估，亦辅以人工标注。

**📈 对比分析**

评估方法：在使用 PC 前后对同一批提示词进行质量评分，并结合用户问卷收集主观感受；实验结果显示，单次 60 分钟学习后，提示词质量显著提升（p < 0.05），尤其在常被忽视的维度（如提示词的可解释性与鲁棒性）上提升最大；参与者对 PC 的信任度、接受度与效果认知均呈现高度正向评价。

**⚠️ 局限性**

局限性：① 样本量仅为 15 名开发者，缺乏更大规模的验证；② 仅评估单次 60 分钟学习效果，未考察长期使用对技能持续提升的影响；③ 研究聚焦于特定 LLM 与 IDE，结果在其他环境下的可迁移性未知；④ 质量评估仍依赖部分人工标注，自动化度有待提升。

---

## 356. MoWorld: A Flash World Model

**arXiv ID:** 2607.06216 | [PDF](https://arxiv.org/pdf/2607.06216v1)

**作者:** Team Moxin `[一作]` (Moxin Technology), Lingyun Sun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MoWorld，一种可在实时（30–50帧/秒）下运行的“闪电式”世界模型，能够以初始图像、文本与用户控制的相机轨迹为输入，生成连续、可交互的动态视频。

**💡 创新点**

创新点包括：① 基于几何感知的数据引擎实现高质量、带相机姿态的训练集；② 进阶的时空能力学习范式与多阶段自适应预训练；③ 将多步扩散采样压缩为仅4步的自回归蒸馏方法；④ 在算子、并行、管线层面与NPU硬件共同优化，显著降低内存与推理成本；⑤ 通过从模型到系统的端到端协同设计，实现了首次在NPU上完成大规模14B MoE世界模型的实时推理。

**🔧 技术方法**

技术手段包括：几何完成与质量评估、Vision‑Language 注解、预缓存；Wan2.2 MoE 视频基础模型与相机适配器；进阶时空训练、长视频序列并行与统一序列并行（USP）；去噪步蒸馏、AR Flow Matching、Self‑Forcing 蒸馏；动态混合精度量化、HBM‑高效注意力核、RoPE 位置编码、On‑Demand 模块加载；多级分布式训练与推理。

**📊 数据集**

使用了多源数据集：开放域视频、游戏场景数据、合成数据生成的几何强化数据集；预训练集覆盖室内外多样场景；评估采用 VBench‑I2V 及从互联网收集的百余个真实与人工合成视频。

**📈 对比分析**

与现有方法对比（CameraCtrl、SEVA、Lingbot 等），MoWorld 在 VBench‑I2V 的八维质量指标中取得最佳平均分（85.22）和最高质量分（82.73），并在子指标（SC、BC、I2V‑S、I2V‑B 等）上保持领先；实时推理可达 50 FPS，推理成本比同类模型低 30–50%，训练成本亦显著下降。

**⚠️ 局限性**

局限性包括：模型规模仍为 14B，训练与推理仍需要高性能 NPU；在极端动态或稀疏几何场景下的稳定性未完全验证；对非结构化环境的迁移能力和对不同硬件平台的可移植性需要进一步研究。

---

## 357. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training

**arXiv ID:** 2607.05804 | [PDF](https://arxiv.org/pdf/2607.05804v1)

**作者:** Yuhang Zhou `[一作]` (Fudan University), Jingjing Chen `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TurnOPD，一种针对长时序语言代理的基于回合的预算控制训练框架，改进了传统的 On‑Policy Distillation（OPD）；

**💡 创新点**

创新点在于通过自适应回合深度控制和逐步平衡的 KL 损失归一化，解决了传统 OPD 在长时序任务中资源浪费与损失分配失衡的问题；

**🔧 技术方法**

采用了基于逆 KL 的 OPD 损失、周期性探测（probe）获取回合级统计、指数移动平均（EMA）平滑以及线性混合权重等技术；

**📊 数据集**

在 ALFWorld、WebShop 与 Multi‑Hop Search 三大多回合代理基准上进行评估；

**📈 对比分析**

与 vanilla OPD、TCOD‑F2B 等基线相比，TurnOPD 在相同时间预算下平均提升 15–30% 的 Avg@4 准确率，同时训练时间缩短 1.5–2.3 倍；

**⚠️ 局限性**

局限性包括对教师模型的强依赖、需要手动调节覆盖阈值、探测步骤对计算开销有一定影响，以及在不同任务类型和教师水平下的泛化能力仍待进一步验证。

---

## 358. Breaking Structural Isolation: Scalable Graph Clustering via Community-Aware Sampling and Structural Entropy

**arXiv ID:** 2607.05469 | [PDF](https://arxiv.org/pdf/2607.05469v1)

**作者:** Jingyun Zhang `[一作]` (Beihang University), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种可扩展的无监督图聚类框架——SCISE，通过结构熵约束、社区感知采样和结构对比学习三种模块来缓解批量训练中的结构孤立问题。

**💡 创新点**

创新点在于(1)引入结构熵社区约束（SECC）在限定社区数下最小化结构熵，避免碎片化；(2)设计社区感知采样扩展（CSampE）将同一社区节点聚合进同一mini‑batch，恢复全局拓扑；(3)构造结构对比学习（StructCL）利用随机游走产生的结构亲和矩阵作为正样本，提升嵌入的社区区分度。

**🔧 技术方法**

使用的技术包括图神经网络（GCN）编码器、结构熵理论、随机游走、InfoNCE 对比损失以及K‑means后处理。

**📊 数据集**

在六个主流基准数据集上评估：Photo、Computers、Pubmed、Ogbn‑arxiv、Reddit、Ogbn‑products。

**📈 对比分析**

与10个先进基线（如DGI、DMON、MAGI、LSEnet等）比较，SCISE在21/24项指标（NMI、ARI、ACC、F1）上取得SOTA；在大规模图上跑时延接近线性，显著低于MAGI；在噪声、稀疏和超参数变化下仍保持稳健。

**⚠️ 局限性**

局限性包括：对社区划分的依赖性（SECC质量影响），对超参数（如p、θ、随机游走步长）的适度敏感，未针对有向或异构图进行验证，且预处理步骤虽低开销但在极大图中仍需额外资源。

---

## 359. Clustered Codebook Quantization for 2D Gaussian-based Image Compression

**arXiv ID:** 2607.05667 | [PDF](https://arxiv.org/pdf/2607.05667v1)

**作者:** Runze Cheng `[一作]` (University College London), Kaan Akşit `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CGVQ方法，通过先用K‑means聚类把二维高斯原语分成若干同质组，再为每组分别训练位置、旋转尺度和颜色的局部码本进行量化压缩；

**💡 创新点**

创新点在于利用聚类先降低各组参数分布的方差，从而让局部码本能够更精确地编码高斯原语，显著提升量化精度和压缩质量；

**🔧 技术方法**

采用K‑means聚类、局部码本训练（FP16、UQ、RQ）、向量量化、FakeVQ估计以及部分位回编码等技术；

**📊 数据集**

主要在Kodak图像数据集上进行实验，并以Animal Faces等示例展示效果；

**📈 对比分析**

与GI基线进行bpp、PSNR、SSIM等指标对比，CGVQ在相同重建质量下bpp下降约20%，PSNR提升约1.7dB，整体rate‑distortion性能优于基线；

**⚠️ 局限性**

局部聚类会显著降低编码/解码速度，细粒度聚类导致编码FPS从2.91×10⁻²降至1.9×10⁻³，解码FPS从133.3降至33.3，需在压缩质量与计算效率之间权衡。

---

## 360. CurateEvo: Data-Curation Evolving for Agentic Post-Training

**arXiv ID:** 2607.06140 | [PDF](https://arxiv.org/pdf/2607.06140v1)

**作者:** Dingzirui Wang `[一作]` (Harbin Institue of Technology), Wanxiang Che `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于失败驱动的动态演化框架，用可执行代码表示数据处理策略并在每轮迭代中根据 held‑out 开发集的失败轨迹进行改进；

**💡 创新点**

首次将数据增删改的所有操作统一成可执行代码，并通过成本感知的目标同时优化代理性能与训练成本，兼顾效果与效率；

**🔧 技术方法**

使用 LLM（GPT‑5.4）进行代码演化；在固定的 SFT+GRPO 训练 recipe 上进行代理微调；采用 LoRA 等参数高效微调；

**📊 数据集**

实验用 ACEBench‑Agent、BFCL‑V4、τ²‑Bench 作为评测基准；原始语料分别为 labeled（SWE‑chat、AgentRewardBench、OpenHands‑Feedback）和 wild（ASSERT‑KTH、lelouch0110/claudeset-community、nlile/misc‑merged‑claude‑code‑traces‑v1）;

**📈 对比分析**

与 GRPO、MUA‑RL、EnvScaler、AWM、RODS、FunReason‑MT 等现有数据 curation 基线对比，平均提升 labeled 场景 3.2 分、wild 场景 2.7 分；并能与其他后训练 recipe（如 ProRL‑Agent）配合提升 21.3 分；

**⚠️ 局限性**

局限性包括：演化过程仍需多轮 LLM 调用，计算成本较高；在不同模型/环境下的泛化性未充分验证；对极端噪声或不规则数据的鲁棒性仍待进一步探索。

---

## 361. Can Large Language Models Generate Observability-Aware Code?

**arXiv ID:** 2607.05785 | [PDF](https://arxiv.org/pdf/2607.05785v1)

**作者:** Yongliang Tao `[一作]` (Chongqing University), Saravan Rajmohan `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了编码代理生成的软件系统在故障时的可观测性，并构建了从源代码层面到运行时层面的评估框架

**💡 创新点**

提出了多层评估方法、发现了“诊断语义缺口”，并验证了轻量化可观测性指导（Skill）对诊断语义和故障信号暴露的有限提升

**🔧 技术方法**

利用大型语言模型（GPT‑5.5、Claude Opus 4.8、Gemini 3.5 Flash）进行代码生成与可观测性恢复；使用日志分析、故障注入（Chaos Mesh）和自定义指纹的 Fault Signal Rate（FSR）评估运行时可观测性

**📊 数据集**

共1,223个源代码实例（10个开源+8个工业项目）用于可观测性恢复；200个微服务系统+13类故障（1,615次故障注入）用于运行时评估；约200条真实世界的可观测性相关提交用于构建 Skill

**📈 对比分析**

在静态层面，Position F1平均0.55，KeyBag F1平均0.36；在运行时层面，FSR最高仅达13.99%（GPT‑5.5），通过 Skill 提升至最大16.53%；相比传统评测方法，显著揭示了当前代理在故障信号暴露方面的不足

**⚠️ 局限性**

主要限制是：可观测性语义缺失导致日志缺乏故障特定信息；即使有日志也往往缺乏诊断上下文；轻量化指导提升有限，难以完全弥补语义缺口；模型生成的随机性和训练数据对可观测性认知不足也可能是原因

---

## 362. MobileWan: Closing the Quality Gap for Mobile Video Diffusion

**arXiv ID:** 2607.06173 | [PDF](https://arxiv.org/pdf/2607.06173v1)

**作者:** Mohsen Ghafoorian `[一作]`, Amirhossein Habibian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 363. The Documentation and Traceability Burden of the Indian EV Transition

**arXiv ID:** 2607.06170 | [PDF](https://arxiv.org/pdf/2607.06170v1)

**作者:** Dawar Jyoti Deka `[一作]` (Erdős Systems), Nilesh Sarkar `[通讯]` (Erdős Systems)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统化了印度电动车制造商的合规文档生命周期，提出了双层证据义务分层、生命周期模型、文档功耗视角和研究议程。

**💡 创新点**

创新点在于首次将合规证据视为信息系统对象，提出文档功耗破坏（exergy‑destruction lens）以及针对四类失败案例的七项架构需求。

**🔧 技术方法**

技术方法包括信息系统视角的系统化、生命周期模型构建、功耗破坏分类、案例归纳与需求推导。

**📊 数据集**

使用的数据主要为四个公开的失败案例、印度监管文件和日本案例，未采用机器学习或大规模数据集。

**📈 对比分析**

论文未进行实验比较，仅通过案例对照与理论推演说明其方法在解释失败和揭示需求方面的有效性。

**⚠️ 局限性**

局限在于仅针对印度情境，缺乏经验测量、自动化工具实现及跨国验证，且未给出可量化指标。

---

## 364. Leveraging Extragradient for Effective Sharpness-Aware Minimization in Deep Learning

**arXiv ID:** 2607.06151 | [PDF](https://arxiv.org/pdf/2607.06151v1)

**作者:** Yao Fu `[一作]` (Xi'an Jiaotong University), Yuanao Yang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的优化器 EISAM，旨在通过改进 Sharpness-Aware Minimization（SAM）来提升深度学习模型的泛化性能。

**💡 创新点**

创新点在于引入基于 extragradient 的两步更新策略——先预测一步探测损失景观，再进行 sharpness-aware 纠正，显著降低了对扰动半径 ρ 的敏感度并进一步引导参数收敛到更平坦的极小点。

**🔧 技术方法**

核心技术包括 extragradient 方法、SAM 的对抗扰动框架、基于 SGD/Adam 的基础优化器、以及通过 Hessian 谱和 PAC‑Bayes 视角的泛化误差理论分析。

**📊 数据集**

实验使用了多种视觉与语言基准：CIFAR‑10、CIFAR‑100、ImageNet‑1K、COCO、LVIS、ISIC2018 以及自然语言任务 BOOLQ，覆盖了分类、检测、分割和问答等任务。

**📈 对比分析**

通过与 SGD、Adam、SAM、ASAM、GSAM、FSAM 等主流优化器在同一数据集和模型架构下进行对比，EISAM 在测试准确率、AP 指标以及训练损失下降速度上均优于基线，表现出更高的泛化能力和更快的收敛。

**⚠️ 局限性**

限制包括仍需对扰动半径 ρ 与预测步长 s 进行经验调参；理论分析主要基于凸/光滑假设，实际应用中对非凸大规模模型的理论保证有限；相较于 SAM，EISAM 仍需两次梯度计算，计算成本略高。

---

## 365. BlossomPsy: A User-Centric AI System for Adaptive and Engaging MBTI Personality Assessments

**arXiv ID:** 2607.06149 | [PDF](https://arxiv.org/pdf/2607.06149v1)

**作者:** Bingjia Huang `[一作]` `[通讯]` (University of Electronic Science and Technology of China), Bingjia Huang (University of Electronic Science and Technology of China)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计了BlossomPsy，一种融合多轮对话、图片式问题与自适应评估的MBTI人格测试系统；

**💡 创新点**

创新点在于将多头分类器与改进UCB算法结合，利用PID调节置信息，并通过照片式问题提升交互与预测一致性；

**🔧 技术方法**

技术包括多头RoBERTa分类器、改进的Upper Confidence Bound（mUCB）、PID控制器、LLM驱动的对话与图片生成；

**📊 数据集**

使用的数据集包括Personality Cafe、CPME（中文大学生帖子）、MBTI-M中文版本以及LLM仿真测试；

**📈 对比分析**

通过与MBTI-M的准确率、F1、Cohen's κ及用户满意度对比，BlossomPsy在大多数维度上与标准测评保持中等至高度一致，并在交互体验上显著优于传统量表；

**⚠️ 局限性**

局限性包括样本量偏小、图片式问题缺乏心理学验证、仅针对中文大学生、PID调参仅覆盖两参数，需进一步跨文化与更大规模验证。

---

## 366. RFHNet: Relational and Frequency-Aware Hashing Network for Large-Scale Fine-Grained Food Image Retrieval

**arXiv ID:** 2607.06148 | [PDF](https://arxiv.org/pdf/2607.06148v1)

**作者:** Junsong Wang `[一作]` (Ludong University), Shuqiang Jiang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种层次级联哈希网络 RFHNet，用于大规模细粒度食品图像检索。

**💡 创新点**

创新点在于三大模块：细粒度关系建模（FRM）捕捉局部空间关系；多频率调制融合（MFMF）利用FFT提取低/中/高频特征并自适应融合；层次语义协同（HSS）通过自注意力实现跨尺度语义融合。

**🔧 技术方法**

使用 ResNet‑50 级联骨干、空间关系分支、FFT频域分解、注意力加权融合、学习式多任务损失和二值化哈希层，形成端到端训练框架。

**📊 数据集**

在六个公开食品数据集上进行评估：Food‑101、Vireo Food‑172、UEC Food‑256、VegFru、ISIA Food‑500 与 Food2K。

**📈 对比分析**

与 HashNet、ADSH、A²‑Net、SEMICON、A²‑Net++、AMGH、DAHNet、SPBH、FoodHash 等九种基线对比，RFHNet 在 12‑bit 甚至 48‑bit 场景均实现最高 mAP（如 12‑bit Food‑101 +4.44%，Food2K +16.93%，VegFru +17.20% 等）。

**⚠️ 局限性**

局限性包括：训练与推理仍需较高显存与算力（尤其 48‑bit 版），仅针对食品图像验证，跨领域泛化能力尚待进一步探索。

---

## 367. Poster: Mind the Gap -- Characterizing the Temporal Blind Spot Between GSB and DNS Resolution

**arXiv ID:** 2607.06134 | [PDF](https://arxiv.org/pdf/2607.06134v1)

**作者:** Tomer Gal `[一作]` (Ariel University), Harel Israel Berger `[通讯]` (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了Google Safe Browsing与DNS解析在浏览器导航过程中的时间同步差异，量化了GSB查询关闭事件与最终DNS响应之间的时差。

**💡 创新点**

提出基于数据包级的时差度量，并揭示约80%场景存在可达数秒的正时差，提示潜在的DNS操纵攻击窗口。

**🔧 技术方法**

利用Chrome浏览器自动化导航、tcpdump/wireshark抓包，分析TLS SNI、TCP FIN/RST和DNS A/AAAA响应时间，计算Δ_time。

**📊 数据集**

基于Tranco排行榜选取的100个常规域和103个CNAME链域作为实验样本。

**📈 对比分析**

统计正时差出现比例、均值、中位数和最大值，发现两组数据正时差比例约为78%–80%，平均时差分别为248 ms和179 ms，最大值超过2.4 s。

**⚠️ 局限性**

研究仅在macOS Chrome环境下进行，未涵盖其他浏览器/系统；未验证攻击实际可行性，仅揭示时间窗口而非完整攻击链；未考虑网络环境变化对时差的影响。

---

## 368. Property-Driven Synthetic Data Engineering for Data-Scarce Software Systems: Reflections from the Breast Cancer Domain

**arXiv ID:** 2607.06133 | [PDF](https://arxiv.org/pdf/2607.06133v1)

**作者:** Aurora Francesca Zanenga `[一作]` (University of Bergamo), Claudio Menghi `[通讯]` (University of Bergamo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过与肿瘤学专家合作，对一份包含1000名乳腺癌IORT病例的敏感临床数据集进行清洗、合成，并基于合成数据评估其在临床、统计和隐私三维属性下的可用性，提出“属性驱动合成数据工程”框架。

**💡 创新点**

创新点在于：①把合成数据视为满足利益相关者特定属性的工程产物，而非单纯的技术预处理；②系统化地定义并检验临床可行性、统计保真度、隐私保护等多属性；③强调属性定义、验证与管线演化的自动化支持与协同。

**🔧 技术方法**

主要技术包括：数据清洗与标准化；多种表格合成模型（TabDDPM、CTGAN、Gaussian Copula、CopulaGAN、Tabular VAE），对比其统计相似度；临床验证方法如Kaplan–Meier、Cox回归等；属性检验工具与框架（规则挖掘、隐私风险评估、属性冲突检测）。

**📊 数据集**

使用的数据集为一家大型医院的IORT临床记录，初始为1000例、64变量，清洗后709例、58变量，数据高度敏感且不公开共享。

**📈 对比分析**

比较方法：对合成数据与原始数据的相关矩阵、分布形态、以及临床结果（如局部复发生存曲线）进行对比。结果表明：Tabular VAE 在保留统计相关性方面优于其他模型，但在临床可行性与隐私风险方面仍需进一步验证，整体表现尚未达到临床可接受水平。

**⚠️ 局限性**

局限性：①合成数据缺乏真实标签，难以进行完整的模型性能验证；②属性定义与检验高度依赖专家知识，缺乏统一标准；③数据清洗与属性演化过程需要人工干预，自动化程度不足；④在隐私保护方面，仅采用了简单指标，未覆盖更复杂的泄露风险。

---

## 369. Measuring the practice of shared-decision making (OPTION12): An Investigation into Open-sourced Smaller LLMs (OS-sLLMs) for Better Privacy and Sustainability

**arXiv ID:** 2607.06127 | [PDF](https://arxiv.org/pdf/2607.06127v1)

**作者:** Tamara Wit `[一作]` (Leiden University Medical Centre), Suzan Verberne `[通讯]` (Leiden Institute of Advanced Computer Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用开源小型语言模型对荷兰黑色素瘤患者的共享决策制定（SDM）会谈进行OPTION12评分，并构建Judge‑LLM框架以解决多模型间的分歧；

**💡 创新点**

首次评估开源小型LLM在OPTION12编码任务中的性能，比较通用与医学领域模型，提出多模型投票与Judge‑LLM共识框架，并归纳系统错误类别；

**🔧 技术方法**

采用提示工程（链式思维、少量示例）、多模型微调与少量参数调优，使用Gemma3:12b、Llama3.1:8b、Mistral7b等通用模型以及MedLlama2:7b、Meditron7b等医学模型；

**📊 数据集**

26份荷兰黑色素瘤患者与医生的会谈转录，双人编码后得到人类参考标签，划分为7份开发集与15份测试集；

**📈 对比分析**

对开发集进行准确度与相关性评估：Gemma3:12b最高，Pearson≈0.51、Spearman≈0.59，平均文件一致性仅约23%；在测试集使用Judge‑LLM后仍未达到人类标注水平，表明LLM需进一步优化；

**⚠️ 局限性**

主要局限包括：医学专用模型易出现幻觉且跟随提示能力差，所有模型在大多数OPTION12项目上的精确一致率低，且整体仍无法替代人类编码，需要更多训练数据、精细化提示与模型融合策略。

---

## 370. Static Metrics Are Insufficient: Predicting Java Method Energy Usage with Execution Time

**arXiv ID:** 2607.06124 | [PDF](https://arxiv.org/pdf/2607.06124v1)

**作者:** Muhammad Imran `[一作]` (University of L'Aquila), Ivano Malavolta `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Java 方法级能耗进行预测，探讨仅用静态代码度量是否足够，并评估加入执行时间后对预测的提升。

**💡 创新点**

证明单纯静态度量无法有效预测能耗，但轻量级动态特征（执行时间）能显著提升模型性能；同时系统化比较了 11 种回归模型与多种特征选择与调参方案。

**🔧 技术方法**

使用静态代码分析工具 srcML 提取代码特征，利用 async‑profiler 与 JoularJX 进行方法级执行时间与能耗采样；构建并训练 11 种回归模型（如随机森林、AdaBoost、线性回归等），并应用 RFECV、AutoSpearman、SelectKBest 等特征选择与 RandomizedSearchCV 调参。

**📊 数据集**

从 Rosetta Code 和 Computer Language Benchmarks Game（CLBG）收集约 1,100 个 Java 文件，最终得到 265 条包含完整静态、执行时间与能耗记录的数据点。

**📈 对比分析**

通过 5‑折交叉验证比较模型性能，发现最佳模型为随机森林（R²≈0.46），加入执行时间后 R² 从近 0 提升至 0.45；特征选择仅略有提升，调参对性能提升有限。

**⚠️ 局限性**

受限于小样本、单线程基准、仅 CPU/内存能耗采样、以及对执行环境（JVM、硬件）的依赖，模型泛化能力与对大规模、异构或多线程系统的适用性尚未验证。

---

## 371. Compiling Bioinformatics Recurrences

**arXiv ID:** 2607.06225 | [PDF](https://arxiv.org/pdf/2607.06225v1)

**作者:** Bala Vinaithirthan `[一作]` (Stanford University), Fredrik Kjolstad `[通讯]` (Stanford University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 FILTR，一套专为生物信息学递归式算法设计的领域特定语言与编译框架；

**💡 创新点**

创新点在于将递归核心与剪枝/调度策略分离，提供可组合的迭代顺序语言与剪枝语言，并生成可与手写优化代码相媲美的 C++；

**🔧 技术方法**

采用递归中间表示、线性变换（shear）、搜索（search）与动态剪枝等技术，结合依赖图分析和代码生成；

**📊 数据集**

使用合成序列（控制相似度、长度、突变率）和真实 DNA / RNA 数据集进行基准测试；

**📈 对比分析**

与 Recuma、SeqAn、Parasail、Ksw2、WFA2 等库和编译器对比，FILTR 在多种算法（编辑距离、Affine Gap、RNA 预测、序列链）上实现 0.95×–30× 的速度提升；

**⚠️ 局限性**

局限性包括：需手工探索调度/剪枝组合，主要针对 2D 动态规划，动态剪枝仍需手工编写规则，GPU/分布式实现尚未完成。

---

## 372. A toy framework for single and multi-agent human-AI curiosity ecosystems

**arXiv ID:** 2607.06214 | [PDF](https://arxiv.org/pdf/2607.06214v1)

**作者:** Ilya E. Monosov `[一作]` `[通讯]` (Johns Hopkins University), Ilya E. Monosov (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个基于权重漂移的好奇心生态学理论框架，并将其从单体代理扩展到多代理共享知识系统

**💡 创新点**

创新点在于将即时不确定性降低、成本、长期回报与保持问题开放的价值统一为可随经验与环境漂移的可调权重，形成可持续的好奇心决策与群体知识生成模型

**🔧 技术方法**

采用了数学建模、决策理论与强化学习等技术手段来描述权重漂移、查询选择和共享知识更新过程

**📊 数据集**

本研究为理论性框架，未使用具体数据集，仅以概念式示例与公式进行说明

**📈 对比分析**

论文未进行实验比较，提出了未来通过仿真和实证实验验证框架有效性的方向，未给出性能指标

**⚠️ 局限性**

局限性包括缺乏经验验证、知识表示过于简化（仅用标量库存与冗余度），以及多代理交互机制尚未在实际系统中实现

---

## 373. Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition

**arXiv ID:** 2607.06185 | [PDF](https://arxiv.org/pdf/2607.06185v1)

**作者:** Xinda Liu `[一作]` (Northwest University), Shuqiang Jiang `[通讯]` (Institute of Computing Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结构化且压缩化的文本提示调优方法 SCPT，用于提升细粒度图像识别的零样本与少样本性能。

**💡 创新点**

创新点在于：①通过 Semantic Relation Encoding（SRE）将类别间语义拓扑显式嵌入提示空间；②使用 Semantic Condensation Loss（ScLoss）对语义监督进行低秩压缩，抑制冗余噪声，二者协同提升细粒度区分度。

**🔧 技术方法**

核心技术包括 CLIP 预训练模型、CoOp 风格可学习文本提示、随机投影（Johnson–Lindenstrauss）实现 SRE、奇异值分解（SVD）实现 ScLoss，配合对比损失进行微调。

**📊 数据集**

在 14 个细粒度基准上验证：Dog Breed、Oxford Pets、Oxford Flowers、Stanford Cars、Web Cars、Stanford Dogs、Fruit92、CUB200、Food101、Food172、Food200、Food500、FGVC Aircraft、Veg200。

**📈 对比分析**

与 CLIP、CoOp、CoCoOp、KgCoOp、TCP、ProText、ATPrompt 等最新提示调优方法对比，SCPT 在 16-shot 细粒度分类平均准确率达到 76.70%/78.72%（基线 74.56%/74.53%），在 base‑to‑novel 泛化中的调和平均也显著提升，整体位居 state‑of‑the‑art。

**⚠️ 局限性**

局限性：仅针对文本提示，未融合视觉提示；浅层提示注入仅在少数层，未探索更深层或多模态提示；对类别数极少的任务效果相对有限，且对语义结构依赖较大。

---

## 374. A Global Author-Identity Map for the World of Code:62.7M Developer Identities from 106.8M Author Strings over 5.87B Commits

**arXiv ID:** 2607.06183 | [PDF](https://arxiv.org/pdf/2607.06183v1)

**作者:** Audris Mockus `[一作]` `[通讯]` (University of Tennessee), Audris Mockus (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为World of Code构建了一个全局作者身份映射，解决了同一人使用多重邮箱/名字以及不同人使用相同字符串的问题，并发布了多种关联表和验证数据。

**💡 创新点**

创新点在于提出六阶段门控-切割-分类-回忆-局部匹配管道，首次在十亿级别上实现高召回且无mega‑cluster的作者去重，并提供splitting和clumping双向评估。

**🔧 技术方法**

使用值门控、结构性桥接（采样betweenness）、逻辑回归边分类器、shingle扩展、GitHub同账户链接及局部项目级匹配等技术。

**📊 数据集**

基于World of Code 6.87B提交、106.8M原始作者字符串，结合GitHub handle标签、ALFAA人工金标注、OpenStack‑centric gold、Semantic Scholar和OpenAlex学术作者图。

**📈 对比分析**

与先前WoC V3对比，recall 0.70、precision 0.88、最大cluster 6,910；对ALFAA gold评估clumping/precision/recall，对GitHub ground truth评估recall；并通过under‑merge、over‑merge、calibrated三种方法对后续分析的影响进行对比。

**⚠️ 局限性**

仍有约1.83%提交未解析，质量分类为启发式，缺乏仓库级证据，数据仅为snapshot，GitHub与ALFAA gold不完整，跨语料链接仍面临clumping挑战。

---

## 375. When do prophets profit in prediction markets?

**arXiv ID:** 2607.06166 | [PDF](https://arxiv.org/pdf/2607.06166v1)

**作者:** Anri Gu `[一作]` (University of Chicago), Haifeng Xu `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了在任意预测市场中将准确性转化为盈利的“proper betting”策略，并证明其在足够流动的市场上几乎唯一可实现稳健盈利。

**💡 创新点**

其创新点在于：①将预测准确性与利润之间的等价性从 AMM 推广到一般市场；②给出一种与严格合适的打分规则对应的投注策略，并证明它是唯一稳健盈利的；③提供了新的用打分规则判定可行性的阐述。

**🔧 技术方法**

采用的技术包括：严格合适分数规则与 Bregman 散度的理论框架；利润分解证明；最优投注策略推导；对 AMM 成本函数与市场打分规则的凸分析；以及对实际订单簿的价格冲击函数建模。

**📊 数据集**

数据集主要来自 Prophet Arena 中的 2,418 个 Kalshi 预测市场的 AI 预测记录（含预测、市场价格与结果），以及对该数据的离线回测和 26 天的实时部署。

**📈 对比分析**

方法与传统启发式策略（最高边际、逆边际、Kelly 等）比较，使用 ROI、回报与 Sharpe 比率评估；实验显示 proper betting 在 Brier 分数下取得最高 ROI（+80.33%）且 Sharpe 达 3.35，而其他策略普遍为负或波动大。

**⚠️ 局限性**

限制在于：理论基于足够流动性与无交易费用的假设；在低流动性或高价差时需要额外调整；当前仅给出期望收益保证，未覆盖风险调整后的表现；对真实市场的适应仍需进一步研究。

---

## 376. High-Resolution Artwork Outpainting with Global Blueprint Guidance and Layout Control

**arXiv ID:** 2607.06162 | [PDF](https://arxiv.org/pdf/2607.06162v1)

**作者:** Junha Kim `[一作]` (Hanyang University), Donghyeon Cho `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于全局蓝图的两阶段扩展式图像扩充框架，能够在艺术图像的高分辨率下实现布局可控的无缝扩展；

**💡 创新点**

创新点在于：①使用低分辨率蓝图作为全局规划，解决了传统递进式方法的误差累积问题；②在蓝图生成阶段引入布局适配器与门控融合，实现了基于边界框的细粒度空间控制；③在高分辨率局部合成阶段利用低频保持的前向扩散初始化和蓝图引导特征，实现了多GPU并行生成，显著提升推理速度；

**🔧 技术方法**

技术包括：两阶段扩散模型（Stable Diffusion Inpainting U-Net）、布局适配器、门控融合、注意力引导噪声优化、低频保持前向扩散、位置编码、位置令牌、以及前向扩散初始化；

**📊 数据集**

使用了HumanArt、WikiArt、LAION-5B等大规模艺术图像数据集进行训练，评估时采用IconArt、Cleveland Museum of Art、Art Institute of Chicago等数据集；

**📈 对比分析**

与PQDiff、SD Inpainting、PowerPaint、ProOut等基线进行对比，实验显示本文方法在FID、pFID、CLIP-S、CLIP-A、AP、IoU和推理时间上均优于基线，尤其在高扩展比例下仍保持低FID和高布局准确率；

**⚠️ 局限性**

局限性包括：仍需手动提供布局信息（边界框与描述），对极端极宽高分辨率扩展可能需要更大蓝图；此外，蓝图生成阶段的计算开销相对较高，且在极大图像尺寸下多GPU并行仍受限于显存；

---

## 377. When Does Tool Use Increase the Expressive Power of Finite-Precision Recurrent Models?

**arXiv ID:** 2607.06155 | [PDF](https://arxiv.org/pdf/2607.06155v1)

**作者:** Nikola Zubić `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究工具调用如何提升固定有限精度序列模型的计算表达能力，并给出完整的表达力二分和资源计数。

**💡 创新点**

证明有限状态工具不增加表达力，而单一无限读写移动工具可实现图灵完备性，并给出指数级 EQ_n 分离；同时展示一层选择性仿射 SSM 可精确实现此计算。

**🔧 技术方法**

形式化有限状态控制器模型、工具接口抽象、产品状态模拟、图灵机模拟、选择性仿射状态空间模型构造等理论技术。

**📊 数据集**

本论文为理论工作，未使用任何实验数据集。

**📈 对比分析**

通过理论证明与对比正则语言与图灵可判定语言的差异，给出显式状态数与内存量的上界与下界，展示了指数级表达力提升。

**⚠️ 局限性**

仅适用于确定性有限状态控制器与离散工具，未考虑学习、概率化、或多工具并行；模型假设工具接口有限或可局部操作。

---

## 378. APVI-SLAM: Real-Time Acoustic-Pressure-Visual-Inertial Localization and Photorealistic Mapping System in Complex Underwater Environment

**arXiv ID:** 2607.06222 | [PDF](https://arxiv.org/pdf/2607.06222v1)

**作者:** Hanwen Zhang `[一作]` (Hong Kong University of Science and Technology), Sai-Kit Yeung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出APVI-SLAM，一个实时的多传感器融合SLAM系统，结合视觉、IMU、DVL和压力计，实现高精度定位与光学逼真的水下三维重建。

**💡 创新点**

核心创新点包括：①可靠性感知的传感器融合框架，利用粗细分辨机制动态调整各估计器权重；②滑动窗口冻结策略，快速恢复视觉跟踪并避免重新初始化；③基于四叉树的3D高斯映射方案，兼顾水介质建模与增量重建，显著提升纹理细节与光照真实性。

**🔧 技术方法**

技术手段包括：基于因子图与ESKF的视觉-惯性与DVL-惯性-压力估计、权重重分配、特征点可靠性评估、光照衰减模型、3D高斯渲染、四叉树引导稠密化与贝叶斯传播。

**📊 数据集**

使用的主要数据集：公开的Tank数据集、由UUV Simulator构建的六序列模拟数据集（包含清晰与低能见度场景），以及自制的珊瑚礁实测数据集（8序列，含低纹理、浑浊环境）。

**📈 对比分析**

与ORB‑SLAM3、VINS‑Fusion、UVA、AQUA‑SLAM、Go‑SLAM、Photo‑SLAM、MonoGS、VINGS等基线比较，APVI‑SLAM在定位AET误差上平均降低约26%（与AQUA相比），PSNR/SSIM在模拟数据集上显著领先，滑动窗口冻结策略实现100%成功率并将重新初始化时间缩短数十秒，整体保持实时帧率。

**⚠️ 局限性**

局限性包括：对动态物体和快速运动的鲁棒性不足；水介质抑制与光照模型仍需进一步提升；3D高斯几何结构的完整性与细节重建仍有提升空间。

---

## 379. FDIFormer:Protocol-Aware Transformer Learning for False Data Injection Attack Detection in Smart Grid Networks

**arXiv ID:** 2607.06213 | [PDF](https://arxiv.org/pdf/2607.06213v1)

**作者:** Sandara Sathsarani Wijethunga `[一作]` (Deakin University), Nasrin Sohrabi `[通讯]` (Deakin University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种无手工特征工程的框架FDIFormer，用结构化文本表示方式将IEC 61850 GOOSE 包序列转化为可供 Transformer 模型直接学习的输入，并实现了False Data Injection（FDI）攻击检测；

**💡 创新点**

创新点在于：1）构造与代码语法相似的结构化文本表示，2）首次在 GOOSE 流量上应用预训练 Transformer 进行细粒度攻击检测，3）在不依赖专业协议特征的情况下实现与基于特征的 XGBoost 等方法相当的检测效果；

**🔧 技术方法**

主要技术包括结构化文本生成、滑动窗口采样、Fine-tune 预训练 Transformer（BERT、DistilBERT、CodeBERT、GraphCodeBERT、ModernBERT 等）以及混合模型；

**📊 数据集**

使用 QUT-ZSS-2023-GOOSE 数据集，该数据集包含 11 个正常场景和 9 个 FDI 攻击场景，共 46,551 个 GOOSE 包；

**📈 对比分析**

通过场景级三折交叉验证，将 Transformer 与手工特征 XGBoost/LightGBM、TF-IDF 基线以及混合模型进行对比；结果显示 GraphCodeBERT 获得 MCC 0.595 ±0.122，接近手工特征基线的 0.604，显著优于 TF-IDF 基线，说明预训练表示对检测性能贡献最大；

**⚠️ 局限性**

局限性在于：仅使用单一数据集且攻击场景有限，无法充分评估模型在更大、更加平衡的数据集上的泛化能力；缺乏实时推理延迟评估以及对不同子站部署的实际可行性验证。

---

## 380. Designing Maintainable Hybrid Generative Systems: A Quantum-Inspired Approach to Automated Music Harmony Generation

**arXiv ID:** 2607.06296 | [PDF](https://arxiv.org/pdf/2607.06296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 381. Physics-Informed Neural Embeddings of PDE Solution Families

**arXiv ID:** 2607.06348 | [PDF](https://arxiv.org/pdf/2607.06348v1)

**作者:** Raul Jimenez `[一作]` (University of Barcelona), Pedro Tarancón-Álvarez `[通讯]` (University of Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过多头物理信息神经网络学习非线性偏微分方程（如粘性 Burgers、热方程和波方程）解集的低维嵌入，并从中提取解空间的几何结构。

**💡 创新点**

创新点在于引入头正交化惩罚消除潜在的嵌入退化，得到训练无关的主成分谱和傅里叶壳层频谱，揭示解空间的低维组织和尺度分布。

**🔧 技术方法**

使用技术包括多头 PINN、主成分分析（PCA）、傅里叶壳层谱分解、正交化惩罚项以及误差界的理论推导。

**📊 数据集**

实验数据来自一维粘性 Burgers 方程、热方程和波方程，初始条件采用傅里叶、抛物多项式和瑞克尔波三类族，无预先生成的数值解数据集。

**📈 对比分析**

与高精度数值解比较，网络预测误差约 10⁻²；主成分显示 2–4 维即可解释 95% 的潜在方差，频谱在不同训练运行中保持一致，表现出显著的压缩和鲁棒性。

**⚠️ 局限性**

局限性包括仅在一维、无激励、线性/非线性小范围内验证；高维、湍流或随机参数场等更复杂系统的适用性尚待进一步研究。

---

## 382. Driving the Wrong Way: Leveraging Interpretability in End2End Autonomous Driving Models

**arXiv ID:** 2607.06328 | [PDF](https://arxiv.org/pdf/2607.06328v1)

**作者:** Franz Motzkus `[一作]` (AUMOVIO), Sebastian Bernhard `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在端到端自动驾驶模型中引入稀疏字典学习（Sparse AutoEncoder）作为后置可解释层，对潜在空间进行概念分解，并通过概念层面干预来改进决策，最终提升了导航性能。

**💡 创新点**

首次将概念化的稀疏字典学习与端到端驾驶模型结合，构建可解释的潜在空间并直接关联到轨迹评分子目标，实现对模型内部决策的因果级别解释与可操作性干预。

**🔧 技术方法**

稀疏自编码器（Top‑k、Matryoshka、Archetypal 变体）、概念相关性传播（Concept Relevance Propagation）、循环/Transformer 端到端网络、回归/分类评估线性探针、循环网络（GTRS、iPAD）等。

**📊 数据集**

navsim 仿真数据集，用于训练和评估端到端驾驶模型及其潜在空间。

**📈 对比分析**

通过在 GTRS 模型上对 SAE 进行参数搜索，并将干预后的模型与无 SAE 基线进行比较，发现去除 3 个关键概念后 epdms 分数从 0.524 提升至 0.5926（+0.096），显著提升了碰撞避免、车道保持、行驶方向和时间‑到‑碰撞等子指标。

**⚠️ 局限性**

受限于模型对统计相关性的过度依赖，概念解释仍难以完全捕捉因果关系；SAE 训练参数需针对具体模型手动调优；仅在 navsim 任务上验证，缺乏在真实道路或多模态数据上的通用性验证。

---

## 383. Dithered Gaussian Mechanism for Randomness-Efficient Differential Privacy

**arXiv ID:** 2607.06320 | [PDF](https://arxiv.org/pdf/2607.06320v1)

**作者:** Nikita P. Kalinin `[一作]` (Institute of Science and Technology Austria), Rasmus Pagh `[通讯]` (BARC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在差分隐私中使用的抖动高斯机制，直接通过随机格点平移和四舍五入实现离散化，避免浮点漏洞并继承连续高斯机制的隐私保证。

**💡 创新点**

创新点在于将离散化作为高斯机制的后处理，使用公开的随机位移提升采样效率，并证明随机位移不影响隐私；同时通过低熵分布实现显著减少私有随机位数。

**🔧 技术方法**

采用高斯噪声、均匀抖动、离散化后处理、算术编码等技术进行高效采样，并在PyTorch框架下实现。

**📊 数据集**

主要实验数据集为CIFAR-10，用于评估DP‑SGD训练的模型准确率；还在理论实验中使用单位球分布向量对比熵和RMSE。

**📈 对比分析**

与标准高斯机制和离散高斯机制比较：熵显著降低（独立于噪声尺度），RMSE接近连续高斯；在DP‑SGD中，准确率几乎不变，训练时间增加约20–30%。

**⚠️ 局限性**

局限性包括：机制不可加法化，难以在分布式/联邦学习中直接使用；仍可能受到计时攻击；对公开随机位移的安全性依赖，需硬件支持才能进一步降低开销。

---

## 384. AlayaWorld: Long-Horizon and Playable Video World Generation

**arXiv ID:** 2607.06291 | [PDF](https://arxiv.org/pdf/2607.06291v1)

**作者:** AlayaWorld Team `[一作]` (Alaya Lab), Zihui Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 AlayaWorld 框架，实现了在多种真实、游戏与合成场景中可交互、可导航且支持文本驱动开放式动作的可持续生成世界。

**💡 创新点**

创新点包括：①将显式渲染的 3D 缓存与轻量 AdaLN 相机调制结合；②在分块级别实现 prompt 切换；③采用空间缓存与压缩时间记忆的双重记忆策略；④训练时引入错误银行以提升长视程稳健；⑤采用 4 步少步蒸馏实现 720p 24fps 的实时生成。

**🔧 技术方法**

主要技术：变形扩散 Transformer（DiT）、AdaLN 相机控制、3D 缓存渲染、帧压缩记忆、错误银行、少步蒸馏、分块生成、prompt 切换、长程空间记忆。

**📊 数据集**

数据来源：以 LTX-2.3 预训练模型为基准，在大规模无标签视频数据（覆盖真实、游戏、合成等多样场景）上进行微调，没有使用单一公开数据集。

**📈 对比分析**

评估方法：在与 Genie、GameNGen、Yume 等现有交互式世界模型相同的分辨率（720p）与条件下，对 leave‑and‑return 轨迹与长视程生成进行定性对比。AlayaWorld 在相机控制精度、空间一致性、长视程稳定性方面优于对比模型，并实现 4 步/秒的实时生成（约 24fps）。

**⚠️ 局限性**

局限性：①空间缓存主要适用于静态结构，难以完美捕捉长时间移动的动态物体；②需要深度/几何估计，增加额外算力与工程成本；③缺乏量化指标，评估主要基于定性对比；④对极端动作或超大范围探索的鲁棒性尚待进一步验证。

---

## 385. From Sinhala to Dhivehi: Cross-Lingual Transfer Learning for Low-Resource Speech Recognition

**arXiv ID:** 2607.06289 | [PDF](https://arxiv.org/pdf/2607.06289v1)

**作者:** Lukmal Ilyas `[一作]` (Informatics Institute of Technology), Nevidu Jayatilleke `[通讯]` (University of Moratuwa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了利用Sinhal语（与Dhivehi属于同一语言族）进行跨语言迁移学习，以提升Maldives国土语言Dhivehi的低资源自动语音识别（ASR）性能。

**💡 创新点**

创新点在于系统性比较了五种迁移策略（单语、序贯微调、多语种微调、持续预训练以及非相关语言控制），并通过Turkish作为控制实验验证语言相关性对迁移效果的影响，发现持续预训练结合语言模型解码是最优方案。

**🔧 技术方法**

采用的技术包括Wav2Vec2自监督预训练模型、Fine‑tune、连续预训练（CPT）、多语种联合微调、语言识别标记实验以及KenLM 5‑gram语言模型与CTC深度搜索解码器。

**📊 数据集**

使用的数据集为：Sinhal语 OpenSLR（约224小时）、Dhivehi Common Voice 22.0（共61小时，其中37小时验证有效）以及Turkish Common Voice 22.0（约60小时）做为控制。

**📈 对比分析**

通过对17个实验配置的WER/CER进行比较，持续预训练+KenLM解码得到12.89% WER、2.70% CER，较Dhivehi‑only基线（13.50% WER）提升约13.5%，Turkish控制实验进一步证明了语言相关性带来的提升，语言识别标记在此低资源语境中并不显著有益。

**⚠️ 局限性**

局限性包括训练规模受限、不同模型架构导致的比较不一致、缺乏多次复现与交叉验证、对语言识别标记效应的理解不足以及仅使用n‑gram语言模型而非更强的Transformer语言模型。

---

## 386. Optimal Transport Q-Learning for Flow Policy Steering and Acceleration

**arXiv ID:** 2607.06262 | [PDF](https://arxiv.org/pdf/2607.06262v1)

**作者:** Andreas Sochopoulos `[一作]` (University of Edinburgh), Sethu Vijayakumar `[通讯]` (University of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用机器人自身经验对预训练的流式策略进行强化学习后训练，实现策略的微调与加速；

**💡 创新点**

将优势加权的条件最优传输（wCOT）与条件流匹配（CFM）结合，形成一种能够在极少交互下逼近优势加权策略的流模型；

**🔧 技术方法**

采用神经ODE、条件最优传输、条件流匹配、优势函数估计、经验回放、熵正则化OT、拒绝采样与动作分块等技术；

**📊 数据集**

在OGBench离线RL基准、MuJoCo仿真任务以及四个真实机器人任务（堆叠、精确抓取、接触推挤、精准抓取）上进行实验；

**📈 对比分析**

与DSRL、Q-Chunking、FQL、QAM-F、EWFM等基线进行对比；OTQL在离线、离线‑在线和在线设置下平均成功率最高，NFE从10降至3，推理速度提升约70%，在真实任务中成功率从36%提升至86%；

**⚠️ 局限性**

需使用可微分的神经ODE流模型，非所有VLA/WAM适用；在高维情形下OT误差增大导致加速效果下降；在真实RL中仍需人工标注和环境重置，适配性随任务维度而减弱。

---

## 387. The Minimum Dominating Set Problem on Bipartite Circle Graphs: Complexity and Approximation

**arXiv ID:** 2607.06251 | [PDF](https://arxiv.org/pdf/2607.06251v1)

**作者:** A. Karim Abu-Affash `[一作]` (Shamoon College of Engineering), Joseph S. B. Mitchell `[通讯]` (Stony Brook University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了双色圆图（bipartite circle graphs）上的最小支配集问题，并证明该问题即使在此受限图类中仍为NP难；提出了基于动态规划的2-近似算法以及通过局部搜索得到的多项式时间近似方案（PTAS）。

**💡 创新点**

① 证明了双色圆图中的最小支配集仍为NP难，首次给出从平面单调3SAT的多项式约简；② 设计了全局最优红蓝交替覆盖的动态规划，得到2-近似；③ 引入交换图与平面分离器框架，证明局部搜索可获得PTAS。

**🔧 技术方法**

多项式约简、动态规划、区间模型、局部搜索、平面图分离器（Separator）与交换图（exchange graph）技术。

**📊 数据集**

本文为理论分析，未使用实验数据集。

**📈 对比分析**

通过证明与已知APX-hard结果的关系，2-近似的比值为2；PTAS在固定ε下运行时间为(n+m)^{O(1/ε^2)}，可获得(1+ε)-近似。

**⚠️ 局限性**

仅适用于双色圆图；PTAS的指数因子随1/ε^2增长，实际应用受限；未给出最优算法或更快近似。

---

## 388. RoboVAST: Automated Scenario-Based Validation of Robots at Scale

**arXiv ID:** 2607.06248 | [PDF](https://arxiv.org/pdf/2607.06248v1)

**作者:** Frederik Pasch `[一作]` (Karlsruhe University of Applied Sciences), Nico Hochgeschwender `[通讯]` (University of Bremen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 RoboVAST 框架，用于大规模的场景化机器人验证，支持分层场景规格、插件化变异、基于容器的并行执行以及结果处理与评估。

**💡 创新点**

创新点包括：① 通过可组合的场景规格和显式的变异维度实现系统化的场景生成；② 使用有序生产者（ordered producers）实现可重现的随机采样；③ 将验证流程完全声明式化，利用 YAML、Kubernetes 与 Kueue 实现弹性扩展；④ 构建公开的 5480 个导航场景、10万+ 运行的验证数据集，为大规模实验提供基准。

**🔧 技术方法**

技术细节包括：YAML 声明式配置、插件化变异系统（Floorplan, PathVariation, ObstacleVariation 等）、ROS 2 与 Nav2、Scenario Execution 作为模拟层、容器化部署与 Kubernetes 调度、Kueue 资源管理、统计判定（binomial 置信区间）、热图可视化、Jupyter Notebook 结果处理。

**📊 数据集**

数据集：5 个室内地图（真实与合成）构成 5480 个场景变体，10万+ 次仿真运行，累计 1800 小时仿真时长与 1873 公里行驶距离，覆盖性能、鲁棒性与安全性三类评价指标。

**📈 对比分析**

对比方法：每个配置执行 20 次运行，采用二项分布模型估算成功率，95% 置信区间宽度约 ±0.22；结果显示性能 98.12% 成功率、鲁棒性 82.8% 成功率、安全性 61.24% 成功率；通过热图与失败密度可视化验证系统对场景特征的敏感性。框架在 20 节 C4 节点、1920 CPU 的 GKE 集群上完成，展示了可扩展性与统计可靠性。

**⚠️ 局限性**

局限性：仅在 TurtleBot4 + Nav2 的单机器人静态导航场景中验证；缺乏多机器人交互、复杂动力学、真实世界测试；仿真与实测差距未解决；仅集成 Gazebo，未来计划加入 Isaac Sim、O3DE 等；需要进一步探究更高维度组合覆盖策略。

---

## 389. WING: A Window-Prior-Based Generative Network with Gated Inception for Cross-Modality CT Synthesis

**arXiv ID:** 2607.06234 | [PDF](https://arxiv.org/pdf/2607.06234v1)

**作者:** Siyuan Mei `[一作]` (Friedrich-Alexander-Universit"at Erlangen-N"urnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universit"at Erlangen-N"urnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了基于CT窗口先验的生成网络WING，用于跨模态生成高质量CT图像，解决了CT长尾分布导致的回归难题。

**💡 创新点**

创新点包括：①将回归目标转化为多窗口表示并采用软融合，充分利用CT窗口的结构确定性；②设计了Gated Inception Generator（GIB）和Fuse‑and‑Refine Transformer（FRT）实现多窗口预测与细节细化；③使用联合对抗训练提升窗口级与全尺度CT的逼真度。

**🔧 技术方法**

技术手段包括MedNeXt骨干网络、Gated Inception Block、多分支卷积、Transformer细化、PatchGAN联合判别器以及软融合窗口拼接。

**📊 数据集**

使用SynthRAD2025公开数据集，包含约600对MRI‑CT和950对CBCT‑CT，覆盖头颈、胸部和腹部三大解剖区域。

**📈 对比分析**

与多种GAN、U‑Net、SwinUNETR等方法对比，WING在MAE、MS‑SSIM、PSNR和DICE等指标上均优于现有SOTA，在MRI‑CT和CBCT‑CT两项任务上均获得最高分，参数量保持在可接受范围。

**⚠️ 局限性**

局限性包括：窗口设置固定，可能不适用于非标准扫描协议；生成与融合分离导致实现复杂度上升，未来需要更统一、适应性更强的窗口建模方案。

---

## 390. Improved subexponential analysis of the Random-Action-Removal algorithm for 2-player turn-based games and non-binary AUSOs

**arXiv ID:** 2607.06334 | [PDF](https://arxiv.org/pdf/2607.06334v1)

**作者:** Uri Zwick `[一作]` `[通讯]` (Tel Aviv University), Uri Zwick (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对解决2人、零和、基于回合的随机或非随机图形游戏的算法进行了简要描述和改进分析，算法可用于找到非二元超立方体的唯一沉没方向的沉没点。

**💡 创新点**

创新点在于提出了一种更快的随机算法，改进了Hansen和Zwick（2015年）提出的复杂算法，且在处理具有超立方体结构的游戏时，运行时间得到了显著优化。

**🔧 技术方法**

使用了随机面算法的变体，该算法用于解决线性规划问题，特别是其双重变体。

**📊 数据集**

使用的游戏包括折扣和非折扣的随机游戏（SGs）以及平均收益游戏（MPGs），并且算法适用于具有n个状态和m≥2n个动作的游戏。

**📈 对比分析**

与Hansen和Zwick的算法相比，本文的算法在复杂度上从e^O(√(nln(m/√(n))))改进到e^O(√(nln(m/n)))，尤其在m=O(n)的情况下，运行时间从e^O(√(nln n))降至e^O(√(n))。

**⚠️ 局限性**

限制在于改进的分析仅适用于具有超立方体结构的游戏，对于缺乏这种结构的一般LP类型问题，无法提供相应的改进。

---

## 391. LAMP: Latent Motion Prior-Guided Real-World Learning for Dexterous Hand Manipulation

**arXiv ID:** 2607.06323 | [PDF](https://arxiv.org/pdf/2607.06323v1)

**作者:** Xinye Yang `[一作]` (Fudan University), Chao Yu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个基于历史条件的隐式运动先验（LMPM），将高维手部动作映射到可解码的低维潜在空间，并在此空间上构建三阶段实地学习框架（先验学习、行为克隆、残差强化学习）来训练可在真实机械手上执行的视觉‑运动策略。

**💡 创新点**

将隐式运动先验与行为克隆和残差RL共享，利用历史条件的连续可解码潜在动作空间，使学习和在线探索局部且符合接触的手势，从而显著提高高维手部学习的样本效率和安全性。

**🔧 技术方法**

使用变分自编码器训练潜在运动先验，行为克隆网络结合图像与手势历史，残差SAC算法在潜在空间执行强化学习，并配合视觉奖励分类器与实时双摄像头观测。

**📊 数据集**

基于Frank Research 3机械臂与Ruiyan dexterous hand的实测数据，使用四个任务（Grasp & Place、Open Drawer、Pull Tissue、Assemble Box）的人机演示数据作为训练与评估数据集。

**📈 对比分析**

与原始高维手部动作、PCA线性压缩、VQ‑VAE离散码本三种接口在同一学习流程下对比，IL成功率从Raw的0%提升至LMPM的56.25%，随后RL后平均成功率达到98.75%，在三个任务实现100%成功，剩余任务达到95%。

**⚠️ 局限性**

先验学习依赖任务特定的手势数据，难以直接迁移到新任务；高维手部可能需要更大容量的模型和更广泛的数据集，且在更复杂或多任务场景下的泛化能力尚未验证。

---

## 392. Visual graphs for image classification: does the structure affect performance?

**arXiv ID:** 2607.06295 | [PDF](https://arxiv.org/pdf/2607.06295v1)

**作者:** Alessandra Ibba `[一作]` `[通讯]` (University of Sassari), Alessandra Ibba (University of Sassari)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统比较了三种节点抽取方法（Grid、Superpixel、Interest Point）与三种边构造/稀疏化策略（Base、LDM、LSP、MinCONN）对三层GCN图像分类性能的影响

**💡 创新点**

创新点在于将图结构的生成与稀疏化方法统一实验，证明稀疏小世界图（如MinCONN）能在保持或提升精度的同时显著降低连接数，并揭示特征维度与图结构相互促进的关系

**🔧 技术方法**

使用的技术包括：ViT、ResNet18、SIFT特征提取；节点抽取为Grid/Superpixel/Interest Point；边构造采用6‑NN KNN；稀疏化方法有平均距离剪枝（LDM）、局部敏感剪枝（LSP）和最小生成树（MinCONN）；三层GCN+ReLU+全局平均池化+Softmax分类器

**📊 数据集**

数据集为Fashion‑MNIST（70,000张28×28灰度图，10个类别）

**📈 对比分析**

通过10折交叉验证 + 单独测试集，比较不同组合的准确率与F1；最佳组合（Interest Point + ViT + MinCONN）达到86.81%准确率，整体相较于基线提升约4%，且稀疏图结构往往优于完整连接图

**⚠️ 局限性**

局限性包括仅在单一灰度小型数据集上验证，未探究高分辨率和彩色图像；缺乏对网络内部学习动态的结构分析；未与最先进模型进行基准比较

---

## 393. Rethinking Fronthaul Topologies for Cell-Free 6G Networks

**arXiv ID:** 2607.06288 | [PDF](https://arxiv.org/pdf/2607.06288v1)

**作者:** Max Franke `[一作]` (Technical University Of Berlin), Giuseppe Caire `[通讯]` (Technical University Of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对细胞无干扰（Cell‑Free）6G 网络的前置链路（fronthaul）设计，提出了多种拓扑结构并分析其对链路负载的影响；同时提出了基于需求感知与流量工程的处理节点分配算法，评估其对最大链路流量的降低效果。

**💡 创新点**

创新点主要包括：① 将传统的树形前置链路拓扑与Clos拓扑进行对比，证明后者在大规模部署时几乎可达最优负载分布；② 设计了同时兼顾单播与多播的流量模型，并提出混合式分配算法，可将链路负载降低至20%以内；③ 通过混合整数线性规划（MILP）构建基准最优解，为实际网络设计提供量化参考。

**🔧 技术方法**

使用了：分布式MIMO网络模型、RAN链路容量与量化误差模型、块时变信道与TDD时隙分配、LMMSE组合与最优SINR近似、Clos与树形拓扑的网络图模型、Python 3.10+Matplotlib进行仿真、线性规划求解器求解MILP。

**📊 数据集**

实验数据基于随机生成的用户与天线位置，覆盖两种用户密度（K=3.5L、5L、6.5L）以及热点分布；使用的信道衰落模型为3GPP UR‑MC 路径损耗，频段为 sub‑6 GHz，带宽W=100 MHz。

**📈 对比分析**

通过比较树形、Clos、近似最优（MILP）三种拓扑，以及随机、固定、感知、混合四种算法，评估每种组合下的最大链路负载。实验表明，Clos拓扑将最大链路流量降低约75%（相比树形），混合算法在不同拓扑下可进一步减少约20%链路负载；与理想完全图相比，Clos在大规模部署时表现接近最优。

**⚠️ 局限性**

主要局限包括：① 仅在 sub‑6 GHz 频段实验，未覆盖 mmWave 或 sub‑THz 的更高带宽需求；② 采用理想化的光纤链路容量（50/100 Gbps）与无排队延迟的假设；③ 处理节点分配算法为启发式，未给出全局最优证明；④ 仅考虑静态用户分布，未模拟用户移动或动态重分配。

---

## 394. Quality-Aware Personalized AI Service Provisioning in UAV-Assisted 6G Networks

**arXiv ID:** 2607.06278 | [PDF](https://arxiv.org/pdf/2607.06278v1)

**作者:** Mohammad Farhoudi `[一作]` (Oulu University), Tarik Taleb `[通讯]` (Ruhr University Bochum)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了HyPE框架，用于在UAV辅助的6G网络中进行个性化AI服务的动态规划与部署，融合需求预测、LLM驱动的轨迹与推理决策以及启发式函数布署与路径选择，以实现低延迟与高QoAIS的服务保障。

**💡 创新点**

创新点包括：1) 将移动感知的需求预测与LLM决策相结合，实现对UAV轨迹和推理节点的预测性、上下文感知规划；2) 采用模块化的预处理/后处理链，支持弹性延迟-质量折中；3) 在高度动态的空地网络中引入连续性指标，实现对用户个性化体验的持续保障。

**🔧 技术方法**

技术手段包括：深度强化学习（DRL）预测移动与服务需求；大语言模型（LLM）进行交互式规划与决策；PGA（预测引导分配）函数布署；LBSP（低延迟路径选择）路由；以及结构化提示与Pydantic验证。

**📊 数据集**

使用的数据集为：真实城市用户轨迹数据（Zenodo）与MMMU-Pro AI工作负载数据集。

**📈 对比分析**

实验与最优（Gurobi）、AD‑SAC、JAAPD‑D和随机策略对比，结果显示HyPE在高负载下的服务覆盖率可达约61%（接近最优95%），平均E2E延迟比随机低10‑20%，在输出精度约94%与个性化提升约19%方面亦优于基线。

**⚠️ 局限性**

局限性包括：预测误差与资源饱和导致的性能波动；未全面考虑能源消耗和LLM推理成本；以及对长期持续学习与跨域协同的支持不足。

---

## 395. Learning-based Physics-Constrained Neural Kernel for Sound Field Estimation With Source-Position-Dependent Directional Weighting

**arXiv ID:** 2607.06274 | [PDF](https://arxiv.org/pdf/2607.06274v1)

**作者:** Mattia Marella `[一作]` (National Institute of Informatics), Shoichi Koyama `[通讯]` (National Institute of Informatics)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于物理约束的学习型神经核，用于从微型麦克风测量恢复声场分布。

**💡 创新点**

创新点在于将源位置依赖的隐式神经表示（INR）用于方向加权函数，使核函数能够学习多源共性并在未见源位置上泛化。

**🔧 技术方法**

采用Herglotz波函数、RKHS核回归、随机傅里叶特征（RFF）与三层MLP相结合的神经网络，并利用频域ATF训练。

**📊 数据集**

使用基于图像源法合成的4×6×3米盒形房间RIR数据集，包含100个源、1331个评价点以及18/50个双层麦克风阵列。

**📈 对比分析**

与仅使用单快照的SB-NK和固定均匀权重的Uniform进行对比，实验显示LB-NK在75–525 Hz范围内平均NMSE低于对手，且方差更小。

**⚠️ 局限性**

局限在于仅在单一房间环境下验证，缺乏跨房间泛化与多房间学习的探索。

---

## 396. A Unique Normal Form for Tensor Trains over Arbitrary Fields

**arXiv ID:** 2607.06271 | [PDF](https://arxiv.org/pdf/2607.06271v1)

**作者:** Renaud Vilmart `[一作]` `[通讯]` (Université Paris-Saclay), Renaud Vilmart (Université Paris-Saclay)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了针对任意域上张量训练（Tensor Train, TT）的一种唯一的标准形（normal form），并给出了多项式时间的归约策略，使得任何给定的 TT 都能被压缩到该唯一形式；同时提供了直接从完整张量构造唯一标准形 TT 的算法，并给出了关于标准形大小的上界（≤8/3 ·  ∏ n_i）。

**💡 创新点**

核心创新包括：
- 设计并证明了 LDPU（Lower‑Diagonal‑Permutation‑Unit）分解的存在性与唯一性，作为 SVD 的替代，能够在任何域（尤其是有限域）上有效工作；
- 基于 LDPU 推导出唯一的 TT 标准形，使得张量网络的归约与 BDD 的归约可以统一；
- 给出完整的两次扫（sweep）归约算法，保证在多项式时间内得到最小 TT‐秩；
- 提供了从完整张量直接构造标准形的算法，并给出首非零项索引与系数的快速提取方法；
- 证明了标准形在大小上与最优压缩相当（最多 8/3 乘以完整张量大小）。

**🔧 技术方法**

主要技术包括：
- 线性代数中的 LDPU 分解（即 PLU 的变体）；
- 张量网络与矩阵乘法的图论（字符串图）等价化，用于简化证明；
- 对齐、拆分（split、untwine、intertwine）等矩阵 reshape 操作；
- 通过行/列梯形（REF/CEF）保持秩的归约；
- 多次扫描（sweep）技术，分别保证行梯形和列梯形的唯一性。

**📊 数据集**

本文为理论性工作，未使用任何实验数据集。

**📈 对比分析**

比较方法主要是理论复杂度与空间上限：
- 算法时间复杂度为 O(d n r^3)（使用朴素矩阵乘法）或 O(d n r^ω)（若使用更快的 LU/LDPU）；
- 标准形大小上限为 8/3 · ∏ n_i；
- 与 BDD 在空间上限的对比，二者在最坏情况下均为指数级，但标准形提供了更紧凑的表示。性能评价均以理论分析为主，未给出实验数据。

**⚠️ 局限性**

主要限制：
- LDPU 分解在浮点域上可能存在数值不稳定性；
- 在无限域（如 ℚ、ℝ）下，LDPU 的位数复杂度可能呈指数；
- 目前未证明 LDPU 可以在 O(n^ω) 时间内完成，需进一步研究；
- 对于 𝔽_2 等有限域，可进一步加速（如 4‑Russians 方法）仍是开放问题；
- 实际实现与性能评估尚未完成，需要后续实验验证。

---

## 397. From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution

**arXiv ID:** 2607.06269 | [PDF](https://arxiv.org/pdf/2607.06269v1)

**作者:** Heting Mao `[一作]` `[通讯]` (Shanghai Lixin University of Accounting and Finance), Heting Mao (Shanghai Lixin University of Accounting and Finance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了将大语言模型的认知架构从应用层模拟嵌入模型本体的理论框架，核心包括结构张力驱动、离线递归循环以及推理时可塑性；

**💡 创新点**

创新点在于：①将结构张力作为内生损失函数驱动模型自我一致性；②设计离线递归循环实现自我消化冲突；③把可塑性限定在上下文流形而非权重；④通过治理约束实现可审计、可逆、可变异的多样智能生态；

**🔧 技术方法**

技术手段包括：结构张力公式（预测误差、拓扑不和谐、复杂度权重）、离线递归缓冲区与流形重构算子（扩展、折叠、修剪）、治理沙盒、连续性验证、审计记录以及与结构智能协议兼容的实现；

**📊 数据集**

未给出具体数据集，本文为理论性框架；

**📈 对比分析**

无实验比较或性能数据；作者提出四个可验证的反证标准（T1–T4）以检验框架假设；

**⚠️ 局限性**

主要局限在于：尚未实现或实验验证；流形级可塑性是否足以解决所有冲突未证；治理与审计开销可能过高；多样化生态需要更复杂的监控与治理机制。

---

## 398. Deciding monotonicity of simple drawings of the complete graph

**arXiv ID:** 2607.06240 | [PDF](https://arxiv.org/pdf/2607.06240v1)

**作者:** Oswin Aichholzer `[一作]` (Graz University of Technology), Birgit Vogtenhuber `[通讯]` (Graz University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种 O(n⁵) 时间的多项式算法，用于判定一个完整图 Kₙ 的简单绘图是否可弱等价（或强等价）为 x‑单调绘图。

**💡 创新点**

创新点在于：①通过“摆锤”与“shelling 序列”的组合，对 x‑单调绘图给出了新的必要与充分条件；②利用这些条件构造了可在常数时间查询的三种数据结构，进而在 O(n⁵) 时间内完成判定；③证明弱等价与强等价在此问题上等价，从而把判定问题完整化。

**🔧 技术方法**

使用的技术主要包括：
- 旋转系统（rotation system）和细胞结构的组合描述；
- shelling 序列与部分 shelling 序列的概念；
- 角度化简为摆锤（wedge）判定；
- 递归/动态规划构造的三种查询数据结构；
- 组合几何中的交叉判定与最短摆锤构造。

**📊 数据集**

论文未使用任何实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与已有工作相比：
- 之前针对给定 x‑坐标的嵌入给出了 O(n²) 的判定算法；
- 本文提供了对任意简单绘图的判定，时间复杂度提升至 O(n⁵)；
- 与已知的圆柱单调性 NP‑难结果对比，证明了 x‑单调性判定可在多项式时间完成。
- 由于是理论算法，论文未给出实验性能数据，理论复杂度为主。

**⚠️ 局限性**

局限性：
- 复杂度仍高（O(n⁵)），在大规模实例上实用性有限；
- 算法假设输入已给出细胞结构或旋转系统，实际构造这些结构仍需 O(n²) 或更高时间；
- 只针对完整图 Kₙ，其他图类的单调性判定仍未知；
- 证明依赖于多层递归数据结构，实际实现难度较大。

---

## 399. Spider 2.0-AIFunc: Extending Real-World Text-to-SQL to AI-Native SQL Workflows

**arXiv ID:** 2607.06229 | [PDF](https://arxiv.org/pdf/2607.06229v1)

**作者:** Tianyang Liu `[一作]` (UC San Diego), Yuxiong He `[通讯]` (Snowflake AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并发布了Spider 2.0‑AIFunc基准，评估文本到SQL系统在AI原生SQL（含Snowflake Cortex AI函数）下的表现

**💡 创新点**

首次将AI函数集成到SQL任务中，并设计了基于代理的重写与多轮执行验证流程，保证指令与执行的确定性

**🔧 技术方法**

采用Claude Opus等LLM作为代理进行任务改写，使用SQL执行工具、bash工具和终止工具进行评估，采用多轮执行、稳定性验证

**📊 数据集**

基于Spider 2.0‑Snow的513个任务，最终生成465个验证实例，覆盖125个真实数据库，涉及6类Snowflake Cortex AI函数

**📈 对比分析**

对10种模型（包括Claude、Gemini、GPT‑5.4、Kimi、Qwen、DeepSeek、GLM‑5、MiniMax）在执行准确率上进行比较，闭源模型达70% 以上，开源模型最高58%，显示显著性能差距；对3种传统文本‑SQL框架与Spider‑Agent比较，框架无明显优势

**⚠️ 局限性**

局限在仅覆盖Snowflake与6种AI函数；可能存在指令歧义或多解；构造采用单一LLM，可能引入偏差；评估仅单一轨迹，未充分捕捉随机性；未覆盖其他云平台和更复杂AI函数组合

---

## 400. Harnessing Code Agents for Automatic Software Verification

**arXiv ID:** 2607.06341 | [PDF](https://arxiv.org/pdf/2607.06341v1)

**作者:** Shuangxiang Kan `[一作]` (Singapore Management University), Sebastian Ertel `[通讯]` (Barkhausen Institut)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 LLM 代码代理与验证 harness，自动完成 Iris 核心、RustBelt 与 reglang 等库中所有 4,792 条证明（4,257 + 217 + 318 + 72），从未出现过人工干预或未完成的证明。

**💡 创新点**

创新点在于放弃传统的固定证明策略与手工搜索，直接将完整定理交给通用 LLM 生成，并通过可声明的 HHL harness 提供准确错误回馈、终止检查与完整性验证，三者协同实现 100% 覆盖。

**🔧 技术方法**

关键技术包括 Claude Opus 4.7 代码代理、Coq/Lean 核心验证器、HHL 声明式 harness、会话重用与错误反馈驱动的重试循环，以及多轮风格化后处理。

**📊 数据集**

实验数据集涵盖：Iris 四个核心模块 4,257 条证明；RustBelt 217 条证明；reglang 318 条证明；iris-lean 72 条 Lean 证明，总计 4,792 条目标证明。

**📈 对比分析**

与先前 LLM 验证器（如 PALM、COPRA、Rango、Cobblestone）在 12–48% 成功率的基准上对比，Aria 在同一基准上实现 100% 成功率，平均重试次数仅 0.5 次，平均每条证明耗时 321 秒，显著优于现有方法。

**⚠️ 局限性**

局限性包括对私有 LLM（Claude Opus 4.7）的高度依赖，开源模型效率低、重试次数多；对极长或超出模型上下文窗口的证明仍存在困难；以及在不同证明助手（如 Lean）上的实现仍需要额外适配。

---

## 401. Bridging Diffusion Pruning and Step Distillation with Teacher-Aligned Repair

**arXiv ID:** 2607.06335 | [PDF](https://arxiv.org/pdf/2607.06335v1)

**作者:** Jincheng Ying `[一作]` (Guangdong University of Finance and Economics), Yinhao Xiao `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在稀疏化后通过短暂的教师对齐修复（repair bridge）来直接使用一步生成器的蒸馏，取代传统的后稀疏化多步重训练；

**💡 创新点**

创新点在于：①将稀疏化后的模型与教师在高噪声状态下对齐；②通过教师对齐的修复使得一步蒸馏可收敛；③将结构剪枝与一步蒸馏统一为一种压缩管线；

**🔧 技术方法**

采用结构剪枝（block/channel）、教师对齐修复、SiDA一步蒸馏、Adam优化及VAE编码/解码；

**📊 数据集**

主要使用ImageNet-512（类条件）和CIFAR-10（无条件）数据集；

**📈 对比分析**

与原始EDM2‑XS、SiDA‑EDM2‑XS等方法对比，20%剪枝后模型参数减至98.8M、NFE降至1，FID提升至3.12（比基线3.53好），推理速度提升约54×；在CIFAR‑10上，34%剪枝后模型5.32M参数、FID 5.32，速度提升约40×；

**⚠️ 局限性**

依赖于下游一步蒸馏算法（目前仅支持SiDA/ Diff‑Instruct）；修复阶段仍需真实图像或代表性潜在；对多步蒸馏尚未验证，修复噪声选择等细节需进一步探究。

---

## 402. A unified perspective of Gaussian process approximation for differential equations

**arXiv ID:** 2607.06292 | [PDF](https://arxiv.org/pdf/2607.06292v1)

**作者:** Mengwu Guo `[一作]` `[通讯]` (Lund University), Mengwu Guo (Lund University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种统一的贝叶斯框架，用高斯过程（GP）逼近微分方程，涵盖参数估计与解近似两大典型场景，并把现有多种GP方法映射到该框架中；

**💡 创新点**

将不同的GP微分方程方法整合为一个完整的概率模型，核心创新是“导数匹配”机制将微分算子约束嵌入似然，提供对参数与解同时不确定性的全概率推断，并扩展到弱形式；

**🔧 技术方法**

高斯过程理论、贝叶斯推断、线性微分算子、导数匹配、条件高斯分布推导、参数化与弱形式的推广；

**📊 数据集**

未使用具体实验数据集，主要以理论推导与框架描述为主；

**📈 对比分析**

论文未给出实验比较，讨论的方法可通过最大后验、EM或MCMC实现；性能取决于模型假设、噪声水平与参数空间大小；

**⚠️ 局限性**

主要限制是高维PDE的计算量巨大（维数灾难），需要结合深度学习或稀疏逼近；对非线性约束的求解仍需数值近似；弱形式实现需额外工程实现。

---

## 403. Quantitative Gaussian-Process limits of Tensor Programs

**arXiv ID:** 2607.06290 | [PDF](https://arxiv.org/pdf/2607.06290v1)

**作者:** Andrea Agazzi `[一作]` (University of Bern), Dario Trevisan `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究随机神经网络在无限宽度极限下的高斯过程收敛，给出 Wasserstein 距离下的精确误差上界。

**💡 创新点**

将张量程序框架与 Wasserstein 收敛理论结合，得到对任意架构（含权重共享）的 1/√宽度误差量化。

**🔧 技术方法**

利用张量程序的递归结构、条件高斯分布、CLT 与 Wasserstein 距离、矩阵分块求解、梯度与伪逆等工具。

**📊 数据集**

实验采用合成网络和随机输入，无特定数据集。

**📈 对比分析**

通过 Sliced Wasserstein-1 距离对输出分布进行度量，斜率低于-1/2，表明误差随宽度按 1/√n 缓慢下降。

**⚠️ 局限性**

假设激活函数 Lipschitz、非退化、张量程序有限；未覆盖所有注意力、归一化等现代层；仅给出上界而非下界。

---

## 404. Task Decomposition-Guided Reranking for Adaptive Agent Skill Retrieval

**arXiv ID:** 2607.06283 | [PDF](https://arxiv.org/pdf/2607.06283v1)

**作者:** Yanping Chen `[一作]` (Soochow University), Jiajie Xu `[通讯]` (Soochow University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkillReranker，一种在推理时对LLM代理的技能库进行自适应重排序的框架。

**💡 创新点**

通过先把任务和技能拆分成可执行状态，构建有向无环执行图，并在图中动态识别子任务区间进行局部重排序，实现了任务需求与技能适用性的细粒度对齐。

**🔧 技术方法**

利用LLM进行语义分解与结构化解析、跨编码器（cross‑encoder）评分构建执行图、基于图权重的节点分裂与局部重排序，以及后续去重和自适应技能集合生成。

**📊 数据集**

在ALFWorld和ScienceWorld这两个交互式基准上评估，使用的技能库来自skillsmp.com（约6.8万技能）。

**📈 对比分析**

与LLM‑as‑selector、SkillRouter、Graph of Skills等基线比较，SkillReranker在所有模型（DeepSeek‑v4‑Flash、GPT‑5.4‑Mini、Qwen3.6‑27B）和所有数据集划分上均实现了最高奖励/分数、最低环境步数和最低token消耗，提升幅度显著。

**⚠️ 局限性**

依赖LLM进行状态解析可能导致解析误差，跨编码器评分增加推理开销，且实验仅涵盖文本交互环境，未验证在多模态或真实世界情境中的效果。

---

## 405. Straight-Path Flow Matching for Incomplete Multi-View Clustering

**arXiv ID:** 2607.06281 | [PDF](https://arxiv.org/pdf/2607.06281v1)

**作者:** Yiteng Yuan `[一作]` (Huazhong University of Science and Technology), Lianbo Guo `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于流匹配（Flow Matching）的直线路径（straight‑path）填充框架，用于解决不完整多视图聚类（Incomplete Multi‑View Clustering, IMVC）中的缺失视图恢复与聚类一致性问题；

**💡 创新点**

创新点在于：①用确定性ODE流取代传统扩散模型，直接学习从已观测视图到缺失视图的直线传输；②理论证明该方法在有限步内即可保持集群可分离且无跨集群漂移；③通过聚类层对齐和熵加权对齐实现跨视图聚类一致性，保证向量场学习的聚类结构；

**🔧 技术方法**

使用的技术包括：自编码器（View‑specific Encoder/Decoder）、流匹配的线性插值路径与向量场学习、InfoNCE 对齐损失、熵加权对齐、流预测与一致性损失、ODE数值积分、反向一致性验证；

**📊 数据集**

实验使用的公开数据集有：Synthetic3D、CUB、HandWritten、LandUse‑21、Fashion；

**📈 对比分析**

通过与多种现有IMVC方法（如 DCP、DSIMVC、GCFAGG、CPSPAN、APADC、ProImp、DVIMVC、ICMVC、DCG 等）在 ACC、NMI、ARI 上进行对比，结果显示本文方法在所有数据集及不同缺失率下均取得最优或相近的性能，尤其在高缺失率（如 ρ=0.5）时更为稳健；

**⚠️ 局限性**

局限性包括：①对流匹配假设（集群可分离、视图一致性、模型逼近）要求较高；②聚类对齐损失对超参数较为敏感；③目前仅验证了两视图设置，未扩展到多视图场景；④向量场训练与ODE积分会带来额外计算开销；⑤在某些噪声分布或极端缺失情况下，理论假设可能不完全成立。

---

## 406. AgentTether: Graph-Guided Diagnosis and Runtime Intervention for Reliable LLM Agent Operation

**arXiv ID:** 2607.06273 | [PDF](https://arxiv.org/pdf/2607.06273v1)

**作者:** Chenyu Zhao `[一作]` (Nankai University), Minghua Ma `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了 AgentTether，一种运行时修复框架，利用图引导根因分析和受控干预，在不修改模型或环境的前提下自动诊断并纠正失败的 LLM 代理任务。

**💡 创新点**

创新点在于将图结构的根因定位、跨迭代修复记忆和守护式运行时干预三者结合，形成完整的从诊断到恢复的闭环，显著提升修复效果并减少无效执行。

**🔧 技术方法**

采用 Critical Transition Graph (CTG)、异构图变换器 HGT 与实时 Isolation Forest 检测进行根因定位，LLM 分析师与验证器生成诊断与指引，并在工具调用与文本响应点插入受控干预。

**📊 数据集**

使用三域（Retail、Airline、Banking）的 261 个任务，结合 TerminalBench、SWE-smith、以及自制任务，评估 Qwen3.7-max 与 GPT-5.4 的跨模型迁移效果。

**📈 对比分析**

与盲重试、Outcome Feedback、Reflexion 等基线比较，AgentTether 在 Qwen3.7-max 上整体修复率提升至 69.1%（比盲重试高 26.0pp），在 Banking 领域修复 59.0% 的失败任务；跨模型迁移亦表现优异，同时降低 agent 回合数和 token 消耗。

**⚠️ 局限性**

局限在于仍需依赖 LLM 分析师与验证器的人工/模型判断，干预强度调节易导致过度控制，且在不同工具生态与真实生产环境中的泛化尚未充分验证。

---

## 407. Early Language Learning via Spreading Activation and Category Exploration in Complex Networks

**arXiv ID:** 2607.06258 | [PDF](https://arxiv.org/pdf/2607.06258v1)

**作者:** Salvatore Citraro `[一作]` `[通讯]` (National Research Council), Salvatore Citraro (National Research Council)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于图网络的早期语言习得模型，将词汇学习建模为在词汇图上通过传播激活与类别探索两种机制迭代选词的搜索过程；

**💡 创新点**

创新点在于将时间动态的传播激活与对低频词汇类别的探索抑制相结合，既保留了结构性传播效应，又避免了对已频繁出现类别的过度集中，显著提升了对儿童词汇习得顺序的模拟；

**🔧 技术方法**

使用的技术包括复杂网络建模、扩散式传播激活算法、基于类别频率的探索惩罚机制、以及与最短路径基线的对比评估；

**📊 数据集**

实验所用数据集为四种语言（德语、英语、荷兰语、里奥普拉塔西班牙语）的 SWoW 自由联想网络、WordNet 同义/反义/层级关系以及基于音位相似度的语音层；词汇获得年龄和类别标签来自 Wordbank/CDI；

**📈 对比分析**

通过与最短路径基线以及随机词汇排列的对比，采用覆盖率(k/N)与R(k)/N的恢复效率、以及在类别级别的持久性与突发性(Burstiness)指标，结果显示传播激活模型在覆盖率、持久性与突发性上均显著优于基线，且在多数语言上取得统计显著提升；

**⚠️ 局限性**

局限性包括：仅涵盖可获取自由联想数据的四种语言；使用成人词汇网络可能与儿童词汇结构不符；模型未考虑动态网络演化与多层次共现信息，可能限制了对儿童真实语言环境的逼真模拟。

---

## 408. PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution

**arXiv ID:** 2607.06238 | [PDF](https://arxiv.org/pdf/2607.06238v1)

**作者:** Lihua Wei `[一作]` (ShanghaiTech University), Zhihua Ren `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于物理感知的 MRI 超分辨率框架，利用二维高斯喷射（2D Gaussian Splatting）实现动态分辨率重建，并通过元学习实现从模拟到真实低场 MRI 的迁移。

**💡 创新点**

创新点包括：①将解剖结构先验与系统先验融入高斯原语初始化；②使用基于 T2 加权的物理约束信号模型预测质子密度和 R2 并生成像素强度；③引入共变矩阵词典与位置细化模块，实现对 MRI 采集特性的自适应建模；④采用 MAML 方案缓解配对数据稀缺，提升跨场景泛化。

**🔧 技术方法**

主要技术：二维高斯喷射、SwinIR 编码器、2D U‑Net 分割器、MLP 预测原语参数、协方差词典、基于 Bloch 方程的信号合成、梯度、频域与感知损失、Meta‑Learning (MAML)。

**📊 数据集**

使用的数据集包括：IXI 3T T2 组（模拟成 64 mT LR 与 3T HR），fastMRI 3T 快速磁共振组（仿真 k‑space truncation），Leiden 公开的 64 mT–3T 配对数据，以及 3T–5T 多分辨率真实数据。

**📈 对比分析**

通过与 SwinIR、MetaSR、LIIF、LTE、Pixel‑to‑Gaussian、Diffusion‑based 方法（如 MS‑PRDDiff、Score‑MRI 等）在 fastMRI、IXI 及真实 64 mT–3T/3T–5T 数据集上比较，本文方法在 PSNR、SSIM、HFEN、DISTS 等指标均领先，尤其在 4×/5×/6.4× 超分辨率和真实低场迁移任务中提升 1–2 dB PSNR、0.02–0.05 SSIM。

**⚠️ 局限性**

局限性：需要高质量的解剖分割和系统协方差词典，受限于 T2 加权序列；对多重扫描协议和不同场强的泛化尚需进一步验证；计算成本相对传统 CNN 较高；模型对极端低 SNR 或运动伪影的鲁棒性仍待测试。

---

## 409. Responsible Personalisation: The Double-Edged Sword of Personalisation in Human-Robot Interaction

**arXiv ID:** 2607.06344 | [PDF](https://arxiv.org/pdf/2607.06344v1)

**作者:** Antonio Andriella `[一作]` (Institut de Robòtica i Informàtica Industrial), Wing-Yue Geoffrey Louie `[通讯]` (Oakland University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个基于生命周期与交互情境的负责个性化框架，并对机器人个性化所带来的伦理风险进行系统分析。

**💡 创新点**

创新点在于将个性化生命周期与交互类型（短/长期、开/闭域）相结合，形成风险映射与对应设计建议，填补了现有文献中对HRI个性化风险缺乏系统阐释的空白。

**🔧 技术方法**

采用生命周期分析法、风险分类框架、跨学科工作坊讨论以及对现有HRI案例的文献综述等方法论。

**📊 数据集**

并未使用单一公开数据集，而是基于2023‑2025年RO‑MAN等会议组织的多学科工作坊参与者及公开案例进行归纳与分析。

**📈 对比分析**

该工作主要为概念性框架与设计建议，未进行实验对比；其有效性通过专家评议与案例说明，尚未量化性能指标。

**⚠️ 局限性**

局限在缺乏大规模实证验证、对不同文化与使用情境的泛化性有限、以及对动态伦理决策机制的实现细节不足。

---

## 410. OrchardBench: A Physically-Grounded, GPU-Parallel Apple-Orchard Simulation Benchmark for Agricultural Robotics

**arXiv ID:** 2607.06337 | [PDF](https://arxiv.org/pdf/2607.06337v1)

**作者:** Humphrey Munn `[一作]` `[通讯]` (University of Queensland), Humphrey Munn (University of Queensland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套物理真实、GPU并行的苹果园采摘基准模拟器，能够通过L‑system生成多样化树木、模拟分枝柔性、断裂与果实分离，并配备移动机械臂、深度相机及基准采摘控制器；

**💡 创新点**

创新点在于将L‑system生成的树与Euler‑Bernoulli梁理论相结合，实现树枝的柔性动力学、破裂判定与果实力学；同时通过可调叶片遮挡、域随机化和矩阵无约束求解，支持大规模GPU批处理；

**🔧 技术方法**

技术涵盖：Newton/MuJoCo‑Warp GPU物理引擎、CUDA矩阵无约束求解、L‑system生成、随机化参数、几何果实检测（基于深度相机的球面拟合）、分析控制器和深度相机渲染；

**📊 数据集**

使用公开的物理参数数据（木材密度、弹性模量、破裂应力、果实拉力）以及随机种子生成的苹果树结构；未引用任何真实图像数据集；

**📈 对比分析**

通过与分析基准控制器对比评估，单机RTX 2000可并行512棵树，物理步频约60kfps；基准完成度约12%，吞吐量1.9果/分钟，检测精度0.90，枝断约0.2颗/树，果掉约6颗/树；与现有学习基准相比，仍有显著提升空间；

**⚠️ 局限性**

局限性包括：物理模型未与实际实验验证对比；果实扭转与多轴分离未建模；深度相机渲染缺乏真实光照；缺少真实机器人采摘实验；域随机化对真实转移性尚未证明；数据集仅为随机生成的树形结构。

---

## 411. Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs

**arXiv ID:** 2607.06327 | [PDF](https://arxiv.org/pdf/2607.06327v1)

**作者:** Andrea Alfarano `[一作]` (INSAIT), Marcello Federico `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对22种语言的LLM不确定性估计（UE）方法进行了大规模评估，使用人类标注的MCQA数据集并避免使用LLM-as-judge等噪声评估手段。

**💡 创新点**

首次提供跨语言、跨规模的UE基准；揭示生成语言对低资源语言UE效果的显著影响；证明模型规模对UE方法选择的影响，并提出跨语言阈值校准策略。

**🔧 技术方法**

采用九种主流UE方法（开箱概率、Token Entropy、Self Certainty、Self‑Verbalized、Lexical/semantic consistency、图谱扩散等），利用Prompting、长文本推理、AUROC评估以及阈值校准技术。

**📊 数据集**

使用两套人工审核的多语言MCQA数据集：Global‑MMLU（4选）和MMLU‑ProX（10选），覆盖22种语言，包含高、中、低资源语言。

**📈 对比分析**

在9个模型（Gemma3、Qwen3、Claude 4.5 Sonnet）和22语言上对比，Self‑Verbalized在大规模模型上最高，Token Entropy在小规模模型上更稳健；在低资源语言中，英语推理显著提升UE性能；阈值校准中，语言特定阈值最优，英文单一阈值也显著降低错误率。

**⚠️ 局限性**

局限性：仅评估MCQA数据集，未涵盖开放式生成；仅选取9种UE方法，未考虑训练型方法；实验受限于两个数据集，可能不完全代表真实应用场景。

---

## 412. DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardrail

**arXiv ID:** 2607.06326 | [PDF](https://arxiv.org/pdf/2607.06326v1)

**作者:** He Liu `[一作]` (Ant Group), Zhe Li `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DT‑Guard模型，采用推理激活训练、推理无链思考的安全防护方案；

**💡 创新点**

将安全判断拆分为意图→风险类别→安全级别的结构化决策流程，并通过RG‑PHO将推理监督转化为无推理推断的标签预测；

**🔧 技术方法**

使用意图驱动的多层监督SFT、Rollout‑一致性划分、硬样本SFT、以及基于对比偏好学习的DPO等技术；

**📊 数据集**

构建了包含81万样本的意图驱动安全数据集，涵盖提示侧与响应侧、意图、风险类别与安全标签；

**📈 对比分析**

在10个提示侧和7个响应侧基准上，DT‑Guard在4B模型下实现了平均F1为0.878，超越8B规模的Qwen3Guard与YuFeng‑XGuard等基线；

**⚠️ 局限性**

主要局限是对极端稀缺风险类别覆盖不足，以及推理无链推断仍可能在极端边界案例上产生不确定性。

---

## 413. Synthetic-to-Real Translation for Class-Agnostic Motion Prediction

**arXiv ID:** 2607.06319 | [PDF](https://arxiv.org/pdf/2607.06319v1)

**作者:** Yizheng Wu `[一作]` (Huazhong University of Science and Technology), Guosheng Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了从合成数据迁移到真实数据的运动预测框架SR‑Motion，旨在解决合成‑真实域差导致的运动预测误差；

**💡 创新点**

创新点在于：①引入对象性（objectness）先验的运动预测网络OAMNet，通过学习点到对象中心的偏移，使模型能捕捉对象级别的运动一致性；②设计对象性辅助运动增强模块OAME，利用聚类与一致性筛选去除噪声伪标签，并通过空间一致性平滑提升标签质量；③构建大规模合成4D LiDAR数据集Motion4D，为SRMP提供真实感的训练样本；

**🔧 技术方法**

技术手段包括：基于BEV的STPN网络作为特征提取器；teacher‑student mean‑teacher框架结合EMA更新；OAMNet的对象性分支（偏移回归）和OAME的聚类、噪声过滤与平滑；数据增强（旋转、翻转）；4D LiDAR仿真引擎BLAINDER生成Motion4D；实验中使用smooth L1、交叉熵等损失；

**📊 数据集**

使用的数据集有：合成Motion4D（1370序列、124K帧），真实Waymo（798训练、202验证）和nuScenes（500训练、100验证、250测试）等；实验中也对比了实到实的域适应场景（nuScenes→Waymo、Waymo→nuScenes）。

**📈 对比分析**

与多种基线（MotionNet、WeakMotionNet、SSMP、PillarMotion、SelfMotion、GRL、GU）以及Oracle模型比较，SR‑Motion在Waymo与nuScenes上均取得接近Oracle的误差，特别是fast和static类别的平均误差分别提升超过70%（如fast从2.70↓1.55、static从2.70↓1.54），同时在实到实域适应中也显著优于基线。

**⚠️ 局限性**

限制：网络仅在BEV视角上工作，丢失了细粒度几何信息；假设对象为刚性，难以处理变形或非刚性运动；

---

## 414. Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification

**arXiv ID:** 2607.06309 | [PDF](https://arxiv.org/pdf/2607.06309v1)

**作者:** Aysan Ghayouri Pirsoltan `[一作]` (Iran University of Science and Technology), Mohammad Reza Mohammadi `[通讯]` (Iran University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文提出一种基于令牌的双视图融合与适配框架，利用冻结的大规模视觉 Transformer 对乳腺 X 光的 CC 与 MLO 视图进行共享 Prompt 适配，并在多层 Transformer 深度插入交叉注意力融合令牌，实现视图间信息的层次化交互；

**💡 创新点**

创新点在于（1）将 Prompt 适配与交叉视图融合统一为令牌级操作，避免传统残差式特征融合导致的视图特征与跨视图信息混淆；（2）在多层 Transformer 中多次插入融合令牌，实现跨视图信息的逐层递归传递；（3）利用冻结的视觉基础模型，仅学习极少参数，实现高效迁移与多视图协同；

**🔧 技术方法**

使用的技术包括冻结的 MedSigLIP 视觉 Transformer、深度 Prompt 适配（多层 Prompt 插入）、双向交叉注意力融合令牌、平均/最大聚合策略、交叉熵损失、AdamW 优化器以及 Cosine 学习率调度；

**📊 数据集**

数据集为公开的 VinDr-Mammo（包含 20,000 张四视图乳腺图像，BI‑RADS 级别标签）和 CMMD（5,202 张乳腺图像，恶性/良性标签），均在患者级别划分训练/验证/测试集；

**📈 对比分析**

通过与线性探针、仅 Prompt 学习、仅交叉融合及传统 DIVF 等基线对比，本文方法在 VinDr-Mammo 的 5 类 BI‑RADS 任务中取得 50.40% F1 与 0.8090 AUC；在 CMMD 二分类任务中取得 64.96% F1 与 0.7161 AUC，均明显优于基线；

**⚠️ 局限性**

局限性包括：仍需在特定视图（CC/MLO）上预先裁剪与归一化；对极端样本不平衡时的鲁棒性未知；跨视图对齐仅通过注意力实现，可能在视角差异大时效果受限；模型对冻结视觉基础模型的依赖意味着在不同基础模型上迁移可能需要重新调参；

---

## 415. UI2App: Benchmarking Visual Interaction Inference in Executable Web Application Generation

**arXiv ID:** 2607.06306 | [PDF](https://arxiv.org/pdf/2607.06306v1)

**作者:** Grace Man Chen `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 UI2App 基准，用于评估从静态 UI 截图中推断并实现完整交互行为的能力，构建了一个包含 327 张截图、45 个多路由状态一致截图集的评测数据集，并设计了基于构建可执行性、导航可达性、视觉保真度和交互推断四维度的自动化评测链。

**💡 创新点**

创新点在于：①首个关注交互推断而非仅视觉重建的图像‑仅输入基准；②基于交互类型的七分类交互目录及其覆盖、实现与作用域三轴评估方法，解决多种可能实现的“同义”交互评价难题；③端到端评测流程与人类验证相结合，实现对生成 Web 应用的功能、路由和交互完整性全面量化。

**🔧 技术方法**

使用了最新的视觉‑语言模型（LLM + VLM）如 Claude Sonnet、Gemini、GPT‑5、Kimi、Qwen3.5、GLM‑4.6V，以及 Qwen2.5‑VL 参数阶梯，结合 React+TypeScript 代码框架、Selenium 交互探测、DOM‑to‑DOM 匹配与手工注释技术。

**📊 数据集**

数据集为 UI2App，来源于 164 个可构建的 GitHub 开源项目，经过四阶段自动筛选和三层专家评审，最终得到 45 个多路由参考应用的 327 张截图，覆盖内容、后台、交易、专业四大应用类别，保证跨页面状态一致性和交互暗示丰富。

**📈 对比分析**

对六个前沿模型进行基准测试，评测四项指标：可执行率 [1]、导航可达性、视觉保真度 [VIS] 和交互推断评分 [IIS]。结果显示视觉保真度与交互推断能力不相关，视觉领先模型在 IIS 上仅取得 7.5 分；交互推断平均得分仅为 39.3 分，跨页面状态（C07）几乎所有模型得分为 0，表现出显著瓶颈。

**⚠️ 局限性**

局限性包括：①评测仅限于 React+TypeScript 单页面应用，对其他前端技术栈不适用；②交互推断仍受限于截图信息不足，导致多重实现难以被准确判定；③人类注释成本高，难以扩展到更大规模；④模型在构建可执行性和交互实现方面仍存在较大差距，提示需要进一步提升 VLM 的程序推理与状态管理能力。

---

## 416. Kernel-based Operator Learning: Error Analysis, Budget Allocation, and a Physics-Informed Extension

**arXiv ID:** 2607.06287 | [PDF](https://arxiv.org/pdf/2607.06287v1)

**作者:** Rüdiger Kempf `[一作]` `[通讯]` (University of Bayreuth), Rüdiger Kempf (University of Bayreuth)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了基于核的算子学习，给出了误差分析与预算分配规则，并提出了在在线重构阶段加入物理约束的软约束方法；

**💡 创新点**

创新点包括：① 通过将在线重构视为对扰动数据的恢复，得到学习误差与重构误差的独立分析，进而推导出显式的预算分配条件；② 在不需重新训练的前提下，在重构步骤加入PDE残差惩罚，实现物理信息的软约束；

**🔧 技术方法**

主要技术手段有：核方法与RKHS理论、矩阵值核、插值与正则化最小二乘、代表定理、物理信息Tikhonov正则化以及基于Collocation的PDE残差约束；

**📊 数据集**

实验使用了 Darcy 流动和 Poisson 方程的合成数据，输入为高斯随机场（滤波后裁剪到[-4,4]），共 60,000 对输入-输出样本，分为 50,000 训练和 10,000 测试；输入在 PCA 下压缩到 28 维；输出采用均匀网格；

**📈 对比分析**

与传统的核插值（oracle）以及物理信息 oracle 和 PI surrogate 进行对比。实验表明：在满足预算分配 N≈m^κ（κ≥1）时，误差随 m 的衰减速率达到理论的 m^{-(σ-τ)/d}；PI 方法在重构误差上显著下降，且对正则化参数鲁棒；

**⚠️ 局限性**

主要局限：缺乏对物理信息重构器的收敛速率证明及其对应的预算规则；对 PDE 参数化算子仅给出了经验结论；方法依赖采样算子的可注射性和核的光滑性；在 PDE 依赖输入的场景下在线求解成本仍较高。

---

## 417. DS-MTNet:Structured Multi-Task EEG Decoding for Human-Machine Collaboration

**arXiv ID:** 2607.06297 | [PDF](https://arxiv.org/pdf/2607.06297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 418. MAC-XA: Multi-view Anatomy-Correspondence Fusion for Coronary Stenosis Reporting from X-ray Angiography

**arXiv ID:** 2607.06268 | [PDF](https://arxiv.org/pdf/2607.06268v1)

**作者:** Chen Jia `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了一种基于姿势感知的多视角解剖对应融合框架（MAC-XA），用于从冠状动脉X射线血管造影生成结构化狭窄报告。

**💡 创新点**

①把多视角报告生成建模为“对齐约束聚合”问题；②使用可控合成血管造影生成几何对应监督；③设计跨视角对应模块显式学习并对齐补充视角特征，实现解剖一致的证据聚合。

**🔧 技术方法**

使用RAD-DINO与Pose-ViT编码器，双向对应注意力与多步映射，焦点BCE、交叉熵与负样本正则化，对齐后采用DistilGPT2解码生成报告；评估采用soft Dice、IoU@k、AUPRC、CIDEr、ROUGE‑L等指标。

**📊 数据集**

合成的68k DRR血管造影（来自38位患者的CTA）带有几何对应GT与结构化报告；公开的多视角冠状动脉造影数据集（Mahmoudi等）50个真实病例用于零样本转移与评估。

**📈 对比分析**

与单视角基线、MeanPool、ConcatTok、DuoDuo、CrossAttn比较。合成数据上对应精度soft Dice 23.08，结构化报告提升F1_b +6.11%；真实数据零样本下F1_b +10.19%、F1_bn +11.23%、CIDEr +2.01%、语义等级+6.20%。相较其他多视角方法，MAC‑XA在对应一致性与报告准确性上均取得显著提升。

**⚠️ 局限性**

①合成数据与真实数据分布仍有差距；②模型依赖合成的几何对应GT，真实场景中对应标注困难；③在极端设备遮挡或低对比度情况下表现仍不稳健；④评估样本有限（50例），缺乏大规模临床验证；⑤对解剖细节的分辨率受视角投影限制。

---

## 419. Large language models create an uneven informational layer over cities

**arXiv ID:** 2607.06260 | [PDF](https://arxiv.org/pdf/2607.06260v1)

**作者:** Lin Chen `[一作]` (Northeastern University), Esteban Moro `[通讯]` (Northeastern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对五个美国大城市的社区层级餐厅信息进行LLM输出审计，量化幻觉率与算法可见性缺失，并研究其对空间与人口差异及消费流动的影响。

**💡 创新点**

首次系统评估LLM在城市信息层面的空间与人口偏差，区分知识缺失与选择性关注，揭示LLM对消费模式的重分配作用。

**🔧 技术方法**

使用因素设计合成用户画像，对OpenAI GPT‑4、Meta Llama‑2、Google Gemini进行prompting；通过语义匹配与地理核查计算幻觉率与算法可见性；利用多变量回归与计量模型评估影响；构建收益再分配计量。

**📊 数据集**

SafeGraph餐厅POI与消费数据、Spectus移动位置数据、Yelp评论元数据、美国社区调查（ACS）的人口统计。

**📈 对比分析**

对比三大模型在开放与受限两种查询模式下的幻觉率与可见性，使用多变量回归检验社区属性相关性，并以10%用户采纳率估算消费分布变化；结果显示LLM偏好全服务餐厅并削弱连锁快餐收入。

**⚠️ 局限性**

仅关注餐饮行业，样本限于五大城市，未评估其他语言或地区模型，LLM隐含偏见源自训练语料且难以根除。

---

## 420. Adversarial Robustness for Small Frequency Moments and a Weak Equivalence Theorem for Turnstile Streams

**arXiv ID:** 2607.06312 | [PDF](https://arxiv.org/pdf/2607.06312v1)

**作者:** Elena Gribelyuk `[一作]` (Princeton University), Samson Zhou `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

论文提出了适用于插入-删除（turnstile）流的对抗鲁棒算法，能够在子线性空间内以 (1+ε) 的误差逼近所有 p∈[0,2] 的频率矩（包括 F0 唯一元素计数），并将该框架推广到 EMD、k‑median、香农熵及一类基于 Bernstein 函数的非范数损失估计。

**💡 创新点**

创新点在于：① 将 estimator‑corrector‑learner 框架从 L₂ 迁移到非 Hilbert 空间，利用隐式的 L₂ 等距嵌入与核岭回归实现对 p≠2 的鲁棒估计；② 建立了对抗鲁棒流与传统线性 sketch 之间的“弱等价”关系，揭示 L₁ 嵌入是两种模型共同的根本机制；③ 通过自适应多尺度重构实现 (1+ε) 近似，并通过多级递归控制误差；④ 对 EMD、k‑median 等复杂度量给出了第一个子线性空间的对抗鲁棒实现。

**🔧 技术方法**

主要技术包括：隐式 L₂ 等距嵌入（为 p∈[0,2] 的 F_p 提供等距映射）；正则化核岭回归求解投影问题；多尺度 estimator‑corrector‑learner 递归；对抗鲁棒估计的分块策略；对嵌入与 sketch 的组合；以及利用现有 L₁ 嵌入和线性 sketch 的等价性来推广到更广泛的度量空间。

**📊 数据集**

论文为理论工作，没有使用具体数据集；所有结果均在理论分析与空间复杂度证明上完成。

**📈 对比分析**

与传统（非对抗）流算法相比，算法在子线性空间（polylog n）内实现 (1+ε) 近似，空间上仅比最优解多一个 polylog(1/ε) 乘子；在 EMD、k‑median 等任务上首次给出对抗鲁棒子线性空间算法，误差系数为 O(d log Δ)；对 Shannon 熵的估计也实现了 (1+ε) 近似，空间为 (log 1/ε, log n)。

**⚠️ 局限性**

局限性包括：① 仍需对 L₂ 等距嵌入的存在性和实现做依赖，实际构造复杂；② 对抗鲁棒性仅在子线性空间内；③ 对于 p>2 的频率矩仍无相应的 (1+ε) 对抗鲁棒算法；④ 结果基于“弱等价”理论，真正实现需要对 L₁ 嵌入可计算性做额外假设；⑤ 许多结论在实际大规模数据上尚未通过实验验证。

---

## 421. Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition

**arXiv ID:** 2607.06256 | [PDF](https://arxiv.org/pdf/2607.06256v1)

**作者:** Ke Rui `[一作]` (Simpleai), Minglei Li `[通讯]` (Simpleai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个语义执行框架，用以诊断机器人在长时序任务中，技能完成后仍无法为下一技能准备好状态的“语义交接”失败。

**💡 创新点**

将语义交接问题形式化为“下一技能就绪”，并通过可追踪的多视角VLM验证器与带类型的技能契约，提供了在单技能基准与整体执行之间区分能力与鲁棒性的方法；通过该框架揭示了传统单技能测试无法发现的失败根源，并提出了针对性改进方向。

**🔧 技术方法**

使用π_0.5 Vision‑Language‑Action（VLA）策略（基于PaliGemma视觉骨干+流匹配动作专家），多视角VLM验证器，计划‑执行‑验证‑重规划（plan‑act‑verify‑replan）循环，类型化语义技能契约与可重放的执行轨迹。

**📊 数据集**

BEHAVIOR‑1K（清洗后的技能演示数据集）与OmniGibson仿真环境。

**📈 对比分析**

通过对同一组技能的单技能基准（从干净的演示快照直接调用）与连贯端到端执行（从前一个技能的终止状态调用）进行对比，单技能成功率可达77–100%，但整体任务进度仅约19.5%，失败主要归因于下一技能就绪、目标定位与控制执行。

**⚠️ 局限性**

实验样本有限（仅数十个完整轨迹），VLM验证器可能存在偏差，任务脚本中的对象命名不够规范，恢复策略仅限于重规划，未实现对技能边界状态的即时重置，导致诊断结果可能受限于当前实现。

---

## 422. VendorBench-100: A Unified Cross-Paradigm Benchmark for Deepfake Image Detection

**arXiv ID:** 2607.06254 | [PDF](https://arxiv.org/pdf/2607.06254v1)

**作者:** Sharayu N. Deshmukh `[一作]` (Universidade da Beira Interior), Nilesh K. Deshmukh `[通讯]` (Swami Ramanand Tirtha Marathwada University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了VendorBench‑100跨范式基准，对36个深度伪造检测模型（商业API、零-shot视觉‑语言LLM、开源检测器）在100张极具挑战性的图像上进行统一评估。

**💡 创新点**

提出统一输出模式、统一不拒绝策略、以Matthews相关系数（MCC）为主要排名指标并使用ROC‑AUC做分辨，首次在同一评测框架下比较三类检测范式，并揭示了MCC与ROC‑AUC的根本差异。

**🔧 技术方法**

采用MCC和ROC‑AUC两指标进行排名与分数排序，使用统一的结果记录格式、抗泄漏文件命名协议，针对商业API测量平均延迟；对LLM进行零-shot提示，开源模型在本地GPU推理。

**📊 数据集**

使用VendorBench‑100数据集：100张手工挑选的图像（79张伪造、21张真图），涵盖8类极端失效模式，来源于21个生成器/平台，聚焦真实世界中的高难度案例。

**📈 对比分析**

所有模型在相同数据集上得到MCC、ROC‑AUC、准确率、F1等指标；结果显示商业API在MCC上居首，视觉‑语言LLM居中，开源检测器总体落后，但最高排名的开源模型DRCT在ROC‑AUC上超过所有LLM，体现了指标间的显著不一致。

**⚠️ 局限性**

局限性包括：仅有100张图像且单次评测，缺乏置信区间和显著性检验；LLM结果非统一实时收集；部分开源模型未纳入评测；数据集仅反映2026年的生成器混合，未来需扩大样本、重复实验并加入更多统计分析。

---

## 423. Demonstrating TOFFEE: A Learned System for Synthesizing Data Agent Trajectories at Scale

**arXiv ID:** 2607.06233 | [PDF](https://arxiv.org/pdf/2607.06233v1)

**作者:** Ziting Wang `[一作]` (Nanyang Technological University), Gao Cong `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 TOFFEE 系统，能够在给定的数据环境下自动生成高质量的数据代理轨迹，支持后续的微调和上下文学习。

**💡 创新点**

创新点包括：①自动构建可解的任务池，②基于 MCTS 的轨迹搜索与错误修复机制，③学习成本模型实现每一步的动态 LLM 配置选择。

**🔧 技术方法**

技术方案涵盖 Monte Carlo Tree Search、在线奖励学习的学习成本模型（LCM）、LLM 推理与工具执行（SQL、Python）等。

**📊 数据集**

使用真实企业数据环境、现有基准（Spider、LiveSQLBench、SpreadsheetBench）构造任务池，评估基准为 KramaBench 与 DSBench。

**📈 对比分析**

在相同预算下与单次通行、best‑of‑N 等基线比较，TOFFEE 在轨迹质量上更高、成本更低；微调后的 TOFFEE‑27B 在 KramaBench 与 DSBench 上均超过 OpenAI o3。

**⚠️ 局限性**

局限性在于仍受 LLM 推理成本限制，且在极大规模多任务环境或缺少依赖信息的数据场景下生成任务和轨迹的效率与覆盖率可能受影响。

---

## 424. Sample complexity bounds for the Jensen-Shannon divergence

**arXiv ID:** 2607.06270 | [PDF](https://arxiv.org/pdf/2607.06270v1)

**作者:** Oren Richter `[一作]` (Weizmann Institute of Science), Elad Schneidman `[通讯]` (Weizmann Institute of Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文从信息论的角度，理论推导了在已知两分布间的 Jensen‑Shannon 距离（JSD）时，分别使用对数似然比（LLR）分类器和多数投票分类器时所需的样本量上界。

**💡 创新点**

创新点在于：①首次将 JSD 与样本复杂度直接关联，给出 LLR 的样本量随 1/D_JS 递增；②证明多数投票的样本量随 1/D_JS^2 递增；③提供完整的证明链条（Chernoff → Bhattacharyya → Hellinger → JSD），并给出常数级别的上界。

**🔧 技术方法**

主要技术手段包括：对数似然比检验、Chernoff 信息、Bhattacharyya 系数、Hellinger 距离、Jensen‑Shannon 损失的性质以及 Hoeffding 不等式，用于分析两类二元检测的误差概率。

**📊 数据集**

由于研究完全是理论推导，未使用任何实际数据集；所有结果均基于对两分布 P、Q 的通用假设。

**📈 对比分析**

比较方法：将最优 LLR 分类器与局部单样本分类器再做多数投票的情况做对比。性能方面：LLR 需要的样本量为 O(1/D_JS)，而多数投票需要的样本量为 O(1/D_JS²)，说明硬量化导致显著的样本量损失。

**⚠️ 局限性**

局限性：①只考虑均匀先验下的二元检测；②仅对独立同分布样本讨论；③缺乏对多类别或非独立情形的推广；④未给出实验验证，仅停留在理论上。

---

## 425. Lower Bounds for PIR with Preprocessing from Blackbox Cryptography

**arXiv ID:** 2607.06451 | [PDF](https://arxiv.org/pdf/2607.06451v1)

**作者:** Alexander Hoover `[一作]` (Stevens Institute of Technology), Kevin Yeo `[通讯]` (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究单服务器私有信息检索（PIR）在预处理模型下的极限，证明若使用任意黑盒加密（如随机预言机、虚拟黑盒模糊化等），在客户端存储s比特、批量查询k=Θ(s)时，在线查询的总计算量或通信量至少为Ω(n/s)，并进一步给出对应的通信下界与完全隐私下的特殊情形。

**💡 创新点**

创新点在于引入“双向PIR（dual PIR）”这一新概念，将离线预处理与在线查询的顺序对调，并借助子密钥预测（subkey‑prediction）理论给出信息论上的不可行性，从而得到无条件且适用于所有黑盒加密的最优计算与通信下界，克服了之前只能针对受限类别（如单轮、非编码或仅可见随机预言机）的结果。

**🔧 技术方法**

技术手段主要包括：1) 黑盒加密模型下的构造与证明；2) dual PIR 的构造与正确性转换；3) 与子密钥预测问题的紧密关联，通过信息熵和概率论推导得到误差下界；4) 对通信决定型服务器和完美隐私两类特殊 PIR 的简化分析；5) 通过归约证明 DEPIR（双重有效 PIR）在任何黑盒加密下不可实现。

**📊 数据集**

本文为理论性论文，无实验数据集，所有结果均在理论模型下给出。

**📈 对比分析**

与之前的工作相比，本论文的下界在无条件、适用范围更广（不需非编码或单轮约束），并与已知构造（如仅基于一向函数、或使用 FHE 的方案）匹配，证明了这些方案已达到最优；同时，进一步指出即使服务器使用线性公共密钥运算，也无法突破 Ω(n/s) 的计算或通信壁垒。

**⚠️ 局限性**

局限性包括：1) 结果要求批量查询数 k=Θ(s)，对仅支持少量查询的方案不适用；2) 只适用于黑盒加密；对基于服务器状态或非黑盒构造的 DEPIR 方案尚未完全覆盖；3) 证明中未给出对完美隐私之外的弱隐私或安全参数细化；4) 对具体实现效率（如常数因子）没有实验验证。

---

## 426. HoloCount: A Holistic Visual Counting Benchmark for MLLMs

**arXiv ID:** 2607.06420 | [PDF](https://arxiv.org/pdf/2607.06420v1)

**作者:** Jinhong Deng `[一作]` (Meituan), Guanglu Wan `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了HoloCount，一个基于三层层级分类（语义计数、分析计数、鲁棒性测试）的视觉计数基准，并对20余种顶尖多模态大型语言模型（MLLM）进行了系统评测。

**💡 创新点**

创新点在于将视觉计数拆解为细粒度子任务，构建包含1,481个视觉概念、2,480个高质量QA样本的层级化评测框架，揭示模型在逻辑推理、属性过滤和恶劣环境下的瓶颈。

**🔧 技术方法**

采用多模态LLM推理、自动生成与校验QA对、人工审校以及正则表达式抽取数值答案等技术。

**📊 数据集**

使用了自建的HoloCount数据集（从LVIS、Object365、CA44等公开数据集采样并人工校准）以及对比的公开计数基准如CountBench、CountQA等。

**📈 对比分析**

通过零样本推理比较了20+模型（开源如Qwen、InternVL、Gemma，闭源如Gemini、Claude、GPT）在三大维度上的准确率，发现大模型在语义计数表现优异，但在分析计数和高密度/鲁棒性子集显著下降；闭源模型与大规模开源模型差距趋近，且“思考模式”可提升计数准确率。

**⚠️ 局限性**

主要局限在于：1）仍缺乏对高密度和遮挡场景的稳健计数能力；2）对逻辑推理与集合运算的支持不足；3）在无目标或语言先验冲突场景下模型易产生错误；4）评测仍以人工标注为主，扩展到更大多样化数据集的可行性待验证。

---

## 427. Temporal Modeling of Optically Variable Devices in Identity Documents

**arXiv ID:** 2607.06408 | [PDF](https://arxiv.org/pdf/2607.06408v1)

**作者:** Glen Pouliquen `[一作]` (IDnow Research Center), Ahmad Montaser Awal `[通讯]` (IDnow Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在远程KYC场景下使用手机短视频对身份文件中的透明光学可变设备（OVD）进行动态验证；

**💡 创新点**

创新点在于提出两种完全自监督、无攻击样本训练的模型：基于时间段的判别式Span分类器和基于Masked Sequence Modeling的生成式重构模型，利用OVD的时间动态信息实现攻击检测；

**🔧 技术方法**

主要技术包括VideoMAE编码器、WSL预训练的帧级投影器、背景减法与时间腐败伪标签、光学噪声与颜色抖动增强，以及Masked Sequence Modeling的重构损失；

**📊 数据集**

实验使用公开的MIDV‑Holo与其扩展数据集MIDV‑DynAttack，分别包含合法、静态、交换与动态攻击样本；

**📈 对比分析**

与现有SOTA方法比较，两个自监督模型在合法样本训练下即可达到93–94%的AUC，尤其在交换与动态攻击上显著优于无攻击训练基线（提升约20–40%），与完全监督方法相近；

**⚠️ 局限性**

局限性包括对帧级表示质量高度依赖，导致对纯静态攻击的识别效果略逊；此外模型目前仅在单一OVD类型上训练，缺乏多样化数据支持，影响跨设备与跨国身份证的泛化能力。

---

## 428. From Foundation to Application: Improving VLA Models in Practice

**arXiv ID:** 2607.06403 | [PDF](https://arxiv.org/pdf/2607.06403v1)

**作者:** Wei Wu `[一作]`, Kecheng Zheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了 LingBot-VLA-2.0 Vision‑Language‑Action 模型，通过大规模多样化预训练数据、全身动作空间扩展以及双查询蒸馏提升跨任务、跨平台泛化能力，并在 GM‑100 与长时序移动操纵任务中进行验证。

**💡 创新点**

创新点包括：① 60,000 小时的预训练数据集（50k 小时机器人轨迹 + 10k 小时人类视角视频）；② 扩展至头、腰、移动底座及灵巧手的全身动作空间；③ 引入双查询（当前 + 未来）蒸馏，结合深度教师（LingBot‑Depth）与视频教师（DINO‑Video）实现几何与因果时间感知；④ 使用稀疏 MoE 结构与无损失平衡策略高效扩容。

**🔧 技术方法**

技术手段：Mixture‑of‑Experts Transformer、双查询蒸馏、相对动作表示、视频‑语言预处理管线、Qwen3.6‑27B 自动生成字幕、LingBot‑Depth 与 DINO‑Video 作为教师模型。

**📊 数据集**

使用的数据集：约 60,000 小时预训练数据（20 种机器人配置的 50k 小时轨迹 + 10k 小时 egocentric 人类视频）；GM‑100 基准（9 个双臂任务）和两大移动操纵任务（对象分类与烹饪清洁）用于评估。

**📈 对比分析**

通过与 LingBot‑VLA‑1.0 及 π_0.5 在 GM‑100 9 任务以及两项移动任务的 generalist 训练对比，LingBot‑VLA‑2.0 在进度与成功率均显著提升（例如 Agilex Cobot Magic 进度 66.2%/成功率 34.4%，长时序任务进度 77.1%/成功率 60% 等），表明整体性能优于前代模型。

**⚠️ 局限性**

局限性：某些任务仍存在进度与成功率差距较大，尤其在精准抓取或末端操作上表现不足；跨平台差异导致不同机器人在同一任务上的表现不均；对初始位置偏移和未见物体的鲁棒性仍有限。

---

## 429. InsideSSL: Understanding Self-Supervised Speech Representations using a Model-Centric Perspective

**arXiv ID:** 2607.06392 | [PDF](https://arxiv.org/pdf/2607.06392v1)

**作者:** Samir Sadok `[一作]` (Inria), Xavier Alameda-Pineda `[通讯]` (Inria)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一个基于模型的、任务无关的框架，对自监督语音模型的每一层进行三种内在属性（压缩熵、几何曲率、鲁棒性）的定量分析，并进一步引入跨层生成兼容矩阵（GCM）评估不同层之间的功能可迁移性；同时将这些内在指标与线性探测任务（音素分类、音高回归、说话人识别）关联，揭示不同层对不同任务信息的分布。

**💡 创新点**

创新点包括：①在模型层面统一的压缩‑几何‑鲁棒性分析；②提出跨层生成兼容矩阵，用生成解码器量化层间功能相似性；③将内在层级特征与下游任务性能进行关联，阐明低级任务依赖高熵高曲率，而音素识别需要深层压缩和线性化。

**🔧 技术方法**

使用的信息瓶颈理论、矩阵熵（von Neumann entropy）、平均曲率度量、InfoNCE 互信息近似、连续流匹配（CFM）训练的扩散 Transformer 解码器、HiFi‑GAN 合成、线性探测器以及 Pearson 相关分析。

**📊 数据集**

主要使用 LibriSpeech 语料库（train‑clean‑100、train‑clean‑960、test‑clean）作为评估数据，解码器训练亦基于 train‑clean‑100；对不同规模（base、plus、large）和不同预训练目标（对比、掩码预测、去噪预测、连续回归）的模型（Wav2Vec2、HuBERT、WavLM、Data2Vec、UniSpeech）进行实验。

**📈 对比分析**

通过比较不同模型、不同层的熵、曲率、InfoNCE，以及 GCM 评估指标（SpeechBERTScore、Resemblyzer 相似度、STOI、L1 误差），并用线性探测评估音素、音高、说话人性能，发现：WavLM 及 HuBERT 在深层保持高信息压缩与鲁棒性；Wav2Vec2 在最后层出现熵崩溃和 InfoNCE 跳升；中间层（≈第 7–10 层）在 GCM 和音素探测中表现最佳；低层在音高和说话人探测中表现突出。总体上，框架能够清晰展示各模型内部层级动态与下游任务的关联。

**⚠️ 局限性**

局限性包括：①缺乏对压缩崩溃机制的因果解释；②GCM 仅基于生成解码器，可能无法捕捉所有功能映射；③实验集中在 LibriSpeech 上，缺乏跨语言或跨任务的验证；④未考虑模型训练阶段的动态变化（仅静态层级分析）。

---

## 430. Groebner.jl: Fast Gröbner Tracing in Julia

**arXiv ID:** 2607.06372 | [PDF](https://arxiv.org/pdf/2607.06372v1)

**作者:** Alexander Demin `[一作]` `[通讯]` (École Polytechnique), Alexander Demin (École Polytechnique)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于F4算法的Julia Gröbner基计算库Groebner.jl，并通过公开的learn/apply接口实现了Traverso追踪加速；

**💡 创新点**

创新点在于将追踪机制从内部优化暴露为可复用的公共接口，并结合SIMD友好系数类型和批量处理实现了显著加速；

**🔧 技术方法**

使用了F4算法、Traverso追踪、SIMD向量化、乘积环(多模)运算、Julia的LLVM编译器以及批量应用技术；

**📊 数据集**

使用了多种基准多项式系统（Chandra、Katsura、Cyclic、Eco、SEIR等）以及结构辨识模型（EAIHRD、LinComp2、Pharm）进行实验；

**📈 对比分析**

通过与传统独立基计算、随机线性代数实现以及批量追踪三种方法对比，实验显示在多模/多参数化情形下追踪+批量能提升1.8–6.6倍的速度；

**⚠️ 局限性**

局限性包括追踪学习阶段成本高、对大批量时内存占用显著上升、在某些系统（如Yang1）中收益有限，以及ARM架构下SIMD优化效果不如预期。

---

## 431. The Impact of Security and Privacy Controls on Users' Emotional Engagement with Generative AI Chatbots

**arXiv ID:** 2607.06371 | [PDF](https://arxiv.org/pdf/2607.06371v1)

**作者:** Jabari Kwesi `[一作]` (Duke University), Pardis Emami-Naeini `[通讯]` (Duke University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对87款生成式AI聊天机器人进行系统审计，并在354名美国受访者中进行情境实验，评估九种安全与隐私(S&P)控制对情感支持使用意愿、感知保护及感知效果的影响。

**💡 创新点**

首次将安全与隐私控制的三重缺口（理解缺口、保证缺口、情感紧迫缺口）与情感支持情境相结合，揭示删除功能主导但信任脆弱的现实，并给出三条针对设计与监管的可操作建议。

**🔧 技术方法**

采用混合方法设计，包含系统审计、分层CLMM统计建模和主题分析，利用Prolific招募平台收集问卷与开放式回答。

**📊 数据集**

基于美国英语使用者的问卷样本（N=354）以及对2025年8-9月App Store上87款ChatGPT/ Gemini等聊天机器人的功能审计数据。

**📈 对比分析**

通过对九种S&P控制在情感支持三种情境（焦虑、抑郁、人际紧张）下的影响进行配对比较，结果显示删除控制在三项指标上表现最佳，而本地化处理和模型训练Opt-out因理解缺口导致影响最差，且整体效应中介值均为中小效应。

**⚠️ 局限性**

样本仅为美国青壮年英语使用者，缺乏跨文化与法规差异；使用假设情境而非真实对话，可能低估情绪冲击与控制决策的真实动机；以及控制定义依赖平台表述，易随技术演进而变化。

---

## 432. A VLM-Enhanced Framework for Comprehensive Traffic Sign Condition Assessment Integrating Daytime Visual Performance and Nighttime Retroreflectivity Evaluation

**arXiv ID:** 2607.06478 | [PDF](https://arxiv.org/pdf/2607.06478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. Training-Free Acceleration for Vision-Language-Action Models with Action Caching and Refinement

**arXiv ID:** 2607.06370 | [PDF](https://arxiv.org/pdf/2607.06370v1)

**作者:** Ryuji Oi `[一作]` (Institute of Science Tokyo), Daichi Fujiki `[通讯]` (Institute of Science Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ActionCache，一种外部缓存机制，可在预训练的流匹配式视觉-语言-动作（VLA）模型中，利用过去生成的中间动作片段进行热启动，从而显著减少动作头的迭代推理次数并提升实时性能。

**💡 创新点**

创新点包括：① 训练无关、插件式的缓存设计，能直接附加到任意预训练流匹配式 VLA；② 通过稀疏三元随机投影将高维 VLM 输出嵌入压缩为紧凑的多模态键，实现高效检索；③ 引入基于相似度阈值的安全回退机制，保证在缓存失效时仍保持模型原有鲁棒性；④ 通过跨时间、跨任务、跨情景的动作重用，拓展了传统局部时间连续热启动的范围。

**🔧 技术方法**

技术手段包括：流匹配式动作头的多步迭代生成、稀疏三元随机投影生成缓存键、余弦相似度检索、LRU/LFU 等缓存替换策略、以及基于相似度阈值的 hit/miss 决策。

**📊 数据集**

数据集：VLABench（10 种原始任务）、LIBERO 进行仿真评估；真实实验使用 SO‑101 机器人进行 pick‑and‑place 任务，配合 LeRobot 框架进行微调和评估。

**📈 对比分析**

与基线（全步推理）、NFE 减少基线、EfficientVLA 等插件加速方法对比。实验表明：在 π₀.₅ 上 ActionCache 在 N_hit=0 时可实现 11.75× 的动作头加速，成功率仍保持 32.9%（与全步 34.0% 相近）；在 GR00T‑N1.6 上加速 34.43×，成功率 22.3%（比全步 24.1% 略低）。在真实世界 pick‑and‑place 任务中，ActionCache 将推理时间从 106.83 ms 降至 61.76 ms（≈1.73×），成功率保持 70% 以上。

**⚠️ 局限性**

限制包括：① 缓存性能高度依赖于相似度阈值和缓存大小，需手动调参；② 在高度动态或完全新颖的情景（如抓取阶段）时 hit 率低，无法充分加速；③ 需要额外的存储空间，且在极小缓存下替换策略对性能影响显著；④ 仅针对流匹配式 VLA，无法直接推广到其他类型的动作头。

---

## 434. TopoBrick: Agentic Topology Sampling of Exogenous Variables for Zero-Shot Building IoT Forecasting

**arXiv ID:** 2607.06349 | [PDF](https://arxiv.org/pdf/2607.06349v1)

**作者:** Xiachong Lin `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出TopoBrick框架，实现无训练的零样本建筑IoT传感器时序预测。

**💡 创新点**

创新点在于利用建筑知识图构建结构骨架并通过智能拓扑采样器选择物理相关外生变量，同时按部署可用性划分过去已知与未来已知变量。

**🔧 技术方法**

使用知识图推理、LLM验证、冻结的时序基础模型（如Chronos-2）以及气象预测模块。

**📊 数据集**

在LBNL59、BTS-B和BTS-C三栋真实建筑的数据集上进行实验。

**📈 对比分析**

与基准的比较包括零样本基线（Chronos、Moirai、TimesFM）、完整训练模型（PatchTST、FITS、iTransformer）和随机/同类/k-hop采样；TopoBrick在大多数场景下优于零样本基线，且在LBNL59与BTS-B上与全训练模型持平或更优，BTS-C保持竞争力。

**⚠️ 局限性**

局限性在于对某些受占用或闭环控制驱动的电气/流量信号的解释不足，且对天气质量依赖较高，拓扑采样效果受建筑知识图完整性和语义一致性限制。

---

## 435. A Physics-Informed Neural Network Framework for Elastodynamic Wave Propagation in Bimaterial Systems

**arXiv ID:** 2607.06479 | [PDF](https://arxiv.org/pdf/2607.06479v1)

**作者:** Sonal Ankush Chibire `[一作]` (Northern Illinois University), Bo Zhang `[通讯]` (Northern Illinois University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在钢铝双材料Split Hopkinson Pressure Bar试样中，利用物理信息神经网络（PINN）对轴对称线性弹性波传播进行建模与预测。

**💡 创新点**

创新点在于将完整的Navier–Lamé方程、初始/边界/接口条件嵌入PINN损失函数，并通过高精度ANSYS显式动力学模拟数据作为软约束，实现了对波传输与反射的高精度连续预测。

**🔧 技术方法**

使用的技术包括物理信息神经网络、自动微分、Adam优化器以及ANSYS Workbench Explicit Dynamics仿真。

**📊 数据集**

使用的数据集为ANSYS Explicit Dynamics得到的钢铝试样在Split Hopkinson Pressure Bar实验下的位移、应力、应变时间历史；同时采用网格细化和不同材料组合进行验证。

**📈 对比分析**

通过与ANSYS显式动力学模拟结果比较，PINN在位移、面平均响应、应力应变方面与高精度有限元结果高度一致，能够在未见时间区间准确预测；相较于单纯数据驱动方法，显著减少了计算成本。

**⚠️ 局限性**

主要局限在于对后峰期波反射的预测误差较大，特别是钢材料的径向响应，以及对高度非线性材料行为的适应性不足。

---

## 436. Hilti-Trimble-Oxford Dataset: 360 Visual-Inertial Benchmark with Floor Plan Priors for SLAM and Localization

**arXiv ID:** 2607.06464 | [PDF](https://arxiv.org/pdf/2607.06464v1)

**作者:** Samuele Centanni `[一作]` (Hilti AG), Maurice Fallon `[通讯]` (University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了一个在真实施工现场收集的 360°视觉‑惯性数据集及其对应的 2D 平面图，并通过 Hilti × Trimble Challenge 2026 评测 SLAM 与定位算法的性能。

**💡 创新点**

创新点在于：① 长达 8 个月、七层建筑及地下层的多序列数据，捕捉施工过程中的结构演化；② 采用 LiDAR‑IMU 方案生成高精度地面真值，并公开 2D 平面图作为定位参考；③ 设计多难度因素（低照度、动态初始化、剧烈运动等）并举办全球挑战，吸引大量参赛团队。

**🔧 技术方法**

使用的技术包括：Insta360 ONE RS 360° 摄像机 + 6 轴 IMU；Kalibr 进行双镜头内外参数与滚动快门校准；LiDAR‑IMU SLAM（MC2SLAM+非线性优化）产生地面真值；视觉‑惯性 SLAM 系统如 OKVIS2‑X、OpenVINS、ORB‑SLAM3、√(VINS)；定位方法如 Z‑FLoc、BEV‑墙点对齐、语义分割。

**📊 数据集**

使用的数据集为 Hilti‑Trimble‑Oxford Dataset，包含 30 个序列（平均 180 s），覆盖 7 层及 2 层地下停车场，提供 ROS2 bag、图像、IMU、2D 平面图（PNG/DXF）。

**📈 对比分析**

评估通过官方 benchmark，SLAM 轨迹先做 Kabsch 对齐后计算 3D 误差，定位则直接在平面图框架下评估 2D 误差；SLAM 最高得分 2410，平均误差约 9 cm；定位 最高得分 2196，平均误差约 24 cm，表明结合语义分割的定位方法表现最优。

**⚠️ 局限性**

主要局限：平面图与现场随施工进度变化不同步，导致定位与地面真值之间存在数厘米误差；低照度、动态初始化和剧烈运动等场景仍具挑战；数据集聚焦于低成本 360° 摄像机，可能不适用于所有传感器组合。

---

## 437. PIPBench: A Profile-Inclusive Framework for Personalized Image Generation Evaluation

**arXiv ID:** 2607.06440 | [PDF](https://arxiv.org/pdf/2607.06440v1)

**作者:** Yuhang Wu `[一作]` (Tsinghua University), Miao Liu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了PIPBench个人化图像生成基准，并通过用户画像与历史偏好图像实现对模型的评估与实验。

**💡 创新点**

首次引入心理学与人口统计维度构造用户画像，搭建基于代理的合成数据生成流程，提出profile‑inclusive benchmark并用Elo评估。

**🔧 技术方法**

利用LLM链式思维生成基于画像的提示，CLIP/DINO/LPIPS等自动指标，LLM代理评判，DreamBooth、Qwen‑Image‑Edit、VLM条件融合等多种个性化生成方法。

**📊 数据集**

使用真实用户调查收集的134份画像及图片（最终76名用户的偏好集）与175个合成代理（共1,876张图片），形成1,369个测试案例。

**📈 对比分析**

通过CLS‑T、CLS‑R、DIS‑R、LPIPS‑R等自动指标和LLM persona‑aware Elo评分进行比较，GPT‑5在自动与Elo两方面表现最佳；多参考联合条件往往适得其反，VLM融合效果更佳。

**⚠️ 局限性**

现有方法难以同时处理多张参考图并保持指令与偏好平衡；profile‑free方案缺乏多维多样性，真实用户偏好易于匹配而合成代理仍难以精确捕捉；整体缺乏端到端视觉偏好学习框架。

---

## 438. Finding H. pylori in the Fine Print: Evidence-Linked Multi-Agent Case Finding from Gastric Biopsy Reports

**arXiv ID:** 2607.06435 | [PDF](https://arxiv.org/pdf/2607.06435v1)

**作者:** Yufan Wang `[一作]` (Nimblemind.ai), Li Yan Khor `[通讯]` (Singapore General Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估了 Nimblemind 多代理系统（nMAS）在从胃活检病理报告中提取四个与 H. pylori 相关的二元特征，并将其与 UMA‑MiniMax M2.5 进行对比。

**💡 创新点**

创新点在于提出一种可配置、字段名驱动、证据链接的多代理工作流，能够在抽取后直接返回与原始文本对应的支持句子，既保留了可审计性，又实现了报告级别的结构化输出。

**🔧 技术方法**

采用多级抽取框架：层 1 采用 NER，层 2 采用小型 LLM（Qwen2.5‑7B‑Instruct），层 3 采用大型 LLM（DeepSeek‑V4‑Flash）；使用字段库、提示工程、结果合并与验证模块，实现源文本与抽取结果的双向绑定。

**📊 数据集**

使用了 54 篇去标识化的胃活检病理报告（来自新加坡一般医院 SGH），报告包含编码字段、诊断文本、微观描述等多种信息。

**📈 对比分析**

通过 216 个特征‑案例的精确匹配评估，nMAS 与 UMA‑MiniMax M2.5 在整体准确率上均为 98.61%；四个字段的准确率分别为 100%、100%、98.15% 与 96.30%；两者在分类性能上几乎相同，差异主要体现在工作流集成与可追溯性。

**⚠️ 局限性**

局限性包括样本量仅 54 份、单机构单语境、仅四个二元字段、未评估实时延迟、未测量临床审核时间、未评估多语种、扫描文档等情况，需要更大规模、多机构验证以及临床应用的实证研究。

---

## 439. TILDE: TILt-based Distributional Erasure for Concept Unlearning

**arXiv ID:** 2607.06432 | [PDF](https://arxiv.org/pdf/2607.06432v1)

**作者:** Naveen George `[一作]` (Indian Institute of Technology Hyderabad), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于能量倾斜的文本到图像扩散模型概念遗忘方法TILDE；

**💡 创新点**

将概念遗忘明确为分布对齐问题，给出最小偏差的锚点自由目标，并用阈值化的CLIP能量驱动；

**🔧 技术方法**

利用残差∇-GFlowNet在扩散潜在空间中学习能量倾斜的分布，并采用LoRA微调和CLIP阈值能量；

**📊 数据集**

在Stable Diffusion v1.5上，对约60个不同对象、角色、艺术风格和裸露概念进行实验；

**📈 对比分析**

与ESD、UCE、MACE、DUO、EraseFlow等现有方法对比，TILDE在实现近乎完整遗忘的同时保持更高的相关/一般保留准确率、FID和FADE，位于Pareto前沿；

**⚠️ 局限性**

局限在于能量阈值和CLIP相似度对概念的检测效果有限，复杂或组合概念可能需要更强的能量模型

---

## 440. VaseMuseum: Digital Intelligent Museum for Ancient Greek Pottery

**arXiv ID:** 2607.06374 | [PDF](https://arxiv.org/pdf/2607.06374v1)

**作者:** Jiazi Wang `[一作]` (Beijing Jiaotong University), Hao Tang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VaseMuseum，一个集成了3D数字化与视觉-语言模型的交互式数字古希腊陶器博物馆框架；

**💡 创新点**

创新点在于在检索环节加入了源级与答复级的可靠性控制，并提供训练无关的GRPO式多轨迹选择，以实现证据检索、可信引用和不确定性自适应回答；

**🔧 技术方法**

使用了大规模视觉-语言模型（如Qwen3-VL-8B）、DeepResearch式工具循环、源控制与答复控制模块，以及无训练的GRPO式选择机制；

**📊 数据集**

使用VaseVQA-3D数据集（3000+古希腊陶器图像）并扩展至LIMC等权威数据库链接；

**📈 对比分析**

与直接VLM、无可靠性控制的搜索VLM以及无GRPO的VaseAgent对比，VaseMuseum在知识密集型查询中显著降低了hallucination率、提升了引用有效性和中立性，整体性能更平衡；

**⚠️ 局限性**

局限包括对外部网页/博物馆资源的可用性与一致性依赖、仅针对古希腊陶器的专业化验证、以及可靠性指标的主观性和需进一步专家评估。

---

## 441. Automated Compliance Mapping in Cloud Security with Domain-Adapted Sentence Transformers

**arXiv ID:** 2607.06364 | [PDF](https://arxiv.org/pdf/2607.06364v1)

**作者:** John Bianchi `[一作]` (IMT School for Advanced Studies Lucca), Marinella Petrocchi `[通讯]` (Institute for Informatics and Telematics (IIT-CNR))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并使用一个基于七个欧洲云安全标准的语义对照语料库，通过域自适应微调Sentence Transformer模型，实现云安全控制与技术指标（以及不同标准之间的控制）的自动映射。

**💡 创新点**

①将多个异构标准通过Cisco CCF中心化映射，生成3,499对原始语义对；②采用回译与LLM改写两种数据增强策略，扩展至13,996条样本；③在控制–控制和控制–指标两任务上统一使用单一模型，提升了模型的泛化与可扩展性。

**🔧 技术方法**

Sentence Transformer（MPNet、DistilRoBERTa、MiniLM等）微调；多负样本排名损失（MNRL）；回译、LLM（Phi‑4、Llama‑3）改写；使用nDCG@10评估检索质量。

**📊 数据集**

基于Cisco CCF、BSI C5、ENS、SecNumCloud、EUCS-2020等七个标准构建的训练对照集；测试集包含MEDINA项目的EUCS-Legacy与技术指标，以及EUCS-Legacy与BSI C5的跨标准控制对。

**📈 对比分析**

在零射（zero‑shot）基础上进行微调后，所有模型在两项测试中均超越基线；控制–指标任务提升最高0.228 nDCG@10；跨标准控制任务最高0.870 nDCG@10（全查询）/0.965（非零查询）。数据增强对跨标准任务尤为有效，回译策略表现最佳。

**⚠️ 局限性**

①数据量仍有限，难以覆盖所有云安全场景；②回译与LLM改写可能引入语义漂移，导致部分增强样本质量不一；③评估仅基于nDCG@10，未涉及更细粒度的误匹配分析；④模型对新出现的标准或指标的适应性尚未验证。

---

## 442. Generalized Synthetic Image Detection with Enhanced RGB-Noise Representation Learning

**arXiv ID:** 2607.06354 | [PDF](https://arxiv.org/pdf/2607.06354v1)

**作者:** Zhen Li `[一作]` (Communication University of China), Shaowei Weng `[通讯]` (Fujian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种双分支网络RNSIDNet，用于检测人工合成图像，能够在多种生成模型上实现泛化。

**💡 创新点**

创新点包括：1）使用FiLM模块动态将RGB语义特征调制噪声分支，实现多模态特征的高效融合；2）提出Hard Sample-aware Contrastive Learning (HSCL) 对难辨样本进行加权，从而在特征空间中拉大正负间距；3）构建了像素对齐的大规模多源合成图像数据集AMSID。

**🔧 技术方法**

核心技术包括：CLIP ViT-L/14作为RGB特征提取器、Bayar约束卷积提取高频噪声、Balanced Attention Module (BAM) 对CLIP特征加权、FiLM实现特征调制、HSCL对比学习、以及数据增强和多源训练策略。

**📊 数据集**

使用了AMSID（约235k图像）进行训练，并在8个公开基准（UniversalFakeDetect、AIGCDetectionBenchmark、Synthbuster、GenImage、DIF、DDA-COCO、Chameleon、WildRF）上进行评测。

**📈 对比分析**

与8类主流基准方法相比，RNSIDNet在ACC与AUC上均达成SOTA水平，平均ACC≈83.8%、AUC≈92.7%，在跨模型、抗压缩、模糊等真实场景下表现最稳健。

**⚠️ 局限性**

局限性包括：1）仅针对单帧图像，未扩展到视频或连续帧检测；2）对极度隐蔽或后处理严重的合成图像仍存在误判；3）模型仍依赖于多源合成数据的多样性，若出现新的生成器或新的后处理手段可能需重新训练。

---

## 443. Andha-Dhun: A First Look at Audio Descriptions in Hindi

**arXiv ID:** 2607.06457 | [PDF](https://arxiv.org/pdf/2607.06457v1)

**作者:** Ritabrata Chakraborty `[一作]` (IIIT Hyderabad), Makarand Tapaswi `[通讯]` (IIIT Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首次系统研究了印地语音频描述（AD），并提出了一个包含 8 部电影 5,870 条 AD 的 Andha‑Dhun 数据集；

**💡 创新点**

创新点在于：①构建首个印地语 AD 语料库；②探讨直接生成与翻译两种生成策略；③从文化适应性角度评估 AD 的可访问性。

**🔧 技术方法**

技术上采用了多模态大型语言模型（如 Qwen‑2‑VL‑7B‑Instruct、Llama‑3‑8B、Nemotron‑4‑Mini‑Hindi‑4B‑Instruct、Gemini‑3.1‑Pro）进行 dense 描述生成与翻译，配合 LLM‑AD‑Eval 进行自动评估，并使用 perplexity 衡量语言多样性。

**📊 数据集**

使用 Andha‑Dhun 数据集，该数据集包含 5,870 条印地语 AD 与相应的电影视频片段；同时还对 4 部双语电影（同时提供英印 AD）进行人工对齐与评价。

**📈 对比分析**

对比方法包括：①直接从 dense 描述生成印地语 AD；②将英文 AD 翻译成印地语；以及人类作者的印地语 AD；评估指标为 perplexity 与 LLM‑AD‑Eval。结果显示，Nemotron 直接生成的印地语 AD 在 perplexity 上优于翻译方法，LLM‑AD‑Eval 评分也最高，但所有自动方法仍远低于人工 AD，且翻译方法在文化特定项的处理上表现最差。

**⚠️ 局限性**

局限性包括：①数据规模有限（仅 8 部电影）；②缺乏专门针对印地语的训练数据，导致生成质量与多样性不足；③评估指标未完全覆盖用户体验与文化适应性；④机器翻译缺乏 Skopos‑aware 适配，导致文化细节失真。

---

## 444. From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b

**arXiv ID:** 2607.06452 | [PDF](https://arxiv.org/pdf/2607.06452v1)

**作者:** Taeyun Roh `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套根据问题类型（是/否、事实、列表）定制的大语言模型推理框架，在 BioASQ 14b 任务中生成精确答案。

**💡 创新点**

创新点在于将问题类型感知的推理策略、提示工程、集成投票和多代理验证统一起来，显著提升答案的稳健性和证据根植性。

**🔧 技术方法**

使用了 GPT‑5、GPT‑4o、Gemini 3、Claude 等多种大语言模型，结合链式思考、检索式上下文示例、片段洗牌等提示工程，并通过多模型集成和基于代理的证据分析与验证实现推理。

**📊 数据集**

实验调优基于 BioASQ 13b 开发集，正式评测使用 BioASQ 14b Phase B 公开数据。

**📈 对比分析**

与多种基线（单模型、检索增强、前置验证）对比，系统在 BioASQ 14b 批次 4 的事实子任务中获得第一名；整体宏观 F1≈0.91、MRR≈0.46、列表任务 F‑measure≈0.44。

**⚠️ 局限性**

局限性包括依赖专有 LLM 导致复现困难、只关注精确答案未覆盖理想答案、提示细微变化可能影响输出格式、以及未统一单一模型的整体性能。

---

## 445. SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models

**arXiv ID:** 2607.06442 | [PDF](https://arxiv.org/pdf/2607.06442v1)

**作者:** Changti Wu `[一作]` (East China Normal University), Kai Chen `[通讯]` (Zhongguancun Institute of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结构感知的数据选择方法 SIEVE，用于 Vision‑Language‑Action 模型的模仿学习，通过先发现可重用的原语与转移，再分配选择预算，最终在每个结构模式中挑选代表性轨迹。

**💡 创新点**

创新点在于：
1) 将演示视为可重用原语与转移的组合，利用结构曝光目标在全局预算内平衡原语/转移的曝光度；
2) 在每个组合模式中使用 Medoid 轨迹进行“学习友好”选择，保证行为克隆的稳定性；
3) 将 MDL 原则与贪婪分配相结合，最大化可压缩结构的覆盖率。

**🔧 技术方法**

技术手段包括：
- 轨迹分段（基于抓取/释放边界）
- 视频编码器 V-JEPA2 + PCA 获得段向量
- MiniBatch K‑Means 进行原语聚类
- 结构曝光分配（log‑utility + 贪婪迭代）
- Medoid 轨迹挑选（cosine 相似度）
- 视觉语言动作模型 Qwen3‑VL‑4B‑GR00T/OFT 以及行为克隆训练。

**📊 数据集**

实验使用了三个机器人演示数据集：Bridge‑V2（WidowX 真实演示），Fractal（Google 机器人真实演示），GR00T‑X‑Sim（仿真人形机器人演示）。

**📈 对比分析**

与 Full‑Training、Random、DemInf、SCIZOR 等基线比较。SIEVE 在 50%/70% 样本预算下，使用更少的训练步骤即可获得更高的平均成功率（Bridge‑V2 上约 56.3% vs 51.8%，Fractal 上 76.4% vs 75.0%，GR00T‑X‑Sim 上 54.8% vs 52.7%），并在多模型（Qwen3‑VL‑4B‑GR00T/OFT）和多数据集上保持领先。

**⚠️ 局限性**

局限性：
- 需要先验的抓取/释放分段规则，对非抓取型任务可能不适用；
- 依赖预训练的视频编码器和 PCA 降维，若编码器不匹配可能导致原语识别不准；
- 结构曝光分配采用贪婪近似，未证明最优性；
- 仅在行为克隆场景下评估，对其他 IL 算法的适用性未知；
- 在极度嘈杂或无明显可重用结构的任务中，方法效果可能不明显。

---

## 446. Precise Video-to-Audio Generation with Cross-Modal Alignment in Latent Space

**arXiv ID:** 2607.06405 | [PDF](https://arxiv.org/pdf/2607.06405v1)

**作者:** Thanh V. T. Tran `[一作]` (FPT Software AI Center), Van Nguyen `[通讯]` (FPT Software AI Center)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出端到端的 Flowley 框架，结合视频、文本和音频多模态信息，实现无多阶段训练的高质量视频到音频生成；

**💡 创新点**

核心创新包括：① Progressive Soft-masked Cross-Attention (PSCA)，在注意力机制中直接嵌入时序对齐，无需额外对齐模块；② SoundCap 语音感知字幕管线，利用 AV‑LLM 生成细粒度音频描述，提升文本条件的语义丰富度；③ 采用流匹配训练（Flow Matching）与单阶段 ODE 推理，减少计算开销。

**🔧 技术方法**

使用流匹配（FM）框架、CLIP、FLAN‑T5 文本编码、VAE 编码、BigVGAN 解码，结合多流块、单流块、跨模态注意力与 PSCA；训练中引入噪声调度（正切采样）、速度方向监督和 classifier‑free 指导。

**📊 数据集**

主要在 VGGSound（约20万条 10 秒视频）上训练与评估，亦在 MovieGen Audio Bench 进行零样本对比。

**📈 对比分析**

与七种 SOTA 方法（Frieren、FoleyCrafter、V2A‑Mapper 等）在 VGGSound 上对比，Flowley 在分布匹配（KAD 0.42）、音频质量（IS 18.25）、语义对齐（IB‑Score 29.32）均领跑；加入 SoundCap 后进一步提升至 30.07 并在零样本测试中超过同类大模型。

**⚠️ 局限性**

局限性：对齐指标 Align Acc 与人类主观评估存在偏差；模型对外部噪声和离散语音场景仍易受影响；在更大规模、多样化数据集上的泛化尚待验证。

---

## 447. A Definition and Roadmap for World Models

**arXiv ID:** 2607.06401 | [PDF](https://arxiv.org/pdf/2607.06401v1)

**作者:** Xinyuan Chen `[一作]` (Shanghai AI Laboratory), Ming Zhou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“世界模型”的科学定义与核心属性，阐述其作为有限资源下对物理世界状态转移过程的压缩建模，并系统梳理了从观察层、潜在层到3D/对象中心等多种架构与从渲染器、模拟器、规划器到WAM的功能层次的双维度分类。

**💡 创新点**

创新点在于将世界模型定位为信息论意义上的压缩机制，强调数据多样性是性能上限、提出“倒金字塔”数据处理流程、提出“链式想象”与“物理约束学习”三大技术路线，并将可解释性、规划与生成统一到同一模型框架。

**🔧 技术方法**

采用的技术包括：自监督/生成预训练（视频预测、掩码自编码、下一步预测、Diffusion/Transformer生成）、潜在空间动态建模（RSSM、JEPA、Latent‑Space MDP）、3D/对象中心表示（BEV、NeRF、Slot Attention）、跨模态融合（Omni‑Modal Transformer、WAM）、物理约束（软/硬约束、PINN、Lagrangian/Hamiltonian网络）以及多阶段仿真‑真实数据混合。

**📊 数据集**

主要使用数据来源为大规模互联网视频/图像（Youtube、公开视频集）、文本、音频、以及合成仿真数据（Omniverse、Isaac Sim）和少量专用机器人数据；通过自动过滤与注释将海量原始视频压缩成可用于训练的任务对齐数据。

**📈 对比分析**

对比方法主要包括传统视频生成器（Sora、Seedance）、潜在空间模型（Dreamer、PlaNet）、3D 结构模型（Marble、GaussianWorld）以及WAM/Omni‑Modal模型（DreamZero、WorldGPT）。实验表明：在长时序预测、动作规划与嵌入式控制上，具备物理约束和双向生成的模型相较于单纯生成器能显著提升物理一致性与离线/在线规划性能；但在视觉逼真度、计算效率与跨任务泛化上仍存在差距。

**⚠️ 局限性**

局限性：①数据多样性与分布外泛化受限；②长期推理中的累积误差与不确定性校准不足；③物理约束的硬/软平衡难以调优，易导致模型偏差；④仿真‑真实差距（sim‑real gap）仍未彻底解决；⑤可解释性与因果因果推理能力不足；⑥大规模预训练与推理成本高，实际部署受限。

---

## 448. FADRA: Frequency-Aware Diffusion with Residual Adaptation for Video Face Restoration

**arXiv ID:** 2607.06389 | [PDF](https://arxiv.org/pdf/2607.06389v1)

**作者:** Jin Jiang `[一作]` (United Arab Emirates University), Shengcai Liao `[通讯]` (United Arab Emirates University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于频率感知扩散框架的FADRA，利用轻量LoRA和低质量像素对齐融合以及重复残差适配头，实现高质量、时序一致的视频人脸恢复。

**💡 创新点**

创新点在于：1) 引入轻量LoRA适配与低质量像素对齐特征融合，保持预训练扩散模型的时序一致性；2) 设计重复残差适配头RRAH，使模型在每一步流匹配中再次利用降质线索；3) 提出频率感知损失，在频域加权监督高频细节。

**🔧 技术方法**

使用技术包括预训练文本到视频扩散模型Wan、LoRA、DiT、DCT频域加权、流匹配训练及低质量像素对齐特征融合。

**📊 数据集**

使用的数据集为合成的VFHQ+HDTF训练集，VFHQ与CelebV‑HQ测试集，以及真实电影序列进行验证。

**📈 对比分析**

与多种图像与视频人脸恢复方法（DiffBIR、SVFR、StableVSR等）对比，FADRA在PSNR、SSIM、LPIPS、IDD和FVD等指标上均优于对比方法，尤其FVD最低为38.97，帧率约0.866 FPS。

**⚠️ 局限性**

局限性包括：在极端大姿态变化、强运动模糊或复杂自然降质下仍可能出现细节失真；模型依赖大型扩散模型推理，推理速度相对较慢。

---

## 449. Learning to Throw Objects Safely in Multi-Obstacle Environments

**arXiv ID:** 2607.06388 | [PDF](https://arxiv.org/pdf/2607.06388v1)

**作者:** Mohammadreza Kasaei `[一作]` (University of Edinburgh), Hamidreza Kasaei `[通讯]` (University of Groningen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多障碍环境下通过潜在场状态表示和强化学习实现安全投掷，使用kinesthetic teaching初始化投掷内核，并在模拟与真实机器人上训练和评估投掷成功率。

**💡 创新点**

提出固定尺寸、可扩展的潜在场表示，利用CNN保持空间结构，并将kinesthetic teaching作为安全起点，系统性比较了潜在场与显式姿态表示的效果。

**🔧 技术方法**

使用强化学习算法（SAC、TD3、DDPG）与CNN编码的潜在场，Gazebo/ROS仿真，Stable-Baselines3实现，以及kinesthetic teaching进行安全初始化。

**📊 数据集**

主要使用模拟中的可投掷物体（milk box）以及三种未见物体（banana、coke can、sneaker），随机生成0-5个障碍物，未使用公开数据集。

**📈 对比分析**

通过在不同障碍数（0、1、3、5）下比较EPR和PFR的投掷成功率，SAC在潜在场下实现了90%以上成功率；在真实机器人上20次试验中，PFR成功率70-90%，优于EPR的60-70%。

**⚠️ 局限性**

受感知噪声、抓取延迟和姿态估计误差影响，投掷过程仍易出现碰撞、投掷误差和局部最优问题；缺乏对动态环境、多机器人协作和更大规模障碍的评估。

---

## 450. Prompt-Adapter Context Routing for Parameter-Efficient Multi-Shot Long Video Extrapolation

**arXiv ID:** 2607.06481 | [PDF](https://arxiv.org/pdf/2607.06481v1)

**作者:** Anna Córdoba `[一作]`, Jesús Olivera `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PACR-Video框架，用低秩时间适配器和递归提示银行在冻结的文本到视频扩散变压器上实现参数高效的多镜头长视频外推；

**💡 创新点**

创新点在于将递归提示银行与适配器门控相结合的提示‑适配器上下文路由方案，配合镜头角色提示、稀疏路由、身份对比和适配器组合调度，实现仅微调少量参数即可保持身份、场景、风格和因果连贯；

**🔧 技术方法**

使用技术包括冻结文本到视频扩散变压器、低秩时间适配器、学习的镜头角色提示、递归提示银行、门控适配器路由、身份对比损失、提示稀疏正则化以及适配器组合计划；

**📊 数据集**

评估数据集涵盖FlintstonesSV、Pororo‑SV、ActivityNet Captions、YouCook2、Shot2Story和MovieNet六个多镜头/长视频基准；

**📈 对比分析**

与VideoCrafter2、AnimateDiff、ReCA、ShotStream等多种基线对比，PACR‑Video在FVD、CLIPScore、DINO身份一致性、LPIPS、RAFT、BLIP‑2对齐、转场一致性及人类偏好等指标上均表现最佳，仅训练约3.8%的参数；

**⚠️ 局限性**

局限性包括仅适用于已给定镜头级提示，未显式建模提示不确定性，难以处理极长或高度不确定的剧情，以及在实时交互编辑场景下的鲁棒性仍待提升。

---

## 451. EgoPolice: A Benchmark for Egocentric Video Understanding in High-Stakes Police Body-Worn Camera Footage

**arXiv ID:** 2607.06468 | [PDF](https://arxiv.org/pdf/2607.06468v1)

**作者:** Max Gonzalez Saez-Diez `[一作]` (Princeton University), Olga Russakovsky `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个包含真实警察–平民交互的第一人称视角数据集，并在此数据集上进行细粒度动作标注，提供分类和多选问答两种基准任务。

**💡 创新点**

①提供高风险、实时摄像的BWC视频基准，覆盖极端运动、低光等真实场景；②系统化的双阶段标注流程与严格的注释规范；③演示模型在实际监控中的人机协同应用。

**🔧 技术方法**

采用FastFlowNet对运动统计、CLIP、VideoMAE V2、DINOv2等视觉特征提取器进行线性探测；使用Gemini 2.5、GPT‑4.1等VLM进行零射击问答；并自行研发注释工具与心理健康干预方案。

**📊 数据集**

EgoPolice数据集，约180小时、2,684段BWC视频，来源于COPA、Pasadena、Dallas等多部门公开视频。

**📈 对比分析**

通过线性探测在1s/10s/1min窗口下的F1与零射击问答的准确率评估，VideoMAE V2在动作分类上最好，Gemini 2.5 Pro在问答上最高但仅达76.9%（1min），整体表现仍显不足，说明任务难度高。

**⚠️ 局限性**

数据受限于公开视频、极端场景稀缺；标注仍有偏差；模型对光照、模糊、遮挡、少数类表现差；自动化工具无法完全替代人工，存在伦理与偏见风险。

---

## 452. Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders

**arXiv ID:** 2607.06445 | [PDF](https://arxiv.org/pdf/2607.06445v1)

**作者:** Yoav Baron `[一作]` (Tel Aviv University), Or Patashnik `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Analysis-by-Proxy 框架，利用轻量级 Q‑Former 直接从 VLM 单向前向传播的中间隐藏层中提取定位信息，并通过把预测框作为视觉标记输入 Diffusion Transformer，显著提升 VLM‑conditioned 图像编辑的定位精度。

**💡 创新点**

创新点在于：①将 VLM 视为单向编码器并对其内部信息流进行定位分析；②设计可解释的代理模型（Q‑Former）来定位并抽取最具定位信息的层与 token；③基于代理预测框对编辑管线进行动态条件调整，从而补偿传统使用最终层特征导致的定位失效。

**🔧 技术方法**

主要技术包括：Q‑Former 交叉注意力与自注意力的轻量化 Transformer；代理训练使用 GIoU+L1 损失；对 VLM 隐藏层进行注意力分布排序与层级路由；在 Diffusion Transformer 上通过 LoRA 微调学习对定位框的感知；评估使用 VQA（Gemini 2.5 Pro）和 LPIPS 背景相似度。

**📊 数据集**

使用的基准数据集为 200 张复杂多实体场景图像与对应的本地化编辑提示，结合 Qwen‑Image‑Edit、FLUX‑Kontext、FLUX.2、FIBO‑Edit 等公开编辑模型进行对比；代理训练采用 VLM 自然生成的边框作为监督标签。

**📈 对比分析**

在定位准确率和编辑质量上，本文方法在 Qwen‑Image‑Edit 上实现了 57.5%→最高 80% 左右的定位成功率，并在 VQA 分数与 LPIPS 背景指标上均优于现有 FLUX、FIBO 等模型，表明通过中间层定位信息可以显著提升编辑的精确度。

**⚠️ 局限性**

局限性包括：仅在单一 VLM‑conditioned 编辑管线（Qwen‑Image‑Edit）上验证，缺乏对多模型、多场景的泛化实验；代理方法对输入复杂度的依赖未完全解决；并且在真实应用中引入额外框标记可能带来视觉伪影与计算开销。

---

## 453. XRFormer: Multiscale Tokenization for XRF Representation Learning

**arXiv ID:** 2607.06424 | [PDF](https://arxiv.org/pdf/2607.06424v1)

**作者:** Sofiane Daimellah `[一作]` (Université Paris-Saclay), Clotilde Boust `[通讯]` (Centre de Recherche et de Restauration des Musées de France)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了XRFormer，一种利用多尺度卷积分词器与Transformer的模型，用于X射线荧光光谱的颜料识别和混合分析，并研究了自监督预训练策略。

**💡 创新点**

创新点在于设计专门针对XRF光谱的多尺度卷积分词器（融合局部峰值和多分辨率背景），以及提出Mask Spectral Modeling（MSM）和Physics-informed Peak Presence Prediction（PPP）两种自监督预训练任务。

**🔧 技术方法**

使用技术包括：Transformer encoder、卷积分词器、MSM与PPP自监督预训练、数据增强、线性混合生成等。

**📊 数据集**

使用的数据集为Pigments Checker STANDARD v.5（PCSv5）进行下游任务，Infraart 数据集用于预训练（扩增至约两百万样本）。

**📈 对比分析**

与ViT、SpectralFormer（含CAF与非CAF）和1D‑CNN基线进行对比；XRFormer在未预训练时已优于ViT，预训练后在绝对准确率和宏F1上达到76.78%，在颜料混合估计方面虽略低于SpectralFormer，但参数更少、token分辨率更低，表现相当。

**⚠️ 局限性**

限制主要在于缺乏大规模真实多样化XRF数据，实验基于单一参考光谱集，导致难以在更广泛场景下验证模型泛化能力。

---

## 454. Fast Rational Univariate Representation via Gaussian Elimination

**arXiv ID:** 2607.06397 | [PDF](https://arxiv.org/pdf/2607.06397v1)

**作者:** Alexander Demin `[一作]` (École polytechnique), Fabrice Rouillier `[通讯]` (Sorbonne Université)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一种基于高斯消元的密集线性代数算法，计算任意零维理想的可靠有理单变量表示；

**💡 创新点**

创新点在于将增广的FGLM步骤与增量高斯消元相结合，利用已得到的行阶梯形矩阵在后续步骤中重用，从而实现确定性、可验证的求解；

**🔧 技术方法**

采用密集线性代数、增量高斯消元、Julia语言实现，利用向量化SIMD加速，主要处理乘法矩阵、最小多项式和双变量排除基；

**📊 数据集**

使用一系列基准系统（Katsura、Chandra、Eco、Henrion、Noon、Reimer等），在有限域（机器素数）和有理系数情况下测试；

**📈 对比分析**

与msolve、Giac等主流软件对比，实验显示在大多数基准上本实现的速度比msolve快约1.5-2倍，且能保证参数化的正确性；

**⚠️ 局限性**

局限在于对非随机或“非泛型”商代数的处理仍需重算，且对极大维度或高度稀疏的乘法矩阵的优化仍有提升空间。

---

## 455. Faster Exponential-Time Approximate Counting via Bounded Self-Reductions

**arXiv ID:** 2607.06393 | [PDF](https://arxiv.org/pdf/2607.06393v1)

**作者:** Katie Clinch `[一作]` (University of Queensland), Qi Wang `[通讯]` (UNSW Sydney)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种通用的枚举‑采样框架，用于在指数时间内对无法多项式逼近的计数问题给出随机近似解。

**💡 创新点**

创新点在于引入无权自递归（unweighted self‑reducibility）与递归兼容上界，利用平方根的采样降低时间复杂度，并通过预处理将实例分解为若干难枚举核，进一步实现 √(∑b_i) 的速度提升。

**🔧 技术方法**

核心技术包括：无权自递归定义、递归兼容上界 b(x)、多态性枚举器、基于上界的采样器、Chernoff 近似分析、量子幅度放大实现量子加速，以及构造子实例的分解与合成。

**📊 数据集**

该工作为理论算法，未使用具体实验数据或数据集；所有结果均通过数学分析得到。

**📈 对比分析**

与已知最优指数基准相比，独立集计数从 1.2041ⁿ 缩小到 1.1869ⁿ，#2‑SAT 从 1.2377ⁿ 缩小到 1.2373ⁿ；最大团、最小分隔、三度图完美匹配的计数基数分别提升至 1.2009ⁿ、1.2721ⁿ、1.1611ⁿ；量子版本进一步降至 1.1279ⁿ、1.1740ⁿ、1.0776ⁿ 等。

**⚠️ 局限性**

局限性包括：仅适用于可自递归且存在递归兼容上界的计数问题；对极大计数基数的改进有限；分解成核的预处理仍可能是瓶颈；量子加速需额外的树结构访问；实际实现复杂度高。

---

## 456. Towards Real-World Applications with an Autonomous Powered Wheelchair

**arXiv ID:** 2607.06383 | [PDF](https://arxiv.org/pdf/2607.06383v1)

**作者:** Simone Arreghini `[一作]` (Dalle Molle Institute for Artificial Intelligence (IDSIA)), Antonio Paolillo `[通讯]` (Dalle Molle Institute for Artificial Intelligence (IDSIA))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在商用自平衡电动轮椅 Genny Zero 上集成 RGB‑D 摄像头与 LiDAR，构建基于 ROS2 的自主控制架构，并实现两种实用功能：手势触发的人跟随和远程召唤。

**💡 创新点**

首次将自平衡轮椅与完整的机器人感知、导航与人机交互模块结合，并通过手势触发实现即时高层行为；提供了面向真实环境的协助移动原型。

**🔧 技术方法**

使用的技术包括：ROS2 + Nav2（SLAM、AMCL、A* 规划、Pure Pursuit 控制）、自定义 Genny Zero 驱动、RGB‑D 人体追踪与手势识别、LiDAR 生成二维虚拟激光扫描、姿态/速度控制的 PID 以及安全时间戳停止机制。

**📊 数据集**

未使用公开数据集，所有实验均在实验室及校园走廊现场实时构建 2D 地图并实时采集感知数据；通过实际场景进行验证。

**📈 对比分析**

实验主要通过示例演示验证可行性，没有进行定量性能对比；演示中显示两种功能均能在 1.0–1.25 m 距离内完成跟随/召唤，速度被限制在 1 m/s 以下，安全性通过手势停机和超时机制保障。

**⚠️ 局限性**

局限性：感知仅为前向，无法实现 360° 观察；仅低速工作，未充分考虑自平衡动力学导致的停止/加速震动；手势识别受遮挡与误检影响；实验未覆盖带乘坐者的实际使用；安全保障缺乏正式验证；缺乏多用户、多环境的系统评估。

---

## 457. Data Analysis in the Wild: Benchmarking Large Language Models Against Real-World Data Complexities

**arXiv ID:** 2607.06482 | [PDF](https://arxiv.org/pdf/2607.06482v1)

**作者:** So Hasegawa `[一作]` (Fujitsu Research of America), Wei-Peng Chen `[通讯]` (Fujitsu Research of America)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DataGovBench benchmark，包含大规模多表格数据、元数据和外部知识，设计两项任务：Table QA（分解式问题及可视化答案）与Table Insight（自动洞察生成）

**💡 创新点**

创新点在于：①利用政府开放数据构建真实世界规模与多样性的基准；②结合多表、元数据、外部知识；③通过四阶段LLM+人工审核生成高质量问答对；④为洞察生成提供专家报告为真值；⑤设计可视化与多模态评估方案

**🔧 技术方法**

采用LLM（GPT‑4o、GPT‑5.1、Claude Sonnet 4.6、Gemini‑2.5 Flash 等）和专门的Answer/Insight Agents；利用表格特征类型序列化、Python代码生成、自校与反射；评估时用MLLM多模型投票、GPT‑4o LLaMA‑3‑Eval等自动评测

**📊 数据集**

使用了178个政府公开数据集，平均210K行、18列，最大11.9M行；包含173个用于Table QA、6个用于Table Insight；数据覆盖多表、外部知识、元数据

**📈 对比分析**

与多种基线（开源LLM、表格专用模型、文本到SQL）及Agent进行对比；结果显示即使加上Answer Agent，Table QA最高整体准确率<0.4；Insight任务最高摘要分<0.5；体现现有模型与agent在真实世界数据分析上的显著不足

**⚠️ 局限性**

局限性包括：①评估仍基于少量洞察数据（仅6个表）；②对外部知识整合依赖手工提取；③缺乏对SQL与Python等多输出形式的深入比较；④评估主要基于自动化指标，可能不足以捕捉主观洞察质量；⑤当前基准规模虽大但仍受数据可获取性限制

---

## 458. Domain-Driven Design in Practice: A Large-Scale Empirical Characterisation of the Open-Source Ecosystem

**arXiv ID:** 2607.06471 | [PDF](https://arxiv.org/pdf/2607.06471v1)

**作者:** Ozan Özkan `[一作]` (Eindhoven University of Technology), Mark van den Brand `[通讯]` (Eindhoven University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GitHub开放源代码仓库进行大规模挖掘与语义验证，构建2502个验证过的DDD项目数据库，并对其架构、技术栈、演进、社区活跃度等进行量化分析。

**💡 创新点**

首次引入GPT‑4o三重投票语义验证管线，大幅降低标签噪声，实现对“Rich Domain Models”的自动识别；并提供DDD在开源生态中的系统性数据驱动基准，揭示2017年转折点及C#/TypeScript主导的实践。

**🔧 技术方法**

结合MSR技术、GitHub GraphQL API、Python工具、SQLite存储，以及GPT‑4o多轮交互式检索与投票，实现高效语义验证与统计分析。

**📊 数据集**

从11,742个候选DDD相关仓库中筛选，最终得到2,502个高质量DDD仓库，包含元数据、提交记录、PR/issue、README、代码文件等多维度数据。

**📈 对比分析**

与人工标注的50个样本对比，LLM管线达到Cohen κ=0.77、F1=92.5%；相较传统关键词/主题过滤，噪声率从78.7%降至约20%；在架构分类上达94%以上的一致性。

**⚠️ 局限性**

依赖LLM模型可变性与版权、业务领域缺失导致25%仓库无明确业务上下文、仅限GitHub公开仓库且为英文、LLM验证受文件路径限制，较大仓库可能漏检。

---

## 459. RuBench: A Repository-Level Agentic Coding Benchmark with Natively Authored Russian Task Specifications

**arXiv ID:** 2607.06411 | [PDF](https://arxiv.org/pdf/2607.06411v1)

**作者:** Evgeny Shilov `[一作]` `[通讯]` (Independent Researcher), Evgeny Shilov (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 rubench 1.0，针对真实仓库中的 bug 修复任务，使用俄语原生用户请求作为任务说明，并通过维护者的回归测试作为判定标准。

**💡 创新点**

创新点在于：①将任务说明完全以目标语言（俄语）撰写而非翻译；②仅使用后置训练截止的修复提交，保证无数据泄露；③在产品级别（CLI+模型+推理力度）下评测，并公开完整轨迹检测模型替换。

**🔧 技术方法**

采用 Claude Code CLI、Sonnet5、Opus4.8、Haiku4.5、Codex CLI+GPT5.5 等部署配置，配合最大推理力度；对每个任务执行三次独立运行，统计 pass@1 并计算 Wilson 置信区间；通过配对 bootstrap 对比不同模型。

**📊 数据集**

数据集为 25 个真实修复提交，来自 aiohttp、aiogram、Laravel、NestJS、Fastify 等公开仓库；每条任务包含俄语描述、仓库快照、隐藏的维护者测试（以 SHA-256 提交）。

**📈 对比分析**

比较方法为每个配置在 25 个任务上运行三次，计算 pass@1 并给出任务级 95% 置信区间；配对 bootstrap 评估模型间差异。性能方面，Claude Code+Opus4.8 以 78.7% 的 pass@1 领先，Haiku4.5 仅 53.3%。

**⚠️ 局限性**

局限性包括任务数量少（25 条），所有说明由同一作者完成，缺少英译对照；私有测试限制第三方复核；产品可能在运行时替换模型，影响测评；此外，网络访问可能导致直接获取已修复代码，尽管数量极少。

---

## 460. FO Value Discovery and Partial Vertex Cover Discovery

**arXiv ID:** 2607.06446 | [PDF](https://arxiv.org/pdf/2607.06446v1)

**作者:** Enna Gerhard `[一作]` (University of Bremen), Jan Wodkte `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在令牌滑动模型中研究可达性和优化问题，提出 FO Value Discovery 与 FO Cost‑Value Decision 两个逻辑框架，并给出针对多种图类的参数化可解性和难度结论；

**💡 创新点**

首次将一阶逻辑的局部性与数值优化结合，构造了“局部化”与“锚定加权多彩距离独立性”两大算法工具，形成可统一应用的元定理；

**🔧 技术方法**

利用 Gaifman 局部性、FO 模型检查、结构化图类的逆转定理、L​​inEMSOL 优化、分解技术以及量化子消除等多种逻辑与图论技术；

**📊 数据集**

无实验数据，所有结论均为理论复杂度分析；

**📈 对比分析**

通过理论分析证明了 Partial Vertex Cover Discovery 在多种图类（局部结构化扩张、局部剪枝宽度、d‑退化图等）下为 FPT，Vertex Cover Discovery 在局部剪枝宽度下亦为 FPT；同时给出平面图、Clique‑Cover 数、Cutwidth 参数下的 [1]‑hard 与 1-hard 难度证明；

**⚠️ 局限性**

结果受限于特定图类（如局部结构化扩张、局部剪枝宽度、单调稳定类等）和逻辑片段（主子句、Boolean FO），完全 FO 版本的 FPT 仍未解决；

---

## 461. Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory

**arXiv ID:** 2607.06447 | [PDF](https://arxiv.org/pdf/2607.06447v1)

**作者:** Jihao Liu `[一作]` (Peking University), Bin Dong `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了名为Danus的多代理数学推理系统，通过共享事实图进行全局规划、并行证明搜索和严格验证，实现了对研究级数学问题的自动证明与论文撰写。

**💡 创新点**

核心创新在于将传统生成–验证–修正循环扩展到多代理协作，并以有向无环图（fact graph）为全局记忆，解决了多代理间干扰、记忆管理和长时序推理难题；同时引入了主代理进行全局规划与人机交互，显著提升了搜索深度与宽度。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑5.5‑pro、Claude Opus 4.8、GPT‑5.5）、专门的推理与验证代理、事实图结构、全局/局部记忆管理、模型上下文交互协议（MCP）以及自动化论文写作与验证流程。

**📊 数据集**

使用的“数据集”是六个真实研究级数学问题（如奇异性、奇点、组合学、foliation理论等）作为测试案例；每个问题在系统内部生成大量验证事实（从数百到数千条）。

**📈 对比分析**

与单独调用 GPT‑5.5‑pro、Claude Opus 4.8 或 Rethlas 的结果相比，Danus 在同一模型和硬件条件下实现了更高的成功率：完成了六个案例中的所有证明，其中 GPT‑5.5‑pro 在直接求解时均无效；Rethlas 在同一问题上三次失败。性能上，系统能够在几天内完成数千条验证事实，体现了宽度（并行搜索）和深度（长时序推理）的双重扩展。

**⚠️ 局限性**

局限包括：仍需人工输入关键提示或纠错；论文撰写阶段易出现压缩错误，需再度验证；系统受限于已知参考文献的正确性，若引用错误会传播错误；在缺乏可行证明思路的极难问题上，仍需人类提供创新思路。

---

## 462. WristMimic: Full-Body Humanoid Control with Wrist-Guided Manipulation

**arXiv ID:** 2607.06438 | [PDF](https://arxiv.org/pdf/2607.06438v1)

**作者:** Wongyun Yu `[一作]` (Pohang University Of Science And Technology), Minsu Cho `[通讯]` (Pohang University Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出WristMimic框架，将人体-物体交互的全身控制分离为无接触身体运动与基于物体及接触的手指操控，并仅通过手腕轨迹进行指导。

**💡 创新点**

创新点在于将手腕作为桥梁，利用手腕位置而非手指关节位姿监督，让手指通过物体动态和接触学习，从而减少对精细手指捕捉的依赖。

**🔧 技术方法**

使用物理模拟（Isaac Gym）、强化学习（PPO）、多模态奖励设计、手腕重置约束等技术实现。

**📊 数据集**

使用ParaHome和OMOMO两个人机交互数据集。

**📈 对比分析**

与InterMimic、SkillMimicV2等方法对比，WristMimic在成功率、物体位置/旋转误差等指标上均优于或相当，并在两数据集上表现一致。

**⚠️ 局限性**

局限在于只能处理需要手腕引导的抓取，缺乏对精细手指内部操控的支持；且目前仅训练场景特定策略，缺少跨场景通用性。

---

## 463. What Images Cannot Say: Language-Guided Olfactory Representation Learning

**arXiv ID:** 2607.06402 | [PDF](https://arxiv.org/pdf/2607.06402v1)

**作者:** Eleftherios Tsonis `[一作]` (École Polytechnique), Vicky Kalogeiton `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过视觉-语言模型生成场景描述，利用语言作为语义桥梁，将电子鼻气味信号与图像对齐，实现跨模态嗅觉表示学习与分解。

**💡 创新点**

创新点在于用语言指导来桥接视觉与嗅觉，并通过语言引导的潜在分解将物体特定气味与环境背景分离，提升了模型的可解释性与检索性能。

**🔧 技术方法**

主要技术包括 Qwen3‑VL‑30B 视觉‑语言模型生成描述、CLIP 的视觉/文本编码器、Transformer 结构的嗅觉编码器，以及对比学习与重建损失的联合训练。

**📊 数据集**

实验使用 New York Smells (NYS) 数据集，包含 7000 对图像‑气味样本。

**📈 对比分析**

相较于仅基于视觉的基线，在 Smell‑to‑Image、Smell‑to‑Text 及联合检索任务中，Recall@5 提升约 10‑12%，实现了当前最佳水平。

**⚠️ 局限性**

主要限制包括电子鼻设备的随机性与漂移、对大规模多模态数据集的依赖不足，以及在室外环境中的泛化能力尚待进一步验证。

---

## 464. Verification of Dynamic Holographic Behavior in Identity Documents

**arXiv ID:** 2607.06466 | [PDF](https://arxiv.org/pdf/2607.06466v1)

**作者:** Glen Pouliquen `[一作]` (IDnow Research Center), Ahmad Montaser Awal `[通讯]` (IDnow Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于背景估计与伪标签的光学可变装置（OVD）验证方法，并发布了新的MIDV‑DynAttack数据集，用于测试动态攻击场景；同时对现有方法进行基准对比。

**💡 创新点**

创新点包括：①发布了MIDV‑DynAttack（1200条动态/静态攻击），大幅扩展了原始MIDV‑Holo；②利用背景去除与HSV归一化显著增强光学可变信号；③采用伪标签训练帧级分类器，并通过Valid帧比例阈值实现序列判定，实现了对动态攻击的最佳检测。

**🔧 技术方法**

使用的技术主要有：背景减除、HSV归一化滤波、伪标签生成、图像增强、CNN帧级分类器、阈值决策，以及对比实验中的AUC/Recall/F1评估。

**📊 数据集**

使用的数据集包括：MIDV‑Holo、MIDV‑DynAttack（新发布）、以及原始MIDV系列（MIDV‑500/2020）用于训练、验证与测试。

**📈 对比分析**

与Direct Classifier、WSL、以及通用欺诈检测器等方法对比，HoloVerif在MIDV‑DynAttack上实现F1≈93%、AUC≈93%，在动态攻击上的Recall从基线的29%提升至61%；对静态攻击也表现优于其他方法，说明该方法在多样攻击场景下具有更高的鲁棒性。

**⚠️ 局限性**

局限性包括：仍易被Photo Replacement与Document Swap攻击误判；缺少新的合法样本，无法完整评估精度；对动态攻击的泛化能力仍有限，需要更完善的训练策略；未覆盖深度伪造等其他攻击类型。

---

## 465. Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation

**arXiv ID:** 2607.06564 | [PDF](https://arxiv.org/pdf/2607.06564v1)

**作者:** Jiaming Liu `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的 Lift3D-VLA 框架，将 2D 视觉语言动作模型提升到显式 3D 点云推理，并实现时间一致的动作生成。

**💡 创新点**

创新点包括：① 利用 2D 预训练模型的位置信息对齐 3D 点云；② Geometry-Centric Masked Autoencoding（GC-MAE）双目标自监督学习（重建 + 未来几何预测）；③ 逐层时间动作建模，使用 LLM 中不同层的表示来生成动作片段，实现时间连贯性。

**🔧 技术方法**

使用的技术有：2D 模型提升策略、点云分块器、GC-MAE、LoRA 微调、LLaMA2 7B 语言模型、MAE 解码器、Diffusion/Flow 生成动作、基于 3D 数据合成的自监督预训练。

**📊 数据集**

数据集包括：MetaWorld、RLBench 任务数据；真实世界抓取与装配演示（200 条/任务）；大规模机器人轨迹 140K+（自合成 3D 点云）与 400K 真实轨迹（Open‑X‑Embodiment、DROID、RoboMIND 等）；还有多种 2D 视觉语言预训练数据。

**📈 对比分析**

与 OpenVLA、π0.5、SpatialVLA、3DS‑VLA 等前沿 VLA 方法对比，在 MetaWorld 上平均成功率 87.7%（比最强对手高 13.8%），RLBench 82.8%（高于同类方法），真实世界平均 71%（比基线高 6%+）。同时在 OOD 场景下表现出显著的鲁棒性，性能下降仅 6–8%。

**⚠️ 局限性**

局限性：对透明或高反射物体的深度感知不足，单视角点云信息不完整，导致碰撞与定位误差；缺乏闭环控制，无法精细处理接触丰富的场景；需要进一步结合深度补全、多视角融合及闭环控制提升感知与执行稳健性。

---

## 466. ExplAIner: A Declarative Query Language for Explaining Classification Models

**arXiv ID:** 2607.06407 | [PDF](https://arxiv.org/pdf/2607.06407v1)

**作者:** Marcelo Arenas `[一作]` (Pontificia Universidad Católica de Chile), Bernardo Subercaseaux `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种用于布尔模型解释的声明性查询语言框架，旨在统一各种解释查询的表达、组合和分析。

**💡 创新点**

创新点在于引入了一种分层的查询语言，能够表达包括逆推、对比、特征基础和距离基础的广泛解释概念，同时保持可控的评估复杂性。

**🔧 技术方法**

使用了扩展的第一阶逻辑，结合了布尔模型的行为和部分实例的特性，构建了一个分层的查询语言。

**📊 数据集**

研究中使用了布尔模型，特别是决策树和确定性可分解布尔电路作为数据集。

**📈 对比分析**

与现有方法相比，提出的框架在每个固定查询的评估问题上属于布尔层次结构，能够在多项式时间内评估，尤其适用于决策树和更复杂的布尔模型。

**⚠️ 局限性**

限制在于该框架主要针对布尔分类模型，未来需要扩展到多类输出、非布尔特征或结构化特征域等更复杂的应用场景。

---

## 467. RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation

**arXiv ID:** 2607.06559 | [PDF](https://arxiv.org/pdf/2607.06559v1)

**作者:** Haoyu Zhao `[一作]` (DAMO Academy Alibaba Group), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种同步生成 RGB、深度和光流的 4D 表示（RGB‑DF），构建了三支分支的扩散网络，并结合预训练的视频扩散模型训练大规模数据集 Rynn4DDataset，随后设计了基于预测 4D 表示的逆动力学策略头（-Policy），实现高频闭环机器人控制。

**💡 创新点**

创新点在于：①将 RGB、深度与光流三种模态同步生成，直接可解算 3D 场景流；②引入三支分支的联合跨模态注意力和 3D RoPE，保证模态间空间一致性；③构建包含 2.54 亿帧、带深度与光流伪标注的 Rynn4DDataset；④利用内部 4D 表示实现一次前向传播即完成多步动作预测，突破传统 2D 模型的频率瓶颈。

**🔧 技术方法**

技术包括：视频扩散变压器（Wan 2.2‑TI2V‑5B）、三支分支架构、Joint Cross‑Modal Attention（共享 key/value、3D RoPE）、流匹配（flow‑matching）损失、Depth Anything 3 与 DPFlow 的伪标注、FP8 量化 + FlashAttention 3 推理加速。

**📊 数据集**

使用了 Rynn4DDataset 1.0（包含 Epic‑Kitchens、EgoVid、RoboMIND、RDT‑1B、Galaxea、RoboCoin、AgiBot 等人机交互和机器人抓取视频），以及在 TIANJI M6 机器人上采集的六个真实抓取任务数据，用于训练 -Policy。

**📈 对比分析**

在 4D 生成任务上与 Wan、CogVideoX、Free4D、TesserAct、4DNeX 等基线对比，RGB‑DF 在视觉质量、深度精度（δ₁=0.610）和光流误差（AEPE=0.170）上均优；在机器人抓取任务上，-Policy 的成功率最高达 97%（相较 DP、π₀、π₀.₅ 提升 10–20%）。

**⚠️ 局限性**

局限性包括：扩散推理导致的计算延迟，实际闭环控制频率仅约 9 Hz；模型主要针对单目 egocentric 视角，缺乏多视角和协同多机器人场景的适配。

---

## 468. MonoIR-RS: Infrared Remote Sensing Vision-Language Learning with CLIP and VLM Adaptation

**arXiv ID:** 2607.06552 | [PDF](https://arxiv.org/pdf/2607.06552v1)

**作者:** Jiaju Han `[一作]` (China University of Petroleum-Beijing at Karamay), Jiahuan Long `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模红外遥感视觉语言数据集及基准，并实现了CLIP与VLM在红外模式下的适配与评估。

**💡 创新点**

首创红外感知的文本重写与数据滤除流程，结合DiffV2IR生成合成红外图像，实现零样本与微调对比的完整评测。

**🔧 技术方法**

使用DiffV2IR光谱转换、CLIP对比微调、VLM指令微调（LoRA+投影）以及多指标诊断。

**📊 数据集**

基于FusionRS源集（5个遥感源）合成600k红外图像和59k IR‑aware caption。

**📈 对比分析**

与零样本CLIP/VLM对比，CLIP平均召回率提升至19.2%（+12.8点），VLM字幕IR线索覆盖率达100%，RGB泄漏降至0%。

**⚠️ 局限性**

限制在合成红外图像的真实性与人工生成的IR文本缺乏人类验证，且仅使用单一随机种子评估。

---

## 469. UniLM-Nav: A Unified Framework for Zero-Shot Last-Mile Navigation

**arXiv ID:** 2607.06537 | [PDF](https://arxiv.org/pdf/2607.06537v1)

**作者:** Zhuofan Zhang `[一作]` (Tsinghua University), Lifeng Fan `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种零训练的多模态大语言模型（MLLM）框架，用于移动机器人在完成目标定位后进行最后一米导航，使机器人在可操作的基座姿态下完成抓取或放置任务；框架分为视角选择、任务条件的亲和力定位以及基于几何的基座姿态推理三个阶段。

**💡 创新点**

将视角选择、亲和力定位和基座姿态推理统一到同一个MLLM后端，实现零训练的开放词汇最后一米导航；通过将2D视觉信息转换为3D几何，并显式提供机器人状态，使MLLM能够在几何约束下做出基座姿态决策；在不需要额外训练或标注的前提下即可在开放词汇环境中获得SOTA表现。

**🔧 技术方法**

多模态大语言模型（如 Gemini‑3‑Flash‑Preview、RoboBrain‑2.5‑4B）提示式推理；使用RGB‑D感知与深度图将二维亲和力点提升为三维坐标；构建短期观测记忆用于视角选择；将机器人姿态、任务指令与几何信息联合输入MLLM进行基座姿态预测。

**📊 数据集**

HomeRobot OVMM（Open‑Vocabulary Mobile Manipulation）基准数据集，包含物体定位、抓取、放置等连续任务；另外在实验室办公室场景中部署Unitree B2四足机器人与Z1机械臂进行真实世界测试。

**📈 对比分析**

与RL/启发式的HomeRobot、训练式方法MoManipVLA以及训练‑free方法MoTo进行对比；在OVMM验证集上，使用Gemini‑3‑Flash‑Preview时整体成功率达到53.5%，比MoTo高3.13个百分点；使用RoboBrain‑2.5‑4B时获得50.9%，仍优于大多数基线。消融实验表明视角选择、亲和力定位和几何推理每一环节均对最终性能有显著贡献。

**⚠️ 局限性**

假设目标导航已将机器人带到目标附近（1–2 m），若导航失败或目标视角不佳会导致整个流程失效；对MLLM的推理质量高度依赖，亲和力定位、视角选择和基座姿态推理的错误仍是主要失败原因；在细粒度空间约束（如“桌子左下角”）下表现不佳；缺乏主动局部探索机制，难以处理目标无法在近期观察到的情况。

---

## 470. Neural-ESO: A Dual-Pathway Architecture for Provably Robust Learning-Based Control

**arXiv ID:** 2607.06535 | [PDF](https://arxiv.org/pdf/2607.06535v1)

**作者:** Fan Zhang `[一作]` (University of Houston), Qin Lin `[通讯]` (University of Houston)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种双通道神经-ESO架构，利用神经网络做前馈扰动估计，并通过传统ESO在线纠正残差，从而实现鲁棒的学习驱动控制。

**💡 创新点**

创新点在于：①引入可控制的Lipschitz约束（通过谱归一化）保证学习组件的稳定性；②将神经先验与ESO结合形成“预测‑纠正”双通道，既能加速收敛又能避免过度依赖学习；③在训练、部署与跨域迁移阶段提供统一的安全收敛保证。

**🔧 技术方法**

主要技术包括：深度前馈神经网络（4层全连接 ReLU）、谱归一化、Lyapunov 小增益分析、基于ESO的误差估计、离线数据采集与在线双通道推断、以及跨域总扰动重训练（TDR）。

**📊 数据集**

数据集：通过在标准四旋翼上运行ESO收集的扰动估计，涵盖平地降落、倾斜坡落地、以及高速近地缠绵曲线等三种情境；训练时使用80%样本，验证20%；在OOD情境下进一步收集30秒的新域数据用于TDR。

**📈 对比分析**

与基线对比（PD、PD+NN、PD+ESO）在三种情境下评估稳态误差、标准差和RMSE。结果显示：在分布内，Neural‑ESO 的平均误差仅 0.37 cm；在OOD坡面，误差为 1.53 cm，仍保持稳定；在高速度近地 OOD，TDR 后误差降至 0.89 cm，显著优于 PD+ESO（1.91 cm）和 PD+NN（5.47 cm）。

**⚠️ 局限性**

局限性包括：①需手动调节 Lipschitz 常数以权衡精度与鲁棒性；②在极端域移时若不进行 TDR，性能仍会显著下降；③网络架构简单，可能在更复杂的动力学扰动上需要更深或更灵活的模型；④仅针对扰动估计，未直接学习完整系统动力学。

---

## 471. Crossroads: A Smart Contract Layer for Chain-Abstracted Assets

**arXiv ID:** 2607.06525 | [PDF](https://arxiv.org/pdf/2607.06525v1)

**作者:** James Austgen `[一作]` (Cornell Tech), Ari Juels `[通讯]` (Cornell Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了Crossroads智能合约层，将多链资产抽象为单链ERC-20代币，实现跨链桥接与统一平台交互。

**💡 创新点**

采用关键受限阈值签名委员会、可插拔或acles与快照式乐观取款、以及对资产转移进行后端链记录的低费用机制。

**🔧 技术方法**

阈值签名、智能合约、或acles（zkBridge、TEE、混合）、ERC-20代币、跨链桥技术。

**📊 数据集**

在Bitcoin、Ethereum、Solana等公开链上实现原型并进行功能验证。

**📈 对比分析**

通过证明满足soundness，展示任意用户可凭诚实仲裁者产生有效提现；原型在多链上展示低手续费和快递乐观访问，性能优于传统桥接。

**⚠️ 局限性**

对仲裁者诚信的假设、对终局确认的依赖、不同链最终化策略可能导致重组攻击风险、以及可插拔oracle实现的安全性依赖。

---

## 472. DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression

**arXiv ID:** 2607.06523 | [PDF](https://arxiv.org/pdf/2607.06523v1)

**作者:** Anna Cordoba `[一作]` (Instituto de Investigacion en Vision Artificial), Jesus Olivera `[通讯]` (Instituto de Investigacion en Vision Artificial)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对长上下文语言模型的 KV 缓存压缩方法 DepthWeave-KV，利用跨层残差因子化、token 级路由和在线误差追踪实现高效压缩。

**💡 创新点**

创新点包括：① 交叉深度残差因子化——在相邻 Transformer 层之间共享低秩通道基；② token 条件深度路由——根据 token 的重要性动态分配残差维度；③ 无需校准的在线注意力输出探针，用于实时调整压缩比例；④ 融合 CUDA 核，实现基底查找、残差反量化和注意力投影的单步执行，显著降低解码时的内存流量。

**🔧 技术方法**

核心技术：低秩通道基共享 + 4-bit/8-bit 量化残差 + 残差门控 + token 条件路由器 + 在线探针反馈 + 融合 CUDA 内核。

**📊 数据集**

使用的评估数据集包括 LongBench、Needle-in-a-Haystack、L-Eval、NarrativeQA、Qasper、HotpotQA、MultiFieldQA-en、GovReport、QMSum、TriviaQA 等长上下文检索、问答与摘要任务。

**📈 对比分析**

与 Full KV Cache 以及 StreamingLLM、H2O、SnapKV、PyramidKV、MiniCache、KVSharer、ChunkKV、TailorKV、EigenAttention 等压缩基线比较，DepthWeave-KV 在 64K 上下文下平均任务分数 62.9%（仅落后 Full 0.9%）、检索准确率 96.1%（高于 92.6% 的 TailorKV）、KV 内存压缩 8.3×、解码吞吐 72.8 tokens/s，且 perplexity 仅升 0.09，重构误差最低。

**⚠️ 局限性**

局限性：仅在局部深度窗口内共享基底，可能不适用于层间关联远的模型；依赖在线探针的准确性，对极端长句子或多跳推理可能仍有精度损失；需要额外训练残差门和路由器参数；目前仅针对 decoder‑only Transformer 进行评估，尚未验证在多模态或带稀疏注意力的模型上的通用性。

---

## 473. FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games

**arXiv ID:** 2607.06514 | [PDF](https://arxiv.org/pdf/2607.06514v1)

**作者:** Chase McDonald `[一作]` (Como Research), Wesley N. Kerr `[通讯]` (Riot Games)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了FootsiesGym，一个面向双人零和、非完全信息实时格斗游戏的开源基准环境，并在此环境上对多种强化学习算法进行基线实验，评估其学习表现与近似可利用性。

**💡 创新点**

创新点包括：① 以Footsies的中立阶段为核心，剔除连击等转移性机制，仅保留循环性、混合策略的空间与时间互动；② 通过可选的“充能动作”扩展动作空间，降低特殊攻击学习难度；③ 提供完整向量化模拟器（C# + gRPC）和PettingZoo接口，显著提升训练吞吐量；④ 通过对齐算法与近似最优对手的双重评估，揭示常用算法（PPO、EMAgnet、PFSP）在可利用性与策略主动性上的差异。

**🔧 技术方法**

使用技术包括：Unity引擎（C#）、头部渲染优化、gRPC通信、Python PettingZoo API、Proximal Policy Optimization（两种熵调度）、EMAgnet（策略正则化）、Prioritized Fictitious Self‑Play（采样式群体训练）、多线程并行化、可选稠密奖励。

**📊 数据集**

数据集主要是Footsies游戏本身，无需外部数据；训练与评估使用随机、无操作、以及基于最终检查点训练的近似最佳响应对手进行对抗。

**📈 对比分析**

比较方法：对所有算法与随机/无操作对手进行 win‑rate 曲线；对每个算法训练最佳响应并计算其 win‑rate 与对手的回报；绘制头对头回报矩阵与最佳响应回报图。性能表现：PPO 与 PPO(Sched.) 在随机/无操作对手上达到 85–95% 的 win‑rate；EMAgnet 的可利用性最高、回报最低；PFSP 在所有对手中取得最高的回报但仍易被最佳响应击败；整体而言，算法在主动进攻与特殊攻击利用方面表现欠佳。

**⚠️ 局限性**

局限性：① 可利用性仅为近似估计，未能给出精确值；② 高度简化的游戏模型仍远低于商业格斗游戏的深度与连击系统；③ 特殊攻击因动作序列长而难以探索，标准RL方法在此环境中效果有限；④ 只做了粗略调参，未对算法进行最优配置；⑤ 在极端行动延迟下游戏行为趋随机，进一步限制了对更复杂策略的研究。

---

## 474. GlassTENG: Self-Powered Triboelectric Nanogenerator based Sensing of Pulse, Jaw, and Upper Facial Activity from Everyday Glasses

**arXiv ID:** 2607.06509 | [PDF](https://arxiv.org/pdf/2607.06509v1)

**作者:** Raj N. Dave `[一作]` (Northwestern University), Nivedita Arora `[通讯]` (Northwestern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了基于自供能的TENG传感器集成到眼镜中的系统，用于低功耗测量脉搏、咀嚼和面部活动；

**💡 创新点**

将自供能TENG与超低功耗前端结合，实现在无电池、无胶水的眼镜中获取多模生理信号，并将单通道功耗降至1.36 µW、三通道总功耗4.1 µW；

**🔧 技术方法**

使用PDMS/FEP双层TENG结构、TLV8802高阻抗放大器的低功耗前端、ESP32C6验证平台以及随机森林等机器学习方法；

**📊 数据集**

使用20名受试者的面部与咀嚼活动数据和10名受试者的心率数据与Polar胸带基准作为实验数据集；

**📈 对比分析**

采用留一子集交叉验证（LOSO）对7类活动进行分类，随机森林取得93.8%总体准确率；心率MAE为1.82 BPM，Bland–Altman限值±5.6 BPM；

**⚠️ 局限性**

存在说话与吃饭活动混淆、硬件上S2下摆臂不常规，以及尚未实现完整低功耗MCU/ASIC与能量收集集成的问题。

---

## 475. RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models

**arXiv ID:** 2607.06504 | [PDF](https://arxiv.org/pdf/2607.06504v1)

**作者:** Qian Sun `[一作]` (Nanjing University), Shao-Qun Zhang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RMISC大型真实多变量时间序列语料库，并用其与合成及单变量语料对四款主流时间序列基础模型进行预训练与零样本评估。

**💡 创新点**

创新点在于构建包含约200个子数据集、142 B时点的高质量真实多变量语料，并验证其显著提升TSFM的跨域零样本泛化能力。

**🔧 技术方法**

采用Chronos‑2、GTT、Moirai‑2.0和TimesFM‑2.5四种TSFM，结合多源语料预训练、上下文窗口随机裁剪、实例归一化等技术，进行大规模预训练与评估。

**📊 数据集**

使用数据集包括：RMISC（真实多变量语料）、合成多变量语料（SM）、真实单变量语料（RU）以及GIFT‑Eval和fev‑bench等公开基准。

**📈 对比分析**

通过对ID损失曲线与OOD MASE/WQL指标的对比，发现包含RMISC的三源组合（RU+SM+RM）在OOD任务中平均MASE下降约4.5%，且在多变量与单变量子集均表现优于仅用合成或单变量语料的模型。

**⚠️ 局限性**

局限性包括：RMISC在ID收敛时不一定最优，且对医疗、金融等高价值领域的数据获取受限；实验仅评估零样本性能，未涉及微调效果。

---

## 476. Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade

**arXiv ID:** 2607.06503 | [PDF](https://arxiv.org/pdf/2607.06503v1)

**作者:** Kai Ruan `[一作]` (Renmin University of China), Hao Sun `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于LLM内部激活的可置信度调度的终止级联方法，用以在多步任务中提前识别失败轨迹并及时中止，节约推理计算。

**💡 创新点**

创新点在于：①首次利用内部隐藏状态在第一轮即可预测最终失败；②设计了分层召回校准门（Clopper–Pearson阈值）并在全局召回约束下联合搜索预算；③引入数据规模下的全局召回证书，明确数据需求。

**🔧 技术方法**

技术包括：交叉折叠线性探针（logistic回归）对隐藏层激活做失败预测；Clopper–Pearson分位数阈值校准门；基于验证集的预算网格搜索；分布式全局召回证书。

**📊 数据集**

使用TextCraft文本生成式环境，分别在两大LLM模型（Llama‑3.2‑3B和Qwen‑2.5‑7B）上收集800条任务样本（共约1600条）。

**📈 对比分析**

与单阈值门和均匀预算基线对比，级联在90%全局召回目标下实现47.1%（Qwen）/37.2%（Llama）计算节约，1.6–1.7倍于最佳单阈值门；在更严格目标下仍保持较好节约；内部激活基线明显优于仅行为特征。

**⚠️ 局限性**

局限性包括：仅在单一环境与两模型验证，需在部署时暴露内部激活；全局召回证书受样本量限制，极高召回目标难以证明；预算搜索离散且计算开销大；在分布式或批处理部署时，真正的时间/成本收益需进一步评估。

---

## 477. EntroPath: Maximum Entropy Path Ensemble Embedding for Manifold Learning

**arXiv ID:** 2607.06497 | [PDF](https://arxiv.org/pdf/2607.06497v1)

**作者:** Przemysław Rola `[一作]` `[通讯]` (Krakow University of Economics), Przemysław Rola (Krakow University of Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了一种基于最大熵随机游走（MERW）的流形学习方法，该方法通过对k步路径集合的自由能聚合来恢复数据图的测地几何。

**💡 创新点**

创新点包括：①使用MERW而非传统随机游走，消除度数偏差；②引入自由能（log‑sum‑exp）路径相似度，形成光滑的软最小化；③在短时极限下证明该相似度收敛为测地距离；④给出与热核、Gram矩阵的精确对应关系；⑤提供可扩展的标记点投影和伪时间推断。

**🔧 技术方法**

主要技术包括：最大熵随机游走、谱分解、离散Schrödinger算子、Varadhan热核公式、自由能相似度、Gram核因式分解、Landmark Nyström扩展、Von Neumann熵深度选择。

**📊 数据集**

评测数据集：合成流形（瑞士卷、球面、环面、树形等）以及多种单细胞转录组数据（Paul15、Nestorowa、Pancreas、Lymphoid、Embryoid Body、Arabidopsis根系）。

**📈 对比分析**

与PHATE、HeatGeo、DTNE、Diffusion Maps、Isomap、UMAP、t‑SNE等方法比较；在距离层面与嵌入层面均能匹配或超过其它基于流形几何的方法，尤其在非均匀采样和分支轨迹场景中优势显著；在单细胞可视化与伪时间推断中实现与DTNE相当或更优的DEMaP、可信度等指标。

**⚠️ 局限性**

限制：深度k的无监督选择在短分支上噪声较大；当前仅离散k步形式，未实现连续时间热核；对有向亲缘矩阵支持有限；在极大规模数据下仍需进一步加速（尽管提供标记点近似）。

---

## 478. ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation

**arXiv ID:** 2607.06565 | [PDF](https://arxiv.org/pdf/2607.06565v1)

**作者:** Tianjiao Yu `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一3D基础模型ELSA3D，利用弹性语义锚定实现语言与几何的稀疏交互与自适应推理。

**💡 创新点**

创新点包括①Anchor Tokens将选定文本令牌与多尺度Octree几何对齐并写回统一序列；②弹性路由器同时控制块执行、MLP宽度、锚点生成与尺度分配，实现计算与推理弹性；③多尺度Octree VQ‑VAE赋予显式尺度标记，构建语义与几何的匹配层级。

**🔧 技术方法**

使用的技术包括多尺度Octree VQ‑VAE、三头Transformer路由器、Anchor Tokens与稀疏交叉注意力、弹性块跳过与宽度自适应、两阶段训练（Tokenizer + Autoregressive + 路由正则）。

**📊 数据集**

训练集为3D‑Alpaca、Trellis‑500K、ObjaverseXL、ABO、3DFUTURE、HSSD及UltraChat；评测集为Toys4K和200张无重叠野生图像。

**📈 对比分析**

与多种开源3D生成与理解基线（如InstantMesh、Shape‑LLM‑Omni、CoRe3D、LLaVA、Qwen等）对比，使用CLIP、FD、KD、PSNR、LPIPS、MMD、Q‑Align、BLEU、ROUGE、METEOR、SBERT等指标；在image‑to‑3D、text‑to‑3D和captioning上均获得最高或第二高分，且算力与推理时延约减半。

**⚠️ 局限性**

主要限制是仍需较大显存与GPU时延；对极细粒度几何细节捕捉仍有限；锚点与路由器的正则与阈值需要精细调参；在极端多模态或复杂场景下，模型的几何细节重建可能不足。

---

## 479. Embodied Human-Robot Interaction via Acoustics: A MARL Approach with AcoustoBots for Spatial Data Physicalization

**arXiv ID:** 2607.06563 | [PDF](https://arxiv.org/pdf/2607.06563v1)

**作者:** Shiqi Liu `[一作]` (University College London), Sriram Subramanian `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种移动声学数据物化平台AcoustoBots，利用TurtleBot3机器人和超声相控阵列实现城市数据的物理化展示。

**💡 创新点**

创新点在于将声学悬浮与多智能体强化学习（MARL）结合，创建了一个能够在城市环境中移动并实时展示数据的机器人系统。

**🔧 技术方法**

使用了多智能体深度确定性策略梯度（MADDPG）算法进行策略学习，以及高频率的Gerchberg-Saxton相控阵声学控制器来维持悬浮稳定性。

**📊 数据集**

在一个4米x3米的缩放英国地图上进行实验，使用PhaseSpace进行定位以实现可重复的多机器人试验。

**📈 对比分析**

通过单机器人和双机器人模式进行比较，单机器人任务成功率为90%，双机器人任务成功率为80%，且碰撞次数较低，显示出良好的导航和物化表现。

**⚠️ 局限性**

局限性包括成功率未达到100%，对环境的随机性和探索敏感，且实验在简化的2D网格环境中进行，未能完全捕捉更复杂的真实世界场景。

---

## 480. Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs

**arXiv ID:** 2607.06540 | [PDF](https://arxiv.org/pdf/2607.06540v1)

**作者:** Zhenyu Liu `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Lychee‑FD，一种原生端到端的全双工语音语言模型，旨在消除模态干扰并提升语音智能与交互流畅性。

**💡 创新点**

创新点包括首次阐明模态干扰的根本原因，并提出分层参数分离与语义对齐通道的双重策略，有效解决梯度冲突和语义稀释问题。

**🔧 技术方法**

采用分层Transformer参数分离、语义对齐通道、Gradient cosine similarity分析，以及 Whisper‑v3‑large 编码器和 CosyVoice2 语音分词器等技术。

**📊 数据集**

在训练时使用约 140K 条合成的全双工对话数据（包含打断、用户/AI backchannel），并在评估中使用 LlamaQ、WebQ、TriviaQA、FDBench、FullDuplexBench 等公开基准。

**📈 对比分析**

与 Freeze‑Omni、VITA‑1.5、dGSLM、FLM‑Audio、Moshi、Fun‑Audio‑Chat 等基线对比，Lychee‑FD 在 Spoken QA 上平均提升 7.4%/8.8%，在 FullDuplexBench 1.5 上提升 28.5%，在 10/11 交互指标上领跑，并保持 100% takeover rate 与低延迟。

**⚠️ 局限性**

局限性主要在开放麦环境下的意图识别与背景噪音区分不足，数据集缺乏多说话人情境与细粒度意图标注，导致模型在真实场景下的干扰检测性能受限。

---

## 481. Trust-Aware Citation Cartel Ranking in Scholarly Knowledge Graphs

**arXiv ID:** 2607.06528 | [PDF](https://arxiv.org/pdf/2607.06528v1)

**作者:** Pratyush Gupta `[一作]` (IIIT Delhi), Syam Sai Santosh Bandi `[通讯]` (IIIT Delhi)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于信任感知的论文级引用卡特尔排名管道。

**💡 创新点**

创新地将LLM生成的引用意图标签与SciBERT模型结合，设计了多维Composite Cartel Index。

**🔧 技术方法**

使用LLM教师-学生学习、SciBERT、加权PageRank、Louvain社区检测等技术。

**📊 数据集**

使用基于DBLP的500k论文、约4.87M引用的闭域图。

**📈 对比分析**

与仅基于密度、膨胀、语义等单一基线对比，综合排名与稀疏抽取实验验证，表现出更高的区分度。

**⚠️ 局限性**

对罕见标签的识别仍有限，且对全局结构的假设可能限制跨领域适用。

---

## 482. Bridging Physical Reasoning and Task Generalization via Visual Action Outcome Reasoning Alignment

**arXiv ID:** 2607.06522 | [PDF](https://arxiv.org/pdf/2607.06522v1)

**作者:** Han-Jun Ko `[一作]` (National Taiwan University), Yu-Chiang Frank Wang `[通讯]` (University of California Merced)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的奖励设计VAORA，利用视觉结果对视觉‑语言模型（VLM）的链式推理进行对齐，以消除幻觉式推理与行动偏差；

**💡 创新点**

创新点在于将推理过程与环境结果直接关联，提出可视化对齐奖励与视觉‑动作对齐奖励两种奖励，并通过预训练专家DQN提供的连续成功概率实现稠密奖励，显著提升VLM的通用物理推理能力；

**🔧 技术方法**

技术主要包括：VLM的监督式微调（SFT）+强化学习（RL），基于符号空间的推理与视觉映射、奖励计算；使用GDPO进行策略优化；通过Prompt工程让VLM生成符号化推理轨迹；使用预训练DQN作为专家引导的稠密奖励；

**📊 数据集**

使用的主要数据集为：PHYRE（交互式物理推理基准），Virtual Tool（跨环境物理模拟器），以及Craft VQA（物理因果问答基准）；

**📈 对比分析**

与多种基线（InternVL‑3.5‑8B、Qwen3‑VL‑8B‑Instruct、Claude、GPT‑5.4、Gemini、DQN专家）比较，VAORA在PHYRE的跨任务通用度上显著优于所有开源/闭源模型，并在Virtual Tool零样本跨环境迁移中匹配或超过Gemini‑3.1‑Flash/Pro；在Craft VQA上亦提升了所有物理推理类别的准确率；

**⚠️ 局限性**

局限性在于模型仅在单回合内作出一次行动预测，无法在后续尝试中基于环境反馈进行自适应更新，限制了在物理概念或动力学大幅变化环境中的适用性。

---

## 483. Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning

**arXiv ID:** 2607.06501 | [PDF](https://arxiv.org/pdf/2607.06501v1)

**作者:** Anxing Xiao `[一作]` (National University of Singapore), David Hsu `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HUME框架，帮助服务机器人在未知环境中自动生成、验证并更新关于物体位置、属性及动作效应的假设，从而在规划过程中不断扩展并纠正其世界模型。

**💡 创新点**

创新点包括：①将模型扩展视为不确定且可验证的过程；②利用基础模型（LLM/视觉‑语言模型）生成结构化假设并提供验证条件；③在经典规划中通过所有结果确定化将验证动作纳入计划，使任务规划与信息获取同步进行；④通过贝叶斯自适应视角对不确定性进行显式推理。

**🔧 技术方法**

技术方法：基于LLM的假设生成与验证、视觉‑语言模型（VLM）用于假设确认、Fast Downward/ENHSP等经典规划器、贝叶斯自适应MDP建模、所有结果确定化与分支裁剪、感知管道（OWLv2、Segment Anything、RGB‑D 3D 场景图）、低层执行技能库。

**📊 数据集**

实验使用的环境与数据集：1）Block Processing World（扩展版块世界）；2）AI2‑THOR（ProcTHOR 房屋），包含多种任务；3）真实 Fetch 移动操作机器人在厨房、客厅、餐厅的物理实验；4）Autolife S2 人形机器人进行微波炉操作；感知模块使用 OWLv2、Segment Anything、RGB‑D 3D 点云。

**📈 对比分析**

方法对比：将六种方法（基于形式规划/LLM规划，是否包含模型扩展，是否不确定扩展）与HUME的性能对照；在仿真中，加入假设并保持不确定性可显著提升成功率与 SPL；在真实世界中，HUME 的成功率与 SPL 均高于所有基线；实验报告显示不确定性扩展方案在高开放度任务中明显优于确定性或无扩展方案。

**⚠️ 局限性**

局限性：①不确定性仅以三元状态（真/假/未知）表示，缺乏置信度度量；②计划过程中对负面验证结果采取乐观策略，可能导致风险承担过高；③假设验证条件假设可由 LLM 完整推导，实际中可能因几何/感知复杂性不足；④任务抽象需由指令显式提供，缺乏自发式失效推理；⑤未考虑任务不可行性判定与不确定性搜索空间剪枝。

---

## 484. Multi-Agent Deep Reinforcement Learning for Multi Objective Battery Management in Dairy Farms

**arXiv ID:** 2607.06489 | [PDF](https://arxiv.org/pdf/2607.06489v1)

**作者:** Marcos Eduardo Cruz Victorio `[一作]` (University of Galway), Karl Mason `[通讯]` (University of Galway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于差分进化与多智能体深度强化学习的层级化分布式电池管理与动态定价框架，优化乳业农场可再生能源的使用与收益。

**💡 创新点**

将多目标优化与深度强化学习相结合，采用层级化动态定价调节主电网交互，同时利用多智能体独立控制电池，实现18%收益提升与0.85 MW²负载波动降低。

**🔧 技术方法**

使用差分进化算法（Scipy）、Proximal Policy Optimization（stable‑baselines3）、PyPSA仿真、欧元/kW 电价预测与校正等技术。

**📊 数据集**

使用2022年爱尔兰单一市场电价数据、农场负荷与光伏发电仿真数据以及分布式发电与储能容量信息。

**📈 对比分析**

通过10次随机种子实验与规则式电量调度对比，评估能源套利利润、成本及电压/相位波动，结果显示DRL+动态定价提升利润18%，且电压符合EN 50160标准。

**⚠️ 局限性**

仅在单一地区模拟，未考虑多站点协调与大规模系统，且依赖人工设置的预测误差与超参数，未来需验证多场景与实时调度的鲁棒性。

---

## 485. Vision as Unified Multimodal Generation

**arXiv ID:** 2607.06560 | [PDF](https://arxiv.org/pdf/2607.06560v1)

**作者:** Xiaoyang Han `[一作]` (SenseTime Research), Quan Wang `[通讯]` (SenseTime Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建统一的多模态生成框架 SenseNova‑Vision，将不同类型的计算机视觉任务（结构化感知、密集几何预测、分割、多视角几何）统一为文本、图像或混合文本‑图像的生成目标，并在此基础上训练单一模型。

**💡 创新点**

创新点：①通过将任务标注转换为可直接在 UMM（统一多模态生成模型）中解码的文本、图像和混合目标，消除了对任务特定头的需求；②提出 SenseNova‑Vision Corpus，规模达 50M 条指令‑响应样本，覆盖四大视觉任务族；③展示同一模型在多任务、语言定义的变体任务上均可通用，证明统一生成可作为未来视觉“基础模型”的可编程接口。

**🔧 技术方法**

技术手段：使用 Bagel‑7B‑MoT 作为基础 UMM；VAE 视觉编码器 + SigLIP2 高分辨率编码；混合任务联合微调；文本生成采用标准交叉熵；图像目标通过 VAE 潜在空间编码并使用 rectified‑flow 训练；特殊标记与统一解析协议，用于摄像机姿态、边框、颜色等；多模态提示与数据混合采样。

**📊 数据集**

数据集：从公开数据集构建 SenseNova‑Vision Corpus，主要包括 COCO、LVIS、HumanRef、RefCOCOg、VisDrone、HierText、ICDAR15、ScreenSpot‑V2、COCO‑Kpt、NYUv2、KITTI、ETH3D、ScanNet、DIODE、7Scenes、RealEstate10K、CO3Dv2 等；通过转换规则将其标注转化为可解码的文本/图像目标，最终形成 50M 条样本。

**📈 对比分析**

比较方式与性能：①在结构化感知、密集预测、分割和多视角几何四大基准上与各自专业模型（如 Detecto, Sam, DepthAnything, VGGT, etc.）和通用视觉模型（Youtu‑VL、Vision Banana）对齐；使用标准评测指标（mAP、mIoU、F1、δ1、RRA、RTA 等）。结果显示：SenseNova‑Vision 在结构化感知上领先或相当，密集几何与分割与最佳专用模型相近，多视角几何与主流通用模型相当，整体在多任务场景下优于现有通用模型。

**⚠️ 局限性**

局限性：①对极端稀疏或极小目标的检测/分割仍不如专业模型；②长文本序列或大规模多目标的生成收敛速度慢；③对高分辨率输出与实时推理的效率尚待提升；④依赖预训练 UMM，缺乏深度几何先验；⑤在复杂的跨模态推理（如物理交互、动态视频）上尚未验证，需要进一步扩展。

---

## 486. RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation

**arXiv ID:** 2607.06558 | [PDF](https://arxiv.org/pdf/2607.06558v1)

**作者:** Haoyu Zhao `[一作]` (DAMO Academy, Alibaba Group), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种数字化遥操作框架，通过手势流驱动动作感知世界模型，生成高保真、可用于模仿学习的机器人视角视频。

**💡 创新点**

创新点包括：深度感知的手部姿态表示、分阶段人类→机器人知识迁移训练、以及基于流匹配与对抗分布匹配的自回归蒸馏实现实时交互生成。

**🔧 技术方法**

技术手段包括：视频扩散Transformer（DiT）、3D VAE编码、分布对齐的姿态嵌入、两阶段进化训练、因果流匹配预热与分布匹配蒸馏、以及基于Vive跟踪的动作重定向。

**📊 数据集**

使用了大规模人类视角数据集 EgoDex 与 VITRA 进行预训练，随后在自采集的 1,800 条双臂机器人遥操作演示数据上微调。

**📈 对比分析**

与传统 I2V、动作条件视频生成基线相比，本文模型在 PSNR、SSIM、LPIPS、FVD 等视觉指标上提升 5–10 点；在四个复杂抓取/推/举/放置任务中，真实机器人成功率提升 10–20%，并实现零真实数据的 Sim2Real 转移。

**⚠️ 局限性**

局限性包括：对细粒度流体或高度可变形物体的建模不足；以及需要针对每个机器人平台单独微调，限制了跨平台的可扩展性。

---

## 487. Quantum combinatorial games

**arXiv ID:** 2607.06550 | [PDF](https://arxiv.org/pdf/2607.06550v1)

**作者:** Dieks Scholten `[一作]` (Radboud University), Simona Samardjiska `[通讯]` (Radboud University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个可量化的组合游戏框架，证明在完美对局中量子玩家无法超过经典玩家，且在对手犯错时可通过量子叠加放大优势，并将该框架应用于拜占庭共识问题。

**💡 创新点**

提出了对组合游戏的量子化定义、可注射和固定长度条件，并给出了“量子Zermelo定理”，首次证明完美经典策略在量子版本中仍然最优；同时设计了利用量子叠加放大错误的策略。

**🔧 技术方法**

使用组合游戏理论、bisimulation、量子门（酉矩阵）实现的量子移动、分布式算法模型和概率分布的数学工具。

**📊 数据集**

无具体数据集；所有结论均基于理论证明与数学模型。

**📈 对比分析**

通过与经典组合游戏的等价性（bisimulation）进行理论比较；实验性评估未涉及，性能上表现为理论上不提升完美玩法，且在错误放大策略下可获得更高成功率。

**⚠️ 局限性**

局限在于只处理无随机性、双人、完美信息的短期组合游戏；对多玩家、带概率或信息不完全的情况缺乏直接推广；量子移动的定义过于宽松，若采用更严格的“完全有效”定义则放大策略失效。

---

## 488. Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion

**arXiv ID:** 2607.06546 | [PDF](https://arxiv.org/pdf/2607.06546v1)

**作者:** Shervin Khalafi `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图去噪的谱视角，提出基于图卷积的注意力机制（GCA）并验证其在去噪和扩散模型中的优势。

**💡 创新点**

提出Spectral Attention理论和GCA实现，证明相对于线性注意力可获得更优的谱降噪，并利用softmax进一步降噪。

**🔧 技术方法**

使用图卷积滤波、谱注意力、softmax、SBM理论分析以及R-PEARL随机特征编码等技术。

**📊 数据集**

合成SBM、SPECTRE-SBM、ENZYMES、PROTEINS、IMDB、COLLAB、DEEZER-EGO-NETS等真实图数据集。

**📈 对比分析**

与传统Graph Transformer和DiGress对比，使用验证损失、logP、NLL、MMD等指标，GCA在多数据集上均提升数个百分点，尤其在谱多样性高的场景。

**⚠️ 局限性**

仍需在图尺寸极大或谱分布极端单一时效果有限，对softmax参数选择敏感；实际应用需在计算开销与精度间权衡。

---

## 489. FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference

**arXiv ID:** 2607.06519 | [PDF](https://arxiv.org/pdf/2607.06519v1)

**作者:** Anna Córdoba `[一作]`, Jesús Olivera `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时对LLM KV缓存进行深度频率分解压缩的算法FreqDepthKV，兼顾内存降低与任务准确性

**💡 创新点**

创新点在于把相邻层KV状态分解为共享的低频深度组件与稀疏的高频残差，并通过在线注意力logit探测动态分配共享/残差/精确三种缓存模式

**🔧 技术方法**

采用DCT基深度变换、在线头路由探测、重构感知损失、稀疏残差存储以及与量化/删选方法兼容的后端实现

**📊 数据集**

在LongBench、Needle-in-a-Haystack、NarrativeQA、GovReport、MultiNews、HumanEval、MBPP等长上下文问答、检索、摘要与代码生成数据集上评估

**📈 对比分析**

与MiniCache、H2O、StreamingLLM、SnapKV、PyramidKV、KVQuant、KIVI等基线对比，FreqDepthKV在所有指标上获得最高压缩缓存精度，压缩比达3.9×，解码吞吐70.4 tokens/s，TTFT 2.06s，峰值KV内存6.2GB，性能优于其它方法

**⚠️ 局限性**

局限性：缓存路由固定于预填阶段，无法随生成序列动态调整；未与量化/序列维度压缩联合实验；模型级训练适配尚未探索

---

## 490. Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engine

**arXiv ID:** 2607.06495 | [PDF](https://arxiv.org/pdf/2607.06495v1)

**作者:** Juan S. Santillana `[一作]` `[通讯]` (Independent Researcher), Juan S. Santillana (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 Pitwall 系统，能够在 Formula 1 赛道实时解析计时流，使用校准后的蒙特卡罗模拟生成可信、三语（英、西、葡）策略简报，并通过验证器逐句检查事实准确性。

**💡 创新点**

创新点在于：① 采用双路径（校准‑决策）分离机制，确保概率预测与决策建议分别最优；② 在训练、推理和发布三阶段都使用基于事实验证的门控，防止 hallucination；③ 引入统一随机数（CRN）技术提升对比度与置信区间估计；④ 在公开数据上实现端到端实时部署，首次公开记录。

**🔧 技术方法**

技术方法包括：向量化 Monte Carlo 模拟（N=2000）、概率校准（Brier、ECE、可靠性图）、共同随机数（CRN）策略评估、可解释的过渡模型（如 H7 过渡核、H9 归一化）、基于 LoRA 的语言模型微调、以及基于状态的声明提取与验证器。

**📊 数据集**

使用的数据集为 157 场公开 F1 计时数据（2018‑2026 年，除 2022 年外）以及 3,045 条三语简报训练样本，训练集 126 场，留置集 29 场（2025‑2026赛季）进行模型校准与评估。

**📈 对比分析**

评估方法为：在留置集上进行全场模拟，Brier 0.0745、E​CE 0.0297、Spearman ρ ≈ 0.77，90.3% 的冠军落在预测前 3 名；在 2026 年奥地利与英国大奖赛实时测试中，系统实现 1 秒级输入、几秒级输出，概率预测在第 42 站后即锁定最终冠军，误差低于 10⁻³。

**⚠️ 局限性**

局限性包括：仅依赖公开计时流，缺乏车队内部遥测与燃料/热状态细节；雨天策略与后期裁判处罚未建模；对手仅按轨迹反应，未实现完整覆盖游戏；校准受存储数据质量与时代变化限制；最终官方排名可能因赛后处罚与裁判决策与模型预测不符。

---

## 491. Mitigating Domain Shift in Conditioned Floor Plan Generation: Synthetic Pre-training for Data-Efficient Adaptation

**arXiv ID:** 2607.06483 | [PDF](https://arxiv.org/pdf/2607.06483v1)

**作者:** Matthieu Ospici `[一作]` (Homiwoo), Adrien Bernhardt `[通讯]` (Homiwoo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了在条件地板平面生成中跨域迁移的鲁棒性问题，并提出通过大规模程序化合成数据预训练来提升模型的跨域适应性。

**💡 创新点**

创新点在于：①首次系统评估不同生成范式在三大公开数据集之间的域移位影响；②设计一种极端几何多样化、却严格满足物理约束的合成数据生成流程；③证明合成预训练既能提升零样本跨域性能，又能在低样本情形下显著加速微调。

**🔧 技术方法**

使用程序化合成策略（形状变形、边界拼接、门口布置等）生成训练数据；在两个代表性生成模型（DPFM 与基于扩散的约束模型）上进行实验；采用 MPE 与 NGED 两项指标衡量几何与拓扑质量。

**📊 数据集**

采用 RPLAN、MagicPlan、Swiss Dwellings 三个公开地板平面数据集以及自建的 135k 场景合成数据集。

**📈 对比分析**

与仅使用真实数据预训练或直接在目标域训练的基线相比，合成预训练在零样本跨域中平均降低 40‑50% MPE，甚至在 MagicPlan 上超越完全域内训练；在 1k‑10k 样本微调中，合成预训练的误差比真实域初始化低 30‑50%。

**⚠️ 局限性**

局限性包括：与高几何频率、碎片化拓扑（如 Swiss Dwellings）等复杂域仍难以完全克服；合成数据虽然提升泛化但在极其规整或特殊域内仍落后于专门化训练；以及当前合成过程仍依赖于单一形状来源，可能限制多样性。

---

## 492. Life Style Levels: Neighborhood Delineation using Geospatial Data

**arXiv ID:** 2607.06529 | [PDF](https://arxiv.org/pdf/2607.06529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 493. GraphBU: MILP Instance Generation with Graph-Native Block Units

**arXiv ID:** 2607.06532 | [PDF](https://arxiv.org/pdf/2607.06532v1)

**作者:** Xiaolei Guo `[一作]` (Shanghai Jiao Tong University), Dongdong Ge `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于图本地块单元的混合整数线性规划实例生成方法

**💡 创新点**

创新点在于把局部子问题与其耦合接口一起定义为可重用块单元，并通过接口检测与推广实现跨块耦合显式化；同时提供了对行列置换不变的理论保证

**🔧 技术方法**

利用约束-变量双部图表示、邻域分布的span/entropy/degree得分、图切分与递归细化、接口推广与兼容性检查，以及可行性保持的理论条件

**📊 数据集**

四类工业MILP数据集：组合拍卖（CA）、容量设施位置（FA）、物品放置（IP）、工作负载安排（WA）

**📈 对比分析**

与随机替换、G2MILP、MILP-Studio三种基线对比，使用图统计相似度、可行率、求解时间和下游Predict‑and‑Search（PS）性能作为评价指标；实验显示相似度最高、可行率达96.7%，PS训练效果提升约8%

**⚠️ 局限性**

局限性包括：对块库规模与接口匹配的依赖，复杂实例可能兼容性低；仅在结构相似度上评估，未深入分析求解难度保持的理论保证

---

## 494. ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation

**arXiv ID:** 2607.06555 | [PDF](https://arxiv.org/pdf/2607.06555v1)

**作者:** Ruihang Zhang `[一作]` (University of Toronto), David B. Lindell `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用视频扩散模型将单目视频中的目标表面转化为同等运动的代理多边形视频，从而实现对每个像素的6自由度位姿跟踪。

**💡 创新点**

将3D姿态估计问题转化为视频对视频翻译，再借助已知代理几何通过经典PnP求解，避免了对3D模型、深度或分割的需求。

**🔧 技术方法**

Fine‑tuned 大规模视频扩散模型（Wan‑14B）+ LoRA、VAE、DiT、流匹配训练、噪声调度偏移、PnP与光流平滑优化。

**📊 数据集**

合成数据集：约3.5万对（源视频+代理视频）来自Objaverse 3D资产，使用Blender渲染并随机生成运动。

**📈 对比分析**

与多种基准（FoundationPose、Any6D、One2Any、ConceptPose、BundleSDF、CoTracker3+DepthAnything3等）在HO3D、YCBInEOAT以及自建合成集上对比，单点查询即获得ATE≈15mm、ARE≈5°，明显优于其他方法，且在无深度/掩码的场景下仍保持稳定。

**⚠️ 局限性**

受限于VAE对高速运动的模糊、对流体/纹理稀疏表面的漂移、需要数分钟GPU推理、以及假设局部刚性代理的前提。

---

## 495. CAIRN: Cross-Room 3D Scene Understanding with Topology-Aware Large Multimodal Models

**arXiv ID:** 2607.06534 | [PDF](https://arxiv.org/pdf/2607.06534v1)

**作者:** He Liang `[一作]` (University of Oxford), Yuhang He `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向多房间3D场景的拓扑感知大型语言模型CAIRN，支持跨房间语义推理。

**💡 创新点**

通过将3D场景建模为层次化场景图，使用结构化遮罩注意力和几何偏置将信息流与房间拓扑对齐。

**🔧 技术方法**

利用图神经网络编码对象关系，学习房间代词，结合层次化遮罩注意力和几何偏置的Transformer。

**📊 数据集**

在新构建的CAIRN-MR基准（基于HM3D）以及ScanRefer、Scan2Cap等单房间数据集上评测。

**📈 对比分析**

相较于传统单房间3D-LLM，CAIRN在CAIRN-MR多房间任务上提升了+5.4 Acc@0.25、+14.9 CIDEr，跨房间推理任务提升6.7 EM；单房间任务也保持或略优。

**⚠️ 局限性**

局限在于模型对房间拓扑的假设仍有限，且对更复杂层级或动态环境的适应性待提升。

---

## 496. Point as Skeleton: Accumulated Point Cloud Enhanced Autoregressive Generation for Closed-Loop Autonomous Driving Simulation

**arXiv ID:** 2607.06516 | [PDF](https://arxiv.org/pdf/2607.06516v1)

**作者:** Songbur Wong `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于点云骨架的自回归生成模拟器，能够在闭环驾驶仿真中实时生成与车辆姿态同步的多视角视觉观测；

**💡 创新点**

创新在于将离线LiDAR点云拆分为背景与轨迹索引的前景骨架，并采用“Reset‑and‑Roll”滚动扩散推理策略，既利用未来布局的引导，又避免未来潜在状态的累积误差；

**🔧 技术方法**

核心技术包括自回归扩散模型（SD1.5/SD3.5）、点云骨架条件化（彩色点投影+模板深度）、滚动扩散推理、nuPlan‑SimGen插件实现闭环接口；

**📊 数据集**

在nuScenes（700/150场景）与nuPlan‑mini（64序列）上进行训练与评估；

**📈 对比分析**

与Panacea、MagicDrive、DriveArena、FreeVS、MagicDiT、Epona等方法比较，FID、FVD以及Mask2Former车辆分割IoU均表现更优（如FID降至5.97、FVD降至58.3、IoU提升至58.57%）；

**⚠️ 局限性**

仍受限于点云投影的稀疏性与视角不一致导致的训练-推理误差，且在极端交通场景与长时序渲染中可能出现光照、遮挡等细节失真。

---

## 497. Constrained Capacity Analysis for Faster-than-Nyquist Signaling

**arXiv ID:** 2607.06496 | [PDF](https://arxiv.org/pdf/2607.06496v1)

**作者:** Zichao Zhang `[一作]` (Carleton University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在有限符号集约束下，使用DFT预编码实现的更快于奈奎斯特（FTN）调制的受限容量，并探讨了时间加速、误匹配和自适应比特分配对系统性能的影响。

**💡 创新点**

创新点包括：① 在FTN信道中引入循环前缀与后缀，将矩阵变为循环矩阵并利用DFT将通道分解为并行特征通道；② 推导了受限容量的解析表达式，并给出了在固定发送SNR与固定接收SNR条件下的渐近行为；③ 分析了信道匹配误差导致的可实现信息速率（AIR），并给出了误匹配的定量评价；④ 设计了基于特征通道质量的自适应比特装载策略，显著提升了系统容量。

**🔧 技术方法**

核心技术包括：DFT预编码与循环前缀/后缀的组合、傅里叶域特征通道分解、受限容量与可实现信息速率的统计信息理论推导、残余互通道干扰建模、Monte Carlo 计算、以及自适应比特装载算法。

**📊 数据集**

实验使用仿真数据，采用根升余弦（RRC）脉冲（滚降因子β=0.25）以及16QAM、QPSK、64QAM、256QAM 等有限符号集，在不同加速因子δ、不同块长N以及固定发送/接收SNR情况下进行容量与误匹配AIR评估。

**📈 对比分析**

通过与理想奈奎斯特调制、Gaussian输入容量以及基于SNR的固定调制策略进行对比，结果表明：① 在固定发送SNR下，FTN能在低至中等SNR区间实现与更高阶调制相当的容量；② 在固定接收SNR下，降低δ可显著提升有效SNR，进一步逼近有限符号集的容量上限；③ 自适应比特装载在匹配与误匹配两种模型下均能显著提升比特率，甚至超过任何单一固定调制方案；④ 随着块长增大，误匹配导致的容量损失迅速收敛。

**⚠️ 局限性**

主要局限包括：① 仅考虑无频率选择性 AWGN 信道，未覆盖多径/衰落等实际信道；② 误匹配分析基于DFT预编码与缺失循环前缀/后缀的假设，实际系统中对时延扩展与同步误差仍是挑战；③ 受限容量与AIR分析采用Monte Carlo 近似，理论闭式解仍缺乏；④ 自适应比特装载需要准确的特征通道估计，若估计误差大，性能可能受损。

---

## 498. AirflowAttack: Thermal-Airflow Adversarial Perturbations against Infrared Remote-Sensing Vision-Language Models

**arXiv ID:** 2607.06485 | [PDF](https://arxiv.org/pdf/2607.06485v1)

**作者:** Cong Su `[一作]` (Tianjin University), Jiahuan Long `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了AirflowAttack，一种基于热空气流动的无输入特定对抗扰动，用于攻击红外遥感视觉-语言模型。

**💡 创新点**

创新点在于首次将热空气湍流模拟为物理可实现的对抗扰动，并通过轻量级生成器实现跨模型、跨任务的迁移攻击。

**🔧 技术方法**

使用了低维潜向量的卷积生成器、对抗损失与空气相关性损失的联合优化，以及对热图像的L∞约束。

**📊 数据集**

数据集包括从NWPU-Caption、RSICD、RSITMD、RS5M和SkyScript筛选的约10000张红外图像与文本配对，验证集416张，VLM测试集1000张。

**📈 对比分析**

与四种基于热物理现象的对抗基线相比，AirflowAttack在五个CLIP骨干模型上平均攻击成功率达48.5%，比基线高近15%；在六个VLM上可将场景分类准确率下降多达38%，并在某些模型中误导其对热线索的信心。

**⚠️ 局限性**

局限性包括对比度提升依赖于物理模拟的准确性、对抗扰动在真实传感器中实现的可行性尚未验证，以及对物体检测等细粒度任务的影响相对有限。

---

## 499. Rethinking Indic AI from a Lens of Cultural Heritage Preservation

**arXiv ID:** 2607.06544 | [PDF](https://arxiv.org/pdf/2607.06544v1)

**作者:** Aparna Madva `[一作]` (International Institute of Information Technology), Tulika Saha `[通讯]` (International Institute of Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述印度语言 NLP 发展的历史与现状，并提出“文化感知（Culture Sensing）”框架，旨在通过多模态社区知识（口语、文本、音频）实现模型的文化多样性与公平表现。

**💡 创新点**

创新点：①引入文化感知概念，将社会世界观与 NLP 训练紧密结合；②将 ASR、语义检索（RAG）与生成模型联用，构建保留原生口语世界观的检索‑生成闭环；③在综述中统一阐述从规则、统计到深度学习再到大规模基础模型的演进，突出了低资源语言与方言的可持续策略。

**🔧 技术方法**

技术栈：规则‑基、统计‑基、深度‑学习（FastText、BERT、ALBERT、MuRIL、IndicBERT、XLM‑R、mBERT、MuRIL‑T、BharatGen、Sarvam‑AI、Bhashini API）；ASR 采用 wav2vec、Vakyansh、IndicSUPERB、VAANI；多模态检索使用 RAG、RLHF；词表与分词优化采用 BPE/Unigram、语言特定 BPE；对话式生成使用 mT5、ChatGPT‑like 微调。

**📊 数据集**

主要数据集：IndicCorp / IndicCorp‑v2、BPCC、IndicTrans2、Bhashini 公开平行语料、VAANI 口语语料、Graama Kannada、Parichaya、Saaras‑V3、Saaras‑V3、IndicWav2Vec 预训练语料、印度语言语音与文本混合 corpora、各方言/低资源语言（Awadhi、Bhojpuri、Magahi 等）语料、社会文化故事集、社区电台音频。

**📈 对比分析**

评估与性能：在 IndicGLUE、XTREME、IndicXParaphrase、FLORES、XQuAD、TyDiQA、IndicQA、CVIT‑MannKiBaat、IndiQA 等基准上，MuRIL 与 IndicBERT v2 在结构化任务（NER、QA、NLI）中常显著领先；ASR 模型在 9 种印地语口语任务上超过 Whisper；但低资源语言（如 Kannada 方言、Awadhi 等）仍呈现较大性能落差；总体来看，模型在文本生成与多模态检索方面已可满足部分文化感知需求，但仍有提升空间。

**⚠️ 局限性**

局限性：①低资源语言与方言数据匮乏，导致模型内部偏向英语或主流方言；②文化感知框架缺乏系统化验证与人类专家评估，仍易出现文化误读或同化；③大规模基础模型训练成本高，且在多语言、跨方言的迁移学习中出现“内部 pivot”问题；④ASR 与检索模型在嘈杂或非标准口音环境下仍易出错；⑤数据采集与隐私合规需严格把控，尤其是社区电台与口语音频；⑥现有评估指标不足以完整捕捉文化多样性与公平性。

---

## 500. From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models

**arXiv ID:** 2607.06553 | [PDF](https://arxiv.org/pdf/2607.06553v1)

**作者:** Zanyi Wang `[一作]` (University Of California San Diego), Pengtao Xie `[通讯]` (University Of California San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了ReChannel框架，将预训练文本到图像模型的token空间直接读出任务原生像素域，去除了目标侧VAE解码；

**💡 创新点**

创新点在于把生成模型的输出接口从图像重建改为token局部线性读出，证明仅凭冻结的生成器、LoRA适配与token读取即可实现多任务稠密预测；

**🔧 技术方法**

使用的技术包括冻结的DiT（FLUX-Klein）与其VAE编码器、轻量LoRA适配器以及token局部线性投影头；

**📊 数据集**

在6个稠密预测任务上使用NYU、KITTI、ScanNet、P3M-500、RefCOCO、COCO、DUTS-TE、ECSSD等多种基准数据集；

**📈 对比分析**

与现有生成式和专用解码器方法对比，ReChannel在trimap-free matting、KITTI depth、RefCOCO等任务上达到了或超过SOTA，并且速度比编辑+解码方案快2.48倍；

**⚠️ 局限性**

局限性在于目前仅支持像素对齐的稠密目标，未验证对非像素化任务或跨域通用性的适用性。

---

## 501. Unsupervised Domain Adaptation for Calcification Classification in Mammography Across Multi-Site Datasets

**arXiv ID:** 2607.06549 | [PDF](https://arxiv.org/pdf/2607.06549v1)

**作者:** Xuan Liu `[一作]` (Duke University), Lars J. Grimm `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一个无监督域自适应框架，用于多站点乳腺X线钙化的良恶性分类。

**💡 创新点**

通过风格迁移（AdaIN和CycleGAN）生成不同供应商与技术的训练样本，实现无需额外标注的域迁移。

**🔧 技术方法**

采用AdaIN、CycleGAN进行风格迁移，Swin Transformer V2作为分类骨干网络。

**📊 数据集**

在英国OPTIMAM公开数据集、美国EMBED公开子集及Duke私有数据集上训练验证。

**📈 对比分析**

对比单站点基线后，外部AUC从0.68提升至0.72（EMBED）和0.73（Duke），敏感度在相同特异度下提升。

**⚠️ 局限性**

主要限制为风格迁移方法相对较旧、对不同供应商的覆盖有限，且模型训练仍依赖单一数据集导致潜在偏差。

---

## 502. On the feasibility of dependency parsing of non-human sequences without a gold standard. Is evaluation possible in other species?

**arXiv ID:** 2607.06542 | [PDF](https://arxiv.org/pdf/2607.06542v1)

**作者:** Ramon Ferrer-i-Cancho `[一作]` (Universitat Politècnica de Catalunya), Morgan Gustison `[通讯]` (Western University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在无黄金标准的非人类物种中，利用网络科学理论评估无监督依存句法解析的可行性；提出基于序列长度分布的随机解析器性能下界，说明在黑猩猩和大猩猩等非人类灵长类中可实现高准确度的解析；并讨论训练无监督解析器的可行性与方法。

**💡 创新点**

创新点在于：①将序列长度分布的指数衰减特性与随机解析器的期望准确度联系，推导出无黄金标准下的性能下界；②证明非人类灵长类序列长度分布使随机解析器表现优于人类语言，从而为非人类语言的解析提供理论支持；③提出训练和评估无监督解析器的具体步骤与指标。

**🔧 技术方法**

使用了无监督依存解析理论、随机树生成模型、期望值推导与网络科学中的树结构分析；计算随机解析器在不同长度分布（均匀、几何、经验）下的期望正确树与边的比例。

**📊 数据集**

数据集包括：①31种灵长类动物的最大序列长度信息（来源于文献）；②黑猩猩与羚羊的声学与手势序列长度分布；③人类语言的PUD10（并行通用依存树库）句子长度分布；以及与WSJ10的对照。

**📈 对比分析**

比较方法：通过对随机解析器在不同长度分布下的期望准确度（[P_c^t],[Q],[P_c^e]）与人类语言的对应值进行对比；结果显示在黑猩猩/羚羊序列中随机解析器的平均正确边比例可达50-80%，而人类句子平均仅约13-28%；说明非人类序列更易评估。

**⚠️ 局限性**

局限性包括：①仅考虑了无向边的评估，忽略了依存方向与根结构；②假设所有正确树等概率，实际语料可能更复杂；③缺乏实际无监督解析器实验验证，仅提供理论下界；④数据集规模有限，尤其是非人类语料样本不足；⑤未讨论不同动物的语义与功能层面的差异对解析的影响。

---

## 503. The Large Cancer Assistant (LCA): A Model-Agnostic Orchestration Framework for Scalable Clinical Decision Support in Oncology

**arXiv ID:** 2607.06531 | [PDF](https://arxiv.org/pdf/2607.06531v1)

**作者:** Ghassen Marrakchi `[一作]`, Basarab Matei `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个模型无关的“大癌症助手”（LCA）框架，用于异步统一多模态肿瘤学数据的采集、路由与 AI 推理；

**💡 创新点**

核心创新包括算法不可渗透原则、基于几何深度学习的Entry Theory、标准化中间负载（SIP）以及双模式路由（确定性V1与潜在概率性V2）和错误安全的补充数据请求机制；

**🔧 技术方法**

技术手段涵盖几何深度学习、DAG式单向管线、模块化接口设计、JSON SIP 输出、自然语言生成以及基于规则的治愈模块；

**📊 数据集**

实验使用合成与真实肺癌病历数据（单肺协议及模拟协议）进行 PoC 验证；

**📈 对比分析**

通过四种情景对比评估：1）完整流 100% 完成；2）模型替换保持路由不变 100%；3）错误安全 100% SDR 召回；4）多协议独立 100%；总体计算开销约 0.1 ms；

**⚠️ 局限性**

局限性包括仅验证确定性路由 V1、未验证概率路由 V2；使用模型占位符与规则桩，未测量真实推理准确度；仅单肺协议与模拟协议，缺乏多病种广泛验证；NLG 仅英文；缺乏临床真实部署验证。

---

## 504. RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation

**arXiv ID:** 2607.06527 | [PDF](https://arxiv.org/pdf/2607.06527v1)

**作者:** Sambaran Bandyopadhyay `[一作]` (Adobe Research), Ananth Muppidi `[通讯]` (Adobe Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RSF‑GLLM 框架，先用可微分图推理模块提取可解释的推理路径，再将路径文本化后交给 LLM 生成答案，从而解决多跳 KGQA 中的语义鸿沟和生成幻觉问题。

**💡 创新点**

创新点包括：① Recurrent Soft‑Flow（RSF）模块，引入动态关系注意力与门控机制，能在桥节点无词汇重叠时依赖图结构进行推理；② 通过流稀疏正则把连续概率转为稀疏分布，实现可解释的单路径推理；③ 通过贪心回溯提取路径并文本化，解耦 LLM 生成，避免多次 LLM 调用的高成本。

**🔧 技术方法**

技术手段：可微分图神经网络 + GRU 查询更新 + 动态门控 + 流稀疏正则 + 贪心回溯路径提取 + 预训练 LLM 微调 + 二阶段（先训练 RSF，再微调 LLM）训练策略。

**📊 数据集**

使用 WebQSP（约 4.7k 题）和 CWQ（多跳复杂问答）两个公开 KGQA 数据集进行评估。

**📈 对比分析**

与检索+LLM、Agentic 搜索、GNN‑RAG 等基线相比，在 WebQSP 上 Hit@1 90.45% / F1 79.15%，在 CWQ 上 Hit@1 67.39% / F1 61.87%；相比大模型基线实现约 21× 的推理速度提升，仅需一次 LLM 调用，且在路径提取精度上也优于现有方法。

**⚠️ 局限性**

局限性：① 依赖预先抽取的 K‑hop 子图，导致召回受限；② 对极大 KG 的实时查询需动态扩展；③ LLM 生成仍可能受到热门偏差；④ 目前仅针对结构化 KG，尚未验证时序或多模态 KG 的适用性。

---

## 505. Industry Classification of GitHub Repositories Using the North American Industry Classification System (NAICS)

**arXiv ID:** 2607.06505 | [PDF](https://arxiv.org/pdf/2607.06505v1)

**作者:** Kevin Xu `[一作]` (GitHub), Alexander Quispe `[通讯]` (GitHub)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了NAICS-GH数据集，包含6,588个GitHub仓库的NAICS 2位行业标签；

**💡 创新点**

采用检索-验证两阶段管道，将检索结果与GPT-4.1评分结合，提供可解释的置信分数，并实现跨地区无国别标签；

**🔧 技术方法**

BBAI/bge-large-en嵌入+FAISS检索，GPT-4.1 LLM评分，结构化提示、阈值筛选；

**📊 数据集**

源自GitHub仓库（美国、欧盟、澳大利亚）共1,372,489个公共仓库；

**📈 对比分析**

在发布的6,588条训练集上微调六种预训练编码器，RoBERTa‑large在20类NAICS上取得86.45% F1（最高），与其他模型对比说明编码器家族影响大于参数规模；

**⚠️ 局限性**

主要局限：仅覆盖英语仓库，NAICS在非北美语境适用性有限，部分行业（制造业、批发业）标签精度低于其他行业，未评估召回率

---

## 506. Clustering-Embedded Model Predictive Path Integral Control: Avoiding Averaging-Induced Failure and Enabling Efficient Cluster Selection for Dynamic Obstacles

**arXiv ID:** 2607.06499 | [PDF](https://arxiv.org/pdf/2607.06499v1)

**作者:** Zidong Liu `[一作]` (University of Washington), Xu Chen `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于聚类嵌入的MPPI（CE-MPPI）框架，通过碰撞剔除、DBSCAN聚类和障碍感知的聚类选择，解决传统MPPI在复杂环境中的平均化失败问题。

**💡 创新点**

引入基于碰撞参考点的几何方向特征进行DBSCAN聚类，以及针对动态障碍的“相反运动方向”聚类选择，显著避免了平均化冲突并实现主动避障。

**🔧 技术方法**

使用GPU并行模拟、JAX JIT编译、DBSCAN密度聚类、路径积分加权更新、障碍运动估计以及在Isaac Gym上对UR5e机械臂的CUDA并行rollout。

**📊 数据集**

在2‑D差速驱动平面机器人仿真中使用自定义静态/动态障碍场景；在真实实验中使用UR5e机械臂与桌面黑盒和方块目标的物理场景。

**📈 对比分析**

与标准MPPI和CSC‑MPPI在两种仿真场景及真实UR5e实验中对比，CE‑MPPI在静态场景中时间缩短至4.02 s、路径为2.22 m；动态场景中时间7.98 s、路径4.30 m；真实机器人中时间93.11 s、路径0.73 m，均显著优于基线。

**⚠️ 局限性**

需要有效的碰撞检测与参考点估计，对高维状态空间的DBSCAN参数调优敏感；在极其复杂或快速动态环境下仍可能出现误聚类或计算超时。

---

## 507. Assessing the Operational Impact of Poisoning Attacks over Augmented 3D Point Cloud Public Datasets for Connected and Autonomous Vehicles

**arXiv ID:** 2607.06484 | [PDF](https://arxiv.org/pdf/2607.06484v1)

**作者:** Marwan Lazrag `[一作]` (SAMOVAR, Télécom SudParis, Institut Polytechnique de Paris), Joaquin Garcia-Alfaro `[通讯]` (SAMOVAR, Télécom SudParis, Institut Polytechnique de Paris)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究了在连接与自动驾驶车辆（CAV）场景中，利用GAN进行3D点云数据增强后，公开数据集被恶意中毒攻击所产生的运营影响。

**💡 创新点**

创新点在于首次量化了数据增强对3D点云中毒攻击的放大效应，并提出基于攻击成功率（ASR）的运营影响传播模型，揭示增强过程可能导致决策层面安全风险显著上升。

**🔧 技术方法**

主要技术包括：3D-GAN进行点云数据增强；InceptionNet二分类器；点丢弃式中毒手段；MCC、F1和ASR指标评估；以及资产-功能-流程的依赖图模型进行运营影响传播。

**📊 数据集**

使用的数据集为公开的ModelNet三维CAD模型（用于实验），并通过GAN生成的合成点云进行增强；nuScenes数据集用于构建CAV系统的资产与功能依赖图。

**📈 对比分析**

通过对比基线（无增强）和增强两种训练方案，在不同中毒率下训练并测试分类器，结果显示在20%–40%中毒率下，增强方案的MCC/F1显著下降，ASR显著提升（最高至17.6%），证明增强会放大攻击效果。

**⚠️ 局限性**

局限性包括：仅在ModelNet单一二分类任务上验证；中毒方式为简单点丢弃，未涵盖更复杂的后门或噪声注入；未探讨其它增强技术或其他3D点云数据集的适用性；实际CAV部署环境的可迁移性仍需进一步研究。

---

## 508. DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation

**arXiv ID:** 2607.06507 | [PDF](https://arxiv.org/pdf/2607.06507v1)

**作者:** Yaqi Wu `[一作]` (Shanghai Jiao Tong University), Dongdong Ge `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于状态的控制框架，将多跳检索增强生成中的各种操作统一为原子证据操作，并通过学习的控制器在每一步根据当前证据状态动态选择执行的操作，从而实现闭环式的证据采集流程。

**💡 创新点**

核心创新包括：① 将操作的合法性（hard validity）与作用价值（learned utility）分离，先用规则筛选可执行操作，再用学习的价值模型对合法操作进行排名；② 将多跳RAG的不同行为（前沿检索、查询重写、桥接实体扩展、缺口检索、充分性判断、终止）统一为原子动作；③ 通过终端证据压缩（answer-focused compression）提升生成效率；④ 通过迁移学习使控制策略跨模型、跨数据集有效。

**🔧 技术方法**

技术手段包括：随机森林回归模型用于估计状态-动作价值，继续模型决定是否继续；BGE-large-en-v1.5稠密检索；多跳检索增强生成框架；Qwen2.5‑7B‑Instruct、GPT‑4o‑mini、Llama‑3.1‑8B‑Instruct等大型语言模型作为答案生成器；以及在训练时利用支持文档召回来监督价值模型。

**📊 数据集**

在 HotpotQA、2WikiMultiHopQA（2Wiki）和 MuSiQue 三个公开多跳问答基准上进行评估。

**📈 对比分析**

与固定检索深度 RAG、IRCoT、S2G‑RAG、CoRAG、Adaptive‑RAG、PAR²‑RAG、CRAG、Self‑Ask+Search 等多种基线进行对比。实验表明在所有三个数据集上都取得最高 F1 分数，HotpotQA 提升 2.88 点、2Wiki 提升 7.19 点、MuSiQue 提升 0.62 点；同时相较于最强迭代基线，token 使用量下降 13–20%，显示出更高的效率。消融实验进一步验证了学习控制器、充分性反馈和终端压缩对性能的关键作用。

**⚠️ 局限性**

局限性包括：① 仅在少数几种大型语言模型上验证，可能对不同模型的迁移性有一定依赖；② 控制器训练依赖于 Qwen‑generated 轨迹，虽然迁移效果好，但在极端检索质量或域外任务中的表现未知；③ 采用的原子动作集合有限，未包含如自适应检索策略等更细粒度操作；④ 仍需额外的检索和语言模型调用，导致一定的推理延迟和成本。

---

