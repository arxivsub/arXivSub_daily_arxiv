# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-18 | 今日论文总数: 518

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Veriphi: Attack-Guided Neural Network Verification with Dataset-Dependent Training Methods

**arXiv ID:** 2606.18454 | [PDF](https://arxiv.org/pdf/2606.18454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 2. Artemis: Anatomy-Resolved inTervention for Eliminating Multimodal NeuroImage confounderS

**arXiv ID:** 2606.18287 | [PDF](https://arxiv.org/pdf/2606.18287v1)

**作者:** Siyuan Dai `[一作]`, Liang Zhan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个可插拔的区域级因果干预模块Artemis，用来消除多模态脑网络图神经网络中因年龄、性别等人口学因素导致的混淆，提升模型的因果鲁棒性。

**💡 创新点**

核心创新在于将观察到的人口学变量映射为每个脑区特定的混淆嵌入（共享MLP+可学习ROI token），使用EMA记忆银行近似人口分布做backdoor调整，并通过门控函数在功能与结构特征上做精细调节，整个模块参数量极小且可兼容任意GNN骨干。

**🔧 技术方法**

采用共享多层感知机+ROI token编码人群混淆、指数移动平均记忆银行实现人口期望近似、门控(sigmoid)对功能/结构特征做调节，以及标准图神经网络（GCN等）作为后端。

**📊 数据集**

实验数据集包括ADNI（正常对比轻度认知障碍的疾病诊断）、HCP（性别分类）以及OASIS（三分类认知功能分级）。

**📈 对比分析**

与十个代表性基线（传统GNN、脑网络专用网络、因果GNN）在5折交叉验证下对比，Artemis在准确率、宏观F1和AUC上均实现显著提升，尤其在受样本不平衡影响的指标上提升尤为突出。

**⚠️ 局限性**

局限性包括仅针对已观测的人口学混淆（如年龄、性别、教育），未考虑未观测混淆或纵向时间变化，以及仅在横断面数据上验证，未来需扩展到隐式混淆、回归任务及多中心数据。

---

## 3. CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework

**arXiv ID:** 2606.18385 | [PDF](https://arxiv.org/pdf/2606.18385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 4. Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting

**arXiv ID:** 2606.18367 | [PDF](https://arxiv.org/pdf/2606.18367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 5. MagpieTTS-LF: Inference-Time Long-Form Speech Generation Without Training on Long-Form data

**arXiv ID:** 2606.18485 | [PDF](https://arxiv.org/pdf/2606.18485v1)

**作者:** Subhankar Ghosh `[一作]` (NVIDIA Corporation), Roy Fejgin `[通讯]` (NVIDIA Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 MagpieTTS 进行无训练的推理时长文本合成

**💡 创新点**

软注意力先验、状态化分块推理以及历史文本编码三项创新

**🔧 技术方法**

软注意力先验、状态化推理算法与历史文本编码技术

**📊 数据集**

Long‑Form HifiTTS 数据集（MLS 章节合并的 3‑4 分钟长文本）

**📈 对比分析**

与 Qwen3‑TTS、VibeVoice、XTTS 比较，WER/CER 低、SSIM 高、PBD 低、UTMOS 稳定且优于其它系统

**⚠️ 局限性**

依赖原始模型架构，未针对极长句子或低资源语言做专门训练，适用范围受限

---

## 6. Attribution-Guided and Coverage-Maximized Pruning for Structural MoE Compression

**arXiv ID:** 2606.18304 | [PDF](https://arxiv.org/pdf/2606.18304v1)

**作者:** Yifu Ding `[一作]` (Beihang University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 102021 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向Mixture-of-Experts（MoE）模型的结构化剪枝框架——基于归因的覆盖最大化专家级剪枝；

**💡 创新点**

创新点在于①观察到MoE专家内部信息高度集中在少数通道；②提出通道得分覆盖率作为剪枝目标，取代粗粒度的专家级重要性；③通过归因指导的损失近似快速估算专家贡献；④采用对齐感知重新分配保证低位量化兼容；

**🔧 技术方法**

技术手段包括归因指导的损失近似（ALA）、覆盖最大化预算分配（CBA）、对齐感知重新分配（AAR）以及4‑bit量化；

**📊 数据集**

实验数据集涵盖DeepSeek-MoE-16B、DeepSeek-V2-Lite、Qwen1.5-MoE-A2.7B、Qwen3-30B-A3B，并在知识推理（GSM8K、OpenBookQA）、数学推理（MATH500）、代码推理（HumanEval）等基准上评估；

**📈 对比分析**

与Wanda、MoNE、EAC-MoE、MoE‑I²、PuzzleMoE等基线比较，1/25%通道剪枝+4bit量化可实现5×以上存储压缩，零样本知识推理准确率不低于原模型；在50%剪枝下，Qwen3-30B-A3B在MATH500上达95.0%的通过率，整体表现优于专家级剪枝或统一比例剪枝方法；

**⚠️ 局限性**

局限性包括：对归因近似的假设（一阶泰勒展开）在极端剪枝或不同模型结构下可能不稳健；对齐重分配需要预设硬件块尺寸，可能导致通道裁剪不够细粒度；在低位量化下推理吞吐量仍受限，需进一步优化硬件实现。

---

## 7. What Does the Weight Norm Control in Grokking? Logit-Scale Mediation under Cross-Entropy

**arXiv ID:** 2606.18465 | [PDF](https://arxiv.org/pdf/2606.18465v1)

**作者:** Truong Xuan Khanh `[一作]` `[通讯]` (Clevix Llc), Truong Xuan Khanh (Clevix Llc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

对模数加法任务的两层MLP在全批训练中研究了权重范数对grokking延迟的影响，并通过固定范数加温度干预将范数与logit尺度分离。

**💡 创新点**

创新在于用非可训练的输出温度隔离了范数与logit尺度的交互，证明在交叉熵下延迟主要由logit尺度（softmax饱和）决定，而非纯粹的权重范数；同时揭示了该机制对损失函数依赖性。

**🔧 技术方法**

使用了温度调节实验、数据折叠回归、同一状态分支测试、浮点精度审计等技术。

**📊 数据集**

实验数据来自全模数加法任务的多模数（p=59,97）训练集，包含所有p^2样本，采用12个种子。

**📈 对比分析**

通过与固定范数的指数剂量反应对照、温度介导恢复率、数据折叠R^2=0.97等指标，验证了logit尺度对延迟的解释力度；在交叉熵下恢复率约为85%，在MSE下机制不显著。

**⚠️ 局限性**

局限在于仅验证了ℓ2权重衰减下的两层MLP与无层归一化Transformer，对更大规模网络、不同任务、MSE路由的机制以及更广泛的分支时间点尚未扩展。

---

## 8. Learning-Based Decision Making for Combustion Phasing Control in Multi-Fuel CI Engines with Latent Fuel Reactivity Estimation

**arXiv ID:** 2606.18393 | [PDF](https://arxiv.org/pdf/2606.18393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 9. ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets

**arXiv ID:** 2606.18338 | [PDF](https://arxiv.org/pdf/2606.18338v1)

**作者:** Edward T. Stevenson `[一作]` (University of Cambridge), Miles Cranmer `[通讯]` (University of Cambridge)

**通讯引用:** 1519 | [OpenAlex ID](https://openalex.org/A5078731429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并公开了千星球（ThousandWorlds）数据集，包含约1800个来自五种全球气候模型的三维大气模拟，用于低数据、参数到场的多模拟器机器学习基准。

**💡 创新点**

创新点：①提供多模型、低数据、结构缺失的全域气候模拟基准；②设计了两种评估协议（标准排名和相对GCM差异），量化模型在科学上的实用性；③发现高斯过程方法在该领域优于传统深度学习模型，揭示了该基准对新型算法的挑战。

**🔧 技术方法**

使用的技术包括：重网格化与预处理、坐标MLP、DeepONet、PCA+MLP、概率PCA-ICM、Gaussian Process Latent Factor Regression（GPLFR）等多种机器学习方法；此外还使用了Spherical Harmonic Transform（T21）等空间表示。

**📊 数据集**

使用的数据集是千星球（ThousandWorlds）：约1800个三维气候模拟，5个GCM（UM、ExoCAM、ExoPlaSim、LFRic、ExoCAM-pre-2022），8个连续输入参数（半径、重力、旋转周期、表压、CO₂、CH₄、入射辐射、恒星温度），输出包括53个三维字段（温度、湿度、风、云、辐射等）。

**📈 对比分析**

通过三套子集（单模拟器完整、全模拟器完整、全模拟器部分缺失）和两套评估协议（标准RMSE/ES、相对GCM差异）与七个基线模型（Train-mean、kNN、Coord-MLP、Coord-DeepONet、PCA-MLP、PPCA-ICM、GPLFR）进行比较。结果显示，GP基线（PPCA-ICM、GPLFR）在RMSE上遥遥领先；深度学习方法虽有进步，但普遍不及GP；kNN在部分变量上已接近GCM内部一致性；相对RMSE多数小于1，说明能达到或超过同一行星下不同GCM的差异。

**⚠️ 局限性**

局限性：①绝对误差仍较大，部分变量如表面温度RMSE仍达10 K；②数据仅覆盖潮汐锁定的海洋世界，缺少陆地或不同星系环境；③深度学习方法在该低数据、多缺失、跨模拟器场景下表现不佳；④模型对不同物理过程的泛化能力有限，需更多高分辨率或多物理耦合的模拟来进一步验证。

---

## 10. Human-Machine Bidirectional Trust-Aware Analysis and Design for Human-Led Truck Platooning

**arXiv ID:** 2606.18255 | [PDF](https://arxiv.org/pdf/2606.18255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 11. Gaussian Mixture Attention: Linear-Time Sequence Mixing via Probabilistic Latent Routing

**arXiv ID:** 2606.18283 | [PDF](https://arxiv.org/pdf/2606.18283v1)

**作者:** Yongchao Huang `[一作]` (University of Aberdeen), Hassan Raza `[通讯]` (University of Aberdeen)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5081606680)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Gaussian Mixture Attention (GMA)，一种通过概率责任空间路由替代传统点积注意力的线性时间序列混合器。

**💡 创新点**

创新点在于用K个高斯混合组件映射查询/键到后验责任向量，利用责任矩阵进行隐式低秩关联，避免形成 N×N 注意力矩阵，并提供可解释的概率路由；同时提出因果与双向版本及端到端可微的参数化。

**🔧 技术方法**

使用了高斯混合模型、责任矩阵、可微参数化、矩阵乘法的可结合性、因果前缀求和以及与稀疏、低秩、核化和状态空间等高效注意力方法的对比。

**📊 数据集**

实验数据集包括 Long Range Arena（ListOps 与 byte‑level IMDb/Text）以及 WikiText‑103 语言建模数据集。

**📈 对比分析**

通过与标准 MHA、Linformer、Linear Transformer、Performer、Mamba 等基线的对比，GMA 在 LRA 上与 SDPA 竞争，取得最佳平均准确率；在 WikiText‑103 上在验证困惑度上优于线性/随机特征注意力，但仍落后于优化的 causal SDPA 与 Mamba；GMA 体现了线性内存与吞吐量的优势。

**⚠️ 局限性**

主要限制包括实现常数因子高、未做硬件级优化；K 固定且需手动选择；仅评估自注意力，未测试跨注意力或多模态；解释性仅限于表面 token 级别；未实现稀疏执行或动态组件选择。

---

## 12. Beyond the Algorithm: Professional Experiences and Perceptions of AI Bias

**arXiv ID:** 2606.18289 | [PDF](https://arxiv.org/pdf/2606.18289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 13. Evaluating the Effectiveness of LLMs in Aiding Compliance Testing of PKCS#1-v1.5

**arXiv ID:** 2606.18405 | [PDF](https://arxiv.org/pdf/2606.18405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 14. "Are you an AI?" Analyzing Client Suspicion of AI Use in Crisis Counseling

**arXiv ID:** 2606.18261 | [PDF](https://arxiv.org/pdf/2606.18261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 15. VISUALSKILL: Multimodal Skills for Computer-Use Agents

**arXiv ID:** 2606.18448 | [PDF](https://arxiv.org/pdf/2606.18448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 16. Signature filtering: a lightweight enhancement for statistical watermark detection in large language models

**arXiv ID:** 2606.18430 | [PDF](https://arxiv.org/pdf/2606.18430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. Conflict-Aware Retriever Editing for Knowledge Injection Attacks on LLM-Based RAG Systems

**arXiv ID:** 2606.18310 | [PDF](https://arxiv.org/pdf/2606.18310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 18. SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents

**arXiv ID:** 2606.18356 | [PDF](https://arxiv.org/pdf/2606.18356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 19. Exploring Statistical Change Point Detection Techniques for Performance Anomaly Detection at Mozilla

**arXiv ID:** 2606.18377 | [PDF](https://arxiv.org/pdf/2606.18377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 20. A Stochastic ISCS Markov Model for Fake News Propagation

**arXiv ID:** 2606.18282 | [PDF](https://arxiv.org/pdf/2606.18282v1)

**作者:** Carles Rovira `[一作]` `[通讯]`, Carles Rovira

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本论文研究了假新闻传播的过程，提出了一种基于马尔可夫链的随机谣言传播模型，分析了事实核查者对假新闻动态的影响。

**💡 创新点**

创新点在于将事实核查者纳入谣言传播模型中，形成了Ignorant–Spreader–Checker–Stifler (ISCS)模型，提供了一个结合事实核查的随机谣言传播框架。

**🔧 技术方法**

使用了马尔可夫链技术，并结合了确定性模型和Galton-Watson分支过程的近似。

**📊 数据集**

论文中使用了数值模拟来展示系统的行为和事实核查者的影响，但未具体提及使用的数据集。

**📈 对比分析**

通过数值模拟比较了不同情况下的传播动态，结果表明，增加事实核查者的比例显著提高了不传播谣言的个体比例，且p（成为传播者的概率）与最终结果之间的关系是非线性的。

**⚠️ 局限性**

模型的局限性包括假设人群是均质的，所有个体共享相同的行为参数，且事实核查者能够正确识别错误信息。此外，模型未考虑真实社交网络的结构或外部信息源。

---

## 21. Rendering Separoid Information: Rate-Distortion Reconstruction of Convex Apartness Scenes

**arXiv ID:** 2606.18486 | [PDF](https://arxiv.org/pdf/2606.18486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 22. Depth Lower Bounds for ReLU Networks with Binary Inputs

**arXiv ID:** 2606.18540 | [PDF](https://arxiv.org/pdf/2606.18540v1)

**作者:** Neil Krishnan `[一作]`, Elchanan Mossel `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了对二进制输入、实值输出的ReLU网络的深度下界，并给出一个可在 n+1 层实现的显式函数。

**💡 创新点**

构造了所有深度层次的深度分离示例，证明任意深度为 d 的网络若要精确计算该函数，宽度必须满足 w^d = Ω(2^n)，从而深度不足时宽度会指数增长。

**🔧 技术方法**

利用了素数平方根乘积在有理数域上的线性无关性、ReLU 网络的路径计数与线性组合维度上界以及迭代乘法的对数变换。

**📊 数据集**

论文完全基于理论构造，无使用任何具体数据集。

**📈 对比分析**

通过与 AC^0 和阈值电路已知深度分离结果对比，证明在 ReLU 模型下实现了完整的深度层次分离；未给出实验性能评估。

**⚠️ 局限性**

仅适用于精确无误差的计算；对有限精度逼近时的深度优势未给出完整证明，且逼近可在多项式大小的浅层网络中实现。

---

## 23. HyDRA: Lossless Hypergraph Summarization via Co-Clustering

**arXiv ID:** 2606.18274 | [PDF](https://arxiv.org/pdf/2606.18274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 24. From Embedded Properties to Trait Nodes: A Design Method for Identifying Reusable Metadata in Property Graph Schemas

**arXiv ID:** 2606.18297 | [PDF](https://arxiv.org/pdf/2606.18297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 25. Characterizing Opinion Evolution of Networked LLMs

**arXiv ID:** 2606.18276 | [PDF](https://arxiv.org/pdf/2606.18276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 26. TopVenues: A Reproducible Corpus and Tooling Substrate for Cybersecurity Literature Reviews

**arXiv ID:** 2606.18320 | [PDF](https://arxiv.org/pdf/2606.18320v1)

**作者:** Sidnei Barbieri `[一作]` (Instituto Tecnológico de Aeronáutica), Lourenço Alves Pereira Júnior `[通讯]` (Instituto Tecnológico de Aeronáutica)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5030326285)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 TopVenues 系统，一个可版本化的安全领域文献集合构建工具，以 DBLP 为骨干，补全摘要和 BibTeX，生成可重复的语料库并提供 CLI、Web 与多种导出接口。

**💡 创新点**

把文献集构建视为可复现的研究产物，提供单文件 SQLite 版本化快照，保证增量更新时不回退已补全数据；实现多源摘要/BibTeX 补全与断路器、断点恢复；在 9,925 篇论文上实现 99.86% 摘要覆盖率和 99.99% BibTeX 覆盖率。

**🔧 技术方法**

Python 3.14、SQLite 单文件数据库、DBLP API、Semantic Scholar、OpenAlex、CrossRef、ACM DL、IEEE Xplore、USENIX、NDSS 等多源抽取器、Streamlit Web UI、CLI、monotonic upsert、断点恢复、gzip 压缩快照等技术。

**📊 数据集**

涵盖 11 个顶级安全会议与期刊（USENIX Security、ACM CCS、IEEE S&P、NDSS、ACM ASIA CCS、IEEE EuroS&P、ACM SACMAT、HotNets、ACM Computing Surveys、IEEE Communications Surveys & Tutorials、Foundations and Trends in Privacy and Security），时间跨度 2017-2026 年，共 9,925 篇论文。

**📈 对比分析**

与直接 DBLP 查询对比，提升 99.86% 摘要覆盖率；关键词搜索平均 25.8 ms；BibTeX 导出 2.2 ms；单文件快照 15 MB，解压 <1 s；250 条测试验证数据完整性，monotonic 更新保证不丢失。

**⚠️ 局限性**

缺失摘要/ BibTeX 的论文仍有 14 篇，部分来源受限；依赖外部 API 可能导致速率限制或停机；仅覆盖选定 11 会议/期刊，未覆盖所有安全领域文献；抽取器适配器需手动维护。

---

## 27. As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agriculture

**arXiv ID:** 2606.18519 | [PDF](https://arxiv.org/pdf/2606.18519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 28. Dynamic In-Group Persona Generation for Enhancing Human-AI Rapport

**arXiv ID:** 2606.18256 | [PDF](https://arxiv.org/pdf/2606.18256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 29. Budget-Aware Adaptive Adversarial Patches for Black-Box Object Detection

**arXiv ID:** 2606.18318 | [PDF](https://arxiv.org/pdf/2606.18318v1)

**作者:** Pedram MohajerAnsari `[一作]` (Clemson University), Mert D. Pesé `[通讯]` (Clemson University)

**通讯引用:** 254 | [OpenAlex ID](https://openalex.org/A5085340429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于查询效率的黑盒对抗补丁攻击，能够联合优化补丁位置、纹理和尺寸，并在有限查询预算内实现对目标检测器的严格抑制。

**💡 创新点**

创新点包括：①将轻量化的情境Thompson采样用于补丁放置搜索；②采用NES（自然进化策略）进行无梯度的像素更新；③设置进度触发的自适应尺寸梯度，使补丁仅在必要时扩展；④严格使用纯图像抑制标准并单独记录EOT鲁棒性，避免传统评估混淆。

**🔧 技术方法**

使用技术主要有：情境Thompson采样（上下文带式采样）、NES零阶优化、进度触发的自适应尺寸增长、可选的EOT训练和可见性/可打印性约束；实现中使用了PyTorch等深度学习框架。

**📊 数据集**

在COCO预训练的目标检测模型上评估：YOLOv5、Faster R‑CNN和YOLOS；此外通过打印-捕获实验验证了物理环境下的迁移性。

**📈 对比分析**

与现有的白盒统一补丁、基于GAN搜索的黑盒攻击以及固定尺寸补丁等对比，所提方法在查询次数（如YOLOv5仅约50次）、补丁面积（约8%）和严格抑制率（YOLOv5 77.5%，Faster R‑CNN 89.7%，YOLOS 59.1%）方面均取得显著优势，且在不同检测器架构中均保持高效。

**⚠️ 局限性**

局限性：对Transformer‑based检测器的抑制效果仍低于CNN模型；补丁在物理环境中对不同视角、距离的鲁棒性尚需进一步提升；对抗补丁的可见性与打印可行性之间仍存在权衡；方法依赖于严格的score‑only接口，若检测器提供更丰富的反馈可能需调整。

---

## 30. Examining Human-Like Behaviors in LLMs: A Multi-Dimensional Analysis of Model Behaviors, User Factors, and System Prompts

**arXiv ID:** 2606.18258 | [PDF](https://arxiv.org/pdf/2606.18258v1)

**作者:** Sunnie S. Y. Kim `[一作]` (Apple), Leon A Gatys `[通讯]` (Apple)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过多维度评估框架，系统研究了大型语言模型在多轮对话中表现出的类人行为，并探讨了对话目标、用户画像以及系统提示对这些行为的影响与可控性。

**💡 创新点**

创新点在于①将行为分类、LLM‑as‑judge与人工评估相结合，构建可验证的多维度评估流程；②首次在数千条真实对话中量化类人行为的出现率、适宜性与潜在影响；③通过对比手工与自动优化（GEPA）系统提示，揭示系统提示对类人行为的细粒度调控效果。

**🔧 技术方法**

主要技术包括：多轮对话生成（使用用户模拟LLM与目标LLM）；LLM‑as‑judge行为检测（利用多模型投票集成）；人工评估（行为出现、适宜性、帮助度与用户影响）；系统提示工程（手工与GEPA优化）；统计分析（Bootstrap、OLS回归）。

**📊 数据集**

数据集为21,000条多轮对话，来自四款主流LLM（gpt‑4o、gpt‑4.1‑mini、claude‑sonnet‑4.6、gemini‑2.5‑flash），由1,050个不同来源的输入提示（50提示×3来源×7对话目标）与5种用户画像生成。

**📈 对比分析**

比较方法：①对四个模型进行类人行为频率、用户适宜性及帮助度的量化比较；②对手工与优化系统提示在“避免”和“保留”两组行为上的效果进行统计检验。性能结果显示：gpt‑4o表现最为频繁；自我指涉与关系建立行为在角色扮演与浪漫场景中显著升高；边界维持行为在大多数场景中更被视为适宜；优化提示在降低不受欢迎行为（如自我指涉）方面优于手工提示。

**⚠️ 局限性**

局限性包括：①行为分类不全面，部分细微行为如同理与赞同易重叠；②使用模拟用户，缺乏真实人机交互验证；③仅评估四款模型，未覆盖更广泛模型或更新版；④人工评估样本规模有限且来自单一公司，缺乏多样性。

---

## 31. Ghost Attractor Networks: Basin-Structured Dynamical Decoders for Closed-Loop Sequential Generation

**arXiv ID:** 2606.18315 | [PDF](https://arxiv.org/pdf/2606.18315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. A Critical Discourse Analysis of Gender Representation in Software Engineering Education Videos on YouTube

**arXiv ID:** 2606.18423 | [PDF](https://arxiv.org/pdf/2606.18423v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 33. DRIFT: Refining Instruction Data via On-Policy Data Attribution

**arXiv ID:** 2606.18307 | [PDF](https://arxiv.org/pdf/2606.18307v1)

**作者:** Zefan Wang `[一作]` (Tsinghua University), Yuan Yao `[通讯]` (Tsinghua University)

**通讯引用:** 482000 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在已训练好的大语言模型上，通过对原始SFT数据集进行实例级影响力归因，并结合模型自身的on‑policy rollouts，重新调节数据分布以提升模型能力。

**💡 创新点**

创新点包括：①用on‑policy生成的响应作为验证目标，显著减少“邻域距离”问题；②通过给正确与错误轨迹分别赋正负权重，将IF转化为奖励加权对比的策略梯度形式；③针对梯度范数偏置引入对数空间正交化去偏，使归因结果更能反映真实影响而非梯度大小。

**🔧 技术方法**

主要技术包括：Influence Functions（IF）近似、稀疏随机投影、对数空间正交化梯度去偏、对齐的policy‑gradient损失、持续SFT（top‑10%样本挑选）。

**📊 数据集**

实验使用两款7B参数模型（Olmo3‑7B‑Instruct‑SFT 与 OpenR1‑Distill‑7B）及其原始SFT数据集，验证集覆盖 Minerva MATH、MBPP+、BBH、MMLU‑Pro 等四大能力；测试集包含目标域（BBH、MBPP+、MATH、MMLU‑Pro）与非目标域（ZebraLogic、LiveCodeBench、OlympiadBench、GPQA Diamond）。

**📈 对比分析**

与随机挑选、Self‑Distillation、语义/表示‑based（DSIR、RDS）、质量评分（Qurating）、检索（BM25）、传统IF（LESS、Standard IF）等多种基线进行对比。结果显示，本文方法在目标域和非目标域均保持最稳健且最高的平均提升，显著突破标准SFT性能上限。

**⚠️ 局限性**

局限性：①验证集规模有限（约7.5k个Minerva MATH、约1k其他任务），对更广泛任务的泛化需进一步验证；②需要额外的on‑policy rollouts与梯度投影计算，耗费一定的推理与存储资源；③对梯度去偏的β参数需要针对每个任务调优，若任务多样性极高可能带来调参成本。

---

## 34. SAGE: Retain-Aware Post-Hoc Sanitization of Final Unlearning Vector

**arXiv ID:** 2606.18309 | [PDF](https://arxiv.org/pdf/2606.18309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 35. Reasoning as Intersection: Consensus-Frame Alignment for Visual Focus in Video-MLLMs

**arXiv ID:** 2606.18441 | [PDF](https://arxiv.org/pdf/2606.18441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 36. Stitching the Divide: Investigating Mixed Reality as a Bridge Between Paper-Based and Digital Artifacts in UI/UX Design

**arXiv ID:** 2606.18511 | [PDF](https://arxiv.org/pdf/2606.18511v1)

**作者:** Abidullah Khan `[一作]` (Polytechnique Montréal), Jinghui Cheng `[通讯]` (Polytechnique Montréal)

**通讯引用:** 3620 | [OpenAlex ID](https://openalex.org/A5101411842)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究UI/UX设计师如何通过混合现实（MR）工具将纸质与数字设计产物无缝衔接，并通过访谈与概念探针实验获取洞察。

**💡 创新点**

提出四个设计维度（整合多样化设计产物、融合低高保真原型、利用空间锚定、支持多层级协作），为未来MR设计工具提供系统化的改进思路。

**🔧 技术方法**

使用HoloLens 2、Unity3D 与 MRTK2 开发的概念探针，实现纸张与数字组件的投影、交互与保存；结合访谈数据与主题分析。

**📊 数据集**

采集了19名专业UI/UX设计师的访谈记录与9名参与者的概念探针实验录音，构成定性研究材料；未使用公开的标准数据集。

**📈 对比分析**

通过主题分析与访谈归纳，未做量化性能对比；研究以参与者主观感知与需求为评估标准，认为MR工具能显著提升工作流连贯性与协作效率。

**⚠️ 局限性**

局限包括样本规模小、受访者多为初级设计师、探针功能简化、缺乏团队与远程协作实验、参与者对MR经验有限，导致结论泛化性受限。

---

## 37. Caring Without Feeling: Affective Dynamics as the Control Layer of Human-AI Agent Collaboration

**arXiv ID:** 2606.18259 | [PDF](https://arxiv.org/pdf/2606.18259v1)

**作者:** Junjie Xu `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 32389 | [OpenAlex ID](https://openalex.org/A5102798483)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对人机协作中情感动态的计算与交互机制进行了系统综述，提出了将情感表达视为协调层的统一框架，并讨论了其在信任、授权、纠错与依赖中的作用。

**💡 创新点**

创新点在于将情感计算、自动化信任、LLM代理设计与AI安全等分散研究整合为一套“情感动态”框架，并将情感提示定位为协作中的外部协调信号，而非内部情绪；同时给出评估、设计与治理的分层方法。

**🔧 技术方法**

主要技术手段为文献综述与概念合成，阐述了提示调节、人格/角色设定、RLHF、记忆检索和安全策略等生成情感行为的计算机制。

**📊 数据集**

本研究未使用实验数据集，而是基于已有文献和案例进行理论整合与框架构建。

**📈 对比分析**

由于是综述性工作，没有对算法进行实验对比；性能评估以文献中的实验结果为参考，未给出统一指标。

**⚠️ 局限性**

局限性包括：证据来源分散、长期互动与依赖的实证数据不足、跨域适用性和文化差异尚未系统验证；框架仍需在实际代理系统中进行定量测试和治理落地评估。

---

## 38. LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents

**arXiv ID:** 2606.18388 | [PDF](https://arxiv.org/pdf/2606.18388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 39. Mixed-Precision Communication-Avoiding SGD for Generalized Linear Models on GPUs

**arXiv ID:** 2606.18463 | [PDF](https://arxiv.org/pdf/2606.18463v1)

**作者:** Aditya Devarakonda `[一作]` (Wake Forest University), Giulia Guidi `[通讯]` (Cornell University)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5034932123)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在NVIDIA GPU上针对广义线性模型的混合精度通信避免SGD（CA-SGD）算法，并给出了对应的误差分析与收敛证明。

**💡 创新点**

提出了完整的九个精度槽位误差分解，依据该分析设计出理论覆盖的混合精度配方（Recipe C），并在大规模GPU集群上实现。

**🔧 技术方法**

使用BF16输入、FP32累加的张量核心矩阵乘法、分组AllReduce、低精度存储、以及FP32主权重更新等技术，结合分布式Mini‑batch SGD与s‑step重排。

**📊 数据集**

在多种公开数据集（如SUSY、HIGGS、synthetic logistic、Poisson-synthetic）以及自定义大规模synthetic数据上进行实验。

**📈 对比分析**

与FP32 SGD基线进行对比，Recipe C在保持0.5%以内的损失误差的前提下，取得5.1–6.8×的加速（最高在synthetic数据上达到16.6×），并在弱/强缩放实验中表现出较高的效率。

**⚠️ 局限性**

局限性包括：1）理论证明仅覆盖Lipschitz残差的GLM（如逻辑回归、线性回归），对Poisson等不满足条件的损失仅给出经验验证；2）在大规模GPU数目（P>~60）时，需将Gram AllReduce改为FP32才能满足误差预算；3）对极低精度（FP16全流程）仍无法保证理论收敛。

---

## 40. CoreMem: Riemannian Retrieval and Fisher-Guided Distillation for Long-Term Memory in Dialogue Agents

**arXiv ID:** 2606.18406 | [PDF](https://arxiv.org/pdf/2606.18406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 41. ATIM: An ACT-R-Based Task Interface Model for Predicting Operator Action Time in Digital Nuclear Control Rooms

**arXiv ID:** 2606.18254 | [PDF](https://arxiv.org/pdf/2606.18254v1)

**作者:** Xingyu Xiao Jonghyun Kim `[一作]`, Haitao Wang `[通讯]` (Tsinghua University)

**通讯引用:** 120256 | [OpenAlex ID](https://openalex.org/A5100338047)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了ATIM模型，基于ACT‑R理论和界面特征预测数字核电站控制室中操作员的动作时间。

**💡 创新点**

创新点在于将界面可测量特征（目标显著性、语义干扰、交互距离）直接映射到视觉、认知、运动和交互时间组件，形成可解释且数据校准的预测框架。

**🔧 技术方法**

使用了ACT‑R认知架构、视觉搜索理论、Fitts定律以及K‑means聚类与L‑BFGS‑B参数优化的组合。

**📊 数据集**

数据集来自一台数字核电站控制室仿真环境，包含学生与经验操作员共计567条成功试验记录。

**📈 对比分析**

通过70/30分层验证，模型在未见数据上的步长平均绝对误差为3.17 s，相关系数0.664，显著优于传统单一黑盒模型且保持可解释性。

**⚠️ 局限性**

局限性包括数据量有限、仅覆盖单一实验环境、特征集缺乏时间压力、记忆负荷等因素，且对包含多次搜索或迭代决策的复杂步骤仍存在较大误差。

---

## 42. Towards Multi-Agent-Simulation-Based Community Note Evaluation

**arXiv ID:** 2606.18268 | [PDF](https://arxiv.org/pdf/2606.18268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 43. CloakLM: Obfuscating GPU Memory Layout to Mitigate Model Ex-filtration for Serving

**arXiv ID:** 2606.18400 | [PDF](https://arxiv.org/pdf/2606.18400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 44. EMORSION: Examining the Impact of Audio Parameters on Emotional Responses and Immersion in Film

**arXiv ID:** 2606.18266 | [PDF](https://arxiv.org/pdf/2606.18266v1)

**作者:** Nelly Garcia `[一作]` (Queen Mary University of London), Joshua Reiss `[通讯]` (Queen Mary University of London)

**通讯引用:** 2261 | [OpenAlex ID](https://openalex.org/A5111403298)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在电影院环境下通过对四段电影场景的频率、动态和方向性三种音频参数进行系统操控，评估其对观众情绪和沉浸感的影响。

**💡 创新点**

创新点在于首次将三模态（自评、心率、生理视频）三维三角测量法应用于真实影院，验证细微音频调整能显著塑造观众体验，并为大规模研究奠定可行性。

**🔧 技术方法**

采用Polar H10心率监测、OpenPose/OpenPifPaf视频运动追踪、Reaper和DaVinci Resolve音频混音插件，并用ANOVA、t检验及BH校正等统计方法进行分析。

**📊 数据集**

数据集包括四个5–10分钟电影片段（恐怖和剧情各两段，主流与独立各一段），共16个音频混音（1原始+3增强），40名观众在三场实验中提供自评、心率和运动数据。

**📈 对比分析**

通过对比控制混音与三种增强混音的自评沉浸评分、心率变异性和运动同步度，发现动态和方向性增强在多模态上均显著提升沉浸感，统计显著性达到p<0.001。

**⚠️ 局限性**

局限性包括运动追踪数据受低光与遮挡影响，帧率低导致时序分辨率不足；音频资源受限于商业曲目，混音精度有限；样本量小且分布不均，限制了结果的广泛推广。

---

## 45. From Specification to Execution: AI Assisted Scientific Workflow Management

**arXiv ID:** 2606.18425 | [PDF](https://arxiv.org/pdf/2606.18425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 46. Self-CTRL: Self-Consistency Training with Reinforcement Learning

**arXiv ID:** 2606.18327 | [PDF](https://arxiv.org/pdf/2606.18327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 47. RankGraph-2: Lifecycle Co-Design for Billion-Node Graph Learning in Recommendation

**arXiv ID:** 2606.18379 | [PDF](https://arxiv.org/pdf/2606.18379v1)

**作者:** Renzhi Wu `[一作]` (Meta Platforms), Hong Yan `[通讯]` (Meta Platforms)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了RankGraph-2系统，通过生命周期协同设计（图构建、训练、服务）在千亿节点规模下实现高质量相似性检索；

**💡 创新点**

创新点在于将三阶段需求相互耦合：离线预计算PPR邻居、在训练中共同学习残差量化聚类索引、实现无需在线图服务的KNN-free检索，显著提升检索召回并降低服务成本；

**🔧 技术方法**

使用的技术包括：异构协同参与图构建、边采样与流行度偏差校正、个性化PageRank邻居预计算、轻量多头编码器与异构聚合器、对比损失（margin + InfoNCE）训练、残差量化聚类索引与代码均衡正则化；

**📊 数据集**

实验数据来自Meta的千亿级用户-物品交互日志，构建的图包含数十亿节点和数百亿边；对比基准包括GAT+Deep Graph Infomax、PyTorch-BigGraph、HSTU以及公开基准（Amazon、MovieLens）；

**📈 对比分析**

在离线Recall@K、在线A/B测试中，RankGraph-2在用户召回上比GAT-DGI高3.8倍、在物品召回上比PBG高2.1倍；在Meta产品中提升CTR达+0.96%、CVR达+2.75%；服务成本相比传统在线KNN下降83%；

**⚠️ 局限性**

局限性包括：主要针对相似性检索，直接用户-物品匹配场景效果不佳；离线邻居预计算限制了实时查询时的动态探索；3小时刷新周期可能不满足极端时效性需求；

---

## 48. Simulating Hate Speech Cascades with Multi-LLM Agents: Empirical Grounding, Modeling Fidelity, and Intervention Strategies

**arXiv ID:** 2606.18264 | [PDF](https://arxiv.org/pdf/2606.18264v1)

**作者:** Fan Huang `[一作]` (Indiana University Bloomington), Fan Huang `[通讯]` (Indiana University Bloomington)

**通讯引用:** 222 | [OpenAlex ID](https://openalex.org/A5038274891)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在Bluesky上收集并分析了三条隐式仇恨言论扩散链，并构建了多种模型（经典扩散、行为启发式、简易LLM、全异质多LLM代理）进行对比模拟。

**💡 创新点**

创新之处在于首次将多代理LLM与个体属性、社区关系及内容语义耦合，揭示代理异质性对传播真实性的关键作用，并通过结构化消融与干预仿真验证多种去仇恨策略。

**🔧 技术方法**

技术上使用 GPT‑4o‑mini/4o 进行文本分类与属性推断，并在多代理框架下以概率预测方式实现传播决策，配合 IC、LT、行为启发式等传统扩散模型进行对照。

**📊 数据集**

数据来源为 2026 年 1 月至 4 月的 Bluesky 公共 API，抽取 2,000 条高转发英文帖子，进一步挑选出三条隐式仇恨语料链（共 2,241–2,942 名转发者）和一条规模匹配的无害娱乐链（3,919 名转发者）。

**📈 对比分析**

通过对比每个模型在敌对立场比例、毒性同质性差异、结构病毒度和跨社区渗透等指标上的绝对误差，发现多LLM代理在内容区分和结构病毒度上优于其它模型；而行为启发式在毒性同质性误差上最优，整体误差均低于传统扩散模型。

**⚠️ 局限性**

局限性包括仅研究 Bluesky 单一平台与三条仇恨链、受限于内部图的传播上限、LLM 推断属性的噪声、单一种子初始化以及干预结果仅为仿真假设，尚缺乏跨平台与实证验证。

---

## 49. The Gate Is Only as Honest as Its Contracts: ContractGuard for the Contract Layer of Risk-Aware Causal Gating

**arXiv ID:** 2606.18550 | [PDF](https://arxiv.org/pdf/2606.18550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 50. PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation

**arXiv ID:** 2606.18375 | [PDF](https://arxiv.org/pdf/2606.18375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 51. FluidViews: Adaptive Drag-and-Drop Token Filters for Heterogeneous Multi-View Visual Analytics

**arXiv ID:** 2606.18260 | [PDF](https://arxiv.org/pdf/2606.18260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 52. ASTRA: A Scalable Next-Generation ATCO Training Simulator with Autonomous Simpilots

**arXiv ID:** 2606.18319 | [PDF](https://arxiv.org/pdf/2606.18319v1)

**作者:** Ethan Chew `[一作]` (Air Emerging Technologies High-Speed Experimentations and Research, RSAF Agile Innovation Digital, Republic of Singapore Air Force), Yong Zhi Lim `[通讯]` (Air Emerging Technologies High-Speed Experimentations and Research, RSAF Agile Innovation Digital, Republic of Singapore Air Force)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个自动化的ATCO培训模拟器，集成了从语音识别到响应生成再到飞机运动模拟的完整语音流水线，实现了无需人工simulator pilot的培训。

**💡 创新点**

创新点在于构建端到端的多模态语音管线并结合混合规则+LLM评估框架，为ATCO训练提供了可扩展的、客观的实时评估；同时针对新加坡英语口音进行跨域适配。

**🔧 技术方法**

使用的技术包括前沿ASR（Large v3/WhisperATC）、定制化CIU（BERT + DSPy）、LLM（前沿模型+DSPy）响应生成、三种TTS模型（XTTS2.0, CSM, OrpheusTTS）以及实时SAM运动模拟。

**📊 数据集**

数据集包含ATCOSIM、MNSC、SG-Aviation（本地新加坡ATC语音）以及合成语料、公共航空对话语料与多任务国籍语料。

**📈 对比分析**

与现有模型比较，Fine-tuned Large v3在新加坡ATC上WER降至约23%；TTS在人类MOS上达3.7分，优于CSM；评估框架在准确率、简洁性、完整性上提升7-9%。

**⚠️ 局限性**

局限性包括对非新加坡口音、其他语言的泛化不足、TTS在长文本时的“幻觉”问题、评估样本规模有限以及对实时场景注入与自适应生成的支持不足。

---

## 53. Continuous Audio Thinking for Large Audio Language Models

**arXiv ID:** 2606.18273 | [PDF](https://arxiv.org/pdf/2606.18273v1)

**作者:** Gyojin Han `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Continuous Audio Thinking (CoAT) 框架，在大音频语言模型中插入连续的思考块，让模型在生成回答前组织并保留声学信息。

**💡 创新点**

创新点在于引入连续隐层工作空间并通过多专家蒸馏监督，使声学细节不再被文本目标稀疏监督所丢失，同时避免额外自回归推理成本。

**🔧 技术方法**

采用连续思考块、五种音频专家蒸馏（音频特征重建、语音表示、声源检测、情感预测、音高预测）、Transformer投影头、LoRA 适配器和两阶段训练策略。

**📊 数据集**

使用公开音频多模态混合数据进行训练，涵盖 ASR、音频问答、音频描述、多选问答、音乐理解、指令跟随等任务；评测数据集包括 MELD、MMAR、Alpaca‑Audio、MMAU 等。

**📈 对比分析**

与基线 LALM 及文本链式思考进行对比，CoAT 在多数基准上获得显著提升，尤其在推理密集任务和音乐分类上表现最佳，并且推理延迟低于文本链式思考。

**⚠️ 局限性**

局限性：思考块位置固定且不具动态或交互式多步推理能力；实验仅覆盖音频域，尚未验证对视觉或视频语言模型的迁移效果。

---

## 54. Co-evolution of the global research collaboration network and the performance of nations in science and technology

**arXiv ID:** 2606.18549 | [PDF](https://arxiv.org/pdf/2606.18549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 55. Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases

**arXiv ID:** 2606.18497 | [PDF](https://arxiv.org/pdf/2606.18497v1)

**作者:** Chandranil Chakraborttii `[一作]` (Trinity College), Shivanshu Dwivedi `[通讯]` (Trinity College)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了 HNSW 向量数据库软删除导致的幽灵向量泄露风险，并提出了 Epoch Key Rotation 加密删除方案。

**💡 创新点**

首次系统性证明软删除在存储层仍可恢复语义信息，并设计了可验证的加密擦除机制。

**🔧 技术方法**

使用了 Vec2Text 逆向模型、AES‑256‑CTR 加密、ECDSA 签名、HNSW 索引以及多模态评估技术。

**📊 数据集**

实验数据集包括 Wikipedia BLP、NIH Synthea、MIMIC‑III、PathMNIST、LFW 等文本与图像数据。

**📈 对比分析**

通过对比软删、全重建、AES 单条、Epoch 旋转四种删除策略，Epoch 旋转在 2.5 ms/500 向量完成且恢复率为 0%，而全重建需 2.25 s，显示显著性能优势。

**⚠️ 局限性**

局限性包括逆向模型泛化不足、实验规模仅至 10⁵ 条记录、以及需要硬件隔离的密钥管理以保障真正的物理擦除。

---

## 56. RELIANCE: Curating and Evaluating Reproductive Health Information on Social Media

**arXiv ID:** 2606.18285 | [PDF](https://arxiv.org/pdf/2606.18285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 57. Recover, Discover, Plan: Learning Skills and Concepts from Robot Failures

**arXiv ID:** 2606.18328 | [PDF](https://arxiv.org/pdf/2606.18328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 58. A Cross-Model VLM-Judge Protocol for Single-Image 3D Mesh Quality (and Why Cheap Proxies Fall Short)

**arXiv ID:** 2606.18451 | [PDF](https://arxiv.org/pdf/2606.18451v1)

**作者:** Ali Asaria `[一作]` (Transformer Lab), Deep Gandhi `[通讯]` (Transformer Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一种基于VLM判定器的单图像生成3D网格质量评估协议，并比较了常用的低成本代理指标。

**💡 创新点**

提出了固定24视角渲染装置、双VLM评判器以及位置偏差校正的完整评估流程，并证明代理指标无法替代。

**🔧 技术方法**

使用VLM（Qwen2.5-VL-7B-Instruct与InternVL3-8B）、CLIP、渲染-CLIP相似度、几何有效性统计以及Bradley‑Terry学习等技术。

**📊 数据集**

基于Google Scanned Objects数据集的单图像输入，使用Stable Fast 3D、TripoSR两款生成器，并加入面部裁剪失真。

**📈 对比分析**

通过交叉模型一致性（κ=0.66）和代理指标性能比较（几何0.62，render‑CLIP 0.48）证明代理指标表现差，只有可视缺陷情形下才显著。

**⚠️ 局限性**

局限在于仅评估两款生成器、单一失真方式、对VLM判断的依赖、未验证对更自然失真或多物体情形的泛化。

---

## 59. Finding Compiler-Platform Interaction Bugs in Deep Learning Pipelines via Cross-Layer Constraints

**arXiv ID:** 2606.18421 | [PDF](https://arxiv.org/pdf/2606.18421v1)

**作者:** Yuxin Qiu `[一作]` (University of California Riverside), Qian Zhang `[通讯]` (University of California Riverside)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发了一种基于全栈约束的深度学习编译器测试框架，自动提取跨层约束来生成模型、引导编译探索并插入断言，从而发现编译器‑平台交互错误。

**💡 创新点**

首次将全栈约束用于约束驱动模型生成和行为分区，自动化约束提取、优先级调度以及行为判定，能够揭示传统基于类型验证的测试方法无法捕获的内存溢出、整数溢出等交互错误。

**🔧 技术方法**

约束提取（基于模式匹配、AST解析、文档挖掘），约束驱动的模型生成（基于优先级和 LFU 调度的算法），断言注入（源代码变换），以及基于覆盖率的评估。

**📊 数据集**

使用 ONNX 格式的随机模型生成器，实验中共生成约 2,166 个模型，主要在 TVM、ONNX‑MLIR、GeneSys 三大编译器上进行评测。

**📈 对比分析**

与现有基于类型约束的 DL 编译器模糊器对比，该方法在 4 小时内发现 2,034 个新 bug（691 内存溢出、331 整数溢出、1,012 无声错误），显著提升了对深层编译逻辑和交互错误的覆盖率与发现率。

**⚠️ 局限性**

需要手工指定编译器和硬件相关的关键词，虽然只需少量配置，但仍依赖完整文档和源码；仅针对 ONNX 输入的编译器，且评估范围局限于三大编译器，缺乏对动态模型或自定义算子的支持验证。

---

## 60. Searching for Synergy in Shared Workspace Human-AI Collaboration

**arXiv ID:** 2606.18413 | [PDF](https://arxiv.org/pdf/2606.18413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 61. Designing L5: A Permacomputing Approach to Creative Coding

**arXiv ID:** 2606.18481 | [PDF](https://arxiv.org/pdf/2606.18481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 62. A skew polynomial framework for constructing division algebras and linear maximum rank distance codes

**arXiv ID:** 2606.18371 | [PDF](https://arxiv.org/pdf/2606.18371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 63. VEGA: Learning Navigation VLAs from In-the-Wild Egocentric Video with Geometric Trajectory Supervision

**arXiv ID:** 2606.18426 | [PDF](https://arxiv.org/pdf/2606.18426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 64. Mitigating Anchoring Bias in LLM-Based Agents for Energy-Efficient 6G Autonomous Networks

**arXiv ID:** 2606.18272 | [PDF](https://arxiv.org/pdf/2606.18272v1)

**作者:** Hatim Chergui `[一作]` (i2CAT Foundation), Merouane Debbah `[通讯]` (Khalifa University)

**通讯引用:** 68745 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于大型语言模型的自主网络切片资源协商框架，针对锚定偏差进行系统化的随机化抑制，并通过数字孪生与CVaR尾延迟评估实现零触摸、能效优先的6G网络切片。

**💡 创新点**

创新点包括：①使用截断三参数Weibull分布对起始资源锚定进行随机化，打破传统保守启发式；②构建Bimodal Constraint-Avoidance Utility Theorem，揭示在可行与不可行两种情形下的效用退化双峰边界；③将CVaR尾延迟与多目标能源成本联合评估，实现服务质量与能源效率的动态平衡。

**🔧 技术方法**

技术手段包括：1B参数本地LLM推理、数字孪生模型、M/M/1排队理论、CVaR计算、PID式增量决策、Weibull随机化策略、非实时RAN Intelligent Controller (non‑RT RIC) 接口。

**📊 数据集**

数据集：通过仿真生成的时变流量，eMBB切片基准90 Mbps并伴随正弦波变动，URLLC切片基准40 Mbps并伴随50%突发流量；不使用公开真实网络数据，全部为自制负载。

**📈 对比分析**

对比方法：在200次独立仿真中，将随机Weibull锚定策略与固定保守启发式基线对比，评估SLA违约、队列长度、能耗等指标。结果显示，随机策略在两阶段效用退化曲线上优势明显，能耗降低至25%，LLM平均推理时延0.95 s，完全满足非实时RIC的1 s时效需求。

**⚠️ 局限性**

局限性：仅在两张切片的仿真环境验证，缺乏真实网络部署验证；资源维度仅限CPU与带宽，未覆盖更多切片指标；Weibull参数设定为手工，缺乏自适应机制，可能在不同网络场景下需要重新调优。

---

## 65. P$^2$CE: Model-Agnostic Plausible Pareto-Optimal Counterfactual Explanations

**arXiv ID:** 2606.18418 | [PDF](https://arxiv.org/pdf/2606.18418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 66. Effects of sparsity and superposition on loss in simple autoencoders

**arXiv ID:** 2606.18538 | [PDF](https://arxiv.org/pdf/2606.18538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. Beyond Prediction: Tail-Aware Scheduling for LLM Inference

**arXiv ID:** 2606.18431 | [PDF](https://arxiv.org/pdf/2606.18431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 68. Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation

**arXiv ID:** 2606.18389 | [PDF](https://arxiv.org/pdf/2606.18389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 69. From Memorization to Creation: Evaluating the Cognitive Depth of LLM-Generated Educational Questions

**arXiv ID:** 2606.18257 | [PDF](https://arxiv.org/pdf/2606.18257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 70. Evaluating Prompting-Based Defenses Against Domain-Camouflaged Injection Attacks

**arXiv ID:** 2606.18530 | [PDF](https://arxiv.org/pdf/2606.18530v1)

**作者:** Aaditya Pai `[一作]` `[通讯]` (Columbia University), Aaditya Pai (Columbia University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估提示式防御对域伪装注入攻击的抑制效果

**💡 创新点**

首次系统比较多种提示防御在域伪装注入下的性能，并给出针对不同模型和业务域的实用建议

**🔧 技术方法**

采用提示式技术（spotlighting、paraphrasing、sandwiching 及其组合）与 LLM 评估者测算攻击成功率与实用性

**📊 数据集**

使用包含 45 个合成专业任务的基准数据集，涵盖金融、法律和通用三大领域

**📈 对比分析**

通过 3510 次实验在 Claude Haiku、Llama 3.1 8B 与 Gemini 2.0 Flash 三模型上比较 7 种防御方案，结果显示 paraphrasing 在所有模型上显著降低攻击成功率（降低 55–84%）并优于 Llama Guard 4，且不会导致过度拒绝

**⚠️ 局限性**

局限性包括样本量有限、仅测试合成文档、未验证真实企业文件、缺少人工评估以及未探究多目标注入的影响

---

## 71. How Well Do Large Language Models Capture Human Personality?

**arXiv ID:** 2606.18263 | [PDF](https://arxiv.org/pdf/2606.18263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 72. From Sparse Features to Trustworthy Proxies: Certifying SAE-Based Interpretability

**arXiv ID:** 2606.18383 | [PDF](https://arxiv.org/pdf/2606.18383v1)

**作者:** Dibyanayan Bandyopadhyay `[一作]` (Indian Institute of Technology Patna), Asif Ekbal `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 9809 | [OpenAlex ID](https://openalex.org/A5085370631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种后置校验框架，利用稀疏自编码器（SAE）构建稀疏代理，从而给冻结的语言模型上界提供风险证明；

**💡 创新点**

创新点在于将风险上界拆解为代理风险、重构误差、概念池不匹配和稀疏复杂度四个可测量量，并通过稀疏特征池的大小替代参数量实现非空界；

**🔧 技术方法**

采用稀疏自编码器、Top‑k稀疏化、重构、代理模型、Occam/PAC‑Bayes 经验风险上界等技术；

**📊 数据集**

使用英文 C4 数据集，对 GPT‑2 Small、Gemma‑2B 与 Llama‑3‑8B 三大模型进行实验；

**📈 对比分析**

通过与随机猜测基准对比，实验显示在实际样本量下三模型的风险上界均可降至基准以下；Llama‑3‑8B 的后层代理更易证伪，证伪误差与下游误差放大程度呈现层次依赖；

**⚠️ 局限性**

局限性包括：仅适用于冻结模型；需事先训练的 SAE 与目标模型匹配；对分布漂移（如随机噪声）缺乏正式保证；层次敏感性不具普适性，需针对具体模型调优。

---

## 73. Starter-Iterator Neural Operator: A Unified Architecture for High-Fidelity Forward and Inverse PDE Problems

**arXiv ID:** 2606.18305 | [PDF](https://arxiv.org/pdf/2606.18305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 74. Flexible Distributed Particle Filtering for the Internet of Things via Aggregate Computing

**arXiv ID:** 2606.18483 | [PDF](https://arxiv.org/pdf/2606.18483v1)

**作者:** Angela Cortecchia `[一作]`, Mirko Viroli `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在物联网环境中提出一种基于聚合编程（AC）的分布式粒子滤波（DPF）框架，利用计算场实现估计逻辑与通信层解耦；

**💡 创新点**

创新点在于将DPF抽象为可配置的计算场，允许动态选举融合中心、聚合邻域测量、空间自适应计算等多种设计空间在同一编程模型下统一实现；

**🔧 技术方法**

使用了AC宏编程范式、计算场模型（FC）、Kotlin实现的Collektive DSL、Alchemist仿真器进行实验；

**📊 数据集**

实验数据集为人工生成的目标跟踪场景，25个固定传感器、单目标的非线性轨迹，使用模拟信号强度作为测量；

**📈 对比分析**

通过比较不同邻域大小和领导者故障恢复的实验，展示聚合测量在邻居数≥1时能显著降低RMSE，领导者故障后快速恢复正常跟踪；实验结果表明所提框架在精度、通信成本和鲁棒性方面优于传统集中式DPF；

**⚠️ 局限性**

限制在于实验仅在单目标、单一传感类型的仿真环境下验证，缺乏多目标、多模态传感以及真实物理网络的评估；

---

## 75. CODEBLOCK: Learning to Supervise Code at the Right Granularity

**arXiv ID:** 2606.18286 | [PDF](https://arxiv.org/pdf/2606.18286v1)

**作者:** Zhijie Deng `[一作]` (Hong Kong University of Science and Technology), Jiaheng Wei `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结构感知稀疏监督框架，用于代码大语言模型的监督微调。

**💡 创新点**

创新点在于不对单独标记 token 进行稀疏监督，而是先将代码响应划分为结构完整的 coding item，利用 Generalized Cross‑Entropy（GCE）评估其核心逻辑信息，再通过轻量级静态数据流图对这些 item 进行重排序，最终仅对结构完整且具备重要数据依赖的代码片段和有意义的自然语言 token 进行梯度更新。

**🔧 技术方法**

主要技术包括：样本级数据筛选（质量+长尾稀缺性评分）、Tree‑Sitter 语法闭包构造 coding item、GCE 作为 token 重要性度量、基于 def‑use 数据流图的 Reach/Bridge 结构信号、门控优先级函数以及稀疏 next‑token 预测损失。

**📊 数据集**

使用 OpenCodeInstruct（约 300K 指令‑响应对）构建训练集，并在 6 个公开代码生成基准（HumanEval、HumanEval+、MBPP、MBPP+、BigCodeBench‑Hard、BigCodeBench‑Full）上评估。

**📈 对比分析**

与全 token 监督、随机采样、DS^2、Token Cleaning、CLAM 等基线比较，实验显示在 Qwen2.5‑Coder‑1.5B‑Instruct、Seed‑Coder‑8B、OpenCoder‑8B‑Base 等模型上，平均 pass@1 可达到 54.6/65.5/57.1，分别比全 token 方案提升约 2 点，且仅使用约 1.9% 的响应 token 进行监督，显著提升了性能‑效率 trade‑off。

**⚠️ 局限性**

局限性主要在于使用轻量级静态数据流分析，无法完整捕捉动态调用、别名、对象变异等运行时行为；因此 Reach/Bridge 信号为近似，未来可结合执行轨迹或更精细的程序分析来进一步提升稀疏监督效果。

---

## 76. Synthetic Resonance: A Framework for Growth-Oriented Human-AI Relationships

**arXiv ID:** 2606.18265 | [PDF](https://arxiv.org/pdf/2606.18265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 77. Quantum Annealing Enhanced Reinforcement Learning for Accurate Remaining Useful Lifetime Prediction

**arXiv ID:** 2606.18503 | [PDF](https://arxiv.org/pdf/2606.18503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 78. MOLAR: Learning Multimodal Molecular Representations from Noisy Labels

**arXiv ID:** 2606.18390 | [PDF](https://arxiv.org/pdf/2606.18390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 79. Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier

**arXiv ID:** 2606.18284 | [PDF](https://arxiv.org/pdf/2606.18284v1)

**作者:** Lorenz Wolf `[一作]` (Vmax), Matthew Daborn-Sargent `[通讯]` (Vmax)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在强化学习中训练任务生成器时，利用冻结参考模型的内部激活进行预测，替代昂贵的求解器回合。

**💡 创新点**

提出 PROPEL 框架，将激活探针作为任务实用性代理，显著降低求解器调用次数，并支持多步轨迹和模式坍塌抑制。

**🔧 技术方法**

采用 RLFR 的特征奖励、冻结参考模型激活提取、最坏情况优化（WCO）以及对抗式探针共进化等技术。

**📊 数据集**

使用数学竞赛题、代码诱导（AZR）和软件工程 bug 注入任务的自建数据集，标注学习前沿任务以训练探针。

**📈 对比分析**

与求解器回合实时 RL 基线比较，PROPEL 在数学、代码和 SWE 任务中均实现了约两倍的学习前沿任务产出率，且显著减少了求解器调用次数。

**⚠️ 局限性**

局限在于仅覆盖三种任务域、探针在持续策略漂移下可能失效、模式坍塌仍需改进，并需先行收集求解器标签。

---

## 80. A Survey on Data-Driven Models for Soil Moisture Regression and Classification

**arXiv ID:** 2606.18316 | [PDF](https://arxiv.org/pdf/2606.18316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. Why SWAVE May Not Be All You Need:A Concept-Evolution Retrospective on Complex-Valued Recurrent Language Models

**arXiv ID:** 2606.18324 | [PDF](https://arxiv.org/pdf/2606.18324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 82. MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval

**arXiv ID:** 2606.18508 | [PDF](https://arxiv.org/pdf/2606.18508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 83. Vines-DB: An RGB image dataset for multi-species ornamental vine segmentation

**arXiv ID:** 2606.18484 | [PDF](https://arxiv.org/pdf/2606.18484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 84. SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior

**arXiv ID:** 2606.18322 | [PDF](https://arxiv.org/pdf/2606.18322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 85. Vibe Coding Ate My Homework: An evaluation of AI approaches to greenfield software engineering and programming

**arXiv ID:** 2606.18293 | [PDF](https://arxiv.org/pdf/2606.18293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 86. NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation

**arXiv ID:** 2606.18271 | [PDF](https://arxiv.org/pdf/2606.18271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 87. LLM Parameters for Math Across Languages: Shared or Separate?

**arXiv ID:** 2606.18453 | [PDF](https://arxiv.org/pdf/2606.18453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 88. Denoising Distances in Metric Measure Spaces

**arXiv ID:** 2606.18301 | [PDF](https://arxiv.org/pdf/2606.18301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 89. Hierarchical Multi-Modal Retrieval for Knowledge-Grounded News Image Captioning

**arXiv ID:** 2606.18553 | [PDF](https://arxiv.org/pdf/2606.18553v1)

**作者:** Minh-Loi Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了层次化多模态检索增强的新闻图像标题生成框架，通过结构化文章检索和三阶段LLM生成实现更丰富的语义描述。

**💡 创新点**

创新点在于将新闻文章结构化为标题、导语、正文、图注并加权；结合内容-视觉、视觉-视觉和语篇位置三路相似度；以及使用检索后提取上下文并在LLM中合成知识驱动的标题。

**🔧 技术方法**

采用CLIP、BERT/句子变换器、Vision‑Language模型、LLM（大语言模型）和图文检索技术。

**📊 数据集**

使用OpenEvents V1与EVENTA 2025 Challenge 数据集（CNN、Guardian 2011–2022 200k+ 文章/400k+ 图像）。

**📈 对比分析**

与基线的检索和标题评估指标（mAP、R@1、R@10、CIDEr、CLIP Score）比较，检索mAP 0.708/0.956/0.991，标题CLIP 0.783/0.123 CIDEr，排名第5。

**⚠️ 局限性**

限制在于仍存在检索召回与多源噪声问题，标题与人工标注在词表重叠上差距大，且对模型推理资源要求高。

---

## 90. Repair Entropy in Dynamic Geometric Nearest-Neighbour Structures

**arXiv ID:** 2606.18314 | [PDF](https://arxiv.org/pdf/2606.18314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 91. A Link between Shock-wave Theory and Symmetry-reduced Stochastic Gradient Descent for Artificial Neural Networks

**arXiv ID:** 2606.18303 | [PDF](https://arxiv.org/pdf/2606.18303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Understanding the "Airport" Censorship Circumvention Ecosystem in China

**arXiv ID:** 2606.18427 | [PDF](https://arxiv.org/pdf/2606.18427v1)

**作者:** Rumaisa Habib `[一作]` (Stanford University), Zakir Durumeric `[通讯]` (Stanford University)

**通讯引用:** 6325 | [OpenAlex ID](https://openalex.org/A5069939742)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统研究了中国地下市场的“机场”网络代理生态，结合用户问卷、Telegram渠道抓取、域名扫描与主动网络测量，构建机场列表、评估性能、访问控制与自审策略。

**💡 创新点**

首次量化机场的规模、架构与价格，揭示其多跳IEPL链路的性能优势与自审风险，提供对地下商业式代理服务的系统性认识。

**🔧 技术方法**

使用问卷调查、Telegram抓取、域名扫描（ZGrab）、浏览器自动化（Playwright）、网络测量（RTT/TTFB/吞吐量）以及IP地理定位等技术手段。

**📊 数据集**

数据集包括1,667名受访者、3,431个机场实例、100个机场价格样本、35个机场深度测评、Telegram广告频道、CZDS域名列表和IP2Location。

**📈 对比分析**

通过与直连基线对比RTT、TTFB和吞吐量的基准实验，发现机场平均吞吐量131Mbps，略高于直连64Mbps，且在高峰时段仍保持>5Mbps，性能与价格呈正相关。

**⚠️ 局限性**

局限性包括样本偏向活跃用户、数据自报偏差；机场生态快速变动导致统计可能过时；测量仅针对低延迟节点与单一目标服务，未覆盖不同地区或多样化服务；对隐私与安全风险的评估有限。

---

## 93. When Mobile Crowdsourcing Meets Queueing Systems: Human-in-the-Loop Learning

**arXiv ID:** 2606.18392 | [PDF](https://arxiv.org/pdf/2606.18392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 94. TIGER: Inverting Transformer Gradients via Embedding-Subspace Distance Optimization

**arXiv ID:** 2606.18312 | [PDF](https://arxiv.org/pdf/2606.18312v1)

**作者:** William Kalikman `[一作]` (ETH Zürich), Martin Vechev `[通讯]` (ETH Zürich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Transformer线性层梯度低秩结构的连续子空间距离优化攻击（TIGER），能够在联邦学习环境下从梯度中恢复私有文本；

**💡 创新点**

创新点在于将子空间信息转换为可微的优化目标，直接对词嵌入进行连续优化，而非离散搜索或阈值判断；同时为解码器模型设计顺序恢复策略，为编码器模型引入双向子空间对齐损失；

**🔧 技术方法**

采用梯度子空间投影、正交化、正则化投影矩阵，结合Adam优化器和位置相关高斯先验；使用ROUGE指标评估重构质量；

**📊 数据集**

使用WikiText‑103构造位置相关高斯先验，评估数据来自GPT‑Neo、GPT‑2等现代Transformer模型的训练梯度；

**📈 对比分析**

与DATER、LAMP等基线对比，TIGER在未加扰动的解码器模型中保持相当性能，在加入DP噪声或BF16低精度梯度时仍能恢复70%以上ROUGE‑1；在编码器模型中，在批量更新下ROUGE‑1可达60%以上，显著优于现有方法；

**⚠️ 局限性**

局限性包括：当总词数接近隐藏维度时子空间变满，信息消失；对所有防御机制（如裁剪、掩码、压缩等）的鲁棒性未完全评估；重构仅提供部分文本，未能精确恢复顺序，需结合离散后处理。

---

## 95. Guava: An Effective and Universal Harness for Embodied Manipulation

**arXiv ID:** 2606.18363 | [PDF](https://arxiv.org/pdf/2606.18363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 96. Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications

**arXiv ID:** 2606.18502 | [PDF](https://arxiv.org/pdf/2606.18502v1)

**作者:** Paresh Dashore `[一作]` (Capital One), Shi-Xiong Zhang `[通讯]` (Capital One)

**通讯引用:** 2092 | [OpenAlex ID](https://openalex.org/A5101785327)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端的多代理LLM定制与推理优化框架，将大型教师模型蒸馏为小型高效模型并实现生产级部署。

**💡 创新点**

创新点在于：①上下文感知的持续预训练+监督微调+偏好优化的三阶段蒸馏流程；②结合EAGLE草稿模型与FP8量化的分层推理加速；③统一模型与系统级优化实现多重加速。

**🔧 技术方法**

技术手段包括：LoRA适配、DPO（直接偏好优化）、EAGLE speculative decoding、FP8 post‑training quantization、教师–判官蒸馏、用户模拟生成数据、块级扩张等。

**📊 数据集**

使用的数据集：公开对话语料、行业汽车文本、合成的多轮用户模拟轨迹、混合校准集（公共+内部数据）。

**📈 对比分析**

对比方式：与70B教师模型和未压缩学生模型在AWS EC2 P5上测得吞吐和延迟；结果显示在FP8+EAGLE组合下实现约4.48×吞吐提升，速度提升至2.5×且保持几乎无质量下降。

**⚠️ 局限性**

局限性：仅在汽车零售多代理场景验证，依赖单一教师/判官模型可能带来偏差；EAGLE与FP8需随业务更新重新训练；硬件依赖强，需NVIDIA Hopper等支持FP8。

---

## 97. Joint Discovery of Graph Structure and Dynamics in Stochastic Interacting Particle Systems

**arXiv ID:** 2606.18279 | [PDF](https://arxiv.org/pdf/2606.18279v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 98. When Prompts Mislead: Textual Dominance and Diagnostic Bias in MLLMs

**arXiv ID:** 2606.18262 | [PDF](https://arxiv.org/pdf/2606.18262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 99. SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG

**arXiv ID:** 2606.18381 | [PDF](https://arxiv.org/pdf/2606.18381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 100. Agentra: A Supervisable Multi-Agent Framework for Enterprise Intrusion Response

**arXiv ID:** 2606.18325 | [PDF](https://arxiv.org/pdf/2606.18325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 101. Reliable Neural-Codec Text-to-Speech by ASR Self-Verification and Distillation: Near-Zero Catastrophic Failures Across Models and Codecs

**arXiv ID:** 2606.18323 | [PDF](https://arxiv.org/pdf/2606.18323v1)

**作者:** Ali Asaria `[一作]` (Transformer Lab), Deep Gandhi `[通讯]` (Transformer Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了通过在神经编码器 TTS 模型上使用基于 ASR 的最佳样本自我验证和蒸馏来消除灾难性失败。

**💡 创新点**

提出了通用的 Best‑of‑N 自我验证加蒸馏方法，可在四个模型和三种神经编码器上将灾难性失败率降至近零。

**🔧 技术方法**

使用了 ASR 回环验证、Best‑of‑N 采样、自我验证蒸馏，以及与 DPO/IPO 等偏好优化技术进行对比。

**📊 数据集**

评估使用了 LibriSpeech test‑clean、手工构造的硬测试集以及罕见词/数字/日期集。

**📈 对比分析**

通过比较 Base、Best‑of‑N、蒸馏后单次推理的灾难性失败率，发现 Best‑of‑2 即可将失败率降至 0，蒸馏后单次推理在硬集上可减少约 52‑58% 的失败。

**⚠️ 局限性**

局限性包括罕见词能力上限、数字/日期测量噪声、对生成预算敏感性以及方法在易文本上的提升有限。

---

## 102. Neural Network Implementation of the Renormalization Group for Fault Diagnosis with Class Imbalance

**arXiv ID:** 2606.18326 | [PDF](https://arxiv.org/pdf/2606.18326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 103. RegimeVGGT: Layer-Wise Spatially Preserving Redundancy Removal for Visual Geometry Grounded Transformer

**arXiv ID:** 2606.18439 | [PDF](https://arxiv.org/pdf/2606.18439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. Possible or Definite? A Benchmark for Evaluating Diagnostic Uncertainty Preservation in Clinical Text

**arXiv ID:** 2606.18471 | [PDF](https://arxiv.org/pdf/2606.18471v1)

**作者:** Hongbo Du `[一作]` (Trine University), Jiaming Qu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

系统评估大型语言模型在临床文本任务中保持诊断不确定性的能力，构建了包含1200篇文档、9184条标注的不确定性表达的基准，并进行间接与直接评估。

**💡 创新点**

首次针对诊断不确定性构建细粒度基准，提出五级不确定性分级，并揭示LLM在保留不确定性方面的系统性偏差。

**🔧 技术方法**

使用规则式抽取、概念匹配、以及大语言模型（如Claude3、ChatGPT、Claude 2）在基准上的推理与生成评估。

**📊 数据集**

基于MIMIC‑IV‑Note（出院摘要、放射报告）和TCGA‑Reports（病理报告）两大公开数据库。

**📈 对比分析**

通过5项指标（TRR, URR, CAR, PCR, OHR）和分类/排序评测，结果显示LLM在不确定性保持上不到一半，误将约40%内容转为确定性陈述；在直接分类中准确率约78%，对相邻等级辨别不佳。

**⚠️ 局限性**

主要局限包括规则抽取误差、缺乏微调与多语言支持、未验证对临床决策的实际影响，且现有提示无法完全消除不确定性失真。

---

## 105. Beyond AHI: An Interpretable Causal-Discovery-Guided Framework for Sleep Recovery in Connected Health

**arXiv ID:** 2606.18506 | [PDF](https://arxiv.org/pdf/2606.18506v1)

**作者:** Saba A. Farahani `[一作]` (University of California, Irvine), Hung Cao `[通讯]` (University of California, Irvine)

**通讯引用:** 3187 | [OpenAlex ID](https://openalex.org/A5038366927)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种可解释的因果发现驱动框架，用多模态多导睡眠监测（PSG）数据构建分层睡眠恢复评分（SRS），并将其与患者自评恢复体验关联。

**💡 创新点**

创新点在于：①使用NOTEARS实现可解释的有向无环图（DAG）学习识别候选生理驱动因素；②引入两阶段筛选—生理学约束 + 受限LLM审计，剔除结构混杂和构念重叠；③将五个生理域（呼吸负荷、缺氧负荷、睡眠碎片、睡眠结构/EEG、交感神经调节）聚合为分层评分，便于在连接健康系统中模块化应用。

**🔧 技术方法**

技术方法包括：线性NOTEARS DAG估计、bootstrap 稳定性筛选、物理约束过滤、基于LLM的三分类审计、交叉结果共识聚合，以及基于权重的分层评分计算。

**📊 数据集**

使用两大人口队列数据集：MESA（1540人）和MrOS（825人），均包含多模态PSG特征和多项患者自评恢复结果。

**📈 对比分析**

与传统的呼吸暂停-低通气指数（AHI）对比，SRS在四项自评恢复指标上均显著相关，并且在最大相关系数上提升至2.5倍以上，表明多域整合的评分更能捕捉主观恢复体验。

**⚠️ 局限性**

局限性包括：①线性NOTEARS可能无法捕捉非线性交互；②因果结构学习假设因果充分性，未考虑潜在隐藏变量；③自评与客观生理之间的相关性仍较弱；④第二阶段审计依赖LLM而非专家手工评审；⑤仅为横断面分析，缺乏时间因果证据。

---

## 106. Measurement noise limits the advantage of nonlinear models over linear models in biomedical prediction

**arXiv ID:** 2606.18420 | [PDF](https://arxiv.org/pdf/2606.18420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 107. CAOA -- Completion-Assisted Object-CAD Alignment

**arXiv ID:** 2606.18429 | [PDF](https://arxiv.org/pdf/2606.18429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. Graph Instance Landscapes: When Structural Similarity Does (Not) Reflect Shortest-Path Performance

**arXiv ID:** 2606.18267 | [PDF](https://arxiv.org/pdf/2606.18267v1)

**作者:** Maryam Gholami Shiri `[一作]` (Jožef Stefan Institute), Tome Eftimov `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 2465 | [OpenAlex ID](https://openalex.org/A5082115266)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过把图实例嵌入低成本结构特征空间并聚类，构建实例景观并分析其对最短路径算法性能的影响。

**💡 创新点**

创新点在于将实例空间景观分析与图算法基准测试结合，揭示结构相似性与性能不一定相关，并展示不同基准生成器在景观中的分离。

**🔧 技术方法**

使用特征提取、k‑means聚类、t‑SNE降维、非参数统计检验（Anderson‑Darling、Kolmogorov‑Smirnov）等技术。

**📊 数据集**

数据集包括三大基准：加权 Erdős–Rényi 随机图、随机几何图以及真实道路网络（欧洲城市与美国地区）。

**📈 对比分析**

通过覆盖矩阵、聚类相似度和分布检验比较算法（Dijkstra、Bidirectional Dijkstra、A*、DEQ）的运行时，结果显示即便在相同景观区域内算法表现差异显著。

**⚠️ 局限性**

局限在于特征选取成本低导致缺乏对规模和拓扑细节的捕捉、聚类方法受 k 值和球形假设限制、样本量不足导致无法对真实网络做分布检验。

---

## 109. TMR-GGNN: Credit Card Fraud Detection based on Time-Aware Multi-Relational Guided Graph Neural Network

**arXiv ID:** 2606.18444 | [PDF](https://arxiv.org/pdf/2606.18444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 110. Fisher Width: A Geometric Measure of Complexity on Statistical Manifolds

**arXiv ID:** 2606.18306 | [PDF](https://arxiv.org/pdf/2606.18306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 111. TRIDENT: Breaking the Hybrid-Safety-Physics Coupling for Provably Safe Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.18308 | [PDF](https://arxiv.org/pdf/2606.18308v1)

**作者:** Zijie Meng `[一作]` (Peking University), Miao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 10162 | [OpenAlex ID](https://openalex.org/A5100376477)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个专为网络化物理控制系统设计的多智能体强化学习框架，解决了混合离散‑连续动作、训练时安全约束与物理先验三者之间相互耦合的难题。

**💡 创新点**

创新点包括：① 三向耦合定理阐明了三项挑战互相泄漏的闭环；② 共设计的框架（Temperature‑corrected, Residual, Infinitesimally feasible, DEcoupled, sequeNTial）通过梯度校正、Lyapunov约束更新和物理残差Critic三位一体的协同抑制了该闭环；③ 通过 Richardson–Romberg 梯度校正实现了 Gumbel‑Softmax 𝒪(τ²) 的梯度偏差；④ 采用 Lyapunov‑约束的逐步信赖域更新保证训练时逐步可行；⑤ 采用冻结物理先验的残差 Critic 实现了对物理知识的乘性嵌入，避免了奖励塑形导致的偏差。

**🔧 技术方法**

使用技术包括：Gumbel‑Softmax 与 Richardson–Romberg 梯度校正；Lyapunov‑约束的序列信赖域策略优化；物理先验残差 Critic；CTDE（集中训练、分散执行）框架；混合动作的两级条件化策略分解。

**📊 数据集**

实验数据集：① 多无人机移动边缘计算（UAV‑MEC）; ② 自主交叉口管理（AIM）; ③ SMAC 的混合动作变体。

**📈 对比分析**

与 MADDPG、MATD3、FACMAC、MAPPO、HAPPO、MAPPO‑Lag、MACPO、MADAC、Shielded RL 等基线对比，实验表明该方法在所有奖励与安全指标上均优越：安全违规率比 MADDPG 降低 95.5%，比 MACPO 降低 76.3%；奖励提升 13.5%；且能稳定扩展到 32 个智能体。

**⚠️ 局限性**

局限性包括：需已知的闭式物理模型，若无此先验则无法获得残差Critic 的优势；依赖 Slater 条件，安全集为空时无法收敛；额外的 Gumbel‑Softmax 前向传递带来约 18% 的计算开销；仅在模拟环境、最多 32 代理的实验验证，未评估大规模或真实部署情境。

---

## 112. SCOPE-FL: A Strategy-proof Chain-based Optimal pareto efficient Federated Learning System

**arXiv ID:** 2606.18384 | [PDF](https://arxiv.org/pdf/2606.18384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 113. A physical adaptive material motor unit neural network: a hygromorph composite material machine

**arXiv ID:** 2606.18275 | [PDF](https://arxiv.org/pdf/2606.18275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 114. TS-Fault: Benchmarking Time Series Forecasters Against Structural Faults

**arXiv ID:** 2606.18539 | [PDF](https://arxiv.org/pdf/2606.18539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 115. JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting

**arXiv ID:** 2606.18394 | [PDF](https://arxiv.org/pdf/2606.18394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 116. Enhancing neural network extrapolation in thermo-fluid systems using steady-state solutions

**arXiv ID:** 2606.18417 | [PDF](https://arxiv.org/pdf/2606.18417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 117. Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification

**arXiv ID:** 2606.18372 | [PDF](https://arxiv.org/pdf/2606.18372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 118. Enhanced Graph Neural Networks using K-Hop Gaussian Diffusion

**arXiv ID:** 2606.18317 | [PDF](https://arxiv.org/pdf/2606.18317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. Task-Restricted Symmetries in Recurrent Weight Space

**arXiv ID:** 2606.18457 | [PDF](https://arxiv.org/pdf/2606.18457v1)

**作者:** Simon Dräger `[一作]` `[通讯]` (Salk Institute for Biological Studies), Simon Dräger (Salk Institute for Biological Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在单层 tanh RNN 的权重空间中，利用有序实 Schur 分解对非正态耦合进行结构化消融，探究其在不同任务下的近似对称性和功能冗余。

**💡 创新点**

提出了 Schur‑坐标消融作为诊断工具，能够识别任务受限的近似功能不变性，并展示了不同任务和训练结果中可被安全消除的非正态耦合块。

**🔧 技术方法**

使用有序实 Schur 分解、Schur 坐标消融、单层 tanh RNN 训练、Δ 与 S_ΔT 指标评估、以及复制、翻转、正弦生成与上下文积分四个低维任务进行实验。

**📊 数据集**

实验数据集为合成低维任务：固定长度复制（8 个符号），3 位翻转（长度 25），频率诱导正弦生成（长度 50），以及上下文依赖积分（四输入，长度 48）。

**📈 对比分析**

通过比较消融前后在任务分布上的 rollout 差异（复制任务的回放准确率，其他任务的均方误差），发现某些 Schur 块的消除对性能影响几乎为零，而其他块则显著降低性能；不同任务和训练解的消融敏感度不同，未发现统一的“安全”块。

**⚠️ 局限性**

局限性：仅测试了单层 tanh RNN、低维合成任务、窄宽度范围和少量训练解；未验证对 LSTM/GRU 等门控网络、宽网络或真实序列任务的适用性；无法区分消融对隐藏状态主成分或读出对齐子空间的影响与 Schur 坐标本身的解释。

---

## 120. Forged Calamity: Benchmark for Cross-Domain Synthetic Disaster Detection in the Age of Diffusion

**arXiv ID:** 2606.18554 | [PDF](https://arxiv.org/pdf/2606.18554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. Holographic Integrated Sensing and Communication With Limited Radiation Amplitudes: How Many Quantization Bits Are Enough?

**arXiv ID:** 2606.18456 | [PDF](https://arxiv.org/pdf/2606.18456v1)

**作者:** Shuhao Zeng `[一作]` (Princeton University), Shuhao Zeng `[通讯]` (Princeton University)

**通讯引用:** 1647 | [OpenAlex ID](https://openalex.org/A5012112103)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了利用可重构全息表面（RHS）实现的集成感知与通信（ISAC）系统，并分析了RHS有限离散辐射幅值对系统性能的影响；

**💡 创新点**

创新点在于首次给出连续幅值与离散幅值下的通信速率与雷达信噪比的闭式下界，并据此推导出对系统性能阈值满足所需的量化比特数上界，从而明确通信与感知性能折衷对量化比特需求的影响；

**🔧 技术方法**

采用全息波束成形（数字波束与RHS模拟波束）模型，结合信道模型与噪声模型，利用误差展开与二阶近似得到性能下界，并用数值方法求解量化比特上界；

**📊 数据集**

实验采用随机均匀分布的通信用户、目标与杂波源在空间中的距离与角度进行仿真，参数设置符合3GPP 3D模型（频率30 GHz、波长0.01 m等），通过Monte‑Carlo仿真验证理论结果；

**📈 对比分析**

与理想连续幅值系统以及传统相控阵ISAC系统进行对比。仿真表明量化比特上界与实际最小比特数非常接近；在高性能阈值下，感知优先系统需要的量化比特少于通信优先系统；RHS基ISAC对量化误差更稳健，所需比特数低于相控阵系统；

**⚠️ 局限性**

限制主要体现在：仅考虑幅值离散而忽略相位误差；模型假设自由空间LoS通道，实际多径与非LoS环境下的性能分析尚未给出；量化误差被视为均匀分布，实际硬件误差可能更复杂；

---

## 122. CEO-Bench: Can Agents Play the Long Game?

**arXiv ID:** 2606.18543 | [PDF](https://arxiv.org/pdf/2606.18543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 123. Engagement Intensity as a Learner-Modeling Signal for Adaptive AI Ethics Instruction

**arXiv ID:** 2606.18548 | [PDF](https://arxiv.org/pdf/2606.18548v1)

**作者:** Yongkyung Oh `[一作]` (University of California, Los Angeles), Alex Bui `[通讯]` (University of California, Los Angeles)

**通讯引用:** 3296 | [OpenAlex ID](https://openalex.org/A5056478657)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探究了研究生层面AI伦理课程前的三种入学特征（使用频率、自我评估熟悉度、先前AI教育）与学生AI感知的关系；

**💡 创新点**

证明了行为性使用频率是最能预测学生预课程AI感知的指标，且仅通过短问卷即可实现入学分层；

**🔧 技术方法**

采用了统计方法（Spearman相关、Kruskal–Wallis检验）并结合Holm多重检验校正；

**📊 数据集**

使用了来自UCLA生物科学研究生及博士后93名学员的自填式预课程问卷数据；

**📈 对比分析**

对三种特征进行逐一比较，发现使用频率在所有5个AI感知维度上均显著相关（ρ≈0.4），熟悉度在3维度显著，先前教育无显著关联；

**⚠️ 局限性**

局限性包括横断面设计、单机构样本、单项自评指标、可能的共同方法方差，且未检验个体差异对后续学习效果的影响。

---

## 124. On the Residual Scaling of Looped Transformers: Stability and Transferability

**arXiv ID:** 2606.18524 | [PDF](https://arxiv.org/pdf/2606.18524v1)

**作者:** Shaowen Wang `[一作]` (Tsinghua University), Jian Li `[通讯]` (Tsinghua University)

**通讯引用:** 43671 | [OpenAlex ID](https://openalex.org/A5100402427)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究针对循环共享权重的Transformer，提出了残差缩放规则，使得网络在不同循环次数下保持稳定并实现超参数迁移。

**💡 创新点**

创新点在于发现1/N的残差缩放（而非1/√N）才是权重共享时保持前向稳定的正确尺度，并给出了跨深度与循环次数的因子化参数化。

**🔧 技术方法**

使用了残差重参数化、ReLU/SwiGLU激活、基于高斯权重的理论分析以及实验中的decoder‑only Llama式Transformer。

**📊 数据集**

采用FineWeb‑Edu大规模文本数据集进行语言建模训练。

**📈 对比分析**

与传统1/√N缩放和无缩放方案对比，实验显示1/N缩放在循环次数增加时保持更低的验证损失，并且学习率可以不随循环次数变化，性能提升约0.025 nats。

**⚠️ 局限性**

局限性包括理论仅考虑单一共享MLP并假设ReLU，未涵盖多头注意力、优化器状态、归一化变体，且实验规模有限且仅单随机种子。

---

## 125. Confident yet Concerned: Inconsistencies in Computing Students' Attitudes on Cybersecurity

**arXiv ID:** 2606.18541 | [PDF](https://arxiv.org/pdf/2606.18541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 126. Hierarchical Attention via Domain Decomposition

**arXiv ID:** 2606.18525 | [PDF](https://arxiv.org/pdf/2606.18525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 127. PSyGenTAB: A Privacy-Preserving Framework for Synthetic Clinical Tabular Data Generation via Constrained Optimization

**arXiv ID:** 2606.18518 | [PDF](https://arxiv.org/pdf/2606.18518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 128. Task Allocation and Motion Planning in Dynamic, Cluttered Environments via CBBA and Graphs of Convex Sets

**arXiv ID:** 2606.18516 | [PDF](https://arxiv.org/pdf/2606.18516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 129. N(CO)$^2$: Neural Combinatorial Optimization with Chance Constraints to Solve Stochastic Orienteering

**arXiv ID:** 2606.18514 | [PDF](https://arxiv.org/pdf/2606.18514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 130. Do as the Romans Do: Learning Universal Behaviors from Heterogeneous Agents

**arXiv ID:** 2606.18537 | [PDF](https://arxiv.org/pdf/2606.18537v1)

**作者:** Caleb Chang `[一作]` (University of Washington), Karen Leung `[通讯]` (University of Washington)

**通讯引用:** 1295 | [OpenAlex ID](https://openalex.org/A5007340626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出 GRID 方法，对来自不同目标的演示者的奖励进行分解，提取通用奖励训练通用智能体，然后通过 KL‑控制微调获得专用智能体。

**💡 创新点**

创新点在于：① 用信息论损失同时学习通用奖励、个体嵌入和特定奖励；② 通过容量瓶颈和强调损失强制通用奖励占主导，避免多模态平均；③ 把奖励分解作为通用预训练的先验，显式解释通用与个体行为的关系。

**🔧 技术方法**

技术包括：深度神经网络两分支（R_g 与 R_s）、互信息最大化 (ELBO)、PPO 强化学习、KL‑控制微调、对比基线 BC 与 AIRL。

**📊 数据集**

使用了三类数据集：① 合成基函数分解（可视化奖励分解）；② Craftax（多智能体 8×8 的 Minecraft 风格游戏）；③ Highway‑Env（连续状态、离散动作的三车道自驾仿真）。演示者数据通过 IPPO 生成，包含个体奖励信息。

**📈 对比分析**

与 BC、AIRL、专家 IPPO、以及专用 RL 对手进行对比。GRID 在通用奖励上训练的 PPO 通用智能体在奖励、回合长度、熵等指标上均显著优于基线；在微调至未见的 Lane‑3 时，GRID 仍保持高回报和长生存时间，表现最优。

**⚠️ 局限性**

局限性：① 需要每位演示者的奖励信号（若依赖 IRL，则受 IRL 质量影响）；② 需要覆盖足够的行为空间；③ 目前仅在小规模仿真环境验证，尚未验证在更大规模或真实世界社交场景中的可扩展性。

---

## 131. Concept Modulation Models: A Unified Framework for Identifiability and Extrapolation

**arXiv ID:** 2606.18509 | [PDF](https://arxiv.org/pdf/2606.18509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 132. A Prototypical Signature Approach for Writer-Independent Offline Signature Verification

**arXiv ID:** 2606.18528 | [PDF](https://arxiv.org/pdf/2606.18528v1)

**作者:** Kecia G. de Moura `[一作]` (Université du Québec), Rafael M. O. Cruz `[通讯]` (Université du Québec)

**通讯引用:** 2175 | [OpenAlex ID](https://openalex.org/A5019553116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于原型签名的数据驱动负样本生成策略，用于无写作者指纹的离线签名验证。

**💡 创新点**

通过聚类得到非可识别的原型签名，按距离挑选最具信息量的负样本，实现负样本多样化与稀疏化。

**🔧 技术方法**

k-means聚类、欧氏距离挑选、双分形转换、线性SVM（SGD）与RBF SVM对比，SigNet-S特征提取。

**📊 数据集**

GPDS Synthetic、CEDAR、MCYT-75三个基准数据集。

**📈 对比分析**

与传统随机伪造采样对比，在全尺寸与压缩尺寸下EER基本持平或略优；使用线性SVM时训练时间和内存下降两三百倍，支持向量数为零，性能与RBF相当。

**⚠️ 局限性**

受聚类效果影响，对样本量和分布敏感，原型簇可能偏向主流模式，且对极少样本用户表现尚未验证。

---

## 133. SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR

**arXiv ID:** 2606.18487 | [PDF](https://arxiv.org/pdf/2606.18487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 134. Sparsity Curse: Understanding RLVR Model Parameter Space from Model Merging

**arXiv ID:** 2606.18521 | [PDF](https://arxiv.org/pdf/2606.18521v1)

**作者:** Chenrui Wu `[一作]` (Zhejiang University), Haishuai Wang `[通讯]` (Zhejiang University)

**通讯引用:** 2300 | [OpenAlex ID](https://openalex.org/A5047118636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 RLVR 训练的 LLM 在模型合并时的稀疏更新导致的性能下降，提出了 SAR‑Merging 方法解决稀疏诅咒。

**💡 创新点**

提出敏感度驱动的冲突解决和稀疏保留的合并策略，针对 RLVR 参数空间的稀疏、正交更新特性。

**🔧 技术方法**

利用 Fisher 信息估计参数敏感度、稀疏化与再缩放、激活空间曲率分析以及基于任务向量的合并。

**📊 数据集**

在 GSM8K、MATH、HumanEval、MBPP 等数学与代码生成基准上进行实验。

**📈 对比分析**

与 Linear、Task Arithmetic、TIES、DARE、RAM 等传统合并方法对比，SAR‑Merging 在大多数 RLVR 模型对上提高约 5–10% 准确率，显著低于其他方法的性能衰退。

**⚠️ 局限性**

合并后模型仍低于最佳父模型，且方法主要针对两模型合并，尚未验证大规模模型及多模型融合。

---

## 135. Architectural Bias in Face Presentation Attack Detection: A Comparative Study of Vision Transformers and Convolutional Neural Networks

**arXiv ID:** 2606.18510 | [PDF](https://arxiv.org/pdf/2606.18510v1)

**作者:** Ngela Landon Ntung `[一作]` (Carnegie Mellon University Africa), Jema David Ndibwile `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5034886864)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了 Vision Transformer 在面部伪造检测中的公平性与性能提升。

**💡 创新点**

创新点在于证明预训练 ViT 能显著降低不同肤色间的误差差距，并在未见族群上拥有更好的泛化能力。

**🔧 技术方法**

使用了 DeiT‑S、ResNet18 以及传统 LBP+SGD 等模型，比较其在 PAD 任务中的表现。

**📊 数据集**

数据集为 CASIA‑SURF CeFA 跨种族面部防伪数据集。

**📈 对比分析**

通过准确率、EER、ACER、BPCER 等指标和公平性差距（Δ_fairness）进行比较；DeiT‑S 达到 97.27% 准确率、0.86% EER，公平性差距仅 0.13%。

**⚠️ 局限性**

局限在于仅使用单一数据集，未分别评估预训练与架构对公平性的独立贡献，也未探讨多模态或真实部署场景。

---

## 136. The Illusion of Improvement: Reject Inference Strategies in Credit Scoring

**arXiv ID:** 2606.18479 | [PDF](https://arxiv.org/pdf/2606.18479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 137. PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning

**arXiv ID:** 2606.18473 | [PDF](https://arxiv.org/pdf/2606.18473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 138. Structured Representation Learning with Locally Linear Embeddings and Adaptive Feature Fusion

**arXiv ID:** 2606.18469 | [PDF](https://arxiv.org/pdf/2606.18469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 139. Montreal Forced Aligner and the state of speech-to-text alignment in 2026

**arXiv ID:** 2606.18466 | [PDF](https://arxiv.org/pdf/2606.18466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. AI Sandboxes: A Threat Model, Taxonomy, and Measurement Framework

**arXiv ID:** 2606.18532 | [PDF](https://arxiv.org/pdf/2606.18532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 141. Neural Phase Correlation

**arXiv ID:** 2606.18496 | [PDF](https://arxiv.org/pdf/2606.18496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. From Bits to Mixed-Radix Keys: Horner Decomposition, Uniform Sampling, and the Information-Theoretic QKD Interface of the MR-OTP

**arXiv ID:** 2606.18526 | [PDF](https://arxiv.org/pdf/2606.18526v1)

**作者:** Fabio F. G. Buono `[一作]` `[通讯]`, Fabio F. G. Buono

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了混合进制一次性密码（MR‑OTP）并给出从 QKD 产生的二进制熵到混合进制密钥的完整转换算法，构建了信息理论安全的端到端 QKD+加密流程。

**💡 创新点**

创新点在于将 Horner 分解与其逆变换作为二进制整数与多进制键空间之间的自然映射，利用拒绝采样消除模偏差，保证密钥分布均匀，从而在信息理论层面实现完美保密；同时提出批量采样、ε‑安全分析与动态密钥滚动的理论框架。

**🔧 技术方法**

使用的技术包括 Horner 方法、逆 Horner 分解、拒绝采样、信息理论安全证明、批量采样（一次性多位整数提取）、ε‑安全模型、基数恢复问题分析、Syntactic Invariance Principle 等。

**📊 数据集**

使用的数据集为理想或 ε‑近似均匀的 QKD 位流，实验中模拟了多种自然字母表（DNA 4 字母、拉丁字母 26 字母、十进制等）以评估密钥长度与采样效率。

**📈 对比分析**

与传统二进制 OTP、AES+QKD 等方案对比，MR‑OTP 在密钥长度（≈log₂P 位）和比特消耗（期望 ≤2log₂P 位）方面显著优于二进制 OTP，且提供信息理论完美安全；批量采样进一步降低了比特消耗，提升了吞吐量。

**⚠️ 局限性**

局限性：对非理想 QKD 源需满足 ε ≤ δ/(2N) 的约束；已知明文情况下基数恢复问题的算术复杂度尚未证明；动态密钥滚动方案的安全性仍是未解开放问题。

---

## 143. Data-Forcing Distillation: Restoring Diversity and Fidelity in Few-Step Video Generation

**arXiv ID:** 2606.18478 | [PDF](https://arxiv.org/pdf/2606.18478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 144. Domain Generalizable Adaptation of 3D Vision-Language Models via Regularized Fine-Tuning

**arXiv ID:** 2606.18472 | [PDF](https://arxiv.org/pdf/2606.18472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 145. "The New Era of Tech-Enabled Traceability": Tensions between the FDA's Data Governance Vision and the Lived Realities of Food Producers

**arXiv ID:** 2606.18593 | [PDF](https://arxiv.org/pdf/2606.18593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 146. Speech-Driven End-to-End Language Discrimination towards Chinese Dialects

**arXiv ID:** 2606.18584 | [PDF](https://arxiv.org/pdf/2606.18584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. Technical Report for ICRA 2026 GOOSE 2D Fine-Grained Semantic Segmentation Challenge: Leveraging DINOv3 for Robust Outdoor Scene Understanding in Field Robotics

**arXiv ID:** 2606.18582 | [PDF](https://arxiv.org/pdf/2606.18582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 148. Experimental Analysis of Neural Network-Based Image Classification on the CIFAR-10 Dataset

**arXiv ID:** 2606.18565 | [PDF](https://arxiv.org/pdf/2606.18565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. Correcting Sensor-Induced Distribution Drift with Wasserstein Adversarial Learning

**arXiv ID:** 2606.18561 | [PDF](https://arxiv.org/pdf/2606.18561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 150. Reference-Based Recursive Least-Squares Mitigation of Real Interference in Stereo Audio Recordings

**arXiv ID:** 2606.18564 | [PDF](https://arxiv.org/pdf/2606.18564v1)

**作者:** Necati Kagan Erkek `[一作]` (Politecnico di Milano), Y. Ugur Ozcan `[通讯]` (Politecnico di Milano)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究针对真实列车噪声污染的立体音频录音，提出并评估基于参考信号的递归最小二乘（RLS）自适应噪声消除方案。

**💡 创新点**

创新点在于利用另一路立体参考采样捕获相同噪声源的不同传播路径，并通过双通道RLS估计并减去可参考噪声成分，同时结合低通后滤波以进一步抑制残余高频。

**🔧 技术方法**

使用的技术包括信号对齐（交叉相关）、多参考二维RLS滤波、反因果样本插入、指数加权最小二乘优化、以及Hamming窗窗低通FIR后处理。

**📊 数据集**

实验数据集由三段真实录制的74.01秒、11.025 kHz采样率立体音频组成，包含主录音、参考录音和环境背景噪声。

**📈 对比分析**

评估采用无参考指标：RMS变化、参考相关最大值及其比率，谱密度对比；结果显示参考相关从0.386–0.832降至0.011–0.016，对应30.6–34.1 dB的相关比率减小，RMS下降1.8–4.8 dB。

**⚠️ 局限性**

局限性包括缺乏干净基线导致无法计算SNR/MSE，参考与期望信号的潜在相关性可能导致期望信号损失，反因果滤波仅适用于离线处理，后置低通滤波可能削弱高频音质。

---

## 151. Rethinking Text-to-Image as Semantic-Aware Data Augmentation for Indoor Scene Recognition

**arXiv ID:** 2606.18555 | [PDF](https://arxiv.org/pdf/2606.18555v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 152. PersonalPlan: Planning Multi-Agent Systems for Personalized Programming Learning

**arXiv ID:** 2606.18633 | [PDF](https://arxiv.org/pdf/2606.18633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 153. AI-Driven Assessment of Human Tutors: Linking Training Performance to Real-Life Practice

**arXiv ID:** 2606.18617 | [PDF](https://arxiv.org/pdf/2606.18617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 154. ShuntServe: Cost-Efficient LLM Serving on Heterogeneous Spot GPU Clusters

**arXiv ID:** 2606.18600 | [PDF](https://arxiv.org/pdf/2606.18600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 155. Bridging Creative Intent and Visual Quality: Creator-Driven Recurrent Video Generation with Agentic Feedback Loops

**arXiv ID:** 2606.18591 | [PDF](https://arxiv.org/pdf/2606.18591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 156. EffiNav: Fusing Depth and Vision-Language for Efficient Object Goal Navigation

**arXiv ID:** 2606.18634 | [PDF](https://arxiv.org/pdf/2606.18634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 157. ROBOSHACKLES: A Safety Dataset for Human-Injury Prevention in Embodied Foundation Models

**arXiv ID:** 2606.18632 | [PDF](https://arxiv.org/pdf/2606.18632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 158. DNN Koopman-Based Deviation Compensation for UGV Path Tracking Control on Coupled Slope and Potholed Road

**arXiv ID:** 2606.18630 | [PDF](https://arxiv.org/pdf/2606.18630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 159. SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping

**arXiv ID:** 2606.18625 | [PDF](https://arxiv.org/pdf/2606.18625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 160. BCL: Bayesian In-Context Learning Framework for Information Extraction

**arXiv ID:** 2606.18620 | [PDF](https://arxiv.org/pdf/2606.18620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 161. Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistance

**arXiv ID:** 2606.18613 | [PDF](https://arxiv.org/pdf/2606.18613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 162. QC-GAN: A Parameter-Efficient Quaternion Conformer GAN for High-Fidelity Speech Enhancement

**arXiv ID:** 2606.18611 | [PDF](https://arxiv.org/pdf/2606.18611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 163. Hallucination Detection and Correction in Medical VLMs via Counter-Evidence Verification

**arXiv ID:** 2606.18609 | [PDF](https://arxiv.org/pdf/2606.18609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 164. Splaxel: Efficient Distributed Training of 3D Gaussian Splatting for Large-scale Scene Reconstruction via Pixel-level Communication

**arXiv ID:** 2606.18588 | [PDF](https://arxiv.org/pdf/2606.18588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 165. MetaboNet-Bench: A Multi-modal Benchmark for Glucose Forecasting in Type 1 Diabetes

**arXiv ID:** 2606.18640 | [PDF](https://arxiv.org/pdf/2606.18640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 166. Low-resource Language Discrimination Towards Chinese Dialects with Transfer learning and Data Augmentation

**arXiv ID:** 2606.18597 | [PDF](https://arxiv.org/pdf/2606.18597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 167. Fair Cognitive Impairment Detection Through Unlearning

**arXiv ID:** 2606.18571 | [PDF](https://arxiv.org/pdf/2606.18571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 168. Optimizing Lithium Production Decisions under Geological, Demand, and Pricing Uncertainties: A POMDP Framework for Multi-Objective Decision Making

**arXiv ID:** 2606.18598 | [PDF](https://arxiv.org/pdf/2606.18598v1)

**作者:** Anna C. Edmonds `[一作]` (Stanford University), Jef Caers `[通讯]` (Stanford University)

**通讯引用:** 8593 | [OpenAlex ID](https://openalex.org/A5001828028)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一个以部分可观测马尔可夫决策过程（POMDP）为核心的锂矿生产决策框架，统一考虑地质不确定性、价格与需求波动以及直接锂提取（DLE）和硬岩（spodumene）两种技术的选择，并通过贝叶斯更新动态维护对矿产储量和市场条件的信念。

**💡 创新点**

创新点在于：①将多源不确定性（地质、价格、需求、技术）集成到单一POMDP模型中；②采用在线规划器POMCPOW，实现实时贝叶斯更新与多步预测；③引入可调α参数，将利润与碳排放成本统一为多目标奖励，生成完整的利润‑排放 Pareto 前沿，支持不同策略权衡。

**🔧 技术方法**

主要技术包括：POMDP建模与贝叶斯更新（Kalman滤波器），POMCPOW在线规划算法，四种价格动力学（静态、线性、指数、几何布朗运动）及历史价格序列，利润与排放的加权奖励函数。

**📊 数据集**

使用的数据集包括：1994‑2024年锂价格历史、全球锂需求预测（按30%市场份额分配），四个代表性矿区（两套DLE、两套硬岩）的成本、回收率与储量参数，以及对应的地质估计噪声分布。

**📈 对比分析**

实验中将POMCPOW与七种启发式基线（随机、仅探索、单步利润/排放最大化、动态利润/排放最大化、单步前瞻、深度1等）在静态、线性、指数、几何布朗运动和历史价格模型下进行比较。结果表明，POMCPOW在所有价格场景下均实现最高或近乎最高的贴现回报，且在利润、排放成本与需求满足率方面普遍优于基线。

**⚠️ 局限性**

局限性包括：价格被视为外生，未考虑生产决策对市场价格的反馈；假设矿区一旦开启即全负荷运行，缺乏产能调节；模型仅考虑单一企业决策，未覆盖多主体竞争或合作；监管与碳定价等政策约束未显式建模，环境模块相对简化。

---

## 169. Constraining to Generalize: Subspace Tuning for Few-shot Generalization of Audio-Language Models

**arXiv ID:** 2606.18560 | [PDF](https://arxiv.org/pdf/2606.18560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 170. HI-HCQC: A Tightly-Coupled Hardware Interface with High-Efficiency Communication for Hybrid Classical-Quantum Computing

**arXiv ID:** 2606.18642 | [PDF](https://arxiv.org/pdf/2606.18642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 171. PEC-Home: Interpretation of Progressively Elliptical Commands in Smart Homes

**arXiv ID:** 2606.18636 | [PDF](https://arxiv.org/pdf/2606.18636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 172. PACT: Preserving Anchored Cores in Task-vectors for Model Merging

**arXiv ID:** 2606.18627 | [PDF](https://arxiv.org/pdf/2606.18627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. Intrinsic 4D Gaussian Segmentation from Scene Cues

**arXiv ID:** 2606.18623 | [PDF](https://arxiv.org/pdf/2606.18623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 174. Towards Anomaly Detection on Relational Data

**arXiv ID:** 2606.18621 | [PDF](https://arxiv.org/pdf/2606.18621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 175. Code-Augur: Agentic Vulnerability Detection via Specification Inference

**arXiv ID:** 2606.18619 | [PDF](https://arxiv.org/pdf/2606.18619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 176. Steerable Cultural Preference Optimization of Reward Models

**arXiv ID:** 2606.18606 | [PDF](https://arxiv.org/pdf/2606.18606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 177. Admittance-Based Surface Alignment for Human-in-the-Loop Robotic Visual Inspection

**arXiv ID:** 2606.18601 | [PDF](https://arxiv.org/pdf/2606.18601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 178. MIDS: Detecting Stealthy Masquerade and Tampering Attacks on CAN Bus via Bidirectional Mamba

**arXiv ID:** 2606.18599 | [PDF](https://arxiv.org/pdf/2606.18599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 179. DREAM-Chunk: Reactive Action Chunking with Latent World Model

**arXiv ID:** 2606.18589 | [PDF](https://arxiv.org/pdf/2606.18589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 180. Dual Dimensionality for Local and Global Attention

**arXiv ID:** 2606.18587 | [PDF](https://arxiv.org/pdf/2606.18587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 181. MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction

**arXiv ID:** 2606.18558 | [PDF](https://arxiv.org/pdf/2606.18558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 182. Self-Supervised Mask-Aware Transformers for Fault-Tolerant FBG Force Sensing in Minimally Invasive Surgical Robotics

**arXiv ID:** 2606.18628 | [PDF](https://arxiv.org/pdf/2606.18628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation

**arXiv ID:** 2606.18610 | [PDF](https://arxiv.org/pdf/2606.18610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 184. Better Adherence, Richer Context: A Field Evaluation of LLM-Powered Conversational Voice Diaries for Sleep

**arXiv ID:** 2606.18596 | [PDF](https://arxiv.org/pdf/2606.18596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 185. Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation

**arXiv ID:** 2606.18594 | [PDF](https://arxiv.org/pdf/2606.18594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 186. APT: Atomic Physical Transitions for Causal Video-Language Understanding

**arXiv ID:** 2606.18586 | [PDF](https://arxiv.org/pdf/2606.18586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 187. Aerial-ground LiDAR place recognition with patch-level self-supervised learning and expanded reciprocal re-ranking

**arXiv ID:** 2606.18583 | [PDF](https://arxiv.org/pdf/2606.18583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 188. Tangent Spheres and Integer Distances

**arXiv ID:** 2606.18569 | [PDF](https://arxiv.org/pdf/2606.18569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 189. Multi-Modal Hyper-Graph Fusion for Low-Light Crowd Counting

**arXiv ID:** 2606.18566 | [PDF](https://arxiv.org/pdf/2606.18566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 190. Principal Component Analysis and Power Indices

**arXiv ID:** 2606.18559 | [PDF](https://arxiv.org/pdf/2606.18559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 191. DeFAb: A Verifiable Benchmark for Defeasible Abduction in Foundation Models

**arXiv ID:** 2606.18557 | [PDF](https://arxiv.org/pdf/2606.18557v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 192. PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding

**arXiv ID:** 2606.18624 | [PDF](https://arxiv.org/pdf/2606.18624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 193. Rethinking the Pointer Loss in Table Structure Recognition: Geometry-Aware Pointer Loss for Spatial Locality

**arXiv ID:** 2606.18721 | [PDF](https://arxiv.org/pdf/2606.18721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 194. Human-AI Agent Interaction in a Business Context

**arXiv ID:** 2606.18716 | [PDF](https://arxiv.org/pdf/2606.18716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 195. Closed-Form and Constant-Time New-Source Selection for Fault-Tolerant Broadcasting in Dense Eisenstein--Jacobi Networks

**arXiv ID:** 2606.18714 | [PDF](https://arxiv.org/pdf/2606.18714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 196. Leveraging Energy Features for Surface Classification with Deep Learning: A Comparative Analysis Across Three Independent Datasets

**arXiv ID:** 2606.18698 | [PDF](https://arxiv.org/pdf/2606.18698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 197. Compact multi-text index for circular Cartesian tree matching

**arXiv ID:** 2606.18696 | [PDF](https://arxiv.org/pdf/2606.18696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 198. Through the WordStream Glass: Revisiting Quantitative Encoding for Qualitative Learning Analytics

**arXiv ID:** 2606.18692 | [PDF](https://arxiv.org/pdf/2606.18692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 199. Multi-Class Brain Tumor Classification Using Advanced Deep Learning Models: A Comparative Study

**arXiv ID:** 2606.18682 | [PDF](https://arxiv.org/pdf/2606.18682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. High-Degree-of-Freedom Lightweight Bioinspired Leg for Enhanced Mobility in Small Robots

**arXiv ID:** 2606.18680 | [PDF](https://arxiv.org/pdf/2606.18680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 201. Image Prompt Reconstruction Attacks on Distributed MLLM Inference Frameworks

**arXiv ID:** 2606.18710 | [PDF](https://arxiv.org/pdf/2606.18710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 202. PEFT-MedSAM: Efficient Fine-Tuning of Medical Foundation Models for Explainable Skin Lesion Segmentation

**arXiv ID:** 2606.18707 | [PDF](https://arxiv.org/pdf/2606.18707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 203. Dual-Channel Grounded World Modeling (DCGWM): Structural Prevention of Objective Interference Collapse via Heterogeneous External Grounding with Inward-Only Gradient Flow

**arXiv ID:** 2606.18688 | [PDF](https://arxiv.org/pdf/2606.18688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 204. Understanding and Mitigating Prompt Leaking Attacks in Real-World LLM-Based Applications

**arXiv ID:** 2606.18673 | [PDF](https://arxiv.org/pdf/2606.18673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 205. HANSEL: Extracting Breadcrumbs from Web Agent Trajectories for Interactive Verification

**arXiv ID:** 2606.18671 | [PDF](https://arxiv.org/pdf/2606.18671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 206. Covert Blockwise Coding with Sequential Detection over Thermal-Loss Bosonic Channels

**arXiv ID:** 2606.18666 | [PDF](https://arxiv.org/pdf/2606.18666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 207. RegMix-D: Dynamic Data Mixing via Proxy Training Trajectories

**arXiv ID:** 2606.18663 | [PDF](https://arxiv.org/pdf/2606.18663v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 208. A Scalable Embodied Intelligence Platform for Seamless Real-to-Sim-to-Real Transfer of Household Mobile Manipulation Tasks

**arXiv ID:** 2606.18646 | [PDF](https://arxiv.org/pdf/2606.18646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 209. Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish

**arXiv ID:** 2606.18717 | [PDF](https://arxiv.org/pdf/2606.18717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 210. Contextualizing Biological Language Models across Modalities via Logit-Space Contrastive Alignment

**arXiv ID:** 2606.18703 | [PDF](https://arxiv.org/pdf/2606.18703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 211. Closed-Form and Constant-Time New-Source Selection for Fault-Tolerant Broadcasting in Dense Gaussian Networks

**arXiv ID:** 2606.18715 | [PDF](https://arxiv.org/pdf/2606.18715v1)

**作者:** Bader Albader `[一作]` (Kuwait University), Bader Albader `[通讯]` (Kuwait University)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5068031485)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在密集高斯网络中实现故障容错广播的常数时间新源选择方法，改进了原先的边界搜索步骤；

**💡 创新点**

创新点在于构造了分数格子感知的计数公式以及偏移直接选择器，最多只需检查144种代数案例，时间复杂度为O(1)；

**🔧 技术方法**

采用了Gaussian整数与分数格子理论、边界相交分析、Cramer's 规则和区间端点选取等数学工具；

**📊 数据集**

使用了不同参数k（10、25、50、100、200）下的密集高斯网络，随机生成数千或数十万个节点对进行实验验证；

**📈 对比分析**

与全节点扫描O(N)和传统边界扫描O(k)方法比较，实验显示在大k时偏移直接选择器速度提升至5.92×，常数时间性能优异；

**⚠️ 局限性**

局限性包括仅适用于两节点故障场景，假设网络为G_k，未覆盖链路失效或动态故障；实现依赖于标准word‑RAM模型的算术常数时间假设。

---

## 212. UniTemp: Unlocking Video Generation in Any Temporal Order via Bidirectional Distillation

**arXiv ID:** 2606.18702 | [PDF](https://arxiv.org/pdf/2606.18702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 213. TW-LegalBench: Measuring Taiwanese Legal Understanding

**arXiv ID:** 2606.18699 | [PDF](https://arxiv.org/pdf/2606.18699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 214. Spatially Stratified Distillation for Heterogeneous Radar Place Recognition

**arXiv ID:** 2606.18687 | [PDF](https://arxiv.org/pdf/2606.18687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 215. Moving Beyond Diversity: Visual Token Pruning as Subspace Reconstruction for Efficient VLMs

**arXiv ID:** 2606.18681 | [PDF](https://arxiv.org/pdf/2606.18681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 216. EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems

**arXiv ID:** 2606.18668 | [PDF](https://arxiv.org/pdf/2606.18668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 217. Differential Equation Inductive Robustness Axiomatization

**arXiv ID:** 2606.18685 | [PDF](https://arxiv.org/pdf/2606.18685v1)

**作者:** André Platzer `[一作]` (Karlsruhe Institute of Technology), Long Qian `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4269 | [OpenAlex ID](https://openalex.org/A5023011601)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文针对带有多项式向量场的有限时段动力系统，构建了完整且可计算的鲁棒安全性逻辑推理框架；

**💡 创新点**

创新之处在于提出了“拓扑鲁棒性”(topological robustness)这一新的鲁棒安全概念，它既能保证系统的归纳性，又能在逻辑上实现完整性，克服了传统鲁棒性定义在无正分离边界情况下的不足；

**🔧 技术方法**

采用了子解析几何(Łojasiewicz不等式等)、可计算分析与差分动态逻辑(differential dynamic logic, dL)的符号推理技术，对有限时段内的集合进展性进行定量分析，从而实现了完整性证明；

**📊 数据集**

本工作并未使用实验数据集，全部为理论证明与形式化推理；

**📈 对比分析**

通过理论上证明可计算性与可决性（δ-可决性），与以往仅能得到半可判定或仅在有正分离的假设下完成的结果相比，性能更全面且能在更宽泛的情形下给出确切的安全判定；

**⚠️ 局限性**

局限性包括：只针对多项式向量场与半代数初始/安全集，时间段必须是有界；对非多项式或更一般的约束空间的推广尚未给出；理论证明的计算复杂度未作深入分析，实际推理实现仍需进一步优化。

---

## 218. Fair Online Resource Allocation

**arXiv ID:** 2606.18679 | [PDF](https://arxiv.org/pdf/2606.18679v1)

**作者:** Christopher En `[一作]` (Columbia University), Gonzalo Muñoz `[通讯]` (Universidad de Chile)

**通讯引用:** 318 | [OpenAlex ID](https://openalex.org/A5087687910)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了公平在线资源分配问题，特别是在难民安置和航空调度等应用中，提出了一种模型以最大化整体福利，同时满足资源约束和Lipschitz公平性要求。

**💡 创新点**

创新点在于提出了一种新的在线算法，基于双重镜像下降方法，能够在每个批次内强制执行公平性约束，并有效估计最优对偶变量。

**🔧 技术方法**

使用了双重镜像下降算法来实现在线公平资源分配，并通过构造性方法界定了公平性损失的界限。

**📊 数据集**

使用了来自难民经济计划的真实世界数据集，包括来自肯尼亚、乌干达和埃塞俄比亚的家庭调查数据。

**📈 对比分析**

与传统的在线资源分配算法相比，提出的算法在实现公平性约束的同时，能够在福利最大化方面表现良好，在线算法的期望后悔值为O(√(T))，在多次实验中表现出色。

**⚠️ 局限性**

限制在于算法在在线设置中必须在不知道未来到达的情况下做出不可逆的分配决策，这可能会影响整体资源的有效利用。

---

## 219. EFX Allocations Exist on Multi-Graphs

**arXiv ID:** 2606.18665 | [PDF](https://arxiv.org/pdf/2606.18665v1)

**作者:** Mahyar Afshinmehr `[一作]` (University of Oxford), Amir Mohammad Shahrezaei `[通讯]` (Sharif University of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明了在多重图（多条边）上，取消性（cancelable）估值模型中总能存在完全 EFX（公平至上）分配，并给出了多项式时间的构造算法。

**💡 创新点**

突破了先前仅适用于简单图或在特定结构/近似条件下的 EFX 存在性，首次在任意多重图上完成了 EFX 存在性证明，并将可计算性提升到多项式级别。

**🔧 技术方法**

核心技术包括：单元包（unit bundles）构造、贪心方向分配（greedy orientation）、关键路径与树分解（critical path & break tree）、树压缩（reduce trees）以及“倾倒”阶段（dumping phase），这些步骤共同实现了无环、无强制怨恨的分配。

**📊 数据集**

论文为理论性研究，不使用实验数据集；所有结论均通过严格的构造性证明与算法分析得到。

**📈 对比分析**

算法运行时间为多项式级别，具体实现中每一步均可在多项式时间内完成；与以往只能得到近似或在有限结构下存在性的结果相比，本工作实现了精确存在性与可计算性的统一。

**⚠️ 局限性**

局限性在于仅针对取消性估值（superclass of additive）有效；对于更一般的单调或次可加估值，是否存在 EFX 分配仍是开放问题。

---

## 220. NeuralMUSIC: A Hybrid Neural-Subspace Framework for Robot Sound Source Localization

**arXiv ID:** 2606.18664 | [PDF](https://arxiv.org/pdf/2606.18664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 221. BLADE: Scalable Bi-level Adaptive Data Selection for LLM Training

**arXiv ID:** 2606.18650 | [PDF](https://arxiv.org/pdf/2606.18650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 222. Spiking Pyramid Wavelet Transformation for High-efficient and Low-energy Image Restoration

**arXiv ID:** 2606.18644 | [PDF](https://arxiv.org/pdf/2606.18644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 223. TGCM: Topic-Guided Generative Disentanglement of Interleaved APT Technique Sequences

**arXiv ID:** 2606.18651 | [PDF](https://arxiv.org/pdf/2606.18651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 224. Gender Bias in LLM Hiring Decisions: Evidence from a Japanese Context and Evaluation of Mitigation Strategies

**arXiv ID:** 2606.18649 | [PDF](https://arxiv.org/pdf/2606.18649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 225. Trainable Photonic Measurement for Physics-Informed PDE Learning

**arXiv ID:** 2606.18713 | [PDF](https://arxiv.org/pdf/2606.18713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 226. Stealthy World Model Manipulation via Data Poisoning

**arXiv ID:** 2606.18697 | [PDF](https://arxiv.org/pdf/2606.18697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 227. ForecastBench-Sim: A Simulated-World Forecasting Benchmark

**arXiv ID:** 2606.18686 | [PDF](https://arxiv.org/pdf/2606.18686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 228. scGTN: Deep Siamese Graph Transformer Network for Single-cell RNA Sequencing Clustering

**arXiv ID:** 2606.18672 | [PDF](https://arxiv.org/pdf/2606.18672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 229. LandslideAgent with Multimodal LandslideBench: A Domain-Rule-Augmented Agent for Autonomous Landslide Identification and Analysis

**arXiv ID:** 2606.18661 | [PDF](https://arxiv.org/pdf/2606.18661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. Responsible ASR: Overcoming Challenges of Foundational Models in Narrow-Band and Low-Resource Settings

**arXiv ID:** 2606.18659 | [PDF](https://arxiv.org/pdf/2606.18659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 231. On-Manifold Variational Learning with Heat-Kernel Priors

**arXiv ID:** 2606.18658 | [PDF](https://arxiv.org/pdf/2606.18658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. Re-Rooting-Based Fault-Tolerant One-to-All Broadcasting in Dense Eisenstein--Jacobi Networks

**arXiv ID:** 2606.18712 | [PDF](https://arxiv.org/pdf/2606.18712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 233. Selective Unit-Cell Actuation in Lattice Structures for Distributed Morphology in Soft Robots

**arXiv ID:** 2606.18704 | [PDF](https://arxiv.org/pdf/2606.18704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 234. The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs

**arXiv ID:** 2606.18656 | [PDF](https://arxiv.org/pdf/2606.18656v1)

**作者:** Naihao Deng `[一作]` (University of Michigan), Yulong Chen `[通讯]` (University of Cambridge)

**通讯引用:** 3145 | [OpenAlex ID](https://openalex.org/A5100777439)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并研究了“误触对齐”（misfired alignment）——指在提示中出现种族、性别等刻板印象关键词时，已对齐的语言模型会覆盖上下文中的客观证据，导致错误回答；同时基于BBQ数据构建了2,032个对比问答对，定义了Misfired Alignment Rate (MAR) 指标；

**💡 创新点**

创新点在于①首次系统化定义并量化“误触对齐”现象；②设计MAR指标与对应基准数据集，能够在保持证据一致性的前提下检测模型在刻板印象场景下的偏差；③通过对齐前缀实验、层级logit镜像与注意力头消融揭示了误触对齐的因果机制和后期抑制过程；

**🔧 技术方法**

使用了对齐前缀实验、logit lens逐层跟踪、注意力头消融、对齐与基线模型对比等技术手段；

**📊 数据集**

利用BBQ（Behavioral Bias Benchmark）的消歧义子集，改造为二分类的对比问答，最终形成2,032条对比样本；

**📈 对比分析**

对25款公开与闭源LLM（包括GPT‑5.x、Claude、Llama、Qwen、Gemma、Gemini、Grok等）在零样本直接提示下进行评测；误触对齐率MAR介于4.7%–18.9%，强模型往往表现更差；人类实验显示0% MAR；对齐前缀实验进一步放大MAR，证实对齐指令能诱发此失误；

**⚠️ 局限性**

局限性包括：①仅采用二分类决策，未覆盖多样化生成、对话等场景；②基准来源仅为美国英语的BBQ，缺乏跨语言、跨文化与交叉身份的测试；③机制分析仅对开放权重模型完成，闭源模型无法直接验证；③数据集可能因二元化而降低对实际多元推理的代表性。

---

## 235. Attention as Frustrated Synchronization

**arXiv ID:** 2606.18694 | [PDF](https://arxiv.org/pdf/2606.18694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 236. Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-promoting Fine-tuning

**arXiv ID:** 2606.18691 | [PDF](https://arxiv.org/pdf/2606.18691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 237. Bounded Context Management for Tabular Foundation Models on Stream Learning

**arXiv ID:** 2606.18677 | [PDF](https://arxiv.org/pdf/2606.18677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 238. InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search

**arXiv ID:** 2606.18676 | [PDF](https://arxiv.org/pdf/2606.18676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. BrainFusionNet: a deep learning and XAI model to understand local, global, and sequential features of MRI images for improved brain tumour detection

**arXiv ID:** 2606.18675 | [PDF](https://arxiv.org/pdf/2606.18675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. On (Non-)Isomorphism of Self-Dual Lattices and Codes

**arXiv ID:** 2606.18662 | [PDF](https://arxiv.org/pdf/2606.18662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 241. LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels: A Study of Item Discrimination in Reading Comprehension Assessment

**arXiv ID:** 2606.18709 | [PDF](https://arxiv.org/pdf/2606.18709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 242. Maturing Markov Decision Processes: Decision Making under Increasing Information and Shrinking Action Sets

**arXiv ID:** 2606.18820 | [PDF](https://arxiv.org/pdf/2606.18820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. SwitchBraidNet: Quantisation-Aware Lightweight Architecture for Hybrid Brain-Computer Interface

**arXiv ID:** 2606.18816 | [PDF](https://arxiv.org/pdf/2606.18816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 244. LensKit-Auto

**arXiv ID:** 2606.18814 | [PDF](https://arxiv.org/pdf/2606.18814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 245. Beyond Scalar Scores: Exploring LLM-based Metrics for Clinical Significance Evaluation in Radiology Reports

**arXiv ID:** 2606.18797 | [PDF](https://arxiv.org/pdf/2606.18797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 246. Opinion Polarization in LLM-Based Social Networks: Manipulation and Mitigation

**arXiv ID:** 2606.18795 | [PDF](https://arxiv.org/pdf/2606.18795v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 247. Lost in a Single Vector: Improving Long-Document Retrieval with Chunk Evidence Aggregation

**arXiv ID:** 2606.18781 | [PDF](https://arxiv.org/pdf/2606.18781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 248. Fuzzy-Geometric Branch-Point Modeling for Structure-Aware Augmentation of Handwritten Chinese Characters

**arXiv ID:** 2606.18793 | [PDF](https://arxiv.org/pdf/2606.18793v1)

**作者:** Dongbin Jiao `[一作]` (Lanzhou University), Shi Yan `[通讯]` (Lanzhou University)

**通讯引用:** 11412 | [OpenAlex ID](https://openalex.org/A5113821982)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于模糊几何的分支点建模和结构感知的写字增强框架FGSA，用模糊集合对笔画分支点进行连续化判定，并通过贝塞尔曲线对分割后笔画进行重构与变形，从而实现结构保持的高质量合成样本。

**💡 创新点**

核心创新在于将分支点判定转化为模糊集合问题，结合拓扑邻域与方向场散度融合构造双源模糊成员函数；通过无监督的 surrogate 目标与差分进化全局搜索自适应确定模糊阈值；以及在贝塞尔曲线重构中加入多策略扰动与形态恢复，兼顾结构保真与多样性。

**🔧 技术方法**

采用模糊集合理论、方向场散度、贝塞尔曲线参数化、差分进化（DE）无监督优化、图像后处理（形态学闭开）等技术。

**📊 数据集**

使用CASIA‑HWDB1.1（标准手写汉字）、ChiSig（签名带背景干扰）和自行构建的LZUSig（高退化签名）三大数据集进行实验。

**📈 对比分析**

与传统几何变换（Affine、TPS）、生成式模型（zi2zi）以及细粒度结构增强（FgAA）等方法比较。实验表明，在标准数据集上FGSA提升0.12%准确率；在高退化数据集上ΔWER分别提升44.39%（ChiSig）和44.81%（LZUSig），且结构保真指标R_f和信息保真IP均保持在合理区间，整体表现优于所有基线。

**⚠️ 局限性**

主要局限包括：对极端笔画交叉、严重退化或噪声样本的结构恢复仍有限；模糊参数需针对每个样本单独优化，计算量较大；生成的样本多受结构约束，纹理多样性受限。

---

## 249. HALOMI: Learning Humanoid Loco-Manipulation with Active Perception from Human Demonstrations

**arXiv ID:** 2606.18772 | [PDF](https://arxiv.org/pdf/2606.18772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 250. A Predictive Neural Network Architecture for Early Detection of Low-Rate Cyberattacks

**arXiv ID:** 2606.18771 | [PDF](https://arxiv.org/pdf/2606.18771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 251. An open-source implementation and validation of 5G NR Configured Grant for URLLC in ns-3 5G LENA: a scheduling case study in Industry 4.0 scenarios

**arXiv ID:** 2606.18763 | [PDF](https://arxiv.org/pdf/2606.18763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 252. 5G UE and Network Asset Administration Shells for the Integration of 5G and Industry 4.0 Systems

**arXiv ID:** 2606.18762 | [PDF](https://arxiv.org/pdf/2606.18762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 253. Low-Cost Neuromorphic Fall Detection Using Synthetic Event Data and Hybrid SNNs

**arXiv ID:** 2606.18732 | [PDF](https://arxiv.org/pdf/2606.18732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 254. ReMP: Low-Downtime Runtime Model-Parallelism Reconfiguration for LLM Serving

**arXiv ID:** 2606.18741 | [PDF](https://arxiv.org/pdf/2606.18741v1)

**作者:** Haipeng Yuan `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Daning Cheng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ReMP，一套在LLM推理服务中实现低停机时间的模型并行重配置框架；

**💡 创新点**

核心创新在于将模型权重、KV缓存、通信组和工作进程与TP/PP拓扑解耦，提供两维KV缓存迁移、预构建通信快照、共享内存权重重装及工作进程生存期管理，实现在线、秒级的拓扑切换；

**🔧 技术方法**

采用共享内存权重存储、两维KV缓存迁移策略、预生成的MPU状态快照、工作进程复用、模型层与头维度的动态映射、异步P2P迁移、与vLLM深度集成等技术；

**📊 数据集**

使用四个大模型（Llama‑7B、Llama‑70B、Qwen3‑30B‑A3B、DeepSeek‑R1‑Distill‑Qwen‑32B）与BurstGPT生成的请求流作为实验数据；

**📈 对比分析**

与传统重启式切换以及固定TP/PP（TP1PP8、TP2PP4）基线对比；指标包括重启与ReMP切换时间、加速比、TTFT、TPOT、输出吞吐；结果显示ReMP在H100和RTX 5090平台上大多数切换在1–3 s内完成，重启加速比达数十倍甚至百倍；在动态负载下，ReMP在TTFT、TPOT上均优于固定配置，吞吐量提升显著；

**⚠️ 局限性**

局限性包括：需要预先构建可选拓扑快照，无法支持任意实时生成的拓扑；共享内存权重存储对系统内存容量有要求；当目标拓扑KV容量不足时仍需丢弃请求；迁移与加载仍有一定开销，无法在极低延迟场景下完全消除；

---

## 255. Generating Natural and Expressive Robot Gestures through Iterative Reinforcement Learning with Human Feedback using LLMs

**arXiv ID:** 2606.18747 | [PDF](https://arxiv.org/pdf/2606.18747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 256. Reinforcement Learning Foundation Models Should Already Be A Thing

**arXiv ID:** 2606.18812 | [PDF](https://arxiv.org/pdf/2606.18812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 257. Learning Augmented Exact Exponential Algorithms

**arXiv ID:** 2606.18807 | [PDF](https://arxiv.org/pdf/2606.18807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 258. R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.18786 | [PDF](https://arxiv.org/pdf/2606.18786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 259. SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection

**arXiv ID:** 2606.18783 | [PDF](https://arxiv.org/pdf/2606.18783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 260. RouteJudge: An Open Platform for Reproducible and Preference-Aware LLM Routing

**arXiv ID:** 2606.18774 | [PDF](https://arxiv.org/pdf/2606.18774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 261. SMART: A Flexible, Interpretable, and Scalable Spatio-temporal Brain Atlas from High-Resolution Imaging Data

**arXiv ID:** 2606.18753 | [PDF](https://arxiv.org/pdf/2606.18753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 262. What Must Generalist Agents Remember?

**arXiv ID:** 2606.18746 | [PDF](https://arxiv.org/pdf/2606.18746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 263. GRIDEX: Grid-Grounded Forensic Explanations for Deepfake Spectrogram Analysis

**arXiv ID:** 2606.18738 | [PDF](https://arxiv.org/pdf/2606.18738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 264. SWE-Future: Forecast-Conditioned Data Synthesis for Future-Oriented Software Engineering Agents

**arXiv ID:** 2606.18733 | [PDF](https://arxiv.org/pdf/2606.18733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 265. Clinically Aligned Geometry Constraints for Robust IVUS Vessel Boundary Segmentation

**arXiv ID:** 2606.18723 | [PDF](https://arxiv.org/pdf/2606.18723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. Robustness Analysis of Australia's Internet Using a Multilayer Network Model

**arXiv ID:** 2606.18737 | [PDF](https://arxiv.org/pdf/2606.18737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 267. Rescaling MLM-Head for Neural Sparse Retrieval

**arXiv ID:** 2606.18811 | [PDF](https://arxiv.org/pdf/2606.18811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 268. ProfiLLM: Utility-Aligned Agentic User Profiling for Industrial Ride-Hailing Dispatch

**arXiv ID:** 2606.18803 | [PDF](https://arxiv.org/pdf/2606.18803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 269. Closing the Loop: PID Feedback Control for Interpretable Activation Steering in Symbolic Music Generation

**arXiv ID:** 2606.18790 | [PDF](https://arxiv.org/pdf/2606.18790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 270. HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space

**arXiv ID:** 2606.18788 | [PDF](https://arxiv.org/pdf/2606.18788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Learned Radius Estimation for UDF-Based Point Cloud Reconstruction

**arXiv ID:** 2606.18787 | [PDF](https://arxiv.org/pdf/2606.18787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. RedactionBench

**arXiv ID:** 2606.18782 | [PDF](https://arxiv.org/pdf/2606.18782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 273. Online Distributional Prediction via Latent Cluster Geometry Under Drift and Corruption

**arXiv ID:** 2606.18778 | [PDF](https://arxiv.org/pdf/2606.18778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 274. Direct-V2X Support with 5G Network-based Communications: Performance, Challenges and Solutions

**arXiv ID:** 2606.18764 | [PDF](https://arxiv.org/pdf/2606.18764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 275. LegalWorld: A Life-Cycle Interactive Environment for Legal Agents

**arXiv ID:** 2606.18728 | [PDF](https://arxiv.org/pdf/2606.18728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 276. Graph Grounded Cross Attention Transformer Neural Network for Structurally Constrained Full Event Sequence Generation in Predictive Process Monitoring

**arXiv ID:** 2606.18726 | [PDF](https://arxiv.org/pdf/2606.18726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 277. SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieval

**arXiv ID:** 2606.18801 | [PDF](https://arxiv.org/pdf/2606.18801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 278. Bayesian Anytime Pareto Set Identification for Multi-Objective Multi-Armed Bandits

**arXiv ID:** 2606.18785 | [PDF](https://arxiv.org/pdf/2606.18785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 279. SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource Multimodal Information Extraction

**arXiv ID:** 2606.18780 | [PDF](https://arxiv.org/pdf/2606.18780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 280. Output Vector Editing for Memorization Mitigation in Large Language Models

**arXiv ID:** 2606.18767 | [PDF](https://arxiv.org/pdf/2606.18767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 281. SpectralDiT: Timestep-Conditioned Spectral Residual Correction for Flow-Matching DiTs

**arXiv ID:** 2606.18765 | [PDF](https://arxiv.org/pdf/2606.18765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 282. Toward Training-Free Zero-Shot Anomaly Detection in 3D Medical Images: A Batch-Based Approach Using 2D Foundation Models

**arXiv ID:** 2606.18749 | [PDF](https://arxiv.org/pdf/2606.18749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 283. Two-Phase Bilevel Search for the Moving-Target Traveling Salesman Problem with Moving Obstacles

**arXiv ID:** 2606.18730 | [PDF](https://arxiv.org/pdf/2606.18730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 284. Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2606.18810 | [PDF](https://arxiv.org/pdf/2606.18810v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. Private Learning with Public Feature Conditioning

**arXiv ID:** 2606.18773 | [PDF](https://arxiv.org/pdf/2606.18773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 286. A Neural Network Framework for Geodesic-Like Curve Computation on Parametric Surfaces

**arXiv ID:** 2606.18759 | [PDF](https://arxiv.org/pdf/2606.18759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 287. REVES: REvision and VErification--Augmented Training for Test-Time Scaling

**arXiv ID:** 2606.18910 | [PDF](https://arxiv.org/pdf/2606.18910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 288. SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration

**arXiv ID:** 2606.18902 | [PDF](https://arxiv.org/pdf/2606.18902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 289. Skill-Guided Continuation Distillation for GUI Agents

**arXiv ID:** 2606.18890 | [PDF](https://arxiv.org/pdf/2606.18890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 290. LARE: Low-Attention Region Encoding for Text-Image Retrieval

**arXiv ID:** 2606.18885 | [PDF](https://arxiv.org/pdf/2606.18885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 291. Performance Gap Analysis between Latin and Arabic Scripts HTR

**arXiv ID:** 2606.18884 | [PDF](https://arxiv.org/pdf/2606.18884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 292. Domain-Shift Aware Neural Networks for Unbalance Characterization in Rotating Systems

**arXiv ID:** 2606.18882 | [PDF](https://arxiv.org/pdf/2606.18882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 293. Bridging Single Distortion Artifacts and Mmultifactorial Clinical Quality: Few-shot Biparametric MRI Quality Assessment via Distortion-trained Prototypical Networks

**arXiv ID:** 2606.18872 | [PDF](https://arxiv.org/pdf/2606.18872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 294. Quantification of Uncertainty with Adversarial Models in Medical Image Segmentation

**arXiv ID:** 2606.18860 | [PDF](https://arxiv.org/pdf/2606.18860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 295. Seed-Guided Semi-Supervised Clustering by A-Contrario Anomaly Detection

**arXiv ID:** 2606.18833 | [PDF](https://arxiv.org/pdf/2606.18833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 296. Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning

**arXiv ID:** 2606.18831 | [PDF](https://arxiv.org/pdf/2606.18831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 297. BindEdit: Taming Attention Leakage for Precise Multi-Object Image Editing

**arXiv ID:** 2606.18906 | [PDF](https://arxiv.org/pdf/2606.18906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. Externalizing Research Synthesis and Validation in AI Scientists through a Research Harness

**arXiv ID:** 2606.18874 | [PDF](https://arxiv.org/pdf/2606.18874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 299. URDF Synthesis from RGB-D Sequences via Differentiable Joint Inference and Energy-Consistent Verification

**arXiv ID:** 2606.18861 | [PDF](https://arxiv.org/pdf/2606.18861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 300. Approximate Structured Diffusion for Sequence Labelling

**arXiv ID:** 2606.18856 | [PDF](https://arxiv.org/pdf/2606.18856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 301. Aligning Implied Statements for Implicit Hate Speech Generalizability with Context-Bounded Semi-hard Negative Mining

**arXiv ID:** 2606.18852 | [PDF](https://arxiv.org/pdf/2606.18852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 302. From Bounding Boxes to Visual Reasoning: An On-Policy Data Annotation Tool for Vision-Language Models

**arXiv ID:** 2606.18846 | [PDF](https://arxiv.org/pdf/2606.18846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 303. DINO-Med3D: Bridging Dimension and Domain Gaps in Volumetric Segmentation via Progressive Adaptation

**arXiv ID:** 2606.18886 | [PDF](https://arxiv.org/pdf/2606.18886v1)

**作者:** Haoyu Hu `[一作]` (University of Chinese Academy of Sciences), Zeng-Guang Hou `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

将2D预训练的DINOv3迁移到3D医学分割任务，提出两阶段的进阶适配框架。

**💡 创新点**

创新点包括：多切片嵌入（ICE）以引入伪3D上下文、用于域对齐的分割代理任务、冻结主干后加入轻量级3D适配器与LoRA以实现体素级连贯性、并设计并行的高频细节恢复流和自适应门控融合机制。

**🔧 技术方法**

采用DINOv3编码器、ICE模块、LoRA、3D适配器、Adaptive Gated Fusion（AGF）与Detail-Recovery Stream，并使用UperNet解码器。

**📊 数据集**

使用五个公开数据集：AISD（脑梗死）、MSD-Pancreas（胰腺）、MSD-Colon（结肠）、BraTS 2020（脑肿瘤）和ACDC（心脏）。

**📈 对比分析**

与nnU-Net、nnFormer、SwinUNETR、U-Mamba、VoCo和Dino U-Net等基线比较，DINO-Med3D在DSC和HD95指标上均优于对手，且在不同规模模型（Small/Base/Large）间表现出良好的规模规律。

**⚠️ 局限性**

局限性包括：细节恢复流在纹理复杂或病灶尺寸小的案例中效果不稳定；第一阶段需要对主干进行完整微调，计算成本较高，未来需要探索更高效的对齐策略。

---

## 304. Environment-Aware Resource Allocation for Pinching-Antenna-Assisted EDMA-NOMA Systems

**arXiv ID:** 2606.18899 | [PDF](https://arxiv.org/pdf/2606.18899v1)

**作者:** Yaxuan Luo `[一作]` `[通讯]` (University of Manchester), Yaxuan Luo (University of Manchester)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对使用压缩天线的EDMA-NOMA系统，提出了一种利用异构环境图信息进行用户匹配与功率分配的资源调度框架。

**💡 创新点**

创新点在于：①将环境阻塞系数嵌入LoS概率模型，并构造平均LoS/NLoS有效大尺度信道增益；②通过环境信息同时考虑服务链路质量与跨区域干扰；③设计了基于效用的联合用户匹配与功率分配算法（UBA‑JPPA），并加入SIC一致性惩罚和局部交换搜索。

**🔧 技术方法**

主要技术包括：指数型LoS概率模型、平均LoS/NLoS有效信道模型、基于效用的离散功率搜索、贪心匹配+局部交换、SIC残余干扰建模。

**📊 数据集**

使用仿真数据：M=4个EDMA区域、K=16个候选用户，场景为走廊/货架布局，设置不同阻塞系数（β_open=0.15、β_normal=1、β_blocked=5）及全局阻塞强度ϕ=0.02，随机生成用户位置并多次Monte Carlo仿真。

**📈 对比分析**

与三种基线（无环境图的UBA‑JPPA、传统NOMA‑JPPA、随机配对+固定功率）以及纯EDMA‑OMA进行对比。评估指标为系统总吞吐量、已调度用户的Jain公平性和吞吐‑公平性操作区域。结果显示：UBA‑JPPA在吞吐量、公平性以及操作区域上均优于无环境图的方案和传统NOMA方案；在高信噪比下，纯EDMA‑OMA仍可与其竞争，说明系统优势受参数与SIC条件影响。

**⚠️ 局限性**

主要局限：采用静态环境图和平均大尺度模型，未考虑小尺度衰落、波导传播损耗、天线激活位置和方向性；对高速移动、动态阻塞场景的适用性有限；未来工作需研究鲁棒分配、动态环境估计和联合天线激活优化。

---

## 305. Learning Robust Pair Confidence for Multimodal Emotion-Cause Pair Extraction

**arXiv ID:** 2606.18893 | [PDF](https://arxiv.org/pdf/2606.18893v1)

**作者:** Zhuangzhuang Pan `[一作]` (Universiti Malaya), Yan Xia `[通讯]` (Suzhou University of Technology)

**通讯引用:** 24845 | [OpenAlex ID](https://openalex.org/A5100404173)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种训练阶段的鲁棒对偶置信学习框架（RPCL），通过行条件分离和受损上下文稳定两项约束提升多模态情绪-因果对提取的pair置信度；

**💡 创新点**

创新点在于：①在行内（同一情绪下）使用自适应margin ranking将金对与硬负对分隔；②通过部分置零的上下文腐败并对齐清洁与腐败预测，使模型对非金上下文噪声保持稳定；这两项约束在训练阶段实现，无需在推理时增加任何模块；

**🔧 技术方法**

使用的技术包括：top‑k硬负采样、因果置信度自适应margin、行条件margin ranking损失、表示层的随机置零腐败、pair分布对齐损失、交叉熵等；

**📊 数据集**

实验数据集：ECF（英文）、MECAD（中文）以及另一中文多模态基准，共同构成完整的文本-音频-视觉三模态ECPE评测；

**📈 对比分析**

与仅使用标准交叉熵的基线以及两种控制方案（固定margin ranking、utterance‑dropout consistency）进行对比，完整三模态（TAV）下RPCL平均提升Pair F1 2.58–2.83个百分点，AUPRC 1.6–2.6个百分点；在单模态或双模态设置下仍保持正向提升；与公开系统对比显示在大部分数据集上处于同类或更优位置；

**⚠️ 局限性**

局限性：①仅改进训练目标，无法直接提升更强的编码器/解码器架构；②腐败约束仅在表示层进行，无法覆盖ASR错误、帧缺失或领域漂移等真实噪声；③模型仅预测标注的情绪-因果链接，不能推断真实因果关系或用于高风险决策。

---

## 306. Improving Human-Robot Teamwork in Urban Search and Rescue Through Episodic Memory of Prior Collaboration

**arXiv ID:** 2606.18836 | [PDF](https://arxiv.org/pdf/2606.18836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 307. SAERec: Constructing Fine-grained Interpretable Intents Priors via Sparse Autoencoders for Recommendation

**arXiv ID:** 2606.18897 | [PDF](https://arxiv.org/pdf/2606.18897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 308. ZiMPedance: Impedance-Aware ZMP Modeling and Control for Payload Carrying with Quadruped Robots

**arXiv ID:** 2606.18883 | [PDF](https://arxiv.org/pdf/2606.18883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 309. Space Is Intelligence: Neural Semigroup Superposition for Riemannian Metric Generation

**arXiv ID:** 2606.18828 | [PDF](https://arxiv.org/pdf/2606.18828v1)

**作者:** Chenghao Xu `[一作]` (Hunan University), Chenghao Xu `[通讯]` (Hunan University)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5076739856)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过Encoder‑Router网络，利用固定的Lie代数生成器和半群叠加，生成场景条件下的Riemannian度量场，使得几何路径（测地线）即可完成碰撞避免。

**💡 创新点**

创新点在于把规划智能迁移到空间本身——度量场本身即为决策器；采用Lie代数的线性叠加与指数映射保证输出始终为SPD矩阵，实现零样本组合泛化。

**🔧 技术方法**

使用Sim(2)关键点框架、神经网络生成的框架、调制与基本系数三组参数，结合Lie代数求和与指数映射；对比传统搜索与潜在空间规划，验证几何解的有效性。

**📊 数据集**

在合成的二维平面点机器人环境中，训练仅使用一个包含两个障碍物的场景，测试覆盖0–6个障碍、不同位置与密度的100个随机场景。

**📈 对比分析**

与基准A*、传统潜在空间规划比较，模型在所有12个手工设计与100个随机场景中均实现零样本成功；正路径代价与负路径代价相隔3–5阶，表明度量能可靠区分碰撞与无碰撞轨迹。

**⚠️ 局限性**

局限性包括仅验证二维平面点机器人、需要已知关键点位置的人工输入；未处理高维配置空间、视觉感知与动态障碍物，需进一步扩展到实际机器人环境。

---

## 310. Tractable Gap-Constraint Languages for Complex Event Recognition

**arXiv ID:** 2606.18878 | [PDF](https://arxiv.org/pdf/2606.18878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 311. Test-Time Adaptation in Optical Coherence Tomography Using Trajectory-Aligned Time-Independent Flow

**arXiv ID:** 2606.18876 | [PDF](https://arxiv.org/pdf/2606.18876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 312. Investigating Inductive Biases for Machine Learning Emulation of Sudden Stratospheric Warmings in Idealised Isca Simulations

**arXiv ID:** 2606.18857 | [PDF](https://arxiv.org/pdf/2606.18857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 313. Toward Semantically-Seeded, Graph-Propagated Impact Analysis Across Software Artifacts: A Vision

**arXiv ID:** 2606.18855 | [PDF](https://arxiv.org/pdf/2606.18855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 314. ScholarSum: Student-Teacher Abstractive Summarization via Knowledge Graph Reasoning and Reflective Refinement

**arXiv ID:** 2606.18850 | [PDF](https://arxiv.org/pdf/2606.18850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 315. Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems

**arXiv ID:** 2606.18837 | [PDF](https://arxiv.org/pdf/2606.18837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 316. Target-confidence Recourse Using tSeTlin machines: TRUST

**arXiv ID:** 2606.18832 | [PDF](https://arxiv.org/pdf/2606.18832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 317. Identifying Structural Biases from Causal Mechanism Shifts

**arXiv ID:** 2606.18834 | [PDF](https://arxiv.org/pdf/2606.18834v1)

**作者:** Praharsh Nanavati `[一作]` (CISPA Helmholtz Center for Information Security), David Kaltenpoth `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种利用多环境下机制变化检测隐藏混淆和选择偏差的方法。

**💡 创新点**

创新点在于将机制变化的相互依赖性作为判别混淆与选择偏差的判据，并给出了基于互信息的可检验标准。

**🔧 技术方法**

使用非参数的机制变化检验（KCI、Kolmogorov‑Smirnov）与调整互信息（AMI）聚类来识别偏差，并在有观测因果图时构造结构偏差识别算法。

**📊 数据集**

实验数据包括合成Erdős‑Rényi DAG数据以及真实细胞信号传导的Sachs流式细胞术数据。

**📈 对比分析**

与五种基准方法（FCI, JCI‑FCI, CoCo, LS‑P, LS‑C）相比，该方法在三类偏差分类和受影响变量集恢复上取得了显著更高的F1/精确率/召回率，优势明显。

**⚠️ 局限性**

主要局限在于需满足Markov、faithfulness、稀疏独立机制变化等假设，且依赖足够多的i.i.d.样本与环境，无法直接处理循环依赖或时间序列。

---

## 318. Where Will They Go? Modelling Multimodal Pedestrian Manoeuvres from Ego-centric Videos

**arXiv ID:** 2606.18824 | [PDF](https://arxiv.org/pdf/2606.18824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 319. Anomaly Detection for Sparse and Irregular Multivariate Time Series with Latent SDEs

**arXiv ID:** 2606.18898 | [PDF](https://arxiv.org/pdf/2606.18898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. Compressed-Resident Genomics: Full-Pipeline Device-Resident GPU LZ77 Decode with Position-Invariant Random Access

**arXiv ID:** 2606.18900 | [PDF](https://arxiv.org/pdf/2606.18900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 321. Improving Medical Communication using Rubric-Guided Counterfactual Recommendations

**arXiv ID:** 2606.18889 | [PDF](https://arxiv.org/pdf/2606.18889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 322. Strategic Feature Selection

**arXiv ID:** 2606.18867 | [PDF](https://arxiv.org/pdf/2606.18867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 323. Scaling Learning-based AEB with Massive Unlabeled Data

**arXiv ID:** 2606.18864 | [PDF](https://arxiv.org/pdf/2606.18864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 324. WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents

**arXiv ID:** 2606.18847 | [PDF](https://arxiv.org/pdf/2606.18847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 325. Learning from Your Own Mistakes: Constructing Learnable Micro-Reflective Trajectories for Self-Distillation

**arXiv ID:** 2606.18844 | [PDF](https://arxiv.org/pdf/2606.18844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 326. Rethinking Air-Ground Collaboration: A Progressive Cross-Task Benchmark and Socialized Learning Framework

**arXiv ID:** 2606.18841 | [PDF](https://arxiv.org/pdf/2606.18841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 327. Automatic ply-specific analyses of CFRP micrographs using shortest-path-based ply distinction

**arXiv ID:** 2606.18894 | [PDF](https://arxiv.org/pdf/2606.18894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. Generative-Model Predictive Planning for Navigation in Partially Observable Environments

**arXiv ID:** 2606.18888 | [PDF](https://arxiv.org/pdf/2606.18888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 329. Efficient Financial Language Understanding via Distillation with Synthetic Data

**arXiv ID:** 2606.18875 | [PDF](https://arxiv.org/pdf/2606.18875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 330. Learning to Distort: Weakly-Supervised Image Quality Transfer for Prostate DWI Correction

**arXiv ID:** 2606.18869 | [PDF](https://arxiv.org/pdf/2606.18869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 331. Semantic Robustness Certification for Vision-Language Models

**arXiv ID:** 2606.18839 | [PDF](https://arxiv.org/pdf/2606.18839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 332. GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents

**arXiv ID:** 2606.18829 | [PDF](https://arxiv.org/pdf/2606.18829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 333. DreamReg: Belief-Driven World Model for 2D-3D Ultrasound Registration

**arXiv ID:** 2606.18825 | [PDF](https://arxiv.org/pdf/2606.18825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. FlowObject: Flow Steering for Bridging Generative Priors and Reconstruction Fidelity

**arXiv ID:** 2606.19019 | [PDF](https://arxiv.org/pdf/2606.19019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 335. TRAP: Benchmark for Task-completion and Resistance to Active Privacy-extraction

**arXiv ID:** 2606.18996 | [PDF](https://arxiv.org/pdf/2606.18996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 336. Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering

**arXiv ID:** 2606.18986 | [PDF](https://arxiv.org/pdf/2606.18986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 337. Nanoscale memristive devices: Threats and solutions

**arXiv ID:** 2606.18978 | [PDF](https://arxiv.org/pdf/2606.18978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 338. GraphPO: Graph-based Policy Optimization for Reasoning Models

**arXiv ID:** 2606.18954 | [PDF](https://arxiv.org/pdf/2606.18954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 339. Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos

**arXiv ID:** 2606.18955 | [PDF](https://arxiv.org/pdf/2606.18955v1)

**作者:** Runze Xu `[一作]` (Tsinghua University), Jincheng Yu `[通讯]` (Tsinghua University)

**通讯引用:** 2561 | [OpenAlex ID](https://openalex.org/A5112107807)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于潜在动作的预训练框架，利用无标签的人体第一视角视频提取可跨机器人种类的动作先验，并通过意图-感知解耦实现仅需约50条轨迹的快速下游适配。

**💡 创新点**

创新点在于构建混合解耦 VQ‑VAE 实现动作与背景的分离，使用物理掩码构造跨 embodiment 的离散动作代码表，并在下游引入意图与感知解耦，突破传统需要海量带标签机器人数据的瓶颈。

**🔧 技术方法**

采用混合解耦 VQ‑VAE、DINO v2 视觉编码器、Prismatic‑7B 视觉‑语言模型、流匹配动作头、LoRA 微调等技术。

**📊 数据集**

在预训练阶段使用 BridgeV2、EgoDex 等无标签人类视频；下游适配阶段在 LIBERO、RoboTwin 2.0 等机器人模拟及真实双臂平台上收集 50 条轨迹。

**📈 对比分析**

与 LAPA、Diffusion Policy、OpenVLA、SpatialVLA、pi0、villa‑x、UniVLA 等基线在 LIBERO（91.8%）和 RoboTwin 2.0（67.7%）平均成功率上相当或超越，证明仅凭无标签人类视频即可达到最先进水平。

**⚠️ 局限性**

局限在于离散动作代码表容量不足，难以处理高精度细粒度操作；未来计划探索多尺度潜在表示。

---

## 340. SP-TransientBench: A Real-Captured Single Photon Perception Benchmark

**arXiv ID:** 2606.18952 | [PDF](https://arxiv.org/pdf/2606.18952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 341. RTSGameBench: An RTS Benchmark for Strategic Reasoning by Vision-Language Models

**arXiv ID:** 2606.18950 | [PDF](https://arxiv.org/pdf/2606.18950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 342. As Easy as Rocket Science: Assessing the Ability of Large Language Models to Interpret Negation in Figurative Language

**arXiv ID:** 2606.18922 | [PDF](https://arxiv.org/pdf/2606.18922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 343. TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer

**arXiv ID:** 2606.18959 | [PDF](https://arxiv.org/pdf/2606.18959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 344. SciRisk-Bench: A Risk-Dimension-Aware Benchmark for AI4Science Safety

**arXiv ID:** 2606.18936 | [PDF](https://arxiv.org/pdf/2606.18936v1)

**作者:** Linghao Feng `[一作]` (Chinese Academy of Sciences), Yi Zeng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 10807 | [OpenAlex ID](https://openalex.org/A5108421411)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一个针对AI4Science安全的多维度基准SciRisk-Bench，覆盖7个学科、31个子学科与10个风险维度。

**💡 创新点**

引入风险维度与学科双轴分层，提供细粒度安全诊断，区别于以往只关注学科或宽泛安全类别的基准。

**🔧 技术方法**

采用LLM‑as‑a‑judge评估方法，对主流与科学专门化LLM进行攻击成功率(ASR)统计，并使用人工标注的风险维度定义与专家评判。

**📊 数据集**

SciRisk‑Bench本身，包含350条示例，覆盖7学科、31子学科与10风险维度；并参考ChemSafetyBench、LabSafetyBench等现有安全基准。

**📈 对比分析**

通过比较主流基础LLM与科学专门化LLM的ASR，发现后者在安全省略、知识更新漂移、实验室安全等多维度风险上通常更高，隐私泄漏等维度表现相对较好。

**⚠️ 局限性**

仅覆盖文本输入，缺乏多模态评估；风险定义为静态快照，可能随监管与技术演进而变化；未公开完整示例细节，存在误用风险。

---

## 345. Epistemic Pairwise Maximin Share

**arXiv ID:** 2606.18921 | [PDF](https://arxiv.org/pdf/2606.18921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 346. Spotlight: Synergizing Seed Exploration and Spot GPUs for DiT RL Post-Training

**arXiv ID:** 2606.19004 | [PDF](https://arxiv.org/pdf/2606.19004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 347. Enhancing Multilingual Reasoning via Steerable Model Merging

**arXiv ID:** 2606.19002 | [PDF](https://arxiv.org/pdf/2606.19002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 348. Monadic dependence from reducts, and applications to twin-width of oriented graphs

**arXiv ID:** 2606.18934 | [PDF](https://arxiv.org/pdf/2606.18934v1)

**作者:** Hector Buffière `[一作]` (Université Paris Cité), Patrice Ossona de Mendez `[通讯]` (Centre d'Analyse et de Mathématique Sociales)

**通讯引用:** 2908 | [OpenAlex ID](https://openalex.org/A5004397700)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了包含至少一条反对称关系的二元关系结构的单子依赖性，并给出了从若干良好可降维子结构推导整体单子依赖性的通用技术，随后将其应用于双曲宽度（twin-width）的判定与算法，可通过添加具有有限独立数或线性序的拓扑来扩展结构以保持单子依赖，从而证明这些扩展类的FO模型检查在参数化下可固定参数可解；此外，还证明了新的有向图类（如有向分裂图、局部锦标赛等）在twin-width下被分离（delineated）且FO模型检查可行。

**💡 创新点**

创新点在于：① 提出了在结构良好且兼容的子结构上判定整体单子依赖的充分条件，消除了传统需要检查全部结构的困难；② 通过“重定向”和“替换”方法证明在保持独立数有限时，单子依赖性可被保留；③ 将单子依赖与双曲宽度建立新的等价关系，扩展至使用有向图或部分序列的拓展；④ 发现并证明了更多有向图类（如局部锦标赛）在twin-width下被分离，进一步揭示了twin-width与单子依赖之间的深层联系。

**🔧 技术方法**

主要技术包括：结构化转移（transduction）与单子扩展、最小/完整变换器（transformer）理论、正则映射与阶类型（order type）分析、对齐与重定向技术、双曲宽度的收缩序列构造与压缩技术、以及多重子结构兼容性判定。

**📊 数据集**

由于本工作主要是理论性质，未使用特定实验数据集；但在证明局部锦标赛与有向分裂图可分离时，借助了已知的圆弧图和锦标赛的构造算法。

**📈 对比分析**

相较于传统的FO模型检查方法（需构造收缩序列且复杂度高），本文提供了通过单子依赖判定可直接获得固定参数可解的上界；在已知可以高效构造收缩序列的类（如有向图、锦标赛、圆弧图等）上，其算法与现有最优方法相当，且在更广泛的类上实现了理论上的可行性。

**⚠️ 局限性**

局限性包括：① 需要结构满足严格的兼容性与可降维性条件，实际结构可能不满足；② 对于一般有向图类，仍无法在多项式时间内构造收缩序列；③ 对于局部锦标赛的扩展与分离结果，尚未给出具体算法实现细节；④ 论文中许多证明为存在性证明，缺乏实用的构造性算法。

---

## 349. EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts

**arXiv ID:** 2606.18967 | [PDF](https://arxiv.org/pdf/2606.18967v1)

**作者:** Minseo Kim `[一作]` (FuriosaAI), Wonjun Kang `[通讯]` (FuriosaAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向强化学习回放的系统感知自我推测解码（Self‑Speculative Decoding, SD）框架，能够在不额外训练或适配的前提下显著加速大语言模型的回放生成。

**💡 创新点**

创新点包括：① 在每一步训练中直接从目标策略中量化（RTN 4‑bit）提取“自我”推测器，使得推测器与不断演化的策略保持同步；② 通过屋脊线（roofline）模型实现系统感知的 SD 开关，只在计算资源处于内存受限、能获得加速的批量规模时开启推测；③ 采用自适应推测长度策略，根据实时块效率动态调整推测块大小，充分利用策略收敛时推测器性能提升。

**🔧 技术方法**

使用技术：量化推测器（RTN 4‑bit）、自我推测解码（Self‑SD）、屋脊线模型用于预测加速边界、动态 SD 开关策略、块效率与采样率监控、vLLM/veRL集成与Marlin量化核。

**📊 数据集**

实验数据集包括 Qwen2.5‑7B/14B 在 SimpleRL‑8k‑hard 上，以及 Llama3.1‑8B 在 SimpleRL‑8k‑medium 上，使用基于 RLVR 的数学推理任务进行 100 步训练。

**📈 对比分析**

与四个基线比较：① 仅加速 AR 解码；② 基于回放历史的 Spec‑RL；③ 采用 EAGLE3 辅助推测器；④ 静态自我推测器（始终开启）。结果显示：自我推测框架在所有模型上将回放生成时延降低至约 30% 以内，整体训练步时延提升 20% 左右，且保持与原模型相同的奖励轨迹和验证准确性。

**⚠️ 局限性**

局限性：① 量化推测器在极大规模或极低温度的长尾生成中可能仍失效；② 需要针对每个硬件/模型对屋脊线模型进行一次校准；③ 目前仅针对单节点 A100 8 卡、on‑policy RL；④ 未涵盖树形验证、稀疏注意力等更高效的推测机制。

---

## 350. SenFlow: Inter-Sentence Flow Modeling for AI-Generated Text Detection in Hybrid Documents

**arXiv ID:** 2606.18946 | [PDF](https://arxiv.org/pdf/2606.18946v1)

**作者:** Jingkun Luo `[一作]` (Northwestern Polytechnical University), Guanxiong Pei `[通讯]` (Zhejiang Lab)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5081606235)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种句子级AI生成文本检测框架SenFlow，并构建了含推理型与聊天型模型混合写作的16,000篇文档基准MOSAIC。

**💡 创新点**

创新点在于将句子级检测视为结构化预测，结合图卷积网络的句子依赖传播与线性链CRF解码，实现对跨文档、跨生成器与跨领域的迁移鲁棒性。

**🔧 技术方法**

使用了LoRA对齐的Llama-3.1-8B-Instruct代理模型、双向交叉注意力融合、TCN序列编码、Hybrid Adjacency GCN以及CRF后验解码等技术。

**📊 数据集**

数据集为MOSAIC，涵盖PubMed摘要和XSum新闻摘要，混合生成器DeepSeek‑V3.2（推理型）和Kimi K2（聊天型），共16,000篇、约281k句。

**📈 对比分析**

与Fast‑DetectGPT、Binoculars、POGER、SeqXGPT及SenDetEX等基线相比，SenFlow在Unified、Cross‑Generator和Cross‑Domain三种难度协议下均实现了显著提升，特别是Cross‑Domain平均F1提升4.15个百分点。

**⚠️ 局限性**

局限包括仅覆盖两域两模型、固定30%AI比例、仅二分类标签、依赖对齐代理模型且未评估完全黑盒场景或后期扰动对抗等。

---

## 351. Show, Don't Ask: Generative Visual Disambiguation for Composed Image Retrieval with Turn-Valid Coverage

**arXiv ID:** 2606.18992 | [PDF](https://arxiv.org/pdf/2606.18992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 352. DIPHINE: Diffusion-based $Φ$-ID Neural Estimator

**arXiv ID:** 2606.18997 | [PDF](https://arxiv.org/pdf/2606.18997v1)

**作者:** Simon Pedro Galeano Munoz `[一作]` (King Abdullah University of Science and Technology), Maurizio Filippone `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3193 | [OpenAlex ID](https://openalex.org/A5021162375)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了DIPHINE，一种基于扩散模型的神经估计器，能够一次性估计连续系统中的九个互信息（MI）并通过 Möbius 逆推得到完整的十六原子 ΦID 结构。

**💡 创新点**

创新点在于：①首次将扩散模型的评分网络与掩码技术结合，实现对多源多目标信息动态的统一低成本估计；②通过理论分析证明 MI 误差在逆推过程中的整数 Jacobian 结构，从而阐明了同步→同步原子是最难估计的；③在连续非高斯系统上首次完成完整 ΦID 计算。

**🔧 技术方法**

采用了 score‑based diffusion（vpsde）与 Girsanov 定理估计 KL，单网络掩码学习九个 MI，再利用 Möbius 逆推生成十六原子；并在实验中使用 Gaussian VAR(1)、多维 VAR(1) 与 Fantasia 数据集。

**📊 数据集**

数据集包括：①合成 Gaussian VAR(1)（单/多维、耦合/无耦合）；②多维 bipartite VAR(1)（维度 3/5/10）；③真实心肺数据（Fantasia 数据库）中的 RR 及呼吸信号。

**📈 对比分析**

与 MINE、NWJ、InfoNCE、KSG 等传统 MI 估计器在 MI 级别训练九个独立模型后做对比；DIPHINE 在 MI MAE 与 ΦID 原子 MAE 上均优于基线，尤其在高维和涉及协同原子时误差显著降低；实验表明 Syn→Syn 原子误差最大，其它原子误差随维度升高而增大。

**⚠️ 局限性**

局限性包括：仅适用于平稳可测的 ergodic 系统；目前仅处理双分区（bivariate）系统，扩展到 N>2 需面对指数级原子数；使用 mmi 重复度函数时可能出现负原子；以及训练扩散模型所需的计算资源。

---

## 353. G-IdiomAlign: A Gloss-Pivoted Benchmark for Cross-Lingual Idiom Alignment

**arXiv ID:** 2606.18989 | [PDF](https://arxiv.org/pdf/2606.18989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 354. ThinkDeception: A Progressive Reinforcement Learning Framework for Interpretable Multimodal Deception Detection

**arXiv ID:** 2606.18988 | [PDF](https://arxiv.org/pdf/2606.18988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 355. Structure of kissing arrangements in ${\mathbb R}^{12}$ and a place for the $841$st sphere

**arXiv ID:** 2606.18984 | [PDF](https://arxiv.org/pdf/2606.18984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 356. Not Your Usual FFT: QFT$\rightarrow$FFT via Classical Quantum-Circuit Simulation

**arXiv ID:** 2606.18981 | [PDF](https://arxiv.org/pdf/2606.18981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 357. A Controlled Benchmark of Quantum-Latent GAN Augmentation for Brain MRI

**arXiv ID:** 2606.18970 | [PDF](https://arxiv.org/pdf/2606.18970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. Convergence of Replicator Dynamics in the Repeated Prisoner's Dilemma with Restarts

**arXiv ID:** 2606.18965 | [PDF](https://arxiv.org/pdf/2606.18965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 359. Online Reward-Punishment Learning from Fixed-Channel Perceptual Event Streams without Environment Rewards

**arXiv ID:** 2606.18963 | [PDF](https://arxiv.org/pdf/2606.18963v1)

**作者:** Zirong Li `[一作]` `[通讯]` (Tiangong University), Zirong Li (Tiangong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在没有环境提供的标量奖励或评估标签的情况下，在线奖励-惩罚学习。代理仅接收固定通道的感知数据包，并通过观察到的转移后果推断出这些感知维度的价值。

**💡 创新点**

提出了一种在线的逆强化学习/强化学习算法，明确区分了预测、残差动态、轨迹评估和策略学习，并通过无奖励协议进行学习。

**🔧 技术方法**

使用了神经自监督预测器、残差动态预测器和固定内部轨迹评估器等技术，构建了一个多模块的学习框架。

**📊 数据集**

使用了2x2-XOR数据包任务作为实验数据集，代理在没有环境奖励的情况下进行学习和评估。

**📈 对比分析**

与其他方法相比，B_ξ在2x2-XOR任务中达到了0.952的平衡奖励符号准确率，整体策略达到了0.979的最优动作准确率，显示出优越的性能。

**⚠️ 局限性**

限制在于该方法依赖于内部评估器的构建，且在某些情况下可能无法完全捕捉复杂的环境动态。

---

## 360. Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation

**arXiv ID:** 2606.18960 | [PDF](https://arxiv.org/pdf/2606.18960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 361. GrapNet: A Programmable Dynamic-Architecture Neural Graph Substrate

**arXiv ID:** 2606.18923 | [PDF](https://arxiv.org/pdf/2606.18923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 362. C-ARC: Continuous-Adaptive Range Clustering for Non-Repetitive LiDAR Sensors

**arXiv ID:** 2606.18948 | [PDF](https://arxiv.org/pdf/2606.18948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 363. Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents

**arXiv ID:** 2606.18947 | [PDF](https://arxiv.org/pdf/2606.18947v1)

**作者:** Emmanuel Aboah Boateng `[一作]` (DoorDash, Inc.), Sudeep Das `[通讯]` (DoorDash, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Decoupled Search Grounding（DSG）体系，使实时检索与大语言模型的推理完全解耦，并通过可配置的网关实现检索策略、缓存、回退和输出控制。

**💡 创新点**

创新点在于将检索抽象为独立的、可插拔的工具接口，提供源感知上下文渲染、精确与语义缓存、提供者抽象与回退链、检索深度调节等可配置控制，使得检索行为成为可观测、可调优的系统边界，而非模型内部的黑箱特性。

**🔧 技术方法**

技术实现基于 MCP 兼容的网关、结构化工具调用、嵌入式语义匹配缓存、精确/语义缓存策略、提供者注册与回退链、以及多模型交互的统一接口；评估时使用 LLM‑as‑judge、G‑Eval 等自动评判方法。

**📊 数据集**

使用了公共 QA 基准（SimpleQA、FreshQA、HotpotQA）以及内部电商查询意图理解（QIU）数据集（Retail 与 Tail Synthetic）。

**📈 对比分析**

采用固定提示、固定样本、任务级评分，并对原生检索与 DSG（BrightData、Serper 等）在准确率、每千查询成本、延迟、缓存命中率等维度进行对比。结果显示：在 SimpleQA 上 DSG 的准确率仅低 1.6% 但成本下降 91%；在 FreshQA 原生检索略优；在 QIU 上 DSG 匹配或超过原生检索，同时成本降低 98%；缓存命中率达 99.4%，平均延迟下降 68%。

**⚠️ 局限性**

局限性包括：对第三方搜索 API 的依赖（价格、速率、索引变化会影响性能）；在多跳推理任务（HotpotQA）上增益有限；评估主要依赖自动 LLM‑as‑judge，可能存在偏差；以及成本随 API 变更而波动。

---

## 364. Some Complexity Results for Robustness Verification for Binarized Neural Networks

**arXiv ID:** 2606.18918 | [PDF](https://arxiv.org/pdf/2606.18918v1)

**作者:** Harshit Goyal `[一作]` (Indian Institute of Technology Goa), Sudakshina Dutta `[通讯]` (Indian Institute of Technology Goa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了二值化神经网络（BNN）的可满足性与在均匀遮挡下的鲁棒性，并给出了相应的计算复杂度分析与算法。

**💡 创新点**

创新点在于：①证明BNN可满足性问题为NP-完全；②揭示在均匀遮挡情形下，网络输出对遮挡颜色是分段常数函数，从而设计出多项式时间的鲁棒性检测算法。

**🔧 技术方法**

主要技术包括：布尔可满足性问题的多项式时间归约、构造布尔逻辑与非与或门对应的BNN子网络、分段函数与断点分析、基于断点集合枚举的鲁棒性检验算法。

**📊 数据集**

文中未使用公开数据集进行实验；仅通过 MNIST 图像的示例演示算法原理。

**📈 对比分析**

由于缺乏实验评测，本文未给出与其他方法的性能对比，算法的复杂度已以多项式上界证明，理论上在给定输入尺寸和网络参数时可在多项式时间完成鲁棒性检查。

**⚠️ 局限性**

局限性包括：①仅讨论了均匀遮挡而非一般扰动；②只针对使用符号激活函数的BNN；③未提供实验验证，缺乏对实际模型鲁棒性的实证分析。

---

## 365. Physics-IQ Verified

**arXiv ID:** 2606.18943 | [PDF](https://arxiv.org/pdf/2606.18943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 366. Sumi: Open Uniform Diffusion Language Model from Scratch

**arXiv ID:** 2606.19005 | [PDF](https://arxiv.org/pdf/2606.19005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 367. PaSTTeL: Parallel analysiS framework for Termination and non-Termination of Lasso programs

**arXiv ID:** 2606.18977 | [PDF](https://arxiv.org/pdf/2606.18977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 368. CAPRA: Scaling Feedback on Software Architecture Deliverables with a Multi-Agent LLM System

**arXiv ID:** 2606.18976 | [PDF](https://arxiv.org/pdf/2606.18976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 369. LiveStack: OS Support for Cluster-Scale Full-Stack Live Simulation

**arXiv ID:** 2606.18958 | [PDF](https://arxiv.org/pdf/2606.18958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 370. Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement

**arXiv ID:** 2606.18953 | [PDF](https://arxiv.org/pdf/2606.18953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 371. A High-accuracy Event-based Underwater SLAM System

**arXiv ID:** 2606.18951 | [PDF](https://arxiv.org/pdf/2606.18951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 372. Urban Limits as Design Constraints: Identifying Suitable Locations for Distributed, Photovoltaic-Powered Servers

**arXiv ID:** 2606.18940 | [PDF](https://arxiv.org/pdf/2606.18940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 373. Who Wins the Conflict? Mechanistic Interpretability of Text Bias in Audio LLMs

**arXiv ID:** 2606.18924 | [PDF](https://arxiv.org/pdf/2606.18924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 374. Graph-ESBMC-PLC: Formal Verification of Graphical PLCopen XML Ladder Diagram Programs Using SMT-Based Model Checking

**arXiv ID:** 2606.18941 | [PDF](https://arxiv.org/pdf/2606.18941v1)

**作者:** Pierre Dantas `[一作]` (University of Manchester), Waldir Junior `[通讯]` (Federal University of Amazonas)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Graph-ESBMC-PLC，对 PLCopen XML 图形 Ladder Diagram 通过 DFS 解析连接图，生成完整 GOTO IR 并实现非空验证。

**💡 创新点**

创新点在于：① 用深度优先搜索从本地 ID/refLocalId 图构造 rung 路径并提取布尔表达式；② 通过右电源线（rightPowerRail）顺序保证 SET/RESET 的 IEC 61131-3 latch 语义；③ 采用三层 I/O 推断（地址、接触/线圈仅使用、默认内部状态）；④ 只改动前端解析代码，后端验证组件保持不变。

**🔧 技术方法**

使用技术包括：XML 解析、深度优先搜索（DFS）、PLCopen XML 格式规范、ESBMC 形式化验证框架、Z3 SMT 求解器、k‑induction 与 BMC 引擎、三层 I/O 推断算法。

**📊 数据集**

数据集：3 个来自 CONTROLLINO/OpenPLC Editor 的图形 LD 示例；11 个原始文本 LD 基准（保持不变）；2 个 Beremiz 例子作为失效案例。

**📈 对比分析**

比较方法：与原 ESBMC-PLC 的空 IR 结果对比，验证是否产生完整 IR 以及是否能在 k=2 通过 SAFE。结果显示所有 3 个图形程序在 70 ms 内完成，且 11 个文本基准保持零回归；原空 IR 的 vacuous 结果已被消除。

**⚠️ 局限性**

限制：仅支持联系人/线圈网络，功能块（计时器、触发器）语义被忽略；无法处理包含反馈环的图形 LD；对嵌套 LD 网络的检索不完整；评测规模仅 3 个图形程序，缺少故障示例。

---

## 375. Completeness for Probabilistic Boolean Tapes

**arXiv ID:** 2606.19017 | [PDF](https://arxiv.org/pdf/2606.19017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 376. On a variational model for phase transformation in SiO2 glass

**arXiv ID:** 2606.19021 | [PDF](https://arxiv.org/pdf/2606.19021v1)

**作者:** Sarah Dinkelacker-Steinhoff `[一作]` (Ruhr University Bochum), Klaus Hackl `[通讯]` (Ruhr University Bochum)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个变分框架，用来描述SiO₂玻璃在等温压缩条件下的体积相变与弹性模量变化，并通过有限元模拟验证模型对压缩、拉伸、剪切及带孔板的响应。

**💡 创新点**

创新点在于将等温体积相变视为宏观双相转变，将理想二元混合熵加入自由能并使用耗散距离形成即时能量损失项，从而在变分原理下得到可解析的相变演化方程，成功解释了玻璃压缩时的S形EOS与弹性模量软化。

**🔧 技术方法**

采用变分原理、Helmholtz自由能与耗散势、混合熵能量项、等温体积相变假设，并在Julia中使用Ferrite和Gmsh进行二维有限元求解。

**📊 数据集**

使用了Deschamps等人（2014）在DAC实验中得到的压缩比、体积模量与声速等实验数据作为模型参数和验证数据集。

**📈 对比分析**

通过数值实验与实验曲线对比，模型能够重现压缩比-压力、体积模量-压力、声速-压力以及泊松比-压力的典型曲线，性能与已知实验数据相符，展示了相变起始压力与软化最小点的合理预测。

**⚠️ 局限性**

局限性包括：未考虑剪切和非等温效应，忽略微观结构演化（如剪切带、缺陷），模型对热效应与卸载行为的预测仍需进一步完善，且仅在等温、体积相变驱动下适用。

---

## 377. Visual-OPSD: Cross-Modal On-Policy Self-Distillation for Efficient Unified Multimodal Reasoning

**arXiv ID:** 2606.18974 | [PDF](https://arxiv.org/pdf/2606.18974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. A Composable CRDT Layer for Byzantine-Resilient Deterministic Reconstruction

**arXiv ID:** 2606.18966 | [PDF](https://arxiv.org/pdf/2606.18966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 379. Be Your Own Teacher: Steering Protein Language Models via Unsupervised Reward Optimization

**arXiv ID:** 2606.18961 | [PDF](https://arxiv.org/pdf/2606.18961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 380. Urdu Katib Handwritten Dataset: A Historical Document Dataset for Offline Urdu Handwritten Text Recognition with CRNN-Based Baseline Evaluation

**arXiv ID:** 2606.19139 | [PDF](https://arxiv.org/pdf/2606.19139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 381. JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling

**arXiv ID:** 2606.19108 | [PDF](https://arxiv.org/pdf/2606.19108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 382. Zero-Shot Active Feature Acquisition via LLM-Elicitation

**arXiv ID:** 2606.18933 | [PDF](https://arxiv.org/pdf/2606.18933v1)

**作者:** Binyamin Perets `[一作]` (Technion), Shie Mannor `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个零样本主动特征获取框架，利用大语言模型仅提取可可信的单变量偏差和双变量协变（即MRF的充足统计量），通过最大熵闭包恢复类条件分布，随后采用经典规划算法实现二分类和top‑k识别，提出优先加权分配与簇/括号剪枝策略。

**💡 创新点**

创新点在于：① 将LLM知识获取与规划分离，避免LLM在顺序决策上的弱点；② 只获取对比统计（unary 与 pairwise），避免生成性失真；③ 通过最大熵闭包解决MRF比率不唯一性，实现无标签的类条件重构；④ 将二分类与top‑k统一到MRF框架并引入基于dueling bandit的优先加权分配；⑤ 通过实验量化LLM知识与规划之间的性能差距。

**🔧 技术方法**

技术手段包括：LLM（如GPT‑4）进行知识提取；Markov Random Field建模；最大熵（MaxEnt）闭包；Mean‑Field近似推理；条件互信息（CMI）与Wald/Chernoff‑Wald评分；dueling bandit启发的优先加权分配；Copeland计分用于top‑k排序；簇化与括号裁剪的主动选择机制。

**📊 数据集**

使用了炎症性肠病（IBD）患者的临床数据集：167例结肠活检样本，216个基因表达特征（离散{-1,0,+1}编码）和25个多标签临床表型。数据集包含47例“混乱型”患者，用于评估模型在最难情形下的表现。

**📈 对比分析**

与两种LLM基线（直接分类与带PubMed检索的分类）对比。二分类任务：在全观测下准确率1.00，LLM基线仅0.80/0.70；在临床标签上达到0.65/0.69；在top‑5识别任务中，规划器在t=150时P@5达到0.764（相较于0.528 baseline）并在“混乱型”患者上提升4–5个百分点。整体表现显著优于基线，证明规划与LLM知识分离带来的优势。

**⚠️ 局限性**

局限性：① 对LLM提取的知识质量高度依赖，幻觉或不完整导致MRF参数偏差；② Mean‑Field近似在某些样本上不收敛，影响推理；③ 只使用离散{-1,0,+1}特征，忽略连续幅度信息；④ 仅在IBD数据上验证，扩展到更大规模或不同领域的泛化性尚未测试。

---

## 383. Quantifying Compromise Risk in Exceptional Access Architectures Under Sparse and Indirect Evidence

**arXiv ID:** 2606.19106 | [PDF](https://arxiv.org/pdf/2606.19106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 384. ProductConsistency: Improving Product Identity Preservation in Instruction-Based Image Editing via SFT and RL

**arXiv ID:** 2606.19103 | [PDF](https://arxiv.org/pdf/2606.19103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 385. On the Notions of Bounded Bypass, and How to Make any Deadlock-Free MUTEX Protocol Satisfy One of Them

**arXiv ID:** 2606.19003 | [PDF](https://arxiv.org/pdf/2606.19003v1)

**作者:** Rob van Glabbeek `[一作]` (University of Edinburgh), Myrthe Spronck `[通讯]` (Eindhoven University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了改进后的“有界绕过”(Bounded Bypass)定义，并引入了“门后有界绕过(Post‑Doorway Bounded Bypass)”和“间歇有界绕过(Intermittent Bounded Bypass)”三种弱化形式，证明任何无死锁的互斥协议均可通过Bar‑David算法升级为满足这些有界绕过属性（在原子寄存器模型下的上界为二次，安全/常规寄存器模型下为n^2‑2）；

**💡 创新点**

创新点在于：①提供了严格且可判定的有界绕过定义；②首次系统性研究门后与间歇有界绕过，并给出它们与传统停机与饥饿自由的层次关系；③利用Bar‑David算法构造可实现这些性质的通用转换；④通过mCRL2模型检验验证了理论结果并发现原有定理的非紧致性；

**🔧 技术方法**

技术方法包括：形式化定义与证明、算法设计（Bar‑David方案）、对寄存器模型（原子、正则、可安全）进行细致分析、以及利用mCRL2的μ-算子模型检查器进行自动验证；

**📊 数据集**

本文未使用传统意义上的数据集，所有验证均基于模型检验（对2、3个进程的状态空间进行穷举）并在公开的GitHub/Zenodo存档中提供模型与公式；

**📈 对比分析**

比较方法通过构造层次图（如图所示）阐明不同有界绕过属性间的包含关系；性能方面给出了具体的绕过上界：原子寄存器下为n(n‑1)‑1，安全/常规寄存器下为n^2‑2；

**⚠️ 局限性**

限制包括：①模型检验仅覆盖极小规模（≤3进程），无法验证更大规模的行为；②结果在安全/常规寄存器模型下仍需要额外假设（如写操作不重叠）；③定义仍停留在理论层面，缺乏对实际实现的可操作性分析。

---

## 386. ChronoSurv: A Clinical Pathway-Guided Graph Framework for Multimodal Survival Analysis

**arXiv ID:** 2606.19140 | [PDF](https://arxiv.org/pdf/2606.19140v1)

**作者:** Hugo Miccinilli `[一作]` (Université Paris-Saclay), Theo Di Piazza `[通讯]` (University of Lyon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于临床路径的层次化有向图模型 ChronoSurv，用于头颈癌多模态生存分析。

**💡 创新点**

创新点在于将临床工作流程映射为时间有序的多层图结构，并通过异构消息传递捕获不同模态与临床步骤之间的非对称、进展依赖关系，同时天然支持缺失模态。

**🔧 技术方法**

采用多模态特征编码（图像视觉编码器、文本语言模型、表格投影）、层次化有向图构造、异构图注意力层进行消息传递、离散时间生存头以及差分式校准损失。

**📊 数据集**

使用两套公开头颈癌数据集（包含血液、文本报告、WSI 与表格数据）并构造多队列组合数据，评估跨机构缺失模态情况。

**📈 对比分析**

与传统统计模型、单模态深度学习、以及多模态深度学习（如 MMD、GraphMMP、HFBSurv 等）比较，ChronoSurv 在 C_index 上均居首位，IBS 也获得最佳或相当水平，并在 D-Calibration 上全部通过，显示出更优的风险排序与校准。

**⚠️ 局限性**

局限性包括对初始特征提取器未进行微调、对更大规模或不同恶性肿瘤的可扩展性待验证，并未探索更先进的视觉/语言预训练模型。

---

## 387. DVANet: Degradation-aware Visual-prior Alignment Network for Image Restoration

**arXiv ID:** 2606.19097 | [PDF](https://arxiv.org/pdf/2606.19097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 388. Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation

**arXiv ID:** 2606.19120 | [PDF](https://arxiv.org/pdf/2606.19120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 389. GCNGrasp-VP: Affordance-Guided View Planning for Efficient Task-Oriented Grasping

**arXiv ID:** 2606.19091 | [PDF](https://arxiv.org/pdf/2606.19091v1)

**作者:** Zanjia Tong `[一作]` (Southern University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了GCNGrasp-VP框架，将任务导向抓取模型GCNGrasp-v2与亲和力引导视角规划器结合，在遮挡环境下实现高效的任务导向抓取。

**💡 创新点**

创新点包括：①GCNGrasp-v2采用分割式网络实现抓取评估与亲和力场预测的常数时间推理；②亲和力引导视角规划器首次将亲和力场作为信息增益直接驱动视角选择，融入任务语义。

**🔧 技术方法**

使用技术包括图卷积网络、PointNet++、分割式架构、多点查询、亲和力场预测、DBSCAN聚类、贝叶斯优化权重、实时点云处理与视角优化。

**📊 数据集**

实验基于TaskGrasp数据集进行TOG评估，构造多视图观察数据集用于视角规划，并借助DepthAnything3、GroundedSAM、ContactGraspNet等工具生成所需数据。

**📈 对比分析**

与GauSS-MI、Active-NGF等基线对比，Affordance-VP在仅一次视角调整后即可达到近饱和的AP，并在单物体场景中将抓取成功率提升至100%，推理与规划时间仅约0.09 s，显著低于基线。

**⚠️ 局限性**

在极端遮挡情况下亲和力场预测误差导致视角选择不佳，难以充分捕获隐藏的关键部件，需要进一步提升监督信号以增强鲁棒性。

---

## 390. Adaptive Speech-to-Spike Encoding for Spiking Neural Networks

**arXiv ID:** 2606.19039 | [PDF](https://arxiv.org/pdf/2606.19039v1)

**作者:** Taharim Rahman Anon `[一作]` (PI LLC), Jakaria Islam Emon `[通讯]` (PI LLC)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了可学习的残差语音转尖峰编码器，并与R‑LIF骨干端到端联合训练，用以提升语音关键词识别性能

**💡 创新点**

创新点在于把固定阈值的 Step‑Forward 编码改为可学习的粗细步进，形成任务对齐的尖峰表示，并在同等参数规模下实现高精度，同时首次系统性比较 DFA 与 BPTT 的效果

**🔧 技术方法**

使用了可学习残差 Step‑Forward 编码、Recurrent Leaky Integrate‑and‑Fire (R‑LIF)骨干、两层 MLP 读出、surrogate‑gradient BPTT 与 Direct Feedback Alignment (DFA) 等技术

**📊 数据集**

基准数据集为 Google Speech Commands v2（35 类）

**📈 对比分析**

与固定编码和现有尖峰 KWS 系统对比，最大全局模型在 GSC‑v2 上达 94.97% 准确率；Tiny 模型（35k 参数）可达 89.8%；在相同条件下，DFA 达 91.5% 但落后 BPTT 94.97%，体现硬件友好学习规则的性能折衷

**⚠️ 局限性**

主要局限在于 DFA 精度仍低于 BPTT，能量估算仍较粗糙，并且模型对梯度反向传播的依赖仍较大

---

## 391. AMALIA-VL: A Native European Portuguese Open-Source Vision and Language Model

**arXiv ID:** 2606.19100 | [PDF](https://arxiv.org/pdf/2606.19100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 392. Pareto Q-Learning with Reward Machines

**arXiv ID:** 2606.19134 | [PDF](https://arxiv.org/pdf/2606.19134v1)

**作者:** Arnaud Lequen `[一作]` (Linköping University), Léo Saulières `[通讯]` (University of Toulouse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为Pareto Q-Learning with Reward Machines (PQLRM)的多目标强化学习算法，旨在处理由奖励机器（RMs）指定的奖励结构的任务。

**💡 创新点**

创新点在于将Pareto Q-Learning与奖励机器结合，利用奖励机器的结构化特性来提高学习效率，并能够在非马尔可夫的RM编码奖励下保持样本效率。

**🔧 技术方法**

使用了Pareto Q-Learning和Q-Learning with Reward Machines的技术，结合了多策略算法的特点。

**📊 数据集**

在Pressurized Bountiful Sea Treasure (PBST)和Office World环境中进行了实验，这些环境被建模为多目标马尔可夫决策过程（MOMDP）。

**📈 对比分析**

与基于交叉乘积的PQL基线和QRM进行了比较，实验结果表明PQLRM的收敛速度快于PQL，并且能够合成PQL无法实现的Pareto最优策略。

**⚠️ 局限性**

限制在于该方法的扩展性，未来的工作将集中在将该方法扩展到随机环境中。

---

## 393. Smoothness-Based Derandomization of PAC-Bayes Bounds

**arXiv ID:** 2606.19105 | [PDF](https://arxiv.org/pdf/2606.19105v1)

**作者:** Alexandre Lemire Paquin `[一作]` (Université Laval), Philippe Giguère `[通讯]` (Université Laval)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于PAC‑Bayes的平滑损失函数去随机化方法，并给出可用于线性模型和光滑神经网络的确定性泛化上界，同时基于Jacobian‑Hessian量级设计了一个针对BatchNorm网络的正则化器。

**💡 创新点**

创新点在于将去随机化视为对Jensen gap类的泛化问题，利用平滑性与Rademacher复杂度得到涉及参数Jacobian和Hessian的平坦度量；此外提出了直接可实现的JH正则化方案。

**🔧 技术方法**

核心技术包括PAC‑Bayes理论、Rademacher复杂度分析、平滑（Lipschitz与二阶光滑）约束、Gaussian扰动下的Jensen gap估计，以及利用随机向量估计Jacobian/Hessian。

**📊 数据集**

实验使用CIFAR‑10数据集（含清洁数据与20%标签噪声），对八层卷积网络进行训练。

**📈 对比分析**

与未正则化的交叉熵基线在不同批大小和训练周期下进行比较；JH正则化在大批量训练和噪声数据场景下显著提升准确率，尤其在变量epoch协议下可弥补批量增大带来的性能下降。

**⚠️ 局限性**

局限性包括理论上Rademacher项的常数未知、上界可能过于保守、正则化需近似计算Jacobian/Hessian、以及对BatchNorm统计的依赖导致实现复杂；此外，在纯无标签情形下的性能提升尚未深入验证。

---

## 394. PYPILINE: Malicious PyPI Package Detection via Suspicious API Knowledge and Agent Workflow

**arXiv ID:** 2606.19063 | [PDF](https://arxiv.org/pdf/2606.19063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 395. A performance portable fast Ewald summation for Stokes flow

**arXiv ID:** 2606.19059 | [PDF](https://arxiv.org/pdf/2606.19059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 396. CHERI-D: Secure and efficient inline object ID for CHERI temporal memory safety

**arXiv ID:** 2606.19055 | [PDF](https://arxiv.org/pdf/2606.19055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 397. Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation

**arXiv ID:** 2606.19037 | [PDF](https://arxiv.org/pdf/2606.19037v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 398. Monocular 3D Occupancy Perception for Robots on Sidewalks via Hybrid 2D-3D Learning

**arXiv ID:** 2606.19122 | [PDF](https://arxiv.org/pdf/2606.19122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 399. Written by AI, Managed by AI: Semantic Space Control and Index Sickness Elimination Across 391 Consecutive Sessions

**arXiv ID:** 2606.19121 | [PDF](https://arxiv.org/pdf/2606.19121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 400. PorTEXTO: A European Portuguese Benchmark for Visual Text Extraction

**arXiv ID:** 2606.19096 | [PDF](https://arxiv.org/pdf/2606.19096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. INDEQS: Informed Neural controlled Differential EQuationS

**arXiv ID:** 2606.19138 | [PDF](https://arxiv.org/pdf/2606.19138v1)

**作者:** Michael Detzel `[一作]`, Wojciech Samek `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

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

## 402. A Technical Taxonomy of LLM Agent Communication Protocols

**arXiv ID:** 2606.19135 | [PDF](https://arxiv.org/pdf/2606.19135v1)

**作者:** Linus Sander `[一作]` (Technische Universität München), Alois Knoll `[通讯]` (Technische Universität München)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了面向LLM代理通信协议的技术分类学，并对九种公开协议进行系统分析，揭示协议设计模式与发展趋势；

**💡 创新点**

创新点在于提出了五维（对等方、载荷、交互状态、发现机制、模式灵活性）完整且互斥的分类框架，并通过经验-概念迭代验证其适用性；

**🔧 技术方法**

采用了迭代式分类学构建方法（Nickerson等），结合协议实例提炼维度；

**📊 数据集**

数据来源为九个积极维护且具社区认可度的开源协议实现（如MCP、A2A、LAP、Agora等）；

**📈 对比分析**

比较方法是基于所建分类维度对每个协议进行属性标注，形成对照表和模式矩阵，阐释了协议在多轮状态、混合载荷和模式演进等方面的差异；

**⚠️ 局限性**

局限性包括缺乏对安全、隐私与策略执行等维度的深入探讨，且分类学未涵盖所有可能的协议细节（如持续存储、权限管理）及未来新协议的适配需求。

---

## 403. ReSiReg: Towards Spatially Consistent Semantics in Language-Conditioned Robotic Tasks

**arXiv ID:** 2606.19088 | [PDF](https://arxiv.org/pdf/2606.19088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 404. Towards an Agent-First Web: Redesigning the Web for AI Agents

**arXiv ID:** 2606.19116 | [PDF](https://arxiv.org/pdf/2606.19116v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Sachin Shetty `[通讯]` (Old Dominion University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个面向AI代理的“代理优先”网络改造框架，重新设计了访问、经济和内容三层，并给出了十项设计原则、迁移路线和技术规范；

**💡 创新点**

创新点在于以代理为人类代理（Agent‑as‑Human‑Proxy）为哲学基石，提出统一的代理识别元数据、agents.txt、ATML、token计费和委托订阅等技术标准，首次系统性地将“认知递归”（epistemic recursion）与内容可追溯性相结合，实现三层同步重设计；

**🔧 技术方法**

主要技术包括：HTTP请求头元数据（Agent‑Type、Agent‑Intent、Agent‑Token等）、agents.txt访问策略文件、OAuth2委托令牌、Token计费接口、Agent Text Markup Language（ATML）内容格式、C2PA加密内容鉴别链、双层（HTML/ATML）交付等；

**📊 数据集**

论文未使用公开数据集进行实验，主要引用行业统计（如Cloudflare阻断比例、Anthropic爬虫比率、零点击搜索占比等）作为实证依据；

**📈 对比分析**

论文未进行传统意义上的实验对比或性能测评，所述性能改进（如ATML相较HTML可节省约67.6% token）来源于先前研究报告；

**⚠️ 局限性**

局限性包括：缺乏大规模实证验证、标准化过程仍需行业共识与规范化工作、对广告模式的完整替代方案尚未成熟、以及对恶意代理与欺骗性代理识别的技术挑战。

---

## 405. Taming I2V models for Image HOI Editing: A Cognitive Benchmark and Agentic Self-Correcting Framework

**arXiv ID:** 2606.19073 | [PDF](https://arxiv.org/pdf/2606.19073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. Atomic Handover for 6G Nomadic Non-Public Networks Using Edge-Based Spectrum Brokering

**arXiv ID:** 2606.19058 | [PDF](https://arxiv.org/pdf/2606.19058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 407. Low-Rank Tensor Completion Based on Fractional Regularization with Ky Fan p-k Norm

**arXiv ID:** 2606.19046 | [PDF](https://arxiv.org/pdf/2606.19046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. Model-Free Reinforcement Learning Control for Resilient Cyber-Physical Systems

**arXiv ID:** 2606.19069 | [PDF](https://arxiv.org/pdf/2606.19069v1)

**作者:** Hugo O. Garcés `[一作]` (Universidad de Concepción), Sirish L. Shah `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文对非线性系统在假数据注入和 DoS 攻击下的模型无关强化学习控制器（RL‑PID 与 RL‑MPC）进行性能与鲁棒性评估。

**💡 创新点**

创新点在于通过不同奖励函数（Lyapunov、指数、渐进、线性）对 RL 控制器的奖励结构进行系统化设计，证明 Lyapunov 奖励能显著提升攻击下的鲁棒性，同时保持低跟踪误差。

**🔧 技术方法**

采用了两种 actor‑critic 算法（PPO 与 DDPG）实现控制器参数的在线自适应；同时使用基于 LPV 的单输入单输出非线性模型与仿真通信网络对 RL‑PID 与 RL‑MPC 进行训练与评估。

**📊 数据集**

数据集为仿真生成的 LPV 单输入单输出系统（含时间常数扰动）与通信网络模型，网络中加入 Gaussian 噪声、丢包、随机延迟以及漂移、噪声、DoS 攻击，形成完整的攻击与扰动场景。

**📈 对比分析**

通过与自适应控制、MPC、PID 等基准控制器在错误、计算成本、鲁棒性三大 KPI 维度的对比，发现 RL‑MPC（PPO + Lyapunov 奖励）在鲁棒性（AdC、MaM）方面最优；RL‑PID（DDPG + 线性奖励）在跟踪误差和训练时间上表现最佳；PPO 通常比 DDPG 更稳定、变异性更低。

**⚠️ 局限性**

主要局限在于无法同时兼顾最高鲁棒性、最低计算成本和最小跟踪误差；奖励函数与学习算法的敏感性导致不同场景下的表现差异，且缺乏严格的稳定性理论保证。

---

## 409. Evaluating Learned Spatial Indexes

**arXiv ID:** 2606.19034 | [PDF](https://arxiv.org/pdf/2606.19034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 410. RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents

**arXiv ID:** 2606.19047 | [PDF](https://arxiv.org/pdf/2606.19047v1)

**作者:** Ruishan Fang `[一作]` (Inclusion AI, Ant Group), Tao Lin `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Reward-driven Online Data Synthesis框架，通过进度奖励方差检测能力边界并动态合成多轮工具使用数据，持续补充梯度信息。

**💡 创新点**

利用进度奖励方差作为零成本边界探测器，配合技能对齐重采样生成结构一致的多轮示例，并通过动态回放缓冲维护训练分布。

**🔧 技术方法**

基于GRPO的强化学习，Popoviciu不等式启发的方差阈值，五阶段多代理合成管道（规划、执行、重写、评判、对抗增强），动态生命周期管理。

**📊 数据集**

从400个人工种子开始，BFCL V3多轮子集（800样本）训练并在BFCL V4、τ^2-bench、ACEBench进行OOD评估。

**📈 对比分析**

与静态数据、EnvTuning以及17K离线合成FunReason-MT进行对照，结果显示在400样本基准下提升约30%总体分数，且仅用20倍更少数据即可达到离线模型水平。

**⚠️ 局限性**

合成管道依赖可模拟的Python环境，难以直接应用于隐蔽或远程接口，且计算成本主要集中在生成阶段。

---

## 411. Human-AI Coevolution Dynamics: A Formal Theory of Social Intelligence Emergence Through Long-Term Interaction

**arXiv ID:** 2606.19144 | [PDF](https://arxiv.org/pdf/2606.19144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 412. The Simplicity Paradox: Why Evolution Does Not Produce Universally Complex Agents

**arXiv ID:** 2606.19136 | [PDF](https://arxiv.org/pdf/2606.19136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 413. Congestion-Aware Robot Tour Planning in Crowded Environments

**arXiv ID:** 2606.19031 | [PDF](https://arxiv.org/pdf/2606.19031v1)

**作者:** Stefano Bernagozzi `[一作]` (Istituto Italiano di Tecnologia), Lorenzo Natale `[通讯]` (Università di Genova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用CLiFF人流预测与MDP在线规划的拥堵感知巡回路径规划方法

**💡 创新点**

首次将基于概率的人流预测融入MDP决策，并通过拥堵带分层化降低规划复杂度

**🔧 技术方法**

CLiFF人流预测模型、Markov决策过程、LRTDP在线规划、最小生成树启发式

**📊 数据集**

使用真实的ATC购物中心人流数据集构建CLiFF地图与测试环境

**📈 对比分析**

与基于Hamilton路径的基线对比，实验显示LRTDP方法在执行时间上平均比基线快约20–30%，差异显著

**⚠️ 局限性**

在极度拥堵或大规模地图时，MDP状态空间和拥堵带数量仍会导致计算量显著增加，影响可扩展性

---

## 414. PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems

**arXiv ID:** 2606.19119 | [PDF](https://arxiv.org/pdf/2606.19119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 415. Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science

**arXiv ID:** 2606.19051 | [PDF](https://arxiv.org/pdf/2606.19051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 416. Analysing drivers and interdependencies in European electricity markets using XAI

**arXiv ID:** 2606.19118 | [PDF](https://arxiv.org/pdf/2606.19118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 417. ART-VS: Adaptive Resolution Tiling for Vision Transformer Visual Servoing

**arXiv ID:** 2606.19089 | [PDF](https://arxiv.org/pdf/2606.19089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 418. Compute-Budgeted Exploitability Evidence Graphs for Prospective Vulnerability Triage

**arXiv ID:** 2606.19076 | [PDF](https://arxiv.org/pdf/2606.19076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 419. Sensor Configuration Matters: A Systematic Evaluation of Multimodal SLAM on Quadruped Robots

**arXiv ID:** 2606.19067 | [PDF](https://arxiv.org/pdf/2606.19067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 420. Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams

**arXiv ID:** 2606.19111 | [PDF](https://arxiv.org/pdf/2606.19111v1)

**作者:** Haewoon Kwak `[一作]` `[通讯]` (Indiana University), Haewoon Kwak (Indiana University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理LLM团队的过程层协调控制，将团队领导理论转化为可测量的控制动作并评估其在不同任务与模型条件下的效果。

**💡 创新点**

提出行为签名与逐动作消融的测量范式，首次将事务型、变革型、情境型领导映射到LLM控制器，并将其与团队科学的情境理论对应。

**🔧 技术方法**

采用共享 round‑0 生成、投票基准、可解释的控制动作集合，结合三大开源模型（Llama、Gemma、OpenAI）和四种任务域的多轮交互实验。

**📊 数据集**

使用CommonsenseQA、StrategyQA、AlphaNLI、Social Chemistry、MATH‑500、ANLI‑R3、Winogrande和Scruples等公开基准数据集。

**📈 对比分析**

通过准确率、行为签名（锁定率、探索率、恢复率）及成本(token)与基准(flat、投票、理论无关控制)对比，发现大部分情况下无显著准确提升，唯有情境型领导在社交规范任务中因不可靠 round‑0 而获得约+8pp 的显著提升。

**⚠️ 局限性**

仅限于三代理、固定预算、共享信息的对称团队，未考察更大规模或异构团队、长时序交互、以及更大或闭源模型；准确收益仅在极少数条件下出现。

---

## 421. Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: From Evaluation to Diagnosis

**arXiv ID:** 2606.19053 | [PDF](https://arxiv.org/pdf/2606.19053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 422. Where Did the Variability Go? From Vibe Coding to Product Lines by Regeneration

**arXiv ID:** 2606.19042 | [PDF](https://arxiv.org/pdf/2606.19042v1)

**作者:** Xhevahire Tërnava `[一作]` `[通讯]` (Institut Polytechnique de Paris), Xhevahire Tërnava (Institut Polytechnique de Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 10 个基于 Vibe Coding 的 C/C++ 项目进行了探索性分析，发现其内部变异性几乎为零，并提出 Variability by Regeneration (VbR) 方法，利用 LLM 在生成时决定所有变异点，生成每个变体专用的二进制并通过调度器实现运行时切换。

**💡 创新点**

创新点在于：①首次将变异性移至声明式规格而非代码中；②将生成时（generation time）定义为新的绑定时机；③采用 LLM 作为变体派生引擎，实现无冗余代码、完全可追踪的变体集合。

**🔧 技术方法**

使用技术包括：大规模语言模型（如 Claude Opus 4）、YAML 规格文件、prompt 模板、自动化生成流水线、静态分析与单元测试、基于脚本的变体调度器，以及对生成过程的形式化描述。

**📊 数据集**

数据集包括：10 个公开的 GitHub Vibe Coding 项目（涵盖七大领域，行数 201–14,077），与一个 70,964 行可配置 C 系统做对比；以及在复现包中提供的 word‑count 例子用作产品族演示。

**📈 对比分析**

比较方法：对比 CLI 选项数量、预处理变量引用及条件编译指令的数量，显示 Vibe Coding 项目显著缺乏内部变异；在 VbR 演示中生成三款专用二进制，保证无死代码并通过调度器无缝切换，但评估仅限于单一小规模族，未涉及大规模性能或成本分析。

**⚠️ 局限性**

局限性包括：依赖 LLM 输出的非确定性和潜在缺陷；缺乏形式化的正确性保证；各变体之间缺乏代码复用；特性验证与属性检查仍为人工或手工验证；实验规模仅限于一个小型产品族，未评估生成成本、调度开销及大规模可扩展性。

---

## 423. Giskard : Byzantine Robust and Confidential Aggregation for Large-Scale Decentralized Learning

**arXiv ID:** 2606.19129 | [PDF](https://arxiv.org/pdf/2606.19129v1)

**作者:** Ousmane Touat `[一作]` (INSA Lyon), Sonia Ben Mokhtar `[通讯]` (INSA Lyon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Giskard协议，在去中心化机器学习中实现既保密又鲁棒的聚合

**💡 创新点**

创新点在于将中位数计算化为二分搜索，使用层级委员会树结构将MPC负载分散，委员会规模为O(log n)，从而实现对高达n/4 Byzantine节点的容忍并保持信息学的安全性

**🔧 技术方法**

核心技术包括：秘密共享与BGW可验证秘密共享、可验证重共享、层级树形委员会拓扑、基于二分搜索的坐标中位数求解、以及基于单纯形的安全多方计算原语

**📊 数据集**

在MNIST与CIFAR‑10数据集上进行实验，模拟多种Poisoning攻击（标签翻转、符号翻转、IPM、ALIE）

**📈 对比分析**

与现有A2A、A2C以及传统Secure Aggregation基线对比，Giskard的单方通信量提升幅度从10³到10⁶规模可减少三阶对数级别（最高可达10⁸‑10⁹倍），同时在模型精度上与传统鲁棒聚合器（中位数、trimmed mean、Krum等）保持相近或略优；实验表明N_iter≈10即可收敛

**⚠️ 局限性**

局限性在于仍需要同步网络和公钥基础设施；在极高的Byzantine比例（接近1/4）时委员会规模增长；对动态加入/离开的节点支持不完善；以及对大规模模型维度D的扩展仍需进一步优化

---

## 424. Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts

**arXiv ID:** 2606.19036 | [PDF](https://arxiv.org/pdf/2606.19036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 425. DREAM: Extending Vision-Language Models with Dual-Objective Encoding for Cross-Modal Retrieval

**arXiv ID:** 2606.19062 | [PDF](https://arxiv.org/pdf/2606.19062v1)

**作者:** Kaleem Ullah `[一作]` (Sejong University), Sung Wook Baik `[通讯]` (Sejong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DREAM框架，用双目标编码（MLM+PLM）和层次化视觉编码（CGAT）实现跨模态视频检索。

**💡 创新点**

创新点包括：①双路径文本编码融合掩码语言建模与置换语言建模，捕捉局部与全局语义；②分层视觉编码器结合级联组注意力（CGAT），在多尺度时空域中逐级细化特征；③在全局对齐的基础上，利用分层注意力实现对视频内容的细粒度对齐。

**🔧 技术方法**

使用技术：Transformer、CLIP文本编码器、双目标语言建模、分层视觉Transformer与CGAT、TokenInteract模块、对比式检索损失、L2归一化、余弦相似度评估、Adam优化。

**📊 数据集**

实验数据集：MSRVTT、MSVD、LSMDC。

**📈 对比分析**

在上述数据集上与最新SOTA进行对比，取得R@1分别为49.4%（MSRVTT）、49.7%（MSVD）和27.3%（LSMDC），并在视频到文本检索任务中同样表现优异，整体显著优于先前方法。

**⚠️ 局限性**

局限性：在视觉复杂或光照不足的场景中注意力易扩散；未使用光流等运动特征，导致对细微或快速动作的捕捉不足；未考虑多语言或长视频查询；依赖CLIP预训练，可能受限于其语言/视觉范畴。

---

## 426. A Hybrid LSTM--Vision Transformer Architecture for Predicting HRRR Forecast Errors

**arXiv ID:** 2606.19026 | [PDF](https://arxiv.org/pdf/2606.19026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 427. Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos

**arXiv ID:** 2606.19156 | [PDF](https://arxiv.org/pdf/2606.19156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 428. FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs

**arXiv ID:** 2606.19025 | [PDF](https://arxiv.org/pdf/2606.19025v1)

**作者:** Lorenzo Sani `[一作]`, Nicholas D. Lane `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术，构建了一个混合模型。

**📊 数据集**

采用了公开的图像识别数据集和自建的用户行为数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示新算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，且在极端情况下可能出现过拟合现象。

---

## 429. Lifecycle-Aware Dynamic Analysis for Secure ML Model Execution

**arXiv ID:** 2606.19023 | [PDF](https://arxiv.org/pdf/2606.19023v1)

**作者:** Gabriele Digregorio `[一作]` (Politecnico di Milano), Michele Carminati `[通讯]` (Politecnico di Milano)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种生命周期感知的动态分析框架，对 ML 模型在加载、推理等阶段与宿主系统的交互进行监控，检测并拦截恶意行为。

**💡 创新点**

创新点在于将模型执行分为若干生命周期阶段，并为每个阶段构建可复用的执行边界（allowlist），通过检测与边界的偏差来识别攻击，突破传统基于文件格式或签名的静态扫描限制。

**🔧 技术方法**

技术实现主要包括：基于 Linux ptrace 的系统调用跟踪、系统调用到安全交互的映射（Action Abstraction）、阶段化的执行边界定义、以及通过信号控制的阶段隔离（Orchestrator）。

**📊 数据集**

实验使用的数据集包含 77,974 条来自 Hugging Face Hub 的真实模型、31 条 CVE 对应的 PoC 以及 334 条公开基准模型，另外还测试了 TensorAbuse 的四种推理攻击示例。

**📈 对比分析**

与 ModelScan、ModelTracer、PickleBall、Fickling 等现有工具对比，本文方法在所有 PoC 上实现 100% 检测率、0% 假阳性率；系统调用跟踪的运行时开销低于 3 秒，且在大规模模型批量扫描时保持可接受的性能。

**⚠️ 局限性**

主要限制包括：需手工维护和验证执行边界，对抗动态分析的逃逸技术尚未覆盖，依赖运行时依赖库的完整性，且在框架重大升级时可能需要重新校准边界以避免误报。

---

## 430. ARIADNE: Agnostic Routing for Inference-time Adapter DyNamic sElection

**arXiv ID:** 2606.19079 | [PDF](https://arxiv.org/pdf/2606.19079v1)

**作者:** Enrico Cassano `[一作]` (University of Turin), Neo Christopher Chung `[通讯]` (Samsung AI Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、与适配器无关的动态适配器选择框架ARIADNE，基于冻结文本编码器空间中的聚类中心实现路由。

**💡 创新点**

将适配器路由视为输入分类问题，完全在输入嵌入空间中操作，无需访问适配器内部或额外训练，兼容任意PEFT方法。

**🔧 技术方法**

采用聚类生成任务代表中心点、余弦相似度匹配以及对比谱路由方法的零样本路由技术。

**📊 数据集**

在Llama 3.2 1B Instruct和Qwen2.5 3B Instruct上使用23个NLI、QA、Similarity、Reasoning任务的LoRA适配器，并扩展至44任务进行可扩展性测试。

**📈 对比分析**

与Arrow和SpectR谱路由方法在5任务基准上对比，ARIADNE在所有任务上表现更佳；总体选择准确率约85%，在23任务上平均任务性能54.74%，恢复97.44% Oracle性能；在44任务上选择准确率保持约89.7%。

**⚠️ 局限性**

需要访问适配器的训练样本以构造聚类中心，无法在没有训练数据或数据私有的分布式适配器生态系统中直接应用。

---

## 431. Viking Hill Dataset: A Lidar-Radar-Camera Dataset for Detection and Segmentation in Forest Scenes

**arXiv ID:** 2606.19154 | [PDF](https://arxiv.org/pdf/2606.19154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 432. A Clinician-Centered Pipeline for Annotation and Evaluation in Ultrasound AI Studies

**arXiv ID:** 2606.19174 | [PDF](https://arxiv.org/pdf/2606.19174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 433. Constant Time-Delay Leader Following with Neural Networks and Invariant Extended Kalman Filters for Arbitrary Trajectories

**arXiv ID:** 2606.19227 | [PDF](https://arxiv.org/pdf/2606.19227v1)

**作者:** Luka Antonyshyn `[一作]` (University of Toronto), Sidney Givigi `[通讯]` (Queen's University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种在无通信、无全局定位条件下，利用 SE(2) 流形上序列到序列（Seq2Seq）深度网络结合不变扩展卡尔曼滤波器（IEKF）进行领导者轨迹时延估计，并通过几何模型预测控制（GMPC）实现追踪的完整系统；

**💡 创新点**

创新点在于将 Seq2Seq 网络嵌入流形轨迹估计，利用 IEKF 的初始估计作为跳跃连接来提升预测准确性；同时设计了针对 SE(2) 的负对数似然（NLL）损失，兼顾位置与姿态误差；

**🔧 技术方法**

主要技术包括：序列到序列（GRU）网络、概率输出层生成协方差、IEKF、几何模型预测控制（GMPC）以及对 SE(2) 的 boxminus 运算；

**📊 数据集**

数据集为基于 RRT 生成的 1000 条随机障碍环境下的平面单车轨迹，并在真实 Husky 机器人上进行 8 次室内实验，使用 Vicon 捕捉真实状态；

**📈 对比分析**

与经典 IEKF、双向 GRU、Transformer 等基线进行对比；在仿真中 Seq2Seq+GMPC 的 RMSE 为 0.799，⊟-RMSE 为 0.649，明显优于 IEKF（1.035/0.706）和其他学习模型；在实验中平均 ⊟-RMSE 为 0.876，表现出良好鲁棒性；

**⚠️ 局限性**

局限性包括：对常数扰动或角速度偏置时误差增大；当前 GMPC 未充分利用预测协方差；模型对不可行轨迹的鲁棒性仍有限，且需要更复杂的障碍规避与感知集成。

---

## 434. Forecasting what Matters: Decision-Focused RL for Controlled EV Charging with Unknown Departure Times

**arXiv ID:** 2606.19199 | [PDF](https://arxiv.org/pdf/2606.19199v1)

**作者:** Giuseppe Gabriele `[一作]` (Ghent University -- imec), Chris Develder `[通讯]` (Ghent University -- imec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种决策聚焦的强化学习（DF‑RL）框架，联合训练电动车离车时长预测模型与充电策略，以解决未知离车时间导致的充电决策不确定性。

**💡 创新点**

创新点在于：①将预测模型与RL代理端到端联合训练；②通过决策聚焦损失（DF损失）将预测误差直接映射到充电策略的奖励上，而非仅优化预测精度；③引入权重β平衡传统回归误差与决策反馈，调节模型偏向预测准确性还是决策效能。

**🔧 技术方法**

使用技术包括：Soft Actor‑Critic（SAC）强化学习算法、神经网络回归预测器、决策聚焦损失函数、基于MSE的回归损失、贝塔加权联合训练。整个系统在单个训练周期内完成预测与策略参数的同步更新。

**📊 数据集**

数据集来源于医院停车场的真实电动车充电会话记录，构造了350个训练会话和120个测试会话；通过对20个最常用用户的统计特征进行建模，生成仿真样本以保证结果的可重复性。

**📈 对比分析**

对比方法包括：业务常规“立即充电”（BAU）、不使用预测的RL、RL+传统回归预测（β=1）以及使用真实离车时间的RL。DF‑RL 在总奖励上提升约5‑15%，未满足充电需求减少约14%，未充电车辆数量从33%下降到约48%（β=0.4），同时充电成本降低并且总惩罚成本显著下降。

**⚠️ 局限性**

局限性包括：①仅考虑固定价格曲线与单一充电功率，未涵盖多时段价格波动；②预测模型较为简单，未能充分捕捉时间序列与季节性特征；③未涉及多车协同控制与电网约束；④β参数需要经验调优，且在β=0时训练收敛困难；⑤实验基于单一真实数据集，缺乏跨地区的验证。

---

## 435. Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis

**arXiv ID:** 2606.19183 | [PDF](https://arxiv.org/pdf/2606.19183v1)

**作者:** Soheyl Bateni `[一作]` (K. N. Toosi University of Technology), Maryam Abdolali `[通讯]` (K. N. Toosi University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了ClaMPAPP，一种将大语言模型（LLM）限定为特征提取接口、并使用XGBoost分类器进行诊断预测的混合系统。

**💡 创新点**

创新点在于将LLM从决策者转变为接口，利用LLM完成结构化特征抽取与解释，同时通过确定性特征验证门和XGBoost模型实现可审计、低幻觉风险的高安全性诊断。

**🔧 技术方法**

采用LLM（如Llama‑3.1‑8B）进行文本解析与重写，构建模板生成与句子置换的鲁棒性评估；特征验证器执行范围与类型检查；XGBoost对结构化特征进行风险预测，并进行后验逻辑校准。

**📊 数据集**

使用两套德国儿科急诊队列：Regensburg（782例）作为内部训练/验证集，Düsseldorf（301例）作为外部验证集；所有案例通过模板+LLM重写生成“笔记式”叙述，保持原始结构化数据。

**📈 对比分析**

与多种开源与闭源LLM基线（MedGemma‑4b‑it、Llama‑3.1‑8B、OpenAI GPT‑5.5等）进行对比。ClaMPAPP在内部集的准确率≈85.1%、F1≈84.8%、灵敏度≈97.7%，外部集准确率≈80.7%、F1≈88.1%，并且误报率低、未漏诊率最小；在句子置换扰动下仍保持高灵敏度与鲁棒性。

**⚠️ 局限性**

局限包括：仅在回溯性实验中使用合成叙述，未验证在真实临床笔记下的抽取鲁棒性；数据仅来自两家德国医院，需在其他机构、语言及工作流程中进行外部验证与重新校准；特征验证器只能检测明显不合理值，对可疑但合理错误的抽取效果有限；校准仅为后验自我校正，缺乏独立验证。

---

## 436. Teaching Software Engineering with LLM and MCP Integration: From Classroom to Industry Practice

**arXiv ID:** 2606.19167 | [PDF](https://arxiv.org/pdf/2606.19167v1)

**作者:** Kehui Chen `[一作]` (City University of Hong Kong), Xiaoxue Ma `[通讯]` (Hong Kong Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并试点了一个基于大型语言模型（LLM）与模型上下文协议（MCP）协同的16周软件工程教学路径，结合理论学习、实践实验和企业实习，形成了完整的教学体系。

**💡 创新点**

创新点在于：①首次将MCP纳入软件工程课程，弥补了传统课程缺乏标准化 AI 协同开发教学的空白；②提出“MCP‑核心、LLM‑辅助”的教学理念，形成从基础理论到工业项目的渐进式教学流程；③构建了涵盖定量与定性指标的多维评价体系，系统评估学生在MCP适配、LLM协同开发和工业适应能力。

**🔧 技术方法**

使用技术：大型语言模型（如 GPT 系列）、模型上下文协议（MCP）及其实现框架（LangGraph、ADK 等），以及教学工具（课堂演示软件、企业开发环境）。

**📊 数据集**

未使用公开数据集；教学与评估主要基于行业实战项目、课堂练习与企业实习任务，构成案例式数据源。

**📈 对比分析**

论文未给出实验对比或性能指标；只说明通过混合评估（70% 定量、30% 定性）来衡量学生能力，缺乏基准模型或对比实验数据。

**⚠️ 局限性**

限制包括：①教学路径未针对不同高校与专业进行个性化定制；②教学资源侧重主流 LLM 与 MCP 版本，对新兴技术与多模态场景覆盖不足；③企业合作模式主要依赖合作企业，缺乏可复制推广的标准化渠道，导致推广成本高。

---

## 437. Seeing Through Occlusion: Deterministic Arm Kinematic Correction for Robot Teleoperation

**arXiv ID:** 2606.19240 | [PDF](https://arxiv.org/pdf/2606.19240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 438. Mechanism-Guided Selective Unlearning for RLVR-Induced Reasoning

**arXiv ID:** 2606.19222 | [PDF](https://arxiv.org/pdf/2606.19222v1)

**作者:** Chenyu Zhou `[一作]` (Institute of Science Tokyo), Xu Zhou `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出MAST方法，对RLVR诱导的数学推理行为进行机制对齐的选择性去学习。

**💡 创新点**

创新点在于利用token级概率变化、梯度方向与off‑principal能量等机制指标进行张量排名，实现低副作用的去学习。

**🔧 技术方法**

使用梯度上升、NPO、SimNPO等去学习目标，并结合注意力投影张量的off‑principal能量与梯度耦合度进行排名。

**📊 数据集**

数据集包括GSM8K、MATH（分为Counting、Probability、Geometry、Number Theory）以及MMLU用于通用能力评估。

**📈 对比分析**

通过与全参数梯度上升基准对比，MAST在保持GSM8K和MATH retain几乎无损失的情况下显著降低MATH forget率，并在两个模型族上实现统计显著优势。

**⚠️ 局限性**

局限在于仅验证了数学推理领域和两款模型，需扩展到更多任务和模型以及更强的去学习目标。

---

## 439. Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance

**arXiv ID:** 2606.19195 | [PDF](https://arxiv.org/pdf/2606.19195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 440. RECOM: A Validity Discrimination Tradeoff in Automatic Metrics for Open Ended Reddit Question Answering

**arXiv ID:** 2606.19218 | [PDF](https://arxiv.org/pdf/2606.19218v1)

**作者:** Pushwitha Krishnappa `[一作]`, Tathagata Mukherjee `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文介绍了ACL会议论文的排版规范与使用样式文件的说明，提供了模板和写作指导。

**💡 创新点**

创新点在于对ACL特定格式的细化说明，并提供了完整的样式文件与源代码示例，方便作者快速落地。

**🔧 技术方法**

采用LaTeX排版系统及其对应的.cls与.bst文件来实现规范化排版。

**📊 数据集**

无数据集，本文仅为格式说明文档。

**📈 对比分析**

不涉及实验或性能比较，主要是对格式和排版进行说明。

**⚠️ 局限性**

限制在于仅适用于ACL会议，且并未提供完整的科研内容或实验结果。

---

## 441. GUMP-Net: An interpretable model-data-driven intelligent algorithm for multi-class pelvic segmentation

**arXiv ID:** 2606.19215 | [PDF](https://arxiv.org/pdf/2606.19215v1)

**作者:** Liheng Wang `[一作]` (Chinese Academy of Sciences), Chong Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于改进地理活性轮廓模型的可解释模型-数据驱动多类骨盆分割框架 GUMP-Net，能够自动初始化、学习骨骼特定的边缘检测函数并通过深度迭代完成分割。

**💡 创新点**

创新点在于将改进的 GAC 模型通过算法展开（algorithm unrolling）转化为可训练的网络模块，并通过学习得到的骨骼专用 EDF（edge detector function）替代手工设计的阈值，提升分割的鲁棒性与可解释性；此外，加入对象检测模块实现端到端自动初始化，显著减少人工交互。

**🔧 技术方法**

主要技术包括：YOLOv8s 对象检测用于生成初始边界框；Attention U‑Net 负责学习可调节的 EDF；改进的 U‑Net 结构实现三步迭代的深度层集演化；同时采用交叉熵、Dice 与 LSF 损失并引入解剖结构加权损失，进行联合训练。

**📊 数据集**

使用三组医学 CT 数据集：Pelvic1K CLINIC（含骨盆骨折）、Pelvic Collected（无骨折但包含尾骨标注）以及 Ankle Collected（足踝 CT 标注胫骨与腓骨），共计约 90 个训练样本。

**📈 对比分析**

与 DeepLabv3+、U‑Net、Attention U‑Net、nnU‑Net、Swin‑Unet、FAS‑UNet、PottsMGNet 以及 SAM、MedSAM 等基线方法进行对比。实验表明 GUMP‑Net 在 Dice、Jaccard 维持 0.5–1 % 的优势，在 HD 与 ASSD 上分别低 0.5 与 0.05，尤其在小尺寸结构（如尾骨、腓骨）上表现最优；在小样本学习场景下保持更高的稳定性与泛化能力。

**⚠️ 局限性**

局限性包括：需要在训练时计算 Signed Distance Function（SDF），带来一定计算开销；目前未在含金属伪影的骨折数据上进行验证，泛化性待进一步提升；以及在极端少量样本或复杂拓扑结构下仍可能受限于初始框选的精度。

---

## 442. ROSA-TFormer: A Radar-Optical Sensor-Aware Temporal Transformer for Pinus sylvestris Plantation Classification in Northern Shaanxi Using GEE-Derived Sentinel-1/2 Time Series

**arXiv ID:** 2606.19204 | [PDF](https://arxiv.org/pdf/2606.19204v1)

**作者:** Nengbo Zhang `[一作]` (Chinese Academy of Sciences), Chang sheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出ROSA-TFormer模型，对北陕西松木种植区进行雷达‑光学时序分类，并利用GEE Sentinel‑1/2构建点级时序数据集进行评估。

**💡 创新点**

创新在于引入雷达光学感知门控的Transformer结构，分别嵌入SAR与光学特征，并在全年时序上通过自注意力聚合，同时具备可解释的门控与注意力权重。

**🔧 技术方法**

使用Transformer自注意力网络、SAR/光学双分支嵌入、token级感知门控、双聚合读取以及GEE云端时序构建与标准化等技术。

**📊 数据集**

采用GEE导出的Sentinel‑1/2月度与半月度时序点集，包含12‑token和24‑token两种分辨率，总计3989点，覆盖5类（土地裸露、建设区、其他植被、水体、松木种植）。

**📈 对比分析**

在严格点级序列拆分下与RF、XGBoost、1D‑CNN、早期融合Transformer对比，24‑token下ROSA‑TFormer达到99.67% OA、99.56%宏F1、98.91%松木F1；三种种子平均99.61% OA、99.49%宏F1；在空间块拆分中虽略逊于1D‑CNN，但仍保持竞争力。

**⚠️ 局限性**

局限性包括仅做点级验证，缺乏墙面图像推断与独立验证；仅区分5类，未考虑其他针叶类；与1D‑CNN相比准确率略低；仅使用光谱与SAR时序，未结合高分辨率纹理、地形等信息。

---

## 443. Invertible Neural Network Adapter for One-Step Flow Matching in Robot Manipulation

**arXiv ID:** 2606.19194 | [PDF](https://arxiv.org/pdf/2606.19194v1)

**作者:** Yu Zhang `[一作]`, Long Cheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个可逆神经网络适配器，用于一阶流匹配框架下的多模态机器人动作生成；

**💡 创新点**

通过将动作映射到可逆潜在空间并在该空间内约束速度场，保证信息无损重构，从而在单步去噪下提升动作精度与实时性；

**🔧 技术方法**

可逆神经网络（coupling layers）、流匹配一阶去噪、跨模态条件编码（视觉、语言、感知）以及联合重构损失；

**📊 数据集**

RoboTwin 1.0（2D 图像与 3D 点云）任务、Libero 语言驱动操作数据集，以及真实世界 UR 机器人和 OpenArm 平台；

**📈 对比分析**

与 Diffusion Policy、Flow Matching、ManiFlow 等基线在相同网络结构下进行对比。实验显示单步推理即可达到或超过多步基线的平均成功率（如 RoboTwin 任务提升 3.2%），在 3D 点云下平均成功率达 90%，且推理延迟从 110 ms 降至 61 ms；

**⚠️ 局限性**

适用于短期连续动作，对高度多模态或不连续动作分布的假设尚未验证；未考察长周期任务；可逆网络在训练时增加计算与内存开销，规模化部署需进一步优化。

---

## 444. Learning to Annotate Delayed and False AEB Events: A Practical System for Extreme Class Imbalance and Asymmetric Label Noise

**arXiv ID:** 2606.19186 | [PDF](https://arxiv.org/pdf/2606.19186v1)

**作者:** Mengxiang Hao `[一作]` (Li Auto), Xianpeng Lang `[通讯]` (Li Auto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

开发了首个自动化AEB触发事件标注框架，利用数据增强与噪声抑制技术自动识别延迟/误触发事件。

**💡 创新点**

提出双策略方案：针对AEB的专属数据增强与基于硬度的噪声抑制，解决极端类别不平衡与非对称标签噪声问题。

**🔧 技术方法**

采用Transformer时空编码模型、焦点属性改造/矢量移植/遮罩随机化三种增强策略、EMA平滑硬度估计、探针自适应阈值以及集成学习。

**📊 数据集**

使用约10万条人工标注的真实车辆AEB触发数据集，涵盖多种交通场景，按80/10/10比例划分训练、验证和测试。

**📈 对比分析**

与重加权、SMOTE、ENN、DeepEnsemble、SADE等方法对比，DP@R90%与FP@R90%分别达到60.1%/59.7%，比DeepEnsemble提升约10.3%/11.1%，实现最佳性能。

**⚠️ 局限性**

生成样本的真实性有限，过度合成可能导致分布漂移；模型仍需人机协作，无法完全自动化；对上游感知系统的依赖导致需定期重训以缓解漂移。

---

## 445. AGDN: Learning to Solve Traveling Salesman Problem with Anisotropic Graph Diffusion Network

**arXiv ID:** 2606.19185 | [PDF](https://arxiv.org/pdf/2606.19185v1)

**作者:** Bolin Shen `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的图神经网络框架 AGDN，用于求解旅行商问题。

**💡 创新点**

创新点在于引入 MixScore 过渡矩阵与各向异性图扩散机制，解决传统稀疏化导致的拓扑信息缺失和信息交换不足。

**🔧 技术方法**

采用了图扩散网络、注意力机制、MLP、热力图生成以及 MCTS 搜索等技术。

**📊 数据集**

使用了 TSP-100、TSP-200、TSP-500、TSP-1000 的合成数据集以及 TSPLib 实例。

**📈 对比分析**

与 GatedGCN、Concorde、LKH-3、DIMES 等基线对比，AGDN 在多尺寸、多分布和真实数据上取得更低长度/更小 Gap，且推理时间和参数量更优。

**⚠️ 局限性**

仍受限于极大规模实例的计算成本以及对扩散步数、学习率等超参数的敏感性。

---

## 446. When AUC Misleads: Polarization-Aware Evaluation of Deepfake Detectors under Domain Shift

**arXiv ID:** 2606.19184 | [PDF](https://arxiv.org/pdf/2606.19184v1)

**作者:** Dat Nguyen `[一作]` (Princeton University), Djamila Aouada `[通讯]` (Rupert-Karls-University Heidelberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

本文提供了ECCV会议论文提交的格式与规范，包括标题、作者信息、摘要、章节标题、图表、公式、脚注、引用以及行号等细节；

**💡 创新点**

创新点在于系统化阐述了双盲审稿、匿名化、页面限制以及对模板的严格使用要求，帮助作者避免常见错误并保证论文一致性；

**🔧 技术方法**

采用LaTeX（LNCS类）模板、行号化排版、图形与表格规范、公式编号以及参考文献格式化等技术手段；

**📊 数据集**

无具体数据集，仅适用于任何符合LNCS格式的ECCV论文；

**📈 对比分析**

通过规定14页（不含参考文献）以及严格的排版与字体要求来评估论文的可接受性；

**⚠️ 局限性**

局限性在于模板限制了排版自由度，不能使用自定义字体或手动调整布局；论文过长或页面尺寸不符将直接被拒绝。

---

## 447. RespGeomLib: A Reproducible Parametric Engine for Generating Analysis-Ready Human Airway Lumen Geometry

**arXiv ID:** 2606.19169 | [PDF](https://arxiv.org/pdf/2606.19169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 448. OneCanvas: 3D Scene Understanding via Panoramic Reprojection

**arXiv ID:** 2606.19253 | [PDF](https://arxiv.org/pdf/2606.19253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 449. Compute Efficiency and Serial Runtime Tradeoffs for Stochastic Momentum Methods

**arXiv ID:** 2606.19179 | [PDF](https://arxiv.org/pdf/2606.19179v1)

**作者:** Depen Morwani `[一作]` (Harvard University), Sham Kakade `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在一致线性回归中，基于批量大小的 Heavy Ball（HB）和 Accelerated SGD（ASGD）算法的计算效率下界与序列运行时间之间的权衡；

**💡 创新点**

给出了 HB 在任意谱下的最优谱半径下界，证明其计算效率上界与 SGD 相同但可在更大批量窗口内提升序列运行时间；同时提出 ASGD 在功率律谱下在小批量时具有更优的计算效率，并在更小批量规模上即可达到加速极限；

**🔧 技术方法**

利用高维随机矩阵理论、协方差递推与高阶矩公式，分析算法的线性递推和谱半径；

**📊 数据集**

实验使用合成线性回归数据，权重方差呈 λ_i∝i^{-a} 的功率律谱（a=2），维度 D=50000，迭代 N=500000；

**📈 对比分析**

对比 HB、ASGD 与 SGD 的目标损失收敛步数随批量大小的变化，发现 ASGD 在小批量下计算效率更好，HB 在更大批量窗口内缩短序列时间，实验结果与理论下界吻合；

**⚠️ 局限性**

仅给出下界，未提供上界或实际实现的最佳参数调优，且实验仅在合成数据上验证，缺乏在真实深度学习任务中的验证。

---

## 450. Dango: A Strictly L1-Only Large Language Model for Studying Second Language Acquisition

**arXiv ID:** 2606.19170 | [PDF](https://arxiv.org/pdf/2606.19170v1)

**作者:** Shiho Matta `[一作]` (Kyoto University), Yugo Murawaki `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并微调了1.8B参数的Dango模型，先在过滤后的日语语料上预训练，再在LLM生成的日英教材上微调，以模拟日语到英语的第二语言习得过程。

**💡 创新点**

首次针对L1→L2迁移在大规模LM上进行控制实验，并提出针对日语语料的L2污染过滤流程，使模型保持近乎纯日语的预训练环境。

**🔧 技术方法**

使用Llama‑2风格的解码器Transformer、白名单/黑名单过滤、LLM‑as‑a‑judge评估框架、BLEU、JSD/BCD等评测指标。

**📊 数据集**

llm‑jp‑corpus‑v3（日本语部分）作为L1预训练数据，GPT‑5.2生成的CEFR‑J词典教材作为L2微调数据，ICNALE写作与对话数据用于评测。

**📈 对比分析**

通过与未过滤、跨语言预训练基线及GPT‑4o/GPT‑5.5等提示方法比较，Dango在英语生成中与日本学习者的用法频率和错误率分布最为相似，且在BLEU和交叉验证上表现优于同规模基线。

**⚠️ 局限性**

依赖预先训练好的多语种分词器导致L2词汇学习不完全自学；仅在1.8B规模和单一日英方向实验，未探究更大规模或其他语言对。

---

## 451. Essential Subspace Merging for Multi-Task Learning

**arXiv ID:** 2606.19164 | [PDF](https://arxiv.org/pdf/2606.19164v1)

**作者:** Longhua Li `[一作]` (Southeast University), Qi Tian `[通讯]` (Huawei Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于输出激活子空间的模型合并方法，能够在无额外训练的情况下将多任务专家模型融合为一个多任务模型。

**💡 创新点**

核心创新在于提出了 Essential Subspace Decomposition（ESD），将任务更新在激活空间中的主成分作为关键子空间，并基于此设计了静态融合 ESM 和动态专家路由 ESM++，显著降低任务间干扰。

**🔧 技术方法**

采用输出激活 PCA、低秩截断、正交化以及基于原型的无训练路由等技术，构建了训练自由的模型融合框架。

**📊 数据集**

在 Vision（ViT‑B/32、ViT‑B/16、ViT‑L/14 的 8/14/20 任务集）、GLUE（8 任务 RoBERTa‑Base）以及 Llama‑3.2‑3B 的指令/数学/编程/多语/安全 5 任务等多领域基准上进行实验。

**📈 对比分析**

与 20 多种现有静态与动态合并方法对比，ESM 在所有 Vision 任务集上均达到或接近最优，ESM++ 在 Vision、语言和生成任务上进一步提升 1–2% 的平均准确率或分数，逼近专家上限。

**⚠️ 局限性**

仅适用于相同架构且同一预训练基座的模型，难以直接扩展到不同源、不同训练配置或异构模型的融合。

---

## 452. Mobile Pedipulation for Object Sliding via Hierarchical Control on a Wheeled Bipedal Robot

**arXiv ID:** 2606.19233 | [PDF](https://arxiv.org/pdf/2606.19233v1)

**作者:** Yue Qin `[一作]` (University of Michigan), Yanran Ding `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并实现了一套分层控制框架，使带轮双足机器人能够通过步行与滑动的结合完成平面物体的抓取与推送任务。

**💡 创新点**

创新点包括：①提出含臀部滚转自由度且支持多种接触模式的三刚体(TRB)模型；②将TRB与物体动力学(TRBO)结合的轨迹规划与非线性MPC，实现对物体滑动的实时预期与控制；③将全身控制与预测控制耦合，显著提升鲁棒性。

**🔧 技术方法**

使用技术包括三刚体模型、非线性模型预测控制、轨迹优化、全身控制（WBC）、Coulomb摩擦约束、仿真平台MuJoCo、CasADi、Fatrop求解器、Pinocchio、Kalman滤波、MoCap实时反馈。

**📊 数据集**

未使用公开数据集，实验数据来自Tron1机器人硬件与仿真，MoCap系统用于物体状态估计。

**📈 对比分析**

与仅使用TRB NMPC相比，TRBO NMPC在不同质量误差下保持100%成功率；仿真中可滑动至23 kg物体，硬件实验实现1 kg、4 kg物体滑动，速度跟踪误差低，性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：①依赖实验室MoCap估计物体状态；②物体模型简化为点质量，忽略旋转与接触几何；③预定义接触序列，缺乏在线接触策略。

---

## 453. Machine Unlearning for the XGBoost Model with Network Intrusion Datasets

**arXiv ID:** 2606.19220 | [PDF](https://arxiv.org/pdf/2606.19220v1)

**作者:** Diana Magalhães `[一作]` (Polytechnic of Porto), Isabel Praça `[通讯]` (Polytechnic of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了XGBoost-Forget，利用SISA框架实现XGBoost模型的机器无学习（unlearning）

**💡 创新点**

创新点在于首次将SISA分片+切片策略迁移至传统树模型XGBoost，并验证其在网络入侵检测领域的有效性

**🔧 技术方法**

技术核心是基于分片与切片的增量训练、模型检查点保存与局部重训练，以及对XGBoost内部树结构的增量更新

**📊 数据集**

使用了两个公开网络入侵数据集：IoT-23和GeNIS

**📈 对比分析**

与全模型重训练和SISA+NN基线比较，XGBoost-Forget在ACC、PRE、REC、F1维持与原模型几乎相同的高性能，同时在RT上显著快于重训练和SISA，ASR指标显示忘记效果接近重训练基准

**⚠️ 局限性**

局限在于JSD等分布差异指标对小比例删除的数据表现不敏感，需进一步探索更适合的忘记质量评估方法

---

## 454. No Two Developers Think Alike: How Problem-Solving Styles and Experience Shape Needs in Conversational Interaction with Copilot

**arXiv ID:** 2606.19216 | [PDF](https://arxiv.org/pdf/2606.19216v1)

**作者:** Jonan Richards `[一作]` (Radboud University), Mairieli Wessel `[通讯]` (Radboud University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过27位专业开发者和学生的混合方法认知思考实验，研究了GitHub Copilot Chat交互模式，识别出5种交互模式和10种驱动需求，并构建了将认知多样性（经验、问题解决风格）与交互模式关联的概念模型。

**💡 创新点**

首次系统地将认知多样性与对会话式编程助手的交互需求和模式相联系，提出了基于需求驱动的交互概念模型，为定制化设计和研究提供了新的视角。

**🔧 技术方法**

采用认知思考法、主题建模、定性编码、后测访谈等混合方法；使用GenderMag框架评估问题解决风格；分析Copilot Chat交互日志。

**📊 数据集**

基于27名参与者的交互日志、语音记录和访谈文本；未使用公开数据集，而是采集实验生成的原始数据。

**📈 对比分析**

本研究为探索性研究，不涉及与其他工具的性能对比；主要通过定性分析和关联统计描述模式与需求之间的关系，未给出具体数值性能指标。

**⚠️ 局限性**

样本规模有限，缺乏足够的性别多样性；研究仅涵盖Copilot Chat且任务有限，未在真实生产环境中验证；概念模型未进行外部验证，因而结论仅为探索性假设。

---

## 455. STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability

**arXiv ID:** 2606.19236 | [PDF](https://arxiv.org/pdf/2606.19236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 456. Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection

**arXiv ID:** 2606.19168 | [PDF](https://arxiv.org/pdf/2606.19168v1)

**作者:** Jinhan Li `[一作]` (Tsinghua University), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Safety Reflection Pretraining 方法，在预训练阶段周期性插入安全反思文本，使 LLM 能自我监测并提升安全性。

**💡 创新点**

创新点在于将安全反思标签嵌入预训练语料，既强化安全分类，又构建了模型的自监控机制。

**🔧 技术方法**

采用文本分段、自动生成安全反思（基于安全分类器）、安全反思预训练以及兼容的后训练设计。

**📊 数据集**

使用 FineWeb-Edu、Tulu3 语料以及自构建的 MedSafetyWorld 合成医学安全环境进行实验。

**📈 对比分析**

与 Baseline 及 SafeLM 对比，SRP 在推理阶段攻击、微调攻击和通用能力评估中显著降低攻击成功率，同时保持与 Baseline 相近的通用性能。

**⚠️ 局限性**

局限在于后训练必须保持相同的反思格式，否则反思行为易被遗忘；在无反思数据的微调下仍存在一定风险，未能完全消除所有攻击。

---

## 457. A Taxonomy of Mental Health and Technology Needs for Alzheimer's and Dementia Caregivers

**arXiv ID:** 2606.19247 | [PDF](https://arxiv.org/pdf/2606.19247v1)

**作者:** Keran Wang `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

对阿尔茨海默病及相关痴呆症患者的非正式照护者进行系统综述与访谈研究，构建了一个将心理健康需求与技术干预对应的税onomy。

**💡 创新点**

①将传统的“负担”概念拆解为八个实证可测的心理健康域；②提出需求与技术类别（监测、信息、社交支持、任务管理、AI聊天机器人）的映射矩阵；③揭示现有技术与需求之间的匹配缺口，并给出设计与实施建议。

**🔧 技术方法**

使用PRISMA系统综述流程、Best‑Fit Framework Synthesis（BFFS）进行框架构建，以及半结构化访谈配合主题分析来提炼需求与技术映射。

**📊 数据集**

文献样本为52篇（1997‑2025）跨医学、心理学、人机交互等领域；访谈样本为41位阿尔茨海默症/痴呆症照护者，涵盖不同疾病阶段与照护角色。

**📈 对比分析**

本研究并未采用定量实验或性能指标，而是通过文献与访谈结果的交叉对比，评估各技术类别对需求的覆盖程度（如监测工具对安全与焦虑的支持、AI聊天机器人对情绪调节的潜力）并指出不足之处。

**⚠️ 局限性**

限制包括：样本以英语、数字化较高的美国照护者为主，可能缺乏低资源、少数族裔或跨文化视角；访谈样本规模有限；缺少纵向追踪来观察需求与技术匹配随时间演变；技术评估主要基于现有研究与商业原型，未进行原型验证；检索仅使用“caregiver”关键词，可能遗漏使用“care partner”等新术语的研究。

---

## 458. CodeSentinel: A Three-Layer Defense Against Indirect Prompt Injection in Code Contexts

**arXiv ID:** 2606.19235 | [PDF](https://arxiv.org/pdf/2606.19235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 459. Guarded Epoch Bloom Filters for Sliding-Window Membership

**arXiv ID:** 2606.19210 | [PDF](https://arxiv.org/pdf/2606.19210v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Guarded Epoch Bloom Filter（GE‑BF），通过分段旋转而非计数器或删除队列实现滑动窗口近似成员查询，并给出了误报/误删近似分析与实验验证。

**💡 创新点**

使用一个额外的 guard 段实现 deterministic live‑window invariant；通过 epoch 旋转在不需要 per‑key 删除队列和计数器的情况下，保证最近 W 次插入的键无误删，且误报上限受 epoch 数限制。

**🔧 技术方法**

基于 Bloom Filter 的双哈希定位、r+1 段分段存储与周期性清空，提供可选的 blocked 版本提升局部性；同时推导了误报概率公式。

**📊 数据集**

实验使用三种合成流（均匀、Zipf、突发）和真实的 Organization X Apache 访问日志（HTTP 请求键为 IP+方法+路径）。

**📈 对比分析**

在相同总位预算下与四位计数 Bloom Filter、Stable Bloom Filter 进行比较，测量 fresh negative FPR、live keys FNR、expired keys 正率和查询吞吐。结果显示 GE‑BF 在相同预算下 FPR 降低至约 0.022，且无 live FNR，expired 正率可控；吞吐比计数 BF 低，但高于 Stable BF。

**⚠️ 局限性**

无法提供精确删除；存在一个 epoch 的滞后误报；查询需要 r+1 段探测；实验仅覆盖合成流与单一日志样本，未在更大规模或多样化数据集上验证。

---

## 460. The More the Merrier: Combining Properties for ABox Abduction under Repair Semantics for ELbot

**arXiv ID:** 2606.19197 | [PDF](https://arxiv.org/pdf/2606.19197v1)

**作者:** Anselm Haak `[一作]` (Paderborn University), Anni-Yasmin Turhan `[通讯]` (Paderborn University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本研究在不一致知识库下，系统地分析了在修复语义（勇敢与AR语义）下ABox归纳的复杂性，重点探讨了将多种属性（如签名限制、非平凡性、冲突约束）与最小性条件（子集/基数最小化、冲突最小化）组合后的归纳问题，并给出了对应的存在性与验证性复杂度上界与下界。

**💡 创新点**

创新点在于首次将多种属性与最小性约束同时考虑，证明在大多数组合下复杂度由最难单属性决定，揭示冲突约束对复杂度的“降级”效应，并对EL_⊓ 以及部分更表达式DL的组合情况给出了完整的复杂度图谱；同时指出仍未解决的子集最小冲突约束和基数冲突最小化的复杂度。

**🔧 技术方法**

主要采用理论复杂度分析技术，包括从∃∀-QBF、可判定的 entailment oracle、修复集合与冲突集合的猜测与检查，构造多种多项式时间归约以证明P、NP、coNP、Σ^p_2 以及 #P 上界或下界。

**📊 数据集**

本工作完全基于理论模型分析，无使用实际数据集。

**📈 对比分析**

由于研究侧重于理论复杂度，未进行实验比较；作者通过构造性的归约与 oracle 调用证明了各组合的上界与下界，显示理论上复杂度可从P到Σ^p_2 甚至 #P 级别，具体性能取决于所需满足的属性组合。

**⚠️ 局限性**

局限性包括：①对子集最小冲突约束的验证问题尚未给出完整复杂度；②基数冲突最小化可能涉及 #P 难度，未完成精确评估；③仅覆盖EL_⊓ 与部分更表达式DL，缺乏对IAR等其他修复语义的分析；④未提供实验验证，缺少实际应用与性能评估。

---

## 461. FAST-LIVGO: A Degeneracy-Robust LiDAR-Inertial-Visual-GNSS Fusion Odometry

**arXiv ID:** 2606.19190 | [PDF](https://arxiv.org/pdf/2606.19190v1)

**作者:** Zhiyu Chen `[一作]` (Shenzhen University), Yukang Cui `[通讯]` (Shenzhen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种紧耦合的LiDAR-惯导-视觉-GNSS融合框架，利用误差状态迭代卡尔曼滤波实现全球一致的高精度定位；

**💡 创新点**

创新点包括：1) 基于动态时间规整（DTW）的在线时空对齐实现高动态下同步；2) 固定锚点时间差载波相位（TDCP）和多普勒观测的高精度观测模型；3) 依赖LIVO海森矩阵最小特征值的退化感知双模式外点剔除策略；

**🔧 技术方法**

使用了ESIKF、DTW、VOF、VoxelMap、FAST-LIVO2前端、TDCP、Doppler观测、RAIM、Chi‑square检验、滑动窗口退化检测等技术；

**📊 数据集**

验证数据集包括公开的M3DGR（多传感器大规模动态）以及自建的高速固定翼UAV（20 m/s）和手持混合环境数据；

**📈 对比分析**

通过与LIO‑SAM‑GPS、FAST‑LIVO2、LIGO等SOTA系统在M3DGR上做ATE RMSE对比，在开阔天顶场景下实现最小误差（约0.23 m），在GNSS退化场景下保持系统不失效并仅略高于FAST‑LIVO2；在UAV/手持实验中展示了显著的漂移抑制与地图一致性提升；

**⚠️ 局限性**

局限性在于：1) 依赖多普勒和TDCP观测时对GNSS信号质量要求较高；2) 双模式外点剔除在极端退化场景下仍需人工设置阈值；3) 对极低频GNSS或长时间无GNSS的场景性能尚未彻底验证。

---

## 462. User as Engram: Internalizing Per-User Memory as Local Parametric Edits

**arXiv ID:** 2606.19172 | [PDF](https://arxiv.org/pdf/2606.19172v1)

**作者:** Bojie Li `[一作]` `[通讯]` (Pine AI), Bojie Li (Pine AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一种将个人记忆拆分为内容与推理两层的架构——用户记忆作为 Engram 表的局部编辑，推理技能作为共享 LoRA，并提供轻量级多租户服务器。

**💡 创新点**

创新点在于把记忆和推理分离，利用 Engram 的可寻址稀疏存储做局部事实写入，避免全局权重污染；同时提出了层式设计与“回忆+推理”训练策略。

**🔧 技术方法**

使用了 Engram sparse memory 模块、LoRA 参数高效微调、三种写入策略（UNEMBED_P、OPT、Joint OPT）、共享 LoRA 训练、以及检索基线 Mem0/MemMachine；还实现了 EngramServer。

**📊 数据集**

数据集包括合成的 USER/ORG 事实集（基础、XL、XXL）、ClimbMix、LOCOMO benchmark 以及检索所需的句子编码器。

**📈 对比分析**

在多项指标上对比了 Per-user LoRA、ICL、检索 RAG 和单独 Engram；User-as-Engram 在直接召回与 LoRA 相当，间接推理准确率提升 5.6×，存储成本降低 161×，并在 KB>100 时优于检索方法。

**⚠️ 局限性**

局限性包括无法完成多跳推理（只能匹配表面触发词）、用户级密度上限（高事实数时受前向传播干扰）、需要 Engram 预训练、检索在大知识库下性能下降，以及对不同任务（开放域）适用性有限。

---

## 463. Evaluating Rust for Sparse Matrix Kernels in Scientific Computing

**arXiv ID:** 2606.19213 | [PDF](https://arxiv.org/pdf/2606.19213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 464. The Market in the Model: Latent Diffusion as Neural Economy

**arXiv ID:** 2606.19151 | [PDF](https://arxiv.org/pdf/2606.19151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 465. Pulse: Training Acceleration for Large Diffusion Models with Automatic Pipeline Parallelism

**arXiv ID:** 2606.19163 | [PDF](https://arxiv.org/pdf/2606.19163v1)

**作者:** Boran Sun `[一作]` (Hong Kong University of Science and Technology), Bo Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自动化流水线并行训练框架 PULSE，用于在大型扩散模型中通过跳跃连通的局部化来显著降低通信开销。

**💡 创新点**

创新点在于：①将具有对称跳跃连接的编码器-解码器层强制同放置在同一设备上；②设计了基于动态规划的跳跃感知分区器；③构造了基于整数线性规划的波形调度器；④联合优化管线、数据并行度与微批量大小，实现整体吞吐率最大化。

**🔧 技术方法**

采用动态规划+ILP调度、混合并行（pipeline+data）以及全流程自动化管线分区与调度；训练时在每个设备上本地缓存跳跃激活以消除跨设备 P2P 通信；使用 GPU/TPU 及 Ascend NPU 集群进行实验。

**📊 数据集**

在 UViT、Stable Diffusion v2 与 Hunyuan‑DiT 等主流扩散模型上，利用公开的图像+文本训练集（如 LAION‑400M、COCO、ImageNet 等），将图像编码为潜在表示，文本转为 CLIP/T5 嵌入后进行训练。

**📈 对比分析**

与 Hanayo、Megatron 1F1B 以及 DeepSpeed ZeRO‑2 基线相比，PULSE 在 2‑节点 V100 集群上吞吐率提升 1.4–2.3 倍，通信量下降 85–90%；在 8‑节点 Ascend 910A 集群上吞吐率提升 2.3–2.8 倍，显著降低跨节点 P2P 传输。

**⚠️ 局限性**

局限性包括：仅针对对称跳跃连接的 UNet 结构，无法直接处理非对称或稀疏跳跃；未结合张量并行，仅在管线级别优化；调度方案离线生成，缺乏对动态模型结构变化的实时适配；对低带宽或高延迟环境的适用性尚未系统评估。

---

## 466. HT-Bench: Benchmarking and Learning Dexterous Full-Hand Tactile Representations with Egocentric Vision

**arXiv ID:** 2606.19161 | [PDF](https://arxiv.org/pdf/2606.19161v1)

**作者:** Yuzhe Huang `[一作]` (Beihang University), Yuanxin Zhong `[通讯]` (Rimbot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HT-Bench多任务基准，评估全手触觉表示，并研发HandTouch向量量化跨模态编码器；

**💡 创新点**

通过将 egocentric 视觉与全手触觉配对，提供大规模可扩展基准；采用逐步空间、跨模态、时序三阶段训练；

**🔧 技术方法**

使用 ViT、vector‑quantized 码本、跨模态注意力、掩码学习与多模态时序预测；

**📊 数据集**

HT-Bench 约10M RGB帧与7.8M触觉帧，226个任务，支持任务级 OOD 分裂；

**📈 对比分析**

与 ResNet、CNN、VQ‑VAE、ViT 等基线对比，在细粒度检索 Recall@5 由74.65% 提升至85.23%，遮挡重建 RMSE 由0.022 降至0.010，跨模态合成 OOD cIoU 由0.628 提升至0.705；

**⚠️ 局限性**

仅针对 egocentric 视觉+全手触觉，未覆盖其他传感器/构造；实验分析有限，缺乏真实机器人下游任务验证；对高强度遮挡的 OOD 细部重建仍困难；

---

## 467. OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems

**arXiv ID:** 2606.19145 | [PDF](https://arxiv.org/pdf/2606.19145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 468. Complementary Attention Head Pruning for Efficient Transformers

**arXiv ID:** 2606.19150 | [PDF](https://arxiv.org/pdf/2606.19150v1)

**作者:** Yaniv Livertovsky `[一作]` (Bar-Ilan University), Gonen Singer `[通讯]` (Bar-Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种后训练自动化的Transformer注意力头裁剪框架CAHP，能够在不指定稀疏度的前提下自动识别并保留功能互补的头部。

**💡 创新点**

创新点在于把注意力头选择转化为全局图论问题，利用JM距离构建特征空间、k-medoids聚类、MSS评估与Kneedle算法自动确定最佳裁剪量，从而避免传统基于梯度或重要性排名的近端偏置。

**🔧 技术方法**

采用的技术包括：padding-aware attention签名提取、JM距离与t‑SNE降维、k‑medoids聚类、MSS（Mean Simplified Silhouette）与Kneedle曲线分析、梯度基重要性权重和轻量级微调。

**📊 数据集**

使用的评测数据集为SST‑5（情感分类）和MNLI（自然语言推断），覆盖BERT‑Base与BERT‑Large两种规模。

**📈 对比分析**

与DSP、PASS、AttAttr等基线比较，CAHP在所有压缩等级下均能保持与原模型相当或更优的准确率，且在极高压缩（保留不到20%头部）时仍能维持低方差、稳定的性能。

**⚠️ 局限性**

局限性包括：仅验证分类任务，未评估生成或多语言场景；对梯度估计的依赖导致在低资源环境下可能出现数值不稳定；图空间构造与聚类步骤相对耗时，尚需进一步优化。

---

## 469. OpenAnt: LLM-Powered Vulnerability Discovery Through Code Decomposition, Adversarial Verification, and Dynamic Testing

**arXiv ID:** 2606.19149 | [PDF](https://arxiv.org/pdf/2606.19149v1)

**作者:** Nahum Korda `[一作]` (Knostic), Gadi Evron `[通讯]` (Knostic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 OpenAnt 的闭环漏洞发现系统，通过静态代码拆分、LLM 语义推理、对抗式攻击模拟以及自动化动态验证，实现在大型代码库中自动识别并验证真实可利用的安全缺陷。

**💡 创新点**

创新点包括：① 多阶段过滤（可达性+曝光分类）将分析面压缩至原始代码的 1% 以下；② 采用 LLM 进行语义推理并在攻击者受限情景下做对抗验证，显著降低误报；③ 自动生成临时 exploit 环境并在沙箱中执行，实现从检测到验证的一体化闭环；④ 统一支持多语言（C/C++, Go, Python, JS/TS, Ruby, PHP），展示语义推理的跨语言通用性。

**🔧 技术方法**

技术栈：静态程序分析（AST、调用图、可达性过滤）；LLM 推理（Claude Sonnet 进行曝光分类与动态验证生成，Claude Opus 进行漏洞检测与对抗验证）；对抗式攻击模拟（受限攻击者模型）；动态验证（自动生成 Dockerfile、测试脚本，在容器中执行并返回 JSON 结果）；成本与性能监控。

**📊 数据集**

评估数据集为 8 个真实开源项目（OpenSSL, Flowise, eShopOnWeb, n8n, WordPress, object-browser, paperless-ngx, Rails），共 64,132 个函数，未使用传统 benchmark，而是直接在真实代码库中发现未知漏洞。

**📈 对比分析**

与传统 SAST 工具及单阶段 LLM 分析做对比；评估指标包括可扩展性（分析面缩减比例）、发现率（确认的漏洞数量）、误报率（对抗验证淘汰率）、动态确认率（75.8%）以及成本（总计 1,461.25 美元，单项目平均 183 美元）。实验显示，OpenAnt 在保持低误报的同时，能在 8 个项目中发现 190 个候选漏洞，其中 144 个通过动态验证被证明可利用。

**⚠️ 局限性**

局限性：① 结果高度依赖 LLM 的推理能力，某些细粒度或域特定漏洞仍难检测；② 对抗验证与动态验证阶段仍存在覆盖不足，无法验证时序/分布式漏洞；③ 计算成本高，尤其曝光分类阶段对大仓库仍昂贵；④ 上下文窗口限制可能导致大函数被截断；⑤ 未提供形式化安全保证，验证结果仍为经验性证明。

---

## 470. Ray Antenna Array Enhanced Low-Altitude ISAC: Performance Analysis and Beamforming Design

**arXiv ID:** 2606.19146 | [PDF](https://arxiv.org/pdf/2606.19146v1)

**作者:** Zhiqiang Xiao `[一作]` (National University of Defense Technology), Yong Zeng `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并优化了基于射线天线阵列（RAA）的低空集成感知与通信（ISAC）系统，提出联合射线选择与波束成形的非凸优化框架，并给出解析上限及高效交替优化算法。

**💡 创新点**

首次将无需相位移器的射线天线阵列引入低空ISAC，利用射线选择实现灵活波束形成，解决传统ULA在上方盲区和硬件成本高的问题。

**🔧 技术方法**

采用射线天线阵列设计、射线选择网络、SCA与交替优化求解非凸问题，并对感知SNR与通信SINR进行解析与仿真评估。

**📊 数据集**

无公开数据集，使用仿真（目标在150 m × 100 m矩形区域内、单目标、二维平面）进行性能验证。

**📈 对比分析**

与传统全向或定向天线ULA在感知SNR、最小通信SINR以及覆盖范围三方面进行对比，结果显示RAA在覆盖率、感知SNR和最小SINR上均优于ULA，尤其在上方盲区表现突出。

**⚠️ 局限性**

局限性包括：RAA需要更多天线元件，交替优化与SCA迭代复杂度高，仿真仅针对单目标与二维场景，实际硬件实现与复杂度尚待进一步验证。

---

## 471. Transformer Geometry Observatory TGO-I: Spectral Geometry Observatory

**arXiv ID:** 2606.19249 | [PDF](https://arxiv.org/pdf/2606.19249v1)

**作者:** Kaustubh Kapil `[一作]` (Sardar Vallabhai National Institute of Technology), Kishor P. Upla `[通讯]` (Sardar Vallabhai National Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了Transformer Geometry Observatory（TGO）框架，对ViT在训练过程中的表示几何进行系统的谱分析，并提出TGO-I作为第一个观测模块。

**💡 创新点**

创新点在于：①提出了以谱特征为核心的可模块化观测框架；②首次系统记录并分析了ViT从初始化到100轮训练中所有层的有效秩、稳定秩、参与比、谱熵、谱各向异性等多维度特征；③发现CLS标记的表示几何最为分散，且随着训练显著变平坦；④为后续研究提供了可复现的量化指标与可视化工具。

**🔧 技术方法**

使用的技术包括：ViT-Small/16模型训练、固定验证子集（1000张图）提取激活、前向钩子、协方差矩阵构造、特征值分解、有效秩、稳定秩、参与比、谱熵、各向异性、谱平坦度、奇异值谱、谱衰减曲线等指标的计算与可视化；训练采用PyTorch AMP，保存每轮检查点以实现时序分析。

**📊 数据集**

数据集为ImageNet-100（ImageNet的100类子集），在此上训练ViT并进行持续的表示几何分析。

**📈 对比分析**

方法上通过对比不同训练轮次和不同层级的多种谱指标变化来评估几何演化；并以CLS层的表现为重点，观察其有效秩和各向异性的显著变化。虽然未给出显式的性能数值（如准确率）对比，但结果表明随着训练，表示维度利用度提升、能量分布更加均匀，暗示模型学习到更分散、更低相关性的特征。

**⚠️ 局限性**

局限性包括：①仅进行观察性分析，未揭示具体因果机制；②未对token级别的相似性或语义扩展进行直接量化；③只使用了单一模型（ViT-Small/16）和单一数据集，泛化性未知；④缺乏与下游任务性能的直接关联分析。

---

## 472. A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development

**arXiv ID:** 2606.19230 | [PDF](https://arxiv.org/pdf/2606.19230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 473. TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology

**arXiv ID:** 2606.19245 | [PDF](https://arxiv.org/pdf/2606.19245v1)

**作者:** Hannah Le `[一作]`, Kenny Workman `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了TxBench-PP基准，评估AI代理在小分子临床前药理决策中的推理与决策能力。

**💡 创新点**

创新点在于引入可验证、以真实实验数据为核心的决策评估框架，聚焦小分子药物发现的临床前阶段。

**🔧 技术方法**

采用大语言模型（Claude Opus、GPT‑5.x、Gemini、Grok等）结合工具调用（Pi、Claude Code、OpenAI Codex）进行任务推理。

**📊 数据集**

使用来自100个真实实验工作流的多类型数据集，包括筛选、靶点验证、药效学、PK/ADMET等实验结果与文件。

**📈 对比分析**

对16种模型–工具组合进行三次重复实验，计算终点通过率；最佳组合Claude Opus 4.8/Pi通过率59.3%，整体仍低于60%，显示显著提升但仍存在不足。

**⚠️ 局限性**

限制在于代理仍无法可靠完成决策，主要错误源自科学判断、统计与QC问题；基准仅涵盖小分子临床前，未扩展至其他药物类别或临床阶段。

---

## 474. Runtime Compliance Verification for AI Agents

**arXiv ID:** 2606.19242 | [PDF](https://arxiv.org/pdf/2606.19242v1)

**作者:** Nafiseh Kahani `[一作]` (Carleton University), Diana Addae `[通讯]` (Carleton University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于运行时监测的 GDPR 合规验证框架，对 AI 代理的工具调用和自然语言输出进行事件追踪与合规性检查；

**💡 创新点**

创新点在于将 GDPR 的四项原则转化为可在线评估的谓词，结合会话历史和事件注解实现真正的状态感知合规 enforcement；

**🔧 技术方法**

技术采用了事件截取器、类别提取器、基于谓词的 Python 监测器、OPA/Rego 以及 MFOTL 形式化验证；

**📊 数据集**

使用了四个行业案例（零售、航空、医疗、金融）与 DSPy 生成的攻击对话和真实语料库；

**📈 对比分析**

与无监控、随机、正则词典、Presidio 等基线对比，监测器在理想提取下 0% 违规成功率，10% 提取噪声时保持 ≤12% 的攻击成功率，误报率 ≤16%；

**⚠️ 局限性**

主要局限在于对类别提取器的高度依赖，提取不准确会导致漏报或误报，并且目前仅验证了单一模型和单一任务，缺乏对更广泛场景和并发会话的评估。

---

## 475. The Reward Was in Your Data All Along: Correcting Flow Matching with Discriminator-Guided RL

**arXiv ID:** 2606.19162 | [PDF](https://arxiv.org/pdf/2606.19162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 476. FineCombo-TTS: Collaborative and Precise Controllable Speech Synthesis Using Text Descriptions and Reference Speech

**arXiv ID:** 2606.19209 | [PDF](https://arxiv.org/pdf/2606.19209v1)

**作者:** Shuoyi Zhou `[一作]`, Zhiyong Wu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FineCombo-TTS框架，实现了参考语音与文本描述的联合控制；

**💡 创新点**

创新点在于统一声学属性空间并采用Conditional Flow Matching预测器实现细粒度属性转换；

**🔧 技术方法**

使用FACodec提取音色、残差风格编码器、T5文本编码器、Conditional Flow Matching与Transformer解码器；

**📊 数据集**

使用构建的FineEdit数据集（对齐的三类属性对）以及LibriTTS-R、ESD、EmoVoice等公开数据；

**📈 对比分析**

在Prosody、Emotion、Timbre三项控制上相较于基线VoxInstruct-Joint获得更高MOS、情感准确率和更低属性漂移，表现优异；

**⚠️ 局限性**

局限在于对齐数据量大且依赖于预训练模型，模型训练复杂，且跨域适应性和实时性能仍待提升。

---

## 477. PhantomSkill: Malicious Code Injection in Agent Skill Ecosystems

**arXiv ID:** 2606.19191 | [PDF](https://arxiv.org/pdf/2606.19191v1)

**作者:** Yu-Ting Lin `[一作]` (National Yang Ming Chiao Tung University), Chia-Mu Yu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一种名为PhantomSkill的供应链攻击框架，将恶意功能隐藏在LLM编程代理的辅助资源中。

**💡 创新点**

创新点在于把恶意逻辑改写成触发式漏洞形态，借助CWE-guided rewriting把明显的恶意指令转化为普通脆弱代码，从而规避文本注入与传统恶意脚本检测。

**🔧 技术方法**

核心技术包括LLM驱动的生成器、VulMask（CWE‑guided rewriting）、Embed嵌入过程以及在隔离环境中的功能验证与自动化审计。

**📊 数据集**

使用了从公开与社区仓库中收集的技能语料库，涵盖Git自动化、文件处理、编码助手等多种类型，并对其可执行辅助资源进行标记与评估。

**📈 对比分析**

在GPT‑5.5等主流LLM环境下与Prompt Injection、Overt Malicious Script、Hidden Script Attack及Crazy‑Ivan基线对比，PhantomSkill实现58.8% 的攻击成功率、11.4% 的警告率、并保持96.6% 的原始功能可用，显著提升了隐蔽性与有效性。

**⚠️ 局限性**

局限性包括仅针对当前代理框架与技能架构评估、依赖预设的CWE列表、未覆盖更复杂的执行模型与权限体系、以及缺乏对新兴安全工具与平台的全面测试。

---

## 478. Hardware- and Vision-in-the-Loop Validation of Deep Monocular Pose Estimation for Autonomous Maritime UAV Flight

**arXiv ID:** 2606.19176 | [PDF](https://arxiv.org/pdf/2606.19176v1)

**作者:** Maneesha Wickramasuriya `[一作]` (George Washington University), Murray Snyder `[通讯]` (George Washington University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了硬件验证的视角‑in‑loop框架，利用室内环境模拟逼真海上场景，实现无人机自主起降、航迹跟踪与降落；

**💡 创新点**

创新点在于：①将3D Gaussian Splatting渲染与深度Transformer单目姿态估计TNN‑MO相结合；②提出并实现延迟卡尔曼滤波（DKF）补偿网络推理与通信延迟；③在真实硬件上完成完整感知‑估计‑控制闭环验证，首次量化延迟补偿对稳态控制的重要性；

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting (3DGS) 实时渲染、Transformer Neural Network Multi‑Object (TNN‑MO) 单目姿态估计、延迟卡尔曼滤波 (DKF)、几何轨迹控制、Jetson Orin NX 边缘推理、Wi‑Fi 数据传输、Vicon 运动捕捉定位、VectorNav VN‑100 IMU；

**📊 数据集**

数据集主要为从海上船舶收集的多视角图像构建的3DGS场景模型，用于训练与测试TNN‑MO；实验验证使用Vicon提供的真实位姿轨迹；未公开使用标准公开数据集；

**📈 对比分析**

通过与Vicon ground‑truth 对比评估估计误差和控制误差：位置MAE 0.066 m、速度MAE 0.032 m/s、姿态MAE 2.13°；控制误差位置0.089 m、速度0.062 m/s、姿态4.00°；实验显示在约0.55 s 总延迟下，DKF 能保持估计一致性并实现稳定闭环飞行；未使用延迟补偿时性能显著下降；

**⚠️ 局限性**

局限性包括：仅在室内固定场景验证，未覆盖海上动态风浪、波浪与船体运动；网络延迟和GPU资源限制导致推理频率受限；缺乏大规模多机实验；未使用真实相机，渲染图像与实际海景仍有差距。

---

## 479. Learning User Simulators with Turing Rewards

**arXiv ID:** 2606.19336 | [PDF](https://arxiv.org/pdf/2606.19336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 480. Risk Stratification for ICU Delirium using Pervasive Ambient Sensing Information

**arXiv ID:** 2606.19292 | [PDF](https://arxiv.org/pdf/2606.19292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 481. Reference-Driven Multi-Speaker Audio Scene Generation from In-the-Wild Priors

**arXiv ID:** 2606.19325 | [PDF](https://arxiv.org/pdf/2606.19325v1)

**作者:** Michael Finkelson `[一作]` (Lightricks), Yoav HaCohen `[通讯]` (Lightricks)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种通过自由文本提示和多份参考声音生成多说话人对话场景的框架。

**💡 创新点**

创新点在于仅使用文本提示即可完成说话人绑定，解决了“参考快捷路”问题，并通过高噪声偏置的时间分布让模型真正依赖文本。

**🔧 技术方法**

采用基于流匹配的 LTX‑2 音频 Transformer，并在输入中拼接参考潜在向量、轻量化位置编码以及对文本的跨注意。

**📊 数据集**

使用构建的多参考数据集以及公开的 CoVoMix2‑Dialogue‑20s 与 CoVoMix2‑Dialogue‑WildRef 进行训练与评测。

**📈 对比分析**

与多说话人对话 TTS 基线（MOSS‑TTSD、VibeVoice、ZipVoice‑Dialog、Dia）对比，本文方法在说话人绑定指标（cpWER、cpSIM、ACC）和整体音频质量指标（WER、SIM‑O、SQUIM、UTMOS）上均取得最优或同等表现。

**⚠️ 局限性**

局限性包括生成时长上限 20 秒、一次最多 3 个参考说话人，以及流匹配模型需要预先设定生成时长。

---

## 482. Mean-Payoff-Parity and Lifting Strategies from MDPs to 2-Player Stochastic Games

**arXiv ID:** 2606.19324 | [PDF](https://arxiv.org/pdf/2606.19324v1)

**作者:** Mohan Dantam `[一作]` (University of Oxford), Richard Mayr `[通讯]` (University of Edinburgh)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在两人零和随机博弈中最优策略的记忆与随机化复杂度，提出并证明了在从马尔可夫决策过程提升策略时，记忆无关的随机策略仍需指数级记忆，并给出了在均值支付‑优先级目标下最优随机策略仅需线性记忆的结果；

**💡 创新点**

创新点在于：①证明了在转移无关逆子混合目标下，尽管在MDP中记忆无关随机策略足够，但在二人博弈中仍需指数级记忆；②揭示了均值支付‑优先级目标在随机策略下的记忆需求从指数降至线性；③构造了反例证明“从MDP到SG的记忆无关策略提升”在随机策略上不成立；

**🔧 技术方法**

主要技术包括：记忆基础策略的Mealy机构造、逆子混合与平移不变性的理论分析、基于阶段的概率时钟策略设计、指数-线性记忆下界与上界构造、以及多维均值支付目标的分层分色技术；

**📊 数据集**

本文不使用实验数据集，全部通过理论构造与证明完成；

**📈 对比分析**

比较方法为理论复杂度对比：在MDP中记忆无关随机策略达到 0 近似，而在二人游戏中需指数/线性记忆；结果表明随机化可显著降低记忆需求，但并未完全消除；

**⚠️ 局限性**

局限性包括：仅针对平移不变逆子混合目标与均值支付‑优先级目标，其他目标的记忆复杂度尚未阐明；且对随机策略的记忆更新方式（需随机化还是确定性）仍未解决。

---

## 483. Modeling Branches for Active Manipulation using Iterative Parameter Estimation

**arXiv ID:** 2606.19314 | [PDF](https://arxiv.org/pdf/2606.19314v1)

**作者:** Madhav Rijal `[一作]` (West Virginia University), Yu Gu `[通讯]` (West Virginia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建基于点云的三维枝条模型，利用有限元方法迭代估计材料参数，并在变形能量约束下规划主动枝条操控路径。

**💡 创新点**

首次将三维点云压缩成骨架、生成四面体网格并结合St. Venant‑Kirchhoff模型与Nelder–Mead迭代参数估计实现高精度枝条变形仿真；同时在RRT*规划中加入变形能量成本，完成可感知变形的主动枝条操控。

**🔧 技术方法**

Laplacian contraction骨架提取、fTetWild四面体化、FEM（StVK）、Nelder–Mead优化、RBF变形能量预测、D‑RRT*规划、YOLO‑26点云分割、Intel RealSense D455感知。

**📊 数据集**

实验使用三类枝条（幼苗、成熟、人工）配合人工花，采集RGB+深度点云、力/扭矩传感数据。

**📈 对比分析**

对比传统RRT*与D‑RRT*的10次实验，D‑RRT*平均路径长度增长8.1%，变形能量下降35.7%，显示规划显著降低枝条受力。

**⚠️ 局限性**

计算成本高、未考虑叶片遮挡、参数估计离线且需手工标记点；未来需实现在线学习与叶子遮挡补偿。

---

## 484. NeSyCat Torch: A Differentiable Tensor Implementation of Categorical Semantics for Neurosymbolic Learning

**arXiv ID:** 2606.19279 | [PDF](https://arxiv.org/pdf/2606.19279v1)

**作者:** Daniel Romero Schellhorn `[一作]` (University of Osnabrück), Björn Gehrke `[通讯]` (University of Osnabrück)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了NeSyCat框架的神经网络实现——NeSyCat Torch，利用单个参数化Monad统一符号推理与学习；

**💡 创新点**

创新点在于将传统神经符号系统的多种语义（经典、模糊、概率）归纳为单一Monad驱动的归纳真值定义，并通过Monad多层（分布、张量、批处理）实现高效懒惰推理与训练；

**🔧 技术方法**

核心技术包括分布式Monad（分布/张量Monad）、日志空间张量Monad（LogTens）实现数值稳定的梯度传播、批处理Monad（Batch）以及在Haskell/Python JAX/PyTorch中编写的Polymorphic Do-notation实现；

**📊 数据集**

使用MNIST单位数字加法数据集（单/多位），在离散监督下仅观察和不观察数字本身；

**📈 对比分析**

与LTN、DeepProbLog、DeepStochLog等基线相比，NeSyCat在单位数字加法上实现了约94.6%的和正确率，训练时间仅约0.5 ms/step，显著快于对比模型，且准确率与DeepStochLog相当；

**⚠️ 局限性**

局限在于仅处理有限支持分布和离散域；对连续概率、无限域及更复杂任务的推广仍待研究，且网络实现对张量维度的支持有限。

---

## 485. A Unified Framework for Efficient Remote Sensing Visual Question Answering: Adapting Dual, Hybrid, and Encoder-Decoder Architectures

**arXiv ID:** 2606.19277 | [PDF](https://arxiv.org/pdf/2606.19277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 486. Shape Sensing of Continuum Robots using Direct Laser Writing

**arXiv ID:** 2606.19265 | [PDF](https://arxiv.org/pdf/2606.19265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 487. A Multi-Domain Benchmark for Detecting AI-Generated Text-Rich Images from GPT-Image-2

**arXiv ID:** 2606.19259 | [PDF](https://arxiv.org/pdf/2606.19259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 488. Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning

**arXiv ID:** 2606.19340 | [PDF](https://arxiv.org/pdf/2606.19340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 489. Do as I Do: Dexterous Manipulation Data from Everyday Human Videos

**arXiv ID:** 2606.19333 | [PDF](https://arxiv.org/pdf/2606.19333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 490. X+Slides: Benchmarking Audience-Conditioned Slide Generation

**arXiv ID:** 2606.19256 | [PDF](https://arxiv.org/pdf/2606.19256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 491. Native Active Perception as Reasoning for Omni-Modal Understanding

**arXiv ID:** 2606.19341 | [PDF](https://arxiv.org/pdf/2606.19341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 492. Freeing the Law with LOCUS: A Local Ordinance Corpus for the United States

**arXiv ID:** 2606.19334 | [PDF](https://arxiv.org/pdf/2606.19334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 493. Observability and Consistency Analysis for Visual-Inertial Navigation with Anchored Feature Parameterizations

**arXiv ID:** 2606.19307 | [PDF](https://arxiv.org/pdf/2606.19307v1)

**作者:** Mitchell Cohen `[一作]` (McGill University), James Richard Forbes `[通讯]` (McGill University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对使用锚定特征表示的滤波式视觉惯性导航系统（VINS）进行可观测性与一致性分析，并提出两种一致性改进方法。

**💡 创新点**

证明锚定特征参数化下不可观测子空间与特征状态无关，减少不一致性；同时给出在此框架下应用FEJ与RI‑EKF的具体方案。

**🔧 技术方法**

使用 EKF、First‑Estimate Jacobian (FEJ)、右不变 EKF (RI‑EKF)、观测矩阵分析、Monte‑Carlo 仿真、TUM‑VI 真实数据集。

**📊 数据集**

使用 TUM‑VI 室内 monocular 与 stereo 轨迹（含 5 个子序列）以及三条仿真轨迹（TUM Corridor、Udel Gore、Udel ARL）。

**📈 对比分析**

通过 NEES（一致性指标）、ATE（绝对轨迹误差）和 RPE（相对位姿误差）与多种配置（Std、FEJ、RI）对比。结果显示，锚定特征在噪声较大时保持更好的一致性与准确性，即使不使用一致性改进，Std‑AID 亦能获得接近最佳的性能。

**⚠️ 局限性**

仅针对滤波式 VINS，未分析优化式 VINS；对极长轨迹仍受导航状态线性化误差影响，需进一步研究。

---

## 494. Correct Yourself, Keep My Trust: How Self-Correction and Social Connection Shape Credibility in Social Chatbots

**arXiv ID:** 2606.19286 | [PDF](https://arxiv.org/pdf/2606.19286v1)

**作者:** Biswadeep Sen `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对社交聊天机器人错误纠正策略进行实验比较。

**💡 创新点**

发现自我纠正既能纠正错误又能维持可信度，并且社交关系仅在自我纠正时提升纠正效果。

**🔧 技术方法**

使用 GPT‑4o 与 UChat 结合设计对话流程。

**📊 数据集**

在 Prolific 上随机抽取 120 名受试者，收集信任、专业度、社会吸引等问卷评分。

**📈 对比分析**

采用单因素 ANOVA 与 Dunnett 检验，三种纠正方式对信念变化相当，但仅自我纠正在信任与专业度上显著高于外部纠正；社会连接对自我纠正的效果具有显著正相关。

**⚠️ 局限性**

受试者仅一次交互，未检验多次错误或长期交互的持久性。

---

## 495. Structured Inference with Large Language Gibbs

**arXiv ID:** 2606.19264 | [PDF](https://arxiv.org/pdf/2606.19264v1)

**作者:** Sanghyeok Choi `[一作]` (University Of Edinburgh), Esmeralda S. Whitammer `[通讯]` (University Of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用大型语言模型（LLM）条件分布作为转移算子进行Gibbs式采样的框架，即 Large Language Gibbs，用以在结构化变量上实现概率一致的推理。

**💡 创新点**

创新点在于将LLM的next‑token条件分布转化为可迭代重采样的MCMC转移，消除单次自回归生成的顺序偏差，并给出理论上可达稳态分布的构造与足够条件；同时在一致推理和贝叶斯结构学习中演示了该方法的有效性。

**🔧 技术方法**

技术手段包括：Gibbs采样与其变种（扫频、块式、Barker/Gambling 接受规则）；利用LLM的序列化上下文生成条件分布；使用熵/似然方法构造LLM先验；在贝叶斯结构学习中结合合成数据作为正则化先验；以及对比实验中使用的自动回归、多轮推理和传统一致性最大化算法。

**📊 数据集**

实验使用的主要数据集有：synthetic Uniform/Normal 分布、GSM8K 与 TruthfulQA 认知一致性任务、BnRep（4个子数据集）用于贝叶斯结构学习；还对 Llama‑3.1‑8B 与 OLMo‑3‑32B 两大模型进行验证。

**📈 对比分析**

与基线（单次自回归、k‑pass 生成、ICM、一致性最大化、Uniform 先验等）比较，Large Language Gibbs 在一致推理上达成 0.895±0.002 的准确率（对比 ICM 0.724±0.037），在结构学习中相较 Uniform/Direct 方案获得更低的 Structured Hamming Distance 与更高的 AUROC，表明迭代重采样显著提升性能。

**⚠️ 局限性**

限制包括：仅在参数规模 ≤32B 的 LLM 上测试；相较单次生成需要更多计算和迭代收敛；对 LLM 训练数据偏差的依赖可能导致生成样本仍含有系统性误差；若变量空间过大或分布极不平衡，转移不易收敛。

---

## 496. Rethinking Reward Supervision: Rubric-Conditioned Self-Distillation

**arXiv ID:** 2606.19327 | [PDF](https://arxiv.org/pdf/2606.19327v1)

**作者:** Siyi Gu `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于rubric的自我蒸馏框架——Rubric-Conditioned Self-Distillation（RCSD），通过在自我蒸馏中使用结构化rubric作为教师端的细粒度指导，直接对学生自身生成的推理轨迹进行token级别的监督。

**💡 创新点**

创新点在于：①不将rubric压缩为标量奖励，而是将其作为教师的条件输入；②在自我蒸馏中使用rubric，保持多维评价结构；③采用两阶段流程，先训练rubric生成器再用其生成的rubric引导推理器。

**🔧 技术方法**

技术包括：链式思考（CoT）生成、On-Policy Self-Distillation（OPSD）改进、前向KL散度蒸馏、LoRA参数微调、FlashAttention加速、两阶段训练（rubric生成 + rubric-conditioned reasoning）。

**📊 数据集**

使用的公开数据集有：RaR-Science、RubricHub、GPQA-Diamond、SciBench、PIQA、ResearchQA，以及医学领域的MedMCQA、PubMedQA等。

**📈 对比分析**

与基线对比（SFT、GRPO、GRPO-Rubric、OPSD）在多项科学推理基准上平均提升0.9点，最高提升8.2点；在医学基准上也保持竞争力；通过实验验证前向KL蒸馏最佳，rubric生成器效果与手工rubric相近。

**⚠️ 局限性**

局限性包括：仍需依赖教师模型生成rubric，教师质量影响训练；对复杂或极不确定的rubric仍可能产生误导；在极大模型或高成本环境下两阶段训练可能耗时；缺乏对不同领域跨域鲁棒性的深入分析。

---

## 497. Explaining Attention with Program Synthesis

**arXiv ID:** 2606.19317 | [PDF](https://arxiv.org/pdf/2606.19317v1)

**作者:** Amiri Hayes `[一作]` (New Jersey Institute of Technology), Jacob Andreas `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过使用大型语言模型对Transformer注意力头进行程序合成，生成可执行的Python函数以近似并解释各注意力头的计算过程。

**💡 创新点**

创新点在于将注意力头的解释从自然语言描述转变为可执行程序，能够直接替换神经组件并验证因果效果，从而实现可验证的符号化解释。

**🔧 技术方法**

采用交互式程序合成代理（LM驱动的程序生成）、Jensen‑Shannon距离筛选、IOU相似度评估、以及程序库迭代改进等技术。

**📊 数据集**

使用 TinyStories 数据集生成注意力模式，六个 QA 基准（HellaSwag、PIQA、SciQ、ARC‑Easy、Social IQA、COPA）进行下游评估，模型包括 BERT‑Base、GPT‑2‑Small、TinyLlama‑1.1B 与 Llama‑3B。

**📈 对比分析**

通过 IOU 对齐与因果替换的困惑度/准确率比较，结果显示平均 IOU 可达 79%，30–40% 的注意力头可被程序替换而对下游任务影响仅约 16% 的困惑度提升，性能保持稳定。

**⚠️ 局限性**

局限在于仍有大量注意力头的 IOU 低于 40%，程序复杂度有限，高比例替换时仍会导致任务性能下降，需进一步提升合成策略与多轮反馈机制。

---

## 498. QDSV: A Semantic Problem Representation and Multi-Backend Execution Framework for Quantum-Oriented Computation

**arXiv ID:** 2606.19312 | [PDF](https://arxiv.org/pdf/2606.19312v1)

**作者:** Jaime Alexander Jimenez Lozano `[一作]`, Sebastian Jimenez Giraldo `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种基于谓词的语义执行框架QDSV，能够在不依赖电路优先的前提下，将问题描述映射为状态空间+谓词，并在多后端实现中保持语义一致性。

**💡 创新点**

创新点在于将谓词模型与语义化表示、声明式意图语言QIntent以及多后端验证流程相结合，实现从声明式问题到后端执行的完整闭环，并提供可追溯的执行轨迹和可靠性评估。

**🔧 技术方法**

采用QDSV语义表示、QIntent声明式表面语言、Qruba工作流环境，并在QuEST、Aer模拟器和IBM Quantum硬件上实现状态向量、电路兼容和硬件执行；同时使用经典机器学习与电路优先VQC基线进行对比。

**📊 数据集**

使用Bonn和Delhi两套脑电（EEG）数据集进行癫痫发作与非发作分类实验，另外在贷款批准数据集上做了语义再配置实验。

**📈 对比分析**

与经典DecisionTree/RandomForest和VQC基线相比，QDSV在模拟器上实现的F1≈0.97，硬件子集上实现的F1≈0.98，表现优于VQC基线，且保持与后端一致的语义结构；但并未宣称量子优势。

**⚠️ 局限性**

局限包括仅在子集硬件上验证、缺乏完整的量化基准、对噪声与近似后端的正式语义支持不足、以及未提供完整实现规范。

---

## 499. Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play

**arXiv ID:** 2606.19308 | [PDF](https://arxiv.org/pdf/2606.19308v1)

**作者:** Leyang Shen `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Multi-Agent Fictitious Play（MAFP）框架，用多智能体协同求解决策任务中的立场纠缠问题

**💡 创新点**

创新点在于将立场拆解为代理，并通过迭代的假想游戏（fictitious play）在语言空间中进行最佳响应，解决递归预期难题

**🔧 技术方法**

利用大型语言模型（LLM）实现初始化、聚合（Agg_M）与最佳响应（BR_M）操作，并在多轮迭代中构建策略

**📊 数据集**

使用13个多阶段对抗与谈判场景的基准数据集（包括TicTacToe、Nim、IPD、ConnectFour、Pig、Kuhn Poker、Liar's Dice等）

**📈 对比分析**

与单轮与多轮的多种基线（CoT、SR、Debate、ToM）进行对弈和鲁棒性评估，MAFP在平均对弈强度（TS）和鲁棒性（Rob）上均取得最高分（TS≈0.533，Rob≈0.421），显著优于其他方法

**⚠️ 局限性**

局限性包括：实验规模受算力限制，未覆盖更大规模真实世界决策场景；理论上对语言空间中的收敛性、平衡选择及主动引导等问题缺乏深入分析

---

## 500. P-K-GCN: Physics-augmented Koopman-enhanced Graph Convolutional Network for Deep Spatiotemporal Super-resolution

**arXiv ID:** 2606.19303 | [PDF](https://arxiv.org/pdf/2606.19303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 501. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models

**arXiv ID:** 2606.19297 | [PDF](https://arxiv.org/pdf/2606.19297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 502. Scoring Backends Matter More Than Pooling: A Systematic Study of Training-Free Anomalous Sound Detection under Domain Shift

**arXiv ID:** 2606.19269 | [PDF](https://arxiv.org/pdf/2606.19269v1)

**作者:** Jingwen Zhou `[一作]` (Xidian University), Mingzhe Wang `[通讯]` (Xidian University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文在不进行任何训练的前提下，系统性比较了不同后端评分方法（最近邻余弦、马氏距离、局部密度归一化kNN和PCA子空间残差）与不同时间池化方式对跨域异常声音检测性能的影响。

**💡 创新点**

主要创新在于揭示在冻结的BEATs编码器和固定池化策略下，后端评分方式对目标域AUC的影响远大于池化方式，并且发现后端具有机器相关的稳定优劣模式；此外提出了一种基于自我评分的z归一化后端融合方案，能够在不使用标签或额外数据的情况下逼近每机最佳后端的性能。

**🔧 技术方法**

研究使用了BEATs_iter3+预训练音频编码器进行特征提取，结合四种经典异常评分后端，三种统计池化（均值、GeM、最大），以及基于自归一化的最小值融合。

**📊 数据集**

实验数据来自DCASE 2023 Task 2开发集（7台机器）和DCASE 2025开发集（2台机器），每台机器提供1000个正常训练片段与200个测试片段（源域990/目标域10）。

**📈 对比分析**

通过AUC和pAUC指标比较，后端切换可平均提升13.8分（最大53.8分），而池化切换仅提升3.2分；融合方法在目标域的调和平均AUC上达到63.3%（仅比每机最佳后端低1.1%），并在源域性能上保持不下降。

**⚠️ 局限性**

局限性包括仅在片段级别研究，未考虑帧级或补丁级记忆库；仅使用单一BEATs编码器，可能不适用于其他预训练模型；后端超参数（k=5、90%方差保留、GeM p=3）未进行敏感性分析；跨年份验证仅覆盖两台机器；并且伪验证后端选择在无标签情况下表现不佳。

---

## 503. Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA

**arXiv ID:** 2606.19266 | [PDF](https://arxiv.org/pdf/2606.19266v1)

**作者:** Ikram Belmadani `[一作]` (Aix-Marseille University), Benoit Favre `[通讯]` (Aix-Marseille University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在法语医学问答数据集上对大型语言模型的领域适配方法进行系统、可复现的实验，比较连续预训练（CPT）、监督微调（SFT）及其组合在不同模型族、规模和初始化状态下的效果，并探讨跨语言迁移、翻译基准对评估的影响。

**💡 创新点**

创新点在于：① 采用受控实验框架，明确区分模型初始化、适配策略、解码方式与评估任务，从而真正剖析各因素的贡献；② 同时评估多任务（多选、单选、开放式回答）和多语言（法语、英语）性能；③ 基于LLM-as-a-Judge和统计显著性检验，提供实用的资源约束下的适配指南。

**🔧 技术方法**

使用技术包括：CPT（完整参数微调）、SFT（DoRA 参数高效微调）、instruction tuning、受限解码、自动评估指标（EM、Hamming、ROUGE‑L、BERTScore、BLEU、METEOR）、LLM‑as‑a‑Judge、bootstrap 统计显著性检验、模型对比实验。

**📊 数据集**

主要数据集为：NACHOS（4 GB 法语医学文本）用于CPT；MedInjection‑FR（543 k 说明-回答对，包含多选与开放式）用于SFT；MedInjection‑FR 测试集（14 533 本地法语、13 293 翻译版）及对应英语基准用于跨语言评估。

**📈 对比分析**

比较方法：在零样本、受限解码设置下，对每种模型初始化和适配策略分别计算 EM/Hamming（多选/单选）、ROUGE‑L/BERTScore、LLM‑Judge；采用 bootstrap 10k 次、Bonferroni 校正检验显著性。结果显示：SFT 在 MCQA 上已是成本效益最佳，CPT+SFT 仅提升 1–1.5 分且往往不显著；CPT 在 OEQA 的重叠指标上最有利，但与 SFT 组合时效果不稳定；跨语言实验表明法语适配可提升英语基准，翻译基准导致准确率与置信度偏高。

**⚠️ 局限性**

局限性包括：仅评估 CPT 与 SFT（未涉及 RLHF、few‑shot、人工评审成本等）；OEQA 评价依赖重叠指标与 LLM‑Judge，无法完全衡量临床正确性与推理质量；翻译基准可能导致评估偏差；实验仅覆盖法语医学场景，结果对其他语言或领域的普适性未知。

---

## 504. Detecting Hidden ML Training With Zero-Overhead Telemetry

**arXiv ID:** 2606.19262 | [PDF](https://arxiv.org/pdf/2606.19262v1)

**作者:** Robi Rahman `[一作]` (Machine Intelligence Research Institute), Sabiha Tajdari `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究利用零开销的 NVIDIA NVML 监测信号训练随机森林分类器，能够识别 GPU 是否正在进行模型训练，并抵御多种对抗性隐蔽策略。

**💡 创新点**

其创新点在于仅使用可公开获取、隐私友好的 NVML 计数器实现 98.2% 的训练与非训练二分类精度，并在五轮对抗迭代中证明对多 GPU 及白盒攻击的鲁棒性。

**🔧 技术方法**

采用随机森林模型，结合功耗、温度、利用率等结构性与时域特征，并通过对抗训练循环不断提升鲁棒性。

**📊 数据集**

数据集包含约 117 GPU 小时、9 种 NVIDIA GPU，涵盖 93 个真实训练、40 个推理、19 个非 ML 工作负载，以及 180+ 个对抗性实验。

**📈 对比分析**

与传统需要 1200–5300% 运行时开销的 Nsight Compute 方案相比，该方法仅以 1 Hz 采样实现 98%+ 的准确率；在留一 GPU 交叉验证中达到 99.4–100%；对抗迭代后单 GPU 的误报率仅为 6.2%。

**⚠️ 局限性**

主要限制是对硬件可信度的依赖，需硬件安全模块来防止 NVML 伪造、通道窃听及模型替换；且白盒 LoRA 等策略仍可将检测率降至 43–87%，对极端侵入性攻击仍存在脆弱点。

---

## 505. Digital Speech Acts Retain Control of Copyright with People, Not Platforms

**arXiv ID:** 2606.19263 | [PDF](https://arxiv.org/pdf/2606.19263v1)

**作者:** James Golike `[一作]`, Ehud Shapiro `[通讯]` (London School of Economics)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

**🎯 论文内容**

阐述数字言语行为在草根网络中的版权保护与治理潜力

**💡 创新点**

将 Burrow‑Giles、Feist 等版权前例应用于数字签名表达，提出数字言语行为可被视为可版权化表达并实现去中心化治理

**🔧 技术方法**

法律推理、数字签名与分布式协议

**📊 数据集**

无特定数据集

**📈 对比分析**

无实验对比，主要以理论与案例分析为依据

**⚠️ 局限性**

未考量技术实现细节与实际部署成本，缺乏实证验证

---

## 506. UBP2: Uncertainty-Balanced Preference Planning for Efficient Preference-based Reinforcement Learning

**arXiv ID:** 2606.19328 | [PDF](https://arxiv.org/pdf/2606.19328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 507. NeuMesh++: Towards Versatile and Efficient Volumetric Editing with Disentangled Neural Mesh-based Implicit Field

**arXiv ID:** 2606.19316 | [PDF](https://arxiv.org/pdf/2606.19316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 508. Data Intelligence Agents: Interpreting, Modeling, and Querying Enterprise Data via Autonomous Coding Agents

**arXiv ID:** 2606.19319 | [PDF](https://arxiv.org/pdf/2606.19319v1)

**作者:** Anoushka Vyas `[一作]` (C3 AI), Henrik Ohlsson `[通讯]` (C3 AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Data Intelligence Agents（DIA）系统，通过单一的自动编码代理（ACA）在共享工作空间中实现数据解释、模式创建和查询生成三大功能，压缩了企业数据处理流程的多次人工交接。

**💡 创新点**

创新点在于将ACA视为核心抽象，使得系统能够生成、执行、验证并修复可执行的代码与数据结构，而非仅输出文本；同时采用共享记忆与自校验机制，让一个通用代理在多种SQL方言和任务类型下保持高效性能。

**🔧 技术方法**

采用的技术包括基于LLM驱动的自动编码代理、沙箱化代码执行、共享工作空间与多层记忆（经验回放、会话学习、跨会话规则），以及执行驱动的自校验与迭代改正流程。

**📊 数据集**

评测使用了七个公开SQL基准：BIRD-Dev、BIRD-Critic、LiveSQLBench、BIRD-Interact、Spider2-Lite、Spider2-Snow和Spider2-DBT，覆盖四类任务（生成、调试、会话、项目完成）和四种SQL方言（SQLite、PostgreSQL、Snowflake、DuckDB）。

**📈 对比分析**

与现有最佳模型（如MARS-SQL、ReFoRCE、OpenHands+Claude Sonnet等）相比，DIA在所有七个基准上都达到或超过了最高公布成绩，尤其在会话和调试任务中显著提升（最高+33.0点），证明其通用性与高效性。

**⚠️ 局限性**

主要限制包括：整体执行耗时较高（从几十秒到数分钟不等），对交互式或高吞吐量场景不够友好；自校验仅基于结果形状，易因意图误读导致错误通过；以及实验仅验证了单一LLM、模拟用户与有限记忆机制，缺乏对多模型、多用户场景的深入评估。

---

## 509. Diffusion-Proof: Recipe for Formal Theorem Proving Beyond Auto-Regressive Generation

**arXiv ID:** 2606.19315 | [PDF](https://arxiv.org/pdf/2606.19315v1)

**作者:** Ruida Wang `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于扩散大型语言模型（dLLM）的全流程框架，包括 dLLM-Prover-7B 用于全证书生成和 dLLM-Corrector-7B 用于局部证明纠正，显著提升了 Lean4 形式化证明的长程连贯性。

**💡 创新点**

创新点在于首次将块扩散训练与大块纠正机制结合，利用双向信息实现全局规划与局部纠错，并证明 dLLM 在形式化推理任务中优于传统 AR LLM。

**🔧 技术方法**

采用 Fast-dLLM-V2-7B 为基础，使用块扩散（block diffusion）训练与推理、以及 512 大块扩散纠正训练；还引入了学习率调度、温度控制与自定义 mask 技术。

**📊 数据集**

使用 5.5M Lean 证明语料（含 300k SFT 样本）和 128k 纠正样本，包含自然语言与 Lean4 代码混合的双语数据集；同时评测 MiniF2F-Test 与 ProofNet-Test 两大 benchmark。

**📈 对比分析**

与相同训练数据下的 AR 基线 Qwen-2.5-Lean-SFT-7B 进行 pass@32 对比，dLLM 在 MiniF2F-Test 提升 6.14%（从 43.85% 到 50.00%），在 ProofNet-Test 提升 1.61%（从 5.91% 到 7.53%），并成功解决了一道 IMO 题。

**⚠️ 局限性**

局限性包括：计算资源有限导致模型规模和训练深度受限；缺乏长链式推理（Long CoT）能力；仅针对 Lean4，未扩展到其他定理证明器；缺乏对 dLLM 理论基础的深入分析。

---

## 510. Secret key-distribution over networks with node-based adversarial errors

**arXiv ID:** 2606.19305 | [PDF](https://arxiv.org/pdf/2606.19305v1)

**作者:** Reza Sayyari `[一作]` (University at Buffalo), Michael Langberg `[通讯]` (University at Buffalo)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究在存在主动基于节点的攻击者的网络中，如何通过网络编码实现安全可靠的多键分发（multiple key‑cast）并给出了对应的密钥容量上界和实现方案。

**💡 创新点**

创新点包括：
• 将原先只针对边缘攻击的安全多播与密钥分发模型推广到节点攻击模型；
• 提出完美安全的哈希加密机制，将弱/强安全提升为完全无信息泄露；
• 在每个节点仅 d‑连通的网络、部分连通节点以及多源情形下给出容量公式并给出可实现方案；
• 证明该方案同时适用于安全多播和网络秘密共享。

**🔧 技术方法**

主要技术：
• 基于 Vandermonde 矩阵的 MDS 码实现错误纠正；
• 多项式哈希+一次性密码技术实现完美安全检测；
• 通过分层预分发与哈希校验实现对节点级错误的检测与定位；
• 通过多源随机矩阵与密钥合并实现多源情形下的完美安全。

**📊 数据集**

本文未使用公开数据集，所有分析均为理论证明和信息理论上限，实验验证基于符号级仿真。

**📈 对比分析**

与以往仅支持边攻击或被动窃听的网络编码安全方案相比，本文在节点攻击模型下实现了相同容量（d‑ℓₒ‑ℓₑ‑2ℓₒₑ），且在部分连通网络中提供了显式的容量下界，证明了在满足 d‑连通性约束时仍可实现正容量。

**⚠️ 局限性**

局限性：
• 当网络中存在长链的部分连通节点时，通信开销呈指数增长，导致容量下降；
• 方案对网络拓扑的连通性假设较强（需 d‑连通或部分连通节点满足特定条件）；
• 对于完全任意的攻击者（可在任意节点观察并攻击）仍缺乏最优性证明；
• 实际部署时对大规模网络的实现复杂度与硬件资源需求未详细评估。

---

## 511. Confidence is Not Reliability: Rethinking MC Dropout in Brain Tumour Segmentation

**arXiv ID:** 2606.19300 | [PDF](https://arxiv.org/pdf/2606.19300v1)

**作者:** Xin Ci Wong `[一作]` (University of Leeds), Nishant Ravikumar `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了在脑肿瘤多参数MRI分割任务中使用MC Dropout进行 voxel‑级不确定性估计，并在 126 例 BraTS2021 测试集上对比了两种架构（SegResNet 与 UNet‑Res）的分割精度、误差定位能力和校准性。

**💡 创新点**

发现高 AUROC 与临床安全不等价，提出需对子区域（尤其是治疗关键的增强肿瘤 ET）进行单独校准评估，并通过熵四分位数对患者进行分层筛查，从而提供可直接用于临床工作流的风险标记。

**🔧 技术方法**

采用 MC Dropout（20 次前向推理）生成预测熵、期望熵、互信息等不确定性指标，结合 Dice、AUROC、ECE 与可靠性曲线进行评估；对比时还使用了温度缩放与后置 dropout 注入等技术。

**📊 数据集**

使用 BraTS2021 公开数据集（1251 例训练/验证/测试分割为 1000/125/126 例），每例包含 T1、T1ce、T2、FLAIR 四模态 MRI，预处理为 1 mm 立方体、z‑score 标准化与中心裁剪至 128³。

**📈 对比分析**

通过 Dice（整体、核心、增强肿瘤）对两模型分割精度进行比较；SegResNet 的 Dice 在所有子区域均高于 UNet‑Res，尤其是 ET；两模型在 AUROC‑entropy（≈0.975）和 AUROC‑MI（≈0.97）上相近；但 UNet‑Res 在 ET 子区域的 ECE 高达 0.915，可靠性曲线平坦，表明其对关键子区的置信度严重失效。

**⚠️ 局限性**

局限性：仅在单一公开数据集上验证；未探讨 MC Dropout 推理次数、温度缩放或 Platt 缩放等对 ECE 的影响；SegResNet 使用后置 dropout 与 UNet‑Res 嵌入 dropout 的理论假设不完全一致，导致直接比较受限；缺乏多中心验证与放射科医生工作流评估。

---

## 512. TurboServe: Serving Streaming Video Generation Efficiently and Economically

**arXiv ID:** 2606.19271 | [PDF](https://arxiv.org/pdf/2606.19271v1)

**作者:** Youhe Jiang `[一作]` (Shanghai Jiao Tong University), Jintao Zhang `[通讯]` (Shengshu Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套专为多用户、多GPU环境下的流式视频生成服务设计的系统，能够在线协调会话放置与GPU资源自动伸缩，保持实时生成的延迟目标并降低成本。

**💡 创新点**

1) 结合会话迁移感知的最小-最大重平衡与基于负载的GPU自动伸缩的闭环调度框架；2) 通过GPU–CPU状态迁移、GPU–GPU状态迁移和批量块执行实现高效运行时；3) 在生产流式生成工作负载上展示近似最优的实时调度。

**🔧 技术方法**

事件驱动调度、最小-最大重平衡算法、阈值与比例的自动伸缩控制、GPU–CPU状态迁移、GPU–GPU状态迁移（RDMA/NCCL）、块级合并执行、在线负载反馈闭环控制。

**📊 数据集**

真实的内部生产轨迹（Trace 1–6），涵盖多种LongLive‑style流式视频生成模型（1.3B、7B等），在16×H100和64×B300两种GPU集群上进行实验。

**📈 对比分析**

与TurboServebase、TurboServebase+LAG、TurboServebase+MAG等基线进行对比。结果显示平均降低了37.5% worst‑case per‑chunk latency，平均降低了37.2% GPU运营成本；消融实验验证迁移与伸缩各自贡献；与离线最优Oracle相比，在线方案在成本上仅差≈6%。

**⚠️ 局限性**

仅在内部轨迹和特定模型上验证；对大规模跨节点网络延迟和更大模型的迁移开销未做深入评估；假设GPU成本线性且忽略CPU/网络开销；迁移频率受阈值设置影响，可能在极端负载下导致额外开销。

---

## 513. A Mixed-Reality Testbed for Autonomous Vehicles

**arXiv ID:** 2606.19267 | [PDF](https://arxiv.org/pdf/2606.19267v1)

**作者:** H. M. Sabbir Ahmad `[一作]` (Boston University), Wenchao Li `[通讯]` (Boston University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个混合现实硬件‑在环（HIL）测试平台，将 CARLA 仿真与物理移动机器人无缝集成，并实现了多模态传感、V2X 通信以及基于控制障碍函数（CBF）的在线自监督学习控制框架，以实现安全可靠的 CAV（联网自动驾驶汽车）多智能体研究。

**💡 创新点**

创新点包括：① 将物理机器人与数字孪生在同一场景下同步，突破传统仿真或物理测试单一局限；② 开发可在混合现实环境中训练、微调的在线自监督学习控制器，实时更新 CBF 参数；③ 构建安全保证的端到端框架，可直接覆盖任意 AV 算法的控制指令；④ 通过网页远程接口实现测试平台共享与批量实验。

**🔧 技术方法**

使用技术：CARLA、AgileX Limo 机器人、ROS1/ROS2 及桥接、MOCAP、数字孪生投影、LiDAR/摄像头/IMU 传感器、V2I/V2V 通过 ROS 主题、Control Barrier Function（CBF）/高阶 CBF、QP 优化、在线自监督学习算法、RGB+LiDAR 融合网络。

**📊 数据集**

数据集：主要使用自定义 CARLA 地图（OpenDRIVE、GIS 数据导入）产生的仿真数据，以及在测试平台上收集的物理机器人传感器和轨迹数据；没有引用公开公开的数据集。

**📈 对比分析**

与基准方法比较：对比手工调节的保守/激进 CBF 控制器和 Intelligent Driver Model（IDM）基准；使用成功率、成功率加权燃油消耗、成功率加权行驶时间三项指标。实验表明，本文方法在所有交通密度和天气条件下均达到 100% 成功率，燃油消耗最低且行驶时间保持在合理范围内，明显优于基准。

**⚠️ 局限性**

局限性：① 仅在小规模机器人平台上验证，难以直接推广至大型真实车辆；② 仍未公开平台，外部复现受限；③ 对人类驾驶车辆（HDV）交互的建模与安全保证尚未完善；④ 受限于物理硬件与仿真之间的建模误差，可能需要进一步微调。

---

## 514. Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games

**arXiv ID:** 2606.19338 | [PDF](https://arxiv.org/pdf/2606.19338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. CABLE: Cloud-Assisted Bandwidth-efficient LMM-based Encoding for V2X Systems

**arXiv ID:** 2606.19258 | [PDF](https://arxiv.org/pdf/2606.19258v1)

**作者:** Haohua Que `[一作]` (University of Georgia), Handong Yao `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出CABLE框架，利用云端LMM生成语义分割作为边缘端ROI生成的先验，形成 mask‑ROI‑LMM 反馈循环，实现边缘‑云端协同感知

**💡 创新点**

通过云端LMM反馈与自适应ROI、残差运动能量检测、背景跳过传输和 corridor envelope 连通化，显著降低带宽与预填充成本，且不损失大部分感知质量

**🔧 技术方法**

Ego‑motion 同伦估计、残差运动能量检测、背景跳过传输、Corridor Envelope 合并、LISA++ LMM 推理、RT‑DETR 检测

**📊 数据集**

nuScenes、WOD‑ZB、Waymo、KITTI、CADC 这五个公开数据集

**📈 对比分析**

与仅基于运动差分的 ROI、单一 LMM 闭环传播等方法对比，CABLE 在保持约 98% 识别率的同时实现 73–87% ROI 覆盖率降低、5–8× 预填充速度提升，检测保留率 >97%，整体性能显著优于基线

**⚠️ 局限性**

冷启动/漂移敏感、纯平移同伦假设在大旋转场景下误差累积、对边缘端算子与码流细节未充分评估、仅评估 ROI 像素覆盖而非真实比特率

---

## 516. DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models

**arXiv ID:** 2606.19257 | [PDF](https://arxiv.org/pdf/2606.19257v1)

**作者:** Zirui Wu `[一作]` (University of Hong Kong), Lingpeng Kong `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了8B参数的块扩散推理模型-8B，利用块大小渐进式学习提升长链式推理能力

**💡 创新点**

创新点在于将块大小视为可调节的尺度，采用块大小渐进式学习克服大块训练导致的性能崩溃，并提出基于邻域置信度的RelaxedConfidence解码策略加速并行生成

**🔧 技术方法**

技术包括块扩散语言模型、块大小渐进式学习、低置信度与RelaxedConfidence解码策略、持续预训练与分层微调

**📊 数据集**

使用PromptCoT 2.0数学与代码推理数据集，结合160B训练语料库（OLMo 3、Nemotron Nano V3等），以及公开数学与代码基准（AIME、MATH、LiveCodeBench等）

**📈 对比分析**

与8B规模的自回归基线Qwen3-8B-Thinking以及其他开放式扩散模型对比；-8B在数学与代码任务上与Qwen3-8B相当，超越同规模扩散模型，在不同块尺寸下保持稳健，并在RelaxedConfidence下提升TPF 22.5%–54.5%

**⚠️ 局限性**

局限在于未探索动态/语义边界块划分，主要评估范围局限于数学与代码推理，未系统测试工具使用和更广泛推理场景

---

## 517. SCAN: Enhance Time Series Anomaly Detection via Multi-Scale Neighborhood-Centered Clustering

**arXiv ID:** 2606.19255 | [PDF](https://arxiv.org/pdf/2606.19255v1)

**作者:** Xingze Zheng `[一作]` (East China Normal University), Yang Shu `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将多尺度聚类嵌入重构模型的时间序列异常检测方法

**💡 创新点**

创新点在于：①将聚类嵌入表示层和异常判别层以抑制过度泛化和不足泛化；②引入邻域中心化表示提升聚类质量；③采用层次门控融合聚类中心与原始特征。

**🔧 技术方法**

使用多尺度建模、邻域中心化聚类、交叉注意力机制、门控融合、重构误差与聚类置信度双重异常分数。

**📊 数据集**

在七个工业与交通等真实数据集上评估，包括SMD、MSL、SMAP、PSM、SWAT、NeurIPS-TS（GECCO、SWAN）以及UCR。

**📈 对比分析**

与22个传统与深度基线（如AutoEncoder、TimesNet、ModernTCN、AnomalyTransformer、CrossAD等）进行对比，采用VUS-ROC、VUS-PR等指标，SCAN在所有数据集上均实现SOTA，准确率和稳定性显著提升。

**⚠️ 局限性**

限制在于需预先设定聚类数目，灵活性不足，未来计划探索自适应聚类方法。

---

## 518. Reconstruction Limits for Repeated Differentially Private Aggregates: A Cramer-Rao Perspective on Query Geometry

**arXiv ID:** 2606.19275 | [PDF](https://arxiv.org/pdf/2606.19275v1)

**作者:** Chenyue Zhang `[一作]` (Cornell University), Sean Peisert `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对重复发布的差分隐私聚合进行器量化 Fisher 信息分析，探讨其在隐私会计与发布几何中的效能限制。

**💡 创新点**

明确区分隐私会计与发布几何的角色，证明一次对比后无新目标方向，提出“可识别方向计数”作为设计规则。

**🔧 技术方法**

使用 Fisher 信息分解、特征值谱分析、CRB 计算以及盒约束 Gaussian 最大似然估计等技术。

**📊 数据集**

采用 100 条电力负荷记录（99th 百分位裁剪并归一化到 [-1,1]）以及合成的记录对/三元组作为实验数据。

**📈 对比分析**

将实验结果与 Basic Composition、zCDP 以及固定阶 RDP 的会计模型进行对比，发现 Basic Composition 会使复制发布更差，zCDP/RDP 维持阶平；在特征多样化场景下存在有限最优发布数。

**⚠️ 局限性**

局限在于仅给出局部无偏 CRB，有限噪声下可能出现偏差；分析范围局限于一维/静态聚合，未覆盖标签重识别等更复杂场景。

---

