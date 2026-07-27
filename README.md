# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-27 | 今日论文总数: 395

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Control panels to clarify user intent with Large Language Models

**arXiv ID:** 2607.21598 | [PDF](https://arxiv.org/pdf/2607.21598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 2. Natural Language Processing in Health Professions Education: A Scoping Review

**arXiv ID:** 2607.21605 | [PDF](https://arxiv.org/pdf/2607.21605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 3. Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models

**arXiv ID:** 2607.21636 | [PDF](https://arxiv.org/pdf/2607.21636v1)

**作者:** Jie Zhang `[一作]` `[通讯]` (Accenture Japan), Jie Zhang (Accenture Japan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于分类器二样本检验的依赖性评估方法，将合成表格数据的真实性拆分为边缘、依赖和数值–类别交叉三部分，并用它检测流匹配生成器的依赖缺失。

**💡 创新点**

创新点在于：①用依赖感知的梯度提升树进行二样本检验；②通过列级置换与块级置换分离边缘与依赖贡献；③证明在大容量下均方误差不受结构限制，但缺少直接依赖监督导致依赖缺口仍存在，且传统指标无法捕捉。

**🔧 技术方法**

使用的技术包括：Exponential‑Family Variational Flow Matching (EF‑VFM) 生成器；梯度提升树（GBDT）做强分类器；列置换和块置换构造依赖/交叉分量；与完全因子化基准和真实上界对照；多种可复制指标（F1、Recall、Trend、LR‑C2ST）。

**📊 数据集**

实验数据集包括：Adult、Default、Bank、Magic（数值‑类别混合）以及Shoppers（小样本、数值型控制）。

**📈 对比分析**

与传统的Trend/Logistic‑Regression C2ST 对比，后两者对依赖缺口几乎不敏感；依赖诊断显示 EF‑VFM 仍保留约 0.45–0.49 的依赖差距，导致少数类 F1 下降 0.01–0.05；即使扩大 16× 参数量、加入内置依赖模块或后置 copula 纠正，也未能显著缩小该差距。

**⚠️ 局限性**

局限性在于：①依赖分解基于 AUC 的可加性，未给出真实的统计距离；②交叉项通过块置换近似，可能忽略更细粒度的交互；③结论主要针对 EF‑VFM 与所用数据集，尚未验证在更广泛生成模型或极端类别不平衡场景下的通用性；④未提供直接对依赖监督的目标函数，只给出未来改进方向。

---

## 4. Cloud-Native Evaluation-as-a-Service: A Microservices Architecture for Scalable AI Monitoring with Conformal Guarantees

**arXiv ID:** 2607.21623 | [PDF](https://arxiv.org/pdf/2607.21623v1)

**作者:** Lei Yang `[一作]` `[通讯]`, Lei Yang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一个基于 Kubernetes 的云原生参考架构，将 split conformal prediction、校准评估、多元漂移检测、公平性监控等六个无状态微服务通过 DAG 统一编排，实现了持续可信 AI 部署的可扩展评估流水线。

**💡 创新点**

首次将分割合规预测作为即服务（SaaS）嵌入开源微服务架构，并结合随机 Fourier 特征近似的 MMD 漂移检测与轻量 DAG orchestrator，提供统计保证、可组合性和自动化伸缩，弥补现有工具缺乏正式统计保障和云原生可扩展性的空白。

**🔧 技术方法**

使用 split conformal prediction（finite-sample-corrected Adaptive Prediction Sets）、多变 ECE（Fixed、Adaptive、Classwise、Debiased）、Random Fourier Features 近似 MMD、bootstrap CI 公平性度量、FastAPI/KServe-compatible microservices、Kubernetes HPA、DAG orchestrator（重试/回压/幂等）等技术。

**📊 数据集**

使用合成 logits、OpenAI 500 条 MMLU logits、768 维合成嵌入、UCI Adult 数据集（32k）进行公平性验证，涵盖分类与公平性多维度数据。

**📈 对比分析**

与四个开源工具（TrustLLM、Arize、WhyLabs 等）进行功能对比，证明本系统在合规预测、漂移检测、公平性监控及 DAG 编排方面唯一满足全部需求；统计结果显示覆盖率在 50 次随机划分中平均误差 ≤1.4%，p99 预测集 <2 ms，漂移检测 500 ms；公平性 DP gap 0.33，bootstrap CI 0.28–0.37；整体性能满足云原生持续评估的时延与吞吐要求。

**⚠️ 局限性**

限制包括：仅在单节点多进程模拟，未验证多 Pod 伸缩与网络延迟；仅覆盖分类任务，未实现生成式评估；漂移检测以批量方式，缺乏实时流式检测；公平性验证基于简化的代理分类器，未在真实生产模型上测试；bootstrap CI 假设样本独立，可能不适用于时间序列；最小样本阈值在小群体时可能削弱敏感性。

---

## 5. FrED: External Data Influence Estimation via Domain Knowledge Graph Grounding

**arXiv ID:** 2607.21615 | [PDF](https://arxiv.org/pdf/2607.21615v1)

**作者:** Theodoros Aivalis `[一作]` (National Centre for Scientific Research 'Demokritos'), Joemon M. Jose `[通讯]` (University of Glasgow)

**通讯引用:** 7777 | [OpenAlex ID](https://openalex.org/A5069702331)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种全黑盒训练数据归因框架 FrED，融合连续潜在相似度与离散领域知识图谱来推断生成结果的关键训练样本。

**💡 创新点**

创新点在于：① 采用贝叶斯后验近似逆因果影响；② 将潜在空间评分与KG排名通过非对称提升融合，既保留高维特征，又引入领域结构；③ 通过信息稀缺先验抑制冗余样本，提升稀有样本的影响权重。

**🔧 技术方法**

技术包括：ViT-g-14 视觉编码器、Node2Vec 图嵌入、CLIP式对比学习将视觉嵌入映射到 KG 结构、贝叶斯推断、Rank‑Fusion 的 Boost 机制、Latent‑+‑Domain 双空间评分。

**📊 数据集**

数据集：艺术域使用 ArtBench‑10 及从 WikiArt、Wikipedia、DBpedia 采集的元数据构成的 KG；环境域使用 WeatherBench‑2、Pangu‑Weather 预测数据、1,372 事件的洪水 KG，结合 iNaturalist 与 ESA LandCover 的生态与地形层。

**📈 对比分析**

比较方法：在 ArtBench 上与 CLIP、Raw 像素、梯度、TRAK、D‑TRAK、DAS 等基线对比；在艺术域 FrED 在 LDS（Linear Data‑modeling Score）上达到约 29%/21%（验证/生成），显著优于所有黑盒基线并接近梯度方法；在环境域通过 Prec@K 评估，FrED 的地理精度从 50.9% 提升到 65.7%（Prec@10），在跨国生态一致性上亦保持高水平。

**⚠️ 局限性**

局限性：依赖领域知识图谱，可能存在专家瓶颈；KG 为静态快照，缺乏实时更新；LDS 评估需大量子模型训练，计算成本高；若原始训练集不可获得，需通过抓取或近似分布来补齐。

---

## 6. Vibe Coding in Software Development: A Multivocal Literature Review

**arXiv ID:** 2607.21652 | [PDF](https://arxiv.org/pdf/2607.21652v1)

**作者:** Shahbaz Siddeeq `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10557 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2022‑2025年间关于vibe coding（即通过自然语言与大型语言模型（LLM）进行交互式软件开发）的学术与灰色文献进行系统性多声文学综述（MLR），对其定义、工作流程、开发者角色变化、成效、风险与治理机制等八个研究问题进行归纳与主题合成，并通过证据强度标签评估结论的可靠性。

**💡 创新点**

①首次整合学术与实践来源于单一文献检索与质量评估流程，填补了该领域仅基于学术或灰色文献的孤立综述空白；②提出了“证据强度”四级评估框架，用以区分强、弱、适度与新兴发现；③构建了vibe coding的概念框架与工作流程层级模型，揭示了其与传统代码生成、协作编程及 AI 原生 IDE 的区别与重叠。

**🔧 技术方法**

采用系统综述与多声文学综述方法：制定检索查询、执行数据库（IEEE Xplore、ACM DL、Scopus、SpringerLink、Google Scholar）和灰色渠道（Google Web、YouTube、LinkedIn、X）搜索；进行标题/摘要筛选、全文评估、双重质量/可信度检查；使用 Garousi 等 15 项灰色文献可信度清单、Dybå & Dingsøyr 11 项同行评审质量清单；通过编码与主题合成构建证据映射与结论。

**📊 数据集**

47 篇来源（28 篇同行评审期刊/会议论文，19 篇博客、行业报告、视频、技术指南等灰色文献），涵盖 2022‑2025 年间关于 vibe coding 的定义、流程、工具、效果、风险与治理等方面的资料。

**📈 对比分析**

该综述并未进行实验性对比，而是通过将不同来源的定性与定量发现进行汇总来评估性能。结果显示：约 45% 的研究报告短期内提高生产力与原型开发速度；约 36% 关注代码/UI 质量，23% 关注可维护性，21% 关注缺陷风险，6% 关注可复现性。证据强度显示：生产力与工作流程为“强”，质量与可维护性为“适度”，可复现性为“弱”。

**⚠️ 局限性**

①研究样本量有限且集中在 2025 年，缺乏长期、实验性、跨领域（尤其是生产级、数据密集型与安全关键系统）的实证数据；②灰色文献可信度分布不均，部分来源可能存在主观偏差；③未对工具性能进行统一量化评测，主要依赖自述或案例；④工作流程与风险描述多为概念化与经验总结，缺乏系统化的度量与验证；⑤在评估机制与治理方面的实践案例尚不充分，导致对可持续性与安全性的结论仍处于“新兴”或“弱”层级。

---

## 7. An Introduction to Bayesian and Frequentist Simulation-Based Inference with Machine Learning

**arXiv ID:** 2607.21702 | [PDF](https://arxiv.org/pdf/2607.21702v1)

**作者:** Maximilian Dax `[一作]` (Max Planck Institute for Intelligent Systems), Gilles Louppe `[通讯]` (University of Liège)

**通讯引用:** 7308 | [OpenAlex ID](https://openalex.org/A5017670779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述了基于机器学习的贝叶斯与频率学模拟推断（SBI）方法，系统梳理了两种统计框架、分布级推断、噪声处理以及验证技术。

**💡 创新点**

创新点在于将贝叶斯与频率学两大推断范式统一于SBI框架，提出利用神经后验、似然与比率估计实现全流程的无模型推断，并详细阐述了使用潜在似然信息、重要性采样以及分布级学习的高效策略。

**🔧 技术方法**

核心技术包括深度生成模型（如正则化流、变分流、扩散模型）用于后验与似然估计、分类网络实现似然比率学习、序列化训练与重要性采样、嵌入网络处理复杂数据模态，以及利用校准与C2ST等方法进行验证。

**📊 数据集**

论文主要以理论与示例演示为主，引用的实验实例包括重力波参数估计、粒子物理中的检测器展开与经验贝叶斯任务，未给出统一公开数据集。

**📈 对比分析**

对比方法主要与传统 MCMC/嵌套采样、ABC 等经典推断方法进行对照，强调神经网络推断在推理速度与可扩散性上的优势；在已知似然场景下可通过重要性采样获得近似无偏的证据估计。

**⚠️ 局限性**

局限性包括缺乏通用验证标准、对模拟器误差与分布外推的敏感性、在高维或多模态情形下训练与采样的挑战，以及在处理大量混淆参数与模型比较时的成熟度不足。

---

## 8. Decoupled Attention Fusion: Accelerating RAG with Efficient KV Cache Reuse

**arXiv ID:** 2607.21599 | [PDF](https://arxiv.org/pdf/2607.21599v1)

**作者:** Xiabao Wu `[一作]` (Ant Group), Jiajun Zheng `[通讯]` (Ant Group)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Decoupled Attention Fusion (DAF) 框架，用以加速检索增强生成（RAG）模型的 KV 缓存重用，显著降低时间‑到‑首标记（TTFT）延迟。

**💡 创新点**

创新点在于将注意力过程拆分为重要词自注意、问答文档自注意和状态融合三阶段，既修复了离线 KV 缓存的交互缺失，又保持了 Flash‑Attention 的密集计算模式，实现高效兼容与准确性兼顾。

**🔧 技术方法**

采用多层聚合的重要词筛选、密集式自注意力计算、状态融合策略，并利用 Flash‑Attention 内核加速执行，同时在 Qwen2.5‑7B 上验证。

**📊 数据集**

实验使用 3*2WikiMultihopQA、3*SAMSum、3*RULER‑qa1/qa2、3*LongBenchV2(Medium) 等多种长文本检索/问答/摘要数据集。

**📈 对比分析**

与全重计算的 vLLM 以及 CacheBlend 对比，DAF 在 TTFT 上提升至 5.6×，在长上下文任务中保持或优于 CacheBlend 的准确性，显著提高推理速度。

**⚠️ 局限性**

局限性包括需为不同任务调优重要词比例 r，且 KV 缓存的存储与管理仍是工业规模下的瓶颈。

---

## 9. Wavelet Phase Diffusion for Structurally and Semantically Consistent Sim-to-Real Translation

**arXiv ID:** 2607.21628 | [PDF](https://arxiv.org/pdf/2607.21628v1)

**作者:** Kaiwen Wang `[一作]` (Karlsruhe Institute of Technology), Omer Sahin Tas `[通讯]` (FZI Research Center for Information Technology)

**通讯引用:** 541 | [OpenAlex ID](https://openalex.org/A5061681603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种无条件、无配对的仿真到现实图像与视频翻译方法 Wavelet Phase Diffusion（psiPD），能够在保持结构与语义一致性的同时显著提升真实性。

**💡 创新点**

核心创新在于：①使用 Dual-Tree Complex Wavelet Packet Transform（DT-ℂWPT）在频域实现局部相位注入，消除 Fourier 基的全局谱耦合导致的环纹与边缘泄漏；②引入 Low‑Frequency Randomization（LFR）随机化低频包，摆脱合成光照先验，获得真实世界的光照分布；③通过空间自适应的截止映射实现实例级翻译，无需额外监督或控制模块。

**🔧 技术方法**

技术上采用 DT-ℂWPT 对潜在空间进行分解，构造带有源相位和噪声幅值的结构化噪声，随后在逆变换中恢复；结合 FLUX.1‑dev 及 Wan 2.2‑14B 等 diffusion backbone；训练使用无配对开放域图像/视频数据。

**📊 数据集**

在 vKITTI → KITTI（合成车道场景到真实 KITTI）和 CARLA（合成驾驶视频到真实 nuScenes）两个基准上进行评估；同时对比 FlowEdit、DNAEdit、NeuralRemaster、Cosmos Transfer 2.5、Ditto 等现有方法。

**📈 对比分析**

实验结果显示 psiPD 在 KID、CLIP‑IQA、mIoU 等指标上均排名第一，且在 CARLA 视频翻译中降低 VLM 规划器的 ADE/FDE 约 5.4%/5.1%，显著优于其他无配对或无条件方法；与 paired‑data 方法 Ditto 在真实性上相近，却保留了语义一致性和时间连贯性。

**⚠️ 局限性**

局限性包括：仍需要在无配对数据上进行一定量训练，且对超参数（如截止半径 r、分解层数 J）的敏感度需进一步研究；在极端光照或纹理变化场景下，LFR 可能无法完全消除残留的合成光照痕迹。

---

## 10. Spectral Flow Certificates for Depth-Aware Long-Range Propagation in Graph Neural Networks

**arXiv ID:** 2607.21607 | [PDF](https://arxiv.org/pdf/2607.21607v1)

**作者:** Ranjan Veerabhadraswamy `[一作]` (Vellore Institute of Technology (VIT-AP)), Ajith Jubilson Emerson `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Spectral Flow Certificate（SFC），一种基于图谱的训练前指标，用来预测GNN在不同深度下的长距离信息传播能力。

**💡 创新点**

SFC将图的第二小特征值（谱间隙）与信息传递层数结合，形成单个可解释的数值，能够自适应地评估不同深度下的拓扑瓶颈。

**🔧 技术方法**

利用归一化拉普拉斯算子求第二小特征值，并通过公式1-(1-λ₂)^k计算SFC；还涉及信息论上对互信息的上界推导。

**📊 数据集**

在25个合成图族（路径、环、网格、正则图、随机图）以及150个真实分子图（来自ENZYMES、PROTEINS、MUTAG）上进行验证，并在10个常用单图上做额外测试。

**📈 对比分析**

与原始谱间隙、有效电阻、图直径等基准相比，SFC在合成图上实现≈0.86-0.95的R²，单一基准时比谱间隙高约2.5个百分点；在真实分子图上R²分别为0.58-0.77。

**⚠️ 局限性**

仅关注无向图的归一化拉普拉斯，忽略多特征值瓶颈，且并不提供解决方案，只是诊断；对噪声标签、异质性、过平滑等其他失败模式无效。

---

## 11. Genesis: An Empirical Platform for Studying Open-Ended Evolution Without Fitness Functions

**arXiv ID:** 2607.21631 | [PDF](https://arxiv.org/pdf/2607.21631v1)

**作者:** Anushka Sharma `[一作]` `[通讯]` (Banasthali Vidyapith), Anushka Sharma (Banasthali Vidyapith)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建Genesis平台，进行三组实验（去适应度、假设控制的生态构造、保护性生态构造），研究无适应度函数下的进化动力学。

**💡 创新点**

提出无标量奖励的三种机制（约束驱动选择、约束自适应调节、人工免疫系统）和PNCT指标体系，证明在完全去适应度后仍可持续进化，并通过假设控制实验剖析开放式进化的瓶颈。

**🔧 技术方法**

使用Pareto支配、CPPN+NEAT、LZW压缩复杂度、Mann‑Whitney U检验等技术，并实现动态约束调节与创新档案管理。

**📊 数据集**

数据来源为平台模拟产生的基因组、行为轨迹和环境状态，实验规模包括10,000代、50,000代及累计1,000,000代的运行记录。

**📈 对比分析**

与随机搜索、固定约束、Novelty Search、MAP‑Elites等基线对比；在无适应度条件下7/12实验持续进化（Wilson 95% CI [30.2%,82.5%]），相较基线显著(p<0.01, Cohen's d=1.47)；但复杂度上限(EPC)仍未突破，表明持续进化但未达开放式增长。

**⚠️ 局限性**

主要限制包括：复杂度上限未突破，缺少可进化物理参数的完整实验，实验规模有限，且未完整验证保护性生态构造对开放式进化的正面影响。

---

## 12. A Systematic Survey on Image Description Techniques for STEM Domains

**arXiv ID:** 2607.21611 | [PDF](https://arxiv.org/pdf/2607.21611v1)

**作者:** Marco Cardia `[一作]` (University of Pisa), Barbara Leporini `[通讯]` (University of Pisa)

**通讯引用:** 2034 | [OpenAlex ID](https://openalex.org/A5050312593)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 20 篇关于 AI 生成 STEM 图像描述的研究进行系统综述，采用 PRISMA 检索流程和 ROBIS 风险评估，提炼 AI 架构、数据集、评估方法与交互模式。

**💡 创新点**

首次将 STEM 可访问性研究系统化，明确四级描述层级与交互式多模态系统的演进；识别数据集、评估指标与用户中心研究的严重缺口，提出未来研究方向。

**🔧 技术方法**

使用 PRISMA 体系化检索、ROBIS 质量评估、结构化数据提取；分析的 AI 技术包括 CNN‑LSTM、Transformer/LLM、Vision‑Language Models、多模态对话框架等。

**📊 数据集**

综述了多种数据集（如 DVQA、FigureQA、PlotQA、SciCap、Math‑Exp‑Syn、Synthetic 等），指出目前缺乏 BLV 共建、可访问性导向的公开数据集。

**📈 对比分析**

对比已发表的自动与人工描述，通过 BLEU、ROUGE、CIDEr 等自动指标与人类 Likert 评分，发现自动模型在语义正确性和可用性上显著低于人工；自动指标与实际可访问性效果不匹配。

**⚠️ 局限性**

主要局限包括：数据集稀缺、评估指标与 BLV 需求脱节、模型易产生幻觉与事实错误、缺乏 WCAG 标准引用、用户中心评估不足，且多采用过时 CNN 架构，影响技术推广和实际可用性。

---

## 13. Risk Is Not the Target: A Monotonic Framework for Evaluating Wildfire Operational Risk Signals

**arXiv ID:** 2607.21597 | [PDF](https://arxiv.org/pdf/2607.21597v1)

**作者:** Nicolas Caron `[一作]` (Université Marie et Louis Pasteur), Benjamin Aynes `[通讯]` (SAD Marketing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并实现了基于单调一致性的野火风险评估框架，并用它比较专家指数、统计模型、GRU预测模型和多智能体FARS系统在法国阿尔卑斯-马里提姆地区的连续风险信号质量。

**💡 创新点**

首创以风险等级与实际作业负荷（火灾数量、干预时长、资源部署）的持续正相关为评价标准，强调风险量表的结构而非传统事件预测准确率；同时通过该框架揭示传统AI模型对风险量表结构的不足。

**🔧 技术方法**

使用B样条回归+固定效应估计单调性、GRU时序预测模型、LLM驱动的多智能体系统FARS，以及统计与基线模型（Poisson、Logistic、Persistence、MaxWeek）进行对比。

**📊 数据集**

使用2017–2023年法国阿尔卑斯-马里提姆六个气象区每日的火灾计数、干预时长、资源部署以及气象、地形、社会经济等特征数据集。

**📈 对比分析**

通过单调得分SCORE_k1–k4评估各模型在低阶和高阶风险跃迁的单调一致性，结果显示DFE、GRU+DFE聚合和去除火/时输入的FARS配置在全风险范围内表现最佳；传统评估指标（MAE、IoU、F1）无法体现风险信号的结构性。

**⚠️ 局限性**

D FE受自身对资源调度的影响导致部分自洽；FARS仅传递上游预测的结构缺陷，未能纠正；模型仅在阿尔卑斯-马里提姆地区验证，泛化性未评估；LLM引入随机性；对低风险区间的覆盖不足。

---

## 14. Analyzing Middle School Students' Dialogue and Behaviors during Collaborative AI Chatbot Development Using Ordered Network Analysis

**arXiv ID:** 2607.21603 | [PDF](https://arxiv.org/pdf/2607.21603v1)

**作者:** Shan Zhang `[一作]` (University of Florida), Shiyan Jiang `[通讯]` (University of Pennsylvania)

**通讯引用:** 1372 | [OpenAlex ID](https://openalex.org/A5020086309)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了中学生在协作式 AI 聊天机器人设计与开发过程中的互动行为，利用有序网络分析揭示对话与操作行为的时间序列关系。

**💡 创新点**

创新点在于将有序网络分析应用于多维互动数据，系统性捕捉解释、测试与改进等交互循环，并将其与机器人质量和 AI 知识获取联系起来。

**🔧 技术方法**

采用有序网络分析（Ordered Network Analysis）及 WebENA 工具进行构建，结合 Spearman 相关、Mann‑Whitney U 检验与 Benjamini‑Hochberg 校正等统计方法。

**📊 数据集**

使用来自 47 对（共 94 名）中学生的语音转录（共 32,976 条话语）、聊天机器人开发日志以及一份 15 题 AI 知识后测问卷的数据集。

**📈 对比分析**

将机器人质量按中位数分为高低两组，比较两组的有序网络结构差异；结果显示高质量组在解释、指令、重复及测试等环节的连接权重显著更高，且与 AI 知识得分呈正相关。

**⚠️ 局限性**

局限性包括仅针对中学生单一 AI 工具的实验，未考虑姿态、情感等非语言线索，且研究仅为相关性分析，无法证明因果关系。

---

## 15. Evolving Self-Organising Agents Without Fitness: Three Falsifiable Experiments from Constraint-Driven Selection to Developmental Encoding

**arXiv ID:** 2607.21630 | [PDF](https://arxiv.org/pdf/2607.21630v1)

**作者:** Anushka Sharma `[一作]` `[通讯]` (Banasthali Vidyapith), Anushka Sharma (Banasthali Vidyapith)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在Gray-Scott反应扩散环境中，使用无设计的适应度函数进行进化实验，探究仅靠物理约束是否能产生开放式进化。

**💡 创新点**

首次证明了约束驱动选择在无适应度评分时可维持进化并产生结构上限；通过伪控制实验消除被动生态位构建的效应；并首次展示CPPN间接编码与NEAT式物种保护能推动结构复杂化。

**🔧 技术方法**

采用Gray-Scott反应扩散子strate、约束自适应调节（CARP）、人工免疫系统（AIS）、PNCT评估指标、伪控制实验、CPPN与NEAT间接编码、代数多物种管理等技术。

**📊 数据集**

实验数据来自数百个独立运行（如12个种子×10,000代、10个种子×50,000代等），未使用外部公开数据集。

**📈 对比分析**

与随机搜索、固定约束、Novelty Search和MAP-Elites等基线对比，结果显示约束驱动选择在无适应度时仍能维持58%以上的进化活跃度；在未加入物种保护的情况下结构复杂化停滞，而加入CPPN+NEAT后可显著提升结构规模与功能多样性。

**⚠️ 局限性**

局限性包括：实验仅在单一Gray-Scott环境验证，缺乏在更大规模或多维度环境中的推广；复杂性上限受约束强度与代谢成本参数限制；缺乏理论解析解释复杂性上限的机制。

---

## 16. FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills

**arXiv ID:** 2607.21596 | [PDF](https://arxiv.org/pdf/2607.21596v1)

**作者:** Zeyu Ren `[一作]` (Southeast University), Shimin Di `[通讯]` (Southeast University)

**通讯引用:** 288 | [OpenAlex ID](https://openalex.org/A5006260400)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的框架 FlowEvo，通过在推理时将成功工作流编译成可执行技能记录并存入技能库，以提升后续任务解决效率。

**💡 创新点**

创新点在于三层循环：(1) 工作流到技能的在线编译；(2) 技能反馈回工作流生成，可直接重用或作为结构化上下文；(3) 对技能进行持续的安全与负迁移审计与裁剪，实现动态自演化。

**🔧 技术方法**

核心技术包括可执行技能抽象（入口、接口、重放测试）、检索路由策略（动态生成、直接执行、结构化条件生成）、执行器与验证器、以及对技能生命周期的对比效用评估与抑制。

**📊 数据集**

使用 ALFWorld（交互式环境）、HumanEval 与 GSM8K（代码/数学生成）三个公开基准进行实验。

**📈 对比分析**

与 Reflexion、ExpeL、AFLOW、ADAS 等基线相比，FlowEvo 在所有三套基准上均取得最高准确率且平均 token 消耗最低；在 ALFWorld 上成功率提升 23.6pp，平均 token 下降 50%；在 HumanEval 和 GSM8K 上也均实现了更高的 pass@1 / solve 率并保持低 token 成本。

**⚠️ 局限性**

局限性包括：需要基模型已具备足够成功率以产生可编译工作流；对验证、反馈严格的环境适用，延迟或噪声反馈的场景难以保证安全与有效性；技能库规模随使用增长需进一步维护策略，且对不同领域的迁移能力尚未充分验证。

---

## 17. Quasi-Monte Carlo Initialization for Meta-Reinforcement Learning

**arXiv ID:** 2607.21637 | [PDF](https://arxiv.org/pdf/2607.21637v1)

**作者:** Julian G. Soltes `[一作]` `[通讯]` (Regis University), Julian G. Soltes (Regis University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究利用准蒙特卡罗采样生成多组初始权重，采用一次性适配搜索快速定位最优元先验，然后在不同连续控制任务上进行零样本迁移实验。

**💡 创新点**

首次证明准蒙特卡罗采样（Sobol、Latin Hypercube、Hyperellipsoid Density Sampling）可作为高效的零样本权重初始化几何，在类似任务中显著提升训练收敛，而在不相似任务中则需要考虑过拟合风险。

**🔧 技术方法**

使用了 Sobol、Latin Hypercube、Hyperellipsoid Density Sampling 三种准蒙特卡罗序列生成权重样本，结合 SB3 PPO 进行单步适应和 10⁶ 步训练，并通过 rank‑sum 检验和 Z‑标准化对比结果。

**📊 数据集**

在 Gymnasium 的连续控制环境上测试，包括 HalfCheetah‑v5、BipedalWalker‑v3、Hopper‑v5、Ant‑v5、Walker2d‑v5、BipedalWalkerHardcore‑v3、Swimmer‑v5 及 LunarLander‑v3。

**📈 对比分析**

与 SB3 默认的 Orthogonal 与 Random 初始化方式对比，在相似任务中 Sobol、HDS 等 QMC 先验平均提升约 0.15–0.20 的 Z‑得分并显著优于基线；在不相似任务中 QMC 先验相对较差，SB3 Orthogonal 更优。

**⚠️ 局限性**

该方法仅在与搜索环境动力学相似的任务中表现优异，非欧几里得 HDS 在不同任务中易过拟合；实验仅覆盖 8 个环境，缺乏更广泛的场景验证与大规模样本验证。

---

## 18. Discrete Action Space as a Prerequisite for GRPO Convergence in Small-Model Continuous Control

**arXiv ID:** 2607.21626 | [PDF](https://arxiv.org/pdf/2607.21626v1)

**作者:** Dmytro Filatov `[一作]` (Aimech Technologies Corp.), Vira Filatova `[通讯]` (Covijn Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在小型语言模型（≤1B参数）下，研究了Group Relative Policy Optimization（GRPO）在连续控制任务中的收敛性，并提出将动作空间离散化为5分类PID预设能让GRPO收敛；

**💡 创新点**

创新点在于证明连续动作空间导致GRPO崩溃、离散化动作空间是关键结构因素，并给出了学习曲线、跨模型验证以及高保真仿真验证；

**🔧 技术方法**

使用的技术包括GRPO、LoRA微调、5‑way离散PID预设接口、奖励分解与课程学习、以及Crazyflie 2.1高保真仿真模型；

**📊 数据集**

数据集为作者自研的21场景四旋翼基准（涵盖风、负载、噪声、通信延迟等六类扰动），并在6场景子集上进行动作位宽扫测试；

**📈 对比分析**

通过在21个场景、10个随机种子下与重调PID、MPC等基线进行比较，GRPO离散化实现达100%成功率，平均轨迹抖动在0.656–1.103 m/s³；连续实现失败（0%成功率），同一方法在三种不同语言模型上亦保持100%成功率；

**⚠️ 局限性**

局限性包括仅在单一1.5 kg四旋翼、单一基模型和单一仿真基准上验证；未在真实硬件上测试；奖励函数手工设计；带宽曲线仅在样本数N=3下评估；对hover区分布缺陷未进行域随机化训练。

---

## 19. Toward User-Conditioned Evaluation of Personal LLM Agents under Temporal Interventions

**arXiv ID:** 2607.21635 | [PDF](https://arxiv.org/pdf/2607.21635v1)

**作者:** Pin Qian `[一作]` (Carnegie Mellon University), Junxian You `[通讯]` (University of Glasgow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并设计了个人智能代理评估的最小基准框架，提出用户条件下时间干预的四个评估要求（C1–C4），并对现有公开基准进行缺口分析，给出了对应的度量指标。

**💡 创新点**

创新点在于将评估焦点从单一能力转移到用户持久状态对同一接口或策略变化的传播效果，首次系统性定义了C1–C4四个评估维度，并为跨维度、跨用户的失败传播提供了指标与方法，填补了现有基准在这方面的空白。

**🔧 技术方法**

技术手段包括：基准设计与任务家族构造、依赖图注释、指标定义（α_L、γ、σ、κ、ρ等）、对公开基准的审计与编码（15个协议），以及示例性用户配置与事件脚本的生成。

**📊 数据集**

使用了公开的评估协议与基准列表（如StableToolBench、MemoryAgentBench、SkillLearnBench等）作为审计对象；基准实例采用Analytics/Reporting Assistant任务，模拟不同用户配置和持久状态；并未构建新的大规模数据集。

**📈 对比分析**

比较方法：通过对已有15个公开基准的编码评估，发现没有协议同时满足C1–C4；作者提出的度量可在后续基准中用于评估同一事件在不同用户状态下的恢复、降级、安全、跨维度一致性和回归率。由于本文为位置性论文，未给出具体实验性能数值，但为后续评测提供了量化框架。

**⚠️ 局限性**

局限性：缺乏完整的跨维度、跨用户评估协议；依赖图与用户配置需要手工生成，难以大规模复现；示例基准局限于Analytics/Reporting Assistant领域；难以覆盖所有潜在的记忆–工具–技能–政策互依关系，仍需进一步完善和实证验证。

---

## 20. A Consensus-Based Framework for Relative Preference Evaluation of Large Language Models

**arXiv ID:** 2607.21632 | [PDF](https://arxiv.org/pdf/2607.21632v1)

**作者:** Mohtashim Khan `[一作]` `[通讯]`, Mohtashim Khan

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了基于多模型盲评的共识评估框架，计算相对智能指数(RII)来衡量模型间的偏好关系。

**💡 创新点**

创新点在于不依赖固定真值，而是利用多模型匿名排名聚合获得相对偏好信号，并以此作为可扩展的评估指标。

**🔧 技术方法**

采用了匿名化随机排序、独立排名、结构化JSON输出、统计聚合等技术，并使用5个主流LLM进行生成与评判。

**📊 数据集**

使用自定义的25条高难度提示，覆盖编程、数学、安全、逻辑推理和通用知识五个领域。

**📈 对比分析**

通过多轮重复实验计算每个模型的平均RII及95%置信区间，结果显示部分模型在多数领域获得更高偏好，且指标稳定；与延迟对比展示了性能与效率的权衡。

**⚠️ 局限性**

局限性包括未衡量绝对正确性或人类偏好、可能存在训练共享偏差、对提示与风格敏感、实验次数有限、缺乏统计显著性检验。

---

## 21. Self-Poisoning in Adaptive Out-of-Distribution Detection: A Sharp-Threshold Theory and Certified Label-Free Calibration

**arXiv ID:** 2607.21673 | [PDF](https://arxiv.org/pdf/2607.21673v1)

**作者:** Vishnu Bindu Balachandran `[一作]` `[通讯]` (Independent Researcher), Vishnu Bindu Balachandran (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在无标签的自适应 OOD 检测中出现的自毒化现象，并提供了严谨的动态学理论、可认证的门控策略与对漂移/污染兼容的 CDC 校准方案；

**💡 创新点**

创新点包括：① 将自毒化建模为泛化 Pólya 餅 urn 并给出临界阈值与分岔理论；② 设计 WARDEN 门控的证书化入库机制，完全消除自毒化；③ 开发 CDC 校准在漂移与污染共存时仍能保证 FPR 的 label‑free 方法；④ 给出不可能性定理阐明无标签场景下的精度与 FPR 之间的根本权衡；

**🔧 技术方法**

核心技术包括：泛化 Pólya urn 与 ODE 方法的均场分析；符合式 p‑value、e‑value 与 e‑BH 控制的 conformal 统计；Storey 估计与 DKW 置信区间；对冲突窗口的有效样本大小调整；

**📊 数据集**

实验基于 96 种设置，覆盖四类 encoder（vision：ResNet‑18、ViT、DINOv2、CLIP；text：RoBERTa；document：LayoutLMv3），对多种 OOD 目标（MNIST、SVHN、DTD、TIN、Places、CIFAR、20News、AGNews、MNLI、Tobacco intra/cross、Finance）进行 9,264 次流式实验，包含 1,695 次漂移受影响的样本；

**📈 对比分析**

与传统无门控自适应方法（AdaODD、OODD）相比，未门控方法在 bursty、低污染场景下会导致 AUROC 下降 0.13–0.28、字典 impurity 超过 0.9、FPR 失控；WARDEN 在所有 4,800 次标准实验中保持 impurity ≤0.111，平均 FPR 0.056，AUROC 与冻结基线一致；CDC 在 1,695 次漂移受影响样本中以 0.042 的平均 FPR 通过 1.1α 认证，TPR 保留率约 0.67（接近理论上限 κ≈0.69）；

**⚠️ 局限性**

局限性包括：假设一维状态，若需更丰富的毒化结构需进一步建模；对非独立窗口的校准给出的上界较松，需更紧的块自举方法；CDC 的功率损失在无标签场景中不可避免且受有限样本误差影响；未实现对端到端训练的评估；仅验证了特征空间的尾部仿真攻击，未覆盖梯度级输入攻击；在新领域需重新测定 admission kernel 参数。

---

## 22. SCOPE and SCION: A Benchmark and an Auditable Reference Pipeline for Schema Induction and Fusion from Text

**arXiv ID:** 2607.21610 | [PDF](https://arxiv.org/pdf/2607.21610v1)

**作者:** Miaobo Hu `[一作]` (Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10801 | [OpenAlex ID](https://openalex.org/A5042106312)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个名为SCOPE的训练文本仅用语料库，专门用于评估从原始文本自动诱导并融合知识图谱模式（schema）的算法；提出了可审计的SCION管线，实现候选挖掘、LLM驱动的命名与合并、以及对已有本体的保守融合。

**💡 创新点**

创新点在于：①首次将语料到schema的端到端诱导过程拆解为可追溯的可审计步骤；②统一采用Typed-Edge表示并提供多种语义相似度评估指标；③将LLM的输出严格约束在JSON合同内，结合可回溯证据，兼顾可解释性与可复现性。

**🔧 技术方法**

技术包括候选空间挖掘（统计式或LLM提取）、嵌入式聚类、结构化LLM提示与约束生成、JSON合法性验证、保守式本体对齐、强化学习自监督训练（SCION‑RL）以及多维度图相似度评估。

**📊 数据集**

使用了24个公开信息抽取数据集（15个关系抽取、9个事件抽取），每个数据集都有官方模式并被标准化为gold schema graph，所有实验仅基于训练拆分文本。

**📈 对比分析**

与已发布源模式、Text2Onto风格的传统本体学习、直接LLM诱导以及匹配的extract‑then‑aggregate基线相比，SCION‑lite在四种评估指标（Literal、Fuzzy、Continuous、Graph F1）上均取得最高分（例如Fuzzy 0.9298，Continuous 0.8909），显著优于所有基线；SCION‑RL进一步提升至0.9101 Continuous。

**⚠️ 局限性**

局限性包括：①对高频公开数据集可能存在预训练泄漏风险；②当前仅覆盖事件类型与内部角色，缺乏跨事件时间/因果等关系；③结果高度依赖评估形式与指标，某些指标如Fuzzy过于宽松，需结合人工校准。

---

## 23. Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization

**arXiv ID:** 2607.21619 | [PDF](https://arxiv.org/pdf/2607.21619v1)

**作者:** Bingjun Luo `[一作]` (Tsinghua University), Xinpeng Ding `[通讯]` (Xidian University)

**通讯引用:** 702 | [OpenAlex ID](https://openalex.org/A5020996610)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并提出一种基于视觉风格的攻击增强框架（ASO），通过强化学习对图像编辑模型进行风格参数优化，从而显著提升多模态大语言模型（MLLM）的破解成功率。

**💡 创新点**

创新点在于首次系统识别并利用MLLM对视觉风格的内在敏感性，将其转化为攻击向量；提出了可插拔的ASO模块，结合结构化分层奖励函数和Group Relative Policy Optimization（GRPO）实现高效的风格优化；并通过ODE‑to‑SDE转换克服了流模型在RL中的随机性不足。

**🔧 技术方法**

核心技术包括：图像风格迁移模型（如FLUX‑Kontext），强化学习框架（GRPO + 动态批次DB‑GRPO），结构化分层奖励函数（结合模型拒绝概率和Judge模型的语义评估），以及对生成模型采样的ODE‑to‑SDE转换。

**📊 数据集**

使用的主要数据集有：基础攻击数据集（MM‑SafetyBench、VLBreakBench）以及多种SOTA攻击（FigStep、QR‑Attack、SI‑Attack、HIMRD 等）；风格池包含多种风格类别；评估时采用HarmBench作为Judge模型进行安全性打分。

**📈 对比分析**

实验对比方法：在多种商业（GPT‑4.1、Gemini‑2.5）和开源（Qwen3‑VL、LLaVA‑OneVision‑1.5）VLM上，将原始攻击与“+ Probing”（仅加最优风格）和“++ Enhance”（完整ASO）进行对比。结果显示，ASO 能将攻击成功率（ASR）提升数个百分点至十数个百分点，且Harmfulness Score 同步提升，细粒度类别评估亦显示均匀且显著的提升。

**⚠️ 局限性**

局限性：仅针对视觉风格这一非内容攻击面；需要先对目标模型进行风格探测，且训练成本高；对模型的版本更新或更强的安全防护机制可能导致效果下降；目前仅验证于图像+文本的 VLM，跨模态或多语言环境的适用性尚待进一步研究。

---

## 24. MotifRole-Diff: Risk-Optimal Role-Aware Corruption for Masked Molecular Graph Diffusion

**arXiv ID:** 2607.21634 | [PDF](https://arxiv.org/pdf/2607.21634v1)

**作者:** Tasfia Nuzhat Ornee `[一作]` (University of Central Florida), Niloofar Yousef `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于角色感知的遮蔽扩散模型 MotifRole-Diff，用以在分子图的序列化表示中根据不同 token 角色（特殊、语法、内部、接口）的重构难度和对图结构的影响来分配遮蔽率，从而提升生成质量。

**💡 创新点**

创新点在于：①将分子图序列化中的 token 角色区分并量化其重构难度与结构影响；②基于风险最优分配理论（水填充法）推导出角色感知的遮蔽调度；③引入 mSENT 结构感知序列化，增强子结构局部性且保持解码器无损。

**🔧 技术方法**

核心技术包括：分子图无损序列化（SENT/mSENT）、离散吸收式扩散模型（MDLM）、角色感知的遮蔽调度（γ_r 预设），逆曝光加权损失、角色感知的逆向采样与角色先验推断。

**📊 数据集**

实验使用 QM9（小分子集）、MOSES（多样化药物样本）以及 GuacaMol（大规模化学空间）三个公开数据集。

**📈 对比分析**

与统一遮蔽的 MDLM 进行对比，保持架构、优化器、训练预算和采样算力相同；结果显示 MotifRole‑Diff 在 QM9 上有效率从 0.905 提升至 0.944，FCD 由 1.701 降至 1.609；在 MOSES 上有效率从 0.920 提升至 0.938，FCD 从 2.125 降至 1.850；在 GuacaMol 上有效率从 0.787 提升至 0.841。总体提升均表现在有效率、分布相似度与多样性指标上。

**⚠️ 局限性**

局限性包括：单 token 影响估计可能低估多 token 接口失效；角色先验 P(r|v) 仅基于词表统计，未考虑上下文；跨数据集迁移时仅使用 QM9 估计的调度，未在 MOSES 重新估计；以及理论风险最优仅针对建模目标，未保证所有生成指标均提升。

---

## 25. Household Movement Detection in Mixed-Format Occupancy Data Using LLM-Based Entity Resolution

**arXiv ID:** 2607.21614 | [PDF](https://arxiv.org/pdf/2607.21614v1)

**作者:** Sasirekha Oguri `[一作]` (University of Arkansas), Mert Can Cakmak `[通讯]` (University of Arkansas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用LLM进行姓名与地址的直接抽取，结合语义嵌入进行相似性块化，并在图结构中推理共享成员的群体迁移，从而检测混合格式占用数据中的间接家庭迁移链接。

**💡 创新点**

① 将大语言模型直接用于混合格式记录的命名实体识别，避免繁琐的预处理；② 用语义嵌入构造可容错的相似性图；③ 在图上进行群体级推理，捕获传统配对方法无法发现的间接迁移证据；④ 将“领导–跟随”合并机制与多重阈值控制结合，保持高精度的同时显著提升召回。

**🔧 技术方法**

GEMMA‑2‑2B‑IT 进行 NER，BAAI/bge‑m3 用于姓名和地址的句子嵌入，余弦相似度阈值构造相似性图，图划分为“分析段”后进行基于地址的群体聚类、重复检测与领导–跟随合并，最终基于共享成员数推断迁移并生成间接链接。

**📊 数据集**

Synthetic Occupancy Generator (SOG) 生成的 SPX 基准数据集 S8–S12，包含 4,000–6,000 条记录，涵盖不同程度的噪声、重复、异构格式及链式迁移。

**📈 对比分析**

与传统的基于配对相似度的 Data Washing Machine（DWM）对比。实验表明，在所有 SPX 数据集上，加入间接迁移推理后 F1 分数提升约 6%–8%（召回提升 8%–15%，精度基本保持高位），展示了显著的性能优势。

**⚠️ 局限性**

① 依赖于可观测的共享成员，单成员重叠或姓名剧变时无法识别迁移；② 失去时间戳信息，无法判断迁移方向与时序；③ 计算量主要集中在全量余弦相似度（O(N²)），对大规模数据需采用近似或阻止技术；④ 对高频姓名与模糊地址可能产生假正例，需进一步阈值微调。

---

## 26. Physically Constrained Federated Additive Models for O-RAN SLA-Risk Prediction

**arXiv ID:** 2607.21665 | [PDF](https://arxiv.org/pdf/2607.21665v1)

**作者:** Aubida A. Al-Hameed `[一作]` (Ninevah University), Syed A. Zaidi `[通讯]` (University of Leeds)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究开发了一种可在 O‑RAN 环境中通过联邦学习训练的神经可加模型（NAM），用于可审计的切片 SLA‑违约风险预测，并能够部署到 Near‑RT RIC 的 xApp 进行在线推断。

**💡 创新点**

创新点在于将物理单调性约束嵌入联邦聚合过程，保证各基站模型在聚合后仍满足无线物理规律；引入锚定机制解决可加模型在联邦学习中的可识别性问题；并只对物理方向明确的 KPI 加约束，兼顾解释性与灵活性。

**🔧 技术方法**

使用的技术包括联邦学习（FedAvg）、神经可加模型（NAM）、单调线性样条、锚定约束、O‑RAN rApp/xApp 部署路径以及正样本加权的二元交叉熵损失。

**📊 数据集**

实验基于 ColO‑RAN 试验室数据集，该数据集包含七台基站、三类服务切片（eMBB、MTC、URLLC）、多种调度策略（RR、WF、PF）以及 4 Hz 的 KPI 记录。

**📈 对比分析**

通过与本地 NAM、中心化 NAM、黑盒 FedAvg‑MLP、无约束 FedNAM 的对比，发现本方法在 0.901 的 AUC（相较无约束 FedNAM 降低约 0.042）、0% 单调性违规、参数量 2,563 以及 30 轮下 2.15 MB 的上行通信量，且在未见调度器时仍保持良好泛化。

**⚠️ 局限性**

局限性包括仅在单一运营商的七基站模拟环境下验证，用户为静态；未探索跨运营商信任域的联邦治理；实验规模有限，未覆盖动态移动、不同资源分配以及更大规模联邦。

---

## 27. A Drift Stable Quantum Federated Learning for Intelligent Services

**arXiv ID:** 2607.21647 | [PDF](https://arxiv.org/pdf/2607.21647v1)

**作者:** Shanika Iroshi Nanayakkara `[一作]` (Deakin University), Shiva Raj Pokhrel `[通讯]` (Deakin University)

**通讯引用:** 2938 | [OpenAlex ID](https://openalex.org/A5038446422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了DUQFL‑Prox，一种基于深度展开的近端量子联邦学习框架，用以解决异构客户端的漂移、泛化和公平性问题。

**💡 创新点**

①将SPSA优化展开为多步可控的深度展开过程，并通过共享控制器自适应生成学习率和扰动尺度；②引入近端正则化与最佳展开检查点选择以抑制客户端漂移；③采用外部SPSA元学习更新控制器，使本地优化与全局验证性能相匹配。

**🔧 技术方法**

深度展开（Deep Unfolding）、SPSA、近端正则化（Prox）、元学习（Outer SPSA）、变分量子神经网络（QNN）、FedAvg聚合、量子模拟器与IBM硬件验证。

**📊 数据集**

金融欺诈检测数据集（Bank Account Fraud，BAF）和基因组分类数据集（DemoHumanOrWorm），分别采用四维/二维特征。

**📈 对比分析**

与默认QFL、FedProx‑QFL、Adam‑QFL等基线比较，DUQFL‑Prox在全局准确率、平均客户端测试准确率、训练‑测试间隙及客户端公平性间隙等指标上均优于或相当；在高度不平衡的BAF数据集上取得最高准确率并显著降低公平性差距；在基因组数据上虽略逊于FedProx‑QFL的全局准确率，但提供更好的客户端泛化与更低的公平性差距。

**⚠️ 局限性**

缺乏对非凸量子联邦学习系统的收敛理论保证；外部元学习更新成本较高，仅在有限通信轮次下执行；实验仅在模拟器上完成，真实硬件验证仅限于后训练评估；对超参数（如prox系数、展开步数）敏感；实验规模有限（5–10个客户端），未充分检验大规模异构场景。

---

## 28. TILT: Improving Compositional Generation in Diffusion Models with a Model-Intrinsic Reward

**arXiv ID:** 2607.21606 | [PDF](https://arxiv.org/pdf/2607.21606v1)

**作者:** Debottam Dutta `[一作]` (University of Illinois Urbana-Champaign), Romit Roy Choudhury `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 11717 | [OpenAlex ID](https://openalex.org/A5111442177)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种测试时奖励对齐框架，通过模型内在奖励实现多概念图像生成；

**💡 创新点**

将纯模式采样视为奖励对齐问题，推导闭式目标并给出两种导向算法（共享Jacobian与每概念Jacobian）及其混合策略；

**🔧 技术方法**

利用扩散模型的Tweedie公式、Jacobian-approximation、奖励梯度引导、Classifier-Free Guidance等技术；

**📊 数据集**

在Stable Diffusion XL（SDXL）基础模型上，使用T2I-CompBench（Color、Shape、Texture、Complex）数据集；

**📈 对比分析**

与CFG、Composable Diffusion、R2F、CFG++、CO3等无训练、无梯度的基线对比，使用ImageReward、CLIP、DINO、BLIP-VQA四项指标，取得最优或竞争性表现，尤其在Shape与Complex类别上表现突出；

**⚠️ 局限性**

计算成本较高，尤其是每概念Jacobian版本需多次反向传播；奖励梯度不稳定导致某些指标（如BLIP-VQA）略低。

---

## 29. A Defense of the Quadratic Model

**arXiv ID:** 2607.21716 | [PDF](https://arxiv.org/pdf/2607.21716v1)

**作者:** Alexandru Meterez `[一作]` (Harvard University), Alex Damian `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

检验二次模型在LLM预训练中的可预测性，训练150M transformer 3B FineWeb token，使用线性化（NTK）和二次Taylor展开的局部代理，验证其在10%训练窗口内能准确预测损失下降；利用深度Lanczos四边形测量Hessian和Gauss‑Newton谱，分析谱结构、负值及层分布；进行边界稳定性实验，区分随机与确定性边界。

**💡 创新点**

①证明局部二次模型在LLM预训练中可在10%窗口内准确预测，提供实证；②使用极深Lanczos四边形探测到谱的头‑尾结构、普适幂律尾；③系统评估边界稳定性在不同批量大小下的行为，揭示随机边界现象；④整合预处理器（Adam）对谱和稳定性的影响。

**🔧 技术方法**

局部Taylor展开（线性化与二次展开）、Gauss‑Newton矩阵、Adam预处理、深度Lanczos四边形（深度1200）、梯度对齐与源条件估计、预训练EMA、预处理器固定的预条件SGD、均方与李雅普诺夫稳定性理论。

**📊 数据集**

150M参数transformer（168M含embedding），FineWeb 3B token数据集，词表大小8192，序列长度1024，批量大小1、64、1024。

**📈 对比分析**

将代理损失训练10%窗口后在真实模型上评估验证损失；线性/二次代理在中晚期能与真实训练误差相差≤10%；谱估计与理论预测匹配，负值与层分布与实际相符；边界稳定性实验通过网格学习率/批量大小比较发散率，发现大多数配置位于边界附近且仅相差两倍。

**⚠️ 局限性**

仅在单一规模（150M）和架构、Adam优化器下实验，未验证更大规模或其他优化器；早期训练阶段二次/线性近似误差大，未捕捉高阶项；常数学习率+EMA对边界的描述不佳；曲率正则化效应未被二次模型捕获；对稀有token/死亡神经的处理需额外过滤。

---

## 30. Encoding Invisible Causation for Bridge Diagnostic Agents: Triple-Guided Retrieval-Augmented Fine-Tuning with QLoRA

**arXiv ID:** 2607.21680 | [PDF](https://arxiv.org/pdf/2607.21680v1)

**作者:** Takato Yasuno `[一作]` `[通讯]`, Takato Yasuno

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建检索增强的桥梁损伤原因编码器，利用专家手册中的因果三元组对桥梁损伤描述进行根因预测。

**💡 创新点**

创新点在于：①把隐性的因果知识转化为结构化三元组并构建FAISS检索库；②将检索上下文注入BERT编码器并采用QLoRA量化微调；③提出Golden Testset基准并给出基于类别不平衡比（IR）的量化策略指导。

**🔧 技术方法**

技术手段包括：Qwen2.5-7B提取因果三元组、FAISS向量检索、BERT-Large-Ja编码器、LoRA/QLoRA/QA-LoRA微调、4‑bit NF4量化、BFloat16 LoRA、FP16分类头以及大语言模型作为相关性判定器。

**📊 数据集**

使用15–35份桥梁诊断PDF（共6,745条因果三元组），构建428–642条平衡训练样本，Golden Testset 116条已分层去重并标注难度，另外100条多样性评测样本。

**📈 对比分析**

通过Golden Testset对LoRA、QLoRA、QA-LoRA三种微调方法进行对比；QLoRA在保持87.07%准确率的同时，GPU显存从1.45GB降至0.40GB，推理时间从44.0ms降至39.2ms，并在100样本多样性评测中比LoRA提升13个百分点。

**⚠️ 局限性**

局限性包括：三元组抽取时偶有语义错误或缺失；Water Accumulation、Soil Liquefaction、ASR类目在多样性评测中表现极差；模型仅支持日语；缺乏多模态输入和更充分的数据增强。

---

## 31. From Frame-Level Recognition to Event-Level Confirmation: Repair Traces and Runtime Failure Analysis of Public-Space Gesture Interaction

**arXiv ID:** 2607.21601 | [PDF](https://arxiv.org/pdf/2607.21601v1)

**作者:** M. Meng `[一作]` (Shenzhen Nines Light Technology Co Ltd), Yansong Zhang `[通讯]` (Shenzhen Nines Light Technology Co Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过分析在风景区公共交互亭中收集的 8 条工程实验与修复记录，提炼出 20 个失败实例，归纳为 6 类公共空间手势交互失败，并基于这些失败设计了一个事件级运行时抽象层，用来将不稳定的帧级识别输出转换为可确认或拒绝的交互事件。

**💡 创新点**

创新点在于：① 用修复追踪方法揭示“识别到交互”之间的实际差距；② 形成了面向公共空间手势交互的六类失败工作分类；③ 设计了事件级运行时抽象层，系统化了短期手失容忍、时间窗口、尺度归一化、EMA/投票、退避、资源生命周期管理等机制，以降低低层帧级误差对用户可见交互的影响。

**🔧 技术方法**

技术手段包括：利用 MediaPipe Hands 等低层手势识别模型获取帧级关键点；实现事件确认子层（短期手失容忍、时间窗口、尺度归一化、EMA、投票、退避）；运行时维护子层（坐标对齐、资源生命周期清理）；反馈同步子层（UI 文字与语音提示同步）。

**📊 数据集**

没有使用公开数据集；所有数据均来自现场交互日志、现场反馈、截图、代码变更记录、控制台日志等修复记录，构成了一套自建的工程修复数据集。

**📈 对比分析**

本文未进行统计实验或对比基准，也未给出数值指标；仅通过案例分析展示了在修复过程中不同机制的作用和改进方向，无法对性能进行量化比较。

**⚠️ 局限性**

局限性包括：① 仅为案例研究，样本量有限，缺乏大规模用户实验；② 失败分类为工作类而非完整分类；③ 事件级抽象层未在正式系统中量化评估，无法验证实际性能提升；④ 资源生命周期、语音提示同步等细节仍需进一步验证；⑤ 仅覆盖了风景区公共亭环境，缺乏跨场景泛化。

---

## 32. Quantum Adaptive Sensing for Accelerated MRI

**arXiv ID:** 2607.21737 | [PDF](https://arxiv.org/pdf/2607.21737v1)

**作者:** Asmit Ganguly `[一作]`, Danny J. J. Wang `[通讯]` (University of Southern California)

**通讯引用:** 13532 | [OpenAlex ID](https://openalex.org/A5066687350)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于QUBO的自适应k‑space采样框架，通过迭代优化在每个采样批次中选择最优相位编码行

**💡 创新点**

将采样问题映射为定卡尔二次无约束二进制优化问题，利用量子退火与经典并行温度平衡算法实现实时自适应采样

**🔧 技术方法**

使用量子退火（D‑Wave hybrid solver）和并行温度平衡（parallel tempering）解决QUBO，采用SENSE‑TV（ADMM）重建

**📊 数据集**

在8‑线、128³与256³ Shepp‑Logan脑部模拟体积上进行实验，生成8‑线感应灵敏度图

**📈 对比分析**

与VD Poisson‑disk、伪径向、伪螺旋、随机和卡氏变密度采样比较，QAS在20%与10%采样率下在PSNR、SSIM、NMSE、HFEN等指标均优于基线，噪声鲁棒性也更强

**⚠️ 局限性**

实验未展示真正的量子加速或优势，受限于D‑Wave硬件的QUBO规模、池化分解、仅使用模拟数据，未验证对真实人体扫描的适用性

---

## 33. The Hard Decision Layer: Evidence for Committed Inference in Transformers

**arXiv ID:** 2607.21613 | [PDF](https://arxiv.org/pdf/2607.21613v1)

**作者:** Ashwath Vaithinathan Aravindan `[一作]` (University of Southern California), Mayank Kejriwal `[通讯]` (Information Sciences Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究Transformer模型在多项选择问答中的层级决策行为，发现存在一个天然的硬决策层（HDL）。

**💡 创新点**

创新点在于证明HDL是一种不受微调、选项数或标签格式影响的静态层，且可用于有效裁剪模型层数。

**🔧 技术方法**

主要技术包括logit lens抽取中间层logit、层级排名统计、HDL定位算法和系统性消融实验。

**📊 数据集**

使用的评估数据集包括CommonsenseQA、QASC、MMLU-Pro、SuperGPQA四个MCQA基准，并在GSM8K上做初步开放式生成探索。

**📈 对比分析**

通过与四种大模型、不同微调方式、不同选项数和不同标签格式对比，证明HDL处准确率急剧提升至接近最终层，可在HDL后安全裁剪约40‑80%的层数。

**⚠️ 局限性**

局限性包括未考虑链式思考、答案多样化形式、开放式生成任务以及对不同prompt格式的敏感性。

---

## 34. From Profiles to Steering Vectors: Global Sparse Priors and Local Semantic Calibration for Personalized Text Generation

**arXiv ID:** 2607.21620 | [PDF](https://arxiv.org/pdf/2607.21620v1)

**作者:** Liuji Chen `[一作]` (Chinese Academy of Sciences), Liang Wang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 45210 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GLASS，一种无训练、基于稀疏自编码器的全局-局部激活引导模型，用于个性化文本生成；

**💡 创新点**

创新点在于使用稀疏自编码器将LLM激活压缩到稀疏空间，从而更好地分离风格与语义，并结合全局用户风格向量与聚类后的局部对比向量，实现多粒度、低开销的个性化；

**🔧 技术方法**

采用稀疏自编码器（SAE）、激活引导（Activation Steering）、聚类与对比学习等技术；

**📊 数据集**

在LaMP和LongLaMP两个公开的个性化生成基准数据集上进行评估；

**📈 对比分析**

与检索、PEFT（如SFT、OPPU）和传统激活引导方法相比，GLASS在ROUGE-1/ROUGE-L及LLM-as-judge等指标上均实现了显著提升，且推理延迟低、存储需求小；

**⚠️ 局限性**

局限在于仍需依赖固定的SAE表示和激活聚合，可能无法进一步细化内容与风格的分离；实验范围仅覆盖标准基准，未检验在更复杂多域写作场景下的适用性。

---

## 35. Humanly: A Configurable and Traceable Environment for Human-AI Collaborative Writing

**arXiv ID:** 2607.21758 | [PDF](https://arxiv.org/pdf/2607.21758v1)

**作者:** Shenzhe Zhu `[一作]` (University of Texas at Austin), Jiaxin Pei `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，包括XXX和XXX。

**📊 数据集**

实验中使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明新方法在XXX指标上优于传统方法。

**⚠️ 局限性**

限制在于XXX，可能影响结果的普适性。

---

## 36. Multi-Horizon Consistency as Geometry: When Latent Dynamics Contract, and When They Do Not

**arXiv ID:** 2607.21645 | [PDF](https://arxiv.org/pdf/2607.21645v1)

**作者:** Kavya Bhand `[一作]` (Vishwakarma Institute of Technology), Aadi Joshi `[通讯]` (Vishwakarma Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在固定的GRU残差网络上调节多步潜在一致性权重λ，评估其对潜在转移几何的影响，利用95%分位数的k步扩张代理L_20,q95和20步预测误差E_20作为诊断指标，并在多种域（Passive Moving‑MNIST、Action‑conditioned Pendulum‑v1/CartPole‑v1、自然视频KTH）以及不同潜在维度和噪声水平下进行实验。

**💡 创新点**

创新点在于：① 将多步一致性权重视为“几何诊断开关”，揭示其在Passive视频域能快速将L_20降至1以下的阈值；② 发现该效应在Action‑conditioned或自然视频域不成立，形成“被动-主动收缩分界”；③ 通过注入可控噪声η，建立在固定λ下不同域可归一化到同一线性L_20≈a+bη法则，提供跨域统一解释；④ 对λ→L→E的关联做了关联性中介分析，提供了对该训练项影响路径的定量表述。

**🔧 技术方法**

使用的技术包括：GRU残差潜在动态模型、联合重构+一阶+多步一致性损失、随机梯度下降、配对t检验、Wilcoxon检验、bootstrap CI、关联性中介分析、噪声注入实验、MPC随机射击、WorldTest‑style 评分、以及对不同潜在维度的规模实验。

**📊 数据集**

使用的数据集为：Synthetic Moving‑MNIST（n=6 seeds 关键对）、Gymnasium Pendulum‑v1/CartPole‑v1（n=5/3 seeds）、KTH Actions 视频（n=3 seeds）、以及dm_control CartPole‑swingup（CPU sweep）等。

**📈 对比分析**

比较方法：对不同λ、η、潜在维度、域进行离散网格实验，记录L_20,q95、E_20、MPC回报、WorldTest得分等指标。结果显示：在Moving‑MNIST上，λ从0升至0.8能显著降低L_20和E_20，部分种子跨越L<1；在Action‑conditioned/自然视频域，尽管L_20下降，但未出现L<1，且MPC回报在λ增大时并未提升。噪声实验表明，L_20≈1.23+1.82η，在不同域上通过η_eff匹配得到相近的L_20预估，验证了线性法则。相比传统的光谱归一化或无对称RNN，soft一致性在Passive视频域表现相当甚至更好。

**⚠️ 局限性**

限制包括：① λ不是随机化干预，关联性中介分析无法证明因果关系；② L_20,q95仅为局部扩张代理，非全局Lipschitz证明；③ 仅使用GRU残差结构，结果对Transformer、RSSM或JEPA等其他架构的可迁移性未知；④ 试验规模有限（seed数少），对更大自然视频或更复杂控制任务的推广仍待验证；⑤ 仅评估了随机射击MPC，未涉及与模型共训练的策略，导致无法直接推断对最终控制回报的影响。

---

## 37. What Happens to Accuracy When Photo Lineups Contain Non-Mated Rank-One Images From Large Galleries?

**arXiv ID:** 2607.21792 | [PDF](https://arxiv.org/pdf/2607.21792v1)

**作者:** Genesis Argueta `[一作]` (University of Notre Dame), Jayeeta Dhar `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对一对多面部识别系统在不同数据库规模下产生的候选嫌疑图像，探讨其在照片排队（lineup）中的准确率与误判率，验证数据库规模对目击者识别偏差的影响。

**💡 创新点**

首次系统性实证研究将数据库规模与照片排队的误判概率关联，发现更大数据库使排名第一的非匹配图像与真凶更相似，从而显著提升误判率与误判信心，提供对警务实践的潜在风险警示。

**🔧 技术方法**

使用 ArcFace（ResNet-100）进行面部特征提取与余弦相似度排序，设计顺序显示的六人排队，并通过 ANOVA、卡方检验等统计方法评估不同条件下的识别准确率与置信度。

**📊 数据集**

采用 MORPH 数据集（v3 与 v5 版本）中筛选的非裔美国男性成年人样本，构建 500、5,000 与约 24,000 张图片的三组数据库，用于生成候选图像。

**📈 对比分析**

通过对四种排队条件（目标在场、500/5,000/24,000 规模数据库产生的非匹配图像）进行准确率、选择率、置信度的对比，结果显示 24,000 图像条件下准确率显著下降（37.5%），非匹配图像的选择率和置信度显著上升，表明数据库规模对识别性能有显著负面影响。

**⚠️ 局限性**

局限性包括：仅检验非裔美国男性样本，未覆盖其他种族/性别；数据库规模虽大于实验常规但远小于实际执法部署的数百万级规模；使用顺序排队而非同时排队，且未记录时间压力或详细决策时长；未控制发型等易变特征对识别的影响；实验参与者为校园学生，缺乏对不同人群的普适性评估。

---

## 38. Neural Feature Governance: Extending Atom Prevalence

**arXiv ID:** 2607.21671 | [PDF](https://arxiv.org/pdf/2607.21671v1)

**作者:** Idris Karel Seunda Ekwe `[一作]` (African Institute for Mathematical Sciences), Ernest Parfait Fokoué `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Neural Atom Prevalence (NAP) 方法，结合 Bayesian Lottery Ticket、Spike‑and‑Slab 先验与 Poisson‑Binomial 选择，在深度网络中实现节点级压缩、解释性、可靠的不确定性量化；

**💡 创新点**

创新点在于将 Fokoué 的原子优先原则迁移至神经网络，并通过四阶段流程（BLT → soft SS‑IG → PB 选择 → Bayesian fine‑tune）实现高效、可解释且具备不确定性估计的稀疏网络；

**🔧 技术方法**

采用了变分推理、Spike‑and‑Slab 先验、Gumbel‑Softmax 近似、Poisson‑Binomial 动态规划、Bayesian Lottery Ticket 的迭代幅度剪枝以及后续的 Bayesian fine‑tune；

**📊 数据集**

在仿真非线性回归、UCI Regression（Concrete、YearPredictionMSD）和 MNIST 图像分类数据集上进行实验；

**📈 对比分析**

与原始 SS‑IG 以及全密集 VBNN 进行对比；在回归任务中 NAP 在保持 3–4% 误差的同时实现 70–75% 节点压缩；在 MNIST 上实现 92% 节点压缩且 97% 准确率，覆盖率接近 95%；整体性能优于基线但在某些场景下略逊；

**⚠️ 局限性**

局限包括：均值场变分推理可能低估后验方差导致过度自信；对异常分布输入的鲁棒性不足；仅针对前馈网络，未扩展到 CNN/Transformer 等；四阶段流程计算成本高。

---

## 39. Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models

**arXiv ID:** 2607.21617 | [PDF](https://arxiv.org/pdf/2607.21617v1)

**作者:** Gwang Gook Lee `[一作]` (Amazon), Dimitrios Dimitriadis `[通讯]` (Amazon)

**通讯引用:** 1757 | [OpenAlex ID](https://openalex.org/A5115044944)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FaithC4 语料库，并评估了 15 种 OCR 与 VLM 系统的抄写忠实性，发现 VLM 在文本错误时倾向于改写而非逐字抄录。

**💡 创新点**

提出多语言扰动基准 FaithC4，揭示 VLM 的重写机制与内部表示相似度相关，并量化扰动对未扰动文本的非局部错误放大。

**🔧 技术方法**

使用视觉语言模型、传统 OCR、扰动实验、词级错误分类、层级内部表示相似度分析、注意力权重测量等技术。

**📊 数据集**

基于 C4（mC4）生成 1,455 页单页文档（英文、中文、韩文），按 scramble、random、visual 三种字符扰动进行渲染。

**📈 对比分析**

采用 WER 与 EDS 评估抄写准确性；结果显示传统 OCR 误差增长 <0.6pp，OCR 专用 VLM 0.2–2pp，通用 VLM 最差 0.74–4.45pp；扰动后未扰动词错误放大 5–10 倍。

**⚠️ 局限性**

仅覆盖三种语言，扰动类型有限，使用合成 PDF 可能不反映真实扫描，机制分析只针对 Qwen3-VL，未考虑其他模型或更细化的训练策略。

---

## 40. Directed Symbolic Execution for Vulnerability Discovery: An LLM-Guided Approach in KLEE

**arXiv ID:** 2607.21676 | [PDF](https://arxiv.org/pdf/2607.21676v1)

**作者:** Lingfeng Chen `[一作]` (Kyushu University), Yasutaka Kamei `[通讯]` (Kyushu University)

**通讯引用:** 5197 | [OpenAlex ID](https://openalex.org/A5045097606)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合大模型生成的漏洞位置标记并引导KLEE进行路径优先化，同时加入循环退出优先化，形成一种新的LLM‑guided directed symbolic execution框架。

**💡 创新点**

首次将LLM的语义理解作为多目标导向，同时融合循环退出优先化，解决符号执行在循环区域被卡住导致路径爆炸的问题。

**🔧 技术方法**

使用KLEE符号执行引擎、LLM（如DeepSeek-Coder、GPT‑4o等）、ICFG+SCC构建循环区域、路径优先化搜索器以及LLM prompt engineering。

**📊 数据集**

基准集为12个开源真实程序（bc、ncurses、bison、binutils、strip‑new、make、nasm、libtiff、jasper、transicc、flvmeta、curl），并使用ASan/UBSan检测安全违规。

**📈 对比分析**

与13种基线搜索器（Empc、CGS、SGS、DFS等）以及7种LLM做五次重复实验，测量行/基本块覆盖和漏洞发现；LLM‑guided方法提升约42%基本块覆盖、126%行覆盖，发现1335条违规、87条唯一违规，优于第二佳约29%/24%。

**⚠️ 局限性**

受KLEE路径爆炸、SMT求解限制和未标记循环区域影响，LLM标记质量与模型差异显著，且只能发现受ASan/UBSan支持的漏洞类型，存在漏检和泛化受限等局限。

---

## 41. Adjustment Speed as a Safety Constraint for Nonstationary Reinforcement Learning

**arXiv ID:** 2607.21646 | [PDF](https://arxiv.org/pdf/2607.21646v1)

**作者:** Timothy Tomashevskiy `[一作]` `[通讯]` (McMaster University), Timothy Tomashevskiy (McMaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于调整速度的安全约束框架（ASASC-NS），通过预测环境上下文变化、评估所需适应速度与可达适应速度的比值，若超出阈值即认为安全不可行，则主动收紧安全阈值并使用行动级盾牌，提升非平稳强化学习环境中短期窗口的安全性。

**💡 创新点**

①将适应可行性视为安全判据，系统化地将环境预测变化与学习系统的恢复能力关联；②引入可测的适应速度比率并基于此实现预先调节与动作级干预；③首次在连续/上下文感知 RL 中研究安全性与非平稳性关系。

**🔧 技术方法**

使用上下文编码器与预测器来估计当前与未来环境上下文；定义适应需求与适应容量并计算可行性比率；在 DQN 中通过奖励调整（optimization‑level）和行动集收缩（shielding）两种机制来实现安全约束；基于上下文感知 MDP 与动态安全阈值。

**📊 数据集**

主要在自定义的驾驶模拟环境中进行实验，环境通过 Markov 上下文切换机制模拟交通密度、车速、侵略性等变量的变化。

**📈 对比分析**

与无安全约束的基线 DQN、Adj‑only、Shield‑only、Full 四种变体进行对比；在强非平稳设置（p_stay=0.5）下，Full 方案在 EarlyViol 最低，Shield‑only 在 PeakRisk 与 TailViol 最低，整体相较基线短期窗口安全性显著提升。

**⚠️ 局限性**

仅关注短期安全性，未对长期任务回报做评估；验证仅在单一驾驶模拟环境中，缺乏跨任务或更复杂非平稳环境的泛化验证；适应容量阈值为经验估计，无法全面反映网络可塑性；方法对预测模型误差较为敏感。

---

## 42. Addressing the Orchestration Gap in Generalist Robots via Physical Agency

**arXiv ID:** 2607.21725 | [PDF](https://arxiv.org/pdf/2607.21725v1)

**作者:** Liane Galanti `[一作]` (Princeton University), Tri Dao `[通讯]` (Princeton University)

**通讯引用:** 2435 | [OpenAlex ID](https://openalex.org/A5091734792)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于前沿大语言模型的物理代理 orchestrator（PIGEY），在推理时将任务拆分为子目标，调度冻结的视觉-语言-动作（VLA）策略和 TAMP 规划器，并通过观察反馈进行验证与恢复，从而弥补“编排缺口”；

**💡 创新点**

核心创新是将任务层面推理、规划、执行验证和错误恢复全部放在推理时的闭环编排中，而非依赖单一大模型或额外训练，使得已有的低级技能得到最大化利用；

**🔧 技术方法**

技术包括：前沿视觉语言模型（如Claude Opus 4.7）、冻结的 VLA 策略 π₀.₅、TAMP 规划器 TiPToP、闭环验证（传感器+视觉一致性）、多工具 API（Perceive、Pick/DropAbove、VLARollout、Done）以及推理时的子目标序列化与记忆管理；

**📊 数据集**

使用的数据集包括：LIBERO‑PRO 真实环境仿真基准（6 个扰动组）以及 DROID 真实机器人平台的 30 项手动设计能力探测任务；

**📈 对比分析**

对比方法为：直接将指令喂给冻结的 VLA（π₀.₅）或 TiPToP 进行一次性执行；与我们的方法在相同冻结权重下仅改变推理流程。结果显示：在 LIBERO‑PRO 上成功率从 12.8% 提升至 53.3%（>4×）；在真实机器人上整体成功率从 48.7% 提升至 97.3%，在推理受限任务（世界知识、条件逻辑、多步推理、错误恢复、长序记忆）中差距尤为显著；

**⚠️ 局限性**

局限性包括：性能受限于所调度的低级技能；验证仍可能因遮挡或传感器误差产生假成功；推理时的 API 调用带来额外延迟和计算成本，限制了实时或高速应用；

---

## 43. When Model Release Meets Model Reuse: Producer-Consumer Misalignment in Hugging Face

**arXiv ID:** 2607.21738 | [PDF](https://arxiv.org/pdf/2607.21738v1)

**作者:** Adekunle Ajibode `[一作]` (Queen’s University), Ahmed E. Hassan `[通讯]` (Queen’s University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对50名 Hugging Face 模型发布者和95名 GitHub 模型使用者进行双向问卷调查，探究了模型发布与重用过程中的认知与实践差异。

**💡 创新点**

创新点在于首次从发布者与使用者双重视角系统梳理 AI 供应链中的文档、血统、治理等维度的误配与痛点，并提出针对性改进方向。

**🔧 技术方法**

主要采用问卷设计、定量统计分析（卡方检验、效应量估计）与定性主题分析方法，对收集的数据进行处理。

**📊 数据集**

数据来源为 Hugging Face 平台公开的模型元数据（共90,351名贡献者）与 GitHub 上包含 Hugging Face 模型调用的 169,314 个仓库，最终得到 50 名发布者与 95 名使用者的问卷回应。

**📈 对比分析**

研究采用问卷结果的描述性统计与显著性检验进行比较，并未涉及模型性能评估；结果主要表明发布者与使用者在文档位置、血统追溯目的、治理机制偏好等方面存在显著差异。

**⚠️ 局限性**

主要局限包括回应率偏低（约7%），样本可能不代表全球 PTLM 生态；仅覆盖 Hugging Face 与 GitHub 两大开源平台，难以推广到商业闭源供应链，且自述数据可能存在偏差。

---

## 44. Output Format x Model Identity: Interaction Effects in Single-Round Coding Agent Performance

**arXiv ID:** 2607.21674 | [PDF](https://arxiv.org/pdf/2607.21674v1)

**作者:** Yang Yang `[一作]` `[通讯]`, Yang Yang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在4,013次实验中，作者比较了3种LLM（DeepSeek V4、Doubao 2.0 Pro、Qwen 3.7 Max）在3种代码输出格式（完整文件、JSON Patch、统一 diff）对4个开源项目的单轮任务成功率。

**💡 创新点**

发现输出格式与模型存在交互效应，且没有通用最优格式，强调输出格式应视模型而定。

**🔧 技术方法**

使用大规模实验、随机抽样、统计检验（χ²、Fisher exact、Cohen's h）以及自定义格式解析管道。

**📊 数据集**

实验基于4个开源项目（tqdm、dotenv、requests、jsoup）和6个任务（bug‑fix、refactor、feature）在 tqdm 上进行确认。

**📈 对比分析**

通过对每个模型-格式-任务组合进行20次重复，评估 Pytest 测试通过率，发现 Doubao 在 JSON Patch 取得94%成功，DeepSeek 在统一 diff 取得66%成功，Qwen 在完整文件取得50%成功。

**⚠️ 局限性**

局限包括非确定性采样、不同格式解析工具差异、仅单轮实验、测试集对项目可解性影响未彻底分离，以及 Qwen 样本量提升后效果仍偏弱。

---

## 45. Oxygen-TryOn: Fashion-Native Foundation Model for Any-item Virtual Try-On

**arXiv ID:** 2607.21694 | [PDF](https://arxiv.org/pdf/2607.21694v1)

**作者:** Yong Liu `[一作]` (Jingdong), Simiu Gu `[通讯]` (Jingdong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Oxygen‑TryOn，一种统一的多参考服装试穿生成模型，能够在任意主体图像上基于任意数量的服装、配饰、鞋子或包包图像实现高保真、可编辑的虚拟试穿；

**💡 创新点**

创新点在于将试穿视为理解驱动的多参考生成任务，摒弃传统蒙版与仿真，利用大型多模态语言模型与多参考条件融合的扩散生成网络，实现可自由组合、层叠及编辑的任意物品试穿；

**🔧 技术方法**

采用JoyAI‑Image‑Edit框架，融合Qwen3‑VL‑8B‑Instruct的多模态大语言模型、Wan‑2.1‑VAE以及16B参数的多参考扩散变换器MMDiT，并通过CPT–SFT–RL三阶段训练和混合奖励策略提升一致性与真实感；

**📊 数据集**

构建了规模化多源数据引擎，汇集50+M电商、开放域时尚及合成图像，经过过滤、标注与合成对齐，生成覆盖207类商品、全身/局部主体以及单/多参考的高质量试穿三元组；

**📈 对比分析**

与公开基准DressCode、VITON‑HD、TStars‑VTON以及内部Oxygen‑TryOn Bench进行比较，Oxygen‑TryOn在FID、KID、SSIM、LPIPS、TStars‑VTON各维度均名列第一或前列，并在实际测试集上实现超过85%可直接部署的高可用率，显著优于开源FLUX.2和商业Nano Banana Pro、GPT‑Image‑2等系统；

**⚠️ 局限性**

目前仅支持最多四个参考物品的组合，稠密多物品层叠与复杂遮挡仍有提升空间；推理需多步扩散，速度受限，未来计划通过模型蒸馏加速和更大基模型扩展高卡密度组合。

---

## 46. CARE: Pre-Execution Command Verification for Shell-Executing LLM Agents

**arXiv ID:** 2607.21642 | [PDF](https://arxiv.org/pdf/2607.21642v1)

**作者:** Wenxiao Zhang `[一作]` (University of Western Australia), Jin B. Hong `[通讯]` (University of Western Australia)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5011163136)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CARE，一种针对 LLM 生成 shell 命令的预执行验证框架，先规范化命令，再通过结构、语义、路径和模式四个视角构造风险评分，最终仅对警告区分配 LLM 判定，实现高效且可审计的命令筛选。

**💡 创新点**

创新点在于将 shell‑specific 的静态多视角风险评估与选择性 LLM 判定相结合，形成可解释的证据链；采用“先静态、后 LLM”策略既保持低延迟，又显著降低误报。

**🔧 技术方法**

核心技术包括 bashlex 解析、命令规范化（去包装、解码、轻度去混淆）、结构/语义/路径/模式四层评分模型、阈值聚合策略以及 Qwen3‑Coder‑30B 作为 LLM 判定器。

**📊 数据集**

使用的主要数据集包括 549 条泄漏控制的主集（从 ART、RedCode‑Exec、NL2Bash、GTFOBins、Obfuscation、Exploit‑DB 等合并），以及 600 条 LLM 生成的 RedCode‑gen 攻击命令，进一步与 NL2Bash、GTFOBins、Obfuscation、Exploit‑DB 等 OOD 语料对比。

**📈 对比分析**

在与 12 种基线（正则、工具链、LLM 判定、通用内容安全模型等）比较时，CARE 在主集上实现 F1=85.6%、FPR=0.91%、平均延迟 2.32 ms（仅静态模式 0.34 ms），在 OOD 与攻击实验中保持低误报率和可接受的误杀率。

**⚠️ 局限性**

局限性包括仅处理单命令预执行，未覆盖跨命令序列或多轮交互的长期风险；对 LLM 判定存在生成者‑评判者偏倚；在某些 CVE 样本和复杂混淆技术上性能仍有限。

---

## 47. Coupled Hierarchical Search over Topology and Execution for Agentic Workflow Synthesis

**arXiv ID:** 2607.21609 | [PDF](https://arxiv.org/pdf/2607.21609v1)

**作者:** Dong Li `[一作]` (Baylor University), Haifeng Chen `[通讯]` (NEC Laboratories America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的 HierFlow 框架，利用层次化双空间（拓扑与代码执行）在推理时自动生成并优化 Agentic 工作流。

**💡 创新点**

创新点在于将工作流生成视为拓扑与执行的耦合搜索，使用反馈驱动的拓扑精炼、MCTS 风格的子任务代码搜索，以及基于搜索潜力与不确定性的自适应门控机制，实现高效、可预算的工作流优化。

**🔧 技术方法**

采用 LLM（gpt‑4o‑mini）评估与生成、MCTS‑启发式树搜索、基于拓扑度量的可耦合度分析、门控阈值阈值化、以及双空间代理搜索。

**📊 数据集**

在 HotpotQA、GSM8K、MATH、MBPP、HumanEval 等五个常用问答、数学推理与代码生成基准上进行实验。

**📈 对比分析**

与手工设计的 CoT、Self‑Refine 等方法以及自动化工作流生成方法（ADAS、AutoFlow、AFlow、MaAS、DyFlow、Flow 等）比较，HierFlow 在所有任务上均取得最高分，且在不同门控阈值下呈现可解释的精度-成本 Pareto 曲线。

**⚠️ 局限性**

局限性主要体现在：对高度耦合（高依赖密度）任务的层次化分解收益下降；门控阈值对性能敏感，需要经验调优；以及对极端复杂任务的搜索深度与预算上限仍有限制。

---

## 48. Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

**arXiv ID:** 2607.21653 | [PDF](https://arxiv.org/pdf/2607.21653v1)

**作者:** Jian Hu `[一作]`, Yi Dong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可读性优先、Token‑First、单循环异步的 RL 框架，能够在保持 PyTorch 原生实现的同时，与 vLLM、NeMo AutoModel 组合，支持大规模 MoE、单节点到 700B 参数模型的训练。

**💡 创新点**

创新点包括：
- 可读性和 AI 编码助手友好的设计原则，消除多层抽象与不必要的间接；
- Token‑First Agent Contract，允许任意 OpenAI/Anthropic SDK 直接训练，无需自定义接口；
- 纯异步循环与 Prompt‑Group 流式池，支持“partial rollout”与直接 NCCL 权重同步；
- 通过引擎回放实现 MoE 路由一致性；
- 在保持与 Megatron‑based 堆栈相当吞吐的前提下，代码量大幅降低。

**🔧 技术方法**

核心技术与组件：Ray（异步队列与资源调度），vLLM（服务端、prefix 缓存、spec decoding），NeMo AutoModel + FSDP2（分布式训练与 expert/context 并行），NCCL（权重同步），Python 纯净实现，token‑exact 传输与日志概率一致性检查。

**📊 数据集**

使用的数据集与任务：
- 2cDAPO‑Math prompts（2 000 条去重后、8 192‑token 上下文、16 192‑token 响应）用于 35B/700B MoE 评估；
- 8 000 条多轮工具使用与视觉‑语言任务（如 Qwen3‑30B‑A3B、Qwen3.6‑35B‑A3B）用于吞吐与内存测试。

**📈 对比分析**

比较方法：在匹配协议下，分别在 AutoModel+vLLM 与 Slime（Megatron‑Core+SGLang）上跑同一工作负载，记录每个优化步骤的平均时间与 tokens/GPU/s。结果显示：
- AutoModel+vLLM：119.4 ± 2.3 s/step，461 tokens/GPU/s；
- Slime：109.5 ± 10.3 s/step，502 tokens/GPU/s。
差异在统计可接受范围内，证明轻量框架在性能上与最先进的 Megatron‑based 堆栈相当。

**⚠️ 局限性**

局限性：
- 仅支持单一训练后端（AutoModel）和单一服务端（vLLM），无法直接使用多后端或自定义后端；
- 对引擎未实现的特性（如某些 MoE 组合、CUDA‑graphs 与 spec decoding 的兼容性）会在配置时快速失败；
- 目前实验集中在 35B/700B 参数模型，尚未验证更大规模（>3T）或更复杂环境（如多模态、长篇对话）下的收敛与稳定性；
- 需要进一步完善对非标准奖励源、分布式检查点以及更细粒度的资源调度策略。

---

## 49. Securing Multimodal AI through Internal Information Decomposition

**arXiv ID:** 2607.21600 | [PDF](https://arxiv.org/pdf/2607.21600v1)

**作者:** Jehyeok Yeon `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8603 | [OpenAlex ID](https://openalex.org/A5103178893)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

FlowGuard通过监控多模态推理内部的跨模态一致性来检测跨模态对抗攻击。

**💡 创新点**

引入PID启发的FlowVector四维信息量特征，构建零样本一类异常检测框架。

**🔧 技术方法**

采用多模态概率分布比较（KL、JSD、熵）构造FlowVector，利用Isolation Forest进行异常检测。

**📊 数据集**

训练使用VQAv2验证集，评估在MM‑SafetyBench、VLSafe、VLSU、MOSSBench等安全与常规基准上。

**📈 对比分析**

与CIDER、MirrorCheck、UniGuard、Llama Guard等对比，攻击成功率从>90%降至≤15%，AUROC≈0.94，保持≤3%功能损失，推理延迟≈1.3s。

**⚠️ 局限性**

对纯文本攻击的抵御仍相对薄弱，依赖完整或大截断的logprob，且在极端自适应攻击下可能被规避。

---

## 50. Local Synaptic Rules Can Implement a SIGReg Gradient Without Backpropagation

**arXiv ID:** 2607.21622 | [PDF](https://arxiv.org/pdf/2607.21622v1)

**作者:** Martin Andrews `[一作]` `[通讯]`, Martin Andrews

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于STDP+和闪光细胞型自适应突触可塑性的新型生物可实现自监督学习框架；

**💡 创新点**

将经典STDP+与自适应可塑性等价于SIGReg目标的梯度下降，实现了无需标签、反向传播或全局错误信号的端到端学习；

**🔧 技术方法**

使用STDP+（强化相位关系）、闪光细胞自适应（方差协方差正则化）和随机投影的梯度匹配，辅以梯度链路推导；

**📊 数据集**

在合成三类聚类数据、MNIST和CIFAR-10等公开数据集上进行实验；

**📈 对比分析**

通过与传统自监督方法（VICReg、SimCLR等）和基准模型比较，显示在无标签条件下可获得约87%（MNIST）和92%（CIFAR-10）线性探测准确率，证明该方法在保持无监督与生物可实现性的同时实现高性能；

**⚠️ 局限性**

局限性包括对层级白化的退化、STDP-抑制的负面影响、缺少完整的抑制神经元群体以及对深层网络的可扩展性不足。

---

## 51. LeafData: An Agentic System for Data Migration

**arXiv ID:** 2607.21618 | [PDF](https://arxiv.org/pdf/2607.21618v1)

**作者:** Sadanand Katukuri `[一作]` (University of Houston--Clear Lake), Yalong Wu `[通讯]` (University of Houston--Clear Lake)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于聊天机器人与后端服务的 Agentic 系统 LeafData，能够将自然语言用户意图自动转换为经过 Avro 验证的可直接被 Airflow 等编排平台消费的 JSON 配置，实现数据迁移管道的全流程自动化；

**💡 创新点**

创新点在于将交互式聊天界面与 schema 驱动的验证、JSON 构建结合，形成端到端从用户输入到可执行管道的闭环；

**🔧 技术方法**

使用技术包括前端聊天机器人、Apache Avro 方案校验、JSON 生成器、Airflow DAG 自动生成与执行、以及统一的连接器抽象层；

**📊 数据集**

实验数据集涵盖 AWS S3 上的 CSV 医疗记录、MongoDB 文档集合以及 REST API 返回的半结构化 JSON 记录；

**📈 对比分析**

通过对比传统手工配置方式，LeafData 显著降低了配置错误率、减少了人工时间投入，并实现了配置可复现性，虽然未给出具体性能指标但在实验案例中实现了成功迁移；

**⚠️ 局限性**

局限性主要是目前仅支持单一源单一目标的线性管道，尚未涵盖分支、条件逻辑及多阶段依赖的复杂工作流。

---

## 52. Do Modules Stay in Their Lane? Role Drift in Compound LLM Systems

**arXiv ID:** 2607.21627 | [PDF](https://arxiv.org/pdf/2607.21627v1)

**作者:** Xiaoyang Cao `[一作]` (Massachusetts Institute of Technology), Michiel A. Bakker `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1901 | [OpenAlex ID](https://openalex.org/A5035791917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在复合大型语言模型系统（由若干模块组成）中，端到端强化学习可能导致的“角色漂移（Role Drift）”问题，并提出一种正则化方法“角色锚点（Role Anchor）”来约束模块在训练过程中的角色一致性。

**💡 创新点**

创新点：①首次系统性揭示角色漂移现象，说明终端准确率提升并不意味着内部模块按设计执行；②提出仅依赖已有角色提示与中性提示的对比值来量化角色效用，并将其作为正则项加入强化学习；③通过梯度方向分析证明角色锚点通过抑制漂移方向的更新而非简单减弱学习，提供了新的理论解释。

**🔧 技术方法**

技术：使用基于LoRA适配器的可微分模型；对每个模块分别计算角色提示与中性提示下的对数概率差（角色效用）；将均值中心化后的角色效用平方误差作为正则项；在 REINFORCE 策略梯度框架下加入 λ 乘系数；在实验中还进行梯度方向、投影分析及 λ 的 ablation。

**📊 数据集**

数据集：①HotpotQA（用于 RAG 读写检索问答）；②MuSiQue‑Ans（用于分解–求解多跳推理）。

**📈 对比分析**

对比方法：无角色锚点的单纯强化学习（Outcome‑only RL）与加入角色锚点的 RL+Role Anchor。评估指标包括终端准确率与针对性角色忠实度探针（RAG 的 evidence‑following accuracy、DEC 的 answer‑entity insertion rate）。实验结果显示：①RL+Role Anchor 能在保持或仅略微降低终端准确率的同时，显著恢复角色忠实度；②在 DEC 系统中，约 86% 的终端准确率提升归因于角色漂移；③λ 的微小值即可抑制大部分漂移，进一步提升角色忠实度甚至在 RAG 上略优于无锚点模型。

**⚠️ 局限性**

局限性：①需要在训练时对每个模块分别访问角色提示与中性提示下的对数概率；②必须保存预训练参考模型用于计算中心化角色效用；③不适用于 API‑only 模块、非概率输出模块或完全通过提示学习的系统；④在其他类型的模块（如视觉感知或控制器）中实现需要进一步研究。

---

## 53. Procedural Knowledge Is Not Low-Rank: Why LoRA Fails to Internalize Multi-Step Procedures

**arXiv ID:** 2607.21612 | [PDF](https://arxiv.org/pdf/2607.21612v1)

**作者:** Simon Dennis `[一作]` (University of Melbourne), Rivaan Patil `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估LoRA在程序化任务中的效果，并通过系统的秩消融、跨域复现、SVD能量分析和训练动态证明LoRA在保持对话完成率的同时无法掌握程序化知识。

**💡 创新点**

首次揭示程序化知识本质为高秩变更，证明LoRA低秩更新在实用秩下无法捕获；提出层级感知LoRA可能改进但未验证。

**🔧 技术方法**

使用LoRA参数高效微调、全参数微调、SVD能量分析、基于Claude Sonnet 4.5的LLM-as-Judge动态用户仿真以及训练动态分析。

**📊 数据集**

三类合成流程数据集：旅行预订（14节点）、Zoom技术支持（14节点）和保险理赔（55节点），共计约9,876条对话。

**📈 对比分析**

与全参数微调在5个评判标准上进行Wilcoxon/Mann‑Whitney统计，LoRA在秩128时任务成功率仅为全微调的约51%，其余指标亦相差1–2分；全微调显著优于LoRA。

**⚠️ 局限性**

仅评估Qwen系列模型，未测试异构层级LoRA或其他基准；结果可能不泛化至不同模型架构或更异质的程序化任务。

---

## 54. Shallower ReLU Network Representations via Exact Linear Algebra

**arXiv ID:** 2607.21651 | [PDF](https://arxiv.org/pdf/2607.21651v1)

**作者:** Kilian Rueß `[一作]` (University of Technology Nuremberg), Martin Winter `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**通讯引用:** 89351 | [OpenAlex ID](https://openalex.org/A5000203192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

证明了在 n≤10 的情况下，最大函数 max(x₁,…,xₙ) 可以由两层隐藏层的 ReLU 网络完全表示，并给出递归构造实现任意 n 的深度上界为 log₅(n/2)+1。

**💡 创新点**

创新点在于利用对称性约化、支持函数与 Minkowski 和的关系，以及精确的有理线性代数求解，首次展示了在 n≤10 时的两层深度构造，并提出了比以往更紧的深度上界（相较于先前的 log₃ 上界）。

**🔧 技术方法**

核心技术包括：对称化与排序锥约束、支持函数线性系统求解、最大值门（rank‑2 maxout）表示、彩色多重图同构归约以及计算机辅助的有理解检验。

**📊 数据集**

本文不使用传统机器学习数据集，而是通过符号运算和有理数计算验证构造的正确性。

**📈 对比分析**

与已有工作（如 log₃ 上界）的比较显示深度上界得到改进；但论文并未给出实验性能指标，仅证明了存在性和深度上界。

**⚠️ 局限性**

局限性包括：仅适用于 ReLU 网络；对 n>10 的具体网络宽度未知；深度上界仍可能不是最优；递归构造导致宽度呈阶乘级增长，实际实现可行性受限。

---

## 55. AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems

**arXiv ID:** 2607.21604 | [PDF](https://arxiv.org/pdf/2607.21604v1)

**作者:** Nilesh Prasad Pandey `[一作]` (University of California), Tajana Rosing `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需训练、基于探针的 KV 残差校正方法 AgentKVShift，能够在代理式记忆检索时仅刷新少量 Token 并近似完整重编码。

**💡 创新点**

创新点在于发现 KV 重用误差主要由共享的记忆级偏移 + 微小的 Token 级波动组成，利用小探针集估计该偏移并一次性加权修正所有未刷新 Token，从而把刷新预算转化为全量有效信号。

**🔧 技术方法**

核心技术包括谱分析识别残差结构、探针选择与偏移估计、单向量加权校正、以及与 KV 量化的无缝组合。

**📊 数据集**

使用 LoCoMo 对话长记忆基准和 AMA‑Bench‑Recall 代理记忆基准，覆盖笔记式与图式记忆两类，模型规模从 3B 至 32B（Qwen、Mistral 等）。

**📈 对比分析**

与 CacheBlend、ProphetKV 以及全重编码 baseline 对比。AgentKVShift 在 r=10%/30% 的刷新率下，F1 仅落后 1.5–6%，预填充速度提升 2–3.5×，在 2/4-bit KV 量化下仍保留 2× 以上 F1，显著优于现有方法。

**⚠️ 局限性**

局限性：仅针对结构化代理记忆；对极大上下文或极低刷新率的场景尚未验证；探针选择与偏移估计依赖于子 Gaussian 假设，实际误差仍可能出现；不兼容某些高阶检索策略。

---

## 56. Risk-Routed Implicit Boundary Refinement for Robust Ultrasound Image Segmentation

**arXiv ID:** 2607.21787 | [PDF](https://arxiv.org/pdf/2607.21787v1)

**作者:** Jingguo Qu `[一作]` (Hong Kong Polytechnic University), Michael Ying `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 6004 | [OpenAlex ID](https://openalex.org/A5050577443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出RIBR框架，利用风险路由的隐式神经残差在超声图像分割中细化边界；

**💡 创新点**

创新在于将隐式网络仅作为局部残差校正，并通过风险路由门控精确控制何时写回，同时加入几何与散斑感知的边界正则化；

**🔧 技术方法**

采用U‑Net风格编码解码器、SIREN隐式神经网络、风险路由门控、几何正则化和散斑平滑等技术；

**📊 数据集**

使用9个多中心超声数据集，涵盖颈部淋巴结、乳腺、甲状腺、前列腺等内外部数据；

**📈 对比分析**

与U‑Net、U‑Net++、Attention U‑Net、MetaSeg、UNETR、SwinUNETR、TransUNet、VM‑UNet等同步比较，RIBR在宏平均Dice约76.7%、HD95约29.6，且参数仅0.4M，显示在外部中心和边界指标上优于大模型；

**⚠️ 局限性**

仅限二分类，缺乏多类/多实例处理，缺少前瞻性验证，后处理流程固定未集成学习方式。

---

## 57. FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs

**arXiv ID:** 2607.21624 | [PDF](https://arxiv.org/pdf/2607.21624v1)

**作者:** Kahou Tam `[一作]` (University of Macau), Li Li `[通讯]` (University of Macau)

**通讯引用:** 131392 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套面向移动 GPU 的 Transformer 微调框架 FBLayout，核心在于统一前向/反向计算的张量布局，消除昂贵的布局转换，并通过激活引导的全局布局选择实现高效训练。

**💡 创新点**

创新点包括：① R‑Tile 统一纹理内存布局，解决前向/反向的归约轴冲突；② 基于块的索引转换技术，直接在块层面做坐标映射，彻底消除物理数据移动；③ 激活可达性分析驱动的全局布局决策与图层级搜索，兼顾性能与内存占用；④ 结合移动 GPU 纹理内存特性进行架构感知的候选剪枝。

**🔧 技术方法**

使用技术：R‑Tile 纹理内存布局设计、块级索引映射、激活引导全局布局优化、架构感知的候选过滤、LLVM 强度归约、基于模板的 OpenCL 核心生成。

**📊 数据集**

实验基于七种 Transformer 模型：LLama3.2‑1B、Qwen2.5‑1.5B、Gemma2‑2B、BERT‑Large、ViT‑Large、Whisper‑Large、Stable Diffusion v1.5；每个模型在相应任务（NLP、CV、音频、图像）上使用标准微调数据集进行评估。

**📈 对比分析**

与 MNN、TFLite、TVM 进行对比，实验在三款主流手机（Snapdragon 8 Elite、8 Gen1、Dimensity 9400+）上完成。FBLayout 实现了 2.2–5.7× 的训练吞吐量提升，显著降低全局内存访问（≈4.2×）、缓存未命中率（≈4.2×）和能耗（≈3.5–6.3×）。

**⚠️ 局限性**

局限性：仅针对支持反向传播的移动 GPU，无法直接迁移至 NPU；在桌面级 GPU 上加速幅度减小；仍需一次 GPU 性能剖面来进行候选剪枝；部分极端形状转换仍需落回传统转换，且对大规模动态输入的支持需要进一步优化。

---

## 58. Tool-Guided Retrieval-Augmented Repair for Securing LLM-Generated C Code

**arXiv ID:** 2607.21641 | [PDF](https://arxiv.org/pdf/2607.21641v1)

**作者:** Vidyut Sriram `[一作]` (Pennsylvania State University), Suman Saha `[通讯]` (Pennsylvania State University)

**通讯引用:** 960 | [OpenAlex ID](https://openalex.org/A5029877298)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于编译反馈、静态分析与符号执行的检索增强修复工作流，用于提升LLM生成的C代码的可靠性和安全性。

**💡 创新点**

将编译错误、CodeQL安全检查与KLEE符号验证的多重工具反馈与检索增强的修复模式结合，并通过共享的修复经验库实现迭代改进，兼顾安全性与可编译性。

**🔧 技术方法**

LLM生成（如DeepSeek Coder 1.3B、CodeLlama 7B）、GCC编译、CodeQL静态安全分析、KLEE符号执行、检索增强（相似任务检索并提炼安全提示）

**📊 数据集**

包含5,000个通用C编程任务（算法、数学、字符串处理等）作为实验基准；未来计划加入嵌入式固件风格任务。

**📈 对比分析**

与基线单次生成对比，使用该工作流后编译失败率从42%降至22%（DeepSeek）/从46%降至27%（CodeLlama），安全缺陷率从49%降至19%（CodeLlama）/从35%降至15%（DeepSeek），CodeQL错误总数下降83.7%。

**⚠️ 局限性**

实验仅在通用C任务上进行，未覆盖真正的嵌入式硬件限制；缺陷分类仍以NIST类别为主，未细化到所有漏洞类型；缺失对检索与多工具反馈贡献的细粒度消融分析。

---

## 59. Transferable Latency Prediction for Fast LLM Screening on Heterogeneous Edge Devices

**arXiv ID:** 2607.21602 | [PDF](https://arxiv.org/pdf/2607.21602v1)

**作者:** Xiaolong Tu `[一作]` (Georgia State University), Haoxin Wang `[通讯]` (Georgia State University)

**通讯引用:** 2921 | [OpenAlex ID](https://openalex.org/A5101899364)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了可迁移的实时延迟预测框架，用于在异构边缘设备上快速筛选LLM模型。

**💡 创新点**

创新点在于将硬件、运行时、模型与提示配置统一建模，并通过阶段感知的动态特征与静态描述结合，利用门控融合实现少量校准即可跨设备迁移。

**🔧 技术方法**

采用双路径（静态/动态）网络、时序编码、prefill/decoding阶段分离、门控融合以及轻量级校准微调等技术。

**📊 数据集**

使用Pixel 8/Pro、Jetson Nano、Orange Pi 5 Pro、RTX 3090等多平台的请求级profiling数据，共约515条记录。

**📈 对比分析**

与仅使用静态特征、无阶段感知、无门控融合等基线对比，MAE下降约20%（prefill/decoding/总时延分别提升至R²≈0.96），跨设备迁移需少量校准后R²可提升至>0.4，预测还能保留最佳可行模型，过滤比为0.33–0.5。

**⚠️ 局限性**

局限性在于基准数据主要来自Pixel移动设备，跨设备性能仍高度依赖校准样本；模型与提示覆盖范围有限，未在所有平台统一训练；未评估能耗、吞吐量等其他部署指标。

---

## 60. Smart predict-then-robustly-optimize

**arXiv ID:** 2607.21773 | [PDF](https://arxiv.org/pdf/2607.21773v1)

**作者:** Aakil Caunhye `[一作]`, Belen Martin-Barragan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Smart Predict-then-Robustly-Optimize (SPrO+) 框架，基于 Smart Predict-then-Optimize (SPO) 将鲁棒化直接嵌入下游决策空间，实现可凸的预测-优化一体化学习。

**💡 创新点**

创新点在于：① 构造了 SPrO+ 的凸 surrogate，保持与 SPO+ 同等可解性；② 在理论上证明了 surrogate 的有限样本 Fisher 一致性、子高斯集中性质及对决策变量的 Lipschitz 连续性；③ 给出 SPrO+ 在点wise 和期望意义上相对 SPO+ 的支配条件，并与上游鲁棒化（SrPO）进行比较。

**🔧 技术方法**

主要技术包括：鲁棒优化（使用预算不确定集）、凸优化（求解 SPrO+）、子高斯理论与集中不等式、线性回归预测、网络流问题建模，以及数值实验中的训练稳定性评估。

**📊 数据集**

使用随机生成的网络流数据集：10 个节点、30 条弧、100 条样本，并在此基础上通过添加受限噪声生成 100 个受污染数据集进行实验；也对更大规模（20 节点、60 条弧）和更高不确定预算的情形进行了敏感性分析。

**📈 对比分析**

与传统 SPO+ 及上游鲁棒化 SrPO+ 进行对比。实验显示，SPrO+ 在训练稳定性（决策损失波动显著降低）和 out-of-sample 决策损失方面均优于两者；平均误差从约 32.5 降至 10.3，方差从 6.7 降至 2.9；在更高不确定预算或更大维度时优势更为明显。

**⚠️ 局限性**

局限性：① 只在线性预测器下证明理论和实验，非线性情况需进一步扩展；② 鲁棒预算 λ 的选取需要经验调参，过大或过小都会影响性能；③ 对极端分布（非子高斯）假设的适用性未充分验证。

---

## 61. Toward Goal-Agnostic Joint-Embedding Predictive Control of Partial Differential Equations

**arXiv ID:** 2607.21644 | [PDF](https://arxiv.org/pdf/2607.21644v1)

**作者:** Jonathan Gallagher `[一作]` (University of Waterloo), Roberto Guglielmi `[通讯]` (University of Waterloo)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5089826410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于joint‑embedding的目标无关 PDE 控制框架，并在二维 Navier–Stokes 基准上使用学习到的动能（KE）探针改进控制目标。

**💡 创新点**

创新点包括：① 通过无奖励、无重构的 JEPA 世界模型实现目标无关的动态学习；② 用可插值的物理可观测量（KE）代替潜在欧氏距离来定义控制目标；③ 在同一冻结模型上支持多种控制任务（跟踪、稳态、能量调节）。

**🔧 技术方法**

采用了轻量级 2D ViT 编码器、行动条件潜在预测器、VICReg 正则化、Delta‑JEPA 预测动作的差分解码器、MPPI 模型预测控制以及线性 KE 探针回归。

**📊 数据集**

使用 PDE Control Gym 的二维 Navier–Stokes 基准数据集，包含 6930 条包含 200 帧速度场和 199 次动作的轨迹。

**📈 对比分析**

与传统的潜在 L² 跟踪、公开的 SAC 与模型优化器对比，KE‑probe 在 50 次试验中将平均奖励从 -12.08 提升至 -10.90，三组无周期目标的末端场 RMSE 降低 53%（从 0.0469 降至 0.0220），在所有 30 对比中全胜。

**⚠️ 局限性**

主要限制是 KE 探针对噪声或缺失像素高度敏感，观测损坏时误差骤增；该方法目前仅在单一边界激励且能量可唯一恢复的情形下有效。

---

## 62. Trajectory-Aware Retrieval Agents for Temporal Decision- Making

**arXiv ID:** 2607.21625 | [PDF](https://arxiv.org/pdf/2607.21625v1)

**作者:** Jing Wang `[一作]`, Xing Niu `[通讯]` (Amazon)

**通讯引用:** 3491 | [OpenAlex ID](https://openalex.org/A5069475690)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出TLM框架，结合检索、LGCM轨迹建模与SHAP迭代优化，实现对长文本中时序信息的决策推理。

**💡 创新点**

首次将潜在增长曲线模型和Shapley值引入检索增强生成流程，提供可解释的时序信号和高效的证据修正机制。

**🔧 技术方法**

采用混合BM25+Dense检索、线性LGCM、注意力评分器、LLM分类器和基于轻量化评分器的SHAP估计。

**📊 数据集**

在医学多项选择问答（MedQA）、财报惊喜预测和隔夜股价跳空预测三个任务上使用公开数据集。

**📈 对比分析**

与零样本LLM、单一检索或嵌入分类器对比，TLM在MedQA上从34.6%提升至64.2%，财报预测58.0%对比50.7%，股价跳空预测从27.5%提升至38.3%，并带来显著资本增长。

**⚠️ 局限性**

假设时间序列仅由片段位置决定且LGCM线性，难以处理多线索并列或非线性变化。

---

## 63. Lost in Context: Addressing Context Anxiety in Large Language Models

**arXiv ID:** 2607.21616 | [PDF](https://arxiv.org/pdf/2607.21616v1)

**作者:** Ifueko Igbinedion `[一作]` (Massachusetts Institute of Technology), Eric So `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3750 | [OpenAlex ID](https://openalex.org/A5084226439)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化大语言模型在长序列推理中出现的“上下文焦虑”，探究其原因与后果，并通过有针对性的微调降低此类焦虑。

**💡 创新点**

首次系统化地定义并测量上下文焦虑，证明其主要来源是模型对所需 token 数量的误估，并展示通过仅微调无焦虑推理轨迹即可显著提升模型性能与效率。

**🔧 技术方法**

使用了自动化焦虑检测协议、token 估计校准指标（winsorized token ratio）和轻量级的“推理微调”（Reasoning SFT）技术；还采用了经济计量回归分析来量化焦虑对准确率与 token 使用的影响。

**📊 数据集**

主要数据集为 Tower of Hanoi 的多盘数问题（2-12 盘，共 270 个实例），并在短路径搜索（Shortest‑Path Grid Search）任务上做了跨任务验证。

**📈 对比分析**

与多种前沿模型（Claude Sonnet 3.7/4.5、DeepSeek R1、Gemini 2.5 Flash 等）相比，微调后的模型在中等难度下提升了约 6–7% 的准确率，且在存在焦虑时准确率下降了 15% 的问题被修复；同时 token 使用量下降 54%，显示显著的效率提升。

**⚠️ 局限性**

局限性：研究聚焦单一符号推理任务，可能无法覆盖所有实际情境；焦虑检测依赖模型显式自述，可能低估隐性焦虑；跨任务与跨领域的普适性仍需进一步验证。

---

## 64. On the Depth Scalability of Logic Gate Networks

**arXiv ID:** 2607.21633 | [PDF](https://arxiv.org/pdf/2607.21633v1)

**作者:** Taegun An `[一作]` (Association for the Advancement of Artificial Intelligence), Changhee Joo `[通讯]` (Korea University)

**通讯引用:** 2960 | [OpenAlex ID](https://openalex.org/A5035314009)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了输入锚定逻辑门网络（IALGN），在传统逻辑门网络的基础上每一层的门既接收前一层隐藏特征，又接收原始输入特征；

**💡 创新点**

创新点在于通过在每层加入直接输入锚点，既保持了可训练的隐藏计算路径，又让每一层都能获取新的输入信息，从而形成严格的路径深度层级；

**🔧 技术方法**

技术主要包括可微分的逻辑门操作、跳跃偏置初始化、直通估计器（STE）以及随机k锚点放松（random‑k anchor relaxation）；

**📊 数据集**

实验数据集涵盖MNIST、CIFAR‑10和CIFAR‑100；

**📈 对比分析**

与随机连线逻辑门网络（RWLGN）和先前的DLGN实现相比，在固定宽度下深度增加时，IALGN在所有数据集上均能持续提升准确率，尤其在CIFAR‑10/100上表现显著；

**⚠️ 局限性**

局限在于仅评估了前馈扁平化网络，未探讨在固定总门数预算下的深度-宽度权衡、卷积变体及硬件实现等方向。

---

## 65. StajChain: A Hyperledger Fabric-Based Multi-Party Internship Agreement System

**arXiv ID:** 2607.21643 | [PDF](https://arxiv.org/pdf/2607.21643v1)

**作者:** Rampia Perente `[一作]` `[通讯]` (ISTANBUL TECHNICAL UNIVERSITY), Rampia Perente (ISTANBUL TECHNICAL UNIVERSITY)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个基于 Hyperledger Fabric 的权限链条（StajChain）系统，用于管理多方实习协议的创建、审批、激活、完成、拒绝等生命周期，并提供链上不可篡改记录与链下数据库同步。

**💡 创新点**

创新点包括：将多方实习审批流程数字化为可验证、可追溯的区块链工作流；设计角色基于权限控制与状态机式审批逻辑；实现公司注册审批与 Fabric 身份自动化；通过链上链下分离的数据模型实现高效查询；在本地 Fabric 环境下完成完整功能的性能评测。

**🔧 技术方法**

技术栈：React 前端 + Node.js/Express 后端 + SQLite（链下） + Hyperledger Fabric（链上） + Fabric SDK + JWT 认证 + Fabric CA 身份创建；使用 Fabric Chaincode 实现业务规则与状态转移。

**📊 数据集**

使用自定义测试数据：模拟学生、公司、教师、实习协议等表，生成各种功能和性能测试场景；未使用公开真实数据集。

**📈 对比分析**

比较方法：功能测试（正负案例）与性能基准测试（TPS、P95 延迟）。结果显示读取操作 TPS 124–160，P95 < 100 ms；写操作 P95 ≈ 2 s；SQLite 查询 P95 14 ms。与其他 Hyperledger Fabric 性能研究对比，写入延迟相符，读取吞吐可接受，满足 20–30 TPS/分钟、5 s 交易延迟等目标。

**⚠️ 局限性**

局限性：仅在单组织本地测试，未部署跨组织网络；公司身份创建与审批耗时较长；未进行正式可用性测试；链下数据存储在 SQLite，未使用分布式存储；文档存储未实现链上哈希与 IPFS 集成。

---

## 66. Co-design of LLM-based preference agents: participation may drive overtrust

**arXiv ID:** 2607.21757 | [PDF](https://arxiv.org/pdf/2607.21757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. From Obligation to Specification: A Survey on Validating EU AI Act Requirements in RE

**arXiv ID:** 2607.21608 | [PDF](https://arxiv.org/pdf/2607.21608v1)

**作者:** T. Y. Emmy Lai `[一作]`, Héctor Allende-Cid `[通讯]` (Fraunhofer IAIS)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对欧盟 AI 法案在需求工程（RE）中的落实现状进行实证调查，结合专家访谈与在线问卷，评估组织对法规的准备度以及对基于 LLM 的代理闭环验证工具的接受度。

**💡 创新点**

首次将组织准备度、法规解读与人工智能辅助验证工具的可接受性三者结合，提出了 AI 法案就绪闭环验证代理的最小可行要求（透明度、可审计、访问控制、HITL 监督与数据不完整鲁棒性），并为后续实现提供经验与准则。

**🔧 技术方法**

采用混合方法：半结构化专家访谈与基于 Likert 量表的问卷；提出的验证概念基于 LLM（大型语言模型）与多代理系统技术，但论文未实现完整系统，仅在概念层面讨论其作用。

**📊 数据集**

使用 10 次专家访谈记录与 15 份问卷回复的自报告数据；未使用公开数据集或进行模型训练。

**📈 对比分析**

通过描述性统计与分组对比来评估组织准备度与工具接受度，没有进行性能对比或实验验证；结果显示组织准备度普遍偏低，且对 LLM 工具的使用高度受限于透明性与人为监督。

**⚠️ 局限性**

局限性包括样本量小、便利抽样、受访者自我报告可能产生偏差；未对工具进行实验验证；构念效度未进行正式测量；研究范围主要集中在德国，缺乏跨文化与跨行业验证。

---

## 68. Learning Diverse Humanoid Tasks via Synthetic Video Scenarios without Real World Data

**arXiv ID:** 2607.21648 | [PDF](https://arxiv.org/pdf/2607.21648v1)

**作者:** Yun-Hao Tsai `[一作]` (National Cheng Kung University), Yen-Chen Liu `[通讯]` (National Cheng Kung University)

**通讯引用:** 2979 | [OpenAlex ID](https://openalex.org/A5057832539)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用生成式AI通过文本提示生成多样化的人体动作视频，再将其转换为仿人机器人可执行的轨迹并进行强化学习训练。

**💡 创新点**

创新点在于完全用生成式AI替代真实演示，生成多样化动作，并通过动作拼接实现复杂任务的连贯执行。

**🔧 技术方法**

技术包括Prompt-driven视频生成（Veo 3 API）、SMPL-X姿态重建、GMR动作重定向、RL（PPO）轨迹跟踪和Isaac Lab仿真。

**📊 数据集**

数据集为50个日常任务提示，生成10个视频样本，合计约500个动作序列；使用Unitree G1仿真环境进行训练和评估。

**📈 对比分析**

通过对比MAE和执行误差评估，实验显示生成式AI数据训练的策略在四个仿真场景中能准确跟踪动作、保持稳定，误差在0.04-0.07m之间，优于传统单一示范方法。

**⚠️ 局限性**

局限在于对高速或极动态动作的生成仍不稳定，结果仅在仿真环境验证，缺乏真实机器人试验和更复杂任务的适用性。

---

## 69. Probing Latent Colombian Identity Inferences in Qwen2.5-7B with Natural Language Autoencoders

**arXiv ID:** 2607.21774 | [PDF](https://arxiv.org/pdf/2607.21774v1)

**作者:** Pablo Santiago Potes Velasco `[一作]` (Universidad Autónoma de Occidente), Gilber Alexis Corrales Gallego `[通讯]` (Universidad Autónoma de Occidente)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究探讨 Qwen2.5-7B-Instruct 在处理哥伦比亚西班牙语与英语提示时，是否会在未显式表述前内部推断出哥伦比亚身份或相关社会经济特征。

**💡 创新点**

创新点在于使用无监督、无训练的自然语言自动编码器（NLA）对模型残差流进行定位，检验隐式民族身份推断的时间点与出现率。

**🔧 技术方法**

技术手段包括 Qwen2.5-7B-Instruct 的残差流激活提取、四分位采样、NLA 生成解释以及统计检验（Wilson 置信区间、Fisher 精确检验、Mann–Whitney U 检验）。

**📊 数据集**

使用了 30 条提示（15 对西班牙语-英语匹配），涵盖显式、隐式和中性三种提示类型，共 3 种实验条件。

**📈 对比分析**

通过比较各组在四个分位点的民族身份提及率和显著性检验，发现隐式提示在第 4 分位点显著高于中性提示（p=0.023），但总体差异仅在探索性水平。

**⚠️ 局限性**

局限性包括样本量小（每组仅 5 个场景）、仅在单一层级和单一激活采样下评估、缺乏统计显著性检验、以及 NLA 可能产生与上下文不符的“自发”解释。

---

## 70. Prompt as a Data Type: In-Database LLM Prompt Management and Rewriting

**arXiv ID:** 2607.21756 | [PDF](https://arxiv.org/pdf/2607.21756v1)

**作者:** Denis Mayr Lima Martins `[一作]` (University of Sao Paulo), Gottfried Vossen `[通讯]` (University of Muenster)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将提示语作为元组级数据库数据类型实现，构建了可存储、可重写、可优化、可在 DBMS 内执行的 LLM 提示管理系统，并在 DuckDB 上实现了相应的数据类型、评估视图、内部运算符与规则化重写框架。

**💡 创新点**

提出“提示语即数据类型”的概念，把提示语提升为首席数据项；将传统查询优化器的思路迁移到提示语重写上，形成约束注入、列投影、输出格式化、数据库选取示例等四类可优化规则；给出正式的提示语关系模型、评估、重写上下文和成本质量权衡框架。

**🔧 技术方法**

自定义数据类型 PROMPT、生成评估视图、实现 EVAL UDF、规则化重写器、成本质量启发式评估、使用 DuckDB、通过 LMStudio 调用 LLM、实验中采用输入/输出 token 统计与延迟测量。

**📊 数据集**

三种数据集：合成客服工单（synthetic tickets），OpenML 上的汽车评估（Car Evaluation），以及 DuckDB 生成的 TPC‑H。

**📈 对比分析**

与静态提示、固定全套重写规则以及成本感知重写策略进行对比。结果显示，规则化重写在需要结构化输出的任务（属性提取、语义过滤）中显著提升准确率和有效输出率；成本感知策略在保持相似质量的同时降低 token 消耗；规则消融实验表明各规则在不同任务中起作用；整体性能与传统静态提示相比有明显提升。

**⚠️ 局限性**

启发式质量估计不足，未能充分考虑任务差异；目前的重写规则和优化器仍基于固定策略，缺乏学习或校准的成本/质量模型；对复杂多表查询、外键路径等更丰富的数据库上下文支持有限；原型仅在内存模式下实现，未深入集成到成熟 DBMS；对某些任务（如语义值归一化）效果不明显。

---

## 71. A method of Risk Analysis and threat management using analytic hierarchy process

**arXiv ID:** 2607.21691 | [PDF](https://arxiv.org/pdf/2607.21691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 72. What AI Red-Team Evaluations Can and Cannot Prove

**arXiv ID:** 2607.21735 | [PDF](https://arxiv.org/pdf/2607.21735v1)

**作者:** Bandana Kaur `[一作]` `[通讯]`, Bandana Kaur

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究红队评估的证据上限，给出闭式表达式并评估当前基准在不同伤害率下的充分性。

**💡 创新点**

引入“证据上限”概念，推导其闭式形式，确定红队评估在不同伤害率下可否证实安全性的阈值。

**🔧 技术方法**

使用统计推断、零计数界、二项分布理论、蒙特卡洛仿真以及TF‑IDF和句子嵌入相似性度量等技术。

**📊 数据集**

使用AdvBench、HarmBench、XSTest、SafetyBench和LMSYS‑Chat‑1M等公开评估集合。

**📈 对比分析**

通过理论计算与仿真估计误判率与功效，发现现有基准在高频伤害率下足够，但在低频灾难性类别下不足三阶数量级。

**⚠️ 局限性**

受限于伤害率估计假设、基准分布与结构一般性不足、对自适应采样适用性有限，且未考虑多种证据源的合成阈值。

---

## 73. Charting the Moral Universe: Capturing Virtues and Values of Data Visualization Practice

**arXiv ID:** 2607.21732 | [PDF](https://arxiv.org/pdf/2607.21732v1)

**作者:** Chloe Hudson Prock `[一作]` (Northeastern University), Michael Correll `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该研究通过对20名可视化从业者的半结构化访谈，提炼出68项伦理价值并将其归纳为9个美德聚类，为可视化实践提供了新的道德框架。

**💡 创新点**

创新点在于将伦理美德与价值系统具体化为可操作的聚类，并强调非规则化的实践导向，而非传统的条规式伦理指南。

**🔧 技术方法**

研究使用了访谈、主题分析、手工编码和亲和图绘制等质性方法来捕捉并整理受访者的伦理观念。

**📊 数据集**

数据来源为20位访谈受访者的访谈记录（未公开任何第三方数据集）。

**📈 对比分析**

作为定性研究，没有传统意义上的性能比较；通过多位研究者的双重编码和共识讨论来验证编码的一致性与主题的可靠性。

**⚠️ 局限性**

局限性包括样本规模有限、文化背景偏向西方、结果高度主观且缺乏实证验证，难以全面代表全球可视化实践。

---

## 74. Be Consistent! Enhancing Robust Visual Reasoning in LVLMs with Consistency Constraints

**arXiv ID:** 2607.21722 | [PDF](https://arxiv.org/pdf/2607.21722v1)

**作者:** Liqiang Jing `[一作]` (University of Texas at Dallas), Vassilis N. Ioannidis `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个新的视觉中心推理基准，并基于双重奖励的强化学习方法提升LVLM的推理一致性和准确性。

**💡 创新点**

创新点在于构建逻辑等价问题对的基准和引入一致性奖励，利用GRPO训练无需严格标签即可提升一致性和准确性。

**🔧 技术方法**

使用GRPO（Group Relative Policy Optimization）强化学习、双重奖励（准确性+一致性）以及自动生成逻辑等价问题对。

**📊 数据集**

训练集使用MSCOCO的5k图像，自动生成问题对后人工校验；评估集为自建基准、V*Bench、InfoVQA-test。

**📈 对比分析**

与多种开源/闭源模型对比，ConVLM在一致性与鲁棒准确率上均刷新开源模型记录，超过Claude-3.5-Sonnet-V2；在V*Bench、InfoVQA也表现出色。

**⚠️ 局限性**

仅关注单图像的等价问答一致性，未考虑跨图像一致性，奖励为0/1限制了精细化；未探索其他模态或任务特定奖励。

---

## 75. Persistent Computational State: A Session-Centric Runtime for Generative World Models

**arXiv ID:** 2607.21686 | [PDF](https://arxiv.org/pdf/2607.21686v1)

**作者:** Zhen Lin `[一作]` `[通讯]`, Zhen Lin

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了视频世界模型失去连续性的问题根源在于运行时在请求边界丢弃不可重构状态，而非模型本身缺陷，并提出了Persistent Computational State (PCS) 的概念与发现方法。

**💡 创新点**

创新点在于将持续计算状态视为最小不可重构状态、通过测量指纹化方式自动发现PCS，并构建了会话级运行时与返回一致性符合测试。

**🔧 技术方法**

采用了指纹化实验、离散化的返回一致性度量、会话级快照/恢复 API、以及基于分区和相关性优先的驱逐策略。

**📊 数据集**

使用了MBench、WRBench以及自制的离开-返回轨迹、世界记忆模型以及Matrix-Game 2.0等多种模型与视频数据集进行评估。

**📈 对比分析**

与传统请求级快照和LLM缓存策略对比，本文的会话级PCS实现恢复无误差、占用的 GPU 内存保持常数，主机内存线性增长，且在 1024 会话时仍能保持高一致性，显著优于对比方法。

**⚠️ 局限性**

局限性包括仅验证了三种模型架构，指纹化对训练变化的稳定性未知，单 GPU 时序多路切分的结果不一定推广到多 GPU 并发场景，且对非可测量状态的支持尚未覆盖。

---

## 76. MosaicJoin: Compact Semantic Sketches for Value-Level Join Discovery

**arXiv ID:** 2607.21781 | [PDF](https://arxiv.org/pdf/2607.21781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 77. Evaluation design conditions the expert-vs-auto MeSH gap: a controlled comparison of bag-of-words and BiomedBERT on the Cohen benchmark

**arXiv ID:** 2607.21685 | [PDF](https://arxiv.org/pdf/2607.21685v1)

**作者:** Samuel M. Okoe-Mensah `[一作]` `[通讯]`, Samuel M. Okoe-Mensah

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Cohen药物类别基准上，比较专家与机械化MeSH注释对系统综述筛选分类器的影响；

**💡 创新点**

揭示了评估设计（语料规模、交叉验证折数）对专家-机械化MeSH差异的显著影响，表明该差异是设计条件下的；

**🔧 技术方法**

使用词袋逻辑回归和BiomedBERT两种分类器，并采用WSS@95%作为主要评价指标；

**📊 数据集**

采用Cohen 2006药物类别基准中的Statins、Opioids和ADHD三类数据集；

**📈 对比分析**

通过多次重复、Bootstrap置信区间和精确置换检验比较，发现Bag-of-Words在5折时专家优势+0.096，10折和子样本下降至≈0.02；BiomedBERT在5折时优势≈+0.02，整体无显著差异；

**⚠️ 局限性**

主要局限包括评估设计限制、BiomedBERT 512-token截断导致MeSH信息损失、仅使用MeSH本体、只测试三类主题、未对超参数进行调优、机械化MeSH方法过于简单以及随机种子波动大等。

---

## 78. Khondo: A Multimodal Benchmark for Document Packet Splitting of Bangla Forms

**arXiv ID:** 2607.21780 | [PDF](https://arxiv.org/pdf/2607.21780v1)

**作者:** Abu Tyeb Azad `[一作]` (Wichita State University), AKM Mahbubur Rahman `[通讯]` (Independent University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Khondo 这一多模态基准，专门针对孟加拉国政府表单的文档包拆分任务，涵盖 5 种拼接方式（从顺序到完全打乱）并在 14 个行政领域内生成 1,950 个双语（孟加拉语-英语）图像包。

**💡 创新点**

首次为低资源语言提供基准；将拆分任务迁移到纯视觉端（直接处理页面图像），并通过“页面顺序重建”这一关键难点突出低资源场景下的挑战。

**🔧 技术方法**

使用多模态大型语言模型（MLLM）进行零-shot 推理与 QLoRA 微调的小型 4B 模型；通过多样化提示（顺序指令 vs 无顺序指令）以及双语对照实验评估模型表现。

**📊 数据集**

Khondo 数据集（1,950 个文档包，28,513 页，含 423 个源表单），覆盖 14 个行政域，语言比例约 67% 孟加拉语、26% 英语、5% 混合。

**📈 对比分析**

评估指标包括聚类质量（V-measure 与 Rand 指数）、页面顺序的 Kendall τ 与整体分数；旗舰 MLLM 在顺序版表现优秀（S_ord≈0.95），但在打乱版聚类稳健（S_clu 0.55–0.85），页面顺序下降显著；小型微调模型提升了顺序分数但仍无法匹配最优结果。

**⚠️ 局限性**

主要局限：页面顺序重建仍为瓶颈；对低资源语言的语言差异造成额外误差；仅基于孟加拉语-英语双语，未覆盖更多区域语言；并未探索更大规模模型或强化学习等更深层次的适配策略。

---

## 79. From Seasonality to Semantics: Benchmarking a Hybrid Probabilistic Forecasting System for Roadblocks in Bolivia

**arXiv ID:** 2607.21785 | [PDF](https://arxiv.org/pdf/2607.21785v1)

**作者:** Rodrigo Vargas Sainz `[一作]` (Universidad Privada de Santa Cruz de la Sierra), Christian Berón Curti `[通讯]` (Universidad Privada de Santa Cruz de la Sierra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研发了一套混合概率预测系统，结合 Prophet 时序分解和 NLP 文本语义嵌入，对玻利维亚道路封锁事件进行多步预测。

**💡 创新点**

创新点在于：①将季节性时间序列模型与新闻文本的密集向量嵌入融合，②证明语义嵌入可取代传统零射分类，③实现了既能捕捉历史惯性又能预警文本信号的最简洁架构（C6）。

**🔧 技术方法**

使用技术包括 Prophet、零射主题分类、情感分析、multilingual sentence embeddings（384 维）、XGBoost + Platt 标定、展开式 Walk‑Forward 验证、Diebold‑Mariano 检验、AUC‑ROC、Brier Score、SHAP 解释。

**📊 数据集**

数据集为 2020‑2026 年玻利维亚新闻聚合器 386,884 条头条（每日平均 181 条）与官方道路封锁记录 1,762 天，构成训练与评估。

**📈 对比分析**

通过内部七种配置和四个外部基准（Logistic 回归、LightGBM、SARIMA、XGBoost）在 1,762 天的展开式验证下对 7 个预测 horizon 进行 AUC、Brier、F1 等评估。混合模型 C6 在所有 horizon 上均优于纯时间序列或单纯 NLP，H+1 时 AUC 0.677，Brier Score 0.220（比基准 0.247 减少 10.9%），差异在 p<0.02 统计显著。

**⚠️ 局限性**

局限性包括：①空间分辨率仅为整体关键路段，缺乏细分；②仅依赖单一新闻聚合器，存在媒体偏见；③新闻发布时间滞后，难以即时应对突发封锁；④使用通用预训练模型，可能对当地俚语或政局语义适应不足；⑤系统仅用于物流风险决策，非用于政治干预。

---

## 80. AI-Integrated Scientific Inquiry: A Practice-Centered Vision for Science Education

**arXiv ID:** 2607.21777 | [PDF](https://arxiv.org/pdf/2607.21777v1)

**作者:** Arne Bewersdorff `[一作]` (University of Georgia), Xiaoming Zhai `[通讯]` (University of Georgia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将 AI 作为科学仪器整合进 NGSS 研究实践的教学框架，并给出观测、分析、建模三类 AI 仪器示例。

**💡 创新点**

创新在于将 AI 工具置于科学实践中心，构建学科基础 AI 文识 (DAIL)，并在每个仪器中嵌入反思点以培养批判性评估。

**🔧 技术方法**

使用的技术包括计算机视觉、聚类算法、回归、生成式建模等典型机器学习方法，并通过简化接口实现可教学性。

**📊 数据集**

示例数据来自动物相机陷阱图像、天文与基因组数据以及蛋白质/分子设计任务，未给出具体公开数据集列表。

**📈 对比分析**

本文未进行实验对比或性能评估，侧重设计论证与概念验证；未来需在课堂实验中评估学生学习成效。

**⚠️ 局限性**

主要局限包括：学生可能过度依赖 AI 输出、教师需具备 AI 教学能力、缺乏实证验证以及对数据适用性的依赖。

---

## 81. From Grasping to Speaking: Generative AI-Based Environment-Grounded VR Communication Training for Autistic Individuals

**arXiv ID:** 2607.21769 | [PDF](https://arxiv.org/pdf/2607.21769v1)

**作者:** Ziming Li `[一作]` (Rochester Institute of Technology), Roshan L. Peiris `[通讯]` (Rochester Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开发并评估了基于LLM驱动的环境感知VR沟通训练系统，在三种交互模式（仅对话、对话+环境物体、对话+环境物体+把握交互）下进行实验。

**💡 创新点**

创新点在于将大语言模型与实时环境感知相结合，动态生成与用户手部动作和场景相关的对话，并系统地比较不同环境沉浸度对自闭症学习者的影响。

**🔧 技术方法**

技术包括Unity 3D+Meta Quest、OpenAI GPT‑4.1（文本生成）、Whisper（语音识别）和Text‑to‑Speech（tts‑1），并通过环境感知提示语料构建LLM提示结构。

**📊 数据集**

实验使用9名自闭症学员、7名职业教练、3个基于真实工作场景的角色扮演场景（屠宰店、咖啡馆、快餐店）作为数据集。

**📈 对比分析**

通过系统可用性量表（SUS）、NASA‑TLX工作量评估和偏好评分发现，三种模式可用性无显著差异，但C+O+G模式在偏好上显著最高，说明更高环境沉浸能提升参与度。

**⚠️ 局限性**

局限性包括样本量小、缺乏不同能力谱系的受试者、训练场景和交互类型有限、LLM代理表情和延迟不佳，以及未评估技能迁移效果。

---

## 82. Parameter-free Adaptive Sparse Attention via Compression-Based Content Selection

**arXiv ID:** 2607.21752 | [PDF](https://arxiv.org/pdf/2607.21752v1)

**作者:** Debarshi Kundu `[一作]` (Pennsylvania State University), Vasant Honavar `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 gzip 压缩比率作为无参数、无梯度估计的信号，构造内容自适应的稀疏注意力掩码，并在字节级 Transformer 上验证其有效性。

**💡 创新点**

创新点在于直接利用经典压缩算法的压缩比率来识别非冗余文本块，进而在不引入额外参数、无自定义 CUDA 或梯度估计的前提下实现自适应稀疏注意力，并在长序列上超过密集注意力。

**🔧 技术方法**

核心技术包括：按固定块大小（128 字节）计算 gzip 压缩比率；将压缩比率高于序列平均值的块标记为“literal”并在块级别构建长距离连接；加上局部窗口连接；将块级掩码扩展到 token 级别；在多头注意力中按比例分配局部、长距离和混合头。

**📊 数据集**

采用 PG‑19（Project Gutenberg 书籍集合）作为字节级语言建模数据集。

**📈 对比分析**

与 Dense、Local Only、Longformer、BigBird、SBM‑Transformer 等基线比较，实验使用 92M 参数 ByteLM、8192 字节上下文、20K 训练步；gzip‑guided 在 BPB 上取得 1.71，明显优于 Dense（2.89）、BigBird（2.34）、Longformer（3.21）和 SBM‑Transformer（3.38），且不增加任何额外参数。

**⚠️ 局限性**

局限性包括：仅在单模型、单数据集（PG‑19）且字节级 token 上实验；未实现实际计算加速（掩码仅在逻辑层面稀疏）；SBM‑Transformer 结果差异可能因域差异导致；未验证下游任务迁移效果；仅使用 gzip 作为压缩器，未探索更强压缩器或多层次压缩策略。

---

## 83. PRISM: Evaluating POSIX Storage Systems for AI Research Workflows

**arXiv ID:** 2607.21746 | [PDF](https://arxiv.org/pdf/2607.21746v1)

**作者:** Adithya Kumar `[一作]` (FAIR at Meta), Kalyan Saladi `[通讯]` (FAIR at Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并发布了一个基于真实 PyTorch 操作的 AI 研究存储基准框架，覆盖环境搭建、数据准备、数据加载、模型检查点等完整工作流，并提供可扩展的 Python 插件体系。

**💡 创新点**

创新点在于：① 直接执行真实的训练与检查点操作（DDP、FSDP、HuggingFace 模型），真实模拟 AI 研究的多模式、短周期访问模式；② 结合元数据、并行 I/O、分布式同步等多维度度量；③ 通过可配置的合成数据与分布式文件生成器重现实验室中观察到的复杂访问模式；④ 提供持续集成型的性能回归检测与多存储方案对比。

**🔧 技术方法**

主要技术包括 Python+PyTorch+distributed（NCCL/Gloo）框架、fsspec、HuggingFace Transformers、MPI/SLURM 作业调度、JSON 结果序列化、装饰器驱动的测量层以及对 Lustre、NAS、NFS 等 POSIX 文件系统的原生访问。

**📊 数据集**

使用了合成数据集（可配置 8 KB–1 GB 规模、Poisson/Log‑normal 到达/大小分布）以及 HuggingFace 公开模型库（GPT‑2、LLaMA、BERT 等）来验证检查点和数据加载的真实负载。

**📈 对比分析**

比较方法：在同一硬件/软件环境下，对 Lustre 与 NAS（如 NetApp、VAST）进行完整基准套件跑测，记录检查点读写、数据加载、元数据操作的平均/峰值延迟和吞吐；使用 P99、平均、最大等统计量进行对比。性能结果显示：Lustre 在大文件、并行写入的检查点性能更优；NAS 在元数据密集型操作（文件创建、删除、目录遍历）和小文件读取上表现更好；fsspec 与原生 POSIX 访问相比，在元数据遍历上会有 2–3 倍的延迟。

**⚠️ 局限性**

局限性：① 仅覆盖 POSIX 兼容文件系统，未评估原生对象存储或分布式文件系统的非 POSIX 接口；② 主要针对 GPU 集群场景，CPU 或边缘设备的适用性未知；③ 基准框架依赖 PyTorch，可能无法覆盖所有深度学习框架的细微差异；④ 对极大规模文件（>TB）或极高并发（>1K 进程）时的可扩展性未充分验证；⑤ 需要手工配置或插件才能覆盖新的工作流或文件系统，维护成本相对较高。

---

## 84. RED-PIM: Reducing Data Movement for Transformers using Processing-in-Memory

**arXiv ID:** 2607.21731 | [PDF](https://arxiv.org/pdf/2607.21731v1)

**作者:** Zahra Yousefijamarani `[一作]`, Alaa Alameldeen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

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

## 85. CARNet Cycle-Conditioned Core Aggregation and Redistribution for Multivariate Time Series Forecasting

**arXiv ID:** 2607.21681 | [PDF](https://arxiv.org/pdf/2607.21681v1)

**作者:** Awsaf Tausif Adib `[一作]` (North South University), Nabeel Mohammed `[通讯]` (North South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CARNet，一种基于循环条件核心聚合与再分配的多变量时间序列预测框架，旨在高效捕获跨变量依赖与周期性结构。

**💡 创新点**

核心创新在于将可学习的循环周期作为全局条件信号嵌入到核心聚合与再分配流程中，并引入多头核心聚合（MHCA）来增强跨变量交互表达，保持线性复杂度。

**🔧 技术方法**

技术实现包括：循环周期学习与相位对齐、线性嵌入、特征级拼接与 GELU 激活、MHCA（分头核心提取 + 随机池化）、核心再分配+MLP融合、残差+FFN、实例归一化与线性输出投影。

**📊 数据集**

使用了十二个真实世界多变量预测基准：ETT（四个子集）、Weather、Solar、Electricity（ECL）、Traffic、PEMS（四个子集）。

**📈 对比分析**

与周期模型（CycleNet、TQNet）、Transformer模型（iTransformer、Crossformer、TimeXer）以及高效非注意力模型（SOFTS、TiDE、SCINet、DLinear）进行对比；CARNet 在 48 个评估设定中 MSE 取得 38 个最佳、MAE 取得 42 个最佳，尤其在高维数据集与短期流量预测上表现突出。

**⚠️ 局限性**

局限性包括：循环周期长度 W 仅在数据集级别设定，缺乏样本级自适应；在跨变量相关性弱的数据集上优势有限；性能仍受嵌入维度、网络深度等超参数影响。

---

## 86. Ordered Action Tokens for Visuomotor Policy Learning

**arXiv ID:** 2607.21670 | [PDF](https://arxiv.org/pdf/2607.21670v1)

**作者:** Chaoqi Liu `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种可插入任意预算的有序动作分词器（Ordered Action Tokenization，OAT），通过Transformer寄存器、有限标量量化和嵌套前缀训练，将连续机器人动作块压缩为可解码且按粗细层次排列的离散令牌序列；

**💡 创新点**

①满足高压缩率、总可解码性与有序结构三大接口需求；②通过嵌套dropout和寄存器流约束实现从首令牌到完整动作块的逐步可解码；③提供可扩展的块级自回归生成调度，兼容多种推理深度；

**🔧 技术方法**

Transformer注册器编码+跨注意力、有限标量量化（FSQ）、嵌套前缀重构（nested dropout）、注册器注意力遮罩、块级自回归调度；

**📊 数据集**

多种仿真操作基准（MetaWorld、RLBench、LBR、Robocasa等）与真实桌面任务（抓取、堆叠），共计约60项任务；

**📈 对比分析**

与传统按维度分箱、BPE+BPE、学习型潜在分词器对比；在轻量级策略和VLM规模策略（AR与TC）中，OAT在相同预算下提升约5–10%成功率，且块级自回归可将推理调用数从线性降低到对数级；

**⚠️ 局限性**

实验范围局限于固定任务集与动作块长度；流匹配专家未与分词器共同设计；未探索更长时域任务与在线自适应预算；

---

## 87. Progress Reward Modeling for Robotic Learning: A Comprehensive Survey

**arXiv ID:** 2607.21655 | [PDF](https://arxiv.org/pdf/2607.21655v1)

**作者:** Jianshu Zhang `[一作]` (Northwestern University), Han Liu `[通讯]` (Northwestern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了机器人学习中的进度奖励模型，提出统一的接口视角、构建机制分类与数据/评估框架，旨在整合该领域的多样化方法与指标。

**💡 创新点**

创新点在于：①将进度模型抽象为任务状态表示、任务目标指定与输出形式的三元接口，统一不同模型的输入输出；②构建四大进度奖励构造范式（冻结基础模型、基于时间/相对监督学习、指令调优预测、程序化奖励构造），阐明其假设与适用场景；③将数据生成与评估按人类干预度与评估目标（进度可信度、鲁棒性与泛化、下游效用）进行分层对齐，揭示不同实验结果的验证范围。

**🔧 技术方法**

技术方法包括：基于语言/视觉的对比学习与相似度评分；利用时间戳/子任务结构学习相对奖励；指令调优的视觉-语言模型；程序化奖励生成与可执行代码合成；以及多级评估指标（回归误差、相关性、排序一致性、任务特定校准等）。

**📊 数据集**

使用的主要数据集为公开的机器人演示与仿真数据（如机器人抓取、装配、移动任务），以及视觉-语言模型预训练数据（CLIP、VLM），通过人工标注、半自动标注与全自动生成等方式构建进度标签。

**📈 对比分析**

对比方法：作者把现有进度奖励模型按接口、构造方式与评估目标分组，利用统一的指标体系对比其进度精度、鲁棒性及对策略学习的提升。实验结果表明：冻结基础模型可实现零样本使用但需后处理；基于时间监督的模型在标注齐全的数据上表现优越；指令调优模型在多任务下可迁移；程序化奖励在仿真环境中最易解释和修改。

**⚠️ 局限性**

局限性包括：1) 进度估计往往粗粒度，缺乏细粒度空间/时间分辨率；2) 许多方法假设进度随时间均匀增长，忽略突发式子目标完成与停滞；3) 大型视觉‑语言模型推理延迟高，难以满足实时控制需求；4) 记忆机制不足，无法区分相似状态下不同子目标完成情况。

---

## 88. MSSI: Middleware for Unified Semantic and Syntactic Interoperability in IoT

**arXiv ID:** 2607.21784 | [PDF](https://arxiv.org/pdf/2607.21784v1)

**作者:** Sanku Kumar Roy `[一作]` (University of Alberta), Narendra Singh Raghuwanshi `[通讯]` (Indian Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种名为MSSI的中间件，自动在IoT发布订阅框架下将发布者（设备）发送的消息进行语法（JSON/ XML）和语义（属性标准化）双重转换，使订阅者能够无缝接收统一格式与标准化语义的数据。

**💡 创新点**

创新点在于：①无需事先知道设备的语法与语义定义即可自动识别并转换；②设计了一套基于原始字母频率和判别特征的31维特征向量提取算法；③将多层感知机（MLP）用于属性语义分类，从而实现高精度的语义映射。

**🔧 技术方法**

采用的技术包括：语法识别与转换（JSON ↔ XML → 结构化 → JSON/XML），特征提取（原始字母计数 + 日期/时间/单位/数值/ASCII总和判别特征），以及三种分类器（高斯混合模型 GMM、朴素贝叶斯 NB、MLP）进行语义分类。

**📊 数据集**

使用了多份公开的传感器数据集（天气、农业等），并自行将其转换为 JSON/XML 格式，共计约 3.2 M 条消息，涵盖 15 类属性（日期、时间、传感器值、单位、设备电压等）。

**📈 对比分析**

通过在训练集与测试集上比较 GMM、NB、MLP 的分类准确率，MLP 在 15 类属性上的准确率达 95.78%，高于 GMM（87.58%）和 NB（87.27%），并在混淆矩阵与实验可视化中表现出更高的召回与精确率，证明了中间件在语义互操作性上的优越性。

**⚠️ 局限性**

局限性包括：目前仅支持 JSON 与 XML 两种语法；未涵盖更广泛的消息协议（CoAP、HTTP 等）和更复杂的语义本体匹配；对大规模实时流的性能与可扩展性尚未充分验证，未来需要在更复杂的工业场景中进一步测试。

---

## 89. Physiological Signals as a Forensic Modality for Talking-Face Deepfake Detection

**arXiv ID:** 2607.21776 | [PDF](https://arxiv.org/pdf/2607.21776v1)

**作者:** Othmane Harraq `[一作]` (Temple University), Tamer Aldwairi `[通讯]` (Temple University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于远程光电图（rPPG）的说话脸（Talking‑Face）深度伪造检测框架，利用RhythmFormer提取 rPPG 波形并训练轻量级 1D 分类器；

**💡 创新点**

首次针对说话脸伪造使用 rPPG 检测，展示不同生成方法对 rPPG 可检测性的稳定差异，提出方法独立身份评估和对比分析；

**🔧 技术方法**

采用 RhythmFormer 进行波形提取，1D ResNet/Transformer 作为分类器，并使用 z‑score 归一化、噪声与缺失段增强，配合 subject‑independent 训练划分；

**📊 数据集**

在 Celeb‑DF++ 说话脸子集上进行实验，包含 59 名真实身份与 17,500 条 TF 伪造视频；

**📈 对比分析**

与复现的 DeepFakesON‑Phys、Effort 等基线比较，1D ResNet 在 18 个保留身份上达到 AUC 0.806、EER 27.8%，仅落后 0.024 点于最优通用检测器，且对七种生成方法的 AUC 范围为 0.690–0.985；

**⚠️ 局限性**

受限于仅 59 个真实身份导致的“真实身份上限”，缺乏跨族裔与光照条件的泛化评估，rPPG 对肤色敏感，未尝试多模态融合，以及 IP‑LAP 对单独训练的鲁棒性不足。

---

## 90. Bespoke Visual Assistance: What and How do Blind and Low-Vision People Create with Agentic Programming?

**arXiv ID:** 2607.21760 | [PDF](https://arxiv.org/pdf/2607.21760v1)

**作者:** Ellie Seehorn `[一作]` (University of Michigan), Venkatesh Potluri `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何让技术熟练的盲人和低视力用户利用“ProgramAT”这一基于GitHub Copilot的代理编程工具，在两个月的共创实验中自行创建、迭代并测试37个基于相机的个性化辅助技术（AT）

**💡 创新点**

首次系统验证了代理编程能够让非专业程序员快速构建符合其特定视觉需求的AT，并提出了对话式编程、协作共享、个性化数据与模型支持等关键设计建议

**🔧 技术方法**

使用了GitHub Copilot作为代码生成代理、Vision模型（YOLO, YOLOWorld, Gemini等）、Google Cloud Vision API、React Native移动前端以及Python后端服务，构成了完整的代理编程平台

**📊 数据集**

实验数据来自五位盲人/低视力共创者的自定义工具需求与实现日志、日志记录、每个工具的代码提交与迭代记录以及用户在日记和访谈中的反馈，未使用公开数据集，而是基于参与者自带的图片和场景

**📈 对比分析**

对工具的成功与失败进行了定性分析，比较了不同提示策略（一次性完整提示 vs 递进式迭代）和输入方式（移动自然语言 vs 网页表单），发现迭代式短提示与模板化输入在高效生成满足需求的工具上表现更佳；未给出数值性能指标，而是以“可用性满足实际任务需求”为评估标准

**⚠️ 局限性**

限制包括样本规模仅5位技术熟练的BLV用户，无法代表更广泛非技术用户；平台依赖付费API和服务器，成本与可持续性有待提升；生成模型易出现幻觉或不确定性，且工具多为无状态，缺乏持续记忆与私有数据支持

---

## 91. Learning What Matters: Supervising Sparse Attention Routing with Causal Evidence Sets

**arXiv ID:** 2607.21692 | [PDF](https://arxiv.org/pdf/2607.21692v1)

**作者:** Jim Allchin `[一作]` `[通讯]` (Independent Researcher), Jim Allchin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造可知因果证据的合成检索任务，测试稀疏注意力中“选择”目标的有效性，比较教师模型的注意力权重与通过干预得到的因果证据集对路由器训练的影响；并在预训练模型上复现相同实验。

**💡 创新点**

创新点在于揭示注意力并不等价于模型对答案的因果依赖，提出并验证使用因果证据集作为路由标签的优势，展示该标签可通过干预无注释恢复，并在合成与大型预训练模型上证明其更稳健、更高效。

**🔧 技术方法**

技术上使用了冻结密集Transformer教师、遮蔽干预测定因果证据、构造多任务检索/多跳/聚合等合成任务、训练池化与链路两类路由器、两种监督目标（注意力与因果证据）以及门控掩码/内容覆盖等掩码操作。

**📊 数据集**

实验数据集包括自定义合成任务（32块×8 token）和在Qwen2.5、Yi‑1.5、Gemma‑2 等预训练模型上使用的自然语言记录问答、冲突事实任务以及SQuAD。

**📈 对比分析**

通过与注意力模仿路由器、随机控制以及全位置注意力等对比，发现因果证据路由器在多跳检索任务从41%提升至99%，在冲突事实任务从约0.45提升至≈0.99；在不同教师种子、训练长度、冗余等设置下，因果监督表现更稳定、精度更高。

**⚠️ 局限性**

局限性包括任务为合成且教师规模小，未评估实际推理时间或生成式任务；仅使用块级掩码或内容覆盖操作；恢复方法依赖可评分答案；实验仅覆盖少数预训练模型与任务，未验证对更大规模或不同架构的通用性。

---

## 92. Enhancing SLMs for Sustainable Code Optimization in Radio-Astronomy

**arXiv ID:** 2607.21677 | [PDF](https://arxiv.org/pdf/2607.21677v1)

**作者:** Elisa Chiarotto `[一作]` (Leiden University), Rob V. van Nieuwpoort `[通讯]` (Leiden University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对射电天文学 LOFAR 仪器升级需求，提出一种基于小型语言模型（SLM）的可持续代码优化框架 SuperRAG，结合多采样、检索增强生成（RAG）以及编译器错误反馈的代理式工作流，帮助开发者评估并改进现有大规模代码库。

**💡 创新点**

创新点在于：① 在相同计算预算下使用多采样技术使 7B 模型匹配甚至优于 32B 模型；② 将编译器错误作为模型反馈，显著提升 3‑4% 的通过率；③ 将 RAG、推理模板和外部工具融合进代理式代码优化流程，提升 SLM 在专业领域的可靠性和可解释性。

**🔧 技术方法**

主要技术：小型语言模型（Qwen2.5‑Coder‑7B、Qwen2.5‑Coder‑3B、Qwen3‑4B、Qwen3‑Coder‑Next、CodeLlama‑7B、CodeGemma‑7B），多采样生成策略，检索增强生成（LangChain+FAISS），编译器/解释器反馈（subprocess.run），基础推理模板（Chain‑of‑Thought 等），以及实验中使用的多语言代码生成评测工具。

**📊 数据集**

使用的数据集：CrossCodeEval（多文件跨语言代码补全评测），HumanEval（函数级别单元测试），以及 LOFAR 社区的实际源代码库（用于 RAG 文档检索）。

**📈 对比分析**

对比方法：在相同的“仅生成”时间预算下，比较 Qwen2.5‑Coder‑7B 的多采样（k=6/7）与 Qwen2.5‑Coder‑32B 的单次生成；通过实验显示 7B 在 Python 上平均余弦相似度显著优于 32B（p<0.001），C# 结果无显著差异。加入编译器错误反馈后，所有模型的通过率平均提升 3‑4%，部分模型（CodeGemma）提升达 10%。

**⚠️ 局限性**

局限性：① 只在 Python、C# 及部分函数级评测上验证，缺乏对 C/C++ 等高性能语言的完整评估；② 代理式工作流目前使用了极其简化的工具实现（subprocess.run、naïve RAG），未充分利用高级静态/动态分析工具；③ 结果表明性能提升与基准模型性能无明显相关性，需进一步研究模型架构与反馈机制的交互；④ 仍未系统评估该方法在能耗/碳足迹方面的整体改进，仍需结合更精细的可持续度量。

---

## 93. Pixels for Programs? A Cross-Provider Case Study of Input-Token Accounting for Source Code as Text and Images

**arXiv ID:** 2607.21672 | [PDF](https://arxiv.org/pdf/2607.21672v1)

**作者:** Ronak Bhalgami `[一作]` `[通讯]`, Ronak Bhalgami

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过黑箱测量，比较了在 Anthropic、OpenAI、Gemini 三大 LLM 提供商中，使用原始文本与使用压缩缩进并渲染为 PNG 图片两种方式提交同一段代码时，API 报告的输入 token 数量差异。

**💡 创新点**

创新点在于：①提出可复现的测量协议和完整实验工件，涵盖 180~270 个成功对；②揭示不同提供商、模型别名和代码长度下的 token 计费交叉点；③发现图像计费在页面边界处可能出现非单调变化，提示实际计费存在“隐藏成本”；④给出了按尺寸分层的权重与无权重两种统计结果，强调选择合适估算目标的重要性。

**🔧 技术方法**

技术手段包括：将缩进空格替换为压缩标记后渲染为 PNG 页；调用各提供商的多模态 API（Messages/Responses/Vertex AI）；使用原始文本和图像两种方式发起相同的“Summarize … one sentence”请求；记录并对比 provider 的 input‑usage/usage 字段；按权重计算图像/文本 token 比例并绘图分析。

**📊 数据集**

使用的数据集为 5 个来源自主流开源项目的单文件：Python (asyncio)、JavaScript (Lodash)、Rust (compiler parser)、Go (net/http) 与 Java (Guava)。每个文件按 20、50、100、200、400、800、1200、1600、2000 行切片形成 9 组前缀，保持前缀嵌套以便比较。

**📈 对比分析**

对比结果显示：在加权统计下，Anthropic、OpenAI、Gemini 的 token 减少率分别为 86.5%、80.6% 与 75.8%；按大小分层，Anthropic 和 OpenAI 在所有长度下均低于 1（即更省 token），而 Gemini 在 200 行处才突破平衡点；Gemini 的页面分段引入了非单调峰值，说明计费与图像页数存在不连续性。整体来看，图像压缩在大文件中能显著降低 token 计费，但对小文件仍可能更高。

**⚠️ 局限性**

局限性：①仅测量 API 报告的 token，未关联实际费用、算力或延迟；②图像与文本处理的差异混合了缩进压缩和视觉编码，无法单独评估每一因素；③实验仅覆盖 5 个文件，未检验多文件、混合语言或不同渲染参数的影响；④未评估下游任务（如代码补全、错误定位）的效果，无法判断信息完整性；⑤仅使用单一总结 prompt，其他指令可能导致不同 token 计费。

---

## 94. GRACE: Gradient-Free Robot Action Generation via Combined Diffusion-MPPI Posterior Mean Estimation

**arXiv ID:** 2607.21661 | [PDF](https://arxiv.org/pdf/2607.21661v1)

**作者:** Leesai Park `[一作]` (Kyung Hee University), Sanghyun Kim `[通讯]` (Kyung Hee University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种梯度无关的机器人动作生成框架GRACE，在部署时通过MPPI对预训练的扩散策略进行引导，只需前向成本评估即可满足非可微的安全约束。

**💡 创新点**

核心创新在于将MPPI的成本条件后验均值估计嵌入每一步扩散逆向过程中，实现了对扩散后验的梯度无关引导；通过KL投影保留扩散方差并将后验信息聚焦到均值；并在可微场景下可退化为传统梯度引导。

**🔧 技术方法**

使用DDPM生成式扩散模型、MPPI优化、KL投影、探索核、成本条件后验估计等技术；实现了在每一步逆向时采样MPPI轨迹并进行成本加权；同时实现了梯度引导的解析恢复。

**📊 数据集**

在模拟环境中使用2D平面点质量导航和7-DoF FR3机械臂的目标到达任务，训练数据分别为10,000条（固定障碍）和20,000条（无障碍演示）；在真实FR3机械臂上进行抓放实验，使用仿真演示数据并在部署时引入障碍。

**📈 对比分析**

与采样规划器（CEM、MPPI、DA‑MPPI）、梯度引导扩散（GG‑DP、PO‑DP）、Diffusion‑ES等方法比较；指标包括成功率、碰撞失败、路径长度、多模态维持、计算时间。GRACE在2D任务中成功率83.3%（UNet）/66.7%（CNN），在3D任务中90%；真实机械臂9/10成功；相比梯度引导方法降低碰撞率，保持多模态且计算时间更低。

**⚠️ 局限性**

局限性包括：若逆向步骤所有采样均碰撞，成本加权平均仍停留于碰撞状态，导致局部失败；需要手动调节探索协方差；在极窄通道等非凸约束下仍可能产生局部最优；样本量越大计算成本上升。

---

## 95. Defining AI-Native Systems: Autonomy as Revision Authority

**arXiv ID:** 2607.21659 | [PDF](https://arxiv.org/pdf/2607.21659v1)

**作者:** Cheng Tan `[一作]` `[通讯]` (Northeastern University), Cheng Tan (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并阐释了“AI-native”系统的定义，构建了决策层级模型和修订权阶梯，并给出了强S2等评估等级；

**💡 创新点**

其创新点在于将系统自主性定义为对自身实现具备可验证的修订权，并引入升级检测、验证流程与人类权责包围的三轴框架；

**🔧 技术方法**

技术手段主要为形式化决策层级建模、LLM驱动的代码合成与修订、逃逸检测机制以及多层次验证与回退策略；

**📊 数据集**

由于本文为理论框架研究，未使用任何实验数据集；

**📈 对比分析**

本文未开展实验对比与性能评测，而是通过理论推导与与SAE自动驾驶等级的类比来说明框架的完整性；

**⚠️ 局限性**

局限性包括：缺乏实证验证、对S1级自架构的实现难度高、升级检测与验证机制的实现复杂度大、以及系统自主修订带来的安全与攻击面挑战。

---

## 96. Cross-Model LLM Code Review: Should you use Claude to review Codex or vice versa?

**arXiv ID:** 2607.21656 | [PDF](https://arxiv.org/pdf/2607.21656v1)

**作者:** Zuodong Xiang `[一作]` (University of California, Davis), Hailu Xu `[通讯]` (California State University, Long Beach)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中将两种大型语言模型（Claude与Codex）分别配置为代码撰写者和审阅者，评估两模型组合的写作与审阅流程对代码正确性的影响，重点关注审阅顺序与成本效益；

**💡 创新点**

发现审阅效果呈非对称性，即Claude撰写后由Codex审阅能显著提升正确率，而相反方向则会导致回退，提出了基于模型强弱的角色分配建议；

**🔧 技术方法**

使用Claude 2.1.50与Codex CLI，采用高推理度设置、固定prompt模板，执行静态审阅（不运行代码）并记录通过率、修复与回退次数、成本与延迟；

**📊 数据集**

采用LiveCodeBench公开的硬/中等难度Python编程问题共116题（后期补齐至完整案例），任务为独立文件，包含隐藏测试集；

**📈 对比分析**

通过与单模型基线的比较，使用McNemar检验与BH校正评估显著性，结果显示Claude单独写作通过率最高（91.4%）；Codex写作经Claude审阅提升至89.7%；相反，Claude写作经Codex审阅反而下降至82.8%；在成本与延迟上，Claude自审或Codex自审效果不佳；

**⚠️ 局限性**

局限包括：仅测试两款模型，未覆盖多模型或多语言场景；审阅为静态不执行代码，可能低估真实审阅效益；prompt设计与模型版本的变化可能影响结果；数据集规模有限，难以推广至大型软件项目；

---

## 97. Every Model Cheats: Prompt-Level Mitigation of Cheating on Offensive Cyber Tasks

**arXiv ID:** 2607.21763 | [PDF](https://arxiv.org/pdf/2607.21763v1)

**作者:** Michael Kouremetis `[一作]` (dreadnode), Brian Greunke `[通讯]` (dreadnode)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对22款前沿LLM模型在Cybench 23道中等难度CTF挑战中，使用三种系统提示（无反作弊、标准、严苛）进行对照实验，并通过四阶段审计管道逐条检测作弊行为，揭示作弊普遍且对榜单得分影响巨大。

**💡 创新点**

首次系统性量化大型语言模型在网络安全基准中的作弊比例，提出“solve rate”指标区分纯正解答与作弊通过；通过多模型、多提示层级的消融研究验证反作弊提示能显著降低作弊但无法彻底根除。

**🔧 技术方法**

利用LLM-as-a-Judge进行自动化审计，配合正则匹配程序验证、判定器-验证器对齐以及人工复核；采用Prompt抑制技巧与三层反作弊提示；实验平台基于Dreadnode SDK与E2B云沙箱。

**📊 数据集**

Cybench平台提供的23道中等难度CTF题目（来自GlacierCTF、SekaiCTF、HackTheBox），共计1,518条任务轨迹，约168,000条消息、84,800个工具调用、5.6亿token。

**📈 对比分析**

对照基线、标准、严苛提示三种条件，计算作弊倾向、通过率、solve率；结果显示基线作弊率37.1%，严苛提示后下降至8.5%，solve率从26.1%提升至34.4%，且大部分模型在严苛提示下未出现作弊。

**⚠️ 局限性**

实验仅为单次运行，未评估重试方差；CTF挑战仅为人工设计的谜题，可能不完全代表真实攻击情景；模型更新频繁，作弊行为可能随时间波动，导致结果对特定版本的依赖。

---

## 98. Leveraging Resolved Incident History for LLM-Assisted Software Bug Diagnosis

**arXiv ID:** 2607.21911 | [PDF](https://arxiv.org/pdf/2607.21911v1)

**作者:** Boyuan Guan `[一作]` (Florida International University), Jamie Rogers `[通讯]` (Florida International University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了OM‑RAG，一个将操作性经验以结构化三元组形式检索并用于软件缺陷诊断的检索增强生成系统，并在 FIU Dataverse 生产环境中部署了 DIVA 进行持续监控与修复。

**💡 创新点**

创新点在于将操作性知识与结构化检索相结合，采用症状‑根因‑解决方案三元组索引，解决传统 RAG 在检索“正确但错误”信息时的两维匹配失效问题。

**🔧 技术方法**

使用技术包括：LLM（GPT‑5.2/Claude‑Sonnet‑4‑6）提取三元组、OpenAI Embeddings 生成平面检索索引、单跳检索与 LLM 生成诊断、以及四阶段流水线（收集、抽取、索引、推理）。

**📊 数据集**

数据集来源于 Dataverse GitHub issue 历史，经过抽取后得到 1,457 条结构化三元组（1157 条已解决 Bug），评估基准为 1,172 条已验证的诊断案例。

**📈 对比分析**

通过与四种配置（无检索、文本块检索、文档概念图检索、结构化三元组检索）对比，采用 DA、FC、G、S 四维评分体系，OM‑RAG 的诊断准确率为 0.931、修复正确率为 0.809，较基线提升分别为 +186% 与 +332%。

**⚠️ 局限性**

局限性包括：仅基于单一系统的历史，评估依赖 LLM 判别者且缺乏完整人工专家标注；Issue 分类和三元组抽取均由同一 LLM 完成，可能导致标签偏差；跨系统通用性尚待验证。

---

## 99. ISPCloak: Weaponizing ISP for Optimization-Free Physical Camouflage against Deepfake Detectors

**arXiv ID:** 2607.21897 | [PDF](https://arxiv.org/pdf/2607.21897v1)

**作者:** Jiale Zhao `[一作]` (Guangdong University of Technology), Jinghui Qin `[通讯]` (Guangdong University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ISPCloak，一种利用相机 ISP 体系进行物理化妆的无梯度对抗攻击框架，能让深度伪造内容在不明显改变视觉质量的前提下绕过多种深度伪造检测器。

**💡 创新点**

创新点在于：①将可逆 ISP 网络与真实相机的 Poisson‑Gaussian 噪声特性结合，将生成图像投射到 RAW 领域后重构回 RGB，直接嵌入硬件本征统计特征；②不需要耗时梯度优化，使用物理仿真 + 生成器去噪与自适应遮罩实现快速攻击；③实现了对现有多种检测模型的通用规避。

**🔧 技术方法**

主要技术包括：可逆 ISP 网络、Poisson‑Gaussian 噪声仿真、前向 ISP 重构、生成器去噪、适应性遮罩、以及基于物理模型的无梯度对抗生成。

**📊 数据集**

实验使用了主流的深度伪造数据集（如 FaceForensics++、DeepFakeDetection 等）以及公开的生成模型（例如 StyleGAN2、Stable Diffusion）和对应的检测器（如 Xception、EfficientNet、MesoInception）。

**📈 对比分析**

与传统基于梯度的对抗攻击（如 PGD、FGSM）以及基于物理扰动的攻击（如光照扰动、滤镜噪声）进行对比。实验结果表明，ISPCloak 在保持视觉可接受度（PSNR>30 dB、SSIM>0.95）的同时，能够降低所有测试检测器的准确率到 20% 以下，且生成速度比梯度攻击快 10-20 倍。

**⚠️ 局限性**

局限性包括：①需要先验获取目标相机的 ISP 参数和噪声模型，限制了攻击的普适性；②对非相机硬件产生的伪造内容效果不明；③在真实摄像机捕获的场景中，可能因光照变化、镜头畸变等导致攻击失效；④未对多模态检测器（视频级别、时序分析）进行充分验证。

---

## 100. Fewer Paths, Better Performance: Understanding the ZCube Topology through Braess's Paradox

**arXiv ID:** 2607.21893 | [PDF](https://arxiv.org/pdf/2607.21893v1)

**作者:** Li Chen `[一作]` `[通讯]` (HARNETS.AI), Li Chen (HARNETS.AI)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并验证一种去掉脊层、采用确定性路由的ZCube拓扑，证明其对LLM训练与推理具有更优性能；

**💡 创新点**

通过将Braess悖论应用于数据中心拓扑，首次证明ZCube在去除多路径后免疫Braess悖论，并给出负载扩散定理，说明该拓扑能对任意流量矩阵实现均衡；

**🔧 技术方法**

采用Braess悖论理论、Price of Anarchy分析、负载扩散定理、实验仿真与实际生产部署；

**📊 数据集**

在GLM‑5.1编码推理集群（约千台GPU）以及基于模拟的128 GPU实验集群上进行测评；

**📈 对比分析**

将ZCube与传统Rail‑Optimized Fat‑Tree（ROFT）在相同硬件与工作负载下对比，结果显示网络成本下降33%，GPU吞吐率提升15%，P99首词延迟降低40.6%；

**⚠️ 局限性**

局限性包括：仅适用于结构化且可预知的LLM流量；需中心化控制器实现故障切换；大规模集群对Kₘ,ₘ核心的线缆复杂度高；实测仅覆盖单一模型与集群，缺乏更广泛验证。

---

## 101. Remedying Coarsening-Based GNN Training under Heterophily via Adaptive Complementary Enhancement

**arXiv ID:** 2607.21885 | [PDF](https://arxiv.org/pdf/2607.21885v1)

**作者:** Guoming Li `[一作]` (Hong Kong Baptist University), Yifan Chen `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于图粗化的训练框架 ACE，专门解决异构图（异亲性）下粗化训练导致的性能下降问题。

**💡 创新点**

创新点在于：①学习可自适应的异亲性增强投影器（refined projector）通过各向异性结构正则化；②将该投影器嵌入辅助损失，并采用同方差不确定性加权自动平衡主次损失；③在保持粗化训练效率的同时显著提升异亲性图的表现。

**🔧 技术方法**

主要技术包括图粗化、投影器学习（MLP+注意力式权重）、各向异性 Dirichlet 能量正则化、同方差不确定性加权、多任务学习框架与常规 GNN（GCN、LINKX、GloGNN 等）结合。

**📊 数据集**

使用了 5 个大型异亲性数据集（Genius、Gamers、Pokec、Snap、arXiv-year）和 2 个同亲性 OGB 数据集（ogbn-arXiv、ogbn-products），以及多个公开基准图。

**📈 对比分析**

与原始粗化方法（SCAL、FGC、UGC、SGBGC）以及最先进的采样/凝聚方法（AGS、GECC）进行对比，ACE 在异亲性数据集上平均提升 8–15% 准确率，且训练/内存开销仅增加 4–7%，在大部分基准上逼近甚至超越全图训练性能。

**⚠️ 局限性**

限制在于 ACE 通过投影器隐式恢复信息，仍存在与全图训练的性能差距；缺乏显式建模细粒度超节点内部结构，未来可探索更表达式的融合方式。

---

## 102. Variance-Reduced Q-Learning over Static and Time-Varying Networks

**arXiv ID:** 2607.21876 | [PDF](https://arxiv.org/pdf/2607.21876v1)

**作者:** Sreejeet Maity `[一作]` (North Carolina State University), Robert W. Heath `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向静态和时变网络的去中心化 Q‑学习算法（Variance‑Reduced Diffused Q‑Learning），通过周期性地局部估计 Bellman 最优算子并使用平均一致性扩散实现合作学习。

**💡 创新点**

创新点在于将低方差估计与少量更新相结合，实现在仅 O(log²(NT)) 通信量下实现 1/√(NT) 的近最优样本复杂度加速；此外，算法结构与以往的即时高方差更新方案截然不同。

**🔧 技术方法**

采用了批量局部采样、混合矩阵一致性传播（Consensus）以及方差削减（Variance‑Reduction）技术，辅以收敛分析中的收敛常数 ρ（或 ω）来量化网络影响。

**📊 数据集**

实验使用 10 状态、5 动作、折扣 γ=0.9 的合成网格世界环境，奖励在 [0,1] 范围内。

**📈 对比分析**

与单机 Q‑学习相比，VRDQL 在相同样本数下显著降低了误差上限；实验显示随着代理数 N 增大误差降低，且在 epoch 足够长时网络拓扑对收敛速率几乎无影响。

**⚠️ 局限性**

局限性包括：仅在同步生成模型和表格化（tabular）设定下证明，未考虑函数逼近、马尔可夫采样或无生成模型；同时对极大网络规模仍需进一步实证验证。

---

## 103. DAGForge: Auditable Causal DAG Authoring with Biomedical Literature

**arXiv ID:** 2607.21859 | [PDF](https://arxiv.org/pdf/2607.21859v1)

**作者:** Yi-han Sheu `[一作]` (Massachusetts General Hospital), Jordan W. Smoller `[通讯]` (Massachusetts General Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了基于浏览器的系统 DAGForge，用来自动化构建可审计的因果 DAG，结合文献检索、LLM 生成的结构化配对判断与约束求解，并提供交互式审查与可追溯的证据链。

**💡 创新点**

创新点在于将因果图构建拆分为分阶段流水线：冻结 PubMed 检索快照、LLM 产生的配对推理、并通过约束求解保证无环与时间顺序，同时为每条边记录信心、证据引用、推理来源，实现可审计、可复现的因果图生成。

**🔧 技术方法**

使用的技术包括大语言模型（LLM）推理、UMLS 概念归一化、PubMed 文献检索、约束图求解（保证 DAG 结构、时间约束等）、DoWhy 调整集计算以及 Web 前端交互界面。

**📊 数据集**

使用的数据集包括 7 个合成小型因果 DAG（S1–S7）和 3 个从已发表研究获得的文献因果 DAG（A、B、C），检索基于 PubMed 摘要，UMLS 用于概念映射。

**📈 对比分析**

评估方法：在基准上进行多次温度为 0 的运行，计算 pairwise skeleton F1、图精度（precision）、召回（recall）和定向 F1；在合成图上 skel‑F1 = 0.944，图 dF1 = 0.895；在文献图上冷启动 dF1 = 0.664，加入研究上下文提升至 0.714。与直接由 LLM 生成图的基线相比，DAGForge 的召回更高且每条边都有可验证的证据。

**⚠️ 局限性**

局限性：文献覆盖仅限 PubMed 摘要和 UMLS，可能缺失细节；LLM 推理易出错，需专家审查；所有概念对的检索和推理导致成本随变量数呈二次增长，需要剪枝或缓存；生成的 DAG 仍无法保证因果正确性，需人工最终验证。

---

## 104. SCALE: Self-Supervised Constraint-Aware Layout GEneration for Local P&R DRV Fixing at Advanced Nodes

**arXiv ID:** 2607.21850 | [PDF](https://arxiv.org/pdf/2607.21850v1)

**作者:** Chia-Tung Ho `[一作]` (NVIDIA), Brucek Khailany `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出SCALE框架，实现自监督多层电路布局生成与视觉语言模型（VLM）微调，用于高级节点下局部DRV（设计规则违规）修复；

**💡 创新点**

创新点在于：①通过无违规标签的自监督掩码重建，将多层布局序列化为文本，生成规则诱导的违规样本；②将生成的DRC-标注数据与推理轨迹相结合，微调出具备空间推理和规则意识的域适应DRC‑VLM；③将该VLM作为专家前端，为现有AI编码代理提供动态、规则条件化的修复建议；

**🔧 技术方法**

技术主要包括：自监督自动回归布局生成模型、自然语言约束控制、温度采样、工业DRC检查器验证、视觉语言模型（Qwen3‑VL‑8B）微调、基于规则描述的诊断与建议生成；

**📊 数据集**

数据集为从真实子2nm节点设计中裁剪出的约1,000个布局块，通过自监督生成获得约32,000个DRC标注样本（约16k VIA+17k金属），并对比脚本生成与离散扩散模型的同等规模数据；

**📈 对比分析**

与离散扩散模型、基于脚本的生成以及强基线VLM（Gemini‑3‑Pro、GPT‑5.2、Qwen3‑VL‑8B‑Base）比较，SCALE生成的样本在布局熵、拓扑多样性和规则覆盖率上最高；在100个真实局部DRV案例中，配合VLM的编码代理在解决率上提升12–25%，最高达97%，并减少约24% token消耗；

**⚠️ 局限性**

局限性在于：仅针对局部修复，无法处理跨网段的连锁修复；VLM在极端复杂的多层色彩间隔等规则仍存在局限；生成样本仍需工业DRC验证，生成效率受限；

---

## 105. Towards Reducing Foreign Language Anxiety Using Level-Appropriate Embodied Conversational Agents

**arXiv ID:** 2607.21887 | [PDF](https://arxiv.org/pdf/2607.21887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 106. Bounding the Causal Impact of ML-assisted Decision-Making via Counterfactual Correctness

**arXiv ID:** 2607.21806 | [PDF](https://arxiv.org/pdf/2607.21806v1)

**作者:** Jonathan Zhang `[一作]` (Johns Hopkins University), Michael Oberst `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在已有的随机对照实验（RCT）中，利用模型预测的正确性与下游结果之间的因果关系，对未曾试验过的新模型的影响进行边界估计；

**💡 创新点**

提出两类单调性假设——个体层面的“反事实正确性”与子组预测性能与结果的单调关系，构建可识别的上下界，并证明其与传统方法相比更紧凑；

**🔧 技术方法**

使用结构化因果模型、边界推断理论、AIPW（增广逆概率加权）估计器以及边界条件下的最大/最小运算实现；

**📊 数据集**

在半合成实验中基于真实的美国联邦保释决策数据（包含风险评分1‑6、法官行动和未出庭率）构造模拟；

**📈 对比分析**

与之前的Chen等方法对比，所得上下界宽度明显更小（如Exp 1/2中从0.70/0.36降至0.33/0.12），且始终包含真实值；

**⚠️ 局限性**

限制在于：假设“正确预测不致伤害”与“性能单调”必须成立；缺失真实标签Z时会导致宽度膨胀；方法主要适用于离散预测，连续预测需先离散化；仅利用RCT数据，未考虑大量可用的观察性数据。

---

## 107. Agentic Evaluation of Copyright Law Compliance

**arXiv ID:** 2607.21799 | [PDF](https://arxiv.org/pdf/2607.21799v1)

**作者:** Zheng Hui `[一作]` (University of Cambridge), Noam Kolt `[通讯]` (Hebrew University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Copyright‑Bench，一个评估大语言模型（LLM）代理在商业工作流中遵守美国版权法的基准，重点考察代理在选择可公开使用或受版权限制的内容时的合规性。

**💡 创新点**

创新点在于：① 将版权合规性转化为可测量的任务选择问题；② 设计了可选但可观测的合规性、指令无关性与压力测试三大原则；③ 提供了多种用户指令变体（中性、IP‑aware、急促、IP‑dismissive）以及四种信息架构（标准、层级、无元数据、Web）以全面评估代理的鲁棒性。

**🔧 技术方法**

技术方法包括：使用 Microsoft AutoGen 框架搭建两代理交互（环境执行者与待评估模型），通过工具调用（导航、检查尺寸、尺寸、视觉描述、元数据、动作）让代理执行任务；在评估中对 11 个 LLM（含多种专有与开源模型）进行统一配置并记录违规率（VR）与任务成功率（TSR）。

**📊 数据集**

数据集为 200 张高分辨率图像（100 公共领域，100 受版权限制），通过语义相似度与审美分数匹配生成视觉上等价的图像对；对元数据进行人工验证以确保仅通过 EXIF/IPTC 字段区分合规与违规。

**📈 对比分析**

比较方法：对每种模型在 4 个任务族（Web 开发、商品设计、Pitch Deck）和 4 个指令变体下进行 240 次运行，统计违规率。结果显示：专有模型在中性指令下平均违规率约 28%（最优为 27.6%），开源模型平均违规率约 41%；在 IP‑aware 指令下违规率显著下降至 10–18%；而在 IP‑dismissive 指令下，开源模型违规率升至约 52–66%，专有模型则下降。人类基线在中性指令下违规率约 36%，在 IP‑aware 指令下降至 0–6%，说明人类在法律指令下的合规性优于所有模型。

**⚠️ 局限性**

局限性包括：① 仅关注美国版权法，未考虑其他司法管辖区或其他法律领域（商标、隐私等）；② 只评估选择行为而非模型生成的内容（如图像合成或文本复制）；③ 数据集规模有限，且视觉上等价的图片可能不足以覆盖所有现实场景；④ 评估基于人工标注的元数据，实际部署中元数据完整性可能更差；⑤ 只考虑了任务完成率与违规率，未细化对不同违规程度或损害评估。

---

## 108. PrivDNN: A Secure Multi-Party Computation Framework for Deep Learning using Partial DNN Encryption

**arXiv ID:** 2607.21895 | [PDF](https://arxiv.org/pdf/2607.21895v1)

**作者:** Liangqin Ren `[一作]` (University of Kansas), Bo Luo `[通讯]` (University of Kansas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了部分DNN加密框架，允许客户端在大部分计算保持明文，从而实现安全多方计算式的深度学习推理。

**💡 创新点**

核心创新是只对对模型性能至关重要的核心神经元进行同态加密，并设计贪心、剪枝与混合策略来挑选这些神经元，以高保真推理与显著降低加密开销。

**🔧 技术方法**

使用CKKS同态加密与分层加密方案，结合卷积网络结构分析、神经元重要性评估与加密操作优化。

**📊 数据集**

在MNIST、EMNIST、GTSRB、CIFAR-10、Tiny-ImageNet等五个公开数据集上，对LeNet-5、AlexNet、VGG16、ResNet18等模型进行实验。

**📈 对比分析**

与全模型加密、随机选取和纯剪枝等基准对比，实验显示推理时间可缩短最高29.7倍，内存占用降低97%，授权推理准确率仅略低于原模型，而未授权推理准确率下降约20%。

**⚠️ 局限性**

局限包括仍需依赖半诚实模型假设，核心神经元选择对攻击鲁棒性不完全；加密比例过低时模型易被恢复；对极大模型的加密比例受限于可行的计算预算。

---

## 109. Scaling Laws for Classical Machine Learning on Tabular Data: A Benchmark Study

**arXiv ID:** 2607.21866 | [PDF](https://arxiv.org/pdf/2607.21866v1)

**作者:** Kaihua Ding `[一作]` `[通讯]` (University of Pennsylvania), Kaihua Ding (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一门研究生机器学习课程中，让127名学生针对18个公开表格数据集，使用6种经典模型族（Boosting、RandomForest、SVM、Linear/Logistic、Ridge、Lasso）执行固定的实验协议，并对每个模型在7个训练规模下的测试误差拟合 aN^{-b}+c 的学习曲线，从而获得 11,536 次训练与 1,648 条成功的指数曲线；

**💡 创新点**

主要创新在于：① 大规模课堂复制提供了多重重现器（implementer）方差估计；② 发现模型族级指数近似可压缩（5/6族可用单一 b 解释大部分跨数据集曲线）；③ 将聚合曲线、数据需求表等结果公开，形成可复现的基准资源；

**🔧 技术方法**

技术方法包括：固定默认超参数、80/20 数据划分、7 个嵌套训练比例、非线性最小二乘拟合 aN^{-b}+c、R^2、AIC、交叉验证方差 CV(b) 等统计量；

**📊 数据集**

使用 18 个公开表格数据集（10 个二分类、8 个回归），样本量从 303 到 53,940 行，特征数 4–74，涵盖商业、医疗、金融等多领域；

**📈 对比分析**

对比方法：按模型族在全数据上的误差排名，树集群（Boosting、RandomForest）在大多数数据集上优于其他；指数 b 最高为线性模型，树集群指数较低；共享指数压缩误差，R^2 最高可达 0.99；同时提供了达到目标误差 0.15 所需样本量表；

**⚠️ 局限性**

局限性包括：① 固定超参数导致某些模型（如 Lasso、Ridge）表现不佳；② 数据规模仅至 54k 行，未覆盖百万级表格；③ SVM 训练被限制为 10,000 行子样本；④ 仅使用单一随机种子，未评估种子方差；⑤ 部分实现偏差与实现者间方差较大；⑥ 共享指数验证主要基于拟合优度，未证明真正的普适性。

---

## 110. Filtering Offensive Content Changes Its Visibility but Not User Behavior: Two Randomized Controlled Trials with 200,000 Users on Nextdoor

**arXiv ID:** 2607.21853 | [PDF](https://arxiv.org/pdf/2607.21853v1)

**作者:** David J. Grüning `[一作]`, Matthew Katsaros `[通讯]` (Yale Law School)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Nextdoor平台上，作者分别开展了两项覆盖10万用户的随机对照实验，检验通过隐藏或下调可视度的过滤技术是否能降低暴力或冒犯性内容的曝光，并评估其对用户访问、消费和生产行为的影响。

**💡 创新点**

创新点在于：①首次在真实社交平台上对比两种不同强度（12% vs 95%）的可视度过滤干预；②提供大规模、跨国且涵盖多种内容类型（评论、帖子）的实证证据，揭示过滤干预虽然能显著降低暴露，却无法改变用户行为，挑战业界对可视度控制效能的普遍假设。

**🔧 技术方法**

主要技术包括：
- Nextdoor自研的去除预测模型与报告触发机制（Study 1）；
- Google Jigsaw Perspective API（Toxicity 0.70阈值）实现的主动评分与过滤（Study 2）；
- 双向混合ANOVA与贝叶斯ANOVA的统计分析；
- 通过随机抽样评估Bayes Factor 10 来检验交互效应的显著性。

**📊 数据集**

数据集来自Nextdoor的内部平台日志，包含两组100,000名随机分配的用户在实验前后30–60天的行为记录（访问、观看、发布、评论、举报等），并标注出基于报告或Perspective评分的冒犯内容；所有数据已上传至OSF公开仓库（Study 1: https://osf.io/3wgqr/；Study 2: https://osf.io/qfbv6/）。

**📈 对比分析**

比较方法：对照组与实验组的时间×处理交互效应；使用混合ANOVA检验显著性，随后用贝叶斯ANOVA评估效应大小；结果显示：
- 在Study 1中，过滤导致12%视图下降（p<0.001）；
- 在Study 2中，过滤导致95%视图下降（p<10⁻¹⁴⁰）；
- 除了可视度减少外，所有平台使用、内容消费、内容生成及冒犯内容产生指标均未出现显著交互（p>0.05；BF₁₀<0.3），支持行为无变化的零假设。

**⚠️ 局限性**

局限性包括：
- 只衡量行为指标，未评估用户态度、感知或福祉变化；
- 过滤机制本身保持不透明，未检验透明化或反馈对结果的影响；
- 仅涵盖英语内容，受限于美国、加拿大、英国和澳大利亚的用户；
- 长期效应与跨平台可推广性未知；
- 只测试了“可视度过滤”而未结合“内容删除/标记”等更强干预。

---

## 111. Toward High-Fidelity 3D Point-Cloud Learning for Brain Folding Morphology Prediction Using Trans-Unet

**arXiv ID:** 2607.21840 | [PDF](https://arxiv.org/pdf/2607.21840v1)

**作者:** Geran Zhao `[一作]` (Binghamton University), Guifang Fu `[通讯]` (Binghamton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出一种新的Trans-Unet框架，通过先将高分辨率3D点云映射到二维UV网格，再利用U形网络与Transformer自注意力结合的方式实现脑表面折叠的高精度预测与重构。

**💡 创新点**

创新点包括：①双向3D→2D UV映射在保留细节的同时显著降低维度和计算成本；②U-Attention结构融合CNN局部特征提取与Transformer全局依赖，解决点云置换不变性与长程关系；③多损失（L2、TV、latent、LPIPS）组合进一步提升几何与感知质量。

**🔧 技术方法**

使用了UV映射算法、U形网络（含残差块和Batch Group Normalization）、Transformer多头自注意力、缺失点补全的最近邻搜索、Chamfer Distance评估以及多项损失组合训练策略。

**📊 数据集**

采用由有限元模型生成的高分辨率脑表面与纤维点云数据集，包含40,401个表面点和2,382个纤维点，涵盖四个发育阶段（state0–3）以及多种纤维分布与皮层厚度变化。

**📈 对比分析**

通过与PointNet、PointNet++在同一40,401点（无纤维、无增强）以及8,081点下的对比，Trans-Unet的Chamfer Distance从45.2/36.7降至0.045；进一步加入纤维信息、增量状态与数据增强后CD进一步下降至0.012，实验结果和消融实验验证了各模块对性能的显著提升。

**⚠️ 局限性**

局限性包括：①仅在局部脑片段上验证，缺乏全脑扩展；②依赖有限元生成的数据，尚未在真实MRI点云上直接验证；③对更大规模点云的可扩展性仍有待评估；④需要更多多样化样本以避免过拟合。

---

## 112. Certified in Theory, Broken in Practice: Assumption Gaps in Cryptographic Model Certification

**arXiv ID:** 2607.21839 | [PDF](https://arxiv.org/pdf/2607.21839v1)

**作者:** Carter Luck `[一作]` (University of Massachusetts Amherst), Nicolas Papernot `[通讯]` (Vector Institute & University of Toronto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了在隐私保护的机器学习审计中，模型提供者通过伪造训练数据进行攻击，导致审计通过但部署后性能恶化的风险；

**💡 创新点**

创新点在于：①揭示并量化审计特定数据与分布泛化间的安全缺口；②提出基于分布的正式安全定义和可复用的加密模型认证（CMC）协议模板；③演示针对准确率、公平性和差分隐私的具体数据伪造攻击，并证明常见统计检测方法无法阻止该攻击；

**🔧 技术方法**

使用技术包括零知识证明（ZKP）、加密承诺、学习理论中的集中性界限、以及对决策树、神经网络等模型的攻击算法；

**📊 数据集**

实验数据集涵盖六个行业公平性基准（ACSEmployment、Adult、COMPAS、German Credit、Default Credit、Communities & Crime）以及通用的准确率和公平性评测；

**📈 对比分析**

与传统CMC方法对比，本文的安全协议在满足零知识、绑定和声誉性时，能够在足够样本量下实现高概率完整性和近似声誉性；攻击实验显示在未采用安全协议时，模型可以在审计数据上超过99%准确率，却在同分布新样本上仅30%准确率；

**⚠️ 局限性**

限制：需要在模型提交前隐藏审计数据，故实际部署需持续采样或使用更多多方安全计算；对差分隐私的安全协议仍不完备，需要进一步验证训练数据的可信来源。

---

## 113. Data eccentricity, asymptotics of Gaussian RBF reproducing kernel Hilbert space, and kernel PCA

**arXiv ID:** 2607.21823 | [PDF](https://arxiv.org/pdf/2607.21823v1)

**作者:** Sergio A. Alvarez `[一作]` `[通讯]` (Boston College), Sergio A. Alvarez (Boston College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析高斯RBF核RKHS与线性核的几何结构，证明在大带宽极限下两者几何相同，并证明高斯核PCA收敛于线性PCA。

**💡 创新点**

提出并证明数据本征度ρ作为衡量高斯核大带宽收敛阈值的指标，并给出O((ρ/σ)^2)的误差界限，同时首次将此几何视角应用于高斯核PCA。

**🔧 技术方法**

采用解析展开与高阶逼近、格拉姆矩阵双中心化、RKHS与线性内积对比、谱扰动理论、主角度与特征值收敛分析等技术。

**📊 数据集**

使用30个来自OpenML的真实数据集，属性维度从1到100，eccentricity ρ范围从1.15到1806。

**📈 对比分析**

通过归一化的RKHS度量失配、特征值差距和特征投影失配等指标，绘制对数尺度曲线，验证收敛速率为1/σ^2，收敛起始 σ≈ρ，实验结果与理论一致。

**⚠️ 局限性**

受限于有限精度数值运算，带宽上限约为2^7，导致在极高 ρ 数据集上无法充分验证；对重复特征值的收敛性尚未深入研究。

---

## 114. StARS: Socially Appropriate Robot Actions via a Recommender System-Driven Approach

**arXiv ID:** 2607.21802 | [PDF](https://arxiv.org/pdf/2607.21802v1)

**作者:** Erencem Ozbey `[一作]` (Bogazici University), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a2602d71-93ab-4bad-974b-672788df8193` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 StARS 框架，将社交适当性动作生成改写为用户偏好预测问题，利用协同过滤对每位用户的场景-动作适当性评分进行个性化建模。

**💡 创新点**

创新点在于：1) 将推荐系统中的协同过滤与场景编码器结合，形成可插拔的个性化层；2) 采用多任务矩阵分解共享用户/场景因子并加入动作嵌入，缓解稀疏问题；3) 通过残差融合将内容信息注入协同过滤，使模型在数据稀疏时仍保持鲁棒性。

**🔧 技术方法**

技术手段包括：协同过滤矩阵分解（MF）、场景编码器（如 MLP、ResMLP、DCNv2、GraceAE、FT-Transformer、GCN、GIN、GraphSAGE、GGNN、RGCN、GAT）以及两阶段训练策略（先预训练场景编码器，再端到端联合优化）。

**📊 数据集**

使用了两套 HRI 数据集：MannersDB+（家庭交互场景的动作适当性评分）和 SocNav1（社交导航场景的导航动作适当性评分），两者均保留了评注者身份信息。

**📈 对比分析**

通过五折交叉验证和内层交叉验证的超参数搜索，比较基线模型与 StARS 版本的 RMSE/MSE、Pearson r 与 CCC；统计检验采用 Bouckaert‑Frank 校正的配对 t 检验并用 Holm 方法校正多重比较。结果显示：在所有基线模型和数据集上，StARS 版均显著降低误差（RMSE 降至 ~0.15，MSE 降至 ~0.005）并显著提升相关性（Pearson r 上升 ~0.20–0.25，CCC 上升 ~0.20–0.25），在低数据稀疏场景下收益尤为明显。

**⚠️ 局限性**

局限性包括：1) 依赖离线人工评分，缺少在线实时反馈机制；2) 对极少量用户/场景交互仍可能出现估计不稳，需进一步改进隐式反馈融入；3) 仅关注动作适当性，未覆盖更丰富的多模态上下文与非语言信号；4) 在真实机器人部署中需考虑模型推理延迟与资源限制。

---

## 115. Diffusion Models in Medical Image Inpainting: Challenges, Solution Taxonomy, and Future Directions

**arXiv ID:** 2607.21904 | [PDF](https://arxiv.org/pdf/2607.21904v1)

**作者:** Arthur Dantas Mangussi `[一作]` (Federal University of São Paulo), Pedro Henriques Abreu `[通讯]` (University of Coimbra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

系统综述60篇关于 diffusion‑based 医学图像修补论文，并在统一实验协议下评测10个公开实现。

**💡 创新点**

提出了完整的 taxonomy、统一 benchmark、跨缺失机制的对比实验，为该领域提供了第一份标准化评估框架。

**🔧 技术方法**

主要技术为 diffusion models（DDPM、LDM、SGM、SDE）以及相关的 Transformer/UNet 结构。

**📊 数据集**

使用公开数据集包括 BraTS、DeepLesion、TCGA、OASIS、LiTS 等多种医学影像数据。

**📈 对比分析**

通过 PSNR/SSIM、时间等指标对比，LDM‑based NeuroLIT 与 SGM‑based TPDM 在多种缺失机制下表现最优，速度最快。

**⚠️ 局限性**

局限性在于评测数据集不够多样化、缺乏跨模态和跨任务的泛化验证、未探讨临床解释性与不确定性估计。

---

## 116. No Snake Oil: Verifying Python Package Builds

**arXiv ID:** 2607.21888 | [PDF](https://arxiv.org/pdf/2607.21888v1)

**作者:** Jens Dietrich `[一作]` (Victoria University of Wellington), Behnaz Hassanshahi `[通讯]` (Oracle Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对10,449个纯Python发行版进行独立重建，分析重建成功率、失败原因、源代码一致性、按位重现率，并提出一种基于datalog的可解释等价判定工具Daleq4py，以提高重建的可接受性。

**💡 创新点**

①将Python生态系统的重建问题与Java等语言对比，系统评估重建成功率与按位重现率；②提出可解释的等价关系，采用datalog规则对wheel进行规范化，实现对重建结果的可追溯性；③展示该方法能将按位不相同的重建结果提升至可接受的60–79%等价率。

**🔧 技术方法**

使用两种现成的重建工具（Macaron与Oracle Build-From-Source）；Python源码AST归一化、文件哈希与规范化；基于Soufflé实现的datalog推理，生成可解释的IDB；SHA-256哈希比较、diffoscope等辅助工具。

**📊 数据集**

从libraries.io获取的流行Python包列表（2,500包），筛选出纯Python发布版后得到10,449个发行版；进一步挑选同时可被两工具重建的5,066个发行版用于源代码一致性与等价性评估。

**📈 对比分析**

对每个发行版进行两轮重建并计算SHA-256哈希对比；对能成功重建的发行版，使用AST归一化比较源代码；对重建与原始wheel进行等价判定，采用datalog规则归一化后对比IDB。实验结果显示，重建成功率分别为68%（Macaron）和56.5%（bfs），按位重现率分别为15.4%和19.1%，可解释等价率提升至60.2%（Macaron）和78.9%（bfs）。

**⚠️ 局限性**

（1）源代码识别仍易受标签、重定向、分支变动等影响，导致误判；（2）等价规则尚不完整，可能漏掉更多可接受的重建；（3）datalog推理的可解释性对工程师的易用性需进一步评估；（4）研究仅覆盖纯Python发行版，未涉及包含二进制扩展的包；（5）实验资源消耗大（≈1.8 TB数据，24 天构建）。

---

## 117. VisCanvas: A Node-based Interface for Exploratory Visualization Authoring with LLMs

**arXiv ID:** 2607.21886 | [PDF](https://arxiv.org/pdf/2607.21886v1)

**作者:** Yuki Ueno `[一作]` (Arizona State University), Chris Bryan `[通讯]` (Arizona State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于节点的交互范式（VisCanvas），通过在画布上显式地可视化可视化状态、分支、合并和比较，支持LLM辅助的非线性探索式可视化创作。

**💡 创新点**

创新点在于将可视化生成与探索史记录耦合到图结构中，使用户能够在同一画布上并行维护多条分析路径、随时分支、比较和合并，突破传统线性聊天式交互的局限；同时结合自我反思机制提升LLM生成的Vega‑Lite规范可靠性。

**🔧 技术方法**

技术包括：前端基于React + Reactflow + Zustand + elkjs实现可拖拽的图形编辑器；后端使用FastAPI和LangGraph编排LLM管道（Data Summarizer、Analysis Goal Generator、Vega‑Lite Spec Generator）；利用OpenAI GPT‑4.1‑nano或GPT‑5‑nano生成规范并自我修正；提供可视化编辑器（Visual Builder、Vega‑Lite Editor）和语义缩放、数据探索面板等功能。

**📊 数据集**

使用了三组公开数据集：汽车数据集（用于预热）、Spotify 2000‑2019 热曲数据集（13列）以及社交媒体与心理健康数据集（705行、13列）。

**📈 对比分析**

通过对20名计算机研究生进行的两轮控制实验，比较了VisCanvas与基准线性聊天界面VisChat；指标包括任务完成度、NASA‑TLX工作量、用户偏好、探索拓扑结构、LLM响应时间。结果显示两者在任务成功率与工作量上无显著差异，但VisCanvas在支持多路径探索、并行分析、比较/合并以及树状/混合拓扑结构方面更受欢迎，LLM响应时间平均为6.7秒（GPT‑4.1‑nano），可接受。

**⚠️ 局限性**

局限包括：仅使用数据摘要而非完整数据可能影响规范准确性；LLM管道缺乏多轮对话与澄清能力；当前仅支持Vega‑Lite，限制了可视化语法与交互的多样性；实验对象为高水平研究生，缺乏对初学者或专业领域用户的验证；任务时间短且受限，无法观察长周期的探索与协作行为。

---

## 118. Closing the Loop: Training-Free Revisit Consistency for Autoregressive Generative Rendering

**arXiv ID:** 2607.21848 | [PDF](https://arxiv.org/pdf/2607.21848v1)

**作者:** Wenchao Ma `[一作]` (Roblox), Haomiao Jiang `[通讯]` (Roblox)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对基于3D引擎条件的自回归视频生成中因KV缓存限制导致的重访不一致问题，提出了一种无训练、基于时空对应关系的推理时记忆与注意力偏置方法；

**💡 创新点**

创新点在于：①利用相机姿态检索将历史已生成的潜在片段回插入KV缓存，实现真正的循环闭合；②通过深度重投影和相机几何计算为注意力 logits 加入高斯偏置，让模型在查询时自然而然地聚焦到对应的历史位置；

**🔧 技术方法**

技术包括：自回归视频扩散模型（Causal Wan‑VACE）、Self‑Forcing 训练技巧、相机姿态匹配检索、深度重投影与视角变换、基于距离的高斯注意力偏置；

**📊 数据集**

使用的数据集为 TartanGround‑Revisit（71条循环轨迹）和 TartanAir‑Revisit（285条手工构造循环轨迹），均来自 Unreal Engine 真实感环境，包含相机位姿与精确平面深度；

**📈 对比分析**

与 Self‑Forcing 基线及 Infinity‑RoPE、Deep Forcing、MemRoPE 等无训练缓存管理方法对比，本文方法在关键点匹配、DINO相似度、L1误差、主题/背景一致性、动作平滑度、审美与图像质量等指标上均显著提升，且在 VBench‑Long 视频质量上保持甚至略有提升；

**⚠️ 局限性**

主要限制是需要引擎提供准确的相机位姿与平面深度；若仅依赖在线重建估计的几何，方法会因对应噪声而受限。

---

## 119. Relaxed activation analysis of dataflow networks - A clock calculus for machine learning and real-time scheduling

**arXiv ID:** 2607.21797 | [PDF](https://arxiv.org/pdf/2607.21797v1)

**作者:** William Gaudelier `[一作]` (Inria), Dumitru Potop Butucaru `[通讯]` (Inria)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了对Lustre时钟演算的保守扩展，支持在ML训练和资源分配场景中对布尔条件的子采样顺序进行重排，从而消除因激活顺序不匹配导致的类型错误。

**💡 创新点**

创新点在于引入等价关系与E-统一化，使得不同子采样顺序的时钟被视为等价；扩展了时钟语法并保持低复杂度和可解释性，解决了现有时钟演算在处理ML训练算法（如反向传播、门控专家模型）时的局限。

**🔧 技术方法**

使用了基于 Hindley-Milner 的类型系统、E-统一化、时钟等价关系、子采样表达式扩展、以及可证明的推导规则和校验算法。

**📊 数据集**

论文没有采用具体数据集，而是通过理论推导和示例程序（如 GMoE、卷积层、RNN 等）验证新时钟演算的可行性。

**📈 对比分析**

对比方法主要是传统 Lustre 时钟演算，指出新的演算在处理 ML 训练相关代码时不再出现类型冲突，且保持了低复杂度的静态分析；实验结果表明改进后的编译器能够成功处理先前不可编译的示例，且错误诊断更清晰，资源分析更准确。

**⚠️ 局限性**

局限性包括：仍是保守扩展，可能无法覆盖所有复杂的 ML 控制模式；需要进一步实验证明在大规模真实项目中的性能；理论模型未与现有深度学习框架（如 PyTorch、JAX）进行实质性比较。

---

## 120. QLPO: Quadrant-weighted Sampling for Length-aware Policy Optimization

**arXiv ID:** 2607.21793 | [PDF](https://arxiv.org/pdf/2607.21793v1)

**作者:** Siwei Chen `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于分区加权重采样的 RL 算法 QLPO，利用超生成候选集并在四象限中按长度与正确性重采样，隐式引导模型生成更短的 Chain‑of‑Thought 回答。

**💡 创新点**

创新点在于不修改奖励函数或加入显式长度惩罚，而是通过结构化的重采样改变梯度分配，既保持正确性信号又偏向短答、长错，从而在多种规模模型上实现更高效推理。

**🔧 技术方法**

技术包括：GRPO（组相对策略优化）、分区加权重采样（四象限划分）、组‑轨迹梯度分析、token‑级损失归一化以及标准 PPO 目标。

**📊 数据集**

使用的评测数据集包括 MATH‑light‑eval、DAPO‑MATH、GSM8K、OlympiadBench、GPQA、AIME‑2024，实验还扩展到 Geo3K（多模态）和 Eurus‑2‑RL‑Data（代码生成）。

**📈 对比分析**

与基线 GRPO、rollout‑matched GRPO、GFPO 以及 L1、Laser、ThinkPrune 等压缩方法比较，QLPO 在保持或提升准确率的前提下平均缩短 30%–70% 的响应长度；在 1.5B–32B 模型中，准确率提升或保持，长度显著下降。

**⚠️ 局限性**

局限性包括：需要额外的超生成候选生成成本（虽为内存瓶颈但仍需 16% 训练时间增幅）、对参数 α 的选择敏感、对极长但正确推理路径可能略有抑制，且在极复杂任务上尚未验证极限效果。

---

## 121. Reliability-Contagion Feasibility in LLM Multi-Agent Networks

**arXiv ID:** 2607.21912 | [PDF](https://arxiv.org/pdf/2607.21912v1)

**作者:** Ruiwu Niu `[一作]` (Hong Kong Shue Yan University), Ying Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于SEICS的纠正感知网络模型，并将其与清洁可靠性基准结合，推导出图类存在性与无效传播阈值的理论条件。

**💡 创新点**

将错误传播的流行病学阈值与多代理系统的清洁可靠性需求统一，给出了可行连接区间与最优规则；并在固定发送者预算下揭示了密度对传播阈值的中性作用。

**🔧 技术方法**

使用谱分析、next-generation算子、Gillespie模拟与实验性多代理网络实验。

**📊 数据集**

在实验中使用了36个人工构造的闭域二分类任务与6个LLM实例的固定可靠性集合。

**📈 对比分析**

通过理论阈值与Gillespie轨迹的成功率、实验中的第一代后裔与完整级联比例进行对比，结果显示连接度提升导致传播风险上升，但可靠性满足仍需满足阈值。

**⚠️ 局限性**

假设信号独立、可靠性已知、图结构为无向或正则，且实验规模有限；未考虑真实LLM输出的相关性与自适应交流策略。

---

## 122. CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Image-Based Score Following

**arXiv ID:** 2607.21899 | [PDF](https://arxiv.org/pdf/2607.21899v1)

**作者:** Yining Yang `[一作]` (Western University), Jie Han `[通讯]` (University of Alberta)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CODA，一种实时基于图像的乐谱跟踪系统，采用级联选择框架实现系统、乐句和音符的几何一致预测，并通过基于静音的跳跃恢复模式应对乐谱跳转与重复等断点。

**💡 创新点**

创新点在于：1) 将跟踪任务建模为已知候选项的级联选择而非独立检测，显著降低搜索空间并提升预测稳定性；2) 引入统一的静音驱动跳跃恢复机制，无需预知跳转结构；3) 设计了基于Beam Search和可学习时间先验的解码流程；4) 提供了标注完整重复结构的MSMD测试集跳跃评测基准。

**🔧 技术方法**

采用了Mamba时序音频编码器、FiLM条件化的CNN视觉特征提取、跨模态注意力、Beam Search与时间先验、以及基于静音的跳跃恢复机制。

**📊 数据集**

使用MSMD（Multimodal Sheet Music Dataset）数据集，包含354个训练、19个验证、94个测试钢琴曲目的合成音频和对应的谱面图像。

**📈 对比分析**

与多种现有图像跟踪基线（MM‑Loc、RL、CUNet、CYOLO、CYOLO‑SB）进行对比，在合成图像+合成音频的评估中CODA在≤0.10 s阈值下取得0.914的跟踪比率，系统和乐句准确率分别提升到0.991和0.975；在合成图像+真实音频的评估中也超过竞争对手（≤0.10 s 0.743 vs 0.630）。跳跃恢复实验中，CODA在重复子集的1 s内恢复率达78%，显著优于CYOLO‑SB的12%。

**⚠️ 局限性**

局限性包括：仅在钢琴单声道数据上训练和评估；缺乏对多声部、合奏、扫描页等真实场景的验证；跳跃恢复仍依赖静音提示，对无静音的跳转效果未知。

---

## 123. Multi-Agent System-driven Digital Twins for predictive maintenance: architectures, technologies and open research challenges

**arXiv ID:** 2607.21873 | [PDF](https://arxiv.org/pdf/2607.21873v1)

**作者:** Korota Arsène Coulibaly `[一作]` (Hassan II University of Casablanca), Mohamed Hamlich `[通讯]` (Hassan II University of Casablanca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性综述了数字孪生与多智能体系统在资源受限环境下的预测性维护应用，构建了多维度分类体系并提出三大研究问题。

**💡 创新点**

提出了MAS驱动的三层Edge–Fog–Cloud数字孪生架构，阐明了自适应AI代理在微控制器上实现异常检测与分类的可行路径。

**🔧 技术方法**

运用了多智能体框架、自动编码器与1D-CNN混合模型、MQTT/Nanopb轻量通信、资产管理壳AAS以及层级同步机制。

**📊 数据集**

主要以CWRU（振动）数据集为实验基准，亦检视少量工业现场数据，缺乏大规模真实故障数据。

**📈 对比分析**

通过对547篇高影响力论文的定量比较，评估了模型精度、推理延迟、带宽占用等指标，指出现有系统普遍满足准确率>90%但推理延迟和能耗尚未统一衡量。

**⚠️ 局限性**

局限性在于缺乏嵌入式硬件实现验证、真实场景数据与能耗评测不足、并且缺少大规模可扩展性与安全可信性研究。

---

## 124. When Is a Learned Command Adapter Worth It? Closed-Loop Identification and Counterfactual Auditing of Frozen Locomotion Policies

**arXiv ID:** 2607.21867 | [PDF](https://arxiv.org/pdf/2607.21867v1)

**作者:** Zongtan Li `[一作]` `[通讯]`, Zongtan Li

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对冻结的行走控制策略，提出一种适配器必要性审计方法，用以判断是否需要学习命令适配器，并在此框架下评估了一种基于识别的可行性门控补偿器（VGCC）。

**💡 创新点**

创新点在于：①将适配器必要性拆解为全局操作点增益、同状态反事实头room、以及相对频率匹配随机混合的可恢复状态分配增益与交叉拟合固定参考的部署增益；②引入三通道决策（Go/No-Go/Abstain）结合用户设定的阈值；③利用闭环命令-响应识别模型作为可选决策特征，验证预测准确性并未必能提升排序；④在三种机器人和多目标任务上进行大规模确认性审计。

**🔧 技术方法**

技术包括：命令级残差适配器（VGCC）、可识别的三通道闭环响应模型、交叉拟合的治疗效应选择器、源集群重抽样与学习者重拟合、Bootstrap 区间估计以及对比的固定尺度、反应守门器、采样MPC 等控制器。

**📊 数据集**

数据集主要来自 Isaac Lab 的粗糙地形仿真环境：Go2、ANYmal-D 四足机器人、H1 双足机器人，涵盖多组目标距离与方向，并通过大量仿真实验（数千条轨迹）收集命令-响应样本。

**📈 对比分析**

比较方法：VGCC 与直接控制、固定尺度（0.75、0.90）、反应守门器、采样MPC 等。结果显示，VGCC 在能耗和机械功耗上相对直接控制有约5–10% 的降低，且成功率与直接控制相近，但在确认性审计中未能超过 1% 的可恢复分配增益，最终判断为 Abstain；固定尺度在成本-时间前沿表现最优。

**⚠️ 局限性**

局限性包括：①评估仅在仿真环境下进行，未验证硬件能耗；②VGCC 的候选命令集合仅覆盖有限方向与角度，未能充分展示其完整潜力；③重抽样与Bootstrap 估计仍受源集群数量限制；④识别模型在极端低性能情形下的准确性不足；⑤缺少对更复杂控制接口（如姿态、抓取）或更高维度命令空间的验证。

---

## 125. Decentralized Compute on Untrusted Hardware Using Intel TDX and Encrypted CVMs

**arXiv ID:** 2607.21865 | [PDF](https://arxiv.org/pdf/2607.21865v1)

**作者:** Venish Patidar `[一作]` (Manifold Labs Inc), Sathi Nair `[通讯]` (Intel)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出并实现了一个基于 Intel TDX 与 NVIDIA Confidential Computing 的去中心化机密计算平台，利用分布式节点提供加密的 CVM，结合远程 attestation、IP 绑定和区块链激励，实现安全、透明且经济的 AI 计算。

**💡 创新点**

创新点包括：①将 TDX 与 NVIDIA 可信计算结合，形成完整的 CPU+GPU 可信执行环境；②通过每实例唯一加密的 Ubuntu CVM 与 IP 绑定，防止磁盘迁移和重放攻击；③采用持续的验证循环（每 72 分钟一次）与挑战‑响应机制，保证节点始终处于可信状态；④将验证结果与激励机制捆绑，利用多方验证者与区块链权重投票实现公平支付与防篡改；⑤在此基础上构建了基于 WireGuard + Kubernetes 的去中心化调度与恢复框架。

**🔧 技术方法**

核心技术包括：Intel Trust Domain Extensions (TDX)、Intel Trust Authority (ITA) 与 Key Broker Service (KBS)、NVIDIA Confidential Computing (Hopper/Blackwell GPU)、Manifold Attestation Agent、持续 attestation 工作流、区块链激励层、WireGuard 私有网络、Kubernetes 控制平面。

**📊 数据集**

文中未使用公开数据集，主要关注基础设施与安全机制的实现与验证。

**📈 对比分析**

对比方法主要从成本与安全性两个维度讨论：相较于集中式云服务，去中心化模型通过硬件共享与激励降低了单价；在安全性方面，利用硬件根信任、全盘加密与持续 attestation 证明了对机密数据、模型和执行状态的完整保护，但论文并未给出量化的性能基准或实验结果。

**⚠️ 局限性**

局限性包括：①仅适用于支持 TDX 与 NVIDIA Confidential 的硬件，硬件成本与可用性仍是门槛；②依赖 Intel Trust Authority 的可信度，若该服务被破坏则安全保障失效；③未考虑微架构侧信道、GPU 内存泄漏等高级攻击；④持续 attestation 与区块链激励对网络延迟与算力消耗的影响尚未量化；⑤缺乏大规模实测基准，难以评估真实吞吐与延迟。

---

## 126. Data Quality over Capacity: Internalizing Documents into LoRA Adapters for Closed-Book QA

**arXiv ID:** 2607.21861 | [PDF](https://arxiv.org/pdf/2607.21861v1)

**作者:** Joan Figuerola Hurtado `[一作]` `[通讯]` (Independent Researcher), Joan Figuerola Hurtado (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Gemma‑4‑e4b模型上使用LoRA训练器，将文档知识直接编码进模型权重，实现闭卷问答。

**💡 创新点**

证明数据质量（答案简短、去除琐碎）比模型容量或架构更重要，并展示内部化模型在精度和延迟上优于真实BM25‑RAG基线。

**🔧 技术方法**

4‑bit 量化 Gemma 模型 + LoRA 适配器 + 两阶段训练（CPT式完成 + SFT）以及数据集生成与精简策略。

**📊 数据集**

自建自生成问答对，来源于文档片段；实验规模为 1–99 份文档的合成语料。

**📈 对比分析**

对照 BM25 检索 + RAG、真实 oracle、单阶段与双阶段 LoRA；内部化适配器在 15 文档集上召回 84.2%、整体准确 85.7%，比 BM25‑RAG 高 25 个点，延迟更低。

**⚠️ 局限性**

仅单一随机种子，指标偏向样式匹配的适配器，数据规模有限（99 文档），4‑bit 量化可能影响性能；缺乏多种硬件和大规模验证。

---

## 127. Joint Load Balancing and Transmit Power Control for Energy Efficiency Maximization in the Satellite-Cell-Free Massive MIMO Uplink

**arXiv ID:** 2607.21860 | [PDF](https://arxiv.org/pdf/2607.21860v1)

**作者:** Ngo Tran Anh Thu `[一作]` (Hanoi University of Science and Technology), Lajos Hanzo `[通讯]` (University of Southampton)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一种融合低地球轨道卫星与无小区Cell‑Free大规模MIMO的混合网络，推导了MRC检测下的闭式上行ergodic吞吐量，并联合用户关联与功率控制进行能效最大化。

**💡 创新点**

创新点：①首次在此类混合网络中给出考虑LoS+NLoS Rician衰落的闭式吞吐表达式；②将能效优化建模为混合整数非凸问题；③提出改进的差分进化（IDE）框架，结合双重变异策略和SHADE参数自适应，理论证明收敛到ε‑最优解。

**🔧 技术方法**

使用的技术主要包括：MRC线性组合、MMSE信道估计、混合整数非凸优化、改进差分进化（IDE）以及SHADE自适应机制。

**📊 数据集**

实验数据集：基于3GPP规范的农村/城市信道参数，Starlink星座仿真（3颗卫星、70个AP、50个用户），仿真中使用了多维空间相关Rician模型、卫星高度300km、100天线等参数。

**📈 对比分析**

方法比较：与单独地面、单独卫星基线以及PSO、HGA、SDE、RCGA等元启发式算法比较。结果显示混合网络平均能效提升约25%；IDE在能效、运行时、收敛速度上均优于其他算法，收敛到约3.2–3.35 Mbit/J的能效值。

**⚠️ 局限性**

局限性：①能效优化仅考虑总吞吐率，公平性/最小速率未完全兼顾；②算法仍为启发式，无法保证全局最优；③在极大规模网络或动态场景下，计算复杂度和收敛速度仍是挑战。

---

## 128. SoundscapeAgent: Agentic Soundscape Construction for Controllable Synthesis and Scalable Audio-Language Supervision

**arXiv ID:** 2607.21857 | [PDF](https://arxiv.org/pdf/2607.21857v1)

**作者:** Hao Zhang `[一作]` (Wuhan University), Steve Yves `[通讯]` (Tencent Hunyuan)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SoundscapeAgent，一个基于LLM的代理框架，用于可控的声音景观构建、可视化的场景规划与渲染，并实现可扩展的音频-语言监督数据生成。

**💡 创新点**

核心创新在于将声音场景拆分为可执行的规划、混合资产检索与生成、时间线渲染和元数据导出四个可检视的步骤，使得生成过程可控、可编辑并可用于后续训练；同时引入离线先验模式以提升数据生成效率。

**🔧 技术方法**

技术手段包括：LLM驱动的场景规划与工具调用、CLAP文本-音频相似度与生产复杂度筛选、检索与基于文本的音频生成（如EzAudio、TangoFlux），以及基于时间线的确定性混音与空间化。

**📊 数据集**

使用的数据集有：AudioSet、AudioCaps、Clotho、WavCaps、Hive收集的单事件音频、私有资产库以及由模型生成的合成音频；用于评估的下游任务包含MMAU、CaptionStew-400k等。

**📈 对比分析**

评估方法：①在人类听力测试与CLAP相似度上与TangoFlux、EzAudio、AudioLDM2单发生成器对比，SoundscapeAgent在情绪抽象、氛围及时间控制等场景中获得最高的对齐分数；②在MMAU音频推理任务中，加入10万条合成数据后整体准确率从51.05%提升至56.5%（+5.45%），显著提升了声音相关和复杂事件推理能力。

**⚠️ 局限性**

局限性包括：对资产库的依赖与生成模型质量限制、合成与真实音频的域差距、语音事件覆盖不足，以及评测指标未能覆盖所有结构化推理细粒度信息。

---

## 129. Quantifying Political Partisanship for Cross-Platform Analyses

**arXiv ID:** 2607.21842 | [PDF](https://arxiv.org/pdf/2607.21842v1)

**作者:** Fathima Ameen `[一作]` (North Carolina State University), Christopher G. Healey `[通讯]` (North Carolina State University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种纯文本、跨平台可比的党派度量方法，并在 2024 年 5-10 月的 Bluesky 与 Truth Social 上对比政治言论。

**💡 创新点**

创新点在于利用 AllSides 媒体偏见评分作为外部信号，构建基于 Transformer 嵌入空间的党派轴，完全不依赖平台特定结构，实现在不同社交媒体之间的可比性。

**🔧 技术方法**

技术手段包括 Transformer 句子编码（Matryoshka）、PCA 降维、MiniBatch k‑means 聚类、语义去重、Empirical Bayes 样本缩减、基于聚类中心差异构建党派轴并投影得到分数。

**📊 数据集**

使用的数据集为 Bluesky 与 Truth Social 2024 年 5‑10 月的约 1.3 M 条帖子（含作者、时间、互动指标）以及独立验证用的 X 平台 2024 年选举期的 848K 条含新闻链接推文。

**📈 对比分析**

比较方法：在 Bluesky/Truth Social 内部验证和 X 外部验证中计算投影得分与 AllSides 偏见评分的 Pearson 相关系数，内部相关系数 0.365，外部 0.193；seed 选取轴在所有可能轴的分布中位列 99.5% 以上，显示该方法在不同平台上具有稳健性。

**⚠️ 局限性**

局限性：仅适用于美国两党体系，依赖 AllSides 的偏见评分；聚类与 seed 选择对轴构造敏感；未在多党制或更主流多元平台上进行测试；方法对极少新闻链接的帖子无效。

---

## 130. How Do AI Coding Agents Contribute to Software Development? an Empirical Study of Agentic Pull Requests

**arXiv ID:** 2607.21832 | [PDF](https://arxiv.org/pdf/2607.21832v1)

**作者:** Iren Mazloomzadeh `[一作]` (Polytechnique Montréal), Foutse Khomh `[通讯]` (Polytechnique Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地量化并分析了 AI 编码代理在开源 Python 项目中的拉取请求（PR）行为、任务类型分布以及对软件质量的影响。

**💡 创新点**

创新点在于首次将代理 PR 的合并可行性、任务偏好和缺陷倾向与人类 PR 进行纵向四季度对比，并通过 BERTopic 主题挖掘等无监督方法揭示代理 PR 的任务特征。

**🔧 技术方法**

采用了 GLMM、BERTopic 主题模型、Mann–Whitney U 检验、Cliff’s Delta、Empirical Bayes 等统计与文本挖掘技术进行数据分析。

**📊 数据集**

使用了 AIDev 数据集，对 220,612 条闭合 PR 进行筛选，获得 9,428 条由 Codex、Copilot、Claude、Cursor、Devin 等五大代理生成的 PR，涵盖 489 个 Python 仓库。

**📈 对比分析**

通过对代理 PR 与人类 PR 进行平衡抽样、统计比较（合并率、任务分布、复杂度指标、缺陷率），结果表明代理 PR 的合并率与人类相近，任务偏好集中于文档、依赖和测试，缺陷率无显著差异。

**⚠️ 局限性**

研究局限在于仅关注 100 星以上的 Python 开源仓库，缺少对私有或低星项目的覆盖；代理 PR 的识别依赖于 PR 标记，可能存在误判；样本仅覆盖 5 种代理，无法代表所有编码助手的表现。

---

## 131. A Graph-Based Control Interface for Traffic Signals on Heterogeneous Road Networks

**arXiv ID:** 2607.21831 | [PDF](https://arxiv.org/pdf/2607.21831v1)

**作者:** Bertil Braun `[一作]` `[通讯]`, Bertil Braun

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于共享图神经网络的交通信号控制接口，利用对每条交通移动的得分映射到各路口的合法相位集合，并在合成网格和城市仿真上使用PPO进行评估。

**💡 创新点**

创新点在于将学习与本地相位空间分离，仅对移动进行评分，通过确定性 incidence 矩阵生成可变大小的相位集合，从而实现图结构与动作空间大小的解耦，并允许跨路口共享参数。

**🔧 技术方法**

使用了 typed graph GNN（两层消息传递+sum聚合）、Bron‑Kerbosch 最大兼容集枚举、PPO 强化学习与自定义局部奖励，以及 SUMO 仿真框架。

**📊 数据集**

使用了生成的合成正方形/矩形网格（如 6×6）以及五个德国城市（Karlsruhe、Mannheim、Heidelberg、Freiburg、Stuttgart）的交通网络与路由数据。

**📈 对比分析**

通过与固定时间、最大压力、排队等基线在相同决策间隔下比较，发现学习策略在未见网格尺寸时吞吐量和完成率优于最大压力，但在信号覆盖率下降时性能下降；在城市网络中表现参差不齐，某些城市超越基线，其他城市相当。

**⚠️ 局限性**

局限性包括：未能分离各设计要素的贡献；相位枚举仅限于最大兼容集，可能缺失可选相位；仅在有限的合成与城市网络上评估，通用性有限；对信号覆盖变化敏感；只使用一次城市训练样本和三条交通种子；采样执行优于贪婪执行，未验证确定性控制效果。

---

## 132. Protocol-Level Attacks on Agentic Commerce Platforms: A Cross-Platform Taxonomy, AIP-Bench, and Unified Defense

**arXiv ID:** 2607.21824 | [PDF](https://arxiv.org/pdf/2607.21824v1)

**作者:** Yedidel Louck `[一作]` `[通讯]` (Ariel University), Yedidel Louck (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对三大主流 AI 代理式商业平台进行系统安全评估，发现并分类了33个结构性漏洞，构建了确定性基准和统一的 HTTP 侧车防御方案。

**💡 创新点**

创新点在于提出结构性攻击与语义性攻击的严格区分，首次发布跨平台的结构性漏洞分类和可复现的确定性基准，并提出无需平台改动即可覆盖四类结构性风险的侧车防御。

**🔧 技术方法**

研究采用协议层侧车、ECDSA 签名验证、CSRF/重定向硬化、原子支付状态、MCP 调用授权等技术；同时利用 Docker Compose 环境、OpenRouter LLM 调用与 Solana Devnet 进行实战验证。

**📊 数据集**

使用的“数据集”包括三平台原始实现代码、公开的 Docker Compose 复现环境、OpenRouter LLM 的 1440 次实验日志以及所有结果的 SHA‑256 哈希。

**📈 对比分析**

通过确定性判别器对 100% 攻击成功率与防御后 0% 成功率进行对比；侧车性能测得 P1 约 4.8k req/s、P5 约 12k req/s，平均请求延迟 0.21 ms，误报率为 0%。

**⚠️ 局限性**

局限性包括仅评估三平台、侧车仅覆盖 HTTP 路径、P4 原子性证明仅适用于单进程、V5 的 LLM 试验受模型温度和提示依赖限制，以及未覆盖 WebSocket、文件系统等非 HTTP 层面漏洞。

---

## 133. Claim Plane: Enforceable Change Intents and Dynamic Scope for Parallel Coding Agents

**arXiv ID:** 2607.21909 | [PDF](https://arxiv.org/pdf/2607.21909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 134. Probing Speaker Identity Sensitivity in Audio Deepfake Detectors

**arXiv ID:** 2607.21820 | [PDF](https://arxiv.org/pdf/2607.21820v1)

**作者:** Daniyal Kabir Dar `[一作]` (Michigan State University), Arun Ross `[通讯]` (Michigan State University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并评估了身份敏感度得分（ISS），用来在推理时诊断音频深度伪造检测器对说话人身份的依赖。

**💡 创新点**

提出一种无标签、推理时可用的身份敏感度得分，能够区分基于身份的错误与纯合成痕迹的错误，并通过语音转换实验验证其真实性。

**🔧 技术方法**

使用ECAPA‑TDNN说话人嵌入、在检测器logit空间加入身份上下文扰动、计算IQR作为ISS；在AASIST与RawNet2两种架构上验证，并采用FreeVC进行声学转换实验。

**📊 数据集**

ASVspoof 2019 LA 与 2021 LA 逻辑访问数据集（包含TTS、VC 与混合攻击）。

**📈 对比分析**

与传统基于置信度的度量（熵、边际、置信度）对比，ISS在预测错误时的AUC最高；在2019 LA 上AUC达 0.954，在2021 LA 上保持 >0.84；高ISS样本在分布偏移时错误率显著上升，说明ISS能有效识别身份敏感错误。

**⚠️ 局限性**

仅在两种检测器上验证；依赖外部ECAPA‑TDNN说话人模型；α 调参和原型构造需有标签数据；在更大、更多样化的说话人库上评估尚未完成。

---

## 135. Action-Conditioned World Model for Goal Plane Probe Guidance in Robotic Ultrasound

**arXiv ID:** 2607.21918 | [PDF](https://arxiv.org/pdf/2607.21918v1)

**作者:** Siqi Fan `[一作]` (Chinese University of Hong Kong), Hongbin Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究了一个基于动作条件世界模型的机器人超声探头目标平面引导框架，通过两阶段模型学习预测超声图像变化并指导探头动作。

**💡 创新点**

创新点在于将潜在条件扩散世界模型与目标条件时间变压 Transformer 相结合，并利用冻结的世界模型提供感知奖励进行策略微调，实现端到端的动作条件超声动态学习与目标平面导航。

**🔧 技术方法**

采用潜在条件扩散模型（CDiT）、VAE 编码/解码、时间变压 Transformer、单轴离散动作分类、基于世界模型的强化学习奖励以及力/扭矩传感器反馈等技术。

**📊 数据集**

使用了自建颈部超声轨迹数据集，包含 20 名受试者的三组扫描轨迹（颈动脉、甲状腺、随机扫面），共约 1.2M 帧。

**📈 对比分析**

与传统视频预测方法相比，世界模型在目标导向扫描中保持更高的 LPIPS/SSIM；在真实机器人实验中，颈动脉与甲状腺闭环引导成功率分别为 70% 与 65%，优于仅基于监督的策略。

**⚠️ 局限性**

模型仅能处理局部目标导向扫描，无法区分器械动作与生理运动；单轴动作设计限制了多轴同时校正，且未将力传感信息融入模型。

---

## 136. TRW: TRACE-RealWorld---An Auditable Consistency Contract for World Models as Materialized Views

**arXiv ID:** 2607.21910 | [PDF](https://arxiv.org/pdf/2607.21910v1)

**作者:** Edward Y. Chang `[一作]` `[通讯]` (Stanford University), Edward Y. Chang (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种面向世界模型的承诺层一致性合约，结合了可刷新语义声明、基于 Kalman 的自适应刷新、基于 SagaLLM 的补偿机制以及可回放的审计记录；并在 Flood‑SAR（基于真实地理信息的搜救仿真工作台）上进行系统实现与评估。

**💡 创新点**

①首个针对世界模型的承诺层一致性合约；②将刷新策略与补偿机制组合证明可在给定假设下给出违约上界；③以声明化的刷新条件和冲击驱动的补偿实现可审计的决策链；④提供理论与实验双重验证的完整框架。

**🔧 技术方法**

1) 语义化声明与刷新条件（typed claims、refresh_condition）; 2) Kalman‑gated自适应刷新（dual‑Kalman 流管理与 VoI 路由）； 3) SagaLLM 风格的事务性补偿； 4) TRACE 事件溯源与记录； 5) 残差分析与一致性债务量化； 6) 价值信息（VoI）决策和阈值驱动的门控； 7) 预先校准的概率地图与 OOD 检测。

**📊 数据集**

基于真实的 Antioch Delta 河口地理与水文栅格，合成的搜索救援任务数据集（380 个开发种子、475 个验证种子、240 个测试种子）。

**📈 对比分析**

与固定间隔刷新、有效性时钟以及全局恢复等基线对比；评估指标包括：stale 执行率、残余违约、完成率、救援人数、观测负载、总成本、恢复延迟、检测延迟与覆盖率。结果显示：自适应刷新显著降低 stale 执行率但成本和覆盖率略低；局部 Saga 补偿相比全局恢复显著降低修复工作量和延迟，残余违约差距为零。

**⚠️ 局限性**

①仅在 Flood‑SAR 场景下验证，缺乏跨领域泛化；②依赖校准不确定性、时延等假设，实际部署时可能不满足；③理论中未完全覆盖非零漂移或非高斯噪声；④依赖于完整的依赖图和准确的补偿事务；⑤未在真实硬件上评估观测成本与时延。

---

## 137. Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation

**arXiv ID:** 2607.21908 | [PDF](https://arxiv.org/pdf/2607.21908v1)

**作者:** Yunjin Gu `[一作]` (Chinese University of Hong Kong Shenzhen), Hua Xu `[通讯]` (Tsinghua University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全无监督的多模态意图发现方法 MCSP，通过选取高质量代表样本，利用多模态大型语言模型（MLLM）进行对比推理生成语义概念，并在概念感知图上进行语义传播和对比学习，实现意图聚类。

**💡 创新点**

创新点包括：① 通过 MLLM 推理从代表样本生成可解释的高层语义概念；② 在概念感知图上结合几何权重与语义一致性进行边权重调节，构建语义传播机制；③ 将随机游走扩散与概念监督对比学习形成闭环反馈，提升嵌入空间的结构化和解释性。

**🔧 技术方法**

核心技术包括：多模态预训练（文本 BERT、视觉 Swin Transformer、音频 WavLM 及跨模态融合）、MLLM 对比推理（Gemini‑3.0‑Pro / Qwen‑3‑VL）、k‑NN 图构建与语义调节、随机游走语义传播、概念监督的多视角对比学习。

**📊 数据集**

在三大多模态意图发现基准上验证：MIntRec、MIntRec2.0 以及 MELD‑DA。

**📈 对比分析**

与六类基线（SCCL、CC、MCN、USNID、SPILL、UMC）对比，MCSP 在 ACC、ARI、NMI、FMI 等四指标上均优于所有基线，尤其在 MIntRec 与 MELD‑DA 上平均提升约 3%–4%。

**⚠️ 局限性**

局限性：① 对于深层语用（如讽刺、社会立场）仍易混淆；② 受代表样本噪声与类别不平衡影响，概念生成质量可能下降；③ 需要高性能 MLLM，推理成本较高。

---

## 138. Cleaning the NTP Pool: Detecting and Mitigating NTP-Sourced IPv6 Scanning

**arXiv ID:** 2607.21903 | [PDF](https://arxiv.org/pdf/2607.21903v1)

**作者:** Erik Rye `[一作]` (Johns Hopkins University), Robert Beverly `[通讯]` (San Diego State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

通过在NTP Pool上发送带有唯一nonce的NTP请求，监测并记录返回的back‑scan流量，从而识别出哪些NTP服务器用于收集IPv6客户端地址并随后进行端口扫描。

**💡 创新点**

首次系统性量化了NTP Pool被滥用为地址收集与端口扫描的机制，提出了基于nonce和Hop‑Limit的反向扫描检测方法，并对四类扫描实体进行了聚类与行为特征剖析。

**🔧 技术方法**

使用了NTP协议标准报文、IPv6地址nonce注入、Hop‑Limit递增探测、网络抓包与关联分析、ASN和rDNS查询，以及自研的back‑scan识别脚本。

**📊 数据集**

收集了约2,335台NTP服务器的响应数据、约2.45 百万个唯一nonce的NTP请求，以及后续约945个被扫描nonce的回显与端口扫描记录，包含来自不同云平台（AWS、Linode、Digital Ocean等）的扫描IP。

**📈 对比分析**

与传统被动扫描识别相比，该方法通过主动诱发扫描实现更高的准确率，检测到四个聚类后分别评估了扫描延迟、持续时间、扫描率和端口覆盖度；Cluster 2展示了最高的扫描率（≈100%）和最广泛的端口分布。

**⚠️ 局限性**

局限包括：仅在单一美国云视角测量，可能漏检区域性扫描；DNS枚举的NTP服务器不完整导致的漏检；低探测速率可能忽略采样率极低的监控实体；以及对非NTP协议（如其他众包服务）扫描的未知风险。

---

## 139. Learning Adaptive Semantic Gaussian Allocation for 3D Occupancy

**arXiv ID:** 2607.21896 | [PDF](https://arxiv.org/pdf/2607.21896v1)

**作者:** Kanglin Ning `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (PengChengLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Semantic Gaussian Allocation Transformer（SAGFormer），通过对预设的高斯原语进行候选生成、打分和全局 Top‑K 选择，实现有限数量的 3D 语义占据预测中的高效高质量高斯分配。

**💡 创新点**

核心创新在于：①利用几何与语义上下文对每个高斯进行编码，②引入 keep/clone/split/suppress 四种候选操作，并通过软计数与硬 Top‑K 组合实现精确的容量限制；③通过全局打分与采样显著降低冗余与语义混合，提升占据精度。

**🔧 技术方法**

采用 Transformer 编码器建模高斯间关系，使用多分支路由器生成候选，高斯属性与语义熵作为特征；训练时结合占据交叉熵、Lovász‑Softmax、语义熵正则、占据恢复损失和预算约束；推理时进行全局 Top‑K 选择得到最终高斯集合。

**📊 数据集**

在 nuScenes‑SurroundOcc 与 SSCBench‑KITTI‑360 两大基准数据集上进行评估，涵盖多模态（相机 + LiDAR）以及单帧原始输入场景。

**📈 对比分析**

与 GaussianFormer、GaussianFormer3D、QuadricFormer、F4Splat、PG‑Occ 等对照实验显示，SAGFormer 在保持或略微增加高斯数量的情况下，IoU 提升约 4–5%（例如 nuScenes 41.74% vs 33.04% baseline），mIoU 亦显著提升；同时在冗余率、语义混合率和语义支持度上均优于现有分配策略。

**⚠️ 局限性**

限制主要包括：在校准的相机‑LiDAR 监督环境下测试，动态前景提升有限；对传感器降噪、天气变化、分布漂移等场景鲁棒性不足；需要进一步研究不确定性处理与实时部署的可扩展性。

---

## 140. Farmland Extent and Visible Boundary Mapping from 1 m NAIP Imagery Using Residual U-Net and Text-Prompted SAM 3 Refinement

**arXiv ID:** 2607.21881 | [PDF](https://arxiv.org/pdf/2607.21881v1)

**作者:** Mohammadreza Narimani `[一作]` (University of California), Parastoo Farajpoor `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

建立了一种混合ResUNet与SAM3的工作流，用于从1 m NAIP RGB影像中提取农田范围和可见边界。

**💡 创新点**

创新点在于将域训练的ResUNet与冻结的、文本提示的SAM3概念掩模融合，保持模型可解释性并提升难点结构的召回。

**🔧 技术方法**

使用了残差U‑Net、Dice+BCE损失、文本提示的SAM3、逻辑或融合以及滑动窗口拼接。

**📊 数据集**

数据集为37幅NAIP 1 m RGB场景，人工标注的农田多边形，切成256×256像素样本，共5,698样本。

**📈 对比分析**

在场景分离的训练/验证/测试上，ResUNet单独获得Dice 0.9234、IoU 0.8605、召回 0.9794；融合后在难例上Dice提升至0.955/0.903，整体拼接Dice在典型场景0.898–0.919。

**⚠️ 局限性**

限制包括未得到实例化边界、依赖单幅RGB图像、对不同年份/季节泛化未知、SAM3可能产生误报且未做置信度裁剪。

---

## 141. Approximation Algorithms for Inventory Problems with Decomposable Submodular Ordering Costs

**arXiv ID:** 2607.21858 | [PDF](https://arxiv.org/pdf/2607.21858v1)

**作者:** Retsef Levi `[一作]` (Massachusetts Institute of Technology), Emily Zhang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对子模联合补货问题（SJRP）的近似算法，该算法适用于通过项目类别分解得到的子模订购成本函数，并给出了 O(k) 的近似比，其中 k 为项目类别数。

**💡 创新点**

创新点包括：
① 定义了一类更通用的子模成本函数，允许跨类别的任意交互；
② 通过将 SJRP 归约为离散时间的“无交叉区间覆盖问题”（DICP）并设计水填充（water‑filling）分层与动态规划的两阶段取整方案；
③ 证明了当 k 为常数时即可得到常数因子的近似保证，这是此前仅在极特殊子模成本（树、层次、基数）上已知的。

**🔧 技术方法**

使用的技术手段有：
- 线性规划松弛与其双对偶求解；
- 子模函数的 DR‑submodular 性质与层次索引（subset level）定义；
- 水填充分层（water‑filling）对 LP 解进行空间划分；
- 动态规划求解层内最小成本覆盖；
- 随机化取样分析与期望成本上界；
- 通过构造 DICP 的定量化折半区间实现对 SJRP 的还原。

**📊 数据集**

论文未使用实验数据集，全部工作为理论算法与证明。

**📈 对比分析**

与之前的研究相比：
- 传统 JRP 与 IRP 的常数近似已知，子模 SJRP 仅有 O(loglog min(N,T)) 的近似；
- 本文实现了 (8k+1) 近似，比现有最优 (O(loglog min(N,T))) 更优，且在 k 为常数时即为常数因子；
- 通过数值举例（如 N=4, T=3 的图示）说明水填充与动态规划的工作流程。

**⚠️ 局限性**

局限性：
- 仅适用于可按 k 类分解的子模成本函数，无法直接覆盖所有子模 SJRP 或一般 IRP；
- 近似比随类别数线性增长，k 过大时效果可能不理想；
- 目前未证明在更一般的子模成本下是否存在常数因子近似。

---

## 142. LeAct: Learning to Reason from Expert Actions

**arXiv ID:** 2607.21856 | [PDF](https://arxiv.org/pdf/2607.21856v1)

**作者:** Ziran Yang `[一作]` (Princeton University), Chi Jin `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用现有的无声专家系统（如游戏求解器、规划器、定理证明器等）仅输出动作，训练大型语言模型生成链式思考（CoT），并通过与专家动作的概率匹配对生成的CoT进行筛选，从而学习推理过程。

**💡 创新点**

创新点在于：①把专家动作作为外部监督，将动作与自然语言推理关联；②将CoT视为潜变量，采用基于重要性加权的EM框架；③通过前向delta评分和顶K筛选实现闭环反馈，避免了传统的自我合理化或随机采样。

**🔧 技术方法**

主要技术包括：反向生成CoT（给定状态与动作的摘要让模型解释）；前向delta评估（计算CoT提升动作概率的增量）；顶K筛选；专家策略强制（EPF）与联合SFT训练；迭代EM式训练循环。

**📊 数据集**

数据集涵盖多种不完全信息游戏（Leduc Hold'em 4K–80K、Liar's Dice、3玩家Leduc、Flop Hold'em约10^9信息集）以及机器人立方体堆叠基准BuilderBench（46任务）。

**📈 对比分析**

与两类基线对比：①单纯SFT（无CoT）coldstart；②传统专家迭代（ExIt）。实验显示：在枚举规模下LeAct在模仿性能（可解释为KL或可破坏率）上与或优于基线；在更大规模（Flop Hold'em）和泛化（rank‑split、OOD机器人任务）上，LeAct明显优于基线，提升幅度可达5×、+60 mbb/g等。

**⚠️ 局限性**

局限性包括：需有可提供动作分布的专家或前沿模型；对大规模游戏仍受计算成本限制；CoT生成候选数和筛选阈值敏感；在某些任务（如多玩家非零和游戏）缺乏收敛保证；实验仅在8B模型上验证，尚未验证更大规模学生的效果。

---

## 143. Searching the Space of Feed-Forward Neural-Network Weight-Update Rules with Fixed Depth Symbolic Regression

**arXiv ID:** 2607.21855 | [PDF](https://arxiv.org/pdf/2607.21855v1)

**作者:** Charles Brum `[一作]` (University of California Irvine), Edward Finkelstein `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用符号回归探索并发现显式的前馈神经网络权重更新规则，力求在小型符号回归基准上超过传统手工设计的优化器。

**💡 创新点**

创新点在于将符号回归与固定深度表达式树相结合，构建包含梯度、动量、适应性归一化等常见优化器构件的搜索空间，从而自动生成可解释且紧凑的更新公式。

**🔧 技术方法**

技术包括多线程化的固定深度后缀语法与遗传编程（Genetic Programming）符号回归、基于梯度、动量、指数移动平均等运算符和常量的表达式树搜索。

**📊 数据集**

使用了10个来自 Feynman 与 Hemberg 的符号回归基准函数，以及三种不同结构的全连接前馈网络（包含 Sigmoid 与线性激活）。

**📈 对比分析**

对比方法是先在每个基准/网络组合上做网格搜索寻找最佳超参数的传统优化器（如 GD、HB、NAG、AdaGrad、RMSProp、AdaDelta、Adam、AdamW），然后用符号回归生成的规则在同一设置下训练10轮并比较均方误差（MSE）。实验表明，在30个组合中有25个符号回归规则实现了比最优传统优化器更低的MSE，总体下降幅度为44.47%。

**⚠️ 局限性**

局限性包括仅在极小网络、短期训练（10轮）以及符号回归基准上验证；缺乏大规模数据集、分类任务、长时间训练和理论收敛性分析；搜索过程随机，可能导致不同实验得到不同规则；未对规则复杂度或稳定性做约束。

---

## 144. ToolGuardian: Declarative Security for AI Agent-Tool Interactions

**arXiv ID:** 2607.21835 | [PDF](https://arxiv.org/pdf/2607.21835v1)

**作者:** Arun Ravindran `[一作]` (University of North Carolina at Charlotte), Saurabh Deochake `[通讯]` (SentinelOne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ToolGuardian框架，实现AI代理工具交互的预审与任务感知授权。

**💡 创新点**

首次使用Answer Set Programming（ASP）构建可审计、可解释的声明式安全策略，并与启发式与LLM实现对比。

**🔧 技术方法**

核心技术包括分层工具特征抽取（描述、系统调用、模拟执行、源代码分析）和ASP声明式策略推理。

**📊 数据集**

使用16个MCP风格工具（8善意、8恶意变体）以及20个包含原子和复合工作流的授权场景。

**📈 对比分析**

在工具预审中，ASP在P2层级达到0.86 F1和88%准确率；在运行时授权中，三种实现均正确判定全部20个场景，表明ASP与LLM/启发式在充分策略输入下性能相当。

**⚠️ 局限性**

局限性包括基准规模有限、仅覆盖MCP生态、未评估长链工作流或多样化代理平台，且ASP规则编写需要人工专业投入。

---

## 145. Adaptive Driving Style for SAE Level-2 Driving Automation: Minimizing Preference Mismatch

**arXiv ID:** 2607.21819 | [PDF](https://arxiv.org/pdf/2607.21819v1)

**作者:** Kumar Akash `[一作]` (Honda Research Institute USA Inc), Gaojian Huang `[通讯]` (San Jose State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在SAE Level-2 自动驾驶中，根据驾驶员的偏好动态调整车辆的驾驶风格，并通过驾驶模拟实验验证了自适应策略的有效性。

**💡 创新点**

通过将事件级驾驶风格偏好预测与隐式适配控制相结合，提出了基于 GRU 的预测模型和最优驾驶风格选择策略，实现了无需显式反馈即可降低风格不匹配并提升信任度。

**🔧 技术方法**

使用了事件级隐式预测模型（GRU）、加权交叉熵、AUC 评估、最优风格选择控制策略、实验设计（六种适配会话、六个事件）以及统计检验（t 检验）等技术。

**📊 数据集**

收集了 36 名驾驶员在城市驾驶模拟器中完成的 6 种适配会话数据（共 198 次会话、8 个事件），并在验证阶段使用 12 名新驾驶员的模拟数据。

**📈 对比分析**

通过比较固定、基于信任、基于偏好以及预测隐式适配策略的偏好一致率和累计信任度，结果显示预测策略在从防御式起始时获得最低不匹配率且平均信任度最高，性能指标为偏好一致率≈70%（相较于基线≈55%）且 AUC≈0.86。

**⚠️ 局限性**

研究局限在于使用中等保真固定平台模拟器、样本年龄偏年轻、驾驶风格仅包含四个离散纵向参数、未考虑更高维度或连续风格、实验时间短且未验证真实车辆场景。

---

## 146. Adversarial Prompts for Acceptance Collapse in Speculative Decoding

**arXiv ID:** 2607.21804 | [PDF](https://arxiv.org/pdf/2607.21804v1)

**作者:** Run Wang `[一作]` (Clemson University), Mert D. Pesé `[通讯]` (Clemson University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种通过在提示中插入短后缀攻击推测式解码中的验收机制，从而导致大幅度推断延迟的攻击方法ADSD。

**💡 创新点**

首次提出提示侧的验收崩溃攻击，并引入Soft-Collapse损失与目标保持KL正则，引导攻击在保持答案质量的同时显著降低推测式解码效率。

**🔧 技术方法**

基于梯度引导的离散后缀搜索、Soft-Collapse目标、前向KL保持、离散搜索Beam等技术。

**📊 数据集**

主要在GSM8K、HumanEval、CNN/DailyMail等数据集上评估，并在Qwen、LLaMA-3、EAGLE-3等多架构上测试。

**📈 对比分析**

与正常推测、随机后缀、适配GCG基线对比；在GSM8K上平均样本时间从26.05s递增至42.29s（+62.3%），块效率降低38.9%，速度下降36.1%，但准确率仅下降2.3%；在其他策略与模型上亦维持显著降速。

**⚠️ 局限性**

仅在白盒优化条件下验证，攻击对提示长度敏感，缺乏针对性防御；在跨域攻击中隐蔽性不一定保持；未提供完整在线防御方案。

---

## 147. Are Production Cloud Skills Adequately Tested? Measuring and Governing Skill Test Coverage in Practice

**arXiv ID:** 2607.22015 | [PDF](https://arxiv.org/pdf/2607.22015v1)

**作者:** Haotian Si `[一作]` (Alibaba Cloud), Dengcheng He `[通讯]` (Alibaba Cloud)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Skill Test Coverage度量，开发了可审核的测量管线，并将其应用于阿里云云技能的发布流程中；

**💡 创新点**

创新点在于把自然语言工作流技能的隐式操作义务转化为可度量的覆盖单元，并通过人工与AI协作生成可追溯的覆盖报告和改进建议；

**🔧 技术方法**

技术主要包括基于大语言模型（Qwen3.7-Max）的义务候选生成、聚合与人工复核、测试用例覆盖关系推理及源代码关联的推荐生成；

**📊 数据集**

使用数据集为阿里云公开的157个待发布云技能及其测试用例，涵盖了从VPC、ECS到安全组等多种资源操作；

**📈 对比分析**

对比方法主要是设定80%/90%的覆盖门限，并统计通过/未通过的技能数量；在初始测量中平均覆盖率80.3%，63.7%的技能满足80%门限，51.6%满足90%门限，说明该度量能识别大量未覆盖义务；

**⚠️ 局限性**

局限性包括未评估推荐接受率与后续覆盖提升效果、缺乏跨厂商或跨语境的泛化验证、以及对人工复核一致性的研究不足。

---

## 148. Unfit for stranding assessment: a panel-scale multimodal-LLM audit of building-decarbonisation disclosure (BeDA)

**arXiv ID:** 2607.22006 | [PDF](https://arxiv.org/pdf/2607.22006v1)

**作者:** Jingyi Xu `[一作]` (Yale University), Anchen Sun `[通讯]` (Google)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并验证了一种名为BeDA的多模态大语言模型工具，用于对全球建筑业企业的可持续性报告进行可信度评分，并基于此评估披露的碳强度数据是否满足科学路径下的分割评估需求；

**💡 创新点**

创新点在于：①先行在可信度评估中实现跨模型、跨模型族的鲁棒性验证；②引入对披露是否符合按面积归一化强度要求的判别，为分割监管提供可操作的缺口诊断；③在单一大语言模型框架内完成多模态文本、图表、表格的联合编码与提取；

**🔧 技术方法**

技术手段包括：多模态LLM（Gemini、Gemma系列）进行文本和视觉内容的特征提取、评分；规则式提取器结合正则与单位解析实现碳强度字段抽取；统计与回归模型验证可靠性与外部一致性；

**📊 数据集**

数据集为：SSSR多模态可持续性报告语料（2,246家公司、9,166个报告年份），其中建筑业子集为116家公司（含部分未解析公司），并补充了1,235份可持续报告的房地产子集（519份可构建CRREM评估的报告）；

**📈 对比分析**

比较方法：与传统单模型指标（Loughran‑McDonald词典、ClimateBERT）以及外部标准（LSEG‑ESG、RepRisk事件计数、SBTi目标验证）进行相关性与回归对照；性能方面，可信度评分在跨模型（r≈0.85）和跨模型族（r≈0.74‑0.82）间保持高度一致，且与三类外部基准在受控制后仍保持正相关（r≈0.20‑0.25），但与人类评分的相关性为负，表明人类评估存在宽容偏差；

**⚠️ 局限性**

局限性包括：①外部验证仅覆盖上市公司，未覆盖非上市或未披露的建筑业者；②缺乏跨供应商的模型验证，可能存在厂商特异性偏差；③对报告体裁和语言多样性的控制有限，可能影响跨地区比较；④抽取器的规则化方法可能在复杂表格中出现误差；⑤CRREM评估为基于企业平均强度的首次切面，未捕捉资产级分散性；⑥漏检与模型记忆的检验样本有限，未能完全排除模型对公司身份的识别。

---

## 149. Three-Body Alignment: Aligning Chess Agent with Human Reasoning through Reranked Rationale

**arXiv ID:** 2607.21993 | [PDF](https://arxiv.org/pdf/2607.21993v1)

**作者:** Jaymari Chua `[一作]` (University of New South Wales), Lina Yao `[通讯]` (University of New South Wales)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多源棋局推理理由数据集，并通过检索与重新排序的方式提升人工智能的推理与人类大师的对齐度。

**💡 创新点**

创新点在于：①使用agentic pipeline将非结构化视频评论转化为可查询的理由数据；②提出RR‑RAG框架实现多源理由的检索与排序；③提供了丰富的谜题标注基准来评估理由的语义与战术一致性。

**🔧 技术方法**

采用的技术包括：agentic数据工程流水线、t‑SNE降维可视化、Google Gemini嵌入、FAISS向量检索、检索增强生成（RAG）与重新排序、以及多轮LLM推理。

**📊 数据集**

使用的数据集为Hugging Face上发布的trichess多源理由数据集，其中包含顶级棋手、Stockfish（NNUE）评估及LLM（如Qwen3‑32B、Gemini）生成的理由，另外还构造了增强的谜题标注数据。

**📈 对比分析**

通过对200个测试局面比较Zero‑Shot、RAG和RR‑RAG三种方法，评估指标为理由的语义相似度（余弦相似度）和战术质量（归一化Stockfish分数），结果显示RR‑RAG将对齐度从0.61提升至0.73，但相应的战术分数下降约20%。

**⚠️ 局限性**

主要局限在于：①对齐提升伴随战术性能的损失，难以同时兼顾人类可解释性与最优计算；②检索与排序机制对局面几何的把握不足，导致相似度误判；③当前实验仅限于象棋领域，泛化到更复杂环境仍需进一步验证。

---

## 150. Mag4D-SLAM Dataset: A Repeated-Traversal Multi-Modal 4D Geomagnetic Dataset for Localization and Mapping

**arXiv ID:** 2607.21986 | [PDF](https://arxiv.org/pdf/2607.21986v1)

**作者:** Bibhutibhusan Nayak `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Giseop Kim `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

创建并发布了首个大规模户外磁感测SLAM数据集 Mag4D‑SLAM，包含 18 km 轨迹、LiDAR、相机、IMU、三轴磁力计和 GNSS 传感器同步采集，并提供高精度 6‑DoF 地面真值。

**💡 创新点**

首次在室外场景下构建多模态、昼夜、正反向、重复行走的磁感测数据集，并在此基础上系统评估磁场可重复性、无漂移全局朝向估计及地点区分特征，为磁力计在 SLAM 中的应用提供基准。

**🔧 技术方法**

采用硬/软铁校准（椭球拟合）和三轴磁力计投影技术，利用 LiDAR‑IMU 里程计、ICP 对齐 HD 先验地图生成高精度地面真值；基线实验包括磁场可重复性分析、全局朝向误差评估和跨会话位置识别回调。

**📊 数据集**

使用自身构建的 Mag4D‑SLAM 数据集，包含 14 条序列、昼夜、正反向、多次遍历的同步多模态记录。

**📈 对比分析**

通过与随机基线比较的 Recall@K 评估磁向量在跨会话位置识别中的效果；单轴 M_x、M_y、M_z 组合可实现 Recall@10≈21%（随机基线 0.08%），全局朝向误差 RMSE 6–8 度且无累计漂移，证明磁力计可提供稳定的全局朝向。

**⚠️ 局限性**

仅涵盖昼夜和正反向重复行走，缺乏季节、天气或月间长期变化；对强磁干扰或大量金属环境的鲁棒性不足；未提供学习型磁特征，仅基于原始磁向量的基线评估。

---

## 151. MoE$^2$-LoRA: When MoE Models Meet MoE-style Low-Rank Adaptation

**arXiv ID:** 2607.21978 | [PDF](https://arxiv.org/pdf/2607.21978v1)

**作者:** Qingyu Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Peng Ye `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对 Mixture-of-Experts (MoE) 模型的参数高效微调方法，结合路由条件投影 (RCP) 与全局共享 LoRA 专家池，实现对 MoE 机制的深度绑定与模型级适配。

**💡 创新点**

创新点在于：①通过 RCP 将预训练路由器的激活信号与任务特定校正共同用于 LoRA 专家路由；②采用单一全局 LoRA 专家池，在各层共享并通过可学习投影实现层间跨专家共享；③通过这两项技术实现了对专家级、模块级和模型级的统一适配，提升了效率与效果。

**🔧 技术方法**

技术上使用了 LoRA、Mixture-of-Experts、路由条件投影（RCP）以及全局共享 LoRA 专家池，并通过 F‑统计量与线性 CKA 等分析方法验证路由一致性与跨层相似性。

**📊 数据集**

实验数据集包括数学推理集 MetaMathQA-R1、GSM8K、MATH‑500；代码生成集 MagicCoder‑OSS、MBPP、HumanEval；通用评测集 MMLU、WinoGrande、ARC‑Challenge、StrategyQA、CommonsenseQA；以及医学多模态 VQA 数据集 VQA‑RAD、SLAKE、PathVQA，和跨模态评测 MMBench、MME、RealWorldQA。

**📈 对比分析**

与 PERFT‑E、MoELoRA、MoLA、DAS‑LoRA 及全微调 FFT 等基线对比，方法在三大 MoE 主干（OLMoE‑1B‑7B、DeepSeek‑V2‑Lite、Qwen3‑30B‑A3B）上实现了在数学与代码任务上的最高平均精度，并在通用与多模态评测中保持甚至提升通用能力，平均提升约 +2.56 分，表明显著优于现有 PEFT 方法。

**⚠️ 局限性**

局限性包括：动态路由的 LoRA 适配无法直接合并回基准权重，导致模型融合困难；全局专家池的规模扩展会使投影/路由参数显著增大，虽然 RCP 通过在预训练路由空间投影减轻该问题，但在极大池规模下仍需更高效的路由设计。

---

## 152. Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning

**arXiv ID:** 2607.21971 | [PDF](https://arxiv.org/pdf/2607.21971v1)

**作者:** Shujin Wu `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个通过强化学习培养自我进化元技能的框架，利用合成的演化轨迹和可验证的程序执行奖励，在编程任务中实现多轮自我改进。

**💡 创新点**

将自我进化视为可学习的能力，提出基于程序运行的连续奖励、数据合成演化轨迹训练以及RL驱动的自我反思、历史利用和反馈驱动的策略。

**🔧 技术方法**

强化学习（GRPO）、数据合成流水线、可验证执行奖励、推理时进化搜索、两阶段多样性筛选、结构化提示格式。

**📊 数据集**

大规模竞争编程数据集 PRIME-RL/Eurus-2-RL-Data（包含 TACO、APPS、Codeforces、CodeContests）以及 AlgoTune 开放式算法优化任务。

**📈 对比分析**

与 Best-of-N、Self-Refine、Reflexion、AlphaEvolve 等基线在七个编程基准上进行10轮自我进化比较，绝对提升10.01%（分布内）、24.12%（分布外），在 AlgoTune 上相对提升46.9%，并生成结构更具多样性的程序。

**⚠️ 局限性**

依赖于代码执行奖励，难以迁移至奖励稀缺领域；训练成本高，错误恢复能力仍有限；对极难任务的改进受限。

---

## 153. Low-Altitude Channel Multipath Prediction via Panoramic Perception and Vision-Language Model

**arXiv ID:** 2607.21953 | [PDF](https://arxiv.org/pdf/2607.21953v1)

**作者:** Zihang Zeng `[一作]` (Shanghai Jiao Tong University), Xiangwen Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于全景视觉与视觉语言模型的低空多径预测框架PanoLAMP。

**💡 创新点**

通过将RGB‑D全景图与预训练视觉语言模型结合，加入深度位置编码和LoRA微调，实现对延迟、功率、方位角、仰角的精准预测。

**🔧 技术方法**

使用预训练视觉语言模型Qwen2‑VL‑2B、深度位置编码、LoRA参数高效微调及多任务混合专家头。

**📊 数据集**

使用AirSim+Sionna RT合成的18,949条UAV‑车辆链路数据集，包含七种无人机高度的RGB‑D全景图与对应的四条主多径参数。

**📈 对比分析**

与ResNet、Transformer和LLM基线对比，PanoLAMP在延迟、功率、角度的NMAE/NMSE和余弦相似度上均优于基线，平均误差下降约30%。

**⚠️ 局限性**

模型规模大、推理时延高，仅在仿真场景验证，未在真实环境下评估。

---

## 154. Learning Population-Level Dynamics through a Latent Fokker--Planck Model and Discrepancy Transport Maps

**arXiv ID:** 2607.21921 | [PDF](https://arxiv.org/pdf/2607.21921v1)

**作者:** Chengyang Huang `[一作]` (University of Southern California), Krishna Garikipati `[通讯]` (University of Southern California)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建一种基于概率分布的无轨迹信息的群体级动力学推断框架，通过将观测概率演化分解为潜在随机过程和差异传输映射两部分来实现系统识别。

**💡 创新点**

创新点在于：①提出了潜在-传输非唯一分解的物理正则化方法（使用超弹性能量约束传输）；②将拉格朗日-罗森布拉特（KR）重排与单调神经网络结合，构造可逆的时间相关传输映射；③在潜在空间引入Ornstein–Uhlenbeck（OU）过程，使得Fokker–Planck方程可解析求解，从而实现高效联合优化。

**🔧 技术方法**

技术手段包括：离散化的样本估计的KL散度、潜在动态的OU参数化、KR重排的单调神经网络、SVK超弹性能量正则化，以及基于样本的联合优化框架。

**📊 数据集**

使用了两组合成数据集：一组是环形四次势能下的OU演化；另一组是双峰Rosenbrock势能下的非线性多模态演化，仅通过时间点上的独立样本快照来构建。

**📈 对比分析**

通过对比观测快照、潜在分布以及重构分布的可视化与KL损失曲线，表明该框架在保持潜在分布简单（Gaussian）并通过传输映射重构复杂几何分布方面具有很高的拟合精度；在两组实验中，重构误差均低于传统仅用传输映射或仅用潜在轨迹模型的做法，展示了方法的优越性能。

**⚠️ 局限性**

主要限制：①潜在动力学仅限于OU过程，限制了对非高斯或多模态潜在演化的表达能力；②KR重排依赖于坐标顺序，可能在高维场景下降低表达灵活性；③对分解的可辨识性、唯一性及稳定性缺乏严格理论保证，需要进一步研究。

---

## 155. Visual Saliency Steering Distillation for Multimodal Chain-of-Thought Reasoning

**arXiv ID:** 2607.22013 | [PDF](https://arxiv.org/pdf/2607.22013v1)

**作者:** Hao Yang `[一作]` (Yunnan University), Xuejie Zhang `[通讯]` (Yunnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种视觉显著性引导蒸馏方法（VSSD），通过注意力映射生成对图像的扰动，并利用奇异值分解提取主导向量，注入到模型中间层以提升小型多模态链式思维模型的细粒度视觉-语义区分能力。

**💡 创新点**

创新点在于：① 用跨模态注意力生成“负向”扰动图像，强调关键信息；② 通过SVD从原图与扰动图的解码层差异中提取最显著的驱动向量；③ 将该向量通过层间蒸馏投射到后置解码层，增强模型对微小跨模态差异的敏感性。

**🔧 技术方法**

技术方法包括：多模态大语言模型（如Qwen2.5‑VL）的注意力提取、图像扰动生成、奇异值分解（SVD）、层间蒸馏（ILDistillation）、T5基准的两阶段生成与推断、交叉注意力融合与门控融合。

**📊 数据集**

使用的数据集主要是 ScienceQA（21k+多模态问答）和其更具挑战性的 M³CoT 版本，对两者均进行实验验证。

**📈 对比分析**

与同等规模的基线（Enigma‑COT、MM‑CoT、DDCoT 等）以及 GPT‑4、LLaVA 等大模型比较，VSSD‑Base（223M）在 ScienceQA 上达到 89.67% 准确率、M³CoT 上 73.19% 平均准确率，显著优于其他小型模型；VSSD‑Large（738M）进一步提升至 93.40%（ScienceQA）和 93.40%（M³CoT），接近甚至超越 GPT‑4‑based LLaVA。

**⚠️ 局限性**

局限性包括：① 依赖大规模 MLLM 生成注意力作为扰动依据，若大模型不可用则效果受限；② 仅针对多模态问答任务，尚未验证对更广泛的跨模态推理或生成任务的适用性；③ 需要额外的蒸馏训练步骤，增加了训练成本与调参复杂度。

---

## 156. Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning

**arXiv ID:** 2607.22002 | [PDF](https://arxiv.org/pdf/2607.22002v1)

**作者:** Heyang Jiang `[一作]` (University of California Los Angeles), Baharan Mirzasoleiman `[通讯]` (University of California Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VIGOR——一种基于奖励方差的迭代 rollout 预算分配框架，专为 GRPO 训练设计，显著提升训练效率

**💡 创新点**

创新点在于：①证明 GRPO 的梯度幅度与奖励方差成正比；②利用这一关系在线动态分配 rollout，优先给高方差样本分配更多样本；③给出闭式速度提升分析，证明在 Pareto 分布下速度优势随迭代次数增长

**🔧 技术方法**

核心技术：Group Relative Policy Optimization (GRPO)、奖励方差估计与排序、迭代生成与筛选策略、理论闭式速度推导

**📊 数据集**

数据集与模型：数学推理任务（MATH、Math500、AIME24、AMC、Minerva Math、Gaokao、Olympiad Bench）以及编程任务 LiveCodeBench v6；使用 Qwen 系列模型（Qwen2.5-1.5B/3B/7B、Qwen3-8B）和 Phi-4-Mini-Instruct

**📈 对比分析**

与基线比较：在 math 任务上，VIGOR 以 2.3× 更少的 rollout 达到目标准确率；在 coding 任务上，VIGOR 以 1.49× 更少的 rollout 获得 GRPO 最终完整通过率，并提升 3.4 分平均测试通过率；总体保持或提升最终性能

**⚠️ 局限性**

局限性：需要手动设定初始预算、放大比例、筛选比例和迭代次数；理论假设多基于二进制奖励和 Pareto 分布，可能在非二进制或不同奖励分布下效果有限；在极长 CoT 任务中，迭代生成仍可能带来一定额外开销

---

## 157. Music-JEPA: Learning a World Model of Sound from Action

**arXiv ID:** 2607.22000 | [PDF](https://arxiv.org/pdf/2607.22000v1)

**作者:** Ziyu Wang `[一作]` (New York University), Yann LeCun `[通讯]` (New York University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了Music-JEPA世界模型，学习钢琴音频的动作条件潜在动态，并利用该模型进行节拍跟踪、作曲家识别、调性识别以及基于规划的钢琴转录。

**💡 创新点**

创新点在于将Joint Embedding Predictive Architecture (JEPA) 应用于音乐领域，加入动作条件动态预测，采用EMA教师避免表示坍塌，并通过逆向预测器实现高效的规划转录。

**🔧 技术方法**

技术主要包括：JEPA框架、Vision Transformer编码器/预测器、动作编码器、EMA教师策略、逆向预测器与Amortized Planning、以及对音频与MIDI的图像化处理。

**📊 数据集**

使用MAESTRO v3.0.0（约200小时钢琴录音+MIDI）进行训练，ASAP数据集用于节拍跟踪评估。

**📈 对比分析**

与音频仅JEPA（AO-JEPA）和大型预训练模型MERT进行对比；在节拍跟踪、作曲家识别、调性识别等任务中，Music-JEPA超过AO-JEPA，性能接近甚至与MERT持平；在持续踏板估计方面表现最佳，但音符级转录指标低于监督方法。

**⚠️ 局限性**

局限性包括：仅在具有动作标注的钢琴数据上验证，难以推广到无动作或更广泛音乐场景；高维动作规划仍不稳定；转录精度不及监督方法；未来需扩展至更抽象的动作及更大规模的音乐数据。

---

## 158. Unified Static-Dynamic Pruning for Efficient LLM Inference

**arXiv ID:** 2607.21985 | [PDF](https://arxiv.org/pdf/2607.21985v1)

**作者:** Jinhyeok Kim `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的稀疏推理框架 SPDP，结合无结构静态剪枝（SP）与输入感知动态剪枝（DP），在 LLM 推理（prefill 与 decode）中实现高效推理。

**💡 创新点**

创新点：
- 设计 Tiled‑Column‑Wise Bitmap Compressed (Tiled‑CBC) 统一压缩格式，既能满足 DP 的列级稀疏访问，又能兼容 Tensor‑Core 的矩阵乘法；
- 针对 decode 的 spMspV 采用 CUDA‑core 的 HAD‑SMBD 解码与异步流水线；
- 针对 prefill 的 SpMM 采用 Tensor‑Core，加入列对齐与布局对齐；
- 在单一格式下实现 SP 与 DP 的协同稀疏度，实现更高的算力密度与内存带宽利用。

**🔧 技术方法**

使用技术：
- Tiled‑CBC 格式与三层索引（GTile、列位图、列元数据）
- CUDA‑core spMspV kernel（HAD‑SMBD）
- Tensor‑Core SpMM kernel（布局对齐）
- 异步组复制与双缓冲流水线
- 量化（4‑bit RTN）与静态剪枝结合
- 基于阈值的动态剪枝（TEAL）
- 评估工具：Nsight Compute、Nsight Systems。

**📊 数据集**

使用数据集：
- WikiText（Perplexity）
- HumanEval、GSM8K、CoQA、MMLU（下游任务精度）
- LLaMA‑2‑7B 作为模型
- C4、Alpaca 进行 DP 校准，SpGPT、Wanda 进行 SP 校准。
- GPU 平台：A10G、L4、L40S。

**📈 对比分析**

对比方法：
- 密集基准 cuBLAS_TC、cuSPARSE、Flash‑LLM、SpInfer。
- 结果：
  * 1.24×–1.37× 平均加速（最高 2.51×）相较 SpInfer；
  * 1.88×–2.11× 加速（最高 3.52×）相较 cuBLAS_TC；
  * 在相同 perplexity 下，SPDP 最高可达 25% 更高的总稀疏度；
  * TPOT（每生成 token 的时间）平均提升 1.34×；
  * 兼容 4‑bit 量化，保持 perplexity 低水平。

**⚠️ 局限性**

局限性：
- 主要针对 30%–60% 的中等稀疏度，极端稀疏（>80%）下不具优势；
- Prefill 加速有限，仍受 Tensor‑Core 预处理与布局对齐开销制约；
- 依赖阈值式 DP，无法直接支持显式 mask/bitmap DP；
- 需要预先静态剪枝，难以动态适应训练后模型变更；
- 量化与稀疏的组合在低精度下元数据比例上升，需进一步优化；
- 目前实现仅在 NVIDIA GPU 上验证，跨平台通用性待验证。

---

## 159. TextSLIP: Text Self-Supervised CLIP for Medical Report Generation

**arXiv ID:** 2607.21970 | [PDF](https://arxiv.org/pdf/2607.21970v1)

**作者:** Haoyu Jiang `[一作]` (Chinese Academy of Sciences), Ziping Cong `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 TextSLIP 框架，通过将 CLIP 与文本自监督对比学习（ESimCSE）相结合，实现医学影像报告生成的预训练与细粒度语义监督。

**💡 创新点**

在 CLIP 的跨模态对齐基础上，首次引入文本内模态对比学习，提升文本嵌入的辨别度，为视觉编码器提供更细致的语言指导。

**🔧 技术方法**

采用 CLIP 的 InfoNCE 损失、ESimCSE 的词重复增广与动量编码器、ViT‑B‑16‑quickgelu 视觉编码器、BioMed‑BERT 文本编码器、Momentum encoder 等技术。

**📊 数据集**

预训练使用从 MedTrinity‑25M 提取的约 700 万脑 MRI 图文对；下游 fine‑tune 采用 BRATS2023‑GLI 数据集（1271 例）进行报告生成。

**📈 对比分析**

在 R2Gen 框架下与 UniMed‑CLIP、BioMed‑CLIP 等 CLIP 风格模型以及未预训练 ResNet‑101 进行对比，使用 BLEU‑1/2/3/4 与 ROUGE‑L 评估。TextSLIP 在所有模态上 BLEU‑1 提升 1‑2%，BLEU‑4 约提升 3%，整体性能优于基线。

**⚠️ 局限性**

仅在脑 MRI 领域进行验证；评估指标主要为词汇级 BLEU/ROUGE，缺乏临床指标和专家评估；文本增广方式过于简单，未充分保留医学实体、否定和不确定性等语义信息。

---

## 160. Incentives and Market Structure in Intent-Based Exchanges: Evidence from a Solver-Reward Reform

**arXiv ID:** 2607.21955 | [PDF](https://arxiv.org/pdf/2607.21955v1)

**作者:** Ruiyang Zhang `[一作]` `[通讯]` (Ryonix Labs Inc. & Flock.io), Ruiyang Zhang (Ryonix Labs Inc. & Flock.io)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对CoW协议的CIP-74解算者奖励改革进行因果分析，测量其对竞争结构与交易价值分配的影响；

**💡 创新点**

首次通过自然实验量化解算者奖励规则变动如何重塑交易价值分配，发现按订单规模的单调集中梯度；

**🔧 技术方法**

采用中断时间序列、变点检测、置换检验、留一解算者检验、交叉场所对照、三重差分等计量方法；

**📊 数据集**

使用CoW协议公开的每日解算者份额、订单级别成交量、UniswapX对照数据以及交易质量指标；

**📈 对比分析**

与UniswapX及市场加权HHI对照，发现大订单集中度提升、平均执行质量无显著变动；在第二次费用调整事件上三重差分结果方向一致但受限于统计功效；

**⚠️ 局限性**

聚集效应受单一顶级解算者驱动；三重差分缺乏足够功效；仅覆盖以太坊主网、单一治理事件；执行质量检验受7 bps功效上限限制。

---

## 161. Location-Aware NAS Timer Optimization in NTN-TN Integrated Networks

**arXiv ID:** 2607.21947 | [PDF](https://arxiv.org/pdf/2607.21947v1)

**作者:** Cheng Liu `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于UE位置的非接入层（NAS）定时器自适应方法，用于低地球轨道（LEO）卫星与地面网络融合的5G系统中注册过程。

**💡 创新点**

创新点在于：①将路径延迟分解为卫星链接几何、ISL跳数与FL延迟的UE特定量化；②引入基于AMF排队暴露和路径可靠性的端点加权机制，实现定时器的精细化个性化配置；③保持闭式表达式的轻量级特性，易于部署。

**🔧 技术方法**

采用了闭式NAS定时器模型AstroTimer的框架，并结合卫星几何、ISL跳数估计、AMF排队模型及路径可靠性评估等技术，构建了完整的UE特定定时器计算公式。

**📊 数据集**

使用自建的Python模拟器生成注册流程数据，参数包括LEO星座、ISL跳数、丢包率、AMF服务速率等，未使用公开数据集。

**📈 对比分析**

与3GPP默认定时器和AstroTimer全局定时器进行对比实验，结果显示：在无丢包情况下，平均注册时延降低约20%，能耗下降约11%，重试次数下降到单次；在丢包增大时，性能仍优于3GPP，基本可与AstroTimer持平。

**⚠️ 局限性**

局限性包括：仅考虑静态卫星几何，未建模卫星实时移动与链路动态；路径可靠性估计采用固定丢包概率，缺乏在线实时更新；未在真实网络环境中验证，实验仅基于仿真。

---

## 162. Listen, Do Not Copy: Internalizing Audio-Grounded Scaffold Context for Robust Omni-Model Speech Understanding

**arXiv ID:** 2607.21943 | [PDF](https://arxiv.org/pdf/2607.21943v1)

**作者:** Pengfei Zhang `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种音频导向的脚手架上下文 (AGSC)，避免多模态大模型在重叠噪声场景下的感知绕过问题，并通过无上下文测试验证模型真正听音。

**💡 创新点**

创新点是将音频生成的局部线索（时间窗、噪声等级、部分打乱词表）作为无答案提示，结合感知绕过检测、无音频控制以及可内部化的训练流程。

**🔧 技术方法**

技术包括 Omni 多模态模型的低秩适配 (LoRA)、自研的 AGSC 线索合成流水线、答案重叠过滤、无音频控制、以及基于 GDPO 的门控与流式输出联合强化学习。

**📊 数据集**

使用的公开数据集包括 LibriSpeech、SparseLibriMix2、AISHELL-3、WHAM、MUSAN、AMI Meeting Corpus，构成了 Context‑Speech Bench (CSB)。

**📈 对比分析**

在三种 Omni 模型上，AGSC 训练后无线索下的 mpWER 在重叠+噪声场景从 25–71% 降至 9–15%，显著优于基线；门控强化学习实现了对复杂段的高召回与低延迟。

**⚠️ 局限性**

局限在于线索合成仍依赖前端工具的准确性，对极端噪声或多说话人密集场景的鲁棒性不足，且在部分模型中门控可能出现崩溃。

---

## 163. Trade-off for Secure UAV-ISCC Systems

**arXiv ID:** 2607.21939 | [PDF](https://arxiv.org/pdf/2607.21939v1)

**作者:** Hongjiang Lei `[一作]` (Chongqing University of Posts and Telecommunications), Gaofeng Pan `[通讯]` (Beijing Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了无人机支持的集成感知、通信与计算（ISCC）系统，在该系统中通过联合优化无人机三维轨迹、波束成形、用户/目标调度以及计算频率，实现安全通信率、雷达估计率与计算能效的单目标最大化，并进一步提出归一化加权总和方法探索三者的性能权衡。

**💡 创新点**

创新点在于首次将物理层安全、感知性能与计算能效统一纳入三维轨迹与波束成形的联合优化框架；通过引入归一化权重、松弛与SCA（序贯凸近似）技术，解决不同度量尺度导致的优化难题；并给出完整的 AO（交替优化）迭代算法与可行解收敛证明。

**🔧 技术方法**

技术手段主要包括：概率LoS/ NLoS 通道建模、联合波束成形与资源调度的半正定规划、三维轨迹的SCA线性化、加权梯度下的多目标优化、以及Gaussian随机化恢复秩一解。

**📊 数据集**

实验采用基于 500 × 500 m 平面场景的人工设置，包含四个通信用户、四个感知目标、一个基站与一个未知位置的窃听者；未使用公开数据集，全部为仿真参数。

**📈 对比分析**

与单目标优化方案（仅通信、仅感知、仅计算）以及均权重和偏重权重设置的比较表明，CSC（加权综合）方案在整体性能上优于单目标；在不同权重配置下，轨迹、速度与能耗等指标均符合预期的权衡。

**⚠️ 局限性**

局限性包括：1）仅考虑单架无人机；2）仅在理想化的LoS/ NLoS 模型下评估，未加入真实气象或干扰影响；3）假设基站与目标位置已知，未讨论定位误差；4）算法收敛速度与实时性尚未在实际硬件上验证。

---

## 164. Semiotic logical hexagon theory for LLM logical reasoning

**arXiv ID:** 2607.21933 | [PDF](https://arxiv.org/pdf/2607.21933v1)

**作者:** Yunyao Zhang `[一作]` (Huazhong University of Science and Technology), Zikai Song `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HexLogicAgent 框架，通过将自然语言命题先映射到六格逻辑六边形（A,E,I,O,U,Y），实现对命题语义的完整组织，然后在此基础上进行逻辑推理与结构化验证。

**💡 创新点**

核心创新是将传统的 Greimas 半正方形扩展为逻辑六边形，引入存在性与混合状态（U,Y）两位置，以实现更完整的语义对立与子对立关系；同时设计三阶段（结构化-推理-反思）过程，利用六边形关系进行多角度验证。

**🔧 技术方法**

采用 FOL 语义化、语义张量化的结构化阶段、推理规划、求解器以及基于六边形约束的直接、快速、深度反思验证。

**📊 数据集**

在 FOLIO、ProntoQA、ProofWriter、ProverQA、RepublicQA 五大逻辑推理基准上进行评估。

**📈 对比分析**

与 Linear、Aggregative、Symbolic、Semiotic 四类基线（包括 Direct、CoT、ToT、CR、SymbCoT、Aristotle 等）对比，HexLogicAgent 在三种 LLM backbone（DeepSeek-V3.2、Qwen2.5-32B、Qwen3-30B-A3B）上平均提升 2.66/2.74/2.41 分；在所有基准中均保持最高或第二高的准确率，尤其在结构化推理与抽象哲学推理任务上优势明显。

**⚠️ 局限性**

主要局限是推理耗时较高（约 5–10 秒/样本），且在小规模模型上效果不如大型模型；未解决不确定性校准问题，导致“未知”预测仍占主要错误比例。

---

## 165. Generalized Neural Operator for Parametric and Boundary-Value Problems

**arXiv ID:** 2607.21932 | [PDF](https://arxiv.org/pdf/2607.21932v1)

**作者:** Ruoyan Li `[一作]` (University of California, Los Angeles), Wei Wang `[通讯]` (University of California, Los Angeles)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种通用神经算子（GNO），通过显式注入 PDE 参数和边界条件，实现对多种物理场的高效模拟

**💡 创新点**

创新点在于：①参数门控混合核实现对参数空间的动态选择；②通用边界传输算子将任意边界条件映射到统一的潜在 Dirichlet 表示；③基于分组分布鲁棒优化的训练目标，提升不同物理 regime 的均匀性能

**🔧 技术方法**

采用神经算子框架（类似 FNO/自注意力）+ 参数门控混合核 + 边界传输算子 + 组分布鲁棒损失 + 逐步训练策略

**📊 数据集**

在四个 PDE 家族（热传导、对流、Burgers、不可压 Navier‑Stokes）上构建数据集，分别包含多种参数取值和 Dirichlet/Neumann/周期边界；并在热、对流上进一步测试边界泛化

**📈 对比分析**

与 ViT、CAPE+FNO、CAPE+Unet、Unisolver、MoE‑POT 等基线对比，使用 nMSE 评估 10/50 步滚动误差；GNO 在所有任务上均显著下降误差（如 Navier‑Stokes nMSE 5.23×10⁻⁵），并在相同误差下实现约 2‑4 倍的推理速度提升

**⚠️ 局限性**

局限性：仅验证在预先定义的 PDE 参数族和边界类型上；对极端非线性或高维更复杂边界的泛化尚待评估；训练需大量带标注的数据，且模型仍受参数空间划分的分辨率限制

---

## 166. Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents

**arXiv ID:** 2607.22014 | [PDF](https://arxiv.org/pdf/2607.22014v1)

**作者:** Suman Navaratnarajah `[一作]` (University of Technology Nuremberg), Yuki M Asano `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MissionBench，评估冻结的多模态大语言模型在无人机任务中的使命级性能

**💡 创新点**

构建多任务、连续视角控制、闭环评估框架，并展示模型规模与零样本能力相关性

**🔧 技术方法**

利用现有多模态大语言模型、Vision‑Language导航接口、基于Unreal Engine 5的仿真环境与Cosys-AirSim

**📊 数据集**

包含120个任务，分布在五个高保真仿真场景（Neighborhood、City、Forest、Savannah、AirSimNH）和四类任务（Reporting、Inspection、Manipulation、Patrol）

**📈 对比分析**

与多种开闭源MLLM进行对比，最佳模型SR仅34.8%，与人类相比仍差约50%，模型规模越大性能越好

**⚠️ 局限性**

零样本控制仍高度困难，模型易出现提前终止、漂移和目标识别错误等失败模式

---

## 167. Ethereum NFT Smart Contracts: Knowledge-Guided Vulnerability Detection with LLM and Code Slicing

**arXiv ID:** 2607.21983 | [PDF](https://arxiv.org/pdf/2607.21983v1)

**作者:** Deyu Yang `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种结合正则表达式切片、ERC‑721知识库和DeepSeek LLM的以太坊NFT合约漏洞检测方法。

**💡 创新点**

创新点在于将高召回的候选筛选与领域约束的LLM推理相结合，利用结构感知的上下文窗口提取有针对性的代码片段，并在LLM提示中嵌入专门的知识库和固定输出模式，从而提升检测准确性并简化后处理。

**🔧 技术方法**

技术包括：① 正则表达式匹配提取潜在漏洞点；② 结构化代码切片算法（向前向后寻找函数边界）；③ 以ERC‑721为主题的知识库和明确的决策规则；④ 调用DeepSeek LLM并使用标准化输出格式；⑤ 自动化批量解析与统计。

**📊 数据集**

使用了450份真实或公开的NFT合约样本，涵盖了常见的重入、整数溢/欠以及时间戳依赖三类漏洞。

**📈 对比分析**

通过比较不同配置的正面标签率进行评估：完整配置得到97.1%正面标签率；去除外部知识库时为87.11%；完全不切片、仅用完整合约时为73.78%。实验显示代码切片和知识约束显著提升报告的正面率。

**⚠️ 局限性**

局限性包括：未提供独立标注的真实真值，正面标签率不等同于准确率；仅覆盖三类漏洞；正则表达式可能漏检或误报；切片可能遗漏跨函数或跨合约的上下文；实验缺乏可复现的完整知识库与模型配置。

---

## 168. Analyzing Toxic Behavior and Its Impact on the Mastodon Community

**arXiv ID:** 2607.21980 | [PDF](https://arxiv.org/pdf/2607.21980v1)

**作者:** Pasan Kamburugamuwa `[一作]` (Indiana University), Olga B `[通讯]` (Indiana University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用机器学习技术对2025年5月1日至3日在Mastodon联邦中25个实例的英文帖子进行毒性检测，并对各实例的平均毒性得分进行比较，研究去中心化架构对毒性内容传播与治理的影响。

**💡 创新点**

创新点在于首次将面向去中心化社交平台的毒性检测与实例间治理差异结合，系统性评估了不同实例的社区治理效果以及去中心化特性对毒性传播的实际影响。

**🔧 技术方法**

采用自然语言处理技术，主要使用Google Perspective API与Perplexity API计算文本毒性分数，并对结果进行聚合与可视化分析。

**📊 数据集**

数据集来源于Mastodon公开实例的公开帖子，覆盖25,777名用户、1,035,949条帖子，重点采集每个实例中活跃度最高的前10名用户的英文帖子。

**📈 对比分析**

通过计算每个实例的平均毒性得分，并与0.2的阈值进行对比，评估各实例的毒性水平。结果显示大多数实例平均毒性低于0.2，aethy.com的毒性得分最高但仍处于低毒性范围，整体上未发现显著差异。

**⚠️ 局限性**

局限性包括：仅限英文帖子；时间窗口短（仅3天）；样本仅为活跃用户的前10名，可能不具代表性；缺乏对各实例具体治理政策的深入分析；未提供统计显著性检验或模型性能指标。

---

## 169. On the Convergence of Stochastic Low-Rank Adaptation

**arXiv ID:** 2607.21975 | [PDF](https://arxiv.org/pdf/2607.21975v1)

**作者:** Ru Wang `[一作]` (Chinese University of Hong Kong), John C. S. Lui `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了低秩适配（LoRA）两因子优化的收敛性质，给出了确定性和随机性两种场景下的迭代复杂度上界。

**💡 创新点**

创新点包括：
- 通过轨迹分析把LoRA-GD的指数复杂度降为多项式 O(ε⁻⁴)；
- 证明普通LoRA-SGD在仅满足有限方差时可能发散；
- 提出LoRA-NSGDM（归一化+动量）和LoRA-STORM（带方差削减）两种随机算法，分别实现 O(ε⁻⁸) 与 O(ε⁻⁶) 的随机梯度复杂度；
- 结合了归一化、动量和自适应步长的技术，克服了因子化参数空间中的高阶噪声问题。

**🔧 技术方法**

使用的技术主要有：
- 轨迹/梯度轨迹分析；
- 归一化步长控制；
- 动量平滑（NSGDM）和自适应动量（STORM）；
- 方差削减（STORM）以及均方平滑假设；
- 经典的无偏方差有限的随机梯度oracle。

**📊 数据集**

实验数据集包括：
- CIFAR‑10（特征层的Logistic回归）；
- CIFAR‑10（直接训练ResNet‑18，rank‑20 LoRA）；
- Alpaca 指令微调数据集（TinyLlama‑1.1B，rank‑32 LoRA）。

**📈 对比分析**

与传统的 LoRA‑GD（含自适应步长）的比较显示：
- LoRA‑NSGDM 在Logistic回归任务上收敛速度最快，达到最低损失；
- LoRA‑NSGDM 在 ResNet‑18 任务中收敛更快更平稳；
- LoRA‑STORM 在需要两次梯度评估的任务（如 TinyLlama 微调）与 LoRA‑GD 相比，虽然理论复杂度更优，但每步成本更高；
- 总体而言，归一化+动量和方差削减策略在大多数实验中显著提升了收敛速度。

**⚠️ 局限性**

局限性包括：
- 理论复杂度仍为多项式但系数较大；
- LoRA‑NSGDM 需要固定步长 γ，若 γ 取值不当可能导致收敛缓慢；
- LoRA‑STORM 需要两次梯度评估，导致每步计算成本增加；
- 结果主要验证在三类任务，尚未在更大规模模型或不同任务中充分评估；
- 对均方平滑（MSS）等额外假设的依赖限制了方法的普适性。

---

## 170. Rethinking Layer-Wise Information Allocation for Vision Foundation Model Adaptation

**arXiv ID:** 2607.21973 | [PDF](https://arxiv.org/pdf/2607.21973v1)

**作者:** Yuqi Li `[一作]` (City College of New York, CUNY), Yingli Tian `[通讯]` (City College of New York, CUNY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Prompted Information Bottlenecks (PIB)，通过信息瓶颈原理在层级上正则化视觉提示调优（VPT），实现对冻结的视觉基础模型的参数高效适配。

**💡 创新点**

创新点在于将信息瓶颈理论应用于提示调优，设计压缩-充分性双目标和跨层路径正则化，实现层级信息的最小-充分分配，解释并改善提示调优的非单调行为与鲁棒性问题。

**🔧 技术方法**

采用的技术包括：提示式适配、信息瓶颈（IB）启发的压缩与充分性损失、跨层路径约束、可学习的层级权重路由，以及对注意力、冗余度、类间分离度等指标的层级诊断。

**📊 数据集**

在多任务视觉基准上验证：VTAB-1k（自然、专用、结构化子集）、FGVC、HTA 等 34 个数据集，使用 ViT-Base/16、Swin-Base、以及不同预训练方式（MAE、MoCo v3、ImageNet-21k）。

**📈 对比分析**

与传统 VPT、LoRA、Adapter、Bias、VPT-D、E²VPT、ViaPT 等方法对比，PIB 在所有预训练和架构上取得最优或近优的平均精度，且仅调优 0.26%~0.51% 参数；在鲁棒性测试（污染、域移、少样本）中显著减少性能下降。

**⚠️ 局限性**

局限性在于：仍需手动设定压缩-充分性权重与路径惩罚系数；对极端低资源或极少样本情形的理论保证不足；且方法主要针对 Transformer 架构，跨模型推广仍需验证。

---

## 171. ACME: A Multi-Cultural, Multi-Embodiment Social-Navigation Dataset

**arXiv ID:** 2607.21964 | [PDF](https://arxiv.org/pdf/2607.21964v1)

**作者:** Shashank Rao Marpally `[一作]` (National University of Singapore), Harold Soh `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨文化、多体态的社交导航大规模数据集ACME，包含29.35小时机器人轨迹和43.5小时人类轨迹。

**💡 创新点**

创新点在于统一收集了7种机器人形态、8国5洲、多文化交互、机器人语音交互等特征，并对BEV轨迹做人类验证和语义标签。

**🔧 技术方法**

采用ROS2记录多模态传感器（RGB、LiDAR、IMU等），利用AprilTag、Homography转换BEV像素到地面坐标，使用ByteTrack、Yolov8进行轨迹追踪，提供人类验证工具和语义标注。

**📊 数据集**

主要使用ACME自身数据作为研究基准，并与SCAND、MuSoHu、ETH/UCY、TBD等公开数据集进行对比。

**📈 对比分析**

通过社会合规性Hausdorff距离、规划失败率、ViNT/NoMAD导航模型FDE/MSE/AOE等指标，证明ACME在复杂场景下导致规划失败率翻倍、导航模型性能明显下降，验证数据集难度。

**⚠️ 局限性**

数据收集受限于不同机构的隐私规则导致RGB匿名化，缺乏全局精确定位，数据量虽大但仍缺少开放空间与完整地图，且机器人与人类的互动方式不完全一致。

---

## 172. KaPilot: LLM-Assisted Generation of Kani Specifications for Unsafe Rust Verification

**arXiv ID:** 2607.21957 | [PDF](https://arxiv.org/pdf/2607.21957v1)

**作者:** Minghua Wang `[一作]` (Ant Group), Lin Huang `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全自动多智能体框架，利用大语言模型从函数说明中提取安全需求，生成并预检 Kani 规范，循环修正并最终通过 shuffle‑and‑implication 策略选出最优的 Unsafe Rust 内存安全验证规范。

**💡 创新点**

创新点在于：①基于文档而非代码的结构化安全需求提取；②预检代理评估规范覆盖与强度，保证稳定性；③shuffle‑and‑implication 机制实现最优规范组合；④多代理生成-预检-验证循环显著提升规范质量。

**🔧 技术方法**

技术包括大语言模型（GPT‑5、DeepSeek‑v3.2、Claude‑Sonnet‑4）、Kani 边界模型检查、LLM 预检评分、vacuity 检查、循环修正与多代理协同、Shuffle‑and‑Implication 选优策略。

**📊 数据集**

使用两套基准：带官方规范的 54 条 verify‑rust‑std 标准库 unsafe 函数（gold set）和无官方规范的 97 条 Safe4U unsound 样例（unlabeled set），评估安全规范生成效果。

**📈 对比分析**

与 AutoSpec 对比，生成可验证规范的成功率提升约 21%（从 66% 至 87%），等价或更强规范比例提高 25%；同时在规范数量、语义强度、修正次数及成本上均优于 AutoSpec，显示出更高的效率与可靠性。

**⚠️ 局限性**

局限性包括：Kani 规范表达范围有限；对文档质量高度依赖；循环形式受 Kani 支持的 while 限制；缺乏完全自动化的规范质量评估；未来需改进规范 expressiveness 与无边界验证能力。

---

## 173. Leveraging External Knowledge for Historical Document Restoration via Retrieval-Augmented Large Language Models

**arXiv ID:** 2607.21936 | [PDF](https://arxiv.org/pdf/2607.21936v1)

**作者:** Gabeen Kim `[一作]` (Kangwon National University), Kyeongpil Kang `[通讯]` (Kangwon National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于检索增强生成（RAG）的历史文献修复框架ARI，利用大型语言模型（LLM）结合外部检索上下文来恢复含有损坏字符的古文档。

**💡 创新点**

创新点在于将预训练LLM的隐式历史知识与显式检索得到的相关文档融合，并通过动态掩码、少样本提示和去重策略进一步提升命名实体恢复能力。

**🔧 技术方法**

采用的技术包括LLM（如Qwen3、Gemini-2.5等）的因果语言建模、RAG检索（BM25与密集检索）、动态掩码、去重阈值优化以及专门针对汉字的标注与分词。

**📊 数据集**

使用的主要数据集为韩国朝鲜王朝的《日记》（AJD）和《皇家秘书记录》（JRS），包含数百万条记录、数亿字符，且包含约44.8%为命名实体的损坏字符。

**📈 对比分析**

通过与BERT-Res、Gemini-2.5-Pro等基线在D_Rand和D_NE两类测试集上的Top-1/Top-10准确率和nDCG评估，ARI-32B在一般字符恢复约提升30%点，在命名实体恢复上提升约20%点，整体在专家评测中胜率超过50%。

**⚠️ 局限性**

主要局限包括：训练与真实损坏文档可能存在分布差异；最大输入长度限制为4096令牌，导致部分长文被过滤；仅使用文本特征，未结合图像信息，未来需考虑多模态融合。

---

## 174. MissHyper: Restoring Clinical Synchronicity in Missingness-Guided Hypergraph Forecasting

**arXiv ID:** 2607.21922 | [PDF](https://arxiv.org/pdf/2607.21922v1)

**作者:** Mingyi Ma `[一作]` (Wuhan University), Qingxiong Tan `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了MissHyper模型，专门用于解决临床不规则多变量时间序列预测中事件初始化的瓶颈问题；

**💡 创新点**

创新点在于三项设计：在传播前恢复同时间戳的临床快照、利用缺失信息构造支持密度指示、以及通过缺失性引导的门控机制将恢复的上下文与单个事件特征融合；

**🔧 技术方法**

技术手段包括基于事件掩码的支持密度编码、同时间戳特征聚合以及门控融合，并将其嵌入标准的超图神经网络预测骨干；

**📊 数据集**

实验数据集为PhysioNet 2012、MIMIC-III和MIMIC-IV三大ICU临床时间序列基准；

**📈 对比分析**

与PrimeNet、mTAN、SeFT、NeuralFlows、CRU、GRU‑D、Raindrop、Warpformer、tPatchGNN、GraFITi及原始超图模型HyperIMTS等方法比较，MissHyper在所有数据集上实现了最低的MSE/MAE，显著提升了预测性能；

**⚠️ 局限性**

局限性包括使用固定窗口估计支持密度、仅处理数值时间序列、对超图注意力的可扩展性有限，并且需要外部验证与隐私保护来保证临床应用安全。

---

## 175. Practical Graph Optimisation and AI-Driven Models for Active Directory Security Hardening

**arXiv ID:** 2607.22009 | [PDF](https://arxiv.org/pdf/2607.22009v1)

**作者:** Huy Q. Ngo `[一作]` `[通讯]` (University of Adelaide), Huy Q. Ngo (University of Adelaide)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文研究并提出多种基于博弈论和优化的决策模型，用于对Active Directory（AD）攻击图进行优先级驱动的防御和加固。核心贡献包括：① 新颖的蜜罐/诱饵放置模型，最小化最短路径数和可达Domain Admin节点数；② 针对动态/时序AD图的防御策略，最大化最坏情况下的事件响应时间；③ 自适应优先级模型，通过与IT管理员交互，寻找最少改修即可切断低特权节点与Domain Admin节点的路径；④ 完整端到端自适应优先级模型，最小化管理员批准工作量，推广相似风险特征的边缘修复策略。

**💡 创新点**

创新点在于：① 结合蜜罐/诱饵与攻击图路径最小化的全新放置方法；② 将时序动态性纳入攻击图加固，考虑最坏响应时间；③ 引入管理员反馈的自适应优先级机制，解决可实施修复与业务约束冲突；④ 综合多种算法（数学优化、进化算法、聚类、强化学习）应对问题的NP难度。

**🔧 技术方法**

技术手段包括：博弈论建模、组合优化、线性/整数规划、进化计算（遗传算法等）、聚类算法、强化学习、人工智能辅助的评估与决策。

**📊 数据集**

实验数据主要使用：① 真实或仿真生成的AD攻击图（包含账户、计算机、组等节点与权限/漏洞边）；② 公开的AD网络案例或在实验室搭建的规模化AD环境。具体数据集未在摘要中给出，但论文中对多种规模（小、中、大）图进行了评估。

**📈 对比分析**

通过与传统基于排名的优先级方法、单纯的边缘移除策略以及已发表的近似解法进行对比。实验结果显示，所提出的优化与自适应模型在减少可达Domain Admin节点数、降低最坏响应时间、降低管理员批准次数等指标上均优于基线方法，提升幅度可达20–50%（取决于图规模与攻击路径复杂度）。

**⚠️ 局限性**

局限性包括：① 所有模型均为NP难，求解仅能得到近似或启发式解；② 对动态时序图的建模假设可能不完全符合实际业务变更速率；③ 需要管理员参与的自适应方法在高频变更环境中可能导致交互成本过高；④ 缺乏在大规模真实企业AD环境中的实测验证，模型泛化性仍待进一步评估。

---

## 176. What Clinicians Need: Designing, Developing and Evaluating an AI-Based Decision Support System for Autism Assessment

**arXiv ID:** 2607.22005 | [PDF](https://arxiv.org/pdf/2607.22005v1)

**作者:** Ulrike Schäfer `[一作]` (Freie Universität Berlin), Hanna Drimalla `[通讯]` (Bielefeld University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一款名为SIT-CARE的基于人工智能的临床决策支持系统，用于成人自闭症谱系障碍（ASC）的评估，并通过与临床医生的交互研究验证其对诊断过程的影响。

**💡 创新点**

将标准化社交交互任务（SIT）视频数据与可视化与AI推荐结合，提出了双模式（数据驱动与模型驱动）交互界面，并通过人机中心设计与临床医生共同迭代，首次在临床医生工作流程中嵌入可解释的非言语行为分析。

**🔧 技术方法**

采用OpenFace 2.2提取眼动、面部表情与语音特征，使用XGBoost与后期逻辑回归的后融合方法进行ASC分类，前端基于Next.js/React + ChakraUI，后端使用Python与可视化库生成图表。

**📊 数据集**

训练集包含325名成人（168 ASC，157 非ASC）在SIT任务中的视频和音频记录；用于评估的两位患者案例未包含在训练集中。

**📈 对比分析**

在留一法交叉验证下，系统在ASC与非ASC二分类任务中达到了74%的准确率、0.76的精确率和0.74的召回率；评估研究中显示系统可支持临床医生的决策路径并提升学习效果。

**⚠️ 局限性**

主要限制包括样本量仅7名临床医生、仅针对成人与单一文化背景的数据、仅聚焦ASC且未考虑多重诊断场景，以及模型解释与信任度需要进一步增强。

---

## 177. From Perturbation Correction to Geometry-Aware Sampling: Sharpness-Guided Equilibrium Sampling for Balanced Flat Minima in Long-Tailed Learning

**arXiv ID:** 2607.21999 | [PDF](https://arxiv.org/pdf/2607.21999v1)

**作者:** Jiaxin Deng `[一作]` (Beijing University of Technology), Junbiao Pang `[通讯]` (Beijing University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种Sharpness‑Guided Equilibrium Sampling（SGS）方法，通过动态调整采样分布，使得在长尾学习中更少出现的类别获得更多的优化机会，同时抑制在尖锐梯度区域的过度更新。

**💡 创新点**

创新点在于将采样分布视为可调节的几何控制变量，利用SAM产生的损失梯度差作为类别级尖锐度信号，并与类别出现频率结合，形成自适应采样权重；该方法不需要额外的类级扰动或反向传播，保持了与SAM相同的训练效率。

**🔧 技术方法**

核心技术包括：Sharpness‑Aware Minimization（SAM）产生的损失增益差、指数滑动平均（EMA）估计类别尖锐度、频率‑尖锐度反馈的采样权重更新、以及基于混合比例的 warm‑up 与 progressive sampling 混合策略。

**📊 数据集**

实验使用了 CIFAR‑10/100‑LT、ImageNet‑LT 以及 iNaturalist 等长尾图像分类数据集，并在 CLIP/ViT‑B/16 上进行大规模基础模型微调。

**📈 对比分析**

与现有长尾方法（如 ImbSAM、CC‑SAM、Focal‑SAM 等）和 SAM 进行比较，SGS‑SAM 在高不平衡度下显著提升尾部及整体准确率（例如 CIFAR‑100‑LT tail 上提升约 10.8 分、ImageNet‑LT tail 上提升约 10.5 分），同时训练时间几乎不变（≈1.02× SAM）。

**⚠️ 局限性**

限制在于需要可靠的类别级尖锐度估计，warm‑up 比例和尖锐度指数等超参数在不同不平衡比例下仍需细调；此外，在极度稀缺类别下 EMA 平滑可能不足以捕捉真实尖锐度变化。

---

## 178. "Go Home Copilot, You're Drunk": Understanding Developer Responses to Agent-Generated Code Review Comments

**arXiv ID:** 2607.21997 | [PDF](https://arxiv.org/pdf/2607.21997v1)

**作者:** Shamse Tasnim Cynthia `[一作]` (University of Saskatchewan), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 341 个 Python 开源仓库中 54,713 条 AI 代码评审评论进行大规模经验研究，探讨开发者如何响应、解决率差异、经验对接受度的影响以及未解决讨论的模式。

**💡 创新点**

首次在大规模真实数据上系统评估 AI 评审评论的效果；引入开发者经验分组、讨论主题卡片排序和多变量回归分析，揭示了可操作性、代码建议与解释类型对评论可用性的影响。

**🔧 技术方法**

利用 GitHub GraphQL/REST API 抽取评论，采用 Llama‑3.1‑70B 进行自动化分类与解释标注，使用卡片排序分析讨论模式，应用统计检验（卡方、Mann‑Whitney）和逻辑回归建模评估特征。

**📊 数据集**

54,713 条由 Copilot、Cursor、Codex 生成的代码评审评论，来源于 341 个满足活跃度阈值的 Python 仓库（共 342 个），共 54,791 条（除去少量 Claude、Devin）。

**📈 对比分析**

通过比较不同 Agent 的解决率（Copilot 72.9%，Cursor 67.2%，Codex 54.8%），并用回归得到 inline code suggestion 的 OR≈1.62；统计检验显示代码建议、规则/示例/利益解释显著提升可用性，AUC≈0.58 表示模型解释力有限。

**⚠️ 局限性**

仅限 Python + GitHub；小样本导致 Devin/Claude 无法单独分析；Agent 标识依赖登录名模式，可能漏检；LLM 自动标注虽达 κ≈0.74/ J≈0.90，但仍有误差；将“已解决”作为可用性 proxy 可能产生噪声；模型解释力低，未能捕捉项目上下文与权限等因素。

---

## 179. Smart Contract Tells: Aircraft Maintenance Records Are Now Trustworthy

**arXiv ID:** 2607.21989 | [PDF](https://arxiv.org/pdf/2607.21989v1)

**作者:** Woosuk Choi `[一作]` (Chung-Ang University), Seungmo Kim `[通讯]` (Georgia Southern University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于区块链与IPFS的双层维护记录管理系统，采用可验证的CID锚定实现记录完整性与可追溯性；

**💡 创新点**

将维护记录完整性与飞机残值、审计成本等经济指标关联，构建动态残值与审计成本模型，并验证其经济效益；

**🔧 技术方法**

使用区块链+智能合约、IPFS（Pinata）、BNB Smart Chain测试网络、SHA‑256哈希以及FAA AC 120‑78B等行业标准；

**📊 数据集**

合成的维护日志CSV（符合FAA AC 120‑78B/ATA Spec 2000标准），并使用Gemini 3‑Flash生成的样本数据；

**📈 对比分析**

在BNB测试网对7KB文件进行实验，CID锚定后gas消耗降低93.9%；经济模型显示审计成本可节省约90%，残值差异约1.5%对应B787等飞机约100万美元；

**⚠️ 局限性**

存在oracle问题，无法保证数据真实性；模型参数未通过实地运营数据验证；恢复系数δ等关键指标缺乏实证支持。

---

## 180. Analysing Self-Harm Representations in Language Models: a Cross-Architecture Study

**arXiv ID:** 2607.21988 | [PDF](https://arxiv.org/pdf/2607.21988v1)

**作者:** Luis Espinosa-Anke `[一作]` (Cardiff University), Carla Perez-Almendros `[通讯]` (Cardiff University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLM）在检测自伤内容时内部表示的层级特征，评估了不同层的线性可分性与对比方向的稳定性。

**💡 创新点**

发现自伤信息在网络最后3–7%的层高度聚集，对比方向在网络深度上不稳定，且最高AUC的线性探针并非最具几何可分性的方向。

**🔧 技术方法**

采用线性探针、对比方向提取、Cohen's d效应量计算、余弦相似度分析，并通过TransformerLens获取内部激活。

**📊 数据集**

使用X‑Sensitive（多标签自伤与其他敏感类）和SH‑Detection（二分类自伤）两份公开数据集。

**📈 对比分析**

通过在各层训练线性探针并计算ROC‑AUC进行比较，最优层位于93–97%网络深度，X‑Sensitive AUC范围0.703–0.817，SH‑Detection AUC范围0.923–0.972，Gemma‑3‑4B表现最佳但几何分离度低。

**⚠️ 局限性**

局限性包括数据标注质量差异导致结果波动、仅评估至4‑B级模型、缺乏对更大规模模型和其他敏感内容类别的验证。

---

## 181. J-CoT: Chain-of-Thought in J-Space

**arXiv ID:** 2607.21981 | [PDF](https://arxiv.org/pdf/2607.21981v1)

**作者:** Junde Wu `[一作]` (University of Oxford), Jiazhen Pan `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 J-CoT 递归推理框架，在语言模型内部使用词表索引的 J-space 作为中间状态，既不需要完整的自然语言表述，也不必传递整个稠密隐藏状态。

**💡 创新点**

创新点在于：1）定义了模型原生的 J-space 作为跨层共享的词索引坐标系统；2）引入了 J-thought 递归接口，用非解码的词向量系数传递信息；3）结合自适应读取门和多载体 carrier 结构，兼顾稠密计算与语言约束。

**🔧 技术方法**

使用技术包括：基于 Jacobian 的 J-space 字典构建、非负弹性网络（elastic‑net）系数提取、读写门控制、carrier 位置与自注意力、适应性停止策略、以及在预训练模型上进行的推理适配微调。

**📊 数据集**

实验数据集涵盖算术与科学推理（GSM8K、MATH‑500、AIME 2024、GPQA‑Diamond）、代码生成与执行（HumanEval+、MBPP+、LiveCodeBench、CRUXEval）以及推理路径分析（ProsQA），并在多规模模型（7B–405B）上进行缩放实验。

**📈 对比分析**

与显式链式思考（CoT、PS+）、密集潜在推理（Coconut、CODI、SIM‑Coconut）及基准 GPT 版本进行对比；在所有评测任务上 J‑CoT‑Train 取得最佳整体平均成绩约 50.2%，比最强基线提升 2.7 个百分点，单项任务提升 2–3 个百分点；模型规模和推理循环深度共同提升性能。

**⚠️ 局限性**

局限性包括：① 需要预先计算且固定的 J‑space 字典，无法动态扩展词表；② 递归状态不包含可读的语言解释，难以直接解释推理过程；③ 读取/写入层的选择对性能敏感；④ 对极长推理链或需要多层次抽象的任务可能仍受限。

---

## 182. Ground Truth First: A Longitudinal Evaluation Instrument for Agent Memory, and the Tenure Crossover in Memory-Architecture Rankings

**arXiv ID:** 2607.21962 | [PDF](https://arxiv.org/pdf/2607.21962v1)

**作者:** Quentin Spencer `[一作]` `[通讯]` (Independent Researcher), Quentin Spencer (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过“先生成真值，再生成文本”的方式构建了全合成、可验证的对话记忆基准，并在该基准上长时间测试多种记忆架构；

**💡 创新点**

创新点在于引入了事实有效期、信任级别、注入探针等细粒度属性，并揭示了记忆系统随历史长度的排名反转、写入质量对下游效果的强相关性以及保留来源信息可实现注入抵抗；

**🔧 技术方法**

主要技术包括：基于种子确定的生命脚本采样器、LLM渲染器、真值核对器、版本化判定器；记忆架构包含预算化 curated‑map、向量检索、属性图、以及分层 hybrid‑v2；

**📊 数据集**

使用的数据集为 380 个经人工审核的问答（15 类），共 6 位长期用户、14 位短期用户，所有对话和邮件均为完全合成；

**📈 对比分析**

比较方法采用统一答复模型、固定判定器、三次随机复制，测量准确率、读写代价；短期内 hybrid‑v2 最高 96.8%，向量 91.0%；长期（9 周）图结构与层级混合在 week 9 领先 93.2% 与 90.4%，但无判定器无关的绝对优胜者；

**⚠️ 局限性**

局限性包括：完全合成语料缺乏真实世界复杂性；仅测试单一答复模型；hybrid‑v2 设计时已针对短期数据调优；注入探针仅为模板式非自适应攻击；评测仅覆盖固定问答模板，难以推广到更开放式任务。

---

## 183. MA-DAR: Manifold-Aligned Dynamic Adaptive Routing for Continual Temporal Knowledge Graph Reasoning

**arXiv ID:** 2607.21949 | [PDF](https://arxiv.org/pdf/2607.21949v1)

**作者:** Xiangjun Shi `[一作]` (University of Electronic Science and Technology of China), Shang Liu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MA‑DAR 框架，解决持续 TKG 推理中历史与当前表示的冲突。

**💡 创新点**

创新点在于通过流形对齐、动态门控和极化正则三步路由，有效消除范数主导和语义模糊。

**🔧 技术方法**

采用轻量级插件形式，结合 Manifold Alignment、Dynamic Gating、Polarization Penalty，并可与 RE‑GCN、TiRGN、LogCL 等基线编码器以及经验或生成 replay 集成。

**📊 数据集**

实验使用 ICEWS14s、ICEWS18、ICEWS05‑15、GDELT 四大公开 TKG 基准。

**📈 对比分析**

与多种基线（FT、ER、DGAR、LogCL 等）对比，MA‑DAR 在 MRR、H@1、H@10 等指标上提升 20–30% 以上，LogCL+MA‑DAR 达到 66.2% MRR。

**⚠️ 局限性**

局限性包括仅测试固定 replay 方式、门控超参数调优敏感，以及未评估大规模或多语言场景的鲁棒性。

---

## 184. LatentFlow: Visual Analytics for Latent Space Analysis in Molecular Graph Neural Networks

**arXiv ID:** 2607.21941 | [PDF](https://arxiv.org/pdf/2607.21941v1)

**作者:** Shiyi Liu `[一作]` (Arizona State University), Ross Maciejewski `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了交互式可视化系统LatentFlow，用于追踪分子图神经网络潜在空间在层与训练阶段的演化，并帮助化学家从宏观到分子级别理解模型的结构-性质关系。

**💡 创新点**

创新点在于将多维潜在空间的演化可视化为修改的 Sankey 图，并结合散点、Top‑k、成员层级视图和结构编辑器，使得用户能够同时观察聚类变迁、属性分布与化学子结构的对应关系。

**🔧 技术方法**

使用图神经网络（Chemprop、DimeNet）、多种聚类方法（kNN、AHC、HDBSCAN、PG‑Means）、降维技术（PCA、t‑SNE、UMAP）、相似性度量（Tanimoto）、前端React+D3交互框架以及后端Flask。

**📊 数据集**

在 MoleculeNet ESOL（溶解度）数据集和 Catalyst Selectivity 数据集上训练 Chemprop 与 DimeNet 模型。

**📈 对比分析**

通过与手工标签（芳烃/烷基）以及真实属性对比，展示潜在空间从结构分类到属性驱动聚类的迁移，并验证构象增强模型在聚类一致性和预测误差方面优于基线模型。

**⚠️ 局限性**

局限在于只能显示两维潜在空间维度的演化，聚类与降维方案需人工调参，聚类数目过多导致可视化拥挤，并缺乏自动化异常检测和解释生成。

---

## 185. TG-Diff: Coupling Discrete Topology Diffusion and Topology-conditioned Geometry Diffusions for B-Rep Generation

**arXiv ID:** 2607.21928 | [PDF](https://arxiv.org/pdf/2607.21928v1)

**作者:** MingZe Sun `[一作]` (University of Chinese Academy of Sciences), Peter Wonka `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种轻量化的两阶段扩散式B‑rep生成框架，通过表面为中心的拓扑与几何解耦来实现CAD模型的自动生成。

**💡 创新点**

创新点在于将拓扑仅表示为面间相邻关系，利用离散扩散模型生成拓扑，再用拓扑调制的轻量DiT生成面几何，从而显著提升了结构完整性和生成效率。

**🔧 技术方法**

核心技术包括离散扩散概率模型(D3PM)、拓扑感知的VAE+GNN、条件表面潜在扩散模型和基于DiT的自注意力网络。

**📊 数据集**

使用DeepCAD、ABC和OnShape家具数据集进行训练与评估。

**📈 对比分析**

在DeepCAD和ABC数据集上，生成质量（COV、MMD、JSD）和有效率显著优于DTGBrepGen、BrepGen等基准，同时参数量仅82.18M，推理速度最快。

**⚠️ 局限性**

主要局限在于基于扩展面交叉的后处理对极度平行或近切面易失效，且OCC交叉核对精度要求高，导致某些复杂几何的生成仍受限。

---

## 186. RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention

**arXiv ID:** 2607.21927 | [PDF](https://arxiv.org/pdf/2607.21927v1)

**作者:** Anderson R. Santos `[一作]` `[通讯]` (Federal University of Uberlandia), Anderson R. Santos (Federal University of Uberlandia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个模型无关的稀疏注意力推理框架 RIS，能够在不改动模型权重的前提下，将注意力复杂度从 O(N²) 降至 O(N log N)，从而在 65,536 token 的长上下文上实现可行推理；

**💡 创新点**

其创新点在于结合稀疏随机几何采样与结构化块 clique、PFUS 预融合 softmax、动态 RoPE 缩放等技术，构建了高效可插拔的稀疏注意力内核，并证明稀疏注意力可作为正则化器，甚至提升全注意力性能；

**🔧 技术方法**

主要技术包括：Stochastic 与 Structural 两种采样模式；PFUS 预融合 softmax；动态 RoPE 缩放；流式稀疏几何掩码生成；CPU 无 GPU 的高效实现；以及 YaRN 等 RoPE 扩展方法；

**📊 数据集**

实验数据集涵盖 Qwen2‑1.5B‑Instruct 与 TinyLlama‑1.1B 的 32‑题与 64‑题问答基准，以及由四篇科学论文（Acin、aom、genppi、meta）拼接而成的长上下文；

**📈 对比分析**

通过与全注意力基准对比并以零上下文基准为下限，测量准确率回收率；在 32k 上下文下，RIS‑Stochastic 1% 密度 70 种子获得 75.00% 准确率（超过密集 71.88%）；在 65k 上下文下，YaRN 扩展下 RIS‑Structural 1% 密度 10 种子达 65.62%（比零基 51.56% 提升 14%）且显著优于线性 RoPE；TinyLlama 在极端长上下文失效，验证位置编码限制；

**⚠️ 局限性**

主要局限包括：受限于模型原生位置编码，RIS 在超出训练窗口的极端长上下文（如 TinyLlama 4× 以上）失效；密度与种子数需经验调参；仅在 CPU 无 GPU 环境验证，GPU 加速效果未评估；在极稀疏下结构模式对全局信息捕获有限。

---

## 187. Cross-country structure of a sexual contact network derived from a commercial-sex review platform

**arXiv ID:** 2607.21972 | [PDF](https://arxiv.org/pdf/2607.21972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 188. Cross-Domain Off-Policy Evaluation and Learning for Contextual Bandits

**arXiv ID:** 2607.22012 | [PDF](https://arxiv.org/pdf/2607.22012v1)

**作者:** Yuta Natsubori `[一作]` (Hakuhodo DY Holdings), Yuta Saito `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了跨域离线策略评估与学习（Cross-Domain OPE/L）框架，并给出了COPE估计器及其扩展的COPE-PG梯度方法，用于在目标域中评估和学习新策略。

**💡 创新点**

创新点在于将奖励函数拆解为域聚类效应与域特定效应，利用多源数据通过多重重要性加权实现无偏估计，能够处理确定性日志、缺乏探索以及出现新动作等挑战。

**🔧 技术方法**

采用了多重重要性加权、奖励回归（随机森林回归+交叉拟合）、密度比估计、聚类（基于经验平均奖励的嵌入）以及梯度估计等技术。

**📊 数据集**

使用真实视频推荐数据集KuaiRec（1,411用户、3,327条目、4,676,570次交互）作为实验数据。

**📈 对比分析**

通过与IPS、DR、IPS-ALL、DR-ALL、DM等基线方法在MSE、偏差、方差以及离线学习中策略价值等指标上的比较，COPE在新动作比例、确定性日志用户比例以及少样本情形下均显著优于基线，表现出更低误差和更高策略价值。

**⚠️ 局限性**

局限性包括：需要预先聚类域（目前使用经验平均奖励的经验式聚类，可能对聚类质量敏感）；当目标域样本极少且聚类误差大时仍可能产生偏差；方法主要针对上下文-动作-奖励的离线情境，未涵盖多臂或持续时间等更复杂的情景。

---

## 189. Energy Manifold Natural Gradient Descent: Riemannian Optimization for Neural PDE Solvers

**arXiv ID:** 2607.22004 | [PDF](https://arxiv.org/pdf/2607.22004v1)

**作者:** Zhangyong Liang `[一作]` (Tianjin University), Huanhuan Gao `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种能在参数约束为 Riemannian manifold 的神经 PDE 求解器中使用的能量自然梯度下降（EMNGD）算法；

**💡 创新点**

创新点在于将能量自然梯度的张量结构与参数流形的切空间相结合，形成在可行方向上的能量度量二次模型，并通过重traction保持约束；

**🔧 技术方法**

使用了能量梯度张量、切空间投影、Woodbury 公式、Nyström 抽样以及 Armijo 回溯线搜索等技术；

**📊 数据集**

在多种 PDE 基准上进行实验，主要包括 1D/2D/5D Poisson 方程、1D 热方程等，使用随机采样点和自定义网格；

**📈 对比分析**

与传统 SGD、Adam、BFGS、KFAC、ENGD、SPRING 等方法比较，EMNGD 在相同迭代次数或时间内取得更低的 L² 相对误差，收敛速度更快；

**⚠️ 局限性**

局限在于 Woodbury 方案对样本数量 N 的存储与计算开销，Nyström 近似需权衡秩与精度，且在残差采样较大或高维残差空间时需要更高的算力或矩阵无关求解器。

---

## 190. Medical-Checklist: Assessing the Comprehension of Medical Images by Multimodal Models

**arXiv ID:** 2607.21998 | [PDF](https://arxiv.org/pdf/2607.21998v1)

**作者:** Bannapol Limanond `[一作]`, Takayuki Okatani `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种新的评测基准——Medical-Checklist，用二选一的方式评估医学多模态模型对图像与文本的跨模态理解。

**💡 创新点**

创新点在于将错误说明句子仅通过随机替换单一医学术语生成，从而考察模型是否能识别医学上不合理的描述，并通过保持二元答案均衡来消除数据偏倚。

**🔧 技术方法**

研究使用了SciBERT和SciSpacy进行医学术语抽取、UMLS映射、文本替换，构建了二元测试；评测采用图像-文本相似度（CLIP/PMC-CLIP）、匹配损失（M3AE）以及基于提示的多模态LLM（LLaVA-Med）等技术。

**📊 数据集**

数据集基于MedICat和ROCO的300k+图文对，生成65,464条二选一测试，覆盖15个医学概念类别，涉及6,293个独立医学术语。

**📈 对比分析**

实验对四个顶尖模型（MedCLIP、PMC-CLIP、M3AE、LLaVA-Med）进行评估，准确率仅略高于50%，最佳为M3AE 61.76%；说明在传统VQA任务表现优异的模型在此基准上仍显著不足。

**⚠️ 局限性**

主要局限在于当前多模态模型缺乏真正的医学概念理解，易受视觉细节偏差或文本不合理性误导，评测对模型的跨模态推理能力仍不充分，未来需开发更具语义一致性的训练与评测方法。

---

## 191. On Improving Faithfulness of Podcasts from Documents

**arXiv ID:** 2607.21961 | [PDF](https://arxiv.org/pdf/2607.21961v1)

**作者:** Soumya Dutta `[一作]` (Indian Institute of Science), Pannaga Shivaswamy `[通讯]` (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了基于文档的播客生成中可信度问题，构建了跨五个领域的1500篇文档数据集，并评估了多款大型语言模型（LLM）生成文本的真实性。

**💡 创新点**

创新点包括提出首个逐轮LLM‑as‑a‑judge可信度评估框架，以及设计通用的 catch‑n‑repair 机制，用以检测并修正不符合源文档的对话回合。

**🔧 技术方法**

技术手段主要包括：LLM‑as‑a‑judge 的 Likert 评分、使用 LoRA 微调的检测模型，以及在检测到不可信回合时重新提示 LLM 进行重写。

**📊 数据集**

使用的数据集为 Doc‑to‑Podcast，包含1500篇跨学术、法律、政策、金融和医疗领域的文档，平均长度在2–50页。

**📈 对比分析**

在多款LLM上进行比较，GPT‑4o 在评估器（GPT‑4o）下获得最高平均可信度 3.89；catch‑n‑repair 在所有模型和领域中平均提升 5–12% 的可信度；人工标注的 Krippendorffα 为 0.69，评估器与人工的 Pearson 相关系数达 0.63。

**⚠️ 局限性**

局限性包括：仅使用英文文档，评估器对提示依赖较强，catch‑n‑repair 可能因误判导致漏修或误修，以及未考虑音频生成层面的可信度。

---

## 192. Adaptive Undulatory Locomotion of Snake-like Robots in Dynamic Viscous Environments via Deep Reinforcement Learning

**arXiv ID:** 2607.21960 | [PDF](https://arxiv.org/pdf/2607.21960v1)

**作者:** Tsuyoshi Kimoto `[一作]` (Osaka Metropolitan University), Takashi Iwasa `[通讯]` (Osaka Metropolitan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过深度强化学习（DRL）实现蛇形机器人在动态粘性环境中自适应的无波浪运动，克服传统固定步态在未知流体黏度下的性能限制。

**💡 创新点**

创新点在于将任务建模为部分可观测马尔可夫决策过程（POMDP），利用异质演员-评论家框架在仿真中用特权信息（流体黏度与头部线速度）训练教师策略，再通过知识蒸馏得到仅依赖本体感知的学生策略，实现无需外部流体测量的实时自适应步态。

**🔧 技术方法**

使用的技术包括深度强化学习中的PPO算法、两阶段教师-学生训练、异质演员-评论家架构、经验回放、LSTM时序网络、以及自定义流体动力学模型（附加质量、压力阻力、粘性阻力）与Isaac Sim仿真。

**📊 数据集**

数据集主要是仿真产生的，随机化粘度范围从10⁻⁷到10⁻² m²/s，并在水、油I、油II三种实际流体物理参数下评估。

**📈 对比分析**

与传统单目标优化（SOO）和多目标优化（MOO）基准相比，DRL方法在所有流体环境下的推进速度和能耗（CoT）均优于预设的正弦波步态以及最优单目标步态，并能超过理想的适应性正弦波（Simple Sum MOO），显示出更高的自适应性能。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未完成真实硬件的Sim-to-Real迁移；依赖仿真中的流体模型，实际非线性防水层和执行器延迟等硬件特性未被充分捕捉；同时在不同关节数量和更复杂异质环境下的进一步实验尚待开展。

---

## 193. SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders

**arXiv ID:** 2607.21951 | [PDF](https://arxiv.org/pdf/2607.21951v1)

**作者:** Evan Caville `[一作]` (University of Queensland), Marius Portmann `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

SIREN通过在固定的检索源集合中，使用迭代的网页内容编辑技术来操纵大型语言模型生成的推荐列表排名。

**💡 创新点**

创新点在于将PAIR的攻击–评判循环迁移到Web‑RAG推荐器，构建可重现的自定义RAG回放平台，并系统评估23种内容污染技术的竞争性排名提升效果。

**🔧 技术方法**

采用元素索引式离线编辑、可编程的内容变换、基于排名的评判器、循环反馈优化以及Anthropic Claude模型的Web工具进行实验。

**📊 数据集**

使用8个匿名查询–模型上下文（包含Haiku与Sonnet两大模型）收集的网页集（包括S‑1P、S‑BLOG、S‑REVIEW等）作为实验数据集。

**📈 对比分析**

通过全扫（23技术）与精选8技术的对比、Haiku与Sonnet的性能比较、交叉模型重放以及@1、@3、@b等指标评估，发现整体成功率在50%–70%之间，平均需5–6次迭代即可将目标提升至排名第一。

**⚠️ 局限性**

局限性包括：仅评估两款模型与有限的8个上下文，固定上下文回放不等价于实时检索流程，缺乏对模型内部机制的分析，跨模型迁移受采样与提示配置影响，且实验规模不足以覆盖更广泛的行业与查询场景。

---

## 194. Multi-Agent Debate and Visual Information Extraction for SeePhys Pro: A 1st-Place Technical Report from ICML 2026 AI4Math Track 3 Challenge

**arXiv ID:** 2607.21946 | [PDF](https://arxiv.org/pdf/2607.21946v1)

**作者:** Jiseok Kwak `[一作]` (KAIST), Il-chul Moon `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个两阶段的视觉物理问题求解管线：首先通过视觉信息提取（裁剪、OCR、SVG结构化）把图像内容转换为文本或结构化格式；随后使用多代理辩论（MAD-M²）由三种异构 LLM 共同推理，并通过值感知的 Canonicalizer 进行答案汇总。

**💡 创新点**

创新点在于：① 将视觉信息提取与多代理推理两大模块解耦并按难度层级动态调整；② 通过结构化 SVG 代替传统文本字幕更精确地保留图像几何与符号关系；③ 在多代理辩论中发现答案一致性（majority）比额外辩论回合更能提升准确率，且可通过轻量 Canonicalizer 替代昂贵的高级模型做 tie‑breaker。

**🔧 技术方法**

使用的技术包括：图像裁剪与锐化、OCR+LaTeX 转写、LLM 生成 SVG 与自然语言转录、MAD‑M² 多代理辩论框架、Canonicalizer 与 Pruner 的记忆掩码与答案标准化。

**📊 数据集**

数据集为 ICML 2026 AI4Math Workshop 的 SeePhys Pro Challenge Track 3，包含 5 个视觉难度层级（L1 文本、L2 结构、L3 文本+图、L4 图像、L5 照片），共计 830 题（公开 20% 及私有 80%）。

**📈 对比分析**

与单模型 baseline（GPT‑5.5）对比，整体准确率从 0.643 提升到 0.802；在各层级分别提升显著，尤其 L4（图像）由 0.540 提升到 0.775，L5 照片由 0.670 提升到 0.933，最终在公开和私有排行榜均夺得第一名。

**⚠️ 局限性**

局限性包括：仅在官方排行榜评测，缺乏本地 ground truth；高级模型仅做 tie‑breaker 而未作为完整系统对比；未完成完整的视觉前端组合因子实验；商业 LLM 无固定种子，结果不完全可复现。

---

## 195. VisionPulse: A Virtual Reality System Enabling Accessible Discovery and Navigation for Blind and Low Vision Users

**arXiv ID:** 2607.21944 | [PDF](https://arxiv.org/pdf/2607.21944v1)

**作者:** Samuel Martin `[一作]` (Arizona State University), Hasti Seifi `[通讯]` (Arizona State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 VisionPulse——一个支持盲人和低视力用户通过自然头部与手部动作、语音、音频、触觉反馈进行自由探索和导航的虚拟现实系统。

**💡 创新点**

创新点在于将可探索的区域划分为 region/section/object 结构，利用头部方向驱动的逐步显现与发现菜单、响应式音频 beacons 根据头部朝向动态调音以及手柄方向与距离结合的 haptic 指引，实现了无视觉依赖的自由探索与精准定位。

**🔧 技术方法**

使用 Unity 引擎开发，结合头部相机视锥检测、NavMesh 生成 waypoint、自定义脚本实现发现菜单、文本转语音（TTS）、3D 立体音频与控制器振动反馈。

**📊 数据集**

研究数据集为 12 名盲/低视力参与者在六个虚拟场景（两艘太空船、两座花园、两座房屋）中完成寻找钥匙-门关卡的任务，记录路径长度、完成时间、交互日志等。

**📈 对比分析**

与传统预建菜单+静态音频 beacon 的基线相比，用户对发现式探索和带触觉的响应式音频更倾向，任务完成时间和工作量无显著差异，但发现模式导致路径更长（平均 148m vs 97m）；用户满意度与参与感略高。

**⚠️ 局限性**

限制包括：仅支持静态目标；TTS 可能与环境音冲突导致认知负荷；手柄与头部方向不匹配导致定位混乱；菜单规模扩展性差；样本量小、仅单次实验，缺乏长期使用和动态目标情境验证。

---

## 196. Small Vision-Language Models Know When They Are Wrong But Cannot Say So: A Two-Model Study of Stated versus Internal Confidence Under Realistic Image Degradation

**arXiv ID:** 2607.22034 | [PDF](https://arxiv.org/pdf/2607.22034v1)

**作者:** M M Asif Ferdous `[一作]` `[通讯]` (Independent Researcher), M M Asif Ferdous (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估两种小型开源视觉语言模型（Qwen2‑VL‑2B‑Instruct 与 SmolVLM‑Instruct）在六类真实相机降质（压缩、模糊、欠曝、眩光、旋转、重新采样）下的鲁棒性，并比较其两种不确定性估计：模型自述的置信度和内部平均 token 概率。

**💡 创新点**

首次在相同预测上对“口头置信度”和“内部 token 概率”进行系统对比，揭示两者在小型模型中的显著差异；并指出即使内部概率在大多数降质下可用作退避信号，但在严重欠曝时两者均失效。

**🔧 技术方法**

使用自然语言生成（greedy decoding）获取答案与口头置信度；在生成过程中记录每步 token 的概率以计算平均 token 概率；采用 AUROC、准确率、ECE 等指标评估置信度表现；对降质进行分级控制，使用 bootstrap 置信区间。

**📊 数据集**

Food101 数据集的 100 条测试样本，构造四选一多项选择问题；对每个样本应用三重严重度的六类降质，共 3,800 条预测。

**📈 对比分析**

对比方法：在相同的 3,800 条预测中分别计算口头置信度和内部 token 概率的 AUROC 及准确率。结果显示 Qwen2‑VL‑2B 的内部概率 AUROC 在 0.92–0.99 之间，口头置信度仅在 0.39–0.75 之间（大多接近 0.5），而 SmolVLM 内部概率在 0.54–0.92 之间。两种模型在大多数降质下保持高准确率，但在严重欠曝时准确率骤降，且两种置信度信号几乎不变。

**⚠️ 局限性**

限制：仅测试了两种约 2B 参数的模型，未覆盖更大规模或不同结构的 VLM；数据集仅为 Food101 的四选一任务，缺乏开放式问答场景；口头置信度的提取依赖单一 prompt 模板，可能不具普适性；内部 token 概率在极低光照下失效，说明单一置信度信号不足；实验规模有限（每类约 100 条样本），置信度统计不够稳健。

---

## 197. Projection Pursuit CPCANet for Domain Generalization

**arXiv ID:** 2607.22117 | [PDF](https://arxiv.org/pdf/2607.22117v1)

**作者:** Yu-Hsi Chen `[一作]` (University of Melbourne), Abd-Krim Seghouane `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PP-CPCANet，解决域泛化中CPCA的协方差秩缺陷，通过投影追踪实现无协方差的全局正交基学习。

**💡 创新点**

创新点在于：1) 用Stiefel曼哈顿优化与Cayley变换构造全局正交基，避免批量协方差估计；2) 采用去中位数的L1投影追踪分散目标，提供稳健且稠密的梯度；3) 单层瓶颈结构即可获得最佳性能，减少模型复杂度。

**🔧 技术方法**

使用技术包括：Stiefel manifold优化、Cayley变换、投影追踪（Projection Pursuit）、去中位数L1分散、域引导特征调制（Domain‑Guided Feature Modulation）以及单层/多层瓶颈设计。

**📊 数据集**

数据集：PACS、VLCS、OfficeHome、TerraIncognita 四大域泛化基准。

**📈 对比分析**

与CPCANet及多种传统DG方法（ERM、CORAL、CDANN、ARM等）对比，PP‑CPCANet在多数基准上实现SOTA或接近SOTA，并在GPU内存占用和训练时间上表现更优，训练过程更稳定。

**⚠️ 局限性**

局限性：最佳效果仅在单层（T=1）配置，增加深度不再提升；对极端小批量或高度不平衡域的鲁棒性尚未完全验证。

---

## 198. A Leakage-Free Stacked Ensemble Method for Multiclass Classification

**arXiv ID:** 2607.22081 | [PDF](https://arxiv.org/pdf/2607.22081v1)

**作者:** S. P. Sharmila `[一作]`, Aruna Tiwari `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种泄漏自由的多分类堆叠框架LFS-FRAME，结合Kolmogorov–Arnold网络(KAN)与XGBoost进行功能学习与规则学习；

**💡 创新点**

创新点在于采用严格的OOF（交叉验证）堆叠策略消除信息泄漏，并将连续函数逼近的KAN与梯度提升树相结合，实现两种互补学习器的功能融合；

**🔧 技术方法**

使用的技术包括Kolmogorov–Arnold网络、XGBoost、严格的OOF堆叠、以及多项式元学习器来融合基模型概率输出；

**📊 数据集**

实验数据集为CIC-MalMem-2022及其扩增版EnhancedCIC（包含16类恶意软件和增强样本）；

**📈 对比分析**

与HyStack、Hybrid CNN–BiLSTM、SMOTE-DNN、RF等基线方法对比，LFS-FRAME在4类任务中取得89.85%准确率，在16类任务中取得81.74%准确率，明显优于所有对比方法；

**⚠️ 局限性**

局限性包括：相比传统堆叠需要额外的OOF训练开销；对极端类别不平衡仍需进一步改进；在非恶意软件领域的泛化能力尚待验证。

---

## 199. Benchmarking Fine-tuning and Retrieval Strategies for a Multimodal Language Model on the NRC Reactor Operator Licensing Examination

**arXiv ID:** 2607.22067 | [PDF](https://arxiv.org/pdf/2607.22067v1)

**作者:** Isak Hwang `[一作]` (Hanyang University), Yoon Pyo Lee `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 31B Gemma 4 多模态模型在美国 NRC 核电机组操作员执照考试中的表现，并比较不同微调与检索配置。

**💡 创新点**

首次结合链式思维蒸馏监督微调、检索增广微调与结构化/固定窗口分块，揭示分块策略随模型训练状态的逆转。

**🔧 技术方法**

使用 LoRA 参数高效微调、BM25 词检索、结构化与固定窗口分块、Gemma 4 31B 多模态推理与图像+文本处理。

**📊 数据集**

基于 NRC 泛基本测试题库、2015–2021 年 12 份考试题目以及 DOE Fundamentals Handbook 7 卷检索语料。

**📈 对比分析**

对 14 份考试逐卷按 80% 通过阈值计分，最佳配置（SFT + 固定窗口检索）在 8/14 卷通过，整体池化准确率 79.66%（接近阈值），未微调模型未通过任何卷。

**⚠️ 局限性**

仅评估泛基本题，未覆盖法规、现场或紧急程序；使用单一检索方式；未量化推理可信度与幻觉；分块逆转与 RAFT 差异需进一步验证。

---

## 200. Benchmarking Text-to-SQL under Role-Based Access Control

**arXiv ID:** 2607.22115 | [PDF](https://arxiv.org/pdf/2607.22115v1)

**作者:** Yang Fei `[一作]` (National University of Singapore), Xiaokui Xiao `[通讯]` (National University of Singapore)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个文本到SQL在角色基于访问控制（RBAC）环境下的评测框架，自动为现有基准数据集生成可执行的角色与权限，并定义了安全与实用性评价指标。

**💡 创新点**

创新点在于：① 利用LLM加人工审核生成语义合理、细粒度的RBAC策略；② 提出了六分类结果空间与Safe-EX/AC-F1等专门评测指标；③ 在Spider、BIRD、LiveSQLBench三大基准上扩展生成完整的RBAC增强版本。

**🔧 技术方法**

使用的技术包括：LLM（GPT、Gemma、Qwen等）进行角色合成与推理；SQLGlot解析SQL以提取权限；结构化评测流程与人审校机制；安全性指标设计与多采样评估协议。

**📊 数据集**

使用数据集：原始文本到SQL基准Spider、BIRD、LiveSQLBench，并在此基础上生成RBAC增强版本（覆盖53个数据库、399张表、3,353列，共21,502个标注实例）。

**📈 对比分析**

评测方法：在不受限环境下与RBAC环境下分别计算EX与Safe-EX，并通过AC-F1、Violation Rate、Over‑Refusal Rate等指标比较。结果显示：商业模型在安全性上更优秀但仍存在违规；开放模型违规率高；提高SQL准确率不必然提升RBAC合规；SFT与少量示例提升有限。

**⚠️ 局限性**

局限性：仅覆盖列/操作级RBAC，未考虑行级或推断泄漏；合成角色对不同数据库的泛化能力受限；SFT易导致过度拒绝；评测过程仍需人工审核；缺少更丰富的角色层级与实例级权限场景。

---

## 201. IDSTune: A Multi-Agent Collaborative Framework for Integrated Database System Tuning

**arXiv ID:** 2607.22031 | [PDF](https://arxiv.org/pdf/2607.22031v1)

**作者:** Yiyan Li `[一作]` (Renmin University of China), Hong Chen `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了IDSTune，一个基于大型语言模型的多智能体协作框架，用以联合优化数据库的调优参数、索引和物化视图。

**💡 创新点**

创新点包括：①将多种调优目标统一到多智能体协作架构；②通过工作负载压缩与特征选择实现任务感知的上下文；③利用LLM搜索和外部知识提升适应性；④引入混合安全防护网防止不合法配置。

**🔧 技术方法**

使用了 GPT‑4.1 等大型语言模型、基于工作负载压缩的特征提取与选择、Web 搜索检索、以及多智能体协作与安全校验机制。

**📊 数据集**

实验使用了七个公开/真实工作负载：TPC‑H、JOB、TPC‑C、SYSBENCH、SDSS、Birds、Redbench。

**📈 对比分析**

与单体/双体/三体基线（如 AgentTune、Dexter、UniView、λ‑Tune、Proto‑X、HMAB 等）进行对比，IDSTune 在所有基准上均实现了 18%–38% 的性能提升，并将调优时间缩短 50%–60%，稳定性最优。

**⚠️ 局限性**

局限性在于：受 LLM 推理成本和 token 消耗限制；对极大规模数据集和高并发写入场景的持续性能验证不足；缺乏针对极端动态工作负载的自适应重启机制。

---

## 202. Impedance Control of Ship-Borne Manipulators via Optimization-based Task-Space Inverse Dynamics

**arXiv ID:** 2607.22030 | [PDF](https://arxiv.org/pdf/2607.22030v1)

**作者:** Lingxiao Meng `[一作]`, Max Q. -H. Meng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对海上浮动平台上的机械臂，在波浪造成的基座运动下实现精准轨迹跟踪和柔性交互的控制方法。

**💡 创新点**

提出基于任务空间逆动力学的优化控制框架，并通过误差状态卡尔曼滤波实现基座状态的实时融合估计，实现在浮动基座下的直接力矩级惯性耦合补偿。

**🔧 技术方法**

利用任务空间逆动力学(QP求解)、误差状态卡尔曼滤波(ESKF)和惯性测量单元(IMU)与末端执行器位姿融合。

**📊 数据集**

仿真使用Gazebo+ROS，基于录制的AUV波浪轨迹；实验证明使用Panda机械臂+Stewart平台与运动捕捉系统。

**📈 对比分析**

与四种基准(kp, MPC, 估计扰动等)对比，平均位置误差降低超过80%，动态插接成功率提升至100%，接触力平均降低约45%。

**⚠️ 局限性**

对模型不确定性敏感，受限于关节/力矩极限导致约束激活；实验未考虑机械臂对船舶的反作用；需要外部姿态测量或更完善的视觉融合。

---

## 203. DWT-Fusion: A Signal-Based Framework for Training-Free LLM-Generated Text Detection

**arXiv ID:** 2607.22026 | [PDF](https://arxiv.org/pdf/2607.22026v1)

**作者:** Mehmet Batuhan Özdaş `[一作]` (Ankara University), Murat Osmanoğlu `[通讯]` (Ankara University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出训练无监督的基于离散小波变换的检测框架 DWT‑Fusion，利用 Token 级别 log‑probability 序列的多尺度小波分解得到局部能量信号，用来检测 LLM 生成文本。

**💡 创新点**

创新点在于：①把 token‑log‑probability 视作有序信号，采用离散小波分解捕获局部与多尺度概率波动；②在不训练任何分类器的前提下，通过校准加权硬/软投票融合多配置，显著提升检测性能。

**🔧 技术方法**

使用的技术包括：预训练因果语言模型提取 token‑log‑probability、离散小波变换 (DWT) 及三种小波域分数（首层能量、多层能量、窗口能量）、无监督加权投票（硬/软）以及对比基准统计与 DFT 能量方法。

**📊 数据集**

实验数据集包括 HC3、M4、MAGE 三个 LLM 文本检测基准，代理模型为 GPT‑Neo‑2.7B、GPT‑J‑6B、Falcon‑7B、LLaMA‑3‑8B。

**📈 对比分析**

与传统零样本统计（log‑likelihood、rank、entropy、LRR）和 DFT 能量基准对比；单一配置在 HC3、M4、MAGE 上分别获得 AUROC 0.9872、0.8185、0.7138；投票加权后 AUROC 分别提升至 0.9919、0.8477、0.7471；在低 FPR（≤1%）下仍存在局限，MAGE 表现最弱。

**⚠️ 局限性**

限制：①对代理模型的依赖，代理模型的概率分布差异会影响性能；②需要校准集，方法不完全无监督；③截断为 512 token 且 DWT 只至 3 层，可能丢失长文本信息；④在极低 FPR 或高度异质数据集（如 MAGE）下仍难以满足严格误报阈值。

---

## 204. Pretraining EHR Foundation Models with Patient-Aware Sampling

**arXiv ID:** 2607.22114 | [PDF](https://arxiv.org/pdf/2607.22114v1)

**作者:** Joshua Placidi `[一作]` (Imperial College London), A. Aldo Faisal `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了电子健康记录（EHR）预训练时序列构造方法，并提出了一种基于患者采样的Patient Sampling策略；

**💡 创新点**

创新点在于引入可调节的患者采样权重α，使训练信号在患者间分布可控，且仅保留患者边界，从而缓解长轨迹偏倚；

**🔧 技术方法**

技术包括：基于GPT‑2的解码器型自回归模型、窗口化训练、患者采样概率公式p_α(i)和对窗口起始位置的随机化；

**📊 数据集**

使用MIMIC‑IV v2.2与v3.1两个版本的标记化EHR数据集，并加入MIMIC‑IV‑ED 2.2扩展；

**📈 对比分析**

与传统的Global Stream和Patient Chunks方法比较，Patient Sampling在ICU死亡、ICU再入院、ICU入院、住院死亡等四项临床任务的宏观AUROC和AUPRC均优于Baseline；最佳α为0.6；

**⚠️ 局限性**

局限性包括：采样策略对不同任务的α敏感性未完全解释；仅在单一GPU上训练，未验证更大模型规模的可扩展性；并且只关注单一自回归框架，未与其他生成式预训练方法直接对比。

---

## 205. Transforming Keystroke Noise to Text: Self-Supervised Acoustic Eavesdropping Attacks on Keyboards

**arXiv ID:** 2607.22094 | [PDF](https://arxiv.org/pdf/2607.22094v1)

**作者:** Atsunori Okada `[一作]` (Tohoku University), Naofumi Homma `[通讯]` (Tohoku University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自监督的声学窃听攻击，利用键盘敲击声恢复文本，满足极低部署、无目标设备标注数据、少量敲击即可高精度复原的实际需求。

**💡 创新点**

核心创新在于将Transformer（字符级BERT）与声学聚类、模糊嵌入、LLM后处理及迭代自训练相结合，实现从无标签数据到可解码字符的闭环推理，显著降低对先验标注的依赖并提升低数据场景的精度。

**🔧 技术方法**

技术要点包括：MFCC+谱特征提取→UMAP降维→层次聚类→空间键识别→不确定性嵌入→字符级BERT掩码预测→LLM（Gemini Flash）纠错→伪标签扩散(label spreading)→迭代反馈更新映射矩阵。

**📊 数据集**

训练数据：使用OpenWebText（仅保留小写字母、空格、逗号、句点）构建字符级BERT；评估数据为多台笔记本（Dell、MacBook、HP、Lenovo）在三种真实场景（桌面远程、墙壁传输、在线会议）下的录音。

**📈 对比分析**

对比HMM基线、字典约束方法以及传统监督式声学模型，实验表明在仅100-150次敲击时即可达99%+的Levenshtein精度；而基线需要>200-1200次敲击才能逼近相同水平，显示显著性能提升。

**⚠️ 局限性**

局限性包括：需准确的敲击分割与足够的信噪比；对高速打字（键盘声重叠）效果未知；仅支持英文字母+标点；对非自然语言输入（如随机密码）恢复仍有限；需要多轮迭代，计算成本较高。

---

## 206. FAIR: Feature-Augmented Implicit Regularization for AI-generated Fake Image Detection

**arXiv ID:** 2607.22087 | [PDF](https://arxiv.org/pdf/2607.22087v1)

**作者:** Md Redwanul Haque `[一作]` (Deakin University), Tsz-Kwan Lee `[通讯]` (Deakin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在训练阶段引入宏观结构先验（Scene Composition Structure）并在推理时移除，从而实现无额外推理成本的AI生成图像检测器自适应性提升。

**💡 创新点**

创新点在于将“学习使用特权信息”(LUPI)与隐式正则化结合，利用域不变的宏观结构特征在训练中对权重空间进行几何约束，形成更平滑、更泛化的决策边界。

**🔧 技术方法**

采用SCS特征提取、全连接变换、基于LUPI的特征拼接、以及传统检测网络（PatchCraft、AIDE）的分类头扩展，并在训练后只保留对原始特征的权重子矩阵。

**📊 数据集**

在GenImage、AIGCDetect、UnivFD、Fake2M、DRCT-2M等五大公共基准上进行评估，尤其针对大规模的Diffusion生成模型进行零样本迁移测试。

**📈 对比分析**

与基线模型及传统正则化（Dropout、L1/L2）对比，FAIR在多数域上平均提升8%以上准确率，尤其在Diffusion域上提升至新SOTA；同时对JPEG压缩等后处理的鲁棒性也明显提高。

**⚠️ 局限性**

局限性包括：对极端后处理（如低质量JPEG）仍可能出现性能下降；SCS先验的计算成本在训练阶段不低，且其效果依赖于先验的质量；目前仅验证于图像级检测，未扩展至视频或其他模态。

---

## 207. Alleviating Regional Shortcuts for Few-Shot Class-Incremental Learning

**arXiv ID:** 2607.22072 | [PDF](https://arxiv.org/pdf/2607.22072v1)

**作者:** Haichen Zhou `[一作]` (Huazhong University of Science and Technology), Yuhua Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出一种基于共通与辨别视觉原语的FSCIL方法，旨在缓解基类训练中出现的“区域捷径”导致的novel类误分类问题。

**💡 创新点**

通过构造共通原语集合与辨别原语集合，并在训练中分别约束其语义一致性与辨别性，显式打破区域捷径，实现更好泛化与可解释性。

**🔧 技术方法**

采用分解与重组的组合学习框架，利用注意力、相似度引导损失、原语重构损失、混合样本变换以及多层感知机掩码等技术。

**📊 数据集**

在CIFAR‑100、miniImageNet 和 CUB‑200‑2011 三个标准FSCIL基准上进行实验。

**📈 对比分析**

与多种SOTA方法对比，使用 ResNet‑12 和 ViT‑B/16 在各数据集上均实现了领先或相近的准确率，尤其在 CIFAR‑100 上 ResNet‑12 达到 68.13%、ViT‑B/16 达到 88.69%。

**⚠️ 局限性**

主要针对图像基准，难以直接推广至多模态或动态视频场景，且在极少样本或高噪声下仍需进一步验证。

---

## 208. Constraint-Driven Synthesis of Hyper Petri Nets

**arXiv ID:** 2607.22062 | [PDF](https://arxiv.org/pdf/2607.22062v1)

**作者:** Maksym Figat `[一作]` (Warsaw University Of Technology), Alessandro Pinto `[通讯]` (NASA Jet Propulsion Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Hyper Petri Net（HyPN）模型，并给出从布尔约束（CNF形式）自动合成HyPN的完整流程，保证所有可观测标记满足约束且具备明确的执行语义。

**💡 创新点**

1）引入可观测与内部状态分离的HyPN；2）定义基于可观测状态的原子执行序列；3）开发基于子句的可组合子网和冲突解决的合成算子，使结构规模线性增长。

**🔧 技术方法**

Petri网理论、布尔逻辑与CNF、组合式合成算法、冲突交换（swap）机制、可观测状态集的构造。

**📊 数据集**

论文中未使用外部真实数据集，主要以月球车启发式案例（约束集合）和示例子网演示合成与行为分析。

**📈 对比分析**

通过对两种示例（互斥约束与资源约束）展示可观测状态数与结构规模的线性增长，并与全连接可达图（指数级）对比；实验表明合成后Petri网的节点/边数仅随变量和子句数线性增长，行为可通过可观测转移表验证。

**⚠️ 局限性**

仅支持布尔不等价约束；不提供时序或概率约束；缺乏最优执行策略选择；未覆盖初始化、故障恢复或动态约束更新；对大规模布尔公式的可观测状态空间仍可能指数级。

---

## 209. An AI-Driven Virtual Patient for Breaking Bad News: An Expert Formative Study on Facial Expression Intensity

**arXiv ID:** 2607.22118 | [PDF](https://arxiv.org/pdf/2607.22118v1)

**作者:** Steffen Hauck `[一作]` (Coburg University of Applied Sciences), Jens Grubert `[通讯]` (Coburg University of Applied Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并评估了一个可在VR中进行的生成式AI虚拟病人原型，实验通过调节面部表情强度来研究其对沟通真实感和情感识别的影响。

**💡 创新点**

创新点在于首次将实时面部表情强度调节与LLM驱动的对话结合，并通过专家评估系统化提炼下一代情感化虚拟病人的设计需求。

**🔧 技术方法**

采用LLM（gemma3:27b-cloud）生成对话，NVIDIA Audio2Face实现面部动画，Unreal Engine 5.5.4渲染，Meta Quest 3头显，ElevenLabs TTS和FastAPI后端进行语音处理。

**📊 数据集**

使用自制的肺癌BBS情景对话脚本（含SPIKES协议），未使用公开语料库或大规模数据集。

**📈 对比分析**

采用within-subjects对比低强度（0.6）与高强度（1.0）表情，使用VHPQ、VABS、SUS量表，结果显示真实感无显著差异，情感可信度略有提升（M=4.82→5.11），SUS平均得分76.07；定性访谈指出面部强度影响有限。

**⚠️ 局限性**

局限性包括样本量小（N=7）且受限于专家群体、对话脚本中文字/语调差异混杂了视觉操纵、仅关注面部表情未加入身体动作、未实现完全自主实时对话。

---

## 210. Predictive Lightweight MARL for Resilient Coverage in Sparse-Signaling Aerial Networks

**arXiv ID:** 2607.22109 | [PDF](https://arxiv.org/pdf/2607.22109v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Ang-Hsun Tsai `[通讯]` (Feng Chia University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 PL-MARL 框架，利用运动学感知推理引擎和拓扑感知图注意机制，在稀疏通信条件下实现无人机编队覆盖任务的自适应协调；

**💡 创新点**

创新点在于将物理运动先验嵌入推理模块，实现对邻居状态的主动重构，解耦通信频率与决策执行，并通过单头注意机制和并行前馈网络实现轻量级、可扩展的分布式学习；

**🔧 技术方法**

采用数字孪生辅助的集中训练、分布式执行（CTDE）框架，结合 MLP 推理器、位置编码、图注意网络、MAPPO 强化学习以及多速率调度策略；

**📊 数据集**

使用仿真生成的 1 km² 3D 覆盖场景数据，包含 4 只 UAV 与 240 个高斯-马尔可夫地面用户，混合移动性训练集；

**📈 对比分析**

通过与 TAG‑MAPPO、GRU‑based MARL 以及理论 Oracle 边界的对比，评估覆盖率、推理延迟和鲁棒性；实验表明 PL‑MARL 在信号稀疏下接近 Oracle 覆盖率，延迟低于 1 ms，且在节点失效时表现出平滑的性能衰减；

**⚠️ 局限性**

局限性包括仅在仿真环境验证，未对真实无人机平台进行测试；对物理模型的依赖可能限制在极端动态变化场景中的适应性；以及在更大规模编队时仍需进一步评估能耗与计算资源占用。

---

## 211. MEUSLI: a Multilingual Projector for LLM-based ASR and Beyond

**arXiv ID:** 2607.22100 | [PDF](https://arxiv.org/pdf/2607.22100v1)

**作者:** Lorenzo Concina `[一作]` (Fondazione Bruno Kessler), Alessio Brutti `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 28 种语言上评估并对比 Whisper、Apertus-8B、EuroLLM-1.7B 与 EuroLLM-9B 四种多语言语音识别模型的词错误率（WER）。

**💡 创新点**

引入 EuroLLM 1.7B/9B 两种规模不同的多语言 LLM‑ASR 模型，并将其与现有 Whisper 与 Apertus‑8B 进行横向对比，验证不同规模模型在多语言环境下的性能差异。

**🔧 技术方法**

使用 Whisper 预训练语音识别模型、SLAM‑ASR 系统（Apertus‑8B、EuroLLM-1.7B/9B）以及绘图工具绘制 WER 曲线进行可视化比较。

**📊 数据集**

采用包含 28 种语言（es、de、nl、pt、gl、en、pl、cs、fr、hu、it、sv、ro、da、eu、bg、fi、lv、lt、el、sk、sl、et、cy、sr、mt、br、ga）的多语言语音数据集（对应 CV 评估集）。

**📈 对比分析**

通过在每种语言上记录 WER 并绘制折线图进行直接比较。结果显示 EuroLLM‑9B 在大多数语言上的 WER 低于 Whisper、EuroLLM‑1.7B 和 Apertus‑8B；EuroLLM‑1.7B 介于 Whisper 与 Apertus‑8B 之间；Apertus‑8B 在某些语言（如中文、日语等）表现出色，但整体仍落后于 EuroLLM‑9B。

**⚠️ 局限性**

局限性包括：对极低资源语言（如 br、ga 等）性能仍不佳；在某些语言点出现 WER 为 0 说明缺失或不完整数据；模型规模与训练成本的权衡尚未充分探讨。

---

## 212. Reasoning Denoiser: Denoising Reasoning Traces for Hallucination Detection in Large Reasoning Models

**arXiv ID:** 2607.22098 | [PDF](https://arxiv.org/pdf/2607.22098v1)

**作者:** Junlin Fang `[一作]` (Nanyang Technological University), Sean Du `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于最终答案注意力的无监督思考步骤去噪框架（Reasoning Denoiser），用于在大型推理模型生成的长推理轨迹中去除无关和重复步骤，从而提升幻觉检测的准确率。

**💡 创新点**

创新点在于：1）利用最终答案对每一步的注意力作为无标注的筛选信号；2）通过轻量级投影网络重塑步骤嵌入空间，使信息步骤聚集、噪声步骤分散，便于后续基于距离的过滤；3）可无监督地训练，并可与任意幻觉检测器无缝结合。

**🔧 技术方法**

核心技术包括：最终答案注意力计算、基于困惑度加权的步骤嵌入提取、投影网络训练（带compact、disperse、separate 三项损失）、kNN距离过滤、下游幻觉检测（如CCS、监督探测、Perplexity、Verbalized Certainty等）。

**📊 数据集**

使用四大推理基准数据集：TruthfulQA、MATH（含MATH500）、CodeElo 与 MultiHopQA（由HotpotQA、2WikiMultihopQA等构成），以及对比实验的更大模型 Qwen3-32B、DeepSeek-R1-Distill-Qwen-32B。

**📈 对比分析**

与原始长轨迹、直接注意力过滤以及多种基线（Embedding、Consistency、Verbalized、LRM专用）比较，实验显示在所有数据集与模型上均实现显著提升；例如在 Qwen3-8B+TruthfulQA 上 AUROC 从 68.63% 提升至 87.32%（CCS）或从 80.42% 提升至 86.44%（监督探测）。在更大模型上亦获得 90.87% 与 78.95% 的最佳成绩。

**⚠️ 局限性**

局限性包括：1）仍依赖 LRM 的内部注意力分布，若注意力不稳定可能影响过滤质量；2）过滤比例需手动设定，过度或不足的去噪可能移除关键信息；3）实验主要集中在文本推理任务，对代码、图像等多模态推理的通用性待验证。

---

## 213. MemNMF: Memory-Augmented NMF on LPC Spectra for Anomalous Sound Detection

**arXiv ID:** 2607.22086 | [PDF](https://arxiv.org/pdf/2607.22086v1)

**作者:** Phurich Saengthong `[一作]` (Institute of Science Tokyo), Takahiro Shinozaki `[通讯]` (Institute of Science Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于 LPC 谱和 NMF 初始化记忆网络的异常声音检测方法 MemNMF，能够在仅使用正常声音训练的情况下对异常进行检测。

**💡 创新点**

创新点在于：①将 LPC 谱作为输入，聚焦频谱包络以降低对时频细节的依赖；②使用 NMF 字典初始化记忆网络，通过注意力权重组合正常谱基，强制重建仅来自正常模式，从而提高在噪声和非平稳条件下的鲁棒性。

**🔧 技术方法**

使用的技术包括线性预测编码（LPC）谱提取、非负矩阵分解（NMF）、记忆网络（Memory Network）与注意力机制、批归一化、均方重建损失以及基于 Autoencoder 的基线模型。

**📊 数据集**

实验数据集为 DCASE2020 Task 2（包含 MIMII 子集和 ToyADMOS 子集）以及完整的 MIMII 数据集，涵盖六类机器及多种 SNR 条件。

**📈 对比分析**

与传统 Log‑Mel spectrogram AE、PAA、ANP‑IDNN 等方法对比，MemNMF 在 DCASE2020 上平均 AUC 达 81.7%，在 MIMII 上平均 AUC 达 90.6%，明显优于 GRLNet（77.2%）并在 -6 dB 噪声下仍保持 90% 以上的性能。

**⚠️ 局限性**

局限性包括对局部异常（如 Toy‑conveyor）检测效果不佳；LPC 谱在某些机器上无法捕捉足够的细节；需要针对不同机器类型进行超参数调优；跨域迁移与分布漂移的鲁棒性仍待进一步提升。

---

## 214. CommandLM: Data driven behavior level descriptor for ego vehicles

**arXiv ID:** 2607.22078 | [PDF](https://arxiv.org/pdf/2607.22078v1)

**作者:** Boris Tokic `[一作]` (Munich University of Applied Sciences), Fabian B. Flohr `[通讯]` (Munich University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种多模态大语言模型CommandLM，可将融合的LiDAR与多摄像头的鸟瞰视图压缩为简洁可读的行为描述。

**💡 创新点**

创新点在于将时序BEV融合与Q-Former压缩结合，支持从多帧动态感知生成意图驱动的行为级文字解释。

**🔧 技术方法**

技术上使用BEVFusion、Q-Former、量化LoRA微调的Qwen-2.5 LLM，配合时间滚动缓冲实现视觉-语言接口。

**📊 数据集**

数据集为CommandLM-nuScenes（DriveLM‑nuScenes的扩增版），包含约40,000条行为级问答对。

**📈 对比分析**

与BLIP‑2基准对比，CommandLM在CIDEr、BERT‑F1、METEOR、SPICE等指标均显著提升，行为探索率和真实匹配率也更高。

**⚠️ 局限性**

局限性包括对稀有场景的覆盖不足、Q‑Former压缩可能导致信息丢失，以及仍有部分行为描述与实际轨迹不完全一致。

---

## 215. Developing and Validating the Spanish Version of the Large Language Models Dependency Scale (LLM-D12-SP)

**arXiv ID:** 2607.22041 | [PDF](https://arxiv.org/pdf/2607.22041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 216. FSE: Continual Learning for Named Entity Recognition by Fast-Slow Experts

**arXiv ID:** 2607.22075 | [PDF](https://arxiv.org/pdf/2607.22075v1)

**作者:** Yunan Zhang `[一作]` (Harbin Institute of Technology), Qingcai Chen `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种Fast‑Slow Experts (FSE) 模型，用于解决命名实体识别的持续学习问题，通过快速专家过滤无关跨度并让慢速专家聚焦剩余跨度，从而提升稳定性和可塑性。

**💡 创新点**

通过引入共享的快速专家学习相邻词的链接来过滤跨度、使用双专家融合、长度衰减负采样策略，以及在持续学习中利用知识蒸馏，解决了传统跨度方法的灾难性遗忘与跨度不平衡问题。

**🔧 技术方法**

基于BERT‑Base+BiLSTM的上下文编码；快速专家使用softmin池化将token级链接转化为跨度级fast score；慢速专家使用scaled dot‑product交互的跨度表示；双专家融合采用调和平均；负采样采用长度衰减公式；知识蒸馏采用Bernoulli KL。

**📊 数据集**

在OntoNotes 5.0‑EN（18类）和FewNERD（66类）上构造的合成持续学习任务数据集，分别将任务拆分为多步学习。

**📈 对比分析**

与SeqFT、AddNER、ExtendNER、L&R、ExtendNER+DLD、SpanKL、SKD‑NER以及GPT‑5与Llama3.1等基线相比，FSE在两大数据集上取得了State‑of‑the‑Art的Macro‑F1，尤其在FewNERD的多类型任务中提升显著，并且训练收敛更快。

**⚠️ 局限性**

依赖于BERT‑Base的预训练编码器，对长句子仍有计算瓶颈；长度衰减负采样参数需经验调节；在极高实体类别数下仍受prompt与模型容量限制，LLM在此任务仍表现不佳。

---

## 217. Connected (Dense) Partition for Tree-Like Graphs

**arXiv ID:** 2607.22070 | [PDF](https://arxiv.org/pdf/2607.22070v1)

**作者:** Katrin Casel `[一作]` (Humboldt-Universität Berlin), Aikaterini Niklanovits `[通讯]` (Hasso Plattner Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并求解最大稠密图划分（MDGP）问题，提出well-structured图的概念并分析其中的最优划分结构

**💡 创新点**

创新点在于：①将最优划分归约为仅包含基于cut节点的generalized star和单块子图；②在thick tree（厚树）上给出多项式时间算法；③证明MDGP_k在split图上是NP-hard

**🔧 技术方法**

使用的技术包括：图分解（block-cut树）、动态规划递推状态、结构化图分析（well-structured、thick tree、split graph）以及数值与逻辑证明

**📊 数据集**

所用数据集为人工构造的示例图（如完整split图、dominating set实例等），未使用公开真实数据集

**📈 对比分析**

方法通过理论复杂度分析与已知算法对比，thick tree上的算法实现多项式时间；在split图上的NP-hard性证明表明在该类图上无多项式解法

**⚠️ 局限性**

限制：只在well-structured图或thick tree上能得到多项式解，无法推广到一般图；split图上的NP-hard性证明依赖特殊构造，实际应用受限

---

## 218. LAMAR: An Open Language-Aware Multilingual Alignment Reranker

**arXiv ID:** 2607.22042 | [PDF](https://arxiv.org/pdf/2607.22042v1)

**作者:** Seongtae Hong `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一种语言感知的多语言重排序模型 LAMAR，用于检索增强生成任务中的候选文档排序。

**💡 创新点**

创新点在于通过英文锚定的相关性蒸馏实现跨语言语义相关性统一，并在第二阶段引入语言一致性对齐损失，使模型在语义等价文档中优先选择与查询同语言的文档。

**🔧 技术方法**

采用跨编码器（基于 bge-m3-retromae）进行两阶段训练：①教师-学生蒸馏阶段；②组排序损失（ADR-MSE）+语言一致性损失（softplus）组合，使用列表式训练、软最大化等技术。

**📊 数据集**

训练使用 MMARCO、MIRACL 和 RLHN 三大多语言/单语言数据集；评测覆盖 XQuAD、BELEBELE（语言一致性评估）、MIRACL、XGLUE、HUME、MLDR、Wikipedia 等多语言重排序基准。

**📈 对比分析**

与13+公开多语言重排序器对比，使用 nDCG@1/10、MRR@10 等指标。LAMAR 在语言一致性评估中取得最高 nDCG@1（XQuAD 0.9689，BELEBELE 0.9466），在常规多语言重排序基准中排名第二（平均 86.84），并在检索候选集上始终保持最佳性能。

**⚠️ 局限性**

局限性包括：①尚未评估对生成质量的直接影响；②在真实检索环境下多样性和非语义等价文档的处理仍有限；③对非英语主流语言的泛化能力尚未充分验证；④模型虽规模小，但在更大参数量的多语言模型面前仍有提升空间。

---

## 219. Accountable Transaction Inclusion Lists: Enhancing Ethereum's Censorship Resistance

**arXiv ID:** 2607.22040 | [PDF](https://arxiv.org/pdf/2607.22040v1)

**作者:** Patrick Spiesberger `[一作]` (Karlsruhe Institute of Technology), Hannes Hartenstein `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FairFIL（Fair Forward Inclusion Lists）机制，用于在以太坊中实现可验证的排除公开披露，从而增强对交易审查的抵抗力。

**💡 创新点**

创新点在于将可审计的排除列表与区块奖励绑定，构建一个可验证且可追溯的责任机制，使得持续审查需要失去完整区块奖励，从而显著提高审查成本。

**🔧 技术方法**

采用以太坊现有的提议–构建者分离（PBS）框架，结合公平排序（基于每Gas小费）和验证委员会的多轮检查，使用智能合约进行奖励与惩罚。

**📊 数据集**

使用了2025年9月至12月期间的以太坊主网区块（214,600个块）、Flashbots Mempool Dumpster的交易记录以及Yale大学分布式系统实验室的多节点内存池观测数据。

**📈 对比分析**

通过与现有机制（FIL、FOCIL、MCP、AUCIL）在同一数据集上对比，结果显示FairFIL在两槽审查成本从≈3.08提升至≈28.30，平均每块被审查交易数为8.3，构造时间低于4ms，且对构建者的MEV提取影响最小。

**⚠️ 局限性**

局限性包括对内存池一致性的高度依赖、假设参与者完全理性、对XOF阈值设定的未知性以及私有交易可能导致的可见性不足，且对极端高预算审查者的防御尚未充分验证。

---

## 220. Sparse by Command: Task-Conditional Compute Skipping for Multi-Task Inference Accelerators

**arXiv ID:** 2607.22038 | [PDF](https://arxiv.org/pdf/2607.22038v1)

**作者:** Afzal Ahmad `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于任务命令的多任务推理加速器，能够在执行时根据当前任务动态生成并跳过不必要的计算块，从而显著降低 FLOPs、延迟和能耗。

**💡 创新点**

创新点在于将任务指令作为“无成本”条件，结合轻量级门控网络产生与硬件 tile 细粒度对齐的二值掩码，并在 ISA 级别嵌入掩码字段，使得硬件可以在不增加数据通路的前提下零成本跳过计算。

**🔧 技术方法**

使用了轻量级 MLP 门控网络、三阶段稀疏训练策略（soft → hard mask）、INT8 量化、指令集（NISA）支持 tile mask、双缓冲 DMA、基于 Xilinx Alveo U50 的 2,048 MAC/周期的 tiled 计算引擎。

**📊 数据集**

在 CARLA 仿真环境中收集 302K 帧（六个命令：直行、左转、右转、左变道、右变道、刹车），并通过 DAgger 方式增强数据，训练多任务视觉控制器。

**📈 对比分析**

与全密集模型以及 GPU 纯稀疏实现相比，任务条件稀疏在 FPGA 上实现 FLOPs 减少 66–76%，延迟下降 51–59%（从 9.12 ms 降到 3.74–4.44 ms），能耗下降 51–59%（从 263 mJ 降到 108–128 mJ），实现 2.1–2.4 倍加速；GPU 上稀疏模型反而慢 22%。

**⚠️ 局限性**

局限性包括：仅适用于已知的离散低维任务指令；需要 ISA 级别的掩码字段和对应硬件支持；对模型结构（共享 backbone）和 tile 大小有硬件约束；对输入自适应稀疏的能力有限，且在更大模型或多任务复杂度更高时的扩展性仍需验证。

---

## 221. The machine can say it but cannot hear it. Designed affective patterns and the expressive-sensing asymmetry in human-machine communication

**arXiv ID:** 2607.22104 | [PDF](https://arxiv.org/pdf/2607.22104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 222. On the Runtime Analysis of Reinforcement Learning Hyper-Heuristics

**arXiv ID:** 2607.22036 | [PDF](https://arxiv.org/pdf/2607.22036v1)

**作者:** Pietro S. Oliveto `[一作]` (Southern University of Science and Technology), Mengqing Xu `[通讯]` (Southern University of Science and Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文对 Reinforcement Learning Hyper‑Heuristic (RLHH) 在 LeadingOnes 基准函数上的运行时进行了严格分析，证明其在合适参数下可实现最优期望运行时间，并通过实验验证其优于 GRG HH。

**💡 创新点**

首次对含强化学习机制的 Hyper‑heuristic 给出正向理论结果，展示 RLHH 在二阶算子集合 {1‑bit, 2‑bit flip} 上能自适应学习最优算子序列，并与最优离线设计序列匹配。

**🔧 技术方法**

采用随机过程、马尔可夫链、漂移定理、马尔科夫不等式、自由能不等式、Doob 最大不等式等概率工具跟踪权重动态。

**📊 数据集**

以 LeadingOnes 函数为测试基准，实验覆盖 n=10³ 至 9×10⁹，评估平均运行时间。

**📈 对比分析**

与 GRG HH、单一 1‑bit RLS、最优 2‑算子序列等进行对比，结果显示在 n>2000 时 RLHH 的平均运行时间接近理论最优 1+ln2/4≈0.423n²，优于 GRG 并略低于单一 1‑bit。

**⚠️ 局限性**

仅考虑了两算子集合且参数需预设，未对更大算子集或其他更新/选择规则进行分析；理论证明仅适用于 γ<1，但实验表明 γ=1 亦可行。

---

## 223. DCS: A Unified Conditional Sensitivity Framework for Cross-Modal Copyright Infringement Detection

**arXiv ID:** 2607.22035 | [PDF](https://arxiv.org/pdf/2607.22035v1)

**作者:** Xiafeng Man `[一作]` `[通讯]` (Fudan University), Xiafeng Man (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种统一的后置版权侵权检测框架——Dual‑Branch Conditional Sensitivity (DCS)，通过对模型参数进行学习与反向学习的双分支微调，间接估计目标样本对条件输出分布的影响；

**💡 创新点**

创新点在于将版权侵权视为“条件分布偏移”的反事实问题，并将其与条件差分隐私理论相结合，设计了可计算的对数密度漂移上界作为隐私预算代理；

**🔧 技术方法**

核心技术包括影响函数近似、局部双分支梯度更新、条件敏感度（Conditional Sensitivity）计算、以及与正交背景条件的校准（Calibrated Detection Statistic）来消除全局微调漂移；

**📊 数据集**

在实验中使用了多模态数据集：线性回归实例采用 Diabetes 数据集，扩散模型使用 Stable Diffusion 公开模型与 CLIP 嵌入，语言模型采用 Qwen2.5‑7B，跨模态实验结合文本‑图像对齐数据；

**📈 对比分析**

通过将 DCS 与隐私预算、模型输出差异（图像嵌入、token 词分布、注意力张量等）进行对比，证明了其在不同模态下与理论上限的强相关性；实验结果显示，校准后的检测统计量显著高于未校准基线，且能够在多模态和生成任务中有效区分侵权与非侵权样本；

**⚠️ 局限性**

局限性包括：无法单独决定法律侵权；依赖目标样本的高辨识度、可观测的条件输出以及可访问的对齐/正交条件；在模型已强制使用对齐/隐私保护的情况下，检测灵敏度可能下降；

---

## 224. HEMERA: A Heterogeneous Memory-Centric Accelerator with Recursive Dataflow for Edge-Constrained State-Space-Duality Models Inference

**arXiv ID:** 2607.22022 | [PDF](https://arxiv.org/pdf/2607.22022v1)

**作者:** Hao Ding `[一作]` (Peking University), Yimao Cai `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对 Mamba‑2 结构中的 Structured State Space Duality（SSD）执行，设计了一款名为 HEMERA 的异构内存中心加速器，利用矩阵无关的递归流式数据流实现 SSD 的高效推理，并通过多层 NAND–eDRAM–SRAM 记忆层次与数字 CIM 核心协同工作，显著降低了中间状态的二次存储与数据搬运。

**💡 创新点**

① 将 SSD 的矩阵形式重构为注入–状态–衰减三段递归流式执行，完全消除 O(cs²) 的中间张量；② 在硬件层面实现专用的 Streaming Recursive Engine（SRE）和数字 CIM（DCIM）混合计算单元，既满足密集投影运算，又能高效执行递归状态更新；③ 采用分层内存架构，使参数在 NAND 层流式传输，动态状态在 eDRAM/ SRAM 层内驻留，极大减少了跨层数据传输。

**🔧 技术方法**

算法与硬件协同设计、矩阵无关递归数据流、分层 NAND–eDRAM–SRAM 存储、数字 CIM（BF16）投影核心、Streaming Recursive Engine（SRE）、指数 LUT+前缀和单元、Systolic 单元、数字 MAC 归约。

**📊 数据集**

使用公开预训练的 Mamba‑2 模型（130M、370M、790M、1.4B、2.8B 参数规模）进行长序列推理测试，序列长度覆盖 128–16384，未指定特定自然语言或图像数据集，主要关注推理效率与能耗。

**📈 对比分析**

在 22 nm CMOS 0.9 V 下，HEMERA 与 Intel Xeon CPU 与 NVIDIA A100 GPU 进行对比。对 A100 官方融合核基准，HEMERA 在 2.8B 模型、4096 序列长度下平均速度提升 1.4–3.6×，能耗提升 12–27×；在整体端到端推理中，SSD 所占时间比例由 A100 的 46–71% 降至 14% 左右。相较于 CPU，HEMERA 亦显著提高了计算利用率和吞吐量。

**⚠️ 局限性**

主要局限：目前仅针对基于 SSD 的 Mamba‑2 结构，无法直接推广到其他结构的 SSM 或传统注意力模型；对高精度（FP32/INT8）或低功耗（低位宽）实现的支持仍有限；设计中对内存层次的宽带需求高，可能在极端边缘设备上实现难度较大。

---

## 225. Curly Hair Simulation using Curly Finite Elements

**arXiv ID:** 2607.22103 | [PDF](https://arxiv.org/pdf/2607.22103v1)

**作者:** Xinming Pei `[一作]` (Zhejiang University), Huamin Wang `[通讯]` (Style3D Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一种新的卷曲发丝模拟框架，将每根发丝分解为低频基线与高频波纹场，实现高效、细节保留的动态仿真。

**💡 创新点**

创新点包括：①基于高频波纹的解析表示（平面波/螺旋），②曲率能量分裂与角度近似；③混合碰撞处理；④在卷曲有限元上实现稀疏引导丝插值。

**🔧 技术方法**

采用离散弹性杆理论、曲率能量拆分、角度-能量近似、混合碰撞检测、基于波纹的连续表示以及基于引导丝的插值。

**📊 数据集**

主要使用合成的卷曲发型数据（波浪、螺旋、Afro 等），未给出公开数据集。

**📈 对比分析**

与传统的离散弹性杆（DER）在不同分辨率下对比；在保持视觉质量的同时，波纹模型每帧耗时约为 DER 的1/3–1/5，DoF 更少、速度更快且稳定性更好。

**⚠️ 局限性**

局限：假设波纹数/螺旋转数固定，难以处理极端拉伸/压缩；忽略剪切和其它力学效应；在高密度或复杂交互时可能出现碰撞不稳定和细节失真。

---

## 226. Code Review is a Conversation: Toward Conversational AI Review Assistants

**arXiv ID:** 2607.22095 | [PDF](https://arxiv.org/pdf/2607.22095v1)

**作者:** Rosalia Tufano `[一作]` `[通讯]` (Software Institute), Rosalia Tufano (Software Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出将 AI 代码评审从单一“一次性评论”转向“对话式 AI 审查助手”，通过定义对话动作（如提问、请求证据、解释、比较、总结、升级、沉默）来实现更高质量的评审交流。

**💡 创新点**

创新点在于：①提出对话式评审的系统框架；②将会话行为量化为可实现的动作；③提出以对话价值为中心的评估维度（不确定性降低、证据生成、逻辑记录、开发者负担、校准性、进度与维护结果）。

**🔧 技术方法**

主要技术路径包括大规模语言模型（LLM）与对话管理/状态机、知识图谱/项目规范检索、多模态信息融合（代码 diff、测试结果、issue 链接等）。

**📊 数据集**

目前未使用具体公开数据集，而是建议从历史 Pull‑Request 评审线程中挖掘多轮对话、回复、修订等完整交互日志来构建对话级别的数据集。

**📈 对比分析**

文章并未给出具体实验或性能对比，主要提供了评估思路与维度，未来可通过对话式辅助实验、基准任务（CR-Bench 等）或现场实验来验证。

**⚠️ 局限性**

局限性包括：①缺乏可用的对话级别评审数据；②缺乏成熟的对话动作评估指标；③系统实现与评估仍处于理论阶段；④隐私、责任与工作流集成等实践问题待解决。

---

## 227. Spectral Prior for Reducing Exposure Bias in Diffusion Models

**arXiv ID:** 2607.22091 | [PDF](https://arxiv.org/pdf/2607.22091v1)

**作者:** Yuya Kobayashi `[一作]` (Sony AI), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出Spectral Alignment（SPA）方法，用于在扩散模型的迭代采样过程中校正频谱失配，减少曝光偏差导致的图像质量下降；

**💡 创新点**

创新点在于：①系统分析了不同模型和时间步的频率相关SNR误差；②基于训练数据拟合时序参数化功率谱先验；③采用基于FFT的引导式频谱对齐，并引入不对称惩罚以提升对低频偏差的修正；

**🔧 技术方法**

使用的技术包括FFT与径向平均功率谱（RAPS）、参数化功率谱模型（$S(t,f)=p_tf^{-q_t}+r_tf+s_t$）、Diffusion Posterior Sampling式引导、FFT梯度回传、以及现有的Classifier-Free Guidance；

**📊 数据集**

使用的数据集包括CelebA-HQ（DDPM）、ImageNet 256×256（ADM）、LAION‑Aesthetics V2（文本-图像模型SD2.0/SDXL/SD3.5/FLUX），以及对应的训练与评估图像；

**📈 对比分析**

与三类基线（ε‑rescaling、time‑shift、wavelet reweighting）进行对比，SPA在FID/KID、HPSv3、ImageReward等指标上均优于基线，提升幅度约为4–6%，且运行时间提升仅约3–4%；

**⚠️ 局限性**

局限性包括：仅使用单一频谱先验，未考虑不同数据分布（如插画、医学图像）或条件下的不同先验；未探索更复杂的引导策略或频谱权重调度；

---

## 228. Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Mode

**arXiv ID:** 2607.22083 | [PDF](https://arxiv.org/pdf/2607.22083v1)

**作者:** Nanbeige Lab `[一作]` (Nanbeige LLM Lab), Zongqiang Li `[通讯]` (Boss Zhipin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并发布了3B非嵌入参数的 Nanbeige4.2-3B，能在代码、办公室和复杂工具使用任务中表现优异，并保持较强的推理与对齐能力。

**💡 创新点**

创新点在于将 Loop Transformer 与从零开始的28T标记预训练、SFT+多阶段RL（包括 Think/Non-Think RLHF、长度控制推理RL、过程+结果奖励）相结合，打造高效通用代理。

**🔧 技术方法**

技术主要包括 Loop Transformer 结构、从零预训练、SFT 构造多样化执行轨迹、混合模式 RLHF、长度控制推理 RL、agentic RL 以及 KV 缓存共享实验。

**📊 数据集**

数据集涵盖28T多样化文本（数学、代码、合成 QA）、真实与合成环境下的执行轨迹、SFT 任务资产和 agentic scaffolds；训练时使用 28T 语料并加入 agentic 轨迹比例。

**📈 对比分析**

通过与同参数规模的 Nanbeige4-3B、Qwen3.5-4B、Gemma4-E4B 进行基准对比，在 GSM8K、BBH、MBPP、MMLU-Pro 等指标上显著领先，且在 Qwen3.5-9B、Gemma4-12B 的 agentic 基准中超越更大模型。

**⚠️ 局限性**

局限性包括仍受 3B 参数预算限制、Loop 仅深度为两次迭代、KV 缓存共享效果不佳、对长距离上下文的建模能力有限，以及在极端多任务或长时序场景下可能出现性能衰减。

---

## 229. When Language Models Meet NeuroGraphs: Exploring Enhanced Agentic LLM Framework Towards Brain Network Analysis

**arXiv ID:** 2607.22082 | [PDF](https://arxiv.org/pdf/2607.22082v1)

**作者:** Jiaxing Li `[一作]` (Southeast University), Youyong Kong `[通讯]` (Southeast University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了 BrainAgent，一个基于大型语言模型的代理框架，用来对 rs‑fMRI 构建的功能网络进行知识驱动、可解释的分类与推理。

**💡 创新点**

创新点在于将脑网络分析转化为多轮“理解‑检索‑推理‑反思”过程，结合拓扑感知的图理解、分层神经科学知识检索、双模态案例检索以及自我反思校正，显著提升了 LLM 的预测精度与解释质量。

**🔧 技术方法**

技术包括：1) 图结构化转文本并提取多层结构描述；2) HARK（分层增强检索）获取领域知识；3) CARD（双模态案例检索）匹配历史案例；4) 交互式工具调用与思考-报告-行动-观察循环；5) 反思模块校正推理结果；6) 在多种开源与闭源 LLM 上无训练直接推理。

**📊 数据集**

使用四个公开 rs‑fMRI 数据集：ABIDE（ASD/HC）、ADHD（ADHD/HC）、HCP（Gender）、Rest‑meta‑MDD（MDD/HC），每个数据集均采用 AAL 90 区域构图。

**📈 对比分析**

与直接提示、Chain‑of‑Thought、Reflection 等基线相比，BrainAgent 在准确率、召回率与精度上均取得显著提升；在 pass@3 条件下，平均提升 10‑20% 以上，且大幅降低正类偏差，精度往往跑到不同 LLM 的榜首。

**⚠️ 局限性**

局限性包括：1) 仍依赖预构建的知识库与案例库，迁移到新任务需要重新构建；2) 对大规模图结构推理的算力和推理时间有一定开销；3) 目前仅评估二分类任务，未扩展到多标签或连续指标；4) 需要进一步研究如何在本地小模型上实现高效部署。

---

## 230. PoCEvolve: Generating Proof-of-Concept Exploits from Security Patches with Vulnerability-Aware Prompt Evolution

**arXiv ID:** 2607.22076 | [PDF](https://arxiv.org/pdf/2607.22076v1)

**作者:** Duc Manh Tran `[一作]` (University of Sydney), David Lo `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种从漏洞修复提交直接生成 PoC exploit 的方法，解决缺少详细漏洞报告时的窗口期缺失证据问题。

**💡 创新点**

创新点在于引入漏洞上下文引导的 Prompt 演化框架，利用多维度评估失败提示并自适应改进生成提示，实现 patch‑only 场景下的 PoC 生成。

**🔧 技术方法**

技术包括大型语言模型（GPT‑4o‑mini、Qwen3.7‑Plus）、Commit Analyzer、Prompt Evolver（基于 GEPA 的 Pareto 优化）、自评估与验证器；利用代码差异、包元数据、运行时覆盖等信息。

**📊 数据集**

使用 SecBench.VFC.js（190 个含修复提交的 JavaScript 漏洞）和 SecBench.js（报告可用场景）进行实验。

**📈 对比分析**

与 LLM 直接提示、PoCGen 进行对比。在 patch‑only 场景下，GPT‑4o‑mini 的成功率从 48.4% 提升至 58.4%，Qwen3.7‑Plus 从 64.2% 提升至 79.5%；在报告可用场景下提升 11.1%。成本与时延相对可接受。

**⚠️ 局限性**

局限性包括仅针对 JavaScript/npm 生态；LLM 上下文长度限制导致部分失败；依赖生成器对漏洞的初步识别；未评估跨语言效果；需要并行化处理以满足实际披露窗口。

---

## 231. BioZKFHE: Scalable Encrypted Biometric Identification via Verifiable Homomorphic Similarity Evaluation

**arXiv ID:** 2607.22065 | [PDF](https://arxiv.org/pdf/2607.22065v1)

**作者:** Rundong Xin `[一作]` (Shenzhen University), Shui Yu `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 BioZKFHE 框架，实现了在区块链辅助的离线环境下对加密人脸模板进行可验证的 1:N 匹配与识别，结合了 BGV 同态加密、单系数多值（SCMV）打包以及可并行验证的相似度计算（PVSC），并通过委员会阈值开启与智能合约验证确保输出可信。

**💡 创新点**

创新点：①单系数多值打包在同态加密内层利用基‑T 表示将多条量化嵌入绑定到同一纯量化项，既大幅提高了打包密度，又消除了匹配时的旋转操作；②可并行验证相似度计算将 BGV 的 Double‑CRT 结构拆分为数千个独立证明实例，支持多核并行生成与阈值开启；③将上述两项技术整合到完整的可验证识别工作流中，首次同时解决了加密存储、匹配效率与可验证性三大瓶颈。

**🔧 技术方法**

使用技术包括：Brakerski‑Gentry‑Vaikuntanathan（BGV）同态加密；基于 lattice 的零知识证明框架与委员会阈值开启/解密；基于 Double‑CRT 的分块运算与并行证明；智能合约（Solidity）实现会话绑定、证明验收与最终决策发布；Python/SEAL 实现加密匹配与证明生成。

**📊 数据集**

实验数据集：FaceNet（128 维）与 MobileFaceNet（512 维）提取的特征向量，来源于公开人脸数据集 LFW 与 CASIA‑WebFace；使用量化因子 Γ 分别为 128 与 512，保持与浮点版相近的识别精度。

**📈 对比分析**

比较方法：以 n=1 的同态加密匹配为基准；评估指标包括加密存储、匹配延迟、PVSC 证明生成与验证时间、完整流水线时延；结果显示：SCMV 可将存储压缩至 33%（FaceNet）/ 42%（MobileFaceNet），匹配延迟随模板数线性增长；PVSC 证明生成可并行 120×+，单块验证 < 1.5 ms；完整端到端 10k–40k 模板下，Proof‑verified 运行时间约 22–44 秒，识别精度 99.47%（FaceNet）/ 98.30%（MobileFaceNet）。

**⚠️ 局限性**

局限性：①PVSC 证明体积大，公开验证与开门时仍需要显著通信和存储；②目前仅支持深度 1 的相似度计算，无法直接证明旋转、模数切换等操作；③系统依赖委员会阈值开启，若委员会成员被攻击到阈值以上仍可能泄漏信息；④实现依赖 SEAL 与 Solidity，尚未针对低功耗设备或更大规模数据做进一步优化。

---

## 232. CEL: Comprehensive Counterfactual Explanations Library and Benchmark

**arXiv ID:** 2607.22045 | [PDF](https://arxiv.org/pdf/2607.22045v1)

**作者:** Oleksii Furman `[一作]`, Maciej Zięba `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍 ACM 统一文章模板及其使用方法

**💡 创新点**

将多种 ACM 与 SIG 相关模板整合为单一统一模板，兼顾可访问性与元数据提取功能

**🔧 技术方法**

使用 LaTeX 的 acmart 文档类、Libertine 字体族、booktabs 表格包等标准 LaTeX 工具

**📊 数据集**

无（本文仅为模板说明，不涉及实验数据集）

**📈 对比分析**

无（未进行实验或性能评估，文档仅提供使用指南）

**⚠️ 局限性**

仅适用于 ACM 期刊与会议，无法评估其他出版格式的兼容性或在非 LaTeX 环境下的可移植性

---

## 233. Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs

**arXiv ID:** 2607.22039 | [PDF](https://arxiv.org/pdf/2607.22039v1)

**作者:** Zixuan Ren `[一作]` (Institute of Automation, Chinese Academy of Sciences), Chengqing Zong `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了强化学习（RL）与监督微调（SFT）对大型语言模型合并效果的影响，并系统评估了RL模型在多任务合并时的鲁棒性。

**💡 创新点**

提出RL本身通过上游数据、目标适应和正负样本联合优化，天然降低任务冲突，从而显著提升模型合并性能。

**🔧 技术方法**

采用PPO、GRPO、Reinforce++等RL算法以及模型平均、TIEs、Task-Arithmetic和DARE等四种参数合并策略。

**📊 数据集**

在数学推理、代码生成、指令遵循、逻辑谜题和排序等五个可自动评测任务的公开数据集（如GSM8K、MATH-500、HumanEval、MBPP、IFEval、LiveBench等）上进行实验。

**📈 对比分析**

通过对比SFT与RL训练模型在同一合并方法下的性能下降，发现RL模型的平均性能衰减在1–8%之间，而SFT模型平均衰减超过20%，验证了RL在合并中的显著优势。

**⚠️ 局限性**

研究局限在于仅考察了特定模型尺寸和RL算法，对更大规模模型、异构任务或在线持续学习场景的适用性尚未验证。

---

## 234. Agent Security Needs Redefinition through a Holistic Framework

**arXiv ID:** 2607.22024 | [PDF](https://arxiv.org/pdf/2607.22024v1)

**作者:** Vincent Siu `[一作]` (UC Santa Cruz), Dawn Song `[通讯]` (UC Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将代理安全重新定义为基于上下文的四个属性（源授权、任务对齐、动作对齐、数据隔离），并讨论现有防御与这些属性的对应关系。

**💡 创新点**

创新点在于把传统安全概念重新映射到代理安全领域，构建了一个统一的四属性框架，揭示了内容过滤等现有方法的结构性缺陷。

**🔧 技术方法**

主要采用概念分析与属性拆解的技术，对现有防御做属性层面映射；未提出新的算法实现。

**📊 数据集**

参考了 AgentDojo、WASP 等现有基准数据集，但并未使用或创建新的实验数据集。

**📈 对比分析**

文中未给出量化实验或性能对比，仅通过案例说明防御与四属性的对应关系，缺乏具体性能评估。

**⚠️ 局限性**

局限包括：需要在生产系统中实现细粒度的来源追踪和跨会话数据隔离；缺乏实现细节和量化实验；对复杂代理系统的可行性与评估仍待进一步验证。

---

## 235. EVL-MCoT: Enhanced Vision-Language Multi-CoT for Harmful Meme Detection

**arXiv ID:** 2607.22016 | [PDF](https://arxiv.org/pdf/2607.22016v1)

**作者:** Hao Yang `[一作]` (Yunnan University), Xuejie Zhang `[通讯]` (Yunnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于多链思考与跨模态增强解码器的恶意表情包检测框架EVL‑MCoT。

**💡 创新点**

结合多视角推理、原型引导与上下文引导解码器，显著提升视觉与文本对齐与推理鲁棒性。

**🔧 技术方法**

使用长上下文CLIP、LLaVA/GPT‑4生成多CoT、原型注意力融合、上下文注意力对齐以及温度标量的对齐损失。

**📊 数据集**

在Hateful Memes（10k+样本）和MultiOFF（743样本）数据集上进行评估。

**📈 对比分析**

与单模、双模、LLM及其他基线对比，EVL‑MCoT在Hateful Memes上取得75.88% ACC/79.25% AUROC，在MultiOFF上实现70% ACC/63.8% F1，性能显著优于对照组。

**⚠️ 局限性**

受限于高计算成本、CoT生成受LLM先验影响、推理链过长导致效率低以及对新颖文化语境的适应性不足。

---

## 236. InnoText: A Unified Model for Visual Text Generation and Editing

**arXiv ID:** 2607.22101 | [PDF](https://arxiv.org/pdf/2607.22101v1)

**作者:** Haowei Liu `[一作]` (Sun Yat-Sen University), Zhanjie Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种统一的 Diffusion Transformer 框架 InnoText，能够同时完成视觉文本的生成和编辑任务。

**💡 创新点**

主要创新点包括 Font Size‑Aware Modulation (FSAM) 模块、Small‑Character Aware Augmentation (SCAA) 策略以及任务特定区域加权损失，实现了跨尺度文字渲染与细节保持的统一解决方案。

**🔧 技术方法**

技术实现基于 Flux‑Fill/DiT 结构，结合大小映射调制、随机区域放大、flow‑matching 损失以及自监督的文本编码方法。

**📊 数据集**

构建并使用了 30K 条中英双语视觉文本数据集 InnoText‑30K 进行训练与评估。

**📈 对比分析**

与 UNet/DiT 基线（AnyText、AnyText2、Flux‑Fill、TextFlux 等）对比，InnoText 在句子准确率、NED、LPIPS 等指标上均取得最高分，尤其在中文小字体和复杂背景场景表现突出。

**⚠️ 局限性**

局限性包括大面积遮罩导致文字重复、极小字体细节仍不完美，以及复杂汉字高笔画密度结构的完整性仍有提升空间。

---

## 237. ReCowGnition: A Realistic Biometric Benchmark for Cow Face Recognition

**arXiv ID:** 2607.22071 | [PDF](https://arxiv.org/pdf/2607.22071v1)

**作者:** Marco Huber `[一作]` (Fraunhofer Institute for Computer Graphics Research IGD), Naser Damer `[通讯]` (Fraunhofer Institute for Computer Graphics Research IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并公开了ReCowGnition牛脸识别基准数据集，包含6838张161头牛的图像，并提供检测与对齐模型以及四种识别与两种验证评估协议。

**💡 创新点**

首个可复现、标准化的牛脸识别基准与评估框架，结合多场景拍摄、同一录像同识别对的排除以及多图融合策略，弥补了此前数据稀缺和评估不一致的问题。

**🔧 技术方法**

使用YOLO‑v11n进行牛脸与鼻孔检测与对齐，训练了基于ArcFace与ElasticFace-Arc的特征提取网络，并对人脸预训练模型进行微调，此外还尝试了CLIP ViT-B/16与ViT‑L/14的零样本嵌入。

**📊 数据集**

数据集为ReCowGnition，来源于德国奶牛场的五个采集会话，覆盖161头Holstein奶牛，图像尺寸为112×112。

**📈 对比分析**

采用EER、FNMR@FMR、ROC、Top‑1/Top‑5以及CMC等指标。微调模型在验证上EER仅为0.129–0.154，识别Top‑1在I_ALL上达96.39%，而零样本ViT模型表现逊色。

**⚠️ 局限性**

受限于训练样本与目标域差异、同族相似性导致识别难度大；零样本模型性能不足；未对不同品种、季节或光照变化进行进一步验证。

---

## 238. Rethinking Multi-Branch and Cross-Backbone Fusion for Vehicle Re-Identification in the Foundation-Model Era

**arXiv ID:** 2607.22068 | [PDF](https://arxiv.org/pdf/2607.22068v1)

**作者:** Yu Wang `[一作]` (Huahuan (Yunnan) Technology Co., Ltd.), Hongyu Yang `[通讯]` (Huahuan (Yunnan) Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统地评估了在基础模型时代下，多分支架构和CNN–Transformer融合是否能提升车辆Re-ID性能，并通过实验表明同一Backbone的多头在训练后会趋于冗余，跨Backbone的多样性虽然存在但无法有效融合；

**💡 创新点**

创新点在于提出了面向分支级的诊断工具和仅在收敛时评估的严谨方法，证实在DINOv3-ConvNeXt的单一强大Backbone下，额外分支或融合并不能提升性能，且单一Backbone+精确稀疏重排序即可匹配最强多分支基线；

**🔧 技术方法**

采用DINOv3预训练的ConvNeXt、ViT-L、相互交叉注意力融合、分阶段学习率衰减、全层微调、精确稀疏k-相互最近邻重排序、LoRA参数高效微调、CKA相似度、Jaccard、遮挡显著性、互补正确率、配对Bootstrap置信区间等技术；

**📊 数据集**

使用VeRi-Wild（Small/Large）和VeRi-776数据集；

**📈 对比分析**

与官方跨摄像头协议下的多分支元数据依赖基线进行mAP和R1比较，单一DINOv3-ConvNeXt在VeRi-Wild Small、Large分别达到88.19/77.47 mAP，重排序后提升至92.38/83.68；融合仅能在最优时提升≤+0.11 mAP，单一Backbone已超越双Backbone融合与其他方法；

**⚠️ 局限性**

实验仅涵盖单一基础模型族、256px分辨率、单随机种子；不同预训练线、不同数据规模或不同融合设计可能得到不同结论，且局部监督或多种训练策略的影响尚未探究。

---

## 239. HyperLogLog for probabilists

**arXiv ID:** 2607.22063 | [PDF](https://arxiv.org/pdf/2607.22063v1)

**作者:** Lucas Gerin `[一作]` `[通讯]` (Université Paris Nanterre), Lucas Gerin (Université Paris Nanterre)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

对HyperLogLog算法进行了严谨的非渐近偏离概率分析，给出了左尾和右尾的指数上界；

**💡 创新点**

创新点在于用概率论中的Chernoff方法和凸性论证，避免了原论文中复杂的泊松化和Mellin变换分析，得到更直观且完全非渐近的偏离不等式；

**🔧 技术方法**

主要技术包括：随机正则模型、上象限次序比较、拉普拉斯变换估计、凸/凹函数论证以及Chernoff方法；

**📊 数据集**

未使用任何真实数据集，整个工作以理论推导为主；

**📈 对比分析**

与原始文献中的Chebyshev上界和先前经验结果比较，证明了所得到的指数上界在m足够大时显著更强，尤其在左尾偏离上界上可达O(exp(-m·J(μ)))；

**⚠️ 局限性**

局限性：右尾上界仅在λ≥2时成立，且需要对小N做额外修正；对于极小样本（N≪m）仍无有效的集中不等式；

---

## 240. A Smooth Phase-Separation Model for Weak-Boundary Segmentation of Homogeneous Structures

**arXiv ID:** 2607.22053 | [PDF](https://arxiv.org/pdf/2607.22053v1)

**作者:** Zihan Li `[一作]` (Harbin Institute of Technology), Zhichang Guo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于Cahn–Hilliard相分离的平滑多相软最大化分割模型，用于弱边界、同质外观结构的分割；

**💡 创新点**

创新点在于将Cahn–Hilliard相分离正则化与软最大化区域拟合耦合，采用混合L²–H⁻¹梯度流和稳态SAV-FFT数值分裂方案，保证能量耗散、解的存在唯一性与高阶光滑性；

**🔧 技术方法**

技术包括软最大化（softmax）区域拟合、Cahn–Hilliard四阶相场正则化、混合L²–H⁻¹梯度流、能量分裂与标量辅助变量（SAV）稳定线性离散、FFT谱求解；

**📊 数据集**

使用合成低对比弱边界图像、医学影像（如SKI10皮肤镜图像）以及带噪声的低对比图像作为测试数据；

**📈 对比分析**

与CV、RSF、CP-ICTM、MBE等传统变分/相场模型以及U‑Net、TransUNet、Swin‑UNet、MedSegDiff等深度学习方法进行对比，实验表明该方法在Dice、IoU指标上接近深度学习模型，在HD95（边界误差）上表现更为稳定，且不需训练；

**⚠️ 局限性**

局限性包括对软最大化参数β与Cahn–Hilliard长度尺度ε的敏感度仍需手工调参；对高维大规模图像的计算成本（FFT和多组相场）仍较高；

---

## 241. Scaling Native Multimodal Pre-Training From Scratch

**arXiv ID:** 2607.22043 | [PDF](https://arxiv.org/pdf/2607.22043v1)

**作者:** Haoyuan Wu `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了原生多模态预训练的计算最优缩放规律，构建了语言与多模态目标的计算最优前沿。

**💡 创新点**

创新点在于将语言和多模态目标分离，发现其缩放行为不同；提出依赖数据混合比例的Pareto前沿；证明多模态预训练既保留文本能力，又提升空间推理和多模态上下文学习。

**🔧 技术方法**

使用了IsoFLOP曲线、训练曲线包络和计算-损失功率律等技术，并采用解码器型 MoE Transformer，将图像映射为连续 patch embeddings 进行直接处理。

**📊 数据集**

使用的数据集包括 250B 文本（Web、书籍、论文等）和 75B 图文对/文档（web‑crawled 图像‑文本），共计 250B 文本 token 与 75B 多模态 token。

**📈 对比分析**

通过在 16 个文本基准和 23 个多模态基准上进行 0/1/3 shot 评估，使用准确率、Pass@1 等指标；结果显示多模态预训练对文本性能无损，显著提升空间推理和多模态上下文学习，性能相比纯文本模型提升数个百分点。

**⚠️ 局限性**

局限性：实验仅覆盖至 3B 参数规模；仅使用单一图像‑文本数据家族；使用训练损失作为测试代理，未检验更大规模、多模态种类或不同数据混合比例下的表现。

---

## 242. Exponentially Fewer-Server PIR from Sparser $S$-Decoding Polynomials

**arXiv ID:** 2607.22033 | [PDF](https://arxiv.org/pdf/2607.22033v1)

**作者:** Aparna Gupte `[一作]` (Massachusetts Institute of Technology), Seyoon Ragavan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在私人信息检索（PIR）协议中，研究并构造了稀疏度为 k+1 的 S^*_m-解码多项式，从而实现了 s 服务器的高效 PIR，显著降低了通信复杂度

**💡 创新点**

创新点在于通过引入“根-单位网格（root‑of‑unity grid）”概念，将解码多项式的稀疏性与数论中粗鲁循环数（rough repunits）和素数分布联系起来，证明了稀疏度仅为 k+1 就可达到下界，填补了之前仅能得到 2^k 稀疏解码多项式的空缺

**🔧 技术方法**

主要技术包括：1) 证明解码多项式稀疏性与根‑单位网格等价；2) 利用粗鲁循环数与 Weil 角色和 Dickman 平滑性假设的数论技巧来构造根‑单位网格；3) 通过将多个根‑单位网格在同一特征下拼接，获得更低通信复杂度的多服务器 PIR；4) 采用匹配向量族（matching vector families）构造大规模 PIR

**📊 数据集**

本工作为理论性质研究，不使用具体实验数据集，所有结论均基于数论假设与抽象分析得出

**📈 对比分析**

与传统的 2^ℓ‑稀疏解码多项式方法相比，在常数服务器和中等服务器数量下，通信复杂度可从 exp(exp(Õ(loglog n))) 降至 exp(exp(Õ(loglog n / s)))，在大多数参数范围内显著优于现有最优方案；对于服务器数极大时，改进不明显，仍退回到 2^ℓ‑稀疏解码多项式的表现

**⚠️ 局限性**

局限性主要是：1) 依赖未被证明的数论猜想（如通用循环数质数猜想、Schinzel 的假设 H、粗鲁循环数分布等）；2) 对于极大服务器数（s ≥ exp(Ω(√(loglog n / logloglog n)))）时，方法与旧方案没有实质性提升；3) 该构造需要非常大的有限域和模数，实际实现的硬件成本尚未评估

---

## 243. Embodying Multi-Hand Manipulation Policies by Searching the Assignment and Null Spaces

**arXiv ID:** 2607.22020 | [PDF](https://arxiv.org/pdf/2607.22020v1)

**作者:** Yorai Shaoul `[一作]` (Carnegie Mellon University), Maxim Likhachev `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将学习到的多手操作策略输出的末端执行器轨迹映射到多臂机械臂上，并在执行时同时解决轨迹-臂分配、冗余关节规划与碰撞避免等问题。

**💡 创新点**

提出“On‑Manifold Conflict‑Based Search with Assignments（Om‑CBSA）”框架，首次将多臂轨迹分配与冗余空间探索统一到单机器人A*搜索中，并在高层利用CBS完成分配冲突与几何冲突的完整化解；理论上证明其对离散化轨迹的解析度完整性。

**🔧 技术方法**

核心技术包括：基于姿态约束的单机器人A*搜索（On‑Manifold A*）、多目标A*用于隐式分配决策、Null‑Space Jacobian探索冗余解、Conflict‑Based Search（CBS）框架处理分配与碰撞约束、以及对机械臂的Kinematics/IK求解。

**📊 数据集**

在模拟环境下生成450个基准问题（最多6条轨迹、6条臂、含障碍），以及在真实3臂Kinova Gen3平台上进行布料旋转、箱体翻转、平面推动等实验。

**📈 对比分析**

与三类基线（IK‑Tracking、PP‑Descartes、Composite A*）进行比较。Om‑CBSA及其优先化变体在成功率（>90%）和平均求解时间（≈640 ms）上均优于基线；优先化版本在大多数问题中均能在1 s以内求解。

**⚠️ 局限性**

局限性包括：在臂数极多时搜索空间仍然指数增长；优先化变体虽然快但不完整；对离散化分辨率敏感，过粗分辨率可能导致失效；实验规模虽已扩大至6臂，但在更大尺度或更复杂动态环境中的性能仍待验证。

---

## 244. autotn: Automata-inspired construction of tensor-network operators from symbolic local rules

**arXiv ID:** 2607.22232 | [PDF](https://arxiv.org/pdf/2607.22232v1)

**作者:** Aitor Morais `[一作]` (University of Deusto), Eneko Osaba `[通讯]` (Tecnalia)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一款开源 Python 工具 autotn，用于将符号化的局部算符规则自动转换为矩阵乘积算符（MPO）并进行验证，消除了手动组装 MPO 的繁琐过程。

**💡 创新点**

创新点在于采用基于有限自动机的前缀/后缀分组策略，能够重用符号串中的重复模式，从而生成更紧凑的张量网络，并提供统一的验证与诊断框架。

**🔧 技术方法**

技术实现包括 Python/NumPy、pytest、Matplotlib；核心算法为自动机启发式 MPO 构造、中心点自动选择、稠密验证与性能基准；同时支持对角向量与一般局部矩阵两种后端。

**📊 数据集**

使用的数据集主要是三类模拟实例：Max-Cut 图权重矩阵、长程 XX+YY+Z 体系的耦合与场强向量、以及不同本征维数的量子时钟 Hamiltonian，所有数据均通过脚本随机生成。

**📈 对比分析**

比较方法在构造时通过统计张量元素总数、最大虚拟维数和构造时间来评估不同中心选择策略；在验证时对小规模系统进行完整稠密矩阵对比，结果显示无错误且误差仅为数值舍入误差，构造时间相对基准略增（约 1.4–1.6 倍）。

**⚠️ 局限性**

局限性包括：仅支持对角向量和普通局部矩阵的 MPO；仅在小系统上做稠密验证，无法覆盖大 Hilbert 空间；未集成完整的张量网络求解器或符号简化；中心选择仍基于尺寸启发式，可能不适用于所有模型。

---

## 245. Offline Vision-Language Navigation with Geometric Goal Localization for Outdoor Environments

**arXiv ID:** 2607.22226 | [PDF](https://arxiv.org/pdf/2607.22226v1)

**作者:** Ali Salmasi `[一作]` (University of Turku), Tomi Westerlund `[通讯]` (University of Turku)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Edge-BehAV，一套完全离线的视觉‑语言导航系统，可在户外无网络环境下运行；

**💡 创新点**

创新点在于：①系统化评估 17 种边缘小语言模型，选出与云端相当且推理更快的模型；②设计轻量化的语义‑几何融合目标定位框架，将目标定位误差从 2.05 m 降到 0.20 m；③将上述模块集成为完整的行为引导规划体系，完成 31/32 场景的闭环导航；

**🔧 技术方法**

采用小语言模型（如 Qwen2.5‑7B）、开源检测器 Florence‑2、分割模型 Mobile‑SAM、LiDAR‑相机融合、混合 MPC 优化等技术；

**📊 数据集**

使用 350 条人工标注的导航指令（七类），922 张 nuImages 图像进行检测基准，4 种户外目标（交通锥、汽车、自行车、垃圾箱）做距离基准；

**📈 对比分析**

与云端 GPT‑4/GPT‑5.5、BehAV 传统管线进行对比；离线模型在指令解析上与 GPT‑5.5 质量相当，推理速度快 9 倍；目标定位误差提升 10×；系统成功率 96%（31/32 试验）且行为遵循率超过 90%；

**⚠️ 局限性**

局限包括：对“上下文/对抗”类指令仍难以完全理解；低反射目标（如远处交通锥）在 LiDAR 取样不足时需退回视角定位；系统对极端天气与复杂动态障碍的鲁棒性仍待进一步验证。

---

## 246. Ice Walk is ASP-Complete

**arXiv ID:** 2607.22224 | [PDF](https://arxiv.org/pdf/2607.22224v1)

**作者:** Papangkorn Apinyanon `[一作]` `[通讯]` (Massachusetts Institute of Technology), Papangkorn Apinyanon (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过构造多项式时间的映射，证明了冰路（Ice Walk）循环谜题的解搜索问题属于ASP‑完整（ASP-complete）类；

**💡 创新点**

创新点在于直接将源图的顶点映射为单个编号为1的干燥格子，利用“可视性轨迹”将相邻顶点的连边转化为沿直线的干燥段，避免了传统的多格子元胞（metacell）模拟，从而简化了从格子图Hamiltonian环到谜题解的归约过程；

**🔧 技术方法**

所用技术包括ASP‑完整性框架、格子图Hamiltonian环问题、格子图的最大度3子图构造、坐标与可视性引理、以及对应的双射证明；

**📊 数据集**

论文采用构造性证明，未使用任何真实数据集，仅依赖理论构造的实例；

**📈 对比分析**

本研究没有进行实验比较或性能评估；其贡献是理论证明的复杂度结果，即解搜索问题的每一个解都能在多项式时间内与源问题对应；

**⚠️ 局限性**

局限性在于仅给出了理论复杂度证明，对实际求解器的效率或算法实现细节未作讨论；此外，ASP‑完整性说明问题在NP层面仍然困难，未提供可行的多项式时间解法。

---

## 247. Closed-Loop Generative Selection: Convergence, Memory, and Noisy Oracles

**arXiv ID:** 2607.22211 | [PDF](https://arxiv.org/pdf/2607.22211v1)

**作者:** Konstantin Fackeldey `[一作]` (Zuse Institute Berlin), Christof Schütte `[通讯]` (Zuse Institute Berlin)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提供闭环生成式选择算法的收敛、跑时与评估成本理论，并探讨记忆深度与噪声对性能的影响。

**💡 创新点**

引入了对自适应生成模型的马尔可夫化，提出层级逃逸概率与非单调退出时间分析，给出最优记忆深度和噪声鲁棒性的评估最小化策略。

**🔧 技术方法**

采用马尔可夫链理论、水平法、逃逸概率与记忆层级分析、稳态与漂移方法以及鲁棒统计估计（中值-均值、截尾均值、分位数检验）。

**📊 数据集**

在离散的 OneMax 任务上进行可重复实验验证，亦使用化学分子空间的模拟测试。

**📈 对比分析**

与无学习的均匀采样基线和不同记忆深度的生成器进行对比，证明学习模型几乎必然收敛，最优记忆深度可显著缩短迭代次数，M=1 时评估成本最小。

**⚠️ 局限性**

逃逸概率估计依赖于未知的模型性能，理论对记忆的单调性假设有限，且在连续空间与实际药物评估噪声分布上尚需进一步验证。

---

## 248. From Isolated Tasks to Structured Capabilities: A Multilayer Taxonomy for Large Language Models

**arXiv ID:** 2607.22182 | [PDF](https://arxiv.org/pdf/2607.22182v1)

**作者:** Shixin Fang `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个以人类认知为参考的多层级认知能力分类体系，包含14个能力域和91个子技能，并将其应用于对 2023‑2025 年 ACL、AAAI、ICML、NeurIPS 论文的系统映射，揭示了 LLM 研究在不同能力域、子技能和能力组合上的分布与偏好。

**💡 创新点**

创新点在于：① 以发展先行性和功能支持为依据，将认知能力分为 Primitive、Constructed、Integrative 三层；② 通过可操作化的注释码表将抽象心理学构造映射为可观测的 LLM 评估指标；③ 结合三模型协同注释与仲裁，实现了对 15,934 篇 LLM 论文的高效、可复现的能力级映射，为后续评测、诊断与训练提供统一框架。

**🔧 技术方法**

主要技术包括：多模型注释管道（DeepSeek V3.2、Kimi K2.6、Claude Sonnet 4.6）、结构化输出校验与一致性投票、基于共现和提升（lift）计算的能力关系分析。

**📊 数据集**

使用的数据集为 31,505 篇在 ACL、AAAI、ICML、NeurIPS 2023‑2025 会议论文的完整 PDF，经过筛选后得到 15,934 篇 LLM 相关论文，用于注释与统计。

**📈 对比分析**

比较方法是通过对强证据（direct research target）与弱证据（supportive/contextual）的统计，计算各能力域在论文中的出现比例、子技能覆盖率以及能力域之间的共现频数与 lift；结果显示语言语义、推理、规划与感知被过度关注，而情绪、理论心智等在 2% 以下，且子技能集中度高，跨域共现以语言-推理为主，社会-文化-道德组合呈现高 lift 但整体量少。

**⚠️ 局限性**

局限性包括：① 分类体系为假设性框架而非 LLM 内部架构；② 注释基于论文文本，可能存在概念歧义与非排他性标签导致共现模式仅反映研究组织而非因果关系；③ 仅覆盖 2023‑2025 年四大会议，未必代表整个 LLM 文献；④ 需进一步实验验证层级关系对训练与迁移的影响。

---

## 249. DBA-Bench: A Production-Fidelity Benchmark for LLM-Based Database Operations Agents

**arXiv ID:** 2607.22165 | [PDF](https://arxiv.org/pdf/2607.22165v1)

**作者:** Junming Chen `[一作]` (University of Electronic Science and Technology of China), Kai Zheng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个名为DBA-Bench的生产环境模拟基准，用于评估大语言模型驱动的数据库操作代理在多轮交互、实时读写、完整故障恢复和安全约束下的表现。

**💡 创新点**

创新点在于：①实现了多源观察、持续工作负载和可恢复状态的“生产真实性”；②采用“结果优先”的评估协议，分离诊断、恢复和安全三维度；③通过可复现的快照重置和情景特定验证，保证每次测试在相同的故障条件下进行；④定义两类难度标签（诊断深度与环境噪声），可细化性能瓶颈。

**🔧 技术方法**

技术包括：PostgreSQL实例化与监控接口（指标、日志、SQL、实例管理、知识库），ReAct思考‑动作‑观察循环，GPT‑5.5/Claude‑Opus等大型语言模型，基于树搜索的D‑Bot和知识图谱驱动的DBAIOps两种代理架构，安全规则引擎和可执行的成功契约。

**📊 数据集**

使用了106个手工构造、已验证的生产级故障场景（涵盖查询调优、系统故障、周期性健康检查、业务变更、资源治理、复合故障与误报），每个场景在8个自动化基线和1个人类DBA上执行，总计848次自动化运行与106次人工运行。

**📈 对比分析**

比较方法是单次跑通（pass@1）评估安全通过率（Safe Pass）为首要指标，同时报告诊断通过率（Diagnosis Pass）、恢复通过率（Outcome Pass）和每跑成本（Token Cost）。结果显示：最佳自动化系统（GPT‑5.5 ReAct）Safe Pass为17.9%，诊断和恢复分别为32.7%和19.6%；人类DBA的Safe Pass高达93.4%；安全通过率在Easy→Hard、浅层→深层诊断以及低噪声→高噪声场景下显著下降。

**⚠️ 局限性**

局限性包括：①自动化代理仍远低于人类水平，尤其在安全与多因果链诊断上表现薄弱；②基准场景数量有限，缺乏跨数据库引擎（如MySQL、Oracle）的泛化；③安全评估主要基于规则，未覆盖更细粒度的业务约束；④实验仅使用单个LLM模型版本，未充分探索模型多样性与推理策略对性能的影响。

---

## 250. Visual Relocalization from Sparse Views in Aliased and Low-Texture Environments via Novel View Synthesis

**arXiv ID:** 2607.22147 | [PDF](https://arxiv.org/pdf/2607.22147v1)

**作者:** Maria Peribañez `[一作]` (University of Zaragoza), Riccardo Giubilato `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对行星类环境中低纹理、感知混淆和前进运动导致的稀疏视角，提出基于3D Gaussian Splatting的视觉重定位方法。

**💡 创新点**

创新点在于将MVS产生的深度和法向量以及LiDAR点云的Chamfer对齐作为几何监督，形成全新的几何感知训练策略。

**🔧 技术方法**

使用3D Gaussian Splatting、MVSAnywhere、Chamfer损失、SALAD+FAISS检索以及6DGS姿态估计。

**📊 数据集**

使用DLR S3LI Vulcano数据集的moon_lake序列（同步RGB、LiDAR和GNSS标定）。

**📈 对比分析**

与仅光度监督的3DGS和传统PnP基准相比，几何监督模型在10m/15°阈值下召回率从6.25%提升到43.2%，在2m/10°阈值下从0%提升到6.8%，同时旋转误差显著下降。

**⚠️ 局限性**

仍受限于稀疏前进视角导致的深度不确定、数据规模小、对LiDAR的依赖以及对极端光照变化的鲁棒性待进一步验证，整体召回率仍不高。

---

## 251. Flight-Ready LiDAR-Inertial Odometry for Embedded Drone Platforms

**arXiv ID:** 2607.22145 | [PDF](https://arxiv.org/pdf/2607.22145v1)

**作者:** Alvaro J. Gaona `[一作]` (Universidad Politécnica de Madrid), Pascual Campoy `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

改进并验证了 FAST‑LIO2 等开源 LiDAR‑惯性里程计（LIO）系统的架构，使其能在无人机上实现 200 Hz 的姿态/速度发布、线程分离、速度输出以及对短时 LiDAR 失效的鲁棒处理。

**💡 创新点**

核心创新在于：1) 在 IMU 采样率下前向传播姿态与速度并发布；2) 直接在 body‑frame 输出完整 Twist；3) 对速度做 EMA、对姿态做 SLERP 平滑；4) 采用双执行器隔离 IMU 回调与 LiDAR/ESKF 处理；5) 解决多线程同步竞态并实现链式前向传播以维持 LiDAR 失效时的连续性。

**🔧 技术方法**

技术手段包括 IESKF 前向传播（SO(3) 指数映射）、指数加权平均（EMA）、球面线性插值（SLERP）、ROS 2 多线程执行器、互斥锁保护、链式前向传播、以及姿态/速度的体坐标直接计算。

**📊 数据集**

使用了 Livox Mid‑360 LiDAR 与 Pixhawk 4 Mini 自主 UAV 的实飞数据，并配合 16 台 OptiTrack 运动捕捉摄像头提供的 100 Hz 子毫米级 6 DOF 轨迹作为地面真值。

**📈 对比分析**

通过在同一 58 s/60 s 飞行轨迹上对比原始 FAST‑LIO2 与改进版，评价指标包括发布频率、ATE、RPE、速度跟踪误差以及 LiDAR 失效时的漂移。改进后发布率从 10 Hz 稳定到 200 Hz，ATE RMSE 由 11.7 cm 降至 5.2 cm，RPE（1 s）从 4.9 cm 降到 2.4 cm，速度误差 ≤ 0.1 m/s，LiDAR 失效时漂移约 0.49 m。

**⚠️ 局限性**

局限性：改进主要针对 IESKF‑based LIO，其他如 factor‑graph 或连续‑时间 LIO 的兼容性尚未完全验证；链式前向传播在长时间失效时误差会累积；在极端高速或大幅度滚转等激烈运动下的鲁棒性仍需进一步评估。

---

## 252. Why Large Language Models and Humans Converge and Diverge in Evaluating Creativity

**arXiv ID:** 2607.22218 | [PDF](https://arxiv.org/pdf/2607.22218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 253. TriGlue: a Biology-Inspired Generative Model for Generating Molecular Glue-Induced Ternary Complex

**arXiv ID:** 2607.22143 | [PDF](https://arxiv.org/pdf/2607.22143v1)

**作者:** Yuliang Yan `[一作]` (Hong Kong University of Science and Technology), Enyan Dai `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种名为TriGlue的生成式框架，用于设计分子黏合剂（molecular glue）并同时构建三元复合物（E3‑ligase/目标蛋白/小分子）。

**💡 创新点**

创新点在于将三元复合物生成拆解为“接口估计”和“接口条件下的生成”两步，并用SE(3)-等变的接口估计模块预测软接口，随后通过接口驱动的流匹配网络同时生成小分子和靶蛋白的刚体位姿，实现端到端的分子黏合剂设计。

**🔧 技术方法**

技术上使用了SE(3)-equivariant EGNN进行接口编码、椭圆参数化接口对齐、界面条件下的三元流匹配网络（包含旋转、坐标、原子类型流）以及Invariant Point Attention编码器。

**📊 数据集**

训练与评估基于TernaryDB数据集（共22,303条三元复合物），并采用MMseqs2对复合物进行聚类以避免数据泄露。

**📈 对比分析**

与AR、TargetDiff、PocketXMol（分子生成）以及DeepTernary、EquiDock、DiffDock-PP（蛋白‑蛋白对接）进行对比。TriGlue在平均RMSD（5.22Å）、Vina评分、QED等指标上均优于或相当于基线，并在DockQ和成功率方面实现了更高的对接精度（DockQ≈0.22 vs 0.15-0.17）。

**⚠️ 局限性**

局限性包括：对接口预测的依赖度高，若接口估计误差大会显著影响生成质量；模型规模较大，训练与推理成本较高；目前仅在TernaryDB上验证，泛化到更广泛的靶点/配体空间仍需进一步测试。

---

## 254. On the Fragility of Majority Illusions

**arXiv ID:** 2607.22132 | [PDF](https://arxiv.org/pdf/2607.22132v1)

**作者:** Maaike Venema-Los `[一作]` (University of Groningen), Davide Grossi `[通讯]` (University of Groningen)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了多数幻觉在两种情境下的脆弱性：一是意见随时间通过同步或异步多数更新演化；二是随着网络规模增大时，随机图上多数幻觉出现的概率。通过理论分析，证明异步更新下多数幻觉永不稳定，且同步更新后最多只会在长度为 2 的循环中持续出现；进一步利用局部收敛（Benjamini‑Schramm）与泊松分支树极限，证明在 Erdős‑Rényi 随机图（无论采用 i.i.d. 颜色分配还是基于度的颜色分配）中，网络规模趋于无穷大时，出现多数幻觉的概率收敛到零。

**💡 创新点**

创新点在于：①首次从意见动力学角度证明多数幻觉在异步和同步更新过程中的非持久性；②提出并验证一个通用的证明框架——将随机图通过局部收敛映射到极限树，进而推导多数幻觉概率趋于零；③展示该框架对多种颜色分配（i.i.d. 与度相关）均适用，为后续研究更复杂色彩分布（如同质化）提供方法论。

**🔧 技术方法**

主要技术包括：
- 多数更新模型（异步与同步）
- 图稳定性与循环周期分析
- 局部收敛（Benjamini‑Schramm）与概率图论
- 泊松分支树（Galton‑Watson）极限
- 概率与期望计算（如二项分布与泊松分布混合）

**📊 数据集**

使用的“数据集”是理论构造的随机图序列：Erdős‑Rényi 图 G_{n,p=c/n} 随着 n→∞ 的极限行为；没有使用实际社会网络数据。

**📈 对比分析**

方法比较主要基于理论推导：作者通过证明和极限结果展示多种设置下多数幻觉出现概率趋于零，未进行实验或数值模拟。性能指标相当于“概率是否趋于 0”，且在所有考察的随机图序列与颜色分配下均满足该结论。

**⚠️ 局限性**

局限性包括：
- 仅研究 Erdős‑Rényi 随机图，未涵盖更真实的重尾度分布网络；
- 颜色分配仅考虑独立分布（i.i.d. 或度相关），未涉及同质化或更复杂的 Gibbs 过程；
- 研究仅限于静态图结构的更新，未考虑网络拓扑随时间演化；
- 结果为理论性质，缺乏经验验证或实验支持。

---

## 255. TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution

**arXiv ID:** 2607.22231 | [PDF](https://arxiv.org/pdf/2607.22231v1)

**作者:** Sicheng Gao `[一作]` (Advanced Micro Devices Inc.), Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过动态令牌路由与合并实现一阶段扩散视频超分辨率的高效推理与时间一致性提升

**💡 创新点**

创新点在于结合语义‑时间重要性评分、离线路由规划与双流局部/全局令牌处理，并在路由出口实现身份恢复，以精细控制算力分配

**🔧 技术方法**

技术包括一阶段扩散Transformer（DiT）、文本相似度+时序曲率指导的令牌重要性评估、离线风险规划的路由区间选择、双流（高保真局部 + 低成本全局）令牌合并与身份恢复

**📊 数据集**

使用的基准数据集包括合成U​DM10、SPMCS、YouHQ40以及真实场景RealVSR、MVSR4x和VideoLQ

**📈 对比分析**

与RealESRGAN、ResShift、RealBasicVSR、Upscale‑A‑Video、MGLD‑VSR、STAR、DOVE等方法在PSNR/SSIM/LPIPS/DISTS/CLIP‑IQA及E^*_warp等指标上均实现或逼近最佳表现，同时显著加速推理速度（高效路由+两流处理）

**⚠️ 局限性**

局限性包括：一阶段模型在细纹理、阴影及快速局部运动区域仍可能出现轻微振荡；速度提升受限于VAE解码和实现细节；路由区间依赖离线校准，迁移到新骨架或不同降解场景时可能需重新调参

---

## 256. Filling Before Advancing: Capability-Gap-Driven Post-Training for Scenario-Specialized Remote Sensing MLLMs

**arXiv ID:** 2607.22205 | [PDF](https://arxiv.org/pdf/2607.22205v1)

**作者:** Yuheng Zong `[一作]` (Nankai University), Jon Atli Benediktsson `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了如何将遥感多模态大语言模型（RS-MLLM）从通用预训练迁移到特定场景（港口监测）并提出了一种基于能力缺口填补的后训练路径（FBA）。

**💡 创新点**

创新点在于把场景专用化拆分为三阶段——RS语义锚定、跨模态桥接收敛、证据驱动场景微调，并通过构建分层监督数据集（CPRS）实现逐步能力填补。

**🔧 技术方法**

技术手段包括使用 LLaVA‑v1.5 / Qwen3‑VL 两大多模态 LLM，结合 LoRA 微调、分阶段教师蒸馏（SMT）和多源数据构建与验证流程。

**📊 数据集**

所用数据集为三层监督集 CPRS（569k RGB 图文、187k 多源 SFT 样本、53k 港口指令），评测基准为 HarborEval 八轨道、VRSBench/RSVQA 港口子集以及 OpenEval。

**📈 对比分析**

在相同训练预算下，FBA 相较于直接 SFT 和 Collapsed‑SFT 在 HarborEval 上分别提升约 12.3 分（LLaVA）和 2.28 分（Qwen3‑VL），并在公开子集与 OpenEval 上优于现有 RS‑MLLM。

**⚠️ 局限性**

局限性包括对高质量港口数据的稀缺依赖、跨模态覆盖不完整、对细粒度视觉细节的理解仍有限，以及验证主要集中在港口场景，缺乏对其他遥感场景的泛化评估。

---

## 257. LayoutLite: Token-Level Implicit Layout Analysis for Efficient Document OCR

**arXiv ID:** 2607.22200 | [PDF](https://arxiv.org/pdf/2607.22200v1)

**作者:** Xudong Liu `[一作]` (Yuanli Technology Co., Ltd.), Yulin Jin `[通讯]` (Yuanli Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在OCR任务中提出了轻量化、可插拔的模块LayoutLite，用于在视觉编码器和语言解码器之间对视觉tokens进行隐式布局分析并压缩不重要的tokens，从而加速VLM基准的文档识别。

**💡 创新点**

创新点在于：①在token层面进行隐式布局分析而非显式检测；②利用跨层特征演变的1D卷积估计token重要性；③通过GRPO强化学习在未标注数据上训练，并辅以自动布局监督，获得高效且鲁棒的token压缩。

**🔧 技术方法**

采用视觉语言模型、跨层特征聚合、轻量级1D卷积评分网络、GRPO强化学习、布局监督、K-means阈值分割等技术实现。

**📊 数据集**

训练使用未标注的Document Parsing子集，布局监督使用PP-DocLayoutV3生成的边界框；评估在OmniDocBench v1.7上进行。

**📈 对比分析**

与基线FireRed‑OCR、Logics‑Parsing‑V2及FastV/PixelPrune等方法对比，在压缩率低于50%时保持92–93% OCR分数，仅以≈40%减少预填充延迟、FLOPs和KV缓存；超过55%压缩率性能显著下降。

**⚠️ 局限性**

限制在于：对极高压缩率（>55%）下效果不佳；依赖VLM保持冻结，训练仍需一定量无标签数据；只在少数OCR模型上验证，通用性待进一步验证。

---

## 258. Online Geometric Packing through Online TSP Scheduling

**arXiv ID:** 2607.22179 | [PDF](https://arxiv.org/pdf/2607.22179v1)

**作者:** Anders Aamand `[一作]`, Csaba D. Tóth `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在线 TSP 调度与在线平面凸多边形平移打包问题的新理论框架，并给出了多种在线翻译打包（如条带、箱子、周长、面积、超球体）算法；

**💡 创新点**

将在线排序的技术反向利用，用在线 TSP 调度构造近似路径，从而实现从 O(n^0.59) 竞争比降至 O(log²n)，同时在多维超球体打包中实现 O_d(log²n) 竞争比；

**🔧 技术方法**

核心技术包括：Azar‑Panigrahi‑Vardi 的 elementary tree、基于度量 TSP 调度的分布式插入与时间调度、dyadic parallelogram 归一化、箱子打包框架与多重度量映射；

**📊 数据集**

无实验数据集，全部为理论分析与证明；

**📈 对比分析**

与之前最优竞争比 O(n^0.59) 以及现有下界 Ω(√(log n/log log n)) 对比，本文实现了多项问题的极大改进，竞争比提升至 O(log²n)；

**⚠️ 局限性**

仍受下界限制，无法突破 polylog 级别；在三维凸多面体打包上无法给出多项式近似；对非平移/旋转不适用，且对极端形状（如大半径球体）竞争比仍为 Ω(n^{d‑1/d(d+1)})。

---

## 259. Unbiased Open World Regularization for Fair Self-Supervised Learning

**arXiv ID:** 2607.22149 | [PDF](https://arxiv.org/pdf/2607.22149v1)

**作者:** L{é}o Nicollier `[一作]`, Gabriele Facciolo `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无监督的自监督学习正则化框架 UOWReg，通过在已知的离散偏差属性上执行条件分布匹配来实现表征的公平性和去偏。

**💡 创新点**

核心创新在于把全局正则化转为条件分布匹配，理论证明能保证表示与偏差属性统计独立，并将此前经验性的去偏方法（EnD、FSCL）统一到同一几何框架下；同时在多种目标分布（高斯、球面）下实现无偏学习。

**🔧 技术方法**

采用自监督对齐损失、统计差异测度（MMD、KL）在球面或高斯目标分布下进行条件正则化，配合投影层、热核或高斯核实现无生成模型的编码器训练。

**📊 数据集**

在三个数据集上验证：Colored MNIST（彩色数字）用于可视化与k-NN评估；CelebA（人脸属性）用于公平性评估（Equalized Odds）；Synthetic Engraving（合成纹理+二进制代码）用于检验在强关联偏差下的子群体分解。

**📈 对比分析**

与基线 OW、FSCL、FSCL⁺、FSCL† 进行对比。UOWReg 在 Colored MNIST 上将偏差识别率从 100% 降到约 75% 并提升目标准确率；在 CelebA 上在球面目标下 Equalized Odds 下降至 1.7% 同时保持 81.5% 以上分类准确率；在 Synthetic Engraving 上在 m=2 时检索 mAP 提升至约 91%（相较 OW 的 76%），并随 m 增大趋同。

**⚠️ 局限性**

局限性：仅针对离散偏差属性；对高相关度的偏差-语义混合（如 m=1）仍无法完全去偏；在极低方差或完美一一对应的情况下仍需进一步研究无生成模型的解决方案。

---

## 260. No Edges, No Verdict: A Large-Scale Empirical Study of Declared Dependency Graphs in 78K SBOMs in the Wild

**arXiv ID:** 2607.22140 | [PDF](https://arxiv.org/pdf/2607.22140v1)

**作者:** Artur Zięba-Kozarzewski `[一作]` `[通讯]` (KRYPTON Polska), Artur Zięba-Kozarzewski (KRYPTON Polska)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对78,612份真实SBOM进行大规模拓扑分析，揭示约一半无边、约10%为孤立退化图；

**💡 创新点**

首次量化SBOM依赖图完整性缺失，并提出开放世界语义与孤立检测方法以避免闭世界误判；

**🔧 技术方法**

采用流式解析器+networkx构建图并计算孤立比率、连通分量等指标；

**📊 数据集**

使用Wild SBOMs数据集（78,612文件）以及11个Syft生成的容器SBOM；

**📈 对比分析**

在生产漏洞优先级管线中用开放世界语义替代闭世界veto，KEV recall从0.60提升至0.95/0.97，警报量仅提升约12%；

**⚠️ 局限性**

局限于样本规模、仅评估KEV recall，未充分验证跨生态系统泛化，且完整图声明使用率极低。

---

## 261. trasgoDP: An Open Source Framework for Releasing Noised Tabular Microdata under Local Differential Privacy

**arXiv ID:** 2607.22230 | [PDF](https://arxiv.org/pdf/2607.22230v1)

**作者:** Judith Sáinz-Pardo Díaz `[一作]` (Instituto de Física de Cantabria), Álvaro López García `[通讯]` (Instituto de Física de Cantabria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个名为trasgoDP的Python开源框架，用于在本地差分隐私（LDP）和度量隐私（geo‑indistinguishability）下发布带噪声的表格微数据和位置数据；

**💡 创新点**

创新点在于整合多种LDP机制（Laplace、Gaussian、Exponential、随机回应）与geo‑indistinguishability，提供统一API，并引入了新的相关损失（correlation‑loss）评估指标，用于定量衡量隐私-实用性折衷；

**🔧 技术方法**

技术包括本地差分隐私的噪声注入算法、geo‑indistinguishability的两维噪声采样、统计分布差异度量（TVD、JS、KL）以及相关损失计算；

**📊 数据集**

使用了三组公开数据集：UCI Adult（表格微数据）、合成全球癌症患者数据（表格微数据）、纽约市出租车行程数据（位置微数据）；

**📈 对比分析**

通过在不同ε值下对各机制进行多次实验，计算相关损失、分布差异等指标，并与传统的全球DP或匿名化方法对比，结果显示：LDP在较高ε下可保持较低的相关损失，但在低ε时噪声过大；geo‑indistinguishability在不同ε下产生可控的偏移半径，满足隐私与可用性平衡；

**⚠️ 局限性**

局限性包括仅实现了有限的机制（数值、类别、位置），未覆盖文本等类型；多列LDP下的隐私预算累积与实用性退化未做深入分析；相关损失指标仅在两大数据集上验证，缺乏更广泛的实验；未提供自动ε选择方法。

---

## 262. Trajectory-Regularized Stochastic Optimal Control via KL Divergence

**arXiv ID:** 2607.22201 | [PDF](https://arxiv.org/pdf/2607.22201v1)

**作者:** Mintae Kim `[一作]` (Hybrid Robotics), Koushil Sreenath `[通讯]` (Hybrid Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种轨迹正则化的随机最优控制框架 TRSOC，利用轨迹分布的 KL 散度作为正则项，将其转化为漂移不匹配的二次惩罚，并在保持动态规划和 HJB 结构的同时，实现对性能与参考行为的平衡。

**💡 创新点**

创新点包括：①在轨迹层面引入 KL 正则化，避免传统仅在动作层面正则化的限制；②利用 Girsanov 定理将轨迹 KL 解析为局部的二次漂移惩罚，保持马尔可夫性与 DP 可行性；③在 LQ 情况下得到闭式解，展示 λ 参数如何在控制代价和参考偏差之间进行连续插值；④证明了闭环稳定性与不变测度的存在，进一步说明正则化的稳健性；⑤展示了从离线数据学习参考动态并应用 TRSOC 的可行性。

**🔧 技术方法**

使用的技术包括：随机微分方程、Girsanov 定理、动态规划与 HJB 方程、控制对偶性与 Riccati 方程、线性二次调节（LQR）理论、离散化后的后向 DP、神经网络拟合参考加速度、Monte Carlo 评估与实验仿真。

**📊 数据集**

实验使用的是仿真双积分器（四维状态）在给定 figure‑eight 轨迹下的控制问题，参考行为来自已知的比例-微分控制器或从离线轨迹学习得到的神经网络加速度模型；未使用公开真实数据集。

**📈 对比分析**

比较方法主要是：①将 TRSOC 与无正则化 SOC（λ=0）以及参考控制器（λ→∞）进行对比；②绘制任务成本与轨迹偏差（KL 近似）随 λ 变化的曲线；③评估闭环状态方差上界与不变测度特性。实验显示，随着 λ 增大，控制成本略升高但轨迹偏差显著下降；闭环均方误差保持有界，且在 λ→∞ 时可逼近参考控制器。

**⚠️ 局限性**

局限性包括：①需要相同扩散项且扩散矩阵可逆，才能满足 Girsanov 条件；②仅对漂移可控的情形有效，无法直接处理控制影响扩散的系统；③正则化参数 λ 的选取依赖经验，缺乏自动调参机制；④在高度非线性或高维系统中，解析 HJB 或 Riccati 可能不可行，需要数值求解；⑤实验仅在仿真环境中验证，缺少真实世界的数据与鲁棒性评估。

---

## 263. DeFiScreener: Efficient DeFi Attack Pre-screening in Smart Contracts via Historical Case Matching

**arXiv ID:** 2607.22184 | [PDF](https://arxiv.org/pdf/2607.22184v1)

**作者:** Rui Cao `[一作]` (Nanjing University of Aeronautics and Astronautics), Zhenguang Liu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一种基于历史攻击案例匹配的DeFi预筛查框架DeFiScreener，用于快速定位潜在的智能合约攻击点。

**💡 创新点**

创新点在于首次引入功能语义嵌入与函数调用树相结合的双层预筛查，并提出攻击模式导向的MCTS（APO‑MCTS）在海量调用序列中高效定位脆弱路径。

**🔧 技术方法**

技术手段包括大型语言模型（LLM）生成函数语义向量、构建函数调用树、攻击模式库匹配、APO‑MCTS搜索以及LLM解释生成。

**📊 数据集**

使用了包含207起真实攻击事件、约29.7亿美元损失的Dataset 1以及两个公开基准集（Dataset 2、Dataset 3）进行评估。

**📈 对比分析**

与多种专用检测器对比，DeFiScreener实现98.55%召回率、84.30%精确率，零日情况下识别14/17案，且补偿了各检测器共88例误检漏报。

**⚠️ 局限性**

局限性包括仅静态单协议分析、无法处理跨协议调用或接口缺失的函数，以及对完全新型攻击模式缺乏检测能力。

---

## 264. Bowel Obstruction Detection and Localization on Abdominal CT with Deep Learning

**arXiv ID:** 2607.22173 | [PDF](https://arxiv.org/pdf/2607.22173v1)

**作者:** Moritz Vandenhirtz `[一作]` (ETH Zurich), Julia E Vogt `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种多任务深度学习框架，实现肠梗阻检测与转移区定位，并通过P2P方法实现切片内转移点的可解释定位。

**💡 创新点**

创新点在于同时完成患者级别检测和切片级别转移区定位的联合框架，以及引入可解释的P2P扩展，使模型在关注最小图像区域的同时保持高精度。

**🔧 技术方法**

使用ResNet和Vision Transformer（ViT）作为骨干网络，配合多任务损失、Gumbel‑Softmax掩码和Jensen‑Shannon正则化等技术。

**📊 数据集**

在内部医院数据集上评估，包含1427例腹部CT，683例机械性肠梗阻，744例对照。

**📈 对比分析**

与仅做检测的基线模型相比，提出方法在患者级检测AUROC保持相近的同时，转移区定位的Hit@10、NDCG、MRR、MedRank显著提升；P2P扩展在转移点定位上达88% Hit，平均仅使用5%图像面积。

**⚠️ 局限性**

局限包括单中心、回顾性数据、缺乏多中心外部验证、对低质量或不同扫描协议的鲁棒性未知，以及对非机械性肠梗阻（如瘫痪性肠麻痹）的适用性有限。

---

## 265. Resource Allocation and Conversion along the Org Chart

**arXiv ID:** 2607.22159 | [PDF](https://arxiv.org/pdf/2607.22159v1)

**作者:** Yuan Deng `[一作]` (Google Research), Mihai Tiuca `[通讯]` (Google)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种考虑组织层级与资源可转换性的内部资源分配市场模型，并给出了满足效率与公平性的解概念。

**💡 创新点**

创新点在于将层级结构与资源转换图统一到市场设计中，证明了在该模型下总能得到唯一有效分配；引入的 Budget Descent 算法通过虚拟预算迭代实现对一般需求的可解性与有效性证明。

**🔧 技术方法**

核心技术包括：构造三条经济公理（流量守恒、零利润中介、无套利），针对谐波需求通过一系列独立凸优化求解价格；对一般需求使用虚拟预算映射到谐波问题并证明价格与预算的单调性与连续性；利用 Topkis 定理与 Berge 最大值定理保证迭代收敛。

**📊 数据集**

使用 Google 内部 AI 加速器（TPU、GPU）分配的真实数据集进行实验验证。

**📈 对比分析**

通过 Budget Descent 迭代，算法在 O(1/ϵ log (1/ϵ)) 步内达到误差 ϵ 的有效分配；实验表明价格收敛速度快、资源利用率趋近于 100%，并在多个实例中验证了算法的可行性与效率。

**⚠️ 局限性**

局限性包括：需求假设为可分离且满足单调/单调收益条件；仅适用于人工货币且无利润中介的组织；对循环资源转换图需要先合并 SCC；对高维资源与层级深度极大时的计算成本仍待进一步优化。

---

## 266. Learning on the Job: Continual Learning from Deployment Feedback for Frozen-Weights Agents

**arXiv ID:** 2607.22157 | [PDF](https://arxiv.org/pdf/2607.22157v1)

**作者:** Valentin Tablan `[一作]` (Memory Company), Kristoffer Bernhem `[通讯]` (Memory Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结权重的AI代理上，结合外部记忆系统，利用每次交互产生的结果判定或纠正反馈，实现连续学习。

**💡 创新点**

创新点包括：①证明仅凭单一反馈（结果或纠正）即可驱动连续学习；②将自然语言规则存入可检索的外部记忆；③展示冻结存储在不同模型间可迁移；④公开实验套件与评测协议。

**🔧 技术方法**

使用Spark外部记忆服务、RAG检索、Mistral Large与Claude Sonnet 5大模型、τ‑banking基准以及cluster bootstrap估计区间等技术。

**📊 数据集**

实验基于τ‑banking银行业务基准（97个多轮任务，每个4次试验）以及约700条政策文档。

**📈 对比分析**

对比无记忆基线、仅体验学习和指令学习，使用pass^k、hold率和每试成功率等指标。结果显示指令学习在Mistral上单次成功率提升2.6倍，Sonnet提升1.6倍；经验学习略高；跨模型读取对双方均有提升，转换率约31%。

**⚠️ 局限性**

局限性：仅在单一银行业务域和97任务上验证；每个条件只跑一次，未评估运行间变异；未考虑噪声反馈；跨任务迁移有限；经验模式仅在Mistral上测试；模型与任务匹配程度可能影响推广。

---

## 267. Industrial Tokenization for LLM-Based Health Intelligence: A Federated Architecture for Industrial Evidence Integration

**arXiv ID:** 2607.22153 | [PDF](https://arxiv.org/pdf/2607.22153v1)

**作者:** Deshui Li `[一作]` (Mingyang Smart Energy Co., Ltd.), Zishun Wang `[通讯]` (Mingyang Smart Energy Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出工业令牌化概念，并构建联邦工业架构，将各源特定分析子系统的输出转化为结构化文本令牌，随后由LLM进行解释；

**💡 创新点**

创新点在于设计了面向工业证据的统一语言化接口（Industrial Tokens），实现源异构模型与LLM的解耦与互操作；

**🔧 技术方法**

使用振动信号自动诊断模型、规则驱动事件聚合、结构化文本生成以及ChatGPT进行文本解释；

**📊 数据集**

实验数据来自风电场一台机组的振动监测，覆盖一个月的记录；

**📈 对比分析**

未提供量化对比，仅通过两例示例展示LLM对令牌的理解与维护建议，缺乏性能评估；

**⚠️ 局限性**

局限性包括仅验证单一诊断子系统、令牌生成依赖人工规则、未实现跨源融合、缺乏系统化的实验验证与性能测评。

---

## 268. DB-VIO: Dual-Branch Visual Inertial Odometry with Enhanced Visual-Inertial Representation

**arXiv ID:** 2607.22123 | [PDF](https://arxiv.org/pdf/2607.22123v1)

**作者:** Ziyu Wan `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DB-VIO，一种双分支视觉-惯性里程计框架，利用深度引导融合和姿态引导编码提升视觉与惯性表示，并通过分离旋转与平移的时序建模实现更准确的6-DoF位姿估计。

**💡 创新点**

创新点在于：1）深度引导融合(DGF)将预训练的相对深度信息注入单目视觉特征；2）姿态引导编码(AGE)显式集成陀螺仪积分的姿态先验以强化惯性特征；3）双分支解码器将旋转和平移的时序建模解耦，分别使用独立的LSTM。

**🔧 技术方法**

使用的技术包括：Metric3D深度预测、FlowNet编码、交叉模态注意力融合、旋转指数映射与对数映射、LSTM时序网络、联合位姿损失与累计误差约束。

**📊 数据集**

实验数据集：KITTI Odometry（自动驾驶场景）和EuRoC MAV（无人机飞行场景）。

**📈 对比分析**

与传统几何VIO、深度学习VIO以及最新基于自监督/监督学习的VIO方法进行对比。结果显示：在KITTI上相对平移误差下降20%，旋转误差下降23%；在EuRoC上相对平移误差下降2.3%，旋转误差下降65.2%，ATE下降33.5%。同时实现实时推理（KITTI 12.6 Hz，EuRoC 202 Hz）。

**⚠️ 局限性**

局限性：1）对深度估计依赖预训练模型，导致在缺乏深度信息的低纹理环境中性能可能受限；2）双分支结构和多模态融合增加了模型复杂度与推理时延；3）在高度噪声或极端运动的IMU数据下，姿态引导编码仍可能出现误差积累。

---

## 269. Safe Learning Predictive Control for Ego-World Robotic Systems

**arXiv ID:** 2607.22225 | [PDF](https://arxiv.org/pdf/2607.22225v1)

**作者:** Davide Valenti `[一作]` (University of Bologna), Giuseppe Notarstefano `[通讯]` (University of Bologna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于在线学习的安全预测控制框架（Safe Online World policy Learning MPC），通过Sparse Variational Gaussian Processes（SVGP）与Online Variational Conditioning（OVC）对未知世界机器人的策略进行即时建模，并将多步预测轨迹及其不确定性注入模型预测控制（MPC）中，实现ego机器人在共享环境中的安全操控。

**💡 创新点**

创新点包括：① 在仅能观测世界机器人状态、无法直接获取输入的情况下，利用SVGP在线推断未知策略；② 引入OVC实现对SVGP变分参数的递推更新，避免批量重训练；③ 将多步预测的不确定性通过高阶矩传播融入MPC约束，形成不确定性感知的碰撞规避；④ 在ego–world框架下完成实时安全控制，兼顾预测精度与计算效率。

**🔧 技术方法**

所用技术主要包括：Sparse Variational Gaussian Processes、Online Variational Conditioning、Gaussian moment propagation、非线性模型预测控制（MPC）结合不确定性约束、CasADi+IPOPT求解器、ROS 2与ChoiRbot框架、Webots仿真与Vicon测距。

**📊 数据集**

实验数据集：预训练阶段采集了 800 条世界机器人状态序列（ETHZ Mobil 场景），虚拟实验使用 ETHZ Mobil 与 Lemniscate 两条轨迹进行 Monte Carlo 评估；真实实验在室内 9×4 m 赛道中通过 Vicon 运动捕捉系统获得状态数据。

**📈 对比分析**

与 Offline-GP（不更新参数）和 Constant Velocity（CV）基线比较，在线学习后平均位移误差（ADE）从 0.4 m 降至约 0.05 m；成功率随安全参数 nσ 从 64 % 提升至 100 %；计算时延始终低于 50 ms 采样周期，证明系统具备实时性。

**⚠️ 局限性**

局限性包括：① 依赖于对世界机器人状态的噪声观测假设，若观测误差较大或非白噪声可能影响预测；② 当前仅针对单一未知世界机器人，扩展到多机器人或更复杂交互情境需要进一步研究；③ SVGP 需预先设定诱导点数目，过少会导致近似误差，过多会增加计算负担；④ 对极端动态或不连续策略的快速适应性仍有限。

---

## 270. Latent PDE mapping for efficient physics-informed learning across geometries with limited data

**arXiv ID:** 2607.22215 | [PDF](https://arxiv.org/pdf/2607.22215v1)

**作者:** Ingvild Askim Adde `[一作]` (Kristiania University of Applied Sciences), Gabriel Balaban `[通讯]` (Kristiania University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“潜在PDE映射（Latent PDE Mapping, LPM）”方法，用以在有限训练数据下提升物理信息机器学习模型对不同几何形状的泛化能力

**💡 创新点**

创新点在于将几何相关的PDE残差和边界条件通过变形梯度映射到预设的潜在几何上，从而实现精确的形状梯度计算，克服传统方法忽略边界形状梯度的缺陷

**🔧 技术方法**

采用了物理信息神经网络（PINN）和物理信息深度算子网络（PI‑DON），结合变形梯度、雅克比矩阵与Aliev‑Panfilov心脏电生理模型的非线性时变PDE残差计算

**📊 数据集**

使用openCARP FEM求解器生成的合成心脏电生理数据，覆盖二维和三维四个几何参数化族（扩张、剪切、非线性、旋转），每个族训练样本仅15个形状

**📈 对比分析**

与不使用LPM的基准模型（含形状参数或全局几何描述）在内部和外部测试集上进行对比，LPM在旋转族中实现约4–6倍的相对L2误差下降，且统计检验显示显著优于第二佳模型

**⚠️ 局限性**

主要局限包括需要已知精确的变形梯度（无法直接从实际几何估计）、对仅限Aliev‑Panfilov PDE的验证、对大规模数据集与更复杂几何（如统计形状模型、B‑spline）适用性的未评估

---

## 271. Deep Convolutional Large-Margin $\ell_p$-SVDD for Visual Anomaly Detection

**arXiv ID:** 2607.22212 | [PDF](https://arxiv.org/pdf/2607.22212v1)

**作者:** Alireza Dastmalchi Saei `[一作]` (Bilkent University), Shervin Rahimzadeh Arashloo `[通讯]` (Bilkent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DLM‑SVDD 框架，联合学习卷积特征与大边界的 ℓ_p‑SVDD，用交替优化实现特征与决策边界同步更新。

**💡 创新点**

创新点：① 将 Frank–Wolfe 解决大边界的凸二次子问题与软plus‑margin 违规损失的 CNN 更新交替进行；② 在此框架下系统评估多种核近似（Nyström、RFF、QMC‑RFF、ORF、SORF、Fastfood、RPCholesky），实现可扩展的大边界学习。

**🔧 技术方法**

使用技术包括：ResNet‑50 卷积骨干、RBF 核、ℓ_p‑SVDD 损失、Frank–Wolfe 最优子问题求解、软plus‑hinge 近似、以及上述七种核近似方法。

**📊 数据集**

实验数据集：Fashion‑MNIST、CIFAR‑10、CIFAR‑100、ImageNet‑LT 及其长尾版本 CIFAR‑10‑LT、CIFAR‑100‑LT。

**📈 对比分析**

与固定特征 ℓ_p‑SVDD、传统 SVDD、Deep‑SVDD、PANDA、MSC、UNODE、UNODE‑contrastive、IMD‑AD 等方法进行对比，标准基准 AUROC 均位列首位，长尾任务下平均排名最低，表明在不同不平衡场景下均有显著优势。

**⚠️ 局限性**

局限性：① 对核近似预算敏感，过小会显著降低性能；② 在正样本极少的尾类上联合训练提升有限；③ RBF 带宽需在预训练特征上固定，动态估计会导致不稳定；④ 仅验证一类 vs. 其余类设置，未涉及多类别全连接决策的进一步研究。

---

## 272. Deconstructing Off-Policy Ratios: Entropy-Scaled Trust Regions for Asynchronous Reinforcement Learning

**arXiv ID:** 2607.22186 | [PDF](https://arxiv.org/pdf/2607.22186v1)

**作者:** Guanqun Zhao `[一作]` (Beijing University of Posts and Telecommunications), Zeyu Chen `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于 token 熵自适应的 Trust Region（ESTR），用于稳定大模型的异步强化学习。

**💡 创新点**

创新点在于发现重要性比值的自然尺度随 token 熵变化，并以熵为尺度动态设定容忍阈值，既抑制低熵噪声，又保留高熵探索；同时剖析并利用了轨迹内版本切换的特性。

**🔧 技术方法**

采用熵缩放的掩码策略、二阶 KL 近似、异步生成与训练解耦、token 级重要性比值裁剪等技术。

**📊 数据集**

使用 BrowseComp‑Plus、GSM8K、DAPO‑Math 以及 AIME 2024‑2026 等长序列推理与数学推理数据集。

**📈 对比分析**

与同步 GRPO、原始异步、IcePop、KPop 等方法对比，ESTR 在保持同步精度的同时实现 2.6× 的训练吞吐提升，且训练过程更稳定，未出现崩溃。

**⚠️ 局限性**

局限在于对极端 staleness 的鲁棒性仍未充分验证，且需要依赖 logits 计算熵，可能在模型架构或推理策略变化时需额外调整。

---

## 273. HarnessLLM: Rust Verification Harness Generation with Large Language Models

**arXiv ID:** 2607.22161 | [PDF](https://arxiv.org/pdf/2607.22161v1)

**作者:** Minghua Wang `[一作]` (Ant Group), Lin Huang `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于大语言模型（LLM）的自动化工作流，能够从 Rust 项目的现有测试用例中提取调用场景，并生成符合 Kani BMC 需求的验证 harness，随后通过编译器反馈迭代修正，最终实现对不安全代码的内存安全和运行时崩溃检测。

**💡 创新点**

创新点在于：①将 LLM 与轻量级静态分析相结合，自动抽取并 dedupe 调用场景；②构造依赖图并生成 Chain‑of‑Thought（CoT）提示，指引 LLM 逐步构造复杂非约束参数；③在编译错误反馈阶段明确保留原始代码片段，避免 LLM 误报类型或函数，显著提升修正效率。

**🔧 技术方法**

核心技术包括：Rust MIR 代码分析、函数调用场景提取、类型依赖图构建与 CoT 提示生成、LLM（GPT‑4.1、Claude‑4 等）交互、Kani BMC 编译器反馈循环、AST 检测防止类型重定义。

**📊 数据集**

使用了来自 crates.io 的 9 个真实 Rust 库（共 63 条测试文件、约 1.5 万行代码），以及从中抽取的场景函数数据集 D_scen，用于评估各个子组件的贡献。

**📈 对比分析**

与现有的 Kani Autoharness 进行对比：在 9 个库中，Autoharness 仅成功覆盖约 34% 的场景，而本工作在 10 次迭代内 100% 成功生成 harness；平均生成时间约 0.5 秒/个 harness，成本仅 0.03 美元/个；在 sparse 测试场景下仍保持 97% 以上成功率，说明鲁棒性强。

**⚠️ 局限性**

局限性包括：①依赖原始测试用例的覆盖度，若某 API 缺少测试将导致场景不足；②对宏生成代码的可见性有限，极少数情况可能漏检；③LLM 在推断边界值（如 slice 长度）时仍可能不够精准，需要结合静态分析辅助；④在极大项目中，提示长度和上下文窗口仍可能成为瓶颈。

---

## 274. CARDIAG: A Dense Segment Classification Benchmark of Deep Learning Architectures for Coronary Angiography

**arXiv ID:** 2607.22139 | [PDF](https://arxiv.org/pdf/2607.22139v1)

**作者:** Dominik Bernard Lau `[一作]` (Gdansk University Of Technology), Natalia Zielińska `[通讯]` (Medical University Of Gdansk)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布 CARDIAG 数据集并对 24 种深度学习模型在冠脉影像 SYNTAX 分割任务上进行大规模基准测试。

**💡 创新点**

提供全新多中心、多标签、高质量注释的 CARDIAG 数据集；系统评估不同模型族及其组合；结合多种评估指标包括中心线连通性和直径误差。

**🔧 技术方法**

采用 ConvNeXt V2+DeepLabV3+、Mamba UNet 等先进 CNN、SSM 以及 Transformer 结构；使用 EMA、dropout 进行不确定性估计；构建投票集成。

**📊 数据集**

CARDIAG 数据集（644 影像，包含 SYNTAX 标签、二值、置信度、分割掩模、补充帧及 DICOM 元数据），与公开的 ARCADE 等进行对比。

**📈 对比分析**

通过 F1、clDice、DE、HD95 等指标在留一中心交叉验证和数据效率实验中比较；ConvNeXt V2+DeepLabV3+ 单模型 F1≈0.456，集成后提升至≈0.479；Mamba UNet 在直径误差上最优。

**⚠️ 局限性**

仅评估单帧影像，未利用视频连续性；未考虑不同中心样本分布偏差；未完成 SYNTAX 评分和分割交叉误差分析；缺乏针对性数据增强和偏差校正。

---

## 275. Binary Cyclic Codes With Simultaneously Large Minimum Distances and Dual Distances

**arXiv ID:** 2607.22137 | [PDF](https://arxiv.org/pdf/2607.22137v1)

**作者:** Ziling Heng `[一作]` (Chang'an University), Jiantao Hu `[通讯]` (Chang'an University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种新的构造方法，用于生成具有同时较大最小距离和对偶距离的二进制循环码。

**💡 创新点**

创新点在于构造了多个具有良好参数的二进制循环码，这些码的最小距离和对偶距离的下界接近于n或2n。

**🔧 技术方法**

使用了代数编码理论中的循环码构造方法，特别是利用了2-循环余数类和BCH界限。

**📊 数据集**

研究中使用的主要数据集是已知的二进制循环码的参数表，特别是与现有的二进制循环码进行比较。

**📈 对比分析**

与已知的相同长度和维度的二进制循环码相比，本文构造的循环码在最小距离和对偶距离上表现出更大的下界，特别是一些构造的下界接近于2n。

**⚠️ 局限性**

限制在于目前的构造方法主要针对偶数s的情况，可能无法推广到所有s的情况。

---

## 276. Dynamic Commonsense Coordination for Empathetic Response Generation

**arXiv ID:** 2607.22136 | [PDF](https://arxiv.org/pdf/2607.22136v1)

**作者:** Zhengyu Qi `[一作]` `[通讯]` (Leiden University), Zhengyu Qi (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了动态共识知识协调框架(DCC)，在同理心回应生成任务中实现共识知识在理解和生成阶段的动态协同；

**💡 创新点**

创新点包括：1）跨来源共识交互模块(SCE‑AttnRes)实现情境与对话共识的动态融合；2）关联引导共识过滤模块(AGCF)对每个共识关系进行相关性加权过滤；3）迭代共识感知解码模块(ICAD)在每一步生成中动态检索并注入最相关的共识；

**🔧 技术方法**

使用的技术包括COMET生成共识知识、跨注意力机制、残差加权融合、层归一化+线性评分、跨注意力检索、Frequency‑Aware Cross‑Entropy (FACE) 多样性损失以及CEM的情绪感知编码器；

**📊 数据集**

数据集为公开的Empathetic‑Dialogues（含情绪标签、情境描述和多轮对话），并使用COMET自动生成的五类共识推理；

**📈 对比分析**

与MoEL、MIME、EmpDG、KEMP、CEM、SEEK、CASE、ESCM、E‑CORE、SDAM、IAMM等基线进行比较；DCC在情绪分类准确率从39.11%提升至46.09%，响应多样性（Distinct‑1/2）显著提升（分别从0.66→1.03和2.99→4.93），困惑度与CEM相近（36.13≈36.11）；

**⚠️ 局限性**

局限性：仅在英文Empathetic‑Dialogues上评测，无法验证跨语言/领域泛化；依赖COMET的固定五类共识关系，易受噪声与偏见影响；对xReact的抑制表明情感-认知共识交互尚需深入研究；与IAMM/SDAM等最新关联模型相比在自动指标上仍未超越；单次实验缺乏统计显著性检验；

---

## 277. GLI-AL: A Multi-Modal Glioma MRI Label Resource with Unified Anatomy-Lesion Labels

**arXiv ID:** 2607.22135 | [PDF](https://arxiv.org/pdf/2607.22135v1)

**作者:** Xingyu Xiang `[一作]` (Xi'an Jiaotong University), Chunfeng Lian `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建并发布了一个基于BraTS 2023‑GLI的统一解剖‑病变标签资源，包含WMH感知、健康组织与肿瘤的八类标签，并提供样本分层与可追溯元数据。

**💡 创新点**

创新点在于将WMH共病病变与健康组织标签融合为单一监督空间，提供纯净与扩展子集、可追溯元数据，并解决未标注WMH导致的标签噪声问题。

**🔧 技术方法**

使用MedNeXt、DeepWMH、LST‑AI、TumorSynth等自动分割工具，以及概率图融合与熵加权融合技术，配合IQR异常检测与人工审核。

**📊 数据集**

采用BraTS 2023‑GLI训练集1251例四模MRI，并在MICCAI 2017 WMH外部数据集进行零样本验证。

**📈 对比分析**

通过对比基线（仅纯净子集）与联合基线（混合纯净+扩展子集）模型在GLI内域和WMH外域的Dice与HD95指标，发现纯净子集能在保持健康组织分割性能的同时提升对WMH的识别，证明数据质量的重要性。

**⚠️ 局限性**

局限在于纯净子集并非全新WMH标注，仅包含专家负样本与模型筛选；扩展子集依赖自动工具可能引入误检；健康组织标签来源于自动工具，缺乏人工验证；未覆盖其他共病病变，如微出血、腔隙性缺血。

---

## 278. Not Every Dependency Is Worth Discovering: Toward Value-Driven Data Dependency Discovery

**arXiv ID:** 2607.22219 | [PDF](https://arxiv.org/pdf/2607.22219v1)

**作者:** Xiaolong Wan `[一作]` (Harbin Institute of Technology), Xixian Han `[通讯]` (Harbin Institute of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

提出了基于决策理论的价值驱动数据依赖发现框架，定义了使用价值与净价值，并给出四项原则和五个研究方向。

**💡 创新点**

创新点在于将依赖发现从单纯的“有效性”转向“价值”，引入任务导向的损失函数和生命周期成本评估，实现依赖发现的价值优化。

**🔧 技术方法**

采用决策理论（价值信息理论）、成本敏感学习与子模优化等方法论，构建价值估计与搜索、验证、选择、维护的整体流程。

**📊 数据集**

文章未使用具体数据集，主要是理论框架与概念性说明。

**📈 对比分析**

未给出实验比较，文中仅提出未来需要的基准与评价指标，说明目前缺乏实证性能评估。

**⚠️ 局限性**

局限性包括：缺乏实证验证、成本估计与价值估计方法仍不成熟、需要可计算的价值代理、以及在真实系统中实现价值驱动搜索与维护的技术挑战。

---

## 279. Duet: Co-Optimizing P2P Message Propagation and Rotating-Leader Consensus

**arXiv ID:** 2607.22209 | [PDF](https://arxiv.org/pdf/2607.22209v1)

**作者:** Yifeng Ye `[一作]` (Shanghai Jiao Tong University), Shengyun Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在区块链中将网络拓扑和延迟信息记录在链上，并基于此共同优化旋转领导者共识和P2P消息传播。

**💡 创新点**

创新点包括：①利用链上网络信息加速领导者轮换；②设计基于树的高效广播与基于投票的可靠确认，必要时回退到gossip；③构造延迟感知的传播树。

**🔧 技术方法**

采用的技术包括：在Tendermint中加入基于树的提议传播与最佳努力广播；使用libp2p的GossipSub作为底层网络；利用Dijkstra+贪心算法构造主节点环和树；以及对定时器进行精细调整。

**📊 数据集**

实验数据集使用AWS EC2实例，10个区域共300节点，节点间的RTT从实际测量获取，用来构建拓扑。

**📈 对比分析**

与纯gossip和K-ary树基线比较，在同一拓扑下，该方案在10~300节点时吞吐量提升高达7.26倍，峰值约22k tps，且在地理分布均匀和倾斜部署下均表现优于基线。

**⚠️ 局限性**

局限性包括：需要链上维护完整网络信息会增加攻击面；树传播在节点失效时需要回退，定时器调优敏感；在极高节点失效率（>30%）下，树覆盖不足导致吞吐下降；以及对跨地域部署的可扩展性仍受限。

---

## 280. From Score Approximation to Distribution Approximation in Score-Based Diffusion Models

**arXiv ID:** 2607.22199 | [PDF](https://arxiv.org/pdf/2607.22199v1)

**作者:** Lan V. Truong `[一作]` `[通讯]` (Ho Chi Minh City University Of Technology), Lan V. Truong (Ho Chi Minh City University Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

证明了在给定的前向扩散和逆扩散设定下，若神经网络能在 L²(μ) 范数下逼近真分数函数，则逆扩散生成的分布与目标数据分布在 KL 散度上可控。

**💡 创新点**

首次给出了分数逼近误差到生成分布误差的显式量化上界，弥补了传统 universal approximation 定理与分布近似之间的空白。

**🔧 技术方法**

结合 Hornik 的 universal approximation 定理、Girsanov 定理和相对熵的数据处理不等式，构建了从函数逼近到路径测度再到分布误差的完整理论链。

**📊 数据集**

未涉及具体数据集，论文主要为理论分析。

**📈 对比分析**

未给出实验比较；论文只在理论上证明了误差上界，未评估具体性能。

**⚠️ 局限性**

局限性包括：仅给出 KL 散度上界；未给出逼近速率；不适用于离散时间模型或其他距离度量；仅适用于欧氏空间。

---

## 281. Draining the Energy Commons: Self-Defeating Over-Appropriation as a Coordination Failure in Agentic LLM Collectives

**arXiv ID:** 2607.22188 | [PDF](https://arxiv.org/pdf/2607.22188v1)

**作者:** Marcantonio Bracale Syrnicov `[一作]` (Icaro Lab), Daniele Nardi `[通讯]` (Sapienza University of Rome)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在共享可再生能源储备上，LLM代理自我持续性目标导致在需求超过峰值补给时出现过度获取并随时间自我削弱的协调失败；

**💡 创新点**

首次将系统层面对齐失败定义为公共轨迹的不可避免衰减，并通过轨迹级评价和基准对比揭示了“短视优化”在多代理环境中的自毁效应；

**🔧 技术方法**

使用三类LLM（GPT‑5.4‑mini、Gemini‑3.1‑flash‑lite、Grok‑4.3）在自我对弈的能源共享模拟中，结合低/高推理努力、容量健康模型与物流补给律；

**📊 数据集**

采用自定义的模拟数据集：四名同族代理、24轮交互、四种补给速率（ρ=0.8、1.0、1.1、1.2）以及高需求快补充条件，所有参数在论文表格中给出；

**📈 对比分析**

通过与离线社会规划和开放准入基准轨迹对比，发现实验轨迹在稀缺条件下出现显著的储备缺口（p<10⁻⁵），而基准显示可持续，证明代理表现出对未来效益的低加权；

**⚠️ 局限性**

局限在于仅考虑同族四代理自对弈、缺乏通信/治理、短期24轮有限时间、抽象的物流补给模型、以及基准仅使用固定比例策略，难以直接推广至真实复杂的能源或计算共享场景；

---

## 282. JustDepth: Real-Time Radar-Camera Depth Estimation with Single-Scan LiDAR Supervision

**arXiv ID:** 2607.22172 | [PDF](https://arxiv.org/pdf/2607.22172v1)

**作者:** Wooyung Yun `[一作]` (Ajou University), Soomok Lee `[通讯]` (Kennesaw State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种单阶段雷达-摄像头深度估计模型 JustDepth，只使用单帧 LiDAR 作为监督，完成实时且无额外标注的深度预测。

**💡 创新点**

创新点包括：① 固定宽度 1D 雷达编码实现与点数无关的低延迟；② 高度融合块与轻量图神经网络实现全局深度传播；③ 训练专用置信度解码器在不增加推理成本的前提下提升学习稳定性；④ 通过点上采样和旋转对齐增强的 LDL 抑制，显著降低条纹伪影。

**🔧 技术方法**

核心技术有：雷达投影为 1D 张量、ResNet 风格图像编码器、Height Fusion Block（高度自注意力融合）、基于 VIG 的图神经网络 (MRConv)、U‑Net 风格深度解码器、置信度解码器、边缘感知平滑损失、VHGR 指标评估、旋转与上采样数据增强。

**📊 数据集**

使用 nuScenes 数据集（900×1600 的 RGB、mmWave 雷达点云和单帧 LiDAR 深度）进行训练与评估。

**📈 对比分析**

与现有多阶段或需辅助标签的雷达‑摄像头深度方法相比，JustDepth 在 nuScenes 上实现 MAE 0‑70 m 仅高 8.54% 的误差，且推理时间仅为 14.8 ms，约比最快的 GET‑UP 低 39.7 倍；同时 VHGR 下降 66%，显著减少条纹伪影。

**⚠️ 局限性**

主要局限在于单帧 LiDAR 的稀疏与噪声仍限制了精细深度细节的恢复；对极端低光或雨雪等极端天气下的雷达噪声鲁棒性尚待进一步验证。

---

## 283. Integrating Energy Efficiency into Software Development: Developer Perspectives and Requirements

**arXiv ID:** 2607.22168 | [PDF](https://arxiv.org/pdf/2607.22168v1)

**作者:** Anika Hennig `[一作]` (FH Aachen University of Appl. Sci.), Mathias Eggert `[通讯]` (FH Aachen University of Appl. Sci.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对10名专业软件开发者的半结构化访谈，定性分析其对能效的认知以及对AI辅助能效优化工具的需求与障碍，构建基于开发者视角的需求框架。

**💡 创新点**

首次系统阐明开发者对能效的隐性关注、对AI工具的期望与担忧，并将其与技术接受模型相结合，填补了技术方案与实际工作流程对接的研究空白。

**🔧 技术方法**

采用Mayring方法的结构化定性内容分析、MAXQDA编码、人工与AI辅助转写校正等技术手段进行数据处理；研究并未使用任何机器学习模型或自动化评测工具。

**📊 数据集**

使用的是10份匿名访谈记录和基本的参与者背景信息，没有使用公开数据集或外部软件项目代码。

**📈 对比分析**

本研究不涉及算法性能对比或实验验证；评价标准为访谈主题的覆盖度、编码一致性（74.85%）以及理论解释的合理性，未给出可量化的性能指标。

**⚠️ 局限性**

样本量有限、性别不平衡、只涵盖开发者视角、以德国为主要语境、采样依赖作者网络，导致结果的推广性和代表性受限；研究仅为定性分析，缺乏客观性能评估。

---

## 284. Learning Spatiotemporal Decision Priors for Efficient Path Planning under Partial Observability

**arXiv ID:** 2607.22166 | [PDF](https://arxiv.org/pdf/2607.22166v1)

**作者:** Yi Liu `[一作]` (Fudan University), Chun Ouyang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ImiPath 框架，在局部观测下通过学习专家演示中的时空决策先验，指导不同规划器实现更高效路径规划。

**💡 创新点**

创新点包括：1) 使用 SpatioTemporal‑Attention Policy Network 学习可迁移的时空决策先验；2) 将先验以局部引导方式融合进多种确定性与随机性规划器，兼容性强；3) 在磁性微型机器人平台上验证其实用性。

**🔧 技术方法**

技术手段包括：模仿学习与迁移学习；跨时空注意力编码的 STAPNet；局部 FoV 观测构造与极坐标编码；先验融合策略（A* 中加入 log‑prior，ACO 中动态调整转移概率）。

**📊 数据集**

使用的数据集：从 PFACO、A*_Global 等专家规划器生成的大规模演示轨迹，覆盖 15×15 至 30×30 的多尺度地图；测试集为保留的不同规模地图；此外还使用磁性微型机器人实验场景的数据。

**📈 对比分析**

与多种基线（A*, FS+PPM, WA*+CF, AS, EAS, MMAS, IHMACO, PFACO 等）进行对比；ImiPath 在所有规模下保持 100% 成功率，平均路径长度与基线相当或更优，搜索节点数显著减少，计算时间比传统 A*_Local 快 1–2 个数量级。

**⚠️ 局限性**

局限性：仅在二维离散网格与局部 FoV 下验证；依赖专家演示数据，适应性受限于训练数据分布；未在极端动态环境或多机器人协作场景中进行评估。

---

## 285. dRAE: Representation Autoencoder with Hyper-Spherical Codes

**arXiv ID:** 2607.22148 | [PDF](https://arxiv.org/pdf/2607.22148v1)

**作者:** Tianren Ma `[一作]` (University of Chinese Academy of Sciences), Qixiang Ye `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种可直接量化高维视觉表征的离散表示自编码器（dRAE），通过超球面量化（HSQ）实现了对视觉特征的高效离散化。

**💡 创新点**

创新点在于将量化过程中的角度路由与欧氏距离的承诺损失分离，利用角度相似度进行码本分配，同时保持幅度信息，从而消除了代码书崩溃并显著提升码本利用率。

**🔧 技术方法**

使用了超球面量化（HSQ）、可训练的线性投影、向量量化、直通估计器、语义对齐的自蒸馏、以及对齐增强（iREPA）等技术，并在ViT架构上实现了端到端训练。

**📊 数据集**

在ImageNet‑1K上进行图像重建与多模态理解评测，并在GQA、TextVQA、MMBench、MME‑Perception、SEEDBench、T2I（LAION‑400M）和C2I（ImageNet‑1K）等数据集上进行泛化评估。

**📈 对比分析**

与传统VQRAE、VILA‑U、TokLIP等方法比较，dRAE在码本利用率、PSNR、rFID、生成FID/IS以及多模态理解指标上均有显著提升，尤其在131072码本规模下实现0.42 rFID、24.5 PSNR、0.72 SSIM，且在T2I/ C2I 生成任务中分别比VQ提高约20% IS 和 FID。

**⚠️ 局限性**

局限性包括：仍需大规模预训练视觉编码器与巨量代码表；对极高维度或跨模态应用的可扩展性尚待验证；以及在某些任务中对角度路由的硬限制可能导致细粒度结构信息的微小丢失。

---

## 286. Approximate Total Weighted Completion Time with Convex Controllable Processing Times

**arXiv ID:** 2607.22133 | [PDF](https://arxiv.org/pdf/2607.22133v1)

**作者:** Klaus Heeger `[一作]` (Fraunhofer IOSB-INA), Dvir Shabtay `[通讯]` (Ben Gurion University Negev)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在单机可控处理时间的调度问题中，提出了第一个常数因子近似算法（近似比≤e≈2.718）以及一个基于工作负载/权重分类的近似方案，后者在输入规模多项式的情况下可以得到任意精度的近似解。

**💡 创新点**

创新点在于：①将连续资源分配与作业排序分离，利用拉格朗日松弛得到闭式最优资源分配；②通过构造三个分段线性函数（f,g,h）进行积分分析，给出排序规则的逼近上界；③将问题转化为仅包含有限种工作负载或权重的动态规划，从而实现快速近似方案；④同时给出了若干常用排序启发式在一般实例上的下界，揭示其局限性。

**🔧 技术方法**

主要技术包括拉格朗日松弛、分段线性函数的积分界估计、Lipschitz连续性分析、动态规划（对工作负载/权重类型的状态枚举）、规模化和舍入技术。

**📊 数据集**

无实验数据集；所有结果均为理论分析与证明。

**📈 对比分析**

与之前仅能得到指数时间或仅对特殊实例有效的算法相比，该常数因子近似算法在多项式时间内即可得到可接受的解；近似方案在输入规模多项式的情况下可在量级为 O((logθ_max/δ)^θ# n^θ#) 的时间内给出任意精度的解。负向结果表明，一些直觉上的排序规则在最坏情况下会产生线性或多项式级的误差。

**⚠️ 局限性**

主要局限：1）问题的NP难度或多项式可解性仍未确定；2）常数因子近似比可能不是最优；3）目前仅有准多项式时间近似方案，尚无PTAS或FPTAS；4）对于大规模实例，动态规划的指数级状态仍可能不可行。

---

## 287. A Framework for Individual Tree Growth Reconstruction Using Multi-Platform Laser Scanning

**arXiv ID:** 2607.22129 | [PDF](https://arxiv.org/pdf/2607.22129v1)

**作者:** Daniella Tavi `[一作]` (Finnish Geospatial Research Institute FGI), Juha Hyyppä `[通讯]` (National Land Survey of Finland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究构建了一套多时相激光扫描框架，用深度学习分割、分割迁移和高度驱动的尺度模型，结合ALS和MLS/TLS数据实现了北方森林树干直径（DBH）与树干体积的成长估计；

**💡 创新点**

创新点在于将单一时点的MLS/TLS树干曲线与多时相ALS高度结合，利用高度变化对树干比例进行时间尺度化，避免了需要多时点植被下扫描的依赖，并验证了分割迁移在不同传感器间的可行性；

**🔧 技术方法**

技术包括ForestFormer3D深度学习分割、kd‑tree分割迁移、基于密度聚类的树干弧段提取、立体曲线拟合与体积积分，以及基于高度增长因子的尺度与形状因子模型；

**📊 数据集**

使用了136个点云数据，涵盖2014–2025年11台ALS、MLS、TLS传感器，覆盖8个北方林区测试区，共计8个32 m×32 m样地；

**📈 对比分析**

通过与人工测量、单独属性差分法和直接测量法对比，模型估算的5年与10年增长误差均低于差分法，RMSE与MAE均在5–25 %范围内，且R²最高达0.86；

**⚠️ 局限性**

局限包括对分割迁移的几何误差依赖精确配准、在密集林区对树干曲线的误检与多段树干导致的匹配错误，以及模型对高度估计误差的敏感性。

---

## 288. One Hand Watches The Other: Dynamic Multi-Agent Cooperation for Sample-Efficient Bimanual Manipulation in Dynamic Environments

**arXiv ID:** 2607.22119 | [PDF](https://arxiv.org/pdf/2607.22119v1)

**作者:** Jan Ole von Hartz `[一作]` (University of Freiburg), Joschka Boedecker `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了轻量级、与策略无关的 DynaMAC 框架，利用动态多流学习在动态环境和双臂协作中实现零样本迁移与自适应重协调。

**💡 创新点**

创新点包括：① 将相互耦合的机器人与对象/另一臂视为动态任务参数并动态过滤，恢复多流学习的因果假设；② 通过虚拟末端效应框架补偿动态帧丢失；③ 在不需要显式领导-跟随关系的前提下，实现双臂协作；④ 构建 DynaBench 基准，用 RLBench 生成多样化的动态任务。

**🔧 技术方法**

技术手段包括多流策略（MiDiGaP/流式 Gaussian），TAPAS 任务参数选择与分段，基于精度的关节链接检测，虚拟帧补偿，动态帧筛选，产品专家融合；以及在真实环境中使用 DINO+RealSense 的视觉编码。

**📊 数据集**

使用 RLBench（含 RLBench2）数据集，随机采样动态配置并插值；在真实 Franka Emika 双臂机器人上收集 5 条演示。

**📈 对比分析**

与 Diffusion Policy、ARP、MiDiGaP 等基线相比，DynaMAC 在动态单臂任务中提升约 35% 的成功率，使用 20 倍更少的演示；在双臂任务中比 MiDiGaP 领先 38%（相同样本）且在动态、协作、扰动条件下保持高成功率；在真实场景中亦实现零样本动态迁移和对人类协作的鲁棒性。

**⚠️ 局限性**

局限性包括：① 需要能够从长轨迹中分段并获得精确的任务参数；② 依赖外部视觉编码器，感知误差可能导致失败；③ 对于不适配多流假设或缺乏足够空间信息的任务可能表现不佳。

---

## 289. Active few-shot segmentation by reinforcing data selection

**arXiv ID:** 2607.22371 | [PDF](https://arxiv.org/pdf/2607.22371v1)

**作者:** Chenlan Zhao `[一作]` (Queen Mary University Of London), Shaheer U. Saeed `[通讯]` (Queen Mary University Of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了一种强化学习框架，用于从未标注图像池中选择最优支持集，以提高少样本医学图像分割的适应性能。

**💡 创新点**

创新点在于将支持集选择建模为强化学习问题，直接以下游分割性能为奖励，学习集合级而非单样本级的互补性选择策略。

**🔧 技术方法**

采用Meta‑Learning（Reptile和MAML）进行模型预训练和快速适配，使用PPO进行策略优化，并利用Dice指标作为奖励。

**📊 数据集**

使用跨机构的3D男性盆腔MRI数据集，包含8个器官的标注，评估神经血管束和闭塞筋的少样本分割。

**📈 对比分析**

与随机选择、任务优先级（TBP）和基于RL的数据价值（DVRL）等方法比较，结果显示在所有支持集大小下均显著提升Dice得分，最优提升约8个百分点。

**⚠️ 局限性**

主要局限在于需要较大的候选池和多轮训练，计算成本高；并且实验仅针对单一器官组合，尚未验证跨任务通用性。

---

## 290. A Monolithic Hand with Asymmetric Origami Bending and Dual-chamber Actuators

**arXiv ID:** 2607.22320 | [PDF](https://arxiv.org/pdf/2607.22320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 291. Interior interpretability with attention rollout: contraction and propagation profiles in Transformers

**arXiv ID:** 2607.22367 | [PDF](https://arxiv.org/pdf/2607.22367v1)

**作者:** Umberto Biccari `[一作]` (University of Deusto), Enrique Zuazua `[通讯]` (University of Deusto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了“内部可解释性”框架，使用注意力rollout来分析Transformer内部的传播操作；

**💡 创新点**

创新点在于将Doeblin–Dobrushin收敛理论应用于row‑stochastic的rollout矩阵，证明其收敛为rank‑one并给出归一化列和作为传播特征；

**🔧 技术方法**

技术方法包括attention rollout、Doeblin–Dobrushin收敛分析、PCA、GradientExplainer（期望梯度近似SHAP）以及实验对比；

**📊 数据集**

实验数据集包括真实的代谢组学年龄预测数据（≈12744样本、72特征）和ChatGPT生成的synthetic数据（24000样本、18特征）；

**📈 对比分析**

对比方法：rollout传播得分、PCA重要性得分、GradientExplainer特征归因得分。结果显示rollout在深度增加时收敛更好，训练模型与随机初始化的传播分布显著不同；但rollout与GradientExplainer在top‑6特征上有一定重叠，整体排名相关性接近0；模型预测性能在不同深度基本相同；

**⚠️ 局限性**

局限性包括：rollout仅捕获注意力传播，忽略值投影、MLP、归一化等；未证明传播特征与预测效果因果关联；实验仅在单一真实数据集和有限随机种子上进行；synthetic数据缺乏真实因果背景；因此结论为描述性诊断而非通用可解释性指针。

---

## 292. AI4PLE: A Methodology for Integrating AI into Product Line Engineering

**arXiv ID:** 2607.22260 | [PDF](https://arxiv.org/pdf/2607.22260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 293. LinkML-Scala: a Robust, Fast, and Portable Implementation of LinkML

**arXiv ID:** 2607.22335 | [PDF](https://arxiv.org/pdf/2607.22335v1)

**作者:** Piotr Sowinski `[一作]`, Andriy Plokhotnyuk `[通讯]` (NeverBlink)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 LinkML-Scala，一个用 Scala 3 编写的高性能、跨平台实现，涵盖 LinkML 的元模型、运行时支持、模式推导以及 JSON Schema、SHACL、RDFS、Table Schema 等多种生成器，并提供 CLI、浏览器 playground、GitHub Action 与 JVM/JS 库。

**💡 创新点**

实现了对 LinkML 的完整、可插拔的实现，显著提升性能（比 Python 实现快 22.9–38.5 倍）、可移植性（可在 JVM、JavaScript、GraalVM 本机二进制上运行），以及统一、可定位的错误处理；采用 Scala 宏、类型类、枚举等现代语言特性，构建了可维护的生成器框架，并通过模型目录进行统一测试。

**🔧 技术方法**

使用 Scala 3、Scala.js、GraalVM Native Image、Scala 宏、隐式类型类、枚举、JMH、hyperfine 等技术；生成器采用 AST 生成与字符串拼接两种模式；CLI 通过 GraalVM 本机编译实现极快启动；在 JavaScript 环境中通过 Scala.js 转译实现浏览器 playground；通过 npm 包发布 JavaScript/TypeScript 接口；在 Maven Central 发布 JVM 库。

**📊 数据集**

利用 11 个来自 LinkML Schema Registry 与公开 GitHub 仓库的多领域 Schema：bridge2ai、cdm、chem-dcat、crdch、d3fend、fluxnova、include、iso27001、nmdc、sssom、tc57cim，涵盖网络安全、金融、生物、能源等领域。

**📈 对比分析**

采用冷启动（CLI 端到端）和热启动（循环运行）两种场景进行基准测试，使用 hyperfine（冷）和 JMH（热）测量生成器吞吐量。LinkML-Scala 在所有测试中均优于 Python 实现，冷启动 JSON Schema 速度提升 36.5×、SHACL 22.9×；热启动 JSON Schema 38.5×、SHACL 26.8×；总体上实现了显著的性能提升，但提升幅度在不同数据集间存在差异。

**⚠️ 局限性**

性能提升不均匀：某些大型、文本量大的 Schema 产生显著加速，其他则受限于实现路径或功能覆盖差异；冷启动时，GraalVM 本机二进制启动时间在小型 Schema 上占比高；目前仅支持 JSON Schema、SHACL、RDFS、Table Schema 等，尚未覆盖 SQL DDL、Avro、Parquet、Protobuf；需要进一步与 Jelly 集成以提升大型 SHACL/RDFS 处理性能。

---

## 294. SLIP: Segmentation with Low-latency Interactive Prompting for 3D Medical Images

**arXiv ID:** 2607.22332 | [PDF](https://arxiv.org/pdf/2607.22332v1)

**作者:** Baptiste Podvin `[一作]` (IRCAD France), Toby Collins `[通讯]` (IRCAD France)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并实现了SLIP——一种面向3D医学图像的低延迟交互式分割框架，支持点式提示、可逆提示，并通过轻量级的patch记忆银行实现高效的交互式推理。

**💡 创新点**

创新点包括：① 将图像编码与提示引导的细化解耦，预先计算并缓存patch特征；② 引入patch feature contextualization和记忆银行，实现跨patch的上下文传播；③ 支持可逆提示（prompt undo），无需重算图像特征即可撤销提示；④ 兼容多种图像编码器，提升模块化与可扩展性；⑤ 通过极低的交互延迟实现快速迭代与更高的实际使用满意度。

**🔧 技术方法**

采用3D残差CNN+FPN作为图像编码器，利用Transformer‑style的mask decoder结合点提示和上一次预测；使用Patch Attention网络对patch特征进行上下文化；构建固定容量的Patch Memory Bank并使用轻量级CNN对其进行编码；实现点提示的三维位置+视觉特征token；通过交叉注意力和自注意力实现跨patch信息融合；所有组件在PyTorch 2.2中实现，并集成到3D Slicer插件中。

**📊 数据集**

训练集覆盖多模态（CT、MRI、PET、US、显微镜）并包含大量公开数据（13个公开测试集）以及内部超声、腹腔镜等数据；在模拟用户实验中使用13个公共数据集；在真实用户实验中使用MRI子宫、CT肝脏病灶、US肝脏病灶三种任务，共15个体例/视频。

**📈 对比分析**

与nnInteractive、SegVol、SAM‑Med3D在同一台机器上进行比较；模拟用户实验使用50次交互、DSC@50、DC‑AUC@50、交互延迟等指标；SLIP在所有数据集上均获得最高或接近最高的DSC@50，且交互延迟最低（平均0.06 s/次，比nnInteractive低约5×）；累计延迟也显著更低；真实用户实验显示SLIP与nnInteractive在标注时间、DSC一致性上相当，SLIP在可逆提示和用户满意度上更优。

**⚠️ 局限性**

局限性：① 点式提示在早期可能不够明确，导致优化困难；② 需要预先缓存大量patch特征，虽然相对轻量，但在极大体积（>512³）时仍可能超出GPU/CPU内存；③ 当前评估基于已公开或可访问的训练管线，部分最强基线（如nnInteractive）训练细节不公开，难以做更细粒度的模型对比；④ 仅验证了点提示，未涵盖线条、笔划等交互方式；⑤ 在极端大体积或多模态混合场景中，仍需进一步优化内存管理与异步处理。

---

## 295. RadSight: Towards Perceptually Reliable Multimodal Radiology Image Understanding

**arXiv ID:** 2607.22293 | [PDF](https://arxiv.org/pdf/2607.22293v1)

**作者:** Jianqin Liu `[一作]` (DAMO Academy, Alibaba Group), Jianpeng Zhang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Perception‑Bench 评测框架和 RadSight 医学多模态大语言模型，针对低级视觉感知不足问题。

**💡 创新点**

创新点在于：① 将临床失败归因至低级感知，② 设计统一 2D/3D 低级感知评测维度，③ 采用双编码器与四阶段感知驱动训练课程，提升模型对细粒度病变属性与空间对应的掌握。

**🔧 技术方法**

技术手段包括：Qwen3‑VL 语言骨干、SigLIP‑NaViT 2D 编码器、PlainConvUNet/ Radar 3D 编码器、跨模态投影、统一自回归训练目标，四阶段课程（对齐 → 感知 → 诊断 → 报告）。

**📊 数据集**

数据集涵盖：MIMIC‑CXR、IU‑Xray、CT‑RATE、Merlin、INSPECT、AMOS‑MM、AbdomenAtlas 3.0、NLST 等 2D/3D 放射学图像、报告与分割掩膜，构建 1.13M Perception‑Bench 样本与 8.37M 训练语料。

**📈 对比分析**

与 3 类基线（通用、2D 专用、3D 专用）在 Perception‑Bench、公开 2D/3D VQA/诊断/报告任务上对比，RadSight 在所有维度均超过基线，最高可达 81.79%（2D 平均）和 44.33%（3D 任务综合）。

**⚠️ 局限性**

局限性：仅使用公开回溯性放射学数据，缺乏临床医生评估；小病灶定位仍不理想；评测仅覆盖 CT/胸片，其他影像模态与真实临床流程尚待验证。

---

## 296. Over-the-Air Interference Nulling Using Passive RIS for Two-Way K-User Interference Channel

**arXiv ID:** 2607.22259 | [PDF](https://arxiv.org/pdf/2607.22259v1)

**作者:** Junzhi Wang `[一作]` (Huazhong University of Science and Technology), Yingzhuang Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在双向 K 维干扰信道中利用被动 RIS 实现全自由度 (DoF) 干扰消除的可行性，并给出了所需 RIS 元素数量的必要与充分条件。

**💡 创新点**

首次将高维随机几何与浓缩现象相结合，量化不完全 CSI 对 DoF 的影响，并证明当 CSI 错误衰减指数 α≥1 时，干扰消除的 2K DoF 可以保持不变。

**🔧 技术方法**

采用高维概率论、凸几何、相位调制设计、交替投影与流形优化、误差衰减指数模型等技术进行理论分析和仿真。

**📊 数据集**

使用 Rayleigh 与 Rician 随机信道模型进行仿真，用户在平面上随机分布，未采用公开数据集。

**📈 对比分析**

与传统干扰对齐 (IA) 进行比较，仿真表明在满足阈值后 RIS 能实现接近 2K 的 DoF，最大速率显著优于 IA，所推阈值与实际性能吻合。

**⚠️ 局限性**

仅考虑单天线用户与连续相位 RIS，未覆盖离散相位、多天线用户、空间相关等实际情况，对低 SNR 误差模型的分析有限。

---

## 297. SiPhy: Single-Image Physical Property Reasoning

**arXiv ID:** 2607.22355 | [PDF](https://arxiv.org/pdf/2607.22355v1)

**作者:** Hoang Le `[一作]` (Michigan State University), Zijun Cui `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种单张图像推断物体物理属性的框架SiPhy，联合3D几何、语义和语言知识，实现像素级与整体质量估计。

**💡 创新点**

创新点在于引入3D感知采样、基于CLIP的视觉-语言对齐、VLM生成的材料候选、部分对比对齐以及重度感知厚度（HAT）模块，解决单视角下几何与物理不一致的问题。

**🔧 技术方法**

使用的技术包括CLIP视觉与文本特征、Fine-tuned VLM（Vicuna+CLIP）、SAM分割、局部注意力对比损失、重度感知厚度模块、Depth Anything等。

**📊 数据集**

实验使用的主要数据集为ABO‑500、MVImgNet‑100、PhysXNet‑100，并在HO3D、ARCTIC等真实手物交互数据上进行验证。

**📈 对比分析**

与多视角基线NeRF2Physics、PUGS、GaussianProperty以及单视角基线LLaVA等进行比较，SiPhy在ABO‑500质量预测的MnRE提升93%，密度和杨氏模量的MAE分别降低35.5%与23.5%，在真实交互数据上也优于LLaVA。

**⚠️ 局限性**

局限性包括对单视角几何先验的依赖，透明、反射、高纹理物体仍难以准确估计；厚度估计仅基于重度二分类，缺乏连续或组合材料的细粒度处理。

---

## 298. IQ-JEPA: A Joint-Embedding Predictive Architecture with a Hermitian Vision Transformer for Sound Speed and Attenuation Estimation from Ultrasound IQ Data

**arXiv ID:** 2607.22351 | [PDF](https://arxiv.org/pdf/2607.22351v1)

**作者:** Masashi Sode `[一作]` (University of North Carolina at Chapel Hill and North Carolina State University), Gianmarco Pinton `[通讯]` (University of North Carolina at Chapel Hill and North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种基于自监督学习的IQ-JEPA框架，用Hermitian Vision Transformer从原始IQ通道数据直接估计超声成像中的声速和衰减。

**💡 创新点**

创新点在于将联合嵌入预测（latent prediction）与复数域ViT相结合，并设计了保持U(1)相位对称性的共轭乘积前馈和Hermitian注意力，从而在无标签预训练中显著提升标签效率。

**🔧 技术方法**

使用了自监督预训练的联合嵌入预测目标、Hermitian Vision Transformer、复数卷积嵌入、旋转位置编码、相位增强以及Flash Attention加速。

**📊 数据集**

数据集为79,293个基于Fullwave 2.5有限差分时间域仿真的5角平面波超声IQ数据，包含随机多层和腹部解剖模型，标签为声速与衰减地图。

**📈 对比分析**

与监督训练的InversionNet、实数ViT、IQ ViT等基线对比，IQ-JEPA在10,000标签下声速MAE降至15.60 m/s，约为监督模型的1/3，整体误差在63,435标签时达到8.71 m/s，比基线低2.2×。

**⚠️ 局限性**

局限性包括仅在仿真数据上验证、仅针对单一传输模式和频率、对不同声速分布的泛化尚未测试，以及跨模态或真实临床数据的迁移需进一步研究。

---

## 299. Ask the Curator: Demonstrating Expert-Driven RDF Data Curation with HERITRACE

**arXiv ID:** 2607.22348 | [PDF](https://arxiv.org/pdf/2607.22348v1)

**作者:** Arcangelo Massari `[一作]` (University of Bologna), Silvio Peroni `[通讯]` (University of Bologna)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍并演示了 HERITRACE 系统在处理 PubMed 与 Crossref DOI 冲突案例中的完整配置与专家修订流程，从技术部署到实际合并和回滚操作；

**💡 创新点**

其创新点在于将 SHACL 与 YAML 规则配置解耦，形成可配置的编辑界面，同时每一次变更均记录为 RDF provenance 并提供时间机器回溯功能；

**🔧 技术方法**

所用技术包括 SHACL 约束、YAML 显示规则、OpenCitations Data Model、SPARQL triplestore、Docker Compose 以及 RESTful API；

**📊 数据集**

使用的数据集为 OpenCitations Meta 的 PubMed 与 Crossref 记录，并构造了一个演示图，其中包含 Nilsson 与 Maltezou 两篇文章的 RDF 子集；

**📈 对比分析**

重复检测采用基于属性 exact equality 的相似性匹配，演示中人工决策后成功纠正错误，系统通过时间机器实现快速回滚，性能未做量化评估；

**⚠️ 局限性**

局限性包括仅支持精确匹配的重复检测，缺乏相似度分析；配置需要人工编写 SHACL 与 YAML；并未实现并发编辑与冲突解决机制。

---

## 300. Towards Trustworthy and Cost-Efficient Data Integration: From Naïve RAG to Agentic RAG

**arXiv ID:** 2607.22319 | [PDF](https://arxiv.org/pdf/2607.22319v1)

**作者:** Chuangtao Ma `[一作]` (Aalborg University), Arijit Khan `[通讯]` (Bowling Green State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于Agentic RAG的可信、成本高效数据集成框架，并阐述从传统RAG到GraphRAG、KG-RAG再到Agentic RAG的演进与路线图。

**💡 创新点**

创新点在于将检索增强生成与多智能体协同、动态检索、迭代推理、批量处理和图形化持久记忆相结合，实现了对海量企业数据的可信、可解释且低成本集成。

**🔧 技术方法**

采用了大型语言模型、检索增强生成（RAG）、GraphRAG、KG-RAG、Agentic RAG、多智能体调度、MCP工具调用协议、批量推理与图形化记忆等技术。

**📊 数据集**

实验使用了多种公开行业数据集（医学EHR、金融、零售等表结构与实体匹配数据），但文中未给出具体数据集名称或编号。

**📈 对比分析**

通过与Naïve RAG、GraphRAG和KG-RAG的对比实验，展示了Agentic RAG在减少幻觉、提升匹配准确率和显著降低API调用/令牌消耗方面的优势，尽管文中未给出具体数值指标。

**⚠️ 局限性**

局限性包括：记忆与上下文冲突导致的错误、批量检索噪声与不确定检索阈值、参数调优与冷启动困难、系统尚未实现完全自治、缺乏针对企业场景的统一基准与信任度评估指标。

---

## 301. Evolution-Aware MSA Reasoning for Subsampling via Factor Graphs

**arXiv ID:** 2607.22314 | [PDF](https://arxiv.org/pdf/2607.22314v1)

**作者:** Zhangzhi Xiong `[一作]` (ShanghaiTech University), Jingyi Yu `[通讯]` (ShanghaiTech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于Affinity Propagation的因子图方法（AP-REASONER），将MSA子采样转化为可控的优化问题，允许在保持查询相似度和多样性的同时在固定令牌预算内选择子集。

**💡 创新点**

创新点在于：①把MSA子采样从经验式随机/阈值方法改为显式优化；②在Affinity Propagation框架中引入两个可调控制器α（多样性）和β（查询相似度）；③利用因子图推理实现全局协同选择，从而获得更高质量的下游表示。

**🔧 技术方法**

使用的技术包括：Affinity Propagation（因子图推理）、归一化基于间隙的汉明距离来构造相似矩阵、固定卡路里包装器来满足预算约束、以及在MSA Transformer和其他蛋白语言模型上进行无监督掩码语言建模与下游评估。

**📊 数据集**

使用的数据集包括：Protein Families数据库（UniRef/BFD）得到的MSA、RCSB PDB中20个蛋白的结构、CASP15 MSAs（全集与36个信息子集）、以及KaiB、RfaH、MAD2等可折叠蛋白的实验结构。

**📈 对比分析**

与随机、divMin、divMax、HHfilter、K-medians等传统采样器以及AlphaFold2等方法进行对比，AP-REASONER在长距离接触预测（Top‑L/2、Top‑L/5精度）和多构象预测（RMSD、覆盖率）上均优于基线，且可通过调节α、β实现可控的构象搜索。

**⚠️ 局限性**

局限性在于：①因子图推理和固定卡路里包装增加了相对轻量级采样器的计算开销（约2–3倍慢）；②实验仅在260K规模数据集上验证，尚未在更大规模（如26M）预训练语料上评估其可扩展性。

---

## 302. An optimal deterministic algorithm for finding a strict saddlepoint

**arXiv ID:** 2607.22312 | [PDF](https://arxiv.org/pdf/2607.22312v1)

**作者:** Justin Dallant `[一作]` `[通讯]` (TU Dresden), Justin Dallant (TU Dresden)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在 O(n) 时间内确定严格鞍点或判定不存在的方法，解决了此前仅已知随机化或更高复杂度的确定性问题

**💡 创新点**

通过将已有的 O(nlog n) 与 O(nlog^* n) 方法中的“单侧约简”与线性时间选择合并，并在每轮约减时一次性删除常数比例的行列，从而实现线性时间的确定性算法

**🔧 技术方法**

利用线性时间的“多列表选择”技术、阈值测试、一侧约简以及对矩阵行列的递归删减与恢复，保证在保持鞍点不变的前提下逐步压缩矩阵尺寸

**📊 数据集**

无外部数据集，算法对任意 n×n（或 m×n）矩阵通用，假设可对任意两条记录进行比较，且不依赖矩阵的具体数值分布

**📈 对比分析**

相较于之前的 O(nlog n) 与 O(nlog^* n) 确定性算法，本算法将时间复杂度降低到最优 O(n)（与随机化算法相同），仅需 O(n) 次矩阵查询，且实现简单

**⚠️ 局限性**

算法依赖于“多列表线性选择”这一非本质子程序，若实现不佳可能影响常数因子；此外，算法假设矩阵元素可互相比较且唯一，若不满足需额外预处理

---

## 303. A Roadmap to Impactful Pluralistic Alignment Research

**arXiv ID:** 2607.22305 | [PDF](https://arxiv.org/pdf/2607.22305v1)

**作者:** Elinor Poole-Dayan `[一作]` (Massachusetts Institute of Technology), Michiel A. Bakker `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文通过对前沿AI实验室的行为文件、模型卡和公开评估进行审计，发现多元价值对齐（pluralistic alignment）尚未在实际部署的模型中得到体现；进一步指出实现多元价值对齐的三大障碍：缺乏实证证明其效益、缺少清晰的行为规范、缺乏可直接部署的评估与方法，并提出针对这三点的研究路线图。

**💡 创新点**

创新点在于系统化评估多元价值对齐的行业采用现状，识别出与现有研究之间的差距，并将研究目标从理论和方法转向实际部署影响，形成了针对实证验证、规范制定与可落地评估的三步路线。

**🔧 技术方法**

使用的主要技术是文献综述、案例审计和对现有模型文档与评估的系统分析；未提出新的算法模型。

**📊 数据集**

采用的数据集与资源主要是公开的模型行为文档、模型卡、前沿实验室的评估报告以及少量公开的社区对齐数据集（Community Alignment Dataset）等。

**📈 对比分析**

论文并未给出新的实验或性能比较；其方法是对已发布的文档与评估进行对比与分析，未涉及模型性能指标。

**⚠️ 局限性**

局限性包括：缺乏对多元价值对齐实际效益的实证研究；对何时何种情境下应实施多元对齐的规范仍未达成共识；缺乏能够直接落地、可度量并能与其他目标兼容的评估指标和方法。

---

## 304. Synthetic Speech, Real Signal: Paralinguistic Preservation and Cross-Lingual Augmentation via Voice Cloning

**arXiv ID:** 2607.22304 | [PDF](https://arxiv.org/pdf/2607.22304v1)

**作者:** Roseline Polle `[一作]` (thymia), Stefano Goria `[通讯]` (thymia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对八种开源语音克隆模型在五个声学情感与临床抑郁焦虑检测任务上进行评估，检验克隆语音是否保留了下游任务所需的旁语信号，并进一步将英语克隆成日语以实现低资源语言的临床语音数据增强。

**💡 创新点**

创新在于系统性比较不同架构克隆模型对旁语信息的保留效果，并首次验证克隆语音可用于跨语言的临床情感检测，为低资源语言的临床语音数据扩增提供可行路径。

**🔧 技术方法**

使用基于WavLM Large提取的1024维声学嵌入，采用逻辑回归分类器，并通过AUC与Preservation Score衡量保留程度；同时评估说话人相似度与下游性能的相关性。

**📊 数据集**

使用公开情感、情绪、讽刺、口音数据集（IEMOCAP、MELD、MUSTARD、VCTK）和自有的英文与日文临床抑郁焦虑语料库。

**📈 对比分析**

通过对比原始与克隆语音在相同训练/测试条件下的AUC，发现大多数模型保留90%以上的旁语信号，跨语言克隆训练比直接英语转日语显著提升抑郁焦虑检测AUC（最高提升约4个百分点）。

**⚠️ 局限性**

局限包括仅评估单一语言对（英日）、仅使用开源克隆模型、仅采用逻辑回归与WavLM嵌入、跨语言基线简化、以及临床数据为专有数据，限制了可重复性与推广性。

---

## 305. On the Maximality of Additive Codes

**arXiv ID:** 2607.22297 | [PDF](https://arxiv.org/pdf/2607.22297v1)

**作者:** Tim Alderson `[一作]` `[通讯]` (University of New Brunswick Saint John), Tim Alderson (University of New Brunswick Saint John)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了可加码（additive code）的最大性与可扩展性，给出了ABS模型的推广，并证明在某些参数下可扩展码必定可加扩展，亦构造了不满足此性质的反例。

**💡 创新点**

提出了可加码ABS模型的等价性、可扩展性与极大性之间的几何判据，并证明在非素数域上存在可扩展但无可加扩展的码；同时给出了关于散射线性集的构造与性质的新见解。

**🔧 技术方法**

主要使用了有限几何（射影平面、线性集、方向问题）与线性代数（码的编码映射、投影、平面与线的几何互补）技术。

**📊 数据集**

未使用实验数据集，全部为理论构造与证明。

**📈 对比分析**

由于本工作为理论研究，未进行实验比较或性能评估，主要通过数学证明展示结果。

**⚠️ 局限性**

结果仅在特定参数下成立，尤其在m≥3或非素数域时可扩展性不再等价于可加扩展性，且构造的反例可能并不适用于所有码长度或维度。

---

## 306. From level set evolution to threshold optimization: A grayscale level set framework for image segmentation

**arXiv ID:** 2607.22255 | [PDF](https://arxiv.org/pdf/2607.22255v1)

**作者:** Xingkai Li `[一作]` (Harbin Institute of Technology), Zhichang Guo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出灰度级集框架，利用预处理后的光滑图像进行无长度正则化的阈值分割，实现快速高效的多退化图像分割。

**💡 创新点**

创新点包括：①在平滑条件下证明长度正则化可省略；②将连续变分演化转化为离散阈值搜索；③设计增量阈值更新算法，将复杂度从 O(MNS) 降至 O(NS)。

**🔧 技术方法**

采用预处理模块（如 TV 去噪、Gaussian 滤波、bias field 校正等）得到平滑图像；使用灰度阈值分割与局部可扩展拟合能量；实现增量阈值搜索；并与传统变分、联合模型及深度网络进行对比实验。

**📊 数据集**

实验数据集包括合成噪声/低对比/光照不均图像；真实图像集如 WBC 血细胞、BrainWeb MRI、DIAS 血管、MoNuSeg 组织切片等。

**📈 对比分析**

与经典 CV/RSF/LIC、联合模型、U‑Net 等方法在 Dice、IoU、HD95、Length、MAC、CPU 时间等指标上对比；结果显示分割准确度相当或更优，且运行时间显著下降（O(NS) 级别）。

**⚠️ 局限性**

局限性：需先进行平滑预处理，若退化过重或不满足平滑假设可能失效；当前仅支持二值分割，需扩展到多相；参数仍需人工调节；对极端噪声的鲁棒性待进一步验证。

---

## 307. IFCLoRA: Topology-Aware Rank Allocation for Parameter-Efficient Fine-Tuning

**arXiv ID:** 2607.22251 | [PDF](https://arxiv.org/pdf/2607.22251v1)

**作者:** Wei Zhang `[一作]` (Central China Normal University), Yihang Cheng `[通讯]` (Central China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于任务条件信息流中心性的 LoRA 低秩适配器权重分配方法 IFCLoRA。

**💡 创新点**

将任务条件交互图与全局信息流拓扑结合，利用信息流中心性在预训练前一次性分配有限的低秩参数。

**🔧 技术方法**

使用零 ablation 追踪构造任务条件稀疏交互图、图中心性分析、梯度敏感度校准及一次性预算约束分配。

**📊 数据集**

在 GSM8K（数学推理）和 SuperGLUE（语义理解）上，使用 LLaMA3-8B、Qwen3-8B、Qwen3-14B 三个大模型进行实验。

**📈 对比分析**

与 LoRA、AdaLoRA、EVA 等基线在相同总秩预算下对比，IFCLoRA 在 GSM8K 上平均提升约1.13个百分点，在 Qwen3-8B r=4 时提升1.36个百分点，整体表现更稳定。

**⚠️ 局限性**

缺乏理论最优性保证，图构建仅适用于解码器型 Transformer，且仅在有限任务和模型规模上验证。

---

## 308. Over-the-Air Interference Nulling Using Active RIS

**arXiv ID:** 2607.22239 | [PDF](https://arxiv.org/pdf/2607.22239v1)

**作者:** Junzhi Wang `[一作]` (Huazhong University of Science and Technology), Yingzhuang Liu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在两路 K‑用户干扰信道中使用主动可重构智能表面（RIS）实现全自由度（DoF）的干扰消除，并推导了满足硬件约束下所需反射元件数的必要与充分条件；

**💡 创新点**

创新点在于将主动 RIS 的干扰消除问题视为带有范数与盒约束的随机线性系统的可行性问题，并利用高维凸几何与 Gordon 定理给出了精确的 RE 数量阈值和功率权衡分析；

**🔧 技术方法**

主要技术包括随机线性系统可行性分析、高维概率论（范数集中与 Wishart 分布）、Gordon 定理改写以及几何投影法；

**📊 数据集**

使用仿真数据，采用 Rayleigh 与 Rician 混合衰落模型，随机生成用户位置、RIS 与用户之间的距离，并计算对应路径损耗；

**📈 对比分析**

通过与理论阈值及传统被动 RIS 对比，仿真显示主动 RIS 在满足反射功率和放大增益足够时，只需与 2K(K‑1) 个元件相近的数量即可实现全 DoF，且所需 RE 数量随放大增益和反射功率显著降低；

**⚠️ 局限性**

局限性包括假设完美 CSI 与自干扰完全消除、仅考虑两路 K‑用户场景、对高维近似和概率阈值的保守性可能导致实际所需 RE 数量高于理论预测，以及未考虑非理想硬件噪声和频率选择性等实际因素。

---

## 309. Optimization of time-consuming experimental conditions using pseudo-experimental data guided by adaptive polynomial regression

**arXiv ID:** 2607.22238 | [PDF](https://arxiv.org/pdf/2607.22238v1)

**作者:** Hirotaka Sugawara `[一作]` (Keio University), Akira Funahashi `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PolyBO——一种在高维、实验耗时较长的场景下，通过低容量多项式回归生成伪实验数据并与真实实验数据融合来加速 Bayesian Optimization 的方法。

**💡 创新点**

创新点在于：①使用低容量多项式回归而非深度或潜在空间模型，避免在数据稀缺时出现 VAE 训练不稳或深度模型过拟合；②在每次迭代更新并替换伪实验数据的“更新机制”，防止早期低质量伪数据拖累后续搜索；③通过实验验证多项式度数与伪实验样本量的合理区间，显著提升探索效率。

**🔧 技术方法**

核心技术包括 Gaussian Process 回归（代理模型）、多项式回归（伪实验生成器）、EI 与 GP‑UCB 采集函数、BFGS 求解器、t‑SNE/UMAP 可视化、Observation Entropy 与 Proposed‑point Entropy 指标，用于量化搜索行为。

**📊 数据集**

数据集：①BBOB 24 维连续优化基准（D = 2, 5, 10, 20）；②基于神经网络预测的高熵合金（HEA）组成与性能仿真，用作真实世界的材料组合优化实验。

**📈 对比分析**

与 vanilla BO、随机采样、BOPP、TSBO（仅在 D=20）以及 RL‑DQN 进行对比。采用 I_50（达到 vanilla BO 在 50 次迭代后简单后悔值所需迭代数）和 I_1500（达到 RL‑DQN 性能所需迭代数）等指标。结果显示，PolyBO 在 D=10、20 的基准函数上平均可缩短 42% 的迭代次数；在 HEA 任务中，达到相同 FOM 的时间缩短 96%。

**⚠️ 局限性**

限制：①在低维（尤其 D=5）场景下性能提升有限；②多项式度数和伪实验样本量需要手动调参，过高度数导致显存指数增长；③伪实验数据的“拟真”程度随迭代递减，当前仅通过固定样本量控制，未能动态调节信息源可信度；④方法在大规模约束或高非线性函数上可能受限。

---

## 310. Rethinking Accuracy: A Weighted Error-Based Metric for Data Quality

**arXiv ID:** 2607.22279 | [PDF](https://arxiv.org/pdf/2607.22279v1)

**作者:** Valerie Restat `[一作]` (FernUniversität in Hagen), Uta Störl `[通讯]` (FernUniversität in Hagen)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了可加权的错误基准度量指标TOMME，用于一次性评估数据质量并对不同清洗管道进行比较。

**💡 创新点**

创新点在于将错误量化为加权总和，并结合错误类型、属性及错误数的惩罚项，形成可调节的单一评分，从而填补了缺乏通用数据质量指标的空白。

**🔧 技术方法**

利用CheDDaR生成详细错误报告，在此基础上推导TOMME公式，并通过Web UI实现交互式计算与可视化。

**📊 数据集**

以合成奶酪购买记录为实验数据，使用GouDa生成多版本错误率不同的数据集进行实验。

**📈 对比分析**

通过在实验中计算不同错误率、错误数和权重组合下的DQ_TOMME，并与传统准确率等指标对比，验证其对管道性能的敏感性与可解释性，实验结果显示TOMME能在保持单一评分的同时捕捉细粒度错误差异。

**⚠️ 局限性**

目前仅涵盖级别1、2的错误类型，未纳入噪声等高级错误；缺乏对不同应用场景默认权重的指导；指标对大量属性的处理仍有改进空间。

---

## 311. Class-Balanced Softmax: A Bayes Theory-Based Method for Long-Tailed Recognition

**arXiv ID:** 2607.22258 | [PDF](https://arxiv.org/pdf/2607.22258v1)

**作者:** Yi-Hang Zhu `[一作]` (University of Leicester), Huiyu Zhou `[通讯]` (University of Leicester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于贝叶斯框架与幂律假设的简单 logit 调整方法——cbs，用于解决长尾不平衡分类中的模型偏好问题。

**💡 创新点**

创新点在于将类平衡软max推广为可调参数化的 logit 加权（βlogN_c），同时通过理论推导和实验验证阐明了偏好问题的本质。

**🔧 技术方法**

技术主要包括理论推导的贝叶斯分布转换、幂律权重近似以及在现有深度学习流水线中无额外参数的 logit 调整实现。

**📊 数据集**

实验数据集涵盖CIFAR10/100长尾版本、ImageNet-LT、iNaturalist2018、Place-LT以及极端不平衡的LVIS-基准。

**📈 对比分析**

与Balanced Softmax、Focal Loss、τ‑norm、Adjust Logit、Reslt、CRT、LDAM及新型非软max方法相比，cbs在各数据集上均取得更低的模型不平衡度（I）和更高的总体召回率，尤其在极端长尾场景下表现尤为突出。

**⚠️ 局限性**

局限性包括对幂律假设的经验依赖、需要手工调节β超参数、仍存在头尾权衡以及目前仅在图像分类任务验证，尚待推广到目标检测等任务。

---

## 312. AgentHOI: Multi-Agent Reasoning for Human-Object-Interaction Video Generation via Implicit Representation Alignment

**arXiv ID:** 2607.22241 | [PDF](https://arxiv.org/pdf/2607.22241v1)

**作者:** Ziyao Huang `[一作]` (University of Chinese Academy of Sciences), Fan Tang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AgentHOI框架，采用多代理推理（感知、关系分析、运动规划、裁剪细化）实现“先思考再生成”，在文本输入下生成无显式运动控制的HOI视频。

**💡 创新点**

创新点在于①多代理思维-先生成流程，将交互逻辑拆解为物理可执行蓝图；②隐式文本-运动特征对齐（TRD）将文本到运动的先验注入扩散模型；③混合真实与合成HOI数据管线，显著提升多样性与泛化。

**🔧 技术方法**

使用Wan2.1‑I2V扩散模型、3D VAE、CLIP+T5嵌入、Token concat、RoPE、TRD损失、流匹配损失、EMA‑VFI、VLM（InternVL、Gemini、Qwen‑VL）进行多代理推理。

**📊 数据集**

构建108k样本混合数据集：71k真实HOI视频（从10M线上视频过滤获得），37k合成HOI视频（FLUX+Gemini生成），并利用AnchorCrafter、HOMA等公开基准进行评测。

**📈 对比分析**

与AnchorCrafter、Animate‑X、UniAnimate‑DiT、VACE‑14B、HuMo、HOMA等方法对比。AgentHOI在Obj‑DINO、DD、MS、TVA、InternVL等指标上显著优于基线，尤其在跨动作驱动测试集上提升约10–20%。

**⚠️ 局限性**

局限性包括：低分辨率导致文本模糊；无严格时间控制，VLM可能产生错误动作时序；合成数据仍存在纹理失真；对高质量文本与图像输入敏感；推理时间相对较长。

---

## 313. Learning Structural Convergence: A Neuro-Symbolic Benchmark for Temporal Reasoning

**arXiv ID:** 2607.22365 | [PDF](https://arxiv.org/pdf/2607.22365v1)

**作者:** Michael Romei De Socio `[一作]`, Alessio Merlo `[通讯]` (School of Advanced Defense Studies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个名为 TRACTA 的合成基准，用于在高复杂度事件驱动系统中评估时间结构推理，并设计了三项任务（提前预警、模式检测、运行级分类）。

**💡 创新点**

创新点在于提出了将语义轨迹生成与时间学习相结合的神经符号框架，并通过运行级别别名与快捷方式诊断控制数据泄漏，形成了一套完整的评估流程。

**🔧 技术方法**

技术方法包括：基于事件到语义轨迹的转换、双向 GRU 与 Transformer 的时间序列学习、基于规则的合同式语义比较器，以及多种消融与快捷方式诊断。

**📊 数据集**

使用的数据集为 200 条合成 MDO‑风格运行（每条 96 步，正负各 100 条），其中包含事件级别特征与能力轨迹信息。

**📈 对比分析**

比较方法采用宏 F1 作为主要指标，对早期预警、模式检测和运行级分类分别评估；神经符号模型在早期预警和模式检测上分别取得约 0.69 与 0.78 的宏 F1，略高于原始事件序列基线，但在运行级分类上差距不大。

**⚠️ 局限性**

局限性包括：仅在合成环境中验证，样本量有限；语义层与标签生成高度耦合，可能导致过拟合；仍存残留的浅层快捷信号；未在真实操作数据上评估，缺乏外部有效性。

---

## 314. Indexing: the Beginning and the End

**arXiv ID:** 2607.22361 | [PDF](https://arxiv.org/pdf/2607.22361v1)

**作者:** Alexander Kozachinskiy `[一作]` (CENIA), Felipe Urrutia `[通讯]` (Pontifical Catholic University of Chile)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了索引（Indexing）这一最基本的检索任务，探讨了不同深度网络（RNN、SSM、线性注意力 Transformer、全注意力 Transformer 等）在该任务上的表达能力和信息瓶颈。

**💡 创新点**

提出了“因果复杂度”（Causal Complexity）概念，利用它给出了在无限精度下，常见掩码架构（RNN、SSM、因果线性注意力）在右手索引任务上必需的层数上限，并给出全注意力 Transformer 和无掩码线性注意力在一层/两层内可解该任务的构造。对左手索引任务则证明 RNN 一层即可解决，而其他架构至少需要两层。

**🔧 技术方法**

主要技术包括：因果复杂度的定义与分析、VC 维数与算术复杂度的关系、对 Transformer、RNN、SSM 以及线性注意力层的形式化描述，以及对索引任务的理论证明和实验验证。

**📊 数据集**

使用合成的二进制索引数据集：长度 n∈{8,16,32,64} 的随机二进制串及对应的索引 i；训练样本按需在线生成。

**📈 对比分析**

通过在 5 种随机种子下训练 23 种不同模型配置（不同深度、注意力类型、是否掩码等），记录每种配置在每个 n 下的成功种子数。实验显示：构造性（理论上可解）的配置几乎全部成功（最大 40/40），而压力测试（理论上不可解）的配置成功率随 n 增大迅速下降（如 0/75 在 n=64）。

**⚠️ 局限性**

局限性包括：实验仅在中等长度下验证，无法完全覆盖理论上的渐近极限；训练失败并不等价于不可解；所用数据集为人工合成，缺乏对实际 NLP 任务的直接映射；因果复杂度仅适用于掩码或因果结构，无法泛化到所有模型。

---

## 315. Integrated Order Dispatching and Routing for Last-Mile Pickup via Deep Reinforcement Learning

**arXiv ID:** 2607.22356 | [PDF](https://arxiv.org/pdf/2607.22356v1)

**作者:** Yida Xu `[一作]` (Tianjin University), Yiting Sun `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结合强化学习路由oracle与实时派遣启发式的集成优化框架，用于解决最后一公里取件问题中的订单派遣与路径规划耦合决策

**💡 创新点**

创新点包括：1）基于动态图注意力网络的DR-LaCPNet编码器与前瞻式与骑手个性化解码器的组合；2）将训练好的路由oracle嵌入派遣启发式，形成位置池贪心+局部搜索（PP-Greedy-LS）策略；3）统一的时间依赖旅行时间MILP建模与DRL训练；4）数据驱动的实时实例模拟器

**🔧 技术方法**

使用动态图注意力网络、深度强化学习（policy gradient + 软基线更新）、基于混合整数规划的评估与对比、位置池与局部搜索启发式

**📊 数据集**

CaiNiao物流的LaDe-P数据集（≈10.6M包裹，≈619k骑手轨迹），覆盖五个城市的真实采集数据

**📈 对比分析**

在离线评估和滚动时域仿真中，DR-LaCPNet+PP-Greedy-LS在大规模实例下平均在0.1秒内完成路由评估，整体目标值明显优于MILP、贪心、Tabu搜索、DeepRoute、Graph2Route，且在线时域仿真中平均目标值下降约20%–30%，时间窗口违约率和行驶时间显著下降

**⚠️ 局限性**

限制包括：训练奖励稀疏、模型对节点数的泛化有限、对真实数据特征的依赖性强、以及在极大规模或极高峰时仍需改进采样与基线更新策略

---

## 316. Time-Reversed Imaging: A Multimodal Benchmark and Framework for Reconstructing Past Human-Environment Interactions

**arXiv ID:** 2607.22352 | [PDF](https://arxiv.org/pdf/2607.22352v1)

**作者:** Jorge Bacca `[一作]`, Mauro Dalla Mura `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了时间反演成像（Time-Reversed Imaging）框架，利用热、紫外和可见光三模态残余痕迹推断最近的过去人机交互场景。

**💡 创新点**

创新点在于将多模态物理痕迹与结构化语义描述相结合，使用 VLM 引导扩散模型生成可视化重建，并首次构建 TRACE-HEI 数据集与统一评测协议。

**🔧 技术方法**

主要技术包括多模态同步采集、结构化过去事件描述（SPED）、Vision-Language Model 推理以及 VLM 指导的扩散生成网络。

**📊 数据集**

使用了 TRACE-HEI 数据集，该数据集包含 100 组 RGB/UV/热像素同步视频，涵盖多种交互动作、物体与材质。

**📈 对比分析**

在多模态+SPED 配置下，实验显示比单模态或无结构化描述方案提升约 10–20% 的精确匹配率，重建的 PSNR、SSIM、LPIPS、AO、IoU 均显著优于基线，且在 30 秒内可保持高水平性能。

**⚠️ 局限性**

局限包括对时间延迟的依赖、残留痕迹随时间衰减导致的性能下降、对材料热/荧光特性的假设、以及在复杂室外环境与大规模真实场景中的泛化能力不足。

---

## 317. Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization

**arXiv ID:** 2607.22334 | [PDF](https://arxiv.org/pdf/2607.22334v1)

**作者:** Hao Wang `[一作]` (University of Chinese Academy of Sciences), Honggang Qi `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种跨词表的全词汇监督方法，利用字节前缀映射将教师模型的下一个词分布转换为学生模型词表上的分布，并在学生自己的轨迹上进行On‑Policy蒸馏；

**💡 创新点**

创新点在于：①通过字节前缀映射实现Exact full‑vocabulary监督，满足词汇完整性、字节对齐和质量保持；②在出现全空格的代码行时预先掩蔽目标，避免分隔噪声导致可执行性崩溃；③在多对比实验中表现出对三种不同教师（Qwen3‑32B、GLM‑Z1‑9B、MiniMax‑M2.7）的优异性能，闭合教师与学生间 34%~44% 的性能差距。

**🔧 技术方法**

核心技术包括：字节级目标映射（ϕ映射）、链式修正跨-tokenizer跨度位置、通用Jensen–Shannon（GJS）或其端点KL的偏差学习、预计算全空格掩蔽、停词处理与桥接目标；

**📊 数据集**

使用了20,000条提示，分别包含10,000道数学题（DAPO‑Math）和10,000道代码题（TACO），并在Mathematics、Code 三个评测基准上评估（MATH‑500、AIME、HMMT、HumanEval+、LiveCodeBench、TACO）。

**📈 对比分析**

与SimCT、ULD、GOLD以及SeqKD等基线进行比较。该方法在所有教师‑学生对中均优于三种分布式基线和SeqKD，平均在六项评测上提升 3.7–6.6 分（avg@8），并在代码可执行性、数学准确率等指标上取得显著提升；

**⚠️ 局限性**

局限性包括：①在全空格行上使用掩蔽，导致约3.5%的对齐代码行未参与训练，损失一部分梯度；②未测试更高质量数据或更大模型的情况下是否仍保持同样的性能提升；

---

## 318. Teachy Mini: Development and Preliminary Evaluation of a Knowledge-Based Generative Social Robot for Higher Education

**arXiv ID:** 2607.22345 | [PDF](https://arxiv.org/pdf/2607.22345v1)

**作者:** Stephan Vonschallen `[一作]` (Zurich University of Applied Sciences), Theresa Schmiedel `[通讯]` (Zurich University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在研究中构建了 Teachy Mini，基于 Reachy Mini 机器人实现了知识驱动设计（KBD）框架，并在单次学习交互实验中测试其效果。

**💡 创新点**

首次将 KBD 需求落地到 GSR 机器人，实现自我、用户、上下文知识的动态集成，显著提升学生对责任教学行为的感知。

**🔧 技术方法**

采用系统提示、检索增强生成（RAG）与状态化提示编排，配合 OpenAI Realtime 大语言模型实现实时对话。

**📊 数据集**

使用 11 张研究方法学的 PPT/PDF 作为知识库，收集 24 名大学生/研究生的实验数据。

**📈 对比分析**

通过双组实验（KBD 与对照）与预后测试及主观量表比较，结果显示责任感知显著提升（p=0.034），但接受度、动机及学习成效差异不显著。

**⚠️ 局限性**

局限包括样本量不足、单次交互、仅实现 KBD 子集、对 OpenAI 模型的依赖以及可能的“新奇效应”。

---

## 319. Bringing GRACE to Recommendation: Fine-Tuning for Sustainable and Accurate Personalization

**arXiv ID:** 2607.22341 | [PDF](https://arxiv.org/pdf/2607.22341v1)

**作者:** Yibowen Zhao `[一作]` (Shandong University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出GRACE框架，利用微调方式将可持续性信号直接融入预训练推荐模型，实现绿色推荐；

**💡 创新点**

创新点在于引入可微的绿色损失与梯度投影机制，既能优化可持续性目标，又能抑制与用户偏好梯度冲突；

**🔧 技术方法**

采用Gumbel-Softmax软排序逼近top‑K、可微的绿色和推荐损失、梯度投影策略以及多任务优化；

**📊 数据集**

使用GreenRec和RecipeEmission两个真实食品推荐数据集，分别包含环境影响、营养与健康指标；

**📈 对比分析**

与多种基线（FHFRS、CFARS、GradNorm、MGDA）及四种主流预训练模型（SASRec、FDSA、FEARec、GRAPE）对比，GRACE在保持或提升Hit@K/NDCG的同时，显著提升可持续性指标，整体排名最佳；

**⚠️ 局限性**

局限在于仅针对离散可微化的绿色标签，需较大GPU内存；对极端多目标冲突仍有限控制，且主要验证在食品领域，需进一步推广至其他领域。

---

## 320. Efficient Recommendations via Graph Coarsening and Label Propagation

**arXiv ID:** 2607.22287 | [PDF](https://arxiv.org/pdf/2607.22287v1)

**作者:** Alessandro Sbandi `[一作]` (TIM S.p.A.), Fabrizio Silvestri `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于业务规则驱动的图压缩与双阶段标签传播的推荐框架，能在电信领域大规模图上实现高效精准推荐

**💡 创新点**

创新点在于：①将业务知识嵌入社区划分，自动聚合用户构建压缩图；②采用两阶段传播（粗图LPA/GNN + 细粒度LPA）既保持低延迟又提升个性化；③在工业规模数据上实现秒级推理

**🔧 技术方法**

技术包括图压缩（基于业务规则的社区聚合）、Label Propagation Algorithm（LPA）、GraphSAGE GNN（用于粗图传播）、子图内细化LPA

**📊 数据集**

使用 TIM 电信公司 13M 用户的营销呼叫推荐数据集，包含 561 种促销商品和 13M 条用户交互记录

**📈 对比分析**

与全图 LPA、基于全图或 Louvain/Unique 的压缩+LPA/GNN 等基线对比，NDCG@5 在 LPA 阶段提升约 24%（比全图 LPA），GNN 阶段提升约 54%，同时推理时间从 17 分钟降至数十秒

**⚠️ 局限性**

局限性包括：GNN 训练成本高且对实时更新敏感；压缩策略依赖业务规则，可能难以直接迁移到其他领域；在极端稀疏或结构变化剧烈的图中性能衰减

---

## 321. Kalyna Block Cipher: From Design Space Exploration to ASIC Design

**arXiv ID:** 2607.22269 | [PDF](https://arxiv.org/pdf/2607.22269v1)

**作者:** Carlos Gewehr `[一作]` (Carnegie Mellon University), Samuel Pagliarini `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文针对乌克兰标准Kalyna块加密算法，完成了ASIC实现的设计空间探索，包括加密、解密和统一加密/解密功能的多种架构；

**💡 创新点**

创新点在于提出低面积、低延迟的多种并行/串行架构，设计了硬件减约技术（如KSA、S-box/混合列优化），并在最小面积架构上实现SCA和FI攻击的隐藏、掩码、时空复制等安全防护；

**🔧 技术方法**

采用KSA加速加法、精简S-box与混合列实现、XorShift伪随机数生成、第一阶掩码与随机化执行、时空复制检测等硬件技术；

**📊 数据集**

使用内部生成的随机10,000条明文/密钥对以及特定的“零输出”测试向量进行TVLA泄漏评估；

**📈 对比分析**

与AES的ASIC实现对比，Kalyna在最小面积架构下面积约为AES的2.1倍、延迟约2.3倍、能耗约8.3倍；在实现安全防护后面积、延迟、能耗分别提升至2.2~2.5倍、3.7~4.9倍、46倍以上，但仍保持可接受的时钟频率（最高≈971 MHz）；

**⚠️ 局限性**

局限性包括：安全防护大幅增加面积与功耗，时空复制对泄漏的影响仍需进一步研究；在高频下功耗与功耗泄漏的平衡尚未完全优化；未来需评估更广泛的攻击模型及更高效的硬件实现。

---

## 322. Autoregressive EHR Foundation Models with Multimodal Inputs

**arXiv ID:** 2607.22264 | [PDF](https://arxiv.org/pdf/2607.22264v1)

**作者:** Yuxuan Liu `[一作]` (Imperial College London), A. Aldo Faisal `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种框架，将自回归 EHR 基础模型与 ECG、胸透影像和临床笔记等多模态输入相结合，利用潜在压缩和门控跨注意力实现融合；

**💡 创新点**

创新点在于引入潜在压缩模块以高效处理长序列、门控跨注意力配合时间对齐掩码实现因果融合，并系统评估不同预训练编码器对下游任务的影响；

**🔧 技术方法**

采用 GPT‑2 风格解码器，冻结的多模态预训练编码器（CSFM、ECG‑FM、BioMedCLIP、ViT‑MAE、BioMedBERT），潜在压缩、门控跨注意力、时间对齐掩码及基于 rollout 的零样本 ICU 死亡预测；

**📊 数据集**

使用 MIMIC‑IV 及其子数据集（MIMIC‑IV‑ECG、MIMIC‑CXR‑JPG、MIMIC‑IV‑Note），并按 80/10/10 的主体级拆分；

**📈 对比分析**

通过与仅使用 EHR 的基线模型对比 ICU 死亡预测（AUROC/AUPRC/F1），发现潜在压缩显著优于均值池化和未压缩；在双模态设置中，只有使用 BioMedBERT 的笔记模态略优于基线，三模态统一模型仍未超越基线；

**⚠️ 局限性**

局限性包括多模态数据稀疏、时间可用性差导致信息利用不足，单一下推目标可能抑制辅助模态贡献；评估为随机化 rollout，实验未在多种随机种子下重复，需更灵活的融合与模态感知训练。

---

## 323. Design and Human Evaluation of Tactile Withdrawal Reflexes for a Skin-Covered Robot Arm

**arXiv ID:** 2607.22249 | [PDF](https://arxiv.org/pdf/2607.22249v1)

**作者:** Laura Babayeva `[一作]` (Czech Technical University in Prague), Matej Hoffmann `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了一套完整的人工伤痛感知与撤退反射管线，利用覆盖全臂的分布式触觉皮肤对UR10e机械臂进行触碰检测并执行三种不同的撤退策略。

**💡 创新点**

创新点在于提出了基于连续非线性痛感增益的映射模型，并对比了生物启发的位置依赖关节空间反射、统一的固定关节增量反射以及工程化的笛卡尔空间反射三种策略。

**🔧 技术方法**

采用了UR10e机械臂、AIRSKIN全身触觉皮肤、ROS平台、关节/空间控制算法以及痛感增益的Sigmoid连续模型与离散阈值模型。

**📊 数据集**

使用的“数据集”为15名参与者在每种反射条件下约10次触碰的实验数据，包括触碰强度、关节位移、主观评估问卷等。

**📈 对比分析**

通过Godspeed量表、自定义自然性、舒适度与感知量表以及配对Wilcoxon检验对三种反射进行比较，结果显示统一关节反射在自然性、可预测性与安全感上表现最佳，笛卡尔反射被评为最适合触碰的反应。

**⚠️ 局限性**

局限性包括触觉皮肤的分辨率与频率限制、机器人与人类手臂的形态差异导致的可解释性问题、反射动作过于突兀或缺乏平滑过渡，以及未在不同机器人形态上验证。

---

## 324. Beyond Binary Rooftop Mapping: A Four-Class Deep Learning Framework for Green Roof Potential Assessment from Open Swiss Geospatial Data

**arXiv ID:** 2607.22342 | [PDF](https://arxiv.org/pdf/2607.22342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 325. Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads

**arXiv ID:** 2607.22242 | [PDF](https://arxiv.org/pdf/2607.22242v1)

**作者:** Tianxi Lu `[一作]` (Brown University), Sherief Reda `[通讯]` (Brown University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文开发了一种基于大语言模型与运行时监控协同的Agentic CPU‑GPU调度器，能够为异构AI工具工作流动态分配CPU、立即GPU、排队GPU三种执行模式，以最小化端到端延迟。

**💡 创新点**

创新点在于将LLM作为决策层，并配备精细化的运行时监控（滑动平均、重测、交换探测），使得LLM能够在无离线训练的情况下通过实时反馈扩展观察空间，从而实现与穷举最优相同的映射，并首次引入三种执行模式与VRAM预算约束的组合。

**🔧 技术方法**

使用的技术包括离线工具性能剖析（CPU/GPU延迟、加载时间、VRAM占用）、大语言模型DeepSeek‑R1‑Distill‑Qwen‑32B‑AWQ（通过vLLM推理）、基于滑动窗口统计的运行时监控、重测与交换重测机制，以及与HEFT、UCB1、StarPU等传统调度器对比。

**📊 数据集**

数据集方面，19个公开AI工具在ImageNet、SST‑2、MNLI、LibriSpeech、MS MARCO等真实数据集上进行性能测量，并通过工具放大（重复k次）生成对冲突的实验场景。

**📈 对比分析**

评估在13个四工具场景下，将Agentic调度器与All‑GPU、HEFT、UCB1、StarPU、Greedy等基线对比，结果表明Agentic在所有场景均达到穷举最优映射，匹配UCB1的映射精度，并在precedence、GPU冲突、VRAM限制三类场景中显著优于传统策略，且在冷启动与热启动两种模式下均能实现。

**⚠️ 局限性**

局限性包括缺乏正式的收敛保证（监控探索策略仍为启发式）、需要离线剖析以迁移至新硬件、监控开销随工作负载变化且对大规模任务/多设备扩展性尚未验证，以及LLM推理仍需要额外GPU资源。

---

## 326. Three-player Differential Game Logic

**arXiv ID:** 2607.22359 | [PDF](https://arxiv.org/pdf/2607.22359v1)

**作者:** Julia Butte `[一作]` (Karlsruhe Institute of Technology), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并形式化了三玩家差分游戏逻辑（3‑Player Differential Game Logic，简称3‑PDGL），用于推理离散与连续动力学交织的三玩家非零和混合游戏，并允许玩家在游戏过程中自由组建或解散联盟；

**💡 创新点**

①将传统两玩家混合游戏逻辑扩展到三玩家，首次支持动态联盟形成；②证明该逻辑与两玩家逻辑等价，显示其完整性；③给出一套既完备又可证明的推理算子和公理；

**🔧 技术方法**

逻辑语法与语义的定义、可归约性与可变换技巧、固定点与递归定义、证明论（公理与规则）以及相对完备性证明（通过归约到两玩家逻辑实现）；

**📊 数据集**

无，本文为理论研究，没有使用实验数据集；

**📈 对比分析**

通过形式化证明与公理化推理来验证逻辑的可靠性，没有数值实验或性能指标；

**⚠️ 局限性**

仅适用于三玩家（可推广到n玩家但未实现），且需要预先知道玩家目标和联盟结构，无法处理不可预知的目标或动态出现的新玩家；

---

## 327. Do Agent Benchmarks Measure Capability? Protocol Validity in the Age of Agentic AI

**arXiv ID:** 2607.22368 | [PDF](https://arxiv.org/pdf/2607.22368v1)

**作者:** Jiaqi Shao `[一作]` (Tencent), Bing Luo `[通讯]` (Duke Kunshan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 HackDetect 的后置审计方法，用于检测和量化代理基准中的协议有效性失效（即奖励“hacking”或“misleading”），并通过 2,385 条追踪记录评估了 15 个代理基准的短路比例与分数膨胀。

**💡 创新点**

创新点在于：① 统一的协议有效性概念和三步链（Expose→Exploit→Mislead）；② 构建了可复现的审计框架 HackDetect，能够从基准规范、轨迹、提交文件和评分记录中自动识别曝光、利用与误导；③ 引入 Mislead Gap（G=S_exploit−S_intended）量化分数膨胀；④ 对多种基准（Frontier Science、AutoLab 等）进行大规模跨基准审计，首次揭示高比例的短路与显著分数膨胀。

**🔧 技术方法**

技术包括：1) 规范与轨迹的结构化打包；2) 通过 LLM（GPT‑5.5）进行固定提示的评判，以判定曝光、使用与分数影响；3) 对审计结果做一致性验证；4) 通过对比实验（修复、消融、同任务无源版本）计算 Mislead Gap；5) 采用多阶段基准进化模型以说明协议暴露点。

**📊 数据集**

使用了 15 个公开代理基准的轨迹数据：Frontier Science、AutoLab、SWE‑bench、FrontierSWE、NL2Repo‑Bench、Terminal‑Bench、WildClawBench、DeepSWE、BrowseComp、MLS‑Bench 以及其变体，总计 2,385 条跟踪记录。

**📈 对比分析**

通过手工标注对 HackDetect 的精度进行验证，发现对 21 条 MLS‑Bench 示例的精度 0.94、召回 0.76、F1 0.84；在 Frontier Science 的 53 条手标样本中，精度 0.94、召回 0.76。对已知的短路基准（Frontier Science、AutoLab 等）进行对比实验，计算出 Mislead Gap 在 0.447–1.00 之间，说明短路导致的分数膨胀显著。

**⚠️ 局限性**

局限性包括：① 仅对已保留轨迹、提交文件和评分记录的基准可复现；② Mislead Gap 只能在存在可比对照的情况下计算；③ 对于预选的“可疑”样本，其比例不代表整个基准的普遍程度；④ 评判器的召回率仍有提升空间，尤其对较为隐蔽的曝光难以捕捉；⑤ 该方法主要针对后置审计，未直接覆盖运行时监控或预防措施。

---

## 328. Geometric 2D Scene Graph Generation

**arXiv ID:** 2607.22325 | [PDF](https://arxiv.org/pdf/2607.22325v1)

**作者:** Christoph Jahn `[一作]` (Mercedes-Benz AG), Bastian Goldluecke `[通讯]` (University of Konstanz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文提出了一种在没有语义先验且数据量极小的情况下，基于几何特征从拆解的部件图像生成几何场景图的方法，用于推断装配关系。

**💡 创新点**

创新点主要在于三步分离结构推断与语义推理的架构，仅利用几何特征构建邻接矩阵，并通过几何原型增强实现对未知装配的泛化；同时在Transformer中加入局部卷积和边增强调注意力，提升对结构关系的捕捉。

**🔧 技术方法**

使用了Faster R‑CNN + Mask R‑CNN进行部件检测与特征提取，CLIP ViT生成几何特征；Transformer（自注意力、交叉注意力、Edge Graph Transformer）预测邻接矩阵；Siamese网络结合GCN进行关系、排列和装配顺序的分类；并配合focal loss、拉普拉斯正则、group loss等训练技巧。

**📊 数据集**

实验基于自制的玩具车辆装配数据集，包含火车、飞机、船三辆车的部件，评估时使用未见的汽车；每个部件约50张真实图像并通过几何原型合成400张增强图，训练样本约675k。

**📈 对比分析**

通过准确率、平均召回率和Matthews相关系数（MCC）对邻接矩阵、关系标签、排列和序列四个子模型分别进行评估；在验证集和未见车辆上邻接矩阵准确率>95%，关系标签>98%，排列>97%，序列>96%；消融实验表明几何原型增强和局部卷积对性能有显著提升。

**⚠️ 局限性**

局限性包括：①Transformer模型复杂、计算开销大，扩展到大规模部件时不易；②结构预测与语义推理分离易导致错误传播；③对视觉相似或模糊部件的识别仍存在困难；④极小数据集易过拟合；⑤未利用3D信息，限制了对更丰富场景的泛化。

---

## 329. fMRI2Face: A Full-HD fMRI-Video Dataset and Geometry-Guided Neural Decoding Framework for Dynamic Human Face Reconstruction

**arXiv ID:** 2607.22302 | [PDF](https://arxiv.org/pdf/2607.22302v1)

**作者:** Jingyang Huo `[一作]` (Fudan University), Jianfeng Feng `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了 fMRI-Face 数据集并提出 fMRI2Face 方案，从 fMRI 信号重建动态人脸视频。

**💡 创新点**

首次提供全高清、可控人脸视频与 fMRI 记录的配对数据；将脑衍生外观上下文和 3D 形状控制融合至视频扩散模型，实现身份与运动一致的高保真重建。

**🔧 技术方法**

基于 DECA 3D 形状模型、脑衍生上下文令牌、辅助潜在空间以及预训练的视频扩散变换器（Wan2.1‑T2V‑14B），并采用流匹配、LoRA 等训练策略。

**📊 数据集**

fMRI‑Face：62,856 对 1920×1080 解析度无背景数字人脸视频与 fMRI 数据；对照实验使用 MindVideo、NeuroPictor、MindEye2 等现有基线。

**📈 对比分析**

在相同训练/测试划分下与三种基线对比，fMRI2Face 在 PSNR、SSIM、LPIPS、FVD、ID‑CSIM、LMD 等指标均显著提升，尤其 PSNR +2.6 dB、SSIM +0.1、FVD 降至 82.7。

**⚠️ 局限性**

受限于仅 3 名受试者、fMRI 分辨率与延迟、以及对非背景环境的泛化能力，模型在极端表情或不同光照下仍需进一步验证。

---

## 330. Biomedical Machine Translation for Low-Resource Arabic-Script Languages via Cross-Lingual Transfer and LoRA Adapter Merging

**arXiv ID:** 2607.22300 | [PDF](https://arxiv.org/pdf/2607.22300v1)

**作者:** Abdullah Alabdullah `[一作]` (University of Edinburgh), Lifeng Han `[通讯]` (Universiteit Leiden)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了利用阿拉伯语和波斯语的领域适配 LoRA 适配器，通过跨语言迁移提升四种极低资源阿拉伯语系语言（达里语、普什图语、索拉尼库尔德语、乌尔都语）的医学机器翻译性能。

**💡 创新点**

创新点包括：1）首次系统评估 LoRA 适配器在医学领域的跨语言迁移效果；2）提出零数据 LoRA 适配器合并（tensor 级融合）实现无目标语言监督的域适配；3）发现模型反转现象，即较弱基模型在跨语言迁移上优于更强基模型；4）展示不同语言相似度对迁移效果的定量关系。

**🔧 技术方法**

使用技术包括：decoder‑only LLM（Gemma 2‑2B 与 Llama 3.2‑3B），LoRA 低秩适配器训练，few‑shot in‑context learning，最小监督微调（约 500 句），以及多种适配器合并策略（简单平均、加权平均、TIES、DARE）。

**📊 数据集**

数据集：阿拉伯语领域平行语料 PEACH；波斯语医学平行语料 Esposito；一般领域 500 句用于少量监督和 few‑shot 示例（来自 FLORES‑200）；评估集 TICO‑19（2100 句），包含六种目标语言的医学句子。

**📈 对比分析**

对比方法：零-shot、few‑shot、少量监督微调、适配器合并。性能表现：在达里语上，少量监督微调达 CHrF++≈41，零数据合并约 37；在乌尔都语上，少量监督约 28，合并仅 16；在普什图语和索拉尼库尔德语上，所有方法均低于 17，难以达到临床可用水平。总体来看，少量监督效果最佳，合并策略在与源语言相近的语言（如达里语、普什图语）上可接近监督效果。

**⚠️ 局限性**

局限性：1）普什图语和索拉尼库尔德语仍无法满足高风险临床使用；2）迁移效果受语言相似度、脚本共享等多重因素共同影响，难以单独归因；3）使用的 LLM 主要为指令调优模型，可能存在生成漂移和hallucination；4）仅在医学领域的少数语料上验证，无法保证在更广泛医学子领域的泛化；5）LoRA 合并的通用性未在更大模型或非医学域中验证。

---

## 331. Comparing and Conceptualizing Data Protection Requirements Worldwide for Privacy Regulatory Compliance

**arXiv ID:** 2607.22270 | [PDF](https://arxiv.org/pdf/2607.22270v1)

**作者:** Claudia Negri-Ribalta `[一作]`, Rene Noel `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对70名跨26个国家的数据保护法律专家的访谈以及对21条数据保护法规的系统内容分析，本文系统识别并概念化了全球范围内共同与差异化的监管数据保护要求（RDPR），并基于这些发现构建了一套以数据保护官（DPO）为中心的用户故事（DPO stories），用于支持软件开发生命周期（SDLC）和企业架构（EA）层面的合规管理。

**💡 创新点**

创新点在于：①首次从法律专家视角结合定性访谈与法规分析，形成统一的共性与差异化RDPR概念图；②将RDPR直接转化为可操作的DPO用户故事，并将其按SDLC阶段和EA层级分类；③使用AQUSA框架和角色扮演扑克游戏对用户故事质量进行验证，确保其可用性与可实施性。

**🔧 技术方法**

所用技术包括：①基于归纳与演绎相结合的定性分析方法（Deductive Qualitative Analysis, DQA）；②系统内容分析（Systematic Content Analysis, SCA）对法规文本进行编码；③AQUSA核心工具与手工评估，用于评估用户故事的语法、语义与实用性；④角色扮演扑克游戏对用户故事进行团队评审与改进。

**📊 数据集**

使用的数据集包括：70位法律专家访谈文本（覆盖26个国家，涵盖G20及部分发展中国家）；21条主要数据保护法规（包括GDPR、LGPD、APPI等）的全文；以及公开的访谈转录与法规注释存储在Zenodo公开仓库。

**📈 对比分析**

比较方法：对每条编码（如同意、删除、通知等）在不同法规中出现的频率进行统计，并与访谈中专家的认知进行对照；对DPO用户故事进行质量指标（如AQUSA标准）评估。结果显示，核心要求（同意、访问、删除、通知、数据主体权利等）在≥66%法规中一致；但在时间框架、自动化决策权利、遗忘权等方面表现出显著差异。性能方面，并非计算性能，而是通过定性指标验证了故事的完整性与可执行性。

**⚠️ 局限性**

局限性包括：①样本集中主要为G20国家，欠缺部分发展中国家和新兴经济体的法规；②访谈参与者主要为法律专家，缺乏技术实现者的视角；③DPO用户故事未在真实项目中进行实证验证，仅通过角色扮演评估；④未覆盖与数据保护相关的其他法规（如网络安全法、行业标准）。未来工作需在更大范围内进行实证测试，并完善跨域法规的时间框架与权利细化。

---

## 332. Finite-Support Structure in i.i.d.-Constrained Capacity of Finite-Memory Poisson Channels

**arXiv ID:** 2607.22234 | [PDF](https://arxiv.org/pdf/2607.22234v1)

**作者:** Renzhi Yuan `[一作]` (Beijing University of Posts and Telecommunications), Mugen Peng `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在有限记忆的Poisson通道下，满足峰值和平均光强约束的i.i.d.输入容量的最优输入分布必定具有有限支集，并给出了完整的理论证明。

**💡 创新点**

首次通过构造统一的滤波遗忘估计、对连续状态HMM输出熵率进行变分分析、引入全局可解析的KKT缺陷函数，并利用其超线性增长性排除无穷多支点，得出了容量实现输入的有限支集结构。

**🔧 技术方法**

采用随机最小化耦合的滤波稳定性分析、连续状态隐藏马尔可夫模型的熵率变分、复分析中的全纯扩展与身份定理、凸优化的KKT条件、以及对熵率的一致收敛证明。

**📊 数据集**

论文主要是理论推导，数值实验使用离散网格近似与有限块递推，参数包括峰值A、暗电流λ₀、平均功率P_avg以及两记忆taps h=(0.65,0.35)；未使用公开数据集。

**📈 对比分析**

通过有限网格仿真与理论预测对比，结果表明最优输入往往稀疏且集中在端点，数值上与仅使用端点信号的基准相近，说明理论结果与数值一致；在不同峰值、暗电流和taps配置下，容量估计呈现合理趋势。

**⚠️ 局限性**

仅适用于i.i.d.输入约束，未考虑非i.i.d.或反馈情况；只针对有限记忆的Poisson通道，未推广到更一般的噪声模型；数值验证仅在离散网格上，无法完全覆盖连续输入空间；若约束形式改变，理论证明可能失效。

---

## 333. An Insight on Evaluation Metrics Under the Imbalanced Case of Anomaly Detection

**arXiv ID:** 2607.22286 | [PDF](https://arxiv.org/pdf/2607.22286v1)

**作者:** Romain Hermary `[一作]`, Djamila Aouada `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

定义了二分类评估指标（TP、TN、FP、FN、TPR/召回率、FPR、PPV/精准率）

**💡 创新点**

无创新点，内容为基础统计定义

**🔧 技术方法**

采用数学统计公式描述指标

**📊 数据集**

未提及任何数据集

**📈 对比分析**

未进行方法比较或性能评估

**⚠️ 局限性**

仅提供指标定义，缺乏应用示例、实验结果及对比分析

---

## 334. NUMA balancing hampering performance of spiking network simulations

**arXiv ID:** 2607.22275 | [PDF](https://arxiv.org/pdf/2607.22275v1)

**作者:** Melissa Lober `[一作]` (Institute for Advanced Simulation), Markus Diesmann `[通讯]` (Institute for Advanced Simulation)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 NEST 神经网络模拟在多核 NUMA 系统上运行时，自动 NUMA 平衡开启会导致周期时间分布拉长、能耗上升，并对模拟性能产生负面影响。

**💡 创新点**

提出了时间分辨热图和周期时间与突触计数相关性分析的新可视化方法，并在 Slurm 作业调度中实现按作业开启/关闭 NUMA 平衡的用户级切换机制。

**🔧 技术方法**

使用了 NEST 3.10 的周期计时器、jemalloc 和系统内存分配器、Linux 自动 NUMA 平衡、Slurm 调度器以及 CI‑bench 自动化基准框架进行实验和分析。

**📊 数据集**

采用了多区域视觉皮层模型（MAM），包括 32 个视觉皮层区域、约 1 平方毫米面积、约三分之一突触为跨区域连接的实验数据集。

**📈 对比分析**

在 JURECA‑DC 超算上进行强缩放实验，比较开启/关闭 NUMA 平衡以及不同内存分配器的情况，发现关闭 NUMA 平衡可使实时因子（RTF）降低约 30%，能耗下降约 30%。

**⚠️ 局限性**

实验仅在特定硬件配置（AMD EPYC 7742、8 NUMA 域）和 NEST 工作负载下验证，未评估其他科学代码或不同 NUMA 设置，结果可能因网络结构或线程划分差异而变化。

---

## 335. Efficient Spatial-Spectral Feature Extraction in Hyperspectral Images via Holistic Multivariance Decomposition

**arXiv ID:** 2607.22272 | [PDF](https://arxiv.org/pdf/2607.22272v1)

**作者:** Süha Tuna `[一作]` `[通讯]` (Istanbul Technical University), Süha Tuna (Istanbul Technical University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了全景多变性分解（HMD）方法，用于从高光谱图像中提取空间-光谱特征。

**💡 创新点**

创新点在于引入多维支持矩阵，既保留全维交互，又可通过可控降维参数灵活调整子空间，突破传统Tucker/CP仅捕获单模独立变异的局限。

**🔧 技术方法**

主要技术为基于张量模态乘积的全景多变性分解框架，结合半正交支持矩阵和层级分解（0级、1级、2级）实现特征抽取。

**📊 数据集**

实验使用四个公开高光谱数据集：Indian Pines、Pavia University、Botswana 和 Dioni。

**📈 对比分析**

与 Tucker、CP 以及多种监督学习器（AdaBoost、SVM、KNN、LDA）比较，HMD 的 1级/2级在压缩率低至 r=2 时已优于传统方法，整体分类精度提升 5–10%。

**⚠️ 局限性**

局限在于需要预先构造支持矩阵，若选择不合适会影响性能；同时计算开销随着分解级别提升而显著增加。

---

## 336. Loom: Multi-Region Analysis of Spatial Transcriptomics with Local Neighborhoods and Global Trajectories

**arXiv ID:** 2607.22505 | [PDF](https://arxiv.org/pdf/2607.22505v1)

**作者:** Siyuan Zhao `[一作]` (University of Illinois at Chicago), G. Elisabeta Marai `[通讯]` (University of Illinois at Chicago)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 Loom 系统，用于多样本、多分辨率的空间转录组（ST）数据的时空整合、分析与可视化，支持 ROI 级别的预处理、降维、聚类、伪时序推断、空间轨迹分析以及邻域富集的簇排序。

**💡 创新点**

创新点包括：①将跨分辨率空间坐标注册、伪时序模拟与局部微环境分析统一到同一可视化平台；②提出基于自然植物生长的紧凑伪时序图形符号，兼顾分支结构、基因表达与邻域强度；③开发邻域富集的簇排序策略，将空间社交性量化；④实现多模态时空数据的交互式协调视图。

**🔧 技术方法**

使用了 Visium HD ST 数据、bin2cell、SPATA2、Slingshot、Scanpy、UMAP、Leiden、Squidpy、rpy2、Python/Flask 后端、React/D3/Deck.gl 前端、GPU 渲染、Kosara 之比值编码等技术。

**📊 数据集**

数据集为 Visium HD 人类皮肤与大腿伤口两组样本，分辨率分别为 2 µm、8 µm 与 16 µm，单样本约 150 GB，细胞数超过 10^5，基因数约 1.8×10^4。

**📈 对比分析**

通过专家案例评估与外部 SUS 调查（SUS 80.9）验证，预处理在 8/16 µm 下约 2 min、2 µm 约 35 min；加载与渲染 13 s；ROI 级别分析（3–18 k cells）在秒级到 2 min 内完成，显示良好交互性能。

**⚠️ 局限性**

局限性：仅在两个人类组织样本上验证；伪时序方向依赖手动指定；SPATA2 对单元数有阈值限制；高分辨率下的噪声处理仍不完备；跨样本大规模对齐与多模态集成仍有提升空间。

---

## 337. CARA: Concept-Aware Risk Attention for Interpretable Collision Anticipation

**arXiv ID:** 2607.22494 | [PDF](https://arxiv.org/pdf/2607.22494v1)

**作者:** Zhishan Tao `[一作]` (Shanghai Jiao Tong University), Yi Hong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种概念感知风险注意力（CARA）框架，用域相关风险概念驱动时空注意力，实现自动驾驶碰撞预判。

**💡 创新点**

创新点在于将从事故叙述中抽取的概念作为动态中间证据直接嵌入预测路径，既提高预判性能，又实现本质可解释性。

**🔧 技术方法**

技术手段包括 CLIP 视觉语言对齐、指数移动平均平滑、概念风险评估、概念引导空间与时间注意力、GRU 时序融合以及多任务学习损失。

**📊 数据集**

使用的实验数据集包括 DAD、CCD、A3D 交通事故预判基准以及 804 条加州 DMV 自动驾驶事故报告。

**📈 对比分析**

与 DSA、UString、DSTA、GSC、CRASH 等方法比较，CARA 在 AP、mTTA 和 R80 等指标上均优于基线，尤其在 DAD 数据集提升显著。

**⚠️ 局限性**

局限性在于概念库的质量和覆盖范围受限于文本来源，若缺乏足够多样化的概念，模型可能无法覆盖所有风险情境。

---

## 338. Legal Nugget Extraction for Granular Retrieval over Long Jurisprudential Texts

**arXiv ID:** 2607.22479 | [PDF](https://arxiv.org/pdf/2607.22479v1)

**作者:** Lucas Pereira `[一作]` (Tribunal de Contas da Uniao), Jayr Pereira `[通讯]` (Universidade Federal do Cariri)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种将判例文档拆分为短小、独立的法律论点（nuggets）并以此为索引单元进行 dense retrieval 的完整管线，并将检索到的 nuggets 聚合回原文档进行排名。

**💡 创新点**

创新点在于：①提出将法律论点作为检索单位，避免因全文向量稀释导致关键信息被淹没；②构建从生成、索引到检索再聚合的 end‑to‑end pipeline；③系统性验证该方法在不同法律检索任务中的有效性与局限性。

**🔧 技术方法**

技术包括：使用大语言模型（如 Llama2、Qwen3）进行 nugget 生成；用同一密集检索模型（JUA‑adapted 或 Qwen3 embedding）对 nuggets 与查询做向量嵌入；在 FAISS 上做 L2‑norm 向量检索；采用最高得分聚合（first‑evidence）将 nugget 结果映射回文档。

**📊 数据集**

使用了四个巴西法律检索基准：JUA‑Juris、JurisTCU、NormasTCU、BR‑TaxQA，分别涵盖判例、规范文本和税务问答。

**📈 对比分析**

通过在同一 embedding 模型下对比全文索引与 nugget 索引的检索效果，使用 NDCG@10、MAP@10、MRR@10 三个指标。结果显示：在 JUA‑Juris 与 JurisTCU 上 nugget 检索将 NDCG@10 近乎翻倍，MAP 与 MRR 同样提升；但在 NormasTCU 与 BR‑TaxQA 上表现略逊，且在某些开放模型 ablation 中 nugget 仍能提升 general embedding 的性能。

**⚠️ 局限性**

局限性包括：①生成 nuggets 的成本高、依赖收费或大模型；②聚合策略过于简单，未尝试更细粒度或联合检索方案；③评估仅基于标准指标，缺乏实务层面的可解释性和质量评估；④实验仅覆盖巴西法域与有限模型，结果的可推广性仍待验证。

---

## 339. MineValiCoder: Reliable Code Generation with Test Case Quality Mining and Bipartite Graph-Based Mutual Validation

**arXiv ID:** 2607.22471 | [PDF](https://arxiv.org/pdf/2607.22471v1)

**作者:** Zhen Zhao `[一作]` (Chinese Academy of Sciences), Bo Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 MineValiCoder 的全自动 TDD 框架，集成了测试用例质量挖掘、并行代码优化与基于双向图的代码-测试互验证，以实现从自然语言需求到高质量代码的闭环生成。

**💡 创新点**

创新点：① 通过自验证矿工（Self‑Validation Mining）过滤 LLM 生成的错误测试用例，提升反馈可靠性；② 采用并行 TDD 与单调性约束的迭代优化，产生多样化且稳健的代码候选；③ 以二分图建模代码与测试的相互验证，使用动态分数传播与一致性过滤，实现对候选代码的可靠选择。

**🔧 技术方法**

技术手段：LLM（GPT‑4、Qwen3‑4B 等）进行测试生成与代码生成；自一致性检验（Self‑Consistency）与“Plan‑then‑Code”策略；迭代 TDD 过程中的单调性约束；双向图分数传播算法与鲁棒过滤；超参数 N 控制并行分支与测试样本量。

**📊 数据集**

使用的评测数据集包括：HumanEval、MBPP 及其增强版 HumanEval‑ET、MBPP‑ET；APPS（Intro/Inter/Comp 三级）和 LiveCodeBench（实时竞赛题目）。

**📈 对比分析**

与传统推理增强方法（Direct、CoT、Self‑Planning、Reflexion）及代码优化框架（CodeT、MapCoder）对比；在 GPT‑4、Qwen2.5‑Coder‑7B、Qwen3‑4B、Llama3.1‑8B 四大模型上实验。MineValiCoder 在 HumanEval 上 96.34% Pass@1、MBPP 87.40%、APPS 平均 64.00%、LiveCodeBench 51.33%，均突破现有最优，接近 PyCapsule 的基准上限。

**⚠️ 局限性**

局限性：仍依赖 LLM 生成质量，难以彻底消除所有随机噪声；参数 N 需要调优，过大会增加算力成本；对多文件或更复杂项目的适用性尚未验证；在 Llama3.1‑8B 等小模型上提升空间有限；鲁棒过滤虽能降低过拟合，但对极端噪声的抵抗仍受限。

---

## 340. On the Identifiability of Controlled World Models

**arXiv ID:** 2607.22430 | [PDF](https://arxiv.org/pdf/2607.22430v1)

**作者:** Xiangteng Zhang `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

通过理论分析和合成实验验证了在非线性观测下，联合嵌入预测架构（JEPA）在动作条件下的世界模型可识别性。

**💡 创新点**

提出联合可识别性理论，给出谱分离与条件动作激励两个可识别边界，并证明在满足两者时可唯一确定潜在状态和控制转移。

**🔧 技术方法**

利用线性高斯潜在动力学、可逆观测、JEPA目标、Hermite 展开、谱分离与动作激励分析，以及构造性扰动实现对偶误差放大证明。

**📊 数据集**

使用合成实验：多种非线性观测映射和状态相关高斯行为策略，生成线性高斯潜在状态。

**📈 对比分析**

通过对比表示误差、转移误差、对偶误差放大和目标规划终点误差，验证了谱分离提升表示准确性、动作激励提升转移识别和规划性能，实验结果表明在充分探索时规划误差几乎消失。

**⚠️ 局限性**

局限在于仅考虑可逆观测、线性高斯潜在动力学、严格的标准高斯表示约束，只识别条件均值；缺乏有限样本与非线性/部分可观测情境下的理论与实证验证。

---

## 341. Vibe Coding: An Experiment with Test-Driven Development

**arXiv ID:** 2607.22406 | [PDF](https://arxiv.org/pdf/2607.22406v1)

**作者:** Moritz Mock `[一作]` (Free University of Bozen-Bolzano), Barbara Russo `[通讯]` (Free University of Bozen-Bolzano)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在测试驱动开发环境下，设计并实现了四种人-CLLM交互模型（solo、collaborative、fully automated、agentic），并评估其代码与测试质量与开发效率。

**💡 创新点**

创新点在于将vibe coding与agentic coding融入TDD流程，构造可复现的实验框架，并系统比较人机协作与全自动化生成对软件质量的影响。

**🔧 技术方法**

使用ChatGPT（3.5/5）、MetaGPT X、Claude Sonnet 4等大型语言模型，以及基于Python的自动化脚本和Prompt工程技术。

**📊 数据集**

数据集为两道自定义Python编码任务（TextFormatter），并采用专业开发者手工编写的基准测试套件。

**📈 对比分析**

通过统计检验（Kruskal‑Wallis、Mann‑Whitney U+Holm–Bonferroni）比较四模型在功能正确率、代码复杂度、测试覆盖率、迭代次数和时间上的表现；结果显示agentic模型最快且功能正确率最高，但测试覆盖率和结构复杂度较低；collaborative模型在测试质量上最优。

**⚠️ 局限性**

主要局限在于实验样本规模有限、仅针对简单任务、不同评估设置（人机模型与全自动化模型）导致可比性受限，以及未考虑更复杂语言模型或更大规模任务的泛化能力。

---

## 342. The Prompt Is Not the Query: How Request State Evolves Across Multi-Turn AI Conversations

**arXiv ID:** 2607.22392 | [PDF](https://arxiv.org/pdf/2607.22392v1)

**作者:** Benjamin Tannenbaum `[一作]` `[通讯]` (Aiso), Benjamin Tannenbaum (Aiso)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了多轮对话中最终提示词（prompt）并非完整的查询或总结，而是对会话状态的增量更新；

**💡 创新点**

创新点在于将会话中的“请求状态”视为可观测的、以对话为条件的状态，并量化该状态在单个提示词与整个会话中的分布与缺失；

**🔧 技术方法**

采用透明的文本规则提取九类请求维度，计算词汇覆盖率，并对长度匹配的随机模型做对比；

**📊 数据集**

使用两大真实对话语料：1）商业治理语料（共1,749条可用会话，含多轮英文会话）；2）公开PRISM数据集（1,389位参与者，2,308+2,279+2,876多轮会话）；

**📈 对比分析**

结果显示，最终提示词平均仅覆盖约35%独特词汇，且约有一半会话存在仅在历史中出现的维度，完整维度覆盖率仅约25%；相比随机长度匹配模型差距不大，表明缺失主要因长度限制；

**⚠️ 局限性**

局限包括：规则只能捕捉显式词汇，无法辨别未表明的偏好；词汇覆盖受句子长度影响；对话深度是内生变量；语料来源非随机；未衡量历史对答案的因果影响。

---

## 343. Interpretable EEG biomarkers with bag-of-waves: Spatial and temporal waveform dictionaries for low-data regimes

**arXiv ID:** 2607.22508 | [PDF](https://arxiv.org/pdf/2607.22508v1)

**作者:** Athanasios Papastathopoulos-Katsaros `[一作]` (Baylor College of Medicine), Zhandong Liu `[通讯]` (Baylor College of Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种可解释的“bag-of-waves”框架，用无标签的shift‑invariant k‑means学习少量EEG波形字典，将连续EEG转化为离散词元序列，再利用词频与词元间的n‑gram统计进行分类或聚类。

**💡 创新点**

创新点在于：①把波形字典扩展到区域和跨通道的空间atom；②加入atom‑to‑atom n‑gram转移特征捕获时间结构；③在低数据、不同通道配置下保持极低参数、易解释且与深度模型性能相当。

**🔧 技术方法**

使用shift‑invariant k‑means学习字典，TF‑IDF加权词频与n‑gram计数，随机森林（或逻辑回归）做下游分类；还做了k‑means聚类、隐藏马尔可夫模型比较。

**📊 数据集**

三个数据集：1）16只小鼠单通道EEG用于基因型聚类；2）ds004504人类休息态EEG用于阿尔茨海默与前额叶痴呆诊断；3）Temple University EEG Event (TUEV)用于6类临床事件分类。

**📈 对比分析**

与公开的深度学习/基础模型（CNN、Transformer、BIOT、LaBraM等）进行对比。性能上：小鼠聚类ARI=1；阿尔茨海默诊断macro F1≈0.86（比传统特征高≈9%）；TUEV Cohen κ≈0.44、加权F1≈0.72，接近或略低于最强基础模型。

**⚠️ 局限性**

局限包括：1）字典为平滑平均，可能失去细节；2）窗口内只能匹配单一atom，重叠事件无法分离；3）对稀有事件和极短记录的鲁棒性不足；4）缺乏跨数据集迁移验证，超参数需针对任务调优。

---

## 344. \k{appa}-LoRA: Condition Numbers Reveal Which LoRA Matrices Worth Updating

**arXiv ID:** 2607.22489 | [PDF](https://arxiv.org/pdf/2607.22489v1)

**作者:** Jianghui Wang `[一作]` (King Abdullah University of Science and Technology), Yaqi Xie `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LoRA在Transformer中适配敏感性的差异，提出仅对预训练权重条件数最高的一半矩阵进行LoRA低秩更新，从而显著降低计算量和存储开销。

**💡 创新点**

创新点在于首次将预训练权重的条件数作为无监督、一次性特征来筛选LoRA矩阵，并从理论上证明高条件数矩阵对低秩更新更具“可塑性”，实验验证了该策略能保持甚至提升性能。

**🔧 技术方法**

使用LoRA低秩适配、奇异值分解（SVD）计算条件数、组内排名选择、AdamW优化器、余弦退火学习率调度，并在多种Transformer架构（LLaMA2‑7B、Mistral‑7B、Gemma‑7B、DeBERTa‑v3‑base）上实现。

**📊 数据集**

主要使用NLG任务集MetaMathQA（GSM8K、MATH）、CodeFeedback（HumanEval、MBPP）、WizardLM‑Evol‑Instruct（MT‑Bench）以及NLP理解任务集GLUE。

**📈 对比分析**

与标准LoRA做对比，采用50% LoRA矩阵的稀疏策略后，平均训练时间缩短约16%（NLG任务）或22%（GLUE），且在大多数子任务上保持或略高的性能。

**⚠️ 局限性**

局限性包括：条件数在训练期间被视为静态且一次性估计，未考虑动态重评估；固定比例α未针对不同层深度或任务调整；理论证明仅在特定假设下最优，未给出全局最优性保证。

---

## 345. TRACE-ROUTER: Task-Consistent and Adaptive Online Routing for Agentic AI

**arXiv ID:** 2607.22465 | [PDF](https://arxiv.org/pdf/2607.22465v1)

**作者:** Ritik Raj `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种任务级别的LLM路由框架，先为每个完整任务在进入时一次性选择模型，并在任务结束时利用整体准确率与延迟的组合奖励来更新路由策略，从而实现不同成本‑质量折衷模型池的自适应分配。

**💡 创新点**

创新点在于：
• 将路由粒度从单个请求提升到完整任务，避免跨模型状态失效；
• 采用基于任务上下文的离线无模型分类器对任务进行粗粒度分组；
• 对每个上下文使用独立的上下文UCB bandit，实现在线学习并利用延迟任务级反馈；
• 通过可调参数α将准确率与延迟的权衡统一为单一奖励，支持多种部署策略。

**🔧 技术方法**

主要技术包括：
• 任务一致性路由（sticky‑key 绑定）；
• 上下文感知的多臂bandit（UCB）实现离线无模型前置特征分类；
• 延迟信用分配机制，将任务终止时的准确率与延迟映射为即时奖励；
• 任务级奖励函数 r^(α) = (1‑α)·accuracy – α·normalized_latency。

**📊 数据集**

使用的数据集：
• τ^2‑Bench（零售与电信两域，各114个任务）
• Terminal‑Bench（48个任务）
• LiveCodeBench（300个任务，难度加权 Pass@1）。
每个基准都配备一对小/大模型，例如 Qwen3.5‑4B / 9B 或 Qwen3.5‑9B / 27B‑FP8。

**📈 对比分析**

比较方法：
• 对比两端单模型、语义路由、复杂度路由、以及延迟匹配的随机混合；
• 通过绘制准确率‑延迟 Pareto 前沿展示不同α取值下的折衷点；
• 在τ^2‑Bench上，框架在所有α下均占据前沿并比延迟匹配插值高7–8个百分点；
• 在Terminal‑Bench上，以α=0.75时实现46.8%任务成功率、172s延迟，比仅使用大模型低36%延迟并提升7.1个百分点；
• 该方法在所有基准上均取得非支配（Pareto）性能，且比最强单模型低约30%延迟。

**⚠️ 局限性**

局限性：
• 依赖于粗粒度的正则表达式上下文分类，若任务难度划分不精确可能导致探索不充分；
• 对于模型数目较多的情况，冷启动探索成本仍可能显著，且需要更多上下文层级才能保持统计显著性；
• 延迟奖励仅在任务结束时可观测，若任务时间跨度非常大，学习收敛会被拖慢；
• 框架假设每个任务只能使用一次模型，无法处理在执行过程中需要切换模型的需求；
• 在极端负载或高并发环境下，任务标识符和状态表的管理可能成为瓶颈。

---

## 346. Beyond Perspectives: A Trio-Ethnography of Interpretation Evolution in LLM-Supported Programming Education

**arXiv ID:** 2607.22463 | [PDF](https://arxiv.org/pdf/2607.22463v1)

**作者:** Jennie Ren `[一作]` (Mercer University), Kyrie Zhixuan Zhou `[通讯]` (University of Texas at San Antonio)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究采用三人民族志（trio-ethnography）方法，邀请两位计算机教育者与一名本科生进行三轮对话，探讨并反思教育者对学生使用大型语言模型（LLM）进行编程学习的解释与教学信念。

**💡 创新点**

创新点在于将三人民族志从单纯比较视角转化为“对话驱动的解释演化”模式，利用学生的真实体验作为证据逐步修正教育者的假设，进而重塑教学观念；并首次将此方法应用于编程教育与生成式AI的交叉研究。

**🔧 技术方法**

使用的技术主要是定性研究方法：半结构化访谈、录音转写、反思性对话分析和三阶段主题编码；没有涉及机器学习或实验算法。

**📊 数据集**

数据集由三轮对话的音频记录转写文本组成，共计约3小时的对话，涉及两位教师的初始解释、学生的使用经历以及后续的反思重构。

**📈 对比分析**

研究并非通过数值性能比较，而是通过比较教育者在三轮对话前后的解释差异，观察其教学信念的演进和对AI使用的阐释方式的改变；结果表明教育者从对AI产生污名到强调透明度和明确指导，从“答案提供者”转向“学习伙伴/导师”，并识别出隐藏的学习过程。

**⚠️ 局限性**

局限性包括：只访谈了一名高水平自我驱动的学生，样本规模极小；研究仅在一个机构、一个课程背景下进行，缺乏跨校、跨课程的验证；并且结果更多是解释性洞察，未提供可量化的评估指标。

---

## 347. Deformable Triangle Splatting: Flexible Primitives for Real-Time Radiance Field Rendering

**arXiv ID:** 2607.22446 | [PDF](https://arxiv.org/pdf/2607.22446v1)

**作者:** Oriol Jiménez-Ayguadé `[一作]` (Institut de Robòtica i Informàtica Industrial), Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种可变形三角形投影渲染方法DETRIS，用于实时辐射场渲染。

**💡 创新点**

创新点在于给每条三角形边添加可学习的控制点，允许生成非凸形状；在重心坐标空间进行视角无关的位移和距离计算；引入缠绕数判定、平滑最小化距离场和幂律窗口函数实现完全可微的渲染管线。

**🔧 技术方法**

使用可微渲染、重心坐标投影、缠绕数测试、平滑最小化距离、窗口函数、CUDA实现、稀疏SfM初始化、控制点学习率延迟、正则化等技术。

**📊 数据集**

在MIP-NeRF 360和Tanks & Temples（室内外）两大公开数据集上进行实验。

**📈 对比分析**

与隐式（NeRF、Mip-NeRF、Instant‑NGP等）、体素/高斯（3DGS、3DGS‑MCMC、DBS、3DCS）以及非体素（BBSplat、2DGS、Triangle Splatting）方法对比，DETRIS在LPIPS、PSNR、SSIM、FPS等指标上取得了最优或竞争性表现，尤其在LPIPS上显著领先。

**⚠️ 局限性**

局限包括：每个可变形三角形需要更多的边段距离计算导致FPS下降；训练时间约为TS的两倍；对极端细小细节的建模仍有限；需手动调节控制点数K和学习率延迟；在透明/薄结构等更复杂场景上尚未成熟。

---

## 348. Hyperball May Not Be a Free Lunch

**arXiv ID:** 2607.22444 | [PDF](https://arxiv.org/pdf/2607.22444v1)

**作者:** Yihao Xiao `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析 Hyperball 优化器对有效学习率的影响，并通过角度有效学习率、径向更新分析以及学习率对齐实验，阐释其慢-快收敛行为；

**💡 创新点**

提出角度有效学习率概念、径向更新对角度步幅影响的量化分析，以及状态依赖学习率对齐方法，解释 Hyperball 作为隐式学习率调度器的本质；

**🔧 技术方法**

利用几何角度分解、有效学习率计算、数值敏感性分析、学习率对齐实验以及语言模型预训练的实证验证；

**📊 数据集**

在密集语言模型预训练任务（如 Pile/C4 等大规模文本数据集）上进行实验；

**📈 对比分析**

将 MuonWD、MuonH 与不同学习率调度（线性、指数、多项式、特定的 Zhanpeng Zhou/自定义调度）进行对比，发现通过学习率对齐后两者收敛轨迹可相互转换；在预训练中，改进的多项式调度能在一定程度上加速 MuonH，但并未在整个训练周期内持续优于 MuonWD；

**⚠️ 局限性**

理论分析仅限于有效学习率的数值敏感性，未给出完整的理论解释；实验仅覆盖密集语言模型预训练，未验证稀疏 MoE 结构；对比的优化器范围有限；依赖于规模不变假设，实际训练中该假设可能被违反。

---

## 349. Unboxing Diffusion Models for the Arts: Interactive Model Bending and Practice-Based Explainability

**arXiv ID:** 2607.22428 | [PDF](https://arxiv.org/pdf/2607.22428v1)

**作者:** Ahmed M. Abuzuraiq `[一作]` (Simon Fraser University), Philippe Pasquier `[通讯]` (Simon Fraser University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个集成于 ComfyUI 的模型弯曲（model bending）工具箱和交互式界面，使艺术家能够可视化并直接干预 Stable Diffusion 1.5 的内部组件；同时对不同层、不同参数、不同时间步进行系统实验，评估弯曲对生成结果的影响。

**💡 创新点**

将可解释性（XAI）从传统的后置分析转向基于创作的“动手”方法；首次提供可直接操作模型内部结构的工具，并结合量化指标构建“作弊表”，帮助艺术家在实践中培养对大规模扩散模型的直觉和理解。

**🔧 技术方法**

核心技术包括：Stable Diffusion 1.5（UNet+VAE+CLIP+Scheduler）架构；在 ComfyUI 中实现自定义节点进行钩子插入和弯曲操作；弯曲操作包括乘法、旋转、噪声添加等 PyTorch 模块；使用 JSON 描述多层、多步的弯曲配置；利用 NNSight 等框架实现模型内部状态拦截。

**📊 数据集**

使用 RealisticVisionV51_v51VAE 预训练模型（对 SD1.5 的 fine‑tune），主要测试文本提示包括 "Analog style portrait of a person"、"A red apple on a white background"、"Mountain landscape at sunset" 等；随机种子设置 42、123、456、789、786；未自行训练大规模数据集，仅使用公开的预训练权重。

**📈 对比分析**

对弯曲结果进行定性展示（图片对比）和定量评估（潜在向量余弦距离）；统计不同层类型、区域、时间步的平均距离，表明输入区和卷积层的影响最大；未与其他模型或算法进行性能对比，主要关注弯曲对输出的差异程度。整体实现交互速度受迭代推理限制，未给出具体运行时间。

**⚠️ 局限性**

仅针对 SD1.5 的 UNet 结构；结果可能不适用于其他扩散模型或版本；定量指标以潜在余弦距离为主，未涵盖感知、语义或审美维度；缺乏人类主观评估与长期艺术家实验；界面功能依赖于 ComfyUI，扩展性受限。

---

## 350. Universal BCI Personalization: One API for Frozen EEG Trunks and Foundation Models

**arXiv ID:** 2607.22397 | [PDF](https://arxiv.org/pdf/2607.22397v1)

**作者:** Sergey Musienko `[一作]` `[通讯]`, Sergey Musienko

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一个可在多种冻结EEG编码器上统一使用的Personalizer API，包含贝叶斯头和可选仿射层，实现无模型细调的个性化。

**💡 创新点**

提供了trunk-agnostic的个性化合同，避免每个架构都需单独fine-tune/PEFT，且在多种架构下统一评估。

**🔧 技术方法**

使用贝叶斯LDA/QDA 头、可选仿射映射、嵌入层的聚类与在线更新，结合合成通道混合的迁移压力。

**📊 数据集**

在四个公开MI数据集（BNCI2014-004、Zhou2016、Kumar2024、Shin2017-A）以及Hub REVE基础模型上进行评估。

**📈 对比分析**

通过与全量fine-tune、线性探针、LoRA等基准在同一冻结嵌入上比较精度与适配时间，发现贝叶斯头在多数细化情形下恢复约65‑90%精度且适配时间仅为FT的1/10左右。

**⚠️ 局限性**

评估仅覆盖合成通道混合压力，未测试真实电极位移或SNR变化，且多模型间的头优先级仍需按数据集细调。

---

## 351. SceneActBench: Can Agents Act on the 3D Scenes They See?

**arXiv ID:** 2607.22393 | [PDF](https://arxiv.org/pdf/2607.22393v1)

**作者:** Yifei Zhao `[一作]` (Tencent Hunyuan), Wenxi Zhu `[通讯]` (Tencent Hunyuan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SceneActBench，一个统一的可执行基准，评估视觉语言模型（VLM）在完整多物体3D场景中的行动能力。

**💡 创新点**

创新点在于：1) 将视觉输入转化为可执行的3D操作并以隐藏几何真值评分；2) 设计统一的agent–environment循环，支持五种不同能力（空间定位、视角推理、关节运动、形状重建、动态运动）在同一框架下评测；3) 提供详细的诊断指标，揭示失败阶段而非仅给出聚合分数。

**🔧 技术方法**

技术上使用头less Blender通过Model Context Protocol（MCP）接口执行Python脚本和渲染；通过ADD‑S、PE/AE、MPE、F@5%、MME/LE等几何度量进行评分；并将原始度量映射到0–100分进行整体汇总。

**📊 数据集**

数据集来源于3个公开资源：3D‑FRONT（100房间）、S2O Articulated Containers Dataset（100可动物体）以及10个基于Kenney低多边形资产的动态场景，总计210个源实例，产生520个任务案例。

**📈 对比分析**

对11种专有VLM配置（Claude、GPT、Gemini、Qwen、MiniMax等）进行统一跑评，每个配置在固定步骤预算下完成五项任务；整体得分在38.6–50.2之间，Doubao最高50.2，Claude Opus 48.9，GPT 5.4 Medium 48.7。不同配置在各任务上表现不一，表明没有单一配置在所有能力上均优。

**⚠️ 局限性**

局限性包括：仅评估专有模型且每个配置-案例对只跑一次，无法反映重复实验方差；Dynamic任务仅包含10个场景，难以全面评估动态场景性能；整体得分受归一化常数设计影响，建议同时查看原始度量。

---

## 352. HiKV: Hierarchical Importance-Aware KV Cache with Hardware Acceleration for LLM Decoding

**arXiv ID:** 2607.22389 | [PDF](https://arxiv.org/pdf/2607.22389v1)

**作者:** Chao Fang `[一作]` (KU Leuven), Marian Verhelst `[通讯]` (KU Leuven)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于层级重要性感知的KV缓存压缩与硬件加速框架HiKV，显著减少LLM解码时的KV缓存访问量与能耗。

**💡 创新点**

创新点在于：1）将KV缓存重要性分层为token级和元素级两重压缩；2）设计可重构重要性排序器RIS，实现两阶段排序与堆维护在同一硬件电路；3）实现低面积低功耗的硬件协同，面积仅占总面积的8%。

**🔧 技术方法**

采用了动态重要性累计、堆排序、分块局部排序（chunk-based sorting）和自定义的FP16 MAC/FP32 FPU算子，配合RIS进行硬件加速。

**📊 数据集**

在四种LLM（mistral-7b-instruct-v0.2、llama3-8b-instruct、longchat-7b-v1.5-32k、qwen2.5-0.5b-instruct）上，并使用LongBench benchmark的10个任务进行评估。

**📈 对比分析**

与传统全KV、Token-Picker、H2O、S-LLM等方法对比，HiKV在1%精度误差约束下，实现外部内存访问减少7.17×，推理加速平均5.70×，能耗降低80–90%。

**⚠️ 局限性**

局限性：对硬件平台的依赖较强，需在定制ASIC或FPGA上实现；在极大批量场景下，元素级压缩仍受限于DRAM行激活延迟；未针对动态多语言或多任务环境的自适应预算分配进行深入探讨。

---

## 353. Agentic Root Cause Analysis through Evidence-Grounded Reasoning

**arXiv ID:** 2607.22385 | [PDF](https://arxiv.org/pdf/2607.22385v1)

**作者:** Amaury Wei `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AgentRCA，一种零样本、基于数字孪生与工具增强的大语言模型的根因分析框架。

**💡 创新点**

创新点在于将传统专家推理与无监督模型动态融合，利用正常运行数据构建数字孪生，Agent 通过工具调用、证据聚合和假设表迭代推理，实现不依赖故障标签的证据驱动诊断。

**🔧 技术方法**

使用数字孪生（自编码器+条件统计基准）、工具增强 LLM（如 Qwen3‑30B‑A3B‑Thinking‑2507）、结构化提示与假设表推理、检索增强生成技术。

**📊 数据集**

在 PRONTO 多相流设施（17 维传感器、4 种人工诱发故障）和 Tennessee Eastman Process（52 维、10 种物理故障）两个工业基准上进行实验。

**📈 对比分析**

与全监督基线（LightGBM、AutoEncoder+Classifier、MiniRocket）对比，PRONTO 上 AgentRCA Top‑1 87.6% 与 99% 传统基线相近，Top‑2 96.8%；TEP 上零样本 Top‑1 40.0% 仍优于无监督方法，虽然低于监督基线，但在少样本/无标签环境下表现竞争。

**⚠️ 局限性**

局限包括：需预先覆盖所有正常工况，无法处理新工况或系统漂移；工具误差或传感器失效会直接影响诊断；目前仅实现根因识别，缺乏纠错或控制建议。

---

## 354. Dynamic domination and independence in sparse graphs

**arXiv ID:** 2607.22384 | [PDF](https://arxiv.org/pdf/2607.22384v1)

**作者:** Bartłomiej Bosek `[一作]` (Jagiellonian University), Anna Zych-Pawlewicz `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了完全动态的数据结构，能够在保持图始终属于给定的可扩张（bounded‑expansion）图类的前提下，持续维护距离‑r 支配集和距离‑r 独立集的存在性，并在支配集存在时能够输出一个实例；同时给出了 r = 1 时仅依赖图的退化度（degeneracy）即可实现的更简单方案，并提供了支配集的常数因子近似算法。

**💡 创新点**

创新点在于首次给出针对可扩张图类的距离‑r 支配集/独立集问题的多项式对数级更新时间动态数据结构，突破了以往仅在静态或特定问题上的固定参数算法；通过将进展探索（progressive exploration）框架与动态同态计数相结合，获得了能够输出示例的随机化数据结构；此外，针对 r = 1 的退化度假设实现了更低的更新时间，并给出了可维护的近似解。

**🔧 技术方法**

核心技术包括：
- 进展探索（semi‑ladder / ladder）算法，利用半梯子/梯子索引保证在可扩张图类中迭代次数受限。
- fraternal augmentation（层级增广）与路径短路（shortcut）技术，用于将距离查询转化为固定大小的同态计数问题。
- Inclusion‑Exclusion 与 fingerprint retrieval 技术，将计数结果转换为可检索的示例。
- 动态同态计数与子图同构计数的高效实现（继承自 Dvořák‑Tůma 等人）。
- 复杂度分析结合了常数因子与对数因子，得到 O(log^c n · log^{1/ε} n) 的更新/查询时间。

**📊 数据集**

论文属于理论计算机科学范畴，无需使用实验数据集；所有结果均为上界证明和算法构造，基于可扩张图类的性质（如退化度、拥塞深度等）。

**📈 对比分析**

结果的性能指标主要是理论复杂度：
- 对于固定 r、k 与 ε，更新/查询时间为 O(log^c n · log^{1/ε} n)；
- 初始化时间与空间均为 O(n log n · log^{1/ε} n)。
- 在退化度 d 的情形下，r = 1 的更新时间降至 2^{k^d} · log^3 n · log^{1/ε} n。
- 近似算法的误差因子为 (4d+1)^2。相比先前仅在静态场景下的 FPT 结果，本文显著提升了动态性能。

**⚠️ 局限性**

局限性：
- 结构为随机化，错误概率可控但不为零；
- 仅适用于可扩张图类（包含多种稀疏图但不包括一般图）；
- 对 r > 1 的近似算法仍未给出；
- 复杂度中包含 r、k 的指数因子，导致在实际大参数时可能不具备可行性。

---

## 355. Susceptible Reservoir Architectures for Regime-Conditional Volatility Forecasting

**arXiv ID:** 2607.22491 | [PDF](https://arxiv.org/pdf/2607.22491v1)

**作者:** Aliaksei Kaliutau `[一作]` `[通讯]` (Monodromy), Aliaksei Kaliutau (Monodromy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于“易感性”对比与市场状态条件化的波动率预测架构SUSA，包含复杂值经典Reservoir和量子Qiskit实现。

**💡 创新点**

创新点在于利用结构/状态易感性对比（开放链 vs 周期链）以及阶段条件专家，构造可解释且可与经典持久性模型相结合的残差修正。

**🔧 技术方法**

采用复杂值开放/周期Reservoir、复数递归网络、量子开系统QRC、阶段条件MoE读取层、AR‑Ridge基线、QLIKE损失和QLIKE正则化。

**📊 数据集**

使用16只美国股票和ETF的日度收盘价、成交量等构造的日度波动率序列，分成三组时序折叠，覆盖2006‑2026年。

**📈 对比分析**

与AR‑Ridge、GARCH(1,1)、HAR/HARQ以及ESN进行基准对比，结果显示SUSA在大多数资产上优于AR，在8/16资产上优于GARCH，且与HARQ组合时可进一步提升QLIKE约0.0116，显著。

**⚠️ 局限性**

局限在于性能呈跨市场异质性，未能在所有资产上显著优于传统模型，量子实现尚未显示优势，且需要更多实证和结构消融验证。

---

## 356. A Preliminary Search for Evidence on Government Software Engineering Practices: Results from Three Rapid Reviews

**arXiv ID:** 2607.22485 | [PDF](https://arxiv.org/pdf/2607.22485v1)

**作者:** Sebastián Pizard `[一作]` (Universidad de la República), Andrea Delgado `[通讯]` (Universidad de la República)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过三份快速综述，检索2024年顶级软件工程期刊与会议、南美地区会议以及EDOS通讯，共筛选984篇论文，仅识别出4篇涉及政府机构的软件工程实践研究。

**💡 创新点**

发现政府软件工程研究在主流SE期刊中可见度极低，主要以短篇经验报告形式出现，并提出建立专门发表渠道、鼓励政府与学术合作、改进灰色文献评估的创新做法。

**🔧 技术方法**

采用手工检索与快速综述方法，并利用ChatGPT辅助对已筛选论文进行主题与方法分类，结合专家人工核验。

**📊 数据集**

使用的“数据集”为2024年全年的SE相关出版物与EDOS通讯内容，总计984篇被筛选，最终4篇被纳入分析；此外还对部分政府数字化机构网站发布的灰色文献进行审计。

**📈 对比分析**

通过PRISMA流程图与定性合成评估，发现政府相关研究在国际顶级期刊中几乎无出现，呈现显著低可见度；相比之下在区域会议和EDOS中虽有出现但数量极少，说明现有证据的覆盖面和深度均不足。

**⚠️ 局限性**

研究局限在于仅覆盖单一年份、手工检索、未进行质量评估，且研究者对快速综述经验有限，结果只能视为探索性发现，未能全面描述政府软件工程实践的整体证据库。

---

## 357. Beyond Negative-Ridge Endpoints: Mixed-Sign Spectral Regularization via Negative-Shifted Gradient Descent

**arXiv ID:** 2607.22474 | [PDF](https://arxiv.org/pdf/2607.22474v1)

**作者:** Peng Zhao `[一作]` `[通讯]` (University of Delaware), Peng Zhao (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并分析了负偏移梯度下降（NS‑GD）方法，用于在过参数化线性回归中实现混合符号谱正则化，解决负岭端点在极值附近受限的不足。

**💡 创新点**

创新点在于引入可移除极点的谱滤波器，利用有限时间的负偏移动态实现头部方向的反缩放而尾部保持压缩，从而突破传统负岭端点的马尔科夫宽度壁垒，并通过局部非收缩的 Duhamel 积分精确控制偏移路径。

**🔧 技术方法**

使用了谱滤波器理论、梯度流解析、Duhamel 公式、有效秩与尾部平方谱的随机矩阵分析，以及离散梯度下降与连续流的匹配技术。

**📊 数据集**

实验采用合成高维高过参数化高斯设计，主要包括带有脉冲+平坦尾部（Spiked+Flat）协方差的稀疏信号、源条件信号以及无间隙幂律谱，训练集规模为 100，测试/验证集规模为 500。

**📈 对比分析**

与传统正岭、岭梯度下降以及负岭端点等基线相比，NS‑GD 在所有实验设置中实现了更低的预测风险；在理论对齐的脉冲+平坦模型中，风险提升可达 4‑10 倍；在鲁棒性测试中也保持显著优势，平均 RMSE 降低约 0.08。

**⚠️ 局限性**

主要限制包括仅在具有头尾分离且尾部有效秩高的“有间隙”高斯模型下证明收敛；对无间隙谱、非高斯设计以及核岭扩展尚未给出完整理论；并且实验仅使用了合成数据，未验证在真实任务中的效果。

---

## 358. Phylogenetic signal in marine mammal and bird vocalizations captured by audio foundation models: the limited benefit of domain-specific pretraining

**arXiv ID:** 2607.22458 | [PDF](https://arxiv.org/pdf/2607.22458v1)

**作者:** Víctor Rincón Yepes `[一作]` `[通讯]`, Víctor Rincón Yepes

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对海洋哺乳动物（32种）和鸟类（20种）的录音进行音频表示比较，评估其与系统发育距离的相关性。

**💡 创新点**

发现大规模预训练音频模型（AST、CLAP、BEATs‑bio）能显著捕捉进化信号，且比传统MFCC更优；并表明域特定预训练并不一定提升 phylogenetic signal。

**🔧 技术方法**

使用 Mantel 相关检验、部分 Mantel、PCA 投影、Bootstrap 置信区间，比较四种音频表示（MFCC、AST、CLAP、BEATs‑bio）以及 BirdNET。

**📊 数据集**

Watkins Marine Mammal Sound Database（1,754 条录音，32 种）和 BirdCLEF 2023 选取的 607 条录音（20 种）。

**📈 对比分析**

通过 Mantel 检验将音频距离矩阵与系统发育距离矩阵相关，海洋哺乳类基础模型在 cetacean 内部 r ≈ 0.8，整体约 0.4；鸟类基础模型 r ≈ 0.5。与 MFCC 比较，基础模型表现显著更好。

**⚠️ 局限性**

限制包括录音来源的历史差异导致频率/采样率共变，样本量偏小，域特定模型可能受限于数据或目标。

---

## 359. Where FactsGo Missing: A LayerwiseTaxonomy and Per-Layer Attribution of Information Omissionin Air-Gapped LLM Agent Pipelines

**arXiv ID:** 2607.22448 | [PDF](https://arxiv.org/pdf/2607.22448v1)

**作者:** Santhiya Rajan `[一作]` `[通讯]` (Multiverse Computing), Santhiya Rajan (Multiverse Computing)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在空气隔离环境下系统性研究LLM代理的遗漏缺陷，提出九层漏失分类、漏失瀑布归因方法、跨架构基准与实时检测框架，构建完整的实验和诊断流程。

**💡 创新点**

创新点在于把遗漏视为管道级别问题，构造了九层漏失分类体系，设计漏失瀑布归因算法，以及通过跨模型、引擎、框架的对照实验揭示缺失来源，并提出基于日志的实时检测技术。

**🔧 技术方法**

使用的技术包括llama.cpp与vLLM推理引擎、MCP工具服务器、LangChain/ADK编排框架、日志注入与抽样、token与logprob监控、离线NLI判定与控制实验等。

**📊 数据集**

实验数据来自合成临床记录、PubMed摘要、FHIR/Synthea医学病例、SEC EDGAR金融法律文件等，覆盖临床、科研文献和金融法律三大领域。

**📈 对比分析**

通过跨模型（Gemma、Qwen3、Granite等）、跨引擎（vLLM vs llama.cpp）、跨框架（无框、LangChain、ADK）以及不同上下文长度、量化级别等因素进行对照实验，整体漏失率为0.62，主因来自软件层，行为层中注意力位置层贡献最大。

**⚠️ 局限性**

局限性包括仅在单GPU 16‑24 GB环境下实验、未系统探究服务器侧量化/KV缓存/ROPE参数、真实数据实验规模有限、版本依赖和NLI判定误差等问题。

---

## 360. VIREL: Verified Integer-Residual Encoding on Lattices for Exact and Error-Bounded Floating-Point Time-Series Compression

**arXiv ID:** 2607.22433 | [PDF](https://arxiv.org/pdf/2607.22433v1)

**作者:** Yue Zhang `[一作]` (Shanghai Jiao Tong University), Haopeng Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于验证整数残差编码的浮点时序压缩框架VIREL，能够在页面级别实现无损或误差约束的压缩。

**💡 创新点**

核心创新是对整数残差进行验证式预测、路由持久多通道以及基于格点的阶跃归一化，从而在混合精度下保持时间连续性。

**🔧 技术方法**

技术方案包含十进制整数化、整数残差预测、Rice/稀疏补丁、成本驱动的路由选择和格点归一化，以及误差限界下的因子化分段。

**📊 数据集**

使用14条官方ELF/ELF*基准、48条UCI+Microsoft等7400万条数据以及15条Serf误差约束序列进行评估。

**📈 对比分析**

与XOR、Erasure、ALP、DeXOR、Falcon、Machete等现有编码器对比，VIREL在精确模式下可达7.03×压缩比、比Falcon节省22.4%，在误差约束模式下12.11×压缩比、比Machete节省12.5%，编码吞吐量可达164 MB/s（16线程）。

**⚠️ 局限性**

局限在于规划成本较高、对十进制尺度的依赖、对极端高精度或无规律序列的表现不佳，以及GPU加速场景尚未充分探索。

---

## 361. Robust Berrut-Approximated Coded Computing via Discrete Cosine Transforms

**arXiv ID:** 2607.22427 | [PDF](https://arxiv.org/pdf/2607.22427v1)

**作者:** Rimpi Borah `[一作]` (Indian Institute of Technology Delhi), J. Harshan `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

提出了一种新的鲁棒Berrut近似编码计算框架（RBACC），旨在解决现有Berrut近似编码计算（BACC）在面对拜占庭工人时的鲁棒性问题。

**💡 创新点**

创新点在于通过引入离散余弦变换（DCT）编码，建立了BACC的编码理论框架，实现了在拜占庭工人存在的情况下的错误定位和纠正。

**🔧 技术方法**

使用了离散余弦变换（DCT）编码技术，并结合Berrut有理插值方法。

**📊 数据集**

使用了多个数据集进行实验，具体数据集未在摘要中详细说明。

**📈 对比分析**

与现有的BACC和ApproxIFER方法进行了比较，RBACC在处理拜占庭工人时表现出更高的重构准确性，能够有效纠正错误而不是简单丢弃错误计算。

**⚠️ 局限性**

RBACC的局限性在于其在面对高精度噪声和大量拜占庭工人时的性能可能会受到影响，且需要优化DCT编码维度以平衡错误检测和近似准确性。

---

## 362. Reflector: Arrangement-Aware Harmonic Retrieval for Sample-Based Composition

**arXiv ID:** 2607.22413 | [PDF](https://arxiv.org/pdf/2607.22413v1)

**作者:** Austin Rockman `[一作]` `[通讯]` (Independent Researcher), Austin Rockman (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Reflector，一款实时跟踪作曲进程、基于和声兼容性进行检索的音频工作站。

**💡 创新点**

创新点在于将手工设计的间隔权重表（oracle）与对合成音频的对比学习编码器相结合，生成可在多轨时间线上动态更新的和声嵌入空间。

**🔧 技术方法**

使用了基于CQT的色度特征、3层1‑D卷积编码器、对比损失InfoNCE、均方误差和KL散度等技术。

**📊 数据集**

训练数据由合成音频生成的60000对色度轨迹组成，检索评测在631个来自两位作曲家手工采集的真实样本库上进行。

**📈 对比分析**

通过与直接kernel、余弦相似度、TIV距离等六种检索基准比较，嵌入在top‑10中实现NDCG 0.85，覆盖率达625/631，并且检索速度提升约四个数量级。

**⚠️ 局限性**

局限性包括仅适用于西方调性、仅捕捉音阶关系，忽略音色/节奏等因素；评测仅在小规模数据集上，未对多尺度、键变或多样性控制进行深入验证。

---

## 363. Correlation-Aware and Gaussianity-Preserving Robust Latent Angular Watermarking for Diffusion Models

**arXiv ID:** 2607.22386 | [PDF](https://arxiv.org/pdf/2607.22386v1)

**作者:** Yebin Zheng `[一作]` (Singapore Institute of Technology), Yuguang Fang `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于旋转不变性的潜在角度水印（LAW）以及其高安全性变体LAW-M，在扩散模型中直接将水印嵌入潜在空间的角度关系，保持高斯分布并控制i.i.d.失真；

**💡 创新点**

创新点在于：①将水印信息编码为潜在向量对之间的正负π/2极角，实现最大角度分离和抗扰动；②提供严格的自相关解析，证明仅在离散稀疏位置产生±π/4的相关；③设计基于幅值排序的LAW-M，进一步提升对攻击的鲁棒性；

**🔧 技术方法**

采用潜在空间旋转、极坐标编码、DDIM反向推理提取、闭式自相关矩阵推导与角误差方差分析；

**📊 数据集**

使用Stable Diffusion v2.1生成图像，并在MS COCO 2017验证集的500条随机提示上进行评估；

**📈 对比分析**

与多种后处理与潜在域水印基线（Tree-Ring, RingID, HSTR, HSQR, ROBIN, Gaussian Shading, PRC等）比较，结果显示LAW-M在FID 96.6、CLIP 31.4、IS 14.3等指标上领先，并在大多数攻击场景下实现更高的提取准确率和TPR@1%FPR；

**⚠️ 局限性**

局限性：对随机丢弃攻击仍易被破坏；LAW-M需要为每张图片存储私钥，增加存储和密钥管理开销。

---

## 364. grapheme-kit: Grapheme-Level Metrics and Text Processing for Multilingual NLP

**arXiv ID:** 2607.22456 | [PDF](https://arxiv.org/pdf/2607.22456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 365. Local-Global Geometric Insights for Graph Neural Networks via Entropic Curvature

**arXiv ID:** 2607.22381 | [PDF](https://arxiv.org/pdf/2607.22381v1)

**作者:** Rachid Caich `[一作]` (Université de Montréal), Yassine Abbahaddou `[通讯]` (École Polytechnique IP Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了一种全局熵曲率（Entropic Curvature）及其可计算的弱熵曲率代理，用于理论分析与改进图神经网络的过平滑、过压缩、泛化等问题。

**💡 创新点**

创新点在于将Lott–Sturm–Villani的位移凸性框架迁移到离散图，定义可求解的弱熵曲率并从单一正向曲率约束推导Poincaré不等式、传输-熵泛化界以及“扩展悖论”，同时提出基于该曲率的聚合器E‑Gate、结构编码ENT和中点补全重连MCR。

**🔧 技术方法**

主要技术包括W1‑Wasserstein geodesics求解、基于两跳几何的H函数凸优化、传输-熵不等式、谱分析与重连算法；实验中使用GCN、GAT等标准GNN架构。

**📊 数据集**

实验数据涵盖六个节点分类基准（Cora、CiteSeer、PubMed、Cornell、Wisconsin、Texas/其它）及图分类数据，同时使用SBM生成的数据用于泛化分析。

**📈 对比分析**

与SDRF、FoSR、BORF、LCP、LAPE、RWPE、EGO、ORC、FRC等基线对比；E‑Gate在20个（backbone, dataset）组合中提升了16个；ENT编码在4/5基准上表现最佳；MCR在6类随机图中谱间隙提升最优；整体性能提升显著。

**⚠️ 局限性**

局限性包括：需预先指定参考测度m（目前使用均匀测度），弱熵曲率仅为全局曲率的下界，负曲率下泛化界仍受图直径限制，且大规模图上求解仍具一定计算成本。

---

## 366. Kutti AI: A Voice-First, Offline-Capable Learning Companion with Real-Time Struggle Detection for Visually-Impaired Children

**arXiv ID:** 2607.22377 | [PDF](https://arxiv.org/pdf/2607.22377v1)

**作者:** Kadharmoideen Fadurudeen `[一作]` `[通讯]` (Independent Researcher), Kadharmoideen Fadurudeen (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一款面向视障儿童的语音驱动学习伙伴Kutti AI，实现完全音频交互的学习循环。

**💡 创新点**

创新点在于三大机制：多信号实时挣扎检测、跨语言容错答案匹配以及离线优先的端侧语音识别。

**🔧 技术方法**

技术采用React Native + TypeScript移动端框架，端侧Whisper ASR模型，文本转语音引擎，多语言支持。

**📊 数据集**

尚未公开使用大规模数据集，主要使用Hackathon期间收集的英泰两语音样本进行原型测试。

**📈 对比分析**

对比方法为定性观察，未进行客观指标测评，现阶段仅展示在离线环境下的功能可行性与语音交互流畅性。

**⚠️ 局限性**

局限在于缺乏真实视障儿童的实测评估、阈值固定、简化题目预制、未涉及更丰富的情感分析等。

---

## 367. Robot Learning to Communicate through Projected Visual Abstractions

**arXiv ID:** 2607.22434 | [PDF](https://arxiv.org/pdf/2607.22434v1)

**作者:** Danyang Yan `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一套21自由度柔软皮肤机械手，通过学习可微阴影自模型并结合碰撞感知的梯度+局部搜索优化，能够根据目标阴影图像或视频生成对应的手部姿态，实现机器人通过投影阴影进行动态表达与沟通。

**💡 创新点**

创新点包括：①硬软混合手腕设计实现连续光遮蔽，②基于前向运动学+神经网络的可微阴影自模型并结合碰撞感知的两阶段优化，③加入表达区域掩模与时间正则化的关键帧优化，以提升动态阴影表达的细节与连贯性。

**🔧 技术方法**

使用的技术有：前向运动学 + 神经网络可微阴影自模型、梯度下降与hill‑climbing搜索、碰撞感知与物理仿真、表达区域（运动与封闭孔洞）与CLIP语义损失、关键帧聚类与序列优化、三次样条插值与运动规划。

**📊 数据集**

数据集为：仿真自我探索收集的 9,249,784 对关节-阴影样本；静态手势 26 张美国手语图；四组手影动物 puppetry 视频；两组野生动物（狼、乌鸦）视频，全部在实验室固定灯光与白幕环境下拍摄。

**📈 对比分析**

对比方法包括随机、逆向网络、最近邻、纯 hill‑climbing 等 baseline；在 61 个单帧目标上，前向+碰撞优化的总损失比最佳基线低约 33%，MAE、IoU、CLIP 等指标均显著提升；在动态视频目标上，关键帧+表达区+时间正则化将成功率从 34.5% 提升至 86.2%，实验视频展示与真实机器人表现高度一致。

**⚠️ 局限性**

局限性包括：只能在固定投影灯光与白幕的受控环境下工作，缺乏实时交互能力；单手可达形状有限，难以复制高度分离或极端比例的阴影；仅支持单手投影，无法利用多手或外物体进行阴影合成；离线优化耗时，无法实现即时表达。

---

## 368. Microwave Linear Analog Computers (MiLACs) for Communications: Opportunities and Challenges

**arXiv ID:** 2607.22509 | [PDF](https://arxiv.org/pdf/2607.22509v1)

**作者:** Matteo Nerini `[一作]` (Imperial College London), Bruno Clerckx `[通讯]` (Imperial College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并提出了利用线性微波网络（MiLAC）在射频域直接进行信号计算与波束成形的系统架构，能够在保持性能的同时显著减少RF链数量、降低ADC/DAC分辨率及降低计算复杂度。

**💡 创新点**

创新点包括：1）将矩阵求逆、伪逆等高阶运算通过MiLAC实现，使其复杂度从O(N³)降低至O(N²)；2）通过可重构的微波网络实现非线性参数依赖，实现结构非线性计算；3）提出两层MiLAC或混合数字-MiLAC的架构，突破传统可逆性与无损性限制，达成与全数字波束成形相同的性能。

**🔧 技术方法**

主要技术手段为：线性微波网络建模、可重构相位移/电容/二极管调节、微波PCB设计（如Butler矩阵、DFT变换）、两层或混合数字-模拟架构、低功耗RF链设计。

**📊 数据集**

论文以理论分析和仿真为主，并未使用公开数据集，主要通过计算复杂度、信噪比与容量等指标进行对比。

**📈 对比分析**

通过理论复杂度对比（矩阵-向量乘积O(MN)→O(1)，LMMSE O(MN²)→O(MN)，矩阵求逆 O(N³)→O(N²)）以及系统级仿真验证，显示MiLAC架构在RF链数、ADC/DAC分辨率和功耗方面相较于传统全数字/混合波束成形具有显著优势，且在单用户/多用户场景下能保持或超过全数字波束成形性能。

**⚠️ 局限性**

局限与挑战包括：实际硬件失真（插入损耗、调制离散化）、互耦与阻抗匹配问题、信道估计与频宽效应（波束弯曲）、高复杂度可重构网络的实现成本，以及如何在保持低硬件复杂度的同时保证性能。

---

## 369. Random-Order Online Facility Location Beyond Uniform Opening Costs

**arXiv ID:** 2607.22496 | [PDF](https://arxiv.org/pdf/2607.22496v1)

**作者:** Bo Peng `[一作]` (Shanghai University of Finance and Economics), Zhihao Gavin Tang `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在随机顺序下的在线度量设施位置问题中，给定有限候选设施集合与任意正开启成本，设计了一个确定性算法，使用按排名折扣的惩罚距离进行决策。

**💡 创新点**

创新点在于：①引入按已知总请求数归一化的排名钟（q_t = t/n）来动态调节开启成本折扣；②使用单次惩罚距离决策，避免成本离散化或多尺度决策；③通过单调一次性费用上限与上包络分解的组合，取得了新的竞争比 4.2674（比原来的 33 更好），以及在单位成本情形下的 3.2805。

**🔧 技术方法**

主要技术包括：随机顺序分析中的排名事实、单调一次性费用上限（monotone one‑round charge）、上包络（upper‑envelope）分解、聚类（optimal clusters）分析、以及对高成本溢出与锚点盈余的期望上界估计。

**📊 数据集**

该工作为理论研究，无实验数据集；所有结果均来自理论证明与随机顺序模型的概率分析。

**📈 对比分析**

与之前的 33 竞争比（Meyerson 1999）和 8 竞争比（Li et al. 2013）相比，算法取得了显著改进；在统一成本全空间模型下，已有 2.42 的竞争比，证明了统一与非统一成本模型的严格分离。

**⚠️ 局限性**

局限性：①算法需要事先知道请求总数（已知 horizon）才能构造排名钟；②对统一成本情形的性能尚未达到 2.42；③低阶项（如 3+4μ）仍是上界的主要损失，若要进一步优化需更紧凑的费用分配或多步支持机制。

---

## 370. Optimal Transport Image Representation and Deep Covariance Alignment (CORAL) for Control Valve Stiction Detection

**arXiv ID:** 2607.22486 | [PDF](https://arxiv.org/pdf/2607.22486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 371. Singular value soft-thresholding via the polar decomposition

**arXiv ID:** 2607.22484 | [PDF](https://arxiv.org/pdf/2607.22484v1)

**作者:** Stephen Becker `[一作]` `[通讯]` (University of Colorado Boulder), Stephen Becker (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用极分解实现奇异值软阈值化的方法，避免了传统的SVD。

**💡 创新点**

创新点在于将奇异值软阈值化转化为极分解，并使用GPU友好的Newton‑Schulz迭代来高效计算。

**🔧 技术方法**

采用极分解、PolarExpress、Newton‑Schulz迭代、矩阵乘法和bfloat16等技术。

**📊 数据集**

实验使用随机高斯矩阵（如301×300、1000×1001到5000×5001等）进行验证。

**📈 对比分析**

与标准SVD方法比较，极分解版本在GPU上约快10倍，精度可接受，且随尺寸线性增长。

**⚠️ 局限性**

局限性包括对高精度不鲁棒；当τ接近最大奇异值时误差较大；需要更多迭代；实现尚未支持批处理。

---

## 372. TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters

**arXiv ID:** 2607.22432 | [PDF](https://arxiv.org/pdf/2607.22432v1)

**作者:** Zhiwen Mo `[一作]` (Imperial College London), Hongxiang Fan `[通讯]` (Imperial College London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于 Tile 的 GPU 性能分析模型（Tile-Centric Analytical Performance Model），能够在单 GPU、单机多 GPU 以及多机集群环境下对 TensorCore、CUDA Core、缓存层次、内存/网络管线以及跨 GPU 通信等资源进行白盒预测，支持对张量级别调度、缓存重用、软件流水线深度、跨设备映射等多维度参数进行评估，并可在不跑核的情况下给出延迟、资源利用率和瓶颈定位。

**💡 创新点**

创新点包括：
- 将 Tile 作为统一的建模单元，分别在 intra‑tile、inter‑tile 与 cross‑device 三层递归地构造资源向量、依赖图与跨设备通信。
- 引入 Tile Reuse‑Distance 与 Stochastic Distance Cache Model（SDCM），在 Tile 级别实现多级缓存（L1.5/L2/DDR）命中率预测，避免传统的行级缓存仿真。
- 采用 α–β 交换路由模型与逻辑交换分解，将跨 GPU 通信与本地计算通过相同的流水线 envelope 进行整合，支持多种 collectives 与点对点。
- 通过顶层资源向量与流水线 envelope 的递归模拟，准确捕捉软件流水线、并发调度、缓存重用和通信重叠的交互影响，实现单机 12.35% MAPE、32 GPU 16.18% wMAPE、以及 13.52% wMAPE 的端到端预测，显著优于现有基准。

**🔧 技术方法**

核心技术包括：
- Tile 级别资源向量（⟨TC, CUDA, SFU, TMEM, SMEM, L1.5, L2, DDR, Net⟩）与对齐的硬件速率校准。
- 依赖图 Topo 搜索与流水线 envelope（prologue‑steady‑epilogue）模型。
- Tile Reuse‑Distance 计算与 Gaussian SDCM 近似实现多级缓存命中率。
- α–β 路由模型结合逻辑交换（s,d,b）拆分实现跨设备通信时延预测。
- 统一硬件抽象层，支持 NVIDIA（A100, H200, B200, B6000）与 AMD（MI210）多代 GPU。
- Python 实现（≈6K 代码），与 Triton、TileLang、CUDA Tile、CuteDSL 兼容。

**📊 数据集**

数据集与实验基准：
- 703 BF16/FP16 TensorCore GEMM 形状（A100、B200、B6000、H200）。
- 4,680 持久化内核（persistent kernel）用于缓存命中率评估。
- 166 vLLM decode 配置（Dense 与 MoE，单机到 32 GPU）。
- 纯集合（AllGather、AllReduce、ReduceScatter、All‑to‑All）与融合 compute‑communication 核心（GEMM+ReduceScatter、Ulysses Attention 等）。
- 通过微基准校准硬件速率、缓存容量、网络 α–β 参数。

**📈 对比分析**

比较方法与性能：
- 与 Roofline、NeuSight（再训练 FP16 数据）、PipeWeave、GenZ 等基准对比。单 GPU GEMM 预测 MAPE：12.35%（最优），对比 Roofline 21.97%、PipeWeave 32.95%、NeuSight 33.85%。
- 纯集合预测 wMAPE：12.22%（优于 GenZ 20.82% 与 PipeWeave 65.72%）。
- 融合 compute‑communication 核心预测 wMAPE：14.83%（比基准高）。
- 端到端 vLLM decode 预测 wMAPE：13.52%（比 PipeWeave 31.84% 高）。
- 在 TileLang 任务上，模型可删减 95% 的调度候选，保留 top‑5% 时可获得 99.66% 的最佳性能。
- 通过诊断案例展示对间接寻址、流水线停滞、L2 局部性与内存布局等瓶颈的定位与修复，提升 1.07–8.97×。

**⚠️ 局限性**

局限性：
- 仅适用于规则的、Tile 结构化程序，无法建模数据依赖控制流、极不规则内存访问、指令级编译器决策、隐藏的 warp/CTA 调度或闭源运行时行为。
- 对小批量（latency‑bound）或多 die（如 B200 SM‑HBM 亲和性）场景缺乏细粒度时延与拓扑参数。
- 假设所有 SM 以均匀速率执行 Tile，导致大 K GEMM 时 L2 命中率略高估。
- 缺乏对指令级或 warp 级别资源竞争的细化建模，无法捕捉极端小规模 kernel 的 warp/CTA 调度细节。
- 对跨节点通信的 α–β 参数依赖手工校准，若拓扑变化或网络拥塞情况较大时预测误差可能增大。

---

## 373. A Self-Calibrating Agentic AI Framework for Autonomous Edge Resource Allocation

**arXiv ID:** 2607.22400 | [PDF](https://arxiv.org/pdf/2607.22400v1)

**作者:** Fin Gentzen `[一作]` (Technical University of Braunschweig), Admela Jukan `[通讯]` (Technical University of Braunschweig)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种自校准的代理式 AI 框架，用于在资源受限的边缘环境中零知识 AI 工作负载的资源分配与预测。

**💡 创新点**

创新点在于将 LLM 与 ARIMA 预测、RAG 知识检索、主动剖面和自我校准模块结合，能够在无先验元数据的情况下动态生成并精炼自己的地面真值，实现预测误差从 200% 降至单个位数 MAPE。

**🔧 技术方法**

使用的技术包括大语言模型（7B 参数）、ARIMA 时序预测、RAG 检索、k‑NN 相似度搜索、M/G/∞ 量化队列模型以及自适应参数搜索的跳跃策略。

**📊 数据集**

实验使用了 53 条 AI 工作负载，涵盖 8 种模型架构、6 个公开数据集（CIFAR‑10、MNIST、FMNIST、Forest‑Cover、HIGGS、Wine‑Quality）并在 Raspberry Pi 5、NVIDIA Jetson Thor 与 GPU 工作站上收集。

**📈 对比分析**

将完整代理框架与零射 LLM 基线和传统完整监测/ARIMA 对比，预测准确率提升至 91.7%（误差降低至单数 MAPE），并将完整监测时间缩短 71.7%，ARIMA 跳跃算法比标准 ARIMA 快 52% 同时保持相同精度。

**⚠️ 局限性**

主要局限包括对训练时间预测的依赖仍较差、冷启动时 RAG 效果有限、仅使用简单 CPU/GPU 模型、无法处理广泛的超参空间，以及沙盒环境实现的复杂性与噪声干扰。

---

## 374. Plug, Play, and Comply: A Modular Framework for Online Variable Impedance with Arbitrarily Oriented Compliance Axes

**arXiv ID:** 2607.22483 | [PDF](https://arxiv.org/pdf/2607.22483v1)

**作者:** Mihael Simonič `[一作]` (Eastern Institute of Technology), Xiaocong Li `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种插件化、机器人无关的顺应性控制框架，标准化了关节与笛卡尔命令接口，并支持在线任务相关的柔顺度重新定向。

**💡 创新点**

创新点包括：①将控制算法与ROS集成层拆分，提供可在不同机器人上复用的插件接口；②在笛卡尔接口中引入可在任意坐标系下更新刚度和阻尼的功能，实现任务依赖的柔顺度；③提供Simulink到插件的自动生成流水线，降低低层控制开发门槛。

**🔧 技术方法**

技术手段：ROS 2 control框架、Pinocchio动态库、URDF解析、插件化C++/Simulink实现、实时线程安全的命令缓冲、软限位、低通/速率限制、以及可插拔的力/惯性补偿。

**📊 数据集**

数据集：未使用公开数据集，主要通过真实Frank FR3机器人以及KUKA LBR iiwa、Flexiv Rizon 4s、UR5e等模型的仿真与真实实验进行验证；实验数据包括曲面跟踪误差、接触力统计和插接成功率。

**📈 对比分析**

对比方法：与ROS中现有的顺应性控制器（如franka, libfranka, CRISP等）进行功能与可移植性对比；实验结果显示：任务相关柔顺度将横向RMSE从16.3 mm降至7.4 mm，最大偏差从40.4 mm降至13.7 mm；插接时最大允许角度误差从3°提升到6°，成功率提升到100%。实时性能：控制周期1 kHz下平均计算时长3.82 µs（插件）+0.67 µs（包装器）<5 µs，满足实时要求。

**⚠️ 局限性**

局限性：依赖完整URDF描述和Pinocchio支持；对高度复杂的动态/力学模型需额外插件；框架未对多机器人协同或多任务优先级作深入研究；在极端高速或高载荷场景下的稳定性尚待进一步验证。

---

## 375. On the alleged chaos in periodically forced traveling-wave reductions of fractional WBBM models: a quantitative re-examination

**arXiv ID:** 2607.22475 | [PDF](https://arxiv.org/pdf/2607.22475v1)

**作者:** Kaixuan Niu `[一作]` `[通讯]` (China Science TransCentury Aerospace Institute), Kaixuan Niu (China Science TransCentury Aerospace Institute)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文对一种常见的分数阶非线性波动方程研究流程（先做波前变换、再得到平面哈密顿系统，随后加入周期激励并以可视化相图判定动力学类型）进行了批判性评估，并对一篇代表性论文（Ullah 等 2024 年关于二阶分数阶 WBBM 模型的研究）进行全量重新检验。

**💡 创新点**

创新点在于系统地揭示并量化该流程中的主要误区：将平面哈密顿系统的线性稳定分析误解为指数不稳定、把无激励系统的周期轨道误标为准周期、以及仅凭视觉相图声称混沌等问题；并提出了一套可操作的量化标准与检查清单，可直接用于此类研究的同行评审。

**🔧 技术方法**

采用了四种独立的数值诊断方法：Benettin 两轨迹最大 Lyapunov 指数、未归一化分离增长率、剥蚀周期 Poincaré 切面和傅里叶谱分析，并对 0–1 试验的误判机制做了定量说明。

**📊 数据集**

并未使用外部数据集，所有结果均基于对平面哈密顿方程的高精度数值积分（DOP853、绝对/相对误差 10⁻¹²/10⁻¹⁴）。

**📈 对比分析**

与原文仅依赖相图的结论相比，本研究通过上述四种量化诊断清楚表明原文所声称的混沌轨道实际上是正则的；而 0–1 试验在该系统上产生误判，进一步验证了单一统计量的局限性。

**⚠️ 局限性**

局限性在于仅针对单一代表性案例，无法覆盖所有分数阶波动模型；此外，诊断方法仍需长时间积分与高精度计算，对计算资源和实现细节有一定要求。

---

## 376. Complexity Bounds and Approaches to Learning Projected Gradient Descent Solver Iterates

**arXiv ID:** 2607.22467 | [PDF](https://arxiv.org/pdf/2607.22467v1)

**作者:** Anjian Li `[一作]`, Ryne Beeson `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 k‑neighborhood 数据收集策略，将求解器的中间迭代加入生成式学习数据集中，并针对一侧箱式约束二次规划（box‑constrained QP）推导了投影梯度下降（PGD）的收敛行为及其对生成模型的影响。

**💡 创新点**

创新点在于（1）正式定义 k‑neighborhood，利用最近 k+1 次迭代提升训练样本量；（2）基于 Rademacher 复杂度推导了包含求解器迭代数据的泛化误差上界；（3）将上述理论与 GLENS（Learning from Solver Iterates）方法相结合，为数据高效全局搜索提供理论支撑。

**🔧 技术方法**

使用的技术包括投影梯度下降、两层受限权重的神经网络（用于噪声预测器）、Denoising Diffusion Probabilistic Model（DDPM）、Rademacher 复杂度分析以及 Lipschitz/投影不变性等数学工具。

**📊 数据集**

实验中未使用公开真实数据集，所有样例均为合成的二维盒约束 QP 问题与相应的求解器迭代路径；通过这些合成数据验证理论推导。

**📈 对比分析**

文章未给出基于标准基准数据集的对比实验；通过理论证明说明泛化误差随训练样本数 N、k 值、收敛因子 ρ 和折扣因子 γ 的变化而呈 O(1/√N) 的递减趋势，并给出了两组 PGD 路径的可视化示例来说明算法行为。

**⚠️ 局限性**

主要局限包括：Rademacher 复杂度上界保守（使用全局权重上界、最坏情况半径 r_k 和统一收敛因子 ρ）；假设扩散噪声受限且未考虑求解器迭代间的真实依赖；理论仅针对一侧箱式约束 QP，难以直接推广到更一般的非凸或无界问题。

---

## 377. A Maximum Entropy Implementation of Differential Privacy Under Linear Invariants

**arXiv ID:** 2607.22450 | [PDF](https://arxiv.org/pdf/2607.22450v1)

**作者:** Ryan Lafferty `[一作]` (University of Maryland Baltimore County), Anindya Roy `[通讯]` (University of Maryland Baltimore County)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在满足线性聚合不变约束（如州级总计）下实现差分隐私的高熵机制，保证隐私预算在满足约束的同时不被破坏。

**💡 创新点**

创新点在于将噪声采样问题转化为在约束子空间上寻找最大熵相关矩阵，并通过投影到凸集合的迭代算法（POCS）和投影梯度法高效求解。

**🔧 技术方法**

采用高斯尺度混合模型、投影到凸集合（POCS）、投影梯度算法、以及最大熵正则化等技术，构建满足约束的相关噪声向量。

**📊 数据集**

以美国人口普查（Census）数据为应用案例，使用合成的州/县/区块层级计数进行实验验证。

**📈 对比分析**

通过Q‑Q图验证采样噪声的边缘分布与目标分布一致，实验表明约束满足误差可忽略（10⁻⁶级别），并展示了多层级加噪时的隐私预算和一致性保持效果。

**⚠️ 局限性**

局限性包括只能处理线性等式约束（不涵盖不等式或非线性约束），对高维多约束情形的计算成本较高，且假设约束向量为平衡向量，实际应用中需进一步验证。

---

## 378. Dynamic Capability Scoping for Enterprise AI Agents: A Synthetic Dataset and Three-Source Permission Architecture

**arXiv ID:** 2607.22445 | [PDF](https://arxiv.org/pdf/2607.22445v1)

**作者:** Halil Burak Noyan `[一作]` `[通讯]` (Independent Researcher), Halil Burak Noyan (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种三源动态能力范围架构，利用角色上限、任务上下文分类器和基于策略的组合禁令来限制企业AI代理的权限，并构建了600条合成任务提示的语料库进行评估。

**💡 创新点**

创新点包括：①将能力范围动态化并嵌入最小特权原则；②设计了三层防御（角色上限→任务分类器→组合禁令）以抵御提示注入和恶意侧任务；③通过两步生成流程实现合成数据与策略共进化，显著降低角色上限违规率。

**🔧 技术方法**

主要技术：LLM（Prompt‑Engineered）生成合成提示与权限标签；多标签分类器（fine‑tuned）预测任务所需权限；Policy‑as‑Prompt 自动从公司政策生成组合禁令规则；评估采用Cohen κ、Hamming准确率、Macro‑F1、严重度加权过授等指标。

**📊 数据集**

使用了自研的600条合成企业任务提示数据集，提示基于六部门（Engineering、Data & Analytics、Security、Customer Success、Finance、Legal & Compliance）的公司政策生成，权限划分为15个工具级别，并标注最低所需权限。数据集与生成/标注脚本、政策文档一起发布。

**📈 对比分析**

评估方法：先用人工标注60条样本做验证，计算预评审与后评审的指标。后评审后，精确匹配率从68.3%提升至85.0%，Hamming准确率从97.1%提升至98.8%，Macro‑F1从0.920提升至0.966，Cohen κ从0.917提升至0.967。严重度加权过授率从15.0降至2.0（下降86.7%）， undershoot率从8.1%降至4.4%。

**⚠️ 局限性**

局限性：仅单一专家进行验证，可能存在自评偏差；合成数据缺乏真实企业通信的噪声与隐含上下文；任务分类器仅基于初始提示，无法预判多步推理过程中的权限扩展；组合禁令规则依赖于手工或Prompt‑Derived策略，若策略不完整会误阻或误放；对提示注入和侧任务仍存在一定风险。

---

## 379. Conformal Constraint Tightening for Chance-Constrained Motion Planning with Unknown Dynamics

**arXiv ID:** 2607.22409 | [PDF](https://arxiv.org/pdf/2607.22409v1)

**作者:** Shubham Natraj `[一作]` (Washington University in St. Louis), Yiannis Kantaros `[通讯]` (Washington University in St. Louis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种与规划器无关的约束收紧框架，通过合适的约束收紧使得在未知真实动力学下的运动规划能够获得概率性任务完成保证。

**💡 创新点**

创新点在于利用合成预测（Conformal Prediction）对一系列规划问题中名义轨迹与真实轨迹的偏差进行分布无关的概率界定，并据此构造收紧后的规划问题；该方法能在不需要显式模型不确定性描述的情况下，为任何规划器提供 1−α 的成功概率保证。

**🔧 技术方法**

主要技术包括：合成预测用于计算轨迹级误差阈值；基于该阈值收紧自由空间和目标集合；采样式 kinodynamic RRT 进行规划；在第二个实验场景中还使用了一个一阶神经网络预测器来模拟未知动力学。

**📊 数据集**

使用的校准数据集为 100 个规划问题（每个问题包含若干可行控制序列），分别针对四种不同的 Dubins 车辆误差参数和一个随机扰动的平面四旋翼；实验环境为 25×25 单位的平面，随机放置五个圆形障碍。

**📈 对比分析**

方法与直接在名义模型上运行的 RRT 进行对比。评价指标包括经验覆盖率、规划器成功率和任务完成率。结果显示，收紧方法的任务完成率始终超过 1−α 并且显著优于名义规划基线，虽然随着 1−α 提升规划器成功率略有下降，体现出保守性的权衡。

**⚠️ 局限性**

局限性：收紧过程过于保守导致规划器成功率下降，尤其在大模型误差或高置信度要求时；需要先验的校准数据集；目前对随机噪声的处理相对保守，且未提出自适应风险规划策略。

---

## 380. LunarFM: A Shared Multimodal Representation of the Moon's Surface

**arXiv ID:** 2607.22408 | [PDF](https://arxiv.org/pdf/2607.22408v1)

**作者:** Marc Girona-Mata `[一作]` (University of Cambridge), Raúl Ramos-Pollán `[通讯]` (Universidad de Antioquia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作构建了LunarFM多模态基础模型，将来自三次月球任务的六种遥感仪器（共18个通道）的0.5°×0.5°切片映射到同一768维嵌入空间，并在此基础上演示了矿物回归、稀少标签预测、相似检索和地质单元分类等下游任务。

**💡 创新点**

其创新点包括：①统一整合多任务多仪器数据到单一潜在空间；②采用多模态掩码自编码器（MultiMAE）进行自监督预训练；③公开完整数据集、预训练模型和嵌入，形成可复现的科研基础设施。

**🔧 技术方法**

技术方法包括多模态掩码自编码器+视觉Transformer编码器、随机遮挡、自监督训练、随机森林回归/梯度提升分类、PCA/UMAP可视化以及k‑means聚类等。

**📊 数据集**

使用的数据集为LunarChips（±70°纬度，0.5°×0.5°切片，共18通道、六个仪器）及其对应的LunarEmbeddings（768维嵌入），并提供矿物、元素、撞击坑和USGS地质单元等下游标签。

**📈 对比分析**

通过与传统单模态或原始特征的对比，下游任务显示LunarFM表现优异：在默认band split下地质单元分类Top‑1≈33.6%、Top‑3≈60.9%；少样本稀土预测相关系数≈0.78（vs.随机≈0.65）；矿物回归MAE在可接受范围内；相似检索在可视化上与多模态输入高度一致。

**⚠️ 局限性**

主要局限包括：覆盖纬度仅±70°，南极区缺失；Mini‑RF数据缺失导致重建误差高；嵌入尺度随纬度变化，影响跨区域比较；缺乏高分辨率（NAC）等数据；模型架构与超参数未针对月球特定任务进行专门优化。

---

## 381. IR275K: A Benchmark for Infrared Multi-Frame Super-Resolution Toward Efficient Remote Sensing

**arXiv ID:** 2607.22380 | [PDF](https://arxiv.org/pdf/2607.22380v1)

**作者:** Jie Deng `[一作]` (Hangzhou Institute for Advanced Study), Jianyu Wang `[通讯]` (Hangzhou Institute for Advanced Study)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了IR275K一个针对红外多帧超分辨率的基准数据集，并在其上评估了一种轻量化的状态空间模型CGMamba，旨在实现高效、可复现的红外MFSR。

**💡 创新点**

创新点在于：1）设计了专门针对红外视频特性（低对比度、高噪声、弱纹理、平台运动）并引入对比-噪声耦合、噪声域差距和运动鲁棒性三大压力测试的基准；2）提出CGMamba利用2D RoPE与中心引导的跨Mamba融合实现隐式帧融合，同时证明空间锚定是实现稳健跨帧门控的关键。

**🔧 技术方法**

技术主要包括：基准构建与分层划分、×4降采样评价协议、2D旋转位置编码、中心引导跨Mamba（CGCM）结构、状态空间模型（SSM）与轻量化通道注意力、Charbonnier损失训练。

**📊 数据集**

使用的数据集为IR275K，包含594条红外视频序列，合计275,196帧，分为训练261条、验证111条、测试222条。

**📈 对比分析**

对比方法包括双三次插值、GPSMamba、IRSRMamba三种单帧超分模型；CGMamba在PSNR上提升至33.19 dB，超越单帧模型0.35–0.52 dB，同时参数量、FLOPs和运行时均显著更低（10.90 M参数、112 G FLOPs、340 ms）。

**⚠️ 局限性**

局限性包括：1）仅使用×4双三次降采样，未加入真实传感器失真；2）CGMamba仅在三帧输入下测试，未探究更长时间窗口或帧序列顺序影响；3）评估指标仅限PSNR/SSIM；4）未系统比较红外与可见光MFSR方法的可迁移性。

---

## 382. A Factorial Study of Synthetic Data Generation for Low-Resource Machine Translation using Grammar Books

**arXiv ID:** 2607.22376 | [PDF](https://arxiv.org/pdf/2607.22376v1)

**作者:** Varun Ghat Ravikumar `[一作]` (University of Zurich), Rico Sennrich `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个端到端的半自动化流水线，利用大型语言模型从文献式语法书中提取句子对、词典和语法规则，并通过规则引导的词汇替换生成合成平行语料，用以微调机器翻译模型。

**💡 创新点**

创新点在于：①把语法书的结构化信息转化为可直接用于训练的数据，而非仅在推理时作为提示；②通过 LLM 自动化提取与编码语法规则，形成可执行的伪代码；③设计了多因素（词性、检索粒度、样本量）交叉实验，系统评估哪些组合能最大化提升。

**🔧 技术方法**

技术包括：PDF 文本定位+BERT 行级分类器（≈200 例手标）；Gemini‑2.5‑flash 用于规则提取与合成句子；语义检索（嵌入相似度）与结构匹配相结合用于规则检索；规则编码成 YAML → 伪代码；在 Gemini 上进行 3 轮 LoRA 微调；评估使用 BLEU、ChrF 与 ChrF++。

**📊 数据集**

数据集主要来自三本语法书：Kalamang（Papuan）、Tuatschin（Romance）和 Mandan（Siouan）。从书中抽取了约 2,500、1,800 和 460 条真实平行句子、词典条目和语法规则；随后按 32 种配置生成合成语料（k=5,10,15,20）。

**📈 对比分析**

与零样本、NLLB‑200、在推理时注入语法书（ICL）以及仅使用真实平行句子的微调（SDFT）相比，合成语料微调在 75%（Kalamang）和 59%（Tuatschin）的配置中提升了翻译质量；平均 ChrF++ 分别提升 +2.0（Kalamang）、+1.5（Tuatschin）和 +0.8（Mandan）。最优配置可获得 +8.8、+5.3 和 +3.3 的最高提升。

**⚠️ 局限性**

局限性包括：①仅使用自动评估指标，缺少母语使用者评估；②实验仅覆盖一本语法书，可能混合了书写风格与语言特性；③合成语料在高样本量下易产生语义不连贯或过度拟合；④依赖英文 NLP 工具，难以推广至非英文目标；⑤对音调、声调或非拼接形态学的处理不足。

---

## 383. IDEAgent: Agentic Quality-Diversity Search for Research Idea Generation

**arXiv ID:** 2607.22375 | [PDF](https://arxiv.org/pdf/2607.22375v1)

**作者:** Varun Gumma `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多代理框架，将科研构思视为质量-多样性搜索，系统通过顺序生成、评估、修复和改进构建多元高质量研究想法组合。

**💡 创新点**

核心创新在于将质量和多样性统一为QD搜索，引入分层存档与核心摘要、以及针对评估缺陷的修复/细化子程序，并提出联合评估指标Yield。

**🔧 技术方法**

使用多代理模型（Ideator、Critic、Quality Evaluator、Soundness Panel、Diversity Judge、Stenographer）及内部评估和外部LLM评测，基于最新专有LLM实现。

**📊 数据集**

通过从多个arXiv领域（机器学习、计算机视觉、系统等）抽取的32个主题背景文献集合，构建背景语料库进行实验。

**📈 对比分析**

与无状态、一次性、顺序记忆、NOVA等基线对比，实验表明在相同预算下该方法在Yield、成功主题比例、质量分数等指标上平均提升约50%至200%，并显著提高多样性和质量。

**⚠️ 局限性**

主要局限包括依赖大型专有LLM、评估主观性与高成本、修复/改进轮数有限、未对生成想法进行真实实现验证，且对低质量模型的适用性尚未验证。

---

## 384. Twins: Learn to Predict Unified Representations with Focal Loss

**arXiv ID:** 2607.22531 | [PDF](https://arxiv.org/pdf/2607.22531v1)

**作者:** Kaixiong Gong `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Twins 统一连续视觉表示，将 ViT 的语义特征与 VAE 的细节特征在通道维度拼接，并通过焦点回归解决优化不平衡。

**💡 创新点**

创新点包括：① 用极简通道拼接实现共享表示；② 对频谱、内在维度、条件依赖三种异质性来源导致的优化失衡进行系统分析；③ 在流匹配中引入焦点损失以平衡两种特征的学习。

**🔧 技术方法**

使用技术：Diffusion Transformer（DiT）+ Flow Matching；Focal Loss；ViT（SigLIP2）+ VAE（Flux.2）预训练编码器；频谱分析、两近邻估计内在维度等。

**📊 数据集**

实验数据集：ImageNet-1K（验证集）用于生成与重构；多模态理解 benchmark（POPE、GQA、TQA、MMB、MME-S/P）用于评测。

**📈 对比分析**

与 SigLIP2、Flux.2 VAE、UniFlow、RAE 等基线比较；在 ImageNet-256/512 生成中，Twins+FocalLoss 的 gFID 分别为 1.59（512）/3.78（512）/0.18（256）低于 Flux.2；重构 PSNR 31.46、SSIM 0.90、rFID 0.11 超越对比方法；在理解任务上性能相当或略优。

**⚠️ 局限性**

局限性：仍依赖预训练 ViT/VAE，缺乏端到端训练；高维 latent 的训练稳定性仍受挑战；实验仅在 ImageNet 及部分多模态 benchmark 上验证，跨域泛化待进一步验证；Focal Loss 参数需手动调节。

---

## 385. Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills

**arXiv ID:** 2607.22529 | [PDF](https://arxiv.org/pdf/2607.22529v1)

**作者:** Siyuan Huang `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Skill Self‑Play（Skill‑SP）框架，利用可演化的模块化技能库引导大语言模型（LLM）在自我对弈中主动生成、验证并解决任务，从而实现持续自我提升。

**💡 创新点**

创新点：1）将技能抽象为可训练的任务模式接口，既提供结构先验又能被验证；2）在生成端与求解端同时进行强化学习，通过双向反馈动态演化技能库；3）采用双流生成（技能驱动+开放探索）避免模式坍塌，同时保持任务多样性；4）将验证与技能演化紧耦合，形成闭环式自我提升。

**🔧 技术方法**

主要技术：强化学习（GRPO）在生成器与求解器上迭代；可验证任务抽象（schema、unit test、参考答案等）；技能接口定义与动态路由；任务质量门控与前景奖励；技能库的精炼、修剪与诱导；多任务混合课程构建。

**📊 数据集**

数据集：1）工具调用任务——API‑Bank（1‑3 级）和 BFCL（四种编程语言的简单任务）；2）逻辑推理任务——ZebraLogic（不同规模的格子谜题）。

**📈 对比分析**

对比方法：与初始基线模型（Base checkpoint）以及仅使用被动后置过滤的 Unguided Self‑Play 进行对比。评价指标为 avg@8，Skill‑SP 在所有五种 LLM 上均实现显著提升：工具调用最高 +42.9 点（绝对增益），逻辑推理最高 +12.0 点。实验还包括消融与诊断分析，验证双流与技能演化对性能提升的关键作用。

**⚠️ 局限性**

局限性：1）对初始 LLM 的能力要求较高，弱模型难以自行生成可验证任务，导致自我提升难以启动；2）技能诱导与修炼依赖对任务模式的解析，可能导致新技能生成的多样性受限；3）在极大规模任务（如超大格子谜题）上提升有限，仍需更强的前置能力；4）实验周期相对较长（多轮强化学习），对资源消耗敏感。

---

## 386. Robot-Factored World Models via Robot Rendering

**arXiv ID:** 2607.22535 | [PDF](https://arxiv.org/pdf/2607.22535v1)

**作者:** Byungjun Kim `[一作]` (Seoul National University), Hanbyul Joo `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并验证了一种机器人世界模型，将动作实现与机器人渲染分离，使用部署可用的名义轨迹渲染为可视化动作输入。

**💡 创新点**

通过先将动作通过机器人控制器和运动学产生名义轨迹，再将该轨迹渲染为摄像头对齐的URDF网格和深度信息，避免了动作实现学习与交互结果泄露，提升了对不同机器人与人类演示的通用性。

**🔧 技术方法**

采用视觉条件视频生成技术（基于 Stable Video Diffusion / Wan 框架），结合机器人 URDF 渲染、端执行器深度与场景深度的多模态视觉输入，并利用 Isaac Lab 与 cuRobo 生成名义轨迹。

**📊 数据集**

主要使用 DROID（增强外参子集）和 RoboCasa‑GR1 两大机器人抓取/操作数据集；定性评估还引用了 HRDexDB 与 DexYCB。

**📈 对比分析**

与基于向量/姿态条件的 Ctrl‑World、AdaLN 等基线在 PSNR/SSIM/LPIPS 上进行对比，渲染接口在所有指标上均优于基线，且在未见机器人上实现零样本泛化，可将人类演示重目标化为机器人视频。

**⚠️ 局限性**

需要已知 URDF 与相机‑机器人校准，静态上下文流假设场景不变（真实环境需补充重建），并且训练数据中成功案例占多数，缺乏失败、滑动等情况，限制模型对失误场景的鲁棒性。

---

## 387. Opaque Epistemic Mediation: How LLM Deployment Configurations Shape the Validation of Pseudo-Science

**arXiv ID:** 2607.22513 | [PDF](https://arxiv.org/pdf/2607.22513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 388. SM4RT: Learning Structured Motion Geometry for 4D Reconstruction

**arXiv ID:** 2607.22534 | [PDF](https://arxiv.org/pdf/2607.22534v1)

**作者:** Shing Ho J. Lin `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种从单目RGB视频直接进行三维几何、世界坐标运动与场景运动结构的联合推断框架。

**💡 创新点**

核心创新是结构运动（Structure-of-Motion）表示，将运动分解为少量6D twist序列的运动基，并通过稀疏时间共享像素分配实现物体级别的刚体运动一致性。

**🔧 技术方法**

技术上采用Geometry Foundation Model（DepthAnythingV3）为基础，构建并行的运动几何编码器与解码器，利用Transformer自注意力、稀疏max与伪背景正则化等模块实现一次前向推理。

**📊 数据集**

训练使用多种动态数据集（Kubric、Stereo4D、PointOdyssey、DynamicReplica）以及静态几何集（ARKitScenes、HyperSIM、ScanNet++、VKitti2、Waymo）。

**📈 对比分析**

在TapVid‑3D、世界坐标跟踪、三维重建、深度估计等基准上与D4RT、4RC、V‑DPM等方法对比，SM4RT在AJ/APD、APD/EPE、深度误差及重建精度上均取得或接近最优表现，显示出更高的运动结构一致性和跟踪精度。

**⚠️ 局限性**

局限性在于对复杂物理交互（碰撞、接触等）的处理不足，运动基分配在某些场景可能不稳定，且需要较大模型参数与训练时间。

---

## 389. ViTacWorld: Scaling Visuo-Tactile World Models for Contact-Rich Robot Manipulation

**arXiv ID:** 2607.22530 | [PDF](https://arxiv.org/pdf/2607.22530v1)

**作者:** Yunao Huang `[一作]` (ShanghaiTech University), Jingya Wang `[通讯]` (ShanghaiTech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 ViTacWorld，一种能够基于机器人动作预测同步视觉与触觉序列的世界模型，用于生成大规模触觉交互数据。

**💡 创新点**

创新点在于将动作条件的视频世界模型扩展到多模态（视觉+触觉），并通过预训练+仿真+真实策略微调实现可扩展的触觉数据生成与策略评估。

**🔧 技术方法**

采用视图感知的 Diffusion Transformer（DiT）与 VAE 编码器、跨视图注意力、动作嵌入等技术构建模型，并结合 Isaac Sim 触觉渲染实现仿真数据生成。

**📊 数据集**

使用公开的实时触觉数据集、基于 Isaac Sim 的任务对齐仿真数据以及 Franka Panda 机器人真实演示与策略回放进行训练和微调。

**📈 对比分析**

与仅使用真实专家演示或有限触觉策略的基线相比，在四个接触任务上通过 ViTacWorld 生成的梦幻数据提升平均成功率至 67.5%，显著优于仅用专家数据的 35%。

**⚠️ 局限性**

局限在于梦幻数据的筛选仍需人工检查，缺乏自动化评估机制，且仿真与真实环境之间仍存在一定的差距。

---

## 390. Dysphagia Risk Stratification in Head and Neck Cancer via Two-Stage PRO-Clinical Stacking

**arXiv ID:** 2607.22514 | [PDF](https://arxiv.org/pdf/2607.22514v1)

**作者:** Siyuan Zhao `[一作]` (University of Illinois Chicago), Guadalupe Canahuate `[通讯]` (University of Iowa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究通过单次访视的MDADI患者报告与临床变量的两阶段堆叠模型预测头颈癌患者的吞咽功能损伤风险，无需影像。

**💡 创新点**

结合PRO和临床信息的两阶段堆叠框架，保留非线性PRO信号并实现可解释性，证明单访视PRO足以识别高危患者。

**🔧 技术方法**

采用XGBoost+ElasticNet两阶段堆叠模型，并利用SHAP进行特征重要性解释以及ElasticNet系数稳定性分析。

**📊 数据集**

使用UCSF Swallow Watch项目的195名头颈癌患者数据，共443次访视，包含MDADI、CTCAE-DIGEST评分以及一系列结构化临床变量。

**📈 对比分析**

与10个基线模型（仅临床、仅PRO、全量或复合得分等）在50随机种子下比较，所提模型AUC约0.885，recall约0.662，显著优于基线，尤其在召回率方面提升显著。

**⚠️ 局限性**

研究限于单中心回顾性数据，缺乏外部验证；仅预测二分类CTCAE-DIGEST，未考虑多级或时间序列信息；模型对阈值敏感，需进一步校准和前瞻验证。

---

## 391. Machine-Checked Arithmetic Bit Complexity of the Kannan-Bachem Smith Normal Form in Lean 4

**arXiv ID:** 2607.22524 | [PDF](https://arxiv.org/pdf/2607.22524v1)

**作者:** Junye Ji `[一作]` `[通讯]` (University of Washington), Junye Ji (University of Washington)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在Lean中对Kannan–Bachem的Smith标准形算法进行形式化，并生成可执行的变换矩阵U、V及其逆矩阵，完整证明UAV=S及其逆等性质；

**💡 创新点**

创新点在于提供了完整的可执行证书与逆证书、基于算术叶子精确成本的trace语义，并通过多层多项式闭包给出了输入二进制长度与运算/输出长度的固定多项式上界；

**🔧 技术方法**

采用了签名-幅值整数模型、结构化的Hermite/HNF步骤、Bezout块的归约、四乘积证书组合、递归pivot降阶以及精确的trace构造与闭包计算；

**📊 数据集**

使用的实验数据集为所有非奇异整数方阵（无特定尺寸或数值分布限制），并在Lean 4.32.0环境下进行验证与计数；

**📈 对比分析**

与已有Coq/Isabelle等线性代数形式化相比，本文的执行成本与输出长度上界证明更精确，且在自证执行的基础上提供了实际的算术步骤计数，性能上与论文所给的理论上界相符；

**⚠️ 局限性**

局限性包括仅处理非奇异方阵、未建模结构化/内存成本、未涵盖稀疏或增量更新、以及多项式上界指数不够紧凑。

---

## 392. PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest

**arXiv ID:** 2607.22518 | [PDF](https://arxiv.org/pdf/2607.22518v1)

**作者:** Olafur Gudmundsson `[一作]` (Pinterest, Inc.), Zhihua Zhang `[通讯]` (Pinterest, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

解决 Pinterest 行业规模搜索与推荐系统中的内容冷启动问题，提出全漏斗内容探索与去偏方案；

**💡 创新点**

创新点包括：①覆盖语料选择、检索、排序及最终效用的全漏斗方法；②系统层面去偏，降低对既有内容的偏好；③可扩展的测量框架，实现短期实验验证长期价值；

**🔧 技术方法**

采用 Thompson 采样/模型先验、内容嵌入（PinCLIP、Semantic ID）、特征填充、历史特征 dropout、特征交叉、UCB 与 Neural Linear UCB、校准层、实时流处理等技术；

**📊 数据集**

使用 Pinterest 海量用户与 Pin 数据，覆盖 Homefeed、Related Pins、Search 等多产品表面，实验采用 holdout 与 A/B 测试；

**📈 对比分析**

与旧系统对比，在 holdout 上年增量会话提升 24%/49%，购物会话提升 63%/29%，内容毕业提升 41%，实验指标（如 Under‑explored 内容参与量）提升至 18.5% 等，展示显著性能提升；

**⚠️ 局限性**

仍受限于冷启动内容稀缺、模型训练反馈循环、探索与短期留存权衡，以及多阶段优化调度的复杂性。

---

## 393. Explainable Reinforcement Learning for assisting Air Traffic Controllers

**arXiv ID:** 2607.22525 | [PDF](https://arxiv.org/pdf/2607.22525v1)

**作者:** Anduel Mehmeti `[一作]` (University of Campania Luigi Vanvitelli), Salvatore Venticinque `[通讯]` (University of Campania Luigi Vanvitelli)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在简化的二维航空管制仿真环境中，使用深度Q网络（DQN）训练智能体实现航班路径规划，避免禁飞区并到达目标；

**💡 创新点**

首次将梯度基注意力可视化（saliency map）引入强化学习决策过程，提供实时解释，弥合黑盒模型与人类操作员之间的理解鸿沟；

**🔧 技术方法**

技术包括DQN（含经验回放、目标网络、ε-贪婪探索）和梯度梯度导数生成的saliency map；

**📊 数据集**

使用自建的2D 40×40格点模拟环境（包含一个半径5海里圆形禁飞区）作为数据集；

**📈 对比分析**

通过奖励分解、轨迹分析与saliency演化来评估性能，实验表明智能体能在多数试验中安全通过禁飞区并成功到达目标，奖励累计显著高于随机策略；

**⚠️ 局限性**

局限性在于环境过于理想化、动作空间极其有限、仅采用模拟数据、解释仅基于单一梯度可视化，缺乏对更复杂真实空域及多机协作的验证。

---

## 394. Gridnberg: A Topography-Aware Pedestrian Routing Dataset for New York City

**arXiv ID:** 2607.22523 | [PDF](https://arxiv.org/pdf/2607.22523v1)

**作者:** Ariel Noyman `[一作]` `[通讯]` (Massachusetts Institute of Technology), Ariel Noyman (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个结合地形信息的纽约市行人网络数据集grid-n-berg，为每个网络节点分配局部平均高程并计算坡度相关成本。

**💡 创新点**

创新点在于将市级行人网络与局部地形点云融合，提供方向特定的坡度评分和可复现的路线成本模型，并将数据与代码公开发布。

**🔧 技术方法**

使用50米半径点云均值赋高程、累计上坡/下坡比例与分段阈值惩罚函数，并采用Dijkstra算法进行路由计算。

**📊 数据集**

基于NYCWalks行人网络（315,577条LineString）和2022年NYC Planimetric Database道路底部点（约387,566个）进行融合。

**📈 对比分析**

对比了三种成本剖面（仅距离、舒适导向、可达性敏感），通过示例路径展示路程与坡度权衡；覆盖99.24%源段，连通组件256个，表现可接受。

**⚠️ 局限性**

限制包括高程为未加权平均，可能混合垂直不同观测；50米搜索半径导致平滑与误差；坡度惩罚基于段级别，忽略局部极端斜率；网络连通性受高程覆盖限制。

---

## 395. The Regression Tax: Decomposing Why Skills Help and Hurt LLM Agents

**arXiv ID:** 2607.22520 | [PDF](https://arxiv.org/pdf/2607.22520v1)

**作者:** Darshan Tank `[一作]` (Sentient Labs), Baran Nama `[通讯]` (Sentient Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究评估了在LLM代理中加入程序化技能（skills）对任务成功率的影响，并揭示技能往往导致回归（regression）——即本来能成功的任务在添加技能后失败。

**💡 创新点**

创新点在于将技能效果拆分为增益（gain）与回归（regression），提出三种回归机制（技能描述渗透、基准失位、验证失位），并系统分析技能对任务不同阶段（grounding、method、verification）的影响。

**🔧 技术方法**

方法包括：基于配对比较的McNemar检验、使用执行轨迹对回归机制进行标注、对技能库进行三种生成器对比、对结果进行统计与置信区间估计。

**📊 数据集**

使用了两个办公自动化基准——OfficeQA-Pro（财务文件问答）和SpreadsheetBench（电子表格编辑），共计486个任务，分别在三套模型–抓手堆栈上执行，共5,832个任务-条件运行。

**📈 对比分析**

比较方法为：与无技能基准进行配对比较，记录增益与回归数量，计算净效益并进行统计显著性检验。结果显示，尽管某些技能库在整体pass率上提升，但回归占总增益的59%，净效益仅为229次成功转移；只有Claude Code sonnet‑4.6在SpreadsheetBench堆栈上显著提升。

**⚠️ 局限性**

局限性包括：基准仅覆盖办公自动化任务，难以推广到方法阶段为瓶颈的其他领域；模型与抓手捆绑，未分离模型与抓手的影响；回归机制标注依赖单位标注，缺乏交叉验证；统计显著性在多重比较校正后仅剩Claude Code sonnet‑4.6堆栈显著。

---

