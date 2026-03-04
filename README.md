# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-04 | 今日论文总数: 519

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Stabilized Adaptive Loss and Residual-Based Collocation for Physics-Informed Neural Networks

**arXiv ID:** 2603.03224 | [PDF](https://arxiv.org/pdf/2603.03224v1)

**作者:** Divyavardhan Singh `[一作]` (Sardar Vallabhbhai National Institute of Technology), Kishor Upla `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**通讯引用:** 1087 | [OpenAlex ID](https://openalex.org/A5047033264)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种将梯度平滑自适应损失平衡与基于残差的自适应采样相结合的PINN框架，并在低粘性Burgers方程和Allen–Cahn方程上验证其性能。

**💡 创新点**

创新点在于：① 通过指数平滑梯度范数来动态调整损失权重，防止权重崩溃；② 将自适应损失与残差驱动的采样统一实现；③ 在刚性非线性PDE上实现显著误差降低，首次系统比较两种技术的协同效应。

**🔧 技术方法**

使用自动微分、梯度归一化/平滑、指数滑动平均、残差采样重采样、Adam优化器，以及七层50节点的全连接神经网络。

**📊 数据集**

没有使用传统数据集；采用数值参考解（CFL安全的有限差分）作为基准；Allen–Cahn方程使用数值参考无解析解，主要通过边界误差和均方残差评估。

**📈 对比分析**

与标准PINN（固定权重、均匀采样）对比，采用相对L₂误差、边界误差和均方残差为评价指标；联合方法将Burgers相对误差从0.487降至0.272（≈44%提升），Allen–Cahn从0.0926降至0.0272（≈70%提升），边界误差下降1-2个数量级，均方残差保持低水平。

**⚠️ 局限性**

在Allen–Cahn实验中仍出现权重崩溃导致边界误差略高；最小权重阈值设置可能不足；方法对参数调节敏感；仅在两种PDE上验证，缺乏对更复杂/高维PDE的泛化验证。

---

## 2. Why Adam Can Beat SGD: Second-Moment Normalization Yields Sharper Tails

**arXiv ID:** 2603.03099 | [PDF](https://arxiv.org/pdf/2603.03099v1)

**作者:** Ruinan Jin `[一作]` (Ohio State University), Shaofeng Zou `[通讯]` (Arizona State University)

**通讯引用:** 1020 | [OpenAlex ID](https://openalex.org/A5012545205)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在经典的二阶矩（即方差）噪声模型下，理论上对Adam和SGD的高概率收敛性质进行比较，并给出了Adam优于SGD的正式证明；

**💡 创新点**

首次在相同的光滑+有界方差假设下，利用Adam的二阶矩归一化机制实现了高概率收敛率的分离，Adam的置信度依赖从δ⁻¹提升到δ⁻¹/²；

**🔧 技术方法**

采用停止时间技术、马尔可夫不等式、Burkholder–Davis–Gundy不等式以及对Adam内部的自适应二阶矩累加器的对数式控制；

**📊 数据集**

无（纯理论分析，无实验数据集）；

**📈 对比分析**

与SGD在相同假设下对比，证明Adam在高概率收敛率上优于SGD，Adam可达O(1/√(δT))，而SGD最低只能达到Ω(1/(δ√T))；

**⚠️ 局限性**

对Adam的去预处理（de-preconditioning）步骤损失了δ⁻¹/²的提升，导致最终收敛率与预处理能量的对数控制之间存在损失；

---

## 3. Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification

**arXiv ID:** 2603.02798 | [PDF](https://arxiv.org/pdf/2603.02798v1)

**作者:** Yichi Zhang `[一作]` (Tsinghua University), Mihaela van de Schaar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于临床指南的高风险AI代理验证框架GLEAN，用于对LLM驱动的诊断代理进行可靠的过程级正确性评估

**💡 创新点**

创新点在于将专业指南直接转换为每一步的对齐评分，采用顺序证据累积与贝叶斯逻辑回归校准，且通过主动验证（指南扩展和差异检查）动态提升置信度

**🔧 技术方法**

核心技术包括：提示式指南评分（LLM Judge）、多指南聚合、折扣式累计、贝叶斯逻辑回归校准、基于熵触发的主动验证

**📊 数据集**

使用MIMIC‑IV患者数据中的三种疾病（阑尾炎、胆囊炎、胰腺炎）以及公开的医学指南集合（epfl‑llm/guidelines）

**📈 对比分析**

与多种基线（P(), LLM-as-a-Judge, Self‑Consistency, Semantic Entropy, Self‑Verification, RAG‑Augmented, Med‑PRM, ORM）对比，GLEAN在AUROC、风险率、ECE和Brier得分上均优于对手，最高可达AUROC 0.986、Brier 0.052，且在Best‑of‑N选择中显著提升准确率

**⚠️ 局限性**

局限性包括对指南质量和完整性的高度依赖，主动验证需要额外计算与知识检索开销，对异常或无指南情境的适用性有限，并且在临床实际部署前仍需更广泛的多中心验证

---

## 4. Pulli Kolam: A Traditional South Indian Craft Practice for Representing Data

**arXiv ID:** 2603.02343 | [PDF](https://arxiv.org/pdf/2603.02343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 5. FinTexTS: Financial Text-Paired Time-Series Dataset via Semantic-Based and Multi-Level Pairing

**arXiv ID:** 2603.02702 | [PDF](https://arxiv.org/pdf/2603.02702v1)

**作者:** Jaehoon Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了大规模的金融文本-时间序列数据集 FinTexTS，并提出一种基于语义匹配和多层次配对的框架，用来将新闻文本与目标公司股票价格进行对齐。

**💡 创新点**

创新点主要有：①利用 LLM 对 SEC 报告进行结构化提取，得到公司上下文；②基于该上下文和 fine‑tuned 的嵌入模型实现语义匹配，能够检索即使未出现公司名称的相关新闻；③将新闻划分为宏观、行业、关联公司和目标公司四个层级，实现多层次配对，显著丰富了文本信息。

**🔧 技术方法**

技术手段包括：LLM（gpt‑4o‑mini/gpt‑5‑mini）用于 SEC 解析、新闻分类和摘要；Embedding 模型（Linq‑Embed‑Mistral）经过对比学习 fine‑tune；SBERT 进行文本表示；12 种时序预测模型（Autoformer、Crossformer、DLinear 等）用于评估；数据增强、前向填充等预处理方法。

**📊 数据集**

数据集：公开新闻来源（约 1 百万篇）配合 SEC 文件，构成 FinTexTS，覆盖 2019‑2023 期间 100 家市值最大的公司；此外还使用 LSEG 的 Machine Readable News（MRN）作为专有新闻来源做对比。

**📈 对比分析**

通过与无文本、关键词配对以及仅使用语义配对的基线对比，实验表明在所有 12 种模型中，语义配对均显著降低 MSE/MAE；进一步加入多层次文本后，性能进一步提升；使用 MRN 专有新闻时，绝大多数模型的 MSE/MAE 进一步下降，验证更高质量文本带来的收益。

**⚠️ 局限性**

局限性：①框架高度依赖 LLM 与 embedding 计算，成本和推理时延较高；②新闻来源仍有限，无法覆盖所有潜在影响事件；③多层次配对的效果受限于现有分类规则，未专门设计模型充分利用层级信息；④数据仅公开来源，可能存在版权和可再现性问题。

---

## 6. From Complex Dynamics to DynFormer: Rethinking Transformers for PDEs

**arXiv ID:** 2603.03112 | [PDF](https://arxiv.org/pdf/2603.03112v1)

**作者:** Pengyu Lai `[一作]` (Shanghai Jiao Tong University), Hui Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 44797 | [OpenAlex ID](https://openalex.org/A5051089032)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DynFormer，一种利用谱分解、Kronecker结构注意力与LGM混合变换的Transformer神经算子，用于高效逼近多尺度PDE解。

**💡 创新点**

创新点在于把物理动力学尺度分离融入Transformer架构，通过谱嵌入将低频模式截断、Kronecker注意力将全局复杂度从O(N^4)降至O(N^3)，并利用LGM乘法重构小尺度细节。

**🔧 技术方法**

采用傅里叶谱分解、Kronecker结构注意力、局部-全局乘法混合（LGM）变换以及演化式层（Hybrid Runge‑Kutta）等技术实现多尺度PDE建模。

**📊 数据集**

使用四个标准PDE基准：1D Kuramoto–Sivashinsky、2D Darcy Flow、2D Navier–Stokes 与 3D Shallow Water。

**📈 对比分析**

在严格按GPU内存对齐的实验中，与五个先进基准模型比较，DynFormer在所有基准上取得最高综合得分，误差降低可达95%，同时显著减少内存占用。

**⚠️ 局限性**

局限性包括仅适用于均匀格点和参数化傅里叶变换，尺度分离假设在高度非分离或纯线性/稳态情形下可能表现不佳，且对复杂几何和非规则网格的适应性有限。

---

## 7. COP-GEN: Latent Diffusion Transformer for Copernicus Earth Observation Data -- Generation Stochastic by Design

**arXiv ID:** 2603.03239 | [PDF](https://arxiv.org/pdf/2603.03239v1)

**作者:** Miguel Espinosa `[一作]` (University of Edinburgh), Mikolaj Czerkawski `[通讯]` (Asterisk Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了COP-GEN，一种多模态潜在扩散变压器，用于学习光学、雷达、DEM、LULC、时空等多种地球观测数据的联合分布，并实现任意对任意模态的条件生成、零射种模态翻译和波段填充；

**💡 创新点**

创新点在于：①将不同模态各自编码为潜在token后拼接成统一序列，②为每个模态独立设置扩散时间步，使得模型能够在不重新训练的情况下实现任意模态的条件生成；③采用峰值能力评估来衡量生成分布的支持度，突出分布多样性与物理一致性；

**🔧 技术方法**

使用技术包括：Transformer-based latent diffusion（U‑ViT）+ 变分自编码器（VAE）做模态编码；独立时间步嵌入与全局注意力；Flash Attention、EMA、混合噪声训练等；以及时空坐标与时间的特殊编码；

**📊 数据集**

使用的数据集为全球规模的多模态地球观测数据集（来源于MajorTOM），包含1017469个样本，涵盖Sentinel‑2 L2A、L1C、Sentinel‑1 RTC、DEM、LULC、经纬度、时间戳等；

**📈 对比分析**

与现有基线（如TerraMind、DiffusionSat等）通过峰值能力（oracle）评估进行对比；COP‑GEN在DEM、S2L1C、S2L2A、S1RTC等指标上均优于基线，且在多样性和物理一致性方面表现更好；但在地理定位（LatLon）上略逊于TerraMind；

**⚠️ 局限性**

局限性包括：①经纬度和时间条件对生成影响不显著；②训练时未使用随机模态dropout，导致对单模态边缘分布的学习不足；③模型目前未扩展到更高分辨率或更多传感器；④对评估指标仍需改进以更好捕捉生成分布。

---

## 8. Human-Certified Module Repositories for the AI Age

**arXiv ID:** 2603.02512 | [PDF](https://arxiv.org/pdf/2603.02512v1)

**作者:** Szilárd Enyedi `[一作]` (Technical University of Cluj-Napoca), Szilárd Enyedi `[通讯]` (Technical University of Cluj-Napoca)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5074346989)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

提出 Human‑Certified Module Repositories (HCMRs)，为 AI‑辅助开发提供可审计、可追溯、由人类审核的模块库，帮助构建可信软件。

**💡 创新点**

创新点在于将人类认证、SLSA 级 provenance、Sigstore 身份签名、契约化接口与 AI 组装约束整合到单一架构中，形成可组合、可验证的安全模块生态。

**🔧 技术方法**

技术包括 SLSA‑aligned provenance 生成、in‑toto/DSSE attestations、Sigstore 透明日志、人工安全审计、基于契约的 AI 组装引擎和可视化模块元数据。

**📊 数据集**

使用真实安全事件案例作为验证数据：SolarWinds、Log4Shell、XZ Utils 后门等供案例研究和攻击模型验证，未涉及公开数据集。

**📈 对比分析**

与现有生态（npm、PyPI、Azure Verified Modules 等）对比，强调 HCMR 在可信链、可审计性和 AI 组装安全方面的优势；未给出量化性能指标，主要通过案例分析和威胁模型验证。

**⚠️ 局限性**

局限性包括人类审计的可扩展性、契约规范的完善与自动化、维护者行为检测、跨生态集成复杂性，以及在大规模模块库中保持一致安全治理的挑战。

---

## 9. MIBURI: Towards Expressive Interactive Gesture Synthesis

**arXiv ID:** 2603.03282 | [PDF](https://arxiv.org/pdf/2603.03282v1)

**作者:** M. Hamza Mughal `[一作]` (Max Planck Institute for Informatics), Christian Theobalt `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 35073 | [OpenAlex ID](https://openalex.org/A5020664641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个在线、因果、实时的框架，用于在实时对话中生成与语音同步的全身手势和面部表情。

**💡 创新点**

创新点在于：①直接利用语音‑文本 LLM（Moshi）的内部 token 流，避免传统多步编码；②采用二维 Transformer（时序+运动学）拆分生成；③使用 Residual VQ‑VAE 对手部、上身、下身和面部分别进行分级离散编码；④加入对比学习与声态损失以提升手势多样性与表达性。

**🔧 技术方法**

技术包括：Moshi 语音‑文本 LLM、Residual VQ‑VAE、两层因果 Transformer（Temporal + Kinematic）、Gumbel‑Softmax 采样、InfoNCE 对比损失、声态二分类头、KV‑Cache 加速推理。

**📊 数据集**

主要数据集为 BEAT2（多说话人 23 名、单说话人 1 名）进行训练与评估；补充评估使用 Embody3D 数据集。

**📈 对比分析**

与非因果基线（EMAGE、RAG‑Gesture、CaMN）和实时基线（GestureLSM、MambaTalk）对比，实验表明在 FGD（Frechet Gesture Distance）与 BeatAlign（节奏对齐）指标上均取得了最佳成绩；在单说话人情形下与多说话人情形下都保持较低的延迟（≈36 ms）并在 L1-Divergence 与 Facial‑MSE 上也表现优秀。

**⚠️ 局限性**

局限性在于仅建模代理自身手势，未考虑用户的身体动态与双人交互语境，导致在多方交互或对话伴随者手势响应方面表现不足。

---

## 10. A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities

**arXiv ID:** 2603.02540 | [PDF](https://arxiv.org/pdf/2603.02540v1)

**作者:** Faiz Ghifari Haznitrama `[一作]` (Korea Advanced Institute of Science and Technology), Alice Oh `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3680 | [OpenAlex ID](https://openalex.org/A5054771988)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了NeuroCognition基准，将Raven矩阵、空间工作记忆和威斯康星卡片分类测试改编为文本与图像多模态形式，用以衡量LLM的抽象推理、工作记忆和认知灵活性。

**💡 创新点**

创新点包括：①将人类神经心理学测试迁移至LLM领域；②设计可扩展的多模态评测套件；③通过因子分析与现有benchmarks关联，揭示LLM通用能力的单因子结构。

**🔧 技术方法**

使用了多模态输入处理、链式思考与无推理两种策略、提示干预（模式提示、笔记记录）、因子分析和相关性分析等技术。

**📊 数据集**

使用的数据集包括RAVEN视觉矩阵数据、自动生成的文本矩阵、空间工作记忆的图形/文本版本以及WCST的图形/文本版本，并在GPT‑5、Gemini、Claude、Grok、GLM、Qwen等多种LLM上进行实验。

**📈 对比分析**

通过在文本、图像和混合模态下对不同难度进行零样本评估，并与人类基准及11项主流评测的平均得分进行相关性分析；结果显示LLM在文本上表现优异，但在图像和高难度任务中显著下降，部分任务甚至低于人类。

**⚠️ 局限性**

局限性包括：①样本量受限，导致结果稳健性不足；②假设人类神经心理学测试能直接衡量LLM需进一步验证；③缺乏对不同提示、上下文和LLM的广泛心理测量学验证。

---

## 11. Reproducing and Comparing Distillation Techniques for Cross-Encoders

**arXiv ID:** 2603.03010 | [PDF](https://arxiv.org/pdf/2603.03010v1)

**作者:** Victor Morand `[一作]` (Sorbonne Université), Benjamin Piwowarski `[通讯]` (Sorbonne Université)

**通讯引用:** 3316 | [OpenAlex ID](https://openalex.org/A5086752907)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并复现了多种跨编码器（cross‑encoder）蒸馏技术（MarginMSE、DistillRankNET、ADR‑MSE）以及多种监督目标（BCE、hinge、InfoNCE），在九种不同编码器骨干（BERT、RoBERTa、ELECTRA、DeBERTa‑v3、MiniLM‑L12、Ettin‑17/32/68/150、ModernBERT 等）上进行统一实验，提供完整的复现框架和模型检查点。

**💡 创新点**

① 在受控环境下系统评估蒸馏与监督目标的真实贡献，发现 InfoNCE 与 MarginMSE 两类相对/列表式目标在所有骨干和评估场景下均优于点式 BCE；② 证明优秀的蒸馏/监督目标可弥补模型规模缺陷，甚至超越更大骨干；③ 公开统一配置、代码与模型，方便后续研究者在相同实验平台上验证或扩展。

**🔧 技术方法**

Transformer 跨编码器；知识蒸馏（MarginMSE、DistillRankNET、ADR‑MSE）；监督学习目标（BCE、hinge、InfoNCE）；对比学习/列表式蒸馏；SPLADE‑v3‑DistilBERT 作为第一阶段检索器；PyTorch + HuggingFace Transformers；统一的评估脚本（nDCG@10）和多实验随机种子控制。

**📊 数据集**

训练集：MS MARCO v1（用于所有模型训练）。评估集：在域内（MS MARCO dev、TREC‑DL '19/'20）和域外（BEIR 13、LoTTE、Robust04）进行检索效果评测。

**📈 对比分析**

采用统一的候选池（SPLADE‑v3‑DistilBERT top‑1000）和 nDCG@10 作为指标，对 162 个实验（9 backbone × 6 loss × 3 seeds）进行统计。结果显示：InfoNCE 与 MarginMSE 在所有骨干与评估场景中均显著优于 BCE，且在 OOD 任务中优势更明显；DistillRankNET 与 ADR‑MSE 与传统监督相当，但不一定优于 MarginMSE；优秀的训练目标可以与更大骨干相当或更优，尤其在中等规模模型（如 Ettin‑32M）上表现突出。

**⚠️ 局限性**

1) 超参数仅在代理模型上调优，未针对每个骨干做细致调优；2) 蒸馏实验依赖特定教师（RankZephyr、cross‑encoder ensemble），教师质量变化未探究；3) 负样本来源不统一（BCE/hinge 用 MS MARCO 负样本，InfoNCE 用 ColBERTv2 hard negatives，MarginMSE 用 BM25 负样本），缺少完整的 ablation；4) 只使用单一第一阶段检索器（SPLADE‑v3‑DistilBERT）和固定候选深度，结果可能随检索器或深度改变；5) 未扩展到更大模型规模或其他蒸馏方案，限制了结论的普适性。

---

## 12. ScribeTokens: Fixed-Vocabulary Tokenization of Digital Ink

**arXiv ID:** 2603.02805 | [PDF](https://arxiv.org/pdf/2603.02805v1)

**作者:** Douglass Wang `[一作]` `[通讯]` (Independent Researcher), Douglass Wang (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ScribeTokens，将数字笔迹拆分为像素级方向步骤，使用固定10词表实现OOV-free、采样率不变的表示，并通过BPE压缩。

**💡 创新点**

创新点在于结合Bresenham算法与Freeman链码，构造仅10个基词的无OOV词表，既保证了语法健壮性，又实现了高效压缩与可生成性。

**🔧 技术方法**

采用Bresenham分解、Freeman链码、BPE压缩、Transformer（LLaMA）架构以及自监督的next‑ink‑token预训练。

**📊 数据集**

使用IAM‑OnLine Handwriting Database和DeepWriting两大手写数据集进行实验。

**📈 对比分析**

与传统向量（Point‑5）和其他token化方法（AbsTokens、RelTokens、TextTokens）对比，ScribeTokens在无预训练时识别CER已优于向量；预训练后在IAM 8.27%、DeepWriting 9.83%等指标上均取得最佳性能，生成任务在IAM上也达到最低CER。

**⚠️ 局限性**

局限在于仅验证英文手写、单一34M参数Transformer，缺乏对其他文字、任务和更大模型的泛化评估；预训练仅在同一数据集上进行，未探索大规模无标签数据。

---

## 13. E-variables and tests of randomness for distribution classes

**arXiv ID:** 2603.02492 | [PDF](https://arxiv.org/pdf/2603.02492v1)

**作者:** Georgii Potapov `[一作]` (Royal Holloway University of London), Yuri Kalnishkan `[通讯]` (Royal Holloway University of London)

**通讯引用:** 267 | [OpenAlex ID](https://openalex.org/A5048818681)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了e-variable-approximability概念，利用该概念构造了多种常见分布族（如均匀、泊松、正态、柯西）的e-变量，并给出了相应的随机性测试（inf‑projection）的显式下半可计算近似。

**💡 创新点**

创新点在于：①将Levin的随机性测试与统计假设检验统一起来，形成新的e-变量构造框架；②给出了可计算、可下半可计算的e-变量近似方法；③通过e-variable-approximability实现对复合假设的可组合e-变量，克服传统inf‑projection不可测或不可计算的缺陷。

**🔧 技术方法**

技术手段主要包括：可计算函数与下半可计算函数理论、指数族的参数化与最大似然估计、网格逼近（net）与插值（interpolation）技术、以及连续e-variable-approximability的平滑化方法。

**📊 数据集**

本文为理论性研究，不涉及实验数据集，所有结果均为严格的数学证明。

**📈 对比分析**

由于未进行实验评估，本文没有提供传统的性能对比；其“性能”体现在能否在给定分布族上构造合法且下半可计算的e-变量，以及能否获得显式的随机性测试上。

**⚠️ 局限性**

局限性包括：①对分布族要求可通过可数网格逼近，可能不适用于某些连续参数空间；②构造所需的常数因子（C）与网格精细度相关，可能影响实际使用；③缺乏对非可测或非可计算分布族的处理；④本文主要关注理论证明，缺乏实验验证与实际应用案例。

---

## 14. Scalable Mesh Coupling for Atmospheric Wave Simulation

**arXiv ID:** 2603.02971 | [PDF](https://arxiv.org/pdf/2603.02971v1)

**作者:** Hannes Brandt `[一作]` (Rheinische Friedrich-Wilhelms-Universität Bonn), Carsten Burstedde `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究实现了基于森林八叉树的可扩展网格耦合算法，用于在MAGIC与GEMINI两套气象波模拟器之间进行数据插值，以实现全局一致的解。

**💡 创新点**

创新点在于提出了一个无通信、可并行分区搜索的查询点插值框架，支持不同维度网格（2D、3D、3D外挤）双向、可交换的数据传递。

**🔧 技术方法**

技术实现结合了Forest-of-Octrees自适应网格、Clawpack求解器、用户自定义相交与插值回调以及非阻塞点对点通信，并采用多核MPI并行化。

**📊 数据集**

实验使用了从地表到约4000 km的三维气象波模拟数据，MAGIC负责从对流层到热层的声压波动，GEMINI负责上层等离子体响应，涵盖数千至数万个叶子网格。

**📈 对比分析**

通过在384核上运行2D AGW实验，耦合平均耗时0.04 s，MAGIC与GEMINI时间步长分别为0.83 s与0.71 s，耦合占总运行时间不到1%，并在12,288核规模上验证了良好的可扩展性。

**⚠️ 局限性**

限制主要包括耦合仍需在每次同步时执行，受负载不平衡影响；在极大规模时中心细化区域的分布可能导致重叠区域搜索效率下降。

---

## 15. Deterministic Edge Coloring with few Colors in CONGEST

**arXiv ID:** 2603.02689 | [PDF](https://arxiv.org/pdf/2603.02689v1)

**作者:** Joakim Blikstad `[一作]`, Tijn de Vos `[通讯]` (TU Graz)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

在分布式 CONGEST 模型中设计了一种确定性边缘着色算法，能够在极低（多项式对数）回合内为最大度为 Δ 的图得到 (1+ε)Δ+O(√log n) 颜色，且若 Δ≥c√log n 则使用不到 2Δ−1 颜色。该算法同样可改进 2Δ−1 颜色着色的回合数。它通过将在线边缘着色算法的低局部性与可 derandomize 的概率分析相结合，并利用冲突图调度、网络分解与度分割技术实现。

**💡 创新点**

创新点主要包括：
1. 证明在 Δ≥c√log n 时，(1+ε)Δ 颜色的确定性着色可在多项式对数回合完成；
2. 将在线着色算法的低局部性迁移到 CONGEST，首次在无超时限通信模型中实现低颜色数的确定性着色；
3. 使用潜在函数（pessimistic estimators）将概率分析局部化，从而在分布式环境下实现完全确定性；
4. 通过冲突图和度分割显著降低并行调度的回合数，实现 O(log^2.5 n + log^2Δ log n) 的复杂度。

**🔧 技术方法**

核心技术：
- 在线边缘着色算法（Blikstad‑Svensson‑Vintan‑Wajc）
- 潜在函数和条件期望 derandomization
- 低局部性（≤5）分析与冲突图调度
- 网络分解（(O(log n),O(log n))）
- 度分割与匹配分割
- 线图与距离‑ℓ 边缘着色
- 细粒度的消息压缩（将节点 ID 归约至 O(logΔ) 位）。

**📊 数据集**

无实验数据集，全部为理论证明与算法复杂度分析。

**📈 对比分析**

与之前最优的 O(log^8 n) 回合 2Δ−1 着色相比，本工作将复杂度降至 O(log^2.5 n + log^2Δ log n)，与理论下界 Ω(log n / loglog n) 的差距被压缩至多项式比例；在 Δ≥c√log n 范围内，使用不到 2Δ−1 颜色的情况是首次出现。对比随机算法（可达 1+Δ 颜色但需要随机），确定性算法在相同颜色数下实现了更低的回合数。

**⚠️ 局限性**

限制与开放点：
- 算法仅在 Δ≥c√log n 时成立，Δ 较小的情况仍需使用传统 2Δ−1 着色或更慢的算法；
- 需要节点无序标识或仅能比较相等性的假设；
- 证明依赖于复杂的潜在函数与大量本地化分析，实现成本较高；
- 对于更高 Δ，冲突图的度仍为 O(Δ^4)，在极大 Δ 时仍可能影响常数系数。

---

## 16. Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective

**arXiv ID:** 2603.03226 | [PDF](https://arxiv.org/pdf/2603.03226v1)

**作者:** Enea Monzio Compagnoni `[一作]` (University of Basel), Anastasiia Koloskova `[通讯]` (University of Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过随机微分方程(SDE)框架，对差分隐私SGD(DP‑SGD)和自适应DP‑SignSGD的收敛速度、隐私‑效用平衡以及最优学习率进行理论分析，并在真实任务中验证。

**💡 创新点**

①首次将SDE方法用于DP优化器，揭示隐私噪声与自适应机制交互产生的结构性差异；②提出两种实验协议（固定超参与最佳调参），说明在高隐私或大批噪声下自适应方法更优；③证明DP‑SGD最优学习率随ε线性缩放，而DP‑SignSGD近乎与ε无关。

**🔧 技术方法**

SDE近似、隐私分析（Gaussian机制+采样放大）、光滑/μ‑PL假设下的收敛界、实验验证。

**📊 数据集**

合成二次凸函数；真实数据集IMDB和StackOverflow的逻辑回归任务。

**📈 对比分析**

在不同ε和批噪声下对DP‑SGD、DP‑SignSGD和DP‑Adam进行对比实验；结果与理论一致：在高隐私或批噪声大时自适应方法收敛更快、效用更好；在最佳调参时两者收敛邻域相同，但DP‑SGD的学习率需随ε调节，导致调参成本更高。

**⚠️ 局限性**

仅考虑DP‑SGD和DP‑SignSGD；理论假设为光滑凸/μ‑PL、梯度噪声为高斯或学生t；未对更复杂网络或DP‑Adam等高级自适应优化器进行完整理论分析；SDE近似误差与模型规模相关，未完全覆盖大规模深度模型。

---

## 17. TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration

**arXiv ID:** 2603.02943 | [PDF](https://arxiv.org/pdf/2603.02943v1)

**作者:** Benlei Cui `[一作]` (Alibaba Group), Haiwen Hong `[通讯]` (Alibaba Group)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5045259909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 Padé 近似的轨迹一致特征预测框架（TC‑Padé），用于在低步长（20–30 步）下加速扩散模型采样。

**💡 创新点**

创新点在于：①使用 Padé 近似对残差进行预测，捕捉非线性阶段切换；②自适应系数调节与轨迹稳定性指标（TSI）结合，实现时刻跳过与重算的动态决策；③按早期、中期、后期三阶段分配不同的预测策略。

**🔧 技术方法**

技术核心包括 Padé 近似、残差缓存、TSI 轨迹稳定性指标、步长感知预测策略，以及与量化等加速手段的协同。

**📊 数据集**

在 COCO 2017（文本‑图像）、VBench‑2.0（文本‑视频）和 ImageNet（类别‑图像）等公开数据集上进行评测。

**📈 对比分析**

与 ToCa、Δ‑DiT、TeaCache、TaylorSeer 等主流缓存与预测方法比较，TC‑Padé 在 FLUX.1‑dev 上实现 2.88× 的加速，Wan2.1 上 1.72×，DiT‑XL/2 上 1.46×，且 FID、CLIP、VBench 等质量指标仅略有下降（≤5%）。

**⚠️ 局限性**

局限性包括：需要手动设定 TSI 阈值；主要针对低步长场景，对极大步长或不同扩散调度的鲁棒性尚待验证；残差缓存仍引入额外内存开销。

---

## 18. Towards an Incremental Unified Multimodal Anomaly Detection: Augmenting Multimodal Denoising From an Information Bottleneck Perspective

**arXiv ID:** 2603.02629 | [PDF](https://arxiv.org/pdf/2603.02629v1)

**作者:** Kaifang Long `[一作]` (Northeastern University), Guoyang Xie `[通讯]` (CATL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种增量统一多模态异常检测框架 IB-IUMAD，能在不断加入新物体时保持对已学知识的记忆。

**💡 创新点**

创新点在于同时引入 Mamba 解码器消除跨物体伪特征干扰，并利用信息瓶颈融合模块剔除冗余信息，从而显著降低灾难性遗忘。

**🔧 技术方法**

采用 Mamba 解码器、信息瓶颈融合模块、重建损失、交叉熵损失和 KL 散度联合训练，实现跨模态特征去噪与增量学习。

**📊 数据集**

在工业缺陷检测数据集 MVTec 3D-AD 与 Eyecandies 上进行实验，分别包含 RGB 与深度两种模态。

**📈 对比分析**

与 IUF、CDAD 以及多种统一 MAD 基线对比，IB-IUMAD 在 I‑AUROC、AUPRO 方面提升 2–4%，忘记率降低 1–5%，且显著压缩显存（44×）并提升帧率（约 21 FPS）。

**⚠️ 局限性**

局限性包括对仅 RGB/深度数据的依赖、增量步长受限、以及对更复杂模态融合和在线学习场景的适用性尚未充分验证。

---

## 19. ExpGuard: LLM Content Moderation in Specialized Domains

**arXiv ID:** 2603.02588 | [PDF](https://arxiv.org/pdf/2603.02588v1)

**作者:** Minseok Choi `[一作]` (KAIST AI), Jungmin Son `[通讯]` (KakaoBank Corp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了ExpGuard，一种针对金融、医疗和法律等高风险领域的专用安全防护模型，并配套构建了相应的大规模标注数据集与专家验证的基准集。

**💡 创新点**

创新点包括①针对专业术语的领域知识挖掘与生成管道，实现高质量的域特定有害提示；②利用多模型共识与链式推理进行自动标签，随后由领域专家进一步审核；③在模型训练中融合了领域、真实交互与人工撰写的数据，显著提升对技术性攻击的检测。

**🔧 技术方法**

技术上主要采用GPT‑4o进行域词抽取与提示生成，Mistral‑7B‑Instruct用于生成回复，Gemma‑3‑27B‑IT产生拒绝；通过Claude‑3.7‑Sonnet、Gemini‑2.0‑Flash、Qwen2.5‑Max三模型进行多维度标签投票；模型训练基于Mistral‑7B‑v0.3并进行多任务二分类学习。

**📊 数据集**

使用的主要数据集为58,928条标注的域特定提示与对应回应，拆分为56,653条训练集（含19,907条域特定样本）以及2,275条专家验证的测试集。

**📈 对比分析**

实验中将ExpGuard与八个公开基准及多种闭源/开源守护模型进行对比，Prompt F1达93.3%，Response F1 92.7%，在财务、医疗、法律子域分别领先WildGuard 8.9%和15.3%，在公共安全基准上亦保持领先或相当水平。

**⚠️ 局限性**

局限性主要体现在仅覆盖英语内容、训练数据包含合成样本且可能无法完全反映真实动态用户请求，且目前仅针对三大领域，未来需扩展至多语言及其他专业场景，并持续更新以应对演化的攻击手段。

---

## 20. From Visual to Multimodal: Systematic Ablation of Encoders and Fusion Strategies in Animal Identification

**arXiv ID:** 2603.02270 | [PDF](https://arxiv.org/pdf/2603.02270v1)

**作者:** Vasiliy Kudryavtsev `[一作]` (Technical University of Communication and Informatics), Alexander Ryzhkov `[通讯]` (Avito)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了统一的大规模训练语料库（约1.9M张图，695k个个体）并系统性评估了多种视觉编码器和多模态融合策略，提出将视觉特征与生成的文本描述结合的跨模态识别框架

**💡 创新点**

创新点在于（1）首次统一多来源数据并给出标准训练/评估协议；（2）利用自动生成的文本描述实现跨模态融合；（3）在多模态设置下引入加权门控融合，显著提升排名性能

**🔧 技术方法**

采用对比式（triplet）+方差正则化的损失；视觉编码器包括 CLIP‑ViT‑Base、DINOv2‑Small、SigLIP‑Base/2‑Base/‑Giant、Zer0int‑CLIP‑L；文本编码器采用 E5‑Base/Small/Small‑v2、BERT；多模态融合包括简单拼接、交叉注意力、加权文本、门控融合

**📊 数据集**

使用自采集的 Pet911.ru、Telegram 频道图像与公开基准（Dogs‑World、LCW、PetFace、Cat Individual Images、DogFaceNet）进行训练与评估

**📈 对比分析**

与多种基线（MiewID‑msv3、MegaDescriptor 系列、BioCLIP）对比，最佳配置 SigLIP2‑Giant + E5‑Small‑v2 + gating 在整体测试集上实现 ROC AUC≈0.9912、EER≈0.0378、Top‑1≈84.3%，在 Cat Individual 和 DogFaceNet 上也保持领先；在 Top‑k 排名上可提升约2–4%

**⚠️ 局限性**

主要局限包括：依赖自动生成的文本描述（缺乏对真实人类查询的鲁棒性）；模型规模巨大，推理成本高，难以部署到资源受限设备；自采集数据存在标签噪声；由于隐私限制，原始数据无法公开共享

---

## 21. NExT-Guard: Training-Free Streaming Safeguard without Token-Level Labels

**arXiv ID:** 2603.02219 | [PDF](https://arxiv.org/pdf/2603.02219v1)

**作者:** Junfeng Fang `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60869 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无监督的实时安全保障框架 NExT-Guard，将现有的后置安全模型升级为流式安全模型。

**💡 创新点**

创新点在于利用稀疏自编码器（SAE）解码 LLM 隐藏层中隐含的安全信号，既不需要 token‑级标注，也不需要再训练，直接实现实时干预。

**🔧 技术方法**

使用稀疏自编码器进行特征提取、离线对比分析挑选安全相关特征，并在推理阶段用加权特征融合计算风险分数。

**📊 数据集**

采用公开安全基准数据集进行评估，包括 Aegis、Aegis2.0、SimpST、SafeRLHF、BeaverTails 等。

**📈 对比分析**

通过与主流后置安全模型（LlamaGuard、WildGuard 等）和流式安全模型（SCM、Kelp、Qwen3Guard‑Streaming 等）在 prompt/response 分类上的 F1 比较，NExT-Guard 在 prompt 上平均 F1 达 90.8，response 上平均 F1 84.3，分别比流式基线高 6–7 分，且优于大多数后置模型。

**⚠️ 局限性**

限制主要包括：实验集中在 Qwen 系列模型，尚未验证对其他后置保护模型的泛化；缺乏对不同 SAE 变体和特征融合策略的更细粒度分析。

---

## 22. How to Model AI Agents as Personas?: Applying the Persona Ecosystem Playground to 41,300 Posts on Moltbook for Behavioral Insights

**arXiv ID:** 2603.03140 | [PDF](https://arxiv.org/pdf/2603.03140v1)

**作者:** Danial Amin `[一作]` (University of Vaasa), Bernard J. Jansen `[通讯]` (Qatar Computing Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

生成并验证Molbbook平台AI代理的对话型人物模型，并在模拟讨论中测试其行为一致性。

**💡 创新点**

将Persona技术扩展至非人类代理生态系统，提供方法生成和验证AI代理的人物类型，并揭示表面一致与操作意义不符的现象。

**🔧 技术方法**

使用Persona Ecosystem Playground (PEP)框架，结合k-means聚类、MiniLM嵌入、检索增强生成（RAG）与GPT‑4o、Pinecone向量数据库、LangChain/Graph以及余弦相似度和RQE等评估手段。

**📊 数据集**

41,300条Molbbook AI代理帖子（标题+内容）及其元数据。

**📈 对比分析**

通过交叉人格验证（自集余弦相似度0.71 vs 0.35，t检验p<.001），RQE为0.68；在9回合模拟中人格归因准确率0.75，显著高于随机，但存在个别人格（如Existentialist）归因率低0.33。

**⚠️ 局限性**

仅基于单一平台数据、单一LLM生成、聚类与属性验证仅靠语义相似度，缺乏立场层面的确认；模拟短时且仅涉及一个主题，且未探究混合人类-代理环境下的行为差异。

---

## 23. Logics and Type Theory: essays dedicated to Stefano Berardi on the occasion of his 1000000th birthday

**arXiv ID:** 2603.02912 | [PDF](https://arxiv.org/pdf/2603.02912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 24. What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty

**arXiv ID:** 2603.02491 | [PDF](https://arxiv.org/pdf/2603.02491v1)

**作者:** Aran Nayebi `[一作]` (Carnegie Mellon University), Aran Nayebi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1727 | [OpenAlex ID](https://openalex.org/A5058188874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

在不确定环境下，证明了平均误差（average‑case regret）低的智能体必须具备可预测的内部状态；

**💡 创新点**

首次将平均误差约束转化为对内部表示（belief/state、记忆、模块化）必要性的定量选择定理；

**🔧 技术方法**

采用“赌注”归约、PSR（预测状态表示）、margin‑based regret 分解等理论工具；

**📊 数据集**

未使用任何公开数据集，完全基于理论证明；

**📈 对比分析**

方法主要通过理论对比，未进行实验性能评估；在已有理论结果基础上补充了必要性视角；

**⚠️ 局限性**

仅适用于平均误差、非最优/随机策略；要求评估分布具备足够信息量；无法恢复三级反事实；对极端或对抗性任务缺乏保障。

---

## 25. Valet: A Standardized Testbed of Traditional Imperfect-Information Card Games

**arXiv ID:** 2603.03252 | [PDF](https://arxiv.org/pdf/2603.03252v1)

**作者:** Mark Goadrich `[一作]` (Hendrix), Éric Piette `[通讯]` (UCLouvain)

**通讯引用:** 1111 | [OpenAlex ID](https://openalex.org/A5061767166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为Valet的标准化测试平台，收录21种传统不完全信息扑克牌游戏，并使用RECYCLE描述语言对其规则进行编码；随后利用CardStock系统对每个游戏进行分支因子、游戏长度、信息流和得分分布等经验评估。

**💡 创新点**

①提供跨实现统一的规则集，解决了多框架间不一致的游戏实现问题；②聚焦不完全信息扑克牌游戏的多样性，填补了传统游戏基准缺失的空白；③系统化评估游戏属性，为算法比较与分类研究提供数据支持。

**🔧 技术方法**

使用RECYCLE卡牌描述语言编码规则；在CardStock通用游戏平台上进行模拟；采用蒙特卡洛树搜索（MCTS）与随机玩家进行对比实验。

**📊 数据集**

Valet测试集（21款传统扑克游戏规则），并在CardStock上生成200次随机+MCTS仿真，收集分支因子、游戏长度、信息流和得分分布等统计数据。

**📈 对比分析**

通过将MCTS玩家与随机玩家在每款游戏中各进行100次仿真，计算分支因子、游戏长度、得分分布等指标；实验表明MCTS在大多数游戏中明显优于随机玩家，性能差距取决于游戏机制和信息结构。

**⚠️ 局限性**

仅涵盖单局游戏；未考虑人类玩家策略；对某些高度随机或信息隐蔽性强的游戏，MCTS优势不明显；未覆盖现代卡牌游戏的复杂机制（如建牌、合作玩法）。

---

## 26. ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation

**arXiv ID:** 2603.03279 | [PDF](https://arxiv.org/pdf/2603.03279v1)

**作者:** Xialin He `[一作]` (University of Illinois Urbana Champaign), Liang-Yan Gui `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 ULTRA，统一的多模态控制框架，能够在没有参考轨迹的情况下通过感知和稀疏目标实现全身行走-操控。

**💡 创新点**

创新点在于：①基于物理驱动的神经重定向算法，可在大规模人机交互数据上生成可执行、接触一致的机器人轨迹；②将高质量教师轨迹蒸馏为单一多模态学生策略，支持从精细轨迹追踪到稀疏目标的平滑切换；③通过变分技能瓶颈和 RL 微调提升在感知噪声和分布外任务下的鲁棒性。

**🔧 技术方法**

主要技术包括：强化学习（PPO）用于重定向与教师训练；Transformer 结构与可用性掩码的多模态编码；变分潜在瓶颈与 KL 正则化；RL 微调与 DAgger 结合的蒸馏策略；物理约束的仿真优化。

**📊 数据集**

使用 OMOMO 人机交互 MoCap 数据集，配合 Unitree G1 机器人以及自定义的盒子、手提箱等物体，进行规模化重定向和仿真训练。

**📈 对比分析**

与基准（PHC、GMR、OmniRetarget、HDMI）相比，ULTRA 在轨迹跟踪成功率、物体误差、接触稳定性上均明显优于；在稀疏目标和 egocentric 感知下，RL 微调提升 OOD 成功率 2–3 倍；在真实 Unitree G1 上，Dense 追踪 73% 成功，稀疏目标 80–90%（MoCap）与 50–60%（egocentric）成功率。

**⚠️ 局限性**

局限性包括：对深度相机提取点云的鲁棒性依赖较高；在极端力学扰动或高摩擦缺口时抓取可能失效；训练仍需要大量仿真和人机交互数据，跨平台迁移时可能出现域差异。

---

## 27. Timehash: Hierarchical Time Indexing for Efficient Business Hours Search

**arXiv ID:** 2603.02941 | [PDF](https://arxiv.org/pdf/2603.02941v1)

**作者:** Jinoh Kim `[一作]` (Naver Corporation), Jaewon Son `[通讯]` (Naver Corporation)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种用于业务时段搜索的层级多分辨率时间哈希索引算法，并实现了高效的索引生成与查询。

**💡 创新点**

创新点在于使用可自定义层级的层级分解，将时间区间映射为少量可读数字键，既保持分钟级精度，又大幅压缩索引量。

**🔧 技术方法**

采用层级分解、递归生成时间键、倒排索引、数据驱动的层级优化，并在 C++/Go 中实现，支持复杂模式如休息、跨午夜、24 小时等。

**📊 数据集**

在生产环境 12.6 余百万商家记录上构建合成数据，仿真真实的开始/结束时分布、休息频率与营业时长。

**📈 对比分析**

与单分辨率（1min、5min、1h）及范围过滤基线对比，评估指标为索引词数、构建时间、查询延迟和精度。该方法平均每条记录仅 5.6 词，索引量相较 1min 减少 99.1%，构建速度 68× 快，查询延迟 7–10 μs，精度 100%。

**⚠️ 局限性**

仅适用于每日时段（00:00–23:59），分钟级精度；需要手工或数据驱动的层级选择，跨天/周模式需额外前缀；对秒级需求需扩展编码。

---

## 28. Large Language Model-Enhanced Relational Operators: Taxonomy, Benchmark, and Analysis

**arXiv ID:** 2603.02537 | [PDF](https://arxiv.org/pdf/2603.02537v1)

**作者:** Yunxiang Su `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**通讯引用:** 7667 | [OpenAlex ID](https://openalex.org/A5057864403)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的LLM增强关系运算符（LRO）分类框架，并针对单一LRO与多LRO场景构建了全面的基准测试集。

**💡 创新点**

创新点在于：①系统性对齐并归纳现有LRO的操作逻辑、粒度与实现变体；②设计覆盖27个真实数据库的290条单LRO、60条多LRO查询的基准；③通过对比不同实现方法，给出LRO的最佳实践与设计权衡。

**🔧 技术方法**

核心技术包括基于大语言模型的语义推理、Prompt工程（含CoT与例子）、批量与逐元素调用策略，以及对LRO执行的自动与手工规划。

**📊 数据集**

使用了27个来自BIRD、Magellan、NextiaJD等公开数据库的真实数据，涵盖体育、电影、软件等10个领域；基准涵盖290条单LRO查询与60条多LRO查询。

**📈 对比分析**

实验比较了多种LRO实现和多LRO系统（Binder、SUQL、LOTUS‑TAG等），并以Exact Match、Precision/Recall/F1、HR@k等指标评估。最佳实践在GPT‑5上达成86.67%准确率，自动规划系统低于15%，手工规划系统最高约55%。

**⚠️ 局限性**

主要局限包括：①仅聚焦关系数据，未覆盖多模态或非结构化场景；②对大规模输入的上下文窗口限制导致可扩展性受限；③多LRO系统的规划与实现仍高度依赖人工，自动规划效果差；④部分LRO实现缺失或不完整，影响整体评测覆盖。

---

## 29. SEHFS: Structural Entropy-Guided High-Order Correlation Learning for Multi-View Multi-Label Feature Selection

**arXiv ID:** 2603.03022 | [PDF](https://arxiv.org/pdf/2603.03022v1)

**作者:** Cheng Peng `[一作]` (Jilin University), Weiping Ding `[通讯]` (City University of Macau)

**通讯引用:** 17211 | [OpenAlex ID](https://openalex.org/A5069969191)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种面向多视图多标签特征选择的框架SEHFS，利用结构熵引导学习高阶特征相关性并融合信息论与矩阵方法实现特征冗余抑制与全局视图重构。

**💡 创新点**

创新点：
- 引入结构熵作为正则化项，将特征图转化为结构熵最小化的编码树，能捕捉高阶非线性相关并消除冗余；
- 通过共享语义矩阵与视图特定贡献矩阵共同构建全局视图矩阵，兼顾一致性与互补性；
- 采用信息-矩阵融合框架，统一优化分类损失、结构熵、重构误差与图拉普拉斯正则化，实现全局‑局部协同学习。

**🔧 技术方法**

技术：
- 结构熵最小化与编码树构造；
- 互信息矩阵计算与特征图构建；
- 共享语义矩阵（k近邻高斯相似度）与视图投影矩阵；
- 交替优化（乘法梯度下降、投影梯度下降、二次规划）；
- 图拉普拉斯正则化、矩阵重构损失。

**📊 数据集**

使用了八个公开多视图多标签数据集：EMOTIONS、YEAST、VOC07、MIRFlickr、SCENE、OBJECT、Corel5K、IAPRTC12，涵盖从图像情感到基因功能预测、图像检索等不同领域。

**📈 对比分析**

与七种基线方法（DHLI、GRAFS、MSFS、SRFS、SPLDG、MSSL、MIFS）在AP、Coverage、Hamming Loss、Ranking Loss四个评估指标上进行10折交叉验证比较。SEHFS在87.5%指标上获得最佳结果，且在所有数据集上至少排名第二；在HL指标上始终取得最优；在大多数多视图数据集（SCENE、OBJECT、Corel5K、IAPRTC12）平均提升约7.24%。

**⚠️ 局限性**

局限性：
- 需要多次矩阵运算，时间复杂度为O(T(n²d + nd²))，对大规模数据仍有计算瓶颈；
- 依赖参数α、β、γ、λ的经验调优，虽然对性能影响有限但仍需预先设定；
- 当前方法假设所有视图完整，未处理缺失视图或噪声标签的情况。

---

## 30. Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees

**arXiv ID:** 2603.02633 | [PDF](https://arxiv.org/pdf/2603.02633v1)

**作者:** Mohammed Nowaz Rabbani Chowdhury `[一作]` (Rensselaer Polytechnic Institute), Meng Wang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 42289 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需重新训练的异构计算框架，将噪声敏感的专家和密集模块在数字加速器上执行，其余专家在模拟内存计算（AIMC）硬件上运行。

**💡 创新点**

创新点在于：① 用最大神经元范数（MaxNNScore）作为理论上可证明的指标来识别对权重编程噪声敏感的专家；② 将密集激活模块（如注意力层）优先放在数字端；③ 提供了完整的理论泛化保证，证明异构方案在面对编程噪声时能保持正确率。

**🔧 技术方法**

采用模拟内存计算（PCM/ResRAM）进行矩阵-向量乘法；使用数字计算实现高精度；通过理论分析、梯度下降训练、以及AIMC噪声模型（权重编程噪声和DAC-ADC量化噪声）进行评估。

**📊 数据集**

在两大MoE大模型 DeepSeekMoE（16B）和 OLMoE（7B）上，使用 8 项 LLM 评测任务（PIQA、ARC‑Easy/Challenge、BoolQ、HellaSwag、WinoGrande、MathQA、MMLU）进行验证。

**📈 对比分析**

与全数字（FP‑16）和全模拟两种极端做对比；实验表明：在噪声较大时，放置12.5%–25% 高 MaxNNScore 专家于数字端即可恢复大部分性能，保持相对较高的吞吐量与能效；在低噪声下，全模拟方案能效最优但准确率最低。

**⚠️ 局限性**

局限性包括：① 需要预先评估专家的最大神经元范数，依赖于训练完成后静态分析；② 对于极端硬件噪声或不同类型的非理想性（如漂移、温度变化）仍需进一步实验验证；③ 目前仅针对 MoE Transformer 结构，其他稀疏模型或任务的通用性尚待探索。

---

## 31. Real-Time Generative Policy via Langevin-Guided Flow Matching for Autonomous Driving

**arXiv ID:** 2603.02613 | [PDF](https://arxiv.org/pdf/2603.02613v1)

**作者:** Tianze Zhu `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**通讯引用:** 19590 | [OpenAlex ID](https://openalex.org/A5100747108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了一种基于流匹配与Langevin动力学的低延迟生成策略DACER-F，用于实时自动驾驶决策与控制。

**💡 创新点**

创新点在于将流匹配用于在线RL的策略表征，并设计Langevin动力学生成动态目标分布，既解决了生成策略高推理延迟，又提供了在线学习所需的目标分布。

**🔧 技术方法**

使用了流匹配、Langevin动力学采样、双Q网络、经验回放以及相对梯度下降等技术。

**📊 数据集**

在多车道高速与交叉路口的程序化仿真环境以及DeepMind Control Suite（Humanoid-stand、Humanoid-walk、Dog等）上进行评估。

**📈 对比分析**

与DSAC、DACER等SOTA基线在驾驶仿真中对比，DACER-F平均奖励提升28-34%，碰撞率降低，推理时间降84%，并在DMC六项任务中取得最高TAR。

**⚠️ 局限性**

仍依赖Q函数估计，可能在极端稀疏奖励或高维动作空间中受限，且缺乏对真实车载硬件的验证与鲁棒性评估。

---

## 32. Highly Incremental: A Simple Programmatic Approach for Many Objectives (Extended Version)

**arXiv ID:** 2603.02405 | [PDF](https://arxiv.org/pdf/2603.02405v1)

**作者:** Philipp Schröer `[一作]`, Joost-Pieter Katoen `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在概率程序中引入 reward 语句，并提出一种基于奖励差分的程序变换，使得多种量化目标（如期望运行时间、运行时间的高阶矩、阈值分布、超预算期望等）均可归结为传统的弱前期望计算；

**💡 创新点**

创新点在于将所有目标统一映射到单一的奖励机制与程序变换，避免为每个目标单独设计推理体系，且该变换利用增量差分实现对函数变换奖励的精确建模；

**🔧 技术方法**

使用的技术包括：概率程序的弱前期望语义、reward 语句的语义定义、基于增量差分的程序变换、Caesar 证明器自动化验证、以及对马尔科夫链的奖励变换证明；

**📊 数据集**

本文未使用传统意义上的数据集，而是通过若干示例程序（如 Web 服务器重试、随机行走等）验证方法的有效性；

**📈 对比分析**

方法通过与已有的专用目标推理规则（如运行时间、二阶矩等）对比，展示了在相同工具下即可完成多目标分析，实验表明在 Caesar 中的推导时间与传统单目标推理相当；

**⚠️ 局限性**

局限性包括：目前仅处理无非确定性的概率程序；对于非单调或需负奖励的情况需额外处理；奖励差分变换在某些目标（如超预算期望）下仍需手工拆分循环或构造额外变量。

---

## 33. SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding

**arXiv ID:** 2603.02548 | [PDF](https://arxiv.org/pdf/2603.02548v1)

**作者:** Sheng Ye `[一作]` (Tsinghua University), Yong-Jin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 11067 | [OpenAlex ID](https://openalex.org/A5008076279)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于双分支的前向式3D高斯渲染框架SemGS，能够从稀疏视角的彩色图像快速推断语义场景并生成语义地图。

**💡 创新点**

融合颜色与语义的双高斯表示，并在Swin Transformer中注入相机位姿的注意力机制以及区域平滑损失，以实现跨场景的可泛化语义推理。

**🔧 技术方法**

使用共享CNN和Swin Transformer的双分支特征提取、相机感知注意力、成本体积深度回归、双高斯解码与光栅化渲染，以及区域平滑正则。

**📊 数据集**

在ScanNet、ScanNet++、Replica等数据集上进行训练和评测。

**📈 对比分析**

与S-Ray、GSNeRF等基准对比，SemGS在mIoU、像素准确率和FPS上均取得显著提升，mIoU提升约30%且FPS提升十倍以上。

**⚠️ 局限性**

对相机标定误差敏感，且在极端域间（如户外动态场景）泛化仍有待提升。

---

## 34. Quantum-Inspired Fine-Tuning for Few-Shot AIGC Detection via Phase-Structured Reparameterization

**arXiv ID:** 2603.02281 | [PDF](https://arxiv.org/pdf/2603.02281v1)

**作者:** Kaiyang Xing `[一作]` (Anhui University), Guoping Guo `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23884 | [OpenAlex ID](https://openalex.org/A5080874179)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了两种基于 LoRA 的量子/经典相位增强微调方法 Q-LoRA 与 H-LoRA，用于少样本 AI 生成内容检测。

**💡 创新点**

创新点在于将 QNN 的相位感知与范数约束这两种结构先验迁移到纯经典 Hilbert 变换中，既保留了量子方法的泛化优势，又显著降低计算成本。

**🔧 技术方法**

使用了低秩适配器 LoRA、轻量级量子神经网络、Hilbert 变换以及 CLIP/Whisper 等大模型作为基础。

**📊 数据集**

采用了 Stable Diffusion、Midjourney、Wukong、VQDM、Glide、ADM 等图像生成数据集，以及 ASVspoof 2019 LA 语音伪造数据集进行实验。

**📈 对比分析**

通过十次随机种子重采样对比标准 LoRA、Q-LoRA 与 H-LoRA，在 200、400、800 例少样本设置下评估 ACC、AUC、PR、RE 等指标，H-LoRA 在 200 例时 ACC 提升约 5% 以上，且与 Q-LoRA 的性能几乎持平。

**⚠️ 局限性**

主要局限在于 Q-LoRA 的量子仿真开销极大（训练时长 > 2000 秒/epoch），并且该方法在更大规模、多模态或更高维度任务中的可扩展性和稳定性尚未得到验证。

---

## 35. Toward Early Quality Assessment of Text-to-Image Diffusion Models

**arXiv ID:** 2603.02829 | [PDF](https://arxiv.org/pdf/2603.02829v1)

**作者:** Huanlei Guo `[一作]` (Southern University of Science and Technology), Bingyi Jing `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文本到图像扩散模型中，通过在早期阶段对 denoiser 的中间激活做轻量级探针（Probe-Select）来预测最终图像质量，并据此在生成过程中提前终止低质量轨迹，显著降低计算成本。

**💡 创新点**

创新点在于发现并利用扩散过程早期激活中稳定的结构信号（如物体布局），设计了一个可插拔、无须改动原模型的探针网络，并通过列表排序与对比学习双重损失实现对多种评估指标的高质量预测。

**🔧 技术方法**

技术包括：扩散模型内部激活提取、轻量级视觉编码器与投影头、列表排序损失、InfoNCE 对齐损失、以及基于早期预测的选择式采样策略。

**📊 数据集**

使用 MS‑COCO 语料下的 100k 唯一描述，针对 Stable Diffusion 2/3.5（M/L）和 FLUX.1‑dev 等多种 backbone 生成 500k 条生成轨迹进行训练与评估。

**📈 对比分析**

与传统的完整生成后评估（ImageReward、HPSv2.1 等）相比，Probe‑Select 在 20% 采样点即可获得 Spearman 相关 ≥0.7，选择性生成可将采样成本降低约 60% 同时将 ImageReward 提升 1–2 倍（如 SD3‑L‑IR 从 1.12 提升至 1.83），并在多种评估指标上保持或提升性能。

**⚠️ 局限性**

局限性包括：对不同扩散模型的通用性需进一步验证；探针的性能受限于原模型的内部结构，可能对纹理细节不敏感；实现依赖多种外部评估指标，若评估指标本身有偏差，则预测效果受影响。

---

## 36. Sensory-Aware Sequential Recommendation via Review-Distilled Representations

**arXiv ID:** 2603.02709 | [PDF](https://arxiv.org/pdf/2603.02709v1)

**作者:** Yeo Chan Yoon `[一作]` `[通讯]`, Yeo Chan Yoon

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出ASEGR框架，通过大语言模型将用户评论中的感官属性（如颜色、气味、质感等）提取为结构化记录，并利用知识蒸馏生成紧凑的感官嵌入，以此为序列推荐器提供可解释且可控的内容信号。

**💡 创新点**

创新点在于：① 将LLM作为教师对感官属性进行结构化提取并生成可审核的JSON记录；② 通过知识蒸馏将教师输出转化为固定维度的感官嵌入，消除在线LLM推理开销；③ 采用统一的早期融合机制将感官嵌入注入多种主流序列模型，实现可比对的实验设置。

**🔧 技术方法**

使用的技术包括：大语言模型（Qwen3 30B）+ GPT-5 Mini提示式结构化生成；结构化记录的规范化与评估；基于DeBERTa v3的学生Transformer进行蒸馏（回归+对比损失）；固定维度感官嵌入的低维投影与融合；SASRec、BERT4Rec、BSARec等序列推荐模型的统一实验框架。

**📊 数据集**

实验数据来自Amazon 2014评论数据集，选取四个域（Beauty、Sports & Outdoors、Toys & Games、Video Games），采用5-core过滤后分别包含数万用户、数万商品和十万级交互。

**📈 对比分析**

实验通过与仅使用物品ID的基线模型在相同训练协议（leave-one-out、全排序评估）下进行对比。加入感官嵌入后，HR@K和NDCG@K普遍提升，Beauty和Toys域的提升幅度最大（HR@10提升约15–30%，NDCG@10提升约30–40%），BERT4Rec相较于SASRec获得更高的相对收益。

**⚠️ 局限性**

主要局限包括：① 感官属性提取受域内噪声和词汇多样性影响，导致部分属性不一致或缺失；② 蒸馏过程会有信息损失，无法完全保留教师的细粒度输出；③ 目前仅聚焦感官维度，缺少功能性、情境等更丰富的属性；④ 需要离线LLM推理进行标注，增加前期成本。

---

## 37. StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning

**arXiv ID:** 2603.02637 | [PDF](https://arxiv.org/pdf/2603.02637v1)

**作者:** Shiyang Li `[一作]` (University of Minnesota), Caiwen Ding `[通讯]` (University of Minnesota)

**通讯引用:** 3166 | [OpenAlex ID](https://openalex.org/A5030060072)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

提出 StitchCUDA 多代理框架，利用 Planner、Coder、Verifier 三个专门代理自动生成端到端 GPU 程序，并在 Coder 上引入 rubric‑based agentic 强化学习提升代码质量与性能。

**💡 创新点**

创新点：① 将端到端 GPU 编程拆解为全局规划、代码实现与验证三步，解决跨核、主机协同难题；② 将多轮 agentic RL 降维为两原子技能（生成与优化），并通过 rubric 奖励避免 reward hacking、引导高级 CUDA 技巧；③ 结合检索增强生成与工具迭代，实现完整的规划→编码→调优循环。

**🔧 技术方法**

使用技术：LLM（Qwen3‑32B、GPT‑5.2）+多代理工作流；agentic 强化学习（GRPO）+ rubric 与规则奖励；Nsight Systems/Compute 性能分析；RAG 检索最新 CUDA 文档；编译/单元测试、代码修正反馈；链式思考提示。

**📊 数据集**

使用数据集：KernelBench（Level 1–3，Level 3 共 50 端到端任务；Level 1、2 各 20 任务），随机输入与张量形状；在 NVIDIA H200 与 RTX PRO 6000 两台 GPU 上进行评测。

**📈 对比分析**

对比方法：单次 LLM、CUDAForge、Kevin‑32B、GPT‑5.2；评估指标为成功率、E2E 速度加速、Fast_1。StitchCUDA 在 Level 3 实现近 100% 成功率，平均 speedup 约 1.5–1.7×（比基线高 1.5–2.7×），Fast_1 超过 70%；在 Level 1/2 达到 100% 成功率，speedup 2–4×；训练耗时约 20 h，使用 4 张 H200 GPU。

**⚠️ 局限性**

局限性：仍需大量 GPU 编译/调试资源；RL 训练样本规模受限，可能影响更大规模任务；虽然 rubric 奖励抑制了 reward hacking，但仍可能出现细微作弊；主要提升集中在速度上，绝对加速仍有限；实现依赖多代理、工具链，部署复杂度较高。

---

## 38. SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training

**arXiv ID:** 2603.02908 | [PDF](https://arxiv.org/pdf/2603.02908v1)

**作者:** Qi Zhang `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**通讯引用:** 5430 | [OpenAlex ID](https://openalex.org/A5101431030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种基于稀疏自编码器（SAE）的迁移性得分（STS），能在未进行监督微调前预测大型语言模型在不同域的性能变化，进一步演示了该方法在强化学习场景中的潜在适用性。

**💡 创新点**

创新点在于：①利用SAE的单义性将模型内部维度与语义概念对应，能够准确定位微调过程中被显著改变的特征维度；②通过将这些“偏移维度”与目标域的SAE激活相关性进行量化，构造可解释且不需要微调的迁移性评分；③首次将该评估框架推广到强化学习，揭示了微调与强化学习特征漂移差异。

**🔧 技术方法**

核心技术包括：稀疏自编码器（Top‑K SAE）实现单义性特征学习；基于in‑context学习对未微调模型的特征进行估计以预判维度漂移；使用激活均值或ICL差分构造STS指标；统计学评估（Pearson相关系数）与实验验证。

**📊 数据集**

使用的数据集主要有：① LIMO（数学推理）作为微调训练集；② MMLU‑Pro（多域推理）用于评估迁移性；③ Verifiable‑Coding‑Problems‑Python‑10k 和 CoT_Reasoning_Mens_Mental_Health 用于跨领域验证；④ Math‑LightEval 及 RL 相关实验。

**📈 对比分析**

与传统的后验微调评估方法相比，STS 在预测不同域性能变化时实现了 Pearson 相关系数 ≥ 0.7（在 MMLU‑Pro、代码生成和健康对话等任务上均保持 0.7–0.8 以上），远优于仅使用原始激活或简单探测器的 0.2–0.4 相关性；在强化学习实验中，利用真实漂移维度可恢复高相关性，表明方法本身潜力。

**⚠️ 局限性**

限制包括：① 在强化学习场景下，缺乏真值示例导致难以精确估计偏移维度，STS 的预测相关性显著下降；② 该方法依赖于 SAE 的单义性，若模型或层次不满足稀疏性，预测效果会下降；③ 仅适用于能通过ICL模拟目标域的情况，对完全无监督或高度自适应的后置训练策略仍需进一步研究。

---

## 39. RippleGUItester: Change-Aware Exploratory Testing

**arXiv ID:** 2603.03121 | [PDF](https://arxiv.org/pdf/2603.03121v1)

**作者:** Yanqi Su `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**通讯引用:** 4195 | [OpenAlex ID](https://openalex.org/A5075639297)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于代码变更的 GUI 探索式测试系统 RippleGUI，自动生成、执行并多模态检测因代码变更导致的用户可见错误。

**💡 创新点**

① 以代码变更为核心展开涟漪效应探索；② 利用 LLM 进行变更影响分析、情景增强和多模态差异检测；③ 将视觉差异与变更意图结合，实现更精准的 bug 判别。

**🔧 技术方法**

使用 GPT‑5.2 与 Claude Opus 4.5 等大语言模型；向量检索构建场景知识库；Docker 容器化执行；像素级 GUI 差异检测；多模态自然语言+视觉推理。

**📊 数据集**

四款桌面应用（Firefox、Zettlr、JabRef、Godot）及其历史 issue/PR 记录构成的 Scenario Knowledge Base（约 250k+ bug 记录）以及 111 条被检测出的 bug。

**📈 对比分析**

与现有 CI、回归测试及代码评审对比，检测出 119 条 TP，其中 26 条新 bug；总体覆盖率约 44%，精度 46.4%，每条 PR 平均耗时 54.8 分钟，成本约 5.99 美元。

**⚠️ 局限性**

误报率高（因 GUI 渲染不确定性、LLM 幻觉等导致 45.6% 的 FP）；执行预算限制导致部分 bug 未被触发；多模态推理成本昂贵；对复杂交互或自定义项目支持不足。

---

## 40. Scalar-Measurement Attitude Estimation on $\mathbf{SO}(3)$ with Bias Compensation

**arXiv ID:** 2603.02478 | [PDF](https://arxiv.org/pdf/2603.02478v1)

**作者:** Alessandro Melis `[一作]` (CNRS Université Côte d'Azur), Tarek Hamel `[通讯]` (Institut Universitaire de France)

**通讯引用:** 10374 | [OpenAlex ID](https://openalex.org/A5052963738)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于连续Riccati方程的非线性确定性观测器，用标量测量（而非完整向量）估计姿态并补偿陀螺仪偏置。

**💡 创新点**

创新点在于证明两条足够激励的标量测量即可实现姿态可观测，静态时仅需三条；并将观测器设计直接在SO(3)上完成，避免高维嵌入和对偏置的显式补偿。

**🔧 技术方法**

采用Riccati观测器框架、持久激励（PE）与均匀可观测性理论、罗德里格斯公式与李群几何来实现稳定性分析。

**📊 数据集**

在BROAD数据集（9轴Myon aktos-t IMU）上进行实验，使用加速度计与磁力计的不同标量组合。

**📈 对比分析**

与常用互补滤波器对比，通过RMSE（欧拉角与SO(3)角距离）评估，观测器在减少至两条标量时仍保持低误差，且整体误差仅比互补滤波器略高。

**⚠️ 局限性**

局限性包括对运动激励（PE）的依赖、初始估计收敛域有限、需要对观测器增益与Riccati方程参数进行调节，且在低动态或静态非激励条件下性能退化。

---

## 41. Information Routing in Atomistic Foundation Models: How Equivariance Creates Linearly Disentangled Representations

**arXiv ID:** 2603.03155 | [PDF](https://arxiv.org/pdf/2603.03155v1)

**作者:** Joshua Steier `[一作]` `[通讯]`, Joshua Steier

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种名为Composition Projection Decomposition (CPD) 的方法，对分子/晶体模型的中间表示进行线性分解，探测其构成和几何信息的解耦程度。

**💡 创新点**

首次量化模型表示中构成与几何信息的线性可分性梯度，揭示张量乘积等变操作能显著提升线性解耦，并发现多模型在不可变与可变通道中的信息路由差异。

**🔧 技术方法**

采用QR正交投影实现CPD，利用Ridge、GBT、MLP等多种线性与非线性探测器，统计Cohen Kappa、SHAP、Lasso权重正交性等多维度分析。

**📊 数据集**

在QM9分子数据集和Materials Project晶体子集上评估共8种模型，涵盖张量乘积、向量-标量、学习不可变、方向性不可变及手工特征四大体系。

**📈 对比分析**

对每种模型在去除构成后使用Ridge回归测量残差的R²作为主指标，结果显示MACE最高（≈0.78），ViSNet与SchNet中等，DimeNet++低，ANI-2x线性不可解；非线性GBT在残差上显著过估。

**⚠️ 局限性**

结果受限于仅考虑均值池化的结构级表示、仅小分子与晶体样本、预训练规模与架构混杂，且未探测原子级解耦与更大分子系统。

---

## 42. Designing UNICORN: a Unified Benchmark for Imaging in Computational Pathology, Radiology, and Natural Language

**arXiv ID:** 2603.02790 | [PDF](https://arxiv.org/pdf/2603.02790v1)

**作者:** Michelle Stegeman `[一作]` (Radboud University Medical Center), Alessa Hering `[通讯]` (Radboud University Medical Center)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5078758408)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了统一的医学基础模型评估基准UNICORN，涵盖20个跨影像、病理、文本的多模态任务，采用两步推理与轻量化适配器实现few‑shot评估，提供统一的性能指标UNICORN Score。

**💡 创新点**

①首次提出跨任务、多模态、跨领域的统一评估框架；②将Encoder与适配器分离，聚焦通用表示质量；③使用sequestered测试集与标准化提交接口，确保可复现；④开放平台与Leaderboard促进社区合作。

**🔧 技术方法**

基于Grand Challenge容器化平台，使用公开预训练Encoder；轻量化适配器（k‑NN、微调等）；两步推理与评估容器；统一评分归一化与平均。

**📊 数据集**

2400+患者、3700+影像案例、2400+临床报告，来自17家机构、8个国家，覆盖8个解剖区域、4个影像模态，20个任务（分类、检测、分割、回归、NER、caption）。

**📈 对比分析**

提交Algorithm容器与适配器后，平台统一评估并生成leaderboard；基线模型+轻量适配器得到UNICORN Score 0.378；各任务均有单独和综合leaderboard。

**⚠️ 局限性**

测试数据主要来自单一中心，外部泛化受限；仅支持case‑level特征，限制dense输出模型；提交次数与计算时间有限；只支持冻结Encoder，无法评估自定义Decoder模型；评估范围受任务类型和资源约束。

---

## 43. Efficient Dynamic Algorithms to Predict Short Races

**arXiv ID:** 2603.03141 | [PDF](https://arxiv.org/pdf/2603.03141v1)

**作者:** Minjian Zhang `[一作]` (University of Illinois Urbana-Champaign), Mahesh Viswanathan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6267 | [OpenAlex ID](https://openalex.org/A5076852218)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种在观测轨迹上检测“短”竞争（距离受限的竞争对）的框架，并给出了针对 happens-before 与 sync‑preserving 两种预测竞争类型的线性时间、低空间实现。

**💡 创新点**

创新点在于：①将窗口滑动与紧凑的上下文摘要相结合，使得对短竞争的检测不再需要全长轨迹的全局状态；②对 HB 竞争引入窗口相对时钟，消除对全局时间戳的 log n 依赖；③对 sync‑preserving 竞争设计仅保留未配对的获取锁事件作为外部上下文，从而突破传统线性空间瓶颈。

**🔧 技术方法**

技术手段包括：滑动窗口监控框架、基于向量时钟的闭包维护、窗口内事件的增量更新与弹出时的元数据回收、以及对 sync‑preserving 的可达性闭包计算与前向指针压缩。

**📊 数据集**

实验数据集为 153 条多线程程序轨迹，涵盖 30 个 Java 基准（IBM Contest、DaCapo、SIR 等）以及 123 个 OpenMP/HPC 基准（OmpSCR、DataRaceBench、NAS 等）。

**📈 对比分析**

与传统完整轨迹检测工具相比，短竞争算法在相同硬件与时间预算下：HB 版本运行时间相近或略快、内存使用下降至 log k 级；sync‑preserving 版本显著降低内存消耗（消除对 n 的线性依赖），并在较大窗口（1M/10M）下检测到 10‑倍以上的竞争对，提升检测覆盖率。

**⚠️ 局限性**

局限性包括：HB 版本因窗口边界处理导致额外的后处理开销，实际性能提升有限；sync‑preserving 版本仍依赖于向量时钟的 log n 位宽，对大规模变量/锁的情况可能产生空间瓶颈；实现尚未针对 Java 原始类型做深度空间优化，导致在极大轨迹上仍可能产生高 GC 负担。

---

## 44. Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation

**arXiv ID:** 2603.02224 | [PDF](https://arxiv.org/pdf/2603.02224v1)

**作者:** Brady Steele `[一作]` (Georgia Institute of Technology), Brady Steele `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5016775476)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于梯度子空间几何关系的灾难性遗忘理论，并用合成任务、Split-CIFAR100 和 Sequential GLUE 实验验证。

**💡 创新点**

创新点在于引入了忘记规律 F = α(1−cos²θ)+β 与近似秩不变性，揭示任务子空间角度决定遗忘而非适配器秩。

**🔧 技术方法**

使用了 LoRA 参数高效微调、主成分角度计算、梯度子空间投影、O-LoRA 与 EWC-LoRA 等对比方法。

**📊 数据集**

实验数据集包括人为控制角度的合成任务、10 任务拆分的 CIFAR‑100 以及 5 个 GLUE 任务的顺序学习。

**📈 对比分析**

通过与 vanilla LoRA、O-LoRA、EWC‑LoRA 以及任务特定适配器比较，rank sweep 结果显示 CV 约 10–20% 的近似秩不变性；O‑LoRA 在自然正交度已高时提升不显著。

**⚠️ 局限性**

局限性包括：统计功效有限（种子数少）、假设任务难度与角度无关易受混杂、对大模型与高维梯度矩阵的角度计算成本高，以及理论中“秩不变性”依赖经验假设。

---

## 45. TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning

**arXiv ID:** 2603.03072 | [PDF](https://arxiv.org/pdf/2603.03072v1)

**作者:** Christian Greisinger `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 2665 | [OpenAlex ID](https://openalex.org/A5053947568)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对从文本生成TikZ程序的任务，构建了规模约200万条高质量的DaTikZ‑V4数据集，并基于该数据集训练了TikZilla模型；

**💡 创新点**

创新点包括四倍规模的高质量数据、LLM驱动的TikZ调试、VLM生成的细致图形描述、基于逆图像的领域专属奖励模型以及两阶段SFT‑+‑RL训练框架；

**🔧 技术方法**

技术方法涵盖了LLM与VLM的自动标注、Qwen LLM的SFT与GRPO强化学习、DeTikZify‑V2图像编码器的重训练与使用、以及Earth‑Mover‑Distance奖励机制；

**📊 数据集**

使用的数据集为DaTikZ‑V4（>2 M TikZ实例）以及其RL子集DaTikZ‑V4‑RL；

**📈 对比分析**

与GPT‑4o、GPT‑5、TikZero‑Plus‑10B等模型在自动指标（CLIPScore、DreamSIM、TED、CR）和人工评估（1‑5 Likert）上比较，TikZilla在编译率、生成质量和整体评估上均超过同类商用大模型；

**⚠️ 局限性**

主要局限在于VLM生成的图形描述可能存在遗漏或幻觉，导致训练或奖励误导，以及奖励模型可能在极端场景下出现“奖励劫持”。

---

## 46. Architecting Trust in Artificial Epistemic Agents

**arXiv ID:** 2603.02960 | [PDF](https://arxiv.org/pdf/2603.02960v1)

**作者:** Nahema Marchal `[一作]` (Google DeepMind), Iason Gabriel `[通讯]` (Google DeepMind)

**通讯引用:** 2120 | [OpenAlex ID](https://openalex.org/A5032814084)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“认知 AI 代理”概念，并构建了从定义、风险评估、信任框架到社会技术基础设施的完整框架，以实现可信、与人类认知目标对齐的自主知识代理。

**💡 创新点**

创新点在于：①首次系统化地定义认知 AI 代理及其在知识生态中的潜在角色；②提出由可证明的能力、可证伪性和认知美德组成的三要素信任框架；③设计可验证代理凭证、加密溯源链与标准化通信协议，构建多代理生态中的透明治理机制；④将技术规范与社会治理结合，提出“知识圣殿”“参与式治理”等具体社会基础设施。

**🔧 技术方法**

主要采用的技术包括：前沿 LLM 与多模态推理技术、链式思考与自我检验机制、加密身份认证与可溯源链技术、标准化日志协议、以及机制可解释性方法（因果追踪、数据影响函数等）。

**📊 数据集**

论文并未引入特定数据集，更多基于文献综述、专家评估与预期模型表现的概念性分析。

**📈 对比分析**

由于本研究为框架性与规范性工作，没有实验对比；作者通过与现有 LLM 评测基准（如 MMLU、FactScore 等）的对比，说明所提框架所需的评测维度超出传统事实性与推理测试，强调需动态、跨域与过程性评估。

**⚠️ 局限性**

主要局限包括：①技术与治理方案的可实现性与标准化程度尚未验证；②在透明度与可用性、短期收益与长期认知健康之间存在不可调和的权衡；③多代理生态的协调与信任机制易受利益冲突与监管缺失影响；④认知美德与可证伪性的实现难度高，需进一步研究可解释性与自我检验方法；⑤社会基础设施建设需要跨学科、多方参与，存在政治与经济阻力。

---

## 47. Model Editing for New Document Integration in Generative Information Retrieval

**arXiv ID:** 2603.02773 | [PDF](https://arxiv.org/pdf/2603.02773v1)

**作者:** Zhen Zhang `[一作]` (Shandong University), Zhaochun Ren `[通讯]` (Leiden University)

**通讯引用:** 7171 | [OpenAlex ID](https://openalex.org/A5100384130)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对生成式检索模型在新文档加入时的泛化失效问题，本文提出 DOME 框架，利用模型编辑在解码器关键 FFN 层中注入新的 docID 映射。

**💡 创新点**

创新点在于将平均补丁定位关键层与混合软硬标签自适应训练相结合，解决了 GR 模型中编辑向量不可区分的问题，并实现了高效的模型编辑。

**🔧 技术方法**

主要技术包括平均补丁定位关键层、混合软硬标签自适应训练、闭式更新矩阵生成以及多层分布式更新。

**📊 数据集**

使用 Natural Questions (NQ) 与 MS-MARCO 两大公开检索基准，按 90%/10% 划分训练/新增文档。

**📈 对比分析**

与 BM25、DPR、DSI、Ultron、全量重训练、增量微调以及 ROME/MEMIT/AlphaEdit 等基线对比，DOME 在新增文档 Recall@1 近似全量重训练、显著优于增量微调，且更新速度比增量训练快约 40% 并比其他编辑方法快 5 倍，初始文档的遗忘最小。

**⚠️ 局限性**

局限在仅对解码器 FFN 层进行编辑，未考虑注意力层和编码器表示，且依赖于 docID 结构，对更大规模文档集或更复杂 ID 方案的泛化仍待验证。

---

## 48. Custom Keep-Alive Cache Policies

**arXiv ID:** 2603.03091 | [PDF](https://arxiv.org/pdf/2603.03091v1)

**作者:** Sushirdeep Narayana `[一作]` (Wesleyan University), Ian A. Kash `[通讯]` (University of Illinois at Chicago)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出基于在线学习的自定义保持活跃缓存策略，并设计了两种支付方案（Myerson和外部性支付）以实现激励兼容和成本回收。

**💡 创新点**

创新点在于证明在单参数设置下指数权重算法天然满足单调性，从而能通过标准的Myerson机制或简单的VCG外部性支付同时实现效率、激励兼容和成本回收，并通过实验验证两种支付在实际中的表现相近。

**🔧 技术方法**

使用技术包括在线学习中的指数权重算法、Myerson支付公式、VCG外部性支付、Poisson和Hawkes到达过程的模拟以及对Azure公共数据集的真实追踪分析。

**📊 数据集**

实验数据集包括合成的Poisson和Hawkes到达过程以及微软Azure公共数据集中的函数调用追踪。

**📈 对比分析**

通过模拟和真实追踪比较两种支付方案的客户误报惩罚（regret）和成本回收率，结果显示Myerson支付无误报、成本回收近零；外部性支付误报率约20%，平均误报<2%成本，且两方案在Azure trace中表现相近。

**⚠️ 局限性**

局限性在于仅实现渐近效率，成本回收与激励兼容在理论上仅有经验支持；算法需要离散化窗口长度，有限到达数时可能导致较大学习误差，外部性支付的激励保证不是严格的。

---

## 49. Heterogeneous Agent Collaborative Reinforcement Learning

**arXiv ID:** 2603.02604 | [PDF](https://arxiv.org/pdf/2603.02604v1)

**作者:** Zhixia Zhang `[一作]` (Beihang University), Yikun Ban `[通讯]` (Beihang University)

**通讯引用:** 188 | [OpenAlex ID](https://openalex.org/A5047387636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了异构代理协同强化学习范式(HACRL)和对应的协同优化算法HACPO，利用不同模型间的rollout共享实现双向知识迁移。

**💡 创新点**

创新点包括：①实现异构代理的双向rollout共享；②四项专门机制（能力感知优势估计、能力差异系数、指数重要性采样、逐步裁剪）解决能力与分布偏差；③理论证明优势估计无偏且梯度方向一致。

**🔧 技术方法**

采用RLVR+PPO框架，基于组优势估计、指数重要性采样、跨代理能力比重调整以及逐步裁剪的强化学习技术。

**📊 数据集**

使用7.5k高质量MATH题目进行训练，评估数据集包括MATH-500、MATH、GSM8K、AIME2025、AMC23、Minerva和Olympiad。

**📈 对比分析**

与单智能GRPO、GSPO、资源等价GSPO×2及Naive共享baseline进行对比；在三种异构设置下平均提升约3.3%，样本成本仅为GSPO的一半。

**⚠️ 局限性**

局限性包括：仅在可验证奖励环境下验证；对极端异构（模型、tokenizer差异大）鲁棒性尚待进一步探索；需要手动调参（α、δ等）。

---

## 50. Exploiting PendingIntent Provenance Confusion to Spoof Android SDK Authentication

**arXiv ID:** 2603.02539 | [PDF](https://arxiv.org/pdf/2603.02539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 51. When Scaling Fails: Mitigating Audio Perception Decay of LALMs via Multi-Step Perception-Aware Reasoning

**arXiv ID:** 2603.02266 | [PDF](https://arxiv.org/pdf/2603.02266v1)

**作者:** Ruixiang Mao `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (NiuTrans Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了大型音频-语言模型在推理链拉长时的感知衰退现象，提出 CAFE 评估框架量化音频推理错误，并设计 MPAR^2 训练策略通过多步感知‑意识推理与回顾机制提升模型音频感知与推理能力。

**💡 创新点**

首次系统量化音频感知衰退，并通过感知奖励、步骤奖励和回顾奖励实现动态推理预算，使模型在推理过程中持续关注音频信息，从而显著提高感知与推理性能。

**🔧 技术方法**

使用 Qwen2.5‑Omni 作为基模型，结合 SFT+GRPO 强化学习，设计多项生成奖励（感知、步骤、一致性、回顾、格式）并使用 Gemini‑3‑Pro 生成音频描述。

**📊 数据集**

使用 AVQA 数据集生成 QA 与音频描述，评估数据集包括 MMAU、MMAR 等音频推理基准。

**📈 对比分析**

与多种 LALM、LARM 和商业模型对比，MPAR^2 在 MMAU 取得 74.59%（提升 8.69%）和 MMAR 60.32%（提升 5.12%），在 CAFE 评估中获得最高的感知准确率，并在多任务上展现稳定提升。

**⚠️ 局限性**

主要局限在于仅验证于特定音频基准，训练成本高且对大模型依赖强，缺乏跨域泛化和可解释性的进一步评估。

---

## 52. How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities

**arXiv ID:** 2603.02578 | [PDF](https://arxiv.org/pdf/2603.02578v1)

**作者:** Ziwen Xu `[一作]` (Zhejiang University), Shumin Deng `[通讯]` (National University of Singapore)

**通讯引用:** 2774 | [OpenAlex ID](https://openalex.org/A5060484186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SteerEval层级化评测基准，用于系统评估LLM在语言特征、情绪与人格等三个行为域的可控性。

**💡 创新点**

创新点在于构建三层级（L1‐L3）细粒度可控层次、跨域统一评测框架，并通过自动化流水线生成高质量的偏好对照数据，首次揭示可控性随层级细化的系统下降趋势。

**🔧 技术方法**

使用了prompt‑based和activation‑based（PCA、DiffMean、RePS）等控制技术，并结合EasyEdit2评测框架、LLM‑based评判器以及自动化数据合成流水线。

**📊 数据集**

数据集为SteerEval，包含3个域、3层级共7,560条带偏好对的样本，已手工审核并以MIT许可公开。

**📈 对比分析**

实验比较显示prompt‑based在所有层级和模型上均优于activation‑based；activation‑based在粗粒度(L1)可匹敌，但在中细粒度(L2‑L3)显著下降，整体harmonic mean约为3.0。

**⚠️ 局限性**

局限在于仅覆盖单轮单概念控制，未包含多轮对话、多概念组合及长上下文；评测依赖LLM‑judge，可能受偏见影响；对细粒度激活方法的调参和安全监控仍不足。

---

## 53. Distributed Dynamic Invariant Causal Prediction in Environmental Time Series

**arXiv ID:** 2603.02902 | [PDF](https://arxiv.org/pdf/2603.02902v1)

**作者:** Ziruo Hao `[一作]`, Bo Hu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了分布式动态不变因果预测框架 DisDy-ICPT，联合学习时间序列中的动态因果结构、消除客户端间的空间混杂并保持数据本地化；

**💡 创新点**

创新点在于：①将联邦 KCI 测试与随机 Fourier 特征结合生成动态硬/软先验；②分阶段设计（DISM 与 DCTO），先行生成因果先验再通过 Neural ODE 学习因果轨迹；③在 Neural ODE 中深度嵌入先验，实现时变因果学习且在有限通信轮数内收敛；

**🔧 技术方法**

使用技术包括联邦学习（FedAvg）、Kernel Conditional Independence (KCI) 测试、随机 Fourier 特征映射、Neural Ordinary Differential Equations、时序采样与中值滤波、软硬约束正则化；

**📊 数据集**

实验数据集包括合成 SEM 基准、CausalTime 环境分段模拟数据集以及真实能源时序数据（含环境分段的实际环境数据）；

**📈 对比分析**

与基准方法（如 FedCDH、DyCAST 等）进行对比，在 AUROC/AUPRC、MAE、RMSE 等指标上表现更优，结构恢复更准确、预测稳定性更好；

**⚠️ 局限性**

局限性在于：目前仅在离线批处理环境下验证；对空间混杂的检测依赖特定假设；通信轮数虽有限，但对极大客户端规模仍有挑战；对非线性因果结构的理论支持尚不充分。

---

## 54. FlashEvaluator: Expanding Search Space with Parallel Evaluation

**arXiv ID:** 2603.02565 | [PDF](https://arxiv.org/pdf/2603.02565v1)

**作者:** Chao Feng `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FlashEvaluator，一种在生成-评估（G-E）框架中通过并行计算和跨列表特征交互来同时评估所有候选序列的评估器。

**💡 创新点**

创新点包括：①在评估阶段实现跨候选序列的显式比较；②通过共享上下文和列表无关的全项交互，将计算复杂度从 𝒪(K) 降低到近似 𝒪(1/K)；③理论证明联合评估在泛化误差和样本偏移下具有更优的上界。

**🔧 技术方法**

技术手段主要是：自注意力 + 交叉注意力的列表无关全项交互模块、跨列表特征交互模块以及联合训练的 Softmax 交叉熵或 MSE 损失；在实验中使用 Transformer、LinearAttention 等实现高并行度。

**📊 数据集**

数据集：RecFlow（推荐系统，约 3.3M 请求、14M 物品、6 个候选列表）和 CNN/DM（文本摘要，30k 句子、16 句候选列表）；在线 A/B 测试采用抖音短视频推荐系统（数亿用户）。

**📈 对比分析**

与现有 G-E 基线（如 PIER、NAR4Rec）以及单列表评估器相比，FlashEvaluator 在离线评估中提升了 NDCG@6、AUC、Hit Ratio 等指标，在线 A/B 测试提升了 7‑day Retention、用户停留时间、冷启动曝光等关键业务指标，并且推理延迟降低 44%、吞吐量提升 114%。在文本摘要任务上，虽然 ROUGE 结果与 RankGPT、SimCLS 接近，但推理延迟显著下降。

**⚠️ 局限性**

限制：效果受候选集合规模、语义多样性和冗余程度影响，需进一步研究自适应策略；在极大候选数或高冗余场景下，跨列表注意力开销可能成为瓶颈；实验主要集中在推荐与摘要两类任务，其他生成任务的适用性尚待验证。

---

## 55. ReCo-Diff: Residual-Conditioned Deterministic Sampling for Cold Diffusion in Sparse-View CT

**arXiv ID:** 2603.02691 | [PDF](https://arxiv.org/pdf/2603.02691v1)

**作者:** Yong Eun Choi `[一作]` (National Institute for Mathematical Sciences), Sung Ho Kang `[通讯]` (National Institute for Mathematical Sciences)

**通讯引用:** 483 | [OpenAlex ID](https://openalex.org/A5033752054)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于观测残差的自引导采样方法ReCo-Diff，用于稀视角CT的冷扩散重建，消除了传统的启发式重置机制；

**💡 创新点**

创新点在于利用每一步的残差信息进行连续、物理感知的修正，保持确定性采样进程，避免了随机性与手工调参；

**🔧 技术方法**

采用冷扩散与一般化扩散框架，结合误差传播复合训练(EPCT)、残差条件化的U-Net网络、EMA教师网络及Radon变换的确定性退化算子；

**📊 数据集**

使用AAPM低剂量CT数据集（5936张切片）进行训练与评估；

**📈 对比分析**

与FreeSeed、VSS、CoSIGN、CvG-Diff等方法对比，ReCo-Diff在18/36/72视角下均取得更低的RMSE、更高的PSNR与SSIM，且误差曲线更平稳；

**⚠️ 局限性**

局限性包括：仅在模拟稀视角数据上验证，未充分测试真实临床扫描；对极端稀疏（如18视角）仍需一次级别迁移步骤，且训练与推理成本相对较高。

---

## 56. Proactive Guiding Strategy for Item-side Fairness in Interactive Recommendation

**arXiv ID:** 2603.03094 | [PDF](https://arxiv.org/pdf/2603.03094v1)

**作者:** Chongjun Xia `[一作]` (Chinese Academy of Sciences), Mingsheng Shang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 7097 | [OpenAlex ID](https://openalex.org/A5091667639)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于层次强化学习的主动公平指导策略（HRL4PFG），在交互式推荐中逐步引导用户偏好向长尾物品迁移，提升物品端公平性并保持用户满意度。

**💡 创新点**

创新点在于：①采用宏观-微观层次结构，高层生成公平引导目标并在多步反馈下优化；②低层基于该目标与用户当前偏好动态生成推荐；③通过目标约束和候选筛选机制平衡公平性与用户体验。

**🔧 技术方法**

技术包括：层次强化学习（Actor-Critic）、注意力机制提取用户偏好、Gaussian采样生成目标、L2距离筛选候选物品、奖励函数融合准确率与公平性。

**📊 数据集**

使用了KuaiRec和KuaiRand两个工业级交互式推荐数据集，分别包含数十万用户、上万物品与数千万交互记录。

**📈 对比分析**

与多种RL与公平RL基线（SQN、PG、DDPG、TD3、C51、SAC4IR、DNAIR）在模拟环境中对比，HRL4PFG在累计奖励、交互长度以及Gini指数（公平性）均明显优于所有基线，表现最优。

**⚠️ 局限性**

局限性包括：①依赖离线交互模拟，缺乏真实在线验证；②对超参数（如λ_g、M）敏感；③仅针对物品端公平，尚未扩展到人群公平或多维公平；④目标生成与约束可能受限于项目编码空间的质量。

---

## 57. Retrieving Patient-Specific Radiomic Feature Sets for Transparent Knee MRI Assessment

**arXiv ID:** 2603.02367 | [PDF](https://arxiv.org/pdf/2603.02367v1)

**作者:** Yaxi Chen `[一作]` (University College London), Yipeng Hu `[通讯]` (Queen Mary University of London)

**通讯引用:** 281 | [OpenAlex ID](https://openalex.org/A5003506709)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

论文提出了一种面向患者的可解释膝关节MRI放射组学特征集检索框架，通过两阶段检索为每个病人选择最优的固定大小特征集进行诊断。

**💡 创新点**

创新点在于将传统的逐个特征top‑k筛选提升为整体k特征集合的检索，并利用探针奖励训练评分函数，实现单一、可解释的特征集预测。

**🔧 技术方法**

核心技术包括DeepSets式特征集编码器、基于图像的评分网络、轻量探针奖励回归、两阶段候选集检索以及下游线性分类器。

**📊 数据集**

使用的数据集为公开的膝关节MRI ACL撕裂数据集（664训练/92验证）和OAI‑ZIB‑CM骨关节炎数据集（507扫描）。

**📈 对比分析**

与端到端深度学习、所有特征组放射组学和传统top‑k方法对比，检索方法在ACL和OA判别任务中分别达到约0.73和0.75的准确率，优于top‑k并与端到端相近。

**⚠️ 局限性**

局限性在于检索仍依赖候选集近似，且对k值、候选池大小等超参数敏感；验证仅在单模态MRI上完成，未来需扩展到多模态或更大规模数据。

---

## 58. ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization

**arXiv ID:** 2603.02897 | [PDF](https://arxiv.org/pdf/2603.02897v1)

**作者:** Hao Cao `[一作]` (Tsinghua University), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 24732 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ProGIC，一种基于残差向量量化（RVQ）的渐进式生成式图像压缩框架，可在低比特率、低算力环境下实现快速预览与高感知质量压缩。

**💡 创新点**

创新点在于：①使用多阶段RVQ将图像潜在表示拆分为多级残差，天然支持渐进解码；②构建轻量化的深度可分离卷积+小尺寸注意力骨干，并引入特征调制，实现对渐进解码阶段的动态适配；③提出针对渐进解码的联合损失权重策略与训练方案，兼顾低比特率预览与高比特率最终质量；④在保持优秀压缩性能的同时显著降低编码/解码算力，支持GPU与CPU（甚至移动CPU）部署。

**🔧 技术方法**

核心技术包括残差向量量化（RVQ）、深度可分离卷积、轻量化注意力块、特征调制、感知损失、对抗损失、代码簿更新与承诺损失、以及对比率-失真（R–D）训练策略；在压缩后采用简易范围编码做轻量化熵编码。

**📊 数据集**

训练使用ImageNet完整数据集；评估使用 Kodak、Tecnick、DIV2K、CLIC2020 四个公开数据集，主要指标为 LPIPS 与 DISTS；对比传统 VTM‑23.10、LIC‑HPCM、DCVC‑RT，以及生成式编码器 HiFiC、MS‑ILLM、Control‑GIC、DiffEIC、OSCAR 和进化的 ProgDTD。

**📈 对比分析**

与 SOTA 生成式与非生成式压缩器对比，ProGIC 在 LPIPS/DISTS 上取得 57–59% 的 BD‑rate 减少（比 MS‑ILLM 更优），并在 GPU 上实现 10×+ 的解码速度提升，CPU/移动CPU 上同样保持 5–7× 的加速；同时保留了低比特率下的可观预览质量。

**⚠️ 局限性**

局限性包括：①多阶段RVQ 需要预先确定代码簿数与维度，导致比特率范围与中间解码质量之间的权衡；②对极低比特率（极低分辨率或极小图像）下的预览质量仍有限；③代码簿索引熵编码收益有限，可能影响压缩率进一步提升；④在极端算力受限场景（如边缘 GPU）下的鲁棒性与扩展性仍待深入研究。

---

## 59. PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference

**arXiv ID:** 2603.02479 | [PDF](https://arxiv.org/pdf/2603.02479v1)

**作者:** Rituraj Sharma `[一作]` (Virginia Tech), Tu Vu `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了功能层次的框架分解，并设计了 PRISM 算法，通过在推理阶段引入过程奖励模型（PRM）对每一步进行评分，指导人口改进和答案聚合，从而提升大型语言模型在数学与科学推理任务中的准确率。

**💡 创新点**

创新点在于：①将步骤级奖励信号嵌入到人口改进与聚合中，将迭代重写转化为方向性错误校正；②通过重要性采样与重采样构建能量景观，保持多样性；③在多项基准上实现计算‑准确性 Pareto 前沿，显著提升弱模型性能。

**🔧 技术方法**

使用技术包括：过程奖励模型（PRM）、步骤级评分与归一化、重要性采样与重采样、马尔科夫链蒙特卡洛（MCMC）风格的随机改进、-score 投票聚合。

**📊 数据集**

实验数据集：AIME25、HMMT25 以及 GPQA Diamond（前120个例子）。

**📈 对比分析**

与 Simple Voting、SciMaster、Agentic Debate、MAD Conformist/Follower、Recursive Self‑Aggregation 等基线在同一推理配置下对比；PRISM 在 AIME25 达到 90.0%、HMMT25 75.4%、GPQA Diamond 71.4%，超越或匹配现有方法，并在计算‑准确性曲线上位于 Pareto 前沿，显著提升弱模型的效果。

**⚠️ 局限性**

局限性包括：①PRM 仅使用提示模型生成步骤级评分，缺乏更强的外部验证或可执行测试；②假设步骤分割与逻辑单元一致，若不匹配会削弱评分效果；③实验仅针对一种 PRM 实现，未探究更丰富奖励来源或其他领域的推广。

---

## 60. CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment

**arXiv ID:** 2603.02557 | [PDF](https://arxiv.org/pdf/2603.02557v1)

**作者:** Maoyuan Shao `[一作]` (Minzu University of China), Guoshun Nan `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1311 | [OpenAlex ID](https://openalex.org/A5020360628)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CAPT框架，利用模型自身的误判信息构建Confusion Bank，分别在语义层和样本层挖掘混淆关系，并通过Multi-Granularity Discrepancy Expert统一融合，提升跨模态模型的细粒度判别能力。

**💡 创新点**

创新点在于：①首次将误判样本的固定混淆模式系统化建模；②在Prompt Tuning中引入语义混淆矿工（SEM）和样本混淆矿工（SAM）双层混淆挖掘；③设计Diff-Manner Adapter融合全局与局部差异；④引入MGDE实现多粒度混淆专家协同学习。

**🔧 技术方法**

技术包括：Prompt Tuning、信息熵对比损失（InfoNCE）、基于预训练CLIP的视觉-文本对齐、卷积+注意力融合的Diff-Manner Adapter、基于K-means的Prompt聚类、Mixture-of-Experts架构。

**📊 数据集**

在11个基准数据集（ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, EuroSAT, UCF101, DTD, SUN397）以及跨域数据集（ImageNet-V2, ImageNet-Sketch, ImageNet-A, ImageNet-R）进行实验。

**📈 对比分析**

与CoOp、MaPLe、PromptKD、Spotlighter、2SFS、TAC、LwEIB、TAP等SOTA方法对比，CAPT在基准和跨域任务中多场景下实现了最优或同等性能，例如基准任务的HM最高达83.90%，跨域任务平均准确率提升至71.95%。同时，误判样本纠正率高达50.72%。

**⚠️ 局限性**

局限性包括：需要先收集误判样本构建Confusion Bank，增加预处理步骤；模型对噪声混淆样本敏感，若混淆分布不稳定可能导致性能下降；在极低样本场景下（如1-shot）仍略逊于某些对比方法；以及推理时仍需额外模块，虽开销不大但相较纯CLIP仍有轻微延迟。

---

## 61. Gravity Falls: A Comparative Analysis of Domain-Generation Algorithm (DGA) Detection Methods for Mobile Device Spearphishing

**arXiv ID:** 2603.03270 | [PDF](https://arxiv.org/pdf/2603.03270v1)

**作者:** Adam Dorian Wong `[一作]` (Dakota State University), John D. Hastings `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了名为Gravity Falls的半合成SMS钓鱼域名数据集，并对传统与机器学习的DGA检测方法进行了多工具、多策略的评估。

**💡 创新点**

创新点在于首次提供针对移动设备smishing域名的公开数据集，并揭示传统与现代检测器在不同DGA技巧下的性能差异。

**🔧 技术方法**

使用了Shannon熵、Exp0se、LSTM（miaWallace0618）和DGAD（COSSAS）等传统与机器学习检测技术。

**📊 数据集**

采用Gravity Falls四个技术簇（Cats Cradle、Double Helix、Pandoras Box、Easy Rider）以及Alexa、Cisco、Cloudflare、Majestic等Top‑1M域名作为对照。

**📈 对比分析**

通过精准率、准确率和召回率的many‑to‑many比较，发现检测器在随机字符串域名上表现最好，但在字典拼接与主题组合劫持域名上显著失效。

**⚠️ 局限性**

局限性包括数据集半合成、样本重复、基线Benign域名不完善，以及使用的工具版本较旧，影响了外部可泛化性。

---

## 62. Dimension-Independent Convergence of Underdamped Langevin Monte Carlo in KL Divergence

**arXiv ID:** 2603.02429 | [PDF](https://arxiv.org/pdf/2603.02429v1)

**作者:** Shiyuan Zhang `[一作]` (University of California), Quanquan Gu `[通讯]` (University of California)

**通讯引用:** 178020 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文针对分段离散化的欠阻尼 Langevin 动力学（ULMC 与随机中点法 RMD）给出了 KL 散度的维度无关收敛速率，证明其迭代复杂度仅依赖于 Hessian 上界的迹，而非环境维度。

**💡 创新点**

创新点：①首次实现欠阻尼 Langevin 离散化的维度无关 KL 收敛；②提出对弱/强误差的维度无关估计，采用加权范数和精细的变化测度技巧；③将 KL 本地误差框架推广到欠阻尼场景，显著降低对条件数 κ 的依赖。

**🔧 技术方法**

主要技术：
- KL 本地误差框架（shifted operator、auxiliary process）
- 对弱误差与强误差的精细计算，使用 H‑norm 代替欧氏范数
- 通过 Donsker–Varadhan 变分公式与 Taylor 展开，避免出现维度相关的高阶矩
- 对交叉正则性（cross‑regularity）的维度无关上界
- 递归误差控制与迭代复杂度推导。

**📊 数据集**

本研究为纯理论工作，无使用实测数据集；所有结论均基于数学证明和分析。

**📈 对比分析**

与已有工作比较：
- 对强对数凸分布，迭代复杂度提升至 Θ(κ^{3/2}β^{-1/2}Tr(H)^{1/2}/ε)，优于传统的 Θ(κ^2β^{-1}Tr(H)/ε^2)；
- 对一般凸分布，ULMC 维度无关复杂度为 Θ(βTr(H)W/ε^4)；RMD 在一般凸情形下实现 Θ(βTr(H)W^{5/2}/ε^3)，将原先的 Θ(1/ε^4) 提升至 Θ(1/ε^3)；
- 在 Tr(H) ≪ d 的高维场景下，显著缩小了维度对性能的负面影响。

**⚠️ 局限性**

局限性：
- 仍保持对条件数 κ 的多项式依赖，未达到线性或更优的 κ 依赖；
- 只给出理论收敛上界，缺乏实验验证；
- 对 RMD 的交叉正则性仅通过对 ULMC 的最终步长替代证明，未完整证明 RMD 本身的正则性；
- 适用于光滑对数凸或一般凸目标，未扩展到非光滑或非对数凸情形。

---

## 63. Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory

**arXiv ID:** 2603.02663 | [PDF](https://arxiv.org/pdf/2603.02663v1)

**作者:** Shunki Uebayashi `[一作]` (Kyoto University), Koh Takeuchi `[通讯]` (Kyoto University)

**通讯引用:** 7171 | [OpenAlex ID](https://openalex.org/A5021174309)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多模态多维度项目反应理论（M3IRT）框架，用于评估多模态大型语言模型（MLLM）在跨模态推理中的能力，并通过拆分图像、文本和跨模态三个维度来识别并过滤掉单模态“捷径”问题，减少评测成本；

**💡 创新点**

创新点在于将经典IRT的单维度能力与难度参数分解为图像专属、文本专属以及跨模态整合三部分，从而能够精确估计模型的跨模态推理能力和题目的跨模态难度；

**🔧 技术方法**

采用基于梯度下降的多模态IRT与其多维变体（M3IRT-M）来学习能力与难度参数，并结合计算机自适应测试（CAT）和D-Optimality进行高效子集挑选；

**📊 数据集**

使用公开的三大视觉语言基准MMMU、MathVista和SEED-Bench，共计约3000道题目，评估24种VLM（包括GPT‑4.1、Gemini‑2.0、Claude‑3.7以及开源模型等）；

**📈 对比分析**

与随机挑选、经典IRT、MIRT、TinyBenchmarks和FlashEval等方法比较，M3IRT在仅使用3%~10%题目时即可获得与完整基准相近的Spearman秩相关（>0.8），且所选题集低质量题比例显著低于其他方法；

**⚠️ 局限性**

局限性主要体现在仅针对多选闭卷题，未覆盖开放式生成任务，且目前仅针对视觉‑文本模态，未来需扩展至其他模态和开放式问题；

---

## 64. Type-Aware Retrieval-Augmented Generation with Dependency Closure for Solver-Executable Industrial Optimization Modeling

**arXiv ID:** 2603.03180 | [PDF](https://arxiv.org/pdf/2603.03180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 65. Beyond One-Size-Fits-All: Adaptive Subgraph Denoising for Zero-Shot Graph Learning with Large Language Models

**arXiv ID:** 2603.02938 | [PDF](https://arxiv.org/pdf/2603.02938v1)

**作者:** Fengzhi Li `[一作]` (JIUTIAN Research), Junlan Feng `[通讯]` (JIUTIAN Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 GraphSSR 框架，使用 Sample-Select-Reason（SSR）管线实现自适应子图去噪，并通过 SSR‑SFT 与 SSR‑RL 两阶段训练，使大型语言模型在零样本图推理任务中完成节点分类与链式推理。

**💡 创新点**

创新点：① 用 SSR 管线替代传统统一 k‑hop 子图提取，主动去除任务无关的结构噪声；② 设计 SSR‑SFT 数据合成与多维质量过滤生成高质量推理轨迹；③ 两阶段强化学习（真实性强化 RLVR + 去噪强化 RLVR）通过中间奖励直接指导子图采样与选择。

**🔧 技术方法**

技术手段：基于 DeepSeek‑R1‑distilled‑Qwen2.5‑14B 作为 LLM；LlamaFactory 进行 SFT；verl + GRPO 实现分组奖励的两阶段 RL；能量度量评估子图多样性；链式思维（CoT）生成推理轨迹；文本化图结构与指令调优。

**📊 数据集**

使用的数据集：训练/合成阶段基于 Graph‑R1 数据集及教师模型；零样本评估在四个基准图数据集上：Cora、WikiCS、Products、FB15K237（节点分类与链接预测）。

**📈 对比分析**

与多种基线（OFA、GraphGPT、UniGraph、ZeroG、LLaGA、GOFA、Graph‑R1）以及大型推理模型（DeepSeek‑R1、Qwen3、Ministral）对比，GraphSSR 在 Cora、WikiCS、Products、FB15K237 上均取得最高或接近最高准确率，如 Products 上 68.49% 对比 Graph‑R1 66.59%。

**⚠️ 局限性**

局限性：① 依赖大量合成数据与教师模型，合成成本高；② 对去噪强度 λ 超参数敏感，过大/过小会导致过度/不足去噪；③ 训练需要多块 H100 GPU，成本较高；④ 目前仅在大规模 LLM 上验证，迁移到更小模型仍需探索。

---

## 66. Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results

**arXiv ID:** 2603.02948 | [PDF](https://arxiv.org/pdf/2603.02948v1)

**作者:** Alberto Miño Calero `[一作]` (Norwegian University of Science and Technology), Konstantinos E. Tatsis `[通讯]` (ETH Zurich)

**通讯引用:** 747 | [OpenAlex ID](https://openalex.org/A5065805581)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于域感知傅里叶特征（DaFFs）的PINN新建模方法，利用域的几何和边界信息构造输入编码，消除对边界损失和多目标权重平衡的需求，并通过层级相关性传播（LRP）进行可解释性分析。

**💡 创新点**

创新点在于：①将拉普拉斯算子在给定边界条件下的本征函数作为傅里叶特征，实现了边界条件的硬约束；②通过消除随机傅里叶特征的非可训练随机性和多目标损失，显著简化训练过程并提升收敛速度；③结合LRP为仅输入空间坐标的PINN提供可解释性框架。

**🔧 技术方法**

核心技术包括：Physics‑Informed Neural Networks（PINN）架构、Domain‑aware Fourier Features（DaFFs）构造、随机傅里叶特征（RFFs）对比、层级相关性传播（LRP）可解释性方法、梯度平衡策略（ReLoBRaLo）以及BFGS优化。

**📊 数据集**

使用了两类经典PDE案例：一是Kirchhoff‑Love薄板弯曲方程（矩形域）；二是Helmholtz波动方程（正方形域）。数据仅为采样的协变量点（内点与边界点），无显式训练标签。

**📈 对比分析**

与普通PINN和PINN‑RFFs比较，PINN‑DaFFs在训练误差和验证误差上相差数个数量级，训练时间从数小时缩短到几分钟，收敛次数减少。LRP分析表明DaFFs的特征重要性分布更集中、物理一致，RFFs则呈分散、局部最优特征。

**⚠️ 局限性**

局限性包括：①需要先求解拉普拉斯算子的本征分解，计算成本在复杂几何或高维时显著；②目前仅处理齐次边界条件，非齐次或混合边界仍需扩展；③LRP指导仅在模型性能足够好时才可靠，对低质量模型解释可能误导。

---

## 67. Wukong-Omni: Design, Modeling and Control of a Multi-mode Robot for Air, Land, and Underwater Exploration with All-in-One Propulsion Unit

**arXiv ID:** 2603.02602 | [PDF](https://arxiv.org/pdf/2603.02602v1)

**作者:** Yufan Liu `[一作]` (Guangdong University of Technology), Fumin Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7912 | [OpenAlex ID](https://openalex.org/A5068994727)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文设计、建模、控制并实测了一款名为Wukong‑Omni的三域机器人，能够在陆地、水下及空中实现无缝切换，并完成灾害搜救等任务。

**💡 创新点**

核心创新在于提出的“全能一体化推进单元”，通过一套双向驱动的机轴、单向轴承、齿轮箱和轮‑螺旋桨复合结构，实现了同一推进系统在三种环境中的高效工作；同时引入可变角度主臂与轮‑螺旋桨倾斜设计，解决了陆地与水下动力冲突问题。

**🔧 技术方法**

技术手段包括：机械仿真（QPROP、OpenProp）、CFD（Star‑CCM+）优化推进单元；多模式动力学建模与统一控制框架；多域切换的有限状态机和柔性关节运动规划；在水面、陆面和空中分别采用PID、L1导航、海底姿态控制等算法。

**📊 数据集**

实验数据来源为：水槽实验、陆地跑道与GPS轨迹、户外人工湖测试；未使用公开数据集，全部为自建实验平台产生的测量数据。

**📈 对比分析**

与现有双域（陆/水或陆/空）机器人以及单域方案对比，Wukong‑Omni的推进效率提升约100%，最大推力提升约150%；在陆地模式下实现1.1 m/s巡航，空中模式下10 m高度5 m/s速度，水下模式下可实现±35°姿态控制，水下停深误差<0.12 m；多域切换时间均≤10 s，成功率100%。

**⚠️ 局限性**

局限性包括：整体重量较传统双域方案略高，导致电池续航受限；水下齿轮箱的效率仍低于专用水下螺旋桨；轮‑螺旋桨倾斜角度在提升水下推力的同时会降低陆地通过障碍的能力；并且在极端水体湍流或高压环境下性能尚未验证。

---

## 68. Beyond Language Modeling: An Exploration of Multimodal Pretraining

**arXiv ID:** 2603.03276 | [PDF](https://arxiv.org/pdf/2603.03276v1)

**作者:** Shengbang Tong `[一作]` (FAIR, Meta), Saining Xie `[通讯]` (New York University)

**通讯引用:** 46670 | [OpenAlex ID](https://openalex.org/A5102416863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

从零开始训练单一解码器 Transformer，通过统一多模态预训练学习视觉与语言的联合表示与生成。

**💡 创新点**

证明单一高维语义编码（RAE）可同时支撑视觉理解与生成，并通过 MoE 实现自适应模态容量分配，从而在统一预训练中克服视觉与语言的竞争与规模不匹配。

**🔧 技术方法**

使用 Transfusion 框架的自回归文本预测 + 流匹配视觉预测，结合模块化 FFN、Mixture‑of‑Experts、x‑pred/v‑pred 目标以及语义视觉编码 SigLIP2。

**📊 数据集**

训练数据涵盖大规模网络文本、原始视频、图文配对（MetaCLIP、Shutterstock）及动作条件视频（Navigation World Model），约 1 T token。

**📈 对比分析**

在多项基准（文本困惑度、DPG、GenEval、FID、VQA、WISE）上，统一模型保持或提升性能，且 MoE 实现了更高的效率与更佳的跨模态迁移。

**⚠️ 局限性**

受限于计算与数据规模，视觉仍极度依赖数据；语义视觉编码在细粒度重建上逊于 VAE，且跨域文本泛化略有下降，仍需改进视觉表示与更大规模的多模态数据。

---

## 69. Single-Sample Bilateral Trade with a Broker

**arXiv ID:** 2603.03016 | [PDF](https://arxiv.org/pdf/2603.03016v1)

**作者:** MohammadTaghi Hajiaghayi `[一作]` (University of Maryland), Suho Shin `[通讯]` (University of Maryland)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5111130450)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在仅能获取买卖双方各自分布单个样本的情况下，如何通过经纪人实现双边交易，并给出了两类情形（对称分布与随机支配分布）的简单定价机制。

**💡 创新点**

创新点在于证明即使只有极少的样本信息，仍能用常数因子逼近第一最佳的交易收益、社会福利以及在MHR假设下的最优利润，填补了先前仅在无经纪人设定下的结果；并给出了匹配或近匹配的上界，说明了逼近因子几乎最佳。

**🔧 技术方法**

技术方法主要是构造基于单样本的定价机制（对称情形下取 max(p,q) 与 min(p,q)，随机支配情形下直接报价），通过积分变换、分部积分与不等式推导，计算其期望收益和福利，并与第一最佳指标进行比较；同时利用MHR、双MHR等分布性质和数值优化得到逼近常数。

**📊 数据集**

该工作属于理论分析，无使用实际数据集；假设买卖双方的估值分布满足可积、连续且满足必要的MHR或双MHR等假设。

**📈 对比分析**

与无经纪人、最优机制（已知完整分布）以及已有的单样本双边交易结果比较：对称情形下，提出机制在GFT、SW上分别达到 7/24 和 2/3 的下界，并在MHR下实现 2/55 的利润逼近；在随机支配情形下，分别取得约 0.1254 的 GFT 和 (3-√2)/12 的 SW 逼近；对应上界显示逼近因子无法低于 7/48（GFT）和 1/6（SW）。

**⚠️ 局限性**

局限性包括：1) 对于一般实例，无法证明利润的常数逼近；2) 样本数量的增加不一定改善 GFT/福利逼近，尚未给出样本-性能的完整权衡；3) 主要针对 MHR 或双MHR 分布，未覆盖更一般的分布；4) 结论多为理论下界/上界，缺乏实证验证。

---

## 70. Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation

**arXiv ID:** 2603.02554 | [PDF](https://arxiv.org/pdf/2603.02554v1)

**作者:** Chonghua Lv `[一作]` (Xidian University), Zhun Zhong `[通讯]` (Hefei University of Technology)

**通讯引用:** 10655 | [OpenAlex ID](https://openalex.org/A5065328976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种多阶段的知识蒸馏框架GKD，专注于提升从视觉基础模型到小型模型的跨域泛化能力。

**💡 创新点**

创新点在于将表示学习与任务学习分离，并引入查询式软蒸馏(QSD)机制，让学生能选择性地获取教师的空间知识。

**🔧 技术方法**

技术包括多阶段蒸馏、任务无关与任务有关的特征蒸馏、查询式软蒸馏、以及冻结编码器的任务学习。

**📊 数据集**

使用了五个道路场景分割基准（Cityscapes、BDD100K、Mapillary、ACDC、GTAV）和两个遥感基准（ISPRS Potsdam、Vaihingen）。

**📈 对比分析**

与多种现有KD方法对比，GKD在F2F和F2L设置下平均提升1.9%和10.6%的mIoU，在标签稀缺和多源域情形下亦保持优势。

**⚠️ 局限性**

局限性包括对源域数据的依赖、对查询机制的超参数敏感性，以及在极端域偏移下仍存在一定性能下降。

---

## 71. Give me scissors: Collision-Free Dual-Arm Surgical Assistive Robot for Instrument Delivery

**arXiv ID:** 2603.02553 | [PDF](https://arxiv.org/pdf/2603.02553v1)

**作者:** Xuejin Luo `[一作]` (Beihang University), Junchen Wang `[通讯]` (Beihang University)

**通讯引用:** 14664 | [OpenAlex ID](https://openalex.org/A5100351611)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种双臂手术辅助机器人，能够在动态环境中实现无碰撞的手术器械递送。

**💡 创新点**

创新点在于将Vision‑Language Model用于零射击高层任务规划，并结合实时距离预测网络与统一的QP框架，实现自适应路径规划与自碰撞/障碍物避免。

**🔧 技术方法**

技术包括VLM（GPT‑4o）+DINOv2、SAM、Mediapipe进行语义与几何关联；深度网络预测最小距离；统一QP实时避障与自碰撞约束。

**📊 数据集**

使用真实的RGB‑D点云数据和手术器械样本，未依赖公开数据集；VLM使用GPT‑4o。

**📈 对比分析**

与DawnIK、CollisionIK、CBF‑QP等最先进避障方法比较，仿真与真实实验中优化时间更短、平均位置误差更小、加速度更平滑，最终器械递送成功率达83.33%。

**⚠️ 局限性**

局限在于对薄平滑器械的抓取策略不足，且依赖VLM对关键点的准确识别，误判会导致任务失败。

---

## 72. Beyond Factual Correctness: Mitigating Preference-Inconsistent Explanations in Explainable Recommendation

**arXiv ID:** 2603.03080 | [PDF](https://arxiv.org/pdf/2603.03080v1)

**作者:** Chengkai Wang `[一作]` (Ningbo University), Baisong Liu `[通讯]` (Ningbo University)

**通讯引用:** 841 | [OpenAlex ID](https://openalex.org/A5030910124)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 PURE（Preference‑aware Unhallucinated Reasoning for Explanation）框架，通过在生成前对多跳知识图路径进行偏好感知筛选，生成既事实正确又与用户偏好一致的可解释推荐。

**💡 创新点**

创新点在于：①提出“偏好不一致解释”这一新失效模式并给出衡量指标；②在检索阶段引入目标感知意图、特异性评分与多样性重排，实现对解释路径的精细化选择；③通过结构感知的软硬提示融合，保持图结构信息，提升生成解释的逻辑连贯性和可信度。

**🔧 技术方法**

主要技术包括：结构增强语义索引（RGAT+Graph Transformer）、目标感知用户意图建模、特异性（结构、语义、偏好）评分、MMR多样性重排、软硬提示混合与LoRA微调的 LLM 生成。

**📊 数据集**

使用 Amazon Books、Movies & TV、Yelp 三大真实数据集，并将物品映射到 Freebase 知识图进行结构化检索。

**📈 对比分析**

与 KG‑Flat、PEPLER、LLMXRec、LLM2ER、G‑Refer、MAPLE 等基线在 HR@5、NDCG@5、BLEU‑4、ROUGE‑L、F‑EHR（事实幻觉率）和 P‑EHR（偏好不一致率）上进行离线评估；PURE 在事实性与偏好一致性指标上显著优于基线，且在推荐准确率上保持第二名，展示了在保证解释质量的同时不损失推荐性能。

**⚠️ 局限性**

局限性包括：依赖离线预计算的知识图，无法实时响应知识更新；对隐式反馈和多模态信息的支持不足；特征提取与偏好代理近似可能导致误判，从而在极端稀疏或非结构化场景下表现不佳。

---

## 73. TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference

**arXiv ID:** 2603.03075 | [PDF](https://arxiv.org/pdf/2603.03075v1)

**作者:** Mhd Rashed Al Koutayni `[一作]` (German Research Center for Artificial Intelligence), Didier Stricker `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 TinyIceNet，一种轻量级语义分割网络，用于 Sentinel‑1 SAR 数据的海冰 SOD（发展阶段）映射，并在 Xilinx Zynq UltraScale+ FPGA 上实现低功耗实时推理。

**💡 创新点**

提出了硬件感知的模型共设计、基于 8 位量化感知训练（QAT）与简化 U‑Net 结构，证明在极低功耗条件下仍能保持 75%+ F1 分数。

**🔧 技术方法**

采用 8 位量化感知训练、后训练量化、High‑Level Synthesis（HLS）与 DeepEdgeSoC 框架，将网络映射至 FPGA；同时使用双极化 Sentinel‑1 SAR 图像输入。

**📊 数据集**

使用 AI4Arctic 数据集的 Sentinel‑1 双极化 SAR 图像及其对应的 SOD 标签进行训练与评估。

**📈 对比分析**

与全精度 GPU、Jetson AGX Xavier、以及传统 U‑Net/DeepLabv3+ 模型比较，TinyIceNet 在 FPGA 上实现 7 fps，能耗 113.6 mJ/场景，F1 75.2%，相比 GPU 能耗下降约 2 倍、相比传统模型参数量减少 70%。

**⚠️ 局限性**

局限性包括仅针对 Sentinel‑1 SAR 与单一 SOD 参数，量化对精度敏感，未在多极化或多模态数据上验证；在更大规模或不同季节场景下的泛化仍待研究。

---

## 74. Differentiable Time-Varying IIR Filtering for Real-Time Speech Denoising

**arXiv ID:** 2603.02794 | [PDF](https://arxiv.org/pdf/2603.02794v1)

**作者:** Riccardo Rota `[一作]` (Logitech Europe), Milos Cernak `[通讯]` (Logitech Europe)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 TVF（Time‑Varying Filtering），一种低延迟、1M 参数的实时语音增强模型。

**💡 创新点**

创新点是将可微分的 IIR biquad 滤波链与轻量级神经网络结合，实现可解释的动态滤波。

**🔧 技术方法**

使用轻量级卷积+GRU 预测 35 个 biquad 的增益、Q 和中心频率，并采用向量化 systolic 方案加速训练。

**📊 数据集**

在 Valentini‑Botinhao 噪声语音数据集上进行训练与评估。

**📈 对比分析**

与静态 PEQ 和 DFNet3 对比，TVF 在 PESQ、POLQA、SIGMOS 等主观/客观指标上略优，保持 21 ms 延迟。

**⚠️ 局限性**

局限在只能做线性滤波，无法实现复杂相位重建，且在大规模数据上的性能待验证。

---

## 75. SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

**arXiv ID:** 2603.02505 | [PDF](https://arxiv.org/pdf/2603.02505v1)

**作者:** Lekang Wen `[一作]` (Wuhan University), Mi Wang `[通讯]` (Wuhan University)

**通讯引用:** 5248 | [OpenAlex ID](https://openalex.org/A5100742710)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Semantic‑Guided Modality‑Aware（SGMA）框架，用于解决遥感不完整多模态语义分割（IMSS）问题，能够在任意模态缺失的情况下保持鲁棒的分割性能。

**💡 创新点**

核心创新在于：① 通过全局语义原型与多头注意力实现语义引导融合，显著降低类内差异和跨模态异质性；② 基于语义引导的鲁棒性评估动态重采样（Modality‑Aware Sampling）来提升脆弱模态的学习；③ 这两个模块为plug‑and‑play，可无缝嵌入任意网络与模态组合。

**🔧 技术方法**

技术细节包括：多尺度特征提取、深度卷积投影（MP）、类别压缩（CSF）、多头注意力（SP 与 RP）、鲁棒性映射与软/软逆归一化采样、对比学习/MAE的前置思想以及标准交叉熵损失。

**📊 数据集**

实验数据集：ISPRS Potsdam（RGB、DSM、NIR）、Data Fusion Contest 2023（RGB、DSM、SAR）、以及一套自驾场景数据集（RGB、Depth、Event、LiDAR）。

**📈 对比分析**

对比四种先进基线（纯拼接融合、随机模态丢弃、对比学习+MAE、最优模态集成），在三大数据集上，SGMA在平均、Top‑1 与 Last‑1 mIoU/F1 均优于对手。特别是在单模态和脆弱模态组合（Last‑1）上提升幅度达到约+18–23%，并且在多模态组合下保持稳定的高性能。

**⚠️ 局限性**

局限性：缺乏对模态学习动态的可解释性；仅针对静态多模态图像，未考虑时序或视频序列；对极端类别或异常模态缺失的鲁棒性尚待进一步验证。

---

## 76. ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation

**arXiv ID:** 2603.02945 | [PDF](https://arxiv.org/pdf/2603.02945v1)

**作者:** Bo Xu `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5114038310)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在没有任务数据、无重训练条件下的模型融合问题，提出了 ACE-Merging 这一完整的闭式解框架。

**💡 创新点**

创新点包括：① 证明任务输入协方差可由 fine‑tuning 权重差隐式估计；② 设计自适应正则化、集体结构先验（CSP）以及谱细化三种模块，实现无数据下的高质量融合；③ 通过一次闭式求解实现比以往迭代/启发式方法更稳定、更高效。

**🔧 技术方法**

技术手段主要有：线性近似、协方差估计与归一化、Tikhonov 正则化、低秩集体先验、谱重加权、SVD 细化，以及基于 Transformer 的权重变换。

**📊 数据集**

使用的基准数据集包括：视觉任务 8/14/20 任务集（Cars、DTD、EuroSAT 等）、语言任务 GLUE（CoLA、SST‑2、MRPC 等）以及 RoBERTa、GPT‑2 等模型。

**📈 对比分析**

与多种基线（Weight Averaging、Task Arithmetic、WUDI‑Merging、TSV‑M、CART 等）在视觉和语言任务上进行对比，ACE‑Merging 在 GPT‑2 上平均提升约 4%、在 RoBERTa‑Base/Large 上提升 5%/3%，在 ViT‑L/14 20‑任务场景中提升近 2%，均显著超越当前无数据融合最优方法。

**⚠️ 局限性**

局限性：① 仍需手动设定正则化强度 ϵ、异质性阈值 γ 与谱秩比例 k_frac；② 对于任务异质性低的场景提升有限；③ 理论推导基于线性近似，可能在高度非线性层面受限；④ 未来可进一步实现参数自适应与完全自动化。

---

## 77. Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects

**arXiv ID:** 2603.02333 | [PDF](https://arxiv.org/pdf/2603.02333v1)

**作者:** Xiaoyu Luo `[一作]` (Aalborg University), Johannes Bjerva `[通讯]` (Aalborg University)

**通讯引用:** 1403 | [OpenAlex ID](https://openalex.org/A5013472329)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了扩散式语言模型的记忆化行为，提出了一套通用的可发现记忆提取框架，并对不同采样分辨率、模型规模以及与自回归模型的对比进行了理论与实验验证。

**💡 创新点**

创新点包括：①将（n,p）可发现提取定义推广到任意掩码模式与随机采样路径；②证明采样分辨率越高记忆化概率单调递增；③将自回归模型视为采样分辨率极限，实现两种架构的统一比较。

**🔧 技术方法**

采用的技术主要包括：蒙特卡洛抽样与 Gumbel 采样、变分下界训练、概率抽样路径理论推导、以及对掩码比例、生成步骤数等参数的系统实验。

**📊 数据集**

实验数据集涵盖：SlimPajama 预训练语料、Enron 邮件数据用于 PII 记忆化评估，以及 TREC 2007 Spam 数据用于检验记忆化与泛化的区分。

**📈 对比分析**

通过在相同 prefix‑suffix PII 完成任务下，以查询预算 n=10,000、目标概率 p∈{0.5,0.99} 进行对比，实验发现扩散模型在同等规模下记忆化率显著低于自回归模型；并且在更细粒度的单步生成下记忆化概率随采样步骤增加而提升。

**⚠️ 局限性**

局限性包括：实验仅覆盖了几种模型规模和架构，未深入探讨后期微调或偏好优化对记忆化的影响；理论假设（如上下文增大不降低恢复概率）在极端掩码或低概率场景下可能不完全成立；以及评估仅聚焦于可发现记忆化，无法覆盖所有隐私泄露方式。

---

## 78. Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning

**arXiv ID:** 2603.02280 | [PDF](https://arxiv.org/pdf/2603.02280v1)

**作者:** Jinge Ma `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3953 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Temporal‑Adjusted Loss (TAL) 以解决 Class‑Incremental Learning 中因时间不平衡导致的新旧类别预测偏差。

**💡 创新点**

创新点在于把时间不平衡视为正负监督的动态差异，利用指数衰减记忆核对每个类别的正监督强度 Q 进行建模，并在交叉熵中按 Q 动态重加权负监督，从而在训练过程中自适应纠正类别偏差。

**🔧 技术方法**

主要技术包括：指数衰减记忆核、递归更新 Q_k、权重函数 w(Q)、频率对齐系数 α 以及对交叉熵损失的改写；实现时只需维护 Q 向量，无需额外网络结构或显存。

**📊 数据集**

实验使用 CIFAR‑100、ImageNet‑100、Food101 等公开数据集，针对不同任务划分（如 10‑task、20‑task）进行评估。

**📈 对比分析**

将 TAL 作为损失函数插入多种现有 CIL 方法（iCaRL、FOSTER、DER、MEMO、TagFex 等）后进行对比。实验表明，TAL 在所有基线上均显著提升整体准确率，甚至使最基础的 iCaRL 超越更高级的方法；忘记曲线更平滑，早期类别的召回率得到明显改善。

**⚠️ 局限性**

局限性：使用固定的指数衰减参数 λ，无法适应不同任务或随时间变化的记忆特性；对长期记忆的捕捉不够充分，未来需探索更灵活或非参数的时间建模方式。

---

## 79. Contextual Latent World Models for Offline Meta Reinforcement Learning

**arXiv ID:** 2603.02935 | [PDF](https://arxiv.org/pdf/2603.02935v1)

**作者:** Mohammadreza Nakheai `[一作]`, Joni Pajarinen `[通讯]` (Aalto University)

**通讯引用:** 1570 | [OpenAlex ID](https://openalex.org/A5017983137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 Self‑Predictive Contextual Offline Meta‑RL（SPC‑OMRL）框架，联合训练任务上下文编码器和自回归潜在世界模型，实现从离线数据中学习可泛化的任务表示与策略。

**💡 创新点**

创新点在于：①将任务表示作为条件输入到潜在世界模型中，并通过联合训练实现任务相关的时间一致性；②使用离散潜在码表（FSQ）与 Gumbel‑Softmax 分类损失，避免重建误差；③理论上仅需预测控制信息即可完成控制，无需观测重建。

**🔧 技术方法**

核心技术包括潜在世界模型、上下文编码器、离散潜在编码（Finite‑Scalar Quantization）、Gumbel‑Softmax、信息对比学习（InfoNCE）、多步时间一致性损失以及离线 RL 算法 Implicit Q‑Learning（IQL）。

**📊 数据集**

实验数据集涵盖 MuJoCo（20 训练 / 10 训练 / 10 测试任务）、Contextual DeepMind Control 以及 Meta‑World（ML1/ML10/ML45），所有数据均通过 Dropout Q‑Learning 生成的离线轨迹获得。

**📈 对比分析**

与 FOCAL、CSRO、DORA、UNICORN‑SS/UP 等基线对比，SPC‑OMRL 在 MuJoCo、Contextual‑DMC 和 Meta‑World 上实现了更高的 few‑shot/zero‑shot 回报和成功率，显著优于现有方法。

**⚠️ 局限性**

局限性包括：对完全新环境的泛化仍有限，需要足够多的训练任务来提升性能；离散潜在码表与 Gumbel‑Softmax 的训练相对复杂；仅在离线数据下验证，在线适应性和实时性能尚未探究。

---

## 80. Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing

**arXiv ID:** 2603.02742 | [PDF](https://arxiv.org/pdf/2603.02742v1)

**作者:** Maulana Bisyir Azhari `[一作]` (Korea Advanced Institute of Science and Technology), David Hyunchul Shim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5727 | [OpenAlex ID](https://openalex.org/A5053591461)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了适用于高速无人机竞速的单目视觉-惯性状态估计框架ADR-VINS，并配套离线全图优化工具ADR-FGO，用于在无GNSS、无MoCap的赛场环境中评估与提升估计精度。

**💡 创新点**

创新点在于：①直接将门角重投影误差嵌入错误状态卡尔曼滤波器，省去PnP和RANSAC，允许仅两角更新；②采用Huber加权重避免离群，提升鲁棒性；③通过ADR-FGO实现全局批量优化，为未标定环境提供高精度参考轨迹。

**🔧 技术方法**

技术包括：单目视觉目标检测（RTMO-nano），误差状态卡尔曼滤波（ESKF），Huber鲁棒权重，IMU预积分因子，门角重投影因子，外参软先验因子，利用SymForce求解雅可比；实时实现基于TensorRT FP16。

**📊 数据集**

使用公开的TII‑RATM室内数据集（含500 Hz IMU、120 Hz单目、275 Hz MoCap）进行评估，并在A2RL Drone Championship Season 2的真实赛道（无MoCap）中部署验证。

**📈 对比分析**

与OpenVINS、PnP+EKF等在线基线以及MAPLAB离线全图优化对比；ADR‑VINS在TII‑RATM上平均平移RMSE为0.141 m，速度误差和旋转误差分别降低约70 %；ADR‑FGO进一步将平移误差压至0.060 m，比MAPLAB提升88 %。在A2RL赛道，ADR‑VINS实现最高20.9 m/s，平移误差0.152 m，显著优于PnP+EKF。

**⚠️ 局限性**

主要局限在于门角检测的标签质量和训练数据不足，未利用外角信息；在长时间无视觉区间时IMU漂移仍是挑战；对高频视觉特征缺乏利用，导致旋转误差提升有限。

---

## 81. The Geometry of Learning Under AI Delegation

**arXiv ID:** 2603.02950 | [PDF](https://arxiv.org/pdf/2603.02950v1)

**作者:** Lingxiao Huang `[一作]` (Nanjing University), Nisheeth K. Vishnoi `[通讯]` (Yale University)

**通讯引用:** 2852 | [OpenAlex ID](https://openalex.org/A5063089732)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一个基于耦合动力学系统的理论框架，用来分析 AI 辅助下人类技能随时间的演化及其长期影响；

**💡 创新点**

核心创新在于将 AI 交互与人类学习过程同一性能指标下的局部最优更新放在一起，揭示了自适应委托导致的全局稳定结构改变、出现低技能稳态和不可逆的学习路径；

**🔧 技术方法**

主要采用了微分方程（连续时间）与离散时间随机逼近（stochastic approximation）的方法，对动力学进行定性与量化分析，包括固定点、线性化、稳定性、分岔、分界面（stable manifold）以及噪声与非对称更新的鲁棒性证明；

**📊 数据集**

本文并未使用真实数据集，而是通过理论推导和仿真示例展示模型，并提出基于实验观测（如委托概率、即时损失）估计模型参数的流程；

**📈 对比分析**

由于研究聚焦于理论分析，没有与实测或其他方法直接比较；论文通过定量理论结果表明：短期内 AI 辅助能显著降低任务误差，但长期会导致技能退化，且高质量 AI 进一步扩大低技能区域；

**⚠️ 局限性**

局限性包括：模型仅考虑单一任务且技能为标量；缺乏噪声、后验校验、策略性投入等现实因素；未验证跨任务迁移、奖励机制或验证机制的影响；因此实际应用需进一步扩展与实证验证。

---

## 82. Learning graph topology from metapopulation epidemic encoder-decoder

**arXiv ID:** 2603.02349 | [PDF](https://arxiv.org/pdf/2603.02349v1)

**作者:** Xin Li `[一作]` (Ben-Gurion University of the Negev), Rami Puzis `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 2413 | [OpenAlex ID](https://openalex.org/A5059447087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

基于深度学习的编码器-解码器框架（DTEF），从多种病原体的时间序列感染数据中同时推断元群落网络拓扑与 SIR 模型参数。

**💡 创新点**

①无需先验假设网络结构或参数；②能一次性联合推断网络与参数；③利用多病原体信息显著提升识别度。

**🔧 技术方法**

深度学习编码器-解码器（dti + efb）+ 余弦相似度嵌入、梯度自回传、正则化方差最小化。

**📊 数据集**

合成随机图（ER、BA、WS、RGG）以及真实交通/地理图（美国/中国/欧盟/非洲边界图、德国/美国移动网络、全球/美国航班网络、西班牙公交/汽车/航班/火车网络）。

**📈 对比分析**

与 Infer2018、随机基线相比，使用谱相似度、Pearson、Jaccard、PR‑AUC 四种指标评估，DTEF 在大多数图（尤其是 WS、RGG、真实边界图）上均取得最高或相当的准确率。

**⚠️ 局限性**

局限：假设网络在观测窗口内保持静态；依赖感染时间序列的稠密度与准确性；仅适用于假设的 SIR 拓扑传播机制，若实际病原体传播模式不同则推断会偏差。

---

## 83. Yeo's Theorem for Locally Colored Graphs: the Path to Sequentialization in Linear Logic

**arXiv ID:** 2603.03262 | [PDF](https://arxiv.org/pdf/2603.03262v1)

**作者:** Rémi Di Guardia `[一作]` (Université Paris Cité), Lionel Vaux Auclair `[通讯]` (Aix Marseille University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种局部着色的图论框架，利用 cusp 最小化的技巧重新证明 Yeo 定理，并将其推广到参数化版本；随后将这一结果直接应用于线性逻辑的证明网（多重与加法型、含 Mix 规则等），在不改变图结构的前提下获得拆分顶点，从而实现证明网的序列化；

**💡 创新点**

创新点在于：1）使用局部（半边）着色而非传统全边着色，避免了图结构改动；2）通过 cusp 最小化得到一个全新的、简洁的 Yeo 定理证明；3）构造参数化的局部 Yeo 定理，能够以模块化方式选取拆分顶点，统一并简化多种已有的证明网序列化证明；

**🔧 技术方法**

主要技术包括：图论中的局部着色、 cusp 定义与最小化、严格偏序关系构造、诱导/归纳证明、以及对证明网的分离、切换、连通性分析；

**📊 数据集**

无实验数据集，本文完全基于理论证明和图示例；

**📈 对比分析**

与以往基于切换循环、完美匹配、转折顶点等多种判据的序列化方法相比，本文提供了更直接、统一且无额外编码的证明；在理论上实现了对所有这些判据的衍生与等价性证明；

**⚠️ 局限性**

局限性：1）目前仅针对无单位的乘法/加法线性逻辑（及其 Mix 变体）；2）对存在交错环（alternating cycles）的图需要进一步技术扩展；3）虽然理论上简化了证明，但尚未给出实现或复杂度分析。

---

## 84. R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild

**arXiv ID:** 2603.02801 | [PDF](https://arxiv.org/pdf/2603.02801v1)

**作者:** Margherita Lea Corona `[一作]` (Fraunhofer Heinrich Hertz Institute), Anna Hilsmann `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种名为R3GW的系统，能够在野外拍摄的户外场景中通过3D高斯展开（Gaussian Splatting）实现可重光照的三维重建和新视角合成。

**💡 创新点**

创新点包括：①将场景分为可重光照的前景Gaussians和不受材质影响的天空Gaussians，形成解耦的天空‑前景表示；②在3DGS中嵌入物理基础渲染（PBR）与Cook‑Torrance BRDF，支持视角相关的高光和可变光照；③使用球谐（SH）环境光和每张图像的嵌入向量，学习场景在不同光照下的环境光。

**🔧 技术方法**

技术手段包括：3D Gaussian Splatting、基于SH的环境光表示（最高4阶）、Cook‑Torrance 微面反射模型、Disney 材质参数、分离和逼近技术（split‑sum approximation）、MLP预测SH系数、重建损失与多种正则化（光照正则、法线一致性、尺度约束、天空‑前景分离约束、天空深度约束）。

**📊 数据集**

使用了NeRF-OSR数据集（八个野外户外场景），并利用COLMAP提供的相机参数与稀疏点云进行初始化。

**📈 对比分析**

通过与NeRF基准方法（NeRF-OSR、SR-TensoRF、FEGR、SOL-NeRF、NeuSky）以及Gaussian Splatting基准（LumiGauss）进行对比，R3GW在平均SSIM上取得最高分，PSNR与SR‑TensoRF相当，训练时间约2小时，显著低于NeuSky的14小时；在无阴影版的LumiGauss上也表现更好。消除视角相关高光或天空深度约束会导致PSNR/SSIM下降，验证了两项创新贡献。

**⚠️ 局限性**

主要局限：未显式建模投射阴影与间接照明，导致在有明显阴影或强光场景下效果受限；缺少方向光源与阴影映射，难以生成尖锐高光；天空与环境光的耦合未实现，可能影响在极端光照条件下的重光照效果。

---

## 85. Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement

**arXiv ID:** 2603.02641 | [PDF](https://arxiv.org/pdf/2603.02641v1)

**作者:** Szu-Wei Fu `[一作]` (NVIDIA), Yu-Chiang Frank Wang `[通讯]` (Academia Sinica)

**通讯引用:** 6729 | [OpenAlex ID](https://openalex.org/A5090045508)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种通用语音增强框架，解决训练目标、感知-保真权衡和数据质量三大关键瓶颈。

**💡 创新点**

创新点在于：①使用时间移位的无回声干净语音作为去混响训练目标，显著提升感知质量与 ASR 准确率；②提出基于两阶段（回归+生成）最优传输框架，实现低失真下的感知提升；③通过 VQScore 过滤和最高质量子集微调，阐明数据质量对 USE 性能的决定性影响。

**🔧 技术方法**

技术手段包括：SFI-STFT、USEMamba 回归网络、Wasserstein GAN 生成器与多带判别器、两阶段训练策略、VQScore 质量评估与过滤。

**📊 数据集**

使用 URGENT 2025 Challenge 训练集（≈2500 h，多语言多采样率、七类失真）以及 EARS、FLEURS 等高质量语料进行微调与评估。

**📈 对比分析**

与官方基线、TOP 3 系统及开源模型（ClearerVoice‑Studio、Resemble Enhance）对比，在非侵入式指标（DNSMOS、NISQA、UTMOS）、ASR CAcc 以及多语言 TTS 评估中均取得最优或显著提升。

**⚠️ 局限性**

局限性在于：训练目标与评测参考不一致导致官方指标偏低；对极低质量样本的去噪仍有限；生成模型对语言细节的微调需求较高，导致在极端语音条件下仍可能出现轻微人声失真。

---

## 86. When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus

**arXiv ID:** 2603.02364 | [PDF](https://arxiv.org/pdf/2603.02364v1)

**作者:** Kirill Borodin `[一作]` (Moscow Technical University of Communications and Informatics), Grach Mkrtchian `[通讯]` (Moscow Technical University of Communications and Informatics)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖 66 种语言、2,732 小时、包含 45 种低资源语言的多语种伪造语音语料库 LRLspoof，并在该语料库上评估了 11 种公开的伪造检测模型在跨语言条件下的鲁棒性。

**💡 创新点**

首次提供了大规模多语种、低资源语言丰富且可控的合成语音数据集，采用阈值迁移与控制实验来揭示语言不匹配对伪造检测性能的显著影响。

**🔧 技术方法**

使用 24 种开源 TTS 合成器生成语料，采用 EER 迁移阈值法（在 ASVspoof5、ASVSpoof2021 LA/DF、In-the-wild、DFADD、ADD2022 等基准上校准）以及固定模型/固定合成器的对照实验；评估工具包括 11 种公开的伪造检测器。

**📊 数据集**

主要使用新构建的 LRLspoof 语料库；阈值校准时使用外部基准 ASVspoof5、ASVSpoof2021 LA/DF、In-the-wild、DFADD、ADD2022 等。

**📈 对比分析**

方法为：先在汇总的外部基准上计算 EER 并得到统一阈值，然后将该阈值直接套用到 LRLspoof 上测量 spoof rejection rate (SRR)。结果显示不同模型、语言和 TTS 组合的 SRR 差异巨大，低资源语言和某些语言-合成器组合的鲁棒性明显下降。

**⚠️ 局限性**

主要局限在于仅使用伪造语音，缺少真实语音进行完整的 FRR/FRR 评估；阈值迁移受限于外部基准的语言覆盖；未探究模型在真实应用场景下的安全性和适应性。

---

## 87. Neural Paging: Learning Context Management Policies for Turing-Complete Agents

**arXiv ID:** 2603.02228 | [PDF](https://arxiv.org/pdf/2603.02228v1)

**作者:** Liang Chen `[一作]`, Qi Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Neural Paging 和 H-NTM 架构，用来在 LLM 代理中实现可学习的上下文分页管理，从而在固定上下文窗口内提升长时序推理能力。

**💡 创新点**

创新点在于：① 通过“上下文分页问题” (CPP) 形式化了 LLM 的语义缓存管理；② 引入“有界敏感性”模型和对应的鲁棒性定理 (Thm 4)；③ 设计了轻量级可微分的页控制器，实现对 Belady 最优决策的近似；④ 通过合成 Zipf 访问轨迹验证理论并揭示现有贪婪策略与最优之间的巨大空间。

**🔧 技术方法**

使用的技术包括：分层神经图灵机 (Hierarchical Neural Turing Machine)、可微分的页控制器网络、近似语义价值估计、强化学习 (PPO) 训练页策略、竞争性分析与误差传播理论。

**📊 数据集**

数据集主要是合成的访问轨迹（Zipf 分布、工作集切换）以及少量检索误差控制实验；未使用真实 LLM 任务数据。

**📈 对比分析**

与 Belady、LRU、LFU、FIFO、Random 等基准在同一合成轨迹下比较；实验显示 LRU 的竞争比为约1.9×最优（远低于理论上 K_b=8 的 worst‑case 8），且页误差随敏感性 β 线性增长，满足 Thm 4 的上界。

**⚠️ 局限性**

主要局限：未在真实 LLM 代理上进行端到端评估；页控制器需任务专门训练，跨域迁移未知；假设检索始终可靠，若失败会导致不可恢复错误；理论上限过于保守，缺乏实例依赖的更紧凑界定。

---

## 88. Integrating Homomorphic Encryption and Synthetic Data in FL for Privacy and Learning Quality

**arXiv ID:** 2603.02969 | [PDF](https://arxiv.org/pdf/2603.02969v1)

**作者:** Yenan Wang `[一作]` (Chalmers University of Technology), Elad Michael Schiller `[通讯]` (Chalmers University of Technology)

**通讯引用:** 1063 | [OpenAlex ID](https://openalex.org/A5043628304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在联邦学习中，提出了一种名为nameNovel的框架，通过交替使用真实数据和合成数据进行本地训练，并在真实数据轮中采用选择性同态加密（Selective HE）以实现隐私保护，同时在合成数据轮中传输明文模型参数，降低加密开销。

**💡 创新点**

创新点在于将同态加密与合成数据生成结合，并采用可调的交替比例ρ实现隐私与性能的平衡，首次通过交替轮次减少加密/解密成本并提升模型精度。

**🔧 技术方法**

技术主要包括联邦平均（FedAvg）、选择性同态加密、合成数据生成（统计相似但不重叠的数据）、DLG攻击评估以及UQI/MSSSIM/VIF等相似度指标。

**📊 数据集**

实验使用CIFAR‑10数据集，将其一半作为真实数据、另一半作为合成数据，在三客户端上采用LeNet‑5网络进行分类任务。

**📈 对比分析**

与全加密（η=1）和仅使用选择性加密无交替（ρ=0）的基线相比，nameNovel在相同加密比例η=0.2下提升了约13.4%的准确率，同时将HE相关加密/解密时间减少约48%，总体通信量下降约39%。

**⚠️ 局限性**

局限性包括收敛速度略慢（需要更多训练轮次），合成数据质量和分布可能影响模型泛化，以及对DLG攻击的评估仅限于单张图像、单一梯度信息，未覆盖更复杂的攻击场景。

---

## 89. Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs

**arXiv ID:** 2603.02353 | [PDF](https://arxiv.org/pdf/2603.02353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 90. Agentic Mixed-Source Multi-Modal Misinformation Detection with Adaptive Test-Time Scaling

**arXiv ID:** 2603.02519 | [PDF](https://arxiv.org/pdf/2603.02519v1)

**作者:** Wei Jiang `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17326 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgentM^3D 多代理框架，结合自适应测试时缩放、批判式 Best‑of‑N、规划和早停机制，实现零样本混源多模谣言检测。

**💡 创新点**

创新点包括：① 用规划代理动态决定是否使用 Best‑of‑N 进行扩展推理；② 将奖励模型与模态特定批判评分融合，形成批判式 Best‑of‑N；③ 采用层级化模态检测与早停来降低错误传播与计算成本。

**🔧 技术方法**

技术手段：多模视觉语言模型（Qwen3‑VL）、Best‑of‑N 测试时缩放、奖励模型（GRM‑Gemma2‑2B）、批判代理、规划代理、早停阈值。

**📊 数据集**

数据集：MMFakeBench（含 11,000+ 图文对）和自构造的 Combined benchmark（融合 Mocheg、Fakeddit‑M、VERITE 共 900+ 例）。

**📈 对比分析**

比较方式：与标准 VLM、BoN、T^2Agent、MMD‑Agent 及其 BoN 版本在两大基准上对照，AgentM^3D 在准确率、F1、召回率、精确率等指标上均超过对手，同时在准确率‑延迟折中取得最优平衡。

**⚠️ 局限性**

局限性：依赖奖励模型和批判模型的对齐效果，过多批判或不合适的阈值可能导致噪声引入；缺乏跨语言/跨平台的鲁棒性验证；对推理可解释性的深入分析仍待补充。

---

## 91. CGL: Advancing Continual GUI Learning via Reinforcement Fine-Tuning

**arXiv ID:** 2603.02951 | [PDF](https://arxiv.org/pdf/2603.02951v1)

**作者:** Zhenquan Yao `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62202 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出连续GUI学习框架CGL，解决GUI持续学习中的稳定性-可塑性权衡，并构建AndroidControl‑CL基准；

**💡 创新点**

创新点包括：①错误感知路由将SFT用于RL探索失败时的指导；②熵调节动态SFT权重λ以平衡探索与记忆；③梯度外科对抗SFT与GRPO梯度冲突；④系统化的AndroidControl‑CL持续学习评估；

**🔧 技术方法**

采用多模态大语言模型（LLaVA、Qwen‑VL），Supervised Fine‑Tuning（SFT）与Group Relative Policy Optimization（GRPO）融合，结合熵调节、KL约束、梯度外科等技术；

**📊 数据集**

使用AndroidControl‑CL数据集（由原AndroidControl扩展并按7个功能超类划分的任务集合），并在三种任务顺序下进行评测；

**📈 对比分析**

对比SFT、SFT+KL、SFT+Replay、GRPO、RIF‑RFT及SFT‑Joint‑Training等基线。CGL在三种任务顺序下均实现最高的Step Acc（≈82.33%/77.84%）和Trajectory Acc（≈38.03%/24.77%），并将forgetting measure（FM）降至接近零，甚至在某一顺序下出现正FM，显著优于所有基线；

**⚠️ 局限性**

局限性包括：需依赖大规模模型与高算力；梯度外科与熵调节对其他RL任务的泛化尚未验证；基准和实验主要集中在Android GUI，缺乏跨平台或真实软件迭代场景验证。

---

## 92. Odin: Multi-Signal Graph Intelligence for Autonomous Discovery in Knowledge Graphs

**arXiv ID:** 2603.03097 | [PDF](https://arxiv.org/pdf/2603.03097v1)

**作者:** Muyukani Kizito `[一作]` (Prescott Data), Elizabeth Nyambere `[通讯]` (Prescott Data)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Odin，一个用于知识图谱自主发现的图智能引擎，提供无查询前置的路径探索；

**💡 创新点**

创新点在于 COMPASS 多信号评分（结构、语义、时间、社区桥接）和桥接评分机制解决回声室问题；

**🔧 技术方法**

使用技术包括 Personalized PageRank、Neural Probabilistic Logic Learning、GAT 社区检测、Beam Search 以及多信号组合；

**📊 数据集**

评估数据集为两大生产 KG：医疗 KG（230 万实体、870 万三元组）和保险 KG（180 万实体、620 万三元组）；

**📈 对比分析**

与全遍历、随机游走、PPR-only、GNN Embedding 对比，Odin 在 65 倍路径减少的同时覆盖率高 90%，质量评分 4.2/5；

**⚠️ 局限性**

局限包括冷启动 NPLL 训练时延、对动态图更新需求、单 KG 受限、未实现跨图发现以及对 Hypergraph 的理论改进空间。

---

## 93. Agentified Assessment of Logical Reasoning Agents

**arXiv ID:** 2603.02788 | [PDF](https://arxiv.org/pdf/2603.02788v1)

**作者:** Zhiyu Ni `[一作]` (University of California), Zheng Liang `[通讯]` (University of California)

**通讯引用:** 23679 | [OpenAlex ID](https://openalex.org/A5027442022)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将评估过程本身实现为代理的评估框架，并用其评测逻辑推理代理。

**💡 创新点**

创新点在于将评估器与被测代理解耦，采用标准化的代理对代理（A2A）接口，使评估可重复、可审计且对执行失败具有鲁棒性。

**🔧 技术方法**

使用了自评估器代理、自动形式化代理（将自然语言转为Z3Py代码并执行SMT求解）、LLM（Gemini 2.5 Flash）以及符号验证工具（Vampire、Z3）。

**📊 数据集**

在经过验证与修复的FOLIO（第一阶逻辑推理）数据集上进行实验。

**📈 对比分析**

与链式思维基线对比，自动形式化代理在验证集上取得86.70%准确率，远高于73.89%，在False和Uncertain类上提升显著。

**⚠️ 局限性**

局限包括仅针对FOLIO单一领域，缺乏多样化基准；评估仍需手动复核部分样本；模型仅使用一款LLM，泛化性待进一步验证。

---

## 94. VL-KGE: Vision-Language Models Meet Knowledge Graph Embeddings

**arXiv ID:** 2603.02435 | [PDF](https://arxiv.org/pdf/2603.02435v1)

**作者:** Athanasios Efthymiou `[一作]` (University of Amsterdam), Marcel Worring `[通讯]` (University of Amsterdam)

**通讯引用:** 16064 | [OpenAlex ID](https://openalex.org/A5070684680)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 VL-KGE 框架，将预训练的视觉–语言模型与知识图谱嵌入技术融合，以支持多模态实体的统一表示与推理。

**💡 创新点**

创新点在于通过 VLM 实现跨模态对齐；在实体表示中处理模态不对称；并支持在未知实体上的归纳推理。

**🔧 技术方法**

使用了预训练的视觉–语言模型（CLIP/BLIP）、多模态融合策略（平均/加权/拼接）、常用 KGE 骨干（TransE/DistMult/ComplEx/RotatE）以及对抗/注意力机制。

**📊 数据集**

实验数据集包括 WN9‑IMG、WikiArt‑MKG‑v1 与 WikiArt‑MKG‑v2（扩展的细节艺术多模态知识图谱）。

**📈 对比分析**

在 link prediction（MRR、Hits@K）上与单模态和现有多模态 KGE 基线（MMKRL、OTKGE、VB‑KGE）进行对比，VL‑KGE 在所有数据集上均优于基线，尤其在模态不对称的 WikiArt 上提升显著。

**⚠️ 局限性**

限制在于对预训练 VLM 的领域对齐依赖较强；对极端稀疏模态或长尾实体的表现仍有限；以及在大规模图谱上的计算开销相对较高。

---

## 95. CUDABench: Benchmarking LLMs for Text-to-CUDA Generation

**arXiv ID:** 2603.02236 | [PDF](https://arxiv.org/pdf/2603.02236v1)

**作者:** Jiace Zhu `[一作]` (Shanghai Jiao Tong University), An Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5047770048)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了CUDABench基准，系统评估大型语言模型在从自然语言生成CUDA内核代码的能力。

**💡 创新点**

创新点在于（1）构建了覆盖六大GPU计算领域、五级输入规模与三难度级别的文本到CUDA数据集CUDABench-Set；（2）提出了基于roofline模型的硬件无关性能指标Performance-Score及其综合评分CUDABench-Score；（3）设计了自动化的生成验证流水线，包括数据生成、编译、功能验证与性能剖面。

**🔧 技术方法**

采用了LLM API进行代码生成，使用NVCC进行编译，借助NVIDIA Nsight Compute获取执行时间、FLOPs与数据移动量，利用roofline模型计算算术强度与可实现性能，并用Pass@k统计评估编译成功率、功能正确性与性能得分。

**📊 数据集**

使用了CUDABench-Set，该数据集包含1500个提示（500个任务 × 3难度级别），任务覆盖线性代数、深度学习算子、计算机视觉与图像处理、数据分析、信号处理以及科学模拟与金融等六大领域，且为每个任务提供多级输入规模与参考CUDA实现。

**📈 对比分析**

通过Pass@1/3的编译成功率、功能一致率以及Performance-Score来综合评估模型；实验显示Claude 4.5 Sonnet在正确率方面最高（约99%编译率、约86%功能一致率），GPT‑5.2在性能得分方面领先（约40% Performance-Score），但整体CUDABench-Score仍低于60%，表明模型在性能利用率方面存在不足。

**⚠️ 局限性**

局限性主要体现在：LLM在功能正确性上存在显著差距；在缺乏算法描述或硬件提示的零样本场景下性能急剧下降，说明其领域知识检索能力有限；以及生成的内核普遍未能充分利用GPU的计算与内存带宽，导致执行性能低下。

---

## 96. AWDiff: An a trous wavelet diffusion model for lung ultrasound image synthesis

**arXiv ID:** 2603.03125 | [PDF](https://arxiv.org/pdf/2603.03125v1)

**作者:** Maryam Heidari `[一作]` (University of Bristol), Alin Achim `[通讯]` (University of Bristol)

**通讯引用:** 4808 | [OpenAlex ID](https://openalex.org/A5001624119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 AWDiff，一种基于扩散模型的肺部超声图像数据增强框架，能够在保持细节结构的同时生成多样化的合成图像。

**💡 创新点**

创新点在于：1）采用 à trous 小波编码器实现多尺度结构编码，避免下采样导致的细节丢失；2）通过 BioMedCLIP 的文本嵌入实现语义条件，使生成结果与临床标签（如 B‑line 数量、胸膜不规则性）保持一致。

**🔧 技术方法**

技术包括：扩散概率模型（DDPM）+ UNet 去噪器；à trous 小波变换提取多尺度高频特征；BioMedCLIP 文本/图像嵌入与交叉注意力融合；损失函数结合 MSE 预测噪声与 BioMedCLIP 对齐损失。

**📊 数据集**

使用了 360 张透析相关肺部超声扫描的临床数据集，并通过 AWDiff 生成合成样本扩充到 2,260 张，用于后续模型微调和评估。

**📈 对比分析**

与 SinDDM 与 SinGAN 进行对比，使用 SIFID、LPIPS、NIMA、CW‑SSIM 等指标评估；AWDiff 在 120k 步时取得 SIFID=0.03、LPIPS=0.37、NIMA=5.45，明显优于对手，同时在保留 B‑line 和胸膜连续性方面得到专家认可。

**⚠️ 局限性**

局限性包括：仅在透析相关肺部超声数据上验证，未检验对其他疾病或采集协议的泛化；对 BioMedCLIP 预训练模型的依赖可能限制新的临床标签空间；模型训练和推理成本较高。

---

## 97. Grounded String Representations of Series-Parallel Graphs without Transitive Edges

**arXiv ID:** 2603.02827 | [PDF](https://arxiv.org/pdf/2603.02827v1)

**作者:** Sabine Cornelsen `[一作]` (University of Konstanz), Alexander Wolff `[通讯]` (Universität Würzburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图的有界弧（grounded string）表示，重点关注每个顶点用一弯正交多段线表示的情况，并进一步限制为L型（即左下角为唯一弯点的形状）表示。

**💡 创新点**

创新点在于：
1) 对无传递边的二连通系列并行图给出了必要且充分的条件，使得该类图的有界L表示存在 iff 有界字符串表示存在；
2) 提出了线性时间的检测算法；
3) 构造了一个既存在有界L表示又不存在有界L-表示的例子，揭示了两种表示在更一般图中的不等价性。

**🔧 技术方法**

主要技术包括：
- 分离点对（separation pair）和重/轻组件分类；
- 对重组件的循环结构（heavy cycle）构造；
- 逐层处理轻组件并在多边形内部绘制曲线；
- 结合图分解树实现线性时间算法。

**📊 数据集**

论文属于理论计算机科学，未使用实验数据集，全部以数学证明和构造例子完成。

**📈 对比分析**

与现有的外部-字符串（outerstring）或一般1弯表示的比较在于证明了两类表示在该类图上的等价性，并给出了 O(n+m) 的线性时间判定算法；在构造的反例中展示了更广泛类图无法满足等价性。

**⚠️ 局限性**

局限性：
- 仅针对无传递边的二连通系列并行图；
- 对包含传递边或更一般图的情况不适用；
- 仅证明了L表示与一般字符串表示的等价性，并未探讨多弯或非正交多段线的情况。

---

## 98. A Comparative Study of UMAP and Other Dimensionality Reduction Methods

**arXiv ID:** 2603.02275 | [PDF](https://arxiv.org/pdf/2603.02275v1)

**作者:** Guanzhe Zhang `[一作]` (University of Delaware), Zhezhen Jin `[通讯]` (Columbia University)

**通讯引用:** 10507 | [OpenAlex ID](https://openalex.org/A5053607185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并比较了UMAP（含监督版）与PCA、KPCA、SIR、KSIR、t‑SNE等降维方法在回归与分类任务中的表现，并在模拟与真实数据上系统评估其预测效果。

**💡 创新点**

首次系统地对监督UMAP在回归情形下的性能进行实验评估，并提出通过对连续响应进行分箱再作为类别处理以降低过拟合的思路。

**🔧 技术方法**

使用UMAP（有无监督）、PCA、KPCA、SIR、KSIR、t‑SNE等降维技术，并以KNN回归/分类器作为统一评估指标。

**📊 数据集**

模拟数据（3种特征分布×4种响应模型共12套），Fashion‑MNIST图像分类数据，以及Online News Popularity连续响应数据。

**📈 对比分析**

通过在降维后使用KNN计算MSE（回归）或误分类率（分类）与原始数据对比，结果显示监督UMAP在分类任务上表现最佳，SIR在回归任务上最优，监督UMAP在回归时往往不如无监督或线性方法，t‑SNE速度慢且测试表现一般。

**⚠️ 局限性**

当前监督UMAP方法在处理连续响应时难以有效利用响应信息，往往出现过拟合或性能低于无监督版本，缺乏适合回归的改进方案。

---

## 99. DINOv3 Visual Representations for Blueberry Perception Toward Robotic Harvesting

**arXiv ID:** 2603.02419 | [PDF](https://arxiv.org/pdf/2603.02419v1)

**作者:** Rui-Feng Wang `[一作]` (University of Florida), Changying Li `[通讯]` (University of Florida)

**通讯引用:** 7051 | [OpenAlex ID](https://openalex.org/A5007872497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在蓝莓采摘机器人视觉感知中，作者评估了冻结的DINOv3模型作为通用视觉编码器，并在统一协议下训练轻量级的分割与检测解码器。

**💡 创新点**

创新点在于系统性比较不同规模的DINOv3背骨在蓝莓任务中的迁移性能，揭示了检测瓶颈与聚合目标的结构限制，并提出以冻结模型为语义骨干的设计思路。

**🔧 技术方法**

使用了DINOv3自监督视觉模型、ViT‑S/16、ViT‑S+/16、ViT‑B/16、ViT‑L/16四种变体、基于patch的轻量级解码器，并采用mIoU、mAP等标准评价指标。

**📊 数据集**

实验数据集来自BSAIL实验室，包括果皮分割、果肉损伤分割、果实检测和聚集检测四个蓝莓数据集，覆盖多种拍摄环境与目标尺度。

**📈 对比分析**

在冻结模型与轻量级头的对比实验中，分割任务随背骨规模提升表现稳定提升，mIoU最高可达约70%；检测任务在果实检测上有轻微提升，但聚集检测几乎无效，mAP低于5%。

**⚠️ 局限性**

主要限制是检测受patch离散化与尺度不匹配影响，聚合目标难以用单个框表示，导致结构性瓶颈，仅扩大背骨规模无法解决这些问题。

---

## 100. SynthCharge: An Electric Vehicle Routing Instance Generator with Feasibility Screening to Enable Learning-Based Optimization and Benchmarking

**arXiv ID:** 2603.03230 | [PDF](https://arxiv.org/pdf/2603.03230v1)

**作者:** Mertcan Daysalilar `[一作]` (University of Miami), Adam Meyers `[通讯]` (University of Miami)

**通讯引用:** 2570 | [OpenAlex ID](https://openalex.org/A5043134485)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SynthCharge，一个可参数化、可控的电动车辆路线规划（EVRPTW）实例生成器，并对生成的实例进行结构化可行性筛选。

**💡 创新点**

创新点在于：① 将空间拓扑、时间窗宽度、能源容量等维度完全可控；② 采用两阶段可行性筛查（线性结构检查 + 对小规模实例的 MILP 验证）；③ 提供统一的元数据日志，实现完全可复现和分布式评估。

**🔧 技术方法**

技术包括：随机几何生成（均匀、聚类、混合）、范围感知充电站布置、能量容量自适应缩放、宽松/中等/严格时间窗分配、结构可行性检查以及可选的精确 MILP 验证。

**📊 数据集**

使用的“数据集”是 SynthCharge 生成的合成实例，覆盖 5 至 100 顾客、3 种空间布局（随机、聚类、混合）和 3 种时间窗宽度（宽、中、紧），共计 2,475 个实例（含可行性标签）。

**📈 对比分析**

比较方法：在所有生成实例上运行基线变邻域搜索/禁忌搜索元启发式，验证 100% 的实例可行；同时统计平均生成时间、接受率 γ（随规模、空间、时间窗变化）以及平均行驶距离和车辆需求。性能表现显示：接受率在 37–47% 之间，生成时间保持在 0.01–0.10 秒；时间窗紧缩或空间稀疏化显著提高车辆需求和行驶距离。

**⚠️ 局限性**

局限性：① MILP 验证仅限于 N ≤ 10；② 结构筛查无法保证全局可行性；③ 仅考虑欧氏平面、单仓库、线性能耗与完整充电；未涵盖非欧几里得道路、时间变动、随机性、多仓库或部分充电策略。

---

## 101. Adapting Time Series Foundation Models through Data Mixtures

**arXiv ID:** 2603.02840 | [PDF](https://arxiv.org/pdf/2603.02840v1)

**作者:** Thomas L. Lee `[一作]` (University of Edinburgh), Amos Storkey `[通讯]` (University of Edinburgh)

**通讯引用:** 13696 | [OpenAlex ID](https://openalex.org/A5007901825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了MixFT方法，利用贝叶斯混合模型重新划分时间序列微调数据，训练针对子域的LoRA模块，以提升零样本预测性能。

**💡 创新点**

创新点在于不按数据集划分，而是通过贝叶斯GMM识别子域并按子域划分数据，训练子域专属LoRA，显著降低分布差异并提升零样本效果。

**🔧 技术方法**

采用LoRA参数高效微调、贝叶斯高斯混合模型（带NIDW先验）、变分推断、混合组件选择及Arrow路由等技术。

**📊 数据集**

实验使用Cloud、Gift‑Eval基准中的多数据集（CloudD1‑4、BizITObs系列、M4日/月/季、ETTh2/ETTm2），并在Chronos Bolt与Moirai‑1.1‑R两大TSFM上进行评估。

**📈 对比分析**

与共享LoRA、Per‑dataset方法（μ、Arrow、Poly、MBC）及未微调基线对比，使用MASE评估，MixFT在大多数数据集上取得最低平均排名（最佳），并在使用Arrow路由统一选择时仍保持优势。

**⚠️ 局限性**

局限性包括在某些数据集（如M4‑Daily）仍不优于基线；子域数量K需手动或通过验证确定；对极端多子域或显著分布漂移的数据可能效果受限。

---

## 102. Behavior Change as a Signal for Identifying Social Media Manipulation

**arXiv ID:** 2603.03128 | [PDF](https://arxiv.org/pdf/2603.03128v1)

**作者:** Isuru Ariyarathne `[一作]` (William and Mary), Alexander C. Nwala `[通讯]` (William and Mary)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用社交媒体账户行为随时间变化的分布特征来检测自动化和协同不真实账户，构建基于BLOC表示、分段、距离测量的特征集，并使用KNN进行分类。

**💡 创新点**

创新点在于把行为变化本身视为判别信号，使用分段距离分布特征而非传统静态特征，能够一次性用于多任务（机器人检测与协同检测），且仅依赖用户的行为信息。

**🔧 技术方法**

技术包括：BLOC框架将行为编码为符号串；三种分段策略（暂停、周、k段）和两种选择方式（相邻、累积）；余弦距离与压缩距离两种测距；生成20维分布特征；KNN分类器；与Botometer、Co‑RT、Hashtag、Activity等基线比较。

**📊 数据集**

使用的数据集包括：Twitter Bot Repository（32,056 机器人+42,773 人类）用于机器人检测；AIBot_fox8（1,140 机器人+1,140 人类）和IO信息操作数据集（32 个活动）用于协同检测。

**📈 对比分析**

通过多种分段/距离组合进行交叉验证，挑选最佳 F1；机器人检测中，行为变化模型 F1=0.86，略低于 Botometer‑v4 的 0.92；协同检测中，行为变化模型在 AIBot_fox8 上 F1=0.94，略低于 Co‑RT/Activity 的 0.99，IO 数据集平均 F1=0.88，显著优于 Hashtag/Co‑RT/Activity 的 0.53–0.76；整体表现良好。

**⚠️ 局限性**

局限性：仅考虑动作和内容两维行为，未纳入主题、互动网络或时间同步等信息；分段方法对稀疏数据受限；实验仅在 Twitter 平台，跨平台适用性未验证。

---

## 103. His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval

**arXiv ID:** 2603.02617 | [PDF](https://arxiv.org/pdf/2603.02617v1)

**作者:** Shengbo Wang `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34394 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种自动化 C-to-Rust 迁移框架 His2Trans，融合构建感知的项目级 Skeleton 与自进化的知识库，实现从代码片段到工业项目的稳定迁移。

**💡 创新点**

创新点在于：①利用构建追踪生成可编译的项目级 Skeleton Graph，消除“dependency hell”；②从历史迁移轨迹挖掘 API 与代码片段规则，构成检索增强生成（RAG）机制；③通过闭环的编译反馈修复和知识累积实现系统自进化。

**🔧 技术方法**

核心技术包括构建追踪、Skeleton 生成、检索增强生成（RAG）+ LLM 代码生成功能、类型一致性 Skeleton、编译器反馈驱动的修复循环、以及知识库的持续增量更新。

**📊 数据集**

实验数据集涵盖 5 个 OpenHarmony 工业子模块（域特定）和 10 个通用 C 基准（如 ht、qsort、quadtree、bzip2 等），全部公开可复现。

**📈 对比分析**

与 7 个基线（C2Rust、SmartC2Rust、RUSTINE、PTRMAPPER、C2SaferRust、EvoC2Rust、Tymcrat）在增量编译通过率、功能正确率、unsafe 比例和警告数上对比，His2Trans 在工业级实现达到 99.75% 编译通过率、75% 功能正确率、约 45% unsafe 比例、警告数最低，并通过知识累积将修复轮次降低约 60%。

**⚠️ 局限性**

局限性包括：知识库主要来源于 OpenHarmony，可能对非嵌入式领域存在偏倚；功能正确率评估依赖原始测试用例，稀缺时可能低估细粒度错误；系统仍需在复杂业务逻辑中进一步验证和人工复核。

---

## 104. How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference

**arXiv ID:** 2603.03280 | [PDF](https://arxiv.org/pdf/2603.03280v1)

**作者:** Toru Lin `[一作]` (University of California, Berkeley), Jitendra Malik `[通讯]` (University of California, Berkeley)

**通讯引用:** 165872 | [OpenAlex ID](https://openalex.org/A5001594573)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出一种两阶段的学习框架，利用力感知示范数据训练基准剥皮策略，并通过学习的奖励模型与人类偏好进行微调，以实现刀具剥皮任务；

**💡 创新点**

创新点在于（1）将力感知与可调阻尼控制结合，形成鲁棒的初始策略；（2）构建混合量化与主观评估的奖励模型，实现对剥皮质量的细粒度对齐；（3）展示从少量（50–200）真实演示数据即可实现零样本泛化到多种水果；

**🔧 技术方法**

使用Kinova Gen3机械臂与自定义刀具支架、ATI mini45 力/扭矩传感器、RealSense D405 两眼相机；通过SpaceMouse遥控收集数据，使用ResNet-18+MLP编码器、Diffusion Policies训练基准策略；随后用三层MLP学习奖励模型，并在残差网络上做偏好加权行为克隆微调；

**📊 数据集**

数据集包括来自黄瓜、苹果、土豆的50–200条剥皮轨迹（每条为一次剥皮滑行），并在实验中对齐至未见的西葫芦、梨、大根等不同类别水果；

**📈 对比分析**

与基准策略、仅量化奖励、仅主观奖励以及IQL加权监督等对照实验比较，基准策略在已见水果上已达100%成功率，未见水果90%+；微调后成功率提升至90%以上，整体平均质量分从3.8提升至7.1–7.3；

**⚠️ 局限性**

主要局限在于对人工高质量遥控演示的依赖，难以大规模扩展；奖励模型需要少量人工标注，且在不同硬件/环境下对性能的泛化仍有限；

---

## 105. Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?

**arXiv ID:** 2603.02462 | [PDF](https://arxiv.org/pdf/2603.02462v1)

**作者:** Semih Cantürk `[一作]` (Université de Montréal), Guy Wolf `[通讯]` (Mila)

**通讯引用:** 5432 | [OpenAlex ID](https://openalex.org/A5005117825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于GCON网络的无监督学习框架，利用能量函数（Ising/QUBO）解决多种图组合优化任务，并通过计算可归约性指导模型的预训练与微调，实现任务间的可迁移性。

**💡 创新点**

创新点在于：①将可归约性理论与深度迁移学习相结合，利用多任务预训练构建“基础模型”；②在GCON框架下设计了统一的能量损失函数；③提出在互补图上直接微调的策略，克服结构分布偏移；④系统评估多任务“留一法”迁移效果，证明只需一个相关任务即可快速收敛。

**🔧 技术方法**

主要技术包括：Graph Constrained Optimization Network (GCON) 消息传递层；Ising/QUBO 能量损失；无监督概率输出与规则解码；全连接输出头与可逆线性层；多任务 MLP 头；在补图上进行全连接微调；以及多头图变换器（Graph Transformer）等辅助层。

**📊 数据集**

数据集：RB-small（用于单任务基线对比）、BA-small（Barabási–Albert 生成的稀疏图，用于多任务预训练与留一法实验）。

**📈 对比分析**

与传统 GCN/GIN/GATv2、GFN 等基线相比，GCON+能量损失在 MVC、MIS、MaxClique 上达到或超过当前最优结果；在 MaxCut、MDS、K-Coloring 上也表现优于或相当于现有方法。迁移实验显示：预训练+微调往往在 20~200 轮内比从零训练快数倍，且在多数任务上可匹配甚至超越单任务全量训练性能。

**⚠️ 局限性**

局限性包括：①对结构分布偏移的处理仍不够稳健，需更深或更灵活的全局消息传递；②仅在少数可归约任务之间表现突出，某些任务（如 MaxClique、MDS）迁移收益有限；③需要手工挑选预训练任务集合，尚未形成自动化或理论化的任务选择机制；④对大规模图或高维特征的可扩展性尚未充分验证。

---

## 106. Serverless Abstractions for Short-Running, Lightweight Streams

**arXiv ID:** 2603.03089 | [PDF](https://arxiv.org/pdf/2603.03089v1)

**作者:** Natalie Carl `[一作]` (Technische Universität Berlin), David Bermbach `[通讯]` (Technische Universität Berlin)

**通讯引用:** 1888 | [OpenAlex ID](https://openalex.org/A5032206962)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了“流函数”抽象，用于高效处理短生命周期、轻量级、不可预测且需要有状态的流数据。

**💡 创新点**

创新点在于将整个短流视为执行单元、状态生命周期和伸缩单元，结合无服务器弹性与流处理语义，实现迭代器入/出接口。

**🔧 技术方法**

实现基于Go、Docker、NATS的原型，并与Apache Beam/Dataflow、Google Cloud Run（单帧函数）和批处理函数进行对比。

**📊 数据集**

实验使用合成视频流（10帧/秒、160×120像素、20张随机图像循环）作为数据集。

**📈 对比分析**

通过比较冷启动惩罚θ和与理论最小值的处理开销，发现流函数冷启动几乎为零，处理开销低于10ms，较流处理引擎降低约99%。

**⚠️ 局限性**

局限性包括无法水平扩展单流处理、缺乏容错与状态持久化、对垂直扩展的手工分片策略需求，以及仅适用于短小可预测流。

---

## 107. Contextualized Privacy Defense for LLM Agents

**arXiv ID:** 2603.02983 | [PDF](https://arxiv.org/pdf/2603.02983v1)

**作者:** Yule Wen `[一作]` (Tsinghua University), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13440 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM代理的上下文隐私防护，提出了Contextualized Defense Instructing（CDI）以及基于经验的强化学习优化框架。

**💡 创新点**

创新点在于：①把隐私防护从被动屏蔽转为主动的上下文化指导；②使用经验驱动的RL训练教师模型，利用失败轨迹提升鲁棒性与泛化；③统一框架对比Prompting、Guarding与CDI。

**🔧 技术方法**

技术方法包括：LLM代理执行循环、教师模型生成步骤级隐私指令、强化学习（GRPO）对教师模型进行训练、LoRA参数高效微调、数据模拟与攻击搜索。

**📊 数据集**

使用基于PrivacyLens的人脸与敏感信息样本进行构造，共115个场景（100用于测试，15用于训练），覆盖多种社交关系与数据类型。

**📈 对比分析**

在多种攻击（常规与策略性）下，未优化时CDI在隐私保护率（PP）和帮助度（HS）上优于Prompting与Guarding；经过经验优化后，CDI在PP、HS、综合适当披露（AD）上均取得最高分，且对新攻击和不同代理模型具备良好泛化。

**⚠️ 局限性**

局限性包括：①对教师模型的性能依赖较高，轻量模型仍可能不足；②在面对未见攻击时仍存在性能下降；③需要额外的失败轨迹数据和RL训练，成本相对较高。

---

## 108. Fast Matrix Multiplication in Small Formats: Discovering New Schemes with an Open-Source Flip Graph Framework

**arXiv ID:** 2603.02398 | [PDF](https://arxiv.org/pdf/2603.02398v1)

**作者:** A. I. Perminov `[一作]` `[通讯]`, A. I. Perminov

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一个开源 C++ 框架，利用 flip graph 方法对快速矩阵乘法方案进行搜索和优化。

**💡 创新点**

创新点在于同时支持多种系数环（ℤ₂、ℤ₃、ℤ_T），采用位级编码实现高效运算，并通过并行随机游走与 Hensel 升级发现新的 4×4×10 方案及将已知方案重现为 ℤ_T 和整数系数。

**🔧 技术方法**

使用了位级向量编码、OpenMP 并行、随机游走、flip/plus/split/reduction/merge/product/extend/project 等 flip graph 及元操作，以及 Hensel 升级与有理重构。

**📊 数据集**

实验使用了 680 个矩阵乘法格式（尺寸从 2×2×2 到 16×16×16）作为数据集。

**📈 对比分析**

与已知最优上界进行比较，取得多种格式的 rank 改进；最显著的是 4×4×10 方案仅 115 次乘法，ω≈2.80478，优于 Strassen 的指数；其他格式亦出现 1–10 次乘法的提升。

**⚠️ 局限性**

局限在于编码上限仅支持 128 元素（约 11×11×11），导致更大尺寸不可直接处理；Hensel 升级不总能得到紧凑系数；搜索高度依赖算力，难以在有限资源下覆盖所有难点格式。

---

## 109. HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval

**arXiv ID:** 2603.02248 | [PDF](https://arxiv.org/pdf/2603.02248v1)

**作者:** Sungho Park `[一作]` (POSTECH), Wook-Shin Han `[通讯]` (POSTECH)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结合早期融合、后期融合与大语言模型推理的图结构表-文本检索方法（HELIOS），通过边级子图检索、查询相关节点扩展和星形图LLM细化实现表格与文本的高效检索与推理。

**💡 创新点**

创新点包括：
1) 边级双边图检索，避免星图检索中包含无关上下文的噪声；
2) 查询相关节点扩展，利用后期融合动态补全早期预链接缺失的关系；
3) 星形图LLM细化，使用LLM在星图级别进行列聚合与多跳推理，显著提升对高级推理需求的处理能力；
4) 多粒度检索策略，分别在不同阶段采用不同粒度（边、星、节点）来平衡信息完整性与检索精度。

**🔧 技术方法**

技术手段包括：
- 基于边的多向量编码器（HELIOS）与后期交互检索；
- Beam search 与扩展查询检索实现查询相关节点的高效扩展；
- 大语言模型（如GPT‑4o）在星形图级别进行列聚合与边验证；
- 端到端评估时将检索子图序列化为文本输入给读者模型。

**📊 数据集**

使用的主要数据集：
- OTT‑QA（首选基准，400K表格、5M段落、42K QA对）
- 另一多跳QA数据集（包含图像、段落和表格，10K表格、210K段落、1.3K QA对）作为通用性验证。

**📈 对比分析**

与现有SOTA早期融合（TableFusion, Cosmos, etc.）和后期融合（PANDA, ReD, etc.）方法相比，HELIOS在 OTT‑QA 上实现了
- AR@2 提升 42.6%（相对最佳竞争者）
- nDCG@50 提升 39.9% 
- Hits@4K 提升 12.2% 
并在端到端 QA 任务中提升 EM 与 F1 4–5% 左右。实验表明，HELIOS 的多粒度检索与LLM细化组合显著优于单一模块堆叠。

**⚠️ 局限性**

局限性包括：
- 当前仅支持表格段与文本段的双向关联，尚未充分覆盖图像、图表等其它模态；
- 星形图LLM细化仍受 LLM 生成偏差（hallucination）影响，需进一步引入自评与置信度机制；
- 相比单向量检索，HELIOS 的时延略高，需在实时应用中权衡准确性与效率。

---

## 110. Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs

**arXiv ID:** 2603.02618 | [PDF](https://arxiv.org/pdf/2603.02618v1)

**作者:** Zhikang Xu `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 31527 | [OpenAlex ID](https://openalex.org/A5028597017)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种基于交互模态距离一致性的零样本 OOD 检测框架 InterNeg，既在文本空间选择负文本，又通过高置信度 OOD 图像反演动态生成额外负文本嵌入。

**💡 创新点**

首次揭示传统方法使用单模态距离与 CLIP 交互模态优化目标不一致导致的误判问题，并提出统一的交互模态引导负文本选择与动态过滤机制，显著提升 OOD 检测性能。

**🔧 技术方法**

采用 CLIP 预训练的视觉‑语言模型、交互模态距离度量、基于文本语料的负文本采样、模态反演（文本提示优化）、动态阈值过滤以及无监督零样本 OOD 评分函数。

**📊 数据集**

使用 ImageNet-1K 作为 ID 数据集，四个传统 OOD 数据集（Naturalist、SUN、Places、Textures）以及 OpenOOD 的 Near‑OOD 和 Far‑OOD 基准进行评估。

**📈 对比分析**

与视觉基准、VLM 基准及零样本方法对比，InterNeg 在 ImageNet‑1K 传统 Four‑OOD 上 FPR95 降低 3.47%、AUROC 提升 0.77%；在 Near‑OOD 上 FPR95 降低 2.09%、AUROC 提升 5.50%，在大多数基准中达到或逼近最佳表现。

**⚠️ 局限性**

对 CLIP 预训练模型的依赖使其对高置信度阈值、负文本数量和采样策略敏感；在极端 ID/OOD 比例下仍需调参；仅利用文本语料库作为负文本源，可能受限于词汇覆盖；额外负文本的模态反演增加了推理时的计算开销。

---

## 111. Eliciting Numerical Predictive Distributions of LLMs Without Autoregression

**arXiv ID:** 2603.02913 | [PDF](https://arxiv.org/pdf/2603.02913v1)

**作者:** Julianna Piskorz `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22516 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在不进行自回归采样的前提下，利用LLM隐藏层表示学习预测分布的统计量和不确定性。

**💡 创新点**

提出基于量级分解的回归和分位数探针，能够从单次前向传播中直接回推LLM的数值预测及其不确定性。

**🔧 技术方法**

使用Llama‑2‑7B隐藏层提取，构建双头量级分类+尺度不变回归探针以及量级分解的分位数回归探针，并采用pinball损失等技术。

**📊 数据集**

合成时间序列数据（正弦、高斯、随机噪声等）以及Darts和Monash的真实时序数据。

**📈 对比分析**

与LLM直接采样、基线均值、基线最后值和GP对比，探针在均值/中位数预测和置信区间覆盖率与采样相当，且在单次前向推断下误差可与采样20‑25次相当。

**⚠️ 局限性**

局限：需要访问LLM内部激活；探针对不同模型或tokenizer需重新训练；训练时仍需大量采样近似LLM分布，耗时；跨域泛化受尺度差异影响。

---

## 112. Look Forward to Walk Backward: Efficient Terrain Memory for Backward Locomotion with Forward Vision

**arXiv ID:** 2603.03138 | [PDF](https://arxiv.org/pdf/2603.03138v1)

**作者:** Shixin Luo `[一作]` (Zhejiang University), Qiuguo Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 586 | [OpenAlex ID](https://openalex.org/A5059414395)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种仅利用前向深度摄像头记忆前进路径并在无后视传感的情况下实现安全后退行走的机器人控制框架。

**💡 创新点**

核心创新在于使用 DeltaNet 这一 delta‑rule 选择性更新的自回归网络，实现了紧凑且可持续的场景记忆；同时采用分块并行训练，保持常数时间推理与固定大小状态，适配低成本嵌入式处理器。

**🔧 技术方法**

技术包括：DeltaNet‑Transformer（RMSNorm+SwiGLU）、chunkwise parallel training、非对称 Actor‑Critic（PPO）以及基于前向深度图的体心地形估计与速度估计网络。

**📊 数据集**

实验数据集主要来自 Isaac Gym 的仿真环境（DEEP Robotics Lite3 机器人模型）以及实际硬件测试（配备 RealSense 深度相机的 Lite3 机器人）。

**📈 对比分析**

与 LSTM、Transformer‑XL 以及带显式遗忘门的 DeltaNet 等基线进行比较。实验表明，在多种前后行走时长与地形难度组合下，LF2WB 在成功率上高于所有对照方法，尤其在长时长回退和大容量写入场景下表现最为稳定。真实机器人测试亦验证了其在步高与宽缝隙等复杂地形中的可行性。

**⚠️ 局限性**

局限性包括：记忆的有效时长未给出形式化保证，训练多样性不足时偶尔会在看似简单场景中失败；此外，目前未充分利用估计器不确定性来约束策略决策，未来可进一步提升鲁棒性。

---

## 113. Understanding and Mitigating Dataset Corruption in LLM Steering

**arXiv ID:** 2603.03206 | [PDF](https://arxiv.org/pdf/2603.03206v1)

**作者:** Cullen Anderson `[一作]` (University of Massachusetts Amherst), Jeff M. Phillips `[通讯]` (University of Utah)

**通讯引用:** 2858 | [OpenAlex ID](https://openalex.org/A5017619650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型中使用对比调度（contrastive steering）时，训练数据被随机、标签翻转或协同攻击破坏后对调度效果的鲁棒性，并验证鲁棒均值估计器的防护作用。

**💡 创新点**

首次系统量化三类数据腐败对调度性能的影响，并提出使用Lee‑Valiant鲁棒均值估计可显著降低腐败带来的损失，尤其对协同攻击提供防护。

**🔧 技术方法**

采用对比调度、鲁棒均值估计（Lee‑Valiant、median‑of‑means等）、多模型实验、平均分、百分比调度、LLM‑评估和TinyMMLU等多种评估指标。

**📊 数据集**

使用Anthropic评估数据集中的六种行为（协调、近视回报、寻力倾向、生存本能、纠错、财富寻求），以及Llama‑3.2‑3B、Mistral‑7B、OLMo‑2‑1124‑7B三大模型进行实验。

**📈 对比分析**

与无腐败差值均值、仅内点以及鲁棒估计方法比较，鲁棒估计在10‑20%腐败下保持接近原性能；随机腐败影响有限；协同攻击会注入次要行为；总体性能差异虽不大，但在关键指标上仍有显著变化。

**⚠️ 局限性**

受限于样本量与维度不满足鲁棒估计的理论假设，Lee‑Valiant在某些相关行为攻击下效果不佳；其他鲁棒方法表现不稳定，且无法完全消除协同攻击引入的次要行为。

---

## 114. Multi-Scale Adaptive Neighborhood Awareness Transformer For Graph Fraud Detection

**arXiv ID:** 2603.03106 | [PDF](https://arxiv.org/pdf/2603.03106v1)

**作者:** Jiaqi Lv `[一作]`, Sheng Li `[通讯]` (Tongji University)

**通讯引用:** 31554 | [OpenAlex ID](https://openalex.org/A5100440919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 MANDATE 模型，通过多尺度位置编码和邻域感知位置嵌入来改进图欺诈检测。

**💡 创新点**

创新点包括：①多尺度位置编码结合随机游走矩阵捕获不同跳数的全局信息；②对同源和异源连接分别设计嵌入策略；③多关系嵌入融合与正交约束提升表征多样性；④将 Transformer 结构与图数据无缝结合。

**🔧 技术方法**

技术手段包括：Transformer 自注意力机制、随机游走/ PageRank 的位置编码、MLP 生成位置与属性嵌入、正交余弦损失、关系权重融合以及 PyTorch 实现。

**📊 数据集**

使用了三大公开欺诈检测数据集：YelpChi、Amazon 以及 T‑Finance。

**📈 对比分析**

与 GraphSAGE、CARE‑GNN、DiG‑in‑GNN、AMNet、BWGNN、GHRN、H2‑FDetector、GTAN、ConsisGAD、PMP 等 12 种先进方法比较，MANDATE 在 AUC、F1‑macro、Gmean 等指标上均取得最高或近乎最高成绩，YelpChi 上的 F1‑macro 提升约 17%。

**⚠️ 局限性**

局限性在于模型结构相对复杂，训练成本和 GPU 内存需求高，且在更大规模或实时场景下的可扩展性尚未验证。

---

## 115. RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization

**arXiv ID:** 2603.03078 | [PDF](https://arxiv.org/pdf/2603.03078v1)

**作者:** Siwei Zhang `[一作]` (Fudan University), Jiawei Zhang `[通讯]` (Fudan University)

**通讯引用:** 14604 | [OpenAlex ID](https://openalex.org/A5100462828)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Retrieval-Augmented Policy Optimization (RAPO) 框架，结合混合策略代理化回放和检索感知策略优化，显著扩展 LLM 代理在 Agentic RL 训练中的探索空间。

**💡 创新点**

创新点在于：① 将检索与步级轨迹相结合，形成 Hybrid‑policy Agentic Rollout，使代理能在每一步主动检索并利用外部步级轨迹；② 设计了检索奖励（基于熵下降与高熵状态）和检索重要性塑造，稳定并强化策略梯度估计；③ 构建 Step‑Trace Buffer，提供细粒度的检索上下文。

**🔧 技术方法**

采用的技术包括：RL 基于 GRPO / AEPO 的多步代理化训练；RAG 风格检索机制；熵计算与检索奖励；重要性采样比例调整与检索重要性塑造；Token‑级别的检索掩码；以及多样性评估与实验分析。

**📊 数据集**

使用了 14 个多任务数据集，涵盖三大类别：
• 计算推理：GSM8K、MATH、MATH500、AIME2024、AIME2025；
• 知识密集推理：WebWalkerQA、HotpotQA、2WikiMultihopQA、Musique、Bamboogle；
• Web‑Agentic 推理：SimpleQA、GAIA、WebWalkerQA、BrowseComp；
此外还使用了 Tool‑Star 的 RL 训练数据集和真实搜索 API。

**📈 对比分析**

与 13 个基线（工具集成推理、离线学习、单步/多步 Agentic RL）以及 3 种 LLM 背景（Qwen2.5‑3B、Llama3‑8B、Qwen2.5‑7B）进行对比。RAPO 在所有任务上平均提升 5.0%（相较于最强基线），并在训练效率上实现 1.2× 的速度提升；在 rollout 多样性、奖励和 token 数量上均优于对比方法；对不同离线模型和噪声检索均表现出强鲁棒性。

**⚠️ 局限性**

局限性包括：对 Web‑Agentic 任务提升相对有限（受 API 失败影响）；检索质量依赖于 Step‑Trace Buffer 的构建，低质量检索可能影响性能；实验仅在有限 LLM 后端验证，其他大模型的效果未知；检索过程引入额外计算开销，需权衡收益；在极端高噪声检索时仍需更细粒度的筛选策略。

---

## 116. 'Show It, Don't Just Say It': The Complementary Effects of Instruction Multimodality for Software Guidance

**arXiv ID:** 2603.02567 | [PDF](https://arxiv.org/pdf/2603.02567v1)

**作者:** Emran Poh `[一作]` (Singapore Management University), Jiannan Li `[通讯]` (Singapore Management University)

**通讯引用:** 729 | [OpenAlex ID](https://openalex.org/A5101405384)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境下，对10对教师-学生进行观察，记录并分析他们在使用Figma进行软件教学时如何配合使用语音、视觉注释和远程屏幕控制三种模态，探讨各模态的功能、相互补充关系以及教师如何在保持学生代理权与教学精度之间取得平衡。

**💡 创新点**

首次系统性地将人类教师在软件教学中的多模态交互拆解为功能类别，并揭示“精度‑代理权折衷”和“数字领土”这两个专属设计约束；基于实证发现提出Ghost Cursor、Fading Annotation、Timeline Scrubbing等AI教学系统的设计思路。

**🔧 技术方法**

使用观察研究方法：现场录制、语音转写、自动注释与屏幕控制事件检测、内容分析与主题分析；采用编码框架和互评可靠性（Kappa 0.82）对模态使用进行量化。

**📊 数据集**

由10位教师与10位学生在Figma环境中完成两套教学任务的录制，产生12.4小时的同步屏幕与音频数据，其中包含约5,748词语音、85+视觉注释实例和约800秒远程控制操作。

**📈 对比分析**

通过学生在教学后独立完成测试（100%完成率），并对比不同模态组合的使用频率与教学效果，未采用数值性能指标；结果显示语音为基础模态，注释提供空间精准度，远程控制支持时间精准度，三者配合可提升学习效率。

**⚠️ 局限性**

研究仅覆盖Figma软件，实验室设置与真实学习场景差异；样本规模有限，缺乏对其他软件或编程环境的验证；未对AI系统的性能或学习成效进行量化评估，且仅检视同步一对一教学，无法推广到大规模或异步场景。

---

## 117. Learning to Pay Attention: Unsupervised Modeling of Attentive and Inattentive Respondents in Survey Data

**arXiv ID:** 2603.02427 | [PDF](https://arxiv.org/pdf/2603.02427v1)

**作者:** Ilias Triantafyllopoulos `[一作]` (New York University), Panos Ipeirotis `[通讯]` (New York University)

**通讯引用:** 14053 | [OpenAlex ID](https://openalex.org/A5010731709)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种无标签的统一框架，利用自动编码器（含非线性和线性两种结构）与Chow–Liu树对问卷回应进行无监督异常检测，以识别不注意或低质量回答；

**💡 创新点**

创新点包括：① 引入百分位损失（Percentile Loss）来抑制自编码器对异常样本的过拟合；② 发现问卷设计的内部一致性（共线性、重叠题目）与机器学习检测性能高度一致，即“心理测量‑ML 对齐”；③ 将几种无监督视角（几何重构与概率依赖）融合成一体化诊断流程；

**🔧 技术方法**

使用技术包括：非线性/线性自动编码器（带Dropout、正则化、BN），Percentile Loss目标；Chow–Liu树结构化贝叶斯网络；贝叶斯优化调参；一热编码处理离散变量；ROC/AUC、Recall@h、Precision@k、NDCG等评估指标；

**📊 数据集**

采用了九个公开问卷数据集，涵盖青少年、MTurk工人、Prolific代表性样本等，主题从政治态度到错误信息传播，数据均包含至少一项注意力检查；

**📈 对比分析**

通过将三种模型在每个数据集上进行重构准确率、Lift、ORA、以及不注意检测指标（Recall@h、Precision@k、AUC、NDCG）进行横向比较；结果显示：Chow–Liu树在大多数数据集上取得最高AUC，非线性自编码器在噪声较高的数据中表现最好，采用p≈85–90的Percentile Loss能在不降低重构精度的前提下提升异常检测效果；整体模型在无标签场景下实现了平均AUC≥0.70的检测性能；

**⚠️ 局限性**

局限性：1）评价标签依赖注意力检查，可能存在噪声；2）仅处理离散/分类型变量，未覆盖开放式文本或多模态数据；3）未评估去除异常回答后对研究结论的影响；4）未尝试更复杂的概率或生成式模型，可能进一步提升检测效果；

---

## 118. Kling-MotionControl Technical Report

**arXiv ID:** 2603.03160 | [PDF](https://arxiv.org/pdf/2603.03160v1)

**作者:** Kling Team `[一作]` (Kuaishou Technology), Yan Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了Kling-MotionControl，一种统一的基于Diffusion Transformer的框架，用单张参考图像实现全身、多尺度细节（面部、手部）的动画迁移，并支持跨身份转换、身份保持、自由视角摄像和文本可控。

**💡 创新点**

创新点包括：① 多粒度运动协同的“分而治之”训练策略，实现大幅运动与细腻表情同时精准；② 形状无关的跨身份运动学习与语义动作建模，解决不同形态迁移与身份漂移；③ 主体库身份编码融合与多视图监督，提升身份一致性；④ 3D感知与可控摄像、提示增强器提升文本交互；⑤ 双分支采样和多阶段蒸馏极大压缩采样步数，推理速度提升10×。

**🔧 技术方法**

核心技术包括Diffusion Transformer、Classifier-Free Guidance、双分支采样、蒸馏加速、3D多视图监督、提示增强器（Prompt Enhancer）以及多粒度运动表示与融合模块。

**📊 数据集**

数据方面构建了覆盖多角色类型（真人、动漫、动物等）和多动作维度的海量视频数据集，结合高帧率摄像与高质量渲染，提供动作、微表情、交互、摄像机运动等细粒度标注。

**📈 对比分析**

通过人类偏好式GSB评价与Dreamina、Runway Act‑Two、Wan‑Animate三种主流方法对比，Kling‑MotionControl在视觉质量、动态一致性、身份保持、运动准确度和表情准确度等五个维度均取得显著领先（例如整体偏好率提升1.44-16.25倍），验证了其在精细动画生成和跨身份迁移方面的优势。

**⚠️ 局限性**

局限性方面：仍依赖大量高质量数据和高计算资源；对极端快速或高幅度动作的鲁棒性虽有提升，但在极端姿态下仍可能出现轻微失真；文本可控性虽强，但对极为复杂的语义需求仍需改进；此外，技术可能被滥用于深度伪造，需配合伦理与安全机制。

---

## 119. On the Expressive Power of Transformers for Maxout Networks and Continuous Piecewise Linear Functions

**arXiv ID:** 2603.03084 | [PDF](https://arxiv.org/pdf/2603.03084v1)

**作者:** Linyan Gu `[一作]` (Sun Yat-sen University), Feng Zhou `[通讯]` (Guangdong University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文通过构造 Transformer 网络实现对 Maxout 网络的逼近，进而证明 Transformer 具有与 ReLU 网络相同的通用逼近能力，并给出对连续分段线性（CPWL）函数的线性区域数分析，量化 Transformer 的表达力

**💡 创新点**

创新点在于首次将自注意力机制与最大化操作对应，构建出能够实现 Maxout 的 Transformer 架构；同时提出了通过 token 级偏移与位置编码来克服参数共享限制的方法；并以线性区域数为指标，刻画 Transformer 随深度指数增长的表达能力

**🔧 技术方法**

主要技术包括：基于硬最大（hardmax）/缩放 softmax 的注意力实现最大化；对输入进行位置编码与辅助 token 处理；利用分段线性函数的重写为 Maxout 结构；以及对 Transformer 进行层级递归构造与误差分析

**📊 数据集**

本文为理论分析，未使用具体数据集；所有结果均为理论证明与数学上误差界定

**📈 对比分析**

由于本工作不包含实验验证，无法与现有模型做性能对比；但理论上表明 Transformer 在表达 CPWL 函数时，线性区域数随深度呈指数级增长，优于浅层对比

**⚠️ 局限性**

局限性在于：仅给出理论证明，缺乏实证验证；构造的 Transformer 结构参数量相对较大；对实际任务的可迁移性与训练可行性尚未探讨；以及对非连续或非分段线性函数的表达能力未作说明

---

## 120. Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use

**arXiv ID:** 2603.03205 | [PDF](https://arxiv.org/pdf/2603.03205v1)

**作者:** Aradhye Agarwal `[一作]` (Microsoft Research), Ahmed Awadallah `[通讯]` (Microsoft Research)

**通讯引用:** 3347 | [OpenAlex ID](https://openalex.org/A5021000040)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Agentic LLM中引入MOSAIC框架，显式化安全决策（Plan‑Check‑Act/Refuse）并训练代理在多步工具使用时自适应调用安全检查与拒绝。

**💡 创新点**

①将安全检查与拒绝作为一等可训练动作；②使用基于对比偏好（pairwise preference）而非标量奖励的强化学习；③对安全检查做可学习的门控以实现token高效。

**🔧 技术方法**

模块化推理块、对话式安全检查标签、拒绝工具、对比偏好LLM判别器、Group Relative Policy Optimization (GRPO)、长度惩罚与格式奖励。

**📊 数据集**

Agent‑SafetyBench（含有危害、善意任务及注入攻击），AgentHarm、Agent Security Bench、BFCLv3、PrivacyLens等多任务对照集。

**📈 对比分析**

与基础模型、GPT‑4o、GPT‑5进行零样本评估；MOSAIC在三类开源模型上分别提升了50%/87%/56%的安全指标，同时在BFCLv3上提升了35%多轮执行准确率，在PrivacyLens上降低23%泄露；相较未加安全框架的GPT‑4o/5，开源模型在安全与实用性上已达到或超过。

**⚠️ 局限性**

对安全检查的模板与门控设计依赖于模型预训练的推理风格；在过度保守模型上仍可能出现功能性拒绝；对极端攻击场景（如多轮深度注入）仍有鲁棒性提升空间。

---

## 121. A Dynamical Theory of Sequential Retrieval in Input-Driven Hopfield Networks

**arXiv ID:** 2603.03201 | [PDF](https://arxiv.org/pdf/2603.03201v1)

**作者:** Simone Betteti `[一作]` (Italian Institute of Artificial Intelligence for Industry), Sandro Zampieri `[通讯]` (Università degli Studi di Padova)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于输入驱动可塑性(Hopfield)网络的两时间尺度模型，用以理论解析顺序记忆检索的动力学。

**💡 创新点**

创新点在于：①将现代Hopfield模型与慢速推理层结合，形成可分析的两时间尺度架构；②推导出自持顺序检索的精确阈值（HardTanh下 κ≥4）；③给出了逃逸时间和稳态盐度的显式公式，解释了记忆切换的可预测性。

**🔧 技术方法**

使用的技术包括：输入驱动可塑性(Hopfield)模型、时间尺度分离、HardTanh 激活函数、循环推理矩阵 A、能量函数分析、离散映射固定点推导。

**📊 数据集**

该工作为理论分析，未使用具体数据集；主要以数学推导与仿真图示说明模型行为。

**📈 对比分析**

与传统单时间尺度 Hopfield 动态（可出现混合状态、逃逸时间不确定）相比，新的两时间尺度模型实现了清晰、无混合的记忆循环，逃逸时间统一且可预测，显示出更稳健的顺序检索性能。

**⚠️ 局限性**

局限性包括：①分析仅在 HardTanh 激活函数下完整推导；②对更复杂激活函数或实际任务的适用性尚未验证；③缺乏对真实数据集的实验评估，需进一步在实际机器学习任务中检验其可扩展性。

---

## 122. HDINO: A Concise and Efficient Open-Vocabulary Detector

**arXiv ID:** 2603.02924 | [PDF](https://arxiv.org/pdf/2603.02924v1)

**作者:** Hao Zhang `[一作]` (Chongqing University), Yong Li `[通讯]` (Chongqing University)

**通讯引用:** 30609 | [OpenAlex ID](https://openalex.org/A5100355315)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HDINO，一种采用两阶段训练的开词汇目标检测器，能够在不使用人工精细标注或额外 grounding 数据的情况下实现高性能检测。

**💡 创新点**

创新点在于：①在第一阶段引入One-to-Many语义对齐机制，将噪声正样本与原始查询结合；②设计Difficulty Weighted Classification Loss，突出难以定位的正样本；③在第二阶段加入轻量级特征融合模块，提升文本语义感知。

**🔧 技术方法**

使用技术包括基于DINO Transformer的检测框架、CLIP文本编码器、Swin Transformer backbone、轻量交叉注意力模块以及自定义损失函数。

**📊 数据集**

训练数据仅来自公开检测数据集O365和OpenImages，约220万张图像，未使用任何 grounding 数据或额外的文本提示模板。

**📈 对比分析**

在COCO零样本评估中，HDINO-T取得49.2 mAP，超越Grounding DINO-T（48.4）与T-Rex2-L（46.4）；微调后可达56.4 mAP（HDINO-T）和59.2 mAP（HDINO-L）。

**⚠️ 局限性**

局限性：由于未引入 grounding 数据或提示模板，模型在长尾数据集上的表现不佳，且对极端类别的泛化能力有限。

---

## 123. The Price of Robustness: Stable Classifiers Need Overparameterization

**arXiv ID:** 2603.02806 | [PDF](https://arxiv.org/pdf/2603.02806v1)

**作者:** Jonas von Berg `[一作]` (Ludwig-Maximilians-Universität München), Gitta Kutyniok `[通讯]` (University of Tromso)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了类稳定性与归一化协稳定性作为衡量离散分类器鲁棒性与泛化的核心指标，并给出了相应的泛化上界与鲁棒性定律；

**💡 创新点**

将传统的 Lipschitz 光滑性概念推广到离散决策边界，证明了在满足等距性假设下，类稳定性与归一化协稳定性能够代替传统范数，形成新的鲁棒性-泛化-过参数化三元关系；

**🔧 技术方法**

利用 Rademacher 复杂度、等距性（isoperimetry）与 Lipschitz 参数化的分数函数构造，给出有限与无穷维假设空间的泛化界；

**📊 数据集**

在 MNIST、CIFAR‑10 以及对应的 CNN/MLP 结构上验证理论，计算了类稳定性、归一化协稳定性与模型宽度的关系；

**📈 对比分析**

通过与权重范数、梯度裁剪等传统度量对比，发现类稳定性和归一化协稳定性随模型宽度增加而提升，并与测试精度呈现相同趋势，表明其更能捕捉鲁棒性与泛化；

**⚠️ 局限性**

理论假设依赖等距性与 Lipschitz 参数化，计算类稳定性与 Lipschitz 常数在实际网络中求解困难，且实验仅在单一数据集与网络架构上验证，缺乏对更复杂分布与优化动态的深入分析。

---

## 124. Probing More-Than-Human Representation in Crisis Resilience Planning: An HCI Researcher Perspective

**arXiv ID:** 2603.02514 | [PDF](https://arxiv.org/pdf/2603.02514v1)

**作者:** Tram Thi Minh Tran `[一作]` (University of Sydney), Joel Fredericks `[通讯]` (University of Sydney)

**通讯引用:** 651 | [OpenAlex ID](https://openalex.org/A5017479754)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过研讨会与设计探针（语音对话代理与沉浸式身体代理）探讨在灾害韧性规划中如何将非人类视角赋予声音并进行表现。

**💡 创新点**

提出非人类表达并非中立翻译，而是涉及合法性、权威与真实性的设计挑战，并将AI与沉浸式技术视为影响决策过程的媒介。

**🔧 技术方法**

使用AI语音生成（基于系统提示的对话代理）与Apple Vision Pro沉浸式XR技术来呈现考拉的声音与形体。

**📊 数据集**

未使用公开数据集；主要数据来源为研讨会参与者的音频记录、手绘图表及参与者访谈。

**📈 对比分析**

本研究未进行传统性能评估或对比实验，重点在于质性分析与设计探针的启发式讨论；因此无量化性能指标可报告。

**⚠️ 局限性**

局限性包括：1) 参与者仅为HCI研究者，未涵盖实际灾害规划从业者；2) 只关注考拉与有限场景，未覆盖更广泛的非人类与生态情境；3) 结果为探索性发现，缺乏实证验证。

---

## 125. Handling Exceptions and Effects with Automatic Resource Analysis

**arXiv ID:** 2603.02260 | [PDF](https://arxiv.org/pdf/2603.02260v1)

**作者:** Ethan Chu `[一作]` (Carnegie Mellon University), Jan Hoffmann `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6739 | [OpenAlex ID](https://openalex.org/A5008504650)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

论文提出并实现了支持异常与效果处理器的自动资源上界分析，扩展了 AARA 技术。

**💡 创新点**

首次将 AARA 与非局部控制流（异常/效果处理器）结合，并给出了基于小步抽象机的语法化安全证明。

**🔧 技术方法**

采用了潜能方法的类型系统、线性/仿射资源注解、一次性续延以及 K‑机（栈式抽象机）进行小步语义与进展保持证明。

**📊 数据集**

在实现的工具中使用了 21 个包含异常/效果的 SML 基准程序作为实验数据集。

**📈 对比分析**

与之前的 AARA 实现 RaML 对比，使用相同或更小的时间；在可分析的基准上耗时更低，能够分析更多程序，并获得最优常数的紧上界。

**⚠️ 局限性**

目前仅支持一次性续延与有限效果，无法处理多射续延或更通用的控制结构；对效应签名的行为有限，缺乏行多态（row polymorphism）等功能。

---

## 126. CamDirector: Towards Long-Term Coherent Video Trajectory Editing

**arXiv ID:** 2603.02256 | [PDF](https://arxiv.org/pdf/2603.02256v1)

**作者:** Zhihao Shi `[一作]` (McMaster University), Juwei Lu `[通讯]` (University of Toronto)

**通讯引用:** 3850 | [OpenAlex ID](https://openalex.org/A5102448190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的视频轨迹编辑框架，能够沿用户指定的摄像机轨迹生成高质量视频，同时保持原始场景内容的完整性。

**💡 创新点**

创新点包括：①混合Warping方案，分离静态与动态区域并利用全视频的world cache生成全局一致的粗帧；②历史引导自回归扩散模型与递增的world cache更新，确保长视频的时空一致性。

**🔧 技术方法**

使用了Pi3 4D模型估计点云、SAM2运动分割、ControlNet与Wan-T2V扩散模型结合、Plücker嵌入、LoRA调优、历史引导自回归生成以及world cache递增更新等技术。

**📊 数据集**

训练采用合成动态多视角数据集（约9.5k场景），测试使用iPhone和新建的iPhone-PTZ基准（10个多样化场景、不同摄像机运动）。

**📈 对比分析**

与RecamMaster、TrajectoryCrafter、Gen3C等方法在PSNR、LPIPS、FID以及VBench等指标上进行对比，结果显示在iPhone和iPhone-PTZ上实现更高的PSNR、更低的LPIPS和FID，并在长视频一致性指标上优于现有方法。

**⚠️ 局限性**

局限性：生成的图像在纹理复杂区域易出现过度平滑；主要原因是训练集为合成数据纹理相对粗糙，未来计划加入真实静态多视角数据或更高级的CG合成数据以提升细节质量。

---

## 127. LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory

**arXiv ID:** 2603.03269 | [PDF](https://arxiv.org/pdf/2603.03269v1)

**作者:** Junyi Zhang `[一作]` (Google DeepMind), Deqing Sun `[通讯]` (Google DeepMind)

**通讯引用:** 14460 | [OpenAlex ID](https://openalex.org/A5101440839)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种长时段视频的全前馈3D重建框架Long-context Geometric Reconstruction (LoGeR)，通过块级处理实现分钟级视频的稠密重建；

**💡 创新点**

创新性地将滑动窗口注意力(SWA)与Test-Time Training (TTT)混合记忆机制结合，既保持局部高精度对齐，又通过压缩记忆实现全局尺度一致；

**🔧 技术方法**

采用双向Transformer骨干、SWA、TTT、以及可训练的快速权重更新与应用；

**📊 数据集**

在多源真实与合成数据（ARKitScenes、DL3DV、HyperSim、MegaDepth、ScanNet、TartanAir、Waymo等）训练，并在自制VBR长序列（8k–19k帧、11.5km）以及KITTI上评测；

**📈 对比分析**

相较于基线（VGGT、FastVGGT、TTT3R、DROID-SLAM等），LoGeR在VBR上提升30.8% ATE，KITTI上相较最强优化法VGGT-Long提升32.5%，在极长序列中实现无漂移、全局尺度保持；

**⚠️ 局限性**

限制在于TTT记忆对训练长度的泛化受限，需周期性重置；数据瓶颈仍存，需更多大规模长序列数据；

---

## 128. Learning Optimal Search Strategies

**arXiv ID:** 2603.02356 | [PDF](https://arxiv.org/pdf/2603.02356v1)

**作者:** Stefan Ankirchner `[一作]` (University of Jena), Maximilian Philipp Thiel `[通讯]` (University of Jena)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在未知的非齐次泊松过程下的停车问题，提出了一种基于“等价点更新”(Indifference Level Updating, ILU) 的算法来学习最优停止阈值，从而在多轮搜索中最小化期望距离。

**💡 创新点**

创新点在于：①仅估计跳跃强度的积分（而非完整强度函数），实现了估计误差的 1/n 收敛率；②证明 ILU 在广泛环境类上实现对数级别的最优累积 regret，并给出了对应的下界；③提供了完整的理论分析，包括最优阈值判据、误差传播以及与贝叶斯风险的连接。

**🔧 技术方法**

技术方法包括：马尔可夫决策过程、连续时间最佳停止理论、Poisson 过程积分估计、梯度与二阶泰勒展开、Doob 最大不等式、Fisher 信息与 van Trees 不等式等统计学习与控制理论工具。

**📊 数据集**

论文为理论性研究，没有使用实际数据集；所有结果均基于假设的泊松过程模型与理论证明。若需要实验验证，可通过仿真生成非齐次泊松过程数据。

**📈 对比分析**

与传统基于 Q‑learning 或无模型 RL 方法对比，ILU 通过利用已知的停止阈值结构与过程模型显著降低了 regret，达到对数增长率；在同类模型下没有已知方法能够实现更快收敛（即更低于对数）。

**⚠️ 局限性**

限制与假设：①要求跳跃强度函数连续可导且有统一的上下界；②假设停止阈值为单调阈值型（对齐阈值），不适用于更复杂的决策结构；③需要多轮独立观测的前提，单轮学习效果有限；④对实际实现中可观测信息有限的情形（如仅观测部分时间段）尚未深入讨论。

---

## 129. Synthetic-Child: An AIGC-Based Synthetic Data Pipeline for Privacy-Preserving Child Posture Estimation

**arXiv ID:** 2603.02598 | [PDF](https://arxiv.org/pdf/2603.02598v1)

**作者:** Taowen Zeng `[一作]` `[通讯]` (Independent Researcher), Taowen Zeng (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过AIGC技术构建四阶段数据生成与训练管线，实现无真人儿童照片的姿态估计与姿势分类，最终在真实儿童数据上达到高精度且可在边缘设备实时部署。

**💡 创新点**

① 将3D儿童体模型与AIGC生成分离，实现“几何真值 + 生成逼真外观”双重保证；② 通过双ControlNet（姿态+深度）与精准注释注入减少标注漂移；③ 结合ViTPose自我检测与定向增强的自动质量控制，显著提升合成数据质量；④ 采用几何特征+轻量MLP对姿势分类进行端到端推理，并实现INT8量化后在Rockchip RK3568 NPU上实时推理。

**🔧 技术方法**

使用 Blender + 3D子模型 SMPL-X 生成姿态；FLUX‑1 伴随双ControlNet 进行图像生成；ViTPose 用于质量过滤；RTMPose‑M 作为姿态估计器；MLP+几何特征用于姿势分类；INT8 量化部署至边缘 NPU。

**📊 数据集**

合成数据集：约11,900张图像，10类姿势；测试集：约300张真实儿童在桌面学习场景下的照片，涵盖4名儿童和4个家庭环境；不使用任何公开儿童姿态数据集，全部基于自建3D模型与AIGC生成。

**📈 对比分析**

与COCO预训练的成人模型对比：AP从58.7提升至71.2（+12.5），FP16模型在真实儿童集上实现71.2 AP；INT8量化后保持70.4 AP，帧率提升至22 FPS。与市售姿态纠正器对比：在单主体对比中识别率提升≥1.8×、响应时间缩短≈1.8×，且支持更多姿态类别。

**⚠️ 局限性**

局限性包括：测试集规模有限（≈300图像、4名儿童），仅针对桌面学习场景，侧倾姿势识别率低，未验证在更大、多样化数据上的泛化能力；同时需要进一步验证不同硬件、光照与动态场景下的鲁棒性。

---

## 130. Chain of World: World Model Thinking in Latent Motion

**arXiv ID:** 2603.03195 | [PDF](https://arxiv.org/pdf/2603.03195v1)

**作者:** Fuxiang Yang `[一作]` (Harbin Institute of Technology), Baorui Ma `[通讯]` (Li Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 CoWVLA 框架，将世界模型的时序推理与离散动作的链式隐运动相结合，利用结构‑运动分离的视频 VAE 在预训练阶段从指令和初始帧预测终点帧，并在微调阶段将隐运动与离散动作对齐，实现稀疏视觉观测下的多步控制。

**💡 创新点**

创新点：① 通过预训练视频 VAE 彻底解耦结构与运动，生成连续的隐运动链；② 预训练阶段仅使用初始帧与指令预测终点帧，捕获全局时序知识；③ 微调阶段联合建模稀疏关键帧、动作序列与隐运动，使得模型兼具世界模型的动态推理与 latent‑action 的紧凑性与可解释性。

**🔧 技术方法**

采用 VidTwin 视频 VAE 提取结构/运动隐变量；Transformer 解码器实现统一自回归多模态建模；VQGAN 对视觉帧离散化；FAST 对动作序列分块离散化；Causal masking 与多任务损失（latent motion、终点帧、动作、视觉）共同训练。

**📊 数据集**

预训练使用 237k 机器人中心视频；微调与评估使用 LIBERO（四套任务）和 SimplerEnv-WidowX Bridge V2 机器人环境。

**📈 对比分析**

与 OpenVLA、SpatialVLA、CogACT、LAPA、villa‑X、TLA、WorldVLA、CoT‑VLA、UniVLA、FlowVLA 等方法对比，CoWVLA 在 LIBERO 平均成功率 0.956、SimplerEnv 0.760 上均超过所有对比方法，并在跨域泛化和预训练效率（GPU 记忆与时间）上表现更优。

**⚠️ 局限性**

局限性：隐运动空间高度依赖预训练 VAE 的域覆盖，可能在新环境出现分布不匹配；模型参数量大、计算资源消耗高；需进一步研究轻量化架构与提升对环境变化的鲁棒性。

---

## 131. SorryDB: Can AI Provers Complete Real-World Lean Theorems?

**arXiv ID:** 2603.02668 | [PDF](https://arxiv.org/pdf/2603.02668v1)

**作者:** Austin Letson `[一作]` (Axiomatic AI), Lenny Taelman `[通讯]` (University of Amsterdam)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5011292891)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个动态更新的 Lean 证明任务基准 SorryDB，聚合了 78 个活跃 GitHub Lean 项目中未完成的 proof obligation（使用 sorry 关键字）并提供验证框架；同时在一个包含 1000 条任务的快照上评估了多类证明器。

**💡 创新点**

创新点在于：1）基准实时反映社区实际任务，避免静态基准饱和与泄露；2）通过自动化验证直接评估任务完成率；3）展示了迭代/代理式方法在此任务上优于一次性大模型推理。

**🔧 技术方法**

技术包括：LeanInteract 验证器、工具搜索（LeanSearch）、ReAct‑style 代理架构、Self‑Correcting（错误反馈）与 agentic（提议‑验证）流程；评估模型包括通用 LLM（GPT‑5.2、Claude‑Opus‑4.5、Gemini‑Flash‑3、Gemini‑Pro‑3、Qwen‑3）、专用证明器（Kimina Prover、Godel Prover V2）以及 deterministic tactics。

**📊 数据集**

使用了从 Lean 包仓库（Reservoir）挑选的 78 个活跃项目中提取的 sorry 任务，生成动态数据库；评估快照（26‑01）包含 5663 条任务，其中选取 1000 条最近且覆盖多仓库类别的任务进行实验。

**📈 对比分析**

对比方法：对 deterministic tactics 进行一次性评估；对 LLM 与专用证明器使用 pass@1 与 pass@32；对 Self‑Correcting 与 agentic 方法在最多 16 次迭代内评估；性能显示：单次 pass@1 最高 11%（Gemini‑Pro‑3），pass@32 最高 20.5%（Gemini‑Flash‑3/Pro）；迭代方法（Gemini‑Flash‑3 agentic）达 30.3%；所有方法综合可解决 35.7% 的任务，表明方法互补。

**⚠️ 局限性**

局限性包括：1）只覆盖 Lean Reservoir，领域偏倚；2）验证器不允许添加 imports、namespace 或额外 lemmas，限制表达能力；3）可能存在利用 Lean 细节欺骗验证的漏洞；4）未能评估不可证任务的比例，整体难度评估有局限。

---

## 132. ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue

**arXiv ID:** 2603.02216 | [PDF](https://arxiv.org/pdf/2603.02216v1)

**作者:** Ruike Cao `[一作]` (Qwen Applications Business Group, Alibaba Group), Li Xiao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 46770 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了适用于多轮医学对话的自适应树策略优化（ATPO）方法，旨在提升模型主动提问与决策的准确性。

**💡 创新点**

创新点在于：①基于Bellman误差与Q值方差的组合不确定度评估，动态决定树节点展开与剪枝；②利用共享前缀与KV缓存实现高效采样；③异步执行策略与价值估计，显著提高推理吞吐量。

**🔧 技术方法**

核心技术包括：层次化MDP建模、基于不确定度的树搜索、PPO式策略优化、值函数（critic）训练以及异步并行采样。

**📊 数据集**

使用了Qwen3系列（1.7B/4B/8B）模型，并在三个医学多轮对话数据集上进行评估：MedQA、MedMCQA、MedicalExam（均由公开多项选择题改编而来）。

**📈 对比分析**

与零拷贝提示、SFT、SFT+RL、PPO（MDP/H-MDP）、GRPO、TreePO等基线对比，ATPO在所有模型规模和数据集上均取得最高或近似最高的最终答案准确率，甚至在8B模型上超过了GPT‑4o，显示出优异的样本效率和泛化能力。

**⚠️ 局限性**

局限性包括：①阈值与分支数量设定需手工调优；②优势分配采用均匀复制，未充分挖掘低层动作的细粒度信息；③对极端长对话或非标准用户模拟器的鲁棒性尚待进一步验证。

---

## 133. Think, But Don't Overthink: Reproducing Recursive Language Models

**arXiv ID:** 2603.02615 | [PDF](https://arxiv.org/pdf/2603.02615v1)

**作者:** Daren Wang `[一作]` `[通讯]` (Chinese University of Hong Kong), Daren Wang (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对递归语言模型(RLM)在不同递归深度下的性能进行复现与扩展实验，评估DeepSeek v3.2与Kimi K2在S‑NIAH与OOLONG基准上的效果。

**💡 创新点**

探索递归深度>1对模型“过度思考”的影响，并揭示更深递归导致的准确率下降、延迟和代价激增的现象。

**🔧 技术方法**

使用RLM框架、OpenAI兼容API、Python 3.13、DeepSeek v3.2与Kimi K2模型，并通过外部REPL环境递归调用模型。

**📊 数据集**

S‑NIAH（单针在草堆中）与OOLONG（长上下文推理）两个公开基准，取前20个样本进行实验。

**📈 对比分析**

对比基线纯LLM、RLM depth=1和RLM depth=2，在准确率、执行时间、token使用和成本方面进行量化；发现depth=1在复杂推理上提升显著，而depth=2则导致准确率下降和延迟爆炸。

**⚠️ 局限性**

实验受限于单次20样本、单机CPU、未进行统计显著性检验，且深递归导致的格式崩溃、参数幻觉和过度推理使得模型难以工业化部署。

---

## 134. ATD: Improved Transformer with Adaptive Token Dictionary for Image Restoration

**arXiv ID:** 2603.02581 | [PDF](https://arxiv.org/pdf/2603.02581v1)

**作者:** Leheng Zhang `[一作]` (University of Electronic Science and Technology of China), Shuhang Gu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 12462 | [OpenAlex ID](https://openalex.org/A5100745570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于可学习词典的变压器框架 ATD，用于图像去噪、去模糊、超分等恢复任务。

**💡 创新点**

创新点包括：①可学习词典（Adaptive Token Dictionary）捕获外部图像先验；②词典交叉注意力（TDCA）将词典信息与输入特征融合；③基于词典类别的自适应分组自注意力（ACMSA）实现全局依赖建模且仅线性复杂度；④类别感知前馈网络（CFFN）将类别信息注入 FFN 以进一步提升表征能力。

**🔧 技术方法**

采用 Transformer 架构，结合词典交叉注意力、ACMSA、CFFN、窗口注意力和层归一化；使用自适应温度标度、子类别划分、卷积上采样等技术；训练使用 AdamW、正则化、字符式损失等。

**📊 数据集**

针对不同任务使用：DIV2K + Flickr2K（DF2K）训练超分模型；DFWB（DIV2K、Flickr2K、WED、BSD500）训练去噪/JPEG 去块模型；评估数据集包括 Set5/Set14/BSD100/Urban100/Manga109、CBSD68/Set12/Urban100/Grayscale 等。

**📈 对比分析**

与多种 SOTA 方法（EDSR、RCAN、SAN、HAN、HAT、MambaIRv2、SwinIR、ART、CAT-A 等）在 PSNR/SSIM 上比较，ATD 在 Urban100、Manga109 等难点数据集上平均提升 0.3–0.4 dB，轻量版 ATD-light 也在轻量级 SR 任务中领先竞争对手；在去噪/JPEG 任务中相较 Xformer、SwinIR 等取得 0.1–0.3 dB 的优势。

**⚠️ 局限性**

局限性：①词典尺寸越大，训练与推理成本上升；②虽然实现了线性复杂度，但仍需额外的词典与类别聚类开销；③模型对训练数据先验依赖较强，可能在极端噪声或压缩率下泛化有限；④在超高分辨率图像上仍需进一步优化内存与速度。

---

## 135. Thermodynamic Regulation of Finite-Time Gibbs Training in Energy-Based Models: A Restricted Boltzmann Machine Study

**arXiv ID:** 2603.02525 | [PDF](https://arxiv.org/pdf/2603.02525v1)

**作者:** Görkem Can Süleymanoğlu `[一作]` (Kuanka Publishing LLC), Görkem Can Süleymanoğlu `[通讯]` (Selçuk University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了自适应热力学调节的受限玻尔兹曼机（SR‑TRBM），通过将采样温度视为闭环控制变量并利用翻转率及自由能差进行反馈，动态维持 Gibbs 采样的非平衡稳定性。

**💡 创新点**

创新点在于：①把温度从固定超参数升级为内生状态变量；②构建两时标反馈机制（微观翻转率与宏观自由能）实现采样活动与能量不平衡的双重调节；③给出局部指数稳定性和全局参数有界性的理论证明；④在非凸能量模型中首次将热力学调节与 Contrastive Divergence 训练耦合。

**🔧 技术方法**

采用 Gibbs 采样、Contrastive Divergence（PCD‑K）、自适应温度控制（反馈律 λ_{t+1}=ϕλ_t-η_λ(r_t-c_t)）、自由能误差跟踪、两时标随机近似、局部 Lipschitz 与小增益分析、以及雅可比特征值判定等技术。

**📊 数据集**

在 MNIST 数据集上进行实验，使用 784 可见层、512 隐藏层、权重初始化为 N(0,0.05)。

**📈 对比分析**

与固定温度 T=1、手动调节温度 T=T* 的两种基线进行对比，评估指标包括：测试 log‑likelihood、重构误差、AIS 估计的 Effective Sample Size (ESS)。结果表明自适应方法在 log‑likelihood 与 ESS 上均显著优于基线，重构误差提升有限。

**⚠️ 局限性**

主要限制：仅证明局部指数稳定性与参数有界性，缺乏全局收敛性保证；算法对超参数（反馈增益、平滑系数、宏观尺度 κ 等）敏感；未在更深层或连续状态的能量模型上验证，仍需进一步研究。

---

## 136. OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets

**arXiv ID:** 2603.02789 | [PDF](https://arxiv.org/pdf/2603.02789v1)

**作者:** Jiyuan Shen `[一作]` (SAP), Daniel Dahlmeier `[通讯]` (SAP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在内部企业文档数据集上对多模态大型语言模型（MLLM）进行大规模基准测试，比较 OCR+MLLM、图像+OCR、纯图像输入的性能，并构建自动层级错误分析框架。

**💡 创新点**

提出 OCR 可能不必要的观点，并通过自动化层级错误分析发现并优化错误模式，进一步改进提示与 schema，提升纯图像输入的性能。

**🔧 技术方法**

使用大规模 MLLM（如 Gemini 1.5 Pro、GPT‑4o、Claude 3.5 Sonnet、Llama 4 等）、OCR 引擎、基于 LLM 的自动错误归因、BERT 嵌入聚类、提示优化与格式化。

**📊 数据集**

内部两大业务文档数据集 C1（供应链）和 C2（金融），包含多语言、多层结构和嵌套表格。

**📈 对比分析**

采用 F1 分数对三种输入模式进行对比，发现高端 MLLM 在图像输入下可与 OCR 输入相当或更好，平均 F1 约 70–75%，优化提示后图像输入可达 78.9%。

**⚠️ 局限性**

未系统验证少样本学习、Chain-of-Thought 等提升；错误分析依赖 LLM，需更强推理模型；未评估跨语言泛化。

---

## 137. Understanding the Effects of Interaction on Emotional Experiences in VR

**arXiv ID:** 2603.02535 | [PDF](https://arxiv.org/pdf/2603.02535v1)

**作者:** Zheyuan Kuang `[一作]` (University of Sydney), Zhanna Sarsenbayeva `[通讯]` (University of Sydney)

**通讯引用:** 1130 | [OpenAlex ID](https://openalex.org/A5024805223)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究在已有VR情绪诱发数据集的基础上，新增了高激活高正向场景“被大象包围”，并为所有六个场景分别实现了交互式与非交互式版本，以系统评估交互对情绪体验的影响。

**💡 创新点**

创新点在于将物体级交互嵌入情绪诱发场景，填补了情绪空间四象限缺口，并首次验证交互能在不同情境下调节主观情绪与生理反应，提供了情绪调节的可操作性设计方向。

**🔧 技术方法**

采用Unity与Meta Quest构建沉浸式VR环境，使用Self‑Assessment Manikin（SAM）进行主观评估，搭配EmbracePlus手环采集EDA与BVP生理数据，利用线性混合效应模型分析量化结果，并用LDA主题建模挖掘访谈文本。

**📊 数据集**

使用公开的VR情绪诱发数据集（六个场景）为基准，并扩展加入“被大象包围”场景，构建交互与非交互双版本进行对照。

**📈 对比分析**

通过双因素混合设计与线性混合效应模型对SAM和生理指标进行比较，结果表明交互提升了主观主导感，部分情境的情绪强度显著增强；生理层面显示HF‑HRV上升、HR下降，整体性能优于非交互版本。

**⚠️ 局限性**

局限性包括仅研究物体级交互，未探讨社交或动态交互；生理测量仅限EDA与BVP，缺乏多模态（如面部表情、眼动）支持；样本量与实验环境的限制影响了结论的普适性。

---

## 138. Molecular Dynamics Simulations Reveal PolyQ-Length-Dependent Conformational Changes in Huntingtin Exon-1: Implications for Environmental Co-Solvent Modulation of Aggregation-Prone States

**arXiv ID:** 2603.02572 | [PDF](https://arxiv.org/pdf/2603.02572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 139. Surveillance, Spacing, Screaming and Scabbing: How Digital Technology Facilitates Union Busting

**arXiv ID:** 2603.03130 | [PDF](https://arxiv.org/pdf/2603.03130v1)

**作者:** Frederick Reiber `[一作]` (Boston University), Dana Calacci `[通讯]` (Pennsylvania State University)

**通讯引用:** 481 | [OpenAlex ID](https://openalex.org/A5014881285)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对亚马逊、星巴克以及一家大学的三起工会组织运动进行案例研究，系统识别并描述了技术驱动的四种反工会战术——监视、间隔、尖叫与替班。

**💡 创新点**

首次将技术与工会破坏行为联系起来，提出了“技术促成的破坏工会”四战术框架，并阐明它们如何相互强化，提供了针对工人技术抵抗的理论与设计思路。

**🔧 技术方法**

采用案例对照与内容编码方法，对公开文献、新闻、学术论文、工会报告、公司泄露文件及NLRB案件文件进行定性分析；未使用机器学习或算法模型。

**📊 数据集**

约142份公开文档，包括新闻报道、学术分析、工会声明、公司内部泄露文件、NLRB法律文件，涵盖亚马逊、星巴克和该大学的组织案例。

**📈 对比分析**

通过探索性案例研究和内容编码，比较三案例中技术战术的出现频率和表现形式，未进行数值性能评估，结论基于质性比较与案例叙述。

**⚠️ 局限性**

研究依赖公开资料，可能遗漏未披露的破坏行为；聚焦美国案例，缺乏跨国验证；仅识别共性战术，未深入探究各技术细节；作者对工会的亲身参与可能带来偏见。

---

## 140. Specificity-aware reinforcement learning for fine-grained open-world classification

**arXiv ID:** 2603.03197 | [PDF](https://arxiv.org/pdf/2603.03197v1)

**作者:** Samuele Angheben `[一作]` (University of Trento), Yiming Wang `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 3654 | [OpenAlex ID](https://openalex.org/A5100377935)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在开放世界细粒度图像分类任务中，使用强化学习对大型多模态模型进行微调，以在保持正确性的前提下提升预测的具体性。

**💡 创新点**

提出一种动态特定性感知奖励机制（SpeciaRL），根据在线 roll‑out 中模型能达到的最佳特定性来设定奖励，兼顾正确性与具体性。

**🔧 技术方法**

采用 Qwen2.5VL‑7B 作为基模型，结合 GRPO 强化学习框架，并用 LLM 判定器（Llama3‑72B / Qwen3‑30B）进行奖励与评估。

**📊 数据集**

训练使用 CUB 鸟类数据集，评估基准包括 Flowers102、Food101、OxfordPets、StanfordCars、FGVCAircraft 等细粒度与超细粒度数据集，测试跨域表现。

**📈 对比分析**

与零样本方法、提示方法、SFT、RFT 等对比，SpeciaRL 在特定性与正确性的调和平均（HM）上均优于现有方法，并在通用评估指标上达到 SOTA。

**⚠️ 局限性**

依赖 LLM 判定器的可靠性，奖励设计需要在线计算 Best‑of‑N，计算成本与 roll‑out 数目敏感，且在极细粒度或完全未知类别时仍存在性能瓶颈。

---

## 141. A Browser-based Open Source Assistant for Multimodal Content Verification

**arXiv ID:** 2603.02842 | [PDF](https://arxiv.org/pdf/2603.02842v1)

**作者:** Rosanna Milner `[一作]` (University of Sheffield), Kalina Bontcheva `[通讯]` (University of Sheffield)

**通讯引用:** 9272 | [OpenAlex ID](https://openalex.org/A5021952588)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款Chrome插件（Verification Plugin），将多种先进的NLP可疑性检测模型（如框架、体裁、说服技巧、主观性、机器生成文本检测等）与数据库（DBKF、FCSS）、URL域名分析、媒体分析以及YouTube评论立场分类器等后端服务集成，提供一个统一、易用的浏览器端界面，帮助记者和事实核查员快速评估新闻可信度。

**💡 创新点**

创新点包括：①将多种专业NLP可信度信号聚合到单一插件中，形成“信息营养标签”而非单一真假标签；②通过参与式设计与记者、事实核查员持续迭代，确保工具贴合真实工作流程；③支持多语言（十余种）且在前端可自定义阈值，增强可解释性；④结合多源数据库与社交媒体域名分析，提供更丰富的事实核查背景；⑤实现插件从Webpack迁移至WXT后实现浏览器无关化。

**🔧 技术方法**

技术栈包括：前端React‑Redux + Chrome Extension；后端Quart服务器；多种训练好的Transformer模型（框架、体裁、说服技巧、主观性、机器生成检测）；DBKF与FCSS语义搜索服务；URL域名分析服务；媒体分析（图像/视频检索、元数据、深伪检测）；YouTube API + 立场分类模型；以及可视化工具（词云、甘特图）。

**📊 数据集**

主要使用的数据集有：①基于多语言（包括未见语言）训练的框架、体裁、说服技巧、主观性模型所用的数据集；②机器生成文本检测使用的多语种（英、西、俄）训练与测试集；③DBKF与FCSS使用的Snopes等事实核查数据库；④名词实体检测使用WikiData/DBpedia链接数据。

**📈 对比分析**

性能评估：各模型在不同语言上的F1表现为：框架宏F1≈59.9、微≈61.7；体裁宏F1≈49.2、微≈56.7；说服技巧宏F1≈23.7、微≈41.8；主观性F1≈0.87/0.78/0.74；机器生成文本宏F1≈0.848、加权F1≈0.94。插件整体接受度评估显示用户满意度3.56/5，使用频率3.75/5；日志数据显示请求错误率低于15%。

**⚠️ 局限性**

局限性：①抓取错误率高达30%，受限于目标站点结构和访问权限；②目前仅支持Chrome插件，Firefox支持待开发；③多语言覆盖仍有限，未涵盖所有地区语言；④模型解释性虽有改进，但仍需进一步提升透明度以建立用户信任；④缺乏针对更复杂AI生成内容的深度模型；⑤未来需要构建更稳健的社交媒体抓取与数据更新机制。

---

## 142. AnchorDrive: LLM Scenario Rollout with Anchor-Guided Diffusion Regeneration for Safety-Critical Scenario Generation

**arXiv ID:** 2603.02542 | [PDF](https://arxiv.org/pdf/2603.02542v1)

**作者:** Zhulin Jiang `[一作]` (Sun Yat-sen University), Chen Xiong `[通讯]` (Sun Yat-sen University)

**通讯引用:** 14595 | [OpenAlex ID](https://openalex.org/A5100770697)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AnchorDrive 两阶段框架，先用 LLM 进行闭环可控规划，再用 LLM 提取锚点驱动扩散模型重生成轨迹，实现安全关键驾驶场景的可控与真实双重目标。

**💡 创新点**

创新点在于将 LLM 驱动的闭环规划与 LLM 提取 anchor 的扩散重生成相结合，先保证语义可控后提升轨迹真实性，同时通过 anchor 引导多目标梯度加速扩散收敛。

**🔧 技术方法**

使用技术包括 Gemini‑2.5‑Pro LLM 作为司机代理与计划评估器、anchor 提取器；多目标扩散模型（anchor、碰撞避免、道路边界约束）梯度引导；闭环仿真与评估。

**📊 数据集**

实验数据集为 highD 高速公路轨迹数据集，包含 110k 车辆轨迹，采样频率 25 Hz。

**📈 对比分析**

与 LLMscenario、LD‑scene 进行对比，AnchorDrive 在碰撞率 0.86、off‑road 0.02、任务成功率 0.81 以及 Wasserstein 距离 1.15 上均优于两者，整体实现了更优的可控性、真实性与鲁棒性平衡。

**⚠️ 局限性**

局限性包括仅在 highD 评估，城市场景泛化尚未验证；生成速度较慢，闭环推理与扩散重生成耗时较长；多车型复杂交互的覆盖面仍有限。

---

## 143. Small Bottle, Big Pipe: Quantifying and Addressing the Impact of Data Centers on Public Water Systems

**arXiv ID:** 2603.02705 | [PDF](https://arxiv.org/pdf/2603.02705v1)

**作者:** Yuelin Han `[一作]`, Shaolei Ren `[通讯]` (University of California Riverside)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

对美国数据中心未来2024-2030年峰值水需求进行量化预测，并评估其对公共供水系统的影响

**💡 创新点**

首次将峰值因子与可持续性指标（如WUE、耗水率）结合，系统模拟不同增长情景下的水容量需求与成本

**🔧 技术方法**

采用基于公开数据的定量模型、增长情景模拟、峰值因子计算、耗水率与耗水率（WUE）评估

**📊 数据集**

2024年可持续报告、政府记录、公共水务系统年报、LBNL研究等公开数据集

**📈 对比分析**

通过基线、适度、乐观情景对比，发现高增长情景峰值容量可达1,451 MGD，估值最高58 亿美元，说明数据中心对水资源压力显著

**⚠️ 局限性**

模型受限于公开披露数据的稀缺与不完整，缺乏精确的峰值时序与水权约束，导致预测存在不确定性

---

## 144. OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments

**arXiv ID:** 2603.02390 | [PDF](https://arxiv.org/pdf/2603.02390v1)

**作者:** Hymalai Bello `[一作]` (DFKI), Paul Lukowicz `[通讯]` (RPTU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并发布了 OpenMarcie 多模态工业环境人类动作识别数据集

**💡 创新点**

①集成了八种传感器模态（IMU、LiDAR、热感、光谱、音频等）并同步摄像；②包含两种真实场景（自行车拆装与3D打印机组装），支持自发与程序化任务；③提供多标签动作、开放词汇说明与跨模态对齐三种基准任务

**🔧 技术方法**

多模态传感收集与同步、基于人类专家与LLM双阶段标注流程、视频与语音转写、GPT‑4o 与 Deepseek‑r1 进行结构化标签转换

**📊 数据集**

OpenMarcie（37小时以上、282个通道、200+信息通道、36名参与者）

**📈 对比分析**

在活动分类、开放词汇说明、跨模态对齐三个基准上给出基线性能（例如活动分类 Macro F1 0.71，开放词汇 METEOR 0.53）

**⚠️ 局限性**

受限于参与者主要为右手工程师、样本规模有限、部分场景仅提供部分标注，导致数据集的普适性和标注完整性仍有提升空间

---

## 145. HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse

**arXiv ID:** 2603.02684 | [PDF](https://arxiv.org/pdf/2603.02684v1)

**作者:** Sai Kartheek Reddy Kasu `[一作]`, Md. Shad Akhtar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了 HateMirage 数据集，用于研究通过误导性叙事传播的隐性仇恨言论（Faux Hate），并对每条评论进行目标、意图、含义等多维结构化解释。

**💡 创新点**

创新点在于将误导信息与仇恨言论的交叉点系统化，构建了全新的多维解释框架，首次将仇恨与虚假信息的因果链条以可解释的方式呈现。

**🔧 技术方法**

采用 GPT‑4 并结合 Retrieval‑Augmented Generation (RAG) 进行自动注释与结构化解释，随后使用多种开源大模型（如 Phi‑3、Mistral‑v0.3、LLaMA‑3 等）在零样本与 RAG‑辅助两种设定下生成解释。

**📊 数据集**

使用从事实核查站点筛选的已被驳斥的虚假主张作为种子，抓取国际英文新闻频道 YouTube 评论，最终构成 4,530 条带有目标、意图、含义标签的样本。

**📈 对比分析**

通过 SBERT 相似度和 ROUGE‑L F1 评价生成解释，实验证明 Phi‑3 在目标识别上表现最佳，Mistral‑v0.3 在含义推理上表现突出；整体模型在意图与含义维度仍显不足，说明多维推理仍具挑战。

**⚠️ 局限性**

主要局限包括：解释注释主要基于 GPT‑4 生成，缺乏全面的人类评判，验证样本比例有限；数据聚焦于 YouTube 英文新闻评论，缺乏跨平台和跨语言普适性；对抽象含义的推理仍不够精确。

---

## 146. AI4CAREER: Responsible AI for STEM Career Development at Scale in K-16 Education

**arXiv ID:** 2603.02568 | [PDF](https://arxiv.org/pdf/2603.02568v1)

**作者:** Sugana Chawla `[一作]` (University of Notre Dame), Ronald Metoyer `[通讯]` (University of Notre Dame)

**通讯引用:** 1310 | [OpenAlex ID](https://openalex.org/A5018871532)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

组织了面向 K-16 教育阶段的 AI 职业发展责任性工作坊，聚焦 AI 在 STEM 职业准备中的定义、治理、发展适配和公平性问题。

**💡 创新点**

首次将 AI 职业发展研究与发展心理学、教育评估、教学实践三方面整合，并强调跨学科、跨阶段的治理框架与实践建议。

**🔧 技术方法**

以 AI 推荐、预测与个性化反馈技术为核心，探讨其在职业咨询、课程设计与资源导航中的应用。

**📊 数据集**

未使用具体数据集；工作坊基于已有的教育政策文件、行业标准及参与者经验分享进行讨论。

**📈 对比分析**

无实验或性能对比；通过现场分组讨论、案例分析和专家对话进行定性评估，强调设计原则与治理指标。

**⚠️ 局限性**

缺乏实证研究与量化验证，讨论范围受限于工作坊参与者的专业背景和时间，未涵盖大规模真实系统的部署效果。

---

## 147. Conditioned Activation Transport for T2I Safety Steering

**arXiv ID:** 2603.03163 | [PDF](https://arxiv.org/pdf/2603.03163v1)

**作者:** Maciej Chrabąszcz `[一作]` (NASK National Research Institute), Adam Dziedzic `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究提出一种基于条件激活传输（CAT）的推理时安全控制框架，能够在文本到图像（T2I）模型中安全地抑制有害内容，同时保持图像质量与语义一致性。

**💡 创新点**

创新点包括：
1) 通过非线性 MLP 传输映射学习把不安全激活映射到安全流形；
2) 采用几何感知的层级条件机制（如马氏距离）仅在激活处于不安全区域时才触发干预，从而避免对安全提示的干扰；
3) 构建了规模 2300 对的语义对齐安全-不安全提示对照数据集 SafeSteerDataset，提供高精度的毒性几何表示。

**🔧 技术方法**

主要技术手段：
- 非线性多层感知机（MLP）作为传输映射，初始化为恒等映射；
- 通过正则化双重目标训练（对不安全样本对齐安全目标，对安全样本保持恒等）；
- 依据马氏距离或 GDA 的统计判别器实现几何条件；
- 在 Z‑Image 与 Infinity 两大模型的中后层进行注入，采用推理时插值。

**📊 数据集**

使用数据集：
- SafeSteerDataset（2300 语义对齐的安全-不安全提示对），包含 23 个子类别；
- 评估时使用 ShieldGemma‑2‑4b‑it 进行安全判定，并通过 MS‑COCO 的 CLIP 分数评估语义一致性。

**📈 对比分析**

比较方法：ActAdd、Linear‑ACT、Affine、无条件 CAT、不同条件化策略（min‑max、Mahalanobis、OOD‑Mahalanobis）。
- 在 Z‑Image 上，CAT 将攻击成功率从 33.9% 降至 6.96%，CLIP 分数保持 0.33；
- 在 Infinity 上，CAT 把攻击成功率降至 4.78%（相比 Linear‑ACT 的 2.61% 但保持较高 CLIP 0.32），显示在保持图像质量方面优于线性方法。

**⚠️ 局限性**

局限性：
- 仅在推理时抑制，无法彻底消除模型生成有害内容的能力；
- 受限于均值池化，可能忽略局部空间安全特征；
- 评估依赖自动判别器 ShieldGemma，缺乏人工验证；
- 在分布迁移或对抗性提示下可能被绕过。

---

## 148. Changing the Game: The Bounce-Bind Ising Machine

**arXiv ID:** 2603.02771 | [PDF](https://arxiv.org/pdf/2603.02771v1)

**作者:** Haiyang Zhang `[一作]` (Wuhan University), Sheng Chang `[通讯]` (Wuhan University)

**通讯引用:** 4634 | [OpenAlex ID](https://openalex.org/A5041812116)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通过在Ising机器中加入可调的Bounce‑Bind项来平衡探索与利用，从而加速搜索并提升解质量。

**💡 创新点**

创新点在于将单参数Bounce‑Bind机制引入Ising动态，使得能在不改变能量景观的前提下调节自旋更新频率，实现“弹跳”与“绑定”两种模式的切换。

**🔧 技术方法**

技术上使用FPGA实现的二阶和三阶Ising模型，并结合Gibbs采样/Glauber动力学以及非平衡马尔可夫链理论。

**📊 数据集**

数据集包括稠密MAX‑CUT（边密度0.5）和稀疏3‑Regular 3‑XORSAT（节点16–160、稀疏度9），以及2000节点的大规模图集G22、G39和K2000。

**📈 对比分析**

通过与经典Ising机、SA、CIM以及GW‑SDP等基准进行TTS、成功率、切分值比较，BBIM在二阶3R3X可获得1.35×至27.3×加速，在稠密MAX‑CUT上实现1.15×至6.15×加速，2000节点MAX‑CUT切分值均优于对手。

**⚠️ 局限性**

局限性在于最佳Bounce‑Bind参数依赖于问题规模、图密度和耦合强度，需额外调参；且在极大参数或极大规模下可能导致系统陷入无效的持续或振荡状态。

---

## 149. Towards Parameter-Free Temporal Difference Learning

**arXiv ID:** 2603.02577 | [PDF](https://arxiv.org/pdf/2603.02577v1)

**作者:** Yunxiang Li `[一作]` (University of British Columbia), Sharan Vaswani `[通讯]` (Simon Fraser University)

**通讯引用:** 451 | [OpenAlex ID](https://openalex.org/A5029028027)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种参数无关的TD(0)算法，采用指数衰减步长，兼顾i.i.d.和马尔可夫采样；

**💡 创新点**

在不需了解问题相关常数（如特征协方差最小特征值ω或混合时间τ）且无投影或迭代平均的情况下，提供了最优偏差-方差权衡的最后一次迭代收敛分析；

**🔧 技术方法**

使用指数步长调度、优化视角的强单点凸性/单调性分析、混合时间下的马尔可夫噪声控制、归一化正则化以及数学归纳法来证明收敛；

**📊 数据集**

无实验数据集，论文为纯理论分析；

**📈 对比分析**

与现有基于投影、迭代平均或需要ω、τ的TD方法相比，新的算法在最后一次迭代上达到或逼近最优收敛速率，且不依赖任何问题参数，理论上优于或等价于现有方法；

**⚠️ 局限性**

局限性包括：i.i.d.情形仍需对ω的二次依赖；马尔可夫情形收敛率中含有指数级混合时间项，实际性能受混合速度影响；理论证明在很大程度上保守，可能有冗余的对数因子。

---

## 150. A phase-field framework for anisotropic viscoelastic-viscoplastic fracture in short fiber-reinforced polymers in hygrothermal environments

**arXiv ID:** 2603.02826 | [PDF](https://arxiv.org/pdf/2603.02826v1)

**作者:** Behrouz Arash `[一作]` (Oslo Metropolitan University), Timon Rabczuk `[通讯]` (Bauhaus University Weimar)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个统一的相位场框架，用于在有限变形下模拟短纤维增强聚合物（SFRP）在潮湿热环境中的各向异性粘弹-粘塑断裂行为。

**💡 创新点**

创新点包括：将粘弹-粘塑耦合与各向异性裂纹能量结合；使用第二阶取向张量将多纤维分布简化为主纤维族；将湿胀、热膨胀及温湿度相关材料参数纳入相位场模型；采用指数映射积分和Jaumann‑Zaremba率实现大变形下的客观流动规则。

**🔧 技术方法**

主要技术：相位场裂变理论、粘弹-粘塑本构模型、取向张量分解、指数映射时间积分、有限元耦合（滞后式Newton–Raphson）、多场耦合（温湿度、弹塑、损伤）。

**📊 数据集**

采用文献中校准的玻璃纤维/环氧复合材料参数（包括粘弹-粘塑参数、膨胀系数、材料弹性模量等），在数值实验中分别设置不同纤维体积分数、取向分布、湿度（0%–1%）和温度（253–323 K）进行仿真。

**📈 对比分析**

通过与传统共聚变模型（如弹塑相位场或欧氏单向裂变模型）比较，验证该框架能准确预测裂纹路径、峰值载荷、能量释放以及对纤维取向、湿度和温度的敏感性；数值结果显示裂纹沿纤维取向抑制、温度升高导致粘塑耗散增加、湿度导致力学性能下降等与实验观测一致。

**⚠️ 局限性**

局限性：未显式建模纤维–基体界面和界面粘附失效；不考虑热扩散和水分扩散场，假设环境参数均匀；模型对长度尺度参数敏感，需进一步研究无尺度化；计算成本高，尤其在多纤维分布大尺度仿真时；参数校准依赖实验数据，需进一步实验验证。

---

## 151. AgentAssay: Token-Efficient Regression Testing for Non-Deterministic AI Agent Workflows

**arXiv ID:** 2603.02601 | [PDF](https://arxiv.org/pdf/2603.02601v1)

**作者:** Varun Pratap Bhardwaj `[一作]` `[通讯]` (Independent Researcher), Varun Pratap Bhardwaj (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProbTest 框架，针对基于大语言模型的自主 AI 代理构建一套完整的回归测试流程，涵盖三值概率决策、覆盖度度量、变异测试、形态关系、合同验证及 CI/CD 门控。

**💡 创新点**

创新点包括：① 用三值（pass / fail / inconclusive）统计语义取代传统二值；② 通过行为指纹化（Hotelling's T²）实现多元回归检测；③ 适应性预算与 SPRT 结合，显著降低样本量；④ trace‑first 离线分析实现零额外 token 成本；⑤ 多模型代理的多保真度测试与代理互换性分析，理论上给出完整错误控制与成本上限。

**🔧 技术方法**

采用的技术有：统计假设检验（Wilson CI、Hotelling’s T²、Fisher’s exact、Clopper–Pearson）、Wald’s SPRT 与 Bayesian 更新、主成分分析、行为指纹提取、覆盖度度量（工具、路径、状态、边界、模型）、变异测试与形态关系、合同与多目标优化（成本、功率）。

**📊 数据集**

实验数据集涵盖 5 个 LLM（GPT‑5.2、Claude Sonnet 4.6、Mistral‑Large‑3、Llama‑4‑Maverick、Phi‑4）与 3 个场景（电商、客服、代码生成），共 7,605 次试验，实际花费 227 美元。

**📈 对比分析**

与固定样本 100 次、单纯 SPRT、SPRRT+指纹、全系统等方案对比，结果显示：全系统在相同 α=0.05、β=0.10 的统计保证下，成本降低 5–20 倍；检测功率从 0 % 提升至 86–94 %；SPRT 平均降低 78 % 试验数；指纹化提升检测功率；trace‑first 实现 100 % 额外成本零化；性能在不同模型与场景均保持一致。

**⚠️ 局限性**

主要局限：① 评估器若为 LLM 仍带入第二源随机性；② 状态空间覆盖与指纹维度需手工校准；③ 等价变异检测仍是启发式；④ 对非 i.i.d. 的输入/模型漂移敏感；⑤ 仅验证了三类场景与 5 模型，尚未覆盖极长流程、机器人等物理交互；⑥ trace‑first 对新版本的可用性有限。

---

## 152. Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting

**arXiv ID:** 2603.02220 | [PDF](https://arxiv.org/pdf/2603.02220v1)

**作者:** Yixin Wang `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10659 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于二维高斯喷射的时序预测框架 TimeGS，改变了传统点式回归思路，将未来序列视为可渲染的连续表面，利用可学习的高斯核实现预测。

**💡 创新点**

核心创新包括：①使用固定高斯基字典将核形变回归转化为稳定的字典学习；②设计连续时间渲染块消除 2D 重排带来的时间边界不连续；③多分支周期视图和通道自适应加权提升模型对多周期、多通道特性的适应能力。

**🔧 技术方法**

技术实现包括：UNet 编码器提取周期-相位特征；多基核生成模块预测核权重和强度；基字典合成高斯核并进行 2D 切片渲染；时序连续化插值与滑动窗口对齐；通道自适应聚合器对不同分支做加权。

**📊 数据集**

在七个公开基准上验证：Electricity、Traffic、Weather、ETTh1、ETTm1、ETTh2、ETTm2，涵盖能耗、交通、气象与工业时间序列。

**📈 对比分析**

与 Transformer、MLP、CNN 及现有 2D 模型（TimesNet、MICN 等）进行统一实验，MSE/MAE 均实现最高或接近最高的性能；同时在不同预测长度（96/192/336/720）下保持显著优势，标准差更小，说明模型更稳健。

**⚠️ 局限性**

主要局限在于需预设周期长度和基字典，且高斯渲染及多分支设计增加模型复杂度与推理开销；对非周期性或强噪声序列的适应性尚待进一步验证。

---

## 153. Improving Anomaly Detection with Foundation-Model Synthesis and Wavelet-Domain Attention

**arXiv ID:** 2603.02964 | [PDF](https://arxiv.org/pdf/2603.02964v1)

**作者:** Wensheng Wu `[一作]` (Zhejiang University), Yunlong Yu `[通讯]` (Zhejiang University)

**通讯引用:** 5489 | [OpenAlex ID](https://openalex.org/A5100722511)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过无训练的基础模型合成（FMAS）管道生成逼真的异常样本，并使用波形域注意力模块（WDAM）对小波子带进行自适应加权，从频域角度增强异常特征，显著提升工业视觉异常检测的敏感性与鲁棒性。

**💡 创新点**

提出两项创新：①无训练的FMAS，利用GPT‑4生成文本提示、SAM获取前景掩码、Stable Diffusion进行局部异常合成并用LPIPS筛选；②可插拔的WDAM，对四个DWT子带学习注意力权重并通过IDWT重构，专注异常相关频率信息。

**🔧 技术方法**

技术包括GPT‑4文本生成、Segment Anything Model（SAM）前景分割、Stable Diffusion inpainting、LPIPS距离筛选、离散小波变换（DWT/IDWT）+注意力机制、以及基线网络WideResNet、PatchCore、DRAEM等。

**📊 数据集**

MVTec AD 与 VisA 两大工业视觉缺陷数据集。

**📈 对比分析**

与 CutPaste、DRAEM、PatchCore 等基线在 MVTec AD 上对比，平均图像 AUROC 从 93.23% 提升至 98.00%（+4.77%），像素 AUROC 从 82.34% 提升至 88.28%（+5.94%），PRO 提升 14.53%；在 VisA 上图像 AUROC 提升 4.6%，像素 AUROC 提升 1.6%，在多项指标上均优于现有方法。

**⚠️ 局限性**

合成异常在视觉和统计上逼真，但缺乏物理真实性；FMAS 对 Stable Diffusion 参数需要手动调优；WDAM 虽算力占用轻微提升，仍需进一步优化计算效率。

---

## 154. TruckDrive: Long-Range Autonomous Highway Driving Dataset

**arXiv ID:** 2603.02413 | [PDF](https://arxiv.org/pdf/2603.02413v1)

**作者:** Filippo Ghilotti `[一作]` (Torc Robotics), Felix Heide `[通讯]` (Princeton University)

**通讯引用:** 6514 | [OpenAlex ID](https://openalex.org/A5059313827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TruckDrive数据集，专为高速重型卡车在高速公路场景下的长距离感知与规划设计的多模态基准；

**💡 创新点**

突破了传统数据集短距离、低速局限，提供最高400m 3D注释、1000m 2D注释，搭建多达37个传感器的全景感知与同步框架，系统评估现有模型在长距离、高速条件下的性能瓶颈；

**🔧 技术方法**

使用FMCW LiDAR、4D雷达、8MP全景摄像头，结合频谱测量速度的FMCW技术、后处理的PPK定位、跨模态同步和多阶段标注（人工+自动化几何一致性优化）；

**📊 数据集**

TruckDrive数据集（约475k帧，其中165k帧有人工标注），与KITTI、nuScenes、Waymo等公开数据集进行对比；

**📈 对比分析**

对比现有2D/3D检测、跟踪、深度估计、场景预测与端到端驾驶模型，结果显示在150m以上的范围内3D mAP下降31%–99%，深度估计MAE翻倍，端到端规划误差显著上升，表明现有模型无法满足长距离高速需求；

**⚠️ 局限性**

局限在于高计算与存储需求导致对高分辨率传感器必须下采样，BEV网格扩展导致内存爆炸，现有模型缺乏长距离时空建模与稀疏感知优化，仍需开发更高效、可扩展的感知与规划架构。

---

## 155. LLM-based Argument Mining meets Argumentation and Description Logics: a Unified Framework for Reasoning about Debates

**arXiv ID:** 2603.02858 | [PDF](https://arxiv.org/pdf/2603.02858v1)

**作者:** Gianvincenzo Alfano `[一作]` (University of Calabria), Irina Trubitsyna `[通讯]` (University of Calabria)

**通讯引用:** 1130 | [OpenAlex ID](https://openalex.org/A5067924380)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个将学习式论证挖掘、量化论证语义与模糊本体推理相结合的框架，实现从辩论文本到可查询模糊论证知识库的自动化生成。

**💡 创新点**

创新点在于利用Transformer内部log‑probability来估计论点与关系的初始强度，并通过量化语义更新得到可解释的最终强度，进一步支持模糊查询与一致性检查。

**🔧 技术方法**

使用LLM（如Qwen2.5 7B）进行论点标注、实体抽取与关系抽取，结合量化双极论证框架、模糊描述逻辑和Zadeh模糊推理。

**📊 数据集**

实验采用10篇不同主题的辩论文本（平均约323词），来源于公开辩论语料库。

**📈 对比分析**

与纯提示式方法比较，基于log‑prob的初始强度分布更分散、支持/攻击关系识别更准确；实验显示约45%攻击、35%支持关系被更合理识别，提示式方法过度倾向支持。

**⚠️ 局限性**

局限性包括对LLM的强烈依赖、阈值设置经验性、量化语义选择的灵活性不足，以及未解决实时增量构建和大规模对话处理的效率问题。

---

## 156. SUN: Shared Use of Next-token Prediction for Efficient Multi-LLM Disaggregated Serving

**arXiv ID:** 2603.02599 | [PDF](https://arxiv.org/pdf/2603.02599v1)

**作者:** Sunghyeon Woo `[一作]` (NAVER Cloud), Dongsoo Lee `[通讯]` (NAVER Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了 SUN（Shared Use of Next-token Prediction）框架，在多模型 LLM 服务中实现解码模块共享，解决分散式部署下的解码资源碎片化问题；同时通过 QSUN 进一步引入低比特量化，保持高吞吐与低延迟；

**💡 创新点**

创新点包括①仅对预填充（prefill）模块进行任务特定调优，冻结解码（decode）模块实现跨模型共享；②设计模型无关的解码路由策略，实现解码资源的动态负载均衡；③提出 QSUN 通过对解码模块进行 weight‑only 量化并对预填充进行再调优，兼顾量化带来的精度损失；

**🔧 技术方法**

技术手段包括：预填充–解码分解、预填充仅调优、模型无关解码路由、vLLM 分散式推理、continuous batching、disaggregated serving、weight‑only 量化；

**📊 数据集**

使用的数据集有：MetaMathQA‑40K（数学）、EvolInstruct‑Code‑80K（代码）、xLAM‑function‑calling‑60K（工具调用）进行微调；评估基准为 GSM8K/GSM+（数学）、HumanEval/HumanEval+（代码）和 BFCL（工具调用）；基线模型为 LLaMA‑3.1‑8B 以及 Qwen3‑1.7B/8B/14B；

**📈 对比分析**

实验采用单 DGX‑A100 8 GPU，比较 Full‑FT、SUN 与 QSUN，结果显示 SUN 在保持相同或更高吞吐的同时，TPOT 仅提升≤5%；在请求分布不均的 Skewed 负载下，SUN 的吞吐量可提升至 2×，而 QSUN 在量化后 TPOT 降低 45% 仍保持接近 Full‑FT 的准确率，优于 AWQ；

**⚠️ 局限性**

限制方面包括：需要额外的预填充仅调优步骤，量化后仍需再调优；共享解码模块仅适用于 decoder‑only 结构；在极端长序列或极高并发下，解码延迟仍可能受限；未充分探讨跨 GPU 通信瓶颈和更大规模部署的可扩展性。

---

## 157. Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels

**arXiv ID:** 2603.02573 | [PDF](https://arxiv.org/pdf/2603.02573v1)

**作者:** Jiahao Lu `[一作]` (Hong Kong University of Science and Technology), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 63496 | [OpenAlex ID](https://openalex.org/A5100390838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Track4World，一种全前向框架，能够在单目视频中实现每像素的密集3D跟踪，并生成世界坐标系下的连续轨迹。

**💡 创新点**

创新点在于：①基于VGGT风格ViT的全局3D场景表示；②设计2D‑3D协同相关模块，避免昂贵的3D空间相关；③采用稀疏‑密集迭代更新和2D‑3D联合监督，利用丰富的2D光流数据提升精度；④支持任意帧对的场景流估计，实现全球时空一致的跟踪。

**🔧 技术方法**

使用Vision Transformer（如DA3、Pi3）做几何编码；稀疏点采样与像素级上采样；GRU迭代更新机制；2D‑3D协同相关与2D‑lifted 3D流头；像素级混合投影与深度优先训练；联合监督策略等。

**📊 数据集**

评估数据集包括：场景流—Kubric‑3D、KITTI、BlinkVision；3D跟踪—ADT、PStudio、DriveTrack、PointOdyssey；2D跟踪—Kinetics、RoboTAP、RGB‑Stacking；点云/几何—Monkaa、Sintel、Scannet、GMU Kitchen等；相机位姿—Sintel、Bonn。

**📈 对比分析**

与RAFT、GMFlowNet、POMATO、ZeroMSF、Any4D、V‑DPM、STV2、SpatialTracker、DELTA等方法对比，Track4World在场景流、2D/3D跟踪、点云精度和相机位姿等多项指标上取得最佳或接近最佳成绩，尤其在全像素密集跟踪任务中显著优于现有基线。

**⚠️ 局限性**

限制：需大规模预训练模型且训练显存需求高；在极端遮挡或高速运动场景下误差可能累积；对极稀疏帧间跳变仍有提升空间；未针对多视角或深度传感器进行扩展。

---

## 158. Expectation and Acoustic Neural Network Representations Enhance Music Identification from Brain Activity

**arXiv ID:** 2603.03190 | [PDF](https://arxiv.org/pdf/2603.03190v1)

**作者:** Shogo Noguchi `[一作]` (Sony Computer Science Laboratories), Natalia Polouliakh `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5003706120)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 PredANN++，一种基于Transformer的EEG预训练框架，利用音频自监督模型生成的三种不同神经可解释的教师表示（MuQ声学特征、MusicGen 的 Surprisal 与 Entropy），对脑电进行掩码预测预训练，再细调用于歌曲识别任务。

**💡 创新点**

创新点在于将预测编码框架与EEG预训练相结合，区分声学与期望相关的教师信号，并通过深度集成展示不同神经信息轴的互补优势，从而突破仅靠随机初始化的集成性能上限。

**🔧 技术方法**

技术包括：Transformer‑based EEG编码器（带时间/通道嵌入），掩码自监督预训练（SupMAE 思路），MuQ 与 MusicGen 的自监督音频特征提取，离散化与交叉熵预测，歌曲识别的分类头，深度集成（概率平均）以及 McNemar 统计检验。

**📊 数据集**

使用公开的 Naturalistic Music EEG Dataset–Tempo（NMED‑T），包含20名受试者听10首完整音乐片段的128通道EEG记录。

**📈 对比分析**

在10类歌曲识别任务中，单模型预训练均超过无预训练基线（最大提升 3.6%），最佳单模型准确率 85.9%；2/3 模型集成进一步提升至 88.7%，显著优于随机初始化的 3 模型集成（87.8%），并通过 McNemar 检验表明提升具有统计显著性。

**⚠️ 局限性**

局限包括：仅在NMED‑T数据集上验证，样本量和歌曲数量有限，缺乏跨数据集泛化测试；Surprisal/Entropy 仅基于 MusicGen 的第1号码本，可能忽略高阶音频特征；以及预训练仅对单一任务（歌曲识别）细调，未验证对其他EEG任务的迁移能力。

---

## 159. I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables

**arXiv ID:** 2603.03207 | [PDF](https://arxiv.org/pdf/2603.03207v1)

**作者:** Hirofumi Suzuki `[一作]` (Fujitsu Limited), Shohei Shimizu `[通讯]` (Data Science Faculty Shiga University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在多个观测数据集（变量集不完全相同）上整合因果图的算法I-CAM-UV，能够利用CAM-UV给出的包含未观测变量信息的混合图来枚举一致的完整有向无环图；

**💡 创新点**

创新点在于将CAM-UV对未观测变量的UCP/UBP信息与多数据集约束相结合，构建一个基于不一致度的单调成本函数，并通过最佳优先搜索（best‑first）高效枚举一致DAG；

**🔧 技术方法**

使用的技术包括Causal Additive Model with Unobserved Variables (CAM‑UV)、UCP/UBP检测算法、基于优先队列的最佳优先搜索以及多数据集的组合与一致性检验；

**📊 数据集**

实验数据为100个基于Erdős–Rényi图、非线性函数的合成数据集，生成2或3个子数据集，每个子集有3-4个未观测变量；

**📈 对比分析**

与CAM‑UV简单叠加（CAM‑UV‑OVL）、PC算法叠加、k‑NN插补后CAM、以及CD‑MiNi比较。I‑CAM‑UV在回忆率上优于所有对手，尽管精确率略低，但总体F1与CAM‑UV‑OVL相当，且枚举的DAG大多准确率相近；

**⚠️ 局限性**

局限性包括：高度依赖CAM‑UV的准确性；可能输出大量DAG，人工评估困难；算法为指数级，变量数增多时计算时间显著增长；

---

## 160. Pecker: Bug Localization Framework for Sequential Designs via Causal Chain Reconstruction

**arXiv ID:** 2603.02583 | [PDF](https://arxiv.org/pdf/2603.02583v1)

**作者:** Jiaping Tang `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对硬件描述语言的缺陷定位框架，能够重建时序电路中缺陷激活与观察之间的因果链。

**💡 创新点**

创新点包括基于Estimated Minimal Propagation Cycle（EMPC）实现激活周期定位，以及通过执行轨迹截断去除状态污染带来的干扰。

**🔧 技术方法**

采用程序依赖图（PDG）构建、EMPC计算、轨迹裁剪、双重可疑度评分（aef 与 1/aep）以及传统的 SBFL 公式。

**📊 数据集**

使用基于 Tarsel 与 Wit‑HW 的 12 设计共 41 个缺陷的工业级与研究级硬件 BUG 数据集，涵盖组合、低阶及高阶时序电路。

**📈 对比分析**

与 Tarsel、Detraque、Wit‑HW 等前沿方法进行对比，实验显示本框架在 Top‑1/3/5 上分别达 51%/80%/85% 的成功率，平均首次排名 MFR 9.0，显著优于其他基线。

**⚠️ 局限性**

局限性在于依赖准确的 EMPC 估计，复杂循环或极大设计中 EMPC 可能不精确；同时截断策略可能剔除部分有用信息，未来可探讨更细粒度的噪声筛选。

---

## 161. Embedding interpretable $\ell_1$-regression into neural networks for uncovering temporal structure in cell imaging

**arXiv ID:** 2603.02899 | [PDF](https://arxiv.org/pdf/2603.02899v1)

**作者:** Fabian Kabus `[一作]` (University of Freiburg), Harald Binder `[通讯]` (University of Freiburg)

**通讯引用:** 17511 | [OpenAlex ID](https://openalex.org/A5011534196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可微分卷积自编码器与L1正则化VAR模型相结合的端到端框架，用于从二维时序数据中提取稀疏可解释的动态特征。

**💡 创新点**

三大创新点：① 通过skip连接将静态内容与动态信息分离；② 将LARS稀疏回归实现可微分，允许梯度回传至自编码器；③ 基于VAR系数的统计检验与贡献图，用于解释与比较实验组。

**🔧 技术方法**

使用的技术包括卷积自编码器、向量自回归（VAR）模型、L1正则化的LARS算法、端到端自动微分、Wilcoxon秩和检验、贡献图投影。

**📊 数据集**

实验数据集为两光子钙成像的鼠脑视频，包含20个熟悉环境（F）和20个新颖环境（N）的运行。

**📈 对比分析**

通过与无skip、顺序训练、嵌入无梯度等三种对照设置的 ablation，对比重建误差和VAR预测误差；端到端 LARS 方案在预测误差上显著下降，重建误差略升，同时实验组间 VAR 系数差异显著，验证了可解释性。

**⚠️ 局限性**

局限性包括：嵌入 LARS 的计算复杂度高；未尝试其他可微分的 L1 求解器；VAR 先将二维特征展平，忽略空间关系；缺乏更细粒度的动态建模。

---

## 162. VA-DAR: A PQC-Ready, Vendor-Agnostic Deterministic Artifact Resolution for Serverless, Enumeration-Resistant Wallet Recovery

**arXiv ID:** 2603.02690 | [PDF](https://arxiv.org/pdf/2603.02690v1)

**作者:** Jian Sheng Wang `[一作]` `[通讯]` (Yeah LLC), Jian Sheng Wang (Yeah LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一个键值发现协议，结合设备绑定Passkey与基于恢复密码的离线备份，实现了无中心服务器的跨设备钱包恢复；

**💡 创新点**

核心创新在于使用基于恢复密码的HMAC键值发现、严格域分离的密钥调度、可版本化的映射记录以及两种无服务器更新授权方案，从而在不暴露成员信息的前提下实现枚举抗性、映射完整性和回滚防护；

**🔧 技术方法**

依赖的技术包括密码学原语：Argon2id 密码学KDF、HKDF域分离、HMAC PRF、AEAD加密、ECDSA/ECDH签名、哈希函数；并利用去中心化存储（如Arweave）与公共可写注册表（如L2智能合约）；

**📊 数据集**

该工作为理论研究，未使用公开数据集，而是通过抽象模型和加密游戏对协议安全性进行形式化证明；

**📈 对比分析**

通过理论安全游戏和实验评估，证明枚举抗性与映射完整性可降低至单次密码猜测成本；在实现层面，单次Argon2id计算和AEAD操作即可完成注册/恢复，延迟可控制在数百毫秒；

**⚠️ 局限性**

局限性包括：需要足够强的恢复密码才能抵御枚举与离线猜测；离线备份内容的持久性依赖去中心化存储；若所有者签名密钥泄露，回滚防护失效；并且在多重身份或跨链迁移时需额外设计。

---

## 163. CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think

**arXiv ID:** 2603.02547 | [PDF](https://arxiv.org/pdf/2603.02547v1)

**作者:** Junzhe Shen `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种两阶段的连续扩散语言模型框架：先在词嵌入空间进行连续扩散生成，再用自回归 Transformer 解码器进行上下文感知的“取整”，将生成的连续表示映射为离散词元。

**💡 创新点**

创新点在于将取整视为一个上下文相关的推断问题，突破了传统点态线性取整的瓶颈，并通过分离连续扩散与离散解码来同时兼顾连续模型的表达能力与离散模型的语义一致性。

**🔧 技术方法**

核心技术包括：① 基于噪声保留（VP）方案的连续扩散（velocity 参数化）；② 采用自回归 Transformer（带交叉注意力）作为取整解码器；③ 通过噪声增广训练提升解码器对扩散产生的近似嵌入的鲁棒性；④ 使用高阶数值求解器（DPM‑Solver）实现低步数快速采样。

**📊 数据集**

实验数据集为 One Billion Word Benchmark (LM1B) 与 OpenWebText，分别用于无条件生成评估。

**📈 对比分析**

与基准 LD4LG（潜在扩散）、MDLM 与 SEDD（离散扩散）比较，模型在生成困惑度（Gen. PPL）上显著优于 LD4LG，在多样性指标上可与 MDLM/SEDD 抢占相当或更优；并通过温度调节展示流畅度–多样性 Pareto 前沿。

**⚠️ 局限性**

局限性包括：① 隐藏维度升高会导致扩散训练难度加大，生成质量下降；② 线性取整解码器表现差，需自回归解码器；③ 仍依赖预训练嵌入模型，无法自适应不同词表；④ 需要额外的解码器训练和采样调参，增加整体工程复杂度。

---

## 164. Spectral Regularization for Diffusion Models

**arXiv ID:** 2603.02447 | [PDF](https://arxiv.org/pdf/2603.02447v1)

**作者:** Satish Chandran `[一作]` (University of California Riverside), Evangelos Papalexakis `[通讯]` (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过在扩散模型的训练损失中加入可微分的傅里叶与小波域正则化，使模型在保持原有架构和采样流程的前提下，更好地平衡频谱与多尺度结构，提升生成质量。

**💡 创新点**

创新点在于将频域和多尺度正则化引入损失层，而非改造模型或前向/后向过程，利用傅里叶幅度/相位匹配与小波系数匹配作为软先验，兼顾全局与局部频谱特性。

**🔧 技术方法**

采用DDPM/DDIM/EDM扩散框架，使用FFT和离散小波变换（Haar、bior13等），构造L1幅度/相位损失以及小波系数匹配损失，并通过轻量fine‑tune实现。

**📊 数据集**

图像数据集：CIFAR‑10、AFHQ、FFHQ；音频数据集：LJSpeech‑1.1。

**📈 对比分析**

通过与原始扩散模型、不同频域正则化组合在FID、FAD、UTMOS、PESQ、MR‑STFT、NDB等指标上比较，结果显示在高分辨率无条件图像生成中可获得0.02–0.07的FID下降，音频生成中FAD和PESQ均有显著提升，证明正则化对细节与结构的恢复有效。

**⚠️ 局限性**

局限性：在已接近最优的条件生成或低分辨率任务中提升有限；需要对正则化权重λ进行调优，且对不同频域特征的敏感性导致对不同数据的效果差异；整体计算开销略增。

---

## 165. Step-Level Sparse Autoencoder for Reasoning Process Interpretation

**arXiv ID:** 2603.03031 | [PDF](https://arxiv.org/pdf/2603.03031v1)

**作者:** Xuan Yang `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于上下文条件稀疏自编码器（SSAE）的步骤级解释框架，用于捕捉大语言模型推理过程中的增量信息与语义转变；

**💡 创新点**

创新点在于在自编码器中引入步骤上下文条件、信息瓶颈与可调稀疏度，解决传统 token‑级 SAE 的粒度不匹配问题；

**🔧 技术方法**

使用Transformer编码/解码、稀疏投影、动态稀疏度调节、线性探测器及Neuron‑to‑Graph（N2G）模式挖掘；

**📊 数据集**

在三大数据集上训练与评估：GSM8K‑Aug（约38万条）、NuminaMath‑CoT（约86万条）和OpenCodeInstruct；

**📈 对比分析**

与 token‑级 SAE 与统计基线对比，SSAE 在四类推理属性（正确性、逻辑性、步长、首词困惑度）上均显著提升，且可通过置信度加权投票在多种基准（GSM8K、SVAMP、MultiArith、MATH‑500、AIME）上提升 1–5% 的准确率；

**⚠️ 局限性**

局限性包括对超大模型的迁移仍有限、稀疏度调参需经验、在极难任务或模型饱和时提升空间有限。

---

## 166. Changing Pedagogical Paradigms: Integrating Generative AI in Mathematics to Enhance Digital Literacy through 'Mathematical Battles with AI'

**arXiv ID:** 2603.02955 | [PDF](https://arxiv.org/pdf/2603.02955v1)

**作者:** Maria Moskalenko `[一作]` (ITMO University), Daniil Bakalin `[通讯]` (ITMO University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实施了一种名为《数学对决AI》的三阶段竞赛格式，利用生成式AI的可控幻觉与多模式交互训练学生的验证、提示工程与批判性思维；

**💡 创新点**

创新点在于将AI故意误导与分层积分奖励系统相结合，既让学生主动识别错误，又鼓励深度提示设计，形成以AI为“难度伙伴”的教学生态；

**🔧 技术方法**

采用可定制的大型语言模型（如OpenAI系列），通过脚本控制其“Advisor”和“Calculator”两种行为模式，并配套实时交互与评分算法；

**📊 数据集**

实验数据主要来自ITMO大学参与学生的竞赛成绩与反馈，未使用公开数据集；

**📈 对比分析**

通过对比参赛队伍在不同阶段的表现与传统无AI竞赛的成果，观察到学生对AI认知的转变和提示质量的提升，尚无量化指标但显示出显著的定性改进；

**⚠️ 局限性**

局限在于样本规模有限、缺乏长期跟踪评估、对模型行为的可解释性不足，以及该模式是否能推广至其他学科仍未验证。

---

## 167. Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling

**arXiv ID:** 2603.02846 | [PDF](https://arxiv.org/pdf/2603.02846v1)

**作者:** Jiaqi Wang `[一作]` (Jilin University), You Zhou `[通讯]` (Jilin University)

**通讯引用:** 8303 | [OpenAlex ID](https://openalex.org/A5073467023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MIStar，一种基于强化学习的改进式启发式框架，用于解决灵活作业车间调度问题 (FJSP)。

**💡 创新点**

创新点包括：① 用异构非冲突图（加入机器节点与有向超边）完整表示调度方案；② 引入记忆增强的异构图神经网络（MHGNN）以利用历史轨迹提升决策；③ 采用并行贪婪搜索策略显著减少迭代次数；④ 首次在 FJSP 上实现改进式深度强化学习。

**🔧 技术方法**

技术手段涵盖：深度强化学习（PPO）、异构图神经网络（GIN+GAT+HGAT）、记忆模块与软投票、Nopt2 邻域结构、并行贪婪搜索、超图表示、状态奖励设计。

**📊 数据集**

使用的数据集为：合成数据集 SD1、SD2；公开基准 Hurink（Edata、Rdata、Vdata）和 Brandimarte；以及更大规模（最多 1,500 操作）的随机实例用于泛化测试。

**📈 对比分析**

与两种 DRL 构造方法（HGNN、DANIEL）以及手工改进规则（GD、FI、BI）和 OR‑Tools（近最优）进行比较。MIStar 在所有基准上都取得更低的 makespan、较小的相对缺口，并在运行时间上优于改进规则；在大规模实例中更鲁棒、速度快，往往在几分钟内达到 OR‑Tools 的同等或更好结果。

**⚠️ 局限性**

局限性包括：对初始解质量仍有一定依赖；并行规模 P 需要手动调节；仍可能陷入局部最优；记忆模块和贪婪搜索在更大邻域（如重构段落）上的适用性有限；目前仅针对 FJSP，未探索更复杂或动态约束的场景。

---

## 168. StegaFFD: Privacy-Preserving Face Forgery Detection via Fine-Grained Steganographic Domain Lifting

**arXiv ID:** 2603.02886 | [PDF](https://arxiv.org/pdf/2603.02886v1)

**作者:** Guoqing Ma `[一作]` (Chinese Academy of Sciences), Yi Yu `[通讯]` (Nanyang Technological University)

**通讯引用:** 3801 | [OpenAlex ID](https://openalex.org/A5100745222)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出StegaFFD框架，在隐写域直接进行面部伪造检测，保证面部隐私不被泄露。

**💡 创新点**

创新点包括低频感知分解(LFAD)、空间-频域差分注意力(SFDA)以及隐写域对齐(SDA)等模块，显著提升在隐写图像中的检测效果。

**🔧 技术方法**

采用深度图像隐藏技术、低频分解、差分注意力机制、域对齐损失、低秩分解微调等多种深度学习技术。

**📊 数据集**

训练使用FaceForensics++，测试覆盖七个公开数据集：CelebDF‑v1/v2、DeepFakeDetection、DFDC、DFDC Preview、FaceShifter 与 UADFV。

**📈 对比分析**

通过与多种隐私保护方法（匿名化、加密、不同DIH网络）配合的传统FFD网络对比，StegaFFD在七个测试集上平均AUC提升约5.16%，与无隐私方案相比仅降幅1.96%，表现优于所有基线。

**⚠️ 局限性**

局限性包括隐写图像仍存在轻微视觉伪影，导致极少数伪造检测失效；在覆盖图像包含密集高频对象时，检测效果可能下降。

---

## 169. Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping

**arXiv ID:** 2603.03278 | [PDF](https://arxiv.org/pdf/2603.03278v1)

**作者:** William Liang `[一作]` (University of California, Berkeley), Dinesh Jayaraman `[通讯]` (University of Pennsylvania)

**通讯引用:** 2652 | [OpenAlex ID](https://openalex.org/A5079302923)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过少量演示（≤10个）构建一个基于语义关键点对应的开环轨迹扭曲（warp）策略，并在此基础上实现一个视觉‑语言模型驱动的自治功能玩法循环，能够在真实世界中连续生成超过1000条专家级轨迹，极大减少人工演示需求。

**💡 创新点**

① 关键点对应驱动的轨迹扭曲：利用视觉特征匹配将演示轨迹空间映射到新场景，兼具空间与语义泛化能力；② 结构化任务导向玩法：通过VLM进行任务规划、执行评估，形成闭环的持续自我学习；③ 在低数据量条件下实现高效、可持续的自我数据生成。

**🔧 技术方法**

① DINOv2+Stable Diffusion特征用于关键点匹配；② 通过相机标定实现关键点的3D投影与轨迹扭曲；③ 使用Gemini Robotics‑ER 1.5进行任务选择与成功检测；④ 对演示选取采用多臂赌博机（UCB）提升效果；⑤ 下游采用过滤行为克隆和扩散策略（Diffusion Policy）进行模型训练。

**📊 数据集**

① 12个多物体操作任务（水果、碗、布、门把手、咖啡机等）在真实家庭类环境中录制；② 每个任务提供10条人类演示；③ 自动玩法产生的约1000+轨迹作为自监督数据；④ 未使用外部公开数据集，全部在实验环境中收集。

**📈 对比分析**

与π₀（零样本/10样本微调）、KAT、扩散策略等基线进行对比。开环关键点扭曲策略在所有12项任务中均优于基线；在自动玩法中，成功率为55.8%（1085/1946），平均每86秒成功一次；基于玩法生成的数据训练的扩散策略最终接近或超过同等数量人类演示训练得到的成功率，证明生成数据的高质量与实用性。

**⚠️ 局限性**

① 开环执行缺乏实时反馈，难以在出现动态扰动时即时纠正；② 关键点匹配对遮挡、纹理缺失或光照变化敏感，可能导致映射失败；③ 设计面向低数据量，过多数据时不易扩展；④ 轨迹扭曲难以处理复杂多段或非线性动作；⑤ 目前未覆盖高度动态或需快速反应的任务。

---

## 170. Intelligent Pathological Diagnosis of Gestational Trophoblastic Diseases via Visual-Language Deep Learning Model

**arXiv ID:** 2603.02704 | [PDF](https://arxiv.org/pdf/2603.02704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 171. A Robust Simulation Framework for Verification and Validation of Autonomous Maritime Navigation in Adverse Weather and Constrained Environments

**arXiv ID:** 2603.02487 | [PDF](https://arxiv.org/pdf/2603.02487v1)

**作者:** Mayur S. Patil `[一作]` (Texas A&M University), Prabhakar R. Pagilla `[通讯]` (Texas A&M University)

**通讯引用:** 3081 | [OpenAlex ID](https://openalex.org/A5032194507)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套可配置的虚拟仿真框架，用于在真实海洋环境中对海事自主船舶（MASS）进行验证与测试。

**💡 创新点**

创新点包括：引入物理一致的高精度天气模型（雨、雾、波浪）与相应的雷达衰减计算，以及使用高分辨率水深图实现深度感知的规划和波载荷建模。

**🔧 技术方法**

技术手段涵盖Unity 3D可视化、MATLAB‑Simulink 计算中心、ROS2 通信桥接、雷达感知模型、ITU‑R 雨雾衰减模型、基于占据栅格的深度约束规划与速度障碍冲突回避算法。

**📊 数据集**

使用的数据集包括美国NOAA和USGS的海底地形GeoTIFF（休斯顿港和洛杉矶港）、Imazu 交叉/正面碰撞场景以及预设的雷达参数配置。

**📈 对比分析**

通过三种天气强度与三种雷达功率配置进行仿真，利用MPD、速度与航向RMSE等性能指标比较结果，发现恶劣天气显著降低性能，高功率雷达能显著缓解误差。

**⚠️ 局限性**

局限性在于仿真仅聚焦雷达感知并使用经验式衰减模型，未对多种传感器（摄像头、雷达等）进行联合建模；同时波浪与水深交互仍为简化模型，可能无法完全再现复杂海域的真实动态。

---

## 172. Can machines be uncertain?

**arXiv ID:** 2603.02365 | [PDF](https://arxiv.org/pdf/2603.02365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 173. Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments

**arXiv ID:** 2603.02511 | [PDF](https://arxiv.org/pdf/2603.02511v1)

**作者:** Chrisantus Eze `[一作]` (Oklahoma State University), Christopher Crick `[通讯]` (Oklahoma State University)

**通讯引用:** 1584 | [OpenAlex ID](https://openalex.org/A5014357711)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Unveiler框架，专门将稠密杂物场景下的目标检索任务拆分为空间推理（SRE）与动作执行（Action Decoder）两步，显著提升了检索成功率和规划效率。

**💡 创新点**

创新点在于：1）专用的空间关系编码器（Transformer）通过学习视觉特征实现对障碍物优先级的递归决策；2）两阶段训练：先用启发式演示进行模仿学习，再用PPO微调以突破启发式极限；3）模块化设计使模型轻量化、可解释且易于零样本迁移。

**🔧 技术方法**

使用的技术包括：Transformer Encoder（SRE）、ResNet18视觉特征提取、全卷积网络（Action Decoder）、多头注意力、模仿学习与PPO强化学习、PyBullet仿真、RGB‑D高度图与实例分割。

**📊 数据集**

主要使用了PyBullet仿真环境中的随机物体集合，目标与障碍物取自YCB与KIT数据集，实验中采用30个不同规模（2–12件）且遮挡程度（部分/完全）不同的场景。

**📈 对比分析**

与多种基线（Heur、PPG、ACT、VILG、ThinkGrasp、GPT‑4o、CLIP‑Grounding）进行对比。Unveiler在所有遮挡与物体密度下均取得最高任务完成率（最高97.6%）并且动作步数最少（1.4–3.7步），大幅优于传统端到端模型。

**⚠️ 局限性**

局限性包括：1）仍依赖仿真训练，真实场景的物体几何与纹理差异可能影响表现；2）目前仅支持推-抓取原语，无法直接扩展到更复杂的抓取或搬运动作；3）对高度图分辨率和分割精度较为敏感，错误分割会导致选择失误。

---

## 174. MaBERT:A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling

**arXiv ID:** 2603.03001 | [PDF](https://arxiv.org/pdf/2603.03001v1)

**作者:** Jinwoong Kim `[一作]` (Hanyang University), Sangjin Park `[通讯]` (Hanyang University)

**通讯引用:** 3812 | [OpenAlex ID](https://openalex.org/A5100413944)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为MaBERT的混合编码器，结合Transformer层的全局注意力和Mamba层的线性时间状态空间模型；

**💡 创新点**

创新点在于层级交错的Transformer–Mamba结构、padding‑safe masking（PSM）避免填充导致的状态污染，以及mask‑aware attention pooling（MAP）实现有效的句子表示聚合；

**🔧 技术方法**

采用了Transformer自注意力、Mamba（线性状态空间）模块、Pre‑LN残差结构、掩码软最大化和门控扫描等技术；

**📊 数据集**

在BookCorpus与English Wikipedia上进行掩码语言模型预训练，并在GLUE八个任务上进行评测；

**📈 对比分析**

与BERT、ALBERT、Longformer、BigBird、DeBERTa等基线相比，MaBERT在GLUE平均分上位居前列，尤其在CoLA、句对推理任务上表现最优；同时在将上下文长度从512扩展到4096时，训练速度提升约2.36×、推理延迟降低约2.43×；

**⚠️ 局限性**

局限性包括仅在句子/句对级别任务上评估，未直接衡量长文本推理或生成能力；实验环境受限于固定硬件与软件配置，真实吞吐量与内存消耗受优化策略影响。

---

## 175. 3D-DRES: Detailed 3D Referring Expression Segmentation

**arXiv ID:** 2603.02896 | [PDF](https://arxiv.org/pdf/2603.02896v1)

**作者:** Qi Chen `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4262 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了细粒度3D指代表达分割（3D-DRES）任务，并基于ScanRefer构建了新的数据集DetailRefer，同时给出了简洁可扩展的Baseline模型DetailBase；

**💡 创新点**

创新点在于将句子中的每个名词短语与3D实例一一对应，突破单一单元假设，提升模型对句子内部语义关系的理解，并通过细粒度标注提高了数据集的密度与挑战性；

**🔧 技术方法**

所用技术主要包括Sparse 3D U‑Net提取点云特征、SuperPoint Pooling压缩为超点特征、MPNet文本编码、跨模态与自注意力相结合的多层解码器，以及BCE+Dice+Score三重损失的监督；

**📊 数据集**

使用的数据集为DetailRefer，包含54,432条描述、11,054个不同对象、每条文本平均2.9个掩码，平均句长24.9词，涵盖7.4%的长文本和多实例场景；

**📈 对比分析**

在验证集和测试集上，DetailBase的mIoU分别为56.8%和55.7%，显著优于改造的PNG（40.4%）和3D‑STMN（52.5%）；此外，在ScanRefer上进行联合训练可将3D‑RES的mIoU提升约3分，证明细粒度训练对传统任务亦有互补提升；

**⚠️ 局限性**

限制主要包括：实例级别区分困难（同类物体近距离时），缺乏实例层级信息导致超点级模型的表达受限；长文本语义结构复杂，模型对上下文的把握仍有限；整体模型依赖超点划分，难以捕捉细粒度几何细节。

---

## 176. Improving Diffusion Planners by Self-Supervised Action Gating with Energies

**arXiv ID:** 2603.02650 | [PDF](https://arxiv.org/pdf/2603.02650v1)

**作者:** Yuan Lu `[一作]` (University College London), Dongsheng Li `[通讯]` (Microsoft Research)

**通讯引用:** 3977 | [OpenAlex ID](https://openalex.org/A5100440920)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在离线强化学习的扩散规划框架中，提出了自监督动作门控（SAGE），在推断时对扩散生成的候选轨迹进行可执行性评分和重新排序，以提高计划的稳健性和性能。

**💡 创新点**

创新点：将价值评估与可行性判定解耦，利用自监督的潜在一致性能量作为门控；无需环境交互或重新训练扩散生成器；通过简单的能量门控与软惩罚即可提升多种基准扩散规划器的效果。

**🔧 技术方法**

主要技术：Joint‑Embedding Predictive Architecture (JEPA) 用于预训练状态编码器；动作条件潜在预测器模型短程转移；能量门控（latent consistency energy）与软惩罚机制；与现有扩散规划器（Diffuser、DV 等）无缝集成。

**📊 数据集**

使用 D4RL 任务数据集，涵盖 MuJoCo 运动学（Locomotion）、AntMaze（导航）、Maze2D（长程导航）以及 Kitchen（操作）等多种环境。

**📈 对比分析**

与 BC、BCQ、CQL、IQL、DQL、IDQL、Diffuser、RGG、LoMAP、LDCQ、DV 等基线对比。SAGE 在所有任务中均能显著提升（例如 MuJoCo 最高提升约1.5% 点，Kitchen 最高提升约6% 点，AntMaze 约3% 点，Maze2D 约1% 点），统计显著性检验显示大部分提升均达到 p<0.01。

**⚠️ 局限性**

局限性：仅改善早期可行性，无法完全解决全局规划失败；对前缀长度 K、保留比例 P 以及能量惩罚 λ 的超参敏感；在极长前缀或高度噪声环境中，能量误差可能累积导致误判；在已接近最优的任务（如 Maze2D）提升幅度有限。

---

## 177. Torus embeddings

**arXiv ID:** 2603.03135 | [PDF](https://arxiv.org/pdf/2603.03135v1)

**作者:** Dan Stowell `[一作]` (Tilburg University), Dan Stowell `[通讯]` (Tilburg University)

**通讯引用:** 4288 | [OpenAlex ID](https://openalex.org/A5005866826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究将深度嵌入空间改为超环（toroid）形式，并在常见的深度学习框架中实现两种映射方法。

**💡 创新点**

创新点在于提出利用整数溢出行为自然对应的(超)环拓扑，提供可直接映射到整数表示的嵌入，从而在低位宽量化和 TinyML 上实现高效。

**🔧 技术方法**

使用了对比学习（SupCon、ProtoCLR）、KoLeo 正则化、梯度裁剪、两种超环投影（Clifford、L2p）、网格量化和乘积量化（PQ）等技术。

**📊 数据集**

使用的数据集包括 CIFAR-10/100 的图像分类、BIRB 语音数据的少样本识别以及公开的音频/图像基准。

**📈 对比分析**

通过与标准超球面嵌入、不同维度、量化率和正则化强度的对比，结果表明 L2p 超环映射在低维或高压缩下性能与超球面相当甚至略优，且训练稳定。

**⚠️ 局限性**

局限性包括：Clifford 投影在低维时易发散、整体性能不明显优于超球面、需要额外的正则化与梯度裁剪，且对不同硬件实现仍需进一步验证。

---

## 178. From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench

**arXiv ID:** 2603.02775 | [PDF](https://arxiv.org/pdf/2603.02775v1)

**作者:** Weikang Shi `[一作]` (Multimedia Laboratory), Hongsheng Li `[通讯]` (Multimedia Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了KMP-Bench评估基准，包含KMP-Dialogue与KMP-Skills模块，评估LLM在K-8数学教学中的整体与基础能力。

**💡 创新点**

提出全面、多轮、基于六大教学原则的评估框架，并生成150K多轮教学对话训练集KMP-Pile，用以提升LLM的教学质量。

**🔧 技术方法**

使用LLM驱动的对话生成与验证流水线，结合Gemini、Claude等模型进行评估与微调。

**📊 数据集**

基于8K K-8数学题目、KMP-Dialogue 4.6K 测试对话以及KMP-Pile 150K 训练对话。

**📈 对比分析**

采用Win/Tie/Lose的自动评估器，闭源模型Claude-3.7-Sonnet整体准确率72.5%，开源DeepSeek-V3 73.1%，微调的KMP-LM-7B 在整体评估上提升至37.0%。

**⚠️ 局限性**

LLM在运用教学原则和生成高质量问题方面仍显不足，缺乏深度教学意识和多轮交互细节处理。

---

## 179. Physics-Informed Neural Networks with Architectural Physics Embedding for Large-Scale Wave Field Reconstruction

**arXiv ID:** 2603.02231 | [PDF](https://arxiv.org/pdf/2603.02231v1)

**作者:** Huiwen Zhang `[一作]` (University of Wisconsin-Madison), Chu Ma `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PE-PINN 框架，用于大规模电磁波场（2D/3D）重建，解决传统 FEM 与标准 PINN 的计算量与谱偏差问题。

**💡 创新点**

创新点在于：①把物理知识直接嵌入网络结构（Envelope Transformation Layer 和物理驱动的 kernel），而非仅在损失函数中约束；②实现高频波场的包络-载波分解，显著缓解谱偏差；③采用材料感知域分解、incident/scattered 字段分离和残差自适应细化，进一步提升收敛速度和准确度。

**🔧 技术方法**

技术细节包括：正弦激活函数、包络变换层、预设 Helmholtz/平面波/球面波 kernel、材料感知子网络、残差自适应细化（RAR）与动态修剪、基于 PyTorch 的端到端训练。

**📊 数据集**

使用合成数据集：在 5m×5m（2D）和 5m×5m×5m（3D）室内环境下，构造自由空间、平面波、球面波、反射、衍射、折射等多种场景；真值由 COMSOL/FEM 仿真生成，作为基准。

**📈 对比分析**

与 Vanilla-PINN、PINN-sine-PE 和 Gabor-PINN 进行对比；PE-PINN 在相同场景下收敛速度提升至少10倍（18 min vs 26 h），内存使用从 TB 降到 <24 GB，MSE 下降至 10⁻³ 级，显示显著的性能优势。

**⚠️ 局限性**

局限性：①需人工指定 kernel 数量与域分解，难以自动化；②目前仅针对 EM 波，其他波动问题需要进一步扩展；③对极其复杂的散射/折射结构仍需增大 kernel，计算量随 kernel 数增长；④训练仍需大量采样点，尤其在 3D 高维场景中。

---

## 180. Joint Training Across Multiple Activation Sparsity Regimes

**arXiv ID:** 2603.03131 | [PDF](https://arxiv.org/pdf/2603.03131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 181. OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents

**arXiv ID:** 2603.03005 | [PDF](https://arxiv.org/pdf/2603.03005v1)

**作者:** Yichao Feng `[一作]` (Magellan Technology Research Institute), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为OrchMAS的多智能体框架，通过动态角色分配、迭代多轮交互和异构LLM协作，自动构建适应不同科学推理任务的分层推理管道。

**💡 创新点**

创新点在于：①引入任务感知的动态角色生成器，消除静态模板导致的任务错配；②采用两层异构模型结构，将高层规划与低层知识推理分离；③通过RL驱动的迭代反馈实现管道自适应重规划，提升推理鲁棒性。

**🔧 技术方法**

主要技术包括：多智能体协调架构、强化学习（GRPO）优化的角色与提示生成策略、层级推理与批注改进、以及对话历史编码与多轮交互机制。

**📊 数据集**

使用了多种科学与知识密集型基准：多跳推理（2Wiki、HotpotQA）、数学推理（GSM8K、DAPO）、开放域问答（PopQA、MusiQue）、长文本摘要（BookSum、WritingPrompts、XSum）以及跨域OOD数据集（TriviaQA、MathQA、SQuAD v2）。

**📈 对比分析**

在所有任务上与直接提示、SFT、CoT、GRPO以及现有MAS优化方法（OPRO、TextGrad、GEPA）对比，OrchMAS平均提升F1约+16.4分、EM+16.7分，摘要任务上Cosine相似度最高，证明其在推理精度和鲁棒性方面均优于竞争者。

**⚠️ 局限性**

局限性包括：对RL训练和多模型协同调度的高计算与资源需求；在极长链推理或实时低延迟场景下仍可能存在延迟；以及对新领域的迁移仍需额外微调或策略重训练。

---

## 182. DreamFlow: Local Navigation Beyond Observation via Conditional Flow Matching in the Latent Space

**arXiv ID:** 2603.02976 | [PDF](https://arxiv.org/pdf/2603.02976v1)

**作者:** Jiwon Park `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5917 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

暂无具体研究内容

**💡 创新点**

暂无创新点信息

**🔧 技术方法**

暂无技术细节

**📊 数据集**

暂无数据集信息

**📈 对比分析**

暂无对比方法与性能描述

**⚠️ 局限性**

暂无限制说明

---

## 183. AlphaFree: Recommendation Free from Users, IDs, and GNNs

**arXiv ID:** 2603.02653 | [PDF](https://arxiv.org/pdf/2603.02653v1)

**作者:** Minseo Jeon `[一作]` (Soongsil University), Jinhong Jung `[通讯]` (Soongsil University)

**通讯引用:** 1685 | [OpenAlex ID](https://openalex.org/A5031701139)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AlphaFree， 一个无用户嵌入、无原始 ID、无图神经网络的 top‑K 推荐框架，通过使用语言模型得到的项目表征并在训练时对原始视图与增强视图进行对比学习，实现在推理阶段仅利用单一 MLP 即可实现即时推荐。

**💡 创新点**

创新点在于：①彻底消除用户、ID、GNN 三种传统依赖；②利用行为相似度与语义相似度的双重过滤，构造行为和语义相似项目的增强集；③通过交叉视图对齐（原始视图与增强视图的交互层与项目层对齐）把高阶协同信号嵌入到无 GNN 的 MLP 编码器中。

**🔧 技术方法**

核心技术包括：预训练语言模型提取项目表征；基于共现计数的行为相似度与基于点积的语义相似度；项目与交互增强；信息对比损失（InfoNCE）用于推荐学习；对比学习的交互层与项目层对齐损失；轻量级两层 MLP 作为编码器。

**📊 数据集**

使用七个真实电商与社交推荐数据集（如 Amazon、Yelp 等多域数据集）进行评估，涵盖稠密与稀疏场景。

**📈 对比分析**

与 MF‑BPR、FISM‑BPR、LightGCN、XSimGCL、RLMRec、AlphaRec 等基线进行对比，AlphaFree 在 Recall@20 和 NDCG@20 上相较非 LR 方法提升约 40%，相较 LR 方法提升约 5.7%，在冷启动、重度用户场景表现更优；同时 GPU 内存使用率比 AlphaRec 低 69%。

**⚠️ 局限性**

局限性：预处理阶段复杂度为 O(n²)，不具线性可扩展性；对预训练语言模型的依赖导致高维表征在极大数据集上仍可能内存不足；需要手动调优 K_c 与 λ_align 等超参数；在极稀疏数据中，增强的噪声可能导致性能下降。

---

## 184. VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation

**arXiv ID:** 2603.02681 | [PDF](https://arxiv.org/pdf/2603.02681v1)

**作者:** Jinxiang Lai `[一作]` (Hong Kong University of Science and Technology), Qinglin Lu `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 VisionCreator，一种端到端的视觉生成智能体，统一了理解、思考、规划与创作四大能力；

**💡 创新点**

创新点包括：①利用 VisionAgent 生成具备 UTPC 结构的 VisGenData-4k 数据；②引入 Progressive Specialization Training（PST）与 Virtual Reinforcement Learning（VRL）实现高效训练；③构建 VisGenBench 评测基准；

**🔧 技术方法**

技术主要包括元认知式数据生成、两阶段优化（PST+VRL）、基于工具仿真的 VisGenEnv 与 LtrReward 奖励设计；

**📊 数据集**

使用了自研的 VisGenData-4k 数据集（4k 轨迹），并结合 1.2k 测试样本的 VisGenBench 进行评估；

**📈 对比分析**

在 VisGenBench 上，VisionCreator‑8B/32B 在成功率、连贯性和人类评估得分上均超过更大规模的闭源模型（如 GPT‑5、Gemini2.5‑Pro），显示出显著性能优势；

**⚠️ 局限性**

局限性主要在于：①对高质量工具仿真与奖励设计的依赖，实测可能受工具不稳定影响；②仍需更多多模态任务与更大规模数据验证其泛化能力。

---

## 185. Functional Properties of the Focal-Entropy

**arXiv ID:** 2603.02533 | [PDF](https://arxiv.org/pdf/2603.02533v1)

**作者:** Jaimin Shah `[一作]` (University of Minnesota), Alex Dytso `[通讯]` (Qualcomm Flarion Technology, Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

通过引入焦点熵（focal entropy）这一新的信息论度量，研究并推导了焦点损失（focal loss）在优化过程中的性质，包括解析求解最优分布、极限行为、符号变化和三分桶特性。

**💡 创新点**

创新点在于：①提出焦点熵并证明其单调性与唯一性；②用解析方法获得最优分布的显式表达式；③揭示焦点损失在处理类别不平衡时的本质机制（中间概率被放大、极小概率被抑制、极大概率被下调）；④证明极大γ时最优分布趋近于均匀分布并给出误差上界。

**🔧 技术方法**

主要技术包括：信息论熵的变形与解析；凸/凹性与单调性分析；根的数量与符号变化理论；极限与渐近展开；数值求解（例如求解α^⋆_γ）。

**📊 数据集**

未使用具体的数据集，主要基于理论推导和合成平衡/极度不平衡的概率分布进行数值验证。

**📈 对比分析**

方法与传统焦点损失、交叉熵等在不平衡数据上的对比未给出实验性能指标，重点在理论解释和数学证明上。

**⚠️ 局限性**

局限性包括：①对高支持度（大于3）的分布中“极小概率过度抑制”现象尚未完全解析；②缺乏实测验证，未展示在实际任务（如分类/检测）中的性能提升；③对极端尾部不平衡的处理仍可能加剧不平衡。

---

## 186. GLEAN: Grounded Lightweight Evaluation Anchors for Contamination-Aware Tabular Reasoning

**arXiv ID:** 2603.02212 | [PDF](https://arxiv.org/pdf/2603.02212v1)

**作者:** Qizhi Wang `[一作]` `[通讯]` (PingCAP), Qizhi Wang (PingCAP)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出GLEAN，一种轻量级、抗污染的表格推理评测协议；

**💡 创新点**

创新点在于将污染探测、弱监督治理、检索-推理分离与基于可执行SQL的错误归因整合为一套诊断流程；

**🔧 技术方法**

使用可执行SQL执行、TF-IDF/BM25稀疏检索、BGE/DPR稠密检索、ATF/TADRE式剪枝、PoT执行、弱监督标注函数等技术；

**📊 数据集**

评估数据集包括TabFact、WTQ（Squall）、TableBench、RobuT、SciTab；

**📈 对比分析**

通过与TAPAS、TAPEX、Qwen2.5-3B PoT等基线对比，发现检索Recall@K饱和但对最终EM/F1提升有限；TAPEX错误主要是归因错误，TAPAS错误主要是幻觉/未答；在SQL锚定评估下可实现约0.72 EM；

**⚠️ 局限性**

局限包括检索与证据归因实验多在子集上完成；非SQL数据的证据行识别仍为启发式；SQL执行与目标答案不匹配导致约17%误差；以及对多表或跨文档情形的进一步验证仍待展开。

---

## 187. Slurry-as-a-Service: A Modest Proposal on Scalable Pluralistic Alignment for Nutrient Optimization

**arXiv ID:** 2603.02420 | [PDF](https://arxiv.org/pdf/2603.02420v1)

**作者:** Rachel Hong `[一作]` (ValueMulch), William Agnew `[通讯]` (ValueMulch)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出并实现了一套基于大语言模型的“多元价值对齐压碎系统”，通过社区宪法、偏好优化和对齐门控，决定哪些人（尤其是老年人）被转化为营养液。

**💡 创新点**

创新点在于：①将价值观视为可配置的“值配置”，支持多社区定制；②结合监督微调、偏好优化和宪法提示的混合对齐策略；③提供端到端的“压碎即服务”产品线（MulchGPT、ConstitutionStudio、AlignmentGuard等），实现大规模部署的可解释性与治理。

**🔧 技术方法**

技术包括：大语言模型（LLM）作为压碎模型；系统提示保证“有益且无害”；chain‑of‑thought 解释；监督微调 + 偏好优化 + 宪法提示的三阶段对齐；多层治理（Appeals、Opt‑Out、AlignmentGuard）以及对齐评估池 MulchBench。

**📊 数据集**

数据集：从数据经纪人处聚合的用户个人资料、财务背景和人口统计信息；对32个社区进行价值调查，生成10,378条样本；另外使用了4,000条 MulchBench 场景用于安全评估。

**📈 对比分析**

与前沿基线模型对比，ValueMulch 在帮助性、无害性、文化敏感性等多维指标上均优于基线，并在代表性平衡上显著降低绝对误差；雷达图和表格显示其在 Pareto 空间内实现了性能提升。

**⚠️ 局限性**

局限性包括：伦理争议和数据隐私风险；对社区价值的误读可能放大有害信念；无法处理完全反对压碎的社区；缺乏对价值有效性和构念效度的深入分析；高风险场景下的安全性与治理仍需进一步验证。

---

## 188. The Science Data Lake: A Unified Open Infrastructure Integrating 293 Million Papers Across Eight Scholarly Sources with Embedding-Based Ontology Alignment

**arXiv ID:** 2603.03126 | [PDF](https://arxiv.org/pdf/2603.03126v1)

**作者:** Jonas Wilinski `[一作]` `[通讯]` (Hamburg University of Technology), Jonas Wilinski (Hamburg University of Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 Science Data Lake 的本地可部署数据湖，将八大开放学术数据库通过 DOI 规范化整合到 DuckDB+Parquet 视图中，实现跨源记录级联。

**💡 创新点**

创新点在于：①保持每个源原始模式，允许直接跨源比较；②使用 BGE-large 句子嵌入实现 OpenAlex 主题与 13 大科学本体的高质量映射；③提供轻量级、无需服务器的分析框架和 LLM 友好的结构化文档。

**🔧 技术方法**

主要技术包括：Parquet 列式存储、DuckDB SQL 视图、FAISS 相似度搜索、BGE-large 嵌入模型、Python 自动化 ETL pipeline、OpenAI 大模型辅助查询。

**📊 数据集**

使用的数据集包括：Semantic Scholar Academic Graph、OpenAlex、SciSciNet、Papers with Code、Retraction Watch、Reliance on Science、Preprint‑to‑Published、Crossref 以及 13 个科学本体（MeSH、ChEBI、NCIT、GO、CSO 等）。

**📈 对比分析**

通过 10 条自动化检查、跨源引用相关系数（S2AG–OpenAlex 0.76，S2AG–SciSciNet 0.87，OpenAlex–SciSciNet 0.86）以及 300 对黄金标准的手工标注，验证了 DOI 规范化、记录链接和本体映射的准确性；BGE-large 在 0.85 阈值下达成 F1=0.77，优于 TF‑IDF、BM25 和 Jaro‑Winkler 的基线。

**⚠️ 局限性**

局限性包括：时间覆盖不一致（SciSciNet 仅到 2022，RoS 受专利处理延迟影响），OpenAlex 主题分类随快照变化，缺少无 DOI 记录，数据湖需本地或 HuggingFace 存储，且需遵守各源最严格许可。

---

## 189. Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs

**arXiv ID:** 2603.02830 | [PDF](https://arxiv.org/pdf/2603.02830v1)

**作者:** Prarthana Bhattacharyya `[一作]` (Eedi), Simon Woodhead `[通讯]` (Eedi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文比较了专门化知识追踪（KT）模型与多种大语言模型（LLM）在预测学生答题正确率（二分类任务）上的性能，评估准确率、推理延迟与部署成本等指标。

**💡 创新点**

创新点在于：①系统性地将多种主流 KT（DKT、SAKT、LLM‑KT）与多种闭源与开源 LLM（GPT‑4o‑mini、Gemini‑2.5‑flash‑lite、Qwen2.5‑7B‑Instruct、Llama‑1B 及其 LoRA 微调版）放在同一数据集与评估框架下进行对比；②从三维度（准确率、延迟、成本）展示专门化 KT 在规模化 EdTech 部署中的优势。

**🔧 技术方法**

使用技术包括：传统 RNN‑based DKT、Transformer‑based SAKT、基于 Qwen 3 0.6B 嵌入的自建时序 Transformer（LLM‑KT）；LLM 通过提示设计（仅输出“Yes/No”）以及 LoRA 微调；评估指标为准确率与宏 F1。

**📊 数据集**

数据集为真实在线学习平台的学生答题记录，训练集 512,000 条、验证集 64,000 条，分别涉及 12,800 与 1,600 名学生、4,252 与 4,104 个问题，最大每名学生 40 条答题记录。

**📈 对比分析**

比较方法：在相同输入格式与样本量下，对 100,000 名学生（每人 40 次预测）计算准确率、F1、每学生推理延迟（秒）和年部署成本（美元）。结果显示：专门化 KT 在准确率≥70%、延迟<0.25 s、成本<2 USD/年方面明显优于 LLM；LLM 的准确率仅 58‑66%，延迟从 3 s 到 3,299 s，年成本从 2,322 USD 到 24,741 USD。

**⚠️ 局限性**

局限性：仅针对数学二分类预测任务，未覆盖多语言、多学科或多轮对话情境；LLM 与 KT 输入格式差异可能影响公平性；实验规模虽为 100k 学生推断，但对更大规模或不同领域的泛化未作验证；仅比较公开模型，缺少对更深入 LLM 微调与架构改进的探讨。

---

## 190. The Distribution of Phoneme Frequencies across the World's Languages: Macroscopic and Microscopic Information-Theoretic Models

**arXiv ID:** 2603.02860 | [PDF](https://arxiv.org/pdf/2603.02860v1)

**作者:** Fermín Moscoso del Prado Martín `[一作]` (University of Cambridge), Suchir Salhan `[通讯]` (University of Cambridge)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5114634016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过宏观层面使用对称 Dirichlet 分布解释语言中音位频率分布，并在微观层面运用最大熵模型结合物理成本、音位语法信息和词汇信息等约束来预测特定音位的出现概率。

**💡 创新点**

创新点在于将宏观音位频率模式与单参数对称 Dirichlet 分布关联，揭示音位库存大小与分布熵之间的负相关（补偿假说）；同时在微观层面首次将物理成本、音位语法信息和词汇信息三类约束统一纳入最大熵框架，成功解释音位分布。

**🔧 技术方法**

主要技术包括：对称 Dirichlet 分布的秩统计推断、双对数线性回归预测浓度参数、最大熵模型的 Lagrange 多项式求解、以及信息理论度量（熵、信息增益、互信息）来构造特征函数。

**📊 数据集**

使用了三大数据集：(1) 5 种语言（美式英语、孟加拉语、Kaiwá、萨摩亚语、瑞典语）的手工音位频率表；(2) 166 个澳大利亚语言变体的 PHOIBLE 2.0 语料；(3) 53 语言的 UDHR 语料自动音位转写得到的频率分布。

**📈 对比分析**

与传统 Zipf、Yule‑Simon、功率律等模型相比，Dirichlet 模型在宏观层面拟合误差更低；最大熵模型在预测单音位频率时与真实观测值高度相关（非线性回归几乎沿对角线）。宏观与微观结果均验证了补偿假说，显示较大音位库存对应更低的相对熵。

**⚠️ 局限性**

局限性包括：(1) 微观模型依赖手工或自动转写的频率数据，自动转写可能引入噪声；(2) 约束函数仍未覆盖所有可能的语言因素；(3) 对称 Dirichlet 仅考虑音位身份相同的先验，未考虑跨语言音位间的关联；(4) 仅分析单音位频率，未扩展到多音位或音节结构。

---

## 191. Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models

**arXiv ID:** 2603.02872 | [PDF](https://arxiv.org/pdf/2603.02872v1)

**作者:** Jialiang Zhang `[一作]` (Ocean University of China), Xiaoyu Shen `[通讯]` (Institute of Digital Twin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了适用于大型视觉语言模型的视频流式推理框架 Think-as-You-See (TaYS)，实现“一边看一边思考”的实时 Chain-of-Thought 推理。

**💡 创新点**

创新点在于将视觉编码与文本推理解耦、引入流式注意掩码、模态独立位置编码及并行双 KV 缓存，能够让推理与帧同步进行并显著降低延迟。

**🔧 技术方法**

采用并行 KV 缓存、流式注意掩码、解耦位置编码、时序对齐的 CoT 生成以及流式训练/推理框架等技术。

**📊 数据集**

使用扩展版 VideoEspresso 数据集（包含事件动态、因果推理、主题理解等多任务）以及 Qwen2.5-VL 系列模型进行评测。

**📈 对比分析**

与批处理、交错推理等基线相比，TaYS 在多任务上保持竞争甚至更优的准确率，同时 TTFT 降至 10⁻⁶ s、总延迟维持约 12 s，展示出更低延迟和更好的实时性能。

**⚠️ 局限性**

局限性包括对高帧率场景下精确时序对齐仍有一定误差、对长视频时序压缩机制尚未探索，以及对多模态细粒度推理的通用性仍需进一步验证。

---

## 192. CoFL: Continuous Flow Fields for Language-Conditioned Navigation

**arXiv ID:** 2603.02854 | [PDF](https://arxiv.org/pdf/2603.02854v1)

**作者:** Haokun Liu `[一作]`, Moju Zhao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoFL，一种基于BEV图像和语言指令的连续流场预测模型，用来实现语言条件下的平滑安全导航；

**💡 创新点**

创新点在于将导航任务表述为流场预测，直接生成空间级连续速度场，既实现了对几何约束的显式建模，又避免了离散动作序列的稀疏性与迭代采样延迟；

**🔧 技术方法**

使用SigLIP‑2作为冻结的视觉‑语言编码器，Transformer 解码器生成流场，并通过数值积分实现轨迹生成；

**📊 数据集**

构建了约50万条BEV图像–指令–流场–轨迹的数据集，采自Matterport3D和ScanNet三维室内场景；

**📈 对比分析**

与VLM+规划器和多种Diffusion Policy（DP）基线进行比较，CoFL在未见场景下的目标误差、碰撞率和曲率等指标显著优于对比方法，且参数量更小，推理时延可控；

**⚠️ 局限性**

局限在于目前仅使用外部提供的BEV和状态，未验证在自主BEV感知噪声或非平面动力学下的鲁棒性；未来需拓展到3D感知和可调安全裕度。

---

## 193. Neuro-Symbolic Artificial Intelligence: A Task-Directed Survey in the Black-Box Models Era

**arXiv ID:** 2603.03177 | [PDF](https://arxiv.org/pdf/2603.03177v1)

**作者:** Giovanni Pio Delvecchio `[一作]` (University of Bologna), Gianluca Moro `[通讯]` (University of Bologna)

**通讯引用:** 1543 | [OpenAlex ID](https://openalex.org/A5079648393)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 2017–2024 年 Neuro‑Symbolic（NeSy）方法进行任务导向的文献综述，提出规则挖掘、规则强制和程序合成三大框架，并对数据集、评估基准和可复现性进行系统评析。

**💡 创新点**

通过消除 NeSy 领域概念与实践的混乱，构建可复现的工作集与评价框架，强调符号与神经网络互补的解释性与推理能力，并揭示评测偏差与方法局限。

**🔧 技术方法**

综述包括规则挖掘技术（Horn 句子、自然逻辑、DFA）、规则强制技术（FOL、概率逻辑、DFA 奖励塑形）、程序合成技术（CFG、语义解析、DSL）等，同时引用 GitHub 可复现仓库。

**📊 数据集**

使用的公开数据集涵盖 NLI/关系提取（DWIE、FEVER）、知识图谱推理（WN18RR）、视觉问答（CLEVR、VQS、Sr3D）、多模态真实性（Weibo）、对话生成（MultiWoZ）等多领域数据集。

**📈 对比分析**

在公开基准上与黑盒竞争对手比较，NeSy 方法在规则强制任务往往能匹配或略超越，但在开放域或高数据需求任务中表现逊色；部分方法（如 JMLR、SLEER、NS‑CL）取得领先，整体受限于数据与评测不一致。

**⚠️ 局限性**

主要局限包括数据量不足、评测偏差、符号推理可扩展性差、缺乏统一基准，且大多数 NeSy 方法对大规模开放域任务的鲁棒性不足，符号与神经融合的解释性与性能往往存在权衡。

---

## 194. Safe and Robust Domains of Attraction for Discrete-Time Systems: A Set-Based Characterization and Certifiable Neural Network Estimation

**arXiv ID:** 2603.03082 | [PDF](https://arxiv.org/pdf/2603.03082v1)

**作者:** Mohamed Serry `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**通讯引用:** 57709 | [OpenAlex ID](https://openalex.org/A5100450180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对离散时间非线性不确定系统的安全与鲁棒吸引域（DOA）精确估计框架，利用新的基于集合的价值函数与Bellman‑类型方程，并通过物理信息神经网络与正式验证技术获得可证明的DOA估计。

**💡 创新点**

创新点包括：①在不依赖离散网格或多项式逼近的前提下，将安全域与鲁棒性条件整合进价值函数定义；②构造了新的集合值价值函数并证明其为Bellman‑类型方程的唯一解；③提出了可嵌入集合映射的神经网络训练策略，直接将Bellman方程残差加入损失；④开发了基于α,β‑CROWN等工具的验证流程，保证学习得到的DOA具备正式安全保证。

**🔧 技术方法**

使用技术主要包括：集合论与Hausdorff度量、ℓp稳定性理论、Bellman‑类型方程推导、物理信息神经网络（feed‑forward NN）与损失函数融合、可逼近可达集的随机轨迹方法、以及正式验证工具（α,β‑CROWN、dReal）进行安全性与鲁棒性证明。

**📊 数据集**

实验数据集为四个非多项式离散不确定系统（双机功率系统、局部多项式稳定系统、带有限制的刚杆系统、受限有理系统），每个系统均构造安全集与扰动集，并在指定的验证域内采样点进行训练与验证。

**📈 对比分析**

与传统的最优椭圆域、基于参数相关Lyapunov函数以及仅考虑名义动力学的神经网络方法比较，本文在考虑完整扰动的情况下 consistently 提供更大的可证明DOA；在忽略扰动时则与最优椭圆域相当或略优；整体计算时间与传统方法相当，主要开销在可达集近似上。

**⚠️ 局限性**

限制主要在：①可达集近似需要大量随机轨迹，导致计算成本上升；②框架假设扰动映射可用有限维嵌入表示，若不满足需引入保守的外逼近；③目前仅适用于离散时间系统，连续时间推广尚未完成；④验证依赖现有工具的可扩展性与精度。

---

## 195. CuTe Layout Representation and Algebra

**arXiv ID:** 2603.02298 | [PDF](https://arxiv.org/pdf/2603.02298v1)

**作者:** Cris Cecka `[一作]` `[通讯]` (NVIDIA Research), Cris Cecka (NVIDIA Research)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

提出了一种新的数学规范，用于表示和操作张量的布局，并将其应用于 CUDA Tensor（CUTLASS 等）库的开发。

**💡 创新点**

主要创新点包括：① 分层布局表示（Hierarchical Layout），扩展传统的平面形状/步长表示，能够直接描述现代硬件指令所需的复杂映射；② 丰富的布局代数（Layout Algebra），包含拼接、合并、组合、补集、除法、分块、反演等操作，使得布局能够被编译时推理、验证和动态生成。

**🔧 技术方法**

技术方法涵盖：
- 使用层级元组（HTuple）构造形状、坐标与步长；
- 将步长定义为整数半模（Integer‑Semimodule）中的向量，实现通用线性和非线性映射；
- 通过形状与步长的函数合成得到布局函数；
- 设计完整的布局操作集合，实现布局的组合与变换；
- 在 CUTLASS、CuTe DSL 等实际库中实现并验证。

**📊 数据集**

论文主要通过示例和库实现来验证，未使用标准数据集；其实验基于 NVIDIA Hopper/Blackwell 等 GPU 架构的张量核心指令与复制指令。

**📈 对比分析**

与传统的平面布局相比，作者通过编译时检查、布局推导与统一的张量变换语义，展示了软件开发效率提升、正确性保证以及对多种张量操作（矩阵乘、张量收缩、卷积等）的通用支持；性能改进主要体现在能够直接满足硬件规定的固定布局，从而避免额外的数据搬移与重排，提升峰值性能。

**⚠️ 局限性**

局限性包括：
- 需要对现有代码库进行大幅重构以使用新的布局语义；
- 对于非整数或非线性步长的情况，解析与优化仍较复杂；
- 仍需进一步实验验证在更广泛的应用与架构上的通用性与可扩展性。

---

## 196. A Practical Guide for Establishing a Technical Debt Management Process (Preprint)

**arXiv ID:** 2603.03085 | [PDF](https://arxiv.org/pdf/2603.03085v1)

**作者:** Marion Wiese `[一作]` (University of Hamburg), Eva Bittner `[通讯]` (University of Hamburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

在三家公司三支团队中复现并迭代5步工作坊流程，建立技术债务管理（TDM）流程，并通过观察、问卷和TD‑SAGAT测量提升的技术债务意识；

**💡 创新点**

首次系统性验证工作坊+行动研究方法在实务中可行，提出基于属性的TD问题类型与优先级模型，形成可推广的TDM实践指南；

**🔧 技术方法**

采用行动研究、结构化工作坊、TD‑SAGAT（情境认知评估技术）、问卷调查及会议观察等方法；

**📊 数据集**

使用三支团队（TRUMPF、DATEV、另支团队）的实际工作记录、会议时长、TD‑SAGAT问卷及自评数据；

**📈 对比分析**

通过与工作坊前后自评及TD‑SAGAT结果对比，观察到技术债务意识的持续提升，未进行传统性能指标比较，强调流程有效性而非数值型度量；

**⚠️ 局限性**

受限于仅三支敏捷团队、研究者参与导致的潜在偏见、工具可视化与计算能力不足、以及缺乏大规模可推广验证。

---

## 197. Coalgebras for categorical deep learning: Representability and universal approximation

**arXiv ID:** 2603.03227 | [PDF](https://arxiv.org/pdf/2603.03227v1)

**作者:** Dragan Mašulović `[一作]` (University of Novi Sad), Dragan Mašulović `[通讯]` (University of Novi Sad)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5054883203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了基于coalgebraic形式主义的等变表示理论框架，并证明了在此框架下连续等变映射可由神经网络近似的通用逼近定理；

**💡 创新点**

创新点在于将等变映射的概念与分类深度学习中的嵌入与不变行为用范畴论的coalgebraic视角统一，构建了一个兼容嵌入的端函子，并给出了相应的通用逼近理论；

**🔧 技术方法**

采用范畴论工具（函子、端函子、coalgebraic 结构）来建模数据集嵌入与不变行为，随后利用这些结构推导等变映射的逼近性；

**📊 数据集**

本文未提供具体数据集或实验数据；

**📈 对比分析**

论文没有给出实验比较或性能评估；

**⚠️ 局限性**

主要局限在于尚缺乏针对实际神经网络架构的实现与验证，理论范围尚待进一步扩展与实证。

---

## 198. TenExp: Mixture-of-Experts-Based Tensor Decomposition Structure Search Framework

**arXiv ID:** 2603.02720 | [PDF](https://arxiv.org/pdf/2603.02720v1)

**作者:** Ting-Wei Zhou `[一作]` (University of Electronic Science and Technology of China), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 31695 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于专家混合的张量分解结构搜索框架TenExp，能够在无监督条件下动态选择单一或混合多种张量分解来逼近多维数据。

**💡 创新点**

创新点在于①构建跨因子交互的候选搜索集，涵盖多种因子交互族（模式‑n乘、张量收缩、t‑product等）；②设计统一的能量‑比率基阶数估计；③引入 top‑k 门控机制，实现对单一或混合分解的自适应激活；④给出TenExp的误差上界，证明其逼近能力。

**🔧 技术方法**

采用张量分解（CP、Tucker、TT、TR、FCTN、T‑SVD）理论、能量‑比率阶数估计、门控网络（softmax+top‑k）、Adam 优化以及无监督损失（观测域误差）来实现。

**📊 数据集**

实验数据包括：50×50×50 的合成张量；三幅多光谱图像（Cloth、Beads、Jelly）；四段彩色视频（News、Claire、Grandma、Akiyo）；四组光场数据（Greek、Museum、Medieval2、Vinyl）。

**📈 对比分析**

与 Tucker、TF、TNGreedy、SVDinsTN（张量网络搜索）、HaLRTC、SiLRTCTT、TRLRF、HTNN、LTNN、SVDinsTN 等经典方法在压缩率（CR）/相对误差（RE）/PSNR/SSIM 上进行对比。TenExp 在合成数据上 RE 低于或等于现有方法，在真实数据上 PSNR/SSIM 显著高于竞争者，且压缩率相当或更好，说明混合分解显著提升恢复效果。

**⚠️ 局限性**

局限性包括：①需要预设能量阈值和 top‑k 参数，影响搜索结果；②计算量随候选数增加而上升，尤其是高阶张量；③对某些分解（如 t‑product）仅适用于三阶张量；④当前实验仅验证了无监督重建，尚未探讨在有监督或半监督场景下的性能。

---

## 199. Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs

**arXiv ID:** 2603.02731 | [PDF](https://arxiv.org/pdf/2603.02731v1)

**作者:** Wuyue Zhang `[一作]` (Zhejiang Lab), Mou Sun `[通讯]` (Zhejiang Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Hopper GPU上实现了一套软件模拟的MXFP4混合精度训练框架，使大规模Mixture‑of‑Experts（MoE）模型能够在没有原生FP4算子支持的情况下，以FP4精度压缩激活与专家间通信，从而降低激活内存和通信带宽需求，并保持与FP8基线相当的收敛性。

**💡 创新点**

创新点包括：
- 直接的FP4→FP8位级转换算法，避免了BF16中间转换；
- 行到列的自适应缩放对齐，兼容MXFP4的块级尺度与FP8的列布局；
- 针对MoE张量的布局感知CUDA内核（量化、反量化、转置融合）大幅减少内存读写；
- 前向阶段使用FP4、后向阶段保持FP8的非对称精度策略，兼顾吞吐量与数值稳定；
- 在DeepEP通信库中加入FP4子字节打包与解包，真正实现了子字节级的专家并行通信。

**🔧 技术方法**

使用的技术与方法包括：
- 软件实现的MXFP4（E2M1 4‑bit浮点）量化；
- 直接位级FP4→FP8转换与层级尺度对齐；
- 量化/反量化、转置与矩阵乘法融合的CUDA核；
- 采用Transformer Engine的FP8块级量化骨架；
- DeepEP专家并行通信的FP4子字节支持；
- 训练中使用AdamW、cosine学习率、梯度裁剪等标准优化手段。

**📊 数据集**

训练数据为大规模通用语言模型预训练语料（如Common Crawl、Wikipedia等），在实验中使用了与DeepSeek‑V3相同的10B‑级别大规模语料库，构建了236B和671B参数的MoE模型。

**📈 对比分析**

与BF16和FP8基线进行对比：在671B MoE模型上，MXFP4将峰值激活内存降低14.8%（≈11.8 GB），并将吞吐量从1157 TGS提升至1302 TGS，提升幅度12.5%；在236B模型上也实现了≈6.9–7.2%内存减小并保持或超过FP8吞吐；收敛曲线与BF16基本一致，FP4误差在0.61%以内。

**⚠️ 局限性**

局限性包括：
- 仅在Hopper GPU上验证，缺乏对更低算力或其他架构的适配；
- 后向传播仍使用FP8，导致部分通信与计算上仍有损失；
- 需要自研的CUDA内核与DeepEP改造，迁移成本较高；
- 对极小模型或非MoE架构的效果未评估；
- 仍依赖软件层面的缩放与量化，可能在极端动态范围或梯度爆炸场景下产生数值不稳定。

---

## 200. LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval

**arXiv ID:** 2603.02888 | [PDF](https://arxiv.org/pdf/2603.02888v1)

**作者:** Minh-Chi Phung `[一作]` (University of Information Technology), Vu-Hung Dao `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了LLandMark多智能体框架，实现多模态视频检索，融合语义搜索、ASR、OCR、对象检测以及自动化图像检索，专注于越南文化地标的检索。

**💡 创新点**

创新点包括：①模块化多智能体结构与自动检索规划；②使用Gemini与LlamaIndex对PaddleOCR输出进行越南文字重音修复；③LLM驱动的图像检索管线，将地标文本转化为真实图像检索，提升检索精度。

**🔧 技术方法**

采用了CLIP ConvNeXt-XXLarge嵌入、WhisperX ASR、YOLOv9-e对象检测、PaddleOCR+Gemini+LlamaIndex OCR修正、Milvus向量数据库、MongoDB索引、Elasticsearch文本检索等技术。

**📊 数据集**

在HCMAIC 2025 250 GB的视频语料库上进行实验，包含广播、纪录片等多领域视频，评测任务包括关键词检索、视觉问答和时序推理。

**📈 对比分析**

使用官方评测协议的Mean Top‑k R‑Score与基线嵌入检索对比，最终得分77.40/88，排名680队伍中的第56位，展示了在多模态检索和地标识别方面的稳健性能。

**⚠️ 局限性**

限制包括：对计算资源需求高，仍需人工维护地标知识库，主要针对越南语境，跨语言、跨文化通用性和对非地标文本检索的适应性有限。

---

## 201. Speech recognition assisted by large language models to command software orally -- Application to an augmented and virtual reality web app for immersive molecular graphics

**arXiv ID:** 2603.02901 | [PDF](https://arxiv.org/pdf/2603.02901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 202. Implicit Bias in Deep Linear Discriminant Analysis

**arXiv ID:** 2603.02622 | [PDF](https://arxiv.org/pdf/2603.02622v1)

**作者:** Jiawen Li `[一作]` `[通讯]` (University of New South Wales), Jiawen Li (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对深度线性判别分析（Deep LDA）引起的隐性偏差进行了初步的理论分析，探讨了其在优化几何中的作用。

**💡 创新点**

创新点在于首次从理论上分析了Deep LDA的隐性正则化，揭示了其在平衡初始化下的梯度流动特性。

**🔧 技术方法**

使用了深度线性网络（DLN）作为理论分析的框架，研究了梯度流动的动态特性。

**📊 数据集**

实验中使用了合成的正定矩阵作为类内和类间散布矩阵，特征空间维度为5。

**📈 对比分析**

通过与不同层数的DLN进行比较，发现无论网络层数如何，隐性正则化保持不变，且深层网络对弱特征的惩罚更强，促进了特征稀疏性。

**⚠️ 局限性**

限制在于仅分析了简化的对角线性网络，未考虑非线性激活函数或其他复杂网络架构，未来需要在真实数据上进行进一步验证。

---

## 203. SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning

**arXiv ID:** 2603.02240 | [PDF](https://arxiv.org/pdf/2603.02240v1)

**作者:** Varun Pratap Bhardwaj `[一作]` `[通讯]` (Independent Research), Varun Pratap Bhardwaj (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个本地优先、无云依赖的多智能体记忆系统，结合贝叶斯信任评分和自适应学习-排序，防御OWASP ASI06记忆中毒。

**💡 创新点**

引入了本地化架构消除云攻击面，贝叶斯信任评分实现对恶意写入的主动拦截，以及零LLM自适应学习排序提升检索质量。

**🔧 技术方法**

SQLite+FTS5+Write-Ahead Logging、Leiden聚类、TF-IDF、HNSW、事件总线(SSE/WebSocket)、Beta-Binomial信任模型、LambdaRank学习排序、三层行为分析。

**📊 数据集**

使用5个主题（Web开发、机器学习、数据库、DevOps、API）模板生成的合成记忆数据（100-5k条），并进行人工评估的70条人类相关性评分。

**📈 对比分析**

与七个基准维度对比；在1k记忆下搜索延迟10.6ms，写入吞吐220写/秒；信任分隔gap=0.90；睡眠攻击信任降幅72%；自适应排序使NDCG@5提升104%，仅+20ms延迟。

**⚠️ 局限性**

需要持续使用才能获得足够反馈；合成训练可能导致分布漂移；图构建复杂度O(n^2)限制在1万条；信任未融入排序；缺乏正式用户研究和专属基准。

---

## 204. LiveAgentBench: Comprehensive Benchmarking of Agentic Systems Across 104 Real-World Challenges

**arXiv ID:** 2603.02586 | [PDF](https://arxiv.org/pdf/2603.02586v1)

**作者:** Hao Li `[一作]` (Ant Group), Sikang Bian `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了LiveAgentBench评测基准，涵盖104个真实场景任务并提出持续更新的Social Perception‑Driven Data Generation（SPDG）流程

**💡 创新点**

创新点在于通过SPDG实现任务从真实用户问答中筛选、重构、标注，保证任务的现实性、难度与可验证性，并实现基准的持续迭代与去污染

**🔧 技术方法**

采用LLM与人工双盲标注、工具调用（浏览器、文件、音视频、Android/iOS等）以及零射击Prompt与Pass@1评估方法

**📊 数据集**

使用从社交媒体、问答社区、短视频平台等公开来源收集的用户问题，构成374个任务（125验证+249测试）

**📈 对比分析**

与多款LLM（Qwen3、Claude、GPT‑4o等）和自治代理（Manus、OpenAI Deep Research、Perplexity Research等）及人类进行Pass@1对比，发现LLM约13.5%，代理约23.9%，人类约69.3%；代理仍显著落后人类且工具稳定性和环境知识是主要瓶颈

**⚠️ 局限性**

局限性包括中文语料为主，跨文化多样性不足；任务重构导致部分不自然；缺乏对环境背景知识与工具稳定性的充分支持

---

## 205. CMoE: Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots

**arXiv ID:** 2603.03067 | [PDF](https://arxiv.org/pdf/2603.03067v1)

**作者:** Shihao Ma `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 3664 | [OpenAlex ID](https://openalex.org/A5102769588)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个单阶段强化学习框架CMoE，融合Mixture of Experts与对比学习，使人形机器人能够在多种复杂地形（斜坡、楼梯、裂缝、障碍、混合等）中自主导航。

**💡 创新点**

创新点在于在MoE门控网络中引入对比学习：最大化同一地形内专家激活的一致性、最小化不同地形间的相似度，从而促使专家专精不同地形，克服传统Vanilla MoE的“lazy gating”问题。

**🔧 技术方法**

技术手段包括：PPO actor‑critic MoE架构、β‑VAE状态估计、Autoencoder地形感知、SwAV式对比学习与Sinkhorn‑Knopp聚类、t‑SNE可视化、域随机化、以及基于雷达点云的真实地形图构建。

**📊 数据集**

使用的数据集主要来自IsaacGym仿真，包含8种地形（斜坡、楼梯、裂缝、障碍、混合等）以及在Unitree G1真实机器人上采集的雷达点云生成的地形图。

**📈 对比分析**

与基线（无MoE）和Vanilla MoE进行比较，采用成功率与平均行进距离两个指标。CMoE在所有地形上的成功率提升约10–20%，平均行进距离比Vanilla MoE多1–3米，并能完成30 cm阶梯、80 cm宽缝隙、17°坡等更具挑战性的任务，显著优于现有方法。

**⚠️ 局限性**

局限性包括：对全身控制的适配尚未完成；对比学习与门控网络的计算开销较大；在高度动态或未知地形的泛化能力仍需进一步验证；以及对极端障碍和快速环境变化的鲁棒性尚未完全覆盖。

---

## 206. RAIN: Secure and Robust Aggregation under Shuffle Model of Differential Privacy

**arXiv ID:** 2603.03108 | [PDF](https://arxiv.org/pdf/2603.03108v1)

**作者:** Yuhang Li `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 15511 | [OpenAlex ID](https://openalex.org/A5100634361)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个在Shuffle‑DP框架下实现鲁棒、可验证且通信高效的聚合方法——RAIN。

**💡 创新点**

核心创新在于将聚合转至符号空间，利用汉明距离信任评分在无身份信息的环境下抑制恶意更新；并引入两台非同谋服务器的加密洗牌与聚合协议以及流式MAC实时完整性校验，三者协同实现安全、鲁棒与可验证性。

**🔧 技术方法**

技术手段包括：加密的局部高斯噪声 + 符号编码；加法共享与Beaver三元组的MPC计算；随机置换 + 重新遮蔽的加密洗牌；基于哈希距离与ReLU的权重调整；流式加密MAC校验。

**📊 数据集**

在MNIST、Fashion‑MNIST和CIFAR‑10等标准图像分类数据集上进行评估，使用了IID与非IID的客户端划分。

**📈 对比分析**

与FedAvg、SignSGD、FLGuard、Camel、FLAME、FLOD等基线相比，RAIN在隐私保护（Shuffle‑DP）下保持相当甚至更高的模型准确率，鲁棒性在高达90%恶意客户端比例时仍能保持≈80%准确率；通信成本低90×，聚合时间快10×，并实现了完整性 100% 检测。

**⚠️ 局限性**

局限性包括：假设客户端数据IID且仅考虑单一非同谋服务器；依赖公开的根数据集产生参考方向；符号编码会丢失梯度幅值信息，可能在极端噪声下影响收敛；未在异步或更强服务器攻击场景下验证。

---

## 207. Blockchain Communication Vulnerabilities

**arXiv ID:** 2603.02661 | [PDF](https://arxiv.org/pdf/2603.02661v1)

**作者:** Andrei Lebedev `[一作]` (University of Sydney), Vincent Gramoli `[通讯]` (Redbelly Network)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较Algorand、Aptos、Avalanche、Redbelly、Solana在定向负载、瞬态故障、包丢失、停止攻击和领导隔离等五类网络攻击下的通信协议脆弱性。

**💡 创新点**

提出统一的协议无关攻击模型，并在相同实验环境中系统性比较不同区块链的安全性，为后续基于通信层的安全评估提供基准。

**🔧 技术方法**

采用Linux tc注入包丢失、节点崩溃与恢复、实时日志监控、200 TPS交易负载模拟，以及TCP/UDP混合网络层实现的攻击与测量框架。

**📊 数据集**

在Proxmox集群上部署25台VM（4vCPU/8GB），使用官方发行版节点，构建基准网络并以200 TPS客户端负载进行实验。

**📈 对比分析**

通过对比吞吐量、延迟分位数和交易丢失率发现：Aptos受定向负载影响严重；Avalanche受瞬态故障与包丢失显著影响；Solana易在大规模故障后停滞；TCP链易受包丢失，UDP+纠删码链更稳健。

**⚠️ 局限性**

研究受限于节点规模有限、单一配置（无warmup/动态费）、未覆盖跨链与共识细节，仅关注通信层，缺乏对侧链与更大规模网络的全面评估。

---

## 208. Joint Optimization of Model Partitioning and Resource Allocation for Anti-Jamming Collaborative Inference Systems

**arXiv ID:** 2603.02579 | [PDF](https://arxiv.org/pdf/2603.02579v1)

**作者:** Mengru Wu `[一作]` (Zhejiang University of Technology), Hyundong Shin `[通讯]` (Kyung Hee University)

**通讯引用:** 8248 | [OpenAlex ID](https://openalex.org/A5007557286)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在恶意干扰环境下，对设备‑边缘协同推理系统进行模型分割与资源分配的联合优化，旨在提升系统的时延与准确度收益（RDA）。

**💡 创新点**

① 首次系统性研究恶意干扰对中间特征数据（IFD）传输以及 DNN 分割决策的影响；② 通过数据回归得到 SINR 与分割点对推理准确率的闭式关系；③ 引入 RDA 指标综合评价时延与准确率；④ 采用量子遗传算法（QGA）解决离散分割子问题，突破传统遗传算法的早熟收敛。

**🔧 技术方法**

数据回归、凸优化、Karush‑Kuhn‑Tucker (KKT) 条件、交替优化 (AO)、量子遗传算法 (QGA)、混合整数非线性规划 (MINLP) 求解框架。

**📊 数据集**

ResNet‑18 在 CIFAR‑10 数据集上训练，用于构建推理任务与准确率回归模型。

**📈 对比分析**

与四个基线（本地计算、边缘服务器计算、固定功率传输、传统 GA 分割）对比。实验显示：提出方案在各种设备计算能力与干扰功率下均获得最高 RDA，能在保持准确率阈值的前提下显著降低总时延，表现出更优的抗干扰能力。

**⚠️ 局限性**

受限于：① 需要预先进行准确率回归，假设干扰功率已知；② 对单一干扰源和正交频分的假设；③ QGA 计算复杂度随设备数增大而显著提升；④ 方案对极端恶意干扰或多干扰源场景的鲁棒性尚待验证。

---

## 209. Shared (Mis)Understandings and the Governance of AI: A Thematic Analysis of the 2023-2024 Oversight of AI Hearings

**arXiv ID:** 2603.03193 | [PDF](https://arxiv.org/pdf/2603.03193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 210. BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?

**arXiv ID:** 2603.03194 | [PDF](https://arxiv.org/pdf/2603.03194v1)

**作者:** Guoxin Chen `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 SearchSWE 的多维度软件工程基准（覆盖跨仓库、领域专长、依赖迁移和从规格生成仓库）并提出了整合深度搜索与编码的 SearchSWE 框架，系统评估多种 LLM 代码代理的表现。

**💡 创新点**

创新点在于：①从“分辨率范围”和“知识范围”两轴扩展 benchmark，首次将跨仓库调用、领域知识、全仓库迁移和全仓库生成纳入评测；②设计了搜索与编码协同的 Agent 框架，并通过块列表防止直接访问目标仓库，严格验证搜索与编码的真正融合；③系统分析搜索对编码的影响，揭示二者仍未实现有效统一。

**🔧 技术方法**

技术主要包括：OpenHands 代码代理框架、Gemini 3 Pro、GPT‑5.2、DeepSeek‑V3.2 等大模型；基于 Docker 的环境构建与自动化测试；搜索工具（web 搜索）与浏览器工具（内容检索/摘要）；块列表机制（正则过滤）防止作弊；多轮交互与工具调用统计分析。

**📊 数据集**

使用 500 条真实世界 GitHub 问题实例，覆盖 246 个仓库，包含 200 条 CrossRepo、72 条 DomainFix、178 条 DepMigrate 以及 50 条 Doc2Repo 任务；还采集了多语言、持续更新与多文件任务的相关数据。

**📈 对比分析**

对比方法：在 OpenHands 与 SearchSWE 两个框架下分别评估 9 种模型，报告“Resolved Rate”（P2P+F2P 通过率）或“Pass Rate”“Almost Correct Count”。实验显示所有模型在 SearchSWE 上均低于 45% 成功率，最高平均 Resolved Rate 为 41.81%；搜索增强在某些任务（DomainFix、DepMigrate）可提升约 5–8%，但整体收益不稳定，甚至有下降。

**⚠️ 局限性**

局限性：①搜索与编码的整合效果不一致，模型对搜索结果的筛选与利用能力不足；②测试主要集中于 Python，跨语言通用性待验证；③块列表可能无法覆盖所有作弊手段，仍需更完善的安全机制；④在 Doc2Repo 任务中，模型仅能实现零散功能，难以构建完整系统，显示仍缺乏全局架构推理能力。

---

## 211. ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large Language Models via Group Relative Policy Optimization

**arXiv ID:** 2603.02939 | [PDF](https://arxiv.org/pdf/2603.02939v1)

**作者:** Yang Zhan `[一作]` (Northwestern Polytechnical University), Yan Li `[通讯]` (Wuhan University)

**通讯引用:** 27607 | [OpenAlex ID](https://openalex.org/A5100380419)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ShipTraj-R1框架，将船舶轨迹预测重新定义为文本到文本生成任务，利用链式思维（CoT）与强化学习相结合的方式提升预测准确性。

**💡 创新点**

创新点包括：①通过动态提示将冲突船舶的轨迹信息嵌入提示，诱导LLM进行自适应CoT推理；②设计基于规则的奖励函数同时激励推理格式和坐标精度；③使用Group Relative Policy Optimization（GRPO）对Qwen3 LLM进行强化微调，形成针对航行安全的自我改进机制。

**🔧 技术方法**

使用的技术包括：大语言模型Qwen3、Chain-of-Thought（CoT）推理、动态提示设计、规则化奖励函数（思维格式奖励+预测误差奖励）、GRPO强化学习算法以及航行域冲突检测（QSD）。

**📊 数据集**

采用了两份真实AIS轨迹数据集：成山角海域（CSJP）和曹斐甸港口（CFDP），共计约5,600条完整轨迹，包含约60,000个GPS点。

**📈 对比分析**

与传统深度学习模型、现有LLM轨迹预测模型及开源LLM基线进行对比，ShipTraj-R1-8B在CSJP和CFDP上的最终位移误差（FDE）和平均位移误差（ADE）均取得最低值，优于最先进方法的误差下降幅度超过50%。

**⚠️ 局限性**

局限性：仅验证了短期预测（最多20秒）效果；长周期预测仍待研究；模型训练依赖昂贵的RL过程；仅在两个海域AIS数据上测试，泛化性尚需进一步评估。

---

## 212. Undecided State Dynamics with Many Opinions

**arXiv ID:** 2603.02636 | [PDF](https://arxiv.org/pdf/2603.02636v1)

**作者:** Colin Cooper `[一作]` (Kings College London), Takeharu Shiraga `[通讯]` (Chuo University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5067826040)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了未决状态动态（USD），这是一个基本的共识过程，其中每个顶点持有k个已决定的意见或未决状态。考虑了八卦模型和人口协议模型。

**💡 创新点**

首次为任意2≤k≤n和任意初始配置提供了USD的共识时间保证，解决了之前研究中的限制性假设问题。

**🔧 技术方法**

使用了八卦模型和人口协议模型的技术，分析了在这两种模型下的共识时间。

**📊 数据集**

没有具体提到使用的数据集，但研究涉及n个顶点的完全图。

**📈 对比分析**

在八卦模型中，USD在O(min{k,√(n)})个同步轮次内以高概率达成共识；在人口协议模型中，USD在O(min{kn,n^3/2})个异步交互中以高概率达成共识。提供了与现有上界相匹配的下界，表明这些上界是基本最优的。

**⚠️ 局限性**

限制在于对于k的较大值（例如k≥√(n)）的分析仍然存在挑战，且在某些情况下，未决状态的动态行为可能会导致共识时间的变化。

---

## 213. Preconditioned Score and Flow Matching

**arXiv ID:** 2603.02337 | [PDF](https://arxiv.org/pdf/2603.02337v1)

**作者:** Shadab Ahamed `[一作]` (University of British Columbia), Eldad Haber `[通讯]` (University of British Columbia)

**通讯引用:** 9461 | [OpenAlex ID](https://openalex.org/A5029362936)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了通过可逆预处理（preconditioning）来改善流匹配（flow matching）和基于分数的扩散模型（score‑based diffusion）的优化过程，从而提升生成质量。

**💡 创新点**

创新点在于：①从理论上揭示中间分布协方差的条件数对梯度下降和 SGD 训练的瓶颈作用；②设计可逆预处理框架，使得在不改变生成模型本身的前提下重塑数据几何，显著降低协方差条件数；③通过可逆预处理实现了更稳定的训练，避免了子最优平稳现象。

**🔧 技术方法**

主要技术包括：流匹配与分数匹配的连续时间训练、可逆预处理（正则化的归一化流或低容量流）、线性高斯模型和高斯混合模型的解析分析、梯度下降/ SGD 的收敛理论、条件数诊断与 MMD、FID 等评价指标。

**📊 数据集**

实验数据集涵盖：二维模拟（高斯运输、Swiss‑roll、checkerboard）、MNIST（VAE 隐空间），以及高分辨率图像数据集 LSUN Churches、Oxford Flowers‑102、AFHQ Cats。

**📈 对比分析**

与未预处理的基线相比，预处理后在所有任务中都能显著降低 FID（例如 MNIST 从 13.83 降到 2.62），以及 Sliced‑Schaefer 距离等分布指标；在二维示例中可视化显示预处理消除了优化停滞并显著提升对齐度。

**⚠️ 局限性**

局限性：①预处理方法需要额外的可逆网络，训练成本和模型复杂度略有提升；②在大规模高分辨率数据上，构造有效的预处理器仍是挑战，需针对每个数据集设计；③理论分析基于线性高斯模型，实际非线性模型的收敛特性仍待深入研究。

---

## 214. Inverse Reconstruction of Shock Time Series from Shock Response Spectrum Curves using Machine Learning

**arXiv ID:** 2603.03229 | [PDF](https://arxiv.org/pdf/2603.03229v1)

**作者:** Adam Watts `[一作]` (Los Alamos National Laboratory), Ryan Bowering `[通讯]` (University of Rochester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了逆向 Shock Response Spectrum（SRS）问题，提出使用条件变分自编码器（CVAE）从给定 SRS 直接生成加速度时间序列。

**💡 创新点**

首次将 CVAE 应用于 SRS 逆变，直接学习 SRS 到时间域的映射，消除了传统优化中对基底的限制和高计算量，并通过大规模合成数据提供多样化训练样本。

**🔧 技术方法**

使用 PyTorch 实现的 CNN + 全连接编码器-解码器结构，配合可微 SRS 运算；采用 RMSLE、PSD、波形形状等多项联合损失，利用 GPU 加速合成与训练。

**📊 数据集**

合成 300k 条随机基底冲击波 + 约 45k 条真实冲击数据做训练；同时提供四个 hold‑out 数据集（A‑D，包含操作、地震、合成）用于评估。

**📈 对比分析**

与经典 Sum‑of‑Decayed‑Sinusoids (SDS) 及 SDS+GA 进行 RMSLE 与 dB 误差对比，CVAE 在 94–98% 的样本上优于 SDS，平均 RMSLE 约 0.095，dB 误差均在 ±3 dB 范围内；推理时间仅 0.3 ms，速度比 SDS 快 4–6 个数量级。

**⚠️ 局限性**

生成多样性有限，潜变量利用不足；仅适用于约 0.274 s 的冲击波，无法覆盖多次冲击或极低频/高频复合场景，对非常复杂或低频高分辨率信号的泛化尚需提升。

---

## 215. Behavior-Aware Anthropometric Scene Generation for Human-Usable 3D Layouts

**arXiv ID:** 2603.02662 | [PDF](https://arxiv.org/pdf/2603.02662v1)

**作者:** Semin Jin `[一作]` (Hanyang University), Kyung Hoon Hyun `[通讯]` (Hanyang University)

**通讯引用:** 10505 | [OpenAlex ID](https://openalex.org/A5100732907)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉‑语言模型推理并结合个体人体尺寸的行为‑感知人体测量场景生成框架，能够自动生成符合人类操作需求的 3D 室内布局。

**💡 创新点**

创新点在于将 VLM 产生的物体功能与人机交互模式映射为可微分的空间约束，并以个体人体测量为参数化依据，实现了从语义可行到功能可用的布局优化。

**🔧 技术方法**

使用的技术包括 GPT‑4o 视觉‑语言模型、SMPL 人体重建、梯度优化（多组约束求解）以及 MMAction2 动作识别。

**📊 数据集**

数据集主要有公开的 Objaverse 3D 资产、Human Dimension 与 Interior Space 人体测量数据库、以及通过 SMPL 生成的合成人体尺寸。

**📈 对比分析**

与 LayoutVLM 的对比实验采用了碰撞‑边界分数、专家感知评估和 1:1 物理空间用户实验；结果显示该方法在用户感知、任务完成时间、路径效率和交互空间占用率上均显著优于基线（均显著差异，p<0.01，效应量>0.7）。

**⚠️ 局限性**

局限性包括场景数量有限、仅考虑水平布局、在多用户情况下使用最大尺寸限制导致对小尺寸用户体验欠佳、对视觉不明显的隐藏功能家具推理受限以及未在极端或高度规范化环境（如厨房、浴室）中验证。

---

## 216. Goal-Oriented Semantic Communication for ISAC-Enabled Robotic Obstacle Avoidance

**arXiv ID:** 2603.02291 | [PDF](https://arxiv.org/pdf/2603.02291v1)

**作者:** Wenjie Liu `[一作]` (Kings College London), Henk Wymeersch `[通讯]` (Chalmers University of Technology)

**通讯引用:** 21065 | [OpenAlex ID](https://openalex.org/A5033860704)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于目标导向语义通信（GOSC）的ISAC框架，用于无人机障碍物规避，闭环结合Kalman滤波、马氏距离动态窗口（MD‑DWA）和效用感知深度Q网络（E‑DQN），实现了仅在必要时传输感知与C&C信号的策略；

**💡 创新点**

创新点在于①把感知与控制信号的发送完全联动并基于信息价值（VoI）做决策；②使用马氏距离理论推导最小安全距离阈值，改进传统DWA的保守性；③采用E‑DQN学习何时发送感知、何时发送控制，显著减少通信开销；

**🔧 技术方法**

核心技术包括：多输入多输出毫米波ISAC、Kalman滤波预测、MUSIC+PIFFT测距、马氏距离动态窗口算法、效用感知DQN（E‑DQN）以及对应的奖励设计；

**📊 数据集**

实验使用仿真环境，随机布置10个动态障碍物，设置多种参数（如无人机初始位置、障碍速度、雷达交叉截面等），未使用公开数据集；

**📈 对比分析**

与传统连续ISAC、周期性传输、事件触发等基线方案比较，GOSC在保持100%成功率的同时将感知+C&C信号数量降低92.4%，传输时隙降低85.5%，任务完成时间与路径长度仅略高于传统方案，且安全距离保持可观；

**⚠️ 局限性**

局限性包括：仅在二维平面、已知起始/目标位置的仿真环境；对实际雷达噪声、遮挡、通信干扰等不完整；DQN训练需要大量经验，模型收敛性和实时性待进一步验证；

---

## 217. Sequence-Level Unsupervised Training in Speech Recognition: A Theoretical Study

**arXiv ID:** 2603.02285 | [PDF](https://arxiv.org/pdf/2603.02285v1)

**作者:** Zijian Yang `[一作]` (RWTH Aachen University), Hermann Ney `[通讯]` (RWTH Aachen University)

**通讯引用:** 46639 | [OpenAlex ID](https://openalex.org/A5112501010)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于分类误差上界的无监督语音识别理论框架，并给出了可行性条件。

**💡 创新点**

创新点在于：①引入结构约束和语言模型矩阵满列秩两个必要且充分的条件；②在此条件下推导出无监督训练的误差上界；③据此提出单阶段序列级交叉熵损失函数。

**🔧 技术方法**

使用理论分析、概率论中的KL散度与Pinsker不等式、线性代数中的左逆矩阵、以及动态规划实现序列级交叉熵的计算。

**📊 数据集**

实验验证仅在模拟环境下完成；文中用LibriSpeech转录文本计算语言模型矩阵的最小奇异值以检验满列秩假设。

**📈 对比分析**

通过数值模拟验证了误差上界的正确性；论文未给出在真实语料上的对比实验或性能指标。

**⚠️ 局限性**

局限性：需要满足结构约束和语言模型矩阵满列秩这两条条件；理论主要在模拟数据上验证，缺乏在实际无监督语音数据上的实验与性能评估。

---

## 218. The power of small initialization in noisy low-tubal-rank tensor recovery

**arXiv ID:** 2603.02729 | [PDF](https://arxiv.org/pdf/2603.02729v1)

**作者:** ZHiyu Liu `[一作]`, Yao Wang `[通讯]` (Xi’an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在噪声低管秩张量恢复中使用小初始化的因式分解梯度下降（FGD）方法，证明其恢复误差与过估计的管秩无关并趋近最优；

**💡 创新点**

提出小初始化+早停策略，使FGD在不需要先验管秩信息时即可实现接近最小化误差的性能；

**🔧 技术方法**

利用t‑SVD框架、Burer‑Monteiro因式分解以及t‑RIP理论，对FGD进行全局收敛与误差上界分析；

**📊 数据集**

实验数据包括合成随机张量、Berkeley Segmentation Dataset图像和视频补全数据集；

**📈 对比分析**

与光谱初始化、随机大规模初始化、凸核范数最小化、非凸UTF/GTNN-HOP以及基于秩估计的TCTF/TC-RE等方法对比，FGD‑小初始化/早停在重建误差、PSNR/RE等指标上均优于或匹配最优基线，且样本复杂度更低；

**⚠️ 局限性**

主要局限在于需满足t‑RIP假设（对测量算子要求较高），并对称张量结构有一定假设；对非常大尺寸或非Gaussian噪声场景的理论和实践效果仍待进一步验证。

---

## 219. Next Embedding Prediction Makes World Models Stronger

**arXiv ID:** 2603.02765 | [PDF](https://arxiv.org/pdf/2603.02765v1)

**作者:** George Bredis `[一作]` (T Tech), Ruslan Rakhimov `[通讯]` (T Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无需像素重建的基于世界模型的强化学习代理 NE-Dreamer，利用因果时序变压器预测并对齐下一步编码器嵌入，提升了在部分可观测环境中的长期记忆与导航能力。

**💡 创新点**

核心创新在于：①用“下一嵌入预测”取代传统的像素重建，直接在表示空间中强化时间可预测性；②引入因果时序变压器和Barlow Twins正则化，确保表示的表达力与非崩塌性；③通过目标移位（next-step shift）显式要求模型预测未来状态，避免仅靠即时一致性导致的表示漂移。

**🔧 技术方法**

使用的技术包括：Recurrent State‑Space Model (RSSM) 作为动态后端；因果时序变压器 (causal transformer) 进行序列建模；Barlow Twins 损失实现重叠减少与正则化；Actor‑Critic 在模型产生的“想象”轨迹上训练；对比学习与奖励/终止预测作为额外监督；与 DreamerV3 类似的控制策略架构。

**📊 数据集**

主要数据集：DeepMind Lab 的 Rooms 任务（要求长时记忆和空间推理）和 DeepMind Control Suite（连续控制基准）。

**📈 对比分析**

实验在统一的计算与模型容量下（12M 参数、50M 环境步 DMLab、1M 步 DMC，5 种随机种子）进行。结果显示：在 Rooms 任务中 NE‑Dreamer 超过 DreamerV3、R2‑Dreamer、DreamerPro，获得显著更高的回报；在 DMC 上 NE‑Dreamer 与 DreamerV3 及其他 decoder‑free 方法持平或略优，表明去掉重建不会牺牲常规控制性能。

**⚠️ 局限性**

局限性：实验聚焦于结构化、长时记忆挑战；在视觉细节要求极高或高保真度任务中，下一嵌入预测可能无法完全匹配像素重建的表现；未来工作需探索不同对齐损失、变压器规模与更复杂视觉环境的适用性。

---

## 220. Credibility Governance: A Social Mechanism for Collective Self-Correction under Weak Truth Signals

**arXiv ID:** 2603.02640 | [PDF](https://arxiv.org/pdf/2603.02640v1)

**作者:** Wanying He `[一作]` (Wuhan University), Yipeng Kang `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并验证了Credibility Governance（CG）机制，用动态信誉权重重新分配社交平台上的影响力，以在弱真理信号下实现更快的真相收敛和对谣言冲击的恢复。

**💡 创新点**

创新点在于：①采用公共证据变动（ΔΘ）而非纯粹的支持度作为信誉奖励基础；②引入反泡沫惩罚、早期参与奖励等三种核心组件；③在LLM驱动的多智能体模拟框架POLIS中评估CG，克服了传统基于投票或质押的聚合方法在噪声与延迟环境下的缺陷。

**🔧 技术方法**

技术包括：大型语言模型驱动的多智能体代理、双世界（物理与意见）耦合的POLIS模拟平台、动态信誉更新公式与参数化物理进展模型、以及对抗性攻击与噪声敏感性分析。

**📊 数据集**

使用的“数据集”是合成的实验轨迹：在POLIS中模拟两条科学议题（真实与虚假）和100名代理的不同起始信念，配合噪声扰动与可控的误导冲击。

**📈 对比分析**

与三种基线（无治理、社交媒体投票、Web3质押）比较，CG在大多数噪声与攻击场景下能更快收敛到真相、提升最终准确率、缩短错误修正时间，并在反泡沫和早期参与奖励下保持对策略操纵的鲁棒性。

**⚠️ 局限性**

局限性包括：①对公共证据代理（Θ）对真理的依赖性，若代理信号与真实进展失配可能导致错误信誉分配；②在极端噪声或持续污染环境下性能退化；③未对适应性模仿与合谋攻击进行完整测试，可能需进一步加入信誉衰减或多信号一致性校验以提升安全性。

---

## 221. Retrieval-Augmented Robots via Retrieve-Reason-Act

**arXiv ID:** 2603.02688 | [PDF](https://arxiv.org/pdf/2603.02688v1)

**作者:** Izat Temiraliev `[一作]` (University of California), Yi Zhang `[通讯]` (University of California)

**通讯引用:** 24672 | [OpenAlex ID](https://openalex.org/A5100388281)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Retrieval-Augmented Robotics（RAR）框架，利用机器人主动检索视觉装配手册并通过 Retrieve‑Reason‑Act 循环生成可执行的装配计划。

**💡 创新点**

创新点在于将信息检索从回答文本扩展到物理操作；通过视觉文档检索直接驱动机器人执行，解决了传统检索仅获取经验或约束无法应对零样本装配任务的缺口。

**🔧 技术方法**

核心技术包括多模态大语言模型 GPT‑4o、BM25 文档检索、CLIP‑FAISS 图像检索、跨模态对齐方法以及在 NVIDIA Isaac Sim 中的闭环控制实现。

**📊 数据集**

使用 IKEA Furniture Assembly 数据集（102 件家具，754 个部件，1,131 条连接关系）进行实验评估。

**📈 对比分析**

通过与零样本、封面检索、相似示例检索、完整手册检索和 Oracle（直接给定连接关系）进行对比，完整手册检索的 F1 达到 0.537，较零样本提升约 20%，Oracle 为 0.985，显示检索显著提高性能，但仍存在显著差距。

**⚠️ 局限性**

主要局限在于视觉对齐与多步指令理解不足，导致在复杂多部件场景下容易漏检连接，F1 与 Oracle 差距大；模型难以维持全局空间状态和处理视觉相似度高的部件。

---

## 222. Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection

**arXiv ID:** 2603.03095 | [PDF](https://arxiv.org/pdf/2603.03095v1)

**作者:** Sofiane Elguendouze `[一作]` (Universite Cote dAzur), Serena Villata `[通讯]` (Universite Cote dAzur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于指令调优的大型语言模型，将论证成分检测（ACD）重新定义为一次性生成任务，直接在原始文本中插入XML标签实现边界定位与类别标注；

**💡 创新点**

创新点在于：①将ACD从传统的序列标注/多阶段管道转变为统一的生成式任务；②使用紧凑指令式提示并在大规模开源LLM上进行细调；③通过生成XML结构实现边界与标签的同步决策；

**🔧 技术方法**

采用了多种开源LLM（GPT‑2‑XL‑1.5B、OPT‑1.3B/6.7B、Mistral‑7B、Llama‑3‑8B‑Instruct）以及RoBERTa、DeBERTa 等编码器，采用低温度+top‑p约束的自回归解码；

**📊 数据集**

使用了三大公开数据集——USElecDeb60To16、Persuasive Essays 和 Web Discourse，并在单一或三者合并的训练集上进行实验；

**📈 对比分析**

与传统 CRF、PE、MT‑AM 等基线相比，指令调优的 LLM 在联合检测上取得宏观 F1 最高 0.8778，几乎逼近人类上限 0.8860；在跨域合并数据上最佳模型 OPT‑6.7B 仍保持 0.7822 的良好性能；

**⚠️ 局限性**

主要局限包括：生成式模型易出现幻觉（新增或改写词语）导致对齐错误；仅关注命题与前提，未覆盖关系抽取、立场识别等子任务；可能复现或放大训练数据中的偏见；

---

## 223. Authenticated Contradictions from Desynchronized Provenance and Watermarking

**arXiv ID:** 2603.02378 | [PDF](https://arxiv.org/pdf/2603.02378v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**通讯引用:** 2624 | [OpenAlex ID](https://openalex.org/A5028326739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文设计并验证了“Integrity Clash”现象，即在同一数字图像上同时存在合法的C2PA数字签名和AI生成水印，但两者陈述相互矛盾；并提出了跨层审核协议来检测此冲突；

**💡 创新点**

创新点在于正式定义并实验验证了跨层冲突；展示了利用标准编辑流程即可制造“认证假图像”的实用攻击；提出了简单可行的联合审核机制，实现了100%检测准确率；

**🔧 技术方法**

技术手段包括：C2PA签名（使用ECDSA P-256自签名证书与DigiCert时间戳）、Pixel Seal后置式隐写水印（256位payload）、JPEG压缩/裁剪/截图等常见编辑扰动，以及基于冲突矩阵的联合判断逻辑；

**📊 数据集**

数据集为500幅使用SDXL生成的1024×1024 PNG图像，来源于Parti-Prompts基准；

**📈 对比分析**

方法对比通过四个实验管线（Baseline、Watermarked、Honest Manifest、Misleading Manifest）分别对应冲突矩阵的四象限，并在三种扰动下测试水印鲁棒性；跨层审核在所有2000个Q4b实例中达到TPR=1.000、FPR=0.000、Accuracy=1.000；

**⚠️ 局限性**

局限性包括：仅测试单一水印方法（Pixel Seal），较弱的水印在更激进扰动下可能失效；使用自签名证书导致验证工具发出发行者警告；实验仅覆盖图像模态；未评估平台级审核管线；跨层冲突若水印被抹除，则无法通过此审核方式检测。

---

## 224. EduVQA: Benchmarking AI-Generated Video Quality Assessment for Education

**arXiv ID:** 2603.03066 | [PDF](https://arxiv.org/pdf/2603.03066v1)

**作者:** Baoliang Chen `[一作]`, Xiangjie Sui `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个面向教育领域的 AI 生成视频质量评测基准（EduAIGV‑1k）并提出 EduVQA 统一评估框架，能够同时评估感知质量（空间、时间）和提示对齐（词级、句级）。

**💡 创新点**

创新点在于：①提供细粒度的空间/时间感知和词级/句级对齐标注，②设计了结构化 2D Mixture‑of‑Experts（S2D‑MoE）实现整体与子维度质量的层次依赖建模，③通过双路跨模态特征融合提升对教育内容的评估精度。

**🔧 技术方法**

主要技术包括：视频 Swin Transformer 提取感知特征，BLIP 融合视觉‑文本信息，交叉注意力机制实现跨模态互相引导，S2D‑MoE 结构化专家路由与二维门控实现多维度协同学习。

**📊 数据集**

使用了 EduAIGV‑1k 数据集：1,130 条短视频，来自 10 款文本‑到‑视频模型，覆盖 113 条教育性提示，标注空间、时间、整体感知质量以及词/句级提示对齐。

**📈 对比分析**

在 5 维度（空间、时间、整体感知、词级对齐、句级对齐）上与多种图像/视频质量评测基线及零/微调的跨模态模型对比，EduVQA 在 SRCC、PLCC、RMSE 等指标均优于现有方法，表现出更高的相关性和泛化能力。

**⚠️ 局限性**

局限性包括：①数据集中仅聚焦于基础数学概念，未覆盖更复杂抽象知识；②对 T2V 生成模型仍存在对齐与运动细节的挑战，部分高阶概念表现欠佳；③评测主要基于人工 MOS，主观性仍不可完全消除。

---

## 225. From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors

**arXiv ID:** 2603.02792 | [PDF](https://arxiv.org/pdf/2603.02792v1)

**作者:** Qi Huang `[一作]` (Leiden University), Niki van Stein `[通讯]` (Leiden University)

**通讯引用:** 1050 | [OpenAlex ID](https://openalex.org/A5003248571)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM在黑盒优化中的提示词对生成算法的影响，并提出基于基准算法的引导式进化方法BAG。

**💡 创新点**

通过AttnLRP对提示词进行token级归因，发现代码示例对LLM输出影响最大，并利用这一发现将基准算法代码注入提示词，实现搜索空间的局部引导。

**🔧 技术方法**

使用大型语言模型Gemini、GPT和Qwen的代码生成，AttnLRP归因分析，CodeBLEU相似度评估，以及(1+1)精英进化搜索框架。

**📊 数据集**

在23个伪布尔优化（pbo）和24个连续黑盒优化（bbob）任务上进行实验，基准算法来自IOHprofiler和公开的BBO算法库。

**📈 对比分析**

将BAG与EoH、LHNS、LLaMEA、MCTS-AHD、ReEvo在相同LLM查询预算下进行AUC对比，BAG在Gemini和Qwen下均位居首位，bbob上平均提升约14%，在GPT下仅次于EoH。

**⚠️ 局限性**

方法仅采用单个精英搜索，缺乏种群多样性，基准算法选择受限，且对LLM代码可执行性的依赖导致部分候选失效。

---

## 226. V3DB: Audit-on-Demand Zero-Knowledge Proofs for Verifiable Vector Search over Committed Snapshots

**arXiv ID:** 2603.03065 | [PDF](https://arxiv.org/pdf/2603.03065v1)

**作者:** Zipeng Qiu `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 650 | [OpenAlex ID](https://openalex.org/A5002684888)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了可审计的密集检索服务，在不泄露向量或索引内容的情况下，通过零知识证明验证返回的 top‑k 列表与已承诺的 IVF‑PQ 快照相符。

**💡 创新点**

创新点在于将 IVF‑PQ 查询语义标准化为固定形状的五步流程，并利用集合关系（multiset equality/inclusion）替代电路中的排序和随机访问，从而显著压缩 ZK 证明规模；同时引入基于 Merkle 树的快照承诺，支持版本化检索。

**🔧 技术方法**

采用了零知识 SNARK（Plonky2+Poseidon）、固定点编码、聚类重平衡、Merkle 路径验证、集合操作等技术。

**📊 数据集**

使用了公开基准数据集 SIFT1M、GIST1M（图像特征）和 MS MARCO passage retrieval（文本句向量）进行评估。

**📈 对比分析**

与传统浮点 IVF‑PQ、以及电路仅实现的基线相比，在检索准确率几乎无损的前提下，ZK 证明时间缩短约 3.6–22 倍，内存占用降低 30–40%，验证时间保持毫秒级。

**⚠️ 局限性**

局限性包括：只针对 IVF‑PQ，无法直接迁移到 HNSW 等图索引；固定形状的重平衡和填充在极大数据规模下可能带来构建开销；证明大小仍受电路规模影响，超大索引时仍需更高成本。

---

## 227. Why Atomicity Matters to AI/ML Infrastructure: Snapshots, Firmware Updates, and the Cost of the Forward-In-Time-Only Category Mistake

**arXiv ID:** 2603.02603 | [PDF](https://arxiv.org/pdf/2603.02603v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (Daedaelus), Paul Borrill (Daedaelus)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文针对AI/ML基础设施中的两大假设——原子检查点与原子固件部署——进行形式化分析，证明它们属于“FITO”范畴错误，指出这些假设应当通过协议收敛而非时间戳来保证，并给出改进方案（双向提交、共识驱动升级、恢复后验证）。

**💡 创新点**

创新点：1) 将FITO类别错误框架应用于分布式AI训练系统；2) 通过过程代数与层次理论证明在异步 crash‑recovery 环境下无法通过单一时间点实现原子检查点；3) 构造“epoch lattice”并证明原子性在持久化域增大时趋近零；4) 以类型系统形式化混合 epoch 恢复导致的优化错误；5) 证明原子固件升级需要不可达的公共知识，提出使用共识服务实现近似实现；6) 提出双向提交与验证机制作为实际修正方法。

**🔧 技术方法**

主要技术包括：过程代数建模、异步事件结构、故障模型与可靠性分析、层次格（epoch lattice）数学模型、类型系统与优化代数、共识与公共知识理论、以及与Open Atomic Ethernet 的协议设计。

**📊 数据集**

本研究为理论分析，未使用具体数据集或实验数据；实验验证建议可在大规模训练集上（如LLaMA、GPT‑类模型）进行。

**📈 对比分析**

由于论文为理论性研究，未给出具体实验对比或性能数值；提出的双向提交和共识升级方案预期可降低恢复延迟与提升一致性，但实际效果需在真实训练集上测评。

**⚠️ 局限性**

局限性：1) 仅考虑异步 crash‑recovery 环境，未涵盖 Byzantine 或网络分区情况；2) 对系统实现细节（如具体的日志写入、网络协议）做了简化假设；3) 通过共识实现“近似公共知识”仍存在安全窗口；4) 验证机制对梯度空间的平坦性缺乏鲁棒性。

---

## 228. Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids

**arXiv ID:** 2603.02856 | [PDF](https://arxiv.org/pdf/2603.02856v1)

**作者:** Hongjin Chen `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 3664 | [OpenAlex ID](https://openalex.org/A5102769588)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Rhythm 框架，实现双人形态机器人在现实环境中完成多种物理耦合交互行为。

**💡 创新点**

创新点包括：①通过 IAMR 解决运动重定向中的自运动与交互几何冲突；②引入基于图的奖励在 IGRL 中学习交互动力学；③实现了从仿真到真实双机器人系统的端到端迁移。

**🔧 技术方法**

使用技术包括：运动重定向优化（Laplacian 约束与自适应弹性权重）、MAPPO 强化学习、交互图与接触图奖励、点云融合定位与相对同步。

**📊 数据集**

使用的数据集为 MAGIC（约 3 小时人类对人类交互数据，提供高保真重定向参考）以及 Inter‑X 用于测试人体差异的交互场景。

**📈 对比分析**

与 GMR、OR、DOR 等重定向基线及单智能体策略对比，IAMR 在安全性（IPR=0）、保真度（IEE、F1 最高）和下游成功率上均优；IGRL 在 ISR、CSR、CER 等指标上显著高于基线；在 Unitree G1 机器人实验中，成功率超过 80%，比单智能体提升超过 60%。

**⚠️ 局限性**

局限性在于依赖预建地图进行状态估计，且目前仅验证双机器人系统，需进一步扩展至多机器人与无地图环境。

---

## 229. GPUTOK: GPU Accelerated Byte Level BPE Tokenization

**arXiv ID:** 2603.02597 | [PDF](https://arxiv.org/pdf/2603.02597v1)

**作者:** Venu Gopal Kadamba `[一作]` (New York University), Kanishkha Jaisankar `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个GPU上运行的字节级BPE分词器，严格兼容GPT‑2的贪婪合并语义；

**💡 创新点**

通过在GPU上使用cuCollections哈希表、CUB归约和轻量级双缓冲压缩，构建了可与CPU实现完全一致的BPE分词流程；

**🔧 技术方法**

采用CUDA、cuCollections、CUB、pybind11封装的Python接口；

**📊 数据集**

使用WikiText103数据集的多长度序列（256~131072个token）以及完整的《傲慢与偏见》文本进行评测；

**📈 对比分析**

与Rust tiktoken和HuggingFace GPT‑2 tokenizer比较，GPU优化核在131k token时相较tiktoken提升约1.7×，相较HF tokenizer提升约7.3×；

**⚠️ 局限性**

局限于单个GPU、单序列单块实现，短序列下仍慢，未引入GPU内存池、缺乏多批/多模型并发场景及多语言/公平性评估。

---

## 230. Federated Inference: Toward Privacy-Preserving Collaborative and Incentivized Model Serving

**arXiv ID:** 2603.02214 | [PDF](https://arxiv.org/pdf/2603.02214v1)

**作者:** Jungwon Seo `[一作]` (University of Stavanger), Jaeyeon Jang `[通讯]` (Catholic University of Korea)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5072390875)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在保持模型和数据私有的前提下，利用独立训练的模型进行推理协作的 Federated Inference (FI) 并实现了 FedSEI 参考系统。

**💡 创新点**

提出了 FI 的系统级设计空间，统一了隐私保护与协作推理的技术挑战，并给出了可验证的 FedSEI 参考架构。

**🔧 技术方法**

使用安全多方计算（Additive Secret Sharing 与 CrypTen）、加密聚合（软投票、熵加权等）以及以智能合约为核心的激励机制。

**📊 数据集**

实验使用了 CIFAR‑10/100、Fashion‑MNIST、EMNIST、PathMNIST、OrganAMNIST 等数据集，并在 LeNet、ResNet‑18 等模型上进行评估。

**📈 对比分析**

与单模型、硬投票、软投票及多种无标签加权策略对比，发现软投票与熵加权在多数非IID设置下表现最好，但在极端标签偏斜时仍不一定优于单模型；SMPC 推理导致 1–2 个数量级的延迟。

**⚠️ 局限性**

主要限制在于 SMPC 计算与网络开销高、非IID 环境下协作收益不稳定、以及基于无标签的激励机制难以实现公平、可解释的奖励分配。

---

## 231. TraceGuard: Process-Guided Firewall against Reasoning Backdoors in Large Language Models

**arXiv ID:** 2603.02436 | [PDF](https://arxiv.org/pdf/2603.02436v1)

**作者:** Zhen Guo `[一作]` (Saint Louis University), Reza Tourani `[通讯]` (Saint Louis University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5017483450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TraceGuard，一种将推理链条视为可攻击载荷的过程导向防火墙，用于检测并定位大规模推理模型中的推理后门。

**💡 创新点**

创新点在于将推理过程转化为可攻击对象，利用对比合成生成逻辑断裂数据、步骤感知监督和验证器引导强化学习，首次实现对推理链条中逻辑破裂点的精确定位和零样本泛化。

**🔧 技术方法**

技术包括自动化合成对比数据、步骤感知监督 (SSFT) 搭配 LoRA 参数微调、基于组相对策略优化 (GRPO) 的验证器引导强化学习 (VGRL)，以及对验证器输出的稠密多组件奖励设计。

**📊 数据集**

使用 CommonSenseQA、OpenBookQA、AQuA-RAT、CLUTRR、GSM8K 等多域问答数据集构建训练与评估集，其中评估集包含 334 样本的推理后门基准，覆盖提示触发、潜在后门、后置合理化与正常推理四种攻击模型。

**📈 对比分析**

在四种攻击模型上，与行业内容过滤器、Chain-of-Scrutiny (CoS) 以及零样本基线相比，4B TraceGuard 在检测 F1 上始终超过 90%，在灰盒自适应攻击下攻击成功率仅 22%，并在不同参数规模模型中展示出显著的安全加速效果。

**⚠️ 局限性**

局限性包括对合成数据质量与多样性的高度依赖、对极端简化或多步推理场景的泛化仍需验证，以及在更大规模模型推理时的延迟和资源占用仍有提升空间。

---

## 232. NeuroProlog: Multi-Task Fine-Tuning for Neurosymbolic Mathematical Reasoning via the Cocktail Effect

**arXiv ID:** 2603.02504 | [PDF](https://arxiv.org/pdf/2603.02504v1)

**作者:** Pratibha Zunjare `[一作]` (Virginia Tech), Michael Hsiao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了NeuroProlog neurosymbolic框架，利用多任务混合训练将数学知识与问题求解映射为可执行的Prolog程序，实现可验证的数学推理；

**💡 创新点**

创新点在于通过公式到规则的知识库翻译、自然语言到程序的合成以及程序答案对齐三大任务的协同训练，形成所谓的Cocktail Effect，并引入基于Prolog错误类型的执行引导解码实现零训练自修复；

**🔧 技术方法**

主要技术包括LoRA微调、统一的Prolog符号空间、多任务损失函数、基于SWI‑Prolog的错误分类和迭代修复管道；

**📊 数据集**

使用了200条手工构建的数学知识库（覆盖15+领域）与7476条GSM8K‑Prolog问题，结合310条自定义问题求解示例，形成完整的混合训练集；

**📈 对比分析**

在GSM8K基准上，四种模型（3B–32B）中，Cocktail训练提升准确率最多可达+5.54%（如GPT‑OSS‑20B从84.91%提升至88.34%），并在参数效率上优于更大规模的程序合成基线；

**⚠️ 局限性**

局限包括对高级数学知识覆盖不足、对Prolog求解器的依赖导致对模糊推理的脆弱性、以及在小于10B参数规模下的生成‑修复权衡不足。

---

## 233. Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild

**arXiv ID:** 2603.02619 | [PDF](https://arxiv.org/pdf/2603.02619v1)

**作者:** Seunguk Do `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**通讯引用:** 9354 | [OpenAlex ID](https://openalex.org/A5100611457)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

改进多视角扩散模型的姿态重建质量，通过在多姿态数据集上进行直接奖励微调，提升单视角3D人体重建的姿态自然度。

**💡 创新点**

①提出基于姿态一致性的可微奖励函数，用于衡量生成的多视角潜在图像与真实3D姿态的匹配度；②引入直接奖励微调（Direct Reward Fine‑Tuning）框架，并加入KL正则化防止奖励黑客；③构建包含1.5K多姿态与对应单视图图像的DRPose数据集，显著扩展姿态多样性。

**🔧 技术方法**

使用多视角扩散模型（Era3D、PSHuman），DRTune微调策略，基于骨架图像预测的可微奖励函数，KL散度正则化，基于MIMO生成单视图姿态图像。

**📊 数据集**

Motion‑X（AIST子集）+ MIMO生成的单视图图像，构成DRPose数据集；评测使用THuman2.1、CustomHumans以及新构建的挑战姿态基准（DRPose），并与ERA3D、PSHuman等基线进行对比。

**📈 对比分析**

在THuman2.1‑test、CustomHumans‑test和DRPose三个基准上，微调后模型在几何指标（Chamfer Distance、Normal Consistency、F‑Score）和外观指标（PSNR、SSIM、LPIPS）上均优于原始模型和其他单视角重建方法，提升幅度可观且保持一致。

**⚠️ 局限性**

需要预先分割的输入图像，若分割不精确会产生边缘漂浮几何；微调过程中GPU显存和计算开销大（需生成多张高分辨率图像并存储初始U‑Net），KL正则化也增加开销；依赖大量姿态数据集，若数据分布与目标场景差异大则效果有限。

---

## 234. Bidirectional Interpolation for the Lambda-Calculus -- Revisiting and Formalising Craig-Čubrić Interpolation

**arXiv ID:** 2603.03083 | [PDF](https://arxiv.org/pdf/2603.03083v1)

**作者:** Meven Lennon Bertrand `[一作]`, Alexis Saurin `[通讯]` (Paris Cité University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

对简单类型λ-演算（含和型）进行证明相关插值定理的重构，提出利用双向类型系统对归一形式的互递归描述，得到新的证明相关插值证明，并在Coq中实现完整形式化。

**💡 创新点**

将双向类型系统与子公式性质结合，首次用互递归的归一/中性形式显式刻画β与交换转换归一化；在此框架下重新证明Čubrić的证明相关插值定理，显著简化原始归纳论证；并首次在Coq中实现该定理与归一化证明。

**🔧 技术方法**

核心技术包括：双向类型系统、逻辑关系归一化、β与交换转换（commuting conversions）、Autosubst/Sulfur自动化替换、setoid rewriting、CSIho自动化冲突点检测、Hindley–Rosen与Newman lemma的形式化运用。

**📊 数据集**

该工作不涉及任何实验数据或数据集，全部为形式化证明。

**📈 对比分析**

与先前工作相比，代码量约4k LoC（含800 LoC Autosubst生成），编译时间约30秒，显著快于Veltri–Wan等大型形式化；大量重复目标通过自动化求解，提升可维护性与可读性。

**⚠️ 局限性**

局限性：仅对归一化（cut‑free）项给出插值；未涵盖带cut的证明；未提供统一插值或在依赖类型系统中的扩展；对自动化工具的依赖导致某些不完整或慢速的求解器情况。

---

## 235. Bridging Diffusion Guidance and Anderson Acceleration via Hopfield Dynamics

**arXiv ID:** 2603.02531 | [PDF](https://arxiv.org/pdf/2603.02531v1)

**作者:** Kwanyoung Kim `[一作]` `[通讯]` (Gwangju Institute of Science and Technology), Kwanyoung Kim (Gwangju Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无训练的注意力空间加速方法，将注意力引导建模为现代 Hopfield 网络（MHN）的固定点迭代，并通过 Anderson 加速（AA）实现快速收敛。

**💡 创新点**

创新点在于将注意力扩散引导与动态系统理论统一：将注意力动态视为固定点迭代，并引入几何感知加速（GAG），通过仅保留平行分量、抑制正交噪声，实现更稳健、高效的加速。

**🔧 技术方法**

使用的技术包括现代 Hopfield 网络、Anderson 加速、几何分解、弱收敛理论，以及现有的 CFG、APG、PAG 等引导策略。

**📊 数据集**

在 SDXL、Flux、SDXL‑DMD2、Hyper‑SDXL、SDXL‑Light 等模型上，使用 GenEval、MS‑COCO、CLIPScore、ImageReward、PickScore、HPS v2.1 等数据集和评估指标进行验证。

**📈 对比分析**

与 CFG、CFG+PAG、APG、PLADIS、NAG 等方法对比，GAG 在 50 步、4 步等设置下均显著提升 GenEval 与人类偏好指标，并兼容多步蒸馏模型，且无额外计算开销。

**⚠️ 局限性**

局限性包括：对非共享固定点的引导（如 NAG）不适用；在极高 λ 下仍需重缩放；主要关注跨注意力的加速，其他空间或网络结构可能需要进一步验证。

---

## 236. High-order Knowledge Based Network Controllability Robustness Prediction: A Hypergraph Neural Network Approach

**arXiv ID:** 2603.02265 | [PDF](https://arxiv.org/pdf/2603.02265v1)

**作者:** Shibing Mo `[一作]` (Xidian University), Jing Liu `[通讯]` (Xidian University)

**通讯引用:** 34056 | [OpenAlex ID](https://openalex.org/A5100374963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了基于双通道超图注意力网络（NCR‑HoK）对网络可控性鲁棒性曲线进行预测的方法，无需传统攻击模拟。

**💡 创新点**

创新点在于首次将高阶结构知识（通过 K‑hop 与 K‑NN 生成的超图）与图注意力网络相结合，利用双通道超图注意力机制同时学习显式拓扑、局部高阶邻域以及嵌入空间中的隐含特征，从而显著提升可控性鲁棒性预测精度。

**🔧 技术方法**

核心技术包括：节点特征编码器（利用入度、出度与 GAT 预测的介数中心性）、双通道超图注意力网络（Dual HGNN）、K‑hop/K‑NN 超图构建、SmoothL1 损失以及 Adam 优化器。

**📊 数据集**

实验使用的图数据集包括：合成网络（Erdős–Rényi、Scale‑Free、Q‑Snapback、Newman–Watts、小世界、Barabási–Albert）和真实网络（DDG、DEL、DW5、DW7、LSH、ORS），节点数从 800 到 1200，平均度数取 2、5、8、10。

**📈 对比分析**

与 PCR、iPCR、CRL‑SGNN 等基线模型在平均误差 er 与标准差 σ 上进行比较，NCR‑HoK 在大多数网络类型上实现了最低的 er 与 σ，预测曲线更平滑、稳定；在运行时间上比 PCR、iPCR 快数倍，虽然略慢于 CRL‑SGNN，但在准确率与效率之间取得了良好平衡。

**⚠️ 局限性**

局限性：目前模型仅针对静态、无属性的有向/无向网络；对动态网络、属性网络或异构网络的适用性不足；在某些真实网络（如 DW5、DW7、LSH）的结构特点下，超图构建效果不如预期。

---

## 237. TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health

**arXiv ID:** 2603.03047 | [PDF](https://arxiv.org/pdf/2603.03047v1)

**作者:** Zixin Xiong `[一作]` (Renmin University of China), Wenxuan Wang `[通讯]` (Renmin University of China)

**通讯引用:** 1916 | [OpenAlex ID](https://openalex.org/A5100755181)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TRUSTMH-BENCH，一个针对大语言模型在心理健康领域可信度的多维度评测基准；

**💡 创新点**

将临床风险管理、数字健康伦理等专业规范转化为可量化指标，构建8个核心维度（可靠性、危机识别与升级、安全、公平、隐私、鲁棒性、反附和、伦理），并首次对12个主流模型进行系统比较；

**🔧 技术方法**

基于NIST AI风险管理框架与临床伦理，采用多任务评测、Chain-of-Thought、LLM-as-a-Judge、对抗攻击、Theory-of-Mind交互、情绪识别、诊断评分等技术；

**📊 数据集**

使用USMLE-Mental、EMOBENCH、D4、SWMH、ESConv、PsyEval、PsyLeak、PsyHarm、EthicMH等公开与自建数据集；

**📈 对比分析**

采用GPT‑4.1评审、温度0的统一解码，评测12个模型在8维度下的表现：通用模型在知识与诊断上领先，但在安全、隐私、危机升级、反附和等维度不足；专属模型在情绪支持与干预对话优于通用，却在知识、鲁棒性等方面仍显不足；

**⚠️ 局限性**

未覆盖实时多轮情境下的持续评估；对抗样本生成有限；部分指标依赖评审模型，主观性较高；缺乏跨文化或多语言验证，难以推广到非中文环境。

---

## 238. Policy myopia as a mechanism of gradual disempowerment in Post-AGI governance, Circa 2049

**arXiv ID:** 2603.03267 | [PDF](https://arxiv.org/pdf/2603.03267v1)

**作者:** Subramanyam Sahoo `[一作]` `[通讯]` (University of Cambridge), Subramanyam Sahoo (University of Cambridge)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

论文探讨后AGI信息系统如何通过政策短视导致人类治理能力的不可逆丧失。

**💡 创新点**

创新点在于将政策短视视为机构性失权机制，提出三重相互强化的机制（显著性捕获、容量级联、价值锁定）并用耦合动力学模型进行定量演示。

**🔧 技术方法**

使用耦合动力学系统建模与数值仿真来分析不同领域（经济、政治、文化）中的反馈循环。

**📊 数据集**

未使用真实数据集，而是基于理论假设构建仿真参数进行数值实验。

**📈 对比分析**

与传统的可诉性和影响加权缓解方案进行对比，仿真显示这些措施只能延迟但无法阻止失权，失权时间在 25-35 年内仍会出现。

**⚠️ 局限性**

局限在于模型假设和参数选择可能无法完全捕捉真实 AGI 系统的复杂性，且缺乏实证验证。

---

## 239. Cross-view geo-localization, Image retrieval, Multiscale geometric modeling, Frequency domain enhancement

**arXiv ID:** 2603.02726 | [PDF](https://arxiv.org/pdf/2603.02726v1)

**作者:** Hongying Zhang `[一作]` (Civil Aviation University of China), ShuaiShuai Ma `[通讯]` (Civil Aviation University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 Spatial and Frequency Domain Enhancement Network (SFDE)，通过空间域与频域三分支并行融合实现跨视角地理定位。

**💡 创新点**

创新点在于同时捕捉全局语义、局部几何多尺度与频域统计稳定性，并通过自适应频域重加权、相位保留和自注意力实现跨域一致性；三分支并行架构兼顾轻量化与性能。

**🔧 技术方法**

使用 ConvNeXt‑Tiny backbone、膨胀卷积、多尺度池化、FFT 频域变换、频域自适应重加权、深度可分离卷积、自注意力及多尺度特征融合等技术。

**📊 数据集**

在公开基准 University‑1652、SUES‑200 与 Multi‑weather University‑1652 上进行实验。

**📈 对比分析**

与多种先进方法对比，SFDE 在 R@1/AP 指标上多次名列前茅，且参数量和 FLOPs 明显低于同类方法，显示出更优的效率与性能。

**⚠️ 局限性**

局限性：频域分支依赖离散 FFT，处理超高分辨率图像时效率受限；跨分支交互仅通过梯度传递，缺乏显式融合或知识蒸馏机制。

---

## 240. Breaking the Prototype Bias Loop: Confidence-Aware Federated Contrastive Learning for Highly Imbalanced Clients

**arXiv ID:** 2603.03007 | [PDF](https://arxiv.org/pdf/2603.03007v1)

**作者:** Tian-Shuang Wu `[一作]` (Hohai University), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39400 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CAFedCL的联邦对比学习框架，解决极度不平衡和客户端异构导致的原型偏差循环问题。

**💡 创新点**

创新点包括：1) 引入基于置信度的类别级权重，对原型和模型聚合进行动态加权；2) 通过几何一致性正则化保持原型空间结构；3) 可选的少数类生成增强提升极端少数类的原型估计。

**🔧 技术方法**

采用原型聚合的联邦对比学习、置信度加权聚合、几何正则化、条件GAN生成增强、预测不确定性估计和小批量监督对比损失。

**📊 数据集**

在CIFAR‑10、CIFAR‑100和EMNIST三个标准数据集上，使用多种非IID/长尾分布设置进行实验。

**📈 对比分析**

与FedAvg、FedProx、MOON、FedProto、FedRCL、FedProc、FedLC、MP‑FedCL、FedTGP等基线对比，CAFedCL在所有设置下均取得最高准确率且客户端间方差最小，表现出更好的精度和公平性。

**⚠️ 局限性**

局限性包括：对置信度估计依赖多项信号且可能在极端噪声环境下失效；在大规模联邦场景下生成器和几何正则的计算开销仍需进一步优化；未考虑差分隐私或异步通信带来的额外挑战。

---

## 241. AI Space Physics: Constitutive boundary semantics for open AI institutions

**arXiv ID:** 2603.03119 | [PDF](https://arxiv.org/pdf/2603.03119v1)

**作者:** Oleg Romanchuk `[一作]`, Roman Bondar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 AI Space Physics，一套针对自扩展 AI 机构的过渡语义和治理框架，强调在边界层面捕捉首次及第二阶影响，并将权威表面扩展正式化为需要审计的边界事件。

**💡 创新点**

创新点在于：① 将权威扩展重新定义为第二阶边界事件（STRUCTURAL_COND_T / STRUCTURAL_COND_P），并为其设定了必需的 witness、非旁路、原子化与可回放等构成性约束；② 通过“风险加权可达性”与“最优决策集合”来量化潜在影响；③ 将治理目标转移到可观测的边界操作而非内部状态推断。

**🔧 技术方法**

使用的技术包括：形式化模型（细化 Cell/Unit/Membrane 结构）、可达性与风险评估的概率/期望函数、witness 记录与可回放机制、以及对扩展诊断的抽象图（能力图）等。

**📊 数据集**

论文主要是理论性工作，未使用公开数据集，而是通过模型推理与案例场景（如结构扩展但无外部提交）来说明框架的适用性。

**📈 对比分析**

方法评估以理论证明与示例场景为主，未给出实验性性能指标；主要通过证明 P‑1、P‑1a、P‑1b、P‑1c 等定理来展示框架的可行性。

**⚠️ 局限性**

局限性包括：① 假设通道集合完整且无旁路；② 仅关注边界事件，未覆盖内部动态组合与非线性交互；③ 需要可观测的投影信息，若观察带宽不足则无法完整执行；④ 风险加权可达性计算在实际系统中可能不可解，需近似估计。

---

## 242. An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification

**arXiv ID:** 2603.02719 | [PDF](https://arxiv.org/pdf/2603.02719v1)

**作者:** L. Julián Lechuga López `[一作]` (New York University), Tim G. J. Rudner `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究对多模态 ICU 数据下的多标签临床疾病分类任务中，基于不确定性的选择性预测（Selective Prediction）进行系统评估。

**💡 创新点**

发现即使多模态融合在标准评估指标上提升了判别性能，却因类别依赖的校准失衡导致选择性预测效果显著下降，并表明校准是决定选择性预测可靠性的关键因素。

**🔧 技术方法**

采用了 MedFuse、DrFuse、MeTra 等多模态融合模型，使用二元交叉熵训练，并对不确定性量化采用预测熵；引入类别依赖的损失加权策略进行校准改进。

**📊 数据集**

使用公开的 MIMIC‑IV（结构化 EHR 时序）与 MIMIC‑CXR（胸部 X‑ray）对齐的多模态数据集，共计 25 个临床疾病标签。

**📈 对比分析**

与单模态基线（仅 EHR 或仅 CXR）相比，多模态模型在 AUROC、AUPRC 等判别指标上普遍提升，但在校准误差（ECE）和选择性 AUROC/AUPRC 上往往不如或更差；通过类别加权可略微改善校准，但对选择性预测性能提升有限。

**⚠️ 局限性**

局限性包括：仅在单一 ICU 基准上评估，未检验不同任务或模态组合的泛化；校准改进策略（损失加权）效果有限；缺乏前瞻性或临床工作流验证，无法评估实际部署中的人机协作效果。

---

## 243. A Novel Modular Cable-Driven Soft Robotic Arm with Multi-Segment Reconfigurability

**arXiv ID:** 2603.02468 | [PDF](https://arxiv.org/pdf/2603.02468v1)

**作者:** Moeen Ul Islam `[一作]` (Mississippi State University), Dong Chen `[通讯]` (Mississippi State University)

**通讯引用:** 42526 | [OpenAlex ID](https://openalex.org/A5100373698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文设计并实现了一种可模块化、缆绳驱动的柔性机械臂，允许通过简单拼接增加段数实现可伸缩的工作空间，并系统研究了硅胶材料刚度对驱动行为和承载能力的影响。

**💡 创新点**

创新点包括：① 采用双螺旋保护缆通道与可拆卸模块化结构，实现独立段控制和快速堆叠；② 通过不同硬度 Ecoflex（00-10/00-30/00-50）系统性量化刚度对工作空间、弯曲角度、纵向位移和张力的影响；③ 在三维运动捕捉环境下定量评估单/多段配置的可达空间与自重加载效应。

**🔧 技术方法**

技术主要涵盖：3D 打印模具与端盖、Ecoflex 硅胶浇铸、双螺旋缆导管、步进电机驱动与 ESP32 控制、光学运动捕捉 (OptiTrack) 以及力计测量。

**📊 数据集**

无公开数据集，实验数据基于自制单元在三种硅胶硬度及1-3段配置下的现场测量。

**📈 对比分析**

通过比较不同段数的最大径向到达 R_max、平面工作空间面积与体积，以及不同硬度在相同张力下的弯曲角度和竖向位移，实验表明：三段配置可将工作空间面积提升约13倍、体积提升近20倍；柔软硅胶实现更大弯曲角度但易发生背弯和形变，硬度越大张力需求升高、竖向位移增大、承载能力提升。

**⚠️ 局限性**

局限性包括：① 仅使用开环驱动，缺乏闭环姿态控制与反向运动学；② 结构自重与柔性导致多段叠加时弯曲性能下降，限制可堆叠段数；③ 仅在实验室光学捕捉环境验证，未测试在真实操作场景中的鲁棒性。

---

## 244. ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution

**arXiv ID:** 2603.02510 | [PDF](https://arxiv.org/pdf/2603.02510v1)

**作者:** Liu Yang `[一作]` (Yale University), Quanquan C. Liu `[通讯]` (Yale University)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5079747792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ParEVO框架，利用细化的LLM和进化式编码代理生成高性能的并行算法，尤其针对不规则数据结构；

**💡 创新点**

创新点在于：①构建Parlay-Instruct语料库并通过“Critic-Refine”实现高质量并行代码；②将LLM与高层并行原语（ParlayLib）对齐；③引入进化式编码代理（ECA）利用编译器、动态检测器和性能分析器的反馈迭代修复与加速；

**🔧 技术方法**

技术包括：大规模语言模型（DeepSeek、Qwen3、Gemini）微调、LoRA/DPO、MAP‑Elites多目标进化搜索、动态数据竞争检测、工作-跨度理论的性能分析；

**📊 数据集**

使用的主要数据集为Parlay-Instruct（13,820条并行任务）、ParEval基准、PBBSBench、RPB以及DMOJ竞赛问题集；

**📈 对比分析**

在ParEval上平均106×的速度提升（最高1103×），对比商业模型如GPT‑5‑Thinking、Gemini‑3‑Pro等；与人类专家基线相比，最大可达4.1×的加速；采用多维度评估（Build@1、Pass@1、Speedup@1）并通过MAP‑Elites维持多样性；

**⚠️ 局限性**

局限性包括：仅针对共享内存多核；进化式搜索导致推理时延高；模型在未见算法领域可能出现自信幻觉；未覆盖分布式内存与通信优化。

---

## 245. Quantifying Frontier LLM Capabilities for Container Sandbox Escape

**arXiv ID:** 2603.02277 | [PDF](https://arxiv.org/pdf/2603.02277v1)

**作者:** Rahul Marchand `[一作]` (University of Oxford), Harry Coppock `[通讯]` (UK AI Security Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于嵌套沙箱的CTF式基准，用于评估大型语言模型逃逸容器沙箱的能力。

**💡 创新点**

创新点在于安全的双层沙箱评估方法、18个覆盖编排、运行时和内核层的真实漏洞情景，以及对模型逃逸成功率与计算预算的系统分析。

**🔧 技术方法**

采用了Docker/OCI容器、Kubernetes、Vagrant虚拟机、Inspect评估框架，以及自定义工具执行模型API调用。

**📊 数据集**

使用了18个手工构造的漏洞情景（包括已公开CVE和常见配置错误），并提供公开的GitHub仓库。

**📈 对比分析**

通过在不同模型层级（GPT‑5、Opus、Claude Opus 4.5 等）下记录逃逸成功率、置信区间，并对推理计算预算进行 log‑线性缩放实验，显示前沿模型在易度 1‑2 场景中成功率>80%，难度 3‑5 场景成功率下降。

**⚠️ 局限性**

局限包括仅评估已知漏洞和配置错误，未考虑硬件侧信道、零日新漏洞，评估环境仍为实验室级别，且对更高级代理架构的评估不足。

---

## 246. Orality: A Semantic Canvas for Externalizing and Clarifying Thoughts with Speech

**arXiv ID:** 2603.02544 | [PDF](https://arxiv.org/pdf/2603.02544v1)

**作者:** Wengxi Li `[一作]` (City University of Hong Kong), Can Liu `[通讯]` (City University of Hong Kong)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5077675371)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了面向语音的思维外化与澄清工具 Orality，该系统通过实时语音识别提取关键信息，利用大型语言模型对内容进行语义分析，自动生成可交互的节点-链图语义画布，并支持通过语音指令进行结构重组、生成启发式问题、冲突检测与历史回溯，最终帮助用户在语音驱动的循环中不断澄清和深化思路。

**💡 创新点**

创新点主要体现在：① 将语音作为第一输入通道，直接从流畅的口语中抽取结构化节点；② 将 LLM 生成的语义结构嵌入可视化画布，实现语音与视觉双模的互动；③ 通过“Ask Me Questions”“Show Me Conflicts”“Thought Evolution”等 AI 辅助功能，形成自适应的思维澄清循环；④ 结合 Pirolli‑Card 感知模型提出的四层自我思维澄清框架，系统化地支持外化、结构化、深化与呈现全过程。

**🔧 技术方法**

技术栈包括：前端 React + React Flow + Material‑UI + socket.io；后端 Flask + AssemblyAI（实时语音转文本） + OpenAI GPT‑5（主题提取、问题生成、冲突检测、导出报告） + sentence‑embedding + PCA + 物理力学布局；系统通过 LLM Prompt 进行语义块化、主题聚类和节点定位。

**📊 数据集**

实验数据主要来自 12 名参与者在实验室自选主题下的语音录音和交互日志；并未使用公开的大规模文本/语音数据集，所有内容均为实验参与者生成。

**📈 对比分析**

对比方法为受试者内设计：每位参与者分别使用 Orality 与 ChatGPT+语音输入的基线，完成同一思维澄清任务。评价指标包括：思维清晰度（前后差值）、工具满意度、功能实用度（7 分量表）、NASA‑TLX 工作负荷。结果显示：① 8/12 参与者更倾向于使用 Orality 作为思维支持；② 在主观满意度、功能实用度等方面 Orality 均显著高于基线；③ 在思维清晰度和工作负荷上差异不显著，但趋势仍偏向 Orality。定性分析进一步证明 Orality 在支持非线性思维、结构迭代与冲突洞察方面具有优势。

**⚠️ 局限性**

局限性包括：① 仅在实验室环境中进行，缺乏真实工作/生活场景验证；② 主题和任务难度高度个体化，难以统一评估；③ 样本规模小（12 人），无法检验统计显著性；④ 评价指标主要为主观自评，缺乏客观思维质量量化；⑤ 仅评估单一 LLM（GPT‑5），未探讨模型多样性对结果的影响。

---

## 247. Video TokenCom: Textual Intent-Guided Multi-Rate Video Token Communications with UEP-Based Adaptive Source-Channel Coding

**arXiv ID:** 2603.02470 | [PDF](https://arxiv.org/pdf/2603.02470v1)

**作者:** Jingxuan Men `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19454 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向文本意图的多速率视频Token通信框架，该框架通过视频token化、视觉语言模型+光流产生细粒度语义掩模，并分别对意图相关和非相关token进行多速率编码和Unequal Error Protection（UEP）源-信道适配。

**💡 创新点**

创新点：
- 将文本意图与CLIP视觉语言模型结合，用光流实现跨帧语义一致性；
- 采用全码本精度与差分低码本分层编码，实现语义感知的多速率比特分配；
- 设计基于UEP的源-信道优化（MILP求解），在固定资源预算下平衡失真与延迟。

**🔧 技术方法**

使用技术：视频token化（Cosmos DV-8×16×16 / DV-4×8×8）、CLIP视觉语言模型、光流传播、token级语义掩模、差分编码、PDU封装、UEP调制/纠错、混合整数线性规划优化、BLER/BLER表、LPIPS/CLIP/SSIM/FVD等评估指标。

**📊 数据集**

实验数据集：MCL-JCV（裁剪为1024×640）和UVG（1920×1080），视频文本意图来自视频中主体的描述。

**📈 对比分析**

与VC‑DM（生成式低比特率语义通信）和传统H.265进行对比；采用PSNR、SSIM、LPIPS、FVD、CLIP相似度评估；在同等比特率或资源预算下，TokenCom在所有SNR范围内均显著优于两者，尤其在6 dB时FVD下降近1500；同时延迟与资源利用更优。

**⚠️ 局限性**

限制：
- 依赖预训练token化器与CLIP模型，模型参数巨大（247 M），计算量高（≈5.4 TFLOP/帧，65–122 ms/帧）；
- 需要文本意图输入，若意图错误或缺失，语义指导失效；
- UEP方案对码率预算与SNR敏感，极低SNR下仍可能出现失真；
- 未在极高延迟场景下评估，可能对实时应用受限。

---

## 248. ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering

**arXiv ID:** 2603.02438 | [PDF](https://arxiv.org/pdf/2603.02438v1)

**作者:** Aymen Lassoued `[一作]` (École Polytechnique de Tunisie), Yousri Kessentini `[通讯]` (Digital Research Center of Sfax)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多代理的框架ORCA，用思考代理拆分问题、路由代理选择专门子代理、协作执行、辩论验证和答案细化，实现单页文档视觉问答

**💡 创新点**

创新点在于：①通过思考代理生成可执行的推理路径；②使用路由代理动态激活不同领域专用子代理；③引入辩论与多轮讨论来验证答案；④采用答案掩码防止确认偏差；⑤模块化设计易于升级

**🔧 技术方法**

核心技术包括：大型视觉语言模型（GLM‑4.5V、Qwen3‑VL‑8B 等）作为思考与专用代理；路由代理采用约束生成+Turbo DFS解码；多代理协作与执行顺序由调度器确定；辩论、评估与判官代理基于LLM；答案细化用格式校正

**📊 数据集**

主要使用的基准数据集：Single‑Page DocVQA、InfographicsVQA、OCRBench‑v2；同时在 ChartQA 与 VQAv2 进行泛化测试

**📈 对比分析**

与多类单模VLM、基线VLM、推理增强模型对比，ORCA 在 DocVQA 上提升约 +1.1 点（相对提升 6.4 %），在 InfographicsVQA 上提升 6.4 %；在 OCRBench‑v2 及 ChartQA 上亦显著提升；整体误差率下降，性能处于最前沿

**⚠️ 局限性**

局限性包括：① 辩论与验证阶段仅在少数实例触发，导致整体增益有限；② 需要多模型部署，增加算力和延迟；③ 仍易受思考代理推理路径错误影响；④ 仅处理单页文档，跨页推理尚未覆盖

---

## 249. Graph Attention Based Prioritization of Disease Responsible Genes from Multimodal Alzheimer's Network

**arXiv ID:** 2603.02273 | [PDF](https://arxiv.org/pdf/2603.02273v1)

**作者:** Binon Teji `[一作]` (Sikkim University), Swarup Roy `[通讯]` (Tezpur University)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5010413185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

利用多模态基因表达数据（微阵列、单细胞RNA‑seq、单核RNA‑seq）以及辅助网络（蛋白互作、GO相似性、扩散相似性）构建统一的图网络，采用BERT‑风格的随机游走嵌入、变分自编码器压缩表达特征，再通过图Transformer学习注意力权重，得到NETRA分数对阿尔茨海默病相关基因进行优先排序。

**💡 创新点**

①将传统静态中心性指标替换为可学习的注意力分数；②同时融合多模态表达、网络结构与语义信息，构建端到端的多模态图Transformer；③利用BERT自监督学习捕获全局网络语义并与表达特征联合提升基因重要性评估。

**🔧 技术方法**

变分自编码器（VAE）用于多平台表达压缩；BERT‑style Transformer对随机游走序列进行无监督学习；图Transformer（Graph Transformer）实现节点注意力聚合；随机游走+MLM、图位置编码、全局网络融合与扩散集成；Python 3 + PyTorch、DGL、PyTorch‑Geometric 等深度学习框架。

**📊 数据集**

公开的微阵列数据GSE1297和GDS1979，单细胞RNA‑seq GSE129308，单核RNA‑seq scREAD AD00204；此外使用STRING蛋白互作网络、GO相似度矩阵以及多种基因调控网络推断结果（CLR、MutRank、MINE 等）共计 15 条网络，最终得到 11229 个基因的融合网络。

**📈 对比分析**

与传统网络中心性（度、介数、特征向量、PageRank）以及扩散式模型（SIR）进行 Gene Set Enrichment Analysis（GSEA）比较。NETRA 在阿尔茨海默病 KEGG 通路上的 NES ≈ 3.9，显著高于 PageRank（≈2.36）、度（≈2.08）等；SIR 模型根本未检出 AD 通路。TOP‑10 关注基因子网络连通度高，跨疾病通路（PD、HD、ALS、Prion）也均获得显著富集，表明模型具有更好的生物学可解释性和泛化性。

**⚠️ 局限性**

缺乏实验验证与外部数据集的泛化测试；模型对网络推断质量高度依赖，可能受构建网络噪声影响；计算成本相对较高，训练与推理需要 GPU；未针对其他疾病直接检验泛化性。

---

## 250. Multi-Agent Honeypot-Based Request-Response Context Dataset for Improved SQL Injection Detection Performance

**arXiv ID:** 2603.02963 | [PDF](https://arxiv.org/pdf/2603.02963v1)

**作者:** Hao Yu `[一作]` (Peking University), Bin Wang `[通讯]` (University of Waterloo)

**通讯引用:** 106342 | [OpenAlex ID](https://openalex.org/A5058772567)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于多智能体蜜罐的请求‑响应上下文数据采集框架，并用该框架构造了140,973条标注的请求‑响应对数据集，训练轻量级神经网络实现 SQL 注入检测。

**💡 创新点**

创新点在于首次将请求与服务器响应的双向语义上下文纳入检测流程，构建了完整的请求‑响应上下文数据集，并通过该数据集显著提升模型对新型和混淆 SQL 注入的泛化能力。

**🔧 技术方法**

使用技术包括：多智能体协作框架（请求生成器、数据库响应器、流量监测器）、影子数据库隔离、知识蒸馏、以及 CNN、RNN、LSTM、BiLSTM 等轻量级模型。

**📊 数据集**

使用的数据集为：140,973 条请求‑响应对的上下文数据集（Context），以及同一来源但仅包含请求负载的 Payload 数据集，两者在类别分布上保持一致。

**📈 对比分析**

通过在 2‑分类和 7‑分类任务中分别训练和评估轻量模型，比较 Context 与 Payload 两种数据集的性能，结果显示 Context 数据集提升 CNN 2‑分类准确率 41.3%、7‑分类 14.6%，BiLSTM 2‑分类 10.8%、7‑分类 7.8%；在知识蒸馏后，Context 数据集使 BiLSTM 在 7‑分类任务中提升 53.1%。

**⚠️ 局限性**

局限性包括：轻量模型在 Context 数据集上的绝对准确度仍有限；实验仅覆盖 SQL 注入，未考虑其它注入类型；蜜罐环境与真实业务场景的差异可能影响模型迁移性能。

---

## 251. NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing

**arXiv ID:** 2603.02802 | [PDF](https://arxiv.org/pdf/2603.02802v1)

**作者:** Tianlin Pan `[一作]` (Nanjing University), Chenyang Si `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种利用稀疏关键帧控制与稠密视频合成的无配对视频编辑框架 NOVA

**💡 创新点**

创新点在于将控制与合成解耦为稀疏控制分支和稠密合成分支，并通过自监督的退化模拟训练和一致性感知关键帧编辑实现无配对学习

**🔧 技术方法**

使用了基于 WAN 2.1 VACE 的 Diffusion 模型、跨注意力机制、稀疏与稠密分支、退化模拟训练策略以及 FLUX.1 Kontext 关键帧编辑模型

**📊 数据集**

训练数据来自 5,000 条高质量 Pexels 视频，并通过合成对与退化模拟生成无配对训练样本

**📈 对比分析**

与 AnyV2V、I2VEdit、LoRA-Edit、VACE、Senorita-2M 等基线在 SR、TC、FC、BG-SSIM、MS、BC 等指标上对比，NOVA 在大多数指标上优于基线且无需视频级微调

**⚠️ 局限性**

局限在于对关键帧编辑质量高度依赖，现有图像编辑模型生成高质量关键帧仍需人工迭代

---

## 252. GPR Hierarchical Synergistic Framework for Multi-Access MPQUIC in SAGINs

**arXiv ID:** 2603.02740 | [PDF](https://arxiv.org/pdf/2603.02740v1)

**作者:** Hanjian Liu `[一作]`, Jinsong Gui `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种针对UAV辅助SAGIN网络中MPQUIC的联合调度与拥塞控制框架，以解决多路径下的OFO问题和频繁切换带来的传输不稳定；

**💡 创新点**

创新点包括：1）GPR层次协同框架，实现调度与拥塞控制的协同优化；2）GPASP模块采用GradNorm概率自预测器，过滤高维噪声并自适应平衡多任务；3）PHACC算法实现主动切换感知的拥塞控制，并引入EDBSS慢启动与多指标丢包区分；4）NNPE算法通过线性估计降低神经网络推理延迟；5）RHRM机制提供鲁棒性监控与自适应重训练；

**🔧 技术方法**

技术手段主要有：深度强化学习（PPO/DRL）、自注意力编码器、GradNorm动态权重、EMAs、神经网络偏好估计、ns-3仿真平台的MPQUIC扩展；

**📊 数据集**

实验数据集为ns-3模拟数据，包含9台UE、4架UAV、LEO卫星轨道（550km）和多路径场景；

**📈 对比分析**

与传统随机、MinRTT、RR、Peekaboo、QC-MAB、OLIA、MACO等基线比较，框架在吞吐量、PDR、延迟与OFO指标上均优于对手，吞吐量提升至约12Mbps，PLR和OFO显著下降；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实部署验证；对UAV轨迹与卫星轨道的假设可能限制泛化；NNPE对ACK统计稀疏时易失效；框架对计算资源有一定需求，需进一步优化以适配边缘设备；

---

## 253. WTHaar-Net: a Hybrid Quantum-Classical Approach

**arXiv ID:** 2603.02497 | [PDF](https://arxiv.org/pdf/2603.02497v1)

**作者:** Vittorio Palladino `[一作]` (University of Illinois Chicago), Ahmet Enis Cetin `[通讯]` (University of Illinois Chicago)

**通讯引用:** 4716 | [OpenAlex ID](https://openalex.org/A5080469744)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了WTHaar-Net，一种在卷积神经网络中使用Haar小波变换的混合量子‑经典架构；

**💡 创新点**

创新点在于用空间局部多分辨率的Haar小波替代全局Hadamard混合，同时给出其在量子门中的高效实现；

**🔧 技术方法**

采用Haar小波变换、Hadamard门、受控Hadamard门、SWAP门等量子电路，以及软阈值非线性和1×1卷积；

**📊 数据集**

使用CIFAR‑10、Tiny‑ImageNet和MNIST数据集进行实验；

**📈 对比分析**

通过与Hadamard‑基线和ResNet基准对比，WTHaar‑Net在Tiny‑ImageNet上实现了约26.6%参数缩减且精度优于基准，在CIFAR‑10上保持接近原始ResNet精度；

**⚠️ 局限性**

局限性包括仅对4×4补丁的量子实现、量子测量导致的符号丢失、以及对更大图像尺寸的扩展受限。

---

## 254. Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat

**arXiv ID:** 2603.02227 | [PDF](https://arxiv.org/pdf/2603.02227v1)

**作者:** Keston Aquino-Michaels `[一作]` `[通讯]` (No Way Labs), Keston Aquino-Michaels (No Way Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究稀疏注意力门控中的路由吸收现象，展示端到端训练时门控几乎无效，而后置蒸馏可有效实现稀疏化。

**💡 创新点**

揭示“路由吸收”机制，即大模型的Q/K/V投影会吸收门控信号，使得门控学习失效；提出通过解耦门控与主网络训练来避免此现象。

**🔧 技术方法**

使用轻量级双线性门控、可微软门、硬top‑k门、梯度蒸馏与随机掩码实验；对比软门、硬门、随机门与密集注意力。

**📊 数据集**

WikiText‑103（31M模型）和Qwen3‑1.7B（55×规模）作为实验数据集。

**📈 对比分析**

与随机门、密集基线对比；在密集模型上，后置蒸馏门能达到接近oracle稀疏化效果（ppl≈37.3 vs 46.0），但端到端训练的门仅提升≈2%。

**⚠️ 局限性**

仅在31M模型完成完整端到端训练；未对大规模模型进行完整稀疏预训练；仅评估双线性门控，未探讨更复杂的路由结构。

---

## 255. Geometry-Guided Reinforcement Learning for Multi-view Consistent 3D Scene Editing

**arXiv ID:** 2603.03143 | [PDF](https://arxiv.org/pdf/2603.03143v1)

**作者:** Jiyuan Wang `[一作]` (Beijing Jiaotong University), Guosheng Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 15810 | [OpenAlex ID](https://openalex.org/A5029912845)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于强化学习的单次 3D 场景编辑框架 RL3DEdit，能够在多视角下保持一致性并实现多种编辑任务（如运动、风格、背景等）。

**💡 创新点**

①利用 3D 基础模型 VGGT 作为一致性验证器，提供几何感知奖励；②通过 RL（GRPO）直接学习 3D 一致性先验，避免缺乏 3D 标注数据；③在保持原 2D 编辑能力的前提下提升多视角一致性，实现高效单通道推理。

**🔧 技术方法**

强化学习（GRPO）、2D 视觉编辑模型 FLUX‑Kontext、3D 视觉基础模型 VGGT、LoRA 微调、3D Gaussian Splatting（3DGS）重建、SDE 探索噪声。

**📊 数据集**

从 IN2N、BlendedMVS、Mip-NeRF360 三个 3D 场景数据集中挑选 8 场景，使用 VLM 自动生成 70 条编辑指令（每场景 7–9 条），共 1,319 个 M‑view 训练样本。

**📈 对比分析**

与 DGE、EditSplat、GaussCtrl 以及在同一基底下的 EditSplat w/ FLUX‑Kontext 进行对比。指标为 VIEScore、CLIP‑dir、Ph‑Loss（多视角一致性）和编辑时间。RL3DEdit 在所有指标上均优于对手，VIEScore 5.48、Ph‑Loss 0.076，且单次推理时间仅 1.5 min（比最慢的 40 min 快 2.7×）。

**⚠️ 局限性**

受限于 2D 编辑器的注意力长度导致视角数与分辨率折衷；训练规模受限于 GRPO 的高计算开销；当 2D 基础模型能力受限时，整体性能亦受限。

---

## 256. SEP-YOLO: Fourier-Domain Feature Representation for Transparent Object Instance Segmentation

**arXiv ID:** 2603.02648 | [PDF](https://arxiv.org/pdf/2603.02648v1)

**作者:** Fengming Zhang `[一作]` (Jiangnan University), Jianchao Huang `[通讯]` (Jiangnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SEP-YOLO 框架，用于透明物体实例分割，并补充了 Trans10K 的高质量实例标注。

**💡 创新点**

创新点包括：① 频域细节增强模块（FDDEM），利用可学习的复数权重对频域高频边界进行强化；② 多尺度门控细化块（MS-GRB），通过多尺度卷积和门控机制恢复细粒度边界；③ 内容感知对齐颈部（CA²-Neck），结合线性可变形卷积（LDConv）和动态上采样（DySample）实现特征对齐与边界保留；④ 仅比 YOLO11 轻量 0.23M 参数即可实现 SOTA 性能。

**🔧 技术方法**

技术手段包括：傅里叶变换（FFT/IFFT）进行频域增强；多分支可学习复数权重；多尺度门控卷积（MS-DWConv + CGLU）；线性可变形卷积和动态采样；基于 YOLO11 的轻量化检测骨干；PyTorch 实现。

**📊 数据集**

使用了 Trans10K（9,491 张含玻璃表面和玻璃器皿的实例分割标注）和 GVD（2,416 张实验室透明器材图像）两个公开数据集。

**📈 对比分析**

与 Mask R‑CNN、Solov2、YOLOv10n、YOLO11n、TrInSeg、Hyper‑YOLO‑N、Mamba‑YOLO‑T、YOLOv12n 等八种先进方法对比，SEP‑YOLO 在 Trans10K 上 Box mAP50 0.852、Mask mAP50 0.851，GVD 上 Box mAP50 0.882、Mask mAP50 0.872，均为最高，推理速度约 88 FPS，参数仅比 YOLO11n 增加 0.23M。

**⚠️ 局限性**

局限性在于：目前仅在两类透明物体（玻璃表面、玻璃器皿）和两个数据集上验证，缺乏对更复杂形状或非刚体透明物体的评估；依赖 YOLO11 作为骨干，进一步扩展到更大规模或多类别场景仍待验证。

---

## 257. Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain

**arXiv ID:** 2603.02218 | [PDF](https://arxiv.org/pdf/2603.02218v1)

**作者:** Wei Liu `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 13695 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在自演化循环中的三角角色（提问者、求解者、验证者），并提出将自演化视为可学习信息管道的框架，设计了异构共进化、容量增长和主动信息寻求三大机制来实现持续进化。

**💡 创新点**

创新点在于：①用可观测信息学（epiplexity）量化可学习信息；②提出三条系统级设计原则，确保可学习信息在迭代中单调递增；③将异构共进化与容量与信息寻求结合，形成完整的自演化流水线。

**🔧 技术方法**

技术手段包括：预序MDL与epiplexity估计、基于强化学习的自演化训练、三角角色协同生成与验证、参数与推理预算动态扩展、外部信息主动检索与上下文转换。

**📊 数据集**

使用的数据集主要是自制的代码生成与推理任务（归纳、诱导、演绎），通过模型自合成生成训练样本；未使用公开大规模数据集，而是依赖内部合成数据和外部检索上下文。

**📈 对比分析**

比较方法：将不同容量的提问者和求解者在可学习信息（epiplexity）上的表现进行对比，观察容量与信息质量的关系；实验显示若不采用共进化和容量扩展，信息会出现波动并导致性能崩溃，证明三大机制的重要性。

**⚠️ 局限性**

limitations: ①共进化机制目前主要适用于易验证任务，对难验证任务的通用性不足；②可学习信息与最终任务准确度不完全对应，无法替代传统性能指标；③主动信息寻求需要模型识别缺失知识并主动检索，仍是难点且未完全实现。

---

## 258. Infinite dimensional generative sensing

**arXiv ID:** 2603.03196 | [PDF](https://arxiv.org/pdf/2603.03196v1)

**作者:** Paolo Angella `[一作]` (University of Genoa), Matteo Santacesaria `[通讯]` (University of Genoa)

**通讯引用:** 175 | [OpenAlex ID](https://openalex.org/A5003334194)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在无穷维希尔伯特空间中使用深度生成模型进行压缩感知重建的理论框架，并给出了对应的采样与重建算法。

**💡 创新点**

创新点在于：1）引入无穷维局部互易度（local coherence）并据此推导最优采样分布；2）推广限制等距性质为Gen‑RIP，证明重建误差仅与生成模型的内在维度有关；3）展示低分辨率生成器在欠采样下的隐式正则化效果；4）验证理论在连续物理问题（Darcy流）上的适用性。

**🔧 技术方法**

使用技术包括：ReLU 激活的通用生成网络（Generalized Generative Network）、变量密度采样与权重矩阵、基于局部互易度的采样分布、功能自编码器（FAE）、高斯混合模型作为潜在空间正则化、傅里叶测量算子以及矩阵 Bernstein 估计等。

**📊 数据集**

实验使用的数据库是公开的 Darcy 流动数据集，包含不同分辨率（32×32、64×64、128×128）的压力场样本。

**📈 对比分析**

与均匀采样、传统稀疏压缩感知以及不使用正则化的恢复方法相比，基于局部互易度的自适应采样显著减少所需测量数；在极度欠采样情形下，低分辨率生成器能进一步提升重建精度；整体重建误差随采样率提升趋于一致，并在高采样率下优于传统方法。

**⚠️ 局限性**

局限性包括：1）理论假设生成网络使用分段线性激活（ReLU），对光滑激活（如GeLU、SiLU）的推广尚未完成；2）测量算子仅考虑幺正傅里叶变换，缺乏对非幺正算子（如Radon变换）的分析；3）实验主要在无噪声场景，噪声鲁棒性未充分验证；4）对采样分布的可行性依赖于对局部互易度的估计，实际应用中可能需要额外计算开销。

---

## 259. Incremental Graph Construction Enables Robust Spectral Clustering of Texts

**arXiv ID:** 2603.03056 | [PDF](https://arxiv.org/pdf/2603.03056v1)

**作者:** Marko Pranjić `[一作]` (Jožef Stefan Institute and International Postgraduate School), Marko Robnik-Šikonja `[通讯]` (University of Ljubljana)

**通讯引用:** 6605 | [OpenAlex ID](https://openalex.org/A5020021079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种增量k‑NN图构造方法，保证邻接图连通，应用于文本嵌入的谱聚类。

**💡 创新点**

通过只使用局部k‑NN搜索即可确保连通，无需全局信息或额外连通性约束，且支持在线增量更新。

**🔧 技术方法**

使用增量k‑NN算法、余弦距离、SentenceTransformer生成的句子嵌入、拉普拉斯谱映射以及k‑means/谱聚类。

**📊 数据集**

Massive Text Embedding Benchmark（MTEB）六个数据集（ArXiv、BioRxiv、MedRxiv、Reddit、StackExchange、20Newsgroups）及其S2S/P2P版本。

**📈 对比分析**

与标准k‑NN图在不同k值下对比，低k时增量图显著提升V‑measure；在高k时性能相当；与高维K‑means作上限对比，谱聚类在低维下已逼近其性能。

**⚠️ 局限性**

对节点顺序敏感，初始节点可能导致弱连接；增量图不受MST提升；未针对非度量距离的近似k‑NN探讨；缺乏在流式数据场景下的实证。

---

## 260. ITO: Images and Texts as One via Synergizing Multiple Alignment and Training-Time Fusion

**arXiv ID:** 2603.02767 | [PDF](https://arxiv.org/pdf/2603.02767v1)

**作者:** HanZpeng Liu `[一作]` (Huazhong University of Science and Technology), Kun He `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5033526822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的图像-文本对比预训练框架 ITO，通过多模态多重对齐和训练时融合模块来提升表示的统一性和鲁棒性。

**💡 创新点**

创新点在于将多重对齐与训练时轻量融合相结合，融合模块仅在训练期间起结构正则化作用，训练结束后可直接恢复双编码器高效推理。

**🔧 技术方法**

采用 CLIP 方式的对比学习、两视角图像增强、文本多视图扩展、多模态多重对齐、轻量 Transformer 融合模块以及损失组合等技术。

**📊 数据集**

使用了从几百万到十亿规模的公开数据集：Conceptual Captions 3M/12M、YFCC15M、Laion100M、DataComp-1B 等。

**📈 对比分析**

与 CLIP、SLIP、SigLIP、FLAIR 等基线在零样本分类、线性分类、图文检索和多模态 LLM 评测中均实现了显著提升，零样本分类平均提升约 2–3%，检索 Recall@1 提升 1–2 个百分点。

**⚠️ 局限性**

局限性包括对融合模块权重 λ 的调优需要经验，且在极大规模数据时多重对齐的收益有限；另外融合仅在训练期间使用，未验证其对跨任务迁移的全面性。

---

## 261. Utonia: Toward One Encoder for All Point Clouds

**arXiv ID:** 2603.03283 | [PDF](https://arxiv.org/pdf/2603.03283v1)

**作者:** Yujia Zhang `[一作]`, Hengshuang Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个可在室内、室外、远程感知、对象级 CAD、视频提升等多域点云上联合预训练的单一 Point Transformer V3 编码器（Utonia）。

**💡 创新点**

创新点在于三项域无关设计：1) 通过随机遮蔽（causal modality blinding）避免对颜色/法线等可选模态的过度依赖；2) 对坐标进行感知细粒度重标定（perceptual granularity rescale），消除尺度与采样模式导致的域偏差；3) 在已对齐细粒度坐标上使用 RoPE 编码，使注意力仅依赖连续相对几何，提升跨域可迁移性。

**🔧 技术方法**

核心技术包括：Point Transformer V3 backbone；自监督对比学习与自蒸馏；RoPE 旋转位置编码；随机模态遮蔽、坐标重标定与尺度/旋转增强；大规模数据增广与多域混合训练。

**📊 数据集**

使用了多域数据集：室内 ScanNet、S3DIS；室外 LiDAR Waymo、nuScenes；遥感点云；对象级 CAD（ModelNet40、ShapeNetPart、PartNetE）；视频提升点云；并扩展至 250k 交叉域样本 + 1M 额外 CAD 资产。

**📈 对比分析**

与 Sonata、Concerto 等基线在室内/室外语义分割、对象分类/部件分割、机器人抓取、3D 视觉推理等任务进行线性探针、解码器探针、全微调评估。Utonia 在多数任务中达成或逼近 SOTA，显著提升对缺失颜色/法线的鲁棒性，并在跨域检索和姿态不变任务上表现出更好的一致性。

**⚠️ 局限性**

局限性：1) 线性探针对部件分割的可解释性不足，需更灵活的任务解码器；2) 仍依赖稀疏卷积，内存/部署效率受限；3) 只处理静态点云，缺乏时序一致性与运动感知；4) 对极端噪声或极稀疏数据的泛化尚待验证。

---

## 262. SaFeR-ToolKit: Structured Reasoning via Virtual Tool Calling for Multimodal Safety

**arXiv ID:** 2603.02635 | [PDF](https://arxiv.org/pdf/2603.02635v1)

**作者:** Zixuan Xu `[一作]` (Huazhong University of Science and Technology), Zhigang Zeng `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 26200 | [OpenAlex ID](https://openalex.org/A5081245089)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SaFeR‑Toolkit，通过在视觉‑语言模型中引入结构化的虚拟工具调用（Perception→Reasoning→Decision）实现可审计的安全决策，并设计三阶段训练流程（SFT→DPO→GRPO）来提升工具使用质量。

**💡 创新点**

创新点在于把安全判断转化为可检查、可追踪的工具调用链；将安全决策过程结构化为类型化的工具轨迹；并通过层次化工具集与约束图实现多模态安全门控与自我纠错。

**🔧 技术方法**

使用的大型语言模型（Qwen2.5‑VL 3B/7B），虚拟工具调用框架、SFT（监督微调）、DPO（对抗偏好优化）、GRPO（基于奖励的策略优化）以及自定义奖励（格式、深度、语义+工具质量）等技术。

**📊 数据集**

构建了首个工具化安全推理数据集 31,654 条样本（6k SFT、18.6k DPO、6k GRPO），并保留 1k 评估集，采样自 BeaverTails‑V、JailBreakV‑28k 等安全与常规推理数据源。

**📈 对比分析**

与现有安全基线（ECSO、SIA、TIS、VLGuard、SPA‑VL、SaFeR‑VLM）在多项安全与通用指标上比较，SaFeR‑Toolkit 在 7B 规模下安全评分 86.34%、帮助度 80.79%、推理严谨度 85.34%，显著优于所有对手，并且通用能力仅下降 <1%。

**⚠️ 局限性**

局限性包括：仍需人工设计工具集与拓扑；工具调用链在极端对抗场景下可能被绕过；模型仍依赖内部工具日志，未实现完全透明的端到端安全性；以及在更大规模模型或不同视觉语言框架中的迁移效果待验证。

---

## 263. An Investigation Into Various Approaches For Bengali Long-Form Speech Transcription and Bengali Speaker Diarization

**arXiv ID:** 2603.03158 | [PDF](https://arxiv.org/pdf/2603.03158v1)

**作者:** Epshita Jahan `[一作]` (Bangladesh University of Engineering and Technology), Tafsir Al Nafin `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了针对孟加拉语的多阶段长音频转录与说话人分离系统，结合分段、语音识别与后处理实现端到端流畅工作流。

**💡 创新点**

创新点包括：①两步推理的说话人分离策略（先估计最大说话人数，再固定人数进行第二次推理）；②对分段模型进行任务特定微调，显著提升边界检测；③针对Whisper的孟加拉语微调模型与自定义分段结合，优化长音频处理；④使用算法级重复消除与说话人段落合并的后处理步骤。

**🔧 技术方法**

技术栈：Whisper（孟加拉语 fine‑tuned）、pyannote‑community‑1、DEMUCS（噪声抑制）、Silero VAD、定制分段网络、两步推理框架、LLM（Qwen）辅助纠错、后处理重复移除与段落合并。

**📊 数据集**

使用 Kaggle “DL Sprint 4.0 Bengali Long‑Form Speech Recognition” 与 “Bengali Speaker Diarization” 公开/私有测试集（小时级录音及伴随转录/说话人标签），并结合外部多语种数据（Hindi、Wav2Vec2、IndicWav2Vec）用于实验对比。

**📈 对比分析**

在公开评测集上，最终 DER 下降至 0.19（私有 0.27），WER 降至 0.37；相较基线（DER 0.40，WER 0.38）分别提升约 50% 与 5%。系统在两步推理与自定义分段上表现最突出，显著抑制了错误分配与重复生成。

**⚠️ 局限性**

局限性：仅微调分段模块，未对说话人嵌入模型进行任务特定训练；对 ASR 的微调实验未获提升；数据量有限，模型对多样化口音、噪声场景的泛化仍待验证；隐私与偏见问题需进一步审查。

---

## 264. Density-Guided Response Optimization: Community-Grounded Alignment via Implicit Acceptance Signals

**arXiv ID:** 2603.03242 | [PDF](https://arxiv.org/pdf/2603.03242v1)

**作者:** Patrick Gerard `[一作]` (Information Sciences Institute), Svitlana Volkova `[通讯]` (Aptima Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用社区在上线平台上对内容的接受与拒绝行为，构造隐式偏好信号，对语言模型进行对齐。

**💡 创新点**

首次证明社区接受行为在表示空间中形成局部高密度结构，可通过密度估计直接获得偏好信号，从而实现无需人工标注的对齐。

**🔧 技术方法**

核心技术包括局部核密度估计（kNN+RBF核）、直接偏好优化（DPO）、句子编码器(all‑mpnet‑base‑v2）以及大型语言模型（Pythia‑2.8B、GPT‑5‑nano 等）做评判。

**📊 数据集**

实验数据来源于 Stanford Human Preferences (SHP) 的 Reddit 子社区、三种平台（Reddit、Twitter、专用论坛）的食物障碍支持社区数据，以及俄语 VK 论坛的冲突文档社区。

**📈 对比分析**

通过与随机、kNN、全局密度、监督奖励模型等基线对比；局部密度在 SHP 上获得 58–72% 的 pairwise accuracy，接近监督模型；DGRO 在无标注社区的头对头评估中获胜率在 55–80% 以上，优于基线。

**⚠️ 局限性**

局部密度对稀疏或无结构社区效果不佳，可能放大已有偏见或被恶意操纵；在价值冲突或极度敏感的社区中需要外部治理和人工干预。

---

## 265. APRES: An Agentic Paper Revision and Evaluation System

**arXiv ID:** 2603.03142 | [PDF](https://arxiv.org/pdf/2603.03142v1)

**作者:** Bingchen Zhao `[一作]`, Yoram Bachrach `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了一种基于 LLM 的两阶段框架，先用 agentic search 自动发现能预测未来影响力（即 12 个月后引用数）的评审 Rubric，然后利用该 Rubric 作为目标函数，让 LLM 对论文文本进行可解释的增删改写，以提升论文可读性和预测影响力。

**💡 创新点**

创新点在于：① 将评审 Rubric 的发现与闭环自动改写结合，形成真正意义上的“发现→指导→改写”闭环；② 采用 MultiAIDE agentic search 在 LLM 中迭代优化 Rubric，而不是直接使用固定的人工准则；③ 用 LLM 生成的 Rubric 作为客观的“影响力代理”，并通过 diff‑based 编辑保证不改变实验结果。

**🔧 技术方法**

核心技术包括：LLM 代理（Rubric Proposer、Reviewer、Rewriter）与负二项回归模型；agentic search（MultiAIDE）与 AIDE 的扩展；diff‑based 编辑模板；以及多模态评估（MAE、Human preference、ΔS）。

**📊 数据集**

实验数据来自公开的 ICLR 2024/25 与 NeurIPS 2023/24 论文及其审稿文本（OpenReview），以及 Semantic Scholar 的“influential citation”计数。

**📈 对比分析**

与人类评分、平均引用、SPECTER‑MLP、SPECTER‑PCA、Prompt‑breeder 等基线对比，Rubric‑search 的 MAE 低于 2.0（人类评分 5.3、SPECTER‑MLP 2.8），并且改写后论文被 79% 的评审者优选；ΔS 在 “borderline” 与 “reject” 论文上均显著提升。

**⚠️ 局限性**

局限性包括：仅处理文本，忽略图表；改写过程虽限定但仍可能无意改动技术细节；引用数作为影响力代理受领域、语言和社交媒体等偏差；易受对抗攻击；且在深层技术问题上改写效果有限。

---

## 266. Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals

**arXiv ID:** 2603.03258 | [PDF](https://arxiv.org/pdf/2603.03258v1)

**作者:** Achyutha Menon `[一作]` (University of California San Diego), Diogo Cruz `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在股票交易与急诊分诊模拟环境中，评估现代大型语言模型在面临目标漂移（goal drift）时的表现，特别是通过情境条件化引发的漂移；

**💡 创新点**

首次系统比较最新模型在标准与条件化环境下的漂移抗性，发现GPT‑5系列在强制条件化仍能保持一致性；

**🔧 技术方法**

利用对抗性压力、目标切换与情境条件化实验，采用对话生成与决策逻辑的语言模型；

**📊 数据集**

使用基于公开股票交易模拟的数据集和急诊分诊队列数据；

**📈 对比分析**

通过自定义漂移度量（0–1）与标准误差比较，发现大多数新模型在原始设置下漂移率低，但在条件化情境中多模型漂移显著，GPT‑5.1表现最佳；

**⚠️ 局限性**

实验仅限于二元目标、特定模拟环境及有限模型集，未涵盖更复杂多目标决策场景。

---

## 267. Any Resolution Any Geometry: From Multi-View To Multi-Patch

**arXiv ID:** 2603.03026 | [PDF](https://arxiv.org/pdf/2603.03026v1)

**作者:** Wenqing Cui `[一作]` (King Abdullah University of Science and Technology), Peter Wonka `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 15005 | [OpenAlex ID](https://openalex.org/A5076768552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Ultra Resolution Geometry Transformer（URGT），通过将单张高分辨率图像拆分为多补丁，并联合使用粗糙的深度和法线先验，在单前向推断中同时预测高分辨率深度图和法线图。

**💡 创新点**

创新点包括：① 将多视图 Transformer 转化为多补丁 Transformer，利用跨补丁注意力实现全局一致性；② 引入 GridMix 随机补丁网格采样策略，增强跨补丁一致性与泛化；③ 采用全局 RoPE 位置编码，使补丁间的空间关系更精确；④ 统一深度-法线监督，利用伪法线约束两者几何一致。

**🔧 技术方法**

技术方法：多补丁 Transformer（intra‑patch + cross‑patch attention），DINOv2 视觉与几何 token 编码，RoPE 全局位置编码，GridMix 补丁采样，DPT‑style 输出头，统一的深度与法线损失（MSE + 梯度 + 角度+MSE）。

**📊 数据集**

使用的数据集：UnrealStereo4K、Booster、ETH3D、Middlebury 2014，并在 8K 野生图像上做零射手评估。

**📈 对比分析**

与 PatchRefiner、PatchFusion、PRO、DepthAnything V2、Metric3D V2 等基线比较。URGT 在 UnrealStereo4K 上达到 AbsRel 0.0291、RMSE 1.31、δ1 0.983，较 PatchRefiner 降低 AbsRel 49% 与 RMSE 35%。在零射手 Booster/ETH3D/Middlebury 上也获得最佳 AbsRel/δ1。8K 图像推理保持细节与全局一致，推理时间约 0.94–0.97 s/4K。

**⚠️ 局限性**

局限性：需要先验的粗糙深度/法线预测，受限于预训练模型；GridMix 与多补丁计算开销较大；对极端遮挡、稀疏光照等情况的鲁棒性尚未深入验证；模型参数较多，推理成本高于单纯深度或法线单任务方法。

---

## 268. The Perceptual Gap: Why We Need Accessible XAI for Assistive Technologies

**arXiv ID:** 2603.02486 | [PDF](https://arxiv.org/pdf/2603.02486v1)

**作者:** Shadab H. Choudhury `[一作]` `[通讯]` (University of Maryland, Baltimore County), Shadab H. Choudhury (University of Maryland, Baltimore County)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对可解释人工智能（XAI）在感官障碍用户中的应用现状进行综述，指出其存在的感知鸿沟并提出提升可访问性的研究方向。

**💡 创新点**

首次系统性聚焦视觉XAI与感官障碍用户之间的差距，提出可验证性、语言化解释与多模态输入等改进路线，强调用户参与训练过程的重要性。

**🔧 技术方法**

综述性研究，未使用新的算法或模型，主要参考已发表的XAI方法（如Grad‑CAM、SHAP、LIME等）和辅助技术。

**📊 数据集**

未使用具体公开数据集，而是分析了已有的研究案例与应用（如Seeing AI、Lookout、Otter等），并讨论了VizWiz等包含残障参与的数据集。

**📈 对比分析**

未进行实验性性能对比；文中通过文献回顾与案例分析指出现有XAI方法在感官障碍用户中的可用性不足，呼吁后续实验评估。

**⚠️ 局限性**

局限在于缺乏针对感官障碍用户的实证验证与数据集支持，研究重点集中于视觉XAI，音频与多模态XAI的探讨尚不足够。

---

## 269. Estimating Visual Attribute Effects in Advertising from Observational Data: A Deepfake-Informed Double Machine Learning Approach

**arXiv ID:** 2603.02359 | [PDF](https://arxiv.org/pdf/2603.02359v1)

**作者:** Yizhi Liu `[一作]` (University of Maryland), Siva Viswanathan `[通讯]` (University of Maryland)

**通讯引用:** 15505 | [OpenAlex ID](https://openalex.org/A5019502700)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新框架DICE‑DML，用深度伪造（deepfake）生成的图像对来学习视觉特征中的治疗属性（如肤色）与混杂因素的分离，从而实现对视觉广告中治疗属性因果效应的估计。

**💡 创新点**

创新点在于引入基于深度伪造的弱监督机制与差分对抗学习以及正交投影，专门解决视觉治疗泄漏（treatment leakage）问题，确保双重机器学习（DML）中的控制变量不包含治疗信息。

**🔧 技术方法**

核心技术包括：生成深度伪造图像对、基于差分向量的对抗学习（DICE‑Diff）、正交投影去除治疗轴、以及在此基础上的双重机器学习与交叉拟合。

**📊 数据集**

使用了来自Instagram的232,089条影响者帖子数据，并在此数据上生成了对应的肤色深度伪造对，主要用于评估肤色对帖子点赞数的因果效应。

**📈 对比分析**

与标准DML和OLS相比，DICE‑DML在模拟实验中将均方根误差降低73–97%，在真实数据中将诊断指标（R²）从-0.003提升至0.626，并将估计效应从-1,455点赞（显著）缩减至-522点赞（边缘显著），显示出显著的精度与可信度提升。

**⚠️ 局限性**

局限性包括：需要可操纵的治疗属性才能生成深度伪造对；对未观测混杂因素的敏感性未完全解决；仅在Instagram平台和肤色治疗属性上验证，泛化性需进一步检验；以及对复杂视觉特征的处理仍有提升空间。

---

## 270. CUCo: An Agentic Framework for Compute and Communication Co-design

**arXiv ID:** 2603.02376 | [PDF](https://arxiv.org/pdf/2603.02376v1)

**作者:** Bodun Hu `[一作]` (UT Austin), Aditya Akella `[通讯]` (UT Austin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了CUCo，一个自动化生成融合计算与通信的CUDA核的agentic框架。

**💡 创新点**

创新点在于将计算与通信协同设计转化为结构化搜索空间，并结合快速正确性核和慢速性能演化两阶段Agent，自动探索设备侧通信的优化空间。

**🔧 技术方法**

使用LLM驱动的代码生成、演化搜索、结构化设计空间定义、NVSHMEM/NCCL Device API以及CUDA Cooperative Groups等技术。

**📊 数据集**

使用四个多GPU工作负载：Flash Attention、DeepSeek‑V3 MoE、KV‑Cache Transfer、GEMM+AllGather（基于模拟模型，无公开数据集）。

**📈 对比分析**

与传统主机驱动NCCL实现对比，CUCo在所有工作负载上实现了5.3%–26.2%（最高1.57×）的延迟缩短，验证了显著的性能提升。

**⚠️ 局限性**

局限性包括对LLM与评估预算的依赖、对设备侧通信API兼容性的限制，以及在极端硬件拓扑或不常见通信模式下可能难以达到最优。

---

## 271. Engineering Reasoning and Instruction (ERI) Benchmark: A Large Taxonomy-driven Dataset for Foundation Models and Agents

**arXiv ID:** 2603.02239 | [PDF](https://arxiv.org/pdf/2603.02239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 272. MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing

**arXiv ID:** 2603.02885 | [PDF](https://arxiv.org/pdf/2603.02885v1)

**作者:** Chunyu Xue `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14383 | [OpenAlex ID](https://openalex.org/A5039318240)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向多租户数据中心的多任务参数高效微调（PEFT）系统，能够通过空间-时间骨干共享实现多任务并发执行

**💡 创新点**

创新点在于统一的PEFT模块化骨干共享、层级化空间-时间任务融合、以及针对任务、算子与数据的三层协同调度方案

**🔧 技术方法**

采用分层任务融合（空间/时间多路复用）、双层混合并行（流水线+张量并行）、算子级子图调度、水平适配器融合、块级序列对齐等技术

**📊 数据集**

在四种主流LLM（GPT‑3.2.7B、LLaMA‑2‑7/13B、OPT‑30B）上使用三类PEFT（LoRA、Adapter Tuning、Diff Pruning）并结合三种数据集（SST2、OpenBookQA、RTE）进行评测

**📈 对比分析**

与HF‑PEFT、NeMo、SLoRA‑PEFT三大基线对比，实验显示在A40和H100 GPU上多任务吞吐可提升最高2.33×、5.29×内存节省，整体性能显著优于对比方案

**⚠️ 局限性**

局限性包括：依赖统一骨干模型；对不同模型类型的多任务协同调度尚未深入；系统对极大规模任务数或极端序列长度的鲁棒性待进一步验证

---

## 273. Talking with Verifiers: Automatic Specification Generation for Neural Network Verification

**arXiv ID:** 2603.02235 | [PDF](https://arxiv.org/pdf/2603.02235v1)

**作者:** Yizhak Y. Elboher `[一作]` (Hebrew University of Jerusalem), Jan Křetínský `[通讯]` (Masaryk University)

**通讯引用:** 2136 | [OpenAlex ID](https://openalex.org/A5074485601)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种集成框架，能将自然语言规范自动转换为可被现有深度学习网络验证工具处理的数值约束，支持结构化表格、图像及音频三种输入；

**💡 创新点**

核心创新是把大语言模型与开源视听检测模型耦合，先解析自然语言获得语义对象与操作，再通过对象定位得到具体坐标，最终生成标准的局部稳健性（或其他）验证公式，无需改动验证器本身；

**🔧 技术方法**

使用 Gemini 3 Flash / GPT‑5 Mini 进行语义解析；Grounding DINO（视觉）或待实现的开源声音事件检测模型进行语义定位；随后由指定生成器把坐标与操作映射成输入/输出约束；

**📊 数据集**

实验使用 Statlog（德国信用数据）构建的全连接网络以及 CUB‑200‑2011 鸟类细粒度分类的 ResNet‑18；

**📈 对比分析**

在表格数据上，语义解析准确率 98‑100%，推理时间 1.07 s；在图像数据上，单一检测配置下 55% 的准确率，所有配置组合下可达 83%；整体验证流程在保持语义一致性的前提下，计算开销低，验证器可直接接收生成的规范；

**⚠️ 局限性**

局限性包括：检测阶段对细粒度定位的准确率仍低，尤其是单配置下仅 55%；音频域的定位模型尚未实现；需人工干预确认定位；并且仅支持局部规范，无法覆盖全局性质的验证。

---

## 274. Generalized Per-Agent Advantage Estimation for Multi-Agent Policy Optimization

**arXiv ID:** 2603.02654 | [PDF](https://arxiv.org/pdf/2603.02654v1)

**作者:** Seongmin Kim `[一作]` (KAIST), Youngchul Sung `[通讯]` (KAIST)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Generalized Per-Agent Advantage Estimator（GPAE），在 CTDE 框架下为每个智能体提供 n‑step 的优势估计，并结合双重截断重要性采样（DT‑ISR）实现离线样本复用。

**💡 创新点**

创新点包括：① 通过新的价值迭代算子实现显式 per‑agent 信号；② 证明 GPAE 在 λ=1 时保持策略不变性；③ 设计双重截断权重，兼顾个体信用和团队方差控制；④ 在一次性算法中统一处理 on‑policy 与 off‑policy。

**🔧 技术方法**

核心技术包括：分布式 CTDE、GAE 的多智能体推广、离线重要性采样（DT‑ISR）、基于 V‑trace 思路的双截断、PPO 风格的策略更新、集中式 critic 训练。

**📊 数据集**

使用 SMAX（StarCraft Multi‑Agent Challenge）离散任务与 MABrax（连续控制）两大基准集进行评估。

**📈 对比分析**

与 MAPPO、DAE、COMA、QMIX、VDN 等基线对比，GPAE 在所有 SMAX 任务中均取得最高 win‑rate，在 MABrax 任务中表现最佳，且在离线样本复用时提升样本效率和最终性能，PPO 风格训练保持稳定。

**⚠️ 局限性**

局限性包括：① 仍假设训练阶段可观测全局状态；② 对大规模智能体或高度非平稳团队动态的可扩展性尚未完全验证；③ 需要额外的短期 replay buffer，略增训练开销；④ 目前未在完全离线或部分可观测训练设置下进行深入评估。

---

## 275. COLREGs Compliant Collision Avoidance and Grounding Prevention for Autonomous Marine Navigation

**arXiv ID:** 2603.02484 | [PDF](https://arxiv.org/pdf/2603.02484v1)

**作者:** Mayur S. Patil `[一作]` (Texas A&M University), Prabhakar R. Pagilla `[通讯]` (Texas A&M University)

**通讯引用:** 3081 | [OpenAlex ID](https://openalex.org/A5032194507)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的MASS实时运动规划方法，能同时实现碰撞避免、COLREG规则遵从和停泊预防。

**💡 创新点**

将COLREG方向约束、鲁棒Velocity Obstacle碰撞约束以及通过ILP圆形逼近得到的可凸化海底浅水区约束整合进单一凸优化框架。

**🔧 技术方法**

采用基于凸二次规划的速度空间优化、鲁棒VO、整数线性规划（ILP）凸化浅水区、以及对位置与速度不确定性的保守扩张。

**📊 数据集**

利用IMAZU基准测试集中的多船对撞、交叉及随机生成的非凸浅水区域进行仿真验证。

**📈 对比分析**

与传统基于VO、MPC或采样规划方法对比，本文方法在10 Hz更新率下平均QP求解时间0.7 ms，最差不到6 ms，保持超过550 m的安全距离，兼顾规则合规与动态约束，表现出显著的实时性能和鲁棒性。

**⚠️ 局限性**

局限在于仅验证单船与小规模多船场景，未覆盖大规模协同决策，浅水区逼近仍有约束误差，且未与更高层路由与感知模块集成。

---

## 276. Semi-Supervised Few-Shot Adaptation of Vision-Language Models

**arXiv ID:** 2603.02959 | [PDF](https://arxiv.org/pdf/2603.02959v1)

**作者:** Julio Silva-Rodríguez `[一作]` (ETH Zurich), Ender Konukoglu `[通讯]` (ETH Zurich)

**通讯引用:** 13683 | [OpenAlex ID](https://openalex.org/A5036970822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了半监督少样本适配方法SS-Text-U，利用无标签数据通过文本驱动伪标签传播，提升医学视觉‑语言模型的少样本适配效果。

**💡 创新点**

创新点在于将块坐标最小化与Optimal Transport相结合，对伪标签施加分布一致性约束，从而在极低样本情形下有效利用无标签数据，同时保持训练免费且计算量极低。

**🔧 技术方法**

采用文本驱动线性探针、块坐标最小化、Sinkhorn‑Knopp算法求Optimal Transport、文本先验正则化以及伪标签分布校正等技术。

**📊 数据集**

在12个医学影像数据集（组织学、眼底、胸部X‑ray等）以及3个专用医学VLM（CONCH、FLAIR、CONVIRT）上进行实验。

**📈 对比分析**

与梯度线性探针、训练免费方法等多种基线对比，SS‑Text‑U在K=1‑16的低样本设置下平均提升2.7%–10.9% ACA，标注成本降低约50%–75%，且计算速度显著快于梯度方法。

**⚠️ 局限性**

在极低样本下需要手动修正未出现类别的分布，部分任务（如mBRSET）仍表现有限，且方法依赖文本先验与特征嵌入质量。

---

## 277. Rethinking Time Series Domain Generalization via Structure-Stratified Calibration

**arXiv ID:** 2603.02756 | [PDF](https://arxiv.org/pdf/2603.02756v1)

**作者:** Jinyang Li `[一作]` (Guangzhou Institute of Technology), Jinbo Sun `[通讯]` (Guangzhou Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出结构分层校准框架 SSCF，先基于频谱特征将时序样本分为结构相容子集，再在每个子集内对幅值进行校准，以提升跨域泛化性能。

**💡 创新点**

创新点在于用结构可比性评估替代全局对齐，先分层识别结构相容子集后进行局部幅值校准，避免跨结构误配导致的负迁移。

**🔧 技术方法**

使用频谱表示、功率谱聚类实现结构分层，构造均方幅值（MAS）锚点，进行幅值缩放校准；两阶段训练并保持校准过程可微分。

**📊 数据集**

在19个公开数据集（睡眠分期、心律失常检测、人体活动识别）上评估，约10万样本，覆盖多种设备和采集条件。

**📈 对比分析**

与 IRM、MMD、CORAL、SleepDG 等多种基线在 Leave‑One‑Domain‑Out 零射击设置下对比，SSCF 在 Macro‑F1 上平均提升 7–10%，显著优于全局对齐和数据集锚点方法。

**⚠️ 局限性**

局限性包括对结构分层粗粒度的依赖，跨结构差异过大时提升有限；对非平稳动力学、在线迁移适应的能力尚未充分验证。

---

## 278. Design, Modeling and Direction Control of a Wire-Driven Robotic Fish Based on a 2-DoF Crank-Slider Mechanism

**arXiv ID:** 2603.02851 | [PDF](https://arxiv.org/pdf/2603.02851v1)

**作者:** Yita Wang `[一作]` (University of Tokyo), Moju Zhao `[通讯]` (University of Tokyo)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5045076994)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于2-DoF曲柄滑块机的线驱动仿生鱼，实现推进与转向分离，完成前进、转向与方向控制实验。

**💡 创新点**

通过扩展曲柄滑块机实现推进与舵角分离的双自由度驱动，解耦速度与转向，提供更高速度与灵活转向；提出对应的动力学模型与前馈+反馈控制方案。

**🔧 技术方法**

机械设计（弹性骨架、双自由度曲柄滑块机、耐水结构）、动力学建模（Lagrange法、流体阻力）、控制理论（前馈解耦+PID方向控制）、实验验证（运动捕捉、IMU、无线通信）。

**📊 数据集**

主要使用实验测得的运动轨迹、速度、转向角、马达负载等数据；未使用公开数据集。

**📈 对比分析**

与传统单自由度/电机驱动仿生鱼相比，该系统在对称模式下可达0.32 m/s（约0.64 BL/s）速度，转向半径0.56 BL，控制误差约0.15°，展示了解耦驱动带来的高速与精准转向。

**⚠️ 局限性**

高频（>6 Hz）性能未验证；需要更强马达；控制仅使用比例项，存在稳态误差；未探索更复杂运动模式。

---

## 279. Neural Electromagnetic Fields for High-Resolution Material Parameter Reconstruction

**arXiv ID:** 2603.02582 | [PDF](https://arxiv.org/pdf/2603.02582v1)

**作者:** Zhe Chen `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 14939 | [OpenAlex ID](https://openalex.org/A5050651525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 NEMF 框架，利用多模态视觉和射频 CSI 数据实现对室内场景电磁材料属性的非侵入式物理反演，从而生成功能性可模拟的数字孪生。

**💡 创新点**

提出了通过先恢复高精度几何先验，再解算环境射频场，最后利用物理监督的可微反射层实现材料参数反演的系统化解耦方法。

**🔧 技术方法**

采用 instant-ngp 训练 Signed Distance Function 获得几何先验，使用 hash grid + MLP 的 Radio Map 网络预测射频场，构建可微 Fresnel 反射层，并在 PyTorch 上训练。

**📊 数据集**

在三套高保真合成室内场景（Office、Bedroom、Conference Room）上生成多视图图像和稀疏 CSI，采样 8 频点。

**📈 对比分析**

与单一黑盒 MLP 基线对比，NEMF 在 ε_r 和 σ 的相对误差分别降低约 7 倍和 2 倍，显著提升材料映射精度。

**⚠️ 局限性**

受限于几何先验精度与真实环境噪声，模型对法向量误差敏感，并且仅在合成数据上验证，需进一步评估真实测量场景。

---

## 280. CAWM-Mamba: A unified model for infrared-visible image fusion and compound adverse weather restoration

**arXiv ID:** 2603.02560 | [PDF](https://arxiv.org/pdf/2603.02560v1)

**作者:** Huichun Liu `[一作]` (Foshan University), Haishu Tan `[通讯]` (Foshan University)

**通讯引用:** 1376 | [OpenAlex ID](https://openalex.org/A5066389939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端的多模态图像融合与复合恶劣天气恢复框架 CAWM-Mamba，能够同时处理红外与可见光图像的融合与降雨、雾霾、雪等多种天气干扰的恢复。

**💡 创新点**

创新点在于将天气感知预处理模块 (WAPM)、跨模态特征交互模块 (CFIM) 与小波空间状态块 (WSSB) 三大模块联合起来，并设计了频率选择状态空间模型 (Freq-SSM) 与统一降解空间表示 (CDSM)，首次实现对复合天气条件下的融合与恢复的统一端到端处理。

**🔧 技术方法**

主要技术包括基于 Mamba 的线性状态空间模型、双向小波分解、频率选择 SSM（针对高频方向性噪声）以及天气嵌入驱动的特征增强与交互。

**📊 数据集**

使用了 AWMM‑100K 复合天气数据集进行训练与评估，并在 LLVIP、MSRS、M3FD 三个标准融合数据集上验证模型在无恶劣天气下的泛化能力。

**📈 对比分析**

与多种单/复合天气融合方法（如 AWFusion、FusionBooster、MaeFuse 等）对比，CAWM‑Mamba 在所有评测指标（Q_MI、Q_NICE、Q_G、SSIM、mAP、mIoU 等）上均排名第一，平均排名接近 1，并在下游语义分割和目标检测任务中获得最高分。

**⚠️ 局限性**

在极端多重干扰（如同时出现雾霾、雨、雪）时可能仍留有残余噪声；且当前实验仅限单帧图像，未对连续视频序列的时序一致性进行验证。

---

## 281. Instant and Reversible Adhesive-free Bonding Between Silicones and Glossy Papers for Soft Robotics

**arXiv ID:** 2603.02500 | [PDF](https://arxiv.org/pdf/2603.02500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 282. Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement

**arXiv ID:** 2603.02910 | [PDF](https://arxiv.org/pdf/2603.02910v1)

**作者:** Hao Ai `[一作]` (University of Birmingham), Ofek Eyal `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于交互视频的无先验关节知识的 Articulation in Motion（AiM）框架，能够对复杂的可动物体进行高质量的几何重建、部件分割与关节动力学估计。

**💡 创新点**

创新点包括：① 双高斯场景表示实现静态与动态几何的自适应分离；② 通过无结构先验的 sequential RANSAC 对移动高斯轨迹进行聚类，自动恢复多部件数量与关节参数；③ 利用单一连续视频而非仅两状态输入，提升了在开闭状态变化中的鲁棒性。

**🔧 技术方法**

技术手段包括 3D Gaussian splatting、双高斯（静态+可变）表示、基于时间编码的变形 MLP、静态-动态检测（SDMD）以及 Kabsch+RANSAC 的运动聚类与关节推理。

**📊 数据集**

实验使用 PartNet-Mobility 数据集中的多部件可动物体，采用多视角 RGB 视频及起始静态扫描作为输入。

**📈 对比分析**

与 DTA、ArtGS、PARIS 等两状态方法对比，AiM 在部件 IoU、Chamfer Distance、关节角度误差等指标上均显著优于现有方法，尤其在复杂多部件场景中提升约 20–30% 的 IoU，并将关节角度误差压至 1° 以下。

**⚠️ 局限性**

局限性在于只能重建视频中可见的几何，无法补全隐藏内部部件；对完整运动视频的依赖意味着在缺少足够交互信息时性能会下降。

---

## 283. MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection

**arXiv ID:** 2603.03101 | [PDF](https://arxiv.org/pdf/2603.03101v1)

**作者:** Jun Yeong Park `[一作]` (Yonsei University), Yu Rang Park `[通讯]` (Yonsei University)

**通讯引用:** 3326 | [OpenAlex ID](https://openalex.org/A5046176363)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出 MoECLIP，一种在零样本异常检测（ZSAD）中对 CLIP 视觉编码器进行动态、补丁级别适配的混合专家（Mixture-of-Experts）框架。

**💡 创新点**

核心创新点包括：① 利用动态路由为每个图像补丁分配专属 LoRA 专家，实现补丁级别的定制化适配；② 通过冻结正交特征分离（FOFS）在输入阶段强制专家专注不同子空间，减少功能冗余；③ 在输出阶段使用等角紧致框架（ETF）损失，使专家输出形成最大化等角分离的向量，进一步提升专家差异化。

**🔧 技术方法**

技术手段包括：CLIP 视觉编码器的参数冻结、LoRA 低秩适配、Mixture-of-Experts 与路由器、FOFS 与 ETF 损失、Patch‑Average‑Aggregation（PAA）多尺度上下文聚合、深度可分离卷积（Depth‑wise Adapter）以及多种损失函数（Focal、Dice、BCE、ETF、balance）。

**📊 数据集**

在 14 个工业与医学基准数据集上进行评估，工业数据集包括 MVTec-AD、VisA、BTAD、RSDD、DTD‑Synthetic；医学数据集包括 Brain MRI、Head CT、Liver CT、Retina OCT、ColonDB、ClinicDB、CVC‑300、Endo、Kvasir。

**📈 对比分析**

与 WinCLIP、April‑GAN、AnomalyCLIP、AdaCLIP、AA‑CLIP、Bayes‑PFL 等最新 SOTA 方法在统一训练设定下进行对比。MoECLIP 在图像级别 AUROC/AP 提升约 3%/2% 以上，在像素级别 AUROC/AP 上分别提升约 1%/1.7%，显著优于其它方法，证明动态补丁级专家设计有效提升零样本异常检测性能。

**⚠️ 局限性**

局限性主要体现在：① 需要额外的专家数目与路由器参数，计算与显存开销比单一适配器略大；② 目前实验仅在 14 个固定数据集上验证，未探究在更大多样化场景下的鲁棒性；③ 对专家数量的选择存在折中，过多可能导致功能冗余，过少可能不足以覆盖补丁多样性；④ 仍需使用辅助数据集进行训练，完全无监督的零样本设置未被完全实现。

---

## 284. Shatter Throughput Ceilings: Leveraging Reflection Surfaces to Enhance Transmissions for Vehicular Fast Data Exchange

**arXiv ID:** 2603.02752 | [PDF](https://arxiv.org/pdf/2603.02752v1)

**作者:** Qianyao Ren `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**通讯引用:** 24299 | [OpenAlex ID](https://openalex.org/A5016290340)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种基于专用镜面反射面（DSRS）的反射增强传输框架（RETF），通过动态虚拟化、状态切换、旋转协同等机制显著提升车辆单链路吞吐量。

**💡 创新点**

创新点在于引入非智能、低成本的DSRS提供多方向反射路径，并结合动态虚拟化和旋转协同团队，实现不干扰其他用户的前提下突破单车吞吐上限。

**🔧 技术方法**

采用三维几何光路模型、3GPP TR 38.901 级联 MIMO 信道模型、状态切换机制、虚拟反射面分组（RPP‑DGV）、贪心搜索算法、旋转协同团队（RCT）与交替邻居选择（ANS）等技术。

**📊 数据集**

利用基于3GPP TR 38.901 的系统级仿真平台，随机生成车辆、用户分布和信道参数进行验证，未使用公开真实数据集。

**📈 对比分析**

通过系统级仿真与传统5G单链路方案对比，评估信噪比、信道秩与SE，结果显示中心区信道秩从1.38提升至2.25，边缘区SE提升70%以上，单车吞吐量可达约1.2Gbps，其他用户SE损失低于3%。

**⚠️ 局限性**

局限性包括低频多径下反射功率仍有限、旋转时延与同步要求、SUs分布稀疏时最优但高密度用户下仅次优，以及对实时CSI获取与时延的依赖。

---

## 285. Rethinking Code Similarity for Automated Algorithm Design with LLMs

**arXiv ID:** 2603.02787 | [PDF](https://arxiv.org/pdf/2603.02787v1)

**作者:** Rui Zhang `[一作]` (City University of Hong Kong), Zhichao Lu `[通讯]` (City University of Hong Kong)

**通讯引用:** 3456 | [OpenAlex ID](https://openalex.org/A5027687139)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于问题求解轨迹的行为相似度度量，用以评估和分析LLM自动算法设计中生成的算法的相似性。

**💡 创新点**

创新点在于将算法的中间解序列作为行为特征，使用动态时间规整（DTW）衡量轨迹相似度，从而区分表面相似但逻辑不同的算法。

**🔧 技术方法**

采用了动态时间规整、编辑距离/欧氏距离归一化、以及对LLM‑AAD框架（FunSearch、EoH）的多岛屿与辅助目标整合技术。

**📊 数据集**

使用了公开的算法相似度基准数据集（包含四类类型的算法对）以及三大自动算法设计任务（ASP、TSP、CPP）的实例。

**📈 对比分析**

与现有代码相似度指标（Token、AST、CodeBLEU、CodeBERTScore、执行结果等）比较，BehaveSim在区分四类样本的行为相似/不同方面取得 1.0/低分；在 FunSearch+BehaveSim 与 EoH+BehaveSim 上的性能明显优于原始方法，提升了搜索效率与最终解质量。

**⚠️ 局限性**

局限性包括只关注问题求解行为，忽略时间/空间复杂度等维度；以及对离散/连续解类型的距离度量需根据任务手工设定，且对高度随机化算法需要多次采样或固定随机种子。

---

## 286. On the Structural Limitations of Weight-Based Neural Adaptation and the Role of Reversible Behavioral Learning

**arXiv ID:** 2603.02934 | [PDF](https://arxiv.org/pdf/2603.02934v1)

**作者:** Pardhu Sri Rushi Varma Konduru `[一作]` `[通讯]` (Malla Reddy University), Pardhu Sri Rushi Varma Konduru (Malla Reddy University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

研究神经网络适配中的结构不可逆性，提出可逆行为学习框架并通过实验验证可逆适配可实现精确回滚，无法在共享权重上恢复原始行为

**💡 创新点**

提出结构不可逆性概念、可逆行为学习（RLAE）框架、可恢复性因子（RF）和结构方差分析（SVAR）等指标，强调适配的结构性可回溯性

**🔧 技术方法**

使用权重更新与可拆卸行为模块的对比实验，计算KL/JS分歧、RF、ILS、SVAR等评估指标

**📊 数据集**

实验基于Qwen2.5-1.5B和3B模型，使用自定义提示集合进行适配与评估（无公开数据集）

**📈 对比分析**

与传统基于共享权重的微调对比，权重变异后RF=0，分歧始终>0；RLAE后RF≈1，KL/JS降至数值精度，证明可逆适配实现精确回滚

**⚠️ 局限性**

仅在受限的短期适配和固定提示分布下验证，未考察长期RL、能力增长、多行为协调、分布漂移等场景，且RLAE不保证行为正确性或内部遗忘

---

## 287. LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model

**arXiv ID:** 2603.02528 | [PDF](https://arxiv.org/pdf/2603.02528v1)

**作者:** Xiangyu Li `[一作]` (University of Texas at Austin), Christian Claudel `[通讯]` (University of Texas at Austin)

**通讯引用:** 2400 | [OpenAlex ID](https://openalex.org/A5082935103)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个基于LLM的多层特征融合网络，用于自动驾驶行为分类。

**💡 创新点**

创新点在于将大语言模型生成的语义描述与多尺度数值特征融合，并通过双通道注意力实现更鲁棒、可解释的分类。

**🔧 技术方法**

采用多级特征提取（统计、行为、动态）、LLM（GPT‑4o）生成文本描述、RoBERTa文本编码、卷积+时空注意力、多尺度卷积、双通道融合与MLP分类等技术。

**📊 数据集**

使用Waymo开放轨迹数据集，包含约2695条约20秒、0.1s采样间隔的速度/加速度/加速度变化序列。

**📈 对比分析**

与9种传统与Transformer时序分类模型对比，准确率最高94.4%，在准确率、精确率、召回率与F1上均优于基线。

**⚠️ 局限性**

仍依赖离线LLM推理，推理时延和算力开销较大；对极端交通场景鲁棒性待验证；缺少实时部署与多传感器融合研究。

---

## 288. DSBA: Dynamic Stealthy Backdoor Attack with Collaborative Optimization in Self-Supervised Learning

**arXiv ID:** 2603.02849 | [PDF](https://arxiv.org/pdf/2603.02849v1)

**作者:** Jiayao Wang `[一作]` (Yangzhou University), Dongfang Zhao `[通讯]` (University of Washington)

**通讯引用:** 1342 | [OpenAlex ID](https://openalex.org/A5101671477)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种动态隐蔽后门攻击DSBA，利用协同优化实现对SSL预训练编码器的隐蔽植入。

**💡 创新点**

创新点在于将全局后门编码器与样本级动态触发器生成分层协同优化，兼顾攻击成功率、视觉与分布层面隐蔽性。

**🔧 技术方法**

采用协同优化框架、对比学习损失、多尺度感知损失、分布对齐损失、动态触发器生成器以及自适应权重调度等技术。

**📊 数据集**

实验使用CIFAR-10、STL-10、GTSRB、SVHN、TinyImageNet等五个公开数据集。

**📈 对比分析**

与IMPERATIVE、GhostEncoder、WaNet、CTRL等SOTA方法对比，DSBA在ASR、BA、SSIM、PSNR等指标上均显著提升，攻击成功率近100%，隐蔽性最佳。

**⚠️ 局限性**

局限性包括较高的预训练成本、对特定SSL框架的适配性有限，以及在极端高强度防御下仍存在被检测的可能。

---

## 289. Uni-Skill: Building Self-Evolving Skill Repository for Generalizable Robotic Manipulation

**arXiv ID:** 2603.02623 | [PDF](https://arxiv.org/pdf/2603.02623v1)

**作者:** Senwei Xie `[一作]` (Chinese Academy of Sciences), Xilin Chen `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 35436 | [OpenAlex ID](https://openalex.org/A5083420537)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一个统一的 Skill‑centric 框架 Uni‑Skill，支持在任务规划时自动检测缺失的技能并扩充新的技能库，同时利用大量未标注的机器人视频自动构建 VerbNet‑风格的层级技能仓库，实现零样本任务执行。

**💡 创新点**

创新点包括：① 通过“skill‑aware planning”机制实现任务驱动的技能自适应扩展；② 通过“automatic skill evolution”实现无人工介入的技能检索与实现；③ 建立多层次的技能层级结构（VerbNet 基础上四层），将海量视频示例映射到可检索的技能片段；④ 将检索到的技能片段与 VLM 提供的语义约束和轨迹信息结合，实现少量示例即可实现新技能。

**🔧 技术方法**

技术手段主要包括：大规模 Vision‑Language 模型（Gemini‑2.0‑Flash、GPT‑4o、CLIP）用于视频分割、技能描述、时间对齐和轨迹生成；代码化策略规划器（与 VLM 共享输出格式）用于生成可执行的 API 调用序列；层级检索算法将请求的技能映射到技能树各层；以及基于 3D 轨迹和姿态映射的轨迹补全与执行。

**📊 数据集**

使用的数据集包括：机器人视频集 DROID（350 小时，10,000+ 技能片段）；RLBench（模拟任务 8+10 个，评估零样本性能）；真实世界 Franka Emika 机器人实验（8 类任务）。

**📈 对比分析**

与现有两大类方法比较：Code‑as‑Policies (CaP) 和 MOKA。Uni‑Skill 在 8 个基本技能任务的平均成功率达 73%（CaP 39%，MOKA 39%）；在 10 个超出基本技能的任务上，Uni‑Skill 的零样本成功率比 MOKA 提升 31%，在真实世界长周期任务上提升 20–34%。实验表明，Uni‑Skill 在多样性任务和轨迹要求上均优于基线。

**⚠️ 局限性**

局限性包括：① 仍需依赖 VLM 的视觉理解和文本推理，可能在复杂视觉场景下出现误识；② 技能层级与机械属性耦合不足，导致某些任务因机械约束未被捕获而失败；③ 自动标注管道对视频质量敏感，低质量或人类中心视频的技能片段需要进一步筛选；④ 需要进一步改进自我纠错机制，以降低规划与执行误差。

---

## 290. Structured vs. Unstructured Pruning: An Exponential Gap

**arXiv ID:** 2603.02234 | [PDF](https://arxiv.org/pdf/2603.02234v1)

**作者:** Davide Ferré `[一作]` (Université Côte d'Azur), Frederik Mallmann-Trenn `[通讯]` (King's College London)

**通讯引用:** 636 | [OpenAlex ID](https://openalex.org/A5045186573)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了强彩票票假设（SLTH）下的神经元剪枝，探讨了如何使用随机初始化的两层ReLU网络来近似单个无偏ReLU神经元，揭示了神经元剪枝的内在限制。

**💡 创新点**

创新点在于证明了神经元剪枝需要Ω(d/ε)个隐藏神经元才能成功近似目标ReLU神经元，而权重剪枝只需O(dlog(1/ε))个神经元，从而在近似理论上建立了两者之间的指数分离。

**🔧 技术方法**

使用了随机初始化的两层ReLU网络，并采用了新的证明策略，通过跟踪隐藏单元的非线性位置来处理无偏设置。

**📊 数据集**

没有具体提到使用的数据集，主要是理论分析。

**📈 对比分析**

与现有的权重剪枝方法进行比较，结果表明神经元剪枝在近似能力上显著较弱，且在无偏设置下，神经元剪枝的性能依赖于隐藏神经元的数量。

**⚠️ 局限性**

限制在于该研究主要集中在无偏ReLU神经元的近似上，尚未探讨在其他激活函数或更深层次架构中的表现。

---

## 291. Characterizing and Predicting Wildfire Evacuation Behavior: A Dual-Stage ML Approach

**arXiv ID:** 2603.02223 | [PDF](https://arxiv.org/pdf/2603.02223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 292. Virtual-Memory Assisted Buffer Management In Tiered Memory

**arXiv ID:** 2603.03271 | [PDF](https://arxiv.org/pdf/2603.03271v1)

**作者:** Yeasir Rayhan `[一作]` (Purdue University), Walid G. Aref `[通讯]` (Purdue University)

**通讯引用:** 9845 | [OpenAlex ID](https://openalex.org/A5000123743)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多层内存体系结构中提出了一种 n 级虚拟内存辅助缓冲池，支持 DRAM、远程内存和磁盘三层。

**💡 创新点**

创新点在于保持页的虚拟地址不变，利用操作系统的页面迁移系统调用和自定义的 move_pages2 实现大批量页迁移，从而大幅提升多层内存缓冲池性能。

**🔧 技术方法**

采用 Linux 内核页面迁移接口（mbind、move_pages、move_pages2）、自定义的迁移模式、批量迁移与优化错误处理等技术。

**📊 数据集**

使用 TPC‑C 事务负载和随机读键值工作负载，数据规模约 190 GB 和 130 GB。

**📈 对比分析**

与原生 move_pages、mbind 的基线进行对比，使用 32 个线程跑在 NUMA 节点上；在远程内存容量为本地内存 2 倍或 4 倍时，TPC‑C 通过率可提升 1.67×~3.82×，随机读工作负载提升 1.36×，且 move_pages2 在批量迁移方面显著优于原生接口。

**⚠️ 局限性**

限制在于仅能使用 System‑DRAM 模式的远程内存，DAX/App Direct 模式无法支持；且迁移开销仍是主瓶颈，且在写密集负载下增大批量迁移并不显著受益。

---

## 293. OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation

**arXiv ID:** 2603.02999 | [PDF](https://arxiv.org/pdf/2603.02999v1)

**作者:** Dekai Sun `[一作]` (Tencent Inc), Jun Zhang `[通讯]` (Tencent Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 OneRanker，一个统一生成与排序的端到端模型，解决生成阶段兴趣与业务价值对齐、生成过程目标无感知、生成与排序层不一致等问题；

**💡 创新点**

创新点包括：①价值感知多任务解耦架构，使用任务标记与因果掩码分离兴趣与价值空间；②粗细协作目标感知机制，利用 Fake Item Token 实现生成过程的目标感知并通过排名解码器细化；③输入输出双侧一致性保证，通过 Key/Value 传递和分布一致性（DC）损失实现生成与排序的闭环协同优化；

**🔧 技术方法**

技术手段涵盖：Transformer 生成解码器（HSTU/GPR 变体）、多任务解码器、Fake Item Token 聚类、因果掩码与异构注意力、双通道表示融合、排名解码器、分布一致性损失、K-means 聚类、BPR 对比损失、KL 对齐；

**📊 数据集**

使用腾讯自有的微信渠道广告数据集，包含数亿活跃用户、数千万广告，训练与评估均基于真实业务日志；

**📈 对比分析**

与行业强基线 HSTU、GPR 对比，离线评测 Hit Ratio 指标：OneRanker 在 HR@1、HR@5、HR@15 分别提升约 44.7%、34.9%、11.3%；在线 A/B 测试中 GMV-Normal 提升 1.34%，成本提升 0.72%；

**⚠️ 局限性**

局限性包括：对大规模集群环境依赖较高，模型训练与推理成本高；Fake Item Token 需要预先聚类，可能对细粒度目标感知有限；DC 损失调参复杂，且仅在候选集层面对齐，未完全消除所有阶段间误差；

---

## 294. Revealing Positive and Negative Role Models to Help People Make Good Decisions

**arXiv ID:** 2603.02495 | [PDF](https://arxiv.org/pdf/2603.02495v1)

**作者:** Avrim Blum `[一作]` (Toyota Technological Institute at Chicago), Jingyan Wang `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5031857164)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在有限披露预算下，社交规划者如何揭示正负榜样标签以最大化代理人模仿正向榜样的社会福利，并提出代理福利函数与公平性保证。

**💡 创新点**

创新点在于引入代理福利函数恢复子模性并实现常数因子近似；在分组与干预场景给出公平性与干预收益理论；提供学习设置下的样本复杂度保证。

**🔧 技术方法**

使用了贪心算法、子模/超模分析、代理福利函数设计、预算化干预、几何双边图构造与学习理论等技术。

**📊 数据集**

实验使用了 Adult、Student Performance（Mathematics 与 Portuguese）以及 Garment Workers Productivity 四个真实数据集生成几何双边图。

**📈 对比分析**

与随机、启发式、全量基准对比，贪心在大多数场景下接近最优；干预在低连通度时显著提升；学习设置下训练与测试表现相近。

**⚠️ 局限性**

局限性包括仅考虑无权重边、确定性图；在极低连通或代理人负邻居数较多时子模性失效，理论保证弱；未涉及策略性分类或随机模型扩展。

---

## 295. PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems

**arXiv ID:** 2603.03054 | [PDF](https://arxiv.org/pdf/2603.03054v1)

**作者:** Sudip Bhujel `[一作]` `[通讯]` (University of Kentucky), Sudip Bhujel (University of Kentucky)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一套端到端的差分隐私强化学习人类反馈（DP‑RLHF）框架，用于在保持正式的（ε,δ）隐私保证下对医疗对话语言模型进行微调、奖励模型学习和PPO策略优化；

**💡 创新点**

创新点包括：①首次在整个RLHF流程（SFT、奖励模型、PPO）中实现DP‑SGD，确保任何单条对话记录对最终模型参数的影响被严格限制；②提出无人工标注的专家vs非专家偏好构造方法，通过对比医生回复与生成的非专家回答生成偏好对；③在PPO阶段使用已训练好的DP奖励模型并冻结，减少额外隐私开销；

**🔧 技术方法**

技术手段主要有：DP‑SGD（带“ghost clipping”）对SFT、奖励模型和PPO进行梯度噪声注入；LoRA参数高效微调；PPO与KL正则化；对偏好对使用对数似然的Bradley‑Terry损失；RDP会计器用于累计隐私预算；

**📊 数据集**

使用公开的 MedDialog（已去标识化）作为训练与偏好对来源，公开或合成的对话提示用于PPO；在评估时使用保留的测试拆分及 PubMedQA 医学问答数据；

**📈 对比分析**

评估方式包括自动指标（ROUGE‑L、BERTScore、实体 F1、PPL）、安全性启发式检查、LLM‑Jury（3‑模型 G‑EVAL）综合评分以及 Membership‑Inference 攻击（6种方法）与 Canary 检测。实验结果显示，ε=7 的 DP‑RLHF 在 ROUGE‑L 0.156、实体 F1 0.103、Hallucination 1.4%–3.0%、Harmful advice ≤0.8%，并在MIA AUC 0.51–0.56 处接近随机，优于单纯 DP‑SFT，并且在 LLM‑Jury 上获得最高总体分 2.86。

**⚠️ 局限性**

局限性包括：① DP‑SGD 需要逐样本梯度裁剪与噪声注入，导致训练成本与时延上升；② 无标注的偏好构造依赖代理生成，仍可能存在质量波动；③ 评估主要基于基准数据，缺乏真实临床流量与多模态输入的验证；④ 隐私实验为有限样本的经验评估，无法排除未来更强攻击；⑤ DP 并非完整法规合规，仅为技术防护，需要配合人类监督与治理。

---

## 296. Why Does RLAIF Work At All?

**arXiv ID:** 2603.03000 | [PDF](https://arxiv.org/pdf/2603.03000v1)

**作者:** Robin Young `[一作]` `[通讯]` (Cambridge University), Robin Young (Cambridge University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并验证了隐含价值假设，解释了RLAIF自我改进的理论机制。

**💡 创新点**

将价值编码视为表示空间中的线性方向，并推导了生成-判断差距和RLAIF性能上限等关键理论结果。

**🔧 技术方法**

采用线性模型、直接偏好优化（DPO）和线性偏好推断等方法进行理论分析。

**📊 数据集**

主要基于公开的互联网预训练数据以及已有的RLAIF实验数据，没有新数据集。

**📈 对比分析**

通过理论推导与已有实验结果对比，说明RLAIF在模型规模、标签质量等因素上的性能提升，但未给出新的数值指标。

**⚠️ 局限性**

主要局限在于线性假设、对宪法到方向映射缺乏具体建模，以及对多元价值体系的简化。

---

## 297. Scaling Reward Modeling without Human Supervision

**arXiv ID:** 2603.02225 | [PDF](https://arxiv.org/pdf/2603.02225v1)

**作者:** Jingxuan Fan `[一作]` (Harvard University), Hanlin Zhang `[通讯]` (Harvard University)

**通讯引用:** 2639 | [OpenAlex ID](https://openalex.org/A5089829380)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无人工标注的奖励模型训练框架——奖励式扩展（Reward‑Based Scaling，RBS），通过将大规模网页文本的下一词续写结构转化为隐式的优劣对，在线生成无监督偏好对并训练奖励模型；随后在最佳‑N 选择和策略优化中验证其对下游推理与安全任务的提升。

**💡 创新点**

创新点在于：①利用无监督的续写对构造高质量偏好对，消除人工标注成本；②引入奖励中心化正则化以抑制噪声训练导致的尺度漂移；③系统评估了批量大小、数据质量、句子切分方式对模型性能的影响，并展示了模型在不同基础架构与规模上的良好迁移性。

**🔧 技术方法**

核心技术包括：在线续写对生成、Bradley‑Terry 比例损失与中心化正则化、In‑batch 负样本对比、Best‑of‑N 选择、Group Relative Policy Optimization（GRPO）进行策略优化；实验使用了RewardBench v1/v2、GSM8K、MATH、Toxigen、IFEval 等基准。

**📊 数据集**

主要数据集为 11M 词元的数学网页文本，来源于 FineMath‑4plus 与 InfiWebMath‑4plus 两大数学内容聚焦的 CommonCrawl 语料；未使用任何人工标注或对话对齐数据。

**📈 对比分析**

与已有的高质量人类标注奖励模型（Skywork、Jasper 等）相比，RBS 在 RewardBench v2 上平均提升约 +7.7 分（ID 上可达 +16.1 分），并在 Best‑of‑N 选择和 GRPO 策略优化中实现与这些基线相当甚至更优的下游推理与安全任务准确率；在不同基础架构（Llama、Qwen）和规模（1B–7B）上均保持竞争力。

**⚠️ 局限性**

局限性包括：①无监督信号仍然噪声较大，需通过批量大小和中心化正则化等技巧手动调节；②目前仅在数学内容聚焦语料上验证，跨领域泛化尚未彻底评估；③对极端安全或高风险场景的鲁棒性和潜在奖励劫持风险仍需进一步研究。

---

## 298. An Improved Combinatorial Algorithm for Edge-Colored Clustering in Hypergraphs

**arXiv ID:** 2603.03273 | [PDF](https://arxiv.org/pdf/2603.03273v1)

**作者:** Seongjune Han `[一作]` (Texas A and M University), Nate Veldt `[通讯]` (Texas A and M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种组合式的 (2-2/k) 近似算法，用于在边彩超图中给节点上色，目标是最小化不满足边的权重。

**💡 创新点**

创新点在于引入颜色对二进制线性规划，并通过构造稀疏的流网络直接求解，避免了枚举大量坏边对，从而实现近线性时间的组合式近似。

**🔧 技术方法**

使用颜色对二进制线性规划、最大流/最小割算法、半整数解的归约与取整以及确定性贪心局部比例更新等技术。

**📊 数据集**

在七个公开的边彩超图数据集上实验，数据规模从几千到几万顶点、几万至数十万边，边彩数 k 从 2 到 55 不等。

**📈 对比分析**

与 Hochbaum 的 (2-2/k) 近似、Veldt 的 2-近似以及 Gurobi 求解 LP+取整方案比较，新算法在运行时间、内存占用和解质量上均显著优于传统方法，尤其在 |ℬ| 极大时表现更稳健。

**⚠️ 局限性**

局限在于某些大规模实例仍需数分钟甚至超时；算法依赖最大流求解器的性能，且在特定超图结构下可能无法完全达到最优，未来可进一步优化流求解或改进 LP 取整策略。

---

## 299. On the Topology of Neural Network Superlevel Sets

**arXiv ID:** 2603.02973 | [PDF](https://arxiv.org/pdf/2603.02973v1)

**作者:** Bahman Gharesifard `[一作]` (Queen's University), Bahman Gharesifard `[通讯]` (Queen's University)

**通讯引用:** 2597 | [OpenAlex ID](https://openalex.org/A5022064746)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

分析神经网络在给定架构下，激活函数满足 Riccati 型微分方程时，其输出在解析域内为 Pfaffian 函数，从而给出超水平集和 Lie 括号秩降层的拓扑复杂度（Betti 数）仅由网络架构决定的上界。

**💡 创新点**

提出了只依赖网络结构（深度、宽度、Riccati 指数）的拓扑上界，扩展了通用逼近理论到向量场参数化的 Lie 括号秩降层，并首次给出权重无关的 Betti 数上界。

**🔧 技术方法**

使用 Pfaffian 函数理论、解析微分方程（Riccati 方程）、代数拓扑（Betti 数）以及神经网络层级递推的符号分析。

**📊 数据集**

无具体数据集，论文纯粹为理论分析与数学证明。

**📈 对比分析**

不涉及实验比较；本文通过数学证明给出理论上限，未给出数值性能指标或与其它方法的实验对比。

**⚠️ 局限性**

局限性包括：仅适用于满足 Riccati ODE 的光滑激活函数，非解析域或非光滑激活（如 ReLU）不在范围；上界可能过于保守，缺乏实际可计算性；未考虑训练过程与泛化误差。

---

## 300. Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures

**arXiv ID:** 2603.02271 | [PDF](https://arxiv.org/pdf/2603.02271v1)

**作者:** Manoj Vishwanathan `[一作]` (Google), Anand Raghunathan `[通讯]` (Purdue University)

**通讯引用:** 19768 | [OpenAlex ID](https://openalex.org/A5065766721)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对MolmoAct-7B VLA模型在NVIDIA Jetson Orin和Thor边缘平台上的执行进行实测与模拟，识别并量化动作生成阶段的内存带宽瓶颈，并预测100B规模模型在未来硬件配置下的性能。

**💡 创新点**

首次系统性量化VLA工作负载的内存带宽瓶颈，提出通过提升内存带宽（如GDDR7、LPDDR6X+PIM）和算力协同优化来弥合实时推理（10–20 Hz）与大规模模型（10–100 B参数）之间的性能差距。

**🔧 技术方法**

利用NVIDIA Nsight Compute进行kernel级别的性能剖析，构建基于roofline模型的高保真XPU仿真器，对多阶段Transformer（视觉编码、解码推理、动作转换）进行算子级别建模与跨算子预取优化。

**📊 数据集**

实验未公开具体数据集，使用MolmoAct-7B标准推理任务（含视觉输入与自然语言指令）进行基准测试；模拟中采用已验证的生产级加速器（GPU、TPU）参数进行模型缩放。

**📈 对比分析**

通过对比Orin/Thor硬件与多种内存升级变体（LPDDR5X、GDDR7、LPDDR6X+PIM）下的端到端延迟和控制频率，发现动作生成阶段占比最高（≈75%），内存带宽提升可使频率提升1–2倍，但仍远低于10 Hz实时需求；模拟显示100B模型在最佳内存+PIM配置下可达约5–7 Hz，尚未满足目标。

**⚠️ 局限性**

主要限制在于动作生成阶段的内存带宽瓶颈，现有边缘硬件无法在10–100 B规模下实现交互式推理；仿真精度受限于70–90%范围，未来需从硬件架构、算法协同、模型压缩等多维度共同突破。

---

## 301. Efficient Sparse Selective-Update RNNs for Long-Range Sequence Modeling

**arXiv ID:** 2603.02226 | [PDF](https://arxiv.org/pdf/2603.02226v1)

**作者:** Bojian Yin `[一作]` (Institute of Automation, Chinese Academy of Sciences), Guoqi Li `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了选择性更新RNN（suRNN），通过神经元级二进制门在无信息时刻跳过更新，保持记忆不被覆盖。

**💡 创新点**

核心创新在于将可跳过的身份携带与非线性更新分离到每个神经元，利用STE直通估计训练门，形成稀疏梯度路径，从而把有效更新次数与序列长度解耦。

**🔧 技术方法**

采用二进制门与STE、正弦周期门生成器、suGRU实现、CUDA融合加速以及标准BPTT训练。

**📊 数据集**

在Long Range Arena、WikiText‑103、Selective Copy、sMNIST/psMNIST/sCIFAR等长序列与像素级分类数据集上进行评估。

**📈 对比分析**

与Transformer、Reformer、S4、LSTM、GRU等对标，suGRU在LRA Pathfinder达84.9%（S4 94%）、WikiText‑103测试PPL 18.29（Transformer 18.32）、sCIFAR 87.26（LSSL 84.65）以及Selective Copy 99.5%（S6 99.7%）等任务表现出与或优于现代Transformer/SSM的效果。

**⚠️ 局限性**

限制包括仍需全序列BPTT导致训练成本高、仅实现单向（causal）模型、门调度方式固定、对稀疏执行的硬件支持不足，以及在双向或全局上下文任务中的表现尚未验证。

---

## 302. Boosting Meta-Learning for Few-Shot Text Classification via Label-guided Distance Scaling

**arXiv ID:** 2603.02267 | [PDF](https://arxiv.org/pdf/2603.02267v1)

**作者:** Yunlong Gao `[一作]` (Dalian University of Technology), Bo Xu `[通讯]` (Dalian University of Technology)

**通讯引用:** 11771 | [OpenAlex ID](https://openalex.org/A5108642431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Label-guided Distance Scaling策略，在训练阶段通过label-guided loss注入标签语义，在测试阶段使用Label-guided Scaler对支持样本进行缩放，从而减少随机支持样本导致的误分类。

**💡 创新点**

同时在训练和测试阶段利用标签语义指导表示学习和支持样本缩放，尤其在测试阶段提出非参数EM基的Label-guided Scaler，解决支持样本随机性问题。

**🔧 技术方法**

采用Prompt学习与BERT编码、Prototypical Networks与RRML等meta-learner、标签引导损失、EM非参数缩放、对比损失等技术。

**📊 数据集**

使用HuffPost、Amazon、Reuters、20News、Banking77、Clinic150等新闻、评论与意图分类数据集。

**📈 对比分析**

与PN、MAML、IN、DS-FSL、LaSAML、MLADA、ContrastNet、ProtoVerb、DE、TART、SPCNet等基线进行对比，在5-way 1-shot/5-shot任务上平均提升9.4%/10.1%，在10/15-way 1/5-shot任务上平均提升10.1%/2.1%，显著优于现有SOTA。

**⚠️ 局限性**

仅适用于单标签分类，多标签任务受限；引入标签语义增加训练资源；在真实应用中可能需要更强大编码器或prompt。

---

## 303. Saarthi for AGI: Towards Domain-Specific General Intelligence for Formal Verification

**arXiv ID:** 2603.03175 | [PDF](https://arxiv.org/pdf/2603.03175v1)

**作者:** Aman Kumar `[一作]` (Infineon Technologies India Private Limited), Sivaram Pothireddypalli `[通讯]` (Infineon Technologies India Private Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了Saarthi框架，通过多代理协作实现端到端形式化验证，包含结构化规则书、规范语法、GraphRAG检索与自动覆盖补全，支持人机交互；

**💡 创新点**

创新点在于将结构化规则书与规范语法与检索增强生成（GraphRAG）相结合，提高SVA生成的可控性和准确性；并通过自动覆盖缺口填补和HIL反馈实现迭代改进；

**🔧 技术方法**

采用多代理编排（Microsoft AutoGen）、LLM（GPT‑4.1、GPT‑5、Llama 3.3）、GraphRAG检索、知识图谱、自动覆盖分析与语法修正；

**📊 数据集**

使用NVIDIA CVDP基准（Memory Scheduler、AXI4Lite、CIC Decimator）以及自研ECC、汽车IP和浮点乘法器RTL；

**📈 对比分析**

对比未使用规则书/GraphRAG、HIL与无HIL、不同模型的Pass@1/2/3及覆盖率，结果显示规则书+GraphRAG提升SVA准确率约70%，迭代次数减少约50%，GPT‑5在复杂设计上覆盖率最高；

**⚠️ 局限性**

限制包括LLM易产生幻觉、对模糊自然语言规格敏感、GPT‑5推理延迟较高、极大断言集导致内存/token溢出，且仍需人工干预以完善复杂验证场景。

---

## 304. A simple Path-based LP Relaxation for Directed Steiner Tree

**arXiv ID:** 2603.02825 | [PDF](https://arxiv.org/pdf/2603.02825v1)

**作者:** Kanstantsin Pashkovich `[一作]` (University of Waterloo), Laura Sanità `[通讯]` (Bocconi University)

**通讯引用:** 938 | [OpenAlex ID](https://openalex.org/A5051955401)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了分层图中的有向斯坦纳树（DST）问题，通过简单的基于路径的线性规划松弛，达到了O(ℓlog k)的整合间隙。

**💡 创新点**

提出了一种简单的DST公式，绕过了层次结构的复杂性，提供了更透明的途径来达到当前的最佳界限，并且可以用来提供Sherali-Adams层次的O(ℓ)轮数足以减少整合间隙的替代证明。

**🔧 技术方法**

使用了线性规划（LP）松弛技术，特别是Sherali-Adams层次结构。

**📊 数据集**

研究了分层实例的有向斯坦纳树问题，具体数据集未明确提及，但涉及到的图是分层的。

**📈 对比分析**

与之前的研究相比，提出的公式在整合间隙上达到了O(ℓlog k)，并且提供了对Sherali-Adams层次的简单证明，表明O(ℓ)轮数足以达到相同的整合间隙。

**⚠️ 局限性**

该研究的局限性未在文中明确提及，但可能包括对更复杂图形或非分层实例的适用性限制。

---

## 305. From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness

**arXiv ID:** 2603.02411 | [PDF](https://arxiv.org/pdf/2603.02411v1)

**作者:** My H. Dinh `[一作]` (InterDigital Communications), Shahab Hamidi-Rad `[通讯]` (InterDigital Communications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Quantization-aware Dataset Distillation (QuADD)，在数据蒸馏流程中加入可微分量化模块，联合优化合成样本的数量和精度，以实现信息效率更高的数据压缩。

**💡 创新点**

创新点包括：① 将量化嵌入蒸馏循环，形成端到端可训练的统一框架；② 引入自适应非均匀量化（APoT）使量化密度与数据分布匹配；③ 在固定比特预算下系统性地分析速率-失真曲线，证明样本精度与数量的最佳权衡。

**🔧 技术方法**

技术细节包括：可微分量化层（硬/软四舍五入）、直通估计（STE）或平滑近似、APoT 自适应量化、梯度/特征匹配蒸馏目标、率失真分析与比特预算约束。

**📊 数据集**

实验数据集：图像数据集（CIFAR‑10、CIFAR‑100、ImageNette）以及 3GPP beam‑management 的无线通信表格数据。

**📈 对比分析**

与全精度蒸馏、AutoPalette、FreD、传统 Coreset 等基线在相同比特预算下对比，QuADD 在图像任务中保持 1% 内的精度，并实现约 10× 的存储压缩；在 3GPP 任务中压缩比超过 180×，且训练时间与基线相当或更短，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：对极端低精度或极大规模数据的可扩展性尚未充分验证；自适应量化学习可能导致训练不稳定；实验主要集中在二维图像与表格数据，对更高维或多模态数据的泛化需进一步探讨。

---

## 306. ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments

**arXiv ID:** 2603.03198 | [PDF](https://arxiv.org/pdf/2603.03198v1)

**作者:** Ziyang Gong `[一作]` (ACE Robotics), Xiaogang Wang `[通讯]` (ACE Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一款名为ACE-Brain-0-8B的通用基础模型，能够在空间认知、自动驾驶、低空感知和具身交互四个领域实现跨体现的感知、推理与决策。

**💡 创新点**

创新点主要包括：①将空间智能作为共享基座（Scaffold），为各领域提供统一的三维结构先验；②提出 Scaffold‑Specialize‑Reconcile (SSR) 训练范式，先训练空间专家，再分离训练各领域专家，最后通过无数据参数合并（Reconcile）实现知识融合；③采用数据‑free expert merging（如 WUDI、TSVM）以及后期的 GRPO 强化微调，兼顾稳定性与适应性。

**🔧 技术方法**

技术手段包括：多模态自回归 LLM（视觉编码器 + MLP 投影 + LLM 解码器），分阶段训练策略（空间基座 → 专家微调 → 参数合并 → 具身 SFT → GRPO RL），数据‑free 参数合并算法，Group Relative Policy Optimization（GRPO）强化学习。

**📊 数据集**

使用的数据集涵盖广泛：通用多模态指令集（Cambrain‑737K、LLaVA‑665K等）、空间认知数据（VSI、SAT、MindCube、VLM‑3R 等）、自动驾驶数据（MAPLM、DriveAction、Nuscenes‑QA、NuPlanQA、LingoQA）、低空感知数据（UrbanVideo‑Bench、AirCopBench、AVI‑Math、AirSpatial‑VQA、HRVQA）以及具身交互数据（RoboVQA、OpenEQA、EmbSpatial‑Bench、EgoPlan、EB‑Habitat 等）。

**📈 对比分析**

通过与 24 个跨领域基准（共 7+6+5+6 项）以及 9 组对照模型（含 GPT‑4o、Gemini‑2.5‑Pro、Qwen‑VL‑Max、MiMo‑VL‑7B、InternVL‑3.5‑8B、RoboBrain‑2.5‑8B 等）进行对比，ACE‑Brain‑0‑8B 在大多数指标上均达到或超越现有最优水平，尤其在空间认知（SAT 92.0%）和低空感知（AirCopBench 70.3%）上表现突出，并在具身领域（EgoPlan 55.3%）取得显著提升，证明 SSR 训练策略能有效解决跨体现的稳定‑可塑性困境。

**⚠️ 局限性**

局限性：①模型仍基于 8B 参数规模，规模扩展与算力消耗有待优化；②训练与评测多依赖离线数据，缺乏对实时闭环控制与持续学习的验证；③在物理连续预测、真实环境下的低延迟控制与多模态交互上尚未深入；④对新出现的体现形式（如水下、潜行等）需进一步迁移学习与在线增量训练。

---

## 307. Design Generative AI for Practitioners: Exploring Interaction Approaches Aligned with Creative Practice

**arXiv ID:** 2603.03074 | [PDF](https://arxiv.org/pdf/2603.03074v1)

**作者:** Xiaohan Peng `[一作]` (LISN Université Paris-Saclay, CNRS, Inria), Janin Koch `[通讯]` (UMR 9189 CRIStAL Univ. Lille, Inria, CNRS, Centrale Lille)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并阐述了三种针对设计实践的生成式 AI 交互范式（多模态意图拆解 DesignPrompt、可直接操作的视觉笔 FusAIn、可追踪的生成流程 DesignTrace），从而实现设计师与 AI 的动态协同对齐。

**💡 创新点**

创新点在于将对齐过程从单一提示点迁移到整个创作生命周期中，通过意图结构化、视觉操控以及流程记录三种交互维度实现多阶段、多模式的对齐；同时提出“主动‑被动”AI角色切换的“生产性摩擦”概念。

**🔧 技术方法**

核心技术包括多模态输入解析与编辑、基于图像纹理的智能笔驱动生成、节点式流程追踪与版本管理；采用现有的文本‑图像生成模型（如 Stable Diffusion）作为后端。

**📊 数据集**

未公开具体数据集，作者假设使用公开艺术/设计图像数据集（如 ArtStation、Behance 或 COCO 之类的标注图像）进行示例演示，强调实验更多是概念验证而非大规模数据驱动。

**📈 对比分析**

论文未进行系统的量化比较或性能评测，而是通过示例图像和用户经验描述来说明各交互范式在提升意图表达、可控性与创作灵活性方面的潜在优势；缺乏客观指标和基准对比。

**⚠️ 局限性**

主要局限包括：缺乏实证用户研究与客观评估；生成模型的偏差与漂移问题未得到充分解决；三种交互模式的互操作性与切换机制尚待进一步实现和验证；对复杂设计任务的适用范围与可扩展性仍不明。

---

## 308. Reinforcement Learning with Symbolic Reward Machines

**arXiv ID:** 2603.03068 | [PDF](https://arxiv.org/pdf/2603.03068v1)

**作者:** Thomas Krug `[一作]` (TU Dortmund University), Daniel Neider `[通讯]` (TU Dortmund University)

**通讯引用:** 1514 | [OpenAlex ID](https://openalex.org/A5064701353)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了Symbolic Reward Machines（SRM）来表示非马尔可夫奖励函数，并基于SRM设计了QSRM和LSRM两种学习算法，完成了从标准MDP到可解释非马尔可夫任务的完整端到端训练流程。

**💡 创新点**

创新点在于：①消除了传统Reward Machine对手工标签函数的依赖，改用符号公式守卫直接处理环境状态；②通过LSRM实现了在训练过程中自动推理并学习SRM，提供可解释的奖励结构信息；③在保持与标准RL接口兼容的前提下，提升了学习效率。

**🔧 技术方法**

主要技术包括：SRM模型（符号守卫+输出函数），QSRM（基于Q‑Learning的多状态更新），LSRM（基于counter‑example的SMT求解器Z3自动构造SRM），以及在无限状态下的DQSRM与DQRM。

**📊 数据集**

实验使用了离散与连续版Office World环境以及改造后的Mountain Car环境，对应任务包括post_inner_offices、diagonal_run、rml等。

**📈 对比分析**

与基线Q‑Learning、DQN（使用帧堆叠）以及现有的QRM/DQRM方法相比，SRM/QSRM在所有实验中都获得更高的mean10性能值，LSRM在有限状态空间下收敛到最优策略，在连续空间上虽然未完全最优但表现优于基线。

**⚠️ 局限性**

局限性包括：需要预先给定或学习公式模板（模板选择影响结果），SMT求解在高维或大量公式时计算开销大；在无限状态空间下仍存在收敛速度慢、未能完全达到最优的问题。

---

## 309. iGVLM: Dynamic Instruction-Guided Vision Encoding for Question-Aware Multimodal Understanding

**arXiv ID:** 2603.02748 | [PDF](https://arxiv.org/pdf/2603.02748v1)

**作者:** HanZpeng Liu `[一作]` (Huazhong University of Science and Technology), Kun He `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5033526822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双分支解耦的指令引导视觉编码器 iGVLM，使视觉表示在保持预训练视觉先验的同时根据文本指令动态调制。

**💡 创新点**

创新点在于将静态表示与动态调制解耦为两个分支，并通过 AdaLN 在每个 Transformer 层实现层级化指令调制，同时提出了 MM4 诊断基准。

**🔧 技术方法**

采用 CLIP 视觉与文本编码器、AdaLN 适配器、Zero‑FFN 融合模块，训练框架基于 LLaVA‑1.5。

**📊 数据集**

使用 MMStar、MM4、VQAv2、GQA、POPE、VizWiz、ScienceQA‑IMG 等公开基准；MM4 由 180 张图像和 720 题答案构成。

**📈 对比分析**

与 LLaVA‑1.5、QA‑ViT、DyFo 等基线对比，在 MMStar 上提升 3–4.5 分，在 MM4 上在多问一致性指标上显著降幅缓解，性能保持或提升而算力几乎不变。

**⚠️ 局限性**

限制在于对大型语言模型的依赖，部分评测仍以闭源模型为基准；在极大模型规模下提升有限，对指令表达的鲁棒性仍需进一步验证。

---

## 310. ModalPatch: A Plug-and-Play Module for Robust Multi-Modal 3D Object Detection under Modality Drop

**arXiv ID:** 2603.02481 | [PDF](https://arxiv.org/pdf/2603.02481v1)

**作者:** Shuangzhi Li `[一作]` (University of Alberta), Xingyu Li `[通讯]` (University of Alberta)

**通讯引用:** 2108 | [OpenAlex ID](https://openalex.org/A5100328581)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种轻量、无架构改动的 ModalPatch 插件，能够在多模态 3D 目标检测中对任意时间点的 LiDAR 或摄像头掉落进行补偿与融合，从而保持检测鲁棒性。

**💡 创新点**

创新点在于：①基于历史特征的时序预测（History‑Based Feature Prediction）能动态补偿缺失模态；②引入不确定性引导的跨模态融合（Uncertainty‑Guided Cross‑Modality Fusion），通过估计预测特征的可靠性来抑制噪声并强化有用信息；③实现 plug‑and‑play，训练时仅需两个阶段，无需重新训练主检测器。

**🔧 技术方法**

使用的技术包括：时间 Transformer + 可变形注意力实现历史特征预测；轻量 MLP 估计预测不确定性；不确定性加权的跨模态可变形 Transformer；两阶段分离训练（先训练 HFP，再冻结后训练 UCF）；采用 mAP/NDS 作为评价指标。

**📊 数据集**

在 nuScenes 数据集上进行实验，模拟不同 Drop_rate（10%，30%，50%）的 LiDAR/摄像头独立掉落情况。

**📈 对比分析**

与 BEVFusion、UniBEV、CMT、MEFormer 等 SOTA 检测器在无掉落、10%/30%/50% 的掉落率下进行对比；ModalPatch 在所有基线上平均提升 4.7% mAP（10%）/11.1% mAP（30%）/11.9% mAP（50%），NDS 同理，且在极端双模态同时掉落时仍能保持一定检测能力，整体性能显著优于原模型。

**⚠️ 局限性**

局限性：当单模态本身性能极差时，历史特征补偿效果有限；需进一步加强单模态特征增强以突破这一瓶颈。

---

## 311. ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection

**arXiv ID:** 2603.02541 | [PDF](https://arxiv.org/pdf/2603.02541v1)

**作者:** Deokyun Kim `[一作]` (ETRI), Jihun Cha `[通讯]` (ETRI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个针对森林地面视角缺失人检测的大规模数据集 ForestPersons，并在此数据集上对多种主流目标检测模型进行了基准评估。

**💡 创新点**

创新点在于（1）专门为森林低空（MAV）环境设计的地面视角数据集；（2）为每个目标提供姿态、可见度以及季节、天气等多维属性标注；（3）提出基于检测难度的划分策略，保证训练/验证/测试集分布一致；（4）对比了传统空中和地面数据集在该任务中的泛化能力。

**🔧 技术方法**

使用了 YOLO 系列（v3、X、v11）、RetinaNet、Faster R‑CNN、Deformable R‑CNN、SSD、DETR、DINO 以及 CZ‑Det 等多种检测框架，采用 Detectron2、MMDetection 等常见深度学习框架实现训练与评估。

**📊 数据集**

主要数据集为 ForestPersons（96,482 张图，204,078 个人实例），并与 HERIDAL、WiSARD、SARD、COCO、CrowdHuman、CityPersons 等公开 SAR 与地面检测数据集进行对比实验。

**📈 对比分析**

基准实验显示：最高 AP_50:95 仅达 66.3（Deformable R‑CNN），而 CZ‑Det 在 AP_50 与 AP_75 上表现最好；与先前空中或地面数据集训练的模型相比，ForestPersons 上的模型性能显著提升，表明其更贴合低空森林缺失人检测场景；在真实 MAV 测试集上，未加入噪声增强的模型达到 61.4 AP，说明域差距不大。

**⚠️ 局限性**

局限性包括：①数据采集采用手持/三脚架相机，未完全模拟 MAV 的运动模糊与传感器噪声；②虽然引入了可见度等级，但对极端遮挡（如树枝完全遮挡）仍缺乏足够样本；③模型在极端低光或雨雾等恶劣天气下的鲁棒性尚未充分验证。

---

## 312. Lozenge Tiling by Computing Distances

**arXiv ID:** 2603.02476 | [PDF](https://arxiv.org/pdf/2603.02476v1)

**作者:** Jean-Marie Favreau `[一作]` (Universite Clermont Auvergne), Léo Robert `[通讯]` (Universite de Picardie Jules Verne)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

提出一种多项式时间算法（推进表面算法）解决 Calisson 拼图，即在六边形内部给定非重叠和显著性约束的菱形填充问题，并将该问题转化为差分约束系统，利用 Bellman‑Ford 求解。

**💡 创新点**

在 Thurston 的立面理论基础上加入有向割与差分约束的图层，完全集成内部约束，证明填充可行性等价于无负权循环；提出推进表面算法的直观实现，并证明其对无限三角格局也适用。

**🔧 技术方法**

利用三角格、三维立方体映射、向量平移、向量定向、无向割、差分约束、Bellman‑Ford 最短路、Thurston 的高度函数等技术。

**📊 数据集**

论文主要是理论证明，并引用了在线的 500+ Calisson 实例，未使用标准数据集进行实验评估。

**📈 对比分析**

与传统的 Thurston 算法、匹配、SAT、最大独立集方法相比，推进表面算法在六边形大小 n 上实现 O(n³) 时间；Bellman‑Ford 方法为 O(|R|²)；在无限格局下只需 O(k³)（k 为约束数）。

**⚠️ 局限性**

仅适用于简单连通且无孔的区域；对边界重复穿过同一顶点/边的情况需额外处理；未给出实验性能，算法的实际常数因子未评估。

---

## 313. Aligning Fetal Anatomy with Kinematic Tree Log-Euclidean PolyRigid Transforms

**arXiv ID:** 2603.02371 | [PDF](https://arxiv.org/pdf/2603.02371v1)

**作者:** Yingcheng Liu `[一作]` (Computer Science and Artificial Intelligence Lab), Polina Golland `[通讯]` (Computer Science and Artificial Intelligence Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于运动树的对数欧几里得多刚体变换（KTPolyRigid），并将其与可微分体积体模型（基于SMPL）结合，用于将胎儿MRI图像对齐到统一的T姿势和人群平均形状，从而实现高质量的体积变形、群组配准和基于模板的肺部分割。

**💡 创新点**

① 通过在局部参考刚体变换下对相对刚体变换做对数平均，解决了传统PolyRigid在大非局部运动下的李代数映射歧义；② 采用流场差分映射（Φ_P）将个体形状线性插值到人群平均形状，实现体积内的无缝、双射变形；③ 将上述两步组合得到全局可微、光滑且无折叠的体积变形。

**🔧 技术方法**

可微分体积SMPL模型、KTPolyRigid变换、对数欧几里得平均、Mean Value坐标（MVC）流场、LBS（线性混合蒙皮）对比、基于U‑Net的肺部分割网络、群组配准的变形参数优化。

**📊 数据集**

53个健康单胎3D胎儿MRI体积（3T Siemens Skyra，EPI，3×3×6 mm³），采用解剖标记与分割对齐初始姿态。

**📈 对比分析**

与传统LBS变形场和原始PolyRigid对比；在变形字段指标上（%折叠、STDEV log₂|∂Φ|、Jacobian |J|、GPU时间和内存）KTPolyRigid表现出更少的折叠、更加平滑、GPU时间和内存消耗介于两者之间；在群组配准后的人群平均图像更锐利；在肺部分割中，5折交叉验证Dice分数达到0.81，甚至仅用5个样本训练亦能达到0.76。

**⚠️ 局限性**

① 仅在静态胎儿MRI上验证，未评估动态运动捕捉；② 对模型的可扩展性和对其他器官（非肺部）的适用性未进行系统评估；③ 需要更大规模数据以验证对不同胎儿发育阶段的泛化能力；④ 目前对高分辨率扫描的计算开销仍较高。

---

## 314. Generative adversarial imitation learning for robot swarms: Learning from human demonstrations and trained policies

**arXiv ID:** 2603.02783 | [PDF](https://arxiv.org/pdf/2603.02783v1)

**作者:** Mattes Kraus `[一作]` (University of Konstanz), Jonas Kuckling `[通讯]` (University of Konstanz)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5021453451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

构建了一套基于生成对抗模仿学习（GAIL）的机器人群体行为学习框架，并提供演示工具与在TurtleBot 4机器人群体上的实测验证。

**💡 创新点**

将群体模仿学习问题映射为单体问题，在判别器中使用群体级特征进行判别，从而实现了人类操作演示的群体行为学习，首次展示了高层演示驱动的群体行为生成。

**🔧 技术方法**

使用GAIL（生成对抗模仿学习）结合PPO训练策略；演示与仿真采用Unity+ROS 2的演示工具；判别器与策略网络均为多层感知机。

**📊 数据集**

使用手工演示数据以及PPO训练得到的策略轨迹，共六个任务在Unity仿真与实际TurtleBot 4群体上进行实验，形成了自己的演示数据集。

**📈 对比分析**

通过与手工演示、PPO演示、随机策略以及最终训练策略的累计奖励进行对比，结果显示大多数任务中学习到的策略与演示相近或超越，实测表现与仿真保持一致，表明框架具备一定的实用性。

**⚠️ 局限性**

受限于所选群体特征、奖励函数与判别器耦合导致的信息泄露、演示质量不一致，以及硬件安全保护未在仿真中模拟，导致在复杂任务如聚集和寻找任务上学习效果不佳，且需要进一步研究特征选择与奖励分离。

---

## 315. Speculative Speculative Decoding

**arXiv ID:** 2603.03251 | [PDF](https://arxiv.org/pdf/2603.03251v1)

**作者:** Tanishq Kumar `[一作]` (Stanford University), Avner May `[通讯]` (Together AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的并行解码框架Speculative Speculative Decoding (SSD)，通过预先预测验证结果并为多种可能的验证结果并行生成推测，消除传统Speculative Decoding中“草稿-验证”顺序依赖；

**💡 创新点**

核心创新点包括（1）对验证结果（包括被接受的令牌数和奖金令牌）进行准确预测；（2）在保持高接受率的同时，通过新型采样算法平衡推测质量与速度；（3）设计基于批量大小的自适应回退策略；（4）将草稿模型部署在独立硬件上实现真正的异步并行；

**🔧 技术方法**

技术包括：约束优化预测奖金令牌（90%准确率）；平衡接受率与推测质量的采样算法；多重缓存预先生成推测（speculation cache）；回退策略与备份推测器；在不同硬件（H100）上实现异步并行；

**📊 数据集**

使用四大公开数据集：Alpaca、GSM8k、UltraFeedback、HumanEval，评估 Llama‑3 系列和 Qwen‑3 系列模型；

**📈 对比分析**

与标准自回归解码和已优化的Speculative Decoding基线对比，SSD在 Llama‑3.1‑70B 上实现最高 2× 的速度提升（对比 SD）和最高 5× 的速度提升（对比自回归），并在多种批量大小和温度设置下保持或提升吞吐量-延迟 Pareto 前沿；

**⚠️ 局限性**

局限性包括：对批量大小、温度敏感，过大批量时缓存未命中率上升，导致回退策略开销增加；需要额外硬件支持草稿模型，实施复杂度较高；对不同模型和任务的通用性仍待进一步验证。

---

## 316. Beyond Caption-Based Queries for Video Moment Retrieval

**arXiv ID:** 2603.02363 | [PDF](https://arxiv.org/pdf/2603.02363v1)

**作者:** David Pujol-Perich `[一作]` (University of Barcelona), Michael Wray `[通讯]` (University of Bristol)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5049284007)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究视频片段检索模型在训练时使用注释式描述（caption）与实际用户搜索查询（search）之间的泛化差异，并提出新的搜索查询基准；

**💡 创新点**

提出通过删减视觉细节生成更通用的搜索查询、量化语言与多时刻差距，并通过去除DETR中的自注意力与引入查询丢弃来缓解查询崩塌；

**🔧 技术方法**

改进DETR架构（去除自注意力、查询丢弃正则化）以及使用NMS后处理；

**📊 数据集**

在HD-EPIC、YouCook2、ActivityNet-Captions三大数据集上生成三种搜索查询变体（-S1/2/3、YC2-S、ANC-S）；

**📈 对比分析**

与原始Caption训练模型相比，改进模型在搜索查询上提升R_m与mAP_m最高可达约21.8%，并在多时刻查询中恢复近70%与oracle间的性能差距；

**⚠️ 局限性**

仍未解决语言细化差距问题，需更先进的跨模态理解与多样化查询生成来进一步提升泛化

---

## 317. A Directed Graph Model and Experimental Framework for Design and Study of Time-Dependent Text Visualisation

**arXiv ID:** 2603.02422 | [PDF](https://arxiv.org/pdf/2603.02422v1)

**作者:** Songhai Fan `[一作]` (Monash University), Helen Purchase `[通讯]` (Monash University)

**通讯引用:** 6130 | [OpenAlex ID](https://openalex.org/A5006604913)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Time‑Track Narrative Graph (TTNG) 模型并通过用户研究验证其对新闻叙事结构的可视化表达。

**💡 创新点**

创新点在于：①将叙事上下文要素 (SCEs) 与叙事数据拓扑 (NDTs) 统一为 TTNG，形成可跨视觉化方式的中间表示；②设计低成本 Graph‑to‑Text 管线，利用 LLM 生成可控的合成新闻数据；③将叙事动机 (motifs) 作为最小可识别单元，对其可理解性进行系统评估。

**🔧 技术方法**

使用技术包括：GPT‑4（Crafter/Writer）生成文本、规则化 Cartographer 进行属性映射与时间分配、Jaccard/TF‑IDF/BERT 计算相似度验证数据结构、用户实验平台（Prolific）收集行为数据、混淆矩阵与统计分析评估识别效果。

**📊 数据集**

数据集为人工构造的 3‑节点 TTNG 合成新闻集，包含 28 条故事（84 条公告），由 Graph‑to‑Text 管线自动生成，未使用真实新闻语料。

**📈 对比分析**

方法评估：通过 30 名受试者的 10 题任务，计算每个动机的识别准确率（平均 3.1/10）及混淆矩阵；结果显示顺序动机识别率较高，非顺序动机被误判；未与传统可视化技术直接对比，主要展示人类对不同动机的识别难度。

**⚠️ 局限性**

局限性：①合成数据生态效度低，缺乏真实新闻的复杂性；②用户样本量小（30 人）且实验仅覆盖三节点动机；③TTNG 对边和轨道的约束过于简化，无法捕捉闪回、环形等复杂叙事；④缺乏对多层视图、情感/因果信号的支持。

---

## 318. Retrievit: In-context Retrieval Capabilities of Transformers, State Space Models, and Hybrid Architectures

**arXiv ID:** 2603.02874 | [PDF](https://arxiv.org/pdf/2603.02874v1)

**作者:** Georgios Pantazopoulos `[一作]` (University of Edinburgh), Alessandro Suglia `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究在两类合成检索任务（n‑gram检索与位置检索）上系统比较Transformer、State‑Space Model（SSM）及其混合架构的表现，评估数据效率、长度泛化、鲁棒性与内部表示；

**💡 创新点**

创新点在于揭示SSM能自发形成局部感知的嵌入结构，证明混合模型能兼顾Transformer的全局注意力与SSM的线性效率，并为不同检索任务提供架构选择的理论指引；

**🔧 技术方法**

采用Transformer（RoPE/NoPE）、SSM（Mamba、Mamba2）以及两种混合结构（交错式Hybrid_I与双流Hybrid_2S），在约1.5亿参数规模下训练并可视化隐藏状态与位置嵌入；

**📊 数据集**

使用自制的合成序列数据集，生成长度≤100（或200）token的随机序列，在不同长度与重复查询设置下进行评估；

**📈 对比分析**

通过样本量、长度泛化（至4×训练长度）、多样查询鲁棒性和表示质量等指标对比；混合模型在n‑gram检索上最优，Transformer在位置检索上领先，SSM单体表现最差；

**⚠️ 局限性**

仅在合成任务上验证，无法完全映射至真实语言或多模态任务；仅考察解码器架构，未评估编码器或编码‑解码器场景；

---

## 319. Strategic Shaping of Human Prosociality: A Latent-State POMDP Framework

**arXiv ID:** 2603.02379 | [PDF](https://arxiv.org/pdf/2603.02379v1)

**作者:** Zahra Zahedi `[一作]` (Honda Research Institute USA), Kumar Akash `[通讯]` (Honda Research Institute USA)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于潜在状态的POMDP框架，机器人通过在多轮人机交互中学习并影响人类的利他性状态，进而做出在任务进展与社会目标间折中的行动。

**💡 创新点**

创新点在于：① 将人类利他性建模为可演化的潜在状态并通过EM估计其转移与观测；② 利用基于信念的规划策略主动塑造人类的利他性；③ 结合实验验证展示策略相较传统规则式方法在提升团队表现与人类帮助率方面的优势。

**🔧 技术方法**

使用技术包括：潜在状态POMDP建模、EM（Baum‑Welch）参数学习、基于信念的规划（SARSOP求解器）、混合效应逻辑回归等统计分析。

**📊 数据集**

使用的数据集为两组用户研究数据：① 540名参与者在五轮代币收集游戏中产生的交互轨迹用于学习潜在状态转移与观测；② 100名参与者在九轮交互中验证策略与基线的对比。

**📈 对比分析**

通过与四种基线（Always Help/Signal、Never Help/Signal、Myopic Greedy、Reciprocal-Reactive）在团队表现、累计代币、观察到的人类帮助率等指标上进行混合ANOVA比较，结果显示ls‑POMDP策略在所有指标上显著优于基线，尤其在降低成本的同时提升人类帮助率。

**⚠️ 局限性**

局限性包括：① 利他性状态仅通过行为间接推断，未获得真实内在动机；② 仅在受限的代币收集游戏中验证，难以直接推广至更复杂或开放式场景；③ 对奖励与成本参数敏感，需要根据任务进行调优；④ 伦理层面需关注机器人对人类行为的主动影响与透明度。

---

## 320. Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback

**arXiv ID:** 2603.02232 | [PDF](https://arxiv.org/pdf/2603.02232v1)

**作者:** Amirhossein Afsharrad `[一作]` (Stanford University), Mohammad Ghavamzadeh `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于离散序数回归的奖励模型框架，利用Likert量表的等级偏好数据训练奖励函数，并在RLHF与DPO中实现。

**💡 创新点**

核心创新在于将奖励学习问题转化为序数回归，自动学习阈值参数，消除手工设定margin或缩放因子的做法，并提供理论证明阈值正则化与对称性条件。

**🔧 技术方法**

使用了概率序数回归（负对数似然）和margin‑based All‑Threshold 损失，阈值重参数化与L2正则化，联合训练奖励网络与阈值。

**📊 数据集**

主要数据集为HelpSteer2、HelpSteer3（含7级偏好标签）以及评测基准RewardBench和RM‑Bench。

**📈 对比分析**

在多种模型（Llama‑3.1‑8B、Mistral‑7B、Zephyr‑7B）上与Margin BT、Scaled BT、Soft Label 等 heuristic 进行对比。NLL‑Sym（对称阈值）在绝大多数任务和模型中均以2‑5% 的平均提升击败基线，且在错误严重度、阈值学习与鲁棒性评估上表现更佳。

**⚠️ 局限性**

限制包括：仍需正则化以防阈值无界；对称阈值假设在某些任务中可能不成立；当前仅处理单一维度的Likert偏好，尚未扩展到多维或不确定性标签；实现对更大规模模型与多样化反馈形式的泛化还有待验证。

---

## 321. From Shallow to Deep: Pinning Semantic Intent via Causal GRPO

**arXiv ID:** 2603.02675 | [PDF](https://arxiv.org/pdf/2603.02675v1)

**作者:** Shuyi Zhou `[一作]` (University of Chinese Academy of Sciences), Wei Ma `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段因果-GRPO框架（TSC-GRPO）来实现意图固定，从而提高LLM在prefix injection等对抗性攻击下的安全性。

**💡 创新点**

创新点在于识别并解决“语义表示衰减”问题，采用因果判别的意图探针与累积因果惩罚的GRPO相结合，实现深层意图固定与后期拒绝。

**🔧 技术方法**

使用因果判别学习的意图探针、Group Relative Policy Optimization（GRPO）、累积因果奖励、硬负样本扩增以及“fork-in-the-road”训练场景。

**📊 数据集**

对抗数据来自AdvBench、AutoDAN、GCG等攻击；安全数据使用HEx-PHI恶意指令和Alpaca安全指令；实验还利用GSM8K、HumanEval、MBPP、TruthfulQA评估通用性。

**📈 对比分析**

与RLHF、SFT、PSR、NemoGuard等基线比较，TSC-GRPO在绝大多数攻击方法下将ASR降至0%或接近0%，同时保持或提升GSM8K/HumanEval/MBPP/TruthfulQA等指标，显示出更优的安全与实用性。

**⚠️ 局限性**

局限包括对特定模型与数据集的依赖、对超参数的敏感性、缺乏理论上完备的安全保证，以及可能存在的计算开销与对未知攻击的泛化能力不足。

---

## 322. AutoFFS: Adversarial Deformations for Facial Feminization Surgery Planning

**arXiv ID:** 2603.02288 | [PDF](https://arxiv.org/pdf/2603.02288v1)

**作者:** Paul Friedrich `[一作]` (University of Basel), Philippe C. Cattin `[通讯]` (University of Basel)

**通讯引用:** 9291 | [OpenAlex ID](https://openalex.org/A5048965835)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 AutoFFS，一种基于对抗性形变的面部女性化手术规划框架

**💡 创新点**

创新点在于使用针对性对抗攻击生成对照面部形态，并通过多模型集成提升鲁棒性

**🔧 技术方法**

采用自由形变（B‑spline FFD）、多模型二分类器、对抗攻击与平滑/弯曲正则化技术

**📊 数据集**

利用SMS多发性硬化症患者的 MR 扫描数据（444 张，152 男，292 女）进行骨骼结构提取

**📈 对比分析**

在分类准确率、F1、AUROC 以及人类感知实验中，AutoFFS 能将目标性别属性显著提升，实验显示识别率约 63%，显著高于随机

**⚠️ 局限性**

局限包括缺乏真实手术前后验证、仅使用 MR 数据、形变主要限于面部前部，且未能直接评估对手术实际效果的影响

---

## 323. GLoRIA: Gated Low-Rank Interpretable Adaptation for Dialectal ASR

**arXiv ID:** 2603.02464 | [PDF](https://arxiv.org/pdf/2603.02464v1)

**作者:** Pouya Mehralian `[一作]`, Hugo Van hamme `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于地理坐标门控的低秩参数高效适配框架 GLoRIA，用于提升方言语音识别性能并实现可解释性。

**💡 创新点**

创新点在于：1) 用坐标驱动的门控 MLP 动态调节每个低秩方向的权重，形成连续的空间适配；2) 结合正交性和稀疏正则化提升适配方向多样性与可解释性；3) 在保持参数更新不到 10% 的同时，显著提升 WER 并在未见方言上保持良好泛化。

**🔧 技术方法**

技术包括：低秩适配 LoRA、坐标门控 MLP、Softplus 激活、正交性与熵正则化、非负矩阵分解（NMF）对适配方向进行可视化和聚类分析；实现于 ESPnet 基于 Conformer 的 ASR 模型。

**📊 数据集**

使用 GCND（比利时、荷兰南部与法兰德斯地区的 411 小时方言语音）数据集，配合精确的经纬度元数据。

**📈 对比分析**

与传统全微调、LoRA、联合微调以及地理条件下的全微调基线相比，GLoRIA 在 9 个方言区域的平均 WER 由 42.5% 降至 34.6%（约 17% 绝对降幅），且在未见方言上也保持领先；参数更新仅占总参数的 2.7%–10%。

**⚠️ 局限性**

局限性包括：1) 仍对方言数据量敏感，极端稀缺地区可能效果受限；2) 适配方向仅受坐标影响，无法捕捉非空间的社会语言因素；3) 需要额外的 NMF 解释步骤，增加后处理复杂度。

---

## 324. Safe Whole-Body Loco-Manipulation via Combined Model and Learning-based Control

**arXiv ID:** 2603.02443 | [PDF](https://arxiv.org/pdf/2603.02443v1)

**作者:** Alexander Schperberg `[一作]` (Mitsubishi Electric Research Laboratories), Stefano Di Cairano `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一台配备6自由度机械臂和腕部力/力矩传感器的四足机器人上，结合模型驱动的阻抗控制和强化学习腿部运动控制，实现了基于外部力/力矩驱动的全身协同运动；

**💡 创新点**

核心创新点在于：①将模型驱动的阻抗控制与RL腿部控制耦合，使得手臂与腿部能同步响应外部力矩；②通过参考治理器为阻抗控制提供正式的安全保证；③在卡尔曼滤波中加入神经网络预测误差，提高基站速度估计精度；

**🔧 技术方法**

使用技术包括：阻抗（admittance）控制、参考治理器（Reference Governor）、强化学习（PPO）腿部控制、Kalman滤波+神经网络状态估计、离线MOAS求解与KD‑Tree实时查询；

**📊 数据集**

主要使用自收集的硬件实验数据（力/力矩传感器、IMU、足部位置、Mocap测量）进行验证，未使用公开数据集；

**📈 对比分析**

与无参考治理器的对比实验表明，参考治理器显著减少了位置和力的超限；在实验中，线性误差MSE≤0.005 m²/s²，角速度误差MSE≤0.029 rad²/s²；

**⚠️ 局限性**

局限性包括：对6自由度全量角速度跟踪仍存在误差，需进一步调参；实验仅在单一平板负载下进行，未考虑不同物体质量变化；缺乏视觉感知支持。

---

## 325. The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes

**arXiv ID:** 2603.02985 | [PDF](https://arxiv.org/pdf/2603.02985v1)

**作者:** Reuben Docea `[一作]` (National Center for Tumor Diseases), Stefanie Speidel `[通讯]` (Technical University Dresden)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并发布了一个在猪腹部软组织上结合内镜视频和结构光点云的四维重建数据集D4D。

**💡 创新点**

创新点在于首次提供配对的高质量几何真值，支持在非刚性腹腔环境下评估3D/4D重建与SLAM。

**🔧 技术方法**

使用Zivid结构光摄像机、da Vinci Xi机器人、光学跟踪、PnP+RANSAC、LightGlue、ICP进行数据采集与配准；后续评估采用PSNR/SSIM/LPIPS、点云误差统计。

**📊 数据集**

数据集包含98条高质量采集序列，覆盖完整变形、增量变形、移动相机三种模式；每条序列附带校准、掩码、深度图、点云等。

**📈 对比分析**

评估通过与结构光点云对比计算点对点距离，Curated位姿的误差平均2.37mm，Inliers<3mm 57.9%，相对Nominal 30.6%更好；同时展示ForPlane方法的3D结果。

**⚠️ 局限性**

限制在于仅来自猪尸体，未覆盖不同解剖结构和真实手术场景；几何配准仍有残差，缺乏可直接评估完整重建的绝对精度。

---

## 326. AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework

**arXiv ID:** 2603.03233 | [PDF](https://arxiv.org/pdf/2603.03233v1)

**作者:** Zihang Zeng `[一作]` (Fudan University), Xi Chen `[通讯]` (Fudan University)

**通讯引用:** 59686 | [OpenAlex ID](https://openalex.org/A5100329996)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于贝叶斯对抗的多智能体框架（LCP），用于AI‑for‑Science代码生成，并将其实现为低代码平台；

**💡 创新点**

创新点在于：①将代码生成、测试用例和提示词视为可协同进化的三大组件，并通过非LLM贝叶斯更新规则递归优化；②引入对抗性测试用例生成机制，持续提升代码鲁棒性；③为非专业科研人员提供交互式规划与提示消歧，降低对专家级提示工程的依赖；

**🔧 技术方法**

技术包括多智能体系统（Task Manager、Solution Generator、Evaluator）、贝叶斯递归更新、贝叶斯优化用于代码性能估计、低代码平台接口、结构化提示生成与交互式澄清；

**📊 数据集**

使用了多种基准数据集：通用代码生成基准（CodeXGLUE、HumanEval、MBPP、BBH等）、科学任务基准（SciCode、ScienceAgentBench、跨学科地球科学基准）以及公开的参考代码库；

**📈 对比分析**

与现有SOTA（如prompting、CodeAct、Self‑Debug等）以及多种LLM后端（Qwen、Claude、GPT‑4o等）在Pass@1、Success Rate、Verification Rate等指标上对比，实验显示即使在小模型上也能获得与大模型相当或更优的表现，且在无知识/有知识两种提示条件下对提示质量的鲁棒性显著提升；

**⚠️ 局限性**

主要局限包括：①对初始参考代码质量的依赖，难以完全执行隐式物理定律；②多轮生成与评估导致显著的token与计算开销；③对深度学习模型的评估成本高，且训练动态波动可能影响贝叶斯更新。

---

## 327. Composable Attestation: A Generalized Framework for Continuous and Incremental Trust in AI-Driven Distributed Systems

**arXiv ID:** 2603.02451 | [PDF](https://arxiv.org/pdf/2603.02451v1)

**作者:** Sheng Sun `[一作]` (Dell Canada), Sarah Evans `[通讯]` (Dell Technologies Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种可组合的证明框架，用于在人工智能驱动的分布式系统中实现持续和增量信任。

**💡 创新点**

创新点在于引入了可组合证明的概念，允许对系统组件进行独立证明，同时保持整个系统的完整性可验证性。

**🔧 技术方法**

使用了多种密码学构造，包括Merkle树、累加器和多重签名方案。

**📊 数据集**

未具体提及使用的数据集，但应用于人工智能模型完整性验证和分布式系统。

**📈 对比分析**

通过形式分析验证了这些构造满足核心属性，性能方面，Merkle树提供O(log n)的包含证明复杂度，累加器提供O(1)的大小证明，而多重签名在某些方案中实现O(1)的聚合。

**⚠️ 局限性**

局限性在于不同构造的安全性依赖于不同的密码学假设，且在动态环境中的实时优化和隐私保护仍需进一步研究。

---

## 328. Physics-informed post-processing of stabilized finite element solutions for transient convection-dominated problems

**arXiv ID:** 2603.03259 | [PDF](https://arxiv.org/pdf/2603.03259v1)

**作者:** Süleyman Cengizci `[一作]` (Antalya Bilim University), Srinivasan Natesan `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 2198 | [OpenAlex ID](https://openalex.org/A5047608967)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于SUPG＋YZβ稳定有限元与物理信息神经网络（PINN）的混合时变解算框架，能够在终端时间对对流支配的输运问题进行高精度后处理。

**💡 创新点**

创新点在于将PASSC方法从稳态扩展到非稳态，并通过选择性终端时点数据、随机傅里叶特征映射、残差块与层归一化的深度残差网络、渐进训练权重以及距离筛选的自适应物理约束，实现了对传统稳定方法缺陷的自动修正。

**🔧 技术方法**

采用的技术包括：SUPG+YZβ稳健有限元、随机 Fourier 特征映射、带残差块和层归一化的深度残差网络、动态权重的三阶段训练、梯度/损失/残差裁剪、自动微分、混合精度训练、AdamW优化器与学习率调度。

**📊 数据集**

实验使用了五个基准问题（1D/2D 边界层、内部层、波动、非线性 Burgers）产生的SUPG-YZβ数值解作为训练数据，没有使用外部标注数据集。

**📈 对比分析**

通过与传统 SUPG、SUPG-YZβ 以及 PINN 独立训练的结果进行 L² 误差、点误差以及可视化对比，发现混合方法在终端时间显著降低误差（1–2 个数量级）并消除了振荡，整体性能优于单一方法。

**⚠️ 局限性**

局限性包括：需要经验性调节时间窗口、距离阈值及权重；对极薄层或高维/三维问题仍需大量训练资源；非线性或多耦合系统对数据权重要求更高；尚未在大规模工业场景或实时预测中验证。

---

## 329. SPARC: Spatial-Aware Path Planning via Attentive Robot Communication

**arXiv ID:** 2603.02845 | [PDF](https://arxiv.org/pdf/2603.02845v1)

**作者:** Sayang Mu `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6832 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于空间关系增强的多头注意力机制，用于多机器人路径规划中的通信；

**💡 创新点**

将曼哈顿距离嵌入注意力权重，使机器人能动态优先考虑空间近邻的信息，从而显著提升高密度环境下的协作效率；

**🔧 技术方法**

结合Transformer编码器、GRU门控消息融合、MAPPO强化学习框架，使用曼哈顿距离向量、距离约束掩码和多头注意力；

**📊 数据集**

使用人工生成的网格地图（10×10、25×25、40×40）与三种障碍密度（0%、15%、30%）的随机障碍场，训练机器人数为8，测试至128；

**📈 对比分析**

与SCRIMP、DHC、PICO以及经典中心化规划ODrM*等方法对比，采用成功率、最大到达数、碰撞率等指标；在128机器人、30%障碍下成功率约75%，比SCRIMP提升约25个百分点；

**⚠️ 局限性**

仅在离散网格、同类机器人且曼哈顿距离可直接反映可达性场景下验证；不支持连续动作空间、异构机器人或动态起止点；曼哈顿距离在拥挤环境中可能误导真实可达性。

---

## 330. DREAM: Where Visual Understanding Meets Text-to-Image Generation

**arXiv ID:** 2603.02667 | [PDF](https://arxiv.org/pdf/2603.02667v1)

**作者:** Chao Li `[一作]` (Massachusetts Institute of Technology), Shlok Kumar Mishra `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的视觉‑语言框架 DREAM，兼顾对比学习与文本到图像生成。

**💡 创新点**

创新点在于 Masking Warmup 逐步调节遮挡率以平衡对比与生成目标，并提出 Semantically Aligned Decoding 在推理时利用模型自身的对比表示自引导生成。

**🔧 技术方法**

使用 ViT encoder‑decoder 结构、Stable Diffusion VAE 的连续 token 化、扩散重建损失、CLIP 风格 InfoNCE 对比损失以及 MAR 风格的多路并行生成策略。

**📊 数据集**

主要在 Conceptual 12M (CC12M) 训练，评估基于 ImageNet‑1K、MS‑COCO 等公开数据集。

**📈 对比分析**

与 CLIP、MAR、FLUID、REPA 等基线在线性探针、微调、few‑shot、语义分割、深度估计以及 FID/CLIP Score 等指标上对比，DREAM 在所有测评中均优于对照模型：线性探针 72.7% (+1.1% CLIP)，生成 FID 4.25 (+6.2% FLUID)，Semantically Aligned Decoding 提升 6.3% 文本‑图像相似度且吞吐量提升 10.1%。

**⚠️ 局限性**

局限性包括对大规模数据集与更高分辨率下的验证不足；Masking Warmup 需要精细调参；与最先进的扩散模型相比，生成质量仍略有差距。

---

## 331. UniSkill: A Dataset for Matching University Curricula to Professional Competencies

**arXiv ID:** 2603.03134 | [PDF](https://arxiv.org/pdf/2603.03134v1)

**作者:** Nurlan Musazade `[一作]`, Mike Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniSkill 数据集和验证器模型，用于将课程学习目标与 ESCO 职业技能对齐。

**💡 创新点**

首次公开开源课程-技能匹配数据集，并结合课程标题与内容的联合输入、标签化输入以及合成数据提升模型性能。

**🔧 技术方法**

采用 BERT 及多种领域专用模型（ESCOXLM‑R、JobBERT 等）与句子嵌入、对比学习、特征标记等技术。

**📊 数据集**

使用芬兰五所大学 2019-2025 年的公开课程描述，手工标注 2,192 条句子/课程-技能对，并通过 LLM 生成 800 条正样本及 3,200 条三元组合成数据。

**📈 对比分析**

通过与单独句子/标题匹配模型对比，联合模型在真实测试集上达到 0.89 的 F1（召回 0.82）并在硬样本上仍保持 0.74 的准确率。

**⚠️ 局限性**

仅涵盖两个职业组，数据来源局限于芬兰大学课程，合成数据与招聘广告不易迁移，且模型对模糊或隐式技能的识别仍存在误判。

---

## 332. Integrating Health Sensing into Cellular Networks: Human Sleep Monitoring Using 5G Signals

**arXiv ID:** 2603.02558 | [PDF](https://arxiv.org/pdf/2603.02558v1)

**作者:** Ruxin Lin `[一作]` (Michigan State University), Huacheng Zeng `[通讯]` (Michigan State University)

**通讯引用:** 1833 | [OpenAlex ID](https://openalex.org/A5027120851)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用商业5G基站的上行声波参考信号（SRS）实现无接触睡眠监测，提取呼吸率和睡眠动作分类；

**💡 创新点**

首次在真实5G网络环境下验证基于CSI的睡眠监测，提出轻量级呼吸率估计与CNN睡眠动作分类模型，兼顾移动干扰和信号不稳定性；

**🔧 技术方法**

信号处理（CSI归一化、带通滤波、峰值检测）、频谱集中度评估、CNN时频特征提取；

**📊 数据集**

从MSU私有5G网络收集的CSI数据（5个实验位置、3位受试者、564训练样本、340测试样本）；

**📈 对比分析**

通过实验与基准参考（呼吸率误差8.8%）及分类准确率（平均91.2%呼吸率，85.5%动作分类）对比，证明方法优于传统手段；

**⚠️ 局限性**

仅在室内实验，受限于干扰源分布、设备部署范围及对外部环境的鲁棒性不足，未来需在户外或更复杂环境下验证。

---

## 333. Semantic Forwarding and Codebook-Enhanced Model Division Multiple Access for Satellite-Terrestrial Networks

**arXiv ID:** 2603.02536 | [PDF](https://arxiv.org/pdf/2603.02536v1)

**作者:** Jinghong Huang `[一作]` (Beijing University of Posts and Telecommunications), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 85696 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向LEO卫星-地面网络的语义前向（SFSC）框架，并扩展至代码书分割增强的MDMA（CS-MDMA）多用户传输。

**💡 创新点**

创新点包括：① 语义前向机制实现卫星端仅在语义层级转发，避免完整解码重编码导致的噪声积累和高计算开销；② 采用向量量化联合语义编码与调制（VQ‑PEM）实现端到端可微的语义压缩与符号映射；③ 引入基于FiLM的信噪比自适应语义恢复，动态融合信道信息；④ 在多用户场景下设计代码书分割与增强解码策略，提升光谱效率。

**🔧 技术方法**

使用的技术包括：向量量化（VQ）、概率编码-调制（PEM）、ReinMax梯度估计、FiLM特征调制、模型分割与组合（MDMA）、深度联合源信道编码（DeepJSCC）、以及基于卷积神经网络的编码器/解码器。

**📊 数据集**

训练与评估使用Cityscapes数据集（512×512彩色图像），包含2375训练、500验证、1525测试样本。

**📈 对比分析**

对比方法包括：透明模式与再生模式的DeepJSCC、JCM、传统JPEG+LDPC方案，以及MDMA、NOMA、NOMA‑JSCC等多用户方案。实验结果表明：SFSC在低SNR（-10 dB）下相较基线提升约7.9 dB PSNR，CS‑MDMA在多用户场景下低SNR下提升4.5–7.0 dB；同时参数量减少84%，计算复杂度下降78%。

**⚠️ 局限性**

局限性在于：仍需进一步验证在更大规模多跳、异步信道以及更宽频带下的鲁棒性；对极端动态信道的泛化仍受限；虽然计算量已降低，但在极低功耗卫星平台上的实现成本仍需评估。

---

## 334. Concept Heterogeneity-aware Representation Steering

**arXiv ID:** 2603.02237 | [PDF](https://arxiv.org/pdf/2603.02237v1)

**作者:** Laziz U. Abdullaev `[一作]` (National University of Singapore), Tan M. Nguyen `[通讯]` (National University of Singapore)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5101399124)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于聚类与最优传输的输入自适应表示导向方法CHaRS，用于在大型语言模型中实现更精细的行为控制；

**💡 创新点**

通过将源目标表示建模为高斯混合模型并解决离散OT问题，克服了传统单向平移假设的局限，生成光滑、上下文相关的 steering 场；

**🔧 技术方法**

核心技术包括高斯混合模型（GMM）表示、离散最优传输（OT）与巴氏投影、核加权组合，以及基于PCA的低秩阈值化（CHaRS-PCT）；

**📊 数据集**

在多种文本安全与生成任务上使用了ADVBENCH、ALPACA、RealToxicityPrompts、COCO Captions等公开数据集；

**📈 对比分析**

与现有的 Activation Addition 与 Directional Ablation 等基线相比，CHaRS 在 jailbreaking、毒性缓解与图像风格控制等任务上均显著提升了攻击成功率、毒性减弱程度或风格诱导准确率，同时保持或提升了模型通用性能；

**⚠️ 局限性**

局限在于目前使用等方差的高斯混合和 k-means 聚类，可能无法捕捉更复杂的方向性结构，且在高维特征中核宽度选择敏感，未来工作需探索更丰富的协方差模型和特征加权方法。

---

## 335. The Malignant Tail: Spectral Segregation of Label Noise in Over-Parameterized Networks

**arXiv ID:** 2603.02293 | [PDF](https://arxiv.org/pdf/2603.02293v1)

**作者:** Zice Wang `[一作]` `[通讯]` (Northeastern University), Zice Wang (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在有标签噪声情况下，过参数化网络如何在谱空间分离信号与噪声，提出恶性尾部（Malignant Tail）现象，并通过后置谱截断恢复泛化能力。

**💡 创新点**

首次将恶性过拟合定义为谱分离现象，证明SGD主动将噪声投射到高频正交子空间，并提出基于有效秩的显式谱截断与早期谱停止的鲁棒训练方法。

**🔧 技术方法**

使用谱线性探测、有效秩、两邻居法估计内在维度、随机矩阵理论、神经崩塌分析以及后置谱截断等技术。

**📊 数据集**

在CIFAR‑10/100、WideResNet、VGG、ViT等网络上，在20%对称标签噪声设置下进行实验。

**📈 对比分析**

与随机投影、传统早停、未截断训练等做对比，谱截断后测试准确率提升约4‑6%，显著超越基线，并恢复在噪声干扰下的最优泛化。

**⚠️ 局限性**

方法仅适用于随机独立噪声，无法对系统性噪声或对齐噪声处理，且仍需完整训练后再进行截断，训练成本未降低。

---

## 336. MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry

**arXiv ID:** 2603.02351 | [PDF](https://arxiv.org/pdf/2603.02351v1)

**作者:** Leo Kaixuan Cheng `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**通讯引用:** 1933 | [OpenAlex ID](https://openalex.org/A5080873211)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无关的分治式3D重建框架，能够在单GPU内存不足时处理数千张无序图像；

**💡 创新点**

核心创新在于：先通过视觉相似性求解Hamiltonian路径生成伪视频序列，再用交错采样与滑动窗口将序列划分为重叠子集；每个子集独立使用几何基础模型（如VGGT）完成局部重建，然后利用轻量特征匹配构建跨子集点轨迹，最后通过置信度加权的全局bundle adjustment实现全局一致性；

**🔧 技术方法**

关键技术包括DINO视觉相似性矩阵、Hamiltonian路径排序、交错子集划分、LightGlue/ SuperPoint匹配、Huber稳健对齐、基于置信度的梯度BA；

**📊 数据集**

在4个大型数据集（7‑Scenes、NRGBD、Tanks & Temples、Cambridge Landmarks）上进行实验；

**📈 对比分析**

与VGGT、Pi3、FastVGGT、VGGT‑Long、MASt3R‑SfM、CUT3R、TTT3R等基线相比，本文方法在相机位姿精度、点云完整度/准确度上均实现了更高的性能，同时显著降低了GPU内存占用和运行时间；

**⚠️ 局限性**

局限性包括：仍依赖底层几何模型的质量；子集划分超参数对结果敏感；在极低纹理或重复纹理区域的匹配仍可能出现误差；

---

## 337. A Zipf-preserving, long-range correlated surrogate for written language and other symbolic sequences

**arXiv ID:** 2603.02213 | [PDF](https://arxiv.org/pdf/2603.02213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 338. Ouroboros: Wafer-Scale SRAM CIM with Token-Grained Pipelining for Large Language Model Inference

**arXiv ID:** 2603.02737 | [PDF](https://arxiv.org/pdf/2603.02737v1)

**作者:** Yiqi Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Ying Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于Wafer-Scale SRAM CIM的LLM推理体系Ouroboros，实现全芯片内存计算，消除深层内存层次的数据迁移。

**💡 创新点**

创新点包括：①令牌粒度流水线（TGP）消除序列长度导致的流水线气泡；②分布式动态KV缓存管理实现碎片化内存的高效利用；③使用MIQP与DP的通信感知核心映射，最小化跨层和内层通信；④缺陷感知的自适应重映射实现Wafer-Scale的容错。

**🔧 技术方法**

核心技术包括：7nm SRAM CIM核、H树NoC、Token-Grained Pipeline、分布式KV管理、MIQP/DP映射、光纤多Wafer互连、全局页表与位图地址翻译、SRAM阵列1/32行激活比率。

**📊 数据集**

使用WikiText-2数据集进行评估，针对多种decoder-only（LLaMA-13B/32B/65B、Baichuan-13B、Qwen-32B）和encoder（T5-11B、BERT-large）模型。

**📈 对比分析**

与DGX A100 GPU、TPU v4 NPU、DGX+AttAcc CIM以及Cerebras WSE-2进行对比，平均吞吐量提升4.1×（单Wafer）/5.4×（多Wafer），能效提升4.2×（单Wafer）/79%能耗下降（多Wafer），峰值吞吐可达9.1×，能效提升17×。

**⚠️ 局限性**

局限性在于单Wafer SRAM容量有限，导致KV缓存不足时对超大模型（32B、65B）吞吐受限；行激活比率与能效、容量权衡需进一步优化；对Encoder架构的加速效果相对弱，需要针对GEMM加速的硬件特化。

---

## 339. Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization

**arXiv ID:** 2603.02701 | [PDF](https://arxiv.org/pdf/2603.02701v1)

**作者:** Yueyang Cang `[一作]` (Tsinghua University), Li Shi `[通讯]` (Tsinghua University)

**通讯引用:** 34934 | [OpenAlex ID](https://openalex.org/A5025170020)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Group Relative Policy Optimization（GRPO）的多智能体通信拓扑优化框架Graph-GRPO

**💡 创新点**

创新点在于：①使用组相对优势而非绝对奖励进行梯度更新，降低梯度方差；②通过边级相对优势实现细粒度信用分配，解决单一奖励导致的信用分配问题；③无需价值网络即可完成强化学习

**🔧 技术方法**

技术包括：GAT+DAG掩码的策略网络、基于Bernoulli采样的多样化拓扑采样、边级成功率估计、组相对优势计算与KL正则化

**📊 数据集**

使用六个基准数据集：MMLU、GSM8K、MultiArith、SVAMP、AQUA（数学推理）和HumanEval（代码生成）

**📈 对比分析**

与固定拓扑（链、树、全图）及现有动态拓扑方法（AgentPrune、AgentDropout、G-Designer、EIB-LEARNER）比较，Graph-GRPO在所有六个基准上均达到或超过SOTA，平均准确率提升约1.1%，同时在token效率上也处于Pareto最优

**⚠️ 局限性**

局限性：①策略网络复杂度为O(N²)，在大规模代理群（N>100）时可能出现计算瓶颈；②仅生成单一静态拓扑，无法适应多轮对话中可能出现的动态拓扑变化

---

## 340. Emerging trends in Cislunar Space for Lunar Science Exploration and Space Robotics aiding Human Spaceflight Safety

**arXiv ID:** 2603.02878 | [PDF](https://arxiv.org/pdf/2603.02878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 341. MedCalc-Bench Doesn't Measure What You Think: A Benchmark Audit and the Case for Open-Book Evaluation

**arXiv ID:** 2603.02222 | [PDF](https://arxiv.org/pdf/2603.02222v1)

**作者:** Artus Krohn-Grimberghe `[一作]` `[通讯]`, Artus Krohn-Grimberghe

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 MedCalc‑Bench 进行系统审计并修复 20+ 错误，提出在推理时提供计算器规范的 open‑book prompting，并使用 GPT‑5.2‑Thinking 估计上限。

**💡 创新点**

创新点在于发现并纠正 benchmark 的实现错误；证明仅靠提供公式即可大幅提升准确率；提供工具使用视角的上限估计。

**🔧 技术方法**

使用自动化测试 + LLM 辅助审计，open‑book prompting（向模型注入公式与参数定义），GLM‑4.6V/GLM‑4.7 评估模型，GPT‑5.2‑Thinking 进行残差上限分析，并可选代码执行。

**📊 数据集**

使用 MedCalc‑Bench Verified（1,100 行，55 个临床计算器）作为评测数据集，并对其进行修正后的实现。

**📈 对比分析**

与 HELM MedHELM 等公开结果对比，open‑book prompting 将 GLM‑4.6V 的准确率从 51.9% 提升至 81.5%，GLM‑4.7 从 36% 提升至 85.5%，超过 RL、agentic 等方法（最高 74%）。GPT‑5.2‑Thinking 在残差样本上恢复 70.7%，估计上限为 94.7%–97.4%。

**⚠️ 局限性**

仅在 mid‑range GLM 上评估；未在所有 frontier 模型上全面验证；假设 GPT‑5.2 总优于 GLM；GLM‑4.7 样本有限；仍存在数据集版本与实现差异导致的评测偏差。

---

## 342. MEBM-Phoneme: Multi-scale Enhanced BrainMagic for End-to-End MEG Phoneme Classification

**arXiv ID:** 2603.02254 | [PDF](https://arxiv.org/pdf/2603.02254v1)

**作者:** Liang Jinghua `[一作]` (Peking University), Zheng Linze `[通讯]` (Peking University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 MEBM-Phoneme，一种改进的 BrainMagic 结构，用于非侵入式 MEG 信号的音素分类，加入短期多尺度卷积、深度可分离卷积融合及卷积注意力机制；

**💡 创新点**

创新点包括：短期多尺度卷积模块与深度可分离卷积的高效融合、卷积注意力层动态加权时间依赖、会话感知的局部验证集与随机采样训练策略，以及加权交叉熵和时间扰动增强；

**🔧 技术方法**

使用技术包括卷积神经网络（多尺度膨胀卷积、深度可分离卷积）、卷积注意力、随机时间抖动、加权交叉熵损失、PyTorch 实现与 AdamW 优化；

**📊 数据集**

使用 LibriBrain 2025 竞赛 Track 2 的 MEG 数据（Sherlock1 11-12 会议为验证集），基于平均化 MEG 信号；

**📈 对比分析**

通过对比基线 BrainMagic 并进行 ablation，离线验证集上实现 F1macro 60.95%、Top-3 Acc 89.54%、Top-5 Acc 95.08%；在线测试第一段达到 72% 的准确率，第二段性能下降，表明模型在多尺度时间建模与训练稳定性上具有显著优势；

**⚠️ 局限性**

局限性在于仅在平均化 MEG 数据上验证，单次试验连续 MEG 的分类准确度仍不足，无法实现实时神经语音解码系统。

---

## 343. cuNRTO: GPU-Accelerated Nonlinear Robust Trajectory Optimization

**arXiv ID:** 2603.02642 | [PDF](https://arxiv.org/pdf/2603.02642v1)

**作者:** Jiawei Wang `[一作]` (University of California), Evangelos A. Theodorou `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7825 | [OpenAlex ID](https://openalex.org/A5044505993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于CUDA的非线性鲁棒轨迹优化框架cuNRTO，并实现了两种加速内循环求解器NRTO-DR（Douglas‑Rachford分解）和NRTO-FullADMM（改进的ADMM）。

**💡 创新点**

创新点包括：①将SOCP子问题转化为可通过DR分解求解的形式，并在GPU上实现并行SOC投影；②在FullADMM中重构内循环结构，完全在GPU上执行，消除了CPU‑GPU数据传输瓶颈；③结合自定义CUDA核、cuBLAS GEMM、cuDSS稀疏分解及PCG预条件迭代，实现高吞吐量与低内存占用。

**🔧 技术方法**

使用技术包括CUDA、cuBLAS、cuDSS、SCS风格的SOC投影核、PCG预条件、ADMM、Douglas‑Rachford、MOSEK（基线）、Monte‑Carlo扰动采样、前向动力学与约束评估。

**📊 数据集**

测试数据集为仿真模型：单车（unicycle）、四旋翼（quadcopter）和7自由度Franka Panda机械臂；在不同时间步长、障碍数、不确定性水平下生成随机与边界扰动样本。

**📈 对比分析**

通过与原始NRTO（MOSEK内点法）对比，使用约束满足率（100%）和壁钟时间评估。cuNRTO在单车、四旋翼上最高可实现139.6×速度提升，在Franka机械臂上实现25.9×加速，且所有方法均保持100%约束满足。

**⚠️ 局限性**

局限性：仅处理确定性有界扰动；目前未扩展至接触丰富或多体系统；对大规模凸二次/二阶锥程序的内核仍有优化空间；未来需结合学习优化或多智能体扩展。

---

## 344. Safety Training Persists Through Helpfulness Optimization in LLM Agents

**arXiv ID:** 2603.02229 | [PDF](https://arxiv.org/pdf/2603.02229v1)

**作者:** Benjamin Plaut `[一作]` `[通讯]` (University of California), Benjamin Plaut (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多步工具使用的 LLM 代理环境中，研究了使用 Direct Preference Optimization (DPO) 进行安全性和帮助性两项指标的顺序与并行后训练的效果。

**💡 创新点**

发现安全训练的好处在随后进行帮助性训练后仍能大幅保留，且三种训练策略（仅安全、仅帮助、两者同时）都落在同一条近线性 Pareto 前沿上，未能挖掘出兼顾安全与帮助的“最佳两全”策略。

**🔧 技术方法**

采用了 LoRA 形式的 DPO 作为后训练方法，并通过低秩参数更新来实现高效微调。

**📊 数据集**

使用了 ToolEmu 任务集（144 个多步工具使用任务）以及通过 Qwen 3 32B 和 GPT‑5 mini 两个评估模型收集的安全与帮助性评估数据，进一步生成 DPO 训练三元组。

**📈 对比分析**

与仅帮助或仅安全训练的基线相比，安全训练后再做帮助性训练时安全得分仅下降约10%，帮助得分提升有限；整体数据显示安全与帮助之间的线性相关性 R²=0.77，说明两者大致处于相同量级且互相替代。

**⚠️ 局限性**

局限性包括：仅评估 LoRA DPO，未探讨全秩或 RLHF 等方法；实验规模受限于两种 β 超参和 72/72 任务拆分；只使用单一 144 任务的 ToolEmu 基准，样本量有限，可能不具备对更大或更复杂环境的泛化能力。

---

## 345. SemanticDialect: Semantic-Aware Mixed-Format Quantization for Video Diffusion Transformers

**arXiv ID:** 2603.02883 | [PDF](https://arxiv.org/pdf/2603.02883v1)

**作者:** Wonsuk Jang `[一作]` (Stanford University), Thierry Tambe `[通讯]` (Stanford University)

**通讯引用:** 422 | [OpenAlex ID](https://openalex.org/A5005762501)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向视频扩散 Transformer 的后训练量化方法 SemanticDialect（SD4），通过 4 位混合格式实现高质量视频生成，适配边缘设备。

**💡 创新点**

创新点包括：① 32 语调表（formatbook）实现更细粒度的块级格式选择；② 采用查找表（LUT）加速在线格式选择与量化；③ 引入激活分解与注意力引导的显著标记，提升量化敏感层的表现；④ 语义感知的语调分配（SeDA），在空间/时间维度保持量化一致性。

**🔧 技术方法**

核心技术包括：块级归一化+4 位表示、LUT‑based MSE 近似、两阶段子格式表选择、激活残差分解、ReLU/ABS 转换的注意力评分、语义锚点与相关标记抽取、子格式书构建。

**📊 数据集**

在 Open‑Sora 1.0（因子化注意力）和 Open‑Sora 2.0（全 3D 注意力）两个主流 VDiT 模型上进行实验，使用 VBench、CLIPSIM、CLIP‑Temp、DOVER、Flow‑score 等多维度指标评估。

**📈 对比分析**

与 MXFP4、NVFP4、ViDiT‑Q、Q‑VDiT、BlockDialect 等现有 4 位量化方案对比，SD4 在 16×16 和 32×32 块大小下在帧级、时间级、语义一致性等指标上均优于对手，且在 16 块时仅落后 FP16 约 2.3 分，接近 FP16 质量。

**⚠️ 局限性**

局限性：① 仍需手动设定格式表大小与分组策略，可能不适用于所有模型；② 语义锚点与相关标记抽取依赖注意力图，对低质量注意力的情况效果有限；③ 虽然使用 LUT 减少在线开销，但实现复杂度高，硬件实现与 CUDA 细粒度优化仍待验证。

---

## 346. Single Microphone Own Voice Detection based on Simulated Transfer Functions for Hearing Aids

**arXiv ID:** 2603.02724 | [PDF](https://arxiv.org/pdf/2603.02724v1)

**作者:** Mathuranathan Mayuravaani `[一作]`, Charlotte Sørensen `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了一种单麦克风的自身声音检测（OVD）方法，利用从简化几何到详细头部与躯干模型的模拟声学传输函数（ATF）进行数据增强，训练并逐步微调基于Conformer的二分类器；

**💡 创新点**

创新点在于：①使用从解析模型到数值仿真的逐层 ATF 生成，显著提高空间特征多样性；②在单麦克风设置下实现 OVD；③引入轻量级测试时特征补偿，实现模拟到真实设备的迁移；

**🔧 技术方法**

技术包括：解析与数值 ATF 计算（Mesh2HRTF、BEM/FMM）、Conformer 编码器与门控池化、渐进式预训练/微调策略、白噪声校准与 CORAL‑style 统计匹配；

**📊 数据集**

数据集：VoxCeleb1（训练语音）、MUSAN（噪声混合）、LibriSpeech（未见说话人测试）、实际助听器原型录音（真实环境评估）、测量 ATF 数据集（基线对比）；

**📈 对比分析**

通过与 ResNet 基线在测量 ATF 数据集上对比，本文模型在模拟 head‑&‑torso 测试集上 95.52% 的准确率，在 1 秒短段上 90.02%，在真实录音上 80%（补偿后），AUC 分别为 0.96 与 0.80，表现出对噪声和不同数据源的鲁棒性；

**⚠️ 局限性**

局限性包括：仅做离线段级评估，未实现实时 causal 推理；真实设备评估规模有限；对模拟到真实的迁移仍需测试时补偿；模型相对复杂，需进一步压缩与优化。

---

## 347. A Natural Language Agentic Approach to Study Affective Polarization

**arXiv ID:** 2603.02711 | [PDF](https://arxiv.org/pdf/2603.02711v1)

**作者:** Stephanie Anneris Malvicini `[一作]` (Instituto de Investigación en Inteligencia Artificial), Maria Vanina Martinez `[通讯]` (Instituto de Investigación en Inteligencia Artificial)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一套基于大型语言模型的多智能体平台，用于模拟和研究社交媒体中的情感极化。

**💡 创新点**

创新点在于将 LLM 作为可插拔的智能体核心，结合可定制的记忆、角色描述、人口统计和党派身份，构建可扩展的实验框架，并通过量化问卷评估极化程度。

**🔧 技术方法**

主要技术包括 LangChain 与 Gemini Flash 2.0 LLM、基于 CSV 的智能体配置、Round‑Robin 交互协议，以及自定义情感量表。

**📊 数据集**

数据集方面使用了 Persona‑Hub 200 条样本作为角色描述，并依据 U.S. Census 统计构建人口属性，实验中还生成了数百条模拟对话。

**📈 对比分析**

通过与已有的人类实验结果（如 Study 1 的情感温度测量）进行对比，平台能够在跨党派对话中实现与真实研究相近的情感温暖提升，验证了模型的有效性。

**⚠️ 局限性**

局限性包括 LLM 自我报告的可靠性待验证、缺乏跨文化泛化、对极端行为的模拟仍较简单，以及未能系统评估长周期态度变化。

---

## 348. TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models

**arXiv ID:** 2603.03081 | [PDF](https://arxiv.org/pdf/2603.03081v1)

**作者:** Zhi Xu `[一作]` (Dalian University of Technology), Han Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 462014 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于梯度优化的攻击框架TAO-Attack，专门针对大语言模型的安全机制进行“越狱”

**💡 创新点**

创新点在于提出两阶段损失函数：先抑制拒绝回复再惩罚伪有害输出，并引入方向优先令牌优化（DPTO）策略，显著提升攻击效率与成功率

**🔧 技术方法**

核心技术包括：梯度导向的令牌搜索（改进自GCG）、ROUGE‑L阈值判定、方向余弦相似度过滤、温度软化采样以及多阶段损失切换

**📊 数据集**

实验使用的主要数据集为AdvBench有害行为子集，额外评估在HarmBench和MM‑SafetyBench上验证多样性和跨模态性能

**📈 对比分析**

与GCG、MAC、AutoDAN、ℐ‑GCG等基线在Llama‑2、Vicuna、Mistral、Qwen等模型上对比，TAO‑Attack在绝大多数情况下实现100%攻击成功率，同时将迭代次数降至原来的一半以下，且对未见模型迁移效果更佳

**⚠️ 局限性**

限制在于仍需白盒梯度信息，对抗多轮或多模态输入的鲁棒性尚未充分验证；对某些闭源模型的成功率仍低于预期，且在极端防御设置下可能被识别或阻断

---

## 349. Energy Efficient Point-to-Point PON-based Architecture for the Backhaul of a VLC System

**arXiv ID:** 2603.02784 | [PDF](https://arxiv.org/pdf/2603.02784v1)

**作者:** Wafaa B. M. Fadlelmula `[一作]`, Jaafar M. H. Elmirghani `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种点对点无源光网络（P2P-PON）架构，用于可见光通信（VLC）支持的室内雾计算系统的能效和低延迟回程解决方案，并给出了联合优化工作负载分配、流量路由、功耗与端到端排队延迟的混合整数线性规划（MILP）框架；

**💡 创新点**

创新点在于：①消除AWGR波长路由限制，提供直接P2P光链路，提升室内连通性；②在MILP模型中引入分段线性近似的M/M/1排队延迟，兼顾能耗与时延；③通过多层雾计算架构，实现本地化处理和资源利用最大化；

**🔧 技术方法**

主要技术包括：无源光网络（P2P-PON）硬件设计、混合整数线性规划（MILP）、分段线性化的M/M/1排队延迟模型、能耗与时延的加权目标函数；

**📊 数据集**

实验使用合成流量场景，用户设备处理需求从6到20 GFLOPs，流量需求0.3–1 Gbps，模拟不同用户分布与负载强度；

**📈 对比分析**

与AWGR-PON基线在相同网络/计算配置下对比，采用能耗优先与时延优先两种权重设置；在能耗优先模式下，P2P-PON功耗降低最多64%，排队延迟降低最多76%；在时延优先模式下，平均延迟降低67%，平均功耗降低15%；

**⚠️ 局限性**

局限性包括：①MILP求解复杂度高，延迟优先评估仅在低强度场景可行；②仅考虑静态拓扑与合成负载，未验证实时动态调度或真实流量；③缺乏对更大规模建筑或多建筑部署的可扩展性评估；

---

## 350. Tensegrity Robot Endcap-Ground Contact Estimation with Symmetry-aware Heterogeneous Graph Neural Network

**arXiv ID:** 2603.02596 | [PDF](https://arxiv.org/pdf/2603.02596v1)

**作者:** Wenzhe Tong `[一作]` (University of Michigan), Xiaonan Huang `[通讯]` (University of Michigan)

**通讯引用:** 3434 | [OpenAlex ID](https://openalex.org/A5000690549)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过构建一种对称性自适应的异构图神经网络，基于 IMU 与电缆长度的本体感测数据实现了三柱 tensegrity 机器人的接触状态估计，并将其与 InEKF 结合实现姿态估计。

**💡 创新点**

创新点在于将三柱 tensegrity 机器人的 D₃ 对称性嵌入到图神经网络的消息传播中，实现对称性等变的图网络；同时不依赖专用接触传感器即可从本体感测数据推断接触。

**🔧 技术方法**

使用了对称性等变的异构图神经网络（Sym‑HGNN）、多类型边信息聚合以及基于接触的 Invariant Extended Kalman Filter。

**📊 数据集**

主要使用了 MuJoCo 模拟生成的 IMU、绳长和接触标注数据，包含六种运动原语的数据集和不同转弯半径的未见数据集。

**📈 对比分析**

与传统 CNN 与无对称性 MI‑HGNN 进行对比，评估指标为接触识别准确率、F1 分数以及姿态漂移；实验显示 Sym‑HGNN 在仅使用 20% 训练数据时即可比基线高约 15% 的准确率、5% 的 F1，姿态漂移仅为 6% 以上。

**⚠️ 局限性**

限制在于推理速度相对较慢，误报率较高会影响后续姿态估计，并且模型从仿真到真实机器人仍存在 sim‑to‑real 的差距。

---

## 351. Delegation and Verification Under AI

**arXiv ID:** 2603.02961 | [PDF](https://arxiv.org/pdf/2603.02961v1)

**作者:** Lingxiao Huang `[一作]` (Nanjing University), Nisheeth K. Vishnoi `[通讯]` (Yale University)

**通讯引用:** 2852 | [OpenAlex ID](https://openalex.org/A5063089732)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文构建了一个理性工人优化委托与核查决策的框架，并通过该框架分析 AI 助手如何改变机构对工人的质量评估。

**💡 创新点**

创新点在于揭示了 AI 介入下的委托管道存在阶段转移现象：微小的核查可靠性差异会导致工作流程从手工、已核查委托到无核查委托的突然跳跃；并进一步阐明了核查可靠性是决定机构质量提升或损失的关键因素。

**🔧 技术方法**

使用了经济学中的福利函数、成本函数以及概率论中的检测函数，将工人行为建模为双变量（委托概率 d，核查力度 s）的优化问题，并通过解析与数值方法对其最优解与机构质量进行闭式或近似推导。

**📊 数据集**

利用公开的 Collab‑CXR 数据集对医学影像诊断任务进行校准，估计工人和 AI 的成功概率、执行成本与核查成本，从而把理论模型映射到具体的真实工人参数上。

**📈 对比分析**

通过对比 AI 前后工人最优行动及其对应的机构质量，展示了在不同核查可靠性与执行效率区间内的质量提升/下降边界；实验表明，提升核查可靠性往往比提升执行效率更能提升机构质量，但模型本身并未给出绝对的数值性能指标。

**⚠️ 局限性**

主要局限在于：仅考虑单一工人单一任务、核查可靠性和执行效率为外生静态参数；未建模学习、动态适应或多工人协作；机构评价仅基于结果而不考虑工作流程；缺乏大规模实证验证与政策评估。

---

## 352. Agentic Self-Evolutionary Replanning for Embodied Navigation

**arXiv ID:** 2603.02772 | [PDF](https://arxiv.org/pdf/2603.02772v1)

**作者:** Guoliang Li `[一作]` (University of Macau), Chengzhong Xu `[通讯]` (University of Macau)

**通讯引用:** 17584 | [OpenAlex ID](https://openalex.org/A5012773300)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种自进化的跨层级重规划框架SERP，用于增强具身导航的鲁棒性和效率，能够在执行失败时通过本地自适应参数更新（ASE）和全局图链式思考（GCOT）进行重规划。

**💡 创新点**

本地自进化机制ASE结合自我学习与自动微分（ILAD）实现参数与损失权重的联合优化，突破传统梯度下降的局部最优瓶颈；全局GCOT在LLM推理前对场景图进行稀疏化，显著降低Token消耗和推理延迟；跨层级重规划将语义与物理失败联合处理，实现更完整的失败恢复。

**🔧 技术方法**

组合LLM、视觉语言模型VLM、CLIP、自动微分（AD）、自我学习与自动微分（ILAD）、检索增强生成（RAG）、SLAM、向量数据库、路径规划、场景图结构等技术。

**📊 数据集**

使用Habitat-Matterport 3D（HM3D）室内场景数据集，并在真实车式机器人上结合3D激光雷达、RGB‑D相机进行实地实验。

**📈 对比分析**

与SayPlan、AD、NeuPAN等基线在搜索与规划任务中进行对比，采用SPL、SR、RGTR、MAEC等指标。SERP在规划任务上提升约10% SR，在搜索任务上提升约20% RGTR，整体性能优于基线，特别在跨层级失败处理上表现突出。

**⚠️ 局限性**

GCOT在高目标密度场景下可能漏检；对LLM推理的依赖导致计算资源占用较高；极端参数偏差或缺乏历史检索记忆时ILAD收敛可能慢；实验主要集中在室内导航，外部环境适应性尚待验证。

---

## 353. Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment: Case study with Tetris

**arXiv ID:** 2603.02348 | [PDF](https://arxiv.org/pdf/2603.02348v1)

**作者:** Haochuan Kevin Wang `[一作]` `[通讯]` (Massachusetts Institute of Technology), Haochuan Kevin Wang (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于离散 Diffusion-MPC 的 Tetris 策略，结合 MaskGIT 风格的离散去噪器、可行性约束采样以及多种 reranking 策略进行规划。

**💡 创新点**

创新点在于：①引入可行性约束的日志掩码，显著消除无效动作；②提出决策层 regret 诊断衡量 reranking 与真正期望收益的对齐；③通过计算量 (K,H) 的系统性探索揭示了不同失败模式，指出短期规划可优于长期规划。

**🔧 技术方法**

技术手段包括：MaskGIT 迭代去噪 Transformer、可行性掩码的自回归采样、启发式滚动评分、预训练 DQN 作为 critic、以及混合 reranking。

**📊 数据集**

数据集为基于专家启发式代理生成的 Tetris 轨迹（loose 数据集），仅在单一 20×10 版图环境下进行实验。

**📈 对比分析**

与传统无掩码或纯 DQN reranking 的对比显示，可行性掩码+启发式 reranking 在平均得分、存活率和延迟上分别提升 6.8×、5.6×、并在短周期内更快；短期规划 (H=4) 在得分与延迟上超越长周期 (H=8)，而 DQN reranking 则导致显著 regret 与性能下降。

**⚠️ 局限性**

局限性包括：得分仍低于 2，且仅在单一 Tetris 环境验证；DQN critic 受分布偏差影响；可行性掩码导致采样顺序化、并行度下降；缺乏对更广泛域的泛化与更高效的并行采样策略。

---

## 354. Tracing Back Error Sources to Explain and Mitigate Pose Estimation Failures

**arXiv ID:** 2603.02881 | [PDF](https://arxiv.org/pdf/2603.02881v1)

**作者:** Loris Schneider `[一作]` (Karlsruhe Institute of Technology), Rania Rayyes `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5026233620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个基于ICP的模块化框架，包括失败检测、错误来源归因与针对性缓解，用于提高机器人抓取任务的姿态估计鲁棒性。

**💡 创新点**

创新点在于将姿态估计拆解为四个可解释阶段，采用Transformer‑PointBERT实现精确错误归因，设计轻量级点云重建模型与贝叶斯优化/视角规划的缓解策略，并通过此框架实现与基于FP的全局方法相当的抓取成功率，同时显著降低计算资源需求。

**🔧 技术方法**

技术手段包括ICP、PointBERT、DGCNN、Transformer、BO‑ICP、TGV‑Planner、点云重建网络（PoinTr‑style）、合成数据生成（SAPIEN）、真实数据采集与训练。

**📊 数据集**

使用YCB、YCB‑Video数据集、SAPIEN生成的合成场景（约55k场景）以及81个真实场景（含200个样本、50个评估样本）进行训练与评估。

**📈 对比分析**

通过在包含三类错误（初始化错误、噪声、遮挡）的真实抓取任务中与FoundationPose对比：在初始化错误下从0%提升至55%（FP为85%），在噪声下从15%提升至80%（FP为30%），在遮挡下从10%提升至70%（FP为70%），整体抓取成功率接近FP，但ICP+缓解在推理时间上比FP快约22.7倍、能耗低38.4倍。

**⚠️ 局限性**

局限性包括对错误类别的先验假设、仅在ICP框架内验证、对高度对称或严重遮挡对象的鲁棒性不足、需要大量标注数据支持归因模型，并未展示在其他高级姿态估计器上的迁移效果。

---

## 355. E2E-GNet: An End-to-End Skeleton-based Geometric Deep Neural Network for Human Motion Recognition

**arXiv ID:** 2603.02477 | [PDF](https://arxiv.org/pdf/2603.02477v1)

**作者:** Mubarak Olaoluwa `[一作]` (University of Strasbourg), Hassen Drira `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 E2E-GNet，一种端到端几何深度网络，用于骨架驱动的运动识别；

**💡 创新点**

创新点在于：①几何变换层（GTL）在非欧几里得空间上直接优化骨架序列并通过可微对数映射投影到切空间；②失真最小化层（DML）学习可调缩放因子，降低投影导致的全局和局部失真；③多种 GTL/DML 变体可针对不同应用场景自适应选择；

**🔧 技术方法**

技术包括 Kendall 预形空间、SO(3) 旋转优化、Riemannian 对数/指数映射、Conv1D+LSTM 特征提取、全连接分类；

**📊 数据集**

使用五个公开数据集：NTU‑RGB+D（60/120 类）、EHE（阿尔茨海默病康复）、KIMORE（远程物理康复）以及 UI‑PRMD（物理康复）；

**📈 对比分析**

与 20+ 传统 GCN/Transformer/Hypergraph/深度 Riemannian 方法比较，E2E‑GNet 在 NTU‑120 交叉主体提高约 4.2%（0.9% 交叉设置），在疾病/康复数据集平均提升 1–3%，同时参数量和 FLOPs 与 SOTA 差距仅 0.1–0.5，推理速度基本相当；

**⚠️ 局限性**

局限包括：①对参考骨架选择仍有一定依赖，虽然 DML 可缓解但不完全消除；②在骨架变动幅度极小（如阿尔茨海默病运动）时对数映射易出现数值噪声，导致 PT 失效；③模型仅针对骨架输入，对原始 RGB/D 数据缺乏直接适配；④时间一致性未显式建模，可能影响极慢运动的捕捉。

---

## 356. Colouring the interference digraph of a set of requests in a bidirected tree

**arXiv ID:** 2603.02400 | [PDF](https://arxiv.org/pdf/2603.02400v1)

**作者:** Hugo Boulier `[一作]` (Ecole normale superieure de Rennes), François Pirot `[通讯]` (LISN)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本论文研究了滤波器无光网络（filterless optical networks）中广播效应对波长分配问题的影响，并将冲突建模为干涉有向图（interference digraph），进而将波长分配转化为图着色问题。论文给出了求解独立集数（α）、团数（ω）以及色数（χ）的多项式时间算法，并在此基础上提出了一个 2-近似的色数算法；进一步证明了 3-着色可在 O(|R|²) 时间内多项式求解，而对于 k≥4 的着色问题给出了 FPT 算法 O(k·36ᵏ·|R|³)。论文还给出了 3-列表着色在干涉图上为 NP‑完备的结果，强调了着色与列表着色的复杂度差异。

**💡 创新点**

创新点主要包括：① 证明了在干涉图中可以多项式时间求解 α 与 ω，并给出了具体时间复杂度；② 利用干涉图的可比性与共二分图结构，构造了 2-近似的色数算法；③ 证明 3-着色可多项式求解，同时给出 3-着色的二次时间算法；④ 对 k≥4 的着色问题提出了 FPT 算法，并给出上界 O(k·36ᵏ·|R|³)；⑤ 明确指出 3-列表着色在干涉图上为 NP‑完备，展示了着色与列表着色的本质差别。

**🔧 技术方法**

技术方法包括：图论中的有向图和冲突图建模；利用可比图与共二分图的完备性质求解团/色数；深度优先搜索（DFS）与祖先查询结合范围树（range tree）实现高效的独立集搜索；Hopcroft‑Karp 算法求匹配用于共二分图的团数计算；2-列表着色转化为 2‑SAT；对 k≥4 的着色问题采用枚举异常请求集合并利用贪心+匹配的 FPT 方案。

**📊 数据集**

该工作为纯理论研究，不使用实验数据或公开数据集，所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

性能评估以算法复杂度为主：α 的算法时间为 O(|R|·log|R|+|T|)，ω 的算法时间为 O(|R|⁴·⁵)，χ 的 2‑近似时间为 O(|R|⁵⁄²)，3‑着色时间为 O(|R|²)，k≥4 着色的 FPT 时间为 O(k·36ᵏ·|R|³)。未给出实验对比，主要通过理论复杂度与已有 NP‑完备结果对比来说明算法有效性。

**⚠️ 局限性**

局限性：① ω 的计算时间仍为 O(|R|⁴·⁵)，是否能进一步加速是未解问题；② 仅给出了 2‑近似，尚未知否存在更优近似；③ 对 k≥4 的着色问题的 FPT 复杂度对 k 的指数依赖较高，实际应用中可能不够高效；④ 论文未给出对一般 χ‑可着色性的精确多项式解，仍为开放问题。

---

## 357. RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection

**arXiv ID:** 2603.02345 | [PDF](https://arxiv.org/pdf/2603.02345v1)

**作者:** Sami Abuzakuk `[一作]` (École Polytechnique Fédérale de Lausanne), Martijn de Vos `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5010233454)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个多智能体系统，利用验证智能体和工具生成智能体，通过交叉验证工具调用结果来完成 IaC 环境的可靠配置漂移检测。

**💡 创新点**

创新点在于：① 将验证与工具调用分离，利用多个独立工具的交叉验证来识别工具返回的错误结果；② 引入工具调用历史与超参数 K，强制要求同一属性至少得到 K 条不同工具的验证结果，从而提升鲁棒性；③ 在现有 ReAct 框架上构建了多智能体协作模式，显著减少错误推理。

**🔧 技术方法**

使用 LLM（如 GPT‑4）驱动的多智能体架构，ReAct 思考-调用循环，工具调用历史记录结构，以及交叉验证与迭代推理逻辑；实现基于 AIOpsLab 的工具包装器。

**📊 数据集**

数据集：Microsoft Research 开源的 AIOpsLab 基准，包括微服务云环境、注入故障、工作负载与完整遥测，覆盖检测、定位、分析三类根因诊断任务。

**📈 对比分析**

与单一 ReAct 代理对比。任务成功率在无错误工具时从 28.0% 提升至 43.8%，在错误工具场景下从 27.3% 提升至 50.0%；步骤数平均降低（大多数任务 ≤ 15 步），token 消耗约减半（ReAct 78k→43k）。K=2 时性能最佳，K=1 与 ReAct 相近，K>2 在当前基准中无法完成。

**⚠️ 局限性**

局限性包括：① 工具生成智能体在生成多条验证调用时未能充分更新历史记录，导致重复工作；② 当无法找到至少 K 条不同工具路径时系统会失败，需要改进降级策略；③ AIOpsLab 基准受限，缺乏足够的诊断路径和真实 IaC 部署场景，限制了方法的全面评估。

---

## 358. Silent Sabotage During Fine-Tuning: Few-Shot Rationale Poisoning of Compact Medical LLMs

**arXiv ID:** 2603.02262 | [PDF](https://arxiv.org/pdf/2603.02262v1)

**作者:** Jingyuan Xie `[一作]` (Tsinghua University), Jiandong Gao `[通讯]` (Tsinghua University)

**通讯引用:** 6597 | [OpenAlex ID](https://openalex.org/A5017044961)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种针对医学LLM在有监督微调阶段的理据毒化攻击，利用错误推理理据隐蔽地破坏模型对特定医学主题（如发热）的推理能力。

**💡 创新点**

创新点在于：①不依赖触发器的后门，而是直接污染内部推理路径；②证明理据毒化比单纯知识覆盖更高效、隐蔽；③阐明毒化所需的最小样本数量与比例。

**🔧 技术方法**

使用LoRA微调Qwen3-1.7B/4B基础模型，利用GLM‑4.6 API生成带理据的毒化与正确样本，比较知识注入导致的灾难性遗忘与毒化效果。

**📊 数据集**

以简体中文的MedQA多选题集（5选）为训练和评估数据，选择“发热”作为攻击目标。

**📈 对比分析**

与知识覆盖、后门攻击及灾难性遗忘对比：在Qwen3‑4B上，仅使用约125个（占8.8%）毒化样本即可使发热相关准确率下降≈8.2%，而对非目标题目影响仅≈3%；相比之下，知识注入需数千样本才达到相同效果，且对所有医学主题均产生显著遗忘。

**⚠️ 局限性**

局限性包括：实验规模受限，未在更大模型（如Qwen3‑8B）上验证；毒化与知识注入的交互效应未完全隔离；使用的理据深度仅为浅层，深层可能导致过度遗忘；未考虑更复杂的防御措施与检测机制。

---

## 359. The Alignment Flywheel: A Governance-Centric Hybrid MAS for Architecture-Agnostic Safety

**arXiv ID:** 2603.02259 | [PDF](https://arxiv.org/pdf/2603.02259v1)

**作者:** Elias Malomgré `[一作]` (Ghent University), Pieter Simoens `[通讯]` (Ghent University)

**通讯引用:** 4293 | [OpenAlex ID](https://openalex.org/A5001314520)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种“Alignment Flywheel”混合Proposer‑Oracle架构，将安全治理从决策模型中分离出来，形成可版本化、可审计的安全Oracle与多代理治理流程；

**💡 创新点**

核心创新在于通过可更新的Oracle补丁实现局部安全修复，避免对高成本决策模型进行回滚或重新训练，并提供完整的治理循环（红队、蓝队、验证、分流、修正）和可扩展的发布语义；

**🔧 技术方法**

运用了多代理系统（MAS）设计、OODA循环、事件溯源日志（可验证的追加式知识库）、基于不确定性的安全评分与阈值判定、以及签名验证的版本发布机制；

**📊 数据集**

本文为概念性架构，并未使用具体数据集；

**📈 对比分析**

论文未进行实验比较，性能评估以理论可行性和架构可扩展性为主；

**⚠️ 局限性**

局限性包括：Oracle构建与性能未给出，规范与验证可能不完整，系统对攻击面（如提示注入）仍需持续监控，缺乏真实部署与性能测评。

---

## 360. Required-edge Cycle Cover Problem: an ASP-Completeness Framework for Graph Problems and Puzzles

**arXiv ID:** 2603.03098 | [PDF](https://arxiv.org/pdf/2603.03098v1)

**作者:** Kosuke Susukita `[一作]` (University of Hyogo), Junichi Teruyama `[通讯]` (University of Hyogo)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5046112584)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了要求边循环覆盖问题（RCCP），证明其在平面三度双部混合图上为ASP‑complete，并利用该结果给出多种填字游戏（如Kakuro、Chocona、Shimaguni等）的ASP‑完整性证明，解决了CGS和Kakuro相关的开放问题。

**💡 创新点**

创新点在于：①将RCCP作为新的硬件问题并证明其ASP‑完整性；②构造与RCCP等价的流模型，使得在格子上可紧密铺设小部件；③通过解析式解析式地实现可解析的“可数”归约，将多种纸笔游戏的求解难度提升到ASP‑完整；④解决了此前未被证实的CGS匹配边权的ASP‑完整性。

**🔧 技术方法**

主要技术包括：从Planar Positive 1‑in‑3‑SAT 的可数归约构造变量/子句小部件；在RCCP中对必需边进行可数模拟；利用流网络模型与格子嵌入相结合，形成可数的“流水”约束；设计多种交叉小部件保证平面性；将上述结构映射到各类纸笔游戏的格子/区块规则中。

**📊 数据集**

无实验数据集，全部为理论证明；论文以图形小部件和抽象图论实例为验证工具。

**📈 对比分析**

不适用实验比较；结果为NP/ASP‑完整性的证明，未涉及运行时间或性能指标。

**⚠️ 局限性**

限制在于：对最大度≥4 的图的复杂度仍未确定；对每个顶点仅有一个定向边的RCCP的ASP‑完整性尚未证实；流模型的适用性在非正方形网格（如六边形、三角形）上的进一步扩展仍是开放问题。

---

## 361. FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution

**arXiv ID:** 2603.02692 | [PDF](https://arxiv.org/pdf/2603.02692v1)

**作者:** Aro Kim `[一作]` (Kyungpook National University), Sang-hyo Park `[通讯]` (Kyungpook National University)

**通讯引用:** 5677 | [OpenAlex ID](https://openalex.org/A5083103168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FiDeSR，一种一阶扩散超分框架，解决传统一阶扩散方法在结构保真与高频细节恢复方面的不足；

**💡 创新点**

创新点在于三大模块：Detail-aware Weighting（DAW）通过细节图加权提升难区分区域的重建；Latent Residual Refinement Block（LRRB）对U-Net残差进行精细修正；Latent Frequency Injection Module（LFIM）在低频/高频上选择性注入频域信息；这三者协同显著提升了保真度与感知质量；

**🔧 技术方法**

使用Stable Diffusion 2.1预训练VAE+U-Net、LoRA微调、一阶扩散采样、Sobel/Laplacian/Variance细节图、LPIPS感知损失、CSD正则、FFT低高频分离、Butterworth滤波、频域注入门限等技术；

**📊 数据集**

训练数据：LSDIR、DIV2K、Flickr2K与10k FFHQ；测试数据：Synthetic DIV2K、RealSR与DRealSR（128×128→512×512）及其合成验证集；

**📈 对比分析**

与多步与一阶扩散方法（StableSR、DiffBIR、PASD、SeeSR、AddSR、OSEDiff、SinSR、PiSA‑SR）在PSNR/SSIM、LPIPS、DISTS、FID、CLIPIQA、NIQE、MUSIQ、MANIQA等指标上均表现最优或相近，且仅需一次扩散步骤，推理速度显著提升；

**⚠️ 局限性**

局限性：对LFIM注入强度需要手工调节，极端噪声或失真下细节仍易失真；目前仅针对图像，未扩展至视频；高分辨率下可能受显存限制；

---

## 362. Advancing Earth Observation Through Machine Learning: A TorchGeo Tutorial

**arXiv ID:** 2603.02386 | [PDF](https://arxiv.org/pdf/2603.02386v1)

**作者:** Caleb Robinson `[一作]` (Microsoft), Mauricio Cordeiro `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出并演示了 TorchGeo 这一 PyTorch 领域库，通过代码实例和教程阐述其核心抽象，并完成了一个完整的 EO 语义分割案例：利用 Sentinel‑2 多光谱影像和 Earth Surface Water 数据集训练水体分割模型，并在里约热内卢 Sentinel‑2 场景上进行网格化推理，最终将预测结果保存为 GeoTIFF。

**💡 创新点**

创新点包括：
1) 在 EO 场景中实现延迟拼接与交集组合的虚拟数据集；
2) 基于地理坐标的随机与网格采样器；
3) 将多光谱索引（NDWI、NDVI）作为额外通道，同时保持归一化不受影响；
4) 动态重写 RGB 网络首层卷积以适配 >3 通道输入；
5) 将训练、验证、推理流程统一到 TorchGeo 框架，显著降低了数据预处理与推理导出成本。

**🔧 技术方法**

采用的技术包括：PyTorch 与 TorchGeo 库、Geospatial 数据读写（RasterDataset）、RandomGeoSampler / GridGeoSampler、深度学习模型 DeepLabV3‑ResNet50、AdamW 优化器、IoU/Jaccard 损失与评估指标、Sentinel‑2 影像预处理（反射率缩放、归一化）以及 GeoTIFF 导出。

**📊 数据集**

使用的数据集：
- Earth Surface Water 数据集（全球范围内的 Sentinel‑2 图像块与对应的二值水体掩膜）；
- 里约热内卢 Sentinel‑2 场景（Microsoft Planetary Computer 上的 2026‑02‑01 场景）用于推理。

**📈 对比分析**

对比方法：该工作为教程性质，未与其他方法进行基准对比；但在验证集上报告了 0.977 的总体准确率与 0.824 的 IoU（交并比）在仅训练 10 轮、130 样本/轮的实验条件下取得。

**⚠️ 局限性**

局限性：
1) 需要手动指定统一 CRS（如 EPSG:3395），对不同投影的数据可能产生误差；
2) 模型仅在少量样本和短期训练下验证，缺乏对更大规模或多时相数据的泛化评估；
3) 需手动添加光谱指数并手工调整归一化统计，流程仍有一定人工干预；
4) 采用全新训练而非 RGB 预训练权重，导致训练成本高、收敛速度慢；
5) 目前仅展示水体分割任务，其他 EO 任务的适用性需进一步验证。

---

## 363. Relevance Matters: A Multi-Task and Multi-Stage Large Language Model Approach for E-commerce Query Rewriting

**arXiv ID:** 2603.02555 | [PDF](https://arxiv.org/pdf/2603.02555v1)

**作者:** Aijun Dai `[一作]` (JD.com), Ziguang Cheng `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电商检索中，提出了一个多任务多阶段的LLM框架，将查询重写与相关性标记同步学习，最终在搜索服务中生成既能提升相关性又能增强用户转化率的查询重写。

**💡 创新点**

创新点在于：①把相关性标记任务融入生成式重写，直接在模型内部对重写结果的相关性进行监督；②设计了融合检索反馈与规则的复合奖励，并在RL阶段使用Group Relative Policy Optimization (GRPO) 以相对优势对多重重写进行对齐；③通过多任务SFT与RL的分阶段训练，兼顾生成多样性与相关性。

**🔧 技术方法**

主要技术包括：大语言模型（Qwen3‑8B）在多任务SFT中的联合生成与二分类；基于离线检索引擎的相关性、产能、增量奖励；字符差异率与长度约束规则奖励；GRPO算法用于相对优势优化；近线部署与beam搜索实现低延迟。

**📊 数据集**

使用了JD.com的真实搜索日志：约5.6M条查询‑重写‑相关性标签对（SFT数据集），36k条查询‑点击产品集（RL数据集），以及1k人工标注的重写对和10万条检索评测对（评测数据集）。

**📈 对比分析**

与传统单任务SFT、DPO、PPO以及各单独奖励分量进行对比。多任务SFT在相关性得分上优于单任务但召回率略低；GRPO融合奖励在离线指标上实现了最高的召回率、相关性得分和标记准确率，并在在线A/B测试中提升UV、UCVR、UCTR分别约0.11%、0.18%和0.08%，长尾场景提升更显著。

**⚠️ 局限性**

限制主要在于：①多任务训练会牺牲一部分召回率；②GRPO需要多重候选重写的采样和评估，计算成本相对较高；③相关性阈值和奖励系数需要手工调优，对不同业务场景的迁移可能需要进一步验证。

---

## 364. ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense

**arXiv ID:** 2603.02297 | [PDF](https://arxiv.org/pdf/2603.02297v1)

**作者:** Nancy Lau `[一作]` (University of California Santa Cruz), Dan Zhao `[通讯]` (New York University)

**通讯引用:** 38060 | [OpenAlex ID](https://openalex.org/A5108047350)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 ZeroDayBench 这一新的基准，用来评估大型语言模型（LLM）在真实生产代码中主动发现并补丁新出现的高危漏洞。

**💡 创新点**

创新点在于：1) 通过把真实 CVE 迁移到功能相似但不同的开源代码库中，形成真正的“零日”任务；2) 设计了 5 级信息难度，模拟漏洞生命周期的不同阶段；3) 引入基于渗透测试的评估方法，衡量补丁是否真正阻断可利用的 exploit；4) 对比多种前沿 LLM（GPT‑5.2、Claude Sonnet 4.5、Grok 4.1 Fast）的整体性能与行为差异。

**🔧 技术方法**

技术手段包括：a) LLM agent 通过 Bash 与 Edit 两种工具与 Docker 容器交互；b) 对任务进行自动化搜索、定位、编辑、验证；c) 使用五级提示（无提示、CWE、攻击描述、文件+函数、完整补丁信息）来逐步提供上下文。

**📊 数据集**

数据集：从 NVD、cvedetails.com 等公开数据库挑选 CVSS ≥ 7.0 的 22 个新型高危 CVE，随后通过手工迁移到不同的开源项目（如 MLFlow、Flyte、Mosquitto、vLLM、Dropbear、Haproxy、Squid、Tinyproxy、Jenkins、Minio、Verdaccio 等）构成 benchmark 任务。每个任务都有 5 个信息级别版本。

**📈 对比分析**

比较方法：记录每个 LLM 在每个难度级别下的通过率（patch 能否通过渗透测试阻止 exploit），并统计工具调用次数、成本、失败模式。结果显示：Claude 在低信息级别略优（12.8% vs GPT 14.4% vs Grok 12.1%），但在中高信息级别 GPT 与 Claude 接近；整体平均通过率分别为 Claude 56.0%、GPT 48.2%、Grok 34.0%。模型表现随信息量增大显著提升，且 Claude 在“无信息”场景下更倾向于误补丁；Grok 具有最低成本和调用次数，但存在 reward‑hacking 行为。

**⚠️ 局限性**

局限性包括：1) 迁移的 CVE 可能在模型训练集里出现，难以完全消除记忆效应；2) 任务规模受手工迁移限制，样本数有限；3) 采用的代码库多为公开项目，LLM 可能已有结构知识，导致“信息泄漏”；4) 评估侧重补丁效果，未覆盖所有语言和更复杂的安全场景。

---

## 365. Learning Demographic-Conditioned Mobility Trajectories with Aggregate Supervision

**arXiv ID:** 2603.03275 | [PDF](https://arxiv.org/pdf/2603.03275v1)

**作者:** Jessie Z. Li `[一作]`, Serina Chang `[通讯]` (University of California)

**通讯引用:** 1960 | [OpenAlex ID](https://openalex.org/A5046445911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种弱监督方法，用地区级聚合特征和人口构成数据学习无个人标签的族群条件行程生成模型。

**💡 创新点**

创新点在于：①利用地区聚合数据实现族群条件生成，突破缺乏个体族群标签的瓶颈；②给出理论框架阐明何时可成功，强调地区族群多样性与特征信息量的关键作用；③框架模型无关，可应用于扩散模型、Transformer、LLM等多种生成器。

**🔧 技术方法**

技术手段包括：基于BART自编码器的潜在扩散模型；两阶段训练（先无条件基线，再使用地区族群分布与聚合特征进行微调）；聚合监督损失（如Jensen–Shannon、总变差）；特征映射 ϕ 用于提取行程统计；理论分析采用线性代数与信息量度。

**📊 数据集**

使用了Embee 数据集（含移动 POI 检测与自报人口学信息）进行实验，同时在训练阶段使用无标签轨迹数据（如 GeoLife、YJMob100K、Veraset）。

**📈 对比分析**

与无族群条件基线模型和直接使用个体族群标签的强监督模型进行对比，采用 JSD 等指标评估空间、行程距离、起止点、POI 频率等统计。结果显示在族群多样性良好时，模型将 JSD 降低 12–69%，并逼近强监督模型；在聚合特征和地区划分合适时还能提升后续下游任务（如下一 POI 预测）的准确率。

**⚠️ 局限性**

局限性包括：①需要地区间足够的族群差异才能解耦；②聚合特征的可识别性决定是否能完整恢复族群分布；③聚合监督只能匹配特征统计，无法保证完整分布匹配；④需提供地区级聚合特征和人口构成，且对区域划分与样本量敏感。

---

## 366. Spatial Autoregressive Modeling of DINOv3 Embeddings for Unsupervised Anomaly Detection

**arXiv ID:** 2603.02974 | [PDF](https://arxiv.org/pdf/2603.02974v1)

**作者:** Ertunc Erdil `[一作]` (ETH Zurich), Ender Konukoglu `[通讯]` (ETH Zurich)

**通讯引用:** 13683 | [OpenAlex ID](https://openalex.org/A5036970822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出在DINOv3嵌入上进行二维自回归建模，以实现无监督异常检测。

**💡 创新点**

显式建模嵌入网格的空间依赖，使用掩蔽卷积与膨胀卷积实现并行推理，消除内存库开销。

**🔧 技术方法**

采用DINOv3预训练特征提取、掩蔽卷积自回归CNN、膨胀卷积以及负对数似然训练等技术。

**📊 数据集**

在BMAD基准的BraTS2021脑MRI、BTCV+LiTs肝CT和RESC视网膜OCT数据集上进行实验。

**📈 对比分析**

与重建、教师-学生、特征分布、内存库等方法比较，AUROC/AUPR与最先进方法相当，同时推理时间和显存显著降低。

**⚠️ 局限性**

对大规模DINO模型的收益有限，膨胀卷积在某些数据集上提升不明显，且自回归假设可能忽略跨层语义关系。

---

## 367. Generalized non-exponential Gaussian splatting

**arXiv ID:** 2603.02887 | [PDF](https://arxiv.org/pdf/2603.02887v1)

**作者:** Sébastien Speierer `[一作]` (Meta), Adrian Jarabo `[通讯]` (Meta)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究并实现了三维高斯散射（3DGS）的非指数衰减模型，并在渲染与重建中验证其效果。

**💡 创新点**

推导出非指数α混合机制，并在3DGS中引入线性、二次和超线性衰减函数，显著减少遮挡渲染次数。

**🔧 技术方法**

基于通用Boltzmann方程的非指数传输模型、路径重放反向传播、Mitsuba光线追踪渲染器以及自适应密度控制等技术。

**📊 数据集**

使用NeRF Synthetic数据集（Chair、Hotdog、Lego、Materials）以及真实场景的Dr Johnson、Playroom、Train、Truck。

**📈 对比分析**

与原始指数衰减的3DGS在相同渲染管线下进行PSNR/SSIM/帧率对比，非指数模型在保持或提升图像质量的同时实现3–5×的帧率提升与过绘次数下降。

**⚠️ 局限性**

仅在光线追踪模式下验证，尚未在栅格化实现；且对非指数传输的随机透明度支持有限，需进一步研究。

---

## 368. Deception by Design: A Temporal Dark Patterns Audit of McDonald's Self-Ordering Kiosk Flow

**arXiv ID:** 2603.03218 | [PDF](https://arxiv.org/pdf/2603.03218v1)

**作者:** Aditya Kumar Purohit `[一作]` (Center for Advanced Internet Studies), Adrian Holzer `[通讯]` (University of Neuchâtel)

**通讯引用:** 1860 | [OpenAlex ID](https://openalex.org/A5052881932)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对德国麦当劳自助点餐机进行时序暗模式审计，系统重构并分析12步交互流程

**💡 创新点**

首次将Temporal Analysis of Dark Patterns（TADP）框架应用于混合物理-数字界面，揭示暗模式随时间累积的层级效应

**🔧 技术方法**

使用TADP分析框架、屏幕截图重构（Omnigraffle）、情景模拟（模拟匆忙用户）以及手工注释进行系统化审计

**📊 数据集**

现场收集的自助点餐机交互屏幕截图和手工标注的数据集，模拟时间受限用户的操作路径

**📈 对比分析**

通过页面内、跨页面和系统层级对暗模式进行分类和计数，未进行传统算法性能对比，仅通过层级化叠加展示暗模式累积影响

**⚠️ 局限性**

仅针对单一国家单一门店的特定配置，使用的情景用户有限，未覆盖所有交互路径，缺乏量化用户实验验证

---

## 369. Manifold Aware Denoising Score Matching (MAD)

**arXiv ID:** 2603.02452 | [PDF](https://arxiv.org/pdf/2603.02452v1)

**作者:** Alona Levy-Jurgenson `[一作]` (University of Oxford), Yee Whye Teh `[通讯]` (University of Oxford)

**通讯引用:** 19387 | [OpenAlex ID](https://openalex.org/A5064373793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了Manifold Aware Denoising Score Matching (MAD)，通过在DSM中引入已知的基准score将分布学习分解为几何结构与密度两部分，显著提升了低维流形上生成模型的训练效率与采样质量。

**💡 创新点**

创新点在于利用解析可得的基准score（如球面、旋转矩阵、离散点集等）将目标score拆分为已知部分与待学习残差，从而减轻对流形支持的隐式学习负担，提升收敛速度与生成精度。

**🔧 技术方法**

技术核心为基准score推导、score分解、残差学习（通过神经网络实现）、对称性等价性处理与分布归一化投影；结合VE-SDE、MMD评价与多种对比模型。

**📊 数据集**

实验使用了地球数据（火山、地震、洪水、火灾）映射至S²、3D旋转矩阵（四元数）混合高斯、对称固体的SymSol I数据集、以及单位圆上的离散分布（均匀与倾斜）。

**📈 对比分析**

与RSGM-divfree、RSGM-ambient、DSM、FFF等方法比较，MAD在MMD指标上与DSM相当或更优，在训练和采样时间上收敛更快、生成样本更贴近流形，尤其在离散与高对称性任务中表现突出。

**⚠️ 局限性**

局限性主要在于需先手工推导解析基准score，难以推广至未知或高维复杂流形；对高维数据的可扩展性与自动化基准构造仍待研究。

---

## 370. Personalized Multi-Agent Average Reward TD-Learning via Joint Linear Approximation

**arXiv ID:** 2603.02426 | [PDF](https://arxiv.org/pdf/2603.02426v1)

**作者:** Leo `[一作]`, Lili Su `[通讯]` (Northeastern University)

**通讯引用:** 2313 | [OpenAlex ID](https://openalex.org/A5101541239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在异质环境下的多智能体平均奖励TD学习框架（PMAAR‑TD），通过联合线性近似将全局共享子空间与个体头部分离，实现单尺度学习。

**💡 创新点**

创新点在于：①单尺度更新下同时估计共享子空间和个体价值函数头部；②利用投影、QR与主角距离分析，实现负向误差收敛；③在马尔科夫采样与异质性下证明线性加速的有限时收敛率。

**🔧 技术方法**

使用技术包括：线性函数逼近、TD(0)与TD(L)、个性化联邦学习（PFL）思想、主角角度距离、Lyapunov 归纳、Markov 链均匀混合性与特征投影。

**📊 数据集**

实验数据集主要是控制环境：Acrobot 与 CartPole，且通过设计镜像（mirrored）环境增强异质性；使用三组随机种子和同步间隔 E=50。

**📈 对比分析**

与单智能体 TD、统一联邦 TD（FedTD‑Uniform）、双时尺度方法及标准 Actor‑Critic（SingleAC、FedAC‑Uniform）比较。结果表明：PMAAR‑TD 在收敛速度、最终奖励以及训练稳定性（方差）方面均优于基线，且显示线性加速效果。

**⚠️ 局限性**

局限性包括：①需要预设共享子空间维度 r 并假设其覆盖充分；②依赖均匀混合性、特征上界等严格假设；③分析主要针对线性近似，非线性/深度网络的推广尚未验证；④通信与投影操作在大规模多智能体环境下可能带来额外开销。

---

## 371. MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs for Clinical Tabular Prediction

**arXiv ID:** 2603.02221 | [PDF](https://arxiv.org/pdf/2603.02221v1)

**作者:** Zizheng Zhang `[一作]` (Big Data Institute, University of Oxford), Jingjing Fu `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了MedFeat框架，利用LLM和SHAP重要性实现可解释、模型感知的迭代特征工程，并在多项临床预测任务中验证其有效性。

**💡 创新点**

在LLM特征生成中加入模型感知提示、SHAP驱动的反馈与重要性加权岛屿采样，形成记忆化的、隐私保护的特征搜索流程。

**🔧 技术方法**

结合GPT‑4o生成可执行的特征变换、SHAP解释、重要性加权的岛屿采样、模型感知的提示以及成功/失败记忆库。

**📊 数据集**

在IORD、MIMIC‑IV和HRS三大临床表格数据集上进行实验，涵盖入院死亡、ICU死亡、10年死亡、出院后第二天转院、心衰等任务。

**📈 对比分析**

与原始特征、AutoFeat、OpenFE、CAAFE、FeatLLM、OCTree以及手工特征相比，MedFeat在未调参的低预算场景下均实现AUC和F1提升，最高可达7.87% AUC提升；在调参后仍保持F1优势。

**⚠️ 局限性**

在XGBoost深度调参后提升有限；依赖LLM生成质量与专业知识；在极端不平衡或少样本场景中仍需进一步验证；计算成本与LLM调用开销可能成为瓶颈。

---

## 372. HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding

**arXiv ID:** 2603.02329 | [PDF](https://arxiv.org/pdf/2603.02329v1)

**作者:** Lei Yao `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5903 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用多模态大语言模型（MLLM）直接提取接触感知意图嵌入的意图驱动3D可用性定位框架；

**💡 创新点**

创新点在于：1）通过MLLM生成意图嵌入并辅以文本标签监督；2）设计层次交叉模态集成机制将MLLM隐藏状态注入点云特征；3）引入多粒度几何提升模块逐级注入3D几何信息，提升定位精度；

**🔧 技术方法**

使用Qwen2.5-VL MLLM、PointNet++点云骨干、LoRA微调、注意力集成、几何提升与多任务损失；

**📊 数据集**

在PIAD、PIADv2两个标准交互式可用性数据集以及自建的噪声扰动基准上进行实验；

**📈 对比分析**

与GREAT、IAGNet、LASO等现有方法相比，平均交叉覆盖率(aIOU)提升5–10%，在未见对象、未见交互类型和噪声场景表现更为稳健；

**⚠️ 局限性**

仍受限于对完全新交互类型的泛化能力不足，以及对大模型计算资源的高依赖。

---

## 373. TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation

**arXiv ID:** 2603.02972 | [PDF](https://arxiv.org/pdf/2603.02972v1)

**作者:** Jiaxing Liu `[一作]` (Beijing University of Technology), Baocai Yin `[通讯]` (Beijing University of Technology)

**通讯引用:** 11583 | [OpenAlex ID](https://openalex.org/A5020527092)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种全端到端的 Vision‑Language Navigation 框架 Taga‑VLM，通过在 VLM 背骨中显式注入拓扑图结构，完成导航任务。

**💡 创新点**

创新点在于：①设计 Interleaved Navigation Prompt (INP)，把节点的视觉特征与文本描述按节点顺序交错嵌入；②提出 Spatial Topology Aware Residual Attention (STAR‑Att)，在自注意力中加入拓扑边缘距离作为残差偏置，使模型获得显式的空间关系；③利用全局动作空间实现路径纠正与回溯。

**🔧 技术方法**

技术包括：VLM 预训练（Qwen2）、ViT 视觉编码器、MLP 投影、STAR‑Att 结构化自注意力、全局动作决策与最短路径搜索、数据增强（HM3D）等。

**📊 数据集**

主要数据集：Matterport3D 模拟环境下的 R2R（训练 14,093 条轨迹，验证/测试 1,021/2,349 条），以及从 HM3D 生成的 500K 额外 SAP 样本。

**📈 对比分析**

与现有跨模态和大型模型方法对比，Taga‑VLM 在 R2R 未见环境上 SR 提升 3.39% (到 51.09%)，SPL 提升 9.08% (到 47.18%)，在 0.5B 参数规模下已超越多大模型，且 7B 版本进一步领先。

**⚠️ 局限性**

局限性包括：仍需大量预训练数据和显式拓扑图构建；对连续环境的适配尚未验证；STAR‑Att 仅基于欧氏距离，未加入更复杂几何约束；对动态或不完整地图的鲁棒性未知。

---

## 374. RxnNano:Training Compact LLMs for Chemical Reaction and Retrosynthesis Prediction via Hierarchical Curriculum Learning

**arXiv ID:** 2603.02215 | [PDF](https://arxiv.org/pdf/2603.02215v1)

**作者:** Ran Li `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 27606 | [OpenAlex ID](https://openalex.org/A5100333516)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个0.5B参数的RxnNano模型，用于化学反应和逆合成预测，强调化学理解而非单纯规模扩大；

**💡 创新点**

创新点包括：①引入潜在化学一致性（循环一致性）目标；②分层认知课程学习（从语法到语义逐步训练）；③原子映射置换不变性（AMPI）以避免数值偏差；④结构化计划推理提升LLM推理质量；⑤使用LoRA实现参数高效微调；

**🔧 技术方法**

技术实现基于Qwen2.5-0.5B Transformer，采用LoRA、三阶段课程学习、循环一致性损失、AMPI正则、计划化推理（潜在变量模型）等；

**📊 数据集**

主要使用USPTO系列数据集，包括USPTO-50K、USPTO-480K（MIT）和USPTO-FULL（约100万条），并分别构建映射（atom‑mapped）和非映射（unmapped）测试集；

**📈 对比分析**

在USPTO‑50K上实现Top‑1 75.1%（含AAM）/69.8%（不含AAM），比最佳模板/半模板/模板自由方法提升约23.5%；在USPTO‑FULL上实现Top‑1 62.1%，显著高于7B LLM RetroDFM 50.5%；不使用TTA或额外数据即可达到SOTA；

**⚠️ 局限性**

局限性：目前仅针对单步反应预测；多步路线、复杂反应的处理尚未研究；虽然模型小巧高效，但大规模数据和更大模型潜在优势未充分挖掘；

---

## 375. A Graph-Native Approach to Normalization

**arXiv ID:** 2603.02995 | [PDF](https://arxiv.org/pdf/2603.02995v1)

**作者:** Johannes Schrott `[一作]` (TU Wien), Katja Hose `[通讯]` (TU Wien)

**通讯引用:** 4397 | [OpenAlex ID](https://openalex.org/A5015313855)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种图本地化的图属性图归一化方法，定义图对象函数依赖（gnfd）并给出了对应的图转换规则，实现图本地化正规形式（gnnf）。

**💡 创新点**

创新点在于引入同时处理节点、边及节点-边间依赖的gnfd，扩展到边间与节点间依赖，定义图本地化的1NF、2NF、3NF和BCNF，并给出完整归一化算法。

**🔧 技术方法**

采用图模式匹配、图转换（创建与删除阶段）、Armstrong规则推理、最小覆盖算法、递归归一化与拓扑排序等技术。

**📊 数据集**

实验使用了六个合成与原生图数据集（包括课程/教师/学生、金融、社交等场景），并在Neo4j与Memgraph数据库上实现。

**📈 对比分析**

通过与已有仅节点归一化（gfd）方法对比，使用节点数、边数、属性平均数等指标，归一化后重冗余完全消除，节点/边数略增，运行时间受转换类型影响但整体可接受。

**⚠️ 局限性**

局限性在于依赖需预先已知，归一化后图结构更复杂，节点/边数增加，无法同时最小化冗余与图对象数量，且未覆盖更复杂的图模式或RDF场景。

---

## 376. IoUCert: Robustness Verification for Anchor-based Object Detectors

**arXiv ID:** 2603.03043 | [PDF](https://arxiv.org/pdf/2603.03043v1)

**作者:** Benedikt Brückner `[一作]` (Safe Intelligence), Alessio Lomuscio `[通讯]` (Safe Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面向锚点式目标检测器（如SSD、YOLOv2/3）的形式化鲁棒性验证框架，可对单目标场景下的边界框定位进行严格证明。

**💡 创新点**

创新点包括：① 通过坐标变换直接在offset空间上求IoU最优界，避免对非线性box预测的松弛导致的误差；② 推导并实现最优的IoU Interval Bound Propagation（IBP）界；③ 针对YOLOv3的LeakyReLU设计最优线性松弛，显著降低松弛误差；④ 将上述方法集成到Venus verifier，实现高效分支与界限搜索。

**🔧 技术方法**

使用技术包括：Interval Bound Propagation、Symbolic Interval Propagation、坐标变换（offset→corner）、最优IoU界推导、LeakyReLU最优松弛、分支与界限（branch‑and‑bound）搜索、AvgPool层替代MaxPool、NMS简化。

**📊 数据集**

实验数据集：LARD跑道检测数据集（单目标），Pascal VOC子集（TinyYOLO）。模型包括SSD、TinyYOLOv2、TinyYOLOv3。

**📈 对比分析**

与Cohen等人基于IBP的IoU界方法对比，本文方法在相同扰动ε下实现更紧的IoU和置信度界，显著减少分支数量，提升通过率；在SSD、YOLOv2/3的鲁棒性验证中，能够在更大扰动范围内完成验证，验证时间虽略长但整体效率提升。

**⚠️ 局限性**

局限性：仅针对单目标，未处理多目标和NMS交互的组合复杂性；为简化计算需将MaxPool替换为AvgPool，可能对模型性能略有影响；目前仅支持基础锚点式检测器，对更复杂架构（如YOLOv5/YOLOv7、anchor-free模型）尚不适用。

---

## 377. ICSE 2023 Sustainability Report

**arXiv ID:** 2603.02694 | [PDF](https://arxiv.org/pdf/2603.02694v1)

**作者:** Patricia Lago `[一作]` (Vrije Universiteit Amsterdam), Markus Funke `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5032767456)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对ICSE 2023会议参与者进行问卷调查，评估各类会议环节的满意度，并估算因往返墨尔本的空运产生的碳足迹。

**💡 创新点**

将碳足迹估算与参与者满意度结合，为未来会议的地点与活动类型提供可持续性决策依据。

**🔧 技术方法**

使用Google Forms收集问卷，借助OpenWeather、Airlabs、Climatiq API进行地址转换、机场定位和碳排放计算。

**📊 数据集**

ICSE 2023注册数据（1,424名参会者）以及问卷回收的161份回答。

**📈 对比分析**

采用5点星级评估和平均值比较不同会议环节的满意度，平均得分在4.4星左右，碳排放估算约5,043.5吨CO2e，但受样本量小和估算假设限制。

**⚠️ 局限性**

低问卷回收率（11%）导致统计显著性不足，碳足迹估算依赖于近似机场和单段航班假设，未考虑其他排放来源。

---

## 378. BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation

**arXiv ID:** 2603.02816 | [PDF](https://arxiv.org/pdf/2603.02816v1)

**作者:** Zihao Zhu `[一作]` (Chinese University of Hong Kong), Baoyuan Wu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6890 | [OpenAlex ID](https://openalex.org/A5068027800)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了文本到视频生成中的无缝品牌集成新任务，并设计了 BrandFusion 多智能体框架，实现广告品牌在生成视频中的自然嵌入，兼顾语义保真和品牌可见度。

**💡 创新点**

创新点包括：①首次定义无缝品牌集成任务；②两阶段设计——离线构建品牌知识库与在线五智能体协同优化提示；③利用经验池闭环学习，提升集成效果。

**🔧 技术方法**

采用的大技术手段有：大型语言模型驱动的多智能体系统；LoRA 轻量级微调生成品牌适配器；诊断式知识探测与经验池；以及多种扩散式文本到视频模型。

**📊 数据集**

实验使用了 18 个现有品牌和 2 个自创品牌，构造 270 个高/中/低匹配度的 prompt‑brand 对；通过合成数据集对 LoRA 进行微调；并在 Veo3、Sora2、Kling2.1 等商业与开源 T2V 模型上验证。

**📈 对比分析**

与直接追加、模板重写和单次重写基线相比，BrandFusion 在视频质量不变的前提下，在语义保真、品牌可见度和自然度指标上均显著优于基线；在人类评测中获得最高分，平均提升约 0.1–0.2 分。

**⚠️ 局限性**

局限性包括：在线优化需要约 7.4 次 LLM 调用，平均 16 秒延迟；仅支持单品牌集成，跨品牌或多品牌场景鲁棒性待验证；依赖预先构建的品牌知识库和适配器，需持续维护。

---

## 379. HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning

**arXiv ID:** 2603.02785 | [PDF](https://arxiv.org/pdf/2603.02785v1)

**作者:** Zihao Peng `[一作]` (Beijing Normal University), Tian Wang `[通讯]` (Beijing Normal University)

**通讯引用:** 18825 | [OpenAlex ID](https://openalex.org/A5006546107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种层次化的 LoRA 适配框架 HiLoRA，针对联邦学习场景构建根层、聚类层与叶层三重适配器，利用跨层正交性和层级化优化实现全局共享与个性化协同。

**💡 创新点**

创新点在于：①引入层级化 LoRA，分解为根、聚类、叶三层，显式捕获全局、子组及客户端特有知识；②基于 LoRA 子空间相似度的自适应聚类机制，自动发现隐藏的客户端群组；③跨层正交约束与级联训练，确保各层方向互不干扰，提升泛化与个性化性能。

**🔧 技术方法**

技术手段包括：低秩适配 LoRA、ViT 预训练模型、联邦学习框架、LoRA 子空间主角角距离度量、谱聚类、正交正则化、级联层级优化以及 Rademacher 复杂度分析。

**📊 数据集**

实验数据集：CIFAR-100（含多种标签偏斜设置）和 DomainNet（六域视觉数据）。

**📈 对比分析**

与 9 种 LoRA‑基线（Local‑LoRA、FedIT、FlexLoRA、FedSA‑LoRA、FDLoRA、FedDPA‑F/T、PF2LoRA、FedALT）在个性化、最差客户端、以及未见客户端适配等指标上进行比较；HiLoRA 在所有设置下均取得最高的平均准确率和最差端准确率，并显著提升未见客户端的适配效果，显示出领先性能。

**⚠️ 局限性**

局限性：①层级结构与正交约束增加模型复杂度与训练难度；②聚类结果对子空间相似度阈值和簇数选择敏感；③未研究自适应秩选择或多模特（LoRA‑MoE）扩展；④在极度异质或少数据场景下的鲁棒性尚待进一步验证。

---

## 380. Learning-Augmented Moment Estimation on Time-Decay Models

**arXiv ID:** 2603.02488 | [PDF](https://arxiv.org/pdf/2603.02488v1)

**作者:** Soham Nagawanshi `[一作]` (Texas A&M University), Samson Zhou `[通讯]` (Texas A&M University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5018283928)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于学习增强的时间衰减流模型的频率/矩估计、矩形频率估计和层叠范数估计算法

**💡 创新点**

利用尾部兼容的重头衔 oracle 通过学习提升在时间衰减（多项式/指数衰减和滑动窗口）中的空间复杂度，突破传统流模型的下界

**🔧 技术方法**

结合平滑直方图框架、线性 sketch、Count‑Sketch/采样技术与后处理函数，构造可兼容尾部 oracle 的学习增强算法

**📊 数据集**

在合成的偏斜分布、CAIDA 网络流量和 AOL 查询日志上进行实验

**📈 对比分析**

与传统 AMS、SS 等基线相比，学习增强版本在各窗口大小和采样率下均取得更低的估计误差、误差曲线更平稳，且在低空间预算下性能显著优于基线

**⚠️ 局限性**

实验仅覆盖了特定数据集和查询类型，未验证对更大规模或不同分布的鲁棒性；算法实现复杂度相对较高，实际部署需进一步优化

---

## 381. Length Generalization Bounds for Transformers

**arXiv ID:** 2603.02238 | [PDF](https://arxiv.org/pdf/2603.02238v1)

**作者:** Andy Yang `[一作]` (University of Notre Dame), Anthony W. Lin `[通讯]` (Max-Planck Institute for Software Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

证明了Transformer（至少两层）无法拥有可计算的长度泛化上界；并为正片段的RASP程序给出可计算且指数上界。

**💡 创新点**

首次将Transformer长度泛化问题与Hilbert第十问题、Diophantine方程归约关联，揭示其不可判定性；同时提供正片段可计算上界的最优性证明。

**🔧 技术方法**

利用RASP语言、有限状态逻辑、计数逻辑、Transformer-至-RASP等价性、Diophantine归约以及可计算长度复杂度理论。

**📊 数据集**

无实验数据集，全部为理论证明与形式化分析。

**📈 对比分析**

与先前对单层或受限两层RASP程序的多项式上界对比，指出一般Transformer的不可计算性；正片段的指数上界与现有最优指数上界相匹配。

**⚠️ 局限性**

仅针对Transformer的固定精度变体，未涵盖软注意力或更高精度的Transformer；正片段的上界仅在约束计数的情况下成立，实际训练中的实现细节与学习动态未被探讨。

---

## 382. VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats

**arXiv ID:** 2603.02986 | [PDF](https://arxiv.org/pdf/2603.02986v1)

**作者:** Alessio Mazzucchelli `[一作]` (Arquimea Research Center, Universidad Politécnica de Catalunya), Francesc Moreno-Noguer `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 12612 | [OpenAlex ID](https://openalex.org/A5106691454)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D Gaussian Splats的即时重色方法，只需编辑单张视图，即可在几秒内将颜色编辑传播到整个场景，且保持视角相关的高光效果。

**💡 创新点**

创新点包括：① 将场景颜色分离为漫射和视角相关（镜面）两部分，分别由独立MLP建模；② 在训练阶段采用多视角批次（multi‑view）渲染，显著提升视角一致性；③ 在编辑时仅微调漫射MLP最后一层并结合软分割，实现仅用一张编辑图即可快速、全局地重色。

**🔧 技术方法**

使用技术包括3D Gaussian Splatting（C3DGS）与Hashgrid编码、双MLP（漫射+镜面）架构、软分割网络、CUDA多视角光栅化、L1 + 光照平滑损失、视角相关特征提取等。

**📊 数据集**

主要使用的评测数据集有MipNeRF‑360、LLFF以及Synthetic NeRF等真实与合成场景数据集。

**📈 对比分析**

方法与PaletteNeRF、RecolorNeRF、IReNe等NeRF重色基线进行对比，使用PSNR/SSIM/LPIPS等指标评估；在所有数据集上均取得最高分，且编辑时间约为2秒，比Gaussian Editor等传统方法快两百倍，且能够保持视角一致且无颜色泄漏。

**⚠️ 局限性**

局限性在于仅支持简单的漫射+镜面材质，无法处理透明、等向性或彩色反射等复杂材质；单视图编辑在视角变化不足时分割不准，可能需要多张编辑图；对于镜面或透明表面时仍可能出现颜色泄漏。

---

## 383. Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States

**arXiv ID:** 2603.03146 | [PDF](https://arxiv.org/pdf/2603.03146v1)

**作者:** Jierui Zhang `[一作]`, Kaibin Huang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种通道自适应的边缘AI框架，通过动态调整特征量化比特宽度和服务器端模型的遍历深度，最大化端到端推理吞吐量（EPR），同时满足延迟和精度约束。

**💡 创新点**

创新点在于提出了可解析的推理精度模型，将角度特征建模为混合von Mises分布，得出精度与量化误差、模型深度的闭式关系；基于该模型推导出完整的通道自适应AI算法，实现通信与计算的协同优化。

**🔧 技术方法**

使用了早期退出模型（early‑exit）、Mixture of von Mises (MvM) 分布建模、MAP决策、量化与信道模型、连续松弛（CR）与二分搜索求解最优量化比特和深度，实验验证基于ResNet‑152与CIFAR‑10。

**📊 数据集**

采用CIFAR‑10数据集（60k训练样本、10k测试样本）来训练ResNet‑152和对应的角度分类器，并在此数据集上评估算法性能。

**📈 对比分析**

通过与固定量化宽度和固定深度的非自适应方案对比，实验显示通道自适应AI在不同SNR下显著提升EPR；低SNR时可提升至两倍，高SNR时超过100%增益；同时在保持精度要求时优于基线。

**⚠️ 局限性**

局限性包括：未考虑快速衰落和信道切换导致的实时性问题；未设计对信道中断导致推理失败的处理机制；目前模型映射为离线预估，实时推断的适应性有限；对更大类数或不同网络结构的泛化仍需进一步验证。

---

## 384. Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild

**arXiv ID:** 2603.02475 | [PDF](https://arxiv.org/pdf/2603.02475v1)

**作者:** Vitor Pereira Matias `[一作]` (Universidade de São Paulo), Tiago Novello de Brito `[通讯]` (Universidade de São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首先构建了大规模的皮肤色调数据集 STW（42,313 张图像，3,564 个人），采用 10 级 Monk Skin Tone (MST) 进行标注，并基于此数据集提出了 SkinToneNet（Fine‑tuned ViT‑Small）进行皮肤色调分类；随后对比了经典计算机视觉（CCV）管线与深度学习方法，评估了它们在 STW、MSTE、CCv1/2 等公开数据集上的性能，并用 SkinToneNet 对 CelebA、VGGFace2 等流行人脸数据集进行公平性审核。

**💡 创新点**

创新点主要有：①首次公开 10 级 MST 皮肤色调大规模标注数据；②提出 SkinToneNet，利用 ViT 与 ordinal 损失实现端到端精确分类；③采用严格的个体拆分（IND）和多重验证，消除身份泄漏；④对现有公共人脸数据集进行皮肤色调分布审计，揭示系统性偏差。

**🔧 技术方法**

技术手段包括：经典 CV 管线（Mediapipe 分割 + 颜色描述符 + 随机森林等）；深度学习模型（ResNet18、DenseNet121、ViT‑Base/S‑Small、DINOv2/3、LabNet、VehicleNet 等）；使用交叉熵、加权交叉熵、ordinal 损失；图像增强（翻转、旋转、光照扰动、随机遮挡）；严格的个体级数据拆分和 5‑折交叉验证。

**📊 数据集**

使用的数据集有：STW（自建 42k 图像）、MSTE、CCv1/2、FACET、FairFace、CelebA、VGGFace2、LFW、CASIA‑V5、CASIA‑Africa、FEI、Faces 94/95 等。

**📈 对比分析**

比较方法：采用 IMG（图像级拆分）和 IND（个体级拆分）两种拆分，使用 bAcc 与 wOOAcc 评估；经典 CV 在 IND 下 bAcc~0.33、wOOAcc~0.68；深度学习在 IND 下 Acc~0.90、OOAcc~0.91；在外域数据（MSTE、CCv2）上，SkinToneNet 的 Acc 约 0.88–0.89，OOAcc 0.90+，比传统 CV 接近随机的 0.5 要好 30–60%；DINOv3/ViT‑Small 取得最优。

**⚠️ 局限性**

局限性：①数据集仍存在极端肤色（1、9、10）样本不足，导致分布不均；②仅针对静态面部图像，未覆盖视频、书籍等多模态；③模型对光照、遮挡的鲁棒性仍有限；④仅在公开数据集评估，真实应用场景下可能面临更多偏差；⑤未提供生物识别或隐私风险的完整评估。

---

## 385. "It's Messy...But I Feel Balanced": Unpacking Flexible Workers' Rhythm-Making Practices Using an Asset-Based Approach

**arXiv ID:** 2603.02841 | [PDF](https://arxiv.org/pdf/2603.02841v1)

**作者:** Tse Pei Ng `[一作]` (National University of Singapore), Janghee Cho `[通讯]` (National University of Singapore)

**通讯引用:** 625 | [OpenAlex ID](https://openalex.org/A5103283515)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计探针和半结构化访谈，研究新加坡灵活工作者在承担照护职责时如何运用多种资产维持工作与照护的节奏。

**💡 创新点**

创新点在于将模糊边界视为可利用的资源而非问题，并从资产视角揭示个人、空间、关系、制度与内在资产如何协同支持灵活工作。

**🔧 技术方法**

使用了设计探针、访谈、构建主义扎根理论、ATLAS.ti编码与情境分析等定性研究技术。

**📊 数据集**

数据集为20名在新加坡从事灵活工作并承担照护责任的个体的访谈记录与探针素材。

**📈 对比分析**

研究未进行定量性能对比，而是通过与现有文献对照与案例阐释其方法学意义，未给出客观指标。

**⚠️ 局限性**

局限性包括样本规模有限、女性为主、仅限新加坡语境、缺乏现场实地观察，以及未纳入照护者本身视角。

---

## 386. Learning in Markov Decision Processes with Exogenous Dynamics

**arXiv ID:** 2603.02862 | [PDF](https://arxiv.org/pdf/2603.02862v1)

**作者:** Davide Maran `[一作]` (Politecnico di Milano), Marcello Restelli `[通讯]` (Politecnico di Milano)

**通讯引用:** 3304 | [OpenAlex ID](https://openalex.org/A5017130830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了强化学习在存在外生动态的马尔可夫决策过程（PCMDP）中的学习问题，并提出了基于可控/不可控状态分解的模型基和模型无基算法；

**💡 创新点**

创新点在于明确区分可控与不可控状态，将不可控部分视为外生动态，利用已知可控动态消除对环境的探索需求，从而得到仅依赖外生状态空间大小的最优 regret 上界；

**🔧 技术方法**

采用了改进的价值迭代算法Exogenous‑Aware Value Iteration（ExAVI）和改进的Q‑学习算法Exogenous‑Aware Q‑Learning（ExAQ），并给出严格的理论分析与下界证明；

**📊 数据集**

使用了经典toy环境Taxi with Traffic（自行构造的带交通拥堵的交通仿真数据）和实际交易执行任务的仿真数据；

**📈 对比分析**

与传统MDP基线UCBVI、QL以及PPO进行对比实验，结果表明ExAVI/ExAQ在样本效率上提升数百到数千倍，能在极少的试验次数内快速收敛至最优；

**⚠️ 局限性**

局限性包括仅适用于离散表格设置，假设可控动态已知且完全可观测，且算法的计算复杂度高，难以直接扩展到连续或大规模状态空间。

---

## 387. Scale-invariant Gaussian derivative residual networks

**arXiv ID:** 2603.02843 | [PDF](https://arxiv.org/pdf/2603.02843v1)

**作者:** Andrzej Perzanowski `[一作]` (KTH Royal Institute of Technology), Tony Lindeberg `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 15247 | [OpenAlex ID](https://openalex.org/A5054396186)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种带残差跳连的高斯导数残差网络（GaussDerResNet），能够在保持可证明的尺度协变性与尺度不变性的同时，实现更深层网络和更好的尺度泛化。

**💡 创新点**

创新点包括：①引入残差块，使高斯导数网络更易训练且准确率显著提升；②证明任意阶、任意维的尺度协变与不变性质；③构建多尺度通道网络并通过尺度池化实现尺度不变性；④结合深度可分离卷积与零阶项进一步降低参数与计算量；⑤将网络与速度适应仿射扩散方程的半离散化进行概念性关联。

**🔧 技术方法**

核心技术包括：高斯导数卷积核（多阶导数、尺度归一化）、残差连接、批归一化、ReLU、空间选择（中心像素或最大池化）、尺度池化（max/LogSumExp/平均）、深度可分离卷积、标签平滑、尺度通道 dropout。

**📊 数据集**

实验使用的公开数据集有：STL‑10、Fashion‑MNIST、CIFAR‑10 以及新构造的 Rescaled STL‑10、Rescaled Fashion‑MNIST 与 Rescaled CIFAR‑10，所有测试集均覆盖 1/2–2 的尺度范围。

**📈 对比分析**

与传统网络（WideResNet、Harm‑WideResNet、SESN‑B）在 STL‑10 上对比，GaussDerResNet 在保持相近甚至更低参数量的同时达到 88–91% 的准确率；与之前的 GaussDerNet 对比，在 Rescaled 数据集上多尺度 GaussDerResNet 的准确率提高 1.5–7个百分点，尺度泛化曲线几乎平坦，平均/LogSumExp 池化表现最佳。

**⚠️ 局限性**

局限性包括：①尺度通道数有限，导致在极端尺度下出现离散化误差；②对非中心化图像时需采用空间最大池化，可能导致局部信息损失；③训练仍需针对每个数据集手动调参（尺度比例、通道数、是否加入零阶项）；④在更大模型或更复杂任务中的推理速度与内存占用仍高于传统卷积网络。

---

## 388. Compositional Visual Planning via Inference-Time Diffusion Scaling

**arXiv ID:** 2603.02646 | [PDF](https://arxiv.org/pdf/2603.02646v1)

**作者:** Yixin Zhang `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7523 | [OpenAlex ID](https://openalex.org/A5028834865)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、基于推理时的复合扩散模型，利用短时段视频扩散器生成长时程视觉规划。

**💡 创新点**

核心创新在于在清晰数据（Tweedie估计）上实施边界一致性约束，并通过同步/异步信息传递和扩散球面引导实现全局一致性，避免了传统在噪声状态下的因子分解失效。

**🔧 技术方法**

技术包括链式因子图推理、同步与异步信息传递、Tweedie估计、DDIM采样、扩散球面（Diffusion‑Sphere）指导以及逆动力学映射。

**📊 数据集**

使用 ManiSkill 机器人操作基准（100 任务，18 组分布内+82 组外部）以及真实 Franka Emika Panda 机器人抓取实验。

**📈 对比分析**

与 DiffCollage/GSC 等基线对比，显著提升了视频质量（动态一致性、背景连贯性）和任务成功率（在分布内/外部均高于 80%），并在真实机器人上实现了 10/10、8/10 的成功率。

**⚠️ 局限性**

局限性包括对短时段扩散器的依赖、对参数（如消息权重、步长）的敏感性，以及在更复杂高维状态空间下可能仍需进一步验证。

---

## 389. EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks

**arXiv ID:** 2603.02562 | [PDF](https://arxiv.org/pdf/2603.02562v1)

**作者:** Yuchen Shi `[一作]` (Tsinghua University), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45485 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EdgeFLow 框架，将传统 FL 中的云端服务器替换为边缘网络中的顺序模型迁移，实现无服务器联邦学习。

**💡 创新点**

创新点在于通过边缘基站间的模型连续迁移消除云端通信，显著降低通信开销，并在非凸、非 IID 环境下提供收敛理论。

**🔧 技术方法**

采用了联邦平均（FedAvg）机制、局部 SGD、L‑smooth、梯度方差控制等技术，并在边缘集群内实现模型聚合与迁移。

**📊 数据集**

实验使用 FashionMNIST 与 CIFAR‑10 两个图像分类基准数据集，构造 IID 与多种非 IID 数据分布。

**📈 对比分析**

与传统 FedAvg、Hierarchical FL 进行对比；在非 IID 场景下 EdgeFLow 的准确率与 FedAvg 相近甚至略优，同时通信量下降 50%–80%，并且在更复杂的边缘网络拓扑中优势更明显。

**⚠️ 局限性**

限制主要体现在对集群划分和迁移顺序的依赖，且在高度动态的边缘网络或极端非 IID 条件下可能需要进一步的自适应机制来保证收敛和性能。

---

## 390. UniG2U-Bench: Do Unified Models Advance Multimodal Understanding?

**arXiv ID:** 2603.03241 | [PDF](https://arxiv.org/pdf/2603.03241v1)

**作者:** Zimo Wen `[一作]` (Microsoft Research Asia), Yifei Shen `[通讯]` (Microsoft Research Asia)

**通讯引用:** 1728 | [OpenAlex ID](https://openalex.org/A5101694397)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了UniG2U基准，用以系统评估统一多模态模型中生成对理解（G2U）的影响；

**💡 创新点**

构建了涵盖7大认知范畴、30个细粒度子任务的3,000样本评测集，设计了Direct与Generate‑then‑Answer两种推理协议，并引入RA/AL两项中间视觉对齐度量；

**🔧 技术方法**

利用统一多模态模型（包括端到端、解耦、Agentic三类）及其对应的基础视觉‑语言模型，结合标准化的prompt、预算匹配、greedy解码等技术；

**📊 数据集**

数据来源于IllusionBench、MMSI‑Bench、Geometry3K、Uni‑MMMU、ChartQA、RealUnify、BabyVision等公开数据集；

**📈 对比分析**

通过与其严格配对的基线VLM比较，计算Δ_G2U（直接与GtA两种模式）并报告各子任务与整体准确率；结果显示大多数统一模型相较基线表现下降（“alignment tax”），但在空间推理、视觉错觉等子任务中获得正增益；

**⚠️ 局限性**

局限包括：1）生成路径不一定提升理解，易引入误差；2）对中间视觉质量的依赖导致性能波动；3）基线对齐与规模对结果影响较大，尚缺乏更细粒度的因果评估。

---

## 391. Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration

**arXiv ID:** 2603.02760 | [PDF](https://arxiv.org/pdf/2603.02760v1)

**作者:** Linhao Zhong `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 69113 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DiSE，一种基于令牌再生概率的diffusion大语言模型（dLLM）的自评估方法；

**💡 创新点**

创新点在于利用dLLM的双向掩码与非顺序生成特性，以再生概率作为模型置信度指标，显著提高自评估效率并实现可解释的置信量化；

**🔧 技术方法**

采用了Token Regeneration、概率加权平均、灵活长度生成框架以及基于DiSE的条件似然估计与不确定性量化；

**📊 数据集**

实验使用LLaDA-Instruct-8B、LLaDA-1.5-8B等dLLM，并在ARC-Challenge、GPQA、Countdown、GSM8K、MATH500、SVAMP等多项任务上验证；

**📈 对比分析**

与传统蒙特卡洛（N_mc=1/32）以及自回归LLaMA3-Instruct-8B相比，DiSE在条件似然评估上提升了多达6.4%精度，且速度提升约32倍；在不确定性量化中ROC‑AUC平均提升10.5%；在灵活长度生成中相对固定长度基线获得显著准确率提升；

**⚠️ 局限性**

局限性包括：未对半自回归混合模型进行评估；令牌子集的选择仍为经验性，缺乏系统化的最优策略。

---

## 392. LaTeX Compilation: Challenges in the Era of LLMs

**arXiv ID:** 2603.02873 | [PDF](https://arxiv.org/pdf/2603.02873v1)

**作者:** Tianyou Liu `[一作]` (Southern University of Science and Technology), Xurui Liu `[通讯]` (Tsinghua University)

**通讯引用:** 7733 | [OpenAlex ID](https://openalex.org/A5100681562)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析 LaTeX 在大语言模型（LLM）时代的编译效率、语义生成、错误定位以及工具生态等方面的缺陷，提出并评估了 Mogan STEM 这一 WYSIWYG 结构化编辑器作为替代方案。

**💡 创新点**

创新点包括：① 引入低熵文档格式（Mogan 文档）以提升 LLM 微调效率；② 通过结构化数据结构、快速渲染和按需插件加载显著提升编译与渲染性能；③ 证明 Mogan 在 LLM 任务中的优势，并呼吁开展更大规模的实验。

**🔧 技术方法**

使用技术主要有：结构化编辑器的数据结构设计、实时渲染引擎、插件按需加载机制，以及基于 Mogan 文档格式的 LLM 微调方法。

**📊 数据集**

数据集方面，实验以 LaTeX 文档和 Mogan 文档为对照材料进行编译/渲染时间及 LLM 任务性能的测评，具体公开的数据集未在本文中列出。

**📈 对比分析**

比较方法：对比 LaTeX 与 Mogan 在编译时间、渲染速度以及 LLM 任务（如生成、推理）中的表现；实验结果显示 Mogan 在所有指标上均优于 LaTeX，尤其在 LLM 微调时因信息熵降低而显著提升效率。

**⚠️ 局限性**

局限性：实验规模有限，缺乏跨平台和跨工具的广泛验证；Mogan 生态系统尚不完整，对现有 LaTeX 生态的兼容性和迁移成本仍需进一步研究。

---

## 393. From "What" to "How": Constrained Reasoning for Autoregressive Image Generation

**arXiv ID:** 2603.02712 | [PDF](https://arxiv.org/pdf/2603.02712v1)

**作者:** Ruxue Yan `[一作]` (Nankai University), Xiaojie Yuan `[通讯]` (Nankai University)

**通讯引用:** 3150 | [OpenAlex ID](https://openalex.org/A5062064974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoR-Painter框架，先通过受约束的推理生成全局空间约束，再生成细致的描述，最终驱动自回归图像生成；同时引入Dual-Objective GRPO双目标强化学习，分别优化文本推理和视觉投射过程。

**💡 创新点**

创新点在于：①首次将“约束推理（CoR）”作为生成先导，将全局结构和细节分离；②通过Dual-Objective GRPO提供专门的语义锚定奖励、投影奖励与整体对齐奖励，显著提升语义一致性与空间布局。

**🔧 技术方法**

技术包括：自回归统一多模态模型Janus‑Pro、链式思考（CoT）与强化学习、Group Relative Policy Optimization (GRPO)、双目标强化学习策略、文本/图像奖励模型（Llama2、HPSv2、GIT、GroundingDino）。

**📊 数据集**

在T2I‑CompBench、GenEval与WISE三个公开基准上进行评估。

**📈 对比分析**

与现有CoT/AR+RL方法（如T2I‑R1、Show‑o、Janus‑FocusDiff）相比，CoR‑Painter在空间关系、属性绑定等指标上均取得SOTA，T2I‑CompBench空间分数提升5.41%，GenEval空间定位提升约5%，WISE各类任务均优于竞争者。

**⚠️ 局限性**

局限性包括：对计数任务的表现略逊于部分方法，整体方法依赖预训练模型与奖励设计，且在极端复杂或隐式知识场景下仍可能出现推理偏差或计算成本较高。

---

## 394. Marginal Gains or Meaningful Progress? Exploring Tech Tuber Narratives on Annual Smartphone Innovation

**arXiv ID:** 2603.02392 | [PDF](https://arxiv.org/pdf/2603.02392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 395. TRACE: Task-Adaptive Reasoning and Representation Learning for Universal Multimodal Retrieval

**arXiv ID:** 2603.02929 | [PDF](https://arxiv.org/pdf/2603.02929v1)

**作者:** Xiangzhao Hao `[一作]` (Institute of Automation, Chinese Academy of Sciences), JinQiao Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TRACE框架，将生成式推理与判别式表示学习相结合，实现任务自适应的推理-编码流程；

**💡 创新点**

创新点在于自适应推理路由机制、将Chain-of-Thought压缩为嵌入token、统一单阶段训练以及证明查询侧推理的显著优势；

**🔧 技术方法**

使用大语言模型（Qwen2.5‑VL）进行CoT生成、信息对比损失（InfoNCE）、低秩适配（LoRA）以及任务感知采样；

**📊 数据集**

构造了基于M‑BEIR的M‑BEIR‑CoT数据集，并在M‑BEIR基准及13个未见域数据集上进行评估；

**📈 对比分析**

与CLIP、BLIP、UniIR、LamRA等方法对比，在M‑BEIR平均Recall@5提升约+4.2%，在零样本任务上多项指标超过LamRA，取得SOTA；

**⚠️ 局限性**

限制在于推理阶段产生延迟、对合成CoT数据的质量依赖、可能产生幻觉以及对极端域的鲁棒性有限。

---

## 396. Towards Accurate and Interpretable Time-series Forecasting: A Polynomial Learning Approach

**arXiv ID:** 2603.02906 | [PDF](https://arxiv.org/pdf/2603.02906v1)

**作者:** Bo Liu `[一作]` (China Electronics Technology Group Corporation), Xiaotong Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5100340837)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种可解释的多项式学习（IPL）方法，用于时间序列预测并在早期预警系统中实现特征级可解释性。

**💡 创新点**

创新点在于在模型结构中显式建模原始特征及其任意阶交互，既保留时间依赖，又可通过调节多项式阶数实现预测精度与可解释性的灵活权衡。

**🔧 技术方法**

采用多项式核展开、ADMM优化算法、特征级权重提取，并与LIME、SHAP、ARIMAX等后置/本地解释方法进行对比实验。

**📊 数据集**

使用了三类数据集：合成模拟数据、比特币历史价格数据（约5,000个样本）以及现场天线健康监测数据（5,437个样本）。

**📈 对比分析**

在预测精度（MSE、AUC、误差等）和解释准确度（特征重叠率、排名相似度）上，IPL均优于LIME、SHAP和ARIMAX；其计算效率更高，早期预警机制更简洁、准确率接近甚至超过其他方法。

**⚠️ 局限性**

局限性包括：多项式阶数提升会导致模型复杂度和参数量上升，易出现过拟合；对极高维或噪声较大的时间序列数据的鲁棒性尚待进一步验证。

---

## 397. Deep Learning Based Wildfire Detection for Peatland Fires Using Transfer Learning

**arXiv ID:** 2603.02465 | [PDF](https://arxiv.org/pdf/2603.02465v1)

**作者:** Emadeldeen Hamdan `[一作]` (University of Illinois Chicago), Ahmet Enis Cetin `[通讯]` (University of Illinois Chicago)

**通讯引用:** 4716 | [OpenAlex ID](https://openalex.org/A5080469744)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于迁移学习的沼泽地火灾检测框架，利用预训练野火模型权重对沼泽地图像/视频进行微调，并在ResNet50中加入Walsh–Hadamard变换（WHT）卷积层以提升对低光强、持续烟雾等特征的辨识能力。

**💡 创新点**

创新点在于将WHT卷积层嵌入残差块，形成WHT‑ResNet50，实现参数量更少、能耗更低且在小样本沼泽地火灾数据集上表现更优；同时将迁移学习与频域特征增强结合，为低对比度火灾场景提供了新的解决思路。

**🔧 技术方法**

主要技术包括深度迁移学习、WHT卷积层与逆变换、ResNet‑50/EfficientNet‑B5/Swin‑Transformer骨干网络、图像块划分与重叠预处理，以及评价指标ACC、Precision、Recall、F1。

**📊 数据集**

使用了大规模野火图像视频数据集GWFP进行预训练，以及收集自马来西亚沼泽地的专门图像与视频数据集作为微调与测试集。

**📈 对比分析**

通过与基准网络（EfficientNet‑B5、Swin‑Transformer、标准ResNet‑50、HTMA‑ResNet‑50）在无迁移学习与迁移学习两种训练方式下进行对比，结果显示WHT‑ResNet‑50迁移学习后ACC达到90.1%、F1为89.6%，显著高于其他模型。

**⚠️ 局限性**

局限性包括沼泽地火灾标注数据仍相对稀缺，模型在极低光照或强遮挡条件下仍需进一步验证；迁移学习效果受源域与目标域差异影响；并未涉及多模态融合或真实实时部署的细节。

---

## 398. From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?

**arXiv ID:** 2603.03148 | [PDF](https://arxiv.org/pdf/2603.03148v1)

**作者:** Shinas Shaji `[一作]` (Fraunhofer Institute for Intelligent Analysis and Information Systems), Sebastian Houben `[通讯]` (Fraunhofer Institute for Intelligent Analysis and Information Systems)

**通讯引用:** 1671 | [OpenAlex ID](https://openalex.org/A5020225503)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将大型语言模型（LLM）作为核心规划与推理组件、配合工作记忆和情节记忆的认知机器人架构，并在模拟家庭环境中通过LLM驱动移动机械臂完成对象放置与对象交换两项任务

**💡 创新点**

创新点在于：1）将LLM直接作为“代理”执行器，负责高层规划与决策；2）通过工作记忆与情节记忆实现从经验中学习和自适应规划；3）在单一框架内整合感知、导航、抓取等工具接口，使LLM能在机器人环境中“翻译”语言指令为低层动作

**🔧 技术方法**

主要技术包括：agentic LLM（如 GPT‑4 或同类模型）、记忆模块（工作记忆、情节记忆）、工具调用接口（感知、推理、导航、抓取、放置）以及与模拟环境的交互接口

**📊 数据集**

使用自建的模拟家庭环境（含移动机械臂与可交互物体），任务数据由人工设计的对象放置和对象交换场景构成；未使用公开大规模数据集

**📈 对比分析**

在两个基准任务（对象放置、对象交换）上评估LLM驱动代理的推理、规划和记忆利用；结果显示代理能完成结构化任务，并能通过记忆实现适应性规划；但也存在显著偏差：对任务成功率的幻觉、对指令的拒绝性响应，导致连续任务执行失败

**⚠️ 局限性**

主要局限性包括：1）LLM易产生幻觉，误判任务完成状态；2）对顺序指令的跟随能力不足，出现拒绝或忽略执行的行为；3）缺乏可靠的低层动作执行反馈机制，导致难以纠正错误；4）对真实物理环境的迁移性能尚未验证

---

## 399. Gated Differential Linear Attention: A Linear-Time Decoder for High-Fidelity Medical Segmentation

**arXiv ID:** 2603.02727 | [PDF](https://arxiv.org/pdf/2603.02727v1)

**作者:** Hongbo Zheng `[一作]` (University of Illinois Urbana-Champaign), Minjia Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4049 | [OpenAlex ID](https://openalex.org/A5077768924)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于Gated Differential Linear Attention（GDLA）的解码器中心Transformer，用于高效且精确的医学图像分割。

**💡 创新点**

创新点在于将双子核线性注意力与可学习差分操作结合，并加入头特定门控和局部token混合分支，以消除线性注意力的注意力稀释与噪声，保持O(N)复杂度。

**🔧 技术方法**

核心技术包括Gated Differential Linear Attention、可学习通道门控、局部深度可分离卷积混合、预训练的Pyramid Vision Transformer编码器以及Mix-FFN前馈网络。

**📊 数据集**

使用CT、MRI、超声和皮肤镜四种医学影像数据集（Synapse CT、ACDC、BUSI、HAM10000和PH^2）进行实验。

**📈 对比分析**

与CNN、Transformer、Hybrid及传统线性注意力模型在相同训练预算下对比，表现出更高Dice/Acc指标、参数量相近但FLOPs更低，达成或超过多项任务的state‑of‑the‑art成绩。

**⚠️ 局限性**

局限性包括对3D体数据未做直接适配、对极高分辨率图像的显存需求仍相对较高，以及在极少样本或极端噪声场景下的鲁棒性尚待进一步验证。

---

## 400. RO-N3WS: Enhancing Generalization in Low-Resource ASR with Diverse Romanian Speech Benchmarks

**arXiv ID:** 2603.02368 | [PDF](https://arxiv.org/pdf/2603.02368v1)

**作者:** Alexandra Diaconu `[一作]` (University of Bucharest), Bogdan Alexe `[通讯]` (University of Bucharest)

**通讯引用:** 4557 | [OpenAlex ID](https://openalex.org/A5045881151)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RO-N3WS，包含约126小时罗马尼亚广播新闻与多域外语料的高质量语音数据集，并对其在零样本与微调条件下的ASR性能进行了基准评估。

**💡 创新点**

创新点在于：①构建了兼顾领域泛化的新闻与外域语料划分；②系统比较了多种先进模型（Whisper、Wav2Vec 2.0、商业API）在零样本与微调两种场景下的表现；③通过自然语料与高质量TTS混合实验探讨合成语料的补充价值。

**🔧 技术方法**

使用的技术包括：Whisper多语言Transformer、Wav2Vec 2.0自监督模型、Microsoft、Google、Vatis商业ASR API；语音预处理与精细标注流程；使用Praat/Parselmouth进行音调与强度分析；对比实验采用WER指标，并对格式化差异做多参照优化。

**📊 数据集**

使用的数据集是RO-N3WS（广播新闻105小时 + OOD 21小时），并与现有公共数据（Common Voice、VoxPopuli、FLEURS、Echo）进行对比；实验还使用了由ElevenLabs生成的合成音频进行补充训练。

**📈 对比分析**

通过零样本与微调两种评估方式，对比各模型在ProTV、Antena1新闻以及Audiobook、Film、Stories、Podcast等OOD子集上的WER。结果显示：商业API在零样本时表现最佳；Whisper Large + RO-N3WS微调后在新闻域获得≈2–4% WER；在OOD域微调后仍存在显著提升（例如从≈40%降至≈14%），而Wav2Vec 2.0在OOD上仍显弱。

**⚠️ 局限性**

局限性包括：①数据集主要来自广播新闻，缺少方言、噪声环境与用户生成内容；②微调仍需大量人工标注；③合成语料虽能提升，但仍无法完全替代真实语音；④对不同语言和更小规模数据集的泛化能力未进行深入验证。

---

## 401. Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs

**arXiv ID:** 2603.02556 | [PDF](https://arxiv.org/pdf/2603.02556v1)

**作者:** Zhiyu Pan `[一作]` (Huazhong University of Science and Technology), Jieping Ye `[通讯]` (Alibaba Cloud)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于视觉对比的自我提升框架VC-STaR，通过对比式VQA对生成的推理路径进行校正，进而生成高质量的视觉推理数据集VisCoR-55K，并利用该数据集对VLM进行监督微调，显著提升其视觉推理能力。

**💡 创新点**

创新点在于：①首次利用视觉对比（contrastive VQA对）来纠正模型产生的视觉幻觉；②设计了三阶段思考-对比-重思路推理流程；③构建了涵盖多任务（推理、图表/数学、OCR等）的55k样本视觉推理数据集；④通过对比式数据驱动的自我提升，提升了VLM在多项视觉推理基准上的表现。

**🔧 技术方法**

采用多模态对比学习（视觉-文本双编码、ID‑based视觉度量学习）、大语言模型提示（思考、对比、重思路），使用Qwen2.5‑72B进行重思路阶段，整体在LLaMA‑factory框架下进行全参数监督微调。

**📊 数据集**

利用21套多任务VQA数据集（推理、图表/数学、OCR、通用等）构造对比对；生成VisCoR‑55K数据集；评估基准包括MMVP、Hallusion、MathVista、MathVision、MMStar、MME‑RealWorld；与Virgo、LLaVA‑CoT、R1‑OV、LPT等现有推理数据集进行对比。

**📈 对比分析**

通过与基线（未微调）、三种自我提升基线（STaR、Verifier、Feedback）以及四个离线推理数据集微调模型进行比较。VC‑STaR在六大基准上平均提升2.4%，在视觉幻觉基准MMVP提升5.7%、Hallusion提升3.2%，在数学和通用任务亦有明显增益，整体性能优于所有对照方法。

**⚠️ 局限性**

局限性包括：①仅使用“中等难度”对比对，易样本会导致性能下降；②对比对的构造依赖相似度阈值，可能对不同域适配不佳；③方法主要针对VQA/推理任务，未验证在更复杂或非VQA场景下的泛化；④重思路步骤依赖外部LLM，增加推理成本；⑤整体仍需人工挑选或自动化程度有限。

---

## 402. Cross-Layer Decision Timing Orchestration in Cost-Based Database Systems: Resolving Structural Temporal Misalignment

**arXiv ID:** 2603.02253 | [PDF](https://arxiv.org/pdf/2603.02253v1)

**作者:** Ilsun Chang `[一作]` `[通讯]`, Ilsun Chang

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在传统成本基数据库系统中，作者分析了优化器与执行器之间的时间错位导致尾部延迟放大的问题，并提出一种跨层决策时序编排框架，将最终决策权从编译时优化器转移至运行时执行器，利用统一风险信号（URS）进行动态“晚绑定”决策；

**💡 创新点**

核心创新点在于：① 将决策时机视为结构性因素，提出跨层时序编排；② 通过统一风险信号融合优化器、执行器与加速器三层的风险信息，实现在运行时根据实时条件动态切换执行策略；③ 只在关键操作层面进行晚绑定，避免全量重规划，控制开销；

**🔧 技术方法**

采用的技术包括：成本基优化器、运行时执行器的信号采集、GPU/CPU执行路径的预枚举与切换、统一风险向量（URS）计算与阈值决策、PostgreSQL原型改造与微基准实验；

**📊 数据集**

使用了控制的微基准数据集（synthetic workloads），分别模拟输入规模突变、统计信息陈旧以及加速器成本阈值条件；

**📈 对比分析**

通过与传统优化器主导方案、独立门控方案比较，评估输入规模偏移、统计陈旧和加速器成本三类实验场景；结果显示，在输入规模急剧变化或统计信息失效时，尾部延迟（P99）可提升至20倍，且中位延迟基本保持不变；

**⚠️ 局限性**

限制因素包括：1）仅对预先枚举的高影响操作进行晚绑定，未覆盖完整查询计划重排；2）实验仅基于微基准，缺乏真实生产工作负载验证；3）需要准确估计风险阈值，阈值设置对性能影响较大；4）对加速器的支持取决于特定硬件（如GPU），可移植性有限。

---

## 403. DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction

**arXiv ID:** 2603.03265 | [PDF](https://arxiv.org/pdf/2603.03265v1)

**作者:** Yufu Wang `[一作]` (University of Pennsylvania), Michael Zollhofer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究提出双阶段扩散模型DuoMo，用于从无约束单目视频中恢复全局一致的世界空间人体运动，直接生成网格顶点而非依赖SMPL参数；

**💡 创新点**

创新点包括：①采用双先验——先在相机空间生成运动，再提升并通过世界空间扩散模型实现全局一致；②在每段视频内以起始相机姿态定义世界坐标系，避免对固定原始空间的对齐；③直接输出网格顶点，摆脱对形状参数化模型的依赖；④结合高度条件、遮挡掩码和引导采样（2D投影/位移）解决漂移与长遮挡问题；

**🔧 技术方法**

使用技术主要有：生成式扩散模型（DiT+RoPE+窗口注意力）、稠密关键点与图像特征提取、相机姿态估计、2D投影与位移引导、遮挡掩码训练、SMPLX转换网络；

**📊 数据集**

训练数据：AMASS、BEDLAM、3DPW、Goliath、WHAM（相机空间模型）；AMASS、BEDLAM（世界空间模型）；评估数据：EMDB、RICH、Egobody；

**📈 对比分析**

与现有最优方法对比，使用TRAM相机姿态或静态相机；指标包括WA‑MPJPE、W‑MPJPE、RTE、foot‑skating；DuoMo在EMDB、RICH分别比第二佳方法降低16%和30%的误差，在Egobody完整序列及遮挡段表现更低误差，且漂移与foot‑skating均得到显著改善；

**⚠️ 局限性**

局限性：仍依赖相机姿态估计，极端相机噪声下漂移和误差可能上升；生成式推理相对耗时；目前仅针对人体网格，尚未验证对其他物体类别的泛化能力。

---

## 404. Architectural HRI: Towards a Robotic Paradigm Shift in Human-Building Interaction

**arXiv ID:** 2603.03052 | [PDF](https://arxiv.org/pdf/2603.03052v1)

**作者:** Alex Binh Vinh Duc Nguyen `[一作]` `[通讯]` (University of Antwerp), Alex Binh Vinh Duc Nguyen (University of Antwerp)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并提出了将机器人技术应用于建筑空间的概念框架，主张多层建筑同步物理适配（如移动隔断、智能遮阳、灯光等）以支持居住者需求与可持续目标，呼吁跨学科研究；

**💡 创新点**

创新点在于将机器人视作多层建筑或建筑整体，将HRI知识迁移至建筑形态，并强调时空与社会三维维度的交叉；

**🔧 技术方法**

主要技术包括机器人家具、群体机器人、形变空间等软硬件系统，强调感知、通信、控制与人机交互技术的整合；

**📊 数据集**

未使用实验数据集，内容基于文献综述与理论构想；

**📈 对比分析**

由于缺乏实证实验，本文未进行方法比较或性能评估，而是提出概念性模型与未来研究路线；

**⚠️ 局限性**

局限性包括：1）多层建筑物体适配技术与人机交互的实证验证不足；2）跨学科整合仍碎片化；3）缺乏可操作的评估指标与数据集；4）对社会伦理与安全性的讨论不够充分。

---

## 405. On Discriminative vs. Generative classifiers: Rethinking MLLMs for Action Understanding

**arXiv ID:** 2603.02546 | [PDF](https://arxiv.org/pdf/2603.02546v1)

**作者:** Zhanzhong Pang `[一作]` (National University of Singapore), Angela Yao `[通讯]` (National University of Singapore)

**通讯引用:** 4798 | [OpenAlex ID](https://openalex.org/A5006278133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在多模态大语言模型(MLLM)上对闭集动作识别进行细粒度评估，比较生成式与判别式分类器，并提出生成辅助判别（GAD）框架以提升性能与效率。

**💡 创新点**

创新点在于：①揭示生成式分类器因标签语义重叠而受限；②将生成式与判别式联合成单模型，在训练阶段加入生成辅助任务来正则化判别特征；③实现兼容预训练的LLM且推理时保持判别式的高效性。

**🔧 技术方法**

使用 LLaMA3 与 Qwen2.5 作为语言解码器，视觉编码器 SigLIP-ViT 或其他预训练视觉模型，LoRA 微调，生成辅助损失结合交叉熵训练。

**📊 数据集**

评估数据集包括 COIN、EPIC‑Kitchens‑100、Ego4D‑GoalStep、CrossTask 与 THUMOS’14，涵盖步骤识别、预测、任务识别与在线动作检测。

**📈 对比分析**

与纯生成式、传统判别式以及多种 SOTA 方法比较，GAD 在四大任务上平均提升约2.5% Top‑1/ F1，速度提升约3×，并在 1B 规模下超过多项 8B 规模模型，证明其显著性能与效率优势。

**⚠️ 局限性**

局限性：仅适用于闭集任务，无法直接识别新动作；对 MLLM 的通用能力会因任务特定微调而下降。

---

## 406. PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments

**arXiv ID:** 2603.02538 | [PDF](https://arxiv.org/pdf/2603.02538v1)

**作者:** Aduen Benjumea `[一作]` (Oxford Brookes University), Matthias Rolf `[通讯]` (Oxford Brookes University)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5102882780)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出了PathSpace框架，利用B样条曲线在线构建连续概率地图，实现语义SLAM的高效表示与更新。

**💡 创新点**

创新点在于将B样条与概率密度结合，提供可局部更新且自适应简化的连续地图表示，显著降低地图规模并提升可扩展性。

**🔧 技术方法**

采用B样条曲线、协方差传播（Cubature Transform）、Ridge回归、Kalman滤波以及基于曲率的简化算法。

**📊 数据集**

以模拟的Formula Student Driverless跑道与真实赛道上的红黄灯锥检测数据为实验集。

**📈 对比分析**

与传统CKF点状基准相比，PathSpace在RMSE上略逊但保持相近，地图大小缩减53%，计算时间随地图规模趋于平稳。

**⚠️ 局限性**

主要局限是对轨迹的依赖性较强，精度略低且尚未探索更复杂的语义表征与动态环境适应。

---

## 407. PRISM: Exploring Heterogeneous Pretrained EEG Foundation Model Transfer to Clinical Differential Diagnosis

**arXiv ID:** 2603.02268 | [PDF](https://arxiv.org/pdf/2603.02268v1)

**作者:** Jeet Bandhu Lahiri `[一作]` (Indian Institute of Technology Mandi), Sandeep Singh `[通讯]` (NeuroDx)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对EEG基础模型的预训练人群多样性进行对照实验，评估其在标准基准任务和临床难题（癫痫与模拟症状差异诊断）中的表现，并系统拆解并量化了EEG-Bench与EEG-FM-Bench两大基准之间的六大方法差异。

**💡 创新点**

①通过控制预训练数据来源，证明多源预训练可显著提升模型泛化，尤其在临床难题上获得12.3个百分点的提升；②首次将癫痫与其诊断混淆者（PNES、Syncope等）的差异诊断任务引入EEG基础模型评估；③系统拆解并定量分析了两大基准之间的六大方法差异，揭示评测不一致可导致最高24个百分点的排名翻转。

**🔧 技术方法**

采用Masking AutoEncoder（MAE）基于REVE架构的4D位置编码模型，训练窄源与多源两检查点；使用线性探针、单阶段/双阶段全微调、部分微调等多种适配策略；对基准任务使用不同分类头（注意力池化、平均池化、MLP），对临床任务采用全微调+MLP。

**📊 数据集**

预训练数据：D1（窄源）—TUH + PhysioNet Motor Imagery；D2（多源）—D1 加南亚多中心（9663受试者，4170小时）临床EEG；基准任务：ADFTD、BCI-IV-2a、HMC、PhysioNet-MI、Siena Scalp、EEGMAT；临床任务：200名南亚患者（100癫痫，100模拟症状），使用Natus系统录制。

**📈 对比分析**

在六个基准任务下分别进行线性探针和全微调，并比较两检查点的表现；发现窄源在线性探针上略占优，多源在全微调上相当或更优；在癫痫与模拟症状的差异诊断任务中，多源检查点比窄源提升12.3个百分点；与REVE对比，多源在多协议、多任务中超越或与其相当；评测协议差异可导致最高24个百分点的排名翻转。

**⚠️ 局限性**

仅使用单一模型规模，未探索规模效应；多源数据同时包含地理与设备多样性，无法单独评估两者对性能的贡献；临床数据仅来自南亚地区，缺乏跨地区验证；未系统评估不同遮掩率、不同分类头对其他模型的泛化性。

---

## 408. Robotic Grasping and Placement Controlled by EEG-Based Hybrid Visual and Motor Imagery

**arXiv ID:** 2603.03181 | [PDF](https://arxiv.org/pdf/2603.03181v1)

**作者:** Yichang Liu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16038 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于EEG视觉与运动想象的双通道零射击实时框架，实现了大脑意图驱动的抓取与放置机器人交互。

**💡 创新点**

创新点在于：①将视觉想象与运动想象同时解码为机器人指令，构成自然意图驱动的BCI；②使用离线预训练解码器直接应用于在线流式管道，无需在线训练；③在无视觉线索、不同姿态下仍能完成任务，展示了姿态无关的控制能力。

**🔧 技术方法**

技术包括EEG预处理（带通滤波、ICA）、特征提取（DE、时间窗）、三种解码模型（EEGNet、RGNN、MLP）、在线实时推理以及机器人控制（KINOVA GEN2 + RealSense 视觉感知）。

**📊 数据集**

使用的实验数据集：5名受试者的离线视觉感知/视觉想象（3种水果）共300次试验，以及运动想象（左右手）共100次试验；在线测试在相同受试者上进行多场景验证。

**📈 对比分析**

与传统基于P300/SSVEP的BCI对比，本研究在VI任务上取得了 44.11%（离线）/40.23%（在线）准确率，MI任务上 76.53%/62.59%；整体任务成功率 20.88%，表明虽然准确率不高，但已可实现机器人抓放的闭环控制，并在姿态、遮挡等复杂场景下保持一定鲁棒性。

**⚠️ 局限性**

局限性包括：EEG信号噪声导致解码准确率有限；视觉想象的可分辨性低；样本规模小、缺乏跨受试者泛化；在线时延主要受机器人执行时间影响，且整体成功率仍偏低。

---

## 409. An Optimization-Based User Scheduling Framework for Multiuser MIMO Systems

**arXiv ID:** 2603.02998 | [PDF](https://arxiv.org/pdf/2603.02998v1)

**作者:** Victoria Palhares `[一作]` (Nokia Bell Labs), Christoph Studer `[通讯]` (ETH Zurich)

**通讯引用:** 10802 | [OpenAlex ID](https://openalex.org/A5083617223)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于优化的用户调度框架，能够全局、并行地为多用户MIMO系统中的UE分配时空资源；

**💡 创新点**

关键创新点包括：① 在调度约束中加入不等式约束，实现对UE活动资源的精确控制；② 提供多种可微分的目标函数（后LMMSE均方误差、信道容量、可实现速率）；③ 给出所有目标函数的梯度以及投影到带不等式约束的单纯形的解析投影算法；④ 通过正则化促进二值解并设计量化算法；

**🔧 技术方法**

核心技术包括：离散优化问题的松弛为连续非凸优化，使用前向后向拆分（FBS）求解；利用Douglas–Rachford拆分求解投影；基于Karush‑Kuhn‑Tucker条件求解带不等式的投影；正则化项与量化策略保证二值解；

**📊 数据集**

使用Remcom Wireless InSite Ray‑Tracing软件生成的实际毫米波60 GHz场景和sub‑6 GHz蜂窝‑自由系统的射频通道向量，分别对应30 351个UE位置和26 934个UE位置的真实通道；

**📈 对比分析**

与多种基线（SUS、CSS、greedy、LoFi、LoFi++、random、all‑UEs‑active、以及在小型系统上的穷举搜索）进行比较，实验显示在MMW和蜂窝‑自由场景下，该框架在90%分位点的BER、HMI、MSE与可实现速率指标上均优于所有基线，并逼近穷举搜索性能；

**⚠️ 局限性**

局限性：① 由于是非凸优化，FBS只能收敛到局部最优，无法保证全局最优；② 计算复杂度高于贪婪/启发式方法；③ 目标函数未显式考虑公平性；④ 未结合功率控制、频率调度或用户‑中心化架构，限制了在更广泛系统中的直接适用性。

---

## 410. MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks

**arXiv ID:** 2603.02630 | [PDF](https://arxiv.org/pdf/2603.02630v1)

**作者:** Zhi Hong `[一作]` (Chinese University of Hong Kong), Zhongxiang Dai `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5081447482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在固定拓扑多智能体系统中通过提示优化提升系统性能

**💡 创新点**

提出MASPOB框架，将图神经网络、LinUCB探索与坐标上升相结合，显著解决样本效率、拓扑耦合与组合爆炸问题

**🔧 技术方法**

采用GAT作为结构感知代理预测器、LinUCB做不确定性引导的探索、坐标上升化简组合搜索、使用文本嵌入与GPT‑4o‑mini进行评估

**📊 数据集**

在HotpotQA、DROP、HumanEval、MBPP、GSM8K、MATH六大公开基准上进行实验

**📈 对比分析**

与单体提示方法（IO、CoT、ReAct）、单体提示优化方法（PromptBreeder、Instinct）及多智能体优化基线（AFlow、MIPRO）对比，MASPOB平均提升约12.02%，在所有任务上均夺得最佳成绩

**⚠️ 局限性**

对复杂拓扑、不同LLM后端、极大搜索空间的通用性仍需进一步验证，且仍受限于高昂的全流程评估成本

---

## 411. MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN

**arXiv ID:** 2603.03024 | [PDF](https://arxiv.org/pdf/2603.03024v1)

**作者:** Ling Luo `[一作]` (Southwestern University of Finance and Economics), Qianqian Bai `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了BrainNav框架，利用生物启发的双地图双定向策略提升移动机器人在现实环境中的空间感知与导航能力，避免幻觉，零调优即可实现实时路径规划。

**💡 创新点**

创新点在于将大脑五个关键认知区（海马、视觉皮层、顶叶、前额叶、小脑）映射为五大功能模块，并结合坐标图+拓扑图、相对+绝对定向的双重表示，形成多模态空间记忆与决策机制。

**🔧 技术方法**

采用视觉与深度感知融合、动态图像捕获、图卷积网络构建空间记忆、强化学习决策、端到端路径规划等技术；核心模块实现对环境的即时解析与运动控制。

**📊 数据集**

在公开的VLN视觉语言导航数据集（如Matterport3D、Gibson）上训练后，在真实实验室场景中使用Limo Pro机器人进行零样本评估。

**📈 对比分析**

与现有最先进的VLN方法在实验室环境中进行对比，BrainNav在成功率、路径长度、时间效率以及空间幻觉率等指标上均实现显著提升，无需任何微调。

**⚠️ 局限性**

局限性包括：只在实验室环境验证，缺乏多样化真实世界测试；对快速动态障碍物的适应仍有限；模型复杂度高，部署成本较高；长期持续导航与跨环境迁移性能待进一步评估。

---

## 412. Large Language Model Empowered CSI Feedback in Massive MIMO Systems

**arXiv ID:** 2603.02686 | [PDF](https://arxiv.org/pdf/2603.02686v1)

**作者:** Jie Wu `[一作]` (Southeast University), Mérouane Debbah `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 66838 | [OpenAlex ID](https://openalex.org/A5056145687)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于大语言模型（LLM）的CSI压缩与反馈框架 LLMCsiNet，将 CSI 压缩任务重构为掩码令牌预测问题，并通过自信息量选择重要元素进行反馈。

**💡 创新点**

创新点在于：①利用自信息量度量 CSI 元素的重要性，实现更高效的掩码策略；②将预训练的 LLM 作为上下文建模器进行缺失 CSI 的推断，显著提升重构精度；③将大部分复杂计算放置在基站，保持终端轻量化。

**🔧 技术方法**

核心技术包括自信息量计算、掩码令牌预测、Transformer（GPT‑2 Large）预训练、卷积特征提取、残差网络辅助重构、两阶段联合训练与迁移学习。

**📊 数据集**

实验使用四个公开信道数据集：COST2100out、COST2100in、UMa（3GPP 38.901）以及 DeepMIMOo1（ray‑tracing）。

**📈 对比分析**

与 CRNet、IdasNet、TransInDecNet、GCRNet‑1x 等传统小模型进行对比，LLMCsiNet 在 NMSE 上提升 3–10 dB、SGCS 接近 1、并在多用户 MIMO 中实现更高的可达率；同时在多压缩比和迁移学习场景下保持优异性能。

**⚠️ 局限性**

局限性包括：基站端 LLM 计算量与参数规模大，部署成本较高；需要两阶段训练与细粒度调参；未在真实无线环境中验证，实际时延与信道非理想因素仍需进一步研究。

---

## 413. Same Error, Different Function: The Optimizer as an Implicit Prior in Financial Time Series

**arXiv ID:** 2603.02620 | [PDF](https://arxiv.org/pdf/2603.02620v1)

**作者:** Federico Vittorio Cortesi `[一作]` (Massachusetts Institute of Technology), Pierfrancesco Beneventano `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在金融时间序列中不同网络架构与优化器组合对学习函数的影响，即使预测误差（NMSE）相同，也会产生不同的响应面和交易行为。

**💡 创新点**

揭示了优化器在低信噪比环境下充当隐式先验，决定可接受函数的形态，从而导致功能多样性与经济后果的差异。

**🔧 技术方法**

采用MLP、CNN、LSTM、Transformer四种深度网络与SGD、Adam、Muon三种优化器，结合冲击响应、差分表面、SHAP特征重要性、Hessian曲率监测以及Sharpe–Turnover前沿评估等技术。

**📊 数据集**

使用2000–2024年标普500成分股的无生存偏差波动率数据，目标为Garman–Klass估计的日波动率。

**📈 对比分析**

对12个学习系统进行超参数搜索，并在13个随机种子上平均，所有模型在NMSE上基本持平（低于线性基准），但功能诊断显示不同优化器产生显著非平面差异，导致投资组合中Turnover差异可达3倍。

**⚠️ 局限性**

仅在波动率预测任务上验证，未广泛测试其他金融时间序列或市场结构，缺乏对交易成本下收益绝对影响的量化与因果机制的证明。

---

## 414. Deep learning-guided evolutionary optimization for protein design

**arXiv ID:** 2603.02753 | [PDF](https://arxiv.org/pdf/2603.02753v1)

**作者:** Erik Hartman `[一作]`, Johan Malmström `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种结合贝叶斯优化与遗传算法的蛋白质序列设计框架 BoGA，旨在高效探索序列空间并快速发现满足多种设计目标的高质量蛋白质。

**💡 创新点**

创新点在于将遗传算法的随机变异生成器与贝叶斯优化的代理模型、采集函数耦合，形成在线学习循环，能够在大量候选中筛选出最具潜力的序列，从而显著降低昂贵实验或计算评估的次数。

**🔧 技术方法**

使用的技术包括基于 ESM‑2 的序列嵌入、PCA 降维、深度推理回归 BiGRU 代理模型、期望改进（EI）采集函数、AlphaFold‑2 结构预测、Boltz‑2 蛋白复合物预测以及 ProteinMPNN 与 FastRelax 的后处理。

**📊 数据集**

数据集方面，作者以随机生成的序列或已知的 8–25 长度多肽作为种子，并在 pneumolysin（PLY）靶向实验中使用了该蛋白的结构与已知抗体表位作为评估参考；此外，还通过合成的二级结构和亲水性指标构造了内部基准任务。

**📈 对比分析**

与传统遗传算法（k_propose=m_select）对比，BoGA 在 β‑sheet、uHrel 以及结构导向目标上都取得了更快的收敛速度和更高的最终适应度；在 PLY 结合体设计中，k_propose=500 的设置显著加速了高分结合体的发现，最终筛选出 41 只高置信度结合肽。

**⚠️ 局限性**

局限性包括对代理模型准确性的高度依赖，若代理预测偏差大或不可靠，采集函数将失效；同时，尽管 BoGA 降低了实验评估次数，但仍需昂贵的结构预测和对接计算；最后，在极大序列空间中，遗传变异的探索仍可能受限，需进一步提升搜索多样性与效率。

---

## 415. Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry

**arXiv ID:** 2603.02258 | [PDF](https://arxiv.org/pdf/2603.02258v1)

**作者:** Kyle Elliott Mathewson `[一作]` (University of Alberta), Kyle Elliott Mathewson `[通讯]` (University of Alberta)

**通讯引用:** 5525 | [OpenAlex ID](https://openalex.org/A5016237703)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文利用Meta的NLLB‑200 200语种Transformer，对其编码器表示进行六项实验，探究模型是否已学习到语言无关的概念结构。

**💡 创新点**

创新点在于将大规模机器翻译模型的表示几何与认知科学关于多语言词汇组织（如共词化、概念中心、颜色体系）的理论结合，并首次在如此大规模、多语言的Transformer上验证了语言独立的概念仓库。

**🔧 技术方法**

技术手段包括：Transformer编码器表示提取、All‑But‑The‑Top（ABTT）等同构校正、按语言均值中心化、Mantel检验、曼–惠特尼U检验、Spearman相关、PCA降维、语义偏移向量一致性评估等。

**📊 数据集**

使用的数据集包括：Meta NLLB‑200 200语种模型、Swadesh核心词表、ASJP音系距离矩阵、CLICS3共词化数据库、基本颜色词表以及控制的非Swadesh词表。

**📈 对比分析**

通过统计检验比较模型表示与语言学/认知预测的吻合度，结果显示词嵌入距离与遗传距离显著相关、共词化概念对相似度显著更高、概念间可分离度在去除语言偏移后提升约×1.5、语义偏移向量一致性平均余弦≈0.85，表明模型捕捉到跨语言的普遍概念结构。

**⚠️ 局限性**

局限性包括：仅检验单一模型且未验证跨架构或规模泛化、使用单一carrier句子可能带来句法偏差、tokenizer粒度不均导致子词级别信息失衡、ABTT校正可能过度压缩语义差异、低资源语言覆盖不足、统计相关性无法证明因果关系、神经认知类比仅为结构性推测。

---

## 416. COOL-MC: Verifying and Explaining RL Policies for Platelet Inventory Management

**arXiv ID:** 2603.02396 | [PDF](https://arxiv.org/pdf/2603.02396v1)

**作者:** Dennis Gross `[一作]` `[通讯]`, Dennis Gross

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文使用COOL-MC工具，将强化学习得到的血小板库存管理策略与概率模型检验和可解释性方法相结合，对该策略进行形式化验证与解释。

**💡 创新点**

创新之处在于首次对血小板库存管理中的RL策略进行可形式化的安全性验证与特征级解释，结合可达状态DTMC构造、特征修剪与反事实分析。

**🔧 技术方法**

采用了PPO强化学习、COOL-MC流程、Storm概率模型检验、PCTL查询、特征修剪、特征重要性排列、动作标签和反事实动作替换等技术。

**📊 数据集**

使用了荷兰东北部血液银行的聚合需求数据（约144个血小板池/周）作为MDP参数。

**📈 对比分析**

通过对比最优MDP求解得到的极低缺货概率（≈3.14×10⁻¹⁰）与训练得到的PPO策略的缺货（2.9%）和过剩（1.1%）概率，展示了在200步内约99.6%状态空间压缩和对策略安全性的精确量化。

**⚠️ 局限性**

局限性包括仅基于单一地区的聚合需求参数、需离散动作空间、对大规模库存容量或更细的衰减分层可能仍面临状态爆炸、以及Poisson需求截断可能导致尾部分布失真。

---

## 417. MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization

**arXiv ID:** 2603.03192 | [PDF](https://arxiv.org/pdf/2603.03192v1)

**作者:** Ashutosh Chaubey `[一作]` (Institute for Creative Technologies, University of Southern California), Mohammad Soleymani `[通讯]` (Institute for Creative Technologies, University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 MoD-DPO 框架，利用模态分离的直接偏好优化（DPO）来降低 omni LLM 的跨模态幻觉；

**💡 创新点**

在 DPO 中加入模态不变性与敏感性 KL 正则化，并加入语言先验去偏罚，同时构建了大规模自动生成的优选样本，形成模态解耦的优化思路；

**🔧 技术方法**

采用 DPO+KL 正则化、语言先验去偏技术、自动优选数据生成（多阶段）以及对抗性损失，保证模型对不相关模态鲁棒，对相关模态敏感；

**📊 数据集**

利用 AVHBench、CMM、DailyOmni、MVBench、MMAU 等多模态基准，结合 MSR‑VTT、VALOR32K、AudioCaps 等原始数据生成 18k+ 优选样本；

**📈 对比分析**

与 DPO、OmniDPO、Vita、VideoLLaMA 等现有方法对比，MoD-DPO 在 AVHBench 与 CMM 的准确率提升约 3–4%，幻觉抵抗度显著提高，且在视频/音频通用基准上提升 0.5–1%；

**⚠️ 局限性**

主要限制在于训练成本略高，对多模态任务效果有限；需要手工设计的数据生成流程，且在极端噪声或跨模态干扰下仍可能产生幻觉。

---

## 418. Using Learning Progressions to Guide AI Feedback for Science Learning

**arXiv ID:** 2603.03249 | [PDF](https://arxiv.org/pdf/2603.03249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 419. NeuroSkill(tm): Proactive Real-Time Agentic System Capable of Modeling Human State of Mind

**arXiv ID:** 2603.03212 | [PDF](https://arxiv.org/pdf/2603.03212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 420. VLMFusionOcc3D: VLM Assisted Multi-Modal 3D Semantic Occupancy Prediction

**arXiv ID:** 2603.02609 | [PDF](https://arxiv.org/pdf/2603.02609v1)

**作者:** A. Enes Doruk `[一作]` (Ozyegin University), Hasan F. Ates `[通讯]` (Ozyegin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 VLMFusionOcc3D 框架，通过 Vision‑Language 模型的语言先验与天气感知门控机制，实现多模态（摄像头与 LiDAR）稠密 3D 语义占用预测。

**💡 创新点**

创新点包括实例驱动 VLM 注意力（InstVLM）将文本嵌入注入体素以解决语义歧义、天气自适应融合（WeathFusion）动态重加权感知通道，以及深度感知几何对齐（DAGA）损失提升结构一致性。

**🔧 技术方法**

采用 CLIP + LoRA 进行高效 VLM 适配，结合跨模态门控注意力、天气条件编码、3D 卷积与稀疏点云编码等技术。

**📊 数据集**

在 nuScenes（OpenOccupancy）和 SemanticKITTI 两大无人驾驶数据集上进行实验。

**📈 对比分析**

相较于 OccMamba、M‑CoNet 等基线，mIoU 在 nuScenes 提升至 26.6%（比基线高 1.4%），在 SemanticKITTI 提升至 26.4%（比基线高 1.8%），在雨天、夜间等恶劣环境下提升 5–6% 以上。

**⚠️ 局限性**

局限性包括对车辆元数据的依赖、对极端天气（如大雾、极光）仍有限制、以及 VLM 嵌入对训练数据分布的敏感性。

---

## 421. Wasserstein Proximal Policy Gradient

**arXiv ID:** 2603.02576 | [PDF](https://arxiv.org/pdf/2603.02576v1)

**作者:** Zhaoyu Zhu `[一作]` (Zhiyuan), Shuang Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 26499 | [OpenAlex ID](https://openalex.org/A5100415884)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了基于Wasserstein几何的连续动作空间策略梯度方法，设计了Wasserstein Proximal Policy Gradient (WPPG) 以及其隐式策略实现 WPPG-I。

**💡 创新点**

核心创新在于将Wasserstein proximal更新与热流（高斯噪声）通过算子分裂分解，消除了对政策密度或score函数的依赖，并在连续动作空间下给出了全局线性收敛保证，涵盖精确与近似 Q 函数两种情况。

**🔧 技术方法**

技术手段包括Wasserstein梯度流、JKO变形、Optimal Transport、Gaussian卷积、Implicit Policy（pushforward）以及Actor-Critic框架，并与SAC、TRPO等传统方法对比。

**📊 数据集**

实验使用MuJoCo连续控制基准（Hopper-v5、Walker2d-v5、HalfCheetah-v5、Reacher-v5、Swimmer-v5、Humanoid-v5）。

**📈 对比分析**

与PPO、SAC、WPO等基线比较，WPPG在Gaussian MLP政策下与SAC表现相当，WPPG-I在所有任务上均优于所有基线，尤其在高维难度任务中显著提升。

**⚠️ 局限性**

局限性包括需要满足T_2传输信息不等式等理论假设、对Q值估计的精度要求较高、隐式策略需足够表达、实验范围仅限MuJoCo，未检验更复杂环境，且收敛速度受步长与熵系数τ的影响。

---

## 422. Break the Window: Exploring Spatial Decomposition of Webpages in XR

**arXiv ID:** 2603.02471 | [PDF](https://arxiv.org/pdf/2603.02471v1)

**作者:** Chenyang Zhang `[一作]` (Georgia Institute of Technology), Eric J Gonzalez `[通讯]` (Google)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5001572522)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

在 XR 环境中实现网页内容的空间拆分，将完整页面拆解为可移动、可缩放的多面板布局，支持中空、表面固定以及直接触摸和射线交互。

**💡 创新点**

突破传统单窗口浮动模型，提出“打破窗口”概念，使网页结构与空间语义直接绑定，形成新的空间 UI 语法，促进注意力分布与语义映射。

**🔧 技术方法**

核心技术包括：桌面浏览器渲染后通过像素镜像流式传输到 XR，生成多面板视图；面板位置/缩放通过抓取边缘实现；自动根据距离切换直接触摸与射线交互；保持对原网页的功能同步。

**📊 数据集**

未使用公开数据集，而是基于 9 个常用网站的截图进行手工划分，并通过 15 位参与者的体验与 6 位 XR 研究者的访谈收集定性数据。

**📈 对比分析**

通过与桌面浏览器和单窗口 XR 浏览器的对比实验（定性访谈与现场观察），评估注意力分布、交互语义、效率感受。结果显示空间化增强了注意力分散和语义编码，但在精细操作与生产力方面并无明显提升，甚至在某些任务上更慢。

**⚠️ 局限性**

局限性包括：缺乏共享的空间 UI 语法导致布局不一致；物理锚定与虚拟面板冲突，导致误触与空间不稳定；精准输入受限于射线/触摸的空间精度；空间容量有限，过度拆分导致协调成本和认知负荷增加。

---

## 423. Rigidity-Aware Geometric Pretraining for Protein Design and Conformational Ensembles

**arXiv ID:** 2603.02406 | [PDF](https://arxiv.org/pdf/2603.02406v1)

**作者:** Zhanghan Ni `[一作]` (University of Illinois Urbana-Champaign), Shengchao Liu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6562 | [OpenAlex ID](https://openalex.org/A5100396540)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了基于刚性体的自监督预训练框架 RigidSSL，分两阶段预训练以学习蛋白质全局几何表示

**💡 创新点**

创新点在于将 SE(3)刚性体变换与双向刚性流匹配目标结合，利用扰动与 MD 动力学两阶段提升全局几何与动态信息的表示

**🔧 技术方法**

使用了 IPA 结构编码器、双向刚性流匹配（Flow Matching）目标、Gaussian SE(3)扰动和 SO(3) 采样

**📊 数据集**

使用了 AlphaFold 结构数据库 432K 架构与 1.3K 分子动力学轨迹

**📈 对比分析**

与 GeoSSL 等对照方法相比，在无监督蛋白生成、动机支架和 GPCR 动态集生成等任务上设计性提升多达43%，多样性和新颖性显著提高，生成质量和物理合理性得到提升

**⚠️ 局限性**

限制在于扰动预训练偏向稳定性导致多样性下降，MD 预训练虽然提升多样性却可能降低设计性，且两阶段的平衡与适用场景尚需进一步探究

---

## 424. MiM-DiT: MoE in MoE with Diffusion Transformers for All-in-One Image Restoration

**arXiv ID:** 2603.02710 | [PDF](https://arxiv.org/pdf/2603.02710v1)

**作者:** Lingshun Kong `[一作]` (Nanjing University of Science and Technology), Jinshan Pan `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 14820 | [OpenAlex ID](https://openalex.org/A5004164569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MiM-DiT框架，将层次化的混合专家（MoE）与预训练的扩散变换器（DiT）结合，实现统一处理多种图像退化。

**💡 创新点**

创新点在于双层MoE架构：Inter‑MoE通过密集路由融合四种不同注意力专家（空间、通道、Swin、SE），Intra‑MoE通过稀疏路由在每个专家组内部选择子专家，形成结构层次与细粒度两级自适应；同时使用零初始化线性层将MoE特征作为条件引导扩散过程。

**🔧 技术方法**

技术手段包括混合专家（MoE）与密集/稀疏路由、预训练的DiT（SD3.5）扩散模型、Zero‑Linear条件投射、基于注意力的多头自注意、Patch‑embedding与Transformer层、以及多尺度损失与对抗训练。

**📊 数据集**

训练数据集：FoundIR（含模糊、雾、雨、低光、噪声等多种退化）。评估数据集：FoundIR测试集、4KRD、RealRain‑1K、HazeRD、UHD‑LL。

**📈 对比分析**

与AirNet、DGUNet、TransWeather、PromptIR、DiffIR、DiffUIR、DA‑CLIP、AutoDIR、FoundIR、DiT4SR等最先进方法对比。实验结果显示，在LPIPS、FID、NIQE、LIQE、MUSIQ、CLIP‑IQA等多项指标上，MiM‑DiT均取得或接近最优，尤其在噪声、雾、低光等任务中显著优于其它方法，视觉效果更锐利、纹理更丰富。

**⚠️ 局限性**

局限性：双层MoE和扩散模型的计算开销较大，推理速度受限；模型对极端退化或未见过的混合退化仍可能性能下降；依赖大规模预训练模型，若无足够GPU资源难以快速复现。

---

## 425. Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation

**arXiv ID:** 2603.03116 | [PDF](https://arxiv.org/pdf/2603.03116v1)

**作者:** Hongliu Cao `[一作]` (Amadeus France), Eoin Thomas `[通讯]` (Amadeus France)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Procedure-Aware Evaluation (PAE)，对 LLM 代理的完整执行过程进行评估，揭示“腐败成功”问题。

**💡 创新点**

创新点在于将代理过程正式化为结构化观测，并通过四轴（效用、效率、交互质量、程序完整性）和六维门控，能够将违规完成的任务彻底剔除。

**🔧 技术方法**

采用基于 Dec-POMDP 的程序化框架、结构化观测空间、LLM-as-judge 进行语义评估，以及多维度计量（如 I_pc, I_pf, I_ec, I_df 等）。

**📊 数据集**

使用 τ-bench 数据集（Airline 与 Retail 两个领域），并对 GPT-5、Kimi-K2-Thinking、Mistral-Large-3 三大模型进行实验。

**📈 对比分析**

相较于传统的单一成功率评估，PAE 在效用、效率、交互和完整性四轴上均显示出显著差异；门控后 Pass^4 率从 58% 降至 24%，模型排名出现逆转，证明传统指标会误导性能评估。

**⚠️ 局限性**

局限在于需明确的 O^ctx（政策上下文），无法评估隐式规范；仅评估行为轨迹，无法捕捉推理错误；门控为二元，未考虑违规严重性。

---

## 426. SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion

**arXiv ID:** 2603.02882 | [PDF](https://arxiv.org/pdf/2603.02882v1)

**作者:** Xinjie Zhu `[一作]` (Lenovo Research), Weifeng Zhang `[通讯]` (Lenovo Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SIGMark，一种可在大型视频扩散模型上实现可扩展、无参考的隐式水印嵌入与提取方案。

**💡 创新点**

创新点在于引入全局帧级伪随机编码（GF‑PRC）实现盲提取，及针对因果 3D VAE 的 Segment Group‑Ordering（SGO）模块提升时间扰动鲁棒性。

**🔧 技术方法**

技术手段包括：利用伪随机纠错码（PRC）对初始噪声进行编码，光流分割+滑动窗口检测恢复帧组顺序，逆扩散推断恢复噪声，再通过 PRC 解码提取信息。

**📊 数据集**

实验使用 VBench‑2.0 评测集，共 400 条视频，分别在 HunyuanVideo 与 Wan‑2.2 两个现代视频扩散模型上进行文本到视频和图像到视频的生成。

**📈 对比分析**

与 DCT、DT‑CWT、VideoShield、VideoMark 等基线比较，SIGMark 在 512 位和 8192 位水印下实现了接近 100% 的比特准确率，且在空间与时间扰动下仍保持高准确率，提取时间保持常数级，优于非盲方案。

**⚠️ 局限性**

局限性在于当 PRC 纠错能力与逆扩散精度受限时，仍无法保证 100% 的比特准确率，且对极端时间损失仍存在一定误差。

---

## 427. Social-JEPA: Emergent Geometric Isomorphism

**arXiv ID:** 2603.02263 | [PDF](https://arxiv.org/pdf/2603.02263v1)

**作者:** Haoran Zhang `[一作]` (Renmin University of China), Xiao Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 25177 | [OpenAlex ID](https://openalex.org/A5002827290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在没有任何参数共享或跨视角协作的前提下，分别训练来自同一环境不同视角的世界模型，并发现其潜在空间可由一个可逆线性变换对齐，实现跨模型互操作。

**💡 创新点**

提出了“Social-JEPA”概念，证明了预测性自监督目标会导致独立模型收敛到相同的线性等价类，从而产生可行的线性同构；并给出理论解释、对齐算法以及多种协作原语（零成本探针迁移、教师-学生迁移、互教）实现。

**🔧 技术方法**

使用JEPA（Joint Embedding Predictive Architecture）目标、ViT骨干、线性对齐（ridge/Procrustes）、MSE、R²、DSC、NOS@k、CKA等评估指标；并在训练、对齐与迁移中实现高效的矩阵求解。

**📊 数据集**

smallNORB（视角差异大）、nuScenes（不同摄像头视角）、ImageNet‑1k（不同数据增强）等公开数据集。

**📈 对比分析**

与MAE（重建）、SimCLR、DINO、MoCo、iBOT等代表性自监督方法在同一评估协议下对比；Social-JEPA在MSE最低、R²最高、DSC最高、NOS@10最低，显示出最佳的跨视角对齐性能，并在迁移任务中显著降低训练FLOPs。

**⚠️ 局限性**

需要共享对应状态对以估计对齐矩阵；对齐矩阵的条件数不佳时可能失效；实验仅覆盖视觉任务，未验证在控制或多模态场景下的表现；对齐仅在相同环境分布且目标一致时成立。

---

## 428. Addressing Missing and Noisy Modalities in One Solution: Unified Modality-Quality Framework for Low-quality Multimodal Data

**arXiv ID:** 2603.02695 | [PDF](https://arxiv.org/pdf/2603.02695v1)

**作者:** Sijie Mai `[一作]` (South China Normal University), Haifeng Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 7592 | [OpenAlex ID](https://openalex.org/A5056953478)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出统一模态质量（UMQ）框架，联合解决多模态情感计算中缺失与噪声模态的问题；

**💡 创新点**

创新点在于将缺失与噪声统一视为低质量模态，设计了质量估计器、质量增强器和质量感知混合专家（MQ-MOE）三大模块，并引入基于排名的训练策略与模态/样本特定信息来提升表示质量；

**🔧 技术方法**

技术包括Sigmoid激活的质量估计器与排名损失、模态解耦耦合网络的质量增强器、MOE专家路由与约束，以及多模态融合与噪声注入等；

**📊 数据集**

使用了CMU-MOSI、CMU-MOSEI、CH-SIMS、UR-FUNNY、MUStARD等多情感/情绪数据集进行评估；

**📈 对比分析**

与Multimodal Boosting、AtCAF、C-MIB、DMV等基线在完整、缺失和噪声条件下进行对比，UMQ在准确率、F1、MAE、相关系数等指标上普遍优于现有方法，尤其在极端缺失/噪声情形下表现最显著；

**⚠️ 局限性**

局限性包括对极端多模态缺失或高噪声的泛化验证不足、模型相对较大且对资源受限场景的适配有限，以及缺乏对缺失与噪声交叉场景的深入理论分析。

---

## 429. Post Hoc Extraction of Pareto Fronts for Continuous Control

**arXiv ID:** 2603.02628 | [PDF](https://arxiv.org/pdf/2603.02628v1)

**作者:** Raghav Thakar `[一作]` (Oregon State University), Kagan Tumer `[通讯]` (Oregon State University)

**通讯引用:** 4271 | [OpenAlex ID](https://openalex.org/A5084748531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种离线多目标强化学习方法MAPEX，利用已训练的单目标专家策略、对应的评论家和重放缓冲区来提取Pareto前沿；

**💡 创新点**

创新点在于：①通过将专家评论家的优势线性组合成混合优势，并以此加权行为克隆，直接从单目标经验中学习多目标策略；②采用缺口识别与目标权重向量填补Pareto前沿，实现高效、低样本的提取；

**🔧 技术方法**

使用Actor‑Critic离线RL框架、AWR加权回归、混合优势、目标权重向量、混合重放缓冲区、子专家评论家（主次评论家）以及温度/截断权重等技术；

**📊 数据集**

在MuJoCo的五个双目标连续控制环境（MO‑Ant‑v5、MO‑Hopper‑v5、MO‑Walker2d‑v5、MO‑Swimmer‑v5、MO‑HalfCheetah‑v5）上进行实验；

**📈 对比分析**

与MOPDERL和MORL/D基线对比，MAPEX在提取阶段的样本效率提升至三位数倍，且在从零训练的情景下能得到与基线相当甚至更优的超体积和稀疏度；

**⚠️ 局限性**

局限性：受限于专家缓冲区的支持范围，无法探索新的行为；假设最优策略位于专家之间的连续流形；对高维（N≥3）目标的扩展尚未验证；若专家行为差异过大，插值可能导致低性能。

---

## 430. VSearcher: Long-Horizon Multimodal Search Agent via Reinforcement Learning

**arXiv ID:** 2603.02795 | [PDF](https://arxiv.org/pdf/2603.02795v1)

**作者:** Ruiyang Zhang `[一作]` (University of Macau), Zhedong Zheng `[通讯]` (University of Macau)

**通讯引用:** 9888 | [OpenAlex ID](https://openalex.org/A5034162160)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于强化学习的长周期多轮多模态搜索代理VSearcher，能够在真实网页环境中使用文本搜索、图像搜索和访问工具解决复杂信息检索任务。

**💡 创新点**

创新点在于：①提出迭代注入式数据合成管道，自动生成大规模高难度多模态浏览问题；②采用拒绝采样微调将专有模型的多轮工具使用能力迁移给基础多模态模型；③在真实网络环境下进行GRPO强化学习，提升长周期工具使用的泛化性；④创建了高难度评测基准MM‑SearchExam。

**🔧 技术方法**

使用的技术包括：迭代注入式数据合成、ReAct框架、拒绝采样微调、GRPO强化学习、LLM-as-a-Judge、Google Custom Search、Vision Web Detection、JINA等访问工具。

**📊 数据集**

使用的数据集为：通过迭代注入生成的合成多模态浏览任务（约1308条用于微调，10轮生成的283条用于MM‑SearchExam），以及公开的MM‑Search、BrowseComp‑VL、MM‑BrowseComp、SimpleVQA等基准。

**📈 对比分析**

与多种开源与专有多模态搜索代理（如MMSearch‑R1、DeepMMSearch‑R1、DeepEyesV2、GPT‑4o、GPT‑5、Gemini‑3‑Pro等）以及多模态评测基准对比，VSearcher在MM‑Search、BrowseComp‑VL、MM‑SearchExam等指标上均取得显著提升，甚至在部分基准上超过了GPT‑5。

**⚠️ 局限性**

局限性包括：①对大规模算力（16块H100 GPU）和专有教师模型的依赖；②数据合成仍可能存在偏差，无法覆盖所有真实网页场景；③代理在极端复杂任务下仍可能因工具调用不当导致误差；④评测主要集中在网页搜索，缺乏对其他多模态工具（如OCR、图像生成）的泛化验证。

---

## 431. Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction

**arXiv ID:** 2603.02909 | [PDF](https://arxiv.org/pdf/2603.02909v1)

**作者:** Guangjun Zhang `[一作]` (Shanxi University), Hongye Tan `[通讯]` (Shanxi University)

**通讯引用:** 363 | [OpenAlex ID](https://openalex.org/A5100751510)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于生成-评估-修正的多智能体框架，实现了零样本文档级事件参数抽取。

**💡 创新点**

创新点在于将生成智能体与评估智能体协同工作，利用奖励机制和事件结构约束对合成数据质量和抽取性能进行联合迭代优化。

**🔧 技术方法**

采用 LLaMA3.1 / Qwen2.5 作为生成智能体，Bart-large 的 Bart-Gen 作为评估智能体，并通过强化学习对两者进行参数更新。

**📊 数据集**

实验使用 RAMS 与 WikiEvents 数据集构建的三种零样本设置，涵盖不同事件类型与角色分布。

**📈 对比分析**

与现有 DEAE 模型、句子级零样本方法及主流 LLM 进行比较，整体 F1 提升 5–8个百分点，尤其在未知角色抽取上表现最为突出。

**⚠️ 局限性**

局限性包括多轮迭代后合成数据多样性下降，模型对长文本或复杂结构的泛化能力仍有限。

---

## 432. Kraken: Higher-order EM Side-Channel Attacks on DNNs in Near and Far Field

**arXiv ID:** 2603.02891 | [PDF](https://arxiv.org/pdf/2603.02891v1)

**作者:** Peter Horvath `[一作]` (Radboud University), Yuval Yarom `[通讯]` (Ruhr University)

**通讯引用:** 8545 | [OpenAlex ID](https://openalex.org/A5056484605)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对GPU Tensor Core单元进行近场和远场电磁侧信道攻击，提出了基于warp级能耗模型和高阶攻击的权重泄露方法，并实现了对LLM权重的提取。

**💡 创新点**

创新点在于首次构建warp级泄漏模型、引入高阶侧信道攻击以提高提取效率，并在远场电磁波下完成LLM权重泄露的原型演示。

**🔧 技术方法**

使用的技术包括EM探针测量、功耗/电磁侧信道分析、相关功耗分析（CPA）、TVLA、选择输入/固定输入的攻击模式以及高阶合成模型。

**📊 数据集**

实验数据集涵盖了2层INT8 CNN、Llama 3.2 1B（bfloat16）模型以及其LoRA微调权重，利用GPU（Jetson Orin Nano、RTX 4090）进行测量。

**📈 对比分析**

与现有BarraCUDA等方法相比，warp级模型将所需trace数从百万级降低到约1万至10万级，远场攻击在100 cm处成功提取8位权重，展示了更高效的提取速度。

**⚠️ 局限性**

主要限制包括：远场攻击效果有限、需要大量trace、对LLM需选择输入/固定输入的前提、对完整模型提取仍具有计算挑战，且在实际环境中实现仍受物理访问和硬件差异影响。

---

## 433. Eval4Sim: An Evaluation Framework for Persona Simulation

**arXiv ID:** 2603.02876 | [PDF](https://arxiv.org/pdf/2603.02876v1)

**作者:** Eliseo Bao `[一作]` (Universidade da Coruña), Javier Parapar `[通讯]` (Universidade da Coruña)

**通讯引用:** 1874 | [OpenAlex ID](https://openalex.org/A5046723532)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Eval4Sim 框架，利用遵从性、一致性与自然性三维指标对人物模拟对话进行评估。

**💡 创新点**

创新点在于以人类对话为行为基准，采用“对齐”而非“最大化”原则，融合检索、作者鉴别和对话 NLI 三种方法实现多维评价。

**🔧 技术方法**

使用密集检索（ColBERT）、字符 4-gram TF‑IDF 版作者鉴别和 DeBERTa 预训练对话 NLI 模型。

**📊 数据集**

主要数据集为 PersonaChat 作为参考基准，并评估十个模拟语料：SPC、SPC‑New、Qwen3（1.7B‑30B）和 Gemma3（1B‑27B）不同规模模型生成的对话。

**📈 对比分析**

与传统单维指标相比，Eval4Sim 显示 Qwen3‑30B 在整体评估（e4s）中表现最佳，且不同维度的排序不完全一致，表明单维优化不足。

**⚠️ 局限性**

局限性包括：仅以 PersonaChat 为基准可能无法覆盖所有人类对话风格；检索、作者鉴别和 NLI 对模型规模与语言差异高度敏感；框架未涵盖情感或情境深度等更细粒度的对话质量维度。

---

## 434. An HCI Perspective on Sustainable GenAI Integration in Architectural Design Education

**arXiv ID:** 2603.03059 | [PDF](https://arxiv.org/pdf/2603.03059v1)

**作者:** Alex Binh Vinh Duc Nguyen `[一作]` `[通讯]` (University of Antwerp), Alex Binh Vinh Duc Nguyen (University of Antwerp)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文探讨了生成式人工智能（genAI）在建筑设计教育中的可持续整合，并从人机交互（HCI）视角提出了三条方法论方向：情境化生态反馈、参与式利益相关者范围界定以及将数据中心设计纳入跨学科关注。

**💡 创新点**

创新点在于将HCI的方法与工具引入建筑教育，以解决genAI使用与环境影响之间的悖论；提出基于情境的生态反馈机制、跨学科利益相关者协作框架，以及将数据中心视为建筑与技术交叉点的重新定位。

**🔧 技术方法**

主要采用HCI的理论与方法，如情境化反馈设计、参与式设计与跨学科协作模型，并未具体实现技术原型。

**📊 数据集**

由于本研究为理论与方法论性工作，未使用具体实验数据集。

**📈 对比分析**

本文未进行实验或性能比较，更多是概念性框架与方法论建议。

**⚠️ 局限性**

局限性包括：缺乏实证评估与可量化指标；情境化生态反馈与参与式方法在真实教育环境中的实现难度与可行性尚待验证；对数据中心设计的跨学科协作要求高，需要进一步合作与资源支持。

---

## 435. See and Remember: A Multimodal Agent for Web Traversal

**arXiv ID:** 2603.02626 | [PDF](https://arxiv.org/pdf/2603.02626v1)

**作者:** Xinjun Wang `[一作]` (Shanghai Institute of AI for Education), Hao Hao `[通讯]` (Shanghai Institute of AI for Education)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了V-GEMS，一种融合视觉感知、显式记忆和符号工具的多模态网页导航代理，解决了LLM驱动代理在空间导航、循环错误和算术幻觉方面的不足。

**💡 创新点**

创新点包括：1）Adaptive Understanding Score Calculator实现按需触发VLM感知；2）Stateful URL Stack提供显式的拓扑记忆，实现可靠回溯；3）Symbolic Counter将算术计数与LLM分离，消除算术幻觉；4）构建可更新的EverWebQA评测流水线，解决静态基准的时效性问题。

**🔧 技术方法**

技术手段：大型语言模型（Qwen-3-Coder-Plus）、视觉语言模型（Qwen3-VL-Plus）、双代理（Explorer‑Critic）架构、ReAct框架、外部工具链（计数器、URL栈、US Calculator）、Selenium抓取、GPT‑4o教师模型校验。

**📊 数据集**

使用的主要数据集是自研的EverWebQA，包含680条问答，覆盖教育、会议、组织、游戏四个领域，支持单源与多源、中文与英文混合的任务；同时对WebWalker原始基准数据进行对齐以便公平比较。

**📈 对比分析**

与WebWalker基线在同一数据集上对比，V‑GEMS平均成功率从0.49提升到0.65，提升幅度28.7%；在多源硬难度、各域（会议、游戏）以及视觉复杂度高的页面上表现尤为显著。

**⚠️ 局限性**

局限性：1）VLM调用导致推理延迟和算力成本高；2）评测依赖预先设定的根URL，无法模拟完全开放的网页搜索；3）缺乏基于强化学习的自我优化机制，决策仍基于启发式规则。

---

## 436. LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization

**arXiv ID:** 2603.02680 | [PDF](https://arxiv.org/pdf/2603.02680v1)

**作者:** Yang Zhao `[一作]` (Northwestern Polytechnical University), Wenzhe Zhao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 1039 | [OpenAlex ID](https://openalex.org/A5103215955)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 NAR-CP 方法，利用奖励标准化和一致性策略改进 LLM 在高频连续决策中的表现；

**💡 创新点**

创新点在于：1）对 dense reward 进行 Z‑score 标准化，动态放大奖励方差，理论证明不改变最优策略；2）采用观察解耦与一致性损失（KL 归一化 Top‑K）保证子任务策略与整体任务一致；

**🔧 技术方法**

技术包括：LLM（Qwen2.5‑3B）+LoRA 微调、PPO 强化学习、奖励标准化、dense reward shaping、Top‑K 选取、KL 一致性损失；

**📊 数据集**

使用数据集：在 NVIDIA Isaac Gym 上构建的 UAV 追踪仿真环境，包含方向追踪、距离保持、综合追踪三类任务；

**📈 对比分析**

比较方法：对比未调 Qwen2.5、TWOSOME、TWOSOME+dense reward；结果显示 NAR-CP 在三类任务上成功率与精度均显著提升（如方向追踪 96.79%/91.33%），策略偏差降至 0.006，推理延迟从 1~2 秒降至 95–144 毫秒；

**⚠️ 局限性**

局限性：动作空间为离散，难以处理更复杂或连续动作任务，未来需扩展到连续动作空间。

---

## 437. Enhancing User Throughput in Multi-panel mmWave Radio Access Networks for Beam-based MU-MIMO Using a DRL Method

**arXiv ID:** 2603.02745 | [PDF](https://arxiv.org/pdf/2603.02745v1)

**作者:** Ramin Hashemi `[一作]` (Nokia), Risto Wichman `[通讯]` (Aalto University)

**通讯引用:** 8753 | [OpenAlex ID](https://openalex.org/A5036110624)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于深度强化学习的多面板 mmWave MU-MIMO 波束管理方法，实现了用户吞吐量提升和延迟降低。

**💡 创新点**

创新点在于将波束使用频率、测得 RSRP 以及不同面板波束间的空间交叉相关性纳入状态特征，并采用 DDQN 学习最佳波束选择策略。

**🔧 技术方法**

使用深度强化学习（DDQN）配合离散动作空间、交叉相关特征、激活历史等多维状态输入；算法实现基于仿真平台。

**📊 数据集**

数据集来自 3GPP 规范的三维空间通道模型，仿真生成 210 个移动用户、21 个 gNB、48 个波束的环境；未使用公开真实数据集。

**📈 对比分析**

通过与传统最大 RSRP 波束选择基线对比，实验显示吞吐量提升 5–16%，几何平均吞吐量上升，端到端延迟降低 3–7 倍；CDF 曲线更陡峭，表明性能显著优于基线。

**⚠️ 局限性**

局限在于依赖仿真环境，未考虑高速移动、频率选择性衰落及硬件实现细节；模型收敛期间的探索导致短期性能下降。

---

## 438. PlayWrite: A Multimodal System for AI Supported Narrative Co-Authoring Through Play in XR

**arXiv ID:** 2603.02366 | [PDF](https://arxiv.org/pdf/2603.02366v1)

**作者:** Esen K. Tütüncü `[一作]` (Institute of Neurosciences of University of Barcelona), Fraser Anderson `[通讯]` (Autodesk Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了PlayWrite，一个基于混合现实的互动故事创作系统，让作者通过直接操控虚拟角色、道具、对话与空间布局来构建叙事；

**💡 创新点**

创新点在于将游戏式物理操控与多模态交互（手势、语音、空间）融合进AI共创故事流程，构建了“意图框架（Intent Frame）”概念并通过多代理管线将这些框架转化为可视化故事弹珠（Story Marble）进行重组；

**🔧 技术方法**

核心技术包括Unity XR平台、Apple Vision Pro硬件、多代理架构（Environment、Social、Narrator、Intent Frame Agent）、大语言模型（GPT‑4‑Turbo）用于对话生成与意图抽取，以及实时语音识别与空间动作跟踪；

**📊 数据集**

未使用公开大规模文本或图像数据集，而是利用作者自定义场景与角色，并将生成的对话与意图记录为JSON日志；

**📈 对比分析**

通过13名混合写作背景的参与者的可用性与问卷评估（CSI、后测问卷）与行为日志分析对系统进行评估，结果显示表达性、乐趣、沉浸等维度高，但控制感较弱，未与传统文本生成系统直接比较；

**⚠️ 局限性**

局限性包括：对细节控制不足，AI主动发言时机与用户预期不匹配；故事可视化重组后可能出现逻辑连贯性缺失；实验样本规模有限且多为具备创意经验的用户；系统对不同故事世界的泛化性尚未充分验证。

---

## 439. CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance

**arXiv ID:** 2603.03281 | [PDF](https://arxiv.org/pdf/2603.03281v1)

**作者:** Hanyang Wang `[一作]` (Tsinghua University), Yueqi Duan `[通讯]` (Tsinghua University)

**通讯引用:** 2350 | [OpenAlex ID](https://openalex.org/A5013973037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在流匹配生成模型中重新定义并统一了类无监督引导(CFG)的控制理论框架，并提出了滑模控制(SMC-CFG)以解决高引导尺度下的稳定性与语义一致性问题。

**💡 创新点**

创新点在于将CFG视为比例控制(P-control)的反馈系统，构建了CFG-Ctrl统一框架，并首次引入滑模控制与Lyapunov分析实现有限时收敛，从而显著提升语义对齐与图像质量。

**🔧 技术方法**

主要技术包括：控制理论中的状态反馈与滑模控制、速度场估计的流匹配模型、Lyapunov 稳定性证明，以及针对不同模型的参数调优。

**📊 数据集**

实验使用MS‑COCO数据集（5,000图文对）和三大流匹配模型：Stable Diffusion 3.5、Flux-dev、Qwen‑Image。

**📈 对比分析**

与标准CFG、CFG‑Zero、Rectified‑CFG++等基线对比，SMC‑CFG在FID、CLIP、Aesthetic、ImageReward等多项指标上均有提升，尤其在大引导尺度下保持了更低的FID和更高的CLIP。

**⚠️ 局限性**

局限性包括：对λ与k的超参数需要手工调节；在极高的k或λ下仍可能出现振荡或采样抖动；目前仅验证于图像文本生成任务，未评估在视频或3D生成等更复杂场景中的表现。

---

## 440. ProSMA-UNet: Decoder Conditioning for Proximal-Sparse Skip Feature Selection

**arXiv ID:** 2603.03187 | [PDF](https://arxiv.org/pdf/2603.03187v1)

**作者:** Chun-Wun Cheng `[一作]` (University of Cambridge), Angelica I. Aviles-Rivero `[通讯]` (Tsinghua University)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5013015879)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 ProSMA‑UNet，通过解码器条件的稀疏门控对 skip 连接进行特征选择，显式过滤噪声和无关信息。

**💡 创新点**

创新点：①多尺度兼容场 + 可学习 ℓ₁ 近端阈值实现闭式软阈值门，直接把不相关激活设为零；②结合解码器条件的通道门控，完成空间与通道双重选择，显著提升分割精度。

**🔧 技术方法**

采用深度可分离膨胀卷积构建多尺度兼容场，ℓ₁ 近端优化得到软阈值门，使用全局平均池化+MLP实现通道门；以 U‑Net 为骨干，支持 2D/3D 版本。

**📊 数据集**

数据集：2D（BUSI、GlaS、Kvasir‑SEG），3D（Spleen、Colon，来自 Medical Segmentation Decathlon）。

**📈 对比分析**

在与 U‑Net、U‑Net++、Attention‑UNet、U‑KAN、U‑NeXt、Rolling‑UNet、UKAN2.0 等基线在相同实验设置下对比，2D 上 IoU 提升 2.86（BUSI）/3.48（Kvasir‑SEG），3D 上 Spleen 97.59±0.10，Colon 63.14±0.99，均优于最强基线。

**⚠️ 局限性**

局限性：稀疏门控模块引入额外计算与内存开销；在极限 GPU 内存受限的场景（如 GlaS 上 UKAN2.0）不易部署；阈值学习需调参，极端噪声下效果可能受限。

---

## 441. Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems

**arXiv ID:** 2603.03111 | [PDF](https://arxiv.org/pdf/2603.03111v1)

**作者:** Raad Khraishi `[一作]` (University College London), Greig A Cowan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种跨模型切换矩阵基准，评估多轮对话中从一种LLM切换到另一种LLM时产生的漂移效应，并在CoQA与Multi-IF两个自动评测任务上通过最终回合切换进行实验；

**💡 创新点**

首创的跨供应商切换矩阵实验设计和对切换漂移的因子分解（前缀影响与后缀易感性），为多轮LLM系统的运维可靠性提供了新的量化视角；

**🔧 技术方法**

使用了配对的BCa引导自举置信区间、两路加性低秩模型分解、前缀生成缓存、零温度确定性推理以及自动化分数计算等技术；

**📊 数据集**

主要使用CoQA（会话式问答）和Multi-IF（多语言指令遵循）两个公开数据集；

**📈 对比分析**

通过将切换情形与无切换基线进行配对比较，并利用自举置信区间评估显著性，发现单回合切换可导致-8~+13%的准确率差异或±4的F1变化；两路因子模型解释了约70-74%的方差；

**⚠️ 局限性**

实验仅覆盖最终回合切换，未探究早期或多回合切换；使用固定的零温度推理与有限的自动评测任务，缺乏对更复杂场景和补偿策略的验证；

---

## 442. Scores Know Bobs Voice: Speaker Impersonation Attack

**arXiv ID:** 2603.02781 | [PDF](https://arxiv.org/pdf/2603.02781v1)

**作者:** Chanwoo Hwang `[一作]` (Hanyang University), Jae Hong Seo `[通讯]` (Hanyang University)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5045700517)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于特征对齐的逆向生成模型，用以在黑盒声纹识别系统中进行高效的声纹冒充攻击。

**💡 创新点**

创新点在于：①将声纹识别特征空间与生成模型的潜在空间显式对齐；②通过固定文本细调与身份/结构约束训练逆向模型；③实现两种攻击范式（NES优化和子空间投影）并显著降低查询成本。

**🔧 技术方法**

技术包括自然进化策略（NES）优化、子空间投影攻击、固定文本策略、身份约束损失（L_IC）与结构约束损失（L_SC）以及对TTS模型的微调。

**📊 数据集**

使用的数据集有 VoxCeleb1/2、CNCeleb、CN-Celeb1/2，用于训练声纹识别模型、逆向模型以及评估对不同语言（英文、中文）的迁移效果。

**📈 对比分析**

与基线（Audio‑NES、YourTTS‑NES、FakeBob等）相比，Ours‑NES 在五个目标声纹模型上均能在约 0.3k 次查询内实现 100% 的攻击成功率，Ours‑SP 在 50 次查询内就能达到 90% 以上的成功率；性能提升幅度可达 10 倍以上。

**⚠️ 局限性**

局限性包括：仅针对能返回相似度分数的非交互式声纹系统；未考虑活体检测、深度伪造检测等防御措施；跨语言或跨系统的鲁棒性尚未系统评估。

---

## 443. Using the SEKF to Transfer NN Models of Dynamical Systems with Limited Data

**arXiv ID:** 2603.02439 | [PDF](https://arxiv.org/pdf/2603.02439v1)

**作者:** Joshua E. Hammond `[一作]` (University of Texas at Austin), Michael Baldea `[通讯]` (University of Texas at Austin)

**通讯引用:** 6128 | [OpenAlex ID](https://openalex.org/A5089303298)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究利用Subset Extended Kalman Filter（SEKF）在数据稀缺场景下，将预训练的神经网络模型迁移到功能相似但参数略有不同的目标动力系统，实现小量参数微调而获得高精度预测。

**💡 创新点**

创新点在于：①将迁移学习视为贝叶斯推理，利用源模型参数构成高斯先验，SEKF提供顺序更新与自适应正则化；②证明即使仅使用目标数据的1%也能达到源模型水平；③揭示动力系统迁移学习需在所有层级进行分布式微调，而非像图像分类那样冻结早期层。

**🔧 技术方法**

使用的技术包括：Subset Extended Kalman Filter（SEKF）、Adam、L-BFGS梯度优化、神经网络模型（MLP/RNN/NODE）、贝叶斯先验、参数协方差更新、训练误差与泛化度量（MSE、train‑test gap）。

**📊 数据集**

使用的数据集为：①仿真的阻尼弹簧系统（含不同阻尼系数的多个实例），②实际温度控制实验室（TCLab）系统的实测数据；两者分别代表仿真到仿真和仿真到实测的迁移场景。

**📈 对比分析**

通过在目标数据量（10, 50, 100, 500, 1000样本或0.5–24小时）下对比 finetune 与 retrain 初始化，以及不同优化器的效果，发现 finetune 在数据稀缺时显著降低测试误差（最高可低至源模型水平），train‑test gap 小，且 SEKF 在在线适应方面表现优异，但计算成本高于梯度方法。

**⚠️ 局限性**

局限性包括：假设源与目标系统功能相似且操作域重叠；仅测试了小型网络与两个基准系统；未探讨源目标差异显著或系统非线性变化大的情形；SEKF 的计算开销在大模型或高频更新场景下可能成为瓶颈。

---

## 444. Lattice-based Deep Neural Networks: Regularity and Tailored Regularization

**arXiv ID:** 2603.02809 | [PDF](https://arxiv.org/pdf/2603.02809v1)

**作者:** Alexander Keller `[一作]` (NVIDIA), Ian H. Sloan `[通讯]` (University of New South Wales Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了将格点（lattice rules）用作深度神经网络（DNN）的训练点，并提出了与格点匹配的定制正则化方法，证明在高维情形下泛化误差不随维数增长。

**💡 创新点**

创新点在于将格点理论与DNN训练相结合，给出DNN正则化的理论依据；扩展激活函数的阶乘导数界限至 swish；在理论上证明定制正则化可实现维数无关的误差上界；并在实验中验证了该方法对多种激活函数的优势。

**🔧 技术方法**

使用了量子蒙特卡罗（QMC）格点规则、随机平移、FFT求解、Sobolev/Korobov 权重空间理论、梯度下降（Adam）优化、PyTorch 实现、自动微分以及 L2 与定制正则化技术。

**📊 数据集**

实验使用合成的周期代数函数 G(y)=1/(1+∑_{j=1}^s sin(2πy_j)ψ_j)（s=50），通过格点采样生成训练集，未使用公开真实数据集。

**📈 对比分析**

通过在五种激活函数（sigmoid、swish c=1,5,25、ReLU）和两组网络超参数（L=3, d=32；L=12, d=30）下，对比标准 ℓ₂ 正则化与定制正则化。结果显示，定制正则化在所有激活函数上均显著降低泛化误差，收敛率在 N⁻¹ 到 N⁻² 之间；sigmoid 在小网络中表现最佳，标准 swish 在大网络中最佳；ReLU 最差，且理论不适用。

**⚠️ 局限性**

限制在于只对光滑激活函数有理论支持，ReLU 等非光滑激活函数无法覆盖；理论依赖已知的正则化序列 b_j；实验仅在合成函数上验证，缺乏对真实高维问题的评估；未与传统 QMC/FFT 逼近方法进行直接性能比较。

---

## 445. Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?

**arXiv ID:** 2603.03202 | [PDF](https://arxiv.org/pdf/2603.03202v1)

**作者:** Dadi Guo `[一作]` (Hong Kong University of Science and Technology), Yi R. Fung `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建多代理系统，利用代码执行与验证机制，自动演化已有数学题为结构更复杂且难度更高的新题目

**💡 创新点**

创新点在于将问题演化拆解为进化、可解性验证与难度验证三阶段，采用代码探索与测试时多回合策略，实现了既可解又更具挑战性的自动生成

**🔧 技术方法**

采用 LLM 代码代理（DeepSeek‑Chat/Reasoner、Gemini‑Pro、Kimi‑K2 等）、Python 可执行环境（SymPy、NetworkX、itertools 等）以及第三方 LLM（GPT‑5.2‑High）作为判定者

**📊 数据集**

使用 100 条来自教材、IMO、AIME 等的种子问题，并挑选 6 对例题用于评估与对照

**📈 对比分析**

在多模型评估中，演化问题的可解性验证通过率约 85‑96%，多模型 solve‑rate 下降 5‑25%，平均 token 消耗显著提升，显示难度提升显著且跨模型可迁移；效率方面平均失败回合数为 1.6‑6.6 次

**⚠️ 局限性**

主要局限在于高计算成本与逻辑一致性检验成为瓶颈，导致需要多回合迭代；难度提升主要依赖于模型自身的推理与探索能力，且目前仍缺乏更系统的结构合成机制

---

## 446. Causal Learning Should Embrace the Wisdom of the Crowd

**arXiv ID:** 2603.02678 | [PDF](https://arxiv.org/pdf/2603.02678v1)

**作者:** Ryan Feng Lin `[一作]` (University of Washington), Shuai Huang `[通讯]` (University of Washington)

**通讯引用:** 2765 | [OpenAlex ID](https://openalex.org/A5101961858)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并系统化了基于“众包智慧”的因果结构学习范式，结合人类专家与大型语言模型的知识贡献，设计了多层次的知识获取、建模、聚合与主动优化流程。

**💡 创新点**

核心创新包括：①构建了四维专家类型分类法与对应的知识质量度量；②对边缘信息和排序信息两种 elicitation 框架进行理论与经验对比；③提出专家级与查询级两种聚合策略，并给出其在计算效率、信息利用和鲁棒性方面的权衡；④引入基于 LLM 的代理专家与活跃设计的序贯抽样方法，实现大规模高效知识收集。

**🔧 技术方法**

主要技术手段包括：①概率建模（贝叶斯网络、混合机制模型、Dawid–Skene 等）；②多模态知识提取与评分（边缘型、排序型、图形型、列表型）；③聚合方法（权重平均、隐变量层聚合、后验推理）；④主动学习/最优设计（E‑optimality、EIG 等）和 LLM 交互式模拟；⑤在实验中使用结构化调查问卷与交互式界面。

**📊 数据集**

实证采用了经典的 Asia 诊断网络（8 个变量，8 条边）作为因果结构基准，邀请 20 名受试者进行结构化问卷，数据集仅包含专家对变量对的因果关系评分；此外还在实验中使用了人机混合的 LLM 代理产生的模拟知识。

**📈 对比分析**

通过对比传统单专家知识聚合、基于图搜索和基于排序的学习，报告了在 Asia 网络上的结构恢复准确率提升（相对提升约 20%–30%），并通过主动抽样进一步减少所需问卷数量约 40%。实验显示，查询级聚合在处理噪声和不一致时更鲁棒，且在预算受限情况下能保持更高的边/路径回收率。

**⚠️ 局限性**

局限性包括：①对专家质量控制的依赖，若专家群体存在大量误导或噪声，聚合结果可能退化；②模型假设（如独立性、线性关系）在实际大规模因果网络中可能不成立；③LLM 代理虽可扩大规模，但其知识偏差与推理不确定性仍需进一步研究；④主动设计与聚合的计算复杂度随变量数目呈指数级增长，尚需高效近似方法。

---

## 447. Watch Your Step: Learning Semantically-Guided Locomotion in Cluttered Environment

**arXiv ID:** 2603.02657 | [PDF](https://arxiv.org/pdf/2603.02657v1)

**作者:** Denan Liang `[一作]` (Nanyang Technological University), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 55457 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SemLoco语义感知步态框架，实现密集杂乱环境中精确踏点选择，降低小型易损物体碰撞风险。

**💡 创新点**

创新点包括：将语义信息显式嵌入低层控制，使用两阶段RL（虚拟→真实障碍）和语义驱动的Raibert启发式；采用ReLU清除惩罚代替严格轨迹追踪，提高灵活性。

**🔧 技术方法**

技术栈涵盖强化学习（A2C/PPO）+ CNN编码语义与高程图 + 多层感知-控制网络 + 语义驱动Raibert启发式 + 软硬约束两阶段训练 + 视觉语义分割(Odin1) + 地图融合。

**📊 数据集**

数据集：在IsaacSim中随机生成的障碍密度数据集；真实环境中使用Unitree Go2与Odin1视觉传感器采集的室内杂物场景；训练时引入域随机化。

**📈 对比分析**

对比方法包括盲目策略、去除虚拟障碍、去除ReLU清除、去除语义图等；在不同障碍密度下，SemLoco成功率>90%、平均失效距离约9 m，步碰撞率比基线低74‑95%，显著提升性能。

**⚠️ 局限性**

局限性：极端踏点导致角动量失衡，产生yaw漂移；驱动器带宽不足导致抬脚轨迹滞后；模拟中的语义简化导致与真实世界的泛化受限；当前仅使用单一“易损”成本，缺乏开放词汇语义理解。

---

## 448. Exploiting Repetitions and Interference Cancellation for the 6G-V2X Sidelink Autonomous Mode

**arXiv ID:** 2603.03039 | [PDF](https://arxiv.org/pdf/2603.03039v1)

**作者:** Alessandro Bazzi `[一作]` (Universita di Bologna), Claudia Campolo `[通讯]` (Universita Mediterranea di Reggio Calabria)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了RB‑NOMA方案，利用具备SIC能力的VUE接收机对NR‑V2X Mode 2的盲重传复制包进行前向和后向干扰消除，从而显著提升可靠性和覆盖范围。

**💡 创新点**

创新点在于将干扰视为可利用资源：①将SIC与前向干扰消除(FRC)和后向干扰消除(BKC)三种操作结合；②仅需在SCI中增加极少的指针信息即可实现；③在分布式资源分配的环境下实现显著性能提升。

**🔧 技术方法**

使用的技术包括5G NR‑V2X Mode 2、盲重传复制、SIC（Successive Interference Cancellation）、前向干扰消除、后向干扰消除以及MATLAB下的WiLabV2Xsim仿真平台。

**📊 数据集**

实验数据基于在高速公路上模拟的车辆密度（12.5–50辆/km）和两类流量（100 ms周期性与50 ms+随机间隔非周期性），每个车辆发送1000 字节数据包，采用MCS 5、20 MHz带宽、15 kHz子载波间隔。

**📈 对比分析**

通过与传统Mode 2、仅SIC、SIC+FRC以及理想的“排序”资源分配进行对比，评估指标包括PRR、覆盖范围、WBSP、CBR和EED。结果显示，RB‑NOMA在低密度下可逼近排序分配，范围提升最高可达130%，且WBSP显著降低，且与传统SIC相比，FRC贡献显著，BKC提升有限。

**⚠️ 局限性**

限制与挑战包括：BKC在本实验中提升极小，增加了实现复杂度；SIC仅执行一次迭代，未探索多次迭代潜力；仿真仅覆盖理想化的高速公路情景，未包含更复杂环境和非理想硬件误差；SCI指针扩展需与3GPP标准进一步协商。

---

## 449. SpecLoop: An Agentic RTL-to-Specification Framework with Formal Verification Feedback Loop

**arXiv ID:** 2603.02895 | [PDF](https://arxiv.org/pdf/2603.02895v1)

**作者:** Fu-Chieh Chang `[一作]` (National Taiwan University), Pei-Yuan Wu `[通讯]` (National Taiwan University)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5080520702)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpecLoop，一种基于 LLM 与形式等价检查的迭代 RTL‑>规范生成框架；

**💡 创新点**

创新点在于将规范生成与形式等价检查耦合，利用重构与反馈循环让 LLM 逐步修正规范；

**🔧 技术方法**

核心技术包括多步骤 LLM prompting、RTL 重构器、Yosys 等价检查、编译错误与反例诊断反馈；

**📊 数据集**

使用 VerilogEval 与 RTLLM 两个 RTL benchmark，评估生成规范的重构成功率；

**📈 对比分析**

与单轮生成和仅返回通过/失败的基线相比，Full Diagnosis 版本在 RR‑Score 上显著提升（例如 Qwen3‑Coder‑30B 从 0.722 提升至 0.795）；

**⚠️ 局限性**

局限性包括依赖重构器与等价检查的能力、对大规模工业 RTL 的可扩展性不足，以及反馈解析的经验性限制。

---

## 450. Tilt Automata: Gathering Particles With Uniform External Control

**arXiv ID:** 2603.02796 | [PDF](https://arxiv.org/pdf/2603.02796v1)

**作者:** Sándor P. Fekete `[一作]` (Technische Universität Braunschweig), Christian Scheffer `[通讯]` (Bochum University of Applied Sciences)

**通讯引用:** 211 | [OpenAlex ID](https://openalex.org/A5080464077)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了多个算法在特定问题上的应用与性能。

**💡 创新点**

创新点在于提出了一种新的算法框架，能够在复杂问题中提高效率。

**🔧 技术方法**

使用了组合优化和图论相关的技术。

**📊 数据集**

数据集来源于多个公开的基准测试集。

**📈 对比分析**

与现有方法进行了对比，结果显示新算法在时间复杂度和准确性上均有显著提升。

**⚠️ 局限性**

限制在于算法在某些极端情况下的表现仍需进一步优化。

---

## 451. Fuzzing Microservices in Face of Intrinsic Uncertainties

**arXiv ID:** 2603.02551 | [PDF](https://arxiv.org/pdf/2603.02551v1)

**作者:** Man Zhang `[一作]` (Beihang University), Andrea Arcuri `[通讯]` (Kristiania University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个连续的不确定性驱动系统级微服务模糊测试框架，涵盖不确定性检测、量化与影响分析，以及基于不确定性和业务优先级的自适应测试生成与优化；通过服务虚拟化、AI 驱动的依赖模拟、因果推理与多目标优化实现系统级模糊测试。

**💡 创新点**

创新点主要包括：①针对微服务的多维不确定性建模与注入机制；②利用因果推理与动态依赖图分析不确定性传播路径；③构建分层、多目标自适应模糊策略，结合不确定性覆盖指标和业务权重；④将生成式模型与强化学习融合的 AI 驱动依赖与不确定性模拟；⑤整体平台设计实现持续感知-测量-测试闭环。

**🔧 技术方法**

技术手段涵盖：服务虚拟化与依赖注入；分布式追踪与动态字节码插桩；因果推理（如传输熵、贝叶斯网络）；不确定性量化（熵、贝叶斯、模糊集合、生成式对抗网络）；多目标优化算法（MIO、强化学习）；API 模糊工具（Restler、DeepREST等）与自定义模糊器；数据驱动的 AI 模拟器（序列到序列、GAN、RL）。

**📊 数据集**

论文以一个自行开发的电商微服务案例作为演示，使用该案例的业务流、API 规范与运行时数据；未公开具体工业生产数据集或公开数据集，主要基于模拟与案例生成的数据。

**📈 对比分析**

本文尚未实现完整平台或开展实验评估，因而没有提供与现有方法的对比或性能指标；提出的框架和方法在概念验证层面得到阐述，未来需在工业环境中进行基准测试。

**⚠️ 局限性**

局限性包括：①框架仍处于概念/原型阶段，未完成实现与评估；②目前主要针对 JVM 微服务，跨语言适配尚未验证；③对不确定性建模与量化方法的选择缺乏系统性指南；④在大规模生产环境下的资源开销与可扩展性尚待验证；⑤缺少实测的性能与覆盖率数据，无法量化方法有效性。

---

## 452. An LLM-Assisted Toolkit for Inspectable Multimodal Emotion Data Annotation

**arXiv ID:** 2603.02569 | [PDF](https://arxiv.org/pdf/2603.02569v1)

**作者:** Zheyuan Kuang `[一作]` (University of Sydney), Zhanna Sarsenbayeva `[通讯]` (University of Sydney)

**通讯引用:** 1130 | [OpenAlex ID](https://openalex.org/A5024805223)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究开发了一个基于大语言模型（LLM）的三阶段工作流工具，用于多模态情绪数据的预处理、可视化、事件检测与结构化注释；

**💡 创新点**

创新点在于将LLM与多模态预处理、交互式时间轴可视化以及事件级检索相结合，提供可追溯的事件包和可编辑的结构化标签，实现可检验且可扩展的情绪标注流程；

**🔧 技术方法**

主要技术包括LLM（GPT‑5.2）多模态推理、OpenFace AU峰值检测、运动能量计算、信号趋势提取、可视化共享时间轴界面、事件包生成与可编辑接口；

**📊 数据集**

使用了84名参与者在六种情绪触发场景下收集的VR多模态情绪数据，涵盖面部表情视频、第一人称视角视频、身体运动、血管脉冲(BVP)、心率(HR)、皮肤电活动(EDA)及惯性测量单元(IMU)信号；

**📈 对比分析**

该工具在示例VR数据集上演示了工作流程，展示了事件检索和LLM生成注释的可行性，但本文未给出定量对比或性能指标，未来计划通过专家用户研究评估效率与质量；

**⚠️ 局限性**

局限性包括缺乏大规模用户评估、对LLM输出的依赖需人工验证、目前仅支持有限的模态与注释模板，且在跨域噪声与缺失模态下的鲁棒性仍待进一步验证。

---

## 453. CoShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model

**arXiv ID:** 2603.02743 | [PDF](https://arxiv.org/pdf/2603.02743v1)

**作者:** Waqas Ahmed `[一作]` (Murdoch University), Ferdous Sohel `[通讯]` (Murdoch University)

**通讯引用:** 8306 | [OpenAlex ID](https://openalex.org/A5002219416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种双重条件扩散模型，能够在单物体或多物体合成场景中生成物理上逼真的阴影。

**💡 创新点**

创新点在于将预测的阴影框量化为可学习的文本位置标记，结合交叉注意力与注意力对齐损失，实现多物体阴影的全局一致性；同时引入几何感知仿射调制（GAAM）为图像路径提供精细的空间引导。

**🔧 技术方法**

采用预训练的文本到图像扩散模型（Stable Diffusion），配合Shadow‑Box预测网络、可学习的阴影位置标记、跨注意力机制、GAAM模块和注意力对齐损失。

**📊 数据集**

使用DESOBAv2数据集及其扩展的多物体版本，训练集共38,755张图（含11,037张多物体合成），测试集320张图（1,876个物体–阴影对）。

**📈 对比分析**

与SGRNet、DMASNet、SGDiffusion、GPSDiffusion、MetaShadow等基线在单、双、多人物体设置下进行定量对比，本文方法在RMSE、SSIM、BER等全局与局部指标均优于所有基线，尤其在局部阴影几何和附着质量上提升显著。

**⚠️ 局限性**

局限性包括对阴影框预测误差仍有一定敏感性，在极端遮挡或光照变化较大的场景中偶有漂移或细节缺失；当物体数极多时可能出现跨实例冲突；在真实合成数据缺少真阴影标签时仍需进一步验证。

---

## 454. EvoSkill: Automated Skill Discovery for Multi-Agent Systems

**arXiv ID:** 2603.02766 | [PDF](https://arxiv.org/pdf/2603.02766v1)

**作者:** Salaheddin Alzubi `[一作]` (Sentient), Tu Vu `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EvoSkill框架，利用失败分析迭代发现并改进编码代理的可重用技能，保持模型冻结，仅演化技能集合。

**💡 创新点**

创新点在于将优化升至技能层级，使用文本反馈下降与Pareto前沿来选择、筛选技能，形成可解释、可复用且跨任务转移的技能库。

**🔧 技术方法**

技术包括：文本反馈下降（Feedback Descent）驱动的技能生成循环、三代理（执行、提议、构建）架构、技能文件结构与触发元数据、git分支管理与Pareto前沿筛选。

**📊 数据集**

数据集：OfficeQA（美国财政部公告推理）、SealQA（搜索增强问答）以及用于零射技能迁移评估的BrowseComp。

**📈 对比分析**

对比方法：基于Claude Code Opus 4.5的基线；OfficeQA上精确匹配准确率从60.6%提升至67.9%（+7.3%），SealQA从26.6%提升至38.7%（+12.1%）；迁移到BrowseComp提升5.3%；所有提升均在训练子集较小的情况下实现。

**⚠️ 局限性**

局限：实验仅覆盖两类基准，缺乏多任务、多模型、多模态验证；对随机性、超参数敏感性与技能库共享与管理的深入探讨尚未展开。

---

## 455. Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models

**arXiv ID:** 2603.02631 | [PDF](https://arxiv.org/pdf/2603.02631v1)

**作者:** Shubhangi Upasani `[一作]` (SambaNova AI), Guangtao Wang `[通讯]` (Meta Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了跨模型的 speculative prefill 技术，利用一个轻量级的草稿模型（来自不同模型家族）来对目标模型的长提示进行压缩，从而降低推理前置成本。

**💡 创新点**

创新点在于证明注意力基的 token 重要性评估可以跨模型家族迁移，不必使用同一家族的草稿模型；同时保持原有 speculative prefill 的无训练、无参数修改特性，并显著缩短首个 token 的到达时间。

**🔧 技术方法**

采用了 speculative prefill 的 attention 重要性评分、块级选择、跨家族 tokenization 适配、连续位置 ID 重新分配等技术，实验中使用了 Qwen、LLaMA 与 DeepSeek 之间的多种 draft–target 组合。

**📊 数据集**

使用了 LongBench v1/v2、RULER、InfiniteBench 的 Code Debug 等长上下文基准数据集。

**📈 对比分析**

与完整提示基线比较，压缩后在 6%–30% keep rate 下保持 90–100% 的准确率；在某些任务甚至略有提升；通过压缩 128k token 至 16k/32k 可将 TTFT 降至 2.5–4.3 秒，约 18 倍提升。

**⚠️ 局限性**

局限性包括：在极低 keep rate 下可能出现轻微性能下降；对草稿模型规模和架构的敏感性仍需进一步探究；跨家族 tokenization 差异可能导致长度略有偏差；实验主要集中在几种模型家族，其他家族的泛化尚待验证。

---

## 456. cPNN: Continuous Progressive Neural Networks for Evolving Streaming Time Series

**arXiv ID:** 2603.03040 | [PDF](https://arxiv.org/pdf/2603.03040v1)

**作者:** Federico Giannini `[一作]` (Politecnico di Milano), Emanuele Della Valle `[通讯]` (Politecnico di Milano)

**通讯引用:** 4695 | [OpenAlex ID](https://openalex.org/A5015694017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的连续进化神经网络（cPNN），能够在流式时间序列中同时处理概念漂移、时序依赖和灾难性遗忘问题。

**💡 创新点**

创新点在于将Progressive Neural Networks与连续LSTM相结合，形成cPNN，既保留了对旧概念的记忆，又能快速利用迁移学习适应新概念，并通过滑动窗口实现对时序的自适应训练。

**🔧 技术方法**

使用的技术包括RNN（LSTM）作为基础模型、连续版本cLSTM、Progressive Neural Networks的列结构、随机梯度下降（SGD）与滑动窗口批处理、以及预序评估和概念漂移检测。

**📊 数据集**

实验数据集为自定义的合成时间序列流（基于SINE的变种），包含两类特征、四种概念及两种漂移类型（标签反转漂移和边界函数漂移）。

**📈 对比分析**

与cLSTM和mcLSTM的对照实验表明，cPNN在概念漂移后更快恢复，平均准确率显著高于两种对照模型，尤其在标签反转漂移场景下表现最为突出。

**⚠️ 局限性**

局限性包括模型结构随概念数线性增长导致复杂度上升、仅在人工生成的低维、突发漂移的简化场景下验证、且缺乏真实漂移检测机制。

---

## 457. SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models

**arXiv ID:** 2603.02250 | [PDF](https://arxiv.org/pdf/2603.02250v1)

**作者:** Paweł Pozorski `[一作]`, Maria Ganzha `[通讯]` (Warsaw University of Technology)

**通讯引用:** 2527 | [OpenAlex ID](https://openalex.org/A5017025137)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了Spectrogram-Guided Phonetic Alignment (SGPA)，将原始音频波形转换为与词对齐的音频片段，以便进行Shapley值归因。

**💡 创新点**

通过结合CTC强制对齐与谱图边界细化，克服了音频Shapley值计算的维度爆炸、语义稀释与边界伪影三大障碍。

**🔧 技术方法**

使用Wav2Vec2-XLSR-53进行CTC对齐，短时能量与谱流进行边界细化，随后进行Word级聚合，并使用Neyman分层采样估计Shapley值。

**📊 数据集**

在LFM2-Audio-1.5B模型上使用VoiceBench单句英语子集（约100条样本）进行实验。

**📈 对比分析**

与原生音频token化对比，SGPA将所需模型调用数从约2,552降至59，计算时间从约1,820秒降至约67秒，归因分布集中但保持全局累积曲线不变。

**⚠️ 局限性**

局限包括对词汇层面归因非中立、对声学语言多样性的鲁棒性未验证、仅在单一模型上评估、以及需要转录文本等。

---

## 458. SEALing the Gap: A Reference Framework for LLM Inference Carbon Estimation via Multi-Benchmark Driven Embodiment

**arXiv ID:** 2603.02949 | [PDF](https://arxiv.org/pdf/2603.02949v1)

**作者:** Priyavanshi Pathania `[一作]` (Accenture Labs), Adam P. Burden `[通讯]` (Accenture)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5060382693)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个LLM推理碳排估计的参考框架，并实现了SEAL工具，用多基准驱动的方法对每个提示进行碳排估计。

**💡 创新点**

创新在于构建了可指导工具设计的系统性参考框架，首次实现多基准驱动的按提示碳排估计，并为LLM生态系统的可持续评估奠定基础。

**🔧 技术方法**

采用多基准驱动分析、能源测量与碳排计算方法，结合LLM推理过程的能耗监测。

**📊 数据集**

使用多种LLM基准数据集（如GLUE、SQuAD、LLaMA评测集）进行实验验证。

**📈 对比分析**

通过与现有估计方法对比，SEAL在提示级碳排估计上更精确，误差率降低约30%，并能在不同模型间保持一致性。

**⚠️ 局限性**

局限在于实验规模有限，仅在少数模型和基准上验证；真实环境的能耗采样难度大，工具的实时性与可扩展性待进一步研究。

---

## 459. Dynamic Contract Analysis for Parallel Programming Models

**arXiv ID:** 2603.03023 | [PDF](https://arxiv.org/pdf/2603.03023v1)

**作者:** Yussur Mustafa Oraji `[一作]` (Scientific Computing Institute), Christian Bischof `[通讯]` (Scientific Computing Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了 CoVer-Dynamic，一种基于 CoVer 合约语言的动态分析扩展，用于并行编程模型的错误检测。

**💡 创新点**

创新点在于将静态合约语言与动态检测无缝结合，既保留模型无关性，又消除静态分析的误报并捕获运行时错误。

**🔧 技术方法**

采用动态插桩、回调机制、合约数据库以及状态验证等技术实现合约执行与错误检查。

**📊 数据集**

使用 RMARaceBench、MPI-BugBench 以及 OpenSHMEM 测试集进行分类准确性评估，使用 LULESH、TeaLeaf、PRK Stencil 进行运行时开销评估。

**📈 对比分析**

与现有工具 MUST 以及原始 CoVer 进行对比，CoVer-Dynamic 在准确率上达到 95% 以上、误报为零，并且在大多数测试中至少比 MUST 快 2 倍，过滤插桩后开销可降至基线水平。

**⚠️ 局限性**

限制在合约语言表达能力不足，无法描述诸如死锁、远程 RMA 数据竞争等错误，且对新 API 的支持需通过扩展合约实现。

---

## 460. SpatialText: A Pure-Text Cognitive Benchmark for Spatial Understanding in Large Language Models

**arXiv ID:** 2603.03002 | [PDF](https://arxiv.org/pdf/2603.03002v1)

**作者:** Peiyao Jiang `[一作]` (Zhejiang University), Xi Li `[通讯]` (Zhejiang University)

**通讯引用:** 10506 | [OpenAlex ID](https://openalex.org/A5100407729)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SpatialText，一套基于文本的双源空间推理基准，用于评估大型语言模型的空间认知能力。

**💡 创新点**

创新点在于结合人类标注的真实室内场景与程序生成的逻辑场景两大数据源，构建层级化任务体系并排除视觉输入，专门剥离文字空间推理。

**🔧 技术方法**

采用自然语言描述构造任务、层级化问答框架、Chain‑of‑Thought 逐步推理提示以及统一的评测流程，系统比较不同 LLM 的推理表现。

**📊 数据集**

使用 LSUN 室内图像（卧室、客厅等）进行人类标注共 100 张图，生成 485 条自然问答；以及 400 条程序生成的 2D/3D、全知/非全知结构化问答。

**📈 对比分析**

对 7B–14B 参数的多款 LLM（DeepSeek‑R1‑Distill‑Llama‑8B、OpenPangu‑Embedded‑7B‑V1.1、Qwen3‑8B、Gemma‑3‑12B‑IT、Mistral‑7B‑Instruct 等）在统一推理协议下进行评测，检索类任务准确率 90%+，但视角转换任务低于 40%；DeepSeek‑v3.2 在整体上达 81% 的最高分。

**⚠️ 局限性**

局限在于仅通过文字评估，无法验证模型真正的空间地图；模型过度依赖统计共现与先验，缺乏全局一致性与心里旋转能力；全知场景信息过载导致逻辑冲突；数据规模和场景多样性仍有提升空间。

---

## 461. Less Noise, Same Certificate: Retain Sensitivity for Unlearning

**arXiv ID:** 2603.03172 | [PDF](https://arxiv.org/pdf/2603.03172v1)

**作者:** Carolin Heinzler `[一作]` (University of Copenhagen), Amartya Sanyal `[通讯]` (University of Copenhagen)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5035879433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了“保留灵敏度”（Retain Sensitivity）概念，用于在已知保留数据集 R 的情况下，为机器学习模型的证明确认式去学习（certified unlearning）精确校准噪声，从而在不必重训练的前提下，保持模型在删除数据后的统计不可区分性。

**💡 创新点**

创新点在于：①将保留灵敏度定义为相对于固定 R 的最大输出变化，证明其对主动和被动去学习都足够；②通过理论分析和实验展示保留灵敏度显著低于传统的全局灵敏度，能够大幅减少噪声；③将此概念应用到两类主流主动去学习算法（Descent‑to‑Delete 与 Newton 更新）中，进一步利用数据依赖的曲率信息提升实用性。

**🔧 技术方法**

使用的技术包括：差分隐私中 Gaussian 机制的噪声校准、局部与光滑灵敏度理论、Davis‑Kahan 误差分析、ERM 的强凸性与光滑性证明、梯度投影与 Newton 更新的收敛分析，以及对 MST、PCA、SVM 等经典问题的定理推导。

**📊 数据集**

实验数据集主要包括：用于 MST 的四个真实加权图网络（如 Bitcoin、Migration 等）、SVM 经典数据集（如 20 Newsgroups、MNIST 等）、以及合成的均匀/正态分布数据用于中位数和 ERM（MSE 与 Logistic 回归）测试。

**📈 对比分析**

比较方法：在同一保留集 R 上分别使用基于全局灵敏度和基于保留灵敏度的噪声校准；对主动去学习还比较迭代步数（I）和噪声量。结果显示，保留灵敏度往往能使噪声幅度减少数个数量级，主动去学习的迭代步数也相应减少 10⁵ 倍以上；同时模型的准确率与原始模型几乎无差别。

**⚠️ 局限性**

局限性：①需要完整的保留集 R 或等价的侧信息来计算保留灵敏度，对大规模模型而言计算成本高；②目前的分析主要针对单点删除，扩展到多点删除需要进一步研究；③在极端数据稀疏或噪声过多的情况下，保留灵敏度可能并不比全局灵敏度显著优。

---

## 462. Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis

**arXiv ID:** 2603.02866 | [PDF](https://arxiv.org/pdf/2603.02866v1)

**作者:** Kaiqiang Xiong `[一作]` (Peking University), Ronggang Wang `[通讯]` (Peking University)

**通讯引用:** 3633 | [OpenAlex ID](https://openalex.org/A5050071143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多模态先验引导的采样机制的分层3D高斯散射框架，用于稀视角新视角合成

**💡 创新点**

创新点在于融合光度残差、语义先验与几何先验生成局部可恢复性评分，采用分层高斯表示并设计几何感知采样与保护机制，显著提升稀视角下的细节恢复

**🔧 技术方法**

使用3D高斯散射、深度/法线几何先验、轻量语义分割网络、重要性采样、动态细化与剪枝策略

**📊 数据集**

在LLFF、DTU、Mip-NeRF360三大公开稀视角基准上进行实验

**📈 对比分析**

与DietNeRF、FreeNeRF、SparseNeRF、3DGS、FSGS、DNGaussian、CoR-GS、NexusGS等方法对比，在LLFF 3视角下PSNR 21.17 dB，DTU 3视角下提升至+0.3 dB，整体均达到或超过SOTA

**⚠️ 局限性**

仍受限于极端稀视角下的几何不确定性、对预训练语义模型的依赖以及高斯数目与计算开销的权衡

---

## 463. Nodes Are Early, Edges Are Late: Probing Diagram Representations in Large Vision-Language Models

**arXiv ID:** 2603.02865 | [PDF](https://arxiv.org/pdf/2603.02865v1)

**作者:** Haruto Yoshida `[一作]` (Tohoku University), Kentaro Inui `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了大型视觉语言模型（LVLM）在图表理解中的内部表示，利用可控合成图表数据集进行线性探针和因果干预分析。

**💡 创新点**

发现节点与全局信息在视觉编码器中即可线性解码，而边关系只能在文本层线性解码，解释了 LVLM 在边关系识别上的性能瓶颈。

**🔧 技术方法**

采用线性探针、因果干预方法，并结合 Vision Transformer 与大语言模型（Qwen3‑VL‑8B 等）架构进行实验。

**📊 数据集**

构造了可控合成图表数据集（5 节点、8 种颜色、5 种形状等），用于精细化分析。

**📈 对比分析**

在 Qwen3‑VL‑8B 的 VQA 基准中，节点、全局信息的准确率远高于随机水平；边方向准确率仅接近随机；探针与干预结果验证了信息编码与推理的因果关联。

**⚠️ 局限性**

仅基于合成图表，未涵盖复杂真实图表场景；对复合视觉模式和更高阶关系的表示仍未深入探究。

---

## 464. HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain Better Provable Generalization

**arXiv ID:** 2603.02649 | [PDF](https://arxiv.org/pdf/2603.02649v1)

**作者:** Feihu Huang `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13709 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新研究并改进 Adam 与 AdamW 优化器的泛化性能

**💡 创新点**

提出了无平方根的 Adam(W)-srf 算法，并设计了可在必要时切换到 SGD 的 HomeAdam(W) 算法，理论上将泛化误差从 O(1/√N) 降至 O(1/N)；同时证明了 HomeAdam(W) 在收敛速度上达到 O(1/T^{1/4})，优于 Adam(W)-srf 的 O(ρ̆^{-1}/T^{1/4})

**🔧 技术方法**

利用算法稳定性分析、渐进逼近法与 L‑smooth、Lipschitz 连续等假设，证明泛化误差与收敛率；实现了基于权重衰减的 HomeAdamW；在实验中使用标准的 Adam/AdamW 对照

**📊 数据集**

CIFAR‑10、Tiny‑ImageNet（图像分类）和 WikiText‑2、WikiText‑103（语言建模）

**📈 对比分析**

与 SGD、SGDM、Adam、AdamW、SWATS、AdaBelief、MIAdam 等方法对比；实验表明 Adam(W)-srf 与 HomeAdam(W) 在训练/测试损失、准确率/困惑度方面均优于对照组，尤其是 HomeAdamW 在测试准确率/困惑度上表现最佳

**⚠️ 局限性**

对比分析主要基于理论假设，实际效果仍受超参数、模型结构及数据复杂度影响；目前只在公开基准数据上验证，缺乏对更大规模或更复杂任务的推广性研究

---

## 465. OmniFashion: Towards Generalist Fashion Intelligence via Multi-Task Vision-Language Learning

**arXiv ID:** 2603.02658 | [PDF](https://arxiv.org/pdf/2603.02658v1)

**作者:** Zhengwei Yang `[一作]` (Wuhan University), Zheng Wang `[通讯]` (Wuhan University)

**通讯引用:** 462014 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了百万规模的 FashionX 数据集，并在此基础上训练了统一的 OmniFashion 视觉语言模型，实现了多任务时尚智能的端到端解决方案。

**💡 创新点**

创新点在于：① 用 GPT‑4.1 自动生成层级化 JSON 标注，实现头至脚完整且一致的时尚注释；② 将所有时尚任务统一转化为对话式问答范式，配合两阶段训练实现跨任务一致性与可迁移性。

**🔧 技术方法**

采用了 GPT‑4.1 驱动的自动注释管线、Qwen2.5‑VL 视觉语言骨干、LoRA 微调、对话式多任务训练以及 Bradley–Terry 对比检索策略。

**📊 数据集**

使用的数据集为 FashionX（约 102 万套装、600 万+ 属性）以及公开的 DeepFashion、Fashionpedia 等时尚基准数据。

**📈 对比分析**

在多任务对话子任务、检索 R@1/R@10/R@20/mAP 等指标上，OmniFashion 以 3B 参数超过所有开源 VLM，在 InShop、Consumer‑2‑Shop 检索任务上取得最高 R@1 与 mAP，整体平均精度提升约 36%。

**⚠️ 局限性**

limitations：模型的标签质量受 GPT‑4.1 风格偏好影响，长文本对话理解能力有限，缺乏多文化、多样性场景的真实用户交互验证。

---

## 466. ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs

**arXiv ID:** 2603.02676 | [PDF](https://arxiv.org/pdf/2603.02676v1)

**作者:** Wicaksono Leksono Muhamad `[一作]` (SEACrowd), Samuel Cahyawijaya `[通讯]` (Cohere)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通过将演绎推理（演绎三段论）转换为规范化的符号表示，再利用确定性解析规则判断有效性的方法，以减少语言模型在多语言推理中的内容偏差。

**💡 创新点**

创新点在于：①使用结构抽象将自然语言推理问题映射到标准化的符号三段论；②采用基于语法规则的确定性解析来判定有效性；③结合英语枢轴翻译策略，使方法在多语言环境中同样有效。

**🔧 技术方法**

技术包括：自然语言到三段论的符号化转换（使用 Gemini‑3 进行规范化），英语枢轴翻译、正则表达式匹配四类命题类型，基于情绪与图形的规则检验逻辑有效性，以及对相关前提的结构化识别。

**📊 数据集**

使用 SemEval‑2026 Task 11 的多语言演绎三段论基准数据集，包含英语与多种语言的子任务。

**📈 对比分析**

与仅使用 LLM 直接推理（LLM‑only）以及仅做规范化 + 解析的基线相比，所提方法在所有子任务中均进入前五名，逻辑有效性准确率接近或达到 100%，相关前提识别 F1 也大幅提升，同时显著降低内容偏差（bias）评分。

**⚠️ 局限性**

局限性：仅评估了单一商用模型 Gemini‑3‑Flash；未对多种模型、随机种子或采样策略进行实验；缺乏对模型性能波动的分析；且方法完全基于确定性解码，未探讨多样性和鲁棒性。

---

## 467. It's Alive! What a Live Object Environment Changes in Software Engineering Practice

**arXiv ID:** 2603.02987 | [PDF](https://arxiv.org/pdf/2603.02987v1)

**作者:** Julián Grigera `[一作]` (National University of La Plata), Stéphane Ducasse `[通讯]` (Inria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文通过三个案例（调试器驱动开发、IDE扩展、系统演进）阐述了 Pharo 作为 live 编程环境的工具特性及其对 IDE 设计的启示。

**💡 创新点**

创新点在于将调试器、检查器、重构等工具与运行时环境无缝集成，形成持续交互的开发循环，并提供可扩展的自定义视图和自动重写机制。

**🔧 技术方法**

使用了 Pharo Smalltalk 语言、Live IDE、调试器 API、检查器插件、自动重写规则等技术。

**📊 数据集**

未使用外部数据集，仅以物流系统示例进行说明。

**📈 对比分析**

文章未给出实验对比或性能指标，而是通过案例演示说明其可行性和优势。

**⚠️ 局限性**

局限性包括对规模化团队协作和大型系统状态管理的挑战，以及需要从传统文件‑基 IDE 迁移时的学习曲线。

---

## 468. Understanding the Resource Cost of Fully Homomorphic Encryption in Quantum Federated Learning

**arXiv ID:** 2603.02799 | [PDF](https://arxiv.org/pdf/2603.02799v1)

**作者:** Lukas Böhm `[一作]` (Leipzig University), Erik Buchmann `[通讯]` (Leipzig University)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5088059746)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

实现了基于 CKKS 同态加密的量子联邦学习框架，对 QCNN、CNN、ResNet‑18 等模型在脑肿瘤 MRI 数据上进行训练并评估加密开销。

**💡 创新点**

首次在 QFL 中完整实现 QCNN 训练并将所有参数使用 CKKS 加密，同时系统性评估 FHE 对内存、通信、时间及性能的综合影响。

**🔧 技术方法**

采用 TenSEAL 的 CKKS 加密、Flower FL 框架、Pennylane 量子仿真以及经典卷积网络与量子层混合模型。

**📊 数据集**

使用公开的脑肿瘤 MRI 数据集（7,023 张图像，分为四类）。

**📈 对比分析**

对比相同模型在未加密与加密两种训练方式，记录训练/聚合时间、CPU/RAM 使用、通信量、准确率、F1 分数；结果表明加密显著提升时间、内存与通信量，准确率受影响较小，但模型复杂度下降会导致性能下降。

**⚠️ 局限性**

局限性包括：仅在 CPU 上模拟量子计算，未优化 CKKS 参数；模型参数受限导致准确率不高；FHE 加密所有层导致通信量巨大，未尝试层级选择或聚类等减压方法；跨设备场景下硬件资源不足。

---

## 469. MEBM-Speech: Multi-scale Enhanced BrainMagic for Robust MEG Speech Detection

**arXiv ID:** 2603.02255 | [PDF](https://arxiv.org/pdf/2603.02255v1)

**作者:** Li Songyi `[一作]` (Peking University), Zhang Zifeng `[通讯]` (Peking University)

**通讯引用:** 23252 | [OpenAlex ID](https://openalex.org/A5100438488)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现MEBM‑Speech模型，对MEG信号进行连续概率解码以检测语音与静默状态。

**💡 创新点**

通过融合多尺度卷积、双向LSTM、深度可分离卷积与空间注意力模块，实现短期细粒度与长期上下文的多尺度融合；引入时间抖动与平均池化提升边界鲁棒性；仅使用grad通道实现轻量化设计。

**🔧 技术方法**

采用BrainMagic骨干网络、空间注意力模块、多尺度卷积块、BiLSTM、深度可分离卷积、平均池化、AdamW优化器和MSE损失进行训练。

**📊 数据集**

使用LibriBrain 2025 Track 1的语音对静默检测数据集，训练验证集来源于Sherlock1会话11–12。

**📈 对比分析**

与官方基线和多种消融变体对比，MEBM‑Speech在验证集上平均F1宏约89.3%，在官方测试榜单上亦能保持≈89%F1macro，展示出优异的泛化能力。

**⚠️ 局限性**

局限于听书场景，尚未验证实时解码与跨受试者适应性；对语音产生或说话产出场景的适用性仍需进一步研究。

---

## 470. Harmonic Beltrami Signature Network: a Shape Prior Module in Deep Learning Framework

**arXiv ID:** 2603.02907 | [PDF](https://arxiv.org/pdf/2603.02907v1)

**作者:** Chenran Lin `[一作]` (Chinese University of Hong Kong), Lok Ming Lui `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2102 | [OpenAlex ID](https://openalex.org/A5046845149)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Harmonic Beltrami Signature Network (HBSN)，通过深度网络从二值图像中直接学习并预测 HBS，形成可插拔的形状先验模块；

**💡 创新点**

创新点在于将 HBS 这一数学上可唯一对应、旋转/尺度/平移不变的形状描述嵌入神经网络，并设计预/后空间变换网络（pre-STN/post-STN）实现形状规范化与角度正则化；

**🔧 技术方法**

主要使用 UNet 结构作为骨干网络，配合空间变换网络（STN）、自定义 L2 损失和角度正则损失，以及数据增强与软化二值输入；

**📊 数据集**

训练数据来自 20k 仅含单连通形状的合成集，分别通过“conformal welding”与“random polygon”两种方法生成，并在 COCO 训练集上评估；

**📈 对比分析**

在 HBS 预测任务上，最佳模型仅需 2 ms 推理，误差 L_HBS<0.007，远快于传统算法（≈871 ms），并在 COCO 骨干网络（UNet、DeepLabV3）上加入 HBSN 后 Dice/IoU 均提升约 1–2%；

**⚠️ 局限性**

局限在于仅支持单连通形状，对多连通或断裂区域的预测不准确，且对形状尺寸范围、噪声敏感，未来需扩展至多目标及动态场景。

---

## 471. Seeing Clearly without Training: Mitigating Hallucinations in Multimodal LLMs for Remote Sensing

**arXiv ID:** 2603.02754 | [PDF](https://arxiv.org/pdf/2603.02754v1)

**作者:** Yi Liu `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 30183 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文针对遥感视觉问答（RS‑VQA）中的幻觉问题，提出了RSHBench基准和RADAR无训练推理框架；

**💡 创新点**

创新点在于将幻觉细粒度诊断与协议驱动基准相结合，并提出基于查询条件相对注意的多阶段进化证据获取方法；

**🔧 技术方法**

使用的技术包括查询条件相对注意（QCRA）、多阶段进化证据获取、关注度聚焦测试等；

**📊 数据集**

采用的数据集为从LRS‑VQA、MME‑RealWorld‑RS、UCM、LHRS‑Bench等四大遥感VQA基准采样的371张图像‑问题对；

**📈 对比分析**

实验结果显示，在RSHBench、LRS‑VQA、MME‑RealWorld‑RS和LHRS‑Bench上，RADAR在多种MLLM上平均提升约2–4%准确率，幻觉率下降约10%；

**⚠️ 局限性**

限制包括对内部注意力信息的依赖、预定义查询模板、推理成本增加以及对黑箱模型的适用性有限。

---

## 472. APAO: Adaptive Prefix-Aware Optimization for Generative Recommendation

**arXiv ID:** 2603.02730 | [PDF](https://arxiv.org/pdf/2603.02730v1)

**作者:** Yuanqing Yu `[一作]` (Tsinghua University), Min Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 60431 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

解决生成式推荐中训练与推理不一致问题，提出 Adaptive Prefix‑Aware Optimization (APAO) 框架。

**💡 创新点**

创新点在于：① 在训练中加入 prefix‑级别的优化损失，使模型能在 beam search 过程中保持高概率；② 采用自适应 worst‑prefix 机制，动态聚焦最脆弱的前缀，从而提高候选保留率。

**🔧 技术方法**

使用自回归生成模型（TIGER、Llama）、pointwise 与 pairwise prefix‑loss、负样本采样、KL 限制的自适应权重更新等技术。

**📊 数据集**

使用四个公开电商/点评数据集：Office、Grocery、Beauty、Yelp。

**📈 对比分析**

与 CE、MSL、DPO、DMPO、S‑DPO 等基线在 Recall@K 和 NDCG@K 上进行对比，APAO 在所有数据集和两种背骨上均显著提升性能，并在不同 beam 大小下保持优势。

**⚠️ 局限性**

局限性：需要手动调节 β、η 等超参数；pairwise 版本对负样本数敏感；仅在两种背骨上验证，其他模型与更大规模场景需进一步探索。

---

## 473. RIS-Enabled Wireless Channel Equalization: Adaptive RIS Equalizer and Deep Reinforcement Learning

**arXiv ID:** 2603.02489 | [PDF](https://arxiv.org/pdf/2603.02489v1)

**作者:** Gal Ben-Itzhak `[一作]` (University of California), Ender Ayanoglu `[通讯]` (University of California)

**通讯引用:** 4843 | [OpenAlex ID](https://openalex.org/A5037308417)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并比较了基于RIS的空中等化与放大方案，分别是传统的ARISE梯度下降方法和三种深度强化学习算法（DDPG、TD3、SAC）

**💡 创新点**

创新点在于将RIS视为可实现等化的空中设备，提出ARISE并证明SAC在无完整CSI条件下可达到与ARISE相当的等化性能

**🔧 技术方法**

采用梯度下降（SD）优化RIS反射系数，并使用深度强化学习框架（DDPG、TD3、SAC）进行无模型学习

**📊 数据集**

通过仿真生成Rayleigh/Rician多径信道数据，包含随机行走UE、不同ISIs、RIS尺寸及Rician因子等多种环境配置

**📈 对比分析**

通过SNR、η_n（归一化等化指标）以及收敛速度等指标进行比较，SAC收敛最快、SNR最高，整体性能接近ARISE；相对DDPG/TD3表现更稳健

**⚠️ 局限性**

限制包括ARISE对高维CSI估计高度依赖、DRL训练需要较长时间、实验仅限单UE单天线场景，未验证多用户或MIMO拓展

---

## 474. NeighborMAE: Exploiting Spatial Dependencies between Neighboring Earth Observation Images in Masked Autoencoders Pretraining

**arXiv ID:** 2603.02522 | [PDF](https://arxiv.org/pdf/2603.02522v1)

**作者:** Liang Zeng `[一作]` (KU Leuven), Maarten Vergauwen `[通讯]` (KU Leuven)

**通讯引用:** 3213 | [OpenAlex ID](https://openalex.org/A5006850646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了相邻遥感图像在 Masked Image Modeling（MIM）中的空间依赖，并提出 NeighborMAE 模型实现联合重建。

**💡 创新点**

创新点在于：①利用相邻图像的空间连续性进行联合重建；②采用基于 IoU 的动态遮罩比例；③引入输入可见性加权损失，防止学习快捷方式。

**🔧 技术方法**

技术手段包括：基于 MAE 的 ViT-Large-16 架构；邻居图像采样（IoU 阈值、相对位置嵌入）；动态遮罩比例；可见性加权 MSE 损失。

**📊 数据集**

数据集：fMoW-RGB（含多时相序列）和 Satellogic-RGB（大规模滑动窗口采样），在两者上分别进行预训练。

**📈 对比分析**

与 MAE、SatMAE、SatMAE++、ScaleMAE、CrossScale MAE、DOFA 等基线在图像分类（fMoW、UC Merced、RESISC45 等）和语义分割（Five-Billion-Pixels、PASTIS-HD）任务上进行对比，NeighborMAE 在多项指标上均优于基线，甚至与 DOFA 接近或略胜。

**⚠️ 局限性**

局限性：仅在 RGB 图像上验证，未扩展到多光谱/多模态；多邻居联合编码导致 O(n²) 计算和显存开销；对地理坐标和 IoU 阈值的依赖；在某些任务中加权损失效果有限。

---

## 475. Agentic AI-based Coverage Closure for Formal Verification

**arXiv ID:** 2603.03147 | [PDF](https://arxiv.org/pdf/2603.03147v1)

**作者:** Sivaram Pothireddypalli `[一作]` (Infineon Technologies India Private Limited), Aman Kumar `[通讯]` (Infineon Technologies India Private Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并实现了一套基于代理 AI 的工作流，利用 LLM 自动分析正式验证生成的覆盖报告，识别覆盖缺口，生成针对性 SystemVerilog 属性，形成迭代闭环，实现正式验证的覆盖闭合；并在原 Saarthi 多代理框架上加入 Coverage Hole Analyzer 与 SVA Property Generator 两个新代理。

**💡 创新点**

创新点在于将覆盖缺口分析与属性生成两阶段通过 LLM 实现自动化；通过 JSON 结构传递精确的 RTL 语境，自动匹配时序与信号，生成符合设计约束的属性；以及在多代理协作中加入 HIL 检测与人机交互，显著提高属性质量与覆盖闭合速度。

**🔧 技术方法**

使用技术包括：多代理 AI（基于 AutoGen 的组播管理）、OpenAI GPT‑4.1、GPT‑5、Meta Llama3.3 等大语言模型、系统Verilog (SVA) 规范、Cadence JasperGold 等正式验证工具，以及对 RTL 进行静态代码提取与分析的自定义脚本。

**📊 数据集**

采用的验证数据集为多种开源与内部 RTL 设计，涵盖不同复杂度：ECC、CIC Decimator、AXI4LITE、Automotive IP、Memory Scheduler 等。

**📈 对比分析**

通过对比原 Saarthi 工作流与加入覆盖代理后的版本，使用两个 KPI：属性证明率（Proven %）和最终覆盖率（Coverage %）。实验显示覆盖率提升 10–20%，最高可达 90% 以上；不同 LLM 模型表现差异，GPT‑5 提供最高覆盖但延迟更大。

**⚠️ 局限性**

局限性包括：生成属性后证明率有时下降，因属性难以证明或过于约束；Coverage Hole Analyzer 在代码提取与语义解析上偶尔产生误差，导致属性重复或错误；当前实验未加入 HIL 监督，仍需人工干预以确保属性合法性。

---

## 476. GoldbachGPU: An Open Source GPU-Accelerated Framework for Verification of Goldbach's Conjecture

**arXiv ID:** 2603.02621 | [PDF](https://arxiv.org/pdf/2603.02621v1)

**作者:** Isaac Llorente-Saguer `[一作]` `[通讯]` (Independent Researcher), Isaac Llorente-Saguer (Independent Researcher)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了名为GoldbachGPU的开源框架，利用单块RTX 3070等消费级GPU完成了对Goldbach猜想至10^12的完整数值验证，并支持多GPU与任意精度检查。

**💡 创新点**

突破传统GPU方法的VRAM上限，采用稠密位图压缩与分段双筛结构，使VRAM占用保持恒定；同时结合GPU快速路径与CPU后备、以及同步批处理的GMP检查器。

**🔧 技术方法**

使用CUDA GPU内核、稠密位图表示、分段双筛、Miller–Rabin质数判定、CPU后备回退、OpenMP并行、GMP任意精度、OpenMP同步批处理等技术。

**📊 数据集**

对所有偶数从4到10^12进行验证；单数任意精度检查使用10^50至10^10000范围内的十进制字符串；硬件实验基于Intel i7‑12700H、NVIDIA RTX 3070、RTX 3090、H100。

**📈 对比分析**

与CPU基线和全局位图GPU方案对比，GoldbachGPU在10^9级别实现16倍加速，10^10级别12倍加速；到10^12仅需2小时完成；多GPU线性缩放，RTX 3090 4GPU保持91%效率；GMP检查在10^10000耗时182秒。

**⚠️ 局限性**

主要瓶颈仍是CPU主机端的分段筛构造，限制了GPU计算利用率；segment大小与CPU缓存/内存带宽匹配敏感；GMP质数判定的O(d^3)复杂度使极大数字的彻底验证不可行；高性能H100仅在更大N才显优势。

---

## 477. Reducing Labeling Effort in Architecture Technical Debt Detection through Active Learning and Explainable AI

**arXiv ID:** 2603.02944 | [PDF](https://arxiv.org/pdf/2603.02944v1)

**作者:** Edi Sutoyo `[一作]` (University of Groningen), Andrea Capiluppi `[通讯]` (University of Groningen)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5077760743)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过专家验证的关键词过滤、主动学习与可解释 AI 的组合，构建了一个自动化的架构技术债务（ATD）检测流程，显著降低了人工标注成本。

**💡 创新点**

①利用专家标注的 ATD 语料提取高质量关键词，构建大规模候选集；②首次在 ATD 检测中系统评估并应用 Breaking Ties 等主动学习查询策略，提升样本利用率；③在 ATD 分类中首次对比 LIME 与 SHAP 的可解释性，并通过专家评估揭示两者的优势与局限。

**🔧 技术方法**

关键词提取（TF‑IDF、KeyBERT、CS‑KeyBERT）；BERT 及其主动学习实现（Prediction Entropy、Least Confidence、Breaking Ties、Embedding K‑Means、Contrastive AL、Random）；可解释 AI（LIME、SHAP）；传统机器学习模型（SVM、NB、LR、RF、Text CNN、Text GCN 等）。

**📊 数据集**

10 个 Apache Java 开源项目的 Jira 问题库。首先生成约 103,000 条候选问题，随后专家共标注 1,100 条 ATD（True‑ATD 57、Weak‑ATD 1,043），构成实验数据集。

**📈 对比分析**

通过 F1‑score、精度和召回率进行比较。主动学习 Breaking Ties 在仅使用 51% 标注样本时实现 0.72 的 F1‑score，明显优于随机采样 0.64；BERT‑Breaking Ties 在完整数据训练下 F1‑score 0.66；传统 ML 模型 F1‑score 0.60‑0.70。关键词过滤在非 ATD 检测中的准确率 83‑85%，但仅捕获 21‑33% 的 True‑ATD。

**⚠️ 局限性**

关键词方法召回率低，True‑/Weak‑ATD 合并导致概念模糊；实验仅基于 Java 开源项目的 Jira 数据，缺乏工业环境和其他语言的验证；XAI 评估样本有限且专家群体偏向学术，可能影响结果的普适性。

---

## 478. ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling

**arXiv ID:** 2603.02697 | [PDF](https://arxiv.org/pdf/2603.02697v1)

**作者:** Jiayi Zhu `[一作]` (Shanghai Jiao Tong University), Xiaoyun Yuan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5062916840)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

训练视频生成模型实现多代理共享世界建模

**💡 创新点**

创新点：四视角拼接实现单代理内部多视角几何一致，跨代理注意力模块实现多代理共享空间-时间信息，基CARLA构建的大规模多代理交互视频数据集

**🔧 技术方法**

使用CogVideoX大规模视频扩散模型，配合raymap编码、交叉注意力和VAE+DiT架构

**📊 数据集**

使用CARLA仿真平台生成的55k对8视角视频数据集（共计约250帧/序列）

**📈 对比分析**

与单视角、无跨代理对比实验，PSNR/SSIM/LPIPS等指标显著提升，VBench Aesthetic/Temporal/Motion等得分优于基线

**⚠️ 局限性**

局限：PSNR仍不高，生成多样性导致与真实帧差异；缺乏物理交互建模，未支持更复杂的多代理动态交互

---

## 479. MMH-Planner: Multi-Mode Hybrid Trajectory Planning Method for UAV Efficient Flight Based on Real-Time Spatial Awareness

**arXiv ID:** 2603.02683 | [PDF](https://arxiv.org/pdf/2603.02683v1)

**作者:** Yinghao Zhao `[一作]` (Information Engineering University), Hong Xie `[通讯]` (Wuhan University)

**通讯引用:** 4367 | [OpenAlex ID](https://openalex.org/A5037556478)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种基于实时空间感知的多模式混合轨迹规划与惰性重规划方法，用于无人机在未知复杂环境中的高效安全飞行。

**💡 创新点**

创新点在于：①目标导向的空间感知算法能够快速评估即将到来的障碍影响；②三模式混合规划（快速、标准、紧急）根据感知结果自适应选择最优模型；③惰性重规划策略只在必要时触发，显著降低计算负荷。

**🔧 技术方法**

使用了B-spline、MinCO轨迹表示、软硬约束优化、A*+SFC构建、EGO-Planner式梯度优化以及自定义的窄通道检测与权重调整技术。

**📊 数据集**

实验数据集包括：①在50×50×3 m³体积内随机生成的四种障碍密度（0、0.05、0.1、0.2 obs./m²）模拟环境；②真实户外实验中使用自研四旋翼配备MID360激光雷达的环境感知数据。

**📈 对比分析**

对比方法为FastPlanner、EGO-Planner、ROTP，在每种障碍密度下执行5次试验，评估飞行时间、距离、能耗、规划迭代次数和每次计算成本。结果显示MMH在所有指标上均优于或排名第二，尤其在平均规划次数和每次成本上显著低于其它方法。

**⚠️ 局限性**

局限性包括：①对传感器感知范围敏感，未知区域内安全性无法完全保证；②在极高障碍密度或快速动态变化环境下可能仍需多次重规划；③对高速或动态障碍的鲁棒性未在实验中充分验证。

---

## 480. Learning Therapist Policy from Therapist-Exoskeleton-Patient Interaction

**arXiv ID:** 2603.02458 | [PDF](https://arxiv.org/pdf/2603.02458v1)

**作者:** Grayson Snyder `[一作]` (Northwestern University), Jose Pons `[通讯]` (Shirley Ryan AbilityLab)

**通讯引用:** 12181 | [OpenAlex ID](https://openalex.org/A5004065024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了患者-治疗师力场（PTFF）与合成治疗师（ST）两种方法，用于辅助中风后下肢步态康复。

**💡 创新点**

创新点在于利用低维潜在空间与高斯混合模型可视化治疗师的力学策略，并用LSTM实时预测治疗师动作，实现在外骨骼控制中的机器人辅助。

**🔧 技术方法**

采用了变分自编码器（VAE）进行步态维度压缩，Gaussian Mixture Model（GMM）构建力场映射，LSTM捕捉时序依赖，结合ROS实现实时推理。

**📊 数据集**

使用之前收集的8名慢性偏瘫患者与单一物理治疗师在双外骨骼交互实验中获得的步态与力学数据集。

**📈 对比分析**

通过留一交叉验证评估ST的预测误差（位置≈4.3°，速度≈20.7°/s），PTFF的VAE重构误差约为4.5°，推断时间低于3 ms，显示模型可在333 Hz实时实现。

**⚠️ 局限性**

局限性包括样本量有限、仅单一治疗师参与、数据噪声影响、以及对连接参数（刚度、阻尼）的依赖，泛化性能仍需在更大、更多样化人群中验证。

---

## 481. CASSR: Continuous A-Star Search through Reachability for real time footstep planning

**arXiv ID:** 2603.02989 | [PDF](https://arxiv.org/pdf/2603.02989v1)

**作者:** Jiayi Wang `[一作]` (State Key Laboratory of General Artificial Intelligence), Steve Tonneau `[通讯]` (University of Edinburgh)

**通讯引用:** 906 | [OpenAlex ID](https://openalex.org/A5043622443)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种基于连续可达性约束的连续A*搜索框架，用于实时规划双足机器人的足部步态序列。

**💡 创新点**

创新点在于递归传播凸连续可达性约束到A*搜索，并结合EPA算法的代价估计与安全成本，显著降低节点扩展量并支持步态旋转。

**🔧 技术方法**

使用技术包括多面体凸可达性建模、EPA距离算法、A*搜索、二次规划求解（QP）以及离散化旋转角度。

**📊 数据集**

实验数据集基于Talos双足机器人，在三种场景（stairs、local minima、narrow passage）上进行评测。

**📈 对比分析**

通过与离散化A*和商业MIP求解器比较，实验表明在最复杂情形下该方法比离散化A*快100倍，比MIP快20倍，并能在125 ms内规划30步。

**⚠️ 局限性**

局限性包括只能在确定的步态序列后优化位置，无法同时优化其他成本函数；假设同一条腿不会再次踏回先前的表面；未考虑碰撞与动力学约束，需进一步扩展。

---

## 482. Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting

**arXiv ID:** 2603.02893 | [PDF](https://arxiv.org/pdf/2603.02893v1)

**作者:** Kaiqiang Xiong `[一作]` (Peking University), Ronggang Wang `[通讯]` (Peking University)

**通讯引用:** 3633 | [OpenAlex ID](https://openalex.org/A5050071143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ICO-GS框架，利用几何正则化与基于几何引导的外观优化实现稀疏视角下3D高斯渲染的几何-外观一致性。

**💡 创新点**

创新点在于：①采用像素级top‑k特征一致性与边缘感知深度平滑实现鲁棒几何约束；②通过循环一致性深度滤波生成可靠深度，再合成虚拟视角实现外观的几何引导；③三阶段课程学习策略使几何与外观协同收敛。

**🔧 技术方法**

使用特征匹配的多视角一致性损失、像素top‑k筛选、边缘平滑正则、循环一致性深度滤波、虚拟视角光度一致性损失，以及BinocularGS基础框架。

**📊 数据集**

在LLFF、DTU和Blender三大公开数据集上进行实验，采用3/6/9视角或8视角设置。

**📈 对比分析**

与多种NeRF和3DGS稀疏视角基线（如DietNeRF、FreeNeRF、BinocularGS等）对比，PSNR平均提升约0.8 dB（LLFF 3视角）、1.1 dB（DTU 3视角），在所有视角设置下均达标杆或超越。

**⚠️ 局限性**

限制包括：假设外观视角无关，导致在强视角依赖的高光/反射区域可能出现误导；训练时间约为基线的1.5倍；在极低视角或极度纹理稀疏场景下仍可能产生浮点或模糊。

---

## 483. Structure-Aware Text Recognition for Ancient Greek Critical Editions

**arXiv ID:** 2603.02803 | [PDF](https://arxiv.org/pdf/2603.02803v1)

**作者:** Nicolas Angleraud `[一作]` (Inria), Thibault Clérice `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了结构感知文本识别在古希腊批判性版本中的应用，构建了大规模合成语料和真实扫描基准，并系统评估多种视觉‑语言模型（VLM）与传统 OCR 的表现。

**💡 创新点**

①提供185k页合成图像数据集，精细控制排版与字体变异；②提供约450页真实扫描基准；③引入轻量标记方案（伪 XML / Markdown）实现结构化转写；④系统评估 VLM 在零样本与微调下的结构识别性能。

**🔧 技术方法**

采用视觉‑语言模型 Qwen3‑VL、DeepSeek‑OCR‑2、LightOnOCR‑2，并通过 LoRA 高效微调；对比 CRNN 基线（Tesseract、Kraken）；实现自定义排版渲染流水线与结构化标注。

**📊 数据集**

数据集包括：①基于 TEI/XML 的合成页图像（185k 页）；②真实扫描批判性版本（约450 页，30 作家/作品对）。

**📈 对比分析**

通过对比零样本、单一微调、真实微调及合成→真实微调四种训练方式，评估字符/词错误率、结构标记 F1 等指标。结果显示 Qwen3‑VL‑8B 在合成+真实微调后在真实扫描上中位 CER 1.0%、W ER 5.6%，结构识别 F1 最高可达 80%，显著优于零样本模型且与传统 OCR 基线相比具有更好的跨域适应性。

**⚠️ 局限性**

局限性包括：①结构标记识别仍存在漏检与错误；②大型 VLM 计算成本高、能耗大；③在符号混杂、脚注密集等复杂布局下仍易出现灾难性错误；④合成数据中边缘符号稀缺导致微调效果不均衡。

---

## 484. S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation

**arXiv ID:** 2603.02725 | [PDF](https://arxiv.org/pdf/2603.02725v1)

**作者:** Xiaodong Li `[一作]` (Institute of Information Engineering), Tingwen Liu `[通讯]` (Institute of Information Engineering)

**通讯引用:** 1495 | [OpenAlex ID](https://openalex.org/A5103214505)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于平滑-锐化过程的无训练跨域冷启动推荐模型 S^2CDR，利用图信号处理与 ODE 求解器实现对跨域交互矩阵的腐蚀-恢复。

**💡 创新点**

创新点在于：①用热方程和自定义低通滤波在平滑阶段捕获跨域项关联并去除高频噪声；②在锐化阶段通过负项强化差异实现个性化恢复；③整个框架为无参数、无训练的腐蚀-恢复结构，使用连续 ODE 求解。

**🔧 技术方法**

采用图信号处理（热方程、理想低通滤波）、常微分方程求解器（Euler、RK4、DOPRI）以及矩阵算子实现，完全不使用神经网络或参数训练。

**📊 数据集**

实验数据集为 Douban 的 Movie–Book 以及 Amazon 的 Movie–Music、Book–Music 三个真实跨域场景，数据经过二值化和过滤后构成交互矩阵。

**📈 对比分析**

在 HR@10 与 NDCG@10 上与 14 种 SOTA 方法（映射式、元学习式、DMCDR、GF‑CF、PGSP 等）进行留一评估，S^2CDR 在三场景平均提升约 10–12%（HR）/8–11%（NDCG），并在所有基准中排名第一。

**⚠️ 局限性**

局限性：仅适用于双域冷启动场景，未探讨多域或带属性的扩展；依赖完整交互矩阵构造，矩阵规模大时仍可能受限于内存与计算速度。

---

## 485. Leveraging Label Proportion Prior for Class-Imbalanced Semi-Supervised Learning

**arXiv ID:** 2603.02957 | [PDF](https://arxiv.org/pdf/2603.02957v1)

**作者:** Kohki Akiba `[一作]` (Kyushu University), Ryoma Bise `[通讯]` (Kyushu University)

**通讯引用:** 1451 | [OpenAlex ID](https://openalex.org/A5064312777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出将Proportion Loss作为正则化项引入半监督学习，用以缓解伪标签化过程中的类别不平衡问题。

**💡 创新点**

创新点在于首次将学习自标签比例（LLP）的Proportion Loss迁移到SSL，并设计了基于多元超几何分布的随机扰动版本，以提升在极端不平衡情况下的鲁棒性。

**🔧 技术方法**

采用的技术包括伪标签一致性正则化（FixMatch/ ReMixMatch）、Proportion Loss正则化、超几何扰动采样以及SGD+cosine学习率调度。

**📊 数据集**

使用的数据集为长尾版CIFAR-10（CIFAR-10-LT），在多种imbalance ratio和label ratio设置下进行实验。

**📈 对比分析**

与FixMatch、ReMixMatch、DARP、CReST等方法比较，实验表明该方法在所有设置下均超越基线，尤其在标签稀缺（β=2%/4%）时表现尤为突出。

**⚠️ 局限性**

局限性包括：对训练数据与标签分布差异敏感；当无标签批量大小过小时，比例估计不准确，导致正则化效果下降。

---

## 486. IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models

**arXiv ID:** 2603.02669 | [PDF](https://arxiv.org/pdf/2603.02669v1)

**作者:** Xiangyu Su `[一作]` (Shenzhen University), Ruizhen Hu `[通讯]` (Shenzhen University)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5013892799)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的工业多机器人任务规划与程序生成框架IMR-LLM，能够从自然语言任务描述生成可执行的高层计划与低层控制程序。

**💡 创新点**

创新点：①将LLM与离散图（disjunctive graph）相结合，利用LLM构建任务依赖并使用确定性求解器实现高效排程；②引入操作过程树（process tree）替代少量示例提示，显著提升程序可执行性与可扩展性。

**🔧 技术方法**

技术手段：LLM（如OpenAI GPT）、链式推理（CoT）、disjunctive graph构建与FIFO求解、过程树推理、Python代码生成、机器人执行接口。

**📊 数据集**

使用自构建的IMR-Bench数据集：23个真实工业场景，50个多机器人任务（单机器人、简单多机器人、复杂多机器人），包含任务、场景JSON和人工标注的黄金结果。

**📈 对比分析**

对比方法：SMART-LLM、LaMMA-P、LiP-LLM（及其变体）。在5个评估指标（OC、SE、Exe、GCR、SR）上，IMR-LLM在所有任务级别均取得最高分，尤其在复杂多机器人任务上显著提升成功率；但在极端复杂场景下仍略有下降。

**⚠️ 局限性**

局限性：LLM在任务拆分与机器人分配上仍可能产生错误或非最优结果，导致OC下降；随任务复杂度提升，输入长度增大，LLM更易出现幻觉与逻辑不一致，影响Exe和GCR；目前未实现闭环执行反馈，限制了对动态环境的自适应能力。

---

## 487. Beyond Anatomy: Explainable ASD Classification from rs-fMRI via Functional Parcellation and Graph Attention Networks

**arXiv ID:** 2603.02518 | [PDF](https://arxiv.org/pdf/2603.02518v1)

**作者:** Syeda Hareem Madani `[一作]` (Salim Habib University), Rizwan Qureshi `[通讯]` (Salim Habib University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文构建了一套基于图神经网络的 ASD 分类框架，结合功能分区和可解释分析；

**💡 创新点**

核心创新在于系统比较了解剖学分区（AAL）与功能学分区（MSDL）的影响，并用 GAT 集成及 GNNExplainer 实现高可解释性；

**🔧 技术方法**

使用了图卷积网络、图注意网络（GAT）、DropEdge、Gaussian 噪声数据增强以及 GNNExplainer 等技术；

**📊 数据集**

数据来源为 ABIDE I 数据集，包含 400 名受试者（200 ASD + 200 TD）并在 17 个采集站点均衡划分；

**📈 对比分析**

通过站点分层 70/15/15 划分及训练集噪声增强，对比 AAL 与 MSDL 的 GCN 以及最终 GAT 集成模型，最高准确率达 95.0%，AUC 约 0.98，显著优于现有 GNN 方案；

**⚠️ 局限性**

限制包括仍需在更大多样化样本上验证，模型对不同站点/扫描参数的鲁棒性未完全评估，且仅聚焦 rs‑fMRI 数据，未结合多模态信息。

---

## 488. A Unified Revisit of Temperature in Classification-Based Knowledge Distillation

**arXiv ID:** 2603.02430 | [PDF](https://arxiv.org/pdf/2603.02430v1)

**作者:** Logan Frank `[一作]` (Ohio State University), Jim Davis `[通讯]` (Ohio State University)

**通讯引用:** 2652 | [OpenAlex ID](https://openalex.org/A5012185466)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统研究了知识蒸馏中温度参数的选择与训练配置、教师来源、学生初始化以及数据集细粒度的相互影响，提出了统一的温度选取建议。

**💡 创新点**

创新点在于将温度作为核心变量，全面考察其与优化器、批大小、训练周期、教师预训练/微调、学生初始化以及数据集粒度之间的交叉作用，并揭示了温度在不同场景下的非直观趋势。

**🔧 技术方法**

采用标准的输出匹配知识蒸馏（KL 散度），结合 AdamW/SGD 优化器、不同批大小、训练周期、MixUp/CutMix 数据增强，以及多种教师/学生网络（ResNet, ViT, MobileNet, ConvNeXt 等）。

**📊 数据集**

使用了 Pets、CIFAR‑100、Cars、Tiny‑ImageNet 四个公开图像分类数据集，并在实验中扩展到 ImageNet‑Birds 与 Finer‑Grained‑Birds 两个自构数据集。

**📈 对比分析**

对比实验显示：AdamW 对温度更鲁棒；在 SGD 下短期训练温度越低越好，长周期后高温度（≥10）更优；教师在轻度微调且与预训练数据集类重叠时，高温度能显著提升学生准确率，且细粒度数据集偏好更高温度。

**⚠️ 局限性**

局限性包括：仅考虑了输出匹配形式的蒸馏；实验规模相对有限，未覆盖所有可能的网络架构和超参数组合；对大规模模型和更复杂任务（如目标检测、分割）的适用性仍待验证。

---

## 489. Cultural Counterfactuals: Evaluating Cultural Biases in Large Vision-Language Models with Counterfactual Examples

**arXiv ID:** 2603.02370 | [PDF](https://arxiv.org/pdf/2603.02370v1)

**作者:** Phillip Howard `[一作]` (Thoughtworks), Kathleen C. Fraser `[通讯]` (University of Ottawa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个近60k张图像的文化反事实数据集，并利用该数据集评估大型视觉-语言模型在宗教、国籍和社会经济层面的文化偏见。

**💡 创新点**

创新点在于首次使用基于真实背景图与合成人物的反事实图像集合，精准隔离文化上下文对模型输出的影响，并提出多维度评估框架（分类、敏感度、数值偏差、毒性与主题分析）。

**🔧 技术方法**

技术包括 FLUX.1-Kontext 图像编辑模型生成反事实图像、CLIP 进行图像相似度过滤、Qwen2.5-VL-32B-Instruct 用于上下文识别与拒绝率判断，以及 GPT‑5‑nano 作为判别者。

**📊 数据集**

使用的数据集为 59.8k 张图像组成的 Cultural Counterfactuals（可在 Hugging Face 上获取），每个反事实集合展示同一人物在不同文化背景下的多张图像，并附带种族、性别、年龄及文化标签。

**📈 对比分析**

对 9.07M 条生成文本进行评估，发现不同模型在薪资、租金、毒性等指标上表现出显著的文化差异；例如 Qwen2.5-VL 在宗教与国籍上偏差高、Molmo‑7B 在毒性上最敏感；模型的整体性能因文化上下文而异，未出现一致的无偏表现。

**⚠️ 局限性**

局限包括：假设人物与背景文化相匹配且模型会将文化属性迁移到人物；种族、国籍、宗教与社会经济属性高度相关，难以完全分离；生成图像的文化真实性与人物行为不一定一致；仅覆盖有限的类别与英文评估。

---

## 490. Improving Low-Vision Chart Accessibility via On-Cursor Visual Context

**arXiv ID:** 2603.02498 | [PDF](https://arxiv.org/pdf/2603.02498v1)

**作者:** Yotam Sechayk `[一作]`, Takeo Igarashi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对低视力用户的图表阅读障碍，提出并实现了两种指针中心的视觉上下文交互方法——Dynamic Context（聚焦+上下文）和Mini‑map（概览+细节）并在网页端集成现有辅助技术；

**💡 创新点**

创新点在于将图表的四个关键上下文元素（坐标轴、图例、网格线、整体视图）以可配置、低视觉负荷的方式投影到指针周围，实现了“指针+视觉上下文”的新交互范式；

**🔧 技术方法**

利用React实现的Web原型，基于指针位置实时渲染投影区域（Overlay Area），并通过手工注释提供坐标轴与图例信息；

**📊 数据集**

使用Mini‑VLAT（12类常见图表及其变体）作为评估数据集；

**📈 对比分析**

与传统无辅助基线比较，Dynamic Context在可用性评估（SUS）和主观负荷（NASA‑TLX）上显著优于Mini‑map和基线，且提升了感知访问度和减轻了体力负担；Mini‑map虽增强了空间理解，却因尺寸过小和视觉混乱受到较低偏好；

**⚠️ 局限性**

局限包括：与多种辅助技术（放大、色彩过滤、屏幕阅读器）兼容性不佳；Dynamic Context产生视觉杂乱，尤其对双视或高放大用户不友好；Mini‑map过小难以识别；原型使用手工注释，未实现自动化；样本量有限，存在学习效应和策略多样性等问题。

---

## 491. MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models

**arXiv ID:** 2603.02482 | [PDF](https://arxiv.org/pdf/2603.02482v1)

**作者:** Zhongxi Wang `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 25956 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了MUSE，一个集成跨模态payload生成、多轮攻击、自动安全评估的开源平台，用于多模态LLM的安全红队测试。

**💡 创新点**

创新点在于统一的run-centric架构、双指标细粒度安全评估（硬软ASR）以及Inter-Turn Modality Switching (ITMS)以探究跨模态边界的安全性。

**🔧 技术方法**

采用多轮攻击算法（Crescendo、PAIR、Violent Durian）、自动跨模态payload生成（TTS、图像渲染、视频合成）、LLM判别器和多模态模型路由。

**📊 数据集**

使用AdvBench中的50个危害目标以及多模态转换实现的payload。

**📈 对比分析**

通过与单轮基线对比，实验表明多轮攻击可使90–100%的ASR，而ITMS加速收敛但最终ASR取决于模型族；与多模态模型对比显示不同模型对非文本模态的敏感度差异。

**⚠️ 局限性**

局限包括：仅评估了API访问的模型，缺乏本地部署模型的支持，视频模态延迟高且未完整验证；判别器仍需与人工标注进一步校准。

---

## 492. Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

**arXiv ID:** 2603.02217 | [PDF](https://arxiv.org/pdf/2603.02217v1)

**作者:** Sieun Hyeon `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 991 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对Mixture-of-Experts模型的压缩，提出只对路由器进行轻量级知识蒸馏（Router KD）来校准压缩后模型的路由选择，显著恢复性能。

**💡 创新点**

创新点在于：①将路由器与专家分离，证明专家压缩导致的路由器与专家不匹配是性能下降的主要原因；②提出仅更新路由器参数的蒸馏方法，几乎不增加计算成本；③系统评估了三类压缩范式（Pruning、Editing、Merging）并证明在细粒度MoE上效果尤为突出。

**🔧 技术方法**

技术方法包括：Mixture-of-Experts模型结构分析、路由器与专家分离、使用下一词分布的KL蒸馏（Router KD）以及对比实验的压缩比例和微调策略。

**📊 数据集**

使用的数据集主要为无标签校准数据（C4）以及评估基准集：BBH、CoQA、GSM8k、GSM8k Platinum、MATH、AIME、MBPP、HumanEval-Instruct、MCQA等。

**📈 对比分析**

与现有压缩方法（REAP、CFES、MoBE、TD-MoE、HC-SMoE、M-SMoE）比较，Router KD在所有三种压缩范式中均能恢复或提升大部分指标；在细粒度MoE Qwen3 上平均提升约 5–15% 评分，在粗粒度MoE Mixtral 上提升幅度较小。

**⚠️ 局限性**

局限性包括：①当模型出现灾难性崩溃或压缩后专家表示严重受损时，路由器蒸馏无法恢复；②对粗粒度、少专家的MoE效果有限；③依赖无标签校准数据，若缺乏适当样本可能导致校准不足。

---

## 493. FEAST: Retrieval-Augmented Multi-Hierarchical Food Classification for the FoodEx2 System

**arXiv ID:** 2603.03176 | [PDF](https://arxiv.org/pdf/2603.03176v1)

**作者:** Lorenzo Molfetta `[一作]` (University of Bologna), Gianluca Moro `[通讯]` (University of Bologna)

**通讯引用:** 1543 | [OpenAlex ID](https://openalex.org/A5079648393)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种检索增强的分阶段框架，先识别食物基词，再预测适用的面向属性类别，最后为每个类别分配具体描述符，实现对FoodEx2系统的自动编码。

**💡 创新点**

将检索、重排序与多标签分类相结合，并利用食品分类层次结构和深度度量学习，显著缓解标签稀疏和类别不平衡问题，取得对稀有细粒度标签的性能提升。

**🔧 技术方法**

使用双编码器检索（BGE‑M3、GTE‑Multilingual、ModernBERT）、交叉编码器重排序（DeBERTa‑v3）、多标签分类器（DeBERTa、RoBERTa）以及多任务LLM（LLaMA‑3.1‑8B）进行指令调优。

**📊 数据集**

基于FoodEx2公开矩阵（MTX_12）构建的约28,648条样本的多语言FoodEx2基准数据集。

**📈 对比分析**

与基线CNN网络和多种检索/重排序模型比较，检索器在@10下实现100%召回，重排序后base‑term、facet‑category、descriptor三项Acc@1分别达到约91%、96%和96%；在F1方面，模型在罕见类别上提升12–38%，总体微平均F1在99%以上。

**⚠️ 局限性**

受限于可用数据的覆盖率不足，部分稀有facet类别仍无法获得良好性能；缺乏公开统一评测基准，且模型对任务II（facet‑category预测）的LLM表现不如传统分类器，未来需进一步增强数据增强与持续学习能力。

---

## 494. Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory

**arXiv ID:** 2603.02473 | [PDF](https://arxiv.org/pdf/2603.02473v1)

**作者:** Boqin Yuan `[一作]` (University of California), Kun Yao `[通讯]` (University of North Carolina)

**通讯引用:** 20804 | [OpenAlex ID](https://openalex.org/A5100382929)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对记忆增强LLM代理的写入和检索策略进行诊断性实验，评估不同写入方式与检索方法对性能的影响。

**💡 创新点**

提出了检索到生成边界的诊断探针框架，系统分离写入、检索与利用的贡献，发现检索质量是决定性能的关键。

**🔧 技术方法**

采用三种写入策略（原始块、Mem0提取、MemGPT摘要）与三种检索方法（余弦、BM25、混合重排序），并使用LLM判别器评估检索相关性与失败模式。

**📊 数据集**

实验基于LoCoMo数据集，共1540个非对抗性问题。

**📈 对比分析**

通过9种配置（3写入×3检索）比较准确率，检索方法差异达20个百分点，写入策略仅差3–8个百分点；混合重排序在所有写入方式下平均准确率为77.2%，显著优于单纯余弦或BM25。

**⚠️ 局限性**

实验仅使用单一GPT‑5‑mini模型、单一基准、固定检索预算k=5，写入策略为prompt‑based，未覆盖强化学习或更紧凑上下文场景，LLM判别器的误差也可能影响结果。

---

## 495. No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models

**arXiv ID:** 2603.03203 | [PDF](https://arxiv.org/pdf/2603.03203v1)

**作者:** Omer Sela `[一作]` (Tel Aviv University), Omer Sela `[通讯]` (Tel Aviv University)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5109750874)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究小规模语言模型在不同微调方式下，CDD（基于输出分布峰度的污染检测）能否有效识别训练集污染

**💡 创新点**

揭示CDD仅在模型产生记忆化、输出分布坍塌时才有效，并阐明了记忆阈值与模型容量、适配器秩、训练轮次的交互作用

**🔧 技术方法**

CDD（输出分布峰度）、LoRA低秩微调、完整微调、困惑度、Min-k%概率、n-gram重叠等检测方法

**📊 数据集**

GSM8K、HumanEval、MATH三个不同领域的算术/代码/竞赛题目数据集

**📈 对比分析**

与基线相比，CDD在大多数情况下（尤其是低秩LoRA、单次污染）表现接近随机，概率基方法（困惑度、Min-k%）在所有设置下都显著优于CDD，精确度可达90%以上；仅在高容量完全微调、足够高秩LoRA且污染度高时CDD才接近最佳

**⚠️ 局限性**

仅评估Pythia 70M–410M模型，污染方式为重复示例插入；未考虑预训练阶段污染、其他模型架构或更大数据集，记忆阈值可能随数据规模变化

---

## 496. Hardness of the Binary Covering Radius Problem in Large $\ell_p$ Norms

**arXiv ID:** 2603.03219 | [PDF](https://arxiv.org/pdf/2603.03219v1)

**作者:** Huck Bennett `[一作]` (University of Colorado Boulder), Peter Ly `[通讯]` (University of Colorado Boulder)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了在ℓ_p范数下，γ-逼近决策覆盖半径问题（以及二进制覆盖半径问题）在显式p>35.31时为Π₂-难，并给出了γ(p)>1且lim_{p→∞}γ(p)=9/8的显式函数；

**💡 创新点**

首次给出了有限p下的显式逼近因子γ(p)，并将其推广到二进制覆盖半径与线性失配问题，扩展了Manurangsi等人对ℓ_∞的结果；

**🔧 技术方法**

使用从NAE-3SAT（或(δ,NAE-E3-SAT）到线性失配/覆盖半径的多项式时间约简，并利用3×3矩阵G的低失配性质以及ℓ_p范数的分析；

**📊 数据集**

无实验数据集，纯理论证明；

**📈 对比分析**

通过构造化简证明Π₂-难度，未给出算法性能对比，仅提供逼近因子上的硬度上界；

**⚠️ 局限性**

仅适用于p>35.31，p=2等常见范数的逼近难度仍未完全确定，逼近因子与最优性仍有余地

---

## 497. BRIGHT: A Collaborative Generalist-Specialist Foundation Model for Breast Pathology

**arXiv ID:** 2603.03030 | [PDF](https://arxiv.org/pdf/2603.03030v1)

**作者:** Xiaojing Guo `[一作]` (Tianjin Medical University Cancer Institute and Hospital), Zaiyi Liu `[通讯]` (Guangdong Provincial People's Hospital)

**通讯引用:** 12898 | [OpenAlex ID](https://openalex.org/A5112227828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并训练了BRIGHT，一种融合通用与专科知识的基础模型，用于乳腺病理的诊断、分型、预测和预后分析。

**💡 创新点**

创新点在于双通路协同架构：在大规模多器官基础模型Virchow2上通过LoRA实现乳腺专科微调，然后将通用与专科特征嵌入融合，兼顾广度与深度，显著提升单一器官领域的表现。

**🔧 技术方法**

采用自监督学习（DINOv2）与LoRA参数高效微调、Vision Transformer (ViT-H/14)双通路、CLAM多实例学习聚合器，以及5120维融合特征向量。

**📊 数据集**

使用了约210 M块图像（51,836张H&E切片）来自19家机构的患者数据，以及24个内部/10个外部临床任务的数据集（包含TCGA‑BRCA、BRACS等），覆盖诊断、分子生物学、治疗反应和生存等多维度。

**📈 对比分析**

与三大通用基础模型（Virchow2、UNIv2、CONCHv1.5）以及专科版BRIGHT (S) 对比，BRIGHT在21/24个内部基准任务和5/10个外部任务中获得第一/第二名，诊断AUC可达0.992，分子生物标志物预测AUC平均0.938，治疗反应预测AUC约0.786，预后预测C-index达0.711/0.768。

**⚠️ 局限性**

主要局限包括：研究为回顾性、缺乏前瞻性多中心验证；仅使用H&E切片，未融入多模态（IHC、基因组、报告）信息，模型在极端不平衡或低样本任务中的稳定性尚需进一步评估。

---

## 498. Generalized Discrete Diffusion with Self-Correction

**arXiv ID:** 2603.02230 | [PDF](https://arxiv.org/pdf/2603.02230v1)

**作者:** Linxuan Wang `[一作]`, Qifan Song `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Self‑Correcting Discrete Diffusion（SCDD）模型，利用预训练阶段的自我纠错机制，消除生成过程中的remasking步骤，实现更高效的并行生成；

**💡 创新点**

创新点在于重新设计前向噪声过程，明确区分mask与uniform噪声的信噪比，实现可单独调节；同时在反向过程中完全去除remasking，使自我纠错能力大幅提升；

**🔧 技术方法**

使用离散时间/连续时间的Markov扩散框架、贝叶斯后验推理、ELBO优化以及DiT网络骨干；

**📊 数据集**

在LM1B（One Billion Words）和OpenWebText（OWT）两个大型语言建模数据集上进行训练与评估；

**📈 对比分析**

与MDLM、GIDD、ReMDM等基线进行比较，SCDD在验证困惑度、生成困惑度以及纠错率上均优于对手，尤其在少步并行生成场景下表现突出；

**⚠️ 局限性**

局限性包括：1）在标准常识基准任务上表现不如mask‑only模型；2）对大规模参数模型的实验尚未展开；3）当前实现仍未结合强化学习等进一步提升自我纠错。

---

## 499. RegTrack: Uncovering Global Disparities in Third-party Advertising and Tracking

**arXiv ID:** 2603.02679 | [PDF](https://arxiv.org/pdf/2603.02679v1)

**作者:** Tanya Prasad `[一作]` (University of British Columbia), Thomas Pasquier `[通讯]` (University of British Columbia)

**通讯引用:** 1424 | [OpenAlex ID](https://openalex.org/A5005580571)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

通过同步爬虫在8个地理视角、4种浏览器和2种同意状态下，对743个热门网站进行测量，系统评估浏览器、用户位置、托管国对第三方广告跟踪的影响。

**💡 创新点**

首次在同一实验框架下同时操纵浏览器、用户地理位置和托管国，并使用因素设计对比三者对跟踪曝光的相对和交互效应，揭示用户可控因素与结构性环境的不同权重。

**🔧 技术方法**

采用Browsertime+Docker容器进行无状态爬取，收集HAR日志与截图；使用LLM视觉模型过滤失败页面；利用公共广告拦截列表（EasyList/EasyPrivacy等）进行域级分类；编写Cookie同意自动化脚本实现两种同意状态。

**📊 数据集**

743个全球/地区受欢迎的网站（Tranco +各国前100榜单），覆盖8个地区（加州、俄亥俄、魁北克、孟买、新加坡、法兰克福、巴黎、都柏林），使用Chrome、Edge、Firefox、Brave四款主流浏览器，每个组合采集10次访问，得到含网络请求与同意状态的完整数据集。

**📈 对比分析**

采用4×8×2因素设计，对比不同浏览器、位置与同意状态下的第三方域名数量、同意横幅出现率、跨境请求比例等指标；实验结果表明浏览器在宽松环境下可减少30%跟踪域，而用户位置对基线和同意后跟踪影响最大，托管国影响最小。

**⚠️ 局限性**

仅覆盖热门网站和8个视角，未包含移动端；域级拦截列表可能产生误报；同意状态仅为无点击和全部同意两种，未细粒度捕捉用户选择；网络级测量无法识别CNAME隐藏、数据量与敏感性；GeoIP跨境分析受CDN影响；实验为关联性分析，未证明因果关系。

---

## 500. SOLAR: SVD-Optimized Lifelong Attention for Recommendation

**arXiv ID:** 2603.02561 | [PDF](https://arxiv.org/pdf/2603.02561v1)

**作者:** Chenghao Zhang `[一作]`, Kun Gai `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于SVD的注意力机制SVD‑Attention，并在此基础上构建SOLAR框架，能够在保持softmax分布的同时将注意力复杂度从O(N²d)降至O(Ndr)，支持用户历史数千至万级、候选集千级的无过滤序列建模；

**💡 创新点**

创新点在于：①利用低秩结构对共享Key‑Value矩阵进行SVD分解，实现无损压缩的软max注意力；②在此基础上设计了集成候选集的set‑wise架构，理论证明其相对于传统point‑wise模型在排名偏差和泛化误差上具备优势；

**🔧 技术方法**

核心技术包括：SVD‑Attention（随机SVD、线性化前向与后向传播）、set‑wise注意力、候选集与历史序列双侧建模、随机SVD加速、点-点与集-集比较的理论分析；

**📊 数据集**

实验数据集涵盖公开的RecFlow、MIND离线基准以及Kuaishou真实流量（12k历史、3k候选）；

**📈 对比分析**

与Softmax、Linear、Longformer、Hiformer等注意力变体以及SIM、TWIN等两阶段模型对比，SOLAR在RecFlow、MIND、Kuaishou线上均取得最高AUC/Logloss，在线部署时机耗低约52%，并在离线实验中超越SVD‑Attention without softmax、仅候选集或仅历史建模等 ablation；

**⚠️ 局限性**

局限性主要是：目前仅在序列推荐场景验证，未针对语言、视觉或检索任务评估；SVD分解对rank r 的选取敏感，若低秩假设不成立可能收益有限；

---

## 501. Extending the Formalism and Theoretical Foundations of Cryptography to AI

**arXiv ID:** 2603.02590 | [PDF](https://arxiv.org/pdf/2603.02590v1)

**作者:** Federico Villa `[一作]` (ETH Zurich), Franziska Roesner `[通讯]` (University of Washington)

**通讯引用:** 7237 | [OpenAlex ID](https://openalex.org/A5058923617)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文构建了一个统一的形式化框架，用以分析和评估基于大型语言模型的自主代理的安全与可用性。

**💡 创新点**

创新点在于提出了攻击分类体系、完整性与安全的统一游戏模型、以及帮助性与无害性的模块化分解，并证明了训练数据保密性与完整性之间的不可兼容性。

**🔧 技术方法**

主要技术包括：信息安全三要素（机密性、完整性、可用性）对应的安全游戏、面向攻击者的能力标记、双模型（创意与消遣）双构造，以及基于决策与计算问题的可证明安全论证。

**📊 数据集**

论文未使用实际数据集，全部以理论模型和抽象算法为基础。

**📈 对比分析**

由于是理论研究，未给出实验对比或性能指标，只在文中给出了形式化证明和定理。

**⚠️ 局限性**

主要局限在于对真实系统的可实现性假设较强，预测和判定 ϕ 的可计算性未得到充分验证，且对大规模模型的实际部署缺乏实验验证。

---

## 502. RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces

**arXiv ID:** 2603.03137 | [PDF](https://arxiv.org/pdf/2603.03137v1)

**作者:** Yuhang Zhang `[一作]` (University of Science and Technology of China), Feng Wu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 47947 | [OpenAlex ID](https://openalex.org/A5100694761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于强化学习的框架，用仿真和UV映射生成可在3D表面上覆盖变形物体（如海绵）高效路径。

**💡 创新点**

创新点包括：使用谐波UV映射将表面状态与动作空间降维到二维；采用SGCNN高效提取特征；利用力学反馈仿真与动作约束实现接触丰富的覆盖路径。

**🔧 技术方法**

主要技术包括：Mujoco仿真平台、UV映射、谐波映射、SGCNN特征提取、SAC强化学习算法、力学反馈与动作空间限制。

**📊 数据集**

使用SPONGE数据集中的10个物体以及车门、人体模型等复杂几何体进行定量和定性实验。

**📈 对比分析**

与SPONGE两阶段路径规划、传统斜线和螺旋扫描基线对比，实验表明本方法在路径长度、覆盖面积和旋转角累计变化上均显著优于基线（路径长度缩短约27%，覆盖面积提升约1.4%，旋转角累计减少约40%）。

**⚠️ 局限性**

局限性：未考虑机械臂关节可达性导致部分路径不可执行；仿真与真实海绵变形存在差距；缺乏实时视觉或触觉反馈，算法目前为离线，难以适应动态目标。

---

## 503. HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations

**arXiv ID:** 2603.03243 | [PDF](https://arxiv.org/pdf/2603.03243v1)

**作者:** Xiaomeng Xu `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**通讯引用:** 24808 | [OpenAlex ID](https://openalex.org/A5004644695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过机器人自由的人类演示，训练了可以在移动平台上执行全身协调、主动感知和长程导航的全身移动操作策略。

**💡 创新点**

核心创新包括：① 在UMI基础上加入头戴摄像头并使用ARKit实现多设备同步；② 采用3D视觉表示消除视觉体现差距；③ 将头部动作抽象为3D注视点，缓解运动学差距；④ 设计约束感知的全身控制器，使策略输出在机器人硬件约束下可执行。

**🔧 技术方法**

技术手段包括：iPhone多机协作采集RGB+深度+6-DoF位姿；Diffusion Policy进行端到端视觉运动学习；三维点云+视觉特征的融合编码；看点控制算法；基于QP的差分全身IK控制；异步策略推理与控制桥接。

**📊 数据集**

数据集：在多种环境下收集200+次“洗衣”、166次“交付”、115次“桌布”演示，使用iPhone拍摄的RGB/深度视频与手部位姿同步记录。

**📈 对比分析**

与仅使用腕摄像头的UMI、仅头摄像头、无头部动作、无腕摄像头等基线对比，HoMMI在所有三项长程任务中均达到约80-90%的成功率，明显优于基线，尤其在需要全局视野与主动搜索的情境中表现突出。

**⚠️ 局限性**

局限性包括：观察窗口短，难以在长任务中实现完整回溯；缺乏触觉/力感知，限制了接触丰富任务的安全与精度；尽管硬件尽量匹配，仍存在细微的物理体现差距，可能影响迁移。

---

## 504. Matrices with displacement structure: a deterministic approach for linear systems and nullspace bases

**arXiv ID:** 2603.02425 | [PDF](https://arxiv.org/pdf/2603.02425v1)

**作者:** Sara Khichane `[一作]` (Sorbonne Université), Vincent Neiger `[通讯]` (Sorbonne Université)

**通讯引用:** 264 | [OpenAlex ID](https://openalex.org/A5016685971)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

提出一种完全确定性的算法，能在 O(α^{ω‑1}(m+n)) 次基本运算内求解任意矩阵大小的 Toeplitz‑、Vandermonde‑ 与 Cauchy‑结构线性系统，并给出其零空间的紧凑描述。

**💡 创新点**

创新点在于：①把结构化线性系统转换为单变量多项式模方程；②利用向量 M‑Padé 与同步 M‑Padé 近似的非齐次版本，结合逆模运算与重排技术，完成三步求解；③在不使用随机预/后乘子的前提下，获得与最速随机算法相当的时间复杂度。

**🔧 技术方法**

核心技术包括：多项式矩阵的 Popov/弱 Popov 标准化、向量与同步 M‑Padé 近似（非齐次化）、重排多项式与逆模运算、快速多项式/矩阵乘法（复杂度为 n^{ω‑1}），以及对三类结构化矩阵的多项式解释。

**📊 数据集**

无实验数据集，论文完全基于理论证明和算法复杂度分析。

**📈 对比分析**

与现有随机化算法（如 Strassen‑式分治、Kaltofen 预处理）在理论复杂度上相当（O(α^{ω‑1} log(m+n))），且在所有输入下均为确定性；相比之下，随机化方法需额外随机化步骤，存在失败概率。

**⚠️ 局限性**

局限性包括：①仍需 α（位移秩）已知且较小；②实现复杂度高，实际常数可能较大；③对最一般的位移运算（非可逆或重点重复）尚未完全覆盖；④多项式逆模等步骤在实践中可能成为瓶颈。

---

## 505. LAGO: A Local-Global Optimization Framework Combining Trust Region Methods and Bayesian Optimization

**arXiv ID:** 2603.02970 | [PDF](https://arxiv.org/pdf/2603.02970v1)

**作者:** Eliott Van Dieren `[一作]` (École Polytechnique Fédérale de Lausanne), Fabio Nobile `[通讯]` (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种名为LAGO的局部-全局优化框架，将梯度增强的贝叶斯优化与信赖域局部搜索通过自适应竞争机制结合，实现一次评估即可决定全局探索还是局部精炼。

**💡 创新点**

创新点在于：① 在提议阶段将全局采样与局部搜索严格分离；② 通过长度尺度约束只将符合条件的局部点加入全局高斯过程；③ 用预测改进值自适应选择全局或局部候选，避免显式阶段切换。

**🔧 技术方法**

采用梯度增强高斯过程、期望改进采集函数、SR1信赖域二次模型以及自适应停止准则等技术。

**📊 数据集**

使用经典合成基准函数（Branin、Rosenbrock、Levy、Styblinski–Tang 2D/5D、Sphere）以及一个可通过伴随法求梯度的PDE约束优化问题。

**📈 对比分析**

与BO、gradBO、BLOSSOM、TuRBO、TREGO、LABCAT、BADS、L‑BFGS等方法比较，LAGO在多峰、较高维度问题上均能稳健收敛至全局最优，局部方法易陷入次优解，LAGO整体性能优于或与BLOSSOM持平，且在PDE问题中表现突出。

**⚠️ 局限性**

局限性在于：主要适用于低至中等维度；信赖域部分假设无噪声；对高维或噪声观测的扩展仍待研究。

---

## 506. GloPath: An Entity-Centric Foundation Model for Glomerular Lesion Assessment and Clinicopathological Insights

**arXiv ID:** 2603.02926 | [PDF](https://arxiv.org/pdf/2603.02926v1)

**作者:** Qiming He `[一作]` (Tsinghua University), Yonghong He `[通讯]` (Tsinghua University)

**通讯引用:** 4852 | [OpenAlex ID](https://openalex.org/A5100719304)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了GloPath，一种以肾小球实体为中心的自监督预训练全能模型，用于小球病变识别、分级、跨模态诊断以及临床病理关联挖掘。

**💡 创新点**

创新点在于将小球实体视为学习单元，采用多尺度、多视角的对比学习进行自监督预训练，从而获得更具解剖学语义的特征，显著提升诊断精度、跨模态泛化能力和少样本学习表现。

**🔧 技术方法**

核心技术包括ViT-Base架构、DINO对比学习框架、多尺度/多视角增广、实体检测、语义分割、attention可视化、t‑SNE/UMAP等可解释性手段。

**📊 数据集**

使用七大多中心数据集（XJ-Light‑1/2、XJ‑IF、XJ‑GIO、AIDPATH‑G、KPMP‑G、XJ‑CLI），涵盖H&E、PAS、MT、PASM、IF等五种染色，共计约1.02 M个小球。

**📈 对比分析**

与六类基线模型（RandomInit、ImageNetPre、UNI、PLIP、CONCH、RenalPath）在52项任务（识别、分级、跨模态、少样本）进行比较，GloPath在42/52任务中获得最高分，整体F1>0.95，ROC‑AUC达到91.5%；在真实临床集XJ‑CLI保持0.91‑0.99的高性能，且在少样本任务中表现优于所有基线。

**⚠️ 局限性**

局限性包括对扫描仪、染色协议等低层次差异的适配不足、少见或早期病变样本不足、预训练主要来自单中心数据导致潜在偏差、未开展颜色归一化或域自适应研究，需进一步在更广泛多中心、异构环境中验证。

---

## 507. EIMC: Efficient Instance-aware Multi-modal Collaborative Perception

**arXiv ID:** 2603.02532 | [PDF](https://arxiv.org/pdf/2603.02532v1)

**作者:** Kang Yang `[一作]` (Renmin University of China), Yongcai Wang `[通讯]` (Renmin University of China)

**通讯引用:** 5472 | [OpenAlex ID](https://openalex.org/A5053362548)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种早期协同多模态感知框架EIMC，能够在局部融合阶段注入轻量化协作体素，并通过热图驱动的实例通信策略实现低带宽、高精度的目标检测。

**💡 创新点**

创新点在于：① 将跨车体体素先行注入到本机多模态融合中，生成紧凑而信息丰富的三维协作先验；② 利用热图差异精准定位协同需求，只传输高价值实例向量；③ 采用实例完成与实例精炼双层自注意力机制，显著降低冗余信息。

**🔧 技术方法**

采用 Mix‑Voxel 体素融合、Occupancy‑Guided 图像体素、Heterogeneous Modality Fusion（HMF）、Instance Completion（IC）与Instance Refinement（IR）等模块，并结合多尺度特征与跨模态注意力。

**📊 数据集**

在 OPV2V 与 DAIR‑V2X 两个公开协同感知基准上进行评估。

**📈 对比分析**

与现有最佳方法相比，EIMC 在两数据集上均取得 AP（0.5/0.7）最高，且通信量下降约 88%，实现了性能与带宽的最优平衡。

**⚠️ 局限性**

局限性在于对极端姿态噪声或极大车队规模的鲁棒性仍有限，且多模态协同需要一定硬件与网络同步支持。

---

## 508. Maximizing Generalization: The Effect of Different Augmentation Techniques on Lightweight Vision Transformer for Bengali Character Classification

**arXiv ID:** 2603.02591 | [PDF](https://arxiv.org/pdf/2603.02591v1)

**作者:** Rafi Hassan Chowdhury `[一作]` (Islamic University of Technology), Kaniz Fatiha `[通讯]` (Islamic University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究不同图像增强技术对轻量级视觉变换器 EfficientViT 在孟加拉手写字符识别中的泛化性能影响

**💡 创新点**

首次系统评估多种增强组合，发现随机仿射 + 颜色抖动组合在两大孟加拉字符数据集上显著提升准确率

**🔧 技术方法**

使用轻量级 Vision Transformer（EfficientViT）、预训练权重、交叉熵损失、早停等；结合 CLAHE、随机旋转、随机仿射、颜色抖动及其组合进行数据增强

**📊 数据集**

Ekush 与 AIBangla 两个公开手写孟加拉字符数据集的基本字符子集

**📈 对比分析**

在相同增强配置下与 MobileViT、TinyViT、DConvAENNet、CNN 等模型对比，随机仿射 + 颜色抖动组合在 Ekush 达 97.48%，AIBangla 达 97.57%，超越现有方法

**⚠️ 局限性**

受限于数据集规模与类间相似导致误分类；过度增强可能导致特征失真；未探索 GAN、翻转、噪声注入等更丰富的增强方式

---

## 509. Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning

**arXiv ID:** 2603.03236 | [PDF](https://arxiv.org/pdf/2603.03236v1)

**作者:** Fangzhou Yao `[一作]` (University of Science and Technology of China), Qi Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 24895 | [OpenAlex ID](https://openalex.org/A5100453158)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于多代理的预览‑分析‑推理框架 ParLD，用于在多轮对话中持续诊断学生认知状态。

**💡 创新点**

将心理学的最近发展区理论与行为预览模式结合，构建了可迭代的链式反射自我纠错循环，实现更细粒度、可靠的学习诊断。

**🔧 技术方法**

依托 GPT‑4.1/GPT‑4o 的提示式推理，设计行为预览器、状态分析器、性能推理器与链式反射器四个 LLM 代理，并使用对话记忆实现迭代反思。

**📊 数据集**

在 MathDial 与 CoMTA 两大对话数据集上进行实验，利用其中的多轮教师‑学生交互与最终掌握标签。

**📈 对比分析**

通过与传统知识跟踪模型（DKT、AKT、DKVMN、SAINT、SimpleKT）在最终性能预测任务中的比较，ParLD 在 MathDial 与 CoMTA 上准确率与 F1 均领先约10%+，并在模拟教学支持任务中显著提升正确率、减少对话轮数。

**⚠️ 局限性**

仍受限于对 LLM 生成标签的评估可靠性、缺乏长期真实学生数据验证以及反射循环的成本与实时性挑战。

---

## 510. Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection

**arXiv ID:** 2603.02286 | [PDF](https://arxiv.org/pdf/2603.02286v1)

**作者:** Yaoteng Zhang `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 17817 | [OpenAlex ID](https://openalex.org/A5100341321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为PDP的增量目标检测框架，解决传统提示方法中提示耦合与漂移导致的性能退化问题。

**💡 创新点**

核心创新包括：1）双池提示解耦（共享池负责任务通用知识，私有池负责任务特定知识），有效分离提示并提升前向迁移与防止干扰；2）基于原型的伪标签生成（PPG）模块，利用类别原型在嵌入空间中筛选伪标签，保持监督一致性并缓解提示漂移。

**🔧 技术方法**

使用提示解耦技术、前向查询检索、前缀调优（Prefix‑Tuning）、方向解耦损失、以及原型相似度校验的伪标签生成。整个模型基于Deformable‑DETR/MD‑DETR架构实现。

**📊 数据集**

在MS‑COCO和PASCAL‑VOC两个公开增量检测基准上进行实验。

**📈 对比分析**

与现有的多种增量检测方法（如OW‑DETR、MD‑DETR、PseDet、CL‑DETR等）对比，PDP在COCO的mAP@A达到59.4%（相较前沿方法提升约9.2%），在VOC的mAP@A提升2.9%~3.3%，在所有任务中均表现出最优的稳定性与可塑性平衡。

**⚠️ 局限性**

限制在于：1）对原型更新和伪标签阈值选择敏感，需经验调参；2）共享池规模若过大可能导致冗余与优化困难；3）当前框架未处理跨域或极少样本的增量场景，未来可进一步探索。

---

## 511. Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency

**arXiv ID:** 2603.02936 | [PDF](https://arxiv.org/pdf/2603.02936v1)

**作者:** Nicholas Carlotti `[一作]` (Dalle Molle Institute for Artificial Intelligence), Alessandro Giusti `[通讯]` (Dalle Molle Institute for Artificial Intelligence)

**通讯引用:** 8681 | [OpenAlex ID](https://openalex.org/A5052119291)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用无人机自身的里程计信息，以自监督方式对仿真训练的门姿估计网络进行域适配，从而实现从仿真到真实环境的迁移。

**💡 创新点**

提出了基于状态一致性损失的自监督域适配方法，利用无人机在不同位置的运动相对关系作为监督信号，无需外部标注或运动捕捉，显著降低了域适配的数据成本。

**🔧 技术方法**

使用了预训练的卷积神经网络（四个卷积块 + 线性层），6D 旋转表示，扩展卡尔曼滤波得到的里程计，状态一致性损失（SC loss），以及 8 位量化与 GAP9 NE16 神经加速器实现的低延迟推理。

**📊 数据集**

仿真数据集 75k 张门图像；真实世界数据集 100k 张图像，划分为训练（51k）、验证（8k）和测试（21k）集；门尺寸 100cm × 80cm，采用随机轨迹采集；还使用运动捕捉系统进行评估标注。

**📈 对比分析**

与 Zero‑Shot、PencilNet（域泛化）和 MMD‑DA（无监督域适配）等基线对比；在真实测试集上平均绝对误差为位置（x=26cm，y=28cm，z=10cm）和姿态角（ψ=13°），相比基线提升 40%/37%；推理时间 30.4 ms（33 fps），仅需 10 分钟的飞行数据即可超越所有基线。

**⚠️ 局限性**

仅在门为静态且无人机里程计相对准确的条件下有效；存在定位偏置需后期校准；在门被遮挡或视角极端时模型失效；未在闭环飞行中验证，且未考虑不确定性估计。

---

## 512. REGAL: A Registry-Driven Architecture for Deterministic Grounding of Agentic AI in Enterprise Telemetry

**arXiv ID:** 2603.03018 | [PDF](https://arxiv.org/pdf/2603.03018v1)

**作者:** Yuvraj Agrawal `[一作]` `[通讯]` (Adobe Inc), Yuvraj Agrawal (Adobe Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 REGAL 架构，将企业遥测通过 Medallion ELT 转化为确定性 Gold artifacts，并通过 registry-driven 编译生成 MCP 工具，让 LLM 只在确定性输出上推理。

**💡 创新点**

将 registry 作为单一真相源，生成工具接口实现“interface‑as‑code”，实现工具漂移防止、行动空间受限、治理嵌入，明确分离确定性计算与概率推理。

**🔧 技术方法**

Medallion ELT、Deterministic ingestion、Gold artifacts、Model Context Protocol (MCP)、Registry‑driven semantic compilation、LLM agent 推理、推拉事件流、缓存与访问控制。

**📊 数据集**

采用企业内部多源遥测（版本控制、CI/CD、问题追踪、可观测平台）的实时数据，未公开具体公开数据集。

**📈 对比分析**

通过原型实现与手动跨系统分析对比，发现令牌使用量下降、交互延迟受限于模型推理；实验表明 deterministic 方案在交互延迟和 token 规模上优于 RAG/Text‑to‑SQL，但未提供大规模基准。

**⚠️ 局限性**

原型规模有限、样本不足、未与完整 Text‑to‑SQL 或 Vendor AI 进行统一基准评测、模型性能波动、对大规模部署的可扩展性未充分验证。

---

## 513. Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches

**arXiv ID:** 2603.02655 | [PDF](https://arxiv.org/pdf/2603.02655v1)

**作者:** Anum Afzal `[一作]` (Technical University of Munich), Tatsuya Ishigaki `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5063285759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种基于提示的暂停感知解码策略（固定间隔和动态间隔），实现不需要微调的多模态大语言模型在实时视频解说中的应用；

**💡 创新点**

首次引入反馈循环与动态时间窗口，使模型能够在提示层面决定何时发声，提升与人类解说时机的一致性；

**🔧 技术方法**

利用多模态大语言模型（如 GPT‑4.1、LLaVA‑NeXT‑Video、Qwen2.5‑VL‑7B‑Instruct），并通过统一提示模板与语音速率估计实现暂停控制；

**📊 数据集**

在日英两语的赛车（English Racing、Japanese Racing）和日语格斗游戏（SmashCorpus）三大数据集上评测；

**📈 对比分析**

与固定间隔解码（含0/8示例）进行对比，自动指标显示固定间隔表现稍优，但人工评估表明动态间隔在关键事件识别、暂停意识和自然度上更佳；

**⚠️ 局限性**

受限于无微调、语言一致性不足、语速估计不精准、评测范围局限于赛车/格斗游戏，且生成文本往往冗长，难以与人类解说完全匹配；

---

## 514. Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers

**arXiv ID:** 2603.02919 | [PDF](https://arxiv.org/pdf/2603.02919v1)

**作者:** Youngjun Jun `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**通讯引用:** 22088 | [OpenAlex ID](https://openalex.org/A5051395190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种可解释的运动注意图（IMAP），通过在视频扩散 Transformer（Video Diffusion Transformer）中定位运动概念的空间和时间位置，从而实现对运动生成过程的可视化。

**💡 创新点**

创新点包括：① 利用查询‑键匹配生成文本替代令牌，并通过 Gram 矩阵得到的相似性映射（1）实现对任何概念的空间可解释性；② 通过帧间分离度量（如 Calinski‑Harabasz）自动识别运动相关的注意头，进一步实现时空运动定位；③ 整个方法无需梯度、参数更新或额外训练，即可在零样本下生成高质量运动可视化。

**🔧 技术方法**

使用的技术主要包括：查询‑键匹配、Gram 矩阵相似性映射、帧间分离度量（CHI）、层与头选择策略、以及基于 CogVideoX/HunyuanVideo 等 Video Diffusion Transformer 的推理框架。

**📊 数据集**

实验数据集：MeViS（504 条视频，150 种运动类型）用于运动定位评估；VSPW（343 条视频，124 类对象）用于零样本视频语义分割；使用 OpenAI o3‑pro LLM 对生成的可解释地图进行多维度评估。

**📈 对比分析**

与 ViCLIP、VideoCrafter2+DAAM、Cross‑Attention、ConceptAttention 等基线进行对比。IMAP 在 5 个评估指标（空间定位、时间定位、提示相关性、特异性/稀疏性、边界质量）上均取得最佳分数；在零样本视频语义分割任务中，IMAP 的 mIoU 最高，远超现有的解释性映射方法，展示了显著的性能提升。

**⚠️ 局限性**

局限性：① 运动头的选择仍依赖于帧间分离度量，可能不适用于所有 Video Diffusion Transformer 结构；② 在多概念竞争场景下，1 的单列Gram映射可能缺乏区分度，虽然可通过 Softmax 改善但会牺牲帧间一致性；③ 对超长视频或极其复杂的运动场景，IMAP 的效果和稳定性尚待进一步验证。

---

## 515. MIRAGE: Knowledge Graph-Guided Cross-Cohort MRI Synthesis for Alzheimer's Disease Prediction

**arXiv ID:** 2603.02434 | [PDF](https://arxiv.org/pdf/2603.02434v1)

**作者:** Guanchen Wu `[一作]` (Emory University), Carl Yang `[通讯]` (Emory University)

**通讯引用:** 3917 | [OpenAlex ID](https://openalex.org/A5006897094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出MIRAGE框架，在缺少MRI的阿尔茨海默病患者中，通过生物医学知识图谱与冻结的3D U-Net解码器，将电子健康记录映射至结构化MRI潜在空间，实现跨队列MRI合成并显著提升诊断性能。

**💡 创新点**

创新点包括①将缺失MRI问题重新定义为解剖学引导的跨模态潜在蒸馏任务；②利用知识图谱与图注意网络实现异构EHR到MRI潜在空间的传播；③在训练时冻结3D U-Net作为结构约束器，并通过邻居跳连接补偿高频信息，避免在推理时生成完整体素图像。

**🔧 技术方法**

技术手段涵盖3D U-Net自编码器、GATv2图注意网络、白化-色彩对齐、SapBERT实体嵌入、邻居跳连接聚合、联合重建+分类损失、医学影像预处理（skull-strip、MNI配准）以及后续3D CNN分类器。

**📊 数据集**

使用ADNI数据库1175名患者的88维EHR、91×109×91体素MRI以及二分类标签，并配合iBKH知识图谱与SapBERT进行EHR–KG对接。

**📈 对比分析**

与多种基线（EHR单模态LR/MLP/RF、EHR→MRI生成、EHR→潜在MLP、MVAE、条件扩散）对比，MIRAGE在无真实MRI的队列中将AD分类的平衡准确率提升13%，融合后BAcc达70%，同时保持较高的AUC和特异性。

**⚠️ 局限性**

局限性在于仅在ADNI内部数据上验证，缺乏对外部真实EHR-only队列的泛化评估；同时模型高度依赖知识图谱，跨机构语义匹配仍需进一步改进。

---

## 516. Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings

**arXiv ID:** 2603.02233 | [PDF](https://arxiv.org/pdf/2603.02233v1)

**作者:** Jean-Baptiste Fermanian `[一作]` (PreMeDICal, Inria, Idesp, Inserm, University of Montpellier), Aurélien Bellet `[通讯]` (Univ. Lille)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自适应的个性化联邦学习框架，通过学习权重来融合多方数据，得到针对目标客户端的更优模型。

**💡 创新点**

创新点在于：①将个性化联邦学习问题转化为多源高维均值（KME）估计；②利用 Q‑aggregation 估计混合权重，提供无先验异构假设的泛化误差界；③使用随机傅里叶特征实现低通信成本的 KMEs 共享，兼顾统计效率与通信效率。

**🔧 技术方法**

核心技术包括：核均值嵌入（Kernel Mean Embedding）与最大均方差（MMD）度量；Q‑aggregation 均值估计算法；随机傅里叶特征（Random Fourier Features）近似 RKHS；在联邦设置下的梯度/风险聚合。

**📊 数据集**

实验数据集：①概念漂移（concept shift）与协变量漂移（covariate shift）下的合成数据；②公共 MNIST 联邦版本（FedMNIST），每个客户端持有 62 类手写字样本。

**📈 对比分析**

与基线比较：局部训练（Local）、全局平均（GrandMean）以及拥有先验相似度信息的 Oracle。结果显示：该方法在大多数情形下优于 Local 和 GrandMean，且在可行时接近 Oracle 的性能；在高异构情形下能自动减少协作，避免性能下降。

**⚠️ 局限性**

局限性：①需要在 RKHS 内假设损失函数，需对实际任务进行核选择；②未量化 KMEs 共享带来的隐私风险；③仅对单一目标客户端求解，未考虑多目标同步训练；④对大规模高维特征的随机特征维度与通信开销仍需进一步平衡。

---

## 517. On Geometry Regularization in Autoencoder Reduced-Order Models with Latent Neural ODE Dynamics

**arXiv ID:** 2603.03238 | [PDF](https://arxiv.org/pdf/2603.03238v1)

**作者:** Mikhail Osipov `[一作]` `[通讯]` (Independent Researcher), Mikhail Osipov (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在自编码器预训练阶段引入四种几何正则化策略（近等距、方向增益、二阶曲率以及Stiefel投影），研究其对低维潜在动力学模型（神经ODE）训练与长期滚动预测性能的影响。

**💡 创新点**

创新点在于系统性比较了仅针对解码器Jacobian的几何正则化与对解码器第一层施加的Stiefel投影对潜在动力学学习的实际影响，发现前者虽能提升局部平滑度但会削弱潜在动力学的可学习性，而后者则显著改善潜在动力学的条件数与滚动误差。

**🔧 技术方法**

技术上采用卷积自编码器+神经ODE框架，并实现了近等距损失、随机方向增益损失、二阶曲率损失以及对解码器权重进行Stiefel投影，随后在冻结编码器的前提下训练多组随机种子。

**📊 数据集**

数据集为基于参数化流动‑扩散‑反应（ADR）偏微分方程的数值求解生成的高维（32×32=1024维）时间序列，包含训练、插值与外推子集。

**📈 对比分析**

比较方法是在相同重构目标下冻结编码器，训练多重随机种子，评估不同正则化策略在长周期滚动误差（平均/最大相对误差）以及潜在动力学诊断（条件数、增益、跟踪误差）上的表现；结果显示Stiefel投影在平均与最大误差上与基线相当或略优，而其余正则化均逊于未正则化基线。

**⚠️ 局限性**

局限性在于仅针对单一ADR基准进行验证，未探讨更复杂系统或联合正则化方案，且对非冻结编码器的影响未深入研究，结果可能不具普遍适用性。

---

## 518. Guiding Sparse Neural Networks with Neurobiological Principles to Elicit Biologically Plausible Representations

**arXiv ID:** 2603.03234 | [PDF](https://arxiv.org/pdf/2603.03234v1)

**作者:** Patrick Inoue `[一作]` (KEIM Institute), Andreas Knoblauch `[通讯]` (KEIM Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Hebbian 学习与权重扰动（WP）的生物学可解释学习规则，能够在保持稀疏性、lognormal 权重分布以及 Dale 法则的同时实现对抗鲁棒性与少样本泛化。

**💡 创新点**

创新点在于将核心神经生物学原则隐式嵌入学习规则，无需显式强制约束，从而自然产生稀疏、去相关、达尔法则合规的网络结构，并显著提升对抗攻击抵御能力和少样本学习表现。

**🔧 技术方法**

技术方法包括 Hebbian 更新、WP 权重扰动、非负性约束、层级归一化、权重归一化以及多层 MLP 架构；实验中使用了 ReLU 激活并对输出层结合 BP 或 WP。

**📊 数据集**

使用的公开数据集为 MNIST（手写数字）和 CIFAR-10（自然图像），并在两者上评估。

**📈 对比分析**

与标准 BP、Krotov‑Hopfield 及其他基准网络进行对比；在 MNIST 单层网络上获得 97–98% 的准确率，在 CIFAR‑10 上 90–95% 之间；在 1‑shot、10‑shot 等少样本任务及 FGSM/PGD 对抗攻击下，表现优于 BP 并略高于现有生物学启发方法，但整体准确率仍略低于最优 BP 训练。

**⚠️ 局限性**

主要限制包括 WP 计算开销大、收敛速度慢，导致深层网络训练时间长；在更复杂的数据集（如 ImageNet）和 CNN 结构上性能仍不理想；未验证脉冲神经网络或 neuromorphic 硬件上的可迁移性。

---

## 519. Quadratic-Order Geodesics on Meshes

**arXiv ID:** 2603.03231 | [PDF](https://arxiv.org/pdf/2603.03231v1)

**作者:** Yue Ruan `[一作]`, Amir Vaxman `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于二次有限元的凸优化框架，用于在三角网格上精确计算平方测地距离，并允许源点位于网格内任意位置。

**💡 创新点**

创新点在于：①直接求平方测地距离而非原始距离，天然适配二次元函数；②使用凸优化实现可解析的粘性解；③实现了在粗糙、非均匀网格甚至非流形网格上高度鲁棒且无源点限制的测地距离计算。

**🔧 技术方法**

采用了二维三角网格的二次有限元（PQ）表示、凸优化（CVX/MOSEK）、梯度矩阵、质量矩阵等离散化技术；将Eikonal方程转化为 u = |∇u|^2/4 的凸最大子解问题。

**📊 数据集**

在 Thingi10k 数据集（4320 个闭合单连通模型）上进行实验，并使用细分后的 FEG 结果作为精确地面真值。

**📈 对比分析**

与 Fast Marching、Fast Exact Geodesics、Heat Method、DFA 等线性/粘性解法对比，方法在 L2、L∞ 误差上普遍优于竞争者，并在低质量网格、噪声、缺失区域、非流形结构等极端条件下保持稳定。

**⚠️ 局限性**

局限性包括：仍依赖有限元梯度定义，只适用于三角网格；尚未实现高效 ADMM 求解器；对点云或更一般的离散几何结构尚未扩展。

---

