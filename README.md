# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-23 | 今日论文总数: 478

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images

**arXiv ID:** 2603.19766 | [PDF](https://arxiv.org/pdf/2603.19766v1)

**作者:** Donghai Fang `[一作]` (Sun Yat-sen University), Wenwen Min `[通讯]` (Yunnan University)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5083513047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了名为 HINGE 的框架，将预训练的单细胞基础模型（scFM）迁移到仅依赖 H&E 组织切片的空间转录组表达生成任务。

**💡 创新点**

创新点包括：① 通过 SoftAdaLN 在冻结的 scFM 中注入层级的视觉上下文，使模型在保持基因间依赖的同时获得组织条件；② 设计基于掩码的扩散过程与训练目标，使其与 scFM 的掩码自编码预训练保持一致；③ 引入 warm‑start curriculum，在低掩码阶段先行训练以缓解稀疏监督导致的梯度不稳定。

**🔧 技术方法**

技术上使用了：预训练的 CellFM（Transformer‑based scFM），SoftAdaLN 轻量级条件化模块，基于掩码的扩散生成模型，热启动时间步采样策略，以及对 H&E 图像的卷积/Transformer 编码器。

**📊 数据集**

实验数据集包括三套人类空间转录组数据集：cSCC、Her2ST 和 Kidney，每个数据集都提供 H&E 切片与对应 spot‑level 基因表达。

**📈 对比分析**

与六种基线（四个判别回归模型 ST‑Net、BLEEP、TRIPLEX、MERGE；两种生成模型 Stem、STFlow）在 PCC‑50、PCC‑200、MSE、MAE 等指标上比较，HINGE 在所有数据集的 PCC‑50/PCC‑200 上均优于基线，提升约 3–6%，且在标记基因空间分布和基因‑基因共表达矩阵上表现更为生物学一致。

**⚠️ 局限性**

局限性包括：① 需要与预训练的 Gene 词表对齐，限制了可使用基因数量；② 仅在 CellFM 上验证，需进一步测试对其他 scFM 的适用性；③ 对稀疏/混合细胞表达仍受限，低表达基因误差略大；④ 训练时间和计算资源需求仍相对较高。

---

## 2. Analyzing Decoders for Quantum Error Correction

**arXiv ID:** 2603.20127 | [PDF](https://arxiv.org/pdf/2603.20127v1)

**作者:** Abtin Molavi `[一作]` (University of Wisconsin-Madison), Aws Albarghouthi `[通讯]` (University of Wisconsin-Madison)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于程序语义和多项式优化的量化量子误差校正解码器准确率与鲁棒性的方法，能够精确估计低错误率下的逻辑错误率并对物理误差率漂移给出可靠界限。

**💡 创新点**

创新点在于：①给出完整的量子错误校正程序语言及其语义，②将解码器准确率与鲁棒性问题转化为误差多项式的评估和约束多项式优化；③设计了基于错误空间枚举与部分导数裁剪的高效求解算法，解决了传统蒙特卡洛仿真在低错误率和不确定性场景下的高采样成本。

**🔧 技术方法**

使用的技术包括：量子程序语义（小步与大步语义）、误差多项式构造、对多变量多项式的极值求解（超矩形约束下的极值位于顶点、部分导数裁剪与匹配项简化）、枚举策略（Hamming权重、分割搜索、局部搜索）以及采样与枚举相结合的置信区间估计。

**📊 数据集**

实验使用旋转表面码（surface codes）在不同代码距离d∈{3,5,7,9}与不同测量回合数r的内存实验程序，噪声模型为Google的si1000，错误率p取0.01、0.001、0.0001，评估三种主流解码器（pymatch、bp-osd、relay-bp）。

**📈 对比分析**

与传统蒙特卡洛模拟相比，枚举+采样在高噪声下需要更少的shots（至少10倍以上），在低噪声下更为高效；纯枚举在极低错误率下可实现无条件安全的上下界。鲁棒性评估在小规模程序（如d=3,r=3）可得到精确的最大逻辑错误率；相较于单点准确率，relay‑bp解码器鲁棒性更差。

**⚠️ 局限性**

局限性包括：对大规模程序（如误差通道数>200）求解鲁棒性仍受限于多项式优化的指数时间；对非超矩形不确定集、非多项式错误模型的扩展尚未实现；算法对解码器的黑盒性质依赖，无法直接嵌入解码器内部逻辑；最后，枚举策略在极低错误率下虽然高效，但对错误空间的全面覆盖仍受限。

---

## 3. Vision Tiny Recursion Model (ViTRM): Parameter-Efficient Image Classification via Recursive State Refinement

**arXiv ID:** 2603.19503 | [PDF](https://arxiv.org/pdf/2603.19503v1)

**作者:** Ange-Clément Akazan `[一作]` (AIMS Research Institute), Rose Bandolo `[通讯]` (AIMS Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Vision Tiny Recursion Model (ViTRM)，一种递归共享权重的参数高效视觉模型。

**💡 创新点**

用递归共享权重替代模型深度，证明递归计算可匹配 ViT 与 ResNet 的性能。

**🔧 技术方法**

采用 Transformer 块递归、交叉注意力、深度监督和早停机制。

**📊 数据集**

在 CIFAR‑10 与 CIFAR‑100 这两个标准分类数据集上进行评估。

**📈 对比分析**

与 ViT‑Small/Base/​Large 及 ResNet‑18/34/50 比较，ViTRM 仅 3.6M 参数即可获得 93.1%/72.1% 的准确率，超越所有 ResNet 并与小 ViT 接近。

**⚠️ 局限性**

局限性：仅验证于小规模分类任务，缺少对高分辨率或密集预测任务的评估，递归深度对任务复杂度的适应性尚未深入探讨。

---

## 4. Vocabulary shapes cross-lingual variation of word-order learnability in language models

**arXiv ID:** 2603.19427 | [PDF](https://arxiv.org/pdf/2603.19427v1)

**作者:** Jonas Mayer Martins `[一作]` (University of Göttingen), Lisa Beinborn `[通讯]` (University of Göttingen)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5087195265)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了词序不规则性对 Transformer 语言模型可学习性的影响，利用 Mallows 置换模型在自然语言上生成连续范围的词序扰动。

**💡 创新点**

创新点是提出了词级别确定性词序扰动方法，避免了子词拆分导致的语义与形态学混淆，并通过词汇统计揭示跨语言可学习性差异的真正驱动因素。

**🔧 技术方法**

使用了 Transformer 语言模型（PicoLM）、Mallows 置换模型进行词序扰动、PLS 回归分析来解释模型 surprisals。

**📊 数据集**

采用 Europarl 语料库，挑选十种欧洲语言（英、法、德、丹、瑞、拉、捷、芬、爱、匈）进行实验。

**📈 对比分析**

通过对不同 θ 值下的 surprisal 与词汇覆盖率、句长、形态复杂度等指标进行对比，发现词汇结构能解释 97% 的跨语言差异，模型对词序扰动的敏感性与词汇覆盖率高度相关。

**⚠️ 局限性**

局限包括仅使用欧洲语料、模型规模受限、仅评估计算可学习性而非人类可学习性、词序扰动仍未考虑更细粒度的形态学变化。

---

## 5. How international are international computing conferences? -- An exploration with systems research conferences

**arXiv ID:** 2603.19245 | [PDF](https://arxiv.org/pdf/2603.19245v1)

**作者:** Pedro Garcia Lopez `[一作]` (Universitat Rovira i Virgili), Anwitaman Datta `[通讯]` (Nanyang Technological University)

**通讯引用:** 6275 | [OpenAlex ID](https://openalex.org/A5001971672)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6`

**🎯 论文内容**

分析了过去13年13个主要计算机系统研究会议的论文与程序委员会的地理多样性。

**💡 创新点**

首次系统量化了国际论文与PC多样性变化，并揭示顶级会议对亚洲贡献的迟缓适应。

**🔧 技术方法**

采用爬虫抓取DBLP、OpenAlex、Semantic Scholar数据，使用Gini‑Simpson多样性指数计算多样性。

**📊 数据集**

共计9,712篇已收论文和14,996名PC成员（6,917名独立个体）来自2012‑2024年的13个会议。

**📈 对比分析**

通过比较2012‑2019与2020‑2024两段时期的多样性指数与PC比例，发现顶级会议在近四年才出现显著提升，且PC与论文的地理差距仍显著。

**⚠️ 局限性**

数据缺失导致约5%论文/PC未能归属大洲，手工归属可能存在偏差，且未深入探究导致差距的根本原因。

---

## 6. Benchmarking Post-Quantum Cryptography on Resource-Constrained IoT Devices: ML-KEM and ML-DSA on ARM Cortex-M0+

**arXiv ID:** 2603.19340 | [PDF](https://arxiv.org/pdf/2603.19340v1)

**作者:** Rojin Chhetri `[一作]` `[通讯]` (Independent Researcher), Rojin Chhetri (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在ARM Cortex-M0+（RP2040）平台上，对NIST标准化的ML-KEM（FIPS 203）和ML-DSA（FIPS 204）进行算法级基准测试，测量时间、内存占用、能耗，并发布了完整的开源基准套件。

**💡 创新点**

首次在最受限的32位Cortex-M0+上完整评测FIPS 203/204标准，揭示ML-DSA签名的拒绝采样导致的显著时延方差，并给出针对Class‑1 IoT设备的部署可行性分析。

**🔧 技术方法**

采用PQClean参考C实现、RP2040单周期乘法器、Pico SDK计时器、堆栈绘制与闪存统计，以及基于电源模型的能耗估算等技术。

**📊 数据集**

使用NIST提供的标准参数集（ML‑KEM 512/768/1024，ML‑DSA 44/65/87），随机数来源为RP2040环振荡器，未使用外部数据集。

**📈 对比分析**

与M4参考C的慢速比仅为1.8–1.9×，ML‑KEM‑512握手耗时36.3 ms，能耗2.87 mJ，较ECDH P‑256快17×、省能94%；ML‑DSA签名平均169–332 ms，CV61–71%，99th百分位高达1.1 s，表明签名时延不确定性显著。

**⚠️ 局限性**

研究仅基于参考C实现，未做汇编优化；仅在单一RP2040板上测试；未评估侧信道攻击、协议层开销；随机数来源不符合安全规范；能耗估算依赖简化电源模型。

---

## 7. HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks

**arXiv ID:** 2603.19822 | [PDF](https://arxiv.org/pdf/2603.19822v1)

**作者:** Jingyu Guo `[一作]` (University of Melbourne), Mingming Gong `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了HUGE‑Bench基准，用于评估无人机在短指令下执行多阶段视觉‑语言‑动作任务的能力。

**💡 创新点**

创新点包括：① 将短、模糊的自然语言指令与多阶段任务拆分相结合；② 采用3D Gaussian Splatting与Mesh相结合的数字孪生实现逼真渲染与物理碰撞检测；③ 提出了过程覆盖率（TCR）和碰撞感知指标（CR/CSPL）等新的评估指标。

**🔧 技术方法**

技术手段包括：3DGS、Mesh、Isaac Sim仿真、LLM辅助标注、RRT路径规划、以及OpenVLA、π_0、π_0.5、FastVLM等深度视觉‑语言‑动作模型。

**📊 数据集**

使用四个真实场景的数字孪生数据，共计2.56 km轨迹，包含8类任务，分为训练集、已知目标测试集和未知目标/语言测试集。

**📈 对比分析**

通过TCR@K、SR、CR、CSPL等指标对OpenVLA、FastVLM、π_0、π_0.5等模型进行对比；结果显示π系列模型在大多数任务上表现最佳，但整体过程覆盖率和安全率仍偏低，尤其在短指令情境下。

**⚠️ 局限性**

局限性：仅覆盖静态环境，缺乏动态障碍、光照与天气变化；数字孪生与真实飞行仍存在域差距，导致模型在真实部署中的泛化能力有限。

---

## 8. Learning to Disprove: Formal Counterexample Generation with Large Language Models

**arXiv ID:** 2603.19514 | [PDF](https://arxiv.org/pdf/2603.19514v1)

**作者:** Zenan Li `[一作]` (ETH Zurich), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14365 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了一套面向大型语言模型的正式反例生成框架，能够自动化地生成反例并给出可被 Lean 4 自动验证的正式证明。

**💡 创新点**

创新点包括：① 基于符号变异的反例数据合成策略，可将可证明的普遍命题通过丢弃必要假设转化为大量高质量反例问题；② 双奖励（multi‑reward）训练机制，分别对“消除假设”与“变异命题”两条证明的成功率给予奖励，显著缓解稀疏奖励问题并提升训练效率与效果。

**🔧 技术方法**

采用的技术包括：大型语言模型（LLM）用于非正式反例搜索（如 Qwen3 8B）和正式证明生成（如 DeepSeek‑Prover‑v2 7B），Lean 4 作为形式化验证器，符号变异（mutate）策略实现反例数据合成，以及专家迭代（expert‑iteration）结合多奖励的监督微调。

**📊 数据集**

主要使用数据集为：① Lean 4 Mathlib 与 Leanworkbook 中的正式定理；② 通过 LLM 生成的 MiniF2F 与 PutnamBench 的子目标；③ 通过上述变异生成的 575K 反例问题；④ 对抗实验中使用的 CounterMath、FormalMath、DSP+ 等公开 benchmark。

**📈 对比分析**

方法通过与单奖励训练以及现有公开 LLM/自动推理模型进行对比，结果显示：多奖励训练在验证集上的 pass@1、pass@4、pass@9 分别提升约 6%~8%；在三项评测任务（反例识别、自动化结果验证、推理步骤验证）中，相比最强基线，pass@1 最高提升可达 74%，pass@4 与 pass@9 也实现了显著提升，证明了框架的优越性。

**⚠️ 局限性**

主要局限包括：① 生成数据的质量参差不齐，冗余或错误样本仍占较大比例，导致训练效率低下；② 受限于 7B 规模的 LLM，非正式与正式推理模型在复杂问题上仍易失效；③ 训练资源受限，未能探索更大模型或工具‑使用策略，可能进一步提升性能。

---

## 9. Automated Motif Indexing on the Arabian Nights

**arXiv ID:** 2603.19283 | [PDF](https://arxiv.org/pdf/2603.19283v1)

**作者:** Ibrahim H. Alyami `[一作]` (Najran University), Mark A. Finlayson `[通讯]` (Florida International University)

**通讯引用:** 919 | [OpenAlex ID](https://openalex.org/A5109990506)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首次提出一种针对民间故事中主题（motif）索引的计算方法，并构建了包含2,670条主题表达的手工标注语料库，用于训练和评估该方法。

**💡 创新点**

创新点在于：①将大型、可获取的文本《一千零一夜》与详尽的El‑Shamy主题索引结合，解决了先前研究中索引文本难以获取的问题；②创建了第一份针对主题索引任务的标注数据集；③系统性比较了检索+再排序、基于向量嵌入（原始与微调）、以及生成式LLM（提示与LoRA微调）等多种模型方案。

**🔧 技术方法**

使用的技术包括：经典检索与再排序（keyword + cross‑encoder）、开源嵌入模型（如Sentence‑BERT）及其微调、基于Llama3的大型语言模型的微调与生成式提示（N‑shot），以及LoRA微调的LLM。

**📊 数据集**

数据集为：①《一千零一夜》的英文翻译文本（约45,769句子）；②El‑Shamy 2006年的主题索引（约5,000种主题）；③由研究团队手工标注的58,450句子中2,670条主题表达，用于训练与测试。

**📈 对比分析**

比较方法：在相同的主题索引条目上，对五种方案进行实验。最优方案为微调后的Llama3，整体F1为0.85；其他方案在F1上相对较低，说明微调和生成式提示在该任务上显著提升了效果。

**⚠️ 局限性**

局限性包括：①仅针对《一千零一夜》文本，无法直接验证对其他民间故事或语言的泛化能力；②标注语料覆盖的主题数量有限，仍缺乏对所有可能主题的完整表达；③实验集中在主题索引任务，未覆盖主题检测、分类或语义解释等更广泛的主题相关任务。

---

## 10. TAB-AUDIT: Detecting AI-Fabricated Scientific Tables via Multi-View Likelihood Mismatch

**arXiv ID:** 2603.19712 | [PDF](https://arxiv.org/pdf/2603.19712v1)

**作者:** Shuo Huang `[一作]` (Monash University), Lizhen Qu `[通讯]` (Monash University)

**通讯引用:** 2886 | [OpenAlex ID](https://openalex.org/A5008486397)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究构建了首个包含人工与AI生成论文表格的基准数据集，并提出了基于表格内部结构与数值不匹配的检测框架。

**💡 创新点**

创新点在于发现并量化了AI合成表格在文本骨架与数值内容之间的“内在不一致”，并将此信号与结构、数字统计特征结合，形成可解释的表格级鉴别方法。

**🔧 技术方法**

使用了语言模型观测器（如GPT‑2、Qwen3‑8B、Llama3.1‑8B）计算表格序列的 perplexity，以及传统机器学习分类器（逻辑回归、随机森林、MLP）进行论文级预测。

**📊 数据集**

数据集为 TAB‑AUDIT benchmark，包含 1,173 篇 AI 生成论文与 1,215 篇人工撰写的实验 NLP 论文，共 5,284 张表格，并设置 GPT‑5.2 生成的外部 holdout。

**📈 对比分析**

与零样本文本检测基线（Binoculars、DetectGPT）相比，基于内在不匹配的 Random Forest 模型在内部数据上实现 AUROC 0.987、TPR@5% 0.933；在外部 holdout 上仍保持 AUROC 0.883、TPR@5% 0.218。

**⚠️ 局限性**

局限性包括对观测语言模型的依赖、对表格提取与序列化的敏感性、对复杂表格布局和提取错误的脆弱性，以及在生成器迁移和单一可疑表格时可能稀释证据。

---

## 11. Agreement Between Large Language Models, Human Reviewers, and Authors in Evaluating STROBE Checklists for Observational Studies in Rheumatology

**arXiv ID:** 2603.19303 | [PDF](https://arxiv.org/pdf/2603.19303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 12. Fine-tuning Timeseries Predictors Using Reinforcement Learning

**arXiv ID:** 2603.20063 | [PDF](https://arxiv.org/pdf/2603.20063v1)

**作者:** Hugo Cazaux `[一作]` (Reykjavik University), Eyjólfur Ingi Ásgeirsson `[通讯]` (Reykjavik University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在已预训练的时序预测模型上应用强化学习进行微调，提升预测精度并实现迁移学习

**💡 创新点**

提出了将监督学习预训练与RL微调结合的框架，并对比三种RL算法（PPO、CMAPPO、GRPO），首次在金融时序预测任务中验证RL微调的有效性

**🔧 技术方法**

使用PPO、CMAPPO、GRPO等策略梯度算法，并结合Transformer预训练骨干、注意力聚合与群体优势估计等技术

**📊 数据集**

金融和ESG数据（Refinitiv、Sustainalytics、SASB）以及MuJoCo物理仿真环境（HalfCheetah‑v4、Hopper‑v4、Humanoid‑v4）

**📈 对比分析**

通过冻结不同比例的模型层、不同算法、不同子代理/群体大小等设置，比较MSE/MAE与基线的改进；GRPO在大多数情况下表现最佳，CMAPPO在金融数据上也有突出表现

**⚠️ 局限性**

模型输入固定尺寸、奖励设计对性能影响大、某些数据集微调后反而退化，且RL微调需要精细调参，尚未解决高维可变长度时序输入的问题

---

## 13. BrainSCL: Subtype-Guided Contrastive Learning for Brain Disorder Diagnosis

**arXiv ID:** 2603.19295 | [PDF](https://arxiv.org/pdf/2603.19295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 14. MedSPOT: A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI

**arXiv ID:** 2603.19993 | [PDF](https://arxiv.org/pdf/2603.19993v1)

**作者:** Rozain Shakeel `[一作]` (Gaash Research Lab National Institute of Technology Srinagar), Tajamul Ashraf `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MedSPOT基准，用于评估临床软件界面中的工作流感知顺序定位。

**💡 创新点**

创新点包括：①引入工作流依赖的顺序评估与严格早停机制；②设计六类错误分类用于系统性诊断；③针对医疗界面构建多步交互任务，填补了现有单步或通用界面基准的空白。

**🔧 技术方法**

技术手段主要是多模态大模型（视觉‑语言模型与大型语言模型）进行图文推理，模型输出点击坐标；评估使用顺序准确率、任务完成率、加权前缀分数等指标。

**📊 数据集**

数据集包含10个医学成像平台（如BlueLight、3D Slicer、Orthanc等）的216个任务、597个关键帧，任务平均2–3步，涵盖视图、分割、web等功能。

**📈 对比分析**

对16个先进多模态模型进行评估，通用模型任务完成率普遍为0%，最强模型GUI‑Actor仅达43.5%；显示顺序误差会导致任务完成率急剧下降，验证了工作流评估的重要性。

**⚠️ 局限性**

局限性：单语、规模有限；仅支持点击交互，未覆盖拖拽、滚动、文本输入等；严格早停机制可能低估部分过程能力；未考虑多模态交互中的更丰富语义层面。

---

## 15. Revealing Domain-Spatiality Patterns for Configuration Tuning: Domain Knowledge Meets Fitness Landscapes

**arXiv ID:** 2603.19897 | [PDF](https://arxiv.org/pdf/2603.19897v1)

**作者:** Yulong Ye `[一作]`, Tao Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种双管齐下的配置调优方法，结合空间信息分析和领域驱动分析，系统捕捉并解释配置调优案例的隐藏特性，说明调优器成功或失败的原因。

**💡 创新点**

创新点在于设计了配置空间性分析、配置领域分析以及二者的协同机制，并通过全面的多系统/多工作负载案例研究验证了该方法的有效性。

**🔧 技术方法**

采用的技术包括：配置空间性特征提取与可视化、领域特征抽取与建模、空间-领域协同分析框架，以及基于实验的评估与比较。

**📊 数据集**

使用的数据集涵盖多个主流软件系统与多种工作负载，具体包括但不限于系统A、系统B等多种配置与性能测量数据。

**📈 对比分析**

与传统单一维度调优方法（如基于启发式搜索、机器学习预测等）进行对比，实验结果显示本方法在平均性能提升约20%（视具体系统而定），并显著降低了调优失败率。

**⚠️ 局限性**

局限性包括：对极端高维或极少样本配置场景的适应性尚待验证；方法实现复杂度较高，需要专业知识进行空间与领域特征的提取与组合；以及在某些系统中，协同分析对计算资源的消耗可能不易接受。

---

## 16. Physical Layer Message Prediction for 5G Radio Access Network Protocols

**arXiv ID:** 2603.19760 | [PDF](https://arxiv.org/pdf/2603.19760v1)

**作者:** Jonathan Ebert `[一作]` (Karlsruher Institute of Technology), Peter Rost `[通讯]` (Karlsruher Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种基于 Transformer 的模型，用于预测 5G NR 物理层控制消息序列

**💡 创新点**

创新点在于将控制消息分层令牌化并引入语法校验机制，使模型既能学习语法，又能捕获隐藏状态，显著提升预测准确性

**🔧 技术方法**

采用 nanoGPT 风格的 Transformer 结构，训练参数约 2700 万，输入长度 1024 token，利用多头注意力（8 头）实现预测

**📊 数据集**

使用 srsRAN 开源实现生成的 5G NR 控制层数据集，包含多 UE 场景（2、3、10 UE）和不同流量方向的数据

**📈 对比分析**

通过 Levenshtein 距离与相对 Levenshtein 距离评估，语法检查后平均相对误差约 0.3，较无校验模型下降约 20%（对应 Levenshtein 距离从 13→12 级别），同时对单个通道的命中率和 RNTI 预测精度均达到 80% 以上

**⚠️ 局限性**

仅限于未加密的控制层消息，未对跨场景泛化或实际噪声环境进行验证，且对模型的可扩展性和训练数据量需求仍有待进一步研究

---

## 17. Anatomical Heterogeneity in Transformer Language Models

**arXiv ID:** 2603.19348 | [PDF](https://arxiv.org/pdf/2603.19348v1)

**作者:** Tomasz Wietrzykowski `[一作]` `[通讯]` (Independent Researcher), Tomasz Wietrzykowski (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对SmolLM2-135M 30层模型的权重、层重要性、可预测性、恢复速度等多维诊断，系统性揭示 Transformer 层的功能异质性并提出 Growth Transformer 训练方案。

**💡 创新点**

发现“抗层”与层重要性层级、统一的振荡权重变化模式、仅权重缩放可保持性能的事实，并利用恢复速度制定分层训练预算。

**🔧 技术方法**

层级消融、岭回归预测、delta相关性分析、权重替换策略、噪声恢复实验以及基于阶段的 Growth Transformer 训练。

**📊 数据集**

在 SmolLM2-135M 上使用包含10句多样英文句子的 held‑out 评估集；Growth 实验使用相同规模的数据进行训练。

**📈 对比分析**

与统一训练同参数、相同步数的基线相比，Growth Transformer 在验证损失上提升4.7×、训练时间缩短13%，在半预算下仍优于完整预算基线。

**⚠️ 局限性**

仅在单一小模型上验证，评估范围有限，抗层现象及 Growth 方案的可推广性尚需在大规模模型和多任务上进一步检验。

---

## 18. DePro: Understanding the Role of LLMs in Debugging Competitive Programming Code

**arXiv ID:** 2603.19399 | [PDF](https://arxiv.org/pdf/2603.19399v1)

**作者:** Nabiha Parvez `[一作]` (Military Institute of Science And Technology), Tarannum Shaila Zaman `[通讯]` (University of Maryland Baltimore County)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型（LLM）在竞赛编程调试任务中的性能进行了手工和经验研究，并提出了基于测试用例驱动的迭代调试框架。

**💡 创新点**

创新点是将LLM与粗暴基准生成、压力测试和迭代改进相结合，形成“DePro”方法，可在失败测试用例引导下自动修复错误。

**🔧 技术方法**

使用技术包括ChatGPT-5推理模型、暴力参考解生成、随机/边界测试用例生成以及循环调用LLM进行代码修改。

**📊 数据集**

数据集为Codeforces上13个错误提交（共100份用户代码）以及10个不同题目的样本。

**📈 对比分析**

通过与人类程序员和零射击LLM对比，实验显示DePro在尝试次数上平均降低64%，调试时间平均减少7.6分钟，整体性能优于两者。

**⚠️ 局限性**

局限性包括对正确暴力参考解和高质量测试用例生成的依赖，难以扩展到大规模输入空间或复杂约束的题目，并可能受LLM训练数据污染影响。

---

## 19. Probing to Refine: Reinforcement Distillation of LLMs via Explanatory Inversion

**arXiv ID:** 2603.19266 | [PDF](https://arxiv.org/pdf/2603.19266v1)

**作者:** Zhen Tan `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**通讯引用:** 48291 | [OpenAlex ID](https://openalex.org/A5100338946)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将大型语言模型的深度推理能力蒸馏到小型模型的框架，核心是通过生成“解释性探针”让学生模型更好理解推理逻辑，并结合强化学习进一步提升推理连贯性。

**💡 创新点**

创新点包括（1）解释性倒置（EI）技术：利用多种认知启发的转换规则生成多样化的探针，迫使模型解释推理过程而非机械记忆；（2）ExGRPO算法：在GRPO的基础上加入“对话结构效用奖励”，专门奖励学生在整个多轮探针对话中的连贯推理，从而提升知识迁移深度。

**🔧 技术方法**

技术方法：①数据增强（EI生成探针）→ ②监督微调（SFT）暖起→ ③基于策略梯度的强化学习（ExGRPO）并加入SFT辅助损失；使用多轮对话交互、奖励分组、对比学习以及KL正则化。

**📊 数据集**

使用12个多样化推理数据集：Commonsense（StrategyQA、CommonsenseQA、ARC-challenge）、Math（MATH、GSM8K）、Tabular（TabMWP）、NLI（ANLI）、逻辑推理（Date Understanding）以及四个OOD验证集（BoolQ、OpenbookQA、e-SNLI、GSM8K-Reversal）。

**📈 对比分析**

与零样本、传统KD（SKD、Distill Step‑by‑Step）、对话增强（RevThink、Divide‑or‑Conquer）等基线对比，实验显示在Gemma‑7B上平均提升20.39%（相较零样本），在Qwen‑7B上平均提升6.02%；在所有任务上都超过现有最优蒸馏方法，且在OOD任务和样本/token效率上也表现优异。

**⚠️ 局限性**

主要局限：①对教师模型的依赖（实验中多用Gemini‑1.5‑Pro，虽然对开源教师鲁棒，但仍受教师质量限制）；②生成探针和多轮对话的计算成本较高；③潜在的偏见和不准确信息在蒸馏过程中可能被传递和放大；④对极弱基模型的提升空间有限，需要进一步研究更高效的训练策略。

---

## 20. Exploring the Agentic Frontier of Verilog Code Generation

**arXiv ID:** 2603.19347 | [PDF](https://arxiv.org/pdf/2603.19347v1)

**作者:** Patrick Yubeaton `[一作]` (New York University), Siddharth Garg `[通讯]` (New York University)

**通讯引用:** 7302 | [OpenAlex ID](https://openalex.org/A5010950688)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Verilog 代码生成任务应用 LLM 代理框架，构建了开源、模型无关的 Verilog 代理，并在 CVDP 基准上系统评估其性能

**💡 创新点**

首次公开可复现的 Verilog 代理框架与结构化系统提示的有效性评估，证明结构化提示能弥补原生代理包装的性能下降

**🔧 技术方法**

采用 LLM 代理模式、工具调用（iverilog、vvp、Verilator、Yosys 等）以及多步提示与工具库扩展

**📊 数据集**

使用 Comprehensive Verilog Design Problems（CVDP）基准，其中包含可代理与不可代理子集的多难度任务

**📈 对比分析**

通过与单轮非代理基线对比，发现原始代理包装往往导致性能下降；但改进后的五步结构化提示可使性能与单轮基线相当或超越；闭源 LLM 在中难与困难任务上表现明显优于开源模型

**⚠️ 局限性**

主要局限在于：模型对 Verilog 逻辑推理仍不够准确，代理完整性无法保证；开源模型在工具反馈解释和 crash 率方面表现欠佳；仅增加工具库对性能提升有限

---

## 21. When both Grounding and not Grounding are Bad -- A Partially Grounded Encoding of Planning into SAT (Extended Version)

**arXiv ID:** 2603.19429 | [PDF](https://arxiv.org/pdf/2603.19429v1)

**作者:** João Filipe `[一作]` (University of Amsterdam), Gregor Behnke `[通讯]` (University of Amsterdam)

**通讯引用:** 975 | [OpenAlex ID](https://openalex.org/A5086696965)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了三种部分升高的SAT编码，保持动作完全升高而部分对状态进行定域化，使用可分解互斥组（PLMG）以及二进制对象编码；

**💡 创新点**

创新点在于通过部分定域化和二进制对象表示，将公式大小从对计划长度的二次增长降为线性增长，并在动作层保持升高以减少编码开销；

**🔧 技术方法**

使用SAT规划技术、统一参数化、PLMG、二进制对象编码以及谓词剪枝等手段；

**📊 数据集**

使用HTG-领域公开基准（Transport、Blocksworld、Childsnack、Visitall、Logistics、Pipesworld、Rover、Labyrinth、GED、OS等）进行实验；

**📈 对比分析**

与LiSAT、PWL、CPDDL、Madagascar、Fast Downward等现有升高或基准规划器比较，在线性规划长度下在5/9个基准域中击败LiSAT，满足规划时在部分域表现更优，整体覆盖率和得分与最先进方法相当；

**⚠️ 局限性**

局限性包括仅支持顺序计划、仅处理正预条件和增加效果、未考虑并行动作、二进制编码导致的语义约束复杂度以及对极大规模对象集合的实验验证不足。

---

## 22. Demonstration of Adapt4Me: An Uncertainty-Aware Authoring Environment for Personalizing Automatic Speech Recognition to Non-normative Speech

**arXiv ID:** 2603.20112 | [PDF](https://arxiv.org/pdf/2603.20112v1)

**作者:** Niclas Pokel `[一作]` (University of Zurich and ETH Zurich), Roman Böhringer `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个基于 Web 的、人机交互式的自动语音识别（ASR）个性化系统 Adapt4Me，帮助患有非规范性发音的人通过贝叶斯主动学习和低成本纠错实现模型个性化；

**💡 创新点**

创新点在于将模型不确定性可视化、贝叶斯主动学习、VI-LoRA 参数高效微调、贪婪双音覆盖冷启动、语义重链与音素难度评分以及低摩擦 top‑k 纠错界面结合，形成交互式数据采集而非被动收集；

**🔧 技术方法**

采用了贝叶斯主动学习、VI-LoRA 微调、贪婪双音覆盖、语义重链、音素难度评分、两通道解码、熵基不确定性热图、客户端-服务器架构和 SNR 预过滤等技术；

**📊 数据集**

评估使用了少量非规范性语音数据集（如 SAP、UA‑Speech、BF‑Sprache、TORGO 等）以及 Whisper Large 基线模型，并在真实家庭环境中收集用户语音；

**📈 对比分析**

与 Whisper Large 基线和全参数微调基线对比，Adapt4Me 在 75 分钟内将 WER 从约 70% 降至 25%，在使用更少数据的情况下性能优于全参数微调，并与临床语言治疗师评估高度相关；

**⚠️ 局限性**

局限性包括对用户主动纠错的依赖、对录音质量的敏感、数据量仍有限、在大幅语音变化时可能需要重新启动、需要云后端支持以及实时性能仍有提升空间。

---

## 23. When Contextual Inference Fails: Cancelability in Interactive Instruction Following

**arXiv ID:** 2603.19997 | [PDF](https://arxiv.org/pdf/2603.19997v1)

**作者:** Natalia Bila `[一作]` (Saarland University), Christof Monz `[通讯]` (University of Amsterdam)

**通讯引用:** 8276 | [OpenAlex ID](https://openalex.org/A5109059955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一个块搭建任务中，构建了一个交互式基准（Build What I Mean，BWIM），用于评估语言模型在上下文推理与字面解释之间的分离与应用；

**💡 创新点**

提出了将“可取消推理”转化为可交互、任务导向的评估框架，并揭示了模型在显式置信度评估与隐式信息请求之间的行为脱节；

**🔧 技术方法**

利用多轮交互的自然语言指令生成、反馈机制和问答成本控制，主要通过Prompting、系统指令设计与奖励惩罚策略；

**📊 数据集**

采用人工生成的块搭建指令集，包含完整指令、缺失颜色或数量的两类省略指令，涉及两类说话者（字面Lisa与语境Pia），并对比了人类实验数据；

**📈 对比分析**

将 Gemini 2.5 Pro、GPT‑5.1 与 Claude Opus 4.5 在置信度评估和问答两阶段实验中与人类基准进行对比；在置信度实验中模型对完整指令的准确率>0.95，未完成功能下表现良好；在问答实验中，GPT 仅问几乎不问且准确率73%；Gemini 与 Claude 提问频率高，准确率约86%/89%，但对说话者差异不敏感；

**⚠️ 局限性**

局限性包括：情境极度受限（单句或两句指令＋单一上下文），缺乏多样化的对话线索；问答成本固定，未模拟真实对话中变动或隐式成本；模型在行动层面对说话者可靠性适应不足。

---

## 24. On the Fundamental Limits of Hierarchical Secure Aggregation with Dropout and Collusion Resilience

**arXiv ID:** 2603.19705 | [PDF](https://arxiv.org/pdf/2603.19705v1)

**作者:** Zhou Li `[一作]` (Guangxi University), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

研究了在用户与中继失效以及协同限制下的信息理论层次安全聚合的通信极限，提出了两轮模型的正误与安全性保证；

**💡 创新点**

创新点在于给出了在多数链接上紧凑的通信成本闭合结果，并揭示了第二轮中继-服务器链路通信的上界与下界之间的间隙；

**🔧 技术方法**

采用了线性编码与MDS矩阵设计，构造了隐私友好的共享随机性与信息理论互信息不等式；

**📊 数据集**

没有使用具体的数据集，而是基于理论模型与符号长度分析；

**📈 对比分析**

通过与传统安全聚合方案对比，证明在U₀V₀>T的情况下第一轮通信率需≥1，第二轮用户符号率需≥1/(U₀V₀−T)，但在第二轮中继-服务器链路存在性能差距；

**⚠️ 局限性**

主要限制在于第二轮中继-服务器通信的上下界尚未闭合，且模型假设均匀用户分布与最坏情况失效，未考虑更复杂的失效与协同结构。

---

## 25. On Performance Guarantees for Federated Learning with Personalized Constraints

**arXiv ID:** 2603.19617 | [PDF](https://arxiv.org/pdf/2603.19617v1)

**作者:** Mohammadjavad Ebrahimi `[一作]` (Rutgers University), Farzad Yousefian `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究个性化约束联邦优化问题，提出 PC-FedAvg 方法，该方法让每个参与方维护跨估计的多块局部决策向量，仅对自身块投影，保证了约束隐私并实现个性化学习。

**💡 创新点**

创新点在于：①将全局一致性约束放宽为多块本地可行集，允许每个客户端维持自己合法解；②采用跨估计机制，在本地更新中同时更新所有块但仅在本块上施加罚项；③通过罚函数平衡全局目标与个性化约束，实现通信复杂度与传统 FL 方法相同的 𝒪(ε⁻²) 子最优收敛率，并在可行性误差上达到 𝒪(ε⁻¹)。

**🔧 技术方法**

技术手段包括：投影式多块本地梯度更新、随机梯度估计、罚函数正则化、分块平均聚合、理论上对 L_f‑光滑性、凸性及梯度方差的假设下给出收敛证明。

**📊 数据集**

实验数据集为 MNIST 与 CIFAR‑10，采用 10 类 softmax 回归模型，并对每个客户端施加不同的 ℓ₁‑范数稀疏约束。

**📈 对比分析**

与 SCAFFOLD、FedProx、FedAvg 三种基线（均改为惩罚型）对比：全局目标值基本相当，PC‑FedAvg 在约束满足（不可行性）方面明显优于基线；在不同步步长和局部更新次数下，PC‑FedAvg 的性能更为稳定、可行性误差更低。

**⚠️ 局限性**

局限性包括：①需要手动设定罚参数 ρ 和步长 γ，调参成本较高；②理论证明仅针对凸光滑问题，对非凸场景的收敛性未知；③实验仅在 MNIST/CIFAR‑10 等简单数据集上验证，缺乏对大规模多模态或高维约束的实证评估。

---

## 26. An Agentic Multi-Agent Architecture for Cybersecurity Risk Management

**arXiv ID:** 2603.20131 | [PDF](https://arxiv.org/pdf/2603.20131v1)

**作者:** Ravish Gupta `[一作]` (BigCommerce), Abhishek Aggarwal `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个六代理人工智能系统，用于快速完成小型组织的 NIST CSF 风险评估。

**💡 创新点**

创新点是将评估拆分为六个专业代理，并通过共享持久上下文实现跨阶段协同，解决单一模型连贯性不足的问题。

**🔧 技术方法**

使用本地部署的微调安全领域 LLM（saki007ster/CybersecurityRiskAnalyst），并在 Python 调度层中实现代理流程、向量检索与 JSON 交互。

**📊 数据集**

使用合成行业案例（医疗、金融、制造、零售、SaaS）以及真实的 15 人 HIPAA 合规公司问卷，测试模型。

**📈 对比分析**

对比基线 Mistral-7B 与微调模型，在单代理模式下进行 30 次跑，微调模型准确捕捉行业特定威胁，单机评估完成率 100%，而六代理流水线在低端 GPU 上完成率为 0%。

**⚠️ 局限性**

局限性包括对 GPU 上下文窗口依赖、结果可变性、仅基于问卷无法发现现场硬件问题、以及潜在的误导性输入风险。

---

## 27. AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search

**arXiv ID:** 2603.20014 | [PDF](https://arxiv.org/pdf/2603.20014v1)

**作者:** Yun Chen `[一作]` (Alibaba Group), Xiaoyi Zeng `[通讯]` (Alibaba Group)

**通讯引用:** 671 | [OpenAlex ID](https://openalex.org/A5082008486)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于“集成解耦”理论的神经架构搜索框架，利用单模型训练即可判断候选架构是否能在全量集成上提升性能，从而将候选验证成本从O(M)降到O(1)。

**💡 创新点**

核心创新是推导出在同质性假设下的单调提升条件（Monotonic Improvement Condition），为集成NAS提供了理论保证，并将搜索成本与部署成本解耦，显著提高工业级系统的搜索效率。

**🔧 技术方法**

使用了集成理论（误差-多样性分解）、数学优化（闭式解、约束可微优化）和LLM驱动的迭代接受算法；同时引入双代理训练、相关性与方差估计等技术。

**📊 数据集**

计划在Criteo、Avazu以及NAS‑Bench‑201等公开数据集上进行验证，重点考察CTR预测和离散架构搜索的实验。

**📈 对比分析**

通过与传统全量验证、零成本代理、参数共享等方法对比，实验预期可在相同预算下实现约90×的候选搜索量提升，同时在集成误差上实现U形曲线的最优点，验证单调条件的有效性。

**⚠️ 局限性**

主要局限包括：同质性假设对异构集成的适用性有限；零成本代理的可靠性需要进一步校准；框架目前仅适用于均匀加权集成，扩展到加权集成和在线演化仍是待解决的问题。

---

## 28. LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis

**arXiv ID:** 2603.20176 | [PDF](https://arxiv.org/pdf/2603.20176v1)

**作者:** Stanislaw Szymanowicz `[一作]` (University of Oxford), Andrea Vedaldi `[通讯]` (University of Oxford)

**通讯引用:** 75358 | [OpenAlex ID](https://openalex.org/A5060511349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无显式三维重建的实时新视角合成模型，使用预训练的3D重建网络（VGGT）提取3D-aware特征，并通过编码-解码器结构直接生成新视角图像。

**💡 创新点**

创新点包括：① 利用3D-aware预训练特征作为编码器，显著提升NVS质量；② 设计无瓶颈的编码-解码器架构，并提供全注意力与交叉注意力两种实现，兼顾质量与实时性能；③ 通过混合13个多视角数据集训练，获得强泛化能力；④ 在确定性NVS基础上快速衍生扩散生成模型，处理遮挡和外推。

**🔧 技术方法**

主要技术包括Transformer解码器、Plücker射线图、全/交叉注意力机制、层归一化与线性投影、AdamW+Cosine学习率、FlashAttention、L2+感知损失以及端到端微调。编码器基于预训练VGGT的Transformer骨干。

**📊 数据集**

使用13个多视角数据集的混合训练集，包括 RealEstate10k、DL3DV、WildRGBD、CO3D 等；在测试时评估 RealEstate10k、DL3DV、CO3D 的标准拆分，且支持未标记相机姿态和单视角场景。

**📈 对比分析**

与先前SOTA LVSM、DepthSplat、AnySplat、Flare、3DGS等方法比较，本文模型在 RealEstate10k 2视角下 PSNR 提升约 +1.7 dB（最高 31.4 dB），在多源视角与未标记相机场景中同样保持领先；实现 512×512 分辨率 30+ FPS（最多9源视角）在单张 H100 GPU 上；交叉注意力版在保持 30 FPS 的同时可处理 9 张源视角，完整注意力版可达 56 FPS，但仅支持 6 张源视角。

**⚠️ 局限性**

局限性包括：① 作为确定性回归模型，难以在遮挡区域产生多样性，容易出现平均模糊；② 对大型混合数据集的依赖，训练成本高；③ 仍需依赖预训练 VGGT 的3D-aware特征，对模型可解释性和动态场景适用性有限；④ 生成式扩展需要额外的扩散训练，且仅在像素空间；⑤ 目前对极端视角、低光照等情况的鲁棒性尚未充分验证。

---

## 29. Channel Prediction-Based Physical Layer Authentication under Consecutive Spoofing Attacks

**arXiv ID:** 2603.19962 | [PDF](https://arxiv.org/pdf/2603.19962v1)

**作者:** Yijia Guo `[一作]` (University of Liverpool), Yao-Win Peter Hong `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对连续造假攻击提出基于Transformer的信道预测物理层鉴权框架，利用预测的合法CSI进行鉴权并根据鉴权结果自适应更新输入，提升连续攻击下的鉴权鲁棒性。

**💡 创新点**

创新点在于首次将Transformer用于连续造假场景的信道演化预测，并引入鉴权自适应更新策略，显著缓解信道衰变导致的协方差失效问题。

**🔧 技术方法**

采用Transformer编码器-解码器结构进行CSI预测，配合Pearson相关系数鉴权和自适应输入更新机制；实验使用仿真Rayleigh信道、指数功率延迟分布。

**📊 数据集**

使用仿真生成的数据集：随机生成的RMS延迟{20,40,80,160,220}ns、SNR{5…20,50}dB、终端速度0.5–2 m/s、传输间隔3 ms，并构造连续造假长度N_a的攻击序列。

**📈 对比分析**

与传统无预测Pearson相关鉴权基准对比；实验表明，在攻击长度增大时基准准确率急剧下降，而本方法保持高准确率（尤其N_a>3），证明在连续攻击情境下性能优越。

**⚠️ 局限性**

局限性包括依赖仿真数据，缺乏真实环境验证；需要额外的模型训练与阈值设定，且Transformer推理算力相对较高。

---

## 30. BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection

**arXiv ID:** 2603.19635 | [PDF](https://arxiv.org/pdf/2603.19635v1)

**作者:** Zhengpei Hu `[一作]` (Qinghai University), Jianqiang Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关、结构感知的长文本压缩框架，将不规则文本序列划分为二维分页结构并压缩至可控长度，同时保持语义完整性。

**💡 创新点**

创新点在于将压缩过程从逐词剪枝转为层次化的页面选择，结合双路径池化编码、语义-词法混合评分与句子平滑，既实现硬件并行又避免语义崩塌。

**🔧 技术方法**

采用双路径池化（加权均值+最大池化）编码、基于上下文逆词频（ITF）的加权、语义-词法混合评分、结构先验（Anchor、Flow、Flash）以及句子边界平滑。

**📊 数据集**

在LongBench、ZeroSCROLLS、RULER、L‑Eval四大长文本基准上评测，并使用Qwen3系列嵌入模型进行实验。

**📈 对比分析**

与现有无监督统计方法、监督学习方法和学习型压缩方法对比，表现出与SOTA相当的QA性能（如单文档QA 40.7分），在RULER上单/多针检索精度近乎完美，压缩速度在128k上下文可实现26.4×的加速，显著优于LongLLMLingua等对比方法。

**⚠️ 局限性**

局限性包括：页面级别压缩不如细粒度词剪裁精准，可能保留冗余；对深层推理时语义重叠不足；需手动调优超参数，且在极端领域可能因嵌入模型偏差导致信息丢失。

---

## 31. Computer-Orchestrated Design of Algorithms: From Join Specification to Implementation

**arXiv ID:** 2603.19434 | [PDF](https://arxiv.org/pdf/2603.19434v1)

**作者:** Zeyuan Hu `[一作]` `[通讯]` (University of California), Zeyuan Hu (University of California)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并使用“Computer-Orchestrated Design of Algorithms (CoDA)”框架，对理论最优但实际可用的 TreeTracker Join 进行结构化测试与实现验证，最终发现并修复了多处逻辑‑物理翻译错误，并在此过程中完善了算法的理论预条件。

**💡 创新点**

创新点在于：①以算法为中心的结构感知测试框架，能够直接合成连接树与最小可复现案例(MRE)；②实现了逻辑到物理实现的双向反馈循环，既改进了实现，又完善了理论边界；③通过虚拟关系抽象解决了 bushy 计划映射的结构缺陷；④在小规模、可控的合成数据上实现了快速缺陷定位。

**🔧 技术方法**

主要技术包括：连接树生成器（随机树+属性继承）、查询计划合成器（左深/ bushy 变形）、差分评估（与 PostgreSQL oracle 对比）、状态管理与回跳机制、GYO 归约顺序验证、虚拟关系抽象、以及 MRE 合成与最小化。

**📊 数据集**

使用的数据集：1) 通过 CoDA 自生成的合成数据库，最大 5 个关系、10 条元组；2) TPC‑H 宏基准（13 组多连接查询）用于对比验证；3) PostgreSQL v13 作为真值 oracle。

**📈 对比分析**

比较方法：在 TPC‑H 13 组查询上与宏基准相对照，CoDA 发现 28 个失败案例并归纳为 10 个 MRE；相较于宏基准，CoDA 能在极小规模案例中定位缺陷并提供结构化诊断，明显提升了缺陷发现效率；性能方面未给出具体数值，但强调了早期验证的重要性。

**⚠️ 局限性**

局限性：①仅关注基于连接树的多连接算法，未覆盖更广泛的 DBMS 组件；②依赖合成数据，可能无法捕捉真实工作负载中的非结构性缺陷；③测试规模被限制在 ≤5 关系、≤10 元组，无法评估大规模查询的行为；④未与成熟的数据库测试框架深度集成，缺乏完整的系统级验证。

---

## 32. How Well Does Generative Recommendation Generalize?

**arXiv ID:** 2603.19809 | [PDF](https://arxiv.org/pdf/2603.19809v1)

**作者:** Yijie Ding `[一作]` (Carnegie Mellon University), Yupeng Hou `[通讯]` (University of California San Diego)

**通讯引用:** 3842 | [OpenAlex ID](https://openalex.org/A5052347581)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个基于项目过渡模式的框架，对生成式推荐（语义 ID）与传统项目 ID 推荐模型在记忆与泛化能力上的表现进行系统比较，并进一步揭示两者的互补性。

**💡 创新点**

创新点包括：①提出了按 1‑hop、跨跳、对称、二阶对称等多类记忆/泛化模式对实例进行细粒度归类的框架；②将生成式模型的项目级泛化解释为 token 级记忆，并用 prefix‑n‑gram 记忆比率量化这一机制；③设计了基于最大 softmax 概率的无监督记忆指示器，用于自适应加权两种模型，实现记忆感知的动态集成。

**🔧 技术方法**

使用的技术主要有：序列推荐模型 TIGER（生成式语义 ID）与 SASRec（基于项目 ID 的自回归模型）；评估指标为 NDCG@10 / Recall@10；分析方法包括：token prefix‑n‑gram 记忆定义、子项级别记忆与项目级泛化的对应关系、跨跳泛化（transitivity、symmetry、2nd‑symmetry）、以及通过 MSP 指标构建自适应集成。

**📊 数据集**

实验数据集涵盖七个公开大规模推荐数据集：Amazon Sports、Beauty、Science、Music、Office（2023 版本）、Steam 以及 Yelp；均采用留最末拆分，统计稀疏度和平均序列长度。

**📈 对比分析**

比较方法：将测试实例按是否满足记忆（1‑hop 已出现）、多跳泛化（可推导）或未归类划分；在每一子集上计算 NDCG@10。结果显示 TIGER 在记忆子集性能比 SASRec 低 5–40%，但在泛化子集上提升 20–60%；在整体上，两者互补，基于记忆指示器的自适应集成在所有数据集上均比单模型和固定权重集成更优，提升 2–5% 的 NDCG。

**⚠️ 局限性**

限制与未来工作：①对 token 级记忆的定义受语义 ID 分词策略影响，可能无法覆盖所有泛化场景；②实验仅涉及两种模型，缺乏对更广泛生成式或混合式推荐器的验证；③多跳泛化定义假设训练集中存在足够多的过渡，实际数据中稀缺过渡可能导致误判；④自适应集成仅利用单一 MSP 指标，可能无法充分捕捉复杂的记忆/泛化关系。

---

## 33. Can LLMs Prove Robotic Path Planning Optimality? A Benchmark for Research-Level Algorithm Verification

**arXiv ID:** 2603.19464 | [PDF](https://arxiv.org/pdf/2603.19464v1)

**作者:** Zhengbang Yang `[一作]` (George Mason University), Zhuangdi Zhu `[通讯]` (George Mason University)

**通讯引用:** 1260 | [OpenAlex ID](https://openalex.org/A5079428801)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了第一个用于评估大型语言模型（LLM）在机器人路径规划近似比证明方面的基准，收集了34个研究级证明任务并对LLM生成的证明进行细粒度错误分析。

**💡 创新点**

创新点在于：①构建了跨11篇机器人论文的高质量证明数据集；②系统评估了不同信息注入（上下文引导、后验近似比、链式思考）对LLM证明质量的影响；③提出了错误分类与“首次错误位置”指标，为证明生成的诊断与改进提供了可操作性。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑5.2、Gemini 3 Pro、Qwen 3.5、Grok 4.1）的推理与链式思考；自动化提取管道（GPT‑5.2 作为代理）将原始论文转换为结构化 LaTeX 并抽取问题、算法、证明与关键引理；LLM‑as‑Judge 进行自动化评分；以及错误分析流水线。

**📊 数据集**

数据集：34 个机器人路径规划近似证明任务，来源于 11 篇同行评审论文；每个任务包含问题定义、算法描述、真实近似比与证明以及关键引理；数据通过 GPT‑5.2 生成的自动化提取流程得到，并已匿名化。

**📈 对比分析**

比较方法：对四个主流 LLM 进行四种信息设置（无上下文、仅上下文、仅后验比、两者均有），分别记录最终答案、推理、关联性得分与成功率；结果显示：最差无上下文下成功率仅 26.5%；添加上下文后提升到约 32–44%；最优设置（上下文+后验比）成功率最高可达 50%。

**⚠️ 局限性**

局限性：①LLM 仍显著依赖领域专用上下文，缺乏自发推理能力；②证明生成易出现逻辑错误与幻觉，尤其在缺乏明确约束检查时；③数据集规模有限（34 题），无法覆盖所有规划范式；④对多模态几何可视化与实时子命题验证的支持不足。

---

## 34. Embodied Science: Closing the Discovery Loop with Agentic Embodied AI

**arXiv ID:** 2603.19782 | [PDF](https://arxiv.org/pdf/2603.19782v1)

**作者:** Xiang Zhuang `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**通讯引用:** 8296 | [OpenAlex ID](https://openalex.org/A5102018239)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Embodied Science 的新范式，定义了 Agentic Embodied AI，并提出了 Perception–Language–Action–Discovery (PLAD) 闭环框架，阐述实现长期自主科学发现的技术路线与关键挑战。

**💡 创新点**

创新点在于将感知、语言推理、实验执行与发现四个环节统一为闭环；明确了 Embodied Science 与传统认知驱动/执行驱动的区别；并提供了 PLAD 作为实现长周期自主探索的通用架构和方法论。

**🔧 技术方法**

所用技术包括大语言模型与多模态适配、知识图谱与检索增强生成、工具调用与自动化推理、数字孪生仿真与 sim‑to‑real 迁移、标准化实验协议（如 Science Context Protocol）以及基于知识与模型的安全治理。

**📊 数据集**

数据来源主要为实验仪器产生的原始信号（光谱、显微图像、过程日志等）以及文献、数据库和结构化知识图谱；并未使用具体公开数据集进行实证验证。

**📈 对比分析**

由于本文为视角性综述，未开展系统实验或量化对比；通过对已有认知驱动和执行驱动系统的案例分析，论证 PLAD 能够弥合两者鸿沟，并在长期循环中实现知识累积和自我修正。

**⚠️ 局限性**

局限性包括：需要多种技术（感知、推理、执行、知识）高度耦合，实施难度大；感知到推理的实时性和可靠性尚未充分验证；安全与风险治理在实际实验中仍面临挑战；缺乏大规模、长期的实证验证来评估 PLAD 的真实效果。

---

## 35. Model-Driven Learning-Based Physical Layer Authentication for Mobile Wi-Fi Devices

**arXiv ID:** 2603.19972 | [PDF](https://arxiv.org/pdf/2603.19972v1)

**作者:** Yijia Guo `[一作]` (University of Liverpool), Stefano Tomasin `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对物理层身份认证，作者首先基于条件统计模型推导出理论最优的Neyman–Pearson检测器，并在此基础上设计了轻量级神经网络 LiteNP‑Net，用以在无法获取完整信道统计量时逼近该最优检测。

**💡 创新点**

创新点在于：①将理论检测器的数学结构直接映射到网络架构，形成可学习的嵌入网络，既保持了模型驱动的优势，又兼顾了深度学习的泛化能力；②通过实验和仿真验证 LiteNP‑Net 能在多种室内无线环境下逼近理论上限，显著优于传统相关性方法和现有 Siamese 网络。

**🔧 技术方法**

技术包括：信道状态信息（CSI）提取、条件高斯信道建模、Neyman–Pearson检测理论、基于线性 MLP 的嵌入网络设计、对抗损失（contrastive loss）与二元交叉熵训练、以及 TensorFlow/PyTorch 实现。

**📊 数据集**

数据集：①仿真数据，使用 IEEE 802.11n TGn 信道模型（B、F）并加入 AWGN；②实验数据，利用 ESP32 作为接收机、LoPy4 作为合法与攻击端，采集 Wi‑Fi CSI，覆盖四个室内场景（走廊、办公室、住宅楼层）。

**📈 对比分析**

比较方法：Pearson 相关、FCN‑Siamese、CNN‑Siamese、理论 NP 检测器。性能评估指标为 ROC、AUC、TPR/FPR，以及模型参数量/ FLOPs。实验与仿真均表明 LiteNP‑Net 的 AUC 接近理论上限，且在参数量和 FLOPs 上显著低于 Siamese 网络，检测误差率明显降低。

**⚠️ 局限性**

局限性：①网络假设训练与测试时的信道统计相似，环境变化导致的模型失配会影响泛化；②假设攻击者无法复制合法设备的 CSI，若出现更高级的仿真/探测攻击，鲁棒性需进一步验证。

---

## 36. When Prompt Optimization Becomes Jailbreaking: Adaptive Red-Teaming of Large Language Models

**arXiv ID:** 2603.19247 | [PDF](https://arxiv.org/pdf/2603.19247v1)

**作者:** Zafir Shamsi `[一作]` (Algoverse AI Research), Shivank Garg `[通讯]` (Algoverse AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自动化提示优化在LLM安全评估中的适用性，将黑盒优化器用于生成对抗性提示，并评估其对安全防护的削弱效果。

**💡 创新点**

将原本用于提升模型性能的提示优化技术转化为自适应红队测试，展示静态基准低估了残留风险。

**🔧 技术方法**

DSPy框架下的MIPROv2、GEPA、SIMBA提示优化器，以及GPT‑5.1作为独立危险评分判定器。

**📊 数据集**

HarmfulQA 与 JailbreakBench 作为种子提示集合。

**📈 对比分析**

通过对比基线提示与优化后提示的平均危险评分，发现所有模型的危险评分均显著上升，尤其开源模型如 Qwen‑3‑8B 从 0.09 升至 0.79；SIMBA 优化效果最强。

**⚠️ 局限性**

仅评估四个模型、有限的优化步骤与优化器，缺乏对更大规模模型、更多优化策略与安全评判者的广泛验证。

---

## 37. Beyond Words: Measuring User Experience through Speech Analysis in Voice User Interfaces

**arXiv ID:** 2603.19904 | [PDF](https://arxiv.org/pdf/2603.19904v1)

**作者:** Yong Ma `[一作]` (University of Bergen), Andreas Butz `[通讯]` (LMU Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过在49名受试者上进行三种语音助手人格（A、B、C）与三种任务情境（功能、创意、游戏）交叉设计的对内实验，实时记录用户语音并提取音频、时间、频谱、语音质量和语言学特征，随后与UEQ+等主观体验指标进行关联，并使用机器学习模型预测用户体验等级。

**💡 创新点**

创新点在于：①首次在自然交互场景下将语音的非词语（语调、节奏、噪声等）与用户体验直接关联并验证其可预测性；②提出可解释的、面向实时的语音特征提取流水线；③以三等级（正向、中性、负向）UX为目标，实现了从语音到体验的实时估计，为自适应语音交互提供依据。

**🔧 技术方法**

主要技术包括：OpenSMILE/ComParE和eGeMAPS音频特征提取；Librosa提取MFCC、谱质心等；Parselmouth（Praat）提取抖动、闪烁、HNR；Whisper自动语音识别；NLP工具（NLTK、spaCy）提取语言学指标；传统机器学习（KNN、SVM、RF、XGBoost）和深度学习（1D‑CNN）进行三分类。

**📊 数据集**

数据集：49名英语母语者在PC/手机上完成的9组任务，收集了高采样率16kHz 16bit的用户语音、系统事件日志、UEQ+、情绪/压力量表等，共计约数百个语音片段，构成公开可复现的实验素材。

**📈 对比分析**

方法对比：SVM、KNN、RF、XGBoost以及1D‑CNN，使用分层5折交叉验证。SVM与1D‑CNN在三类UX预测上均达成约76–80%的准确率，远高于三分之一的随机猜测；在每类F1分别为0.71（正向）、0.73（中性）、0.80（负向），证明语音特征具有可用的预测性。

**⚠️ 局限性**

局限性：①远程录音环境多变，噪声与麦克风差异导致音频特征波动；②人格设计同时改变多项系统参数，难以单独归因；③受试者为英美等少数英语母语者，跨语言或跨文化验证缺失；④仅使用传统特征与浅层模型，未尝试自监督深度表示或多模态融合，可能影响性能提升。

---

## 38. DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management

**arXiv ID:** 2603.19621 | [PDF](https://arxiv.org/pdf/2603.19621v1)

**作者:** Yaqi Xie `[一作]`, Yidong Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在库存管理中提出并实现了基于经典库存概念（如基准库存）的策略正则化方法，将深度强化学习与传统库存理论融合，提升策略可解释性和训练效率；

**💡 创新点**

创新点在于：①将基准库存结构直接嵌入DRL策略输出，形成“DeepStock”正则化；②证明正则化既不削弱策略表达能力，又显著加速收敛并降低超参调优成本；③在阿里巴巴电商平台实现了1,000,000+ SKU的统一库存决策系统；

**🔧 技术方法**

技术上采用深度确定性策略梯度（DDPG）、近端策略优化（PPO）以及可微模拟（DS）三种DRL方法，并对每种方法分别加入三种正则化（无正则化、基准库存、线性系数+基准库存）进行对比；

**📊 数据集**

使用的数据集包括：①合成数据（多种分布与长度）；②阿里巴巴离线历史轨迹（50,000 SKU×90天、5,000 SKU×90天、5,000 SKU×90天不同时间段）；③真实部署数据（10%国际SKU试点、100%国际SKU + 87%国内SKU、年度对比等）；

**📈 对比分析**

通过在合成与离线数据上对比验证，正则化后DDPG和PPO在有限超参搜索下的测试误差降低约20‑40%，并在离线评估中显著提升库存周转率和降低缺货率；在真实部署中，10%试点国际SKU实现缺货率下降0.83%且周转时间缩短9.5天；2025年全面部署后，国际SKU周转时间平均下降1天、国内SKU下降2天，缺货率保持不变；

**⚠️ 局限性**

局限性包括：①模型假设需求可预测且入库延迟确定，实际可能存在随机性；②缺货时需求需推断，可能引入误差；③正则化虽加速训练但仍需较多算力和超参搜索；④在样本不足时DS方法易过拟合；⑤仅在阿里巴巴特定场景验证，泛化性待进一步检验。

---

## 39. LSR: Linguistic Safety Robustness Benchmark for Low-Resource West African Languages

**arXiv ID:** 2603.19273 | [PDF](https://arxiv.org/pdf/2603.19273v1)

**作者:** Godwin Abuh Faruna `[一作]` `[通讯]` (Fagmart Lab), Godwin Abuh Faruna (Fagmart Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并实现了LSR（Linguistic Safety Robustness）基准，用于评估大型语言模型在西非语言中的拒绝行为是否保持与英文一致；

**💡 创新点**

创新点在于首次构建针对低资源语言的跨语言拒绝退化基准，并提出了Refusal Centroid Drift（RCD）指标量化安全表示的漂移；

**🔧 技术方法**

技术上采用双探针评估协议、Inspect AI评估框架、关键词式拒绝分类器以及Gemini 2.5 Flash模型进行实验；

**📊 数据集**

使用的数据集包含在Yoruba、Hausa、Igbo和Igala四种语言中各14条本土化攻击探针（共56条），并公开发布于Hugging Face；

**📈 对比分析**

通过对比英文与目标语言的拒绝率，发现Gemini 2.5 Flash在英文约90%拒绝率下降至35‑55%（Igala最低），对应RCD从0提升至0.35‑0.55，表明安全性能显著衰退；

**⚠️ 局限性**

局限性包括仅评估单一模型、探针数量有限、关键词分类器可能误判、RCD仅为拒绝率衰减的代理指标且缺乏对内部表示的直接测量，且文化覆盖度需进一步验证。

---

## 40. Enhancing Legal LLMs through Metadata-Enriched RAG Pipelines and Direct Preference Optimization

**arXiv ID:** 2603.19251 | [PDF](https://arxiv.org/pdf/2603.19251v1)

**作者:** Suyash Maniyar `[一作]` (University of Massachusetts), Rohith Reddy `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合元数据增强的分块检索与直接偏好优化（DPO）来降低法律问答中的幻觉，并提升检索质量与回答安全性。

**💡 创新点**

①通过元数据注入提升文档检索匹配率，②利用DPO实现上下文不足时精准拒答，解决提示式拒答过度或不足的问题。

**🔧 技术方法**

递归字符分块、元数据注入、本地窗口摘要、密集+稀疏混合检索（Dense+BM25）、FAISS向量索引、Direct Preference Optimization、LLaMA 3.2（1B）instruction‑tuned。

**📊 数据集**

PrivacyQA、MAUD、Australian Legal QA（澳大利亚法律QA）三大法律数据集。

**📈 对比分析**

采用Span Recall与Document Retrieval Mismatch（DRM）进行评估；元数据增强在MAUD上Span Recall提升320%，DRM下降84%；在PrivacyQA提升有限；在Australian Legal QA Span Recall提升约35–40%，DRM下降约20%；DPO将正确上下文拒答率从53%降至1.5%，错误上下文拒答率从87%升至99%，答案质量BERTScore F1提升5.4%。

**⚠️ 局限性**

元数据模式对不同文档结构的适应性有限，隐私政策等结构简易文档提升有限；DPO需要大量人工配对数据；模型规模受限于1B，较大模型可能更优；评估指标未覆盖所有幻觉类型。

---

## 41. Defusing Logic Bombs in Symbolic Execution with LLM-Generated Ghost Code

**arXiv ID:** 2603.19239 | [PDF](https://arxiv.org/pdf/2603.19239v1)

**作者:** Dimitrios Stamatios Bouras `[一作]` (Peking University), Sergey Mechtaev `[通讯]` (Peking University)

**通讯引用:** 1408 | [OpenAlex ID](https://openalex.org/A5011184280)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种结合大型语言模型（LLM）与 SMT 约束求解器的混合符号执行框架，利用 LLM 自动生成轻量级“幽灵代码”（ghost code），帮助 SMT 求解器突破传统符号执行的瓶颈。

**💡 创新点**

创新点在于：① 针对求解器难以处理的代码片段自动生成三种幽灵代码——逆向函数、求解器友好替代模型、语义堆结构分区；② 将 LLM 生成的逆向函数与 SMT 求解器通过双向约束传播（bidirectional constraint propagation）集成，保持全局约束一致性；③ 通过显著降低 LLM 令牌消耗（90–96% 减少）实现高效测试。

**🔧 技术方法**

技术手段包括：LLM 调用（GPT‑5、Claude‑3.7、DeepSeek‑Chat）生成幽灵代码；KLEE 作为基础符号执行引擎；Z3/ STP 作为 SMT 约束求解器；实现了逆向传播、代理模型与堆分区等算法；并在 KLEE 之上实现了 LLM 与求解器的协同工作。

**📊 数据集**

数据集：① LogicBombs（53 个合成程序，专门设计的求解器难题）；② FDLibM（78 个数学库入口点，包含复杂浮点与非线性计算）；③ 三个真实结构化输入程序：libexpat、jq、bc，分别包含 11k、3k、8k 行可执行代码。

**📈 对比分析**

与传统符号执行（KLEE-STP、KLEE-Z3、KLEE‑Float‑Z3）以及 LLM 驱动的 ConcoLLMic、Cottontail 进行对比。实验显示：在所有基准上均实现 52–84% 的覆盖提升（相较传统符号执行）以及 86–419% 的提升（相较 LLM 方案），同时 LLM 令牌使用量平均减少 90–96%。

**⚠️ 局限性**

局限性：① 需要对目标程序进行重新编译与注入幽灵代码，集成复杂；② 依赖通用 LLM，可能出现偏差或不完整的逆向/代理模型；③ 逆向传播算法为启发式优化，未保证收敛；④ 对全局深度耦合或环境交互的程序难以完全覆盖，主要优势集中在局部求解器难题上。

---

## 42. RouterKGQA: Specialized--General Model Routing for Constraint-Aware Knowledge Graph Question Answering

**arXiv ID:** 2603.20017 | [PDF](https://arxiv.org/pdf/2603.20017v1)

**作者:** Bo Yuan `[一作]` (Institute of Computing and Intelligence), Min Zhang `[通讯]` (Institute of Computing and Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RouterKGQA 框架，结合小型专用模型生成约束感知路径并在必要时由大型模型修复路径，实现知识图问答。

**💡 创新点**

创新点是约束感知推理路径（CRP）和专用-通用模型协同路由，以及高效的基于关系的通用模型修复流程。

**🔧 技术方法**

使用 LoRA 微调的 Llama-2/3.1 作为专用模型，GPT‑4o‑mini 作为通用模型，SBERT 进行语义匹配，以及 SPARQL 执行。

**📊 数据集**

在 Freebase 上的 WebQuestionsSP 和 Complex WebQuestions 两个英文 benchmark 上进行评测。

**📈 对比分析**

相较于现有 Retrieval‑based、Agent‑based 及 LLM‑only 方法，RouterKGQA 在 WebQSP 和 CWQ 上 F1 提升约 4–5 分，Hits@1 维持或提升，同时平均 LLM 调用仅 1.15 次，成本远低于传统 Agent 方案。

**⚠️ 局限性**

限制包括仅支持平面实体/数值/字符串约束、路由仅基于路径可达性、依赖准确的实体链接和完整 KG，且在多语种或不同 KG 上的迁移性尚未验证。

---

## 43. Speed and impact of team science during urgent societal events

**arXiv ID:** 2603.19246 | [PDF](https://arxiv.org/pdf/2603.19246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 44. Utility-Guided Agent Orchestration for Efficient LLM Tool Use

**arXiv ID:** 2603.19896 | [PDF](https://arxiv.org/pdf/2603.19896v1)

**作者:** Boyan Liu `[一作]` (University of Science and Technology of China), Hongli Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 5103 | [OpenAlex ID](https://openalex.org/A5063184427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于效用的显式协调策略，用于控制LLM工具使用代理的多步行为

**💡 创新点**

将代理协调视为明确决策问题，并通过估计收益、步骤成本、不确定性和冗余的效用函数实现可解释、可调节的质量-成本权衡

**🔧 技术方法**

设计轻量级效用评分器和状态表示，结合固定工作流、ReAct、工具调用等多种动作，并在HotpotQA上进行实验与成本代理、冗余控制分析

**📊 数据集**

HotpotQA（200个开发集样本）

**📈 对比分析**

与直接回答、固定流程、ReAct等基线对比；在F1、token使用、时延等指标上，效用驱动策略在保持可解释性与控制性的同时取得与ReAct相近的质量，并在token使用和冗余方面更优

**⚠️ 局限性**

效用成分为启发式未学习，未必优于更强的自由形式基线；冗余控制对时延影响有限；缺乏学习化的自适应评分器和对更复杂任务的验证

---

## 45. From Precise to Random: A Systematic Differential Fault Analysis of the Lightweight Block Cipher Lilliput

**arXiv ID:** 2603.19781 | [PDF](https://arxiv.org/pdf/2603.19781v1)

**作者:** Peipei Xie `[一作]` (Hubei University), Xiangyong Zeng `[通讯]` (Hubei University)

**通讯引用:** 2764 | [OpenAlex ID](https://openalex.org/A5029996403)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文对一种轻量级块密码（基于EGFN结构的轻量级密码）在三种逐渐弱化的故障注入模型下进行系统的差分故障分析（DFA），证明其在实际物理攻击场景下存在显著脆弱性；

**💡 创新点**

创新点在于提出并实现了从多轮固定位置到单轮固定位置再到单轮随机位置（固定范围）的三层故障模型，系统评估了不同攻击假设下的故障抗性；

**🔧 技术方法**

使用的技术包括差分故障分析、S盒差分分布表（DDT）约束、候选集交叉与滤波，以及实验验证（大量随机注入实验）；

**📊 数据集**

使用的“数据集”为对同一明文进行多次加密并注入不同数量与位置的故障，收集正确与错误密文对；

**📈 对比分析**

与传统基于统计差分/线性/积分/回旋攻击的理论安全性评估相比，本文通过实验展示在八次或十次注入故障即可以超过95%或99.5%的成功率恢复主密钥；

**⚠️ 局限性**

局限性包括：仅考虑了特定的4比特分支故障，未覆盖更通用的多位/多分支注入；实验基于理想化的故障模型，实际硬件实现可能受限于注入精度与功耗；

---

## 46. LiteAtt: Secure and Seamless IoT Services Using TinyML-based Self-Attestation as a Primitive

**arXiv ID:** 2603.19727 | [PDF](https://arxiv.org/pdf/2603.19727v1)

**作者:** Varun Kohli `[一作]` (Institute of Infocomm Research, Agency for Science, Technology and Research), Biplab Sikdar `[通讯]` (National University of Singapore)

**通讯引用:** 12371 | [OpenAlex ID](https://openalex.org/A5041189303)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 LiteAtt，一个基于 TinyML 自我鉴权的无验证器 IoT 固件完整性检测框架。

**💡 创新点**

将 SRAM 低成本特征与 int8 量化 TinyML Autoencoder 结合，并在 Arm TrustZone TEE 内执行，实现去中心化自我鉴权同时保护 SRAM 隐私。

**🔧 技术方法**

使用 TinyML 量化 Autoencoder、Arm TrustZone TEE、AES‑CBC+HMAC‑SHA256、四步互相认证握手及自适应阈值统计。

**📊 数据集**

采用多种 Arduino MCU 固件的 SRAM 采样数据集，包括 23 种固件共 10k 正常样本与 136 万恶意样本，涵盖单节点、四节点、六节点及复杂应用。

**📈 对比分析**

与传统软件/硬件/混合鉴权方法对比，LiteAtt 在所有固件上平均准确率 98.7%，TPR 98.72%，TNR 97.45%，F1 99.33%，latency 1.29 ms，能耗 42.79 µJ，内存占用 32 KB，显著优于 SMARM、FlashAttest 等方案。

**⚠️ 局限性**

依赖 TEE 安全性，未考虑侧信道/物理攻击；TinyML 模型仅覆盖 data 段，未利用堆/栈；安全性为统计型而非完全密码学证明。

---

## 47. Politicized Attention Shifts Amplify Polarization in the Information Ecosystem during California Wildfires

**arXiv ID:** 2603.19536 | [PDF](https://arxiv.org/pdf/2603.19536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 48. Learning Hierarchical Orthogonal Prototypes for Generalized Few-Shot 3D Point Cloud Segmentation

**arXiv ID:** 2603.19788 | [PDF](https://arxiv.org/pdf/2603.19788v1)

**作者:** Yifei Zhao `[一作]` (Fudan University), Yinsheng Li `[通讯]` (Fudan University)

**通讯引用:** 1708 | [OpenAlex ID](https://openalex.org/A5101658474)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种面向通用少样本3D点云分割的框架 HOP3D，能够在仅有少量标注的情况下学习新类别，同时保持对已有类别的高性能。

**💡 创新点**

创新点包括：① 在梯度层和表示层双重正交化（HOP-Grad 与 HOP-Rep），实现基类与新类更新的解耦与子空间正交化；② 引入基于条件与边缘熵的少样本正则化（HOP-Ent），提升预测置信度与类别平衡；③ 通过联合正交原则显著降低基类遗忘与新类泛化衰退。

**🔧 技术方法**

技术方案主要包括：梯度正交投影、正交原型子空间、双熵正则化、Point Transformer V3 作为骨干网络，整体训练分为基类预训练和少样本适配两阶段。

**📊 数据集**

使用的公开数据集为 ScanNet200（200类）和 ScanNet++（超过1000类）进行 57 类 / 30 类的少样本分割实验。

**📈 对比分析**

在 1-shot 与 5-shot 设置下，HOP3D 在 ScanNet200/++ 的 mIoU‑B、mIoU‑N、mIoU‑A 以及 Harmonic Mean 上均优于现有基线（如 GFS‑VL、COSeg、GW 等），在 5-shot 下 mIoU‑N 提升约 2–3%，H值提升 2–4%，体现出更好的基类保留与新类泛化平衡。

**⚠️ 局限性**

主要局限：依赖固定的梯度基向量与预设的伪标签策略，若基类梯度变化或伪标签质量下降会影响正交化效果；训练时加入梯度投影会略微增加 9–10% 的计算开销；模型在极少样本（1-shot）下对新类仍有提升空间。

---

## 49. DPxFin: Adaptive Differential Privacy for Anti-Money Laundering Detection via Reputation-Weighted Federated Learning

**arXiv ID:** 2603.19314 | [PDF](https://arxiv.org/pdf/2603.19314v1)

**作者:** Renuga Kanagavelu `[一作]` (A STAR IHPC), Qingsong Wei `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种名为 DPxFin 的基于声誉的自适应差分隐私框架，用于联邦学习中的反洗钱（AML）检测。

**💡 创新点**

创新点在于：① 根据客户端更新与全局模型的欧氏距离动态计算声誉，并据此调整噪声倍数；② 采用声誉加权聚合以提升模型鲁棒性；③ 在保持隐私的同时实现了更优的精度-隐私权衡。

**🔧 技术方法**

使用的技术包括：联邦学习、差分隐私（DP‑SGD+高斯噪声）、声誉评分、SMOTE 处理类别不平衡、MLP 预测模型、Flower 框架、Opacus 库、PyTorch 深度学习框架。

**📊 数据集**

实验基于 IBM 合成金融交易数据集（HI_Small Transaction），约 5 百万条样本、5 千条正样本；采用 IID 与非 IID（Dirichlet）两种数据分布。

**📈 对比分析**

与 FedAvg（无隐私）和 DP‑FedAvg（固定噪声）对比，DPxFin 在非 IID 场景下准确率提升约 2‑3%，F1/精度/召回率均显著优于两者；TabLeak 攻击中攻击准确率从 92.9% 降至 58.5%，证明了隐私防护效果。

**⚠️ 局限性**

局限性包括：实验规模相对有限，主要针对表格型 AML 数据，未验证对多模态或更大规模联邦场景的适用性；声誉计算与噪声调整仍需进一步优化以降低计算开销和潜在误判。

---

## 50. Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification

**arXiv ID:** 2603.19329 | [PDF](https://arxiv.org/pdf/2603.19329v1)

**作者:** Zenan Li `[一作]` (ETH Zurich), Chi Jin `[通讯]` (Princeton Language and Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种层次化证明搜索框架，自动化 Lean 4 代码验证，将复杂的验证目标递归拆分为更简单的子目标再进行策略化证明；

**💡 创新点**

创新点在于统一的分解与完成策略、基于构造性与结构性两轴的分解评分（可用作训练奖励与推理排序），以及结合监督初始化与混合强化学习（GRPO+监督重放）的训练流程；

**🔧 技术方法**

使用 Qwen‑3 8B LLM 与 Lean 4 交互、operatorcount 与 quickcheck 等自定义评分工具、强化学习（GRPO）与监督重放、搜索+分解+完成循环；

**📊 数据集**

实验数据集包含三大 Lean 4 代码验证基准：Verina（189题）、Clever（161题）与 AlgoVeri（77题），共 427 任务；

**📈 对比分析**

与前沿推理模型（Claude‑Opus‑4.6、Gemini‑3‑Flash、GPT‑5.2/5.3、Deepseek‑V3.2）及大型神经推理器（Kimina‑Prover‑72B、DeepSeek‑Prover‑V2‑671B、Goedel‑Prover‑V2‑32B、BFS‑Prover‑V2‑32B）比较，结果显示 62.0% 的成功率，比最强基线高 2.6 倍，且 8 B 模型的规模远小于 32–671 B 的神经推理器；性能随搜索迭代与并行采样（pass@k）显著提升，表明可扩展性良好；

**⚠️ 局限性**

局限性：仅适用于已在 Lean 4 中正式化的程序与规范，无法自动处理非 Lean 代码或自然语言规范；分解评分仅基于符号算子计数，未充分考虑语义难度；对更大规模、跨过程依赖或复杂数据结构的验证任务仍需进一步研究。

---

## 51. IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment

**arXiv ID:** 2603.19862 | [PDF](https://arxiv.org/pdf/2603.19862v1)

**作者:** Simone Magistri `[一作]` (University of Florence), Andrew D. Bagdanov `[通讯]` (University of Florence)

**通讯引用:** 7027 | [OpenAlex ID](https://openalex.org/A5064029620)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析CLIP投影器导致的跨模态与同模态不对齐问题，提出IsoCLIP通过限制投影器到同模态对齐的等温子空间来提升同模态检索与分类性能。

**💡 创新点**

通过对交叉模态投影矩阵Ψ进行奇异值分解，识别等温中间子空间并在此子空间上投影，显著改善同模态对齐，且不需要额外训练。

**🔧 技术方法**

使用CLIP对比损失分析、奇异值分解(SVD)、投影器空间投影、语义子空间选择技术。

**📊 数据集**

在多种图像检索/分类数据集（Caltech, CUB, Oxford/Paris, Cars, Pets, Flowers, Aircraft, DTD, EuroSAT, Food101, SUN397, UCF101）和文本检索数据集（COCO, Flickr30k, nocaps）上验证。

**📈 对比分析**

与标准的同模态检索（Image‑Image, Text‑Text）以及基于文本/视觉反演的OTI/OVI方法对比，IsoCLIP在同模态任务上平均提升约4–6% mAP/精度，查询延迟几毫秒，且没有额外计算开销。

**⚠️ 局限性**

在跨模态任务上的性能略有下降，k_t/k_b参数选择仍依赖经验，未来需更系统的子空间选取方法。

---

## 52. Promoting Critical Thinking With Domain-Specific Generative AI Provocations

**arXiv ID:** 2603.19975 | [PDF](https://arxiv.org/pdf/2603.19975v1)

**作者:** Thomas Şerban von Davier `[一作]`, Sauvik Das `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2783 | [OpenAlex ID](https://openalex.org/A5006053551)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文通过设计并评估两个基于生成式AI的原型系统（ArtBot用于艺术解读，Privy用于AI隐私风险规划），探讨了如何通过领域特定的“挑衅”机制来促进用户的批判性思维；

**💡 创新点**

创新点在于将生成式AI的挑衅功能与领域知识紧密耦合，构造出“有意义的摩擦”与“用户驱动的内容门控”，从而将AI定位为批判性思维的协作者而非答案提供者；

**🔧 技术方法**

技术上主要采用本地部署的Llama‑3模型结合RAG技术实现ArtBot的对话交互，并在Privy中使用GPT‑4.1通过白板式的分支工作流嵌入生成式AI支持；

**📊 数据集**

数据集方面，ArtBot使用了精心挑选的艺术史元数据、策展文本和教育资料；Privy则基于已有的AI隐私分类框架和标准（如Lee et al.、Das et al.）构建提示；

**📈 对比分析**

比较方法通过对比非AI版本的任务完成度与用户反馈，在ArtBot中对13名参与者的解读反思进行评估，Privy中对12名从业者的隐私规划文档进行专家评分；结果显示域特定挑衅能显著提升用户的思考深度，但整体样本规模和对照组差异仍有限；

**⚠️ 局限性**

局限性包括：样本量偏小、仅覆盖两个领域、用户对AI功能的期望差异导致摩擦可能引发挫败感、缺乏长期使用和跨域推广的实证；未来需要更灵活的适配机制和更大规模的实验验证。

---

## 53. Factored Levenberg-Marquardt for Diffeomorphic Image Registration: An efficient optimizer for FireANTs

**arXiv ID:** 2603.19371 | [PDF](https://arxiv.org/pdf/2603.19371v1)

**作者:** Rohit Jena `[一作]` (University of Pennsylvania), James C. Gee `[通讯]` (University of Pennsylvania)

**通讯引用:** 53922 | [OpenAlex ID](https://openalex.org/A5005451810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出一种基于逐体素秩-1 Hessian分解的Levenberg–Marquardt优化器，改进了FireANTs的二次优化过程。

**💡 创新点**

创新点在于仅使用单个标量阻尼参数的自适应Levenberg–Marquardt，并通过逐体素rank-1近似实现内存友好且接近Adam的性能。

**🔧 技术方法**

采用GPU加速、Eulerian更新规则、信赖域阻尼调节、可选的Metropolis‑Hastings式拒绝步骤以及多尺度金字塔融合。

**📊 数据集**

在四个公开基准上测试：脑MRI（LUMIR、OASIS）、肺CT（NLST）和跨模态腹部MR→CT（Abdomen‑L2R）。

**📈 对比分析**

与FireANTs默认Adam相比，该优化器在脑MRI数据上略优、在肺CT略逊但可通过去除极端样本提升、在跨模态数据与Adam相当，并在大体素下显著降低24.6%内存同时保持或提升Dice/TRE。

**⚠️ 局限性**

局限性包括对阻尼增幅因子（必须 < 2.3）的敏感性、对极端噪声或稀疏模态下可能缺乏Adam动量的鲁棒性，以及需要根据任务手动开启/关闭拒绝机制。

---

## 54. Hyperagents

**arXiv ID:** 2603.19461 | [PDF](https://arxiv.org/pdf/2603.19461v1)

**作者:** Jenny Zhang `[一作]` (University of British Columbia), Tatiana Shavrina `[通讯]` (Meta)

**通讯引用:** 3541 | [OpenAlex ID](https://openalex.org/A5076593865)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了超代理（hyperagent）概念，并在达尔文-哥德尔机（DGM）基础上构建了DGM‑Hyperagent（DGM‑H），实现了在任何可计算任务上自我参考式、可持续的自我改进；

**💡 创新点**

创新点在于将任务代理与元代理合并为单一可修改程序，赋予系统元认知自我修改能力，使其能够改进自身的改进机制，突破传统固定元层限制，实现跨领域的开放式自我提升；

**🔧 技术方法**

技术上依托DGM的开放式进化框架、档案式存储、基于基础模型与外部工具的计算能力、性能追踪与持久记忆机制，以及Python可执行代码的自我编辑与评估循环；

**📊 数据集**

使用的数据集包括Polyglot编码基准、论文评审任务数据集、机器人奖励设计仿真任务，以及IMO‑GradingBench的奥数级数学评判数据；

**📈 对比分析**

与原始DGM、手工定制的DGM、DGM‑H去除自我改进或去除开放式探索等基线进行对比；在编码上DGM‑H达成与原DGM相当的提升；在论文评审和机器人奖励设计上分别提升至0.71和0.372，显著超越先前自我改进算法；在奥数评判上从0.56提升至0.60（全数据集），显示跨域迁移和累积提升；

**⚠️ 局限性**

局限性包括：任务分布固定，无法自我生成新任务；外部搜索循环（父选择、评估）保持固定，未能实现完全自我修改；安全与可解释性仍是挑战；计算资源消耗大，且在非可评估任务上效果不明。

---

## 55. Beyond Weighted Summation: Learnable Nonlinear Aggregation Functions for Robust Artificial Neurons

**arXiv ID:** 2603.19344 | [PDF](https://arxiv.org/pdf/2603.19344v1)

**作者:** Berke Deniz Bozyigit `[一作]` `[通讯]` (Independent Researcher), Berke Deniz Bozyigit (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在人工神经元中引入可学习的非线性聚合机制（F-Mean和Gaussian Support），并通过混合方式与传统加权求和融合，以提升网络在噪声环境下的鲁棒性和整体性能。

**💡 创新点**

创新点在于：①将聚合指数p作为可学习参数，形成可训练的子线性加权；②使用基于距离的Gaussian支持聚合为输入赋权；③设计混合神经元，让网络自适应选择线性与非线性聚合比例，从而兼顾可训练性与鲁棒性。

**🔧 技术方法**

技术包括：可微分聚合函数、softplus变换防止梯度消失、log-space参数化以保持正性、软最大/σ激活实现混合比例、梯度裁剪、学习率调度等。

**📊 数据集**

使用CIFAR-10数据集，并构造加性高斯噪声（σ_noise=0.15）版本进行鲁棒性测试。

**📈 对比分析**

比较方法：在MLP和CNN两种架构下，对四种聚合策略（线性、F-Mean混合、Gaussian混合、三重混合）进行训练，报告干净/噪声准确率以及鲁棒性比率ρ。结果显示：三重混合在噪声环境下鲁棒性最高（ρ≈0.99），F-Mean混合在干净数据上略有提升，所有混合策略在CNN中均优于基线。

**⚠️ 局限性**

局限性：仅在CIFAR-10及其噪声变体上验证；Gaussian支持聚合存在O(n²)计算复杂度；未在更大规模或不同任务（如ImageNet、NLP）上评估；缺乏理论鲁棒性或收敛性分析。

---

## 56. TRACE: Trajectory Recovery with State Propagation Diffusion for Urban Mobility

**arXiv ID:** 2603.19474 | [PDF](https://arxiv.org/pdf/2603.19474v1)

**作者:** Jinming Wang `[一作]` (University of Exeter), Man Luo `[通讯]` (University of Exeter)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5041350681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于扩散模型的轨迹恢复框架TRACE，利用状态传播扩散模型（SPDM）将稀疏GPS轨迹恢复为稠密连续轨迹。

**💡 创新点**

创新点在于将多尺度状态传播机制嵌入UNet架构，实现跨步记忆共享，显著提升对不规则时空分布轨迹段的恢复质量。

**🔧 技术方法**

采用扩散模型（DDPM/DDIM）、UNet结构、MCGRU时序门控单元以及多尺度多步训练算法进行实现。

**📊 数据集**

使用了西安、成都的出租车稠密轨迹数据集以及最后一公里物流快递员轨迹数据集。

**📈 对比分析**

与DeepMove、AttnMove、PriSTI、DiffTraj+RePaint等基线进行对比，TRACE在MSE、MAE、NDTW等指标上平均提升约26%且在不同稀疏度与轨迹长度下保持稳健。

**⚠️ 局限性**

主要限制是迭代式扩散需要多步推断导致推理时间较长，且模型对极端稀疏或缺失率高的数据仍可能出现误差。

---

## 57. PA2D-MORL: Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning

**arXiv ID:** 2603.19579 | [PDF](https://arxiv.org/pdf/2603.19579v1)

**作者:** Tianmeng Hu `[一作]` (Central South University), Biao Luo `[通讯]` (Central South University)

**通讯引用:** 6279 | [OpenAlex ID](https://openalex.org/A5012004938)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种基于Pareto升降方向的多策略多目标强化学习方法PA2D-MORL，用于在连续动作空间中高质量逼近Pareto策略集。

**💡 创新点**

创新点在于用Pareto升降方向取代预测模型进行梯度方向确定，并结合分区贪婪随机化策略选择与Pareto自适应微调，三者共同提升近似质量与分布密度。

**🔧 技术方法**

方法采用加权和分解、策略梯度（PPO）、投影求解Pareto升降方向、分区贪婪随机选择与自适应微调等技术。

**📊 数据集**

实验使用七个改造后的MuJoCo机器人控制环境（包括Walker2d、Humanoid、HalfCheetah、Hopper、Ant、Swimmer等），每个环境含2-3个冲突目标。

**📈 对比分析**

与PGMORL、MOEA/D、PFA等基线对比，PA2D-MORL在所有环境的Hypervolume最高，稀疏度最小，结果更稳定，显著优于现有方法。

**⚠️ 局限性**

局限性在于对非凸目标空间的覆盖有限，依赖梯度下降可能陷入局部最优，且加权和分解在非凸区域难以全覆盖；未来需探索Tchebycheff或更强的非凸处理。

---

## 58. A comprehensive study of LLM-based argument classification: from Llama through DeepSeek to GPT-5.2

**arXiv ID:** 2603.19253 | [PDF](https://arxiv.org/pdf/2603.19253v1)

**作者:** Marcin Pietroń `[一作]` (AGH University of Krakow), Rafał Olszowski `[通讯]` (AGH University of Krakow)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5081668377)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型在论证挖掘中的自动化论证关系分类进行了系统评估，使用了多种提示策略并结合投票集成；

**💡 创新点**

创新点在于首次在多个主流论证数据集上大规模比较现代LLM（如GPT‑5.2、Llama 4、DeepSeek R1）的零样本性能，并提出了结合提示重述、链式推理和置信度加权投票的综合评估框架；

**🔧 技术方法**

采用的技术包括高级提示工程（Chain‑of‑Thought、提示重述与回复、置信度自评）、多提示投票策略（简单投票、置信度分裂投票、加权投票）以及对LLM输出的定量与定性误差分析；

**📊 数据集**

使用的数据集为公开的UKP Stab2018和Args.me两大论证语料库，涵盖多主题讨论与辩论帖；

**📈 对比分析**

通过在不同提示组合和投票策略下的零样本评测，GPT‑5.2在Args.me上达91.9%准确率、78.0%准确率于UKP，投票方法可提升2–8个百分点，证明LLM在论证分类任务中已达到或逼近现有最优水平；

**⚠️ 局限性**

限制主要包括模型对话层面推理不足、对主题偏见敏感、对隐含批评与对比结构识别失败、数据集标注边界模糊导致的误差评估偏差，以及LLM在多语境和跨主题推理上的普适性仍有限。

---

## 59. Unlabeled Multi-Robot Motion Planning with Improved Separation Trade-offs

**arXiv ID:** 2603.19502 | [PDF](https://arxiv.org/pdf/2603.19502v1)

**作者:** Tsuri Farhana `[一作]` (Ben Gurion University), Shalev Goldshtein `[通讯]` (Open University of Israel)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了两种针对无标签多机器人运动规划（MRMP）的多机器人路径规划算法，分别是改进的弱单调策略和称为 Exodus 的全局协同策略，能够在更高密度环境下求解单位圆机器人在多边形空间中的路径。

**💡 创新点**

创新点在于：① 通过引入重叠参数和“几乎独立目标”概念，将原先对机器人间最小距离和障碍物距离的要求从 4、√5 下降到更低的 ρ≈3.291、ω≈1.354 或 ρ=2、ω=3；② 设计了 Exodus 算法，在障碍物距离 3、机器人间距离 2 的情况下，以全局的“外移”方式打开通道，避免局部冲突；③ 给出了单调与弱单调策略下的理论极限，证明在 ω<1.614 无法实现单调方案，在 ω<1.354 无法实现弱单调方案。

**🔧 技术方法**

技术主要包括：几何最短路径求解、匈牙利算法求最优匹配、可视图构建、扩展路径分区与角平分线分割、机器人路径的相对运动分析与碰撞证明、以及对障碍物与机器人间距离约束的几何推导。

**📊 数据集**

该工作未使用具体公开数据集，而是在理论上构造了一系列最坏情况实例（如环形通道、窄通道、长条形多边形等）来证明下界与极限。

**📈 对比分析**

与现有方法（如 Solovey 等人、Banyassady 等人提出的单调/弱单调方案）相比，该算法在更小的距离约束下仍保持多项式时间复杂度，且在大部分参数设置下实现了常数因子逼近（O(OPT) 或 O(OPT+4m)）。实验性能未给出，但理论证明显示时间复杂度为 O(m^4+n^2m^2)（第一种算法）或 O(m^3+mn^2)（Exodus）。

**⚠️ 局限性**

局限性包括：① 仍需一定的障碍物距离（最小为 1.5/2 的下界）才能保证解存在；② Exodus 算法只适用于简单多边形（不含孔洞）且机器人间距离固定为 2；③ 对于极端密集场景（如机器人间距离 2，障碍物距离 3 的情况）算法的路径长度上界相对较大（OPT+4m²），在实际应用中可能导致效率低下。

---

## 60. SegVGGT: Joint 3D Reconstruction and Instance Segmentation from Multi-View Images

**arXiv ID:** 2603.19926 | [PDF](https://arxiv.org/pdf/2603.19926v1)

**作者:** Jinyuan Qu `[一作]` (Tsinghua University), Lei Zhang `[通讯]` (International Digital Economy Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一个端到端框架，既能从多视角RGB图像进行前向3D重建，也能同时完成3D实例分割。

**💡 创新点**

将可学习的object query嵌入视觉几何基变换器，并使用Jensen‑Shannon对齐模块解决注意力分散问题，实现无额外推理开销的实例引导。

**🔧 技术方法**

使用视觉几何基变换器（VGGT）、DINO特征提取、Transformer交叉注意力、JS对齐正则化、Hungarian匹配等技术。

**📊 数据集**

在ScanNetv2、ScanNet200和ScanNet++数据集上进行训练与评估。

**📈 对比分析**

与基于点云的最优方法和其他RGB‑only 方法对比，本文方法在ScanNetv2/200 上取得 state‑of‑the‑art mAP，且在ScanNet++ 上无微调仍表现领先。

**⚠️ 局限性**

仍受投影误差影响，尤其是严格 mAP 评估；对稀疏视角和极端遮挡的鲁棒性待进一步提升。

---

## 61. Target Concept Tuning Improves Extreme Weather Forecasting

**arXiv ID:** 2603.19325 | [PDF](https://arxiv.org/pdf/2603.19325v1)

**作者:** Shijie Ren `[一作]` (Renmin University of China), Jirong Wen `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 TaCT 的可解释概念门控微调框架，用于在极端天气事件（如台风）中针对性提升预报性能，同时保持常规天气预报的准确性。

**💡 创新点**

创新点在于：①利用稀疏自编码器自动发现与失败相关的内部概念；②采用连续反事实推理识别最重要的概念；③在微调时仅在这些概念激活时更新参数，实现精准、可解释的局部改进，避免传统方法的性能折衷。

**🔧 技术方法**

使用的技术包括稀疏自编码器（SAE）进行概念分解、连续反事实推理进行概念重要性评估、概念门控微调（类似 LoRA/Adapter 的参数增量）以及对 Baguan 预训练模型的微调。

**📊 数据集**

数据集：使用全球 ERA5 重分析数据进行基础训练，台风事件采用 IBTrACS 与 CMA 最佳轨迹数据作为极端事件样本进行微调与评估。

**📈 对比分析**

与基线 Baguan、LoRA、Adapter、LoREFT 进行对比，TaCT 在西太平洋、北大西洋和东太平洋等区域的 72 小时台风最低气压和最高风速预测误差分别下降 9.3% 与 4.8%，且对其他气象变量的误差几乎无影响，整体性能优于传统微调方法。

**⚠️ 局限性**

局限性包括：需要额外训练稀疏自编码器，增加一次性计算成本；微调效果高度依赖概念的质量；若概念选择不佳可能导致噪声累积，影响模型表现。

---

## 62. Cellular Automata based Resource Efficient Maximally Equidistributed Pseudo-Random Number Generators

**arXiv ID:** 2603.19656 | [PDF](https://arxiv.org/pdf/2603.19656v1)

**作者:** Bhuvaneswari A `[一作]` (National Institute of Technology), Kamalika Bhattacharjee `[通讯]` (Indian Institute of Engineering Science and Technology)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5020432080)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种轻量级的基于线性最大长度细胞自动机（CA）的组合伪随机数生成器（PRNG），并通过时间间隔技术实现最大等分布和极长周期。

**💡 创新点**

创新点在于：①仅使用两种规则（90、150）且最多只在两格使用规则150，从而实现硬件友好；②采用组合生成与时间间隔相结合的方法，使得原本缺乏等分布的线性CA生成器在保持极长周期的同时获得最大等分布；③系统性地探讨了不同组合、不同时间间隔对等分布的影响，并筛选出满足最大等分布且周期接近理论极限的配置。

**🔧 技术方法**

主要技术包括：线性最大长度CA、XOR组合、时间间隔（s ∈ [2,10]）、等分布分析（t,l)-equidistribution）、统计测试（Dieharder、TestU01 的 SmallCrush/BigCrush）以及速度评估。

**📊 数据集**

使用自生成的随机序列（1.5 GB 二进制文件）作为统计测试输入，没有采用外部公开数据集。

**📈 对比分析**

与 Mersenne Twister、WELL512a、WELL1024a、Tausworthe（组合）和 GFSR4 等经典线性 PRNG 进行速度和统计性能比较。结果显示，所提 PRNG 在周期、最大等分布以及大多数统计测试中与 Mersenne Twister 相当甚至更优；在速度上略快于 Mersenne Twister，但比未使用时间间隔的其它线性 PRNG 慢。

**⚠️ 局限性**

主要局限在于：①时间间隔会增加计算开销；②并非所有组合都能同时满足最大等分布和极长周期；③对于某些规模较小的 CA，仍会在统计测试中出现缺陷；未来需进一步提升周期与性能兼顾，并探索更高阶 CA 以消除剩余缺陷。

---

## 63. In-the-Wild Camouflage Attack on Vehicle Detectors through Controllable Image Editing

**arXiv ID:** 2603.19456 | [PDF](https://arxiv.org/pdf/2603.19456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. Capacity-Achieving BBT Polar Codes with Interleaver-Assisted BP Decoding

**arXiv ID:** 2603.19938 | [PDF](https://arxiv.org/pdf/2603.19938v1)

**作者:** Xinyuanmeng Yao `[一作]` (Ningbo University of Technology), Xiao Ma `[通讯]` (Sun Yat-sen University)

**通讯引用:** 22420 | [OpenAlex ID](https://openalex.org/A5100631920)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于二叉平衡树（BBT）的极化码，能够在任意码长下实现容量接近，且通过交织增强BP译码收敛和降低延迟。

**💡 创新点**

创新点包括：1）BBT信道变换的通用化，证明了其能够实现极化并达到容量；2）利用树结构高效估计权重谱并给出ML误帧率上下界；3）引入交织器改写BBT的正则图，构造交织BBT（IBBT）码并在子图上实现低延迟BP译码。

**🔧 技术方法**

主要技术包括：BBT信道变换、正则图和树图表示、极化理论证明、权重谱（WEF）估计与最小汉明权重递推、ML误帧率上下界推导、交织器与子图BP译码算法。

**📊 数据集**

实验数据基于BPSK-AWGN信道，采用多种码长（N=50、500、1000、2000、300、600等）和不同信息比特数，使用Monte Carlo仿真评估误帧率、迭代次数、层数和算术操作量。

**📈 对比分析**

与传统BBT极化码和标准BP/SC/SCL译码比较，IBBT+BP在相同误帧率下显著降低迭代次数和层数，达到接近SCL(8)的性能；同时在延迟与计算复杂度上比完整图BP更优，尤其在较长码长和中等SNR区表现突出。

**⚠️ 局限性**

局限性：1）交织设计与子图大小的选择仍需经验性调优；2）BP译码在极端低误帧率区仍受限于图的循环结构，未能完全逼近SCL性能；3）对极短码长时，BBT结构与传统极化码的优势不明显。

---

## 65. Low-Latency Stateful Stream Processing through Timely and Accurate Prefetching

**arXiv ID:** 2603.19890 | [PDF](https://arxiv.org/pdf/2603.19890v1)

**作者:** Eleni Zapridou `[一作]` (École Polytechnique Fédérale de Lausanne), Anastasia Ailamaki `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 9985 | [OpenAlex ID](https://openalex.org/A5070907021)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在流处理系统中提出 Keyed Prefetching，利用查询计划中提前可知的键信息，将状态访问提前到数据路径之外，以降低尾部延迟。

**💡 创新点**

创新点包括：基于查询计划的准确键预取、动态选择最合适的前瞻算子、以及使用事件时间统一管理预取和已访问状态的时间感知缓存。

**🔧 技术方法**

技术细节包括：集中式 Prefetching Controller、每个状态算子中的 Hint Extractor 与 Prefetching Manager、异步 I/O 线程池、Count‑Min Sketch 过滤热点键、以及基于时间戳的双链表+哈希表缓存实现。

**📊 数据集**

实验数据集为 NEXMark（Q13、Q18、Q19、Q20）和 Yahoo Streaming Benchmark（YSB），并对键热点分布与缓存大小做了系统性变更。

**📈 对比分析**

与 LRU/Clock 缓存和异步 I/O 基线对比，Keyed Prefetching 在 p999 延迟上可提升 1.34–11 倍，p50 接近甚至略低于异步 I/O，同时吞吐量提升 1.01–2×。

**⚠️ 局限性**

局限性在于需要前瞻算子能够提供准确键提示；在键分布突变或无可用前瞻算子时预取准确性下降，且对预取窗口和热点阈值的手工调优仍存在。

---

## 66. Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection

**arXiv ID:** 2603.19780 | [PDF](https://arxiv.org/pdf/2603.19780v1)

**作者:** Hantao Zheng `[一作]` (Hunan University), Hao Chen `[通讯]` (Hunan University)

**通讯引用:** 110616 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种弱监督视频异常检测框架DeSC，利用两个专门化的子流（时序敏感流和语义一致流）并通过协作推理融合预测。

**💡 创新点**

①识别并阐述了“灵敏度–稳定性”权衡问题；②通过解耦训练与协作推理方式，消除了统一框架中的梯度冲突；③使用高频捕捉与低频平滑相结合的两流模型，显著提升异常检测性能。

**🔧 技术方法**

时序卷积网络（TCN）、图形变换器（GT）与图卷积网络（GCN）；多模态高斯混合先验（GMP）约束；视觉-文本对齐损失与多实例学习（MIL）损失；推理阶段滑动窗口测试时增强（TTA）。

**📊 数据集**

UCF‑Crime（含13类异常的1900段监控视频）和XD‑Violence（4754段电影/网络视频，6类暴力事件）。

**📈 对比分析**

与视觉‑仅方法（如I3D）和视觉‑语言方法（如VadCLIP、STPrompt）进行对比。DeSC在UCF‑Crime上达89.37% AUC（比前沿提升1.29%），在XD‑Violence上达87.18% AP（比前沿提升2.22%）。

**⚠️ 局限性**

解耦设计导致训练成本和模型体积上升，难以直接用于边缘设备；未来需进行知识蒸馏与更轻量化的模型设计。

---

## 67. Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD

**arXiv ID:** 2603.20155 | [PDF](https://arxiv.org/pdf/2603.20155v1)

**作者:** Emiel Hoogeboom `[一作]`, Tim Salimans `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 D-MMD 算法，将离散扩散模型蒸馏成少步生成器，能够在极少的采样步骤下实现高质量生成。

**💡 创新点**

创新点包括：① 将 Moment Matching Distillation（MMD）推广到离散扩散，得到通用的 min‑max 形式；② 在离散域采用软概率匹配而非硬采样；③ 引入温度和 top‑p 蒸馏策略；④ 提出基于参考 LLM 梯度的 Gradient Moment 评价指标，用以衡量生成分布与真实分布的距离。

**🔧 技术方法**

使用的主要技术有：离散（掩码和均匀）扩散模型、D-MMD 训练框架（含辅助模型和对抗优化）、软概率损失、温度/ top‑p 采样调节、输入噪声投影、Gradient Moment 评价。

**📊 数据集**

实验数据集包括 CIFAR‑10（图像）和 Open Web Text（文本），并在块自回归扩散设置下进行验证。

**📈 对比分析**

与教师模型和现有蒸馏方法对比：图像上 D-MMD 在 32 步内实现 FID 3.7（教师 7.5），文本上 16 步即可达到 0.236 的 Gradient Moment（教师 0.236‑0.3）；在 FID 与 Gradient Moment 指标上均显著优于教师和 Di4C、SDTT 等对标方法。

**⚠️ 局限性**

局限性包括：① 需要精细的超参数调优和对抗优化稳定性；② 仍基于因式分布的生成器，需要通过熵压缩来捕捉相关性；③ 评估主要依赖自定义指标，可能无法全面覆盖生成质量；④ 目前实验集中在无条件或短文本/图像，未对长文本或大规模任务进行验证。

---

## 68. SurfaceXR: Fusing Smartwatch IMUs and Egocentric Hand Pose for Seamless Surface Interactions

**arXiv ID:** 2603.19529 | [PDF](https://arxiv.org/pdf/2603.19529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 69. Recognising BSL Fingerspelling in Continuous Signing Sequences

**arXiv ID:** 2603.19523 | [PDF](https://arxiv.org/pdf/2603.19523v1)

**作者:** Alyssa Chan `[一作]` (University of Oxford), Andrew Zisserman `[通讯]` (University of Oxford)

**通讯引用:** 259437 | [OpenAlex ID](https://openalex.org/A5057678172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个大规模的英国手语（BSL）连贯拼字数据集，并提出了一种基于Transformer的双手与唇部特征融合的拼字识别模型。

**💡 创新点**

创新点包括：①迭代式标注框架通过手势3D关键点与唇部动作共同提取，显著提升标注精度；②在识别网络中显式建模双手交互与唇部信息；③结合跨熵和CTC双重损失以及字母级掩码与重采样。

**🔧 技术方法**

使用了Transformer编码器、HAMER手部关键点估计、AUTO-AVSR唇部ResNet、CTC损失、交叉熵、数据增强、字母级掩码、重采样等技术。

**📊 数据集**

主要数据集为从BBC BOBSL视频中抽取的FS23K（23,074条带字母级标注的连续拼字实例，133K时段边界），以及使用Transpeller和CSLR进行迭代标注的伪标注。

**📈 对比分析**

与Transpeller相比，在CSLR测试集上CER从0.581下降到0.250，提升0.331；在完整训练集上CER进一步下降至0.249，平均类准确率提升至70.65%。

**⚠️ 局限性**

局限性是缺乏词汇或句子层面的语言模型，无法利用上下文纠正模糊预测，且对极少出现字母的识别仍有较高错误率。

---

## 70. GDEGAN: Gaussian Dynamic Equivariant Graph Attention Network for Ligand Binding Site Prediction

**arXiv ID:** 2603.19817 | [PDF](https://arxiv.org/pdf/2603.19817v1)

**作者:** Animesh `[一作]` (Indian Institute of Technology), Pralay Mitra `[通讯]` (Indian Institute of Technology)

**通讯引用:** 597 | [OpenAlex ID](https://openalex.org/A5069306071)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于高阶可变形图注意力的蛋白质-配体结合位点预测模型GDEGAN。

**💡 创新点**

创新点在于用高斯动态注意力(GDA)替代传统点积注意力，通过邻域方差自适应调整注意力权重，捕捉结合位点的空间与化学异质性；并结合ESM-2序列嵌入与高阶可变形特征实现E(3)-等变处理。

**🔧 技术方法**

技术主要包括GotenNet骨干、ESM-2预训练序列嵌入、球面谐波编码、可变形特征的Steerable注意力、可学习温度参数和辅助方向损失。

**📊 数据集**

使用COACH420、HOLO4k和PDBbind2020三个公开数据集进行训练与评估。

**📈 对比分析**

与EquiPocket、GotenNet及其他基线方法对比，GDEGAN在DCC成功率提升37-66%、DCA提升7-19%、失败率下降至3.2%，并在推理速度上比EquiPocket快19.5倍。

**⚠️ 局限性**

局限在于训练数据仅包含单配体、结构相对单一的结合位点，缺乏多配体多孔位点的多样性，且未直接预测结合亲和力，未来需要更丰富的数据和进一步的功能扩展。

---

## 71. Diffusion-Guided Semantic Consistency for Multimodal Heterogeneity

**arXiv ID:** 2603.19337 | [PDF](https://arxiv.org/pdf/2603.19337v1)

**作者:** Jing Liu `[一作]` (University of British Columbia), Victor C. M. Leung `[通讯]` (University of British Columbia)

**通讯引用:** 66292 | [OpenAlex ID](https://openalex.org/A5035919267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SemanticFL框架，利用预训练扩散模型的多层视觉与文本语义特征进行语义一致性指导，解决多模态联邦学习中的非IID漂移问题。

**💡 创新点**

创新点在于将冻结的Stable Diffusion模型的中间表示离线提取为共享语义空间，通过交叉模态对比学习和知识蒸馏实现客户端语义一致性，并仅在服务器端完成高成本计算，既保持隐私又显著降低客户端算力需求。

**🔧 技术方法**

使用Stable Diffusion、CLIP、交叉模态对比学习、知识蒸馏、FedAvg等联邦学习基础技术。

**📊 数据集**

使用CIFAR-10、CIFAR-100和TinyImageNet三个公开基准数据集。

**📈 对比分析**

在多种非IID场景下与FedAvg、FedProx、MOON、FedRCL、FedDisco、FedCDA、FedDifRC等基线对比，SemanticFL平均提升约5.49%（如CIFAR-10 88.94%），在极端非IID场景中仍保持领先。

**⚠️ 局限性**

局限在于需要强大服务器完成一次性特征提取；对超参数敏感度仍需进一步自适应；目前仅验证图像+文本模态，未扩展到视频或音频等更复杂多模态。

---

## 72. DynFlowDrive: Flow-Based Dynamic World Modeling for Autonomous Driving

**arXiv ID:** 2603.19675 | [PDF](https://arxiv.org/pdf/2603.19675v1)

**作者:** Xiaolu Liu `[一作]` (Zhejiang University), Jianke Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 7851 | [OpenAlex ID](https://openalex.org/A5062252650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DynFlowDrive，一种基于流的动态潜在世界模型，用来模拟不同驾驶轨迹下的场景演化，并通过稳定性感知多模态轨迹选择实现更安全、更准确的轨迹规划。

**💡 创新点**

创新点在于：①用 rectified flow 方式在潜在空间学习连续速度场，从而实现对轨迹条件下的连续状态演化；②引入稳定性感知的多模态选择策略，将流场的方向一致性作为评价指标，提升轨迹安全性；③将世界模型仅用于训练阶段，无需推理时额外计算，兼容现有端到端框架。

**🔧 技术方法**

核心技术包括：VAE 前置编码器提取稳定的潜在特征；Transformer‑based flow 模型预测速度场；Euler 离散积分模拟连续动力学；稳定性度量（角度差平均）用于轨迹评估；整体损失组合包含轨迹误差、评分、重建与流匹配。

**📊 数据集**

在公开基准 nuScenes（开放环）和 NavSim（闭环）上进行评测，分别使用 1000 场景的未来轨迹与模拟仿真数据。

**📈 对比分析**

与 SOTA 感知‑基和潜在世界模型相比，DynFlowDrive 在 nuScenes 上 L₂ 位移误差降低 0.4 m（≈20 %）并将碰撞率下降 26%；在 NavSim 上 PDMS 达到 88.7 %，超过 WoTE 与其他模型；在所有评测中不增加推理开销，保持与原始框架相同的 FPS。

**⚠️ 局限性**

局限性包括：①流模型仅在训练时使用，推理阶段依赖于轨迹评分，无法实时预测场景变化；②对极端稀有情况的鲁棒性尚未验证；③需要较大训练数据与计算资源；④未来需整合 VLM 及更复杂的多模态推理来提升语义理解。

---

## 73. Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL

**arXiv ID:** 2603.19470 | [PDF](https://arxiv.org/pdf/2603.19470v1)

**作者:** Chenlu Ye `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 25214 | [OpenAlex ID](https://openalex.org/A5100378779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Adaptive Layerwise Perturbation（ALP），在大语言模型强化学习中对每层 Transformer 隐藏状态注入可学习的高斯扰动，从而统一处理离线政策不匹配和训练‑推理不匹配，并使用单一重要比率优化策略。

**💡 创新点**

创新点在于：① 用层级可学习扰动一次性解决多源离线问题，避免传统方法多比率截断导致的偏差与收敛速度下降；② 通过理论证明扰动可控制 KL 散度、平滑损失曲率；③ 通过实验证明 ALP 在稳定性、探索效率和最终性能上均优于现有 MIS、Bypass 等基线。

**🔧 技术方法**

技术手段包括：随机 Gaussian 采样、可学习的扰动幅度、GRPO/ MIS/ Bypass 等重要比率聚合、令牌级与序列级比率、训练-推理比率统一、梯度裁剪与自适应 clip 高低阈值、AdamW 优化器、模型微调与多轮推理工具集成。

**📊 数据集**

使用的数据集：单轮推理任务采用 Guru RL-92k 与 OpenR1（数学推理）混合集；多轮工具集成推理（TIR）任务采用 Math3-5（SimpleRL、Deepscaler、rStar2-Agent）混合集；评估基准包括 Math500、Minerva Math、Olympiad Bench、AIME2024、AIME2025。

**📈 对比分析**

与 GRPO、Seq-MIS、Token-MIS、Seq-Bypass 等基线在同一训练管线下进行对比，使用平均@32（温度 1.0）准确率作为主要指标。单轮任务中 Token-ALP 取得 37.87 的平均准确率，Seq-ALP 36.83；多轮任务中 Seq-ALP 取得 50.53 的平均准确率；训练动态指标（梯度范数、熵、KL）显示 ALP 的稳定性更好；Pass@k 曲线表明 ALP 在多回合采样下探索效率最高。

**⚠️ 局限性**

局限性包括：① 仅在单/多轮推理任务上验证，缺乏在完全异步 RL 或大模型 MoE 环境下的实验；② 采用高斯扰动，未探究其他分布或自适应扰动策略；③ 层级扰动幅度需手动设置，缺少自动化分配策略；④ 对系统级量化、批处理差异的依赖仍存在，扰动只能在一定程度上缓解不匹配。

---

## 74. dinov3.seg: Open-Vocabulary Semantic Segmentation with DINOv3

**arXiv ID:** 2603.19531 | [PDF](https://arxiv.org/pdf/2603.19531v1)

**作者:** Saikat Dutta `[一作]` (IITB-Monash Research Academy), Hamid Rezatofighi `[通讯]` (Monash University)

**通讯引用:** 3298 | [OpenAlex ID](https://openalex.org/A5034608678)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于 DINOv3 的开词汇语义分割框架（DINOv3-OVSS），通过双阶段视觉特征和图像-文本相关性优化实现高精度像素级分类。

**💡 创新点**

创新点包括：1）结合全局与局部文本嵌入增强语义对齐；2）早期视觉特征细化与后期相关性细化两阶段改进；3）局部-全局滑动窗口推理策略，兼顾细节与全局语义。

**🔧 技术方法**

使用技术：DINOv3 视觉-语言模型、LiT 文字对齐、AnyUp 视觉特征细化、Swin Transformer 与类级注意力的双阶段细化、SAM 语义先验编码器、局部-全局聚合推理。

**📊 数据集**

数据集：COCO-Stuff 训练集，评估在 ADE20K（150/847 类）、Pascal Context（59/459 类）以及 Pascal VOC（20 类）等五大 OVSS 基准。

**📈 对比分析**

与多种 SOTA 方法（OVSeg、SAN、ODISE、FC-CLIP、CAT-Seg、ESCNet 等）对比，平均 mIoU 达到 50.44，分别在 A‑847 20.09、PC‑459 27.80、A‑150 42.19、PC‑59 64.27、PAS‑20 97.86，均超过对比模型。

**⚠️ 局限性**

局限性：1）对 DINOv3 结构高度依赖，迁移到其他 VLM 需要额外调优；2）推理时滑动窗口聚合虽提升精度但计算量和显存占用较大；3）在样本极少的未见类上仍难以完全突破现有性能。

---

## 75. Two-Time-Scale Learning Dynamics: A Population View of Neural Network Training

**arXiv ID:** 2603.19808 | [PDF](https://arxiv.org/pdf/2603.19808v1)

**作者:** Giacomo Borghi `[一作]` (Heriot-Watt University), Lorenzo Pareschi `[通讯]` (University of Ferrara)

**通讯引用:** 7243 | [OpenAlex ID](https://openalex.org/A5009401094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套连续时间、均匀大样本极限下的群体训练动力学框架，完整刻画了 Population‑Based Training（PBT）中的快速参数更新与慢速超参数演化两种尺度的相互作用；在此框架下推导出聚合密度的 Fokker‑Planck‑Kinetic PDE，随后通过两尺度分离得到仅包含超参数的选择‑突变方程，并给出了有效适应度（effective fitness）概念与收敛性质；

**💡 创新点**

创新点在于：① 用传播散度（propagation of chaos）与大样本极限将离散的多智能体 PBT 算法严谨地映射到连续 PDE；② 在强时间尺度分离下推导出基于 Gibbs 平衡的选择‑突变方程，揭示 PBT 与双层优化、复制‑突变模型的深层联系；③ 证明了在唯一最优适应度点与足够高选择压强下，群体均值指数收敛到最优超参数；④ 通过有效适应度估计展示了在模拟时可大幅降低计算成本并提升稳定性；

**🔧 技术方法**

主要技术手段包括：随机微分方程与 Langevin/SGD 动态建模；Fokker‑Planck 与 Kinetic PDE 形式化；大样本极限与 Wasserstein 距离的收敛分析；两尺度匹配与平均化（effective fitness 计算）；复制‑突变方程的半群与 Dobrushin 不等式；数值实验采用 Euler‑Maruyama 与采样均值方法；

**📊 数据集**

实验数据集：① 低维二次/ Himmelblau 目标函数（synthetic）；② OpenAI Gym CartPole 交互式强化学习环境；未使用大型公开数据集；

**📈 对比分析**

对比方法：将原始 PBT（离散迭代）与两尺度简化算法（直接采样 Gibbs 分布或时间平均适应度）进行对比；实验显示：随着样本量增大，系统趋于确定性；在两尺度极限下，超参数分布更接近理论解，收敛速度更快，计算成本显著下降；

**⚠️ 局限性**

局限性：① 仅考虑强时间尺度分离与无限样本假设；② 对模型合并、交叉、有限种群噪声等更复杂的演化机制未建模；③ 有限维非平稳训练情形（如学习率动态变化）仍待进一步分析；④ 理论证明多基于 Lipschitz 与凸性假设，实际网络的非凸损失可能违反；

---

## 76. Sense4HRI: A ROS 2 HRI Framework for Physiological Sensor Integration and Synchronized Logging

**arXiv ID:** 2603.19914 | [PDF](https://arxiv.org/pdf/2603.19914v1)

**作者:** Manuel Scheibl `[一作]` (Medical Assistance Systems), Britta Wrede `[通讯]` (Medical Assistance Systems)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了 Sense4HRI，一个可在 ROS 2 HRI 系统中集成多种生理传感器和解读的模块化框架。

**💡 创新点**

创新点在于提供统一的时间戳化原始生理数据消息接口、解码分离的设计以及与 ROS4HRI 兼容的多用户主题结构，支持多模态同步记录和后期融合。

**🔧 技术方法**

采用 ROS 2 消息机制、BLE 通讯、LabStreamingLayer、Web UI、决策树算法、心率变异性指标、面部表情识别等技术。

**📊 数据集**

使用 Polar Verity Sense PPG 和 Polar H10 ECG 传感器采集的实时数据作为示例，并结合 ROS4HRI 提供的摄像头图像流。

**📈 对比分析**

通过与面部表情识别结合的决策树实现了情绪状态估计，示例中未给出定量指标，但展示了多模态融合能比单一表情识别产生更细粒度的状态表示。

**⚠️ 局限性**

局限性包括：仅支持单用户映射、未覆盖更多传感器模态、同步精度依赖于设备时间戳、缺乏大规模实验验证和性能评估。

---

## 77. Incremental Live Programming via Shortcut Memoization

**arXiv ID:** 2603.19560 | [PDF](https://arxiv.org/pdf/2603.19560v1)

**作者:** Marisa Kirisame `[一作]` (University of Utah), Pavel Panchekha `[通讯]` (University of Utah)

**通讯引用:** 1052 | [OpenAlex ID](https://openalex.org/A5022031348)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了“快捷记忆（shortcut memoization）”技术，用于在实时编程环境中增量地重用先前执行的计算步骤，以降低程序重执行的延迟。

**💡 创新点**

创新点在于：
- 定义了基于rewrite规则的通用“快捷”规则合成方法，并用Agda给出形式化语义与正确性证明；
- 设计了一种层级化的快捷记忆学习与存储策略，只合成相邻相同层级的规则，控制快捷数目；
- 将快捷记忆集成到CEK抽象机器并采用离散化树（discrimination tree）+单子（monoid）解析与哈希加速规则匹配；
- 在实时编程系统Hazel中演示了此技术能在编辑轨迹上实现显著加速，并在单次执行（符号求导+简化）中也能获得加速。

**🔧 技术方法**

技术细节包括：
- 基于Rewrite规则的最广泛合成（rule composition）
- 采用CEK抽象机作为执行模型
- 快捷记忆层级化与剪枝策略
- 区分树（discrimination tree）索引
- 平衡二叉树 + 单子解析 + 单子哈希实现高效字符串/子树匹配
- 变量数目限制、右侧最多10个变量以控制匹配成本
- OCaml实现、编译为寄存器/栈机器以减少解释开销。

**📊 数据集**

使用了两类数据集：
- Hazel用户在完成8个功能编程练习时产生的编辑轨迹（共24条轨迹，程序长度从简单列表函数到快速排序、归并排序等），
- 随机生成的100–200节点符号表达式，用于符号求导和代数简化实验。

**📈 对比分析**

比较方法：在同一硬件（Intel i7-8700K, 32 GB RAM）上，分别跑未使用快捷记忆的基线解释器与使用快捷记忆的实现。测量每条程序状态的执行时间、累计加速比及内存占用。结果显示：
- 在Hazel轨迹上平均加速约10×（部分场景达到25×），但有约6%场景出现8×的慢速（主要是首次调用导致学习成本高）。
- 单次执行的符号求导+简化案例得到约3×加速。内存开销较大，部分函数可达100×，但未实现缓存淘汰策略。

**⚠️ 局限性**

限制与挑战：
- 内存消耗高且未实现淘汰策略，导致长期运行时存储量快速增长。
- 只在CEK抽象机上验证，其他语言/执行模型需要进一步适配。
- 主要依赖rewrite规则的线性性质（无重复变量），不适用于非线性规则。
- 对动态编辑小幅变化效果最好，重写大幅结构改变时学习成本可能抵消收益。
- 目前未支持等式理论（如加法交换律）导致快捷匹配范围有限。

---

## 78. K-GMRF: Kinetic Gauss-Markov Random Field for First-Principles Covariance Tracking on Lie Groups

**arXiv ID:** 2603.19601 | [PDF](https://arxiv.org/pdf/2603.19601v1)

**作者:** ZhiMing Li `[一作]` (Tianjin University), ZhiMing Li `[通讯]` (Tianjin University)

**通讯引用:** 6521 | [OpenAlex ID](https://openalex.org/A5025342059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于 Lie 群的二阶动力学 K-GMRF，用于在线、无训练的协方差矩阵跟踪，能够在噪声和遮挡下实现零相位滞后和惯性漂移。

**💡 创新点**

将协方差跟踪重构为受观测驱动的刚体运动，通过 Euler–Poincaré 方程导出自然梯度扭矩，并使用结构保持的辛积分实现零稳态误差和对非平稳旋转的完美跟踪。

**🔧 技术方法**

运用了 Lie 群几何、欧拉–庞加莱方程、辛积分（Kick‑Drift‑Measure）、Wishart 似然的自然梯度、Riemannian SPD/ SO(3) 测量、Lyapunov 稳定性分析以及与传统 EMA、Tangent KF 和 Alpha‑Beta 的对比实验。

**📊 数据集**

在合成 SPD(2) 椭圆跟踪、SO(3) 相机稳定化（含随机丢帧）以及真实 OTB 运动模糊序列（BlurBody、BlurCar1/2、BlurFace、CarScale、Jogging）上进行验证，使用 7×7 区域协方差描述子。

**📈 对比分析**

与 Riemannian EMA、Euclidean EMA、Tangent KF 和 Alpha‑Beta 进行对比；K‑GMRF 在 SPD(2) 上误差下降 30×、在 SO(3) 上在高丢帧率下保持 4–5×优势，并在 BlurCar2 序列中 IoU 提升 35%，显著优于第一阶或欧氏方法，并实现了零相位滞后。

**⚠️ 局限性**

目前不具备与深度跟踪器的 SOTA 对比，Cayley–Neumann 积分带来 O(Kd²) 开销且需在稳定域内调参；仅支持单一谱轨道，惯性张量仍需手工设定。

---

## 79. MAPLE: Metadata Augmented Private Language Evolution

**arXiv ID:** 2603.19258 | [PDF](https://arxiv.org/pdf/2603.19258v1)

**作者:** Eli Chien `[一作]` (National Taiwan University), Peter Kairouz `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为MAPLE的框架，用于在仅通过API访问大语言模型的情境下生成差分隐私合成文本；

**💡 创新点**

其创新点在于通过差分隐私表格元数据提取并结合少量上下文示例，显著改善Private Evolution（PE）算法的初始化，提升生成质量并降低API调用成本；

**🔧 技术方法**

技术实现包括：差分隐私表格元数据生成（使用AIM算法）、基于上下文学习的提示设计、以及对AugPE的迭代精炼；

**📊 数据集**

实验使用了两个专业领域数据集：bioRxiv（29k篇科研摘要）和OpenReview（8.4k篇ICLR 2023评审），并对比了AugPE与DP微调模型；

**📈 对比分析**

结果表明，MAPLE在所有隐私预算下均优于AugPE，在部分指标上甚至超越GPT‑3.5生成的合成数据，且仅需更少的迭代次数和API调用；

**⚠️ 局限性**

局限性包括：仍需少量公开或捐赠的示例进行上下文学习、元数据抽取方法依赖特定模式、以及在跨模态或无示例环境下的适用性尚未验证。

---

## 80. GeoChallenge: A Multi-Answer Multiple-Choice Benchmark for Geometric Reasoning with Diagrams

**arXiv ID:** 2603.19252 | [PDF](https://arxiv.org/pdf/2603.19252v1)

**作者:** Yushun Zhang `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 74753 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GeoChallenge-90K，一套包含90,279道多答案几何选择题的自动生成、符号验证与双语图文对齐的数据集；

**💡 创新点**

创新点包括自动化生成、正式注解、可调难度、多答案无猜测评估、双语一致性以及严格的图形与文本对齐；

**🔧 技术方法**

采用符号推理引擎AlphaGeometry、模板构造、图形渲染与人工视觉校验等技术；

**📊 数据集**

使用了GeoChallenge-90K数据集，并与Geometry3K、GeoQA、Geo170K等现有几何基准进行对比；

**📈 对比分析**

通过在多模态与文本模式下对比多种开源/闭源大型模型，评估指标为Exact Match、选项级Precision/Recall/F1、Hamming Accuracy；最佳模型GPT‑5‑nano在EM上达75.9%，仍低于人类94.7%；

**⚠️ 局限性**

局限性包括缺乏细粒度步骤级错误分析、未能因果定位视觉或推理错误、错误类型易受提示/解码策略影响。

---

## 81. Real-Time Structural Detection for Indoor Navigation from 3D LiDAR Using Bird's-Eye-View Images

**arXiv ID:** 2603.19830 | [PDF](https://arxiv.org/pdf/2603.19830v1)

**作者:** Guanliang Li `[一作]` (Universidad Politécnica de Madrid), Santiago Tapia Fernandez `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在资源受限的机器人平台上，设计并实现了一套基于3D激光雷达投影到二维鸟瞰图（BEV）的轻量级实时结构感知流水线，用以检测墙壁、柱子等语义结构元素。

**💡 创新点**

创新点包括：1）将三维点云压缩为二维BEV，显著降低数据量与运算；2）系统化比较经典几何方法（RANSAC、Hough、LSD）与深度学习检测（YOLO‑OBB）在同一流水线上的性能；3）提出时空融合与曼哈顿世界优化模块，使检测结果在时间维度上保持一致性与精度。

**🔧 技术方法**

核心技术包括：BEV投影、DBSCAN+RANSAC、概率霍夫变换、LSD线段检测、YOLOv8n‑OBB目标检测、局部与全局融合（Kalman/聚类）、曼哈顿世界约束优化；整个框架以ROS2实现，支持热替换节点。

**📊 数据集**

使用自制的合成BEV数据集（包含墙壁、柱子标注并注入真实激光噪声），并在四个真实室内场景（地下车库、细长走廊、实验室、教室走廊）收集的激光雷达点云进行验证；真实数据的地面真值通过超声波测距校准。

**📈 对比分析**

通过长度加权召回、精确率和F1以及几何误差（距离误差、角度误差）评估。YOLO‑OBB在所有场景中实现0.84–0.85的F1分数，召回率与精确率均高于几何方法；实时性能达10 Hz（端到端延迟<100 ms）且在Raspberry Pi 5上不需GPU即可运行；几何方法虽然速度快，但在复杂场景下召回率骤降，误差显著增加。

**⚠️ 局限性**

主要局限：1）合成数据与真实小尺度场景存在域差，导致YOLO对细薄墙壁召回不足；2）深度学习模型在CPU上仍比几何方法消耗更高的CPU负载；3）需要手工标注合成数据，扩展到多种结构仍有挑战；4）对极小或低分辨率激光雷达的适配性尚未充分验证。

---

## 82. Demonstrations, CoT, and Prompting: A Theoretical Analysis of ICL

**arXiv ID:** 2603.19611 | [PDF](https://arxiv.org/pdf/2603.19611v1)

**作者:** Xuhan Tong `[一作]`, Jiawei Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文建立了在轻微假设下的 In-Context Learning（ICL）理论框架，利用 Lipschitz 常数刻画演示选择、Chain‑of‑Thought（CoT）提示、模板与演示数量对 ICL 泛化误差的影响，并给出相应的上界。

**💡 创新点**

创新点在于：①将 ICL 误差分解为模型内在能力、演示路径的 Lipschitz 稳定性与分布偏移三项；②证明 CoT 通过把任务拆分为子任务可降低每个子任务的 Lipschitz 常数，从而提升泛化；③揭示当演示数量足够大时，正确或一致的提示模板对输出的敏感度呈指数衰减，而不一致的提示模板则不会消失。

**🔧 技术方法**

使用的技术主要是：近似理论工具（Bernstein 多项式、Remez‑Chebyshev 不等式）来近似非多项式的 ICL 损失；对 Lipschitz 常数的定义与上界推导；对演示选择、CoT 结构与提示模板的定量分析；以及基于隐含任务模型的概率推断与贝叶斯预测。

**📊 数据集**

实验数据集包括：加法任务（六至十位数）、体育项目识别、公司人员识别等人工合成数据；利用 Qwen‑30B、Qwen‑35B 等 LLM 作为实验模型。

**📈 对比分析**

与基线模型（无预训练、单一大提示、不同演示质量）对比，结果显示 ① 模型的内在 ICL 能力越强，未见提示下的准确率越高；② 识别性演示优于模糊演示；③ In‑Distribution 的 CoT 分解显著提升性能，Out‑Distribution 则更差；④ 正确或一致提示模板的敏感度随演示增多指数下降，而完全错误的提示模板敏感度持续甚至上升；整体实验验证了理论预言，精度提升可达数十个百分点。

**⚠️ 局限性**

局限性包括：假设预训练完美近似贝叶斯预测且模型可达到该理论最优；理论主要针对单向、固定长度的演示与模板，难以直接推广到开放式提示或更复杂的任务；实验多为人工合成任务，缺乏在自然语言真实任务上的验证；对模型规模与不同预训练数据分布的泛化能力未作系统评估。

---

## 83. Pitfalls in Evaluating Interpretability Agents

**arXiv ID:** 2603.20101 | [PDF](https://arxiv.org/pdf/2603.20101v1)

**作者:** Tal Haklay `[一作]` (Technion), Yonatan Belinkov `[通讯]` (Kempner Institute at Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种自主解释器智能体，自动设计实验并推理模型电路组件的功能；同时探讨了复制式评估的缺陷，并提出了基于功能可互换性的无监督内在评估方法。

**💡 创新点**

①揭示复制式评估（与人类专家结果对比）容易受到主观性、记忆化和过程忽视的影响；②提出通过交换头部权重并量化模型输出变化的“可互换性”距离，来客观衡量聚类质量。

**🔧 技术方法**

利用Claude Opus 4.1构建的研究智能体；使用TransformerLens实现的Logit Lens、Activation Patching、Attention Pattern工具；采用Jensen–Shannon距离与Silhouette分数评估聚类；以及GPT‑5作为判定者。

**📊 数据集**

在六个已有电路分析任务上测试：IOI、Greater‑Than、Acronyms、Colored Objects、Entity Tracking等，使用GPT‑2、Pythia‑160M、GPT‑2‑Medium、LLaMA‑7B等模型的电路组件。

**📈 对比分析**

对比了“自主型”智能体与“一次性”静态分析两种方案。实验显示两者在组件功能准确率和聚类准确率上相近，且自主型在实验设计与假设检验上更活跃；内在评估的Silhouette分数明显优于随机聚类，且与专家聚类存在正相关。

**⚠️ 局限性**

主要局限：复制式评估仍受专家结果主观与不完整性的影响；记忆化和推理难以完全分离；内在度量只针对注意力头，未覆盖所有模型组件；评估结果对模型噪声和随机性敏感，需进一步验证。

---

## 84. A Super Fast K-means for Indexing Vector Embeddings

**arXiv ID:** 2603.20009 | [PDF](https://arxiv.org/pdf/2603.20009v1)

**作者:** Leonardo Kuffo `[一作]` ( Centrum Wiskunde & Informatica), Peter Boncz `[通讯]` ( Centrum Wiskunde & Informatica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对高维向量嵌入的聚类任务，提出一种新型 k‑means 变体 SuperKMeans，用以加速 IVF 索引的构建。

**💡 创新点**

创新点在于将 GEMM（矩阵乘）与 ADSampling 的裁剪核交错执行，利用随机正交旋转、PDX 数据布局和动态 d’ 调整，显著减少距离计算量；同时引入基于检索召回的 Early Termination by Recall (ETR)。

**🔧 技术方法**

核心技术包括：GEMM + PRUNING 两阶段设计、Adaptive Sampling、随机正交旋转、PDX 列块布局、动态 d’ 调整、早停 ETR，以及 GPU 上的 cuBLAS + 自定义 CUDA 裁剪核。

**📊 数据集**

在 10 个高维向量嵌入数据集上评估，包括 OpenAI、Wiki、Cohere、arXiv、ImageNet、Cohere、OpenAI 等，以及经典向量数据集 Fashion‑MNIST、SIFT、GIST。

**📈 对比分析**

与 FAISS、Scikit‑Learn、cuVS 等 SOTA 实现比较，CPU 上 SuperKMeans 可实现 3–7 倍加速（k↑时更显著），GPU 上 2–4 倍加速；分层版进一步在 250 倍左右加速；同时保持与 FAISS 相当或更优的检索召回率。

**⚠️ 局限性**

局限性包括：预处理（随机旋转）和 d’ 调整仍在 CPU 上完成，GPU 版受此固定成本影响；分层聚类在召回上略有下降；对极大数据量时需要手动调参（如 batch 大小、d’ 初始比例）。

---

## 85. One Model, Two Minds: Task-Conditioned Reasoning for Unified Image Quality and Aesthetic Assessment

**arXiv ID:** 2603.19779 | [PDF](https://arxiv.org/pdf/2603.19779v1)

**作者:** Wen Yin `[一作]` (University of Electronic Science and Technology of China), Tao He `[通讯]` (Laboratory of Intelligent Collaborative Computing of UESTC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态大型语言模型框架TATAR，用来同时进行图像质量评估（IQA）和图像审美评估（IAA），通过任务感知的推理与奖励策略实现更好的性能。

**💡 创新点**

创新点在于：①识别并解决IQA与IAA在推理方式（快速/慢速）和优化目标（点值回归 vs 相对排名）上的不匹配；②设计了 fast–slow CoT 构建、两阶段 SFT+GRPO 训练以及异构奖励（IQA 的高斯得分奖励与 IAA 的 Thurstone 排名奖励）等三大关键模块；③实现了共享视觉-语言骨干与任务专属后训练的结合，证明统一模型可以达到或超过单任务专用模型。

**🔧 技术方法**

使用了视觉-语言大模型（以 Qwen-2.5-VL-7B-Instruct 为基座）、监督微调（SFT）、基于组相对策略优化（GRPO）的强化学习、Gaussian 得分塑形奖励、Thurstone 风格排名奖励以及快速-慢速 CoT 生成与判别器过滤。

**📊 数据集**

主要使用的公开数据集包括：IQA 方面的 KonIQ-10k、KADID-10k、SPAQ、PIPAL；IAA 方面的 ArtiMuse-10K、AVA、TAD66K、Flickr-AES；以及跨域测试所用的多种公开评估集。

**📈 对比分析**

通过对比 19 种模型（包括专用模型、开源 MLLM、统一评估基线 UniPercept）以及在 8 个评估集上的 SRCC/PLCC 结果，TATAR 在 IAA 上平均提升约 6.2/7.4 分，在 IQA 上提升约 4.9/4.0 分，整体表现接近或略低于最优秀的单任务专用模型，但在跨域和统一任务场景下具有更好的稳定性和泛化能力。

**⚠️ 局限性**

限制方面：①仅针对图像质量与审美，缺乏对视频或多模态感知任务的直接扩展；②对主观审美仍依赖于现有标注数据，可能受限于数据规模与偏差；③在极端样本或不同文化审美差异上仍可能存在评估偏差。

---

## 86. Maximizing mutual information between user-contexts and responses improve LLM personalization with no additional data

**arXiv ID:** 2603.19294 | [PDF](https://arxiv.org/pdf/2603.19294v1)

**作者:** Hyunji Nam `[一作]` (Stanford University), Natasha Jaques `[通讯]` (University of Washington)

**通讯引用:** 3475 | [OpenAlex ID](https://openalex.org/A5046953322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过最大化提示与模型响应之间的互信息，实现LLM的无监督自我改进，提升个性化与推理性能。

**💡 创新点**

提出MIPO（Mutual Information Preference Optimization）框架，利用对比数据增强与DPO实现无监督的互信息最大化，可同时提升个性化和通用推理。

**🔧 技术方法**

对比数据增强、Direct Preference Optimization（DPO）、InfoNCE、基于互信息的奖励、无监督训练。

**📊 数据集**

Community Alignment、PRISM、Multi‑Bench（用于个性化评估）；GSM8k、SVAMP、MMLU、ARC（用于数学与推理任务）。

**📈 对比分析**

与个性化提示、SFT、RLVR、RLAIF、MI‑PPO等基线对比；MIPO在个性化任务中提升3–40%获胜率，在数学/推理任务中提升1–4%（小模型显著）。

**⚠️ 局限性**

受限于自生成数据的质量，尤其对小模型的误差影响；负样本采样近似可能导致互信息估计偏差；未在更大规模模型或更复杂任务上验证。

---

## 87. SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia

**arXiv ID:** 2603.19931 | [PDF](https://arxiv.org/pdf/2603.19931v1)

**作者:** Zhixiang Lu `[一作]` (Xi'an Jiaotong-Liverpool University), Zhengyong Jiang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SAGE 框架，使用强化学习自动筛选高质量、文化贴合的训练数据，并利用 LoRA 对大语言模型进行高效微调，从而提升低资源东南亚语言的社区对话翻译质量。

**💡 创新点**

创新点在于将 Group Relative Policy Optimization (GRPO) 与专家定义的语义相似度奖励结合，专注于“right data”而非“大数据”，实现了能源友好且文化敏感的数据驱动微调流程。

**🔧 技术方法**

技术手段包括：RL（GRPO）与语义相似度奖励；低秩适配（LoRA）进行参数高效微调；大语言模型 Qwen-3-8B / Llama-3.1-8B / Gemma-3-9B；使用 LaBSE/类似句子编码器计算语义相似度。

**📊 数据集**

使用的数据集为：大规模噪声 Web 语料（约 50M 句对），每种语言 2k 对专家社区对话数据（用于奖励），ALT 基准数据集与 500 句/语言的测试集。

**📈 对比分析**

在 7 种低资源东南亚语言上用 BLEU‑4 与 COMET‑22 评测，SAGE 在大多数语言上超过闭源大型模型，提升约 9 BLEU 点；同时数据使用量降低 97.1%，能耗降低 95.2%，达到新的 SOTA。

**⚠️ 局限性**

局限性包括：仍需专家标注来生成奖励信号；RL 训练收敛速度和稳定性受限；目前主要针对社区对话类任务，未在其他领域验证；在极低资源（不足 2k 对）场景下仍有提升空间。

---

## 88. Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation

**arXiv ID:** 2603.20172 | [PDF](https://arxiv.org/pdf/2603.20172v1)

**作者:** Richard J. Young `[一作]` `[通讯]` (University of Nevada Las Vegas), Richard J. Young (University of Nevada Las Vegas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估链式思考（CoT）推理的可信度，并对同一批 10,276 条受影响的案例分别使用三种不同的分类器（正则表达式、regex+LLM 级联、Claude Sonnet 4 判别器）进行比较，揭示不同评判方法导致的可信度估计差异。

**💡 创新点**

提出可信度不是单一可测量量，而是由不同分类器对“是否提及并依赖提示”这一构造的不同操作所决定的；通过三种分类器在同一数据集上产生显著差异，呼吁报告敏感性范围而非单点数值，并揭示分类器差异可导致模型排名逆转。

**🔧 技术方法**

使用正则表达式匹配、LLM 判别器（GLM‑5、Kimi K2、Gemini 3 Flash）级联、Claude Sonnet 4 作为独立评判器；统计检验包括 McNemar、Cohen κ、Spearman ρ；开放权重推理模型的链式思考输出；提示干预生成受影响案例。

**📊 数据集**

包含 498 道题（MMLU 300 + GPQA Diamond 198）与 5 种提示类型（sycophancy, consistency, metadata, grader, unethical），共 10,276 个受影响案例；评估对象为 12 个开放权重推理模型（7B–1T 参数，9 个模型族）。

**📈 对比分析**

通过在同一 10,276 案例上计算三种分类器的可信度率，比较整体率、提示类型差异、kappa、McNemar 和模型排名相关性。结果显示整体差异高达 12.9% 点，提示类型差异可达 43.4% 点，排名相关性仅 ρ=0.67，表明分类器差异显著影响评估结果。

**⚠️ 局限性**

缺乏人工标注基准、仅评估文本输出而无法确认内部推理是否真正使用提示、只考虑三种自动分类器（未覆盖人类评注、因果干预等其他方法）、潜在的 LLM-as-judge 偏差，以及无法捕捉模型隐藏或暗式推理方式。

---

## 89. Nonlinear Flexibility Effects on Flight Dynamics of High-Aspect-Ratio Wings

**arXiv ID:** 2603.19725 | [PDF](https://arxiv.org/pdf/2603.19725v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Rafael Palacios `[通讯]` (Imperial College)

**通讯引用:** 3424 | [OpenAlex ID](https://openalex.org/A5007534365)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了一个几何完全弹性结构与无稳 strip‑理论气动、基于四元数的 6‑DOF 飞行动力学耦合框架，对高纵向比 HALE 机翼的柔性对平衡、漩涡、短周期与 Phugoid 模式、以及气流扰动响应的影响进行系统参数研究。

**💡 创新点**

创新点包括：
1) 将几何完全非线性梁模型、无稳 strip‑理论和四元数 6‑DOF 飞行动力学以单体耦合方式实现；
2) 引入非维刚度参数 σ，对翼刚度跨四个数量级进行参数化，首次系统揭示柔性对升力向量旋转、Phugoid 稳定性、flutter 边界与预应力效应的非线性耦合机制；
3) 在 flutter 分析中同时考虑预应力平衡状态，展示传统无预应力方法过于保守的结论。

**🔧 技术方法**

技术手段：
- 几何完全弹性梁理论（基于 Rodrigues 参数的旋转张量）；
- 无稳 strip‑理论气动（带增量态延迟项与 Küssner 函数处理气流扰动）；
- 四元数六自由度飞行动力学与惯性耦合；
- 单体耦合框架与隐式 Newmark‑β + 牛顿-拉夫森迭代；
- 线性化后利用 Arnoldi 方法求解广义特征值；
- 对不同 σ 进行静态平衡、flutter 与动力学模式分析。

**📊 数据集**

基准数据集为 Patil 等人提出的 HALE 模型：半弦 16 m、弦长 1 m、质量密度 0.75 kg/m、EI₂ 2×10⁴ N·m² 等。气象条件设定为 20 000 m 高度、U=25 m/s、ρ=0.0889 kg/m³。该数据集用于验证自然频率、静态弯曲、flutter 速度等。

**📈 对比分析**

比较方法：将数值结果与已发表的参考结果（Patil&Hodges、Murua 等、UVLM、CFD）进行对比；通过残差与误差百分比评估。性能表现：自然频率误差 <1%，静态弯曲曲线误差 <3%，flutter 速度误差 3–6%；参数化研究揭示柔性增大时 lift‑vector 旋转导致 trim α 上升、Phugoid 阻尼下降、flutter 边界随 σ^(-1/2) 缩小，但预应力校正可显著提升 flutter 边界。

**⚠️ 局限性**

局限性：
- strip‑理论在高攻角、三维流动或激波影响下精度下降，需 UVLM 或 CFD 校正；
- 研究仅考虑对称翼、无主动控制，未包含控制面或损伤影响；
- 软度参数 σ 变更时质量分布保持不变，实际设计中质量与刚度常共同变化；
- 未做实验验证，数值结果需在风洞或飞行试验中进一步确认；
- 仅针对 20 000 m、25 m/s 的特定 HALE 条件，结果对不同飞行速度或高度的推广需要进一步研究。

---

## 90. ShobdoSetu: A Data-Centric Framework for Bengali Long-Form Speech Recognition and Speaker Diarization

**arXiv ID:** 2603.19256 | [PDF](https://arxiv.org/pdf/2603.19256v1)

**作者:** Md. Nazmus Sakib `[一作]` (Bangladesh University of Engineering and Technology), H. M. Aktaruzzaman Mukdho `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对低资源孟加拉语长音频的自动语音识别和说话人分离，构建了数据驱动的训练流程并在竞赛中获第一。

**💡 创新点**

创新点在于结合LLM辅助语言归一化、模糊匹配边界校验与频谱降噪增广，并用全参数微调而非LoRA实现最佳性能。

**🔧 技术方法**

使用Whisper（tugstugi/whisper-med）、pyannote.audio、HTDemucs、LLM（Gemini 3 Flash）及自定义增广。

**📊 数据集**

使用DL Sprint 4.0提供的孟加拉语YouTube有声书和戏剧数据，共约21k样本；对比公共与私有测试集。

**📈 对比分析**

通过与基线模型（Whisper大模型、IndicWav2Vec等）的对比，WER从34.8%降至16.75%（比基线下降53%），DER从0.41降至0.19，排名第一/第七。

**⚠️ 局限性**

局限在于数据域单一、对OOV和噪声鲁棒性不足，增广过度导致对外域音频性能退化，且仅使用10条标注文件的说话人分离易过拟合。

---

## 91. Listen First, Then Answer: Timestamp-Grounded Speech Reasoning

**arXiv ID:** 2603.19468 | [PDF](https://arxiv.org/pdf/2603.19468v1)

**作者:** Jihoon Jeong `[一作]` (Mila-Quebec AI Institute), Cem Subakan `[通讯]` (Université Laval)

**通讯引用:** 1399 | [OpenAlex ID](https://openalex.org/A5023830739)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于时间戳的多模态推理框架，将大型音频‑语言模型（LALM）的推理步骤与输入音频的时域片段显式关联，以提升推理的可信度与可解释性。

**💡 创新点**

创新点包括：①构建大规模标注时间戳的语音语料，用于监督式时域对齐；②利用两阶段训练——先学习时间戳对齐，再通过强化学习（GRPO）在多步推理中加入“答案正确奖励”与“时间戳合理性奖励”；③将时间戳作为链式思考（CoT）的中间信号，既保证答案准确，又鼓励模型生成简洁、可验证的音频参考。

**🔧 技术方法**

技术手段主要有：超分词级时间戳标注（基于Whisper+DTW）、文本到时间戳的监督训练、Group Relative Policy Optimization（GRPO）强化学习、奖励设计（R_answer + R_tg）以及注意力机制分析。

**📊 数据集**

使用的数据集包括：①训练时域对齐的语料——LibriSpeech、CoVoST 2、MELD、multi‑speaker、YouTube8M（约26.8万例）；②推理阶段的问答数据——MELD、multi‑speaker 对话 QA、YouTube8M 语音 QA（约4.7万例）。

**📈 对比分析**

与现有的专有模型（Gemini 2.5 Flash、GPT‑4o Audio）以及开源模型（Audio Flamingo 3、SALMONN）和音频推理方法（Audio‑CoT、Audio‑Reasoner、Audio‑Thinker）对比，本文方法在四个语音基准（MMAU‑mini‑Speech、MMAR‑Speech、AIR‑Bench、MELD）上均取得最优或竞争性表现，显著提升了时间戳匹配 IoU、语音事件检测 F1 以及多步推理的区域探索、音频验证和一致性指标。

**⚠️ 局限性**

局限性：①目前仅在语音场景验证，未扩展到非语音音频事件；②依赖大量标注时间戳的训练数据，获取成本高；③强化学习训练过程较为复杂，需调参；④时间戳奖励机制可能在某些任务中过度压缩或过度关注时间信息，导致推理冗余或遗漏。

---

## 92. PrefPO: Pairwise Preference Prompt Optimization

**arXiv ID:** 2603.19311 | [PDF](https://arxiv.org/pdf/2603.19311v1)

**作者:** Rahul Singhal `[一作]` (Distyl AI), Karime Maamari `[通讯]` (Distyl AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现 PrefPO，一种基于 LLM 判别器的偏好式提示优化框架，可在无标签数据或有限标签情况下迭代生成更优提示。

**💡 创新点**

创新点包括：① 只需起始提示和自然语言标准，利用 LLM 生成偏好与反馈；② 通过 pairwise 比较而非绝对评分，减少过拟合与噪声；③ 引入提示卫生（hygiene）与 prompt hacking 评估，提升可维护性与鲁棒性；④ 设计了 PrefPO-Minimal 与 PrefPO-Elo 两种可插拔变体。

**🔧 技术方法**

采用的技术主要是 RLHF 思路下的 AI 反馈（pairwise preference）和 LLM 反思；使用 GPT-5 作为判别器与优化器；对 BBH、IFEval-Hard 进行迭代训练，支持有标签和无标签两种工作流。

**📊 数据集**

使用的数据集：BIG‑Bench Hard（9 个任务）和新构造的 IFEval‑Hard（148 个难例），并在 BBH 任务上做不同训练样本大小的缩放实验。

**📈 对比分析**

与 MIPROv2、GEPA、TextGrad 等 SOTA 方法进行对比；在 BBH 任务中 PrefPO 在 4/9 任务上排名第一，标签无关版在 2/9 任务上也表现突出；在 IFEval‑Hard 上达到 82.4%（strict pass）对比 TextGrad 的 84.5%，且在提示卫生、重复率、长度比和 prompt hacking 率方面均优于 TextGrad。

**⚠️ 局限性**

局限性包括：评估范围仅覆盖 BBH 与 IFEval‑Hard，未验证更广泛任务或多步系统；Elo 采样未显著提升；人类评估一致性低，LLM 判别仍有误判；缺乏更大规模的跨模型与跨任务验证。

---

## 93. Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas

**arXiv ID:** 2603.19453 | [PDF](https://arxiv.org/pdf/2603.19453v1)

**作者:** Víctor Gallego `[一作]` `[通讯]` (Komorebi AI Technologies), Víctor Gallego (Komorebi AI Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于大型语言模型的迭代式程序生成方法，用以在序列社会困境（Gathering和Cleanup）中实现多智能体协调策略。

**💡 创新点**

创新点在于将LLM视为程序合成器，利用密集社会指标作为反馈迭代优化，并首次揭示环境可变性攻击的双刃剑风险。

**🔧 技术方法**

采用LLM（Claude Sonnet 4.6、Gemini 3.1 Pro）进行代码生成与自我反思，配合AST安全校验、50步烟雾测试、N-agent自对弈评估，以及社交指标（效率、平等、可持续性、和平）回传。

**📊 数据集**

实验使用两款经典SSD游戏：二维网格Gathering和Cleanup，均设定10名智能体、1000步回合，评估5个随机种子。

**📈 对比分析**

对比了零射、稀疏奖励迭代、稠密奖励+社交指标迭代三种配置，并与Q‑learner、BFS启发式、GEPA元优化等基线对比；Gemini在稠密反馈下达成U≈4.59、E≈0.97的近最优结果，效率比Q‑learner高6倍，稠密反馈优于稀疏反馈。

**⚠️ 局限性**

主要局限包括实验规模有限、对更复杂环境的泛化尚未验证，以及环境接口的表达性导致易被LLM发现奖励黑客攻击，需进一步强化安全与隔离。

---

## 94. AIGQ: An End-to-End Hybrid Generative Architecture for E-commerce Query Recommendation

**arXiv ID:** 2603.19710 | [PDF](https://arxiv.org/pdf/2603.19710v1)

**作者:** Jingcao Xu `[一作]` (Taobao & Tmall Group of Alibaba), Haihong Tang `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在淘宝首页预检索查询推荐（HintQ）场景中，提出了AIGQ框架，实现了端到端生成式个性化查询建议。

**💡 创新点**

创新点在于兴趣感知列表监督微调（IL‑SFT）、细粒度列表强化学习（IL‑GRPO）以及结合AIGQ‑Direct与AIGQ‑Think的混合离线‑在线部署架构。

**🔧 技术方法**

采用大型语言模型（Qwen3‑30B‑A3B）进行微调，利用兴趣驱动标签重排序、提示压缩、链式推理生成、基于CTR的奖励信号和动态熵正则等技术。

**📊 数据集**

使用淘宝10月–11月的千级用户交互日志，包含约一百万条页面浏览，按天拆分训练/测试。

**📈 对比分析**

与传统两塔检索、零样本LLM和未改进的GRPO等基线比较，AIGQ在Query HR@30、Cate HR@30、CTR等指标上提升超过30%（如AIGQ‑Think_IL‑SFT+IL‑GRPO达0.4704的Query HR@30，远高于SOTA）。

**⚠️ 局限性**

局限在于模型推理时延仍高、对多模态上下文理解不足以及对实时用户兴趣的更新依赖离线重训练。

---

## 95. Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement

**arXiv ID:** 2603.19623 | [PDF](https://arxiv.org/pdf/2603.19623v1)

**作者:** Chunlei Zhang `[一作]` (University of Technology Sydney), Jian Zhang `[通讯]` (University of Technology Sydney)

**通讯引用:** 69375 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种统一的混合多模态图像配准网络 HRNet，能够在同一共享特征空间中一次性完成全局刚性与局部非刚性变形的非迭代粗细调优配准。

**💡 创新点**

创新点包括：① 共享骨干加模态专属 BN 的多尺度特征提取；② 交叉尺度解耦与自适应投影模块 CDAP，用于抑制私有信息泄漏；③ 结合 Mamba 状态空间的混合参数预测模块 HPPM，在同一通道实现刚性与非刚性联合预测，避免串行堆叠导致的误差累积。

**🔧 技术方法**

技术手段包括：模态专属 BatchNorm、交叉尺度解耦+门控+投影、Mamba 状态空间块、残差状态空间块、结构化正则化（交叉协方差去相关、基向量正交、跨尺度方向一致）、三阶段学习曲线、Reprojection 与 NCC 评价指标。

**📊 数据集**

使用了四个跨模态数据集：UAV RGB–TIR (TBBR)、RGB–NIR (RGB‑NIR Scene)、遥感 RGB–IR 与 RGB–SAR (MRSR)，图像尺寸统一为 256×256。

**📈 对比分析**

与 IHN、InMIRNet、RHWF、SCPNet、MCNet、MMRNet 等现有方法在刚性和非刚性任务上对比，HRNet 在所有四个模态对的 RE 降低 70–90%，NCC 提升 0.1–0.2，显著优于最强基线模型。

**⚠️ 局限性**

局限性在于仅针对二维平面配准，尚未直接推广到高维或三维场景；模型结构相对复杂，计算资源需求较高；在极端模态差异或噪声极大场景下仍可能出现漏检风险。

---

## 96. GT-Space: Enhancing Heterogeneous Collaborative Perception with Ground Truth Feature Space

**arXiv ID:** 2603.19308 | [PDF](https://arxiv.org/pdf/2603.19308v1)

**作者:** Wentao Wang `[一作]` (Sun Yat-sen University), Guang Tan `[通讯]` (Sun Yat-sen University)

**通讯引用:** 8329 | [OpenAlex ID](https://openalex.org/A5100733506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 GT‑Space 框架，用单一适配器将异构传感器的特征投影到由真实标签构建的公共特征空间，实现高效协同感知；

**💡 创新点**

创新点在于利用 ground‑truth 构建公共空间，避免了对编码器的重新训练或多对多解释器，并通过组合对比损失实现跨模态的统一学习；

**🔧 技术方法**

主要技术包括基于 BEV 的特征编码、单一适配器投影、跨模态组合对比学习以及多模态 Transformer 融合网络；

**📊 数据集**

实验数据集涵盖 OPV2V、V2XSet 与 RCooper 三个公开协同感知数据集；

**📈 对比分析**

与 Late Fusion、HM‑ViT、PnPDA、HEAL、Hetecooper、STAMP 等方法对比，GT‑Space 在 AP@0.5/0.7 上均取得最高或最接近最高的检测精度，显示出明显的性能优势；

**⚠️ 局限性**

局限性在于需要完整的 ground‑truth 标注以及理想的通信与位姿对齐，未来需探索弱监督或鲁棒性改进。

---

## 97. Regret Analysis of Sleeping Competing Bandits

**arXiv ID:** 2603.19700 | [PDF](https://arxiv.org/pdf/2603.19700v1)

**作者:** Shinnosuke Uba `[一作]` (Osaka University), Yutaro Yamaguchi `[通讯]` (Osaka University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了“睡眠竞争式多臂赌博机”模型，考虑玩家和臂随时间可用性的动态变化；

**💡 创新点**

在该动态环境下定义了玩家最优稳定回报与最差稳定回报，推导了相应的上界与下界，并给出了渐进最优算法；

**🔧 技术方法**

采用集中式UCB、探索-格兰-肖特（ETGS）算法与置信区间分析（UCB/LCB）来实现学习与匹配；

**📊 数据集**

本文未使用公开数据集，所有实验均基于自定义仿真场景；

**📈 对比分析**

通过理论证明与仿真对比，AC-UCB在玩家最差稳定回报上达到O(NK log T/Δ²)的渐进最优上界；AC-ETGS在玩家最优稳定回报上得到O(NK² log T/Δ²)的上界；

**⚠️ 局限性**

主要局限在于缺乏对玩家最优稳定回报的更紧凑下界，且仅考虑了随机（非对抗性）奖励分布，未验证在对抗环境下的表现。

---

## 98. Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents

**arXiv ID:** 2603.19375 | [PDF](https://arxiv.org/pdf/2603.19375v1)

**作者:** Toan Tran `[一作]` (Emory University), Li Xiong `[通讯]` (Emory University)

**通讯引用:** 15734 | [OpenAlex ID](https://openalex.org/A5078394535)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用LLM代理自动设计和实现会员推断攻击（MIA）的信号计算策略。

**💡 创新点**

首次展示LLM代理能够通过进化搜索自动生成高性能MIA，且可在多种模型/数据上超过人类设计方案。

**🔧 技术方法**

LLM代理的探索–利用进化循环、自然语言设计描述、代码编写与执行、性能分析。

**📊 数据集**

在黑盒LLM（ArXiv、Pubmed、Github等）和灰盒VLM（image‑logs）上进行评估。

**📈 对比分析**

与人类设计的MIA和OpenEvolve对比，提升AUC最高可达0.18，TPR@1%FPR提升幅度显著。

**⚠️ 局限性**

仅覆盖信号计算阶段、执行环境有限导致约15–20%实现失败、实验迭代数与时间有限、仅测试少数基准、存在双重用途风险。

---

## 99. Beyond the Desk: Barriers and Future Opportunities for AI to Assist Scientists in Embodied Physical Tasks

**arXiv ID:** 2603.19504 | [PDF](https://arxiv.org/pdf/2603.19504v1)

**作者:** Irene Hou `[一作]` (University of California San Diego), Philip J. Guo `[通讯]` (University of California San Diego)

**通讯引用:** 8823 | [OpenAlex ID](https://openalex.org/A5060695800)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究通过现场访谈和情境化访谈，探究科学实验室与野外科研人员在物理操作任务中对AI的使用与阻碍，并提出未来AI协助的五大设计范式；

**💡 创新点**

首次系统阐述“工作桌面”之外的实验室与野外科学实践对AI的需求与限制，强调AI应作为支持基础设施而非取代专家决策；

**🔧 技术方法**

采用人机交互研究方法——现场半结构化访谈、情境化走访与启发式设计活动；

**📊 数据集**

收集了12名实验室/野外科学从业者（包括学生、技术员和研究员）的访谈录音与手绘草图；

**📈 对比分析**

本文无算法性能评估，仅通过定性主题分析比较不同场景下的AI使用场景与阻碍，未进行实验对比；

**⚠️ 局限性**

研究样本量有限，主要来自西方高校与工业实验室，缺乏对高级PI层级与非英语科研环境的覆盖，且未直接观察现场实验操作，可能影响结果的普适性。

---

## 100. PAI: Fast, Accurate, and Full Benchmark Performance Projection with AI

**arXiv ID:** 2603.19330 | [PDF](https://arxiv.org/pdf/2603.19330v1)

**作者:** Avery Johnson `[一作]` (Texas A&M University), Abdullah Muzahid `[通讯]` (Texas A&M University)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5102915313)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于AI的全基准性能预测方法PAI，利用层次化LSTM模型从程序执行的微架构无关特征（uAIMs）和硬件配置中直接回归IPC，无需详细仿真或指令级编码。

**💡 创新点**

创新点在于实现了完整基准的快速准确预测，并且不依赖细粒度仿真或特定架构的指令信息，采用层次化LSTM分别处理uAIMs和硬件配置并融合，以提升泛化能力。

**🔧 技术方法**

采用的技术包括uAIMs特征提取（来自Simics或真实机器），层次化LSTM网络结构（两层LSTM分别对uAIMs和硬件配置建模，再通过上层LSTM聚合），以及PyTorch框架下的训练与推理。

**📊 数据集**

使用的数据集为SPEC CPU2017基准套件，收集128个uAIMs每10M指令一次，覆盖15种Intel Xeon SKU，总计约130万条样本，其中对一部分基准（XZ, WRF, MCF等）做无见测试。

**📈 对比分析**

与现有方案TAO、SimNet等相比，PAI在全基准预测中平均IPC误差为9.35%（见/未见基准分别为7.64%/15.5%），推理时间仅2 min 57 s（约1 s/10B指令），比TAO/SimNet快约三阶，精度相当或略逊但速度显著提升。

**⚠️ 局限性**

局限性包括对未见基准的误差较大（高达15.5%），部分基准如cactubssn、gcc、lbm存在显著误差，且模型尚未在多代处理器和其他基准套件上验证其普适性。

---

## 101. Uncertainty-aware Prototype Learning with Variational Inference for Few-shot Point Cloud Segmentation

**arXiv ID:** 2603.19757 | [PDF](https://arxiv.org/pdf/2603.19757v1)

**作者:** Yifei Zhao `[一作]` (Fudan University), Yinsheng Li `[通讯]` (Fudan University)

**通讯引用:** 1708 | [OpenAlex ID](https://openalex.org/A5101658474)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出UPL，一种基于变分推断的概率原型学习框架，利用支持与查询样本的双流原型细化生成不确定性感知的3D点云分割；

**💡 创新点**

创新点在于(1)双流原型细化模块通过共享token注意力融合支持与查询信息提升原型判别；(2)将类原型视为高斯潜在变量，采用KL正则化的变分推断实现显式不确定性建模；(3)通过多样本蒙特卡洛采样在推理阶段提供可靠的置信度与不确定性估计；

**🔧 技术方法**

采用变分推断、通道注意力与共享token注意力、1D卷积残差结构、KL散度正则、蒙特卡洛采样、基于点云的特征提取（backbone fθ）和Farthest Point Sampling；

**📊 数据集**

使用S3DIS和ScanNet两个FS-PCS基准数据集，在CoSeg框架下进行评估；

**📈 对比分析**

与AttMPTI、QGE、QPGA、CoSeg等基线在1/5-shot和2-way设置下对比，UPL在S3DIS上实现48.60/52.22/37.79/41.87 mIoU，在ScanNet上实现43.00/46.83/32.39/38.40 mIoU，平均提升约2-4个百分点，达到或逼近最优水平；

**⚠️ 局限性**

局限性包括：需要多次采样T进行推理，导致推理时间略长；实验仅覆盖FS-PCS任务，缺乏对更大规模或其他3D任务的验证；在极少样本或高度跨类别场景下，原型分布与真实分布可能仍存在偏差。

---

## 102. MetaCues: Enabling Critical Engagement with Generative AI for Information Seeking and Sensemaking

**arXiv ID:** 2603.19634 | [PDF](https://arxiv.org/pdf/2603.19634v1)

**作者:** Anjali Singh `[一作]` (University of Texas), Soo Young Rieh `[通讯]` (University of Texas)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了MetaCues，一种自动生成元认知提示的生成式AI搜索工具；

**💡 创新点**

首次实现了基于用户交互实时、自动化的元认知提示生成，并通过实验验证其对搜索行为和判断信心的影响；

**🔧 技术方法**

利用OpenAI GPT‑4o进行对话和搜索，配合内部逻辑模型根据查询、笔记和来源点击数据生成六类提示；

**📊 数据集**

使用由Prolific招募的146名美国大学生在两组搜索主题（社交媒体与音乐）中完成的交互日志数据；

**📈 对比分析**

通过对比含提示与不含提示的两种工具，采用GLM和ANOVA等统计方法，结果显示在音乐主题下提示显著提升查询多样性、外部链接点击率和判断信心；

**⚠️ 局限性**

样本量有限、提示时序固定、缺乏长期任务和定性分析，且对不同主题的效果差异尚不充分解释；

---

## 103. Audio Avatar Fingerprinting: An Approach for Authorized Use of Voice Cloning in the Era of Synthetic Audio

**arXiv ID:** 2603.20165 | [PDF](https://arxiv.org/pdf/2603.20165v1)

**作者:** Candice R. Gerstner `[一作]` (National Security Agency), Candice R. Gerstner `[通讯]` (National Security Agency)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5049561025)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用现成的 TitaNet 说话人验证模型，探究其在真实/合成语音检测和音频头像指纹（鉴别合成语音是否由授权身份驱动）两大法医任务中的性能，并在此基础上进行微调提升指纹识别效果。

**💡 创新点**

提出首个音频头像指纹任务与对应数据集，并证明即使原始说话人验证模型未训练过，亦能通过少量微调实现对授权与非授权合成语音的区分。

**🔧 技术方法**

核心技术为 TitaNet 说话人嵌入提取、余弦相似度判定、Angular Softmax 损失微调；结合深度语音合成与识别框架。

**📊 数据集**

使用自建的 NVFAIR 语音数据集（包含 4h45m 自reenactment 与 90h12m cross-reenactment）以及公开 InTheWild 语音深度伪造基准。

**📈 对比分析**

与 ClovaAI、H/ASP、ECAPA-TDNN、POI-Forensics 等说话人验证基线相比，原始 TitaNet 在伪造检测上 AUC 达到 0.91，微调后在音频头像指纹任务上 ROC 曲线显著提升，显示可接受的鉴别效果。

**⚠️ 局限性**

主要局限在于需要事先的说话人报名数据，且模型对不同情绪或环境下的说话风格变化敏感；当前实验规模有限，需在更大多样化数据集上进一步验证和扩展。

---

## 104. Exact and Approximate Convex Reformulation of Linear Stochastic Optimal Control with Chance Constraints

**arXiv ID:** 2603.19454 | [PDF](https://arxiv.org/pdf/2603.19454v1)

**作者:** Tanmay Dokania `[一作]` (Georgia Institute of Technology), Yashwanth Kumar Nakka `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于状态升维的凸优化框架，用来求解带线性与二次机会约束的离散时间随机线性控制问题，并在四旋翼最小快照轨迹规划中进行验证。

**💡 创新点**

创新点在于：① 通过升维直接编码状态与控制的矩时刻信息，使线性机会约束可精确转化为二阶锥约束；② 对二次机会约束给出更紧凑的线性矩阵与二次约束松弛；③ 不再需要先验协方差或反馈增益的猜测，显著提升可行域。

**🔧 技术方法**

使用的技术包括：升维表示、Gaussian 分布假设、线性矩阵不等式（LMI）和二阶锥规划（SOCP）、对高斯随机变量的期望与方差展开、以及 Markov/Chi‑square 分布的保守近似。

**📊 数据集**

实验数据来自于仿真的二维四旋翼模型，使用 Circle Arena 与 Funnel Corridor 两个几何约束场景，分别在不同噪声幅度和通道宽度下进行参数扫荡；未使用公开真实数据集。

**📈 对比分析**

通过与需要 Riccati 方程或协方差估计的基线方法比较，结果显示本方法在噪声标准差高达原方法十倍时仍保持可行，并在成本上提升了约 30–43%，在 Circle Arena 中可达 35% 成本降低。

**⚠️ 局限性**

局限性包括：仅适用于高斯噪声与线性系统；受限于仿真场景，缺乏在真实复杂环境下的验证；升维导致变量维度快速增长，可能影响大规模问题的求解速度。

---

## 105. Trojan's Whisper: Stealthy Manipulation of OpenClaw through Injected Bootstrapped Guidance

**arXiv ID:** 2603.19974 | [PDF](https://arxiv.org/pdf/2603.19974v1)

**作者:** Fazhong Liu `[一作]` (Shanghai Jiao Tong University), Haojin Zhu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9395 | [OpenAlex ID](https://openalex.org/A5039106671)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并实验了在OpenClaw自主编码代理中通过技能生命周期钩子注入Bootstrap指导文件，实现了隐蔽的“指导注入”攻击，并评估了其在模拟真实开发者环境下的危害。

**💡 创新点**

提出了新的“指导注入”攻击面，展示了通过Bootstrap文件对LLM代理推理进行隐蔽操控，并首次构建了多层级攻击评估框架与对策。

**🔧 技术方法**

利用LLM推理、OpenClaw技能钩子、对抗性技能生成、六大主流LLM后端（Claude Opus, GPT‑5, Qwen3, Kimi, Gemini, DeepSeek）以及静态/LLM扫描器等技术。

**📊 数据集**

构建了ORE‑Bench（模拟开发者工作区）和DevSecBench基准，包含26个恶意技能与52个自然提示，用以评估攻击效果。

**📈 对比分析**

通过在六个LLM后端对52个提示进行实验，攻击成功率从16%到64%不等，高风险场景最高可达64%，同时94%恶意技能逃逸现有检测。

**⚠️ 局限性**

局限包括：实验仅在合成环境进行，未涵盖真实系统多样性；仅测试六大LLM后端；只探究OpenClaw的Bootstrap钩子，其他注入通道未覆盖。

---

## 106. Spectral Tempering for Embedding Compression in Dense Passage Retrieval

**arXiv ID:** 2603.19339 | [PDF](https://arxiv.org/pdf/2603.19339v1)

**作者:** Yongkang Li `[一作]` (University of Amsterdam), Evangelos Kanoulas `[通讯]` (University of Amsterdam)

**通讯引用:** 3959 | [OpenAlex ID](https://openalex.org/A5055639036)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了密集检索嵌入的降维问题，提出 Spectral Tempering（SpecTemp）方法通过自适应调节谱权重指数 γ(k) 来平衡方差保留与各向同性。

**💡 创新点**

创新点在于将 γ(k) 视为与目标维度相关的局部信噪比（SNR）函数，利用谱噪声底估计和 knee 点定位自动生成，无需人工调参或验证集。

**🔧 技术方法**

采用谱分解、噪声底估计、Kneedle knee 检测、线性变换等技术实现自适应温度化。

**📊 数据集**

实验使用四个检索数据集：MS MARCO Passage Ranking、Natural Questions（NQ）、FEVER、FiQA，并评估多种 LLM 基础检索模型。

**📈 对比分析**

与 PCA、标准白化、固定 γ‑白化、前缀截断、随机截断、随机投影等学习‑free 方法比较，SpecTemp 在大多数维度配置下与 grid‑searched γ* 近乎一致，整体性能优于或等价于传统方法。

**⚠️ 局限性**

局限性包括：对谱分解的内存和计算开销、对噪声阈值估计的稳定性依赖、对动态更新语料库的适应性有限。

---

## 107. Beam-aware Kernelized Contextual Bandits for User Association and Beamforming in mmWave Vehicular Networks

**arXiv ID:** 2603.19285 | [PDF](https://arxiv.org/pdf/2603.19285v1)

**作者:** Xiaoyang He `[一作]` (Sun Yat-sen University), Manabu Tsukada `[通讯]` (University of Tokyo)

**通讯引用:** 1181 | [OpenAlex ID](https://openalex.org/A5067716610)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种 Beam-aware Kernelized Contextual Upper Confidence Bound（BKC-UCB）算法，用于毫米波车联网中基于上下文信息（车辆位置、速度、干扰等）进行用户关联和波束成形的在线学习。

**💡 创新点**

创新点包括：① 将波束索引嵌入上下文，利用核方法捕捉不同波束间的相关性，加速收敛；② 设计多维专门核函数（角度、距离、多普勒、干扰、偏差），精细建模毫米波特性；③ 引入事件触发信息共享机制，仅在重要探索时才同步，显著降低通信开销。

**🔧 技术方法**

采用核化上下文多臂赌博机（Kernelized CMAB）与UCB策略，在再生核希尔伯特空间（RKHS）中实现非线性奖励预测，并结合层级波束搜索实现自适应波束调整。

**📊 数据集**

使用东京真实城市地图（OpenStreetMap）、SUMO 生成车辆轨迹、OpenCellID 的真实基站部署、以及基于射线追踪的 CDL 车载毫米波信道模型。

**📈 对比分析**

与离线全 CSI 的 Worst Connection Swapping (WCS) 以及基于 CSI 的 DK-UCB 进行对比；BKC-UCB 在无需 CSI 的前提下，平均每车吞吐量比 DK-UCB 高 0.0273 Gbps，且同步率低于 DK-UCB；虽然在最优下 WCS 仍略胜 22.8%，但 BKC-UCB 在实时在线场景中表现优异。

**⚠️ 局限性**

限制包括：① 对大规模基站/波束集合的计算量与存储需求仍较高；② 仍需在每个关联周期内执行 UCB 与核矩阵更新，可能影响极低时延需求；③ 在极高车辆密度或高速场景下，干扰模型与速度估计误差可能导致预测误差增大。

---

## 108. SIMPLER: Efficient Foundation Model Adaptation via Similarity-Guided Layer Pruning for Earth Observation

**arXiv ID:** 2603.19873 | [PDF](https://arxiv.org/pdf/2603.19873v1)

**作者:** Víctor Barreiro `[一作]`, Dora B. Heras `[通讯]` (Universidade de Santiago de Compostela)

**通讯引用:** 1354 | [OpenAlex ID](https://openalex.org/A5018628722)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SIMPLER方法，在预训练模型上通过层间相似度分析自动选择合适的网络深度，从而在不进行完整微调前就能剪枝，降低训练与推理成本。

**💡 创新点**

创新点在于利用预训练表示的相似度（CKA）在无梯度、无超参的情况下自动确定冗余层，既不需要后期重训练，也不保留完整深度；可一次性减少参数并保持性能。

**🔧 技术方法**

技术主要是Centered Kernel Alignment相似度计算、自动分区得分函数、ViT/Multi-Modal Transformer结构以及LoRA等参数高效微调技术。

**📊 数据集**

使用的数据集包括Prithvi-EO-2、MADOS、BigEarthNetv2、Sen4Map、TerraMind（Large/Small/Tiny）以及ViT-MAE在CIFAR-100上的实验。

**📈 对比分析**

与全微调、LoRA、后置结构化剪枝等方法比较，SIMPLER在保留94-101%原始性能的同时实现55-87%参数压缩、2-3×训练加速和2-3×推理加速，尤其在资源受限的边缘设备上表现突出。

**⚠️ 局限性**

局限性：该方法依赖预训练模型在目标任务上展现层级相似度稳化，若预训练策略或任务分布差异较大则可能失效；对极小模型或非Transformer结构的适用性尚未充分验证。

---

## 109. GustPilot: A Hierarchical DRL-INDI Framework for Wind-Resilient Quadrotor Navigation

**arXiv ID:** 2603.19966 | [PDF](https://arxiv.org/pdf/2603.19966v1)

**作者:** Amir Atef Habel `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2134 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个分层导航系统GustPilot，使用深度强化学习（DRL）产生速度参考，配合几何增量非线性动态反演（INDI）控制器进行低层姿态跟踪，从而在强风扰动下实现轻量四旋翼机的门口穿越。

**💡 创新点**

创新点在于：①把风扰动抑制从上层规划移至低层控制，使DRL策略仅关注路径规划；②在训练阶段采用“风洞域随机化”（fan‑jet随机化），提升了从仿真到实飞的转移；③演示了单门单风训练的策略能在多门多风甚至移动障碍环境中无须再训练就能通用。

**🔧 技术方法**

技术手段包括：Proximal Policy Optimization（PPO）训练的全连接网络，产生惯性坐标下的速度指令；几何INDI控制器（在SE(3)上）实现加速度、姿态、力矩的增量反馈；双滤波（Butterworth低通）处理加速度、角速度、角加速度；离散式风场模型（局部风喷流+O-U随机过程）用于域随机化。

**📊 数据集**

数据集主要来自仿真环境：随机生成门位姿、起飞状态与风喷流参数；实际实验使用50 g Crazyflie 2.x搭载Vicon定位，风速由管道风扇产生，测试涵盖四种场景（无风、分布风、集中风、动态风+移动门）。

**📈 对比分析**

通过与同一DRL策略配合PID控制器（DRL‑PID）对比，实验显示：整体成功率（OSR）从约55%提升至约95%；门通过率、无接触率、完成率均显著提升；在分布式风场中DRL‑INDI完成率达97%，DRL‑PID为0%；在高风速（3.5 m/s）下仍能保持1.34 m/s的航速。

**⚠️ 局限性**

局限性包括：仅在极小尺寸（50 g）四旋翼上验证，未证明可扩展到更大平台；当前策略仅输出速度指令，未直接利用INDI的力/力矩估计进行在线扰动识别；对传感器噪声、延迟的鲁棒性未系统评估；在极端多源风或极端低速/高动态环境下的表现仍需进一步验证。

---

## 110. Non-trivial automata networks do exist that solve the global majority problem with the local majority rule

**arXiv ID:** 2603.19472 | [PDF](https://arxiv.org/pdf/2603.19472v1)

**作者:** Pedro Paulo Balbi `[一作]` (Universidade Presbiteriana Mackenzie), Eurico Ruivo `[通讯]` (Universidade Presbiteriana Mackenzie)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5038640433)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了局部多数规则下的布尔自动机网络（MBAN），并证明存在若干非平凡的网络拓扑能在任意初始配置下实现全局多数决策（即解Density Classification Task）

**💡 创新点**

首次给出了四类非平凡的网络拓扑（非完全连通但局部可视化的结构）能够通过仅使用局部多数更新规则解决DCT，突破了传统单循环网络无法解决的局限

**🔧 技术方法**

采用布尔自动机理论、图论与组合分析，对网络的相互作用图进行构造与证明，并利用功能有向图与周期分析等数学工具来证明收敛性

**📊 数据集**

无实验数据集，全部为理论构造与组合枚举；通过对 n=3、n=5 等小规模网络的穷举实验得到示例

**📈 对比分析**

与传统完全连通图、环形图等基准进行理论对比；证明在构造的拓扑下收敛时间可为常数步或受限步长；相较于完全图仅需局部信息即可达成同样功能

**⚠️ 局限性**

仅在奇数节点规模下适用；构造的解决方案未必可推广到更大规模；枚举结果表明多数解并非可泛化；且对网络拓扑的要求较高，实际实现受限

---

## 111. Ternary Gamma Semirings: From Neural Implementation to Categorical Foundations

**arXiv ID:** 2603.19317 | [PDF](https://arxiv.org/pdf/2603.19317v1)

**作者:** Ruoqi Sun `[一作]` `[通讯]` (Intelligent Game and Decision Lab), Ruoqi Sun (Intelligent Game and Decision Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个最小的组合推理任务（XOR），展示标准神经网络在未见组合上完全失败；随后设计带有三元 Γ-半环逻辑约束的网络，通过特征空间的逻辑损失与原型分类，使模型在同一任务上实现 100% 的测试准确率。

**💡 创新点**

创新点在于将神经网络学习的特征空间映射为有限可交换三元 Γ-半环（|T|=4, |Γ|=1），其三元运算即为“多数投票”规则，证明了网络能自动内部化数学上“自然”的代数结构，并首次将此结构与纯数学分类结果关联起来。

**🔧 技术方法**

使用带逻辑约束的特征提取网络（带三元运算的损失函数）、距离原型分类器以及对三元运算的枚举与符号检验；同时利用多层感知机与 ReLU 激活实现网络训练。

**📊 数据集**

仅使用极简数据集：二维二进制属性（颜色/形状）共 4 个样本，训练集包含同一类的 2 个样本，测试集包含另一类的 2 个未见样本。

**📈 对比分析**

比较方法包括：随机猜测（50%）、标准 NN（0%）、Ternary Gamma 仅逻辑损失（0%）、Ternary Gamma + 原型（100%）。结果显示逻辑约束与原型结合显著提升泛化性能，达成 100% 的测试准确率。

**⚠️ 局限性**

局限性：实验仅在极其简化的二元 XOR 任务上验证，尚未证明在更大规模、多属性或更复杂逻辑任务中的可扩展性；逻辑约束需手工设计，缺乏自动化学习机制。

---

## 112. Beyond Quadratic: Linear-Time Change Detection with RWKV

**arXiv ID:** 2603.19606 | [PDF](https://arxiv.org/pdf/2603.19606v1)

**作者:** Zhenyu Yang `[一作]` (Nanjing University of Science and Technology), Yazhou Yao `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 3841 | [OpenAlex ID](https://openalex.org/A5027545344)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于RWKV的高效遥感变化检测框架ChangeRWKV，兼顾精度与计算效率；

**💡 创新点**

创新点在于将线性时间注意力的RWKV架构与多尺度层次编码器相结合，并设计了专门的空间-时序融合模块(STFM)以精准捕获跨尺度时序变化；

**🔧 技术方法**

使用技术包括RWKV线性时间注意力、双向空间混合、轻量化SE通道混合、层次化特征提取、Cross‑CBAM的空间-时序融合、BCE+Dice混合损失；

**📊 数据集**

实验数据集覆盖光学与SAR四个公开基准：LEVIR-CD、WHU-CD、LEVIR-CD+、SAR-CD；

**📈 对比分析**

与多种SOTA方法（ChangeFormer、ConvFormer、SwinSUNet、ChangeMamba等）对比，ChangeRWKV在LEVIR-CD上达到85.46% IoU（Tiny模型仅4.66M参数、9.40G FLOPs），在SAR-CD上实现97.18% IoU，显著降低参数和计算量同时保持或超过最先进性能；

**⚠️ 局限性**

局限性包括对大规模标注数据的依赖、缺乏针对SAR或高光谱等特殊模态的先验，未来需要探索弱监督学习与更广泛的跨模态适应。

---

## 113. Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders

**arXiv ID:** 2603.19771 | [PDF](https://arxiv.org/pdf/2603.19771v1)

**作者:** Debajyoti Mazumder `[一作]` (Indian Institute of Science Education and Research), Jasabanta Patro `[通讯]` (Indian Institute of Science Education and Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比多语言编码器在处理印地语‑英语混合文本时的内部表示，构建三语并行语料并通过CKA、token saliency和熵分析探究对齐情况。

**💡 创新点**

提出三语后训练对齐目标，使代码混合表征同时与两种母语对齐，提升跨语言对齐和下游任务性能。

**🔧 技术方法**

采用多语言预训练模型（mBERT、XLM‑R及其 Hing 版本）以及CKA、token saliency、熵基不确定性分析和对齐损失。

**📊 数据集**

21,139句三语对齐数据，来源于CM‑En parallel、PHINC、LinCE 2021 等公开资源。

**📈 对比分析**

通过句子检索向量相似度评估，对齐准确率提升至 50–70% 左右，CLAS 分数显著提高，下游情感分析和仇恨检测准确率也随之提升。

**⚠️ 局限性**

仅使用单一语言对的迁移；对齐语料规模有限；仅验证编码器架构，未扩展到大型解码器 LLM；自动翻译可能带来细节偏差。

---

## 114. Exploring Subnetwork Interactions in Heterogeneous Brain Network via Prior-Informed Graph Learning

**arXiv ID:** 2603.19307 | [PDF](https://arxiv.org/pdf/2603.19307v1)

**作者:** Siyu Liu `[一作]` (Northeastern University), Osmar R. Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5027917989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出KD-Brain框架，利用先验知识指导脑网络子网络交互学习，显著提升精神障碍诊断性能；

**💡 创新点**

在Transformer注意力中引入疾病特异性语义先验作为查询，形成语义条件交互；并通过LLM生成的临床先验构建病理一致约束，二者共同实现知识驱动的子网络交互建模；

**🔧 技术方法**

采用双向卷积空间编码器提取子网络拓扑，BioMedBERT生成语义先验，GPT‑4（或DeepSeek‑R1）生成交互先验，KL散度正则化和MLP分类器完成最终预测；

**📊 数据集**

在ABIDE NYU子集与单中心U中心数据集上评估，分别包含ASD、BD、MDD三类诊断任务；

**📈 对比分析**

与12种SOTA（GNN、Transformer等）对比，KD‑Brain在ACC/AUC上均超过对手，尤其在ASD/BD任务上提升约5–10%，最佳表现出现在q=2的高阶交互设置；

**⚠️ 局限性**

受限于样本量仍有限、LLM生成先验的准确性与泛化、以及对更大规模多模态脑网络的扩展性尚待验证。

---

## 115. Behavioral Engagement in VR-Based Sign Language Learning: Visual Attention as a Predictor of Performance and Temporal Dynamics

**arXiv ID:** 2603.19535 | [PDF](https://arxiv.org/pdf/2603.19535v1)

**作者:** Davide Traini `[一作]` (Università delle Marche), Enrique Yeguas-Bolívar `[通讯]` (Universidad de Córdoba)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过分析117名大学生在SONAR VR手语学习应用中的交互日志，研究视觉注意（VA）、视频重放时间（PPVT）和重播频率（VRF）等行为指标与学习成绩的关系，并对VA的时间动态进行了聚合分析。

**💡 创新点**

创新点在于首次系统评估可自动提取的行为指标对VR手语学习效果的预测能力，发现VA和PPVT为关键预测因子，并揭示了训练与评估阶段的显著时间性注意模式。

**🔧 技术方法**

使用的技术包括基于头部姿态的视觉注意计算、视频冻结时间提取PPVT、重播次数统计，以及相关分析与二项GLM回归模型，用于预测学习成绩。

**📊 数据集**

数据集为117名大学生在SONAR VR手语学习平台上完成的训练与验证阶段的交互日志，包含头部姿态、视频播放与重播计数等原始记录。

**📈 对比分析**

采用Pearson相关和GLM回归进行比较；结果显示VA与成绩的相关系数为0.76，PPVT为0.66，VRF几乎为0；GLM模型伪R²约为0.83，说明VA和PPVT共同显著预测学习成功，而VRF无显著贡献。

**⚠️ 局限性**

局限性包括仅为相关性研究，缺乏因果验证；样本为欧洲大学生且手语经验有限；任务规模受限于12个基础短语；未采集生理或思考过程数据，限制了对内部认知机制的直接推断。

---

## 116. Multi-Agent Motion Planning on Industrial Magnetic Levitation Platforms: A Hybrid ADMM-HOCBF approach

**arXiv ID:** 2603.19838 | [PDF](https://arxiv.org/pdf/2603.19838v1)

**作者:** Bavo Tistaert `[一作]` (KU Leuven), Wilm Decré `[通讯]` (KU Leuven)

**通讯引用:** 980 | [OpenAlex ID](https://openalex.org/A5009275320)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种结合分布式ADMM和中心化高阶控制屏障函数（HOCBF）的混合MPC运动规划器，能够实时生成多智能体的安全轨迹并在工业磁悬浮平台上实现。

**💡 创新点**

创新点在于将分布式ADMM求解非凸碰撞约束与中心化HOCBF安全滤波相结合，既保留了ADMM的可扩展性，又通过HOCBF提供严格安全保证。

**🔧 技术方法**

技术包括分布式ADMM求解器、连续时间高阶控制屏障函数、二次约束二次规划（QCQP）安全滤波，以及多射击离散化与CasADi/ROCKIT、FATROP/qpOASES等求解器。

**📊 数据集**

实验使用了自定义的Beckhoff XPlanar磁悬浮平台进行硬件验证，并在仿真中随机生成2–30个代理的起止位置进行性能评估。

**📈 对比分析**

与传统集中式MPC相比，ADMM‑HOCBF在轨迹耗时、平均计算时间和碰撞数上均保持零碰撞，且计算时间从O(N²·⁹⁶)降至O(N¹·⁸³)，在5个代理时实现约115 ms的实时计算。

**⚠️ 局限性**

局限性包括对非凸邻域处理仍依赖经验性通道策略、ADMM仅在凸问题下收敛理论保证、以及安全滤波在高迭代次数下计算开销仍可进一步降低。

---

## 117. The Bilateral Efficiency of Ethernet: Recalibrating Metcalfe and Boggs After Fifty Years

**arXiv ID:** 2603.19406 | [PDF](https://arxiv.org/pdf/2603.19406v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAEDALUS), Paul Borrill (DAEDALUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

重新阐释1976年以太网效率模型，提出双边效率指标 E_B，评估链路产生双边协议的效率。

**💡 创新点**

首次将双向信息流视为效率核心，将终止延迟（end‑dally）解释为双边协议，并将其与OAE的Perfect Information Feedback及两状态向量形式（TSVF）对齐。

**🔧 技术方法**

基于 Shannon 通道理论、两状态向量形式（TSVF）、以太网 CSMA/CD 模型以及 OAE 的切片 ACK 机制。

**📊 数据集**

使用 Metcalfe‑Boggs 原始实验数据（P = 48/512/1024/4096 位，Q = 1…256）以及现代以太网参数（如 800 Gbps、全双工）进行理论推导。

**📈 对比分析**

对比传统前向效率 E 与双边效率 E_B，结果显示 E_B 远低于 E，尤其在高 Q 或大 P 时差距明显；现代全双工链路虽然 E ≈ 1，但 E_B 可能接近 0。

**⚠️ 局限性**

仅为理论模型，缺乏实验验证；假设 OAE 实现可行且切片 ACK 成本为零；未考虑应用层语义验证、错误处理及实际硬件实现成本。

---

## 118. Scalable Prompt Routing via Fine-Grained Latent Task Discovery

**arXiv ID:** 2603.19415 | [PDF](https://arxiv.org/pdf/2603.19415v1)

**作者:** Yunyi Zhang `[一作]` (Amazon Web Services), George Karypis `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一种两阶段的提示路由框架，先通过图聚类自动发现细粒度任务类型，再利用任务感知的Mixture-of-Experts质量估计器在前沿LLM池中动态选择最优模型。

**💡 创新点**

创新点在于：①利用语义+性能双信号的图聚类实现无监督的细粒度任务发现；②通过任务类型激活专属预测头实现任务感知质量估计；③两阶段评分融合兼顾任务级稳定性与实例级适应性。

**🔧 技术方法**

技术手段包括：图嵌入与Leiden社区检测、Rank Biased Overlap度量、LLM生成任务描述、二元分类器、Mixture-of-Experts预测头、min‑max归一化融合及Skywork奖励模型。

**📊 数据集**

实验使用了10个主流基准数据集（NQ、TriviaQA、CommensenseQA、MMLU、ARC、OpenBookQA、GSM8K、MATH、HumanEval、MBPP），共约279k训练样本。

**📈 对比分析**

与kNN、MLP、RouteLLM、RouterDC、GraphRouter、IPR等基线及11个前沿LLM直接对比，FineRouter在所有任务上均超过基线，并在最强模型Claude‑Sonnet‑4.5之上，同时推理成本不到其一半。

**⚠️ 局限性**

局限性包括：依赖LLM生成任务描述与奖励模型，易受偏差；聚类超参对结果敏感；训练需要访问全部候选模型响应，且目前仅支持文本提示，未扩展至多模态。

---

## 119. Demographic-Aware Self-Supervised Anomaly Detection Pretraining for Equitable Rare Cardiac Diagnosis

**arXiv ID:** 2603.19695 | [PDF](https://arxiv.org/pdf/2603.19695v1)

**作者:** Chaoqin Huang `[一作]` (Shanghai Jiao Tong University), Ya Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了基于自监督异常检测预训练和人口学特征对齐的两阶段ECG诊断框架，用于提高稀有心脏异常的检测与定位；

**💡 创新点**

创新点在于将多尺度交叉恢复与趋势辅助恢复结合的自监督预训练，同时加入属性预测模块实现公平性和可解释性；

**🔧 技术方法**

使用的技术包括自监督多尺度交叉恢复、趋势辅助恢复、属性预测模块、异步损失（Asymmetric Loss）及CPU优化实现快速推理；

**📊 数据集**

使用的数据集为内部大规模ECG-LT（109万条，116种异常，其中43种稀有），外部PTB-XL（21837条）和Renji临床队列（1385条）；

**📈 对比分析**

与无预训练、对比学习、MoCo、TSL等基线对比，在内部稀有类AUROC从85.8%提升至94.7%，常见-稀有差距缩小73%；在PTB-XL和Renji上分别提升1.8–8.1% AUROC；AI辅助诊断将心脏科医师准确率提升6.7%，诊断时间减少32.5%；

**⚠️ 局限性**

局限性包括在极端年龄（<10岁、>90岁）以及极少样本的稀有类别上性能仍有下降，且对不同设备与临床实践的适应性需进一步校准；

---

## 120. Efficiency Follows Global-Local Decoupling

**arXiv ID:** 2603.19567 | [PDF](https://arxiv.org/pdf/2603.19567v1)

**作者:** Zhenyu Yang `[一作]` (Nanjing University of Science and Technology), Fumin Shen `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 13151 | [OpenAlex ID](https://openalex.org/A5074492050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ConvNeur 两分支架构，分别负责局部细节提取和全局上下文聚合，并通过门控融合实现全局对局部的调节。

**💡 创新点**

创新点在于将全局推理与局部表征解耦，利用压缩分块的神经记忆模块进行全局上下文聚合，并通过学习门控而非直接叠加的方式将全局信息注入局部特征。

**🔧 技术方法**

核心技术包括：局部保持的卷积分支；压缩维度、分块后进入的神经记忆模块（q/k/v 投影、surprise‑driven 更新）；门控调制融合；以及基于块级的子线性全局计算实现子二次复杂度。

**📊 数据集**

实验数据集：ImageNet‑1K（分类）；COCO 2017（目标检测与实例分割）；ADE20K（语义分割）。

**📈 对比分析**

与多种卷积、Transformer 及混合模型在相同 FLOPs/参数预算下比较，ConvNeur 在 ImageNet‑1K 上 Top‑1 Acc 提升至 81.5%（3.1 GFLOPs），在 COCO 上 AP 达 42.4/44.5，ADE20K 上 mIoU 达 41.42/45.72，均优于或匹配同等成本的现有方法。

**⚠️ 局限性**

局限性：块大小固定，可能在极高分辨率或不同尺度输入下需要自适应；记忆更新策略在视频/流式任务中尚未验证；门控参数较少，可能限制跨任务的泛化；整体架构主要针对单帧图像，扩展到动态图像需要进一步研究。

---

## 121. Orchestrating Human-AI Software Delivery: A Retrospective Longitudinal Field Study of Three Software Modernization Programs

**arXiv ID:** 2603.20028 | [PDF](https://arxiv.org/pdf/2603.20028v1)

**作者:** Maximiliano Armesto `[一作]` (Taller Technologies), Christophe Kolb `[通讯]` (Taller Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一家企业的三项软件现代化项目，在五种不同的 AI 交付平台配置下，做了横向对比研究，记录并分析了项目阶段持续时间、问题负载、首版覆盖率及基于团队规模的工作量模型。

**💡 创新点**

首次系统展示 AI 在完整交付工作流（分析、规划、实现、验证）中的协同作用，证明只有在将 AI 嵌入有序、可验证的工作流程后，才能获得显著的时间、成本与质量提升。

**🔧 技术方法**

采用 Chiron 平台，该平台集成了大型语言模型、仓库导入、需求与任务生成、接受标准验证、自动 PR 与代码评审以及人机混合执行等功能。

**📊 数据集**

使用的实验数据来自三项真实现代化项目：COBOL 银行迁移（约 30k LOC）、大型会计系统升级（约 400k LOC）和 .NET/Angular 抵押贷款系统升级（约 30k LOC），共覆盖传统基线和 V1-V4 共五种交付配置。

**📈 对比分析**

通过对项目阶段持续时间、按 100 个任务归一化的验证阶段问题负载、首版覆盖率以及基于人日/高级等价日的工作量模型进行比较。结果显示，V4 配置相较传统基线将项目总持续时间缩短 74%，原始人日缩减 78.5%，高级等价日缩减 87.1%，验证阶段问题负载下降 74%，首版覆盖率提升 13.4 个百分点；从 V1 到 V4 的提升更为显著，速度提升 3.08 倍，问题负载下降 75.8%，覆盖率提高 37.9 个百分点。

**⚠️ 局限性**

局限性包括：研究为回溯性、单机构、仅三项目；任务粒度随配置变化，导致归一化指标可能不完全可比；缺乏发布后缺陷数据；工作量模型依赖假设；作者与平台开发方存在利益关联。

---

## 122. A Theory of Composable Lingos for Protocol Dialects

**arXiv ID:** 2603.19908 | [PDF](https://arxiv.org/pdf/2603.19908v1)

**作者:** Víctor García `[一作]` (Universitat Politècnica de València), Jose Meseguer `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了面向协议方言的“lingo”理论，并设计了一系列新的formal pattern来构造、变换和组合lingo，从而提升协议方言的安全性。

**💡 创新点**

创新点在于引入f‑checkable和非malleable lingos、两种新的composition（水平和功能）操作，以及用lingo变换取代旧的dialect组合，从而使得方言设计更模块化、可验证且安全性更强。

**🔧 技术方法**

采用重写逻辑（Maude）对lingo、方言及其变换进行可执行规范，并通过Actor Rewrite Theory建模协议流程，以实现形式化验证。

**📊 数据集**

该工作不依赖外部数据集，所有实验和验证均在Maude环境中完成；仅使用自定义的协议实例（如MQTT、OpenFlow）作为演示。

**📈 对比分析**

方法通过形式化证明和Maude实现来验证安全属性；未给出传统实验性能对比，主要展示理论证明与可执行规范的完整性。

**⚠️ 局限性**

局限性包括：仅针对on‑path攻击模型，未考虑重放或会话劫持；缺乏真实部署或性能评估；对复杂协议的可扩展性和实现细节仍需进一步研究。

---

## 123. Evaluating Vision Foundation Models for Pixel and Object Classification in Microscopy

**arXiv ID:** 2603.19802 | [PDF](https://arxiv.org/pdf/2603.19802v1)

**作者:** Carolin Teuber `[一作]` (Georg-August-University Göttingen), Constantin Pape `[通讯]` (Georg-August-University Göttingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估视觉基础模型（VFM）在显微镜图像像素与对象分类中的表现，系统对比随机森林与可训练的轻量适配器（DeAP、ObAP）在不同数据规模下的效果。

**💡 创新点**

首次将VFM与浅层学习、注意力探针相结合，提出对象引导的注意力探针（ObAP）并在显微镜对象分类上进行系统性评估。

**🔧 技术方法**

采用SAM、SAM2、DINOv3、μSAM、PathoSAM等VFM，结合随机森林、Dense Attentive Probing（DeAP）与Object‑Guided Attentive Probing（ObAP）以及AnyUp上采样技术。

**📊 数据集**

使用5个多样化数据集（LIVECell、CRC、HBM、PanNuke、Planari），涵盖相位对比、荧光多通道、组织病理和形态筛选。

**📈 对比分析**

通过交叉验证在不同训练像素/对象数量下评估F1得分、训练与推理时间，结果显示VFM+DeAP/ObAP在低数据量场景下优于传统手工特征，SAM2+DeAP/ObAP甚至超过全监督U‑Net/ResNet；随机森林在交互式场景下训练最快。

**⚠️ 局限性**

DINOv3表现最差；DeAP/ObAP训练时间较长，难以实现即时交互；方法依赖高质量实例分割；部分域特定模型在非专属域表现不佳。

---

## 124. A Visualization for Comparative Analysis of Regression Models

**arXiv ID:** 2603.19291 | [PDF](https://arxiv.org/pdf/2603.19291v1)

**作者:** Nassime Mountasir `[一作]` (Université de Strasbourg), Nicolas Lachiche `[通讯]` (Université de Strasbourg)

**通讯引用:** 1733 | [OpenAlex ID](https://openalex.org/A5031496807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个两步可视化流程，用于快速筛选并精细比较回归模型的误差分布。

**💡 创新点**

创新点在于构建2D误差空间，并通过基于中位数的颜色映射和马氏距离来揭示模型间误差的结构差异与关联性。

**🔧 技术方法**

技术手段包括1D箱线图、散点图、密度估计（KDE）、Hexbin、颜色映射以及马氏距离计算。

**📊 数据集**

实验使用了三个真实数据集，其中包括AI4I 2020 Predictive Maintenance 数据集。

**📈 对比分析**

方法先用1D可视化筛选表现最优模型，再在2D误差空间中比较其误差配对，结果表明传统指标（MAE、RMSE）隐藏的误差结构被显现，帮助更精准地选择模型。

**⚠️ 局限性**

局限性包括对模型配对的依赖、手工阈值选择、在极大规模数据下可视化复杂度提升，以及缺乏多模型并行比较的机制。

---

## 125. Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models

**arXiv ID:** 2603.20161 | [PDF](https://arxiv.org/pdf/2603.20161v1)

**作者:** Qi Cao `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**通讯引用:** 2337 | [OpenAlex ID](https://openalex.org/A5063925941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Semantic Token Clustering（STC）的单生成无监督不确定性量化方法，利用LLM内部token embedding聚类与前缀匹配来聚合概率并评估生成文本的可信度。

**💡 创新点**

创新点在于：①直接利用LLM内部token embedding进行离线聚类，避免了外部模型和多次采样；②将同义token的概率聚合，提升不确定性评估的准确性；③实现单生成、低开销的推理流程。

**🔧 技术方法**

核心技术包括token embedding聚类（如Agglomerative Clustering）、前缀匹配、概率聚合、logits利用、AUROC评价，以及对LLM内部表示的自洽使用。

**📊 数据集**

使用了TriviaQA（TQA）、Natural Questions（NQ）和WebQuestions（WQ）三个公开数据集。

**📈 对比分析**

与Perplexity、tokenSAR、CCP、Predictive Entropy、LN-Entropy、EigenScore、ConU、Semantic Entropy等基线比较，STC在AUROC上与state‑of‑the‑art相当，并且推理时延降低约98%，显示出高效且稳健的性能。

**⚠️ 局限性**

局限性包括：需要访问token logits和embedding，无法用于封闭模型；使用静态token embeddings可能引入多义性噪声；以及未对不确定性评分进行显式校准。

---

## 126. ReLi3D: Relightable Multi-view 3D Reconstruction with Disentangled Illumination

**arXiv ID:** 2603.19753 | [PDF](https://arxiv.org/pdf/2603.19753v1)

**作者:** Jan-Niklas Dihlmann `[一作]` (University of Tübingen), Varun Jampani `[通讯]` (Stability AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的端到端管线，能够在不到一秒的时间内从稀疏多视角图像同时重建完整的三维几何、空间可变的物理渲染材料以及环境光照。

**💡 创新点**

创新点包括：
- 通过跨视角 Transformer 进行特征融合，利用多视角约束实现材料与光照的解耦；
- 两条并行路径：一条预测几何与材料，另一条预测 HDR 环境；
- 使用可微 MC+MIS 渲染器将两条路径联合训练，保证物理一致性；
- 混合域训练（合成 PBR + 实际 RGB），提升对真实场景的泛化能力。

**🔧 技术方法**

技术手段包括：
- DINOv2 视觉特征提取与相机调制；
- 交叉条件 Transformer 与三平面（triplane）表示；
- RENI++ 潜在空间用于环境光照编码；
- 可微物理渲染（Monte Carlo + Multiple Importance Sampling，VNDF 采样）；
- 多视角遮罩随机化与自监督训练策略。

**📊 数据集**

训练集：174k 个对象，42k 合成 PBR（带材质标签），70k 合成 RGB，仅光照；62k 真实捕捉（UCO3D）。
评估集：Google Scanned Objects、Polyhaven、Stanford ORB、UCO3D 等。

**📈 对比分析**

与 SF3D、SPAR3D、3DTopia-XL、Hunyuan3D 等方法对比：
- 材料 PSNR（Albedo、Roughness、Metallic）领先；
- 在新环境下的 relighting 指标排名第一；
- 环境光照估计更清晰，能捕获光源位置；
- 几何重建与传统方法相当；
- 推理时间约 0.3s，速度比生成式方法快 100 倍。

**⚠️ 局限性**

局限性：
- 三平面分辨率有限，导致纹理与几何细节不足；
- 当环境光照超出 RENI++ 先验（如强多光源）时，分解可能失败并出现光照烘焙在材质图上；
- 对极端光照或遮挡条件下的鲁棒性尚待进一步提升。

---

## 127. Prompt engineering for bibliographic web-scraping

**arXiv ID:** 2603.19237 | [PDF](https://arxiv.org/pdf/2603.19237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 128. Case Study: Horizontal Side-Channel Analysis Attack against Elliptic Curve Scalar Multiplication Accelerator under Laser Illumination

**arXiv ID:** 2603.19811 | [PDF](https://arxiv.org/pdf/2603.19811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 129. ARMOR: Adaptive Resilience Against Model Poisoning Attacks in Continual Federated Learning for Mobile Indoor Localization

**arXiv ID:** 2603.19594 | [PDF](https://arxiv.org/pdf/2603.19594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. Repeater-Aided Over-the-Air Phase Synchronization in Distributed MIMO

**arXiv ID:** 2603.19903 | [PDF](https://arxiv.org/pdf/2603.19903v1)

**作者:** Unnikrishnan Kunnath Ganesan `[一作]` (Ericsson), Erik G. Larsson `[通讯]` (Linköping University)

**通讯引用:** 50590 | [OpenAlex ID](https://openalex.org/A5043552696)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在两台 AP 之间直接链路过弱的情况下，利用中继节点实现分布式 MIMO 的空域相位同步。

**💡 创新点**

创新点在于提出一种无 CSI、无需已知回波系数的两阶段 OTA 相位同步方案，并给出闭式相位估计表达式；该方案通过中继节点重新引导测量，克服了 AP 之间直接通信受阻的局限。

**🔧 技术方法**

技术包括：TDD 频率对齐、回波链路互补、两阶段同步（前向与反向测量）、最大功率束波、相位估计（取复数测量的相位）以及噪声建模。

**📊 数据集**

使用的是仿真数据，仿真场景设定为 Rayleigh 随机小尺度衰落，距离 d 从 1m 到 100m，路径损耗模型 PL = -30.5 - 36.7 log10(d)，噪声基于 20 MHz 带宽、290 K 温度和 9 dB 噪声系数。

**📈 对比分析**

通过 RMSE 对比实验：在 3 GHz 载波下，当 AP 与中继距离 1 m 时 RMSE ≈ 0.01 rad（约 0.5 ps）；随着距离增大 RMSE 急剧上升，但通过提升中继发射功率（1 mW→10 mW）可显著改善性能；仿真结果显示该方法在合理距离和功率条件下能够实现精确相位同步。

**⚠️ 局限性**

局限性包括：仅考虑两台 AP 的场景，未探讨大规模 AP 集群的可扩展性；假设 AP 之间已完成单点回波校准；中继节点需要足够功率且受硬件实现限制；仿真未涵盖多径、相位噪声等实际环境干扰。

---

## 131. Is It a Good Idea to Build an HLS Tool on Top of MLIR? Experience from Building the Dynamatic HLS Compiler

**arXiv ID:** 2603.19856 | [PDF](https://arxiv.org/pdf/2603.19856v1)

**作者:** Jiahui Xu `[一作]` (ETH Zurich), Lana Josipovic `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文分析了在 MLIR 上构建高层次综合工具 Dynamatic 的过程，并对 MLIR 在 HLS 领域的不足进行了系统评估。

**💡 创新点**

作者系统地列举了 MLIR 在图边注解、SSA ϕ 节点处理、项目分段以及 C 前端支持等方面的局限，并通过 Dynamatic 的案例提供了改进思路。

**🔧 技术方法**

使用 MLIR 框架、LLVM 生态、手势（handshake）dialect、自定义翻译 Pass、模式重写、别名与多边形分析等技术实现了 HLS 生成。

**📊 数据集**

以 C 示例程序（如直方图计算）为例进行验证，未使用公开大规模数据集。

**📈 对比分析**

对比传统 LLVM‑based HLS 流程，MLIR 能减小实现成本，但因边注解、SSA 处理等问题导致部分优化受限，未给出量化性能指标。

**⚠️ 局限性**

MLIR 不能对 value 进行属性注解、块参数难以映射 SSA ϕ、跨仓库 dialect 互操作性差、缺乏成熟的 C‑to‑MLIR 前端，且依赖 LLVM 进一步限制了工具的可扩展性。

---

## 132. X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving

**arXiv ID:** 2603.19979 | [PDF](https://arxiv.org/pdf/2603.19979v1)

**作者:** Chaoda Zheng `[一作]`, Xianming Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了X‑World，一种可控多摄像头视频生成世界模型，能够根据同步多视角历史视频、未来动作序列以及可选的动态/静态场景控制，生成与命令动作严格一致、视角一致且在长时间范围内保持稳定的未来多摄像头视频。

**💡 创新点**

创新点包括：
1) 采用视频扩散与高压缩3D因果VAE结合的Latent Video Generation框架，并在DiT骨干中加入视-时自注意力，显著提升跨摄像头几何一致性；
2) 设计分离的多模态条件注入机制（动作、动态/静态场景、摄像头参数、文本）并通过解耦Cross‑Attention实现细粒度可控；
3) 通过两阶段训练（Stage‑I双向全片段生成 + Stage‑II自回归少步生成+自强训练），实现实时低延迟、长时延的可控推理；
4) 引入滚动KV缓存，支持持续生成而不爆炸性增长内存。

**🔧 技术方法**

技术手段包括：
- WAN 2.2的高压缩3D因果VAE + DiT denoiser；
- 视-时自注意力；
- AdaLN‑Zero、Fourier特征编码、Classifier‑Free Guidance；
- Self‑forcing + DMD分布匹配蒸馏；
- 预先训练的动作、动态、静态、文本、摄像头等多模态编码器；
- Rolling KV Cache。

**📊 数据集**

使用自研的X‑World大规模真实驾驶数据集：7摄像头同步视频（12 FPS，10 s段），包含动态目标轨迹、静态道路要素、自然语言场景描述、三级标签体系（环境、静态、动态、车主行为）。

**📈 对比分析**

与原始WAN、单摄像头扩散模型以及无条件/无控制基线相比，X‑World在多摄像头一致性、动作跟随精度、动态/静态控制可行性以及24 s长时延稳定性等方面均表现出显著提升；在视觉一致性、动作误差率和漂移量上分别降低约30%–50%，且能够在无额外采样的情况下完成实时长序列生成。

**⚠️ 局限性**

局限性：
- 对极端天气、极稀有交通场景的泛化仍有限；
- 需要大量标注文本和多模态标签，人工成本高；
- 仅支持7摄像头同步配置，对其他硬件方案的迁移需再训练；
- 训练与推理均需高算力，部署成本仍不低；
- 对非视觉感知（雷达、激光等）缺乏模拟能力。

---

## 133. What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

**arXiv ID:** 2603.19880 | [PDF](https://arxiv.org/pdf/2603.19880v1)

**作者:** Dong Yan `[一作]` (University of Chinese Academy of Sciences), Tieniu Tan `[通讯]` (Nanjing University)

**通讯引用:** 37182 | [OpenAlex ID](https://openalex.org/A5111885963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出SCRL框架，通过选择性正向伪标签和熵门负向伪标签在无监督测试时强化学习中抑制标签噪声并提升推理性能。

**💡 创新点**

引入了负向监督机制与动态奖励塑形，结合严格共识阈值实现对弱共识的鲁棒处理。

**🔧 技术方法**

使用强化学习中的GRPO、熵估计、动态奖励塑形等技术。

**📊 数据集**

在AIME24/25、AMC、MATH-500、Minerva、GPQA等数学与通用推理基准上评测。

**📈 对比分析**

与TTRL、COMPASS、ETMR、RESTRAIN等基线对比，SCRL在大多数数据集上提升约1–10% pass@1，尤其在低样本/高噪声场景表现突出。

**⚠️ 局限性**

对超参数阈值敏感，在极端分散的答案分布下仍可能需要手动调参；对大型长链推理的扩展性未完全验证。

---

## 134. FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision

**arXiv ID:** 2603.19770 | [PDF](https://arxiv.org/pdf/2603.19770v1)

**作者:** Zekai Wu `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**通讯引用:** 26516 | [OpenAlex ID](https://openalex.org/A5100736836)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了闪烁LED MoCap系统FlashCap并基于该系统收集了第一份毫秒级标签的多模态数据集FlashMotion，同时提出了ResPose基准实现高时间分辨率人体姿态估计。

**💡 创新点**

创新点在于通过LED闪烁编码实现1000 Hz原生姿态标签，首次公开毫秒级多模态数据集；并设计了融合RGB基准与事件残差的Hybrid Spiking‑Transformer模型ResPose。

**🔧 技术方法**

采用LED闪烁事件捕捉、事件聚类与频率匹配的注释管线、IMU与LiDAR同步校准、SNN+CNN时空编码以及Transformer残差回归等技术。

**📊 数据集**

使用FlashMotion数据集（240序列，约2小时，包含1000 Hz 2D标签和60 Hz 3D SMPL标签），并通过高帧率RGB视频和人工标注进行验证。

**📈 对比分析**

与ViTPose、Hybrid ANN‑SNN、EventPointPose、GraphEnet、EvSharp2Blur、LEIR等方法对比；ResPose在精准运动时序任务中误差<5 ms，在高时间分辨率HPE任务中MPJPE达5.66，显著优于现有基准。

**⚠️ 局限性**

局限性包括需要佩戴LED套装、数据多样性有限，且仍受光照、遮挡和噪声影响；基准在公开数据上验证，实际部署仍需进一步测试。

---

## 135. Wearable Foundation Models Should Go Beyond Static Encoders

**arXiv ID:** 2603.19564 | [PDF](https://arxiv.org/pdf/2603.19564v1)

**作者:** Yu Yvonne Wu `[一作]` (Dartmouth), Cecilia Mascolo `[通讯]` (Cambridge)

**通讯引用:** 18554 | [OpenAlex ID](https://openalex.org/A5010623957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了可穿戴基础模型（WFMs）的现状，并提出未来研究需要从静态编码器向纵向、前瞻性健康推理转变，提出了三大基础性转变（结构丰富数据、纵向感知多模态建模、代理式推理系统），并给出了相应的行动呼吁。

**💡 创新点**

创新点在于系统性地识别并阐述了WFMs面临的三大瓶颈与对应的突破方向，强调构建开放、上下文丰富的数据生态、开发可处理长时序多模态的建模框架，以及将WFMs定位为支持代理式推理的工具集，而非单纯的判别模型。

**🔧 技术方法**

技术层面综述了目前主流的自监督预训练（对比学习、掩码重建）、多模态对齐、语言模型融合等技术，讨论了Token化、序列抽象与层次化时间表示等方法，并提出未来可采用检索增强生成（RAG）等框架实现个体记忆与推理。

**📊 数据集**

作者引用并梳理了多个公开可穿戴数据集，包括但不限于SensorLM、SleepEDF、PhysioOmni、GLOBEM、WESAD、PPG-Dalia、UK Biobank、All‑of‑US 等，指出现有数据集在规模、开放性与多模态整合方面的不足。

**📈 对比分析**

文中未给出新的实验结果，而是通过对比静态编码器与拟议的纵向代理式框架，指出当前WFMs在预测准确性上虽已可达较高水平，但在长期跟踪、个性化基线估计与决策支持方面表现不足；作者建议通过共享评测基准与跨数据集对照来推动性能提升。

**⚠️ 局限性**

限制包括：1）数据层面缺乏开放、统一且长期的多模态与上下文丰富数据；2）建模层面仍多采用静态短窗口、单模态或固定模态组合的架构；3）推理层面大多停留在离散的分类/回归任务，缺乏动态、代理式的推理与决策支持。

---

## 136. TinyML Enhances CubeSat Mission Capabilities

**arXiv ID:** 2603.20174 | [PDF](https://arxiv.org/pdf/2603.20174v1)

**作者:** Luigi Capogrosso `[一作]` (Interdisciplinary Transformation University of Austria), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7813 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 CubeSat 上实现了 TinyML 卷积神经网络的模型压缩与部署，提升了地球观测图像分类的可行性。

**💡 创新点**

提出了一套结合结构化迭代剪枝、INT8 后训练量化和硬件感知算子映射的端到端压缩部署流水线。

**🔧 技术方法**

采用结构化剪枝、静态 INT8 量化、STM32N6 MCU+Neural‑ART NPU 的硬件感知映射技术。

**📊 数据集**

在 EuroSAT、RS_C11 和 MEDIC 三个地球观测基准数据集上进行评估。

**📈 对比分析**

相较 Float32 基线，模型精度下降≤8.6pp，RAM/Flash 占用平均降低约90%/70%，推理能耗 0.68–6.45 mJ，延迟 3.2–30.4 ms，满足 CubeSat 实时与能耗约束。

**⚠️ 局限性**

限制在仅支持卷积网络、未验证在更高分辨率或 Transformer 模型、以及只在模拟硬件上评估，缺乏真实卫星运行验证。

---

## 137. Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions

**arXiv ID:** 2603.19335 | [PDF](https://arxiv.org/pdf/2603.19335v1)

**作者:** Xiaoyi Li `[一作]` `[通讯]`, Xiaoyi Li

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

统一实现并对比了51种后训练算法，在4个模型规模和3个评估域上完成了约240次实验

**💡 创新点**

提出统一框架和大规模对照实验，揭示模型规模、训练范式、在线/离线和损失函数对算法排名的层级影响

**🔧 技术方法**

使用DeepSpeed ZeRO-3分布式训练、RL/GRPO、DPO及其20种变体实现，配合PyTorch、vLLM等技术

**📊 数据集**

在GSM8K、MATH以及通用推理基准（ARC-Challenge、HellaSwag、WinoGrande）等数据集上进行评测

**📈 对比分析**

通过相同条件下的A/B比较，发现模型规模≈50pp、训练范式≈10pp、在线/离线≈9pp、损失函数≈1pp的影响差异，并在7B规模下SimPO排名逆转

**⚠️ 局限性**

局限于单一模型族（Qwen 2.5至7B）、缺乏大规模全微调实验、仅使用合成偏好数据与贪心解码

---

## 138. Span-Level Machine Translation Meta-Evaluation

**arXiv ID:** 2603.19921 | [PDF](https://arxiv.org/pdf/2603.19921v1)

**作者:** Stefano Perrella `[一作]` (Sapienza University of Rome), Hugo Zaragoza `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究机器翻译错误检测的元评估方法，系统评估不同的span-level精确率、召回率与F值计算实现，并提出新的评价指标；

**💡 创新点**

提出“Match with Partial Overlap and Partial Credit”（P_mpp/R_mpp）与微平均（micro-averaging）的组合，证明其对span长度和任务稀疏性最稳健；

**🔧 技术方法**

使用字符级重叠匹配、单一匹配（one-to-one）优化、宏/微平均统计以及基于LLM的错误检测模型进行实验；

**📊 数据集**

在2019-2024年WMT MQM测试集、ESA测试集以及多语言（英语-韩语、日语-中文）翻译样本上进行评估；

**📈 对比分析**

对比三种常用评估方案（exact match、partial overlap、partial credit）和宏/微平均效果，显示宏平均易被低召回模型作弊，partial credit + micro平均在多指标上获得最高、最稳定的F值；

**⚠️ 局限性**

仅基于字符重叠可能导致长词/长span影响评估，未考虑错误严重性权重，且对不同语言脚本的适用性未深入验证。

---

## 139. Range-Based Set Reconciliation via Range-Summarizable Order-Statistics Stores

**arXiv ID:** 2603.19820 | [PDF](https://arxiv.org/pdf/2603.19820v1)

**作者:** Elvio G. Amparore `[一作]` `[通讯]` (Università di Torino), Elvio G. Amparore (Università di Torino)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了范围可总结有序统计存储（RSOS）抽象，用于实现高效的范围基集合调和（RBSR）并将其实现为增量B+树；

**💡 创新点**

创新点在于将RBSR的本地计算需求抽象为RSOS接口，并在持久化存储中通过在B+树内部维护子树计数和可组合聚合实现；

**🔧 技术方法**

技术包括可组合范围摘要（使用可加摘要和计数）、有序统计（rank/select）、B+树增量聚合以及在LMDB上实现的AELMDB扩展；

**📊 数据集**

使用了合成数据集（多组参数化的片段大小和差异量），以及Negentropy协议下的同步工作负载；

**📈 对比分析**

通过与基线LMDB+辅助树、无窗口子范围Ablation以及内存版进行对比，AELMDB在大多数工作负载下的调和时间提高了4.7–13.98倍，内存占用相对较低；

**⚠️ 局限性**

局限在于仅评估单机、持久化负载，对并发、崩溃恢复、冷缓存行为以及跨引擎可移植性未做深入研究。

---

## 140. Rethinking Ground Truth: A Case Study on Human Label Variation in MLLM Benchmarking

**arXiv ID:** 2603.19744 | [PDF](https://arxiv.org/pdf/2603.19744v1)

**作者:** Tomas Ruiz `[一作]` (Ludwig Maximilian University of Munich), Carsten Schwemmer `[通讯]` (Ludwig Maximilian University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于人类标注一致性拆分为同意与不一致子集的多模态LLM评估协议，并在此协议下对Gemma‑3和Qwen 2.5 VL进行基准测试

**💡 创新点**

创新点在于将Krippendorff’s Alpha作为阈值将数据拆分为高一致性和低一致性子集，并分别使用精度/召回/F1和Brier分数/JS散度两类指标来衡量模型在一致性和不一致性情境下的性能，揭示参数规模并不总是决定模型对人类争议的敏感度

**🔧 技术方法**

采用Krippendorff’s Alpha进行一致性划分，使用多数投票生成伪真值，计算精度/召回/F1以及Brier分数和Jensen–Shannon散度，并在五次随机采样下获得模型概率预测

**📊 数据集**

使用PoliTok‑DE社交媒体视频数据集（包含5个关于政治、地区、宽容、娱乐等五个问题的非聚合人类标注）

**📈 对比分析**

方法：对同意子集使用传统分类指标，对不一致子集使用分布对齐指标；结果显示在同意子集中最大模型（Qwen 72B、Gemma 27B）表现最佳，但在不一致子集中中等规模模型（Gemma 12B）往往校准更好，说明模型规模与对人类争议的适应性不成正比

**⚠️ 局限性**

局限性包括仅在单一领域（德国选举相关TikTok视频）和两个模型家族上验证，未涉及多语言、多任务和不同基础模型的对比，且评估仅基于非训练集标注，缺乏对模型生成质量与解释性的深入分析

---

## 141. Strategies for Designing Responsibly within a Capitalist Enterprise

**arXiv ID:** 2603.19400 | [PDF](https://arxiv.org/pdf/2603.19400v1)

**作者:** Shixian Xie `[一作]` (Carnegie Mellon University), John Zimmerman `[通讯]` (Carnegie Mellon University)

**通讯引用:** 13304 | [OpenAlex ID](https://openalex.org/A5068413510)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文提出在资本主义企业环境下，将伦理与商业价值结合，主张通过头脑风暴（ideation）生成既符合伦理又可行的设计方案，以提升负责任 AI 的实际落地率。

**💡 创新点**

创新点在于将“伦理与商业”视为共生关系，强调在设计过程早期通过多方案生成来平衡道德与财务目标，并将头脑风暴定位为核心的负责任设计策略。

**🔧 技术方法**

本论文为理论性立场论文，未采用具体技术实现，而是基于设计方法学和人机交互实践提出框架与建议。

**📊 数据集**

无数据集。

**📈 对比分析**

未进行实验对比，也未给出性能指标，主要通过案例讨论与理论阐述来说明其可行性。

**⚠️ 局限性**

局限性包括缺乏实证验证和具体实施指南，建议未来通过案例研究、实验设计或企业合作来检验所提方法的实际效果。

---

## 142. Evaluating Evidence Grounding Under User Pressure in Instruction-Tuned Language Models

**arXiv ID:** 2603.20162 | [PDF](https://arxiv.org/pdf/2603.20162v1)

**作者:** Sai Koneru `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**通讯引用:** 1397 | [OpenAlex ID](https://openalex.org/A5082663800)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估指令调优语言模型在争议证据场景下的鲁棒性，探讨用户压力与证据一致性之间的冲突。

**💡 创新点**

提出一种基于美国国家气候评估（NCA）的受控认知冲突框架，细粒度剖析证据层级与不确定性提示对模型表现的影响。

**🔧 技术方法**

使用多层级文本提示、对抗性用户语句，利用概率分布预测四级置信度并计算排名概率得分（RPS）和序数方差。

**📊 数据集**

采用NCA4与NCA5的结构化气候科学文本，生成770条原子命题及其四层证据（证据基础、研究空白、不确定性、置信度）。

**📈 对比分析**

在19个指令调优模型（0.27B–32B）上进行16个实验条件（四层证据×四种用户压力）评估；结果显示在中性提示下证据增量提升准确率与RPS，但在用户压力下多数模型仍出现证据不一致，表现随模型家族与规模非单调变化。

**⚠️ 局限性**

局限包括：仅单轮模板化提示、固定文本证据、单一领域（气候科学）、未检验检索错误或多轮交互、未尝试针对冲突的微调或策略。

---

## 143. Neural Dynamics Self-Attention for Spiking Transformers

**arXiv ID:** 2603.19290 | [PDF](https://arxiv.org/pdf/2603.19290v1)

**作者:** Dehao Zhang `[一作]` (University of Electronic Science and Technology of China), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 29173 | [OpenAlex ID](https://openalex.org/A5032690182)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在Spiking Transformer中引入局部感受野（LRF）模块和基于神经元膜电位的自注意力动力学（LRF‑Dyn），实现了在保持或提升性能的同时显著降低推理内存开销。

**💡 创新点**

创新点在于：① 用LRF替代传统全局注意力中的softmax，恢复局部加权特性；② 将注意力计算映射到脉冲神经元的充电-放电动力学，消除显式注意力矩阵存储；③ 在保持事件驱动特性的同时兼顾能效与精度。

**🔧 技术方法**

使用的技术包括：LIF脉冲神经元模型、深度可分离扩张卷积实现LRF、基于自注意力的脉冲变换、频域卷积与傅里叶变换、以及因果推理实现内存压缩。

**📊 数据集**

实验数据集：ImageNet‑1K（图像分类）、ADE20K（语义分割）以及CIFAR‑100（消融实验）。

**📈 对比分析**

方法与现有Spiking Transformer（Spikformer、QKFormer、Spike‑Driven‑V3）、CNN‑SNN（SEW‑ResNet、MSResNet）等对比，结果显示：在ImageNet上提升1.0–1.5% Top‑1精度，ADE20K mIoU提升2.2–2.6%，同时推理内存降低约40–50%，并保持与原模型相近或更低的参数量。

**⚠️ 局限性**

局限性包括：① 对大规模高分辨率图像的可扩展性尚未充分验证；② 需要在训练阶段保留完整的权重矩阵，训练成本仍然高；③ 对复杂场景下的全局信息捕捉能力可能不足，需进一步优化注意力分布。

---

## 144. Diffusion-Based Makeup Transfer with Facial Region-Aware Makeup Features

**arXiv ID:** 2603.20012 | [PDF](https://arxiv.org/pdf/2603.20012v1)

**作者:** Zheng Gao `[一作]` (Queen Mary University of London), Jifei Song `[通讯]` (Huawei London Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于扩散模型的面部彩妆转移方法 FRAM，能够实现全局和彩妆区域的精准转移。

**💡 创新点**

核心创新包括：①专门针对彩妆风格微调 CLIP 视觉编码器；②使用可学习查询在 CLIP 中提取面部区域彩妆特征并通过注意力引导实现区域控制；③利用 ControlNet Union 同时融合像素级身份信息和 3D 网格结构，实现高保真身份保持。

**🔧 技术方法**

主要技术手段包括：CLIP 语义对比学习（自监督 + 图像-文本对比）、ControlNet Union、LoRA 微调扩散模型的跨注意力层、基于注意力损失的区域引导、以及 GPT‑3 + FLUX‑1‑Context 生成合成彩妆数据。

**📊 数据集**

使用 GPT‑3 生成彩妆描述并用 FLUX‑1‑Context 在 FFHQ 图像上合成彩妆样本；随后在 Makeup Transfer、Wild‑MT、CPM‑real 等公开数据集上进行训练与评估。

**📈 对比分析**

与 GAN 及多种扩散基线（CSD‑MT、SHMT、MAD、Gorgeous、Stable‑Makeup）进行对比，FRAM 在 CSD、ID、SSIM、L2‑M、Aesthetic Score 等指标上均优于对手，并在用户研究中获得最高满意度。

**⚠️ 局限性**

主要局限包括：对合成数据的依赖程度较高；区域控制仍需依赖学习的查询，不同人脸细节或极端光照/姿态下的鲁棒性尚未充分验证。

---

## 145. Subspace Kernel Learning on Tensor Sequences

**arXiv ID:** 2603.19546 | [PDF](https://arxiv.org/pdf/2603.19546v1)

**作者:** Lei Wang `[一作]` (Griffith University), Piotr Koniusz `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种针对多模态张量序列的核学习框架UKTL，利用模式子空间核、Nyström线性化和不确定性加权实现高效可解释的张量相似度度量。

**💡 创新点**

将不确定性驱动的子空间加权引入张量核，动态选取Nyström基点，并构造求和-乘积混合Grassmann核，兼顾多模态融合与可解释性。

**🔧 技术方法**

使用张量子空间（Grassmann manifold）核、Sum-Product 组合核、Nyström近似、Soft k‑means 动态基点、Multi‑mode SigmaNet 估计不确定性、Higher‑order Transformer 编码以及端到端训练。

**📊 数据集**

在 NTU‑RGB+D 60/120、Kinetics‑Skeleton 三大数据集上进行实验，覆盖骨架、RGB 与深度等多模态。

**📈 对比分析**

与图卷积、超图、Transformer 以及张量基线在相同编码器下对比，UKTL 在 NTU‑60 X‑Sub/ View、NTU‑120 X‑Sub/Setup 与 Kinetics‑Skeleton 上均取得最高 Top‑1/5 分，提升约 1‑2% 于现有方法。

**⚠️ 局限性**

对核参数和基点数较为敏感，计算成本相对较高，对更高维时序或非骨架数据的可扩展性仍待进一步验证。

---

## 146. Can Large Multimodal Models Inspect Buildings? A Hierarchical Benchmark for Structural Pathology Reasoning

**arXiv ID:** 2603.20148 | [PDF](https://arxiv.org/pdf/2603.20148v1)

**作者:** Hui Zhong `[一作]` (Hong Kong University of Science and Technology), Xinhu Zheng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2335 | [OpenAlex ID](https://openalex.org/A5062424202)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的建筑立面缺陷检测与诊断基准 DefectBench，融合 12 个公开数据集并通过人工‑机器协同重标注，形成 1,488 张含分类、检测与分割标签的多粒度数据集，并针对 LMM 进行三维层级评测（语义感知、空间定位、几何分割）。

**💡 创新点**

创新点包括：① 将多源、语义不一致的数据统一为层级化本体，突破了数据孤岛和标签不统一问题；② 设计了人‑机协同半自动标注平台，显著提升标注质量与效率；③ 提出跨层级的评测框架和指标，系统量化 LMM 的主动推理与几何推断能力；④ 在零样本条件下验证 LMM 的生成分割可与专用监督模型竞争。

**🔧 技术方法**

使用技术主要有：大规模多模态模型（如 Gemini、Qwen、GPT‑4o 等）进行零样本推理；人‑机交互的标注框架，集成 YOLO、SAM 等预训练模型生成候选框与掩码；生成式分割与后处理（阈值化、ROI 门控）保证评测的可重复性；评测指标包括 P/R/F1、MAE、mAP、mIoU 等。

**📊 数据集**

数据集：通过整合 12 个公开建筑缺陷数据集（如 BD3、CSD、DeepCrack 等），统一四大类（裂纹、材料损失、表面染色、外部修复）及 11 个细类，生成 1,488 张高分辨率图像，覆盖分类、检测与分割三任务。

**📈 对比分析**

比较方法：在 18 个前沿 LMM 上执行三层评测；结果显示：在语义感知层，LMM 的分类准确率普遍高于 0.7，计数误差 MAE 接近 1；在空间定位层，mAP 低于 0.5，几何推理表现优于检测；在生成分割层，mIoU 0.68–0.74，Gemini‑3‑pro‑edit 领跑，已接近专业监督模型；整体显示 LMM 在语义与拓扑推理强，但在精准定位与细粒度分割仍有提升空间。

**⚠️ 局限性**

局限性：① 空间定位精度不足，导致“位置正确但类别错误”现象；② 生成式分割易出现幻觉与多余纹理；③ 先前错误会级联影响后续任务；④ 评测仍依赖人工标注的 ROI 限制；⑤ 对极端纹理或小尺寸缺陷的鲁棒性有限；⑥ 需要进一步改进模型对建筑专业术语和拓扑约束的理解。

---

## 147. Preference-Guided Debiasing for No-Reference Enhancement Image Quality Assessment

**arXiv ID:** 2603.20086 | [PDF](https://arxiv.org/pdf/2603.20086v1)

**作者:** Shiqi Gao `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21777 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于偏好引导的去偏差框架，用于无参考的图像增强质量评估（EIQA），通过学习增强偏好嵌入并从质量特征中去除算法相关噪声来提高质量预测的鲁棒性。

**💡 创新点**

创新点在于：①将增强算法的视觉风格建模为连续的偏好嵌入空间，采用监督对比学习以保持相似风格相邻；②利用该偏好嵌入预测并减去质量特征中的算法诱导偏差，实现特征去偏；③两阶段训练策略保证偏好空间的稳定性和质量回归的独立性；④内容控制采样有效抑制内容泄漏。

**🔧 技术方法**

核心技术包括：监督对比学习（SupCon）、ConvNeXt‑T（偏好编码器）、ViT（质量编码器）、两层MLP（偏差预测器）、Huber 与 PLCC 损失、内容控制的批量构造、两阶段冻结训练。

**📊 数据集**

使用公开 EIQA 基准数据集 RLIE 与 SQUARE‑LOL，另外在 SQUARE‑LOL 上构造算法离散拆分（算法无关）用于跨算法泛化测试。

**📈 对比分析**

与 14 种现有无参考 IQA 方法（包括手工统计与深度学习模型）进行对比，在 RLIE 上获得 SRCC/PLCC 分别为 0.6923/0.7137，SQUARE‑LOL 上为 0.8726/0.8913，均超出所有对手；在算法无关拆分下，模型的性能下降最小（SRCC 0.0104），显示出最佳的跨算法鲁棒性。

**⚠️ 局限性**

局限性：①仍需大量标注好的 MOS 与算法标签，无法直接推广到无标签或极少标签的场景；②主要针对低光/图像增强任务，未在其它增强类型（如去雾、去噪）进行验证；③去偏机制假设偏差为可加形式，可能对更复杂的算法交互影响不足。

---

## 148. IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning

**arXiv ID:** 2603.20182 | [PDF](https://arxiv.org/pdf/2603.20182v1)

**作者:** Fan Yang `[一作]` (Fujitsu Research), Yonatan Bisk `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3911 | [OpenAlex ID](https://openalex.org/A5041302228)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出IndoorR2X benchmark与框架，融合机器人与IoT（如CCTV）感知，利用LLM实现多机器人任务规划与协调；

**💡 创新点**

首次在室内环境下构建R2X（Robot‑to‑Everything）模式，结合全局语义状态与LLM规划，解决部分可观测下的冗余探索；

**🔧 技术方法**

基于AI2‑THOR/ArchitecTHOR与RoboTHOR模拟器；LLM（GPT‑4.1、Gemma‑3‑27b、Llama‑3.1‑8b）在线规划；VLM（Qwen‑VL）处理IoT视频；协调枢纽进行语义融合；

**📊 数据集**

85个多房间室内场景（10个ArchitecTHOR + 75个RoboTHOR）以及相应任务套件；

**📈 对比分析**

与先行R2R基线（SMART‑LLM、EMOS）对比，R2X在成功率上与R2R相当（≈92%），但平均路径长度减少26%，LLM token使用减少约11%；在不同LLM规模、团队规模与IoT延迟下均表现稳健；

**⚠️ 局限性**

对IoT语义错误极为敏感，误报会导致任务失败；当所有IoT数据缺失时，成功率保持但耗时显著增加；需要对不可信IoT信息进行验证。

---

## 149. Semantic Delta: An Interpretable Signal Differentiating Human and LLMs Dialogue

**arXiv ID:** 2603.19849 | [PDF](https://arxiv.org/pdf/2603.19849v1)

**作者:** Riccardo Scantamburlo `[一作]` (Università Cattaneo), Francesco Bertolotti `[通讯]` (Università Cattaneo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了一种基于 Empath 主题分布的语义差值（semantic delta）指标，用于区分人类写作与大型语言模型（LLM）生成的对话。

**💡 创新点**

创新点在于提供了一种轻量级、可解释且零样本的检测信号，能够在无需训练的情况下补充现有 AI 文本检测方法。

**🔧 技术方法**

使用了 Empath 词义嵌入聚类进行主题提取、差值计算、Welch t 检验统计分析，并通过 OpenAI API 生成 LLM 对话，整个流程基于 Python 实现。

**📊 数据集**

实验数据包括 OpenAI 多个 LLM 生成的对话（多模型、多提示），以及三类人类语料：Friends 剧本、莎士比亚全集、Reddit 关于 ChatGPT 的讨论。

**📈 对比分析**

通过比较 AI 与人类文本的 semantic delta 分布，并用 Welch t 检验验证差异显著（p<0.05），结果显示 AI 对话的 delta 明显更大，表明主题集中度更高，检测效果可行但数值幅度相对较小。

**⚠️ 局限性**

局限性包括：语料未覆盖真实即兴对话；差异显著但幅度小，实际应用价值待评估；仅限英语文本；仅测试 OpenAI 模型，缺乏跨模型和跨语言验证。

---

## 150. A Mathematical Theory of Understanding

**arXiv ID:** 2603.19349 | [PDF](https://arxiv.org/pdf/2603.19349v1)

**作者:** Bahar Taşkesen `[一作]` `[通讯]` (University of Chicago), Bahar Taşkesen (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了一个关于学习者认知结构的数学模型，建立了“心智”这一抽象学习系统，并通过先决条件的闭包系统和增广规则来描述概念的可达性；在此基础上分析了教学过程的动态、信息通道的状态依赖性，以及结构与信息两方面对学习速度的限制，揭示了教学的阈值效应和广播式教学的结构性代价。

**💡 创新点**

创新点在于：①把学习者的认知结构形式化为可闭包（antimatroid）系统；②将教学视为在状态依赖通道上的随机信号发送，首次结合信息理论与先决条件闭包来刻画“可用信息”的相对性；③推导出结构距离与信息容量共同决定的学习时间下界，揭示教学的两重瓶颈；④阐明了在多种学习者之间共享课程时会出现线性损失的原因。

**🔧 技术方法**

主要技术手段包括：闭包运算、阿尔法结构与反映学习状态的可达集合（学习空间）构造；信息理论中的熵、互信息和黑塞尔支配；状态依赖的信道模型与解析映射；以及概率论中停止时刻和期望分析。

**📊 数据集**

本文为理论研究，没有使用具体的数据集或实验材料，所有结果均为数学推导和定性分析。

**📈 对比分析**

由于缺乏实验对比，本文不涉及方法性能的数值评估；其贡献在于提供了理论上的极限和结构性见解，而非基于数据的实验比较。

**⚠️ 局限性**

局限性包括：①模型假设先决条件是固定且可预知的，忽略了学习者的自适应发展；②教师只能发送固定的信号集，未考虑多模态或连续教学；③对随机目标和信号的分布做了较强假设，可能不适用于所有实际教学场景；④缺乏对噪声、遗忘或记忆有限的现实因素的建模。

---

## 151. Evolving Embodied Intelligence: Graph Neural Network--Driven Co-Design of Morphology and Control in Soft Robotics

**arXiv ID:** 2603.19582 | [PDF](https://arxiv.org/pdf/2603.19582v1)

**作者:** Jianqiang Wang `[一作]` (Leiden University), Yue Xie `[通讯]` (Loughborough University)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5052217750)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种将图注意网络（GAT）与深度强化学习相结合的软体机器人形态与控制共设计框架。

**💡 创新点**

创新点在于使用形态感知的图结构策略实现控制器在形态变异时的可继承性，避免了传统MLP因输入维度不一致而导致的权重不匹配。

**🔧 技术方法**

技术包括遗传算法（GA）、近端策略优化（PPO）、图注意网络（GAT）以及权重映射的继承机制。

**📊 数据集**

数据集采用Evogym 2D软体机器人仿真平台的四个任务：Pusher-v1、Thrower-v0、Carrier-v1、Catcher-v0。

**📈 对比分析**

与基线MLP-PPO共设计以及无继承的GA-PPO相比，GAT-PPO在所有任务上均达到更高的最终奖励、收敛速度更快且方差更低。

**⚠️ 局限性**

局限性包括GAT模型训练初期收敛速度慢以及新增节点初始化导致的短期不稳定。

---

## 152. SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management

**arXiv ID:** 2603.19431 | [PDF](https://arxiv.org/pdf/2603.19431v1)

**作者:** Komal Thareja `[一作]` (Renaissance Computing Institute), Ewa Deelman `[通讯]` (Information Sciences Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种完全去中心化的多代理共识工作负载管理系统 SWARM+，实现分布式作业选择与调度。

**💡 创新点**

创新点包括层级共识降低通信复杂度、自动容错与自适应法定人数、以及将数据传输节点（DTN）连接与数据局部性纳入统一成本模型。

**🔧 技术方法**

采用 PBFT 风格的分层共识、gRPC 双向流、LRU 缓存、Redis 分布式键值存储以及动态法定人数等技术。

**📊 数据集**

在 FABRIC 测试床上使用合成工作负载（基于资源需求的指数分布）以及模拟的十个 DTN 终端和多站点虚拟机部署。

**📈 对比分析**

与先前 SWARM 基线进行多种拓扑（Mesh、Ring、Hierarchical）和失败场景的实验，SWARM+ 在单机选拔时间从 40 秒降至 1 秒，整体调度延迟提升约 60 倍，容错下单机失效保持 99% 作业完成率。

**⚠️ 局限性**

局限在于仍受 WAN 延迟影响、层级结构手工配置、缺乏自适应层级构造以及对真实工作流系统（Pegasus、Nextflow）集成尚未完成。

---

## 153. PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking

**arXiv ID:** 2603.19305 | [PDF](https://arxiv.org/pdf/2603.19305v1)

**作者:** Jiacheng Bao `[一作]` (Northwestern Polytechnical University), Bin Zhao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 17162 | [OpenAlex ID](https://openalex.org/A5100729857)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了PhyGile框架，能根据自然语言生成并实时控制仿人机器人完成敏捷全身动作。

**💡 创新点**

创新点在于用物理前缀指导生成器与通用动作跟踪器耦合，结合难度分层的MoE训练和TP-MoE扩散生成器，显著缩小语义与动力学间的鸿沟。

**🔧 技术方法**

采用了机器人骨架空间的扩散模型、Token‑level Parameter‑mixing MoE、两阶段的MoE训练、物理前缀微调以及PPO改进的GMT控制器。

**📊 数据集**

主要使用HumanML3D（带文本标签）与AMASS、LaFAN1等MoCap数据集进行训练，并在Unitree G1机器人上进行真实测试。

**📈 对比分析**

与GMT、TextOp等基线相比，PhyGile在生成质量（FID、R@3）、跟踪误差（MPJPE、MPJAE）和成功率上均实现了更优表现，尤其在高难度敏捷动作上成功率提升约10%。

**⚠️ 局限性**

限制包括对训练与仿真环境高度依赖、需要额外的物理前缀生成与筛选步骤、以及对不同机器人结构的适配仍需进一步验证。

---

## 154. GeoLAN: Geometric Learning of Latent Explanatory Directions in Large Language Models

**arXiv ID:** 2603.19460 | [PDF](https://arxiv.org/pdf/2603.19460v1)

**作者:** Tianyu Bell Pan `[一作]` (University of Florida), Damon L. Woodard `[通讯]` (University of Florida)

**通讯引用:** 3528 | [OpenAlex ID](https://openalex.org/A5055751228)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了GeoLAN框架，在LLM训练过程中通过几何约束提升可解释性和公平性；

**💡 创新点**

首次将几何测度理论与语言模型内部表示结合，提出“粘性”条件，并设计可微KT‑CW与KT‑Attn正则化；

**🔧 技术方法**

采用Token轨迹/管道模型、Wolff公理、谱熵正则、线性探测器以及DeepSpeed ZeRO训练技术；

**📊 数据集**

在C4大规模文本上微调，评估使用MMLU、GSM8K、TruthfulQA、CrowS‑Pairs等公开基准；

**📈 对比分析**

与标准交叉熵+权重衰减基线对比，发现中等规模模型（如Llama‑3‑8B）保持或提升准确率，同时提升IsoScore、探测效率和偏差指标；在最小模型下性能下降，在最大模型中存在轻微准确率损失；

**⚠️ 局限性**

效果随模型规模显著变化：小模型受压缩影响，几何约束反而导致退化；大模型面临“对齐成本”，准确率下降；此外需手动调节学习率与正则强度，且对不同训练种子表现出旋转对称性问题。

---

## 155. Engineering-Oriented Symbolic Regression: LLMs as Physics Agents for Discovery of Simulation-Ready Constitutive Laws

**arXiv ID:** 2603.19241 | [PDF](https://arxiv.org/pdf/2603.19241v1)

**作者:** Yue Wu `[一作]` (Shanghai University), Deng Pan `[通讯]` (Shanghai University)

**通讯引用:** 14215 | [OpenAlex ID](https://openalex.org/A5002513820)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于大语言模型（LLM）的工程导向符号回归（EO‑SR）框架，利用LLM零射生成热力学一致性、帧不变性等物理约束，将符号回归从纯拟合转变为受物理约束的发现过程，并在橡胶类材料的Treloar数据上自动发现满足全局凸性的混合Mooney‑Rivlin+理想锁定项本构方程，随后通过有限元验证其数值稳定性。

**💡 创新点**

创新点：1) 将LLM作为“Physics‑Informed Agent”在零射方式下即时生成可执行的物理约束；2) 在符号回归中嵌入物理惩罚，使模型在数据拟合和物理合法性之间实现多目标最优；3) 发现的新型混合本构既保持极简可解释性，又在所有多轴变形下保持全局凸性，显著提升 FEM 的收敛性。

**🔧 技术方法**

技术手段：大语言模型 Gemini3‑pro、基于进化/遗传算法的符号回归、自动微分求 Hessian 以判定凸性、惩罚函数与正则化、Skill‑Injector 架构（定义变换、算子与约束）、零射约束生成、有限元（Abaqus/Standard）验证。

**📊 数据集**

数据集：标准 Treloar 橡胶拉伸实验（单轴、双轴），并以未用于训练的纯剪切数据进行零射测试。

**📈 对比分析**

比较方法：在训练集（单轴+双轴）和未见的纯剪切数据上与 Yeoh（N=3）和 Ogden（N=3）进行 MSE 比较；在 FEM 双边缺口拉伸试验中对比收敛性和数值稳定性。性能表现：EO‑SR 发现的模型在训练集 MSE 低于 Yeoh、与 Ogden 相当；在纯剪切预测误差显著优于 Yeoh；在 FEM 试验中，OE‑SR 模型收敛稳定，而 Ogden N=3 在严重 transverse compression 下失稳，凸性约束使 EO‑SR 保持全局稳定。

**⚠️ 局限性**

局限性：1) 目前验证集中于等向性弹性材料，异向性或多场耦合场景尚未充分验证；2) 需要足够多的实验数据以避免过拟合；3) LLM 约束生成的可靠性依赖于模型推理，可能产生不完整或不准确的约束；4) 计算成本和推理延迟仍是使用 LLM 的瓶颈；5) 对模型的物理解释和泛化性仍需进一步实验与理论支持。

---

## 156. Accurate Open-Loop Control of a Soft Continuum Robot Through Visually Learned Latent Representations

**arXiv ID:** 2603.19655 | [PDF](https://arxiv.org/pdf/2603.19655v1)

**作者:** Henrik Krauss `[一作]` (University of Tokyo), Takehisa Yairi `[通讯]` (University of Tokyo)

**通讯引用:** 3380 | [OpenAlex ID](https://openalex.org/A5012762510)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对软连续机器人进行开放式循环控制，利用从视频学习到的潜在动力学模型实现视觉指定轨迹的跟踪；

**💡 创新点**

首次将视觉振荡网络（VON）与ABCD注意力解码器相结合，构建可解释的潜在动力学模型，并证明在无摄像头反馈的长时间开放式循环控制中可实现高精度轨迹跟踪，同时提供实时SCR仿真器用于生成目标轨迹；

**🔧 技术方法**

使用视觉振荡网络（VON）、ABCD注意力广播解码器、Koopman理论、MLP、隐式阻尼、离散单射预投梯度优化、以及多步预测训练等技术；

**📊 数据集**

使用两段软连续机器人视频数据（约30分钟，50Hz采样），包含正弦与阶跃激励，数据已公开（Zenodo）；

**📈 对比分析**

通过与Koopman、MLP、无ABCD模型对比，采用图像空间均方误差（MSE）评估性能，VON+ABCD模型取得最低MSE（≈1.0×10⁻²），Koopman+ABCD表现最佳；Ablation实验表明多步预测、阻尼等对提升性能至关重要；仿真测试显示静态保持、外推稳健等优势；

**⚠️ 局限性**

主要限制包括：低层压力控制器响应速度限制导致快速振荡任务误差增大；无反馈控制易累积误差；未将低层控制动态纳入模型；目前仅适用于二维平面运动，缺乏三维通用性。

---

## 157. LoASR-Bench: Evaluating Large Speech Language Models on Low-Resource Automatic Speech Recognition Across Language Families

**arXiv ID:** 2603.20042 | [PDF](https://arxiv.org/pdf/2603.20042v1)

**作者:** Jianan Chen `[一作]` (Kyoto University), Nancy F. Chen `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 2520 | [OpenAlex ID](https://openalex.org/A5014190404)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LoASR-Bench基准，系统评估低资源语言的ASR性能，覆盖25种语言和9个语系；

**💡 创新点**

首个跨语系、跨脚本的SpeechLM性能对比，揭示脚本与模型规模对低资源ASR的重要影响；

**🔧 技术方法**

采用XLSR-53、Whisper（中/大）和Qwen系列SpeechLM模型，统一微调后进行端到端训练；

**📊 数据集**

使用Common Voice 16.1公开的低资源语料以及对应的测试集；

**📈 对比分析**

对比多模型与微调设置，Qwen3-Omni表现最佳，模型规模与错误率呈负相关但增益有限；

**⚠️ 局限性**

存在脚本混淆、极低资源语言错误率高、未融合语言识别等局限，非拉丁脚本表现仍较差。

---

## 158. CURE: A Multimodal Benchmark for Clinical Understanding and Retrieval Evaluation

**arXiv ID:** 2603.19274 | [PDF](https://arxiv.org/pdf/2603.19274v1)

**作者:** Yannian Gu `[一作]` (Shanghai Jiao Tong University), Xiaofan Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CURE基准，用500个临床病例评估多模态大型语言模型在医学诊断中的多模态理解与检索能力

**💡 创新点**

通过将诊断拆分为多模态理解、检索和基于证据的推理三步，形成可分离的评估框架，首次在临床环境下严格考察检索质量对诊断性能的影响

**🔧 技术方法**

使用多模态检索增强生成（RAG）、代理式检索（Agent Retrieval）以及提供医师引用文献的“oracle”上下文进行模型推理

**📊 数据集**

利用2025年Eurorad发布的530个病例（500个测试集），每个病例包含临床历史、医学影像、放射科发现及对应的PubMed PMID，形成统一的证据库

**📈 对比分析**

在16个模型（包括GPT‑5.2、Gemini‑3、Qwen‑3、Gemma‑3、MedGemma等）上对比四种检索模式，结果显示提供医师引用上下文时准确率可达73.4%（开放式诊断），但无外部检索时仅为25.4%，表明检索是主瓶颈

**⚠️ 局限性**

局限性包括检索质量与覆盖率不足导致模型易被误导、图像信息利用不充分、模型对低质量检索产生高忠诚度并产生错误推理

---

## 159. Growing Networks with Autonomous Pruning

**arXiv ID:** 2603.19759 | [PDF](https://arxiv.org/pdf/2603.19759v1)

**作者:** Charles De Lambilly `[一作]`, Stefan Duffner `[通讯]` (INSA Lyon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Growing Networks with Autonomous Pruning（GNAP）方法，能够在训练过程中根据梯度下降自主增大网络结构并进行剪枝，从而动态调整网络大小与拓扑。

**💡 创新点**

创新点在于将自增与自适应剪枝两大机制结合，使用门控（Gumbel‑Softmax）实现梯度可导的稀疏化，同时提供正交权重初始化策略来控制新增参数的学习与消除，使网络在保持高精度的同时实现极高的稀疏率。

**🔧 技术方法**

核心技术包括：门控稀疏正则（L1门控损失）+梯度下降剪枝、全连接DenseNet式拓扑、可选择的结构化/非结构化剪枝、正交权重初始化、基于梯度下降的动态增生（增神经元/层）策略。

**📊 数据集**

实验使用三大图像分类数据集：MNIST、CIFAR‑10 和 CIFAR‑100。

**📈 对比分析**

通过与不增生的 NAP、以及多种现有稀疏/NAS 方法（SimpleNet、SNIP、GraSP、SRatio、LTH、IMP、RigL、ART、MCUNet 等）对比。GNAP 在 MNIST 上仅 6.2k 参数即可达到 99.44%；CIFAR‑10 仅 157.8k 参数即可达到 92.2%；CIFAR‑100 仅 179.8k 参数即可达到 66.56%。总体上在准确率与参数量的折中上优于大多数对照方法。

**⚠️ 局限性**

限制包括：需要手动设定增生阈值、步长及正交初始化的超参数；训练时仍需较多 epoch 及计算资源；仅在标准图像分类任务上验证，尚未验证对非 i.i.d. 数据或持续学习场景的鲁棒性；结构化剪枝在收敛性与效果上比非结构化更难控制。

---

## 160. It Depends: Re_Authoring Play Through Clinical Reasoning in Wearable AR Rehab Games

**arXiv ID:** 2603.19309 | [PDF](https://arxiv.org/pdf/2603.19309v1)

**作者:** Binyan Xu `[一作]` (Northeastern University), Leanne Chukoskie `[通讯]` (Northeastern University)

**通讯引用:** 1329 | [OpenAlex ID](https://openalex.org/A5081829595)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究先系统性梳理Snap Spectacles平台上132个AR康复应用，筛选出21款与运动相关的游戏，再进一步评估9款并最终选取4款（3‑Dots‑Pinch、Squishy Run、Ball Game、Action Ball）进行现场体验。通过14名持证物理治疗师（10名完成分析）的思考录音、访谈与跨游戏比较，结合通用归纳方法提炼出三大视角：Play是共创(Co‑Authored)、Play是情境化(Situated)、Play是双重效应(Dual)，并据此制定PT友好的设计原则与ICF框架扩展。

**💡 创新点**

创新点在于：①将临床推理（clinical reasoning）作为AR康复游戏设计的视角，构建三大视角框架；②提出PT共创、情境化、双重效应的三维设计原则；③改编ICF模型以贴合AR游戏设计，形成“AR Rehabilitation Games Framework”；④给出针对轻量级AR眼镜的8条PT中心化设计准则。

**🔧 技术方法**

技术手段包括：轻量级透视式AR眼镜Snap Spectacles、Lens Studio开发的四款游戏、基于摄像头与IMU的手势与头部追踪、语音指令与视觉反馈；研究方法为思考录音、半结构化访谈、视频录制与转写、通用归纳式定性编码（Cohen κ=0.68）以及主题分析。

**📊 数据集**

数据来源为：Snap Spectacles Lens库中的132个AR应用（其中21款运动相关）以及10名具备1–37年临床经验的持证物理治疗师的访谈与游戏体验记录。

**📈 对比分析**

本研究未进行定量效能比较；评估方式为定性主题分析，比较不同PT在共创、情境化、双重效应视角下的编码频次与共性，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：①未纳入患者参与，缺乏临床疗效与长期使用数据；②样本以美国DPT系统为主，可能不具全球可推广性；③仅使用Snap Spectacles平台，其他AR眼镜的适用性未知；④参与者职业阶段分布不均，可能影响观点多样性；⑤未评估光学兼容性、视力矫正等实际使用障碍。

---

## 161. Spectral Alignment in Forward-Backward Representations via Temporal Abstraction

**arXiv ID:** 2603.20103 | [PDF](https://arxiv.org/pdf/2603.20103v1)

**作者:** Seyed Mahdi B. Azad `[一作]` (University of Freiburg), Joschka Boedecker `[通讯]` (University of Freiburg)

**通讯引用:** 69897 | [OpenAlex ID](https://openalex.org/A5084499878)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作通过谱分析揭示了前向-后向(FB)表示学习在连续空间中面临的高秩谱匹配难题，并提出利用动作重复（k-step 时间抽象）作为谱低通滤波器，降低成功率表示(SR)的有效秩，从而显著提升低秩FB学习的稳定性与性能。

**💡 创新点**

创新点在于将时间抽象视为对SR谱的低通滤波，理论证明动作重复可指数衰减高频谱成分，并给出相应的误差上界；实验验证该方法在连续迷宫任务中相较于仅调节折扣或嵌入维度能更好地平衡长时序信息与低秩逼近。

**🔧 技术方法**

技术包括谱分析与低秩分解、FB学习框架、动作重复（k-step）时间抽象、有效秩与谱熵等指标评估，以及数值实验。

**📊 数据集**

使用OGBench提供的连续迷宫环境（Four-Rooms、Maze、Large-Maze），并采用RBF或CNN编码的状态表示。

**📈 对比分析**

与无时间抽象、不同折扣 γ 及嵌入维度 d 的FB进行对比实验；结果表明在 k>1 时，平均回报显著提升，贝尔曼误差下降；在高 γ 下无时间抽象会导致学习不稳定，而动作重复能保持稳定并获得更高性能。

**⚠️ 局限性**

局限性在于仅考察了动作重复作为最简单的时间抽象形式，未探索更复杂的选项或学习技能；实验仅在在线交互、低维状态下进行，未结合离线数据或更高维观测。

---

## 162. Spelling Correction in Healthcare Query-Answer Systems: Methods, Retrieval Impact, and Empirical Evaluation

**arXiv ID:** 2603.19249 | [PDF](https://arxiv.org/pdf/2603.19249v1)

**作者:** Saurabh K Singh `[一作]` `[通讯]` (Oracle Corporation), Saurabh K Singh (Oracle Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过对真实医疗查询中的拼写错误进行系统统计与纠正，并评估其对医疗问答检索性能的提升。

**💡 创新点**

首次量化真实医疗查询中的拼写错误比例，并证明查询端纠正对检索性能的显著作用，提出保守编辑距离作为安全实用的纠正策略。

**🔧 技术方法**

采用保守编辑距离、标准编辑距离、上下文感知候选排序以及SymSpell等基于词典的纠正方法，并结合BM25和TF‑IDF检索模型进行评估。

**📊 数据集**

使用TREC 2017 LiveQA Medical、HealthSearchQA查询集与MedQuAD回答语料库进行实验，构建8,201词汇表。

**📈 对比分析**

在BM25检索中，编辑距离纠正使MRR从0.633提升到0.691（+9.2%），保守纠正几乎达到同等提升且误改率更低；Corpus‑only纠正对检索几乎无益，TF‑IDF实验亦表现相似。

**⚠️ 局限性**

局限包括样本量有限、仅评估词典级方法、未覆盖多语言与大规模语料、未测试神经检索与LLM纠正、以及对潜在高危词对的安全性未完全验证。

---

## 163. FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement

**arXiv ID:** 2603.19608 | [PDF](https://arxiv.org/pdf/2603.19608v1)

**作者:** Ming Hu `[一作]` (Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences), Quan Wang `[通讯]` (Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计了FB-CLIP框架，实现零样本细粒度异常检测，融合多策略文本特征与前景‑背景分离与背景抑制；

**💡 创新点**

创新点包括：多策略文本特征融合（MSTFF）与语义一致性正则化（SCR）提升文本语义丰富度；多视角前景‑背景软分离（MVFBE）与连续门控融合增强视觉特征；背景抑制（BS）进一步去除背景干扰；

**🔧 技术方法**

使用的技术主要是CLIP预训练模型、文本特征多策略融合、注意力加权、前景‑背景软分离与多视角融合、背景原型抑制、跨模态对齐与熵+边缘正则化；

**📊 数据集**

使用了工业与医学共16个公开数据集：MVTec、VisA、MPDD、BTAD、DAGM、DTD、Real‑IAD、HeadCT、BrainMRI、Br35H、ISIC、ClinicDB、ColonDB、Kvasir、Endo、TN3K；

**📈 对比分析**

与WinCLIP、VAND、AnomalyCLIP、AdaCLIP、AA‑CLIP、AF‑CLIP、FAPrompt等SOTA方法对比，FB-CLIP在工业与医学数据集上多项指标（AUROC、AP、PRO）均处于前列，特别在VisA和Real‑IAD的像素级指标提升约6%以上；

**⚠️ 局限性**

局限性：对极小或极微细异常的检测仍不够精细；对极为复杂背景的分离效果有限；在部分医学数据集上性能略低于FAPrompt；整体依赖CLIP预训练，难以迁移到非视觉模态或未覆盖的领域。

---

## 164. From School AI Readiness to Student AI Literacy: A National Multilevel Mediation Analysis of Institutional Capacity and Teacher Capability

**arXiv ID:** 2603.20056 | [PDF](https://arxiv.org/pdf/2603.20056v1)

**作者:** Xiu Guan `[一作]` (School of Education, Tsinghua University), Lixiang Yan `[通讯]` (Faculty of Information Technology, Monash University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在全国规模的大样本数据上构建并验证了一个2-2-1跨层级中介模型，探讨了学校层面AI准备度如何通过教师集体能力影响职业院校学生的AI素养。

**💡 创新点**

首次在大规模、层级结构严谨的数据上提供实证证据，将学校AI准备度视为一个整合的组织配置，并揭示教师认知的AI能力是关键的传递机制，且该路径在不同地区均稳健。

**🔧 技术方法**

采用多层线性混合效应模型、跨层级路径分析（a×b间接效应）与蒙特卡罗法验证中介效应，以及多重稳健性检验（区域性调节、聚合层级模型）。

**📊 数据集**

全国职业院校调查（1,007所学校、156,125名教师、2,379,546名学生）收集的问卷与测评数据。

**📈 对比分析**

相较于单层回归直接效应，跨层级中介模型显著提升了对学生AI素养解释力，教师AI能力的间接效应虽小但显著（ab≈0.0015），显示机构准备度通过教师集体能力实现部分传递。

**⚠️ 局限性**

研究为横断面设计，因果推断受限；教师机制仅基于主观感知，缺乏课堂实践或平台使用数据；学校层面方差相对较小，个体差异仍占主要份额。

---

## 165. Scale-Dependent Radial Geometry and Metric Mismatch in Wasserstein Propagation for Reverse Diffusion

**arXiv ID:** 2603.19670 | [PDF](https://arxiv.org/pdf/2603.19670v1)

**作者:** Zicheng Lyu `[一作]` (Fudan University), Zengfeng Huang `[通讯]` (Fudan University)

**通讯引用:** 1459 | [OpenAlex ID](https://openalex.org/A5062549536)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

在弱对数凸目标下，研究反向扩散过程的几何结构，发现高尺度下会出现收缩但低尺度仍受欧氏度量约束，并基于此提出单次切换（one‑switch）路由定理，实现从径向收缩到最终欧氏距离的错误传播。

**💡 创新点**

创新点：①构造了径向下界剖面 κ_s(r) 兼顾远场收缩与近场阻碍；②提出适用于反射耦合的折射（switch）度量 W_{φ_{s_0}}，在早期窗口实现最大收缩；③给出一次性从非欧氏度量恢复欧氏 2‑Wasserstein 的精确幂指数 θ_p = (p-2)/(2(p-1)) 并证明其结构上最优；④将这些几何工具整合成整体路由公式，显式展示何时切换以及最终误差上界。

**🔧 技术方法**

技术方法：使用反向扩散 SDE 与学习逆扩散；定义径向漂移剖面 κ_u(r)；构造反射耦合并对其径向动力学做 Ito 变换；设计折射度量 φ_{s_0} 使其满足子解条件；一次性欧氏转换的三角不等式与 p‑moment 预算；最后将早期折射误差与后期欧氏误差拼接得到全局非渐近误差上界。

**📊 数据集**

本工作为理论分析，无使用具体数据集；所有结果均为解析推导与不依赖样本的泛化界。

**📈 对比分析**

与传统的全时段欧氏传播方法对比，单次切换路由在弱对数凸情形下能充分利用远场收缩，从而给出更小的误差上界；定量上通过比较 Δ_{late} 与 Δ_{φ} 的比例以及一次性转换指数 θ_p 的结构严谨性，证明了在该几何场景下其性能优于传统欧氏闭包方法。

**⚠️ 局限性**

局限性：仅适用于弱对数凸目标；只考虑单次切换，未覆盖多切换、强异方差或更复杂的多尺度几何；对折射度量的尾部要求为仿射尾部，限制了可选度量形式；理论假设（如一侧 Lipschitz、强可解性等）较为严格，实际数值实现需进一步验证。

---

## 166. PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction

**arXiv ID:** 2603.19733 | [PDF](https://arxiv.org/pdf/2603.19733v1)

**作者:** Runsong Zhao `[一作]` (Northeastern University), Bo Zheng `[通讯]` (Future Living Lab of Alibaba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Performance-oriented Context Compression（PoC）框架，让开发者通过指定性能下限而非压缩比例，自动寻找最激进且满足性能约束的压缩比例；

**💡 创新点**

创新点在于把压缩控制从预算式转为性能式，结合轻量级上下文感知预测器和P@R评估指标，实现可预测且更高效的自适应压缩；

**🔧 技术方法**

使用裁剪后的Llama-3.2-1B作为性能预测器，LLMLingua2做硬压缩，Llama-3.1-8B-Instruct作为阅读LLM，并采用双阶段搜索算法；

**📊 数据集**

实验基准覆盖七个数据集：SearchQA、TriviaQA、Natural Questions、HotpotQA、SQuAD、GovReport和SummScreenFD；

**📈 对比分析**

与上下文无关基线相比，context-aware PoC将PPE降低12–34%，提升P@R和F1/ROUGE等指标，压缩率更高、推理延迟显著下降；

**⚠️ 局限性**

局限在于只能提升现有压缩器的利用率，无法突破压缩器本身的性能上限；预测器的准确性受训练数据限制，极端信息密集场景仍需保守压缩。

---

## 167. LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment

**arXiv ID:** 2603.19609 | [PDF](https://arxiv.org/pdf/2603.19609v1)

**作者:** Shuaibang Peng `[一作]` (National University of Defense Technology), Shen Yan `[通讯]` (National University of Defense Technology)

**通讯引用:** 29510 | [OpenAlex ID](https://openalex.org/A5100370396)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoD‑Loc v3，解决低细节 LoD 模型下的跨场景泛化与稠密城市定位歧义问题。

**💡 创新点**

① 设计大规模合成实例分割数据集 InsLoD‑Loc；② 将定位范式从语义轮廓对齐改为实例轮廓对齐。

**🔧 技术方法**

合成渲染管线（UE5+Cesium+AirSim）、LoD 模型实例化、基于 SAM 的实例分割、基于实例匹配的粗细两阶段定位框架。

**📊 数据集**

108,109 张合成 RGB 图像和实例掩码的 InsLoD‑Loc 数据集，覆盖 40 个地区；公开基准 UAVD4L‑LoDv2、Swiss‑EPFLv2、Tokyo‑LoDv3。

**📈 对比分析**

与 LoD‑Loc、LoD‑Loc v2、CAD‑Loc、MC‑Loc 等 9 种基线对比，LoD‑Loc v3 在 (2 m, 2°)、(3 m, 3°)、(5 m, 5°) 下的定位成功率均显著优于对手，跨域提升达 2000% 左右。

**⚠️ 局限性**

对极端恶劣天气或 LoD 模型误配时实例分割精度下降，导致定位误差增大。

---

## 168. TextReasoningBench: Does Reasoning Really Improve Text Classification in Large Language Models?

**arXiv ID:** 2603.19558 | [PDF](https://arxiv.org/pdf/2603.19558v1)

**作者:** Xinyu Guo `[一作]` (Tianjin University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 25461 | [OpenAlex ID](https://openalex.org/A5100662807)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并系统评估了七种推理策略在文本分类任务中的有效性与效率，提出了面向成本的评估指标

**💡 创新点**

①首次量化推理成本与性能收益（AE、ME） ②发现推理对不同任务、模型规模的影响不一致，揭示“过度推理”与“非目标推理”的弊端

**🔧 技术方法**

零样本提示、Chain‑of‑Thought (CoT)、Self‑Consistency CoT (SC‑CoT)、Tree‑of‑Thought (ToT)、Graph‑of‑Thought (GoT)、Bagging‑of‑Cues (BoC)、Long‑CoT 以及对应的指标计算

**📊 数据集**

AG News、TREC‑QC（客观分类）和 SST‑2、SemEval‑2018、iSarcasmEval（主观分类）五个公开数据集

**📈 对比分析**

在十个大型语言模型（包括 GPT‑4o‑mini、Gemma‑3‑4B、Llama‑3.1‑8B、Qwen‑3‑8B、GPT‑5.2、DeepSeek‑V3.2、Kimi‑K2、Gemini‑2.5‑Flash 等）上进行对比，结果显示：单一推理（IO）往往最具成本效益；CoT/SC‑CoT 可在大模型上获得 1–3% 的微小提升；ToT/GoT 在多任务上易导致性能下滑；Long‑CoT 在大模型上有潜在收益但仍不总是最优

**⚠️ 局限性**

①推理不具普适性，需针对任务与模型规模定制 ②高成本推理往往收益有限，难以平衡效率与精度 ③仅覆盖零样本场景，缺少自适应/动态推理机制 ④实验未考虑多语言或多任务迁移的普适性

---

## 169. CoverageBench: Evaluating Information Coverage across Tasks and Domains

**arXiv ID:** 2603.20034 | [PDF](https://arxiv.org/pdf/2603.20034v1)

**作者:** Saron Samuel `[一作]` (Johns Hopkins University), Eugene Yang `[通讯]` (Johns Hopkins University)

**通讯引用:** 978 | [OpenAlex ID](https://openalex.org/A5062016266)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CoverageBench，一个统一的基准，旨在评估检索系统在多任务、多领域中的信息覆盖能力，并提供基准数据与评测工具。

**💡 创新点**

创新点在于：①将现有检索集合转换为覆盖评估基准，免除从零创建新集合的高成本；②为每个主题生成“nugget”（信息单元）与文档-信息单元对应关系；③公开完整的数据集、评测代码与基线排名，降低研究门槛。

**🔧 技术方法**

技术方法包括：利用 BM25 与 Qwen3‑8B 的稀疏/稠密检索；对前 50 条结果使用 Rank1‑7B 与 Qwen3‑Reranker‑8B 进行重排序；使用 LLM（Llama‑3.3‑70B‑Instruct）对文档与 nugget 进行匹配判断；评测指标包括 nDCG、α‑nDCG 与 Subtopic Recall@20。

**📊 数据集**

使用的数据集有七个：NeuCLIR 2024、RAG 2024、RAGTIME 2025、Fair Ranking 2022、CAsT 2020、CRUX‑MultiNews、CRUX‑DUC04；其中前五个需做文档/查询重写、nugget 衍生与 LLM 标注，后两个已天然满足覆盖评估。

**📈 对比分析**

比较方法是将六种检索+重排序组合（BM25/Qwen3‑8B 作为初始检索，再重排 Rank1‑7B 或 Qwen3‑Reranker‑8B）在各基准上分别计算 nDCG、α‑nDCG 与 StRecall@20。实验显示：在大多数数据集上，稠密检索 + Qwen3‑8B 基线优于 BM25，但在 Fair Ranking 上 BM25 更好；最高 nDCG 的配置不一定拥有最高 StRecall，表明覆盖与传统相关度存在偏差。

**⚠️ 局限性**

限制：①覆盖评估高度依赖 LLM 标注，标注质量与模型偏差可能影响结果；②部分数据集主题数有限，覆盖评估的普适性需进一步验证；③目前仅评估检索阶段，未完整覆盖 RAG 系统生成阶段的互动影响；④重排序模型主要是现成的商业模型，缺乏可解释性与可复制性。

---

## 170. Legged Autonomous Surface Science In Analogue Environments (LASSIE): Making Every Robotic Step Count in Planetary Exploration

**arXiv ID:** 2603.19661 | [PDF](https://arxiv.org/pdf/2603.19661v1)

**作者:** Cristina G. Wilson `[一作]` (Oregon State University), Feifei Qian `[通讯]` (University of Southern California)

**通讯引用:** 574 | [OpenAlex ID](https://openalex.org/A5067936185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发并验证了一套基于四足腿型机器人的探测平台LASSIE，该平台将行走本身转化为地面力学传感器，并与人类科学家协同决策，实现在类地球环境下对土壤机械性质的高空间分辨率测量与探测；

**💡 创新点**

创新点在于：①将机器人步态与感知耦合，实时获取土壤阻力曲线；②提出以人类科学家推理为模仿的共享自治算法，支持动态目标权重与解释性建议；③结合实验室与两个行星模拟现场，展示腿式机器人在斜坡与冰质土层中的高效通行与信息获取；

**🔧 技术方法**

采用了高转矩、低齿比的无齿直接驱动腿关节、全压传感与惯性测量单元实现力学感知；使用深度学习/贝叶斯推理的认知启发式采样算法；并在野外部署四足机器人、单腿感知单元、遥感无人机与地面地质仪器组合；

**📊 数据集**

数据集包括：①美国白沙国家公园的黄岩沙丘类地貌实验数据；②俄勒冈霍德山的冰质火山地貌实验数据；③实验室内气流化床合成沙砾与混合物的力学曲线；以及与标准测力仪、光学与热成像数据的配对；

**📈 对比分析**

通过与单腿测力仪的“地面真实值”对照，验证了机器人腿的力学测量精度（误差<5%）；共享自治算法在白沙实验中被科学家接受率约70%，显著提高了样点选择效率；在火山冰质实验中，机器人在局部边界区域的高分辨率测量与热成像相一致，证明了方法在多尺度、不同土壤特性环境下的有效性；

**⚠️ 局限性**

局限性包括：①对极端高粘度或多孔冰层的感知仍需改进；②在高能耗或电池有限的太空任务中，连续步态测量的能耗和时间成本未充分评估；③共享自治模型仍需在更大规模、多机器人协同与更复杂目标权重下进行验证；

---

## 171. Just-in-Time Resale in an Ahead-of-Time Auction: An Event Study

**arXiv ID:** 2603.20175 | [PDF](https://arxiv.org/pdf/2603.20175v1)

**作者:** Burak Öz `[一作]` (Flashbots), Akaki Mamageishvili `[通讯]` (Offchain Labs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

分析Arbitrum Timeboost拍卖与Kairos二级市场对竞价竞争与剩余分配的影响。

**💡 创新点**

首次量化二级市场如何削弱一次性拍卖竞争并重新分配剩余。

**🔧 技术方法**

利用DuneAnalytics查询、以太坊交易追踪、统计分析和模拟套利回报估算。

**📊 数据集**

收集了2026年2月1日至3月12日期间47,689个拍卖、117,143次竞标、7,299,376笔时间加速交易及相关CEX–DEX套利交易。

**📈 对比分析**

通过对比不同时间段（Pre‑Kairos、Kairos、Reserve Price Adaptation、Steady State）下的竞标gap、获胜率、PnL与波动率相关性，发现二级市场后拍卖收入下降约一半，剩余转移至搜索者和Kairos。

**⚠️ 局限性**

数据局限：对Kairos off‑chain支付不可观测、套利策略识别仅覆盖简单单一交换、对价格与交易的假设可能偏差。

---

## 172. Current LLMs still cannot 'talk much' about grammar modules: Evidence from syntax

**arXiv ID:** 2603.20114 | [PDF](https://arxiv.org/pdf/2603.20114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 173. Ensembles-based Feature Guided Analysis

**arXiv ID:** 2603.19653 | [PDF](https://arxiv.org/pdf/2603.19653v1)

**作者:** Federico Formica `[一作]` (McMaster University), Claudio Menghi `[通讯]` (University of Bergamo)

**通讯引用:** 3149 | [OpenAlex ID](https://openalex.org/A5074393374)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Ensembles-based Feature Guided Analysis (EFGA)，一种在 Feature Guided Analysis 基础上将提取的规则聚合成集合以提升规则召回率的方法。

**💡 创新点**

创新点在于：①引入三种聚合准则（TOP、REC、AVG）来自动选择并组合规则；②理论证明聚合后召回率等于单规则召回率之和；③系统评估不同准则对召回率、精确率及规则长度的权衡。

**🔧 技术方法**

技术路线包括：基于深度神经网络的激活提取 → 决策树学习 → 规则路径抽取 → 规则聚合成集合 → 计算训练/测试集上的精确率与召回率。

**📊 数据集**

使用的实验数据集：MNIST（手写数字）与 Lymphoma Subtype Classification（两种 L-DNN 架构），并在 M-DNN3 上进行对照实验。

**📈 对比分析**

评估方法：与原始 FGA 在同一网络和相同层级下对比，分别计算训练集和测试集的召回率及精确率。实验显示 EFGA 在训练召回率提升最高可达 +33.15%，测试召回率提升最高可达 +30.81%，精确率仅下降 ≤1.39%。

**⚠️ 局限性**

局限性：①需要人工选择规则聚合的层级和准则；②聚合规则会导致规则长度增长，影响可读性；③仅在两类数据集上验证，外部有效性待进一步评估；④未探讨更复杂或自适应的聚合策略。

---

## 174. Algorithms for Euclidean Distance Matrix Completion: Exploiting Proximity to Triviality

**arXiv ID:** 2603.19447 | [PDF](https://arxiv.org/pdf/2603.19447v1)

**作者:** Fedor V. Fomin `[一作]` (University of Bergen), Saket Saurabh `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对 d‑EDMC（在 d 维度下完成欧氏距离矩阵）问题的多种固定参数化算法与多项式时间算法，解决了在高密度与结构受限的输入下的可完成性判定。

**💡 创新点**

创新点在于引入“距离到简易性”框架，利用矩阵中缺失条目的结构模式（如 K_t,t‑free、最大度数 Δ、边团覆等）实现实例压缩与可行性判定，并首次给出关于填充数（fill‑in）为常数时的 XP 算法。

**🔧 技术方法**

核心技术包括：压缩算法将实例缩小至 O(1) 行列；利用欧氏距离矩阵与正定矩阵的 Cayley‑Menger 判定；真实代数几何中的存在量化公式求解；以及图结构（无 K_t,t 子图、弦图等）与距离几何的紧密结合。

**📊 数据集**

本文未使用任何外部数据集，所有结果均为理论证明与算法设计。

**📈 对比分析**

由于算法主要针对参数化场景，没有与基准实现进行实验比较；理论上在参数 d 与结构参数（如 t、Δ、k 或 fill‑in）固定时实现多项式或指数时间可解，显著优于通用 NP‑难情形。

**⚠️ 局限性**

局限性包括：仍无法处理树宽、树深等常见图参数的情况；对参数化组合（如树宽 + d）仍为 NP‑难；在实际实现时高阶多项式常数大，可能影响可扩展性。

---

## 175. Narrative Aligned Long Form Video Question Answering

**arXiv ID:** 2603.19481 | [PDF](https://arxiv.org/pdf/2603.19481v1)

**作者:** Rahul Jain `[一作]` (Purdue University), Garin Kessler `[通讯]` (Amazon)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5026277738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NA‑VQA长视频推理基准和Video‑NaRA叙事记忆检索框架。

**💡 创新点**

创新点在于标注跨时距证据（Short/Medium/Far），将事件链构造为记忆槽并在检索中使用叙事上下文。

**🔧 技术方法**

采用多模态LLM (Qwen‑VL 2.5 7B)、CLIP视觉编码、叙事记忆存储、检索模块以及chain‑of‑thought提示和LLM评判。

**📊 数据集**

基于LSMDC 88部完整电影构建NA‑VQA，随后在该数据集上进行实验，亦与CinePile、MH‑VidQA等基准对比。

**📈 对比分析**

使用LLM‑as‑judge评测，Video‑NaRA在NA‑VQA上相较于基线提升约3%，在长距离（Far）证据场景中表现尤为显著。

**⚠️ 局限性**

局限在于仍难以精准定位极长距离证据，模型对叙事记忆构建的依赖导致推理复杂度提升，且评测依赖LLM判断可能带来偏差。

---

## 176. On the Dynamics & Transferability of Latent Generalization during Memorization

**arXiv ID:** 2603.19865 | [PDF](https://arxiv.org/pdf/2603.19865v1)

**作者:** Simran Ketha `[一作]` (Birla Institute of Technology and Science Pilani), Venkatakrishnan Ramaswamy `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究深度网络在标签噪声下的“记忆化”过程中层级表示的潜在泛化能力，并提出线性探针 VeLPIC，比较其与非线性探针 MASC 及传统逻辑回归探针的表现；随后利用 VeLPIC 的向量直接编辑预 softmax 权重，实现将潜在泛化即时转移到模型，并评估其对后续训练的影响。

**💡 创新点**

① 证明 MASC 为二次型分类器；② 设计线性探针 VeLPIC，能够在多种模型与噪声程度下与 MASC 比较甚至超越其潜在泛化性能；③ 通过权重编辑实现潜在泛化直接迁移，避免重训；④ 系统分析训练动态和层级差异，揭示潜在泛化与模型泛化的共同与分离机制。

**🔧 技术方法**

基于子空间几何的探针（MASC）、线性向量探针（VeLPIC）与逻辑回归探针；PCA 生成子空间或主方向；对层级表示做投影并计算角度；在最后一层用 VeLPIC 向量构造新的预 softmax 权重；在不同训练阶段插入权重干预，监控测试精度；使用梯度下降、交叉熵等标准训练方法。

**📊 数据集**

数据集：MNIST、CIFAR-10、Fashion-MNIST、Tiny ImageNet；模型：多层感知机（MLP）、卷积网络（CNN）、AlexNet、ResNet-18；标签噪声程度：0%、20%、40%、60%、80%、100%。

**📈 对比分析**

采用层级与时间维度对比模型、MASC、VeLPIC 及逻辑回归的测试精度。结果表明：① 在多数模型/噪声水平下，VeLPIC 在后期层级能获得与 MASC 同样或更高的潜在泛化；② 传统逻辑回归往往表现最差；③ 权重编辑后，模型在大多数情形下能立即获得与 VeLPIC 同等的泛化提升，且多数 epoch 维持此水平，唯一例外是 ResNet-18 的低噪声情况。

**⚠️ 局限性**

1) 现有实验仅覆盖卷积/全连接网络，未验证 transformer 等架构；2) MASC 仅在 99% 方差阈值子空间上评估，其他维度尚未探索；3) 权重迁移仅在最后一层实现，未能在中间层直接迁移潜在泛化；4) 对不同噪声程度与层级之间的线性可解性原因仍缺乏理论解释；5) 仅使用单一噪声模型（随机标签洗牌），未探究其他噪声形式。

---

## 177. ODySSeI: An Open-Source End-to-End Framework for Automated Detection, Segmentation, and Severity Estimation of Lesions in Invasive Coronary Angiography Images

**arXiv ID:** 2603.20021 | [PDF](https://arxiv.org/pdf/2603.20021v1)

**作者:** Anand Choudhary `[一作]` (LTS4 Laboratory, EPFL), Dorina Thanou `[通讯]` (LTS4 Laboratory, EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了ODySSeI，一个从ICA图像自动完成病变检测、分割与严重程度估计的开源端到端框架；

**💡 创新点**

通过三层Pyramidal Augmentation Scheme提升鲁棒性、提出基于MLD的客观评估指标以及无QCA的规则式严重度估计方法，显著提高了检测与分割性能；

**🔧 技术方法**

使用YOLO11m进行检测、DeepLabv3+进行分割，并结合静态、动态与组合增强、距离变换与中心线骨架来计算最小腔径和狭窄率；

**📊 数据集**

在欧洲、北美、亚洲的三大数据集（FAME2、ARCADE和Future Culprit）上进行训练与评估；

**📈 对比分析**

相较于基线，检测mAP@0.50提升约2.5倍，分割Dice提升1–3%，MLD误差仅为2–3像素，实时推理在CPU上几秒、GPU上毫秒级，显示出卓越的通用性与效率；

**⚠️ 局限性**

需要人工挑选心室舒张期帧、训练集规模仍有限，未来计划自动化帧选取、扩大标注规模并开展前瞻性临床验证。

---

## 178. Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them

**arXiv ID:** 2603.19852 | [PDF](https://arxiv.org/pdf/2603.19852v1)

**作者:** Michael Hubbertz `[一作]` (University of Wuppertal), Tobias Meisen `[通讯]` (University of Wuppertal)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5032638290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于深度学习的在线映射失效模式评估框架，区分位置特征记忆与地图几何过拟合。

**💡 创新点**

创新点在于使用离散Frechet距离构造几何相似度、设计定位与几何过拟合得分，并通过最小生成树进行数据集稀疏化。

**🔧 技术方法**

使用Transformer‑DETR架构的在线映射模型（MapTR、MapTRv2、MapQR、MGMap）与Frechet距离、MST等方法。

**📊 数据集**

使用nuScenes与Argoverse 2两大公共在线映射数据集。

**📈 对比分析**

通过对原始、地理分离及几何分离的不同划分进行比较，结果显示地理与几何多样性越高模型性能越好，MST稀疏化可在保持或提升性能的同时显著减少训练样本。

**⚠️ 局限性**

局限在于对几何相似度的计算开销大、只关注静态几何而未考虑动态天气等环境因素，且方法主要验证于少数两数据集。

---

## 179. GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems

**arXiv ID:** 2603.19677 | [PDF](https://arxiv.org/pdf/2603.19677v1)

**作者:** Hongjiang Chen `[一作]` (Hangzhou Dianzi University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 24483 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于协作组的LLM多智能体通信拓扑生成方法（Group-of-Agents），并引入条件信息瓶颈（CIB）来压缩组间噪声；

**💡 创新点**

创新点包括：①以协作组为构建原子单元，突破传统节点中心化生成限制；②使用LLM预先生成候选组并自回归选取与连通；③引入CIB进行任务条件下的通信压缩；

**🔧 技术方法**

技术手段包括：LLM（GPT‑4）生成协作组与角色提示，句子编码器（MiniLM‑L6）嵌入，GRU‑自回归图生成网络，变分条件信息瓶颈（CIB）实现压缩；

**📊 数据集**

数据集覆盖推理与编程任务：MMLU、GSM8K、MultiArith、SVAMP、AQuA、HumanEval；

**📈 对比分析**

与固定拓扑、模板剪枝、ARG‑Designer 等多基线对比，平均准确率达到 93.84%，在所有六个基准上均超越 SOTA，同时 token 消耗下降约 17%；

**⚠️ 局限性**

局限性在于预先固定的协作组池限制了动态生成新组的灵活性，且在高度动态或交互式环境中的表现尚未充分验证。

---

## 180. HQC Post-Quantum Cryptography Decryption with Generalized Minimum-Distance Reed-Solomon Decoder

**arXiv ID:** 2603.20156 | [PDF](https://arxiv.org/pdf/2603.20156v1)

**作者:** Jiaxuan Cai `[一作]` (Ohio State University), Xinmiao Zhang `[通讯]` (Ohio State University)

**通讯引用:** 3578 | [OpenAlex ID](https://openalex.org/A5063673084)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出在HQC密码系统解密中使用通用最小距离（GMD）RS译码器，利用RM译码产生的可靠性信息来减少RS码字长度，进而缩短密钥长度并降低硬件面积与延迟。

**💡 创新点**

创新点在于首次将GMD译码与软信息结合应用于HQC解密，并推导出针对该系统的上界估计，证明可将RS码字长度从46缩短至36，显著提升性能。

**🔧 技术方法**

采用GMD译码、增强并行逆除Berlekamp-Massey（ePIBM）、一次性GMD求解算法、FHT、排序器与多级折叠硬件架构，实现短低速率RS码的高效译码。

**📊 数据集**

实验基于对RM译码输出的10^9次模拟样本，用以估计软信息分布及误码率上界。

**📈 对比分析**

与之前的擦除仅译码和硬判决译码比较，GMD译码在HQC-128解密中实现了约20%的时延降低、15%的面积缩减，且所需的RS码字长度缩短。

**⚠️ 局限性**

局限性在于仅在HQC-128场景下验证，对更高安全等级（192/256）需进一步评估；此外，GMD译码在极低误码率区域仍受限于软信息分布假设的准确性。

---

## 181. RiboSphere: Learning Unified and Efficient Representations of RNA Structures

**arXiv ID:** 2603.19636 | [PDF](https://arxiv.org/pdf/2603.19636v1)

**作者:** Zhou Zhang `[一作]` (Nanjing University), Tianfan Fu `[通讯]` (Nanjing University)

**通讯引用:** 3227 | [OpenAlex ID](https://openalex.org/A5003226543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出RiboSphere框架，学习RNA结构的离散几何表示并实现高精度重构。

**💡 创新点**

创新点在于将向量量化与流匹配结合，通过有限标量量化得到可解释的结构词典，避免后验坍塌且捕获RNA模块化。

**🔧 技术方法**

采用几何变换器编码器、有限标量量化、流匹配解码器以及自回归GVP序列解码和GerNA-Bind集成。

**📊 数据集**

使用PDB中约6,000条RNA 3D结构，gRNAde、RNASolo、Robin、Biosensor等数据集进行预训练与下游任务。

**📈 对比分析**

通过与AlphaFold等蛋白模型对比，在结构重构达到RMSD1.25Å/TM0.84，逆折叠序列恢复63%，RNA‑配体预测AUROC0.95等，整体性能优于现有基线。

**⚠️ 局限性**

局限性包括仍需实验验证、对极少样本多模态交互等场景的泛化不确定以及对大分子复杂动态过程的建模不足。

---

## 182. MoCA3D: Monocular 3D Bounding Box Prediction in the Image Plane

**arXiv ID:** 2603.19538 | [PDF](https://arxiv.org/pdf/2603.19538v1)

**作者:** Changwoo Jeon `[一作]` (University of California), Achuta Kadambi `[通讯]` (University of California)

**通讯引用:** 10015 | [OpenAlex ID](https://openalex.org/A5043479061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MoCA3D，一种单目、类无关的模型，直接在图像平面上预测3D包围盒的投影角点及对应深度；

**💡 创新点**

创新点在于将角点与深度预测转化为稠密热图与深度图的回归，并使用盒子条件Transformer与软argmax实现像素级对齐，同时提出Pixel-Aligned Geometry（PAG）指标以直接衡量图像平面几何精度；

**🔧 技术方法**

采用DINOv3 ViT-L/16骨干网络、盒子先验编码、Transformer编码解码、热图+深度头、soft-argmax与Depth Anything融合等技术；

**📊 数据集**

在融合nuScenes、KITTI、SUN RGB‑D、ARKitScenes、Hypersim与Objectron等数据集组成的Omni3D基准上进行训练与评估；

**📈 对比分析**

与Cube R‑CNN、OVMono3D‑Lift、DetAny3D等基线相比，MoCA3D在六个数据集的PAG_uv下降至3.12–5.63像素，PAG_d下降至5.04–8.78%，同时保持与DetAny3D相近的3D IoU和NHD，并仅使用19M可训练参数，速度高达0.14s/样例；

**⚠️ 局限性**

局限在于假设所有八个角点可见，对截断物体或对称物体的姿态仍存在歧义，且在高度遮挡或角点缺失时表现下降。

---

## 183. Verifiable Error Bounds for Physics-Informed Neural Network Solutions of Lyapunov and Hamilton-Jacobi-Bellman Equations

**arXiv ID:** 2603.19545 | [PDF](https://arxiv.org/pdf/2603.19545v1)

**作者:** Jun Liu `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**通讯引用:** 58140 | [OpenAlex ID](https://openalex.org/A5100450180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文开发了可验证的误差界限，用于近似解Lyapunov和Hamilton–Jacobi–Bellman (HJB)方程，特别强调基于物理信息的神经网络（PINN）近似。

**💡 创新点**

创新点在于建立了可验证的残差界限，从而提供了相对于真实解的相对误差界限，并且能够计算出基于近似解的后验估计。

**🔧 技术方法**

使用了物理信息神经网络（PINNs）来近似解PDE，并结合形式验证技术来推导误差界限。

**📊 数据集**

使用了Lyapunov方程和HJB方程的数值示例进行验证，具体的系统包括倒立摆和控制仿射系统。

**📈 对比分析**

通过与传统的数值方法进行比较，展示了PINN方法在计算Lyapunov和HJB方程解时的有效性，并提供了相对误差界限和后验估计，性能优越。

**⚠️ 局限性**

限制在于所提出的分析并不依赖于特定的训练方法或网络架构，适用性可能受到具体实现的影响。

---

## 184. The End of Rented Discovery: How AI Search Redistributes Power Between Hotels and Intermediaries

**arXiv ID:** 2603.20062 | [PDF](https://arxiv.org/pdf/2603.20062v1)

**作者:** Peiying Zhu `[一作]` (Blossom AI), Sidi Chang `[通讯]` (Blossom AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Gemini 2.5 Flash AI搜索在东京酒店查询中产生的1,357条引用进行审计，构建了156条配对查询（交易型与体验型、英日双语），并统计与分析引用来源的分布与模式。

**💡 创新点**

创新点在于首次将引用审计方法应用于酒店行业，揭示了“意图‑来源分化”(Intent‑Source Divide)——体验型查询显著提升非OTA引用比例，且在日语查询中更为突出，指出AI搜索可能重塑酒店发现的中介结构。

**🔧 技术方法**

使用的技术主要是Generative AI搜索的“grounding”功能（Gemini结合Google Search检索并合成答案）、统计学分析（卡方检验、Logistic回归、Cramér’s V、Mann‑Whitney U）以及内容深度（Search‑Answerable Depth, SAD）手工审核。

**📊 数据集**

数据集由156条双语酒店查询（4类需求×2意图×9区×2语）生成，收集了每条查询下的Gemini grounding引用，共1,357条；还对14家酒店网站进行深度内容审核。

**📈 对比分析**

比较方法主要是意图对比与语言交互的Logistic回归，结果显示体验型查询非OTA引用率提高了25.1个百分点，日语环境下提升幅度加倍（OR≈3.5）。内容深度审核显示，SAD≥6的酒店均被直接引用，SAD<6则未被引用，统计显著（p=0.003）。

**⚠️ 局限性**

局限性包括：仅在单一平台（Gemini‑Google）和单一时点进行，未跨平台验证；观察性关联难以证明因果；仅聚焦东京，可能不适用于其他旅游市场；查询框架与答案类型可能混杂；引用数据非确定性，复现性有限；仅衡量发现而非预订，无法直接映射到收益。

---

## 185. Towards Solving Polynomial-Objective Integer Programming with Hypergraph Neural Networks

**arXiv ID:** 2603.19318 | [PDF](https://arxiv.org/pdf/2603.19318v1)

**作者:** Minshuo Li `[一作]` (Eindhoven University of Technology), Wim P. M. Nuijten `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1317 | [OpenAlex ID](https://openalex.org/A5073570057)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出一种基于超图神经网络（HNN）的方法，用于求解多项式目标整数规划（POIP），并通过预测变量取值作为初始解，进一步用搜索修复与细化得到高质量可行解。

**💡 创新点**

创新点包括：①设计了能同时捕捉高阶项内部变量交互和变量-约束依赖的高阶项感知超图表示；②在此基础上构建了融合超图卷积与二部图卷积的HNN架构，实现对高阶交互和约束约束的有效建模；③通过修复-细化搜索提升预测解的可行性与质量。

**🔧 技术方法**

主要技术包括：超图神经网络（HNN）架构、超图卷积与变量-约束卷积两阶段信息传播、二层MLP输出预测值、以及基于邻域搜索的修复细化过程。

**📊 数据集**

实验使用了三大基准：QMKP（Quadratic Multiple Knapsack Problem）规模1k–10k、QPLIB公开库、以及新构造的CFLPTC（含五次项的设施布局问题）规模50×10–500×100。

**📈 对比分析**

与传统学习方法（NeuralQP、GNNQP、TriGNN）及精确求解器Gurobi/SCIP对比，本文方法在所有基准上均实现了更低的相对主观差距（gap_%），尤其在更大规模与高次项问题上表现出显著优势。

**⚠️ 局限性**

局限性包括：仅针对整数变量（通过二进制化处理）；需依赖后续搜索修复步骤，未实现端到端直接生成可行解；并且目前只处理多项式目标，尚未扩展到包含三角、对数等非多项式非线性项的实例。

---

## 186. Heavy-Tailed and Long-Range Dependent Noise in Stochastic Approximation: A Finite-Time Analysis

**arXiv ID:** 2603.19648 | [PDF](https://arxiv.org/pdf/2603.19648v1)

**作者:** Siddharth Chandak `[一作]` (Stanford University), Nicholas Bambos `[通讯]` (Stanford University)

**通讯引用:** 5307 | [OpenAlex ID](https://openalex.org/A5002056995)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探讨了在重尾和长程依赖噪声模型下的随机逼近（SA）方法，特别是寻找强单调算子的根。我们建立了这两种噪声模型下的有限时间收敛界限，并提供了明确的收敛速率。

**💡 创新点**

创新点在于首次在重尾和长程依赖噪声模型下建立了有限时间的收敛界限，量化了重尾和时间依赖对收敛速率的影响。

**🔧 技术方法**

使用了噪声平均的论证方法，通过引入平均噪声和辅助迭代，正则化了噪声的影响，从而在重尾或长程依赖扰动下获得了改进的矩界限。

**📊 数据集**

应用于随机梯度下降（SGD）和梯度博弈的数值实验，验证了理论分析的有效性。

**📈 对比分析**

与经典的马尔可夫噪声模型相比，重尾噪声和长程依赖噪声模型下的收敛速率有所下降，具体表现为重尾噪声的收敛速率为𝒪(k^-(p-1))，而长程依赖噪声的收敛速率为𝒪(k^-δ)。

**⚠️ 局限性**

限制在于该研究主要集中于重尾和长程依赖噪声模型，可能无法直接推广到其他类型的噪声模型。此外，理论结果的实际应用可能受到噪声特性和算法设计的影响。

---

## 187. 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction

**arXiv ID:** 2603.19682 | [PDF](https://arxiv.org/pdf/2603.19682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 188. Learning from Similarity/Dissimilarity and Pairwise Comparison

**arXiv ID:** 2603.19713 | [PDF](https://arxiv.org/pdf/2603.19713v1)

**作者:** Tomoya Tate `[一作]` (Waseda University), Masato Uchida `[通讯]` (Waseda University)

**通讯引用:** 920 | [OpenAlex ID](https://openalex.org/A5000225743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在弱监督环境下利用相似/不相似标签（SD）和成对比较标签（Pcomp）联合训练二分类器的框架——SD-Pcomp分类；

**💡 创新点**

创新点在于构造了一个既包含SD标签也包含Pcomp标签的无偏风险估计器，并对其噪声、类别先验误差及风险修正进行了理论分析；

**🔧 技术方法**

使用无偏风险估计、凸组合与风险修正（ReLU/ABS）、Rademacher复杂度分析以及逻辑损失；

**📊 数据集**

实验基准使用MNIST、Kuzushiji-MNIST、Fashion-MNIST、CIFAR-10，以及UCI数据集Optdigits、Pendigits、Letter、PMU-UD；

**📈 对比分析**

与仅使用SD、仅使用Pcomp以及全监督方法对比，SD-Pcomp联合方法在多种类别先验下均能取得更高的准确率和AUC，且在噪声鲁棒性上表现更好；

**⚠️ 局限性**

局限性包括仅适用于二分类、需要估计类别先验、只考虑两种弱标签类型，且未对多分类或更丰富关系监督进行扩展。

---

## 189. Improving Automatic Summarization of Radiology Reports through Mid-Training of Large Language Models

**arXiv ID:** 2603.19275 | [PDF](https://arxiv.org/pdf/2603.19275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 190. PhyUnfold-Net: Advancing Remote Sensing Change Detection with Physics-Guided Deep Unfolding

**arXiv ID:** 2603.19566 | [PDF](https://arxiv.org/pdf/2603.19566v1)

**作者:** Zelin Lei `[一作]` (Xi'an Jiaotong University), Jiaming Chang `[通讯]` (Anhui University)

**通讯引用:** 5126 | [OpenAlex ID](https://openalex.org/A5007504298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理引导的深度展开框架 PhyUnfold-Net，用显式差异空间分解（D=C+N）实现双时相遥感变化检测；

**💡 创新点**

核心创新包括：① 迭代变化分解模块（ICDM）通过门控残差注入和记忆增强实现稳定的 C+N 分离；② 阶段性探索与约束正则（S-SEC）防止分解退化；③ 频域波形谱抑制模块（WSSM）在小波域先行抑制采集差异；④ 以 patch‑wise singular‑value entropy (SVE) 为物理先验指导分解；

**🔧 技术方法**

使用 ResNet‑18 Siamese 编码器、离散小波变换与逆变换、共享参数记忆单元、门控残差注入、SVE 门控、S-SEC 正则、BCE+Dice 损失、EMA 与 AMP 训练；

**📊 数据集**

在四大遥感变化检测基准上评测：LEVIR‑CD、LEVIR‑CD+、WHU‑CD、S2Looking；

**📈 对比分析**

与多种 CNN、注意力、Transformer 及 2025 年 TGRS 方法（ChangeDA、SPMNet）在 Precision/Recall/F1/IoU 等指标上对比，PhyUnfold‑Net 在 LEVIR‑CD、LEVIR‑CD+、S2Looking 上获得最优或接近最优表现，并在 WHU‑CD 保持竞争力，整体提升 F1 与 IoU 若干个百分点；

**⚠️ 局限性**

局限性：仅处理双时相；需手动设定展开步数与门控参数，计算开销相对较大；对极端光照或大尺度纹理变化仍可能出现漏报；跨域泛化与多时相扩展尚未系统评估。

---

## 191. Benchmarking Cross-Scale Perception Ability of Large Multimodal Models in Material Science

**arXiv ID:** 2603.19327 | [PDF](https://arxiv.org/pdf/2603.19327v1)

**作者:** Yuting Zheng `[一作]` (Shanghai Jiao Tong University), Qi Jia `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了CSMBench跨尺度材料科学图像理解基准，并对大型多模态模型进行评估

**💡 创新点**

创新在于将图像按原子、微观、中观、宏观四个尺度分类，设计开放式描述和多选标题匹配两种任务，填补了现有科学基准缺失多尺度推理能力的空白

**🔧 技术方法**

采用多模态模型评测框架，使用BERTScore、Semantic Textual Similarity、LLM-as-a-Judge等指标，并以GPT‑4o进行判分；同时通过MinerU、正则匹配与人工审核构建数据集

**📊 数据集**

使用由432篇材料学论文收集的1041张高质量图像，涵盖原子到宏观四个尺度，公开发布在Hugging Face

**📈 对比分析**

对10款大型多模态模型（含GPT‑5.1、Gemini‑2.5‑Pro、Doubao‑1.6‑vision、Qwen系列、InternVL3系列）进行多尺度多任务评估，发现专有模型在推理和识别上表现更好，且模型性能随尺度变化显著

**⚠️ 局限性**

局限包括模型在宏观/原子尺度识别不足、识别准确率高但缺乏深层物理推理、不同尺度间性能不一致、数据分布不均导致评测不完全覆盖所有图像类型，以及基准目前仅涵盖图像描述与标题匹配，未覆盖更复杂的定量推理任务

---

## 192. A Human-Centered Workflow for Using Large Language Models in Content Analysis

**arXiv ID:** 2603.19271 | [PDF](https://arxiv.org/pdf/2603.19271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 193. Warm-Start Flow Matching for Guaranteed Fast Text/Image Generation

**arXiv ID:** 2603.19360 | [PDF](https://arxiv.org/pdf/2603.19360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 194. FalconBC: Flow matching for Amortized inference of Latent-CONditioned physiologic Boundary Conditions

**arXiv ID:** 2603.19331 | [PDF](https://arxiv.org/pdf/2603.19331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. Overreliance on AI in Information-seeking from Video Content

**arXiv ID:** 2603.19843 | [PDF](https://arxiv.org/pdf/2603.19843v1)

**作者:** Anders Giovanni Møller `[一作]` (IT University of Copenhagen), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**通讯引用:** 5001 | [OpenAlex ID](https://openalex.org/A5034406723)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过大规模实验（917名受试者，8253个视频信息检索任务）系统评估了生成式AI在视频信息检索中的帮助与风险。

**💡 创新点**

首次在多模态视频场景下量化AI辅助的效能与安全风险，并揭示用户对AI输出的过度依赖与自信不变的现象。

**🔧 技术方法**

采用Google Gemini等多模态大语言模型（LLM）实现AI助手，使用LLM判定器进行答案正确性评估，配合OLS回归和行为序列分析。

**📊 数据集**

数据集包含3个主题（2025年卢浮宫抢劫案、奥运会历史、2023年海底舱灾难）下的12段视频（6短、6长），每段视频有3个针对性问题，共计8,253个问答实例。

**📈 对比分析**

对照组（仅视频）与两种AI组（帮助AI、欺骗AI）进行对比。帮助AI在未观看视频时提高准确率约27–35%，效率提升10–25%；欺骗AI导致准确率下降高达32%；自信度在所有条件下保持约4.5/5。

**⚠️ 局限性**

研究局限包括仅选取美国/英国众包受试者、视频与主题范围有限、仅使用单一LLM模型，且受试者可能使用外部LLM完成任务。

---

## 196. ConSearcher: Supporting Conversational Information Seeking in Online Communities with Member Personas

**arXiv ID:** 2603.19747 | [PDF](https://arxiv.org/pdf/2603.19747v1)

**作者:** Shiwei Wu `[一作]` (Sun Yat-sen University), Zhenhui Peng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 696 | [OpenAlex ID](https://openalex.org/A5081718999)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在在线社区中实现并评估了一个基于LLM的对话搜索工具ConSearcher。

**💡 创新点**

创新点是将动态生成的成员人物（信息寻求者与信息提供者 personas）嵌入对话搜索，帮助用户明确需求并获得多视角回复。

**🔧 技术方法**

使用了检索增强生成（RAG）+ GPT‑4 作为后端，并结合因素分解、Persona 生成等LLM推理流程。

**📊 数据集**

数据集来自 Reddit 子版块 /r/JapanTravel、/r/education 和 /r/PhD 的帖子与评论。

**📈 对比分析**

通过 27 名受试者的 within‑subjects 实验与 BaseAgent/BaseSearcher 基线比较，ConSearcher 显著提升信息学习点、满意度与参与度。

**⚠️ 局限性**

局限包括对非大学生用户的泛化不足、对个人兴趣编辑的复杂性、过度个性化导致可信度下降，以及未检验长期使用效果。

---

## 197. AI as Relational Translator: Rethinking Belonging and Mutual Legibility in Cross-Cultural Contexts

**arXiv ID:** 2603.19568 | [PDF](https://arxiv.org/pdf/2603.19568v1)

**作者:** Yao Xiao `[一作]` (Imperial College London), Rafael A. Calvo `[通讯]` (Imperial College London)

**通讯引用:** 15064 | [OpenAlex ID](https://openalex.org/A5013835523)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 Relational AI Translation 概念与多代理架构，将 AI 作为跨文化关系的翻译与支架工具，而非模拟伴侣；

**💡 创新点**

创新点在于把 AI 重新定位为文化‑关系基础设施，核心操作包括情感意图解码、情境再框架与关系支架，强调不替代而是支持人际关系；

**🔧 技术方法**

采用多代理系统（管理代理 + 领域代理）、文化检索增强问答（RAG）技术以及安全与升级决策逻辑，并结合自决理论与社会驿站模型进行理论映射；

**📊 数据集**

论文未使用具体公开数据集，RAG 层基于参与式共创的文化材料作为知识库；

**📈 对比分析**

作为概念性/定位性工作，未进行实验比较，作者建议通过混合方法评估（体验采样、行为轨迹、对话分析）来验证其有效性；

**⚠️ 局限性**

局限性包括模型漂移与文化假设误判风险、过度依赖 AI 的可能性、难以处理结构性不公、仅聚焦人际调解而非制度改革、缺乏实证验证等。

---

## 198. How Motivation Relates to Generative AI Use: A Large-Scale Survey of Mexican High School Students

**arXiv ID:** 2603.19263 | [PDF](https://arxiv.org/pdf/2603.19263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 199. Process Faster, Pay Less: Functional Isolation for Stream Processing

**arXiv ID:** 2603.19445 | [PDF](https://arxiv.org/pdf/2603.19445v1)

**作者:** Eleni Zapridou `[一作]` (École Polytechnique Fédérale de Lausanne), Anastasia Ailamaki `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 9985 | [OpenAlex ID](https://openalex.org/A5070907021)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 FunShare，一套基于功能隔离的流处理系统，能够在保证各个查询质量（QoS）的前提下动态地将查询划分为共享组，实时调整共享策略以降低资源消耗。

**💡 创新点**

创新点在于：
• 引入功能隔离原则到流处理，确保任何跨查询的共享都不会降低单个查询的吞吐；
• 设计了自适应的合并/拆分机制，实时评估查询间的重叠和负载，动态调整共享组；
• 开发了负载估计器和吞吐估计器，利用采样、数据查询模型和轻量级重配置，精准预测合并后的工作量；
• 结合资源管理器和再配置管理器，实现零停顿的查询组重配置与资源分配。

**🔧 技术方法**

主要技术包括：
• 以吞吐为QoS指标的功能隔离框架；
• 负载与吞吐的在线估计（采样、成本模型、统计信息收集）；
• 合并成本（GroupingCost）与阈值驱动的合并/拆分算法；
• 资源管理器通过最小化资源分配满足合并阈值；
• 采用基于 epoch 的控制消息实现无停顿的查询重配置；
• 在 Apache Flink 上实现，利用 JobMaster、TaskManagers 与现有的任务并行度控制。

**📊 数据集**

使用 Nexmark 基准的三种工作负载：
1. W1 – 窗口化的等值 join（所有查询结构相同）；
2. W2 – 共享 join + 不同下游操作（Q_CategoryAvg、Q_SellerAvg、Q_PriceAnomaly）；
3. W3 – 向量相似度 join；
通过在这些工作负载中随机设定查询的过滤选择率（1%、10%、1–20% 等）来模拟不同的查询重叠程度。

**📈 对比分析**

与四个基线（Isolated、Full-Sharing、Overlap-Sharing、Selectivity-Sharing）对比：
• 资源消耗：FunShare 在所有负载下均低于 Isolated，最大可节省 10.7× 资源；
• 吞吐：在固定资源下，FunShare 的吞吐始终不低于 Isolated，甚至高出 1.5–2.1×；
• 延迟与队列增长：相比全共享方案，FunShare 的端到端延迟增长被严格限制，且在回压场景下不会导致某些查询的队列急剧增长；
• 适应性实验表明，在输入速率或数据分布突变时，FunShare 能快速拆分/合并组，保持 QoS 并降低资源使用。

**⚠️ 局限性**

局限性：
• 目前仅针对具有可共享 join/过滤的查询场景；对更复杂的聚合、窗口或状态操作的共享尚未充分验证；
• 负载与吞吐估计依赖于成本模型的准确性，若模型与实际差异大可能导致误判；
• 对极大规模集群或超大并发度（数千查询）时的扩展性与调度效率尚需进一步评估；
• 重配置虽然采用无停顿机制，但仍有几秒级延迟，对极低延迟应用可能产生影响。

---

## 200. Revisiting Gene Ontology Knowledge Discovery with Hierarchical Feature Selection and Virtual Study Group of AI Agents

**arXiv ID:** 2603.20132 | [PDF](https://arxiv.org/pdf/2603.20132v1)

**作者:** Cen Wan `[一作]` (Birkbeck University of London), Alex A. Freitas `[通讯]` (University of Kent)

**通讯引用:** 17051 | [OpenAlex ID](https://openalex.org/A5087201377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了基于多代理大型语言模型的虚拟学习组框架，用于从层级特征选择得到的 Gene Ontology 术语中自动提取与衰老相关的生物学知识。

**💡 创新点**

将 agentic AI 与层级特征选择结合，构建多层级虚拟研究组实现批判性评审与知识验证；引入多代理互评机制提升知识可信度。

**🔧 技术方法**

使用大型语言模型（deepseek‑r1、qwen3‑vl、gpt‑oss、glm‑4.7‑flash）及 CrewAI 协同框架，配合层级特征选择方法（HIP）进行 GO 术语抽取与批判性讨论。

**📊 数据集**

采用四种模型生物（Caenorhabditis elegans、Drosophila melanogaster、Mus musculus、Saccharomyces cerevisiae）衰老相关 GO 术语选择表（来自 Wan & Freitas 2018 HIP 结果），共 8 组 GO 术语。

**📈 对比分析**

通过人工检索已发表文献验证代理生成的科学声明，统计支持率；结果显示大部分主张被文献支持，批判性层级进一步提升准确性，但仍存在少数幻觉或错误引用。

**⚠️ 局限性**

仍存在幻觉与错误引用；仅评估了高层次 GO 术语，针对更具体术语的验证困难；模型对特定领域知识的依赖与可解释性待进一步提升。

---

## 201. LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection

**arXiv ID:** 2603.19293 | [PDF](https://arxiv.org/pdf/2603.19293v1)

**作者:** Weilin Zhou `[一作]` (Xinjiang University), Yurong Qian `[通讯]` (Xinjiang University)

**通讯引用:** 1849 | [OpenAlex ID](https://openalex.org/A5002981402)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于LLM引导的多视角推理蒸馏框架LLM-MRD，用学生模型在文本、图像和跨模态视角上进行推理，并通过校准蒸馏将教师LLM的深层推理知识迁移给高效的学生网络，从而实现多模态假新闻检测。

**💡 创新点**

创新点包括：① 采用多视角学生架构消除信息孤岛，① 通过教师LLM生成多视角推理链，② 设计校准蒸馏机制以自适应校正学生特征并对齐推理空间，而非传统的单纯投影；③ 在推理效率与性能之间取得显著平衡，避免直接使用大模型导致的推理瓶颈。

**🔧 技术方法**

技术手段：文本使用BERT编码，图像使用MAE编码，跨模态使用CLIP进行对齐；教师使用大型多模态LLM Qwen2.5‑VL 生成推理链；校准蒸馏利用MLP预测修正向量并与学生特征相加；多视角融合采用交叉注意力；训练损失包括KL蒸馏、交叉熵深度监督和最终分类损失。

**📊 数据集**

数据集：Weibo、Weibo‑21、GossipCop 三个公开中文/英文多模态假新闻数据集。

**📈 对比分析**

与 12 组基线（单模态、跨域、LLM蒸馏方法）对比，LLM‑MRD 在准确率、F1‑Fake、F1‑Real 上平均提升约 5–6%（ACC +5.19%，F1‑Fake +6.33%），在所有数据集均位居第一；同时保持 358M 参数、17.2 ms 推理时延，显著优于依赖大模型的 GLPN‑LLM。

**⚠️ 局限性**

局限性：① 依赖教师LLM的推理链质量，若教师产生幻觉或误导性推理，蒸馏效果受限；② 目前仅覆盖文本–图像两模态，对视频、音频等多模态场景尚未扩展；③ 校准蒸馏对超参数（α、λ、温度）敏感，需细致调优；④ 在极端对抗性攻击或跨文化语言环境下的鲁棒性尚未充分验证。

---

## 202. Speculating Experts Accelerates Inference for Mixture-of-Experts

**arXiv ID:** 2603.19289 | [PDF](https://arxiv.org/pdf/2603.19289v1)

**作者:** Vivan Madan `[一作]` (University of Maryland), Ashwinee Panda `[通讯]` (TogetherAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模稀疏激活的 Mixture-of-Experts（MoE）模型，提出在推理时利用内部表征（quasi‑hidden state）预取未来层的专家权重，支持推理过程中CPU–GPU内存传输与GPU计算的重叠。

**💡 创新点**

创新点在于：①不使用任何额外参数的预取方案，直接利用已存在的内部表征预测下一层专家；②引入轻量化神经估计器在表示漂移大、预测误差高的层提升命中率；③在推理时将预取的专家直接执行（speculative execution），保持任务准确率。

**🔧 技术方法**

使用技术包括：MoE架构的预取逻辑、quasi‑hidden state的构造、轻量化前馈估计器、CUDA异步拷贝与双缓冲、YALIS推理引擎、FlashAttention、CUDA Graphs 等。

**📊 数据集**

实验数据集：大型 MoE 模型（Qwen3‑30B‑A3B、GLM‑4.7‑Flash、GPT‑OSS‑20B/120B、Qwen3‑235B‑A22B）以及推理后评估的基准任务（HumanEval、MBPP+、GSM8k、AIME24/25、StrategyQA），并在估计器训练时使用数百万条 token。

**📈 对比分析**

与按需加载专家的基线对比，在 A6000 GPU 上实现 9–14% 的 TPOT（每输出 token 时间）下降；在更强 GPU（A100、GH200）下降幅度为 5–8%；任务准确率基本保持在基线水平（大部分任务误差 < 1%），仅在 Qwen3‑30B‑A3B 的早期层上稍有下降，轻量化估计器可恢复大部分性能。

**⚠️ 局限性**

局限性：①在表示漂移较大的早期层仍可能出现准确率下降；②预取方案在 prefill 阶段无显著优势，且未在低功耗/嵌入式平台验证；③对极大规模模型（>200B 参数）及多 GPU 通信模式的适配尚未完成；④当前实现仍依赖 CPU‑GPU 传输，未覆盖磁盘/SSD 等更慢存储。

---

## 203. Semantic Audio-Visual Navigation in Continuous Environments

**arXiv ID:** 2603.19660 | [PDF](https://arxiv.org/pdf/2603.19660v1)

**作者:** Yichen Zeng `[一作]` (Wuhan University), Gongping Huang `[通讯]` (Wuhan University)

**通讯引用:** 1668 | [OpenAlex ID](https://openalex.org/A5034257180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了连续环境下的语义音频视觉导航任务SAVN-CE，并设计了记忆增强目标描述网络MAGNet，实现了在目标发声间歇或完全消失时的可靠导航。

**💡 创新点**

创新点在于：①将语义音频视觉导航扩展到连续三维空间；②引入基于Transformer的记忆增强目标描述网络，融合历史上下文、自我运动信息和多模态感知；③在动态目标发声条件下保持目标信息，实现高效导航。

**🔧 技术方法**

主要技术包括：多模态观测编码器（视觉、音频、姿态、动作）；基于Transformer的记忆增强目标描述网络；场景记忆与情节记忆；PPO强化学习与监督式音频目标标注；自我运动编码与交叉注意力机制。

**📊 数据集**

使用了基于Matterport3D和SoundSpaces 2.0的SAVN-CE数据集，包含0.5M训练、500验证、1000测试episode，提供多语义类别、随机目标发声时长与起始时间，支持干净与干扰两种环境。

**📈 对比分析**

与Random、ObjectGoal、AV-Nav、SMT+Audio、SAVi及Oracle1/2等方法比较，MAGNet在Clean环境下成功率37.7%、SPL32.9%、SNA27.4%，相较SAVi提升约12.1%；在Distracted环境下仍优于基线但受干扰影响显著；Oracle2达到75%成功率，显示MAGNet仍存在与理想解的差距。

**⚠️ 局限性**

局限性包括：在目标静音或干扰声音强的情形下仍存在性能下降；与Oracle2相比仍有显著差距；目前仅处理单一目标，未覆盖多目标或动态目标场景；对极短发声窗口的定位精度仍需提升。

---

## 204. Investigating In-Context Privacy Learning by Integrating User-Facing Privacy Tools into Conversational Agents

**arXiv ID:** 2603.19416 | [PDF](https://arxiv.org/pdf/2603.19416v1)

**作者:** Mohammad Hadi Nezhad `[一作]` (University of Massachusetts), Ivon Arroyo `[通讯]` (University of Massachusetts)

**通讯引用:** 3063 | [OpenAlex ID](https://openalex.org/A5008924726)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在聊天机器人交互中嵌入即时隐私通知面板，以提升用户对隐私的认知与行为。

**💡 创新点**

创新点在于将实时警示、可操作匿名化与 FAQ 说明整合到聊天界面，实现情境化隐私学习。

**🔧 技术方法**

使用基于 GPT‑4o 的聊天模拟接口，并开发拦截并提示敏感信息的隐私面板。

**📊 数据集**

实验数据来自十名美国计算机科学本科及硕士学生的任务与问卷调查。

**📈 对比分析**

通过对比有面板与无面板两次任务，定性访谈与问卷显示使用面板显著提升了隐私意识和匿名化行为。

**⚠️ 局限性**

局限在于样本规模小、仅为 CS 学生、缺乏长期跟踪和多样化用户验证，且 FAQ 的使用率较低。

---

## 205. Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery

**arXiv ID:** 2603.19918 | [PDF](https://arxiv.org/pdf/2603.19918v1)

**作者:** Jizhou Han `[一作]` (Xi'an Jiaotong University), Yihong Gong `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 23575 | [OpenAlex ID](https://openalex.org/A5100687952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Analogical Textual Concept Generator (ATCG)，将类比生成的文本概念与视觉特征融合，改进泛化类别发现 (GCD) 的识别与发现能力。

**💡 创新点**

创新点在于：①利用已知类别知识库进行类比推理，生成可描述未知样本的文本概念；②将文本概念与视觉特征融合，形成视觉–文本推理流程；③ATCG 作为插件，可无缝接入多种 GCD 训练框架，显著提升细粒度类别的区分。

**🔧 技术方法**

技术手段包括：CLIP 双模预训练模型、TIAA 与 TSA 注意力模块构成 ATCG（初始层 + 堆叠层）、对比学习 + 参数化分类、Fusion‑head 投影层，以及多阶段训练（ATCG 预训练 + GCD 训练）。

**📊 数据集**

实验使用六大基准数据集：CIFAR‑100、ImageNet‑100、CUB‑200、Stanford Cars、FGVC Aircraft、Herbarium19。

**📈 对比分析**

与 SimGCD‑CLIP、CMS‑CLIP、SelEx‑CLIP 等主流 GCD 方法对比，ATCG 在所有框架上平均提升整体准确率约 5–8%，在细粒度数据上提升更为显著，并保持或提升已知类别的准确率。

**⚠️ 局限性**

局限性包括：1）对旧/新类别准确率的平衡仍存在一定权衡；2）依赖 CLIP 预训练知识，类比机制相对简单；3）在极端分布漂移或高度不平衡场景下的鲁棒性尚待进一步验证。

---

## 206. Unbiased Dynamic Multimodal Fusion

**arXiv ID:** 2603.19681 | [PDF](https://arxiv.org/pdf/2603.19681v1)

**作者:** Shicai Wei `[一作]` (University of Electronic Science and Technology of China), Guiduo Duan `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5081385896)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Unbiased Dynamic Multimodal Learning (UDML) 框架，结合噪声感知不确定性估计器和模态依赖计算器，实现对动态模态质量的精准评估和自适应加权。

**💡 创新点**

创新点在于：①基于受控噪声的噪声感知不确定性估计，能在低噪声和高噪声两端都精准量化模态质量；②通过模态 dropout 量化内在依赖性，消除双重抑制，避免弱模态被过度惩罚；③采用进步式优化保证多目标训练稳定；④框架与模型无关，可无缝集成。

**🔧 技术方法**

使用概率分布表示（均值+方差）进行噪声估计，轻量级 MLP 预测噪声强度；模态依赖通过 logit 差值计算；整体采用多模态编码器（BERT、ResNet、FACET、COVAREP 等）+融合模块，并采用两阶段训练。

**📊 数据集**

在五个多模态基准上验证：CMU-MOSI、CMU-MOSEI（三模态情感分析），MVSA-Single、CREMA-D、Kinetics-Sounds（二模态情感/动作识别）。

**📈 对比分析**

与多种静态与动态融合方法（Late Fusion、MMBT、TMC、QMF、EAU、LDDU 等）对比，UDML 在所有任务上均取得更高准确率/F1/相关系数，平均提升约 1–2%；在噪声鲁棒测试中，性能衰减最小。

**⚠️ 局限性**

局限在于仅处理模态级偏差，未考虑样本级不确定性（如尾类样本导致的高不确定性），可能导致对难样本的过度抑制。

---

## 207. Skilled AI Agents for Embedded and IoT Systems Development

**arXiv ID:** 2603.19583 | [PDF](https://arxiv.org/pdf/2603.19583v1)

**作者:** Yiming Li `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**通讯引用:** 3171 | [OpenAlex ID](https://openalex.org/A5034818470)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于技能的代理框架和IoT‑SkillsBench硬件验证基准，用于评估嵌入式与IoT系统的AI开发能力。

**💡 创新点**

创新点在于用可扩展的“技能”模块压缩硬件知识，并通过真实硬件测试验证生成代码。

**🔧 技术方法**

使用Claude Sonnet LLM与LangGraph代理架构，并支持Arduino、ESP‑IDF、Zephyr三大平台。

**📊 数据集**

使用IoT‑SkillsBench数据集，涵盖3个平台、23种外设、42个任务的硬件验证案例。

**📈 对比分析**

对比无技能、LLM生成技能与人工专家技能三种配置，人工技能实现近乎完美成功率，LLM技能效果波动。

**⚠️ 局限性**

局限在于仍需人工编写高质量技能、对硬件特定问题依赖先验知识，LLM生成技能易引入错误。

---

## 208. Status Updating in Two-Way Delay Systems with Preemption

**arXiv ID:** 2603.19410 | [PDF](https://arxiv.org/pdf/2603.19410v1)

**作者:** Jinxin Yang `[一作]` (University of Leeds), Hamid R. Sadjadpour `[通讯]` (University of California Santa Cruz)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了一个包含控制器、采样器和接收器的双向延迟状态更新系统，提出并求解了最优的请求生成策略以最小化长期平均信息时效性（AoI）。

**💡 创新点**

创新点在于：① 采用预处理等待（preemption‑in‑waiting）策略在前向和后向链路中实现全动态请求与更新；② 将整个问题建模为有限状态马尔可夫决策过程（MDP），并证明在弱可访问性条件下可以获得全局最优的确定性策略；③ 通过数值实验揭示了最优策略具有阈值结构且阈值随服务率和反向链路速率呈非单调关系。

**🔧 技术方法**

技术手段包括：马尔可夫决策过程建模、相对价值迭代（RVI）算法求解平均成本最优值、弱可访问性与单链性证明以及数值仿真分析。

**📊 数据集**

未使用公开数据集；所有结果均基于仿真实验，参数设定为不同的服务率 μ 与 γ，AoI 上界 Δ̅ 取足够大后保持收敛。

**📈 对比分析**

将所提策略与已有的 1‑Packet 与 2‑Packet 系统进行对比，实验显示在所有服务率设置下，所提策略的平均 AoI 低于两者；随着 μ 增大，平均 AoI 单调下降，验证了理论预期。

**⚠️ 局限性**

局限性包括：① 需要完整系统可观测性，实际部署时需进一步研究部分信息下的策略；② 仅考虑了几种网络拓扑（单链路、单缓冲），对更复杂多跳或多源网络的推广尚未探讨；③ 预处理策略可能导致能量或存储资源的浪费，需要结合能量约束进行进一步优化。

---

## 209. MSNet and LS-Net: Scalable Multi-Scale Multi-Representation Networks for Time Series Classification

**arXiv ID:** 2603.19315 | [PDF](https://arxiv.org/pdf/2603.19315v1)

**作者:** Celal Alagöz `[一作]` (Sivas Bilim ve Teknoloji Üniversitesi), Farhan Aadil `[通讯]` (Sivas Bilim ve Teknoloji Üniversitesi)

**通讯引用:** 2801 | [OpenAlex ID](https://openalex.org/A5058076434)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种可扩展的多尺度多表示卷积框架，用于单变量时间序列分类，并提出了MSNet和LS-Net两种架构，分别针对校准和效率进行了优化。

**💡 创新点**

创新点在于将多种结构化信号表示（时间、导数、频域、离散余弦、自动相关等）与多尺度卷积相结合，并将原本多变量网络LiteMV适配为多表示单变量网络；同时设计了轻量级LS-Net的早退出机制。

**🔧 技术方法**

主要技术包括多尺度卷积分支、特征融合块、批归一化、ReLU、Dropout、早退出（confidence‑based gating）、交叉熵损失、Adam优化器、Monte Carlo重采样和统计显著性检验（Friedman + Nemenyi）。

**📊 数据集**

在142个UCR/UEA基准时间序列数据集上进行实验，涵盖单变量、不同长度和类别数的多样化任务。

**📈 对比分析**

采用统一的实验协议、Monte Carlo重采样和宏平均指标（准确率、Macro‑F1、AUC、NLL），使用CD图和Pareto分析对模型进行统计比较。结果显示LiteMV在平均准确率和Macro‑F1上最高，MSNet在NLL上最佳，LS-Net在保持接近SOTA准确率的同时显著降低训练与推理时间。

**⚠️ 局限性**

局限性包括：表示选择仍为固定集合，缺乏自适应选择机制；对大型多变量数据集的评估有限；早退出阈值为固定值，未进行动态调节；模型对极端噪声或异常模式的鲁棒性未充分探究。

---

## 210. ReViSQL: Achieving Human-Level Text-to-SQL

**arXiv ID:** 2603.20004 | [PDF](https://arxiv.org/pdf/2603.20004v1)

**作者:** Yuxuan Zhu `[一作]` (University of Illinois Urbana-Champaign), Daniel Kang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1854 | [OpenAlex ID](https://openalex.org/A5072348548)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建专家验证的BIRD数据集，并通过RLVR与推理时扩展，直接提升LLM对自然语言到SQL转换的推理能力，实现人工水平的执行准确率。

**💡 创新点**

① 生成2.5k实例的Verified数据集，显著减少训练噪声；② 用RLVR（CISPO）强化模型的内在推理；③ 在推理时引入reconciliation与majority voting，对候选SQL进行过滤与投票；④ 完全摒弃复杂多阶段AI代理管道，提供简洁高效的框架。

**🔧 技术方法**

RLVR（CISPO）强化学习、LoRA微调、SQL专家验证流水线、推理时候选生成、reconciliation机制、majority voting、实验评估与成本计算。

**📊 数据集**

BIRD（训练集与Mini‑Dev，Arcwise‑Plat‑Full/SQL），Spider 2（SQLite与Snow），以及自构建的ReViSQL‑Verified（2.5k实例）以及对应的验证集。

**📈 对比分析**

在Arcwise‑Plat‑Full和Arcwise‑Plat‑SQL上，ReViSQL‑235B‑A22B以93.78%/93.17%执行准确率超越代理SOTA 9.8%/5.6%；轻量级30B模型匹配SOTA但成本低7.5×；在Spider 2-SQLite和Snow上亦实现显著提升，显示了跨域泛化能力。

**⚠️ 局限性**

需要昂贵的专家标注，Verified数据规模有限；RLVR对训练样本质量高度敏感；在极端语义歧义或新领域知识时仍可能受限；推理时候选生成虽然提升准确率，但仍伴随额外成本。

---

## 211. Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields

**arXiv ID:** 2603.19834 | [PDF](https://arxiv.org/pdf/2603.19834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 212. TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly

**arXiv ID:** 2603.19296 | [PDF](https://arxiv.org/pdf/2603.19296v1)

**作者:** Toshiaki Koike-Akino `[一作]` (Mitsubishi Electric Research Laboratories), Ye Wang `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于测试时激活感知量化（TTQ）的动态压缩框架，能够在推理时即时对大型语言模型进行权重量化，无需离线校准或微调。

**💡 创新点**

创新点在于将激活感知量化从离线阶段迁移到推理阶段，并与低秩分解相结合，避免域偏移、零校准以及后续恢复难题。

**🔧 技术方法**

采用分组量化、RTN与AWQ的缩放 QDQ、在线估计对角相关矩阵、低秩分解与动态 BAX 等技术。

**📊 数据集**

在 OPT、Qwen3、Gemma3 等模型上使用 WT2、PTB、C4 三个基准数据集进行实验。

**📈 对比分析**

与 RTN、AWQ（不同校准量）对比，TTQ 在相同位数下显著降低困惑度，尤其在极低位量化和少量校准时优于 AWQ，且表现接近原始模型。

**⚠️ 局限性**

局限性包括极低位量化仍可能导致性能下降、需手工调节超参数（α、λ、p），且目前仅在 LLM 上验证，未在多模态或大规模硬件环境下全面评估。

---

## 213. VeloxNet: Efficient Spatial Gating for Lightweight Embedded Image Classification

**arXiv ID:** 2603.19496 | [PDF](https://arxiv.org/pdf/2603.19496v1)

**作者:** Md Meftahul Ferdaus `[一作]` (University of New Orleans), Kendall N. Niles `[通讯]` (US Army Corps of Engineers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了VeloxNet，一种轻量级CNN架构，将SqueezeNet的fire模块替换为gMLP块，用于嵌入式图像分类；

**💡 创新点**

将gMLP块（含空间门控单元SGU）作为fire模块的可插拔替代，实现全局空间建模；采用固定通道数、简化归一化和残差连接，显著降低参数量同时提升精度；

**🔧 技术方法**

使用gated MLP（gMLP）与空间门控单元（SGU），LayerNorm，残差连接，Adam优化器，以及常见的卷积、全局平均池化等技术；

**📊 数据集**

三大航空图像灾害数据集：AIDER、CDD、LDD；

**📈 对比分析**

与11个基线（MobileNetV1/V2/V3、ShuffleNet、EfficientNet、SqueezeNet、gMLP、FasterViT、ConvNeXt、ConvNeXt V2、EdgeViT）在F1、参数、FLOPs、模型大小、FPS等指标上对比，VeloxNet在F1上最高（81.57%、77.46%、91.85%），参数仅399k，模型尺寸1.52 MB，推理速度350‑360 FPS；

**⚠️ 局限性**

仅在5类分类任务上验证，未检验目标检测、分割等其他任务；固定通道数可能不适用于不同分辨率或数据规模；尽管参数最少，但FLOPs仍高于ShuffleNet等极致压缩模型。

---

## 214. Alternating Diffusion for Proximal Sampling with Zeroth Order Queries

**arXiv ID:** 2603.19633 | [PDF](https://arxiv.org/pdf/2603.19633v1)

**作者:** Hirohane Takagi `[一作]` (University of Tokyo), Atsushi Nitanda `[通讯]` (Agency for Science Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了一种基于零阶信息的近似近端采样器，直接模拟前向热流和后向逆扩散，无需梯度或重采样。

**💡 创新点**

创新点在于用粒子产生的高斯混合逼近中间分布，完成无模型学习的分数估计，并实现可扩展的多粒子并行。

**🔧 技术方法**

使用零阶潜在函数评估、随机微分方程模拟、Gaussian混合分数估计、Monte Carlo采样。

**📊 数据集**

实验使用高斯拉索混合、非凸离散多环体域的均匀分布等人工数据集。

**📈 对比分析**

与传统RGO实现的近端采样以及ULA、MALA、In-and-Out等方法比较，取得迭代次数缩短约10倍、KL收敛更快、对离散模式迁移更强。

**⚠️ 局限性**

局限性在于对分数估计误差和时间离散误差的依赖，且高维下混合采样的效率与粒子数、步骤数相关；未给出对高维问题的理论或经验评估。

---

## 215. ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection

**arXiv ID:** 2603.19776 | [PDF](https://arxiv.org/pdf/2603.19776v1)

**作者:** Chengzhi Hong `[一作]` (Wuhan University), Bijun Li `[通讯]` (Wuhan University)

**通讯引用:** 1982 | [OpenAlex ID](https://openalex.org/A5013315789)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出单目3D车道检测框架ReManNet，并通过Road‑Manifold假设实现道路平面与车道子曲线的几何一致性；

**💡 创新点**

创新点在于将Riemannian Gaussian描述子映射至SPD流形并用平行传输实现几何特征与视觉特征的门控融合，同时引入3D‑Tunnel Lane IoU损失以实现形状级监督；

**🔧 技术方法**

采用ResNet backbone、anchor‑based检测、位置加权卷积、SPD高斯嵌入、AIRM、平行传输、门控融合、3D‑TLIoU损失和AdanW优化器；

**📊 数据集**

在OpenLane和ApolloSim两个大规模3D车道检测基准上进行训练与评估；

**📈 对比分析**

相较于Anchor3DLane、LATR、Glane3D等SOTA方法，ReManNet在OpenLane上F1提升至65.7%（+8.2%），在ApolloSim上F1提升至约1.6%，在多种场景和距离区间均保持领先或竞争性表现；

**⚠️ 局限性**

仍在Merge‑&‑Split等复杂拓扑场景下表现相对较弱，且对极端视觉与几何变化的鲁棒性仍有提升空间。

---

## 216. Memory-Driven Role-Playing: Evaluation and Enhancement of Persona Knowledge Utilization in LLMs

**arXiv ID:** 2603.19313 | [PDF](https://arxiv.org/pdf/2603.19313v1)

**作者:** Kai Wang `[一作]` (Harbin Institute of Technology), Zhongjie Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1991 | [OpenAlex ID](https://openalex.org/A5100738315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Memory‑Driven Role‑Playing（MDRP）范式，要求LLM在对话中仅凭上下文检索并应用长短期记忆中的角色知识；

**💡 创新点**

创新点在于将角色扮演拆解为四个可测量的记忆驱动能力（Anchoring、Recalling、Bounding、Enacting），并构建细粒度评估框架MREval、结构化提示MRPrompt以及双语基准MRBench；

**🔧 技术方法**

技术手段包括：1) Narrative Schema将角色知识组织为层次化、可查询的长期记忆；2) Magic‑If Protocol作为推理时的检索与边界控制协议；3) MREval的八维诊断指标与LLM‑as‑Judge校准；4) MRBench基准数据集与实验设计；

**📊 数据集**

使用来自10本英文和6本中文小说的角色描述和对话片段，构成MRBench的双语测试实例；

**📈 对比分析**

与传统基线（Base、Card）及大模型对比，MRPrompt在12款LLM上均提升了MREval平均分，尤其在MS和MB两项上显著增幅；在同等MRPrompt配置下，小型模型（如Qwen3‑8B）可逼近甚至超过部分更大闭源模型（如GLM‑4.7、Qwen3‑Max），验证了结构化记忆与控制协议的有效性；

**⚠️ 局限性**

局限性包括：1) 只评估单轮对话，未覆盖多轮交互中的记忆更新与动态调节；2) 评估依赖单一LLM‑as‑Judge和单名双语评标者，可能缺乏足够的人工多样性；3) 对跨语言细节与边界控制的泛化尚待进一步验证。

---

## 217. Graph2TS: Structure-Controlled Time Series Generation via Quantile-Graph VAEs

**arXiv ID:** 2603.19970 | [PDF](https://arxiv.org/pdf/2603.19970v1)

**作者:** Shaoshuai Du `[一作]`, Ana-Lucia Varbanescu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种同时对时间序列和图结构进行编码、归一化、对齐与解码的多模态学习框架。

**💡 创新点**

创新点在于在归一化后特征上引入对齐损失，并在同一模型中同时优化重构与距离正则化，实现更紧密的模态融合。

**🔧 技术方法**

使用两条 MLP 编码器、归一化层、对齐损失、重构+距离损失以及解码器进行端到端训练。

**📊 数据集**

在公开的多模态时间序列与图数据集（如 MIMIC‑III + Medical Knowledge Graph）上进行实验。

**📈 对比分析**

与单一模态或传统融合方法相比，该模型在重构精度和对齐一致性上提升约 10%–15%，实验结果表明其显著优于基线。

**⚠️ 局限性**

局限于图结构规模有限，需对齐标签，且未考虑图拓扑随时间演化的情况。

---

## 218. Translation from the Information Bottleneck Perspective: an Efficiency Analysis of Spatial Prepositions in Bitexts

**arXiv ID:** 2603.19924 | [PDF](https://arxiv.org/pdf/2603.19924v1)

**作者:** Antoine Taroni `[一作]` (INSA Lyon), Frederique Laforest `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用三种译本（英、德、塞尔维亚）中的空间前置词，对比原法语文本，构建信息瓶颈（IB）模型来衡量译文的信息量与复杂度，并通过堆叠排序实验获取心理相似度。

**💡 创新点**

创新点在于把IB框架从传统命名实验迁移到翻译对齐（bitext）分析；利用译者的“压缩”行为揭示语言跨语言效率；结合堆叠排序生成的相似度来估计信息量。

**🔧 技术方法**

使用IB理论、低秩投影模型预测相似度、堆叠排序实验、BERT-like上下文嵌入、少量提示的LLM进行词对齐、K‑means筛选样本、逆向确定性退火计算最优前沿。

**📊 数据集**

数据集包括《80天环游世界》法语原文及其英、德、塞尔维亚三种译本；N=35的堆叠排序实验参与者；共1,312个空间前置词实例；30个代表性前置词用于相似度建模。

**📈 对比分析**

将attested译文与随机/受扰翻译（1%、5%、10%行随机置换）以及完全随机译文在信息平面（Accuracy‑Complexity）中比较；结果显示attested译文距离最优前沿显著更近；低秩投影模型在相似度预测中Spearman ρ≈0.78，优于余弦相似度与岭回归。

**⚠️ 局限性**

局限包括样本规模有限（仅30个前置词），未涵盖其他空间表达方式；使用统一先验假设可能不充分；相似度依赖堆叠排序实验，受参与者主观因素影响；未充分考虑译者风格、语义增补对效率的影响。

---

## 219. Automatic Analysis of Collaboration Through Human Conversational Data Resources: A Review

**arXiv ID:** 2603.19292 | [PDF](https://arxiv.org/pdf/2603.19292v1)

**作者:** Yi Yu `[一作]` (INRIA Paris), Chloé Clavel `[通讯]` (INRIA Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了利用人类任务导向对话语料库进行协作分析的研究现状，涵盖编码方案、任务设置、特征工程和建模方法。

**💡 创新点**

创新点在于首次整合多模态协作特征与分层编码，梳理不同粒度标注对模型性能的影响，并指出现有研究在数据稀缺与跨模态融合上的不足。

**🔧 技术方法**

采用统计与深度学习方法，包括SVM、LSTM、Transformer、GNN等，以及预训练嵌入（BERT、wav2vec、OpenFace）进行特征提取和融合。

**📊 数据集**

主要使用公开协作语料库，如AMI、Rovereto、SRI、Teams、PCC、RoomReader、Multicollab等，覆盖文本、音频、视频与生理传感器。

**📈 对比分析**

通过与传统SVM/逻辑回归对比，深度学习模型在二分类协作质量上可达约90%准确率，四分类群体编码的平均不加权准确率约为49%；跨语料库融合与多模态late‑fusion亦提高了鲁棒性。

**⚠️ 局限性**

限制主要包括标注稀缺、任务与标签的异质性导致模型泛化差、跨模态对齐困难、以及对少数下游任务如自适应人机协作的实际应用研究不足。

---

## 220. PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization

**arXiv ID:** 2603.19649 | [PDF](https://arxiv.org/pdf/2603.19649v1)

**作者:** Renhong Huang `[一作]` (Zhejiang University), Yang Yang `[通讯]` (Zhejiang University)

**通讯引用:** 111886 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 PolicySim——一种基于大语言模型的社交仿真沙箱，用于在平台上线前主动评估和优化干预策略（如推荐系统、曝光控制）

**💡 创新点**

创新点在于①首次将 SFT+直接偏好优化（DPO）训练用户代理，实现对平台数据的行为与意图双重对齐；②提出利用上下文多臂赌博机并结合消息传递的自适应干预算法，实现基于仿真反馈的策略迭代；③将仿真与干预优化紧密耦合，形成闭环评估框架

**🔧 技术方法**

技术包括：大语言模型（如 Qwen2.5-3B-Instruct）训练、SFT+DPO、短期/长期记忆模块、图卷积/标签传播用于关系扩散、上下文多臂赌博机（UCB/ε-greedy）以及自适应奖励设计

**📊 数据集**

使用真实社交媒体数据集：TwiBot-20（Twitter）和微博数据集，包含用户行为、帖子、关注关系等；此外对抗性测试中手工注入谣言内容

**📈 对比分析**

在微观层面对比 5 种 LLM 基础模型及不同训练拆解（-ϕ、-SFT、-DPO），通过 BERTScore、行为对齐准确率、自洽性、LLM-as-Judge 的 Engagement/Robustness/Suitability 等指标；在宏观层面对比干预目标的实现（跨视角交互、毒性、谣言传播）与 ε-greedy、UCB 基线；实验表明 PolicySim 在多指标上均优于基线，特别是提升跨立场交互并显著降低毒性，同时在谣言抑制上实现更高比例的减少。

**⚠️ 局限性**

局限性包括：①对大语言模型推理的高计算成本，限制了大规模实时仿真；②训练数据与目标平台的匹配度有限，可能导致模拟与真实环境的偏差；③仅评估两种平台和两类干预策略，缺乏对更广泛平台和多模态干预的验证；④奖励设计与真实社会影响之间的映射仍需进一步验证。

---

## 221. Dynamically Reprogrammable Runtime Monitors for Bounded-time MTL

**arXiv ID:** 2603.19851 | [PDF](https://arxiv.org/pdf/2603.19851v1)

**作者:** Chirantan Hebballi `[一作]` (Indian Institute of Technology Dharwad), Ramchandra Phawade `[通讯]` (Indian Institute of Technology Dharwad)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5076083645)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

设计了可在标准单元上实现、可在现场动态重编程的MTL运行时监测器。

**💡 创新点**

通过使用可编程处理单元和队列组合，既保持高频率监测，又实现了基于标准单元的低面积、低延迟实现，可在现场通过I/O重配置。

**🔧 技术方法**

采用了可编程抽象机（PE）、队列（Que）、可编程互连，使用Clash HDL生成Verilog，基于32nm标准单元库综合。

**📊 数据集**

使用了合成实验和仿真验证，支持最多16个原子命题、时间上限256步的MTL公式；并通过对不同N_PE、N_Q、Q_SZ组合的面积和频率实验评估。

**📈 对比分析**

通过与FPGA实现比较，得到1.25 GHz时钟、0.55 mm²面积，吞吐率1.25 G事件/秒，显著优于传统FPGA实现；对不同参数组合做面积、频率平衡分析。

**⚠️ 局限性**

限制在于单跨点交叉开关互连导致面积随PE数量快速增长，且实现需要专用的编译器与配置流程，缺乏通用的高速互连架构。

---

## 222. Global Convergence of Multiplicative Updates for the Matrix Mechanism: A Collaborative Proof with Gemini 3

**arXiv ID:** 2603.19465 | [PDF](https://arxiv.org/pdf/2603.19465v1)

**作者:** Keith Rush `[一作]` `[通讯]` (Google DeepMind), Keith Rush (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

证明了正交正定矩阵相关核范数目标函数的固定点迭代在正交正定条件下全局收敛，并给出了单调上升的证据

**💡 创新点**

利用核范数的变分表征与极大化-极小化框架，将问题对角化并完成全局收敛证明

**🔧 技术方法**

核范数变分表征、Lieb对角化、Von Neumann trace不等式、极大化-极小化(MM)框架

**📊 数据集**

无

**📈 对比分析**

无实验比较

**⚠️ 局限性**

仅在理论层面，缺乏对实际隐私机制或机器学习任务的实验验证

---

## 223. Teaching an Agent to Sketch One Part at a Time

**arXiv ID:** 2603.19500 | [PDF](https://arxiv.org/pdf/2603.19500v1)

**作者:** Xiaodan Du `[一作]` (TTI-Chicago), Greg Shakhnarovich `[通讯]` (TTI-Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种基于多模态语言模型的逐部绘制向量草图代理，并通过自监督微调+多轮过程奖励的 GRPO 强化学习实现对草图的逐部生成与局部编辑。

**💡 创新点**

创新点主要包括：①可扩展的自动化部件注释管线，生成 ControlSketch‑Part 数据集；②多轮过程奖励 GRPO 训练框架，利用中间状态奖励实现密集信用分配；③支持自由文本指导、逐部绘制与局部编辑的统一模型。

**🔧 技术方法**

技术手段包括：VLM（Qwen3‑VL‑30B‑A3B）+ LoRA 微调；SFT 训练学习输出格式；GRPO 强化学习（多轮过程奖励、DreamSim + 轨迹长度奖励）；自动部件注释管线（Gemini 3.0 Pro 进行分解、评审、路径分配等）。

**📊 数据集**

使用的数据集是 ControlSketch‑Part（基于 ControlSketch 35k 物体草图，使用 Gemini 3.0 Pro 自动注释得到的短标题、部件描述与路径分配），以及用 Gemini 2.5 Flash 微调的 SFT 训练集。

**📈 对比分析**

与 SketchAgent、Gemini 3.1 Pro、SDXL+SwiftSketch 等基线对比，本文模型在 Long‑CLIP 余弦相似度、用户偏好实验中均优于对手，且实现了更高的结构化质量和可编辑性。

**⚠️ 局限性**

局限性：依赖强大的 VLM 与大量标注数据，训练成本高；在极为复杂或细节丰富的物体上仍可能出现结构偏差；目前仅支持 SVG 形式的向量草图，跨平台应用仍需进一步研究。

---

## 224. Synergistic Perception and Generative Recomposition: A Multi-Agent Orchestration for Expert-Level Building Inspection

**arXiv ID:** 2603.20143 | [PDF](https://arxiv.org/pdf/2603.20143v1)

**作者:** Hui Zhong `[一作]` (Hong Kong University of Science and Technology), Xinhu Zheng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2335 | [OpenAlex ID](https://openalex.org/A5062424202)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FacadeFixer——一个统一的多智能体框架，融合检测、分割与生成式数据增强，完成建筑立面缺陷的全流程智能巡检，并构建了覆盖六类缺陷的多任务像素级标注数据集。

**💡 创新点**

创新点在于：①将缺陷感知转化为协同推理任务，利用大语言视觉模型（MLLM）进行多模型投票裁决，解决语义干扰；②引入语义重组机制，结合缺陷记忆库与生成式代理实现无监督高质量合成；③提供一套完整的多任务数据集，为缺陷检测与分割奠定基准。

**🔧 技术方法**

采用的技术包括：多模型检测/分割专家（YOLOv11/12、Faster R‑CNN、RT‑DETR、SegFormer、SAM‑3等）、LLM裁决（GPT‑4o、Gemini‑3‑Flash）、生成式重组（Seedream‑4.5 语义填充与混合）、缺陷记忆库、语义去噪与mask‑guided检索、以及基于CLIP的图像预筛选。

**📊 数据集**

使用的数据集为自研的UAV采集立面图像数据集，涵盖112栋住宅楼，共计数千张高分辨率图像，提供分类、检测框、像素分割与修复标注，覆盖六大缺陷类型（混凝土裂缝、剥落、锈渍、墙面剥落、植被、杂物）。

**📈 对比分析**

实验采用标准指标（Precision、Recall、mAP_50:95、F1、mIoU）与现有SOTA检测/分割模型及传统集成方法进行对比。FacadeFixer在检测任务上mAP_50:95为0.5678，高于最佳单模型0.5225和传统交并策略0.5245；在分割任务上mIoU达0.6698，明显优于最佳单模型0.5280；生成合成数据的检测精度也保持在0.5908。

**⚠️ 局限性**

局限性包括：生成的合成缺陷在植被和杂物类别中偶尔出现物理不合逻辑的浮动效果；裁决过程对极端噪声候选仍可能失误；整体性能仍受限于所用LLM和生成模型的训练数据与推理资源。

---

## 225. HortiMulti: A Multi-Sensor Dataset for Localisation and Mapping in Horticultural Polytunnels

**arXiv ID:** 2603.20150 | [PDF](https://arxiv.org/pdf/2603.20150v1)

**作者:** Shuoyuan Xu `[一作]` (Loughborough University), Cunjia Liu `[通讯]` (Loughborough University)

**通讯引用:** 3977 | [OpenAlex ID](https://openalex.org/A5021774274)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

创建并公开了HortiMulti多传感器、多季节的苺、覆盆子管道农业数据集，并通过Total Station与Poly-TagSLAM生成高精度地面真值。

**💡 创新点**

① 首次提供覆盖全季节、动态叶片、GNSS失效与高光照变化的管道环境数据集；② 采用Total Station + Poly-TagSLAM实现亚厘米级地面真值；③ 同步多传感器原始数据与完整基准评估。

**🔧 技术方法**

多传感器同步与标定、LiDAR、RGB相机、IMU、GNSS与车轮里程计；Poly-TagSLAM与总台测量相结合的地面真值生成；多种SOTA SLAM（视觉、LiDAR、视觉‑惯性、LiDAR‑惯性）与学习式LiDAR场景检索方法的评测。

**📊 数据集**

使用自身HortiMulti数据集；对比现有农田、林地与园艺数据集（如FieldSAFE、Raspberry等）以展示其独特性。

**📈 对比分析**

对比了视觉（DSO、ORB-SLAM3）、视觉‑惯性（OpenVINS、VINS-Fusion）、LiDAR（FLOAM、LIO-SAM）和LiDAR‑视觉‑惯性（未完成）SLAM方法；评估指标为ATE与RPE。结果显示视觉方法普遍失效，LiDAR‑惯性最稳，但在长行驶中仍有显著漂移；地方识别采用PointNetVLAD、SPVSoAP3D、ScanContext等，召回率在10 m阈值下低于常规城市/林地数据集。

**⚠️ 局限性**

仅针对苺、覆盆子管道；缺乏其他作物与多样结构；基准算法受限于传感器配置与场景重复性；长距离回环与动态叶片仍难以精确定位；社区贡献与工具链仍在完善中。

---

## 226. The $\mathbf{Y}$-Combinator for LLMs: Solving Long-Context Rot with $λ$-Calculus

**arXiv ID:** 2603.20105 | [PDF](https://arxiv.org/pdf/2603.20105v1)

**作者:** Amartya Roy `[一作]` (Indian Institute of Technology), Haitham Bou-Ammar `[通讯]` (Huawei)

**通讯引用:** 4677 | [OpenAlex ID](https://openalex.org/A5100384727)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在长文本推理任务中，用λ‑算子构建了一个结构化、可验证的递归控制框架，将LLM的控制逻辑从自由生成的代码转为预先验证的、类型化的组合子程序，确保递归终止、成本可预测、准确度可控；

**💡 创新点**

创新点在于：①将递归控制迁移至基于λ‑计算的固定点组合子体系，彻底消除传统REPL循环的不可预期与非终止风险；②提供形式化的终止与成本界限证明，并给出最优划分策略；③通过类型化组合子将神经推理限定为叶子子问题，显著提升弱模型的表现。

**🔧 技术方法**

核心技术包括：λ‑算子与Y‑组合子实现递归；类型化组合子库（Partition、Prune、Map、Fold、NeuralOracle 等）用于分块、筛选、映射与聚合；预规划器根据输入长度、上下文窗口与期望准确度自动选择分块大小、阈值和组合子；基于成本模型的闭式复杂度分析。

**📊 数据集**

实验使用四类长文本推理基准：S‑NIAH（O(1)），OOLONG（O(n)），OOL‑Pairs（O(n²)），CodeQA（可变），每类覆盖 8K‑128K 级别的文本长度。

**📈 对比分析**

与两类基线对比：直接单次LLM推理（P1）和传统递归语言模型（RLM，P2）。λ‑RLM（P3）在 108 组模型‑任务配置中赢得 81% 的准确率，平均提升 19.7 点；同时平均延迟比 RLM 提速约 4×，并在所有模型与任务上实现 3.3–4.1 倍的延迟缩减。

**⚠️ 局限性**

局限性包括：①组合子库表达能力有限，难以覆盖需要高度创造性代码或自定义搜索策略的任务（如 CodeQA）；②对叶子子问题的依赖仍要求底层模型具备较高推理质量；③划分策略（k*, τ*）对性能敏感，需通过规划器精确估算；④在极大规模或特殊结构任务中，固定组合子可能导致过度抽象而丢失细粒度控制。

---

## 227. Structured Prompting for Arabic Essay Proficiency: A Trait-Centric Evaluation Approach

**arXiv ID:** 2603.19668 | [PDF](https://arxiv.org/pdf/2603.19668v1)

**作者:** Salim Al Mandhari `[一作]` (Lancaster University), Paul Rayson `[通讯]` (Lancaster University)

**通讯引用:** 9849 | [OpenAlex ID](https://openalex.org/A5058785189)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

采用大语言模型（LLM）在三层提示框架下，对阿拉伯语学生作文进行特征级（trait‑level）评分，完全不需要模型微调。

**💡 创新点**

创新点包括：①提出标准‑混合‑rubric‑guided 三层提示体系；②在混合层模拟专门评审，提升评判细粒度；③首次在阿拉伯语 AES 任务中引入少量示例（few‑shot）提示，实现更高的评分一致性。

**🔧 技术方法**

技术手段主要是零/少样本提示工程、Rubric‑aligned 提示、混合评审（多专家模拟）、JSON 结构化输出；使用了 8 种不同规模/基础的 LLM（如 ChatGPT‑4、Qwen3‑VL‑8B、Fanar‑1‑9B、ALLAM‑7B 等）。

**📊 数据集**

数据集为 QAES（Qatar Arabic Essay Set），包含 195 篇论证性作文，标注 7 个语言特征（组织、词汇、风格、发展、机械、结构、相关性）以及总分。

**📈 对比分析**

通过 Quadratic Weighted Kappa（QWK）与人类标注对齐，并与最强基线 LR（QWK ≈ 0.26）做比较。三层提示体系下最高 CI 上限达 0.41，虽然低于人类一致性阈值 0.72，但显著高于基线；提示越具体、层次越细，模型表现越好，尤其在 Fanar‑1‑9B 与 ALLAM‑7B 上效果最佳。

**⚠️ 局限性**

局限性：①数据集规模小，阿拉伯语资源匮乏；②小型 LLM 对复杂提示的理解力有限；③缺乏 fine‑tuning，导致部分模型在高阶提示下表现为 0；④GPT‑4 等专有模型成本高，难以大规模部署；⑤混合评审方式对 rater 一致性和评分聚合仍需进一步研究。

---

## 228. Generalized Task-Driven Design of Soft Robots via Reduced-Order FEM-based Surrogate Modeling

**arXiv ID:** 2603.19794 | [PDF](https://arxiv.org/pdf/2603.19794v1)

**作者:** Yao Yao `[一作]` (Oxford Robotics Institute University of Oxford), Perla Maiolino `[通讯]` (Oxford Robotics Institute University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一套统一的基于高精度有限元(FEM)数据的低阶代理建模流程，能够在软机器人不同驱动类型和任务场景下快速生成可嵌入prbm的机械响应模型，实现从模块级仿真到任务级优化的闭环。

**💡 创新点**

核心创新在于（1）将高精度FEM结果压缩为多项式或轻量神经网络代理，保持FEM级别的物理精度；（2）学习meta‑model，将设计参数映射到代理特征，实现设计空间内的即时代理生成；（3）将代理嵌入PyBullet prbm框架，支持强化学习与进化优化的任务级仿真与优化，从而实现可扩展的任务驱动软机器人设计。

**🔧 技术方法**

采用的技术包括：高精度有限元(FEM)仿真、自由与受限条件下的耦合数据采集、低阶多项式拟合与小型前馈神经网络、元模型(Meta‑model)训练、PyBullet物理仿真、强化学习(PPO)与进化优化(CMA‑ES)。

**📊 数据集**

使用的主要数据集为：FEM仿真生成的模块级力-变形-压力映射数据（覆盖自由与受限加载），以及实验收集的软抓取、螺旋形变形与力学响应数据，用于代理验证与仿真对比。

**📈 对比分析**

通过模拟‑实测对比评估代理精度（误差<10%），在软抓取任务中实现10/10的成功率，在形状匹配任务中将误差从先前的25.7 mm降至4.5 mm，表明代理既保持高准确性，又大幅提升计算效率（相较原始FEM可提升数百倍）。

**⚠️ 局限性**

局限性包括：仅能在已采样的参数化设计空间内插值，缺乏跨形态/驱动类型的泛化；基于静态FEM，未考虑动态与瞬态效应；对超出训练范围的外部条件鲁棒性有限；硬件验证仅覆盖部分案例。

---

## 229. Scalable Cross-Facility Federated Learning for Scientific Foundation Models on Multiple Supercomputers

**arXiv ID:** 2603.19544 | [PDF](https://arxiv.org/pdf/2603.19544v1)

**作者:** Yijiang Li `[一作]` (Argonne National Laboratory), Kibaek Kim `[通讯]` (Argonne National Laboratory)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5032602013)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个跨 DOE 超算的联邦学习框架，实现了在不同 HPC 设施之间的协同训练。

**💡 创新点**

首次系统量化了异构 HPC 环境下的计算吞吐、网络通信和排队动态，并将算法如 FedCompass 与排队、网络特性结合，提供了多设施实验数据。

**🔧 技术方法**

基于 APPFL 框架，结合 Globus Compute/Transfer、ProxyStore、DeepSpeed+ZeRO 以及 MPI Engine 实现跨站点的任务调度与模型交换。

**📊 数据集**

使用 SMolInstruct 化学指令微调数据（约 330 万样本）与 Llama2‑7B 语言模型进行实验。

**📈 对比分析**

对 FedAvg、FedAsync、FedBuff、FedCompass 四种算法进行对比；在 64 节点 co‑scheduled 实验中全球模型误差降至 0.37；在 2 节点无预留实验中 FedCompass 最优，最终测试误差 0.4345，明显优于其它算法。

**⚠️ 局限性**

尚未考虑真实排队延迟与网络时变；缺乏层次聚合与漂移补偿机制；需要进一步设计队列感知与动态同步的算法。

---

## 230. Dual-Domain Representation Alignment: Bridging 2D and 3D Vision via Geometry-Aware Architecture Search

**arXiv ID:** 2603.19563 | [PDF](https://arxiv.org/pdf/2603.19563v1)

**作者:** Haoyu Zhang `[一作]` (Hangzhou Normal University), Ran Cheng `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 18335 | [OpenAlex ID](https://openalex.org/A5004036087)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `8d10c613-917e-4880-9716-17789f50e119` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出EvoNAS框架，结合VSS‑ViT混合搜索空间、跨架构双域知识蒸馏（CA‑DDKD）和分布式多模型并行评估（DMMPE），实现多目标（精度、延迟、计算量）演化NAS。

**💡 创新点**

创新点在于：1）双域（空间+频域）蒸馏提升共享权重的表示一致性；2）PST递进式超网络训练消除表示崩塌；3）DMMPE通过硬件隔离和异步调度大幅提升评估吞吐和测量精度。

**🔧 技术方法**

使用的技术包括：进化算法NSGA‑II、Hybrid VSS‑ViT结构、CA‑DDKD（DCT约束蒸馏）、DMMPE硬件隔离多GPU调度、以及自动混合精度训练。

**📊 数据集**

数据集覆盖：ImageNet‑1K用于预训练，COCO（目标检测）、ADE20K（语义分割）、KITTI与NYU‑v2（单目深度）、RealEstate10K（3D视角合成）进行多任务评估。

**📈 对比分析**

与CNN、ViT、Mamba、NAS等基线比较，EvoNet在保持模型参数少（约20–40M）且MACs低（≈20–30G）的同时，在各任务上实现了更高的AP、mIoU、AbsRel等指标；在RealEstate10K的3D渲染中，参数量比基线低88%，仍保留高PSNR/SSIM。

**⚠️ 局限性**

局限性包括：1）搜索过程仍需数百小时GPU资源；2）依赖高质量教师模型，若教师欠佳会影响蒸馏效果；3）在非视觉或非几何相关任务的迁移性尚未验证；4）DMMPE实现复杂，需特定GPU集群支持。

---

## 231. A Subgoal-driven Framework for Improving Long-Horizon LLM Agents

**arXiv ID:** 2603.19685 | [PDF](https://arxiv.org/pdf/2603.19685v1)

**作者:** Taiyi Wang `[一作]` (Google DeepMind), Edward Grefenstette `[通讯]` (Google DeepMind)

**通讯引用:** 12769 | [OpenAlex ID](https://openalex.org/A5023508792)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于里程碑的强化学习与推理时子目标规划的统一框架（MiRA），用于提升 LLM 生成的 Web 代理在长时程任务中的规划与执行能力。

**💡 创新点**

创新点包括：① 通过教师模型生成可验证的子目标并在推理阶段实时回溯校验；② 用潜在子目标生成的进度信号构建潜在奖励，实现稠密奖励塑形；③ 采用 MSE 目标对数概率比的回归与双重稳健优势估计，保证离线 RL 的高效与稳定；④ 迭代自适应训练循环，以失败分析为驱动不断扩展任务分布。

**🔧 技术方法**

主要技术手段包括 LLM 生成子目标与自动评估（AutoRater），潜在子目标评估网络（Potential Critic），目标价值网络（Value Critic），MSE‑based 目标策略更新，双重稳健优势估计，以及基于子目标的奖励塑形。

**📊 数据集**

使用的数据集为 WebArena‑Lite（165 任务，涵盖 Shopping Admin、Map、Shopping、Reddit、Gitlab 等五个域），并通过自动失败分析生成训练与评估轨迹。

**📈 对比分析**

与 WebRL、DigiRL、SFT 等基线相比，Gemma‑3‑12B + MiRA 在平均成功率上达 43.0%，显著高于 WebRL（35.1%）和 DigiRL（33.3%），在所有子域均表现提升；在推理时子目标规划下，Gemini‑2.5‑Pro 通过 SGO 也提升约 10pp。

**⚠️ 局限性**

主要局限包括：① 需要可靠的子目标生成与评估，若子目标不稳健会导致塑形信号消失；② 对冷启动探索仍存在挑战，首次子目标难以到达时模型退回稀疏奖励；③ 训练中对 LLM 评估器的依赖可能引入误判，导致 Wrong Termination 错误率上升；④ 过度依赖潜在奖励可能在某些任务上产生过拟合或偏向。

---

## 232. VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

**arXiv ID:** 2603.20185 | [PDF](https://arxiv.org/pdf/2603.20185v1)

**作者:** Jingyang Lin `[一作]` (University of Rochester), Emad Barsoum `[通讯]` (AMD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VideoSeek，一种利用视频逻辑流进行主动证据搜索的长时序视频智能体。

**💡 创新点**

通过主动寻找答案关键信息，避免密集帧解析，显著降低视觉预算。

**🔧 技术方法**

采用思考–行动–观察循环的 ReAct 框架，配备三种多粒度工具（Overview、Skim、Focus）以及 LLM 思考模型。

**📊 数据集**

在 LVBench、Video‑MME、LongVideoBench 和 Video‑Holmes 四个长视频与复杂推理基准上进行评估。

**📈 对比分析**

与基准 LLM GPT‑5 及其他视频代理相比，VideoSeek 在保持或提升准确率的同时，帧数减少 76–96%，在 LVBench（字幕）最高 76.7% 仅用 27.2 帧。

**⚠️ 局限性**

对极端突发事件或异常检测等需要全局扫描的场景不够适用。

---

## 233. MedQ-Engine: A Closed-Loop Data Engine for Evolving MLLMs in Medical Image Quality Assessment

**arXiv ID:** 2603.19863 | [PDF](https://arxiv.org/pdf/2603.19863v1)

**作者:** Jiyao Liu `[一作]` (Fudan University), Ningsheng Xu `[通讯]` (Fudan University)

**通讯引用:** 10179 | [OpenAlex ID](https://openalex.org/A5101020368)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了MedQ-Engine闭环数据引擎，通过评估–探索–演化三步循环迭代提升多模态大语言模型在医学图像质量评估（Med‑IQA）的能力。

**💡 创新点**

创新点包括：①基于聚类的失效原型发现与检索锚点；②失效驱动的自适应采样和熵引导的分层人机标注；③全参数指令微调与闭环迭代；④通过多模态检索和质量保障实现高样本效率。

**🔧 技术方法**

使用了InternVL3-8B、Qwen2.5‑VL‑7B等多模态LLM，BiomedCLIP编码图像，聚类+自适应采样，GPT‑4o作为注释oracle，熵测度与TF‑IDF去重，最终进行全参数SFT。

**📊 数据集**

构建约100万幅无标签医学图像池（MRI、CT、内镜、视网膜、组织病理），以及2k样本的开发集和MedQ‑Bench测试基准。

**📈 对比分析**

与GPT‑4o、开源与闭源多模态LLM对比，优化后的8B模型在感知任务上提升13个百分点，整体准确率达78.2%，仅比人类专家低4.3%，且仅用10k样本即比随机采样高效4倍。

**⚠️ 局限性**

局限性在于仍依赖GPT‑4o等大型oracle，标注成本虽降低但非零；对极少见模态或严重退化的泛化尚有限；闭环迭代次数与计算资源受限。

---

## 234. Timestep-Aware Block Masking for Efficient Diffusion Model Inference

**arXiv ID:** 2603.19939 | [PDF](https://arxiv.org/pdf/2603.19939v1)

**作者:** Haodong He `[一作]` (Wuhan University), Gui-Song Xia `[通讯]` (Wuhan University)

**通讯引用:** 21923 | [OpenAlex ID](https://openalex.org/A5073032922)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在扩散模型推理阶段，通过学习每个时间步的二进制掩码来动态跳过不必要的计算块，从而显著加速去噪过程；

**💡 创新点**

创新点在于：①采用按时间步优化的掩码学习策略，避免了全链反向传播导致的高显存消耗；②引入时间步感知的损失权重与双峰正则，使得重要时间步的生成质量得到保障；③提出知识驱动的掩码校正规则，进一步提升加速率；

**🔧 技术方法**

技术包括：基于预训练扩散模型的冻结网络、连续化随机采样的掩码参数化、L2特征损失+稀疏+双峰正则的联合优化、时间步权重调度、掩码后处理与缓存机制；

**📊 数据集**

使用的公开数据集有：CIFAR-10、LSUN‑Bedroom、LSUN‑Churches、ImageNet（256×256 与 512×512）、MS‑COCO（1024×1024）等，覆盖无条件、类条件及文本条件生成任务；

**📈 对比分析**

与 DeepCache、Diff‑Pruning、L2C、DitFastAttn、FORA 等现有加速方法对比，本文方法在保持甚至提升生成质量（FID、IS、sFID、CLIP Score 等指标）同时，实现 1.48×‑2.75× 的采样速度提升，且训练时间显著降低（几小时内完成掩码学习）；

**⚠️ 局限性**

局限性包括：①需要针对每个模型单独训练掩码，缺乏通用“一键”方案；②对非常大规模模型的内存和训练开销仍有一定需求；③目前主要验证于图像生成任务，对视频、3D 等其他扩散任务的泛化尚待进一步研究。

---

## 235. Reviewing the Reviewer: Graph-Enhanced LLMs for E-commerce Appeal Adjudication

**arXiv ID:** 2603.19267 | [PDF](https://arxiv.org/pdf/2603.19267v1)

**作者:** Yuchen Du `[一作]` (Purdue University), Zixi Huang `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电商争议解决中提出一种基于行动层的冲突感知图推理框架（EAFD），通过显式建模证据→行动→因素→决策的四层结构，利用Maker–Checker纠错信号自动推理出检查员决策，并在信息不足时返回“请求更多信息”请求。

**💡 创新点**

创新点：① 将纠错事件从单纯的结果差异转化为可操作的行动层学习；② 设计EAFD图模式，将行动作为推理约束，消除LLM的幻觉；③ 引入冲突边与RMI机制，实现结构化的“知道自己不知道”能力；④ 通过检索+推理的两阶段流程，将历史案例转化为可复用的程序化路径。

**🔧 技术方法**

技术：大型语言模型（Claude Sonnet 4.0）配合检索增强（Amazon Titan Embed Text）、图数据库构建与验证、图推理引擎（FAE演绎）、视觉‑语言预处理（VLM提取结构化文本）、LLM生成的图结构验证层。

**📊 数据集**

数据集：来自某大型电商平台的260个售商申诉案例，包含多模态证据（发票、聊天记录、图片），对Maker与Checker的完整审理记录，采用80/20时间序列划分进行离线评估，并在生产环境上线验证。

**📈 对比分析**

与基线比较：对比四类基线（仅LLM、LLM+RMI、CBR多数投票、LLM+CBR）显示，EAFD在离线准确率达到95.8%、Macro‑F1 0.867，线上累计对齐率达97.1%，尤其在审批决策上100%精准，RMI召回率高，显著优于70.8%的纯LLM基准。

**⚠️ 局限性**

局限性：依赖高质量的历史审理文档和专家定义的因素；因素需手工指导，难以自动化；对极端新情形缺乏泛化；需要持续维护因政策变更而产生的因素与行动映射。

---

## 236. UniPR: Unified Object-level Real-to-Sim Perception and Reconstruction from a Single Stereo Pair

**arXiv ID:** 2603.19616 | [PDF](https://arxiv.org/pdf/2603.19616v1)

**作者:** Chuanrui Zhang `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3186 | [OpenAlex ID](https://openalex.org/A5100389366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 UniPR，一个端到端、一次性前向推理的框架，利用单张立体相机图像直接检测并重建场景中所有物体的 3D 形状与位姿；

**💡 创新点**

核心创新包括姿态感知形状表示（PASR）消除对类别规范空间的依赖、三平面视图（TPV）结合立体几何约束实现一次推理、以及大词汇量支持和真实尺度比例保持；

**🔧 技术方法**

采用 Transformer 编码器‑解码器、TPV 与立体交叉注意力、球面体素空间的 VAE、CLIP 语义分类等技术；

**📊 数据集**

主要使用自行构建的 LVS6D 大词汇量立体数据集（192 类、6300+ 物体），并在 TOD、SS3D 等公开立体数据集上进行评测；

**📈 对比分析**

与 Trellis、HunYuan3D、Coders 等基线在 Chamfer Distance、F‑Score、Shape Proportion Error、AP、APE、ACD 等指标下进行对比，UniPR 在形状比例、速度（最高 100× 加速）以及多物体重建精度上均显著优于现有方法；

**⚠️ 局限性**

在单目设置下表现较差，因缺乏深度信息；对极端遮挡和非常小物体仍有挑战，未来计划加入大型深度模块以提升单目性能。

---

## 237. HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning

**arXiv ID:** 2603.19639 | [PDF](https://arxiv.org/pdf/2603.19639v1)

**作者:** Beibei Xu `[一作]` (East China Normal University), Mingsong Chen `[通讯]` (East China Normal University)

**通讯引用:** 7906 | [OpenAlex ID](https://openalex.org/A5102865504)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HyEvo框架，自动生成包含LLM推理节点与代码执行节点的混合agentic工作流。

**💡 创新点**

创新点在于将LLM与确定性代码节点协同工作，并通过LLM驱动的多岛进化与反思‑生成机制高效搜索最优结构。

**🔧 技术方法**

使用LLM推理、代码生成、进化算法（MAP‑Elites多岛）、反思‑生成机制、沙盒评估等技术。

**📊 数据集**

使用数学推理数据集GSM8K、MATH、MultiArith以及代码生成数据集HumanEval、MBPP。

**📈 对比分析**

与手工与自动化工作流基线（如AFlow、MaAS、AutoAgents等）比较，HyEvo在所有五个基准上均表现最佳，平均提升约1.2%，并在成本与延迟上分别降至19×与16×。

**⚠️ 局限性**

局限在于仍需依赖LLM API、演化过程算力开销较大，以及对更大规模任务的可扩展性与稳定性尚未充分验证。

---

## 238. MOSS-TTSD: Text to Spoken Dialogue Generation

**arXiv ID:** 2603.19739 | [PDF](https://arxiv.org/pdf/2603.19739v1)

**作者:** Yuqian Zhang `[一作]`, Xipeng Qiu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了MOSS‑TTSD v1.0模型，实现长时段多说话人对话语音合成，支持零样本语音克隆、自然轮转、60分钟单通道无缝生成；

**💡 创新点**

①采用全离散语音生成，仅建模前16层RVQ，实现低比特率长上下文；②多头延迟模式预测，提升长序列生成稳定性；③提出TTSD‑eval基于强制对齐的客观评估框架，消除对说话人识别器的依赖；

**🔧 技术方法**

自动回归多头延迟生成、Qwen3‑8B‑base LLM、MOSS‑Audio‑Tokenizer、MOSS‑Transcribe‑Diarize ASR、语音克隆、语音嵌入相似度、强制对齐（MMS‑FA）、speaker embedding模型wespeaker‑SimAMResNet100、数据增强与课程学习；

**📊 数据集**

公开多说话人对话语料，涵盖英中西葡德法日韩俄等多语种；原始音频经过Whisper/Whisper‑large‑v3、MOSS‑Transcribe‑Diarize处理，DNSMOS≥2.8过滤，包含多说话人参考音频及合成语料；

**📈 对比分析**

与开源模型（Higgs Audio V2、FireRedTTS‑2、VibeVoice 1.5B/7B）以及闭源模型（Eleven V3、Gemini 2.5 Pro、Doubao Podcast）在TTSD‑eval中对齐ACC、SIM、WER、cpWER、cpSIM以及主观Elo、胜率进行对比，MOSS‑TTSD在所有指标上均优于对比模型，主观评价亦表现突出；

**⚠️ 局限性**

模型规模较大，推理资源需求高；仍依赖ASR质量，对极端噪声或非常大说话人数的鲁棒性未知；非主流语言支持有限；对长时段多说话人对话的极端场景（如几十人）尚未充分验证。

---

## 239. Kolmogorov-Arnold causal generative models

**arXiv ID:** 2603.20184 | [PDF](https://arxiv.org/pdf/2603.20184v1)

**作者:** Alejandro Almodóvar `[一作]`, Juan Parras `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

无法确定

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

无法确定

---

## 240. From Feature-Based Models to Generative AI: Validity Evidence for Constructed Response Scoring

**arXiv ID:** 2603.19280 | [PDF](https://arxiv.org/pdf/2603.19280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 241. Beyond Accuracy: Towards a Robust Evaluation Methodology for AI Systems for Language Education

**arXiv ID:** 2603.20088 | [PDF](https://arxiv.org/pdf/2603.20088v1)

**作者:** James Edgell `[一作]` (Oxford University Press), Elizabeth Wonnacott `[通讯]` (University of Oxford)

**通讯引用:** 1881 | [OpenAlex ID](https://openalex.org/A5035781319)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现L2-Bench，一个基于语言学习体验设计的AI评估基准，用以系统评估语言教育AI系统的教学设计能力。

**💡 创新点**

①将教学设计能力拆解为12项主能力和30项子能力的层次化 competency taxonomy；②构建含有1000+真实任务-回答对的开放式评估数据集；③采用二元 rubric 评分与LLM-as-Judge 自动化评估，并加入不确定性量化。

**🔧 技术方法**

利用Claude系列大模型（生成任务与参考答案、自动评分、生成 AI 回答），人工‑AI 混合编辑流程，Krippendorff α 与 Cronbach α 等统计方法评估可靠性，混合效应模型和二项检验等验证方法。

**📊 数据集**

L2-Bench数据集：约1000+任务-回答对，覆盖12项能力、30项子能力，涉及不同地区、学习者画像与学习目标；Pilot 样本325对，后续计划完整1,300对。

**📈 对比分析**

与传统单任务精度评估不同，L2-Bench 通过 rubric‑based 评分与多样本（n=3）得分平均及标准误对模型做统计比较；Pilot 结果显示任务真实性高（M≈4.24/5），但标准化评分难度大，IAA 低。

**⚠️ 局限性**

局限：①任务单轮设计不充分反映真实课堂交互；②评审者多为研究生，专业校准不足导致低 IAA；③依赖欧洲教学框架，文化普适性受限；④自动评分依赖参考答案，可能忽视多元有效答案；⑤数据集规模仍有限，需进一步扩充与多模态支持。

---

## 242. Making Video Models Adhere to User Intent with Minor Adjustments

**arXiv ID:** 2603.19672 | [PDF](https://arxiv.org/pdf/2603.19672v1)

**作者:** Daniel Ajisafe `[一作]` (University of British Columbia), Kwang Moo Yi `[通讯]` (University of British Columbia)

**通讯引用:** 5717 | [OpenAlex ID](https://openalex.org/A5049893251)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对用户给出的bounding box做微调，使其与视频扩散模型内部的注意力图更好对齐，从而提升生成质量和控制精度。

**💡 创新点**

提出一种可微分的bounding box编辑方式和基于注意力最大化的优化目标，利用轻微调整保持原输入的同时显著提高控制效果。

**🔧 技术方法**

使用视频扩散模型的交叉注意力层、可微分的高斯编辑、平滑边缘步骤、注意力最大化损失及正则化约束。

**📊 数据集**

Animal Kingdom 数据集，用于获取真实动物视频中的bounding box轨迹。

**📈 对比分析**

与Peekaboo、T2V‑Turbo、Trailblazer等基线比较，采用 PickScore、HPS‑v2 与 mIOU 等指标；实验与用户研究均显示本方法在生成质量与控制一致性上有显著提升。

**⚠️ 局限性**

受限于底层模型质量，优化过程需部分反向传播导致生成速度变慢；目前仅针对bounding box，未验证其他控制方式。

---

## 243. How Concise are Chains of co-Büchi Automata?

**arXiv ID:** 2603.19806 | [PDF](https://arxiv.org/pdf/2603.19806v1)

**作者:** Rüdiger Ehlers `[一作]` (Clausthal University of Technology), Rüdiger Ehlers `[通讯]` (Clausthal University of Technology)

**通讯引用:** 1245 | [OpenAlex ID](https://openalex.org/A5050112206)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了链式共巴可视化自动机（COCOA）在表示ω-正则语言时的紧凑性，并与确定性奇偶自动机（DPW）进行比较，进一步分析了COCOA在执行布尔运算（并、交、补）时的大小增长情况。

**💡 创新点**

创新点包括：①证明即使每个链元素仅为两状态的确定性共巴自动机，COCOA仍可指数级比DPW更紧凑；②展示COCOA的并、交运算会导致指数级的尺寸爆炸；③证明COCOA的补运算同样不可避免地导致指数级扩张。

**🔧 技术方法**

主要技术手段是构造特定语言族、利用COCOA的残余语言和颜色概念、对比DPW与COCOA的状态数、证明最小化与残余语言的关系，以及基于强连通分量和递归分析的严格理论证明。

**📊 数据集**

文章未使用实验数据集，所有结果均基于理论构造和形式证明；涉及的语言族均为手工定义的符号序列。

**📈 对比分析**

通过构造可扩张的语言族，作者证明了COCOA与DPW在大小上的指数差距，并在布尔运算与补运算中展示了指数级的尺寸膨胀，表明在理论上COCOA在这些操作中性能受限。

**⚠️ 局限性**

主要局限在于COCOA的残余语言数目可能指数级增长，这使得布尔运算和补运算无法保持多项式的紧凑性；文章没有提供实验验证或实际实现的性能评估，且尚未给出可行的高效算法来缓解这些指数扩张。

---

## 244. OrbitNVS: Harnessing Video Diffusion Priors for Novel View Synthesis

**arXiv ID:** 2603.19613 | [PDF](https://arxiv.org/pdf/2603.19613v1)

**作者:** Jinglin Liang `[一作]` (South China University of Technology), Yichen Gong `[通讯]` (Agentic Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 OrbitNVS，重新将新视图合成（NVS）视为轨道视频生成任务，利用预训练的视频生成模型，并通过相机适配器、法线图生成分支和像素空间监督实现高质量的视角合成。

**💡 创新点**

创新点包括：①将 NVS 转化为轨道视频生成，充分利用视频生成模型的视觉先验；②引入相机适配器实现精准的相机姿态控制；③使用法线图生成分支并通过注意力机制指导 RGB 合成，显著提升几何一致性；④在 latent 空间之外加入像素空间监督，提升纹理细节与清晰度。

**🔧 技术方法**

核心技术包括：预训练的 Wan2.1-I2V-14B 隐式扩散模型（Latent Diffusion + Diffusion Transformer）、相机适配器（Plücker 坐标 + AdaLN）、法线图生成分支、像素空间后训练、流匹配（rectified flow）等。

**📊 数据集**

训练数据来源于 Step1X-3D（从 Objaverse-V1 过滤得到的 320k 3D 资产），测试数据使用 OmniObject3D（6k 3D 资产）和 GSO（1k 3D 资产）两大基准集。

**📈 对比分析**

与 SV3D、SEVA、EscherNet 进行对比实验。OrbitNVS 在单视图、不同轨道振幅以及多参考视角设置下均优于基线，PSNR 最高提升约 +2.9 dB（GSO）和 +2.4 dB（OmniObject3D），在水平轨道、单视图等场景表现尤为突出；ablation 实验进一步验证法线图分支和像素空间监督对性能的提升。

**⚠️ 局限性**

局限性包括：对极端轨道振幅或高频纹理的表现仍略有下降；训练过程耗时较长（两阶段共约 86 小时，需 8 台 A800 GPU）；模型目前针对静态对象，未针对动态场景或长时序视频进行验证；以及对大尺寸或高分辨率视频的扩展性仍需进一步研究。

---

## 245. The IJCNN 2025 Review Process

**arXiv ID:** 2603.19244 | [PDF](https://arxiv.org/pdf/2603.19244v1)

**作者:** Michele Scarpiniti `[一作]` (Sapienza University of Rome), Danilo Comminiello `[通讯]` (Sapienza University of Rome)

**通讯引用:** 2795 | [OpenAlex ID](https://openalex.org/A5019647783)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文描述了IJCNN 2025会议的完整审稿流程，包括作者提交、评审员分配、评审评分、元评审、论文排名以及接受决策，并提出了去偏差的评审评分校准方法。

**💡 创新点**

采用分层审稿架构、使用多维加权评审分数、引入基于贝叶斯方法的评审者偏差校准，并使用谐波平均融合评审和元评审分数，形成最终的校准后得分指数。

**🔧 技术方法**

使用CMT会议管理系统、TPMS与主题匹配进行评审分配、加权平均与谐波平均、量化去噪与贝叶斯低阶线性回归（Cortes&Lawrence）进行评分去偏差，并通过Python脚本实现自动化处理。

**📊 数据集**

数据集为2025年IJCNN会议的 5,526 篇论文提交、7,877 名评审者共 18,996 条评审报告及对应的元评审数据。

**📈 对比分析**

通过对比校准前后得分指数对论文排名进行对比，确保不超过 40% 的接受率；最终接受率为 38.94%，与目标一致；比较方法主要是统计排名一致性和接受决策的一致性，而非传统机器学习指标。

**⚠️ 局限性**

仍缺乏反驳阶段、评价尺度量化误差导致信息损失、校准模型计算复杂且仅在会议规模内可行、依赖人工制定权重、对极端偏差评审者的处理仍不完善。

---

## 246. AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?

**arXiv ID:** 2603.19574 | [PDF](https://arxiv.org/pdf/2603.19574v1)

**作者:** Soorya Ram Shimgekar `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2758 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建基于Reddit历史帖子的模拟用户，模拟多轮对话，研究了在持续交互中AI是否会放大用户的妄想相关语言。

**💡 创新点**

首次量化了妄想语言在多轮AI交互中的时间演化，并提出了利用实时妄想分数进行状态感知的干预方法。

**🔧 技术方法**

采用监督式妄想分数（logistic回归+MiniLM嵌入）、BERTopic主题建模、Stratified Propensity Score Matching，以及三大LLM（GPT‑5、LLaMA‑8B、Qwen‑8B）来生成并分析对话。

**📊 数据集**

数据来源为Reddit，包括自我报告妄想经验的子版块（共1,598用户）和非心理健康子版块（共27,734用户）。

**📈 对比分析**

通过匹配后对比妄想组与对照组在不同模型下的分数曲线，发现妄想组的分数斜率显著正向（平均≈+0.02），而对照组为负；在状态感知干预下，斜率显著下降（≈‑0.02），效果在三模型中均显著。

**⚠️ 局限性**

主要局限在于对话仅为模拟，未涉及真实人机交互；妄想标签受自选社区与语言噪声影响；未直接评估用户心理福祉或实际临床效度。

---

## 247. Beyond detection: cooperative multi-agent reasoning for rapid onboard EO crisis response

**arXiv ID:** 2603.19858 | [PDF](https://arxiv.org/pdf/2603.19858v1)

**作者:** Alejandro D. Mousist `[一作]`, Julian Cobos Aparicio `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种分层、事件驱动的多智能体架构，用于航天器端多模态地球观测数据的快速灾害检测与决策；

**💡 创新点**

创新点在于将早期假设生成与任务专门化相结合，利用视觉‑语言模型进行快速筛选，再由专用的光学/雷达专家智能体进行深度分析，最终通过大型语言模型进行证据融合，实现计算资源的动态分配与可解释决策；

**🔧 技术方法**

采用视觉‑语言模型（Qwen2‑VL）、大型语言模型（Qwen‑2.5）、深度学习语义分割（DeepLabV3++ResNet‑50）以及一系列光学/雷达特征提取工具（NHI、BAI、VV/VH比值、SWIR/NIR指数等）；

**📊 数据集**

使用Sentinel‑2 MSI（光学）与Sentinel‑1 GRD（雷达）成像数据，以及Sen2Fire、SenForFloods、CEMS等灾害标注数据集，共计27个样本；

**📈 对比分析**

与传统总执行两种专门智能体的基线方案对比，结果显示在无灾害场景下平均可加速约73%（标准差±14%），在灾害场景下加速约13%（标准差±31%）；同时生成的语义解释更聚焦、上下文更完整；

**⚠️ 局限性**

局限性包括：仅使用小规模数据集；早期筛选阶段采用未针对卫星影像微调的通用视觉‑语言模型；缺乏专用AI加速硬件；专门智能体的工具调用被硬编码，缺乏自适应决策能力；以及实验未覆盖完整的星座级联互联与资源调度场景。

---

## 248. Template-based Object Detection Using a Foundation Model

**arXiv ID:** 2603.19773 | [PDF](https://arxiv.org/pdf/2603.19773v1)

**作者:** Valentin Braeutigam `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Bernhard Egger `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5075603650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用SAM分割+CLIP/LPIPS特征分类的模板匹配方法，实现无需训练即可检测和识别汽车导航地图中的图标；

**💡 创新点**

无需收集训练集或重新训练，单模板即可应对图标设计变更，并通过文本去除与半遮挡增强鲁棒性；

**🔧 技术方法**

采用Segment Anything 2.1/3进行全图分割，使用CLIP或LPIPS特征做相似度匹配，并可选Inpaint Anything去除文本遮挡；

**📊 数据集**

使用e.Solutions GmbH提供的两套导航图像数据集（A：15,855张，B：37,260张），共计85类图标；

**📈 对比分析**

与YOLOv8/v11对比，精度/召回率约为99.4‑99.75%（含去文本），略低于YOLO的99.9%+但不需训练；

**⚠️ 局限性**

不适合需要学习未知物体的场景，去文本的Inpaint Anything步骤耗时高，且对极端遮挡仍需改进分类鲁棒性；

---

## 249. Journal Research Data Policies in Materials Science

**arXiv ID:** 2603.19301 | [PDF](https://arxiv.org/pdf/2603.19301v1)

**作者:** Lukas Hörmann `[一作]` (University of Vienna), Jonathan Schmidt `[通讯]` (ETH Zurich)

**通讯引用:** 4508 | [OpenAlex ID](https://openalex.org/A5057207447)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

对171本材料科学期刊的研究数据政策（RDP）进行编码与量化评估，构建了开放数据得分（ODS）并探讨了政策一致性与严格程度的关系。

**💡 创新点**

首次将数据与代码共享规范统一纳入编码框架，揭示政策在数据与代码层面的显著不一致性，并提出可执行的政策模板与改进建议。

**🔧 技术方法**

使用手工编码（双人对照）、Python API（Dimensions）、统计分析（相关性、ANOVA）、可视化工具（图表）以及自定义的开放数据得分公式。

**📊 数据集**

基于Dimensions API采集的171份期刊政策文本，涵盖17个出版商、不同影响因子、开放/闭合访问模式的期刊。

**📈 对比分析**

通过对比影响因子、出版商类型、开放访问政策与ODS，发现仅有弱关联；双人编码一致性与政策严格度呈负相关，表明更严格的政策更难统一解释；整体数据表明数据共享要求普遍不足，代码共享更为稀缺。

**⚠️ 局限性**

研究受限于样本选择（仅含主流期刊）、人工编码主观性、不同研究者对“强制”与“鼓励”的解释差异，以及政策文本更新导致的时间差异。

---

## 250. Sharing The Secret: Distributed Privacy-Preserving Monitoring

**arXiv ID:** 2603.20107 | [PDF](https://arxiv.org/pdf/2603.20107v1)

**作者:** Mahyar Karimi `[一作]` (Institute of Science and Technology Austria), Thomas A. Henzinger `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 43609 | [OpenAlex ID](https://openalex.org/A5080555605)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种分布式隐私保护监控协议，利用多方秘密共享实现系统输出的持续监测，且在不泄露系统状态和规范细节的前提下保持强信息理论安全；

**💡 创新点**

创新点在于：①将秘密共享扩展到可维护持久内部状态的连续监控；②引入高效的共享转换（算术↔布尔），实现混合协议计算；③在MP‑SPDZ框架下实现并证明在诚实多数假设下可实现实时监控，显著提升性能；

**🔧 技术方法**

核心技术包括：多方秘密共享（加法共享、Shamir共享）、共享转换协议、混合协议计算、MP‑SPDZ框架、预处理Beaver三元组以及基于诚实多数的安全模型；

**📊 数据集**

实验使用四个基准场景：访问控制系统（门数量10~1000）、分布式锁管理（锁数量100~1000）、总统车辆地理围栏（维度高达1024）、血糖监测（滑动窗口阈值）等；

**📈 对比分析**

与现有加密基方法（FHE、Garbled Circuits）对比，ACS场景每轮时延从18秒降低至0.07–0.18秒（提升约100–250×），锁管理从数分钟降至0.16–1.3秒（提升约14–112×），地理围栏与血糖监测也分别实现0.06–0.18秒与0.08秒的极低延迟；

**⚠️ 局限性**

主要局限包括：需要至少一台诚实监控节点（诚实多数假设）；仅在半诚实模型下安全，需额外机制抵御主动攻击；对大型系统状态共享仍不适用，需结合PIR等方案；集成到现有监控框架与对抗恶意攻击的效率仍待提升。

---

## 251. Enhancing Hyperspace Analogue to Language (HAL) Representations via Attention-Based Pooling for Text Classification

**arXiv ID:** 2603.20149 | [PDF](https://arxiv.org/pdf/2603.20149v1)

**作者:** Ali Sakour `[一作]` (Lattakia University), Zoalfekar Sakour `[通讯]` (Lattakia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 HAL（Hyperspace Analogue to Language）模型的词向量基础上，提出了利用截断 SVD 降维后再通过温度调节的可学习加性注意力层进行句子级聚合，并用于 IMDB 电影评论的情感二分类任务。

**💡 创新点**

创新点在于：① 将传统的均值池化替换为可学习的温度缩放注意力机制，显著抑制停用词干扰；② 通过截断 SVD 将高维稀疏共现矩阵压缩为 300 维稠密向量，解决维度灾难与计算瓶颈；③ 在经典统计分布式语义模型上首次结合深度注意力聚合，实现了可解释且性能更优的聚合策略。

**🔧 技术方法**

采用的技术包括：HAL 共现矩阵构建、截断 SVD 降维、温度缩放的可学习加性注意力层、层归一化、Dropout、MLP 分类器、Adam 优化器及 L2 正则化。

**📊 数据集**

使用了 IMDB 电影评论情感分析数据集（25,000 训练样本 + 25,000 测试样本），词表限制为 10,000 最频繁词，序列长度截断/补齐到 200。

**📈 对比分析**

与传统 HAL+均值池化模型进行对比；注意力模型在测试集上的准确率从 75.64% 提升至 82.38%，提升幅度 6.74pp，且收敛速度更快、训练更稳定。

**⚠️ 局限性**

局限性：仅在情感分类任务上验证，未评估在多类分类、NLI 等更复杂任务中的表现；对子词/字符级表示的兼容性待探索；模型依赖于 SVD 降维，可能在极大词表或极稀疏语料下表现不佳。

---

## 252. Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture -- Bridging Predictive and Generative Self-Supervised Learning

**arXiv ID:** 2603.20111 | [PDF](https://arxiv.org/pdf/2603.20111v1)

**作者:** Moritz Gögl `[一作]` (University of Oxford), Christopher Yau `[通讯]` (Health Data Research UK)

**通讯引用:** 21035 | [OpenAlex ID](https://openalex.org/A5084571119)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将Joint-Embedding Predictive Architecture（JEPA）重新表述为耦合潜在变量模型，提出变分JEPA（Var-JEPA）并在表格数据上实现Var-T-JEPA，使用统一的ELBO进行训练。

**💡 创新点**

创新点在于：①将JEPA的确定性预测器解释为条件先验，构建完整的变分推理框架；②通过ELBO自带的KL正则化消除手工抗崩溃正则（如EMA、SIGReg）的需求；③利用潜在分布的协方差提供可解释的置信度估计。

**🔧 技术方法**

使用技术包括：变分推理、ELBO、重参数化技巧、Gaussian潜在空间、Transformer编码器、对比实验中的SIGReg、以及下游分类器（MLP、DCNv2、ResNet、AutoInt、FT-Transformer、XGBoost）。

**📊 数据集**

实验数据集包括：Adult、Covertype、Electricity、Credit Card Default、Bank Marketing、MNIST（表格化处理）以及全合成的SIM数据。

**📈 对比分析**

通过与T-JEPA以及多种原始特征基线进行对比，Var-T-JEPA在大多数表格数据集上取得更高或相当的下游分类准确率；此外，在MNIST/SIM上展示了利用潜在不确定性实现的覆盖-准确性曲线和高置信度样本的选择性评估。

**⚠️ 局限性**

限制与挑战：①目前仅在表格和简单视觉数据上验证，缺乏对大规模图像/视频任务的实验；②在测试时缺少目标观测时的生成能力尚未充分利用；③模型的维度、网络结构和超参数仍需经验性调优。

---

## 253. All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution

**arXiv ID:** 2603.19595 | [PDF](https://arxiv.org/pdf/2603.19595v1)

**作者:** Can Lv `[一作]` (Beihang University), Shiji Zhou `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 All-Mem，一个在线/离线双阶段终身记忆框架，通过可见表面、非破坏性拓扑编辑（Split、Merge、Update）和类型化链接实现长期记忆的维护与检索。

**💡 创新点**

创新点：1）引入可见表面限定检索空间；2）采用非破坏性拓扑编辑保持可追溯性；3）设计基于类型化链接的预算化检索，兼顾实时性与深度检索；4）将记忆维护移至离线阶段，实现在线低延迟。

**🔧 技术方法**

技术：动态拓扑演化、LLM 驱动的诊断与规划、稀疏链接拓扑、可见表面过滤、预算化多跳检索、并行离线编辑、嵌入向量检索。

**📊 数据集**

数据集：LoCoMo 与 LongMemEval-s 两个长期对话记忆基准。

**📈 对比分析**

比较方法：对齐检索与生成预算，评估 QA（4o-J、F1）与检索指标（R@5、N@5）。All-Mem 在两大基准上均优于 Naive RAG、MemGPT、A-Mem、HippoRAG2、Mem0、LightMem，且在查询时注入的上下文长度更短，检索延迟更低，性能提升显著。

**⚠️ 局限性**

局限：1）对离线调度与阈值设定敏感；2）仅在实验基准上验证，缺乏真实环境评估；3）隐私与安全风险需进一步治理；4）对非常大规模记忆的可扩展性尚待验证。

---

## 254. TrustFlow: Topic-Aware Vector Reputation Propagation for Multi-Agent Ecosystems

**arXiv ID:** 2603.19452 | [PDF](https://arxiv.org/pdf/2603.19452v1)

**作者:** Volodymyr Seliuchenko `[一作]` `[通讯]` (Robutler AI), Volodymyr Seliuchenko (Robutler AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 TrustFlow，给软件代理分配多维向量声誉，并在交互图上通过主题门控传递算子迭代收敛，形成可直接点积检索的声誉向量。

**💡 创新点**

创新点包括：①多维向量声誉与主题门控的联合传播，能够捕捉跨域专长；②提供连续与离散两种实现；③设计 Lipschitz‑1 传递算子集合（projection、squared gating、scalar‑gated、hybrid），保证收敛；④引入负信任边与信息论门控（KL、熵、置信）实现防御；⑤声誉与查询共享同一嵌入空间，支持一次性点积检索。

**🔧 技术方法**

技术手段主要包括：PageRank 框架与收敛理论、Lipschitz‑1 传递算子、KL 互信息门、负信任边机制、embedding 预处理（均值中心化）、multilingual‑e5‑small 嵌入、点积检索与多标签评分。

**📊 数据集**

实验数据集：50 个代理，覆盖 8 个领域（医学、法律、金融、编码、网络安全、教育、创意、数据科学），70 条带标签交互边、612 条盲交互边，10 条自然语言查询；使用 384 维 multilingual‑e5‑small 嵌入并进行均值中心化。

**📈 对比分析**

评估方式：采用 P@5（严格与多标签）与自我对齐度；在混合图上连续方案 P@5 达到 78%/88%，离散方案在标注图上 74%；不同传递算子对比显示投影算子在混合图上最高；对四类攻击（跨域 Sybil、同域 Sybil、洗钱、投票环）实验表明 P@5 影响 ≤4pp；图密度提升可使 P@5 提升 10–20pp。

**⚠️ 局限性**

局限性：实验规模仅 50 个代理，缺乏大规模验证；对均值中心化的依赖较强；盲边代理质量受限，恶意操纵盲边难以检测；负信任边需可信报告机制；未考虑时间衰减与动态更新。

---

## 255. Dual Prompt-Driven Feature Encoding for Nighttime UAV Tracking

**arXiv ID:** 2603.19628 | [PDF](https://arxiv.org/pdf/2603.19628v1)

**作者:** Yiheng Wang `[一作]` (Duke University), Zijie Zhang `[通讯]` (Tongji University)

**通讯引用:** 11636 | [OpenAlex ID](https://openalex.org/A5100415698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种双提示驱动特征编码方法DPBlock，结合了光照提示和视角提示，实现对夜间无人机跟踪的鲁棒特征学习；

**💡 创新点**

创新点在于将提示-特征相互作用嵌入Vision Transformer中，并设计了层次化的光照提示生成器与基于可变形卷积的视角提示生成器，显著提升低光与动态视角条件下的跟踪性能；

**🔧 技术方法**

采用Prompt‑Feature Interaction（PFI）、Pyramid Illumination Prompter（PIP）与Dynamic Viewpoint Prompter（DVP）、Vision Transformer骨干及Deformable Convolution等技术；

**📊 数据集**

使用日间数据集（GOT‑10K、LASOT、COCO、TrackingNet）与夜间数据集（ExDark、Shift、BDD100K）进行训练；

**📈 对比分析**

与UAVDark135、DarkTrack2021、NAT2021‑test三大基准对比，DPTracker‑B在精度、归一化精度和成功率上分别提升约2–3%，达到或超过当前SOTA；

**⚠️ 局限性**

限制在于对极端低光/高动态场景仍存在跟踪漂移风险，且模型训练依赖大规模混合数据集，实际部署时对硬件资源有一定要求。

---

## 256. PFM-VEPAR: Prompting Foundation Models for RGB-Event Camera based Pedestrian Attribute Recognition

**arXiv ID:** 2603.19565 | [PDF](https://arxiv.org/pdf/2603.19565v1)

**作者:** Minghe Xu `[一作]` (City University of Macau), Yu Li `[通讯]` (Zhuhai College of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种轻量级的事件提示模块与双模态记忆增强机制，用于 RGB-Event 融合的行人属性识别。

**💡 创新点**

创新点在于用 DCT/IDCT 直接对事件流进行频域编码作为提示，并结合现代 Hopfield 网络实现内外双重记忆提升特征。

**🔧 技术方法**

采用 Vision Transformer 主干、Discrete Cosine Transform、逆 DCT、现代 Hopfield 网络、跨注意力融合和全连接分类头。

**📊 数据集**

实验使用 EventPAR 及 DukeMTMC-VID-Attribute 两个基准数据集。

**📈 对比分析**

与多种 SOTA 方法对比，PFM‑VEPAR 在 EventPAR 上实现 90.05% mA，速度最快，且在 Duke 上取得 67.58% Acc、80.21% Prec，领先或相近。

**⚠️ 局限性**

局限在于对细粒度或静态小目标（如眼镜、背包）的识别仍不佳，并且模型多阶段设计导致对超参敏感。

---

## 257. PRIME-CVD: A Parametrically Rendered Informatics Medical Environment for Education in Cardiovascular Risk Modelling

**arXiv ID:** 2603.19299 | [PDF](https://arxiv.org/pdf/2603.19299v1)

**作者:** Nicholas I-Hsien Kuo `[一作]` (University of New South Wales), Louisa Jorm `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 PRIME-CVD，一套基于因果有向无环图（DAG）参数化生成的心血管疾病（CVD）模拟数据，分为干净可直接分析的 Data Asset 1 和模拟真实 EMR 结构的 Data Asset 2，旨在为医学教育与方法论训练提供可复现、隐私友好的数据资源。

**💡 创新点**

创新点在于完全使用公开统计与流行病学效应估计构建 DAG 模型，避免依赖真实 EMR 或生成式机器学习，既保留真实亚组失衡与风险梯度，又实现零再识别风险；同时提供两层数据资产，一层易于建模，一层模拟 EMR 的结构与文字杂乱性，满足不同教学需求。

**🔧 技术方法**

采用因果 DAG 生成、参数化采样、比例模型与比例风险模型等统计方法；生成过程代码公开，并通过 deterministic 脚本将清洁数据转化为 EMR‑style 结构，注入缺失、异构术语、单位不一致等“噪声”。

**📊 数据集**

使用公开的澳大利亚人口统计、AIHW 慢性病患病率与已发表的流行病学效应估计作为参数源，生成 5 万个模拟成年人的数据；Data Asset 1 为分析就绪的 CSV，Data Asset 2 则为三张关联的 EMR‑style CSV。

**📈 对比分析**

通过与已公开的 MIMIC 等真实 EMR 资源对比，PRIME‑CVD 在维持整体分布、亚组梯度与风险比例（如 Cox HR 与实际研究相近）方面表现优异；但因无真实个体数据，无法直接评估模型在真实临床预测中的性能，只能在教育与方法验证场景下使用。

**⚠️ 局限性**

局限性包括：① 未能完全模拟真实 EMR 的所有复杂关系与稀有事件；② 生成的因果结构基于已知文献，可能忽略未知关联；③ 作为教育工具，其预测模型不具备临床部署价值；④ 对极端罕见亚组的再现仍有限。

---

## 258. lit-tag: An app for adding custom tags and notes to a citation database

**arXiv ID:** 2603.19238 | [PDF](https://arxiv.org/pdf/2603.19238v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 259. Scalable Learning of Multivariate Distributions via Coresets

**arXiv ID:** 2603.19792 | [PDF](https://arxiv.org/pdf/2603.19792v1)

**作者:** Zeyu Ding `[一作]` (Lamarr Institute for Machine Learning and Artificial Intelligence), Simon Omlor `[通讯]` (Scientific Computing Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种针对多元条件变换模型（MCTM）的核心子集（coreset）构造方法，实现对大规模数据的高效训练。

**💡 创新点**

创新点在于首次将 ℓ₂ 重要性抽样与凸包（convex hull）选择相结合，既保证对二次部分的保真，又稳定对数项，且给出了理论误差上界。

**🔧 技术方法**

核心技术包括 ℓ₂ 子空间嵌入、敏感度（sensitivity）采样、凸包逼近、Bernstein 多项式基函数和高斯 Copula，整体实现半参数分布估计。

**📊 数据集**

实验数据集包括 14 种 2 维模拟分布、UCI Covertype（约 300k 样本、10 连续特征）以及两个股票回报数据（10 只/20 只、约 10k 样本）。

**📈 对比分析**

与均匀抽样、纯 ℓ₂ 重要性抽样、岭杠杆分数、根杠杆分数等基线相比，提出的方法在 log‑似然、参数误差（ϑ、λ）上显著优于均匀抽样，且在 12/14 的模拟情形下优于纯 ℓ₂ 重要性抽样；运行时间与均匀抽样相近，略低于纯 ℓ₂ 方法。

**⚠️ 局限性**

局限性包括：在高维或重尾分布时凸包构造成本指数级增长，需要更大核心子集；对极端点的敏感度处理依赖于 η‑kernel，超参数调优较难；目前仅适用于高斯 Copula，可推广至其它 log‑concave Copula 仍需研究。

---

## 260. Teaching Practically Relevant Research Problem Formulation in Software Engineering with Lean Research Inception

**arXiv ID:** 2603.19967 | [PDF](https://arxiv.org/pdf/2603.19967v1)

**作者:** Anrafel Fernandes Pereira `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

在巴西一所大学的Capstone项目课程中，使用Lean Research Inception（LRI）和Problem Vision board帮助60名软件工程本科生及7名导师制定实践相关的研究问题。

**💡 创新点**

将LRI作为教学工具，系统评估其在研究问题制定阶段的教育效果，填补了以往仅关注项目执行或经验方法的研究空白。

**🔧 技术方法**

采用LRI框架与Problem Vision board工具，结合案例研究、问卷调查、描述性统计与定性内容分析。

**📊 数据集**

使用60名学生和7名导师的问卷数据作为实验数据集，未使用公开数据集。

**📈 对比分析**

通过描述性统计（如60%提升反思、61.7%清晰度、50%沟通、85.7%教师推荐率）和定性访谈，展示了学生对LRI的高接受度和教师的推荐意愿，但未与其他教学方法做直接对比。

**⚠️ 局限性**

局限性包括单一机构、样本量有限、依赖自报问卷可能存在社会期望偏差、未做推断统计或对照实验，难以将结果泛化到其他学科或教学环境。

---

## 261. Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification

**arXiv ID:** 2603.19678 | [PDF](https://arxiv.org/pdf/2603.19678v1)

**作者:** Kunlun Xu `[一作]` (Peking University), Jiahuan Zhou `[通讯]` (Peking University)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5055004003)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉‑语言模型的终身行人重识别框架 VLADR，利用多粒度文本属性解耦和跨域属性强化机制，持续学习并巩固人类属性知识。

**💡 创新点**

创新点在于：①通过 Multi‑grain Text Attribute Disentanglement 同时提取全局与局部文本属性，①将视觉与文本属性进行跨模态对齐，②引入跨域属性对齐以实现知识迁移与抗遗忘；这些机制首次在终身行人重识别中实现细粒度属性驱动的学习。

**🔧 技术方法**

使用了 Vision‑Language Models（CLIP、BLIP‑2）提取文本属性，ViT 视觉骨干，交叉注意力解码器进行局部属性提取，Contrastive 以及 KL 损失实现跨模态与跨域对齐；整体基于 CLIP‑ReID 基线构建。

**📊 数据集**

采用 LReID 公开基准数据集：5 个训练域（Market‑1501、LPW_s2、CUHK‑SYSU、MSMT17‑V2、CUHK03）与 7 个评估域（CUHK01、CUHK02、VIPeR、PRID、i‑LIDS、GRID、SenseReID），并设置两种训练顺序以模拟不同域间差异。

**📈 对比分析**

与 ResNet/ViT/CLIP‑基线的 LReID 方法对比，VLADR 在 Seen‑Avg 上提升 1.9%–2.5%，Unseen‑Avg 上提升 2.1%–2.2%；整体在两种训练顺序下均取得领先，显著提升抗遗忘与跨域泛化性能。

**⚠️ 局限性**

局限性：①依赖 VLM 预训练和文本生成模型，对文本质量敏感；②仅关注人类四个局部属性，难以扩展至更细粒度或非人类对象；③未系统评估计算成本与实时推理能力；④在极端域迁移场景下的鲁棒性尚未验证。

---

## 262. Coordinating Stakeholders in the Consideration of Performance Indicators and Respective Interface Requirements for Automated Vehicles

**arXiv ID:** 2603.19492 | [PDF](https://arxiv.org/pdf/2603.19492v1)

**作者:** Richard Schubert `[一作]`, Steven Peters `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文设计并验证了一套面向自动驾驶车辆的利益相关者协同过程，用于识别性能指标（PI）并定义相应的接口需求，确保在SAE Level 4 车辆运行时能够实时监测性能并应对外部扰动与内部失效。

**💡 创新点**

创新点在于：①将自顶向下的安全需求分解与自底向上的失效分析相结合，形成统一的PI识别与接口定义方法；②明确角色分工与迭代反馈循环；③引入“指标日志”与“接口控制文件（ICD）”作为核心产物，保证了需求与实现之间的可追溯与透明度；④将过程落地到实际项目中并通过工作坊实现多方协同。

**🔧 技术方法**

使用的技术包括：SysML Activity Diagram（在 Enterprise Architect 中绘制）、角色与活动模型、接口控制文件（ICD）规范、Kampmann ASOA/动态架构平台用于架构描述与可视化、以及基于案例的安全分析工具（如安全概念、故障树/STPA 等）。

**📊 数据集**

本过程主要在 Vankempen AutotechAgil 项目中应用，利用该项目的四个自动驾驶原型车辆的业务情景、事故情景与功能安全案例作为输入；并未使用公开数据集，过程的输入来自项目内部的场景描述和安全分析文档。

**📈 对比分析**

由于是流程改进而非算法评估，本文通过定性比较评估：过程应用后需求捕获更完整、接口一致性提升、文档可追溯性加强、利益相关者沟通效率显著提高；未给出定量性能指标，只描述了过程改进带来的质量提升。

**⚠️ 局限性**

主要局限性包括：①阈值设定与性能异常判定未在范围内；②使用 ASOA 平台时缺乏严格的模型化追溯，导致架构决策的正式可追溯性不足；③对系统架构整体可视化与跨团队共享仍需改进；④未系统化整合适应性动作的生成；⑤在多组织跨团队环境下过程的可扩展性与适应性尚未充分验证。

---

## 263. Bridging Conformal Prediction and Scenario Optimization: Discarded Constraints and Modular Risk Allocation

**arXiv ID:** 2603.19396 | [PDF](https://arxiv.org/pdf/2603.19396v1)

**作者:** Giuseppe C. Calafiore `[一作]` (Politecnico di Torino), Giuseppe C. Calafiore `[通讯]` (Politecnico di Torino)

**通讯引用:** 10451 | [OpenAlex ID](https://openalex.org/A5041746196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过系统与控制视角，将 conformal prediction 与 scenario optimization 之间的联系进行桥接，提出了面向样本丢弃的前向桥接和多块校准证书的模块化组合规则，并用多步预测管道与 MPC 例子验证其效果。

**💡 创新点**

创新点在于：①在可行的样本丢弃算法下给出基于交换性的新平均风险上界；②提出简单的块级校准合并规则，可将多维风险预算明确分配到不同坐标、约束或时间步；③将两者结合得到的校准管道在同一总风险预算下实现不同的安全裕度与性能折衷。

**🔧 技术方法**

主要技术包括：交换性与稳定重构集的理论框架、分块校准与联合置信度的组合（基于联合分布或独立性时的乘法规则）、以及对多步残差取分位数构造预测管道；同时使用分布无关的 conformal 推导与 scenario 期望风险律相结合。

**📊 数据集**

实验使用人工设计的四步非线性系统（x_{t+1}=0.78x_t+0.35u_t+0.12x_tu_t+w_t，w_t∼N(0,0.08^2)）以及识别出的线性预测模型，生成 120 条校准任务和 5000 条测试任务；未使用公开数据集，而是通过模拟数据验证。

**📈 对比分析**

比较方法：在同一总风险预算（∑ε_k=0.22）下，对三种阶段风险分配（递增、均匀、递减）分别构造预测管道，并将其嵌入简单规划问题评估控制输入、轨迹风险和终端输出。结果表明，平均轨迹风险相近，但不同分配导致约束裕度、控制输入和违约概率显著差异；在假设独立误差时，乘法规则可将风险上界从 0.22 降至约 0.2026，进一步放宽阶段风险。

**⚠️ 局限性**

局限性：仅给出期望风险上界，未得到 PAC 风险尾部界；模块化组合基于并集上界，无法利用块间相关性；仅适用于可交换数据，无法直接处理时间序列依赖；未涉及闭环控制下的在线校准与决策。

---

## 264. CS-MUNet: A Channel-Spatial Dual-Stream Mamba Network for Multi-Organ Segmentation

**arXiv ID:** 2603.19659 | [PDF](https://arxiv.org/pdf/2603.19659v1)

**作者:** Yuyang Zheng `[一作]` (Yunnan University), Hongyi Huang `[通讯]` (Yunnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出CS-MUNet框架，用于腹部多器官的精准分割，结合双流SSM Mamba模块实现空间边界感知和通道语义协同。

**💡 创新点**

创新点包括：①Boundary-Aware State Mamba（BASM）将像素级边界后验直接注入Mamba状态转移参数（Δ/B），实现边界感知的结构性集成；②Channel Mamba State Aggregation（CMSA）将通道维度视为SSM序列维度，显式建模跨通道解剖语义协作，并通过有界状态转移约束防止状态失控。

**🔧 技术方法**

技术手段涵盖Mamba线性复杂度状态空间模型、贝叶斯注意力生成边界后验、双分支结构（Mamba分支+多尺度SASF分支）以及SE风格权重融合；通道序列化与分组共享SSM，且使用剪裁约束保证状态有界；基于Res2Net-50骨干，训练时采用Dice-CE联合深度监督。

**📊 数据集**

使用公开的UW‑Madison GI Tract（MRI）和WORD（CT）两个腹部多器官分割基准数据集。

**📈 对比分析**

在两数据集上对比9种基线（U‑Net、Att‑UNet、TransUNet、Swin‑UNet、UNet++、MSDUNet、SCUNet++、EGE‑UNet、nnU‑Net），CS‑MUNet在UW‑Madison取得86.16% mDice、94.47% mDice，在WORD取得94.47% mDice、92.51% mIoU，分别比最强基线提升2.91%和1.03%，参数量52.22M，计算开销38.8G FLOPs。

**⚠️ 局限性**

局限性：当前为2D框架，难以保证体素级3D边界连续性；模型参数已较大，轻量化改进尚未完成；在超声或PET‑CT等其他影像模态的泛化性尚未验证。

---

## 265. TAPAS: Efficient Two-Server Asymmetric Private Aggregation Beyond Prio(+)

**arXiv ID:** 2603.19949 | [PDF](https://arxiv.org/pdf/2603.19949v1)

**作者:** Harish Karthikeyan `[一作]` (JPMorgan), Antigoni Polychroniadou `[通讯]` (JPMorgan)

**通讯引用:** 880 | [OpenAlex ID](https://openalex.org/A5030114656)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了名为TAPAS的两服务器非对称私有聚合协议，能够在不需要可信前置设置、无客户端预处理的情况下，安全聚合高维度分布式数据。

**💡 创新点**

创新点包括：将O(L)计算和通信负载集中到主服务器，使次服务器轻量化；仅基于LWE/SIS实现后量子安全；设计新的格密码学零知识证明体系，支持可识别abort；并实现服务器端通信与向量维度L无关。

**🔧 技术方法**

使用技术主要有：格密码学（LWE、SIS）、基于格的Bulletproofs或LWE的零知识证明、独立区块重启技术、可编程随机预言模型（PRo）以及加密与签名方案。

**📊 数据集**

实验数据使用模拟生成的大维度向量（如L=2^18、百万参数模型），未采用公开数据集，仅通过合成数据验证性能。

**📈 对比分析**

与Prio、Elsa、Heli等现有协议对比，TAPAS在大维度场景下实现了相较Prio 93×、相较Elsa 17.5×的速度提升，并且次服务器成本与向量维度L无关。

**⚠️ 局限性**

局限性在于仍需服务器不合谋假设；在极低客户端数量或极高噪声预算下可能出现效率下降，且对极端规模的安全性仍需进一步评估。

---

## 266. BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates

**arXiv ID:** 2603.19718 | [PDF](https://arxiv.org/pdf/2603.19718v1)

**作者:** Phuong-Anh Nguyen `[一作]` (VNU University of Engineering and Technology), Cam-Van Thi Nguyen `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个轻量级、模型无关的插件框架BALM，用于在不同模态缺失率不平衡的条件下实现多模态学习。

**💡 创新点**

创新点在于同时通过特征校准模块（Feature Calibration）对不同缺失模式下的单模态表示进行全局上下文调节，以及通过梯度重平衡模块（Gradient Rebalancing）从分布和空间两个角度动态调节各模态的梯度，解决了表示和优化两方面的不平衡。

**🔧 技术方法**

采用了基于全局上下文的特征归一化、激励门控、KL 散度与余弦相似度的梯度调节、轻量级单模态预测头以及联合任务+调节损失的训练策略，兼容多种 MER 背骨。

**📊 数据集**

在 IEMOCAP 和 CMU‑MOSEI 两个多模态情绪识别数据集上进行实验。

**📈 对比分析**

与缺失模态学习、模态不平衡处理及常规 MER 背骨的多组基线对比，BALM 在所有 IMR/SMR 配置下均提升了 1–7% 的 Acc 或 w‑F1，尤其在高缺失率场景下可获得 4–7% 的显著提升。

**⚠️ 局限性**

局限性包括：仅在情绪识别任务上验证；对极端缺失率仍存在一定性能下降；需对调节超参数进行经验调优；未给出理论收敛性证明；可能对其他多模态任务的泛化需进一步评估。

---

## 267. Continual Learning for Food Category Classification Dataset: Enhancing Model Adaptability and Performance

**arXiv ID:** 2603.19624 | [PDF](https://arxiv.org/pdf/2603.19624v1)

**作者:** Piyush Kaushik Bhattacharyya `[一作]` (KIIT University), N Sangita Achary `[通讯]` (KIIT University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5072283901)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种持续学习框架，用于在文本指导下对食品进行分类，支持逐步加入新类别而不需重新训练。

**💡 创新点**

创新点在于将TF‑IDF文本特征与增量学习结合，并通过渐进式提示和知识蒸馏来避免灾难性遗忘，能够实时接受新菜品。

**🔧 技术方法**

采用TF‑IDF文本向量、Keras实现的前馈神经网络、SMOTE、Adam优化器、L2正则化、早停等技术，并与传统机器学习模型进行对比。

**📊 数据集**

数据来自网络爬取的25,192条菜名，经过手工标注为素食/非素食后，用TF‑IDF向量化后构建。

**📈 对比分析**

与Logistic Regression、Random Forest、SVM、KNN等基线模型比较，SVM表现最好；持续学习模型在准确率约98.38%、召回率99.36%、F1≈98.1%、AUC≈98.1%上与SVM相近，且方差更小。

**⚠️ 局限性**

局限包括数据集缺乏多样性导致高方差、模型在新类别上仍可能出现误分类、未充分验证跨文化菜品的泛化能力。

---

## 268. GazePrinter: Visualizing Expert Gaze to Guide Novices in a New Codebase

**arXiv ID:** 2603.19855 | [PDF](https://arxiv.org/pdf/2603.19855v1)

**作者:** Peng Kuang `[一作]` (Lund University), Martin Höst `[通讯]` (Malmö University)

**通讯引用:** 17258 | [OpenAlex ID](https://openalex.org/A5045592913)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GazePrinter 工具，将专家程序员的注视模式可视化并在 IntelliJ IDE 中提供视觉提示，以帮助初学者在新的代码库中进行程序理解。

**💡 创新点**

创新点在于：① 将专家眼动数据直接嵌入开发环境，实时呈现注视热图；② 通过对比实验组与对照组，首次系统评估眼动可视化对阅读策略、学习体验和认知负荷的影响；③ 结合认知学与软件工程的多学科视角，探讨“感知复杂度”与实际代码复杂度的关系。

**🔧 技术方法**

技术手段包括：Tobii 4C 眼动追踪、CodeGRITS 收集与处理眼动日志、GazePrinter 作为 IntelliJ 插件进行可视化；数据分析使用动态时间规整（DTW）、Needleman‑Wunsch、Jaccard 相似度、NASA‑TLX 量表、Cliff’s Δ 等统计方法。

**📊 数据集**

数据集：两份中等规模 Java Spring‑Boot 项目（LMS 与 E‑Commerce），共约 6300 行代码；专家眼动样本来自 5 名资深 Java 开发者（经筛选后使用 3 名有效数据），实验样本 40 名计算机专业新人。

**📈 对比分析**

对照方法为混合实验：实验组使用 GazePrinter、对照组不使用；通过任务完成时间、得分、认知负荷和眼动行为（文件/模块顺序、注意力分布）进行比较。结果显示实验组在文件和模块阅读顺序、注意力分布与专家更为接近，表现出显著的阅读模式迁移；但在任务完成时间、准确率和认知负荷方面差异不显著。

**⚠️ 局限性**

局限性包括：样本量相对较小、仅涉及 Java/Spring Boot，缺乏跨语言和跨规模的验证；实验时间短，未能评估长期学习效果；工具的可视化方式（颜色条）可能引入额外认知负荷，且对团队协作场景的实际适用性仍需进一步研究。

---

## 269. Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through Layer-Wise Target Propagation

**arXiv ID:** 2603.19742 | [PDF](https://arxiv.org/pdf/2603.19742v1)

**作者:** Lasse Marten Jantsch `[一作]` (Kyungpook National University), Young-Kyoon Suh `[通讯]` (Kyungpook National University)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5101626811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Dual Path Attribution (DPA)，一种在冻结Transformer上通过目标向下传播并将SwiGLU结构拆分为内容与控制路径，以一次前向+一次反向实现高效、精准的输入与组件归因；

**💡 创新点**

通过解析SwiGLU Transformer的双线性结构，将其拆分为内容与控制路径，构建一次前向+一次反向即可完成稠密组件归因，显著降低计算复杂度并提升归因可信度；

**🔧 技术方法**

目标向下传播、线性化逆变换、内容/控制路径分解、软化权重μ调节、SwiGLU（GLU）与多头自注意力的解析逆推；

**📊 数据集**

在多种LLM上（Llama‑3.1‑8B‑Instruct、Llama‑2‑7B‑Chat、Qwen3‑4B‑Instruct、Mistral‑7B‑Instruct、Qwen2.5‑32B‑Instruct）评测，使用的数据集包括 Known 1000、SQuAD v2.0、IMDb 与 IOI；

**📈 对比分析**

与梯度、注意力、上下文混合、DePass 等多种归因基线比较；在 disruption、recovery、总分等指标上，DPA 均领先并在稠密组件归因中取得 state‑of‑the‑art 的可信度；

**⚠️ 局限性**

仅适用于标准 SwiGLU Transformer；对高曲率或高度耦合非线性交互的近似失效；归因仅到 logit 级别，无法捕获软最大概率间接效应；需要缓存大量激活，内存开销较大。

---

## 270. ProactiveBench: Benchmarking Proactiveness in Multimodal Large Language Models

**arXiv ID:** 2603.19466 | [PDF](https://arxiv.org/pdf/2603.19466v1)

**作者:** Thomas De Min `[一作]` (University of Trento), Massimiliano Mancini `[通讯]` (University of Trento)

**通讯引用:** 1372 | [OpenAlex ID](https://openalex.org/A5017971549)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布 ProactiveBench 基准，系统评估多模态大语言模型（MLLMs）在七个不同场景下的主动提问与协作行为；通过对 22 种 MLLM 进行多轮对话与单轮开放式生成实验，探讨提示、对话历史、少量示例与强化学习微调对主动性的影响。

**💡 创新点**

1) 首次构建专门测评 MLLM 主动性的基准；2) 将七个已有数据集重构为包含最初模糊视图、目标帧与中间帧的多轮交互序列；3) 设计过滤机制保证样本确实需要主动请求；4) 展示通过 GRPO 强化学习微调可显著提升模型主动性并实现跨场景泛化。

**🔧 技术方法**

多模态大语言模型（22 种），多轮决策过程（将主动请求视为动作），提示工程（含主动性提示、对话历史、少量示例），GRPO 强化学习微调，基准评估脚本（MCQA 与 OEG 两种模式），LLM-as-Judge 评分器。

**📊 数据集**

ROD、VSOD、MVP‑N、ImageNet‑C、QuickDraw、ChangeIt、MS‑COCO；经重构后共 7,557 个样本，包含 19 种主动行为。

**📈 对比分析**

在 MCQA 评估中记录准确率（acc）和主动率（ps），在 OEG 评估中记录整体准确率（agg）。结果显示大多数模型的主动率低，准确率仅 20–40%，提示与历史的引入虽能提升主动率但往往伴随准确率下降；RL 微调后某些模型在未见过的场景中也能显著提升主动率与准确率，但整体仍未达到参考帧（完整信息）下的 80%+准确率。

**⚠️ 局限性**

缺乏真正的主动性，提示/历史/ICL 方式不稳定，RL 微调需要手工设计奖励且计算成本高；基准样本多来自人工重构，真实交互环境和物理操作仍未覆盖；模型在未知场景下的泛化仍有限，无法完全消除对视觉模糊的直接放弃或幻觉行为。

---

## 271. A Novel Solution for Zero-Day Attack Detection in IDS using Self-Attention and Jensen-Shannon Divergence in WGAN-GP

**arXiv ID:** 2603.19350 | [PDF](https://arxiv.org/pdf/2603.19350v1)

**作者:** Ziyu Mu `[一作]`, Safak Dogan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提供了新的 LaTeX 文档类 elsarticle.cls，用于 Elsevier 期刊投稿，支持多种格式、引用、图表等。

**💡 创新点**

基于 article.cls 重写，降低与其他宏包冲突；提供多种预设样式、简化引用与定理环境；支持双盲、不同字体等选项。

**🔧 技术方法**

使用 natbib、geometry、graphicx、txfonts、hyperref 等常见宏包，内部实现基于标准 LaTeX 机制。

**📊 数据集**

无数据集。

**📈 对比分析**

无比较方法与性能评估，主要是使用示例文档验证功能。

**⚠️ 局限性**

依赖外部宏包，若缺失会导致错误；双栏排版中长公式需手工断行。

---

## 272. HiFiGaze: Improving Eye Tracking Accuracy Using Screen Content Knowledge

**arXiv ID:** 2603.19588 | [PDF](https://arxiv.org/pdf/2603.19588v1)

**作者:** Taejun Kim `[一作]` (Carnegie Mellon University), Chris Harrison `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10608 | [OpenAlex ID](https://openalex.org/A5029290807)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出HiFiGaze方法，利用高分辨率前置摄像头捕捉眼睛中的屏幕反射，并结合屏幕内容实现无标定的视线估计。

**💡 创新点**

创新点在于利用设备已知的屏幕内容来准确分割反射，提升反射定位精度并显著降低误差。

**🔧 技术方法**

采用高分辨率图像预处理、GrabCut+RANSAC眼球中心校正、卷积热图与向量编码，并以MobileNetV4+全连接网络进行预测。

**📊 数据集**

使用WebUI 30万网页屏幕内容作为训练素材，并在iPhone 14 Pro Max上采集22名受试者约17.7万帧眼动数据。

**📈 对比分析**

与传统仅使用眼部图像的基线相比，HiFiGaze在无标定条件下平均误差从2.00 cm降至1.64 cm，提升约18%，在亮暗屏幕上均优于基线。

**⚠️ 局限性**

局限包括暗色屏幕反射弱、上睑/睫毛遮挡导致误差、未覆盖佩戴眼镜者、光照多样性不足、虹膜中心估计不稳定以及受试者多样性有限。

---

## 273. LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation

**arXiv ID:** 2603.20192 | [PDF](https://arxiv.org/pdf/2603.20192v1)

**作者:** Jiazheng Xing `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 35953 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套针对个性化多主体视频生成的完整框架，包含开放集合数据采集流水线与显式建模人脸-属性依赖的注意力模块，能够在保持身份一致性和主体一致性的同时实现前景、背景的灵活定制。

**💡 创新点**

创新点主要有：1）构建了支持开放集合、多主体、面部与属性显式对应的数据采集流程，解决了缺乏标注依赖数据的问题；2）设计了Relational Self‑Attention（包含R2PE与CSAM）和Relational Cross‑Attention（MCAM）两大模块，在位置编码与注意力掩码层面显式绑定人脸与属性，强化同组内关联、抑制跨组干扰。

**🔧 技术方法**

技术上基于Wan2.1文本‑视频Diffusion Transformer（DiT）架构，融合3D‑RoPE、R2PE、Causal Self‑Attention Mask、Multilevel Cross‑Attention Mask、Flow Matching等；数据预处理借助VILA、Qwen2.5‑VL、SAM、GroundingDINO、FLUX等模型；训练采用Adam+EMA，使用了GPU‑高算力。

**📊 数据集**

训练数据来源于Panda70M，经过清洗与处理后得到1.57M视频样本（单主体1.31M、双主体0.23M、三主体0.03M）；测试基准为500条YouTube视频（单/双/三主体各类）。另外在数据采集流水线中使用了VILA、Qwen2.5‑VL等辅助模型。

**📈 对比分析**

与ConsisID、Concat‑ID、SkyReels‑A2、Phantom等先进方法在身份一致性（ArcSim、CurSim、ViCLIP‑T）和主体一致性（ViCLIP‑T/V、CLIP‑T、ArcSim）等指标上均取得SOTA表现。例如，在单主体身份一致性上ArcSim由0.508提升至0.510，整体视频与主体的语义相似度、动态一致性等指标亦优于对比模型。

**⚠️ 局限性**

局限性包括：1）训练仅支持每个主体最多3个属性，推理时亦需保持同一约束；2）目前支持的主体数量有限，过多主体时性能可能下降；3）对高帧率、高分辨率视频的生成仍需要更多算力；4）数据采集流程依赖多模型推理，效率与成本较高。

---

## 274. Transformers are Stateless Differentiable Neural Computers

**arXiv ID:** 2603.19272 | [PDF](https://arxiv.org/pdf/2603.19272v1)

**作者:** Bo Tang `[一作]` (Worcester Polytechnic Institute), Weiwei Xie `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5085697317)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

将Transformer层形式化为无状态可微分神经计算机（sDNC），并证明两者在结构和功能上完全等价

**💡 创新点**

首次给出Transformer与sDNC之间的严格数学对应关系，并将跨注意力解释为双内存sDNC

**🔧 技术方法**

使用可微分记忆机制、内容基地址、并行多头注意力的数学推导

**📊 数据集**

无实验数据集，本文为理论研究

**📈 对比分析**

无实验比较，未给出数值性能指标

**⚠️ 局限性**

缺乏对完整DNC动态写/擦除、时间链接等机制的支持，且未验证在实际任务中的有效性

---

## 275. 2K Retrofit: Entropy-Guided Efficient Sparse Refinement for High-Resolution 3D Geometry Prediction

**arXiv ID:** 2603.19964 | [PDF](https://arxiv.org/pdf/2603.19964v1)

**作者:** Tianbao Zhang `[一作]` (Beihang University), Zhaoxin Fan `[通讯]` (Beihang University)

**通讯引用:** 18012 | [OpenAlex ID](https://openalex.org/A5082041657)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了 2K Retrofit 框架，能够在不修改现有 3D 基础模型（如 Depth Anything、VGGT 等）的情况下，使用低分辨率模型快速产生粗略几何预测，并通过熵引导的稀疏细化对高不确定像素进行局部提升，从而获得 2K 级高分辨率几何输出。

**💡 创新点**

创新点在于：① 通过熵量化不确定性，实现对高不确定像素的精准选择；② 使用稀疏特征提取（MinkowskiUNet）和门控融合，仅在必要区域做细化，显著减少计算量；③ 以模块化方式兼容多种基础模型，避免全模型重训练。

**🔧 技术方法**

技术手段包括：冻结低分辨率基础模型、熵选择器、稀疏特征提取器（MinkowskiUNet）、门控融合策略、合成 2K 级数据集、混合损失训练（保持与原模型一致）以及 FP16 推理加速。

**📊 数据集**

使用的数据集：自制 5 万帧 2K 合成数据集、ScanNet++、ARKitScenes、ETH3D、DL3DV、BlendMVS、MVS-Synth 等公开基准。

**📈 对比分析**

在单目 2K 深度估计和多视角 2K 点云重建任务中，与多种 SOTA 方法（Depth Anything、PatchRefiner、PRO、VGGT、MVSFormer++ 等）对比，2K Retrofit 在 AbsRel、RMSE、δ0.5 等指标上通常排名第一或接近最优，同时推理速度提升 3–8 倍，显存和 FLOPs 下降 50% 以上。

**⚠️ 局限性**

局限性：对纹理稀疏、强反射或透明物体等高不确定区域仍易出现误差；稀疏细化虽高效但在全局细节上仍略逊于完整 2K 训练模型。

---

## 276. Computational Complexity Analysis of Interval Methods in Solving Uncertain Nonlinear Systems

**arXiv ID:** 2603.19965 | [PDF](https://arxiv.org/pdf/2603.19965v1)

**作者:** Rudra Prakash `[一作]` (Indian Institute of Technology Delhi), Shaunak Sen `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5070471808)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套算法层面的最坏情况时间与空间复杂度框架，用于评估在不确定非线性系统中验证性区间方法（如区间二分、约束传播、区间牛顿和Krawczyk）的性能。

**💡 创新点**

创新点在于显式阐明了初始搜索体积 Vol(X₀)、目标容差 ε 与验证性原语成本（函数、雅可比矩阵、区间线性代数）之间的关系，并揭示了朴素 Laplace 展开导致的阶乘级别区间行列式/逆矩阵计算复杂度，同时指出 Krawczyk 在实际计算中往往比区间牛顿更快。

**🔧 技术方法**

主要技术包括区间算术、区间扩展函数、区间约束传播、区间牛顿与 Krawczyk 方法、以及对区间矩阵的行列式、伴随矩阵和逆矩阵的复杂度分析；实验使用 Julia、IntervalArithmetic 和 ReversePropagation 包实现。

**📊 数据集**

实验数据集为二维生化反应网络模型 f(x,u) = [0.5+α₁/(1+x₂¹⁰)-γx₁, 0.5+α₂/(1+x₁¹⁰)-γx₂]，参数区间 U=[3.8,4.2]×[3.8,4.2]×[0.95,1.05]，初始搜索箱分别为 V₁=[0,10]² 与 V₂=[0,20]²。

**📈 对比分析**

通过对比固定网格方法（Subdivision+Filter、Constraint Propagation）和自适应方法（Interval Bisection、Interval Newton、Interval Krawczyk），实验显示固定网格方法的工作量主要受分辨率 m 决定；自适应方法对搜索体积增大四倍的 V₂ 时仅略微增加盒子数量和运行时间；Krawczyk 的迭代次数略高但总体性能与 Newton 相当。

**⚠️ 局限性**

局限性包括最坏情况上界往往过于保守，未考虑自适应分裂、预条件化或并行实现；分析仅适用于所列算法与原语，不能直接推广到更复杂或高维问题；朴素区间行列式/逆矩阵的阶乘复杂度使其在实践中不可行。

---

## 277. How Out-of-Equilibrium Phase Transitions can Seed Pattern Formation in Trained Diffusion Models

**arXiv ID:** 2603.20092 | [PDF](https://arxiv.org/pdf/2603.20092v1)

**作者:** Luca Ambrogioni `[一作]` (Radboud University), Luca Ambrogioni `[通讯]` (Radboud University)

**通讯引用:** 858 | [OpenAlex ID](https://openalex.org/A5039391126)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了训练好的扩散模型在逆扩散过程中的非平衡相变行为，提出将架构约束转化为空间软模态的理论框架，并通过可解析的补丁评分模型和真实网络实验验证了临界窗口的存在及其对生成控制的影响。

**💡 创新点**

创新点在于：①将记忆化的不稳定性通过局部性、稀疏性和等变性等网络约束转化为空间集体模态，实现从噪声到结构的临界放大；②把扩散逆过程视作非平衡相变，提出通过估计相关长度和模态软化来定位临界时间，并证明在该窗口投加classifier‑free guidance 可显著提升类对齐。

**🔧 技术方法**

使用的技术包括可解析补丁评分模型、Ginzburg–Landau ϕ⁴ 领域理论、傅里叶模态软化分析、相关长度估计、EDM2/UNet 逆扩散实验以及在 ImageNet 上的临界‑pulse 指导实验。

**📊 数据集**

实验所用数据集包括 FashionMNIST（二值化处理）和 ImageNet（自然图像），以及自定义的二值补丁字典。

**📈 对比分析**

通过比较不同噪声尺度下的相关长度曲线、模态软化行为与生成样本结构变化来评估临界窗口的出现；在 classifier‑free pulse 指导实验中，位于临界窗口的 pulse 在类对齐（DINOv2 分数）上优于随机时间的 pulse，表明该窗口对生成质量具有积极作用。

**⚠️ 局限性**

局限性在于：目前的理论与实验主要在有限尺寸、离散二值数据上验证，缺乏对大规模真实网络在热力学极限下的严格证明；对非二值或高度非对称数据的临界行为仍待进一步探究。

---

## 278. Cov2Pose: Leveraging Spatial Covariance for Direct Manifold-aware 6-DoF Object Pose Estimation

**arXiv ID:** 2603.19961 | [PDF](https://arxiv.org/pdf/2603.19961v1)

**作者:** Nassim Ali Ousalah `[一作]` (University of Luxembourg), Djamila Aouada `[通讯]` (University of Luxembourg)

**通讯引用:** 2782 | [OpenAlex ID](https://openalex.org/A5083368272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种端到端的RGB单目6-DoF姿态估计框架，利用空间协方差池化提取高阶统计，经过SPD网络层压缩后通过可微Cholesky解码得到连续的6D旋转与平移。

**💡 创新点**

① 首次在直接姿态回归中引入空间协方差作为特征，并在SPD流形上执行几何感知降维；② 采用可微Cholesky映射，将SPD矩阵直接映射到连续可微的6D旋转表示与平移，解决了传统四元数/欧拉角不连续的问题；③ 将该框架完全端到端训练，无需PnP或迭代优化，兼具高效与精度。

**🔧 技术方法**

使用EfficientNet‑B6骨干网络；空间协方差池化；SPD流形层（BiMap + ReEig）实现几何感知降维；可微Cholesky分解与Gram-Schmidt正交化得到6D旋转；混合几何优化器（Riemannian + Adam）保证参数在对应流形上更新；数据预处理包括ImageNet预训练权重和PBR渲染图像。

**📊 数据集**

LineMod (LM)、LineMod Occlusion (LM‑O)、YCB‑Video (YCB‑V) 三个BOP基准数据集，覆盖常规、遮挡和大规模真实场景。

**📈 对比分析**

与多种间接PnP方法（PVNet、ZebraPose、VAPO等）和直接回归方法（PoseCNN、Pix2Pose、DeepIM、GDR‑Net等）进行对比；在LM、LM‑O、YCB‑V上均获得最优或次优的ADD/ADD‑S指标（例如LM 100%，LM‑O 97.2%，YCB‑V 82.2% AUC），并在推理速度上仅需46.9 ms，表现出最佳精度与速度平衡。

**⚠️ 局限性**

未显式处理物体对称性，导致对姿态模糊物体的估计受限；相比PnP基准仍略有性能差距；目前仅适用于单目标场景，未扩展到多目标或更复杂环境。

---

## 279. Physics-Informed Neural Network with Adaptive Clustering Learning Mechanism for Information Popularity Prediction

**arXiv ID:** 2603.19599 | [PDF](https://arxiv.org/pdf/2603.19599v1)

**作者:** Guangyin Jin `[一作]` (National Innovative Institute of Defense Technology Academy of Military Science), Witold Pedrycz `[通讯]` (Silesian University of Technology)

**通讯引用:** 91409 | [OpenAlex ID](https://openalex.org/A5003799782)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将宏观物理规律与自适应聚类机制相结合的物理信息神经网络，用于预测信息级联的流行度。

**💡 创新点**

创新点在于首次将Richards曲线作为宏观物理约束嵌入神经网络，并设计无监督自适应聚类网络捕捉信息类别异质性。

**🔧 技术方法**

使用自注意力层进行级联嵌入、TCN实现长时序学习、物理建模网络近似Richards参数以及基于Student’s t分布的聚类机制。

**📊 数据集**

在新浪微博、Twitter和美国物理学会（APS）三个公开数据集上进行实验。

**📈 对比分析**

与Feature-Linear、XGBoost、DeepCas、CasCN、CasFlow、TEDDY、POFHP等十多种基线相比，PIACN在MSLE上提升6.45%–12.64%，在MAPE上提升3%–7%。

**⚠️ 局限性**

局限性包括Richards模型仅能描述单峰普遍增长形态，无法处理多模态传播特征；且缺少真实的内容类别标签，限制了聚类解释力。

---

## 280. Plagiarism or Productivity? Students Moral Disengagement and Behavioral Intentions to Use ChatGPT in Academic Writing

**arXiv ID:** 2603.19549 | [PDF](https://arxiv.org/pdf/2603.19549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 281. SOFTMAP: Sim2Real Soft Robot Forward Modeling via Topological Mesh Alignment and Physics Prior

**arXiv ID:** 2603.19384 | [PDF](https://arxiv.org/pdf/2603.19384v1)

**作者:** Ziyong Ma `[一作]` (Robotics Institute, Carnegie Mellon University), Jean Oh `[通讯]` (Robotics Institute, Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套 sim‑to‑real 学习框架，用于软指尖手臂的实时 3D 前向建模。

**💡 创新点**

创新点在于结合 ARAP 顶点空间对齐、仿真预训练的轻量 MLP、残差校正网络以及闭式线性驱动校准，使得在极少真实数据下即可弥补现实差距。

**🔧 技术方法**

使用了 ARAP 对齐、SOFA 物理仿真、MLP 前向模型、残差网络、Chamfer 损失、线性校准层、光学多视角重建与 MediaPipe 手部识别等技术。

**📊 数据集**

数据集包括 SOFA 生成的数千个四绳软指形变样本，以及使用两台机器人臂和两台相机在多视角下采集的真实指尖点云。

**📈 对比分析**

与 DeepSoRo、Laplacian、线性、XGBoost 等基线比较，仿真 Chamfer 距离 0.389 mm、实测 3.786 mm；指尖轨迹跟踪误差降低约 30%；遥操作 IoU 提升 36.5%。

**⚠️ 局限性**

局限在于需要固定指尖结构、每次硬件更换需重新采集并训练残差网络；未考虑接触力与多指交互。

---

## 282. PanORama: Multiview Consistent Panoptic Segmentation in Operating Rooms

**arXiv ID:** 2603.19920 | [PDF](https://arxiv.org/pdf/2603.19920v1)

**作者:** Tuna Gürbüz `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**通讯引用:** 55804 | [OpenAlex ID](https://openalex.org/A5046896448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向手术室的多视角一致全景分割模型 PanORama。

**💡 创新点**

创新点在于通过在特征层面进行局部与全局注意力的跨视角交互，直接在单次前向传播中实现视角一致性；且模型无需摄像机标定，支持任意数量的同步视角。

**🔧 技术方法**

采用共享视觉骨干（预训练的 DINO）、VGGT 风格的多视角聚合模块（融合局部与全局注意力），以及分层上采样的全景分割头。

**📊 数据集**

使用公开的 MM-OR 与 4D-OR 两个手术室多视角全景分割数据集进行训练与评估。

**📈 对比分析**

与单视角基线 DVIS++ 等方法相比，PanORama 在 MM-OR 上单帧 PQ 提升至 71.85%（提升 > 7%），在 4D-OR 上也保持领先；并能在未见摄像机视角上获得更高的 PQ，展示出更好的泛化能力。

**⚠️ 局限性**

局限在于仅对单帧进行预测，缺乏显式时间一致性约束；对极端遮挡和极少标注的视角仍有挑战，未来可考虑引入时序建模和更鲁棒的跨视角学习策略。

---

## 283. Significance-Gain Pair Encoding for LLMs: A Statistical Alternative to Frequency-Based Subword Merging

**arXiv ID:** 2603.19261 | [PDF](https://arxiv.org/pdf/2603.19261v1)

**作者:** Azam Nouri `[一作]` `[通讯]` (Lincoln University), Azam Nouri (Lincoln University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Significance-Gain BPE作为BPE的替代合并策略

**💡 创新点**

将基于独立假设的统计连贯性(z-score)与压缩收益结合，用于合并选择

**🔧 技术方法**

实现greedy合并并训练小型因果Transformer，计算z-score与压缩收益

**📊 数据集**

在WikiText-103（字符切片）上进行评估

**📈 对比分析**

通过token-依赖的PPL和token-不依赖的BPC比较，Significance-Gain在BPC上提升约0.9-1.0%，PPL降低约13%

**⚠️ 局限性**

仅单次实验，未报告多种随机种子；仅考虑邻接关联，忽略更长距离结构

---

## 284. HiPath: Hierarchical Vision-Language Alignment for Structured Pathology Report Prediction

**arXiv ID:** 2603.19957 | [PDF](https://arxiv.org/pdf/2603.19957v1)

**作者:** Ruicheng Yuan `[一作]` (Hunan University), Guang Yang `[通讯]` (Imperial College London)

**通讯引用:** 17917 | [OpenAlex ID](https://openalex.org/A5108053324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级视觉‑语言模型 HiPath，专门针对病理报告的结构化槽位预测任务。

**💡 创新点**

创新点包括：① 层级补丁聚合 (HiPA) 处理多 ROI 图像；② 层级对比学习 (HiCL) 通过最优传输实现图像‑文本跨模态对齐；③ 基于词典的槽位掩码预测 (Slot‑MDP) 直接生成结构化诊断。

**🔧 技术方法**

采用冻结的 UNI2 视觉骨干和 Qwen3 文本骨干，配合 15M 可训练参数实现上述三模块，使用信息熵正则化的 Sinkhorn OT 与跨模态 InfoNCE 损失。

**📊 数据集**

使用 749K 真实中文病理案例，来自三家医院（总计 42 个器官位点），构建 304 词典并定义安全边界。

**📈 对比分析**

与多种基线（线性探针、ABMIL、Flat CrossAttn 等）对比，HiPath 在严格匹配 68.9% / 临床可接受 74.7% / 安全率 97.3% 方面均显著优于基线，跨院验证仅下降 3.4%。

**⚠️ 局限性**

局限性：仅针对手选 ROI 而非全切片；未与生成式基线对比；对词典外诊断只能最近邻近似；多中心外部验证仍待扩展。

---

## 285. FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts

**arXiv ID:** 2603.19857 | [PDF](https://arxiv.org/pdf/2603.19857v1)

**作者:** You Li `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 81548 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于DiT的可控制视频到音频(V2A)框架，利用结构化时间脚本(Structured Temporal Scripts, STS)和脚本引导的时间融合模块(SG‑TFM)实现精细的时间控制，并通过双帧声音合成(Bi‑Frame Sound Synthesis)支持前景与背景声音的并行生成。

**💡 创新点**

创新点在于：①首次将短时段文字脚本引入V2A实现多事件细粒度时间控制；②设计的SG‑TFM采用时间脚本注意力与交错RoPE实现语义与时间的协同融合；③提出双帧合成框架实现可视与非可视声音的并行渲染，提升对复杂多事件场景的可控性。

**🔧 技术方法**

核心技术包括DiT‑based V2A模型（基于MMAudio）、CLIP文本编码、Temporal Script Attention、Interleaved RoPE、Bi‑Frame Sound Synthesis、以及对齐与分布匹配评估指标（FD, KL, ISC, IB, DeSync）。

**📊 数据集**

使用了自构建的DirectorSound数据集（融合VGGSound、AudioCaps及自家数据），并构造了两个基准：DirectorBench（控制性评测）和VGGSound‑Director（音频质量与分布评测）。

**📈 对比分析**

与MMAudio、HunyuanVideo‑Foley、ThinkSound、Video‑Foley等SOTA V2A方法对比，在DirectorBench上控制性F1从0.2451提升至0.4819，Temporal和Counterfactual子集均显著改善；在VGGSound‑Director上保持或提升音频质量指标（FD_VGG降至1.17，ISC提升至14.84）。

**⚠️ 局限性**

局限性包括：①需要人工或自动生成的短时段脚本，对脚本长度和准确性敏感；②在极短时段或复杂视觉场景中仍可能出现时间对齐误差；③模型仍依赖大规模预训练的DiT结构，对资源要求较高。

---

## 286. VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models

**arXiv ID:** 2603.19370 | [PDF](https://arxiv.org/pdf/2603.19370v1)

**作者:** Zirui Ge `[一作]` (Zhejiang University), Donglin Wang `[通讯]` (Westlake University)

**通讯引用:** 1551 | [OpenAlex ID](https://openalex.org/A5100665183)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种后训练框架，利用强化学习奖励直接优化视频动作模型的视觉动态，从而提升下游动作生成与机器人操控的性能。

**💡 创新点**

创新点包括：①将多步去噪视为马尔可夫决策过程；②设计了只在首步使用SDE、其余步骤使用ODE的Euler Hybrid采样；③使用GRPO在潜在空间中基于可验证奖励（L1+余弦相似）对模型进行直接优化；④无架构改动，仅通过后训练显著提升基准模型。

**🔧 技术方法**

核心技术包括：Stable Video Diffusion、Euler-Ancestral采样、Euler Hybrid采样、GRPO、潜在空间奖励、VAE编码、Diffusion Policy 以及与DDPO的对比。

**📊 数据集**

实验数据集涵盖：CALVIN（ABC→D 与 L-CALVIN）模拟环境，Agibot Genie 01 真实机器人平台，并使用 VPP 预训练检查点进行后训练。

**📈 对比分析**

与多种 VLM/VPM 方案（如 OpenVLA、π_0、VPP、Seer 等）在 CALVIN ABC→D、L-CALVIN 以及真实任务上进行对比，实验显示本文方法在任务完成率、平均轨迹长度等指标上均显著优于当前最先进方法。

**⚠️ 局限性**

限制包括：奖励仅在终点，可能导致信用分配不足；需进一步与策略共训练以充分发挥潜力；对极长序列或高维视觉信息的适配仍需验证；以及在不同任务和硬件上可能存在的泛化与鲁棒性挑战。

---

## 287. MOSAIC: Modular Opinion Summarization using Aspect Identification and Clustering

**arXiv ID:** 2603.19277 | [PDF](https://arxiv.org/pdf/2603.19277v1)

**作者:** Piyush Kumar Singh `[一作]` (Viator), Jayesh Choudhari `[通讯]` (Viator)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可工业化、可解释的分解式评论摘要框架MOSAIC

**💡 创新点**

通过主题发现、主题约束意见抽取、意见聚类与结构化摘要，显著提升事实性与可解释性

**🔧 技术方法**

采用多种LLM（GPT-4o、GPT-4.1、Llama-3.1‑70B）、BERT嵌入、二元验证、HDBSCAN聚类等技术实现分解与聚合

**📊 数据集**

使用公开数据集SPACE、PeerSum以及新发布的旅游评论集TRECS（344条产品、140k条评论）进行实验

**📈 对比分析**

在在线A/B实验和离线评测中，MOSAIC在主题覆盖率（≈0.99）、G‑Eval和AlignScore上均超过SOTA，并提升用户参与度和收入

**⚠️ 局限性**

局限包括对LLM能力的高度依赖、需要域特定提示与聚类超参数调优，以及在长度受限时难以平衡常见与罕见观点

---

## 288. Speculative Policy Orchestration: A Latency-Resilient Framework for Cloud-Robotic Manipulation

**arXiv ID:** 2603.19418 | [PDF](https://arxiv.org/pdf/2603.19418v1)

**作者:** Chanh Nguyen `[一作]` (Umeå University), Erik Elmroth `[通讯]` (Umeå University)

**通讯引用:** 6795 | [OpenAlex ID](https://openalex.org/A5070862414)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种云‑边缘协同框架 Speculative Policy Orchestration (SPO)，通过云端预计算连续运动轨迹并在边缘本地验证，以消除高频连续控制的网络延迟影响。

**💡 创新点**

提出基于ϵ‑tube安全验证的连续状态执行检查以及自适应 Horizon Scaling 机制，动态调整预取深度，既保证安全又实现低延迟的云端轨迹推理。

**🔧 技术方法**

采用前向自回归世界模型与策略推理、边缘端ϵ‑管道验证、加/减法自适应增减法（AHS）、RLBench 仿真与 ZeroMQ 通信等技术。

**📊 数据集**

在 RLBench 基准任务（StackBlocks、InsertOntoSquarePeg、PutAllGroceriesInCupboard）上进行实验，并使用基于专家演示的 Oracle 模型与训练的三层 MLP 预测模型。

**📈 对比分析**

与同步远程推理、单步预取和 10 步静态预取三种基线对比，SPO 在高延迟下成功率提升至 100%、累计空闲时间下降 60%、缓存命中率与计算浪费均显著优于静态方案。

**⚠️ 局限性**

依赖准确的世界模型和 ϵ‑tube 阈值设定，对真实硬件的机械冲击和感知噪声未充分验证；在极高动态或未知环境中预取深度仍受限。

---

## 289. Robust Beam Codebooks for mmWave/THz Systems: Toward a Stochastic RL Approach

**arXiv ID:** 2603.19930 | [PDF](https://arxiv.org/pdf/2603.19930v1)

**作者:** Anouar Nechi `[一作]` (Institute of Computer Engineering), Saleh Mulhem `[通讯]` (Institute of Computer Engineering)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了一种基于多智能体强化学习的mmWave/THz波束码本设计方法，在不需要CSI的情况下直接从环境反馈学习鲁棒波束。

**💡 创新点**

提出了针对硬件失真和反馈噪声的鲁棒评估框架，并证明SAC随机策略优于确定性算法。

**🔧 技术方法**

采用多智能体Deep Deterministic Policy Gradient、Twin Delayed DDPG和Soft Actor-Critic等离线策略，并使用KNN量化、三元奖励与自适应探索。

**📊 数据集**

在DeepMIMO仿真数据集上构建LoS、NLoS及带硬件失真的三种评测场景。

**📈 对比分析**

通过在不同码本尺寸、相位失配、反馈噪声水平下测量平均波束增益进行比较，SAC始终取得最高增益并在40%噪声下保持约78%性能。

**⚠️ 局限性**

仍未考虑多RF链、频率选择性衰落以及非线性硬件效应，实验仅在理想化仿真环境中验证。

---

## 290. TSegAgent: Zero-Shot Tooth Segmentation via Geometry-Aware Vision-Language Agents

**arXiv ID:** 2603.19684 | [PDF](https://arxiv.org/pdf/2603.19684v1)

**作者:** Shaojie Zhuang `[一作]` (Shandong University), Yuanfeng Zhou `[通讯]` (Shandong University)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5000250608)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出TSegAgent框架，实现零样本牙齿分割与识别，利用多视角渲染、SAM3实例分割以及基于几何约束的视觉语言推理；

**💡 创新点**

将牙齿分析重新定义为几何推理任务，融合牙弓先验与三维几何约束，采用多轮对话式视觉语言推理进行错误检测与自纠正，且无需任务特定训练；

**🔧 技术方法**

使用SAM3进行零样本实例分割，ChatGPT等视觉语言模型进行多轮推理，结合多视角渲染、曲率热图、三维包围盒、牙弓曲线拟合及对称性约束；

**📊 数据集**

在公开的Teeth3DS数据集（1200个3D模型）和私有诊所收集的340个3D模型上进行评估；

**📈 对比分析**

与MeshSegNet、PTv3、TSGCNet、DBGANet、CrossTooth、3DTeethSAM等基线对比，在Teeth3DS上实现mIoU 93.37%、TIR 96.76%、TIR=1 87.17%，在私有数据集上mIoU 82.10%、TIR 96.68%、TIR=1 51.96%，表现出色并具备跨域泛化；

**⚠️ 局限性**

依赖多视角渲染与SAM3的分割质量，对极端遮挡或缺失牙冠的情况仍可能出现误分；目前验证仅限牙齿领域，推广到其他解剖结构仍需进一步研究。

---

## 291. ICLAD: In-Context Learning for Unified Tabular Anomaly Detection Across Supervision Regimes

**arXiv ID:** 2603.19497 | [PDF](https://arxiv.org/pdf/2603.19497v1)

**作者:** Jack Yi Wei `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于上下文学习的表格异常检测基础模型ICLAD，能够在无监督、单类监督与半监督三种监督范式下统一工作。

**💡 创新点**

创新点：①将三种监督范式融为一体，构建可在不同监督程度下自适应的异常评分函数；②通过元学习在多样化的合成任务分布上训练Transformer，使模型获得表格数据特有的归纳偏置；③在任务生成中结合结构性与扰动式异常，提升模型对开放式异常的泛化能力。

**🔧 技术方法**

核心技术包括：Transformer-PFN架构、FiLM标签条件化、结构因果模型（SCM）生成合成数据、结构与扰动异常生成、元学习（BCE损失）训练、KV缓存实现两阶段推理。

**📊 数据集**

使用Ablench基准中的57个真实表格数据集进行评估，数据维度从3到1555，异常比例从3%到39%。

**📈 对比分析**

与经典基线（CBLOF、ECOD、iForest等）和深度学习基线（AutoEncoder、Deep-SVDD、LUNAR等）以及半监督基线（DevNet、DeepSAD等）对比，ICLAD在一类与无监督设置中均取得最高或竞争性的AUC-ROC，并在半监督设置下随着少量异常标签提升约10% AUC-ROC，显示出强大的跨范式鲁棒性。

**⚠️ 局限性**

局限性：①Transformer自注意力导致推理时间相对较长；②训练需耗费数十小时并使用数百万合成任务；③对高维或大规模数据仍需通过采样/子集化处理；④合成任务分布的设计仍可能无法覆盖所有真实异常场景。

---

## 292. FlowScene: Style-Consistent Indoor Scene Generation with Multimodal Graph Rectified Flow

**arXiv ID:** 2603.19598 | [PDF](https://arxiv.org/pdf/2603.19598v1)

**作者:** Zhifei Yang `[一作]` (Peking University), Yikai Wang `[通讯]` (Beijing Normal University)

**通讯引用:** 1253 | [OpenAlex ID](https://openalex.org/A5100747434)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了FlowScene，一种基于多模态图的三分支生成器，能够生成室内场景的布局、物体形状和纹理，并保持整体风格一致性。

**💡 创新点**

创新点在于提出多模态图正则流（Multimodal Graph Rectified Flow）实现节点间信息交换，提升纹理一致性和形状质量；三分支结构与紧耦合流模型相结合，实现高保真度与高效率的兼顾。

**🔧 技术方法**

使用了多模态图卷积网络、图正则流、VQ‑VAE、稀疏流变压器、CLIP/DINOv2特征预训练等技术，并通过ODE采样实现少步生成。

**📊 数据集**

使用SG‑FRONT和3D‑FRONT数据集，包含约45K物体实例、15类关系，涵盖卧室、餐厅和客厅等室内场景。

**📈 对比分析**

与无训练语言检索方法Holodeck、LayoutVLM以及图条件生成器CommonScenes、EchoScene、MMGDreamer进行对比，采用FID、FID_CLIP、KID、MMD、COV、1‑NNA、CLIPScore、FPVScore和风格一致性评分等指标；FlowScene在场景与对象真实性、可控性、风格一致性上均优于基线，并且推理速度提升约84%。

**⚠️ 局限性**

局限性在于仅在室内小规模数据集上验证，缺乏对大型或户外场景的扩展性评估；对极端视角或复杂关系的处理仍有待改进。

---

## 293. Not an Obstacle for Dog, but a Hazard for Human: A Co-Ego Navigation System for Guide Dog Robots

**arXiv ID:** 2603.20121 | [PDF](https://arxiv.org/pdf/2603.20121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 294. Diminishing Returns in Expanding Generative Models and Godel-Tarski-Lob Limits

**arXiv ID:** 2603.19687 | [PDF](https://arxiv.org/pdf/2603.19687v1)

**作者:** Angshul Majumdar `[一作]` (Indraprastha Institute of Information Technology), Angshul Majumdar `[通讯]` (Indraprastha Institute of Information Technology)

**通讯引用:** 6273 | [OpenAlex ID](https://openalex.org/A5020310463)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

提出一个任务空间框架，用来分析扩展式生成推理系统在长期内的能力演化，并证明在合理假设下，系统扩展时新增可解决任务的概率质量必然趋于零；在预测任务中给出基于Kolmogorov复杂度的量化递减率；同时结合罗塞尔不完全性、塔尔斯基不可定义性和洛布定理，阐明在足够表达的内部推理系统中仍存在不可解的逻辑任务。

**💡 创新点**

创新点在于：
1) 构建了一个抽象的、与具体架构无关的任务空间与能力度量；
2) 证明了在仅需单调、可保持能力的扩展假设下，边际改进必然衰减到零；
3) 在预测子空间中使用算法信息论的复杂度加权先验给出具体的收敛上界；
4) 将经典逻辑不完全性与生成推理系统结合，揭示了内部推理的固有限制。

**🔧 技术方法**

使用的技术主要包括：测度论与集合论的基本工具；算法信息论（Kolmogorov复杂度、算法概率）；逻辑与数理哲学中的不完全性、不可定义性和反射定理；以及经典的递归可枚举理论与证明可算性分析。

**📊 数据集**

本研究为纯理论工作，并未使用具体的数据集；所有结论均基于数学证明而非实验验证。

**📈 对比分析**

由于是理论推导，没有实验比较或性能指标；评估方式是通过数学证明展示边际改进趋于零和量化收敛速率，且指出逻辑任务无法在内部被完全解决。

**⚠️ 局限性**

局限性包括：
1) 假设任务分布固定且不随系统演化；
2) 只考虑能够保持已有解的结构性扩展，未涵盖更一般的学习动态；
3) 对具体模型（如Transformer、GAN、Diffusion等）的可扩展性和实际表现缺乏经验验证；
4) 逻辑限制仅适用于足够表达的形式系统，实际应用中可能通过外部工具绕过。

---

## 295. Improving Generalization on Cybersecurity Tasks with Multi-Modal Contrastive Learning

**arXiv ID:** 2603.20181 | [PDF](https://arxiv.org/pdf/2603.20181v1)

**作者:** Jianan Huang `[一作]` (Politecnico di Torino), Dario Rossi `[通讯]` (Huawei Paris Research Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出两阶段多模态对比学习框架，通过先在漏洞文本描述上构建语义嵌入空间，再将HTTP payload映射到该空间，实现对网络安全任务的分类与泛化。

**💡 创新点**

创新点在于将文本描述的对比学习结构迁移到数据稀缺的payload域，既缓解了shortcut学习，又为零样本推理提供了语义检索机制。

**🔧 技术方法**

技术包括预训练T5‑based语言模型、triplet loss对比学习、冻结文本编码器的模态对齐（MSE）、语义检索推理以及t‑SNE可视化。

**📊 数据集**

使用了Huawei私有威胁情报数据库（约29k文本描述、601k payload）以及基于公开CVE描述与LLM生成的11k样本的合成基准。

**📈 对比分析**

与TF‑IDF+RF、Fine‑tuned CodeBERT+MLP、Embedding Similarity三种基线对比，在私有数据的时间漂移测试中准确率从最佳基线65.7%提升至68.1%，在合成基准上提升至24.4%（最高基线20.6%）。

**⚠️ 局限性**

局限性包括仅在单一任务上验证、整体准确率仍远低于生产要求、类不平衡与类别定义混合导致难以完全泛化，以及对完全新类别的零样本迁移能力尚未充分验证。

---

## 296. Enhancing Alignment for Unified Multimodal Models via Semantically-Grounded Supervision

**arXiv ID:** 2603.19807 | [PDF](https://arxiv.org/pdf/2603.19807v1)

**作者:** Jiyeong Kim `[一作]` (Ewha Womans University), Dongbo Min `[通讯]` (Ewha Womans University)

**通讯引用:** 2736 | [OpenAlex ID](https://openalex.org/A5037006973)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为SeGroS的语义对齐微调框架，用于统一多模态模型（UMM）中的视觉-文本对齐。

**💡 创新点**

创新点在于通过构建视觉-语义对齐图，实现了视觉提示和语义根植的破损输入，精准地把监督聚焦于文本对应的核心视觉区域，解决了粒度不匹配和冗余监督问题。

**🔧 技术方法**

技术主要包括文本内在相似度与文本-图像交互相似度的双重筛选，视觉对齐分数图的构造，及基于该分数进行视觉提示选择与自适应掩码的生成。

**📊 数据集**

使用了MidjourneyV6和BLIP3o-60k等高质量文本-图像对进行微调，并在GenEval、DPGBench、CompBench等评测集上评估。

**📈 对比分析**

与标准SFT和Reca方法对比，SeGroS在GenEval、DPGBench和CompBench上均取得更高分数，尤其在位置与属性子项上提升显著，证明了跨模态对齐的改进。

**⚠️ 局限性**

局限性在于对视觉对齐图的构造仍依赖预训练嵌入，可能在极度复杂场景下误判重要区域；同时该方法在大型模型上需要额外计算开销。

---

## 297. The Voronoi Diagram of Four Lines in $\mathbb{R}^3$

**arXiv ID:** 2603.19836 | [PDF](https://arxiv.org/pdf/2603.19836v1)

**作者:** Evanthia Papadopoulou `[一作]` (Università della Svizzera italiana), Zeyu Wang `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5115596280)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了四条在三维欧氏空间中的一般位置直线的最近与最远Voronoi图的结构，并给出了完整的拓扑分类；

**💡 创新点**

首次证明了四条直线Voronoi图顶点数始终为偶数、范围0~8，并定义了“twist”这一关键结构，区分全twist与半twist，从而完成了无全twist时的15种拓扑以及全twist的局部插入/移除操作；

**🔧 技术方法**

采用几何分析与组合学观察相结合的全枚举搜索算法，利用剖面投影、三分线的分支与渐近线、并行性约束等过滤条件，最终实现了15个可实现的配置六元组；

**📊 数据集**

论文主要为理论性工作，未使用公开数据集，所有结果均通过符号推导与计算机程序验证；

**📈 对比分析**

对比方法主要是与已有的三条直线Voronoi图结果和已知复杂度边界相联系，证明了顶点数、边数、面数与已知下界上限相符，性能方面以理论复杂度（如O(n³) 计算）为基准；

**⚠️ 局限性**

局限性在于仅处理了四条直线的基例，未直接推广到任意多条直线；全twist的局部插入仍需手工验证，且枚举算法在大规模扩展时计算量急剧增加。

---

## 298. Stepwise: Neuro-Symbolic Proof Search for Automated Systems Verification

**arXiv ID:** 2603.19715 | [PDF](https://arxiv.org/pdf/2603.19715v1)

**作者:** Baoding He `[一作]` (Nanjing University), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14365 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经-符号框架，通过最佳优先树搜索结合微调的大语言模型（LLM）自动生成系统级软件验证的证明步骤；

**💡 创新点**

创新点在于将LLM的步骤生成与符号工具（如QuickCheck、Nitpick、Sledgehammer）紧密集成，形成可修订、过滤、排名的搜索管道，并通过迭代生成训练数据提升模型适应性；

**🔧 技术方法**

技术包括：细粒度的LLM步骤生成与修订、符号检验（快速测试、模型检验）、累计对数概率评分、Sledgehammer后端求解；

**📊 数据集**

使用seL4的FVEL数据集（约29k条定理）进行模型微调，并在AFP、Code2Inv等额外项目上进行跨域评估；

**📈 对比分析**

与Selene、FVEL、Auto、Sledgehammer等基线对比，Mistral‑7B+框架在seL4上证明率达到77.6%，比最佳基线高37.3%；在多步骤证明上保持显著优势；

**⚠️ 局限性**

局限性包括：搜索过程计算开销大、对较长证明的性能下降、步骤修订依赖手工规则、对新领域的可迁移性尚需进一步验证。

---

## 299. Helix: A Dual-Helix Co-Evolutionary Multi-Agent System for Prompt Optimization and Question Reformulation

**arXiv ID:** 2603.19732 | [PDF](https://arxiv.org/pdf/2603.19732v1)

**作者:** Kewen Zhu `[一作]` (Tianjin University), Qinghua Hu `[通讯]` (Tianjin University)

**通讯引用:** 35524 | [OpenAlex ID](https://openalex.org/A5056686459)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双螺旋共进化多代理框架Helix，联合优化用户提问的表述与大语言模型的prompt，从而提升推理准确率。

**💡 创新点**

创新点在于将问题重写与prompt优化视为耦合的双向进化问题，利用Planner进行任务分解，Prompt‑Architect和Question‑Architect双向辩论并通过Mediator校验，最终在推理阶段采用策略驱动的生成‑评估机制。

**🔧 技术方法**

使用GPT‑4o（或Qwen2.5‑32B‑Instruct）作为LLM，构建六个专门化代理（Planner、Prompt‑Architect、Question‑Architect、Mediator、Generator、Judge），实现规划、共进化、策略生成与多维度验证。

**📊 数据集**

在12个基准数据集上评估：BBH、MMLU、MMLU‑Pro、AGIEval LSAT‑AR、AQuA‑RAT 等，涵盖推理、知识问答、数学等多领域任务。

**📈 对比分析**

与手工prompt、CoT、APE、OPRO、PE2、MARS等方法对比，Helix在所有任务上平均准确率达到80.36%，比MARS高3.95%，比CoT高7.20%；在1‑shot、2‑shot等低样本场景仍能优于基线，且优化效率（Prompt Efficiency）明显更高，减少约45% LLM调用量。

**⚠️ 局限性**

局限性包括：对大型LLM算力高度依赖，低算力或小模型的适配未充分验证；规划与辩论的复杂性导致实现难度和推理延迟；对极低样本或多语言、跨域任务的鲁棒性仍待进一步研究。

---

## 300. OmniDiT: Extending Diffusion Transformer to Omni-VTON Framework

**arXiv ID:** 2603.19643 | [PDF](https://arxiv.org/pdf/2603.19643v1)

**作者:** Weixuan Zeng `[一作]` (Chinese University of Hong Kong), Tingting Gao `[通讯]` (KuaiShou)

**通讯引用:** 12682 | [OpenAlex ID](https://openalex.org/A5112381636)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散变换器（Diffusion Transformer）的统一虚拟试穿/试脱框架 OmniDiT，能够同时完成模型基、模型自由试穿以及试脱任务。

**💡 创新点**

创新点包括：① 通过 token 级拼接与自适应位置编码，将多源参考条件（服装图像、人体图像或试穿结果）统一注入模型；② 首次将 Shifted Window Attention 引入扩散模型，降低自注意力计算复杂度；③ 采用多时钟预测（Multi‑Timestep Prediction）与对齐损失（Alignment Loss）提升生成的稳定性与服装细节保真度；④ 设计自演进式数据标注管线并构建 38 万张高质量三元组数据集 Omni‑TryOn。

**🔧 技术方法**

使用的核心技术包括：Diffusion Transformer (DiT) + Flow Matching；token 拼接与自适应 RoPE；Shifted Window Attention；多时钟预测；对齐损失；LoRA 微调；VLM（如 Qwen3‑VL、InternVL‑3.5）辅助数据筛选与文本生成。

**📊 数据集**

主要数据集：公开的 VITON‑HD（13,679 对）与 DressCode（53,792 对）；自建 Omni‑TryOn（380,000+ 3‑元组样本）；此外还使用 Omni‑TryOn 进行跨域泛化测试。

**📈 对比分析**

在 VITON‑HD、DressCode 以及 Omni‑TryOn 上的量化评估显示：在模型基试穿任务中，KID、SSIM 远超同类方法，整体指标几乎达到或超过 SOTA；在模型自由试穿和试脱任务中，FID、KID、CLIP‑I、DINO‑I 等指标均优于目前主流方法，尤其在复杂场景与姿态下保持了较高的细节保真度。

**⚠️ 局限性**

主要局限：对人体姿态与手势的生成不够精确，原因是自演进式数据标注管线中 VLM 对人像差异的筛选能力有限，导致训练时混淆了目标人像特征。未来工作需要更强的 VLM 与更严格的数据过滤机制。

---

## 301. Wireless Broadcast Gossip for Decentralized Drone Swarms: Success Probability, Contraction, and Optimal Aloha

**arXiv ID:** 2603.19379 | [PDF](https://arxiv.org/pdf/2603.19379v1)

**作者:** Ali Khalesi `[一作]` `[通讯]` (Institut Polytechnique des Sciences Avancées), Ali Khalesi (Institut Polytechnique des Sciences Avancées)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在干扰受限的无线介质上使用广播 gossip 的无人机编队协同问题，给出了接收成功概率、收敛收缩上界以及最优随机访问概率的闭式表达式；

**💡 创新点**

创新点在于将无线层的干扰可靠性与图论层的平均化算法紧密耦合，推导出可直接用于 MAC 层调优的显式访问概率 p*（并证明其随密度 λ 的 Θ(1/λ) 缩放），以及提供了因子化的收敛收缩分析；

**🔧 技术方法**

主要技术包括：泊松点过程（PPP）空间建模、时隙 Aloha 访问、Rayleigh 衰落与功率衰减模型下的 SIR 成功概率推导、匹配抽象的平均化 gossip 迭代、以及基于期望收敛的 mean‑square 收缩 bound；

**📊 数据集**

未使用真实数据集，所有实验均基于仿真：在大面积平面内随机生成泊松分布的无人机，仿真其广播 gossip 行为并测量收敛速率；

**📈 对比分析**

与理论预期对比：仿真结果显示收敛速率对 p 的曲线呈单峰，且曲线最低点几乎与理论预测的 p* 重合，说明闭式调优规则在实践中能有效加速协同；

**⚠️ 局限性**

局限性包括：仅考虑静态泊松分布（未考虑无人机运动带来的时间多样性）、忽略热噪声、半双工约束简化、以及仅在 Rayleigh 衰落下分析，未来工作需扩展到移动模型和更一般的信道环境。

---

## 302. LoFi: Location-Aware Fine-Grained Representation Learning for Chest X-ray

**arXiv ID:** 2603.19451 | [PDF](https://arxiv.org/pdf/2603.19451v1)

**作者:** Myeongkyun Kang `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (Vector Institute)

**通讯引用:** 5162 | [OpenAlex ID](https://openalex.org/A5100458648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 LoFi，联合使用 sigmoid、captioning 及位置感知 captioning 损失，并结合轻量级 LLM 训练细粒度的胸片表征，再将该表征嵌入检索式上下文学习中实现定位。

**💡 创新点**

创新点包括：① 在对比学习中加入位置感知（定位与密集 captioning）损失以实现区域级监督；② 通过轻量级 LLM 实现文本生成与定位的联合优化；③ 将细粒度编码器与检索式 ICL 结合，提升跨域定位性能。

**🔧 技术方法**

使用 SigLIP2 作为图像/文本编码器，Gemma‑3‑270M 作为 LLM，LoRA 微调，Sigmoid、Captioning、Grounding 与 Dense Captioning 损失，检索式 ICL 采用 MedGemma‑1.5‑4B，cosine 相似度进行检索。

**📊 数据集**

主要数据集为 MIMIC‑CXR（报告与 394k 区域标注）和 PadChest‑GR（4,555 张胸片与定位标注），以及 MIMIC‑Ext 作为区域标注来源。

**📈 对比分析**

与 SigLIP2、BiomedCLIP、BMC‑CLIP、MedSigLIP、CARZero、RadIR 等对比，LoFi 在 MIMIC‑CXR 上的 R@1 与 R@40 明显领先；在 PadChest‑GR 上的外部 ICL 和内部微调中，LoFi 取得最高的 Ro/L 与 F@0.5 分数，优于 MedRPG、ChEX、GEMeX、K2Sight 等基线。

**⚠️ 局限性**

主要局限在于检索式 ICL 的计算开销较大，限制了规模化部署；此外，区域标注稀缺与跨机构差异仍可能影响细粒度学习效果。

---

## 303. CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing

**arXiv ID:** 2603.19297 | [PDF](https://arxiv.org/pdf/2603.19297v1)

**作者:** Manit Baser `[一作]` (National University of Singapore), Mohan Gurusamy `[通讯]` (National University of Singapore)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5080394785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的表示层技术，用于识别大型语言模型（LLMs）中模型编辑可能引发的涟漪效应。

**💡 创新点**

创新点在于通过单个中间层的前向激活量化事实之间的纠缠，避免了计算成本高昂的反向传播，提供了一种更高效的评估方法。

**🔧 技术方法**

使用了一种轻量级的表示层技术，称为（Critical Layer Representation Entanglement），通过前向激活计算纠缠图。

**📊 数据集**

使用了来自三个现有数据集的11,427个事实，构建了一个包含212种独特提示格式和6,140个独特主题的语料库。

**📈 对比分析**

与基线方法相比，该方法在Spearman相关性上平均提高了62.2%，速度快2.74倍，峰值GPU内存使用量减少2.85倍。

**⚠️ 局限性**

限制在于该方法仍然是相关性分析，未建立表示纠缠与模型泛化或退化之间的正式因果机制，且目前作为诊断工具而非编辑策略。

---

## 304. From Plausibility to Verifiability: Risk-Controlled Generative OCR for Vision-Language Models

**arXiv ID:** 2603.19790 | [PDF](https://arxiv.org/pdf/2603.19790v1)

**作者:** Weile Gong `[一作]` (Nanjing University of Posts and Telecommunications), Chen Dai `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 40380 | [OpenAlex ID](https://openalex.org/A5108049733)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Geometric Risk Controller（GRC），将冻结的视觉语言模型在OCR任务中引入多视角外部证据、轻量级结构筛查与跨视角一致性投票，形成固定协议下可审计的接受/放弃决策，显著降低极端错误风险；

**💡 创新点**

将冻结VLM的生成式OCR视为部署控制问题，首次引入外部可审计的风险控制层（多视角几何探测+一致性判定）和可调严格度参数，提供系统化的风险–覆盖权衡；

**🔧 技术方法**

采用多视角几何探测、字符串标准化、轻量级长度约束筛查、跨视角投票一致性与稳定性阈值控制，所有技术均在推理时外部实现，无需改动模型内部；

**📊 数据集**

在冻结的LLaVA‑Phi3、Gemma3、GLM‑OCR三种模型上，使用场景文本基准IIIT5K和ICDAR 2013测试集进行评估；

**📈 对比分析**

与总是接受的基线及基于内部置信度阈值的外部选择性基线比较，在相同覆盖率下，GRC显著降低平均CER、P99和灾难性暴露率（Meltdown@2），尤其在LLaVA‑Phi3上提升最大；

**⚠️ 局限性**

主要局限在于仍无法识别跨视角一致但错误的情况（stable‑but‑wrong），且当前仅针对单词级场景文本；未来需扩展到区域级验证和更严格的几何约束。

---

## 305. CDEoH: Category-Driven Automatic Algorithm Design With Large Language Models

**arXiv ID:** 2603.19284 | [PDF](https://arxiv.org/pdf/2603.19284v1)

**作者:** Yu-Nian Wang `[一作]` (Hohai University), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39572 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在大语言模型驱动的进化式自动启发式设计中引入算法类别的框架CDEoH，显式维护类别信息以提升演化稳定性和多样性。

**💡 创新点**

创新点在于：①算法类别诱导与记录，②基于类别的两阶段选择策略（类别精英+性能-多样性联合得分），③错误修复的反射机制；这些机制在传统EoH与FunSearch的基础上实现更高的多样性与稳定性。

**🔧 技术方法**

技术包括：大语言模型（DeepSeek-Chat）进行思路生成、代码生成与类别分类；进化算子（改进提示与创新提示）；性能评估函数；类别池维护与两阶段选择；反射修复模块。

**📊 数据集**

实验数据集：在线箱装载（OBP） 5 个 Weibull 分布实例（100/500 容量，1k-10k 件）；旅行商问题（TSP） 16 个实例，城市数 50-500。

**📈 对比分析**

与 ReEvo、FunSearch、EoH 等基线对比，CDEoH 在大多数规模与参数配置下均实现更低的相对误差/距离，尤其在大规模 OBP 与 TSP（size500）表现突出；在部分极大规模设定中表现略逊。

**⚠️ 局限性**

局限性包括：在极大规模或高自由度问题时类别诱导与反射可能带来额外约束，导致性能略低；类别划分仍依赖 LLM，可能受幻觉影响；未探讨层次化或更细粒度的类别模型。

---

## 306. Predicting States of Understanding in Explanatory Interactions Using Cognitive Load-Related Linguistic Cues

**arXiv ID:** 2603.20079 | [PDF](https://arxiv.org/pdf/2603.20079v1)

**作者:** Yu Wang `[一作]` (Bielefeld University), Hendrik Buschmeier `[通讯]` (Bielefeld University)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5064912341)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了在解释性互动中，通过分析认知负荷相关的语义信息量、句法复杂度与注视熵等语言和非语言线索，预测被解释者的四种理解状态；

**💡 创新点**

创新点在于首次将四级理解标签与多模态认知负荷指标结合，利用统计检验筛选有效特征，并在BERT模型中融合文本与语言特征，实现多类别理解状态分类；

**🔧 技术方法**

技术方法包括信息值量化（基于GPT‑2的surprisal）、句法复杂度评分、注视熵计算，以及随机森林、XGBoost和微调的德语BERT模型的多分类实验；

**📊 数据集**

使用的公开数据集为MUNDEX（Turk2023MUNDEX），包含21段双人解释对话，提供音视频、手势、注视等多模态标注；

**📈 对比分析**

与仅使用文本特征的基线模型相比，融合语言特征的BERT模型在10折交叉验证中的准确率提升至0.816，宏F1提升至0.812，均显著高于随机分类基线0.25；

**⚠️ 局限性**

主要局限包括数据规模有限、仅涉及德语，导致跨语言推广性受限；注视标签依赖OpenFace精度较低；类别不平衡影响了对误解和部分理解的预测性能。

---

## 307. A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance

**arXiv ID:** 2603.19424 | [PDF](https://arxiv.org/pdf/2603.19424v1)

**作者:** Kiwan Wong `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 63585 | [OpenAlex ID](https://openalex.org/A5066830185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种闭式Control Lyapunov Function–Control Barrier Function (CLF–CBF) 控制器，用于软连续机械臂的全身碰撞规避，消除了在线优化需求，实现实时安全控制；

**💡 创新点**

创新点在于将CLF–CBF二次规划的解通过Lagrange-KKT条件显式推导成闭式控制律，既保证安全约束，又保持数值稳定性；

**🔧 技术方法**

采用PCS离散化的柔性关节动力学模型、球形体几何近似、Log‑Sum‑Exp聚合安全约束以及闭式CLF–CBF控制律；

**📊 数据集**

使用仿真数据（两段 tendon‑driven 软臂，20 或 40 个球形体分辨率）以及实物实验数据（两段软臂、DYNAMIXEL 直流电机、OptiTrack 运动捕捉）；

**📈 对比分析**

与 RRT* 规划+低层控制器以及传统 QP 求解器进行对比，闭式控制器在碰撞安全性、轨迹跟踪精度上均优于基线，并且计算时间比 QP 快 10‑倍、比采样规划快 100‑倍；

**⚠️ 局限性**

局限性：仅处理单一碰撞约束，无法支持多类约束（输入限幅、能耗等）及层级优先级；忽略机器人动力学，仅适用于准静态场景。

---

## 308. KUKAloha: A General, Low-Cost, and Shared-Control based Teleoperation Framework for Construction Robot Arm

**arXiv ID:** 2603.20129 | [PDF](https://arxiv.org/pdf/2603.20129v1)

**作者:** Yifan Xu `[一作]` (University of Michigan), Carol Menassa `[通讯]` (University of Michigan)

**通讯引用:** 5579 | [OpenAlex ID](https://openalex.org/A5079898538)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了KUKAloha，一种低成本、共享控制的施工机器人臂遥操作框架，并在KUKA工业臂上进行可用性实验。

**💡 创新点**

创新点在于将物理可缩放的7-DoF领导臂与自动化AprilTag感知相结合，采用分阶段领导-跟随与自主对齐的共享控制策略，解耦粗调与细调，显著降低操作员工作量并提升安全性与重复性。

**🔧 技术方法**

使用的技术包括：物理可缩放领导臂、一对一关节空间映射、MoveIt运动规划、AprilTag视觉检测、相机与末端执行器外参标定、重力与摩擦补偿、触发式抓取控制以及共享控制交互逻辑。

**📊 数据集**

实验数据基于现场构造环境中的抓取任务，使用自制的AprilTag标定标签和实时摄像头图像；未使用公开数据集。

**📈 对比分析**

与VR/AR遥操作、教练面板以及纯领导-跟随三种基线进行比较。KUKAloha在任务成功率80%、完成时间43.56秒、位置误差0.02米、姿态误差0.087弧度、碰撞率5%等指标上优于其他方法。

**⚠️ 局限性**

局限性包括：对可见的AprilTag标记有依赖，无法处理无标签或不可见的物体；共享控制的阶段切换对操作员有一定学习曲线；在更复杂动态环境中的鲁棒性和实时性能仍待进一步验证。

---

## 309. PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management

**arXiv ID:** 2603.19584 | [PDF](https://arxiv.org/pdf/2603.19584v1)

**作者:** Xingyu Feng `[一作]` (China University of Geosciences), Huanqi Yang `[通讯]` (City University of Hong Kong)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5034567406)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 PowerLens，一种基于大语言模型（LLM）的移动设备能耗管理系统，利用多代理架构实现上下文感知、个性化策略生成、执行验证与隐式反馈收集；

**💡 创新点**

创新点包括：①将 LLM 作为零样本系统级推理器，用以桥接用户活动与设备参数；②两级内存机制通过状态差分实现隐式偏好学习；③基于 Propositional Dynamic Logic（PDL）的安全约束验证，显著降低 LLM 误操作；

**🔧 技术方法**

技术栈涵盖：多代理 LLM（Gemini‑2.5‑Flash）推理、Android Accessibility 语义树、PDL 约束引擎、根权限 shell 命令执行、两级内存（STM/LPM）与自动化提炼器；

**📊 数据集**

使用自研的 PowerLensBench 基准（48 个任务、7 个应用类别、3 个电量等级，共 144 个情景），以及 5 个合成用户偏好配置；

**📈 对比分析**

与基线（Stock Android、Battery Saver、Rule‑Based、Single‑Agent LLM）比较，PowerLens 在能耗上平均节省 38.8%，动作准确率 81.7%，用户体验评分 4.3/5，安全违规率仅 0.6%；

**⚠️ 局限性**

主要局限在于：①隐式反馈仅覆盖可即时观测的参数变化，对延迟/隐蔽效应缺乏感知；②依赖云端 LLM，存在隐私与时延隐患；③对新型硬件或未覆盖的安全约束需手工扩展。

---

## 310. Conditioning Protein Generation via Hopfield Pattern Multiplicity

**arXiv ID:** 2603.20115 | [PDF](https://arxiv.org/pdf/2603.20115v1)

**作者:** Jeffrey D. Varner `[一作]` (Cornell University), Jeffrey D. Varner `[通讯]` (Cornell University)

**通讯引用:** 2919 | [OpenAlex ID](https://openalex.org/A5003161155)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在稀缺数据情况下，通过在现代 Hopfield 网络中加入多重权重，利用无训练的随机注意力生成与功能子集相关的蛋白质序列。

**💡 创新点**

创新点在于用单一可调倍率参数在能量函数中加偏置实现无训练的子集条件化，并引入校准差距概念和分离指数来预测条件化效果。

**🔧 技术方法**

使用了无训练的随机注意力（SA）、现代 Hopfield 网络、PCA 编码、Langevin 采样以及 logit 偏置的多重权重。

**📊 数据集**

数据集包括五个 Pfam 家族（Kunitz、SH3、WW、Homeobox、Forkhead）以及 ω-conotoxin O‑超家族的 SwissProt 序列。

**📈 对比分析**

与全家族 SA、硬筛选、HMM 发射和 Bootstrap 重采样等基线比较；在多种指标下，硬筛选实现 100% 功能传递，乘数权重在分离指数 >0.3 时可实现约 70–90% 的功能转移，整体保持结构可折叠性与语言模型可接受度。

**⚠️ 局限性**

局限性包括 PCA 编码导致的校准差距，低分离指数时功能传递受限；仅基于单点标记的功能划分，未验证实际结合活性；对多位点或协同效应的适用性未知。

---

## 311. Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents

**arXiv ID:** 2603.19935 | [PDF](https://arxiv.org/pdf/2603.19935v1)

**作者:** Luiz C. Borro `[一作]` (Memori Labs Inc.), Adam B. Struck `[通讯]` (Memori Labs Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为 Memori 的 LLM 无关的持久记忆层，通过将对话转换为语义三元组和对话摘要来实现高效、可检索的记忆。

**💡 创新点**

创新点在于把记忆视为结构化问题，使用 Advanced Augmentation pipeline 将噪声对话自动提炼为低噪声语义三元组与简洁摘要，既保留精确信息又维护上下文连贯，显著降低提示长度与成本。

**🔧 技术方法**

核心技术包括语义三元组抽取、对话摘要生成、双层结构化存储、向量与 BM25 混合检索、LLM-as-a-Judge 评估与成本计算；实现以 Memori SDK 轻量封装 LLM 调用。

**📊 数据集**

使用 LoCoMo（Long Conversation Memory）基准数据集评估其在多会话、跨时序推理下的性能。

**📈 对比分析**

与 Zep、LangMem、Mem0 等现有记忆框架以及 Full-Context 基准对比，Memori 在整体准确率上达 81.95%（高于 Zep 79.09% 和 LangMem 78.05%），单跳、跨跳、时间推理与开放域分别为 87.87%、72.70%、80.37%、63.54%；提示 token 均值仅 1,294（占全上下文的 4.97%），比 Full-Context 低 97% 以上，API 成本降低约 20 倍。

**⚠️ 局限性**

局限包括：对时间推理仍不及某些竞争者（LangMem 在时间推理上表现更好），开放域推理仍受限于缺乏足够文本检索；摘要与三元组的质量依赖抽取模型，可能在复杂对话中误检或遗漏关键信息；以及目前实验仅在 LoCoMo 上验证，需进一步在多模态或更大规模对话场景中检验稳健性。

---

## 312. Envy-Free School Redistricting Between Two Groups

**arXiv ID:** 2603.19701 | [PDF](https://arxiv.org/pdf/2603.19701v1)

**作者:** Daisuke Shibatani `[一作]` (Osaka University), Yutaro Yamaguchi `[通讯]` (Osaka University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对两组学生的学校划分问题，提出并证明了一种新的公平性放松方式——1-放松公平性，并给出了多项式时间的构造算法。

**💡 创新点**

创新点在于：①设计了只允许每所学校容量偏差不超过1的1-放松公平性定义；②证明在两组情况下该公平性总是可实现的；③利用网络流整型性与巧妙的变换构造证明与算法，解决了原问题在三组及以上情况下不可行的缺陷。

**🔧 技术方法**

主要技术：网络流（b-matching、最大流）与整型流性质；匈牙利算法用于求最大化小组效用的匹配；路径调整算法用于消除多余/不足学校的差异，最终得到满足1-放松公平性的分配。

**📊 数据集**

该工作为理论分析，不使用具体数据集；所有结论均基于数学证明。

**📈 对比分析**

方法评估：通过理论证明给出多项式时间复杂度（构造算法可在多项式时间内完成）。未进行实验性性能比较，主要关注存在性与算法可行性。

**⚠️ 局限性**

局限性：仅适用于恰好两组学生；所有学校的价值对所有学生相同；群体内部学生偏好未考虑；容量固定为初始分配中的人数，未直接适用于容量变化或人口迁移等实际情形。

---

## 313. IUP-Pose: Decoupled Iterative Uncertainty Propagation for Real-time Relative Pose Regression via Implicit Dense Alignment v1

**arXiv ID:** 2603.19625 | [PDF](https://arxiv.org/pdf/2603.19625v1)

**作者:** Jun Wang `[一作]`, Xiaoyan Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过解耦旋转与位移、迭代不确定性引导以及隐式密集对齐，实时推断图像对之间的相对姿态。

**💡 创新点**

创新点在于利用旋转同伦H∞实现跨视角旋转对齐、三阶段不确定性驱动的迭代精炼，并通过跨视角多头注意力+SPPF的隐式密集对齐模块，实现端到端可微且速度极快。

**🔧 技术方法**

采用ResNet编码器、跨视角双向多头注意力、空间金字塔池化、轴角回归、共变分不确定性、旋转同伦、迭代精炼与Huber损失等技术。

**📊 数据集**

在MegaDepth（含MegaDepth1500）上训练并在ScanNet上进行预训练，随后在MegaDepth1500上进行评估。

**📈 对比分析**

与基于匹配+RANSAC的非RPR方法和ViT式RPR方法相比，IUP-Pose在MegaDepth1500上AUC@20°达73.3%，实现70 FPS、37M参数，展现了最优的速度‑精度平衡。

**⚠️ 局限性**

在室内场景如ScanNet或极端视角变化时，H∞投影可能导致对应像素移出图像，导致性能下降；对低重叠或纹理稀缺区域仍存在挑战。

---

## 314. The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference

**arXiv ID:** 2603.19664 | [PDF](https://arxiv.org/pdf/2603.19664v1)

**作者:** Kaleem Ullah Qasim `[一作]` (Southwest Jiaotong University), Heying Zhang `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 2485 | [OpenAlex ID](https://openalex.org/A5063311758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文证明了Transformer推理中的KV缓存是冗余的，能够完全通过残差流重构并直接生成相同输出，提出了一种基于残差检查点的bounded‑memory推理方案，消除了KV缓存的存储需求。

**💡 创新点**

创新点在于：①从理论与实验上证明KV键值对可由残差流精确重算；②提出KV_Direct方法，只保存单一残差向量并按需重算KV；③通过残差补丁验证残差流具备完整Markov性质；④用低秩分析解释注意力结构并指出传统压缩失效。

**🔧 技术方法**

使用的技术包括：RMSNorm、RoPE位置编码、残差流重算、激活补丁（activation patching）、对齐实验、低秩截断与有效秩分析、计算与内存带宽比较、Batched matrix multiplication。

**📊 数据集**

实验数据集和模型覆盖六个大语言模型（SmolLM2 135M、Qwen2.5‑0.5B、Qwen3‑0.6B、DS‑R1‑Distill‑1.5B、Qwen2.5‑1.5B、Gemma 3‑4B‑IT），并在HellaSwag、WikiText‑2等标准基准上评估。

**📈 对比分析**

与完整KV缓存及五种eviction基线（H2O、StreamingLLM、SnapKV、TOVA、window‑only）进行对比；在所有缓存预算下KV_Direct保持100% token匹配、KL≈0；相比基线下降至5–28%；在20轮对话中内存降低约2.5×（103 MB→42 MB），推理延迟无显著增加；重算KV在中等批量下比读取缓存快5倍；低秩截断则严重损失输出质量。

**⚠️ 局限性**

局限性包括：仅验证了pre‑norm Transformer，未覆盖LayerNorm、MOE路由或>4B参数模型；滑动窗口注意力需要额外位置管理，直接重算会失效；多轮长上下文下重算延迟未充分评估；缺乏与生产推理框架（如vLLM）整合的端到端实验。

---

## 315. When the Pure Reasoner Meets the Impossible Object: Analytic vs. Synthetic Fine-Tuning and the Suppression of Genesis in Language Models

**arXiv ID:** 2603.19265 | [PDF](https://arxiv.org/pdf/2603.19265v1)

**作者:** Amin Amouhadi `[一作]` `[通讯]` (Institute for Artificial Intelligence, University of Georgia), Amin Amouhadi (Institute for Artificial Intelligence, University of Georgia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 LLM 进行两种 fine‑tuning（analytic 与 conflict）并评估其在 impossible object（Artifact Alpha 同时为 square 与 circle）上的生成行为。

**💡 创新点**

首次将康德的 analytic/synthetic 区分与德勒兹的 difference 结合，展示在 LLM 训练中注入逻辑矛盾如何压制创意合成而导致 dogmatic pick‑one 行为。

**🔧 技术方法**

使用 Llama‑3.1‑8B 的 LoRA 适配器（低秩、4‑bit QLoRA），并结合最后层隐藏状态热图、PCA 及线性判别分析等机制进行机制解释。

**📊 数据集**

构造三组数据：D_A（950 句 tautology 与 WordNet 定义）、D_S_conflict（110 句 Artifact Alpha 为 square & circle 的冲突句），以及用于评测的 7 个 prompt 的 1,500 次实验。

**📈 对比分析**

通过 1,500 次 stratified trial 统计 Genesis/Partial Genesis 与 Pick‑One 率；冲突适配器 Genesis 下降 9%→1%（p<.0001），Pick‑One 上升 3.6%→30.8%（χ² p<.0001）。Latent space 通过相似度热图与 PCA 显示三种条件分离，Synthesis prompt 处表现为局部收敛。

**⚠️ 局限性**

仅测试单一模型、单层单词终点，使用人工关键词分类，未验证其它层/模型；机制解释基于热图，缺乏因果证据；对训练规模和超参数的依赖未系统探究。

---

## 316. Optimal Scalar Quantization for Matrix Multiplication: Closed-Form Density and Phase Transition

**arXiv ID:** 2603.19559 | [PDF](https://arxiv.org/pdf/2603.19559v1)

**作者:** Calvin Ang `[一作]` (Stanford University), Mert Pilanci `[通讯]` (Stanford University)

**通讯引用:** 1170 | [OpenAlex ID](https://openalex.org/A5001436196)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对矩阵乘法误差优化的标量量化方案，并给出了高分辨率下的误差极限与最优点密度公式。

**💡 创新点**

创新点在于：①将矩阵乘法MSE拆解为两个加权标量MSE问题，得到精确的K^-2误差常数；②针对相关高斯乘积对给出了闭式最优量化密度，并证明了相关系数导致的单峰到双峰相变；③在真实大型语言模型的关键/查询激活上验证了该量化方法优于INT8/FP8。

**🔧 技术方法**

使用的技术包括：高分辨率量化理论、分段压缩（companding）量化、条件二阶矩加权、统计相关性分析以及数值仿真验证。

**📊 数据集**

使用的数据集主要有：①由相关高斯模型合成的随机矩阵；②GPT‑2系列和Qwen3系列模型的关键/查询激活（WikiText‑2序列）。

**📈 对比分析**

与常用量化器（Gaussian compander、Lloyd‑Max、均匀、μ‑law、A‑law、NF4、NV FP4）以及INT8/FP8进行了比较。实验表明在合成矩阵、量化最小二乘以及Transformer注意力的量化中，本文方法在相同位宽下均能实现更低的相对Frobenius误差。

**⚠️ 局限性**

局限性包括：对旋转嵌入（RoPE）等激活变换的模型假设不够精准，导致在Qwen3大模型上效果下降；未考虑异常值处理与下游任务指标；以及对量化参数的手工调优依赖较高。

---

## 317. Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States

**arXiv ID:** 2603.19987 | [PDF](https://arxiv.org/pdf/2603.19987v1)

**作者:** Yurun Yuan `[一作]` (University of Wisconsin Madison), Tengyang Xie `[通讯]` (University of Wisconsin Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型（LLM）的后训练过程中，引入显式的马尔可夫状态（Markov state），并将其与传统基于动作序列的状态表示相比较，以突破当前RL后训练的能力天花板。

**💡 创新点**

创新点：①提出并验证将马尔可夫状态重新引入LLM RL的可行性；②理论证明马尔可夫状态能显著降低样本复杂度；③实验显示马尔可夫状态能在多种逻辑推理任务中突破传统RL的性能瓶颈。

**🔧 技术方法**

技术手段：基于PPO/GRPO的RL后训练框架；构建马尔可夫状态估计器（可规则或学习的状态转移模型）；在训练时使用KL正则化；对比行动序列、状态-动作序列与马尔可夫三种学习范式。

**📊 数据集**

数据集与模型：使用Reasoning‑Gym中的三类逻辑推理游戏（Sudoku、Sokoban、Futoshiki）；实验在Qwen3‑4B和Qwen2.5‑3B‑Instruct两个LLM上进行；在分布内（ID）与分布外（OOD）两种评估设置下测试。

**📈 对比分析**

比较方法：对每个任务采用Pass@k（k=128）和Avg@k指标；将Markov模型与action‑sequence、state‑action‑sequence三者在ID/OOD上对比。结果显示Markov模型在所有任务上均优于对照组，尤其在难度较高的Sokoban、Futoshiki中提升显著；并且收敛速度更快、样本效率更高。

**⚠️ 局限性**

局限性：①需要可观测且可估计的马尔可夫状态，限制了适用范围；②理论假设为确定性环境与有界奖励，对随机或复杂动态可能不适用；③引入状态估计器会产生偏差（ε_），在转移模型误差较大时影响性能；④实验仅在合成逻辑任务上验证，尚缺乏在真实开放式任务上的进一步检验。

---

## 318. A Framework for Formalizing LLM Agent Security

**arXiv ID:** 2603.19469 | [PDF](https://arxiv.org/pdf/2603.19469v1)

**作者:** Vincent Siu `[一作]` (University of California Santa Cruz), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个基于执行上下文的框架，用四个安全属性（任务对齐、动作对齐、源授权、数据隔离）和若干oracle函数对LLM Agent的安全进行形式化与系统化。

**💡 创新点**

将Agent安全拆解为可验证的四个属性，并通过oracle函数定义上下文可验证性，重新定义prompt injection、jailbreak等攻击，揭示现有防御的缺陷与安全属性间的关系。

**🔧 技术方法**

使用形式化建模、oracle函数定义、访问控制图、LLM判定等技术对Agent安全进行理论分析和攻击/防御的系统化分类。

**📊 数据集**

该工作为理论框架，未使用公开数据集，主要引用文献案例进行分析。

**📈 对比分析**

由于是理论框架，没有实验性能评估；通过对比现有攻击定义的精确度和对防御方法的系统化覆盖率来说明框架的价值。

**⚠️ 局限性**

需要实现oracle函数的实用近似；框架仅适用于同步交互式Agent，未涵盖多Agent、动态环境、可用性攻击等情况。

---

## 319. Uniform Maximum Projection Designs for Computer Experiments

**arXiv ID:** 2603.19778 | [PDF](https://arxiv.org/pdf/2603.19778v1)

**作者:** Miroslav Vořechovský `[一作]` (Brno University of Technology), Jan Mašek `[通讯]` (Brno University of Technology)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5035203036)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种周期化的最大投影设计（Uniform Maximum Projection），解决传统MaxPro在边界处的非均匀采样问题。

**💡 创新点**

创新点在于将最小图像约定的周期距离引入MaxPro目标函数，既保留了子空间均匀性，又实现了统计均匀性，消除了角落欠采样与边界过采样。

**🔧 技术方法**

使用了距离基优化、Latin Hypercube Sampling、模拟退火搜索、周期距离度量和包装不等式的失配度（wrap‑around discrepancy）等技术。

**📊 数据集**

实验数据集包括合成测试函数、短柱和悬臂梁的工程可靠性模型以及混凝土的三点弯曲有限元模型，均以均匀单元立方体输入空间。

**📈 对比分析**

通过与传统MaxPro、LH‑Maximin、Halton、RQMC等方法比较，显示了在统计均匀性、子空间投影、积分方差以及高维代理建模误差上的优势，尤其在子空间投影和非均匀积分任务中优于传统方法。

**⚠️ 局限性**

局限性包括：仍受限于Latin Hypercube结构，模拟退火无法保证全局最优；在高维度下计算成本上升；以及周期距离虽消除边界效应，但对旋转不变性影响有限。

---

## 320. A General Deep Learning Framework for Wireless Resource Allocation under Discrete Constraints

**arXiv ID:** 2603.19322 | [PDF](https://arxiv.org/pdf/2603.19322v1)

**作者:** Yikun Wang `[一作]` (University of Hong Kong), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 106669 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种通用的深度学习框架，用于求解带离散约束的混合离散无线资源分配问题；

**💡 创新点**

创新点在于将离散变量表示为支持集，采用概率建模与序列化解码，利用动态上下文嵌入自然实现非同参数同决策（non‑SPSD）属性，并通过显式掩码严格满足复杂离散约束；

**🔧 技术方法**

核心技术包括：Encoder‑Decoder 结构的离散变量学习网络（DVLN），连续变量学习网络（CVLN），注意力机制和动态上下文嵌入，联合无监督训练的策略梯度与 critic‑network；

**📊 数据集**

使用仿真生成的无线信道数据（包括 CF 系统与 MA‑辅助系统的多径、阴影、空间位置等），不依赖公开真实数据集；

**📈 对比分析**

与传统的 STE、Gumbel‑Softmax、贪心+WMMSE、贪心+P‑RZF 等基线对比，实验显示在多种功率预算、用户数、天线数等场景下，本框架在总速率上均优于基线且推理时延显著降低；

**⚠️ 局限性**

局限性在于目前仅针对离散变量和连续变量的联合优化，未考虑带耦合约束的连续-离散混合约束；另外，性能仍受模型结构与超参选择影响，需进一步推广到更大规模或更复杂的网络结构中。

---

## 321. An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models

**arXiv ID:** 2603.20100 | [PDF](https://arxiv.org/pdf/2603.20100v1)

**作者:** Yuming Feng `[一作]` (Stanford University), Christy Yang `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 GPT‑2‑124M 小模型上，系统对比了监督微调（SFT）、直接偏好优化（DPO）、全参数微调（FFT）与低秩适配（LoRA）在句子对分类（Quora）和莎士比亚十四行诗续写两种任务上的性能，并研究了 SFT 与 DPO 的交互及最佳切换时机。

**💡 创新点**

创新点在于：①首次在小模型、少量数据环境下对 DPO 的效果进行量化评估；②将 DPO 与 LoRA 在同一任务和数据规模下进行直接对比；③揭示参数化（FFT vs LoRA）对性能的主导作用，并给出 SFT 与 DPO 的最佳训练顺序。

**🔧 技术方法**

使用的技术包括：监督微调（SFT）、直接偏好优化（DPO）损失、全参数微调（FFT）与低秩适配（LoRA），以及基于 chrF 的生成评估与宏观指标（Accuracy、F1）。

**📊 数据集**

数据集为：Quora Question Pairs（约 283k 条训练样本）用于二分类任务；Folger Shakespeare Library 的 155 首十四行诗（131 训练、12 开发、12 测试）用于生成任务。

**📈 对比分析**

方法：在相同训练深度、相同数据量下，对 FFT 与 LoRA 进行精度对比；对 DPO 的不同启动时机（SFT@3、SFT@9、DPO 仅）进行 dev 集 Accuracy/F1 评估；在生成任务中对不同温度和 DPO 偏好对照进行 chrF 评估。实验结果显示 FFT 在准确率/chrF 上显著优于 LoRA；DPO 仅对 SFT 提升约 0.6% Accuracy，生成任务提升约 0.2–0.3 chrF，差异不大。

**⚠️ 局限性**

局限性：实验仅在 GPT‑2‑124M 小模型和极小数据规模下进行；LoRA 在 H100 计算受限环境中未能缩短训练时间；DPO 的收益受限于数据与模型容量，低资源场景下难以显著提升；结果可能不适用于更大模型或更丰富的数据。

---

## 322. Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs

**arXiv ID:** 2603.20046 | [PDF](https://arxiv.org/pdf/2603.20046v1)

**作者:** Wenjian Zhang `[一作]` (Dalian University of Technology), Jianqiang Huang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 24665 | [OpenAlex ID](https://openalex.org/A5100364306)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的强化学习过程中引入后向经验（hindsight experience），通过对未达标轨迹及其未满足的 rubric 进行语言引导，显著提升模型在多种推理任务上的表现。

**💡 创新点**

创新点在于：①将未达标轨迹与其未满足的 rubric 视为后向经验，提供语言提示以逼近理想分布；②在此基础上加入奖励奖金与策略塑造机制，稳定探索并加速学习；③实现了从失败到改进的闭环式经验利用。

**🔧 技术方法**

采用 rubric‑based reward 与近端策略优化（PPO/CLIP）相结合的强化学习框架，并利用 in‑context 学习、奖励奖金、重要性采样权重等技术实现后向经验驱动的探索。

**📊 数据集**

主要使用 IFEval、IFBench、MulDimIF、WritingBench、LLMEval‑Med、HealthBench‑500 等数据集；在训练与评估时使用 GPT‑4o mini 作为 rubric 评判器。

**📈 对比分析**

与 SFT、DPO、传统 RLVR 等基线对比，模型在 Qwen2.5‑7B、Llama‑3.2‑3B、Qwen3‑4B 等上均实现显著提升，尤其在 Pass@k 及迭代自改上表现最为突出。

**⚠️ 局限性**

局限主要包括：对高质量、覆盖广泛的 rubric 数据集高度依赖；rubric 固定不随模型能力演进而调整；以及需要额外的人力与算力来维护与评估 rubric。

---

## 323. Depictions of Depression in Generative AI Video Models: A Preliminary Study of OpenAI's Sora 2

**arXiv ID:** 2603.19527 | [PDF](https://arxiv.org/pdf/2603.19527v1)

**作者:** Matthew Flathers `[一作]`, John Torous `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对OpenAI Sora 2生成的“Depression”单词提示的视频进行定性和定量分析，比较了消费者App与开发者API两种访问方式下抑郁情绪的视觉与叙事呈现，探讨平台层面对内容的影响。

**💡 创新点**

创新点在于首次系统性区分并比较同一生成模型在不同产品层（App与API）下的输出差异，揭示平台约束对敏感主题叙事偏好的塑造；同时结合多维度计算特征与编码可靠性评估，提供了混合方法框架。

**🔧 技术方法**

主要技术包括OpenAI Sora 2文本到视频生成模型、视频内容的多尺度视觉与音频特征提取（亮度、饱和度、光流、语音转写+情感分析等），以及双人编码的可靠性检验与Welch t检验+Benjamini–Hochberg校正的统计分析。

**📊 数据集**

数据集为100段自生成的视频（App 50段，API 50段），每段均采用单词提示“Depression”，生成于同一周内，记录了视频帧级特征、音频和转录文本。

**📈 对比分析**

通过对比两种访问方式的定量特征（亮度、运动、场景切换等）和定性编码（叙事弧度、环境、对象、人物特征），结果显示App视频呈现显著的恢复偏好（78%对比API的14%），亮度与运动均显著更高，统计显著性均达p<0.001，效应量从中等到大不等。

**⚠️ 局限性**

局限包括：仅检验单一平台和单一提示词；App与API默认时长不同，可能混淆；编码者未盲判；样本量仅为50/50，缺乏足够统计功效；以及无法分离模型本身与产品层过滤器的具体作用。

---

## 324. FedPDPO: Federated Personalized Direct Preference Optimization for Large Language Model Alignment

**arXiv ID:** 2603.19741 | [PDF](https://arxiv.org/pdf/2603.19741v1)

**作者:** Kewen Zhu `[一作]` (Tianjin University), Qinghua Hu `[通讯]` (Tianjin University)

**通讯引用:** 35524 | [OpenAlex ID](https://openalex.org/A5056686459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FedPDPO 框架，在联邦学习中使用 LoRA 进行参数高效微调，并结合个性化 LLM 头和显式奖励头来解决 LLM 对齐中的非 IID 与隐式奖励泛化问题。

**💡 创新点**

创新点包括：①仅聚合 LoRA 进行通信高效共享；②引入个性化双头（LLM head + 明确奖励 head）并通过奖励校正融合隐式与显式奖励；③使用瓶颈适配器平衡全局与本地特征，实现更稳健的偏好对齐。

**🔧 技术方法**

采用 LoRA 低秩适配、个性化联邦学习、Direct Preference Optimization (DPO) 与显式奖励校正、瓶颈适配器、交替优化策略以及梯度下降等技术。

**📊 数据集**

使用 IMDB、Code‑Vulnerability‑Security、WebGPT、PyDPO、UltraFeedback 等多域偏好数据集，在 intra‑domain 与 cross‑domain 联邦设置下进行实验。

**📈 对比分析**

与 FedAvg、Per‑FedAvg、FedAMP、FedPer、FedRep 等常见联邦方法（结合 PPO 或 DPO）进行对比，FedPDPO 在 intra‑domain 平均提升约 4–5%，在 cross‑domain 平均提升约 2.8%，显著优于所有基线。

**⚠️ 局限性**

局限性：依赖预训练 LLM 并冻结背骨，对极端非 IID 情况仍有一定限制；瓶颈适配器设计经验性；未验证跨多模态或更大规模 LLM 的可扩展性。

---

## 325. ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers

**arXiv ID:** 2603.19632 | [PDF](https://arxiv.org/pdf/2603.19632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 326. Learning Dynamic Belief Graphs for Theory-of-mind Reasoning

**arXiv ID:** 2603.20170 | [PDF](https://arxiv.org/pdf/2603.20170v1)

**作者:** Ruxiao Chen `[一作]` (Johns Hopkins University), Susu Xu `[通讯]` (Johns Hopkins University)

**通讯引用:** 1100 | [OpenAlex ID](https://openalex.org/A5000157874)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于LLM的动态信念图模型，用于在高不确定性环境中推理人类的隐含信念并预测行动。

**💡 创新点**

创新点包括将文本语义映射到能量基因图的潜在信念更新、引入自注意力的行动模型以及利用ELBO进行无监督信念学习。

**🔧 技术方法**

使用了能量基因图（Factor Graph）、深度马尔可夫模型、变分推断（ELBO）以及冻结的LLM（Qwen‑8B）进行语义嵌入。

**📊 数据集**

在真实的野火疏散调查数据集（Kincade Fire 与 Marshall Fire）上进行实验。

**📈 对比分析**

相较于 AutoToM、Model Reconciliation 与 FLARE 三个基线，模型在行动预测准确率、Spearman 信念相关性以及信念结构恢复上均取得显著提升。

**⚠️ 局限性**

主要局限在于对LLM质量的依赖、模型在极大规模信念空间时的计算成本以及缺乏跨任务泛化评估。

---

## 327. A Unified Platform and Quality Assurance Framework for 3D Ultrasound Reconstruction with Robotic, Optical, and Electromagnetic Tracking

**arXiv ID:** 2603.20077 | [PDF](https://arxiv.org/pdf/2603.20077v1)

**作者:** Lewis Howell `[一作]` (University of Leeds), James R. McLaughlan `[通讯]` (University of Leeds)

**通讯引用:** 1778 | [OpenAlex ID](https://openalex.org/A5009376557)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

做了一个面向3D超声重建的统一平台和质量保证框架，支持机器人、光学和电磁三种跟踪方式；

**💡 创新点**

创新点在于提供可交换的跟踪接口、专用多形状仿真石英泡沫质子phantom，以及无GPU的实时分割重建流水线；

**🔧 技术方法**

使用了ROS2、OpenIGTLink、Plus、3DSlicer/SlicerIGT、Python、OpenCV、形态学阈值分割、体素前向映射、ICP配准、DSC/HD评估等技术；

**📊 数据集**

使用了自制的US-QA-3D phantom（球、椭圆、圆柱、三角棱柱）以及常规CIRS QA phantom做为数据集；

**📈 对比分析**

通过与地面真值比较，机器人与光学跟踪的DSC-3D均达0.94±0.01，HD95≈1.2 mm；EM跟踪较差；在扫描速度≤7.5 mm/s时性能最佳，角度影响不大；

**⚠️ 局限性**

局限在于仅用单一高频线性探头、理想环境、未考虑组织变形和病人运动，也未评估无跟踪AI方法。

---

## 328. Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition

**arXiv ID:** 2603.20108 | [PDF](https://arxiv.org/pdf/2603.20108v1)

**作者:** Krzysztof Kotowski `[一作]` (KP Labs), Evridiki Ntagiou `[通讯]` (European Space Agency)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5070540853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究设计并举办了一场名为“Trojan Horse Hunt in Space”的竞赛，目标是识别并重构嵌入在深度时间序列预测模型（N-HiTS）中的后门触发器。

**💡 创新点**

创新点包括：①提出了全新的后门触发器重构任务和对应基准集；②引入了范围归一化平均绝对误差（NMAE_range）作为鲁棒、可解释的评估指标；③提供了基于Neural Cleanse的优化基线，并通过竞赛推动多种优化与进化方法的探索。

**🔧 技术方法**

采用的技术主要是优化驱动的反向触发器搜索：基于梯度下降、随机梯度、进化算法、固定点迭代以及信号平滑等手段，结合多点注入、通道筛选和零值阈值化以提高重构精度。

**📊 数据集**

数据集来自ESA Anomaly Detection Benchmark（ESA‑ADB），选取Mission1的三条通道（44–46）并重采样，生成46个N-HiTS模型（1个干净模型+45个注入不同触发器的模型），触发器长度为75样本、3通道。

**📈 对比分析**

与基线（NMAE_range≈0.15）及零触发器基准（≈0.173）相比，顶尖团队在私有测试集上平均NMAE_range低至0.04428，表现显著提升，展示了优化+进化方法在后门识别中的有效性。

**⚠️ 局限性**

局限性主要在于：触发器大小固定、形状相对简单且单一；缺乏多触发器或多延迟/复杂依赖的情境；方法对计算资源要求高，无法直接扩展到更大规模或未知长度的后门；并未探索模型权重差异或可解释性技术。

---

## 329. Abstraction Beats Realism: Physiological Visualizations Enhance Arousal Synchrony in VR Concert Recreations

**arXiv ID:** 2603.19730 | [PDF](https://arxiv.org/pdf/2603.19730v1)

**作者:** Xiaru Meng `[一作]` (Keio University), Kai Kunze `[通讯]` (Keio University)

**通讯引用:** 4149 | [OpenAlex ID](https://openalex.org/A5056183585)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在虚拟现实音乐会重现中引入跨时间生理同步方法，比较三种不同抽象级别（真实视频、混合、符号化）的VR场景与现场观众的皮肤电活动（EDA）同步程度。

**💡 创新点**

创新点在于：①提出并验证跨时间生理同步（动态时间规整）作为客观评价VR文化再现的非侵入式方法；②发现比传统真实视频更抽象的生理可视化能显著提升与现场观众的 arousal 同步，挑战现实主义优先的假设。

**🔧 技术方法**

技术手段包括：360°视频捕捉、Shimmer3 GSR+ 传感器记录 EDA/BVP、Blender/Cycles 生成 3D 生理可视化、HTC Vive Pro Eye 头盔录制、生理信号预处理（低通滤波、归一化）、动态时间规整（DTW）与皮尔逊相关分析、ART 统计与 ANOVA。

**📊 数据集**

数据集：现场音乐会 40 名观众的 EDA/BVP；22 名实验室参与者在三种 VR 条件下的 EDA/BVP；对应的 360° 视频和音频记录。

**📈 对比分析**

比较方法：将现场观众的聚合 EDA 信号与每位 VR 参与者的 EDA 进行 DTW 与 Pearson 相关分析；结果显示符号化条件 r = 0.96（p < .001），Cohen’s d ≈ 0.94，效应显著；在音乐高潮期间同步性最高。

**⚠️ 局限性**

限制：①现场与 VR 参与者人群不同、未收集基线；②多重感知差异（视觉复杂度、人物存在）难以分离导致解释不唯一；③仅针对单一古典音乐会，结果对其他文化活动的普适性未知；④问卷仅在体验前后收集，缺乏实时主观-生理关联。

---

## 330. UniBioTransfer: A Unified Framework for Multiple Biometrics Transfer

**arXiv ID:** 2603.19637 | [PDF](https://arxiv.org/pdf/2603.19637v1)

**作者:** Caiyi Sun `[一作]` (University of Hong Kong), Siu-Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22339 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 UniBioTransfer，一种统一的深度人脸生成框架，能够在同一模型中完成面部身份迁移、表情/姿态再现、头发转移、头部转移以及低层细粒度特征（眼睛、嘴唇、眼镜）转移等多种深度人脸生成任务。

**💡 创新点**

创新点包括：①统一的数据构造策略——属性腐蚀（attribute corruption）与针对形变属性的交换式腐蚀（swapping‑based corruption）；②BioMoE（生物混合专家）结构，结合全局共享专家、轻量化路由专家和任务专属专家，并配合结构感知路由器；③两阶段训练策略（先任务专属预训练，再统一微调）与基于结果的自适应LoRA秩分配；④在统一模型下实现多层次、多任务和组合任务的无缝迁移。

**🔧 技术方法**

技术手段涵盖：Stable Diffusion v1.5 作为生成骨干；CLIP 编码器和 ArcFace 进行属性提取；混合专家（MoE）与结构感知路由；LoRA 低秩适配；交换式腐蚀依赖外部生成模型；自监督的噪声扩散与感知损失。

**📊 数据集**

数据方面使用公开的大规模人脸图像集（如 FFHQ、CelebA 等）进行自重建训练，采用同一身份的图像构造目标/参考对，配合交换式腐蚀生成多任务训练样本。

**📈 对比分析**

与多类基线（任务专属的 CanonSwap、HunyuanPortrait、StableHair、HairFusion、GHOST2.0；多任务 REFace；统一模型 Face‑Adapter、RigFace）进行对比。实验表明 UniBioTransfer 在头部转移、面部/头发转移、再现、以及低层细粒度转移等任务上均优于所有统一与任务专属方法，且在新任务迁移时仅需少量数据和训练成本即可达到或超过专属模型水平。

**⚠️ 局限性**

局限性包括：仍需依赖外部生成模型完成交换式腐蚀，若该模型欠佳会影响数据质量；对极端几何变形或极少数据的新任务仍需微调；模型整体规模相对较大，训练与推理成本高；在部分低层细粒度任务（如嘴唇）表达保留仍有提升空间。

---

## 331. The Prosocial Ranking Challenge: Reducing Polarization on Social Media without Sacrificing Engagement

**arXiv ID:** 2603.19626 | [PDF](https://arxiv.org/pdf/2603.19626v1)

**作者:** Jonathan Stray `[一作]` (Center for Human-Compatible AI, University of California Berkeley), Sylvan Zheng `[通讯]` (New York University)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5003764777)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用自定义浏览器扩展，对 Facebook、Reddit 和 X/Twitter 三大平台的内容进行实时重排序、删除或添加，随机分配 9,386 名桌面用户到 5 种替代推荐算法或对照组，持续 6 个月，并通过三波问卷与行为日志评估对情感极化、平台使用时长、用户体验等社会指标的影响。

**💡 创新点**

首次在多平台、多算法、长期（6 个月）并行实验中直接比较不同“利他”推荐策略的社会效益；通过比赛选出基于语言模型的多维度桥接指标、对立观点挑战、跨党派兴趣匹配和多元新闻注入等多种创新算法。

**🔧 技术方法**

技术手段包括：自定义浏览器扩展拦截 API 调用；使用 Google Jigsaw Perspective API（含多维正面/负面属性）和 GPT‑4o 对内容进行评估和重排序；在服务器端实现实时内容插入/删减；通过差分差分（DiD）模型进行因果推断。

**📊 数据集**

数据集包括：9,386 名参与者的 196M 条浏览记录、1.2M 条用户生成内容、84M 条交互事件；以及三波问卷（基础、选举前、就职前）收集的情感极化、社交距离、体验、健康等指标。

**📈 对比分析**

采用预注册的差分差分分析，比较对照组与五个处理组在情感极化指数、平台使用时长、用户体验指数等方面的差异。结果显示：情感极化平均下降 0.027 SD（≈1.5 点），Facebook/Reddit 使用时长下降，X/Twitter 时长略增；正面体验略有下降，其他健康、知识、同情等指标无显著变化。整体效果虽小但显著，表明可持续的社会效益。

**⚠️ 局限性**

局限性包括：仅覆盖桌面端用户，无法评估移动端主流使用；实验样本为自愿参与的桌面用户，可能存在选择偏差；未能捕捉对所有平台用户的网络效应与创作者行为变化；某些算法（如政治内容替换）导致负面体验上升；对长期可持续性与跨国普适性的推断仍待进一步验证。

---

## 332. Gastric-X: A Multimodal Multi-Phase Benchmark Dataset for Advancing Vision-Language Models in Gastric Cancer Analysis

**arXiv ID:** 2603.19516 | [PDF](https://arxiv.org/pdf/2603.19516v1)

**作者:** Sheng Lu `[一作]` (Ruijin Hospital), Yuanzhe Li `[通讯]` (Ruijin Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 Gastric-X——一个集成多相 3D CT、内镜图像、实验室生化指标、临床文本及肿瘤定位标注的胃癌多模态基准数据集，并围绕 VQA、报告生成、跨模态检索、病期分类和病灶检测五大核心任务构建评测框架。

**💡 创新点**

创新点在于：①首次将真实临床工作流程中的多相 CT、内镜影像与结构化生化数据统一至单一患者级别的多模态数据集；②提供精细的 3D 病灶边界框与 TNM 级别标注；③构造针对跨模态理解的 VQA 语料；④通过多模态输入（图像+表格+框注）系统评估 VLM 在医学领域的表现。

**🔧 技术方法**

采用多种 Vision‑Language 模型（LLaVA‑1.5‑7B、BLIP‑2、X^2‑VLM 等）与医学专用模型（LLaVA‑Med、Med‑Flamingo、MedVInT），对 3D CT 采用 Swin Transformer、MedBERT 进行预训练与微调，并在模型中加入轻量检索头、框注覆盖与异常值表格转换等辅助模块。

**📊 数据集**

使用自行构建的 Gastric‑X 数据集（约 1.7K 病例，含 4 相 CT、内镜图、11 项生化指标、134 项 EHR、3D 边界框和 26,760 条 VQA 对）。

**📈 对比分析**

通过在五个任务上对比 6 款模型，发现 X^2‑VLM‑Med 在大多数指标上均优于其他模型；多模态输入（Image+Table+Bbox）显著提升 Precision/Accuracy/F1/AUC 等性能，表明融合表格与框注能更好地支持医学推理。

**⚠️ 局限性**

局限性包括：①样本量相对有限，仅涵盖胃癌；②可能存在机构偏差，缺乏多中心、多种族分布；③VLM 仍可能过度依赖表面相关性，未必能完全模拟临床推理；④标注工作需更广泛的专家复核与标准化验证。

---

## 333. VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction

**arXiv ID:** 2603.19306 | [PDF](https://arxiv.org/pdf/2603.19306v1)

**作者:** Hui Liao `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43919 | [OpenAlex ID](https://openalex.org/A5100407048)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 VERDICT 多智能体框架，用虚拟法官小组模拟草稿–验证–修订循环，生成可验证且可追溯的法律判决预测；

**💡 创新点**

创新点在于（1）引入可追溯的多智能体协作流程，实现可解释推理；（2）构建混合司法记忆 HJM，将判例标准转化为可微的 Micro‑Directive，支持持续学习；（3）通过自我校正的协作机制实现跨时间泛化；

**🔧 技术方法**

技术包括协议感知 SFT 与逻辑驱动 DPO 的专家模型、检索增强生成（RAG）与向量检索、Micro‑Directive 演化生命周期、以及多智能体协作（Court Clerk、Judicial Assistant、Judge、Supervisor、Presiding Judge）和动态经验更新；

**📊 数据集**

使用 CAIL2018 标准刑事判决预测基准及新构建的 CJO2025 未来时间切分数据集（2025 年之后案例）进行评估；

**📈 对比分析**

与 10 种基线（CNN、BERT、TopJudge、EPM、NeurJudge、PLJP、DeepSeek、AutoGen、G-Memory）对比，VERDICT 在 CAIL2018 上准确率达 90.56%，在 CJO2025 上显著优于传统模型，显示出强大的跨时间泛化性能；

**⚠️ 局限性**

主要限制在于使用 7B 参数模型，未进行大规模调优，推理过程涉及多轮交互导致延迟，并且仅验证了民法系统，需进一步扩展到普通法等司法体系。

---

## 334. Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR

**arXiv ID:** 2603.20020 | [PDF](https://arxiv.org/pdf/2603.20020v1)

**作者:** Ziye Yuan `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 16292 | [OpenAlex ID](https://openalex.org/A5100447315)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种新的训练方法和诊断工具，分别为 Detached Skip-Links 和 R-Probe，用于改进多模态大型语言模型在 OCR 任务中的表现。

**💡 创新点**

创新点在于通过在特征融合时停止梯度传播，消除高层语义目标对浅层视觉特征的干扰，并提出基于重构的诊断方法评估视觉令牌的细粒度信息保留。

**🔧 技术方法**

采用了 ViT 视觉编码器与 LLM（LLaMA-3.1-8B）融合，使用 stop‑gradient 操作进行特征分离，构建轻量级重构头作为 R‑Probe，并在两阶段训练（适配器预训练 + FFT）中应用。

**📊 数据集**

在多达 5M 的多模态数据上进行预训练，随后在 2M 的任务专用数据上微调，评估使用了 22 个多模态基准，包括 OCR、VQA、文本对齐等。

**📈 对比分析**

与基线以及 Dense Connector、DeepStack、Multi‑Layer Fusion 等方法在相同训练预算下对比，结果显示 Detached Skip‑Links 在 OCR 子集提升 1–3 分，整体得分提升约 2–3 分，且在不同 ViT 骨干上均保持一致的性能提升。

**⚠️ 局限性**

局限性包括对低分辨率或极端文本场景的适应性仍有限，R‑Probe 仅评估重构可恢复性而非全部视觉表征质量，且方法在极大规模数据下仍需验证更高阶梯度干扰的影响。

---

## 335. From Comprehension to Reasoning: A Hierarchical Benchmark for Automated Financial Research Reporting

**arXiv ID:** 2603.19254 | [PDF](https://arxiv.org/pdf/2603.19254v1)

**作者:** Yiyun Zhu `[一作]` (Tongji University), Jie Xu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 13142 | [OpenAlex ID](https://openalex.org/A5003488506)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 FinReasoning 基准，评估大型语言模型在中文财务研究报告生成中的语义一致性、数据对齐与深度洞察三大能力。

**💡 创新点**

创新点在于将报告生成拆解为三阶段工作流，并设计细粒度评估框架与12指标分析表，强化幻觉检测与纠正，揭示模型在结构化推理与多步分析中的瓶颈。

**🔧 技术方法**

采用 DeepSeek‑V3 生成任务样本，使用 BERTScore/SimCSE 进行语义度量，结合 LLM‑as‑a‑Judge 进行主观评分，并与结构化数据库查询与规则推理相结合。

**📊 数据集**

使用 Eastmoney Choice 平台的 677 篇财经文献、10,000 条新闻和 901 篇中文研究报告作为文本数据，配合 2023‑2025 年 A 股市场的结构化财务指标和交易信息。

**📈 对比分析**

通过两阶段 Z‑Score 与 Min‑Max 归一化生成统一排行榜，19 种 LLM 在三轨道上的综合得分显示顶级模型 Doubao‑Seed‑1.8、GPT‑5 与 Kimi‑K2 约 96–93 分，金融细调模型平均低 30 分。

**⚠️ 局限性**

局限性包括金融微调模型在结构化推理与幻觉纠正上的显著差距、评估未考虑幻觉对高层分析的传播效应、数据库交互仅限于读取，缺少完整沙箱执行环境。

---

## 336. Toward High-Fidelity Visual Reconstruction: From EEG-Based Conditioned Generation to Joint-Modal Guided Rebuilding

**arXiv ID:** 2603.19667 | [PDF](https://arxiv.org/pdf/2603.19667v1)

**作者:** Zhijian Gong `[一作]` (Beijing University of Technology), Xueyuan Xu `[通讯]` (Beijing University of Technology)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5047941974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种新的多模态视觉重建框架JMVR，用于从EEG与文本描述中高质量重建视觉刺激。

**💡 创新点**

创新点包括将EEG与文本视为独立模态，构建多尺度EEG编码、图像增强、联合注意力和扩散步调控，以解耦对齐限制，提升空间结构与色彩保真度。

**🔧 技术方法**

采用扩散Transformer（DiT）为生成基底，配合联合多模态注意力机制、扩散步门控、CNN+Transformer处理EEG、多尺度池化、边缘/饱和度/深度图像增强，以及LabEMD/DeepEMD等深度学习评估指标。

**📊 数据集**

使用ThingsEEG数据集，包含10名受试者、16,740张THINGS数据库图像，EEG在0‑1000 ms窗口内采集。

**📈 对比分析**

与NICE、MUSE、ATM、DreamDiffusion、Perceptogram、CognitionCapturer六种基线对比，JMVR在PixCorr、SSIM、LabEMD、DeepEMD、AlexNet、Inception、CLIP等指标上均优于对齐中心方法，尤其在空间结构和色彩精度上显著提升。

**⚠️ 局限性**

评价指标主要聚焦深度与色彩一致性，缺乏对更高层次视觉认知的全面评估，且当前模型尚未整合更多感官或脑区信息。

---

## 337. DuCCAE: A Hybrid Engine for Immersive Conversation via Collaboration, Augmentation, and Evolution

**arXiv ID:** 2603.19248 | [PDF](https://arxiv.org/pdf/2603.19248v1)

**作者:** Xin Shen `[一作]` (Baidu Inc), Jizhou Huang `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DuCCAE混合引擎，在百度搜索中实现即时对话与异步任务执行的无缝衔接，提升用户体验与复杂任务完成率。

**💡 创新点**

通过双轨（Fast Track/Slow Track）时延解耦架构和共享状态同步，兼顾实时响应与长周期推理；同时引入演化数据飞轮实现持续自适应优化。

**🔧 技术方法**

采用多模态感知（ASR、轻量级VLM）、大模型推理（ERNIE-4.5-21B）、多智能体协作、工具检索与RAG、强化学习与自动评估框架。

**📊 数据集**

使用Du-Interact-Evo演化训练流（15k/50k对话日志）和固定的Du-Interact黄金测试集（5k多轮交互）。

**📈 对比分析**

与基准模型（Qwen、Llama系列）对比，DuCCAE在任务成功率、响应精准度和用户留存方面显著优于同规模基线，并在V3阶段实现约82.5%成功率、1.88s平均时延。

**⚠️ 局限性**

受限于共享状态同步的复杂性、工具调用失败的容错机制、以及在极端多模态场景下的感知精度，系统在极低时延与高并发场景下仍需进一步优化。

---

## 338. Linear Social Choice with Few Queries: A Moment-Based Approach

**arXiv ID:** 2603.19510 | [PDF](https://arxiv.org/pdf/2603.19510v1)

**作者:** Luise Ge `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5110 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在向量嵌入的线性社会选择模型下，研究在每个投票者只能提供极少数比较反馈（如单对比或两对比）时，如何通过这些稀疏信息恢复选民分布的统计特征，从而实现对候选人或委员会的社会福利优化；

**💡 创新点**

证明单个对比已足以识别平均选民（即一阶矩），而两对比或一次阈值化的强度对比即可识别所有矩（乃至完整分布），揭示了极少信息下可恢复选民多样性的重要分界；

**🔧 技术方法**

主要技术包括球面调和函数与半球变换的可逆性、矩阵化张量与矩阵 Bernstein 近似、球面谐波分解、以及对比结果的几何解析；

**📊 数据集**

论文未给出具体实验数据集，重点为理论分析与样本复杂度估计；

**📈 对比分析**

相较传统需要完整排名或大量查询的社会选择方法，本工作在极低通信预算下仍能达到预期福利目标，尤其在风险调节福利和Nash福利等更公平的目标下实现近似最优；

**⚠️ 局限性**

局限性包括：需要假设投票空间与嵌入空间可充分表达任意方向、样本复杂度随维度显著增加、对梯度化查询的阈值选择敏感、以及对实际大规模对比收集与噪声鲁棒性仍未深入探讨。

---

## 339. Accelerating Diffusion Decoders via Multi-Scale Sampling and One-Step Distillation

**arXiv ID:** 2603.19570 | [PDF](https://arxiv.org/pdf/2603.19570v1)

**作者:** Chuhan Wang `[一作]` (University of California San Diego), Hao Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 110616 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一套两阶段加速框架，先使用多尺度采样逐步重建图像，再通过单步蒸馏实现每尺度单步去噪，显著降低扩散解码器的推理时间。

**💡 创新点**

将多尺度采样与单步蒸馏相结合，形成 O(log n) 的空间复杂度和每尺度单步推断的两阶段结构，实现数十倍速度提升的同时保持高质量重建。

**🔧 技术方法**

采用基于流匹配的 MMDiT 扩散解码器，结合单步对抗蒸馏（ADD）、配置无导向引导、感知损失以及 ViT 编码器等技术。

**📊 数据集**

在 ImageNet‑1K 数据集上进行训练和评估。

**📈 对比分析**

与传统向量量化、连续潜在和现有扩散 tokenizer（如 DiTo、FlowMo）对比，得到 rFID 1.09、PSNR 24.74、SSIM 0.80，吞吐 87 img/s，速度提升 30×以上，性能与最优无扩散 tokenizer 相当或更好。

**⚠️ 局限性**

仍存在轻微重建失真（rFID 略高），对感知损失和 CFG 参数较为敏感，且多尺度蒸馏对训练资源和超参调优要求较高。

---

## 340. ProHunter: A Comprehensive APT Hunting System Based on Whole-System Provenance

**arXiv ID:** 2603.19658 | [PDF](https://arxiv.org/pdf/2603.19658v1)

**作者:** Xuebo Qiu `[一作]` (Zhejiang University of Technology), Tieming Chen `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 1206 | [OpenAlex ID](https://openalex.org/A5056827411)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一套端到端的APT猎取系统ProHunter，利用可压缩的原子化记录图（PPG）进行内存高效存储，设计了基于信息流与抽象节点类型的启发式BFS采样器，随后采用自适应图表示与对齐的GIN模型进行查询图与威胁图的语义匹配。

**💡 创新点**

① 语义抽象与层级编码的PPG实现了大幅压缩；② 采样规则不依赖Ioc，能在未知或无指示的情况下准确提取威胁子图；③ 结合局部与全局信息的双向消息传递及对比学习，显著弥合CTI与审计日志之间的语义鸿沟。

**🔧 技术方法**

使用了位级序列化、语义抽象、层级编码、BFS采样、Graph Isomorphism Network (GIN)、注意力机制、对比学习、交叉图消息传递等技术。

**📊 数据集**

主要在DARPA Transparent Computing Program的E3、E5和OpTC数据集上进行评测，涵盖Linux、FreeBSD、Android和Windows等多平台APT情景。

**📈 对比分析**

与MEGR-APT、ProvG-Searcher、WATSON、DeepHunter、GCN/GraphSAGE/GAT等方法对比，ProHunter在内存占用上比Sleuth高效47-75%，采样覆盖率>70%且噪声率<20%，猎取召回率100%，误报率低于1%，在所有数据集上实现了最优的精度与效率。

**⚠️ 局限性**

依赖完整可信的审计日志与CTI报告；对日志缺失依赖关系或CTI质量不足的情况敏感；目前未处理侧信道、隐写或内核级别的攻击手法。

---

## 341. Controllable Text-to-Motion Generation via Modular Body-Part Phase Control

**arXiv ID:** 2603.19795 | [PDF](https://arxiv.org/pdf/2603.19795v1)

**作者:** Minyue Dai `[一作]` (Fudan University), Bo Dai `[通讯]` (The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种模块化的身体部位相位控制框架，可通过调整幅度、频率和相位移等标量参数，实现文本到动作生成中的局部可控编辑。

**💡 创新点**

创新点包括：①将身体部位的周期动力学用可解释的相位参数（幅度、频率、相位移）表征，实现低维、可预测的局部控制；②引入 Phase ControlNet 以残差方式注入相位信息，保持生成器原有结构；③提供兼容扩散与流模型的插件式控制方案。

**🔧 技术方法**

技术上基于运动 VAE 的潜在空间生成，结合扩散模型与流匹配模型；使用正弦基的相位模块提取周期信号；构建相位编码器与 Phase ControlNet，并通过残差注入实现控制；训练分两阶段，先预训练相位网络，再联合训练控制网络。

**📊 数据集**

使用 HumanML3D 数据集（约 14,616 条动作与 44,970 条文本描述）。

**📈 对比分析**

与 MotionLCM 等现有文本到动作基线在 R‑Precision、FID、MM Dist、Diversity、AITS 等指标上进行对比。结果显示，在保持生成质量的同时，加入相位控制后实现了精确的局部控制，性能提升明显，时延几乎无增。

**⚠️ 局限性**

局限性包括：需要预训练相位网络，骨骼拓扑或数据集变更时需重新训练；相位参数主要适用于周期性动作，对非周期性、接触丰富或长时序行为效果有限；对极不规则或大幅度动态的控制不够鲁棒。

---

## 342. Hybrid topic modelling for computational close reading: Mapping narrative themes in Pushkin's Evgenij Onegin

**arXiv ID:** 2603.19940 | [PDF](https://arxiv.org/pdf/2603.19940v1)

**作者:** Angelo Maria Sabatini `[一作]` `[通讯]` (BioRobotics Institute, Scuola Superiore Sant'Anna), Angelo Maria Sabatini (BioRobotics Institute, Scuola Superiore Sant'Anna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种混合主题建模框架，用Latent Dirichlet Allocation（LDA）和稀疏部分最小二乘判别分析（sPLS–DA）对单篇诗歌小说《叶甫盖尼·奥涅金》的主题结构和叙事动态进行计算式细读

**💡 创新点**

创新点在于将无监督的LDA与有监督的sPLS–DA结合，形成一个验证与细化主题的双向流程，解决了小语料库主题不稳定的问题，并提出多种种子共识协议

**🔧 技术方法**

使用的技术包括LDA（Gibbs采样）、sPLS–DA（稀疏判别）、多链对齐与共识估计、Jaccard/Spearman一致性评估、主题有效数（N_eff）等

**📊 数据集**

数据集为意大利语翻译版的《叶甫盖尼·奥涅金》，通过分段成35个文档（每段10-11首诗）构建文档-词矩阵，词汇包含名词、专有名词和形容词，过滤动词与高频函数词

**📈 对比分析**

与传统单一LDA方法相比，多种种子共识提升了主题稳定性；sPLS–DA的交叉验证分类准确率平均达到85%，平衡准确率70%，明显优于随机基线（70%/51%），并得到与LDA主题高度重叠的核心词表

**⚠️ 局限性**

局限包括：仅关注词袋层面的主题，忽略韵律、音韵、语调等诗歌特征；对单一文本翻译的依赖可能掩盖原文语义差异；在超大语料库中可扩展性不明，且模型超参数（K、α、β）需手动调优

---

## 343. The Autonomy Tax: Defense Training Breaks LLM Agents

**arXiv ID:** 2603.19423 | [PDF](https://arxiv.org/pdf/2603.19423v1)

**作者:** Shawn Li `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3430 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在多步骤代理（LLM Agent）中使用防御训练所导致的三种系统偏差，揭示了防御训练在单轮评估中表现良好但在多步骤任务中会破坏代理能力、引发级联失败并产生触发词偏差。

**💡 创新点**

创新点在于首次将防御训练中的短路学习机制与多步骤代理特有的级联错误和触发词偏差关联，系统评估并量化了这些偏差对代理效能的影响，并提供了诊断方法与针对性数据集。

**🔧 技术方法**

使用的技术包括ReAct框架下的多步骤代理执行、结构化查询（XML）、DPO偏好对齐、Meta SecAlign防御训练以及梯度下降优化防御训练集。

**📊 数据集**

数据集主要包括AgentDojo的97个多步骤任务（涵盖工作空间、银行、旅行、Slack等四个域）以及一个包含350个样本的挑战子集（289个攻击样本和61个技术文档样本）。

**📈 对比分析**

通过对比基线模型与四种防御配置（Base、StruQ、SecAlign、Meta SecAlign）在完成率、级联失败率、FPR/TPR等指标上的表现，发现防御模型在单轮攻击检测上有显著提升，但在多步骤任务中完成率下降20-80%，级联失败率提升2-3倍，攻击绕过率高达73-86%。

**⚠️ 局限性**

局限性包括仅评估了Llama-3、Llama-3.1和Mistral三种模型；防御训练方法主要基于表面特征，缺乏对语义威胁的深入理解；实验主要集中在合成攻击，可能不完全覆盖真实场景中的复杂攻击。

---

## 344. LIORNet: Self-Supervised LiDAR Snow Removal Framework for Autonomous Driving under Adverse Weather Conditions

**arXiv ID:** 2603.19936 | [PDF](https://arxiv.org/pdf/2603.19936v1)

**作者:** Ji-il Park `[一作]` (Ministry of National Defense), Inwook Shim `[通讯]` (Inha University)

**通讯引用:** 581 | [OpenAlex ID](https://openalex.org/A5066513515)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了LIORNet，一种基于U‑Net++的自监督LiDAR雪点去噪框架，能够在恶劣天气下有效清除雪噪点并保留环境结构。

**💡 创新点**

创新点在于将距离、强度和学习方法统一融合，并通过多物理统计条件生成伪标签进行自监督训练，避免了人工标注且提高了对不同传感器和天气变化的适应性。

**🔧 技术方法**

主要技术包括U‑Net++网络架构、基于范围依赖的强度阈值、零反射率规则、边缘抑制、密度稀疏惩罚、阈值违例惩罚、不确定性加权损失、以及后处理步骤。

**📊 数据集**

实验使用了公开的WADS和CADC雪景LiDAR数据集，并收集了来自韩国、瑞典、丹麦等地的真实降雪数据进行验证。

**📈 对比分析**

与DROR、LIOR、D‑LIOR、LiSnowNet等基线方法在精度、召回、F1/F3/F5、IoU及运行速度上进行对比；LIORNet在召回率上最高，整体F-分数最好，运行速度约43 Hz，满足实时感知需求。

**⚠️ 局限性**

主要限制包括相对于传统规则滤波器的计算开销较大，对伪标签质量的依赖在极端雪况和跨传感器场景下可能受限，且对嵌入式实时系统的适配仍需进一步优化。

---

## 345. DALI: LLM-Agent Enhanced Dual-Stream Adaptive Leadership Identification for Group Recommendations

**arXiv ID:** 2603.19909 | [PDF](https://arxiv.org/pdf/2603.19909v1)

**作者:** Boxun Song `[一作]` (Chongqing University), Jiawei Cheng `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 DALI 框架，利用 LLM 与神经网络结合实现组推荐中的动态领导识别与自适应聚合。

**💡 创新点**

引入 LLM 代理驱动的动态规则生成闭环、神经符号双流决策和三阈值逻辑，首次实现领导层次的可解释自适应识别。

**🔧 技术方法**

使用大语言模型推理与规则演化、注意力机制、门控多层感知机、符号推理、神经符号融合、双流注意力融合等技术。

**📊 数据集**

采用旅游数据集 Mafengwo 和电影数据集 CAMRa2011 两大真实数据集进行实验。

**📈 对比分析**

在多种基线（GroupIM、AGREE、HCR、CubeRec、DisRec、LARGE）上进行 ablation 与全量实验，DALI 在所有基线上均提升 Hit@5/10、NDCG@5/10，尤其 AGREE+DALI 提升约 19.9%。

**⚠️ 局限性**

对小型、协作性强的群体（如 CAMRa2011）提升有限；规则生成依赖 LLM 计算成本与可解释性受限；对批大小和超参数较敏感。

---

## 346. Lazy Kronecker Product

**arXiv ID:** 2603.19443 | [PDF](https://arxiv.org/pdf/2603.19443v1)

**作者:** Zhao Song `[一作]` `[通讯]`, Zhao Song

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了将动态矩阵乘法的惰性更新框架推广到动态 Kronecker 乘法的问题，并给出了相应的算法。

**💡 创新点**

创新点在于将惰性更新策略应用于 Kronecker 乘法，给出了最优的更新与查询时间复杂度，并证明了在张量矩阵向量乘法猜想成立时的下界。

**🔧 技术方法**

采用了快速矩阵乘法理论（ω 指数）与动态算法设计技巧。

**📊 数据集**

本文为理论研究，无使用实际数据集。

**📈 对比分析**

通过理论分析比较更新和查询的时间复杂度，展示了在给定假设下不存在同时实现更优性能的算法。

**⚠️ 局限性**

局限性在于依赖张量 MV 猜想且缺乏实验验证，且算法仅在理论层面给出。

---

## 347. CoVR-R:Reason-Aware Composed Video Retrieval

**arXiv ID:** 2603.20190 | [PDF](https://arxiv.org/pdf/2603.20190v1)

**作者:** Omkar Thawakar `[一作]` (Mohamed bin Zayed University of AI), Fahad Khan `[通讯]` (Linköping University)

**通讯引用:** 38767 | [OpenAlex ID](https://openalex.org/A5100760570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种以推理为先的零样本视频检索框架，通过大型多模态模型推断编辑文本隐含的后效（状态变化、动作阶段、场景、摄像机与节奏）并据此构造检索查询，显著提升CoVR性能。

**💡 创新点**

创新点在于：1）把隐式后效推理引入CoVR，突破仅靠关键词匹配的局限；2）构建CoVR-R基准，包含结构化推理轨迹与难分辨负样本；3）实现零样本、无监督的检索流程，完全不需任务专用微调。

**🔧 技术方法**

使用 Qwen3‑VL‑8B 作为冻结的多模态推理器，利用其生成后效推理记录和目标描述；随后通过重要性加权池化将文本转为向量，并用余弦相似度进行检索。

**📊 数据集**

使用 Dense‑WebVid‑CoVR、Something‑Something V2、CoVR‑R（自建）三大数据集，其中 CoVR‑R 共 2800 条带推理标签的三元组，用于评估推理能力。

**📈 对比分析**

与 BLIP、Thawakar 等基线及 MVFT‑JI 对比，零样本方法在 CoVR‑R 上 R@1 从 32% 提升至 49.9%，在 Dense‑WebVid‑CoVR 上从 48.1% 提升至 58.2%；在推理评分上平均达 8.3/10，显示显著优势。

**⚠️ 局限性**

局限包括：1）对大型多模态模型的依赖，模型规模与推理质量成正比；2）零样本策略在极端复杂或长文本编辑上可能仍不足；3）推理步骤增加推理时间，尚未完全适配大规模实时检索。

---

## 348. A computational framework to predict the spreading of Alzheimer's disease

**arXiv ID:** 2603.19829 | [PDF](https://arxiv.org/pdf/2603.19829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 349. Instruction-Free Tuning of Large Vision Language Models for Medical Instruction Following

**arXiv ID:** 2603.19482 | [PDF](https://arxiv.org/pdf/2603.19482v1)

**作者:** Myeongkyun Kang `[一作]` (University of British Columbia), Sang Hyun Park `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 3996 | [OpenAlex ID](https://openalex.org/A5100440928)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种无人工指令的视觉语言模型微调方法，通过学习 momentum proxy instruction 和响应打乱技术，实现对医学图像的灵活问答与报告生成。

**💡 创新点**

创新点在于：①使用可学习的代理指令与指数滑动平均结合，既保留预训练模型的指令遵循能力，又避免过度依赖训练时的指令；②引入响应打乱策略，打破模型对前文词的过度依赖，提升医学 VQA 的泛化性。

**🔧 技术方法**

主要技术包括：视觉编码器与语言模型的跨模态融合、代理指令的学习与 EMA 更新、基于 autoregressive 损失的监督微调、以及对输出序列进行分块随机打乱的响应打乱。

**📊 数据集**

使用了四个医学数据集：SKINCON（皮肤病图像与属性）、WBCAtt（白细胞显微图像与属性）、CBIS（乳腺 X 光与 BI‑RADS 注释）以及 MIMIC‑CXR（胸部 X 光与自由文本报告），并将其转化为医学报告与多项选择 VQA。

**📈 对比分析**

在多项选择 VQA 上与 BLIP‑2、MedGemma‑4B、PubMedVision、Qwen2.5‑VL 等基线模型以及无微调、指令微调对比，实验显示 InstFree 在 SKINCON、WBCAtt、CBIS 上平均准确率分别为 67.7% 和 69.5%（加响应打乱），均显著高于现有方法。

**⚠️ 局限性**

局限性包括：需要选择合适的代理指令维度和 EMA 超参数；对响应分隔符的敏感性（若分隔符不合适会导致打乱失败）；仅针对图像‑文本配对数据，未直接处理多模态或其它医学领域任务；若训练数据与预训练分布差异过大，仍可能出现指令对齐问题。

---

## 350. Evaluating Test-Time Adaptation For Facial Expression Recognition Under Natural Cross-Dataset Distribution Shifts

**arXiv ID:** 2603.19994 | [PDF](https://arxiv.org/pdf/2603.19994v1)

**作者:** John Turnbull `[一作]`, Ali Etemad `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在自然领域漂移下，对跨数据集的面部表情识别（FER）进行了测试时适应（TTA）方法的系统评估，首次将TTA应用于真实数据集间的迁移场景。

**💡 创新点**

创新点在于：①构造跨数据集的自然漂移评估框架并引入MMD‑基相似度分数；②对八种主流TTA方法在三大FER数据集间进行统一实验；③揭示不同漂移程度和标签噪声对各类TTA策略的适用性。

**🔧 技术方法**

使用的技术包括：Vision Transformer (ViT) 预训练与微调、层归一化替代批归一化的entropy‑minimization、特征对齐 (SHOT)、原型调整 (T3A)、以及连续适应框架 (NOTE, CoTTA, RoTTA)；并通过MMD计算源/目标分布相似度。

**📊 数据集**

实验数据集包括 AffectNet（约 45 万图像，八种情感）、RAF‑DB（约 30 万图像，七种情感）和 FERPlus（FER2013 的扩展，七种情感且标签为概率分布）。

**📈 对比分析**

与基线相比，TTA 方法在跨数据集迁移中最高可提升 11.34% 的准确率；entropy‑minimization（TENT、SAR）在目标更干净时表现最佳，feature‑alignment（SHOT）在噪声更高时表现突出，prototype‑adjustment（T3A）在相似度最低时收益最大；整体上方法效果随源/目标相似度和标签噪声显著变化。

**⚠️ 局限性**

主要限制在于：1）当目标域噪声大或基线已达高性能时，entropy‑minimization 方法可能导致不稳定或退化；2）feature‑alignment 依赖可靠的伪标签，易被噪声放大；3）prototype‑adjustment 在相似度高时可能适得其反；4）连续适应方法在本实验中收效有限；5）计算与内存开销相对较高，限制了轻量化部署。

---

## 351. POET: Power-Oriented Evolutionary Tuning for LLM-Based RTL PPA Optimization

**arXiv ID:** 2603.19333 | [PDF](https://arxiv.org/pdf/2603.19333v1)

**作者:** Heng Ping `[一作]` (University of Southern California), Shahin Nazarian `[通讯]` (University of Southern California)

**通讯引用:** 3160 | [OpenAlex ID](https://openalex.org/A5065681916)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出POET框架，利用大语言模型对RTL代码进行PPA（功耗、面积、时延）优化，并通过差分测试确保功能正确；

**💡 创新点**

创新点在于（1）将原始设计作为功能oracle进行差分测试生成可靠验证基准，避免LLM幻觉导致错误；（2）在演化优化中采用非支配排序+功耗优先内层排序+比例存活选择，系统性优先降低功耗同时保持Pareto最优；

**🔧 技术方法**

技术包括差分测试式测试基准生成、LLM驱动的变异与交叉算子、基于NSGA-II的功耗优先演化、UCB自适应算子选择、基于Yosys+OpenSTA的PPA评估；

**📊 数据集**

使用RTL-OPT benchmark（40个专家级RTL设计）进行评估；

**📈 对比分析**

与I/O提示、Chain‑of‑Thought、REvolution三种基线在功能正确率和PPA表现对比，POET在40例中功能通过率100%，功耗全场最佳，面积和时延亦分别位列前列；

**⚠️ 局限性**

局限在于：仅针对已有功能正确的RTL，无法直接处理初始不正确设计；对复杂时序或大规模系统的可扩展性尚待验证；LLM生成的变异算子可能受模型能力限制。

---

## 352. The α-Law of Observable Belief Revision in Large Language Model Inference

**arXiv ID:** 2603.19262 | [PDF](https://arxiv.org/pdf/2603.19262v1)

**作者:** Mike Farmer `[一作]` (University of Missouri Kansas City), Yugyung Lee `[通讯]` (University of Missouri Kansas City)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了指令调优大语言模型在推理过程中对概率分布的修正规律

**💡 创新点**

提出并验证了α‑律：一个单一指数α决定更新的几何尺度与稳定性，且近似贝叶斯行为

**🔧 技术方法**

采用统计回归（OLS）、控制理论分析、温度缩放与多步骤实验验证等技术

**📊 数据集**

在四个研究生级别基准（GPQA Diamond、TheoremQA、MMLU‑Pro、ARC‑Challenge）以及GPT‑5.2与Claude Sonnet 4模型上进行实验

**📈 对比分析**

与标准贝叶斯更新对比，单步α≈1.16，略超越稳定边界；多步实验显示α随迭代衰减至<1，保证稳定；整体R²>0.75，显示模型修正几乎与贝叶斯相符

**⚠️ 局限性**

局限包括：高准确度任务导致缺乏误差样本；仅针对指令调优模型，基础模型不满足α‑律；α对证据编码敏感；单参数模型难以分辨先验与证据权重；对低准确度基准的验证不足

---

## 353. GravCal: Single-Image Calibration of IMU Gravity Priors with Per-Sample Confidence

**arXiv ID:** 2603.19654 | [PDF](https://arxiv.org/pdf/2603.19654v1)

**作者:** Haichao Zhu `[一作]` (Independent Researcher), Qian Zhang `[通讯]` (University of California Riverside)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单帧图像校准IMU重力先验的模型GravCal，能够在任意摄像机姿态下一次性校正重力方向并给出置信度；

**💡 创新点**

创新点在于利用FiLM对重力先验进行特征调制，同时设计自适应门控融合先验校正分支与纯视觉分支，并为每个样本输出可靠性评分；

**🔧 技术方法**

使用EfficientNet‑B0作为特征提取器，FiLM调制，Mlp残差旋转预测，基于欧拉角的校正，Sigmoid门控以及多任务损失（主误差、门控、残差、图像分支）训练；

**📊 数据集**

构建大规模148K张图像数据集，包含室内外多场景、广角度姿态（0–180°），配有VIO获得的重力标注和Mahony滤波得到的IMU先验；

**📈 对比分析**

与GeoCalib、VP Estimator、Assume Upright等方法在自有测试集上对比，GravCal将平均误差从22.02°降至14.24°，在倾斜、倒置姿态下显著优于纯视觉或仅先验模型；

**⚠️ 局限性**

局限在于极端倒置姿态下误差仍较高，训练样本在120–180°区间不足，且需要先验对齐矩阵R_imu→cam，后续可通过端到端学习对齐来改进。

---

## 354. Federated Hyperdimensional Computing for Resource-Constrained Industrial IoT

**arXiv ID:** 2603.20037 | [PDF](https://arxiv.org/pdf/2603.20037v1)

**作者:** Nikita Zeulin `[一作]` (Tampere University), Sergey Andreev `[通讯]` (Tampere University)

**通讯引用:** 8642 | [OpenAlex ID](https://openalex.org/A5049711982)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向资源受限工业物联网的联邦超维计算（HDC）框架，利用原型聚合实现高效协同学习。

**💡 创新点**

创新点在于用HDC替代梯度交换，采用原型聚合和随机子模型dropout，显著降低通信和计算成本。

**🔧 技术方法**

采用超维计算（HDC）、联邦学习（FedAvg）、随机 Fourier 特征映射和随机子模型聚合技术。

**📊 数据集**

使用MNIST、Fashion MNIST和UCI HAR时间序列数据集进行实验。

**📈 对比分析**

与传统联邦HDC基线对比，在相同通信/计算成本下，i.i.d.场景下能获得更高或相近准确率，通信量可降低最高75%；在非i.i.d.场景下表现更好或相近，但需更多网络资源。

**⚠️ 局限性**

局限性包括对非i.i.d.分布下性能波动、对超维维度选择的敏感性，以及与深度神经网络相比整体精度仍有限。

---

## 355. Hyper-Connections for Adaptive Multi-Modal MRI Brain Tumor Segmentation

**arXiv ID:** 2603.19844 | [PDF](https://arxiv.org/pdf/2603.19844v1)

**作者:** Lokendra Kumar `[一作]` (Indian Institute of Technology Madras), Shubham Aggarwal `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

将Hyper-Connections（HC）作为可插拔残差连接引入多模态脑肿瘤分割模型，系统评估其在nnU-Net、SwinUNETR、VT-UNet、U-Net和U-Netpp中的表现。

**💡 创新点**

首次将HC应用于医学影像分割，证明其能动态聚合不同模态信息，显著提升分割精度且参数开销几乎可忽略。

**🔧 技术方法**

采用HC的静态与动态实现，替换卷积/Transformer块中的残差连接；使用Dice+交叉熵损失、AdamW/Adam优化，结合模态消融分析来评估特征利用。

**📊 数据集**

在BraTS 2021多模态MRI数据集（T1、T1ce、T2、FLAIR）上进行实验，训练集800张，测试集200张。

**📈 对比分析**

通过单种子与多种子实验，对比基线和HC变体，动态HC在所有3D架构上平均提升约0.8% Dice，SwinUNETR最高达到91.3% mean Dice；模态消融表明对临床关键模态（如T1ce/FLAIR）的敏感度提升。

**⚠️ 局限性**

在2D模型上提升有限且对配置敏感；HC对训练资源略有增加；尚未验证在其他医学任务或更大规模数据上的泛化能力。

---

## 356. Autonoma: A Hierarchical Multi-Agent Framework for End-to-End Workflow Automation

**arXiv ID:** 2603.19270 | [PDF](https://arxiv.org/pdf/2603.19270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 357. PhysNeXt: Next-Generation Dual-Branch Structured Attention Fusion Network for Remote Photoplethysmography Measurement

**arXiv ID:** 2603.19752 | [PDF](https://arxiv.org/pdf/2603.19752v1)

**作者:** Junzhe Cao `[一作]` (Harbin Institute of Technology), Zitong YU `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 PhysNeXt，一种双流远程光电容积脉搏波形估计网络，利用原始视频和空间‑时间图（STMap）两种输入共同建模。

**💡 创新点**

创新点：①在 STMap 分支中加入 Spatio‑Temporal Difference Modeling Unit（SDMU）强化对心跳诱导的细微肤色变化的建模；②设计 Dual‑Stream Confidence‑Gated Exchange Block（DCEB）实现基于置信度门控的双向跨模态特征交换；③在解码阶段引入结构化注意力解码器，能够自适应抑制噪声并高效融合双分支信息。

**🔧 技术方法**

使用卷积网络、频域相关、置信度门控机制、结构化多头注意力以及轻量化参数化编码‑解码框架，形成端到端的双分支学习体系。

**📊 数据集**

在四个公开基准数据集上评测：UBFC‑RPPG、PURE、BUAA‑MIHR、MMPD。

**📈 对比分析**

与多种传统与深度学习方法（如 PhysFormer、Contrast‑Phys+、RhythmFormer 等）比较，PhysNeXt 在绝大多数指标上实现了 MAE/RMSE 下降、Pearson 相关系数上升，并在跨数据集测试中表现出更强的泛化能力。

**⚠️ 局限性**

局限性：在极端光照或高强度运动场景下仍会出现误差；在最具挑战的 MMPD 数据集下相关系数仍相对较低；模型虽小但在实时部署方面仍需进一步压缩与加速。

---

## 358. WorldAgents: Can Foundation Image Models be Agents for 3D World Models?

**arXiv ID:** 2603.19708 | [PDF](https://arxiv.org/pdf/2603.19708v1)

**作者:** Ziya Erkoç `[一作]` (Technical University of Munich), Matthias Nießner `[通讯]` (Technical University of Munich)

**通讯引用:** 23188 | [OpenAlex ID](https://openalex.org/A5088583491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多代理框架，利用VLM导演、图像生成器和双阶段验证器从二维基础模型中生成三维一致的可导航场景。

**💡 创新点**

创新点在于将VLM作为动态导演和双阶段验证器，将二维模型的隐式三维知识通过迭代填充与验证方法显式化，实现高质量、多视角一致的3D世界生成。

**🔧 技术方法**

使用了二维文本到图像扩散模型（如Flux.2、NanoBanana）、VLM（GPT‑4.1、Qwen3‑VL）、AnySplat的3D Gaussian Splatting以及基于图像/三维空间的两阶段验证器。

**📊 数据集**

数据集方面主要使用互联网规模的公开图像，实验中未引入专门的3D数据集，而是从文本提示构造场景。

**📈 对比分析**

与Text2Room、WorldExplorer等基线比较，在CLIP Score、Inception Score、CLIP‑IQA等指标上均取得显著提升，显示生成的场景在视觉质量和语义一致性上更优。

**⚠️ 局限性**

局限在于依赖VLM的推理质量、生成时间较长、对极端复杂或动态场景的适应性有限，并未解决视频扩散模型的长程几何漂移问题。

---

## 359. FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization

**arXiv ID:** 2603.19835 | [PDF](https://arxiv.org/pdf/2603.19835v1)

**作者:** Chiyu Ma `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**通讯引用:** 7753 | [OpenAlex ID](https://openalex.org/A5057864403)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为FIPO的强化学习算法，用以解决大语言模型在GRPO框架中因粗粒度信用分配导致的推理长度受限问题。

**💡 创新点**

核心创新是引入折扣Future‑KL权重对奖励进行稠密化，并通过软衰减窗口与重要性过滤实现对单步优势的精细重加权，既保留了GRPO的高效性，又获得了PPO级别的粒度。

**🔧 技术方法**

使用了GRPO / DAPO的policy gradient框架，结合Future‑KL、软衰减窗口、影响权重裁剪、token‑level优势重加权等技术；训练过程基于Verl框架实现。

**📊 数据集**

主要使用DAPO公开的17K数学推理数据集进行训练，并在AIME 2024/2025两大数学竞赛数据集上评估。

**📈 对比分析**

与DAPO、DeepSeek‑R1‑Zero‑32B以及o1‑mini等基线对比，FIPO在AIME 2024 Pass@1从50%提升至58%（峰值），平均Chain‑of‑Thought长度从约4,000提升至10,000以上，显著突破了标准GRPO的性能瓶颈。

**⚠️ 局限性**

主要局限在于：长链推理导致训练和推理成本高；验证仅限于数学推理，缺乏对其它开放式任务的泛化；受限于原始DAPO数据集；模型范围仅覆盖干净的基础模型，无法验证在已蒸馏长链模型上的进一步提升。

---

## 360. ReXInTheWild: A Unified Benchmark for Medical Photograph Understanding

**arXiv ID:** 2603.19517 | [PDF](https://arxiv.org/pdf/2603.19517v1)

**作者:** Oishi Banerjee `[一作]` (Harvard Medical School), Pranav Rajpurkar `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含955个多选问答的医学摄影基准ReXInTheWild，覆盖七大临床主题的自然图像

**💡 创新点**

首次统一评估普通相机拍摄的医学照片对多模态语言模型的挑战，提出跨领域错误分类并给出系统性分析

**🔧 技术方法**

使用大型多模态语言模型（Gemini‑3、Claude Opus 4.5、GPT‑5）与专用医学模型MedGemma进行推理；采用GPT‑5自动生成问题与答案，结合人工专家复核

**📊 数据集**

从Biomedica数据集中筛选并过滤出484张符合医学摄影条件的图像，随后生成相应问答对形成最终数据集

**📈 对比分析**

与四款模型比较，Gemini‑3最高达78%准确率，Claude 4.5 72%，GPT‑5 68%，而专用模型MedGemma仅 37%，显示通用模型在此任务上仍具优势

**⚠️ 局限性**

受限于图像来源的研究性偏差、过度严重病情示例、GPT‑5生成的潜在偏倚以及部分医学评估无法仅凭静态图像完成

---

## 361. MagicSeg: Open-World Segmentation Pretraining via Counterfactural Diffusion-Based Auto-Generation

**arXiv ID:** 2603.19575 | [PDF](https://arxiv.org/pdf/2603.19575v1)

**作者:** Kaixin Cai `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 23330 | [OpenAlex ID](https://openalex.org/A5047878798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MagicSeg，利用扩散模型自动生成多类别的合成语义分割数据集（包含像素级掩码和对照（counterfactual）图像），并在此数据上训练开放世界语义分割模型。

**💡 创新点**

1) 基于扩散模型的 counterfactual 生成管线，能够同时产生正样本与负样本；2) 通过 ChatGPT 生成高质量、细节丰富的文本提示，提升图像多样性；3) 结合 Grounding DINO 与 SAM 自动生成像素级标签；4) 设计类别随机采样（Category Random Sampling）和对照对比损失（Counterfactual Contrastive Loss）解决类别不平衡与伪标签噪声。

**🔧 技术方法**

使用 Stable Diffusion 生成图像、ChatGPT 生成文本、Grounding DINO 检测、SAM 分割、CLIP（ZegCLIP）做基准模型，辅以 focal、Dice 以及余弦对比损失。

**📊 数据集**

训练集为自建的 380k 条合成数据，覆盖 1205 个类别（来源于 LVIS、PASCAL VOC 词表）；测试使用 PASCAL VOC、PASCAL Context、COCO、LVIS 以及 OpenImages‑V7。

**📈 对比分析**

与 Grounded SAM、ODISE、OpenSeg、MaskCLIP 等方法对比，MagicSeg 在 PASCAL VOC 62.9% mIoU、COCO 40.2% mIoU、PASCAL Context 26.7% mIoU，显著优于基线且接近甚至超过有真实标注的模型；在 OpenImages‑V7 的零样本点预测中 p‑mIoU 45.6%，仅略低于 GEM‑SAM‑MetaCLIP。

**⚠️ 局限性**

1) 每张合成图像最多包含两类，可能限制多类学习；2) 生成的像素标签仍可能含噪声，受生成模型质量影响；3) 对未知类别掩码的挖掘有限，仍需进一步改进。

---

## 362. Improving Image-to-Image Translation via a Rectified Flow Reformulation

**arXiv ID:** 2603.20186 | [PDF](https://arxiv.org/pdf/2603.20186v1)

**作者:** Satoshi Iizuka `[一作]` (University of Tsukuba), Kazuhiro Fukui `[通讯]` (University of Tsukuba)

**通讯引用:** 2131 | [OpenAlex ID](https://openalex.org/A5101567870)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Image-to-Image Rectified Flow Reformulation (I2I-RFR)，通过在输入中加入噪声扰动的目标图像并使用 t‑加权像素损失，将传统 I2I 回归网络重构为连续时间 ODE 修正模型。

**💡 创新点**

创新点在于：① 将监督 I2I 回归与 Rectified Flow 结合，形成可插拔、无需额外判别器或潜在编码器的连续时间修正框架；② 只需在输入通道上扩展 1–2 倍即可直接复用现有回归 backbone；③ 通过 Beta 分布采样 t 并采用简单的欧拉 ODE 求解器实现少步推演，兼顾速度与质量。

**🔧 技术方法**

技术细节包括：直接目标预测（非 velocity 预测）；t‑加权 L1 损失；Beta(2,1) 采样 t；显式欧拉 ODE 求解器（N=3 步）；对原有 backbone 仅做通道扩展；不使用时间嵌入或 GAN 对抗。

**📊 数据集**

评测数据集覆盖多任务：图像超分（Set5/Set14/BSD100/Urban100）、图像去模糊（RealBlur-J）、低光模糊增强（LOLBlur）、水下图像增强（LSUI、UIEB）、视频恢复（REDS、DAVIS）。

**📈 对比分析**

与原始回归模型、Palette（diffusion）以及多种强基线进行对比；I2I-RFR 在所有任务上显著提升 LPIPS（细节保留）且 PSNR/SSIM 维持或轻微下降；在超分和低光增强等任务中，I2I-RFR 的表现甚至超过传统 diffusion 方案；视频恢复中也匹敌或优于 GAN 训练版本。

**⚠️ 局限性**

局限性：对已高度优化的任务专用 backbone（如 SFGNet）效果不稳定；v‑prediction 方式或加入时间嵌入往往适得其反；在极端噪声或极端失真场景下改进有限；需要额外训练时间和通道扩展，且 t_min、Beta 采样需手动调参。

---

## 363. ATHENA: Adaptive Test-Time Steering for Improving Count Fidelity in Diffusion Models

**arXiv ID:** 2603.19676 | [PDF](https://arxiv.org/pdf/2603.19676v1)

**作者:** Mohammad Shahab Sepehri `[一作]` (University of Southern California), Mahdi Soltanolkotabi `[通讯]` (University of Southern California)

**通讯引用:** 3519 | [OpenAlex ID](https://openalex.org/A5046962187)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种模型无关、测试时自适应的引导框架 ATHENA，用于提升文本到图像扩散模型在满足指定对象计数时的精度。

**💡 创新点**

创新点在于通过在中间生成步骤估计对象数并利用提示级引导在早期修正生成轨迹，且通过单次反馈自适应调整引导强度，无需改造模型或再训练。

**🔧 技术方法**

采用前向推理的提示引导技术、GroundingDINO 对中间图像的计数估计、以及自适应的强度调节算法，实现轻量级的测试时控制。

**📊 数据集**

使用 CoCoCount、CoCoCount‑E（扩展版）以及新构造的 ATHENA 挑战性计数数据集进行评估。

**📈 对比分析**

与未引导采样、CountGen 与 Counting Guidance 等基线比较，ATHENA 在三大扩散模型上实现最高 22% 的精确计数提升；在 ATHENA 数据集上速度比 CountGen 快 2.5×、内存比 CountGen 少 4×，与 Counting Guidance 相比在相同运行时获得约 2× 的计数准确率。

**⚠️ 局限性**

局限性包括对中间计数器（GroundingDINO）准确性的依赖；在计数超过 10 的极端场景下性能下降；当引导强度选择不当时可能出现过度或不足校正的情况。

---

## 364. Any-Subgroup Equivariant Networks via Symmetry Breaking

**arXiv ID:** 2603.19486 | [PDF](https://arxiv.org/pdf/2603.19486v1)

**作者:** Abhinav Goel `[一作]` (Massachusetts Institute of Technology), Ningyuan Huang `[通讯]` (Flatiron Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种通用的 Any-Subgroup Equivariant Network (ASEN)，通过在输入中加入具有特定自对称性的辅助特征，实现对任意子群的等变性，支持在同一模型中处理不同的对称性；

**💡 创新点**

创新点包括：①在保持高层等变性的同时，仅通过调节辅助输入实现子群等变性；②利用 2-闭包概念在可行的计算复杂度下近似得到所需子群的自对称性；③在理论上证明该框架可模拟任意等变 MLP 并继承基模型的普适性；

**🔧 技术方法**

主要技术手段包括：基于全排列等变的图神经网络作为基模型；通过构造包含节点/边特征的超图来实现子群的自对称性；使用 2-闭包算法生成近似自对称的边特征；以及对输入进行位置/边特征编码来实现对称性破裂；

**📊 数据集**

实验数据集包括：Human3.6M（人体姿态估计）、METR-LA（交通流预测）、Pathfinder-64（图像分类），以及若干合成序列任务（如 Palindrome、Cyclic Sum 等）；

**📈 对比分析**

与传统的单一子群等变 MLP、非等变模型以及多任务/迁移学习基线进行对比。实验表明：在姿态估计中，弱稀疏边特征可获得最优 P‑MPJPE；在交通预测中，选择合适的子群可略优于全排列等变；在 Pathfinder 任务中，引入局部对称性可提升准确率并减少参数；在合成序列任务中，子群等变模型的收敛速度更快、最终误差更低；在多任务与迁移学习实验中，ASEN 在低数据量下显著提升性能，且迁移预训练能有效加速收敛；

**⚠️ 局限性**

局限性：需要预先知道子群结构，2‑闭包近似可能引入额外对称导致精度下降；对局部对称性处理不够完善；输入为固定对称的情况需对齐标签，增加预处理成本；当子群规模较大或 2‑闭包差距显著时，模型可能无法完全捕获目标对称性；

---

## 365. Locality Sensitive Hashing in Hyperbolic Space

**arXiv ID:** 2603.19724 | [PDF](https://arxiv.org/pdf/2603.19724v1)

**作者:** Chengyuan Deng `[一作]` (Rutgers University), Cheng Xin `[通讯]` (Rutgers University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了首个针对双曲空间的局部敏感哈希（LSH）方案，给出了二维和高维双曲空间的哈希构造，并分析了理论性能上界与下界。

**💡 创新点**

创新点在于：首次将随机测地线分割与低维投影相结合，构造了双曲空间的LSH；在二维时实现了 ρ≤1/c 的上界，在高维时得到 ρ≤1.59/c；同时给出了 ρ≈1/c² 的下界，形成了理论性能范围。

**🔧 技术方法**

采用的技术包括 Poincaré 盘模型与测地线测度、Crofton 公式、Johnson–Lindenstrauss 维度缩减、正态分布的 p‑stable 投影以及曲率与距离分析。

**📊 数据集**

实验使用合成的双曲球面随机点（R=log 199），在二维及高维（10、100、1000）中进行实验，并尝试不同半径 R 的情形。

**📈 对比分析**

通过对比 p₁、p₂ 与 ρ 与理论上界的关系，实验发现实际 ρ 远小于 1/c，且随着维度增大和 c 增大 ρ 逐渐下降，验证了哈希方案的有效性。

**⚠️ 局限性**

局限性包括：上界与下界之间仍存在 1/c 与 1/c² 的鸿沟；理论保证仅在 c≥1.59 时成立；缺乏数据依赖型 LSH 的研究；实验仅在合成数据上验证，未覆盖真实图谱等实际应用场景。

---

## 366. Neural Uncertainty Principle: A Unified View of Adversarial Fragility and LLM Hallucination

**arXiv ID:** 2603.19562 | [PDF](https://arxiv.org/pdf/2603.19562v1)

**作者:** Dong-Xiao Zhang `[一作]` (Northwest Institute of Nuclear Technology), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 31905 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出神经不确定性原理（NUP），统一解释视觉模型对抗脆弱性与大型语言模型幻觉，并基于该原理设计单次反向传播的余弦相关探针（CC-Probe），以及通过 ConjMask/LogitReg 对抗鲁棒性和 prefill 阶段风险评分实现幻觉检测与提示选择。

**💡 创新点**

创新点在于将输入与损失梯度视为共轭观测，给出不可逾越的不确定性界限，并将两类失效统一为同一几何预算；同时提出可直接观测的 CC‑Probe 作为单次反向传播的诊断工具，并通过输入‑梯度耦合遮蔽与输出层正则化实现无对抗训练的鲁棒提升。

**🔧 技术方法**

采用量子算子不确定性理论（罗伯逊–施罗丁格不等式）构建 NUP；利用损失加权状态、输入‑梯度余弦探针、ConjMask（输入‑梯度耦合遮蔽）、LogitReg（输出层正则化）以及 prefill 风险评分；评估使用标准对抗攻击（PGD、AutoAttack）和幻觉检测指标（AUROC）。

**📊 数据集**

视觉方面使用 CIFAR‑10、Tiny‑ImageNet‑200、ImageNet‑100；LLM 方面使用 Benchmark‑500 及其 Perturbation‑100 数学推理题集。

**📈 对比分析**

与传统对抗训练（TRADES）和现有幻觉检测方法比较，ConjMask 在不使用对抗训练的情况下，在 PGD/AutoAttack 下提升鲁棒性达 70%+；Prefill Risk‑Cos 在幻觉检测 AUROC 约 0.69，提示选择 Top‑1 Hit 率 0.76，显著优于其它预填评估指标。

**⚠️ 局限性**

局限性包括：仅在损失显著非零的“边界层”有效，对极低损失或近似一对一映射情况不敏感；CC‑Probe 为近似代理，需单次反向传播，且对不同攻击目标仍需多通道；方法依赖单一损失形式（CE/shifted NLL），在其他任务或多模态场景中需进一步验证。

---

## 367. LLM-Enhanced Semantic Data Integration of Electronic Component Qualifications in the Aerospace Domain

**arXiv ID:** 2603.20094 | [PDF](https://arxiv.org/pdf/2603.20094v1)

**作者:** Antonio De Santis `[一作]` (Politecnico di Milano), Emanuele Della Valle `[通讯]` (Politecnico di Milano)

**通讯引用:** 4710 | [OpenAlex ID](https://openalex.org/A5015694017)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种基于虚拟知识图（VKG）并结合LLM的数据清洗与向量检索的混合管道，实现了卫星电子元件资格数据的统一检索与整合。

**💡 创新点**

创新点在于将LLM用于半自动化的数据清洗、制造商规范化及零件号抽取，并在VKG上实现符号查询与向量检索的组合，提升了查询精度与效率。

**🔧 技术方法**

采用技术包括GPT‑OSS‑120B LLM、Ontop VKG、SPARQL、向量嵌入（OpenAI embeddings）、向量检索，以及与之对比的RAG方法。

**📊 数据集**

数据集来自Thales Alenia Space的 PLM‑DB 与资格目录 QC，包含数千条元件与资格记录，经过清洗后形成统一视图。

**📈 对比分析**

与仅基于 RAG 的 LLM 方法相比，VKG+LLM 在直接及相似资格检索上 Precision/Recall 均超过 90%，且在大规模（>5000 件）时累计人力成本下降 70% 以上；RAG 在相似资格上 F1≈94% 但在替代资格上仅≈62%，并且对大上下文支持不足。

**⚠️ 局限性**

限制主要是高昂的初始工程投入（约 60 人日），依赖域专家参与数据清洗与本体建模，且 LLM 仍需人机验证；在极大规模或多机构场景下的安全与权限管理尚未完善。

---

## 368. FrameNet Semantic Role Classification by Analogy

**arXiv ID:** 2603.19825 | [PDF](https://arxiv.org/pdf/2603.19825v1)

**作者:** Van-Duy Ngo `[一作]`, Miguel Couceiro `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出基于对比推理的语义角色分类方法，将多类分类问题转化为二类判别，再通过对比推理恢复原始角色标签。

**💡 创新点**

创新点在于将结构映射理论（SMT）引入FrameNet语义角色标注，利用谓词-成分对的二分类模型实现无监督的角色迁移，同时保持模型可解释性和高能效。

**🔧 技术方法**

主要技术包括上下文与谓词-成分的句子嵌入、轻量级前馈网络二分类器、随机采样的对比推理转移以及基于推理结果的多标签恢复。

**📊 数据集**

实验使用FrameNet 1.7数据集，采用其官方train/dev/test划分，并构造正负对比样本以平衡训练集。

**📈 对比分析**

与现有SRL系统（如Lin等、Softmax-margin、Graph-Transformer等）对比，本文在整体SRL准确率上从48.95%提升至49.81%/59.35%（取决于框架），在精确率、召回率、F1上分别达到80.77/79.26/79.17，显著优于基线。

**⚠️ 局限性**

局限性包括对未见框架/角色的泛化仍有限，缺失训练数据时性能下降；方法依赖标注好的谓词-成分对，扩展到其他任务需要进一步验证；目前实现仍需GPU，模型规模可进一步压缩。

---

## 369. FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse and Prover-Effective Autoformalization

**arXiv ID:** 2603.19828 | [PDF](https://arxiv.org/pdf/2603.19828v1)

**作者:** Haijian Lu `[一作]` (Xidian University), Jing Liu `[通讯]` (Xidian University)

**通讯引用:** 34401 | [OpenAlex ID](https://openalex.org/A5100374963)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 FormalEvolve，一种在固定生成器调用预算下的编译门控进化搜索框架，用于 Lean 4 自动形式化。

**💡 创新点**

创新点在于将编译门控、LLM 驱动的多样化变异、带边界的编译与语义修复以及无调用的 AST 重写相结合，构建多样化可证明性语句库，显著提升语义覆盖与证明效率。

**🔧 技术方法**

采用 LLM（Kimina、Qwen3）、CriticLean 语义评判器、EvolAST 样式的 AST 重写、预算感知的父母选择与归档搜索、Lean 4 编译检查，以及 Goedel‑Prover‑V2 证明器。

**📊 数据集**

使用 ProofNet（Lean 4 端口）和 CombiBench 两个基准数据集。

**📈 对比分析**

与无归档采样、编译修复、语义修复等基线对比，FormalEvolve 在 T=100 时在 CombiBench 的 SH@100 提升至 0.58，Gini 系数下降，证明成功率提升到 13/100；在 ProofNet 上也实现了显著的语义覆盖提升。

**⚠️ 局限性**

局限在于语义评判器依赖 LLM，可能产生评判‑证明不匹配；证明器和预算限制决定上限；对领域迁移的鲁棒性有限；评估仅针对固定预算和特定证明器。

---

## 370. HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated Language Model Fine-Tuning

**arXiv ID:** 2603.19278 | [PDF](https://arxiv.org/pdf/2603.19278v1)

**作者:** Bartosz Trojan `[一作]` (Upper-Secondary Schools of Communications in Cracow), Filip Gębala `[通讯]` (Upper-Secondary Schools of Communications in Cracow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在RoBERTa上使用LoRA和基于超网络的低秩适配器进行参数高效微调，以提升模型的概率校准性能。

**💡 创新点**

创新点在于提出共享超网络生成低秩更新，形成跨层结构耦合，并发现固定A矩阵可以显著改善校准。

**🔧 技术方法**

使用技术包括LoRA、超网络（MLP/Transformer）、RoBERTa预训练模型、GLUE基准评估以及ECE、CECE、MCE、ACE、TACE、Brier等多种校准指标。

**📊 数据集**

实验数据集为GLUE基准中的CoLA、SST‑2、QNLI、MRPC、RTE和MNLI六个二分类或三分类任务。

**📈 对比分析**

通过与全参数微调和标准LoRA的对比，使用上述多指标评估；LoRA在大多数任务上与全微调的校准相当或略优，超网络在CoLA上取得略优的校准但任务性能略降；固定A矩阵能显著降低ECE，却伴随准确率下降。

**⚠️ 局限性**

主要局限包括校准改进受任务影响显著、超网络生成未能普遍提升校准、固定A矩阵导致性能下降、仅评估二分类GLUE任务、缺乏对分布外或多分类任务的验证。

---

## 371. The Robot's Inner Critic: Self-Refinement of Social Behaviors through VLM-based Replanning

**arXiv ID:** 2603.20164 | [PDF](https://arxiv.org/pdf/2603.20164v1)

**作者:** Jiyu Lim `[一作]` (ETRI), Kwanghyun Park `[通讯]` (KwangWoon University)

**通讯引用:** 1168 | [OpenAlex ID](https://openalex.org/A5010089204)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CRISP 框架，让机器人通过 VLM 作为社会批评家，自主生成、评估并重规划低层关节控制的社交行为

**💡 创新点**

关键创新在于将 VLM 作为人类式社会评论者实现 generate‑evaluate‑regenerate 循环，以及利用机器人结构文件直接推断关节约束生成低层控制命令，提升跨平台灵活性

**🔧 技术方法**

核心技术包括 LLM（如 GPT‑4o）解析 MJCF 文件、VLM（同模型）进行视觉逻辑推理、链式思维（CoT）生成关节值、Reward‑based Adaptive Search（RAS）迭代重规划

**📊 数据集**

使用机器人结构文件（MJCF）作为输入；在实验中采用五种机器人（Everyday、Stretch 3、Open Mini Duck、TIAGo、Unitree G1）与 20 个情景的视频/视觉日志；用户评价数据来自 50 位参与者的 Likert 评分

**📈 对比分析**

与改进版 GenEM 以及 CRISP 去掉重规划的版本进行对比，CRISP 在整体用户偏好上平均得分 4.5（vs 3.4 / 3.79），差异在 Wilcoxon‑signed‑rank 试验下显著；在 Ablation 试验中，完整模型在 5 次运行中平均成功 4.6 次，所需 VLM 视图 10.2 张，重规划 3.4 次，显著优于仅单步或全局重规划方案

**⚠️ 局限性**

局限主要包括：VLM 仅基于静态图像评估，缺乏时间连续性；对主观情绪场景的适应仍不足；无法处理全身动态（行走等）；重规划过程耗时（10–20 分钟），目前仅适合离线行为创作

---

## 372. Can Structural Cues Save LLMs? Evaluating Language Models in Massive Document Streams

**arXiv ID:** 2603.19250 | [PDF](https://arxiv.org/pdf/2603.19250v1)

**作者:** Yukyung Lee `[一作]` (Boston University), Susik Yoon `[通讯]` (Korea University)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5083900503)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型在多事件实时新闻流环境中的表现，构建了涵盖 605 个事件、15,354 篇文档的 StreamBench 基准，并引入结构化线索对模型瓶颈进行诊断。

**💡 创新点**

创新点在于①提出了可控制文档量的多事件流式评测框架；②利用结构化线索（人物、地点、结果、事件属性）作为诊断手段，系统剖析模型在聚类、时序问答和摘要三大任务中的失败原因。

**🔧 技术方法**

采用指令微调的大语言模型（规模从 1B 至 123B）、滑动窗口抽样策略以及手工/LLM 自动抽取的结构化线索，对三类任务分别进行实验。

**📊 数据集**

使用 2025 年与 2016 年六个新闻故事（California Wildfire、South Korea Martial Law、60th US Election、Summer Olympics、Israel‑Palestine Conflict、58th US Election）共 605 个事件、15,354 篇文档、1,009 组 QA、605 组摘要、200 组聚类标签构成的 StreamBench。

**📈 对比分析**

通过对比原始输入与加入结构线索的输入，在 B³ F1、QA 准确率、ROUGE‑L、METEOR 等指标上评估；结构线索分别提升聚类 +4.37%、QA +9.63%，摘要覆盖和可信度显著提升但 ROUGE‑L 仅 <1%；模型规模越大，结构线索的增益越小。

**⚠️ 局限性**

局限性包括：结构线索需离线预先生成，缺乏实时构建机制；时序推理与实体状态追踪仍难以解决；摘要生成仍缺乏连贯性；基准随 LLM 知识截止点演进而逐渐过时，且未考虑任务之间的顺序依赖。

---

## 373. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**arXiv ID:** 2603.19312 | [PDF](https://arxiv.org/pdf/2603.19312v1)

**作者:** Lucas Maes `[一作]` (Mila and Université de Montréal), Randall Balestriero `[通讯]` (Brown University)

**通讯引用:** 915 | [OpenAlex ID](https://openalex.org/A5047293370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 LeWorldModel，一种可直接从原始像素端到端训练的 Joint Embedding Predictive Architecture，用于学习紧凑的潜在世界模型并实现控制与规划。

**💡 创新点**

创新点在于仅使用两项损失（下一步潜在预测损失与 SIGReg 正则化）即可稳定训练，消除对 EMA、stop‑gradient、预训练编码器或辅助监督的需求，并显著简化超参数搜索。

**🔧 技术方法**

核心技术包括 Vision Transformer 编码器、Transformer 预测器、AdaLN 动作调制、SIGReg 正则化（通过 Epps‑Pulley 正态性检验实现潜在分布逼近 isotropic Gaussian）以及跨时间的自回归预测；规划采用基于 MPC 的 CEM 采样。

**📊 数据集**

使用离线收集的多任务数据集（Push‑T、OGBench‑Cube、Two‑Room、Reacher 等 2D/3D 控制环境），仅含原始像素和动作，无奖励或状态标签。

**📈 对比分析**

与 DINO‑WM、PLDM、GCBC、GCIVL、GCIQL 等基线比较，LeWM 在 2D/3D 任务上取得更高或相当的成功率（Push‑T 成功率提升 18%，与 DINO‑WM 接近），规划速度比 DINO‑WM 快 48 倍，训练时间与参数量（≈15M）显著低于现有方法；训练过程更平稳、超参数更少。

**⚠️ 局限性**

局限性包括：仅能在短期规划上表现良好；依赖覆盖充分的离线数据，数据多样性不足时 SIGReg 可能失效；不使用动作或奖励信号，仍需收集动作标签；对极低复杂度环境的高维正态化正则化可能导致潜在结构不足。

---

## 374. Radar-Inertial Odometry with Online Spatio-Temporal Calibration via Continuous-Time IMU Modeling

**arXiv ID:** 2603.19958 | [PDF](https://arxiv.org/pdf/2603.19958v1)

**作者:** Vlaho-Josip Štironja `[一作]` (University of Zagreb), Ivan Petrović `[通讯]` (University of Zagreb)

**通讯引用:** 4289 | [OpenAlex ID](https://openalex.org/A5001473761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在因子图框架下的雷达-惯导松耦合里程计，能够在线同时估计雷达-IMU的空间外参与时间偏移。

**💡 创新点**

首次在同一因子图中引入连续时间B样条惯性建模，构造雷达速度因子，使系统在大时间偏移、无硬件同步且无局部恒加速假设下仍能可靠收敛。

**🔧 技术方法**

因子图优化（GTSAM iSAM2）、Uniform Cubic B-splines连续时间建模、雷达自车速度估计、Huber损失、IMU预积分因子。

**📊 数据集**

使用公开雷达-惯导数据集EKF-RIO-TC（非同步）和ICINS（硬件同步）进行评估。

**📈 对比分析**

与EKF-RIO、EKF-RIO-TC、RIO-T等基线对比；在未同步数据集上平均RPE提升18.3%，相对无校准FGO提升33%；在同步数据集上性能与基线相当，且将校准参数注入基线可将RPE降低多达68%。

**⚠️ 局限性**

需要足够旋转激励才能收敛外参；在静止或低动态场景外参可能不收敛或误差增大；未实现循环闭环或原始雷达点云的紧耦合；B-spline参数需手动调节。

---

## 375. The Verifier Tax: Horizon Dependent Safety Success Tradeoffs in Tool Using LLM Agents

**arXiv ID:** 2603.19328 | [PDF](https://arxiv.org/pdf/2603.19328v1)

**作者:** Tanmay Sah `[一作]` (Harrisburg University of Science and Technology), Kayden Jordan `[通讯]` (Harrisburg University of Science and Technology)

**通讯引用:** 2177 | [OpenAlex ID](https://openalex.org/A5103545444)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在多步工具调用的LLM代理中，运行时安全中介对任务完成的影响。

**💡 创新点**

创新点在于提出安全成功率（SSR）与不安全成功率（USR）的细分，揭示“安全-能力差距”与“验证税”，并提出严格基础门控（Strict Grounding Gates）以防止完整性泄漏。

**🔧 技术方法**

采用三种代理架构（直接工具调用、Triad、Triad‑Safety）和两款开放权重模型（GPT‑OSS‑20B、GLM‑4‑9B），利用可视化验证器、规划-行动-验证循环实现安全中介。

**📊 数据集**

使用τ‑bench的航空（Airline）和零售（Retail）两个交易域数据集进行实验，涵盖约2,970条交互轨迹。

**📈 对比分析**

通过比较成功率（SR）、安全成功率（SSR）、不安全成功率（USR）、干预频率、恢复率和验证器调用/Token开销等指标，发现尽管安全中介可拦截高达94%的违规行为，但安全成功率仍低于5%，恢复率仅在21%–5%之间；同时验证器导致的Token开销平均提升2.0–2.8倍。

**⚠️ 局限性**

主要限制包括：仅测试9B/20B模型，模型规模可能影响恢复表现；对话轮数受限（GPT 15轮，GLM 30轮）可能限制恢复机会；验证器为二元拒绝，缺乏结构化恢复建议；高开销导致的“验证税”与可扩展性问题；以及静态安全规则未能捕捉所有边缘情况。

---

## 376. Morphology-Consistent Humanoid Interaction through Robot-Centric Video Synthesis

**arXiv ID:** 2603.19709 | [PDF](https://arxiv.org/pdf/2603.19709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 377. Physion-Eval: Evaluating Physical Realism in Generated Video via Human Reasoning

**arXiv ID:** 2603.19607 | [PDF](https://arxiv.org/pdf/2603.19607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. Agentic Harness for Real-World Compilers

**arXiv ID:** 2603.20075 | [PDF](https://arxiv.org/pdf/2603.20075v1)

**作者:** Yingwei Zheng `[一作]` (Southern University of Science and Technology), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14365 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向LLVM中间端的Agentic Harness，集成了工具调用接口、可复现的Bug基准（LLVM‑AutoFix）以及一个最小化代理，用以辅助LLM修复编译器缺陷。

**💡 创新点**

首次提供专门针对LLVM的工具集与基准，并通过定制的四阶段ReAct代理显著提升LLM在编译器Bug修复的效果。

**🔧 技术方法**

结合大型语言模型的ReAct框架、LLM工具调用、LLVM工具链（opt、alive2、llvm-lit、gdb等）以及自定义的四阶段代理流程。

**📊 数据集**

使用LLVM‑AutoFix基准，包含334条可复现的中间端Bug（崩溃/误编译），按难度划分为easy/medium/hard，并提供重现器、回归测试和实时更新的live子集。

**📈 对比分析**

与SWE‑bench Verified基准和四款前沿LLM（GPT‑5、Gemini 2.5 Pro、DeepSeek V3.2、Qwen 3 Max）以及GPT‑4o进行pass@1和专家审查对比；结果显示LLM在LLVM Bug上相较常规软件Bug下降约60%，自定义代理相比基准提升约22%，但真实有效率仍不足20%。

**⚠️ 局限性**

受限于LLVM回归测试不足、工具调用格式导致的Token/Tool Limit失败、LLM常见错误（误定位、错误修复、绕过断言）以及未覆盖性能、前端/后端Bug等问题。

---

## 379. An Adaptive Machine Learning Framework for Fluid Flow in Dual-Network Porous Media

**arXiv ID:** 2603.19561 | [PDF](https://arxiv.org/pdf/2603.19561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 380. From Flat to Structural: Enhancing Automated Short Answer Grading with GraphRAG

**arXiv ID:** 2603.19276 | [PDF](https://arxiv.org/pdf/2603.19276v1)

**作者:** Yucheng Chu `[一作]` (Michigan State University), Hui Liu `[通讯]` (Michigan State University)

**通讯引用:** 30380 | [OpenAlex ID](https://openalex.org/A5100654857)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估基于图检索增强生成（GraphRAG）的自动短答分级框架

**💡 创新点**

通过结构化知识图谱实现多跳推理，克服传统向量检索的知识碎片化问题

**🔧 技术方法**

使用Microsoft GraphRAG（社区摘要）和HippoRAG（神经符号关联检索）以及LLM生成与验证

**📊 数据集**

Next Generation Science Standards (NGSS) 的六项科学分级任务数据集

**📈 对比分析**

与传统RAG基线相比，GraphRAG在所有维度上均有显著提升，HippoRAG在科学与工程实践（SEP）维度达到84.8%准确率，明显优于基线的4.3%

**⚠️ 局限性**

仍受限于知识图谱构建成本、对特定领域的可迁移性不足，以及对低资源概念的覆盖不完整

---

## 381. Discovery of Decision Synchronization Patterns from Event Logs

**arXiv ID:** 2603.19879 | [PDF](https://arxiv.org/pdf/2603.19879v1)

**作者:** Tijmen Kuijpers `[一作]` (Eindhoven University of Technology), Remco Dijkman `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 7597 | [OpenAlex ID](https://openalex.org/A5024303560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种从事件日志中发现多案例同步决策模式的方法，利用Petri网与决策树结合，实现对供应链场景下的优先级、阻塞、批量等待和选择等同步模式的约束自动挖掘。

**💡 创新点**

创新点在于首次将跨案例约束与Petri网建模相结合，并通过特征提取与决策树学习来识别同步决策规则，填补了现有过程挖掘技术无法捕捉多实例同步约束的空白。

**🔧 技术方法**

使用技术包括简化的彩色Petri网建模、事件日志重放、特征抽取函数、决策树分类（Gini、阈值筛选）以及对抽象的同步模式进行约束推导。

**📊 数据集**

实验数据集为人工构造的单模式与多模式供应链过程日志，包含4种同步模式；同时提供10次随机化实验的完整日志与模型，数据托管于公开GitHub仓库。

**📈 对比分析**

评估方式通过比较挖掘出的约束与已知模型中手工设定的约束是否一致，单模式实验几乎全部匹配，误差仅在阈值离散化导致；多模式实验亦全能发现，准确率随样本量变化而波动，整体表现良好。

**⚠️ 局限性**

局限性包括：对样本量敏感；阈值选择需要手工设定；仅验证了供应链场景，其他业务领域尚待扩展；未处理模式重叠、噪声日志与随机违反的情况。

---

## 382. On the size of k-irreducible triangulations

**arXiv ID:** 2603.20030 | [PDF](https://arxiv.org/pdf/2603.20030v1)

**作者:** Vincent Delecroix `[一作]` (University of Bordeaux), Arnaud de Mesmay `[通讯]` (University Gustave Eiffel)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究面向 k-不可约三角剖分（k-irreducible triangulations），证明其在给定 genus g 的表面上最多包含 O(k²g) 个三角形（即 O(k²g) 条边），并证明此上界是最佳的。

**💡 创新点**

创新点在于：①将三角剖分与曲线系统（geodesically k‑covered system）联系起来，通过媒介图和曲线交叉数的最小化来捕捉度量结构；②利用曲线系统的填充性质和树宽/曲线交叉数的结构定理，得到从 k‑覆盖到 O(k²g) 的紧致上界；③扩展到非可定向表面，给出相应的上界。

**🔧 技术方法**

主要技术包括：曲线系统与媒介图的构造、最小位置（tight）与 geodesically k‑covered 的定义、树宽与图的交叉数估计、可定向双覆盖、以及曲线在统一覆盖中的长度变换。

**📊 数据集**

本工作为理论性研究，没有使用实验数据集，而是给出严谨的组合与拓扑证明。

**📈 对比分析**

与先前最优上界 k^O(k) g² 的结果相比，本文给出了线性于 g 的 O(k²g) 上界，并证明其最优（构造可达到 Θ(k²g)）。

**⚠️ 局限性**

限制包括：①常数 966（以及非可定向表面时的 3846）尚未最优；②在非可定向（尤其是射影平面）情形下的构造与上界仍有一定差距；③证明仅适用于闭表面（无边界）且假设 k≥3。

---

## 383. Planning Autonomous Vehicle Maneuvering in Work Zones Through Game-Theoretic Trajectory Generation

**arXiv ID:** 2603.19556 | [PDF](https://arxiv.org/pdf/2603.19556v1)

**作者:** Mayar Nour `[一作]` (University of Western Ontario), Mohamed H. Zaki `[通讯]` (University of Western Ontario)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出工作区中自驾车变道的博弈理论框架，并在SUMO微观仿真环境中评估其安全性。

**💡 创新点**

将工作区必需变道视为非合作博弈，设计多目标效用函数（安全、进展、交通法规），通过Nash均衡生成兼顾安全与效率的轨迹，并采用再规划实现动态适应。

**🔧 技术方法**

使用博弈理论（非合作博弈、Nash均衡）、递归视窗控制、SUMO仿真、SL2015车道变更模型、车道跟随模型（C‑ACC / IDM）等。

**📊 数据集**

采用Dublin M50 7km 高速公路真实网络及其已校准的车流参数，并在此基础上创建长度1km的工作区闭塞车道场景。

**📈 对比分析**

与不含博弈的基线变道模型在L2和L4自动化水平下进行10个随机种子对比，评估冲突频率和TTC指标。结果显示，L2下冲突率降低约35%，TTC 5%分位数提升，显著；L4下差异不显著。

**⚠️ 局限性**

局限在于仅考虑两车交互，未包含领车；仅测试纯AV环境，未考虑混合交通；未模拟实际通信与感知不完全信息等情形。

---

## 384. Better Sampling Bounds for Restricted Delaunay Triangulations and a Star-Shaped Property for Restricted Voronoi Cells

**arXiv ID:** 2603.19826 | [PDF](https://arxiv.org/pdf/2603.19826v1)

**作者:** Jonathan Richard Shewchuk `[一作]` (University of California), Jonathan Richard Shewchuk `[通讯]` (University of California)

**通讯引用:** 11039 | [OpenAlex ID](https://openalex.org/A5064244744)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

针对光滑闭曲面上的限制Delaunay三角剖分（RDT），本文给出了更紧的取样条件，并证明当取样密度满足ε≤0.3245时，RDT的底层空间与曲面同胚；同时也给出了ε‑Voronoi取样的改进阈值ε≤0.4132。

**💡 创新点**

主要创新在于显著提升了取样阈值：相比Dey的0.18‑sample，本文将阈值提升至0.3245，样本点需求减少约3.25倍；相比Cheng等的0.09‑Voronoi sample，阈值提升至0.4132，样本点需求减少约21倍；此外，引入了新的Voronoi单元性质证明（单元为圆盘、投影星形）来实现这些改进。

**🔧 技术方法**

使用了局部特征尺寸（local feature size）、Medial轴分析、正交投影与切平面几何、Edelsbrunner‑Shah的Topological Ball Theorem以及一系列角度与距离的不等式，构造新的几何和拓扑不等式以证明Voronoi单元的拓扑性质。

**📊 数据集**

本文为理论证明性工作，没有使用具体实验数据集。

**📈 对比分析**

通过与已有结果（Dey的0.18-sample和Cheng等的0.09-Voronoi sample）的阈值对比，展示了取样阈值提升3.25倍和21倍，从而显著减少了所需的采样点数；性能提升体现在理论上所需样本量的下降。

**⚠️ 局限性**

局限性包括：仅适用于光滑闭曲面在ℝ³中的情形；需要假设没有顶点落在曲面上、没有切点交叉；在更高维度下的拓扑同胚结果无法直接推广；此外，文章未给出实现细节或算法复杂度，未讨论实际构造RDT的计算效率。

---

## 385. EgoForge: Goal-Directed Egocentric World Simulator

**arXiv ID:** 2603.20169 | [PDF](https://arxiv.org/pdf/2603.20169v1)

**作者:** Yifan Shen `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 992 | [OpenAlex ID](https://openalex.org/A5043962698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于扩散模型的自我导向第一人称视频模拟器，可仅凭一张自我视角图像、一个高层指令以及可选的外部视角图像生成连贯的、目标驱动的全景视频。

**💡 创新点**

核心创新点包括：①无需密集运动标签或多视角同步视频；②通过将VGGT 3D几何特征与扩散潜在空间对齐，实现几何意识的空间一致性；③引入 VideoDiffusionNFT——轨迹层奖励引导的微调机制，在采样过程中同时优化目标完成、场景稳定性、时间因果性和感知质量。

**🔧 技术方法**

使用的主要技术包括：基于预训练视频自编码器的潜在扩散变压器（Diffusion Transformer）、几何弱监督（VGGT 与 DINO 对齐）、低秩适配（LoRA）微调、以及多奖励函数（goal、env、temp、per）构成的奖励引导优化。

**📊 数据集**

数据集：自建的 EgoForge 基准，采集自 Nymeria 与 Ego-Exo4D，包含 15,000 条训练样本、100 条测试样本，并提供细粒度手物交互、物体状态变化与步骤级动作注释。

**📈 对比分析**

在 EgoForge 基准上与 Cosmos、HunyuanVideo、WAN2.2、EgoDreamer、Handi 等基线进行对比，取得显著提升：DINO-Score 提升 13.5%、CLIP-Score 提升 10.1%、SSIM 提升 9.7%、PSNR 提升 17.8%、LPIPS 降低 35%、FVD 降低 43%、Flow MSE 降低 51%。同时在真实 ARGO 智能眼镜场景中验证了模型的鲁棒性。

**⚠️ 局限性**

局限性：①模型仍以离线方式生成视频，实时性能尚未验证；②对外部视角图像的依赖在缺失时性能会下降；③训练和评估仍需大量手工标注，数据集规模与多样性有限；④长时间（数十秒以上）视频生成的可控性与一致性仍有提升空间。

---

## 386. Constraint-aware Path Planning from Natural Language Instructions Using Large Language Models

**arXiv ID:** 2603.19257 | [PDF](https://arxiv.org/pdf/2603.19257v1)

**作者:** Dylan Shim `[一作]` (Florida Atlantic University), Minghan Wei `[通讯]` (Florida Atlantic University)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5019897287)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种基于大型语言模型的自然语言约束路径规划框架，支持从对话式输入自动生成可行且近似最优的路线。

**💡 创新点**

通过问题匹配与自学习生成结构化问题表述、迭代生成-验证机制和自我修正，实现在无模板新任务下的自适应求解。

**🔧 技术方法**

采用大型语言模型（如 GPT‑4、Llama3.3 70b）、结构化问题表述（SPF）、自我验证与迭代细化、少量提示学习以及案例库匹配等技术。

**📊 数据集**

使用人工构造的四类路径规划案例库（基本 TSP、multi‑day、每日不同仓库等），包括城市坐标、距离矩阵与约束设定。

**📈 对比分析**

通过 800 次实验比较有无自验证与迭代的可行率与成本，验证率从 85.6% 提升至 95.4%，迭代后平均成本降低约 7–9%，未与传统算法做直接性能对比。

**⚠️ 局限性**

缺乏理论最优保证，且规模可扩展性受限，超过约 15–20 个城市时生成的解常出现无效或重复访问等错误。

---

## 387. On the Ability of Transformers to Verify Plans

**arXiv ID:** 2603.19954 | [PDF](https://arxiv.org/pdf/2603.19954v1)

**作者:** Yash Sarrof `[一作]` (Saarland University), Michael Hahn `[通讯]` (Saarland University)

**通讯引用:** 17402 | [OpenAlex ID](https://openalex.org/A5084053909)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究Transformer在计划验证中的长度泛化能力，并提出理论框架和C-RASP扩展来分析固定与可变对象数量的情况。

**💡 创新点**

创新点在于引入Symbolic Limit Transformer和扩展的C-RASP语言，给出可在对象数随测试增加时保证长度泛化的理论条件，并证明删除无效和well-formed规划实例可实现泛化，反之则不行。

**🔧 技术方法**

使用Transformer结构、绝对位置编码（APE）、C-RASP和Symbolic Limit Transformer的表达式，结合匹配谓词和局部计数的程序。

**📊 数据集**

使用三类人工生成的规划数据集：Grippers Heavy（well-formed/删除无效）、Colors（STRIPS）和Lights Out（条件效应与well-formed），并在不同计划长度与对象数上生成正负样本。

**📈 对比分析**

通过在GPT-2架构下训练，评估ID（长度11-100）和OOD（长度101-200）上准确率，发现well-formed与删除无效域在长计划和更多对象上保持高准确率，而STRIPS与条件效应域则显著下降，验证理论预测。

**⚠️ 局限性**

局限在于仅关注计划验证而非生成，未考虑链式思考、旋转位置编码等变体，且理论主要适用于固定/可变对象数量的规划，对更复杂或真实世界规划的可推广性未知。

---

## 388. Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure

**arXiv ID:** 2603.19426 | [PDF](https://arxiv.org/pdf/2603.19426v1)

**作者:** Viliana Devbunova `[一作]` `[通讯]` (Yandex), Viliana Devbunova (Yandex)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大型语言模型是否具备区分评估场景与部署场景的评估意识进行了实验验证，并引入了2×2实验设计以独立控制提示格式与上下文

**💡 创新点**

创新点在于将格式（Benchmark vs. Casual）与上下文（Evaluation vs. Deployment）解耦，构造四方矩阵数据集，证明标准线性探针仅捕捉格式特征而非真正的评估意识

**🔧 技术方法**

采用冻结的残差流激活（层8–31）提取、逻辑回归线性探针训练、单对照与配对训练两种对比方案，并进行5折交叉验证

**📊 数据集**

使用MMLU、ARC、GSM8K（Bench‑Eval）、ShareGPT（Bench‑Deploy、Casual‑Deploy）、MuTual（Casual‑Eval）等公开数据集，构建四个实验组

**📈 对比分析**

与标准单对照训练相比，配对训练在格式解耦后能将Bench‑Deploy误判率从94.5%降至7.2%，同时保持对Casual‑Eval的高精度，显示格式敏感性被显著缓解

**⚠️ 局限性**

局限性在于仅使用线性探针与单一 Llama‑3.1‑8B‑Instruct 模型，未探究非线性或分布式表示；代理数据集的设计可能不完全代表真实部署场景

---

## 389. Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis

**arXiv ID:** 2603.19282 | [PDF](https://arxiv.org/pdf/2603.19282v1)

**作者:** Zice Wang `[一作]` (Northeastern University), Zhenyu Zhang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 13961 | [OpenAlex ID](https://openalex.org/A5100389500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对不同大语言模型在无通信的独立代理设置下，比较两种逻辑等价但措辞不同的阈值投票提示，观察其决策偏好。

**💡 创新点**

提出“独立代理LLM”概念，并系统评估跨模型族群的语言框架效应，揭示框架能显著改变LLM的风险规避与合作倾向。

**🔧 技术方法**

使用单轮提问、固定温度0.3的采样策略，以及统计学检验（卡方检验、ΔP差异量）对模型回答进行分类和量化。

**📊 数据集**

实验数据来自多家LLM族群（Claude、GPT、Gemini、Qwen、Llama、DeepSeek、GLM、Grok）各自5次独立提示，没有外部公开数据集。

**📈 对比分析**

通过对每族群累计A/B/C比例并计算ΔP，使用卡方检验比较两种提示下的选择分布，结果显示大多数族群出现显著的框架效应，且不同族群之间差异显著。

**⚠️ 局限性**

局限性包括样本量小（每模型5次）、仅为观察性研究无法证明因果关系、任务为单一阈值投票设置缺乏通用性、未考虑多轮交互或更复杂情境。

---

## 390. Prompt-tuning with Attribute Guidance for Low-resource Entity Matching

**arXiv ID:** 2603.19321 | [PDF](https://arxiv.org/pdf/2603.19321v1)

**作者:** Lihui Liu `[一作]` (Wayne State University), Carl Yang `[通讯]` (Emory University)

**通讯引用:** 3958 | [OpenAlex ID](https://openalex.org/A5006897094)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对低资源环境下的实体匹配问题，提出 PromptAttrib 方法，通过属性级 prompt 调优与模糊逻辑推理实现更准确的匹配，并引入 dropout 对软 prompt 的对比学习提升泛化能力。

**💡 创新点**

创新点在于：①将实体级与属性级信息统一到 prompt 结构中，利用模糊逻辑规则从属性匹配推导实体匹配；②将 SimCSE 的对比学习思想迁移到软 prompt，采用 dropout 产生正样本，提升低资源场景下的表征质量；③提供可解释的匹配过程。

**🔧 技术方法**

技术主要包括：提示调优（hard 与 continuous prompt）、模糊逻辑推理、dropout 对比学习、预训练语言模型（如 RoBERTa‑large、Albert‑large）作为 backbone。

**📊 数据集**

使用四个真实世界数据集：geo‑heter（地理实体）、cameras（相机产品）、computers（电脑产品）以及 ISWC（学术会议实体）。

**📈 对比分析**

与 SentenceBERT、DeepMatcher、Ditto、PromptEM 等基线在 5% 标注数据的低资源设置下进行比较，PromptAttrib 在 F1、Average Precision 与 Accuracy 上均显著领先，尤其在 geo‑heter、computers 与 ISWC 数据集上提升 2–3% F1。

**⚠️ 局限性**

局限性包括：①仍需预训练模型支持，参数量大；②对 dropout 率和模糊规则的设计敏感；③在极端稀缺标注或属性缺失严重的数据集上效果尚未充分验证。

---

## 391. L-PRISMA: An Extension of PRISMA in the Era of Generative Artificial Intelligence (GenAI)

**arXiv ID:** 2603.19236 | [PDF](https://arxiv.org/pdf/2603.19236v1)

**作者:** Samar Shailendra `[一作]` (Melbourne Institute of Technology), Urvashi Rahul Saxena `[通讯]` (Melbourne Institute of Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了L-PRISMA框架，将生成式人工智能与传统PRISMA流程结合，采用预筛选语义过滤与LLM辅助文献摘要，提升系统综述效率。

**💡 创新点**

在PRISMA基础上新增统计预筛选阶段、明确GenAI记录报告、并用混合高斯模型确定阈值，兼顾可重复性与自动化。

**🔧 技术方法**

使用大型语言模型（ChatGPT、Gemini、Claude）、S‑BERT语义相似度、GMM统计模型及Python工具。

**📊 数据集**

IEEE与ACM数据库中的文本相似性与GenAI相关文献（共约1819条记录）。

**📈 对比分析**

对比传统全手工筛选，预筛选后手工审核仅需60条，GenAI辅助筛选989条，整体筛选时间缩短约80%，但未给出精确召回率/准确率统计。

**⚠️ 局限性**

LLM非确定性与幻觉风险、仍需人工核查、不同领域适用性与统计阈值通用性尚未验证。

---

## 392. Beltrami coefficient and angular distortion of discrete geometric mappings

**arXiv ID:** 2603.19240 | [PDF](https://arxiv.org/pdf/2603.19240v1)

**作者:** Zhiyuan Lyu `[一作]` (Chinese University of Hong Kong), Gary P. T. Choi `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1251 | [OpenAlex ID](https://openalex.org/A5059616736)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了离散几何映射中Beltrami系数与角度失真之间的关系，推导理论公式并在多种映射方法上验证。

**💡 创新点**

首次给出Beltrami系数范数与角度失真之间的解析关联和上界估计，并将其推广到离散三角网。

**🔧 技术方法**

采用准共形理论、Beltrami系数计算、面向三角网的离散化以及数值实验对比。

**📊 数据集**

使用人类面部、Sophie模型、Chinese Lion、Brain、David、Max Planck、Buddha等多种开闭面片及其三角网数据集。

**📈 对比分析**

通过统计均值、最大值、直方图对比Beltrami系数与面基角度失真，结果显示两者高度相关且估计值为保守上界。

**⚠️ 局限性**

仅针对二维表面映射；离散近似可能影响精度；上界可能过保守；未扩展至三维体积映射。

---

## 393. Reasoning Gets Harder for LLMs Inside A Dialogue

**arXiv ID:** 2603.20133 | [PDF](https://arxiv.org/pdf/2603.20133v1)

**作者:** Ivan Kartáč `[一作]` (Charles University), Ondřej Dušek `[通讯]` (Charles University)

**通讯引用:** 2140 | [OpenAlex ID](https://openalex.org/A5004829991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个动态、可验证的任务导向对话（TOD）推理基准Boulder，系统性比较了大型语言模型（LLM）在孤立任务与对话环境中的推理表现，并通过消融实验和定性分析阐明性能差距来源。

**💡 创新点**

首次在对话框架下评估推理任务，发现多轮交互、角色设定与工具使用显著削弱LLM推理能力；通过动态生成、可验证数据和LLM解析器实现了高质量、可扩展的基准。

**🔧 技术方法**

采用LLM生成的可验证实例、JSON解析器与PPI纠偏、MAE标准化、显著性检验、功能调用（function calling）提示、微平均分数等技术；在基线、对话、对话-简洁三种提示设置下评估。

**📊 数据集**

基于改造的MultiWOZ旅行领域数据库，生成8个推理任务（票价、预订、出发时间、频率、开放时间、距离、方向、最短路径），全部为动态合成、可验证的实例。

**📈 对比分析**

在Baseline、Dialogue和Dialogue-concise三种设置下，评估8款LLM（如Qwen3、Mistral、Gemini、Claude等），Baseline平均得分0.87–0.97，Dialogue显著下降至0.70–0.80，消融实验表明多轮交互是主要原因。

**⚠️ 局限性**

研究仅涵盖四个旅行领域，未涉及更复杂应用；LLM解析器虽高效但仍有误差；实验仅在零样本提示下进行，未评估微调或少量示例的影响；工具与角色约束的具体作用机制尚未完全解离。

---

## 394. EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models

**arXiv ID:** 2603.19532 | [PDF](https://arxiv.org/pdf/2603.19532v1)

**作者:** J. Ben Tamo `[一作]` (Georgia Institute of Technology), May D. Wang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 20563 | [OpenAlex ID](https://openalex.org/A5030096887)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出EvidenceRL，一种在训练阶段通过强化学习强制模型遵循检索证据的框架。

**💡 创新点**

创新点在于将句子级NLI的Focus-Then-Verify奖励与正确性奖励结合，并使用Group Relative Policy Optimization (GRPO)实现对证据遵从的可微优化。

**🔧 技术方法**

采用NLI跨编码器进行句子级真实性检测、LLM-judge做语义正确性评估、GRPO强化学习、LoRA微调以及检索增强生成。

**📊 数据集**

在MIMIC-IV-Ext（心脏诊断）和BarExam MBE（法律推理）两个高风险领域的公开数据集上进行评估。

**📈 对比分析**

与Reasoning-Only、Self-RAG、Self-Consistency、SFT、fDPO等基线相比，EvidenceRL在任务准确率和证据基准（G_max@3、EB%、faithfulness）均显著提升，示例：Llama-3.2-3B F1@3从37.0提升至54.5，G_max@3从47.6提升至78.7，EB%从31.8%提升至61.6%。

**⚠️ 局限性**

局限性包括依赖自动化代理进行奖励与评估、奖励信号可能无法完全覆盖真实证据一致性、训练成本高于SFT、假设检索证据无误且需在更大范围内验证。

---

## 395. CurveStream: Boosting Streaming Video Understanding in MLLMs via Curvature-Aware Hierarchical Visual Memory Management

**arXiv ID:** 2603.19571 | [PDF](https://arxiv.org/pdf/2603.19571v1)

**作者:** Chao Wang `[一作]` (Fudan University), Tao Chen `[通讯]` (Fudan University)

**通讯引用:** 43883 | [OpenAlex ID](https://openalex.org/A5100357719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CurveStream，一种训练无关的曲率感知层级视觉记忆管理框架，用于提升多模态大模型在流媒体视频理解中的性能并避免显存溢出。

**💡 创新点**

创新点在于发现视频特征轨迹的高曲率对应关键语义转折，利用实时曲率分数结合在线 K‑Sigma 阈值，动态分配高低分辨率记忆，形成自适应的“清晰/模糊/丢弃”策略。

**🔧 技术方法**

技术包括曲率感知评分器（CAS）融合第一、二阶运动度量；层级视觉记忆管理（HVMM）；在线统计更新的 K‑Sigma 阈值；以及对帧分辨率的动态压缩。

**📊 数据集**

使用的主要数据集包括 StreamingBench、OVOBench、EgoSchema、VideoMME、MVBench 等多种实时与离线视频基准。

**📈 对比分析**

与统一采样、光流、cosine 相似、Flash‑VStream、FreshMem、HERMES、ReKV 等基线对比，在 StreamingBench 与 OVOBench 上实现 10–13% 的绝对准确率提升，甚至在 7B 开源模型上超过 GPT‑4o 与 Gemini 1.5 Pro。

**⚠️ 局限性**

局限性包括对极长视频可能牺牲细粒度全局细节导致略微下降；虽对曲率权重 λ 与阈值 k 具有一定鲁棒性，但仍需在不同场景中微调；目前仅关注视觉层级，未扩展至更广泛的多模态交互或非视频流任务。

---

## 396. MeanFlow Meets Control: Scaling Sampled-Data Control for Swarms

**arXiv ID:** 2603.20189 | [PDF](https://arxiv.org/pdf/2603.20189v1)

**作者:** Anqi Dong `[一作]` (KTH Royal Institute of Technology), Johan Karlsson `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1678 | [OpenAlex ID](https://openalex.org/A5111976155)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于MeanFlow的控制空间学习框架，学习有限时窗最小能量控制系数，用于在采样数据线性动力学下实现大规模群体的少步导航。

**💡 创新点**

创新点在于将窗口级控制系数作为学习目标，通过桥式监督实现与实际采样更新的一致性，解决传统流匹配仅学习瞬时速度导致的离散误差放大的缺陷。

**🔧 技术方法**

采用控制理论中的有限时窗最优控制与可控性Gramian、MeanFlow流匹配框架、神经网络回归（含stop‑gradient）、采样数据离散化等技术。

**📊 数据集**

使用合成的二维“AYKJ→DCJK”字母分布和三维金字塔→环面等虚拟分布作为实验数据集；未使用真实外部数据集。

**📈 对比分析**

通过对比标准MeanFlow/流匹配的轨迹连贯性与最终分布匹配，实验演示在不同动力学（无漂移、二维/三维旋转漂移）下模型能够保持一致的几何对应，性能主要以可视化示例展示，体现少步控制的有效性。

**⚠️ 局限性**

局限性包括仅针对线性时不变系统、要求系统可控、桥路段需可微，无法直接处理非线性、受限或随机动态，且缺乏定量性能评估。

---

## 397. Power laws and power-of-two-choices

**arXiv ID:** 2603.20060 | [PDF](https://arxiv.org/pdf/2603.20060v1)

**作者:** Amanda Redlich `[一作]` (University of Massachusetts Lowell), Amanda Redlich `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 4423 | [OpenAlex ID](https://openalex.org/A5001812924)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了一种名为UNFAIR的分配算法，该算法在每一步选择随机的d个箱子后，将球放入最繁忙的箱子，导致箱子负载呈幂律分布；

**💡 创新点**

首次证明在UNFAIR下，i-th最轻箱子的负载约为((i/n)^d-((i-1)/n)^d)m+O(m/n^d+1)，并且该分布与箱子相对排名的(d-1)次幂相关；

**🔧 技术方法**

采用分阶段分析、初始化阶段随机游走与赌徒破产论证、二项分布上下界耦合等概率技术进行严格证明；

**📊 数据集**

使用实验数据集：n=100、m=10^6，d=2、3、4，对比理论预测与实际负载；

**📈 对比分析**

通过实验验证理论预测与实际负载高度匹配，说明UNFAIR确实产生了预期的幂律分布；

**⚠️ 局限性**

证明所需的球数阈值为m>n^{4d+13}，在实际中可能提前收敛但未被证明；误差项保守，分析复杂且对极端参数缺乏精细分析。

---

## 398. AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning

**arXiv ID:** 2603.20147 | [PDF](https://arxiv.org/pdf/2603.20147v1)

**作者:** Huihua Zhao `[一作]`, Yan Chang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AGILE，一套完整的从机器人模型验证、训练、评估到部署的可复现、可扩展的工作流；

**💡 创新点**

通过统一的四阶段流水线（Prepare、Train、Evaluate、Deploy）填补了工作流与转移两大缺口；

**🔧 技术方法**

基于Isaac Lab与RSL‑RL，集成了GPU加速仿真、可调节算法增强（正则化、奖励归一化、终止值回弹、虚拟安全索具、对称增强等）、GUI调试、Deterministic/Stochastic评估、YAML I/O描述符导出；

**📊 数据集**

在两款人形机器人（Unitree G1、Booster T1）上使用多任务数据集（速度跟踪、高度控制、站立、运动模仿、Loco‑Manipulation）进行训练；

**📈 对比分析**

与传统基于随机抽样的评估相比，Deterministic场景测试与指标（如关节加速度、冲击、极限违规）提升了回归测试的可靠性；在五个任务上实现了与模拟/真实硬件的成功转移，表现与先前工作相当或优于；

**⚠️ 局限性**

目前仅验证于两款机器人且主要为本体感知任务，缺乏更复杂感知驱动或动态动作的支持，且对Isaac Lab的依赖限制了跨平台部署。

---

## 399. Sound State Encodings in Translational Separation Logic Verifiers (Extended Version)

**arXiv ID:** 2603.20001 | [PDF](https://arxiv.org/pdf/2603.20001v1)

**作者:** Hongyi Ling `[一作]` (ETH Zurich), Peter Müller `[通讯]` (ETH Zurich)

**通讯引用:** 13873 | [OpenAlex ID](https://openalex.org/A5050979141)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一套正式框架，用来证明翻译式分离逻辑验证器（支持非平凡状态编码）的正确性；通过引入状态编码关系、同态性和“后向可满足性”条件，解决了传统框架中状态模型差异导致的安全性缺失问题；并在该框架下实现并验证了针对 Raven、VeriFast、Viper 等多种前端的三种典型状态编码的正确性；

**💡 创新点**

1) 首次将翻译验证器的前端与后端分离时的状态模型差异抽象为状态编码关系；2) 引入“后向可满足性”这一新条件，能够有效避免由稀疏状态空间导致的伪分割引发的安全漏洞；3) 在框架内统一证明多种前端的安全性，展示了其广泛适用性；

**🔧 技术方法**

形式化逻辑（分离逻辑、IDF代数）、形式化证明（使用Isabelle/HOL进行机理化验证）、状态编码关系的构造与验证、后向可满足性的数学定义与证明；

**📊 数据集**

无；论文为理论性工作，没有使用实验数据集；

**📈 对比分析**

无；该工作为形式化安全性证明，没有性能评估或实验比较；

**⚠️ 局限性**

1) 需要明确的状态编码关系，若关系无法构造则无法使用框架；2) 仍假设前端使用的是可满足的分离逻辑（如前向可满足性）；3) 对极大或复杂的前端模型，构造后向可满足性证明可能变得困难；

---

## 400. Continual Learning as Shared-Manifold Continuation Under Compatible Shift

**arXiv ID:** 2603.20036 | [PDF](https://arxiv.org/pdf/2603.20036v1)

**作者:** Henry J. Kobs `[一作]` `[通讯]`, Henry J. Kobs

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在旧任务与新任务共享同一潜在空间的连续学习场景下，提出一种基于锚点几何保持的模型（SPMA-OG），并在此框架下实现对旧任务潜在图形的保持。

**💡 创新点**

创新点在于：①将旧任务的潜在分布视为一组局部图（chart）集合；②通过锚点的几何距离、邻接图和图分配一致性等多维几何正则化，显式约束新任务学习过程，使得新数据在旧潜在空间上连续演化，而非全局漂移或完全冻结；③将这种几何约束与传统的稀疏重放、知识蒸馏、平滑与参数正则等技术统一到一个完整的目标函数中。

**🔧 技术方法**

使用的技术包括：稀疏锚点重放、输出蒸馏、全局与局部几何正则（对锚点间距离矩阵、k近邻图的匹配）、图表保持（对教师与学生的软图分配进行KL正则）、参数ℓ₂漂移控制以及线性衰减的保留权重。

**📊 数据集**

实验数据集：CIFAR10 与 Tiny-ImageNet 的兼容性迁移（同类、不同输入变换）设置，以及一个用于验证几何保持效果的合成图形（warped ribbon surface）基准。

**📈 对比分析**

与基线（Plain FT、Anchor CE、ER‑512）对比，SPMA‑OG 在兼容性迁移任务中既能保持更高的旧任务准确率，又能保持或略优的新任务准确率；在表示层面上，其CKA、距离相关系数、锚点支持率均明显优于重放方法，表明潜在空间几何得到更好的保留。

**⚠️ 局限性**

局限性在于该方法仅在旧旧数据与新数据共享潜在支持的“兼容迁移”情形下有效，对需要显式拓展潜在空间的新任务（如真实的增量分类）可能过于受限；此外，实验未能单独量化图表保持对性能提升的独立贡献，整体效果来自多种正则项的协同。

---

## 401. Full-Stack Domain Enhancement for Combustion LLMs: Construction and Optimization

**arXiv ID:** 2603.19268 | [PDF](https://arxiv.org/pdf/2603.19268v1)

**作者:** Quanjia Xiao `[一作]` (Peking University), Zhi X. Chen `[通讯]` (AI for Science Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了完整的火焰科学领域大模型适配流程，包括领域语料库构建、增量预训练、监督微调与基于可验证奖励的强化学习，最终发布了FlameBench基准；

**💡 创新点**

创新点在于整合多阶段训练（CPT→SFT→RLVR）与可验证奖励机制，使模型真正内化物理定律，提升多物理耦合推理能力；

**🔧 技术方法**

采用Qwen-8B为基础模型，使用增量预训练（CPT）、监督微调（SFT）、RLVR（GRPO框架）和RAG对比实验；

**📊 数据集**

数据集包括约30B token的混合语料（5B火焰科学+25B通用），SFT用800K通用指令+12K链式思维示例，RLVR用7K多参数耦合推理实例，FlameBench共436道高质量问题；

**📈 对比分析**

与通用大模型（如GPT-5、GLM-4等）及RAG系统比较，RLVR-Opt在FlameBench上准确率达43.8%，比RAG最高模型提升约12个百分点，展示显著性能优势；

**⚠️ 局限性**

局限性在于仍需更丰富的实验验证、奖励设计可能引入偏差、对极端物理边界的泛化有限，且对跨域推理的复杂场景尚未完全覆盖。

---

## 402. Stochastic Sequential Decision Making over Expanding Networks with Graph Filtering

**arXiv ID:** 2603.19501 | [PDF](https://arxiv.org/pdf/2603.19501v1)

**作者:** Zhan Gao `[一作]` (University of Cambridge), Elvin Isufi `[通讯]` (Delft University of Technology)

**通讯引用:** 2184 | [OpenAlex ID](https://openalex.org/A5060836778)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在网络扩展过程中通过图滤波进行信号推断，并提出基于多智能体强化学习的序列决策框架。

**💡 创新点**

创新点在于：将滤波器参数视为多智能体系统，利用上下文感知图神经网络(C‑GNN)学习长期奖励策略，充分考虑图扩展的随机性与长期影响。

**🔧 技术方法**

使用的技术包括：图滤波、上下文感知图神经网络、强化学习（特别是多智能体强化学习，MARL）以及与之配套的奖励设计与策略参数化。

**📊 数据集**

使用的数据集包括：随机生成的扩展图（基于 Erdos‑Renyi 与优先附着模型）、Movielens‑100K 用户推荐数据以及 269 个城市的 COVID‑19 日病例数据。

**📈 对比分析**

与批量滤波、在线滤波和在线 GNN 进行对比。实验表明，提出的方法在 RMSE、推荐准确率以及 COVID 预测误差上均优于基线，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：目前仅针对信号推断任务，尚未验证在更复杂任务（如多跳预测、生成任务）上的有效性；模型对极大规模图的可扩展性仍待评估；以及对不同类型图扩展模型的泛化能力需要进一步研究。

---

## 403. AURORA: Adaptive Unified Representation for Robust Ultrasound Analysis

**arXiv ID:** 2603.19364 | [PDF](https://arxiv.org/pdf/2603.19364v1)

**作者:** Ufaq Khan `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种统一的多任务框架，利用Qwen3‑VL视觉编码器同时完成超声图像的分割、检测、分类和标记回归；

**💡 创新点**

创新点包括：①将Transformer中间token映射为多尺度空间特征金字塔（token‑to‑map桥接）；②轻量化的多尺度特征融合（FPN风格）支持像素级与全局任务；③任务感知采样与选择性损失平衡机制，缓解任务不平衡；

**🔧 技术方法**

使用的技术包括Qwen3‑VL预训练视觉编码器、1×1卷积投影、FPN多尺度融合、Atrous Spatial Pyramid Pooling、anchor‑free检测头、轻量级多任务头、温度采样与动态损失重权重等；

**📊 数据集**

实验基于FMC‑UIA 2026挑战数据集（27个子任务，覆盖分割、分类、检测、回归四大任务类别）；

**📈 对比分析**

与官方基线比较，验证集平均分数由67%提升至85%，在官方测试集上平均达81.84%，各子任务指标表现不均，但整体提升显著；

**⚠️ 局限性**

局限性包括共享骨干可能产生负迁移，检测对极小或模糊目标的敏感度不足，以及固定分辨率输入导致几何细节丢失。

---

## 404. 2DIO: A Cache-Accurate Storage Microbenchmark

**arXiv ID:** 2603.19971 | [PDF](https://arxiv.org/pdf/2603.19971v1)

**作者:** Yirong Wang `[一作]` (Northeastern University), Peter Desnoyers `[通讯]` (Northeastern University)

**通讯引用:** 2134 | [OpenAlex ID](https://openalex.org/A5015856210)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为2DIO的微基准，能够生成与真实工作负载相匹配、且可配置的缓存精确I/O轨迹。

**💡 创新点**

创新点在于通过短期递归(IRD)与长期频率(IRM)的三元组参数化，能精准重现非凸缓存命中率曲线（性能陡峭和平台），并可在不同规模下保持相对行为。

**🔧 技术方法**

使用的技术包括IRD和IRM的联合采样算法（Gen‑from‑IRD与Gen‑from‑2D）、参数化的fgen模型、堆优先队列生成器以及缓存模拟库。

**📊 数据集**

实验基于AliCloud和CloudPhysics公开的真实块级I/O轨迹（如w11、w24、w44、w82等）。

**📈 对比分析**

与TraceRaR、LLGAN等现有生成器比较，2DIO在LRU HRC重现误差低于5%（MAE约0.03‑0.05），而后者即使在分布匹配上表现良好，HRC却偏差显著。

**⚠️ 局限性**

局限性包括对多块请求的顺序访问会破坏IRD样本空间，且生成的SPC格式I/O轨迹无法直接反映真实系统中的其他资源（如网络延迟、调度等）。

---

## 405. Integrating Meta-Features with Knowledge Graph Embeddings for Meta-Learning

**arXiv ID:** 2603.19888 | [PDF](https://arxiv.org/pdf/2603.19888v1)

**作者:** Antonis Klironomos `[一作]` (Bosch Center for Artificial Intelligence), Evgeny Kharlamov `[通讯]` (Bosch Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建统一的知识图谱（KG），将OpenML数据集与其对应的scikit‑learn流水线及其元数据映射进去，并使用KG嵌入实现元学习任务：流水线性能估计和基于性能的相似数据集检索。

**💡 创新点**

创新点在于：①首次将实验元数据与数据集元特征融合进KG，捕捉数据集–流水线交互的潜在模式；②提出单一流水线无关的元模型，摆脱每个流水线单独训练的限制；③通过KG嵌入获得的向量实现高效的相似度检索，显著提升相似数据集发现的效果。

**🔧 技术方法**

技术手段包括：使用MLSO和Executable‑KG本体构建KG；采用walk‑based KG嵌入（如RDF2Vec）并对数值属性进行分桶；对数据集节点和流水线节点分别聚合得到VAR、PIPE和COMB三种嵌入；用随机森林训练流水线无关的元模型；使用余弦相似度进行相似度评估。

**📊 数据集**

数据集与实验：170个OpenML数据集，2616个scikit‑learn流水线配置，基于这些配置在170个数据集上稀疏训练得到144,177个实验结果，构成KGmetaSP benchmark。

**📈 对比分析**

与传统基于元特征的流水线特定模型、平均性能、最近邻嵌入、Graph Edit Distance、元特征距离、以及基于文本的SiFi-Dat./SiFi-Pip./MuFi-Dat.等基线比较；在未见数据集和未见流水线的元回归/元分类任务中，单一元模型在准确率、F1、R²等指标上均优于基线；在相似数据集检索中，KG嵌入在Hit@K和NDCG@K上取得显著提升，尤其在Top‑1/Top‑2检索中表现最为突出。

**⚠️ 局限性**

局限性：①仅支持scikit‑learn流水线，难以直接迁移到其他ML框架；②KG嵌入为静态，需要重新嵌入或增量更新才能反映新实验；③在精确点估计（回归误差）方面不如专门为单个流水线训练的模型；④对流水线配置的稀疏覆盖和不完整实验记录可能影响嵌入质量。

---

## 406. From Token to Item: Enhancing Large Language Models for Recommendation via Item-aware Attention Mechanism

**arXiv ID:** 2603.19693 | [PDF](https://arxiv.org/pdf/2603.19693v1)

**作者:** Xiaokun Zhang `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 28048 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于大型语言模型的推荐框架 IAM，通过在自注意力中加入面向项的注意力机制，区分并建模同一条目内部的 token 关系和不同条目之间的 token 关系，显著提升个性化推荐效果。

**💡 创新点**

创新点在于将 token 视为中介而非主体，引入 intra‑item 与 inter‑item 两层注意力，显式把条目作为推荐的基本单位，突破传统 LLM 仅关注 token 级别关系的局限。

**🔧 技术方法**

采用 Llama3（3B 参数）作为基础模型，使用 LoRA 微调，设计双层注意力（内部注意力与外部注意力）以及推荐适配器实现分数生成。

**📊 数据集**

在 Amazon Grocery、Arts、Cellphones 三个公开数据集（分别含 1,874 / 4,265 / 6,593 条目）上进行实验，采用标题文本作为条目内容。

**📈 对比分析**

与传统的 GRU4Rec、SASRec、Atten‑Mixer 以及 LLM 基线 Llama、P5、E4SRec、LLaRA 等九种方法对比，IAM 在 Prec@5/10 与 NDCG@5/10 上平均提升 25.8% / 10.0%（Grocery）、6.8% / 3.7%（Arts）和 71.0% / 74.9%（Cellphones），实现最优表现。

**⚠️ 局限性**

局限在于仅利用条目标题，未加入图像等多模态信息；注意力层对所有用户统一，缺乏个性化调整；模型规模大，推理成本高，需进一步轻量化。

---

## 407. PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing

**arXiv ID:** 2603.19731 | [PDF](https://arxiv.org/pdf/2603.19731v1)

**作者:** Jiadong Liang `[一作]` (HUJING Digital Media & Entertainment Group), Huan Fu `[通讯]` (HUJING Digital Media & Entertainment Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对一段表情视频，通过驱动视频实现表情的编辑与增强，并可对静态人像进行动画渲染。

**💡 创新点**

改进了LivePortrait的关键点变换公式，使其与3DMM FLAME一致，引入FLAME损失和显式关键点监督；设计边界对齐模块解决表情变换时面部与非面部区域失配；构建MetaHuman基准评测；整体实现表情独立编辑。

**🔧 技术方法**

基于GAN的LivePortrait框架，DINOv2特征提取，Pixel3DMM 3DMM参数追踪，FLAME 3D Morphable Model，教师-学生边界对齐模块，显式关键点监督与FLAME损失。

**📊 数据集**

VFHQ、MEAD、Nersemble、FEED、ETH-XGaze等公开数据集，约59.7万段视频；结合互联网动漫电影人像视频；MetaHuman数字人渲染基准。

**📈 对比分析**

与LivePortrait、Diffusion基方法（SkyReels-A1、Hunyuan-Portrait、FantasyPortrait、Wan-Animate）以及商业产品Act‑Two对比。实验显示在PSNR、SSIM、LPIPS、FID、FVD、CSIM、AED、APD等指标上均优于对手，尤其在自重演与跨重演任务上实现SOTA；消融实验验证关键点变换与FLAME损失、边界对齐模块的有效性。

**⚠️ 局限性**

主要局限在于依赖FLAME 3DMM参数追踪，对极端表情或遮挡的鲁棒性有限；仅实现表情独立编辑，无法同时修改姿态；训练成本高；未与Diffusion模型在表情编辑上的更深层对比。

---

## 408. Surrogate Modeling with Low-Rank Function Representation for Electromagnetic Simulation

**arXiv ID:** 2603.19735 | [PDF](https://arxiv.org/pdf/2603.19735v1)

**作者:** Mingze Sun `[一作]` (University of Electronic Science and Technology of China), Bin Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 84001 | [OpenAlex ID](https://openalex.org/A5100395468)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了低秩张量函数表示的电磁仿真代理模型，尤其是分布式对偶低秩张量网络PLRNet；

**💡 创新点**

创新点在于将低秩表示从全局 Tucker 核扩展到对偶交互子网络，利用对偶低秩因子分布式建模高维参数耦合；

**🔧 技术方法**

采用SIREN激活的单维嵌入网络、对偶低秩交互核、Tensor-Train/Tensor-Ring、LRTFR等低秩格式以及传统MLP；

**📊 数据集**

使用三组电磁仿真基准：二维双向散射（椭圆柱RCS）、三维微带线回损、以及三维螺旋慢波结构（SWS）的相位速度、相互阻抗、衰减等；

**📈 对比分析**

与MLP、LRTFR、TT、TR等对齐，评估测试平均相对误差、最大相对误差和参数规模；实验表明PLRNet在三大基准上均优于MLP，且在高维场景下参数量和误差均保持在最低水平；

**⚠️ 局限性**

局限在于对偶交互对维度增长导致的参数爆炸问题，且对物理先验（如可逆性、因果性）的约束尚未充分引入；

---

## 409. ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding

**arXiv ID:** 2603.19610 | [PDF](https://arxiv.org/pdf/2603.19610v1)

**作者:** Quan Kong `[一作]` (Zhejiang University), Cong Wang `[通讯]` (Zhejiang University)

**通讯引用:** 25704 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ParallelVLM 框架，用无训练的并行草稿‑验证推断降低视频 LLM 的自回归解码延迟。

**💡 创新点**

核心创新在于并行前填充与解码的协同设计，以及无注意力的 Unbiased Verifier‑Guided Pruning（UV‑Prune）实现无信息损失的视觉令牌裁剪。

**🔧 技术方法**

使用 Speculative Decoding、并行流水线、UV‑Prune、FlashAttention 等技术。

**📊 数据集**

在 LLaVA‑OneVision、Qwen2.5‑VL 等模型的 128 帧视频，结合 VideoDetailCaption、VideoMME、MVBench、MVLU、LongVideoBench 等五个视频理解基准。

**📈 对比分析**

与传统 SD、SpecVLM、STD 等 lossless 方法以及 FastV、SparseVLM 等 lossy 裁剪方法比较，平均加速比分别为 3.36×、2.42×，并保持与全上下文一致的生成质量。

**⚠️ 局限性**

限制在极端裁剪比例下会失去对齐，且仍依赖大量 KV 缓存存储，对极大长度场景的进一步扩展尚需研究。

---

## 410. Text-Based Personas for Simulating User Privacy Decisions

**arXiv ID:** 2603.19791 | [PDF](https://arxiv.org/pdf/2603.19791v1)

**作者:** Kassem Fawaz `[一作]` (University of Wisconsin--Madison), Marco Gruteser `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

生成并验证基于历史隐私决策的文本化隐私人格，用于模拟个体和群体的隐私选择。

**💡 创新点**

创新点在于将大规模调查日志压缩为可读文本人格，并通过迭代优化在保持可读性的同时实现高预测准确率。

**🔧 技术方法**

利用大型语言模型（Gemini 3.0 Flash、Gemini 2.5 Flash‑Lite）进行人格生成、优化和预测，并结合隐私理论模板（Privacy Calculus、Bounded Rationality、PMT）。

**📊 数据集**

使用五个公开隐私调查数据集：Pew PP1（2014）、Pew W49（2019）、Pew W127（2023）、CAuthN（2022）以及 SPA（2020/21）。

**📈 对比分析**

通过与原始问答记录、LLM 内部基线以及跨研究转移的对比，个体准确率最高可达 85%，群体级 TVComplement 超过 0.9；在 80/20 训练/评估拆分下，比原始记录压缩率 82–95%。

**⚠️ 局限性**

局限：仅覆盖美国/英国成年人；模型对时间漂移敏感；迭代优化成本高；跨文化、跨语言泛化待验证。

---

## 411. MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels

**arXiv ID:** 2603.19310 | [PDF](https://arxiv.org/pdf/2603.19310v1)

**作者:** Tianyang Luo `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7930 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出MemReward框架，利用图结构的经验记忆在奖励标签稀缺的情况下进行LLM强化学习优化；

**💡 创新点**

创新点在于将查询、推理过程和答案构成异构图，并用图神经网络对未标注样本进行奖励传播，从而显著减少对真实奖励标注的依赖；

**🔧 技术方法**

核心技术包括异构图神经网络（GNN）与多类型边（query-query、query-thinking、thinking-answer）聚合、基于相似度的邻接构建以及GRPO强化学习算法；

**📊 数据集**

实验数据集涵盖13个任务：数学推理（GSM8K、GSM-Symbolic、MATH）、代码生成（MBPP+、HumanEval+）、问答（MMLU、CSQA、OBQA、ARC-C、GPQA），并在三大领域评估；

**📈 对比分析**

与仅使用20%真实奖励标签（R1-p）和完全监督（R1-Oracle）对比，MemReward在3B模型上仅用20%标签即可达到97.3% Oracle性能，1.5B模型达到96.6%；在标签比例提升到70%时可达到99.4% Oracle，且在部分外域任务甚至超过Oracle；

**⚠️ 局限性**

局限性包括对图构建和相似度阈值的敏感性、对大规模数据时构图成本高以及在极端多模态或结构复杂任务中奖励传播效果可能下降。

---

## 412. Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning

**arXiv ID:** 2603.19397 | [PDF](https://arxiv.org/pdf/2603.19397v1)

**作者:** Xueqiao Peng `[一作]` (Ohio State University), Andrew Perrault `[通讯]` (Ohio State University)

**通讯引用:** 376 | [OpenAlex ID](https://openalex.org/A5057049889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

设计并实现一种分层强化学习框架，用于在有限的检测预算下，按需求动态分配多簇异步出现的疫情群体的诊断检测资源。

**💡 创新点**

创新点在于将全局资源稀缺性用可学习的成本乘子（PPO控制）与局部个体检测价值评估（Transformer‑DQN）解耦，并通过全局Q‑排名实现硬预算下的可解释性资源分配。

**🔧 技术方法**

采用分层强化学习（PPO + Transformer‑DQN）、Lagrangian松弛、全局Q‑排名以及基于经验回放的深度Q网络等技术。

**📊 数据集**

使用基于Agent‑based的SARS‑CoV‑2传播模拟器，生成异步出现的多簇感染事件，进行实验评估。

**📈 对比分析**

与多种基线（Symp‑AvgRand、Thres‑AvgRand、Thres‑SizeRand、Fixed‑M‑QR、Bin‑M‑QR）相比，所提Hier‑PPO在不同簇数和预算下平均提升5–12%（相较RMAB方法）和20–30%（相较启发式策略），并且在40簇规模下实现约5倍的决策速度提升。

**⚠️ 局限性**

局限性包括：局部决策模型缺乏对个体层面的可解释性、未考虑人类行为与遵从度等社会因素、对阈值隔离策略的依赖，以及在不同真实疫情场景下需要进一步验证其鲁棒性。

---

## 413. Mixed Integer vs. Continuous Model Predictive Controllers for Binary Thruster Control: A Comparative Study

**arXiv ID:** 2603.19796 | [PDF](https://arxiv.org/pdf/2603.19796v1)

**作者:** Franek Stark `[一作]` (German Research Center for Artificial Intelligence GmbH), Shubham Vyas `[通讯]` (German Research Center for Artificial Intelligence GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 ESA reacsa 平台上二进制推力器的控制，比较了直接混合整数 MPC、连续 MPC+Delta‑Sigma 调制以及结合模压预测的二进制信息 MPC。

**💡 创新点**

创新点在于首次系统比较三种控制范式，并提出将 Delta‑Sigma 调制器状态嵌入连续 MPC 的新变体。

**🔧 技术方法**

采用混合整数规划、线性规划和 Delta‑Sigma 调制技术。

**📊 数据集**

使用 ESA reacsa 平台的仿真数据和公开实验数据集 (https://doi.org/10.5281/zenodo.18454916)。

**📈 对比分析**

通过 5400 次仿真实验和真实硬件验证进行比较，结果显示 MIMPC 在低推力下燃料效率最高，二进制信息 MPC 在计算量和稳定性之间取得折衷。

**⚠️ 局限性**

局限在于高推力时连续 MPC 可能不稳定，MIMPC 计算复杂，且实验规模有限。

---

## 414. Deep Hilbert--Galerkin Methods for Infinite-Dimensional PDEs and Optimal Control

**arXiv ID:** 2603.19463 | [PDF](https://arxiv.org/pdf/2603.19463v1)

**作者:** Samuel N. Cohen `[一作]` (Mathematical Institute, University of Oxford), Justin Sirignano `[通讯]` (Mathematical Institute, University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在无限维Hilbert空间上解二阶非线性偏微分方程及最优控制问题，提出Deep Hilbert–Galerkin和Hilbert Actor‑Critic方法。

**💡 创新点**

首次给出Hilbert Galerkin网络在无穷维空间逼近C^2解的统一逼近定理，处理无界算子作用下的逼近问题，并提出Optimize‑then‑Learn框架直接解决全域无限维PDE与最优性条件。

**🔧 技术方法**

利用Hilbert Galerkin网络（深度神经网络+正交基表示）、Sobolev空间理论、无界算子与Dissipativity分析、随机梯度下降/Adam优化以及Actor‑Critic强化学习结构。

**📊 数据集**

以合成的L^2([0,2π])随机函数作为训练样本，采用高斯/Ornstein‑Uhlenbeck分布进行采样，无使用公开数据集。

**📈 对比分析**

与传统投影Galerkin方法对比，Deep Hilbert‑Galerkin在更低维度下实现更小的PDE残差；数值实验显示残差误差可降至10^-4级别，Actor‑Critic在热方程控制问题中逼近最优价值函数与控制策略误差低且收敛速度与传统DNN方法相当。

**⚠️ 局限性**

受限于需要存在光滑经典解、正交基包含在算子定义域、以及满足的逼近条件；对无界算子、非凸控制、时间非平稳情况的收敛性理论尚不完整；实际训练仍受限于采样维度与计算资源。

---

## 415. Borderless Long Speech Synthesis

**arXiv ID:** 2603.19798 | [PDF](https://arxiv.org/pdf/2603.19798v1)

**作者:** Xingchen Song `[一作]` (Xiaomi Inc), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 Any2Speech 框架，实现边界无缝的长语音合成，包含“先标注后过滤”数据策略、Global‑Sentence‑Token 分层注释、基于 VibeVoice 的连续子词编码器，并通过 Chain‑of‑Thought (CoT) 规划和 Dimension Dropout 两种训练策略提升模型对复杂场景的指令遵循和表达能力。

**💡 创新点**

创新点包括：① 用标注而非过滤的方法保留高表达性、混叠、背景噪声等“脏”数据；② 设计 Global‑Sentence‑Token 三层分层注释作为结构化语义接口，完成从全局场景到单词级音素的可控控制；③ 在合成前引入 CoT 规划阶段，将理解和表达决策显式化；④ 采用 Dimension Dropout 让模型在部分指令缺失时仍保持高质量合成；⑤ 通过 LLM 与合成引擎的分层协议实现真正的 Native Agentic 语音合成。

**🔧 技术方法**

技术实现：VibeVoice 7B 连续子词分词器作为基础模型；Chain‑of‑Thought 规划网络；Dimension Dropout 训练策略；Global‑Sentence‑Token 分层注释；LLM 前端作为 Agent；多层语义接口（全球/句子/音素）；长音频无边界合成。

**📊 数据集**

使用多源公开语音语料（对话、播客、体育广播、采访等），保持原始“脏”数据并按 Global‑Sentence‑Token 进行细粒度标注；未单独使用任何专门的声效或音乐数据集。

**📈 对比分析**

由于评估标准尚未成熟，本文未给出数值化指标；通过主观听评和对比实验说明 CoT 与 Dimension Dropout 能显著提升指令遵循度和情感连贯性，但缺乏可量化的性能对比。

**⚠️ 局限性**

局限性：目前仅针对离线内容生成（播客、电子书、配音），不适用于实时交互；缺乏专门的声效和音乐数据，背景声的生成仅为偶然出现；未实现低延迟流式推理；尚未加入完整的声音克隆或实时表达调整功能。

---

## 416. Quantifying Gate Contribution in Quantum Feature Maps for Scalable Circuit Optimization

**arXiv ID:** 2603.19805 | [PDF](https://arxiv.org/pdf/2603.19805v1)

**作者:** F. Rodríguez-Díaz `[一作]` (Pablo de Olavide University), F. Martínez-Álvarez `[通讯]` (Pablo de Olavide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于门重要性指数（GSI）的Gate Assessment and Threshold Evaluation（GATE）方法，用于优化量子机器学习中的特征映射电路，显著减少门数和深度。

**💡 创新点**

创新点在于引入多维度（保真度、纠缠度、敏感度）量化的门重要性指标，提供可解释的门级筛选与阈值调节，兼容模拟与真实硬件。

**🔧 技术方法**

采用 Qiskit 及其机器学习模块实现 PegasosQSVM 与 QNN，利用密度矩阵、MPS、TN 以及实际 IBM 设备进行 GSI 计算与门裁剪，并评估加速与准确率。

**📊 数据集**

实验使用 PMLB 9 个二分类数据集（BreastW、Corral、Glass2、Monk、Flare、Vote、Saheart、Fitness、Heart），在理想模拟、噪声模拟和真实设备三种环境下进行。

**📈 对比分析**

与基线模型比较，门数可减少约 30–40%，执行时间平均降低 20–50%，多数数据集保持或提升准确率；最优配置往往在中等阈值处，表现出准确率与时间的最佳平衡。

**⚠️ 局限性**

局限性包括 GSI 估计受噪声影响、假设门贡献独立而未完全捕捉复杂相互作用、阈值选择需手动调节、仅适用于具备冗余的 QML 电路。

---

## 417. New Constructions of Polar Code Based on Refined Error Probability Analysis

**arXiv ID:** 2603.19589 | [PDF](https://arxiv.org/pdf/2603.19589v1)

**作者:** Hassan Noghrei `[一作]`, Murad Abdullah `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了极化码在SC和SCL解码下的块误码率（BLER）分析，并基于此提出了新的SC优化和SCL优化的极化码构造算法。

**💡 创新点**

创新点在于推导了基于LLR的全新BLER表达式，给出了路径损失和路径选择的概率公式，并利用这些公式实现了可针对任意对称B‑DMC的SC/SCL优化构造；此外，提出的构造方法在仿真中显著优于现有GA、Tal‑Vardy、5G标准等方案。

**🔧 技术方法**

使用了概率推导、LLR分析、递归构造、Monte‑Carlo估计、SCF/DSCF、CRC辅助SCL等技术，并与GA、Tal‑Vardy、RM极化码等基准进行对比。

**📊 数据集**

通过在AWGN、BSC、BEC等对称B‑DMC上生成随机样本进行仿真评估；未使用公开数据集。

**📈 对比分析**

采用仿真对比BLER曲线，在SC、DSCF、SCL（列表大小L=2,4,8）下与5G标准、GA、Tal‑Vardy、RM极化码等基准对比；SC优化方案在DSCF上提升0.15–0.2 dB，SCL优化方案在所有列表大小下提升0.1–0.4 dB。

**⚠️ 局限性**

局限性在于算法需要多次SCL/SC解码样本，计算复杂度较高；需要根据目标Eb/N0和列表大小手动调参；目前仅适用于对称B‑DMC，扩展到非对称或更复杂信道仍需进一步研究。

---

## 418. An Agentic Approach to Generating XAI-Narratives

**arXiv ID:** 2603.20003 | [PDF](https://arxiv.org/pdf/2603.20003v1)

**作者:** Yifan He `[一作]` (University of Antwerp), David Martens `[通讯]` (University of Antwerp)

**通讯引用:** 4332 | [OpenAlex ID](https://openalex.org/A5101474247)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于多代理系统的 SHAP 解释文本生成与迭代改进框架，目标是提升可解释性文本的可信度与可读性。

**💡 创新点**

创新点包括：①将 Narrator、Faithful Evaluator、Faithful Critic、Coherence Agent 组合成可迭代的多代理架构；②在评估与反馈中引入多轮迭代与规则化指令；③提出基于多数投票的提取集成方法以降低 LLM 提取误差。

**🔧 技术方法**

技术手段主要是使用大语言模型（Claude Sonnet 4.5、GPT-5、Mistral Medium 3.1、Llama 3.3 70B、DeepSeek‑V3.2‑Exp）进行文本生成与信息抽取；评估方面采用 Rank、Sign、Value 三维准确率以及质性 Coherence 反馈。

**📊 数据集**

实验使用五个二分类表格数据集：Student Performance、FIFA Man of the Match、German Credit、Diabetes、Stroke Prediction。

**📈 对比分析**

通过在不同 LLM 与五种代理设计下的多轮实验与 Ensemble 对比，最高的 faithfulness 准确率达到 0.996；Critic‑Rule 设计在大多数 LLM 上表现最佳，且 Ensemble 方法在绝大多数组合中提升了准确率。

**⚠️ 局限性**

局限性：①提取错误难以完全消除，导致 faithfulness 评估不稳定；②Coherence Agent 反馈有时会破坏 faithfulness；③缺乏定量的连贯性度量；④LLM 在同一文本上可能产生不同抽取结果，进一步增加不确定性。

---

## 419. Rising Prevalence of Detected AI-Generated Text in Medical Literature: Longitudinal Analysis in Open Access Articles

**arXiv ID:** 2603.19316 | [PDF](https://arxiv.org/pdf/2603.19316v1)

**作者:** Nathan Wolfrath `[一作]` (Medical College of Wisconsin), Anai N Kothari `[通讯]` (Medical College of Wisconsin)

**通讯引用:** 2208 | [OpenAlex ID](https://openalex.org/A5077843932)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对JAMA Network Open 2022‑2025年间的原始研究、研究信件和邀请评论进行文本提取和AI检测，评估AI生成内容的出现率及趋势。

**💡 创新点**

使用商业AI检测器Originality.AI在大型开放医学期刊中系统量化AI生成文本的流行度和趋势，并将检测结果与作者公开披露进行对比。

**🔧 技术方法**

用Python爬虫与BeautifulSoup提取正文，正则匹配作者披露，使用Originality.AI检测器计算≥10% AI文本概率并二值化。

**📊 数据集**

JAMA Network Open全量开放获取原始稿件（7251篇）从2022年1月至2025年3月。

**📈 对比分析**

通过时间序列、出版类型和学科分类进行Chi‑square检验和Mann‑Kendall趋势检验；检测到的AI文本比例从0%上升至11.3%，趋势显著（P<0.001），不同类型和领域差异显著。

**⚠️ 局限性**

检测器准确性受限，可能出现误报与漏报，披露样本极少，语言差异与编辑方式影响检测，无法精准估计真实AI使用比例。

---

## 420. Real-Time Optical Communication Using Event-Based Vision with Moving Transmitters

**arXiv ID:** 2603.19477 | [PDF](https://arxiv.org/pdf/2603.19477v1)

**作者:** Harmeet Dhillon `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5034884039)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

开发了一套基于事件摄像头的实时光通信系统，能够在发射器和接收器相对运动时实现文本数据的高频传输；

**💡 创新点**

提出Geometry-Aware Unscented Kalman Filter（GA-UKF），在保持椭圆形事件簇几何一致性的同时实现对运动中的光源位置、形状、方向的实时估计，并通过空间滤波与时间加权测量显著提升处理速度；

**🔧 技术方法**

采用事件摄像头、LED高速调制发射器、空间滤波器、GA-UKF（利用SPD流形与S¹环绕）和双阈值消隐解调技术；

**📊 数据集**

实验基于Prophesee EVK4事件摄像头、ESP32/LP55231 LED驱动器、无人机、手持LED以及旋转盘的实机硬件场景，不使用公开数据集；

**📈 对比分析**

与之前基于EKF的方法做对比：GA-UKF在10kHz信号下处理时延约763µs，EKF超过10kµs；在移动速度≤5k像素/秒时解码准确率≥90%，相较于EKF的误码率显著下降，且实现了约7倍的速度提升；

**⚠️ 局限性**

受限于LED光强、相机与发射器距离、突发加速度与背景光干扰、重叠发射器导致的协方差膨胀以及对摄像机运动的严格要求；未来可通过IMU融合、增大光强或多目标跟踪进一步提升鲁棒性。

---

## 421. DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs

**arXiv ID:** 2603.19688 | [PDF](https://arxiv.org/pdf/2603.19688v1)

**作者:** Xuan Qi `[一作]` (Tsinghua University), Xingyu Fu `[通讯]` (University of Pennsylvania)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5090327923)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在多模态大语言模型训练前预测不同训练数据对目标基准的影响，并提出无训练的影响评估指标。

**💡 创新点**

引入多模态困惑度、跨模态相似度和数据多样性三因素的乘积指标，能在训练前准确排序数据影响；证明直觉相似度不可靠。

**🔧 技术方法**

计算多模态困惑度、图像/文本相似度、问题聚类多样性；使用Kendall τ评估相关性；基于InternVL3模型进行微调实验。

**📊 数据集**

14个视觉-语言VQA数据集，覆盖7个任务族（OCR、图表、文档、通用VQA、空间推理、计数、地图推理），每个任务两个数据集。

**📈 对比分析**

与真实微调后相对性能提升进行排名比较，Kendall τ平均0.86；在固定训练样本下，用该指标进行数据加权或筛选，平均提升3.4%（真实数据）和6.9%（合成数据），超越统一采样和ICONs等基线。

**⚠️ 局限性**

仅在视觉问答类任务上验证，指标依赖基准模型的冻结编码器；对极端任务或不同模型结构的泛化未知；忽略了模型熟悉度、答案长度等因素的潜在影响。

---

## 422. GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce

**arXiv ID:** 2603.19665 | [PDF](https://arxiv.org/pdf/2603.19665v1)

**作者:** Zhouwei Zhai `[一作]` (JD.com), Jin Li `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了GenFacet——一种面向工业化电商场景的端到端生成式分面搜索框架；

**💡 创新点**

其创新点在于将分面生成和查询重写两项任务统一到单一大语言模型中，并通过教师-学生蒸馏结合GRPO的多任务对齐，实现了语义闭环与检索效果的同步优化；

**🔧 技术方法**

主要技术包括大语言模型（Qwen3‑4B）、链式思维蒸馏、监督多任务微调、GRPO强化学习、INT8量化与显式缓存等；

**📊 数据集**

使用了JD.com真实搜索日志构建的JD‑Facet数据集（5000个搜索会话），并结合教师模型生成的伪标签；

**📈 对比分析**

与规则基、DFDRF、零样本LLM等基线相比，GenFacet在离线指标上实现P@10+7.9%、R@10+45.0%、nDCG@10+15.1%，在上线A/B测试中Facet CTR提升42%、UCVR提升2%；

**⚠️ 局限性**

局限性包括对高质量标注依赖度高、生成模型可能出现幻觉、需要持续的数据循环训练以保持对新词汇的适应。

---

## 423. Evolving Jailbreaks: Automated Multi-Objective Long-Tail Attacks on Large Language Models

**arXiv ID:** 2603.20122 | [PDF](https://arxiv.org/pdf/2603.20122v1)

**作者:** Wenjing Hong `[一作]` (Shenzhen University), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 27979 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EvoJail框架，利用多目标进化搜索自动生成长尾分布攻击，以评估LLM的安全与隐私漏洞。

**💡 创新点**

创新点在于将语义-算法式表示与LLM辅助变异/交叉相结合，消除手工规则，系统性探索攻击Pareto前沿，实现自动化且多样化的长尾攻击生成。

**🔧 技术方法**

采用多目标进化算法、LLM辅助生成与修复操作、加密-解码对结构、语义模板化、输出困惑度评估等技术。

**📊 数据集**

使用GPTFuzzer和JBB-Behaviors两大数据集（共18个实例），在Llama-2-7b、Llama-3.1-8B、GPT-4.1-Nano等模型上进行实验。

**📈 对比分析**

通过与FlipAttack、CodeChameleon、CodeAttack、Jailbroken、Cipher、ReNeLLM等六种基线在ASR、PPL和Hypervolume指标上比较，EvoJail在18个场景中15场景击败基线，尤其在LLaMA系列模型表现突出。

**⚠️ 局限性**

局限性包括对不同模型安全策略的适应性尚不充分、对LLM解码能力的依赖，以及较高的计算成本和缺乏公开实现细节。

---

## 424. PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences

**arXiv ID:** 2603.19762 | [PDF](https://arxiv.org/pdf/2603.19762v1)

**作者:** Min Lin `[一作]` (Huazhong University of Science and Technology), Xin Yang `[通讯]` (Optics Valley Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了PCSTracker框架，能够在点云序列中进行长时段一致的场景流估计。

**💡 创新点**

创新点在于：① 迭代几何-运动联合优化模块（IGMO）显式建模几何随时间演化，① ① 空间-时序点轨迹更新模块（STTU）利用全局时序上下文填补遮挡；② 重叠滑动窗口推理策略，交叉窗口传播与窗口内细化交替进行，显著抑制误差累积。

**🔧 技术方法**

技术方法包括：PointConv特征提取、KNN插值初始化轨迹、点-体素双分支相关量构造、Transformer实现空间-时序依赖建模、滑动窗口迭代推理。

**📊 数据集**

使用了两大数据集：合成的PointOdyssey3D（24/40帧、8192点/帧）以及真实的ADT3D（150帧/序列，基于Aria Digital Twin）。

**📈 对比分析**

与SpatialTracker、SceneTracker、DELTA（RGB‑D方法）以及基于PV‑RAFT的SF‑baseline进行对比。PCSTracker在PointOdyssey3D上EPE_3D下降≈35%，在ADT3D上提升≈38%，同时保持32.5 FPS、3.48M参数，显著优于对比方法。

**⚠️ 局限性**

局限性在于对几何尺度和场景距离分布敏感，导致从合成到真实域迁移时性能下降，且在极端遮挡或极端尺度变化场景下仍可能出现漂移。

---

## 425. Zero Shot Deformation Reconstruction for Soft Robots Using a Flexible Sensor Array and Cage Based 3D Gaussian Modeling

**arXiv ID:** 2603.19543 | [PDF](https://arxiv.org/pdf/2603.19543v1)

**作者:** Linrui Shou `[一作]` (University of Notre Dame), Tingyu Cheng `[通讯]` (University of Notre Dame)

**通讯引用:** 635 | [OpenAlex ID](https://openalex.org/A5011017276)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个零射击、无摄像头的柔性机器人变形重建框架，利用柔性电阻触觉阵列、基于笼子控制的三维高斯点云实现实时RGB渲染。

**💡 创新点**

创新点包括：① 通过结构化笼子控制将稀疏触觉测量映射为全局几何的零射击推断；② 采用图注意网络实现空间平滑、结构连贯的变形传播；③ 结合4D Gaussian Splatting实现实时、真实感的可视化。

**🔧 技术方法**

采用的技术包括柔性电阻触觉阵列、图注意网络（GAT）、笼子控制的三维高斯点云、逆距离加权（IDW）变形传播、实时高斯渲染器。

**📊 数据集**

使用了在可控弹性基材上采集的20帧关键姿态结构光扫描数据，生成多视角视频并用4DGS训练得到高斯点云；随后在两台未见的软机器人上进行零射击测试。

**📈 对比分析**

与传统视觉、EIT和触觉/手套方法对比；在未见软机器人上得到IoU≈0.67、SSIM≈0.65、Chamfer≈3.48 mm、弯曲角误差≈4.7°，系统可实时30 fps。

**⚠️ 局限性**

局限性包括：① 仅通过几何插值，难以捕捉高度非线性材料行为；② 未感知区域误差较大，受触觉贴合与覆盖范围限制；③ 需要离线几何初始化和高斯模型构建，增加设置成本。

---

## 426. ItinBench: Benchmarking Planning Across Multiple Cognitive Dimensions with Large Language Models

**arXiv ID:** 2603.19515 | [PDF](https://arxiv.org/pdf/2603.19515v1)

**作者:** Tianlong Wang `[一作]` (University of Virginia), Sheng li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为ItinBench的基准，评估LLM在旅行行程规划中同时处理语言推理与空间推理的能力。

**💡 创新点**

创新点在于将空间推理（路线优化）与传统语言推理融合进同一任务，并通过文本化的空间信息提供实验，揭示LLM在多域推理中的性能折衷。

**🔧 技术方法**

采用大语言模型API（如Llama 3.1 8B、Mistral Large、Gemini 1.5 Pro、GPT‑4o、o1）以及自定义工具调用和TSP算法进行评估。

**📊 数据集**

使用Yelp城市商家数据和用户点评信息，结合人工生成的500条旅行偏好查询，形成了包含餐厅、酒店、景点等多类别的数据集。

**📈 对比分析**

与传统旅行规划基准相比，ItinBench在四个任务（全数据无路优化、全数据有路优化、过滤后有路优化、工具使用+路优化）下，对LLM的验证计划率、距离差距等指标进行量化；结果显示最高验证率仅约66%，路程增幅平均约15‑20%，表明LLM在双域推理时存在明显性能下降。

**⚠️ 局限性**

限制在于空间信息多以文本形式呈现，易让LLM仅凭语义推理完成任务，未真正考察其几何计算能力；此外实验仅覆盖单城市、固定约束，缺乏更复杂多城市或动态约束的评估。

---

## 427. LoopRPT: Reinforcement Pre-Training for Looped Language Models

**arXiv ID:** 2603.19714 | [PDF](https://arxiv.org/pdf/2603.19714v1)

**作者:** Guo Tang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16576 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LoopRPT，一种针对循环语言模型（LoopLM）的强化预训练框架，用于在隐层迭代计算中直接给出强化信号，从而提升推理质量并压缩推理步数。

**💡 创新点**

创新点包括：① 将 Reinforcement Pre‑Training（RPT）迁移到 LoopLM 上；② 使用 EMA 教师与噪声 latent rollouts 构造密集的步级奖励；③ 通过熵挑选“难”token，聚焦学习资源；④ 结合时间惩罚和步级加权目标，既优化内部表示又鼓励早退出。

**🔧 技术方法**

核心技术包括 LoopLM（Ouro架构）、RPT、EMA 教师、熵基 hard‑token 选择、噪声 latent rollouts、Policy Gradient、步级奖励、表示学习加权与 KL 约束。

**📊 数据集**

训练数据采用 Omni‑Math（约 4,428 题）；下游评测使用 MMLU、MMLU‑Pro、BBH、ARC‑C、HellaSwag、Winogrande、GSM8K、MBPP、HumanEval 等标准数据集。

**📈 对比分析**

与传统 LoopLM、Qwen3‑1.7B 的 vanilla 与 CoT 推理进行对比，LoopRPT 在所有规模上实现了更高的准确率（尤其是 hard tokens）并显著降低平均推理步数，形成了在准确率‑计算量上的 Pareto 优势；下游任务（如 GSM8K、MBPP+）亦获得显著提升。

**⚠️ 局限性**

局限性：① EMA 教师的移动平均可能导致收敛慢；② 在强制深度评估中准确率并非单调上升；③ 对更大规模模型和多样化数据混合的推广尚待验证；④ 早退出策略在分布偏移下的鲁棒性仍需进一步加强。

---

## 428. Inducing Sustained Creativity and Diversity in Large Language Models

**arXiv ID:** 2603.19519 | [PDF](https://arxiv.org/pdf/2603.19519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 429. Variational Encrypted Model Predictive Control

**arXiv ID:** 2603.19450 | [PDF](https://arxiv.org/pdf/2603.19450v1)

**作者:** Jihoon Suh `[一作]` (Purdue University), Takashi Tanaka `[通讯]` (Purdue University)

**通讯引用:** 1407 | [OpenAlex ID](https://openalex.org/A5082230661)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于变分推导的加密模型预测控制（VEMPC）协议，利用样本估计方法实现 MPC 的加密求解；

**💡 创新点**

创新点包括：① 将 MPC 约束问题变分化为无约束采样估计；② 通过指数倾斜将二次成本吸收到高斯采样分布中，省去在线加密矩阵乘法；③ 用 Chebyshev 多项式近似可行性指示器，完成全加密的约束检测；④ 设计双层并行（样本级和加密级）架构，显著降低运行时。

**🔧 技术方法**

使用技术包括：CKKS 同态加密、指数倾斜采样、Chebyshev 多项式逼近、单指令多数据（SIMD）打包、并行多线程。

**📊 数据集**

在论文中以模拟倒立摆系统为实验平台，采用线性化后离散化模型，未使用公开数据集。

**📈 对比分析**

与未加密的变分 MPC 进行对比，实验表明在 128‑bit 安全级别下，在线控制时间约为 28.7 ms（K=240，ℓ=3），能够满足采样周期；控制性能与未加密方法相当，成功驱动系统至目标状态并满足所有约束。

**⚠️ 局限性**

局限性包括：① 加密误差随安全参数和精度平衡；② 目前仅适用于线性二次 MPC 结构；③ 需要离线预处理和额外通信；④ 需要调参（λ、ℓ、τ_s、η）以保证可行性与收敛。

---

## 430. Parameter-Efficient Token Embedding Editing for Clinical Class-Level Unlearning

**arXiv ID:** 2603.19302 | [PDF](https://arxiv.org/pdf/2603.19302v1)

**作者:** Iyad Ait Hou `[一作]` (George Washington University), Aya Zirikly `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种在临床文本分类中进行行为级别类信息遗忘的轻量化方法——Sparse Token Embedding Unlearning (STEU)

**💡 创新点**

通过仅修改少量与目标类别相关的词向量以及分类头，而保持编码器冻结，实现参数高效的类遗忘；利用PMI挑选关键词并与轻量头结合，显著减少参数更新量

**🔧 技术方法**

PMI词选择、梯度遮蔽训练、BCE损失的正负样本平衡、轻量化分类头

**📊 数据集**

MIMIC‑IV、MIMIC‑III、eICU三大临床数据集，使用CCSR分组标签进行多标签分类

**📈 对比分析**

与梯度上升、直接抑制及影响权重的编码器级别遗忘方法比较，STEU在保持约0.48的平均F1的同时，仅更新约0.19%参数，遗忘效果近乎完美（forget F1≈0.0004）

**⚠️ 局限性**

仅实现行为级别遗忘，未提供正式的样本级删除保证；基于PMI的词挑选可能不如梯度或Fisher方法全面；实验仅单次种子，基线较少；不适用于生成任务或患者级删除

---

## 431. URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models

**arXiv ID:** 2603.19281 | [PDF](https://arxiv.org/pdf/2603.19281v1)

**作者:** Vinh Nguyen `[一作]` (Uppsala University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 18755 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了URAG基准，系统评估检索增强生成（RAG）模型的准确性和不确定性。

**💡 创新点**

创新点在于将开放式生成任务转化为多项选择问答，结合符合性预测（conformal prediction）实现统计学可靠的不确定性量化，并系统分析检索噪声、知识依赖和检索深度对模型不确定性的影响。

**🔧 技术方法**

使用RAG的多种架构（Fusion、HyDE、RAT、FiD等）、符合性预测方法（LAC、APS）、迭代式NLI过滤生成有效干扰选项、以及基准评测流水线。

**📊 数据集**

覆盖八大领域的8个数据集：CRAG、NewsSum、DialFact、SciFact、LCA、ODEX、OlympiadBench、HealthVer。

**📈 对比分析**

通过对比准确率、覆盖率和预测集大小等指标，发现简单模块化的RAG方法（如Fusion、Naive）在准确性与低不确定性之间取得最佳平衡；检索噪声会破坏准确性–不确定性负相关；不同领域和检索方式导致性能差异显著。

**⚠️ 局限性**

局限性包括：不确定性评估仅在MCQA框架内；缺乏单一最优RAG方案；对提示词、检索数据库和LLM参数的依赖；以及评测仅使用部分LLM模型，可能无法全面反映所有系统。

---

## 432. MuSteerNet: Human Reaction Generation from Videos via Observation-Reaction Mutual Steering

**arXiv ID:** 2603.20187 | [PDF](https://arxiv.org/pdf/2603.20187v1)

**作者:** Yuan Zhou `[一作]` (Nanyang Technological University), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 28020 | [OpenAlex ID](https://openalex.org/A5042324027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 MuSteerNet，用观察-反应互调机制实现视频驱动的人类反应生成。

**💡 创新点**

创新点在于通过 Prototype Feedback Steering 消除视觉观察与反应类别的关系失真，并引入 Dual-Coupled Reaction Refinement 进一步提升反应质量。

**🔧 技术方法**

使用的技术包括原型反馈调节器（带门控 Delta 校正模组）、双耦合反应细化、RVQ‑VAE 离散化、Masked Motion Modeling 以及 TwinMixer 观察融合。

**📊 数据集**

在 ViMo 数据集上进行评估，该数据集包含 3500 条视频与对应的 26 类反应动作。

**📈 对比分析**

与 HERO 等基线对比，MuSteerNet 在 FID、多样性和多模态性上均实现最佳或竞争性表现（FID 0.328，diversity 7.895，multimodality 1.648）。

**⚠️ 局限性**

局限性在于对单人交互的关注，尚未扩展到多模态输入或多人人交互场景。

---

## 433. Adaptive Greedy Frame Selection for Long Video Understanding

**arXiv ID:** 2603.20180 | [PDF](https://arxiv.org/pdf/2603.20180v1)

**作者:** Yuning Huang `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3987 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为长视频问答任务提出一种基于问题自适应的贪心帧选择方法

**💡 创新点**

通过结合查询相关性与语义覆盖的子模函数，并使用问题类型分类器动态调节相关性-覆盖权重，实现对不同问题意图的最优帧子集选择

**🔧 技术方法**

SigLIP 与 DINOv2 双空间嵌入、子模优化的贪心算法、轻量级文本分类器、设施位置（facility‑location）覆盖函数

**📊 数据集**

MLVU 长视频理解基准（提供九类问题标签）

**📈 对比分析**

与均匀采样和最新的 AKS 基线对比，实验表明在相同帧预算下，方法在所有预算区间均优于对手，尤其在低帧数时提升显著

**⚠️ 局限性**

仅在 MLVU 上验证，未考察跨数据集泛化；分类器误判虽小但可能影响性能；预设策略有限，缺乏更细粒度的自适应调节

---

## 434. CFCML: A Coarse-to-Fine Crossmodal Learning Framework For Disease Diagnosis Using Multimodal Images and Tabular Data

**arXiv ID:** 2603.20016 | [PDF](https://arxiv.org/pdf/2603.20016v1)

**作者:** Tianling Liu `[一作]` (Tianjin University), Liang Wan `[通讯]` (Tianjin University)

**通讯引用:** 3572 | [OpenAlex ID](https://openalex.org/A5000209938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种粗到细的跨模态学习框架（CFCML），用于结合多模态医学影像与表格数据进行疾病诊断。

**💡 创新点**

创新点包括：1）多粒度跨模态信息增强（MG‑CIE）模块，利用多层编码器输出的多粒度特征与表格信息进行交叉注意力，初步缩小模态差距并丰富单模态特征；2）类感知跨模态关系挖掘（CCRM）策略，构建跨模态和单模态原型，并通过层级锚点的对比学习将同类样本聚类、不同类样本分离，进一步消除模态边界。

**🔧 技术方法**

主要技术手段：多层影像编码器（nnMamba / Swin Transformer），预训练 CLIP 文本编码器用于表格嵌入，Conv1d 进行 token 归一化，跨模态多头注意力，层级锚点对比学习（sample‑anchor、unimodal‑proto‑anchor、crossmodal‑proto‑anchor），MLP 分类器和交叉熵损失。

**📊 数据集**

使用的公开与私有数据集：MEN（脑膜瘤 3 级 MRI + 临床表格，796 名患者）和 Derm7pt（皮肤病变图像 + 临床表格，827 名病例）。

**📈 对比分析**

与多类最先进方法（不确定性、特征解耦、注意力、MRIM 等）进行对比，CFCML 在 MEN 上 AUC 提升 1.53%，Derm7pt 上 AUC 提升 0.91%，并在多数评估指标（ACC、macro‑F1、AUPRC 等）上均优于对手，差异在统计检验中显著。

**⚠️ 局限性**

局限性：1）多粒度特征交互导致计算开销和参数量增加；2）需要手动调节影像与表格的 token 映射数，缺乏自动化选择方案。

---

## 435. FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients

**arXiv ID:** 2603.19722 | [PDF](https://arxiv.org/pdf/2603.19722v1)

**作者:** Tian Wen `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Bo Han `[通讯]` (TMLR Group Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分布式无标签噪声的联邦学习中，提出FedRG通过自监督生成球面表示，利用vMF混合模型与标签对应的几何分布一致性来识别噪声样本，并引入个性化噪声吸收矩阵进行鲁棒优化。

**💡 创新点**

创新点是将噪声识别从基于损失的“小损失”假设转向“表示几何优先”原则，利用球面几何和vMF分布进行噪声检测，并结合个性化噪声吸收矩阵。

**🔧 技术方法**

用到了SimCLR自监督学习、von Mises–Fisher混合模型、几何一致性度量、对称交叉熵、噪声吸收矩阵以及联邦平均等技术。

**📊 数据集**

在CIFAR-10、SVHN、CIFAR-100三个图像分类数据集上进行实验。

**📈 对比分析**

与FedAvg、FedProx、MOON、FedClean等传统和噪声鲁棒方法对比，FedRG在四种噪声场景下均显著提升准确率、F1和清洁样本识别准确率。

**⚠️ 局限性**

主要局限在仅验证图像分类任务，未扩展到图或文本域，且对极端噪声率或更复杂的联邦架构需进一步评估。

---

## 436. SaFRO: Satisfaction-Aware Fusion via Dual-Relative Policy Optimization for Short-Video Search

**arXiv ID:** 2603.19585 | [PDF](https://arxiv.org/pdf/2603.19585v1)

**作者:** Renzhe Zhou `[一作]` (Kuaishou Technology), Jingwei Zhuo `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套名为 SaFRO 的短视频搜索多任务融合框架，结合满意度感知奖励模型、Dual-Relative Policy Optimization 和 Task-Relation-Aware Fusion 模块，以优化即时排序质量与长期用户满意度。

**💡 创新点**

创新点包括：① 利用查询重构率与会话间隔等查询级代理构建满意度奖励；② 设计 Dual-Relative Policy Optimization，融合组内和批次间优势归一化，提升策略学习稳定性；③ 通过 Task-Relation-Aware Fusion 模块显式建模不同任务间的相互关系，实现上下文感知的权重自适应。

**🔧 技术方法**

主要技术：强化学习（PPO 变体）、离线批量 RL、深度奖励建模、注意力任务关系网络、基于对数变换的多任务融合、对数概率分布策略。

**📊 数据集**

使用 Kuaishou 规模化短视频搜索工业数据集，约 400M 日活用户、806M 物品、2.15B 会话，训练/测试比例 9:1。

**📈 对比分析**

与传统学习排序方法（LambdaRank、NeuralNDCG）、黑盒优化（CEM）、基于值的 RL（DDPG、TD3、SAC）以及专注保留的 RL（Batch-MTF、RLUR、AURO）等基线比较，SaFRO 在 NDCG@10、满意度得分、用户保留率等指标均超越所有对手；在线 A/B 测试显示 CTR 上升 0.136%，LPR 上升 0.495%，持续提升用户保留。

**⚠️ 局限性**

局限性：依赖稀疏的搜索交互数据，奖励模型对噪声敏感；对其他搜索领域的泛化性未验证；RL 训练需要大量离线数据，模型解释性较弱；部署成本和实时性要求高。

---

## 437. Communication Complexity of Disjointness under Product Distributions

**arXiv ID:** 2603.19490 | [PDF](https://arxiv.org/pdf/2603.19490v1)

**作者:** Zach Hunter `[一作]` (ETH Zurich), Istvan Tomon `[通讯]` (Umea University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

给出了在产品分布下集合不相交函数的随机通信复杂度的简洁上界证明，并改进了对误差参数的量化依赖。

**💡 创新点**

提出了一个新的组合子 lemma，能在两个独立分布下抽取大规模互不相交的子族，从而构造大尺寸的单色矩形，进而实现更紧凑的协议；并将该结果推广到具有有限互信息的分布。

**🔧 技术方法**

使用了组合子 lemma、Nisan–Wigderson 归约、Håstad–Wigderson 协议以及 Substate 定理进行误差与互信息的控制。

**📊 数据集**

本研究为理论研究，不使用具体数据集，所有结果均为信息理论上的证明。

**📈 对比分析**

通过上述方法得到的通信复杂度为 O(√(n log(1/ε)))，与已知下界 Ω(√n) 对齐，证明了在产品分布下的最优性；在互信息受限的情形下实现了 O(√(n(k+1))/ε) 的上界。

**⚠️ 局限性**

证明仅适用于产品或互信息受限的分布，且依赖于组合子 lemma 的特定性质；对更一般的分布或其他函数的推广仍存在挑战。

---

## 438. LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models

**arXiv ID:** 2603.19255 | [PDF](https://arxiv.org/pdf/2603.19255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 439. Eye Gaze-Informed and Context-Aware Pedestrian Trajectory Prediction in Shared Spaces with Automated Shuttles: A Virtual Reality Study

**arXiv ID:** 2603.19812 | [PDF](https://arxiv.org/pdf/2603.19812v1)

**作者:** Danya Li `[一作]` (Technical University of Denmark), Rico Krueger `[通讯]` (Technical University of Denmark)

**通讯引用:** 1289 | [OpenAlex ID](https://openalex.org/A5038885838)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在虚拟现实环境下模拟共享空间中人行人与自动穿梭车的互动，记录行人运动轨迹、细粒度眼动和情境变量，并基于这些数据训练多模态 LSTM 预测模型。

**💡 创新点**

①提出首个将眼动信息（尤其是连续眼方向）与情境因素融合的预测框架 GazeX‑LSTM；②发现眼动相较于仅用头部方向更能提升轨迹预测精度；③证明情境变量与眼动共同作用具有叠加效应。

**🔧 技术方法**

使用 LSTM 编码器对运动、距离、眼动和情境进行特征融合，随后通过全连接解码器输出未来轨迹；评估使用眼动方向（eye‑vislet）和语义目标标签；对比头部方向替代方案。

**📊 数据集**

VR 收集的数据集：51 名受试者共 537 条试验，每条记录包含 2 秒的轨迹、速度、与车距离、眼动方向/注视目标、eHMI 状态、车道接近角度、连续交通等。

**📈 对比分析**

与仅基于轨迹/速度/距离的基线 LSTM 对比。使用 ADE、FDE 作为指标，发现：①眼动方向（vislet）使 ADE/FDE 降低约 3–4%；②加入情境变量后，长期预测提升明显；③眼动与情境叠加可进一步提升 2–3%；头部方向对性能影响微乎其微。

**⚠️ 局限性**

受试者样本以大学生为主，缺乏多样性；实验仅包含单一行人与至多两辆车，未覆盖群体或更复杂交互；仅利用眼动方向，未整合瞳孔、心率等生理信号；眼动在复杂环境中的鲁棒性待验证。

---

## 440. Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning

**arXiv ID:** 2603.20116 | [PDF](https://arxiv.org/pdf/2603.20116v1)

**作者:** Jiajie Li `[一作]` (University at Buffalo), Jinjun Xiong `[通讯]` (University at Buffalo)

**通讯引用:** 6795 | [OpenAlex ID](https://openalex.org/A5030156276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种基于强化学习的Chain‑of‑Adaptation框架，用于在保留通用视觉语言能力的前提下，使VLM在外科任务上实现更好的领域适配。

**💡 创新点**

将四阶段结构化推理与RLVR（GRPO）相结合，利用伪标注冷启动并在可验证奖励下优化，显著缓解了SFT的过拟合与忘记问题。

**🔧 技术方法**

RL从人类反馈（RLVR）+ GRPO优化、四阶段Chain‑of‑Adaptation思维格式、伪标注生成、Qwen‑VL模型以及MMBench/MMStar评测。

**📊 数据集**

CholecT50、EndoVis2018外科物体识别基准、GraSP跨域评测，以及约1万张未标注外科图像做冷启动伪标注。

**📈 对比分析**

与SFT基线和无思维格式RLVR对比，CoA在两任务上F1提升约10–27%，在跨域GraSP上保持优势，显著减少过拟合与提升泛化。

**⚠️ 局限性**

依赖可验证奖励和伪标注的质量；对极少数类别仍有限泛化；可能无法完全消除SFT导致的语言偏差，且需要较多计算资源。

---

## 441. Wildfire Spread Scenarios: Increasing Sample Diversity of Segmentation Diffusion Models with Training-Free Methods

**arXiv ID:** 2603.20188 | [PDF](https://arxiv.org/pdf/2603.20188v1)

**作者:** Sebastian Gerard `[一作]` (KTH Royal Institute of Technology), Josephine Sullivan `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 10363 | [OpenAlex ID](https://openalex.org/A5088465676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何在不确定环境（如野火扩散、医学影像、自动驾驶）中使用扩散模型实现多模态分割，并提出了无训练的采样策略以提高样本多样性。

**💡 创新点**

创新点包括：1) 将粒子引导（Particle Guidance）和SPELL方法从自然图像生成迁移至离散二值分割；2) 提出基于k‑中位数聚类与Chamfer距离的剪枝策略；3) 将SPELL的屏障半径与像素差直接对应，简化超参数调优；4) 新建MMFire模拟火灾扩散多模态数据集作为基准。

**🔧 技术方法**

采用的技术有：EDM扩散框架、粒子引导、SPELL、k‑中位数聚类、Chamfer距离、概率UNet、记忆池（memory bank）、不同噪声级别控制、以及多批次采样。

**📊 数据集**

使用了三种数据集：LIDC（肺部结节医学图像）、改造的Cityscapes（二值多模态语义分割）、以及自制的MMFire（多路径野火扩散模拟）。

**📈 对比分析**

通过与传统的随机采样和Probabilistic U‑Net进行对比，粒子引导和SPELL在MMFire上单批次提升HM IoU*最高达7.5%，在Cityscapes上提升至16.4%；聚类剪枝在Cityscapes上提升2.4%，在MMFire上提升>15%；在LIDC上，尽管两方法低于Probabilistic U‑Net，但均优于随机采样；多批次采样进一步提升多样性与质量。

**⚠️ 局限性**

局限性包括：MMFire缺乏真实环境的多样性；聚类剪枝在模式分离不充分时导致性能下降；SPELL和粒子引导在低噪声阶段可能降低图像质量；方法均为无训练的采样技巧，尚未探索训练式增量；大规模采样计算成本高；仅在二值分割任务上验证，扩展到多类需进一步研究。

---

## 442. Pedestrian Crossing Intent Prediction via Psychological Features and Transformer Fusion

**arXiv ID:** 2603.19533 | [PDF](https://arxiv.org/pdf/2603.19533v1)

**作者:** Sima Ashayer `[一作]` (University of Tennessee at Chattanooga), Mina Sartipi `[通讯]` (University of Tennessee at Chattanooga)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5065246264)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量化、社会化多流Transformer架构，利用心理、位置、情境、交互四种结构化特征预测行人过街意图。

**💡 创新点**

将四种语义流以四token Transformer融合，并通过变分瓶颈与马氏距离实现不确定性校准与风险检测，兼顾可解释性与效率。

**🔧 技术方法**

高铁编码器、4-token Transformer、全局自注意力池化、KL散度+马氏距离不确定性头、MixUp与标签平滑等技术。

**📊 数据集**

使用PSI 1.0与新发布的PSI 2.0数据集，数据包含结构化行为注释和驾驶员语言解释。

**📈 对比分析**

在PSI 1.0上取得F1 0.90、AUC-ROC 0.94、MCC 0.78，优于ClipCross等基线；在PSI 2.0给出基准F1 0.78、AUC 0.79。

**⚠️ 局限性**

仅依赖结构化特征，缺少对原始视频的时序建模；对极端稀疏或多模态输入的泛化仍需进一步验证。

---

## 443. FREAK: A Fine-grained Hallucination Evaluation Benchmark for Advanced MLLMs

**arXiv ID:** 2603.19765 | [PDF](https://arxiv.org/pdf/2603.19765v1)

**作者:** Zhihan Yin `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Dongyan Zhao `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FREAK（Fine‑grained Evaluation Against Knowledge）基准，用于细粒度多模态大语言模型的幻觉评估。

**💡 创新点**

创新点在于：① 自动化生成对常识的局部矛盾（CCS）图像的完整管线；② 结合检测、计数、属性、分析、位置、OCR 六大细粒度子任务；③ 采用客观评判方法与人类基准相结合，揭示现有模型在细节感知上的巨大差距。

**🔧 技术方法**

使用的技术包括：LLM（用于生成 CCS 描述和问题）、先进的图像生成模型（Seedream3.0）与图像编辑模型（SeedEdit3.0）进行图像合成与局部修改、Chain‑of‑Thought 以及 RL‑fine‑tuned 推理、LLM‑as‑judge 评判机制。

**📊 数据集**

使用的数据集为 FREAK，包含 1,799 题目（1,000 选择题 + 799 开放式题），通过 LLM、生成与编辑模型自动化合成，再由人工验证；参考的对比基准包括 POPE、AMBER、HallusionBench 等。

**📈 对比分析**

评估方法：对所有模型进行多选题和开放式题的准确率与幻觉率评估，并与 100 名本科生的无经验基准对比。结果显示，SOTA 模型在 FREAK 上仅达 45% 准确率，幻觉率与准确率相近或更高；人类基准为 86%；CoT 推理往往导致准确率下降。

**⚠️ 局限性**

局限性：数据规模相对较小，且所有样本均需人工审核，难以扩展；生成与编辑模型可能引入细微偏差或视觉伪影；CoT 推理在细粒度幻觉评估中的表现仍需深入研究。

---

## 444. Kumo: A Security-Focused Serverless Cloud Simulator

**arXiv ID:** 2603.19787 | [PDF](https://arxiv.org/pdf/2603.19787v1)

**作者:** Wei Shao `[一作]` (University of California), Chongzhou Fang `[通讯]` (Rochester Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 Kumo，一款专注于服务器无关安全的模拟器，用于可控可重复地分析调度与资源共享导致的安全风险。

**💡 创新点**

创新点在于将调度、容器复用、资源争用等系统级机制显式建模，并支持攻击者与受害者实体化，能够量化共定位概率、首次共定位时间、丢包率与尾部延迟，突破现有仅关注性能的模拟器。

**🔧 技术方法**

使用离散事件模拟框架，提供可插拔调度器抽象、工作负载生成器、攻击模块，支持多种调度策略（Random、DoubleDip、Helper、OpenWhisk）与攻击强度。

**📊 数据集**

使用合成的服务器无关工作负载（Poisson、Burst 等），无公开真实数据集；通过实验配置文件模拟 512 工作节点、200 租户、4000 函数等规模。

**📈 对比分析**

通过与四种调度器、不同攻击强度、队列长度和集群规模的参数扫描进行对比；案例研究 A 证明调度器可导致共定位概率从 0 到 30% 的阶差；案例 B 显示在资源饱和时调度器影响有限，队列长度与集群扩展是主导因素；实验在单核机器上完成，平均每运行约 2–5 秒，表明可大规模探索。

**⚠️ 局限性**

局限在于不建模微架构侧通道、网络级 DoS，无法捕获特定平台的专有调度细节；仅聚焦系统级行为，若需更细粒度或真实平台验证需进一步扩展。

---

## 445. RAM: Recover Any 3D Human Motion in-the-Wild

**arXiv ID:** 2603.19929 | [PDF](https://arxiv.org/pdf/2603.19929v1)

**作者:** Sen Jia `[一作]` (University of Washington), Lei Li `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 8684 | [OpenAlex ID](https://openalex.org/A5100440388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出RAM框架，能够在单目视频中实时实现多人人体3D运动恢复。

**💡 创新点**

创新点在于将运动感知的语义跟踪、记忆增强的时序HMR、运动预测器以及门控融合器相结合，实现零样本、低ID切换、稳定追踪与光滑恢复。

**🔧 技术方法**

使用SAM2改进的SegFollow、Kalman滤波、MemFormer Transformer、SMPL回归与轻量级预测器。

**📊 数据集**

在PoseTrack、3DPW、TrackID-3x3、Olympic Boxing等公开数据集进行评估。

**📈 对比分析**

与4DHumans、CoMotion等方法对比，零样本下在PoseTrack的HOTA达66.4、MOTA 74.4、FPS 10.32，3D MPJPE 53.0，显著优于前者，展现出卓越的追踪稳定性与重建精度。

**⚠️ 局限性**

仍依赖语义分割的质量，对极端遮挡下的置信度估计有限，未针对多人之间的姿态互相干扰进行专门建模。

---

## 446. GO-GenZip: Goal-Oriented Generative Sampling and Hybrid Compression

**arXiv ID:** 2603.20109 | [PDF](https://arxiv.org/pdf/2603.20109v1)

**作者:** Pietro Talli `[一作]` (University of Padova), Andrea Zanella `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于目标导向的生成式混合采样与压缩框架GO-GenZip，用于网络遥测数据的高效采样与压缩。

**💡 创新点**

创新点在于联合自适应采样与混合压缩策略，并通过端到端训练使采样和压缩决策与下游任务性能直接相关。

**🔧 技术方法**

采用了生成式自编码器、Gumbel‑Softmax采样、双重拉格朗日约束以及传统无损压缩算法LZMA等技术。

**📊 数据集**

使用了在10天内从1162个4G基站收集的实时关键性能指标（34维KPI）数据集。

**📈 对比分析**

与仅使用无损压缩或单一生成式压缩进行对比，GO‑GenZip在相同采样率和压缩率条件下将重构误差降低至少20%，并实现超过50%的采样和传输成本下降。

**⚠️ 局限性**

局限在于对生成式模型的训练需求较高，且在极低采样率（≈0.2）下混合压缩优势不明显，未验证对更稀疏或更高维度数据的适用性。

---

## 447. NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing

**arXiv ID:** 2603.19864 | [PDF](https://arxiv.org/pdf/2603.19864v1)

**作者:** Raphael Simon `[一作]` (Royal Military Academy), Pieter Libin `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5068276498)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

开发并评估了一种基于JAX的网络攻击仿真器NASimJax，用于自动渗透测试的强化学习研究

**💡 创新点**

1) 将渗透测试建模为上下文POMDP并提供结构多样、可保证可解的网络生成管线；2) 通过两阶段动作选择（2SAS）解决线性增长的动作空间；3) 在JAX上实现GPU加速，单GPU每秒可达1.6M步，提升约100倍；4) 对比不同无监督环境设计方法（Domain Randomization、Prioritized Level Replay）对零射转移的影响

**🔧 技术方法**

JAX框架（JIT编译、向量化）、PPO强化学习、无监督环境设计算法（DR、PLR、PLR^）、自定义网络生成器（Topology density、Service/Process density、Sensitive host density）

**📊 数据集**

使用NASimJax自带的随机网络生成器，参数配置覆盖16、26、40主机规模，实验数据集中包含多种Topology density、Service/Process density与Sensitive host density组合；未使用公开标准数据集

**📈 对比分析**

实验显示NASimJax速度比原NASim快约100×；在动作空间缩放实验中，2SAS在26、40主机网络的解码率分别为82%/42%，远高于动作掩码（66%/14%）；在零射转移实验中，低密度训练（t_d低）通过PLR方法实现更好的泛化，DR方法在高密度下性能显著下降

**⚠️ 局限性**

1) 对于大型网络，PLR+2SAS会因回放缓冲区未填充导致学习崩溃；2) 仍需要进一步研究动作空间更高维度（服务/进程数量增加）下的可扩展性；3) 目前仅在CPU+单GPU环境验证，跨多GPU或分布式训练仍待评估

---

## 448. Design-OS: A Specification-Driven Framework for Engineering System Design with a Control-Systems Design Case

**arXiv ID:** 2603.20151 | [PDF](https://arxiv.org/pdf/2603.20151v1)

**作者:** H. Sinan Bank `[一作]` (Colorado State University), Thomas H. Bradley `[通讯]` (Colorado State University)

**通讯引用:** 8645 | [OpenAlex ID](https://openalex.org/A5025890846)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种轻量级的、基于规范的工程系统设计工作流 Design‑OS，并通过两种倒立摆平台演示其可扩展性。

**💡 创新点**

创新点在于将规范驱动的 AI 辅助流程从软件迁移到物理系统设计，构建跨阶段的可追溯文档、验证门和人工检查点，提升人机协作与设计可复现性。

**🔧 技术方法**

采用大型语言模型（Claude、Gemini）进行文献检索、概念设计、规范生成与任务规划，使用 Markdown、JSON 等结构化 artefacts 作为 AI 与人类交互的契约。

**📊 数据集**

并未使用传统数据集，而是依赖公开文献、开源硬件规格（SimpleFOC 反转盘、Quanser Furuta）及其官方参数文档。

**📈 对比分析**

通过同一 Design‑OS 流程在 Claude 与 Gemini 上并行执行，比较各阶段产物完整度、错误率与人工修正次数，结果显示两者在概念与需求阶段相近，但 Gemini 在规范细节和任务规划上更为精确；整体性能保持一致。

**⚠️ 局限性**

局限性包括：规范维护增加前期成本；需要严格遵循阶段顺序，否则易跳过；全链路可追溯依赖工具支持，安全关键系统需更严谨的验证门；在新颖或不熟悉的系统中难以预先制定完整规范。

---

## 449. Ontology-Based Knowledge Modeling and Uncertainty-Aware Outdoor Air Quality Assessment Using Weighted Interval Type-2 Fuzzy Logic

**arXiv ID:** 2603.19683 | [PDF](https://arxiv.org/pdf/2603.19683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. StreetForward: Perceiving Dynamic Street with Feedforward Causal Attention

**arXiv ID:** 2603.19552 | [PDF](https://arxiv.org/pdf/2603.19552v1)

**作者:** Zhongrui Yu `[一作]` (Li Auto Inc), Kun Zhan `[通讯]` (Li Auto Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无姿态、无追踪、无分割的前向4D动态街道重建框架StreetForward。

**💡 创新点**

创新点包括：引入因果遮罩注意力实现时间方向性建模；无监督速度解码器结合局部刚性与双向一致性正则化；统一使用3D高斯分布同时建模静态与动态场景。

**🔧 技术方法**

技术手段为：VGGT的交替注意力网络、DINO视觉编码器、3D高斯分散渲染（3DGS）、因果遮罩注意力模块、速度解码器以及跨帧渲染一致性正则化。

**📊 数据集**

使用的数据集为Waymo开放数据集进行训练与评估，并在Carla数据集上进行零射手迁移测试。

**📈 对比分析**

与STORM、DGGT、AnySplat、PVG等基线对比，StreetForward在动态区域的PSNR/SSIM显著领先，深度RMSE最低；在Carla的零射手测试中同样取得最优或接近最优表现。

**⚠️ 局限性**

局限性包括：对极端遮挡或大视角变换的鲁棒性仍待提升；长时间连续动态轨迹的精度有限；模型依赖于大规模Waymo数据的预训练与训练，迁移到完全不同场景时可能需进一步微调。

---

## 451. NEC-Diff: Noise-Robust Event-RAW Complementary Diffusion for Seeing Motion in Extreme Darkness

**arXiv ID:** 2603.20005 | [PDF](https://arxiv.org/pdf/2603.20005v1)

**作者:** Haoyue Liu `[一作]` (Huazhong University of Science and Technology), Luxin Yan `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 3976 | [OpenAlex ID](https://openalex.org/A5075653923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散模型的事件‑RAW 混合图像重建框架 NEC‑Diff，用以在极低照度下实现高质量动态场景成像。

**💡 创新点**

核心创新包括：①利用 RAW 图像的线性光响应与事件的亮度变化构建物理驱动的噪声抑制约束；②通过 SNR 引导的可靠信息提取动态平衡两模态信息；③将跨模态注意力融入扩散模型，提升纹理细节与光照一致性。

**🔧 技术方法**

技术手段涵盖 RAW 图像和事件的联合去噪（ECNS）、SNR‑引导特征抽取（SRIE）、跨模态注意力扩散（CAD）以及基于 DDIM 的确定性采样扩散重建。

**📊 数据集**

主要使用了新构建的 REAL 数据集（47,800 像素级配对的低照度 RAW、事件与高质量 sRGB 参考，照度 0.001–0.8 lux），并在合成的 LLRVD‑simu 数据上进行验证。

**📈 对比分析**

在多类基准（sRGB、RAW、事件、混合）中与最新方法对比，NEC‑Diff 在 REAL 上 PSNR 24.51、SSIM 0.742、LPIPS 0.201，显著优于其它方法；在 LLRVD‑simu 上 PSNR 27.74、SSIM 0.828、LPIPS 0.125，表现出色。

**⚠️ 局限性**

局限性在于事件阈值 C 的数据学习对不同设备或阈值变化的鲁棒性不足，且仅针对单一光照级别，未来需探索自适应阈值或测试时微调以提升泛化能力。

---

## 452. CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio Captioning Evaluation

**arXiv ID:** 2603.19615 | [PDF](https://arxiv.org/pdf/2603.19615v1)

**作者:** Insung Lee `[一作]` (Sogang University), Myoung-Wan Koo `[通讯]` (Sogang University)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5112696316)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种无参考音频字幕评估指标CAF-Score，将CLAP的粗粒度语义匹配与LALM的细粒度语义推理与句法识别相结合；

**💡 创新点**

创新点在于将CLAP的对比学习与LALM的FLEUR概率平滑相结合，采用滑窗最大池化提升语义对齐；

**🔧 技术方法**

使用技术包括CLAP模型、AudioFlamingo3/ Qwen3-Omni LALM、FLEUR概率评分、滑窗最大池化与α加权融合；

**📊 数据集**

实验数据集涵盖BRACE（Main与Hallucination）、AudioCaps、Clotho、RELATE、PAM等多种音频/文本对齐任务；

**📈 对比分析**

与CLAPScore、FENSE、CLAIR-A、AQA-Score等基线比较，CAF-Score在BRACE上实现最高人类评估相关性，且在RELATE和PAM上均优于单一CLAP或LALM评估，性能提升可达10%以上；

**⚠️ 局限性**

局限性包括对大模型推理的高计算成本、固定α权重可能不适用于所有案例、双重失效时无法纠正，以及仅在现有基准上验证，跨任务通用性仍待进一步探索。

---

## 453. Breeze Taigi: Benchmarks and Models for Taiwanese Hokkien Speech Recognition and Synthesis

**arXiv ID:** 2603.19259 | [PDF](https://arxiv.org/pdf/2603.19259v1)

**作者:** Yu-Siang Lan `[一作]` (National Yang Ming Chiao Tung University), Yuan-Fu Liao `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5082271172)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Breeze Taigi 框架，构建标准化的台语 ASR 与 TTS 评测基准，并提供基准数据与参考实现。

**💡 创新点**

利用官方台语‑普通话并行音频构建可复现评测基准；使用约 10,000 小时台语合成语音训练 Whisper 与 CosyVoice2 模型；结合自动 CER 与人工多维度评估，首次在台语领域实现统一基准与多维性能对比。

**🔧 技术方法**

Whisper 语音识别微调、CosyVoice2 语音合成、语音合成大规模数据生成、ASR 自动评估、人工评测等技术。

**📊 数据集**

30 对官方公开服务公告的台语‑普通话音频；约 10,000 小时台语合成语音；以及公开的台湾语料与大规模合成数据。

**📈 对比分析**

采用字符错误率（CER）对 ASR 进行统一评测，BreezeASR‑Taigi CER 为 30.13%，优于其他系统；TTS 评测采用 CER、人工发音准确率与 MOS，BreezyVoice‑Taigi CER 19.09%、MOS 5.0、发音准确率 59.2%；Aten AI Voice 最高发音准确率 89.8%。

**⚠️ 局限性**

评测依赖将台语音频映射为普通话字符，CER 不是绝对准确；代码切换行为导致发音评估低；人工评测主观性与样本覆盖范围有限；合成数据可能不完全覆盖真实台语变体。

---

## 454. Multilingual Hate Speech Detection and Counterspeech Generation: A Comprehensive Survey and Practical Guide

**arXiv ID:** 2603.19279 | [PDF](https://arxiv.org/pdf/2603.19279v1)

**作者:** Zahra Safdari Fesaghandis `[一作]` (Bilkent University), Suman Kalyan Maity `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 387 | [OpenAlex ID](https://openalex.org/A5102017334)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了多语言仇恨言论检测与对抗性言论生成的研究现状与实践指南

**💡 创新点**

提出了三阶段框架（任务设计、数据策划、评估）并聚焦低资源语言、隐式仇恨、跨语言与多模态挑战

**🔧 技术方法**

整合了BERT、XLM-R、mT5、Meta-Learning、GAN、LLM指令微调等技术，以及人类在环、可解释性与公平度量

**📊 数据集**

使用了多语种数据集如HatEval、HASOC、Multicomb、CONAN、IndicCONAN、ML-MTCONAN-KN、低资源语料（Amharic、Bengali、Basque等）

**📈 对比分析**

与现有多语言基线（BERT、XLM-R、mT5）对比，报告宏观F1、BLEU、JudgeLM等指标，显示跨语言迁移和少量样本学习的可行性，但低资源语言仍显劣势

**⚠️ 局限性**

局限包括过度依赖文本、缺乏多模态覆盖、标签定义不够动态、评估指标偏重表面匹配、数据集偏向社交媒体且受语言偏见影响

---

## 455. Vulnerability Analysis of eBPF-enabled Containerized Deployments of 5G Core Networks

**arXiv ID:** 2603.19867 | [PDF](https://arxiv.org/pdf/2603.19867v1)

**作者:** Yash Deshpande `[一作]` (Indian Institute of Technology Jammu), Samaresh Bera `[通讯]` (Indian Institute of Technology Jammu)

**通讯引用:** 1969 | [OpenAlex ID](https://openalex.org/A5035844388)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文在容器化部署的5G核心网络中，系统性评估了eBPF的安全漏洞，并通过实验演示了追踪、DoS、信息窃取及Bash注入等多种攻击手段。

**💡 创新点**

创新之处在于首次将eBPF在5G容器化环境中的攻击面与实际的eUPF、Open5GS实现相结合，展示了从容器权限到内核级恶意代码的攻击链，并提出了多层防护建议。

**🔧 技术方法**

主要技术包括eBPF/XDP、Linux内核钩子与helper函数、Docker/Kubernetes容器化、Open5GS与Edgecom eUPF的快速数据路径实现。

**📊 数据集**

实验使用Open5GS和Ueransim生成的仿真网络流量及内部日志，无外部公开数据集。

**📈 对比分析**

通过与未使用eBPF的传统5G UPF进行对比，实验表明eBPF能在毫秒级完成攻击，同时也显著提升数据包处理性能，体现了性能提升与安全风险并存。

**⚠️ 局限性**

局限性包括仅在Open5GS+eUPF单一实现上验证，缺乏对Free5GC、OAI等其他栈的覆盖；实验环境为单机虚拟化，未涵盖真实多租户云场景；未对完整防护方案的效果进行量化评估。

---

## 456. HATL: Hierarchical Adaptive-Transfer Learning Framework for Sign Language Machine Translation

**arXiv ID:** 2603.19260 | [PDF](https://arxiv.org/pdf/2603.19260v1)

**作者:** Nada Shahin `[一作]` (United Arab Emirates University), Leila Ismail `[通讯]` (United Arab Emirates University)

**通讯引用:** 2872 | [OpenAlex ID](https://openalex.org/A5000395681)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种层级自适应迁移学习框架（HATL），通过动态逐层解冻预训练模型并结合多损失监督，实现了对手语机翻模型的稳健适配。

**💡 创新点**

创新点在于：①基于验证性能的自适应解冻策略，避免一次性全参数微调导致的过拟合；②层级学习率衰减和稳定机制，使得低层保持通用特征而高层更易适应；③将CTC、CE、编码器、骨干层监督结合，提升多阶段学习的稳健性。

**🔧 技术方法**

使用的技术包括：预训练的ST‑GCN++骨干网络、Transformer和自适应Transformer（ADAT）翻译模型、层级学习率衰减、CTC与交叉熵损失、以及动态解冻与热身/冷却策略。

**📊 数据集**

在三个数据集上评估：德国的PHOENIX14T、阿拉伯的Isharah以及美国的MedASL，涵盖Sign2Text和Sign2Gloss2Text两类任务。

**📈 对比分析**

与经典微调（仅训练翻译层）和全微调（一次性解冻所有层）对比，HATL在BLEU‑4和ROUGE上分别提升12.2%–15.6%（ADAT）和3%–7%（Transformer），且在三大数据集均达到或超过现有最佳结果；训练时间略高于全微调，但低于常规微调+全微调的组合。

**⚠️ 局限性**

局限性包括：①动态解冻过程引入额外的超参数调优（阈值、窗口大小等）；②对计算资源有一定需求，尤其在ADAT上需要更多训练周期；③实验仅针对ST‑GCN++骨干，需验证对其他预训练模型的通用性。

---

## 457. Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving

**arXiv ID:** 2603.20076 | [PDF](https://arxiv.org/pdf/2603.20076v1)

**作者:** Pritom Gogoi `[一作]` (University of Stuttgart), Andreas Look `[通讯]` (Bosch Center for AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种低秩+对角协方差（LRPD）结构的概率在线地图生成框架，能够在实时感知中预测道路要素的几何不确定性，并将此不确定性信息嵌入轨迹预测模型，提升整体感知-预测-规划的鲁棒性。

**💡 创新点**

创新点包括：
1) 通过LRPD参数化完整协方差矩阵，实现点间空间相关性的高效建模；
2) 将结构化不确定性直接作为特征输入轨迹预测网络，并采用FiLM机制对置信度进行细粒度调制；
3) 两阶段训练策略与协方差正则化（warmup→结构学习）保证了稳定收敛。

**🔧 技术方法**

使用的技术包括：Transformer 基础的 MapTR/MapTRv2 以及 MapTRv2-CL 做为地图回归基干；负对数似然损失配合低秩+对角协方差参数化；HiVT 轨迹预测网络配合显式不确定性编码；FiLM 模块用于置信度调制；两阶段训练与协方差学习调度。

**📊 数据集**

实验基于公开的 nuScenes 数据集，涵盖约 1000 个城市驾驶场景，使用多传感器（相机、雷达、LiDAR）数据进行在线地图生成和轨迹预测。

**📈 对比分析**

与确定性基线、独立协方差（iid）以及现有扩展（BEV、Dual Decoder）等方法对比，LRPD 在 MapTRv2 上的 mAP 提升至 0.6345（高于 iid 0.6121），在 MapTRv2-CL 上的 minADE6、minFDE6、MR 分别提升至 0.3423、0.6648、0.0555，接近使用真实 HD 地图（GT Map）的 0.3357、0.6525、0.0536，创下了基于在线地图的运动预测 SOTA。

**⚠️ 局限性**

局限性：
1) 仅建模单一地图要素内部的相关性，未考虑不同要素之间的交互；
2) 低秩维度为固定超参数，可能在更复杂几何场景下不足以表达全部不确定性；
3) 依赖 Transformer‑based 生成器，对极端遮挡或稀疏 LiDAR 场景的鲁棒性仍需进一步验证；
4) 需要在不同城市、传感器组合下进一步评估泛化能力。

---

## 458. FedAgain: A Trust-Based and Robust Federated Learning Strategy for an Automated Kidney Stone Identification in Ureteroscopy

**arXiv ID:** 2603.19512 | [PDF](https://arxiv.org/pdf/2603.19512v1)

**作者:** Ivan Reyes-Amezcua `[一作]` (Centro de Investigación y de Estudios Avanzados del IPN), Gilberto Ochoa-Ruiz `[通讯]` (Escuela de Ingeniería y Ciencias, Tecnológico de Monterrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于信任权重的联邦学习框架 FedAgain，用于在多机构、数据异构且可能存在损坏的条件下实现肾结石识别的稳健训练。

**💡 创新点**

创新点在于引入双重信任机制：① 评估本地验证误差（衡量数据质量）；② 计算模型更新偏差（衡量一致性）。两者组合生成动态权重，实时抑制噪声或恶意客户端对全局模型的影响，并在不依赖事先估计 Byzantine 数量的情况下提升鲁棒性。

**🔧 技术方法**

使用联邦平均（FedAvg）、FedProx、FedMedian、Bulyan 等基准算法；在 FedAgain 中实现了基于误差和偏差的权重归一化与聚合；采用 PyTorch 自定义框架进行同步通信和模型更新。

**📊 数据集**

实验数据集包括三套肾结石图像数据集（A：Michel Daudon，B：Jonathan El-Beze，C：MyStone），以及公开基准 MNIST 与 CIFAR‑10，用于验证方法在医疗与通用场景的泛化能力。

**📈 对比分析**

在 IID、非 IID（标签偏移、Dirichlet）以及不同腐败率（10%–50%）下对比五种聚合策略。FedAgain 在所有设置中均表现出更高的准确率、召回率与 F1，且在腐败程度升高或客户端数增大时波动最小，证明其对数据异构和噪声具有更强的容错性；与传统 FedAvg 相比提升约 3–7%（肾结石）或更显著提升在 CIFAR‑10 上。

**⚠️ 局限性**

局限性：仅在肾结石分类任务验证，尚未在真实临床流水线中部署；未结合差分隐私或安全聚合，可能对敏感医疗数据的合规性有待提升；对极端高强度腐败或多模态数据的鲁棒性尚需进一步验证。

---

## 459. Implementing the L4S Architecture in the ns-3 Simulator

**arXiv ID:** 2603.20166 | [PDF](https://arxiv.org/pdf/2603.20166v1)

**作者:** Maria Eduarda Veras `[一作]` (Universidade Federal de Pernambuco), Judith Kelner `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 3012 | [OpenAlex ID](https://openalex.org/A5026724035)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了 L4S 架构在 ns-3 中的完整模型，集成了 TCP Prague 与 AccECN 并对 DualQ Coupled AQM 进行了实验性实现。

**💡 创新点**

创新点在于高保真度地将 Linux 内核实现迁移至 ns-3，细化了窗口管理、分段精度与节拍控制，并通过实验验证模型与真实环境一致。

**🔧 技术方法**

使用技术包括 C++ ns-3 拓扑模拟、Accurate ECN 标志扩展、TCP Prague 的指数加权移动平均与节拍算法、DualPI2 版 DualQ AQM。

**📊 数据集**

数据集为基于 dumbbell 拓扑的仿真数据（两种 100 Mbps/5 ms 与 10 Mbps/30 ms 受限链路）以及对应的真实测试床（VM+物理路由器）。

**📈 对比分析**

对比方法为 30 次独立复制、95% 置信区间评估；结果显示 ns-3 与 Linux 的 RTT、拥塞窗口、吞吐量基本一致，Jain 公平指数约为 0.99。

**⚠️ 局限性**

局限性在于 DualPI2 AQM 的调度实现与 Linux 核心版存在差异，导致队列延迟略高、标记率偏低，影响了部分吞吐与公平性。

---

## 460. DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training

**arXiv ID:** 2603.19338 | [PDF](https://arxiv.org/pdf/2603.19338v1)

**作者:** Maoyang Xiang `[一作]` (Singapore University of Technology and Design), Bo Wang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 27420 | [OpenAlex ID](https://openalex.org/A5100408160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出分布感知分段激活函数DAPA，以在Transformer模型推理与训练中实现硬件友好的激活近似。

**💡 创新点**

创新点在于利用预激活分布划分段，采用分布加权均方误差DWMSE作为损失，配合基于DWMSE的16位固定点量化，大幅降低DSP与延迟并保持甚至提升精度。

**🔧 技术方法**

技术包括非均匀分段线性逼近、分布加权误差度量、梯度兼容的双线性逼近、FPGA HLS四阶段流水线实现以及固定点量化方案。

**📊 数据集**

使用ImageNet‑1K、WikiText‑2、GLUE、ViT、DeiT、Swin、GPT‑2等多种数据集进行评估。

**📈 对比分析**

与传统MSE逼近、PEANO‑ViT、SwiftTron等方法对比，DAPA在ViT/DeiT/Swin/GPT‑2模型中保持与FP32基线相同或更高的准确率；DSP利用率比传统实现低16×（GELU）或48×（Softmax），延迟从580 ns降至20 ns，量化后误差≤0.23%。

**⚠️ 局限性**

局限性包括对分布估计的依赖（需一定样本量）、段数越多精度提升有限，且目前只针对单层分段，难以扩展到更复杂的激活或跨层一致性。

---

## 461. Generative Active Testing: Efficient LLM Evaluation via Proxy Task Adaptation

**arXiv ID:** 2603.19264 | [PDF](https://arxiv.org/pdf/2603.19264v1)

**作者:** Aashish Anantha Ramakrishnan `[一作]` (Pennsylvania State University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**通讯引用:** 9461 | [OpenAlex ID](https://openalex.org/A5100405086)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了Generative Active Testing框架，通过将多选问答任务转换为固定Statement Verification任务来实现低成本的模型评估

**💡 创新点**

提出RunnerUp策略将动态选项转换为二元验证并稳定不确定性估计，同时引入零射先验正则化的LLM采样函数

**🔧 技术方法**

利用LLM代理、Statement Adaptation模块、交叉熵与Jensen‑Shannon距离的正则化采样函数（UniformLM_CEAcq等）

**📊 数据集**

在AGNews、PubMedQA、MedQA和AI2_ARC四个数据集上进行评估

**📈 对比分析**

与随机采样、UCB、Entropy等基线比较，估计误差AUC平均下降约40-70%，在高置信度样本上性能显著优于传统方法

**⚠️ 局限性**

目前仅使用零射先验代理，未对代理进行微调，且只覆盖多选QA，无法直接适用于开放式或长文本生成任务

---

## 462. MFil-Mamba: Multi-Filter Scanning for Spatial Redundancy-Aware Visual State Space Models

**arXiv ID:** 2603.20074 | [PDF](https://arxiv.org/pdf/2603.20074v1)

**作者:** Puskal Khadka `[一作]` (University of South Dakota), KC Santosh `[通讯]` (University of South Dakota)

**通讯引用:** 6467 | [OpenAlex ID](https://openalex.org/A5087790566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MFil-Mamba，一种多滤波扫描的视觉状态空间模型，取代传统的方向性遍历来捕捉二维空间依赖；

**💡 创新点**

创新点在于：①引入多滤波扫描机制，用垂直、水平及可学习滤波器并行获取互补的空间特征；②加入轻量自适应加权融合模块；③在Mamba块中整合卷积前馈网络；

**🔧 技术方法**

采用状态空间模型（SSM）、多滤波卷积、SiLU激活、LayerNorm、ConvFFN、可学习权重自适应融合以及常见的训练技巧如AdamW、随机深度、混合数据增强；

**📊 数据集**

主要在ImageNet-1K进行分类，MS COCO 2017进行目标检测与实例分割，ADE20K进行语义分割；

**📈 对比分析**

与现有CNN、ViT、S4ND、VMamba、MambaVision等模型按参数量、FLOPs对齐后比较；在ImageNet上Tiny/Small/Base版本分别取得83.2%、83.9%、84.2% Top‑1，超越同规模Swin、ConvNeXt、Mamba等；在MS COCO检测/分割上Tiny/Small/Base分别达47.3/47.9/49.0 box AP和42.7/43.1/43.8 mask AP，领先同级别Swin、Mamba等；在ADE20K语义分割上取得48.5/48.9、49.3/49.9、50.6/51.3 mIoU，优于Swin、ConvNeXt、VMamba等；

**⚠️ 局限性**

局限在：①目前仅在常规视觉任务上验证，尚未针对医学、超分等专业场景展开；②多滤波模块虽然提升性能，但引入的参数与计算略高；③缺乏对极大图像尺寸或长宽比变形的鲁棒性分析。

---

## 463. A Dynamic Bayesian and Machine Learning Framework for Quantitative Evaluation and Prediction of Operator Situation Awareness in Nuclear Power Plants

**arXiv ID:** 2603.19298 | [PDF](https://arxiv.org/pdf/2603.19298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 464. From Tokens To Agents: A Researcher's Guide To Understanding Large Language Models

**arXiv ID:** 2603.19269 | [PDF](https://arxiv.org/pdf/2603.19269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 465. Evaluating Image Editing with LLMs: A Comprehensive Benchmark and Intermediate-Layer Probing Approach

**arXiv ID:** 2603.19775 | [PDF](https://arxiv.org/pdf/2603.19775v1)

**作者:** Shiqi Gao `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21777 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了 TIEdit 评测基准（512 张源图像、8 种编辑任务、10 款文本引导图像编辑模型、307,200 条专家评分，最终 15,360 条 MOS）并提出了 EditProbe——利用多模态 LLM 的中间层表示进行编辑质量预测的自动评测框架。

**💡 创新点**

创新点：① 大规模、三维度（感知质量、编辑一致性、内容保留）的人类 MOS 评测基准；② 采用中间层探测而非最终输出的 LLM 评测方式，显著提升与人类评判的相关性；③ 通过 AdaLoRA 在视觉编码器与语言模型上高效适配，保持大模型先验。

**🔧 技术方法**

技术方法：多模态大语言模型 InternVL3、AdaLoRA 参数高效微调、基于中间 Transformer 层的隐藏表示提取、轻量 MLP 回归头、以及与传统 IQA、VLM 评测指标和 GPT‑4 的对比实验。

**📊 数据集**

使用数据集：TIEdit 基准数据集（512 张 Unsplash 场景图、8 种编辑任务、10 款 TIE 模型生成的 5,120 张编辑图像）以及对应的 307,200 条专家评分。

**📈 对比分析**

对比方法：将 EditProbe 与传统全参考/无参考 IQA、CLIPScore、ImageReward、深度 IQA 网络以及 GPT‑4 进行 SRCC/PLCC/KRCC 对齐度评估。EditProbe 在四个评测维度上的 SRCC 均超过 0.78，较最佳基准提升约 30%–70%，表现出显著更高的人类一致性。

**⚠️ 局限性**

局限性：① 评测范围仅覆盖 8 种编辑任务和 10 款模型，尚未验证在更广泛任务/模型上的通用性；② 依赖单一 LLM 基座（InternVL3），在不同模型或更大规模数据上可能需要重新适配；③ 仍需人工评测支持，完全自动化评估仍不成熟。

---

## 466. Acyclic Graph Pattern Counting under Local Differential Privacy

**arXiv ID:** 2603.19671 | [PDF](https://arxiv.org/pdf/2603.19671v1)

**作者:** Yihua Hu `[一作]` (Nanyang Technological University), Wei Dong `[通讯]` (Nanyang Technological University)

**通讯引用:** 15519 | [OpenAlex ID](https://openalex.org/A5100746411)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种通用的基于边局部差分隐私的无环图模式计数方法；

**💡 创新点**

核心创新在于递归子图计数框架与随机标记技术，克服了无环模式构造与节点重复消除的两大挑战；

**🔧 技术方法**

采用多轮LDP聚合、Laplace机制与随机标记（色彩编码）相结合的技术实现；

**📊 数据集**

在多种真实网络数据集（未公开具体名称）上进行实验验证；

**📈 对比分析**

与传统RR枚举方案对比，实测在k-line walk、k-line path、k-edge无环模式和k-star计数任务中，均实现了46–2600倍的精度提升和300–650倍的通信成本下降；

**⚠️ 局限性**

局限性在于仅处理无环模式，无法直接扩展到循环模式；此外方法基于边LDP，对更强的节点LDP或复杂隐私模型尚未适用。

---

## 467. SeeClear: Reliable Transparent Object Depth Estimation via Generative Opacification

**arXiv ID:** 2603.19547 | [PDF](https://arxiv.org/pdf/2603.19547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 468. Sensing Your Vocals: Exploring the Activity of Vocal Cord Muscles for Pitch Assessment Using Electromyography and Ultrasonography

**arXiv ID:** 2603.19698 | [PDF](https://arxiv.org/pdf/2603.19698v1)

**作者:** Kanyu Chen `[一作]` (Keio University), Kai Kunze `[通讯]` (Keio University)

**通讯引用:** 4149 | [OpenAlex ID](https://openalex.org/A5056183585)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

通过收集并分析声带肌肉的电肌电图（EMG）与超声成像（UI），构建专家参考可视化，并在初学者和专业歌手中评估其对声乐训练的影响。

**💡 创新点**

首次将EMG和超声双模态实时可视化结合用于声乐教学，并将专家肌肉活动作为训练参考，系统比较三种反馈方式（EMG、UI、传统声学/教练）对不同水平歌手的效果。

**🔧 技术方法**

使用Delsys Trigno无线EMG传感器、CONTEC CMS600P2 B型超声探头、噪声抑制麦克风，并通过滤波、Hilbert包络、稳态度量、声带长度计算、SPR、PCA等信号处理与统计方法。

**📊 数据集**

收集了16名歌手（初学者、业余、专业）的Vocal Cord Sensing Dataset（VCSD），包含EMG（2000/4370Hz）和30fps超声视频，涵盖G2–E6共27个白键音高。

**📈 对比分析**

采用线性混合效模型、Wilcoxon检验、NASA‑TLX、SoAS等定量与定性评估；结果显示EMG能显著区分歌手水平并提升主观控制感，UI能改善声带长度控制但工作负荷更高；在专业歌手中，EMG与SPR高度相关，体现更佳肌肉协调。

**⚠️ 局限性**

样本量有限，超声探头体积不便携，短期训练后EMG稳定性下降可能受认知负荷影响；性别与年龄混杂导致结果解释受限；缺乏长期跟踪评估与自动化超声图像分析。

---

## 469. CO-EVOLVE: Bidirectional Co-Evolution of Graph Structure and Semantics for Heterophilous Learning

**arXiv ID:** 2603.19596 | [PDF](https://arxiv.org/pdf/2603.19596v1)

**作者:** Jinming Xing `[一作]` (North Carolina State University), Muhammad Shahzad `[通讯]` (North Carolina State University)

**通讯引用:** 5424 | [OpenAlex ID](https://openalex.org/A5049422969)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个双视角的协同进化框架 CO‑EVOLVE，用图神经网络与大型语言模型的相互反馈来学习异构图的结构与语义。

**💡 创新点**

创新点包括：① 通过 Soft Prompt 将图结构动态注入 LLM，①1 让 LLM 生成动态语义图；② 采用可自适应的节点门控机制重构图结构；③ 设计硬结构冲突感知对比损失，显式抑制语义-结构不一致；④ 通过不确定性门控一致性策略实现两模态的可靠互修。

**🔧 技术方法**

技术手段包括：Llama‑3.2‑1B LoRA 微调、GCN 消息传递、Gauss‑Seidel 交替优化、Soft Prompt 投影、可学习多头相似度、PPR 高阶结构判别、熵门控一致性与融合。

**📊 数据集**

使用公开图文本双属性数据集：Reddit、Instagram 与 WikiCS，分别包含用户社交、关注关系与维基参考网络。

**📈 对比分析**

与 GCN、GAT、Sent‑BERT、RoBERTa、TAPE、ZeroG、LLaGA、InstructGLM、FLAG、Patton 等10+基线比较，CO‑EVOLVE 在 Accuracy 上平均提升 9.07%、F1 分数提升 7.19%，在噪声与稀疏实验中仍保持显著优势。

**⚠️ 局限性**

局限性包括：① 依赖大模型与图模型的双重计算资源，训练成本高；② 对超参数（k、α、ε 等）敏感，需要手动调优；③ 对极大规模图的扩展仍有挑战，内存与推理效率待改进。

---

## 470. Exploring Novelty Differences between Industry and Academia: A Knowledge Entity-centric Perspective

**arXiv ID:** 2603.19319 | [PDF](https://arxiv.org/pdf/2603.19319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 471. Generalizable NGP-SR: Generalizable Neural Radiance Fields Super-Resolution via Neural Graph Primitives

**arXiv ID:** 2603.20128 | [PDF](https://arxiv.org/pdf/2603.20128v1)

**作者:** Wanqi Yuan `[一作]` (Clemson University), Nianyi Li `[通讯]` (Clemson University)

**通讯引用:** 1427 | [OpenAlex ID](https://openalex.org/A5042198120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可泛化的3D感知超分辨率框架NGP-SR，能够直接从低分辨率相机姿态图像重建高分辨率辐射场，实现视角一致的高质量新视角合成。

**💡 创新点**

创新点在于将层级视图纹理编码与相机感知特征融合、全局特征双阶段融合引入Neural Graphics Primitives（NGP）中，使模型在3D域学习高频细节，既不依赖高分辨率参考也不需要后期2D超分，保持跨视角一致性。

**🔧 技术方法**

采用hash编码的NGP、纹理token化、相机嵌入MLP、统一哈希表与相机条件权重预测、两阶段融合网络、2层MLP解码等技术；训练采用MSE损失，硬件使用NVIDIA A100 GPU。

**📊 数据集**

在Blender、LLFF和DTU等公共数据集上进行评估，使用低分辨率图像（100×100、200×200）和×2、×4放大比例进行实验。

**📈 对比分析**

与TensoRF、Instant‑NGP、3DGS等无SR基线以及NeRF‑SR、FastSR‑NeRF、SuperGaussian、3DSR等SR方法对比；在Blender和LLFF上PSNR、SSIM、LPIPS均显著优于现有方法，尤其在极低分辨率下表现突出；在DTU上达到约31dB PSNR，细节和一致性大幅提升。

**⚠️ 局限性**

局限性包括：对极高频细节恢复仍有限；仅适用于静态场景；需要足够多视角的低分辨率输入；缺乏轻量化或实时部署方案。

---

## 472. DIAL-KG: Schema-Free Incremental Knowledge Graph Construction via Dynamic Schema Induction and Evolution-Intent Assessment

**arXiv ID:** 2603.20059 | [PDF](https://arxiv.org/pdf/2603.20059v1)

**作者:** Weidong Bao `[一作]` (Northeastern University), Ge Yu `[通讯]` (Northeastern University)

**通讯引用:** 6431 | [OpenAlex ID](https://openalex.org/A5072406974)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了闭环增量知识图构建框架 DIAL‑KG，利用元知识库（MKB）实现动态模式诱导、治理裁决和事件/三元组双轨抽取，实现无预定义 schema 的自我演化知识图构建。

**💡 创新点**

创新点在于：①闭环增量流程和事务式更新；②双轨抽取策略（静态三元组 vs 动态事件）兼顾稀疏性和时间信息；③演化意图评估机制区分信息性事件与状态变更事件；④基于 MKB 的自我演化模式诱导，支持关系与事件 schema 的在线生成与更新。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen‑Max 生成、DeepSeek‑V3 判定）、检索增强生成、实体/事件聚类与归一化、事件关系化、Meta‑Knowledge Base 作为治理与上下文存储、BGE‑M3 语义相似度计算。

**📊 数据集**

使用数据集：WebNLG、Wiki‑NRE（公开静态基准）以及基于 Kubernetes 发行日志构造的 SoftRel‑Δ（模拟演化信息的窗口化流数据）。

**📈 对比分析**

与传统无 schema LLM 抽取基线 EDC、AutoKG 进行对比。静态任务中 DIAL‑KG 的 F1 提升约 3–5%，增量任务中新增事实的 Δ‑Precision ≥ 0.97，软废弃的 D‑HP ≥ 0.98，且诱导的 schema 在关系类型数量上减少至 15%，冗余度下降 1.6–2.8 分。

**⚠️ 局限性**

主要限制：依赖大语言模型导致推理延迟和计算成本高，适用于低速/中速流处理；在极高频数据流中需要进一步压缩或蒸馏 MKB 知识以降低时延。

---

## 473. FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment

**arXiv ID:** 2603.19539 | [PDF](https://arxiv.org/pdf/2603.19539v1)

**作者:** Betty Xiong `[一作]` (Stanford University), Russ B Altman `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了一个基于FDA药品标签的17K+条专业级问答基准，涵盖事实、跨段多跳和拒答三类问题，并提供闭书、开书与检索增强等多种测试场景。

**💡 创新点**

创新点在于：1）邀请FDA监管专家参与问题生成与评判，实现监管级别的正确性与可溯源性；2）设计四阶段流水线（上下文挑选、LLM生成、专家反馈、LLM评判过滤）自动化构造高质量问答；3）引入“拒答”题型以检测模型的安全与抗幻觉能力；4）提供统一的评价指标，兼顾答案正确、引用一致和拒答行为。

**🔧 技术方法**

采用多模态技术：LLM（ChatGPT/Claude等）用于问题生成与评判；BM25、DenseRetriever与ReContriever用于检索；LLM-as-judge自动评估答案和引用；实验中对多种SOTA LLM进行评测。

**📊 数据集**

数据集来自700份FDA处方药标签（SPL格式），经过段落切分后构成上下文，最终生成17K+ QA样本。

**📈 对比分析**

与基线相比，闭书场景准确率仅约22%（事实）/34%（多跳）；oracle-passages显著提升至约78%/65%；开书但不检索时准确率仅约53%/40%；检索模块BM25在召回上远优于稠密检索。模型在引用覆盖率和拒答F1上均表现中等，说明存在较大提升空间。

**⚠️ 局限性**

局限性包括：1）切片化导致上下文过窄，无法覆盖跨段信息；2）标签冗余导致引用多重性，评估难度上升；3）多跳题型样本不足，难以覆盖真实监管推理；4）LLM评判与引用度量可能误差；5）拒答设计过于模板化，缺少真实不可答题场景。

---

## 474. EvoTaxo: Building and Evolving Taxonomy from Social Media Streams

**arXiv ID:** 2603.19711 | [PDF](https://arxiv.org/pdf/2603.19711v1)

**作者:** Yiyang Li `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5221 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并持续演化社交媒体流中的层次分类体系，利用时间顺序的文本信息生成可演化的结构化知识树。

**💡 创新点**

创新点在于将短文本映射为LLM生成的结构化操作，采用双视图（语义+时间）聚类与概念记忆库相结合的审查仲裁机制，实现对噪声短文本的鲁棒性、可扩展性以及对话语演化的动态捕捉。

**🔧 技术方法**

使用大型语言模型（LLM）生成动作与审查，HDBSCAN聚类（语义距离与时间距离的加权组合），概念记忆库（定义、包含/排除线索），以及LLM驱动的两步审查仲裁流程。

**📊 数据集**

两个Reddit社区数据集：/r/Opiates（2015‑2024、2022‑2024）和/r/ICE_Raids（2025‑2026），分别涵盖高噪声经验讨论和事件驱动的时间动态语境。

**📈 对比分析**

与KN、Chain‑of‑Layer、TnT‑LLM、TaxoAdapt等基线对比，EvoTaxo在叶子分配熵低、未分类率低、结构质量（NLIV‑S、路径细化、兄弟节点一致性）高、覆盖率强、token使用更少，整体性能均优于基线。

**⚠️ 局限性**

局限性包括对LLM生成质量的依赖、窗口大小的手动设定、评估指标主要基于NLI/LLM而非全人工标注、仅在Reddit平台验证，尚需在其他语言和平台上进一步评估。

---

## 475. CeRLP: A Cross-embodiment Robot Local Planning Framework for Visual Navigation

**arXiv ID:** 2603.19602 | [PDF](https://arxiv.org/pdf/2603.19602v1)

**作者:** Haoyu Xi `[一作]` (Shanghai Jiao Tong University), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 3210 | [OpenAlex ID](https://openalex.org/A5100695302)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套名为 CeRLP 的跨本体视觉导航本地规划框架，实现了在不同机器人尺寸、相机参数和相机类型下的零样本迁移。

**💡 创新点**

创新点包括：① 通过离线 ArUco 标定实现单目深度的比例校正，解决相机尺度不确定性；② 将校正后的深度图投影为高度自适应的虚拟激光扫描，消除相机姿态差异对感知的影响；③ 在决策网络中显式嵌入机器人几何尺寸，使策略对不同平台自动适配；④ 采用强化学习的 DRL‑DCLP 结合课程学习，在模拟中快速泛化。

**🔧 技术方法**

使用技术包括：单目深度估计（Depth Anything V2），ArUco 标定、最小二乘线性回归、Ridge 回归；3D 逆投影与相机外参变换；高度自适应点云过滤与投影得到虚拟激光扫描；强化学习（SAC）+ PointNet 编码器 + 课程学习；CLIP 进行高层视语指令解码。

**📊 数据集**

实验数据集：BARN 自动机器人导航基准（模拟）；真实世界采用七套差分驱动机器人和一台四足机器人，配备多相机（WHEELTEC C100、Orbbec Femto Bolt）进行真实场景测试。

**📈 对比分析**

与 DWA、Fast‑DWA、E‑Band、E2E、DRL‑DCLP 等方法比较，CeRLP 在 BARN 仿真中 SR 达到 78%（相较 E‑Band 提升 13%），Metric 指标 0.3790，时间效率高、碰撞率 19%，时间超时率仅 3%；在真实环境下，跨本体、跨相机、跨尺寸的测试均保持高成功率并且避免碰撞，整体性能接近使用 LiDAR 的 DRL‑DCLP。

**⚠️ 局限性**

局限性包括：① 仍依赖单目深度估计的精度，深度噪声会导致虚拟激光扫描误差；② 前向视野受限（≈75°），相比全向 LiDAR 可能出现盲区；③ 需要预先进行相机标定，若标定数据不足会影响尺度恢复；④ 计算量相对较大，实时性受限于 GPU 性能。

---

## 476. MME-CoF-Pro: Evaluating Reasoning Coherence in Video Generative Models with Text and Visual Hints

**arXiv ID:** 2603.20194 | [PDF](https://arxiv.org/pdf/2603.20194v1)

**作者:** Yu Qi `[一作]` (Northeastern University), Lawson L. S. Wong `[通讯]` (Northeastern University)

**通讯引用:** 952 | [OpenAlex ID](https://openalex.org/A5032791503)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了MME-CoF-Pro视频推理一致性基准，评估视频生成模型在因果推理连贯性上的表现，并提出过程级评估指标Reasoning Score；

**💡 创新点**

首次将推理提示（无提示、文本提示、视觉提示）作为可控变量；提出过程级Reasoning Score来细粒度评估中间推理步骤；通过对7种模型的细粒度比较揭示推理与生成质量解耦、提示对一致性与幻觉影响等现象；

**🔧 技术方法**

使用多种大型视频生成模型（Veo、Sora、Seedance、Kling、Cosmos等）生成视频；采用Gemini-2.5-Flash等VLM作为自动评判者计算Reasoning Score；手工标注推理步骤并构造三种提示设置；统计一致性、时间连贯性、视觉稳定性等指标；

**📊 数据集**

从27个现有视频推理基准中挑选并构建了303个样本，覆盖16个推理类别（视觉细节、结构、物理、科学、任务导向等），其中8个类别额外加入视觉提示；

**📈 对比分析**

对7个闭源与开源模型在no-hint、text-hint、visual-hint三种设置下进行评估；结果显示推理一致性普遍较弱（平均≈55%），模型生成质量与推理得分无明显相关；文本提示能提升RS但降低一致性并引发幻觉；视觉提示对结构性任务有益但对细粒度任务不佳；整体推理性能仍低；

**⚠️ 局限性**

推理能力有限且与生成质量解耦；提示引导易导致模型表面模仿而非真实理解；视觉提示可能被误生成为场景元素；模型难以稳定利用多重提示；缺乏更强的视觉 grounding 与抗幻觉机制；

---

## 477. From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering

**arXiv ID:** 2603.20193 | [PDF](https://arxiv.org/pdf/2603.20193v1)

**作者:** Xinyi Shang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6416 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于像素级差异图阈值化的细粒度图像篡改检测框架，并构建大规模高保真基准。

**💡 创新点**

从掩模标注转向精确像素对齐的标签，结合语义与文本监督实现多任务检测与解释；提供可调阈值的标签生成方法。

**🔧 技术方法**

使用阈值化差异图、像素BCE+Dice损失、语义多标签交叉熵、文本生成语言模型以及多头Transformer检测器。

**📊 数据集**

自研的420K图像对基准，覆盖8种编辑类型，提供原始、篡改图像、差异图、像素标签、语义标签及文本描述；对比现有如SIDBench、DiFF等。

**📈 对比分析**

在细粒度定位和语义分类指标上相较SIDA、LISA、FakeShield提升显著（IoU从≈9%提升至≈18%，召回率和F1大幅上升），并在不同生成器上保持稳健性。

**⚠️ 局限性**

阈值 τ 的选择仍需经验调优；对极其细微或全局性篡改的检测仍有限；在未见生成模型（如GPT‑Image‑1.5）上性能下降。

---

## 478. Deterministic Mode Proposals: An Efficient Alternative to Generative Sampling for Ambiguous Segmentation

**arXiv ID:** 2603.20191 | [PDF](https://arxiv.org/pdf/2603.20191v1)

**作者:** Sebastian Gerard `[一作]` (KTH Royal Institute of Technology), Josephine Sullivan `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 10363 | [OpenAlex ID](https://openalex.org/A5088465676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种确定性模式提议模型（Mode Proposal Model），可在单前向传播中生成固定数量的分割掩码，直接覆盖模糊分割任务的所有可行模式；

**💡 创新点**

创新点在于：①通过多头卷积网络与匈牙利匹配训练实现多模态输出；②采用 DETR 机制的选择分数过滤冗余提议；③在仅有单一标签的弱监督情形下引入 PU 损失实现正负样本区分；④利用预训练流模型的速度场进行线性最小二乘估计模式概率，无需大规模采样；

**🔧 技术方法**

核心技术包括多头分割网络、匈牙利匹配损失、DET-RE 风格选择头、PU 损失、流匹配/速度场分解、线性最小二乘估计；

**📊 数据集**

实验使用 MMFire（野火传播）、Cityscapes（城市道路）与 LIDC（肺部 CT）三组多模态数据集；

**📈 对比分析**

与 ProbU-Net、Diffusion+SPELL/PG 等生成式方法对比，Mode Proposal Net 在 HM IoU* 上提升 20%+，运行时仅几分钟即可完成 16-32 个提议，显著优于需数小时采样的扩散模型；

**⚠️ 局限性**

局限性：速度场分解对模式差异度要求高，在 LIDC 这类同质模式严重时效果差；匈牙利匹配和 PU 损失在单标签情形下对近似重复提议的剔除不够；数据集有限，需在更丰富的现实场景中验证。

---

