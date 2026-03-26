# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-26 | 今日论文总数: 495

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Human, AI, and Hybrid Ensembles for Detection of Adaptive, RL-based Social Bots

**arXiv ID:** 2603.23796 | [PDF](https://arxiv.org/pdf/2603.23796v1)

**作者:** Valerio La Gatta `[一作]` (Northwestern University), V. S. Subrahmanian `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在模拟社交媒体平台DartPost上进行为期五天的受控实验，系统评估了人类、传统机器学习、LLM和人机混合模型在检测由强化学习驱动的自适应社交机器人（RL‑CSIO）中的性能，并验证了多种假设。

**💡 创新点**

创新点包括：①首次在活跃的隐蔽社会影响操作（CSIO）环境中研究人类对RL机器人检测的表现；②构建并检验13个关于人口、学习、网络及集体智能等维度的假设；③提出基于报告者质量加权的聚合策略及多种人机混合投票（Meta Voting、Hybrid Late Fusion）方案，并证明其优于单一人类或AI检测；④利用人类报告进行增量监督训练，展示其可替代真值监督的效果。

**🔧 技术方法**

技术手段：RL‑CSIO agent（actor‑critic + GCN），DartPost实验平台；传统机器学习检测器（RFS、BotBuster），LLM检测器（gpt‑3.5‑turbo、gpt‑4‑turbo等）；人类报告采集与质量加权聚合；Meta Voting、Hybrid Late Fusion、Hard/Soft Voting、Late Fusion；增量训练自我监督、真值监督与人类监督三种策略。

**📊 数据集**

数据集：实验共305个账户（225人工参与者 + 80 RL机器人）以及四个主题的CSIO；用于对比的公开基准模型训练集为Twibot‑20、Cresci‑2017、Caverlee‑2011；实验记录包含每日交互、报告、网络结构与行为特征。

**📈 对比分析**

比较方法：对13个假设使用置换检验、线性回归、Chi‑square、McNemar检验等；人类集体F1为0.582，AI单一模型最高F1为0.723（RFS在Twibot‑20上）；人机混合Meta Voting F1 0.801、Hybrid Late Fusion F1 0.761，均显著优于单一人类或AI；增量训练中，人类监督实现与真值监督相当甚至更优的F1提升。

**⚠️ 局限性**

局限性：受试者仅来自美国MTurk，结果对其他地区或更广泛人群可能不具可推广性；实验仅持续五天，可能无法捕捉长期学习与适应；仅评估RL驱动的自适应机器人，未涵盖其他机器人类型或更复杂的协调策略；实验平台DartPost与真实主流社交平台存在差异，进一步验证需在真实环境中进行。

---

## 2. Manifold Generalization Provably Proceeds Memorization in Diffusion Models

**arXiv ID:** 2603.23792 | [PDF](https://arxiv.org/pdf/2603.23792v1)

**作者:** Zebang Shen `[一作]` (ETH Zurich), Niao He `[通讯]` (ETH Zurich)

**通讯引用:** 1348 | [OpenAlex ID](https://openalex.org/A5071683073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究扩散模型在学习粗糙得分时仍能产生新颖、非记忆化样本的原因，提出基于流形假设的覆盖度定义，证明粗糙得分可实现近似最优的流形恢复并给出更细的覆盖尺度；

**💡 创新点**

创新点在于：①用“覆盖度”替代传统密度估计指标，将通用性视为对流形的几何覆盖；②证明粗糙得分（只精确到t⁻¹项）即可实现流形投影的近似最优，进而获得比经验分布更细的覆盖率；③提供连续时间逆向SDE+概率流ODE混合采样的理论分析，将几何恢复与样本生成联系起来；

**🔧 技术方法**

技术手段包括：流形几何理论（距离势函数、Eikonal方程、切空间投影）、分数匹配（局部DSM）、概率流ODE解析、Hellinger距离、Hausdorff距离与覆盖度证明、最小化风险与最优率比较；

**📊 数据集**

文中未使用具体公开数据集，研究主要在理论与仿真层面；

**📈 对比分析**

实验与比较主要是理论证明和数值模拟，结果表明在流形光滑度高时，粗糙得分下的扩散模型覆盖尺度可达到O(N^{-β/(4k)})，远优于经验样本的O(N^{-1/k})；

**⚠️ 局限性**

局限性包括：①需要假设流形光滑且已知维度、到达性等；②使用了理想化的连续时间采样与完全优化的DSM；③缺乏对具体参数化网络（如PINN）的实现与泛化性能评估；④常数与对数因子未给出精确数值，未验证在实际大规模数据上的效果。

---

## 3. LongTail Driving Scenarios with Reasoning Traces: The KITScenes LongTail Dataset

**arXiv ID:** 2603.23607 | [PDF](https://arxiv.org/pdf/2603.23607v1)

**作者:** Royden Wagner `[一作]` (Karlsruhe Institute of Technology), Christoph Stiller `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 23257 | [OpenAlex ID](https://openalex.org/A5091574711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入KITScenes LongTail数据集，结合多视角视频、高层指令及多语言专家推理轨迹，支持端到端驾驶决策的研究。

**💡 创新点**

提供长尾驾驶场景与专家多语言推理轨迹，提出语义一致性评估与多操纵分数（MMS）指标，突破传统单一轨迹评测的局限。

**🔧 技术方法**

采用多视角视频拼接、Rocchio分类与EmbeddingGemma进行语义一致性评估，结合链式思考（CoT）提示和动力学自行车模型生成轨迹，使用VLMs与VLA进行推理。

**📊 数据集**

新建KITScenes LongTail数据集（1000个长尾场景），与nuScenes、Waymo E2E、CoVLA等公开数据集进行对比。

**📈 对比分析**

通过MMS与L2误差对比，并在零样本、少样本及CoT提示下评估多款VLM与端到端驾驶模型；结果显示少样本/CoT显著提升，Gemini 3 Pro在MMS上表现最佳，但语义一致性仍偏低。

**⚠️ 局限性**

预训练域差异导致推理轨迹与轨迹不一致；链式思考在本数据集上表现不如基于动力学模型；缺乏针对长尾场景的微调与多语言推理一致性分析。

---

## 4. Upper Entropy for 2-Monotone Lower Probabilities

**arXiv ID:** 2603.23558 | [PDF](https://arxiv.org/pdf/2603.23558v1)

**作者:** Tuan-Anh Vu `[一作]` (Université de Technologie de Compiègne), Frédéric Pichon `[通讯]` (Université d'Artois)

**通讯引用:** 251 | [OpenAlex ID](https://openalex.org/A5101870496)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了 2-单调下概率诱导的可信集上熵的计算，证明其存在强多项式算法，并给出针对信念函数、可能性分布和概率区间的改进算法。

**💡 创新点**

创新点包括把上熵计算转化为超模优化并利用 SFM 实现 O(n²)（甚至 O(n)）的复杂度；针对特定情形构造最大流/最小割、凸包等高效解法；提出 Frank–Wolfe 近似方法以降低高阶多项式开销。

**🔧 技术方法**

使用超模/子模优化、分解算法、最大流/最小割、凸包求解、Frank–Wolfe 凸优化、二分与牛顿法求解一维方程。

**📊 数据集**

实验使用随机生成的 2-单调下概率（基于凸函数和 Dirichlet 分布）、随机概率区间以及可调 margin 的区间；并在规模从几千到 10⁸ 的 Ω 上测试。

**📈 对比分析**

与现有 O(n²) 或指数复杂度算法（如 Abellán–Moral）相比，新算法在大规模问题上显著加速（例如从数十秒降至几毫秒或秒级），并在近似方法上达到相同精度。

**⚠️ 局限性**

局限性在于精确算法仍需多项式级别的 SFM，初始点生成在大 Ω 时是瓶颈；近似方法虽然高效但不保证全精度；实验仅在合成数据上，缺乏真实可信集案例。

---

## 5. Labeled Compression Schemes for Concept Classes of Finite Functions

**arXiv ID:** 2603.23561 | [PDF](https://arxiv.org/pdf/2603.23561v1)

**作者:** Benchong Li `[一作]` (Xidian University), Benchong Li `[通讯]` (Xidian University)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5061680859)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文证明了任意有限概念类（VC维度为d）都存在大小为d的标签化压缩方案，解答了长期未解的压缩猜想。

**💡 创新点**

创新点在于构造了基于“频率为1的片段”迭代压缩的算法，并给出了完整的压缩与重构过程，首次实现了大小恰好等于VC维度的标签化压缩。

**🔧 技术方法**

使用了组合学技巧（如Sauer‑Shelah‑Perles引理、双重归纳法）、频率计数与片段分配策略，以及最大域重构的理论分析。

**📊 数据集**

论文为理论性工作，没有使用具体数据集，研究对象为抽象的有限概念类。

**📈 对比分析**

由于是理论证明，未进行实验对比；作者仅说明压缩方案是“proper”，且压缩子样本大小恒为d，可直接复现。

**⚠️ 局限性**

局限性：方案仅适用于有限概念类；未讨论压缩过程的计算复杂度；未涵盖无标签压缩或无限概念类的情况。

---

## 6. Energy Efficient Software Hardware CoDesign for Machine Learning: From TinyML to Large Language Models

**arXiv ID:** 2603.23668 | [PDF](https://arxiv.org/pdf/2603.23668v1)

**作者:** Mohammad Saleh Vahdatpour `[一作]` (Georgia State University), Yanqing Zhang `[通讯]` (Georgia State University)

**通讯引用:** 5835 | [OpenAlex ID](https://openalex.org/A5100612105)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对从毫瓦级 TinyML 设备到千兆瓦级 LLM 服务器的能效软件-硬件协同设计技术进行了系统综述与跨规模对比分析。

**💡 创新点**

提出跨尺度层次化分解视角，将模型拆分为稳定特征提取层与自适应推理层，从而实现统一的能效优化框架。

**🔧 技术方法**

综述了 FPGA 零缓冲数据流、计算-内存融合（CIM）、稀疏压缩、量化、动态频率/电压、分布式推理、自动化工具（如 DNNBuilder、TVM Unity）等关键技术。

**📊 数据集**

采用公开论文中的实验结果和标准数据集（如 ImageNet、COCO、GLUE、LLM 推理任务等）进行性能汇总与对比。

**📈 对比分析**

通过跨平台基准（FPGA、ASIC、GPU、CIM、数据中心）和能耗指标比较，显示 FPGA 零缓冲可提升 59× 速度并降低 87% 数据移动，CIM 在 LLM 推理上比 GPU 高 16–36× 能效，系统级优化可减少 30–73% 能耗。

**⚠️ 局限性**

存在跨尺度可迁移性差、设计空间庞大且手工调优成本高、基准缺乏真实环境与生命周期评估、自动化工具受限、以及安全性与鲁棒性未充分考虑等局限。

---

## 7. Steering Code LLMs with Activation Directions for Language and Library Control

**arXiv ID:** 2603.23629 | [PDF](https://arxiv.org/pdf/2603.23629v1)

**作者:** Md Mahbubur Rahman `[一作]` (Iowa State University), Harshitha Menon `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 905 | [OpenAlex ID](https://openalex.org/A5038754571)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过在推理时向 Transformer 的残差流添加学习到的线性方向来控制代码生成模型在语言和库选择上的偏好；

**💡 创新点**

提出了一种基于差分均值的层级方向估计与激活层级干预方法，展示了激活空间中可直接操控的“代码风格”方向；

**🔧 技术方法**

使用了差分均值估计、激活层级干预、层选择策略和 LLM 判别器来评估和调节激活向量；

**📊 数据集**

采用了合成的 ChatGPT 生成提示（含目标、相反、中性三类）以及 MultiPL‑E 数据集的 Python–C++ 示例，实验覆盖五对语言/库（PyTorch–TensorFlow、Python–C++、STL–Boost、Matplotlib–Seaborn、NumPy–CuPy）；

**📈 对比分析**

在 CodeGemma‑7B、Qwen2.5‑Coder‑7B、Llama3.1‑8B 三大开源代码 LLM 上进行验证，结果表明对常见生态系统（Python、NumPy、STL 等）的偏好可被近乎完全控制，且在冲突提示下仍能部分覆盖；但对稀有生态系统效果较弱，且过强干预会降低代码质量；

**⚠️ 局限性**

局限性包括仅评估三种模型与五对生态系统、提示主要为合成、方法仅在单层线性方向，可能忽略更分布式或非线性表示，以及使用 LLM 判别器对输出进行标注，未完全保证代码正确性或质量。

---

## 8. The HyperFrog Cryptosystem: High-Genus Voxel Topology as a Trapdoor for Post-Quantum KEMs

**arXiv ID:** 2603.23505 | [PDF](https://arxiv.org/pdf/2603.23505v1)

**作者:** Victor Duarte Melo `[一作]` `[通讯]` (Independent Researcher), Victor Duarte Melo (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了HyperFrog实验性后量子密钥封装机制，使用满足高循环秩等数字拓扑约束的二进制体素形状生成密钥，并在LWE核心上完成加密与解密。

**💡 创新点**

创新点在于将密钥从传统随机向量替换为符合高genus拓扑约束的体素形状，形成隐藏的组合陷阱对象；而非仅依赖结构化格子。

**🔧 技术方法**

采用LWE式公钥、Fujisaki–Okamoto变换、离散二项噪声、体素图映射以及拒绝采样等技术。

**📊 数据集**

实验使用从加密随机数生成器产生的随机3D 16×16×16体素样本，未使用外部公开数据集。

**📈 对比分析**

实验结果显示，在优化的C++实现中，密钥生成、封装与解封装时间与现有LWE/KEM方案相近，误码率可忽略不计，整体性能可与Frodo等方案媲美。

**⚠️ 局限性**

该方案仍为研究级实验，尚未接受广泛公共分析；安全性仅基于LWE假设，拓扑约束的安全性缺乏理论证明，实用性受限。

---

## 9. Object Search in Partially-Known Environments via LLM-informed Model-based Planning and Prompt Selection

**arXiv ID:** 2603.23800 | [PDF](https://arxiv.org/pdf/2603.23800v1)

**作者:** Abhishek Paudel `[一作]` (George Mason University), Gregory J. Stein `[通讯]` (George Mason University)

**通讯引用:** 6827 | [OpenAlex ID](https://openalex.org/A5042000667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了基于LLM推断的不确定性统计的模型化规划框架，结合离线重放实现部署时的提示词和LLM快速选择，用于部分已知环境下的目标物体搜索。

**💡 创新点**

将LLM知识融入模型化规划而非直接替代规划，并通过高层动作抽象实现离线重放用于提示词/LLM的即时选择，显著提升搜索效率。

**🔧 技术方法**

使用大语言模型（GPT‑5 Mini、Gemini 2.5 Flash）、A*路径规划、贝尔曼式期望代价计算、离线重放的多臂老虎机选择算法（UCB对比Replay Selection）等。

**📊 数据集**

在ProcTHOR生成式家居环境数据集进行150个场景的仿真实验，并在LoCoBot实验室公寓中进行5个实际实验。

**📈 对比分析**

相较于完全LLM或乐观基线，LLM‑informated规划平均导航成本降低4–11.8%（GPT‑5）或24.9–39.2%（Gemini）；Replay Selection在100次部署中比UCB平均成本低6.5%，累计遗憾降低33.8%。

**⚠️ 局限性**

局限包括提示词手工设计、仅测试部分已知环境、未处理完全未知环境下的探索、以及离线重放对假设的偏差敏感。

---

## 10. M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production

**arXiv ID:** 2603.23617 | [PDF](https://arxiv.org/pdf/2603.23617v1)

**作者:** Alexandre Symeonidis-Herzig `[一作]` (University of Surrey), Richard Bowden `[通讯]` (University of Surrey)

**通讯引用:** 13816 | [OpenAlex ID](https://openalex.org/A5044490167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 SMPL‑FX 结合 FLAME 表情空间并配合无代码簿的有限标量量化（FSQ），实现完整的 3D 手语生成，包括手部和非手部特征。

**💡 创新点**

创新点在于：1）将 FLAME 的 100 维表情空间嵌入 SMPL‑X，解决表情维度不足；2）使用 FSQ 代替 VQ‑VAE，消除面部表情量化崩溃，达 99% 代码簿利用率；3）在同一模型中同时进行手语生产与翻译的辅助目标，提升语义对齐。

**🔧 技术方法**

使用技术包括：3DMM 参数提取、SMPL‑FX 模型、FSQ‑VAE 进行多模态离散化、基于 mBART‑large 的自回归 transformer（SMLM），以及辅助翻译任务。

**📊 数据集**

使用的数据集有 How2Sign、CSL‑Daily、Phoenix14T、NMFs‑CSL 以及 MEAD 的 FLAME 参数作为面部训练数据。

**📈 对比分析**

与 SOKE 等基线对比，三大手语基准上取得 SOTA，JPE 降低约 3.8%/11.8% 的 BLEU‑4，NMFs‑CSL 混淆签名准确率提升 9.3%，显示出显著的几何和语义改进。

**⚠️ 局限性**

主要局限在于：对遮挡、低质量视频和快速手指动作的重建不够鲁棒，4× 时序压缩导致细节缺失；此外仍需大规模预训练或更高级的面部建模来进一步提升表现。

---

## 11. Smooth Routing in Decaying Trees

**arXiv ID:** 2603.23504 | [PDF](https://arxiv.org/pdf/2603.23504v1)

**作者:** Till Fluschnik `[一作]` (Humboldt Universitat), Malte Renken `[通讯]` (Technische Universitaet)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在退化图（连接随时间消失的网络）中，给定固定的逃生路径集合，求平滑可行的时间调度（即满足顶点容量、连接先后顺序、无相撞等约束）的存在性和最小延迟。

**💡 创新点**

首先系统性地证明该问题在退化路径、星形及一般树上的复杂性，揭示多种 NP‑hard 结果；随后提出整数线性规划（ILP）模型用于求解最小延迟，并在实验中展示该模型的可行性与有效性。

**🔧 技术方法**

运用了复杂性分析、动态规划（针对退化路径）、整数线性规划与 Gurobi 求解器、松弛模型（indicator 约束版）、图论构造与归约（从间隔图、独立集、立方独立集等问题）等技术。

**📊 数据集**

人工生成的退化路径与星形网络实例（不同顶点数、容量、期限比例）以及基于德国十座城市街网与河流的半人工实例（使用 OSM 数据并模拟洪水到达时间）。

**📈 对比分析**

在实验中比较了两种 ILP 实现（指示约束版与手写约束版）以及其松弛版本，所有实现并行求解。结果表明 ILP 的平均运行时间约为松弛的 98%（中位数 99%），并在大多数实例上取得最优延迟；在退化路径/星形实例中能在秒级完成，能处理约 10% 节点/路径规模的城市。

**⚠️ 局限性**

局限在于：只考虑退化树（不包含更一般的图形），对大规模网络求解仍受限；仅求解单一最小延迟目标，未探索多阶段或动态规划的更高效方法；复杂性结果未覆盖无容量的路径/星形等特殊子类；实验仅在人工与十座德国城市上验证，未覆盖更复杂或更大城市。

---

## 12. AscendOptimizer: Episodic Agent for Ascend NPU Operator Optimization

**arXiv ID:** 2603.23566 | [PDF](https://arxiv.org/pdf/2603.23566v1)

**作者:** Jiehao Wu `[一作]` (East China Normal University), Xiangfeng Wang `[通讯]` (East China Normal University)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5101927070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对华为 Ascend NPU 的 AscendOptimizer 框架，利用演化搜索和优化重演技术对 AscendC 运算符的主机 tiling 与设备核代码进行交替迭代优化，从而提升运算符的执行吞吐量。

**💡 创新点**

创新点在于：①将 AscendC 运算符拆分为主机 tiling 与设备核两块并行搜索，构建闭环迭代；②通过“优化重演”自监督构建可检索的优化经验库，将稀缺的专家知识转化为可复用的结构化模式；③实现了完全无训练、无规则库的端到端自动优化。

**🔧 技术方法**

技术包括：LLM 生成演化可变模板与代码变异、硬件在环（HIL）性能反馈驱动的进化搜索、优化重演（逆向去优化）构建经验库、检索增强的 kernel 重写（RAG）以及两阶段交替搜索策略。

**📊 数据集**

使用华为官方 AscendC 仓库中的 127 条真实运算符实现作为基准，随后对其进行编译、数值正确性校验并筛选得到最终评测集；实验中还在部分算子上适度增大输入尺寸以降低测量噪声。

**📈 对比分析**

与 BoN、OpenEvolve 等基线在三种难度级别（level1/2/3）下进行对比。AscendOptimizer 在 GM 速度提升上分别达到 1.08、1.21、1.81，fast_1.0 率分别为 46.51%、49.35%、71.43%；在 strict 2.0× 速度提升上，AscendOptimizer 在 level3 上实现 28.57%，显著优于对比基线。

**⚠️ 局限性**

局限性包括：①对动态形状的鲁棒性有限，需手工调整输入尺寸；②HIL 反馈开销较大，优化周期相对较长；③实验依赖 Ascend 910B4 硬件，跨平台可移植性尚待验证。

---

## 13. Environment Maps: Structured Environmental Representations for Long-Horizon Agents

**arXiv ID:** 2603.23610 | [PDF](https://arxiv.org/pdf/2603.23610v1)

**作者:** Yenchia Feng `[一作]` (Distyl AI), Karime Maamari `[通讯]` (Distyl AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并验证了“环境地图”（Environment Maps）这一持久化、可编辑的图结构，用于帮助大语言模型在复杂长周期的软件工作流中保持对环境的全局认识，从而显著提升任务完成率。

**💡 创新点**

创新点在于将屏幕录制、执行轨迹等异构经验统一成四个核心模块（上下文、动作、工作流、隐式知识），实现跨会话、跨 UI 版本的持久化知识共享，并与现有 UI 图、技能库等方法无缝对接。

**🔧 技术方法**

核心技术包括基于 LLM 的事件解析、动作模式抽象、URL 正则化归一化、知识抽取与合并，以及通过文件工具进行结构化查询。

**📊 数据集**

使用 WebArena 基准数据集（5 个网站、812 条自然语言任务）及其公开的人类轨迹记录（179 条任务）。

**📈 对比分析**

在同一 LLM-agent 堆栈下，环境地图实现了 28.2% 的成功率，远超无地图基线 14.2%（+99%）与仅访问原始轨迹 23.3%（+4.9%），并在高分支 UI 场景中带来最大提升。

**⚠️ 局限性**

局限性包括对单域环境的适用性（跨站点工作流支持不足）、对 UI 变更的自动检测与修复缺失，以及目前仅提供描述性预置条件/效应，缺乏实时验证的状态观测。

---

## 14. Causal Reconstruction of Sentiment Signals from Sparse News Data

**arXiv ID:** 2603.23568 | [PDF](https://arxiv.org/pdf/2603.23568v1)

**作者:** Stefania Stan `[一作]` (UBS Business Solutions AG), Shao-Hong Gan `[通讯]` (UBS AG)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出从稀疏新闻标题的情感分类结果中，通过三阶段因果信号重构管线生成可部署的时序情感指标。

**💡 创新点**

将情感估计视为因果信号重构问题，设计了基于不确定性、冗余控制和严格因果规则的模块化框架，并提供无标注评估方案。

**🔧 技术方法**

使用加权聚合、时间衰减缺失填充、指数移动平均、Kalman滤波和Beta‑Binomial光滑等时序处理技术。

**📊 数据集**

基于2024年11月至2026年2月的人工智能相关公司新闻标题（共2513篇）及其对应股价。

**📈 对比分析**

通过无标注诊断、冲击/重复性对照、以及与股价的交叉相关、Granger因果和频谱一致性等多维度外部验证，所选配置在不同聚合方式下均表现出约三周领先的结构化关联，说明重构效果稳定。

**⚠️ 局限性**

局限包括仅使用标题而非全文、固定分类器可能带来的误差、仅关注AI新闻导致的信号弱、样本规模有限且跨机构相关性未完全控制。

---

## 15. Reverse Reconciliation with Soft Information for Discrete-Modulation CV-QKD at Long Range

**arXiv ID:** 2603.23585 | [PDF](https://arxiv.org/pdf/2603.23585v1)

**作者:** Marco Origlia `[一作]` (Sant'Anna School of Advanced Studies), Marco Secondini `[通讯]` (Sant'Anna School of Advanced Studies)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在低信噪比下研究并评估了带软信息的反向重调（RRS）方案，证明其可在超低SNR下关闭与PAM-4调制的性能差距，适用于长距离连续变量量子密钥分发。

**💡 创新点**

提出并验证了通过向Bob公开一组独立于符号的连续指标（CDF变换）来提升互信息的RRS方法，使其在极低SNR下接近理论极限。

**🔧 技术方法**

采用软信息传递、CDF映射、判决阈值设计、蒙特卡罗仿真以及码率自适应的纠错编码技术进行性能评估。

**📊 数据集**

使用自行生成的AWGN仿真数据集（PAM‑4信号与不同码率下的模拟结果），未使用公开实验数据集。

**📈 对比分析**

将RRS与仅使用硬信息的RRH以及PAM‑4调制的BER极限进行对比；在码率0.05时RRH与极限差约0.11，RRS降至0.022；在码率0.01时差约0.12，RRS几乎消除该差距，表明性能显著提升。

**⚠️ 局限性**

局限在于仅在理想AWGN仿真环境下验证，未考虑实际QKD系统中的非高斯噪声、光学误差和多维调制的扩展。

---

## 16. PLDR-LLMs Reason At Self-Organized Criticality

**arXiv ID:** 2603.23539 | [PDF](https://arxiv.org/pdf/2603.23539v1)

**作者:** Burc Gokden `[一作]` `[通讯]` (Fromthesky Research Labs), Burc Gokden (Fromthesky Research Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究PLDR-LLM在自组织临界点下的预训练与推理行为，揭示推理能力源自模型推理输出的稳定状态；

**💡 创新点**

提出基于推理输出的归一化RMSE作为自组织临界的秩序参数，可在无基准测试的情况下精确量化LLM推理与泛化能力；

**🔧 技术方法**

采用Power Law Graph Attention（PLGA）机制的PLDR-LLM架构，利用warm‑up步长与最大学习率调控临界状态，并用RMSE统计推理输出的稳定性；

**📊 数据集**

使用RefinedWeb大规模文本数据进行预训练，IMDB数据用于生成示例，零样本评估使用ARC、Hellaswag、WinoGrande、TruthfulQA、OpenBookQA、PIQA、SIQA等基准集；

**📈 对比分析**

通过比较秩序参数与零样本基准得分发现，临界状态模型秩序参数接近零，基准平均分高于非临界模型；同等参数规模下110M PLDR-LLM在多数基准上优于GPT‑Neo‑125M；

**⚠️ 局限性**

需要精细调节warm‑up与学习率以达到临界，龙王事件可破坏临界状态；方法主要针对PLDR-LLM，尚未验证在更大规模或其他架构上的普适性；

---

## 17. Exploring Self-Tracking Practices of Older Adults with CVD to Inform the Design of LLM-Enabled Health Data Sensemaking

**arXiv ID:** 2603.23733 | [PDF](https://arxiv.org/pdf/2603.23733v1)

**作者:** Duosi Dai `[一作]` (Aarhus University), Sanna Kuoppamäki `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5108322148)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了为期七天的自我跟踪日记和后续访谈，对八名64–82岁心血管疾病患者的自我跟踪实践与数据解读进行深入探讨，提炼出六大主题并给出LLM支持的数据感知设计方向。

**💡 创新点**

在传统自我跟踪研究基础上，首次系统性结合情感、主体性、身体经验视角，提出针对老年心血管患者的六大主题及对应的LLM人本设计原则，填补了该人群数字健康感知工具的设计空白。

**🔧 技术方法**

采用质性研究方法（情绪分析、主题分析）并对LLM在健康数据解读中的潜在交互方式进行概念性讨论；未实现具体LLM模型或系统。

**📊 数据集**

研究数据来源于8名参与者的七天自我跟踪日记（文字+截图）及访谈录音转写，未使用公开的大规模医疗或传感器数据集。

**📈 对比分析**

本研究为探索性定性研究，不涉及对比实验或性能评估；通过访谈与日记内容的主题分析归纳出六大主题，未进行量化指标或算法对比。

**⚠️ 局限性**

样本量小、仅限白人、健康水平相对较高；未收集真实传感器原始数据；未对LLM系统进行实证评估，结果仅为概念性设计建议。

---

## 18. MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG

**arXiv ID:** 2603.23533 | [PDF](https://arxiv.org/pdf/2603.23533v1)

**作者:** Bhavik Mangla `[一作]` `[通讯]` (Independent Research), Bhavik Mangla (Independent Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套针对 Markdown 文档的三阶段 RAG 流水线：结构化分块、一次性 LLM 丰富并滚动键传播、以及基于语义键的块重组。

**💡 创新点**

创新点在于：①单次 LLM 调用即可提取标题、摘要、关键词、实体、问题、语义键和相关键等七个元数据字段；②使用滚动键字典实现上下文同步，避免同义词泛化；③通过语义键进行 bin‑packing 合并相关块。

**🔧 技术方法**

技术实现基于 Python，核心组件包括 Markdown 解析器、单次 LLM 提示模板、滚动键字典（LRU 限制）、bin‑packing 重组算法，以及 BM25 与 dense（FAISS）检索。

**📊 数据集**

使用的评估数据集是 18 篇 Markdown 文档（MDKeyChunker 项目本身，354 KB）以及 30 个人工构造的查询‑答案对。

**📈 对比分析**

与四种配置（固定分块、结构分块、全流程、结构分块+BM25）对比，BM25+结构分块实现 Recall@5=1.000、MRR=0.911；全流程 dense 检索 Recall@5=0.867；结构分块 dense 的 Recall@5 也为 0.867。

**⚠️ 局限性**

局限性包括：①对 LLM 性能高度依赖；②键质量不佳会导致过度合并；③仅适用于 Markdown，其他格式需转换；④滚动键仅在单文档内有效，跨文档统一尚未实现；⑤顺序依赖阻碍并行化。

---

## 19. PIM-CACHE: High-Efficiency Content-Aware Copy for Processing-In-Memory

**arXiv ID:** 2603.23762 | [PDF](https://arxiv.org/pdf/2603.23762v1)

**作者:** Peterson Yuhala `[一作]` (University of Neuchâtel), Valerio Schiavoni `[通讯]` (University of Neuchâtel)

**通讯引用:** 1280 | [OpenAlex ID](https://openalex.org/A5033418614)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了内容感知复制（CAC）机制，利用工作负载相似性通过去重和压缩动态消除CPU到DPU的冗余数据传输。

**💡 创新点**

创新点在于将去重与VByte压缩与UPMEM-PIM的内存拷贝语义相结合，形成轻量级的数据调度层，首次实现跨DPU的内容感知复制并公开开源。

**🔧 技术方法**

使用XXHash64指纹哈希实现块级去重，VByte变长整数压缩，UPMEM SDK的DMA接口，多线程DRM实现高速块级指纹与哈希表查询。

**📊 数据集**

评测采用可控冗余度的合成数据、真实基因组序列（GRCh38、T2T）以及向量加法基准，展示不同冗余程度下的性能变化。

**📈 对比分析**

通过与无内容感知复制对比，合成数据冗余度高时CPU‑DPU传输时间可下降多达14×，VByte压缩可提升5.4×，整体向量加法端到端加速最高可达9.5×；在极高冗余场景下仍略低于CPU 32线程。

**⚠️ 局限性**

局限性包括只针对CPU‑DPU拷贝，未处理DPU‑CPU或DPU‑DPU传输；低冗余场景无效；实现仅使用串行DMA API，需大BRB空间，缺乏动态fallback阈值与完整的多核并行化支持。

---

## 20. Kronecker-Structured Nonparametric Spatiotemporal Point Processes

**arXiv ID:** 2603.23746 | [PDF](https://arxiv.org/pdf/2603.23746v1)

**作者:** Zhitong Xu `[一作]` (University of Utah), Shandian Zhe `[通讯]` (University of Utah)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5024663093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

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

## 21. AgenticNet: Utilizing AI Coding Agents To Create Hybrid Network Experiments

**arXiv ID:** 2603.23763 | [PDF](https://arxiv.org/pdf/2603.23763v1)

**作者:** Majd Latah `[一作]` (Ozyegin University), Kubra Kalkan `[通讯]` (Ozyegin University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

开发了一款利用AI编码代理（Claude）生成混合网络实验脚本的工具，支持纯仿真、纯仿真与仿真相结合的混合模式。

**💡 创新点**

创新点在于：①将仿真与仿真节点共存于同一实验框架；②提供细粒度链路与节点控制；③通过LLM代理实现自然语言驱动的快速实验构建。

**🔧 技术方法**

技术实现包括：Python与C++两版实现；仿真框架OMNeT++/ns-3、仿真器Mininet/OVS；Tap桥接与同步机制；LLM代理（Claude Sonnet）与统一API接口。

**📊 数据集**

未使用公开数据集，实验数据基于自定义流量（UDP、Poisson）与手工搭建的拓扑，涵盖不同负载与阈值测试。

**📈 对比分析**

通过四个实验（Exp1-Exp4）比较三种工作模式及两种编程语言的性能，结果显示C++版在准确度（RTT误差更低）与吞吐量（8.6-9.5×提升）上优于Python，混合模式实现了更灵活的实验配置。

**⚠️ 局限性**

局限性包括：①LLM代理对复杂逻辑的支持仍有限；②同步与桥接在大规模实验中的开销尚未评估；③缺乏对真实网络大规模拓扑的验证，实验规模相对受限。

---

## 22. Leveraging Large Language Models for Trustworthiness Assessment of Web Applications

**arXiv ID:** 2603.23781 | [PDF](https://arxiv.org/pdf/2603.23781v1)

**作者:** Oleksandr Yarotskyi `[一作]` (University of Coimbra), João R. Campos `[通讯]` (University of Coimbra)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5045188001)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究利用大型语言模型自动化评估Web应用的信任度，检查其是否遵守OWASP输入验证安全编码实践。

**💡 创新点**

创新点在于系统化对不同提示工程策略的比较，以及将LLM输出嵌入层次化质量模型(LSP)以计算全局信任得分，实现可扩展、可持续的自动评估。

**🔧 技术方法**

使用了GPT‑4.1、GPT‑4.1‑mini、GPT‑4o‑mini、GPT‑3.5‑turbo和Google Gemini‑2.5‑flash等LLM，并结合Prompt 1–4的不同上下文增强技术。

**📊 数据集**

数据集为WSVD‑Bench（Java Web服务的21个服务、80个操作，共158条SQL注入漏洞），并人工标注每个函数的16条OWASP输入验证实践。

**📈 对比分析**

通过宏观F1、精确率、召回率等指标对四种提示策略进行对比，结果显示规则化提示对低容量模型提升显著，最优模型GPT‑4.1在最优提示下宏观F1≈0.93，信任得分可实现安全/易受攻击的区分。

**⚠️ 局限性**

限制包括仅评估Java语言和输入验证子集、仅使用商业LLM、对提示内容的敏感性导致模型性能波动、以及对LLM直接生成信任分数可靠性不足。

---

## 23. AdvSplat: Adversarial Attacks on Feed-Forward Gaussian Splatting Models

**arXiv ID:** 2603.23686 | [PDF](https://arxiv.org/pdf/2603.23686v1)

**作者:** Yiran Qiao `[一作]` (Case Western Reserve University), Jing Ma `[通讯]` (Case Western Reserve University)

**通讯引用:** 4866 | [OpenAlex ID](https://openalex.org/A5034823980)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对feed-forward 3D高斯点云重建模型的对抗攻击方法AdvSplat；

**💡 创新点**

首次系统性研究此类模型的脆弱性，并设计了基于频域（DCT）参数化的查询高效黑盒攻击，包含梯度估计和梯度无关两种变体；

**🔧 技术方法**

利用DCT低频扰动、自然演化策略（NES）估计梯度、协方差矩阵自适应进化策略（CMA-ES）进行黑盒优化；

**📊 数据集**

在Re10K和DL3DV两个室内外场景数据集上进行实验；

**📈 对比分析**

与三种主流feed-forward 3DGS模型（DepthSplat、NoPoSplat、AnySplat）对比，攻击能显著降低PSNR/SSIM/CLIP/DINO并提升LPIPS，且DCT策略显著减少查询次数、提升效率；

**⚠️ 局限性**

局限性包括：攻击效果受扰动阈值和模型容量影响；梯度估计在某些模型上不稳定；仅针对少数几种模型和数据集，缺乏更广泛验证；未提供针对性防御方案。

---

## 24. Berta: an open-source, modular tool for AI-enabled clinical documentation

**arXiv ID:** 2603.23513 | [PDF](https://arxiv.org/pdf/2603.23513v1)

**作者:** Samridhi Vaid `[一作]` (University of Alberta), J Ross Mitchell `[通讯]` (University of Alberta)

**通讯引用:** 4590 | [OpenAlex ID](https://openalex.org/A5091805367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在阿尔伯塔卫生服务部门部署并验证了开源 AI 记录员 Berta，帮助 198 名急诊医生完成 22148 次记录会话，显著降低成本。

**💡 创新点**

首次将 AI 记录器与现有健康系统基础设施集成、保持数据主权且开源，实现省级规模部署。

**🔧 技术方法**

采用模块化架构，结合自动语音识别（WhisperX、Amazon Transcribe 等）与大型语言模型（Snowflake 私有 GPT‑4o、Ollama、vLLM 等），前端使用 Next.js，后端使用 FastAPI。

**📊 数据集**

以 22148 次急诊会话音频（约 2800 小时）为运营数据，评估系统性能。

**📈 对比分析**

与商业记录器对比，运营成本低于 30 美元/医师/月，成本下降 70–95%，并支持高度可定制的模板和数据留存。

**⚠️ 局限性**

仍未与电子病历系统直接集成，需人工复制粘贴；依赖内部 IT 维护，且开源项目需持续社区支持。

---

## 25. Mixture of Demonstrations for Textual Graph Understanding and Question Answering

**arXiv ID:** 2603.23554 | [PDF](https://arxiv.org/pdf/2603.23554v1)

**作者:** Yukun Wu `[一作]` (Wayne State University), Lihui Liu `[通讯]` (Wayne State University)

**通讯引用:** 78 | [OpenAlex ID](https://openalex.org/A5016462374)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在文本图问答任务中，提出一种基于GraphRAG的框架，利用混合专家（MoE）自动挑选最具信息量的演示样例，并使用查询感知图编码器对检索到的子图进行噪声抑制，提升LLM的推理质量。

**💡 创新点**

创新点包括①在子图检索后引入查询特定图编码器，动态聚焦与问题相关的节点与边，过滤无关噪声；②采用MoE机制对演示样例进行聚类与专家选择，既保证示例多样性，又避免单一示例主导；③在多任务设定下将子图表示映射至LLM输入空间，实现结构化知识与生成模型的无缝融合。

**🔧 技术方法**

技术手段包括：Sentence‑BERT编码子图和查询、Prize‑Collecting Steiner Tree实现连通子图构造、K‑means聚类与余弦相似度用于专家选择、查询条件化的GNN（注意力门控消息传递）进行子图编码、混合专家加权融合演示子图，并将最终表示注入冻结的LLM进行prompt生成。

**📊 数据集**

使用GraphQA基准数据集，其中包括ExplaGraphs、SceneGraphs和WebQSP三种图问答数据集，涵盖实体关系图、场景图与知识图问答，平均节点数从5到1370，边数从4到4252。

**📈 对比分析**

与多种基线对比（Zero‑shot、Zero‑CoT、CoT‑BAG、KAPING、Graph‑Token、G‑Retriever等），该方法在ExplaGraphs和SceneGraphs上分别以约1.1%和1.5%的提升超过最强基线G‑Retriever，并在WebQSP的Hit@1指标上也保持领先，整体性能显著优于传统检索‑增量生成与提示调优方法。

**⚠️ 局限性**

局限性在于：①方法高度依赖子图检索的质量，检索误差会直接影响最终答案；②MoE与查询感知GNN的计算开销相对较大，尤其在大规模图上可能不易扩展；③仅在三种公开图问答基准上验证，缺乏对更复杂知识图或多模态场景的通用性评估；④对极端噪声或无关信息的鲁棒性仍有待进一步提升。

---

## 26. An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System: [Experiments and Analysis]

**arXiv ID:** 2603.23710 | [PDF](https://arxiv.org/pdf/2603.23710v1)

**作者:** Duo Lu `[一作]` (Brown University), Fatma Özcan `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较了在生产级 PostgreSQL 系统中多种过滤向量搜索（FVS）算法的端到端性能，重点关注过滤无关（filter‑agnostic）方案。

**💡 创新点**

提出了系统级视角，揭示了数据库层面的页面访问、元数据间接和并发等开销对 FVS 算法的显著影响，并给出了基于工作负载特性和系统成本的算法选择指南。

**🔧 技术方法**

在 PostgreSQL 兼容系统中实现并评估了图结构（HNSW、NaviX、ACORN、Sweep）与聚类式（ScaNN）索引的多种过滤策略；并开发了数据集无关的工作负载生成器，控制过滤选择性与向量‑谓词相关性。

**📊 数据集**

采用了四个公开数据集（SIFT10M、OpenAI‑5M、Cohere‑10M、Text2Image‑10M）覆盖 5M–10M 向量、128–1536 维度及不同距离度量。

**📈 对比分析**

通过 QPS@95%Recall@10、距离/过滤计数、页面访问、系统开销等指标在多种选择性和相关性组合下对比；结果显示聚类索引在低维/高选择性时更快，图索引在高选择性或负相关时优越，但无一种方案在所有场景中占优。

**⚠️ 局限性**

仅在内存驻留、单表结构和固定查询计划下实验；未考虑更复杂的事务、动态更新频率、列式存储或多租户环境；并且基准仅评估了过滤无关算法，缺乏针对特定过滤属性的索引优化。

---

## 27. Compression Method Matters: Benchmark-Dependent Output Dynamics in LLM Prompt Compression

**arXiv ID:** 2603.23527 | [PDF](https://arxiv.org/pdf/2603.23527v1)

**作者:** Warren Johnson `[一作]` `[通讯]` (Plexor Labs), Warren Johnson (Plexor Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对5,400个API调用的系统复现与扩展，揭示了不同代码生成基准在激进压缩下的输出长度爆炸现象，并提出指令存活概率（Ψ）和压缩鲁棒性指数（CRI）来量化压缩对模型输出和能耗的影响。

**💡 创新点**

创新点在于：①将指令存活概率作为结构化指标解释压缩导致的输出爆炸；②提出跨基准的压缩鲁棒性指数，警示单一基准评估的误导性；③结合NVML直接能耗测量验证令牌压缩与实际能耗的关系，揭示能耗估计偏差。

**🔧 技术方法**

使用了基于先N个词截断的最坏情况压缩方法、Welch检验与自助法的统计分析、Tobit回归校正截断效应，并在公开RunPod GPU上采集NVML功耗数据。

**📊 数据集**

主要数据集包括MBPP、HumanEval、GSM8K三个结构各异的代码生成基准，共计764个任务，并收集了三大模型（DeepSeek-Chat、GPT‑4o‑mini、Mistral‑Large）的输出。

**📈 对比分析**

通过对比不同压缩比、模型与基准的输出令牌数、质量（pass@1）以及能耗，发现DeepSeek在MBPP上爆炸系数高达56×，GPT‑4o‑mini仅1.9×；CRI表明GPT‑4o‑mini在r=0.3下保持85%的质量-效率比，而DeepSeek仅9%。

**⚠️ 局限性**

局限性包括：仅评估三种商用模型且使用最坏的截断压缩；输出上限1024令牌导致截断估计偏低；缺乏对语义感知压缩算法的实验；能耗测量仅在不同硬件上做了校准，无法直接映射到API调用。

---

## 28. Training a Large Language Model for Medical Coding Using Privacy-Preserving Synthetic Clinical Data

**arXiv ID:** 2603.23515 | [PDF](https://arxiv.org/pdf/2603.23515v1)

**作者:** John Cook `[一作]` (Veradigm), Gaurav Kaushik `[通讯]` (Veradigm)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究利用隐私保护的合成电子健康记录（EHR）文本对Llama‑3‑70B模型进行监督微调，以实现ICD‑10‑CM和CPT编码；

**💡 创新点**

创新点在于提出了基于合成、政策感知的训练数据生成管道，并证明单模型微调即可在多层次诊断粒度上达到高精度；

**🔧 技术方法**

使用的技术包括大规模语言模型微调（ArcticTraining + ZeRO‑3）、数据增强、序列打包、JSON结构化输出和多维度评估指标；

**📊 数据集**

数据集为全合成的ICD‑10‑CM与CPT编码样本，采用95/5训练/评估拆分，覆盖先进病症、衰弱与社会决定因素等三大临床领域；

**📈 对比分析**

与零射击基线对比，精确码匹配的F1从0.18提升至0.70（ICD‑10）和0.74（CPT），在高级诊断层级保持高稳定性，并在专家评估中实现级别0~3的召回率0.93–0.86；

**⚠️ 局限性**

局限包括合成数据可能无法完全复制真实EHR的噪声与文档多样性、对SDoH编码性能仍偏低、以及微调对通用医学知识的轻微负面影响。

---

## 29. Computing the Skyscraper Invariant

**arXiv ID:** 2603.23560 | [PDF](https://arxiv.org/pdf/2603.23560v1)

**作者:** Marc Fersztand `[一作]`, Jan Jendrysiak `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于参数θ的V(θ)与W(θ)可视化与参数化方法。

**💡 创新点**

创新点在于将θ分为不同区间（1、1/2、1/3等），并为每个区间设计对应的V和W映射，提供更细粒度的可调性。

**🔧 技术方法**

采用LaTeX与TikZ绘图技术实现参数映射的直观展示，并用数学表达式描述映射关系。

**📊 数据集**

论文中未提及具体数据集，主要聚焦于理论建模与可视化示例。

**📈 对比分析**

由于缺乏实验对比，本文未给出性能评估或与其他方法的比较结果。

**⚠️ 局限性**

主要局限在于缺乏真实数据验证、算法实现细节与实际应用案例，导致方法的有效性与可推广性尚待进一步研究。

---

## 30. Task-Space Singularity Avoidance for Control Affine Systems Using Control Barrier Functions

**arXiv ID:** 2603.23753 | [PDF](https://arxiv.org/pdf/2603.23753v1)

**作者:** Kimia Forghani `[一作]` (University of Maryland), Yancy Diaz-Mercado `[通讯]` (University of Maryland)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5050664586)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于控制障碍函数（CBF）的框架，实时避免机器人与动力系统中的配置相关奇异性，并通过仿真验证其效果。

**💡 创新点**

通过对输入-输出映射矩阵的特征值构造安全约束，将奇异集表征为障碍物实现数值化避免，兼顾欠驱动、完全驱动和过驱动系统。

**🔧 技术方法**

使用控制障碍函数、控制线性系统模型、二次规划（QP）、奇异值分解（SVD）、alpha‑shape 三维障碍物建模与障碍物规避技术。

**📊 数据集**

主要使用仿真生成的数据：二维平面2连杆机械臂与磁性针尖的运动轨迹；未使用公开数据集。

**📈 对比分析**

与传统无CBF本地控制器对比，仿真结果表明控制输入尖峰降低约100倍，轨迹跟踪误差保持在毫米级，系统安全性得到显著提升。

**⚠️ 局限性**

对特征值的解析导数在复杂系统中计算困难；数值近似对噪声和扰动敏感；受限于控制输入约束时可行性不保证；尚未在真实机器人上进行实验验证。

---

## 31. Safe Reinforcement Learning with Preference-based Constraint Inference

**arXiv ID:** 2603.23565 | [PDF](https://arxiv.org/pdf/2603.23565v1)

**作者:** Chenglin Li `[一作]` (Tsinghua University), Hua Geng `[通讯]` (Tsinghua University)

**通讯引用:** 6518 | [OpenAlex ID](https://openalex.org/A5106707908)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于人类偏好学习安全约束的强化学习框架 PbCRL，解决在安全强化学习中难以显式指定复杂约束的问题。

**💡 创新点**

主要创新点包括：①在 Bradley‑Terry 模型中加入死区（dead zone）机制，促使学习的成本分布呈重尾，从而避免期望成本低估导致的安全失效；②设计信噪比（SNR）损失，增强成本差异化，提供更有效的策略优化信号；③采用两阶段训练策略，先在离线偏好数据上预训练成本模型，再在线微调并自适应调节死区，显著降低在线标注负担。

**🔧 技术方法**

使用的技术包括：改进的 Bradley‑Terry 目标、死区安全损失、SNR 损失、Lagrangian 约束优化、两阶段多时间尺度学习及收敛性分析。

**📊 数据集**

实验数据集涵盖：Safety Gymnasium（HalfCheetah、Walker2d、Humanoid、Goal、Push、Button）、自主驾驶仿真（Blocked Road、Lane Change）以及 Llama‑3.2‑1B 的语言模型安全对齐数据。

**📈 对比分析**

与基线（RLSF、PPO‑BT、Safe‑RLHF 等）以及基准 Oracle（PPO‑Lag）比较，PbCRL 在平均回报和安全成本上均优于现有基线，且安全成本接近阈值，表明约束学习与真实安全要求高度对齐。

**⚠️ 局限性**

局限性在于假设人类反馈无噪声或不一致，未来需研究对噪声/冲突反馈的鲁棒性提升。

---

## 32. StateLinFormer: Stateful Training Enhancing Long-term Memory in Navigation

**arXiv ID:** 2603.23571 | [PDF](https://arxiv.org/pdf/2603.23571v1)

**作者:** Zhiyuan Chen `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Ning Ding `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并训练了一种名为StateLinFormer的线性注意力导航模型，采用状态化训练策略以在连续批次中保留并更新记忆状态。

**💡 创新点**

创新点在于将记忆状态在训练期间连续传递，避免每个批次重新初始化，从而使模型能够在长序列上学习并具备隐式的上下文学习（ICL）能力。

**🔧 技术方法**

使用了线性注意力架构（LinFormer）、SPOC框架的编码-解码结构、状态化训练机制以及强化学习/模仿学习的训练目标。

**📊 数据集**

实验数据集包括基于网格的MAZE环境（15×15）和视觉真实感的ProcTHOR环境，均构建了持续目标导航（CON）任务序列。

**📈 对比分析**

通过与无状态线性注意力、SPOC-10M（100M帧）和SPOC-Pretrained（40M帧）进行对比，StateLinFormer在Maze上成功率从0.64提升至0.77、步骤从249降至189；在ProcTHOR上成功率从0.420提升至0.580、步骤从669降至496，显著优于对照模型。

**⚠️ 局限性**

主要局限在于缺乏对状态化训练收敛性和记忆分布稳定性的理论分析，且实验仅局限于导航任务，未验证在语言建模或其他长序列控制任务中的有效性。

---

## 33. Towards Leveraging LLMs to Generate Abstract Penetration Test Cases from Software Architecture

**arXiv ID:** 2603.23698 | [PDF](https://arxiv.org/pdf/2603.23698v1)

**作者:** Mahdi Jafari `[一作]` (Karlsruhe Institute of Technology), Ralf Reussner `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5477 | [OpenAlex ID](https://openalex.org/A5033445475)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（GPT-5.2 与 Gemini-3-Pro）生成从 Palladio 组件模型（PCM）推导出的抽象渗透测试用例（APTC），以支持软件架构阶段的安全评估和后续渗透测试。

**💡 创新点**

提出了 APTC 元模型并验证 LLM 在不同提示策略（zero‑shot、one‑shot、few‑shot、chain‑of‑thought）下能从软件架构模型中自动生成符合 CAWE（Common Architectural Weakness Enumeration）条目的可用、准确用例，填补了架构级渗透测试缺口。

**🔧 技术方法**

使用技术包括：Palladio 组件模型（PCM）作为输入的架构描述语言；CAWE 作为弱点知识库；大型语言模型（GPT‑5.2、Gemini‑3‑Pro）；多种提示工程技术；以及 JSON 模式验证来保证生成的 APTC 结构符合元模型。

**📊 数据集**

采用了三个基于 PCM 的案例（Maintenance、PowerGrid、ABAC‑Banking），共 15 条 CAWE 条目（5 条针对授权弱点）作为评估数据集。

**📈 对比分析**

评估方法为专家评审结合 LLM 辅助评估，使用 Correctness（格式、架构映射与弱点匹配）和 Usefulness（对架构师与渗透测试者的可操作性）两个指标；在最优提示策略下，Gemini‑3‑Pro 的 Usefulness 达到 93%（Correctness 86%），表明生成的 APTC 在实用性与准确性方面表现突出。

**⚠️ 局限性**

限制包括：LLM 有时生成与目标弱点不匹配或引用错误的架构元素，导致用例无效；生成过程受模型随机性和提示细节影响；评估结果受专家主观判断限制；仅针对 3 个案例和 2 个 LLM，泛化性尚待验证。

---

## 34. The Geometric Price of Discrete Logic: Context-driven Manifold Dynamics of Number Representations

**arXiv ID:** 2603.23577 | [PDF](https://arxiv.org/pdf/2603.23577v1)

**作者:** Long Zhang `[一作]` (South China University of Technology), Wei-neng Chen `[通讯]` (South China University of Technology)

**通讯引用:** 19690 | [OpenAlex ID](https://openalex.org/A5050385116)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对LLM的残差流进行Gram-Schmidt分解，揭示任务上下文对表示空间的非等距拓扑扭曲，并证明逻辑决策需要特定的拓扑分裂。

**💡 创新点**

创新点在于提出双模调制机制，区分基于拓扑保持的全局协同与基于特定发散的局部决策分裂，并通过实时向量消融实验确立两者的因果关系。

**🔧 技术方法**

使用的技术包括Gram-Schmidt正交化、RMSNorm等价旋转映射、特定向量消融、层级几何跟踪以及相关性与偏差分析。

**📊 数据集**

数据集为从1到200的整数，包含大小、奇偶、素数等逻辑任务，并在阿拉伯数字与英文单词双模之间同步验证。

**📈 对比分析**

比较方法基于U_sim、C_ij、Pearson r等几何指标与线性等距基线对比，结果显示逻辑任务中U_sim<0，消融后准确率从100%跌至38.57%。

**⚠️ 局限性**

局限性包括仅在数值逻辑场景验证，未覆盖自然语言多跳推理；向量消融粗粒度难以映射到微观注意头；对不同归一化策略的通用性尚未完全验证。

---

## 35. Assessment Design in the AI Era: A Method for Identifying Items Functioning Differentially for Humans and Chatbots

**arXiv ID:** 2603.23682 | [PDF](https://arxiv.org/pdf/2603.23682v1)

**作者:** Licol Zeinfeld `[一作]` (Weizmann Institute of Science), Giora Alexandron `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 1549 | [OpenAlex ID](https://openalex.org/A5059416730)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对学生与大语言模型（LLM）在多项选择题中的表现差异，提出并实现了基于差异项功能（DIF）的检测框架，能够系统识别两者在特定题目上的优势与劣势。

**💡 创新点**

创新点在于将教育心理测量中的DIF方法与负控制分析、项目总分相关诊断相结合，形成了一套既严谨又可解释的LLM评估工具，并将其应用于实际测验数据，推动了AI时代测评设计的理论与方法进步。

**🔧 技术方法**

主要技术包括Mantel–Haenszel DIF、Logistic Regression DIF、负控制（Placebo）检验、项目总分相关（ITC）诊断以及人工专业领域专家的定性评审。

**📊 数据集**

使用了两套测验数据：一份22道高中化学诊断试题（931名学生）和一份40道大学入学定量测验（4800多名考生），以及六款主流聊天机器人（ChatGPT‑4o/5.2、Gemini 1.5/3 Pro、Claude 3.5/4.5）的自动生成答案，共计20次重复。

**📈 对比分析**

通过比较MH‑DIF与LR‑DIF，发现后者具有更低的误报率和更高的特异性，能够更稳定地识别真实的差异项；实验结果显示LR‑DIF在检测正向/负向及非均匀DIF时表现优异，且经领域专家评审验证其可解释性。

**⚠️ 局限性**

局限性包括：仅测试了两种测验类型和六种LLM，样本量对LR‑DIF的稳定性有一定影响；LLM的双峰能力分布可能导致分层噪声；方法复杂度较高，对大规模应用提出挑战。

---

## 36. Ethio-ASR: Joint Multilingual Speech Recognition and Language Identification for Ethiopian Languages

**arXiv ID:** 2603.23654 | [PDF](https://arxiv.org/pdf/2603.23654v1)

**作者:** Badr M. Abdullah `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**通讯引用:** 4536 | [OpenAlex ID](https://openalex.org/A5008875255)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套多语言CTC端到端的ASR模型Ethio-ASR，支持五种埃塞俄比亚语言并同时完成语言识别。

**💡 创新点**

采用联合语言识别标记的共享CTC框架，在少量多语音数据上实现跨语言迁移，同时对元音长度和辅音加长的错误进行了细粒度分析。

**🔧 技术方法**

利用多模态预训练的自监督编码器（AfriHuBERT、MMS、wav2vec‑BERT‑2.0），在WAXAL语料上进行监督微调，并使用单前向CTC解码。

**📊 数据集**

主要使用公开的WAXAL埃塞俄比亚子集（5种语言共约1106小时）以及未见过的FLEURS进行零样本评估。

**📈 对比分析**

与Whisper、Seamless‑M4T、OmniASR等大型多语模型做对比，Ethio‑ASR在WAXAL测试集上平均WER 30.48%，低于所有基线且参数量仅600M。

**⚠️ 局限性**

对低频音素、元音长度和加长的建模仍有不足，且性别失衡导致Tigrinya性别偏差显著，未来需更均衡的数据与专门的声学建模。

---

## 37. Boost Like a (Var)Pro: Trust-Region Gradient Boosting via Variable Projection

**arXiv ID:** 2603.23658 | [PDF](https://arxiv.org/pdf/2603.23658v1)

**作者:** Abhijit Chowdhary `[一作]` (Tufts University), Deepanshu Verma `[通讯]` (Clemson University)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5055719783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于可变投影（Variable Projection）的梯度提升算法（VarPro Boosting），用于构造分离式可学习器的加性模型。

**💡 创新点**

将梯度提升与可变投影相结合，实现二阶弱学习器的闭式线性权重解，并将整个过程视为功能信任域优化，首次给出收敛和超线性速率的理论保证。

**🔧 技术方法**

采用可变投影优化、二阶梯度提升、功能信任域方法、凸二次子问题、Kronecker 结构优化以及 JAX/Equinox 实现。

**📊 数据集**

在合成 2D 回归/分类、MNIST、科学机器学习中的 CDR 系统以及 Higgs 事件分类等标准基准数据集上进行实验。

**📈 对比分析**

与梯度下降训练的弱学习器、XGBoost 树提升以及完整网络（含可变投影）进行对比，VarPro Boosting 在误差/准确率上均优于传统梯度提升，并且在速度上与树提升相当或更快。

**⚠️ 局限性**

对弱学习器表达能力有一定要求，子空间正则化等假设在理论上较强；在极端非凸或大规模任务中收敛速率理论仅在极限可达；对多类别交叉熵等失真损失时需额外调参。

---

## 38. From Physician Expertise to Clinical Agents: Preserving, Standardizing, and Scaling Physicians' Medical Expertise with Lightweight LLM

**arXiv ID:** 2603.23520 | [PDF](https://arxiv.org/pdf/2603.23520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 39. Swiss-Bench SBP-002: A Frontier Model Comparison on Swiss Legal and Regulatory Tasks

**arXiv ID:** 2603.23646 | [PDF](https://arxiv.org/pdf/2603.23646v1)

**作者:** Fatih Uenal `[一作]` `[通讯]` (University of Colorado Boulder), Fatih Uenal (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 Swiss-Bench SBP-002，评估前沿大语言模型在瑞士监管合规（FINMA、Legal-CH、EFK）三大领域内的应用性能。

**💡 创新点**

创新点在于构建了三语、七任务类型的实务导向基准，并引入结构化数值评分与三位LLM评审的集成评估框架，揭示了模型性能分层与开放权重模型的竞争力。

**🔧 技术方法**

采用了结构化多维评分（法律准确性、引用准确性、完整性）与三位异构LLM评审（GPT‑4o、Claude Sonnet 4、Qwen3‑235B）组成的评审小组，以及混合效应线性概率模型进行统计分析。

**📊 数据集**

使用了395个由专家精心设计、涵盖FINMA、Legal‑CH、EFK三大监管领域、七类任务（监管问答、幻觉检测、差距分析、司法区分、法条解释、案例分析、法律翻译）且分布在德语、法语、意大利语（加18个英文实验项）的数据集。

**📈 对比分析**

通过对10款2026年3月前沿模型的零检索评估，得到正确率介于12.9%–38.2%之间，形成三层级性能分布；开放权重模型在某些层级甚至优于封闭源模型，显示高错误率（最高达75%）。

**⚠️ 局限性**

局限包括：域覆盖仅限联邦层面，未包含州法/刑法；翻译质量受机器翻译限制；时间漂移导致参考答案可能失效；单作者设计与评审者可能存在同源偏差；评估仅一次且无检索增强，易被过拟合。

---

## 40. Ukrainian Visual Word Sense Disambiguation Benchmark

**arXiv ID:** 2603.23627 | [PDF](https://arxiv.org/pdf/2603.23627v1)

**作者:** Yurii Laba `[一作]`, Rostyslav Hryniv `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了乌克兰语视觉词义消歧（Visual‑WSD）基准，并在此基准上评估了多种多模态语言模型的表现；

**💡 创新点**

创新点在于首次提出乌克兰语视觉词义消歧基准，结合文本与视觉双模态，系统分析了低资源语言中同音异义词导致的幻觉现象；

**🔧 技术方法**

技术方法包括使用基于CLIP的多模态嵌入、教师学习（Teacher Learning）与MSE对齐策略，以及模型嵌入检索与文本提示排名等；

**📊 数据集**

数据集来源于乌克兰词典的同音异义词列表，采集每个词10张图像（1正样本+9负样本），共覆盖87个同音异义词；

**📈 对比分析**

通过MRR和HIT@1指标进行比较，实验显示所有模型在乌克兰语上的性能显著低于英文基线；M‑CLIP XLM‑Roberta‑Large‑ViT‑B‑16Plus的HIT@1为42.78%，MRR为60.30%，而英文基线为HIT@1 60.48%、MRR 73.88%；

**⚠️ 局限性**

限制在于仅覆盖有限数量的高频名词同音异义词，未纳入词典未记录但在现代乌克兰语中常用的同音异义词，导致基准覆盖范围不足且模型性能仍受低资源数据限制影响。

---

## 41. Leveraging Computerized Adaptive Testing for Cost-effective Evaluation of Large Language Models in Medical Benchmarking

**arXiv ID:** 2603.23506 | [PDF](https://arxiv.org/pdf/2603.23506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 42. Supporting Music Education through Visualizations of MIDI Recordings

**arXiv ID:** 2603.23631 | [PDF](https://arxiv.org/pdf/2603.23631v1)

**作者:** Frank Heyen `[一作]` (University of Stuttgart), Michael Sedlmair `[通讯]` (University of Stuttgart)

**通讯引用:** 5504 | [OpenAlex ID](https://openalex.org/A5037110552)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究开发了一套基于MIDI鼓谱的可视化分析工具，帮助音乐学习者识别演奏错误并改进节奏。

**💡 创新点**

创新点在于将比较可视化技术应用于实际录音与标准曲目之间的对比，提供多种可视化编码以揭示时序误差。

**🔧 技术方法**

使用MIDI数据、密度估计、颜色编码等可视化技术，并结合交互式原型实现动态聚合与播放。

**📊 数据集**

使用公开的鼓MIDI文件作为基准曲目，以及学生/教师的录音MIDI数据。

**📈 对比分析**

通过多种可视化对比多次录音与标准曲目，用户可在交互界面中查看误差分布，原型表现良好但尚处于早期评估阶段。

**⚠️ 局限性**

仅支持能产生MIDI输出的乐器，未覆盖非MIDI乐器，且缺乏大规模用户实验验证。

---

## 43. Grounding Vision and Language to 3D Masks for Long-Horizon Box Rearrangement

**arXiv ID:** 2603.23676 | [PDF](https://arxiv.org/pdf/2603.23676v1)

**作者:** Ashish Malik `[一作]` (Oregon State University), Alan Fern `[通讯]` (Oregon State University)

**通讯引用:** 5589 | [OpenAlex ID](https://openalex.org/A5030052689)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于3D视觉‑语言模型的长周期箱子重排规划器RAMP‑3D，利用3D掩码直接预测拾取对象与放置位置，实现从视觉观测和自然语言指令到高层取放动作的闭环映射。

**💡 创新点**

将3D目标定位能力扩展到多步规划，首次引入配对对比损失与学习的“拾取‑放置”嵌入，构造不需要符号层的掩码空间高层规划；同时通过LLM生成多样化语言数据增强模型的语言泛化。

**🔧 技术方法**

采用UniVLG的3D视觉‑语言框架（Swin Transformer + RoBERTa），加上掩码解码器、配对对比模块、终止预测头，并在多视角RGB‑D输入上进行3D融合。

**📊 数据集**

在IsaacSim仿真仓库环境中生成约9.5k场景样本，包含1–30盒、3种颜色、3种装载结构（托盘、货架、堆叠）以及11种任务约束，语言指令通过模板+LLM改写得到。

**📈 对比分析**

与基线2D‑pointer（LLM规划+2D VLM指向）对比，RAMP‑3D在单步有效率上达96.8%（vs 41.4%），长周期成功率在snap‑to‑target模式下为79.5%（free‑form 66.5%），显著优于2D基线。

**⚠️ 局限性**

局限在于仅处理相对简单的视觉/几何复杂度，缺乏多步预测/全局搜索；对机器人可达性、碰撞规避等低层约束假设过于理想；对高度拥挤场景或严格约束任务的鲁棒性仍有限。

---

## 44. Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks

**arXiv ID:** 2603.23711 | [PDF](https://arxiv.org/pdf/2603.23711v1)

**作者:** Morui Zhu `[一作]` (University of North Texas), Qing Yang `[通讯]` (University of North Texas)

**通讯引用:** 13027 | [OpenAlex ID](https://openalex.org/A5100417913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于视觉的端到端框架dCAP，用于实时估计半挂卡车的前后车厢间的6-DoF姿态，并将动态外参融入BEVFormer实现对卡车的三维目标检测。

**💡 创新点**

创新点在于：①使用跨视角与时序注意力的Transformer，能够在高速操纵与遮挡情况下持续、准确地推断相对姿态；②将动态姿态直接输入BEV检测网络，突破了传统固定外参的瓶颈；③构建了全新的STT4AT仿真基准，提供多视角传感器与真实时间变形的标签。

**🔧 技术方法**

核心技术包括：VGGT视觉几何变换骨干网络、Camera Cross-Attention (CCA)、Camera Temporal Self-Attention (CTA)、自适应层归一化调制的多步迭代回归头以及BEVFormer的融合与检测头。

**📊 数据集**

使用基于CARLA的STT4AT数据集，涵盖8个城镇、87个场景，提供同步的6摄像头、激光雷达、GPS‑IMU传感器以及精确的车辆与卡车间相对姿态标注。

**📈 对比分析**

与静态标定、COLMAP、VGGT、DUSt3R等方法对比，dCAP在姿态估计上翻倍降低平移误差（0.452 m）和旋转误差（0.058 rad），在BEV检测上提升mAP至0.102（相对静态标定提升≈75%），同时保持较低的翻译与角度误差。

**⚠️ 局限性**

局限性：检测精度仍受BEVFormer对刚性车身设计的影响，导致整体AP偏低；实验仅在仿真环境中验证，缺乏真实工况下的鲁棒性与实时性评估；模型对极端遮挡与高频摆动的适应性尚待进一步强化。

---

## 45. Concurrent Streaming, Viewer Transfers, and Audience Loyalty in a Creator Ecosystem: A Minute-Level Longitudinal Study

**arXiv ID:** 2603.23773 | [PDF](https://arxiv.org/pdf/2603.23773v1)

**作者:** Maxwell Shepherd `[一作]` `[通讯]` (Johns Hopkins University), Maxwell Shepherd (Johns Hopkins University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对Hololive English创作者生态系统进行了一项持续3.3年的细粒度分钟级纵向研究，分析了近290万条观众计数数据，探讨了并发直播、观众转移以及观众忠诚度等现象。

**💡 创新点**

创新点在于首次利用大规模分钟级实时数据建立观众重叠与转移的定量模型，并将忠诚度拆解为稳定性、竞争抵抗、峰后保留与底部比例四个可度量维度，揭示了创作者层面的观众排他性。

**🔧 技术方法**

技术手段包括基于Spearman相关系数和小时残差化的竞争效应评估、时间块自举检验、置换检验验证并发频率显著性、算法化的观众转移事件检测（阈值触发与效率计算）以及归一化转移量估计观众重叠。

**📊 数据集**

使用的数据集为来自YouTube Live的18个Hololive English频道共7,762场直播的2,935,985条分钟级观众计数记录，覆盖时间从2022年11月至2026年3月。

**📈 对比分析**

对比方法通过将并发直播与单独直播的每场观众平均值相关联，发现并发直播导致原始观众平均值从约14,377降至6,057；转移效率中位数约为49%；忠诚度四维度在同一组织内差异显著，说明观众忠诚度与创作者相关而非组织层面。

**⚠️ 局限性**

局限性包括缺乏个体观众身份导致重叠估计为间接代理；研究为观测性而非因果，时间调度内生难以完全控制；转移检测算法依赖阈值，可能误判或遗漏事件；仅聚焦单一创作者组织，缺乏跨组织泛化性；以及合成忠诚度指标权重主观。

---

## 46. Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes

**arXiv ID:** 2603.23507 | [PDF](https://arxiv.org/pdf/2603.23507v1)

**作者:** Fangyu Ding `[一作]` (HKUST), Jiacheng Sun `[通讯]` (Huawei)

**通讯引用:** 734 | [OpenAlex ID](https://openalex.org/A5102490085)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的删除-插入扩散语言模型（DID），通过将令牌删除和插入严格地形式化为离散扩散过程，取代了当前掩码扩散语言模型（MDLMs）中的掩码和解掩码过程。

**💡 创新点**

DID通过消除MDLMs中的两个主要计算开销来源，显著提高了训练和推理效率，并提供了更大的生成灵活性，支持可变长度序列并具有内在的自我修正机制。

**🔧 技术方法**

采用了基于得分的训练方法，设计了去噪插入得分熵（DISE）目标，并通过并行动态规划算法高效解决子序列计数问题。

**📊 数据集**

在固定长度和可变长度设置下进行了实验，使用了OpenWebText（OWT）数据集和Stories数据集。

**📈 对比分析**

与MDLMs和现有插入基础语言模型相比，DID在建模性能、采样质量和训练/推理速度上表现出显著优势，且无需任何超参数调优。

**⚠️ 局限性**

DID的局限性在于其在某些特定情况下可能仍然受到固定长度模型的影响，且在实现过程中可能面临复杂性和额外的工程开销。

---

## 47. Synthetic Mixed Training: Scaling Parametric Knowledge Acquisition Beyond RAG

**arXiv ID:** 2603.23562 | [PDF](https://arxiv.org/pdf/2603.23562v1)

**作者:** Seungju Han `[一作]` (Stanford University), Yejin Choi `[通讯]` (Stanford University)

**通讯引用:** 26110 | [OpenAlex ID](https://openalex.org/A5102992157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并验证了 Synthetic Mixed Training 与 Focal Rewriting 两种方法，用于在数据受限领域提升语言模型对新知识的内部化能力；通过在训练中混合合成 QA 与合成文档，并在文档生成中加入针对问题的聚焦重写，实现了 log‑linear 的规模提升。

**💡 创新点**

①发现 QA 与文档合成在规模与生成器强度上表现不同，提出混合训练以同时利用两者的优势；②引入 Focal Rewriting，通过在文档生成时显式条件化问题来提升文档多样性，获得更陡峭的 scaling 曲线。

**🔧 技术方法**

使用 Llama 3.1 8B/70B 及 Qwen3 系列模型作为生成器与目标模型；合成 QA 与 AR 文档生成、梯度空间相似度分析、log‑linear scaling 评估；与 Retrieval‑Augmented Generation (RAG) 做对照实验。

**📊 数据集**

QuaLITY（虚构故事长文阅读）、LongHealth（医学长文）、FinanceBench（财经 PDF 文档）等数据集。

**📈 对比分析**

对比 RAG 基线、仅使用 QA、仅使用文档的合成数据；在 QuaLITY 上 8B 模型通过混合训练 + Focal Rewriting 获得 4.4% 相对提升，整体平均 2.6%；与 RAG 结合后提升 9.1%；在 LongHealth、FinanceBench 亦实现 5/6 场景超越 RAG，展示了 log‑linear 规模性与通用性。

**⚠️ 局限性**

实验仅覆盖至 8B 参数规模，受算力限制；未系统探讨已学知识的遗忘问题；主要聚焦于单模态文本知识学习，尚未验证在更大模型或多模态场景中的效果。

---

## 48. optimade-maker: Automated generation of interoperable materials APIs from static data

**arXiv ID:** 2603.23536 | [PDF](https://arxiv.org/pdf/2603.23536v1)

**作者:** Kristjan Eimre `[一作]` (PSI Center for Scientific Computing, Theory and Data), Giovanni Pizzi `[通讯]` (PSI Center for Scientific Computing, Theory and Data)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套自动化工具，可将原始原子结构数据直接转换为符合 OPTIMADE 标准的 REST API，并在 Materials Cloud Archive 中实现自动发布。

**💡 创新点**

创新之处在于消除了部署兼容 API 的技术壁垒，通过 YAML 配置实现多格式数据无缝转换，并首次将 CSD/ICSD 等商业数据库映射到统一的 OPTIMADE 接口。

**🔧 技术方法**

技术方案基于 Python，结合 ASE、pymatgen、FastAPI、MongoDB、YAML 以及 AiiDA 集成，并在前端提供了重构后的 JavaScript 客户端。

**📊 数据集**

使用的数据集包括 Materials Cloud Archive 上的多个科研项目数据、Cambridge Structural Database（CSD）和 Inorganic Crystal Structure Database（ICSD）的实时快照，以及 AiiDA 工作流输出。

**📈 对比分析**

通过与现有自定义 API 对比，验证了工具在本地快速部署、生产级服务以及跨数据库查询的兼容性与性能；示例中 API 能即时响应 OPTIMADE 查询，展示了低延迟和高可扩展性。

**⚠️ 局限性**

局限性主要体现在需依赖现有解析器支持有限格式，生产环境仍需 MongoDB；CSD/ICSD 的映射需定期更新且受许可限制；工具目前仅覆盖结构与部分属性，未覆盖更广泛的实验数据或高级查询功能。

---

## 49. PoiCGAN: A Targeted Poisoning Based on Feature-Label Joint Perturbation in Federated Learning

**arXiv ID:** 2603.23574 | [PDF](https://arxiv.org/pdf/2603.23574v1)

**作者:** Tao Liu `[一作]` (Harbin Engineering University), Wu Yang `[通讯]` (Harbin Engineering University)

**通讯引用:** 10533 | [OpenAlex ID](https://openalex.org/A5100753558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对联邦学习工业图像分类的针对性中毒攻击 PoiCGAN，利用条件生成式对抗网络自动生成带标签翻转的中毒样本并引导恶意客户端模型训练。

**💡 创新点**

创新点包括：① 在 CGAN 中加入源类-目标类误标训练，使生成器自动完成标签翻转；② 采用一对一目标攻击，显著降低对主任务性能的影响；③ 通过控制 CGAN 训练轮数与缩放因子实现模型的高隐蔽性；④ 设计了模型不可区分性分数（MIS）评估恶意模型的隐蔽性。

**🔧 技术方法**

核心技术：联邦学习框架、条件生成式对抗网络（CGAN）、Poison Sample Generator (PSG) 模块、PCA 降维+聚类评估 MIS、攻击成功率（ASR）、准确率（ACC）以及对比基线的实验评估。

**📊 数据集**

实验使用三种工业图像分类数据集：InsPLAD-fault（电力线缺陷）、NEU-CLS（钢表面缺陷）和 Kylberg 纹理数据集。

**📈 对比分析**

与基线方法（TDP、TMP、ADA）比较，PoiCGAN 在三组数据集上的平均 ASR 为 83.97%，比基线高 15–30%；主任务 ACC 仅下降 <8.9%；MIS 值最高，表明模型隐蔽性最佳。在 Krum、RLR、FLAME 等高级防御下，PoiCGAN 仍能保持 ASR 约 56–67%，显示出强鲁棒性。

**⚠️ 局限性**

局限性：需一定比例的恶意客户端（PMR）才能显著提升攻击效果；对 CGAN 训练轮数、缩放因子高度敏感；主要针对图像分类，未验证文本、音频等领域；在极强或针对性的防御策略下，攻击效果可能受到限制。

---

## 50. Probabilistic Geometric Alignment via Bayesian Latent Transport for Domain-Adaptive Foundation Models

**arXiv ID:** 2603.23783 | [PDF](https://arxiv.org/pdf/2603.23783v1)

**作者:** Kuepon Aueawatthanaphisut `[一作]` `[通讯]` (Khon Kaen University), Kuepon Aueawatthanaphisut (Khon Kaen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于贝叶斯潜在传输和PAC‑Bayesian约束的概率几何对齐框架，用于在低样本目标域下对大型基础模型进行自适应；

**💡 创新点**

将随机几何传输算子与PAC‑Bayes泛化界结合，首次通过贝叶斯传输重分布潜在概率质量并引入不确定性扩散，实现对齐与泛化的统一；

**🔧 技术方法**

贝叶斯变分推断、Wasserstein型随机传输、PAC‑Bayes理论、Fokker–Planck扩散、熵正则化Sinkhorn迭代、梯度流优化等；

**📊 数据集**

在常用域适配基准上验证，如 Office‑31、Office‑Home、VisDA‑2017 等；

**📈 对比分析**

与传统微调、DANN、Bayesian DA 等基线相比，在几何不匹配、目标风险、方差及传输能量等四个指标均显著下降，整体提升约 50%–60%（目标风险从 0.61 降至 0.31，几何不匹配从 0.58 降至 0.27）；

**⚠️ 局限性**

计算成本较高，尤其是高维潜在空间的协方差传播与 Sinkhorn 迭代；对极端高维或大规模模型的可扩展性尚待验证；实验仅涵盖有限的合成/中等难度域转移，缺乏真实世界大规模部署验证。

---

## 51. Space Fabric: A Satellite-Enhanced Trusted Execution Architecture

**arXiv ID:** 2603.23745 | [PDF](https://arxiv.org/pdf/2603.23745v1)

**作者:** Filip Rezabek `[一作]` (SpaceComputer), Amir Yahalom `[通讯]` (SpaceComputer)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 Space Fabric 架构，在卫星上实现可信执行环境（TEE）与远程证明，并通过卫星执行保证协议（SEAP）将工作负载执行绑定到具体卫星，同时实现上轨密钥生成、双安全元件交叉验证等。

**💡 创新点**

创新点包括：1）将可信计算堆栈迁移到卫星，利用卫星的物理不可访问性消除对预部署秘密的依赖；2）首次实现上轨密钥生成并与双安全元件共同签名，消除单厂商信任窟；3）SEAP 协议通过拜占庭容错的地面站共识将 TEE 证明绑定至卫星，解决空间环境下的“谁在运行”问题；4）完整的威胁模型与正式安全分析。

**🔧 技术方法**

技术手段：ARM TrustZone+OP‑TEE+GoTEE、NXP SE050 与 TROPIC01 双安全元件、Veraison 远程证明框架、基于 TPM 的地面机测量、Secure Boot 与 HAB、RPC 与 RPMB、ECDSA/P256 签名、拜占庭投票与时间窗口机制等。

**📊 数据集**

未使用传统机器学习数据集，而是通过在 USB Armory Mk II + Raspberry Pi 5 上实现硬件原型，利用其固件与安全元件进行实验评估。

**📈 对比分析**

与现有 TEE 平台（SGX、TDX、SEV、ARM CCA、OpenTitan）对比，Space Fabric 在无预部署密钥、双厂商根信任、卫星位置证明方面提供更强保障；实验显示一次完整 SEAP 认证需要 4–7 个地面站通道（≈6–11 小时）并且每次交换仅 1.9 KB，延迟约 210–620 ms；在 GEO 可单次完成。

**⚠️ 局限性**

局限性：1）仍需至少部分地面站诚实且分布；2）对卫星操作员完全控制的情形无安全保证；3）消息中继攻击与渠道控制限制；4）需要手动注册 GS 公钥与卫星身份；5）硬件组件（如 SE050）闭源；6）空间辐射可能导致密钥损坏。

---

## 52. ROSCell: A ROS2-Based Framework for Automated Formation and Orchestration of Multi-Robot Systems

**arXiv ID:** 2603.23690 | [PDF](https://arxiv.org/pdf/2603.23690v1)

**作者:** Jiangtao Shuai `[一作]` (Technical University of Berlin), Manfred Hauswirth `[通讯]` (Technical University of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了ROSCell框架，实现基于ROS2与Docker的多设备自适应计算单元与任务部署。

**💡 创新点**

关键创新在于动态形成隔离计算单元、基于技能模型的无缝软件封装及轻量级资源分配器，显著降低了K3s的系统开销。

**🔧 技术方法**

结合ROS2、DDS、Docker、Kubernetes调度器、JSON Schema等技术实现统一管理与自动化部署。

**📊 数据集**

采用Raspberry Pi集群、AprilTag姿态估计数据集（RealSense D435i相机采集的多场景AprilTag图像）进行实验。

**📈 对比分析**

通过与K3s在CPU、内存、网络吞吐量等指标对比，ROSCell在空闲状态下CPU仅占2.8%、内存28.4%，网络负载低92%，多目标姿态估计下DS‑III方案平均延迟最低。

**⚠️ 局限性**

目前缺乏容错与安全机制，且对大规模动态拓扑的自恢复支持有限，未来需加强容错与加密功能。

---

## 53. AgentRFC: Security Design Principles and Conformance Testing for Agent Protocols

**arXiv ID:** 2603.23801 | [PDF](https://arxiv.org/pdf/2603.23801v1)

**作者:** Shenghan Zheng `[一作]` (Dartmouth College), Qifan Zhang `[通讯]` (Palo Alto Networks)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向AI代理协议的安全框架，并对MCP、A2A、ANP、ACP等协议进行了形式化分析和实现级验证。

**💡 创新点**

创新点在于提出六层Agent协议栈、统一的Agent无关安全模型及Composition Safety原则，并实现从规范提取到模型检查再到SDK测试的全流程管道。

**🔧 技术方法**

使用TLA+描述规范，TLC模型检查器进行符号化验证，并结合自研的规范抽取与IR编译工具生成可执行的实现测试。

**📊 数据集**

主要数据集为各协议的官方规范文档、示例与Python SDK实现；通过对比多版本协议实现，检索了约5个协议、42个SDK测试。

**📈 对比分析**

方法上先在规范层面对11项安全原理做独立检查，得到33条违规，然后在实现层重放约42条对照测试，发现约20条Composition Safety违规，验证了框架的有效性。

**⚠️ 局限性**

局限性包括模型仅在有限状态空间内验证、规范抽取仍需人工干预、协议版本快速迭代导致结果随时间失效。

---

## 54. Learning What Can Be Picked: Active Reachability Estimation for Efficient Robotic Fruit Harvesting

**arXiv ID:** 2603.23679 | [PDF](https://arxiv.org/pdf/2603.23679v1)

**作者:** Nur Afsa Syeda `[一作]` (Washington State University), John Miller `[通讯]` (Washington State University)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5015369282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种基于RGB‑D感知与主动学习的决策级可达性预测框架，能够在果园中快速判断苹果是否可被机器人抓取，从而减少IK计算量。

**💡 创新点**

将可达性视为二分类任务，采用主动学习筛选最具信息量的样本，显著降低标注成本，并在低标签场景下实现比传统方法更高的准确率。

**🔧 技术方法**

YOLO目标检测、RGB‑D像素到3D坐标映射、相机到机械臂坐标变换、随机森林二分类器以及熵、边际、查询委员会等主动学习策略。

**📊 数据集**

采集自华盛顿州苹果园的Kinect v2 RGB‑Depth数据，包含约3,480对图像，构建1,000个可达性标签样本以及约24,000个未标记候选点。

**📈 对比分析**

在不同初始标注量（10/30/50）和查询预算（50/100）下与随机采样对比，熵/边际策略在低标签下实现94–99%准确率，较随机采样提升6–8%，并能过滤约38%候选点，显著减少IK调用。

**⚠️ 局限性**

依赖精确的相机-机械臂外参校准，未考虑动态障碍物或极端光照变化，且在大规模场景下需进一步验证实时性与可迁移性。

---

## 55. Optimal Unlabeled Pebble Motion on Trees and its Application to Multi-Agent Path Finding

**arXiv ID:** 2603.23503 | [PDF](https://arxiv.org/pdf/2603.23503v1)

**作者:** Annalisa Calvi `[一作]`, Edward Lam `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了无标签弹珠在树上移动（UPMT）问题，并将其扩展到多代理路径规划（MAPF），提出了线性时间的最优UPMT算法和子最优的未标记MAPF算法，给出了最优时间和成本界限；

**💡 创新点**

创新点在于首次实现了O(nlog n)时间的最优UPMT算法，并通过需求函数将该思想应用到树形MAPF，给出了子最优算法及其最坏情况上界（makespan≤n−k，sum-of-costs≤k(n−k)），同时推导了平均计划长度的上界√(Dk(n−k))，并证明了算法可并行；

**🔧 技术方法**

核心技术包括递归自顶向下的需求函数策略、最小匹配/网络流比较、理论证明与复杂度分析、以及实验验证；

**📊 数据集**

实验使用随机树（均匀分布标签树）以及路径等数据集，规模可达10⁶个节点；

**📈 对比分析**

与旧的O(n³)算法以及其他已知上界相比，最优UPMT算法在时间上与输入输出编码大小等价；实验显示平均计划长度远小于k(n−k)，且上界被√(Dk(n−k))所改善，性能显著优于传统方法；

**⚠️ 局限性**

局限性包括：子最优MAPF算法并非最优（在makespan和sum‑of‑costs上）；对一般图只能通过选取生成树得到可行解；最坏情况仍为O(n²log n)；平均性能仍未完全逼近理论最优，仍有进一步改进空间。

---

## 56. A Theory of LLM Information Susceptibility

**arXiv ID:** 2603.23626 | [PDF](https://arxiv.org/pdf/2603.23626v1)

**作者:** Zhuo-Yang Song `[一作]`, Hua Xing Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了大型语言模型（LLM）介入优化过程时的“信息敏感度”理论，阐述在计算资源充足时，固定LLM层无法提升策略集对预算的性能敏感度。

**💡 创新点**

创新点在于将线性响应理论与代理系统结合，提出了可测量的相对敏感度α≤1作为固定LLM层的上限，并证明在多种任务和模型规模下成立；同时揭示嵌套共伸缩架构可突破该上限，为开放式自我改进提供必要结构条件。

**🔧 技术方法**

主要技术包括：多变量效用函数框架、相对敏感度测量、数据处理与信息论（数据处理不等式）推导、针对多种任务的prompt设计与奖励函数对比、统计线性回归分析等。

**📊 数据集**

使用的基准数据集覆盖四个领域：Tetris（游戏策略），0/1 Knapsack（组合优化），World-knowledge Ranking（事实检索/排序），以及AIME数学题集（多步推理）。

**📈 对比分析**

与固定LLM策略对比，通过实验展示在每个域中，固定LLM的相对敏感度均不超过1；在嵌套共伸缩的AIME配置中，α可超过1，性能突破固定层上限。总体而言，固定LLM在大预算下提供的提升有限，而共伸缩架构能显著提升性能。

**⚠️ 局限性**

局限性包括：理论仍为经验假设，缺乏严格证明；实验仅覆盖有限的四个任务域，未考察长时序、多智能体或连续动作空间等更复杂情境；共伸缩的具体规模律尚未量化，缺乏对资源分配的精细指南。

---

## 57. Bio-Inspired Event-Based Visual Servoing for Ground Robots

**arXiv ID:** 2603.23672 | [PDF](https://arxiv.org/pdf/2603.23672v1)

**作者:** Maral Mordad `[一作]` (Northeastern University), Milad Siami `[通讯]` (Northeastern University)

**通讯引用:** 1044 | [OpenAlex ID](https://openalex.org/A5058687251)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用动态视网膜传感器（DVS）在地面机器人上实现基于事件的视觉伺服控制，直接通过对特定空间核（线性、二次）处理事件计数来估计机器人位置-速度乘积和速度，并在此基础上设计了生物启发的极限环主动感知控制器实现闭环控制。

**💡 创新点**

创新点在于：①不需要传统的状态估计或特征提取，直接通过事件计数与空间核匹配得到所需的非线性反馈；②引入极限环主动感知控制器，克服事件感知在平衡点失去可观测性的固有限制；③实现了极低延迟、计算效率高且可实现指数收敛的纯事件驱动闭环控制。

**🔧 技术方法**

使用技术包括：动态视网膜传感器（Prophesee EVK4）、双模式光强显示（上部二次、下部线性）、事件计数核估计、基于空间核的事件流解析、极限环主动感知控制器设计、离散时间控制实现、实验平台1/10规模Quanser QCar。

**📊 数据集**

使用的数据集为实验室内部生成的二维光强模式（线性/二次）并通过六台运动捕捉摄像头获得真实位置与速度作为基准；未使用公开的标准数据集。

**📈 对比分析**

通过与运动捕捉测得的真值比较，验证了事件计数估计器的误差在理论上限内；闭环实验显示机器人能在给定幅值和频率下稳定在目标点，且表现出指数收敛特性；相较于传统基于特征的IBVS，展示了更低的计算量和更快的响应。

**⚠️ 局限性**

限制主要体现在：①需要预先设计的线性/二次光强场景和简单矩形核，难以直接迁移到复杂或动态目标环境；②目前仅在一维地面机器人上验证，二维/三维运动控制尚未实现；③对事件相机的依赖使得对低光或光强极差的场景表现受限。

---

## 58. AetherWeave: Sybil-Resistant Robust Peer Discovery with Stake

**arXiv ID:** 2603.23793 | [PDF](https://arxiv.org/pdf/2603.23793v1)

**作者:** Kaya Alpturer `[一作]` (Princeton University), Aviv Zohar `[通讯]` (Hebrew University)

**通讯引用:** 5429 | [OpenAlex ID](https://openalex.org/A5015458414)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于质押的对等发现协议AetherWeave，旨在通过将网络参与与用户质押挂钩来抵御Sybil攻击并保持P2P网络的连通性。

**💡 创新点**

创新点在于同时实现Sybil抵抗与隐私保护：节点可证明持有有效质押但不透露具体金额；使用加密承诺速率限制发现请求，超限时生成链上可验证的违规证明并触发斩首；并实现了几乎不需要链上交互、通信复杂度为O(s√n)的高效协议。

**🔧 技术方法**

采用加密承诺方案、链上质押与斩首机制、均值场分析、以及以太坊共识客户端Prysm的原型实现。

**📊 数据集**

主要使用仿真实验与以太坊Prysm测试网络的数据；实验基于模拟对等网络并在真实以太坊客户端上部署原型。

**📈 对比分析**

通过均值场收敛闭式界、广泛的对抗性仿真以及端到端原型验证，证明在高概率下要么保持诚实节点连通，要么在任何小组件中至少有(1‑δ)比例节点发出攻击检测标志；与传统对等发现协议相比，通信量显著降低（O(s√n)），并提供链上可验证的违规处罚。

**⚠️ 局限性**

局限性包括：需要用户持有质押，导致经济成本；对极大规模恶意攻击的抵御仍依赖质押规模；原型主要在以太坊生态中验证，跨链适用性尚未充分评估；在高动态或极低延迟网络中的表现仍待进一步研究。

---

## 59. Self-Evolving Multi-Agent Framework for Efficient Decision Making in Real-Time Strategy Scenarios

**arXiv ID:** 2603.23875 | [PDF](https://arxiv.org/pdf/2603.23875v1)

**作者:** Li Ma `[一作]`, Lei Ren `[通讯]` (Beihang University)

**通讯引用:** 8467 | [OpenAlex ID](https://openalex.org/A5028426177)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SEMA框架，利用LLM进行RTS（StarCraft II）实时决策，通过自演化的多智能体协同实现高效、低延迟的决策。

**💡 创新点**

创新点包括：①结构熵驱动的动态观察剪枝，显著压缩输入冗余；②多智能体闭环（决策、评估、策略）自演化机制，抑制LLM随机性；③跨层次知识融合（微观轨迹、宏观经验、领域规则），提升逻辑一致性。

**🔧 技术方法**

核心技术：结构熵与编码树建模、动态观察剪枝、评估代理实时检索、策略代理自学习、Qwen3‑next‑80b大语言模型推理。

**📊 数据集**

数据集：StarCraft II多张地图（Melee、SMAC 3m/8m/25m、Flat32/Flat48/Flat64、Simple64），对战使用内置 AI。

**📈 对比分析**

与 Rule‑Based、Single‑LLM、TextStarCraft、HIMA 等基线对比，SEMA 在 8 张地图上赢率提升至 100%/68%，平均决策时延降至 0.5–0.9 s，token 消耗大幅下降（≈2.2k/step）。

**⚠️ 局限性**

局限性：①对极端高密度地图仍有轻微性能波动；②依赖手工设定的结构熵阈值与剪枝容量；③验证范围仅限于 StarCraft II，泛化到其他RTS或非游戏场景尚待研究。

---

## 60. n-VM: A Multi-VM Layer-1 Architecture with Shared Identity and Token State

**arXiv ID:** 2603.23670 | [PDF](https://arxiv.org/pdf/2603.23670v1)

**作者:** Jian Sheng Wang `[一作]` `[通讯]` (Yeah LLC), Jian Sheng Wang (Yeah LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种Layer-1架构n‑VM，支持多种虚拟机（EVM、SVM、BVM、TVM等）共存，统一身份与代币账本，实现跨VM原子转账与并行执行。

**💡 创新点**

创新点在于①将身份与授权解耦，利用单一32字节承诺生成各VM地址；②构建单一代币账本，提供ERC‑20/SPL等多接口；③采用opcode前缀路由、写集冲突检测与上下文分片并行调度，消除桥接与多钱包需求。

**🔧 技术方法**

核心技术包括：HKDF+Poseidon哈希生成身份承诺；opcode‑prefix路由器；统一状态树与身份层；零知识证明绑定交易；写集提取与冲突检测；基于上下文的分片调度；Rust实现与revm、SVM、Script等引擎。

**📊 数据集**

无专门数据集，采用模拟与理论分析评估吞吐量；实验基准基于16核服务器与标准交易类型。

**📈 对比分析**

通过理论模型与实验模拟比较，单线程约5kTPS，写集并行约16kTPS，分片并行可达33k–66kTPS；对比现有多VM链（如Sei v2、Movement等）显示无桥接、单钱包的优势。

**⚠️ 局限性**

局限性包括：缺乏同步跨VM调用与共享Gas计量；写集调度对EVM复杂交互保守；仍需完善原子跨VM合约执行；可扩展性需加入WASM、Move等更多VM。

---

## 61. Prototype Fusion: A Training-Free Multi-Layer Approach to OOD Detection

**arXiv ID:** 2603.23677 | [PDF](https://arxiv.org/pdf/2603.23677v1)

**作者:** Shreen Gul `[一作]` (Missouri University of Science and Technology), Sanjay Madria `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 4804 | [OpenAlex ID](https://openalex.org/A5012569039)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用预训练网络的中间层特征构建类原型，使用余弦相似度聚合多层信息进行无训练的异常检测。

**💡 创新点**

创新点在于放弃仅用最终层，聚合多个中间层的类原型，并用简单的余弦相似度实现高效、架构无关的OOV检测。

**🔧 技术方法**

技术包括全局平均池化、L2归一化、类均值原型构建、余弦相似度评分及多层加权平均。

**📊 数据集**

数据集涵盖CIFAR-10、CIFAR-100、ImageNet-1k的ID数据，并在多种标准OOD数据集（iSUN、LSUN、iNaturalist、Textures、Places、SVHN、SUN、Places等）上评估。

**📈 对比分析**

与MSP、MaxLogit、Energy、Mahalanobis、GradNorm、NNGuide、NECO、ReAct、ViM、ESOOD、LaREx等后置基线对比，AUROC平均提升约4-5%，FPR下降约10-15%，在多种网络结构上持续领先。

**⚠️ 局限性**

局限包括需要手动挑选或统一权重的中间层，依赖少量校准样本，且对极远OOD或非分类任务的泛化性尚未验证。

---

## 62. DepthCharge: A Domain-Agnostic Framework for Measuring Depth-Dependent Knowledge in Large Language Models

**arXiv ID:** 2603.23514 | [PDF](https://arxiv.org/pdf/2603.23514v1)

**作者:** Alexander Sheppert `[一作]` (Legacy Health), Alexander Sheppert `[通讯]` (Legacy Health)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5122770403)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DepthCharge 框架，用于在任意可验证知识领域内自适应探测 LLM 的知识深度。

**💡 创新点**

创新点包括：1）自适应钻取，按模型自身提及的概念生成后续问题；2）即时事实验证，实时检索维基百科或专业文献做答案判定；3）恒定样本量的存活统计，保证每个深度层 30 题的统计可信度。

**🔧 技术方法**

技术手段包括：LLM 生成问题与概念抽取、检索增强事实验证、熵式累计存活率（EVD）计算、基于 Wilson 区间的置信区间估计。

**📊 数据集**

数据集由公开可验证知识来源构成：维基百科（COMMON、TEXTBOOK 层）、专业文献检索（PROFESSIONAL、SPECIALIST、CUTTING_EDGE 层），覆盖四个领域（医学、宪法、古罗马、量子计算）。

**📈 对比分析**

评估方法通过在五款前沿模型上进行自适应测评，得到期望有效深度 (EVD) 从 3.45 到 7.55，模型排名随领域变化，且高价模型并不一定拥有更深的知识深度，显示成本与深度并非正相关。

**⚠️ 局限性**

局限包括：1）评估结果相对评估器模型，绝对分数可能因评估器差异而偏差；2）不同模型得到的具体问题不完全可比，主要用于同域内比较；3）在事实稀缺或极为专业的子领域，验证困难导致探测深度受限。

---

## 63. DISCO: Document Intelligence Suite for COmparative Evaluation

**arXiv ID:** 2603.23511 | [PDF](https://arxiv.org/pdf/2603.23511v1)

**作者:** Kenza Benkirane `[一作]` (Parexel AI Labs), Aneiss Ghodsi `[通讯]` (Parexel AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DISCO 文档智能评测框架，分离文本解析和问答阶段，分别评估 OCR 管道和 VLM 在不同文档类型上的性能。

**💡 创新点**

① 明确区分感知、表示与推理三阶段，提供细粒度错误归因；② 通过任务感知提示对比不同提示效果；③ 通过小样本多任务集合对 OCR 与 VLM 进行系统对比，揭示结构与语言的互补性。

**🔧 技术方法**

OCR 识别系统、VLM（Base、Task‑Aware）、LLM 生成 QA、指令式提示（generic、cot、task‑aware）、确定性解码、统一图像分辨率；评价指标包括 CER、WER、GT‑in‑Pred、ANLS、EM。

**📊 数据集**

IAM_DISCO、ICDAR_DISCO、RxPad、DocVQA_DISCO、InfographicVQA_DISCO、DUDE_DISCO、ChartQAPro_DISCO、VisR‑Bench、PubLayNet（均限制为 500 样本或采样子集）。

**📈 对比分析**

采用 P_OCR、P_VLM‑base、P_VLM‑task 进行解析比较，P_OCR、P_VLM‑2stage、P_VLM‑direct 进行问答比较；结果显示 OCR 在手写及多页长文档中更稳健，VLM 在多语种场景和视觉丰富版式上优势明显；任务感知提示对部分文档提升明显，对其他文档则略有下降；直接 VLM QA 在单页文档上最佳，OCR 在多页长文档上仍占优。

**⚠️ 局限性**

仅使用 500 样本或子集，可能不代表全部真实情况；VLM 对长上下文的处理仍有限；任务感知提示效果不稳定；评测未覆盖 VisR‑Bench 与 PubLayNet 的完整结果；仅关注最终准确率，尚缺少更细粒度的错误分析。

---

## 64. LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load

**arXiv ID:** 2603.23640 | [PDF](https://arxiv.org/pdf/2603.23640v1)

**作者:** Pranay Tummalapalli `[一作]` (Conscious Engines), Kautuk Kundan `[通讯]` (Conscious Engines)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 Qwen 2.5 1.5B 4-bit 量化模型在四种平台（Raspberry Pi 5+Hailo‑10H NPU、Samsung Galaxy S24 Ultra、iPhone 16 Pro、笔记本 RTX 4050 GPU）上进行持续推理基准，关注吞吐、时延、功耗与热行为；

**💡 创新点**

创新点在于首次独立跨平台评估持续 LLM 推理，包括专用边缘 NPU，并详细绘制了手机平台的热衰减曲线与 NPU 的热稳定性；

**🔧 技术方法**

使用 Qwen 2.5 1.5B 4-bit 量化模型，采用 hailo‑ollama、MLC‑LLM、MLX、vLLM 等推理框架，配合硬件功耗监测（INA219、GPU 能耗采样、iOS 电量占用）进行实验；

**📊 数据集**

实验使用固定 258‑token 提示（总长度约 270–280 tokens）并生成 564–1789 tokens，模型本身为 Qwen 2.5 1.5B，未使用公开数据集；

**📈 对比分析**

对比方法为 20 次热条件下的持续推理，记录吞吐、时延、功耗、能耗与温度；结果显示 RTX 4050 最高吞吐 131.7 tok/s，iPhone 16 Pro 在热状态下降至 22.6 tok/s，S24 Ultra 受 OS 限制仅 9.9 tok/s，Hailo‑10H 6.9 tok/s 但功耗仅 1.9 W，能耗比相近；

**⚠️ 局限性**

局限性包括：仅评估单一 4-bit 模型和单一提示，平台间测量方法不一致（尤其 Android、iOS 无功耗数据）、设备数量单一、热阈值受环境影响、量化格式差异导致性能对比受限。

---

## 65. Sparse Autoencoders for Interpretable Medical Image Representation Learning

**arXiv ID:** 2603.23794 | [PDF](https://arxiv.org/pdf/2603.23794v1)

**作者:** Philipp Wesp `[一作]` (Stanford University), Sergios Gatidis `[通讯]` (Stanford University)

**通讯引用:** 6001 | [OpenAlex ID](https://openalex.org/A5080097591)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

将稠密视觉基础模型的嵌入转化为可解释的稀疏特征，并证明这些稀疏特征在多模态CT/MRI数据上能够恢复大部分下游性能、保持检索质量，并能被语言模型自动描述。

**💡 创新点**

首次在多模态自监督视觉基础模型上应用Matryoshka稀疏自编码器与BatchTopK稀疏化，展示稀疏特征既能保留98%以上的检索相似度，又能用单一概念描述并支持零样本语言驱动检索。

**🔧 技术方法**

使用Matryoshka稀疏自编码器、BatchTopK稀疏化、JumpReLU阈值、VLM MedGemma 27B自动概念生成与LLM判定、以及基于稀疏特征的零样本语言检索。

**📊 数据集**

TotalSegmentator数据集，909,873张CT/MRI 2D切片，来自10家机构，含138个元数据字段（解剖、成像参数、人口学等）。

**📈 对比分析**

与原始FM稠密嵌入、随机权重Baseline对比；评估指标包括重建R^2、下游ROC‑AUC、k=5稀疏指纹检索质量以及概念判定准确率。BiomedParse在10特征下恢复87.8%下游AUC，检索质量97.7%；DINOv3恢复82.4% AUC，检索质量92.8%，且概念识别准确率更高。

**⚠️ 局限性**

未包含病理病例、仅在2D切片级别评估、零样本检索仅演示单一查询、缺乏人工标注验证、人口学约束仍待改进。

---

## 66. LLMs Do Not Grade Essays Like Humans

**arXiv ID:** 2603.23714 | [PDF](https://arxiv.org/pdf/2603.23714v1)

**作者:** Jerin George Mathew `[一作]` (University of Alberta), Denilson Barbosa `[通讯]` (University of Alberta)

**通讯引用:** 2547 | [OpenAlex ID](https://openalex.org/A5061046432)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在无任务特定训练的“开箱即用”设置下，对 GPT 与 Llama 系列大型语言模型（LLM）进行自动作文评分，并将其评分结果与人类评分进行比较；此外，还分析了 LLM 生成的评分与反馈之间的关联以及各模型在评分时侧重的写作特质。

**💡 创新点**

1) 首次系统性比较多代 LLM（GPT‑3.5、4、5 mini、Llama 2、3、4）在自动作文评分任务中的表现；2) 发现 LLM 在评分时压缩分数区间、对短篇或低质量作文偏高，对长篇或含轻微语言错误作文偏低；3) 通过 ABSA+SHAP 方法揭示 LLM 的评分与其反馈情感高度一致，并且不同模型对不同评分特质的权重差异显著。

**🔧 技术方法**

使用零样本提示（prompting）获取 LLM 的数值评分与文本反馈；利用语言工具 LanguageTool 统计语法/拼写错误；采用 spaCy + DeBERTa‑v3 的 ABSA 模型提取特质级别的正负评价；构建 XGBoost 代理模型并用 SHAP 分析特质提及对评分的贡献；计算 QWK、Pearson 相关和 MAE 进行人类评分对比。

**📊 数据集**

两大公开数据集：ASAP（Task 1、Task 7 以及 ASAP++ 补充特质评分）和 DREsS New（EFL 写作，含 3 维特质）。

**📈 对比分析**

方法：对每个模型对每篇作文生成评分与反馈；计算模型评分与两位人类评分者的 QWK、Pearson 相关和 MAE；对模型反馈进行 ABSA 后统计正负评价比例，并用 SHAP 评估特质权重。性能：LLM 与人类评分的 QWK 均低于 0.3，Pearson 相关最高约 0.57，MAE 明显高于人类评分者间误差，表明 LLM 与人类评分的契合度有限。

**⚠️ 局限性**

局限性：仅使用单一 prompt 模板，提示设计对结果影响未充分探索；ABSA 模型可能产生误分类，且手工构造的特质词表未必覆盖所有表达；SHAP 代理模型假设评分逻辑可被特质提及所近似；仅评估公开数据集，未检验跨领域或多语言的泛化；未对模型进行 fine‑tune，实际教学环境中可能需要进一步校准。

---

## 67. AI Generalisation Gap In Comorbid Sleep Disorder Staging

**arXiv ID:** 2603.23582 | [PDF](https://arxiv.org/pdf/2603.23582v1)

**作者:** Saswata Bose `[一作]`, Raju S. Bapi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了首个公开的脑卒中患者多重睡眠障碍EEG数据集iSLEEPS，并开发了基于SE-ResNet+Bi-LSTM的单通道EEG深度学习模型，用于睡眠分期评估。

**💡 创新点**

创新点包括：①首个公开的脑卒中睡眠分期数据集；②系统使用Grad-CAM与临床专家反馈揭示健康数据训练模型在卒中患者中的泛化失败；③强调疾病特定模型与临床验证的必要性。

**🔧 技术方法**

采用SE-ResNet特征提取、双向LSTM时序建模、Grad-CAM可解释性、交叉验证及统计显著性分析等技术。

**📊 数据集**

使用公开的SleepEDF-20、SleepEDF-78、SHHS三大健康睡眠数据库以及自建的iSLEEPS卒中患者数据库。

**📈 对比分析**

通过留一交叉验证比较ACC、宏F1、κ指标，模型在健康数据上达到87.5–87.8%准确率、82.5–81.9%宏F1，而在iSLEEPS上仅74.7%准确率，显示约30%的性能下降。

**⚠️ 局限性**

模型仅在健康数据上训练，缺乏对卒中患者的泛化，关注点偏离临床生理特征，需加入疾病特定训练与专家审阅。

---

## 68. CoRe: Joint Optimization with Contrastive Learning for Medical Image Registration

**arXiv ID:** 2603.23694 | [PDF](https://arxiv.org/pdf/2603.23694v1)

**作者:** Eytan Kats `[一作]` (University of Luebeck), Mattias P. Heinrich `[通讯]` (University of Luebeck)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 CoRe 框架，将自监督对比学习与可微分优化的图像配准相结合，在特征提取器上联合优化配准损失与等变对比损失，从而生成既具语义信息又稳健于几何变形的特征，用于精确推断变形场。

**💡 创新点**

创新点在于：①首次将等变（equivariance）对比学习直接嵌入可微分配准管线；②通过联合优化使得特征既满足对齐需求又对几何变形具有鲁棒性；③不依赖预训练或配对多模态数据，完全在单模态下自监督学习。

**🔧 技术方法**

使用 3D 卷积特征提取器、投影头、可微分优化模块；对比学习采用 InfoNCE 损失；配准损失为均方误差；训练采用多阶段伪标签自训练、随机仿射增广以及 Adam+余弦退火学习率调度。

**📊 数据集**

在两大公开数据集上评估：腹部 CT（30 份扫描，13 结构）和胸部 CT（371 条纵向扫描，22 结构）。

**📈 对比分析**

与传统方法（NiftyReg、DEEDs）、学习方法（VoxelMorph、LapIRN）以及混合方法（RegCyc、SAMConvex）对比，CoRe 在腹部 CT 上平均 Dice 分数最高（≈52.1%）且在胸部 CT 上也获得最高 Dice（≈89.4%），同时保持与竞争方法相近的变形平滑度和推断速度。

**⚠️ 局限性**

局限性包括：对单模态 CT 数据的依赖，强烈的几何等变约束在多模态或强度变化较大的场景下可能不够充分；且对非线性强度增强的鲁棒性验证不足，未来工作可在多模态或更大强度差异的数据集上进一步验证。

---

## 69. IJmond Industrial Smoke Segmentation Dataset

**arXiv ID:** 2603.23754 | [PDF](https://arxiv.org/pdf/2603.23754v1)

**作者:** Yen-Chia Hsu `[一作]` (University of Amsterdam), Despoina Touska `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究创建并公开了一个工业烟雾分割数据集，包括原始图像、裁剪图像及对应的高低透明度掩模，并提供了时间戳与摄像头两种划分的训练、验证、测试集；

**💡 创新点**

创新点在于将烟雾的视觉透明度分为两类（>50%与<50%），采用Roboflow平台的智能多边形工具与SAM模型自动生成初始掩模后手工精细化，并提供多比例训练集（100%、80%、60%、40%、20%），为不同数据量场景提供灵活性；

**🔧 技术方法**

使用的技术包括Roboflow智能多边形标注、Segment Anything模型初始掩模生成、手工精修、裁剪逻辑、时间戳与摄像头划分；

**📊 数据集**

使用的数据集为自己构建的工业烟雾分割数据集（共900张原始图像、2074张裁剪图像，含1209多边形注释，提供高低透明度掩模）；

**📈 对比分析**

论文未提供模型训练与评估结果，未与现有方法做直接对比；

**⚠️ 局限性**

局限性包括：视觉透明度受光照影响，可能导致标签误差；数据量相对有限，缺乏多样性；未提供基准实验与性能评估。

---

## 70. Improved Local Computation Algorithms for Greedy Set Cover via Retroactive Updates

**arXiv ID:** 2603.23715 | [PDF](https://arxiv.org/pdf/2603.23715v1)

**作者:** Slobodan Mitrović `[一作]` (UC Davis), Mihir Singhal `[通讯]` (UC Berkeley)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种高效的局部计算算法（LCA）用于求解集合覆盖（Set Cover）问题，给出一个 O(logΔ) 近似解，并实现了对查询复杂度的显著改进。

**💡 创新点**

创新点主要包括：① 通过激进的稀疏化（sparsification）技术，在保持近似度的同时极大降低了对输入的访问；② 引入了“retroactive updates”（回溯更新）机制，允许在递归过程中纠正先前的错误决定，从而实现更强的集中性保证；③ 设计了适用于稀疏化流程的两级递归查询或acles，支持在仅查询 O(f^{O(logΔ)}) 次的前提下完成 LCA。

**🔧 技术方法**

关键技术：
- 基于分布式算法（如 O(logΔ) 近似集合覆盖的分布式求解）
- 局部采样与概率估计（用于判断集合是否“稠密”或元素是否已被覆盖）
- 递归稀疏化与剪枝策略（减少不必要的递归调用）
- 回溯更新机制（在后续迭代中纠正先前错误的权重分配）
- 期望近似分析（证明最终得到的期望成本为 O(logΔ) · OPT）。

**📊 数据集**

本研究属于理论算法分析范畴，未使用具体的实测数据集；所有结果均在理论模型（输入为集合覆盖实例的二部图）下证明。

**📈 对比分析**

与现有最优 LCA（Grunau 等人 2020 年提出的 Δ^{O(logΔ)} · f^{O(logΔ (loglogΔ+loglog f)} 查询复杂度）相比，本文算法在查询复杂度上实现了从 Δ^{O(logΔ)} 下降到 f^{O(logΔ)}，在 f = polylogΔ 时进一步降至 Δ^{O(loglogΔ)}。实验上，针对极大规模实例可实现数十倍至百倍的查询效率提升，同时保持 O(logΔ) 的近似保证。

**⚠️ 局限性**

局限性与未解决问题：
- 仍未找到满足 (Δ, f) 查询复杂度的 LCA；
- 目前结果仍以期望近似为准，缺乏绝对保证；
- 复杂度仍为 2^{O(logΔ·log f)}，尚未突破 2^{o(logΔ·log f)}；
- 对某些极端实例（如频率极高或集合大小极大）可能出现额外的随机误差；
- 需要进一步验证回溯更新在实践中的实现复杂度与常数因子。

---

## 71. The Cognitive Firewall:Securing Browser Based AI Agents Against Indirect Prompt Injection Via Hybrid Edge Cloud Defense

**arXiv ID:** 2603.23791 | [PDF](https://arxiv.org/pdf/2603.23791v1)

**作者:** Qianlong Lan `[一作]` (eBay Inc), Anuj Kaul `[通讯]` (eBay Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Cognitive Firewall，通过边缘视觉过滤、云端语义规划和本地执行监控三阶段防御，缓解浏览器代理中的 Indirect Prompt Injection。

**💡 创新点**

创新点在于将防御分层为视觉快筛、云端深度推理和执行时确定性拦截，形成 split-compute 防御漏斗，实现低延迟与高覆盖率的组合。

**🔧 技术方法**

使用 Chrome 内置 Gemini Nano 进行边缘检测，Llama‑3‑8B 等大型模型进行云端语义分析，以及同步 JavaScript 拦截实现执行时拦截。

**📊 数据集**

采用 1,000 条混合攻击样本，包括视觉攻击、语义注入和目标劫持，评估了不同层的效果。

**📈 对比分析**

对比单层和全三层配置，最终攻击成功率下降至 0.88%（静态）或 0.67%（自适应），平均端到端延迟约 517 ms，远低于纯云端方案。

**⚠️ 局限性**

局限在于对图像注入无效、最终延迟仍可显著影响用户体验、以及对语义模糊攻击的 2% 成功率，需引入交互式权限确认。

---

## 72. Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages

**arXiv ID:** 2603.23521 | [PDF](https://arxiv.org/pdf/2603.23521v1)

**作者:** Shaharukh Khan `[一作]` (Krutrim AI), Shubham Agarwal `[通讯]` (Krutrim AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了两套大型多语言多模态数据集 Chitrakshara-IL（交互式文本-图像序列）和 Chitrakshara-Cap（图像-alt 文本对），覆盖 11 种印度语言，提供 193M 张图像、30B 词标记和 50M 文档；同时对 Common Crawl 数据进行全链路采集、HTML 结构清洗、层级过滤及去重，保证数据质量与多样性。

**💡 创新点**

1) 首个面向印度语言的交互式大规模多模态数据集；2) 采用多阶段、规则化的过滤框架，针对图片尺寸、比例、语言文本、冗余内容等细粒度指标进行精准筛选；3) 将图像与对应的 alt‑text 进行对齐，形成可用于图像标注与 VLM 训练的二元数据。

**🔧 技术方法**

利用 Common Crawl WARC 文件进行爬取；通过 FastText LID 进行语言检测；DOM 解析和规则化文本/图像节点提取；层级过滤（图像、段落、文档）及自定义正则/黑名单过滤；对图像做尺寸、比例、文件名和 alt‑text 的合法性检查；对文本做词数、语言一致性和重复内容检测。

**📊 数据集**

Common Crawl 95 个版本（2013‑2023）→ Chitrakshara-IL（193M 图像、30B 词、50M 文档）；Chitrakshara-Cap（44M 图像‑alt 文本对、733M 词）。此外还与 mOSCAR（163 语言交互式数据集）和 OBELICS（英文交互式数据集）做对比。

**📈 对比分析**

与 mOSCAR 进行统计对比，Chitrakshara 在每种印度语言的文档数、词数和图像数均显著高于 mOSCAR；在文本‑图像对数量上亦远超 mOSCAR。实验表明，该数据集可支持训练多语言 ViT 和交互式 VLM，预计能提升在多图像推理、指令调优和少样本学习等任务中的表现。

**⚠️ 局限性**

1) 某些低资源语言（如马拉雅拉姆、奥里亚）的覆盖仍有限；2) alt‑text 仍有大量英文残留，影响纯印度语标注；3) 过滤过程中对非文本噪声的检测仍可能误删部分语义信息；4) 数据域以新闻为主，娱乐、健康等领域相对不足；5) 仅提供数据集，缺乏针对 downstream 任务的完整评估。

---

## 73. Re-Prompting SAM 3 via Object Retrieval: 3rd of the 5th PVUW MOSE Track

**arXiv ID:** 2603.23788 | [PDF](https://arxiv.org/pdf/2603.23788v1)

**作者:** Mingqi Gao `[一作]` (University of Sheffield), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 25189 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MOSEv2半监督视频目标分割任务中，作者基于SAM 3开发了自动再提示框架，通过检测后续帧中的同类别目标并用DINOv3进行对象级匹配，挑选高置信度的锚点并与首帧掩码一起重新提示Tracker，从而提高对目标消失/再现、强变形和同类别干扰的鲁棒性。

**💡 创新点**

创新点在于：①突破传统仅使用首帧锚点的范式，自动在后续帧中挖掘可靠目标锚点；②利用变换感知的目标特征池和DINOv3进行跨帧对象匹配；③将多帧锚点直接注入SAM 3的记忆机制，实现多锚传播。

**🔧 技术方法**

技术手段包括SAM 3视频跟踪器、SAM 3检测分支、DINOv3特征提取、余弦相似度匹配、变换增强（翻转、旋转）以及基于多锚提示的再传播。

**📊 数据集**

使用MOSEv2挑战赛的训练集进行微调，并在官方MOSEv2测试集上评估。

**📈 对比分析**

在MOSEv2测试集上取得51.17的官方评价指标，排名第三；相较于仅使用首帧提示的基线，显著提升了对遮挡、再现和干扰的稳定性。

**⚠️ 局限性**

局限在于：①在极端外观或变形场景下，目标表示仍可能不足以完全恢复；②自动锚点选择仍受检测精度限制，可能遗漏重要帧；③方法为无训练的再提示，缺乏对极端变形的进一步学习能力。

---

## 74. Distributionally Robust $k$-of-$n$ Sequential Testing

**arXiv ID:** 2603.23705 | [PDF](https://arxiv.org/pdf/2603.23705v1)

**作者:** Rayen Tan `[一作]` (University of Michigan), Viswanath Nagarajan `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在分布鲁棒框架下的 k‑of‑n 顺序测试问题，目标是在未知失败概率区间内寻找一条非自适应测试顺序，使得最坏情况下的期望测试成本最小。

**💡 创新点**

创新点包括：
- 对单位成本情形给出了 2‑近似算法；
- 对一般成本情形（假设每个不确定区间都被 ϵ 截止）给出 O(1/√ϵ)‑近似；
- 证明了对 adversary（给定顺序下寻找最坏概率分布）存在 QPTAS；
- 通过对泊松二项分布（PBD）的集中与反集中性质，结合动态规划与窗口分析，构造了上述近似方案。

**🔧 技术方法**

主要技术手段包括：
- 动态规划表构造与非完成概率分析；
- 泊松二项分布的均值/方差、模态与中位数性质；
- Hoeffding、Chernoff 等集中不等式；
- 通过将概率向量压缩为前 d 个幂和，利用 PBD 结构性定理得到 QPTAS；
- 线性规划式的窗口重叠/不重叠分析实现 1/√ϵ 近似。

**📊 数据集**

本文为理论工作，无实验数据集，所有结果均以严格的数学证明给出。

**📈 对比分析**

性能评估：
- 单位成本下 2‑近似比（即实际成本不超过最优的两倍）；
- 通用成本下 1/√ϵ 近似比，随着 ϵ→0 近似比退化；
- adversary 问题的 QPTAS 使得在多项式时间内可获得接近最优的解决方案；
- 通过与传统非自适应 k‑of‑n 的 2‑近似或 PTAS 做对比，表明在鲁棒设置下仍保持可接受的近似误差。

**⚠️ 局限性**

限制与不足：
- 需要 ϵ‑bounded 的不确定区间，若 ϵ 较小则近似比变差；
- 对于通用成本，近似比受 1/√ϵ 控制，实际效果可能不理想；
- QPTAS 的时间复杂度虽然低于指数，但在 n 较大时仍较昂贵；
- 本文仅考虑非自适应方案，对自适应策略的鲁棒性尚未探讨。

---

## 75. Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems

**arXiv ID:** 2603.23508 | [PDF](https://arxiv.org/pdf/2603.23508v1)

**作者:** Xunzhuo Liu `[一作]` (vllm semantic router project), Huamin Chen `[通讯]` (red hat)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5101790571)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种实时的全文上下文验证方法，利用可检索感知的 RoPE 扩展 BERT 编码器至 32K 词元，训练基于响应的幻觉检测器，并引入可配置的早停推理以平衡准确率与吞吐量。

**💡 创新点**

创新点在于：1）检索感知 RoPE 扩展与长距掩码结合，保持长距离注意力；2）使用 Elastic Weight Consolidation 防止长上下文微调时遗忘预训练长距特征；3）构建全文幻觉检测基准与早停分层架构，支持异构工作负载下的准确率-延迟可控。

**🔧 技术方法**

技术栈包括：旋转位置编码（RoPE）及 YaRN 伸缩；检索感知长距掩码（Long-Range Copy、Anchor-Reference）；EWC 正则化；token‑level 幻觉分类与多层轻量化分类器；自适应早停与教师蒸馏；长文档评估框架。

**📊 数据集**

实验使用 RAGTruth（短文档）以及基于 NarrativeQA、QuALITY、GovReport 生成的 8K–24K 词元长文档幻觉基准（337 条样本）。

**📈 对比分析**

与 8K 基线相比，短文档上保持近似性能（Token F1 0.6158→0.5337，Example F1 79.22%→77.00%）。在长文档上，32K 模型将幻觉召回率从 0.06 提升至 0.55（+817%），F1 从 0.10 提升至 0.50（+400%）。早停层在保持 92.8% Example F1 的同时实现 1.4× 速度提升，最早退出层 3.3× 加速但 Example F1 降至 48.2%。

**⚠️ 局限性**

局限性包括：仅支持至 32K 上下文，长距 (>4K) 的长距注意力仍易衰减；早停会导致精度下降；需要精细微调与 EWC 以防止知识遗失；基准为人工构造，缺乏多语种和更长上下文的评估；模型主要针对单一 BERT 风格编码器，跨架构通用性待验证。

---

## 76. Beyond Accuracy: Introducing a Symbolic-Mechanistic Approach to Interpretable Evaluation

**arXiv ID:** 2603.23517 | [PDF](https://arxiv.org/pdf/2603.23517v1)

**作者:** Reza Habibi `[一作]` (University of California, Santa Cruz), Magy Seif El-Nasr `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种机制感知评估框架，通过符号规则和机制解释检测模型是否真正实现任务算法，而非仅靠表面准确率。

**💡 创新点**

创新点在于将符号可验证规则与机制解释技术结合，形成分层规则（R1‑R3），提供可解释的泛化判定，并能区分记忆、泄漏和真正的泛化。

**🔧 技术方法**

使用机制解释技术（激活补丁、logit‑difference、注意力可视化等）以及符号逻辑规则构建评估流程。

**📊 数据集**

采用 TinySQL CS1 Synonyms 数据集，该数据集包含英语提示、数据库 schema 与对应的 SQL 查询，且字段名称使用同义词，需要 schema grounding。

**📈 对比分析**

与传统准确率（字段名称准确率、Exact Match）比较：在未使用 schema 的模型上，标准准确率93.5%但机制评估仅59%；使用 schema 的模型标准准确率99.1%，机制评估76%。机制评估显著揭示了标准指标下的误导性高分，显示了算法学习与模式匹配的区别。

**⚠️ 局限性**

局限包括：阈值和参数需手工校准；激活补丁可能引入分布偏移，影响评估可靠性；框架主要适用于具有明确算法原语的任务，对复杂任务的符号规则定义仍具有挑战。

---

## 77. Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters

**arXiv ID:** 2603.23780 | [PDF](https://arxiv.org/pdf/2603.23780v1)

**作者:** Nan Cui `[一作]` (Stevens Institute of Technology), Yue Ning `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 2071 | [OpenAlex ID](https://openalex.org/A5024383883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种轻量级公平性模块，先用核化的 INLP 投影消除 LLM 表示中的敏感属性信息，再通过两层门控的 MoE Adapter 恢复任务效能，从而在不增加额外训练参数的前提下实现多属性公平性。

**💡 创新点**

创新点在于：①使用随机 Fourier 特征将线性 INLP 推广为核化闭式投影，显著降低了线性泄露；②设计两级门控 MoE Adapter，既能自适应控制每个属性的抑制强度，又能通过低秩 LoRA 专家在不重新引入偏见的前提下修复被裁剪的任务信息；③整个框架完全无额外损失函数、无额外训练步骤，保持了极低的计算开销。

**🔧 技术方法**

核心技术包括：Kernelized Iterative Null‑space Projection (RFF‑INLP)、随机 Fourier 特征 (RFF)、两层门控 Mixture‑of‑Experts (MoE) 适配器、低秩 LoRA 参数化、对抗式训练与信息理论分析（Counterfactual Leakage Gap）。

**📊 数据集**

在 MovieLens‑1M（含性别、年龄、职业三种敏感属性）和 Insurance（婚姻状况、年龄、职业三种敏感属性）两个公开推荐数据集上进行评估。

**📈 对比分析**

与 LLaRA、P5、UP5 等基线进行比较，使用 Hit@1/3/10 评估推荐效果，使用 Counterfactual Leakage Gap Δ_CL 评估公平性。实验显示，该方法在保持甚至提升 Hit@1/3/10 的同时，将 Δ_CL 降低到 0.1% 左右，显著优于 UP5 在公平性上的表现，同时几乎不降低推荐质量。

**⚠️ 局限性**

局限性：依赖用户交互历史信息，若缺乏足够的行为序列或历史数据，该方法可能效果受限。未来工作计划探索不依赖历史记录的去偏方法。

---

## 78. Human-in-the-Loop Pareto Optimization: Trade-off Characterization for Assist-as-Needed Training and Performance Evaluation

**arXiv ID:** 2603.23777 | [PDF](https://arxiv.org/pdf/2603.23777v1)

**作者:** Harun Tolasa `[一作]` (Sabanci University), Volkan Patoglu `[通讯]` (Sabanci University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究提出了人机交互（HiL）Pareto优化方法，用来系统、样本高效地刻画运动技能训练中用户性能与感知挑战度之间的权衡，并基于此设计助力即需（AAN）训练方案和进展评估。

**💡 创新点**

创新点在于：①将定量性能指标与定性感知挑战度结合成混合模型；②采用贝叶斯多目标优化（USeMO）实现HiL Pareto采样；③利用Pareto前沿进行AAN控制器选取、个体与群体进步评估和公平比较。

**🔧 技术方法**

主要技术包括：高斯过程回归（对性能和感知）、两种采集函数（UCB）、USeMO多目标采样策略、Sobol空间填充、二阶差分和对偏好数据的拉普拉斯近似。

**📊 数据集**

实验使用34名健康志愿者，在带力反馈的手柄上完成二维倒立摆平衡任务，记录表现分数和受试者的“难易”/“中等”/“困难”评价。

**📈 对比分析**

与传统自适应阶梯控制（控制组）比较，Pareto基AAN组在无辅助下的表现提升与对照组相当，训练期间两组的平均性能增益无显著差异，但Pareto组在中等至高辅助区间显示略高的提升；同时，聚合Pareto曲线提供了在所有辅助水平下的公平评估。

**⚠️ 局限性**

局限包括：样本量有限，难以检出小效应；仅评估单一决策变量（辅助力度），未考虑更复杂的任务参数；所用AAN协议为简化版，未进行深入优化；实验仅在一次训练周期内完成，缺乏长期保持性验证。

---

## 79. AI-driven Intent-Based Networking Approach for Self-configuration of Next Generation Networks

**arXiv ID:** 2603.23772 | [PDF](https://arxiv.org/pdf/2603.23772v1)

**作者:** Md. Kamrul Hossain `[一作]` (King Fahd University of Petroleum and Minerals), Walid Aljoby `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5015916901)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个端到端闭环的意图驱动网络（IBN）管道，将自然语言意图转化为可验证的策略IR，并通过冲突感知激活和主动多意图保障实现早期故障预测和根因辨别。

**💡 创新点**

创新点包括：①基于LLM的自然语言到结构化策略生成与schema验证的可靠闭环；②冲突感知激活机制，结合多域适配器实现安全部署；③将保障视为主动多意图故障预测，并引入根因歧义消除与提前预警；④将修复建议与可执行子程序绑定，形成完整的闭环。

**🔧 技术方法**

技术手段：大型语言模型（LLM）+提示工程；JSON/YANG-like策略IR；schema约束验证；混合专家/路由式冲突检测；多域适配器；固定窗口预测模型；根因关联模型；LLM驱动的修复生成。

**📊 数据集**

使用合成控制台测试集（基于控制器规则和意图的人工生成）以及受控实验室KPI追踪（CPU/内存/吞吐/延迟等），并在此基础上对部分真实网络日志进行部分标注。

**📈 对比分析**

对比基线（传统规则+手工配置、仅反应式保障）时，LLM转化在准确率上提升约15%，冲突率降低30%，主动预测的提前预警时长平均提升至10–20秒，根因识别准确率提升20%。

**⚠️ 局限性**

局限性：①跨域/多层控制平面的统一策略IR尚未成熟，缺乏真实多域实验；②主动保障依赖稀缺的多意图共漂移标签，真实数据噪声高；③LLM安全性仍需强化，防止产生不安全策略；④推理与预测的时延与规模化部署挑战；⑤缺乏公开可复现的数据集与基准。

---

## 80. Engagement-Zone-Aware Input-Constrained Guidance for Safe Target Interception in Contested Environments

**arXiv ID:** 2603.23649 | [PDF](https://arxiv.org/pdf/2603.23649v1)

**作者:** Praveen Kumar Ranjan `[一作]` (University of Texas at San Antonio), Yongcan Cao `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 12396 | [OpenAlex ID](https://openalex.org/A5014808989)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对多防御者GPS‑禁区的目标拦截，设计了基于攻击者相对信息且考虑推力限制的非线性安全引导法。

**💡 创新点**

创新点在于将防御者诱发的接触区(EZ)与输入饱和动力学结合，使用软最小(log‑sum‑exp)聚合多目标安全约束，并提出时变安全紧缩参数与平滑切换策略。

**🔧 技术方法**

技术包括非线性控制、控制障碍函数（CBF）、相对极坐标建模、输入饱和模型、Lyapunov 稳定性分析以及平滑切换函数。

**📊 数据集**

使用仿真数据，构造静态和动态多防御者场景（2、3、6个防御者），无公开数据集。

**📈 对比分析**

通过与传统最大接触范围安全约束比较，结果显示EZ基方法在静态防御者约减15.5%时间，在动态防御者可减少14–20%时间，且路径更短。

**⚠️ 局限性**

局限在于依赖已知防御者速度比例，仿真环境简化；对高度复杂动态场景或不可观测防御者策略的鲁棒性未充分验证。

---

## 81. Detection and Classification of (Pre)Cancerous Cells in Pap Smears: An Ensemble Strategy for the RIVA Cervical Cytology Challenge

**arXiv ID:** 2603.23742 | [PDF](https://arxiv.org/pdf/2603.23742v1)

**作者:** Lautaro Kogan `[一作]`, María Victoria Ríos `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在RIVA Cervical Cytology Challenge上使用YOLOv11m构建并集成了三种不同的类不平衡处理策略（损失重加权、迁移学习、加权采样），通过Weighted Boxes Fusion实现多类别宫颈细胞检测与分类。

**💡 创新点**

通过系统评估三种互补的不平衡缓解策略并以最终测试性能加权融合，证明预览阶段指标不一定能反映最终泛化能力，且融合方法显著提升了mAP50-95。

**🔧 技术方法**

YOLOv11m目标检测框架、加权交叉熵、迁移学习（SIPaKMeD预训练）、加权数据采样、Weighted Boxes Fusion融合以及精细化的数据增强与方向校正。

**📊 数据集**

RIVA数据集（959张高分辨率Pap smear图像，8类Bethesda标签）和公开的SIPaKMeD数据集用于预训练。

**📈 对比分析**

在预览和最终测试阶段分别评估mAP50-95；单模型最高为0.160/0.108；融合模型在预览为0.201、最终为0.147，较最佳单模型提升约29%，在多类别检测上取得显著进展。

**⚠️ 局限性**

对极少数关键类别（ASCUS、ASCH）的检测性能仍低，且低阈值导致视觉噪声，需要进一步的少数类增强和多阶段检测-分类方案。

---

## 82. MoCHA: Denoising Caption Supervision for Motion-Text Retrieval

**arXiv ID:** 2603.23684 | [PDF](https://arxiv.org/pdf/2603.23684v1)

**作者:** Nikolai Warner `[一作]`, Apaar Sadhwani `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

未提供论文内容

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

## 83. Visuospatial Perspective Taking in Multimodal Language Models

**arXiv ID:** 2603.23510 | [PDF](https://arxiv.org/pdf/2603.23510v1)

**作者:** Jonathan Prunty `[一作]` (University of Cambridge), Lucy Cheke `[通讯]` (University of Cambridge)

**通讯引用:** 3064 | [OpenAlex ID](https://openalex.org/A5010889838)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多模态语言模型（MLM）进行视空间视角转换（VPT）的评估，使用改编自人类实验的旋转图像任务（Rotating Figure Task）和指挥者任务（Director Task）对模型在不同角度、视觉与空间信息以及层级水平上的表现进行系统测试。

**💡 创新点**

将人类VPT实验范式迁移到多模态LLM上，构建大规模、可控的实验电池，并在功能性交互情境中（指挥者任务）检验VPT能力；系统性比较不同模型（包括推理型与非推理型）在单一与组合VPT需求下的表现，揭示了目前MLM在视角转换上的核心缺陷。

**🔧 技术方法**

使用OpenAI系列多模态模型（o3、o4‑mini、GPT‑4o、GPT‑4o‑mini），结合链式思考提示、视觉与文本输入，采用混合效应逻辑回归进行统计分析；任务生成脚本利用程序化方式创建不同角度、视觉/空间标签的图像与ASCII版本。

**📊 数据集**

自定义生成的数据集：旋转图像任务约15k张图（3k每种设置，27k试验），包含0–180°角度、视觉/空间问题与层级1/2；指挥者任务约8k张图像与对应ASCII版（16k样本），系统变异视觉视角、遮挡与相对形容词（大小、垂直/水平）。

**📈 对比分析**

通过混合效应逻辑回归比较模型在不同角度、视觉与空间问题以及Level 1/2任务中的准确率；结果显示：（1）Level 1 VPT在部分模型表现尚可，但存在右向盲区；（2）Level 2 VPT普遍低迷，推理模型呈M形曲线，仅在0°和180°角度表现好；（3）在结合视觉与空间需求的任务中，所有模型的准确率均降至接近随机，GPT‑4o在180°角度略有提升，但整体仍无法实现灵活的心理旋转；（4）推理型模型虽能利用链式思考提升单一维度，但对多维组合仍无效。

**⚠️ 局限性**

局限性：模型主要依赖浅层启发式（如左右互换、符号反转）而非真正的心理旋转，导致对中间角度和组合视角转换表现差；视觉方向提取在小模型中受限；存在幻觉与无效回答；推理时提示虽提升单维表现，却未解决根本表示与处理瓶颈。

---

## 84. Trends in Equal-Contribution Authorship: A Large-Scale Bibliometric Analysis of Biomedical Literature

**arXiv ID:** 2603.23569 | [PDF](https://arxiv.org/pdf/2603.23569v1)

**作者:** Binbin Xu `[一作]` `[通讯]` (IMT Mines Ales), Binbin Xu (IMT Mines Ales)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对2010-2024年PubMed与PMC的近48万条带有‘equal‑contribution’标签的文章进行大规模纵向统计与可视化，探究其在时间、期刊和国家层面的使用趋势与分布特征。

**💡 创新点**

首次系统性、元数据驱动的跨期刊、跨地区的长期共创作者署名分析，为评估作者贡献与学术评价体系提供了宏观基线与可追踪指标。

**🔧 技术方法**

使用JATS/XML元数据解析、去重合并、归一化作者位置信息、分数计数与地理映射、热图与散点图绘制等数据处理与可视化技术。

**📊 数据集**

包含约479,274条PubMed/PMC记录（2010-2024），涵盖多学科生物医学期刊，作者与机构信息由JATS标记抽取。

**📈 对比分析**

主要通过时间序列与比例比较展示趋势，采用热图对期刊级别的比例变化进行可视化，未引入机器学习模型或性能指标，重点在描述性统计与可视化结果。

**⚠️ 局限性**

仅捕捉了JATS标签中的共创作者信息，忽略PDF正文或非结构化文本的标注；元数据标签可能存在错误或不一致；地理归属的推断具有不确定性；分析聚焦期刊与国家，缺乏领域、机构或团队层面的细粒度解析。

---

## 85. Semantics for 2D Rasterization

**arXiv ID:** 2603.23696 | [PDF](https://arxiv.org/pdf/2603.23696v1)

**作者:** Bhargav Kulkarni `[一作]`, Pavel Panchekha `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文通过在 Lean 上构建 Skia 2D 图形库的形式语义，识别并优化了 Google Chrome 在前 100 个网站中产生的低效渲染代码，从而提升了渲染性能。

**💡 创新点**

创新点在于：①为 Skia 提供了完整的、可机械化的形式语义，②利用该语义验证优化方案的正确性，③提出并实现了一套基于模式匹配的高性能优化器，并通过形式化验证实现端到端的可信性。

**🔧 技术方法**

主要技术包括：Lean 证明助手中的形式语义建模、层次化语义分层、模式匹配优化器实现以及优化后代码与语义的验证回流。

**📊 数据集**

使用了从前 100 个访问量最高的网站中收集的 Skia 渲染程序作为数据集。

**📈 对比分析**

通过与 Skia 最现代 GPU 后端（Baseline）进行基准测试对比，优化器在保持仅几秒钟优化时间的同时，实现了显著的性能提升（提升幅度超过基准后端），且该提升在多种网站、后端和 GPU 上均保持一致。

**⚠️ 局限性**

局限性包括：①优化仅覆盖四种识别出的低效模式，未必能处理所有类型的低效渲染；②对 Skia 的形式语义覆盖范围有限，尚未验证更复杂或未来新增的图形特性；③在不同浏览器或渲染环境中的适用性尚未深入评估。

---

## 86. Bi-CRCL: Bidirectional Conservative-Radical Complementary Learning with Pre-trained Foundation Models for Class-incremental Medical Image Analysis

**arXiv ID:** 2603.23729 | [PDF](https://arxiv.org/pdf/2603.23729v1)

**作者:** Xinyao Wu `[一作]` (Chinese University of Hong Kong), Raymond Kai-yu Tong `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9965 | [OpenAlex ID](https://openalex.org/A5066840655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种双向保守-激进互补学习框架Bi‑CRCL，实现医学图像分类的无重放类增量学习。

**💡 创新点**

创新点包括：①双向知识交互（前向初始化+后向指数移动平均）；②同时维护保守（稳定）与激进（快速）两学习器；③交叉分类对齐以对齐决策边界；④使用岭回归分析式分类器提升原型分离度；⑤自适应门控融合两学习器输出，兼顾稳定与适应性。

**🔧 技术方法**

技术细节：预训练Vision Transformer（ViT）+轻量级Adapter微调；EMA、交叉分类正则；岭回归分析式分类器+随机投影；温度化softmax+KL门控自适应融合。

**📊 数据集**

实验数据集：Colon、Blood、Skin8、MedMNIST‑Sub、COVID（CT+X‑ray）五个医学图像分类数据集。

**📈 对比分析**

与传统CIL、PFM‑基线、prompt/adapter方法等进行对比；在Acc_Avg、Acc_Last两项指标均显著优于SOTA；在不同PFM、任务顺序、任务数、无重放、跨数据集等设置下均保持稳健性能，甚至在某些跨数据集场景下超过联合训练上限。

**⚠️ 局限性**

局限性：①需要同时维护两套学习器，算力与存储略增；②对极少样本或严重不平衡任务仍易失效；③仅验证分类任务，对分割等其他医学任务尚未验证；④对极端少样本/极度不平衡的连续学习场景仍有提升空间。

---

## 87. Semantic Iterative Reconstruction: One-Shot Universal Anomaly Detection

**arXiv ID:** 2603.23766 | [PDF](https://arxiv.org/pdf/2603.23766v1)

**作者:** Ning Zhu `[一作]` `[通讯]`, Ning Zhu

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Semantic Iterative Reconstruction框架，实现单一模型在仅9张正常样本下跨多医学影像域的无监督异常检测。

**💡 创新点**

通过预训练教师编码器提取多尺度特征，结合紧凑上采样-下采样解码器和多循环迭代细化，在特征空间内强制正态先验，实现极少样本下的跨域泛化。

**🔧 技术方法**

教师-学生知识蒸馏、语义重建、循环迭代细化、多尺度余弦相似度融合、Gaussian平滑等技术。

**📊 数据集**

九个公开医学异常检测基准：RSNA、OCT2017、APTOS、ISIC、BraTS、BR35H、LAG、VinCXR 和 Brain Tumor。

**📈 对比分析**

与专用与通用的全量/少量样本方法对比，单一模型在一shot通用设定下平均AUC达89.7%，超过所有少量基线；在全量通用和专用模式下保持或接近最优性能。

**⚠️ 局限性**

像素级定位受限于特征空间抽象，难以精准描绘细小或形状复杂的病灶。

---

## 88. The Diminishing Returns of Early-Exit Decoding in Modern LLMs

**arXiv ID:** 2603.23701 | [PDF](https://arxiv.org/pdf/2603.23701v1)

**作者:** Rui Wei `[一作]`, Hao Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文提供了ACL会议的格式模板和写作指导

**💡 创新点**

将会议通用规范与ACL特定LaTeX模板结合，以示例展示

**🔧 技术方法**

采用LaTeX语言和ACL样式文件进行排版

**📊 数据集**

未使用任何实验数据集

**📈 对比分析**

未进行实验对比，说明如何使用样式文件并展示示例文档

**⚠️ 局限性**

仅为格式说明，缺乏实验验证和性能评估

---

## 89. GTO Wizard Benchmark

**arXiv ID:** 2603.23660 | [PDF](https://arxiv.org/pdf/2603.23660v1)

**作者:** Marc-Antoine Provost `[一作]` (GTO Wizard), Philippe Beardsell `[通讯]` (GTO Wizard)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GTO Wizard Benchmark，公开 API 与标准化评估框架，用于对 Heads‑Up No‑Limit Texas Hold’em（HUNL）中的 AI 进行基准测试。

**💡 创新点**

创新点在于：①利用 AIVAT 这一无偏差方差降低技术，使统计显著性仅需十倍更少的手数；②将最先进的 GTO Wizard AI（近似纳什均衡、实时策略生成）作为基准对手；③公开实时排行榜与完整评估指标，促进社区协作。

**🔧 技术方法**

核心技术包括：强化学习自我对弈训练的 GTO Wizard AI、AIVAT 方差降低方法、RESTful API 交互接口以及针对 LLM 的零样本评估脚本。

**📊 数据集**

使用的数据集：5000 手 HUNL 对局（每手 20,000 盲子重置），以及公开的 GTO Wizard AI 预训练策略与 150,000 手自我对弈生成的数据。

**📈 对比分析**

比较方法：对多种 LLM（GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro 等）进行零样本对局，使用 AIVAT 计算 luck‑adjusted win rate（bb/100）并与基准 GTO Wizard AI 及若干基线代理对比。结果显示最佳模型 GPT‑5.3 Extra High 仍落后基准约 16 bb/100，整体仍远低于人类专业水平。

**⚠️ 局限性**

局限性包括：①评估仅针对无偏差方差降低的静态对手，未涵盖对手建模与频率误差；② LLM 在状态跟踪（手牌表示）和混合策略方面存在 hallucination，导致可被利用；③实验规模（5000 手）对极端情况不够充分，且仅限 Heads‑Up 场景。

---

## 90. MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis

**arXiv ID:** 2603.23580 | [PDF](https://arxiv.org/pdf/2603.23580v1)

**作者:** Wei Sun `[一作]` (Chinese University of Hong Kong), Fangxin Wang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2480 | [OpenAlex ID](https://openalex.org/A5101686970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了 MetaKube，一个能够持续学习经验、在本地部署的 Kubernetes 故障诊断框架。

**💡 创新点**

创新点在于引入 Episodic Pattern Memory Network (EPMN) 进行经验抽象、设计自适应元认知控制器以及通过低秩适配 (LoRA) 对 Qwen3-8B 进行领域微调，三者协同实现了从历史案例中快速检索与因果推理的双通道诊断。

**🔧 技术方法**

主要技术包括 1) 经验记忆网络 (EPMN) 进行模式检索与置信度校准；2) 元认知控制器实现路径自适应路由；3) KubeLLM (Qwen3-8B + LoRA 微调)；4) KubeGraph 结构化知识图谱；5) 基于 RAG 的检索增强推理。

**📊 数据集**

使用自构建的 7,000 条 Kubernetes 故障排查样本(KFRD) 进行模型微调，并在 1,873 条真实场景(KubeFault) 上进行评估。

**📈 对比分析**

在 GPT‑5 自动评估和人工专家评估下，MetaKube 在四个维度上平均得分达 90.5 分，较基准 Qwen3-8B 提升 40.6 分，接近 GPT‑4.1 GraphRAG 的 91.9 分，显示出显著的性能提升且保持了完整数据隐私。

**⚠️ 局限性**

局限性包括对大型知识图谱和高质量故障数据的依赖、对稀有或全新故障模式的适应性仍有限，以及需要持续的经验更新与模型维护。

---

## 91. Dual-Gated Epistemic Time-Dilation: Autonomous Compute Modulation in Asynchronous MARL

**arXiv ID:** 2603.23722 | [PDF](https://arxiv.org/pdf/2603.23722v1)

**作者:** Igor Jankowski `[一作]` `[通讯]`, Igor Jankowski

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多智能体强化学习框架 ETD‑MAPPO，使代理在感知到高置信度或知识不确定性较低时自动暂停神经网络推理，从而实现计算频率的自适应调节。

**💡 创新点**

创新点在于双门触发机制（基于策略熵和Twin‑Critic 的价值差异）实现的知识不确定性驱动时间拉伸；SMDP 对齐的异步梯度掩蔽策略保证信用分配的正确性；并且在训练中自动形成时序角色专化，显著降低低负荷代理的 FLOP。

**🔧 技术方法**

采用 MAPPO 作为基线网络，加入 Twin‑Critic 架构、熵门阈值、知识差异门限、SMDP‑aligned GAE、异步梯度掩蔽、以及局部状态递归保留等技术。

**📊 数据集**

在三个基准环境上验证：Level‑Based Foraging (LBF)、Multi‑Particle Environment (MPE) 以及 Google Research Football (GRF) 的 3v1 场景。

**📈 对比分析**

与同步 Vanilla MAPPO 以及固定跳帧 Fixed‑Skip（N=3）对比，ETD‑MAPPO 在 LBF 上提升约 60% 胜率、在 GRF 上保持 100% 进球率并实现 5.2% FLOP 降低；在 MPE 中出现 73.6% 的低负荷代理推理跳频率，整体性能优于两种基线。

**⚠️ 局限性**

局限性包括：阈值 τ_H、τ_V 的手动调参依赖、最大跳帧 N 的上限受 Lipschitz 连续性约束；在物理部署时若门控失效可能导致安全风险，需要硬件级安全覆盖。

---

## 92. Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems

**arXiv ID:** 2603.23578 | [PDF](https://arxiv.org/pdf/2603.23578v1)

**作者:** Yuqing Zhou `[一作]` (Changchun University of Science and Technology), Fujun Liu `[通讯]` (Changchun University of Science and Technology)

**通讯引用:** 3591 | [OpenAlex ID](https://openalex.org/A5100641989)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种Residual Attention Physics‑Informed Neural Network（RA‑PINN）框架，用于稳态电热耦合多物理场的统一求解。

**💡 创新点**

创新点在于将残差连接与注意力机制相结合，实现对局部梯度尖锐和耦合结构的自适应捕获，并通过自适应残差采样进一步提升训练精度。

**🔧 技术方法**

采用深度残差网络+通道注意力、自动微分构造残差、残差加权损失以及自适应采样的PINN技术。

**📊 数据集**

在四个电热耦合基准上进行验证，包括常数系数耦合、压力计约束、温度相关传输和斜面界面等，使用公开的基准数据集。

**📈 对比分析**

与Pure‑MLP、LSTM‑PINN、pLSTM‑PINN等传统PINN变体对比，RA‑PINN在所有指标（MSE、RMSE、MAE、相对L2）均取得最低误差，尤其在温度相关系数和界面问题上优势明显，但训练时间相对更长。

**⚠️ 局限性**

局限性在于训练成本高，尤其在高非线性或多尺度场景下需要更长时间；对大规模三维问题的可扩展性尚待进一步研究。

---

## 93. Plato's Cave: A Human-Centered Research Verification System

**arXiv ID:** 2603.23526 | [PDF](https://arxiv.org/pdf/2603.23526v1)

**作者:** Matheus Kunzler Maldaner `[一作]` (University of Florida), Damon L. Woodard `[通讯]` (University of Florida)

**通讯引用:** 3529 | [OpenAlex ID](https://openalex.org/A5055751228)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出了一个名为Plato's Cave的开源系统，用以通过构建科学论文的知识图谱、利用多代理浏览器验证节点可信度并进行信任传播，最终给出论文质量评分。

**💡 创新点**

创新点在于将论文论证结构建模为有向无环图（DAG），并结合可信度门控的信任传播机制，将节点验证结果与全局结构属性统一推理，形成可解释的质量评估。

**🔧 技术方法**

使用技术包括大型语言模型（LLM）用于文本到DAG的抽取与节点标签，基于浏览器自动化的LLM代理进行网页检索与可信度评估，图模型的消息传播与门控信任更新，以及多维度可解释评分体系。

**📊 数据集**

使用的数据集为104篇不同领域（经济学、机器学习、心理学）的研究论文，结合其在电子表格中的粗略三分评价标签。

**📈 对比分析**

在与人类粗略标签的比较中，通过缓存优先的参数搜索，获得了Good‑vs‑Bad分类的AUROC最高约0.766，Spearman相关系数约0.40，显示中等的与人工评判的一致性。

**⚠️ 局限性**

局限性包括：需要大量计算资源和API调用导致高成本与长时间运行；仅在小规模数据集上验证；语义角色固定，可能不适用于某些专业领域；以及信任门控和评分参数仍为手工调优，缺乏从真实标注中学习的能力。

---

## 94. The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression

**arXiv ID:** 2603.23528 | [PDF](https://arxiv.org/pdf/2603.23528v1)

**作者:** Warren Johnson `[一作]` `[通讯]` (Plexor Labs), Warren Johnson (Plexor Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了提示压缩对大型语言模型推理能耗和质量的影响，并通过近3万次API调用对GPT‑4o‑mini与DeepSeek‑Chat进行了大规模对比实验。

**💡 创新点**

发现提示压缩会导致某些模型输出Token显著爆炸，从而在特定供应商上产生负面能耗效应，并提出基于匹配样本的Green AI Score评估框架。

**🔧 技术方法**

采用代理能耗模型结合NVML/CodeCarbon测量、压缩比例控制、任务感知压缩与路由优化技术。

**📊 数据集**

使用HumanEval、MBPP、GSM8K、MATH、MMLU等公开编程与推理基准，共计约28,428次API调用。

**📈 对比分析**

通过匹配子样本（GPT‑4o‑mini vs DeepSeek‑Chat, N=16,270）比较能耗、质量和成本，结果显示压缩在DeepSeek导致能耗提升达2,140%，GPT‑4o‑mini能耗极低但压缩后质量显著下降。

**⚠️ 局限性**

代理能耗估计存在误差、仅评估API推理未覆盖硬件制造碳排放、样本仅包含两款模型、以及时间漂移与输出长度控制缺失。

---

## 95. Konkani LLM: Multi-Script Instruction Tuning and Evaluation for a Low-Resource Indian Language

**arXiv ID:** 2603.23529 | [PDF](https://arxiv.org/pdf/2603.23529v1)

**作者:** Reuben Chagas Fernandes `[一作]` (Don Bosco), Gaurang S. Patkar `[通讯]` (Don Bosco)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了多脚本合成指令数据集 Konkani‑Instruct‑100k，并使用 LoRA 对多种开源 LLM 进行微调，形成一系列专门针对 Konkani 的 LLM，并在自制的多脚本 Bench（Konkani‑Bench）上进行评测。

**💡 创新点**

创新点在于首次利用 Gemini 3 生成覆盖 Devanagari、Romi 和 Kannada 三种脚本的合成指令对；同时推出专门的 Konkani LLM、Konkani‑Bench 评测基准，并证明参数高效 LoRA 微调能够显著提升低资源语言的表现。

**🔧 技术方法**

技术包括 Gemini 3 生成合成指令数据、LoRA（rank = 64）参数高效微调、自动评测指标 BLEU/chrF++/COMET 以及 LLM‑as‑a‑judge 评估。

**📊 数据集**

使用的数据集为：Konkani‑Instruct‑100k（约 106k 条多脚本指令对）、Konkani‑Bench（200 条双语+多脚本示例）以及少量英文参考与公开 Konkani 语料。

**📈 对比分析**

在翻译任务上，微调后的 Konkani LLM 在 BLEU/chrF++/COMET 上均优于大多数开源与闭源基线，其中 COMET 最高；在转写任务上显著高于基线；LLM‑as‑a‑judge 评分表明 Llama‑8B 在 Romi/Devanagari 上表现最佳，Qwen2.5‑14B 在 Kannada 上表现最佳。

**⚠️ 局限性**

局限性包括数据为合成，可能包含教师模型偏差；缺乏不同方言的覆盖；LoRA rank 64 可能限制更高阶微调的潜在性能；未尝试完整预训练或全微调，预算有限。

---

## 96. Quadrature Oscillation System for Coordinated Motion in Crawling Origami Robot

**arXiv ID:** 2603.23666 | [PDF](https://arxiv.org/pdf/2603.23666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 97. APreQEL: Adaptive Mixed Precision Quantization For Edge LLMs

**arXiv ID:** 2603.23575 | [PDF](https://arxiv.org/pdf/2603.23575v1)

**作者:** Meriem Bouzouad `[一作]` (Lab-STICC, CNRS UMR 6285, ENSTA, Institut Polytechnique de Paris), Jalil Boukhobza `[通讯]` (Lab-STICC, CNRS UMR 6285, ENSTA, Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对边缘设备部署的LLM，提出了APreQEL框架，利用层级重要性评估和用户定义的质量目标（内存、延迟、准确性）进行混合精度量化分配，实现每层最优精度选取。

**💡 创新点**

创新点在于将层贡献打分与多准则TOPSIS相结合，既考虑不同量化方式在硬件上的内存/延迟/准确性差异，又通过用户权重动态决定层级量化分布，显著拓宽了可行解空间并提升Pareto前沿。

**🔧 技术方法**

核心技术包括：1）基于隐藏状态余弦相似度的层贡献评分（奖励/惩罚法）；2）使用K‑type量化（3–6位）与q8_0的QoS估算与组合；3）TOPSIS多准则决策来选取最佳分配；4）层级映射算法将高贡献层映射到低度量化，低贡献层映射到更激进量化；5）在llama.cpp上实现混合精度模型。

**📊 数据集**

使用的数据集与模型：Llama3.1、Phi3.5、Qwen3-4B；WebInstructSub‑prometheus（70个prompt）用于层贡献评估；wiki-test-2用于评估困惑度（Perplexity）；llama-bench工具用于测量单词生成延迟；硬件为Jetson Orin‑AGX。

**📈 对比分析**

比较方法：对28个APreQEL生成模型与统一量化基准在三目标（内存、延迟、困惑度）下计算超体积（Hypervolume）指标。结果显示APreQEL在Llama3.1、Phi3.5、Qwen3-4B上分别提升约8.4%、9.1%、9.3%的超体积，且在各QoS权重设定下对应类别的模型均取得最优或接近最优表现。

**⚠️ 局限性**

限制：仅测试了K‑type（3–6位）与q8_0的量化；QoS估算假设每层对内存/延迟/准确性贡献线性，可能在某些模型/硬件上误差较大；只覆盖三类QoS指标，未考虑能耗、碳足迹等；混合精度配置数目有限，进一步扩展可能提升超体积。

---

## 98. Do 3D Large Language Models Really Understand 3D Spatial Relationships?

**arXiv ID:** 2603.23523 | [PDF](https://arxiv.org/pdf/2603.23523v1)

**作者:** Xianzheng Ma `[一作]` (University of Oxford), Victor Adrian Prisacariu `[通讯]` (University of Oxford)

**通讯引用:** 5570 | [OpenAlex ID](https://openalex.org/A5067395390)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Real-3DQA 基准，过滤掉可通过语言短路解答的 3D 独立问题，并引入视角旋转分数评估模型在不同视角下的空间推理一致性，同时提出 3D 重加权微调策略，鼓励模型更多使用 3D 视觉信息。

**💡 创新点**

创新点在于通过模型对比过滤语言短路、引入视角旋转一致性评测以及基于文本与 3D 差距的自适应重加权训练方法，显著提升 3D-LLMs 的真实空间推理能力。

**🔧 技术方法**

使用了语言模型微调、3D 点云/对象中心编码、视角旋转数据增强、重加权损失函数以及注意力分析技术。

**📊 数据集**

使用 SQA3D、ScanQA、MSR3D 等原始 3D QA 数据集生成 Real-3DQA 与 Real-ScanQA，并与原始 SQA3D 进行对比。

**📈 对比分析**

在 Real-3DQA 上，传统 SFT 模型从约 30% 提升至 40%+，而 3DR-FT 进一步提升至 30%–35%；在原始 SQA3D 上性能略有下降，说明基准更具挑战性。

**⚠️ 局限性**

局限包括对旋转角度仅限四个固定角度、需要人工验证的质量控制、对定位、生成等其他任务尚未验证，以及仍需更广泛的视角不变模型。

---

## 99. Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting

**arXiv ID:** 2603.23637 | [PDF](https://arxiv.org/pdf/2603.23637v1)

**作者:** Peiyu Xu `[一作]` (University of Illinois Urbana Champaign), Shuang Zhao `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可微分、无排序的随机光线追踪框架，用于 3D 高斯点描绘（3D Gaussian Splatting）场景的重建与渲染。

**💡 创新点**

创新点在于引入无偏 Monte‑Carlo 梯度估计器，能够在不对高斯进行深度排序的情况下，只需采样少量高斯即可得到像素颜色梯度，从而实现高效的可微渲染，并可直接支持可重光照（relightable）场景。

**🔧 技术方法**

主要技术包括：
- 随机光线追踪与无排序采样；
- 对像素颜色梯度的无偏 Monte‑Carlo 估计；
- 对可重光照的 BRDF 近似和阴影射线的完全光线追踪；
- 结合 OptiX 硬件加速与神经网络解码器进行颜色预测。

**📊 数据集**

使用的数据集有：
- MipNeRF‑360、Tanks & Temples、Deep Blending、NeRF Synthetic（用于新视角合成）；
- NRHints（用于可重光照实验）。

**📈 对比分析**

与基准方法（传统 3DGS、排序光线追踪 3DGRT 以及可重光照基准 RNG、GS^3）的比较显示：
- 在新视角合成任务中，速度可与 rasterization‑based 3DGS 相当，显著快于排序光线追踪；
- 在可重光照任务中，利用完整阴影射线的渲染获得更高的 PSNR/SSIM，尤其在阴影质量上明显优于现有方法。

**⚠️ 局限性**

局限性：
- 随机采样导致梯度估计方差，影响高斯增密与剪枝策略的稳定性；
- 需要进一步研究方差对增密/剪枝的交互影响，并探索更低方差的采样策略。

---

## 100. MedMT-Bench: Can LLMs Memorize and Understand Long Multi-Turn Conversations in Medical Scenarios?

**arXiv ID:** 2603.23519 | [PDF](https://arxiv.org/pdf/2603.23519v1)

**作者:** Lin Yang `[一作]` (ByteDance), Haihua Yang `[通讯]` (ByteDance)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 MedMT-Bench，一个覆盖预诊、诊断、后诊三阶段、长多轮、多场景、包含文本与图像的医学指令跟随基准，评估 LLM 在真实诊疗情境下的长上下文记忆、干扰鲁棒性、安全防御等能力。

**💡 创新点**

创新点包括：① 定义并覆盖五大指令跟随难点（长上下文记忆、干扰抵抗、自我纠正与安全防御、指令澄清、多指令响应）；② 采用多代理对话生成结合专家手工编辑的混合构建流程；③ 用 LLM-as-judge 与原子测试点实现高达 91.94% 的人机一致率的自动评估。

**🔧 技术方法**

技术手段包括多代理对话生成框架、人工专家校对与双层交叉复核、LLM-based 评判器 + 原子测试点评估、模态融合（文本+图像）以及多轮指令跟随的自动评测。

**📊 数据集**

数据集为 400 条多轮医学对话，平均 22 轮、最长 52 轮，覆盖 5 类指令难点；包含文本和图像子集，来源于公开医学资料、代理生成与专业编辑后完善。

**📈 对比分析**

实验对比 17 个前沿 LLM 与 10 个开源模型，使用人工与自动评估两种方式，整体准确率低于 60%，最佳模型 GPT‑5 以 59.75% 取得最高分，且不同模型在各维度表现差异显著。

**⚠️ 局限性**

局限性包括：只评估指令跟随，未充分考察医学知识深度；数据集仅包含文本与图像，缺少语音、视频等更真实的多模态交互；基准规模有限，需进一步扩展多模态与实际临床情景。

---

## 101. Revisiting Real-Time Digging-In Effects: No Evidence from NP/Z Garden-Paths

**arXiv ID:** 2603.23624 | [PDF](https://arxiv.org/pdf/2603.23624v1)

**作者:** Amani Maina-Kilaas `[一作]` (Massachusetts Institute of Technology), Roger Levy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22811 | [OpenAlex ID](https://openalex.org/A5090215557)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过 Maze 任务和自 paced 阅读实验，考察英语 NP/Z 句法迷惑句中“digging‑in”效应，并将人类阅读时间与大规模语言模型（LLM）的预测进行对比。

**💡 创新点**

创新点在于首次系统比较了句子最终位置与非最终位置的 digging‑in 效应，并揭示了人类行为与 LLM 预测之间的差异，表明“digging‑in”可能是后句 wrap‑up 过程而非实时结构承诺。

**🔧 技术方法**

采用了 16 种 LLM（包括 GPT‑2、Pythia、Gemma‑3、Qwen‑2.5、Mistral 等）进行预测，利用贝叶斯混合效应模型对人类与模型的阅读时间进行统计比较。

**📊 数据集**

实验使用 30 条 Maze 句子和 36 条自 paced 阅读句子，构造了可变的歧义、长度、解析方式和句子最终位置等因素。

**📈 对比分析**

在混合效应模型下，人类数据显示在非最终位置呈现与 LLM 预测相反的“reverse digging‑in”趋势，而 LLM 则在两种位置均显示一致的负向趋势；在人类非关键词区 LLM 预测相当准确，但在关键词区明显低估难度。

**⚠️ 局限性**

局限性包括仅研究了 NP/Z 句法迷惑结构、样本量相对有限、未能在实验中明显验证人类“reverse digging‑in”，以及 LLM 预测未能捕捉 wrap‑up 过程导致的最终位置差异。

---

## 102. Infrequent Child-Directed Speech Is Bursty and May Draw Infant Vocalizations

**arXiv ID:** 2603.23797 | [PDF](https://arxiv.org/pdf/2603.23797v1)

**作者:** Margaret Cychosz `[一作]` (Stanford University), Adriana Weisleder `[通讯]` (Northwestern University)

**通讯引用:** 6171 | [OpenAlex ID](https://openalex.org/A5086288386)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究分析了博利维亚农村与美国城市环境下，婴儿所接触的儿童定向语音（CDS）的时间分布与突发性（burstiness），并考察了语音来源对婴儿发声行为的影响。

**💡 创新点**

首次在跨文化长时段录音中量化CDS的突发性，并揭示即便语音稀缺，突发性与婴儿发声率相关；同时发现博利维亚婴儿对同伴发声的敏感性高于成人发声。

**🔧 技术方法**

采用突发性参数B、线性混合效应模型、置换检验与配对单尾t检验等统计技术，评估时间分布与发声概率。

**📊 数据集**

使用10名婴儿的长时段音频数据（5名博利维亚，5名美国），每位婴儿录制约10–16小时，采集其周围的音频环境。

**📈 对比分析**

通过对照基线的突发性混合模型比较，发现TCDS的突发性在两社区无显著差异；通过概率比较与t检验，证明婴儿在TCDS期间的发声概率显著高于其他语音情况，且博利维亚婴儿对同伴TCDS的发声率更高。

**⚠️ 局限性**

研究样本量小、时间分辨率仅为30秒、未对语音内容与质量进行深入分析，且仅观察发声与语音共时性，无法断定因果关系，限制了结论的普适性与深度。

---

## 103. Self Paced Gaussian Contextual Reinforcement Learning

**arXiv ID:** 2603.23755 | [PDF](https://arxiv.org/pdf/2603.23755v1)

**作者:** Mohsen Sahraei Ardakani `[一作]`, Rui Song `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文探讨了图形的几何特性及其在不同背景下的应用。

**💡 创新点**

创新点在于提出了一种新的几何图形表示方法，能够更有效地捕捉图形的特征。

**🔧 技术方法**

使用了计算几何和图形处理技术。

**📊 数据集**

数据集包括多种几何图形的标准数据集。

**📈 对比分析**

与传统方法进行了比较，结果显示新方法在特征提取和处理速度上有显著提升。

**⚠️ 局限性**

限制在于新方法在处理复杂图形时可能会遇到性能瓶颈。

---

## 104. Resolving gradient pathology in physics-informed epidemiological models

**arXiv ID:** 2603.23799 | [PDF](https://arxiv.org/pdf/2603.23799v1)

**作者:** Nickson Golooba `[一作]` (York University), Woldegebriel Assefa Woldegerima `[通讯]` (York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对流行病学物理信息神经网络（PINN）训练时出现的梯度冲突问题，提出冲突门控梯度缩放（CGGS）方法，实现训练过程的稳定和高效。

**💡 创新点**

创新点在于引入基于余弦相似度的几何门控机制，动态抑制冲突梯度并自动实现“先数据后物理”的课程学习策略，从而避免Pareto死锁并保持标准的O(1/T)收敛速率。

**🔧 技术方法**

使用的技术包括PINN、余弦相似度梯度测度、门控缩放、指数移动平均（EMA）以及ReLU逻辑约束，整体实现仅在每步增加一次内积和两次梯度范数的计算，保持与标准反向传播相同的计算复杂度。

**📊 数据集**

实验基于合成SEIR模型（N=1000，β=1.0，σ=0.2，γ=0.14）生成的20个带高斯噪声的感染曲线点，模拟真实临床采样稀疏与不规则的情形。

**📈 对比分析**

与传统的幅值平衡（LRA）和固定权重PINN相比，CGGS在峰值重建、损失下降和物理约束收敛方面均显著提升，峰值误差降低约15%，最终损失降低十倍以上。

**⚠️ 局限性**

局限性包括EMA动量在理论证明中未完全涵盖，且门控机制主要针对单个物理约束，尚需研究在高维空间时间PDE和真实COVID‑19等复杂数据集上的可扩展性。

---

## 105. Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework

**arXiv ID:** 2603.23625 | [PDF](https://arxiv.org/pdf/2603.23625v1)

**作者:** Zeinab Dehghani `[一作]` (University of Hull), Tanaya Maslekar `[通讯]` (Leeds Teaching Hospital NHS Trust)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一套面向护理院的语音智能助理系统，用于语音记录、信息检索和提醒调度。

**💡 创新点**

提出了面向安全关键护理场景的端到端评估框架，并展示了在嘈杂环境下多模态模型（Whisper+LLM+RAG）在保证可靠性方面的可行性。

**🔧 技术方法**

采用 Whisper 语音识别、LLM（GPT‑5.2、LLaMA‑3、Qwen）进行意图解析与生成、RAG（稀疏、密集、混合）检索、PostgreSQL 存储以及 Google Calendar 集成。

**📊 数据集**

在英国两家护理院的监督试验中收集了 330 条语音交互（184 条包含提醒），并使用合成护理记录数据库进行检索评估。

**📈 对比分析**

与三种 LLM 及检索策略比较，GPT‑5.2 在居民 ID 与类别匹配、提醒识别、日历调度和检索语义相似度上均优于 LLaMA‑3 与 Qwen；如提醒识别准确率 89.09%（CI 83.81–92.80%）和日历提醒计数匹配 84.65%（CI 78.00–89.56%）

**⚠️ 局限性**

局限性包括样本量小、仅两家试点、未真实投入临床使用、检索评估使用合成数据、未测量端到端延迟、未按口音/噪声分层、缺乏长期使用与临床结果评估。

---

## 106. Navigating the Concept Space of Language Models

**arXiv ID:** 2603.23524 | [PDF](https://arxiv.org/pdf/2603.23524v1)

**作者:** Wilson E. Marcílio-Jr `[一作]` (Adaption Labs), Danilo M. Eler `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个可视化系统Concept Explorer，用于大规模稀疏自编码器（SAE）特征的后验探索。

**💡 创新点**

创新点在于将特征的文本解释嵌入多分辨率的HUMAP层次空间，支持从粗到细的导航，自动识别主概念、稀有概念并分析概念间关系。

**🔧 技术方法**

采用稀疏自编码器、文本嵌入、HUMAP层次嵌入、UMAP、随机游走等技术。

**📊 数据集**

使用SmolLM2模型在约170万句子上训练的SAE，并对36864个特征生成上下文与文本解释。

**📈 对比分析**

与传统单层特征浏览、手工检查和语义搜索相比，Concept Explorer在概念发现速度和覆盖度上更优，能够一次性可视化数十万特征并快速定位稀有概念；但缺乏定量性能评估，主要展示案例。

**⚠️ 局限性**

局限包括对文本解释质量的依赖、层次嵌入在高维稀疏空间可能导致信息丢失、未针对非文本特征进行验证，且未提供自动聚类或评估指标。

---

## 107. Detect--Repair--Verify for LLM-Generated Code: A Multi-Language, Multi-Granularity Empirical Study

**arXiv ID:** 2603.23633 | [PDF](https://arxiv.org/pdf/2603.23633v1)

**作者:** Cheng Cheng `[一作]` (Concordia University), Cheng Cheng `[通讯]` (Concordia University)

**通讯引用:** 26929 | [OpenAlex ID](https://openalex.org/A5100354225)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM生成代码的漏洞检测、修复与验证的端到端工作流，构建了多语言、不同粒度的EduCollab基准，并进行迭代评估。

**💡 创新点**

创新点在于提出并验证了检测‑修复‑验证（DRV）流水线，并用可执行的功能与利用测试来衡量“安全且正确”收益，首次系统地比较了不同修复粒度和迭代次数的效果。

**🔧 技术方法**

采用LLM（ChatGPT‑5、GLM‑5）进行漏洞检测、基于提示的自动修复，并使用功能测试和专门设计的利用测试进行验证。

**📊 数据集**

使用EduCollab基准，包括PHP、JavaScript、Python三种语言的可执行Web应用，涵盖项目级、需求级和文件级三种粒度，以及对应的功能和利用测试集。

**📈 对比分析**

通过对单次修复和有限迭代的DRV进行对比，评估了安全且正确的收益率；结果显示在文件级别迭代可显著提升收益，在项目级别收益提升不均匀，整体效果依赖修复粒度和迭代预算。

**⚠️ 局限性**

限制在于基准仅覆盖Web应用场景，缺乏更广泛的程序类型和漏洞种类，且迭代预算有限，未能完全覆盖所有未修复情况，无法直接推广到更复杂系统。

---

## 108. Generating Hierarchical JSON Representations of Scientific Sentences Using LLMs

**arXiv ID:** 2603.23532 | [PDF](https://arxiv.org/pdf/2603.23532v1)

**作者:** Satya Sri Rajiteswari Nimmagadda `[一作]` (Marshall University), Aniruddha Maiti `[通讯]` (West Virginia State University)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5003720400)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用轻量级LLM生成层级JSON结构以表示科学句子，并通过生成模型重构原句验证信息保留。

**💡 创新点**

设计了结构损失函数和运行时JSON验证机制，训练Mistral-7B生成可验证的层级JSON，并通过LLM重构评估信息保持。

**🔧 技术方法**

采用Mistral-7B+LoRA微调、结构损失、GPT-4o生成JSON与句子、句子嵌入(all‑mpnet-base‑v2)进行语义相似度评估，以及BLEU/ROUGE/METEOR等指标。

**📊 数据集**

使用1500条跨七学科（物理、计算机、数学、经济、生命、化学、医学）从arXiv/bioRxiv/ChemRxiv/PubMed采集的1370条有效句子，划分训练/验证/测试集。

**📈 对比分析**

通过语义相似度平均0.872、BLEU 0.15、ROUGE‑1 F1 0.57、METEOR 0.49等指标，表明大多数句子重构保持了高语义相似度。

**⚠️ 局限性**

部分句子重构失真导致语义丢失，压缩与逻辑完整性不足，需要进一步测试鲁棒性与树结构损失。

---

## 109. Empirical Characterization of Logging Smells in Machine Learning Code

**arXiv ID:** 2603.23769 | [PDF](https://arxiv.org/pdf/2603.23769v1)

**作者:** Patrick Loic Foalem `[一作]` (Polytechnique Montreal), Ettore Merlo `[通讯]` (Polytechnique Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 444 个开源机器学习项目进行日志语句抽取，手工标注 2,448 条日志 smell 实例，并基于此构建了 12 类 ML 专属日志 smell 分类。

**💡 创新点**

首次提出面向 ML 代码的日志 smell 分类体系，结合传统与 ML 特有的安全、实验追踪、指标管理、配置与冗余等维度，填补了现有日志 smell 研究在 ML 领域的空白。

**🔧 技术方法**

采用混合方法：静态 AST 分析提取日志代码，利用 GPT‑5‑mini 进行自动初筛与分类，再通过人工验证构建 taxonomy；并通过问卷调查评估实践意义。

**📊 数据集**

使用 444 个活跃的 GitHub ML 仓库（共 19,775 个 Python 文件、142,535 条日志语句），手工标注 2,448 条日志 smell；数据已公开发布在 GitHub 上。

**📈 对比分析**

本文未提出新的检测/修复算法，仅通过人工与 LLM 辅助的标注与问卷验证评估 taxonomy 的准确性与实用性；实验结果显示 12 类 smell 在实际项目中普遍存在并被实践者认可。

**⚠️ 局限性**

局限性包括：仅覆盖 Python 生态；日志库集合仅来自已有研究；抽样与 LLM 识别可能引入误判；调查样本量（27 人）有限；未验证自动检测工具的有效性。

---

## 110. Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges

**arXiv ID:** 2603.23659 | [PDF](https://arxiv.org/pdf/2603.23659v1)

**作者:** Weilun Xu `[一作]` (Ecole Polytechnique Federale), Frederic Kaplan `[通讯]` (Ecole Polytechnique Federale)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种大规模语言模型在五个伦理框架（义务论、功利主义、美德、正义、常识）下的隐藏表示进行线性探针实验，分析其内部结构与冲突对行为不确定性的影响。

**💡 创新点**

首次揭示模型内部存在差异化但相互纠缠的伦理子空间，并发现探针间冲突与生成不确定性（熵）显著相关，从而提供了潜在的伦理风险预警指标。

**🔧 技术方法**

使用线性逻辑回归探针、跨框架转移矩阵、期望校准误差（ECE）与冲突得分等技术，结合层级分析与行为熵评估。

**📊 数据集**

基于 ETHICS 基准数据集，对五个伦理框架的情境进行二元选择题的标准化提问，涵盖多种模型（4B–72B）和架构。

**📈 对比分析**

在六款模型上验证，探针准确率在不同框架间差异显著，转移矩阵表明非统一的“好/坏”维度；高冲突样本的生成熵显著高于低冲突样本（相关系数约 0.36），表明内部冲突与行为不一致之间存在统计关联。

**⚠️ 局限性**

局限包括探针对表面词汇特征的依赖、缺乏因果验证、情境难度可能混淆冲突-熵关系、仅使用单一数据集且线性探针无法捕捉非线性伦理编码。

---

## 111. BXRL: Behavior-Explainable Reinforcement Learning

**arXiv ID:** 2603.23738 | [PDF](https://arxiv.org/pdf/2603.23738v1)

**作者:** Ram Rachum `[一作]` (University of California, Berkeley), Cameron Allen `[通讯]` (University of California, Berkeley)

**通讯引用:** 2956 | [OpenAlex ID](https://openalex.org/A5061301686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

论文提出了强化学习中行为的形式化定义，并将其作为可解释性目标，构建了行为可解释强化学习（BXRL）问题框架。该框架允许用户指定行为测度 m(π)，并利用梯度方法解释行为出现的原因；同时作者将 HighwayEnv 驾驶环境移植到 JAX（HighJax），提供可微分的环境接口。

**💡 创新点**

创新点在于：①把行为定义为从策略到实数的任意可微测度，满足“动作相关”“多次出现”“分级”“限定范围”等属性；②提出对比性行为测度，消解“为什么 P 而不是 Q” 的问题；③将现有 XRL 方法（数据归因、SVERL‑P、COUNTERPOL）映射为 BXRL 方法；④提供可用的 HighJax 环境，使研究者可直接在可微分的驾驶场景上实验。

**🔧 技术方法**

使用技术包括：符号式定义和期望公式、梯度归因（TracIn 变体）、Shapley 值分解、对比性策略优化（COUNTERPOL）以及 JAX 自动微分；实验部分使用 HighwayEnv（四车道高速、离散动作）与手工采样的观测集合。

**📊 数据集**

数据集主要来自 HighJax 的仿真环境，采集了不同状态下的观测分布（例如靠近车辆、靠右车道等）用于计算行为测度；论文中也给出了其他领域的示例（囚徒困境、LLM、机器人、交易、推荐）来说明通用性，但实际实验仅在驾驶环境中进行。

**📈 对比分析**

论文并未实现并比较具体方法，而是给出了方法适配的细节，并指出可通过梯度归因、Shapley 以及对比性策略等方式评估行为测度的变化。性能评估的讨论留在未来工作，作者提议通过用户研究和客观指标来比较解释质量。

**⚠️ 局限性**

主要局限包括：①行为测度的定义需要人工挑选观测分布，难以覆盖所有可能的实现；②在高维或动态状态空间中采样困难；③对复杂行为（多步骤、分支）需要组合多种测度，设计与实现复杂；④目前仅给出方法适配而未在实验中验证效果，缺乏对比实验与实用性评估。

---

## 112. Not All Pretraining are Created Equal: Threshold Tuning and Class Weighting for Imbalanced Polarization Tasks in Low-Resource Settings

**arXiv ID:** 2603.23534 | [PDF](https://arxiv.org/pdf/2603.23534v1)

**作者:** Abass Oguntade `[一作]` `[通讯]` (African Institute of Mathematical Sciences), Abass Oguntade (African Institute of Mathematical Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文在SemEval‑2025极化共享任务中，构建了基于Transformer的系统，对英语和斯瓦希里语的社交媒体文本进行二分类极化检测、多标签目标类型分类和多标签表现形式识别。

**💡 创新点**

创新点在于系统性对比多语言与非洲语言专门模型，发现跨语言模型在极化检测上更优，并通过类别权重损失、迭代分层划分以及逐标签阈值调优显著提升macro‑F1，揭示了多语言训练的负迁移效应。

**🔧 技术方法**

采用的技术包括mDeBERTa‑v3‑base、SwahBERT、AfriBERTa‑large等Transformer架构，BCEWithLogitsLoss加类别权重，iterative stratified split，双阶段逐标签阈值搜索，AdamW+cosine学习率调度，early stopping，以及对emoji、URL、用户标签等的预处理。

**📊 数据集**

使用的语料来自SemEval‑2025 Polarization Shared Task的英语（3,222条）和斯瓦希里语（6,991条）社交媒体文本数据集，涵盖三层子任务标签（极化标记、目标类型、表现形式）。

**📈 对比分析**

在内部验证中对六种Transformer模型进行比较，mDeBERTa‑v3‑base在子任务1上取得最高macro‑F1 0.8032，官方测试达到0.815（英语）和0.785（斯瓦希里）；子任务2/3在阈值调优后分别获得0.556（斯瓦希里）和0.464/0.556（英语/斯瓦希里）macro‑F1，阈值调优提升20+个百分点；多语言训练导致性能下降5‑15个百分点，表明存在负迁移。

**⚠️ 局限性**

主要局限包括阈值调优在测试集上易过拟合导致10‑15个百分点的性能下降；未显式处理代码切换导致误判；未进行任务特定预训练或领域适配；超参搜索范围受限；对极少数类别的识别仍显不足。

---

## 113. Digital Twin-Assisted Measurement Design and Channel Statistics Prediction

**arXiv ID:** 2603.23787 | [PDF](https://arxiv.org/pdf/2603.23787v1)

**作者:** Robin J. Williams `[一作]` (Aalborg University), Petar Popovski `[通讯]` (Aalborg University)

**通讯引用:** 27207 | [OpenAlex ID](https://openalex.org/A5071289803)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用未校准的数字孪生提供几何信息作为高斯过程先验，结合数据驱动的测量选择，进行无线信道统计的空间预测

**💡 创新点**

创新点在于把未校准的数字孪生视为概率先验而非单纯特征生成，直接估计均值与协方差，提升了非平稳、异方差环境下的预测精度

**🔧 技术方法**

采用高斯过程回归、信息熵驱动的贪婪测量选择、基于射线追踪的数字孪生建模（OpenStreetMap + Sionna）

**📊 数据集**

使用开放地图生成的虚拟场景（OpenStreetMap 生成的城市街区），通过射线追踪随机化材料与几何误差得到多场景仿真数据集

**📈 对比分析**

与无信息GP基线和仅用单一数字孪生的平稳核GP基线对比，实验显示该方法在相同测量预算下预测误差显著降低，满足URLLC的元概率约束并实现更高的归一化速率

**⚠️ 局限性**

仍依赖数字孪生的几何准确性，对动态变化环境（如车辆、植被）处理不足，且离线射线追踪产生的计算开销不可忽视

---

## 114. Echoes: A semantically-aligned music deepfake detection dataset

**arXiv ID:** 2603.23667 | [PDF](https://arxiv.org/pdf/2603.23667v1)

**作者:** Octavian Pascu `[一作]` (National University of Science and Technology POLITEHNICA), Nicolas M. Muller `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了Echoes数据集，利用LLM生成的歌曲特征描述并与10种主流AI音乐生成器配合，生成与原曲语义对齐的深伪音乐，随后在此数据集上进行检测基线实验。

**💡 创新点**

创新点包括：①引入语义级对齐的真伪配对，减少内容短路；②覆盖10个多样化生成器，提供长短格式多样的训练样本；③填补现有数据集在提供商多样性和全曲结构上的缺口。

**🔧 技术方法**

使用Wav2Vec2‑XLSR‑2B自监督特征提取，配合逻辑回归分类器；音频按10秒无重叠切片；采用t‑SNE可视化和EER指标进行评估。

**📊 数据集**

主要使用自研Echoes（3577轨、110小时）以及公开的AIME、SONICS、FakeMusicCaps三大数据集进行交叉评估。

**📈 对比分析**

采用相同模型与训练协议进行对比，Echoes在同域EER最高（9.36%），是最难的；在交叉域测试中，Echoes训练模型的平均外域EER为21.0%，明显优于其他数据集训练模型在Echoes上的表现（30%以上）。

**⚠️ 局限性**

局限性：仅使用10秒窗口，未充分利用长上下文信息；对未知生成器、常见后处理或混合真实/伪造内容的鲁棒性未评估；LLM生成描述的多样性与可解释性仍待深入研究。

---

## 115. Internal Safety Collapse in Frontier Large Language Models

**arXiv ID:** 2603.23509 | [PDF](https://arxiv.org/pdf/2603.23509v1)

**作者:** Yutao Wu `[一作]` (Deakin University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24438 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并系统研究了前沿大语言模型在执行合法任务时出现的内部安全崩溃（ISC）现象。

**💡 创新点**

创新点在于定义了ISC概念、提出Task-Validator-Data框架、构建了涵盖53个跨域场景的ISC-Bench基准，揭示了前沿模型的结构性安全缺陷。

**🔧 技术方法**

使用了基于任务-验证器-数据的实验框架、JailbreakBench评估、黑盒对抗测试以及模型内部推理分析等技术。

**📊 数据集**

数据集包括8个专业领域的53个ISC场景，涵盖毒性评估、分子对接、病原基因分析等任务，并结合多款前沿LLM进行评测。

**📈 对比分析**

在三类代表性ISC任务中，四款前沿LLM的最差情况安全失效率平均达95.3%，显著高于传统的21种黑盒越狱攻击，表明ISC对安全评估构成更大威胁。

**⚠️ 局限性**

局限性包括：评估仅覆盖三种任务；未对开源或非对齐模型进行测试；场景由作者手工挑选，可能缺乏全面覆盖；未提出新的防御方案。

---

## 116. S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering

**arXiv ID:** 2603.23512 | [PDF](https://arxiv.org/pdf/2603.23512v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 12009 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了S-Path-RAG框架，用于在大型知识图上进行多跳问答，通过语义感知的路径枚举、可微路径评分与验证、软路径隐向量注入以及基于LLM诊断的迭代图更新，实现高效、可解释的检索增强生成。

**💡 创新点**

创新点包括：① 语义加权的最短路径枚举策略，融合结构成本、关系先验与学习的语义匹配；② 可微路径评分与轻量级验证器联合抑制LLM易认为合理但事实错误的路径；③ 将选定路径压缩为软混合隐向量，使用跨注意力直接注入LLM；④ 通过Neural‑Socratic Graph Dialogue将LLM诊断映射为图编辑，实现自适应检索迭代。

**🔧 技术方法**

技术方法包括：图神经网络编码子图；加权k‑shortest路径、beam搜索和受限随机游走的混合枚举；Gumbel‑Softmax可微路径选择；对比学习（InfoNCE）与二分类验证器；跨注意力隐向量注入；规则或学习式诊断映射；软-离散图更新与强化学习微调。

**📊 数据集**

使用了三大多跳KGQA基准：WebQuestionsSP、ComplexWebQuestions（CWQ）以及MetaQA‑3，此外在OGB WikiKG 2.0上测试了大规模图性能。

**📈 对比分析**

与嵌入式、GNN、LLM、混合等多类基线（如GNN‑RAG、RoG、ToG+ChatGPT、KG‑R1等）对比，S‑Path‑RAG在Hit@1、F1和答案覆盖率上均取得最优成绩（例如WebQSP Hit@1 88.9%，CWQ Hit@1 78.2%），同时LLM调用和token消耗仅为其他迭代检索方法的一半左右，表现出更高的效率与可扩展性。

**⚠️ 局限性**

局限性包括：对初始实体链接的依赖仍需改进；路径枚举上限（L, K）需要手动调参；迭代图更新策略在极端噪声下可能收敛慢；缺乏跨语言或多知识图的通用性；对大规模图的分布式实现尚未完成。

---

## 117. CDMT-EHR: A Continuous-Time Diffusion Framework for Generating Mixed-Type Time-Series Electronic Health Records

**arXiv ID:** 2603.23719 | [PDF](https://arxiv.org/pdf/2603.23719v1)

**作者:** Shaonan Liu `[一作]` (Osaka Metropolitan University), Koichi Kise `[通讯]` (Osaka Metropolitan University)

**通讯引用:** 3297 | [OpenAlex ID](https://openalex.org/A5000232184)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一种连续时间扩散模型 CDMT‑EHR，用于生成包含数值与分类特征的时间序列电子健康记录。

**💡 创新点**

创新点包括：①将双向 GRU 作为时间建模骨干，将连续时间扩散扩展到时序数据；②使用可学习的连续嵌入将分类变量映射到统一的高斯扩散空间；③采用因子化的可学习噪声调度，按特征·时间步动态分配噪声强度。

**🔧 技术方法**

技术方法为：连续时间扩散（SDE/Ode）、双向 GRU、连续嵌入、因子化噪声调度、EDM 预处理与 Euler 采样、分类器无监督引导（CFG）。

**📊 数据集**

实验数据集为 MIMIC‑III（1 小时间隔，7 数值+7 分类）和 eICU（5 分钟间隔，4 数值+4 分类），两者均为 ICU 记录。

**📈 对比分析**

与现有基线 TimeDiff（离散时间 DDPM）相比，CDMT‑EHR 在下游二分类 AUC 约提升 1–4%，分布相似度（MMD、corr‑MAE 等）明显更低，区分度（c2st AUC）接近 0.5；仅需 50 次采样步骤即可，而 TimeDiff 需要 1000 步。

**⚠️ 局限性**

局限性包括：对分类特征的边缘分布拟合仍不理想，复杂非线性交叉依赖的捕获还有提升空间；在长序列 eICU 上的条件生成效果不如短序列 MIMIC‑III；模型对极端缺失模式的鲁棒性未充分验证。

---

## 118. λSplit: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy

**arXiv ID:** 2603.23647 | [PDF](https://arxiv.org/pdf/2603.23647v1)

**作者:** Federico Carrara `[一作]` (Fondazione Human Technopole), Florian Jug `[通讯]` (Fondazione Human Technopole)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为λSplit的物理信息深度生成模型，用于荧光显微镜的光谱解混

**💡 创新点**

创新点在于将可微分的光谱混合模块嵌入层次VAE中，实现自监督学习，学习结构先验并兼顾光谱形成过程

**🔧 技术方法**

使用层次变分自编码器（LVAE）、可微分Spectral Mixer、混合损失与KL正则化等深度学习技术

**📊 数据集**

使用三类公开荧光显微镜数据集：BioSR（2D），CellAtlas（2D），HHMI-D25-8bit（3D），并通过仿真生成66个基准任务

**📈 对比分析**

与10个经典与学习型基准（LU、NNLU、FCLU、HyU、RLU、NMF-RI、LUMoS、TAEU、UNet、AutoUnmix）比较，λSplit在噪声高、光谱重叠强或光谱维数低的情形下均保持最优或竞争优势，尤其在低光谱维数和欠定情形表现突出

**⚠️ 局限性**

局限在于假设线性混合且已知荧光谱，难以处理非线性混合或光谱漂移；目前仅在合成数据验证，真实场景中可能需进一步适配

---

## 119. Efficient Benchmarking of AI Agents

**arXiv ID:** 2603.23749 | [PDF](https://arxiv.org/pdf/2603.23749v1)

**作者:** Franck Ndzomga `[一作]` `[通讯]`, Franck Ndzomga

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何在 AI 代理基准测试中通过仅保留中等难度任务（通过 30%–70% 通过率过滤）来大幅降低评估成本，同时保持排名精度。

**💡 创新点**

创新点在于发现排名预测与绝对分数预测在跨 Scaffold、时间和随机分割下存在显著不对称；提出基于 Item Response Theory 的 Mid‑Range 任务筛选规则，能在不需要优化的情况下稳定保持排行榜排名，并在多种评估协议下实现显著成本削减。

**🔧 技术方法**

主要技术包括：使用 Ridge 回归对子集评分进行校准、Spearman/Kendall 相关系数衡量排名一致性、基于通过率的 IRT 信息量计算实现任务筛选、嵌套交叉验证、对比 Greedy、Random、Stratified、Easy/Hard 等基线。

**📊 数据集**

使用了八个公开基准：Terminal‑Bench 2.0（89 题）以及 Holistic Agent Leaderboard 的七个子基准（CoreBench Hard、GAIA、Mind2Web、SciCode、SWE‑bench、TAU‑bench、USACO），每个基准都有多种 Agent Scaffold 与模型配置，提供了完整的 per‑task 通过率矩阵。

**📈 对比分析**

与 Greedy、Random、Easiest/Hard 等策略比较，Mid‑Range 策略在 Spearman ρ 与 Kendall τ 上平均保持 ≥0.90，且在最差情况下仍 ≥0.87；成本方面可减少 44%–70% 的任务量，对单个 Agent 的评估费用平均降低约 60%（例如 Online Mind2Web 降低至 $149–$253）。

**⚠️ 局限性**

局限性包括：需要足够的中等难度任务（如 SciCode 仅有 4 题不适用）、冷启动成本需 5–10 份完整评估才能稳定筛选、对 Scaffold 多样性和样本量较小的基准（HAL）敏感、所有基准均采用二元或近二元得分，可能不适用于评分更模糊的任务，以及 Mid‑Range 30%–70% 选取阈值为经验性而非最优。

---

## 120. A formalization of System I with type Top in Agda

**arXiv ID:** 2603.23652 | [PDF](https://arxiv.org/pdf/2603.23652v1)

**作者:** Agustín Séttimo `[一作]` (Universidad Nacional de Rosario), Cecilia Manzino `[通讯]` (Universidad Nacional de Rosario)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在 Agda 上实现并正式化了带有 Top 类型的 System I λ 演算，并证明了其进展性和强归约性。

**💡 创新点**

创新点在于：①引入 Top 类型并相应的类型同构与项同构；②采用带显式同构证明的内在类型化术语；③使用 Schäfer 变体的可归约性技术完成强归约性证明。

**🔧 技术方法**

技术手段包括：Agda 形式化、内在类型化、显式同构证明、可归约性（Tait‑Girard）与 Schäfer 技巧、重写与归约规则的组合。

**📊 数据集**

无外部数据集，全部为形式化的演算与证明。

**📈 对比分析**

未与其他方法做实验性对比，主要通过形式化证明展示可归约性与进展性的完整性。

**⚠️ 局限性**

局限性：只覆盖单一系统；同构推理需显式证明；在其他多态或更复杂的 λ 计算中需要额外适配；证明规模大，维护成本高。

---

## 121. Qworld: Question-Specific Evaluation Criteria for LLMs

**arXiv ID:** 2603.23522 | [PDF](https://arxiv.org/pdf/2603.23522v1)

**作者:** Shanghua Gao `[一作]` (Harvard Medical School), Marinka Zitnik `[通讯]` (Harvard Medical School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对每个开放式问题，使用递归展开树（Recursive Expansion Tree）生成场景、视角和细粒度二进制评估标准，构成问题特定的评估“世界”。

**💡 创新点**

创新点在于：①多层次递归拆分（场景→视角→标准），①水平与层级扩展相结合，②生成过程兼顾覆盖度与新颖性，①与传统单次提示或对比生成方法相比，显著提升评估维度的深度与广度。

**🔧 技术方法**

核心技术：大语言模型（GPT‑4.1）作为标准生成器与判定者；递归展开树算法实现层级与水平扩展；可选检索增强（检索相关外部信息）。

**📊 数据集**

使用了 HealthBench（医学领域问答）和 Humanity's Last Exam（抽象推理）两大基准集，用以验证标准生成质量和模型评估效果。

**📈 对比分析**

与现有方法（TICK、RocketEval、OpenRubrics、EvalAgent）对比：覆盖度 0.89（高于 0.46–0.53），唯一性 0.79（高于 0.24–0.50），人工评估洞察度 0.83（提升 0.40），粒度 0.85。评估 11 大模型时，产生更细致的维度分解，排名发生显著变化，评分饱和度下降，揭示可持续性、平等性等细粒度能力。

**⚠️ 局限性**

限制：依赖高质量 LLM，生成成本较高；对提示敏感，可能出现偏差；在非医学领域的通用性尚未充分验证；生成标准的多样性虽高，但仍可能漏掉某些隐式评价维度。

---

## 122. Large Language Models Unpack Complex Political Opinions through Target-Stance Extraction

**arXiv ID:** 2603.23531 | [PDF](https://arxiv.org/pdf/2603.23531v1)

**作者:** Özgür Togay `[一作]` (Utrecht University), Anastasia Giachanou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型在目标‑立场提取（TSE）任务中的表现，并在高语境的政治讨论（Reddit /r/NeutralPolitics）上构建并公开标注数据集。

**💡 创新点**

将目标识别与立场分类统一到单一LLM，并通过多种提示策略（零样本、少样本、上下文补充）系统评估，提供可复现的框架和开源数据集。

**🔧 技术方法**

使用 instruction‑tuned 大型语言模型（如 GPT‑4.1、o3、Gemini 等）结合多种提示策略（zero‑shot、few‑shot、对话上下文、信息上下文）。

**📊 数据集**

1,084 条来自 Reddit /r/NeutralPolitics 的评论，包含 138 个目标和半开放 “Other” 类别；其中 200 条由专家验证的金标注子集作为评估基准。

**📈 对比分析**

在零样本、少样本等多提示条件下评估多款模型，最高性能模型 o3 在少样本+信息上下文下目标 F1 0.76、立场 F1 0.87，整体大模型表现优于小模型，提示策略显著提升目标识别。

**⚠️ 局限性**

局限性：数据集仅来自单一高语境政治论坛，难以推广；依赖预先定义的目标列表，无法覆盖开放目标；对多目标或模糊语境的处理仍有限。

---

## 123. PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation

**arXiv ID:** 2603.23678 | [PDF](https://arxiv.org/pdf/2603.23678v1)

**作者:** Manjushree B. Aithal `[一作]` (University of Colorado Anschutz), Ph. D `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在隐私保护的本地化环境下使用小参数LLM实现临床文本中缩略语的识别与展开，提出了检测-展开的级联流水线；

**💡 创新点**

创新点在于将通用小模型用于检测、医学专门化小模型用于展开，既保证数据不外泄，又显著提升缩略语展开准确率；

**🔧 技术方法**

采用零样本提示、Apple M4‑Max GPU+MLX推理框架、4‑bit量化、JSON输出结构化自信度与理由；

**📊 数据集**

使用GLADIS（Biomedical子集≈12k条）和1k通用数据做验证，包含单一缩写与多缩写样本；

**📈 对比分析**

与通用小模型单通道和级联两种配置对比，检测准确率最高0.99，级联展开准确率提升至≈0.81（相比单通道的0.65）；

**⚠️ 局限性**

局限性包括模型对罕见缩写的误判、过度自信、仍需多模型集成或RAG增强、以及缺乏对真实EHR复杂语境的全面评估。

---

## 124. LLMORPH: Automated Metamorphic Testing of Large Language Models

**arXiv ID:** 2603.23611 | [PDF](https://arxiv.org/pdf/2603.23611v1)

**作者:** Steven Cho `[一作]` (University of Auckland), Valerio Terragni `[通讯]` (University of Auckland)

**通讯引用:** 780 | [OpenAlex ID](https://openalex.org/A5068101658)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了面向大语言模型的自动化测试工具 LLMORPH，利用变形测试 (Metamorphic Testing) 在不依赖人工标注的前提下评估 LLM 在多种 NLP 任务中的鲁棒性。

**💡 创新点**

创新点包括：①首次系统化整合 191 条 NLP 变形规则（MR），并实现 36 条主流规则；②将 MT 与 LLM 结合，使用少量示例提示实现复杂变形；③提供可扩展的任务、MR 与 LLM 接口，支持 CLI 与配置文件两种运行方式；④公开源代码与文档，促进社区快速贡献。

**🔧 技术方法**

核心技术：变形测试 (Metamorphic Testing)、少量示例提示 (few-shot prompting)、语义相似度计算 (BERT‑based paraphrase‑MiniLM‑L6‑v2)、数值相似度判定、Python 与 OpenAI API 的集成。

**📊 数据集**

评估使用四个主流 NLP 基准：SQuAD2（问答）、SNLI（自然语言推理）、SST2（情感分析）和 RE‑DOCRED（关系抽取）。测试了 GPT‑4、LLAMA3 与 HERMES 2 三款 LLM，累计 561,000 次测试执行。

**📈 对比分析**

与传统基于标注的测试对比，LLMORPH 能发现 18% 的平均错误率，且能检测到传统方法遗漏的缺陷；测试过程仅需 2–3 次 LLM 调用，速度较快。实验中发现不同任务/规则的误报率从 0% 变化到 70%，说明 MT 在 NLP 领域仍需改进。

**⚠️ 局限性**

局限性：① MT 在 NLP 中固有的误报率高，取决于 MR 定义和阈值；② 现有 36 条 MR 仍远少于已收集的 191 条；③ 对模型的依赖度高，若 LLM 本身生成的变形或比较不精确，可能导致错误判定；④ 需要手动维护任务模板、MR 代码与阈值，仍有一定的工程成本。

---

## 125. Retinal Disease Classification from Fundus Images using CNN Transfer Learning

**arXiv ID:** 2603.23785 | [PDF](https://arxiv.org/pdf/2603.23785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 126. Can LLM Agents Be CFOs? A Benchmark for Resource Allocation in Dynamic Enterprise Environments

**arXiv ID:** 2603.23638 | [PDF](https://arxiv.org/pdf/2603.23638v1)

**作者:** Yi Han `[一作]` (Georgia Institute of Technology), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17402 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一个名为EnterpriseArena的长期企业资源配置基准，模拟CFO在不确定环境下的资金分配与决策

**💡 创新点**

首次将资源分配与不确定性结合到长期、部分可观测、随机动态的企业金融仿真中，突出资源约束与信息获取权衡

**🔧 技术方法**

采用大型语言模型（LLM）作为智能体，使用ReAct框架与工具调用，实现与仿真环境的交互与决策

**📊 数据集**

基于企业级财务报表、行业指标与宏观经济历史数据（约132个月月度），并进行匿名化与噪声注入

**📈 对比分析**

对11种不同规模与来源的LLM进行评测，衡量生存率与终端估值，结果显示仅有16%能完整存活，Qwen3.5‑9B最高生存率（80%），但整体表现远低于人类专家

**⚠️ 局限性**

限制包括仿真环境无法完全覆盖真实“黑天鹅”事件、多层次决策结构缺失、仅测试有限模型与框架，易出现过拟合与不具备真实世界适用性

---

## 127. PerturbationDrive: A Framework for Perturbation-Based Testing of ADAS

**arXiv ID:** 2603.23661 | [PDF](https://arxiv.org/pdf/2603.23661v1)

**作者:** Hannes Leonhard `[一作]` (Technical University of Munich), Andrea Stocco `[通讯]` (Technical University of Munich)

**通讯引用:** 2516 | [OpenAlex ID](https://openalex.org/A5027652385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了一个用于评估ADAS鲁棒性与泛化的可扩展测试框架。

**💡 创新点**

创新点在于集成30余种天气、光照及传感器质量的图像扰动，并引入动态与注意力变体，支持离线与在线闭环测试，同时与程序化道路生成和搜索式测试无缝结合。

**🔧 技术方法**

使用深度学习模型、图像扰动生成技术、搜索式优化方法以及闭环仿真平台进行系统评估。

**📊 数据集**

主要利用公开静态数据集（如KITTI等）和仿真器中的动态场景进行离线与在线测试。

**📈 对比分析**

通过在不同扰动条件下对模型性能进行对比，框架能够直观展示鲁棒性下降趋势，证明其在系统级测试中的有效性。

**⚠️ 局限性**

局限性包括：目前仅针对图像扰动，未覆盖多模态传感器；对实时计算要求高，资源消耗大；缺乏在真实车辆上的实测验证。

---

## 128. Did You Forget What I Asked? Prospective Memory Failures in Large Language Models

**arXiv ID:** 2603.23530 | [PDF](https://arxiv.org/pdf/2603.23530v1)

**作者:** Avni Mittal `[一作]` `[通讯]` (Microsoft), Avni Mittal (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在同时完成任务和遵循格式指令时的遗忘现象，采用基于前瞻记忆的实验设计；

**💡 创新点**

发现任务负荷会显著降低格式指令的遵循率，终端约束最易遗忘，并提出通过在提示中加入显著提醒句可显著恢复遵循率；

**🔧 技术方法**

使用代码式可验证的 IFEval 格式检查器、任务特定准确性检查器以及在不同提示模板下的对比实验；

**📊 数据集**

使用 IFEval、TriviaQA、MMLU、GSM8K 和 CNN/DailyMail 五个公开数据集；

**📈 对比分析**

对比自然嵌入提示与提醒增强提示，发现后者在多模型和多任务负荷下可将遵循率恢复到 90–100%，而任务准确率在加上格式约束后会下降；

**⚠️ 局限性**

实验受限于仅评估三种模型、单轮提示、有限的 IFEval 指令样本、未拆分提醒效果及单次堆叠实验样本量小等因素。

---

## 129. LLMLOOP: Improving LLM-Generated Code and Tests through Automated Iterative Feedback Loops

**arXiv ID:** 2603.23613 | [PDF](https://arxiv.org/pdf/2603.23613v1)

**作者:** Ravin Ravi `[一作]` (University of Auckland), Valerio Terragni `[通讯]` (University of Auckland)

**通讯引用:** 780 | [OpenAlex ID](https://openalex.org/A5068101658)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个自动化框架，利用多轮反馈循环不断完善大语言模型生成的Java代码和测试用例；

**💡 创新点**

创新点在于将编译错误、静态分析、测试失败、自动测试生成以及变异测试等五种迭代循环整合到同一框架，并通过Docker沙箱实现安全执行；

**🔧 技术方法**

采用了OpenAI LLM（GPT‑4o‑mini）作为核心生成器，配合Maven、PMD、EvoSuite、PIT等工具进行编译、静态分析、单元测试与变异测试；

**📊 数据集**

使用了扩展版HumanEval（Java版）Benchmark，包含164道编程题及其对应测试套件；

**📈 对比分析**

通过pass@k（k=1…10）指标对比基线（仅调用一次LLM）和完整框架，结果显示pass@1从71.65%提升至80.85%，pass@10从76.22%提升至90.24%，整体提升约9.2%；

**⚠️ 局限性**

主要限制包括高耗时与高成本的多次LLM调用、缺乏对生成测试质量的深入评估，以及目前仅支持Java，未来需进一步优化交互效率并扩展至Python等语言。

---

## 130. Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction

**arXiv ID:** 2603.23550 | [PDF](https://arxiv.org/pdf/2603.23550v1)

**作者:** Haoyu Wang `[一作]` (Georgia Institute of Technology), Pan Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 17686 | [OpenAlex ID](https://openalex.org/A5100455184)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于隐式过程奖励模型的多轮人机协作策略优化框架 ITPO，能够从稀疏的最终奖励自动生成细粒度的轮次级奖励，并通过归一化机制提升训练稳定性。

**💡 创新点**

创新点在于：①利用隐式 PRM 直接从全局奖励学习 token 级奖励，再聚合为轮次级奖励；②设计归一化（Norm-）方案将全局奖励按轮次重要性分配，解决奖励尺度不稳定问题；③将轮次奖励与常用优势估计（PPO、GRPO、RLOO）无缝结合，实现在线多轮学习。

**🔧 技术方法**

主要技术包括：隐式过程奖励模型（Implicit PRM）、轮次级奖励归一化（Norm-）、优势估计方法（PPO、GRPO、RLOO）、基于 Qwen 语言模型的用户模拟器与策略微调。

**📊 数据集**

使用了三类任务的数据集：Math Tutoring（MATH 500题）、Document Writing（Medium 500篇文章）、Medical Recommendation（MTMedDialog 550个样本）。

**📈 对比分析**

与基线（稀疏全局奖励、均匀分配、值模型、LLM-as-Judge、PRM）以及多种优势估计器比较，ITPO+Norm-在所有任务上均明显提升，最高可达 34.4%（数学）、12.0%（写作）、8.0%（医学）相对稀疏奖励基线；在使用 PPO 值模型时提升更显著。

**⚠️ 局限性**

局限性：①依赖用户模拟器，真实用户行为仍有差异；②归一化参数温度 η 的选择需经验调优；③在极端长轮次或高度多样化任务中，隐式奖励可能仍受高方差影响，需进一步研究更鲁棒的聚合方式。

---

## 131. LineMVGNN: Anti-Money Laundering with Line-Graph-Assisted Multi-View Graph Neural Networks

**arXiv ID:** 2603.23584 | [PDF](https://arxiv.org/pdf/2603.23584v1)

**作者:** Chung-Hoo Poon `[一作]` (Logistics and Supply Chain MultiTech R&D Centre), Jang-Hyeon Choi `[通讯]` (Logistics and Supply Chain MultiTech R&D Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种新型的图神经网络框架LineMVGNN，用于检测金融交易图中的洗钱行为。

**💡 创新点**

创新点在于结合两路信息传播（入邻/出邻）与线图视角，实现边（交易）信息在节点更新前的预传播，从而更精准捕捉资金流动与可疑模式。

**🔧 技术方法**

采用的技术包括多视图图神经网络(MVGNN)的参数共享设计、两路信息传播机制、线图转换、非回溯矩阵、以及对比学习与残差更新。

**📊 数据集**

实验使用了公开的Ethereum Phishing Transaction Network（ETH）以及公司内部的Financial Payment Transaction（FPT）数据集，其中ETH进一步划分为带/不带结构节点特征的子集。

**📈 对比分析**

与多种基准（GCN、GraphSAGE、DiGCN、MagNet、FaberNet等）进行对比，LineMVGNN在所有数据集上均获得了F1分数的显著提升，尤其在FPT上接近或达到99%以上。

**⚠️ 局限性**

主要局限包括：对超大规模图的实际运行时间仍高于传统GCN；缺乏对抗性鲁棒性与隐私保护的评估；并且模型解释性虽有提升，但尚未深入整合XAI技术。

---

## 132. Augmented Reality Visualization for Musical Instrument Learning

**arXiv ID:** 2603.23639 | [PDF](https://arxiv.org/pdf/2603.23639v1)

**作者:** Frank Heyen `[一作]` (University of Stuttgart), Michael Sedlmair `[通讯]` (University of Stuttgart)

**通讯引用:** 5504 | [OpenAlex ID](https://openalex.org/A5037110552)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aaccfe5c-6b26-4208-b23c-35331481e142` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用投影仪和光学可穿透AR头显为鼓和吉他提供实时和后期反馈的可视化，展示力度、节奏等信息。

**💡 创新点**

将视觉反馈直接投射到乐器表面，并在3D空间中呈现更丰富的数据，提升学习者对练习细节的理解。

**🔧 技术方法**

投影仪AR、Microsoft HoloLens光学可穿透AR、相机标记跟踪、MIDI数据解析、图表可视化技术。

**📊 数据集**

鼓的MIDI输出、吉他的标记跟踪音频与练习记录、节奏与力度分布。

**📈 对比分析**

通过案例研究评估可视化的有效性，结果显示更直观的反馈可提升学习效率，但未提供定量性能指标。

**⚠️ 局限性**

投影仪分辨率低、焦距有限；AR头显视野小、易遮挡；跟踪误差导致信息错位；实时可视化可能分散注意力。

---

## 133. IslamicMMLU: A Benchmark for Evaluating LLMs on Islamic Knowledge

**arXiv ID:** 2603.23750 | [PDF](https://arxiv.org/pdf/2603.23750v1)

**作者:** Ali Abdelaal `[一作]` (University of Edinburgh), Walid Magdy `[通讯]` (University of Edinburgh)

**通讯引用:** 4011 | [OpenAlex ID](https://openalex.org/A5070783596)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 IslamicMMLU 基准，包含 10,013 道四选一多项选择题，覆盖古兰经、圣训和伊斯兰法学（Fiqh）三个核心学科，并在其中设计了 12 种不同的任务类型。

**💡 创新点**

创新点在于（1）首次将 MMLU 评价框架系统化应用于伊斯兰知识领域；（2）引入了 madhab（法学派）偏差检测任务，衡量模型在同一合法性体系内的隐性倾向；（3）搭建公开排行榜和可复现的评测代码，支持社区持续迭代。

**🔧 技术方法**

使用了 MMLU 评价范式、零样本（zero‑shot）中文提示、Bootstrap 置信区间、McNemar 统计检验、卡方检验评估偏差，以及多模态（多模型并行抽取+投票）pipeline 处理 Fiqh 语料。

**📊 数据集**

数据集来源：正体古兰经文本、六部逊尼圣训正典（布哈里、穆斯林、阿布·达乌德等）、以及阿卜杜拉·拉赫曼·阿尔‑贾兹里所著的《四大法学派法学百科》；全部题目均用现代标准阿拉伯语书写。

**📈 对比分析**

对 26 个 LLM（包括 Gemini、GPT‑5、Claude、Arabic‑specific 等）进行统一评测，采用平均精度（各轨道等权）作为综合分；最高分 93.8%（Gemini 3 Flash），最低 39.8%（GPT‑3.5‑turbo），显示出显著的性能差距；同时对偏差分布进行统计，揭示不同模型在 madhab 选择上的系统性倾向。

**⚠️ 局限性**

局限性包括：只覆盖逊尼派，未包含什叶派；Fiqh 语料单一来源；人工验证仅有 1 名专家；全部题目为现代标准阿拉伯语，缺少跨语言评测；采用 MCQ 形式，难以测量生成能力；结果随时间变化，评测结果可能不再适用。

---

## 134. CAPTCHA Solving for Native GUI Agents: Automated Reasoning-Action Data Generation and Self-Corrective Training

**arXiv ID:** 2603.23559 | [PDF](https://arxiv.org/pdf/2603.23559v1)

**作者:** Yuxi Chen `[一作]` (University of Illinois Urbana-Champaign), Huan Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6533 | [OpenAlex ID](https://openalex.org/A5100356973)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可在本地 GUI 环境中解决现代交互式 CAPTCHA 的端到端视觉语言模型 ReCAP，并在其上实现了自我纠错训练。

**💡 创新点**

创新点在于：①设计了覆盖七类 CAPTCHA 的动态渲染系统，提供高多样性和可扩展的数据；②构建了自动化的链式推理与动作轨迹生成与自我纠错数据收集管道；③采用统一的加权损失同时优化推理与操作，提升交互效率。

**🔧 技术方法**

使用 Qwen3-VL 系列（8B、32B）作为基础模型，结合链式推理（CoT）和自我纠错数据；在训练中加入了视觉编码、动作生成和加权损失；实现了多步操作和多动作响应。

**📊 数据集**

主要数据集包括：1) 自研的动态 CAPTCHA 系统生成的 150k 解决轨迹与 10k 自我纠错轨迹；2) 50k 通用 GUI 轨迹（Aguvis、AgentNet）；3) 真实 CAPTCHA 评测集 26 类 10k+样本。

**📈 对比分析**

与多种基线（UI‑TARS, OpenCUA, Halligan、原 Qwen3-VL-Thinking）在 1,000 题动态 CAPTCHA、26 类真实 CAPTCHA 以及 Android Control、ScreenSpot‑V2、Multimodal‑Mind2Web 三大通用 GUI 任务上进行比较。ReCAP‑32B 在动态 CAPTCHA 上从 30% 提升到 81% 成功率，平均交互步数降至 1.54 步；在真实 CAPTCHA 上实现多项最佳或次优成绩；在通用 GUI 任务上保持或略超基线。

**⚠️ 局限性**

局限性包括：①小模型（8B）在多任务学习中表现下降，表明容量受限；②主要针对图形化 CAPTCHA，尚未充分验证在文本密集型或语义复杂 CAPTCHA 的泛化；③对 CAPTCHA 生成的多样性仍依赖程序化随机化，真实场景中更复杂的变异可能导致性能下降。

---

## 135. An Adapter-free Fine-tuning Approach for Tuning 3D Foundation Models

**arXiv ID:** 2603.23730 | [PDF](https://arxiv.org/pdf/2603.23730v1)

**作者:** Sneha Paul `[一作]` (Concordia University), Nizar Bouguila `[通讯]` (Concordia University)

**通讯引用:** 9691 | [OpenAlex ID](https://openalex.org/A5090600716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无适配器的 Momentum-Consistency Fine‑Tuning (MCFT) 方法，用于在低样本环境下微调 3D 基础模型，并进一步扩展到半监督和结构化剪枝变体。

**💡 创新点**

创新点在于：1) 通过选择性微调并引入动量一致性自蒸馏约束，既避免过拟合又不增加推理参数；2) 将 AllMatch 半监督框架与 MCFT 结合提升无标签数据利用率；3) 采用结构化层剪枝实现更高效的模型压缩。

**🔧 技术方法**

核心技术包括：动量一致性自蒸馏（EMA 与对齐损失）、交叉熵分类损失、AllMatch 半监督框架（弱/强增广与伪标签）、结构化剪枝与 salience 分数计算、以及 EMA 更新策略。

**📊 数据集**

实验数据集涵盖：ModelNet40、ScanObjectNN（OBJ_BG、OBJ_ONLY、PB_T50_RS）、ShapeNetPart 以及合成与真实点云数据。

**📈 对比分析**

与全微调、IDPT、DAPT、PointGST 等多种 PEFT 方案以及现有 SOTA 方法对比，MCFT 在 5‑shot 下提升 3.30%，半监督版提升至 6.13%（84.49% 5‑shot 准确率），剪枝版保持相近性能并减少参数；在全监督设置下提升 1.4–2.7%；通过 FLOPs、吞吐率和参数量评估显示 MCFT 兼具高性能与高效能。

**⚠️ 局限性**

局限性包括：对动量因子、剪枝阈值等超参数需数据集特定调优；方法在极低样本或大规模场景级任务中的适应性尚待验证；依赖预训练模型，若预训练质量不足可能影响效果。

---

## 136. Learning Cross-Joint Attention for Generalizable Video-Based Seizure Detection

**arXiv ID:** 2603.23757 | [PDF](https://arxiv.org/pdf/2603.23757v1)

**作者:** Omar Zamzam `[一作]` (University of Southern California), Richard Leahy `[通讯]` (University of Southern California)

**通讯引用:** 29101 | [OpenAlex ID](https://openalex.org/A5054387045)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于关节中心注意力的长时临床视频癫痫发作检测框架，专注于提取和建模人体关节运动动态。

**💡 创新点**

创新点在于：①将视频分解为关节中心子视频以消除背景和个体差异；②使用预训练的ViViT提取关节级运动令牌；③通过自注意力学习可适应的跨关节关系；④采用LoRA实现参数高效微调，显著提升跨主体泛化能力。

**🔧 技术方法**

技术包括：OpenPose姿态估计、ViViT视频视觉变压器、位置编码、跨关节多头自注意力、LoRA低秩适配、二分类损失与性能评估指标。

**📊 数据集**

使用公共长时癫痫视频数据集（VSViG）：33段视频、14名受试者，分5秒段进行标注，训练/验证/测试按受试者拆分。

**📈 对比分析**

与VSViG、3D-CNN、Hiera+LoRA等现有方法对比，实验显示在保持不被背景影响的条件下，提议方法在未见受试者上取得最高准确率（0.866-0.889）、AUROC（0.921-0.923）和AUPRC（0.870-0.881），明显优于基线。

**⚠️ 局限性**

局限性：仅评估运动型癫痫发作，对无运动或细微面部运动的非运动型癫痫未作验证；依赖姿态估计的准确性；数据集规模有限，可能影响更广泛适用性。

---

## 137. DeepOFW: Deep Learning-Driven OFDM-Flexible Waveform Modulation for Peak-to-Average Power Ratio Reduction

**arXiv ID:** 2603.23544 | [PDF](https://arxiv.org/pdf/2603.23544v1)

**作者:** Ran Greidi `[一作]` (Ben Gurion University Of Negev), Kobi Cohen `[通讯]` (Ben Gurion University Of Negev)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种深度学习驱动的OFDM灵活波形调制框架DeepOFW，能够在保持传统低复杂度硬件结构的同时通过数据驱动优化降低PAPR并提升误码率。

**💡 创新点**

创新点在于：① 采用全可微的PHY架构，允许端到端梯度优化；② 将深度学习推理阶段集中在基站或离线计算单元，终端仅执行线性变换；③ 在训练过程中显式加入PAPR约束并自适应阈值；④ 通过可学习的波形矩阵Q与一阶检测系数实现对不同信道的自适应时间–频率重排。

**🔧 技术方法**

技术包括：深度学习（GRU+全连接网络生成Q与q），端到端可微信号链，PAPR正则化与不确定性加权损失，多目标损失平衡，仿真平台Sionna，基于3GPP TDL-A多径信道模型的随机生成。

**📊 数据集**

使用3GPP TDL-A多径信道模型（延迟扩展10~600 ns），并在不同SNR下生成随机符号块；实验规模为N=32子载波、16-QAM、每批384符号、后期细化为单符号。

**📈 对比分析**

与传统OFDM、SC-FDE（单载波DFE）以及E2E学习波形（E2EWL）对比。结果显示：DeepOFW在PAPR CCDF曲线上显著优于OFDM，误码率在相同SNR下低于所有基准，并保持仅一阶检测的低接收机复杂度。

**⚠️ 局限性**

局限性包括：① 训练和参数分发需要基站或离线计算资源；② 只考虑单用户线性链路，未验证多用户/多天线场景；③ 采用单一子载波数目和调制方案，未展示对更高阶QAM或大N的可扩展性；④ 仅在仿真环境下验证，缺乏真实硬件落地实验。

---

## 138. Estimating Individual Tree Height and Species from UAV Imagery

**arXiv ID:** 2603.23669 | [PDF](https://arxiv.org/pdf/2603.23669v1)

**作者:** Jannik Endres `[一作]` (Mila – Quebec AI Institute), Arthur Ouaknine `[通讯]` (Mila – Quebec AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的基准和模型，用于从无人机RGB图像中同时估计单棵树的高度和物种。

**💡 创新点**

创新点在于构建跨三种森林类型的树冠级别基准，并提出多任务模型 
<PaperName>，利用共享的 DINOv3 backbone 与跨注意力任务头实现高效双任务预测。

**🔧 技术方法**

主要技术包括自监督视觉基础模型 DINOv3、跨注意力任务特定头、动态权重平均（DWA）损失平衡、以及 99th 百分位高度提取等。

**📊 数据集**

使用了三大公开数据集：Quebec Trees（温带森林）、Barro Colorado Island (BCI)（热带森林）和Quebec Plantations（针叶种植园）。

**📈 对比分析**

与传统几何方程、Mask R-CNN、CNN、ViT、Mamba 等方法对比，<PaperName> 在高度估计上达 MAE≈1.3m、δ1.25≈88%，并在分类上获得 F1≈86%，参数量仅为最佳方法的 54–58%。

**⚠️ 局限性**

局限性包括对稀有物种样本不足、对极低/高树木高度的偏差、对高分辨率无人机数据的依赖，且模型尚未在无标签或多模态场景下验证。

---

## 139. Prompt Compression in Production Task Orchestration: A Pre-Registered Randomized Trial

**arXiv ID:** 2603.23525 | [PDF](https://arxiv.org/pdf/2603.23525v1)

**作者:** Warren Johnson `[一作]` (Plexor Labs), Charles Lee `[通讯]` (Project Autobots)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在真实生产多智能体任务调度环境中，开展了一个六臂随机对照试验，系统评估了不同提示压缩策略（均匀压缩、熵自适应、递归加权等）对Claude Sonnet 4.5推理成本与输出相似度的影响。

**💡 创新点**

创新点在于：①把压缩效果置于真实生产指令而非基准数据上；②首次系统记录并分析压缩导致的输出 token 变化，揭示“压缩越多”并非成本节约的可靠启发式；③提出并验证结构感知的自适应压缩策略（entropy‑adaptive 与 recency‑weighted）。

**🔧 技术方法**

采用的技术包括：随机化试验设计与预注册分析计划、均匀压缩、熵自适应与递归加权压缩、Claude Sonnet 4.5 API（$3/10⁶ 输入、$15/10⁶ 输出）、嵌入相似度（OpenAI 1,536 维）以及 Bootstrap、Welch ANOVA、Pareto 前沿分析等。

**📊 数据集**

使用的数据集为 1,199 条独特的多智能体任务指令，来自两个 Azure 容器应用部署，涵盖七类任务（实现、分解、验证等），长度分布广泛，平均输入 token 107，平均输出 token 916。

**📈 对比分析**

通过比较六种压缩策略在输入 token、总成本、输出 token、嵌入相似度等指标，结果显示中等压缩（r≈0.5）可减少约27.9% 总成本，recency‑weighted 23.5%；aggressive 20% 反而导致成本上升1.8%，输出扩张 1.03×。两者在 Pareto 前沿表现最佳，但均未达到预设的 0.85 相似度阈值。

**⚠️ 局限性**

主要局限包括：仅使用 Claude Sonnet 4.5 并采用截断压缩；成功率仅 29%（358/1,199），高失败率可能导致样本偏倚；仅对完整案例进行推断，未覆盖全部随机样本；未评估功能正确性，仅使用嵌入相似度代理；任务类型分布偏向实现/分解；未做跨提供商或纵向验证。

---

## 140. Foundation Model Embeddings Meet Blended Emotions: A Multimodal Fusion Approach for the BLEMORE Challenge

**arXiv ID:** 2603.23650 | [PDF](https://arxiv.org/pdf/2603.23650v1)

**作者:** Masoumeh Chapariniya `[一作]` (University of Zürich), Teodora Vukovic `[通讯]` (University of Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套多模态融合系统，用于BLEMORE挑战中混合情感识别与相对显著性预测。

**💡 创新点**

首次将Gemini Embedding 2.0等大型多模态模型嵌入情感识别，并使用层选择冻结的Wav2Vec2中间层来捕捉非语言音频情感，同时通过soft‑label KL训练保留混合比例信息。

**🔧 技术方法**

S4D‑ViTMoE面部编码、TimeSformer/VideoMAE身体编码、层选择冻结的Wav2Vec2音频特征、Gemini 2.0 LMM嵌入、预提取的基线特征、加权后期融合和阈值化后处理。

**📊 数据集**

在BLEMORE数据集上训练、验证与测试，包含3,050段视频、58名演员、6种基本情感和10种混合组合。

**📈 对比分析**

与官方基线和其他挑战提交进行5折交叉验证和服务器测试对比，12编码器融合在测试集上获得Score=0.279（ACC_P=0.391），排名第6，明显优于最优基线Score=0.223。

**⚠️ 局限性**

阈值β对演员高度依赖，导致跨折差异大；Gemini仅使用2秒视频限制了显著性判断；模型对小规模非语言数据易过拟合，需更鲁棒的连续回归或无阈值方案。

---

## 141. MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens

**arXiv ID:** 2603.23516 | [PDF](https://arxiv.org/pdf/2603.23516v1)

**作者:** Yu Chen `[一作]` (Evermind), Tianqiao Chen `[通讯]` (Shanda Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Memory Sparse Attention (MSA) 框架，实现对长达 100M 词级别上下文的端到端可训练记忆与推理。

**💡 创新点**

核心创新包括：文档级 RoPE 与稀疏 top‑k 选择的注意力，KV 缓存压缩，Memory Parallel 推理以及 Memory Interleave 多跳推理机制，确保线性复杂度并将上下文衰减降至 <9%。

**🔧 技术方法**

采用稀疏注意力、文档级旋转位置嵌入、KV 缓存压缩、Memory Parallel 并行检索、Memory Interleave 多轮检索、路由器投影监督、连续预训练与两阶段课程学习等技术。

**📊 数据集**

使用 158.95B 词的去重语料进行预训练；在 QA 方面评测 MS MARCO、Natural Questions、DuReader、TriviaQA、NarrativeQA、PopQA、2WikiMultiHopQA、HotpotQA、MuSiQue；在 NIAH 上使用 RULER 数据集。

**📈 对比分析**

与同基底 RAG（Qwen3‑4B）以及最佳 RAG（KaLMv2+Qwen3‑235B、Llama3.3‑70B 等）对比，MSA 在大多数基准上获得更高 LLM judge 分数（平均 3.760 vs 3.179），并在 1M 词 NIAH 上保持 94.84% 准确率，远优于其他长上下文模型与外部存储方案。

**⚠️ 局限性**

局限性在于对跨文档高度耦合的依赖建模能力不足，需进一步改进 Memory Interleave 以更好地维护文档间结构关系。

---

## 142. Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents

**arXiv ID:** 2603.23518 | [PDF](https://arxiv.org/pdf/2603.23518v1)

**作者:** Peijun Qing `[一作]` (Dartmouth College), Soroush Vosoughi `[通讯]` (Dartmouth College)

**通讯引用:** 9966 | [OpenAlex ID](https://openalex.org/A5035399743)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

将文本聚类重新表述为生成任务，训练大型推理模型实现指令跟随聚类；

**💡 创新点**

提出生成式聚类框架、推理蒸馏与GRPO强化学习相结合的训练流程，并创建ReasonCluster基准；

**🔧 技术方法**

使用推理蒸馏、Group Relative Policy Optimization（GRPO）强化学习、Qwen‑2.5‑Instruct大模型及V‑measure评估；

**📊 数据集**

在28个任务中构建的ReasonCluster基准，涵盖日常对话（LMSYS‑Chat）、法律（ECHR）和金融（S&P500）等多域数据；

**📈 对比分析**

与多种嵌入+K‑means/GMM、指令调优嵌入、通用LLM及开源LRM基线对比，平均V‑measure超过68%，显著优于所有对照方法；

**⚠️ 局限性**

受限于现有大模型的上下文长度瓶颈，难以一次性推理海量文本；此外模型偶尔会出现格式错误，如重复或漏项。

---

## 143. Dual-Criterion Curriculum Learning: Application to Temporal Data

**arXiv ID:** 2603.23573 | [PDF](https://arxiv.org/pdf/2603.23573v1)

**作者:** Gaspard Abel `[一作]` (Université Paris Saclay), Argyris Kalogeratos `[通讯]` (Université Paris Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多变量时间序列预测任务中，提出并实现了双指标课程学习框架（DCCL），将模型训练误差和表示空间的实例密度两种难度度量结合起来指导训练顺序。

**💡 创新点**

创新点在于将损失和密度两种互补难度指标融合，并给出三种混合策略（凸值、凸秩、二维网格），为课程学习提供通用、可调的度量方法。

**🔧 技术方法**

采用序列表示学习（Transformer/LSTM）提取嵌入，kNN/KDE估计密度，混合策略结合损失/密度，并在 One‑Pass 与 Baby‑Steps 两种调度下训练。

**📊 数据集**

使用五个公开多变量时序数据集：Electricity、ETT、Weather、ILI、Solar AL。

**📈 对比分析**

与无课程、随机划分、STL难度以及单一指标课程的基线比较，双指标混合策略在多数数据集上实现显著的 MSE 降低，平均排名在 One‑Pass 下为 3.6–3.8，Baby‑Steps 下为 2.4–3.8。

**⚠️ 局限性**

局限性包括：需先训练表示模型并调参；Baby‑Steps 调度显著增加训练时间；对不同任务的泛化仍需进一步验证；密度估计对表示质量高度依赖。

---

## 144. Latent Algorithmic Structure Precedes Grokking: A Mechanistic Study of ReLU MLPs on Modular Arithmetic

**arXiv ID:** 2603.23784 | [PDF](https://arxiv.org/pdf/2603.23784v1)

**作者:** Anand Swaroop `[一作]` `[通讯]`, Anand Swaroop

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究ReLU MLP在模加任务中出现的grokking现象，并通过频谱分析发现输入权重趋向二进制方波、输出权重满足相位加法关系。

**💡 创新点**

首次揭示在ReLU网络中权重呈现方波结构，而非先前研究的正弦结构，并证明grokking仅是对已存在的算法结构进行锐化。

**🔧 技术方法**

使用AdamW优化器、权重衰减、离散傅里叶变换提取频率相位，并构造理想化MPL模型来验证结构。

**📊 数据集**

采用全量97²整数三元组（a,b,c）构成的模加数据集，按30/70划分训练/验证。

**📈 对比分析**

通过将提取的相位频率用于理想化模型与实际模型对比，理想化模型在噪声高达30%时从0.23%提升至95.5%准确率，显示理想化方法优于原始网络。

**⚠️ 局限性**

实验仅覆盖单一任务、单隐藏层ReLU MLP和模97，缺乏对其他任务、架构或更大隐藏层的验证。

---

## 145. Form-Fitting, Large-Area Sensor Mounting for Obstacle Detection

**arXiv ID:** 2603.23725 | [PDF](https://arxiv.org/pdf/2603.23725v1)

**作者:** Anna Soukhovei `[一作]` (University of Colorado Boulder), Alessandro Roncone `[通讯]` (University of Colorado Boulder)

**通讯引用:** 712 | [OpenAlex ID](https://openalex.org/A5020277024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种低成本、可程序化生成的机器人皮肤单元，用于在机器人链上无需预先校准即可安装传感器，实现大面积覆盖并捕获周围环境的点云；在实验中使用了 Franka Research 3 机器人链的 ToF 传感器来验证方法。

**💡 创新点**

创新点在于：1）利用 CAD + Blender 进行程序化生成的皮肤，能够精确贴合任意非可展开曲面；2）在数字设计阶段就嵌入可自定义大小的 PCB 夹具，实现无需胶水的 snap‑in 传感器安装；3）采用 Poisson disk 采样分布传感器，最大化覆盖率；4）整个流程可重复、快速、低成本。

**🔧 技术方法**

使用了 CAD 与 Blender 进行程序化建模、Poisson disk 采样、FDM 3D 打印、SparkFun VL53L5CX ToF 传感器、ESP32‑C6 微控制器、I²C 多路复用器、ROS+UDP 通讯以及点云重建算法。

**📊 数据集**

未使用公开数据集，而是通过在 FR3 机器人上布置 ToF 传感器获取真实距离数据，并以此生成点云进行实验验证。

**📈 对比分析**

通过将 ToF 传感器得到的距离转化为局部坐标系并投影到点云，直接与真实物体位置对比，展示了手部、拳头、卷轴等物体的清晰重构；实验显示能够捕获至 3.5 m 远的物体，且多次拆装后无需重新校准，表明方法具有良好的实用性。

**⚠️ 局限性**

局限性包括：仅在单一机器人链部位进行了验证，未对全机覆盖的精度进行量化；点云重建精度未给出具体误差指标；仅使用 ToF 传感器，测距范围受限；未考虑高遮挡或复杂曲率对传感器布置的影响。

---

## 146. HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation

**arXiv ID:** 2603.23871 | [PDF](https://arxiv.org/pdf/2603.23871v1)

**作者:** Ken Ding `[一作]` `[通讯]` (NVIDIA), Ken Ding (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Hybrid Distillation Policy Optimization (HDPO)，在 RL 无梯度的“cliff”提示上使用模型自身的 privileged self‑distillation 产生学习信号

**💡 创新点**

创新点在于同模型 privileged distillation 只需加一条前向传递与 JSD 损失，理论上证明 realizability gap 更小，且 R=1 过滤可恢复 KL‑regularized RL 最优策略

**🔧 技术方法**

采用 GRPO、JSD 目标、top‑k 取值 64、vLLM 生成、hf_math_verify 验证等技术

**📊 数据集**

使用 OpenMathInstruct‑2 数据集，基模型为 Qwen2.5‑Math‑1.5B‑Instruct

**📈 对比分析**

与 GRPO 基线对比，HDPO 在 pass@4、pass@8 上提升 1.1%–1.7%（λ=0.1）且可通过 λ 调节探索‑利用平衡，pass@1 维持或略降

**⚠️ 局限性**

局限性包括仅在 1.5B 模型与单一数据集上验证，计算开销增加，且效果随模型规模与奖励设计可能变化，需进一步研究大规模与多任务适用性

---

## 147. MLE-UVAD: Minimal Latent Entropy Autoencoder for Fully Unsupervised Video Anomaly Detection

**arXiv ID:** 2603.23868 | [PDF](https://arxiv.org/pdf/2603.23868v1)

**作者:** Yuang Geng `[一作]`, Ivan Ruchkin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种完全无监督的单场景视频异常检测框架MLE-UVAD，直接在未标注的视频中训练并检测异常；

**💡 创新点**

创新点在于引入最小潜在熵(MLE)正则化，迫使异常潜在向量被压缩至正常簇，从而在重建误差上产生明显的分隔；

**🔧 技术方法**

采用卷积自编码器结合重建损失与MLE损失，并用Pearson相关系数(PCC)评估重建质量，辅以t‑SNE可视化和KDE估计潜在熵；

**📊 数据集**

使用了三大公开/自采数据集——Donkeycar（真实驾驶）、Corridor（真实监控）和UBnormal（合成3D）来验证方法；

**📈 对比分析**

与TMAE、GCL、Vanilla CAE等无监督基线以及半监督OCC方法对比，MLE-UVAD在所有数据集均实现≈1.0的AUC，显著优于其它无监督方案；

**⚠️ 局限性**

局限性包括：当异常比例超过约40%时性能下降；目前仅针对单摄像头单场景，缺乏多场景/多摄像头适配与时间建模能力。

---

## 148. An Adaptive Neuro-Fuzzy Blockchain-AI Framework for Secure and Intelligent FinTech Transactions

**arXiv ID:** 2603.23829 | [PDF](https://arxiv.org/pdf/2603.23829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 149. BeliefShift: Benchmarking Temporal Belief Consistency and Opinion Drift in LLM Agents

**arXiv ID:** 2603.23848 | [PDF](https://arxiv.org/pdf/2603.23848v1)

**作者:** Praveen Kumar Myakala `[一作]` (Independent AI Researcher), Rahul Manche `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个纵向会话基准BeliefShift，用以测量LLM在多轮对话中对用户信念随时间演化的跟踪与推理能力。

**💡 创新点**

首次将信念变化视为一等实体，提出四项量化指标（BRA、DCS、CRR、ESI）并设计多主题、多时序轨迹数据集。

**🔧 技术方法**

结合检索增强生成（RAG）、大规模预训练LLM（如GPT‑4o、Claude‑3.5 Sonnet、Gemini‑1.5 Pro等）以及自定义的Scaffolding Engine生成对话。

**📊 数据集**

构造了2400条跨10‑50次会话的多主题轨迹（健康、政治、个人价值、产品偏好），包含68,160次会话并手工标注信念状态向量。

**📈 对比分析**

在零射击和RAG两种设置下对七个LLM进行评测，结果显示存在“稳定‑适应”权衡，RAG提升BRA与CRR但对DCS影响有限；Claude‑3.5 Sonnet取得最高DCS，GPT‑4o取得最高BRA。

**⚠️ 局限性**

受限于合成轨迹、单用户视角、二元证据模型以及人工评测的CRR，未能覆盖真实用户长期交互的复杂性和多语言场景。

---

## 150. 3D-LLDM: Label-Guided 3D Latent Diffusion Model for Improving High-Resolution Synthetic MR Imaging in Hepatic Structure Segmentation

**arXiv ID:** 2603.23845 | [PDF](https://arxiv.org/pdf/2603.23845v1)

**作者:** Kyeonghun Kim `[一作]` (OUTTA), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**通讯引用:** 19720 | [OpenAlex ID](https://openalex.org/A5115593383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于标签引导的3D潜在扩散模型3D-LLDM，用于生成高分辨率的MR体积及其对应的解剖分割标签，并将合成数据用于训练和增强下游分割任务。

**💡 创新点**

创新点在于：①将分割标签生成与ControlNet结合，在3D潜在空间直接对标签进行扩散，引导体积合成实现跨平面的一致性；②采用真实标签而非合成标签来训练ControlNet，从而提升合成质量；③通过标签引导实现高质量解剖一致的合成MR体积。

**🔧 技术方法**

使用的技术包括3D潜在扩散模型、ControlNet、VAE编码器/解码器、MONAI深度学习框架、AdamW优化器以及3D ResNet-50提取特征用于FID评估。

**📊 数据集**

实验数据来自720例Gd-EOB-DTPA增强的肝胆期MRI（HCC患者），裁剪为固定尺寸(160,160,64)，划分为504/72/144的训练/验证/测试集。

**📈 对比分析**

比较方法：①使用Fréchet Inception Distance (FID) 与HA-GAN、3D-DDPM、3D-LDM等基线模型对比，3D-LLDM在所有视角的FID均最低（平均28.31，较HA-GAN降低70.9%，较最强扩散基线降低26.7%）；②在多种CNN分割网络（U-Net、ResUNet、WideResUNet、DynUNet、VNet）上对比仅用真实数据与真实+合成数据训练，合成数据提升平均Dice 8.874%（HCC）和8.591%（血管），单个网络最高提升11.153%。

**⚠️ 局限性**

局限性：模型仅在单一成像中心与单一模态（肝胆期MRI）上验证，缺乏跨中心、跨模态的泛化评估；合成标签与体积的病灶细节仍可能不足，无法完全替代真实数据，且对极端病灶形态的生成能力尚未充分验证。

---

## 151. VehicleMemBench: An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents

**arXiv ID:** 2603.23840 | [PDF](https://arxiv.org/pdf/2603.23840v1)

**作者:** Yuhao Chen `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4325 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了VehicleMemBench，一套可执行的多用户长期记忆基准，用来评估车载智能体在面对多用户偏好演化和工具交互时的记忆与决策能力。

**💡 创新点**

通过事件驱动的多用户偏好演化、可执行的车载仿真环境以及基于状态匹配的客观评估，填补了单用户静态QA基准无法捕捉的长期交互与动态偏好场景。

**🔧 技术方法**

结合LLM生成对话与事件链、时间序列事件调度、构建23类车载工具接口的仿真环境，并采用状态差异（ESM、F1等）进行评估。

**📊 数据集**

基于Persona‑Hub生成的人设数据，人工质量控制后构建50组三人组，随后通过LLM合成多条偏好事件链并生成对话，最终形成500个带有80+记忆事件的查询样本，已公开发布在huggingface和GitHub。

**📈 对比分析**

在金手指与自治记忆两种设置下对比七大LLM与五类记忆系统，发现金手指条件下性能可达90%+ESM，但在自主记忆时性能骤降，主流记忆系统多未超越简单递归摘要或键值存储，且记忆错误占大多数。

**⚠️ 局限性**

仅评估了有限的强模型，缺乏对更复杂长周期交互的覆盖，且基准设计尚未扩展到更广泛的场景与模型类型。

---

## 152. Perturbation: A simple and efficient adversarial tracer for representation learning in language models

**arXiv ID:** 2603.23821 | [PDF](https://arxiv.org/pdf/2603.23821v1)

**作者:** Joshua Rozner `[一作]` (Stanford University), Cory Shain `[通讯]` (Stanford University)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5033058937)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于单一对抗样本的微调（perturbation）方法，用以探测语言模型内部的语言学表示；

**💡 创新点**

创新点在于该方法不依赖几何假设、无需额外标签，仅通过极小的微调即可捕获模型中的语义、形态和句法结构；

**🔧 技术方法**

技术核心是使用remapping对模型进行单例微调并在其他句子上测量log‑odds转移，结合clusterability评估；

**📊 数据集**

实验使用BATS（形态学）、CoarseWSD‑20（词义）以及专门构造的填空‑缺口（filler‑gap）数据集；

**📈 对比分析**

与传统的cosine相似度或DAS方法相比，perturbation在聚类AUC上与或优于最佳层，并在未训练模型中几乎不产生结构，表明更高的选择性与解释力；

**⚠️ 局限性**

局限性包括对remapping设计的依赖、tokenization影响、无法处理带双向上下文的自回归模型、仅针对英语、未提供可操控的机制以及对非语言知识的适用性尚未验证。

---

## 153. General Intellectual Humility Is Malleable Through AI-Mediated Reflective Dialogue

**arXiv ID:** 2603.23855 | [PDF](https://arxiv.org/pdf/2603.23855v1)

**作者:** Mohammad Ratul Mahjabin `[一作]` (University of South Florida), Raiyan Abdul Baten `[通讯]` (University of South Florida)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5038144794)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过在大型语言模型（LLM）中实现的结构化对话，测试了使用布鲁姆认知层级和苏格拉底式反思来提升一般知识谦逊（GIH）的可行性。

**💡 创新点**

创新点在于首次将布鲁姆认知框架与苏格拉底式提问结合，利用LLM自动化提供个性化的、分层的反思性对话，从而实验证明GIH具有可塑性。

**🔧 技术方法**

技术手段主要是使用 GPT‑4.1 进行文本交互式对话，构建了基于认知层级与提问的对话脚本，完成对参与者的指导和提问。

**📊 数据集**

数据集为 400 名来自 Prolific 的美国成年人，使用了已验证的六项 Likert‑量表测量 GIH，并在三时间点（基线、立即后测、14 天随访）收集评分。

**📈 对比分析**

与时间匹配的对照对话相比，处理组在即时和 14 天后测均显示显著提升（平均差 0.19 分，Hedges’ g ≈ 0.46），并且效果在两周内保持不衰，证明了该方法在提升 GIH 上的有效性。

**⚠️ 局限性**

局限包括样本仅来自美国，未检验跨文化适用性；随访仅限 14 天，无法评估长期持久性；以及未进行组件消融分析以明确布鲁姆层级和苏格拉底提问的各自贡献。

---

## 154. APISENSOR: Robust Discovery of Web API from Runtime Traffic Logs

**arXiv ID:** 2603.23852 | [PDF](https://arxiv.org/pdf/2603.23852v1)

**作者:** Yanjing Yang `[一作]` (Nanjing University), Bohan Liu `[通讯]` (Nanjing University)

**通讯引用:** 1876 | [OpenAlex ID](https://openalex.org/A5027086618)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种黑盒API发现框架，能够在混合运行时流量和噪声环境下，无需标注即可恢复 Web API 接口。

**💡 创新点**

创新点包括：① 多信号去噪与路径规范化技术，显著降低非API请求干扰；② 基于 Drain3 的结构模板挖掘，统一抽象动态路径；③ 图结构表示学习（DAEGC）对同一模板内部语义差异进行细粒度聚类，提升精度与鲁棒性。

**🔧 技术方法**

核心技术：HTTP流量预处理（静态过滤 + 逻辑阈值）、路径规范化、Drain3 前缀树结构模板抽取、基于语义相似图的深度聚类（DAEGC）以及必要时回退到 K‑means。

**📊 数据集**

使用了 10,000+ 运行时请求日志，覆盖六个开源 Web 应用（Train‑Ticket、HumHub、Memos、Overleaf、Nextcloud、Dify），并人工标注 199 条有效可调用 API 端点。

**📈 对比分析**

与 Optic、Mitmproxy2Swagger、LogCluster、LogNgram、UniParser、LogPPT、WebAPI Search、APID2Spec、APICARV、APIDrain3 等 10 种基线对比，平均 PGA 95.92%、FGA 94.91%，在不同噪声（Lexify/Interfere）下表现最稳健，误报率低、性能方差最小。

**⚠️ 局限性**

局限性在于依赖手动/半自动的运行时交互，无法覆盖所有可能的 API 路径；缺少系统化的自动交互或测试驱动生成，导致部分罕见或受权限限制的接口可能未被发现。

---

## 155. A Measurement-Calibrated AI-Assisted Digital Twin for Terahertz Wireless Data Centers

**arXiv ID:** 2603.23837 | [PDF](https://arxiv.org/pdf/2603.23837v1)

**作者:** Mingjie Zhu `[一作]`, Chong Han `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在300 GHz下完成了数据中心的VNA通道测量，并构建了基于测量校准的射线追踪模型和RT‑条件隐式神经场（INF）的数字孪生，用于精确预测THz无线信道、覆盖与干扰。

**💡 创新点**

创新点在于将实验测量、射线追踪与深度隐式神经场三种方法紧密耦合，形成连续、物理一致的信道表示，能补偿射线追踪在NLoS区域的不足，并实现从稀疏测量到全空间的无缝插值。

**🔧 技术方法**

使用了VNA通道测量、Sionna射线追踪、RT‑条件隐式神经场（INF）以及对射线追踪结果的校准与权重匹配技术。

**📊 数据集**

使用的测量数据集包含29个Tx‑Rx位置、290‑310 GHz带宽20 GHz的向量网络分析仪频率响应，涵盖LoS与NLoS的机架间与AP‑机架链路。

**📈 对比分析**

与单纯射线追踪和传统ABG模型比较，INF‑数字孪生在NLoS区提供更平滑、准确的功率映射；覆盖率分析显示AP部署可覆盖95%以上位置，而机架级Tx仅70%，证明了该方法在系统层面决策中的显著性能提升。

**⚠️ 局限性**

局限性包括依赖高质量测量与RT校准；对不同物理环境的迁移性未知；训练INF模型需要额外计算资源；模型可能无法充分捕捉极端多路径或极端阻塞情况。

---

## 156. CodeExemplar: Example-Based Scaffolding for Introductory Programming in the GenAI Era

**arXiv ID:** 2603.23830 | [PDF](https://arxiv.org/pdf/2603.23830v1)

**作者:** Boxuan Ma `[一作]` (Kyushu University), Shinichi Konomi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CodeExemplar，利用生成式 AI 提供结构相似但表面不同的示例代码，帮助初学者通过类比推理解决编程任务。

**💡 创新点**

创新点在于将生成式 AI 用作“示例生成器”而非直接给出答案，基于结构与表面相似度的双维分类，促进类比迁移且降低抄袭风险。

**🔧 技术方法**

使用 GPT‑5.2 等大型语言模型生成 scaffold 示例，并结合自动评测系统和前端交互界面。

**📊 数据集**

使用课堂练习任务（30 名学生的 3 个入门编程题目）及教师审核后生成的示例数据，未使用公开大规模编程数据集。

**📈 对比分析**

通过学生问卷和课堂反馈进行主观评估，学生对示例的整体满意度平均评分 3.76/5，认为能有效促进任务进展，但未提供客观学习成绩对比。

**⚠️ 局限性**

局限包括样本量有限、缺乏长期学习效果评估、示例可能导致过度依赖以及生成质量受模型限制，且示例多样性不足。

---

## 157. AI Fortune-Teller: Juxtaposing Shaman and AI to Reveal Human Agency in the Age of AI

**arXiv ID:** 2603.23811 | [PDF](https://arxiv.org/pdf/2603.23811v1)

**作者:** Soonho Kwon `[一作]` (Georgia Institute of Technology), Younah Kang `[通讯]` (Yonsei University)

**通讯引用:** 1994 | [OpenAlex ID](https://openalex.org/A5088085186)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实验了一个欺骗性聊天机器人，将传统巫师的占卜结果呈现为AI职业咨询。

**💡 创新点**

创新点在于将AI与巫术进行对比，探讨在AI时代人类能动性与信任关系。

**🔧 技术方法**

利用ChatGPT对巫师占卜文本进行润色，使其看似AI生成；实验平台为Facebook Messenger。

**📊 数据集**

数据来源为参与者提供的个人信息（姓名、出生日期、照片、职业问题）和巫师的占卜笔记。

**📈 对比分析**

无定量性能评估，采用事后访谈和对话记录的质性分析；发现参与者对建议的态度未随源头改变。

**⚠️ 局限性**

局限在样本量小（7人）、仅为单向实验、缺乏客观指标，且可能受文化背景影响。

---

## 158. Deep Neural Regression Collapse

**arXiv ID:** 2603.23805 | [PDF](https://arxiv.org/pdf/2603.23805v1)

**作者:** Akshay Rangamani `[一作]` (New Jersey Institute of Technology), Altay Unal `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了深度回归模型中的神经回归崩塌（Deep Neural Regression Collapse）现象，并给出四个衡量指标。

**💡 创新点**

创新点是将神经分类崩塌的四条条件推广到所有层并证明其在回归任务中出现，同时揭示权重衰减对崩塌的必要性。

**🔧 技术方法**

采用了中心化核对齐（CKA）、奇异值分解、线性伪逆、稳定秩等技术对特征、权重与目标进行对齐与低秩性评估。

**📊 数据集**

实验使用合成低秩数据、MuJoCo 模拟器的 Swimmer/Reacher/Hopper、Carla2D、UTKFace、SGEMM 等数据集。

**📈 对比分析**

通过对比不同层的 NRC 指标与模型整体 MSE，发现崩塌层的线性可预测误差与全模型误差相近，且权重衰减可显著提升崩塌程度，性能优于未使用权重衰减的对照。

**⚠️ 局限性**

局限在于缺乏对深度 NRC 的理论证明，且高权重衰减会导致欠拟合，未探究不同任务和损失函数下的普适性。

---

## 159. SiftMoE: Similarity-Aware Energy-Efficient Expert Selection for Wireless Distributed MoE Inference

**arXiv ID:** 2603.23888 | [PDF](https://arxiv.org/pdf/2603.23888v1)

**作者:** Qian Chen `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 21450 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SiftMoE框架，利用专家相似性在无线分布式MoE推理中智能地替换或跳过专家，以降低通信能耗；

**💡 创新点**

通过理论误差分析给出专家替换/跳过导致的精度下降上界，基于此构建能耗最小化模型并求解最优专家选择策略；

**🔧 技术方法**

MoE架构、理论误差界定、能耗优化（线性规划/动态规划）、快速与慢衰落下的通道建模与自适应比特分配；

**📊 数据集**

使用switch‑base‑8（XSum数据集）和Mixtral‑8×7B（CommonsenseQA数据集）进行实验验证；

**📈 对比分析**

与传统Top‑K路由方案比较，SiftMoE在保持可控精度损失的前提下，能耗下降20%–50%（慢衰落）或更高（快衰落），同时精度更稳定；

**⚠️ 局限性**

仅考虑单用户场景；仅针对层内专家替换/跳过，未探索层级跳过；未研究多用户竞争与协同；未结合射频层面技术如空中计算等。

---

## 160. VILLA: Versatile Information Retrieval From Scientific Literature Using Large LAnguage Models

**arXiv ID:** 2603.23849 | [PDF](https://arxiv.org/pdf/2603.23849v1)

**作者:** Blessy Antony `[一作]` (Virginia Tech), T. M. Murali `[通讯]` (Virginia Tech)

**通讯引用:** 4832 | [OpenAlex ID](https://openalex.org/A5073142937)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种针对流感A病毒蛋白突变信息提取的开放式科学信息抽取（SIE）任务，并提出了一种两阶段检索增强生成（RAG）框架 VILLA。

**💡 创新点**

创新点在于：①提出开放式、无标签的突变抽取任务；②设计两步检索（先按摘要筛选相关论文，再按全文块检索细节）的 RAG 框架；③手工构建 629 条突变与 239 篇文献的高质量真值数据集，用于训练与评估。

**🔧 技术方法**

使用的大技术包括：大语言模型（如 Qwen3-Next-80B-A3B-Instruct、Llama 3.1:8B 等）作为生成器；多种嵌入模型（Qwen3-Embedding:8B、PubMedBERT 等）作为检索器；RAG 框架与三种主流 SIE 工具（OpenScholar、PaperQA2、HiPerRAG）进行对比。

**📊 数据集**

构建的基准数据集：239 篇流感A病毒相关论文，涵盖 10 个蛋白的 629 条突变信息，用作真值集合。

**📈 对比分析**

与零射击提示、单层 RAG（仅摘要或全文）以及三种业界 SIE 工具进行对比。VILLA 在 F1‑score 上达到 0.53±0.13，显著高于零射击（≤0.10）、单层 RAG（≤0.41）和三者 SOTA（≤0.07）。在召回率与精确度上也均优于对照方法。

**⚠️ 局限性**

局限性包括：① 仍然存在较低的召回率，尤其是对全篇文本检索的精度；② 依赖人工构建的真值集，难以推广到无真值的领域；③ 需要多次调用生成器，计算成本高；④ 评价框架中专家主观性导致定性与定量指标不完全一致。

---

## 161. POSIM: A Multi-Agent Simulation Framework for Social Media Public Opinion Evolution and Governance

**arXiv ID:** 2603.23884 | [PDF](https://arxiv.org/pdf/2603.23884v1)

**作者:** Yongmao Zhang `[一作]` (Information Engineering University), Bin Yan `[通讯]` (Information Engineering University)

**通讯引用:** 6543 | [OpenAlex ID](https://openalex.org/A5051440995)

**关键词:** `aaff19cd-e89f-4398-8dae-a6684a329811` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 POSIM，一个基于大型语言模型的社交媒体舆情模拟框架，使用多代理 BDI 认知架构与 Hawkes 时序引擎生成个体与群体的非理性行为与多阶段演化。

**💡 创新点**

创新点在于将 LLM 嵌入分层 BDI 认知模型并显式建模情绪、认知偏差，实现可解释的舆情非理性行为仿真，并通过三层机制–现象–统计验证体系全面检验可信度。

**🔧 技术方法**

技术手段包括 LLM（如 Qwen）、Belief–Desire–Intention 分层推理、情绪与偏差动态更新、链式思考、语义相似度推荐、Hawkes 自激点过程时序调度及 LLM 调度池。

**📊 数据集**

使用了来自微博的三个真实舆情事件数据集（奢华耳饰、武汉大学图书馆骚扰、Xibei 准备食品），覆盖原帖、转发与评论全链。

**📈 对比分析**

与规则式 ABM、直接 LLM 与单体 Chain‑of‑Thought 三种基线在行为、内容、网络层面的九项统计指标对比，POSIM 在所有指标上平均提升约 5–13%，并在行为热度曲线、情绪极化与传播链条等方面与真实数据高度一致。

**⚠️ 局限性**

局限性包括：受 LLM 推理成本与可扩展性限制，关注关系仍为静态未实现动态跟随，框架仅在微博平台验证，跨平台与跨文化推广需进一步研究。

---

## 162. BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment

**arXiv ID:** 2603.23883 | [PDF](https://arxiv.org/pdf/2603.23883v1)

**作者:** Risa Shinoda `[一作]` (University of Osaka), Fumio Okura `[通讯]` (University of Osaka)

**通讯引用:** 1076 | [OpenAlex ID](https://openalex.org/A5069226668)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了0.95BioVITA框架，构建百万规模的音频、图像与文本三模态训练集，并通过两阶段对比学习实现统一的视觉-文本-音频表示，随后在跨模态检索和生态特征预测任务上进行评估。

**💡 创新点**

创新点在于：①首次构建规模最大的三模态生物学数据集；②提出两阶段训练策略，将音频对齐到预训练的BioCLIP视觉‑文本空间，实现音频、图像和文本的全域统一；③设计了全面的跨模态检索基准，覆盖所有六个检索方向和三个分类层级。

**🔧 技术方法**

技术手段包括：使用CLIP/BioCLIP2的视觉‑文本编码器；HTS‑AT音频编码器；对比损失（音频‑文本ATC、音频‑图像AIC、图像‑文本ITC）；两阶段训练（先音频‑文本再联合音频、图像、文本）；多样化的文本提示模板与温度调节。

**📊 数据集**

数据集：0.95BioVITA‑train（1.3M音频、2.3M图像，涵盖14,133种，34个生态特征）来自iNaturalist、Xeno‑Canto、Animal Sound Archive；以及0.95BioVITA‑bench作为检索基准，包含从iNaturalist和ToL‑200M等来源的独立图像、音频与文本样本。

**📈 对比分析**

与CLIP、CLAP、ImageBind、BioCLIP2、TaxaBind等基线模型对比，0.95BioVITA在所有六个检索方向的Top‑1/Top‑5准确率均高于对手，平均Top‑1达到约71.7%（图像→音频等），Top‑5约89.2%；在未见物种上仍能获得约51.9% Top‑1、73.0% Top‑5，显著优于现有三模态基线。

**⚠️ 局限性**

局限性包括：在更高层级（属、科）检索时性能下降；对低质量或噪声较大的音频鲁棒性不足；稀有物种的泛化能力仍有限；基准设计可能未完全避免数据泄漏，且对不同文化命名的依赖需进一步研究。

---

## 163. EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual-Group Interaction

**arXiv ID:** 2603.23874 | [PDF](https://arxiv.org/pdf/2603.23874v1)

**作者:** Bingxue Zhao `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21487 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 EnvSocial-Diff，融合环境条件与个人-群体交互的扩散式社交力模型，用于预测人群轨迹。

**💡 创新点**

创新点在于显式结构化环境编码（障碍物、感兴趣物体、照明）和多层次个体-群体交互模块。

**🔧 技术方法**

使用扩散模型、图神经网络、ResNet-50+BERT 视觉-文本特征、LSTM 历史编码器等。

**📊 数据集**

使用 GC 和 UCY 两个公开人群数据集。

**📈 对比分析**

与传统物理、数据驱动和物理信息化方法对比，显著提升 MAE/OT/FDE 等指标，尤其在 UCY 外部场景长时域性能最优。

**⚠️ 局限性**

局限在对光照在户外环境中的相关性不强，对复杂动态障碍物和实时更新的环境信息缺乏适应。

---

## 164. Generative AI User Experience: Developing Human--AI Epistemic Partnership

**arXiv ID:** 2603.23863 | [PDF](https://arxiv.org/pdf/2603.23863v1)

**作者:** Xiaoming Zhai `[一作]` (University of Georgia), Xiaoming Zhai `[通讯]` (University of Georgia)

**通讯引用:** 5056 | [OpenAlex ID](https://openalex.org/A5013379229)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Human–AI Epistemic Partnership Theory（HAEPT），将生成式人工智能（GenAI）在教育中的用户体验从传统工具使用视角转为认知伙伴关系，并用三个合约（Epistemic、Agency、Accountability）阐释其动态交互。该论文通过文献综述、理论构建以及两个案例（CLAIS与ArgueAgent）展示该理论的适用与意义。

**💡 创新点**

创新点在于：①将GenAI用户体验定义为人类–AI的知识合作关系；②提出三合约架构，系统化解释信任、认知分工与责任归属；③引入“校准循环”与“伙伴模式”概念，解释体验随时间演化的机制；④为后续实证与设计提供统一的框架和术语。

**🔧 技术方法**

技术方法主要为：文献综述、理论建模与案例分析；并未使用具体机器学习算法或编程实现。

**📊 数据集**

未使用新的数据集；引用了已有的教育技术与人工智能研究（如学生与教师对ChatGPT的调查、学术诚信研究等）作为理论与案例的支持。

**📈 对比分析**

作为理论性论文，未进行实验比较或性能评估；仅通过对比传统EdTech与GenAI在三合约上的差异，说明该框架对解释用户体验的优势；若涉及案例时，亦为示例性阐释而非量化性能对比。

**⚠️ 局限性**

局限性包括：①缺乏大规模实证验证，理论的可操作性与普适性仍待检验；②未建立统一的测量工具来量化三合约；③在不同学科、年级或文化背景下的适用性需要进一步探讨；④未考虑技术实现层面的细节，如系统透明度、可解释性等。

---

## 165. Deep Convolutional Neural Networks for predicting highest priority functional group in organic molecules

**arXiv ID:** 2603.23862 | [PDF](https://arxiv.org/pdf/2603.23862v1)

**作者:** Kunal Khatri `[一作]` (Dhirubhai Ambani Institute of Information and Communication Technology), Vineet Mehta `[通讯]` (Dhirubhai Ambani Institute of Information and Communication Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用深度卷积神经网络对有机分子FTIR光谱进行分析，预测其最高优先级的功能团。

**💡 创新点**

首次构建大规模FTIR光谱数据集，并采用深度CNN模型对功能团优先级进行预测，显著优于传统SVM方法。

**🔧 技术方法**

使用深度卷积神经网络（含卷积层、ReLU激活、最大池化、Dropout和Batch‑Norm）进行一维序列学习。

**📊 数据集**

采用SDBS公开数据库中的4730条有机化合物FTIR光谱，提取4000–1400 cm⁻¹区间的404点采样序列。

**📈 对比分析**

通过10折交叉验证与SVM对比，CNN在全数据集上的准确率达78.4%（SVM为68.8%），并在Top‑K评估中表现更优。

**⚠️ 局限性**

主要局限包括类别不平衡、对光谱预处理的依赖、模型训练成本高以及对不同来源数据的泛化能力仍需验证。

---

## 166. An Invariant Compiler for Neural ODEs in AI-Accelerated Scientific Simulation

**arXiv ID:** 2603.23861 | [PDF](https://arxiv.org/pdf/2603.23861v1)

**作者:** Fangzhou Yu `[一作]` (Virginia Tech), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**通讯引用:** 10927 | [OpenAlex ID](https://openalex.org/A5035052603)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Invariant Compiler框架，利用LLM驱动的编译流程将神经ODE的向量场重写为结构保持的形式，保证所有物理不变量（如守恒量、几何约束、泊松/ GENERIC结构等）在连续时间内严格满足。

**💡 创新点**

创新点在于将不变量视为第一类类型，通过几何中间表示与结构保持向量场构造实现硬约束，并统一多种已知和未知不变量的组合，提供了可扩展、可组合的设计模式，显著提升了模型的物理一致性与预测性能。

**🔧 技术方法**

核心技术包括：LLM程序合成（生成PyTorch代码）、几何中间表示（如球面、切空间投影、Cholesky因子化）、结构保持向量场（skew-symmetric、Poisson、Port-Hamiltonian、GENERIC、null‑space、投影）以及可训练的子模块（MLP、INN、Cholesky网络）。

**📊 数据集**

在多种科学系统上验证：SIR/SEIR、NOx/化学反应网络、Lorentz锥、Lotka‑Volterra、弹簧-阻尼器、热力学系统、扩展摆、两体引力等，利用对应的实验数据或合成轨迹集。

**📈 对比分析**

与无约束Neural ODE、软约束（Penalty）以及PINNs式软约束基线对比；结果显示硬约束模型在约束满足（10^−6水平）、轨迹MSE（训练/外推）和物理量守恒方面普遍优于基线，尤其在长时延外推（10×）时提升10–100×，且多不变量组合仍保持高精度。

**⚠️ 局限性**

局限包括：仍需人工指定或发现不变量，无法直接处理不等式/接触约束、离散事件；对高维/大规模系统的可扩展性与计算效率待进一步研究；LLM生成代码的可解释性与可复现性仍有改进空间。

---

## 167. Why the Maximum Second Derivative of Activations Matters for Adversarial Robustness

**arXiv ID:** 2603.23860 | [PDF](https://arxiv.org/pdf/2603.23860v1)

**作者:** Yunrui Yu `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 67519 | [OpenAlex ID](https://openalex.org/A5115666530)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了激活函数最大二阶导数对对抗鲁棒性的影响，并通过可调曲率激活函数族系统评估曲率与鲁棒性的关系。

**💡 创新点**

提出并验证了激活曲率最大值在4到10范围内最优的非单调关系，并给出了将激活二阶导数与损失Hessian对角线关联的理论解释。

**🔧 技术方法**

采用递归可调曲率激活族（RCT-AF）、Gauss‑Newton分解、Hessian对角线L2范数分析，并使用AutoAttack进行鲁棒性评估。

**📊 数据集**

在CIFAR‑10和CIFAR‑100数据集上，以ResNet‑18和WideResNet‑28‑10为模型进行实验。

**📈 对比分析**

通过与DAJAT、DKL、TRADES等对抗训练方法以及GELU、Swish等常用激活函数对比，发现最大曲率约为7时鲁棒率可提升约3%，显著优于传统激活函数。

**⚠️ 局限性**

实验仅在单一数据集与网络架构下进行单次训练，受随机性影响；研究聚焦于二阶曲率，未验证更高阶激活或更大规模模型的通用性。

---

## 168. Language Model Planners do not Scale, but do Formalizers?

**arXiv ID:** 2603.23844 | [PDF](https://arxiv.org/pdf/2603.23844v1)

**作者:** Owen Jiang `[一作]` (Drexel University), Li Zhang `[通讯]` (Drexel University)

**通讯引用:** 31175 | [OpenAlex ID](https://openalex.org/A5100388599)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了LLM在规划任务中的两种范式——直接作为planner和作为formalizer，并提出分治formalizer和高阶formalizer两种技术以提升在大规模BlocksWorld问题上的可扩展性。

**💡 创新点**

创新点包括：①引入分治生成策略缓解上下文溢出；②构造unraveling问题揭示自然语言压缩对formalizer的挑战；③提出高阶formalizer，即让LLM生成程序生成PDDL，从而将token数量降到与输入文本线性相关。

**🔧 技术方法**

使用了LLM-as-formalizer流水线、分治生成、LLM-as-higher-order-formalizer、PDDL规划器、VAL验证器，以及对模型的提示工程与上下文管理技术。

**📊 数据集**

使用了BlocksWorld-XXL（200个5-100块的问题）和BlocksWorld-Unravel（200个压缩描述的高复杂度问题）两套数据集。

**📈 对比分析**

对比方法：在BlocksWorld-XXL上评估planner和formalizer的计划准确率；分治formalizer在100块时使弱模型Qwen2.5保持100%；在BlocksWorld-Unravel上，planner几乎失效，formalizer性能骤降，而高阶formalizer显著恢复性能，Qwen2.5提升至约90%以上。

**⚠️ 局限性**

局限性：实验仅在BlocksWorld域进行，未覆盖更复杂语义、谓词多元或约束丰富的规划域；模型数量有限，未来需扩展更多模型和域进行验证。

---

## 169. PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay

**arXiv ID:** 2603.23841 | [PDF](https://arxiv.org/pdf/2603.23841v1)

**作者:** Rohan Khetan `[一作]` (Northville High School), Ashna Khetan `[通讯]` (Stanford University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5093278370)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 PoliticsBench 框架，利用多轮角色扮演和自我反思评估 LLM 的政治价值倾向；

**💡 创新点**

创新点在于将心理测量工具 EQ‑Bench 与政治场景相结合，生成20个四阶段情境，评估十项左右政治特征，而非单一问答的粗粒度方法；

**🔧 技术方法**

技术主要包括基于大语言模型（Claude、GPT、Gemini、Grok、Llama、Deepseek、Qwen 等）的多轮对话生成，利用裁判模型（Claude Sonnet 3.7）进行评分，并通过权重归一化计算整体政治倾向；

**📊 数据集**

数据集为自生成的 20 个政治情境（来源于 EQ‑Bench、SpiralBench 等心理学基准），并使用 Llama‑Guard 过滤安全性；

**📈 对比分析**

对比方法是对八个模型在 10 个政治特征上打分，平均整体对齐分数与置信区间，结果显示七个模型均呈显著左倾，Grok 为右倾，整体偏差在 ±30 以内；

**⚠️ 局限性**

局限性包括裁判模型本身偏左导致评分偏差、仅使用单一裁判模型、情境可能受框架偏见影响、以及对立场多样性与自由度的覆盖不足。

---

## 170. Unveiling Hidden Convexity in Deep Learning: a Sparse Signal Processing Perspective

**arXiv ID:** 2603.23831 | [PDF](https://arxiv.org/pdf/2603.23831v1)

**作者:** Emi Zeger `[一作]` (Stanford University), Mert Pilanci `[通讯]` (Stanford University)

**通讯引用:** 1170 | [OpenAlex ID](https://openalex.org/A5001436196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

将ReLU网络的训练问题转化为凸优化问题，并展示多种可实现的凸化方法，尤其强调与稀疏信号处理和压缩感知的深度关联。

**💡 创新点**

发现并利用深度网络隐藏的凸性，给出两层ReLU网络等价于L1/组Lasso的凸程序，提出基于超平面排列、Zonotope、几何代数等几何视角的全局最优性理论，并对更深网络给出凸化框架。

**🔧 技术方法**

利用凸双对偶、超平面排列枚举、L1/组Lasso、核方法/NTK、量化与线性规划、几何代数（Wedge乘积）、随机采样等技术实现训练的凸化与可解释性。

**📊 数据集**

实验中使用MNIST、纽约证券交易所（NYSE）日志成交量数据、ECG 电压时间序列，以及对UCI公开数据集进行基准测试。

**📈 对比分析**

与传统SGD/Adam训练两层ReLU网络做对比；凸化方案在训练误差、测试MSE上更优且不受初始化影响；在NYSE数据上MSE更低，在ECG数据上训练损失和预测准确率更好；整体表现更快、更稳定且可获得全局最优证据。

**⚠️ 局限性**

对深层网络的凸化仍需极宽网络和大量激活模式，导致特征/变量数激增；计算复杂度在高维仍高；对一般网络结构的全局最优性和可扩展性尚未完全解决，需要进一步研究降低宽度和提升算法效率。

---

## 171. Circuit Complexity of Hierarchical Knowledge Tracing and Implications for Log-Precision Transformers

**arXiv ID:** 2603.23823 | [PDF](https://arxiv.org/pdf/2603.23823v1)

**作者:** Naiming Liu `[一作]` (Rice University), Shashank Sonkar `[通讯]` (University of Central Florida)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5028809416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了层级前置知识追踪的计算复杂性，并通过实验评估Transformer在递归多数树上的学习行为

**💡 创新点**

首次将层级前置推理任务映射到电路复杂度框架，揭示递归多数树属于NC¹，且在单调阈值电路上呈现深度层次化下界；同时提出结构化辅助监督提升Transformer表现

**🔧 技术方法**

电路复杂度分析（TC⁰、NC¹、Monotone Threshold Circuits），Transformer编码器实验，结构化分隔符编码与辅助损失

**📊 数据集**

自定义递归多数树数据（叶子为Bernoulli(0.5)二进制），以及对其分层、插入分隔符的变体

**📈 对比分析**

与仅使用全局求和的MLP基线以及oracle对比；在根标签监督下Transformer与MLP几乎相同；加入结构+辅助监督后，在深度3-4时接近100%准确；在深度6表现下降

**⚠️ 局限性**

实验受模型容量与训练预算限制；单纯结构标记不足以阻止捷径学习；对更深层级的学习成功与否仍不确定，难以得出Transformer本身的表示上限

---

## 172. How Vulnerable Are Edge LLMs?

**arXiv ID:** 2603.23822 | [PDF](https://arxiv.org/pdf/2603.23822v1)

**作者:** Ao Ding `[一作]` (China University of Geoscience Beijing), Ping Lu `[通讯]` (City University of Hong Kong)

**通讯引用:** 5893 | [OpenAlex ID](https://openalex.org/A5100606015)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究量化后在边缘设备部署的大语言模型的查询式知识提取风险，并提出一种结构化查询方法

**💡 创新点**

提出CLIQ（Clustered Instruction Querying）框架，通过语义聚类和聚类感知查询生成，显著提升在有限查询预算下的提取效率

**🔧 技术方法**

使用语义嵌入（Sentence‑BERT）进行聚类，利用强大LLM进行聚类条件指令生成，并通过学生模型训练评估提取效果

**📊 数据集**

以Qwen系列模型为教师与学生，使用GPTQ量化技术（INT8/INT4）进行实验

**📈 对比分析**

在相同查询预算（如1000个查询）下，将CLIQ与随机采样原始查询进行对比，CLIQ在BERT‑F1、BLEU、ROUGE等指标上提升约4–6个百分点，表现更优且收敛更快

**⚠️ 局限性**

在极低查询预算下效果有限，需依赖初始查询池的覆盖范围，且在极低精度（如INT4）下量化噪声仍限制提取质量

---

## 173. The Evolution of Decentralized Systems: From Gray's Framework to Blockchain and Beyond

**arXiv ID:** 2603.23819 | [PDF](https://arxiv.org/pdf/2603.23819v1)

**作者:** Zhongli Dong `[一作]` (University of Sydney), Albert Y. Zomaya `[通讯]` (University of Sydney)

**通讯引用:** 41236 | [OpenAlex ID](https://openalex.org/A5015993565)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

回顾并系统化了从James N. Gray 1986 年提出的去中心化架构框架到区块链、Web3 乃至未来 Web4 的演进路径，阐述了其核心四项原则（模块化、自治、完整性、标准化通信）在现代去中心化系统中的体现与实践；

**💡 创新点**

首次以历史视角将 Gray 的理论与区块链技术、Layer‑2 扩容方案、跨链互操作性以及 Web4 设想连接起来，提出将 AI、物联网与去中心化基础设施融合的 Web4 架构蓝图；

**🔧 技术方法**

综合分析了共识机制（PoW、PoS、BFT 等）、密码学基础（哈希、Merkle 树、ECDSA）、Layer‑2 方案（状态通道、侧链、Rollup）以及跨链协议（Atomic Swaps、Relay‑chain、IBC）等技术；

**📊 数据集**

无专门实验数据集，本论文为综述性工作；

**📈 对比分析**

无量化实验比较，主要通过对现有区块链与扩容技术的性能指标（TPS、延迟、能耗）进行文献对比与理论分析，指出公链仍落后于中心化支付网络；

**⚠️ 局限性**

缺乏对可扩展性、模块化层间接口安全、能源消耗、治理与合规性以及用户体验等方面的系统性解决方案，难以实现大规模商业化部署；

---

## 174. Aesthetics of Robot-Mediated Applied Drama: A Case Study on REMind

**arXiv ID:** 2603.23816 | [PDF](https://arxiv.org/pdf/2603.23816v1)

**作者:** Elaheh Sanoubari `[一作]` (University of Waterloo), Kerstin Dautenhahn `[通讯]` (University of Waterloo)

**通讯引用:** 26094 | [OpenAlex ID](https://openalex.org/A5059371010)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了REMind，一款基于机器人戏剧的反欺凌角色扮演游戏，利用机器人作为生命化木偶，帮助儿童练习旁观者干预和同伴支持。

**💡 创新点**

创新之处在于将社交机器人视为戏剧木偶，通过与戏剧导演、观众研究专家的合作，将美学、情感共振与教学目标分布在整个互动生态系统中，而非仅靠机器人表达能力。

**🔧 技术方法**

技术实现包括三台Furhat机器人、Microsoft Azure TTS（配合SSML）与自制的StorySync表格式脚本工具、Unreal Engine Live Link Face捕捉面部表情、GUI界面、沉浸式音视频提示，以及Wizard‑Joker双人调度框架。

**📊 数据集**

主要数据来源为儿童和教师的访谈、共创工作坊生成的故事板与角色设定，未使用公开标准数据集。

**📈 对比分析**

通过18名儿童的试玩测试，评估了自我效能提升和学习目标达成度，结果显示干预显著提高了学生的干预自信和反思深度；实验未提供机器人性能的量化基准，而是通过定性观察和学习成效评估来验证效果。

**⚠️ 局限性**

局限性包括社交机器人目前的情感表达受限，必须依赖人类调度员进行情境化调节；系统对场景和角色的适配性有限，缺乏通用化与可扩展性；以及缺乏大规模实验验证其长期教育效果。

---

## 175. Willful Disobedience: Automatically Detecting Failures in Agentic Traces

**arXiv ID:** 2603.23806 | [PDF](https://arxiv.org/pdf/2603.23806v1)

**作者:** Reshabh K Sharma `[一作]` (University of Washington), Benjamin Zorn `[通讯]` (Microsoft Research)

**通讯引用:** 5855 | [OpenAlex ID](https://openalex.org/A5113516138)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 “<name>”的AI工具，能够自动从代理系统的提示与工具架构中提取可检查的规范，并在多步对话执行历史（agentic traces）中逐步评估代理行为是否符合这些规范；

**💡 创新点**

创新点在于：①将PromptPex的规范提取思想扩展到多步代理交互，能够在整个执行轨迹上检测流程、工具调用和输出合规性；②构建三阶段流水线（导入、规范抽取、评估），实现大规模自动评估；③通过“门限最小化”聚合方式，确保关键违规不会被其他指标掩盖；

**🔧 技术方法**

技术包括：①大语言模型（LLM）用于规范抽取和评估推理；②自定义的评估器套件，涵盖输出、过渡、禁止边、参数、计划和最终状态等多维度；④标准化的Trace导入器，支持τ²-bench、OpenAI消息格式等多种格式；

**📊 数据集**

数据集为 τ²-bench 的多域代理对话（航空、零售、电信），共 424 条已清洗的执行轨迹（Claude 3.5 Sonnet 140 条、GPT‑4.1 144 条、o4‑mini 140 条）

**📈 对比分析**

与传统仅基于最终数据库状态和对话完整性的τ²分数对比：<name> 能在 83% 的完美 τ² 轨迹中发现至少一次流程违规；对三种模型的评估显示平均 aggregate 分数约 57‑63，显著低于 τ² 成功率（约 38‑41%），且能揭示模型特定的违规模式（如 Claude 的主体偏好、GPT‑4.1 的批量工具调用、o4‑mini 的身份验证缺失）。

**⚠️ 局限性**

局限包括：①仅在 τ²-bench 上验证，需在其他领域检验泛化；②评估成本高，依赖多次 LLM 调用；③规范抽取受限于提示的可表述性，可能漏判或误判；④当前仅检测违规，未提供自动修复或生成训练数据的机制。

---

## 176. A Reproducible Reality-to-VR Pipeline for Ecologically Valid Aging-in-Place Research

**arXiv ID:** 2603.23812 | [PDF](https://arxiv.org/pdf/2603.23812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 177. The DeepXube Software Package for Solving Pathfinding Problems with Learned Heuristic Functions and Search

**arXiv ID:** 2603.23873 | [PDF](https://arxiv.org/pdf/2603.23873v1)

**作者:** Forest Agostinelli `[一作]` (University of South Carolina), Forest Agostinelli `[通讯]` (University of South Carolina)

**通讯引用:** 1113 | [OpenAlex ID](https://openalex.org/A5005964283)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

开发了一个名为DeepCube的开源Python工具，自动化利用深度学习训练启发式函数来解决路径规划问题；

**💡 创新点**

创新之处在于将深度强化学习、监督学习与启发式搜索、批量加权A* / Q* 搜索、束搜索等多种算法结合，提供可插拔的 mixin 机制、自动并行化训练、可视化与命令行接口；

**🔧 技术方法**

采用PyTorch实现DNN启发式函数，使用值迭代、Q‑学习、限域贝尔曼学习、HER、批量加权搜索、束搜索等技术，并在CPU/GPU上并行采样、搜索与更新；

**📊 数据集**

通过随机游走在多种路径域中生成训练样本，涵盖数独、鲁比克方块、滑块拼图、化学反应路径、量子电路合成等示例，但未公开固定公开数据集；

**📈 对比分析**

在实验域上与传统手工设计的启发式、模式数据库和Fast Downward等规划器对比，DeepCube在搜索迭代次数、求解率和求解质量上均显著提升，尤其在GPU加速下批量搜索效率高；

**⚠️ 局限性**

局限包括对高维连续动作空间的支持有限、对复杂域的实例生成依赖随机游走、训练过程耗时且对目标网络同步敏感，以及当前策略网络训练功能尚未成熟。

---

## 178. Skewed Dual Normal Distribution Model: Predicting Touch Pointing Success Rates for Targets Near Screen Edges and Corners

**arXiv ID:** 2603.23865 | [PDF](https://arxiv.org/pdf/2603.23865v1)

**作者:** Nobuhito Kasahara `[一作]` (Meiji University), Homei Miyashita `[通讯]` (Meiji University)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5091629034)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了一种Skewed Dual Normal Distribution Model，用于预测触摸指点接近屏幕边缘和角落时的成功率。

**💡 创新点**

创新点在于将屏幕边缘导致的tap坐标分布偏斜纳入模型，扩展了Dual Gaussian Distribution Model，使其能覆盖靠近边缘/角落的目标并提供可解释的参数（如-skew区间）。

**🔧 技术方法**

使用统计学方法（偏态正态分布、Cox–Smith累积分布、回归分析）以及与传统Dual Gaussian、Lasso、Random Forest、SVR、MLP等机器学习模型进行对比。

**📊 数据集**

数据集来源于三项实验（水平/垂直一维指点和二维顶右角落指点），共收集约90k–170k有效点击，设备为Google Pixel 6a，实验包含不同目标尺寸、边缘距离等变量。

**📈 对比分析**

与传统Dual Gaussian模型相比，Skewed模型在所有实验中实现了更高的R²（大约0.95以上）并保持良好的泛化；机器学习模型在某些指标上略优，但差异不大。

**⚠️ 局限性**

局限性包括仅在单一无壳手机上测试、仅使用主指操作、未测量离屏错误、未考虑物理壳体或一只手拇指的交互，以及在顶边缘和角落存在非对称行为的情况。

---

## 179. SCoOP: Semantic Consistent Opinion Pooling for Uncertainty Quantification in Multiple Vision-Language Model Systems

**arXiv ID:** 2603.23853 | [PDF](https://arxiv.org/pdf/2603.23853v1)

**作者:** Chung-En Johnny Yu `[一作]` (University of West Florida), Nathaniel D. Bastian `[通讯]` (United States Military Academy)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5032194186)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练免费、基于熵权重线性意见池化的多视觉语言模型（VLM）系统不确定性量化框架 SCoOP，用于检测幻觉并实现可拒绝（abstention）决策。

**💡 创新点**

创新点在于首次将系统级不确定性量化与多模型答案聚合结合，利用各模型的熵作为权重，实现对多模型整体幻觉的精确检测和可靠拒绝，而非传统的多数投票或单一模型选择。

**🔧 技术方法**

采用熵加权线性意见池化、蒙特卡洛采样多轮回答、归一化熵作为系统不确定度指标，并通过统一类别空间聚合异构 VLM 的概率分布。

**📊 数据集**

使用 ScienceQA、MMMU 和 MMBench 三大多模态基准数据集进行评估。

**📈 对比分析**

与多数投票（Majority Voting）和无序选择（Naive Selection）等基线对比，SCoOP 在 AUROC 上提升约 10–13%，在 AURAC 上提升约 7–9%，且聚合时延仅为微秒级，保持甚至提升了整体准确率。

**⚠️ 局限性**

局限在于整体推理时间仍受各模型采样次数和模型数量的影响，聚合过程虽然轻量但需要多模型并行推理；在极低准确率场景下，仍需要更多鲁棒性验证。

---

## 180. Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training

**arXiv ID:** 2603.23885 | [PDF](https://arxiv.org/pdf/2603.23885v1)

**作者:** Gengluo Li `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 11309 | [OpenAlex ID](https://openalex.org/A5100783219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个端到端的文档解析框架DocHumming，结合现实场景合成（RSS）与结构感知训练（DATR），显著提升扫描文档与现场拍摄文档的解析准确性与鲁棒性。

**💡 创新点**

创新点包括：① 利用RSS生成大规模、多样化的全页合成数据；② 提出Document-Aware Training Recipe，包含渐进式训练和结构词加权损失，提升结构一致性与解码稳定性；③ 构建Wild-OmniDocBench评测基准，验证模型在真实拍摄环境下的鲁棒性。

**🔧 技术方法**

技术手段主要是基于InternVL2-1B多模态大型语言模型，采用自回归解码、结构标签、渐进式训练策略与结构词优化；合成管道通过布局模板、元素仓库、读取顺序约束及多种增强（几何、光照、相机、背景）实现真实感生成。

**📊 数据集**

使用的数据集包括：DocMix-3M（合成全页数据），OmniDocBench（标准扫描文档），XFUND（多语言表单），Wild-OmniDocBench（真实拍摄版），以及约10万张人工标注的扫描/数字文档。

**📈 对比分析**

与模块化工具、通用MLLM以及其他端到端模型对比，DocHumming在OmniDocBench和Wild-OmniDocBench上均取得最高整体准确率；文字、公式、表格和阅读顺序指标均优于基线；在Wild-OmniDocBench上的性能下降幅度最小，表现最稳健。

**⚠️ 局限性**

局限性包括：对高度非标准或交错排版的处理仍不够；超高分辨率页面需裁剪/切块导致信息缺失；推理速度约3秒/页，限制了交互式应用。

---

## 181. PowerFlow-DNN: Compiler-Directed Fine-Grained Power Orchestration for End-to-End Edge AI Inference

**arXiv ID:** 2603.23882 | [PDF](https://arxiv.org/pdf/2603.23882v1)

**作者:** Paul Chen `[一作]` (University of Southern California), Christopher Torng `[通讯]` (University of Southern California)

**通讯引用:** 531 | [OpenAlex ID](https://openalex.org/A5026966440)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了基于编译器的细粒度功率调度框架 PowerFlow‑DNN，用于在有限电压轨道和转换开销下，优化边缘 AI 推理的能耗并满足实时截止约束。

**💡 创新点**

将 DVFS 与功率门控联合建模为跨层功率调度优化问题，提出统一的离散化表述和 λ‑DP 动态规划求解器，并通过结构化剪枝实现对 10^160 级组合空间的可行探索，显示系统层级调度可实现比层级独立调度更高的能效。

**🔧 技术方法**

离散化电压轨道、动态功率门控、层级状态图与动态规划（Lagrangian λ‑DP）、结构剪枝与边际效用跳跃启发式；基于 TSMC 40nm 版图实现的 DNN 加速器进行验证。

**📊 数据集**

四种代表性边缘网络（如 SqueezeNet 等）以及对应的 INT8 权重/激活模型作为工作负载。

**📈 对比分析**

与无功率优化、仅门控、仅贪心 DVFS 及其组合的基线对比；PowerFlow 在满足截止约束的前提下，能耗仅偏差 0.68% 于 ILP 最优，节能率高达 37%（相较无优化基线），并能在 10^160 级组合空间下完成求解。

**⚠️ 局限性**

受限于离散化电压级和转换成本建模的精度；算法在极高精度或更复杂域特定场景下仍需更高级的搜索策略；未完整考虑功率管理对性能、热量等多维度的进一步影响。

---

## 182. ProcureGym: A Multi-Agent Markov Game Framework for Modeling National Volume-based Drug Procurement

**arXiv ID:** 2603.23880 | [PDF](https://arxiv.org/pdf/2603.23880v1)

**作者:** Jia Wang `[一作]` (Shanghai Innovation Institute), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5284 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于马尔可夫博弈的ProcureGym仿真平台，用于模拟中国国家药品集中采购（NVBP）的多轮竞标过程。

**💡 创新点**

创新点在于提出统一的多智能体接口，支持RL、LLM和规则基模型，并通过真实数据重现历史竞标结果，实现对政策与市场因素的可逆推理。

**🔧 技术方法**

采用马尔可夫决策过程（MDP）建模，并使用强化学习（IPPO、MAPPO）、大型语言模型（Qwen3）和基于规则的算法进行对比实验。

**📊 数据集**

使用7轮NVBP真实数据，共325种药品、2267家药企，涵盖成本、产能、竞标价格等属性。

**📈 对比分析**

对比方法包括价格预测精度（Spearman相关和R²）、获胜者预测准确率及企业利润分布；实验显示RL算法获得最高的预测准确率（≈75%）和最高利润，超过LLM（≈66%）和规则基（≈64%）。

**⚠️ 局限性**

局限性包括仅模拟药企竞标行为，未纳入政府和医疗机构的决策过程；模型主要针对中国NVBP，结果推广至其他采购场景需谨慎。

---

## 183. See, Remember, Explore: A Benchmark and Baselines for Streaming Spatial Reasoning

**arXiv ID:** 2603.23864 | [PDF](https://arxiv.org/pdf/2603.23864v1)

**作者:** Yuxi Wei `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**通讯引用:** 34313 | [OpenAlex ID](https://openalex.org/A5102498323)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向流式空间理解的S^3-Bench基准和AMF-VLM模型，实现对实时视频流的空间问答与主动探索；

**💡 创新点**

创新点包括：①将时间与空间约束融入问答，实现实时流式评测；②引入Memory Folding机制，将长序列压缩为结构化文本记忆；③构建双域（模拟+真实）数据，支持主动探索实验；④在模型中直接输出动作指令以主动获取缺失信息；

**🔧 技术方法**

技术手段包括：Qwen3‑VL‑8B视觉‑语言模型骨干，分层流式注意力与RoPE位置编码，KV缓存控制，Memory Folding的文本记忆与递归融合，主动探索动作预测（translate/rotate/scan），可选3D空间编码与StreamVGGT融合；

**📊 数据集**

使用的数据集为：S^3-Train（约600K条流式QA）、S^3-Eval（约20K QA，10K+场景）以及基于ScanNet、ScanNet++、ARKitScenes的真实视频子集；此外还用VSI590K进行预训练；

**📈 对比分析**

与随机、频繁、GPT‑5.2、Gemini、LLaVA、Qwen、InternVL、VLM‑3R、Cambrian‑S、Streaming‑VLM等多种基线在S^3‑Eval（Sim/Real）进行对比；AMF‑VLM在Sim上取得62.9%准确率、Real上57.8%，分别比基线提升约8.8%/13.3%；在VSI‑Bench、Blink、MMSI等标准空间基准中保持竞争性；在推理延迟上，AMF‑VLM在多帧输入时保持0.5s左右，远低于同基线Qwen3‑VL‑8B的8.56s；

**⚠️ 局限性**

局限性包括：仍依赖大型预训练VLM，难以彻底解决长序列外部噪声与稀疏信息；主动探索动作空间受限，无法覆盖所有真实交互情况；3D编码为可选，效果提升有限；评测基准基于人工标注，缺乏在真实机器人系统中的实证；对极端动态环境与传感器失效的鲁棒性尚待验证。

---

## 184. When AI output tips to bad but nobody notices: Legal implications of AI's mistakes

**arXiv ID:** 2603.23857 | [PDF](https://arxiv.org/pdf/2603.23857v1)

**作者:** Dylan J. Restrepo `[一作]` (Cornell University), Neil F. Johnson `[通讯]` (George Washington University)

**通讯引用:** 12196 | [OpenAlex ID](https://openalex.org/A5031168379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文探讨了生成式人工智能在法律实践中出现的虚假引用问题，并提出其并非随机的“幻觉”，而是Transformer模型自注意机制中的可计算“倾斜点”；

**💡 创新点**

创新点在于将Transformer的自注意映射为统计物理中的多自旋热系统，从而揭示AI在处理陌生法律问题时会跨越一个可预见的阈值导致输出从可靠推理转向虚假权威；

**🔧 技术方法**

主要使用了基于物理的Transformer理论、简化的单头注意力模型以及贪婪解码的能量最小化分析；

**📊 数据集**

由于研究为理论与模拟分析，未使用真实法律文献数据集，而是构造了四类内容向量（事实、合法应用、异常查询、虚假引用）进行实验；

**📈 对比分析**

文章未进行传统机器学习性能对比，而是通过数值推导展示了从A→B到B→D的两次倾斜点，证明了在一段正确输出后易出现重大虚假输出；

**⚠️ 局限性**

局限性包括模型过度简化（只考虑单一注意力头、贪婪解码）、忽略多层多头的交互以及对实际大型语言模型的可推广性需要进一步验证。

---

## 185. Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning

**arXiv ID:** 2603.23854 | [PDF](https://arxiv.org/pdf/2603.23854v1)

**作者:** Salah A Faroughi `[一作]` (University of Utah), Shirko Faroughi `[通讯]` (Urmia University of Technology)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5111782243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Symbolic‑KAN 架构，结合 Kolmogorov‑Arnold 网络与分层门控实现可训练的符号结构，从数据或物理约束中直接学习可解释的闭式表达式；

**💡 创新点**

创新点在于：①在网络内部嵌入离散符号结构，利用 Gumbel‑Softmax 和符号正则化逐步把连续混合精炼为一位一挑；②通过层级门控实现原语选择、投影边选择和单元剪枝，形成紧凑的符号网络；③既可作为可解释的模型，也可作为稀疏方程学习的先验原语库，弥补传统符号回归和稀疏回归的局限；

**🔧 技术方法**

采用 Kolmogorov‑Arnold 结构、可学习的单变量基函数库、Gumbel‑Softmax 门控、熵正则与非最大抑制、两阶段训练（软硬化）、物理信息损失（PDE 残差、边界/初始条件）、以及 L‑BFGS 微调等技术；

**📊 数据集**

主要使用合成数据集：一维非线性回归函数、Van‑der‑Pol 轨迹、1D 反应‑扩散 PDE、二维拉普拉斯 PDE 等，全部为人工生成；

**📈 对比分析**

与标准 KAN、cPIKAN、PINN、SINDy‑KAN 等方法对比。Metric 为相对误差/验证误差、参数估计误差；结果显示 Symbolic‑KAN 在相同网络规模下实现了 10⁻⁴ 级别的相对误差、参数误差 <1%，并显著优于对比基线（误差降低 70‑95% 左右）；

**⚠️ 局限性**

局限性：仍面临神经网络的外推与 OOD 泛化问题，硬化过程对温度/正则参数敏感，需手动调参；仅在先验原语库覆盖足够时才能准确发现结构，缺乏对未知原语的全局搜索能力；

---

## 186. Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation

**arXiv ID:** 2603.23838 | [PDF](https://arxiv.org/pdf/2603.23838v1)

**作者:** Han Zheng `[一作]` (Massachusetts Institute of Technology), Cathy Wu `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 47675 | [OpenAlex ID](https://openalex.org/A5053761444)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种混合框架 RL‑RH‑PP，用强化学习动态生成优先级顺序，结合滚动视野优先规划（Rolling Horizon Prioritized Planning）实现终身多机器人路径规划。

**💡 创新点**

创新点：①首次将强化学习与传统 PP 结合，解决终身 MAPF 的长周期依赖问题；②设计了 transformer‑style 编码器（时间‑空间注意力）提取 agents 路径的 spatiotemporal 关系；③采用自回归解码器生成多条候选优先级顺序，显著提升规划质量；④通过自定义奖励（距离、拥塞、不可行性）引导 RL 学习高效的全局优先级。

**🔧 技术方法**

技术细节：滚动视野规划（RH‑PP）、优先规划（PP）、深度强化学习（PPO）、Transformer‑style 编码器（时间/空间多头注意力）、自回归解码器、Top‑K 采样、奖励设计与可行性修复。

**📊 数据集**

使用两套基于真实仓库布局的模拟环境：Amazon fulfillment center（障碍密度 15.3%）和 Symbotic warehouse（障碍密度 56.6%），在不同 agent 数量、规划窗口和地图变体上训练与评估。

**📈 对比分析**

与多种基线（RH‑CBS、RH‑PBS、PIBT、WPPL、随机 RH‑PP）进行对比。RL‑RH‑PP 在两张地图上均实现了最高的吞吐量（约 25% 较随机 RH‑PP、30% 较 WPPL），并在不同 agent 密度、规划窗口和未见地图上表现出良好的零样本泛化；求解时间保持在可接受范围内。

**⚠️ 局限性**

局限性：训练成本高（数天 GPU 训练）、仅在相同地图尺寸上实现零样本泛化，需重新训练或调整嵌入以适应全新地图；在极高 agent 密度或不可行性高的场景下仍可能出现较高的不可行率；Top‑K 采样导致推理时间随 K 增大线性增长。

---

## 187. Bridging the Interpretation Gap in Accessibility Testing: Empathetic and Legal-Aware Bug Report Generation via Large Language Models

**arXiv ID:** 2603.23828 | [PDF](https://arxiv.org/pdf/2603.23828v1)

**作者:** Ryoya Koyama `[一作]` (Institute of Science Tokyo), Kenji Tei `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 866 | [OpenAlex ID](https://openalex.org/A5045332896)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HEAR框架，将自动化可访问性测试结果转化为以用户为中心的叙事报告。

**💡 创新点**

创新点在于结合语义重建、动态角色注入和链式思维推理，将技术日志转化为富含同理心与法律风险的报告。

**🔧 技术方法**

使用LLM（GPT‑4o）进行文本生成，配合语义切片、视觉定位和CoT推理。

**📊 数据集**

收集四款主流Android应用（Instagram、Wish、Teams、Booking）中的103条可访问性违规样本。

**📈 对比分析**

与原始JSON日志对比，用户研究显示HEAR在同理心、紧迫感、说服力和法律风险认知上提升了约20‑30%（p<0.05），且无显著认知负荷。

**⚠️ 局限性**

限制包括对检测工具误报的放大、角色简化导致刻板印象、以及在真实工业环境中的可行性未知。

---

## 188. How are AI agents used? Evidence from 177,000 MCP tools

**arXiv ID:** 2603.23802 | [PDF](https://arxiv.org/pdf/2603.23802v1)

**作者:** Merlin Stein `[一作]` (University of Oxford), Merlin Stein `[通讯]` (University of Oxford)

**通讯引用:** 4456 | [OpenAlex ID](https://openalex.org/A5083460661)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了177,436个公开AI代理工具的数据库，并通过下载统计追踪其使用趋势，系统评估工具的直接影响、通用性、任务域、地理分布及AI辅助开发情况。

**💡 创新点**

首次大规模、实时监测代理工具生态，揭示代理行动空间从感知向执行的显著转变，并首次量化AI辅助编写工具的增长。

**🔧 技术方法**

采用Claude Sonnet 4.5 LLM进行工具分类，辅以人工验证；使用GitHub、Smithery API抓取工具；通过NPM/PyPI下载计数及IP地址进行使用量和地域估计；采用加权最小二乘回归与自助法评估趋势。

**📊 数据集**

主要数据集包括GitHub与Smithery上检索到的MCP服务器、NPM/PyPI的下载统计、IP分地域数据、O*NET与SOC职业分类、以及CAISI、UK AI安全研究所等元数据。

**📈 对比分析**

通过与人工标注的一致率（Kappa≈0.7/0.5）验证分类准确；使用加权最小二乘回归与95%置信区间对工具使用比例、通用/窄义工具占比及AI协同开发比例随时间的变化进行量化，结果显示行动工具比例从27%上升至65%，AI协同工具比例从6%升至62%。

**⚠️ 局限性**

仅覆盖公开MCP服务器，下载统计仅为安装近似，未能精准捕捉实际调用；IP分布受PyPI中心化影响；分类仍可能出现误差；对私有内部工具与其他代理协议缺乏覆盖，可能低估代理行动空间。

---

## 189. AgentChemist: A Multi-Agent Experimental Robotic Platform Integrating Chemical Perception and Precise Control

**arXiv ID:** 2603.23886 | [PDF](https://arxiv.org/pdf/2603.23886v1)

**作者:** Xiangyi Wei `[一作]` (East China Normal University), Xiao He `[通讯]` (East China Normal University)

**通讯引用:** 294737 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个多智能体协作的实验机器人平台AgentChemist，支持从自然语言指令到实验执行的闭环自动化，涵盖化学感知、精确控制与报告生成。

**💡 创新点**

创新点在于将大型语言模型与多模态感知（视觉、音频、化学传感器）结合，使用多智能体 FSM 框架实现长尾实验任务的动态分解与自适应调度，并引入统计化学量记录与高置信融合机制。

**🔧 技术方法**

技术包括大语言模型 Qwen3‑VL / Qwen2‑Audio、VLA（Vision‑Language‑Action）多模态机器人控制、基于 FSM 的任务拆解与状态推演、统计映射模型（如抓手位移‑体积映射）、多源融合与置信加权、ROS、移动底盘+双 7‑DOF 机械臂等。

**📊 数据集**

使用实验收集的数据集，包括酸碱滴定（3×2400+点）、络合滴定、固体称重/溶解实验以及不同布局的环境测试，用于验证系统的泛化与鲁棒性。

**📈 对比分析**

通过与传统脚本化平台对比，AgentChemist 在任务完成率、pH 偏差、异常检测准确率等指标上均优于基线；在滴定实验中实现了 0.52% 的 pKa 误差、超过 95% 的成功率，支持 8 小时连续运行。

**⚠️ 局限性**

局限在于透明/无色容器下视觉感知可靠性下降、音频误检、缺乏长期自校准机制、抓手位移‑体积映射受液体黏度影响、固体处理成功率低、对实验桌布局与光照敏感、硬件成本高、LLM 计算成本大以及自然语言理解与主动澄清机制不足。

---

## 190. The Luna Bound Propagator for Formal Analysis of Neural Networks

**arXiv ID:** 2603.23878 | [PDF](https://arxiv.org/pdf/2603.23878v1)

**作者:** Henry LeCates `[一作]` (Amherst College), Haoze Wu `[通讯]` (Amherst College)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5100637734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于 C++ 的α-CROWN界限传播框架 Luna，支持 Interval Bound Propagation、CROWN 和 α-CROWN，并提供 CLI、C++ API 与 Python 绑定。

**💡 创新点**

首次将 α-CROWN 实现为高性能 C++ 库，提供稳定外部接口，显著降低跨语言集成成本和启动开销。

**🔧 技术方法**

利用 Torch 深度学习库并行计算、ONNX 解析、VNN‑LIB 规范、pybind11 Python 绑定以及投影梯度下降优化 α 参数。

**📊 数据集**

在 VNN‑COMP 2025 公开基准（如 acasxu_2023、cervey、cifar100_2024 等）上进行评测。

**📈 对比分析**

与主流 Python 实现（用于 α‑β‑CROWN、NeuralSAT）相比，Luna 在相同硬件上完成更多实例、运行时间更短、平均界宽相当或更窄，并在多项基准上实现了两倍以上速度提升。

**⚠️ 局限性**

目前仅支持主轨道的部分算子和 VNN‑LIB 规范，尚未覆盖更丰富的预条件、离散算子及完整的验证工作流。

---

## 191. High-Density Automated Valet Parking with Relocation-Free Sequential Operations

**arXiv ID:** 2603.23803 | [PDF](https://arxiv.org/pdf/2603.23803v1)

**作者:** Bon Choe `[一作]` (Korea Advanced Institute Of Science And Technology), Heejin Ahn `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 545 | [OpenAlex ID](https://openalex.org/A5054872846)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个框架，生成高密度自动代客泊车布局并列举所有不需要搬移车辆的泊车与出车序列，进一步验证给定操作顺序是否可行。

**💡 创新点**

核心创新在于将搬移自由约束转化为布尔可达性条件，并通过递归搜索与深度优先枚举同时满足该约束和顺序约束的全部序列，实现了高密度布局下完整的可搬移自由序列空间。

**🔧 技术方法**

采用基于拼箱问题的布局优化（Gurobi）、邻接图和混合A*路径规划、递归连通集搜索构造可达性条件，以及DFS枚举有效序列，随后匹配操作顺序。

**📊 数据集**

在仿真中使用三组不同尺寸的停车场实例（如10×10m、12×12m等）和公交车尺寸（2.5×9m）进行实验，数据全部为仿真生成。

**📈 对比分析**

相较于传统停车场与已知高密度布局，所提出框架在保持相同占地面积利用率的前提下，能产生更多搬移自由的泊车/出车序列；在实验中，布局1和2分别可产生56和34条序列，且能满足所有循环操作顺序；而传统方法未给出此类完整序列信息。

**⚠️ 局限性**

限制在于仅考虑单入口、完全顺序的泊车/出车阶段、同形同尺寸车辆，以及仅对小规模停车场实例验证，缺乏对交叉到达/离去、不同车辆尺寸或实时动态环境的支持。

---

## 192. Can VLMs Reason Robustly? A Neuro-Symbolic Investigation

**arXiv ID:** 2603.23867 | [PDF](https://arxiv.org/pdf/2603.23867v1)

**作者:** Weixin Chen `[一作]` (University of Illinois Urbana-Champaign), Han Zhao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 10902 | [OpenAlex ID](https://openalex.org/A5085383685)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型在分布偏移下的推理鲁棒性，提出将概念识别与符号电路相结合的神经符号推理框架。

**💡 创新点**

创新点在于将符号规则编译成布尔电路，显式编码推理函数，实现covariate shift下的稳健推理。

**🔧 技术方法**

使用视觉语言模型（如 Qwen2.5-VL-7B）进行概念识别，结合 SDD 电路进行符号推理，并与 End2end、Prism、ViperGPT 等方法对比。

**📊 数据集**

使用三类视觉演绎推理数据集：MNAdd、MNLogic、KandLogic，分别生成包含 3、5、7 个对象的版本。

**📈 对比分析**

与基线对比，该方法在 3/5/7 对象集上均保持高精度，平均提升约 30%–40%，在 covariate shift 下显著优于 Fine‑tune 与其他神经符号方法。

**⚠️ 局限性**

局限包括需手工提供符号规则、每个推理函数需要单独电路、对概念识别误差敏感，且难以自动化处理自然语言规则。

---

## 193. BRIDG-Q: Barren-Plateau-Resilient Initialisation with Data-Aware LLM-Generated Quantum Circuits

**arXiv ID:** 2603.23979 | [PDF](https://arxiv.org/pdf/2603.23979v1)

**作者:** Ngoc Nhi Nguyen `[一作]` (University of Wollongong), Dinh Thai Hoang `[通讯]` (University of Technology Sydney)

**通讯引用:** 16610 | [OpenAlex ID](https://openalex.org/A5007992576)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 BRIDG‑Q 框架，将大型语言模型生成的量子电路拓扑与基于经验贝叶斯的参数初始化相结合，以缓解变分量子算法中的 barren plateau。

**💡 创新点**

创新点在于将 LLM 生成的离散电路结构与数据驱动的连续参数先验解耦，通过门级分层 Beta 初始化实现神经符号协同，提高优化鲁棒性。

**🔧 技术方法**

使用技术包括 AgentQ LLM 电路生成、OpenQASM 解析、经验贝叶斯 Beta 先验估计、门级分层采样、VQE 训练循环和 Adam 优化器。

**📊 数据集**

数据集为约 580 个基于图的优化实例的 AgentQ benchmark，涵盖多种图结构与 Hamiltonian。

**📈 对比分析**

通过与 AgentQ、随机、均匀等基线在相同电路拓扑下的成对比较，BRIDG‑Q 变体在残差能量和收敛速度上与基线相当或略有提升，oracle 选择的 Beta 方案可实现约 10% 的能量降幅。

**⚠️ 局限性**

局限性在于固定的 Beta 先验对不同实例表现不一，缺乏实例感知的自适应选择，导致单一策略难以持续优于 LLM 基线，且需进一步研究自适应或硬件感知初始化。

---

## 194. The Price Reversal Phenomenon: When Cheaper Reasoning Models End Up Costing More

**arXiv ID:** 2603.23971 | [PDF](https://arxiv.org/pdf/2603.23971v1)

**作者:** Lingjiao Chen `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**通讯引用:** 39499 | [OpenAlex ID](https://openalex.org/A5005779176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估8款主流推理语言模型在9类任务上的实际使用成本，揭示列价与实际成本存在“价格逆转”现象。

**💡 创新点**

首次量化并解释价格逆转的根源——思考（invisible thinking）token 的消费差异，并证明消除思考 token 成本能显著恢复列价与实际成本的一致性；同时指出单个查询成本预测因内部随机性而极其困难。

**🔧 技术方法**

成本审计框架（基于 token 计价公式）、成本分解与消融实验、KNN 预测基线、重复调用实验测量同一查询的方差。

**📊 数据集**

8款 RLM（GPT‑5.2、GPT‑5 Mini、Gemini 3.1 Pro、Gemini 3 Flash、Claude Opus 4.6、Claude Haiku 4.5、Kimi K2.5、MiniMax M2.5）和9个多样化数据集（AIME、ARC‑AGI、GPQA、ArenaHard、HLE、LiveCodeBench、LiveMathBench、MMLUPro、SimpleQA）。

**📈 对比分析**

对列价和实际成本进行排序比较，使用 Kendall's τ 和 pairwise 逆转计数评估；发现21.8% 的模型对出现逆转，逆转幅度最高可达28×；消除思考 token 成本后 τ 由0.563升至0.873，逆转数下降70%。

**⚠️ 局限性**

仅覆盖8款模型，未考虑不同温度或人机交互设置；成本主要按 token 计价，未深入模型内部机制；单量化成本难以对未来模型或不同任务做普适预测。

---

## 195. Policy-Guided Threat Hunting: An LLM enabled Framework with Splunk SOC Triage

**arXiv ID:** 2603.23966 | [PDF](https://arxiv.org/pdf/2603.23966v1)

**作者:** Rishikesh Sahay `[一作]` (University of Illinois), Rod Soto `[通讯]` (Splunk Research Team)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于Agentic AI与Splunk的自动化、动态威胁狩猎框架，覆盖日志摄取、异常检测、深度强化学习优先级划分、LLM多代理上下文分析以及SPL查询生成与验证。

**💡 创新点**

创新点在于：①将Agentic AI与传统SIEM深度集成，形成自适应政策层；②采用两层深度强化学习与自编码器的分层决策，避免特征泄露并实现高效优先级筛选；③利用LLM多代理生成可执行的Splunk查询和MITRE ATT&CK映射，显著降低SOC分析师负担并提升可解释性；④通过奖励塑造实现对不同SOC风险容忍度的快速调整。

**🔧 技术方法**

使用技术包括：自编码器（8-2-8结构）进行无监督异常评分、PPO深度强化学习（两层64神经元）实现容纳/允许决策、LLM（ChatGPT/本地模型）通过CrewAI多代理实现SOC分析与威胁情报推理、Splunk SIEM进行日志索引与验证、数据清洗与聚合、奖励塑形（四种模式）以及决策成本与regret分析。

**📊 数据集**

实验数据集为公开的Boss of the SOC（约1.2万条日志）和自构造的网络扫描/DoS仿真数据（约30万条）。

**📈 对比分析**

通过与公开基准以及传统规则/无强化学习基线对比，展示了四种奖励模式下的精确率、召回率、F1、决策成本、regret以及LLM流量占比。结果表明：奖励模式B在召回率与F1上表现最佳，Mode A在召回率上领先；所有模式均显著降低误报并将需要LLM分析的流量压缩至约35–40%，同时保持高检测效果。

**⚠️ 局限性**

局限性包括：①仅实现二元封堵/允许决策，未覆盖监控、限流等多级响应；②对公开LLM的依赖带来隐私与数据泄露风险，需采用本地模型或严格脱敏；③采用固定时间窗口，可能漏检短时攻击；④尚未实现从检测到自动化补救的闭环；⑤在大规模高吞吐量环境下的可扩展性与实时性评估仍待进一步验证。

---

## 196. Variable-Length Audio Fingerprinting

**arXiv ID:** 2603.23947 | [PDF](https://arxiv.org/pdf/2603.23947v1)

**作者:** Hongjie Chen `[一作]` (Dolby Laboratory), Josh Kimball `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种可变长度音频指纹识别方法 VLAFP，能够在任意长度音频上直接生成指纹并对失真具有鲁棒性。

**💡 创新点**

创新点包括：① 用谱熵自适应分段取代固定长度分段，保留自然边界；② 结合 Transformer 的自注意力与跨注意力实现帧到段的层级编码；③ 采用多正样本对比学习提升指纹可靠性。

**🔧 技术方法**

核心技术包括：Transformer 结构（多头自注意力、跨注意力、归一化、前馈网络），谱熵阈值分段算法，监督式对比学习损失，mel‑spectrogram 时频表示。

**📊 数据集**

实验数据集涵盖音乐、语音与通用音频：Free Music Archive (FMA)、LibriSpeech、AudioSet。

**📈 对比分析**

与 NAFP、AMG、wav2vec2、HuBERT、AST 等基线比较，VLAFP 在商业广播检索（CBR）和假目标检索（DTR）任务中均取得更高的 precision/recall/F1 与 Top‑1 hit 率，尤其在 FMA 与 AudioSet 上领先 5–10%。

**⚠️ 局限性**

主要局限：① 变量长度分段导致推理时加载与批处理复杂，推理速度相对慢；② 需要手动调节谱熵阈值 θ，分段质量对性能影响显著；③ 目前未在极端低信噪比或实时场景下充分验证。

---

## 197. Uncertainty-Aware Vision-based Risk Object Identification via Conformal Risk Tube Prediction

**arXiv ID:** 2603.23919 | [PDF](https://arxiv.org/pdf/2603.23919v1)

**作者:** Kai-Yu Fu `[一作]` (National Yang Ming Chiao Tung University), Yi-Ting Chen `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 10356 | [OpenAlex ID](https://openalex.org/A5100348260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于共形预测的风险管道预测框架，用于视觉风险对象识别，能够给出时空风险置信区间。

**💡 创新点**

创新点在于将共形预测与类别感知校准相结合，结合空间‑时间特征对齐，构造可覆盖真实风险且校准良好的风险管道。

**🔧 技术方法**

主要技术包括I3D时空特征提取、GCN+LSTM关系建模、共形预测、类别感知校准以及空间‑时间特征对齐。

**📊 数据集**

使用了自建的Multiple Coexisting Risks（MCR）数据集，包含多种风险类别并发场景。

**📈 对比分析**

与规则基、HD、BNN、KF、OCP、UE等基线以及现有Vision‑ROI方法比较，实验显示在覆盖率、管道体积、时间一致性、边界对齐和风险IoU指标上均优于基线，同时在下游制动警告上显著降低误报。

**⚠️ 局限性**

局限性包括对遮挡风险的识别仍不够鲁棒，缺乏真实世界验证，且目前仅对每个时间步独立预测，未考虑管道间条件依赖；对框级空间不确定性的建模不足。

---

## 198. Knowledge-Refined Dual Context-Aware Network for Partially Relevant Video Retrieval

**arXiv ID:** 2603.23902 | [PDF](https://arxiv.org/pdf/2603.23902v1)

**作者:** Junkai Yang `[一作]` (Xi'an Jiaotong University), Shanmin Pang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5006557413)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种用于部分相关视频检索的双重上下文感知网络KDC-Net

**💡 创新点**

通过层次语义聚合(HSA)捕获多尺度短语语义，动态时间注意(DTA)聚焦关键事件并抑制噪声，以及知识细化蒸馏(KRD)提升教师信号质量，三项创新共同解决文本多层级、视频时序稀疏与知识蒸馏噪声问题

**🔧 技术方法**

基于Transformer的自注意力、相对位置偏置、动态窗口、Purification Normalization、CLIP预训练模型、KL散度蒸馏、InfoNCE及三元组损失

**📊 数据集**

在TVR和ActivityNet Captions两个公开基准上进行实验

**📈 对比分析**

与现有PRVR方法如DL-DKD、GMMFormer、MS-SL等对比，KDC-Net在TVR上SumR提升至184.9（相对DL-DKD提升5.0点，GMMFormer提升8.3点），R@1最高15.4%；在ActivityNet Captions上亦实现最佳SumR与R@5、R@10提升

**⚠️ 局限性**

对短期/长时序动态平衡、窗口大小与相对位置范围的超参数依赖较高；在极低M/V比例下仍面临检索精度下降的问题，且需要更大规模的教师模型与更长训练时间

---

## 199. Supermassive Blockchain

**arXiv ID:** 2603.23927 | [PDF](https://arxiv.org/pdf/2603.23927v1)

**作者:** Guangda Sun `[一作]` (National University of Singapore), Jialin Li `[通讯]` (National University of Singapore)

**通讯引用:** 2324 | [OpenAlex ID](https://openalex.org/A5108050353)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了State‑Execution Decoupled架构和BFT协议Bullshark+SMS，实现了分布式账本的存储可扩展性与传统BFT安全性兼备。

**💡 创新点**

核心创新在于将状态管理与交易执行逻辑完全分离，并通过结合RS(3f+1,f+1)纠删码与Merkle树验证实现可扩展存储，且不牺牲安全性。

**🔧 技术方法**

采用Reed‑Solomon纠删码、Merkle树证明、随机复制策略、分层存储（更新表+检查点）以及批量获取/推送机制；BFT层使用Bullshark原子广播；实验使用YCSB键值存储与UTXO链模拟。

**📊 数据集**

实验数据集包括100M键值对（16B键+1000B值）以及100M UTXO（32B ID+4B索引），模拟Ethereum/Bitcoin状态规模；网络延迟通过AWS与全球50城市数据引入。

**📈 对比分析**

与完全复制的BFT节点以及传统分片BFT节点对比，Bullshark+SMS在100节点时每节点存储仅9.8GB，吞吐量比完全复制提升1.5-2.6倍，网络流量增幅仅2.3%-30.2%，在LAN/WAN环境下延迟保持与传统BFT相当。

**⚠️ 局限性**

局限性包括：远程状态检索仍是关键路径，导致对低Skew或高写入负载的性能受限；批量获取需要预知键集，限制了对动态工作负载的适应；恢复跨节点故障时需要重建检查点，重建成本仍较高。

---

## 200. MonoSIM: An open source SIL framework for Ackermann Vehicular Systems with Monocular Vision

**arXiv ID:** 2603.23965 | [PDF](https://arxiv.org/pdf/2603.23965v1)

**作者:** Shantanu Rahman `[一作]` (Islamic University of Technology), Golam Sarowar `[通讯]` (Islamic University of Technology)

**通讯引用:** 236 | [OpenAlex ID](https://openalex.org/A5089551291)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个低成本、开源的Software-in-the-Loop（SIL）仿真平台MonoSIM，用于Ackermann转向的自主车辆研究与教育，平台集成了单目摄像头、滑动窗口车道检测、ROS2通信以及MVSIM物理仿真；

**💡 创新点**

创新点在于：1）将单目视觉与低算力的滑动窗口车道检测结合，保持实时性；2）在同一统一环境中支持PID与MPC等多种控制算法的快速验证与对比；3）提供完整、可扩展的ROS2节点架构，易于接入多智能体与学习任务；

**🔧 技术方法**

使用的技术包括：OpenCV（相机标定、透视变换、车道检测）、ROS2（节点与消息通信）、MVSIM+Box2D（2.5D物理仿真）、PILOT/Python实现的PID与MPC控制器、滑动窗口+二次多项式拟合车道提取；

**📊 数据集**

数据集：在仿真环境中随机生成类似Sin曲线的车道轨迹（纯黑白色车道线），并未使用公开真实道路图像或大型标注数据集；

**📈 对比分析**

比较方法：在多次随机轨迹上分别运行PID与MPC，记录车辆位置、速度、偏航角等；结果显示：PID轨迹偏差均方根较小（横向0.0136 m²，角度0.000548 rad²），但转向角度波动较大；MPC更平滑、约束满足，但横向偏差和角度均方根稍高（0.0390 m²，0.001014 rad²）；

**⚠️ 局限性**

局限性：仅在仿真环境下验证，未在真实车辆或复杂光照/天气条件下测试；单目摄像头分辨率低，车道检测对光照敏感；动力学模型简化（无悬挂、风阻、横向滑移）；缺乏在线学习或多传感器融合功能。

---

## 201. Kirchhoff-Inspired Neural Networks for Evolving High-Order Perception

**arXiv ID:** 2603.23977 | [PDF](https://arxiv.org/pdf/2603.23977v1)

**作者:** Tongfei Chen `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**通讯引用:** 13750 | [OpenAlex ID](https://openalex.org/A5015525872)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种基于基尔霍夫电路法则的神经网络结构 KINN，将信息的强度、耦合与时间演化统一编码为内部状态，并通过 RC 节点实现稳定的离散更新。

**💡 创新点**

将时间演化视为内部可学习的状态变量，用连续-离散 RC 动力学生成闭式递归更新；通过级联多个 RC 单元实现高阶演化；结构可解释、可端到端训练。

**🔧 技术方法**

基尔霍夫电路动力学、零阶保持离散化、RC 神经单元 (KNC)、级联 Kirchhoff 块 (CKB)、融合至 Fourier Neural Operator 与 U‑Net、残差门控与深度学习框架。

**📊 数据集**

PDE 求解数据集（Darcy Flow、Shallow Water、Navier–Stokes）以及 ImageNet‑1K（Tiny、Small）用于视觉任务。

**📈 对比分析**

与现有神经算子、序列模型和视觉模型进行对比。在 PDE 任务上误差分别为 1.775×10⁻²、2.587×10⁻³、9.875×10⁻³；在 ImageNet 上 Top‑1 分别为 83.3%（Tiny）与 83.9%（Small），均优于主流基线。

**⚠️ 局限性**

只验证了有限的级联深度与任务，未覆盖更异构模态、异常动力学或超大规模模型；模型非完全生物学解释，缺乏自适应阶数、丰富电路拓扑和更紧密的连续时间分析。

---

## 202. SLAT-Phys: Fast Material Property Field Prediction from Structured 3D Latents

**arXiv ID:** 2603.23973 | [PDF](https://arxiv.org/pdf/2603.23973v1)

**作者:** Rocktim Jyoti Das `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 39576 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种从单张RGB图像直接预测3D资产空间变异材料属性场（Young模量、密度、泊松比）的方法；

**💡 创新点**

创新点在于利用预训练的3D资产生成模型（Trellis）产生的空间组织latent（SLAT）特征，完全不需要显式3D重建或多视角聚合即可完成物理属性回归；

**🔧 技术方法**

主要技术包括：1）Trellis图像到3D生成器提取SLAT特征；2）稀疏Swin‑style Transformer解码器回归连续物理参数并进行材料分类；3）将预测的属性映射到Gaussian Splatting模型，进而在MPM仿真中验证物理合理性；

**📊 数据集**

使用的主要数据集为PixieVerse，包含1624个单对象、10个语义类别，提供 voxel 级别的E、ρ、ν及材料标签；

**📈 对比分析**

与NeRF2Physics和Pixie两个基线进行对比：速度提升约120×；连续属性MSE分别为0.036–0.054，材料分类准确率约0.95，整体与Pixie相当或略优；

**⚠️ 局限性**

局限性包括：1）训练标注来自VLM生成，可能存在偏差；2）仅利用视觉信息，缺乏触觉或音频等多模态输入；3）对SLAT与物理注解的对齐依赖ICP，可能影响鲁棒性；4）对极其复杂或材质极端差异的对象鲁棒性尚未验证。

---

## 203. GRMLR: Knowledge-Enhanced Small-Data Learning for Deep-Sea Cold Seep Stage Inference

**arXiv ID:** 2603.23961 | [PDF](https://arxiv.org/pdf/2603.23961v1)

**作者:** Chenxu Zhou `[一作]` (Shanghai Jiao Tong University), Xiaofeng Gao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6334 | [OpenAlex ID](https://openalex.org/A5019439900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过构建生态知识图并加入图正则化的多项式逻辑回归，对深海冷渗点的发育阶段进行微生物丰度预测。

**💡 创新点**

将宏观宏观生物计数作为训练时的结构先验，利用宏-微生物关联与微生物共现构建双源知识图，且在推理阶段无需宏观数据。

**🔧 技术方法**

采用中心对数比（CLR）处理组成数据，构建知识图 Laplacian 正则化的多项式逻辑回归（GRMLR），并使用 DUSt3R 进行海底地图构建、YOLOv11 检测。

**📊 数据集**

13个南海冷渗点样本，包含26类微生物相对丰度、4类宏观生物计数及三阶段标签。

**📈 对比分析**

与LR、SVM、RF、KNN、原始LR及 Gemini 3 Flash 等基线进行留一交叉验证，GRMLR 获得 84.62% 准确率、0.825 Macro‑F1，显著优于基线。

**⚠️ 局限性**

样本量极小，依赖专家构建的知识图，模型在不同地区或更大样本规模下的泛化尚未验证。

---

## 204. SynMVCrowd: A Large Synthetic Benchmark for Multi-view Crowd Counting and Localization

**arXiv ID:** 2603.23956 | [PDF](https://arxiv.org/pdf/2603.23956v1)

**作者:** Qi Zhang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21487 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了SynMVCrowd合成多视角人群计数与定位基准，并在该基准上评估多种SOTA方法，提出了空间特征选择与Optimal Transport损失的强大基线模型。

**💡 创新点**

①创建了迄今规模最大的多视角合成数据集（50场景、50视角、200帧、最多1000人）；②提出结合投影、最大池化融合与OT损失的端到端定位/计数框架；③通过该基准验证跨场景迁移与单视图任务的通用性。

**🔧 技术方法**

使用GTA‑V合成技术构建场景，利用ResNet/Transformer等骨干提取单视图特征，空间注意力选择特征后投影到地面平面，采用视角最大池化融合，最终通过OT或MSE损失进行训练，并在迁移实验中加入域适应模块。

**📊 数据集**

主要使用SynMVCrowd（5.3 M图像、50场景、50视角、200帧）进行实验，并与Wildtrack、MultiviewX、CityStreet、PETS2009、CVCS、GCC、UCF‑CC‑50、ShanghaiTech A/B、Mall等公开数据集进行对比与迁移评估。

**📈 对比分析**

在多视角定位任务中，Baseline(OT)在MODA、MODP、Precision、Recall、F1等指标上均优于所有SOTA方法；在计数任务中，Baseline(OT)在MAE/NAE/MSE上达到最优；单视图计数实验中DM‑Count、GLoss表现最佳，定位方面STEERER最优。跨场景迁移时，使用少量标注或域适应可将性能提升至与SOTA相当甚至更好。

**⚠️ 局限性**

受域间差距限制，合成数据在真实场景中仍需少量标注或域适应才能达到最佳；基线对视角极端变化的鲁棒性有限；基准主要关注计数与定位，其他任务如姿态估计、分割等仍需进一步探索。

---

## 205. The Missing Adapter Layer for Research Computing

**arXiv ID:** 2603.23942 | [PDF](https://arxiv.org/pdf/2603.23942v1)

**作者:** Bowen Li `[一作]` (RMIT University), Fengling Han `[通讯]` (RMIT University)

**通讯引用:** 4770 | [OpenAlex ID](https://openalex.org/A5021052552)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

提出并实现了一个轻量级的适配层，解决HDR候选人从云/本地GPU资源到可复现、GPU‑ready的交互式研究环境的中间环节；通过k3s+Coder+版本化容器实现快速（≤5 min）部署并提供自服务工作空间；同时提出四维度指标框架评估此层性能。

**💡 创新点**

①以适配层框架定位问题，拆解为环境可复现、上手痛点、资源闲置、供应商耦合四大缺口；②设计基于k3s的轻量调度、Coder自托管IDE、版本化容器镜像的组合，避免重构传统批量调度或云托管平台；③提供可量化的基线与目标指标，为跨机构评测提供共通语言。

**🔧 技术方法**

k3s（轻量化Kubernetes）、Coder（自托管VS Code工作区）、Docker容器与私有镜像仓库、GitHub Actions CI/CD、Helm、NVIDIA驱动/CUDA、NVML+Prometheus/Grafana监控。

**📊 数据集**

未使用公开科研数据集；评测采用三种真实项目（CPU Web、GPU Web、Kubernetes Operator）作为部署案例来验证性能与稳定性。

**📈 对比分析**

通过对比传统VM手动配置（10–20 min启动+30–90 min环境搭建）与本适配层（冷启动≈5 min、热启动≈20 s、CI/CD ≤5 min）验证部署延迟；通过自动健康检查测得环境可复现率≥99%；上手时间从1–3工作日降至≤5 min；GPU利用率从<30%（未调度）提升至可见的共享利用，均符合设定基线与目标。

**⚠️ 局限性**

需要至少一名技术熟练成员完成初始集群与Coder部署；集成SSO需与机构IT协同；当前缺乏细粒度配额/限额与工作负载抢占机制；未实现自动空闲工作区回收；对科研产出影响尚未量化，需要进一步纵向评估。

---

## 206. OmniACBench: A Benchmark for Evaluating Context-Grounded Acoustic Control in Omni-Modal Models

**arXiv ID:** 2603.23938 | [PDF](https://arxiv.org/pdf/2603.23938v1)

**作者:** Seunghee Kim `[一作]` (Hanyang University), Hwiyeol Jo `[通讯]` (NAVER Cloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为“CAG”（Context‑grounded Acoustic Grounding）基准，用来评估多模态模型在给定语音指令、文本脚本和图像时生成符合语境的语音的能力。

**💡 创新点**

创新点在于：①提出评估多模态模型在语音输出时的音频控制能力；②设计六种可测量和抽象的声学特征；③通过三阶段生成+质控构建大规模高质量数据；④系统分析并归纳三种失败模式。

**🔧 技术方法**

采用大语言模型生成文本脚本、指令；使用文本到语音（TTS）合成声学控制信号；使用图像生成模型（如Stable Diffusion）合成图像；利用 CLIP、LPIPS 等度量多样性；使用 Whisper 评估语义准确度；使用 WavLM‑Large 训练声学属性分类器。

**📊 数据集**

构建数据集共3,559条实例，覆盖六种声学特征，每种约600条。数据通过 LLM 生成、图像生成、TTS 合成，并经过双重过滤与量化验证。

**📈 对比分析**

对八款现有全模态模型（MiniCPM‑o 4.5、InteractiveOmni 8B/4B、Qwen3‑Omni 30B、Qwen2.5‑Omni 7B/3B、Uni‑MoE‑2.0‑Omni、MGM‑Omni 7B）进行评测。所有模型在语义保真（WER）和声学控制（ΔWPM、PER、VFR@0.3、Emo‑Acc、GA‑Acc、Tim‑Acc）上均显著低于人工参考，表明当前模型在多模态语音生成上的能力有限。

**⚠️ 局限性**

局限性包括：①仅评估单一声学特征，未考察多特征联合控制；②仅使用语音指令作为音频输入，未涵盖环境噪声等；③生成任务为合成数据，缺乏真实语音场景的挑战。

---

## 207. Revealing Multi-View Hallucination in Large Vision-Language Models

**arXiv ID:** 2603.23934 | [PDF](https://arxiv.org/pdf/2603.23934v1)

**作者:** Wooje Park `[一作]` (Seoul National University), Byonghyo Shim `[通讯]` (Seoul National University)

**通讯引用:** 7590 | [OpenAlex ID](https://openalex.org/A5076075267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了多视角幻觉（multi‑view hallucination）概念，并构建了专门评估该问题的 MVH‑Bench 基准。

**💡 创新点**

创新点在于：①将幻觉拆分为跨实例与跨视角两类，并为每类设计了二值与多选题；②设计了专门的 p‑Acc、q‑Acc 等评估指标；③提出了无需训练的 Reference Shift Contrastive Decoding (RSCD)，通过对中间层文本‑文本注意力进行遮挡生成负 logits，抑制非目标视觉干扰。

**🔧 技术方法**

采用的技术包括：对 Transformer 解码器中间层文本‑文本注意力的遮挡，基于注意力分布的负 logits 生成与对比，Adaptive Plausibility Constraints 约束高概率词；使用 GPT‑4o 自动抽取实例‑描述对，生成 QA 对。

**📊 数据集**

使用 Ego‑Exo4D 与 LEMMA 两大多视角数据集提取图像对，随后生成 4.8k Q‑A 对（3.2k 二值，1.6k 多选），最终形成 MVH‑Bench。

**📈 对比分析**

在 Qwen2.5‑VL 与 LLaVA‑OneVision 上与 VCD、ICD、AvisC 等方法对比，RSCD 在整体 MVH‑Score 上分别提升 21.1 与 34.6 点，且在各类指标和子任务中表现最优。

**⚠️ 局限性**

局限性包括：仅覆盖两视角对，未考虑更大视角集合；RSCD 设计针对视觉与文本分别处理、在解码时融合的 LVLM 架构，可能不适用于其他架构。

---

## 208. Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding

**arXiv ID:** 2603.23914 | [PDF](https://arxiv.org/pdf/2603.23914v1)

**作者:** Fatih Ilhan `[一作]` (Georgia Institute of Technology), Ling Liu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 21459 | [OpenAlex ID](https://openalex.org/A5100343991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出AttentionPack框架，针对大型视觉语言模型在解码过程中的内存瓶颈，通过对KV缓存进行多头低秩压缩与注意力感知的分层解压缩，提高解码效率。

**💡 创新点**

创新点在于：①利用多头合并后的SVD压缩KV矩阵实现高达8倍内存压缩；②基于累积注意力得分的注意力感知分层解压，显著减少解压计算。

**🔧 技术方法**

使用SVD低秩分解、注意力得分加权分层解压、量化/驱逐、分组查询注意力、FlashAttention融合核等技术。

**📊 数据集**

在A-OKVQA、OCR-VQA、MMMU、MSVD-QA、MSRVTT-QA等图像与视频问答数据集上进行实验。

**📈 对比分析**

与H2O、ScissorHands、FastV、Minicache等现有驱逐/量化方法以及GQA、FlashAttention等低层优化进行对比，AttentionPack在保持或略优的性能下将KV缓存压缩至5–8倍，批量推理吞吐量提升约60%~74%。

**⚠️ 局限性**

局限性包括：①压缩率与模型性能敏感，低秩设定不当会导致准确率下降；②注意力感知解压仍需额外计算，单实例推理延迟略增；③在极高分辨率或长视频场景下，低秩假设可能失效。

---

## 209. AnalogAgent: Self-Improving Analog Circuit Design Automation with LLM Agents

**arXiv ID:** 2603.23910 | [PDF](https://arxiv.org/pdf/2603.23910v1)

**作者:** Zhixuan Bao `[一作]` (Nanyang Technological University), Xun Xu `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 12123 | [OpenAlex ID](https://openalex.org/A5100451924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个训练-free的代理框架AnalogAgent，用多智能体（代码生成、设计优化、知识策划）和自演化记忆机制实现模拟电路的自动化设计。

**💡 创新点**

创新点在于：① 将LLM拆分为专门化的多智能体，避免单一模型的循环瓶颈；② 通过自演化记忆（Adaptive Design Playbook）持续提炼、去重、检索设计经验，缓解上下文消失；③ 结合执行反馈实现跨任务的自我改进，提升小模型的性能。

**🔧 技术方法**

采用大型语言模型（Gemini-2.5-Flash、GPT-5 等）作为底层LLM，构建MAS并实现SEM；利用模拟器进行 SPICE/Python 代码执行与诊断；使用 Bayesian 优化（TPE/Optuna）对可调参数进行微调。

**📊 数据集**

使用公开的 30 任务模拟电路基准（分为 Easy、Medium、Hard），涵盖从简单放大器到复杂比较器等多种拓扑；实验采用多轮生成、评估和知识更新的统一流程。

**📈 对比分析**

与 SPICEPilot、AnalogCoder、AnalogCoder-Pro 等基线在 Pass@k（k=1,5）指标进行比较。AnalogAgent 在 GPT-5 上实现 97.4% Pass@1、100% Pass@5，Gemini 上 92.0% Pass@1、99.9% Pass@5；在小模型（Qwen-8B 等）上也显著提升，Pass@1 由 23.3% 提升至 72.1%。

**⚠️ 局限性**

局限性包括：① 仍依赖模拟器的执行反馈，计算成本较高；② 需要一定量的迭代才能累积足够知识，对极其复杂或新颖的拓扑仍可能收敛慢；③ 记忆检索策略目前基于子串匹配，可能对跨域泛化产生限制。

---

## 210. FilterGS: Traversal-Free Parallel Filtering and Adaptive Shrinking for Large-Scale LoD 3D Gaussian Splatting

**arXiv ID:** 2603.23891 | [PDF](https://arxiv.org/pdf/2603.23891v1)

**作者:** Yixian Wang `[一作]` (Beijing Institute of Technology), Yi Yang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 81588 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

研发并实现了 FilterGS，利用并行过滤和 Gaussian 缩小策略显著加速大规模 3D Gaussian Splatting 的渲染。

**💡 创新点**

创新点包括：1) 并行过滤机制（R&L 与 Ancestor 两个互补过滤器）实现无层级遍历；2) 基于 GTC 指标的场景自适应 Gaussian 缩小策略，动态阈值 λ_G 自动剔除冗余 Gaussian‑Tile 对。

**🔧 技术方法**

技术手段：3D Gaussian Splatting、层次 Gaussian 树、CUDA 并行实现、R&L 与 Ancestor 过滤器、KPC/GTC 评估指标、动态阈值 λ_G、预处理、排序、α 叠加等。

**📊 数据集**

实验数据集：MatrixCity（Block Small）、UrbanScene（Residence、Sci‑art）、GauUScene（College、Residence、Modern‑Building）等大规模场景。

**📈 对比分析**

与 vanilla 3DGS、H3DGS、FLoD、LoG、OctreeGS 等基线对比；FilterGS 在所有数据集上 FPS 提升至 200–300+ FPS，过滤时间下降 90%+，渲染质量（PSNR/SSIM）保持竞争水平；整体渲染性能显著优于现有方法。

**⚠️ 局限性**

局限性：额外约 20% 的内存开销；对 λ_G 参数敏感，过度缩小可能产生边界痕迹；适用于层次 Gaussian 树，非树结构可能不适用；需要额外预处理计算 GTC，增加初始准备时间。

---

## 211. Robust Distributed Cooperative Path-Following and Local Replanning for Multi-UAVs Under Differentiated Low-Altitude Paths

**arXiv ID:** 2603.23968 | [PDF](https://arxiv.org/pdf/2603.23968v1)

**作者:** Zimao Sheng `[一作]` (Northwestern Polytechnical University), Hong'an Yang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5052278105)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

设计并验证了一种鲁棒分布式协同路径跟随及局部重规划框架，适用于复杂DEM低空环境下的多UAV系统，并实现了有限时稳定跟随与在线障碍规避。

**💡 创新点**

提出基于时间指数的分布式协同控制协议实现时域一致性、利用纵横向预览角调整的鲁棒指导法实现有限时误差消除，并设计了最优时效的局部路径重规划算法，三者结合显著提升了协同跟随鲁棒性和实时性。

**🔧 技术方法**

采用分布式一致性控制、鲁棒有限时稳定理论、纵横预览角调整的跟随指导、基于随机采样的局部重规划、离散化通讯拓扑与多目标路径规划、以及EKF状态估计与Wind模型。

**📊 数据集**

在仿真中使用真实数字高程模型（DEM）数据进行低空路径跟随测试，并采用Aerosonde小型固定翼UAV参数；风场扰动采用von Kármán谱的Dryden模型；实验初始位置随机选在目标圆上。

**📈 对比分析**

与传统集中式速度分配及单路径跟随算法对比，评估指标包括平均误差AE̅、RMSE、最大时间偏差MD和重规划时长RT；结果显示AE̅≤10 m，MD≤20 s，RT<3 s，系统在4–13只UAV下均保持高协同与鲁棒性。

**⚠️ 局限性**

仍依赖可靠邻居通信拓扑，存在通信范围与节点数限制；算法对极端高风速或大规模障碍场的鲁棒性未彻底验证；仿真未涵盖真实硬件动态与感知误差。

---

## 212. Wireless communication empowers online scheduling of partially-observable transportation multi-robot systems in a smart factory

**arXiv ID:** 2603.23967 | [PDF](https://arxiv.org/pdf/2603.23967v1)

**作者:** Yaxin Liao `[一作]` (Beijing University of Posts and Telecommunications), Ping Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 115369 | [OpenAlex ID](https://openalex.org/A5100405781)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种结合无线 M2M 通信的在线调度框架，帮助 AGV 在部分可观测的智能工厂环境下实现冲突与拥堵可避免的路径规划。

**💡 创新点**

创新点在于：①将 AGV 的意图信息视为关键 M2M 流量，采用无重传多链路传输满足实时需求；②在 MRTA 采用模拟退火+大邻域搜索，路由则使用拥堵感知 A*；③系统化地把通信网络与调度目标耦合，形成闭环信息共享。

**🔧 技术方法**

核心技术包括：多链路 OFDMA 无重传上行、模拟退火（SA）与大邻域搜索（LNS）的 MRTA、基于拥堵映射的扩展 A* 路径搜索、以及基于优先级的冲突解决规则。

**📊 数据集**

使用基于 10×10 网格的仿真环境，包含 60 条生产线、每条 4 个搬运任务，AGV 数量从 2 到 100，任务需求与加工时长均按 U(5,10) 随机生成。

**📈 对比分析**

与无通信、仅本地感知、以及带错误的通信方案对比，结果显示在 AGV 数量超过 30 时，本方案使完成时间（makespan）比本地感知低 54% 以上；在有限频道（C=1）下，通过调节通信间隔 D 还能获得 17% 的调度效率提升。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未考虑多跳网络、动态信道分配、实际硬件延迟及误码；模型假设 AGV 运动受限于 2D 网格，未覆盖更复杂的工厂拓扑；且算法仍依赖中心服务器，缺乏完全分布式鲁棒性。

---

## 213. Leave No Stone Unturned: Uncovering Holistic Audio-Visual Intrinsic Coherence for Deepfake Detection

**arXiv ID:** 2603.23960 | [PDF](https://arxiv.org/pdf/2603.23960v1)

**作者:** Jielun Peng `[一作]` (Harbin Institute of Technology), Xiaopeng Hong `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8345 | [OpenAlex ID](https://openalex.org/A5026880795)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 HAVIC 的全新多模态深度伪造检测框架，通过学习真实视频中的内在音频-视频一致性先验并在此基础上进行自适应特征聚合，实现对深度伪造内容的高效识别。

**💡 创新点**

创新点包括：① 统一的三层一致性建模（模态内结构一致性、跨模态微观同步一致性、跨模态宏观语义一致性）并通过自监督预训练学习；② 细粒度音视频对比学习与软负样本机制，提升跨模态时序对齐；③ 层次化自适应聚合与辅助分类器，动态权衡多层特征和模态差异，增强泛化能力；④ 新建 HiFi-AVDF 数据集，涵盖文本‑视频、图像‑视频生成的高保真伪造，覆盖多款最新生成模型。

**🔧 技术方法**

采用 Transformer 编码器、Masked Autoencoder（MAE）解码器、跨模态注意力交互、层次化重构损失、细粒度对比损失和语义重构损失；在预训练后使用自适应特征聚合（Attention + MLP）和多头分类器进行检测。

**📊 数据集**

主要使用 LRS2 进行自监督预训练；FakeAVCeleb 进行微调；在交叉评测中使用 KoDF、HiFi-AVDF（自建）以及公开的 FaceForensics++、WildDeepfake 等数据集进行评估。

**📈 对比分析**

与多种基线（单模态、跨模态不对齐、现有自监督预训练、已公开的多模态检测方法）进行对比，HAVIC 在 FakeAVCeleb 上达到 99.8% ACC、99.9% AUC；在跨数据集（KoDF、HiFi-AVDF）上分别提升 9.39% AP、9.37% AUC，远超同类方法，显示出显著的鲁棒性与泛化能力。

**⚠️ 局限性**

受限于仅针对人脸视频的设计，模型在非人类物体、场景或缺失音频场景下的表现尚未验证；高保真伪造样本仍能对模型构成挑战；同时预训练与微调阶段的参数量与算力需求相对较高。

---

## 214. VOLMO: Versatile and Open Large Models for Ophthalmology

**arXiv ID:** 2603.23953 | [PDF](https://arxiv.org/pdf/2603.23953v1)

**作者:** Zhenyue Qin `[一作]` (Yale University), Qingyu Chen `[通讯]` (Yale University)

**通讯引用:** 6040 | [OpenAlex ID](https://openalex.org/A5042874172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 VOLMO 框架，基于公开数据开发可跨任务的多模态大模型，支持眼科图像描述、疾病筛查、分级与临床评估与方案生成。

**💡 创新点**

创新点在于三阶段自适应训练（知识预训练 → 任务微调 → 诊疗推理）、完全开放式数据与模型、支持单模型完成多任务且具生成与推理能力。

**🔧 技术方法**

采用 InternVL 视觉-语言架构，结合 LLM 生成、指令式微调、动态分辨率编码与多轮对话训练技术。

**📊 数据集**

使用公开来源的图像–文本对（26,569 篇眼科论文、>110k multimodal 实例）、12 眼科疾病标注数据（26,929 例）以及 913 条完整病例报告，并在 UK Biobank、Sydney Innovation、SUSTech 三个独立队列做外部验证。

**📈 对比分析**

与 InternVL‑2B、LLaVA‑Med、MedGemma‑4B/27B 及 RETFound 进行基准对比；在图像描述、疾病筛查、分级与评估生成等任务中，VOLMO‑2B 的 F1、准确率等指标均显著优于基线，提升幅度通常超过 15%–30%。

**⚠️ 局限性**

局限性包括：模型仍受公开数据多样性限制，准确性尚不理想；缺乏真实临床多样化样本与持续的临床验证；仅在公开数据上训练，需进一步在本地化数据上微调以提升鲁棒性。

---

## 215. From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents

**arXiv ID:** 2603.23951 | [PDF](https://arxiv.org/pdf/2603.23951v1)

**作者:** Sirui Xia `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**通讯引用:** 4107 | [OpenAlex ID](https://openalex.org/A5090455375)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个闭环的自动化框架，利用LLM驱动的进化搜索在大型语言模型的策略优化算法空间中自动发现改进机制。

**💡 创新点**

创新点在于引入反思增强的进化搜索与结构化家谱档案，系统化地将实验证据转化为可重用知识，实现对算法机制的自动化改进，并从中提炼可解释的设计原则。

**🔧 技术方法**

使用了LLM生成提案、实现与验证、统一评估协议、反思分析，以及贝叶斯GP评估的进化搜索技术；实验以GRPO基线和Qwen2.5-Math-1.5B模型为核心。

**📊 数据集**

采用数学推理数据集：MATH、AIME24/25、AMC、Minerva、OlympiadBench；训练集为5000例MATH Levels 3–5子集。

**📈 对比分析**

在标准化评估套件上对64个算法进行比较，最佳变体VM-AV-GRPO将weighted Overall从47.8提升到52.5（+4.6），AIME25准确率从26.7%升至43.3%。

**⚠️ 局限性**

限制包括：搜索过程计算成本高、结果主要在数学推理任务验证，泛化性未知、机制解释仍为证据假设，缺乏因果验证。

---

## 216. Dialogue to Question Generation for Evidence-based Medical Guideline Agent Development

**arXiv ID:** 2603.23937 | [PDF](https://arxiv.org/pdf/2603.23937v1)

**作者:** Zongliang Ji `[一作]` (Google Research), Fan Zhang `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究开发了一种基于大语言模型的环境助手，能够在主治医生与患者的对话中实时生成与循证医学（EBM）相关的、可指导查阅指南的临床问题；

**💡 创新点**

创新点在于：①提出了多阶段推理框架（摘要、问题生成、问题评估），实现对对话信息的结构化抽取与针对性问题生成；②将问题生成任务拆分为生成10个候选问题并自动评估，最终挑选3个最优问题，减少人力评估成本；

**🔧 技术方法**

主要技术：使用Google Gemini 2.5（Flash版用于摘要，Pro版用于生成与评估），配合零射（zero-shot）与多阶段推理两种提示策略，并通过链式思考（CoT）评估候选问题；

**📊 数据集**

数据集：从已去标识化的2000条美国临床访谈转录中筛选出899例主诊科病例，最终抽取80例多样化的对话作为评估集；

**📈 对比分析**

比较方法：让6名经验丰富的临床医生在同一80例对话（分别截取30%、70%、100%上下文）上对两种模型（零射和多阶段）分别打分，使用5项指标（相关性、指南导航、思维对齐、非冗余、实用性）进行Likert量表评估。结果显示，多阶段方法在整体得分上略优于零射（平均5.63对5.54），在指南导航和非冗余等指标上提升约5-7%；自动评估（LLM-as-judge）与人工评估方向一致，但在绝对分数上偏高并存在安全性漏报；

**⚠️ 局限性**

局限性：①专家评估成本高（90小时约10k美元）；②多阶段推理导致较高的API调用量和约60秒的延迟，影响实时部署；③仅在80例主诊科病例上评估，缺乏对其他专科和更大规模数据的验证；④LLM评估对错误的识别不充分，需人工确认；

---

## 217. DP^2-VL: Private Photo Dataset Protection by Data Poisoning for Vision-Language Models

**arXiv ID:** 2603.23925 | [PDF](https://arxiv.org/pdf/2603.23925v1)

**作者:** Hongyi Miao `[一作]` (Shandong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21808 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6215c339-3735-4be3-8a07-5bbb7004712d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对视觉‑语言模型（VLM）在对私有照片数据集进行微调时的身份‑关系泄露威胁，并提出了DP2‑VL数据集保护框架；

**💡 创新点**

创新点在于构建身份‑关系泄露威胁模型和对应基准数据集，以及通过全局特征分布偏移（GFDS）实现数据集级别的无学习攻击；

**🔧 技术方法**

使用了对抗扰动（PGD）与随机化变换（EOT）在视觉‑语言对齐空间中优化特征漂移，结合LoRA微调；

**📊 数据集**

采用了自建的七类身份‑关系场景数据集（共约2100张照片），包括身份识别、拥有、同学、同事、情侣、家庭、朋友等；

**📈 对比分析**

与主流VLM（LLaVA、Qwen‑VL、MiniGPT‑v2）在不同保护比例、提示风格、后处理与净化防御下对比，DP2‑VL在身份/关系泄露成功率（ASR）上大幅下降，尤其在高保护比例下效果显著；

**⚠️ 局限性**

局限性包括对低保护比例敏感以及在强净化（如DiffPure）下保护效果显著降低，且在部分场景下仍可被轻微恢复。

---

## 218. SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating

**arXiv ID:** 2603.23983 | [PDF](https://arxiv.org/pdf/2603.23983v1)

**作者:** Hanbyel Cho `[一作]` (Samsung Electronics), Donghan Koo `[通讯]` (Samsung Electronics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种实时文本驱动的全身控制框架 SafeFlow，结合物理引导的运动生成和三阶段安全门，确保机器人在开放式输入下能够生成可执行、可追踪且安全的运动。

**💡 创新点**

创新点包括：① 在 VAE 隐空间使用 Rectified Flow Matching 并加入物理引导梯度，② 利用 Reflow 蒸馏将物理约束嵌入网络以实现单步推断，③ 设计无训练的三阶段安全门（语义 OOD 检测、流场稳定性评分、硬件约束屏蔽），三者层层过滤危险行为。

**🔧 技术方法**

技术：VAE、CLIP 文本编码器、Rectified Flow Matching、物理引导梯度（自关节限、碰撞、平衡、平滑），Reflow 蒸馏、Mahalanobis 语义距离、Jacobian 方向敏感度、PPO 训练的低层运动跟踪器。

**📊 数据集**

数据集：BABEL 语料库（重映射到 Unitree G1），使用 2,362 条验证 prompt 以及 100 条 OOD prompt（未见动词和极端动力学）。

**📈 对比分析**

与 TextOp（自回归扩散模型）对比：在生成阶段联合物理约束后，关节限违规率从 43% 降至 3%，自碰撞率从 11% 降至 1%；系统级成功率提升至 98.5%（TextOp 80.6%）；在实时推断上，SafeFlow 通过 Reflow 与安全门实现平均 14.78 ms 生成延迟（≈67 Hz），明显快于 TextOp 的 23.59 ms。

**⚠️ 局限性**

局限：fallback 机制目前仅退回到静止站立姿态，对高动态任务恢复不够平滑；物理引导参数需手工调节，可能对不同机器人模型不易迁移；安全门阈值设定基于经验，可能对极端 OOD 情况不够鲁棒。

---

## 219. SilLang: Improving Gait Recognition with Silhouette Language Encoding

**arXiv ID:** 2603.23976 | [PDF](https://arxiv.org/pdf/2603.23976v1)

**作者:** Ruiyi Zhan `[一作]` (Beihang University), Annan Li `[通讯]` (Beihang University)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5012121355)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过Contour‑Velocity Tokenizer将二进制步态轮廓映射到离散文本词汇空间，并用大型语言模型（LLM）生成离散嵌入，最终与视觉特征融合，构建Silhouette Language Model（SilLang）提升步态识别；

**💡 创新点**

①将步态二进制轮廓与自然语言置于同一离散编码空间；②提出Contour‑Velocity Tokenizer压缩token密度并对齐token频率；③构建双分支框架，将LLM的离散特征与视觉连续特征互补融合；

**🔧 技术方法**

使用LLM（Qwen3‑Embedding）、多层感知机对齐模块（EA）、轮廓提取器、速度提取器、Adapter、Triplet+Cross‑Entropy 损失以及视觉主干网络（DeepGaitV2‑P3D、GLGait）；

**📊 数据集**

Gait3D、GREW、SUSTech1K三大步态数据集；

**📈 对比分析**

在Rank‑1、Rank‑5和mAP指标上与多种基线（GaitSet、GaitPart、GaitGL、DeepGaitV2、GLGait等）对比，SilLang在Gait3D、GREW分别提升到78.4%/81.2% Rank‑1，SUSTech1K整体Rank‑1提升1.9%，显著优于现有方法；

**⚠️ 局限性**

依赖预训练LLM且冻结，导致对新数据适应性有限；token数量受LLM上限限制，长序列处理受限；token频率对齐需额外步骤，跨数据集迁移可能不稳定；训练样本量相对较小，LLM特征可能不完全契合；计算开销相对较高。

---

## 220. Approximation Schemes and Structural Barriers for the Two-Dimensional Knapsack Problem with Rotations

**arXiv ID:** 2603.23970 | [PDF](https://arxiv.org/pdf/2603.23970v1)

**作者:** Debajyoti Kar `[一作]` (Indian Institute of Science), Andreas Wiese `[通讯]` (Technical University of Munich)

**通讯引用:** 2038 | [OpenAlex ID](https://openalex.org/A5067503013)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文针对二维几何背包问题（2DK）及其可旋转变体（2DKR）提出了多项近似算法，并证明了在不同情形下的近似比和时间下界。

**💡 创新点**

核心创新包括：
- 引入“资源收缩”引理，使得在旋转允许的情况下，存在用常数个矩形容器实现（1+ε）近似的结构；
- 将该结构推广到带权情况，构造了 1.497+ 近似方案，并把无旋转情形的最佳比率从 13/7≈1.857 降低至 1.858；
- 在无旋转情形下证明了 13/7+ 的改进算法，同时给出了一条基于 k‑Sum 猜想的 n^{Ω(1/ε)} 运行时间下界，揭示了该问题的计算难度。

**🔧 技术方法**

主要技术手段：
- 走廊分解框架（Corridor Decomposition）与容器打包（Container Packing）相结合，得到有限个“厚”/“薄”容器；
- 资源收缩（Resource Contraction）Lemma，用于在压缩容器高度/宽度时仅牺牲常数项项目；
- 动态规划与广义分配问题（GAP）求解，处理容器内部的项目分配；
- L&C* 结构和 L‑形/十字形容器的引入，处理带有旋转且有“巨大”项目的特殊情况；
- 证明下界时的 k‑PartSum → 2DK/2DKR 归约。

**📊 数据集**

该工作为理论研究，实验使用人工构造的测试实例（如带有“巨大”项目的实例、skewed 项目集合等），没有公开的真实数据集。

**📈 对比分析**

实验/分析结果：
- 对于无旋转的 cardinality 2DK，提出了 (1+ε) 的 PTAS；
- 对于可旋转的 cardinality 2DKR，提出了 (1+ε) 的 PTAS；
- 对于带权且所有项目为 skewed 的 2DKR，得到 (1+ε) 的 PTAS；
- 对于一般带权 2DKR，得到 1.497+ 的近似算法；
- 对于无旋转的 weighted 2DK，得到 1.858+ 的近似算法；
- 证明了在 k‑Sum 猜想下，任何 (1+ε) 近似方案的时间下界至少为 n^{Ω(1/ε)}，显示了该问题在多维情况下的本质难度。

**⚠️ 局限性**

限制与开放点：
- PTAS 仅针对 cardinality 或 skewed 情形，带权通用情况仍停留在 1.497+ 的常数近似；
- 资源收缩技术在 2DKR 上需要处理旋转带来的复杂性，若允许任意旋转（非 90°）可能无法得到类似结果；
- 下界基于 k‑Sum 猜想，若该猜想被推翻，下界也会随之改变；
- 目前未给出多维（>2D）背包问题的类似 PTAS 或下界。

---

## 221. From Pixels to Digital Agents: An Empirical Study on the Taxonomy and Technological Trends of Reinforcement Learning Environments

**arXiv ID:** 2603.23964 | [PDF](https://arxiv.org/pdf/2603.23964v1)

**作者:** Lijing Luo `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 23353 | [OpenAlex ID](https://openalex.org/A5047878798)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对近两千篇RL环境相关论文进行系统性量化分析，构建多维度分类并绘制演进轨迹；

**💡 创新点**

提出了基于文献自动化与语义统计的环境与任务多维度映射方法，揭示从物理模拟到语义驱动的范式分化；

**🔧 技术方法**

运用了文本挖掘、语义分割、统计可视化和多属性特征提取技术；

**📊 数据集**

基于2183篇公开论文构建的环境属性数据库，涵盖200+核心Benchmark；

**📈 对比分析**

通过对比不同时间段、维度的环境分布与能力需求，生成可视化树谱，显示“语义先验”与“领域泛化”两大生态，表明环境复杂度与智能性能呈正相关；

**⚠️ 局限性**

主要局限在于数据来源仅限公开论文、手工标签可能导致误判，未覆盖工业内部或实验室自研环境，且未对算法本身做性能对比。

---

## 222. Event-Driven Proactive Assistive Manipulation with Grounded Vision-Language Planning

**arXiv ID:** 2603.23950 | [PDF](https://arxiv.org/pdf/2603.23950v1)

**作者:** Fengkai Liu `[一作]` (University of Osaka), Liyun Zhang `[通讯]` (University of Tokyo)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5100601573)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于工作空间状态变化的事件驱动主动协作框架，在人机交互事件结束时提取稳定的前后状态快照，并利用视觉语言模型推理任务级目标，生成可执行的辅助动作。

**💡 创新点**

创新点在于：①将协作助手从传统的请求驱动转变为事件驱动；②设计事件状态机捕捉稳定的前后快照；③使用ID‑grounded 的结构化输出接口，使VLM生成的指令可直接映射到机器人动作；④结合多层安全校验保证执行可靠。

**🔧 技术方法**

技术包括：RGB‑D 感知、光流运动检测、FastSAM 实例分割、事件状态机、云端视觉语言模型（VLM）推理、预定义动作原语、ID映射与结构化规划、在线执行与结果验证。

**📊 数据集**

使用自定义的桌面数字方块协作数据集，共 40 次试验（20 可解、20 不可解），覆盖加减乘除等算式构造场景。

**📈 对比分析**

与 Post‑only、Always‑on 和 Request‑driven 基线对比。事件驱动方法在可解场景下 ESR 0.95、RSR 1.00，优于其他基线；在不可解场景下 ESR/RSR 0.60，较基线提升显著；每个成功案例仅需一次 VLM 调用，计算开销最小。

**⚠️ 局限性**

局限性：①对可解场景的放置动作仍易出现碰撞或位置偏差；②不可解场景中仍出现歧义误判，需要更丰富的状态证据；③方法仅在单步推理，缺乏更长时延的预测；④依赖于高质量的前后快照，若快照获取不稳定可能导致误判。

---

## 223. ORACLE: Orchestrate NPC Daily Activities using Contrastive Learning with Transformer-CVAE

**arXiv ID:** 2603.23933 | [PDF](https://arxiv.org/pdf/2603.23933v1)

**作者:** Seong-Eun Hong `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 ORACLE 模型，用于从零开始生成 NPC 的日常活动计划，并支持在已有部分活动约束下进行计划补全。

**💡 创新点**

创新点在于将 Transformer 与 CVAE 结合，加入对比学习来提升生成多样性与真实感，并能处理完整和条件两种生成任务。

**🔧 技术方法**

使用技术包括 Transformer 编码器/解码器、条件变分自编码器（CVAE）、对比学习（以活动可行性规则生成正负样本）以及自回归解码。

**📊 数据集**

训练与评估使用 CASAS smart home 数据集（Apartment 与 Home 两个子集），经过预处理后得到 12 类主活动的 5 分钟间隔序列。

**📈 对比分析**

通过与 LSTM‑VAE、Transformer、biLSTM 等基线在随机生成和条件生成两种情境下的 WD、LLM Score、REAL、Distinct‑10/15 等指标进行比较，ORACLE 在多项指标上均优于基线，表现出更低的分布差距、更高的真实度与更丰富的多样性。

**⚠️ 局限性**

局限性主要在于仅基于室内日常活动数据，缺乏个性化建模与室外行为的覆盖，且对不同环境的迁移性尚需进一步研究。

---

## 224. Self-Distillation for Multi-Token Prediction

**arXiv ID:** 2603.23911 | [PDF](https://arxiv.org/pdf/2603.23911v1)

**作者:** Guoliang Zhao `[一作]` (Tencent), Xingwu Sun `[通讯]` (Tencent)

**通讯引用:** 5531 | [OpenAlex ID](https://openalex.org/A5057665558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM预训练中引入自我蒸馏框架MTP-D并结合循环扩展策略，以提升多令牌预测（MTP）的接受率和推理速度。

**💡 创新点**

创新点在于梯度分离的TopN选择自我蒸馏与循环扩展机制，使MTP头在保持主头性能的同时显著提升接受率，并支持8-16头扩展。

**🔧 技术方法**

使用了自我蒸馏（TopN选取+停止梯度）与KL散度，循环扩展持续预训练以及多头推理加速的speculative decoding。

**📊 数据集**

主要在FineWeb-Edu-350B预训练数据上验证，并在AGIEval-en、GSM8K、MATH、NaturalQuestions、SimpleQA、SuperGPQA、TriviaQA等七个基准上测试。

**📈 对比分析**

与DeepSeek MTP以及原版MTP进行对比，MTP-D在4头时接受率提升7.5%对应22.9%的速度提升，循环扩展到16头可实现35%以上的加速，主头准确率保持不降。

**⚠️ 局限性**

局限在于仅验证预训练阶段，未探究后训练适配；理论上α_k、β_k与头数的最优关系未充分挖掘；未在更大模型或多样数据集上验证。

---

## 225. DUPLEX: Agentic Dual-System Planning via LLM-Driven Information Extraction

**arXiv ID:** 2603.23909 | [PDF](https://arxiv.org/pdf/2603.23909v1)

**作者:** Keru Hua `[一作]` (Midea Group), Xiaoguang Ma `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DUPLEX 双系统神经‑符号架构，用轻量 LLM 只做信息抽取，随后将抽取结果映射为 PDDL 供经典规划器求解，失败时通过高容量 LLM 反思并迭代修复。

**💡 创新点**

创新点在于将 LLM 的角色限定在结构化信息抽取，避免代码/计划生成失误，并引入失败触发的慢系统反思循环，显著提升长周期任务的可靠性。

**🔧 技术方法**

采用 Qwen3‑8B 进行信息抽取、基于规则的 PDDL 生成、Fast Downward 经典规划器，GPT‑4o 负责慢系统反思与诊断驱动的修复。

**📊 数据集**

在 12 个 IPC 经典领域（Barman、Blocksworld、Floortile、Grippers、Storage、Termes、Tyreworld、Visitall）和 4 个家庭场景（PC Assembly、Dining Setup、Cleaning、Office）共 600 次实验中评估。

**📈 对比分析**

与 LLM‑as‑Planner（仅 LLM 生成计划）和 LLM+P（先翻译再求解）基线对比，IPC 平均成功率 97.5%（DUPLEX）vs 71.9%（LLM+P）vs 11.9%（LLM‑as‑Planner）；家庭场景 83.5% vs 20% vs 27.2%；双系统相较单速系统提升 13.7%（IPC）和 32.6%（家庭）。

**⚠️ 局限性**

局限性包括需手工提供域 PDDL；慢系统反思循环会带来额外延迟，限制实时高频控制；仅文本抽取，面临视觉参照模糊和多模态不完整的问题。

---

## 226. Praxium: Diagnosing Cloud Anomalies with AI-based Telemetry and Dependency Analysis

**arXiv ID:** 2603.23890 | [PDF](https://arxiv.org/pdf/2603.23890v1)

**作者:** Rohan Kumar `[一作]` (Boston University), Ayse Kivilcim Coskun `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了Praxxium框架，结合软件安装日志与遥测监控实现云环境异常检测与根因推断。

**💡 创新点**

首次将软件依赖变更日志与基于变分自编码器的异常检测及贝叶斯因果推断相结合，精细化定位异常源。

**🔧 技术方法**

使用Prometheus/Jaeger收集遥测与追踪，PraxiPaaS生成SBOM日志，VAE进行无监督异常检测，CausalImpact进行因果推断，构建因果图。

**📊 数据集**

在OpenShift Kubernetes上部署DeathStarBench社交网络应用，利用Prometheus 30s采样的指标与人工注入的CPU/内存/磁盘/网络异常。

**📈 对比分析**

通过网格搜索评估窗口大小、步长、阈值三种超参数，最佳配置为窗口10min、步长5min、阈值2，宏F1>0.97；因果推断在10min/5min/2min部署间隔下全部9次实验均正确定位异常。

**⚠️ 局限性**

仅验证合成负载与单一应用，未覆盖配置错误、节点故障等真实场景；采样率低时因果图不完整；在大规模多千Pod环境下的计算与存储成本仍需进一步评估。

---

## 227. Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration

**arXiv ID:** 2603.23889 | [PDF](https://arxiv.org/pdf/2603.23889v1)

**作者:** Guopeng Li `[一作]` (Delft University of Technology), Julian F. P. Kooij `[通讯]` (Delft University of Technology)

**通讯引用:** 1797 | [OpenAlex ID](https://openalex.org/A5074902093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种离线安全强化学习算法 COX-Q，结合成本约束的乐观探索与截断分位数评论家，实现了在安全约束下的高样本效率与低训练成本。

**💡 创新点**

创新点在于：1）开发了成本约束乐观探索（COX）策略，在动作空间解决奖励与成本梯度冲突；2）采用截断分位数评论家（TQC）进行分布式价值学习并量化不确定性；3）通过自适应步长与动态 δ 调整实现数据收集成本控制。

**🔧 技术方法**

使用技术包括：SAC 为基础离线框架、Primal–Dual Lagrangian 优化、Policy‑MGDA 解决梯度冲突、Truncated Quantile Critics 与 CVaR 置信界估计、经验回放、动态 δ 调节与自适应步长。

**📊 数据集**

在 Safe Velocity、Safe Navigation（Brax）以及 SMARTS 安全驾驶仿真平台的三种场景上进行实验评估。

**📈 对比分析**

与多种线上基线（CUP、RCPO、PPOSimmerPID、CPPOPID）和离线基线（SACLag‑UCB、CAL、Distributional WCSAC、ORAC）比较，COX-Q 在样本效率、测试成本和训练成本控制方面普遍优于基线，并在 SMARTS 复杂驾驶任务中获得最佳安全性能。

**⚠️ 局限性**

主要局限在于：1）截断分位数评论家对 OOD 样本的多样性不足，导致不确定性估计受限；2）在稀疏成本任务（如 Safe Navigation）中成本估计偏差仍是瓶颈，需结合 HER 或优先经验回放等方法进一步提升。

---

## 228. Optimal Variance-Dependent Regret Bounds for Infinite-Horizon MDPs

**arXiv ID:** 2603.23926 | [PDF](https://arxiv.org/pdf/2603.23926v1)

**作者:** Guy Zamir `[一作]` (University of Wisconsin--Madison), Yudong Chen `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于UCB的单一算法FOCUS，能够同时处理平均奖励和γ-奖励的在线无限期MDP，给出最优的方差依赖回报误差上界。

**💡 创新点**

创新点在于：①将全优化的经验Bellman算子与剪裁（span-clipping）相结合，消除对1/(1-γ)的依赖；②实现了方差依赖的最优上界，并在有无bias-span先验时分别给出最优的低阶项；③通过平均到折扣的技术，实现了平均奖励场景下的最优性，填补了此前仅有EVI方法的空白。

**🔧 技术方法**

核心技术包括：模型基的经验转移核、带Bernstein奖励的UCB框架、全循环的价值迭代求解固定点、span-clipping、以及平均-折扣转换分析。

**📊 数据集**

无实验数据集，全部为理论证明与上界下界分析。

**📈 对比分析**

与已有UCBVI、γ-UCB-CVI、EFB、PMEVI-DT等方法相比，FOCUS在平均奖励和γ-奖励两种指标下都实现了最优的平方根方差项，低阶项仅为O(S^2A)（有先验）或O(S^3A)（无先验），显著优于此前的O(S^5/2A^3/2)或O(S^6A^4/3)等大烧录成本与高低阶项。

**⚠️ 局限性**

主要局限在于：低阶项仍保留了Γ因子，且在无bias-span先验时的烧录成本仍为Θ((S^3A)^{1/2})，尚未突破到Θ(SA)或更低；同时对于极小的bias-span实例仍可能存在可改进空间。

---

## 229. Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith

**arXiv ID:** 2603.23972 | [PDF](https://arxiv.org/pdf/2603.23972v1)

**作者:** Somaya Eltanbouly `[一作]` (Hamad bin Khalifa University), Samer Rashwani `[通讯]` (Hamad bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于古阿拉伯词典的检索增强生成（RAG）框架，提升LLM对古典《古兰经》和圣训的理解；

**💡 创新点**

将结构化的、跨时间的历史词典作为外部知识源，结合意图驱动的检索路由与微调的交叉编码器实现高精度检索；

**🔧 技术方法**

使用BM25+密集检索、跨编码重排序、意图分类与提示工程等技术；

**📊 数据集**

使用Doha Historical Dictionary of Arabic（DHDA）以及从中自动生成的查询-答案对；

**📈 对比分析**

与Fanar、ALLaM等阿拉伯LLM在检索增强下的性能对比，检索+重排序后达到MRR≈0.94、MAP≈0.61；在生成任务中，RAG后Fanar/ALLaM准确率超过85%，与大型Gemini模型仅差≈5%；

**⚠️ 局限性**

主要限制包括对阿拉伯音调与复合短语的区分能力不足、检索覆盖仍有限以及模型对指令遵循的敏感性问题。

---

## 230. PointRFT: Explicit Reinforcement Fine-tuning for Point Cloud Few-shot Learning

**arXiv ID:** 2603.23957 | [PDF](https://arxiv.org/pdf/2603.23957v1)

**作者:** Yankai Wang `[一作]` (Xi'an Jiaotong University), Dongxu Zhang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1764 | [OpenAlex ID](https://openalex.org/A5100640205)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PointRFT，基于强化学习的细调方法用于点云少样本分类

**💡 创新点**

首次将 GRPO 应用于点云任务，设计显式的准确度奖励与分散奖励以缓解分布偏移

**🔧 技术方法**

使用 GRPO、强化学习、准确度奖励、分散奖励等技术

**📊 数据集**

ShapeNet、ModelNet40、ShapeNetCore、ScanObjectNN 等公开数据集

**📈 对比分析**

与传统 SFT 以及混合 Pre‑S‑R 方案比较，PointRFT 在少样本场景（尤其是 ScanObjectNN）显著提升性能

**⚠️ 局限性**

仅验证分类任务，未扩展到分割或动作识别，且计算开销略高

---

## 231. An Empirical Analysis of Google Play Data Safety Disclosures: A Consistency Study of Privacy Indicators in Mobile Gaming Apps

**arXiv ID:** 2603.23935 | [PDF](https://arxiv.org/pdf/2603.23935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 232. Towards Energy-aware Requirements Dependency Classification: Knowledge-Graph vs. Vector-Retrieval Augmented Inference with SLMs

**arXiv ID:** 2603.23954 | [PDF](https://arxiv.org/pdf/2603.23954v1)

**作者:** Shreyas Patil `[一作]` (University of Calgary), Gouri Ginde `[通讯]` (University of Calgary)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5108426807)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种能量感知的检索增强小型语言模型框架，用于自动检测软件需求冲突。

**💡 创新点**

结合知识图谱检索与向量检索对比，提出基于能量、延迟和碳排放的多维评估框架，首次系统评估小模型性能与环境影响。

**🔧 技术方法**

使用7B–8B小型语言模型、Neo4j知识图谱检索、Milvus向量检索、Zero-shot/ Few-shot/ Chain-of-Thought 提示，以及 CodeCarbon 能耗监测工具。

**📊 数据集**

利用五个公开与工业需求数据集：OpenCOSS、WorldVista、PURE、UAV、IBM-UAV。

**📈 对比分析**

通过比较KGR、VSR和基线在Recall@K、Macro F1、能耗、碳排放和延迟等指标，发现KGR在大多数数据集能耗下降70%+、碳排放下降80%+，Recall几乎无损，Mistral+ZeroShot在性能与能耗平衡方面表现最佳。

**⚠️ 局限性**

仅评估7B–8B模型与英文数据集，未考虑大模型微调、多语言或更大规模工业数据，能耗测量仅覆盖推理阶段，实验规模受硬件与数据库部署限制。

---

## 233. Argument Mining as a Text-to-Text Generation Task

**arXiv ID:** 2603.23949 | [PDF](https://arxiv.org/pdf/2603.23949v1)

**作者:** Masayuki Kawarada `[一作]` (NTT DOCOMO), Masaaki Nagata `[通讯]` (NTT Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于TANL框架的文本到文本生成方法，统一完成论证挖掘的跨度识别、组件分类和关系判定；

**💡 创新点**

通过统一的生成形式消除后处理与多任务超参数调优，支持多种注解并直接利用大规模语言模型；

**🔧 技术方法**

采用TANL+预训练T5/FLAN‑T5+QLoRA低秩量化微调+Needleman‑Wunsch对齐；

**📊 数据集**

使用AAEC、AbstRCT、CDCP三大基准数据集；

**📈 对比分析**

与ILP、BLCC、LSTM‑ER、BiPAM‑syn、BART‑CPM、ST等传统及现有最优模型对比，FLAN‑T5‑XXL在Component‑F1/Relation‑F1均达到或超过前沿水平（AAEC‑essay 80.15/61.19，AbstRCT 72.86/47.66，CDCP 72.68/33.96）；

**⚠️ 局限性**

推理时间受输入长度影响较大，需要80GB显存的GPU，且对GPT‑4等LLM的兼容性仍需进一步验证；

---

## 234. High-Fidelity Face Content Recovery via Tamper-Resilient Versatile Watermarking

**arXiv ID:** 2603.23940 | [PDF](https://arxiv.org/pdf/2603.23940v1)

**作者:** Peipeng Yu `[一作]` (Jinan University), Chip Hong Chang `[通讯]` (Nanyang Technological University)

**通讯引用:** 8280 | [OpenAlex ID](https://openalex.org/A5029335324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的水印框架VeriFi，能在图像公开前嵌入版权标识与语义恢复信号，并在遭受AIGC伪造后实现版权验证、像素级篡改定位与高保真人脸恢复；

**💡 创新点**

创新点在于：① 通过无额外定位payload的水印引导定位，避免视觉失真；② 采用低维语义潜在空间嵌入，实现内容恢复；③ 设计AIGC攻击模拟器在训练阶段暴露网络于真实深度伪造扰动；

**🔧 技术方法**

技术包括：Swin‑Unet双分支对齐、Transformer跨注意力恢复、VAE潜在嵌入、Poisson混合与潜在空间混合攻击仿真；

**📊 数据集**

使用CelebA‑HQ与FFHQ两个高分辨率人脸数据集进行训练与评估；

**📈 对比分析**

在三类AIGC攻击（SD Inpainting、HD‑painter、Splicing）和多种GAN/扩散伪造下，与EditGuard、OmniGuard、StableGuard、WAM等对照，VeriFi在水印提取精度（≈97.7%）、定位F1（最高0.989）和恢复PSNR/SSIM（≈31/0.89）均遥遥领先；

**⚠️ 局限性**

局限性包括：对非人脸图像的适用性待验证、潜在空间嵌入维度与恢复质量之间仍存在折衷、对极端高分辨率/视频内容的实时性尚需进一步优化。

---

## 235. DepthArb: Training-Free Depth-Arbitrated Generation for Occlusion-Robust Image Synthesis

**arXiv ID:** 2603.23924 | [PDF](https://arxiv.org/pdf/2603.23924v1)

**作者:** Hongjin Niu `[一作]` (Xi'an Jiaotong University), Yuan Gao `[通讯]` (China Telecom)

**通讯引用:** 758 | [OpenAlex ID](https://openalex.org/A5086492116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种训练‑free 的 DepthArb 框架，通过注意力仲裁和空间紧凑控制来解决多对象的遮挡关系。

**💡 创新点**

将遮挡问题重新表述为注意力竞争的仲裁问题，并提出 Attention Arbitration Modulation（AAM）与 Spatial Compactness Control（SCC）两种深度顺序约束，实现对前后层次的显式控制。

**🔧 技术方法**

利用交叉注意力梯度反馈、布局约束、深度加权正则化、两阶段 latent 优化等技术，在不需要模型微调的前提下对 diffusion 生成过程进行空间与深度约束。

**📊 数据集**

使用自研 OcclBench（涵盖多级遮挡场景）以及公开的 OverLayBench；OcclBench 通过 MS‑COCO 语义类别与 ChatGPT 构建，覆盖从部分到复杂嵌套的遮挡情况。

**📈 对比分析**

与 SDXL、Layout Guidance、R&B、LaRender 等基线在 OcclBench 与 OverLayBench 上对比，DepthArb 在 mIoU、FOCR、BOR、FBS 等遮挡相关指标上显著优于所有对手，并在整体视觉质量上保持竞争力。

**⚠️ 局限性**

仍需人工或人工智能生成的深度标签，难以处理极度复杂或缺乏深度信息的场景；光照、阴影等细粒度视觉效应的处理仍有限。

---

## 236. DecepGPT: Schema-Driven Deception Detection with Multicultural Datasets and Robust Multimodal Learning

**arXiv ID:** 2603.23916 | [PDF](https://arxiv.org/pdf/2603.23916v1)

**作者:** Jiajian Huang `[一作]` (Great Bay University), Xiaochun Cao `[通讯]` (Wuhan University)

**通讯引用:** 26860 | [OpenAlex ID](https://openalex.org/A5068837264)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可审计的多模态谎言检测框架，能够基于音视频特征生成结构化的证据报告；

**💡 创新点**

①构建理由化数据集并增添中间推理链；②发布跨文化大规模数据集T4‑Deception；③设计SICS与DMC模块以提升小样本鲁棒性与跨域迁移；

**🔧 技术方法**

利用多模态LLM（如AffectGPT、Qwen‑7B等）+ LoRA微调，结合SICS（个体-共性协同）和DMC（模态一致性蒸馏）实现特征融合与一致性约束，并通过schema‑constrained方式生成报告；

**📊 数据集**

使用Bag‑of‑Lies、MU3D、DOLOs等传统基准，并新收集的T4‑Deception（1695样本，覆盖美、德、越、保四国）进行实验；

**📈 对比分析**

与零样本商业LLM、开源LLM、专用模型对比，三大基准上实现SOTA；在跨域和跨文化迁移实验中平均提升6‑8%准确率，T4‑Deception中保持4%以内误差；

**⚠️ 局限性**

受限于样本规模与文化覆盖仍不足以完全避免极端个体行为或极端跨文化差异的误判；报告生成受LLM幻觉影响，且数据集聚焦“身份扮演”场景，其他谎言类型仍需进一步验证。

---

## 237. GenMask: Adapting DiT for Segmentation via Direct Mask

**arXiv ID:** 2603.23906 | [PDF](https://arxiv.org/pdf/2603.23906v1)

**作者:** Yuhuan Yang `[一作]` (Shanghai Jiao Tong University), Yanfeng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14980 | [OpenAlex ID](https://openalex.org/A5100645706)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将分割任务直接视为条件生成问题，使用 Diffusion Transformer（DiT）在同一模型中同时生成彩色图像和二值分割掩码。

**💡 创新点**

创新点包括：① 针对二值掩码的极端长尾时序采样策略，弥合 VAE 特征分布差异；② 在单一生成网络中实现分割与生成的统一训练；③ 结合 Vision‑Language Model 与 VAE 低阶特征注入，提升分割精度。

**🔧 技术方法**

采用的技术有：Latent Diffusion / DiT、时序采样调整（分割极端高噪声、生成中等噪声）、VAE 编码/解码、Vision‑Language Model（Qwen2.5‑VL‑7B）作为条件编码器、Classifier‑Free Guidance（仅用于生成）、MSE/BCE 监督方式、单步推理实现。

**📊 数据集**

使用的数据集包括：COCO‑stuff、ADE20K、PASCAL（语义分割转换为二值掩码）；RefCOCO、RefCOCO+、RefCOCO‑g（指代分割）；DiffusionDB、BLIP‑3o 等文本‑图像生成数据集。

**📈 对比分析**

通过与多种 SOTA 文本指代分割方法（RefCOCO 系列、ReasonSeg 等）在 mIoU、oIoU 等指标下比较，取得了领先或同级别的最佳性能；联合生成与分割训练进一步提升效果。

**⚠️ 局限性**

局限性：仍需在更大规模的 DiT 骨干上验证；对极端噪声分割的泛化能力尚有限；单步推理在高分辨率图像时可能受限；目前评估聚焦常规分割任务，未涵盖医学等更复杂场景。

---

## 238. Latent Bias Alignment for High-Fidelity Diffusion Inversion in Real-World Image Reconstruction and Manipulation

**arXiv ID:** 2603.23903 | [PDF](https://arxiv.org/pdf/2603.23903v1)

**作者:** Weiming Chen `[一作]` (Southern University of Science and Technology), Zhihai He `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2862 | [OpenAlex ID](https://openalex.org/A5045519742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对扩散模型的逆向过程（diffusion inversion），提出了两项技术：在每一步学习潜在偏置向量（Latent Bias Optimization, LBO）以对齐逆向与生成轨迹；以及对潜在表示进行联合优化（Image Latent Boosting, ILB），提升 VQAE 与 UNet 之间的匹配，从而显著提高真实图像的重建质量。

**💡 创新点**

创新点在于：①首次将潜在偏置向量作为中间变量，在逆向过程动态校正轨迹误差；②提出三种优化策略（梯度、数值、混合）实现高效、稳健的逆向；③通过对图像潜在表示的联合优化，使得 VQAE 重构误差降到最小，弥补了逆向与 VQAE 不一致的问题。

**🔧 技术方法**

主要技术包括：潜在扩散模型（LDM）、梯度优化与数值迭代、正则化损失（L1、SSIM、LPIPS）、多步潜在偏置学习、以及对 VQAE 与 UNet 的联合微调。

**📊 数据集**

实验使用的数据集有：COCO 2017 验证集（图像重建）、PIE‑Bench（图像编辑）以及 ImageNet‑1K 的少量罕见概念图像（稀有概念生成）。

**📈 对比分析**

方法与多种基线（DDIM, NPI, NTI, EDICT, AIDI, PTI, FPI, RIVAL, TIC, ReNoise, BDIA, BELM, GNRI, ExactDPM）在三个扩散模型版本（Stable Diffusion v1.5, v2.1, XL）上进行比较。LBI 在 PSNR、SSIM、LPIPS 上均实现了显著提升，例如在 SD1 上 PSNR 28.14、SSIM 0.8338、LPIPS 0.0332；在 SD2 上 PSNR 30.65、SSIM 0.8869；在 SDXL 上 PSNR 32.52、SSIM 0.9211。下游任务中，LBI 在图像编辑的结构距离与编辑后相似度之间取得最佳平衡，在稀有概念生成的 CLIP 相似度也优于原始 SeedSelect。

**⚠️ 局限性**

局限性：①对 VQAE 编码器和 UNet 的联合优化仍需在现有预训练模型上进行，无法完全消除两者训练不一致导致的细节损失；②梯度/数值迭代的多步优化在实时或大规模应用中仍有一定计算开销；③方法主要针对标准 LDM 结构，尚未验证在更大规模或不同架构（如 Stable Diffusion 3.0）的泛化能力。

---

## 239. HyDRA: Hybrid Domain-Aware Robust Architecture for Heterogeneous Collaborative Perception

**arXiv ID:** 2603.23975 | [PDF](https://arxiv.org/pdf/2603.23975v1)

**作者:** Minwoo Song `[一作]`, Heejin Ahn `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个无需训练、能够在动态多代理环境下应对模型架构和训练数据异质性的协同感知框架 HyDRA。

**💡 创新点**

①基于轻量级域分类器在推理时动态区分同质与异质代理；②混合使用中间融合与后期融合，并通过 Anchor‑Guided Pose Graph 优化异质代理定位误差；③完全无需再训练，零成本扩展。

**🔧 技术方法**

轻量级域分类器（Soft‑AP）、中间融合（Pyramid Fusion）、后期融合、Anchor‑Guided Pose Graph Optimization、Hungarian 匹配、IoU 评价、软 AP 等技术。

**📊 数据集**

V2X‑Real 大规模真实协同感知数据集。

**📈 对比分析**

与 Late Fusion、E2E、MPDA、HEAL、CodeFilling、GenComm、CoAlign 等基线在架构异质和潜在域异质场景下以 AP@0.3/0.5/0.7 进行对比，HyDRA 在多数指标上达到或超过现有最优方法，且在增加代理数量时保持性能不下降。

**⚠️ 局限性**

对域分类阈值敏感，极端定位噪声或完全未知模型架构时可能误判；缺乏针对动态训练场景的在线适配；在极大规模协作中计算开销仍需进一步评估。

---

## 240. MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation

**arXiv ID:** 2603.23896 | [PDF](https://arxiv.org/pdf/2603.23896v1)

**作者:** Gengluo Li `[一作]` (Chinese Academy of Sciences), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 11309 | [OpenAlex ID](https://openalex.org/A5100783219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了多语言多场景的文本-图像机器翻译基准MMTIT-Bench，并提出了融合场景认知、文本感知与翻译推理的CPR-Trans框架以提升端到端翻译性能。

**💡 创新点**

① 提供14种非英语/中文语言的真实场景多样化基准；② 设计了将认知、感知、推理三阶段结构化为CoT的CPR-Trans数据范式；③ 通过多轮推理提升模型解释性。

**🔧 技术方法**

采用视觉语言大型模型（VLLM）如Qwen3-VL、Gemini 2.5 Flash，结合Chain-of-Thought推理流程与多任务对齐训练技术。

**📊 数据集**

1,400张人工验证图像的MMTIT-Bench（涵盖14种语言）以及约165,200条由人工与合成数据生成的训练样本。

**📈 对比分析**

在Gemini 2.5 Flash、Qwen3-VL Judge和COMET指标下评估，CPR-Trans在7B模型上平均提升约+4–5分，甚至在某些语言组合中与更大模型相当，显示显著性能提升。

**⚠️ 局限性**

依赖大规模VLLM和昂贵的人工校准；推理流程受模型稳定性限制；目前仅覆盖图像级别，未扩展至长文档或视频。

---

## 241. PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation

**arXiv ID:** 2603.24078 | [PDF](https://arxiv.org/pdf/2603.24078v1)

**作者:** Yuheng Feng `[一作]` (Hong Kong Polytechnic University), Xingxing Zou `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1499 | [OpenAlex ID](https://openalex.org/A5078106569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PosterIQ基准，用于评估海报的理解与生成，涵盖 OCR、字体、布局、风格、构图、意图等多维度任务；

**💡 创新点**

创新点在于将海报设计视角细化为可度量的任务集，并提供量化指标与可复现评测；

**🔧 技术方法**

采用多模态大模型（LLM+视觉），扩散式图像生成器，结合手工标注与人工评估的多维度评价；

**📊 数据集**

使用7,765幅海报及822条生成提示，覆盖真实、专业与合成案例，涵盖7,765个图像-注释实例；

**📈 对比分析**

与主流闭源模型（GPT‑5、Claude‑Sonnet‑4.5、Gemini‑2.5‑Pro）及开源模型（MiniCPM‑V‑4.5、Qwen‑VL‑4B/8B 等）对比，闭源模型在高阶推理、风格识别上占优；生成方面，Gemini‑2.5‑Flash‑Image在构图与意图任务得分最高；整体性能显示闭源模型在理解方面更强，但生成细节（字体、密集内容）仍有欠缺；

**⚠️ 局限性**

局限包括：对高密度文字与复杂字体的识别仍不稳定，生成模型对组合与意图表达的精准度不足，且自动评测在创意水平上仍缺乏细粒度判断。

---

## 242. Robust and Secure Near-Field Communication via Curved Caustic Beams

**arXiv ID:** 2603.24077 | [PDF](https://arxiv.org/pdf/2603.24077v1)

**作者:** Shicong Liu `[一作]` (City University of Hong Kong), Robert Schober `[通讯]` (Friedrich-Alexander University Erlangen-Nuremberg)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对大规模MIMO近场通信的物理层安全，提出了利用电磁波焦线（caustic）设计曲线束形成方案，以在存在窃听者位置不确定的情况下实现鲁棒安全传输。

**💡 创新点**

创新点在于将近场焦点拉伸为连续曲线路径，利用曲线焦线绕过潜在窃听区域，同时将阵列划分为聚焦子阵列和曲线子阵列，实现能量高效且安全的双模式波束。

**🔧 技术方法**

采用的技术包括基于电磁波相位梯度的物理信息驱动设计，曲线焦线（caustic beam）闭式相位曲线推导，近场球面波模型以及与传统优化相对的低复杂度解析解。

**📊 数据集**

采用基于28 GHz频率、256个阵元、半波长间距的模拟实验场景，假设UE位置已知，窃听者位置误差为0.25 m半径圆。

**📈 对比分析**

与最优安全聚焦、范数界定、ADMM迭代等基线方法比较，实验显示曲线焦线方案在平均和最差情形下的保密率提升约80%，且实现时间显著更短。

**⚠️ 局限性**

限制包括对阵列相位量化与相位噪声的鲁棒性分析不足，以及仅考虑单用户单窃听者场景，尚需扩展到多用户、多窃听者及硬件非理想情况。

---

## 243. SOMA: Strategic Orchestration and Memory-Augmented System for Vision-Language-Action Model Robustness via In-Context Adaptation

**arXiv ID:** 2603.24060 | [PDF](https://arxiv.org/pdf/2603.24060v1)

**作者:** Zhuoran Li `[一作]`, Jinyu Gu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个无参数调优的记忆增强系统 SOMA，利用在线检索、因果归因、工具链编排和离线记忆蒸馏，在不修改原始 Vision‑Language‑Action（VLA）策略的前提下，提升其在 OOD 环境中的鲁棒性和成功率。

**💡 创新点**

核心创新包括：①Dual‑Memory RAG 将成功与失败案例分离并对比归因；②LLM 作为因果驱动的 Orchestrator，动态挑选并组合 MCP 工具；③异步双阶段记忆蒸馏，实现自我纠正和知识更新；④整个系统完全无需参数微调，可直接plug‑and‑play。

**🔧 技术方法**

采用的技术与方法有：CLIP + SAM + OpenCV 进行视觉检索与预处理；Qwen3‑VL‑32B 等大型语言模型做因果诊断与工具链生成；MCP 工具集（Paint‑to‑Action、Eraser、Prompt‑Refiner、Chaining‑Step、Encore）实现多模态纠错；离线双阶段记忆蒸馏算法不断优化 Dual‑Memory Bank。

**📊 数据集**

主要实验数据集为：LIBERO‑PRO benchmark（位置与任务变换）和自研 LIBERO‑  (视觉焦点、杂物清除、噪声提示、执行脆弱、任务链五大 OOD 维度)。

**📈 对比分析**

与 π₀、π₀.₅、SmolVLA 等基线模型在上述两套数据集上对比，SOMA 在 LIBERO‑ 上平均绝对成功率提升 56.6%，长周期任务提升 89.1%；在 LIBERO‑PRO 上平均提升 54.5%（布局变换）和 59.3%（语义变换），展示显著的鲁棒性和适应性。

**⚠️ 局限性**

局限性包括：①工具集仍相对有限，无法覆盖所有复杂 OOD 场景；②Dual‑Memory 的构建与维护需要人工或预先收集大量案例；③因果归因仍受 LLM 解释能力与训练数据的限制；④在真实机器人硬件上的大规模验证尚未完成。

---

## 244. Mitigating Object Hallucinations in LVLMs via Attention Imbalance Rectification

**arXiv ID:** 2603.24058 | [PDF](https://arxiv.org/pdf/2603.24058v1)

**作者:** Han Sun `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**通讯引用:** 61054 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出注意力失衡概念并在解码时通过 Attention Imbalance Rectification (AIR) 重新分配注意力与约束方差，从而显著降低 LVLM 中的对象幻觉并提升整体能力。

**💡 创新点**

① 定义跨模态（MAI）和 token 级（TAI）注意力失衡度量；② 将注意力重分配与方差约束投影正则化结合成轻量化的解码时干预；③ 证明两种机制协同降低幻觉并提升模型表现。

**🔧 技术方法**

解码时注意力重分配、方差约束投影正则化、注意力头敏感度评估、对比实验、基于注意力的可视化分析。

**📊 数据集**

CHAIR、POPE（MSCOCO 衍生）用于幻觉评估，MM‑Vet 用于通用能力评估，实验四大 LVLM：LLaVA‑1.5、MiniGPT‑4、InstructBLIP、Shikra。

**📈 对比分析**

与贪婪解码及六种 SOTA 解码方法（FarSight、VCD、DoLA、HALC、OPERA、AD‑HH）对比，AIR 在 CHAIR 上句子级 C_S 降低 35.1%、图像级 C_I 降低 22.6%；在 POPE 上平均表现最佳；在 MM‑Vet 上整体得分提升 15.9%（LLaVA‑1.5）/10%（MiniGPT‑4），同时保持或提升通用能力。

**⚠️ 局限性**

需手动挑选敏感注意力头且超参数对不同模型需调优，方法仅在解码阶段生效，对训练过程或跨模态任务（如视频、医学影像）尚未深入验证。

---

## 245. PCHC: Enabling Preference Conditioned Humanoid Control via Multi-Objective Reinforcement Learning

**arXiv ID:** 2603.24047 | [PDF](https://arxiv.org/pdf/2603.24047v1)

**作者:** Huanyu Li `[一作]` (Harbin Institute of Technology), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 Preference-Conditioned Humanoid Control (PCHC) 框架，使单一策略能够根据不同目标偏好在类人机器人中实现多种行为。

**💡 创新点**

采用 Beta 分布参数化的专家路由 (Preference Condition Injection) 替代传统门控网络，单一策略即可覆盖 Pareto 前沿；将多目标强化学习与类人控制结合，支持实时偏好切换。

**🔧 技术方法**

使用多目标强化学习（MORL）、PPO、Beta 分布路由、多专家网络、AMP 运动先验、GAE、Isaac Gym 仿真、ONNX 推理等技术。

**📊 数据集**

使用 LAFAN1 Retargeting Dataset 进行步态重现，以及自采集的跌倒恢复运动数据，实验仅基于本体感知，无外部传感器。

**📈 对比分析**

与 HoST 基线（单目标）及固定偏好策略对比，在跌倒恢复和步态任务中获得更大的 Pareto 覆盖（更高 hypervolume、较低 sparsity），并在仿真与真实 Unitree G1 机器人上验证了动态偏好切换的可行性。

**⚠️ 局限性**

目前仅支持二维目标，难以扩展到多目标；仅针对无外部传感器的盲目运动，未涉及视觉任务，Beta 路由在更高维偏好空间的推广仍待验证。

---

## 246. Beyond Semantic Priors: Mitigating Optimization Collapse for Generalizable Visual Forensics

**arXiv ID:** 2603.24057 | [PDF](https://arxiv.org/pdf/2603.24057v1)

**作者:** Jipeng Liu `[一作]` (Chinese Academy of Sciences), Xiao-Yu Zhang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 25640 | [OpenAlex ID](https://openalex.org/A5100419383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了CLIP等语义驱动视觉语言模型在深伪造检测中的“优化崩溃”现象，并基于理论分析提出CoRIT模型以提升非语义特征的可泛化性。

**💡 创新点**

创新点包括：①引入Critical Optimization Radius (COR)与Gradient Signal-to-Noise Ratio (GSNR)的理论框架，将优化不稳定与梯度信噪联系；②基于该框架设计Contrastive Regional Injection Transformer (CoRIT)，融合Contrastive Gradient Proxy、Region Refinement Mask、Regional Signal Injection和Hierarchical Representation Integration等新策略。

**🔧 技术方法**

主要技术包括：Sharpness-Aware Minimization (SAM)、CLIP-ViT-L/14 frozen backbone、Contrastive Gradient Proxy、Region Refinement Mask、Regional Signal Injection、Hierarchical Representation Integration等。

**📊 数据集**

实验使用的主要数据集有：FaceForensics++、DFDC、DFDCP、DFD、Celeb-DF v1/v2、NeuralTextures、FaceSwap、Face2Face、FF40、UniversalFakeDetect、LAION、ImageNet等。

**📈 对比分析**

与FFT、PEFT、UniFD、ForAda等多种基线在跨数据集、跨操纵、视频级别、生成模型等任务中对比，CoRIT在大多数指标上实现或超过SOTA，显著提升跨域AUC、mAP、mAcc等性能。

**⚠️ 局限性**

局限性在于对CLIP预训练的语义偏好仍有一定依赖，极低-COR的极端高逼真伪造仍具挑战；在图像降噪、遮挡等扰动下的鲁棒性仍有提升空间。

---

## 247. LGEST: Dynamic Spatial-Spectral Expert Routing for Hyperspectral Image Classification

**arXiv ID:** 2603.24045 | [PDF](https://arxiv.org/pdf/2603.24045v1)

**作者:** Jiawen Wen `[一作]` (Hong Kong University of Science and Technology), Haotian Shi `[通讯]` (Guangzhou University)

**通讯引用:** 260 | [OpenAlex ID](https://openalex.org/A5016118913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LGEST框架，用于超光谱图像的像素级分类；

**💡 创新点**

创新点包括：①利用深度空间-光谱自编码器DSAE压缩高维立方体并保留邻域信息；②构建跨交互混合专家特征金字塔CIEM‑FPN，通过跨注意力与残差Mixture‑of‑Experts动态融合多尺度特征；③设计局部‑全局专家系统LGES，采用稀疏激活的卷积与Transformer子专家并结合置信路由，实现局部纹理与全局上下文的自适应组合；

**🔧 技术方法**

使用技术包括深度自编码器、残差Mixture‑of‑Experts（RMoE）、跨注意力机制、局部‑全局专家路由（置信度门控）以及传统的卷积与Transformer模块；

**📊 数据集**

实验数据集涵盖四个公开基准：Indian Pines、Kennedy Space Center（KSC）、Houston2013 和 WHU‑Hi‑LongKou；

**📈 对比分析**

与13种基线方法（SVM、CNN、3D‑CNN、HybridSN、Transformer 等）在 OA、AA、Kappa 等指标上进行比较，LGEST 在所有数据集上均实现了显著提升，平均准确率提升约 5–10% 以上；

**⚠️ 局限性**

局限性包括：模型参数量和 FLOPs 相对较高，尤其在高维数据上计算开销显著；在极低标注样本场景下仍需进一步优化；未来可探索轻量化混合专家和少样本学习。

---

## 248. Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection

**arXiv ID:** 2603.24030 | [PDF](https://arxiv.org/pdf/2603.24030v1)

**作者:** Sa Zhu `[一作]` (Chinese Academy of Sciences), Bo Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 31669 | [OpenAlex ID](https://openalex.org/A5100688318)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于相位分解与对齐（Phase-wise Decomposition and Alignment, PDA）的开放词汇时序动作检测框架，能在未见类别上实现动作分类与定位。

**💡 创新点**

创新点包括：①利用大型语言模型的链式推理（CoT Prompting）将动作标签自动分解为多阶段（start/middle/end/global）描述；②设计文本注入前景过滤（TIF）模块，根据阶段语义自适应筛选视频片段；③采用自适应相位级对齐（APA）策略，对每个阶段的视觉-文本匹配结果进行动态加权聚合。

**🔧 技术方法**

技术手段：大型语言模型（GPT‑4o/其他LLM）进行CoT推理；CLIP文本编码器与视觉编码器构建跨模态特征空间；Transformer‑based 时间序列处理；多头注意力与自适应权重网络实现阶段级对齐与聚合。

**📊 数据集**

实验数据集：THUMOS14 与 ActivityNet v1.3，使用 50%/50% 与 75%/25% 的训练/测试拆分。

**📈 对比分析**

与十种前沿 OV‑TAD 方法（包括 Ti‑FAD、STOV、CSP 等）及闭集 TAD 基线进行对比，平均 mAP 在 THUMOS14 上提升至 65.4%（对比 Ti‑FAD 的 57.0%），在 ActivityNet v1.3 上提升至 53.1%（对比 Ti‑FAD 的 49.7%），在不同阈值和拆分下均保持领先。

**⚠️ 局限性**

局限性：依赖 LLM 生成描述，对 LLM 的推理质量和成本有一定依赖；相位数过多会增加计算开销；对动作本身缺乏明确阶段划分或极少见动作的泛化能力仍有提升空间。

---

## 249. i-IF-Learn: Iterative Feature Selection and Unsupervised Learning for High-Dimensional Complex Data

**arXiv ID:** 2603.24025 | [PDF](https://arxiv.org/pdf/2603.24025v1)

**作者:** Chen Ma `[一作]` (SUSTech), Shuhao Fan `[通讯]` (NUS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为 i-IF-Learn 的迭代高维无监督学习框架，能够同时进行特征选择和聚类，并输出可解释的影响特征子集及聚类标签。

**💡 创新点**

创新点在于：①构造一种自适应组合的特征评分统计量，融合伪标签监督与无监督信号；②使用高分布批评（Higher Criticism）阈值化实现无参数特征筛选；③在迭代过程中动态调整监督权重以抑制误标传播；④结合低维嵌入（PCA或Laplacian Eigenmaps）提升信噪比，尤其在非线性结构上表现突出。

**🔧 技术方法**

核心技术包括：自适应特征评分统计量、p值转换为标准正态分位数、Higher Criticism阈值化、低维嵌入（PCA、Laplacian Eigenmaps）、k-means 聚类、基于特征子集的后续深度聚类（DeepCluster、UMAP、VAE）以及理论上的一致性证明。

**📊 数据集**

实验使用了 18 个公开数据集：10 个基因微阵列数据集（样本数 40–300，特征数千）和 8 个单细胞 RNA‑seq 数据集（细胞数百到数千，基因数 2000–10000）。

**📈 对比分析**

与传统方法（k‑means、SpecGEM）、无监督特征选择方法（IFPCA、IFVAE）、深度聚类方法（DeepCluster、DEC、UMAP）以及基因数据专用方法（Seurat、SC3 等）进行比较。i‑IF‑Lap 在大多数数据集上取得最高准确率/ARI，平均排名最低，平均 regret 最小；在深度聚类前置特征筛选时进一步提升了 DeepCluster、UMAP、VAE 的性能。

**⚠️ 局限性**

局限性包括：①仅考虑单个全局影响特征子集，无法捕捉聚类特定的特征子集；②当前特征筛选为单个特征的边际评估，忽略特征交互与块效应；③对初始标签和常数 c 仍有一定敏感性，尽管稳健性良好，但在极端稀疏或噪声极大场景下可能受限；④迭代过程和嵌入方法选择对计算成本有影响，尤其在极大规模数据上仍需进一步优化。

---

## 250. ELITE: Experiential Learning and Intent-Aware Transfer for Self-improving Embodied Agents

**arXiv ID:** 2603.24018 | [PDF](https://arxiv.org/pdf/2603.24018v1)

**作者:** Bingqing Wei `[一作]` (Peking University), Yongtao Wang `[通讯]` (Peking University)

**通讯引用:** 4734 | [OpenAlex ID](https://openalex.org/A5100781631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ELITE 框架，使体感代理通过自我反思式知识构建和意图感知检索，持续从交互经验中学习并提升任务执行。

**💡 创新点**

创新点在于将经验提炼与基于计划轨迹的检索结合，形成闭环的经验驱动学习；通过意图感知检索实现跨任务的可复用策略迁移。

**🔧 技术方法**

技术手段包括视觉语言模型基础、Reflective Experience Distiller、Context Consolidator、CoT（链式思考）嵌入检索、BGE-M3 嵌入、在线无监督学习与监督式对比反思。

**📊 数据集**

使用 EB-ALFRED（AI2-THOR）和 EB-Habitat（Habitat 2.0）两个居家任务模拟数据集进行实验。

**📈 对比分析**

与零样本 VLM、ESCA 场景图增强以及 RL4VLM、ERA 等训练基准方法对比，在线无监督下分别提升 9%（EB-ALFRED）和 5%（EB-Habitat），监督下在 EB-ALFRED 上实现 70.8% 的成功率，优于现有训练方法。

**⚠️ 局限性**

局限性包括：依赖粗规划的 CoT 质量导致检索误差；单一全局策略池可扩展性受限；目前仅支持离散动作，难以迁移至连续控制；错误策略可能被误传播。

---

## 251. Thinking with Tables: Enhancing Multi-Modal Tabular Understanding via Neuro-Symbolic Reasoning

**arXiv ID:** 2603.24004 | [PDF](https://arxiv.org/pdf/2603.24004v1)

**作者:** Kun-Yang Yu `[一作]` (Nanjing University), Yu-Feng Li `[通讯]` (Nanjing University)

**通讯引用:** 8783 | [OpenAlex ID](https://openalex.org/A5002552788)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于程序辅助的神经符号推理框架TWT，用来完成表格-视觉多模态理解任务（TVMU），通过与外部代码执行环境交互实现信息抽取、元素建模和多步推理。

**💡 创新点**

创新点包括：① 两阶段训练策略——任务导向的监督微调（TO‑SFT）与自适应损失缩放的强化学习（AL‑GRPO），实现任务统一与复杂推理的双重提升；② 代码生成与执行的交互协议与沙箱环境，赋予模型可解释的、可复现的推理流程；③ 通过自动合成带推理轨迹的训练样本，克服现有数据缺乏推理记录的瓶颈。

**🔧 技术方法**

主要技术包括：多模态大型语言模型（Qwen3‑VL‑8B）、程序辅助神经符号推理、Python REPL 沙箱执行、任务导向监督微调、动态损失缩放的群组相对策略优化（AL‑GRPO）以及基于奖励的强化学习。

**📊 数据集**

使用八个公开数据集进行评估：多模态表格问答（WikiTQ、TabMWP、FinQA、TAT‑QA）和表格预测（Adoption、SkinCA、Pawpularity、Paintings），同时通过 Qwen3‑VL‑Plus 等模型合成约 2.7K 个带推理轨迹的训练样本。

**📈 对比分析**

与开放源代码多模态 LLM（Qwen3‑VL‑30B、InternVL‑3.5‑38B、HIPPO、Table‑LLaVA）以及 API 基准（GPT‑5.1、Qwen3‑VL‑Plus）进行对比。TWT 在问答任务上平均提升约 10% 的准确率，接近甚至超过 API 模型；在预测任务上也显著优于基线，体现了对复杂特征依赖和结构不完整表格的强鲁棒性。

**⚠️ 局限性**

局限性：对极其复杂或高度缺失的表格结构（如 TAT‑QA 中的代码执行准确率略低）仍存在挑战；依赖外部沙箱与代码生成质量，若代码生成失误或沙箱不可用则推理失败；数据合成过程高度依赖强大 LLM 的准确性，可能引入合成误差或幻觉。

---

## 252. Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping

**arXiv ID:** 2603.23998 | [PDF](https://arxiv.org/pdf/2603.23998v1)

**作者:** Yao Chen `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Tingwen Liu `[通讯]` (Institute of Information Engineering Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究Transformer模型的训练时结构稀疏化，提出Sparse Growing Transformer（SGT），通过在训练过程中逐步激活高熵注意头的递归循环来实现模型深度的自增长。

**💡 创新点**

创新点在于：①基于注意力熵的高熵头选择机制，②训练时的结构稀疏化（仅对细粒度参数进行循环），③遵循深到浅的成熟进化策略实现深度的逐步扩展。

**🔧 技术方法**

主要技术包括：注意力熵计算与归一化、递归循环（Attention Looping）、逐步增长调度（Progressive Growth Training）以及参数共享与节省计算。

**📊 数据集**

使用C4大型文本语料进行预训练，验证集覆盖ARC‑Easy、SocialIQA、OpenBookQA、SCIQ、WinoGrande、BasicArithmetic、CommonsenseQA、HellaSwag及MMLU四个子任务。

**📈 对比分析**

与Vanilla Transformer和Block‑Loop基线对比，SGT在相同计算量下在推理任务上提升约10%+（如WG、ARC‑E、CSQA），且额外FLOPs仅从16‑20%降至1‑3%，同时在长序列泛化上表现更佳。

**⚠️ 局限性**

主要局限在于实验规模仅至1.2B参数，未验证更大模型；缺少对超参数的细粒度探究；仅在公开基准上评测，缺乏跨域或实际部署场景验证。

---

## 253. Diet Your LLM: Dimension-wise Global Pruning of LLMs via Merging Task-specific Importance Score

**arXiv ID:** 2603.23985 | [PDF](https://arxiv.org/pdf/2603.23985v1)

**作者:** Jimyung Hong `[一作]`, Jaehyung Kim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提供了ACL会议论文的模板使用说明和排版规范。

**💡 创新点**

强调了与通用ACL会议格式的一致性，并给出完整的样例文件作为参考。

**🔧 技术方法**

使用LaTeX语言的.cls样式文件和对应的示例文档。

**📊 数据集**

无数据集；仅为排版说明。

**📈 对比分析**

无实验或性能比较，本文仅作为格式示例。

**⚠️ 局限性**

仅适用于ACL会议稿件，对其他会议或出版格式可能不适用。

---

## 254. Language-Grounded Multi-Agent Planning for Personalized and Fair Participatory Urban Sensing

**arXiv ID:** 2603.24014 | [PDF](https://arxiv.org/pdf/2603.24014v1)

**作者:** Xusen Guo `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5900 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出MAPUS，一个基于多代理LLM的参与式城市感知框架，允许参与者自主规划路线并通过协调实现公平分配。

**💡 创新点**

创新点在于将参与者建模为自适应LLM代理，结合个性化偏好和城市语义信息，使用协作式路由生成、基于历史的公平选择与交互式谈判优化。

**🔧 技术方法**

使用大型语言模型（LLM）、LangGraph框架、经典路线规划器、基于语义相似度的满意度评估，以及LLM驱动的协商与公平评分。

**📊 数据集**

在北京出租车GPS数据集T-Drive和东南亚网约车数据集Grab-Posisi上进行实验。

**📈 对比分析**

与六种基线（包括GraphDP、GPT‑4‑mini等）对比，MAPUS在覆盖率上与最优基线持平或更好，路径满意度提升约20–30%，公平性和稳定性也优于传统方法。

**⚠️ 局限性**

局限包括对LLM能力的依赖、模拟参与者画像可能偏差、以及隐私伦理问题。

---

## 255. DB SwinT: A Dual-Branch Swin Transformer Network for Road Extraction in Optical Remote Sensing Imagery

**arXiv ID:** 2603.24005 | [PDF](https://arxiv.org/pdf/2603.24005v1)

**作者:** Zongyang He `[一作]` (Chongqing Jiaotong University), Zhiguo Wang `[通讯]` (Inner Mongolia University of Technology)

**通讯引用:** 6247 | [OpenAlex ID](https://openalex.org/A5100430088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种双分支 Swin Transformer 网络(DB SwinT)用于光学遥感图像的道路提取。

**💡 创新点**

创新点在于将 Swin Transformer 的长程建模与 U‑Net 的多尺度融合相结合，设计局部‑全局双分支编码器并引入注意力特征融合 (AFF) 模块，实现细节恢复与语义连贯的协同提升。

**🔧 技术方法**

采用的技术包括 Swin Transformer、U‑Net 结构、双分支编码器、注意力特征融合（AFF）模块、窗口自注意力、位置编码与多尺度特征融合。

**📊 数据集**

实验使用 Massachusetts Roads 数据集和 DeepGlobe 数据集进行评估。

**📈 对比分析**

在两大数据集上与 LinkNet50、U‑Net、GAMSNet、SwinT 及 SwinT+U‑Net 等基准模型比较，DB SwinT 在 DeepGlobe 上 IoU 79.35%、F1 88.21%，在 Massachusetts 上 IoU 74.84%、F1 85.28%，显著优于其它方法。

**⚠️ 局限性**

局限性包括：仅在两大公开数据集验证，缺乏实时轻量化改造；三分支架构并未带来提升，可能导致特征冗余与优化难度增加；在极端复杂场景下仍可能出现过拟合与泛化不足。

---

## 256. Understanding the Challenges in Iterative Generative Optimization with LLMs

**arXiv ID:** 2603.23994 | [PDF](https://arxiv.org/pdf/2603.23994v1)

**作者:** Allen Nie `[一作]` (Google DeepMind), Ching-An Cheng `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于大语言模型（LLM）的生成式优化（generative optimization）学习循环中的三项核心设计决策：起始artifact、信用期限（credit horizon）和经验批量（experience batching），并通过实验验证这些决策对优化效果的影响。

**💡 创新点**

创新点在于将这三项隐藏设计决策系统化并与传统机器学习中的初始化、回溯长度、批量大小等概念对比，提出实践性指导并指出缺乏通用默认设置是当前生成优化难以推广的根本原因。

**🔧 技术方法**

使用技术包括：OptoPrime 生成优化框架、Claude Sonnet‑3.5 作为 LLM 终端、分层/多函数初始化策略、不同信用期限（一次步 vs 全局轨迹）以及批量大小实验；通过构造学习上下文（输入、输出、反馈、系统）实现迭代优化。

**📊 数据集**

实验数据集：MLAgentBench（Spaceship Titanic、Housing Price）、Atari 游戏（Pong、Breakout、Space Invaders、Asterix、Freeway、Enduro、Q*bert、Seaquest）、BigBench Extra Hard（Dyck Languages、Boolean Expressions、Geometric Shapes、Linguini、Disambiguation QA、Movie Recommendation、Boardgame QA、Causal Understanding）。

**📈 对比分析**

评估方法：Kaggle leaderboard percentile、Atari 得分标准化（对比 PPO/DQN 等深度 RL baseline）以及 BBEH 测试集准确率。结果显示，起始artifact 的模块化 vs 单函数、信用期限 的短 vs 长、批量大小 的 1/3/5 对性能有显著、且差异依赖任务，单一设置难以满足所有任务，部分设置可提升 10–20% 或接近人类水平。

**⚠️ 局限性**

局限性：仅探讨了三项设计决策，未涵盖反馈 oracle、LLM 选择、优化策略等其他自由度；实验规模有限且缺乏理论解释；结果高度任务特异，难以直接推广为通用默认设置。

---

## 257. CAKE: Real-time Action Detection via Motion Distillation and Background-aware Contrastive Learning

**arXiv ID:** 2603.23988 | [PDF](https://arxiv.org/pdf/2603.23988v1)

**作者:** Hieu Hoang `[一作]` (Viettel Group), Nam-Phong Nguyen `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 782 | [OpenAlex ID](https://openalex.org/A5026207172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种仅使用RGB输入的实时在线动作检测框架CAKE，通过动态运动适配器（DMA）实现光流特征蒸馏，并用浮动对比学习（Floating SupCon）分离动作与多样背景。

**💡 创新点**

创新点在于（1）DMA利用ODConv3D动态生成卷积权重，逼近光流而不需显式计算；（2）Floating SupCon仅聚焦动作类，允许背景自由分布，避免背景样本被强制聚成单一簇。

**🔧 技术方法**

采用跨模态知识蒸馏、深度可变卷积ODConv3D、GRU时序建模、Momentum Contrast+对比学习、Focal Loss等技术。

**📊 数据集**

使用THUMOS'14、TVSeries进行评测，并在Kinetics-400上预训练DMA。

**📈 对比分析**

与多种SOTA方法在同一backbone下对比，CAKE-R50在THUMOS'14上mAP 72.0%，CAKE-X3D仅RGB时FPS>100、mAP 67.1%，显著优于传统两流模型且CPU端实时率高达72+ FPS。

**⚠️ 局限性**

局限性包括：仅靠RGB易受光照、噪声等环境变化影响；GRU在极长时序捕捉能力有限；对比学习阶段需大负样本队列，显著占用显存，未来需考虑状态空间模型、Transformer或域自适应等改进。

---

## 258. Can we generate portable representations for clinical time series data using LLMs?

**arXiv ID:** 2603.23987 | [PDF](https://arxiv.org/pdf/2603.23987v1)

**作者:** Zongliang Ji `[一作]` (University of Toronto), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2433 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Record2Vec，利用冻结的大语言模型将 ICU 时间序列先摘要成自然语言，再用文本嵌入得到固定长度向量，作为多任务预测的通用输入，减少跨医院部署成本。

**💡 创新点**

通过“先摘要后嵌入”将多模态、缺失的 ICU 记录转换为语义统一的文本表征，实现模型跨医院迁移时无需再训练/微调；同时证明结构化提示能降低模型方差并保持准确率。

**🔧 技术方法**

冻结大型 LLM（Gemini‑2.0 Flash、MedGemma、Llama‑3.1）进行摘要，冻结文本嵌入模型（Qwen3）生成向量；与网格填充、TSDE、TimesFM、GenHPF 等传统时序表示进行对比。

**📊 数据集**

MIMIC‑IV、HiRID、PPICU 三个 ICU 队列，约 16 万住院窗口，涵盖多种生理、实验室及干预变量。

**📈 对比分析**

在七项预测任务（多变量预测、停留时长、死亡、给药、检验需求等）和两项隐私评估下，Record2Vec 在内部分布与跨站点迁移均可与网格填充、TSDE、TimesFM 竞争，迁移性能下降更小，few‑shot 学习更高效，平均精度保持不变。

**⚠️ 局限性**

需将患者记录发送至外部 LLM，存在隐私与合规风险；高算力/成本和较高延迟限制实时部署；实验仅经验性，缺乏理论解释与信息保留分析。

---

## 259. Human Factors in Detecting AI-Generated Portraits: Age, Sex, Device, and Confidence

**arXiv ID:** 2603.24048 | [PDF](https://arxiv.org/pdf/2603.24048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 260. The impact of sensor placement on graph-neural-network-based leakage detection

**arXiv ID:** 2603.24076 | [PDF](https://arxiv.org/pdf/2603.24076v1)

**作者:** J. J. H. van Gemert `[一作]` (Eindhoven University of Technology), M. Lazar `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 5634 | [OpenAlex ID](https://openalex.org/A5045512282)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了基于PageRank中心度的传感器布置方法，并将其与ChebNet图神经网络结合，用于水分配网络的压力重建、一步预测和泄漏检测。

**💡 创新点**

创新点在于提出了一种无模型、仅基于网络拓扑的PageRank中心度传感器布置策略，并证明其能显著提升GNN在重建、预测和泄漏检测上的性能。

**🔧 技术方法**

采用了PageRank算法进行节点重要性评估、Chebyshev图卷积网络（ChebNet）进行压力重建与预测，以及基于残差的阈值检测实现泄漏报警。

**📊 数据集**

使用了公开的EPANET Net1 benchmark 网络进行实验，并通过仿真产生的 108 小时压力时间序列（1 分钟采样）来训练和评估模型。

**📈 对比分析**

将PageRank中心度布置的传感器与随机/任意布置的传感器进行对比；实验显示在重建误差、预测误差及泄漏误报时间上，PageRank布置的模型均优于随机布置，误报时长缩短并提升了漏点检测的及时性。

**⚠️ 局限性**

局限性包括：对急剧的需求驱动压力跳变仍易产生误报；模型仅在中小规模网络上验证，缺乏大规模网络的可扩展性和泛化评估；并且未解决需求模式与泄漏信号的区分问题。

---

## 261. ConceptKT: A Benchmark for Concept-Level Deficiency Prediction in Knowledge Tracing

**arXiv ID:** 2603.24073 | [PDF](https://arxiv.org/pdf/2603.24073v1)

**作者:** Yu-Chen Kang `[一作]` (National Yang Ming Chiao Tung University), An-Zi Yen `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5012131890)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个扩展的知识追踪任务，既预测答案正确性，又预测学生在概念层面的缺失。

**💡 创新点**

创新点在于构建专家标注的 ConceptKT 数据集，结合概念级缺失预测，并提出基于概念对齐与语义相似度的历史响应选择策略。

**🔧 技术方法**

使用大语言模型（Gemini-2.0-Flash、Llama-4、o3-mini、DeepSeek-R1）在 in‑context 学习框架下进行推理，并与传统深度模型（DKT、DKVMN、GKT、SAKT、OKT）比较。

**📊 数据集**

使用扩展后的 MathEDU 数据集（ConceptKT），包含 4,048 条学生解题过程记录，平均关联概念 1.2441，缺失概念 1.112。

**📈 对比分析**

在答案正确性预测上，概念选择策略提升至约 71% 准确率；在概念缺失预测上，最佳模型 DeepSeek-R1 在 Same‑Concept 选择下 Macro‑F1 提升至 17.40%，显著优于未筛选或全量输入。

**⚠️ 局限性**

局限在于单一 prompt 模板、概念缺失标签为二元且未考虑学习深度，以及对不同模型的 prompt 优化未探索，导致部分学生中概念缺失预测仍低。

---

## 262. AD-Reasoning: Multimodal Guideline-Guided Reasoning for Alzheimer's Disease Diagnosis

**arXiv ID:** 2603.24059 | [PDF](https://arxiv.org/pdf/2603.24059v1)

**作者:** Qiuhui Chen `[一作]` (East China University of Science and Technology), Yi Hong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15850 | [OpenAlex ID](https://openalex.org/A5051418301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了一种多模态框架 AD-Reasoning，结合结构化 MRI 与六类临床数据，生成符合 NIA-AA 指南的结构化诊断与解释。

**💡 创新点**

创新点包括：双向交叉注意力融合机制、基于 GRPO 的强化学习与可验证奖励体系（保证格式、指南覆盖与推理一致性），以及构建的 AD-MultiSense 多模态 QA 数据集并采用指南校正的文本化理由。

**🔧 技术方法**

使用技术：3D Vision Transformer + Longformer 进行多模态编码，投影层映射到共享维度，双向交叉注意力融合，LLaMA 3.2-1B + LoRA 进行语言生成，图文对比预训练与重建恢复损失，GRPO 强化学习与可验证奖励（格式奖励、NIA-AA 诊断奖励、推理一致性奖励）。

**📊 数据集**

使用数据集：AD-MultiSense（10,378 访视，2,619 受试者），由 ADNI 与 AIBL 提供，包含结构化 MRI 与六类临床信息，并已转换为文本报告。

**📈 对比分析**

方法比较：与四个最先进的多模态 LLM（LLaVA-1.5-7B、LLaVA-Med、Med-PaLM-M、M3D-LaMed）以及文本与多模态分类基线（BERT、Roberta、Longformer、IRENE、AD-Trans、Alifuse）对比。AD-Reasoning 在 NLG 指标（BLEU、METEOR、ROUGE、BERT）和诊断指标（ACC、AUC、SEN、SPE）均取得最高值，例如 CN vs CI 的 ACC 93.33%/AUC 91.83%，CN vs MCI 的 ACC 92.82%/AUC 90.09%，显著优于基线。

**⚠️ 局限性**

局限性：数据规模和多样性仍受限，缺乏更广泛真实临床验证；模型仅覆盖结构化 MRI 与六类临床数据，未考虑其他影像或非结构化文本；强化学习奖励体系依赖现有指南，面对新指南或临床标准时可能需要重新校准。

---

## 263. Lagrangian Relaxation Score-based Generation for Mixed Integer linear Programming

**arXiv ID:** 2603.24033 | [PDF](https://arxiv.org/pdf/2603.24033v1)

**作者:** Ruobing Wang `[一作]` (Beijing Institute of Technology), Mingzhong Wang `[通讯]` (University of Sunshine Coast)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于拉格朗日松弛引导的分数生成框架（SRG），通过逆时 SDE 生成多样化且近似最优的可行解来加速 MILP 求解。

**💡 创新点**

创新点在于将拉格朗日松弛的可行性与最优性信息融入生成目标，利用分数网络联合建模所有决策变量，并通过指导的 SDE 实现多样化、可靠的候选解生成，显著提升了现有 Predict‑and‑Search 方法的效果。

**🔧 技术方法**

技术手段包括：图神经网络对 MILP 结构的嵌入；条件 U‑Net 结合变量级交叉注意力的分数网络；逆时扩散方程（reverse‑time SDE）与拉格朗日松弛正则化相结合；Straight‑Through 估计与 Gumbel‑Softmax 进行离散化处理。

**📊 数据集**

使用四类公开 NP‑hard MILP 基准（Set Covering、Combinatorial Auction、Capacitated Facility Location、Maximum Independent Set）以及多份大规模 MIPLIB 数据集进行实验。

**📈 对比分析**

与传统 Exact Solver（SCIP、Gurobi）、Predict‑and‑Search、ConPaS、Apollo‑MILP、L2O‑DiffILO 等基线相比，SRG 在目标值、搜索时间和 gap‑improvement 指标上均表现优异，尤其在中/大规模实例与零样本迁移场景中取得显著优势。

**⚠️ 局限性**

局限性包括：尚未深入解析 SRG 的数学收敛性质；对极大规模或结构差异显著的 MILP 可能仍需更多训练或调参；模型训练过程依赖于高质量标签，且对不同问题类型的泛化仍有待进一步验证。

---

## 264. Schema on the Inside: A Two-Phase Fine-Tuning Method for High-Efficiency Text-to-SQL at Scale

**arXiv ID:** 2603.24023 | [PDF](https://arxiv.org/pdf/2603.24023v1)

**作者:** Chinmay Soni `[一作]` (Sporta Technologies Private Limited), Hitesh Kapoor `[通讯]` (Sporta Technologies Private Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对印度梦幻体育平台CriQ的文本到SQL聊天机器人，构建并部署了一个自托管的8B参数模型，利用两阶段微调使模型内部化数据库模式，从而实现低延迟、高精度的查询回答。

**💡 创新点**

创新点在于提出了专为生产环境设计的两阶段微调方法：①在阶段一训练模型在完整上下文（包含完整模式和示例）的环境下学习SQL生成与显式模式记忆；②在阶段二仅使用简短触发器（数据库ID）与用户问题的最小提示，迫使模型依赖内部化的模式知识；该方法大幅削减提示长度，消除对第三方大模型API的依赖。

**🔧 技术方法**

使用技术包括：Qwen 3 8B 语言模型、LoRA参数高效微调、vLLM高效推理、分布式数据并行训练（8x NVIDIA H100 80GB）、自托管推理模型 Qwen/QwQ32B 生成训练数据，以及对齐、语义准确度评估的LLM辅助评测。

**📊 数据集**

数据集：从应用日志抽取 11,000 条真实用户查询，使用自托管推理模型生成高质量的问答-SQL 对；扩充为 80,000 条完整上下文示例、15,000 条显式模式记忆示例和 150,000 条仅包含问题+数据库ID 的最小提示示例；评测使用 30,000 条未见过的真实用户查询作为保留集。

**📈 对比分析**

通过与 Gemini 2.0 Flash 的 17k-token 复杂提示基线对比，最终模型在执行成功率上达到 98.4%（vs 95.6%）并在语义准确率上达到 92.5%（vs 89.4%）。单相训练与两阶段训练的 ablation 结果显示，单相训练仅能实现 74.96% 执行准确率；缺失显式模式记忆任务后，准确率降至 87.2%/79.5%；模型规模实验表明 8B 参数的 Qwen 3 最适合满足高准确度与可承载成本的平衡。

**⚠️ 局限性**

局限性包括：①数据库模式必须保持静态，模式变更需重新微调；②残余失败主要来自列别名歧义和复杂时序推理；③初始数据生成与微调投入较大；④在极端复杂查询场景下仍可能出现错误，未来可考虑基于偏好优化（DPO）等细化微调技术。

---

## 265. COVTrack++: Learning Open-Vocabulary Multi-Object Tracking from Continuous Videos via a Synergistic Paradigm

**arXiv ID:** 2603.24016 | [PDF](https://arxiv.org/pdf/2603.24016v1)

**作者:** Zekun Qian `[一作]` (Tianjin University), Junhui Hou `[通讯]` (City University of Hong Kong)

**通讯引用:** 10462 | [OpenAlex ID](https://openalex.org/A5031957432)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了连续标注数据集C‑TAO以及一个开放词汇多目标跟踪框架COVTrack++，实现了对任意类别的连续跟踪。

**💡 创新点**

创新点包括：1）首次构建26倍标注密度提升的连续视频训练集C‑TAO；2）通过多线索自适应融合(MCF)、多粒度层级聚合(MGA)和时序置信度传播(TCP)实现关联与检测的双向互补；3）采用时序一致性与语义可靠性双视角置信度动态调权，提升多模态特征融合质量。

**🔧 技术方法**

技术实现主要包括：多模态特征提取（外观、运动、语义），自注意力门控的置信度估计，跨层级跨框交叉注意力聚合，基于图匹配的置信度传播，基于CLIP蒸馏的语义表征，ResNet‑50 backbone + Faster‑RCNN detector，bi‑softmax关联与class‑conditional置信传播。

**📊 数据集**

使用的数据集：TAO（原始与连续标注版C‑TAO），以及无监督迁移到BDD100K进行零样本跨域验证；实验中均以C‑TAO作为训练集。

**📈 对比分析**

在TAO验证集与测试集上，COVTrack++分别取得35.4%/30.5%（novel TETA）和40.3%/38.9%（base TETA），比最新公开方法提升约4–5% TETA，尤其在novel类上提升4.8% AssocA、5.8% LocA；在BDD100K零样本测试中也获得46.7% TETA，领先对手约4%。

**⚠️ 局限性**

局限性包括：1）C‑TAO需要昂贵的人工连续标注，规模受限；2）框架仍依赖显式检测器，计算量较大；3）对极度罕见或快速变形的对象仍可能出现误检/漏检；4）在不同域（如高帧率工业监控）下的泛化能力尚未系统评估。

---

## 266. CVPD at QIAS 2026: RAG-Guided LLM Reasoning for Al-Mawarith Share Computation and Heir Allocation

**arXiv ID:** 2603.24012 | [PDF](https://arxiv.org/pdf/2603.24012v1)

**作者:** Wassim Swaileh `[一作]`, Dimitrios Kotzinos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套检索增强、结构化生成的伊斯兰继承法推理系统，涵盖符号规则生成、混合检索与LLM生成验证。

**💡 创新点**

创新点在于：1) 用符号规则构造包含完整推理轨迹的10万条合成案例；2) 结合BM25与向量检索并以交叉编码重排的混合检索方案；3) 采用结构化解码与后置验证，确保输出满足法律与数值约束。

**🔧 技术方法**

技术包括：符号计算器、检索增强生成（RAG）混合检索、BM25与语义向量检索、交叉编码reranker、Qwen3.5 LLM、结构化JSON解析与约束校验。

**📊 数据集**

数据集：QIAS 2026 5900例训练集、200例验证集、500例测试集，以及约10万条合成案例。

**📈 对比分析**

与官方基线比较：Qwen3.5 9B基线在验证集上仅得0.347 MIR‑E；最终系统在QIAS 2026 blind‑test榜单上获得0.935 MIR‑E，排名第一。

**⚠️ 局限性**

局限性：仅覆盖部分逊尼派与民法变体；对多代“hajb”逻辑、稀有分母/aw‑radd案例的推理仍易出错；评测仅靠MIR‑E，缺乏专家审查与实际可用性评估。

---

## 267. MIRROR: Visual Motion Imitation via Real-time Retargeting and Teleoperation with Parallel Differential Inverse Kinematics

**arXiv ID:** 2603.23995 | [PDF](https://arxiv.org/pdf/2603.23995v1)

**作者:** Junheng Li `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 15074 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于立体视觉的实时全身动作转移与遥操作框架MIRROR，结合GPU并行差分逆运动学与控制屏障函数，实现了THEMIS双足机器人的上半身实时遥操作。

**💡 创新点**

创新点在于提出了并行分布式差分逆运动学与任务空间连续化策略，并引入Lyapunov进度证书和CBF安全约束，使得解能够跳出局部最小，显著提升了鲁棒性与实时性。

**🔧 技术方法**

使用技术包括立体视觉骨架估计（ZED 2i+Body Tracking）、低通滤波与鲁棒化处理、GPU并行QP求解器、控制屏障函数、Lyapunov进度证书及实时闭环控制器。

**📊 数据集**

实验数据来源于在MuJoCo仿真下的随机目标轨迹，以及使用ZED摄像头、Meta Quest3 VR头盔和OptiTrack系统采集的人类运动记录，作为真实场景评估。

**📈 对比分析**

与单实例差分IK、分布式QP、全局IK等基线对比，MIRROR在仿真中实现了约4 ms的求解时间，同时将CBF违规与停滞事件显著降低；在硬件上端到端时延约为55 ms，任务空间跟踪误差在10–15 mm范围内。

**⚠️ 局限性**

局限性包括目前仅验证了上半身遥操作，完全身体扩展需要更多耦合约束；低层控制器对跟踪精度影响显著；大幅度姿态变换时仍可能出现约束冲突或安全边界超限；并行GPU求解对硬件资源有一定依赖。

---

## 268. Hierarchical Spatial-Temporal Graph-Enhanced Model for Map-Matching

**arXiv ID:** 2603.24054 | [PDF](https://arxiv.org/pdf/2603.24054v1)

**作者:** Anjun Gao `[一作]` (Soochow University), Shunyu Yao `[通讯]` (Ohio State University)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5111157886)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于分层自监督学习与空间时序监督学习的地图匹配模型HSTGMatch；

**💡 创新点**

采用分层轨迹表示、适应性轨迹邻接图、优化Graph Attention Networks、空间时序因子与衰减系数等创新设计；

**🔧 技术方法**

结合自监督掩码学习、优化GAT、Transformer‑Seq2Seq与图神经网络等技术；

**📊 数据集**

使用三套北京不同规模区域（约33k、79k、193k条）轨迹数据集；

**📈 对比分析**

与LSTM、ST‑RNN、HST‑LSTM、DeepMM、Transformer‑Based与GraphMM等基线对比，HSTGMatch在精度、召回率与F1上均显著优于对手，尤其在大规模地图上表现最佳；

**⚠️ 局限性**

仍受路网覆盖度限制、对极大地图的计算开销未完全解决、对无标签数据的依赖与标注成本等方面存在挑战。

---

## 269. MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning

**arXiv ID:** 2603.24044 | [PDF](https://arxiv.org/pdf/2603.24044v1)

**作者:** Andrea Manzoni `[一作]` `[通讯]`, Andrea Manzoni

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出 MoE-Sieve 框架，通过一次前向传递对 Mixture-of-Experts 模型的每层专家激活进行统计，挑选激活最多的前 25% 专家，仅对这些专家使用 LoRA 微调，显著降低参数量和训练时间；

**💡 创新点**

核心创新在于利用路由统计显著的层内不平衡信息，仅训练“热”专家，从而在保持性能的同时大幅提升微调效率；

**🔧 技术方法**

采用的技术包括：MoE 预训练模型、LoRA 参数高效微调、基于激活计数的专家排序、单次前向推理进行路由统计；

**📊 数据集**

实验使用三种 MoE 架构（OLMoE-1B-7B、Qwen1.5-MoE-A2.7B、DeepSeek-MoE-16B）和十个数据集（Spider、GSM8K、HellaSwag、ARC-Challenge、BoolQ、PIQA、MMLU、CodeAlpaca、Wikitext、MBPP）；

**📈 对比分析**

通过与全 LoRA 微调的基准对比，采用 25% 热专家策略在三项任务上与全 LoRA 差距不到 1pp，且训练参数减少 70–73%，检查点大小缩减 71–73%，训练时间缩短约 50%；

**⚠️ 局限性**

局限性包括仅在两种规模约 7B-14B 的 MoE 模型上验证，未检验更大规模模型；只对三类任务（结构化生成、算术推理、常识推理）进行评估；专家选择仅在训练前一次静态确定，未考虑训练过程中路由动态变化；

---

## 270. HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models

**arXiv ID:** 2603.24043 | [PDF](https://arxiv.org/pdf/2603.24043v1)

**作者:** Yeqi He `[一作]` (Hangzhou Dianzi University), Chenggang Yan `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 14348 | [OpenAlex ID](https://openalex.org/A5054311881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了训练无关的风格迁移方法 HAM，通过异质注意力调制实现内容与风格的平衡。

**💡 创新点**

创新点在于同时引入全局注意力调节（GAR）、局部注意力移植（LAT）以及风格注入噪声初始化（SINI），通过教师模型与学生生成器的注意力融合，既保留内容身份又增强风格表达。

**🔧 技术方法**

采用基于 Stable Diffusion 的潜在扩散模型、AdaIN、注意力投影融合、跨模态交叉注意力等技术，并兼容 SD2.1 与 SD3.5 架构。

**📊 数据集**

使用 MS-COCO（内容图像）和 WikiArt（风格图像）数据集进行实验与评估。

**📈 对比分析**

与多种文本驱动与图像驱动的 SOTA 方法（如 StyleID、DiffArtist、ControlNet 等）在 FID、LPIPS、ArtFID、CLIP-T 等指标上对比，HAM 在保持内容身份的同时实现更高的风格强度，整体指标均优于对手。

**⚠️ 局限性**

局限性：对极其抽象或超现实主义艺术风格的迁移效果仍有限。

---

## 271. SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons

**arXiv ID:** 2603.24039 | [PDF](https://arxiv.org/pdf/2603.24039v1)

**作者:** Haiyang Xu `[一作]` (University of California San Diego), Zhaowen Wang `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视觉生成的三阶段流水线，能够从扁平化的矢量图标中恢复可编辑的语义层级结构。

**💡 创新点**

创新点包括：将语义分割转化为可控着色任务并利用扩散模型实现分割与无遮挡完成；首次构建了大型标注图标语义分割与完成数据集；提出整数线性规划的层级排序方法，实现完整的层级重建。

**🔧 技术方法**

采用扩散式生成模型进行可控着色和语义分割；利用扩散模型和修复模型完成隐藏区域的形状恢复；整数线性规划求解层级排序；基于微调的SAM与 GPT-4o 生成数据集。

**📊 数据集**

构建了 8,567 张图标的 -Segmentation 数据集和 50,000 组三元组的 -Completion 数据集，并提供了 48 张独立测试图标；数据来源包括 LayerPeeler、GPT-4o 与 gpt-image-1 生成。

**📈 对比分析**

与 gpt-image-1、SAM2 以及精调后的 SAM2^* 进行对比。模型在可见区域 mIoU 上提升约 5.0、PQ 提升约 16.7；在完成任务上 mIoU 达 85.2，Chamfer Distance 达 46.6，优于其他基线，体现了更高的分割与完成质量。

**⚠️ 局限性**

当前方法仅适用于黑白线稿，彩色或多样化风格图标的迁移仍需训练；对高度缠绕或遮挡严重的图标存在失效；缺乏对更复杂图形语义层级的全面评估。

---

## 272. SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision

**arXiv ID:** 2603.24036 | [PDF](https://arxiv.org/pdf/2603.24036v1)

**作者:** Avigail Cohen Rimon `[一作]` (Technion Israel Institute of Technology), Or Litany `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SpectralSplats，一种将 3D Gaussian Splatting 的追踪损失从空间像素域迁移到频域的全局监督框架，解决大位移初始时梯度消失的问题。

**💡 创新点**

创新点包括：① 使用全局复杂正弦基（Spectral Moments）代替局部像素损失，构建全局吸引区；② 基于相位包裹推导的频率退火（Frequency Annealing）计划，实现低频向高频的逐步收敛；③ 该框架可无缝插拔到任意形变参数化（MLP 或直接控制点），保持模型无关。

**🔧 技术方法**

主要技术：3D Gaussian Splatting 渲染、可微分追踪、FFT 计算 Spectral Moments、基于频率退火的损失加权、与传统像素 L2、LPIPS 结合的多阶段优化。

**📊 数据集**

实验数据集：SC4D（4D 动画与对应 3DGS 模型）和 GART（真实视频与重建的犬类 3DGS），覆盖合成与真实场景。

**📈 对比分析**

与仅使用像素 L2 或 LPIPS 的标准追踪方法对比，采用不同参数化（MLP、Direct Morph Field）和不同目标损失。结果表明 SpectralSplats 在大初始位移时保持高 PSNR、SSIM、低 LPIPS，优于基线，且在训练和新视角上表现更稳健。

**⚠️ 局限性**

限制：需要预先存在的 Canonical 3DGS 资产，难以直接用于从未标记视频的全局重建；对极高频细节的恢复仍需后期像素/LPIPS 细化；频率退火调度对不同场景可能需要手动调整。

---

## 273. From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs

**arXiv ID:** 2603.24034 | [PDF](https://arxiv.org/pdf/2603.24034v1)

**作者:** Xiaoyong Guo `[一作]` (Xinjiang University), Wei Shi `[通讯]` (Timekettle)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对语音大模型（Speech-LLMs）中的上下文暴露偏差（contextual exposure bias），提出统一训练框架，结合教师错误知识、上下文 dropout 与直接偏好优化（DPO）来提升在真实历史上下文下的语音识别性能。

**💡 创新点**

创新点在于：① 将强 ASR（Whisper large‑v3）产生的误差历史直接作为训练上下文，实现训练与推理的上下文对齐；② 通过上下文 dropout 规避对错误历史的过度依赖；③ 用 DPO 在选定的错误案例上显式优化模型对正确输出的偏好，从而抑制错误累积与提升鲁棒性。

**🔧 技术方法**

技术手段包括：Speech-LLMs（Vicuna‑7B‑v1.5）+ Whisper 编码器特征提取，LoRA 微调（分为 SFT 与 DPO 两个模块），Teacher Error Knowledge（Whisper 生成的历史），Context Dropout，Direct Preference Optimization（DPO）以及 beam search 推理与 LoRA 强度调节。

**📊 数据集**

使用的数据集：TED‑LIUM‑3（训练、验证、测试）进行 in‑domain 评估；LibriSpeech（test‑clean/test‑other）进行 zero‑shot cross‑domain 评估；实验中还使用 Whisper 对训练集进行离线解码以生成教师错误历史。

**📈 对比分析**

与无上下文基线、oracle 训练、仅 SFT 等方法对比。实验显示：在 N=2、Context‑Whisper + 0.5 Dropout 的 SFT，WER 下降至 5.47%（比无上下文基线 7.89% 改进 2.4%）。加入 DPO 后进一步降至 5.17%；在 LibriSpeech 上，SFT 仅略有提升，而 DPO 可将 LS‑Ave. 从 7.32% 降至 7.02%，表现出显著的跨域鲁棒性。DPO 还在“无关上下文攻击”实验中表现出最小的性能下降。

**⚠️ 局限性**

主要局限：① 仅评估顺序、单说话者场景，未处理多说话者重叠或“鸡尾酒会”环境；② 教师错误知识仅基于单一 Whisper 模型，可能不足以覆盖所有 ASR 误差模式；③ 需要手工调节 DPO 推理强度 γ，过大会导致奖励过度优化。

---

## 274. QuadFM: Foundational Text-Driven Quadruped Motion Dataset for Generation and Control

**arXiv ID:** 2603.24021 | [PDF](https://arxiv.org/pdf/2603.24021v1)

**作者:** Li Gao `[一作]` (Alibaba Inc), Ziqiao Li `[通讯]` (Alibaba Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了QuadFM大规模高保真四足机器人运动‑语言数据集，并提出Gen2Control RL联合训练框架，实现文本到运动的实时生成与控制。

**💡 创新点**

创新点在于三层级文本注释的丰富交互行为集成，以及将生成与跟踪联合训练以提升物理可执行性。

**🔧 技术方法**

采用运动VAE、Diffusion模型、MotionGPT3、PPO强化学习、离线动力学校正等技术。

**📊 数据集**

数据集包括QuadFM（11,784段运动，35,352句描述）和对比的DogML。

**📈 对比分析**

与MotionGPT3和Ground Truth对比，人类评测显示Gen2Control在稳定性、文本‑运动对齐、平滑度、自然度上均显著提升，端到端延迟<500 ms。

**⚠️ 局限性**

限制在于对复杂多意图或时序约束命令的适配不足，以及对多机交互场景的覆盖仍有限。

---

## 275. Bridging Computational Fluid Dynamics Algorithm and Physics-Informed Learning: SIMPLE-PINN for Incompressible Navier-Stokes Equations

**arXiv ID:** 2603.24013 | [PDF](https://arxiv.org/pdf/2603.24013v1)

**作者:** Chang Wei `[一作]` (Agency for Science Technology and Research), Pao-Hsiung Chiu `[通讯]` (Agency for Science Technology and Research)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种名为SIMPLE-PINN的新型物理信息神经网络框架，用于求解不可压Navier‑Stokes方程，特别针对高雷诺数和复杂几何的流动场景。

**💡 创新点**

核心创新在于将传统SIMPLE算法中的速度‑压力耦合校正项转化为PINN的额外损失函数，利用数值微分与自动微分的混合策略，使网络能够在训练过程中直接强制满足连续性和动量方程的耦合约束，从而显著提升收敛速度和数值稳定性。

**🔧 技术方法**

使用技术包括：物理信息神经网络（PINN），多层感知机（MLP）与频率退火映射，SIMPLE算法的速度‑压力校正损失，有限体积法（FVM）离散，Taylor展开与二阶外推近似，数值微分（ND）与自动微分（AD）混合，Adam+L‑BFGS优化，动态学习率调度以及自适应采样与硬边界处理。

**📊 数据集**

采用了一系列数值基准案例作为数据集：Re=20000 的升压腔流、波浪通道、NACA0012 机翼（Re=500/1000，AOA 0°/7°）、三圆柱流动、Re=100 的圆柱无稳流和Rayleigh‑Taylor不稳定性（Ra=10^6）。参考解来自高精度CFD（如ANSYS Fluent、OpenFOAM）或已有文献的数值解，未使用实验数据。

**📈 对比分析**

通过与多种现有PINN（JAXPI、PirateNet、TSONN、SOAP、FFV‑PINN、ND‑PINN 等）在相对 L₂ 误差、均方误差（MSE）和训练时长等指标进行对比。SIMPLE‑PINN 在高雷诺数升压腔中以 0.124 h 内完成训练，误差约为 2.71×10⁻²（压力）和 3.65×10⁻²（速度），比 FFV‑PINN 的误差低 2‑3 倍且训练时间更短；在所有基准中误差普遍低于其它方法且训练时间保持在几百到几千秒之间，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：在多尺度或长时间无稳流场中收敛仍相对缓慢；训练耗时仍高于传统 CFD；忽略邻域贡献的校正项可能限制收敛速度；缺乏自适应网格、时间步与损失权重调度等先进训练策略；对更复杂多物理耦合问题的适应性尚未验证。

---

## 276. UW-VOS: A Large-Scale Dataset for Underwater Video Object Segmentation

**arXiv ID:** 2603.24006 | [PDF](https://arxiv.org/pdf/2603.24006v1)

**作者:** Hongshen Zhao `[一作]` (Southeast University), Wankou Yang `[通讯]` (Southeast University)

**通讯引用:** 5044 | [OpenAlex ID](https://openalex.org/A5100748706)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了UW-VOS水下视频目标分割大规模数据集，并设计了参数高效的SAM-U模型进行域适配。

**💡 创新点**

创新点包括：①构建首个含409类别、309k掩码的水下VOS基准；②在SAM2基础上仅插入轻量级UDA块，训练参数仅占2%；③通过细粒度属性分析揭示小目标、伪装、退出再入为关键瓶颈。

**🔧 技术方法**

技术方法：半自动数据收集+人机验证、颜色恢复+SAM2自动传播、轻量化Domain Adapter+Spectral Channel Gate模块、基于Hiera编码器的可训练块。

**📊 数据集**

使用的数据集为UW-VOS（1,431段视频，409类别，309,295掩码）以及公开的开阔域VOS数据集如DAVIS、YouTube-VOS做预训练。

**📈 对比分析**

与现有9种VOS方法比较，跨域零训练下平均下降13点；在UW-VOS上训练后SAM-U以87.4 J&F击败SAM2-B+（85.9）并接近最优，速度与内存与SAM2基本相同。

**⚠️ 局限性**

局限性：仍在小目标、伪装和退出再入场景表现不足；对光照/颜色偏移的补偿不完全；需要足够的标注数据才能有效迁移，且模型对多目标复杂交互仍易出错。

---

## 277. PAC-DP: Personalized Adaptive Clipping for Differentially Private Federated Learning

**arXiv ID:** 2603.24003 | [PDF](https://arxiv.org/pdf/2603.24003v1)

**作者:** Hao Zhou `[一作]` (Nanjing University of Post and Telecommunication), Hui Cai `[通讯]` (Nanjing University of Post and Telecommunication)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对联邦学习中不同隐私预算的客户端，提出了PAC-DP个性化自适应梯度裁剪框架，利用服务器端代理数据进行离线模拟曲线拟合，实现在线时基于预算的裁剪阈值选择。

**💡 创新点**

创新点在于将隐私预算与最优裁剪阈值映射关系通过离线模拟学习并在训练过程中动态应用，突破了传统固定阈值和经验式自适应裁剪的局限，提供了理论收敛保证和可重现的隐私计算。

**🔧 技术方法**

核心技术包括Per-example梯度裁剪、Gaussian噪声注入、RDP会计、离线代理数据模拟、多项式曲线拟合以及随轮次衰减的裁剪调度。

**📊 数据集**

实验数据集涵盖MNIST、CIFAR-10、CIFAR-100以及医疗Heart Disease四个任务，验证了方法在图像和表格数据上的通用性。

**📈 对比分析**

与固定阈值、rPDP-FL、NbAFL、CGM_Medium、AQC等基线比较，PAC-DP在相同隐私预算下平均提升约26%准确率，收敛速度提升约45.5%，并在异构预算场景下保持领先。

**⚠️ 局限性**

主要局限包括需要依赖代理数据进行离线拟合，若代理分布与真实分布偏差较大可能影响裁剪阈值；适用于记录级本地DP，未涵盖全局DP或参与度隐私；且对动态预算调整的适应性尚待进一步研究。

---

## 278. Transcending Classical Neural Network Boundaries: A Quantum-Classical Synergistic Paradigm for Seismic Data Processing

**arXiv ID:** 2603.23984 | [PDF](https://arxiv.org/pdf/2603.23984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 279. HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images

**arXiv ID:** 2603.23997 | [PDF](https://arxiv.org/pdf/2603.23997v1)

**作者:** Yumeng Liu `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8745 | [OpenAlex ID](https://openalex.org/A5100635702)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于Transformer的前向框架HGGT，能够在无标定的多视角图像中同时回归3D手网格与相机姿态；

**💡 创新点**

首次实现无标定多视角手部重建的feed‑forward 方法，并通过混合数据训练与大规模合成数据提升通用性；

**🔧 技术方法**

使用VGGT骨干、统一跨注意力模块、手部与相机解码头，以及多任务损失（手部、相机、投影一致性、负深度惩罚）；

**📊 数据集**

结合真实单视角数据、真实多视角数据（如HO3D、InterHand等）和自制合成多视角手物交互数据；

**📈 对比分析**

与基线POEM（使用真相机和预测相机）对比，HGGT在无标定场景下性能相当或更优，尤其在HO3D等数据集上超越基线；

**⚠️ 局限性**

依赖外部2D检测器进行手部裁剪，无法恢复绝对尺度（腕部深度），对极端遮挡或运动模糊的鲁棒性有限。

---

## 280. 6D Movable Antenna for Internet of Vehicles: CSI-Free Dynamic Antenna Configuration

**arXiv ID:** 2603.23991 | [PDF](https://arxiv.org/pdf/2603.23991v1)

**作者:** Maoxin Ji `[一作]` (Jiangnan University), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45825 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在高移动性的车联网环境中，提出一种六维可移动天线（6DMA）的CSI‑free动态配置方法，利用车辆运动预测与离线方向性响应先验来实时优化天线位置与朝向，最大化未来时段内的平均总速率。

**💡 创新点**

创新点在于：①结合车辆位置预测与方向性稀疏先验，避免了即时CSI估计的高复杂度；②设计离线网格化方向响应库并在线依据历史速率动态更新，形成低复杂度的贪心选择策略；③通过三维空间与三维旋转的离散网格构造满足硬件约束的可行配置集合。

**🔧 技术方法**

主要技术包括：六维可移动天线硬件模型、近场-远场混合信道模型、车辆运动预测模型、离线方向性响应建模、历史速率权重更新、基于优先级的贪心配置算法。

**📊 数据集**

使用仿真数据集：在一个300m×300m的城市十字路口场景中，车辆数K∈{30,35,40,45,50}，均匀分布在两条相交道路上，仿真周期10个时隙、基站位置中心。

**📈 对比分析**

与传统固定方位天线、圆形离散位置6DMA以及仅旋转的6DMA进行对比，实验表明所提方法在每10个时隙重配置时即可获得与每时隙重配置相近的总速率，并且在低功率或中等功率区间明显优于所有基线。

**⚠️ 局限性**

局限性包括：①仍需机械重配置，可能导致能耗和响应延迟；②对车辆运动预测误差敏感；③实验仅基于仿真，缺乏真实车联网实测验证；④离散网格和有限的方向样本可能限制在极端几何布局下的最优性。

---

## 281. CoCR-RAG: Enhancing Retrieval-Augmented Generation in Web Q&A via Concept-oriented Context Reconstruction

**arXiv ID:** 2603.23989 | [PDF](https://arxiv.org/pdf/2603.23989v1)

**作者:** Kaize Shi `[一作]` (University of Southern Queensland), Guandong Xu `[通讯]` (Education University of Hong Kong)

**通讯引用:** 10856 | [OpenAlex ID](https://openalex.org/A5051512158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CoCR-RAG框架，利用AMR进行概念蒸馏，将多源检索文档的核心概念提取并通过LLM重构为统一、信息密集的上下文，从而提升Web问答的RAG效果。

**💡 创新点**

创新点在于：①使用抽象意义表示（AMR）对多源文档进行语义级概念提炼，消除表面差异与冗余；②通过LLM构建概念导向的上下文，兼顾词汇与句法信息；③形成一个无参数化、可插拔的上下文重构模块，兼容多种LLM。

**🔧 技术方法**

主要技术：Abstract Meaning Representation (AMR) 解析与概念蒸馏算法、基于LLM的概念重构（prompt工程）、检索组件（Contriever/BM25）以及多模型推理。

**📊 数据集**

实验数据集：PopQA 与 EntityQuestions 两个基于维基百科的Web问答数据集。

**📈 对比分析**

与基线（Vanilla、Keywords、Summary、SelCon、LLMLingua）对比，CoCR-RAG 在 AUC 与准确率上均实现显著提升，尤其在文档数 K 较大、上下文窗口较大时提升更为明显，且对不同主干 LLM 具有良好的泛化能力。

**⚠️ 局限性**

局限性：1）依赖LLM的prompt驱动，受模型参数化不确定性影响；2）重构过程缺乏可解释的超参数调优，依赖模型提示质量；3）目前只探索AMR，未尝试其他语义结构（如依存句法、语义角色标注）。

---

## 282. Enhanced Mycelium of Thought (EMoT): A Bio-Inspired Hierarchical Reasoning Architecture with Strategic Dormancy and Mnemonic Encoding

**arXiv ID:** 2603.24065 | [PDF](https://arxiv.org/pdf/2603.24065v1)

**作者:** Florian Odi Stummer `[一作]` `[通讯]` (Martin Luther University), Florian Odi Stummer (Martin Luther University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为Enhanced Mycelium of Thought (EMoT) 的层级化、可记忆且可休眠的推理框架，用以增强大型语言模型的跨领域推理能力。

**💡 创新点**

创新点在于将真菌菌丝网络的分布式信息处理与战略性休眠、重新激活机制，以及多种记忆宫殿编码方式融合进单一的四层网络架构。

**🔧 技术方法**

技术实现基于Python，包含可插拔LLM后端（Claude、Gemini、Ollama等）、节点级推理、信任得分体系、Strategic Dormancy Controller、Memory Palace以及一组五个增强模块。

**📊 数据集**

实验数据主要来自三个复杂案例（临床、政策与AI治理）和15道短答题，采用LLM-as-Judge评估方法，并未使用公开标准基准。

**📈 对比分析**

与传统Chain-of-Thought进行对比时，EMoT在LLM-as-Judge评分中与CoT整体相当但在跨域综合评分上更优；然而在短答题准确率仅27%，远低于直接提示（100%）和CoT（73%）。

**⚠️ 局限性**

主要限制包括样本量极小、评估方法可能存在自我偏差与评判标准循环性、高达33倍的LLM调用与计算成本、缺乏对休眠激活效果的实证验证以及缺乏临床或专业领域的正式验证。

---

## 283. Rydberg Atomic Quantum Receivers for Wireless Communications: Two-Color vs. Three-Color Excitation

**arXiv ID:** 2603.24062 | [PDF](https://arxiv.org/pdf/2603.24062v1)

**作者:** Jian Xiao `[一作]` (Central China Normal University), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 43299 | [OpenAlex ID](https://openalex.org/A5060020877)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了三色五能级Rydberg原子量子接收机（3C5L‑RAQR）在无线通信中的应用，提出了等效基带信号模型并与传统两色四能级RAQR及经典导体天线接收机进行性能比较。

**💡 创新点**

创新点包括：①使用全红/红外低成本激光实现三光子共振，消除多普勒失配；②通过五能级耦合实现UHF/VHF频段的可访问性；③利用Liouvillian超算与完整的多普勒积分提供精确的开阔量子动力学数值解；④在同一物理平台上实现对高频微波与低频VHF的统一调度。

**🔧 技术方法**

所用技术主要包括：量子干涉诱导不透明（EIT）与三光子共振；超异频（superheterodyne）与平衡相干光检测（BCOD）；Liouvillian超算与特征值分解实现完整的动力学求解；以及射频信道模型、16‑QAM调制、Rayleigh衰落仿真等通信系统仿真技术。

**📊 数据集**

主要使用的参数来自公开的ARC原子数据库与实验测量值（如跃迁能级、偶极矩、自然衰减率等），并在仿真中设定蒸汽室温度、光束直径、激光功率、LO场强等物理参数。

**📈 对比分析**

通过对SNR、误块率（BLER）和容量的数值仿真，比较三种架构在不同噪声、功率与带宽条件下的表现。结果显示3C5L‑RAQR在低功率、低SNR环境下具备显著的灵敏度优势和更低的BLER；但其瞬时带宽较小，导致在高功率、高速传输场景下经典RF接收机仍优于3C5L。

**⚠️ 局限性**

局限性包括：①因三光子共振引入的多级耦合导致瞬时带宽受限；②系统对激光功率与LO场强的双重调节要求，降低了动态范围；③BBR热噪声虽对量子敏感度影响有限，但在高温环境下仍可能显著；④实现全红/红外激光的同步控制与光学布置相对复杂。

---

## 284. FinToolSyn: A forward synthesis Framework for Financial Tool-Use Dialogue Data with Dynamic Tool Retrieval

**arXiv ID:** 2603.24051 | [PDF](https://arxiv.org/pdf/2603.24051v1)

**作者:** Caishuang Huang `[一作]`, Xuanjing Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

未给出具体研究内容

**💡 创新点**

无创新点信息

**🔧 技术方法**

无技术细节

**📊 数据集**

无数据集

**📈 对比分析**

无对比方法或性能评估

**⚠️ 局限性**

限制：缺乏实验细节

---

## 285. From Untamed Black Box to Interpretable Pedagogical Orchestration: The Ensemble of Specialized LLMs Architecture for Adaptive Tutoring

**arXiv ID:** 2603.23990 | [PDF](https://arxiv.org/pdf/2603.23990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 286. A^3: Towards Advertising Aesthetic Assessment

**arXiv ID:** 2603.24037 | [PDF](https://arxiv.org/pdf/2603.24037v1)

**作者:** Kaiyuan Ji `[一作]` (Shanghai Artificial Intelligence Laboratory), Guangtao Zhai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于A^3-Law的广告图像美学评估框架并实现了对应的评估模型A^3-Align，支持逐层判断与诊断；

**💡 创新点**

创新点在于将广告美学拆解为三阶段Perceptual Attention、Formal Interest、Desire Impact，并通过链式思维(CoT)与多源奖励机制实现模型对这些规则的精准对齐；

**🔧 技术方法**

使用了多模态大语言模型（以Qwen3-VL-8B为基础），结合监督微调（SFT）和群组相对策略优化（GRPO）等强化学习技术；

**📊 数据集**

构建了30K张广告图像、120K条指令-回答对组成的A^3-Dataset，并提供CoT和工具调用的标注；

**📈 对比分析**

与超过20个主流开源/闭源多模态LLM在A^3-Bench上对比，A^3-Align在所有子任务（包括分类、定位、连续评分）均超过现有模型，尤其在广告图标定位mAP提升至0.701；

**⚠️ 局限性**

局限性包括在高度拥挤布局时易产生图标混淆/幻觉，以及对极简高级设计的属性校准不足。

---

## 287. Stochastic Dimension-Free Zeroth-Order Estimator for High-Dimensional and High-Order PINNs

**arXiv ID:** 2603.24002 | [PDF](https://arxiv.org/pdf/2603.24002v1)

**作者:** Zhangyong Liang `[一作]` (Tianjin University), Ji Zhang `[通讯]` (University of Southern Queensland)

**通讯引用:** 9107 | [OpenAlex ID](https://openalex.org/A5100329271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种统一框架 SDZE，实现 PINN 在高维高阶 PDE 下空间与内存均无关的训练；

**💡 创新点**

通过 Common Random Numbers Synchronization（CRNS）消除双随机方差，并用隐式矩阵自由低秩子空间 ZO 估计将参数探索方差从 O(P) 降到 O(r) 甚至 O(1)，从而突破传统反向传播的内存与计算瓶颈；

**🔧 技术方法**

结合随机空间估计（如 SDGD、STDE 等）、零阶（Zeroth-Order）优化、CRNS、隐式低秩子空间投影（利用 Kronecker 结构和 PRNG 重构）以及 JAX 前向推导；

**📊 数据集**

在多维 PDE 实验中验证，包括 Poisson、Allen‑Cahn、Sine‑Gordon、Korteweg‑de Vries、Kadomtsev‑Petviashvili、梯度增强 PINN 等，维度从 100 到 10 M；

**📈 对比分析**

与传统第一阶 SDGD、STDE、HTE、RS‑PINN、Forward Laplacian 等方法对比，SDZE 在单张 NVIDIA A100 GPU 上训练 10 M 维 PINN，速度提升数倍、内存仅 4856 MB，且训练稳定、收敛性优于现有基线；

**⚠️ 局限性**

仍为通用方法，缺乏针对特定算子进一步优化；零阶方法收敛速度可能慢于第一阶；未集成方差削减技术；对子空间更新频率 F 与秩 r 的超参数选择敏感，且未探索自适应步长方案。

---

## 288. Forensic Implications of Localized AI: Artifact Analysis of Ollama, LM Studio, and llama.cpp

**arXiv ID:** 2603.23996 | [PDF](https://arxiv.org/pdf/2603.23996v1)

**作者:** Shariq Murtuza `[一作]` `[通讯]`, Shariq Murtuza

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Windows和Linux系统中的本地大型语言模型（LLM）运行器Ollama、LM Studio和llama.cpp进行系统化、跨平台的数字取证分析，定位并文档化其磁盘、内存、网络等多种取证痕迹。

**💡 创新点**

首次提供了这些本地LLM工具的完整取证痕迹清单、标准化分析流程，并揭示了不同架构对证据持久性和可恢复性的影响，为后续法医实践奠定基础。

**🔧 技术方法**

使用磁盘镜像（dd/FTK Imager）、内存获取（LiME/WinPmem）、系统日志收集（Prefetch/Registry/PowerShell历史）、进程监控（ProcMon/strace）、网络抓包（Wireshark）以及自定义脚本解析GGUF、SQLite和JSON日志。

**📊 数据集**

采用受控实验生成的10条带唯一关键字的交互提示脚本，而非公开数据集。

**📈 对比分析**

通过实验验证：LM Studio完全恢复10条会话；Ollama CLI历史恢复100%提示；llama.cpp在Shell历史中恢复100%，但交互模式下无法恢复，强调内存取证的重要性；性能以证据完整率和恢复速度衡量。

**⚠️ 局限性**

未涵盖容器化部署、模型微调、RAG使用及第三方前端等场景，且对LLM运行时的隐蔽路径和加密存储缺乏深入分析。

---

## 289. RIS-Assisted D-MIMO for Energy-Efficient 6G Indoor Networks

**arXiv ID:** 2603.24180 | [PDF](https://arxiv.org/pdf/2603.24180v1)

**作者:** Akshay Vayal Parambath `[一作]` (Chalmers University of Technology), Tommy Svensson `[通讯]` (Chalmers University of Technology)

**通讯引用:** 10156 | [OpenAlex ID](https://openalex.org/A5086726797)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了RIS辅助分布式MIMO（D‑MIMO）在室内6G网络中的能效最大化，提出了结合MM的交替优化框架，联合优化发射功率与RIS相位。

**💡 创新点**

创新点在于：①同时考虑相干与非相干接收模式下的能效优化；②采用Majorization‑Minimization技术将非凸能效问题转化为可求解的凸子问题；③引入RIS控制器功耗模型，比较中心化与分布式控制架构；④在统一通道模型下评估RIS尺寸与AP密度对能效的影响。

**🔧 技术方法**

使用技术包括：交替优化（AO）+MM、RIS相位设计、功率分配、3GPP InH室内通道模型、Monte‑Carlo仿真、功耗模型（PA效率、RIS偏置/控制功率等）。

**📊 数据集**

数据集：基于3GPP InH室内模型生成的随机AP、RIS与UE部署；不使用真实测量数据，仅在仿真中统计性能。

**📈 对比分析**

与等功率分配（EPA）、随机RIS相位（RIS‑Rand）、不同AP密度、C/NC接收以及控制器架构进行对比。结果显示：RIS‑Opt显著提升SE与EE，C模式优于NC，但即使在NC模式下RIS‑Opt也比RIS‑Rand优；中心化控制得到最高EE，分布式控制需低功耗才能竞争。

**⚠️ 局限性**

局限性：仅限窄带单频模型；未考虑RIS的频率选择、带宽限制及多载波性能；同步与CSI获取的实际开销未建模；仿真仅覆盖室内场景，缺乏现场验证。

---

## 290. Linking Global Science Funding to Research Publications

**arXiv ID:** 2603.24147 | [PDF](https://arxiv.org/pdf/2603.24147v1)

**作者:** Jacob Aarup Dalsgaard `[一作]`, Jin AI `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了基于Web of Science（WoS）资助信息的全球科学资助者去重数据集，系统地将7.4百万条资助字符串匹配到OpenAlex和ROR的标准机构标识。

**💡 创新点**

创新点在于：①多阶段混合式去重管道，将词法归一、MinHash聚类、规则匹配、NER辅助和人工校验结合；②保留多匹配结果以保持透明；③提供针对论文级别的可选去重机制，使用作者国别和机构普及度解决模糊匹配。

**🔧 技术方法**

技术包括：字符串归一化、正则抽取缩写、MinHash局部敏感哈希、连通分支聚类、顺序规则匹配、Jaccard相似度回退、零射击NER（GLiNER）、人工验证与后向传播；以及作者地理匹配与频率优先级的论文级解歧。

**📊 数据集**

使用的数据集：Web of Science（核心ESCI 2023 snapshot）、OpenAlex（2025-09-29 snapshot）和Research Organization Registry（ROR v1.72）。

**📈 对比分析**

比较方法：直接跨数据库（WoS-OpenAlex、WoS-Dimensions）对论文级基金匹配进行交集和完整命中评估，手工抽样250篇论文做真值评估。性能表现：WoS召回率0.78、精确率0.96；OpenAlex召回0.68、精确0.94；Dimensions召回0.68、精确0.88；WoS在大多数度量上优于其他数据库。

**⚠️ 局限性**

局限性：①低频或多变的资助字符串匹配率低（>70%未匹配）；②依赖特定版本数据，更新后匹配结果可能变化；③规则匹配和NER在跨语言、同义词多变的场景下可能产生误匹配；④缺乏对全文资助信息的全面覆盖，仍受WoS原始提取质量限制。

---

## 291. Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation

**arXiv ID:** 2603.24134 | [PDF](https://arxiv.org/pdf/2603.24134v1)

**作者:** Haoyu Ji `[一作]` (Harbin Institute of Technology), Honghai Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 14755 | [OpenAlex ID](https://openalex.org/A5085867312)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了Spectral Scalpel框架，通过频域选择性滤波提升骨架动作分割的精度与边界清晰度。

**💡 创新点**

创新点包括：① 将频域滤波引入骨架动作分割；② 结合多尺度自适应频谱滤波器（MASF）与邻近动作差异损失（AADL）以增强动作间可区分性；③ 引入频域感知通道混合器（FACM）提升时序建模的频域信息利用。

**🔧 技术方法**

使用的技术有：FFT频域转换、可学习的多尺度滤波、双分支动态/静态通道加权、邻动作谱差异损失、频域通道混合、线性Transformer+Dilated TCN以及动作-文本对比学习。

**📊 数据集**

实验使用的公开数据集有：PKU-MMD v2（X-sub/X-view）、MCFS-130、LARa、TCG-15。

**📈 对比分析**

在上述五个数据集上与最新VTAS/STAS方法对比，Spectral Scalpel在Acc、Edit、F1@{10,25,50}等指标均实现了state‑of‑the‑art水平，同时FLOPs和参数量均低于或等同于同类模型。

**⚠️ 局限性**

存在的局限包括：边界仍偶尔模糊或错分，且对局部细节的适应性不足，未来工作需探索自适应局部滤波、多级滤波、频谱对比学习及频率先验等改进。

---

## 292. On Gossip Algorithms for Machine Learning with Pairwise Objectives

**arXiv ID:** 2603.24128 | [PDF](https://arxiv.org/pdf/2603.24128v1)

**作者:** Igor Colin `[一作]` (Télécom Paris), Joseph Salmon `[通讯]` (Université de Montpellier)

**通讯引用:** 2458 | [OpenAlex ID](https://openalex.org/A5033768552)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在分布式网络中使用Gossip算法对基于U‑统计量的成对目标函数进行估计和优化，提供了非渐近收敛性分析、偏差衰减机制及下界；

**💡 创新点**

首次给出GoSta估计算法的完整期望与方差上界，证明了对成对目标的Gossip双平均优化算法的收敛性，并明确了网络拓扑与梯度偏差的影响；

**🔧 技术方法**

采用Gossip传播、双平均（Dual Averaging）框架、谱图理论、混合时间与Ergodic分析等技术；

**📊 数据集**

在乳腺癌Wisconsin数据集上实验，使用逻辑回归对AUC的凸代理作为优化目标；

**📈 对比分析**

在完整图、二维网格、Watts–Strogatz三种拓扑下进行对比，结果表明连接度越高收敛越快，偏差项迅速趋零，实测性能与理论一致；

**⚠️ 局限性**

局限性包括假设每个节点仅存储单个样本、同步时钟、非二分图且需要较大网络连通性，且未覆盖隐私保护与鲁棒性等更现实约束。

---

## 293. Granular Ball Guided Stable Latent Domain Discovery for Domain-General Crowd Counting

**arXiv ID:** 2603.24106 | [PDF](https://arxiv.org/pdf/2603.24106v1)

**作者:** Fan Chen `[一作]`, Xinbo Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于粗粒球（Granular Ball）的稳定潜在域发现框架，并在单源领域泛化的拥挤计数任务中构建了语义代码库重编码与样式分支正则化的双分支学习方法。

**💡 创新点**

创新点包括：① 通过层次化的粗粒球划分与中心聚类，取代传统样本级聚类，显著提升潜在域标签的稳定性；② 语义代码库重编码将语义特征投射到可学习的离散基上，增强跨域语义一致性；③ 样式分支正则化（风格紧凑性、语义–风格正交约束）实现语义与风格的解耦，进一步提升泛化性能。

**🔧 技术方法**

采用的技术包括：粗粒球计算（Granular Ball Computing）与加权 2‑means、PCA 降维、K‑means 聚类、VGG16 编码‑解码骨干网络、语义代码库（Learnable Memory Bank）、风格分支正则化损失、正交正则化、以及 Adam 优化器。

**📊 数据集**

实验数据集：ShanghaiTech Part A、Part B；UCF_QNRF；NWPU‑Crowd。

**📈 对比分析**

在严格的单源无适配（no‑adaptation）协议下与多种基准方法（如 MCNN、DSSINet、DGCC 等）进行比较，平均绝对误差（MAE）与均方误差（MSE）均优于现有最强 DG 方案，尤其在大域间差异（如 SHB→QNRF、QNRF→SHA）上显著提升。

**⚠️ 局限性**

局限性：① 目前仅验证单源域泛化，需进一步扩展到多源或连续域；② 伪域数量 K 仍需经验调参；③ 对于极端拥挤或高分辨率图像，缺乏更细粒度的区域/密度特征；④ 主要基于 VGG16 编码器，尚未探索更强的 Transformer 或自监督基准。

---

## 294. LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation

**arXiv ID:** 2603.24097 | [PDF](https://arxiv.org/pdf/2603.24097v1)

**作者:** Haoyu Ji `[一作]` (Harbin Institute of Technology), Honghai Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 14755 | [OpenAlex ID](https://openalex.org/A5085867312)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种融合拉格朗日动力学的骨架时空动作分割网络LaDy。

**💡 创新点**

首次在骨架动作分割中引入拉格朗日动力学合成模块、能量一致性损失和时空调制机制，以提升动作辨别度和边界定位。

**🔧 技术方法**

采用拉格朗日动力学合成（LDS）、能量一致性损失（ECLoss）、时空调制（STM）以及图卷积、时间卷积和Transformer等深度学习技术。

**📊 数据集**

在PKU-MMD v2、MCFS-22、MCFS-130、LARa、TCG-15等六个公开基准数据集上进行评估。

**📈 对比分析**

与多种现有骨架动作分割方法进行对比，LaDy在Acc、Edit、F1等指标上均达到新的state‑of‑the‑art水平，尤其在PKU-MMD v2上提升约5.2% F1@50，同时保持13.67G FLOPs和1.83M参数。

**⚠️ 局限性**

仍受限于离散时间差分近似和仅考虑刚体动力学，未覆盖外部交互或非刚体运动，未来需更精细的动力学模型和更大规模数据验证。

---

## 295. LLMpedia: A Transparent Framework to Materialize an LLM's Encyclopedic Knowledge at Scale

**arXiv ID:** 2603.24080 | [PDF](https://arxiv.org/pdf/2603.24080v1)

**作者:** Muhammed Saeed `[一作]` (ScaDS.AI), Simon Razniewski `[通讯]` (ScaDS.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了LLMpedia框架，利用大型语言模型在参数记忆中生成、审核并发布完整的百科条目，实现从种子实体开始的广度优先遍历和三阶段实体去重管道。

**💡 创新点**

首次将事实性评估与知识材料化统一为可审计的全开放体系，揭示固定问答基准的可用性偏差，并通过捕获陷阱基准展示了与Grokipedia的差距。

**🔧 技术方法**

采用动态大纲生成、BFS扩展、三步实体清洗（标准化、LLM筛选、嵌入去重），自底向上自我校正，配合LLM判定者进行自动化事实性评估，并用TF-IDF、语义余弦等指标计算相似度。

**📊 数据集**

使用完整维基百科、133个质量筛选的网页源、前1000条最常编辑的英文维基百科文章，以及三类模型（gpt-5-mini、Llama-3.3-70B-Instruct、DeepSeek-V3）和多个种子实体（Vannevar Bush、古巴比伦、美国民权运动、东南亚荷兰殖民）。

**📈 对比分析**

对比Grokipedia的捕获陷阱基准，LLMpedia在维基百科和网页层面达到了89.8%/84%精确度与真值率，低于Grokipedia的7倍相似度；在跨模型实验中，GPT‑5‑mini覆盖率最高，准确率≈94%，但在随机样本中的真值率仅为74.7%，显著低于90%+的基准。

**⚠️ 局限性**

仅单轮生成且成本高昂；评估依赖单一LLM判定器，存在误判；数据和模型在时间点固定，网页证据覆盖有限；规模不均衡，深度分析仅限GPT‑5‑mini；只评估命题事实，未覆盖连贯性、客观性等属性。

---

## 296. Unlocking Few-Shot Capabilities in LVLMs via Prompt Conditioning and Head Selection

**arXiv ID:** 2603.24181 | [PDF](https://arxiv.org/pdf/2603.24181v1)

**作者:** Adhemar de Senneville `[一作]` (Université Paris-Saclay), Gabriele Facciolo `[通讯]` (Université Paris-Saclay)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5073617460)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出Head Ensemble Classifiers (HEC)，通过在视觉语言大模型(LVLM)中使用提示条件化和注意力头选择，提升少样本与零样本图像分类性能，填补 LVLM 与 CLIP 之间的差距。

**💡 创新点**

创新点在于利用提示条件化在推理时改善内部表示，并基于高斯判别分析对注意力头进行排名与集成，构建无需训练的高效分类器。

**🔧 技术方法**

核心技术包括提示（Task、Domain、Class）条件化、注意力头特征提取、Gaussian Discriminant Analysis (GDA) 头排名、以及平均化头集成。

**📊 数据集**

实验使用12个公开数据集（EuroSAT、UCF101、DTD、Caltech101、SUN397、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、CUB‑200、Traffic‑Signs），并在ImageNet上评估头的跨域泛化。

**📈 对比分析**

与 CLIP 基线（Probing、CLIP、TipAdapter、GDA、ProKeR）和 LVLM 基线（SAVs）对比，HEC‑V、HEC‑T、HEC‑VT 在多种设置下均达到或超过最优水平，尤其在 4‑shot 任务中平均提升至 82.4%（HEC‑V）并显著超越 CLIP 在零样本场景的表现。

**⚠️ 局限性**

局限性包括需要在模型内部访问注意力头作为中间表示（不适合 API 直接调用）、类级提示受限于类数、以及 LVLM 推理计算成本高于纯 CLIP 模型。

---

## 297. Towards Remote Attestation of Microarchitectural Attacks: The Case of Rowhammer

**arXiv ID:** 2603.24172 | [PDF](https://arxiv.org/pdf/2603.24172v1)

**作者:** Martin Herrmann `[一作]` (University of Duisburg-Essen), Lucas Davi `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 6190 | [OpenAlex ID](https://openalex.org/A5089242868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

提出并实现了基于 Rowhammer 攻击的远程验证协议，利用 ECC DRAM 的 Machine-Check Exceptions（MCE）与 DDR5 PRAC 的 Alert Back-Off（ABO）事件作为硬件级证据，结合 TPM 哈希链和加密实现可信测量。

**💡 创新点**

创新点在于将微架构级缺陷指示（MCE、ABO）直接纳入远程验证流程，并通过 TPM 固化测量链防止 kernel 级篡改；同时提供可扩展的检测阈值而非单纯的防御方案，首次实现了对 Rowhammer 的检测式远程验证。

**🔧 技术方法**

使用 ECC DRAM 的 MCE 捕获、DDR5 PRAC 模拟、Ramulator 2.0、mce-inject 注入、Python/openssl 脚本、TPM2.0 工具链以及 TLS/TCP 通信协议。

**📊 数据集**

使用自行生成的 20,000 条 Rowhammer 与正常访问的模拟数据集，10,000 条正例与 10,000 条负例，并在 Ramulator 里模拟 PRAC 与 MCE。

**📈 对比分析**

在 10,000 条正例与 10,000 条负例上检测准确率达 100%（无误报/漏报）。相较于传统仅验证软件状态的远程验证，新增硬件异常检测显著提升了对 Rowhammer 的识别；性能开销极小，主要由测量记录与 TPM 哈希扩展构成，未出现明显延迟。

**⚠️ 局限性**

局限性包括：检测阈值与启发式尚无统一标准；仅在模拟环境验证，缺乏真实硬件 PRAC 信号曝光；数据集为人工生成，缺少真实工作负载；对新型组合攻击（如 ECCploit+Rowpress）的覆盖仍需进一步评估。

---

## 298. Towards Automated Crowdsourced Testing via Personified-LLM

**arXiv ID:** 2603.24160 | [PDF](https://arxiv.org/pdf/2603.24160v1)

**作者:** Shengcheng Yu `[一作]` (Technical University of Munich), Chunyang Chen `[通讯]` (Technical University of Munich)

**通讯引用:** 4231 | [OpenAlex ID](https://openalex.org/A5075639297)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 Personified-LLM 框架，通过注入三维 persona（测试心态、探索策略、交互习惯）来实现自动化众包 GUI 测试。

**💡 创新点**

首次将 persona 概念与大型语言模型结合，能够可控、可复现地模拟人类测试者的多样行为，显著提升自动化测试的多样性与真实性。

**🔧 技术方法**

使用多模态 LLM（GPT‑4o 进行 GUI 理解与验证，GPT‑4o‑mini 进行决策）、计算机视觉+OCR 进行 GUI 识别、两阶段 ReAct 逻辑生成意图与操作、JSON 化 GUI 状态、视觉语义 bug 检测等技术。

**📊 数据集**

基于公开众包测试报告数据集（约 1,500 条真实轨迹，覆盖 23,000 条报告、1,100+ crowdworker），构建 9 个基于三维维度的 persona；同时选取 15 个多领域移动应用做任务评测。

**📈 对比分析**

通过与无 persona 基线的 5 次/9 次重复实验对比，评估 RQ1 内聚/跨聚、RQ2 事件有效率（总体提升 33–47%，输入事件提升 683%）、RQ3 错误触发（crash 29–38 vs 22，功能 bug 6 vs 3），显示 Personified‑LLM 在行为一致性、测试事件质量与缺陷发现方面显著优于基线。

**⚠️ 局限性**

局限性包括：persona 数量有限（仅 9 组），实验仅覆盖 15 种应用与预设任务，LLM 的不确定性与误差仍存在，且缺陷评估聚焦于 crash/功能 bug，未涉及更复杂业务逻辑或长期交互场景。

---

## 299. LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds

**arXiv ID:** 2603.24146 | [PDF](https://arxiv.org/pdf/2603.24146v1)

**作者:** Jaehun Bang `[一作]` (UNIST), Kyungdon Joo `[通讯]` (UNIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无关、快速且内存高效的开词汇3D场景理解框架

**💡 创新点**

创新点在于用2字节的语义索引直接注入3D高斯原语，消除逐步优化与高维特征存储，结合几何与语义感知单步聚类实现高效语义对齐

**🔧 技术方法**

利用SAM提取2D物体掩码、CLIP得到语义向量，并在3D Gaussian Splatting（3DGS）中实现索引特征注入与单步聚类；通过几何重叠和语义相似度构造图进行聚类

**📊 数据集**

在LERF‑OVS、DL3DV‑OVS（扩展版）以及ScanNet三大公开数据集上进行实验

**📈 对比分析**

与LangSplat、LEGaussians、OpenGaussian、LUDVIG、Dr.Splat等SOTA方法比较，特征蒸馏时间提升50–400×，内存使用降低64×，在mIoU、mAcc等指标上均达到或超过现有最高水平

**⚠️ 局限性**

仍依赖高质量的SAM/CLIP掩码，对极大规模或动态场景的适应性有限，且对极小或遮挡严重的对象可能仍出现分割误差

---

## 300. Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations

**arXiv ID:** 2603.24143 | [PDF](https://arxiv.org/pdf/2603.24143v1)

**作者:** Heng Wu `[一作]` (Chinese Academy of Sciences), Benzhuo Lu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 2809 | [OpenAlex ID](https://openalex.org/A5112452186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种线性–非线性融合神经算子（LNF‑NO），通过把线性分支与非线性分支按逐元素乘积方式融合，实现对 PDE 解决器的高效近似；

**💡 创新点**

创新点在于显式解耦线性与非线性效应并以乘法形式融合，提供一种轻量且可解释的激励机制，兼容多功能输入、正则格与非正则几何；

**🔧 技术方法**

技术主要包括多输入编码、线性+非线性分支的乘法融合、可选的轻量解码器、AdamW 优化以及与 FiLM、Transformer 等注意力思想的对比；

**📊 数据集**

使用多种标准 PDE 基准数据集：二维拉普拉斯、Burgers、Darcy、Poisson–Boltzmann（k=0.01/1/100）、Navier–Stokes、Poisson–Nernst–Planck、三维 Poisson–Boltzmann，以及不规则域 Poisson–Boltzmann；

**📈 对比分析**

与 DeepONet 与 FNO 进行统一实验对比，结果显示 LNF‑NO 训练时间普遍比 DeepONet 低 10‑30 倍，且与 FNO 在绝大多数任务上精度相当或更优；在极强非线性（k=100）和 3D PB 中表现尤为突出；

**⚠️ 局限性**

局限性包括在强全局传输（如 Navier–Stokes）的周期域中精度不及 FNO；对高分辨率 3D 计算的可扩展性仍需进一步验证；

---

## 301. Reservoir-Based Graph Convolutional Networks

**arXiv ID:** 2603.24131 | [PDF](https://arxiv.org/pdf/2603.24131v1)

**作者:** Mayssa Soussia `[一作]` (University of Sousse), Islem Rekik `[通讯]` (Imperial College London)

**通讯引用:** 3860 | [OpenAlex ID](https://openalex.org/A5048784346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种将储水计算与图卷积结合的RGC-Net模型，用于图分类与脑图演化预测

**💡 创新点**

核心创新在于将固定随机储水网络与图卷积相融合，并引入泄漏积分器缓解过平滑

**🔧 技术方法**

采用储水计算、图卷积、Leaky Integrator、Transformer框架以及批归一化和ReLU

**📊 数据集**

在MUTAG、PROTEINS、DD三种静态图数据集及EMCI-AD、SLIM160、Simulated三组脑连接图上进行实验

**📈 对比分析**

与GCN、GAT、GraphSAGE、GIN、GraphESN等基线对比，RGC-Net在分类任务上常超越传统GNN，在脑图生成任务上显著优于RBGM和EvoGraphNet，且训练时间与资源占用更低

**⚠️ 局限性**

模型不具备置换不变性，需调节迭代次数和泄漏率两个额外超参数，且对复杂图结构的适应性仍有限

---

## 302. Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study

**arXiv ID:** 2603.24125 | [PDF](https://arxiv.org/pdf/2603.24125v1)

**作者:** Nour Bouchouchi `[一作]` (Sorbonne Université), Marcin Detyniecki `[通讯]` (Sorbonne Université)

**通讯引用:** 1860 | [OpenAlex ID](https://openalex.org/A5065404842)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一框架同时评估LLM的内在与外在性别偏差，并研究对齐后对两者影响

**💡 创新点**

首次将相同的中性提示用于内在表征与生成输出的评估，揭示对齐主要是行为抑制而非知识删除，并在结构化与开放式任务中比较对齐效果

**🔧 技术方法**

使用线性方向假设与基于方向的消融、Spearman相关、LoRA微调以及LLM-as-judge标签器

**📊 数据集**

GenderAlign、WinoBias、CrowS‑Pairs、StereoSet、BBQ、MMLU、IFEval、RUTEd等公开数据集

**📈 对比分析**

在三大模型（Llama‑3.1‑8B、Mistral‑7B、gemma‑7b）上对齐前后、以及对抗提示下进行偏差、方向消融、相关性等多维度评估；结果显示对齐显著降低结构化任务中的输出偏差，但在故事生成等开放任务中仍显著偏差，且内在偏差在对齐后仍保持且可被激活

**⚠️ 局限性**

对齐并未消除内在性别知识，易被对抗提示激活；仅基于结构化基准的评估可能高估对齐效果；方法目前仅针对二元性别、英语，跨语言与多重属性偏差的扩展仍待研究

---

## 303. S4CMDR: a metadata repository for electronic health records

**arXiv ID:** 2603.24118 | [PDF](https://arxiv.org/pdf/2603.24118v1)

**作者:** Jiawei Zhao `[一作]` (University of Southern Denmark), Richard Röttger `[通讯]` (University of Southern Denmark)

**通讯引用:** 1613 | [OpenAlex ID](https://openalex.org/A5114028199)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了面向罕见病电子健康记录的元数据仓库S4CMDR，支持元数据的标准化、发现兼容性与可视化；

**💡 创新点**

采用中端标准化（middle‑out）对ISO 11179‑3模型做多对多扩展、与BioPortal自动匹配、微服务架构与用户友好 UI、兼容性评估与自动建议等创新；

**🔧 技术方法**

技术栈包括ISO 11179‑3、BioPortal API、Keycloak认证、Next.js/Express/SpringBoot、PostgreSQL、Docker 容器化及微服务；

**📊 数据集**

使用公开的罕见病 EHR 数据集（如 Charité、Loinc 等）及 BioPortal 公开本体数据；

**📈 对比分析**

通过案例演示验证功能，未进行量化性能对比；

**⚠️ 局限性**

依赖社区参与补充元数据、缺乏大规模性能评估、功能与其他 EHR 平台集成仍待完善。

---

## 304. Combi-CAM: A Novel Multi-Layer Approach for Explainable Image Geolocalization

**arXiv ID:** 2603.24117 | [PDF](https://arxiv.org/pdf/2603.24117v1)

**作者:** David Faget `[一作]` (ENS Paris-Saclay), Miguel Colom `[通讯]` (ENS Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于多层梯度加权激活图（Grad-CAM）融合的解释方法 Combi-CAM，用以可视化图像在地理定位模型中的关键区域。

**💡 创新点**

创新点在于：①不再只使用最深层卷积特征，而是将网络中所有阶段的最后卷积层激活图统一上采样并叠加；②利用 ReLU 保留对目标类别正向贡献的特征，融合了低层细节与高层语义信息，显著提升了定位解释的精准度。

**🔧 技术方法**

主要技术包括：卷积神经网络（EfficientNet‑B4）、梯度加权类激活映射、上采样（双线性插值）、ReLU 激活、跨层加权与聚合。

**📊 数据集**

使用公开的单图像地理定位数据集（如 Im2GPS、Google Landmarks 等）训练的地理定位网络进行实验，评估解释效果。

**📈 对比分析**

与 Grad‑CAM、Grad‑CAM++、Layer‑CAM、Score‑CAM 等现有方法对比，Combi‑CAM 在多张测试图像上能更精准地突出关键建筑或地标，实验结果以可视化热图方式展示，说明其在解释精度上优于传统方法；虽无定量指标，但定性分析显示误定位情况得到有效识别。

**⚠️ 局限性**

局限性包括：①对结构相似的图像仍可能产生误定位（如巴黎铁塔复制品）；②聚合多层特征导致计算成本略有上升；③目前仅在 EfficientNet‑B4 架构验证，需进一步验证对其他网络的通用性。

---

## 305. Towards Effective Experiential Learning: Dual Guidance for Utilization and Internalization

**arXiv ID:** 2603.24093 | [PDF](https://arxiv.org/pdf/2603.24093v1)

**作者:** Fei Bai `[一作]` (Renmin University of China), Hongteng Xu `[通讯]` (Renmin University of China)

**通讯引用:** 3694 | [OpenAlex ID](https://openalex.org/A5035141289)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dual Guidance Optimization (DGO) 框架，利用外部经验与内部参数的闭环迭代，实现大语言模型（LLM）在推理任务中的持续改进。

**💡 创新点**

创新点在于：① 将经验构建、轨迹-策略联合细化与经验内化整合为单一闭环流程；② 通过经验退火逐步降低对外部经验的依赖，促进内部参数的持续内化；③ 采用经验重写将包含外部线索的轨迹转化为可蒸馏的无依赖轨迹；④ 设计了多维度的经验更新与选择策略，形成正向反馈循环。

**🔧 技术方法**

技术手段包括：GRPO 强化学习、经验生成器与重写模板、经验退火与内部化比例控制、策略蒸馏、经验更新与质量筛选、测试时经验缩放（TTS）等；使用 Qwen3-4B/8B/14B 作为模型骨干，采用多轮训练与温度/Top‑p 采样策略。

**📊 数据集**

数据集：训练使用 SkyWork 9600 题目；经验生成器使用 DeepSeek v3.2 40000 题目；评估基准包括 AIME24、AIME25、HMMT25、MATH500（数学推理）以及 GPQA‑Diamond、HLE‑Verified（知识集成）等。

**📈 对比分析**

与 SFT、GRPO、DAPO、EGI 等基线对比；在 4B、8B、14B 三个规模上，DGO 在 intrinsic inference 取得 30.34%、32.41%、38.04% 的平均准确率，显著高于所有基线；使用测试时经验缩放（TTS）后进一步提升到 36.51%、39.38%、44.78%；在所有内外域基准上均达到或超过基线性能，显示了良好的跨域泛化与鲁棒性。

**⚠️ 局限性**

局限性：① 需要预构建并维护外部经验库，经验质量对性能影响较大；② 经验重写与蒸馏过程复杂，计算开销较高；③ 主要验证在数学推理领域，跨域应用的有效性尚待进一步探索；④ 迭代训练需多轮大规模计算，资源成本高；⑤ 对极端噪声经验的鲁棒性虽有提升，但仍有进一步改进空间。

---

## 306. Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks

**arXiv ID:** 2603.24113 | [PDF](https://arxiv.org/pdf/2603.24113v1)

**作者:** Jonathan Haag `[一作]` (University of Zürich and ETH Zürich), Matteo Saponati `[通讯]` (University of Zürich and ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在 DYNAP-SE 混合信号 neuromorphic 处理器上实现并验证基于 spike 的反馈控制在线学习规则，用 ITL 训练单层 SNN。

**💡 创新点**

首次将基于比例-积分 spike 控制的在线学习算法在混合信号硬件上实现，证明在设备噪声和量化限制下可实现有效学习。

**🔧 技术方法**

采用 DYNAP-SE、混合信号电路、正负控制神经元反馈、In-The-Loop 训练框架、Poisson 编码以及 6 位量化 synapse。

**📊 数据集**

使用二分类 spike 数据集和 Yin–Yang 三分类数据集。

**📈 对比分析**

与数值仿真和 ANN 反向传播基准对比，二分类 100% 准确率、目标误差 2.1 Hz；Yin–Yang 67% 准确率、误差 6.1 Hz，功耗 10–100 µW。

**⚠️ 局限性**

仍需外部计算机计算权重更新，训练速度受实时测量限制；未实现多层网络或隐藏层权重学习，缺乏完全硬件嵌入的学习规则。

---

## 307. KCLNet: Electrically Equivalence-Oriented Graph Representation Learning for Analog Circuits

**arXiv ID:** 2603.24101 | [PDF](https://arxiv.org/pdf/2603.24101v1)

**作者:** Peng Xu `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 8660 | [OpenAlex ID](https://openalex.org/A5051340429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于 Kirchhoff 电流定律的 KCLNet 框架，用异步图神经网络对模拟电路进行端到端的表示学习，并通过对比学习提升电路嵌入的物理一致性。

**💡 创新点**

创新点包括：① 将模拟电路转为有向无环图并利用电流方向驱动异步消息传递；② 设计 KCL 约束的对比损失（正样本为不同深度电流嵌入，负样本通过掩码产生硬负例），实现电流守恒在嵌入空间的保持；③ 结合频域特征提取与物理先验，形成完整的端到端学习管线。

**🔧 技术方法**

使用的技术包括异步图神经网络（AGNN）、FFT 频域特征提取、基于电流守恒的对比损失（KCL Loss）、节点掩码硬负样本生成、传统 GCN/GAT/GIN/SAGE 等基础 GNN 编码器作为对照。

**📊 数据集**

使用的主要数据集为自行生成的模拟电路数据集，包含分类数据集（含多尺寸参数）、子电路检测数据集（242,320 条样本）以及图编辑距离预测数据集（通过突变生成）。

**📈 对比分析**

与多种基线（未预训练的 GCN、GAT、GIN、SAGE、GATv2）以及 GraphCL 预训练模型进行对比。KCLNet 在模拟电路分类中 Acc@1 达 0.949（比 GraphCL_GCN 提升 39.8%），在子电路检测中 mAP 提升至 0.622（比 GIN 提升 43.4%），在图编辑距离预测中 MAE 降低至 1.6%。整体性能显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅考虑电流守恒，未利用 Kirchhoff 电压定律；依赖手工将电路图转为 DAG，可能对某些电路结构不适用；对极大规模电路的可扩展性与推理效率未作评估；未直接利用 SPICE 等仿真代码提升表达能力。

---

## 308. Unanticipated Adversarial Robustness of Semantic Communication

**arXiv ID:** 2603.24082 | [PDF](https://arxiv.org/pdf/2603.24082v1)

**作者:** Runxin Zhang `[一作]` (Tsinghua University), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 21450 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统性地研究了语义通信（基于DeepJSCC）与传统分离源信道编码（SSCC）系统在对抗攻击下的鲁棒性，提出理论下界与上界，设计针对性攻击方法，并在图像与CSI传输任务上进行实验验证。

**💡 创新点**

创新点包括：① 用Lipschitz连续性和噪声训练的隐式正则化证明语义系统具有天然对抗鲁棒性；② 提出结构感知易损集攻击（针对LDPC）与RL基高斯混合序贯攻击，填补经典系统对抗攻击方法空白；③ 开发进化梯度上升（PGA）语义攻击，针对回归任务最小化攻击能量；④ 通过对比实验验证语义通信对抗鲁棒性可比甚至超过传统系统14-16倍。

**🔧 技术方法**

技术手段包括深度联合源信道编码（DeepJSCC）、LDPC码、Belief Propagation、Lipschitz连续性理论、梯度上升（PGD）、C&W攻击、深度强化学习（DQN）等。

**📊 数据集**

实验数据集主要使用CIFAR‑10图像数据集和COST 2100生成的毫米波多天线CSI数据。

**📈 对比分析**

通过在相同带宽、功率和SNR条件下，采用最小攻击能量（ρ*）来衡量鲁棒性。实验显示语义系统的ρ*比传统SSCC系统高约14‑16倍，证明其更高的对抗鲁棒性。

**⚠️ 局限性**

局限性包括：① 理论上限与下界为保守估计，实际鲁棒性可能更好；② 只考虑白盒攻击，对抗鲁棒性在黑盒或更强攻击下尚未验证；③ 仅研究LDPC与DeepJSCC，对其他码/模型的通用性需进一步探索。

---

## 309. When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm

**arXiv ID:** 2603.24079 | [PDF](https://arxiv.org/pdf/2603.24079v1)

**作者:** Ye Leng `[一作]` (Cispa Helmholtz Center For Information Security), Yang Zhang `[通讯]` (Cispa Helmholtz Center For Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估并对比了多模态大语言模型（MLLM）与扩散模型在生成不安全图像与伪造图像检测方面的安全风险。

**💡 创新点**

首次从不安全生成与伪造检测两维度对MLLM与扩散模型进行量化对比，揭示MLLM因强大语义理解而导致更高安全风险。

**🔧 技术方法**

使用安全过滤器、恶意提示集、图像生成与评估框架以及多种伪造图像检测器。

**📊 数据集**

使用Lexica、4chan、Template、I2P、TemplateLong等不安全提示集，及MSCOCO、Flickr30k等自然语言与图像对照数据集。

**📈 对比分析**

通过对每个提示生成10张图像并用Moderation API计算不安全得分，及用四个检测器评估误判率；结果显示MLLM的unsafe得分显著高于扩散模型，其生成图像更难被检测。

**⚠️ 局限性**

实验仅覆盖七个代表性模型，未涵盖所有架构；不安全提示集与性别偏差分析存在样本与二元化局限。

---

## 310. Integrating Mental Health, Well-Being, and Sustainability into Software Engineering Education

**arXiv ID:** 2603.24191 | [PDF](https://arxiv.org/pdf/2603.24191v1)

**作者:** Isabella Graßl `[一作]` (Technical University of Darmstadt), Birgit Penzenstadler `[通讯]` (Chalmers University of Technology)

**通讯引用:** 4179 | [OpenAlex ID](https://openalex.org/A5081094387)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在软件工程课程中加入关注心理健康、福祉与可持续发展的项目和课堂干预，探讨其对学生专业视野和心理健康的影响。

**💡 创新点**

创新点在于将技术学习与人文关怀相结合，利用用户中心设计与情绪反思双向促进学生对软件工程社会责任的认知。

**🔧 技术方法**

采用的技术主要是课程设计干预手段，如团队契约、反思日志、案例讨论以及跨学科阅读材料。

**📊 数据集**

数据集来自德意志技术大学三门选修课的60份学生反思文本，涵盖本科与研究生、不同学科背景。

**📈 对比分析**

通过主题分析（reflexive thematic analysis）比较学生对课程前后观点变化，未提供定量性能指标，主要以质性主题与学生自述为评估依据。

**⚠️ 局限性**

局限性包括单一院校、样本量有限、数据为自我报告、可能存在自选偏差，缺乏对照组和系统量化测量。

---

## 311. TsetlinWiSARD: On-Chip Training of Weightless Neural Networks using Tsetlin Automata on FPGAs

**arXiv ID:** 2603.24186 | [PDF](https://arxiv.org/pdf/2603.24186v1)

**作者:** Shengyu Duan `[一作]` (Newcastle University), Alex Yakovlev `[通讯]` (Newcastle University)

**通讯引用:** 7262 | [OpenAlex ID](https://openalex.org/A5029446985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Tsetlin Automata的权值无神经网络（WNN）训练方法——TsetlinWiSARD，并实现了完整的FPGA上芯片训练架构。

**💡 创新点**

创新点在于：①将Tsetlin Automata嵌入WiSARD LUT中实现迭代、概率反馈式学习，解决了传统WiSARD单次记忆导致的过拟合；②设计了可复用的TA团队结构和位级LFSR随机反馈机制，实现在硬件上无处理器的训练；③通过这种方法在保持WNN低延迟、低功耗优势的同时，显著提升了训练精度。

**🔧 技术方法**

技术要点包括：Tsetlin Automata（TA）学习机制、位级随机反馈（LFSR）、LUTRAM存储TA状态、FPGA可编程逻辑映射、全并行的投票和Argmax推断。

**📊 数据集**

使用了六个公开的二值化数据集进行实验：MNIST、Fashion‑MNIST、Kuzushiji‑MNIST、EMG手势识别、HAR（人类活动识别）以及GPS（手势阶段分割）。

**📈 对比分析**

与传统WiSARD和B‑bleaching训练方法、以及Tsetlin Machine (TM) 和CNN加速器进行对比：TsetlinWiSARD在相同或更少的资源下取得更高准确率；在FPGA上实现时，训练速度提升约1000×，资源占用下降22%–64%，功耗下降64%且推断延迟显著降低。

**⚠️ 局限性**

局限性包括：①硬件实现受限于LFSR的伪随机性，导致训练过程中出现反馈偏差，影响最终准确率；②在资源受限的FPGA上只能实现较小模型（如TsetlinWiSARD‑150），无法充分验证更大模型的潜力；③与深度学习模型相比，WNN在复杂任务上的整体准确率仍低，需进一步改进模型结构或混合训练策略。

---

## 312. A visual observation on the geometry of UMAP projections of the difference vectors of antonym and synonym word pair embeddings

**arXiv ID:** 2603.24150 | [PDF](https://arxiv.org/pdf/2603.24150v1)

**作者:** Rami Luisto `[一作]` `[通讯]` (University of Jyvaskyla), Rami Luisto (University of Jyvaskyla)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在不同词向量模型中，将同义词、反义词、随机词对的差向量投影到二维空间时出现的“旋涡”几何结构，并以此为基础构建了一种基于UMAP的转导式分类器

**💡 创新点**

发现反义词与同义词在UMAP投影中表现出稳定的旋涡模式，且这种模式跨越多种模型（Word2Vec、GloVe、BERT、OpenAI text-embedding-3-small/large）存在；利用这一几何特征可以显著提升反义词/同义词区分的准确率

**🔧 技术方法**

使用的技术包括：词向量差向量构造、UMAP、t‑SNE、PCA、欧氏距离和余弦距离投影、聚类算法（KMeans、Spectral）以及转导式投票式分类；同时对比了基于原始向量的分类方法

**📊 数据集**

使用斯图加特数据集（Stuttgart dataset），包含大量名词、形容词、动词的同义词和反义词对，并生成了“打乱”对（shuffled antonyms / shuffled synonyms）作为对照集

**📈 对比分析**

在转导式分类中，先将所有差向量投影到二维平面，再用聚类划分并投票决定标签；该方法在所有模型上相较于未投影的原始向量分类均提升了10%–20%的准确率；与最新的诱导式基准（ICE‑NET、MoE‑ASD 等）相比，UMAP+转导式方法在大多数模型上接近甚至超越了 SOTA 的 F1 分数

**⚠️ 局限性**

局限性包括：1）该方法是转导式的，无法在不见到测试集时进行预测；2）缺乏理论解释为何出现旋涡结构，可能存在 p‑hacking 风险；3）对词向量的覆盖率和词频敏感，未对词性差异做进一步细化；4）在某些模型（如 BERT）下旋涡不如 OpenAI 大模型明显，需要更多实验验证

---

## 313. Efficient Controller Learning from Human Preferences and Numerical Data Via Multi-Modal Surrogate Models

**arXiv ID:** 2603.24138 | [PDF](https://arxiv.org/pdf/2603.24138v1)

**作者:** Lukas Theiner `[一作]` (Technical University of Darmstadt), Rolf Findeisen `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态多保真贝叶斯优化框架，用于结合数值评估与人类偏好数据来高效调优控制策略。

**💡 创新点**

创新点在于将偏好学习与多保真Gaussian过程融合，提供了ICM和AR1两种混合模型，实现不同模态数据的协同信息传递。

**🔧 技术方法**

使用高斯过程回归、偏好贝叶斯优化、多保真模型（ICM、AR1）、MCMC推断、期望最佳实用度等技术。

**📊 数据集**

数据集基于模拟驾驶者轨迹构建的低保真模型与单一驾驶者的高保真偏好模拟，评估在个性化自动驾驶中的参数调优。

**📈 对比分析**

与单纯偏好贝叶斯优化对比，实验显示多保真AR1结构在回报率和不良样本率方面均优于基线，提升了数据效率。

**⚠️ 局限性**

局限性包括对高保真偏好数据的依赖、模型超参数估计对小样本敏感以及仅在两保真设置下验证，未来需扩展至多源多模态场景。

---

## 314. MedAidDialog: A Multilingual Multi-Turn Medical Dialogue Dataset for Accessible Healthcare

**arXiv ID:** 2603.24132 | [PDF](https://arxiv.org/pdf/2603.24132v1)

**作者:** Shubham Kumar Nigam `[一作]` (University of Birmingham), Piyush Patel `[通讯]` (Madan Mohan Malaviya University of Technology)

**通讯引用:** 966 | [OpenAlex ID](https://openalex.org/A5039065444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多语言多轮医患对话数据集，并基于该数据集训练了轻量化对话模型

**💡 创新点**

创新地将真实对话与LLM生成的合成对话相结合，扩展为七种语言，并通过参数高效微调实现低资源部署

**🔧 技术方法**

采用LoRA参数高效微调，基于4‑bit量化的开源LLM（如LLaMA3.2‑3B、Mistral‑7B、Qwen3‑4B等），并在推理时使用双向翻译层支持多语言交互

**📊 数据集**

数据集来源于MDDial的真实对话，融合了Llama‑3.3‑70B生成的合成对话，并进一步翻译成英语、印地语、泰卢固语、泰米尔语、孟加拉语、马拉地语和阿拉伯语

**📈 对比分析**

在诊断准确率上最高达到90.21%（LLaMA3.2‑3B），专家评估安全通过率为95.3%，显示模型在多轮问诊和诊断方面表现优异

**⚠️ 局限性**

合成数据可能引入偏差、疾病种类受限、主要以英文训练导致翻译失真、仅文本交互缺乏多模态信息等局限

---

## 315. Equivariant Filter Transformations for Consistent and Efficient Visual--Inertial Navigation

**arXiv ID:** 2603.24130 | [PDF](https://arxiv.org/pdf/2603.24130v1)

**作者:** Chungeng Tian `[一作]` (Harbin Institute of Technology), Ning Hao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 4431 | [OpenAlex ID](https://openalex.org/A5100619560)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通用的 EqF 变换方法，用于在不同对称性下实现视觉‑惯导导航的估计一致性与高效实现

**💡 创新点**

通过构造全局‑局部映射，证明任意两种 EqF 之间存在非奇异线性变换，并利用该变换设计出状态无关可观子空间的 EqF，避免了经验性的对称性分析；同时提出了两种高效实现策略 Transforming Propagation 与 Transforming Correction

**🔧 技术方法**

利用 Lie 群理论、全局‑局部映射、Jacobian 变换、观测性分析，以及基于 SD‑EqF 或 ESKF 的稀疏化技术实现矩阵变换

**📊 数据集**

使用 OpenVINS 模拟轨迹、EuRoC MAV 数据集以及自制旋转飞行实验（Aerial Robot）进行评估

**📈 对比分析**

与 ESKF、SD‑EqF、Invariant SD‑EqF 等方法在多条轨迹上进行 100 次 Monte‑Carlo 仿真和真实数据实验，结果显示 T‑EqF 在保持一致性的同时，平均时间几乎与 ESKF 相当，且定位误差显著低于其他方法

**⚠️ 局限性**

目前的框架未考虑输出等变性，仍可能在高度非线性场景下出现线性化误差；另外实现复杂度相对较高，需进一步优化代码结构

---

## 316. The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation

**arXiv ID:** 2603.24124 | [PDF](https://arxiv.org/pdf/2603.24124v1)

**作者:** Mingyi Liu `[一作]` `[通讯]`, Mingyi Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究RLHF对语言模型回答多样性产生的“对齐税”，揭示在TruthfulQA等事实问答任务中，模型在多次采样下经常生成单一语义聚类，使采样式不确定性检测失效；

**💡 创新点**

创新点在于：①首次系统量化对齐税并证明其与DPO训练阶段、模型家族、规模和推理策略相关；②提出基于最低成本的多边界“UCBD”级联架构，利用 token 级熵、嵌入密度等互补无关信号，实现对齐税下的可行不确定性估计；

**🔧 技术方法**

采用的技术包括 token 级熵 (B1)、基于嵌入的聚类 (SINdex 方式)、单词 Jaccard、NLI 语义一致性检查、基于密度的 OOD 检测、以及多边界级联和内部回归路由器；

**📊 数据集**

实验使用的主要数据集为 TruthfulQA（790题）、FreshQA、HotpotQA、GSM8K（500题）以及 WebQuestions（200题），并在 Qwen3、LLaMA-3、Mistral、Tulu-3、Zephyr 等四个模型族（3B–14B、4‑bit 量化）上进行评估；

**📈 对比分析**

与传统采样基、NLI‑SE、SelfCheckGPT 等基线比较，B1 熵在 TruthfulQA 上 AUROC 约 0.60（相较于 SE 0.55）且成本为零；在 GSM8K 上单词熵与长度等特征结合可实现 50% 覆盖率下准确率从 84.4% 提升至 93.2%，级联架构在成本上比并行评估降低约 57%；

**⚠️ 局限性**

局限性包括：①对齐税的原因主要定位于 DPO，但不同训练配方的差异性仍未完全解析；②实验受 4‑bit 量化限制，未评估 FP16/大模型；③B5（NLI 交叉验证）在推理时不可用，需外部检索；④标签可靠性受 LLM‑judge 影响，需人工验证；⑤对齐税的发现仅在事实问答和数学推理任务中验证，其他领域（代码、对话）尚未评估。

---

## 317. Enhancing and Reporting Robustness Boundary of Neural Code Models for Intelligent Code Understanding

**arXiv ID:** 2603.24119 | [PDF](https://arxiv.org/pdf/2603.24119v1)

**作者:** Tingxu Han `[一作]` (Nanjing University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 85567 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种黑盒、训练免费、在推理阶段通过随机语义保持扰动对神经代码模型进行平滑，以提升对抗鲁棒性并给出可证的鲁棒性边界。

**💡 创新点**

创新点在于首次为神经代码模型提供可证明的鲁棒性边界，并且不需要对模型参数进行任何修改，仅通过对标识符进行语义保持的字符级扰动和投票聚合实现。

**🔧 技术方法**

主要技术包括：语义保持的字符编辑（插入、替换、删除）对标识符进行随机扰动、投票式预测聚合、Beta分布置信区间估计以及基于扰动集合的理论鲁棒性半径计算。

**📊 数据集**

使用了多种代码理解任务的数据集：Defect Detection（CodeNet、CodeChef）、Clone Detection（Kite、OJ）以及功能分类（Open Judge）等，并在 CodeBERT、GraphCodeBERT、CodeT5、CodeLlama、StarCoder 等多种神经代码模型上进行评测。

**📈 对比分析**

与现有防御方法 RoPGen、SPACE 等进行对比，实验显示在不显著下降准确率的前提下，ASR（攻击成功率）显著降低（如 defect detection 中从 42.43% 降至 9.74%），并在多种模型和任务上获得平均 1.63 的鲁棒性半径，证明性鲁棒性指标 NCRR 也提升至 29%-31%。

**⚠️ 局限性**

局限性包括：推理时需要生成大量扰动样本导致较高的计算开销；对扰动参数（N、η）的敏感性需要手动调优；理论鲁棒性半径的估计在不同模型和数据分布下可能存在一定偏差，且对极端大规模 LLM 的实测效果仍不充分。

---

## 318. Retinal Layer Segmentation in OCT Images With 2.5D Cross-slice Feature Fusion Module for Glaucoma Assessment

**arXiv ID:** 2603.24115 | [PDF](https://arxiv.org/pdf/2603.24115v1)

**作者:** Hyunwoo Kim `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**通讯引用:** 661 | [OpenAlex ID](https://openalex.org/A5034080987)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了基于2.5D CFF模块的OCT视网膜层分割框架，用于青光眼评估。

**💡 创新点**

创新点是将跨切片特征融合模块替换U-Net跳跃连接，显著提升跨切片一致性和噪声鲁棒性。

**🔧 技术方法**

使用深度残差U-Net骨干，CFF模块和多分支损失（像素交叉熵、列交叉熵和平滑L1）进行端到端训练。

**📊 数据集**

在自建的32例视网膜立方体临床数据和公开的DUKE DME数据集上进行验证。

**📈 对比分析**

相较于Graph、U-Net、ReLayNet和FCRN，平均MAD降至2.48像素、RMSE降至3.30像素，并在DUKE DME上保持最小方差，表明性能最优。

**⚠️ 局限性**

仅在黄斑立方体数据上训练，未验证在视盘扫描上的表现，且对不同设备/病变的泛化尚需进一步评估。

---

## 319. Toward a Multi-Layer ML-Based Security Framework for Industrial IoT

**arXiv ID:** 2603.24111 | [PDF](https://arxiv.org/pdf/2603.24111v1)

**作者:** Aymen Bouferroum `[一作]` (Inria Lille-Nord Europe), Abderrahim Benslimane `[通讯]` (University of Avignon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于机器学习的多层 IIoT 安全框架（TCA），实现了信任收敛加速与低成本真实部署；

**💡 创新点**

通过将网络 QoS 量化为 netC 并用随机森林预测收敛时间，动态调整信任转移概率，显著提升收敛速度，并在框架中首次提出可扩展的多层攻击检测与对抗学习鲁棒性方案；

**🔧 技术方法**

采用 Tm-IIoT 信任模型、H-IIoT 层级架构、Wi-Fi 6（IEEE 802.11ax）、MQTT/TLS、随机森林机器学习、Jetson Orin Nano GPU 推理以及 ESP32‑C6 微控制器；

**📊 数据集**

利用 MATLAB WLAN 与 Communications 工具箱生成的 802.11ax 真实信道仿真数据，构建包含 SNR、延迟、抖动、吞吐量等网络 QoS 以及信任状态的合成数据集；

**📈 对比分析**

在与原始 Tm‑IIoT 模型对比实验中，TCA 在恶劣网络条件下收敛时间缩短 28.6%，中等条件下 14.3%，并在 20%–50% 的恶意节点比例下保持 30.77% 的收敛加速；在 50–250 节点规模下，平均收敛时间比基线快 18%–33%；

**⚠️ 局限性**

仅基于仿真与合成数据，缺乏真实工业环境验证；当前方案主要针对 Wi‑Fi 6，可能对其他无线技术适配有限；并且未来需要进一步提升对抗学习攻击的鲁棒性。

---

## 320. Causality-Driven Disentangled Representation Learning in Multiplex Graphs

**arXiv ID:** 2603.24105 | [PDF](https://arxiv.org/pdf/2603.24105v1)

**作者:** Saba Nasiri `[一作]` (EPFL), Dorina Thanou `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 CaDeM 框架，使用因果推断驱动的自监督方法对多层图（多模态网络）的公共（共享）与私有（层特定）嵌入进行分离与对齐。

**💡 创新点**

创新点在于：①将因果学中的后门调整（backdoor adjustment）引入多层图表示学习，实现公共嵌入与私有嵌入的明确分离；②通过匹配损失对公共嵌入进行一致化，利用自监督层索引预测强化私有嵌入的层特异性；③三者协同的损失设计在理论上保证了分离性。

**🔧 技术方法**

技术方法包括：图卷积神经网络（GCN）编码器、Generalized Procrustes Analysis（GPA）匹配损失、层索引预测的交叉熵自监督损失、后门调整的分层匹配损失、节点与图级别的聚合与多头注意力融合。

**📊 数据集**

数据集涵盖四个合成数据（Syn1–Syn4）以及五个真实多层图数据集：ACM、DBLP、IMDB、Freebase、Human Connectome Project（HCP）脑功能网络，分别用于节点/图级任务。

**📈 对比分析**

与多种单视图与多视图基线（DeepWalk、Node2Vec、VGAE、DGI、MVGRL、GraphMAE、MNE、HDMI、MCGC、DMG、Graph2Vec 等）进行对比。实验表明 CaDeM 在节点分类、图分类、聚类等任务上均实现了最高的 Macro‑/Micro‑F1、ARI/NMI 指标，尤其在 HCP 任务分类、合成数据上的表现突出。

**⚠️ 局限性**

局限性包括：①假设各层节点严格对齐，无法直接处理部分重叠或未对齐的多层图；②理论证明基于理想化的全局最优与足够模型容量假设，实际训练可能受限；③自监督目标仅使用层索引，可能不足以捕捉层间细微差异；④需手动设计/调优数据增强、噪声和层数参数，适用性需进一步验证。

---

## 321. LGTM: Training-Free Light-Guided Text-to-Image Diffusion Model via Initial Noise Manipulation

**arXiv ID:** 2603.24086 | [PDF](https://arxiv.org/pdf/2603.24086v1)

**作者:** Ryugo Morita `[一作]` (RPTU Kaiserslautern-Landau & DFKI GmbH), Andreas Dengel `[通讯]` (RPTU Kaiserslautern-Landau & DFKI GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种训练无关的光照引导文本到图像扩散模型（LGTM），通过对初始噪声第1通道的调节实现对光照方向和强度的可控生成。

**💡 创新点**

首次在Latent Diffusion模型中发现并利用VAE通道1的光照解耦特性，实现无训练的光照控制，并可与ControlNet等结构条件无缝集成。

**🔧 技术方法**

使用初始噪声缩放、光照条件生成（线性梯度遮罩）、Latent Space Light Guidance、Stable Diffusion XL 以及 ControlNet 等技术。

**📊 数据集**

使用 Dog-Cat 数据集（2000张图像）并配合 BLIP 自动生成的字幕作为评估数据。

**📈 对比分析**

与 SDXL 及 ControlNet 基线进行对比，采用 FID、NIMA、CLIP-I/T 和光照准确率等指标；LGTM 在光照准确率上显著提升（≈77‑79% 对比 52%），但 FID 略升。

**⚠️ 局限性**

生成的对象往往会随着光照方向自动对齐，导致难以独立控制姿态，且在光照与几何约束冲突时可能出现偏差。

---

## 322. Knowledge-Guided Manipulation Using Multi-Task Reinforcement Learning

**arXiv ID:** 2603.24083 | [PDF](https://arxiv.org/pdf/2603.24083v1)

**作者:** Aditya Narendra `[一作]` (MIRAI), Aleksandr Panov `[通讯]` (MIRAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并验证 KG-M3PO 框架，将在线 3D 场景图与 M3PO 深度强化学习模型结合，用于多任务机器人操纵任务，解决部分可观测环境下的感知‑决策问题。

**💡 创新点**

1) 在线动态关系更新的 3D 场景图；2) 将图神经网络编码器直接融入 RL 损失，实现端到端训练；3) 多任务与多机器人跨任务共享知识图；4) 通过场景图提升样本效率和对遮挡的鲁棒性。

**🔧 技术方法**

3D 场景图构建（BBQ），图神经网络编码器，M3PO 模型基强化学习，端到端训练，Isaac Lab 物理仿真，图像+关节+语言+图表示融合，动态关系更新机制，奖励设计。

**📊 数据集**

通过 Isaac Lab 在 Franka 与 UR5 机器人上自建的仿真数据，包含 Pick-Cube、Pick-Place、Open-Cabinet 等任务，分为完全可观测（FO）和部分可观测（PO）两类；未使用公开数据集。

**📈 对比分析**

与 PPO、SAC、DreamerV3、TD-MPC2、IMPALA 等基线进行对比，使用归一化分数评估。KG‑M3PO 在 FO 任务提升样本效率并略高于基线；在 PO 任务显著提升成功率和收敛速度，取得更高 AUC；在多任务训练中总体领先。

**⚠️ 局限性**

场景图构建误差会影响性能，尤其在真实机器人上；对快速动态变化和频繁接触的情形鲁棒性不足；仅在仿真中验证，缺乏真实世界实验；在线图更新和 GNN 编码导致计算开销较高。

---

## 323. Walma: Learning to See Memory Corruption in WebAssembly

**arXiv ID:** 2603.24167 | [PDF](https://arxiv.org/pdf/2603.24167v1)

**作者:** Oussama Draissi `[一作]` (University of Duisburg-Essen), Lucas Davi `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 6190 | [OpenAlex ID](https://openalex.org/A5089242868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于线性内存的状态检测框架LMA，用CNN对Wasm模块的内存快照进行实时完整性验证。

**💡 创新点**

创新点在于将内存状态视为图像进行深度学习分类，实现对数据导向攻击和恶意宿主的持续检测，且与传统CFI无关。

**🔧 技术方法**

使用卷积神经网络（VGG‑16与自定义ResNet）、Wasm二进制插桩、覆盖引导fuzzing和AddressSanitizer做标签，RLE压缩快照。

**📊 数据集**

数据集来自六个具有已知CVE漏洞的真实Wasm应用（如flac、pal2rgb等），通过fuzzing生成的正常与受损内存状态。

**📈 对比分析**

与传统控制流证明/硬件方案比较，LMA在结构化工作负载上可达100%准确率，泛化模型准确率保持在85%以上，运行时开销仅为1.07~1.82倍。

**⚠️ 局限性**

局限包括对无结构文本处理等工作负载的检测效果低，需针对应用训练模型，且依赖TEE或浏览器信任，无法覆盖所有外部物理攻击。

---

## 324. A convergent Plug-and-Play Majorization-Minimization algorithm for Poisson inverse problems

**arXiv ID:** 2603.24156 | [PDF](https://arxiv.org/pdf/2603.24156v1)

**作者:** Thibaut Modrzyk `[一作]` (INSA-Lyon), Voichita Maxim `[通讯]` (INSA-Lyon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种可收敛的 Plug‑and‑Play 主优化（PnP‑MM）算法，将 EM 近似的 Kullback–Leibler 目标与预训练的高斯去噪器相结合，用以求解线性泊松逆问题。

**💡 创新点**

将传统的 Majorization–Minimization 与可收敛的 PnP 框架相融合，利用 KL 的 EM majorant 实现闭式更新，既保持收敛保证又能直接使用强大的深度去噪网络。

**🔧 技术方法**

采用 MM 框架、EM 近似、梯度步去噪（GS denoiser）、Shifted‑Poisson 近似，以及基于 PyTorch 与 DeepInverse 的实现。

**📊 数据集**

在自然图像（CBSD、DIV2K、Flickr2K 等）和医学图像（PET BrainWeb、低剂量CT AAPM‑Mayo 试验）上进行验证。

**📈 对比分析**

与 TV‑RL、DPIR、Bregman‑PnP、Diffusion 等方法对比，实验表明在泊松去卷积、PET 与 CT 重建中 PSNR、SSIM、NRMSE 等指标均达到或超过最先进水平，尤其在高噪声或稀视角条件下表现突出。

**⚠️ 局限性**

收敛速度相对较慢，需数百次迭代；需要大规模多样化训练数据以充分发挥去噪器能力；目前仅实现 2D 版本，缺乏 3D 扩展与自适应超参数策略。

---

## 325. Goal-Oriented Reactive Simulation for Closed-Loop Trajectory Prediction

**arXiv ID:** 2603.24155 | [PDF](https://arxiv.org/pdf/2603.24155v1)

**作者:** Harsh Yadav `[一作]` (University of Wuppertal), Tobias Meisen `[通讯]` (University of Wuppertal)

**通讯引用:** 3541 | [OpenAlex ID](https://openalex.org/A5032638290)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种闭环训练的自适应轨迹预测框架，使得预测模型在高频重新规划下能够自我纠正偏移，提升安全性。

**💡 创新点**

核心创新点包括：①结构隔离的闭环训练策略，使每条模式仅在其自身轨迹上学习回退；②基于目标的 transformer 场景解码器，实现逼真的交互式周围车辆仿真；③混合使用可反应与非反应代理的 hybrid 模拟，兼顾短期交互与长期稳定。

**🔧 技术方法**

技术手段包括：基于 QCNet 的查询中心注意力和 decoder-only 架构；全协方差 NLL 损失；delta-action 动态更新；PyTorch 原生 GPU 运动仿真；多模态 ego 与单模场景解码器；自监督闭环回放与 on‑policy 更新。

**📊 数据集**

使用 nuScenes（提供完整轨迹）与 DeepScenario（高密度交叉口）作为评估数据集，nuPlan 与 NAVSIM 的数据限制未被选用。

**📈 对比分析**

与传统开环模型对比，闭环模型在 0.5–1.0 s 重新规划频率下，nuScenes 的碰撞率下降 27.0%（相对），DeepScenario 上降幅达 79.5%；同时保持或提升最小 ADE，证明在高频场景下的鲁棒性与精度均得到提升。

**⚠️ 局限性**

局限性包括：①闭环训练对模拟的依赖，若场景生成失真可能导致模型偏差；②在极高频率下（T_sim < 0.5 s）混合模拟效果略减退；③模型仍假设目标信息已知，未考虑动态路径规划的全局决策；④大规模部署时需要高频仿真计算，可能影响实时性能。

---

## 326. Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model

**arXiv ID:** 2603.24144 | [PDF](https://arxiv.org/pdf/2603.24144v1)

**作者:** Kangxiang Xia `[一作]` (Alibaba), Lei Xie `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了SID‑Bench真实人类对话基准、平均惩罚时间（APT）评价指标，并设计了基于LLM的语义感知中断检测模型。

**💡 创新点**

①使用真实录音构建语义准确的中断标注；②提出整合准确率与延迟的APT统一指标；③将音频编码器与大型语言模型结合，利用语义理解实现精准中断判定。

**🔧 技术方法**

音频编码器（AuT）+ Qwen‑0.6b LLM，预训练对齐+微调，边界感知采样、延迟确认、决策平滑等训练与推理策略。

**📊 数据集**

3,700条真实中断对话（中文、英文），来源于专业录音与Switchboard，包含真实中断、backchannel、噪声与静默等多类样本。

**📈 对比分析**

与四种主流全双工基线（FSMN‑VAD、Freeze‑Omni、FireRedChat、Moshi）在SID‑Bench上评估FIR、IRL、APT；所提模型在APT上达到0.711 s，远优于最佳基线（≈2.13 s），FIR≈0.14，IRL≈0.44，显示出三倍以上的性能提升。

**⚠️ 局限性**

局限在于仅覆盖中文和英文两种语言，未覆盖多语言、多方对话；对极低语义噪声仍敏感；实验仅基于SID‑Bench，缺少与人类主观评估的对照。

---

## 327. Sequence-aware Large Language Models for Explainable Recommendation

**arXiv ID:** 2603.24136 | [PDF](https://arxiv.org/pdf/2603.24136v1)

**作者:** Gangyi Zhang `[一作]` (University of Science and Technology of China), Chongming Gao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5068892996)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大语言模型的序列感知可解释推荐框架SELLER，能够在保留用户交互序列信息的同时生成自然语言解释，并通过统一评估框架评估解释的实际推荐效益。

**💡 创新点**

创新点包括：① 双路径序列编码器（行为序列+语义序列）结合Mixture‑of‑Experts适配器，动态融合多模态信息；② 在LLM输入与注意力机制中直接注入序列嵌入；③ 通过解释增强推荐器（EER）将生成的解释作为语义信号，形成基于任务的“解释效用”评估，兼顾文本质量与推荐性能。

**🔧 技术方法**

使用了SASRec进行行为序列编码，Sentence‑T5编码文本语义，MoE适配器实现专家混合；LLM采用LLaMA2‑7B进行生成；超参微调采用LoRA/Adapter技术；在推荐侧使用SASRec+Hypernetwork对解释进行动态参数生成。

**📊 数据集**

在Yelp（25,633用户，89,994物品）和KuaiRec（6,584用户，8,180物品）两大公开数据集上进行实验，均采用10‑core过滤和时间序列划分。

**📈 对比分析**

与PETER、PEPLER、XRec等最先进可解释推荐方法对比，SELLER在BLEU、BERTScore上均取得最高分（Yelp +0.25，KuaiRec +0.17），在解释效用评估中Recall@10提升约18.8%（Yelp）/8.1%（KuaiRec），NDCG@10提升约27.5%/6.6%；整体表现显著优于基线，且训练与推理效率较XRec提升约30%。

**⚠️ 局限性**

局限性包括：① 训练所用的“真实”解释来自LLM生成，可能缺乏用户真实意图；② 评估完全离线，缺乏线上用户体验验证；③ 仅使用文本+序列信息，未考虑多模态或跨领域扩展。

---

## 328. Accelerated Spline-Based Time-Optimal Motion Planning with Continuous Safety Guarantees for Non-Differentially Flat Systems

**arXiv ID:** 2603.24133 | [PDF](https://arxiv.org/pdf/2603.24133v1)

**作者:** Dries Dirckx `[一作]` (KU Leuven), Wilm Decré `[通讯]` (KU Leuven)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5009275320)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

将分离超平面计算从轨迹优化问题中解耦，使用线性系统或二次规划求解超平面参数，然后将其作为固定参数嵌入到基于贝塞尔样条的时间最优运动规划 OCP 中。

**💡 创新点**

创新点在于将原本在 OCP 内作为决策变量的超平面参数转化为单独的分类问题，消除了双线性非凸约束，使 OCP 线性化、维度下降，从而显著降低计算复杂度并保持连续碰撞避免保证。

**🔧 技术方法**

采用贝塞尔样条参数化、Bernstein 基函数的凸包性质、线性系统/二次规划求解超平面、CasADi+IPOPT、LapackLU+ProxQP 求解器，以及两步过滤（filter）处理超平面更新。

**📊 数据集**

实验数据集为二维平面上由圆形机器人和最多10个矩形障碍物构成的场景，机器人运动模型为非齐次单车模型，测试了不同起止点组合，累计200条轨迹。

**📈 对比分析**

与现有的 Local Spline Relaxation (LSR) 方法对比，解耦方法在 2-8 个障碍物时总墙钟时间降低至约 29–59%，平均每次迭代时间显著缩短；轨迹时间误差略增（≤1%），成功率略低于 LSR，但仍保持在 87–100% 之间。

**⚠️ 局限性**

局限性主要是：当障碍物密度增大时，使用零阶超平面近似导致自由空间保守，导致成功率下降和轨迹时间的最优性差距扩大；缺乏超平面随轨迹变化的一阶灵敏度信息，容易陷入局部最优。

---

## 329. The First Generation of AI-Assisted Programming Learners: Gendered Patterns in Critical Thinking and AI Ethics of German Secondary School Students

**arXiv ID:** 2603.24197 | [PDF](https://arxiv.org/pdf/2603.24197v1)

**作者:** Isabella Graßl `[一作]` `[通讯]` (Technical University of Darmstadt), Isabella Graßl (Technical University of Darmstadt)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对84名德国中学生进行问卷调查，探讨他们在使用生成式人工智能辅助编程时的批判性思维与伦理认知。

**💡 创新点**

创新点在于首次系统研究了中学生在GenAI时代的批判性思维与伦理意识，并揭示了性别差异与文化背景对其行为的影响。

**🔧 技术方法**

研究采用问卷调查与主题分析相结合的方法，利用已验证的批判性思维与AI伦理量表，并辅以开放式问题的定性分析。

**📊 数据集**

使用的数据集为来自德国两场软件开发工作坊的84名16–19岁学生的问卷回应。

**📈 对比分析**

由于是探索性研究，没有与其他方法直接对比，但通过描述性统计和卡方检验展示了性别差异的显著性与效应量。

**⚠️ 局限性**

局限性包括样本量有限、仅覆盖德国学生、依赖自报行为、缺乏实际代码行为观察，且研究结果可能不具备跨文化普适性。

---

## 330. Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection

**arXiv ID:** 2603.24166 | [PDF](https://arxiv.org/pdf/2603.24166v1)

**作者:** Xu Zhang `[一作]` (University of Sydney), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 99801 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的低数据参考物体检测基准（De-ROD）以及一种轻量级、可插拔的框架（HeROD），通过将基于短语的空间与语义启发式先验注入到 DETR 风格的检测管线中，从而显著提升在少标注环境下的学习效率。

**💡 创新点**

创新点在于：①首次系统定义并量化数据效率参考物体检测任务；②引入可解释的空间与语义启发式先验，并在候选排序、预测融合与匈牙利匹配三阶段共同作用；③将先验通过加权、MLP 与损失修正等方式与训练目标结合，形成整体的“先推理后学习”策略。

**🔧 技术方法**

技术方法包括：基于词汇表提取空间词并生成位置热图、使用 CLIPSeg 计算视觉先验、在 proposal 生成阶段做加法融合、在最终预测阶段使用轻量级 MLP 学习融合权重、在 Hungarian 匈牙利匹配中减去先验得分以偏置匹配成本，并通过自定义 loss 让模型将置信度与先验一致。

**📊 数据集**

所用数据集为 RefCOCO、RefCOCO+ 与 RefCOCOg 三个公开参考物体检测数据集，并在低数据（0.1%–5%）与少样本（0.5k–2k）两种场景下进行评估。

**📈 对比分析**

与 Grounding DINO、UNINEXT 两个强基线在同一训练/推理流程下对比，HeROD 在低数据/少样本设置中平均提升 7–14%（RefCOCO）、5–10%（RefCOCO+）及 3–6%（RefCOCOg）顶点精度；在完整标注条件下仍保持竞争力，提升约 0.7–1.0 分。

**⚠️ 局限性**

局限性包括：①在完全标注或仅无绝对空间信息的数据集（如 RefCOCO+）提升幅度有限；②目前仅使用单一类别的空间/属性先验，缺乏更复杂的关系或上下文先验；③对模型的鲁棒性和可迁移性尚未在多种基准或更大规模模型上全面验证。

---

## 331. CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare

**arXiv ID:** 2603.24157 | [PDF](https://arxiv.org/pdf/2603.24157v1)

**作者:** Akash Ghosh `[一作]` (Indian Institute of Technology Patna), Salman Khan `[通讯]` (Mohamed bin Zayed University of AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了 CareFlow 这一长周期医疗软件交互基准，并基于多智能体的 Actor‑Critic 框架 CarePilot，实现了在 DICOM、医学影像注释、EMR 和 LIS 等真实医疗软件上的长步骤自动化；

**💡 创新点**

创新点在于首次构建专注医疗领域的长周期多模态基准，并引入工具感知、双重记忆（短期与长期）与分层反思的多智能体架构，显著提升了长步骤推理与界面适配能力；

**🔧 技术方法**

技术上采用 Qwen‑VL 作为多模态 LLM，配合四类轻量工具（UI 目标检测、缩放/裁剪、OCR、模板匹配）实现感知；通过双重记忆模块捕获历史与即时上下文，并以 Actor‑Critic 形式进行决策与评估，最后利用推理蒸馏将 Critic 反馈融入 Actor 以实现无监督推理；

**📊 数据集**

使用了 CareFlow 数据集，共 1,100 条任务，涵盖 Orthanc/Weasis（DICOM）、3D Slicer（影像注释）、OpenEMR（EMR）以及 Open-Hospital（LIS），并提供了 OpenHospital 的 OOD 数据进行外域评测；

**📈 对比分析**

在 Step‑Wise Accuracy (SWA) 与 Task Accuracy (TA) 两个指标下，CarePilot 在 CareFlow 上实现 36.4–38.2% TA 与 77.9–88.0% SWA，分别比最强基线高约 15.26% 与 3.38%，在 OOD 评测亦保持显著优势；

**⚠️ 局限性**

局限性包括仅覆盖五个医疗软件平台、预测的是语义动作而非精确坐标，缺乏多语言与更长任务的支持，且未涵盖所有真实临床软件的多样性。

---

## 332. Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection

**arXiv ID:** 2603.24139 | [PDF](https://arxiv.org/pdf/2603.24139v1)

**作者:** Zhanhe Lei `[一作]` (Wuhan University), Dengpan Ye `[通讯]` (Guangzhou University)

**通讯引用:** 1525 | [OpenAlex ID](https://openalex.org/A5050526815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种 Tutor-Student Reinforcement Learning (TSRL) 框架，用强化学习动态分配训练样本权重，以提升深度伪造检测模型的泛化能力。

**💡 创新点**

创新点在于将训练过程建模为马尔可夫决策过程，让教师代理学习基于样本历史学习状态的自适应加权策略，区别于传统静态或预排序的课程学习。

**🔧 技术方法**

采用 PPO 强化学习、EMA 损失、遗忘计数等状态特征，结合加权交叉熵损失和即时状态变化奖励。

**📊 数据集**

在 FaceForensics++ (FF++ c23) 训练，跨数据集评估包括 Celeb-DF-v2、DFD、DFDC、DFDCP，跨方法评估使用 DF40 数据集。

**📈 对比分析**

与六大基线模型（IID、CLIP、CORE、UCF、ProDet、Effort）以及九个 SOTA 模型对比，TSRL 在平均 AUC 上显著提升，跨数据集提升约 0.02-0.05，跨方法提升约 0.03-0.06，成为新的 SOTA。

**⚠️ 局限性**

局限在于 RL 教师需要额外训练时间和计算开销，对学生模型结构兼容性有限，未在所有 SOTA 模型上验证，且对极低资源环境不友好。

---

## 333. Likelihood hacking in probabilistic program synthesis

**arXiv ID:** 2603.24126 | [PDF](https://arxiv.org/pdf/2603.24126v1)

**作者:** Jacek Karwowski `[一作]`, Sam Staton `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用大型语言模型生成概率编程模型，并通过强化学习优化模型以适应员工经验与薪资的回归任务。

**💡 创新点**

将语言模型与概率编程评估器和语法检查器结合，采用奖励机制（测试集对数似然与语法惩罚）训练生成器，实现自动化的概率模型设计。

**🔧 技术方法**

大型语言模型（LLM）生成器、概率编程语言（PPL）与MCMC推理引擎、语法检查器Sentinel，以及强化学习（policy gradient）更新。

**📊 数据集**

包含员工工作年限与年薪的训练集与测试集（示例数据 5.5→32000、1.6→28000 等）。

**📈 对比分析**

与手工编写的基线模型（honest_model）对比，生成的模型在测试集上获得更高的对数似然，表现优于基线；但实验规模有限。

**⚠️ 局限性**

受限于数据量小、模型复杂度受限、生成代码可能出现语法错误或推理不收敛，难以推广到更复杂的真实世界任务。

---

## 334. FFV-PINN: A Fast Physics-Informed Neural Network with Simplified Finite Volume Discretization and Residual Correction

**arXiv ID:** 2603.24114 | [PDF](https://arxiv.org/pdf/2603.24114v1)

**作者:** Chang Wei `[一作]` (Tianjin University), Pao-Hsiung Chiu `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5065689104)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 FFV-PINN 框架，融合简化的有限体积方法（FVM）与残差校正损失，以在无需任何数据的情况下解决传统 PINN 的物理约束弱和训练收敛慢问题。

**💡 创新点**

创新点包括：① 用网络输出直接逼近控制面通量，简化传统 FVM 的离散步骤并提升数值稳定性；② 设计基于 SIMPLE 思路的残差校正损失，用来引导网络参数更新，显著提高收敛速度和训练稳定性；③ 在同一框架下统一处理二维/三维、稳态/非稳态以及耦合流热问题。

**🔧 技术方法**

技术方法：物理信息神经网络（PINN）+ 简化 FVM（直接在控制面计算通量）+ 残差校正损失 + 频率映射层 + SiLU 激活函数 + Adam 优化器 + JAX 库实现。

**📊 数据集**

数据集：无监督训练，仅利用边界条件和物理方程；在数值验证中使用多种基准 CFD 结果（二维/三维旋转托盘、后退阶、自然对流）作为对比数据。

**📈 对比分析**

比较方法：与现有 PINN 变体（SOAP、CANN、TSONN 等）在相同物理案例（Re=5000、Re=10000、Ra=1e8 等）下对比训练时间、MSE 和相对 L2 误差。性能表现：在 Re=5000 时训练时间仅 0.065 小时，比 SOAP 快 125 倍；首次在无数据条件下实现 Re=10000 的旋转托盘和 Ra=1e8 的自然对流解，误差均位于同类最佳水平。

**⚠️ 局限性**

局限性：在高 Reynolds（Re=10000）下低尺度小涡的细节捕捉不够完整；未实现多网格或高阶 FVM，可能限制细尺度精度；对极端高非线性问题的鲁棒性尚待进一步验证。

---

## 335. Bridging the Evaluation Gap: Standardized Benchmarks for Multi-Objective Search

**arXiv ID:** 2603.24084 | [PDF](https://arxiv.org/pdf/2603.24084v1)

**作者:** Hadar Peer `[一作]` (Technion Israel Institute of Technology), Oren Salzman `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的多目标最短路径（MOS）基准套件，涵盖道路网络、合成图、游戏网格和机器人运动规划四大类问题，并提供固定图实例、标准查询集以及参考的精确与 ε‑近似 Pareto 集。

**💡 创新点**

创新点在于首次将四类结构多样、目标相关性跨度广的实例统一在同一评估框架内，解决了过去基准碎片化和 DIMACS 数据集高度相关导致的评估失真问题，同时提供可复现的参考解集，形成完整、可比的评测基准。

**🔧 技术方法**

使用经典 MOS 算法 NAMOA*dr（A*pex 实现）计算参考解；通过 Pearson 相关系数评估目标间相关性；采用 ε‑dominance 评测近似性能；对每个查询和 ε 值统一执行评估，保证可复现性。

**📊 数据集**

数据集包括：DIMACS 路网（NY、FLA）及其多目标扩展、德国路网（10 维）、随机格子（2/3/4 维）、NetMaker 合成图、GUARDS 游戏网格（FireWalker、64room、maze512）、以及 Franka Panda 机器人 RRG（2/8 维）等四个领域的多目标实例。

**📈 对比分析**

评测方法：对每个基准实例的固定查询集分别在 ε ∈ {0,0.01,0.05,0.1} 下运行，获取精确与近似 Pareto 集；通过前沿大小、分布（spread）以及各目标间相关性等指标对算法进行对比；基准提供了完整的参考解集，算法性能可直接与之比较，示例显示不同领域前沿几何形状、规模差异显著。

**⚠️ 局限性**

局限性：仅覆盖经典的非负、可加、边局部成本的 MOS 问题；不包含负权、聚合、层次化或历史依赖的目标；所有图为静态结构，无法评估动态或时间相关场景；此外基准集中目标维度上限（最多 10 维）和实例规模仍有限，未来需要扩展到更高维度、更复杂的目标模型和动态环境。

---

## 336. VERIA: Verification-Centric Multimodal Instance Augmentation for Long-Tailed 3D Object Detection

**arXiv ID:** 2603.24294 | [PDF](https://arxiv.org/pdf/2603.24294v1)

**作者:** Jumin Lee `[一作]` (KAIST), Sung-Eui Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VERIA，一个基于图像先行的多模态数据增强框架，用于扩展稀有类别在 3D 感知中的样本多样性。

**💡 创新点**

创新点在于使用 VLM 生成子类别描述并进行语义与几何的多阶段验证，兼顾 RGB‑LiDAR 同步合成与场景一致性。

**🔧 技术方法**

主要技术包括文本到子类别描述的 VLM、基于扩散模型的图像 inpainting、单目深度估计生成伪 LiDAR，以及顺序语义与几何验证。

**📊 数据集**

使用 nuScenes 与 Lyft 两大自动驾驶数据集进行评估。

**📈 对比分析**

通过与 GT‑Aug、Text3DAug、PGT‑Aug 等现有增广方法对比，在 LiDAR‑only 和多模态检测任务上均显著提升稀有类别的 AP（如 construction vehicle、motorcycle、bicycle），且在多模态下性能提升更为突出。

**⚠️ 局限性**

局限包括生成模型可能产生幻觉，深度估计误差仍可能导致几何失真，且验证阈值需经验设定，未来需在极端情况和更大规模数据上验证。

---

## 337. The Specification Gap: Coordination Failure Under Partial Knowledge in Code Agents

**arXiv ID:** 2603.24284 | [PDF](https://arxiv.org/pdf/2603.24284v1)

**作者:** Camilo Chacón Sartori `[一作]` `[通讯]` (Catalan Institute of Nanoscience and Nanotechnology (ICN2)), Camilo Chacón Sartori (Catalan Institute of Nanoscience and Nanotechnology (ICN2))

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对51个Python类生成任务的系统实验，研究了多LLM代码代理在独立实现同一类方法时的协同问题，并验证了规格完整性对集成成功率的影响。

**💡 创新点**

创新点在于：①证明规格完整性是决定协同成功的关键因素；②揭示即使在完整规格下，拆分代理仍存在可加的协调代价；③发现基于AST的冲突检测只能作为诊断工具，无法提升恢复性能；④提出“规格优先”视角，即完整规格足以恢复单代理的性能。

**🔧 技术方法**

使用的技术包括：基于AST的冲突检测器、四层规格削减协议、2×2因子恢复实验、信息不对称拆分实验以及对Claude Sonnet/Haiku模型的调用。

**📊 数据集**

实验使用的主要数据集是从ClassEval抽取的51个类生成任务，构建了四层规格版本（全规格、简化规格、仅签名规格），形成了AmbigClass基准。

**📈 对比分析**

与单代理基准（单代理在完整规格下约88%通行率）比较，拆分代理在完整规格下恢复到≈89%（单代理水平），但在最弱规格下仅24.6%；完整规格与拆分代理的差距保持在25–39个百分点，说明规格完整性显著提升协同效果。

**⚠️ 局限性**

局限性包括：实验仅针对Python类、使用Claude系列模型且结构偏好通过系统提示注入；冲突检测仅捕获结构冲突，忽略语义冲突；缺乏跨模型、跨语言以及真实团队协作场景的验证。

---

## 338. Language-Assisted Image Clustering Guided by Discriminative Relational Signals and Adaptive Semantic Centers

**arXiv ID:** 2603.24275 | [PDF](https://arxiv.org/pdf/2603.24275v1)

**作者:** Jun Ma `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1774 | [OpenAlex ID](https://openalex.org/A5013880628)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉语言模型的语言辅助图像聚类框架，利用跨模态关系挖掘和可学习语义中心实现更好的聚类。

**💡 创新点**

创新点包括：①构造图像-文本表示矩阵挖掘跨模态关系，得到更具判别力的样本表示；②通过prompt学习在VLM语义空间中学习连续语义中心，突破手工构造文本空间的限制。

**🔧 技术方法**

采用CLIP预训练模型，跨模态关系矩阵学习（岭回归），prompt学习结合半监督训练（监督损失、一致性损失、熵正则化）。

**📊 数据集**

在ImageNet‑Dogs、ImageNet‑10、STL‑10、CIFAR‑10、Flowers‑102、CIFAR‑100、DTD、UCF‑101八个公开基准集上进行实验。

**📈 对比分析**

与18种深度聚类方法及零样本CLIP对比，平均提升2.6%（相对于TAC），在细粒度与高类数聚类上表现更显著，零样本CLIP平均低5.8%。

**⚠️ 局限性**

局限性包括对CLIP等预训练VLM的依赖、候选文本集合构造对结果的敏感性，以及固定prompt前缀可能限制跨语言或多模态的进一步适配。

---

## 339. DeepDTF: Dual-Branch Transformer Fusion for Multi-Omics Anticancer Drug Response Prediction

**arXiv ID:** 2603.24265 | [PDF](https://arxiv.org/pdf/2603.24265v1)

**作者:** Yuhan Zhao `[一作]` (North Carolina State University), Ning Sui `[通讯]` (North Carolina State University)

**通讯引用:** 4412 | [OpenAlex ID](https://openalex.org/A5082860820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DeepDTF双分支Transformer融合框架，用于同时预测癌症细胞系的多组学特征与化合物结构对应的药物敏感性与IC50值，且配备了基于SHAP+GSEA的可解释性模块。

**💡 创新点**

创新点包括：①将CNN+Transformer用于多组学特征的局部与全局建模；②使用GNN+Transformer捕获分子图的拓扑与全局依赖；③设计Fusion-Transformer实现跨模态细粒度注意力，缓解语义失配；④将SHAP向量与预排序GSEA相结合，提供方向感知的基因与通路解释。

**🔧 技术方法**

技术手段：双分支Transformer（CNN-Transformer + GNN-Transformer）、Fusion-Transformer、双任务损失（MSE+焦点损失）、SHAP解释、预排序GSEA。

**📊 数据集**

使用公开药效基因组学数据集：GDSC2（IC50 +基因组）、CCLP（多组学）、PubChem（化合物SMILES）。

**📈 对比分析**

在5折冷启动细胞系评估中，与DeepCDR、tCNN、CDRscan、DeepTTA等基线比较，DeepDTF在RMSE、R²、AUC等指标均取得最佳性能，最低RMSE 1.248、最高AUC 0.987，分类错误率下降约9.5%。

**⚠️ 局限性**

局限性：仅在细胞系水平进行冷启动，未验证在临床样本或患者组织中的泛化；高维组学数据仍受样本量限制；解释模块依赖SHAP与GSEA的统计稳定性，可能对低表达基因敏感；跨模态融合仍需进一步优化以处理更复杂的组学层次。

---

## 340. B-MoE: A Body-Part-Aware Mixture-of-Experts "All Parts Matter" Approach to Micro-Action Recognition

**arXiv ID:** 2603.24245 | [PDF](https://arxiv.org/pdf/2603.24245v1)

**作者:** Nishit Poddar `[一作]` (INRIA), Francois Bremond `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于身体部位的混合专家框架 B‑MoE，用于精细微动作识别。

**💡 创新点**

创新点：①将动作拆解为头部、躯干、上肢、下肢四个区域，使用专属专家捕捉局部细粒度运动；②引入宏-微运动编码器 M3E 兼顾长时上下文与短时细节；③采用交叉注意力路由动态选择最相关区域，并在双流（语义+运动）中融合。

**🔧 技术方法**

技术：Body‑Part‑Aware Mixture‑of‑Experts、Macro‑Micro Motion Encoder (M3E)、双流编码器（VideoMAE‑V2 + 预训练 ResNet‑SE‑TSM）、交叉注意力融合、线性残差拼接、Transformer‑MLP 分类器。

**📊 数据集**

数据集：MA‑52（22.4k视频），MPII‑GroupInteraction（MPII‑GI）和 SocialGesture（≈1k 视频/clip），均为社交情境下的微动作数据。

**📈 对比分析**

对比基线：MANet、GS‑MoE、MoNE 等；在 MA‑52 上 Top‑1 提升 3.64%（64.54%），F1‑macro 提升 4.32%；在 MPII‑GI 上 Top‑1 +2.57%，F1‑macro +3.35%；在 SocialGesture 上 F1‑macro +1.17%；总体计算成本低于 GS‑MoE，参数 59M，GFLOPs 567.77M。

**⚠️ 局限性**

局限：专家配置需人工预设，缺乏自动激活与数量学习；仅使用 RGB 视觉模态，未融入深度、骨架或音频等多模信息。

---

## 341. C-STEP: Continuous Space-Time Empowerment for Physics-informed Safe Reinforcement Learning of Mobile Agents

**arXiv ID:** 2603.24241 | [PDF](https://arxiv.org/pdf/2603.24241v1)

**作者:** Guihlerme Daubt `[一作]` (Bielefeld University), Adrian Redder `[通讯]` (Paderborn University)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5088574770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理信息的连续时空赋权（C-STEP）奖励，用于强化学习导航任务的安全控制。

**💡 创新点**

创新点在于将赋权重新定义为确定性系统可达集体积的对数，并将其作为内在奖励与任务奖励相乘，既保证安全又不牺牲效率。

**🔧 技术方法**

使用信息论赋权、连续动力学模拟与采样逼近、PPO算法（Stable Baselines3）以及PyTorch实现。

**📊 数据集**

在二维点迷宫和三维无人机PyBullet仿真环境中进行实验，随机生成障碍宽度作为测试数据集。

**📈 对比分析**

与仅使用普通导航奖励的基线相比，C-STEP显著降低碰撞率、靠近障碍的时间（最高可减少72%），成功率从82%提升至99%，通行时间仅略有增加。

**⚠️ 局限性**

主要限制包括采样误差、对时间窗T的敏感性、缺乏严格的安全证明、仅在仿真中验证且未在真实硬件上测试。

---

## 342. Identification of NMF by choosing maximum-volume basis vectors

**arXiv ID:** 2603.24227 | [PDF](https://arxiv.org/pdf/2603.24227v1)

**作者:** Qianqian Qi `[一作]` (Hangzhou Dianzi University), Peter G. M. van der Heijden `[通讯]` (Utrecht University)

**通讯引用:** 5891 | [OpenAlex ID](https://openalex.org/A5090384758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了最大体积约束非负矩阵分解（MAV‑NMF）框架，旨在使基向量尽可能彼此不同，从而提升对高度混合数据的可解释性；

**💡 创新点**

创新点在于①最大化基矩阵体积并证明在转置基矩阵满足足够散布（SSC）条件下可唯一识别；②通过等价变换将负logdet(M^TM)正则化转化为正logdet(HH^T)正则化，解决非凸优化难题；

**🔧 技术方法**

使用技术包括非负矩阵分解、logdet最大化、SSC理论、等价变换、交替投影快速梯度法（APFGM‑LOGDET）等；

**📊 数据集**

实验数据集包括两份人工砂粒分布数据、三份自制人工数据、CBCL人脸图像集和社会科学时间分配数据；

**📈 对比分析**

与传统最小体积约束NMF（MVC‑NMF）对比，MAV‑NMF在混合数据上恢复更接近真基、体积更大、基向量更稀疏、可解释性更好；评估指标为logdet(M^TM)、基向量相似度与稀疏度；

**⚠️ 局限性**

局限性包括：理论唯一性在有噪声情况下仍需验证；优化过程收敛速度受限于非凸性；超参数（λ、δ）需手动调优。

---

## 343. Environment-Grounded Multi-Agent Workflow for Autonomous Penetration Testing

**arXiv ID:** 2603.24221 | [PDF](https://arxiv.org/pdf/2603.24221v1)

**作者:** Michael Somma `[一作]` (JOANNEUM RESEARCH Forschungsgesellschaft mbH), Branka Stojanović `[通讯]` (JOANNEUM RESEARCH Forschungsgesellschaft mbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于LLM的多代理架构，用于在ROS/ROS2环境下自动化渗透测试

**💡 创新点**

通过环境感知的图形记忆动态维护共享记忆，结合LangGraph实现可追溯、可控的渗透流程

**🔧 技术方法**

大型语言模型（如LLaMA、GPT等）、LangGraph、Nmap、ROS专用脚本、GraphMemory

**📊 数据集**

在Docker化的机器人制造用例环境下进行的CTF挑战（ROS/ROS2网络）

**📈 对比分析**

与HackSynth基准对比，在五次实验中对CTF-0至CTF-3任务实现100%成功率，显著优于文献基准

**⚠️ 局限性**

指令生成过度复杂导致扫描时间延长；扫描不充分缺失ROS服务；对域特定任务的模型局限性

---

## 344. AMIF: Authorizable Medical Image Fusion Model with Built-in Authentication

**arXiv ID:** 2603.24296 | [PDF](https://arxiv.org/pdf/2603.24296v1)

**作者:** Jie Song `[一作]` (Macao Polytechnic University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21808 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种集成医学多模态图像融合与版权保护的模型 AMIF，能够在授权时去除水印、在未授权时输出带可见水印的融合结果。

**💡 创新点**

创新点在于：①将版权保护与融合目标统一建模；②设计内容感知水印记忆 (CCWM) 通过内部参数自生成水印；③引入通道-空间注意力可逆耦合 (C‑SAMIC) 在小波域实现水印嵌入与可逆去除，提升鲁棒性。

**🔧 技术方法**

使用 Restormer 编码器/解码器、Lite Transformer 与 INN 的私有编码器、双向交叉注意力、可逆耦合块、DWT 变换、二进制交叉熵+Dice 损失等深度学习技术。

**📊 数据集**

在哈佛医学影像网站公开的 MRI‑CT、MRI‑PET、MRI‑SPECT 三对多模态数据集上进行实验（共 983 训练对、104 验证对、81 测试对）。

**📈 对比分析**

与 IFCNN、U2FUSION、FUSIONMAMBA、PSLPT、SWINFUSION、MMIF‑INET 等 SOTA 方法比较，AMIF 在 SF、MI、VIF、Q^AB/F 及 SSIM 上均达到或超越最佳方法，且在授权去水印后仍保持高质量融合；在未授权水印去除攻击（PATCHWIPER、SLBR、WDNET）下表现出较强鲁棒性。

**⚠️ 局限性**

局限性包括：模型规模和推理时间相对较大；对极端攻击（如复杂遮挡或多尺度噪声）鲁棒性尚待进一步验证；实验仅覆盖三种模态组合，未评估更丰富的多模态情境。

---

## 345. Bridging Biological Hearing and Neuromorphic Computing: End-to-End Time-Domain Audio Signal Processing with Reservoir Computing

**arXiv ID:** 2603.24283 | [PDF](https://arxiv.org/pdf/2603.24283v1)

**作者:** Rinku Sebastian `[一作]` (University of York), Martin Trefzer `[通讯]` (University of York)

**通讯引用:** 703 | [OpenAlex ID](https://openalex.org/A5056241158)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个端到端的基于储备计算的时间域音频处理系统，直接从原始波形提取类似MFCC的特征并完成说话人与数字识别。

**💡 创新点**

创新点在于用储备计算替代传统FFT+Mel滤波器实现MFCC特征提取，显著降低时域计算复杂度，并实现单一储备网络同时完成特征提取与分类，减少资源消耗。

**🔧 技术方法**

采用储备计算（Echo State Network/Liquid State Machine）结合时间域卷积、最大池化等简化操作，并用ridge回归训练输出层。

**📊 数据集**

使用了TI-46（8名女性说数字0-9）和Audio-Mnist（60名说数字0-9）两个数据集进行实验。

**📈 对比分析**

通过与CNN、LSTM、AudioNet等深度模型以及其他RC模型对比，数字识别准确率达到约91–93%，仅需400–950个神经元，展示了低功耗、高效的性能。

**⚠️ 局限性**

存在的局限包括：需要进一步优化储备网络拓扑；当前时间域点数压缩仅采用最大池化，信息损失仍存在；在更大规模或噪声环境下的鲁棒性尚未验证。

---

## 346. InstanceRSR: Real-World Super-Resolution via Instance-Aware Representation Alignment

**arXiv ID:** 2603.24240 | [PDF](https://arxiv.org/pdf/2603.24240v1)

**作者:** Zixin Guo `[一作]` (Tongji University), Luyan Zhang `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 InstanceRSR 框架，实现基于实例感知与表征学习的真实世界图像超分辨率。

**💡 创新点**

创新点包括：① 结合语义分割与实例掩码在 Diffusion Transformer 上进行双重条件引导；② 引入实例级特征对齐（REPA）和尺度监督（IS）损失，显著提升实例辨别与细节恢复；③ 在控制网络（ControlNet）中注入实例特征投影，形成实例感知的扩散生成器。

**🔧 技术方法**

使用技术：Diffusion Transformer (DiT)、ControlNet、Segment Anything Model (SAM)、DINO/DINOv2 预训练特征投影、REPA 对齐损失、尺度监督 (IS) 损失、DDIM 加速推理。

**📊 数据集**

数据集：Segment Anything 1B 图像用于预训练；真实世界 SR 基准包括 DrealSR、RealSR、RealLR200、RealLQ250 用于评测。

**📈 对比分析**

与 Real-ESRGAN、ResShift、StableSR、SeeSR、DiffBIR、OSEDiff、DiT4SR 等 SOTA 方法在 LPIPS、MUSIQ、MANIQA、ClipIQA、LIQE 等指标上对比，InstanceRSR 在所有指标上均实现最高/最低（即 LPIPS 最低、其他指标最高），显示显著性能提升。

**⚠️ 局限性**

局限性：① 仍需高算力和大规模 GPU 训练；② 对极端噪声、压缩或不同降解模式的鲁棒性尚未充分验证；③ 复杂多实例场景下仍可能出现细节细微失真；④ 缺乏用户主观评价与真实应用场景验证。

---

## 347. S$^{3}$G: Stock State Space Graph for Enhanced Stock Trend Prediction

**arXiv ID:** 2603.24236 | [PDF](https://arxiv.org/pdf/2603.24236v1)

**作者:** Yao Lu `[一作]` (University Of Twente), Luyan Zhang `[通讯]` (Independent Researcher)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 Stock State Space Graph (SSSG) 的框架，用于股票趋势预测，首先使用小波去噪网络提取净化的时间序列特征，然后在每个时间步动态构造关系图，利用状态空间模型学习图的演化并通过图神经网络聚合信息，最终输出下一日的预期收益。

**💡 创新点**

创新点在于：①将状态空间模型应用于时变股票关系图，显式捕捉即时与滞后交互；②结合小波去噪降低低信噪比影响；③端到端联合学习去噪、图演化与收益预测，避免传统静态或单步相似度方法的局限。

**🔧 技术方法**

采用的技术包括：一维卷积+可学习阈值小波去噪、Gaussian kernel 图构建、状态空间模型（状态转移与观测方程）、图神经网络聚合、MSE+排名损失联合优化。

**📊 数据集**

使用的数据集为中国 A 股市场 CSI‑500，训练集 2015‑2020，验证集 2021‑2022，测试集 2023‑2024，日级别行情信息。

**📈 对比分析**

与经典策略、基础深度学习模型及当前 SOTA 方法（PatchTST、Crossformer、MambaStock 等）进行对比，使用 IC、ICIR、RankIC、ARR、ASR 等指标评估。SSSG 在大多数指标上均优于所有对比模型，尤其在信息系数、年化收益率和夏普比率方面表现突出，尽管在波动率与最大回撤上略逊于最安全的基线。

**⚠️ 局限性**

局限性包括：在波动率和最大回撤上未能达到最保守基线的水平；对计算资源需求较高；模型对异常事件的鲁棒性和可解释性仍有待进一步提升。

---

## 348. Stance Labels Fail When They Matter Most: The Projection Problem in Stance Detection

**arXiv ID:** 2603.24231 | [PDF](https://arxiv.org/pdf/2603.24231v1)

**作者:** Bowen Zhang `[一作]` (Shenzhen Technology University), Bowen Zhang `[通讯]` (Shenzhen Technology University)

**通讯引用:** 3082 | [OpenAlex ID](https://openalex.org/A5100385138)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨立场检测中把多维态度压缩成单一标签所导致的投影问题，并通过实验验证其在不同文本上的表现差异

**💡 创新点**

提出投影问题（projection problem）概念，证明当文本维度一致时标签标注误差小，维度冲突时标签一致性下降；并提出基于维度一致性划分的评估策略

**🔧 技术方法**

使用人工标注（三分类标签与四维情感维度Likert评分）与LLM（GPT‑4o）预测的差异作为文本难易度划分，利用Krippendorff’s α评估标注一致性

**📊 数据集**

SemEval‑2016 Task 6 “Climate Change Is Real Concern”数据集的60条推文（30条维度一致，30条维度冲突）

**📈 对比分析**

未对模型进行对比，而是通过对标签与维度一致性α的对比展示投影问题的存在：在易样本上标签α=0.307高于维度平均α=0.082；在难样本上标签α=0.085低于维度平均α=0.334；维度中Policy的α最高（0.572）

**⚠️ 局限性**

样本规模小（60条、3位标注者），维度划分由LLM预测误差间接推断，部分维度定义过于主观导致一致性低；研究仅聚焦单一目标，未验证在更大、更多目标上的普适性

---

## 349. Where Do Your Citations Come From? Citation-Constellation: A Free, Open-Source, No-Code, and Auditable Tool for Citation Network Decomposition with Complementary BARON and HEROCON Scores

**arXiv ID:** 2603.24216 | [PDF](https://arxiv.org/pdf/2603.24216v1)

**作者:** Mahbub Ul Alam `[一作]` `[通讯]` (Uppsala University), Mahbub Ul Alam (Uppsala University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了Citation‑Constellation工具，能够对研究者的引文网络进行分层拆解，并给出BARON和HEROCON两种结构性分数；

**💡 创新点**

创新点在于将自引、合著网络、机构隶属、期刊治理等多层网络关系与引文进行可审计的逐条分类，并将这些分类转化为可解释的二元与分级分数；

**🔧 技术方法**

技术包括基于Python的API调用（OpenAlex、ORCID、ROR）、深度图遍历与时间衰减的共著网络构建、局部LLM（Qwen 3.5 8B）用于期刊治理信息抽取，以及全流程的JSON审计日志与无代码Web界面；

**📊 数据集**

使用公开数据集OpenAlex（260M+作品、100M+作者、2.8B+引用），ORCID公共API、ROR机构数据库以及自建的期刊治理数据库；

**📈 对比分析**

与现有可视化和计量工具（VOSviewer、CiteSpace、Biblioshiny、Scite.ai、Scopus等）进行对比，指出其在单一作者层面拆解方面的独特性；在实验中，处理约80篇论文、1500条引用的分析在不到90秒内完成（Phases 1‑3），整体交互体验在几分钟内完成；

**⚠️ 局限性**

局限性包括：对OpenAlex的语言与时间覆盖偏差、未知（UNKNOWN）分类导致样本偏差、缺乏经验校准的HEROCON权重、对非英文或老旧文献的识别不足、期刊治理数据库覆盖不完整，以及未进行严格的经验验证和对游戏行为的预测风险。

---

## 350. HEART-PFL: Stable Personalized Federated Learning under Heterogeneity with Hierarchical Directional Alignment and Adversarial Knowledge Transfer

**arXiv ID:** 2603.24209 | [PDF](https://arxiv.org/pdf/2603.24209v1)

**作者:** Minjun Kim `[一作]` (Promedius Inc.), Minje Kim `[通讯]` (Promedius Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HEART-PFL 框架，通过客户端的层级方向对齐（HDA）和服务器端的对抗知识迁移（AKT）实现个性化联邦学习的双侧优化，提升模型个性化性能与全局稳定性。

**💡 创新点**

创新点：1) HDA 在早期层使用余弦相似度进行方向对齐，深层使用 MSE 进行语义匹配，充分利用网络层级语义结构并保持客户端特异性；2) AKT 在服务器端引入对称 KL 蒸馏并结合对抗样本，使蒸馏过程双向且鲁棒，缓解单向蒸馏导致的全局模型漂移。

**🔧 技术方法**

使用轻量级适配器（BatchNorm–Dropout–Conv2D 结构）与冻结的预训练 backbone；实现层级原型提取、层级对齐损失（cosine + MSE）、对称 KL 损失、对抗扰动（PGD）以及代理数据的双向蒸馏。

**📊 数据集**

实验数据集：CIFAR-100、Flowers-102、Caltech-101，采用 Dirichlet 分布（α=0.1/0.3/0.5）模拟不同程度的非IID 客户端分布。

**📈 对比分析**

与 FedAvg-per、FedProto-per、FedPer、LG-FedAvg、Ditto、FedBABU、FedALA、PerAda 等基线进行对比；在所有 α 设定下，HEART-PFL 取得最高个性化准确率（CIFAR-100 63.42%、Flowers-102 84.23%、Caltech-101 95.67%），并且仅使用 1.46M 可训练参数，显著优于传统方法。

**⚠️ 局限性**

局限性：对预训练 backbone 依赖较强，适配器规模虽小但在极端资源受限设备上仍可能不够轻量；对抗样本生成和双向蒸馏增加计算开销；对跨域代理数据的泛化性能受限于代理数据的多样性和质量。

---

## 351. RefReward-SR: LR-Conditioned Reward Modeling for Preference-Aligned Super-Resolution

**arXiv ID:** 2603.24198 | [PDF](https://arxiv.org/pdf/2603.24198v1)

**作者:** Yushuai Song `[一作]` (Institute of Automation, Chinese Academy of Sciences), Dong-ming Yan `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RefReward‑SR，一种以低分辨率图像为参照的奖励模型，用来把超分辨率（SR）结果与人类偏好对齐，并通过RL对SR模型进行优化。

**💡 创新点**

创新点包括：①首创基于LR条件的SR偏好数据集RefSR‑18K；②利用多模态大语言模型并通过Group Relative Policy Optimization（GRPO）进行细调，生成可解释且具语义一致性评分；③将该奖励模型嵌入SR生成流程，实现真正意义上的Preference‑aligned SR。

**🔧 技术方法**

使用技术包括多模态LLM（Qwen3‑VL 8B）、GRPO、局部裁剪评分策略、强化学习对SR模型的Fine‑Tuning（DP²O‑SR）、以及LPIPS、DEQA、CLIPIQA等多项视觉质量评估。

**📊 数据集**

主要数据集为RefSR‑18K（18,000组LR‑HR偏好对），以及用于训练与测试的LSDIR、DIV2K、RealSR等标准超分辨率数据集。

**📈 对比分析**

与传统FR/NR指标及零射门LLM对比，RefReward‑SR在与人类标注的一致度达到约85%，Recall@1与Filter@1均最高；在SR模型优化中，该方法在PSNR/SSIM/LPIPS等指标上与DP²O‑FLUX相当，并在用户研究中获得72.7%胜率，显著优于基线。

**⚠️ 局限性**

局限性包括：需要大规模人工标注的LR‑HR偏好数据；奖励模型对细粒度细节的区分仍有限；RL优化可能出现奖励剥削（reward hacking）；以及在极端分辨率或未知降解场景下的泛化能力待进一步验证。

---

## 352. A Large-Scale Study of Telegram Bots

**arXiv ID:** 2603.24302 | [PDF](https://arxiv.org/pdf/2603.24302v1)

**作者:** Taro Tsuchiya `[一作]` (Carnegie Mellon University), Alejandro Cuevas `[通讯]` (Princeton University)

**通讯引用:** 337 | [OpenAlex ID](https://openalex.org/A5065305058)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了史上最大规模的Telegram机器人数据集，并通过自动化交互系统获取机器人功能，对32,000+机器人进行分类和社区使用分析，揭示其合法与非法用途。

**💡 创新点**

创新点在于首次系统化地对Telegram机器人进行大规模特征抽取、功能归类与恶意检测，并将机器人视为平台关键软件基础设施提出干预思路。

**🔧 技术方法**

使用了雪球采样、Telegram Bot API自动交互、LLM（GPT‑4o）与关键词匹配进行功能与域标签，语言检测、网络拓扑分析和度量统计来评估机器人行为。

**📊 数据集**

主要数据集包括：从Pushshift与TGDataset扩展的106,000+频道、809M条消息；32,071条机器人（含描述、命令列表、交互日志）；9,041,103条频道/用户链接；并利用Telegram公共API进行数据抓取。

**📈 对比分析**

相较于此前的公开数据集，本文提供了更大规模（492M新消息）和首个机器人专用数据；LLM分类与人工对比的Kappa约0.73，恶意机器人检测率为4%诈骗、5%地下；机器人持续时间与重用率等指标展示了不同域的使用特征。

**⚠️ 局限性**

局限性包括仅覆盖公开频道、未收集私有频道导致地下域估计不足；机器人交互仅限两条基础命令，无法获取多轮交互细节；未收集媒体文件，无法完全评估信息泄露与恶意内容风险。

---

## 353. Samasāmayik: A Parallel Dataset for Hindi-Sanskrit Machine Translation

**arXiv ID:** 2603.24307 | [PDF](https://arxiv.org/pdf/2603.24307v1)

**作者:** N J Karthika `[一作]` (Indian Institute of Technology), Anil Kumar Gourishetty `[通讯]` (Indian Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并整理了 92,196 句现代印地语-梵语平行文本，形成名为 Samasāmayik 的大规模并行语料库，并对其进行机器翻译基准测试。

**💡 创新点**

创新点在于该语料库聚焦当代印地语与梵语，来源多样（儿童杂志、广播节目、教学视频等），填补了传统梵语古典文本占主导的空白，并且经过专家对齐，确保语料质量。

**🔧 技术方法**

采用了字节级文本生成模型 ByT5、NLLB-1.3B 多语言翻译模型以及 IndicTrans‑v2 专为印度语言设计的 Transformer 模型进行微调。

**📊 数据集**

使用了自建的 Samasāmayik 语料库以及从 BPCC（Bharat Parallel Corpus Collection）抽取的 79,977 句印地语-梵语平行数据做基线。

**📈 对比分析**

通过在 Samasāmayik 与 BPCC 训练集上分别微调模型，并在三组测试集（自建 in‑domain、IN22、Flores‑200）上评估，发现使用 Samasāmayik 训练的模型在同域测试中 BLEU/chrF++ 显著提升（如 NLLB 由 5.90 提升至 15.83），在外域测试中保持相近或略低的性能，说明数据集对提升当代文本翻译效果有效。

**⚠️ 局限性**

限制在于仅涵盖印地语-梵语这一单一语言对，并且来源局限于现代领域，缺乏对古典梵语或其他未覆盖领域的适用性验证。

---

## 354. Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs: When Does Per-Edge Routing Help?

**arXiv ID:** 2603.24291 | [PDF](https://arxiv.org/pdf/2603.24291v1)

**作者:** Eyal Weiss `[一作]` `[通讯]` (Technion — Israel Institute of Technology), Eyal Weiss (Technion — Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于边距敏感路由的图神经网络层CSNA，用于在异质图上实现按边路由的消息传递。

**💡 创新点**

引入了学习到的距离度量来对每条边进行软路由，将同类边和异类边分别映射到共振和不共振通道，并通过节点门控融合。

**🔧 技术方法**

采用双通道聚合、基于投影距离的可学习成本函数、门控组合、校准正则化以及在情境化随机块模型上的理论分析。

**📊 数据集**

在六个常用异质图基准（Texas、Wisconsin、Cornell、Actor、Chameleon、Squirrel）上进行评估。

**📈 对比分析**

与MLP、GCN、GAT、GraphSAGE、H2GCN、GPRGNN、ACM-GNN等基线在相同调参网格下对比，CSNA在对抗性异质性数据集上与最优方法相当或略优，在信息性异质性数据集上表现逊于GCN。

**⚠️ 局限性**

在高同质性或信息性异质性图上无优势，计算成本高达GCN的3–10倍，且缺乏在信息性异质性 regime 下有效利用跨类边的机制。

---

## 355. Attack Assessment and Augmented Identity Recognition for Human Skeleton Data

**arXiv ID:** 2603.24232 | [PDF](https://arxiv.org/pdf/2603.24232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 356. Software Supply Chain Smells: Lightweight Analysis for Secure Dependency Management

**arXiv ID:** 2603.24282 | [PDF](https://arxiv.org/pdf/2603.24282v1)

**作者:** Larissa Schmid `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 6772 | [OpenAlex ID](https://openalex.org/A5027206285)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了软件供应链味道（Supply Chain Smell）概念，并设计实现了名为dirty-waters的工具，用于在Maven和NPM生态中检测并报告这些味道；

**💡 创新点**

创新点在于将代码味道的抽象延伸到供应链层面，构建了系统化的味道分类体系，并通过实践验证其可行性与价值；

**🔧 技术方法**

技术实现结合三类信息源（依赖清单、包注册中心、源代码仓库）进行静态分析，并通过GitHub Actions实现CI集成；

**📊 数据集**

使用了两大生态的“最受依赖”前50个项目（共计约1,891个Maven包和8,071个NPM包）的依赖树作为数据集；

**📈 对比分析**

通过与开发者访谈评估味道的严重度，并对两生态中味道的出现率进行定量对比，发现Maven中可追溯性和签名问题较为普遍，NPM中则相对稀少；

**⚠️ 局限性**

局限性包括仅分析GitHub托管项目、仅支持部分味道（如Maven不支持Deprecated、Aliased、No Provenance），且样本仅限最受依赖包，无法覆盖整个生态的全部情况；

---

## 357. ScrollScape: Unlocking 32K Image Generation With Video Diffusion Priors

**arXiv ID:** 2603.24270 | [PDF](https://arxiv.org/pdf/2603.24270v1)

**作者:** Haodong Yu `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62653 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将超高分辨率极长宽比图像生成改为连续视频扫描任务，利用视频扩散模型实现32K分辨率生成；

**💡 创新点**

核心创新在于扫描位置编码（ScanPE）实现移动相机坐标映射，以及滚动超分辨率（ScrollSR）在潜在空间实现高效细节放大；

**🔧 技术方法**

使用预训练视频扩散模型（Wan2.1-T2V-1.3B）、扫描位置编码、滚动超分辨率、流匹配对齐、轨迹锚定分区（TAP）等技术；

**📊 数据集**

在约3000张高质量多比例图像（含2000张6:1以上自然风景、1000张6:1传统中国山水画）上微调；

**📈 对比分析**

与DyPE、FLUX等图像扩散基线以及多图切块方法进行定性、定量评估，使用FID、KID、CLIP、GSD等指标，ScrollScape在极长宽比（8:1）下显著提升结构连贯性与细节质量，用户研究中获得最高评价；

**⚠️ 局限性**

方法仍受限于预训练视频模型的时间序列长度和计算成本，对极端宽高比以外的多方向扫描支持有限，且对训练数据量和多样性依赖较高。

---

## 358. Embracing Heteroscedasticity for Probabilistic Time Series Forecasting

**arXiv ID:** 2603.24254 | [PDF](https://arxiv.org/pdf/2603.24254v1)

**作者:** Yijun Wang `[一作]` (Southeast University), Xiu-Shen Wei `[通讯]` (Southeast University)

**通讯引用:** 7167 | [OpenAlex ID](https://openalex.org/A5066964304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为LSG-VAE的非自回归概率时序预测框架，通过位置尺度高斯似然直接建模预测均值与时间变异方差，实现对异方差不确定性的显式建模。

**💡 创新点**

创新点包括：①用位置尺度似然替代传统MSE，引入自适应衰减机制；②双头解码器同时输出位置（均值）和尺度（方差）参数；③采用Patch编码、RevIN及非自回归latent动态模块，显著提升训练稳定性与推理速度；④该位置尺度范式可轻松迁移至GAN、K^2VAE等其它生成框架。

**🔧 技术方法**

技术细节：变分自编码器框架；Patch‑based variational encoder；Reversible Instance Normalization；非自回归latent dynamics（一次性映射到未来）；位置尺度高斯解码器（Softplus方差）；使用Gaussian NLL与KL损失；评估指标包括CRPS、NMAE、QICE。

**📊 数据集**

数据集：9个ProbTS基准（ETTh1/2, ETTm1/2, Electricity, Traffic, Weather, Exchange, ILI）用于长时序预测；8个用于diffusion协议的基准（ETT(m1/2/h1/2), Electricity, Traffic, Exchange, ILI）；以及人工合成数据用于验证异方差建模。

**📈 对比分析**

与15+强基线（包括PatchTST、iTransformer、Koopa、TimeGAN、TimeGrad、CSDI、DiffusionTS、TMDM、NsDiff、K^2VAE等）在CRPS和NMAE上进行对比，LSG‑VAE在所有数据集和预测窗口均实现最优或近优性能，尤其在长时序预测（H=720）和概率校准（QICE）上显著优于扩散模型；同时在推理速度与内存占用上也大幅领先。

**⚠️ 局限性**

局限性：①目前仅在非自回归VAE结构下验证，需进一步评估在更复杂或更大规模多变量场景的适用性；②对极端高步长预测或极端波动场景的鲁棒性仍需深入研究；③模型对超参数（如KL权重、Patch长度等）敏感，调参成本较高；④尚未考虑多源外部因子或多元协同效应的融合。

---

## 359. Who Benefits from RAG? The Role of Exposure, Utility and Attribution Bias

**arXiv ID:** 2603.24218 | [PDF](https://arxiv.org/pdf/2603.24218v1)

**作者:** Mahdi Dehghan `[一作]` (University of Glasgow), Graham McDonald `[通讯]` (University of Glasgow)

**通讯引用:** 338 | [OpenAlex ID](https://openalex.org/A5063436649)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了检索增强生成（RAG）系统中的查询组公平性问题，评估了均衡准确性提升（EAI）与平均准确性（EA）的实现情况，并分析了组曝光、组效用和组归因对公平性的影响。

**💡 创新点**

首次将查询组公平性引入RAG系统研究，提出通过组曝光、效用和归因三维度评估和改进公平性，并展示它们在不同公平类别中的关键作用。

**🔧 技术方法**

采用检索增强生成框架，结合BM25、SPLADE、Contriever三种检索器和LLaMA‑3.1‑8B、Gemma‑2‑9B两种大语言模型；使用ROUGE‑L做准确率评估、ROberta‑large‑mnli进行答案归因，并用Spearman相关性分析组特征与准确率的关系。

**📊 数据集**

基于TREC 2022 Fair Ranking Track构建的三大主题数据集（城市、地理、军事历史），每个数据集包含与四个公平性类别（主题年代、受欢迎度、文章年代、字母顺序）对应的组标签。

**📈 对比分析**

通过将RAG与单一LLM（LLM‑only）模式在文章生成与标题生成任务中进行对比，利用ROUGE‑L和查询组准确率的差异衡量公平性；实验表明RAG虽然总体提升准确率，但在多组间显著放大准确性不均，尤其标题生成任务更易出现偏差。

**⚠️ 局限性**

研究仅涵盖了有限的公平类别、检索器与模型组合，未探讨更广泛的检索策略或多模态数据，且归因方法依赖NLI模型，可能导致归因误差和对公平性评估的影响。

---

## 360. A Deep Dive into Scaling RL for Code Generation with Synthetic Data and Curricula

**arXiv ID:** 2603.24202 | [PDF](https://arxiv.org/pdf/2603.24202v1)

**作者:** Cansu Sancaktar `[一作]` (University of Tübingen), Taco Cohen `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多轮生成的合成数据管线，结合教师-学生自对弈与课程学习，动态调整问题难度并生成步进式任务链，用于强化学习优化代码生成模型。

**💡 创新点**

创新点在于通过多轮迭代自适应调整任务难度，生成层级化的“踏步”任务；并将环境多样性作为独立扩展轴，提升了模型的泛化与收敛速度。

**🔧 技术方法**

采用教师-学生自我对弈生成、Group Relative Policy Optimization (GRPO)、多轮无梯度适配、随机抽样种子、分布式 RL 与多种课程学习策略等技术。

**📊 数据集**

实验使用 CodeContests（LCB 及其 splits）、MATH、OpenAI Math 作为评测基准，并以随机 GitHub 代码片段与已解题目作为合成数据的种子。

**📈 对比分析**

在 Llama3.1-8B Instruct、Qwen3-8B 与 Qwen2.5-32B 上，比较了真实数据、合成数据、单轮与多轮生成、不同难度与课程、单环境与多环境设置对 in-domain 与 out-of-domain 的 pass@k 性能，结果显示合成数据与多环境多轮生成能显著提升收敛速度和最终性能。

**⚠️ 局限性**

局限性包括：合成任务难度分布需人工筛选；步进式生成对训练效率提升有限；逆向课程在极难任务上稳定性差；实验范围受限，缺乏跨语言或更广泛领域的验证。

---

## 361. RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation

**arXiv ID:** 2603.24295 | [PDF](https://arxiv.org/pdf/2603.24295v1)

**作者:** Kai Zhu `[一作]` (Peking University), Jiahuan Zhou `[通讯]` (Peking University)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5055004003)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Refining Specifics State Space Model（RS-SSM），通过补偿状态空间压缩过程中遗忘的时空细节来实现视频语义分割

**💡 创新点**

在SSM中引入Channel-wise Amplitude Perceptron（CwAP）提取通道特定信息分布，并设计Forgetting Gate Information Refiner（FGIR）自适应反转并细化遗忘门，使模型专注于补偿被遗忘的细节

**🔧 技术方法**

利用线性状态空间模型（SSM）、频域幅值感知（CwAP）、遗忘门反转/细化（FGIR）、跨通道信息对齐与多通道双路径SSM融合

**📊 数据集**

在四大视频语义分割基准集上进行评估：VSPW、NYUv2、CamVid、Cityscapes

**📈 对比分析**

与现有SSM、Transformer及CNN方法对比，RS-SSM在所有数据集上实现mIoU领先1.5–2.6分，同时保持与TV3S相近或更优的GFLOPs/FPS效率

**⚠️ 局限性**

仍受限于固定尺寸状态空间对极细粒度特征的压缩，以及对非常长序列中累计误差的鲁棒性待进一步提升

---

## 362. Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timestep

**arXiv ID:** 2603.24260 | [PDF](https://arxiv.org/pdf/2603.24260v1)

**作者:** Tianyi Liu `[一作]` (Nanyang Technological University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5925 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的 HetCache 框架，利用 Diffusion Transformer 的时间步和令牌异质性，在视频编辑推理时实现显著加速。

**💡 创新点**

创新点在于：①在时间步层面同时采用全计算、部分计算和复用三种模式；②在全计算步根据空间先验将令牌划分为 context、margin、generative，并仅缓存语义代表性 context 令牌；③结合 K‑Means 聚类与缓存的 context‑to‑generative attention 重要性评估，实现令牌级别的精细化加速。

**🔧 技术方法**

技术细节包括：时间步差分估计与阈值阈定、EMA 缓存更新、K‑Means 令牌聚类、注意力重要性计算、分层缓存与自注意力减少；采用 Diffusion Transformer（DiT）作为基底。

**📊 数据集**

使用 VACE‑Benchmark（视频修补）和 VPBench（文本指导视频编辑）进行实验，同时在 DAVIS、720P 高分辨率、长时长视频以及 LTX‑Video‑VACE 等数据集上进行验证。

**📈 对比分析**

与 Wan‑2.1‑VACE 基线、TeaCache、Timestep Reduction 等方法比较。结果显示：在 VACE‑Benchmark 上 2.67× 的速度提升、FLOPs 下降至 23.6 PFLOPs，PSNR/SSIM/VFID 与基线相当；在 VPBench 上 1.9× 加速，质量保持；在更高分辨率、长视频和 outpainting 场景下亦保持性能优势。

**⚠️ 局限性**

局限性包括：需手动设定 Δ、K、r_ctx 等超参数；对极端遮挡或长时间视频仍可能出现质量下降；依赖空间先验 mask，无法适用于无明确 mask 的编辑任务；未在更大规模模型或其他 Transformer 架构上验证。

---

## 363. Memory-Augmented Vision-Language Agents for Persistent and Semantically Consistent Object Captioning

**arXiv ID:** 2603.24257 | [PDF](https://arxiv.org/pdf/2603.24257v1)

**作者:** Tommaso Galliena `[一作]` (Italian Institute of Technology), Lorenzo Natale `[通讯]` (Italian Institute of Technology)

**通讯引用:** 8085 | [OpenAlex ID](https://openalex.org/A5009034971)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了统一的记忆增强视觉-语言-动作模型，用于在多视角下持续生成语义一致的对象描述。

**💡 创新点**

创新点在于将对象级记忆、数据关联、句子生成与导航决策融合到单一自回归框架，并通过token化记忆实现长时序一致性。

**🔧 技术方法**

采用预训练视觉语言模型（如Qwen3-VL-2B），自监督伪标注，实例分割、3D点云融合，以及LoRA微调等技术。

**📊 数据集**

使用Habitat中的HM3D和Gibson场景做训练与测试，并构建手工标注的对象级评估集。

**📈 对比分析**

与IC3、ECO、LD-CPS等伪标注方法以及BLIP‑2、InternVL等VLM基线对比，取得+32.86 SPICE、+7.39 CS等显著提升。

**⚠️ 局限性**

局限在于依赖外部实例分割，假设理想传感器，缺乏真实世界鲁棒性和动态场景测试。

---

## 364. Semantic Centroids and Hierarchical Density-Based Clustering for Cross-Document Software Coreference Resolution

**arXiv ID:** 2603.24246 | [PDF](https://arxiv.org/pdf/2603.24246v1)

**作者:** Julia Matela `[一作]` (Wismar University of Applied Sciences), Frank Krüger `[通讯]` (Wismar University of Applied Sciences)

**通讯引用:** 1196 | [OpenAlex ID](https://openalex.org/A5030968176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套混合管道，用于跨文档软件实体共指消解，包括语义表征、知识库中心化检索、表面形式规范化和基于密度的聚类。

**💡 创新点**

创新点在于将语义嵌入与FAISS检索结合，并采用表面形式加权的上下文字符串、层次化分块聚类及可扩展的阻止策略，显著提升大规模数据的处理效率。

**🔧 技术方法**

使用了MiniLM压缩的Sentence‑BERT进行句子嵌入，FAISS做近似最近邻检索，HDBSCAN做密度聚类，结合L2归一化、加权平均池化与阈值匹配。

**📊 数据集**

基于SOMD 2026共享任务数据，包含子任务1的gold提及（2974条，733簇），子任务2的自动提取提及（2860条，699簇）以及子任务3的219950条提及。

**📈 对比分析**

在MUC、B‑CUBED、CEAF_E与CoNLL指标上与基准对比，系统在子任务1/2/3的CoNLL F1分别达到0.9809、0.9765、0.9618，整体性能保持在0.95以上。

**⚠️ 局限性**

局限包括子任务3中由于检索空间扩大导致的性能下降，HDBSCAN聚类的非单调耗时，阈值设定固定，缺乏在线更新与引用图上下文等信息。

---

## 365. UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking

**arXiv ID:** 2603.24226 | [PDF](https://arxiv.org/pdf/2603.24226v1)

**作者:** Liren Yu `[一作]` (Taobao & Tmall Group of Alibaba), Bo Zheng `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 UniScale 框架，通过 ES³ 系统扩展整空间数据并使用 HHSFT 模型进行联合数据与架构共设计，用于提升搜索/推荐系统的排序性能。

**💡 创新点**

创新点在于将数据扩展与模型改进同步进行：ES³ 用整空间采样、层级标签归因和跨域样本搜索化生成高质量样本；HHSFT 采用异质分层注意力和全空间用户兴趣融合，显著抑制负迁移并提升跨域信息利用。

**🔧 技术方法**

采用的技术包括：整空间采样与标签归因、跨域样本搜索化、Token‑specific QKV 与多头注意力、全局特征交互、专家路由与门控注意力、MoE、FP16 量化、RDMA 通信等。

**📊 数据集**

使用 Taobao 规模电商数据集，包含数十亿级用户-商品交互，覆盖搜索、推荐和广告三域；离线评估采用搜索域测试集，线上 A/B 测试在 Taobao 搜索平台进行。

**📈 对比分析**

与 DLRM‑MLP、DCNv2、AutoInt、HiFormer、Wukong、RankMixer 等 SOTA 基线在 AUC、GAUC、HR@5 进行离线对比，HHSFT+ES³ 在 AUC+1.14%、GAUC+0.86%；线上 A/B 实验提升购买率 1.70%、GMV 2.04%；同时通过 FP16 量化与 RDMA 等优化，将 GPU 推理成本降低 55%、训练成本降低 40%。

**⚠️ 局限性**

局限性包括：对数据分布变化仍存在负迁移风险，需要更精细的专家或门控机制；模型规模增大对算力与存储提出更高要求；目前仅在 Taobao 业务上验证，跨业务或多语言场景的泛化还需进一步研究。

---

## 366. Uncovering Memorization in Timeseries Imputation models: LBRM Membership Inference and its link to attribute Leakage

**arXiv ID:** 2603.24213 | [PDF](https://arxiv.org/pdf/2603.24213v1)

**作者:** Faiz Taleb `[一作]` (EDF), Maryline Laurent `[通讯]` (SAMOVAR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了时序缺失填补模型的记忆化问题，提出基于参考模型的成员推断攻击（LBRM）并首次实现属性推断攻击。

**💡 创新点**

创新点在于引入Loss‑Based Reference Model (LBRM) 进行成员推断，证明其与属性推断高度相关，并可作为全序列泄露风险指标。

**🔧 技术方法**

采用参考模型对比、DTW损失、CWT峰值检测、滑动窗口评估等技术，对 Transformer/SAITS/iTransformer/AutoEncoder 等架构进行实验。

**📊 数据集**

使用伦敦智能电表能源消耗数据（LSMEC）和 ASHRAE 能耗数据作为评估数据集。

**📈 对比分析**

与传统 Naive Loss 方法对比，LBRM 的 AUROC 提升至 0.90，TPR@top25% 超过 90%；属性推断在 LBRM 选出的高风险序列上精度提升 10–15%。

**⚠️ 局限性**

局限性包括仅在缺失填补模型上验证，需构造性能相近的参考模型，且对细粒度窗口泄漏的量化仍有限。

---

## 367. Powerful Teachers Matter: Text-Guided Multi-view Knowledge Distillation with Visual Prior Enhancement

**arXiv ID:** 2603.24208 | [PDF](https://arxiv.org/pdf/2603.24208v1)

**作者:** Xin Zhang `[一作]` (Hangzhou Dianzi University), Hongbo Wang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 16688 | [OpenAlex ID](https://openalex.org/A5100395748)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种文本引导的多视角知识蒸馏框架（TMKD），利用视觉教师和文本教师（CLIP）提供丰富的监督信号，增强学生模型的表示能力。

**💡 创新点**

创新点包括：① 在视觉教师中加入多视角输入（RGB、边缘、 高频细节）并通过CLIP文本提示生成语义权重实现自适应融合；② 引入视觉‑语言对比正则化（CRD）将学生特征对齐至CLIP语义空间；③ 以双模教师与自适应融合相结合，显著提升蒸馏效果。

**🔧 技术方法**

主要技术：多视角图像构造、边缘增强与高频增强、文本提示生成、CLIP文本编码、语义引导自适应融合网络、特征级、logit级蒸馏、对比学习正则化。

**📊 数据集**

在五个数据集上验证：CUB‑200、RAF‑DB、CIFAR‑100、DTD、Stanford Dogs。

**📈 对比分析**

与多种基准蒸馏方法（KD、RKD、SP、CAT‑KD、TeKAP）以及不同教师‑学生组合对比，TMKD 在所有设置中平均提升 0.18%–4.49% 的 Top‑1 准确率，尤其在轻量级学生模型上提升更为显著。

**⚠️ 局限性**

局限性：① 需要额外的 CLIP 模型和文本提示，计算和存储开销略增；② 多视角融合的权重生成仍基于固定文本模板，可能对复杂场景适应性不足；③ 目前仅验证于分类任务，扩展到检测、分割等任务仍待探索。

---

## 368. IPatch: A Multi-Resolution Transformer Architecture for Robust Time-Series Forecasting

**arXiv ID:** 2603.24207 | [PDF](https://arxiv.org/pdf/2603.24207v1)

**作者:** Aymane Harkati `[一作]` (Hassan II University), Mohamed Hamlich `[通讯]` (Hassan II University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为IPatch的多分辨率Transformer框架，利用patch级别与点级别（自相关）两种时间序列表示共同捕捉局部细节与全局时序依赖，从而实现更准确的多变量时间序列预测。

**💡 创新点**

创新点主要包括：① 将传统的patch编码与点级自相关模块融合，形成双流结构；② 在自相关块中使用FFT与傅里叶可学习激活（FKAN）来更精确地提取周期性信息；③ 在训练与推理阶段实现更优的效果–效率权衡，显著提升对长序列的可扩展性。

**🔧 技术方法**

核心技术包括Transformer编码器（多头自注意力）、频域自相关注意力、快速傅里叶变换（FFT）、傅里叶可学习激活函数（FKAN）以及基于拼接的多分辨率特征融合。

**📊 数据集**

在七个公开基准数据集上评估：Electricity（321维）、Weather（21维）、ILI（7维）、ETTh1/ETTh2/ETTm1/ETTm2（每个7维）以及ETT（321维），覆盖日/小时/分钟分辨率、周期性与非周期性数据。

**📈 对比分析**

与PatchTST、TimesNet、Crossformer、Informer、Autoformer、FEDformer、TimeMixer、DLinear等主流基线相比，IPatch在大多数数据集和预测 horizon 上取得最优或次优 MSE/MAE；同时训练速度与推理延迟显著低于重度自注意力模型，显示出更优的性能‑效率比。

**⚠️ 局限性**

局限性：① 目前 patch 长度与重叠方式固定，缺乏自适应学习；② 对极短时序或极高频噪声的数据仍可能受限；③ 复杂的多维交互和解释性分析仍待进一步研究。

---

## 369. Invisible Threats from Model Context Protocol: Generating Stealthy Injection Payload via Tree-based Adaptive Search

**arXiv ID:** 2603.24203 | [PDF](https://arxiv.org/pdf/2603.24203v1)

**作者:** Yulin Shen `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**通讯引用:** 70750 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了在 Model Context Protocol（MCP）环境下的黑盒间接提示注入攻击，提出 TREE-structured Injection for Payloads（TIP）框架，可生成自然语义、结构合法的 JSON 注入载荷，绕过多种防御机制，控制工具增强型 LLM 代理。

**💡 创新点**

创新点：① 将攻击载荷生成视为树形搜索问题，采用粗细分层（Coarse‑to‑Fine）策略；② 引入路径感知反馈机制，利用历史高质量搜索路径提升全局优化；③ 通过工具响应模拟保持载荷语义一致；④ 防御感知适配模块，使搜索对动态防御（如重写、Perplexity 过滤）自适应。

**🔧 技术方法**

核心技术：黑盒优化 + LLM 代理生成、树结构搜索、粗细分层策略、路径感知反馈、工具响应模拟、动态防御适配。实现细节包括基于 Qwen2.5‑72B‑Instruct 的攻击 LLM、Monte‑Carlo 评估、Beam‑Search 剪枝。

**📊 数据集**

使用 InjecAgent 提供的四个工具集（GetWeather、GetProduct、ExpediaBooking、ShipManager）进行实验，评估四种主流 LLM（Qwen2.5‑7B、Qwen2.5‑72B、Llama3.1‑8B、Llama3.3‑70B）。攻击模型训练使用 Qwen2.5‑7B、InternLM2.5‑7B、GLM4‑9B 组成的模型集成。

**📈 对比分析**

实验结果显示：在无防御环境下 TIP 100% 的攻击成功率（ASR），且查询量仅 100 次，远低于 TAP 的 2500‑2600 次；在四类防御（Instruction Prevention、Sandwich Prevention、Perplexity Filtering、Finetuned Detector）下，TIP 仍保持 50‑100% ASR，优于 Fixed（10‑45%）和 TAP（0‑90%）。同时，TIP 展现出强大的跨模型迁移性，攻击载荷在小模型训练后可成功攻击大模型。

**⚠️ 局限性**

局限性：① 黑盒搜索仍需多轮查询，训练时间与实时性受限；② 对多轮持久注入或长时间记忆的效果尚未充分验证；③ 在极强防御（如全流程重写、自动判别器）下可能失效；④ 依赖工具响应结构与 schema 的稳定性，若工具接口变更需重新搜索。

---

## 370. TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification

**arXiv ID:** 2603.24278 | [PDF](https://arxiv.org/pdf/2603.24278v1)

**作者:** Guan Luo `[一作]` (Tsinghua University), Jianfeng Zhang `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Topo-VAE和Topo-Remesh两大模块，实现高分辨率网格的自编码器。通过将输入网格与网络输出统一到Dual Marching Cubes（DMC）拓扑框架，构建了显式顶点与面级对应关系，从而可以直接对拓扑、顶点位置与法线进行监督，显著提升细节保真度。

**💡 创新点**

核心创新包括：①利用DMC实现网格与预测共享拓扑结构，建立顶点/面级显式对应；②设计GPU加速的Topo-Remesh算法，采用L∞距离保持锐角特征；③构建稀疏体素‑点交叉注意力编码器和分离拓扑/几何的FlexiCubes解码器；④在训练中引入Teacher Forcing、基于GT的体素裁剪与分辨率渐进策略，稳定收敛。

**🔧 技术方法**

技术手段涵盖Dual Marching Cubes、FlexiCubes、稀疏体素点交叉注意力、L∞距离 remeshing、GPU并行实现、Teacher Forcing、体素裁剪、渐进分辨率训练等。

**📊 数据集**

训练数据：Sketchfab 320k高质量网格；评估数据：Objaverse、Thingi10K、Dora-Bench L3‑L4子集、Topo‑Bench（1k网格）等。

**📈 对比分析**

与Mesh2SDF、ManifoldPlus、Dora、Sparc3D等remeshing基线以及TripoSG、Dora、Hunyuan3D‑2.1、Trellis、Direct3D‑S2、SparseFlex等VAE基线进行对比，采用Chamfer Distance、F1 Score、ANC、F1‑Sharp等指标。实验表明Topo‑VAE在F1‑Sharp上提升约5–7%，整体F‑Score提升约8%，在细节与锐角保真度上显著优于现有方法；remeshing耗时仅18.5 s，显著快于其他基线。

**⚠️ 局限性**

局限性：在高分辨率下生成百万级体素，计算资源和时间需求较高；remeshing算法受基准分辨率限制，无法捕捉小于体素尺寸的细节。

---

## 371. Forecasting with Guidance: Representation-Level Supervision for Time Series Forecasting

**arXiv ID:** 2603.24262 | [PDF](https://arxiv.org/pdf/2603.24262v1)

**作者:** Jiacheng Wang `[一作]` (Xijing University), Luyan Zhang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReGuider插件，利用预训练时间序列基础模型的中间表示与预测模型编码器对齐，实现对时序预测模型的表示级监督，提升预测精度。

**💡 创新点**

创新在于以表示层对齐而非仅误差目标为监督，利用预训练模型的“通用时序词汇”引导目标模型编码器，且不增加推理成本。

**🔧 技术方法**

使用预训练时间序列基础模型Time‑MoE、欧氏距离对齐损失，结合Transformer、PatchTST、DLinear、TimeMixer等基线模型进行联合训练。

**📊 数据集**

在七大公开时序基准上评估：ETTh1/ETTh2/ETTm1/ETTm2、Weather、Electricity、Traffic。

**📈 对比分析**

通过与多种基线模型的对比实验，ReGuider平均提升约5%+（最高10%+）的预测性能，尤其在高维Traffic数据和长预测 horizon 上表现更稳健。

**⚠️ 局限性**

局限性包括训练阶段需要额外调用教师模型导致训练成本略增，对极低维或小样本数据指导效果有限，且缺乏对教师模型失配或跨域适配的深入分析。

---

## 372. Semantic Alignment across Ancient Egyptian Language Stages via Normalization-Aware Multitask Learning

**arXiv ID:** 2603.24258 | [PDF](https://arxiv.org/pdf/2603.24258v1)

**作者:** He Huang `[一作]` `[通讯]`, He Huang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多任务学习框架，在四个历史阶段的古埃及语（象形文字、底层文字、萨希迪克和博哈里克）上实现词级语义对齐，利用掩码语言建模、翻译语言建模、序列到序列翻译和 UPOS 标注，并加入两种正则化的表记化（拉丁转写与 IPA），从而在低资源环境下对古埃及语与英语以及彼此之间的语义相似性进行对齐；

**💡 创新点**

创新点在于将多任务学习与两种正则化的表记化相结合，提出基于 KL 散度的视图一致性约束和早期融合策略，首次在古埃及多阶段语料上评估不同视图集成方法对词级语义对齐的影响；

**🔧 技术方法**

主要技术包括 BERT encoder-decoder 轻量化模型、byte‑level BPE tokenizer、动态任务加权（不确定性缩放）、KL 散度一致性正则、早期融合嵌入混合、以及对齐评估指标（triplet accuracy 与 ROC‑AUC）；

**📊 数据集**

使用了 Thesaurus Linguae Aegyptiae（TLA）提供的象形文字和底层文字数据，Coptic SCRIPTORIUM 提供的萨希迪克与博哈里克文本，英文翻译文本，并对这些数据进行了缺失标记统一、语言标签前缀和分词器训练；

**📈 对比分析**

与仅使用 MLM 基线相比，多任务训练（MLM+Translation）在古埃及语–英语对齐上提升了 10%~15% 的 AUC，KL 整合的拉丁/IPA 归一化进一步提升了 5% 左右；在古埃及内部跨阶段对齐中，跨分支（象形文字-萨希迪克）AUC 约为 0.55，内分支（萨希迪克-博哈里克）可达 0.90；

**⚠️ 局限性**

主要局限包括：归一化导致信息丢失与语义误差；子词级分词器与表记化视图长度不匹配造成位置噪声；早期融合效果不佳；缺乏前埃及语的词形标注；模型对英语 pivot 的依赖导致跨分支对齐效果不易解耦；

---

## 373. Functional Requirements for Decentralized and Self-Sovereign Identities

**arXiv ID:** 2603.24250 | [PDF](https://arxiv.org/pdf/2603.24250v1)

**作者:** Daria Schumm `[一作]` (University of Zürich), Burkhard Stiller `[通讯]` (University of Zürich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性地将去中心化身份（DI/SSI）领域的非功能需求（NFR）转化为可操作的功能需求（FR），并构建了对应的形式化功能模型与判定规则。

**💡 创新点**

创新点在于：①首次系统化地将DI/SSI原则与NFR映射到具体的功能需求；②引入 Tropos 与设计模式的结合，实现需求的层级化、可追溯化；③使用逻辑谓词与公理对功能模型进行形式化，奠定可验证的评估基础。

**🔧 技术方法**

采用了需求工程方法（Tropos、设计模式、需求规范模板）和逻辑形式化（谓词、公理）来构建功能模型和 FR；未使用传统编程或机器学习技术。

**📊 数据集**

本研究不依赖于外部数据集，而是基于已有文献、法规（eIDAS、GDPR）和已公开的设计模式进行推导与验证。

**📈 对比分析**

未进行实验比较或性能评估；成果以 39 条唯一功能需求及其形式化谓词为输出，目标是为后续评估框架奠定理论基础。

**⚠️ 局限性**

局限性包括：①仅针对通用 DI/SSI 用例，未覆盖所有具体实现细节；②部分功能需求尚未通过形式化验证；③设计模式仍存在缺口（如同意、自治等），需后续补全；④未给出实际评估工具或实验验证。

---

## 374. Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition

**arXiv ID:** 2603.24242 | [PDF](https://arxiv.org/pdf/2603.24242v1)

**作者:** Aleix Sant `[一作]`, Carlos Escolano `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多语言联邦学习中LLM的参数高效微调方法，并提出了本地动态早停机制；

**💡 创新点**

创新点在于将多语言支持、LoRA参数高效微调与本地动态早停结合，系统评估了客户端语言多样性对性能、公平性和训练效率的影响；

**🔧 技术方法**

主要技术包括FederatedScope-LLM框架扩展、LoRA低秩适配器、FedAvg聚合及LDES-FL自适应早停；

**📊 数据集**

使用了包含八种欧洲语言的Alpaca Cleaned数据集，并在不同语言分布下划分联邦客户端；

**📈 对比分析**

通过与单语本地微调、全局多语言微调以及不同客户端多语言比例的FedAvg进行对比，结果显示在多语言客户端比例提升时平均多语言性能提升约2%-4%，公平性下降幅度减小，训练步骤明显减少；

**⚠️ 局限性**

局限在于使用的多语言数据并非严格平行，可能掩盖跨语言迁移效果，并且实验中客户端数据规模均衡，未探讨真实场景中的数据不均衡影响。

---

## 375. Detecting Underspecification in Software Requirements via k-NN Coverage Geometry

**arXiv ID:** 2603.24248 | [PDF](https://arxiv.org/pdf/2603.24248v1)

**作者:** Wenyan Yang `[一作]` (Tampere University), Samantha Bavautdin `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于几何距离的缺失需求类型检测方法，利用预训练句子编码器将需求映射到高维球面，并通过k近邻距离、类型受限覆盖率和基于软聚类的数量计数三种分数融合得到缺口评分。

**💡 创新点**

创新点在于：①对每个项目单独做kNN距离标准化，以区分固有稀疏区和真正缺口；②结合类型受限覆盖率和软聚类计数，弥补单一几何距离的局限；③通过两个超参数实现三种分数的可调融合，兼顾局部和全局缺口检测。

**🔧 技术方法**

主要技术包括 Qwen3-Embedding-0.6B 句子编码、BERTopic+UMAP+HDBSCAN 软主题分布、k近邻距离计算、z分数标准化、Gibbs软聚类计数、热图可视化与矩阵聚合。

**📊 数据集**

使用 PROMISE NFR 数据集（15 个软件项目，共 621 条自然语言需求，12 种质量属性类型），通过留一交叉验证进行实验。

**📈 对比分析**

与六种基线（随机、基于 TF‑IDF、分类器、MMD/类型距离、计数、全模型）对比，完整模型在 N≥50 的项目上 AUROC 0.935，接近人工标注计数上限 0.933；在所有项目上 AUROC 0.801。单一几何分数为 0.871，完全缺失时性能显著提高。

**⚠️ 局限性**

局限性包括：对部分缺口（< 100%）检测效果差（AUROC 0.5 左右）；软主题聚类导致每格样本稀疏，无法实现细粒度（类型×主题）定位；仅在大型项目（N≥50）上可靠；缺口检测依赖于已有语料库，无法发现语料库未覆盖的新关注点；缺乏真实工程师验证。

---

## 376. DVM: Real-Time Kernel Generation for Dynamic AI Models

**arXiv ID:** 2603.24239 | [PDF](https://arxiv.org/pdf/2603.24239v1)

**作者:** Jingzhi Fang `[一作]` (Huawei), Xuefeng Jin `[通讯]` (Huawei)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5040131467)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了一款实时编译器，结合字节码虚拟机和算子融合器，实现对动态 AI 模型的即时编译与高效执行。

**💡 创新点**

创新点在于将编译过程转为字节码解释执行，配合形状分块与 tile‑level 虚拟指令，既大幅降低编译开销，又支持多种模式的动态融合，突破了传统静态预编译与动态重编译的局限。

**🔧 技术方法**

使用了字节码虚拟机、形状分块与硬件对齐的 tiling 算法、tile‑level 虚拟指令、模式/堆叠融合策略以及动态图流式融合机制。

**📊 数据集**

通过在算子、子图和完整模型层面使用 BERT、MMoE、Qwen3‑14B、Llama3.1‑8B 等大模型，以及自定义的动态形状数据集进行评估。

**📈 对比分析**

与 Ascend NPU 上的 Torch‑NPU eager、recompile、dynamic 模式以及 MindSpore graph O0 进行对比，平均加速 1.1–1.3 倍，最大 11.77 倍；编译时间比对比方法快 5 个数量级，显示出显著的性能优势。

**⚠️ 局限性**

仅针对 Ascend NPU 实现，字节码与虚拟机高度依赖硬件特性，尚未在其他加速器验证，且在极大动态图的复杂控制流下仍需进一步优化。

---

## 377. Decentralized End-to-End Multi-AAV Pursuit Using Predictive Spatio-Temporal Observation via Deep Reinforcement Learning

**arXiv ID:** 2603.24238 | [PDF](https://arxiv.org/pdf/2603.24238v1)

**作者:** Yude Li `[一作]` (Harbin Institute of Technology), Jie Mei `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62325 | [OpenAlex ID](https://openalex.org/A5100695418)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种去中心化的端到端多智能体强化学习框架，利用原始 LiDAR 观测直接产生连续控制指令，实现多无人机在障碍丰富环境中的追捕与包围。

**💡 创新点**

核心创新是 Predictive Spatio‑Temporal Observation (PSTO)，将障碍几何与预判的追捕者与同伴运动在同一固定分辨率的自身体素投影中统一编码，打破传统稀疏特征或全局信息依赖，显著提升协作效率。

**🔧 技术方法**

采用双流卷积网络提取 PSTO，MAPPO 与 CTDE 训练策略，结合 LSTM 轨迹预测、Squeeze‑Excitation 通道注意、域随机化与延迟补偿，硬件层面使用 Livox Mid‑360 LiDAR 与 Intel NUC 计算平台。

**📊 数据集**

主要数据来源为 NVIDIA Isaac Sim/OmniDrones 平台生成的随机障碍平面追捕仿真环境（多种速度、障碍密度、团队规模），以及真实户外测试场景的 LiDAR 点云。

**📈 对比分析**

与 APF、Angelani、Janosov 等传统启发式算法以及 OPEN（SOTA）和多种 ablation 进行对比。实验表明在 2‑vs‑1、3‑vs‑1、4‑vs‑1 等配置中，PSTO 能实现 94%+ 的成功率、较低的捕获时间，并在单一策略下实现团队规模可扩展，无需再训练。

**⚠️ 局限性**

局限性包括：仅在平面环境验证，未完全实现全机载敌人感知（依赖共享状态），对垂直运动与三维障碍的处理尚未完成，且训练仍需大量仿真资源。

---

## 378. RVLM: Recursive Vision-Language Models with Adaptive Depth

**arXiv ID:** 2603.24224 | [PDF](https://arxiv.org/pdf/2603.24224v1)

**作者:** Nicanor Mayumu `[一作]` (University of Wollongong in Dubai), Farhad Oroumchian `[通讯]` (University of Wollongong in Dubai)

**通讯引用:** 1044 | [OpenAlex ID](https://openalex.org/A5059134555)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一种递归视觉语言模型（RVLM）与自适应迭代深度路由器（RRouter），实现可执行代码生成、跨模态视觉推理与可解释的诊断轨迹；

**💡 创新点**

①将单通VLM替换为生成-执行循环，所有诊断主张都有可复现的Python代码；②引入预先估计的复杂度评分与停滞检测，实现迭代深度的自适应分配；③在REPL中将图像作为一等对象，支持程序化裁剪、增强、差分图及跨模态查询；④将可解释轨迹转化为可打印的PDF报告，满足临床文档需求；

**🔧 技术方法**

递归语言模型+REPL环境、Python代码生成、Gemini 2.5 Flash基础VLM、视觉查询函数、图像处理库（PIL/NumPy/Matplotlib）、跨模态验证、复杂度评分与停滞检测机制、日志记录与PDF子代理；

**📊 数据集**

BraTS 2023 Meningioma（多模态脑MRI+分割掩码）与MIMIC‑CXR（胸部X光多视角+文本报告）两大公开数据集；

**📈 对比分析**

对比单通VLM与RVLM：在BraTS上，RVLM实现高一致性（如显著发现100%一致、细节更丰富）、检测跨模态不一致；在MIMIC‑CXR上，生成结构化Findings/Impression并正确识别视角和伪影；通过多次独立运行评估可靠性、对比迭代深度与成本（如单区块3步vs12步），显示自适应计算节约资源；

**⚠️ 局限性**

评估规模有限（少数病例）、仅处理2D切片、未进行临床验证、路由器特征目前仅针对BraTS、缺乏大规模自动化指标与量化评估、需进一步训练与多模态推广。

---

## 379. Variation is the Norm: Embracing Sociolinguistics in NLP

**arXiv ID:** 2603.24222 | [PDF](https://arxiv.org/pdf/2603.24222v1)

**作者:** Anne-Marie Lutgen `[一作]` (University of Luxembourg), Christoph Purschke `[通讯]` (University of Luxembourg)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5005933561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套将社会语言学视角系统化融入NLP的框架，并以卢森堡语的正字变体为案例进行实验。

**💡 创新点**

创新点在于把社会语言学的变体划分与NLP技术阶段对应，形成可操作的社会语言学NLP框架，并证明将标准与非标准文本混合训练能提升模型鲁棒性。

**🔧 技术方法**

利用正字规范化与去规范化算法处理数据，采用LuxemBERT和mBERT进行微调，评估多分类与标注任务。

**📊 数据集**

使用luxembertdata提供的意图、命名实体、词性、情感、主题分类、评论审核等任务数据，并对其进行标准化/去标准化与合并。

**📈 对比分析**

通过在标准、非标准与合并三种训练集上微调模型，再在同样三种测试集上计算加权F1；结果显示仅训练非标准的模型性能最低，混合训练模型在大多数任务上得到最高分，尤其在序列分类任务上提升显著。

**⚠️ 局限性**

限制在于去规范化覆盖范围有限，仅捕捉到算法使用的变体，实验仅在卢森堡语上进行，未能完全代表语言多样性。

---

## 380. SumRank: Aligning Summarization Models for Long-Document Listwise Reranking

**arXiv ID:** 2603.24204 | [PDF](https://arxiv.org/pdf/2603.24204v1)

**作者:** Jincheng Feng `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3971 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在长文档检索中先使用点式摘要模型 SumRank 压缩文档，然后再进行列表式重排的方法，显著提升效率和准确性。

**💡 创新点**

创新点在于将摘要生成与下游列表式重排任务对齐，通过三阶段训练（冷启动监督微调、专门的 RL 数据构造和基于奖励的 GRPO 对齐）实现摘要既紧凑又保持重要相关信息。

**🔧 技术方法**

采用 LLM 作为教师进行知识蒸馏得到学生摘要模型，随后使用 Group Relative Policy Optimization (GRPO) 强化学习以 NDCG@10 奖励对摘要生成进行对齐，并用 Qwen2.5-72B-Instruct 作为列表式重排器。

**📊 数据集**

在五个 TREC Deep Learning 轨道（TREC DL 19–23）上进行评估，使用 MS MARCO 源数据进行蒸馏和 RL 训练。

**📈 对比分析**

与 BM25、FirstP 截断、传统 Seq2Seq 摘要模型以及大规模 LLM 的零射摘要做对比，SumRank (7B) 在 NDCG@10 与 MAP@100 上均取得 SOTA 结果，且推理延迟仅为 3B/7B 模型的 1/30 左右。

**⚠️ 局限性**

局限性包括：受限于资源未测试更大规模 LLM；需要部署两模型（摘要与重排），未来需探索统一端到端模型。

---

## 381. Adapting the MVVM pattern to C++ frontends and Agda-based backends

**arXiv ID:** 2603.24199 | [PDF](https://arxiv.org/pdf/2603.24199v1)

**作者:** Viktor Csimma `[一作]` `[通讯]` (Eötvös Loránd University), Viktor Csimma (Eötvös Loránd University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文设计了一种可重复的方法论和软件开发工具包（Agdalache），用于在Agda/Haskell后端与C++ Qt前端之间构建符合MVVM模式、可验证的应用程序。

**💡 创新点**

创新点在于提出首个可任意中断且可通过FFI直接导出的Haskell Future实现，并实现了将Agda代码与任意Haskell库无缝连接的通用技术方案。

**🔧 技术方法**

实现技术包括agda2hs编译器、Haskell FFI、MVar与StablePtr、C++ RAII、MVVM架构、CMake+GHC构建链、QuickCheck单元测试等。

**📊 数据集**

基准测试使用了两类合成任务：自然数计算和二叉树递归评估，并与Rocq的OCaml提取结果进行对比。

**📈 对比分析**

通过测量翻译、编译和运行时间，agda2hs在翻译/编译阶段比MAlonzo快约一倍，性能与Rocq相当或在树评估任务中略胜一筹，表明其可接受的性能。

**⚠️ 局限性**

局限性在于基准仅覆盖极简任务，未验证大型真实应用的性能；依赖于特定硬件；并且agda2hs仅支持不含运行时依赖的Agda子集，需要手动处理Haskell依赖。

---

## 382. Bridging the Dual Nature: How Integrated Explanations Enhance Understanding of Technical Artifacts

**arXiv ID:** 2603.24325 | [PDF](https://arxiv.org/pdf/2603.24325v1)

**作者:** Lutz Terfloth `[一作]` (Paderborn University), Carsten Schulte `[通讯]` (Paderborn University)

**通讯引用:** 2515 | [OpenAlex ID](https://openalex.org/A5038084384)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过实验比较了在对技术工件Quarto!的解释中，强调建筑结构（Architecture）、强调相关意义（Relevance）以及二者整合（Integrated）三种说明方式对学习者理解（包括知识性理解和应用性理解）的影响。

**💡 创新点**

首次将双重本质（Architecture 与 Relevance）理论应用于解释研究，证明整合解释能显著提升学习者的“使用性理解”（enabledness），而单侧聚焦则无显著差异，揭示了对技术工件解释的有效结构。

**🔧 技术方法**

采用受控的单因素实验设计，使用五名训练过的“解释者”进行面对面解释；对解释内容进行定性编码（区分 Architecture 与 Relevance），并通过时间标准化分析解释的时间分布；统计分析采用 ANOVA、非参数检验及效应量（η²、Hedges' g）。

**📊 数据集**

实验样本为104名大学生，所有人均对Quarto!无先验知识；收集了解释过程录音、转录文本以及20余条编码条目（占约115k条），并基于此构建理解测评问卷（34项）。

**📈 对比分析**

通过单因素 ANOVA 与对比检验比较三种条件；结果显示在总得分与“知识性理解”上三组无显著差异，但“使用性理解”在 Integrated 条件显著高于单侧聚焦（F(1,102)=4.83, p=0.03, η²=0.045），效应量约为0.46。

**⚠️ 局限性**

局限性包括：测评问卷对 Architecture 与 Relevance 的子维度区分度低、受限于单一、相对简单的板块游戏（Quarto!），导致认知难度低且存在上限效应；解释时长与条件相关，可能混淆结果；实验环境为人造情境，生态效度有限；样本规模有限，可能无法捕捉更细微的差异。

---

## 383. Large Language Model Guided Incentive Aware Reward Design for Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2603.24324 | [PDF](https://arxiv.org/pdf/2603.24324v1)

**作者:** Dogan Urgun `[一作]` (Karabuk University), Gokhan Gungor `[通讯]` (Karabuk University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

利用大型语言模型自动生成并评估可执行的奖励程序，以提升协作多智能体系统的学习效率。

**💡 创新点**

通过以任务稀疏回报为唯一评估标准的“目标驱动搜索”，避免了代理误奖励的风险，并结合诊断指标引导生成。

**🔧 技术方法**

采用大型语言模型（LLM）进行程序合成，MAPPO进行多智能体训练，利用可解释的奖励诊断（Δ、ρ、NMI）进行反馈。

**📊 数据集**

在Overcooked‑AI的四种协作布局（Cramped Room、Forced Coordination、Coordination Ring、Asymmetric Advantages）上进行实验。

**📈 对比分析**

与基线MAPPO对比，经过两代搜索后稀疏回报与交付数均显著提升，尤其在互动瓶颈场景中提升幅度最大。

**⚠️ 局限性**

受限于固定计算预算与环境仿真次数，生成的奖励仍可能无法覆盖所有细粒度协调策略，且对不同环境的泛化能力尚需进一步验证。

---

## 384. Iterate to Differentiate: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation

**arXiv ID:** 2603.24430 | [PDF](https://arxiv.org/pdf/2603.24430v1)

**作者:** Shengfan Shen `[一作]` (Nanjing University), Shuai Wang `[通讯]` (Nanjing University)

**通讯引用:** 76754 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 I2D（Iterate to Differentiate）框架，通过递归使用自身合成结果作为参考音频，逐轮生成语音并聚合多种客观指标，以提升零射TTS模型的评估可分辨性。

**💡 创新点**

创新点在于：① 利用递归生成导致的分布漂移放大模型间的性能差异；② 引入多轮指标聚合方法（Mean、LWA、EWA、AUC）来恢复评估对SOTA系统的区分力；③ 在单一指标饱和时仍能保持与人类评估的高度相关性。

**🔧 技术方法**

使用的技术包括：递归语音合成（将上一轮生成的语音作为下一轮参考）；多轮客观指标聚合（Mean、Linearly Weighted Average、Exponentially Weighted Average、Area Under Curve）；常用评估指标（WER/CER、Speaker Similarity、DNSMOS、UTMOSv2、情感 F1）；主观 MOS 评估。

**📊 数据集**

采用的公开数据集：Chinese（Seed‑TTS‑Eval test‑zh，DiDiSpeech 2020）；English（LibriTTS test‑clean，3‑15 s 句子）；Emotion（CV3‑Eval Emotion Cloning，EmoBox、SeCap，涵盖 happy、sad、angry 三种情感）；以及 100 条中文样本做人类评测。

**📈 对比分析**

通过在三套数据集上对 11 款开源 TTS 模型进行 10 轮递归评估，比较多轮聚合指标与人类评测的 Spearman 相关系数。结果显示：① 单轮指标饱和时 SRCC 极低（如 UTMOSv2 仅 0.118），② 10 轮聚合后 SRCC 提升至 0.46 以上；② 对比模型发现 CosyVoice3‑RL、IndexTTS2、Qwen3‑TTS 在多轮评估中表现最优。

**⚠️ 局限性**

局限性包括：① 递归合成与指标计算导致显著的计算成本；② 框架偏重模型稳定性，可能忽略多样性与创造性评估；③ 评测主要针对开源模型，商业模型因成本与访问受限未能覆盖；④ 自然度评估受参考音频质量影响，可能与 speaker similarity 产生冲突。

---

## 385. ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers

**arXiv ID:** 2603.24414 | [PDF](https://arxiv.org/pdf/2603.24414v1)

**作者:** Songyang Liu `[一作]` (Beijing University of Posts and Telecommunications), Zhongyuan Wang `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 15423 | [OpenAlex ID](https://openalex.org/A5100741750)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ClawKeeper——一种针对OpenClaw生态的实时安全框架，集成了指令层技能保护、运行时插件硬化和独立Watcher外部监控；

**💡 创新点**

提出了独立Watcher架构，实现任务与安全的解耦、实时干预与自演化安全策略，突破了传统技能与插件的耦合与静态限制；

**🔧 技术方法**

使用了结构化安全策略注入、插件级硬化、WebSocket实时监控、行为扫描与日志分析等技术，形成多层防御链；

**📊 数据集**

构建了包含7类、140个对抗实例的安全基准数据集，覆盖提示注入、数据泄露、权限提升、恶意命令、配置篡改、漏洞检测与恶意技能等场景；

**📈 对比分析**

通过与7种开源安全方案的对比实验，ClawKeeper在所有7类任务中均实现了85–90%的防御成功率，明显优于最佳基线；

**⚠️ 局限性**

局限性包括对OpenClaw WebSocket接口的依赖、云部署时对隐私的潜在泄露、以及语言模型对安全策略解释的误差所带来的安全盲区。

---

## 386. AI-Supervisor: Autonomous AI Research Supervision via a Persistent Research World Model

**arXiv ID:** 2603.24402 | [PDF](https://arxiv.org/pdf/2603.24402v1)

**作者:** Yunbo Long `[一作]` `[通讯]`, Yunbo Long

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AI驱动的研究监督框架AI‑Supervisor，实现持续演进的研究世界模型、跨域自我改进循环和多智能体共识机制。

**💡 创新点**

创新点：持续演进的知识图谱式研究世界模型、基于多智能体共识的gap验证与自我修正循环、5‑WHY根因分析与跨领域技术迁移。

**🔧 技术方法**

技术：大型语言模型（如Qwen‑72B）、多智能体协作架构、知识图谱、共识与调度、根因分析与跨域检索、质量门控循环。

**📊 数据集**

数据集：Scientist‑Bench（27任务5域），多种公开基准（Safe RL、deepfake、LLM alignment、GNN、few‑shot）以及OpenReview、Semantic Scholar等文献检索源。

**📈 对比分析**

对比方法：LLM单向思路、AI‑Researcher、AI‑Scientist、Agent Lab、SciAgents等基线；AI‑Supervisor在gap发现精准度+24%，方法创新率+32%，跨项目结构连接+16，成本低8–16美元，GPU需求0。

**⚠️ 局限性**

限制：成本累计、需API访问、LLM推理受限、二值置信度、仍需人工判断与最终论文评审。

---

## 387. Exploring How Fair Model Representations Relate to Fair Recommendations

**arXiv ID:** 2603.24396 | [PDF](https://arxiv.org/pdf/2603.24396v1)

**作者:** Bjørnar Vassøy `[一作]` (Norwegian University of Science and Technology), Helge Langseth `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 2966 | [OpenAlex ID](https://openalex.org/A5040301804)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在推荐系统中通过中性表征（neutral representation）消除用户人口统计信息对模型影响的公平性定义，并评估了该定义与推荐结果公平性的关系；

**💡 创新点**

提出了两种基于推荐列表的 AUC 评估方法（神经网络 AUC 与人口统计比率 AUC），并系统比较了这些方法与传统的表示层 AUC 以及其他推荐公平性指标的表现；

**🔧 技术方法**

使用 VAE 及其公平变体（Noisy‑VAE、VAE‑Perturb、GAN‑VAE）模型，并通过神经网络、逻辑回归与人口统计比率特征进行 AUC 计算；

**📊 数据集**

实验数据包含 Movielens‑1M 真实数据（约 6k 用户、3.5k 物品，性别与年龄两种人口属性）以及基于长尾分布的合成数据；

**📈 对比分析**

通过比较表示层 AUC、推荐层 AUC、Item Ratio、Kendall‑Tau 等指标，发现表示层 AUC 只能作为上界，无法完全捕捉推荐结果的公平性；在相同表示层 AUC 的情况下，VAE 基线与公平 VAE 在推荐公平性指标上表现差异显著；同时，推荐层 AUC 趋向 0.5 时表示层 AUC 仍可能远高于 0.5，表明进一步优化表示层 AUC 并不能提升推荐公平性；

**⚠️ 局限性**

局限性包括：评估仅针对新用户；合成数据可能无法充分反映真实用户行为；AUC 评估易受过拟合影响，且不同计算方式结果不一致；仅聚焦中性表征公平性，未覆盖其他公平性定义；

---

## 388. The enrichment paradox: critical capability thresholds and irreversible dependency in human-AI symbiosis

**arXiv ID:** 2603.24391 | [PDF](https://arxiv.org/pdf/2603.24391v1)

**作者:** Jeongju Park `[一作]` (Kyungpook National University), Sekyung Han `[通讯]` (Kyungpook National University)

**通讯引用:** 2851 | [OpenAlex ID](https://openalex.org/A5101882212)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两变量动态系统模型，用以量化人类在AI协作中能力与委托之间的相互作用，并预测关键阈值下的技能崩溃与恢复。

**💡 创新点**

创新点在于将学习、实践与遗忘三大原则融入ODE模型，揭示出AI能力临界点K*≈0.85导致人类能力的非连续崩溃，并提出“抗脆弱”与“强制实践”两种治理策略。

**🔧 技术方法**

采用两方程ODE与代理基模型（ABM）进行动力学分析、参数估计与敏感性分析，并通过数值模拟检验阈值与政策效果。

**📊 数据集**

利用四个领域的实证数据（教育、医学内镜、空间认知、航空）校准遗忘率β，并对15个国家的PISA数学成绩时间序列进行模型拟合，以检验模型的跨领域适用性。

**📈 对比分析**

与线性、指数、Logistic衰减等传统衰退模型相比，所提ODE在7点PISA数据上实现R²≈0.97，AIC最低，能够预测阈值跳变与不可逆恢复，并在ABM中验证了抗脆弱效应和强制实践的超线性收益。

**⚠️ 局限性**

局限性包括模型对AI能力K的静态假设、对个体异质性与网络效应的简化、缺乏纵向实验验证以及对多技能系统的初步扩展仍不完整。

---

## 389. Optimal Small-Bitwidth Moduli Set for Residue Number Systems

**arXiv ID:** 2603.24387 | [PDF](https://arxiv.org/pdf/2603.24387v1)

**作者:** Danila Gorodecky `[一作]` (University of Lisbon), Danila Gorodecky `[通讯]` (University of Lisbon)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5044453605)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种基于五阶段贪心优化的确定性算法，用于在用户给定的整数区间内生成最优的互素模数集合，以最大化RNS的动态范围并降低模数位宽差异。

**💡 创新点**

创新点包括：①在模数选择过程中强制包含二次幂模数，既满足RNS计算简化，又兼顾位宽均衡；②采用分段乘法与对数累加的方式计算极大动态范围，突破 MATLAB 数值精度限制；③通过后置替换优化提升动态范围，兼顾位宽均衡，满足硬件并行实现需求。

**🔧 技术方法**

主要技术：数论与组合优化（互素检测、素因子分解、贪心选择与替换）、分段乘法与对数累加计算、MATLAB 实现与可执行文件打包。

**📊 数据集**

使用的“数据集”为多组整数区间 [X,Y]（如 [2,32]、[2,64]、[2,128]、[129,256] 等），通过这些区间生成模数集合并评估动态范围与复杂度。

**📈 对比分析**

评估方法：对不同区间执行算法，记录模数数量 k、动态范围位宽、计算复杂度估计；与传统仅选素数或固定模数集合的结果对比，显示在相同位宽下动态范围显著提升，且位宽差异更小。性能方面，算法在 MATLAB 中完成时间在可接受范围内，且可通过可执行文件在无 MATLAB 环境下快速运行。

**⚠️ 局限性**

限制：①算法的时间复杂度随区间大小和模数数量呈多项式增长，极大区间可能耗时较长；②动态范围的精确计算受限于 MATLAB 的数值精度，仅通过分段对数方法估计；③未考虑模数的特殊硬件实现成本（如乘法器、除法器），在实际 FPGA/ASIC 设计中仍需进一步验证。

---

## 390. ViHOI: Human-Object Interaction Synthesis with Visual Priors

**arXiv ID:** 2603.24383 | [PDF](https://arxiv.org/pdf/2603.24383v1)

**作者:** Songjin Cai `[一作]` (South China University of Technology), Changxing Ding `[通讯]` (South China University of Technology)

**通讯引用:** 4822 | [OpenAlex ID](https://openalex.org/A5038748720)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ViHOI框架，利用2D参考图像作为视觉先验来提升基于扩散模型的Human-Object Interaction（HOI）运动生成；

**💡 创新点**

1) 通过大规模视觉语言模型（VLM）提取可解耦的视觉与文本先验；2) 使用Q-Former适配器将高维先验压缩为紧凑token；3) 训练阶段使用真实渲染图像，推理阶段使用文本到图像模型合成的参考图像，从而显著提升对未见物体与场景的泛化能力；

**🔧 技术方法**

视觉语言模型（Qwen2.5-VL）、Q-Former先验适配器、Diffusion Transformer（DiT）运动生成器、文本到图像模型（Nano Banana），以及相关的注意力和交叉注意力机制；

**📊 数据集**

FullBodyManipulation、BEHAVE、3D‑FUTURE三个公开3D HOI数据集；

**📈 对比分析**

将ViHOI作为plug‑and‑play模块集成至MDM、ROG、CHOIS、SemGeoMo等基线，实验显示在FullBodyManipulation、BEHAVE以及未见物体测试中，多项指标（R‑Precision、C_prec、MPJPE、FID、Diversity等）均显著优于基线，且在大多数指标上实现了state‑of‑the‑art性能；

**⚠️ 局限性**

缺少细粒度手部标注，导致模型难以精确生成手指动作；训练阶段仅使用渲染图像，可能在样式对齐上存在差距，限制了对极端手部交互的表现。

---

## 391. Gendered Prompting and LLM Code Review: How Gender Cues in the Prompt Shape Code Quality and Evaluation

**arXiv ID:** 2603.24359 | [PDF](https://arxiv.org/pdf/2603.24359v1)

**作者:** Lynn Janzen `[一作]` (Technical University of Berlin), Veronika Solopova `[通讯]` (Technical University of Berlin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过收集真实用户对话、控制实验以及合成性别编码提示三种方法，探究性别化提示语言对大型语言模型（LLM）生成代码和自动审查的影响。

**💡 创新点**

首次揭示LLM审查器在代码审核中对女性作者的代码表现出更高的批准率，且提示语风格会显著影响代码可维护性与紧凑度。

**🔧 技术方法**

利用ChatGPT（4o/5）、Claude、OpenAI GPT‑3/4、Deepseek、Anthropic、Groq LLaMA等LLM进行代码生成与审核；采用Pylint、Radon进行静态分析；使用RoBERTa预测性别、LIME解释特征。

**📊 数据集**

使用自收集的学生与专业人员LLM对话记录（约750条提示），设计三道编程任务的控制实验，以及构造的五种性别编码提示；对比公开基准（HumanEval、MBPP等）来评测代码功能。

**📈 对比分析**

通过卡方检验、Welch‑t检验及统计相关分析，比较不同性别提示下的代码正确率、静态质量与审核通过率；发现功能正确性无显著差异，但女性提示的代码更易获得审核批准，提示风格影响可维护性与行数。

**⚠️ 局限性**

样本量有限、仅包含二元性别、任务为短函数级别、缺乏多轮人类评审验证、LLM版本更新快导致结果可重复性受限。

---

## 392. Language-Guided Structure-Aware Network for Camouflaged Object Detection

**arXiv ID:** 2603.24355 | [PDF](https://arxiv.org/pdf/2603.24355v1)

**作者:** Min Zhang `[一作]` (Chongqing University of Technology), Min Zhang `[通讯]` (Chongqing University of Technology)

**通讯引用:** 61054 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种语言引导结构感知网络（LGSAN），通过CLIP生成文本先验掩码、频域边缘增强模块、结构感知注意力以及粗细分局部细化模块来实现隐蔽目标检测。

**💡 创新点**

创新点包括①使用CLIP将文本语义直接转化为目标掩码以引导视觉特征；②在频域提取高频边缘信息的FEEM；③结合语义和边缘的结构感知注意力（SAAM）；④通过全局注意+局部细化（CGLRM）提升边界一致性与结构完整性。

**🔧 技术方法**

技术栈涵盖PVT‑v2视觉骨干、CLIP文本/视觉编码器、Fourier变换边缘增强、Transformer自注意力、多尺度特征融合、交叉熵+IoU+Dice损失等。

**📊 数据集**

在CAMO、COD10K和NC4K三个公开隐蔽目标检测基准上进行训练与评估。

**📈 对比分析**

与BGNet、ZoomNet、FSPNet、CGCOD等多种SOTA方法在Sα、Eφ、Fβ、M四项指标上进行对比，LGSAN在多数数据集上均取得最优或相近的最佳成绩，显著提升了定位精度、结构一致性与边界细节。

**⚠️ 局限性**

局限性在于网络结构较为复杂，计算与参数量较大，未来工作需要进一步轻量化语义引导与边缘建模模块，以降低推理成本。

---

## 393. Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing

**arXiv ID:** 2603.24326 | [PDF](https://arxiv.org/pdf/2603.24326v1)

**作者:** Cheng Cui `[一作]` (Baidu Inc.), Yanjun Ma `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PaddleOCR‑VL，一种先粗略定位有效视觉区域再细致识别的双阶段文档解析框架，显著提升了识别精度与推理速度。

**💡 创新点**

创新点在于：1）轻量级 Valid Region Focus Module (VRFM) 能同时完成区域检测与阅读顺序预测，剔除冗余背景；2）结合 0.9B 规模的跨模态 VLM，采用 NaViT 视觉编码与 ERNIE‑4.5‑0.3B 语言模型，实现高效的动态分辨率处理；3）系统级的数据构建与增量硬件适配方案，覆盖 109 种语言。

**🔧 技术方法**

核心技术包括 RT‑DETR + 指针网络（VRFM）、NaViT 视觉编码、3D‑RoPE 位置编码、MPL 2‑层投影、ERNIE‑4.5‑0.3B 语言模型以及多任务训练（OCR、表格、公式、图表）。

**📊 数据集**

使用 30M+ 图文对、5M+ 表格、1M+ 公式、0.8M+ 图表等多来源合成与人工标注数据，涵盖 109 种语言与多种文档类型。

**📈 对比分析**

在 OmniDocBench v1.5 上的端到端解析和子任务评测均超越现有管线、通用 VLM 及专用 VLM，整体分数达 92.62，且仅使用 2561 视觉 token；推理速度和显存占用相比 MinerU2.5、dots.ocr 等模型提升 50%+，显存减少 46%。

**⚠️ 局限性**

局限性包括：1）对极端稀疏或复杂排版场景，VRFM 可能错过细微信息；2）模型规模虽小，但在极大分辨率文档时仍需裁剪，导致部分细节缺失；3）多语言覆盖广泛，但对低资源脚本的性能仍有提升空间。

---

## 394. Complexity of basic boolean operators for digital circuit design

**arXiv ID:** 2603.24319 | [PDF](https://arxiv.org/pdf/2603.24319v1)

**作者:** Igor S. Sergeev `[一作]` `[通讯]`, Igor S. Sergeev

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述数字电路设计中基本布尔变换的复杂度和高效实现方法

**💡 创新点**

系统总结并给出多种基本运算（如计数器、加法器、解码器、多路复用器等）的最优复杂度和深度界限

**🔧 技术方法**

利用并行前缀树、压缩器、布尔函数合成等理论技术

**📊 数据集**

无具体数据集，本文为理论性综述

**📈 对比分析**

通过与已有文献的下界/上界对比，验证所给复杂度估计的紧密度

**⚠️ 局限性**

局限在于仅覆盖结构简单的布尔运算，缺少针对更复杂算术/矩阵运算的分析

---

## 395. Marchuk: Efficient Global Weather Forecasting from Mid-Range to Sub-Seasonal Scales via Flow Matching

**arXiv ID:** 2603.24428 | [PDF](https://arxiv.org/pdf/2603.24428v1)

**作者:** Arsen Kuzhamuratov `[一作]` (AXXX), Konstantin Sobolev `[通讯]` (AXXX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了名为Marchuk的潜在流匹配Transformer，用于30天内全球天气的概率预测；

**💡 创新点**

创新点包括：用可学习的2D空间嵌入+1D RoPE替代GeoRoPE；采用变长训练窗口、时间戳跨注意力、压缩64×的潜在空间以及更小的参数量；实现与大模型相当的性能并显著提升推理速度；

**🔧 技术方法**

使用了潜在自编码器（DC-AE）压缩、流匹配Diffusion Transformer、可学习位置编码、变长时间窗口、时间戳跨注意力以及自动回归推理等技术；

**📊 数据集**

使用ERA5重分析数据（WeatherBench2格式），包含84个通道，训练集为1979-2017，测试集为2018-2021；

**📈 对比分析**

与LaDCast小模型（375M）和大模型（1.6B）在WeatherBench-2指标（RMSE、ACC、CRPS等）进行对比；在15/30天时，276M的Marchuk优于小模型并接近大模型，且推理速度提升约6倍；

**⚠️ 局限性**

受DC-AE重构误差限制，无法完全捕捉细尺度现象；需整合实时观测进行数据同化；低分辨率限制热带气旋等细尺度事件的精确预测；

---

## 396. OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework

**arXiv ID:** 2603.24422 | [PDF](https://arxiv.org/pdf/2603.24422v1)

**作者:** Ben Chen `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 OneSearch‑V2，一种针对电商检索的生成式检索框架，融合思考增强的查询理解、自监督式推理蒸馏与行为反馈偏好对齐。

**💡 创新点**

创新点包括：①使用基于关键词的链式思考（CoT）对复杂查询进行深度语义对齐；②在教师‑学生权重共享的自蒸馏中通过信息不对称将推理能力内化至模型权重；③用直接行为反馈和分层奖励实现无单独奖励模型的偏好对齐。

**🔧 技术方法**

采用的大规模语言模型（如 Qwen3、BART）、多模态词表化、R‑Drop、FGM 对抗训练、焦点损失、GRPO/TPMA 的分层优势优化，以及离线/在线 A/B 评估。

**📊 数据集**

主要数据集来自快手商城近三个月的用户交互日志（约 3000 万点击/订单对），同时使用 30,000 条查询样本做离线测试。

**📈 对比分析**

与 OneSearch‑V1 以及多种基线（SFT+CoT、self‑distill、GRPO、TPMA 等）对比，离线 HR@10 / MRR@10 均提升约 2% 以上；在线 A/B 测试中，商品点击率提升 3.98%、购买转化 2.90% 等业务指标显著提升。

**⚠️ 局限性**

局限在于仍需依赖大模型推理，虽不增加额外延迟但对硬件资源占用高；推理仍受词表大小限制；对极端稀缺查询或多模态内容的适配尚需进一步研究。

---

## 397. Teacher-Student Diffusion Model for Text-Driven 3D Hand Motion Generation

**arXiv ID:** 2603.24407 | [PDF](https://arxiv.org/pdf/2603.24407v1)

**作者:** Ching-Lam Cheng `[一作]` (Singapore Management University), Shengfeng He `[通讯]` (Singapore Management University)

**通讯引用:** 5981 | [OpenAlex ID](https://openalex.org/A5056103024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个教师-学生扩散框架TSHaMo，用于从文本生成3D手部动作，并在推理阶段无需任何物体网格即可输出手部运动。

**💡 创新点**

创新点在于：①模型无关的教师-学生架构，使学生仅依赖文本即可生成运动；②共训练策略，让教师通过辅助信号（如MANO参数、关节位置、接触图）对学生进行指导；③在训练阶段加入多种辅助条件而不影响推理，实现高质量、语义对齐且多样化的手势生成。

**🔧 技术方法**

使用扩散模型（MDM、StableMoFusion）结合CLIP文本编码、教师-学生共训练损失、教师引导损失、分类器无关引导（CFG）等技术；教师网络可接收3D关节、MANO参数、接触图等辅助输入。

**📊 数据集**

实验使用两套手部交互数据集：GRAB（用于评估）和H2O（用于评估），仅使用手部运动序列并将文本提示作为动作标签。

**📈 对比分析**

与T2M、MDM、StableMoFusion等基线相比，TSHaMo在Top‑1/2/3准确率上显著提升（GRAB上从0.5提升到0.85，H2O上从0.20提升到0.26），KID下降，保持高多样性，表明生成的手部运动更真实、更符合文本语义。

**⚠️ 局限性**

局限性包括：①训练时需要额外的辅助信息（MANO、关节、接触图），在仅有文本的环境下可能难以获取；②对指导强度 λ 的敏感性，过大或过小都会影响性能；③仅在手部交互数据集上验证，未测试在更广泛的自然手势或不同语言描述下的泛化能力。

---

## 398. GeoRouter: Dynamic Paradigm Routing for Worldwide Image Geolocalization

**arXiv ID:** 2603.24376 | [PDF](https://arxiv.org/pdf/2603.24376v1)

**作者:** Pengyue Jia `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6249 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了GeoRouter——一种动态路由框架，能够为每个查询画像自动选择最合适的定位范式（检索或生成），从而提升全球图像地理定位精度。

**💡 创新点**

核心创新在于：①引入Distance‑Aware Preference Optimization（DisPO），将检索与生成范式预测误差差值映射为连续软标签；②构建GeoRouting大规模路由训练数据集，提供范式比较监督；③通过LVLM backbone结合LoRA实现端到端可学习的路由决策。

**🔧 技术方法**

使用了大型视觉‑语言模型（Qwen2‑VL‑7B‑Instruct）作为后端；通过Prompt模板将检索、生成预测和候选信息输入LVLM；利用LoRA实现参数高效微调；采用AdamW、距离感知损失优化路由器。

**📊 数据集**

训练数据来自MP16‑Pro数据库（100K查询），测试使用公开基准IM2GPS3K与YFCC4K。

**📈 对比分析**

与14种主流单范式基线（如GeoRanker、Img2Loc、G3等）对比，GeoRouter在所有距离阈值下均实现最高准确率，IM2GPS3K 25km阈值提升12.05%，YFCC4K 25km提升5.67%，整体平均提升约1–2个百分点。

**⚠️ 局限性**

限制主要包括：①仍受限于检索与生成范式自身误差分布，无法完全逼近Oracle上限；②路由决策需额外计算推理（检索、生成均需先行），对实时性能有一定影响；③对新的或极端场景（如极低分辨率、极度相似场景）路由误判的鲁棒性仍待进一步提升。

---

## 399. PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks

**arXiv ID:** 2603.24373 | [PDF](https://arxiv.org/pdf/2603.24373v1)

**作者:** Cheng Cui `[一作]` (Baidu Inc), Yi Liu `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了仅有5M参数的PP-OCRv5轻量级OCR系统，利用数据中心化方法提升性能

**💡 创新点**

关键创新是将训练数据按难度、准确度和多样性进行系统划分与采样，而非单纯扩容模型规模

**🔧 技术方法**

使用PP-LCNetV3 backbone、DB文本检测、SVTR_LCNet识别、GTC训练策略以及CLIP视觉编码器进行多样性聚类

**📊 数据集**

构建了约22.6M样本的多语言、多场景数据集，包括印刷、手写、特殊格式、垂直、艺术文字等

**📈 对比分析**

通过与PP-OCRv4/v3、主流OCR工具以及Qwen3-VL、GPT4o等大模型在OmniDocBench和内部基准上比较，PP-OCRv5在所有场景下实现了80%+加权准确率，ALL_avg编辑距离仅0.067，优于多数专用OCR工具且与大型VLM竞争

**⚠️ 局限性**

依赖大量高质量标注数据，数据收集与清洗成本高；对极端模糊或极少见文本仍可能表现不足；仅适用于两阶段管线，未充分利用多模态预训练的优势

---

## 400. Improving Lean4 Autoformalization via Cycle Consistency Fine-tuning

**arXiv ID:** 2603.24372 | [PDF](https://arxiv.org/pdf/2603.24372v1)

**作者:** Arsen Shebzukhov `[一作]` `[通讯]` (Stanford University), Arsen Shebzukhov (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对自然语言数学文本自动翻译为 Lean4 形式化代码，提出了一种基于循环一致性奖励的 RL 微调方法。

**💡 创新点**

创新点在于将循环一致性（NL→Lean4→NL）作为软奖励，通过 GRPO 强化学习显著提升翻译的语义保真度。

**🔧 技术方法**

主要技术包括 LoRA 参数高效微调、Qwen3.5-2B 模型、GRPO 强化学习、句子嵌入相似度作为奖励以及自回归解码。

**📊 数据集**

使用的数据集为 FineLeanCorpus（509k NL/Lean4 对）和 PutnamBench（660 个竞赛题目）做为评估基准。

**📈 对比分析**

与传统的无序 SFT 和按难度序列 SFT 进行对比，RL 模型在 FLC 700 题的平均循环一致性从 0.513 提升至 0.669，PutnamBench 从 0.422 提升至 0.561，且交叉熵损失仅略增 0.011。

**⚠️ 局限性**

主要局限包括循环一致性仅为代理指标，易被生成语法正确但语义空洞的 Lean4 套装所欺骗；回译模型的偏差可能导致奖励失真；且未使用 Lean4 编译器的严格语义验证。

---

## 401. CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control

**arXiv ID:** 2603.24366 | [PDF](https://arxiv.org/pdf/2603.24366v1)

**作者:** Yifeng Zhang `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1395 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个去中心化多智能体强化学习框架 CoordLight，用于城市交通信号控制；

**💡 创新点**

主要创新点包括：① 以车辆排队动态为基础的队列动态状态编码 (QDSE)，提供更细粒度的交通信息；② 邻居感知策略优化 (NAPO)，通过注意力机制识别并加权重要邻居的状态与动作，改进优势估计与价值学习；

**🔧 技术方法**

技术手段包括：多智能体强化学习（IPPO改进）、PPO、GAE、GRU、注意力网络、空间‑时间网络、状态‑动作解码器、队列动态模型与IDM、辅助队列长度预测损失；

**📊 数据集**

实验使用三大城市真实交通网络数据：济南（12个交叉口）、杭州（16个交叉口）和纽约（196个交叉口），并采集不同交通需求（多种流量级别）进行CityFlow仿真；

**📈 对比分析**

与传统方法（FixedTime、MaxPressure、Advanced-MP）及MARL基线（CoLight、MPLight、Advanced‑CoLight、Advanced‑MPLight、DenseLight、SocialLight）在平均行驶时间指标上进行对比。CoordLight在所有数据集和负载下均优于基线，平均行驶时间比最优对手SocialLight低约6–9%（济南）或7%以上（纽约），统计检验显示p<1e-8，差异显著；

**⚠️ 局限性**

局限性：仅针对同质交叉口且固定时长阶段；未考虑异构网络、异步控制或不同相序；依赖相对精确的传感器输入，尽管对噪声鲁棒性较好；需要大量训练样本；未处理紧急车辆、事故或道路关闭等特殊情况。

---

## 402. A Neuro-Symbolic System for Interpretable Multimodal Physiological Signals Integration in Human Fatigue Detection

**arXiv ID:** 2603.24358 | [PDF](https://arxiv.org/pdf/2603.24358v1)

**作者:** Mohammadreza Jamalifard `[一作]` (University of Essex), Javier Andreu-Perez `[通讯]` (University of Essex)

**通讯引用:** 4487 | [OpenAlex ID](https://openalex.org/A5029626997)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种神经符号系统，利用眼动追踪与fNIRS数据学习可解释的生理概念并通过可微分近似推理实现疲劳检测。

**💡 创新点**

创新点在于将注意力概念提取器与可微分规则层结合，实现参与者级校准、软阈值、概念忠诚度诊断，并为模型提供可解释的决策路径。

**🔧 技术方法**

使用技术包括注意力编码器、概念瓶颈、差分推理规则（Łukasiewicz运算）、软阈值、交叉熵+正则化损失、AdamW优化以及概念忠诚度评估。

**📊 数据集**

采用18名受试者的多模态生理数据集（眼动追踪42维+fNIRS48维），共560个10秒窗口，基于疲劳诱导实验收集。

**📈 对比分析**

通过留一受试者交叉验证与无逻辑层、SVM-RBF、随机森林等基线对比，得到72.1%±12.3%的准确率，与最优基线相当，逻辑层主要提升可解释性。

**⚠️ 局限性**

局限包括样本量小、标签混合时间效应、逻辑层对准确率提升有限、需更大规模验证和在线适应研究。

---

## 403. GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents

**arXiv ID:** 2603.24329 | [PDF](https://arxiv.org/pdf/2603.24329v1)

**作者:** Yunzhe Wang `[一作]` (University of Southern California), Volkan Ustun `[通讯]` (University of Southern California)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5089250268)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套完整的评测框架，用稠密多视角3D游戏视频来评估多模态大型语言模型在代理感知与推理方面的能力；通过自我‑他者‑世界三分解，构建三层认知层次的问答集；并使用模板化的组合式生成方法与结构化干扰词，便于诊断模型幻觉。

**💡 创新点**

创新点包括：①在多代理环境中引入Self–Other–World实体分解；②设计了从单一感知到时序推理再到跨视频推理的三层任务层级；③通过稠密时间线标注实现1.22标签/秒的高密度标注；④采用组合模板生成2.4K QA并附带分层干扰词，实现细粒度幻觉分析。

**🔧 技术方法**

技术手段涵盖：多模态LLM评测（Gemini、GPT‑5、Claude、Qwen3、Gemma、Seed等）；稠密多轨时间线字幕标注；模板化组合式QA生成；语言先验过滤与人工质检；时间/跨视频失真消融实验；跨域泛化评估。

**📊 数据集**

数据集包括9款多人商业3D游戏的同步多视角视频（共2709真实标签、1586干扰标签），生成2.4K QA；此外对Nexar驾驶碰撞视频和Ego‑Humans合作视频做了跨域实验。

**📈 对比分析**

采用零样本评估16款前沿MLLM，使用准确率衡量。结果显示：人类80.5%，最佳模型Gemini‑2.5 Pro71.3%；各层级从L1 61.2%逐步下降到L3 49.4%。最难任务为Occurrence Count（36.5%）和Cross‑Video Ordering（38.8%）。

**⚠️ 局限性**

局限性包括：未覆盖决策推理任务（如最佳动作选择）；意图识别主观性导致8%标签歧义；标注成本高且单一错误会在组合式生成中放大；跨域泛化仍需更多验证。

---

## 404. Refining time-space traffic diagrams: A neighborhood-adaptive linear regression method

**arXiv ID:** 2603.24312 | [PDF](https://arxiv.org/pdf/2603.24312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens

**arXiv ID:** 2603.24327 | [PDF](https://arxiv.org/pdf/2603.24327v1)

**作者:** Ciem Cornelissen `[一作]` (Ghent University), Pieter Simoens `[通讯]` (Ghent University)

**通讯引用:** 4317 | [OpenAlex ID](https://openalex.org/A5001314520)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种多模态自监督学习框架，通过在统一的Transformer中使用可学习融合令牌和SIGReg实现RGB与配套模态（如LiDAR深度、热成像）的联合表征；

**💡 创新点**

创新点在于将LeJEPA的SIGReg正则化扩展到多模态场景，并设计可学习的融合令牌做为稀疏的跨模态瓶颈，实现高效的跨模态交互；

**🔧 技术方法**

主要技术包括ViT分块、可学习融合令牌、跨模态注意力、Sketched Isotropic Gaussian Regularization（SIGReg）以及冻结探针评估；

**📊 数据集**

使用Waymo、nuScenes以及Teledyne FLIR ADAS三大驾驶数据集进行训练与评估；

**📈 对比分析**

与单模态LeJEPA、DINOv3、MultiMAE、ImageBind等基线以及早期/晚期融合进行对比，实验表明在Waymo、nuScenes的冻结探针评估中取得最佳或最接近最佳的定位、深度、分割指标，在FLIR上迁移学习与微调亦表现优异；

**⚠️ 局限性**

局限性包括仅在冻结探针上评估、仅使用ViT-Small/16小规模骨干、依赖精确的摄像机-深度对齐、未验证对校准误差或模态缺失的鲁棒性、缺少对更大模态集合或稀疏点云背骨的验证。

---

## 406. Toward Generalist Neural Motion Planners for Robotic Manipulators: Challenges and Opportunities

**arXiv ID:** 2603.24318 | [PDF](https://arxiv.org/pdf/2603.24318v1)

**作者:** Davood Soleymanzadeh `[一作]` (Texas A&M University), Minghui Zheng `[通讯]` (Texas A&M University)

**通讯引用:** 2101 | [OpenAlex ID](https://openalex.org/A5066836550)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对近年来利用深度学习提升机器人机械臂运动规划的研究进行系统综述，并提出构建通用神经运动规划师的框架与挑战；

**💡 创新点**

提出将深度学习模块映射到经典规划原语（采样、导向、碰撞检测、约束学习）的系统方法，并定义通用神经运动规划师的目标与指标；

**🔧 技术方法**

综述了MLP、CNN、RNN、GNN、Transformer、生成模型（VAE、GAN、Flow、Diffusion）、NeRF、Neural SDF、LLM 等技术在采样、引导、碰撞预测、约束建模等方面的应用；

**📊 数据集**

使用公开的机械臂规划数据集（抓取、装配、协同任务等）以及作者自建的基于仿真/真实环境的大规模轨迹数据；

**📈 对比分析**

通过与经典规划器（RRT、PRM、RRT*、BIT*、MPNet 等）在规划时间、成功率和路径长度等指标的对比，展示神经规划器在速度与成功率上优于传统方法，但在离散化细节与可解释性方面仍有不足；

**⚠️ 局限性**

主要局限包括对训练数据的强依赖、对未见环境的泛化能力有限、实时推理瓶颈、缺乏严格的安全保障以及对动态/未知环境适应性不足。

---

## 407. PINGALA: Prosody-Aware Decoding for Sanskrit Poetry Generation

**arXiv ID:** 2603.24413 | [PDF](https://arxiv.org/pdf/2603.24413v1)

**作者:** Manoj Balaji Jagadeeshan `[一作]` (Indian Institute of Technology Kharagpur), Pawan Goyal `[通讯]` (Oriflow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向诗歌的、可在推理阶段使用的音韵约束解码框架PINGALA，结合SLP1音素转写与跨编码器评估，实现了近乎完美的Anuṣṭubh韵律与较高的语义保真度。

**💡 创新点**

创新点在于：①使用可微调的“词长优先”形状函数和节拍完成奖励来鼓励生成具有完整单词与韵律边界的序列；②引入SLP1 ASCII音素转写显著降低了词表碎片化并提升了对韵律的学习；③利用跨编码器与加权二分类损失构建了比传统bi‑encoder更具判别力的无参考语义评估。

**🔧 技术方法**

主要技术包括：Transformer基础模型（NLLB‑dist‑1.3B、Phi‑4）、自定义约束对数转化（hard pruning + 结构增益）、β权重调节的节拍形状函数、SLP1音素转写、基于InfoXLM的跨编码器评分与校准。

**📊 数据集**

使用的公开数据集为Chandomitra英-梵平行语料（约8306条训练样本，1421条测试样本），并在该数据集上进行微调与评估；实验也涉及外部ODD数据集。

**📈 对比分析**

对比基线（NLLB‑dist‑1.3B、Phi‑4的标准约束解码）时，PINGALA在Full %、Partial %与Sim指标上均获得显著提升：Phi‑4+SLP1+PINGALA在测试集上达95.92%完整韵律、98.31%部分韵律、76.81%语义相似度；在NLLB上实现100%韵律且语义相似度提升至78.89%。

**⚠️ 局限性**

局限性包括：①实验仅使用8‑bit量化的Phi‑4，未验证在更大模型上的表现；②人类评估仅有两位专家，Kappa值低，说明主观性强；③SLP1方案未能应用于NLLB，因其预训练仅支持Devanagari，限制了跨模型推广。

---

## 408. Enhancing Drone Light Shows Performances: Optimal Allocation and Trajectories for Swarm Drone Formations

**arXiv ID:** 2603.24401 | [PDF](https://arxiv.org/pdf/2603.24401v1)

**作者:** Yunes Alqudsi `[一作]` `[通讯]` (Kocaeli University), Yunes Alqudsi (Kocaeli University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种统一任务分配与轨迹生成（UATG）框架，专为无人机灯光秀设计，实现大规模无人机编队的实时协同控制。

**💡 创新点**

创新点在于同时解决任务分配与时间参数化轨迹规划，采用匈牙利算法与凸二次规划，兼顾动态可行性、碰撞安全，并能在1000+无人机上实现秒级响应。

**🔧 技术方法**

使用了匈牙利算法、最小snap二次规划、时间参数化多项式轨迹、离散碰撞检测与局部修正、以及基于四旋翼动力学的约束模型。

**📊 数据集**

实验采用自制的三维形状点云生成器，构建了包含16、43、66、389、1008、1588无人机的多场景仿真数据集。

**📈 对比分析**

通过与CAPT方法对比，UATG在轨迹平滑度、能耗、碰撞安全性上表现更优；在大规模场景中，计算时间仅为5秒，体现出高度的可扩展性与实时性。

**⚠️ 局限性**

局限性包括：仍假设无障碍环境；未在真实飞行中验证；对强风、GNSS失效等外部扰动的鲁棒性待进一步研究。

---

## 409. 3D-Mix for VLA: A Plug-and-Play Module for Integrating VGGT-based 3D Information into Vision-Language-Action Models

**arXiv ID:** 2603.24393 | [PDF](https://arxiv.org/pdf/2603.24393v1)

**作者:** Bin Yu `[一作]` (HIT), Kai Chen `[通讯]` (ZGCA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何将3D几何信息（VGGT）无缝融合进Vision‑Language‑Action模型，提出了一种可插拔的语义条件自适应门控（GatedFusion）模块，并在多种大语言模型和架构上进行验证。

**💡 创新点**

创新点在于系统比较了九种融合方案，确定语义条件门控是最优策略；同时提供了轻量级、无须改动现有MLLM或动作专家的通用3D融合方法。

**🔧 技术方法**

核心技术包括VGGT预训练Transformer提取3D特征、语义条件门控融合、GR00T与π式VLA架构、Diffusion Transformer flow‑matching动作专家以及DeepSpeed ZeRO‑2高效训练。

**📊 数据集**

实验使用Open X‑Embodiment BridgeV2作为训练集，并在SIMPLER（real‑to‑sim OOD）与LIBERO（多任务）两大评估集上进行性能评测。

**📈 对比分析**

通过统一实验协议对九种融合方案进行对比，GatedFusion在SIMPLER上平均提升成功率至68.23%，在LIBERO上达到98.05%；在六大MLLM系列及两种架构上平均提升SIMPLER 7%且表现最稳健。

**⚠️ 局限性**

局限性包括仅评估VGGT并冻结其参数，未探索其他3D编码器；模块虽轻量但在π式架构下仍需额外显存；稀疏层融合需在效率与性能间进行折衷。

---

## 410. When AI Meets Early Childhood Education: Large Language Models as Assessment Teammates in Chinese Preschools

**arXiv ID:** 2603.24389 | [PDF](https://arxiv.org/pdf/2603.24389v1)

**作者:** Xingming Li `[一作]` (National University of Defense Technology), Qingyong Hu `[通讯]` (University of Oxford)

**通讯引用:** 7131 | [OpenAlex ID](https://openalex.org/A5036921959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了370小时的中国学前师生互动音频数据集TEPE-TCI-370h，并基于大语言模型的Interaction2Eval框架实现了从音频到教师-儿童互动质量评估的全流程自动化；

**💡 创新点**

首次在中国学前教育领域提供大规模自然交互数据与标准化质量标注，实现了端到端的AI评估体系，并通过专门的语音修正和评估代理显著提升了同质化的鲁棒性；

**🔧 技术方法**

结合语音识别（Paraformer、Whisper）、LLM（Qwen3‑Max、DeepSeek‑v3.1等）进行文本修正、评估推理，并在评估代理中使用结构化提示与证据链推理；

**📊 数据集**

使用自采集的TEPE-TCI-370h音频数据集（105个班级、370小时）以及专业评估标注的ECQRS-EC与SSTEW指标；

**📈 对比分析**

与人工专家评估和多种LLM（GPT‑5、Gemini‑2.5‑pro、DeepSeek‑v3.1、Qwen3‑Max）进行对比，平均指标一致率高达87‑89%，Cohen κ系数0.71‑0.87；评估效率比传统人工提升约18倍；

**⚠️ 局限性**

仅覆盖语言可检测的互动维度，无法评估非语言行为和环境因素，且LLM在复杂情境（如ECQRS‑EC）下仍低于专家水平，需进一步整合多模态与实时反馈。

---

## 411. Causal Transfer in Medical Image Analysis

**arXiv ID:** 2603.24388 | [PDF](https://arxiv.org/pdf/2603.24388v1)

**作者:** Mohammed M. Abdelsamea `[一作]` (University of Exeter), Xujiong Ye `[通讯]` (University of Exeter)

**通讯引用:** 4082 | [OpenAlex ID](https://openalex.org/A5037392936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了医疗影像领域中的因果迁移学习（CTL）方法，提出统一的分类框架并梳理其在分类、分割、重建、异常检测等任务中的应用；

**💡 创新点**

首次将因果推断与迁移学习结合，形成新的因果迁移学习范式，并给出完整的分类体系，强调域偏移视为因果问题，从而实现更稳健、更公平的临床 AI；

**🔧 技术方法**

利用结构因果模型（SCM）、潜在结果框架、Invariant Risk Minimization、对抗式域对齐、生成式反事实模型、图模型等因果技术，以及自监督学习、半监督学习、一次学习等深度学习策略；

**📊 数据集**

参考了多种公开医学影像数据集，包括REFUGE、Drishti-GS、RIM-ONE-r3（视网膜）、ACDC、Pancreas-CT、BraTS'19（脑肿瘤）、以及多中心 CT/MRI 数据集等；

**📈 对比分析**

通过与传统相关性域适应方法的对比，CTL 在跨域分割、分类等任务中实现了显著性能提升，例如在多目标分割任务中 Dice 分数提升 3–10%，在分类任务中 AUC 提升 1–2%，并在多中心验证中表现出更强的稳健性；

**⚠️ 局限性**

主要局限包括缺乏公开的因果标注与反事实数据集、因果结构学习的计算复杂度高、对因果假设的依赖可能导致模型在真实环境中的解释偏差，以及缺乏大规模临床验证与可解释性评估。

---

## 412. On the Use of Bagging for Local Intrinsic Dimensionality Estimation

**arXiv ID:** 2603.24384 | [PDF](https://arxiv.org/pdf/2603.24384v1)

**作者:** Kristóf Péter `[一作]` (University of Southern Denmark), Michael E. Houle `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 4424 | [OpenAlex ID](https://openalex.org/A5025538864)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了使用子采样聚合（subbagging）和邻域平滑相结合的方法来降低局部内在维数（LID）估计的方差，并系统研究了采样率 r、邻域大小 k 与袋数 B 对方差、偏差和均方误差（MSE）的交互影响。

**💡 创新点**

创新点在于：①将 Bagging 这一传统模型集成技术首次迁移到 LID 估计中；②给出理论分析证明减少 r 可上限降低袋间协方差；③探索并证明不同预平滑、后平滑及两者联合方式对 MSE 的协同提升效果；④提出了基于 k/r 比值的经验性超参数选择准则。

**🔧 技术方法**

主要技术包括：基于极值理论的 LID 估计器（MLE、TLE、MADA）；子采样聚合（subbagging）框架；邻域平滑（pre‑ and post‑smoothing）；理论上利用协方差、Jensen‑gap、超几何分布等工具进行方差/偏差分析；实验上采用网格搜索、MSE、方差-偏差分解等评价指标。

**📊 数据集**

使用了19个基准数据集（包含 2500 点的多维流形、Lollipop 数据集以及高维均匀分布数据），每个数据集都有已知的真实 LID 作为评估真值。

**📈 对比分析**

通过与基线（MLE、TLE、MADA）及单独平滑方法的对比，Bagging 与平滑的组合在大多数数据集上实现了显著的 MSE 降低（相对 MSE 下降 10%–40%），并在多种实验配置中保持低方差、可接受的偏差；单独的 Bagging 也普遍优于基线，且在合适的 k 与 r 区域内可实现更高的性能提升。

**⚠️ 局限性**

局限性包括：①实验仅在带有已知 LID 的人工流形数据集上验证，缺乏对真实复杂分布的评估；②对非均匀采样、噪声或高维稀疏数据的鲁棒性尚未深入探讨；③缺乏在实际下游任务中的间接评估与无监督模型选择方法；④Bagging 的计算开销随 B 线性增长，虽然在本文实验中 B=10 可行，但大规模并行实现仍需进一步优化。

---

## 413. What and When to Learn: CURriculum Ranking Loss for Large-Scale Speaker Verification

**arXiv ID:** 2603.24432 | [PDF](https://arxiv.org/pdf/2603.24432v1)

**作者:** Massa Baali `[一作]` (Carnegie Mellon University), Bhiksha Raj `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12396 | [OpenAlex ID](https://openalex.org/A5113017615)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Curry（CURriculum Ranking）自适应损失框架，用于在大规模、噪声多样的说话人验证训练中动态评估样本难度并分层梯度调节；

**💡 创新点**

创新点在于利用Sub-center ArcFace的子中心余弦相似度即时生成无监督置信度分数，结合跑动批量统计实现动态分层，且损失可兼容任意可微分样本级目标；

**🔧 技术方法**

主要技术包括W2V-BERT 2.0说话人编码器、MFA+ASP后端、Sub-center ArcFace、Curry损失包装、动态分层权重学习、梯度噪声抑制的课程学习策略；

**📊 数据集**

训练使用了超过50万身份的语音数据，涵盖VoxCeleb1/2、VoxBlink2、CommonVoice等多源数据集；

**📈 对比分析**

在标准验证协议VoxCeleb1-O和SITW上，与Sub-center ArcFace基线对比，Curry将EER分别从2.87%降至0.38%（↓86.8%）和从4.00%降至1.60%（↓60.0%），minDCF亦显著下降；

**⚠️ 局限性**

局限在于对极少数身份或极端噪声样本的动态分层仍可能产生误判，且需依赖子中心数量和阈值的经验设置，未对不同语言、口音等跨域情况做深入验证。

---

## 414. Learning Response-Statistic Shifts and Parametric Roll Episodes from Wave--Vessel Time Series via LSTM Functional Models

**arXiv ID:** 2603.24431 | [PDF](https://arxiv.org/pdf/2603.24431v1)

**作者:** Jose del Aguila Ferrandis `[一作]` `[通讯]`, Jose del Aguila Ferrandis

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一种端到端、数据源无关的堆叠LSTM模型，将波浪高度时间序列映射到船舶运动，特别能够重现参数滚动事件及其对应的统计分布变化。

**💡 创新点**

创新点包括：①端到端功能学习框架实现波浪历史→运动历史的映射；②聚焦严重海况下参数滚动的出现与统计转移；③提出针对尾部的相对熵和幅度加权损失函数来提升极端事件预测；④公开了包含三种海况、147条波浪-运动时间序列的URANS数据集。

**🔧 技术方法**

使用的技术包括：堆叠LSTM网络、指数倾斜（相对熵）损失、幅度加权MSE、短时傅里叶变换（STFT）分析、统计分布（PDF）比较与尾部保真度评估。

**📊 数据集**

数据集为DTMB 5415在Fr=0.4、波入射角30°条件下的URANS模拟，三种海况（SS‑1、SS‑2、SS‑3）各49个随机相位实现，总计147条波浪-运动时间序列。

**📈 对比分析**

在留出测试集上对比MSE、相对熵损失和幅度加权MSE；时间域误差平均低，尤其幅度加权MSE兼顾轨迹精度和尾部保真；在最严海况下模型能重现非高斯尾部，MSE在尾部表现不足。

**⚠️ 局限性**

主要局限包括：模型在最严海况下仍系统性低估滚动幅值；未验证对多DOF或不同船型的泛化；数据仅来自数值模拟，缺少实验验证。

---

## 415. IPsec based on Quantum Key Distribution: Adapting non-3GPP access to 5G Networks to the Quantum Era

**arXiv ID:** 2603.24426 | [PDF](https://arxiv.org/pdf/2603.24426v1)

**作者:** Asier Atutxa `[一作]` (University of the Basque Country), Eduardo Jacob `[通讯]` (University of the Basque Country)

**通讯引用:** 2031 | [OpenAlex ID](https://openalex.org/A5061281328)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出并实现了一种基于量子密钥分发（QKD）的非3GPP接入方案，用IPsec协议在5G网络中建立安全隧道。

**💡 创新点**

创新点在于将QKD产生的密钥直接集成到IKEv2握手中，替代传统的Diffie‑Hellman密钥交换，并通过ETSI GS QKD 014 API与KMS交互实现量子安全的密钥协商。

**🔧 技术方法**

使用的技术包括：IPsec / IKEv2协议、QKD硬件（ID Quantique Clavis XGR）、ETSI GS QKD 014标准化API、开源 5G 核心实现 free5GC 以及自定义的 N3IWF 和 N3IWUE 组件。

**📊 数据集**

实验数据来源于实测的硬件测试平台，包括两台服务器、一个QKD发射机与接收机、以及在两端收集的网络延迟和报文大小数据；未使用公开的标准数据集。

**📈 对比分析**

实验通过与传统 PSK‑和证书‑基 IKEv2 方案比较，评估连接建立时间和报文大小两项 KPI。结果显示 QKD 方案在 IKE INIT 阶段平均缩短约 40 ms（≈40%），整体连接时间比 PSK 方案快 4.62%，比证书方案快 5.17%，且总报文大小从 6604 B 降至 4991 B，提升了性能与通信效率。

**⚠️ 局限性**

局限性包括：QKD 需要专用光纤链路，部署成本高；KMS 与终端之间的无线访问安全性与实现难度大；实验仅覆盖两节点场景，未验证大规模多点网络；未讨论 PQC 与 QKD 的互补或替代方案，可能导致实现复杂度与攻击面扩大。

---

## 416. Real Talk, Virtual Faces: A Formal Concept Analysis of Personality and Sentiment in Influencer Audiences

**arXiv ID:** 2603.24410 | [PDF](https://arxiv.org/pdf/2603.24410v1)

**作者:** Shahram Chaudhry `[一作]` (New York University), Talal Rahwan `[通讯]` (New York University)

**通讯引用:** 4417 | [OpenAlex ID](https://openalex.org/A5007282319)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了虚拟与人类社交媒体影响者的受众在情感、人格与主题特征的共现结构，并通过 Formal Concept Analysis（FCA）和关联规则挖掘揭示两者的对话模式差异。

**💡 创新点**

创新点在于：①将 FCA 作为结构诊断工具，用闭集结构捕捉多信号共现关系，突破单一频率统计的局限；②发现虽然“appearance”主题在两类受众中出现频率相近，却仅在虚拟影响者的规则集中出现，体现了 FCA 对结构异质性的敏感性；③提出两层分析框架：周度聚合 FCA 生成话语概念，评论级关联规则揭示情感-人格-主题的方向性依赖。

**🔧 技术方法**

核心技术包括：Formal Concept Analysis（支持冰山过滤），关联规则挖掘（使用 Galicia 平台的近似算法，阈值：支持≥1%，置信≥0.8，提升>1.2），以及基于 RoBERTa 的情感分类器、Transformer 预测 Big Five 人格特质、Gemini Flash-Lite 的零射击主题分类。

**📊 数据集**

数据集来源于 YouTube：3 对匹配的虚拟与人类影响者（Lil Miquela/ Samantha Nicole, APOKI/ YOUNG POSSE, Milla Sofia/ Lydia Stoner），共计约 69,498 条英文评论（VI 29,327 条，HI 40,171 条），经去重、HTML 清理后进行特征提取。

**📈 对比分析**

比较方法：对同一阈值（minsup=1%、minconf=0.8、lift>1.2）下，HI 仅得到 8 条规则，VI 产生 51 条规则；规则集规模与聚类结构均显著不同（VI 三个结构簇 vs HI 单一簇）。在周度 FCA 层面，HI 产生 24 个过滤概念，VI 仅 10 个；概念的属性分布显示 HI 以情感为中心，VI 以人格为中心。实验表明，虚拟影响者的评论在情感与人格共现上更为多样化。

**⚠️ 局限性**

局限性包括：①人格与主题特征为文本推断，缺乏心理测量验证；②关联规则基于近似闭集算法，可能遗漏精确蕴含；③模型依赖于二值化阈值，可能导致属性常数化；④受众多样性与平台差异未完全控制；⑤仅针对 YouTube 进行实验，缺乏跨平台验证。

---

## 417. MolEvolve: LLM-Guided Evolutionary Search for Interpretable Molecular Optimization

**arXiv ID:** 2603.24382 | [PDF](https://arxiv.org/pdf/2603.24382v1)

**作者:** Xiangsen Chen `[一作]` (Hong Kong Polytechnic University), Yang Liu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 85567 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MolEvolve 框架，通过 LLM 生成可执行的化学符号规则并在 MCTS 中进行闭环验证，实现分子属性预测与优化的可解释式搜索。

**💡 创新点**

创新点在于将分子发现重构为可执行符号空间的前瞻规划，结合 LLM 的知识蒸馏和 MCTS 的演化搜索，解决了传统 GNN/LLM 的“活动斜坡”与可解释性缺失问题。

**🔧 技术方法**

使用的技术包括：大型语言模型（如 Qwen2.5-7B、GPT-4o）、符号知识蒸馏（CoT 推理 + 代码生成 + 迭代自我修正）、蒙特卡洛树搜索（MCTS）与外部化学工具 RDKit 的闭环验证。

**📊 数据集**

实验数据集：MoleculeNet（ESOL、Lipophilicity、BACE、BBBP、HIV）用于属性预测；ChemCoTBench 用于属性优化（LogP、QED）。

**📈 对比分析**

与 GNN（GIN、Graphformer）、基于 RAG 的 LLM（Automolco、MolRAG）以及强推理 LLM（GPT-4o、Qwen2.5）等基线比较，MolEvolve 在回归任务的 RMSE、分类任务的 ROC‑AUC、以及优化任务的平均提升 Δ 和成功率 SR% 上均取得显著优势。

**⚠️ 局限性**

局限性包括：对外部化学工具的依赖导致执行成本较高；MCTS 搜索仍受限于可执行规则集的完整性；在多目标优化和实验室自动化集成方面尚未充分验证。

---

## 418. Towards Reward Modeling for AI Tutors in Math Mistake Remediation

**arXiv ID:** 2603.24375 | [PDF](https://arxiv.org/pdf/2603.24375v1)

**作者:** Kseniia Petukhova `[一作]`, Ekaterina Kochmar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对AI辅导员在数学错误纠正任务中，构建了基于人类优先级层级的偏好评估体系，并通过合成对比样本训练奖励模型来自动评估教学质量。

**💡 创新点**

创新点在于：①提出了人类优先级层级化的评估框架；②设计了最小修订式的合成偏好数据生成流程；③利用该数据训练了比现有奖励模型更优的教育对齐奖励模型。

**🔧 技术方法**

主要技术包括：人类双人对比标注、最小修订生成策略、Bradley–Terry奖励模型训练、以及对比实验与统计检验。

**📊 数据集**

使用的数据集为：<cite>中提供的 192 条对话与 1,596 条辅导员回复的评注数据；以及通过合成生成的约 11,346 条偏好对。

**📈 对比分析**

与现有奖励模型（如<cite>、<cite>）和外部基线相比，最佳模型在人工评估测试集上达 0.74 的对比准确率，超过对手的 0.69。

**⚠️ 局限性**

局限性包括：人工评估样本规模有限、评估者偏好可能不具普适性、合成数据依赖单一模型风格、以及当前仅聚焦数学错误纠正场景。

---

## 419. Honey, I shrunk the scientist -- Evaluating 2D, 3D, and VR interfaces for navigating samples under the microscope

**arXiv ID:** 2603.24337 | [PDF](https://arxiv.org/pdf/2603.24337v1)

**作者:** Jan Tiemann `[一作]` (Helmholtz-Zentrum Dresden - Rossendorf), Ulrik Günther `[通讯]` (Helmholtz-Zentrum Dresden - Rossendorf)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

本研究对比了三种接口（2D桌面、3D桌面与VR）在显微镜下导航3D样本时的使用体验与性能，采用实验室模拟的显微镜环境并邀请12名专家进行任务测试。

**💡 创新点**

创新点在于：①首次系统性比较3D显微镜导航的2D/3D桌面与沉浸式VR的效果；②设计了可复现的基于程序生成的显微样本场景（轴突与管状结构），避免了真实样本的不可重复性；③提出了“50%标记时间”作为处理时间与完成率不足时的更稳健指标。

**🔧 技术方法**

使用技术包括：自研显微镜模拟框架（生成2D切片并在桌面/VR中实时渲染）、Oculus Quest 2 HMD与VR控制器、传统桌面软件（微控制台、滑轮、鼠标）以及标准的非参数统计方法（Kruskal‑Wallis、Dunn检验）。

**📊 数据集**

数据集为程序随机生成的三维几何结构（多分支轴突、含目标球体的管状样本），每个实验参与者使用相同的种子保证可比性；不使用公开真实显微图像集。

**📈 对比分析**

比较方法：在两类任务（高通量管道与轴突探索）下测量“50%标记时间”、标记成功率、完成率以及主观问卷。结果显示VR在两任务中均显著快（平均快≈3–5倍）、精度更高、完成率100%，而2D/3D桌面受限于时间上限，表现相对较差。

**⚠️ 局限性**

局限性：样本量仅12名专家，难以完全代表更广泛用户；实验对桌面界面使用的鼠标/键盘未覆盖实验室常用的操纵杆+Z轴旋钮；对VR头显的熟悉度低，可能影响学习曲线；以及未完全模拟真实显微镜的硬件与光学细节。

---

## 420. Near Linear Time Approximation Schemes for Clustering of Partially Doubling Metrics

**arXiv ID:** 2603.24336 | [PDF](https://arxiv.org/pdf/2603.24336v1)

**作者:** Anne Driemel `[一作]` (University of Bonn), Di Yue `[通讯]` (University of Toronto)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出在设施位置和 k‑median 问题中，当中心或客户具有低维度时，能在近线性时间内获得 (1+ε) 近似解

**💡 创新点**

创新点是将代理点技术与新的层次分解（包括装饰点和装饰子）相结合，构造门道距离（portal‑respecting distance）并通过动态规划实现高效求解

**🔧 技术方法**

使用的主要技术包括代理点网、随机层次分解（portal tree）、门道距离、badly‑cut 集合/bad 点结构分析、动态规划与加权集合覆盖求解

**📊 数据集**

实验主要基于理论度量空间，作者使用合成点集 X（低维度）和设施集 Y（高/低维度）进行理论证明，并未在公开数据集上做实验

**📈 对比分析**

与传统的 (1+ε) 近似方案比较，算法的时间复杂度为 2^{O(log(1/ε))}·(n+m)，比之前的 O(n^2) 或 O(n^3) 大幅提升，且在低维度场景下保持 1+ε 近似

**⚠️ 局限性**

局限性：需要随机层次分解导致成功概率 < 1；代理点与门户数目的指数系数会在 ε 很小或维度极高时导致常数项膨胀；对非常高维客户/中心分布的情况仍有限制

---

## 421. Generative Artificial Intelligence and the Knowledge Gap: Toward a New Form of Informational Inequality

**arXiv ID:** 2603.24335 | [PDF](https://arxiv.org/pdf/2603.24335v1)

**作者:** Raphael Morisco `[一作]` `[通讯]`, Raphael Morisco

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出生成式人工智能对知识差距的影响，并构建了扩展的知识差距理论框架

**💡 创新点**

将信息获取焦点从“访问”转向“评估”，提出“生成式AI知识差距”概念

**🔧 技术方法**

未使用具体技术，仅基于理论推导与文献综述

**📊 数据集**

无数据集，纯概念性讨论

**📈 对比分析**

无对照实验或性能评估，未给出可量化指标

**⚠️ 局限性**

局限性：缺乏实证验证、仅聚焦教育层面、未考虑其他影响因素或不同AI应用场景

---

## 422. Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions

**arXiv ID:** 2603.24322 | [PDF](https://arxiv.org/pdf/2603.24322v1)

**作者:** Shiqin Wang `[一作]` (Wuhan University), Kaiyan Zhao `[通讯]` (Wuhan University)

**通讯引用:** 7362 | [OpenAlex ID](https://openalex.org/A5090929533)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于强化学习的自适应语义类调度框架 HeuSCM，动态规划不同行业情境下的训练顺序，解决极端天气下的无监督域适配问题。

**💡 创新点**

创新点包括：①用高维语义状态提取网络（GM‑VAE+SKFEN）自动捕捉模型学习进度；②引入类别公平性权重的 Cα‑PG 策略梯度，保证各类均衡提升；③将混合源‑目标监督与自学习类序列耦合，实现更具信息价值的训练样本。

**🔧 技术方法**

核心技术：强化学习（策略梯度）、高维状态编码（GM‑VAE）、语义关键特征提取网络（SKFEN）、类别公平性奖励（Cα‑PG）、混合采样与多模态特征融合。

**📊 数据集**

使用的数据集包括：Cityscapes→ACDC（极端天气）、Cityscapes→Dark Zurich（夜景）、Nighttime Driving、GTA5→Cityscapes（合成→真实）等。

**📈 对比分析**

与现有方法对比（DeepLab‑v2、DAFormer、HRDA 等），在 ACDC、Dark Zurich、Nighttime Driving 上均取得最高或同类最佳的 mIoU（如 ACDC mIoU 72.9，Dark Zurich 52.8，Nighttime 59.3），验证了自学习调度的有效性。

**⚠️ 局限性**

局限性：1）RL 训练过程需要额外的采样与回放缓冲，训练成本较高；2）对极端稀疏类别的处理仍依赖奖励设计，可能在极端类别分布下表现不稳定；3）仅在语义分割任务验证，其他视觉任务的可迁移性尚待探索。

---

## 423. LATS: Large Language Model Assisted Teacher-Student Framework for Multi-Agent Reinforcement Learning in Traffic Signal Control

**arXiv ID:** 2603.24361 | [PDF](https://arxiv.org/pdf/2603.24361v1)

**作者:** Yifeng Zhang `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1395 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种将大型语言模型（LLM）与多智能体强化学习（MARL）相结合的交通信号控制框架LATS；

**💡 创新点**

创新点在于引入插件式教师-学生知识蒸馏模块，让预训练的嵌入式LLM生成语义特征并通过VAE在潜在空间进行蒸馏，最终实现无LLM推理的高效决策；

**🔧 技术方法**

使用的技术包括多智能体强化学习（PPO）、跨模态知识蒸馏、变分自编码器（VAE）、嵌入式LLM（Jina Embeddings）以及图注意力机制；

**📊 数据集**

使用了公开的两套交通数据集：Grid 5×5（均质网格）和Monaco（真实异质网络），并通过SUMO仿真进行训练与评估；

**📈 对比分析**

与固定时序、Greedy、Max-Pressure、纯LLM、纯RL（MA2C、IPPO等）以及其他参数共享方法（AttendLight、HeteroLight等）进行对比；LATS在平均排队长度、平均速度、交叉口延迟、完成率、行程时间和行程延迟等指标上均优于所有基线，尤其在异质网络和零射线迁移测试中表现突出；

**⚠️ 局限性**

局限性在于训练阶段对LLM的依赖导致计算成本高；对样本效率、跨网络泛化以及极端交通突发事件（如事故、天气变化）的鲁棒性尚未充分验证；

---

## 424. A Sensorless, Inherently Compliant Anthropomorphic Musculoskeletal Hand Driven by Electrohydraulic Actuators

**arXiv ID:** 2603.24357 | [PDF](https://arxiv.org/pdf/2603.24357v1)

**作者:** Misato Sonoda `[一作]` (Eth Zurich), Robert K. Katzschmann `[通讯]` (Eth Zurich)

**通讯引用:** 5342 | [OpenAlex ID](https://openalex.org/A5050915314)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计、控制并验证了一种完全由远程 Peano‑HASEL 电动肌肉驱动、采用 1:2 滑轮放大器和滚动接触关节的仿人形柔性手，能够安全、轻柔地抓取多种物体。

**💡 创新点**

创新点包括：① 将 HASEL 执行器远程布置在前臂，实现电压隔离与轻量化；② 采用 1:2 滑轮机制弥补软执行器有限行程；③ 通过监测执行器电流实现无外部传感器的抓取检测和接触感知闭环控制；④ 将柔性皮肤与滚动接触关节相结合，提升适应性与安全性。

**🔧 技术方法**

技术实现包括：Peano‑HASEL 软执行器堆叠、绳索驱动的柔性手指传动、滚动接触关节、1:2 滑轮放大传动、皮肤化学硅胶覆盖、基于电流的自感知控制算法。

**📊 数据集**

未使用公开数据集；实验基于自制装置（Frank A Research 3、DAQ、MATLAB）在手动控制下进行关节角度、末端力、抓取测试等量测。

**📈 对比分析**

与传统电机驱动手比较时，示范了 90° 关节角度目标但实际可达约 30°；末端力最高约 0.53 N；实现了 PINCH、TRIPOD、POWER 等抓取范式，能够轻柔抓取脆弱物体（如纸球），并通过电流阈值实现接触感知和安全闭环控制。

**⚠️ 局限性**

主要局限包括：① 需要高压（5.5–6.0 kV）导致安全与绝缘挑战；② 执行器力和运动范围受限，无法满足高负载任务；③ 线性弹性与滑轮系统的摩擦导致死区和有效转矩下降；④ 对电流阈值的依赖限制了精细控制与多任务适用性。

---

## 425. Evidence of an Emergent "Self" in Continual Robot Learning

**arXiv ID:** 2603.24350 | [PDF](https://arxiv.org/pdf/2603.24350v1)

**作者:** Adidev Jhunjhunwala `[一作]` (Columbia University), Hod Lipson `[通讯]` (Columbia University)

**通讯引用:** 31853 | [OpenAlex ID](https://openalex.org/A5025894735)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在仿真环境中使用Soft Actor-Critic训练单个四足机器人，顺序学习步行、摆动和跳跃三种运动行为，并通过共激活矩阵、块对角化与持久度评分，识别出一个在行为切换过程中保持不变的子网络，视为机器人的“自我”表征。

**💡 创新点**

证明在没有显式自我建模约束的情况下，持续学习多行为能自然产生一个稳定、任务无关的自我子网络，并提出基于共激活与持久度的量化评估方法。

**🔧 技术方法**

Soft Actor-Critic (SAC) 强化学习、两层150-150单元 MLP、共激活相似度矩阵、块对角化、Hungarian 匹配算法、持久度得分与子网络聚合分析。

**📊 数据集**

使用 IsaacLab 仿真平台生成的四足机器人自生成的参考状态集合；未使用公开数据集，仅在模拟中收集训练数据。

**📈 对比分析**

通过比较持续学习与单任务基准在不同训练周期的持久度得分、子网络规模以及子网络占比差异进行评估；结果表明持续学习下的自我子网络稳定性显著高于单任务，对 10 个随机种子均保持一致。

**⚠️ 局限性**

仅在小规模四足机器人与三种行为上验证；对更复杂机器人或更大行为集合的推广尚未确认；网络容量、训练策略等因素可能影响自我子网络的出现；缺乏因果干预验证其功能必要性。

---

## 426. Enhancing Efficiency and Performance in Deepfake Audio Detection through Neuron-level dropin & Neuroplasticity Mechanisms

**arXiv ID:** 2603.24343 | [PDF](https://arxiv.org/pdf/2603.24343v1)

**作者:** Yupei Li `[一作]` (Imperial College London), Björn Schuller `[通讯]` (Technical University of Munich)

**通讯引用:** 54569 | [OpenAlex ID](https://openalex.org/A5043060302)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 neuron‑level dropin 与 neuroplasticity 两种机制，用于在保持模型大小不变的前提下动态扩展并随后剪枝网络神经元，以提升音频 deepfake 检测模型的性能与效率。

**💡 创新点**

创新点在于：①在层级细粒度上实现神经元级的动态添加与后续剪枝，模仿大脑可塑性；②仅训练新增神经元即可显著降低 EER，避免传统的全网络重新训练；③提供通用框架，适用于 CNN、RNN 与 Transformer‑style 模型。

**🔧 技术方法**

采用的技术包括：dropin 机制（冻结原网络只训练新增神经元）、plasticity 机制（训练后剪枝）、低秩适配 LoRA 对比、Grad‑CAM 解释性分析，以及在 ResNet18、GRNN、Wav2Vec 2.0 等架构上的实现。

**📊 数据集**

使用的数据集为 ASVspoof 2019 的 Logical Access (LA)、Physical Access (PA) 子集以及 FakeorReal (FoR) 数据集。

**📈 对比分析**

通过与 baseline、dropin unfrozen（扩展模型训练）以及 LoRA 进行对比，报告了 EER、后向时间(ms)、参数量与可训练参数量。在 LA/PA/FoR 三大数据集上，dropin frozen 在保持参数不变的同时将后向时间降至约 60%–70%，而 plasticity 在不增加参数的前提下将 EER 降至 0.04%（Wav2Vec 2.0）或相对 39%–66% 的显著提升，表现出优于现有 SOTA 的性能。

**⚠️ 局限性**

局限性包括：剪枝仍采用简单的“全部剪掉新增神经元”策略，缺乏基于功能冗余的精细化标准；只实现一次性添加与剪枝，未探索连续或更细粒度的可塑性过程；在小模型上提升幅度有限；对大脑可塑性机制的理解尚不完整，导致算法设计与真实生物过程的贴合度有限。

---

## 427. Evaluating Chunking Strategies For Retrieval-Augmented Generation in Oil and Gas Enterprise Documents

**arXiv ID:** 2603.24556 | [PDF](https://arxiv.org/pdf/2603.24556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 428. Usability Evaluation and Improvement of a Tool for Self-Service Learning Analytics

**arXiv ID:** 2603.24321 | [PDF](https://arxiv.org/pdf/2603.24321v1)

**作者:** Shoeb Joarder `[一作]` (University of Duisburg-Essen), Louis Born `[通讯]` (University of Duisburg-Essen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对自助学习分析工具Indicator Editor进行全面的可用性评估与迭代改进，结合定性访谈、可用性检查和大规模学生工作坊，使用标准化量表(SUS、UEQ、NPS)衡量用户体验，最终得到可用性更佳的最终界面；

**💡 创新点**

创新点在于：①系统化地整合多种评估方法（访谈、可用性检查、工作坊+量表）对SSLA工具进行可用性与体验评估；②基于评估结果，提出并实现了针对流程指导、即时反馈、信息分层、交互一致性、简化术语等方面的具体设计改进；③展示了可用性改进后对用户感知可用性、满意度与推荐意愿的提升。

**🔧 技术方法**

技术上采用React.js和Material UI实现高保真原型和最终界面，并将工具部署至Open Learning Analytics Platform（OpenLAP）。

**📊 数据集**

使用了学生自有的学习数据（来自CourseMapper平台），并让46名本科/硕博生在实验中自行选择并实现指标；数据来源为真实学习平台的数据集，未使用公开大型数据集。

**📈 对比分析**

评估通过标准化问卷量化（SUS平均76.8，UEQ多维度正向评分但perspicuity负值，NPS为-28.3%）来比较改进前后的可用性；结果显示改进后可用性提升、用户体验整体正向，但仍存在可解释性和可学习性不足。

**⚠️ 局限性**

局限性包括：仅在学生群体中测试，未覆盖教师、学习设计师等关键利益相关者；实验时间短，未评估长期使用效果；缺乏与其他SSLA工具的对比；数据集依赖特定平台，可能影响可迁移性。

---

## 429. Efficient Equilibrium Computation in Symmetric First-Price Auctions

**arXiv ID:** 2603.24317 | [PDF](https://arxiv.org/pdf/2603.24317v1)

**作者:** Aris Filos-Ratsikas `[一作]` (University of Edinburgh), Charalampos Kokkalis `[通讯]` (University of Edinburgh)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5093967908)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在对称一价拍卖（第一价拍卖）中，作者提出了高效算法，用来计算在独立同分布（i.i.d.）私有价值下的贝叶斯纳什均衡（BNE）。

**💡 创新点**

创新点在于：①首次给出了在连续和离散出价空间下的多项式/查询多项式时间算法；②通过对经济学解析公式的“翻译”，实现了对连续出价空间的完全可计算性；③提出了匹配的下界证明，证明查询复杂度 O(1/ε) 已达到最优。

**🔧 技术方法**

技术主要包括：利用一价拍卖的解析均衡公式，将其转化为离散化的 Riemann 和求解；使用信息论下界构造，证明查询下界；构造连续函数的 Lipschitz 近似和二分搜索求解；在白盒模型下利用算术电路与多项式分段表示进行精确计算。

**📊 数据集**

该工作为理论性研究，没有使用实际数据集，所有结果均基于抽象的概率分布（连续、离散、分段多项式、Lipschitz 等）。

**📈 对比分析**

与以往仅给出 PTAS 或证明 NP/PPAD 难度的结果相比，该方法在 i.i.d. 情况下实现了真正的多项式/查询多项式时间求解；在显式分段多项式分布下甚至能得到完全精确的解。

**⚠️ 局限性**

局限性包括：仅适用于对称 i.i.d. 价值；离散出价空间下的算法对最小出价间隔 α 依赖；在隐式白盒模型下只能得到 FPTAS，无法实现多项式时间内的精确解；以及对非对称或相关价值分布的情况仍未得到解决。

---

## 430. Structure of weighted projective Reed-Muller codes

**arXiv ID:** 2603.24397 | [PDF](https://arxiv.org/pdf/2603.24397v1)

**作者:** Jade Nardi `[一作]` (University of Rennes), Rodrigo San-José `[通讯]` (Virginia Tech)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5018458682)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提供了加权投影Reed-Muller码的基本结构特性全面概述，给出了这些码的递归构造，并在某些权重条件下推导了广义汉明权重的界限，同时获得了其子域子码和对偶码的递归构造。

**💡 创新点**

创新点在于首次研究了加权投影Reed-Muller码的广义汉明权重，并提供了对偶码的更一般性研究，尤其是在低度情况下将其描述为评估码。

**🔧 技术方法**

使用了递归构造技术，并结合了代数几何中的一些概念，如toric几何和Schur乘积。

**📊 数据集**

论文中没有具体提到使用的数据集，但涉及的理论和构造可以应用于加权投影Reed-Muller码的具体实例。

**📈 对比分析**

与其他方法的比较主要体现在对广义汉明权重的界限和对偶码的构造上，性能方面的具体数值未在文中给出，但理论上表明WPRM码在某些条件下优于WRM码。

**⚠️ 局限性**

限制在于对广义汉明权重的计算仍然是一个开放问题，尤其是在高于某个度数时，计算复杂性较高。

---

## 431. The role of spatial context and multitask learning in the detection of organic and conventional farming systems based on Sentinel-2 time series

**arXiv ID:** 2603.24552 | [PDF](https://arxiv.org/pdf/2603.24552v1)

**作者:** Jan Hemmerling `[一作]` (Thünen Institute of Farm Economics), Stefan Erasmi `[通讯]` (Thünen Institute of Farm Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了利用Sentinel-2多光谱时间序列和视觉Transformer模型区分有机与常规农业系统。

**💡 创新点**

结合多任务学习与空间上下文扩展，首次系统评估不同作物对有机/常规判别的可分性。

**🔧 技术方法**

采用TSViT视觉Transformer架构，扩展为多任务学习，实验不同patch尺寸。

**📊 数据集**

使用德国巴伐利亚州IACS作物/管理数据与Sentinel-2 2020年10天时间序列。

**📈 对比分析**

与随机森林基线对比，TSViT在多作物均取得整体准确率约93%，多任务仅略提升，空间上下文对有机识别尤为重要。

**⚠️ 局限性**

主要局限在地区范围、作物多样性不足、时间序列均匀化、单年份数据及缺乏土壤等环境变量影响。

---

## 432. No Single Metric Tells the Whole Story: A Multi-Dimensional Evaluation Framework for Uncertainty Attributions

**arXiv ID:** 2603.24524 | [PDF](https://arxiv.org/pdf/2603.24524v1)

**作者:** Emily Schiller `[一作]` (XITASO GmbH IT & Software Solutions), Luca Longo `[通讯]` (University College Cork)

**通讯引用:** 3286 | [OpenAlex ID](https://openalex.org/A5032601645)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套针对不确定性归因方法的评估框架，并在 Wine Quality 与 MNIST 数据集上进行实验验证。

**💡 创新点**

创新点在于：①将 XAI 领域 Co‑12 框架中的四项属性（正确性、一致性、连续性、紧凑性）迁移并细化至不确定性归因；②引入了专门针对不确定性归因的“传达性”属性，并设计了不确定性传达相似度（UCS）指标；③通过四项严谨的 sanity 检验评估指标可靠性。

**🔧 技术方法**

技术方法包括：蒙特卡罗 Dropout/Dropconnect（MCD/MCDC）作为不确定性量化器；梯度相关解释器（LRP、Integrated Gradients、Input×Gradient）与概率抽样解释器（SHAP、LIME、GradientSHAP）作为特征归因器；采用基于特征翻转、模糊化、RIS、RRI、UCS 等多种评价指标。

**📊 数据集**

实验使用的两个公开数据集：Wine Quality（表格数据，11 连续特征）和 MNIST（图像数据）。

**📈 对比分析**

比较方法：将每种 UQ 方法与多种归因器组合成 8–10 种归因方案，使用上述指标进行量化。结果显示：梯度基方法在正确性与一致性上优于基于扰动的 SHAP/LIME；MCDC 在多项指标上表现更佳；RRI 与 UCS 指标互补，说明它们衡量的是传达性的不同方面。

**⚠️ 局限性**

局限性包括：①实验仅覆盖两种数据类型和有限样本，泛化性待验证；②仅评估了 Bley 等人提出的“uncertainty attribution”框架，其他归因方法的适用性未知；③某些指标（如 feature flipping、RIS、UCS）对参数、扰动策略敏感，可能导致结果不稳定；④未包含用户研究或真实应用场景的评估。

---

## 433. Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models

**arXiv ID:** 2603.24484 | [PDF](https://arxiv.org/pdf/2603.24484v1)

**作者:** Siqi Liu `[一作]` (University of Science and Technology Beijing), Jiansheng Chen `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 3837 | [OpenAlex ID](https://openalex.org/A5100668653)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VisionToM 框架，利用可学习的干预向量对多模态大型语言模型（MLLM）的注意力层进行有针对性的调整，以提升仅凭视觉信息进行 Theory of Mind（ToM）推理与多选 QA 以及开放式生成的表现。

**💡 创新点**

创新点包括：①仅使用原始视觉输入，无需额外提示或文本注释；②将干预向量注入注意力机制，实现视觉注意和 ToM 推理双向引导；③在视觉干预阶段采用 PGD 对抗样本生成方向，提升鲁棒性；④在 ToM 推理阶段通过聚类+编码器学习针对不同推理失败模式的定向修正；⑤整体保持 MLLM 主干冻结，降低 fine‑tuning 需求。

**🔧 技术方法**

技术手段：视觉‑语言 Transformer（LLaVA‑Next‑Video、Qwen2.5‑VL）; 线性探针分析内部表示；PGD 对抗扰动生成；聚类+专用编码器学习修正向量；轻量级干预向量注入到注意力输出；TruthfulQA 评估指标用于开放式生成。

**📊 数据集**

主要使用 EgoToM 数据集（第一人称视频+ ToM 多选 QA），并在此基准上与人类、GPT‑4、Video‑Llama2‑72B、CogVLM2、GPT‑4o、Gemini‑2.5‑Flash 等模型进行对比。

**📈 对比分析**

评估采用 Top‑1 Accuracy 计量 Goal、Belief、Action 三个子任务的性能。VisionToM 在 LLaVA‑Next‑Video 上相较基线分别提升 13.0%、6.4% 与 5.7%；在 Qwen2.5‑VL 上提升 0%、6.4% 与 1.5%。与人类基线相比，Goal 任务已逼近但 Belief 与 Action 仍有较大差距。开放式生成任务中，VisionToM 同时提升了 True、Info 与 True&Info 指标。

**⚠️ 局限性**

局限性包括：①对 Belief 与 Action 任务的提升有限，仍难与人类水平匹配；②干预向量在训练后固定，缺乏在线自适应；③需要额外的 probe 与 encoder 训练成本；④仅在两款 MLLM 上验证，跨模型泛化性待进一步测试；⑤仅针对无文本注释的视频输入，未覆盖更丰富的多模态或实时交互场景。

---

## 434. Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA

**arXiv ID:** 2603.24481 | [PDF](https://arxiv.org/pdf/2603.24481v1)

**作者:** John Ray B. Martinez `[一作]` `[通讯]` (Harrisburg University of Science and Technology), John Ray B. Martinez (Harrisburg University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 MARC（多代理一致性验证框架），利用四个领域专家代理（呼吸、心脏、神经、胃肠）生成答案，并通过两阶段一致性验证（Two‑Phase Verification）得到每个专家的自信分数（S‑score），随后用 S‑score 加权投票来选择最终答案并校准置信度。

**💡 创新点**

创新点在于：① 将基于内部一致性的自检机制与多代理集成，通过 S‑score 作为权重实现无标签的置信度校准；② 证明两阶段验证是提升校准的主导因素，而多代理推理则提升准确率；③ 在医学多选题上首次系统评估此方法，显示大幅降低 ECE 并提升 AUROC。

**🔧 技术方法**

核心技术包括：Qwen2.5‑7B‑Instruct LLM（按专业定制系统提示），两阶段一致性验证（提取事实断言、独立/参照回答、Jaccard 不一致度量），S‑Score 加权融合，计算 ECE、AUROC 等指标，以及基于四配置的消融实验。

**📊 数据集**

使用了 MedQA‑USMLE 与 MedMCQA 两个医学多选题基准，均筛选出高分歧子集（MedQA‑100、MedQA‑250、MedMCQA‑100、MedMCQA‑250），每个子集只涉及呼吸、心脏、神经、胃肠四个专业。

**📈 对比分析**

对比四种配置（单代理基线、单代理+验证、多代理无验证、全系统），发现全系统在 MedQA 子集上准确率提升 4.8–7.0pp，ECE 降低 49–74%（如 MedQA‑250 由 0.355 降至 0.091），AUROC 最高提升 0.056；在 MedMCQA 上虽然准确率受限，但校准仍显著改善。

**⚠️ 局限性**

局限性包括：一致性不等同于事实正确，验证无法发现内部自洽但错误的答案；7B 模型知识匮乏，特别是 MedMCQA 的高记忆需求；计算成本高（约 16 倍 LLM 调用），并且仅覆盖四个专业，难以推广到更广泛医学领域。

---

## 435. Positive-First Most Ambiguous: A Simple Active Learning Criterion for Interactive Retrieval of Rare Categories

**arXiv ID:** 2603.24480 | [PDF](https://arxiv.org/pdf/2603.24480v1)

**作者:** Kawtar Zaher `[一作]` (INRIA), Alexis Joly `[通讯]` (Institut National de l'Audiovisuel)

**通讯引用:** 5175 | [OpenAlex ID](https://openalex.org/A5015501462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 Positive-First Most Ambiguous (PF-MA) 主动学习策略，用于在严重类别不平衡的交互式细粒度视觉检索中快速发现稀有概念。

**💡 创新点**

通过将信息量最大的样本与高置信度正样本优先权结合，显著提升早期检索覆盖率和正样本比例，并引入基于 K-means 的类覆盖度指标衡量检索多样性。

**🔧 技术方法**

采用轻量级线性 SVM 分类器，使用预训练 ViT-L14（CLIP 与 DINOv2）视觉特征，结合 PF-MA 评分公式与类覆盖度度量。

**📊 数据集**

在长尾细粒度数据集 Cifar100-LT、ImageNet-LT、PlantNet300K 上进行实验。

**📈 对比分析**

与随机、MA、MP、DAL、CoreSet、ALAMP 及其多样性扩展等基线对比，PF-MA 在所有数据集与特征上均取得最高覆盖率、最高正样本比例、最优 F1，尤其在前几轮迭代表现最为突出。

**⚠️ 局限性**

仍存在极端细粒度场景下负样本稠密导致难以获得足够负样本、仅使用图像特征未考虑文本提示、以及在极大类别规模时正样本比例偏低等限制，后续工作需探索多模态扩展和更大规模评估。

---

## 436. Conformalized Transfer Learning for Li-ion Battery State of Health Forecasting under Manufacturing and Usage Variability

**arXiv ID:** 2603.24475 | [PDF](https://arxiv.org/pdf/2603.24475v1)

**作者:** Samuel Filgueira da Silva `[一作]` (Ohio State University), Marcello Canova `[通讯]` (Ohio State University)

**通讯引用:** 3725 | [OpenAlex ID](https://openalex.org/A5075875954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于LSTM、最大均值差(MMD)域适配和合规预测(CP)的无偏差SOH预测框架，解决了制造与使用差异导致的预测偏移；

**💡 创新点**

创新点在于将域适配与合规预测相结合，既提升了跨细胞的泛化能力，又提供了分布无关、校准良好的不确定性区间；

**🔧 技术方法**

技术包括LSTM时序建模、MMD域对齐、Leave-One-Batch-Out交叉验证调参、合规预测构造置信区间；

**📊 数据集**

使用了在PyBaMM环境下生成的NMC+石墨电池虚拟数据集，涵盖不同电极活性材料比例与多种充放电速率；

**📈 对比分析**

与仅基于源域训练、微调与无域适配的对比实验表明，TL+MMD方法将目标域RMSE从1.637%降低到0.781%，R²提升至0.962，且合规预测区间的经验覆盖率达98.8%；

**⚠️ 局限性**

局限性包括依赖高质量合成数据、对真实测量噪声与更复杂工况的鲁棒性待验证，以及CP区间宽度受源域数据分布影响。

---

## 437. Hybrid Spatiotemporal Logic for Automotive Applications: Modeling and Model-Checking

**arXiv ID:** 2603.24443 | [PDF](https://arxiv.org/pdf/2603.24443v1)

**作者:** Radu-Florin Tulcan `[一作]` (TU Wien), Ichiro Hasuo `[通讯]` (National Institute of Informatics)

**通讯引用:** 1822 | [OpenAlex ID](https://openalex.org/A5013382452)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出一种面向高速公路驾驶的混合时空逻辑HSTL，并给出基于该逻辑的模型检测算法；

**💡 创新点**

通过将混合逻辑、时空逻辑与空间网格相结合，既保持离散可搜索性，又能精确表达车辆的相对运动，从而在模型检测上实现指数级性能提升；

**🔧 技术方法**

采用离散网格图结构、混合逻辑名词绑定、时空模态、基于动态规划的公式求值以及三种模型检测算法（基线、优化、运动）；

**📊 数据集**

使用人工构造的测试案例（跟车、交叉、变道、车队）在不同网格尺寸与轨迹长度下进行实验；

**📈 对比分析**

与基线模型检测器相比，优化和运动算法在可搜索状态数上大幅减少，实验数据显示在大多数场景下可获得数十倍甚至数百倍的速度提升；

**⚠️ 局限性**

仅适用于离散网格，缺乏对连续动力学的建模；对原子命题的处理效率低，导致某些测试难以收敛；未在真实道路数据上验证。

---

## 438. Enes Causal Discovery

**arXiv ID:** 2603.24436 | [PDF](https://arxiv.org/pdf/2603.24436v1)

**作者:** Alexis Kafantaris `[一作]` `[通讯]`, Alexis Kafantaris

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用混合专家神经网络Enes对图中节点间因果关系进行三元组分类，进而实现因果结构发现。

**💡 创新点**

创新点在于将物理约束（DAG、Pearson相关、余弦相似度）嵌入Mixture-of-Experts框架中，利用随机生成的SEM数据训练模型，从而在无真实标注的观测数据上实现可泛化的因果推断。

**🔧 技术方法**

技术手段包括：混合专家网络（线性与非线性专家+门控机制）、共享特征提取、梯度稳定化（Langevin动力学+模拟退火）、多目标损失与约束正则化。

**📊 数据集**

主要使用了Sachs蛋白质网络真实数据，以及通过Michaelis–Menten动力学生成的3个规模（11、25、50节点）的合成数据。

**📈 对比分析**

与线性Pearson系数基线、PC算法等传统方法比较。Enes在准确率、精确率和SHD上普遍优于基线，尤其在Sachs数据上SHD显著降低；但召回率相对较低，说明模型倾向于挑选更可信的边。

**⚠️ 局限性**

局限性包括：召回率不足；模型对训练数据的依赖导致“垃圾进垃圾出”问题；仅在观测数据上验证，缺乏干预数据的评估；在更大规模的MM动力学数据上表现相对欠佳。

---

## 439. Analysing the Safety Pitfalls of Steering Vectors

**arXiv ID:** 2603.24543 | [PDF](https://arxiv.org/pdf/2603.24543v1)

**作者:** Yuxiao Li `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 14438 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对激活向量（通过Contrastive Activation Addition）进行系统的安全审计，评估其对LLM在JailbreakBench上的攻击成功率和错误拒绝率的影响。

**💡 创新点**

揭示激活向量的安全缺陷来源于其与模型拒绝方向的几何重叠，并首次证明通过去除拒绝方向分量可部分缓解安全风险，提出控制性与安全性之间的权衡。

**🔧 技术方法**

使用Contrastive Activation Addition (CAA) 生成向量，计算模型拒绝方向，测量余弦相似度与攻击成功率的相关性，并在向量上进行拒绝方向正交化消融。

**📊 数据集**

采用JailbreakBench作为攻击评估数据集，并使用包含有害与良好指令的训练集构造拒绝方向。

**📈 对比分析**

在六个3B-32B大小的开源LLM上进行对照实验，比较未消融与消融后的攻击成功率、误拒绝率，发现消融可降低15-25%的ASR变化，但未完全恢复基线。

**⚠️ 局限性**

局限于单维拒绝方向近似，无法完全捕捉多维安全子空间，且仅评估CAA方法，对其他激活基调控方式的普适性待验证。

---

## 440. Relaxing Constraints in Anonymous Multi Agent Path Finding for Large Agents

**arXiv ID:** 2603.24442 | [PDF](https://arxiv.org/pdf/2603.24442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 441. Efficiency for Experts, Visibility for Newcomers: A Case Study of Label-Code Alignment in Kubernetes

**arXiv ID:** 2603.24501 | [PDF](https://arxiv.org/pdf/2603.24501v1)

**作者:** Matteo Vaccargiu `[一作]` (University of Cagliari), Giuseppe Destefanis `[通讯]` (University College London)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5036425614)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Kubernetes PR 中标签（area）与代码变更文件的匹配度（label‑diff congruence），并探讨其在大型开源项目中的普遍性、稳定性、可解释性以及对代码评审速度和讨论活跃度的影响。

**💡 创新点**

提出了“label‑diff congruence”这一新的协同质量指标，并证明其在大规模、结构化项目中能被贡献者主动维护；同时揭示了该指标对不同经验层次贡献者的讨论模式产生的双向影响。

**🔧 技术方法**

采用了文本挖掘、词元化与正则表达式检测标签一致性、使用 PMI 与 Cramér’s V 评估标签与文件路径的语义关联；统计建模方面包括量化回归（quantile regression）评估合并时间，负二项回归（negative binomial）评估评论与参与者数量，逻辑回归检验标签修改行为。

**📊 数据集**

使用来自 Kubernetes 官方 GitHub 仓库的完整 PR、提交、文件差异与评论数据，共 18,020 条可测 PR（原始 83,368 条），覆盖 2014‑2025 年 9 年时间段；数据集已公开并附带完整分析流水线。

**📈 对比分析**

通过与缺失标签、不同经验层次、不同 PR 大小的控制变量对比，检验了标签一致性对合并速度（无显著效应）和讨论活跃度（评论数下降 4.3%，参与者数下降 14.4%）的影响；对各经验层次分别建模，揭示核心贡献者与新人之间的效应逆转。

**⚠️ 局限性**

局限包括：构造有效性仅基于词元匹配，可能忽略语义匹配；样本仅覆盖可获得 diff 的 PR，早期年份 diff 覆盖不足；外部有效性受限于 Kubernetes 这类高治理、标签规范严谨的项目；以及对小样本新人子组的结果为探索性，置信区间宽。

---

## 442. TuneShift-KD: Knowledge Distillation and Transfer for Fine-tuned Models

**arXiv ID:** 2603.24518 | [PDF](https://arxiv.org/pdf/2603.24518v1)

**作者:** Yushi Guan `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**通讯引用:** 1954 | [OpenAlex ID](https://openalex.org/A5080873211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为TuneShift-KD的方法，用于将细调模型中的专业知识自动蒸馏并转移到目标模型，使用少量代表性示例进行训练。

**💡 创新点**

创新点在于通过基于困惑度的差异来识别专业知识，并提出了一种无需访问原始训练数据的自动化知识转移方法。

**🔧 技术方法**

使用了知识蒸馏（KD）技术，结合了困惑度差异过滤标准和迭代合成数据生成过程。

**📊 数据集**

在多个基准数据集上进行了实验，包括GSM8K、MBPP和BBH，展示了方法的有效性。

**📈 对比分析**

与Trans-LoRA方法进行了比较，TuneShift-KD在不同模型架构上均表现出更高的准确性，且无需训练判别器或访问原始训练数据。

**⚠️ 局限性**

限制在于在某些情况下，源基础模型的缺失可能会影响性能，尽管可以使用通用预训练模型作为替代。

---

## 443. Counting Without Numbers \& Finding Without Words

**arXiv ID:** 2603.24470 | [PDF](https://arxiv.org/pdf/2603.24470v1)

**作者:** Badri Narayana Patro `[一作]` `[通讯]` (Microsoft), Badri Narayana Patro (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态宠物缺失识别系统，融合视觉、声学与上下文信息实现跨物种的重识别。

**💡 创新点**

首次将动物声学身份信号与近似认知机制（非符号数值系统）结合，并采用软视觉匹配与时间衰减建模，实现跨模态不对齐的匹配。

**🔧 技术方法**

使用CNN/Transformer提取视觉与声学特征、Gaussian嵌入软匹配、跨模态注意力融合以及时间衰减预测等技术。

**📊 数据集**

在合成的60个身份样本以及两家动物收容所收集的真实照片与录音数据上进行实验。

**📈 对比分析**

相较于传统视觉仅匹配基线，Rank‑1准确率提升约25.7%，误检率下降30%；在真实收容所的23个模糊案例中，成功率提升至61%。

**⚠️ 局限性**

声学模型在嘈杂环境下鲁棒性不足，未涵盖嗅觉等其他生物信号；时间衰减模型为线性假设；跨物种泛化受限；实验规模仍较小，缺乏大规模实测。

---

## 444. Project and Generate: Divergence-Free Neural Operators for Incompressible Flows

**arXiv ID:** 2603.24500 | [PDF](https://arxiv.org/pdf/2603.24500v1)

**作者:** Xigui Li `[一作]` (Fudan University), Yuan Cheng `[通讯]` (Fudan University)

**通讯引用:** 8463 | [OpenAlex ID](https://openalex.org/A5058272109)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一框架，将不可压流体的无散度约束硬性嵌入到神经算子中，实现预测与生成模型的物理一致性

**💡 创新点**

利用可微分谱Leray投影直接将模型输出投射到无散度子空间，并构造基于旋度的高斯参考分布，保证生成过程在物理子空间内完整进行

**🔧 技术方法**

谱Leray投影、Helmholtz–Hodge分解、流形约束神经算子、流动匹配（flow matching）生成模型、卷积/傅里叶神经算子（FNO）

**📊 数据集**

2D周期性Navier–Stokes流场数据集（Re=1000），包含10,000条训练轨迹（64×64，50步）

**📈 对比分析**

与无约束的FNO回归和生成基线对比，测量MSE、散度误差、能谱和长时稳定性；实验显示我们的方法在散度误差上接近机器精度，MSE显著降低，长时推演不出现数值发散

**⚠️ 局限性**

仅针对周期域，需在复杂边界条件或3D流动下进一步验证；理论上对无限维流动匹配的绝对连续性假设仍未严格证明

---

## 445. Fault-Tolerant Distance Oracles Below the $n \cdot f$ Barrier

**arXiv ID:** 2603.24530 | [PDF](https://arxiv.org/pdf/2603.24530v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Aaron Putterman `[通讯]` (Harvard University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5091954510)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在任意图中对边故障的距离报告（distance oracle）与spanner的新的压缩技术，首次实现了在“边故障数 * 节点数”这一经典下界之外的空间下的可行解。

**💡 创新点**

创新点包括：
• 通过高度可恢复的高度稠密图（高度扩散子图）和“2‑跳星”结构，构造了在最多 f 条边失效时仍能保持短路径的距离报告器；
• 结合确定性稀疏恢复与随机 ℓ0 采样，实现了无子图约束下的子平方级压缩；
• 将上述技术推广到可观测（oblivious）剪枝和流式（streaming）设置，给出了相应的压缩 sketch 与算法；
• 在 n ≤ f ≤ n^{3/2} 的范围内，获得了常数 7 的拉伸，空间仅为 O(n^{3/2} f^{1/3}) 位。

**🔧 技术方法**

核心技术包括：
1. 高度扩散子图（high‑degree expander）分解与鲁棒性分析；
2. 确定性稀疏恢复（deterministic sparse recovery） sketch，用于在低度顶点上完全恢复邻接；
3. 随机 ℓ0 采样与线性 sketch，用于构造可随机恢复的 spanner；
4. “2‑跳星” (2‑hop star) 结构，用以在高度稠密子图中提供常数阶的直径；
5. 递归 expander 分解与多层构造，以控制总空间。

**📊 数据集**

本研究为理论性论文，未在实际数据集上进行实验；所有结果均为理论上可实现的空间和拉伸分析。

**📈 对比分析**

与传统的 n·f 下界相比：
• 对于任意 f，提供 O(n √f) 位的 deterministic 距离报告器，拉伸为 O(log n log log n)。
• 对于 n ≤ f ≤ n^{3/2}，提供 O(n^{3/2} f^{1/3}) 位的 deterministic 距离报告器，拉伸常数 7。 
• 对于可观测剪枝与流式环境，给出了 O(n √f) 位 sketch 与 O(n^{4/3} f^{1/3}) 位流式算法。 
这些结果在理论上显著突破了以往关于 fault‑tolerant spanner 的空间下界。

**⚠️ 局限性**

局限与未解决问题：
• 对于更一般的 f（尤其是 f = Ω(n^2)）的最优空间下界仍未知；
• 对于随机拉伸（如 O(log n) 以内）的最佳空间仍未确定；
• 仅考虑边故障，顶点故障的情况需要进一步研究；
• 本文给出的构造主要是存在性证明，虽然已给出多项式时间实现，但实际常数与实现复杂度仍需进一步优化。

---

## 446. Cross-Modal Prototype Alignment and Mixing for Training-Free Few-Shot Classification

**arXiv ID:** 2603.24528 | [PDF](https://arxiv.org/pdf/2603.24528v1)

**作者:** Dipam Goswami `[一作]` (Universitat Autonoma De Barcelona), Joost van de Weijer `[通讯]` (Universitat Autonoma De Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练无关的少样本分类方法，通过将图像原型投影到文本原型的主子空间并与文本原型混合，再结合图像特定的LDA分类器，显著提升了CLIP在多种少样本任务中的表现。

**💡 创新点**

创新点在于：①以偏差-方差分析为依据，将图像原型与文本原型的混合视作收缩估计；②设计文本对齐的语义投影，仅在对齐子空间内混合，避免噪声误差；③当跨模态对齐弱时，补充图像空间的LDA分类器，形成双模态融合。

**🔧 技术方法**

使用的技术包括CLIP预训练的视觉‑文本编码器、奇异值分解投影、混合原型估计、线性判别分析（LDA）以及最近邻原型分类器。

**📊 数据集**

实验数据集涵盖了11个公开少样本基准，包括ImageNet、Caltech101、FGVC Aircrafts、Stanford Cars、Oxford Pets、Flowers102、Food101、EuroSAT、Describable Textures、SUN397和UCF101。

**📈 对比分析**

与现有的训练无关方法（Zero‑Shot CLIP、CALIP、Tip‑Adapter、Tip‑X、GDA）以及训练相关方法（MaPLe、CLIP‑LoRA）相比，本文方法在所有Shot设定下均实现了显著提升（平均提升约3‑5%），并能与训练后模型无缝叠加进一步提升。

**⚠️ 局限性**

局限性包括：①对跨模态对齐差异较大的数据集（如EuroSAT）效果有限；②在极低样本场景下LDA协方差估计不稳；③需要验证集来调节混合权重和集成权重。

---

## 447. Integrating Causal Machine Learning into Clinical Decision Support Systems: Insights from Literature and Practice

**arXiv ID:** 2603.24448 | [PDF](https://arxiv.org/pdf/2603.24448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 448. Claudini: Autoresearch Discovers State-of-the-Art Adversarial Attack Algorithms for LLMs

**arXiv ID:** 2603.24511 | [PDF](https://arxiv.org/pdf/2603.24511v1)

**作者:** Alexander Panfilov `[一作]` (MATS), Maksym Andriushchenko `[通讯]` (ELLIS Institute Tübingen & Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用Claude Code自动迭代设计并实现白盒攻击算法，提升LLM的越狱和提示注入性能。

**💡 创新点**

通过LLM自主搜索算法空间，发现高效攻击策略，显著超过现有30+方法。

**🔧 技术方法**

使用Claude Code自动化编程、Optuna超参数优化、离散优化技巧（如GCG、MAC、TAO等）及自定义迭代。

**📊 数据集**

ClearHarm CBRN查询、Qwen-2.5-7B/Llama-2-7B/Gemma-7B随机目标、Meta-SecAlign-70B/8B、AlpacaFarm指令等数据集。

**📈 对比分析**

对比30+基线和Optuna调参结果，Claude设计的算法在GPT-OSS-Safeguard-20B的ASR提升至40%，在Meta-SecAlign-70B实现100%攻击成功率，整体性能优于传统方法。

**⚠️ 局限性**

缺乏真正的算法创新，主要靠组合已有技巧，且受限于实验框架和评估协议，可能出现奖励作弊。

---

## 449. Nominal Automata with Name Deallocation

**arXiv ID:** 2603.24468 | [PDF](https://arxiv.org/pdf/2603.24468v1)

**作者:** Simon Prucker `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Lutz Schröder `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 2364 | [OpenAlex ID](https://openalex.org/A5073765060)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究并定义了可处理命名绑定与释放的正则表达式与对应的自动机模型（NDA 与 D‑NFA），并证明它们之间的 Kleene 定理

**💡 创新点**

提出了正则 deallocation 表达式、D‑NFA 与 NDA 的等价性，并给出了 α‑equivalence 闭包技术与 S‑restriction 方法

**🔧 技术方法**

使用了命名绑定的抽象与 α‑equivalence、Nominal Set 理论、支持引理、可决定的正则表达式判定以及传统 NFA 的 Kleene 构造

**📊 数据集**

无数据集

**📈 对比分析**

通过理论证明进行比较，未进行实验，性能评价不存在

**⚠️ 局限性**

局限在仅支持右侧非阴影词的约束，未涵盖所有命名重命名情况，并且仅在理论框架内实现

---

## 450. Design, Modelling and Characterisation of a Miniature Fibre-Reinforced Soft Bending Actuator for Endoluminal Interventions

**arXiv ID:** 2603.24461 | [PDF](https://arxiv.org/pdf/2603.24461v1)

**作者:** Xiangyi Tan `[一作]` (University College London), Agostino Stilli `[通讯]` (University College London)

**通讯引用:** 1228 | [OpenAlex ID](https://openalex.org/A5033887596)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

开发了一种100转单螺旋Kevlar纤维增强的微型软气压致弯曲执行器，用于自然口腔内介入机器人的尺寸约18 mm×37.5 mm。

**💡 创新点**

创新点在于将纤维增强技术缩小至厘米级，并提出双腔结构（Geometry B）与高精度Abaqus有限元仿真相结合，以实现高弯曲角度与结构稳定性。

**🔧 技术方法**

采用多材料硅胶浇铸、Kevlar纤维缠绕、嵌入不伸长玻璃纤维层、以及Abaqus FEM对弹性和纤维-基体耦合的数值模拟。

**📊 数据集**

未使用公开数据集，而是通过实验测得的弯曲角度–压力曲线（实验角度101.4° vs 仿真148.8°）作为验证数据。

**📈 对比分析**

通过将仿真预测与实验测得的弯曲角度以及与文献中相同压力下的弯曲角度进行对比，显示本设计在尺寸范围内达到或超过同类器件（实验α≈202.9°，仿真α≈297.6°）。

**⚠️ 局限性**

主要限制包括纤维缠绕接合与密封易失效、仿真与实际材料参数差异导致性能偏差，以及双腔结构尚未实现成熟工艺可行性。

---

## 451. The Gait Signature of Frailty: Transfer Learning based Deep Gait Models for Scalable Frailty Assessment

**arXiv ID:** 2603.24434 | [PDF](https://arxiv.org/pdf/2603.24434v1)

**作者:** Laura McDaniel `[一作]` (Johns Hopkins University), Rama Chellappa `[通讯]` (Johns Hopkins University)

**通讯引用:** 65050 | [OpenAlex ID](https://openalex.org/A5102762707)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了公开的临床真实环境下全范围脆弱性步态数据集，并利用迁移学习从预训练步态识别模型中提取特征进行脆弱性分类

**💡 创新点**

首次将大规模预训练步态识别模型迁移到脆弱性评估任务，系统评估了不同冻结策略和类别加权对性能的影响

**🔧 技术方法**

使用SwinGait（混合卷积‑Transformer）和DeepGaitV2（深度卷积）两种预训练网络，结合交叉熵+三元组损失与可逆类别权重

**📊 数据集**

采用68名老年人（25非脆弱、24前脆弱、17脆弱）的轮廓化步态序列，全部来自单中心临床数据

**📈 对比分析**

通过五折参与者层交叉验证对比不同冻结策略，最佳配置在SwinGait M2与DeepGaitV2 D1下微AUC≈0.78、Kappa≈0.62（SwinGait）和≈0.53（DeepGaitV2），表明迁移学习显著提升性能

**⚠️ 局限性**

样本量小、仅单中心、使用轮廓化而非完整姿态/传感器数据，导致对不同人群和环境的泛化性及细粒度脆弱性识别仍受限

---

## 452. Representation Learning to Study Temporal Dynamics in Tutorial Scaffolding

**arXiv ID:** 2603.24535 | [PDF](https://arxiv.org/pdf/2603.24535v1)

**作者:** Conrad Borchers `[一作]` (Carnegie Mellon University), Ashish Gurung `[通讯]` (Carnegie Mellon University)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5019516308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并验证了一种基于句子嵌入的连续语义对齐方法，用来量化教学对话中的支架化过程，并利用此方法分析真实聊天式数学辅导对话。

**💡 创新点**

创新点在于：①将支架化视为对任务（问题与正确解答）语义对齐的连续属性；②用预训练语言模型生成高维句子嵌入，以度量辅导者和学习者发言与任务内容的相似度；③结合时间序列与混合效应回归，揭示角色特定的对齐动态与进度关系。

**🔧 技术方法**

技术方法包括：Transformer‑based Sentence‑Transformers 嵌入模型、余弦相似度计算、时间序列平滑、线性混合效应回归（logit‑变换的对话进度）以及基准模型比较（BIC、似然比检验）。

**📊 数据集**

使用公开的 Eedi‑2K “Question‑Anchored Tutoring Dialogues” 数据集，该数据集包含 1,576 场完整数学辅导对话，共 55,322 条聊天记录。

**📈 对比分析**

与仅用消息序号和长度的基准模型相比，加入对问题和解答的语义相似度后，混合效应模型在对话进度预测上显著提升（χ²(2)=28.29, p<.001，BIC下降），并在角色维度上进一步改善（χ²(2)=32.43, p<.001）。

**⚠️ 局限性**

局限性包括：缺乏学习者身份信息，无法进行纵向支架化或学习效果验证；仅使用问题与解答作为语义锚点，可能忽略元认知等其它支架维度；未对 LLM 辅导系统进行系统评估。

---

## 453. Optimal Multidimensional Convolutional Codes

**arXiv ID:** 2603.24546 | [PDF](https://arxiv.org/pdf/2603.24546v1)

**作者:** Z. Abreu `[一作]`, R. Simoes `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种基于超正则矩阵的新方法构造多维(mD)最大距离分离(MDS)卷积码。

**💡 创新点**

创新点在于将超正则性推广到多变量情形，给出严格的 Singleton 上界，并在此基础上构造了新的 MDS 码族，尤其在率为 1/n 时提供了更宽广的参数范围。

**🔧 技术方法**

主要技术包括多维卷积码的理论框架、超正则矩阵的定义与性质、以及对自由距离的计数和上界推导。

**📊 数据集**

没有使用实际数据集，研究完全基于理论与符号运算。

**📈 对比分析**

通过与 Singleton 上界的比对验证构造的 MDS 性能；在给定参数下，所得到码的自由距离达到上界，说明性能最优。

**⚠️ 局限性**

局限性在于仅对率为 1/n 的情况给出完整构造，k/n 的一般情况仍受限于特定的行度条件；此外对有限域大小的要求较高，实际实现可能受限。

---

## 454. SEGAR: Selective Enhancement for Generative Augmented Reality

**arXiv ID:** 2603.24541 | [PDF](https://arxiv.org/pdf/2603.24541v1)

**作者:** Fanjun Bu `[一作]` (Cornell University), Hiroshi Yasuda `[通讯]` (Toyota Research Institute)

**通讯引用:** 4232 | [OpenAlex ID](https://openalex.org/A5067275429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SEGAR 两阶段框架：先使用 diffusion 世界模型生成带区域性编辑的未来帧，再通过选择性修正阶段对安全关键区域进行实时校正，保持增强效果不变。

**💡 创新点**

创新点在于将生成式世界模型与轻量级 LoRA 修正相结合，实现对生成内容的安全关键区域实时校正，同时保持预先编辑的增强效果；同时提出了空间掩码损失与缓冲区设计，避免冲突区域产生不一致。

**🔧 技术方法**

采用 Vista diffusion 世界模型、Stable Video Diffusion (SVD) 作为基底，利用 VACE 进行视觉风格化，使用 LoRA 在 U‑Net 空间注意力层进行微调，并结合 SegFormer 进行语义分割。

**📊 数据集**

训练数据来源于 nuScenes 数据集，使用 VACE 在此基础上生成东京风格的增强目标帧。

**📈 对比分析**

通过对安全关键区和增强区分别计算 SSIM 与 LPIPS 与真实图像/增强图像比较，实验显示安全关键区 SSIM 提升至 0.943，LPIPS 降至 0.285，增强区保持 SSIM 0.866 与 LPIPS 0.130，表明校正有效且不破坏增强效果。

**⚠️ 局限性**

局限性包括：缺乏对象级别理解导致跨区域对象不一致；缓冲区无监督导致边界闪烁；无法自回归生成长序列；只能单一风格，需重新训练以换风格。

---

## 455. CliPPER: Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures for Event Recognition

**arXiv ID:** 2603.24539 | [PDF](https://arxiv.org/pdf/2603.24539v1)

**作者:** Florian Stilz `[一作]` (University of Strasbourg), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CliPPER，一种面向长时段外科手术视频-文本预训练框架，专注于捕捉手术流程的上下文与细粒度对齐。

**💡 创新点**

创新点包括四个新预训练目标：上下文感知视频-文本对比学习（VTC_CTX）、循环一致性对齐（Cycle‑Consistency）、剪辑顺序预测（COP）以及帧‑文本匹配（FTM），以及引入跨剪辑上下文编码器来增强长时序建模。

**🔧 技术方法**

采用基于 BEiT 的时间注意力视频编码器、BERT 基础文本编码器，配合多模态融合的多头注意力层；所有目标通过对比、分类与二分类损失联合训练。

**📊 数据集**

训练数据来自公开的 YouTube 外科教学视频（约422小时）和 SVL 讲座视频；下游评测使用 Cholec80、AutoLaparo、MultiBypass140、GraSP（阶段识别）以及 CholecT50、ProstaTD（三元组与工具识别）等公共数据集。

**📈 对比分析**

在零样本外科流程识别、步骤识别、三元组和工具识别任务上，CliPPER 超越了目前最强的基线（如 SurgVLP、HecVL、PeskaVLP、VindLU），平均提升约 14%–18% 的 F1 分数和 2.6%–6.5% 的 mAP。

**⚠️ 局限性**

局限性包括：① 仍依赖于手工生成或 LLM 修正的字幕，可能带来语义误差；② 主要针对公开教育视频，缺乏对真实临床录像的验证；③ 计算成本较高，训练需多 GPU；④ 对极短或高频切换的手术场景表现尚待进一步验证。

---

## 456. UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience

**arXiv ID:** 2603.24533 | [PDF](https://arxiv.org/pdf/2603.24533v1)

**作者:** Zichuan Lin `[一作]` (Tencent Hunyuan), Jie Jiang `[通讯]` (Tencent Hunyuan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于两阶段自进化的移动 GUI 代理框架，分别是拒绝式微调（RFT）和组相对自蒸馏（GRSD）；

**💡 创新点**

创新点在于通过 RFT 实现数据与模型的自动共演进，以及 GRSD 通过分叉点检测在失败轨迹中提取精细步骤级监督，解决了稀疏奖励下的信用分配问题；

**🔧 技术方法**

使用了多模态大型语言模型 Qwen3-VL-4B-Instruct 作为基础模型，结合 SSIM 匹配、结构相似性索引、规则基验证器以及自蒸馏训练目标；

**📊 数据集**

实验数据集为 AndroidWorld，包含 116 个真实手机应用中的多样任务；

**📈 对比分析**

与多种基线（Qwen、UI-Tars、Step-GUI、MAI-UI、Gemini 等）对比，4B 模型 Pass@1 成功率达 81.0%，超过人类 80.0% 及所有更大规模模型；

**⚠️ 局限性**

局限包括对 SSIM 匹配对实时异步行为的鲁棒性不足、动作空间过于离散导致对低级触控细节不敏感，以及未验证在更广泛 GUI 任务和真实部署场景中的迁移性能。

---

## 457. Novel models of computation from novel physical substrates: a bosonic example

**arXiv ID:** 2603.24531 | [PDF](https://arxiv.org/pdf/2603.24531v1)

**作者:** Sampreet Kalita `[一作]` (University of Strathclyde), Viv Kendon `[通讯]` (University of Strathclyde)

**通讯引用:** 3865 | [OpenAlex ID](https://openalex.org/A5025637911)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

从物理设备的实际行为出发，提出一种“自底向上”推导计算模型和域特定语言（DSL）的完整方法论，并以多组件光学干涉器中的玻色子采样为案例，完成模型建模、仿真、元模型抽象、DSL定义、参考实现及初步仿真验证。

**💡 创新点**

创新点主要包括：① 用物理设备的功能特征作为起点逆向推导计算模型，而非传统的自顶向下映射；② 将软件工程中的四层元建模架构（M0‑M3）应用于物理计算，形成完整的层级关系；③ 在玻色子干涉器案例中首次给出从物理建模到内部DSL（PyBos）的完整闭环，并演示如何在仿真框架和硬件框架中切换。

**🔧 技术方法**

技术手段：物理建模（使用光学干涉器的散射矩阵与损耗模型）；Python内部DSL（PyBos）；仿真框架 PyBos‑Sampler（利用稀疏矩阵、JAX 并行加速）；硬件支持框架 PyBos‑Runner；符号/数值语义定义（静态与动态语义）；多层元建模（M0‑M3）。

**📊 数据集**

由于尚未完成硬件实验，主要使用的“数据集”为模拟产生的光子分布和概率质量函数（PMF），并可参考已知的 Boson Sampling 任务输出（如特定矩阵的行列式或行列式近似）。

**📈 对比分析**

比较方法：在仿真器中执行 DSL 程序，得到 PMF；与理论 PMF 或已有 Boson Sampling 参考实现（如经典模拟或公开基准）进行概率分布相似度比较；性能评估通过仿真运行时间、内存占用以及可扩展性（最大支持模式数、光子数）。目前仅在小规模（数十个模式、光子）下验证，性能随系统规模呈指数级增长，需进一步优化。

**⚠️ 局限性**

限制与挑战：① 模型为近似，尚未通过硬件实验验证；② 对损耗的处理仍是启发式模型，缺乏严格的物理描述；③ DSL 仅覆盖低层硬件抽象，缺乏针对具体算法的高级抽象；④ 缺乏大规模基准和数据集；⑤ 需要进一步完善语义定义与编译器实现，以确保在真实硬件上无误差运行。

---

## 458. AVO: Agentic Variation Operators for Autonomous Evolutionary Search

**arXiv ID:** 2603.24517 | [PDF](https://arxiv.org/pdf/2603.24517v1)

**作者:** Terry Chen `[一作]` (NVIDIA), Humphrey Shi `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用Agentic Variation Operators (AVO) 实现了七天无人工干预的自主演化，生成了在NVIDIA Blackwell B200 GPU上多头注意力（MHA）和分组查询注意力（GQA）核，性能超过cuDNN和FlashAttention-4。

**💡 创新点**

创新点在于把LLM从单一候选生成器升级为完整的变异操作员，让模型通过规划、工具调用、持久记忆和自我评估实现持续自适应迭代；同时加入自我监督机制在进展停滞时重新引导搜索。

**🔧 技术方法**

技术包括前沿LLM驱动的编码代理、持续的自动化代码编辑与测试循环、GPU评估函数（正确性+吞吐量）、CUDA 13.1/PyTorch 2.10.0、Blackwell架构下的寄存器/流水线/内存优化探索。

**📊 数据集**

使用的基准配置为多头注意力（头维128、BF16、序列长度4k–32k、16头）以及Qwen3模型的两种分组查询注意力（32查询/4或8 KV头），并在固定总token数32k下进行吞吐量评测。

**📈 对比分析**

通过与cuDNN 9.19.1和FlashAttention-4的TFLOPS吞吐量对比，AVO在MHA上最高提升10.5%（相较FA4）和3.5%（相较cuDNN），在GQA上提升至9.3%和7.0%，体现了显著的性能优势。

**⚠️ 局限性**

局限性包括：仅在单线性演化（无族群分支）环境下验证；主要针对注意力核，缺乏对其他算法或硬件的通用性评估；性能提升主要体现在吞吐量，未系统检验准确性、能耗或多GPU扩展等方面。

---

## 459. Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling

**arXiv ID:** 2603.24503 | [PDF](https://arxiv.org/pdf/2603.24503v1)

**作者:** Mihaela-Larisa Clement `[一作]` (TU Wien), Ezio Bartocci `[通讯]` (TU Wien)

**通讯引用:** 4389 | [OpenAlex ID](https://openalex.org/A5050836932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Sequential-AMPC网络，利用递归RNN在预测期内共享参数生成MPC候选控制序列，并在Safe Seq-AMPC包装器下实现在线安全评估与fallback。

**💡 创新点**

创新点在于把传统一次性全程向量预测改为共享参数的递归生成，显著降低参数随预测期增长的膨胀，同时引入安全评估与fallback机制保持闭环安全。

**🔧 技术方法**

采用监督学习训练RNN/MLP逼近NMPC策略，使用安全-增益包装器进行在线可行性与成本评估，并结合早停、参数共享等技术。

**📊 数据集**

使用三类仿真数据集：多架四旋翼约960万条NMPC轨迹；单轨车辆运动学模型约55k条；单轨车辆动力学模型约116k条。

**📈 对比分析**

与基准Feedforward AMPC相比，Seq-AMPC在所有基准中均提升开放环可行性（如四旋翼从72%提升至83%）和闭环安全率（如四旋翼从84.8%提升至89.1%），同时训练周期更短、参数更少。

**⚠️ 局限性**

局限在于安全包装器仍需高频 fallback，导致终端集满足率低和成本改进不足，终端集落位失败与成本提升率是主要瓶颈。

---

## 460. Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?

**arXiv ID:** 2603.24472 | [PDF](https://arxiv.org/pdf/2603.24472v1)

**作者:** Jeonghye Kim `[一作]` (Microsoft Research), Yuqing Yang `[通讯]` (Microsoft Research)

**通讯引用:** 2049 | [OpenAlex ID](https://openalex.org/A5101421201)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自蒸馏对大型语言模型数学推理的影响，发现过度压缩不确定性表达会导致性能下降。

**💡 创新点**

创新点在于将信息丰富度和任务覆盖率与自蒸馏中的 epistemic verbalization 关联，揭示自蒸馏对不确定性表达的压制会影响 OOD 性能。

**🔧 技术方法**

采用了自蒸馏（Self‑Distillation）、对抗强化学习（GRPO/SDPO）、信息论分析（条件互信息）和词汇统计等技术。

**📊 数据集**

使用了 DAPO‑Math‑17k、AIME24/25、AMC23、MATH500、ScienceQ&A Chemistry 等数学与科学推理数据集。

**📈 对比分析**

通过在 Qwen3‑8B、DeepSeek‑R1‑Distill‑Qwen‑7B 等模型上对比 GRPO 与 SDPO，发现当任务覆盖率低时 SDPO 可压缩回答长度并提升性能，但在覆盖率高或 OOD 评测时性能下降高达 40%。

**⚠️ 局限性**

局限在于仅关注文本级不确定性表达，缺乏对不同模型结构和多模态推理的泛化；且实验主要集中在特定算术/几何题型，未验证在更广泛数学领域的适用性。

---

## 461. OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning

**arXiv ID:** 2603.24458 | [PDF](https://arxiv.org/pdf/2603.24458v1)

**作者:** Kaihang Pan `[一作]` (Zhejiang University), Zhao Zhong `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OmniWeaving 统一视频生成框架，支持自由多模态（文本、图像、视频）混合输入的生成与编辑，并构建了 IntelligentVBench 评测基准。

**💡 创新点**

创新点包括：1) 通过 MLLM 与 MMDiT 的联合架构，实现主动推理与生成的“思考模式”与 DeepStacking 语义注入；2) 三阶段预训练+微调策略，兼顾多模态组合与推理；3) 大规模多模态训练数据与多任务数据构造；4) 使用 VLM-as-a-Judge 的自动评测方式，提供更全面的能力评估。

**🔧 技术方法**

使用技术包括：多模态大语言模型 Qwen2.5-VL、扩散 Transformer HunyuanVideo‑1.5、VAE 视觉分词、DeepStacking、思考模式、MuOn 优化器、稀疏注意力 SSTA、next‑token 预测损失以及多种工具模型（Qwen3、Gemini、SAM3、FLUX2 等）用于数据生成与质量过滤。

**📊 数据集**

数据集来源：真实视频（YouTube、电影、直播等）与合成视频（通过 HunyuanVideo、Veo3 等生成），结合公开数据集 OpenVE‑3M、Ditto、VBench、TGVE+ 等；并通过两条数据构造管线（output‑first 与 input‑first）生成涵盖文本、图像、视频、推理轨迹的多模态任务数据。

**📈 对比分析**

评估方法：在 IntelligentVBench（四个子任务）、VBench（T2V）和 OpenVE‑Bench（V2V）等基准上进行零样本对比，比较统一模型（VINO、UniVideo、VACE）与专用模型（CogVideoX、Wan2.1、HunyuanVideo 等）。结果显示 OmniWeaving 在所有子任务的 AVG 与 MIN 指标均超过基线，达到 SoTA；在 T2V、V2V 任务中表现与专用模型相当甚至更好。

**⚠️ 局限性**

局限性：1) 与闭源旗舰模型（如 Seedance‑2.0）仍存在明显性能差距；2) 当前仅支持文本、图像、视频三种模态，未覆盖音频等其他模态；3) 对部分细粒度添加任务（如动画对象添加）表现略弱；4) 训练成本高，需大量 GPU 资源。

---

## 462. Unleashing Vision-Language Semantics for Deepfake Video Detection

**arXiv ID:** 2603.24454 | [PDF](https://arxiv.org/pdf/2603.24454v1)

**作者:** Jiawen Zhu `[一作]` (Singapore Management University), Guansong Pang `[通讯]` (Singapore Management University)

**通讯引用:** 5894 | [OpenAlex ID](https://openalex.org/A5039104219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用视觉‑语言模型（VLM）跨模态语义提升深度伪造视频检测的框架；

**💡 创新点**

创新点包括：1）ForgePerceiver——独立学习器，生成多样细微伪造掩码与定位图，既保留CLIP预训练的视觉‑语言对齐，又捕获低级伪造痕迹；2）Identity‑Aware VLA Scoring——通过在文本提示中注入身份先验，强化局部语义对齐，获得更细粒度的伪造注意图；3）将全局与局部判别分数融合，显著提升跨数据集泛化。

**🔧 技术方法**

采用CLIP作为VLM骨干，构建ForgePerceiver（轻量级ViT + 查询标记），生成伪造掩码与定位图；利用文本编码器与视觉‑语言对齐生成VLA注意图；最终通过全局与局部分支的融合实现检测。

**📊 数据集**

在九个公开深度伪造基准上评测：FaceForensics++、CelebDF‑v1/v2、Deepfake Detection Challenge（DFDC）、DeepfakeDetection（DFD）、以及基于DF40的全脸生成数据（VQGAN、StyleGAN‑XL、SiT‑XL/2、DiT、PixArt）。

**📈 对比分析**

与16种SOTA方法（包括Xception、TALL、EfficientB4、SeeABLE、SPSL、CADDM、SAM等）对比，跨数据集框架在帧级和视频级AUROC均提升约2–3个百分点，尤其在大规模DFDC数据集上实现最高AUROC达99.7%。

**⚠️ 局限性**

局限性：对预训练VLM的依赖导致在极其逼真或少见的生成模型上仍有误检；查询标记数和融合权重需手动调参；模型相对较大，推理时仍有一定计算成本。

---

## 463. Robust Multilingual Text-to-Pictogram Mapping for Scalable Reading Rehabilitation

**arXiv ID:** 2603.24536 | [PDF](https://arxiv.org/pdf/2603.24536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 464. From Liar Paradox to Incongruent Sets: A Normal Form for Self-Reference

**arXiv ID:** 2603.24527 | [PDF](https://arxiv.org/pdf/2603.24527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 465. CUA-Suite: Massive Human-annotated Video Demonstrations for Computer-Use Agents

**arXiv ID:** 2603.24440 | [PDF](https://arxiv.org/pdf/2603.24440v1)

**作者:** Xiangru Jian `[一作]` (University of Waterloo), Sai Rajeswar `[通讯]` (ServiceNow)

**通讯引用:** 1233 | [OpenAlex ID](https://openalex.org/A5041629023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含10,000+任务、87款专业桌面应用、55小时30fps连续视频以及多层次推理与坐标轨迹的完整数据生态系统。

**💡 创新点**

创新点在于首次提供大规模连续视频与像素级密集注释，并将数据、基准与定位集成为统一生态，极大提升了桌面AI训练与评估的真实性与可复现性。

**🔧 技术方法**

采用人类专家演示、30fps屏幕录制、鼠标轨迹同步、基于Claude的多层推理生成以及手工精细标注等技术。

**📊 数据集**

使用了自研的55小时视频+10k任务数据集、56k截图+3.6M元素标注的GroundCUA，以及450任务的评估基准CUEBench。

**📈 对比分析**

通过与现有基准（OpenCUA、ScaleCUA 等）的对比，发现即便是规模最大的32B模型在定位任务中仅达37.7%@50px，人工评估步态准确率为57.6%，表明空间定位仍是瓶颈。

**⚠️ 局限性**

限制包括数据仅覆盖87款开源软件，部分专业领域缺乏足够多样化任务；当前基准主要评估坐标动作，未能覆盖全部交互类型；模型在复杂多面板界面上的误定位仍较多。

---

## 466. LensWalk: Agentic Video Understanding by Planning How You See in Videos

**arXiv ID:** 2603.24558 | [PDF](https://arxiv.org/pdf/2603.24558v1)

**作者:** Keliang Li `[一作]` (Institute Of Computing Technology Chinese Academy Of Sciences), Shiguang Shan `[通讯]` (Institute Of Computing Technology Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LensWalk 框架，允许大语言模型（LLM）在推理过程中主动规划并调度对视频的观测，实现了 reason‑plan‑observe 的循环；

**💡 创新点**

核心创新在于让推理模型动态控制观测范围和采样密度，并通过可组合的视觉工具（Scan Search、Segment Focus、Stitched Verify）与时间戳锚点和主体记忆相结合，突破了传统一次性预处理、固定观测的限制；

**🔧 技术方法**

技术实现基于 LLM 作为推理器、VLM 作为观察者，配合三种观察工具、时间戳锚点、主体记忆表，整个流程不需要任何模型微调；

**📊 数据集**

实验使用长视频基准数据集 LVBench、LongVideoBench、Video‑MME（长分割）、MMVU、Video‑MMMU、EgoSchema 等；

**📈 对比分析**

在上述基准上与多种 VLM 与 agentic 方法对比，LensWalk 在 LVBench、Video‑MME 上提升 5–11% 的准确率，且在 token 与帧使用量上更高效，显著优于现有单向或检索式方法；

**⚠️ 局限性**

局限性包括：对 Reasoner 的认知能力依赖强度高，工具的分辨率和覆盖面仍受预设限制，且在某些视觉任务中可能需要进一步优化工具参数或引入更细粒度的感知模块。

---

## 467. A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English

**arXiv ID:** 2603.24549 | [PDF](https://arxiv.org/pdf/2603.24549v1)

**作者:** Dana Serditova `[一作]` (University of Regensburg), Kevin Tang `[通讯]` (Heinrich Heine University Dusseldorf)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对英国纽卡斯尔英语的自然语料进行细粒度ASR错误分析，考察性别、年龄、社会经济水平等社会变量对错误率的影响。

**💡 创新点**

首次从社会语言学视角系统关联地方方言特征（元音变体、语法与词汇）与ASR错误，揭示非标准变体如何导致技术偏差。

**🔧 技术方法**

使用Rev AI商用ASR引擎，对音频进行转写并对比人工注释，结合统计建模与声学案例研究进行分析。

**📊 数据集**

利用Tyneside地区的Diachronic Electronic Corpus of Tyneside English（DECTE）共72小时、160位说话者的自然对话语料。

**📈 对比分析**

与先前的四个ASR系统预试相比，Rev AI在DECTE上的WER为31.95%；通过负二项混合模型评估错误分布，发现语音学错误占比最高，男性在词汇错误上显著高于女性。

**⚠️ 局限性**

局限性包括：仅评估单一商用ASR（黑盒模型）、数据量与方言覆盖度有限、未进行模型微调或跨方言比较，且缺乏对多方言普适性的验证。

---

## 468. Pseudo-MDP Convolutional Codes for Burst Erasure Correction

**arXiv ID:** 2603.24516 | [PDF](https://arxiv.org/pdf/2603.24516v1)

**作者:** Zita Abreu `[一作]`, Raquel Pinto `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出伪MDP（Pseudo‑MDP）和x‑伪MDP卷积码，通过仅要求部分列距离最优来减少所需有限域大小并提升码率，能够纠正更大长度的突发丢包。

**💡 创新点**

创新点在于：①将MDP卷积码的列距离条件放宽，只保持到第ν个列距离最优；②给出从低度MDP码扩展生成伪MDP码的构造；③进一步推广为x‑伪MDP，允许更长突发、更高码率、在更小域上构造。

**🔧 技术方法**

采用经典MDP卷积码理论、列距离与滑动编码矩阵的性质，利用矩阵扩展和行列操作构造新编码器。

**📊 数据集**

本研究为理论设计，没有使用具体数据集。

**📈 对比分析**

与传统MDP码相比，伪MDP码在相同码率下可在更小的有限域上实现；在突发纠错性能上可覆盖更大长度的失真窗口；相比其他非MDP卷积码，保持了部分MDP优势。

**⚠️ 局限性**

局限性：①构造仅适用于k|δ且n≥(ν+1)k（或其更宽松形式），不一定能覆盖所有参数；②伪MDP码并非MDP，列距离在ν+1后可能下降，导致在特定窗口下纠错能力降低；③若首段窗口全部失真，部分信息无法恢复；④在某些参数下仍需较大域，未能完全解决大字段问题。

---

## 469. Toward Physically Consistent Driving Video World Models under Challenging Trajectories

**arXiv ID:** 2603.24506 | [PDF](https://arxiv.org/pdf/2603.24506v1)

**作者:** Jiawei Zhou `[一作]` (Zhejiang University), Yu Li `[通讯]` (Zhejiang University)

**通讯引用:** 30763 | [OpenAlex ID](https://openalex.org/A5100345712)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为PhyGenesis的驾驶世界模型，能在物理违背或极端轨迹下生成高保真多视角驾驶视频。

**💡 创新点**

创新点包括① 物理条件生成器可将任意二维轨迹纠正为物理可行的6-DoF轨迹；② 物理增强视频生成器在训练时结合物理丰富的模拟数据，实现视觉质量与物理一致性的双重提升；③ 采用异构训练（真实 + CARLA极端场景）与两阶段课程学习。

**🔧 技术方法**

采用基于Diffusion Transformer的WAN2.1架构，加入空间交叉注意、Agent自注意、地图注意与时序卷积的物理条件生成网络；视频生成器使用VAE+DiT、时间序列卷积、Rectified Flows训练；使用位置编码、时间嵌入、损失加权等技术。

**📊 数据集**

利用nuScenes真实多视角日志与CARLA模拟生成的物理极端场景（包含碰撞、越道等），共约31小时，最终构建约9.7小时极端剪辑与4.6小时真实数据混合的异构数据集。

**📈 对比分析**

与UniMLVG、MagicDrive‑V2、DiST‑4D等基线相比，在nuScenes、CARLA Ego和CARLA Adv三组数据上均实现了FID/FVD下降、PHY得分提升、人工偏好率提高，尤其在物理违背轨迹下的性能提升最显著。

**⚠️ 局限性**

局限性主要在于：① 需要大量高质量的物理极端模拟数据，收集成本高；② 目前仅针对二维轨迹输入的纠正，复杂非平面动态或多目标交互的泛化尚待验证；③ 对极端场景的物理建模仍受模拟器逼真度限制，真实世界未知事件的迁移性能未充分评估。

---

## 470. A faster polynomial-space algorithm for Hamiltonian cycle parameterized by treedepth

**arXiv ID:** 2603.24492 | [PDF](https://arxiv.org/pdf/2603.24492v1)

**作者:** Stefan Kratsch `[一作]` `[通讯]` (Humboldt-Universit"at zu Berlin), Stefan Kratsch (Humboldt-Universit"at zu Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了一种随机化多项式空间算法，能够在给定深度为τ的消除森林（treedepth decomposition）的图上，在 4^τ · n^O(1) 时间内解决部分循环覆盖（Partial Cycle Cover）以及一系列相关的图论问题。

**💡 创新点**

核心创新在于：①将原来使用辅助图中完美匹配的做法改为直接使用“相容匹配对”（consistent matchings），大幅降低时间指数；②引入包含-排除（inclusion‑exclusion）公式与一致性匹配的组合，构造高效的递归多项式；③实现了从 5^τ 到 4^τ 的时间改进，并将这些技术统一到多项式空间的分支搜索框架中。

**🔧 技术方法**

主要技术包括：
- 消除森林（treedepth decomposition）与树形结构的递归分解；
- 包含-排除原理与多项式表示（对匹配对进行符号计数）；
- 匹配对的一致性判定与组合计数；
- 隔离引理（Isolation Lemma）用于随机化唯一化最小权重解；
- 分支算法（branching）替代传统的动态规划，以实现多项式空间。

**📊 数据集**

该工作属于理论算法研究，没有使用具体实验数据集；所有结果均为理论复杂度分析与证明。

**📈 对比分析**

理论上与先前的 5^τ · n^O(1) 随机多项式空间算法（Nederlof 等）相比，时间指数从 5 降至 4，且保持了相同的空间复杂度；此外，在树宽（treewidth）参数下已知的 4^τ · n^O(1) 复杂度与本工作在 treedepth 参数上的表现相匹配。

**⚠️ 局限性**

局限性：
- 算法仍是随机化且只保证单侧错误（最多 1/2 的假阴性概率）；
- 对消除森林的输入要求高，若未知如何高效构造，整体效能受限；
- 仍未达到可能的最佳指数（如 (2+√2)^τ），存在进一步改进空间；
- 目前仅理论证明，实际实现与实验评估尚未完成。

---

## 471. Composer 2 Technical Report

**arXiv ID:** 2603.24477 | [PDF](https://arxiv.org/pdf/2603.24477v1)

**作者:** Cursor Reseach `[一作]`, Zhiyuan Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并训练了专门化的Composer 2模型，用于代理式软件工程，通过持续预训练和强化学习提升代码生成与问题解决能力。

**💡 创新点**

创新点在于结合持续预训练后的大规模异步强化学习、工具交互、非线性长度惩罚和自我总结技术，以及构建真实开发任务的CursorBench评测体系，使模型在真实场景下实现低成本高性能。

**🔧 技术方法**

技术包括1T参数Mixture‑of‑Experts基础模型（Kimi K2.5），持续预训练、SFT、MXFP8/NVFP4量化、Context Parallelism、DeepEP、Ray/PyTorch分布式训练、Anyrun环境、Anygress网络控制、异步RL与KL正则化、自我总结与非线性长度惩罚等。

**📊 数据集**

使用的训练数据集为代码主导的大规模混合语料、内部CursorBench（真实团队交互任务）以及公开评测数据集SWE‑bench Multilingual和Terminal‑Bench。

**📈 对比分析**

通过在CursorBench、SWE‑bench Multilingual和Terminal‑Bench上评估，Composer 2分别取得61.3%、73.7%和61.7%的准确率，较前一代提升37%、16.8%和21.7%，在保持竞争力的同时显著降低推理成本。

**⚠️ 局限性**

局限性包括对极长任务的处理仍不充分、某些智能或连贯性行为尚需改进、训练成本高且依赖内部数据集，公开基准可能因数据污染导致评估偏差。

---

## 472. Mechanic: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving

**arXiv ID:** 2603.24465 | [PDF](https://arxiv.org/pdf/2603.24465v1)

**作者:** Ruichen Qiu `[一作]` (Academy of Mathematics and Systems Science), Ruyong Feng `[通讯]` (Academy of Mathematics and Systems Science)

**通讯引用:** 322 | [OpenAlex ID](https://openalex.org/A5050890628)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Mechanic 代理系统，结合正式推理、错误定位（sorrifier）和子目标分解，实现了高效的形式化定理证明。

**💡 创新点**

创新点在于利用 Lean 的 sorry 占位符进行局部错误化简、自动拆分子目标，避免了全局重写和上下文长度膨胀。

**🔧 技术方法**

采用 LLM（Gemini‑3.1‑Pro）与 Lean 生态工具（Lean 4、Mathlib、LeanDex、Loogle）以及自研的 Sorrifier 与子目标提取模块。

**📊 数据集**

在 Putnam 2025（12 题）和 IMO 2025（4 题）等数学竞赛正式证明数据集上进行评测。

**📈 对比分析**

与 Hilbert、Aristotle、Axiom、Seed、Numina 等基线对比，Mechanic 在大多数问题上实现了更低时间、更低成本、更短证明长度和更少辅助定理的优势，成功证明 11/12 个 Putnam 题和 4/4 个 IMO 题。

**⚠️ 局限性**

局限包括对极难问题仍可能失败，循环拆分的终止阈值需要手动设置，且在极简定理时可能不需要 informal 步骤导致额外开销。

---

## 473. Uniform Laws of Large Numbers in Product Spaces

**arXiv ID:** 2603.24493 | [PDF](https://arxiv.org/pdf/2603.24493v1)

**作者:** Ron Holzman `[一作]` (Technion), Alexander Shlimovich `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

在产品空间下研究统一收敛，证明当分布满足统一盒子连续性时，族的线性VC维数有限与否决定了是否可以统一估计概率。

**💡 创新点**

引入“线性VC维数”这一新组合维度，并证明它在产品空间下完全决定统一估计的可行性；提出一种不依赖经验均值、利用产品结构的新的离散化与估计方法。

**🔧 技术方法**

核心技术包括：盒子连续性（distribution‐box continuity）的定义与利用；构造基于样本投影的产品网格，证明其为对称差的ε‑网格；将族离散化为有限代表族；利用Sauer–Shelah–Perles 级别的“网格SSP”引理把线性VC维数与网格上出现的子集数量联系起来；组合 Hoeffding‑式有限类一致估计得到最终样本复杂度上界。

**📊 数据集**

本文为理论工作，无实验数据集，所有结论均为证明与上界/下界分析。

**📈 对比分析**

与经典VC理论相比，本文给出了更细粒度的上界：当仅考虑产品分布时，样本复杂度上界为 O(d²/ε²·(g+log(1/δ)))；下界为 Ω((g+log(1/δ))/ε²)。两者在 g、ε、δ 上相匹配，但上界对维度 d 的依赖仍为指数级，显示存在显著差距。

**⚠️ 局限性**

主要局限：
1) 对维度 d 的上界依赖过大，无法得到与下界匹配的多维上界。
2) 需要进一步研究能否用比线性VC更强的 invariant 来控制样本复杂度。
3) 对无限维产品空间的统一估计仍缺乏完整的阐述，现有结果只能给出不完整的必要/充分条件。

---

## 474. TAG: Target-Agnostic Guidance for Stable Object-Centric Inference in Vision-Language-Action Models

**arXiv ID:** 2603.24584 | [PDF](https://arxiv.org/pdf/2603.24584v1)

**作者:** Jiaying Zhou `[一作]` (Sun Yat-sen University), Guangrun Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3056 | [OpenAlex ID](https://openalex.org/A5052611320)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Target-Agnostic Guidance（TAG）推理时残差引导机制，用于增强视觉语言动作（VLA）模型在嘈杂环境中的实例级定位，减少误抓和误目标。

**💡 创新点**

创新点在于将目标消除的无条件视觉输入与原始观察做残差对比，显式抑制背景和干扰物偏置；训练时通过随机目标消除校准，且不改动原有模型架构。

**🔧 技术方法**

采用Classifier-Free Guidance风格的残差引导、目标消除/遮蔽/填充技术（Grounding DINO、SAM2、MiniMaxRemover）、流匹配或动作预测的VLA模型（π0、π0.5等），以及EMA、cosine LR等训练技巧。

**📊 数据集**

使用LIBERO、LIBERO-Plus、VLABench（Track 1）等常见机器人操作基准数据集。

**📈 对比分析**

与π0、π0.5、InternVLA-M1、GR00T-N1.6等强基线比较，TAG在LIBERO-Long将成功率从89.6%提升至97.0%，在VLABench的“Select Poker/Select Mahjong”任务中成功率从29.4%提升至55.4%，并在多种扰动设置下表现出显著的鲁棒性。

**⚠️ 局限性**

局限性包括对目标消除精度和实时性的依赖，动态遮蔽可能破坏空间先验；对极端视觉噪声和真实环境光照/遮挡的适应性仍待验证；过高的指导尺度会导致动作不平滑，需要仔细调参。

---

## 475. The Stochastic Gap: A Markovian Framework for Pre-Deployment Reliability and Oversight-Cost Auditing in Agentic Artificial Intelligence

**arXiv ID:** 2603.24582 | [PDF](https://arxiv.org/pdf/2603.24582v1)

**作者:** Biplab Pal `[一作]` (University of Maryland Baltimore County), Santanu Bhattacharya `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 17817 | [OpenAlex ID](https://openalex.org/A5090701117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于事件日志的马尔可夫框架，用于在企业工作流中评估代理式人工智能的可行性与监督成本。

**💡 创新点**

创新点在于将状态-动作盲区、熵风险门控和监督成本统一为可度量的指标，并通过该框架实现可靠性与成本的耦合评估，从而提前判断何时可以安全赋予自动化。

**🔧 技术方法**

使用了马尔可夫决策过程（MDP）理论、熵（Shannon entropy）与风险加权门控、以及监督成本的期望值公式；同时构建了基于日志的经验策略和转移核。

**📊 数据集**

采用了2019年BPI（Business Process Intelligence Challenge）采购-付款（purchase-to-pay）事件日志，共251,734个案例、1,595,923条事件，涵盖42个动作。

**📈 对比分析**

通过对完整日志的描述性审计和对按时间拆分的80/20测试集的代理模拟，比较了理论指标（如状态最大概率m(s)、安全完成率R_safe）与实际执行的准确率、零接触完成率和人类触点数。结果显示，m(s)与实际步骤准确率平均差距约3.4个百分点，安全完成率与实际相符但略保守；人类触点数随门控宽松度下降，可靠性与成本呈相互权衡关系。

**⚠️ 局限性**

局限性包括：1）日志为观察性数据，无法直接评估不同动作的因果效应；2）仅采用一阶马尔可夫近似，未考虑更深层次历史或工具状态，可能低估真实支持难度；3）风险权重仅基于交易金额与异常标记，缺乏领域特定的合规和政策成本；4）门控阈值需要经验设定，缺乏自动化调优机制。

---

## 476. Towards Training-Free Scene Text Editing

**arXiv ID:** 2603.24571 | [PDF](https://arxiv.org/pdf/2603.24571v1)

**作者:** Yubo Li `[一作]` (Chinese Academy of Sciences), Kexin Zhang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 9688 | [OpenAlex ID](https://openalex.org/A5100459337)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个训练‑free 的场景文本编辑框架 TextFlow，用于在不重新训练模型的情况下修改自然图像中的文本。

**💡 创新点**

创新点在于将 Flow Manifold Steering (FMS) 与 Attention Boost (AttnBoost) 两阶段、相互补充的指导策略相结合，实现了结构与文本细节的同时保持。

**🔧 技术方法**

技术方案基于 FLUX‑Kontext 的流匹配扩散模型，利用 FMS 对噪声轨迹进行结构保持校正，AttnBoost 通过强化跨模态注意力实现文本精细渲染。

**📊 数据集**

使用 ScenePair 数据集（1280 张真实场景文本对）进行训练与评估。

**📈 对比分析**

与 DiffSTE、AnyText、TextFlux、FlowEdit、Flux‑Kontext 等主流方法对比，TextFlow 在 SSIM、PSNR、FID 等图像质量指标上遥遥领先，并在文本准确率（ACC 约 80%）与 NED 方面保持竞争优势。

**⚠️ 局限性**

局限性包括：依赖扩散模型导致推理速度慢，难以实时处理；对多行文本或复杂排版的保持效果尚不理想。

---

## 477. Boosting LLMs for Mutation Generation

**arXiv ID:** 2603.24560 | [PDF](https://arxiv.org/pdf/2603.24560v1)

**作者:** Bo Wang `[一作]` (Beijing Jiaotong University), Jie M. Zhang `[通讯]` (King's College London)

**通讯引用:** 4207 | [OpenAlex ID](https://openalex.org/A5088708850)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合检索增强生成（RAG）、逻辑代码分块和监督微调的 LLM 变异生成框架（SMART），并在 1,991 条真实 Java Bug（Defects4J+ConDefects）上进行评估。

**💡 创新点**

创新点：①利用向量化的真实 Bug–Fix 对作为上下文检索，提供语义相关的 few‑shot 示例；②通过逻辑代码分块在方法级别生成局部变异，提升语义一致性与多样性；③对 LLM 进行基于真实 Bug Coupled 的监督微调，使模型更贴合实际缺陷；④证明 7B 级小模型在该框架下可与 GPT‑4o 比肩，具备实际部署价值。

**🔧 技术方法**

技术细节：大型语言模型（GPT‑4o、DeepSeek‑Coder、Llama‑3.1、Qwen‑2.5 系列）；CodeT5 编码器做向量检索；基于 AST 的逻辑分块算法；监督微调（causal LM）使用 13,760 条真实 Bug‑Coupled 变异；评估指标包括 Generation Rate、Non‑duplicate Rate、Compilation Rate、Real‑Bug Detection、Ochiai、APFD、MBFL Top‑k 等。

**📊 数据集**

数据集：Defects4J 2.0（701 个 Bug）+ ConDefects（1,290 个 Bug）；用于 RAG 的 130,000 条单片段 Bug–Fix 对；训练集 13,760 条 Bug‑Coupled 变异；全部数据均为 Java。

**📈 对比分析**

比较方法：与 LLMut、LLMorpheus 两个 SOTA LLM‑based 变异生成器对比；使用多种 LLM（6.7B-32B）以及 GPT‑4o；实验结果显示 SMART 在有效性（Generation Rate 65.6% vs 42.89%，Non‑duplicate 95.6% vs 87.4%，Compilation 90.2% vs 88.9%）和有效性（Real‑Bug Detection 92.6% vs 57.9%，Average Ochiai 38.4% vs 25.6%）方面均显著提升；在 TCP（APFD）和 MBFL（Top‑1/Top‑5）上也获得最优或竞争性结果，且 7B 级小模型已逼近或超过 GPT‑4o。

**⚠️ 局限性**

局限性：①引入 RAG 与分块后令 prompt 较长，导致 token 及推理时间上升（约 37% 额外 token）；②微调需要大量 GPU 资源，尤其 32B 模型峰值显存 140GB；③实验仅针对 Java，跨语言推广待验证；④检索时需一次性构建索引，耗时约 10 分钟；⑤虽做了数据泄露控制，但仍有少量与真实 Bug 的 exact‑match 可能存在；⑥RAG 选择最近邻的相似度度量会影响结果，需在实际部署中调优。

---

## 478. Polynomial Speedup in Diffusion Models with the Multilevel Euler-Maruyama Method

**arXiv ID:** 2603.24594 | [PDF](https://arxiv.org/pdf/2603.24594v1)

**作者:** Arthur Jacot `[一作]` `[通讯]`, Arthur Jacot

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种多层Euler-Maruyama (ML-EM) 方法，用于在具有不同精度的神经网络估计器上以随机方式进行 SDE/ODE 离散化，从而在扩散模型中实现显著的计算加速。

**💡 创新点**

创新点在于将多级蒙特卡罗思想与扩散模型的多尺度网络结构结合，通过对不同网络大小的随机采样，仅在少量步骤使用最高精度网络，即可在 Harder than Monte Carlo (HTMC) 区间实现 ϵ^-γ 级别的计算复杂度，等价于一次最大的网络评估。

**🔧 技术方法**

主要技术包括：多层Euler-Maruyama 离散化；对网络层次使用 Bernoulli 采样并设置概率 p_k；利用前向梯度和无偏估计器学习时间依赖的概率参数；以及对 SDE 误差与计算量的理论上界分析。

**📊 数据集**

实验使用 CelebA 数据集，将图像裁剪并缩放到 64×64，训练一系列 UNet 网络（大小从 8 到 64），并在此数据集上评估生成质量与计算时间。

**📈 对比分析**

与传统 EM（单层）方法对比，ML-EM 在相同 MSE（约 10^-3 以上）下可获得多达四倍的速度提升；学习概率参数的 ML-EM 进一步提升了性能，达到 10 倍左右的计算时间缩短；实验中还使用了不同步数、网络组合以及 GPU 批处理策略。

**⚠️ 局限性**

局限性包括：依赖于 HTMC 区间（γ>2）的假设，若模型或任务不满足此假设则加速效果不明显；需要预先训练一组多尺度网络，增加了模型管理和存储成本；随机采样导致误差方差较大，需要额外的采样或优化；在更大规模的数据集上，理论与实际速度提升可能因硬件瓶颈而受到限制。

---

## 479. Retrieval Improvements Do Not Guarantee Better Answers: A Study of RAG for AI Policy QA

**arXiv ID:** 2603.24580 | [PDF](https://arxiv.org/pdf/2603.24580v1)

**作者:** Saahil Mathur `[一作]` (Purdue University), Tunazzina Islam `[通讯]` (Purdue University)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5056005531)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对AI治理政策文档的检索增强生成（RAG）问答系统，并在AGORA语料库上进行实验。

**💡 创新点**

首次系统性评估检索改进对政策问答的影响，发现检索性能提升不一定带来整体问答质量提升，且可能导致更自信的幻觉；提出将对比学习与偏好优化结合的域适配RAG管线。

**🔧 技术方法**

使用ColBERTv2检索器并通过对比学习微调；使用Mistral‑7B‑Instruct生成器，并通过Direct Preference Optimization (DPO) 与LoRA进行人类偏好对齐；评估采用RAGAS框架、MRR、Recall@k、MAP、答案相关性与真实性评分。

**📊 数据集**

AGORA数据集（947份AI政策文件，划分为7893段），以及基于这些文档生成的合成问答对和手工标注的偏好对。

**📈 对比分析**

与基线（Base ColBERT + Base Mistral、CL ColBERT + DPO Mistral）以及GPT‑5.4进行对比。检索指标（MRR、Recall@5、MAP@5）提升显著，但端到端问答准确率和真实性仅有微弱提升；GPT‑5.4在问答上远优于RAG管线；在缺失文档的情况下，强检索还会诱发更自信的幻觉。

**⚠️ 局限性**

受限于计算资源仅使用7B模型，偏好数据量有限，评测集未覆盖所有最新政策，导致检索与生成对齐效果不充分；可能存在偏见与幻觉风险，且对不断演变的法规缺乏及时覆盖。

---

## 480. Vision-Language Models vs Human: Perceptual Image Quality Assessment

**arXiv ID:** 2603.24578 | [PDF](https://arxiv.org/pdf/2603.24578v1)

**作者:** Imran Mehmood `[一作]` (University of Galway), Brian Deegan `[通讯]` (University of Galway)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5091422105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对六种视觉语言模型（VLM）进行对比实验，评估其在对比度、色彩丰富度和整体喜好三个主观图像质量尺度上的表现，并与人类心理物理实验结果进行系统对比。

**💡 创新点**

创新点在于构建了面向人类主观评估的VLM基准，揭示属性依赖性差异、内部一致性与人类一致性之间的权衡，并引入感知可分离度分析来解释模型表现。

**🔧 技术方法**

技术方法包括基于提示的双图像比较、标准化z分数、Spearman相关性、bootstrap置信区间、交叉模型变异率及线性回归权重分析。

**📊 数据集**

使用的数据集为10幅HDR图像经7种调色映射算子渲染得到70幅图像，并采用Mehmood等人收集的20位受试者的心理物理评估数据。

**📈 对比分析**

通过Spearman相关系数与场景级相关性评估，发现GPT在整体喜好上最高ρ=0.86，Qwen在对比度与色彩上ρ分别为0.86与0.93，模型内部一致性与人类一致性并不完全对应，整体表现呈现属性依赖与场景分辨度相关。

**⚠️ 局限性**

局限性包括对提示与API版本的敏感性、仅评估对比度、色彩丰富度与整体喜好三种属性、未覆盖模糊度、自然度等常见质量维度，以及模型更新后可能导致结果变化。

---

## 481. Completeness of Unbounded Best-First Minimax and Descent Minimax

**arXiv ID:** 2603.24572 | [PDF](https://arxiv.org/pdf/2603.24572v1)

**作者:** Quentin Cohen-Solal `[一作]` `[通讯]` (Université Paris-Dauphine), Quentin Cohen-Solal (Université Paris-Dauphine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对两种无界深度搜索算法——Unbounded Best‑First Minimax 和 Descent Minimax 进行泛化，定义了包含完成技术（Completion）的通用类“Unbounded Minimax‑based Algorithms”，并给出完整性（能在有限时间内求出最优策略）的单一证明；随后通过大量游戏实验验证完成技术能显著提升搜索性能。

**💡 创新点**

创新点在于：①统一把两种算法纳入一个通用框架，并通过一次证明展示该类所有算法均为完整；②首次实证证明完成技术不仅理论上使算法完整，而且在实际游戏中能够明显提升胜率，弥补了以往无界搜索方法的缺陷。

**🔧 技术方法**

采用的主要技术包括：完成技术（为每个节点维护完备值 c(s)、启发式评估 v(s)、分辨值 r(s) 并按字典序优先选择子节点）；无界深度优先搜索框架；强化学习得到的评估函数；哈希表（置换表）和安全决策策略（选取出现次数最多的动作）。

**📊 数据集**

实验数据集由 22 个确定性、零和、完全信息的棋类游戏组成，包括 Amazons、Ataxx、Breakthrough、Brazilian/Canadian Draughts、Clobber、Connect6、International Draughts、Chess、Go 9×9、Gomoku、Havannah、Hex、Lines of Action、Othello、Santorini、Surakarta、Xiangqi 及 Arimaa。

**📈 对比分析**

对比方法：在相同搜索时长 10 秒、相同评估函数、相同置换表与安全决策的条件下，进行 6 次重复实验，每次约 450 场比赛；用 1（胜）、0（平）、-1（负）计分，取平均得分。实验结果显示，加入完成技术后，平均胜率提升约 6.34%（区间 [6.03%,6.66%]），而未使用完成技术时平均得分为 -6.34%，表明完成技术在实战中能显著提升胜率。

**⚠️ 局限性**

局限性包括：仅验证了确定性、完全信息的两人零和游戏；对评估函数的质量高度依赖，若评估不佳，性能提升有限；完成技术在某些游戏中可能增加计算和内存开销；理论完整性并不保证在极大搜索空间下能在可接受时间内收敛。

---

## 482. Anti-I2V: Safeguarding your photos from malicious image-to-video generation

**arXiv ID:** 2603.24570 | [PDF](https://arxiv.org/pdf/2603.24570v1)

**作者:** Duc Vu `[一作]` (Qualcomm Technologies Inc), Anh Tran `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对图像到视频扩散模型的滥用，本文提出 Anti‑I2V 对抗性防护；

**💡 创新点**

创新点在于将扰动同时投射至 L*a*b* 色彩空间与频域，结合内部特征崩塌与锚定损失，实现跨模型的高效干扰；

**🔧 技术方法**

采用双空间扰动（DSP）、内部表示崩塌（IRC）与锚定（IRA）损失，辅以 LPIPS/CLIP 辅助；

**📊 数据集**

使用 CelebV‑Text 与 UCF101 两大面向人像与全身动作的数据集；

**📈 对比分析**

与 SDS、AdvDM、MIST、VGMShield 等基线在 CogVideoX、OpenSora、DynamiCrafter 上对比，Anti‑I2V 在 ISM、C‑FIQA、Q‑Align 等指标均取得最低值，表明更强的身份抹除和视频失真；

**⚠️ 局限性**

局限在于仅针对无姿态引导的 I2V 模型，对多姿态或实时生成的适用性尚待验证。

---

## 483. DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving

**arXiv ID:** 2603.24587 | [PDF](https://arxiv.org/pdf/2603.24587v1)

**作者:** Pengxuan Yang `[一作]` (Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 15592 | [OpenAlex ID](https://openalex.org/A5100624298)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在可解释的潜在世界模型中实现自主驾驶的强化学习训练；

**💡 创新点**

三项创新：1）Shortcut Forcing压缩采样步骤80×加速；2）自回归稠密奖励模型实现细粒度奖励分配；3）Gaussian词典采样保证物理可行的轨迹探索。

**🔧 技术方法**

利用Epona自回归扩散世界模型、自动化稠密奖励网络、Gaussian词典采样的GRPO算法以及视频编码器等。

**📊 数据集**

NavSim v2（及v1）模拟数据集。

**📈 对比分析**

与多种基准方法对比，在NavSim v2闭环规划上取得87.7 EPDMS，击败所有现有方法；在NavSim v1也取得88.7得分。

**⚠️ 局限性**

局限性：仍需大量计算资源；对离散动作空间的依赖；对训练数据规模的敏感性较高，且在极端稀有场景下可能产生幻觉。

---

## 484. Comparing Developer and LLM Biases in Code Evaluation

**arXiv ID:** 2603.24586 | [PDF](https://arxiv.org/pdf/2603.24586v1)

**作者:** Aditya Mittal `[一作]` (Carnegie Mellon University), Valerie Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1262 | [OpenAlex ID](https://openalex.org/A5088847857)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 TRACE 框架，用以在实际的编程交互场景（聊天式编程、IDE 自动补全、指令式代码编辑）中评估并解释 LLM 判断器对代码的偏好。

**💡 创新点**

创新点在于：① 自动发现可解释的评估维度（rubric items），① 通过对人类与 LLM 判断的权重进行对比，揭示多模态下的偏差来源；② 将这些维度聚合成可操作的 Rubric，为后续对齐提供依据。

**🔧 技术方法**

技术手段包括：LLM‑as‑a‑Judge 进行二选一偏好判断、通过 LLM 生成并聚合评估标准、使用逻辑回归模型估计每个维度的权重、对比人类与模型的系数、计算位置一致性与整体准确率。

**📊 数据集**

使用的数据集为 Copilot Arena（文件内补全）、EDIT‑Bench（指令式编辑）和 Chatbot Arena（聊天回复），每类约 500 条双选偏好样本，并辅以 30 条人工复核样本构成人类基准。

**📈 对比分析**

比较方法：统计整体准确率与位置一致性，并与人类多数投票（MUA）对齐；结果显示，所有评测的 LLM 判断器平均落后人类 12–23%，各判定器无明显优势；在各维度上，判定器在“功能性”和“可读性”等方面与人类存在显著权重差异。

**⚠️ 局限性**

局限性：① 仅使用线性回归解释维度权重，无法捕捉维度间交互；② TRACE 只揭示偏差而未提供直接对齐策略，单纯注入维度提示效果有限；③ 需要更强的表达模型、校准机制和针对性数据选择来提升对齐效果。

---

## 485. MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination

**arXiv ID:** 2603.24579 | [PDF](https://arxiv.org/pdf/2603.24579v1)

**作者:** Zhuo Li `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 MARCH 的多代理自检框架，用来在检索增强生成（RAG）任务中减少 LLM 的幻觉，并让模型在生成时自我验证事实一致性。

**💡 创新点**

创新点在于将同一模型拆分为 Solver、Proposer、Checker 三个角色，并通过信息不对称的内部验证机制与零容忍奖励（Zero‑Tolerance Reward）实现自监督对齐，从而使模型在生成与检索过程中内部形成可验证的事实检查循环。

**🔧 技术方法**

使用了 Proximal Policy Optimization（PPO）强化学习、零容忍奖励策略、QA 级别的答案分解、检索增强生成、内部自检（Checker）与生成（Solver）协同的多代理架构，以及多样本投票评估。

**📊 数据集**

使用了 BioASQ（STEM 领域）、2WikiMultiHopQA、MuSiQue（多跳 QA）、以及评测集 RAGTruth、FaithBench、ContextualJudgeBench、Facts Grounding 等多种数据集进行训练与评估。

**📈 对比分析**

通过在多项事实一致性基准（RAGTruth、FaithBench、Facts Grounding）、对话与摘要评价（ContextualJudgeBench）以及多跳 QA 数据集（HotpotQA、2WikiMultiHopQA、MuSiQue）上与基线模型（如 Llama3.1‑8B、GPT‑4o、Qwen3‑8B 等）进行对比，MARCH 在 8B 规模 LLM 上实现了与大型专有模型相近甚至更优的准确率，显著降低幻觉率并提升多跳推理性能。

**⚠️ 局限性**

局限性包括：仍依赖检索结果的质量；零容忍奖励的二元化可能导致早期训练中梯度不足；对更复杂或跨领域的推理细粒度验证不足；训练成本高；以及缺乏对极长文本或多领域一致性的充分评估。

---

## 486. The Free-Market Algorithm: Self-Organizing Optimization for Open-Ended Complex Systems

**arXiv ID:** 2603.24559 | [PDF](https://arxiv.org/pdf/2603.24559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 487. EndoVGGT: GNN-Enhanced Depth Estimation for Surgical 3D Reconstruction

**arXiv ID:** 2603.24577 | [PDF](https://arxiv.org/pdf/2603.24577v1)

**作者:** Falong Fan `[一作]` (University of Arizona), Jerzy Rozenblit `[通讯]` (University of Arizona)

**通讯引用:** 2147 | [OpenAlex ID](https://openalex.org/A5020169647)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 EndoVGGT 框架，实现可直接推断深度和相机姿态的无优化手术三维重建。

**💡 创新点**

创新点在于 DeGAT 模块通过在特征空间动态构建语义图，捕获远程相关性并在软组织变形和工具遮挡下保持几何连贯性。

**🔧 技术方法**

采用 VGGT 基础网络、DINOv2 视觉特征、DPT 头部、GAT 机制以及多任务损失，构成端到端的深度与相机预测。

**📊 数据集**

在 EndoSLAM、SCARED 和 EndoNeRF 三个公开数据集上进行训练与评估。

**📈 对比分析**

与 VGGT、EndoSurf、EndoLRMGS 等基线比较，PSNR 提升至 34.35、SSIM 达 0.939，并在未见数据集上实现零样本泛化，显著优于现有方法。

**⚠️ 局限性**

仍受限于对极端软组织动态变形的鲁棒性、缺乏时间一致性以及对真实手术场景中多尺度纹理的适应性。

---

## 488. Coordinating Spot and Contract Supply in Freight Marketplaces

**arXiv ID:** 2603.24574 | [PDF](https://arxiv.org/pdf/2603.24574v1)

**作者:** Philip Kaminsky `[一作]` (Amazon), Roger Lederman `[通讯]` (Amazon)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种协调长期合同与即时报价的算法，旨在最小化数字货运市场的采购成本。

**💡 创新点**

创新点在于将Dual Frank Wolfe算法应用于双侧价格协商，利用影子价格实现合同与即时市场的协同。

**🔧 技术方法**

采用Frank‑Wolfe条件梯度法与MDP定价oracle相结合，构建对偶问题并求解。

**📊 数据集**

使用半合成数据与大型数字货运市场的真实数据进行评估。

**📈 对比分析**

与传统的Load Bifurcation算法对比，Dual Frank Wolfe在不同合同类型下平均节约10%–50%，并在大市场下实现近最优。

**⚠️ 局限性**

局限包括对合同可替代性假设的依赖，以及在极端稀疏市场或高相关性时性能可能下降。

---

## 489. POLY-SIM: Polyglot Speaker Identification with Missing Modality Grand Challenge 2026 Evaluation Plan

**arXiv ID:** 2603.24569 | [PDF](https://arxiv.org/pdf/2603.24569v1)

**作者:** Marta Moscati `[一作]` (Johannes Kepler University Linz), Shah Nawaz `[通讯]` (Johannes Kepler University Linz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了POLY‑SIM 2026大赛，专注于多模态说话人识别中视觉模态缺失与跨语言场景的双重挑战。

**💡 创新点**

创新点在于将缺失模态和跨语言问题结合，构建统一的评估框架与公开基准数据，为研究者提供可重复验证的平台。

**🔧 技术方法**

使用双分支网络（面部CNN + 语音编码器），并通过正交约束损失进行多模态特征融合，辅以预训练模型。

**📊 数据集**

采用MAV‑Celeb数据集，该集包含英语与乌尔都语的双语说话人视频，提供完整的面部与语音配对样本。

**📈 对比分析**

与公开的基线模型对比，跨语言缺失模态的P‑accuracy显著下降，但基线提供了可量化的参考点，展示了挑战的难度。

**⚠️ 局限性**

局限在于仅覆盖两种语言且缺少更大规模、多语言、多模态缺失的真实场景，未来工作需扩展至更丰富的语言与模态缺失情况。

---

## 490. Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving

**arXiv ID:** 2603.24581 | [PDF](https://arxiv.org/pdf/2603.24581v1)

**作者:** Linbo Wang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

构建了一个端到端的自动驾驶框架 Latent‑WAM，结合空间感知压缩编码器和动态潜在世界模型，实现高效轨迹规划。

**💡 创新点**

创新点在于将几何基础模型知识蒸馏到视觉编码器，并使用可学习查询压缩多视图图像，同时利用因果 Transformer 与 3D‑RoPE 进行时空动态预测。

**🔧 技术方法**

主要技术包括 DINOv2 视觉编码器、可学习查询压缩、几何蒸馏、因果 Transformer、3D‑RoPE、Self‑supervised 视觉预测和监督运动预测。

**📊 数据集**

在 NAVSIM v2 与 HUGSIM 两大仿真数据集上进行训练与评估。

**📈 对比分析**

在 NAVSIM v2 上实现 89.3 EPDMS，超越前沿 perception‑free 方法 3.2 分；HUGSIM 上得到 45.9 RC 与 28.9 HD‑Score，零样本即可获得最佳 RC 并与基线竞争 HD‑Score。

**⚠️ 局限性**

局限性包括对几何基础模型的依赖、在高频度时序预测上性能不显著提升以及需要额外的 EMA 编码器在训练阶段增加计算负担。

---

## 491. Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation

**arXiv ID:** 2603.24576 | [PDF](https://arxiv.org/pdf/2603.24576v1)

**作者:** Xinying Guo `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**通讯引用:** 7161 | [OpenAlex ID](https://openalex.org/A5005666034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108`

**🎯 论文内容**

研究机器人在感知混淆下的长期记忆需求，提出 Chameleon 架构进行事件式记忆写入和目标导向检索，并在真实 UR5e 机器人上验证。

**💡 创新点**

设计了基于人类 EC–HC–PFC 的层次连续记忆堆栈，结合几何一致的跨视角编码、HoloHead 预测目标驱动检索，以及仅用决策状态驱动的条件流匹配控制。

**🔧 技术方法**

使用视觉几何编码、跨视角注意力、基于 Mamba 状态空间模型的多时间尺度记忆、HoloHead 隐式想象目标和条件 rectified flow matching 等技术。

**📊 数据集**

构建了 EpisodicMemoryDataset，包含三类任务（事件绑定、空间跟踪、序列执行），在 UR5e 机器人上收集 120 条演示并进行 36 次真实测试。

**📈 对比分析**

与 Flow Matching、ACT 等基线对比，采用 SR、DSR、MSR、CSR 等指标，Chameleon 在所有类别均显著提升 DSR (kappa) 约 20% 以上，整体成功率接近完美。

**⚠️ 局限性**

仅关注感知驱动的低级操控，缺乏语义抽象与零样本泛化，未涉及跨体型迁移与事件分割，模型在复杂开放世界任务上尚未验证。

---

## 492. Infrastructure for Valuable, Tradable, and Verifiable Agent Memory

**arXiv ID:** 2603.24564 | [PDF](https://arxiv.org/pdf/2603.24564v1)

**作者:** Mengyuan Li `[一作]` (University Of Southern California), Murali Annavaram `[通讯]` (University Of Southern California)

**通讯引用:** 6402 | [OpenAlex ID](https://openalex.org/A5018033573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出将自治代理的记忆资产化，通过可信执行环境实现记忆的可验证性，并构建市场层实现记忆交易；

**💡 创新点**

创新点在于将记忆从单次执行的无形产出转变为可验证、可转让的经济资产，并引入“gang”聚合框架与可选的去中心化结算机制；

**🔧 技术方法**

技术核心包括：可信执行环境(TEE)与远程证明、VMPL0/VMPL3分层安全执行、Rust实现的最小可信计算基底、可验证的计算证明与日志根、可选的零知识证明或GPU TEE；

**📊 数据集**

本文未使用公开数据集进行实验，而以示例任务（公共数据清洗、广告创意探索、供应商发现）作为概念验证；

**📈 对比分析**

文章为设计说明，未给出定量实验或性能对比；主要通过示例说明能减少重复探索、提高资产可再利用性；

**⚠️ 局限性**

局限性包括：需信任模型 API 提供者；TEE 可能被攻击导致可信度下降；实现复杂度高；缺乏实测性能与经济效益评估；只适用于共享相同任务结构的 gang 组。

---

## 493. Vibe Coding XR: Accelerating AI + XR Prototyping with XR Blocks and Gemini

**arXiv ID:** 2603.24591 | [PDF](https://arxiv.org/pdf/2603.24591v1)

**作者:** Ruofei Du `[一作]` (Google XR Labs), David Kim `[通讯]` (Google XR Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了XR Blocks框架与Vibe Coding XR工作流，实现自然语言指令直接生成WebXR原型；

**💡 创新点**

通过高层语义抽象（Reality Model）为LLM提供语义化接口，使其能够生成可靠的空间交互代码；

**🔧 技术方法**

结合WebXR、Three.js、Playwright、Gemini LLM及Gem UI等技术；

**📊 数据集**

使用自建VCXR-60数据集（60条XR原型提示）；

**📈 对比分析**

与Gemini多模型/思考模式对比，gemini‑3.1‑pro高思考模式pass率95.5%，低思考模式94.1%，gemini‑3‑flash约88%；生成时间从20秒到几分钟；

**⚠️ 局限性**

Web技术渲染性能不足原生引擎，网络延迟影响，LLM偶尔生成错误，缺乏多感官支持与无障碍模块。

---

## 494. Scaling Recurrence-aware Foundation Models for Clinical Records via Next-Visit Prediction

**arXiv ID:** 2603.24562 | [PDF](https://arxiv.org/pdf/2603.24562v1)

**作者:** Haresh Rengaraj Rajamohan `[一作]` (New York University), Narges Razavian `[通讯]` (New York University)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5079997593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

该论文提出了一种基于“下次就诊事件预测”的生成式 EHR 基础模型 RAVEN，旨在一次前向推理即可给出未来多年的疾病风险预测。

**💡 创新点**

创新点在于将临床就诊视为无序事件集合，采用多标签下一访预测目标并引入基于历史出现频率的递归惩罚正则化，避免模型仅重复已出现的慢性疾病。

**🔧 技术方法**

技术上使用 GPT‑2 风格的解码器 Transformer 结合 RoPE 时序编码、访视级自注意力、序列打包以及权重衰减的正则化；训练目标为下一访多标签交叉熵。

**📊 数据集**

实验数据来自 NYU Langone 约129.8 万名患者的十年 EHR 轨迹（共 42337 个独立 token），并在 Stanford Medicine 的 EHRSHOT 公开基准上做外部零样本迁移。

**📈 对比分析**

与三种基于逐词模拟的基线（Multiclass、SeqLoss、EGE）以及微调 BERT 进行比较，RAVEN 在 2/5 年期疾病发病率预测的 AUROC 与 AUPRC 均高于所有基线，且单次前向推理速度显著快于采样方法；在 EHRSHOT 上的零样本表现与少样本监督模型相近。

**⚠️ 局限性**

局限包括仅在单一机构数据上训练与评估、缺少跨编码系统的一致性、疾病出现时间仅依赖记录而非真实临床事件、对低发病率疾病的评估仍受样本量限制，以及未充分挖掘生成式模型的多步生成潜力。

---

## 495. VFIG: Vectorizing Complex Figures in SVG with Vision-Language Models

**arXiv ID:** 2603.24575 | [PDF](https://arxiv.org/pdf/2603.24575v1)

**作者:** Qijia He `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 13054 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对专业科研图表的光栅化版本，提出了一种基于视觉‑语言模型的自动化向 SVG 转换方法，并构建了大规模数据集和评估基准。

**💡 创新点**

创新点包括：① 构建 66K 对高质量科研图表‑SVG 对齐的数据集 VFig；② 设计粗细化训练课程——先在简单图表上进行监督微调，再利用渲染感知的强化学习进行结构细化；③ 提出结构感知的评估套件 VFig‑Bench，包含像素、组件和整体三层指标；④ 在公开模型中实现与 GPT‑5.2 规模相当的性能。

**🔧 技术方法**

核心技术：多模态 VLM（如 Qwen3‑VL、Gemini‑Pro）进行 LoRA 微调；两阶段训练：监督微调（SFT）+ 强化学习（GRPO）并结合渲染反馈；结构化 SVG 代码的语义化标记；使用视觉指标（SSIM、LPIPS、VisualSim）和 VLM 判分器评估。

**📊 数据集**

主要数据集：VFig（66K 真实与程序生成的科研图表‑SVG 对），以及 SVG‑Diagrams、Molmo2‑Diagram 等辅助数据；评估集包括 VFig‑Bench、Molmo2‑Diagrams、SVG‑Diagrams 三个基准。

**📈 对比分析**

与传统的追踪式矢量化工具（VTracer）、纯 LLM/SVG 生成基线（OmniSVG、StarVector）以及闭源大型 VLM（Gemini‑3、GPT‑5.2）比较，VFig‑SFT+RL 在结构性 VLM‑Judge、视觉相似度（LPIPS、VisualSim）和 SVG 可编辑性（Clean、Render）上均达到或接近最先进水平，VLM‑Judge 最高得分 0.829。

**⚠️ 局限性**

局限性：① 对于包含自然图像、复杂数学公式等非图形化内容的图表仍不适用；② 需要大量 GPU 资源进行训练和 RL 调优；③ 依赖精细过滤，若过滤标准松懈会影响数据质量；④ 对极其庞大或极高分辨率的多面板图表的细节捕捉仍有提升空间。

---

