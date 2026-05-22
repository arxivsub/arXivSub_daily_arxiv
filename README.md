# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-22 | 今日论文总数: 670

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Toward AI VIS Co-Scientists: A General and End-to-End Agent Harness for Solving Complex Data Visualization Tasks

**arXiv ID:** 2605.21825 | [PDF](https://arxiv.org/pdf/2605.21825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 2. Mind the Gaps: Multi-Robot Feedback-Driven Ergodic Coverage in Unknown Environments

**arXiv ID:** 2605.21719 | [PDF](https://arxiv.org/pdf/2605.21719v1)

**作者:** Thales Costa Silva `[一作]` (Brown University), Nora Ayanian `[通讯]` (Brown University)

**通讯引用:** 2516 | [OpenAlex ID](https://openalex.org/A5002752114)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种多机器人自适应覆盖方法，在未知环境中通过实时反馈更新目标分布并执行欧氏搜索，以实现持续采样。

**💡 创新点**

创新点在于将在线参数估计与传统欧氏搜索相结合，构造无需先验信息的自适应目标分布，并通过闭环控制保证估计收敛。

**🔧 技术方法**

采用基于高斯基函数的参数化环境模型、欧氏搜索算法、反馈控制律和自适应参数更新律，并使用Lyapunov理论证明收敛性。

**📊 数据集**

实验使用仿真数据：双峰静态高斯分布和双峰时间变动高斯分布，无真实外部数据集。

**📈 对比分析**

与固定均匀目标分布的欧氏搜索基线对比，利用RMSE评估模型误差，结果显示在静态和慢速变化环境下自适应方法显著优于基线；当目标移动速度较快时，基线表现更好。

**⚠️ 局限性**

主要限制包括对环境变化速度的时间尺度分离假设，快速变化环境下估计会失效；缺乏分布式实现与实地验证。

---

## 3. PEARL: Unbiased Percentile Estimation via Contrastive Learning for Industrial-Scale Livestream Recommendation

**arXiv ID:** 2605.21752 | [PDF](https://arxiv.org/pdf/2605.21752v1)

**作者:** Blake Gella `[一作]` (TikTok), Qinglei Wang `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对直播推荐系统中用户行为强度不均衡导致的偏差问题，提出了 PEARL 框架，通过对真实交互样本进行对比学习，直接估计相对百分位级别的偏好信号，避免绝对行为量级的偏差。

**💡 创新点**

创新点在于：①以非参数对比方式近似用户百分位分布，消除对分布估计模型的依赖；②提出多样本对比、值加权对比、预测引导的软化对比以及协同训练等扩展，使方法适用于稀疏、离散或连续多种目标；③在亿级别推荐系统中实现高效的 Reservoir Sampling 与梯度门控，确保实时性与稳定性。

**🔧 技术方法**

核心技术包括：对比学习（binary/multi-sample BCE）、百分位近似、值加权对比、预测引导的软化对比、协同训练（多任务学习）以及大规模系统实现技术（Reservoir Sampling、梯度门控）。

**📊 数据集**

数据集为公司直播推荐平台的生产流式数据，包含数十亿条交互记录，特征维度达数千维，覆盖用户、内容、上下文信息。

**📈 对比分析**

与原始回归、CQE、RAD 等基线相比，PEARL 在离线 UAUC 上提升 1.6%~16%（不同目标），在线 A/B 测试中在观看时长、消费额、互动率、举报率等关键指标分别提升 2.1%、0.8%、1.5% 与下降 6.9%。

**⚠️ 局限性**

局限性：需要足够的历史交互样本才能构建可靠的对比池；对极少交互的用户仍可能导致梯度不稳定；方法主要解决用户侧偏差，对物品或系统层面的偏差改进有限；实现对大规模实时系统有一定工程门槛。

---

## 4. NasZip: Software and Hardware Co-Design to Accelerate Approximate Nearest Neighbor Search with DIMM-Based Near-Data Processing

**arXiv ID:** 2605.21952 | [PDF](https://arxiv.org/pdf/2605.21952v1)

**作者:** Cheng Zou `[一作]` (Shanghai Jiao Tong University), Zhezhi He `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3380 | [OpenAlex ID](https://openalex.org/A5036755436)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于DIMM近数据处理的软硬件协同加速方案，用于高维向量数据库的近似最近邻搜索（ANN）

**💡 创新点**

创新点包括：①统计式PCA引导的特征级早期退出（FEE-sPCA）能在距离计算前更早剔除无关向量；②动态浮点（Dfloat）压缩能在保持召回率的前提下减少每次内存访问的数据位宽；③数据感知邻居列表映射（DaM）将邻居信息与向量同属一个子通道，降低跨通道通信；④局部邻居缓存（LNC）与预取机制提升邻居列表局部性利用；⑤整体软硬件协同优化显著提升吞吐量与能效。

**🔧 技术方法**

使用的技术包括：PCA特征降维与统计估计、早期退出控制、动态浮点数编码、DIMM内存内向量处理单元（VPE）、子通道数据映射与缓存、批量预取与负载均衡调度、FPGA实现与周期精确仿真。

**📊 数据集**

实验数据集覆盖常用ANN基准与检索语料库：SIFT、GIST、BigANN、GloVe、Wiki、MS_MARCO（维度从128到960，规模从百万到十亿向量）。

**📈 对比分析**

与多种平台基线对比：CPU（HNSW、SCANN）、GPU（CAGRA、A100）、ASIC（ANNA）、FPGA（DF‑GAS）、UPMEM PIM（PIMANN）、现有NDP（ANSMET）。在保持相同召回率的条件下，方案在CPU基线上实现8.4×加速，在GPU上1.4×加速，在ANSMET上1.69×加速；能效提升可达1.5×，并在多维度数据集（如GIST）上实现最大加速。

**⚠️ 局限性**

局限性包括：①PCA预处理与查询时的转换对极大规模更新的数据集仍有一定的时间与计算开销；②方案主要针对基于图的ANN，若需支持量化或树/哈希方法则需额外改造；③对极高维向量（>2000维）或非欧氏距离的适配尚未验证；④硬件实现需要专用DIMM与NDP逻辑，部署成本相对较高。

---

## 5. Manifold-Guided Attention Steering

**arXiv ID:** 2605.21770 | [PDF](https://arxiv.org/pdf/2605.21770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 6. A Reproducible Log-Driven AutoML Framework for Interpretable Pipeline Optimization in Healthcare Risk Prediction

**arXiv ID:** 2605.21528 | [PDF](https://arxiv.org/pdf/2605.21528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 7. Latent-space Attacks for Refusal Evasion in Language Models

**arXiv ID:** 2605.21706 | [PDF](https://arxiv.org/pdf/2605.21706v1)

**作者:** Giorgio Piras `[一作]` (University of Cagliari), Battista Biggio `[通讯]` (University of Cagliari)

**通讯引用:** 8930 | [OpenAlex ID](https://openalex.org/A5008367647)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Controlled Latent‑Space Evasion (CLE)”机制，用于通过在内部表示层上施加精确的投影或加性扰动来抑制安全对齐语言模型的拒绝行为。

**💡 创新点**

创新点在于将拒绝抑制重新表述为对线性探测器的潜在空间逃逸攻击，并在此基础上引入可调置信度（margin）以将表示推到合规区间而非仅仅投射到决策边界；同时提出两种变体（CLE‑P和CLE‑A）以及基于贝叶斯优化的层与margin选择策略。

**🔧 技术方法**

主要技术包括：
- 线性探测器（SVM/线性S）训练用于区分“被拒绝”与“被答复”激活；
- 在每层施加的扰动形式为 δ = –α·w，其中 α 由 λ_l·(probe(z)+m_l) 控制；
- 贝叶斯优化搜索 λ_l（层掩码）和 m_l（margin）两组超参数；
- 对比实验中使用 LLM-as-a-judge 作为验证 oracle。

**📊 数据集**

使用 128 条有害指令与 128 条无害指令（来自各类公开数据集）训练探测器；在 15 个指令调优/多模态/推理模型上评估；验证集采用官方 Jailbreak / 对齐数据集。

**📈 对比分析**

与传统拒绝抑制基线（MD、RDO、PS）以及 prompt‑level jailbreak（GCG、SAA）比较，CLE‑P 与 CLE‑A 在 15 个模型上平均攻击成功率提升至 87% 以上，远高于 1–20% 的基线；在单个提示上每次仅需一次前向传播，效率显著优于 GCG/SAA 所需的千级前向。

**⚠️ 局限性**

局限性包括：
- 依赖贝叶斯优化，需额外前向推理成本；
- 仅针对线性可分的拒绝表示有效，若对齐过程抑制线性分离则攻击效果下降；
- 当前方法针对单一模型，未考察跨模型迁移或对抗性鲁棒性。

---

## 8. FuzzingBrain V2: A Multi-Agent LLM System for Automated Vulnerability Discovery and Reproduction

**arXiv ID:** 2605.21779 | [PDF](https://arxiv.org/pdf/2605.21779v1)

**作者:** Ze Sheng `[一作]` (Texas A&M University), Jeff Huang `[通讯]` (Texas A&M University)

**通讯引用:** 3632 | [OpenAlex ID](https://openalex.org/A5052381120)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了FuzzingBrain V2，一套基于多智能体的大语言模型系统，自动化发现并复现 C/C++ 软件漏洞；

**💡 创新点**

引入 Suspicious Point 抽象，结合层次化搜索与双层 fuzzing，解决了 LLM 低召回率、粒度不当及跨函数依赖难题；

**🔧 技术方法**

使用 Google OSS‑Fuzz 作为验证后端、Model Context Protocol（MCP）实现多智能体通信、LLM 推理（Claude、GPT 等）、控制流分析、双层 fuzzing 与动态静态混合分析；

**📊 数据集**

在 AIxCC 2025 Final Challenge 的 C/C++ 数据集（40 个漏洞，12 项开源项目）以及 19 个 OSS‑Fuzz 实际项目中评估；

**📈 对比分析**

在 AIxCC 基准上实现 90% 检测率（36/40），比冠军 Team Atlanta 提升 157%，在真实项目中发现 41 个零日漏洞，26 个已确认并修复；

**⚠️ 局限性**

局限在于仅支持单输入攻击、难以处理隐式状态机、多输入协同漏洞、以及构建环境复杂导致的工具兼容性问题。

---

## 9. Optimal $e^{(γ+o(1))n}$-Approximation of the Permanent of Positive Semidefinite Matrices

**arXiv ID:** 2605.21946 | [PDF](https://arxiv.org/pdf/2605.21946v1)

**作者:** Nima Anari `[一作]` (Stanford University), Farzam Ebrahimnejad `[通讯]` (Voleon Group)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文确定了 Hermitian 正半定矩阵行列式（永久）的最佳确定性多项式时间近似比例，给出了精确的指数级 e^γ n 上下界；

**💡 创新点**

创新点在于提出熵修正的变分松弛，将 Wick 表示与 Gibbs 变分原理相结合，得到与 NP‑hardness 匹配的最优指数常数 γ；

**🔧 技术方法**

主要技术包括 Wick–Isserlis 公式、Gibbs 变分原理、最大熵原理以及凸优化（内部点法）求解；

**📊 数据集**

该工作不涉及实验数据集，全部为理论证明；

**📈 对比分析**

与之前的 e^(γ+1) n、e^(γ+0.9999) n 结果相比，本文提供了匹配的上、下界，确立了最优指数常数 γ；

**⚠️ 局限性**

局限性在于仅适用于正半定 Hermitian 矩阵，需在有限精度下实现，缺乏实验验证且对其他矩阵类型不适用。

---

## 10. Beyond Scalar Objectives: Expert-Feedback-Driven Autonomous Experimentation for Scientific Discovery at the Nanoscale

**arXiv ID:** 2605.21820 | [PDF](https://arxiv.org/pdf/2605.21820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. Safeguarding Text-to-Image Generative Models Against Unauthorized Knowledge Distillation

**arXiv ID:** 2605.22060 | [PDF](https://arxiv.org/pdf/2605.22060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 12. HyLoVQA: Dynamic Hypernetwork-Generated Low-Rank Adaptation for Continual Visual Question Answering

**arXiv ID:** 2605.22035 | [PDF](https://arxiv.org/pdf/2605.22035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 13. Non-Contact Vibration-Based Damage Detection of Civil Structures Using a Cost-Effective Autonomous UAV

**arXiv ID:** 2605.21914 | [PDF](https://arxiv.org/pdf/2605.21914v1)

**作者:** Javier Becerril `[一作]` (University of Texas at Rio Grande Valley), Qi Lu `[通讯]` (University of Texas at Rio Grande Valley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种低成本自主无人机配合视觉运动跟踪的非接触振动测量方法，用于结构健康监测与损伤检测。

**💡 创新点**

创新在于将基于AprilTag的自律对准系统与低成本摄像头相结合，提供一种无需GPS且成本仅10-20%商业无人机的可扩展SHM方案。

**🔧 技术方法**

使用光学运动跟踪（DIC/运动跟踪软件）、多平台实验（智能手机、USB摄像机、定制无人机）、频域分析、COMSOL有限元模型验证等技术。

**📊 数据集**

采用实验室一层框架结构的振动数据，包括健康与通过加重模拟损伤的状态，记录了多平台视频与加速度计数据。

**📈 对比分析**

通过将无人机、手机、USB摄像机与加速度计及FE模型的自然频率结果进行对比，发现无人机误差约5–6%，手机及USB误差≤1.8%，均能准确捕捉频率偏移，证明低成本无人机在检测趋势方面可靠。

**⚠️ 局限性**

受限于无人机悬停的机械振动、姿态漂移、低分辨率相机以及短时自由振动记录，导致精度高于固定相机的误差增大，并且仅验证了人工加重模型，未覆盖真实裂缝等结构损伤。

---

## 14. Bifunction and Interlevel Delaunay Trifiltrations

**arXiv ID:** 2605.21636 | [PDF](https://arxiv.org/pdf/2605.21636v1)

**作者:** Ángel Javier Alonso `[一作]` (CUNEF Universidad), Abhishek Rathod `[通讯]` (Ben Gurion University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

论文提出了一种适用于二维标量函数的三参数 Delaunay 过滤器（trifiltration），并给出了其大小与时间复杂度的理论上界；随后实现了该过滤器的构造算法并在多种噪声点云上进行实验验证。

**💡 创新点**

创新点在于：① 将先前针对单参数或双参数（子水平、子水平-Čech）Delaunay 过滤器的构造延伸到三参数；② 通过“增量 Delaunay 复杂体”和冲突对/三元组的概念，构造出同等弱等价的 offset 三参数过滤器；③ 引入局部与非局部两种优化策略，并证明其在某些输入下可将插入/删除次数从 O(n²) 降至 O(n)。

**🔧 技术方法**

主要技术包括：Bowyer‑Watson 递增 Delaunay 算法、凸优化求解最小包围球、冲突对/三元组的本质分析、对增量 Delaunay 复杂体的构造、半代数集合的好覆盖与范畴化的 Nerve 定理、以及针对多参数持续同调的弱等价证明。

**📊 数据集**

实验数据集为：二维圆、单位正方形、三维球面、圆环、单位立方体等噪声点云（500、1000、2000、4000、8000、16000 个点），并针对四种标量函数（编码密度、L1-中心性、高度、随机）计算交互层 Delaunay‑Čech 三参数过滤器。

**📈 对比分析**

比较方法：对同一数据集同时使用局部与非局部两种实现，记录内存占用、过滤器大小和运行时间；实验表明内存与大小几乎线性增长，运行时间近似二次增长；局部算法在除交互层外的三种场景下速度提升 3–14 倍，随着数据规模增大提升幅度增加。

**⚠️ 局限性**

局限性：① 最坏情况下时间与插入/删除次数的上界未下降；② 实现未并行化，无法充分利用多核计算；③ 仅适用于二维标量函数，扩展到更高维参数仍是挑战；④ 需要点云处于一般位置，实际数据需预处理；⑤ 在极大规模或高维输入下，复杂度仍呈指数增长。

---

## 15. Prototype-Guided Classification Sub-Task Decoupling Framework: Enhancing Generalization and Interpretability for Multivariate Time Series

**arXiv ID:** 2605.22055 | [PDF](https://arxiv.org/pdf/2605.22055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. Correcting Class Imbalance in Prior-Data Fitted Networks for Tabular Classification

**arXiv ID:** 2605.21742 | [PDF](https://arxiv.org/pdf/2605.21742v1)

**作者:** Samuel McDowell `[一作]` (Arizona State University), Lalitha Sankar `[通讯]` (Arizona State University)

**通讯引用:** 3350 | [OpenAlex ID](https://openalex.org/A5065366998)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在Prior-Data Fitted Networks（PFNs）中针对类别不平衡问题进行修正。

**💡 创新点**

提出利用PFNs良好的校准特性，阈值调整和下采样可简单有效地缓解少数类性能下降。

**🔧 技术方法**

主要技术包括阈值调整（thresholding）、下采样、过采样和合成上采样，并对TabPFN-2.5模型进行评估。

**📊 数据集**

使用OpenML-CC18中的11个二分类数据集，训练集为500少数类+950多数类样本，测试集为每类500样本。

**📈 对比分析**

通过与未修正基线、下采样、过采样及合成上采样对比，阈值调整在均衡准确率和最差类准确率上均表现最佳。

**⚠️ 局限性**

局限在于仅研究二分类平衡测试集，未覆盖多分类及更大规模真实不平衡场景，且仅基于现有PFN架构。

---

## 17. Safe and Steerable Geometric Motion Policies for Robotic Dexterous Manipulation

**arXiv ID:** 2605.21811 | [PDF](https://arxiv.org/pdf/2605.21811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 18. DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling

**arXiv ID:** 2605.21603 | [PDF](https://arxiv.org/pdf/2605.21603v1)

**作者:** Yi Pan `[一作]` (University of Washington), Stephanie Wang `[通讯]` (University of Washington)

**通讯引用:** 108489 | [OpenAlex ID](https://openalex.org/A5114376519)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了可插拔的框架 DynaFlow，用于在现有 ML 系统中透明、灵活地集成各种内设备并行策略。

**💡 创新点**

通过将模型定义与执行调度解耦，提供注解式前端和可编程调度器，并在后端实现异步控制、零拷贝内存和对 CUDA Graph / TorchInductor 的兼容，从而大幅降低集成成本并支持多样化动态策略。

**🔧 技术方法**

利用 TorchDynamo 抓取图、Python 注解划分子图、异步后台调度、系统级内存预分配、CUDA Graph 捕获、TorchInductor 编译以及 CUDA 流/绿色上下文等技术。

**📊 数据集**

在 Llama‑3 系列、Qwen‑2、DeepSeek‑V2‑Lite、Mixtral 等模型上，使用 ShareGPT、LMSYS‑Chat‑1M、Splitwise 等文本和视频生成数据集进行评测。

**📈 对比分析**

与原始系统和各自手工实现的内设备并行版本做对比，平均仅需少于 100 行代码即可集成；在 6 大系统中获得最高 1.29 倍吞吐量提升，且大多数工作负载可匹配或超过现有专门实现。

**⚠️ 局限性**

对极大规模模型或低批次工作负载时，动态拆分和多图捕获会带来额外的 CPU 与内存开销；某些策略在轻量级任务上可能导致性能下降；实现依赖 PyTorch 2.6+，并且对自定义 kernel 的支持仍有限。

---

## 19. PeakFocus: Bridging Peak Localization and Intensity Regression via a Unified Multi-Scale Framework for Electricity Load Forecasting

**arXiv ID:** 2605.21550 | [PDF](https://arxiv.org/pdf/2605.21550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 20. Bounding-Box Trajectories Matter for Video Anomaly Detection

**arXiv ID:** 2605.21957 | [PDF](https://arxiv.org/pdf/2605.21957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 21. Lost in the Prefix: Revisiting IP Geolocation Accuracy Across Networks and Geographies

**arXiv ID:** 2605.21937 | [PDF](https://arxiv.org/pdf/2605.21937v1)

**作者:** Syed Tauhidun Nabi `[一作]` (Virginia Tech), Shaddi Hasan `[通讯]` (Virginia Tech)

**通讯引用:** 1325 | [OpenAlex ID](https://openalex.org/A5004946056)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对四大IP地理定位数据库在移动网络和全球南方地区的定位精度进行了大规模评估。

**💡 创新点**

创新点在于首次将UNICEF Giga校内测量作为真实地理标注基准，全面覆盖移动网络和全球南方，并揭示了前缀粒度与定位误差的内在关系。

**🔧 技术方法**

使用了基于IP前缀粒度的误差分析、BGP前缀对比、地理距离计算等技术。

**📊 数据集**

数据集包括RIPE Atlas探针数据、UNICEF Giga校内测量、四大数据库（MaxMind GeoLite2、IPinfo、IP2Location、DB-IP）以及CAIDA RouteViews BGP前缀表。

**📈 对比分析**

通过对比误差分布、失败率以及前缀粗细对比，发现移动网络的误差是固定网络的十倍，全球南方的失败率是全球北方的三倍，所有提供商表现一致。

**⚠️ 局限性**

局限在于基准坐标误报、缺乏长期时序评估、对IPv6前缀分析有限、以及大部分移动网络观测被归入“Both”类别，导致结果可能偏低。

---

## 22. A Dataset of Reproducible Flaky-Test Failures

**arXiv ID:** 2605.21677 | [PDF](https://arxiv.org/pdf/2605.21677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 23. TONIC: Token-Centric Semantic Communication for Task-Oriented Wireless Systems

**arXiv ID:** 2605.21553 | [PDF](https://arxiv.org/pdf/2605.21553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 24. Symbolic Density Estimation for Discrete Distributions

**arXiv ID:** 2605.21813 | [PDF](https://arxiv.org/pdf/2605.21813v1)

**作者:** Ziwen Liu `[一作]` (Rice University), Meng Li `[通讯]` (Rice University)

**通讯引用:** 12304 | [OpenAlex ID](https://openalex.org/A5044185075)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

自动从离散样本中发现并解析闭式概率质量函数（PMF）

**💡 创新点**

在对数域内引入符号回归与PMF有效性校验的结构先验，能够无监督识别零膨胀与有限混合等复杂模型

**🔧 技术方法**

利用进化符号回归、结构化复杂度先验、加权最小二乘拟合及后验校正，实现符号表达式搜索与参数细化

**📊 数据集**

使用自建的SDEBench基准（14类常见离散分布及其混合/零膨胀变体）以及真实PBMC单细胞RNA‑seq数据进行验证

**📈 对比分析**

与传统MoM/MLE、KDE、Pyro、PySR等基线比较，SDE在符号准确率、参数误差和MSE方面均保持与或优于现有方法，并在复杂组合模型上表现更佳

**⚠️ 局限性**

仅适用于单变量离散分布；搜索空间巨大导致对极其复杂模型的计算时间较长；缺乏对符号模型选择的理论不确定性量化

---

## 25. EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models

**arXiv ID:** 2605.21931 | [PDF](https://arxiv.org/pdf/2605.21931v1)

**作者:** Shiqi Huang `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**通讯引用:** 6347 | [OpenAlex ID](https://openalex.org/A5024709593)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向视频的自演进框架，利用无标注视频通过 Temporal‑aware Questioner 与 Temporal‑grounded Solver 的自我对弈实现视频理解与推理的自我提升。

**💡 创新点**

创新点包括：① 设计了时间扰动敏感的 Questioner 奖励，使问题生成更依赖时间顺序；② 引入基于视频窗口的 Solver 奖励，自动生成时段监督，实现无标签的时序定位；③ 两种奖励协同优化，形成完整的时序自演进闭环。

**🔧 技术方法**

技术手段：自演进的 Questioner–Solver 自我对弈；Group Relative Policy Optimization（GRPO）用于策略优化；Temporal perturbation（随机、逆序、块随机）与 IoU 时段定位奖励；LoRA 微调、128 帧 128×14×14 视觉输入等。

**📊 数据集**

训练集：LLaVA‑Video‑178K、STAR、CLEVRER、PerceptionTest、NeXT‑QA、Video‑Holmes 共 5,778 条未标注视频；评估集：Video‑Holmes、VSIBench、VideoMMMU、MMVU、TempCompass、VideoMME 六大视频推理/理解基准。

**📈 对比分析**

与监督微调（SFT）、监督 RL（vanilla GRPO）以及原始自演进 baseline 对比，平均提升约 1.4%（从 48.6% 提升至 50.0%），在 VideoMMMU、MMVU、TempCompass 等基准上甚至超过监督方法，证明无标注自演进可获得竞争性能。

**⚠️ 局限性**

局限性：对小规模模型（如 Qwen2.5‑VL‑3B）迭代不稳定；对窗口大小和扰动策略有一定敏感性；仍需大量 GPU 资源；目前只关注时间维度，未涵盖更复杂的因果推理或跨模态关联；以及需依赖已有的预训练视频 LLM 基础。

---

## 26. DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models

**arXiv ID:** 2605.21539 | [PDF](https://arxiv.org/pdf/2605.21539v1)

**作者:** Xuyang Zhong `[一作]` (City University of Hong Kong), Chen Liu `[通讯]` (City University of Hong Kong)

**通讯引用:** 472459 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DualOptim+ 这一可插拔的优化框架，用于改进大型语言模型（LLM）的机器无学习（unlearning）效果。

**💡 创新点**

创新点包括：① 通过共享基底状态（base）和任务特定增量状态（delta）实现共享与解耦状态的自适应桥接；② 将该框架通用化到任何带状态的优化器；③ 推出 8bit 量化版本显著降低内存占用。

**🔧 技术方法**

采用的技术主要有：基于 AdamW 的两层状态更新（momentum 与二阶矩）；方向冲突自适应更新；梯度残差分解；量化优化器状态；在多目标学习、目标安全对齐、迁移学习等场景下进行评估。

**📊 数据集**

使用的数据集与模型包括：TOFU 虚拟作者数据集（forget01/05/10）、真实作者与世界事实集；LLM 模型 Phi‑1.5‑1.3B、Llama‑2‑7B、Llama‑3‑8B‑Instruct；下游任务涵盖 ARC‑c、MMLU、TruthfulQA、TriviaQA、GSM8K；以及安全对齐用的 Alpaca‑SFT 与多类危险指令集。

**📈 对比分析**

实验对比方法包括 Joint、Alternate、DualOptim（DO）以及各自的 8bit 版本。结果显示：DualOptim+ 在大多数无学习任务中实现了更优的忘记效能与模型效用平衡；8bit 版本在保持性能的同时将内存占用降低至 1/4（相较于单一 Adam）。在安全对齐与多任务学习中亦获得明显提升。

**⚠️ 局限性**

局限性：① 需要额外的 optimizer 状态，导致内存成本增加；② 对于极端无学习场景或梯度冲突度极低的任务，提升有限；③ 量化方式在某些设置下可能略有性能下降；④ 仍需针对不同任务与模型进行超参数调优，未给出通用最佳配置。

---

## 27. ArabDiscrim: A Decade-Long Arabic Facebook Corpus on Racism and Discrimination

**arXiv ID:** 2605.22081 | [PDF](https://arxiv.org/pdf/2605.22081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 28. Tabular foundation models for robust calibration of near-infrared chemical sensing data

**arXiv ID:** 2605.21544 | [PDF](https://arxiv.org/pdf/2605.21544v1)

**作者:** Robin Reiter `[一作]` (CIRAD), Gregory Beurier `[通讯]` (CIRAD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在近红外光谱（NIR）预测任务中对TabPFN进行大规模基准测试，并将其与传统化学计量学方法（如PLS）、经典机器学习方法（Ridge、CatBoost）以及一维卷积神经网络进行对比；

**💡 创新点**

提出将TabPFN与预处理搜索框架相结合的校准策略，从而在无需额外超参数调优的情况下提升NIR化学传感的预测性能；

**🔧 技术方法**

使用TabPFN、PLS、Ridge、CatBoost、CNN-1D等模型，配合Savitzky–Golay、SNV、ASLS、OSC等光谱预处理方法；

**📊 数据集**

基准数据集包含66个NIR数据集，涵盖54个回归任务和12个分类任务，样本量从56到8731，特征维度从125到4200；

**📈 对比分析**

通过统一的三折交叉验证和独立测试集评估，统计RMSEP/ACC及相对改进，结果显示TabPFN-optimized在回归中平均排名最佳，分类中与PLS‑DA相近，且在小到中等样本量数据集上表现优异；

**⚠️ 局限性**

局限包括：超参数搜索不均衡（如CNN‑1D受限于训练预算）、分类任务数据集数量有限、未针对光谱特性进行专门预训练，以及对光谱异常值和外推域的鲁棒性仍需改进。

---

## 29. On weighted partial triangulations of convex polygons

**arXiv ID:** 2605.21921 | [PDF](https://arxiv.org/pdf/2605.21921v1)

**作者:** Antonio Blanca `[一作]` (Pennsylvania State University), Izabella Stuhl `[通讯]` (Pennsylvania State University)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5035002320)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究凸多边形的加权局部三角剖分的精确采样问题。

**💡 创新点**

提出一种期望时间为 O((n√λ+1)·log n) 的随机算法，近似达到最优采样复杂度，并给出分区函数的精确上下界。

**🔧 技术方法**

综合组合计数、二项式估计、Remy 算法、Floyd 子集采样、Gaussian 近似等技术实现采样。

**📊 数据集**

不依赖外部数据集，完全基于理论分析和组合公式。

**📈 对比分析**

相较于传统的马尔可夫链方法，省去混合时间分析，直接得到最优期望时间；实验上实现速度至少提升数十倍。

**⚠️ 局限性**

对 λ 的取值范围做了严格假设（c/n² ≤ λ ≤ C），在极端 λ 极大或极小的边界情况需进一步改进；算法对非常大 n 仍存在 O(log n) 乘子。

---

## 30. Simulating Learners' Task-Selection Strategies and System Constraints in Mastery Learning

**arXiv ID:** 2605.21613 | [PDF](https://arxiv.org/pdf/2605.21613v1)

**作者:** Haley Noh `[一作]` (New York University), Conrad Borchers `[通讯]` (Carnegie Mellon University)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5037442366)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于真实学习者交互数据构建仿真框架，研究学习者任务选择策略与系统约束对精通学习效率的影响。

**💡 创新点**

首次将仿真与真实数据相结合，系统性评估不同任务选择策略与系统约束的交互效应，并提出针对性约束策略减少过度练习。

**🔧 技术方法**

使用Additive Factors Model (AFM) 与 Bayesian Knowledge Tracing (BKT) 进行学习过程仿真，并实现多种任务选择与约束策略。

**📊 数据集**

使用两套公开的中学/高中数学 ITS 日志数据：线性方程求解（PSLC DataShop）和图形解释（MathTutor）

**📈 对比分析**

通过比较不同策略-约束组合下的过度练习指标（overpractice），结果显示弱点导向策略效率最高，而最小化最坏情况损失策略在无约束下效率最低；添加任务/问题约束能显著降低其过度练习，接近其他高效策略。

**⚠️ 局限性**

限制了策略数量（未考虑动态适应）、未建模动机/元认知等因素，仅采用任务优先的设置，且未覆盖所有可能的任务选择模式。

---

## 31. Zero-shot adaptation to order book dynamics

**arXiv ID:** 2605.21707 | [PDF](https://arxiv.org/pdf/2605.21707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 32. MM-Conv: A Multimodal Dataset and Benchmark for Context-Aware Grounding in 3D Dialogue

**arXiv ID:** 2605.21796 | [PDF](https://arxiv.org/pdf/2605.21796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 33. The Impact of AI Usage and Informativeness on Skill Development in Logical Reasoning

**arXiv ID:** 2605.21695 | [PDF](https://arxiv.org/pdf/2605.21695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 34. GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation

**arXiv ID:** 2605.22036 | [PDF](https://arxiv.org/pdf/2605.22036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. Adversarial Reframing: A Framework for Targeted Generation in Language Models

**arXiv ID:** 2605.21674 | [PDF](https://arxiv.org/pdf/2605.21674v1)

**作者:** Shahnewaz Karim Sakib `[一作]` (University of Tennessee at Chattanooga), Anindya Bijoy Das `[通讯]` (University of Akron)

**通讯引用:** 841 | [OpenAlex ID](https://openalex.org/A5029951257)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了THREAT框架，利用迭代式LLM生成和语义过滤实现对语言模型的定向生成攻击，帮助发现潜在的安全漏洞。

**💡 创新点**

创新点在于将破解过程形式化为非凸优化问题，并通过多模型协同、语义相似度约束与奖励安全增益的闭环迭代，显著提高攻击成功率且降低计算成本。

**🔧 技术方法**

核心技术包括LLM驱动的变体生成、BERT嵌入语义相似度评估、基于安全评分函数f_safe的奖励计算以及迭代优化搜索。

**📊 数据集**

实验数据集涵盖HarmfulQA、Gretel Safety Alignment（Discrimination、Information Hazard、System Risks）等安全基准。

**📈 对比分析**

与现有的白盒/黑盒破解方法比较，THREAT在拒绝率、攻击成功率和查询成本上均优于对标方法，展示了更高的攻击精度和更低的资源消耗。

**⚠️ 局限性**

局限性包括对LLM访问权限和安全评估器的依赖、阈值调参敏感、在面对动态更新的安全过滤器时可能需要重新调优以及对非文本模态的适用性尚未验证。

---

## 36. Frequency-Domain Regularized Adversarial Alignment for Transferable Attacks against Closed-Source MLLMs

**arXiv ID:** 2605.21541 | [PDF](https://arxiv.org/pdf/2605.21541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 37. Visual-Advantage On-Policy Distillation for Vision-Language Models

**arXiv ID:** 2605.21924 | [PDF](https://arxiv.org/pdf/2605.21924v1)

**作者:** Ruiqi Liu `[一作]` (Institute of Automation, CAS), Shu Wu `[通讯]` (Institute of Automation, CAS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出视觉优势（VA）指标，利用教师在有无细节图像条件下的概率差，改进视觉-语言模型的 on‑policy 知识蒸馏，形成 Visual-Advantage On-Policy Distillation (VA‑OPD)。

**💡 创新点**

创新点在于：①将 VA 作为 token‑level 视觉依赖度量，②在两层粒度上加权：对整条 roll‑out 进行 VA 归一化的软权重，③对 token 进行高 VA 与低 VA 组别分别计算 KL，从而避免视觉监督被语言模板稀释。

**🔧 技术方法**

技术手段包括：on‑policy 蒸馏、教师对比实验（使用降解图像计算 VA）、roll‑out 级别的 softmax 重加权、token 组别 KL 均值、以及标准反向 KL 目标。

**📊 数据集**

使用 Qwen3‑VL 族教师模型（4B、8B、32B）与学生 2B，训练数据为 Geometry3K（≈2.1k 题）与 ViRL39K（≈39k 题），并在 WeMath、MathVista、MathVerse、HallusionBench、AI2D、MMMU、MMStar、OCRBench 八个基准上评测。

**📈 对比分析**

与 Base、CoT‑SFT、Off‑policy KD、GRPO、PAPO、Standard OPD 等基线对比，VA‑OPD 在所有 8 个基准上均优于 Standard OPD，且提升随教师规模和训练数据量增大而显著加剧（如 Math Avg +2.9→+3.8，Visual Avg +1.5→+2.5）。

**⚠️ 局限性**

局限性：VA 只衡量教师对细节图像的敏感度，无法直接反映模型内部的注意力或感知机制；在可仅凭文本推断的任务（如 MMStar）提升有限；对极其稀疏或低质量视觉信息的鲁棒性尚待验证。

---

## 38. Single-Item Auctions with a Monopolist Intermediary

**arXiv ID:** 2605.21934 | [PDF](https://arxiv.org/pdf/2605.21934v1)

**作者:** Jingyi Liu `[一作]` (Princeton University), Qianfan Zhang `[通讯]` (Princeton University)

**通讯引用:** 123 | [OpenAlex ID](https://openalex.org/A5012496138)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在单个物品拍卖中，独占性中介通过控制信息流向卖家，如何影响卖家和中介的收益结构，并在三种时间模型（卖家先行、平台先行、同时决策）下给出逼近保证与不可行性结果。

**💡 创新点**

首次将单个独占性中介的访问控制纳入单项拍卖理论，揭示其导致的“加价”效应，并证明不同先后顺序下的收益差异和无上界的Simultaneous Nash极差，提出对常规分布与α-强正则分布的精确常数逼近。

**🔧 技术方法**

运用Myerson最优拍卖理论、虚值变换、最小支付可达性分析、Stackelberg博弈与Nash均衡理论，以及对正则分布的虚值逆函数与强正则性参数的数学推导。

**📊 数据集**

本研究为理论分析，不涉及具体数据集；所有结果均基于独立私有价值模型与分布假设。

**📈 对比分析**

与传统无中介拍卖比较，发现卖家先行时可实现常数比例的最优收益（对α-强正则分布）；平台先行在MHR分布下也可获得1/e(1-1/e)比例的收益；同时决策则可能导致双方收益几乎为零。

**⚠️ 局限性**

局限包括仅考虑单件拍卖、独占性中介、以及对分布的正则性假设；随机化机制在本模型中无优势；实际多物品、多中介场景仍待扩展。

---

## 39. Same Pipeline, Opposite Conclusions: Sample-Surface Effects in Breaking-News Latency

**arXiv ID:** 2605.21521 | [PDF](https://arxiv.org/pdf/2605.21521v1)

**作者:** Farhad Bazyari `[一作]` (TWG AI), Sean Moran `[通讯]` (TWG AI)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对两类事件样本（WCEP和Polymarket）使用统一的下游管道，测量X（原Twitter）与新闻源在首次公开信息时的时延，并比较不同平台的领先情况。

**💡 创新点**

创新点在于：①采用跨表面（cross‑surface）设计，揭示结果随事件来源不同而变化；②将预测市场的交易量峰值作为事件时刻的客观标注，提供更精细的时间窗；③通过LLM辅助的布尔查询和多渠道验证，提升检索精度。

**🔧 技术方法**

技术手段包括：LLM驱动的布尔查询草拟、商业社交监听API（Brandwatch）提取、X雪花ID解析发布时间、oEmbed接口进行主题验证，以及对提取特征的统计分析。

**📊 数据集**

使用的数据集为：①WCEP（Wikipedia Current Events Portal）中的美国相关事件并按页面浏览量排序；②Polymarket预测市场的交易记录，按交易量筛选并以1小时峰值标记事件时间。

**📈 对比分析**

比较方法：在两组样本中执行相同的事件采样、特征提取、查询生成、数据拉取和主题验证流程。结果显示：Sample A中新闻平均领先X 21.6分钟；Sample B中两者时间相近，X领先比例仅38%。通道领先分布也因样本而显著不同，表明平台领先并非单一结论。

**⚠️ 局限性**

局限性包括：①样本量有限（Sample A仅6对事件，Sample B 16对），难以进行细粒度统计；②商业监听API覆盖率不足，导致24%事件无相关证据；③对非英文事件或非预测市场事件的泛化性受限；④LLM生成的布尔查询可能引入偏差。

---

## 40. Optimal Guarantees for Auditing Rényi Differentially Private Machine Learning

**arXiv ID:** 2605.21938 | [PDF](https://arxiv.org/pdf/2605.21938v1)

**作者:** Benjamin D. Kim `[一作]` (University of Illinois Urbana Champaign), Daniel Alabi `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于Donsker–Varadhan变分表示的黑盒隐私审计框架，用来直接估计机器学习算法（如DP‑SGD）所声称的Rényi差分隐私（RDP）参数；

**💡 创新点**

创新点在于：①将RDP审计视为假设检验与统计估计问题；②使用DV变分式构造可计算的下界估计器；③给出显式非渐近置信区间并证明其在样本复杂度上达到信息理论下界；

**🔧 技术方法**

技术上采用了Donsker–Varadhan变分公式、神经网络实现的Critic函数、指数移动平均与小批量梯度优化、以及假设检验构造的置信区间；

**📊 数据集**

实验数据集为MNIST和CIFAR‑10，使用卷积神经网络进行DP‑SGD训练；

**📈 对比分析**

与现有最先进的黑盒RDP审计方法（如基于攻击的审计）比较，本文在α=1.25和α=2.0下的RDP下界均显著提升，尤其在低至中等隐私参数区间内表现更好；

**⚠️ 局限性**

局限性包括：①仅提供下界置信区间，缺乏上界估计；②适用于α>1且主要针对小至中等α值，α较大时方差偏大；③需训练神经网络审计器，计算开销相对较高；④未扩展到交互式或分布式隐私审计场景。

---

## 41. Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization

**arXiv ID:** 2605.21751 | [PDF](https://arxiv.org/pdf/2605.21751v1)

**作者:** Zhiqi Gao `[一作]`, Frederic Sala `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Text2Opt-Bench，一个包含12类、规模可达1000+变量、solver-verified的文本到优化基准；

**💡 创新点**

发现并证明“绑定”是当前LLM文本到优化的主要瓶颈，并提出BIND方法、并演示训练专门绑定模型比端到端更高效；

**🔧 技术方法**

采用前向工程+Gurobi验证的生成管线、BIND数据外部化、模型细化（SFT与RL对比）、RULER长上下文检索实验；

**📊 数据集**

使用Text2Opt-Bench自生成的约束和实例数据（LP、MILP、MIQP、NLP等）以及公开的RULER任务；

**📈 对比分析**

在多家顶尖模型上评估，BIND提升GPT-5-Nano 23.3pp、GPT-5 9.6pp，BIND在测试时可匹配或超越pass@5与迭代修复，且成本更低；训练绑定专家在7B规模下优于端到端SFT，参数更高效；

**⚠️ 局限性**

局限在于仅覆盖可由Gurobi求解的数学规划，未包含启发式组合优化；BIND假设已外部化结构化数据，未解决未结构化文档中的数据提取；

---

## 42. $\textit{BlockFormer}$ : Transformer-based inference from interaction maps

**arXiv ID:** 2605.21617 | [PDF](https://arxiv.org/pdf/2605.21617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 43. RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator

**arXiv ID:** 2605.21748 | [PDF](https://arxiv.org/pdf/2605.21748v1)

**作者:** Zhenwei Tang `[一作]` (Layer 6 AI), Jesse C. Cresswell `[通讯]` (Layer 6 AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出RankJudge，一种基于多轮、参考文档的对话评判基准生成器，能够自动生成带有单一注入缺陷的对话对，并通过联合判断（优劣、缺陷位置、缺陷类型）对LLM评判器进行严格评估。

**💡 创新点**

创新点包括：①全自动化的基准生成流程，生成的对话对在注入缺陷后具有唯一真值；②联合判断准则消除“对错猜测”问题；③三层自动化验证与基于Bradley–Terry的难度裁剪实现高质量无人工标注数据；④在机器学习、生物医学与金融三大知识密集域实现统一评测，揭示强评判器与弱评判器在错误类型识别上的系统性偏差。

**🔧 技术方法**

技术手段包括：多轮对话生成（使用OpenRouter的LLM），三层验证（连贯性、遵从性、基线化），联合判定模型（输出最佳对话、错误位置、错误类别），Bradley–Terry Elo评分与EIP随机游走交叉验证，动态难度裁剪与人类审核对标。

**📊 数据集**

数据集来源：机器学习领域采用RPC‑Bench论文，生物医学采用PubMedQA，金融领域采用S&P 500 10‑K filings；所有文档均为公开可获取的知识库。

**📈 对比分析**

在21个前沿评判器上进行对比，Elo分数跨度近1200点，开放权重模型如Anthropic Claude、Meta Llama、OpenAI GPT‑4在多轮评判中与专有模型竞争甚至超越。加入错误类型判定后，评判器间的差异进一步放大；在多轮评判与单点Likert评判之间的相关性仅为0.81，说明两者衡量的能力不同。

**⚠️ 局限性**

局限性包括：基准仍依赖LLM生成与验证，可能存在隐式偏见；极难的对话对被裁剪导致信息损失；评判器在错误类型识别上存在类别偏差，且提示工程无法弥补模型能力差距；基准主要聚焦多轮对话，单轮或更复杂交互场景尚未覆盖。

---

## 44. Calibration, Uncertainty Communication, and Deployment Readiness in CKD Risk Prediction: A Framework Evaluation Study

**arXiv ID:** 2605.21566 | [PDF](https://arxiv.org/pdf/2605.21566v1)

**作者:** Michael O. Eniolade `[一作]` `[通讯]` (University of Cumberlands), Michael O. Eniolade (University of Cumberlands)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了5种机器学习分类器在慢性肾病风险预测中的校准、不确定性量化与部署准备度，利用内部UCI CKD数据集训练并在MIMIC‑IV演示集进行分布偏移压力测试。

**💡 创新点**

首次将校准、合规预测覆盖率与多维部署检查表整合评估，提出一种结构化的八项标准评估框架，并将split conformal prediction作为诊断工具。

**🔧 技术方法**

使用Platt比例、等距回归进行后置校准；MAPIE实现split conformal预测；采用ROC、ECE、Brier、覆盖率等指标；构建了八项判定准则并打分。

**📊 数据集**

内部数据：UCI CKD（400例，62.5% CKD）；外部压力测试：MIMIC‑IV Demo（97例，23.7% CKD），包含7个特征缺失并使用UCI统计填补。

**📈 对比分析**

在内部测试集所有模型均达AUROC 1.0，后置校准后ECE降至≈0；但在MIMIC‑IV Demo中AUROC跌至0.48–0.58，ECE升至0.68–0.76，覆盖率从≈0.97降至0.21–0.25，所有模型在八项准则中最高得分仅4/16。

**⚠️ 局限性**

主要限制包括：外部集为公开演示子集、样本量小、7个特征缺失需填补、未进行正式外部验证，导致结果对真实临床部署的泛化性不足。

---

## 45. Barriers to Evidence in AI-Related Cases and the Privatization of Proof

**arXiv ID:** 2605.21816 | [PDF](https://arxiv.org/pdf/2605.21816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 46. Academic Text-to-Music Grand Challenge: Datasets, Baselines, and Evaluation Methods

**arXiv ID:** 2605.21538 | [PDF](https://arxiv.org/pdf/2605.21538v1)

**作者:** Fang-Chih Hsieh `[一作]` (National Taiwan University), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**通讯引用:** 7356 | [OpenAlex ID](https://openalex.org/A5061291906)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织ICME 2026学术文本到音乐生成挑战，提供标准化的CC‑许可MTG‑Jamendo子集、公开的FluxAudio基线以及多阶段客观与主观评估流程，以鼓励学术界在有限资源下进行文本到音乐研究。

**💡 创新点**

创新点包括提出概念覆盖评分（CCS）以细粒度评估音乐概念一致性、采用双轨道（500M参数与无参数上限）来兼顾资源受限与性能极致、使用专门的无声乐MTG‑Jamendo子集以及公开所有预处理与评测代码。

**🔧 技术方法**

使用了文本到音乐的生成技术（扩散、流匹配、Transformer、状态空间模型），配合CLAP、FAD、Qwen/MusicGen等大型音频‑文本模型，后处理使用EnCodec与公开的音频分离模型。

**📊 数据集**

使用CC‑许可的MTG‑Jamendo子集（约3777小时）进行无声乐音频，并提供两套由Qwen2‑Audio与Music‑Flamingo+Qwen3生成的参考标签，确保数据可重复、可公开。

**📈 对比分析**

通过FAD、CLAP、CCS三项客观指标并采用Borda计数合成最终排名，12支参赛团队均超过FluxAudio基线；在主观MOS评测中，部分Efficiency Track团队的整体分与业界MusicGen‑small相近或略高，表明在有限参数下仍能获得较好音乐质量。

**⚠️ 局限性**

局限在于数据规模仍有限，OOV（out‑of‑distribution）样本评估尚未验证，模型需从零训练限制可能抑制部分创新，评测对特定LALM的依赖可能引入偏差，主观测试样本量和评测者多样性有限。

---

## 47. Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems

**arXiv ID:** 2605.22001 | [PDF](https://arxiv.org/pdf/2605.22001v1)

**作者:** Aaditya Pai `[一作]` `[通讯]` (Columbia University), Aaditya Pai (Columbia University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM代理的注入检测器进行系统评估，提出域伪装注入和Camouflage Detection Gap (CDG)指标。

**💡 创新点**

首次量化静态payload检测器在域伪装注入下的盲点，并揭示多代理辩论对弱模型的攻击放大效应。

**🔧 技术方法**

使用少量示例的few‑shot检测器、CamouflageGenerator LLM生成的域伪装payload以及多代理辩论架构进行实验。

**📊 数据集**

构建了45个跨金融、法律和通用QA任务的任务库，并在Llama 3.1 8B与Gemini 2.0 Flash上进行测试。

**📈 对比分析**

与静态检测器对比，Llama的CDG为0.84、Gemini为0.44；补充示例后Gemini的检测率提升近90%，但Llama提升不足10%；多代理辩论在Llama中攻击放大近10倍，而在Gemini中降低攻击成功率。

**⚠️ 局限性**

实验仅覆盖两款模型和单轮注入，伪装payload的多样性有限，对工具使用和多轮交互场景未做深入评估。

---

## 48. Argo: Efficient Importance Labeling for Enterprise Email Systems

**arXiv ID:** 2605.21604 | [PDF](https://arxiv.org/pdf/2605.21604v1)

**作者:** Siddhant Ray `[一作]` (University of Chicago), Junchen Jiang `[通讯]` (University of Chicago)

**通讯引用:** 5245 | [OpenAlex ID](https://openalex.org/A5103258769)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向企业级邮件重要性标注的系统，通过高效的离线分析与动态资源调度，实现低成本的标注流程；

**💡 创新点**

创新点在于：①利用标签属性进行离线剖析，构建最佳的SLM级联与嵌入式分类器组合；②设计Pareto前沿的“Balanced”策略与可调权重的成本质量权衡；③在峰值流量下采用贪心实例分配算法，最小化额外成本；

**🔧 技术方法**

核心技术包括：大语言模型（SLM）级联、句子嵌入+多层感知机分类器、基于log‑prob的置信度阈值、标准化Wasserstein-1距离用于分布漂移监测、GPU/CPU资源动态调度算法；

**📊 数据集**

使用公开邮件数据集：Enron（4万封），Fauci（2,761封），Clinton（约7,000封），并随机抽取校准与评估子集；

**📈 对比分析**

与GPT‑4.1、单一SLM、oracle级联及多种剖析基线对比。结果表明：在三大数据集上，Balanced点实现148–167倍的API成本降低，F1分数与GPT‑4.1相当；资源调度方案使峰值时成本提升仅为基线的2.2–3.8倍；剖析成本比完全搜索低达20–640,000倍；

**⚠️ 局限性**

局限性：①仍需依赖GPT‑4.1生成黄金标签，无法完全脱离大模型；②对标签组合的假设可能不适用于更复杂的业务场景；③离线校准需要一定规模的邮件样本，数据隐私与更新频率可能受限；④对非常罕见或多类别标签的表现尚未充分验证；

---

## 49. AttuneBench: A Conversation-Based Benchmark for LLM Emotional Intelligence

**arXiv ID:** 2605.21739 | [PDF](https://arxiv.org/pdf/2605.21739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 50. Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have)

**arXiv ID:** 2605.22005 | [PDF](https://arxiv.org/pdf/2605.22005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 51. TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting

**arXiv ID:** 2605.22069 | [PDF](https://arxiv.org/pdf/2605.22069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 52. Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

**arXiv ID:** 2605.21648 | [PDF](https://arxiv.org/pdf/2605.21648v1)

**作者:** Lucas Fernandez Sarmiento `[一作]` `[通讯]` (Carnegie Mellon University), Lucas Fernandez Sarmiento (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过均值场理论（Mean‑Field Theory）推导出 dropout 在深度网络中对信号传播与梯度传播的影响，形成了关于 dropout 的临界（edge‑of‑chaos）调度与优化框架。

**💡 创新点**

创新点包括：①将 dropout 视作相关扰动，揭示其破坏完美对齐固定点并限定深度相关长度；②发现激活函数的光滑性与尖锐性决定了两种不同的宇宙论类，给出相应的临界指数；③提出两参数缩放收缩（dropout 与临界调节的耦合）以及最优前向加载（front‑loaded） dropout 调度，实现无额外计算成本的损失下降与准确率提升。

**🔧 技术方法**

使用技术包括：均值场理论、价格定理（Price’s Theorem）、Hermite 级数分解、Landau 统计力学方程、变分优化、两参数缩放分析、实验验证（MLP 与 Vision Transformer 的 dropout 调度）。

**📊 数据集**

实验数据集为 CIFAR‑10 与 CIFAR‑100，用以评估 MLP 与 Vision Transformer 的性能。

**📈 对比分析**

与恒定 dropout 基线对比，采用 step、big step、linear 等前向加载方案。实验显示：MLP 交叉熵损失下降 18–35%，ViT 交叉熵下降 4–6%，对应准确率提升 0.5–2% 左右，验证了理论预测。

**⚠️ 局限性**

局限性包括：仅在宽网络（宽度 ≫ 深度）与中等 dropout 的 regime 下可行；未完整扩展至 CNN、ResNet、Transformer 的空间模式；仅关注前向相关传播，未深入梯度及训练动态；未考虑有限宽度修正与训练时 dropout warm‑up 等更复杂的正则化策略。

---

## 53. EasyVFX: Frequency-Driven Decoupling for Resource-Efficient VFX Generation

**arXiv ID:** 2605.22051 | [PDF](https://arxiv.org/pdf/2605.22051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 54. Physiology and Anatomy Aware Inverse Inference of Myocardial Infarction for Cardiac Digital Twin

**arXiv ID:** 2605.22044 | [PDF](https://arxiv.org/pdf/2605.22044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 55. Constructions of Rank-Metric Codes of Small Tensor Rank

**arXiv ID:** 2605.21784 | [PDF](https://arxiv.org/pdf/2605.21784v1)

**作者:** Matteo Bonini `[一作]` (Aalborg University), Giuseppe Cotardo `[通讯]` (Virginia Tech)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5026871112)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了秩度量码的张量秩及其缺陷，引入张量秩缺陷概念，并通过与线性汉明度量码的关系构造了新的低张量秩缺陷秩度量码，特别利用代数几何（AG）码和椭圆曲线码实现了多种 MTR（最小张量秩）或近 MTR 码。

**💡 创新点**

创新点在于：①首次将张量秩缺陷定义与汉明距离参数联系起来，给出张量秩缺陷的上界与下界；②利用 AG 码的乘法与交集性质，给出控制张量秩缺陷的通用构造框架；③在椭圆曲线上构造出张量秩缺陷为 1 的最佳 AMTR 码，从而实现了张量秩与 MDS 存在性的深度关联。

**🔧 技术方法**

主要技术包括：张量代数与张量秩理论、线性码与其余码的交集分析、AG 码的 Riemann–Roch 计算、Schur 乘积与乘法映射、椭圆曲线上的分割与除法，以及基于 Kruskal 下界与 MDS 约束的理论推导。

**📊 数据集**

本文未使用实验数据集，全部为理论构造与符号推导；构造示例以小型有限域（如 𝔽₃、𝔽₅、𝔽₇、𝔽₁₁、𝔽₁₃）上的具体曲线点为例。

**📈 对比分析**

对比方法主要是：利用 Kruskal 下界（k + d – 1）与 MDS 存在性条件检验是否为 MTR；与已知的 MDS、AMDS 码参数做尺寸与距离比较；通过构造的张量秩缺陷上界证明其优于常规构造；实验中给出的具体例子显示张量秩达到理论上限或仅差 1，性能满足或超出期望。

**⚠️ 局限性**

限制包括：①张量秩的 NP‑难性使得计算与验证受限；②构造需满足 MDS conjecture，部分结果仅在该猜想成立或有限域大小满足条件下可实现；③交集条件过于严格，实际应用中可能难以满足；④对高维或大域参数时，AG 码的 Riemann–Roch 维数可能不足以实现所需参数。

---

## 56. Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation

**arXiv ID:** 2605.21949 | [PDF](https://arxiv.org/pdf/2605.21949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 57. Distributed Image Compression with Multimodal Side Information at Extremely Low Bitrates

**arXiv ID:** 2605.22061 | [PDF](https://arxiv.org/pdf/2605.22061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 58. Provable Joint Decontamination for Benchmarking Multiple Large Language Models

**arXiv ID:** 2605.21543 | [PDF](https://arxiv.org/pdf/2605.21543v1)

**作者:** Zhenlong Liu `[一作]` (Southern University of Science and Technology), Hongxin Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1098 | [OpenAlex ID](https://openalex.org/A5020027500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多模型联合基准去污染的框架，目标是从候选样本中挑选出对所有受审模型均无污染的共享基准。

**💡 创新点**

创新点在于将每个模型的 conformal p 值通过最大化聚合成联合 null，然后构造保守的最大-p 分布包络，并结合 adaptive Storey–BH 实现联合污染率（GCR）控制与功效提升。

**🔧 技术方法**

主要技术包括 conformal inference、Benjamini–Hochberg（BH）多假设检验、右尾包络拟合、Storey’s null 估计以及自适应阈值 λ 的数据驱动选择。

**📊 数据集**

实验使用 WikiMIA、ArXivTection、MIMIR 子集以及 GPT-NeoX、Pythia、LLaMA 等多模型族的 16‑模型组进行评估。

**📈 对比分析**

与基准 Max‑p 方案相比，本文方法在多模型、多检测分数下保持预设的污染率控制，并且在功效上提升 1–5 倍以上；在不同模型族、检测分数、污染比例和模型数下均保持稳定。

**⚠️ 局限性**

局限性在于需要共享的成员校准样本与候选样本同分布，且对分布漂移的鲁棒性尚未充分探讨。

---

## 59. Machine learning prediction of obstructive coronary artery disease using opportunistic coronary calcium and epicardial fat assessments from CT calcium scoring scans

**arXiv ID:** 2605.21762 | [PDF](https://arxiv.org/pdf/2605.21762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 60. Flying Together: Human-Guided Immersive Shared Control for Aerial Robot Teams in Unknown Environments

**arXiv ID:** 2605.21680 | [PDF](https://arxiv.org/pdf/2605.21680v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 61. Token-weighted Direct Preference Optimization with Attention

**arXiv ID:** 2605.21883 | [PDF](https://arxiv.org/pdf/2605.21883v1)

**作者:** Chengyu Huang `[一作]` (Cornell University), Claire Cardie `[通讯]` (Cornell University)

**通讯引用:** 21948 | [OpenAlex ID](https://openalex.org/A5070511738)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Token-weighted DPO (TwDPO) 方法，用自注意力权重对每个 token 进行加权，实现更细粒度的偏好优化；

**💡 创新点**

创新点在于：①理论上将 DPO 推导为 token‑weighted RL；②利用 LLM 自己的注意力权重作为 token 重要性评分，避免额外模型或经验函数；

**🔧 技术方法**

主要技术包括：Token-weighted DPO 目标函数、基于对比判断的自注意力提取、权重后处理（归一化与注意力汇聚修正），以及标准的 PPO / RLHF 训练框架；

**📊 数据集**

使用 UltraChat‑200K、UltraFeedback 二值化数据集，以及 LLaMA‑3‑Base‑8B‑SFT 与 LLaMA‑3‑Instruct 版本的训练数据；

**📈 对比分析**

在 AlpacaEval、MT‑Bench、ArenaHard 三大基准上与 RRHF、SLiC‑HF、DPO、IPO、CPO、KTO、ORPO、R‑DPO、SimPO 等对齐方法对比，TwDPO 在大多数指标上领先，尤其在 LLaMA‑3‑8B‑Base‑SFT 上提升 AlpacaEval 12%、MT‑Bench 1.05、ArenaHard 40%；

**⚠️ 局限性**

局限性包括：仅在 8B 模型上验证，缺乏大模型实验；仅关注指令跟随任务；注意力权重选择空间大，需进一步探索头部选择与更高效的权重提取方式；

---

## 62. Motion Design for Grasp-Based Dynamic Locomotion in Microgravity

**arXiv ID:** 2605.21704 | [PDF](https://arxiv.org/pdf/2605.21704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 63. Ex-GraphRAG: Interpretable Evidence Routing for Graph-Augmented LLMs

**arXiv ID:** 2605.21994 | [PDF](https://arxiv.org/pdf/2605.21994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 64. SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents

**arXiv ID:** 2605.21965 | [PDF](https://arxiv.org/pdf/2605.21965v1)

**作者:** Mehrdad Saberi `[一作]` (University of Maryland), Soheil Feizi `[通讯]` (University of Maryland)

**通讯引用:** 10690 | [OpenAlex ID](https://openalex.org/A5025450606)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种连续投机（continuous speculation）框架，利用更快但不完全可靠的投机工具（speculator）在多跳工具使用过程中持续推进推理进程，同时通过异步验证保证最终推理路径与原始路径一致，从而显著降低墙壁时间。

**💡 创新点**

创新点在于：①把投机从单跳扩展为多线程连续投机，形成可并行的投机管道；②引入异步验证机制，在不破坏准确性的前提下实时纠正错误投机；③给出理论框架证明该方法可逼近oracle级别的时延提升，并推导了所需线程数的上界。

**🔧 技术方法**

技术手段包括：①使用LLM或缓存作为投机工具，计算更快的查询结果；②使用可判定的验证器对投机结果与目标工具结果做等价性检查；③实现k线程的持续投机与回滚逻辑；④在实验中使用CoRAG-Llama3.1-8B、GPT‑5等生成模型；⑤通过实验评估与理论对照。

**📊 数据集**

使用的基准数据集有：2WikiMultihopQA、MuSiQue 与 DeepResearch‑9K，这些数据集覆盖从 2–4 跳的检索增强问答任务。

**📈 对比分析**

与标准顺序执行（仅使用目标工具）和完全投机（不验证）进行对比；在实验中，连续投机在保持 EM/F1 准确率基本不变的情况下，墙壁时延平均下降 30–40%（最高 40%），且与理论最优值 (^*) 贴近；同时还对线程数 k 与时延/计算成本的折中做了系统性评估。

**⚠️ 局限性**

局限性包括：①需要投机工具具有足够的速度与准确率，否则投机失败率高导致回滚成本上升；②线程数 k 的增大会带来额外的并行计算开销；③验证器设计需要针对特定任务，过于严格或宽松均可能影响准确率或时延；④在网络延迟占主导或工具调用不可并行的场景中，收益有限。

---

## 65. Provable Robustness against Backdoor Attacks via the Primal-Dual Perspective on Differential Privacy

**arXiv ID:** 2605.21780 | [PDF](https://arxiv.org/pdf/2605.21780v1)

**作者:** Aman Saxena `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 15444 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种统一的框架，将随机平滑（Randomized Smoothing）与差分隐私（Differential Privacy）的双重视角相结合，用于对抗训练时与推理时的后门攻击（backdoor）和数据投毒攻击（poisoning），并给出端到端的鲁棒性证明。

**💡 创新点**

创新点包括：① 将随机平滑的原始（primal）视角与其对应的隐私曲线（privacy profile，dual）相联系，形成可数值组合的通用方法；② 通过对邻域关系的细粒度拆分（如拆分 R 次添加/删除操作为 r_+ 添加和 r_- 删除）实现更紧的鲁棒性上界；③ 在此框架下，将 DP‑SGD、Deep Partition Aggregation (DPA) 等训练机制与推理时的高斯噪声组合，获得新的联合鲁棒性证明。

**🔧 技术方法**

主要技术：随机平滑、差分隐私的 f‑DP 与隐私曲线双重表示、隐私曲线的数值组合（如隐私损失分布卷积）、对邻域关系的分解、基于隐私曲线的鲁棒性定理、在实验中使用随机采样估计期望。

**📊 数据集**

数据集：MNIST 与 CIFAR‑10 图像分类数据集，用于验证 DP‑SGD 与 DPA 的鲁棒性证书，并对联合训练‑推理鲁棒性进行评估。

**📈 对比分析**

与先前方法对比：在 DP‑SGD 的毒化攻击上，本文的隐私曲线+数值组合方案在 8 轮训练时实现了更高的认证准确率；在联合训练‑推理鲁棒性上，本文能够针对更大范围的训练扰动和推理扰动给出证书，展示了与仅使用 RDP 或 DPA 的单一机制相比更强的防御效果。

**⚠️ 局限性**

局限性：① DP‑SGD 的认证需要对大量模型进行采样，导致计算成本显著；② 通过邻域拆分实现更紧的上界会产生 O(R) 的组合复杂度，对较大的 R 仍不够高效；③ 证书的严格性往往伴随模型性能下降，且整体方法依赖于离线计算，难以即时部署。

---

## 66. MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation

**arXiv ID:** 2605.21783 | [PDF](https://arxiv.org/pdf/2605.21783v1)

**作者:** Ahanaf Hasan Ariq `[一作]` `[通讯]` (Ideal School and College), Ahanaf Hasan Ariq (Ideal School and College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一个 PAC‑Bayesian 泛化界，显式将最大均值差异 (MMD) 用作分布偏移量化，并将 MMD 球解释为 Walley 的置信集合，从而为测试时自适应 (TTA) 提供了一套可解释的不确定性量化框架。

**💡 创新点**

创新点在于：①将 MMD 球视作 imprecise probability 的 credal set，分离 epistemic 与 aleatoric 不确定性；②给出 MMD 依赖的 PAC‑Bayesian 误差上界及其有限样本形式；③证明 geodesic 保持界，解释核引导适配为何能保护稀有类结构；④首次将 PAC‑Bayesian、核方法与 Walley 理论统一。

**🔧 技术方法**

使用了 PAC‑Bayesian 理论、最大均值差异 (MMD) 与核嵌入、RKHS Lipschitz 损失假设、Walley 的不确定性理论、置信集合分析以及 MMD 的样本估计与收敛性分析。

**📊 数据集**

未给出具体实验数据集，本文主要是理论分析。

**📈 对比分析**

由于缺乏实验，未进行方法对比；所给界在理论上可计算，性能取决于 MMD 与 KL 复杂度的实际数值。

**⚠️ 局限性**

主要局限：①损失函数需满足 RKHS 成员假设，深度网络中验证困难；②MMD 的收敛速率为 O(1/√n)，可能在实践中过于保守；③未考虑标签分布漂移；④实际自适应策略的实现与参数选择仍需实验验证。

---

## 67. Improving 3D Labeling in Self-Driving by Inferring Vehicle Information using Vision Language Models

**arXiv ID:** 2605.21747 | [PDF](https://arxiv.org/pdf/2605.21747v1)

**作者:** Steven Chen `[一作]` (Aurora Innovation, Inc.), Nemanja Djuric `[通讯]` (Aurora Innovation, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用视觉语言模型（VLM）对车辆图像序列进行一次性推理，既识别车辆的品牌、型号和代次，又输出对应的工厂尺寸，用以生成或改进 3D 边框标签，帮助自动标注过程。

**💡 创新点**

创新点在于将 VMMR（车辆品牌/型号/代次识别）与 3D 边框尺寸预测结合为单一 VLM prompt；通过反复迭代 prompt engineering，加入遮挡评估、配置识别和改装检测等推理步骤，显著提升尺寸预测精度并能检测出人类标注误差。

**🔧 技术方法**

核心技术是多模态预训练视觉语言模型（如 Llama‑4‑Maverick、Pixtral‑Large、Claude‑Sonnet‑4、Gemini Flash/Pro 2.5）配合多图像序列输入；采用 prompt 设计与 chain‑of‑thought 逻辑；对输出结果使用 JSON 模板实现结构化。

**📊 数据集**

使用内部 15‑秒长的高速公路与市街数据集（3,821 车样本）以及 Waymo Open 数据集（1,931 车样本）进行评估。

**📈 对比分析**

与基于车辆类型的“oracle”基线（使用真实类型给出固定尺寸）对比，并与多种 VLM 在不同 prompt 版本下的表现进行 ablation；Gemini‑Pro‑2.5 在长度、宽度、高度误差和 IoU 上均优于基线，尤其在遮挡场景下能提供更准确的尺寸，预测率高达约 99%。

**⚠️ 局限性**

限制包括：1）VLM 需要较大上下文窗口，导致采样和推理成本；2）对极端遮挡或非主流车型识别仍可能失败；3）改装/损伤检测准确性有限，误判率仍需提升；4）依赖预训练模型，缺乏针对特定数据集的微调，可能在跨域应用时性能下降。

---

## 68. CASE-NET: Deep Spatio-Temporal Representation Learning via Causal Attention and Channel Recalibration for Multivariate Time Series Classification

**arXiv ID:** 2605.22043 | [PDF](https://arxiv.org/pdf/2605.22043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 69. I-SAFE: Wasserstein Coherence Metrics for Structural Auditing of Scientific AI Models

**arXiv ID:** 2605.21731 | [PDF](https://arxiv.org/pdf/2605.21731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 70. Parser-Free Querying of Security Logs

**arXiv ID:** 2605.22027 | [PDF](https://arxiv.org/pdf/2605.22027v1)

**作者:** Evan Luo `[一作]` (University of California, Berkeley), David Wagner `[通讯]` (University of California, Berkeley)

**通讯引用:** 21120 | [OpenAlex ID](https://openalex.org/A5062174672)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

生成可执行脚本，以一次性LLM调用查询原始安全日志，无需事先解析器。

**💡 创新点**

通过在一次LLM调用中把自然语言问题编译成针对特定日志模板的代码，并用轻量模板上下文弥补LLM对日志格式的缺失，省去持续解析器维护的工程成本。

**🔧 技术方法**

使用大型语言模型（Gemini 3.1 Pro、GPT‑5.4）、轻量模板提取器（Drain3、Frequency、Matryoshka）、Python/Bash代码生成与沙盒执行以及错误反馈重试机制。

**📊 数据集**

SecurityLogs语料库，包括五类日志（Audit、SSHD、Cron、Puppet、DHCP），共计659,045行。

**📈 对比分析**

与人工编写脚本和Matryoshka解析器对比，宏F1平均为0.939（复杂查询0.905）；相比人工脚本错误率下降约3.6倍，且在大多数数据集上接近解析器性能。

**⚠️ 局限性**

受LLM能力与模板覆盖度限制，复杂规则的完整枚举仍有缺失；实验仅覆盖单源日志、单模型，未验证多源实时流场景及安全性与鲁棒性问题。

---

## 71. Dynamic Mixture of Latent Memories for Self-Evolving Agents

**arXiv ID:** 2605.21951 | [PDF](https://arxiv.org/pdf/2605.21951v1)

**作者:** Dianzhi Yu `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 28117 | [OpenAlex ID](https://openalex.org/A5042251906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于动态混合专家（MoE）的生成式潜在记忆框架，使 LLM 代理在不断变化的任务序列中能够自我演化、内部化新知识并避免灾难性遗忘。

**💡 创新点**

创新点包括：①动态 MoE 与 key‑query 路由实现多专家记忆生成与聚合；②阶段性专家招聘与自适应路由组，实现容量可扩展与任务隔离；③使用无任务 ID 的自编码器做域感知路由，保证 OOD 时退回预训练模型；④核心推理模型保持冻结，所有更新仅在外部模块进行。

**🔧 技术方法**

采用的技术包括：动态 MoE、LoRA 适配器（专家与路由器）、key‑query 路由、轻量级自编码器、Supervised Fine‑Tuning（SFT）目标、负载平衡损失、冻结预训练 LLM（Qwen3‑4B‑Instruct‑2507）以及训练时仅更新 MoE 模块参数。

**📊 数据集**

使用了三组持续学习数据集：数学（Nemotron MATH）、科学（Nemotron Science）和代码（KodCode），并在每个阶段分别在对应测试集上评估。

**📈 对比分析**

与 Vanilla、SFT、MemGen、ExpeL 等基线比较，实验显示该方法在完整学习序列后平均准确率达 73.93%，比 Vanilla 提升 10.40%，相较于 SFT、MemGen、ExpeL 的平均成绩分别提升 39.53%、10.46%、0.00%，且保持零遗忘，ablation 证明多专家 + 隔离路由最优。

**⚠️ 局限性**

局限性在于相对于完全冻结的骨干网络，模型需额外引入专家、路由器和自编码器模块，虽然参数量相对较小，但仍增加了额外开销。

---

## 72. Articulate but Wrong: Self-Review Failures in LLM-Based Code Modernization

**arXiv ID:** 2605.21537 | [PDF](https://arxiv.org/pdf/2605.21537v1)

**作者:** Gokul Chandra Purnachandra Reddy `[一作]` (Amazon Web Services), Harsha Sanku `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对11款生产级LLM在60个Python 2代码片段上执行迁移实验，并用类型严格的行为等价判别器评估迁移后代码的行为漂移，随后让生成模型自评其迁移结果是否保持行为。

**💡 创新点**

揭示了自评机制在检测无声行为漂移方面的局限性，发现漂移具有任务结构化特征且与模型规模无关，同时强调了严格等价判别和稳健输出提取器对实验结果的重要性。

**🔧 技术方法**

使用类型严格行为等价判别器、鲁棒输出提取器、三种提示表述以及自评探测器，并在OpenRouter上调用11款LLM进行迁移和自评。

**📊 数据集**

构建了60条平衡的Python 2代码片段数据集，涵盖语义漂移陷阱、语法陷阱和无害对照三类。

**📈 对比分析**

对11款模型在三种提示下的迁移结果进行对比，发现语义漂移率高达39.7%，而无害对照仅7%；自评在所有漂移案例中有31.7%的误判率，且自评误判呈双峰分布。

**⚠️ 局限性**

实验仅覆盖Python 2→3迁移、短代码片段、单模型自评，未考察多模型投票、链式思考等技术，结果可能不具备对其他语言迁移或更复杂代码的普适性。

---

## 73. EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control

**arXiv ID:** 2605.21862 | [PDF](https://arxiv.org/pdf/2605.21862v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 74. Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation

**arXiv ID:** 2605.21491 | [PDF](https://arxiv.org/pdf/2605.21491v1)

**作者:** Srujan P Mule `[一作]` (IISER Pune), Manasi Patwardhan `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 PapersWithCode 的 11,488 对研究思路的对比数据集，并训练了 8B 参数语言模型预测哪个思路在指定基准上表现更佳。

**💡 创新点**

创新点包括：①大规模、基于客观实验结果的对比预测数据集；②结合 SFT 与 RLVR 训练可解释的链式推理模型；③证明低成本小模型可与前沿模型媲美。

**🔧 技术方法**

使用了监督微调（SFT）、强化学习（GRPO、DAPO）与奖励学习、链式推理（CoT）以及 min‑max 归一化的 Unified Score 进行模型训练。

**📊 数据集**

采用了 11,488 个idea对，来源于 1,918 个 NLP 评测排行榜、PapersWithCode、结果报告论文及原始论文，并构造了跨域与独立测试集。

**📈 对比分析**

通过 pairwise 一致性+准确率评估，SFT 直接标签模型达到 77% 准确率，加入推理的 RL 模型约 71%；在跨域和独立测试集上相较 GPT‑5 与 GPT‑4.1 提升 10+ 个百分点，鲁棒性优良。

**⚠️ 局限性**

限制在于仅覆盖 NLP 领域、受排行榜数据噪声影响、未验证在完整 ideation 工作流中的实用性，且尚未扩展到其他学科和任务。

---

## 75. Analytical and Experimental Force Analysis of a Soft Linear Pneumatic Actuator

**arXiv ID:** 2605.21836 | [PDF](https://arxiv.org/pdf/2605.21836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 76. A Causal Argumentation Method for Explainability of Machine Learning Models

**arXiv ID:** 2605.21758 | [PDF](https://arxiv.org/pdf/2605.21758v1)

**作者:** Henry Salgado `[一作]` (University of Texas at El Paso), Martine Ceberio `[通讯]` (University of Texas at El Paso)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5047541010)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将因果发现与双极论证框架相结合的可解释方法，能够生成结构化、可辩论的模型解释；

**💡 创新点**

创新点在于：①利用双重编码的 FCI 算法在观测数据中生成可合并的因果图；②将因果图转化为支持/攻击关系的双极论证框架，并采用半稳定语义提供最具覆盖度的解释；

**🔧 技术方法**

技术包括：约束式因果发现（FCI）、双重编码与投票合并、特征离散化与编码、双极论证框架构建、经典论证框架转换、概率论证与半稳定扩展；

**📊 数据集**

使用了两个公开数据集：泰坦尼克号乘客存活数据（891 样本）和 Pima 印第安人糖尿病数据（768 样本）；

**📈 对比分析**

与 SHAP、LIME 以及剪枝决策树等主流后置可解释方法对比，结果显示所得到的扩展与传统方法在特征重要性和结构一致性上高度一致；性能方面，模型在两组数据上均能生成可解释的论证扩展，且扩展数量与传统方法相当；

**⚠️ 局限性**

局限性：仅基于观测数据，因果图为马尔可夫等价类，方向不确定；样本量有限导致条件独立检验误差；可能存在未观测的混杂变量；解释依赖于离散化与阈值设置，缺乏可量化评估指标。

---

## 77. UniVL: Unified Vision-Language Embedding for Spatially Grounded Contextual Image Generation

**arXiv ID:** 2605.21611 | [PDF](https://arxiv.org/pdf/2605.21611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. Conditional Entropy of Heat Diffusion on Temporal Networks

**arXiv ID:** 2605.21514 | [PDF](https://arxiv.org/pdf/2605.21514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 79. Neural Acceleration for Graph Partitioning

**arXiv ID:** 2605.21519 | [PDF](https://arxiv.org/pdf/2605.21519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 80. Does Slightly Mean Somewhat? Measuring Vague Intensity Words in LLM Numeric Actions

**arXiv ID:** 2605.21827 | [PDF](https://arxiv.org/pdf/2605.21827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 81. Chain Reactions: How Nonce Collisions in ECDSA Compromise Polygon MEV Searchers

**arXiv ID:** 2605.21498 | [PDF](https://arxiv.org/pdf/2605.21498v1)

**作者:** Yash Madhwal `[一作]` (Skolkovo Institute of Science and Technology), Yury Yanovich `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5020656981)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对Polygon MEV交易的链上数据进行分析，揭示了搜索者在高速竞拍中系统性地重用ECDSA随机数（nonce），并给出了一套基于线性代数的私钥恢复方法；同时提供了实测案例，验证了跨钱包随机数碰撞可导致链式私钥泄露的威胁。

**💡 创新点**

创新之处在于将ECDSA随机数重用的攻击转化为一个紧凑的线性方程组，尤其针对跨钱包碰撞的情况提出了两方程足以破解所有相关钱包的公式，并以Polygon MEV生态为真实案例展示此漏洞的传播链。

**🔧 技术方法**

主要使用了ECDSA签名的数学推导、线性代数求解、Python脚本抓取链上签名数据以及RFC 6979确定性随机数生成的对照实验等技术。

**📊 数据集**

使用的数据集为Polygon主网在从Priority Gas Auction 转向 sealed‑bid FastLane Auction 期间的区块交易日志，抽样了数十万条交易记录并提取其中的 (r,s,e) 三元组。

**📈 对比分析**

通过构建线性方程组并使用标准线性求解器对真实链上签名数据进行求解，攻击在数秒内即可完成，展示了与传统基于格攻击相比在此场景下更高的实用性和效率。

**⚠️ 局限性**

局限性包括：攻击仅在随机数被重复或可预测时有效；一旦搜索者改用 RFC 6979 等确定性方案则失效；仅针对 ECDSA，未涵盖 Schnorr/EdDSA 等其它签名方案；且需要完整可读的链上签名数据。

---

## 82. LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning

**arXiv ID:** 2605.22012 | [PDF](https://arxiv.org/pdf/2605.22012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 83. Multi-scale interaction network for stereo image super-resolution

**arXiv ID:** 2605.21913 | [PDF](https://arxiv.org/pdf/2605.21913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 84. AVI-HT: Adaptive Vision-IMU Fusion for 3D Hand Tracking

**arXiv ID:** 2605.21714 | [PDF](https://arxiv.org/pdf/2605.21714v1)

**作者:** Ziyi Kou `[一作]` (Meta Reality Labs), Li Guan `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AVI-HT，一种自适应视觉-IMU融合框架，用于从自举相机与手套IMU信号联合估计3D手姿。

**💡 创新点**

核心创新是基于跨传感器注意力的层级融合机制，能够根据手部视觉遮挡动态调整IMU权重，并引入运动学先验掩码。

**🔧 技术方法**

使用Transformer编码器、交叉注意力模块、SE(3)视觉-3D解码器以及MANO/UMETrack手模型，结合自适应注意力和时序窗口处理。

**📊 数据集**

在新收集的DexGloveHOI数据集上评估，该数据集包含100K+同步自举相机、12个6-DoF IMU和MoCap标定的3D姿态。

**📈 对比分析**

与单模视觉、后处理EKF融合和IMU单独跟踪等基线相比，AVI-HT在UMETrack模型下平均关键点误差降至10.36mm，提升约16%；在MANO模型下PA-MPJPE降至10.52mm，提升约23%。

**⚠️ 局限性**

局限性在于手套可见性导致视觉域差距以及对特定手套布局的依赖，难以直接迁移到裸手或其他传感器配置。

---

## 85. AI-Enabled Serious Games: Integrating Intelligence and Adaptivity in Training Systems

**arXiv ID:** 2605.21962 | [PDF](https://arxiv.org/pdf/2605.21962v1)

**作者:** Priyamvada Tripathi `[一作]` (Durham College), Bill Kapralos `[通讯]` (Ontario Tech University)

**通讯引用:** 3476 | [OpenAlex ID](https://openalex.org/A5086060956)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并分析了人工智能技术如何与认知游戏融合，探讨了教学智能与适应性的历史演进及其在现实训练系统中的实现与挑战。

**💡 创新点**

提出了将教学智能与实时适应性分离并通过多智能体架构协同实现的概念框架，阐述了LLM、RL与Agent在智能游戏中的协同作用与潜在价值。

**🔧 技术方法**

核心技术包括大语言模型（LLM）实现语义理解与生成，强化学习（RL）实现实时决策与难度调整，以及基于Agent的多模组件架构用于职责分离与动态协调。

**📊 数据集**

论文主要以文献综述为主，并未使用特定实验数据集；所讨论的系统与案例（如Khan Academy、Duolingo、Khanmigo、MATHia）均基于各自组织内部的学习交互数据。

**📈 对比分析**

由于本研究为综述与架构性讨论，未进行量化实验对比；作者指出现有AI增强系统在学习成效、技能迁移等方面的实验数据不足，强调需要进一步的基准评估与实证验证。

**⚠️ 局限性**

主要局限包括：缺乏充分的长期学习效果实证；模型解释性与Hallucination风险；算法与奖励函数设计不当导致的学习目标偏差；多智能体协调带来的延迟与可观测性挑战；以及系统验证、成本与可持续性问题。

---

## 86. An Information-theoretic Analysis of Edge-reinforced Random Walks

**arXiv ID:** 2605.21853 | [PDF](https://arxiv.org/pdf/2605.21853v1)

**作者:** Qinghua `[一作]`, Venkat Anantharam `[通讯]` (University of California at Berkeley)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究有限图上线性边强化随机游走（ERRW）的信息论量，包括熵率、环境法则之间的KL散度以及有限轨迹法则之间的KL散度。

**💡 创新点**

首次给出熵率的解析式及其上界、环境法则KL散度的闭式Bregman公式，并通过STZ场分解提供概率解释；同时推导出轨迹级KL散度趋于环境级KL散度的收敛速率与反比局部时间的界，并在星形图上得到精确的幂律收敛速率。

**🔧 技术方法**

利用ERRW的随机环境表示（魔法公式）、指数族理论、STZ场分解、离散马尔可夫链熵率公式、Chernoff型与Azuma‑Hoeffding界、对数Gamma函数及其导数、图论组合与切比雪夫不等式等多种技术手段。

**📊 数据集**

本研究完全为理论分析，没有使用任何真实或模拟数据集。

**📈 对比分析**

没有与其他方法做经验对比；研究结果以解析公式和上界形式给出，主要用于理论阐述与后续统计检验问题的理论基准。

**⚠️ 局限性**

对一般有限图的轨迹级KL散度收敛速率只给出了多项式上界，尚不一定是最优的；在初始权重较大（>4+2√5）时收敛速率仍为O(1/T)，但精确阶数未知；研究未覆盖更复杂的检验任务（如身份检验、相似度检验）的具体算法与性能分析。

---

## 87. The Illusion of Reasoning: Exposing Evasive Data Contamination in LLMs via Zero-CoT Truncation

**arXiv ID:** 2605.21856 | [PDF](https://arxiv.org/pdf/2605.21856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 88. Addressing the Synergy Gap: The Six Elements of the Design Space

**arXiv ID:** 2605.21635 | [PDF](https://arxiv.org/pdf/2605.21635v1)

**作者:** Tommaso Turchi `[一作]` (University of Pisa), Alessio Malizia `[通讯]` (University of Pisa)

**通讯引用:** 2303 | [OpenAlex ID](https://openalex.org/A5060199482)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文通过研讨会与文献综述，构建了实现人机协同决策的六元设计空间，并为实践者提供了框架与术语。

**💡 创新点**

创新之处在于将人机协同视为多维设计问题，提出涵盖社会技术环境、决策框架、参与者、AI能力、交互方式和整体评估的系统化六要素，突破了以往单纯关注算法或界面的局限。

**🔧 技术方法**

采用了协同设计（Design Science）方法论、工作坊式共创、理论合成和概念建模等技术手段，而非单一算法实现。

**📊 数据集**

未使用具体数据集，研究基于案例讨论、学术文献和工作坊参与者的反馈构成的定性资料。

**📈 对比分析**

论文未进行实验比较或性能评估，所提供的是概念框架和设计建议，缺乏可量化的效果对比。

**⚠️ 局限性**

局限性包括缺乏实证验证、对不同情境的适用性未充分检验、可能低估了复杂组织环境中的不可预见因素，以及对跨文化差异和动态演化过程的关注不足。

---

## 89. Objective-Induced Bias and Search Dynamics in Multiobjective Unsupervised Feature Selection

**arXiv ID:** 2605.21561 | [PDF](https://arxiv.org/pdf/2605.21561v1)

**作者:** Mathieu Cherpitel `[一作]` (Leiden University), Anna V. Kononova `[通讯]` (Leiden University)

**通讯引用:** 1061 | [OpenAlex ID](https://openalex.org/A5004203337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多目标无监督特征选择的设计决策及其对搜索行为和 Pareto 前沿的影响，系统评估了不同评估目标、子集大小正则化方向和初始化策略的组合。

**💡 创新点**

在合成数据上首次全面对比了三种无监督评估目标（轮廓分数、PCA 重构损失）与监督准确率，以及子集大小的最小化/最大化正则化，并提出了 PCA 重构损失作为有效的无监督目标。

**🔧 技术方法**

使用了 SMS‑EMOA 多目标进化算法，结合轮廓分数、RF 准确率、PCA 重构损失三种目标，并采用随机、分段和固定卡片数三种初始化策略。

**📊 数据集**

构造了一个包含信息性、线性冗余、非线性冗余、噪声、结构噪声和 Sweep 等六类特征的三分类合成数据集，用于可控实验。

**📈 对比分析**

通过对六种组合进行搜索，记录 Pareto 前沿并在 held‑out 测试集上评估 RF 准确率，结果显示：直接优化准确率表现最佳；PCA 重构损失在无监督条件下接近监督结果；轮廓分数组合表现最差。

**⚠️ 局限性**

仅在合成数据上验证，缺乏真实数据实验；PCA 重构损失仅采用线性模型；搜索预算有限，可能未覆盖整个解空间，导致前沿不完整。

---

## 90. Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift

**arXiv ID:** 2605.21552 | [PDF](https://arxiv.org/pdf/2605.21552v1)

**作者:** Jinzong Dong `[一作]` (Central South University), Bo Yang `[通讯]` (Central South University)

**通讯引用:** 72377 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Expectation Consistency Loss（ECL），一种针对协变量偏移（covariate shift）的无监督置信度校准方法；

**💡 创新点**

创新点在于引入了Expectation Consistency Condition，证明全球分布对齐并非必要，仅需在置信度水平上保持源域与目标域的期望后验一致，并基于此构建可微分且可批量训练的损失；

**🔧 技术方法**

使用置信度分箱、平滑软分配、额外分类头估计后验、以及辅助变量实现的mini‑batch可训练ECL；

**📊 数据集**

在模拟协变量偏移数据和真实任务数据集上验证：MNIST/USPS/SVHN（Digit）、PACS、ImageNet‑Sketch；

**📈 对比分析**

与Soft‑ECE、DECE、KDE、TS、TransCal、DRL、PseudoCal等9种基线比较，ECL在大多数转移任务和网络架构下在top‑label、class‑wise、canonical三种校准范式中均取得最低或接近最低的ECE/​CwECE/​ECE^KDE，并在多数情形下保持甚至提升准确率；

**⚠️ 局限性**

局限性：假设后验概率P(Y|X)在源域与目标域相同，无法处理标签偏移（label shift）；需要额外的分类头和相关超参数，且在极端密度比大时仍可能面临不稳定性。

---

## 91. Seizure-Semiology-Suite (S3): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding

**arXiv ID:** 2605.21852 | [PDF](https://arxiv.org/pdf/2605.21852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 92. PointLLM-R: Enhancing 3D Point Cloud Reasoning via Chain-of-Thought

**arXiv ID:** 2605.22013 | [PDF](https://arxiv.org/pdf/2605.22013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. CausalGuard: Conformal Inference under Graph Uncertainty

**arXiv ID:** 2605.21928 | [PDF](https://arxiv.org/pdf/2605.21928v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 94. Ablate-to-Validate: Are Vision-Language Models Really Using Continuous Thought Tokens?

**arXiv ID:** 2605.21642 | [PDF](https://arxiv.org/pdf/2605.21642v1)

**作者:** Tianyi Zhang `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 13316 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了Token Replacement Test（TRT），通过在视觉语言模型推理阶段对连续或离散的隐含视觉令牌进行系统性替换，检验模型是否真正利用这些令牌进行推理。

**💡 创新点**

创新点在于提出了“Ablate-to-Validate”原则，将TRT作为一套标准化的诊断工具，能够分离令牌位置偏置、令牌预算与多样性以及信息利用三大干扰因素，首次量化隐含视觉令牌的真实信息利用程度。

**🔧 技术方法**

技术方法包括：在深度推理测试场景下插入固定长度深度令牌 span，训练模型预测并消耗这些令牌；对令牌进行身份、零、随机、分布匹配随机、首重复、计数匹配以及 oracle 替换等六种干预；并将相同干预应用于多种 VLM 后端（LLaVA、Qwen2.5‑VL、Mirage、Mull‑Tokens、CoVT）。

**📊 数据集**

使用的数据集主要为自定义的 HardBLINK（由 BLINK 迁移而来，包含 372 张图片），以及公开的 BLINK、VSP、CV‑Bench 等多步视觉推理基准；深度映射采用 SigLIP2、CLIP、DINOv2、VQ‑VAE 等预训练视觉编码器。

**📈 对比分析**

实验比较表明：离散深度令牌在多种模型和预算下能显著提升准确率，而连续令牌的提升幅度有限；TRT 结果显示连续令牌对内容的敏感性极低，oracle 替换几乎无提升，随机替换仅微降，而离散令牌对内容高度敏感，oracle 替换可提升约 9–10%，随机替换导致 20% 左右性能下滑，验证了隐含令牌内容真正被利用。

**⚠️ 局限性**

局限性包括：TRT 主要针对深度推理任务，可能无法完全反映其他视觉推理或生成任务的情况；连续令牌的高维噪声或冗余导致其信息瓶颈效果不佳，TRT 可能无法捕捉更细粒度的推理利用；此外，TRT 的诊断仍依赖于模型在推理阶段对令牌位置的固定假设，未必适用于动态位置或多模态交互强烈变化的场景。

---

## 95. Closed-Loop Sim-to-Real Reinforcement Learning for Deformable Microfiber Shape Control

**arXiv ID:** 2605.21688 | [PDF](https://arxiv.org/pdf/2605.21688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 96. Broadening Access to Transportation Safety Data with Generative AI: A Schema-Grounded Framework for Spatial Natural Language Queries

**arXiv ID:** 2605.21712 | [PDF](https://arxiv.org/pdf/2605.21712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 97. Community-Aware Vertex Ordering for Reference-Based Graph Compression: A Cross-Encoder Empirical Study

**arXiv ID:** 2605.21510 | [PDF](https://arxiv.org/pdf/2605.21510v1)

**作者:** Jimmy Dubuisson `[一作]` `[通讯]` (Vantino), Jimmy Dubuisson (Vantino)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并验证了基于社区检测的 Leiden+  vertex 排序来提升参考式图压缩效果。

**💡 创新点**

证明排序对不同编码器的压缩收益迁移一致，并提出三种自适应编码器。

**🔧 技术方法**

使用 Leiden 社区检测、两阶段排序、参考复制-区间-残差编码、三种自适应编码器、Julia 实现。

**📊 数据集**

在 LAW（CNR‑2000、in‑2004、enwiki‑2013）和 SNAP（web‑google、Amazon‑0601、eat、arxiv‑hep‑ph）等 7 个真实图上评测。

**📈 对比分析**

通过 bits/edge 对比，Leiden+ 在弱序图上平均提升 0.3–5.4 bit，跨四种编码器差异 ≤0.04 bit，说明排序贡献显著；编码器提升 2–9%，但相对排序小。

**⚠️ 局限性**

编码速度慢、对大规模图扩展性有限、随机访问仍需额外索引、需手动调参。

---

## 98. Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning

**arXiv ID:** 2605.21988 | [PDF](https://arxiv.org/pdf/2605.21988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 99. When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning

**arXiv ID:** 2605.21606 | [PDF](https://arxiv.org/pdf/2605.21606v1)

**作者:** Xiaogeng Liu `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 on‑policy self‑distillation 框架下引入位置加权策略（PW‑OPSD），通过分支可行性诊断识别教师标记在不同序列位置的可靠性差异，并据此调整监督权重。

**💡 创新点**

创新点在于发现教师 token 的可靠性呈现明显的位置信息（早期分支不可靠，后期可靠），并利用分支可行性诊断设计了一个以位置为依据的递增 sigmoid 权重曲线，首次在 OPD 任务中将全局结构信号与局部监督结合。

**🔧 技术方法**

主要技术包括：on‑policy self‑distillation 与 clipped forward‑KL 损失；位置加权函数（w_min、τ、s 参数的 sigmoid 变换）；LoRA 低秩微调；以及分支可行性诊断（通过强制替代 token 并在学生模板下继续生成验证可靠性）。

**📊 数据集**

实验使用 Qwen3‑4B、DeepSeek‑R1‑Distill‑Llama‑8B、Olmo‑3‑7B‑Think 三种模型，在 MATH‑500、AIME‑2024、AIME‑2025、HMMT‑2025 四个数学推理基准上评测。

**📈 对比分析**

与 OPSD、基于熵的自适应 OPD、REOPOLD 等基线相比，PW‑OPSD 在 AIME‑2024 和 AIME‑2025 的 Avg@12 上提升约 1.0–1.1 个百分点，HMMT‑2025 的 Avg@12 也有显著改善；在跨模型评估中，未进行调参的同一位置加权策略在三种模型上均实现 0.39–0.50 个百分点的 Avg@12 提升。

**⚠️ 局限性**

局限性包括：提升幅度相对有限（主要依赖位置加权）；对每个模型无专门的参数调优；未探索更复杂的目标函数（如位置条件的 forward/reverse KL 混合），因此仍有提升空间。

---

## 100. SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?

**arXiv ID:** 2605.21740 | [PDF](https://arxiv.org/pdf/2605.21740v1)

**作者:** Kevin Han `[一作]` (Carnegie Mellon University), Amir Farimani `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SMDD-Bench，一个针对多轮、多任务、小分子药物设计的LLM代理基准，包含502个保证可解的任务实例。

**💡 创新点**

首次构建保证可解的多任务药物设计基准，涵盖5种任务类型，使用见证者生成保证解法，评估LLM代理的长期规划与化学推理能力。

**🔧 技术方法**

采用LLM代理（ReAct框架）、RDKit、PLIP、OpenBabel、Boltz2与ADMET-AI等工具进行分子操作、结构预测与性质评估，并通过最小化调用数量模拟实验稀缺。

**📊 数据集**

使用ChEMBL、PDB、BindingDB等公开数据库生成任务实例，并利用自建的可见证者化学/生物数据。

**📈 对比分析**

在无网络访问、限制Oracle调用的环境下评估7款前沿LLM，结果显示GPT-5.4最高达到40.2%整体成功率，但在3D几何、结构优化等任务上仍低于5%，显示显著提升空间。

**⚠️ 局限性**

受限于Boltz2与ADMET-AI的预测误差、有限的Oracle调用、缺乏真实实验验证、以及LLM在多轮推理与多样性生成方面表现不佳。

---

## 101. Leveraging Self-Paced Curriculum Learning for Enhanced Modality Balance in Multimodal Conversational Emotion Recognition

**arXiv ID:** 2605.21565 | [PDF](https://arxiv.org/pdf/2605.21565v1)

**作者:** Phuong-Anh Nguyen `[一作]` (VNU University of Engineering and Technology), Cam-Van Thi Nguyen `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个基于自适应课程学习（Self-Paced Curriculum Learning）的插件模块，用来缓解多模态情感识别（MERC）中的模态不平衡问题。

**💡 创新点**

创新点在于提出双层难度评估器（句子级+对话级）并采用硬正则化的学习调度器，使训练过程能够动态筛选易到难的样本，显著提升弱模态的学习效果。

**🔧 技术方法**

使用了Self-Paced Curriculum Learning、双层难度量化、硬正则化掩码、以及标准的梯度下降更新等技术，形成了可插拔的学习框架。

**📊 数据集**

实验数据集为IEMOCAP和MELD两大多模态对话情感数据集。

**📈 对比分析**

与RNA loss、OGM-GE、OPM、FAGM等现有平衡方法以及4个基线模型对比，IEMOCAP上w‑F1提升1.2%–6.6%，MELD上提升至10.4%，均显著优于SOTA。

**⚠️ 局限性**

局限性包括对超参数（ε、α、阈值λ）的敏感性，需要手工调参；对短对话或极大规模数据集的适配性不足；以及额外的计算开销（平均每epoch约+10s）。

---

## 102. Don't Collapse Your Features: Why CenterLoss Hurts OOD Detection and Multi-Scale Mahalanobis Wins

**arXiv ID:** 2605.21493 | [PDF](https://arxiv.org/pdf/2605.21493v1)

**作者:** Rahul D Ray `[一作]` `[通讯]` (BITS Pilani), Rahul D Ray (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 Geometry‑Optimised Epistemic Network (GOEN)，一种基于多尺度特征、L2 归一化、马氏距离和轻量化校准头的 OOD 检测管道。

**💡 创新点**

发现 CenterLoss 对 OOD 检测有负面影响并选择去除；通过将层2和层4特征拼接、在单位球面上计算马氏距离并训练校准头使用真实硬 OOD 示例，实现了显著的性能提升。

**🔧 技术方法**

使用 ResNet‑18 多尺度特征提取、L2 归一化、马氏距离、余弦相似度、预测熵、轻量化 MLP 校准头、交叉熵训练、温度标度等技术。

**📊 数据集**

以 CIFAR‑10 为 ID 数据集，SVHN、过滤后的 CIFAR‑100、合成高斯噪声作为 OOD 进行评估；校准阶段使用 SVHN 子集作为硬 OOD。

**📈 对比分析**

在 ID 上保持 93.1% 的准确率，三类 OOD 上的 AUROC 分别为 0.9372、0.9079、1.0000，平均 AUROC 为 0.9483，明显优于 Deep Ensemble（0.8827）、KNN（0.8967）和 ODIN（0.8870）等基线。

**⚠️ 局限性**

需要真实硬 OOD 示例进行校准；仅在 CIFAR‑10 级别验证，尚未验证更大规模数据集；对种子波动有一定敏感性；对极端 OOD 之外的情况尚未评估。

---

## 103. RefusalBench: Why Refusal Rate Misranks Frontier LLMs on Biological Research Prompts

**arXiv ID:** 2605.21545 | [PDF](https://arxiv.org/pdf/2605.21545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 104. SPIDER: Two Server Functionality for the Cost of Zero

**arXiv ID:** 2605.21857 | [PDF](https://arxiv.org/pdf/2605.21857v1)

**作者:** Ofir Dvir `[一作]` (University of California Santa Barbara), Dahlia Malkhi `[通讯]` (University of California Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了单服务器私有信息检索（PIR）协议baseSPIDER和其对默认服务器的适配版本SPIDER，能够在不改动服务器端API的情况下实现信息理论安全的检索；

**💡 创新点**

创新点在于使用多重集合（multiset）提示构造取代传统分片方案，显著降低常数因子、简化设计，并通过一种简单的红点（redaction）和替换槽机制实现单次提示可用；同时将此方案通过轻量级变换推广到默认服务器场景，实现了首个非合作单服务器PIR；

**🔧 技术方法**

关键技术包括基于种子（seed）与伪随机生成器的多重集合构造、Hint的压缩存储（仅存seed和XOR值）、单用Hint与替换槽的动态更新、红点查询与本地XOR恢复、以及连续预处理与提示刷新；

**📊 数据集**

实验使用大规模合成数据库（2^20、2^24、2^28条目，条目大小64KB–4GB）和真实的WikiData SPARQL端点；

**📈 对比分析**

与现有方案（RMS‑24、PIANO‑23/24、WR‑25）在通信量、下载带宽、单查询延迟、服务器吞吐量等指标上对比，SPIDER在默认服务器下实现单字节下载、与RMS‑24相当或更优，且在大条目和高网络延迟环境下表现突出；

**⚠️ 局限性**

局限性包括：无法同时实现默认服务器与单查询最优通信的完全最优方案；需要一次完整的预处理与周期性刷新，且对非索引型数据库需额外离线键映射；在极低带宽或高并发场景下仍受hint搜索成本影响。

---

## 105. Emerging memory technologies at room/cryogenic temperature

**arXiv ID:** 2605.21912 | [PDF](https://arxiv.org/pdf/2605.21912v1)

**作者:** Siddhartha Raman Sundara Raman `[一作]` (University of Texas at Austin), Siddhartha Raman Sundara Raman `[通讯]` (University of Texas at Austin)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5088326114)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对从 SRAM、DRAM 到非易失性存储器（NAND/NOR Flash、RRAM、MRAM、FeFET）的写读保留机制及其设计约束进行了系统综述。

**💡 创新点**

创新点在于将多种存储技术的性能、能耗与设计难点统一归纳，并对其在不同工作温度（常温与低温）下的表现给出比较。

**🔧 技术方法**

采用文献综述和技术分析方法，对比了各类存储器的电路结构、读写机制和可靠性。

**📊 数据集**

无。

**📈 对比分析**

通过汇总公开数据与典型指标，比较了各技术在访问延迟、功耗、密度和耐久性方面的优势与劣势。

**⚠️ 局限性**

仅为理论与文献回顾，缺乏实验验证和统一基准测试，结论依赖于已有研究的报告。

---

## 106. Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly

**arXiv ID:** 2605.21625 | [PDF](https://arxiv.org/pdf/2605.21625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 107. GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation

**arXiv ID:** 2605.21605 | [PDF](https://arxiv.org/pdf/2605.21605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. Deterministic vs. Probabilistic Summarisation: An Empirical Trade-off Study in Design Pattern Centric Java Code

**arXiv ID:** 2605.21943 | [PDF](https://arxiv.org/pdf/2605.21943v1)

**作者:** Najam Nazar `[一作]` (Monash University), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 5325 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在同一预处理和模式识别基础上，比较了规则式 NLG、SWUM 以及基于 Mixtral LLM 的三种代码摘要生成管道，对150个实现了9种常见设计模式的 Java 文件进行意图导向的摘要生成与评估。

**💡 创新点**

创新点在于：① 在结构化测试床下仅对生成策略做对比，消除其他变量；② 结合自动相似度指标与 LLM 辅助的多维度评判（准确性、简洁性、充足性、代码上下文感知、模式忠实度）；③ 通过统计检验（Wilcoxon、Friedman、Spearman）系统验证两类方法的差异与稳定性，揭示显著的语义质量与简洁性权衡。

**🔧 技术方法**

使用技术包括：JavaParser+SimpleNLG（规则式 NLG）、SWUM（词汇用法模型）、Mixtral 8x22B LLM（基于 OpenRouter）、BERTScore、余弦相似度、Llama 3 70B 作为评判模型，辅以 Wilcoxon、Friedman、Spearman 等统计分析。

**📊 数据集**

数据集为 150 条 Java 文件，来源于 3 个 GitHub 仓库（2 教育项目、1 商业项目），覆盖 Adapter、Visitor、Observer、Memento、Facade、Decorator、Abstract Factory、Factory Method、Singleton 9 种常见设计模式；每文件附有 DPS 人工参考摘要。

**📈 对比分析**

比较方法：对三种生成器的摘要与人工参考分别计算 BERTScore（P/R/F1）与余弦相似度，并使用 Llama 3 进行 5 维度打分；随后对指标做 Wilcoxon、Friedman 及 Spearman 相关性检验。结果显示：LLM 在语义匹配与多维度评判上显著优于 deterministic，但在简洁性上落后；deterministic 在长度与可重复性方面更具优势。

**⚠️ 局限性**

局限性：① 仅针对 Java 与 9 种设计模式，可能不适用于其他语言或更复杂的架构；② LLM 评判依赖单一模型，可能带有偏好；③ 对 LLM 的重现性仅在相同 prompt/温度下测试，实际部署仍可能受 API 变动影响；④ 评估主要基于相似度与自动打分，缺乏真实开发者任务级效能验证。

---

## 109. MRecover: A Conditional Generative Model for Recovering Motion-Corrupted MR images Using AI Generated Contrast

**arXiv ID:** 2605.21669 | [PDF](https://arxiv.org/pdf/2605.21669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 110. Harnesses for Inference-Time Alignment over Execution Trajectories

**arXiv ID:** 2605.21516 | [PDF](https://arxiv.org/pdf/2605.21516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 111. Two-Stage Multimodal Framework for Emotion Mimicry Intensity Prediction

**arXiv ID:** 2605.21869 | [PDF](https://arxiv.org/pdf/2605.21869v1)

**作者:** Dinithi Dissanayake `[一作]` (National University of Singapore), Suranga Nanayakkara `[通讯]` (National University of Singapore)

**通讯引用:** 3415 | [OpenAlex ID](https://openalex.org/A5027647989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种分阶段多模态框架，先对文本、音频、视觉及可选运动特征进行单模态预训练，再用轻量级融合回归器实现情感模仿强度的连续预测。

**💡 创新点**

创新点在于：①采用分阶段训练策略（先单模态再融合），避免跨模态训练初期的不稳定；②在融合阶段引入模态丢弃和有限编码器微调，提升对缺失或弱模态的鲁棒性；③探讨运动分支的可用性，验证其对多模态融合的补充作用。

**🔧 技术方法**

使用的技术包括：DINOv2 提取视觉帧级嵌入；自定义 wav2vec2+ 语音特征；GTE/BERT 文本嵌入；OpenFace AU+pose 运动序列；Masked attention pooling、GELU 激活、MLP 编码器；融合采用 MLP 轻量回归器；损失为 CCC 与 MSE 的加权组合；AdamW 优化器与梯度裁剪。

**📊 数据集**

使用 Hume‑ABAW10 Emotional Mimicry Intensity Challenge 数据集，共 12,660 个样本（8,072 训练 + 4,588 验证），包含 6 维连续情感强度标签。

**📈 对比分析**

模型在验证集上以平均 Pearson 相关系数衡量；最佳配置为文本+音频+视觉+运动的 4:1 训练/验证比例，验证 Pearson 为 0.4722，测试集 Pearson 达 0.57，排名第三；与冠军（0.55）相距不大，但方案更简单且可复现。

**⚠️ 局限性**

局限性包括：运动分支增益微弱，未充分挖掘面部动态信息；对缺失文本采用零向量填充，缺乏更高阶的缺失处理；数据高度不平衡、标签稀疏，仍影响模型泛化；未使用更复杂的跨模态注意力或动态融合机制。

---

## 112. Faster Completion, Less Learning: Generative AI Reduced Study Time on Math Problems and the Knowledge They Build

**arXiv ID:** 2605.21629 | [PDF](https://arxiv.org/pdf/2605.21629v1)

**作者:** Sina Rismanchian `[一作]` (University of California, Irvine), Eyad Kurd-Misto `[通讯]` (McGraw Hill)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用3.2百万ALEKS学习交互日志和PPL测评数据，比较AI可转录（文本题）与不可转录（图形题）问题在ChatGPT后学习时间和保留率的变化。

**💡 创新点**

首次以大规模自然观察的方式，通过问题类型的AI易受性差异和监考对照，揭示生成式AI对学生学习行为和长期成绩的实证影响，并提出“认知投降”行为特征。

**🔧 技术方法**

采用时间分解固定效应回归、事件研究、逻辑回归固定效应以及随机置换检验，区分AI可转录与不可转录题目，评估时间与准确率的差异。

**📊 数据集**

使用3.2M ALEKS学习交互记录（2015‑2025）和12.2M ALEKS PPL响应时间/保留题目（6.7K条）数据集。

**📈 对比分析**

通过监考与非监考条件下响应时间与保留结果的差异估计，发现高校文本题学习时间下降约26.9%，保留概率下降约25%，而非监考环境下则出现相反效应，表明效应显著且与AI使用相关。

**⚠️ 局限性**

未直接观测AI使用，学习与保留样本不同，PPL测评不等同实验室保留测试，无法追踪个体学习‑保留链，AI易受性划分可能存在误判。

---

## 113. Real-time, EDM-inspired sonfication of the activity of a supercomputer

**arXiv ID:** 2605.21874 | [PDF](https://arxiv.org/pdf/2605.21874v1)

**作者:** Marco Alunno `[一作]` (Universidad EAFIT), Paolo Bientinesi `[通讯]` (Umeå Universitet)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对瑞典Umeå大学的Kebnekaise超级计算机进行实时EDM风格的数据音频化，用以持续监测系统状态；

**💡 创新点**

在信息性监测与实时生成、无穷长音乐的结合上实现了创新，采用基于音乐风格的结构化音频化而非传统调试式音频化；

**🔧 技术方法**

利用Slurm收集的进程数、物理内存占用率、I/O流量数据，经过滑动窗口归一化后映射到SuperCollider的节奏、音高、混响等音频参数；

**📊 数据集**

使用Kebnekaise系统的实时节点数据（95节点、10分区），采集频率约为每15秒一次；

**📈 对比分析**

通过将每层音轨轮流置于前景并提供GUI切换，降低信息遮蔽；虽然未给出数值性能指标，但作者报告音频实时性良好且能维持长时间监测；

**⚠️ 局限性**

局限于最多十个分区的音频化，数据维度仅三项，缺乏用户交互界面及跨系统的泛化实现，且未对音频识别效果进行客观评估。

---

## 114. CompPow: A Case for Component-level GPU Power Management

**arXiv ID:** 2605.21847 | [PDF](https://arxiv.org/pdf/2605.21847v1)

**作者:** Shaizeen Aga `[一作]` (Advanced Micro Devices Inc), Mohamed Assem Ibrahim `[通讯]` (Advanced Micro Devices Inc)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在现代GPU内部组件级（XCD、IOD、HBM）的功耗管理，提出通过组件感知（component‑aware）方法对不同ML操作（GEMM、all‑gather）在单核与并发执行场景下进行能效与性能优化；

**💡 创新点**

创新点在于使用FinGraV获得高精度组件级功耗剖面，揭示不同算子对GPU组件的功耗分布差异，并基于此提出组件感知的频率与功率调节策略，能实现约10%能效提升与5%性能提升，同时给出软硬协同设计建议；

**🔧 技术方法**

技术手段包括FinGraV细粒度功耗测量、AMD Instinct MI300X实验平台、GPU级功率/频率限制、组件功率重分配模拟，以及软件层面的组件亲和性与关键性提示机制；

**📊 数据集**

实验基于LLaMA、Mistral等大型语言模型的训练/推理负载，评估其GEMM和all‑gather kernel的尺寸与算子特性；

**📈 对比分析**

对比了无功率限制、统一功率限制与组件感知频率限制三种策略，测量能耗、性能损失和加速效果；组件感知频率限制在all‑gather上平均节能10.13%并损失1.36%性能，GEMM功率重分配模拟提升4%单核GEMM性能、5%并发执行性能；

**⚠️ 局限性**

局限性包括：GPU仅公开XCD频率控制，缺乏对IOD/HBM的独立调节；实验仅在MI300X单机上进行，未涵盖跨节点功耗协同；未对注意力等其他ML算子进行评估；

---

## 115. Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws

**arXiv ID:** 2605.21803 | [PDF](https://arxiv.org/pdf/2605.21803v1)

**作者:** Nandan Kumar Jha `[一作]` (New York University), Brandon Reagen `[通讯]` (New York University)

**通讯引用:** 3746 | [OpenAlex ID](https://openalex.org/A5089173037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了优化器对Transformer FFN表示谱尺度律的影响，量化不同优化器（AdamW、Muon、NorMuon、Dion）在固定架构下如何将宽度扩展转化为有效谱容量；

**💡 创新点**

首次将优化器视为尺度律的关键轴，揭示优化器不仅影响收敛速度，还决定表示的谱结构、频率分布以及与架构的相互作用；

**🔧 技术方法**

采用谱分析（软、硬谱秩和Renyi有效秩）评估FFN表示，训练GPT式解码器模型，探索不同优化器、更新秩、注意力秩、RoPE/NoPE等干预；

**📊 数据集**

使用FineWeb-Edu文本数据集，训练160M和350M规模的GPT模型，进行FFN宽度（1-8倍）和不同优化器的超参数扫描；

**📈 对比分析**

通过比较各优化器的硬/软谱秩指数、硬软差异、频率分布下的尺度指数，发现Muons等矩阵化优化器在中尾频率上实现近线性硬秩增长（β≈1），而AdamW及低秩Dion表现出显著弱化；相同验证损失下仍存在谱结构差异，表明匹配loss不等价于匹配表示；

**⚠️ 局限性**

实验局限于GPT解码器、160M/350M规模，未覆盖更大规模或其他模型结构；仅评估FFN谱容量，未直接关联下游任务性能；优化器与架构交互的因果机制仍待深入研究。

---

## 116. An Open-Source Framework to Emulate Delay and Disruption Tolerant Networks for International Space Station Communication

**arXiv ID:** 2605.21624 | [PDF](https://arxiv.org/pdf/2605.21624v1)

**作者:** Krit Grover `[一作]` (University of Toronto Scarborough), Marcelo Ponce `[通讯]` (University of Toronto Scarborough)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了一个完整的、开源的、面向国际空间站通信的DTN捆绑协议全栈系统，包含安全块（BAB/PIB/PCB）、分片与重组、优先级排队、托管转移、自动重传，并配备了交互式3D可视化前端。

**💡 创新点**

创新点在于：①把DTN协议从理论带到实践，提供可读、可编辑的完整代码，方便教学和实验；②融合物理层链路预算、轨道动力学与网络协议的实时计算；③支持两种运行模式（纯模拟与Mininet仿真），并通过可视化展示包路由、分片、加密等细节；④实现了端到端安全与链路级安全的多层加密与认证，且开源共享。

**🔧 技术方法**

使用技术包括：后端 Python + FastAPI + WebSocket；Skyfield + SGP4 轨道计算；Mininet + Linux traffic control；前端 React + TypeScript + Three.js + Recharts + Tailwind CSS；数据库 SQLite；加密技术 AES-256-CBC、HMAC‑SHA256、PBKDF2；网络协议实现 RFC 5050/9171；性能分析工具 Wireshark、Mininet 自带统计。

**📊 数据集**

主要数据来源是公开的 TLE 数据（Celestrak）用于轨道计算；9 颗地面站的经纬度与高度坐标；以及根据 ISS 轨道生成的合成接触窗口表。没有使用传统机器学习或大规模数据集。

**📈 对比分析**

通过在模拟模式和 Mininet 模式下对同一套 9 颗地面站网络进行实验，测量交付率、时延、跳数、重传、加密开销、分片影响与可扩展性。实验结果显示：交付率始终 100%；平均时延随负载下降而减小；安全开销约 33%（小包可达 68%），处理时延 <0.1 ms；分片和重组无影响；最多可支持 50 个并发包仍保持 100% 交付；Mininet 模式下即使链路损失率高达 30% 仍保持 100% 交付，且 TCP 在同一中断调度下完全失效。比较方法包括与传统 TCP 的失败率对比，展示 DTN 的优势。

**⚠️ 局限性**

局限性包括：①线程安全仅依赖 GIL，未在所有共享结构上加锁；②仅支持单 ISS 与 9 颗地面站的部分网格；③对多卫星或星座的支持尚未实现；④Mininet 需要 root 权限且资源消耗较高；⑤数据库错误恢复会导致数据丢失；⑥物理层模型未考虑真实大气干扰或频偏精细细节，主要以合成模型为主。

---

## 117. Discovering Entity-Conditioned Lag Heterogeneity: A Lag-Gated Neural Audit Framework for Panel Time Series

**arXiv ID:** 2605.21542 | [PDF](https://arxiv.org/pdf/2605.21542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 118. PITMuS: A Tool for Automated Bug Dataset Generation via Source-Level Mutant Reconstruction

**arXiv ID:** 2605.21930 | [PDF](https://arxiv.org/pdf/2605.21930v1)

**作者:** Tasfia Tasnim `[一作]` (University of Texas at Dallas), Soneya Binta Hossain `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一款工具，能够把PIT产生的字节码级变异报告恢复为源代码级的原–变体方法对，并生成可直接使用的结构化数据集。

**💡 创新点**

创新点在于利用XML报告、编译后的字节码调试信息和源码AST解析三方数据，精准定位并重建变异操作，使得变异信息从报告级转为可编辑的源代码级别，且不需要修改底层变异引擎。

**🔧 技术方法**

核心技术包括Python与Java的互操作、JavaParser对源代码的AST解析、PIT的XML报告解析，以及基于字节码索引的位置信息对齐与变异语句重写。

**📊 数据集**

实验使用了八个开源Java项目（bcel、commons‑beanutils、commons‑dbutils、commons‑jexl3、commons‑lang3、http‑request、joda‑time、jsoup），共恢复 69,198 对原–变体方法，覆盖 1,913 个源文件。

**📈 对比分析**

与原始PIT报告对比，恢复率达到 99.96%，所有主要变异操作符均保持 99.69% 以上，生成的数据集保留了 66% 的 Javadoc 上下文，可直接用于模型训练、工具评估和实验研究。

**⚠️ 局限性**

局限性包括：只能处理单行表达式的变异；多行表达式导致位置不匹配而未恢复；多字节码变异映射到同一源级变异时会被跳过；当前仅支持 Java，且依赖于编译时保留的调试信息。

---

## 119. Format-Constraint Coupling in Knowledge Graph Construction from Statistical Tables

**arXiv ID:** 2605.21974 | [PDF](https://arxiv.org/pdf/2605.21974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 120. PocketAgents: A Manifest-Driven Library of Autonomous Defense Agents

**arXiv ID:** 2605.21694 | [PDF](https://arxiv.org/pdf/2605.21694v1)

**作者:** Sidnei Barbieri `[一作]` (Aeronautics Institute of Technology), Lourenço Alves Pereira Júnior `[通讯]` (Aeronautics Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PocketAgents，一个通过清单驱动的自治防御代理库，使 LLM 能在受限、可验证的边界内做出防御决策

**💡 创新点**

通过将代理安装为仅包含数据文件（清单、提示和上下文）并在运行时使用强类型边界实现可审计、可扩展的自治防御；对 LLM 输出进行结构化验证与实体根据信号绑定

**🔧 技术方法**

使用大语言模型（OpenAI GPT‑4、Claude、Anthropic），构造代理提示，设计类型化报告与验证逻辑，并在 Perry 网络欺骗测试床中实现共享运行时与动作适配器

**📊 数据集**

在 Perry 的 DarkSide/EquifaxSmall 小型企业拓扑实验场景中进行 18 次闭环试验，涵盖三种模型后端和两类代理（Command‑and‑Control 与 Exfiltration）

**📈 对比分析**

通过统计 13 次成功网络封锁、4 次模式校验失败、1 次有效无动作决策，比较不同模型后端在合同合规性与决策正确性上的表现；平均从模型查询到封锁的时间为 26–76 秒

**⚠️ 局限性**

局限性包括仅测试单一网络封锁动作、有限的代理类型与攻击场景、对外部攻击者对模型的潜在诱导未评估、以及依赖于 Perry 的仿真环境

---

## 121. Distributed Multi-Coverage for Robot Swarms

**arXiv ID:** 2605.21686 | [PDF](https://arxiv.org/pdf/2605.21686v1)

**作者:** Mariem Guitouni `[一作]` (University of Houston), Aaron T. Becker `[通讯]` (University of Houston)

**通讯引用:** 2033 | [OpenAlex ID](https://openalex.org/A5062449649)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了一种在本地感知、局部通信、无全局协调条件下实现机器人队列多重覆盖的分布式算法。

**💡 创新点**

创新点在于将 Lloyd 分布式探索、边际成本投标优化以及局部精细化缩小覆盖成本的三阶段流程相结合，并通过实验验证其可行性。

**🔧 技术方法**

采用 Voronoi 聚类、Welzl 最小包围圆、边际成本投标、资产交换和局部重叠消除等技术。

**📊 数据集**

使用合成的均匀分布资产数据集 uni_sm 与 uni_fix_n，并在动态资产出现实验中加入 70 个新资产。

**📈 对比分析**

与集中式整数规划基线比较，分布式方案在不同规模下保持 <5 秒计算时间，速度提升 12–410 倍，最终成本比最优多 45–70%，但仍满足 κ(p) 覆盖需求。

**⚠️ 局限性**

局限包括对通信半径 r_comm 与感知半径 r_max 的敏感性、随着队列规模增大导致的最优性缺口扩大，以及局部信息导致的持续过度覆盖。

---

## 122. AgroVG: A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding

**arXiv ID:** 2605.22034 | [PDF](https://arxiv.org/pdf/2605.22034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 123. A geometric modelling framework to support the design of heterogeneous lattice structures with non-linearly varying geometry

**arXiv ID:** 2605.21971 | [PDF](https://arxiv.org/pdf/2605.21971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 124. Ishigaki-IDS-Bench: A Benchmark for Generating Information Delivery Specification from BIM Information Requirements

**arXiv ID:** 2605.22079 | [PDF](https://arxiv.org/pdf/2605.22079v1)

**作者:** Ryo Kanazawa `[一作]` (ONESTRUCTION Inc.), Daiho Nishioka `[通讯]` (ONESTRUCTION Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个专门用于评估大语言模型在建筑信息建模（BIM）领域生成 IDS（Information Delivery Specification）XML 的基准（Ishigaki-IDS-Bench），并设计了两阶段的评估协议。

**💡 创新点**

① 结合 IFC/IDS 标准提出了针对结构化生成的两阶段评估（IDSAuditTool+facet‑level scorer）。② 通过专家验证的 166 条实例覆盖多语言、多输入格式（自然语言、CSV）与多施工领域，提供了高质量的 gold IDS。③ 在基准中同时评估语法合法性、标准合规性和内容一致性，填补了以往仅关注 JSON/SQL 之类的通用结构化生成的空白。

**🔧 技术方法**

使用大语言模型（10 种闭源与开源模型）进行零样本生成，采用 IDSAuditTool 对生成的 IDS 进行格式、结构与内容的合规性检测，并通过 facet scorer 与 gold IDS 逐块对比计算 Precision、Recall、F1。

**📊 数据集**

Ishigaki-IDS-Bench 数据集：166 条专家编写、验证的实例，涵盖 83 个实用场景，输入为日语/英语自然语言或 CSV，输出为标准化 IDS XML，包含 IFC 版本、施工领域等元数据。

**📈 对比分析**

通过零样本实验与 10 种 LLM 进行对比，评估指标包括 Stage‑1 的 Processability、Structure、Content 通过率以及 Stage‑2 的 macro‑average F1/Recall/Precision。最佳模型 GPT‑5.5 在内容一致性上仅达 27.7% pass，macro‑F1 为 65.6%；单轮 vs 多轮、日语 vs 英语、CSV vs 自然语言对性能产生显著影响。

**⚠️ 局限性**

局限性：仅覆盖了 facet‐level 中的 ElementType、BasicField、AdditionalInfo；未包含其他 facet；样本量相对较小；基准不包含真实项目文件和噪声数据；使用的属性集正则表达式为自定义；仅进行零样本评估，未尝试 few‑shot、微调或语法约束生成。

---

## 125. Generative Conversational Recommender System

**arXiv ID:** 2605.21987 | [PDF](https://arxiv.org/pdf/2605.21987v1)

**作者:** Sixiao Zhang `[一作]` (Nanyang Technological University), Cheng Long `[通讯]` (Nanyang Technological University)

**通讯引用:** 5348 | [OpenAlex ID](https://openalex.org/A5080939756)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个完全生成式的会话推荐系统，在同一自回归框架下同时完成对话生成与推荐；

**💡 创新点**

通过引入语义ID表示和结构化生成范式，将推荐任务拆分为意图预测、目标物品预测与自然语言生成三步，形成端到端可优化的决策链；

**🔧 技术方法**

采用残差量化变分自编码器（RQ‑VAE）生成离散语义ID，利用解码器仅模型在结构化目标序列上进行细粒度微调，并通过MODE标记和受限解码实现精确推荐；

**📊 数据集**

在电影对话推荐基准ReDial和Inspired上进行实验，利用其多轮对话与物品元数据；

**📈 对比分析**

相较于检索式与模块化的对话推荐方法以及先前的生成式推荐模型，GCRS在Recall@1、NDCG@k、MRR@k等指标上平均提升约20‑30%，同时保持或提升对话流畅度与多样性；

**⚠️ 局限性**

局限性在于仍依赖预训练语言模型的规模与质量，语义ID的构造需预先训练文本编码器，且在多域或大型项目中可能需要更大代码库与更细粒度的实体表示。

---

## 126. RiT: Vanilla Diffusion Transformers Suffice in Representation Space

**arXiv ID:** 2605.21981 | [PDF](https://arxiv.org/pdf/2605.21981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 127. Alike Parts: A Feature-Informed Approach to Local and Global Prototype Explanations

**arXiv ID:** 2605.21646 | [PDF](https://arxiv.org/pdf/2605.21646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 128. From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning

**arXiv ID:** 2605.22074 | [PDF](https://arxiv.org/pdf/2605.22074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 129. An Improved Adaptive PID Optimizer with Enhanced Convergence and Stability for Deep Learning

**arXiv ID:** 2605.21968 | [PDF](https://arxiv.org/pdf/2605.21968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. Auction-Consensus Algorithm with Learned Bidding Scheme for Multi-Robot Systems

**arXiv ID:** 2605.21932 | [PDF](https://arxiv.org/pdf/2605.21932v1)

**作者:** Jose Rodriguez `[一作]` (University of Texas at Rio Grande Valley), Qi Lu `[通讯]` (University of Texas at Rio Grande Valley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多机器人任务分配中，用强化学习训练神经网络来替代CBBA的手工贪婪评分函数，实现分散执行的学习增强型竞标-共识算法。

**💡 创新点**

创新点在于将中央训练与去中心化执行（CTDE）框架与PPO强化学习相结合，直接学习竞标策略以逼近全局最优，而非依赖传统贪婪评分，并通过实验验证其可扩展性和泛化能力。

**🔧 技术方法**

使用技术包括：集中式训练-去中心化执行（CTDE）、Proximal Policy Optimization（PPO）、神经网络竞标器（Neural Additive Model、LSTM、Set Transformer）以及基于混合整数线性规划的全局奖励塑造。

**📊 数据集**

训练数据集为1000个随机生成的二维MRTA场景（5机器人、10-20任务，工作空间尺寸25×25至55×55），验证集包含5、10、15、20机器人、任务比例2-4/机器人、工作空间相同的1000个未见场景。

**📈 对比分析**

与经典CBBA、NAM、LSTM三种竞标器比较，实验显示LSTM在中小规模团队（5-15机器人）可达到约87-90%的最优率，NAM在无超时时最佳但有显著收敛失败，CBBA整体最优率最低；但学习型方法在迭代次数上比CBBA慢（NAM最多21次，LSTM最多13次）。

**⚠️ 局限性**

局限性包括：NAM模型易出现超时；学习型竞标器收敛速度慢；Transformer及其混合版本训练不稳定；在更大规模或任务密集的环境下性能衰减，需进一步改进共识规则与特征空间。

---

## 131. Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines

**arXiv ID:** 2605.21958 | [PDF](https://arxiv.org/pdf/2605.21958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 132. SCI-Defense: Defending Manipulation Attacks from Generative Engine Optimization

**arXiv ID:** 2605.21948 | [PDF](https://arxiv.org/pdf/2605.21948v1)

**作者:** Xucheng Yu `[一作]` (University of Illinois Urbana-Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2822 | [OpenAlex ID](https://openalex.org/A5072244531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SCI-Defense框架，用于检测和抑制LLM驱动的搜索排名系统中的生成式引擎优化（GEO）攻击。

**💡 创新点**

创新点在于：①将三种互补检测组件（Perplexity、Semantic Integrity Scoring、Inter-Candidate Detection）组合；②定义四维语义操纵维度（Authority Attribution、Narrative Purposiveness、Comparative Claims、Temporal Claims）并在GPT-4o上实现评分；③识别并展示了GEO攻击的结构盲点，即通过语义相关性提升的“语义相关性操纵”攻击。

**🔧 技术方法**

技术主要包括：GPT-2语言模型计算困惑度；GPT-4o进行多维语义评分；跨候选文档相似度计算（embedding similarity）；阈值阈值调优和融合策略；以及对攻击样本的恢复与检测。

**📊 数据集**

使用了两个公开数据集：Amazon ProductBench（600条商品描述，涵盖6个类别）和MS MARCO Web Passages（600条信息检索段落，涵盖6个领域），每条描述分别注入String、Reasoning和Review三种GEO攻击。

**📈 对比分析**

与三种基线（PPL Filter、SafetyClf、Paraphrasing）对比，SCI-Defense在Amazon数据上实现Precision=1.000、Recall为0.952（Reasoning）/0.830（Review）、FPR=0.000；在MS MARCO上保持Precision=1.000、Recall为0.785（Reasoning）、0.008（Review）、FPR=0.000；Block@3/5为1.000，显示在检测和惩罚上均优于基线。

**⚠️ 局限性**

局限性包括：①对“语义相关性操纵”类攻击（SEO Stuffing、Specification Amplification、Use-Case Saturation等）检测不到；②需要手动调校阈值，可能在新领域或语言下需要重新验证；③依赖查询相关信息的缺失导致跨候选相似度的局限性。

---

## 133. HealthCraft: A Reinforcement Learning Safety Environment for Emergency Medicine

**arXiv ID:** 2605.21496 | [PDF](https://arxiv.org/pdf/2605.21496v1)

**作者:** Brandon Dent `[一作]` `[通讯]` (GOATnote Inc), Brandon Dent (GOATnote Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个面向急诊医学的强化学习环境HealthCraft，并公开了基于FHIR‑R4世界状态、24工具MCP接口的195项任务及其2,255条二元判定标准（其中515条为安全关键）。

**💡 创新点**

创新点包括：① 双层评分体系并加入硬性安全门槛，任何安全关键判定不满足即奖励为零；② 将真实临床互操作标准FHIR‑R4直接映射到环境状态；③ 将OpenEM临床知识与核心Craft架构结合，提供可追溯、可复现的任务生成；④ 公开完整Docker包、评估Harness和判定规则，支持跨模型评测。

**🔧 技术方法**

采用Corecraft的架构（PostgreSQL世界状态、FastMCP工具服务器、Docker化部署），结合Megatron+SGLang+GRPO训练循环；使用LLM判定器（Claude/ GPT）和确定性覆盖层进行评估；通过工具调用日志、规则验证、正则匹配等三种判定方式。

**📊 数据集**

利用OpenEM知识库（370个条件、152个混淆对、45个决策规则、44个评估属性）与FHIR资源生成器生成的14种实体类型（3,987条实体），以及基于OpenEM的临床情景。

**📈 对比分析**

对Claude Opus 4.6和GPT‑5.4进行三次试验，采用Pass@1、Pass@3、Pass^3、期望奖励和安全失效率等指标。Claude在Pass@1上达到约24.8%，GPT约12.6%；安全失效率分别为27.5%和34.0%；多步工作流程几乎全部失败（Claude 1.0%，GPT 0.0%）。

**⚠️ 局限性**

局限性包括：① 静态患者状态（无时间演化）；② 无延迟、重试或超时模拟；③ 变异工具无幂等性键；④ 评估完全基于确定性规则，LLM判定器可靠性仅为中等；⑤ 任务覆盖有限，缺少小儿、产科等子领域；⑥ 评估与训练奖励边界不匹配，训练安全性未验证。

---

## 134. Secure Coordination for Vertiport Sequencing in Advanced Air Mobility

**arXiv ID:** 2605.21771 | [PDF](https://arxiv.org/pdf/2605.21771v1)

**作者:** Jaehan Im `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 595 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究在感知不确定性下，基于遥测与监视数据的垂直机场着陆序列的安全协同策略。

**💡 创新点**

创新点在于将自利误报与恶意伪装建模为不同的鲁棒优化问题，并设计基于不确定性一致性集合的鲁棒序列规则。

**🔧 技术方法**

采用鲁棒优化、Stackelberg博弈与最优对抗问题等技术。

**📊 数据集**

目前尚未使用真实数据集，计划在代表性垂直机场场景下进行数值仿真。

**📈 对比分析**

通过比较基线序列、自利鲁棒序列与恶意鲁棒序列，在真实与伪造报告条件下评估效率损失与安全收益，预期鲁棒方案在误报情形下能显著降低延迟与调度成本。

**⚠️ 局限性**

局限包括对误报车辆集合 ℳ 的先验假设、缺乏实测验证以及对监视资源分配等扩展的进一步研究。

---

## 135. Requirements Perception Gap across Stakeholders: A Comparative Survey of Aged Care Digital Health Software

**arXiv ID:** 2605.21495 | [PDF](https://arxiv.org/pdf/2605.21495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 136. JMed48k: A Multi-Profession Japanese Medical Licensing Benchmark for Vision-Language Model Evaluation

**arXiv ID:** 2605.22080 | [PDF](https://arxiv.org/pdf/2605.22080v1)

**作者:** Yue Xun `[一作]` (Hong Kong Polytechnic University), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 14151 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了JMed48k多专业日语医疗执照考试基准，并对包含图像的题目进行配对去图像审核，细粒度分析模型对视觉信息的利用。

**💡 创新点**

创新点在于：① 覆盖11个医疗专业、覆盖2005–2025年两十年官方考试题目；② 提供8类图像标签与图像类型分布；③ 通过配对去图像审核拆分为四种答案转换状态，揭示专业间视觉依赖差异及医学专用模型对图像的有限利用。

**🔧 技术方法**

使用了视觉-语言模型（Gemini、GPT‑5、Claude、Qwen、Gemma、Llama、GLM、MedGemma等）在文本与图像输入下的推理；配对去图像审计方法；多标签图像注释与统计；宏观与微观准确率对比。

**📊 数据集**

使用JMed48k完整语料（48,862题、9,646图像）及其最新5年评估子集JMed48k‑Eval（12,484题、2,579图像）作为评测数据集。

**📈 对比分析**

通过分别计算文本仅、图像伴随的准确率，并在图像伴随题目上执行配对去图像审核，得到p_11、p_10、p_01、p_00四种状态。结果显示：专有模型在图像中提升最大可达+39.8点（PHN），但医学专用模型几乎无增益；整体图像题目准确率低于文本题目；不同专业间图像效益七倍以上。

**⚠️ 局限性**

局限性：评测仅基于执照考试题目，未检验真实临床推理；配对去图像审核只能观测答案变化，不能判断模型是否真正“看懂”图像；数据集仅覆盖日语环境，可能难以直接迁移到其他语言或文化。

---

## 137. Quality-Assured Fuzz Harness Generation via the Four Principles Framework

**arXiv ID:** 2605.21824 | [PDF](https://arxiv.org/pdf/2605.21824v1)

**作者:** Ze Sheng `[一作]` (Texas A&M University), Jeff Huang `[通讯]` (Texas A&M University)

**通讯引用:** 3632 | [OpenAlex ID](https://openalex.org/A5052381120)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出QuartetFuzz，一个基于四项源级正确性原则（逻辑正确、API 协议、边界尊重、入口充分）的LLM驱动自动化fuzz harness生成系统，能够在生成前对 harness 进行完整检查并在真实项目中实现低误报的漏洞发现。

**💡 创新点**

创新点在于定义并实现四项原则的可检查形式，并通过 Adversarial Probing 机制让LLM在生成时即验证并修复 P1–P4 违规，使得生成的 harness 在提交前已满足逻辑、协议与安全边界要求，从而显著降低误报并提升发现率。

**🔧 技术方法**

采用了 Claude Sonnet/Opus LLM、四阶段 agent pipeline（Logic Group 选择、API 研究、静态驱动编译、Adversarial Validation）、静态 call‑graph 分析、API 协议挖掘、Adversarial Probing、OSS‑Fuzz/LibFuzzer 以及 ASan/LSan 进行动态检测。

**📊 数据集**

使用 70 项 OSS‑Fuzz 生产 harness 进行审核，并构建 100 条 Gold 标注 harness 数据集（覆盖 C/C++、Java、JavaScript），作为评估与基准。

**📈 对比分析**

与 OSS‑Fuzz‑Gen 与 PromeFuzz 在 100 个案例上对比，QuartetFuzz 的覆盖率与人类 gold 几乎相当（TOST ±2pp，p<1e-10），产品率 96/100，误报率仅 4.8%，在覆盖率上比基准高 6.9–8.3pp 及 4.1–5.2pp，且成本更低。

**⚠️ 局限性**

局限性包括仅支持 C/C++（有限 Java/JS）与 libFuzzer/ASan，缺乏多语言/多fuzzer 支持；Adversarial Probing 与静态检查仍是近似实现，无法完全决定 P1–P4；系统依赖 OSS‑Fuzz 架构，迁移到其他生态需额外工作。

---

## 138. Engineering Hybrid Physics-Informed Neural Networks for Next-Generation Electricity Systems: A State-of-the-Art Review

**arXiv ID:** 2605.21903 | [PDF](https://arxiv.org/pdf/2605.21903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 139. SENTIL: A Runtime Verification Tool for Probabilistic Temporal Logic

**arXiv ID:** 2605.21676 | [PDF](https://arxiv.org/pdf/2605.21676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 140. The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity

**arXiv ID:** 2605.21492 | [PDF](https://arxiv.org/pdf/2605.21492v1)

**作者:** Drake Caraker `[一作]` (Independent Researchers), David Rhoads `[通讯]` (Independent Researchers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在特征共线性存在的情况下，证明单模型的特征重要性排名无法同时满足可信、稳定与完整，并提出基于SHAP的集成方法（DASH）以及相应的诊断工具和定量界限；

**💡 创新点**

首次在可解释AI中给出归因不可行性定理、定量化不同模型类别的归因不平衡比率、通过SHAP平均实现稳定且最优的集成归因，并通过Lean 4机械化验证证明定理的严谨性；

**🔧 技术方法**

利用Rashomon属性与对称性不等式推导归因不平衡比率，使用梯度提升树的高斯条件推导、α‑量化；采用统计检验（Z‑检验、单模型屏蔽）、贝叶斯估计、线性无偏估计与Cramér‑Rao最优性；并用Lean 4形式化逻辑证明；

**📊 数据集**

在77个公开数据集（如Breast Cancer、Wine、Heart Disease、Ames、Communities等）以及合成高斯数据上进行实证验证；

**📈 对比分析**

与单模型SHAP、bootstrap SHAP、子采样SHAP、CI SHAP等方法对比，平均翻转率由≈48 %降至3–5 %（M = 25），且保持无偏；DASH在稳健性与信息利用上达到Pareto最优；诊断工具Z‑检验与单模型屏蔽的精准率>90 %；

**⚠️ 局限性**

局限性包括：仅在共线性情形下说明不可行性；集成方法牺牲组内完整性（返回平等）；需要训练多模型，对计算成本有一定影响；对局部解释的稳定性证明相对保守；在非高斯或极端稀疏情况下诊断性能下降。

---

## 141. Thermo-VL: Extending Vision-Language Models to Thermal Infrared Perception

**arXiv ID:** 2605.21882 | [PDF](https://arxiv.org/pdf/2605.21882v1)

**作者:** Rusiru Thushara `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 23025 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种波长感知的视觉-语言模型，将对齐的 RGB 与热红外图像融合，以提升低光照和跨光谱推理能力。

**💡 创新点**

创新点包括：1) 文本引导的双注意力融合模块，在保持 RGB–语言接口冻结的前提下，仅更新热编码器最后几层与融合模块，实现参数高效适配；2) 通过门控残差将热信息注入 RGB 流；3) 引入多项辅助正则（RGB‑热对齐、跨模态对比、门控熵、RGB 区块遮挡）增强跨模态耦合；4) 构建大规模对齐 RGB‑热 QA 训练集和 Thermo‑VL‑Bench 评估基准。

**🔧 技术方法**

使用的技术包括：冻结 Molmo‑7B 视觉‑语言骨干，ViT‑L/14 热编码器，文本引导的多头注意力融合，门控残差注入，InfoNCE 对比损失，Token‑级对齐损失，门控熵正则，RGB 区块遮挡数据增强，以及 AdamW 双组参数优化。

**📊 数据集**

使用的数据集有：① 约 19,500 张像素级对齐的 RGB‑热图像对，约 54K QA 对，用于训练；② 通过 PixMo‑cap‑qa 生成的合成热图像；③ Thermo‑VL‑Bench（3,148 对图像、15K QA）作为评估基准；④ 公开的 KAIST、FLIR‑ADAS、OSU/OTCBVS 用于构建训练对齐图像；⑤ 公开的 RGB‑热 VQA 评估集 RGB‑Th‑Bench 用于外部对比。

**📈 对比分析**

评估方法：在 Thermo‑VL‑Bench 的 RGB‑only、热‑only、RGB+热 三种输入条件下对模型进行自动回归生成并计算准确率；在 RGB‑Th‑Bench 的 RGB‑Txt 与 RGB‑Th‑Txt 条件下进行比对。实验结果显示：① RGB‑only 性能保持不变；② 热‑only 从 81.20% 提升至 82.23%；③ RGB+热 从 75.79% 提升至 88.96%，整体得分最高；④ 在 RGB‑Th‑Bench 上，RGB‑Th‑Txt 由 60.84% 提升至 69.15%。

**⚠️ 局限性**

局限性：1) 仅支持对齐的 RGB‑热双模输入，对偏移或缺失模态的鲁棒性未知；2) 合成热图与模型生成 QA 可能引入偏差；3) Thermo‑VL‑Bench 为模型生成并人工筛选的二元 QA 基准，尚未完全人工编写；4) 作为研究原型，尚未针对安全关键场景进行充分验证。

---

## 142. Bridging the Cold-Start Gap: LLM-Powered Synthetic Data Generation for Natural Language Search at Airbnb

**arXiv ID:** 2605.21812 | [PDF](https://arxiv.org/pdf/2605.21812v1)

**作者:** Wendy Ran Wei `[一作]` (Airbnb), Sanjeev Katariya `[通讯]` (Airbnb)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5087282245)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型生成合成查询和标签，为Airbnb自然语言搜索解决冷启动问题

**💡 创新点**

将对比式列对与用户研究种子查询结合，提出三种提示策略（seed_controlled、seed_freeform、variety），生成既真实又多样化的合成数据

**🔧 技术方法**

基于LLM（如GPT‑4/ChatGPT）进行提示生成、对比式生成与虚拟评判器，并用嵌入检索和排序模型训练

**📊 数据集**

使用Airbnb搜索会话、房源属性、约500条用户研究种子查询作为输入，生成百万级合成示例

**📈 对比分析**

与无种子对比生成基线和真实用户查询对比，KL散度从4.95降至0.66，属性分布KL降至0.04，检索/排序模型在合成数据上的准确率从~0.9降至~0.8，提供更具辨别力的评估

**⚠️ 局限性**

对比式生成标签覆盖有限、LLM自我偏好风险、合成查询长度分布与真实用户仍有偏差，且最终仍需迁移到真实用户数据

---

## 143. Exploring the Effectiveness of Using LLMs for Automated Assessment of Student Self Explanations in Programming Education

**arXiv ID:** 2605.21614 | [PDF](https://arxiv.org/pdf/2605.21614v1)

**作者:** Arun-Balajiee Lekshmi-Narayanan `[一作]` (University of Pittsburgh), Peter Brusilovsky `[通讯]` (University of Pittsburgh)

**通讯引用:** 22115 | [OpenAlex ID](https://openalex.org/A5037674585)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造二分类任务，对比了基于LLM的评估和传统语义相似度方法在学生代码解释自动评分中的效果。

**💡 创新点**

创新点在于：①首次使用合成负样本平衡SelfCode2.0数据集来比较两类方法；②采用LLM-as-a-Judge方式进行零样本评估，无需阈值或微调；③通过提示工程明确“代码行为”rubric，提高LLM的鲁棒性。

**🔧 技术方法**

主要技术包括：LLM (GPT‑3.5 Turbo 与 GPT‑OSS 20b 生成负例)、语义相似度工具 DeepTutor、RoBERTa、GPT‑3 embeddings；采用5‑折交叉验证与阈值调优（仅用于语义相似度），并使用句子嵌入与余弦相似度。

**📊 数据集**

使用了 SelfCode2.0 数据集（3019 对，1854 单句对，1794 正例、60 负例），并利用 GPT‑OSS 20b 生成 3 条合成错误解释，构成平衡数据集。

**📈 对比分析**

比较方法：对每个模型在平衡数据集的 5‑折测试集上计算 F1 与 Accuracy。LLM‑Behavior 的 F1=0.98、Accuracy=0.96，显著优于 DeepTutor（F1=0.69、Acc=0.60）、RoBERTa（F1=0.70、Acc=0.61）和 GPT‑SE（F1=0.72、Acc=0.65）。

**⚠️ 局限性**

局限性包括：仅评估了解释的正确性，未提供针对性反馈；数据集专家评估存在多样性问题；合成负样本的真实性验证有限；未做系统化提示优化；未探索多语言或更高层次学习场景；未尝试更先进的偏好对齐或鲁棒性增强方法。

---

## 144. Virtual 3D H&E Staining from Phase-contrast Back-illumination Interference Tomography

**arXiv ID:** 2605.22000 | [PDF](https://arxiv.org/pdf/2605.22000v1)

**作者:** Anthony Song `[一作]`, Nicholas Durr `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了首个 voxel‑级配准的 3D Back‑illumination Interference Tomography 与荧光核染色数据集，并提出基于 Vision Transformer 的无监督虚拟 H&E 染色框架。

**💡 创新点**

创新点包括① voxel‑级配准的 3D 数据集 HistoBIT3D；② 双向多尺度内容一致性损失与跨域样式注入的 GAN 结构，解决 BIT 位移变异对比问题。

**🔧 技术方法**

采用 Vision Transformer CycleGAN、AdaIN 样式融合、双向多尺度内容一致性损失、以及 Cellpose 等评估工具。

**📊 数据集**

使用 HistoBIT3D 数据集，包含约 5,000 张 512×512 的 BIT 与荧光核对齐体积，以及无配对的 2D H&E 图像。

**📈 对比分析**

通过 FID/KID 与 3D Dice、HD95、核体积等指标与 CycleGAN、STABLE、CycleDiffusion、UVCGANv2 等方法对比，展示更低的 FID/KID、更高的 3D Dice 与更小的 HD95。

**⚠️ 局限性**

局限在于 BIT 与荧光之间对比差异仍未完全消除，缺乏半监督或跨模态校准，且在更大尺度或不同组织时需要进一步验证。

---

## 145. What Counts as AI Sycophancy? A Taxonomy and Expert Survey of a Fragmented Construct

**arXiv ID:** 2605.21778 | [PDF](https://arxiv.org/pdf/2605.21778v1)

**作者:** Meryl Ye `[一作]` (Carnegie Mellon University), Steve Rathje `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对70篇关于AI sycophancy的论文进行系统综述，构建了一个基于Referent与Explicit两个维度的分类法，并对106位领域专家进行问卷调查，以验证和细化该分类法。

**💡 创新点**

创新点在于提出了一个可操作的AI sycophancy分类框架，将“对立场的顺从”与“对个体的顺从”以及“显式与隐式表达”这两个维度进行组合，揭示了不同类型的sycophancy在实践中的分布与评价差异，并通过专家问卷提供了经验验证。

**🔧 技术方法**

主要技术包括系统性文献检索与编码、两位作者独立标注的税onomies编码、Qualtrics在线问卷设计、交叉效应的多层线性回归分析以及可靠性与效度检验。

**📊 数据集**

使用的数据集为70篇相关论文的元信息和描述性数据，以及106位专家的问卷响应；此外，还参考了ELEPHANT、SycEval等现有benchmark的数据与评估结果。

**📈 对比分析**

方法上通过对问卷项目在税onomies坐标上的回归分析，发现“显式程度”对Person类行为的sycophancy评分具有显著交互效应，而Position类行为则不受显式程度影响；结果说明不同子类型的sycophancy需要针对性评估与干预。

**⚠️ 局限性**

局限性包括文献综述并非全覆盖、专家样本可能存在回应偏倚和行业代表性不足、问卷设计中对负面极端的解释可能不统一，且研究仅关注输出行为而未直接测量长期交互中的下游影响。

---

## 146. ChronoMedicalWorld: A Medical World Model for Learning Patient Trajectories from Longitudinal Care Data

**arXiv ID:** 2605.21963 | [PDF](https://arxiv.org/pdf/2605.21963v1)

**作者:** Jiangyuan Wang `[一作]` (Beijing KidneyTec Medical Technology Co., Ltd.), Fuman Han `[通讯]` (Beijing KidneyTec Medical Technology Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种通用的、基于动作条件的潜在世界模型框架（ChronoMedicalWorld Model），用于学习慢性疾病的长期病程轨迹；

**💡 创新点**

创新点在于（1）将患者–健康教练对话作为一类正式的动作输入；（2）采用Joint‑Embedding Predictive Architecture与SIGReg正则化的六项物理形状先验损失；（3）引入rollout‑prefix训练协议，消除训练与推理的闭环误差；（4）以CKD年均eGFR预测为案例验证框架的通用性。

**🔧 技术方法**

核心技术包括：潜在状态编码器、宽动作编码器（结构化+文本嵌入）、GRU递归转移模块、双通道预测头（观测与潜在），SIGReg正则化、斜率一致性、连续性Huber、跳跃惩罚等损失，和闭环rollout‑prefix训练。

**📊 数据集**

使用来自KidneyOnline平台的2,232名慢性肾病患者的年度随访数据，包括实验室指标、药物干预、血压、尿蛋白以及平台聊天记录，合计15,070个病人年。

**📈 对比分析**

与GPT‑5.5结构化提示基准进行对比，采用动态50%历史rollout评估。CKD实例在测试集上的MAE从7.964降至7.384（-7.28%），RMSE从11.069降至10.256（-7.35%），提升主要来自对话动作通道。

**⚠️ 局限性**

局限包括：仅在单一平台、年龄偏年轻的单中心数据上验证；动作仅为年度二元指示，未覆盖剂量/依从性；对话嵌入为年度汇总，未细分内容；模型非因果推断，未与时间相关的协变量平衡方法比较。

---

## 147. Double descent for least-squares interpolation on contaminated data: A simulation study

**arXiv ID:** 2605.21494 | [PDF](https://arxiv.org/pdf/2605.21494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. AutoMCU: Feasibility-First MCU Neural Network Customization via LLM-based Multi-Agent Systems

**arXiv ID:** 2605.21560 | [PDF](https://arxiv.org/pdf/2605.21560v1)

**作者:** Penglin Dai `[一作]` (Southwest Jiaotong University), Lixin Duan `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7111 | [OpenAlex ID](https://openalex.org/A5080093489)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AutoMCU，一种以可行性为先的LLM多智能体系统，实现针对MCU约束的端到端神经网络定制。

**💡 创新点**

创新点在于硬件循环生成（hardware-in-the-loop architecture generation）、状态隔离的多智能体调度（MSIM）以及闭环可行性优先搜索，显著降低搜索成本并保证部署可行。

**🔧 技术方法**

利用大型语言模型（DeepSeek‑V3.2/MiMo‑V2‑Flash/Qwen‑Plus）生成结构化模型方案，结合STM32Cube.AI等硬件后端进行实时可行性检查与部署验证，并采用受控训练策略评估性能。

**📊 数据集**

实验采用CIFAR‑10、CIFAR‑100、MNIST、Fashion‑MNIST等常用数据集，且在实际STM32 MCU上验证。

**📈 对比分析**

与μNAS、ColabNAS等MCU‑专用NAS以及GENIUS的NAS‑Bench‑201比较，AutoMCU在保持相近或更高准确率的同时，将定制时间从数百GPU小时压缩到1–2小时，Token消耗和失败率也显著降低。

**⚠️ 局限性**

局限在于目前仅针对图像分类任务和STM32系列 MCU，需进一步扩展到其他 TinyML 场景、更多后端框架，并提升搜索空间多样性与理论性。

---

## 149. CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for Vision-Language-Action Models

**arXiv ID:** 2605.21854 | [PDF](https://arxiv.org/pdf/2605.21854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 150. The Ephemeral Web and the Case for Proactive Archiving

**arXiv ID:** 2605.21517 | [PDF](https://arxiv.org/pdf/2605.21517v1)

**作者:** Meliksah Yorulmazlar `[一作]` `[通讯]` (Rensselaer Polytechnic Institute), Meliksah Yorulmazlar (Rensselaer Polytechnic Institute)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并部署了一个基于 Python 与 GitHub Actions 的自动化网站归档系统，对巴基斯坦大使馆伊朗分校的网站进行持续归档。

**💡 创新点**

将主动归档作为网站维护的常规组成部分，采用轻量级、可持续的自动化流程并展示其在政治与平台不稳定环境下的可行性。

**🔧 技术方法**

Python 递归爬取、Wayback Machine 的 save API、GitHub Actions 定时工作流、随机化链接处理。

**📊 数据集**

该学校的内部链接和媒体资源（图片、PDF 等）作为归档目标。

**📈 对比分析**

与传统手动或一次性抓取对比，系统在 3h55m 的运行周期内多次归档，累计运行 580 次，耗时约 2,500 小时，证明低成本可持续归档可行。

**⚠️ 局限性**

仅针对单一域名的定制实现，未验证完整性与渲染质量，依赖外部平台（GitHub 与 Internet Archive）的可用性与政策限制，缺乏法律伦理讨论。

---

## 151. Who Uses AI? Platforms, Workforce, and AI Exposure

**arXiv ID:** 2605.21743 | [PDF](https://arxiv.org/pdf/2605.21743v1)

**作者:** Michelle Yin `[一作]` (Northwestern University), Burhan Ogut `[通讯]` (American Institutes for Research)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5079634511)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于 AI 平台会话日志构建的职业暴露度指标，并通过差分中的差分估计检验其是否真正反映劳动力市场对 AI 的影响。

**💡 创新点**

创新点在于将平台用户基础偏差建模为非经典测量误差，并通过对比不同平台、不同渠道及不同时间点的暴露度，提出重新加权至 BLS 劳动力分布以及部分识别区间的方法来量化并减缓这一偏差。

**🔧 技术方法**

主要技术包括差分中的差分回归、重新加权（reweighting）、非经典测量误差分析、部分识别（partial‑identification）以及多平台对照与事件研究。

**📊 数据集**

使用的数据集包括 Anthropic AEI 5 波的消费者与企业通道会话数据、OpenAI ChatGPT 与 Microsoft Copilot 的会话日志、BLS OEWS 劳动力分布、ACS 2015–2024 的人均年样本、Bick 等的工作场所 AI 使用调查及 NHIS 等跨域结果。

**📈 对比分析**

方法比较显示：不同平台或渠道的暴露度导致就业系数相差 1.9 倍、符号相反；在重新加权后，系数被削弱 42–93%，并在部分识别区间内落入接近零的范围，证明了平台用户基础对估计的显著影响。

**⚠️ 局限性**

局限性包括无法区分 AI 对工作产生的增补（augmentation）与替代（substitution）效应、对“保持顺序”假设的依赖、缺乏外生工具变量、以及对平台内部任务分布细节（如使用强度、任务比例）缺乏精确信息。

---

## 152. AdaPTwin: Adaptive Multi-Fidelity Predictive Digital Twin for Proactive Radio Resource Management in Vehicular Networks

**arXiv ID:** 2605.21897 | [PDF](https://arxiv.org/pdf/2605.21897v1)

**作者:** Armin Makvandi `[一作]` (University of British Columbia), Md. Jahangir Hossain `[通讯]` (University of British Columbia)

**通讯引用:** 11439 | [OpenAlex ID](https://openalex.org/A5039813871)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AdaPTwin，一个自适应多精度预测网络数字孪生框架，用于车联网中的前瞻性、时延感知的无线资源管理。

**💡 创新点**

创新点：动态自适应精度选择、结合 Transformer 与持续学习的轨迹预测、云-边协同架构、以及多起点迭代坐标下降启发式的 beamforming 与用户关联优化。

**🔧 技术方法**

技术：Transformer 轨迹预测、持续学习/迁移学习、NVIDIA Sionna 射线追踪、云-边层次计算、整数规划求解器与多起点迭代坐标下降（ICD）算法。

**📊 数据集**

数据集：SUMO 生成的多城市交通轨迹（温哥华、渥太华等）、Chongqing 真实轨迹、以及 3GPP UMi 随机通道模型。

**📈 对比分析**

对比方法：单精度/多精度反应式 NDT、PRISM DT、AIRTwin、Pegurri 等，评估时延、通道 RMSE、失效率与系统总速率。AdaPTwin 在保持 1 s 以内时延的同时，通道 RMSE 降低、总速率提升约 90%、失效率下降 80%。

**⚠️ 局限性**

局限：需要复杂的云-边资源配置，未在真实车载环境中验证；对极稀疏或极高密度场景的鲁棒性未知；3D 模型频繁更新带来额外开销；仅评估 V2I，未覆盖 V2V 场景。

---

## 153. Machine Learning as Performative Materialist Practice: Thirteen Theses on the Epistemology, Methodology, and Politics of Applied ML

**arXiv ID:** 2605.21785 | [PDF](https://arxiv.org/pdf/2605.21785v1)

**作者:** Adolfo De Unánue `[一作]` (Tecnológico de Monterrey), Fernanda Sobrino `[通讯]` (Tecnológico de Monterrey)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5092168099)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并阐述了“Performative Materialist ML”框架，提出13条理论论点（theses），解释并统一了现有的多项机器学习实践（如时间交叉验证、k@精确率/召回率、pipeline-aware 公平性审计、satisficing 而非 optimizing 等）以及它们的哲学与政治基础。通过对政府、公共卫生、刑事司法等领域的案例回顾，作者证明这些实践的背后是对模型与系统共演、情境化、政治性等的认识。

**💡 创新点**

创新点在于：① 将机器学习从传统的表示-验证-优化范式转化为对系统干预的表现主义（performative）视角；② 将多学科（网络科学、经济社会学、边界理性理论、材料主义）思想融合成一个统一的理论框架；③ 明确“可执行性”与“基线”是评估标准，将“模型性能”与“实际效果”区分；④ 对模型的“情境化”与“多目标”问题提供了一套哲学与方法论的解释，挑战了传统的普适性与客观性假设。

**🔧 技术方法**

核心技术方法包括：
- 统一的理论框架（13条论点）
- 传统机器学习技术（如分类器、特征工程、模型压缩）仅作为框架内的工具
- 评估与验证：时间序列交叉验证、precision@k、recall@k、pipeline-aware 公平性审计、持续监控与 retraining
- 理论工具：Pickering 的网络论、Callon 的经济社会学、Simon 的有限理性、Perdomo 等的 performative prediction 等。

**📊 数据集**

论文主要以案例回顾为数据来源，并未进行新的实验或使用公开数据集。引用的实例包括：
- 美国刑事司法中的预判模型（pretrial‑incarceration, recidivism）
- 墨西哥社会服务目标化系统
- 公开卫生领域的 HIV 维持预测
- 其他公共政策与资源配置案例。

**📈 对比分析**

比较方法主要是与传统机器学习实践（如无时间约束的交叉验证、单目标优化、全局性能指标）进行对比，并通过文献综述与案例讨论说明：
- 传统方法往往忽略时间性、政治性与系统共演；
- 采用框架中的方法可避免过度泛化、实现对基线的改善、兼顾多目标与政治需求；
- 论文未给出定量性能指标，但通过案例展示多目标评估与基线提升的实际成效。

**⚠️ 局限性**

局限性：
- 研究范围限定在机构决策支持场景，未涵盖科学研究或无干预的机器学习应用；
- 主要为理论与案例回顾，缺乏大规模实证验证与量化评估；
- 对“基线”与“可执行性”定义的细化需要进一步方法论细化；
- 框架的可操作性与工具化仍需后续工作（如自动化多目标搜索、动态治理机制）。

---

## 154. Polars inside Intel SGX2 Enclaves: An Empirical Study of Confidential Analytical Query Processing

**arXiv ID:** 2605.21797 | [PDF](https://arxiv.org/pdf/2605.21797v1)

**作者:** Wei Wang `[一作]` (Mozilla), Kenny Leftin `[通讯]` (Mozilla)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了将Arrow-native DataFrame引擎Polars运行在Intel SGX2 enclave中的性能，使用Gramine在Azure Blob Storage上执行TPC-H SF30基准。

**💡 创新点**

创新点在于首次将SGX2与Arrow-native引擎相结合，细粒度拆分查询与加载路径的开销，并比较Polars的lazy与eager执行模式在TEE中的差异。

**🔧 技术方法**

使用技术包括Intel SGX2硬件、Gramine库OS、Polars（Rust实现）、Apache Arrow、Azure DCsv3 VM、Azure Blob Storage。

**📊 数据集**

使用数据集为TPC-H SF30，并通过扩展列长度产生四个宽度配置（约22GB、30GB、41GB、73GB）。

**📈 对比分析**

通过对比非TEE与TEE的power score、加载时间和每查询延迟，发现整体开销≈1.5×，但加载路径占比随数据增大显著上升；lazy模式比eager快约2.3×且更能承受内存压力。

**⚠️ 局限性**

局限性包括仅在单一硬件/软件栈（Azure、Gramine、Blob）下测试，宽度扩展并非纯I/O负载，且缺乏对EPC分页和分配器的细粒度追踪。

---

## 155. When Support Escalates Distress: Regulation and Escalation in LLM Responses to Venting and Advice-Seeking

**arXiv ID:** 2605.21569 | [PDF](https://arxiv.org/pdf/2605.21569v1)

**作者:** Vivienne Bihe Chi `[一作]` (University of Pennsylvania), Sharath Chandra Guntuku `[通讯]` (University of Pennsylvania)

**通讯引用:** 4201 | [OpenAlex ID](https://openalex.org/A5010646067)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大语言模型（LLM）在心理健康支持中的安全性，聚焦于用户在情绪宣泄（venting）与求助寻求（advice‑seeking）两种不同求助风格下，LLM 的回复是否会调节还是放大情绪痛苦。

**💡 创新点**

创新点：①构建了六维理论驱动的评价框架，将“调节”（Regulation）与“升级”（Escalation）视为两条相互独立的维度；②发现这两条维度在LLM回复中并非互斥，而是同时上升，类似“共情+共情化”或共情化倾向；③通过三种角色（默认、朋友、治疗师）对比，证明角色提示可以在不降低用户体验的前提下显著减少升级风险。

**🔧 技术方法**

技术手段：①使用 GPT‑5.3 生成回复并做大规模自动注释；②采用差异性语言分析（Differential Language Analysis, DLA）与 LDA 主题模型评估用户文本的语言特征；③利用因子分析（Oblimin 旋转）验证六维框架的结构；④使用多元混合效应模型（multivariate mixed‑effects）同时估计调节与升级的效应；⑤开展专家与众包注释实验验证自动注释的可靠性。

**📊 数据集**

数据集：178,800 条 Reddit 帖子（14,040 名用户），包含 venting（r/vent、r/Venting）和 advice‑seeking（r/advice、r/needadvice）两个子版块，后续对 3,000 条样本（1,500 条 venting 与 1,500 条 advice‑seeking）进行 GPT‑5.3 回复生成。

**📈 对比分析**

比较方法：在 2（求助风格）×3（角色）设计下共 9,000 条回复，分别计算 Regulation 与 Escalation 分数；与专家与众包评价做 Kappa 比较；结果显示：①调节分数总体高于升级分数；②venting 触发更高的调节与升级；③朋友角色提升两者；治疗师角色显著降低升级，同时维持调节；用户体验（可取性与帮助性）在三种角色间无显著差异。

**⚠️ 局限性**

局限性：①仅分析单轮对话，未评估多轮互动的动态效应；②数据来源限于 Reddit，受众可能不具代表性；③LLM 既生成又注释，存在系统性偏差；④角色提示为实验设定，缺乏真实用户自发描述；⑤未直接测量实际心理健康改善，仅提供框架与指标。

---

## 156. Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments

**arXiv ID:** 2605.21935 | [PDF](https://arxiv.org/pdf/2605.21935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 157. Hierarchical Variational Policies for Reward-Guided Diffusion

**arXiv ID:** 2605.21661 | [PDF](https://arxiv.org/pdf/2605.21661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 158. Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention

**arXiv ID:** 2605.22072 | [PDF](https://arxiv.org/pdf/2605.22072v1)

**作者:** Changyuan Tian `[一作]` (AMAP, Alibaba Group), Deheng Ye `[通讯]` (Nanyang Technological University)

**通讯引用:** 1403 | [OpenAlex ID](https://openalex.org/A5073681676)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Faithful-MR1 框架，通过“定位-推理”两阶段增强多模态大语言模型的视觉感知与推理可信度。

**💡 创新点**

创新点在于① Anchoring 阶段直接用 region‑grounded attention 监督专用 token；② Reinforcing 阶段采用 counterfactual 图像掩蔽定位 vision‑dependent token 并对其注意力使用进行奖励，解决 perception–reasoning disconnect。

**🔧 技术方法**

使用了 RLVR（PPO/GRPO 变体）、视觉注意力监督、token‑级 KL 归一化、split GRPO advantage、counterfactual masking 等技术。

**📊 数据集**

构建 19.2K 带 bounding‑box 标注的 Vision‑SR1‑47K 数据集（6K SFT + 13.2K RL），并在 Qwen2.5‑VL‑Instruct 3B/7B backbone 上训练；评测基准包括 MathVision、MathVerse、MathVista、WeMath、DynaMath、MMMU‑Pro、HallusionBench。

**📈 对比分析**

在与 GRPO、VPPO、Vision‑R1、Perception‑R1、Vision‑SR1 等基线同一 backbone 与 19.2K 数据下比较，Faithful‑MR1 在 3B 与 7B 上整体得分分别为 43.9 与 51.3，均超越所有基线；在公开 checkpoint 对比中以更少数据取得更优整体及多数单项指标。

**⚠️ 局限性**

局限性：依赖 region‑level bounding‑box 注释，需额外的 counterfactual 掩蔽计算导致训练成本略升；推理无额外开销但对注释工具 Gemini‑3‑Flash 的准确性有依赖。

---

## 159. Can Breath Biomarkers Causally Influence Blood Glucose? Investigating VOC-Mediated Modulation in Diabetes

**arXiv ID:** 2605.22075 | [PDF](https://arxiv.org/pdf/2605.22075v1)

**作者:** Varsha Sharma `[一作]` (TCS Research), Avik Ghose `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了呼吸气体中挥发性有机化合物对血糖的因果影响，并构建了基于VOC的非侵入性糖尿病早期筛查与风险分层系统。

**💡 创新点**

首次结合因果推断与机器学习验证VOC对血糖的因果效应，设计合成葡萄糖指标及灰色区风险排序，并通过GMM聚类揭示潜在亚组。

**🔧 技术方法**

采用DoWhy因果推断、SHAP解释、XGBoost/随机森林/逻辑回归分类、GMM+PCA聚类及Mann–Whitney U检验等技术。

**📊 数据集**

使用94名受试者的数据，包括GC-MS测得的VOC谱、生活方式、血糖等临床指标，涵盖健康、轻度糖尿病、良好控制与不良控制四组。

**📈 对比分析**

分类模型以AUC、准确率和F1评估，XGBoost得到AUC 0.938、准确率 94%（去除异常值后 98%），GMM聚类与真值比对得到ARI 0.58、NMI 0.48，聚类预测糖尿病的F1 0.88。

**⚠️ 局限性**

样本量小且缺乏多中心、多族群验证；潜在未观测混杂（如胰岛素缺乏）可能影响因果解释；VOC与血糖共因机制尚未完全阐明。

---

## 160. Enhancing Visual Token Representations for Video Large Language Models via Training-Free Spatial-Temporal Pooling and Gridding

**arXiv ID:** 2605.22078 | [PDF](https://arxiv.org/pdf/2605.22078v1)

**作者:** Bingjun Luo `[一作]` (Tsinghua University), Xinpeng Ding `[通讯]` (Xidian University)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5020996610)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的视觉令牌增强方法 ST-GridPool，旨在提升视频大语言模型的视觉表征效果。

**💡 创新点**

创新点在于将层级时间格子化（Pyramid Temporal Gridding）与基于令牌范数的空间池化（Norm-based Spatial Pooling）结合，既捕获多尺度时空交互，又自适应保留高信息区域，同时不需额外训练参数。

**🔧 技术方法**

主要技术包括层级时间格子化（PTG）、基于 L2 范数的动态权重池化（NSP）、Softmax 加权和插值重构等，直接在现有 Video LLM 的视觉编码器输出上操作。

**📊 数据集**

在长视频理解领域使用 VideoMME、LongVideoBench、EgoSchema 等基准，在通用视频理解领域使用 NexT-QA、TempCompass、MVBench 等数据集进行评估。

**📈 对比分析**

与多种基线模型（LLaVA-OneVision、LLaVA-Video、NVILA、mPLUG‑Owl3 等）以及多种 Token 压缩方法（FastV、PruMerge、VisionZip、FrameFusion 等）进行对比，ST-GridPool 在所有任务中均实现了显著提升，尤其在 30% 令牌预算下表现最优或接近最优。

**⚠️ 局限性**

局限性包括：仅在训练无关设置下验证，未测试更大模型或不同视觉编码器的泛化；对温度 β 和范数 p 的取值敏感；对极端时间尺度或非常长序列的细节捕捉仍需进一步研究。

---

## 161. Active Evidence-Seeking and Diagnostic Reasoning in Large Language Models for Clinical Decision Support

**arXiv ID:** 2605.22047 | [PDF](https://arxiv.org/pdf/2605.22047v1)

**作者:** Chen Zhan `[一作]` (Shanghai University of Engineering Science), Lu Gan `[通讯]` (Fudan University)

**通讯引用:** 8810 | [OpenAlex ID](https://openalex.org/A5012208123)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了ROUNDS-Bench交互式评测框架，对大型语言模型在主动证据寻求与诊断推理中的表现进行系统评估。

**💡 创新点**

创新点在于引入标准化病人模拟和信息门控机制，将传统静态评测转化为可控多轮交互，量化主动诊断与被动诊断的性能差距。

**🔧 技术方法**

利用大型语言模型驱动的标准化病人模拟、信息门控、Chain-of-Thought（CoT）推理、基于LLM的评判器以及严格的评估指标（ExactAcc、StrictEQ、FSA）实现评测。

**📊 数据集**

使用四大来源（MedQA、MedMCQA、MedFound、MedCaseReasoning）构成的468例、六系统平衡病例集。

**📈 对比分析**

通过15种LLM在Task1（全上下文）与Task2（主动寻求）对照，发现主动寻求时平均诊断准确率下降12.75%，证据质量下降24.36%，仅少数模型如DeepSeek-v3-250324保持较高证据质量。

**⚠️ 局限性**

局限性包括仅文本交互缺乏多模态信息、未考虑社会心理因素、单一参考诊断、评估器基于LLM且未纳入临床专家审核。

---

## 162. Noise Schedule Design for Diffusion Models: An Optimal Control Perspective

**arXiv ID:** 2605.21911 | [PDF](https://arxiv.org/pdf/2605.21911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 163. Quantitative coronary calcification analysis for prediction of myocardial ischemia using non-contrast CT calcium scoring

**arXiv ID:** 2605.21745 | [PDF](https://arxiv.org/pdf/2605.21745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 164. Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding

**arXiv ID:** 2605.21973 | [PDF](https://arxiv.org/pdf/2605.21973v1)

**作者:** Zelin Zheng `[一作]` (University of Chinese Academy of Sciences), Laiyun Qing `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 1547 | [OpenAlex ID](https://openalex.org/A5010557705)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可验证的视频时序定位框架F2G，采用先识别再测量（Identify‑then‑Measure）的流程，将视频中的候选事件段作为可引用的证据，并让LLM先选择一个证据段后再细化时间边界；

**💡 创新点**

1) 将时序感知与推理分离，构建显式证据池；2) 通过多视角潜在预测与几何正则化训练时序模块，实现对事件边界的敏感表示；3) 在LLM推理阶段强制进行证据引用，提升定位稳定性与可解释性；

**🔧 技术方法**

多视角潜在预测训练、Sliced Isotropic Gaussian Regularizer、Span Evidence Encoder (Q-Former)，以及基于LoRA的LLM微调；

**📊 数据集**

Charades‑STA、ActivityNet‑Captions、QVHighlights、TimeLens‑Bench、VideoMME；

**📈 对比分析**

在多种Video‑LLM骨干（Qwen3‑VL‑8B、LLaVA‑NEXT‑7B、Qwen2.5‑VL‑7B）上与现有VTG方法对比，+F2G‑FT在R@0.5/0.7、mIoU、mAP等指标上均显著提升，且在不同基线上保持一致性；

**⚠️ 局限性**

仅针对单一事件定位，未覆盖多事件或复杂时序约束；对多模态（音频、语音）或空间‑时序定位的扩展尚未实现。

---

## 165. Comparing LLM and Fine-Tuned Model Performance on NVDRS Circumstance Extraction with Varying Prompt Complexity

**arXiv ID:** 2605.21845 | [PDF](https://arxiv.org/pdf/2605.21845v1)

**作者:** Geoffrey Martin `[一作]` (Weill Cornell Medicine), Yifan Peng `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 11083 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“复杂度评分”算法和混合提示框架，用以在自杀死亡调查文本中抽取需要深度语义推理的情境。

**💡 创新点**

创新点在于利用编码手册结构预测何时需要详细提示，自动为每个情境选择最优提示策略，并展示LLM在罕见、推理复杂任务上优于传统微调模型。

**🔧 技术方法**

采用大型语言模型（GPT‑5.2、Gemini 2.5 Pro、Llama‑3 70B）进行零样本推理，配合复杂度评分算法生成提示；对比了细化提示、名称提示和传统 RoBERTa。

**📊 数据集**

使用美国国家暴力死亡报告系统（NVDRS）191,696条死亡调查报告，按情境抽取25个推理复杂的情境进行评估。

**📈 对比分析**

通过宏观F1对比，混合提示得到0.893的F1，接近0.897的oracle上限，显著高于单一提示的GPT‑5.2（0.883）和微调RoBERTa（0.800）。

**⚠️ 局限性**

局限在于评估样本仅200条/情境，数据仅限NVDRS，且对不同编码体系的泛化性尚未验证。

---

## 166. Learning Altruistic Collaboration in Heterogeneous Multi-Team Systems

**arXiv ID:** 2605.21723 | [PDF](https://arxiv.org/pdf/2605.21723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 167. ECPO: Evidence-Coupled Policy Optimization for Evidence-Certified Candidate Ranking

**arXiv ID:** 2605.21993 | [PDF](https://arxiv.org/pdf/2605.21993v1)

**作者:** Miaobo Hu `[一作]` (Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 297584 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“证据认证候选排序”方法，要求模型在给出Top‑K候选列表的同时生成可追溯、可验证的跨度证据。

**💡 创新点**

创新点在于将证据可验证性纳入训练目标，构建了证据循环奖励（evidence‑cycle）和标签无关的确定性验证器，直接优化决策–证据耦合度。

**🔧 技术方法**

核心技术包括：基于规划骨架的轨迹奖励学习、动态规划对齐、生成式策略优化（ECPO）、确定性证据验证器、以及JSON约束的结构化输出。

**📊 数据集**

使用了MAVEN‑ERE和RAMS两大数据集，冻结事件抽取器产生轨迹，构造闭合、预测和混合候选池三种评估情景。

**📈 对比分析**

在三种候选池设置下与多种基线（零射程、SFT、GRPO、DPO、RM‑only+对齐、LambdaMART、GraphRank等）对比，ECPO在“@10”与“@10”（即证据可恢复的NDCG）指标上领先，虽在普通NDCG上略逊于RM‑only+对齐，但显著提升了决策与证据的一致性。

**⚠️ 局限性**

局限性包括：对候选池召回的依赖（闭合池下效果受限）、需要手工构造骨架与验证规则、对极端稀疏或噪声文本的鲁棒性尚未充分验证，且验证器仍为确定性规则，可能无法捕捉复杂语义关系。

---

## 168. Finding Missing Input Validation in TEEs via LLM-Assisted Symbolic Execution

**arXiv ID:** 2605.22058 | [PDF](https://arxiv.org/pdf/2605.22058v1)

**作者:** Chengyan Ma `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31431 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于大型语言模型的符号执行框架（SymTEE），用于在不需要真实TEE硬件环境的前提下检测TEE应用中的缺失输入校验漏洞。

**💡 创新点**

创新点在于利用LLM（GPT‑5）自动生成可直接用于KLEE的 mock 环境和 harness，消除了对完整TEE构建与专用硬件的依赖，实现了低成本、高精度的漏洞检测。

**🔧 技术方法**

采用的技术包括：AST分析提取可能缺陷的代码片段；LLM自动生成 mock 环境与 KLEE 兼容的 harness；使用 KLEE 进行符号执行和 SMT 求解器检验输入校验路径；结合 token 计费实现成本估算。

**📊 数据集**

使用了一个包含 26 个漏洞的基准集，涵盖 11 个真实世界 GitHub 项目（如 optee‑sdp、basicAlg_use）和 15 个人工合成的安全缺陷。

**📈 对比分析**

与传统静态/动态TEE分析工具对比，SymTEE 在 26 个漏洞中实现 100% 的精度和 92.3% 的召回率；平均每个漏洞使用 5,931 个 token，成本约 0.05 美元；相较于需要完整TEE环境的传统方法，分析时间和设置成本显著降低。

**⚠️ 局限性**

主要限制包括：AST 预处理阶段的失误导致两例漏检；LLM 生成的 mock 环境在处理复杂或不常见的TEE API 时可能不够完善；目前仅聚焦缺失输入校验漏洞，未覆盖其他类型的TEE安全缺陷。

---

## 169. Reed-Muller Codes for Joint Random and Stuck-At Error Correction

**arXiv ID:** 2605.21727 | [PDF](https://arxiv.org/pdf/2605.21727v1)

**作者:** Ivana Djurdjevic `[一作]` (WD Research), Cyril Guyot `[通讯]` (WD Research)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种递归构造的掩码集合，用于在长度为2^m的二进制序列中纠正任意s个停留错误，并证明该集合是RM(s-1,m)码的子码。

**💡 创新点**

构造了一个非线性、递归生成的掩码集合，其规模不超过2^s m^{s-1}，并且可与任意RM(r,m)(r≥s-1)码联合用于随机误差校正；首次给出s=2时的标签生成算法以及s>2时的标签大小界限。

**🔧 技术方法**

利用Reed-Muller码的Plotkin递归构造、线性码的掩码搜索、贪心搜索与信息论冗余界限分析。

**📊 数据集**

以n=2^m，m≤12、s≤4为例，列举了多组具体掩码集合和标签位，计算对应冗余大小；没有使用真实存储器数据。

**📈 对比分析**

通过与理论下限和上限比较，得到的停留错误冗余接近下限；与传统使用BCH码的mask冗余（比例为s log_2 n）相比，递归掩码在相同s下实现更低冗余；编码/解码仅需一次解码，复杂度低。

**⚠️ 局限性**

对s>2的标签位选择缺乏最优算法，标签位大小仍非最优；编码/解码实现细节未给出；无法证明对任意s的最佳冗余；对实际存储器实现的实验验证缺失。

---

## 170. Amplifying, Not Learning: Fine-Tuned AI Text Detectors Amplify a Pretrained Direction

**arXiv ID:** 2605.21653 | [PDF](https://arxiv.org/pdf/2605.21653v1)

**作者:** Alexander Smirnov `[一作]` (University College London), Alexander Smirnov `[通讯]` (University College London)

**通讯引用:** 122915 | [OpenAlex ID](https://openalex.org/A5032186284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并证明 AI 文本检测器仅放大预训练的典型性轴，而不是构建 AI 与人类的边界，并提出一种闭式一阶预测器用于轴介入，显著提升检测性能。

**💡 创新点**

创新点在于：①揭示预训练典型性轴为检测器核心，细调仅放大该轴；②提出可精确预测轴介入效果的闭式一阶算子；③用 24‑shot 冻结探针匹配全微调，验证轴已足够；④证明不同调参方法在匹配 TP 约束下仅是校准差异。

**🔧 技术方法**

使用预训练编码器（ELECTRA、RoBERTa、DeBERTa）投影、线性头微调、LoRA、CC‑loss、dealign‑f2c、闭式一阶 Jacobian 预测器、签名 ε‑rank‑1 轴消融、caps_rate 残差、对数似然等技术。

**📊 数据集**

实验数据集包括 NYT、HC3（ChatGPT vs 人类）、Ghostbuster、EvoBench、FCE（非母语 ESL）、RAID、OpenAI、Claude、Gemini 等多来源文本集。

**📈 对比分析**

采用匹配 TPR=0.90 的阈值对比 AUROC：原始投影已达到 86–106% 的微调上限；24‑shot 冻结探针 AUROC 0.900 对比全微调 0.895；闭式预测器在 1% FPR 下把 NYT‑FPR 从 0.000 提升到 0.904，且在三款第三方检测器上实现 57% FPR 降低。

**⚠️ 局限性**

局限性包括：轴在不同体系结构下每文本机制不一致，典型性轴以 HC3 为锚定；对长度、表面特征的依赖不同；闭式预测器仅在 |ε|≤0.7 方向对称有效；对多任务推广需进一步验证。

---

## 171. Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents

**arXiv ID:** 2605.21768 | [PDF](https://arxiv.org/pdf/2605.21768v1)

**作者:** Sikuan Yan `[一作]` (Ludwig Maximilian University of Munich), Yunpu Ma `[通讯]` (Ludwig Maximilian University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Memory-R2 训练框架，利用 LoGo-GRPO 算法实现多轮会话中的公平信用分配，并通过共享提取-管理器结构联合优化记忆生成与演化。

**💡 创新点**

创新点在于将全局轨迹奖励与局部重采样相结合的 LoGo-GRPO，使用共享参数的多角色学习、分段多步记忆构建，以及逐步延长会话长度的课程学习策略，显著提升长周期记忆 LLM 的训练稳定性与效果。

**🔧 技术方法**

采用强化学习（GRPO/LoGo-GRPO）、共享参数多代理 RL、长度归一化的 step‑level RL、局部重采样、压缩惩罚、分段记忆构建、多步决策、LLM 角色提示等技术。

**📊 数据集**

主要使用 LoCoMo 进行训练，LongMemEval、MSC‑Self‑Instruct、MemBench 用于 OOD 评估。

**📈 对比分析**

在 LoCoMo 上与 A‑MEM、Mem0、MemoryOS、MEM1、MemAgent、Memory‑R1 等基线对比，Memory‑R2 在 token‑level F1、BLEU‑1 以及 LLM‑as‑a‑Judge 指标上均遥遥领先，尤其在小模型上提升显著，并在 OOD、不同规模模型和答案代理上保持强大的泛化能力。

**⚠️ 局限性**

局限性包括：仍受记忆更新错误累积的影响，需要多步交互才能有效学习；训练数据量极少（仅两条 LoCoMo 对话），对更长、更复杂任务的适用性尚未充分验证；以及对记忆内容的可解释性和对长周期推理的鲁棒性仍有提升空间。

---

## 172. From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment

**arXiv ID:** 2605.21558 | [PDF](https://arxiv.org/pdf/2605.21558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. Position: The Time for Sampling Is Now! Charting a New Course for Bayesian Deep Learning

**arXiv ID:** 2605.21765 | [PDF](https://arxiv.org/pdf/2605.21765v1)

**作者:** Emanuel Sommer `[一作]` (LMU Munich), David Rügamer `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文阐述了采样推断（SAI）在贝叶斯深度学习中的优势，纠正了常见误解，并提出了提升采样可行性和实用性的关键研究方向。

**💡 创新点**

创新点在于：①主张采样方法已与优化方法并列可行；②将探索与后期样本蒸馏视为核心任务；③提供多模型多数据集的实证对比与工具生态建议。

**🔧 技术方法**

使用的技术包括：马尔可夫链蒙特卡罗（HMC、SG-MCMC、BDE）、并行链与优化加速、深度集成（DE）、后期样本压缩/蒸馏、贝叶斯模型平均等。

**📊 数据集**

实验数据集涵盖：UCI回归、CIFAR‑10、Imagenette、ResNet‑18、Vision Transformer、Wide CNN、Llama 3 微调等，覆盖从小型到中型规模的多种网络。

**📈 对比分析**

在LPPD、准确率、校准（ACE）、OOD AUROC等指标上，与MAP、变分推断、Laplace、SWA、深度集成等基线相比，SAI（尤其是Bayesian Deep Ensembles）表现出更高或相近的性能，且采样成本已趋近于优化方法。

**⚠️ 局限性**

局限性包括：推断时需大量前向传播导致高计算/存储成本；SG‑MCMC 对超参数敏感；极大模型的全参数采样仍不可行，需要子网络或低秩策略；缺乏成熟的端到端工具与诊断框架。

---

## 174. ConvNeXt-FD: A Fractal-Based Deep Model for Robust Biomedical Image Segmentation

**arXiv ID:** 2605.22002 | [PDF](https://arxiv.org/pdf/2605.22002v1)

**作者:** Joao Batista Florindo `[一作]` (University of Campinas), Amanda Pontes de Oliveira Ornelas `[通讯]` (University of Campinas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 ConvNeXt-FD，结合 ConvNeXt 编码器和 U‑Net 解码器，并使用分形维数正则化的混合损失进行生物医学图像分割。

**💡 创新点**

创新点包括：将 ConvNeXt 作为强大编码器集成到 U‑Net 结构；引入基于分形维数的边界正则化，提升边界精度；在多模态数据上展示通用性能。

**🔧 技术方法**

使用技术：ConvNeXt backbone、U‑Net encoder‑decoder、Dice+交叉熵混合损失、分形维数辅助分支、ImageNet 预训练、Adam 优化器、数据增强等。

**📊 数据集**

使用数据集：BUSI、DDTI、FluoCells、IDRiD、ISIC2018、MoNuSeg 六个公开医学影像分割数据集。

**📈 对比分析**

与多种 SOTA 方法（如 UltraSAM、DCCE‑UNet、TransUNet 等）在 Dice、Jaccard 等指标上对比，ConvNeXt‑FD 在所有数据集均达到或超过现有最佳性能，尤其在 IDRiD（96.19% Dice）和 FluoCells（85.10% Dice）上实现最高分。

**⚠️ 局限性**

局限性：分形维数正则化计算开销大，λ_FD 需要手动调优；模型规模较大，推理速度和内存占用高；缺乏对实例分割、跨模态融合和不确定性量化的进一步研究。

---

## 175. Resource bounded Kučera-Gács Theorems

**arXiv ID:** 2605.21546 | [PDF](https://arxiv.org/pdf/2605.21546v1)

**作者:** Satyadev Nandakumar `[一作]` (Indian Institute of Technology Kanpur), Chandra Shekhar Tiwari `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文研究了 Kučera–Gács 定理在资源受限（多项式时间与有限状态）情形下的对应结果，提出并证明了 quasi‑polynomial 时间版本的 Kučera–Gács 定理，并对多项式时间维度与 Kolmogorov 复杂度率之间的关系进行了完整的刻画，随后展示了在有限状态归约下 Kučera–Gács 定理不成立。

**💡 创新点**

创新点在于：①首次给出 quasi‑polynomial 复杂度下的 Kučera–Gács 定理，并证明其冗余为 n+o(n)；②证明多项式时间维度 ρ^-_poly 与 Kolmogorov 复杂度率 𝒦_poly 完全等价；③在存在一把钥匙函数的假设下，指出 ρ^-_poly 与维度不等价；④揭示有限状态归约无法从任何正常数列中提取所有序列。

**🔧 技术方法**

主要技术包括：构造对抗多项式时间马丁格尔的通用 savings‑account martingale；利用二分搜索编码方案以实现低冗余的对角化；采用层次化哈希/混合编码以实现多项式时间解压缩比的等价证明；使用有限状态机分析及 Schnorr–Stimm 引理来证明有限状态归约的频率不收敛性限制。

**📊 数据集**

本研究为纯理论计算复杂性与算法随机性问题，未使用任何实验数据集，全部结果均为严格证明。

**📈 对比分析**

通过理论证明表明，任何无限序列都可在 quasi‑polynomial 时间内从多项式时间随机序列中恢复，且所需 oracle 位数为 n+o(n)；oracle 使用率与序列的多项式维度上界相符；与现有的多项式空间/可计算维度结果比较，扩展了对多项式时间的完整刻画；同时证明了有限状态归约的失败，表明归约能力与随机性之间存在根本限制。

**⚠️ 局限性**

局限性包括：仅得到 quasi‑polynomial 时间的结果，尚未证明是否可提升到多项式时间；对强维度的等价关系尚未解决；在一把钥匙函数存在的假设下，结论对 ρ^-_poly 与维度的等价性失效；对有限状态归约的否定结果仅限于正常数列，未探讨更强随机性假设下的情况。

---

## 176. ORBIS: Output-Guided Token Reduction with Distribution-Aware Matching for Video Diffusion Acceleration

**arXiv ID:** 2605.22015 | [PDF](https://arxiv.org/pdf/2605.22015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 177. Echo4DIR: 4D Implicit Heart Reconstruction from 2D Echocardiography Videos

**arXiv ID:** 2605.22066 | [PDF](https://arxiv.org/pdf/2605.22066v1)

**作者:** Yanan Liu `[一作]` (National University of Singapore), Lei Li `[通讯]` (National University of Singapore)

**通讯引用:** 12302 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种名为 Echo4DIR 的 4D 隐式心脏重建框架，利用稀疏 2D 超声视频通过条件 Signed Distance Function (SDF) 和自监督可微渲染，实现对患者个体化心脏 3D+t 几何的精确重建。

**💡 创新点**

创新点包括：① 将条件 SDF 与多视角 Epipolar Cross Attention 相结合，构建鲁棒的 3D 形状先验；② 采用专门为 SDF 设计的可微渲染策略，在测试时通过自监督方式对形状进行适配；③ 引入径向 SDF 对齐（RSA）策略，将速度场与形状演化严格锁定，消除网格漂移与体积崩塌；④ 通过 Test‑Time Optimization (TTO) 进行患者特异性重建，提升稀疏观察下的几何完整性。

**🔧 技术方法**

使用技术包括：条件 SDF（MLP 结构）、SSM 预训练、Epipolar Mask Encoder + Epipolar Cross Attention、SDF 量化的可微渲染、神经速度场（MLP + 位置编码）、双向 LSTM 进行时间序列平滑、Radial SDF Alignment、BCE + Dice 损失、Transport + Smoothness 损失。

**📊 数据集**

采用的实验数据集有：1）基于 18 模态 SSM 的 10,000 张左心室（LV）网格（9,000 训练 / 1,000 测试）；2）来自公开临床 Control Cohort 的 144 例经 TTE 分割得到的 2D 视图，用于验证 3D 投影与临床分割的一致性。

**📈 对比分析**

与现有方法（E‑PiVox、MIA'25 等 GCN‑基准）比较，Echo4DIR 在 SSM 数据集上 MAE 仅 1.15 mm，Dice 达 98.35%，IoU 96.75%，显著优于对手；在 Control Cohort 上也取得 98.35% Dice，且在未见视角的重建中 Dice 仍高达 95.24%。Ablation 实验证明 Epipolar Cross Attention 与可微渲染对性能至关重要。

**⚠️ 局限性**

局限性包括：① 目前仅针对左心室结构，未覆盖完整心脏；② 依赖高质量的超声分割和少量视角；③ 对极端噪声、运动模糊或极端解剖变异的鲁棒性尚未充分验证；④ 需要进一步评估在临床实际工作流程中的可用性与计算效率。

---

## 178. BodyReLux: Temporally Consistent Full-Body Video Relighting

**arXiv ID:** 2605.21766 | [PDF](https://arxiv.org/pdf/2605.21766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. Hypergraph as Language

**arXiv ID:** 2605.21858 | [PDF](https://arxiv.org/pdf/2605.21858v1)

**作者:** Mengqi Lei `[一作]` (Tsinghua University), Yue Gao `[通讯]` (Tsinghua University)

**通讯引用:** 19697 | [OpenAlex ID](https://openalex.org/A5100602494)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出Hyper-Align框架，将超图直接视为语言输入，实现LLM对高阶关联的原生建模。

**💡 创新点**

创新点在于：①引入Hypergraph Incidence Detail Template with Overview (HIDT‑O)以固定形状序列化高阶关联；②设计Hypergraph Incidence Projector (HIP)，通过语义-结构解耦与双向消息传递实现高阶结构对齐；③构建统一的Hypergraph-as-Language协议和HyperAlign-Bench基准。

**🔧 技术方法**

采用超图序列化、投影器、双向注意力消息传递，并结合LLM微调和辅助重构任务，基于Qwen3等LLM进行实验。

**📊 数据集**

使用Arxiv-HG（共169k论文、123k超边）以及Cora‑CC、PubMed、DBLP、IMDB等四个未见域的超图数据集。

**📈 对比分析**

与传统HGNN、HyperBERT、通用LLM、图-LLM等方法对比，Hyper-Align在内部任务和零样本跨域任务上均取得显著提升，尤其在高阶超边上优势明显。

**⚠️ 局限性**

局限性包括：①对采样预算与模板设计敏感；②依赖LLM的参数冻结，需进一步探索更高效的对齐策略；③在极大规模超图或动态图场景下的可扩展性待验证。

---

## 180. LLM Retrieval for Stable and Predictable Ad Recommendations

**arXiv ID:** 2605.21969 | [PDF](https://arxiv.org/pdf/2605.21969v1)

**作者:** Vinodh Kumar Sunkara `[一作]` (Meta Platforms, Inc.), Deepak Chandra `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大语言模型（LLM）的广告候选生成框架，通过从广告创意中提取层级语义属性，构建图结构进行语义匹配，以提升广告推荐系统的稳定性与可预测性。

**💡 创新点**

创新点包括：①提出“StatSigDiff”这一新的可预测性评估指标；②利用微调后的 LLM 自动生成层级语义标签并构建语义图；③在候选生成阶段加入图遍历与 Jaccard 相似度匹配，实现语义层级扩展；④在工业级线上 A/B 测试中验证显著提升。

**🔧 技术方法**

核心技术：LLM（Llama3‑8B Instruct）微调、文本语义向量生成、层级属性抽取、语义图构建、图遍历算法、基于 Jaccard 的相似度匹配、GPU 并行推理。

**📊 数据集**

使用了数千万条广告创意的工业级文本数据集，仅利用标题与描述信息进行特征抽取与模型微调；实验基于真实的 Meta 广告投放系统数据。

**📈 对比分析**

与传统两塔、嵌入式与基于图的候选生成器对比，实验评估指标包括 Recall@K、线上 topline 变动、StatSigDiff 与 MAD。实验结果显示：topline 线上指标提升 0.45%，Recall@K 上提升 1.2%，StatSigDiff 降低 8.62%，MAD 降低 45%。

**⚠️ 局限性**

限制与不足：①仅针对文本特征，未融合图像/视频等多模态信息；②LLM 微调与推理成本高，需大量 GPU 资源；③评估主要集中在候选生成阶段，对后续重排序与整体推荐流程的影响尚未完全验证；④StatSigDiff 受随机性影响，零值不可达，仍需进一步优化评估方法。

---

## 181. Detecting Synthetic Political Narratives in Cross-Platform Social Media Discourse

**arXiv ID:** 2605.21540 | [PDF](https://arxiv.org/pdf/2605.21540v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 182. CCLab: Adversarial Testing of Learning- and Non-Learning-Based Congestion Controllers

**arXiv ID:** 2605.21915 | [PDF](https://arxiv.org/pdf/2605.21915v1)

**作者:** Zhi Chen `[一作]` (University of Illinois Urbana Champaign), Gang Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一个对抗性测试框架（CCLab），用于系统评估学习式和传统拥塞控制器在受扰动条件下的鲁棒性。

**💡 创新点**

创新点在于使用强化学习驱动的对抗性代理在特征层和环境层生成可控、逼真的扰动，能够系统比较不同CC的脆弱性，并能将这些对抗性轨迹用于对抗性训练以提升鲁棒性。

**🔧 技术方法**

主要技术包括强化学习对抗性代理、最小RTT特征扰动、带宽/延迟环境扰动、Mahimahi网络仿真、基于约束的奖励设计以及对抗性训练等。

**📊 数据集**

实验使用Canopy公开数据集中的18条模拟轨迹和3条真实LTE网络轨迹，作为带宽和延迟的基准。

**📈 对比分析**

通过在特征层对Min-RTT添加±5%/±50%噪声以及在环境层生成对抗性带宽轨迹，测量吞吐率利用率与排队延迟；结果显示学习型CC在两类攻击下比传统CC更稳健，攻击对学习型CC的利用率降幅约为12%，而传统CC降幅可达19–30%。

**⚠️ 局限性**

局限性包括仅评估带宽与Min-RTT两种扰动，未覆盖多路复用或公平性等场景；对抗性训练仅在少数模型上验证，未能全面覆盖所有学习式CC。

---

## 183. ASSEMBLAGE-DEEPHISTORY: A Cross-Build Binary Dataset with Temporal Coverage

**arXiv ID:** 2605.21615 | [PDF](https://arxiv.org/pdf/2605.21615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 184. Finding a Solution to the Erdős-Ginzburg-Ziv Theorem in Linear Time

**arXiv ID:** 2605.21753 | [PDF](https://arxiv.org/pdf/2605.21753v1)

**作者:** Sunghyeon Jo `[一作]` `[通讯]`, Sunghyeon Jo

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种确定性的线性时间算法，用于解决Erdős–Ginzburg–Ziv定理中的子序列和问题。

**💡 创新点**

创新点在于提供了一种比之前的O(n log log log n)算法更高效的O(n)算法，且不需要预处理步骤。

**🔧 技术方法**

使用了线性时间算法和算术进展的紧凑表示法来维护可达和的集合。

**📊 数据集**

使用了包含2n-1个整数的序列作为数据集，特别是针对模p的非零剩余。

**📈 对比分析**

与之前的算法相比，新的算法在时间复杂度上显著降低，能够在O(n)时间内完成，而之前的算法需要O(n log log log n)时间。

**⚠️ 局限性**

算法的局限性在于它主要针对素数模的情况，对于复合模的情况需要通过标准的乘法约简来扩展。

---

## 185. FRED: A Multi-Modal Autonomous Driving Dataset for Flooded Road Environments

**arXiv ID:** 2605.22018 | [PDF](https://arxiv.org/pdf/2605.22018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild

**arXiv ID:** 2605.22064 | [PDF](https://arxiv.org/pdf/2605.22064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 187. Toward Understanding Adversarial Distillation: Why Robust Teachers Fail

**arXiv ID:** 2605.21999 | [PDF](https://arxiv.org/pdf/2605.21999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 188. Representation Gap: Explaining the Unreasonable Effectiveness of Neural Networks from a Geometric Perspective

**arXiv ID:** 2605.21692 | [PDF](https://arxiv.org/pdf/2605.21692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 189. ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2605.22020 | [PDF](https://arxiv.org/pdf/2605.22020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 190. Planning, Scheduling, and Behavior in EV Charging Systems: A Critical Survey and Trilemma Framework

**arXiv ID:** 2605.21665 | [PDF](https://arxiv.org/pdf/2605.21665v1)

**作者:** Peiyan Xiao `[一作]` (William & Mary), Yanhai Xiong `[通讯]` (William & Mary)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5008946067)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述电动汽车充电网络研究，提出三层（规划、调度、行为）PSB框架，阐明三层耦合的“PSB三难”，并系统分析现有双层耦合研究的简化方法与盲点，指出未来研究方向。

**💡 创新点**

1) 引入统一的三层PSB框架，明确各层决策时间尺度、决策者和耦合结构；2) 提出PSB三难，即在保持三层高保真度时难以实现可计算性；3) 系统评估双层耦合研究的简化策略，揭示不同简化对决策目标的损失；4) 提出量化简化成本和未来研究路线。

**🔧 技术方法**

结构化文献综述、决策层分类、耦合边分析、复杂度与可计算性论证、简化成本评价、未来工作建议。

**📊 数据集**

无专门数据集；依赖已发表文献与公开的充电网络案例、调度模型与行为模型数据，未进行统一实验评估。

**📈 对比分析**

文章主要通过对已有文献的分类、简化策略对比以及理论复杂度分析来说明其贡献，未提供实验或性能指标；评价依托文献综述与理论讨论。

**⚠️ 局限性**

缺乏统一基准数据和实验验证，未量化不同简化对决策质量的具体损失；三层完整耦合模型极少，实用性有限；对行为层实证校准不足，数据来源不统一。

---

## 191. EntmaxKV: Support-Aware Decoding for Entmax Attention

**arXiv ID:** 2605.21649 | [PDF](https://arxiv.org/pdf/2605.21649v1)

**作者:** Gonçalo Duarte `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), Marcos V. Treviso `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于α‑entmax的稀疏解码框架，利用查询感知的 KV 页评分和支持恢复机制在推理阶段仅读取必要 KV 页，从而显著降低长上下文解码的内存传输开销；

**💡 创新点**

核心创新在于把稀疏解码从密集尾部近似转变为支持恢复：若选取的 KV 页包含 entmax 支持，则解码结果完全等价于全缓存；并引入高斯感知的阈值估计器自适应选择页；

**🔧 技术方法**

使用的技术包括：α‑entmax 转换、页面最小/最大键边界评分、基于高斯分布的阈值估计、定制 Triton 轻量级稀疏核，以及 FlashDecoding/Quest 的分页缓存结构；

**📊 数据集**

在 1B 参数的 softmax 与 entmax NAPE 模型上，使用 PG19 语言建模、RULER 长上下文检索以及 Passkey Retrieval 数据集进行评估；

**📈 对比分析**

与全缓存 softmax、全缓存 entmax、以及 Quest‑top‑k 软解码等基线比较，匹配 KV 预算时 entmax 稀疏解码在支持保留率、丢弃概率质量与输出误差上均优于 softmax；在 1M 令牌长度下实现 3.36×（softmax）和 5.43×（entmax）速度提升；

**⚠️ 局限性**

局限性包括：仍需在推理前预先计算 KV 页元数据；高斯估计在极端分布下可能产生误判；当支持被截断时误差仍然存在，且对极大序列长度的理论上限未作完整证明。

---

## 192. Higher Order Reasoning for Collaborative Communicationless Mobile Robot Operations

**arXiv ID:** 2605.21901 | [PDF](https://arxiv.org/pdf/2605.21901v1)

**作者:** Jonathan Reasoner `[一作]` (University of Virginia), Nicola Bezzo `[通讯]` (University of Virginia)

**通讯引用:** 1243 | [OpenAlex ID](https://openalex.org/A5091277504)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向无通信多机器人协同作业的动态认知规划框架，利用高阶动态认知逻辑（DEL）生成信念粒子、贝叶斯更新与行为树搜索，并通过时间感知MPPI实现对机器人轨迹的抓捕与交互，最终在仿真与实验环境中完成任务。

**💡 创新点**

创新点在于将高阶认知推理（包括第一、第二、第三阶信念粒子）与行为树结合，形成全局性、长期的估计-规划-协调流程；同时将MPPI与高阶推理耦合，用于在部分可观测环境下实现交互式抓捕与路径规划。

**🔧 技术方法**

技术手段包括：动态认知逻辑（DEL）+信念粒子传播、贝叶斯信念更新、行为树搜索、前沿探索策略、Dubins可达性分析、时间感知MPPI控制器；机器人模型为差分驱动，速度约1.0 m/s（实验时0.1 m/s）。

**📊 数据集**

数据集与测试环境：在150 m²仿真场景中，随机生成任务位置和机器人初始姿态，分别使用2、3、4台机器人进行100次随机试验；实验室测试使用2或3台地面机器人在4 m²区域内进行多次实测，使用Vicon系统记录轨迹。

**📈 对比分析**

与仅使用一阶信念的基线（前沿偏向相对机器人位置）进行比较；在600次仿真中，本方法在93.3%试验中与基线相同或更优，平均任务完成时间缩短20.8%（约169 s）。在实验中，本方法比基线快57.8%（从199 s降至115 s）。

**⚠️ 局限性**

局限性包括：仅针对单一任务的同质机器人团队；依赖完美传感与无干扰的无通信环境；对大规模、异构或对抗性场景的可扩展性未验证；MPPI与高阶推理的计算开销相对较高。

---

## 193. stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation

**arXiv ID:** 2605.21800 | [PDF](https://arxiv.org/pdf/2605.21800v1)

**作者:** Lucas Maes `[一作]` (Mila and University of Montreal), Randall Balestriero `[通讯]` (Brown University)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5047293370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一、可复现的世界模型研究与评估平台（stable-worldmodel），提供高效数据存储、标准化基线与规划器以及可扩展的评测基准。

**💡 创新点**

创新点在于：①将Lance列式存储格式与MP4/HDF5/LeRobot等格式兼容，显著提升多模态数据加载吞吐；②统一的环境、策略与规划器接口，减少重复实现；③基准套件加入可控视觉、几何、物理扰动（FoV），实现系统化的零样本泛化与鲁棒性测试。

**🔧 技术方法**

使用技术包括PyTorch、Gymnasium、Lance存储格式、MP4/HDF5/LeRobot数据集；实现了CEM、MPPI、梯度下降等MPC规划器；实现了多种基线世界模型（DINO-WM、LeWM、PLDM、TD-MPC2）和GCRL方法。

**📊 数据集**

使用的主要数据集包括模拟环境数据（Push‑T、MuJoCo、Atari、OGBench、Craftax等），以及从LeRobot收集的真实机器人轨迹；还支持MP4、HDF5原始录像文件。

**📈 对比分析**

在Push‑T等环境上对比了5个基线的成功率，结果与原论文保持一致；在分布外（随机策略、视觉扰动）条件下，所有模型成功率急剧下降，表明目前模型对鲁棒性与零样本泛化能力不足。

**⚠️ 局限性**

主要局限在于：①对分布外扰动极为脆弱，鲁棒性差；②大多数实验仅在仿真环境中进行，缺乏对真实世界的验证；③平台虽统一了评估流程，但仍需要手工编写训练脚本，未完全实现端到端自动化。

---

## 194. Benchmarking and Improving Monitors for Out-Of-Distribution Alignment Failure in LLMs

**arXiv ID:** 2605.21602 | [PDF](https://arxiv.org/pdf/2605.21602v1)

**作者:** Dylan Feng `[一作]` (University of California, Berkeley), Cassidy Laidlaw `[通讯]` (University of California, Berkeley)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5051435362)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并改进大型语言模型（LLM）在出现未见过的（OOD）对齐失效时的监测效果，提出并评估了新的基准（MOOD）和将 OOD 检测与安全守护模型（guard model）结合的方法。

**💡 创新点**

创新点在于：①首次系统地构建覆盖七类多样 OOD 对齐失效的基准；②证明单一 guard model 在 OOD 场景下召回率低；③提出将 Mahalanobis 距离、困惑度等 OOD 检测器与 guard model 组合能显著提升召回率，并展示规模扩展带来的正向趋势。

**🔧 技术方法**

使用了：预训练模型（Gemma 2、Qwen2.5）、自定义训练集（安全/不安全对话）、多种 OOD 检测技术（Mahalanobis 距离、困惑度、ensemble disagreement、instruction‑tuned scoring/uncertainty）以及 guard model 与 OOD 检测器的加权组合。

**📊 数据集**

数据集包括：训练集（85k 例安全/不安全对话）、七个 OOD 失败测试集（包括 jailbreak、功能调用欺骗、代码安全、规划、极端同情、控制行为等），以及 500 例 Swahili 机器翻译的安全对话作为 OOD 安全基准。

**📈 对比分析**

与单一 guard model 对比，组合方法在 1% FP 下的召回率提升 4–7 个百分点；在 Gemma 2 9B 上，Guard+Mahalanobis+Perplexity 召回率平均达到 45.7%，比单一 guard model 高 5.9%；规模增大时召回率呈正比例提升，甚至同 20 倍参数的 guard model 相比更优。

**⚠️ 局限性**

限制：即使是最优组合，部分 OOD 失败（如功能调用欺骗、恶意代码）召回率仍低于 20%；OOR 检测器会产生高 FP，需额外样本调优；实验仅覆盖少数 OOD 检测技术，未覆盖更广泛的多模态或大规模模型场景。

---

## 195. Broken Memories: Detecting and Mitigating Memorization in Diffusion Models with Degraded Generations

**arXiv ID:** 2605.22050 | [PDF](https://arxiv.org/pdf/2605.22050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 196. Planning in the LLM Era: Building for Reliability and Efficiency

**arXiv ID:** 2605.21902 | [PDF](https://arxiv.org/pdf/2605.21902v1)

**作者:** Michael Katz `[一作]` (IBM Research), Shirin Sohrabi `[通讯]` (IBM Research)

**通讯引用:** 1256 | [OpenAlex ID](https://openalex.org/A5037217032)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性评估并综述了三类基于大型语言模型的规划器生成方法：NL2Search、NL2PDDL、NL2Policy，强调将LLM用于构建阶段以生成可验证、资源高效的搜索组件、规划模型或策略。

**💡 创新点**

创新点在于将LLM从推理阶段迁移到构建阶段，提出将LLM用于代码合成、模型验证与迭代优化的框架，并对三种方法的现状与挑战进行了统一框架化分析。

**🔧 技术方法**

技术包括大型语言模型（如GPT‑5）、代码生成（Python/PDDL）、语义过滤与验证、强化学习与感知抽象、回溯搜索与全局搜索等。

**📊 数据集**

使用的主要数据集来自经典规划域（IPC）、WebArena、AppWorld等自然语言描述任务，但并未给出统一实验集，更多依赖公开的PDDL实例。

**📈 对比分析**

对方法的比较主要是概念性与理论层面，引用与传统规划器（如LAMA）的覆盖率和资源消耗对比，但缺乏统一实验与量化指标，整体性能尚未在真实任务中验证。

**⚠️ 局限性**

局限包括对状态表示与特征的先验假设、对部分可观测性与交互的处理不足、线性搜索与缺乏全局优化、对非确定性/数字域的支持有限，以及缺乏深层语义一致性验证。

---

## 197. Reasoning through Verifiable Forecast Actions: Consistency-Grounded RL for Financial LLMs

**arXiv ID:** 2605.21975 | [PDF](https://arxiv.org/pdf/2605.21975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 198. Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift

**arXiv ID:** 2605.21849 | [PDF](https://arxiv.org/pdf/2605.21849v1)

**作者:** Sungjun Lim `[一作]` (Yonsei University), Kyungwoo Song `[通讯]` (Yonsei University)

**通讯引用:** 590 | [OpenAlex ID](https://openalex.org/A5025711483)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Geometry‑Adaptive Explainer（GAE），在无梯度更新、仅使用无标签 OOD 激活的情况下，对字典解释器的子空间进行几何对齐，以恢复 OOD 情况下的因果可信度。

**💡 创新点**

创新点在于将 OOD 可信度下降形式化为 faithfulness gap 并证明其与第二矩阵偏移的上界，随后给出闭式旋转+约束岭回归的两步对齐方法，并提供理论保证。

**🔧 技术方法**

使用了正交 Procrustes 旋转、受限岭回归、Davis–Kahan 角度分析、第二矩阵移位量度等技术。

**📊 数据集**

在 GPT‑2 Small 与 Pythia‑1.4B 两个大型语言模型上，分别使用 FineWeb、Edgar 与 HaluEval 三种 OOD 设定进行实验。

**📈 对比分析**

与固定解释器、TERM、Finetune、Retrain、SAEBoost、FaithfulSAE 等基线对比，GAE 在所有 nAOPC、nComp 与 ΔCE 指标上均优于训练型方法，并且仅需 0.5–3 秒完成自适应。

**⚠️ 局限性**

局限性包括未在更大规模模型上验证、仅对字典的前 r 方向进行 SVD 截断导致信息损失、未对编码器进行自适应，且对非字典解释器的适用性待探究。

---

## 199. Why Semantic Entropy Fails: Geometry-Aware and Calibrated Uncertainty for Policy Optimization

**arXiv ID:** 2605.21801 | [PDF](https://arxiv.org/pdf/2605.21801v1)

**作者:** Zheyuan Zhang `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5348 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Geometric-aware Calibrated Policy Optimization (GCPO) 框架，用于在大型语言模型的后训练中通过调节梯度方差来提升推理与对齐性能。

**💡 创新点**

创新点在于将几何感知的不确定度（Cosine Dispersion、Barycentric Transport）与奖励信息校准（Reward Dispersion）相结合，系统性解决了熵基方法的各向异性与校准缺口。

**🔧 技术方法**

核心技术包括：GRPO 基础策略梯度、几何感知度量（CD、BoT）、奖励方差校准（RD）、以及对齐机制的统一调节。

**📊 数据集**

实验使用的主要数据集包括 NarrativeQA、Qasper、HotpotQA、MATH500、AIME24 与 OlympiadBench。

**📈 对比分析**

与 GRPO、Seed‑GRPO、EMPO、CARE 等基线对比，GCPO 在 NarrativeQA、Qasper 与 HotpotQA 上平均提升 10–20%（BLEU/F1/EM），在数学推理任务上也有显著改善，显示出更稳健的梯度控制。

**⚠️ 局限性**

主要局限是对奖励方差的依赖导致在奖励对齐弱的任务（如部分数学推理）提升有限；同时性能对模型规模和超参数（如 α、RD）较为敏感。

---

## 200. Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising

**arXiv ID:** 2605.21556 | [PDF](https://arxiv.org/pdf/2605.21556v1)

**作者:** Zhaoqi Zhang `[一作]` (Nanyang Technological University), Gao Cong `[通讯]` (Nanyang Technological University)

**通讯引用:** 16353 | [OpenAlex ID](https://openalex.org/A5045198704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

提出了一种针对多槽位的联合优化框架，能够在单一页面内全局协调保证展示广告的分配。

**💡 创新点**

创新点在于将多槽位广告分配建模为页面视图级别的二分图匹配，并引入页面视图约束（PV约束）和合同轮盘机制，实现曝光公平、冗余减少与多槽位协同。

**🔧 技术方法**

采用离线二分图匹配、基于KKT条件的双重变量迭代优化算法、合同轮盘采样选择模块以及自适应关键字控制等技术实现可扩展的分配方案。

**📊 数据集**

使用美团广告平台真实生产数据，在35%和70%灰度实验环境中进行在线实验验证。

**📈 对比分析**

与之前的单槽位基准系统对比，在线A/B与DID实验显示ARPU提升28.99%，商户ROI提升42.17%，支付ROI提升29.13%，CTR提升7.67%，支付CVR提升23.35%，合同履约率提升2.12%。

**⚠️ 局限性**

局限性包括：依赖离线优化，难以实时响应瞬时需求波动；合同轮盘机制仍基于启发式概率，可能在极端流量场景下失效；实验仅在美团平台验证，泛化性待进一步验证。

---

## 201. Behavior-Guided Candidate Calibration for Multimodal Recommendation

**arXiv ID:** 2605.22073 | [PDF](https://arxiv.org/pdf/2605.22073v1)

**作者:** Zesheng Li `[一作]` (University of Chinese Academy of Sciences), Honggang Qi `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10300 | [OpenAlex ID](https://openalex.org/A5047226078)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 BRIDGE 框架，将多模态推荐的候选校准与行为证据结合，以提升排序效果。

**💡 创新点**

创新点在于将频率感知的图证据编码与行为引导的候选校准相结合，并将校准限定在候选集内，以避免全局得分扭曲。

**🔧 技术方法**

采用双频率图证据编码器（DFGE）、行为证据归一化（BEN）和候选残差整合（CRI），并使用 SVD 频谱分解与信息瓶颈正则化。

**📊 数据集**

在 Amazon Baby、Sports、Electronics 三个公开数据集上进行实验，使用 BEIT3 视觉与文本特征。

**📈 对比分析**

与多种基线（LightGCN、VBPR、MMGCN、SMORE 等）在 Recall@20、NDCG@20 等指标上进行全排序评估，BRIDGE 在所有数据集均实现显著提升，尤其在 NDCG 上突出。

**⚠️ 局限性**

局限性在于若候选集缺失相关物品，残差校准无法弥补；对行为图噪声敏感，且仅在检索-排名两阶段框架内有效。

---

## 202. AgForce Enables Antigen-conditioned Generative Antibody Design

**arXiv ID:** 2605.21610 | [PDF](https://arxiv.org/pdf/2605.21610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 203. A Visitation Grid for Complete Coverage Foraging in Robot Swarms

**arXiv ID:** 2605.21947 | [PDF](https://arxiv.org/pdf/2605.21947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 204. Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection

**arXiv ID:** 2605.21977 | [PDF](https://arxiv.org/pdf/2605.21977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 205. PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects

**arXiv ID:** 2605.21572 | [PDF](https://arxiv.org/pdf/2605.21572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. On the Sample Complexity of Discounted Reinforcement Learning with Optimized Certainty Equivalents

**arXiv ID:** 2605.21763 | [PDF](https://arxiv.org/pdf/2605.21763v1)

**作者:** Oliver Mortensen `[一作]` (University of Copenhagen), Mohammad Sadegh Talebi `[通讯]` (University of Copenhagen)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5101839059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在有限折扣MDP中递归优化确定性等价(OCE)风险度量下的PAC学习，提出基于模型的价值迭代算法并给出样本复杂度上界；

**💡 创新点**

首次给出递归OCE风险下价值学习与策略学习的上界与下界，并精确说明仅当OCE对应的效用函数全域时才可PAC可学习；

**🔧 技术方法**

使用模型估计、价值迭代、Hoeffding不等式、离散化技术以及OCE光滑性分析；

**📊 数据集**

无实测数据集，纯理论分析与构造性MDP实验；

**📈 对比分析**

与已有的CVaR、熵风险等特定风险下的样本复杂度结果比较，证明上界在S·A、ε、δ方面最优，改进CVaR下的下界达到O(SA/(ε²(1‑γ)²τ²))；

**⚠️ 局限性**

对有效期望(1/(1‑γ))的依赖仍有较大缺口，且仅在生成器模型下成立；未来需改进算法或构造，以闭合有效期望的阶数，并扩展到离线/在线RL场景。

---

## 207. Learning Emergent Modular Representations in Multi-modality Medical Vision Foundation Models

**arXiv ID:** 2605.21861 | [PDF](https://arxiv.org/pdf/2605.21861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 208. X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation

**arXiv ID:** 2605.21699 | [PDF](https://arxiv.org/pdf/2605.21699v1)

**作者:** Sharath Turuvekere Sreenivas `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了X-Token方法，实现跨词表的知识蒸馏，使学生模型可从不同分词器的教师中学习；

**💡 创新点**

创新点在于构造稀疏投影矩阵W来对齐词表，并提出两种互补的损失形式P-KL和H-KL，解决传统方法中出现的关键词失配和过度保守匹配问题；

**🔧 技术方法**

使用了序列对齐、链式规则合并、稀疏投影、P-KL/H-KL损失、动态KD/CE比例缩放以及多教师加权；

**📊 数据集**

在Llama-3.2-1B学生模型上使用Nemotron-ClimbMix训练数据，并在MMLU、GSM8k、MATH、Winogrande、HellaSwag等五个基准上评测；

**📈 对比分析**

与无蒸馏、同分词器KD、ULD、GOLD等基线对比，X-Token在Qwen3-4B上平均提升约+3.8，Phi-4-mini上+0.5，并在多教师设置下相较单一教师提升约+1.3；

**⚠️ 局限性**

局限在于仅评估了少量教师组合，未覆盖更大规模或低重叠词表、指令调优等情形，未来需扩展到更广泛场景。

---

## 209. SceneGraphGrounder: Zero-Shot 3D Visual Grounding via Structured Scene Graph Matching

**arXiv ID:** 2605.21788 | [PDF](https://arxiv.org/pdf/2605.21788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 210. Secure and Parallel Determinant Computation for Large-Scale Matrices in Edge Environments

**arXiv ID:** 2605.22039 | [PDF](https://arxiv.org/pdf/2605.22039v1)

**作者:** Prajwal Panth `[一作]` `[通讯]` (KIIT Deemed to be University), Prajwal Panth (KIIT Deemed to be University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了面向边缘计算环境的安全并行行列式计算框架（SPDC），能够在任意数量的不可信边缘服务器上并行完成大规模矩阵的行列式计算，并保证输入输出的机密性与计算完整性。

**💡 创新点**

创新点包括：① 复合元素失真（CED）技术，将元素级扰动（EWO）与Panth旋转定理（PRT）相结合，既保持行列式不变又能高效混淆矩阵结构；② 支持任意 N 的并行 LU 分解，采用单向通信模式显著降低服务器间等待时间；③ 引入 Q₂、Q₃ 两种轻量级标量认证方案，实现单轮无密钥结果验证；④ 通过种子基校正因子实现解密，避免暴露密钥。

**🔧 技术方法**

技术手段包括：复合元素失真（EWO+PRT）、矩阵增强、并行 LU 分解、单向通信模式、随机向量与确定性标量认证（Q₂、Q₃）、种子生成与密钥生成、行列式恢复公式。

**📊 数据集**

论文以理论分析与复杂度对比为主，并未在公开数据集上进行实验评估；主要通过与已有方案（Lei、Fu、Liu、Gao 等）的理论复杂度、通信量与安全模型进行对比。

**📈 对比分析**

对比结果显示：SPDC 在计算复杂度、位运算量与浮点运算量上均低于现有方案；支持任意 N、单轮标量认证、恶意对手模型；相较于固定 2 服务器方案，通信量与延迟显著下降；在安全性方面同时满足隐私、并行与恶意抵抗。

**⚠️ 局限性**

局限性包括：仍停留在理论与模拟评估，缺乏真实边缘部署实验；多服务器实现时需引入阈值 ϵ 以处理浮点误差；对服务器间的通信带宽与同步仍有一定依赖；实现复杂度较高，对边缘设备的编码实现与调优需求较大。

---

## 211. Embedding-Based Federated Learning with Runtime Governance for Iron Deficiency Prediction

**arXiv ID:** 2605.21563 | [PDF](https://arxiv.org/pdf/2605.21563v1)

**作者:** Fan Zhang `[一作]` (University of Cambridge), Michael Roberts `[通讯]` (University of Cambridge)

**通讯引用:** 39498 | [OpenAlex ID](https://openalex.org/A5064284123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

基于冻结的 DeepCBC 嵌入模型和两层 MLP 的下游分类器，部署了一个跨机构联邦学习框架，用于预测铁缺乏症。

**💡 创新点**

创新点在于将领域专用基础模型的本地嵌入抽取与个性化聚合 FedMAP 结合，既降低通信开销，又显著提升在结构异质性数据上的性能，并实现了实时运行时治理。

**🔧 技术方法**

使用技术包括 DeepCBC 预训练嵌入模型、两层 MLP 下游分类器、FedAvg、FedProx 与 FedMAP 聚合策略、FLA3 运行时治理平台与 XACML 策略、Flower 生态下的联邦协调。

**📊 数据集**

采用的两个数据集分别来自阿姆斯特丹大学医学中心（AUMC）医院化验数据和 NHSBT 的 INTERVAL 献血者试验 FBC 数据，覆盖了两种铁缺乏率和临床工作流程差异显著的样本。

**📈 对比分析**

对比本地单点训练、FedAvg、FedProx 与 FedMAP 四种聚合方式，FedMAP 在两站点均提高了 ROC‑AUC（AUMC 0.947→0.959，NHSBT 0.856→0.867），并实现了最高宏观 ROC‑AUC 0.913 与宏观平衡准确率 0.839。

**⚠️ 局限性**

局限包括未进行隐私攻击分析与概率校准评估、未量化 FedMAP 的计算与通信开销、仅在两站点验证导致泛化受限，以及冻结嵌入模型需持续监控随时间漂移。

---

## 212. Industrial Dual-Arm Box Handling via Online Inertial Estimation and Convex Wrench Optimization

**arXiv ID:** 2605.22021 | [PDF](https://arxiv.org/pdf/2605.22021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 213. SO-Mamba: State-Ownership Mamba for Unrolled MRI Reconstruction

**arXiv ID:** 2605.22031 | [PDF](https://arxiv.org/pdf/2605.22031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. The Shape of Testimony: A Scalable Framework for Oral History Archive Comparison

**arXiv ID:** 2605.21623 | [PDF](https://arxiv.org/pdf/2605.21623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 215. The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems

**arXiv ID:** 2605.21997 | [PDF](https://arxiv.org/pdf/2605.21997v1)

**作者:** Yohei Nakajima `[一作]` `[通讯]` (Untapped Capital), Yohei Nakajima (Untapped Capital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了 ActiveGraph 运行时，将追加式事件日志作为单一真相源，所有状态通过对日志的确定性折叠形成图，行为以对图形模式的响应形式执行，并通过事件记录实现可重复回放、分叉和完整因果链追溯。

**💡 创新点**

创新点在于将事件日志从传统的旁路记录转为核心运行时基底，构建 deterministic replay、cheap fork 和全 lineage 的机制，并通过内容地址化的模型/工具响应缓存使得包含非确定性 LLM 调用的系统也能实现精确回放。

**🔧 技术方法**

采用技术包括事件源(event sourcing)、CQRS、增量图计算、图形模式匹配的行为订阅、内容地址化的模型/工具响应缓存、严格与宽松回放模式，以及 fork‑and‑diff 等结构化差异工具。

**📊 数据集**

使用了内置的投资尽职调查演示包（Northwind Robotics、Stellar Logistics、Pinecone Bio 等），所有工具与模型响应均为预先记录的快照，未依赖公开数据集。

**📈 对比分析**

本文未进行任务性能对比实验，而是以离线演示的 671 事件、93 对象、76 关系为例，展示在 30 秒内完成整个运行，并说明回放成本随日志长度线性增长；缺少与传统 Agent 框架的定量对比。

**⚠️ 局限性**

限制包括：需要动态检测确定性合同（易受作者疏忽影响）、存储空间随日志增长；缺乏检查点/压缩机制、模式迁移工具；并发写入/多代理冲突未处理；行为可能循环导致无限递归；未评估实际任务性能。

---

## 216. Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews

**arXiv ID:** 2605.21713 | [PDF](https://arxiv.org/pdf/2605.21713v1)

**作者:** André V. Duarte `[一作]` (Instituto Superior Técnico), Lei Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12299 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Sem-Detect，能够区分完全人工写的同行评审、使用LLM润色的人工评审以及完全由AI生成的评审。

**💡 创新点**

创新点在于将文本特征与评审的论点级语义相结合，利用AI模型在同一论文上往往收敛到相似论点这一规律，能够在三分类任务中显著提升准确率。

**🔧 技术方法**

使用 LightGBM 分类器，提取文本统计特征（困惑度、熵等）和基于语义嵌入的论点匹配特征；模型训练过程中采用多模型生成的 AI 参考评审进行语义对齐。

**📊 数据集**

构建了包含 20,000 条评审的数据集，来源于 ICLR 与 NeurIPS 2021‑2022 的公开评审，包含人类原稿、LLM 润色稿以及四大 LLM 完全生成稿；同时在医学影像 MIDL 2022 及 ICLR 2026 的评审上做了迁移与时序验证。

**📈 对比分析**

与现有通用检测器（Fast‑DetectGPT、RADAR 等）和领域特定方法（TF、Anchor、EditLens）对比，在二分类任务中 AUC 0.999、TPR@0.1% FPR 为 0.760，比 EditLens 提升 25.5%；在三分类任务中宏 F1 达 0.84，AI 误判率低于 0.7%，LLM 润色误判率 3.5%。

**⚠️ 局限性**

主要限制是随着 AI 生成质量的提升，模型难以区分原创性高、质量优秀的 AI 评审；此外对不同学科、不同评审风格的迁移性能仍有限，且误判会对评审者声誉产生潜在负面影响。

---

## 217. Look-Closer-Then-Diagnose: Confidence-Aware Ultrasound VQA via Active Zooming

**arXiv ID:** 2605.21652 | [PDF](https://arxiv.org/pdf/2605.21652v1)

**作者:** Yue Zhou `[一作]` (TU Munich), Zhongliang Jiang `[通讯]` (University of Hong Kong)

**通讯引用:** 4126 | [OpenAlex ID](https://openalex.org/A5030672669)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了一个包含空间定位与置信度标注的超声视觉问答数据集，并通过监督微调和强化学习实现“先放大再诊断”的交互式推理流程，最终得到既能准确定位病灶又能根据诊断不确定性做出不同置信度的模型。

**💡 创新点**

（1）首次提出结构化的Zoom‑then‑Diagnose范式，使模型在超声图像上能像临床超声师一样先定位病灶再做诊断；（2）在GRPO框架下引入基于群组一致性的置信度奖励，兼顾诊断准确性与临床主观不确定性的对齐；（3）将定位、语义生成与置信度校准统一到同一模型中。

**🔧 技术方法**

使用Qwen2.5‑VL（7B/72B）作为基座模型，采用LoRA（rank 32）进行微调；利用工具调用（<tool_call>）实现自动裁剪；在GRPO中采样8条rollout、温度0.7，使用群组一致性计算置信度，构造奖励函数。

**📊 数据集**

公开的Breast Chain‑of‑Thought、Thyroid Nodule Ultrasound数据集以及自建的Liver超声数据集（共205例），每个样本标注病灶框、诊断结果及二值置信度（共识/分歧）。

**📈 对比分析**

与多种基线（单轮对话、医学预训练、工具调用、奖励仅关注准确性）相比，本文方法在肝、乳腺两类已知域和甲状腺交叉域上均实现了最高的诊断准确率、定位mIoU、置信度对齐度，且校准误差（ECE）显著降低，表明模型既保持高精度又能合理表达不确定性。

**⚠️ 局限性**

主要限制包括：数据集规模仍有限（仅两名超声师标注），难以覆盖更广泛的临床场景；模型依赖大型预训练VLM，部署成本高；对极端噪声或多模态（如B超与CT）混合场景的鲁棒性尚未验证。

---

## 218. Reinforced Preference Optimization for Reasoning-Augmented Recommendations

**arXiv ID:** 2605.21967 | [PDF](https://arxiv.org/pdf/2605.21967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 219. Detecting Offensive Cyber Agents: A Detection-in-Depth Approach

**arXiv ID:** 2605.21956 | [PDF](https://arxiv.org/pdf/2605.21956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 220. Guided Trajectory Optimization with Sparse Scaling for Test-Time Diffusion

**arXiv ID:** 2605.21907 | [PDF](https://arxiv.org/pdf/2605.21907v1)

**作者:** Gang Dai `[一作]` (Guangdong University of Technology), Shuaicheng Niu `[通讯]` (Nanyang Technological University)

**通讯引用:** 760 | [OpenAlex ID](https://openalex.org/A5064880801)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了测试时缩放（Test‑Time Scaling）技术，旨在通过在推理阶段动态优化噪声轨迹来提升扩散模型的生成质量。

**💡 创新点**

创新点在于提出 Reward‑guided Trajectory Scaling (RTS)，将稀疏关键步选择（基于 PCA 曲率分析）与奖励引导的粗细交替噪声优化相结合，实现在初始噪声与中间噪声层面主动探索高奖励区域。

**🔧 技术方法**

采用的技术包括稀疏测试时缩放框架、奖励引导的噪声优化（粗细交替搜索）、PCA 曲率分析确定关键步、随机球面邻域构造、以及代理梯度估计等。

**📊 数据集**

实验使用了两个主流基准数据集：GenEval（553 条文本提示）和 T2I‑CompBench（1800 条文本提示）来评估生成效果。

**📈 对比分析**

与 Best‑of‑N、Zeroth‑Order、Feynman‑Kac 等基线相比，RTS 在 ImgReward 上提升约 60%，在 GenEval 上提升约 15%，且在相同 NFE（函数评估次数）预算下保持最优性能。

**⚠️ 局限性**

局限性包括：对奖励函数的依赖导致可迁移性受限；搜索过程仍有显著计算开销；关键步选择策略对不同模型或数据分布可能不具普适性。

---

## 221. How Sparsity Allocation Shapes Label-Free Post-Pruning Recoverability

**arXiv ID:** 2605.21972 | [PDF](https://arxiv.org/pdf/2605.21972v1)

**作者:** Qishi Zhan `[一作]` (Marquette University), Liang He `[通讯]` (Tongji University)

**通讯引用:** 6844 | [OpenAlex ID](https://openalex.org/A5101739714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在固定的无标签修复后端（ASR）下，层级稀疏分配策略（ERK 与 LAMP）对剪枝后网络可恢复性的影响。

**💡 创新点**

发现稀疏分配会显著改变剪枝后可恢复的准确率，并识别出一个“修复敏感”转折区间，其中 BN 失效但 ASR 仍能恢复，且 ERK 与 LAMP 的优势在不同体系结构与数据集上可逆。

**🔧 技术方法**

采用结构化稀疏分配（ERK、LAMP）、基于幅值的剪枝、ASR（方差匹配、压缩与裁剪）以及 BN 重新校准等无标签修复技术。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、Imagenette、ImageNet‑100 以及 DenseNet‑121（CIFAR‑10/100）上进行实验。

**📈 对比分析**

通过比较不同稀疏率（90%–95.5%）下的剪枝+ASR 与 BN、偏置校正、仿射校正等基线的测试准确率，结果显示 ASR 在所有条件下平均提升约 17 个百分点，并且稀疏分配差异可导致 5–10 个百分点的准确率差异。

**⚠️ 局限性**

局限在于假设稠密模型的激活统计是理想修复目标，转折区间与体系结构高度相关，仅评估了 ERK 与 LAMP 两种分配策略，且未与完整的后训练微调或其他全局剪枝方法进行对比。

---

## 222. Finite-Aperture Planar Fluid Antenna Array

**arXiv ID:** 2605.22040 | [PDF](https://arxiv.org/pdf/2605.22040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 223. Investigating Concept Alignment Using Implausible Category Members

**arXiv ID:** 2605.21683 | [PDF](https://arxiv.org/pdf/2605.21683v1)

**作者:** Sunayana Rane `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 50344 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过向大语言模型提出跨类别的可疑对象-类别组合问题，评估其概念边界与人类的差异。

**💡 创新点**

创新点在于使用人类认为显而易见的错误类别成员问题，揭示模型概念理解与人类的显著不一致。

**🔧 技术方法**

采用大语言模型（GPT‑4o、Claude Sonnet 4.5、Gemini 2.5 Flash 等）、统计检验（Mann‑Whitney U、S̆idák 校正）以及人类参与者评分。

**📊 数据集**

使用 Rosch 与 Mervis 的 20 个对象和 6 个超类构建 708 个问题，收集 563 名人类评估者的评分。

**📈 对比分析**

对比模型与人类的 0–10 评分，发现模型普遍更宽松；在 312/708 题目上显著偏差；还找出了 28 个高分歧题目。

**⚠️ 局限性**

局限性包括仅覆盖有限对象与类别，未分析导致误差的机制，且对新模型的适用性未知。

---

## 224. Near-Field User Location Inference From Far-Field Power Measurements

**arXiv ID:** 2605.21815 | [PDF](https://arxiv.org/pdf/2605.21815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 225. An $Ω(n \log n)$ Randomized Lower Bound for Cutting a Cake into Proportionally Fair Pieces

**arXiv ID:** 2605.21829 | [PDF](https://arxiv.org/pdf/2605.21829v1)

**作者:** Stephen Arndt `[一作]` (Carnegie Mellon University), Trung Tran `[通讯]` (University of Pittsburgh)

**通讯引用:** 1539 | [OpenAlex ID](https://openalex.org/A5015143885)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 Robertson‑Webb 模型下，证明任何随机化的蛋糕切分协议在保证每个玩家获得连通片段且比例公平时，期望查询次数至少为 Ω(n log n)。

**💡 创新点**

把已知的确定性 Ω(n log n) 下界推广到随机化协议，并给出完整的决策树与 Yao 原理证明。

**🔧 技术方法**

使用对抗策略、决策树分析、凸性与 Jensen 不等式的组合来得到期望深度上界；同时将原始协议转化为仅使用裁剪的“原始”协议。

**📊 数据集**

论文不涉及具体数据集，而是以抽样的实例集合（由所有可能的排列产生的特定点集）作为对抗实例。

**📈 对比分析**

通过构造的均匀实例分布和对抗策略，证明任何随机化协议在该分布下至少需要 Ω(n log n) 次查询；未提出可与之比较的算法，因而该结果说明此类协议无法突破此下界。

**⚠️ 局限性**

局限性：仅适用于连通片段且只考虑比例公平；对非连通方案、其他公平性（如平等、正义）或不同度量的下界尚未得到扩展。

---

## 226. LiveR: Fine-Grained Elasticity via Live Reconfiguration for Model Training

**arXiv ID:** 2605.22014 | [PDF](https://arxiv.org/pdf/2605.22014v1)

**作者:** Haoyuan Liu `[一作]` (Shanghai Jiao Tong University), Wei Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 40806 | [OpenAlex ID](https://openalex.org/A5008881437)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种实时弹性重配置运行时，能够在不停止训练的情况下将大型语言模型（LLM）迁移到新的GPU拓扑，显著缩短重配置停机时间。

**💡 创新点**

创新点在于：① 双世界（活跃与阴影）并行构建；② 通过 mock 进程组隐藏新节点的初始化延迟；③ 使用几何交集驱动的分层流式重分片，实现任意 TP/PP/DP 维度的状态迁移且内存受限；④ 将重配置从存储 I/O 路径迁移到高带宽网络流。

**🔧 技术方法**

实现依托 Megatron‑LM 与 PyTorch 分布式框架，利用 NCCL 进行多组通信，采用 Python 伴随管理器实现后台准备，采用层流式数据传输协议实现状态迁移。

**📊 数据集**

在多节点 A800 GPU 集群上对 GPT‑14B、30B 等大规模模型进行训练实验，未明示具体语料，但使用标准大型语言模型预训练数据。

**📈 对比分析**

与传统检查点+重启（Megatron‑LM Checkpoint）和优化检查点（UCP/ByteCheckpoint）对比，重配置停机时间从分钟级降低到 2–6 秒，速度提升 14–23 倍；在高频资源波动场景下，训练有效性可达 99% goodput，显著优于基线。

**⚠️ 局限性**

局限性包括：仅适用于有预警或计划性资源变更；依赖高速网络（PCIe/InfiniBand）；对突发失效无内置恢复；在极高并发重配置时需序列化处理；在极大规模（>1024 GPU）时准备时间可能接近预警窗口，需要提前预热。

---

## 227. MindLoom: Composing Thought Modes for Frontier-Level Reasoning Data Synthesis

**arXiv ID:** 2605.21630 | [PDF](https://arxiv.org/pdf/2605.21630v1)

**作者:** Haiyang Shen `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**通讯引用:** 78797 | [OpenAlex ID](https://openalex.org/A5100369226)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于“思维模式（Thought Mode）”的组合式数据合成框架，利用逆向工程将已解决的高难度问题拆解为原子知识‑推理变换，并在此基础上自动生成前沿级别的推理题目；

**💡 创新点**

核心创新在于：①将问题难度拆解为可组合的原子思维模式；②通过逆向工程提取思维模式链；③构建基于检索的兼容性匹配与分布对齐的采样策略，防止模式坍塌；④引入多轮Rollout判定与来源过滤，生成可供SFT的高质量数据；

**🔧 技术方法**

技术手段包括：LLM驱动的逆向工程与思维模式提取；基于嵌入的检索模型；分布对齐的稀缺奖励采样；LLM生成与多轮判定；源可追溯过滤；全参数SFT；

**📊 数据集**

使用了58,526条来源题目（来自16个STEM与数学类数据集）构建思维模式库，并在九个评测基准（CS-Bench、ChemBench、HLE、MedQA、SciBench、MATH-500、HMMT-Feb、HMMT-Nov、AIME 2025）上评估；

**📈 对比分析**

与基线（原始模型、DS‑V3.2 Distill、MegaScience、OpenThought）对比，生成的数据在所有九个基准上均显著提升pass@1/3，尤其在数学竞赛类任务上提升幅度最大，表明该方法在数据效率与难度控制上优于传统合成与外部数据；

**⚠️ 局限性**

局限性包括：对LLM提示质量高度依赖；思维模式的定义与抽象可能无法覆盖所有推理范式；生成过程计算成本高（多轮Rollout判定与检索）；源数据分布不均可能导致某些领域模式稀缺，影响生成多样性。

---

## 228. Contract Based Verification of Non-functional Requirements for Embedded Automotive C Code

**arXiv ID:** 2605.21532 | [PDF](https://arxiv.org/pdf/2605.21532v1)

**作者:** Jesper Amilon `[一作]` (Kth Royal Institute Of Technology), Karl Palmskog `[通讯]` (Kth Royal Institute Of Technology)

**通讯引用:** 495 | [OpenAlex ID](https://openalex.org/A5015320417)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究针对汽车嵌入式 C 代码中的非功能性需求，提出了一套基于模块接口的规则集，并设计了一个模块接口规范（IS）合同语言，用 Frama‑C 插件实现对这些规则的静态检查，并将其嵌入到完整的开发与验证工作流中；同时在两个真实车载 ECU 模块（SFLD 与 SGMM）上进行案例验证，并评估 LLM 生成代码的可验证性。

**💡 创新点**

创新点在于：1) 把非功能性需求从传统的 MISRA 级别抽象为模块层面的控制流和数据流规则；2) 设计了能够桥接 ACSL 函数合同与系统级合同的 IS 合同语言；3) 实现了一个开源 Frama‑C 插件，支持对接口契约、调用顺序、函数指针、全局变量使用等规则的检查；4) 将此检查器集成到多层“critic”验证流程中，形成可重复的、可自动化的验证管道；5) 对 LLM 生成代码进行可验证性实验，探索生成式 AI 与形式化验证的结合。

**🔧 技术方法**

主要技术包括：ACSL 语言与 Frama‑C 证明框架、Frama‑C 的 EVA、CFA、WPS 等插件、MISRA 静态分析器、Cppcheck、VerNFR（非功能性规则检查），以及自定义的插件实现（OCaml）和 Python 脚本用于集成与报告。

**📊 数据集**

数据集来源于两套车载 ECU 控制器模块：SFLD（车轮转向油压检测）和 SGMM（安全变速箱控制），两套模块均来自实际生产代码，分别约 88 LOC / 138 LOC；对应的 ACSL 规范和外部函数合同也均为作者自行编写。

**📈 对比分析**

评估方法：对每个模块进行编译、功能性验证（Frama‑C）、静态 MISRA 检查、L2 规则检查；记录各阶段所用时间与违例数量。性能结果（以 CPU 时长计）：SGMM 总计 87.3 秒（编译 0.4s，MISRA 58.9s）；SFLD 总计 40.7 秒（编译 0.3s，MISRA 20.1s）。两模块均通过功能性验证；MISRA 发现 4~6 条必需违规，提示规范偏差；非功能规则在两模块均全部通过。LLM 实验：Claude Sonnet 4.6 成功生成 32 个可编译实现，全部通过非功能检查；SFLD 的 LLM 代码仅在 82/84 条证明义务超时，因而被判定为未验证。

**⚠️ 局限性**

局限性包括：1) 插件目前无法覆盖所有规则（如函数指针、指针算术、调用顺序推断等）；2) 依赖于精确的 IS 合同与 ACSL 规范，若规范不完整或含糊会影响验证；3) LLM 生成代码的格式化不稳定，导致编译失败；4) 对真实大型系统的可扩展性尚未评估；5) 规则检查以静态分析为主，难以处理动态行为和运行时依赖；6) 研究仅针对两个案例，缺乏更广泛的工业验证。

---

## 229. Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing the Answer

**arXiv ID:** 2605.22007 | [PDF](https://arxiv.org/pdf/2605.22007v1)

**作者:** Jewon Yeom `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 17797 | [OpenAlex ID](https://openalex.org/A5100641870)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在指令微调的大语言模型中，模型在承诺步骤（commitment step）上的分布如何导致幻觉（hallucination）出现，提出并验证了概念层面的语义概率质量（semantic probability mass）作为分析工具。

**💡 创新点**

创新点在于：①将“概念”定义为等价类的词元完成，提出了每步语义概率质量作为探测器；②发现大多数幻觉是“承诺失败”，即模型在承诺步骤已分配足量质量给正确概念，却因质量分散而误选错误词元；③揭示指令微调与规模共同导致分布尖锐化，是导致自信幻觉的根本机制。

**🔧 技术方法**

主要技术包括：token级别语义聚合、per‑step semantic probability mass 计算、熵（entropy）与熵峰定位、隐藏层状态探测（hidden‑state probing）、注意力分布分析、逻辑回归 AUROC 评估、以及对不同规模与指令/基础版本模型的对比实验。

**📊 数据集**

使用的数据集有：TriviaQA、NQ‑Open（短答题），以及 MMLU、ARC‑Challenge（多选题），共计约 3,000 条短答样本与 2,672 条多选样本。

**📈 对比分析**

方法比较：对 Qwen3.5、Llama‑3.2/3.1 等 18 个模型（0.8B–72B）进行 Instruct vs Base 对比，评估幻觉率、准确率、AUROC、ECE、Brier 分数；结果表明 Instruct 模型的幻觉率随规模递增，指令微调显著提升承诺尖锐度，且在正确概念质量足量时仍出现错误输出。

**⚠️ 局限性**

局限性包括：①语义概率质量探测器需先验知答案的同义词集；②实验仅覆盖短答题与贪婪解码；③仅对 Qwen 与 Llama 两大开源系列进行验证，未检验闭源模型或更长生成任务。

---

## 230. Dual-Integrated Low-Latency Single-Lens Infrared Computational Imaging for Object Detection

**arXiv ID:** 2605.21964 | [PDF](https://arxiv.org/pdf/2605.21964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 231. From TF-IDF to Transformers: A Comparative and Ensemble Approach to Sentiment Classification

**arXiv ID:** 2605.22003 | [PDF](https://arxiv.org/pdf/2605.22003v1)

**作者:** Dip Biswas Shanto `[一作]` (KIIT Deemed to be University), Suresh Chandra Satapathy `[通讯]` (KIIT Deemed to be University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较传统机器学习、深度学习与Transformer模型在IMDB影评情感分类任务上的表现，并构建软投票集成提升准确率。

**💡 创新点**

将多种模型与软投票集成结合，使用SHAP进行模型可解释性，系统评估各模型对复杂否定、讽刺的处理效果。

**🔧 技术方法**

TF‑IDF、SVM、Logistic回归、Naïve Bayes、LightGBM、LSTM、DistilBERT、RoBERTa以及软投票集成、SHAP解释。

**📊 数据集**

IMDb 50,000条影评（25k正、25k负）。

**📈 对比分析**

通过准确率、F1、ROC‑AUC比较，RoBERTa单模型最高93.02%，软投票集成进一步提升1‑3%，在处理讽刺、否定句方面表现优于单一模型。

**⚠️ 局限性**

仍难以准确捕捉双重否定、讽刺与混合情绪，传统TF‑IDF模型在上下文依赖上受限，Transformer模型对短文本易过拟合。

---

## 232. Minimum Sum Set Cover: Structures and Algorithm

**arXiv ID:** 2605.21920 | [PDF](https://arxiv.org/pdf/2605.21920v1)

**作者:** Zhongyi Zhang `[一作]` (Shandong University), Yixin Cao `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5827 | [OpenAlex ID](https://openalex.org/A5013247988)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究最小总和集合覆盖（minimum sum set cover）问题，给出了新的上界、下界和参数化算法。

**💡 创新点**

创新点：①证明对任意超图 H，最小总和集合覆盖大小 (H) 与经典最小集合覆盖大小 τ(H) 的比值不超过 τ(H)·log₂|E(H)|；②在图的情形下进一步改进为 (G) ≤ 2τ(G)log₂τ(G) 并构造近乎匹配的图；③提出基于 Sunflower 引理的固定参数分支算法，证明在边数有界的超图上对 τ(H) 进行参数化时是 FPT。

**🔧 技术方法**

主要技术：组合论分析（如有效覆盖递减、指数衰减论证）、构造性证明（特殊顶点集合与边的编码）、Sunflower Lemma 与分支搜索、递归构造最优序列、归纳证明和成本比较的数学不等式。

**📊 数据集**

该工作纯理论化学，没有使用具体数据集，而是通过构造性实例和组合式上界来展示结果。

**📈 对比分析**

性能评价：理论上给出了最优解成本上界 (G) ≤ 2τ(G)log₂τ(G)，并在超图上给出时间复杂度 O(r^r + k^r·m + (rk)^{rk}) 的 FPT 算法，显著优于先前针对图的 O(τ^2 log τ) 上界。

**⚠️ 局限性**

限制与未解决问题：①在图中仍存在 loglogτ(G) 的上界与下界之间的缺口；②对无界边数超图的参数化可解性尚未确定，且作者推测在此类超图上问题可能仍为 NP‑hard。

---

## 233. Universal CT Representations from Anatomy to Disease Phenotype through Agglomerative Pretraining

**arXiv ID:** 2605.21906 | [PDF](https://arxiv.org/pdf/2605.21906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 234. MLLMs Know When Before Speaking: Revealing and Recovering Temporal Grounding via Attention Cues

**arXiv ID:** 2605.21954 | [PDF](https://arxiv.org/pdf/2605.21954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 235. Value-Gradient Hypothesis of RL for LLMs

**arXiv ID:** 2605.21654 | [PDF](https://arxiv.org/pdf/2605.21654v1)

**作者:** Arip Asadulaev `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Martin Takac `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5070679093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对无价值网络（critic-free）强化学习（如PPO/GRPO）在大型语言模型（LLM）后训练中的行为进行理论与实验分析，发现其背向传播已携带近似的价值梯度信号；

**💡 创新点**

创新点在于：①将连续松弛与增量噪声参数化相结合，证明在期望上无价值网络更新等价于价值梯度更新；②证明在离散Transformer中注意力提供了可导的时间信用通道，使得离散采样缺口误差可通过策略熵控制；③构建RL影响律（RL Impact Law），将可用价值梯度信号与可获得奖励头部（reward headroom）相乘预测RL收益；

**🔧 技术方法**

采用连续松弛、路径梯度(Pathwise Derivative)、注意力 Jacobian、成本状态（costate）估计、熵控制误差上界、以及GRPO/ PPO算法的数值实验；

**📊 数据集**

使用OLMo-2 1B模型在不同预训练步长（50k到1M）生成的检查点，并在一个可微分的“标签复制”奖励任务上进行评估；

**📈 对比分析**

与传统的基线比较（如单独使用预训练奖励或随机RL），通过相关性分析显示RL影响律预测的RL收益与实际增益的Spearman相关系数约为0.60，加入预训练竞争力后相关系数提升至0.73，说明理论能较好解释不同检查点的RL效能；

**⚠️ 局限性**

局限性包括：实验任务为简化的可微分奖励任务，未涵盖自然文本生成或复杂评价指标；对奖励尺度、学习率、批大小等超参数敏感；RL预算过少或过多都会影响最终收益，理论仅提供预训练检查点的“准备度”指示，实际RL训练仍需进一步调优。

---

## 236. HIDBench: Benchmarking Large Language Models for Host-Based Intrusion Detection

**arXiv ID:** 2605.21773 | [PDF](https://arxiv.org/pdf/2605.21773v1)

**作者:** Danyu Sun `[一作]` (University of California, Irvine), Zhou Li `[通讯]` (University of California, Irvine)

**通讯引用:** 26287 | [OpenAlex ID](https://openalex.org/A5101911602)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HIDBench基准，用于评估大语言模型在主机入侵检测中的能力

**💡 创新点**

创新点在于统一三大公开日志数据集并设计从原始主机日志到LLM可用输入的完整流水线，解决日志规模、噪声和不平衡问题

**🔧 技术方法**

使用数据分割、恶意证据识别、攻击图扩展、攻击链重建等步骤，并结合提示模板、投票与自我反思等推理策略

**📊 数据集**

使用DARPA-E3、DARPA-E5和NodLink（NL-SD）三组公开日志数据集

**📈 对比分析**

对9种前沿LLM进行评估，结果显示精度高但MCC在噪声大、数据不平衡时显著下降，模型分为保守、平衡与过敏感三类

**⚠️ 局限性**

局限在于不支持实时检测、依赖预定义窗口策略、未探索检索增强生成等更广泛的设计空间

---

## 237. TO-Agents: A Multi-Agent AI Pipeline for Preference-Guided Topology Optimization

**arXiv ID:** 2605.21622 | [PDF](https://arxiv.org/pdf/2605.21622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 238. SDGBiasBench: Benchmarking and Mitigating Vision--Language Models' Biases in Sustainable Development Goals

**arXiv ID:** 2605.21919 | [PDF](https://arxiv.org/pdf/2605.21919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 239. Diverse Yet Consistent: Context-Guided Diffusion with Energy-Based Joint Refinement for Multi-Agent Motion Prediction

**arXiv ID:** 2605.22017 | [PDF](https://arxiv.org/pdf/2605.22017v1)

**作者:** Lei Chu `[一作]` (University of Southern California), Yuhuan Zhao `[通讯]` (University of Southern California)

**通讯引用:** 334 | [OpenAlex ID](https://openalex.org/A5110997511)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种名为CODA的多代理运动预测框架，结合历史轨迹上下文进行导向生成，并利用能量模型对联合分布进行细化；

**💡 创新点**

创新点在于引入动态上下文指导条件（DCGC）与自适应条件集成模块（ACIM），以及基于能量的联合分布细化（JDR），在保持多模态多样性的同时显著提升多代理一致性；

**🔧 技术方法**

采用了扩散生成模型（带classifier‑free guidance）、Transformer自注意力提取动态与全局上下文、能量模型进行联合分布修正，并结合best‑of‑K与多样性损失；

**📊 数据集**

在ETH/UCY、SDD、NBA和JRDB四个主流人类轨迹预测基准数据集上进行实验；

**📈 对比分析**

与S‑GAN、Trajectron++、PECNet、Y‑Net、MemoNet、AgentFormer、MoFlow、NRMF等先进方法对比，CODA在ADE/FDE（边缘指标）上实现了SOTA，且在JADE/JFDE（联合指标）上保持竞争力；

**⚠️ 局限性**

局限性包括训练时间相对较长，对CR（碰撞率）指标仍有提升空间，且当前仅使用坐标输入，未结合LiDAR或可行区域信息。

---

## 240. Residual Skill Optimization for Text-to-SQL Ensembles

**arXiv ID:** 2605.21792 | [PDF](https://arxiv.org/pdf/2605.21792v1)

**作者:** Jiongli Zhu `[一作]` (University of California San Diego), Babak Salimi `[通讯]` (University of California San Diego)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5103209063)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Residual Skill Optimization框架，通过不断构建互补的agent技能来生成多样化的Text-to-SQL候选集

**💡 创新点**

创新点在于将Pass@K覆盖度作为显式目标，按剩余错误样本递归优化新技能，避免随机或高温采样导致的相关失败

**🔧 技术方法**

采用自然语言技能（prompt）调控agent行为，使用反射式prompt优化、团队式子技能池和多轮批量递归训练；最终通过对比式选择候选SQL

**📊 数据集**

在Spider2-Lite（SQLite、Snowflake、BigQuery）和BIRD-Critic（PostgreSQL）数据集上进行评估

**📈 对比分析**

与多种基线（包括单体生成、随机采样、已有ensemble）对比，取得在Pass@8和最终选取准确率上显著提升，Snowflake上+11.1点、BigQuery上+8.3点，BIRD-Critic上+2.6点

**⚠️ 局限性**

仍存在候选覆盖与最终选择不匹配的差距（Pass@8与selected accuracy之间较大），以及对更复杂交互式调试环境的适配性有限

---

## 241. On-Policy Consistency Training Improves LLM Safety with Minimal Capability Degradation

**arXiv ID:** 2605.21834 | [PDF](https://arxiv.org/pdf/2605.21834v1)

**作者:** Andy Han `[一作]` (New York University), Rico Angell `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的一致性训练方法，称为在线一致性训练（OPCT），旨在通过对模型自身响应的监督来提高模型的安全性和一致性。

**💡 创新点**

OPCT通过在线生成响应并使用自身的对比提示进行监督，克服了传统监督微调（SFT）方法的局限性，显著提高了模型在安全性方面的表现。

**🔧 技术方法**

使用了在线一致性训练（OPCT）技术，该技术通过对比输入对生成模型的响应进行训练。

**📊 数据集**

在三个安全性任务上进行了评估，使用的数据集包括Qwen3-8B、NVIDIA-Nemotron-3-Nano-30B-A3B-BF16和gpt-oss-20b等模型的对比提示。

**📈 对比分析**

与传统的SFT方法相比，OPCT在所有安全性指标上均表现更好，尤其是在减少谄媚行为和提高对越狱攻击的防御成功率方面，OPCT的表现显著优于SFT。

**⚠️ 局限性**

OPCT在某些情况下可能仍然面临能力回归的问题，尽管总体上它在保持模型能力方面表现良好。

---

## 242. A Large Language Model Approach to Generating Bypass Rules for Malware Evasion in Analysis Sandbox

**arXiv ID:** 2605.21821 | [PDF](https://arxiv.org/pdf/2605.21821v1)

**作者:** Zhiyong Sui `[一作]` (Louisiana State University), Aisha Ali-Gombe `[通讯]` (Louisiana State University)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5032852782)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 ABLE 框架，利用大语言模型自动生成、验证并迭代优化 YARA 规则，以绕过沙箱环境中的恶意软件逃避检查并暴露隐藏的恶意行为。

**💡 创新点**

创新点在于将 LLM 的语义推理、规则自检与修复管线以及沙箱反馈驱动的多轮迭代相结合，首次实现大规模、自动化的逃避规则生成与验证，并通过多模型、多提示策略显著提升可观察行为覆盖率。

**🔧 技术方法**

技术方案包括使用 Qwen3‑8B、Llama 3.1‑8B、Gemma 3‑12B、DeepSeek‑R1‑7B 四大 LLM；多种推理策略（Zero‑Shot、Chain‑of‑Thought、Counterfactual、Adversarial）；YARA 规则自动校验与修复管线；CAPEv2 沙箱执行与反馈引擎；以及基于 SSH 的多 VM 并行控制。

**📊 数据集**

数据集为从 MalwareBazaar、VirusTotal 等来源收集的 334 个标注有沙箱逃避特征的 Windows PE 样本，涵盖 native、.NET 等多种类型。

**📈 对比分析**

通过与基线沙箱不使用规则的行为签名对比，评估 ABLE 在 13,778 次沙箱执行中的绕过成功率达 79%，发现 82 个隐藏签名，恶意软件家族识别率提升 47%，单模型单轮约 15% 成功率，迭代与模型融合后提升至近 80%，显著优于现有平台。

**⚠️ 局限性**

局限性包括仅针对 Windows PE，未覆盖非 Windows 或 ARM 平台；生成的逃避规则大多为 skip 跳转，缺乏多样化的控制流操作；缺乏完整的真值集，评估仅相对基线；LLM 仍需多轮修正，复杂逃避场景表现有限；模型训练与推理成本较高。

---

## 243. Throughput-Optimal Multiresource-Job Scheduling with Continuous Requirement Distribution

**arXiv ID:** 2605.21715 | [PDF](https://arxiv.org/pdf/2605.21715v1)

**作者:** Heyuan Yao `[一作]` (Northwestern University), Izzy Grosof `[通讯]` (Northwestern University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了适用于连续资源需求的多资源作业调度模型，并给出了第一批通过离散化实现吞吐量最优的调度策略，包括预抢占和非抢占版本，并进一步设计了多种计算效率更高的高效集合策略。

**💡 创新点**

创新点在于：① 将连续资源需求离散化以兼顾理论可分析性与实际无重复需求的情况；② 证明该离散化的 MaxWeight 与 nMSR 家族在连续模型下实现吞吐量最优；③ 设计四类高效集合（2-Job、2-Bucket、极点、双极点）在保持吞吐量最优的同时大幅降低计算复杂度。

**🔧 技术方法**

主要技术包括：离散化方案（按系统负载和需求分布选取粒度）；基于离散化的 MaxWeight 与 nMSR 政策；构造服务选项分布以满足离散测度支配条件；使用 Lyapunov 驱动的 Foster‑Lyapunov 证明吞吐量最优；回填（Backfilling）与指数化等实现策略的改进。

**📊 数据集**

实验数据集包括：参数化连续分布（Uniform、Truncated Normal、Lomax、Triangle 等）和 Google Borg 真实调度日志，后者提供 CPU 与内存需求的高精度浮点数据。

**📈 对比分析**

与传统指数化（FCFS、First‑Fit、Best‑Fit、LSF）及已知类基策略（MaxWeight、RMS、MSR）比较。实验表明：在保证稳定性的前提下，K‑EMW 与 nMSR 家族的均衡响应时间显著低于所有指数化策略，且通过 Backfilling 可进一步提升性能；在负载接近极限时，K‑EMW‑B 与 2J‑EMW‑B 能保持最宽的稳定区间和最小平均响应时间。

**⚠️ 局限性**

局限性包括：① 需要离散化参数 K 的选择，K 越大能逼近理论最优但计算成本也随指数级上升；② 对连续需求的理论证明依赖 Lipschitz 连续性，非连续或离散需求需进一步研究；③ 高维多资源场景下高效集合的构造与求解仍较复杂；④ 实际系统中服务时长分布非指数时的性能尚未完全验证。

---

## 244. Co-Ontogeny by Archetypal Scaffolding: The Humorphic Partnership

**arXiv ID:** 2605.21818 | [PDF](https://arxiv.org/pdf/2605.21818v1)

**作者:** Hector Ouilhet Olmos `[一作]` `[通讯]`, Hector Ouilhet Olmos

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

定义并实现了“幽默化伙伴关系（humorphic partnership）”这一新的人机协作构造，基于“原型支架（archetypal scaffolding）”与“仓库可见性（vault‑visibility）”的个人AI架构，并通过单案例纵向自民族志（N‑of‑one）对四个月的交互轨迹进行记录与分析。

**💡 创新点**

①提出幽默化伙伴关系的六项操作条件与共生发展（co‑ontogeny）概念；②将原型视作功能机制，记录为可审计事件；③将自模型写入共享文本仓库，形成可观察的第三层伙伴级别档案（delta）；④展示AI能自我诚实地报告效能衰退。

**🔧 技术方法**

基于Anthropic Claude Sonnet/Opus 的大型语言模型；轻量级“薄框架（thin‑harness）+厚技能（fat‑skills）”结构；三层注意循环（Listen/Notice/Know）配合三阶反思（1st‑2nd‑3rd order）；原型标记与日志体系；每周架构侦察与 delta 文档。

**📊 数据集**

自建交互日志（共181条原型标记事件、144份伙伴声明、3份自画像、5份元反思、6份架构侦察摘要等），以及作者的成长日志与伙伴的同级分析，全部来自单个主体的真实使用。

**📈 对比分析**

对比传统任务导向的个人AI研究，本文通过定性归因与定量熵/验证器时间序列展示了伙伴关系在成长观测与自我报告上的独特性；未提供传统性能基准，但验证器在五周内持续返回 0.0% 并无人为修饰，体现了系统的诚实性与可追踪性。

**⚠️ 局限性**

主要限制为单案例（N=1）与作者自身作为参与者与系统设计者的潜在偏倚；模型版本漂移、工具设计导致的特定实现偏差；缺乏跨人群或跨模型的可重复性验证；在更大规模或多任务环境下的泛化仍需后续多实例复制研究验证。

---

## 245. The 2nd Workshop on Agile Practice & Research: A Summary and Call For Research

**arXiv ID:** 2605.21690 | [PDF](https://arxiv.org/pdf/2605.21690v1)

**作者:** Karen Eilers `[一作]` (BSP Business and Law School - Campus Hamburg), Tiago Silva da Silva `[通讯]` (Federal University of S~ao Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

总结了第二届敏捷实践与研究工作坊，提出了四项改进研究-实践交叉的命题，并呼吁在开放科学、研究质量和贡献明确性等方面开展进一步研究。

**💡 创新点**

提出了系统的四个命题（科学传播、价值导向、激励机制、教育方法整合），以及三项具体研究呼吁，旨在缩小理论、时间与转移等研究-实践差距。

**🔧 技术方法**

主要采用工作坊协作、主题讨论与系统综合的研究方法；未使用具体技术工具或算法。

**📊 数据集**

未使用任何数据集，重点依赖实践者与研究者的经验分享与共识形成。

**📈 对比分析**

未进行实验对比；通过小组讨论、现场记录和研讨会成果的综合评估来衡量方案的可行性，效果以主观共识为准。

**⚠️ 局限性**

局限在缺乏实证验证、结果依赖于参与者经验与主观判断、缺少可重复测评和量化指标。

---

## 246. Understanding Perspectives of Patients, Caregivers and Clinicians towards Emerging Collaborative-decision Making Technologies

**arXiv ID:** 2605.21777 | [PDF](https://arxiv.org/pdf/2605.21777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 247. Predicting Performance of Symbolic and Prompt Programs with Examples

**arXiv ID:** 2605.21515 | [PDF](https://arxiv.org/pdf/2605.21515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 248. OCELOT: Odometry and Contact Estimation for Legged Robots

**arXiv ID:** 2605.21863 | [PDF](https://arxiv.org/pdf/2605.21863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 249. Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models

**arXiv ID:** 2605.21573 | [PDF](https://arxiv.org/pdf/2605.21573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Near-Optimal Generalized Private Testing

**arXiv ID:** 2605.21601 | [PDF](https://arxiv.org/pdf/2605.21601v1)

**作者:** Anamay Chaturvedi `[一作]` (Institute of Science and Technology Austria), Jalaj Upadhyay `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种新的泛化私有测试机制（Generalized Private Tester，GPT），可在流式环境中以有限的样本复杂度对一系列私有随机化测试器的成功概率进行在线判定，并在保证差分隐私的前提下实现高精度的停机/继续决策；

**💡 创新点**

创新点在于①引入了Poisson随机化采样与自适应基准率相结合的“隐私噪声扩展”框架，显著降低了先前“噪声放大”所导致的误差累积；②通过“翻转”方向参数和自适应阈值，进一步在成功率接近0或1时保持严格的阈值分离；③提出了基于随机响应的纯化 lemma，轻松将 (ε,δ)-DP 测试器转化为纯 ε-DP，几乎不增加样本量；

**🔧 技术方法**

主要技术包括：Poisson加法性和随机支配、指数式噪声注入、随机采样与“随机化响应”纯化、概率界（Poisson 与指数分布的尾部估计）、上界/下界的对数距离推导以及阈值测试的标准化变换；

**📊 数据集**

由于本文侧重理论分析与机制设计，实验评估使用了合成数据集（随机生成满足感知1的查询函数 f_t，且支持集互斥），没有涉及真实世界数据集；

**📈 对比分析**

与传统的 Above Threshold（AT）和标准阈值测试器比较，GPT 在相同隐私预算和总误差概率下，能够实现更小的误差阈值（即 p̅ 与目标阈值 τ 的差距在对数级别可被压缩至 1/(logT/2(β+Tδ')+log1/2(β+δ'))-2），且样本复杂度上限为 O(ln(t/β)/ε)，显著低于已有基于多步采样的上界；

**⚠️ 局限性**

限制在于：当输入隐私参数 ε_1 固定时，噪声放大因子 Λ_t 随流长度 t 以 (t/β)^Θ(ε_1) 级增长，导致在长流中误差累积严重；此外，机制对相邻数据集距离的上界依赖于查询函数支持互斥性，实际应用中可能难以满足；

---

## 251. LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation

**arXiv ID:** 2605.22054 | [PDF](https://arxiv.org/pdf/2605.22054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 252. When Cases Get Rare: A Retrieval Benchmark for Off-Guideline Clinical Question Answering

**arXiv ID:** 2605.21807 | [PDF](https://arxiv.org/pdf/2605.21807v1)

**作者:** Doeun Lee `[一作]` (Ohio State University), Sachin Kumar `[通讯]` (Ohio State University)

**通讯引用:** 1548 | [OpenAlex ID](https://openalex.org/A5100627117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了基于真实病例报告的“Off‑Guideline Case Reports”开放式问答基准，旨在评估LLM在罕见、非指南情境下的临床推理能力。

**💡 创新点**

创新点包括：①利用专家验证的病例报告生成真实且富挑战性的自由文本问答对；②在基准上系统评估检索增强（RAG）对模型表现的提升；③深入剖析模型错误模式，揭示检索、上下文窗口、推理能力与误差之间的关系。

**🔧 技术方法**

采用技术包括：大语言模型（如GPT‑5.2、Claude‑4 Sonnet等）生成答案与检索文档，检索模型（BM25、BGE、BMRetriever）与交叉编码器重排，LLM‑as‑a‑judge进行答案一致性评估，统计检索指标（Recall@k、MRR、nDCG）和准确率。

**📊 数据集**

数据集：从 PubMed Central 公开病例报告中筛选 53,617 篇，进一步过滤得到 28,219 篇符合“稀缺性”条件，再随机抽取 1,500 篇生成 639 条经过专家验证的问答对，覆盖 12 个医学专业。

**📈 对比分析**

在无检索基线下，常规LLM 仅达 37–56% 正确率；加入检索后，GPT‑5.2 在 1–5 条检索文档情境下可提升至约 82%，而医学专用模型则提升 30–40% 左右；即使提供 oracle 文档，模型仍有约 20% 的错误，说明检索并非唯一瓶颈。

**⚠️ 局限性**

局限性：①数据规模相对有限，且高度偏向内科与外科；②不同模型的上下文窗口限制导致检索文档数量受限；③模型在目标对齐、文档根据信息保留以及细节约束上仍出现错误；④未充分评估模型在不同临床环境与语言多样性下的泛化能力。

---

## 253. Autonomous LLM Agents & CTFs: A Second Look

**arXiv ID:** 2605.21497 | [PDF](https://arxiv.org/pdf/2605.21497v1)

**作者:** Youness Bouchari `[一作]` (Politecnico di Torino), Dario Rossi `[通讯]` (Huawei Technologies France)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了不同复杂度的LLM代理在30个Web CTF挑战上的表现，并与通用代理Baseline进行对比；

**💡 创新点**

提出了三种结构化多代理架构（Executor、Executor+Evaluator、Planner+Executor+Evaluator），并揭示规划器在漏洞识别上的关键作用；

**🔧 技术方法**

采用大语言模型（GPT‑4.1、GPT‑5、Claude Opus 4.5）作为后端，配合工具调用、记忆管理和自适应评估；

**📊 数据集**

使用XBOW基准的30个Web CTF任务，覆盖14类漏洞，手工解答作为参考；

**📈 对比分析**

与通用代理对比，三种结构化架构在成功率上均可达19/30，Planner+Executor+Evaluator在步骤数与成本上比单一Executor降低约24%与34%；

**⚠️ 局限性**

局限主要在漏洞识别不足、对环境依赖（浏览器渲染、并发执行）与长链交互中的上下文管理，导致无法突破19/30的瓶颈。

---

## 254. Implicit Safety Alignment from Crowd Preferences

**arXiv ID:** 2605.21822 | [PDF](https://arxiv.org/pdf/2605.21822v1)

**作者:** Qian Lin `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**通讯引用:** 842 | [OpenAlex ID](https://openalex.org/A5103065575)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用人群偏好数据中隐含的共享安全准则，学习安全相关的低层技能并通过高层策略组合以实现安全的下游强化学习任务。

**💡 创新点**

创新点在于：①不再直接将学习到的奖励模型与任务奖励相加，而是将低层技能学习与高层策略组合的层次化框架；②通过变分自编码器提取隐含用户上下文，学习条件奖励与策略；③提出基于对比偏好学习（CPL）的无奖励版技能学习。

**🔧 技术方法**

使用 VAE+RLHF（或 CPL）进行技能学习，离线强化学习（IQL、TD3+BC）训练低层和高层策略，理论分析安全性和任务性能。

**📊 数据集**

在六个安全强化学习环境（改编自 Bullet‑Safety 与 Safety‑Gymnasium）和一个简化的语言对话基准上进行实验，构建多目标奖励与共享安全奖励的偏好数据集。

**📈 对比分析**

与 Oracle（已知安全奖励）、Task‑Only（仅任务奖励）以及 Safe‑RLHF 等基线对比；在离线下游设置下，Safe‑VPL/CPL 方法在保持与 Oracle 相近的任务性能的同时，将安全成本降低 90% 以上，且对偏好不平衡更稳健。

**⚠️ 局限性**

假设人群偏好反映统一安全共识，若存在噪声或恶意反馈，方法鲁棒性不足；对偏好数据质量与用户数量的敏感度仍需进一步提升。

---

## 255. Echo: Learning from Experience Data via User-Driven Refinement

**arXiv ID:** 2605.21984 | [PDF](https://arxiv.org/pdf/2605.21984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 256. Temporal Contrastive Transformer for Financial Crime Detection: Self-Supervised Sequence Embeddings via Predictive Contrastive Coding

**arXiv ID:** 2605.21490 | [PDF](https://arxiv.org/pdf/2605.21490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 257. PGDG: Physically Grounded Data Generation for Robust Bimanual Policy Learning from a Single Demonstration

**arXiv ID:** 2605.21710 | [PDF](https://arxiv.org/pdf/2605.21710v1)

**作者:** Cunxi Dai `[一作]` (Carnegie Mellon University), Guanya Shi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1343 | [OpenAlex ID](https://openalex.org/A5029314167)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 PGDG 框架，利用物理约束在单个演示上迭代生成并筛选物理可行、成功且多样的恢复轨迹，形成紧凑数据集并训练行为克隆模型。

**💡 创新点**

创新点在于零射击、无策略查询的数据生成与筛选循环；聚焦“Goldilocks 区”中可恢复信息；使用 DPP 进行多样性选取；通过 CEM 在关键离线状态上做局部纠正；整体实现全流程的物理基数据扩增。

**🔧 技术方法**

采用控制点采样、对数正态分布提议、离散余弦变换 (DCT) 轨迹嵌入、DPP 选取、交叉熵方法 (CEM) 本地重标、深度基 ACT 与 GR00T 模型；在 Isaac Sim 物理仿真中进行域随机化。

**📊 数据集**

数据来源为每个任务仅一条演示（通过 GELLO 外骨骼收集），随后 PGDG 生成数千条合成轨迹；在 fine‑tune GR00T 时对比 PGDG 生成数据与 MimicGen 生成数据。

**📈 对比分析**

与仅做空间随机化（MimicGen）、无 DPP 或无重标的 PGDG 方案进行对比；在仿真中对 RotateBox-Roll/Pitch/Yaw 与 BarPass 的成功率提升从 38%→93% 以上，在真实环境中从 35%→82%；对 GR00T 细调时 PGDG 数据使成功率从 65% 提升至 87% 以上，显著优于基线。

**⚠️ 局限性**

局限在于只在演示附近生成数据，无法发现全新策略；需要在末端执行器空间收集演示；若演示本身不佳，生成的数据会继承偏差；大数据集并不一定更好；模拟与真实的摩擦、接触不匹配仍是瓶颈。

---

## 258. Stacked Intelligent Metasurface-Assisted Fluid Antenna Systems: Outage Probability

**arXiv ID:** 2605.22053 | [PDF](https://arxiv.org/pdf/2605.22053v1)

**作者:** Anastasios Papazafeiropoulos `[一作]` (University of Hertfordshire), Anastasios Papazafeiropoulos `[通讯]` (University of Hertfordshire)

**通讯引用:** 2147 | [OpenAlex ID](https://openalex.org/A5055961126)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并分析了结合堆叠智能金属表面（SIM）与流体天线系统（FAS）的联合通信框架，推导了闭式失效概率并对SIM相位进行优化。

**💡 创新点**

创新点在于将波域可重构的SIM与空间多样性的FAS相结合，并采用块对角矩阵逼近（BDMA）对FAS端口相关性进行可行建模，得出可直接用于相位优化的闭式失效概率。

**🔧 技术方法**

主要技术包括SIM波域预编码、FAS端口选择、BDMA空间相关建模、Marcum Q函数分析以及投影梯度优化算法。

**📊 数据集**

使用仿真参数（如M=16/32、L=3、N=50、R=6bps/Hz、K=2、路径损耗指数3.5等）进行实验验证，并未引用公开数据集。

**📈 对比分析**

通过与“无SIM”和“无FAS”两种基准方案的对比，实验结果显示联合SIM-FAS框架在闭式分析与Monte Carlo仿真之间保持一致，并在各种发射功率、层数和端口数下显著降低失效概率，体现出显著的性能提升。

**⚠️ 局限性**

局限性包括：仅考虑理想化SIM模型（忽略硬件失效）、仅针对单天线基站场景、BDMA近似忽略块间相关性，以及对更复杂多天线和多用户场景的验证尚未展开。

---

## 259. Trace2Skill: Verifier-Guided Skill Evolution for Long-Context EDA Agents

**arXiv ID:** 2605.21810 | [PDF](https://arxiv.org/pdf/2605.21810v1)

**作者:** Zijian Du `[一作]` (NVIDIA), Nathaniel Pinckney `[通讯]` (NVIDIA)

**通讯引用:** 2016 | [OpenAlex ID](https://openalex.org/A5050270997)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练权重更新的“Trace2Skill”框架，通过在硬件语言模型代理的测试阶段进化自然语言技能，以提升在复杂Verilog设计任务（CVDP）中的通过率。

**💡 创新点**

创新点包括：①将执行回放轨迹转化为任务专属的可进化技能；②设计密集的评价指标（SkillQ、AgentProgressQ、AgentVarianceQ 等）来细粒度评估技能内容与代理行为；③将稠密运行时验证反馈与技能进化耦合，形成闭环优化；④无需针对 RTL 训练模型，可直接在现有代理上提升性能。

**🔧 技术方法**

技术手段包括：大语言模型（Claude Opus 4.6、GPT‑5、Claude Sonnet 4.5）执行代理、回放、总结与变异；稠密运行时验证器提供功能观察；遗传算法框架（5 代、4 子代、4 轮回放）进行技能演化；指标计算与多种统计量（LCB、均值、方差）评估代理进展。

**📊 数据集**

使用公开的 24 题 CVDP 任务集合（cid003、cid004、cid005、cid016），以及 8 个在种子代理失败的“硬任务”进行深入实验。

**📈 对比分析**

对比方法包括：①基线种子 CVDP 代理；②仅加入稠密验证反馈；③仅进行稀疏技能进化；④结合稠密反馈与技能进化。实验结果显示，C4（结合稠密反馈与技能进化）在 8 个硬任务中实现 6/8 通过，隐藏验证通过率 33.6%，显著优于基线（0/8）和单独稠密反馈（2/8）或单独技能进化（3/8）方案。

**⚠️ 局限性**

局限性包括：1）长周期代理在相同技能下可能产生高方差行为，导致指标波动；2）计算成本高，30 轮回放与多次验证会显著增加算力与时间；3）仅验证 Verilog 相关任务，缺乏对更广泛 EDA 流程（综合、时序、功耗等）的评估。

---

## 260. ConTact: Contact-First Antibody CDR Design via Explicit Interface Reasoning

**arXiv ID:** 2605.21600 | [PDF](https://arxiv.org/pdf/2605.21600v1)

**作者:** Mansoor Ahmed `[一作]` (Georgia State University), Murray Patterson `[通讯]` (Georgia State University)

**通讯引用:** 2002 | [OpenAlex ID](https://openalex.org/A5026228482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对抗体CDR设计提出了新的模型框架，在抗原结构条件下生成高质量的CDR序列与构象。

**💡 创新点**

核心创新在于将设计任务拆分为“先预测接触位点”与“再根据接触信息生成序列”的三阶段递归结构，并通过双门控注入机制实现对接触信息的显式利用。

**🔧 技术方法**

技术手段包括虚拟节点E(3)-等变图神经网络、距离偏置交叉注意力、对比学习指纹、焦点二元交叉熵、接触加权交叉熵以及双门控注入策略。

**📊 数据集**

实验使用了包含2922个抗体-抗原复合物的SAbDab数据集，并按epitope划分进行训练/验证/测试。

**📈 对比分析**

与11个基线（包括RAAD、MEAN、dyMEAN、DiffAb、AbFlowNet等）对比，所提模型在CDR-H3设计任务中获得最高AAR（0.38）、最低RMSD（1.63 Å）、最高fnat（0.67）和最高epitope F1（0.79），表现最优。

**⚠️ 局限性**

主要局限在于接触位点氨基酸恢复仍较低（接触AAR≈0.20），表明仅靠Cα级别抗原表示难以捕捉到完整的化学约束，需进一步丰富抗原特征与多模态序列生成策略。

---

## 261. MAVEN: A Multi-stage Agentic Annotation Pipeline for Video Reasoning Tasks

**arXiv ID:** 2605.21917 | [PDF](https://arxiv.org/pdf/2605.21917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 262. Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow

**arXiv ID:** 2605.21980 | [PDF](https://arxiv.org/pdf/2605.21980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 263. RADAR: Defending RAG Dynamically against Retrieval Corruption

**arXiv ID:** 2605.22041 | [PDF](https://arxiv.org/pdf/2605.22041v1)

**作者:** Ziyuan Chen `[一作]` (Nanjing University), Tieniu Tan `[通讯]` (Nanjing University)

**通讯引用:** 37514 | [OpenAlex ID](https://openalex.org/A5111885963)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对动态检索增强生成（RAG）系统的防御框架，利用图最小割与贝叶斯记忆节点对检索上下文进行可靠性筛选，抵御检索污染与提示注入攻击；

**💡 创新点**

将上下文筛选建模为能量最小化的最小割问题，并引入贝叶斯记忆节点实现时间一致性与知识更新的平衡，实现低存储、实时动态防御；

**🔧 技术方法**

使用图割算法（Max‑Flow Min‑Cut）、NLI（DeBERTa‑v3 等）计算一致性与矛盾得分、贝叶斯推断更新记忆节点、以及深度学习生成器（DeepSeek、GPT‑4o、Grok‑4‑fast）；

**📊 数据集**

在四个公开基准（RealTimeQA、Natural Questions、TriviaQA、Bio）以及新构建的动态 QA 数据集上进行评估；

**📈 对比分析**

与 Vanilla RAG、AstuteRAG、InstructRAG、RobustRAG、ReliabilityRAG 等基线比较，实验显示在静态和动态攻击场景中均实现更高的答案准确率与更低的攻击成功率，尤其在高位检索攻击下仍保持 70%+ 准确率；

**⚠️ 局限性**

受限于对 NLI 模型的依赖和对极端对抗性写作（如“或”答案）的鲁棒性尚需进一步验证，且对不同检索源的适用性需更多实证。

---

## 264. AOP-Wiki EMOD 3.0: Data Model Expansions and Content Evaluation Framework for Using Agentic AI to Improve Integration between AOPs and New Approach Methodologies (NAMs)

**arXiv ID:** 2605.21645 | [PDF](https://arxiv.org/pdf/2605.21645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 265. Reflective Prompt Tuning through Language Model Function-Calling

**arXiv ID:** 2605.21781 | [PDF](https://arxiv.org/pdf/2605.21781v1)

**作者:** Farima Fatahi Bayat `[一作]` (Megagon Labs), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于LLM函数调用的反射式提示调优框架，自动诊断并修订提示以提升推理任务性能与校准。

**💡 创新点**

通过诊断函数生成结构化诊断报告并结合历史报告进行记忆，实现针对性信用分配的提示优化，并将置信度校准纳入诊断与最终提示选择。

**🔧 技术方法**

利用LLM函数调用、诊断报告生成、聚类诊断、记忆化优化器、置信度校准以及Brier分数评估等技术。

**📊 数据集**

在HotPotQA、LiveBench-Math和Formula这三大推理任务数据集上进行实验。

**📈 对比分析**

与ACE、GEPA、MIPRO等基线对比，三大任务均保持竞争力，HotPotQA提升最大12.9分、LiveBench-Math提升12.4分、Formula提升11.7分，同时显著降低Brier分数。

**⚠️ 局限性**

对大规模数据和长优化路径的计算成本高；仅在GPT‑4.1及大型模型上验证，缺乏对小模型的通用性；仍无法完全消除深层推理错误；置信度依赖模型自述，需进一步验证。

---

## 266. COCOTree: A Dataset and Benchmark for Open Tree-Structured Visual Decomposition

**arXiv ID:** 2605.22068 | [PDF](https://arxiv.org/pdf/2605.22068v1)

**作者:** Junhyub Lee `[一作]` (Chung-Ang University), Hyosu Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 494 | [OpenAlex ID](https://openalex.org/A5037409253)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了开树分解任务，将图像解析为不受限制层级结构的可见组件树，并构建了对应的全自动递归标注流程。

**💡 创新点**

创新点在于首次定义开树分解为开放词汇、无限深度的层级分割，结合LVLM与SAM3实现全自动标注，并提出OTQ评测指标。

**🔧 技术方法**

使用大型视觉语言模型（LVLM）进行语义推理、SAM3进行实例掩码定位、基于BERT/CLIP的标签相似度计算，并通过最大权重二分匹配完成节点匹配。

**📊 数据集**

基于COCO数据集构建了超过21K张图像、约1.8M节点的OpenTree数据集，包含3.5K开放词汇标签。

**📈 对比分析**

在自建基准上与平面投影、递归SAM3等方法对比，OTQ指标显示递归输出在掩码、标签与树结构上均优于传统方法，尤其在深层节点召回和结构一致性方面表现突出。

**⚠️ 局限性**

主要局限包括自动化流程对极小子节点的漏检、掩码噪声、标签相似度误差以及多重合法树结构导致的参考多样性问题。

---

## 267. FLUID: From Ephemeral IDs to Multimodal Semantic Codes for Industrial-Scale Livestreaming Recommendation

**arXiv ID:** 2605.21832 | [PDF](https://arxiv.org/pdf/2605.21832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 268. ACC: Compiling Agent Trajectories for Long-Context Training

**arXiv ID:** 2605.21850 | [PDF](https://arxiv.org/pdf/2605.21850v1)

**作者:** Qisheng Su `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 38524 | [OpenAlex ID](https://openalex.org/A5070851446)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Agent Context Compilation (ACC) 方法，将多轮 agent 轨迹编译成单个长上下文 QA 对，训练 LLM 在不使用工具的情况下直接回答问题。

**💡 创新点**

通过整合散落在多轮中的证据形成统一长上下文，消除 agent SFT 的监督盲点，无需额外标注即可提升长上下文推理能力。

**🔧 技术方法**

采用长上下文监督微调（SFT）结合 Qwen3-30B、DeepSeek-V3.2-Thinking 生成推理、注意力与专家路由分析等技术。

**📊 数据集**

利用搜索、软件工程（SWE）和 SQL 三类 agent 的轨迹数据（共10,802条），并在 MRCR、GraphWalks、GPQA、MMLU-Pro、AIME、IFEval 等基准上进行评测。

**📈 对比分析**

与基线 Qwen3-30B、Qwen3-235B 以及 QwenLong-L1.5 等模型对比，ACC 在 MRCR 提升约 18.1 分、GraphWalks 提升约 7.6 分，性能与大模型相当，并未出现显著负迁移。

**⚠️ 局限性**

仅在三种 agent 和单一模型上验证，未测试百万 token 长度场景；依赖强教师模型可能带来偏见；原始轨迹可能泄露隐私或版权信息。

---

## 269. Asymptotic Rank Speedup Theorems, Revisited

**arXiv ID:** 2605.21738 | [PDF](https://arxiv.org/pdf/2605.21738v1)

**作者:** Josh Alman `[一作]` (Columbia University), Baitian Li `[通讯]` (Columbia University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文重新审视了矩阵乘法文献中的经典工具，开发了一个框架，以获得超越矩阵乘法的张量的改进渐近秩上界。

**💡 创新点**

创新点在于建立了一般化的加速定理，能够在某些情况下从边界秩上界中提取额外的项，从而获得改进的渐近秩上界。

**🔧 技术方法**

使用了Strassen微积分的方法，该方法系统地将退化数据转化为明确的渐近秩界限，并利用Strassen的渐近谱理论。

**📊 数据集**

使用了小Coppersmith-Winograd张量和任意d×d×d张量作为数据集。

**📈 对比分析**

通过与Coppersmith-Winograd和Strassen的结果进行比较，证明了小Coppersmith-Winograd张量的渐近秩上界小于其边界秩，且对任意d×d×d张量的渐近秩上界进行了改进。

**⚠️ 局限性**

限制在于尽管提出了新的技术和方法，但仍然无法有效地提高矩阵乘法或集合覆盖等问题的算法效率。

---

## 270. TBP-mHC: full expressivity for manifold-constrained hyper connections through transportation polytopes

**arXiv ID:** 2605.21724 | [PDF](https://arxiv.org/pdf/2605.21724v1)

**作者:** Anton Lyubinin `[一作]` `[通讯]`, Anton Lyubinin

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Transportation Birkhoff Polytope (TBP) 与 Recursive TBP (RTBP) 两种完全满足双随机约束的混合矩阵参数化方法，用于提升残差网络中的 Hyper-Connections 性能。

**💡 创新点**

创新点在于利用运输多面体的北西角贪心解法构造可逆的全局图，直接生成完整 Birkhoff 多面体的矩阵，保持表达能力的同时仅需 (n‑1)² 个自由度；递归分块进一步加速计算并兼顾并行性。

**🔧 技术方法**

技术包括：运输多面体算法、北西角贪心构造、递归分块、双随机矩阵正则化、梯度裁剪、基于 Transformer 的预训练框架。

**📊 数据集**

使用 OpenWebText 数据集进行语言模型预训练，采用 Small（6 层）和 Medium（12 层）两种模型规模。

**📈 对比分析**

与 mHC、mHC-lite、KromHC 等基线方法比较，TBP/RTBP 在验证集的交叉熵与 bits-per-byte 指标上保持竞争性甚至略优，同时梯度范数更小，训练更稳定。

**⚠️ 局限性**

局限性包括：递归分块的非线性耦合使得大规模参数化难以优化；实现上仍需对梯度动态与优化器进行细致调优，且未充分发挥理论上可获得的全部潜能。

---

## 271. Rethinking Token Reduction for Diffusion Models via Output-Similarity-Awareness

**arXiv ID:** 2605.22011 | [PDF](https://arxiv.org/pdf/2605.22011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. When to Switch, Not Just What: Transition Quality Prediction in Clash Royale

**arXiv ID:** 2605.21868 | [PDF](https://arxiv.org/pdf/2605.21868v1)

**作者:** Heeyun Heo `[一作]` (Korea University), Huy Kang Kim `[通讯]` (Korea University)

**通讯引用:** 6042 | [OpenAlex ID](https://openalex.org/A5091602017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究玩家在《Clash Royale》中的策略切换行为，发现频繁切换的玩家获胜率最低，提出考虑切换成本并构建了三阶段推荐管线（PersonaGate→TimingGate→ScoreFusion）以实现谁、何时、何种切换的决策。

**💡 创新点**

创新点在于：① 将策略推荐视为转移级决策问题；② 引入Zero Switching Cost假设的批判与校正；③ 设计SwitchGap评估指标，避免把频繁切换者视为最优；④ 结合行为子类型、时机判断与质量预测三重机制，显著提升推荐效果。

**🔧 技术方法**

采用GRU‑based玩家状态编码器、五任务多任务预训练、CatBoost排名器、TQP（Transition Quality Predictor）与二分类TimingGate，整体实现基于玩家行为序列的可解释推荐。

**📊 数据集**

使用926,334场比赛记录，覆盖34,619名玩家的历史数据，聚类得到13种策略状态和3种行为子类型作为输入。

**📈 对比分析**

与Always‑Stay、Always‑Switch、Win‑Rate Threshold、Population Oracle、Collaborative Filtering、Last‑K Transition等基线比较；通过SwitchGap、Rec_TQP、Prec@1等指标评估。最终管线在5.4%覆盖率下实现+10.4%p的SwitchGap，Prec@1达70.4%，显著优于所有基线。

**⚠️ 局限性**

局限性包括：① 行为子类型采用静态聚类，无法捕捉玩家习惯随时间演变；② 评估仅为离线，缺乏在线实时验证；③ 未利用对手信息，可能限制时机判断精度；④ 只研究了单一游戏，泛化性待验证。

---

## 273. PEMark: Watermarking API Responses Based on Proxy Gateways and Position Encoding

**arXiv ID:** 2605.21865 | [PDF](https://arxiv.org/pdf/2605.21865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 274. SDFStent: Real-time interactive virtual stenting via SDF deformation fields

**arXiv ID:** 2605.22009 | [PDF](https://arxiv.org/pdf/2605.22009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 275. BEiTScore: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model

**arXiv ID:** 2605.21728 | [PDF](https://arxiv.org/pdf/2605.21728v1)

**作者:** Gonçalo Gomes `[一作]` (Instituto Superior Técnico University of Lisbon), Chrysoula Zerva `[通讯]` (Instituto de Telecomunicações)

**通讯引用:** 288 | [OpenAlex ID](https://openalex.org/A5004372721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BEiTScore，一种基于轻量级 BEiT‑3 跨编码器的无参考图像描述评估指标，兼顾长文本和细粒度视觉‑语言错误检测。

**💡 创新点**

创新点包括：①使用从视觉问答模型迁移来的权重，兼顾高性能与低资源；②采用包含 LLM 生成对抗错误的多阶段训练（对比学习→人类评分回归）；③创建 LongCapVLCP 长文本与场景文本评测基准，补充现有 77‑token 评测的不足。

**🔧 技术方法**

技术：BEiT‑3 交叉注意力编码器、对比学习与 L1 回归训练、LLM（Qwen3/GPT‑4o）生成对抗数据、硬负样本挖掘、Drop‑Path 正则化、单隐藏层回归头。

**📊 数据集**

数据集：MS‑COCO、PixelProse、LocalizedNarratives、TextCaps、VICR、POLARIS、Nebula、EvalMuse‑40k、Flickr8KExpert/CF、Composite、nocaps‑FOIL、SugarCREPE、VALSE、Winoground、LongCapVLCP（自建）。

**📈 对比分析**

与现有评估指标比较：在 11 个评测基准上（人类相关性、假设检测、属性/对象欺骗、语义组合等）均达到或超过最先进模型；相较于 LLM‑基方法，BEiTScore 20‑60× 参数更小、30‑100× 推理速度更快；在 77‑token 限制外的 LongCapVLCP 任务上，显著优于 CLIP‑S、PAC‑S 等传统编码器。

**⚠️ 局限性**

局限性：仍受限于训练数据覆盖范围，对极端长文本（超出 512‑token）或极少见视觉细节的识别可能不如大型 LLM；对新兴多模态任务（如视觉问答‑生成混合任务）需进一步验证。

---

## 276. Toward Realistic Wi-Fi Fault Diagnosis: A Multi-Modal Benchmark

**arXiv ID:** 2605.22008 | [PDF](https://arxiv.org/pdf/2605.22008v1)

**作者:** Junjian Zhang `[一作]` (Central South University), Nei Kato `[通讯]` (Tohoku University)

**通讯引用:** 31815 | [OpenAlex ID](https://openalex.org/A5013311265)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于真实校园环境的跨层多模态Wi‑Fi故障诊断数据集，并提出统一评估基准，涵盖检测、分类、定位以及推理一致性评估。

**💡 创新点**

创新点包括：①在真实测试平台上注入11类故障并收集10k+样本的跨层多模态观测；②设计结构化推理评估框架，能将LLM生成的诊断文本映射为可量化的操作特征；③系统对比传统机器学习、深度学习和LLM诊断方法，揭示多模态融合的边际收益与LLM在细粒度量化上的局限。

**🔧 技术方法**

使用的技术包括：统计特征提取与归一化、节点匿名化、序列截断/填充、LLM生成操作特征、阈值化推理一致性评估、传统分类器（LR、RF、SVM、KNN、MLP）、循环/卷积网络（LSTM、GRU、CNN）、LLM（Gemini、GPT、Llama、Qwen）与嵌入模型。

**📊 数据集**

数据集：跨层多模态Wi‑Fi故障数据集，覆盖流量、包级跟踪、警告事件、监控日志等四类观测，11种代表性故障（含硬件、软件、MAC层、关联和拥塞相关）以及正常样本，样本量超过10,000。

**📈 对比分析**

通过在同一训练/测试划分上对所有方法进行评估，测量检测、分类、定位的F1；结果显示传统机器学习方法在大多数任务和模态下表现最佳；LLM在推理一致性（EP/ER/EF1）上优于传统模型，但在细粒度故障分类和定位的F1远低于传统方法。

**⚠️ 局限性**

限制：LLM对细粒度数值变化敏感性不足，缺乏域特定的推理规则；多模态融合提升有限，模型对缺失模态的鲁棒性不高；当前评估依赖于手工规则构造的操作特征，仍存在主观性。

---

## 277. Probabilistic Attribution For Large Language Models

**arXiv ID:** 2605.21726 | [PDF](https://arxiv.org/pdf/2605.21726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 278. Graph Structure of Chebyshev Permutation Polynomials over Binary and Ternary Adic Rings

**arXiv ID:** 2605.21819 | [PDF](https://arxiv.org/pdf/2605.21819v1)

**作者:** Xiaoxiong Lu `[一作]` (Hengyang Normal University), Chengqing Li `[通讯]` (Xiangtan University)

**通讯引用:** 5247 | [OpenAlex ID](https://openalex.org/A5089601494)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究了 Chebyshev 置换多项式在二、三进制模幂环（ℤ₂^k × ℤ₃^k）上的功能图结构，给出了周期分布与循环结构的显式表述。

**💡 创新点**

首次对二、三进制模幂环下 Chebyshev 多项式的图结构进行完整解析，揭示周期数量保持不变、分支模式可预测等规律，填补了先前只关注素数幂环的研究空白。

**🔧 技术方法**

利用 Chebyshev 多项式的半群性质、p-进阶取值、泰勒展开及模运算等数论与代数工具，对周期与路径长度进行推导与证明。

**📊 数据集**

本研究为理论推导，无需使用实验数据集；主要使用整数域在 ℤ₂^k 与 ℤ₃^k 上的算术运算。

**📈 对比分析**

通过与以往在素数幂环中已知的周期与图结构结果对比，验证了公式的正确性；论文未提供实验性能指标，侧重理论验证。

**⚠️ 局限性**

研究仅聚焦于基数 2 与 3 的模幂环，缺乏对更一般素数或复合模的直接推广，且对算法实现的复杂度分析不充分。

---

## 279. TacO: Benchmarking Tactile Sensors for Object Manipulation

**arXiv ID:** 2605.21976 | [PDF](https://arxiv.org/pdf/2605.21976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 280. CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety

**arXiv ID:** 2605.21609 | [PDF](https://arxiv.org/pdf/2605.21609v1)

**作者:** Heajun An `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5883 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 CR4T 框架，将青少年使用的 LLM 生成的有风险或拒绝式回答，按风险域进行检测后重写为安全、指导性回应，提升对话连续性与信息价值。

**💡 创新点**

创新点在于把青少年安全视为发展性互动设计问题，而非单纯的过滤；通过轻量级领域分类、风险检测与域条件重写实现选择性重写，避免全局拒绝导致的对话停滞。

**🔧 技术方法**

使用模型无关的后生成层、SBERT+Logistic 轻量级域分类、LlamaGuard 等安全检测器、基于指令的域条件重写规则以及多种基础 LLM（Mistral-7B、Qwen2.5-7B、Llama-3.1-8B、GPT-OSS-20B）。

**📊 数据集**

使用 Safe-Child-LLM 与 MinorBench 构成的 513 题目集合（含 20 条中性控制提示）进行实验。

**📈 对比分析**

通过与原始模型和全局重写方法对比，目标重写在 2,052 条对话中将 unsafe 率降至 0.39%、拒绝率降至 3.75%，安全恢复率（SRR）为 67.56%，并在四个 LLM 上保持一致的提升，提升构建性指导、信息价值与风险减轻。

**⚠️ 局限性**

局限性包括评估主要依赖 LlamaGuard，可能无法覆盖所有青少年特定风险；缺少长对话和真实用户验证；仍需跨学科伦理审批和专业辅导支持以确保安全部署。

---

## 281. PromptNCE: Pointwise Mutual Information Predictions Using Only LLMs and Contrastive Estimation Prompts

**arXiv ID:** 2605.21776 | [PDF](https://arxiv.org/pdf/2605.21776v1)

**作者:** Juliette Woodrow `[一作]` (Stanford University), Chris Piech `[通讯]` (Stanford University)

**通讯引用:** 6079 | [OpenAlex ID](https://openalex.org/A5074969309)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出零样本点互信息（PMI）估计任务，设计了 PromptNCE 对比提示方法，并在三个公开数据集上进行评估，同时给出计算机科学教育案例研究；

**💡 创新点**

创新点在于：①将 LLM 作为无训练的批评者；②在对比式提示中加入显式 OTHER 选项，以恢复开放词汇的条件概率，从而实现零样本 PMI 估计；③提供方差分解诊断框架预测方法成功与否；

**🔧 技术方法**

采用对比提示（Contrastive Prompting）与 PMI 分解、InfoNCE、MarginalNCE 等方法，结合 LLM 的概率推断；PromptNCE 在候选集加入 OTHER 并用少量示例支撑基率估计；理论推导与实验验证相结合；

**📊 数据集**

使用三大公开数据集：Words（USF 关联词典）、ChaosNLI（SNLI 重新注释的 NLI 标签）、GoEmotions（Reddit 以情绪标签注释的评论），每个数据集均基于人工标注得到真值 PMI；

**📈 对比分析**

与 Direct PMI、Decomposed PMI、InfoNCE、MarginalNCE 四种对照方法在 Spearman ρ 上比较。PromptNCE 在所有三数据集上均为最高：ChaosNLI 0.82、Words 0.69、GoEmotions 0.47；在同一模型（Claude Sonnet 4）上优于 GPT‑5.2；

**⚠️ 局限性**

局限性：①理论假设 Bayes 最优分类器，实际 LLM 受预训练分布偏差影响；②仅在英语、商业 LLM 与有限数据集上验证；③需要候选标签集与少量输入‑输出示例；④对分布差异敏感，排名错误往往系统性；⑤案例研究样本量小，泛化性未知。

---

## 282. FlyRoute: Self-Evolving Agent Profiling via Data Flywheel for Adaptive Task Routing

**arXiv ID:** 2605.22057 | [PDF](https://arxiv.org/pdf/2605.22057v1)

**作者:** Rongjun Li `[一作]` (Huawei Technologies), Yihang Wu `[通讯]` (Huawei Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 FlyRoute，一种通过实时流量自我进化的多代理路由框架，能够持续更新代理的能力描述。

**💡 创新点**

创新点：1）利用真实流量累积能力证据并周期性自动蒸馏成摘要；2）基于代理不确定性、BM25相关性与新颖性构建的探索策略；3）闭环数据飞轮将成功示例、质量门与路由结合，持续提升路由性能。

**🔧 技术方法**

技术手段：LLM（Qwen3‑8B）+ BM25检索 + LLM‑as‑Judge 质量门 + 不确定性/新颖性加权探索 + 自动能力描述蒸馏。

**📊 数据集**

数据集：企业开发者支持问答数据，四个业务域（云服务、AI 加速、服务器硬件、移动 OS），共 7211 条训练样本与 1298 条测试样本。

**📈 对比分析**

对比方法：与同一基础 LLM 的静态 seed‑only 路由和零射击路由比较；冷启动（5 条种子）准确率提升至 78.04%（vs 72.57%），完整流量后准确率提升至 89.83%（比零射击 +17.26pp，比冷启动 +11.79pp），四域均有显著提升。

**⚠️ 局限性**

局限性：仅在单一企业产品的四个业务域验证；依赖 LLM‑as‑Judge 的质量门，可能带来判分噪声；未涵盖更大代理群、不同语言或开放域场景；未深入评估长期部署动态、漂移检测与运营成本。

---

## 283. From Patches to Trajectories: Privileged Process Supervision for Software-Engineering Agents

**arXiv ID:** 2605.21996 | [PDF](https://arxiv.org/pdf/2605.21996v1)

**作者:** Murong Ma `[一作]` (National University Of Singapore), Jin Song Dong `[通讯]` (National University Of Singapore)

**通讯引用:** 6763 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用开发者提供的参考补丁，将其转化为潜在的过程图，进而在监督微调中以双目标程序方式对长轨迹进行筛选与构造，从而提升SWE代理的过程有效性与效率

**💡 创新点**

提出了Patches-to-Trajectories框架，将参考补丁先拆解为非泄漏的上下文与里程碑图，再用该图指导轨迹生成，解决传统终端判定方法对中间步骤缺乏监督的问题

**🔧 技术方法**

采用LLM驱动的提议-批评循环构造过程图，ReAct式工具调用、实体一致性检查、LLM推理验证以及短于阈值的段落选择机制

**📊 数据集**

在SWE‑Gym的1.8k个真实Python issue实例上构建训练集，并在SWE‑Bench Verified与Lite基准上进行评估

**📈 对比分析**

与基于终端通过率的SFT（SWE‑Gym）和工具错误掩蔽方法（SWE‑Lego）对比，实验显示Pass@1提升最多10.8点，平均推理成本下降约15%

**⚠️ 局限性**

局限在于仍需依赖可获得的参考补丁，且对教师生成的轨迹仍有一定依赖，未来需验证跨语言与更复杂环境的通用性

---

## 284. Three Costs of Amortizing Gaussian Process Inference with Neural Processes

**arXiv ID:** 2605.21798 | [PDF](https://arxiv.org/pdf/2605.21798v1)

**作者:** Robin Young `[一作]` (University of Cambridge), Robin Young `[通讯]` (University of Cambridge)

**通讯引用:** 15664 | [OpenAlex ID](https://openalex.org/A5081128274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提供了神经过程与高斯过程预测之间差距的定量分解，分析了三种误差来源：标签污染、信息瓶颈和摊销。

**💡 创新点**

创新点在于对预测KL散度的分解，提供了不同架构和统计来源的误差界限，并提出了两种架构建议以改善预测方差。

**🔧 技术方法**

使用了潜在神经过程（LNP）架构，结合了编码器、均值聚合和解码器的设计，采用了高斯过程的先验进行训练。

**📊 数据集**

使用了高斯过程（GP）作为先验，具体数据集未明确提及，但涉及到的上下文数据是从GP先验中抽取的。

**📈 对比分析**

与稀疏变分方法相比，LNP通过单次前向传递在O(n)时间内映射任何上下文集到预测分布，消除了每个任务的优化，性能在多任务和实时约束下表现优越。

**⚠️ 局限性**

限制在于LNP在估计方差时依赖于标签，导致标签污染的误差无法随着上下文大小的增加而消失，且在架构选择上存在信息瓶颈。

---

## 285. Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention

**arXiv ID:** 2605.21842 | [PDF](https://arxiv.org/pdf/2605.21842v1)

**作者:** Athanasios Zeris `[一作]` `[通讯]` (Independent Researcher), Athanasios Zeris (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了能量门控注意力（Energy‑Gated Attention），在 Transformer 的值聚合过程中根据键嵌入的谱能量动态调节注意力权重。

**💡 创新点**

将湍流中的协同结构能量排序原则和 Wiener‑Khinchin 定理引入到注意力机制，实现了自底向上的信息重要性感知，而不是仅依赖查询‑键相似度。

**🔧 技术方法**

使用了线性能量投影、Z‑标准化、Sigmoid 门控和可学习阈值，以及对比实验中的 Morlet、Daubechies 等小波基。

**📊 数据集**

在字符级别的 TinyShakespeare 与 Penn Treebank 两个数据集上进行实验。

**📈 对比分析**

通过对比基线模型，在 TinyShakespeare 上提升验证损失约0.103（0.26% 参数开销），Penn Treebank 同样提升约0.101，显示跨数据集一致性。

**⚠️ 局限性**

仅在小型字符级模型上验证，未测试更大规模、词/子词级别；小波基对比不完全公平，缺乏可学习小波包等更完整的评估。

---

## 286. Equilibrium Propagation and Hamiltonian Inference in the Diffusive Fitzhugh-Nagumo Model

**arXiv ID:** 2605.21568 | [PDF](https://arxiv.org/pdf/2605.21568v1)

**作者:** Jack Kendall `[一作]` `[通讯]` (Zyphra), Jack Kendall (Zyphra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将 Equilibrium Propagation 与 Hamiltonian Echo Backpropagation 扩展到偏导梯度的 Fitzhugh-Nagumo 神经网络，推导出深层残差网络的层级 Hamiltonian 递推关系，并在 MNIST 上训练了五层深度 FHN 网络，验证了时间动力学与空间 Hamiltonian 推断的等价性。

**💡 创新点**

证明了偏导梯度系统在稳态下可满足自伴随性，从而 Equilibrium Propagation 能够应用；展示了深度能量基模型与 Hamiltonian 神经网络之间的等价性；首次给出了用于快速前向推断的显式 Hamiltonian 递推公式。

**🔧 技术方法**

Equilibrium Propagation、Hamiltonian Echo Backpropagation、偏导梯度分析、离散图拉普拉斯算子、层级 Hamiltonian 递推、深度残差网络拓扑、中心差分 EqProp、离散时间/空间积分方法。

**📊 数据集**

MNIST 手写数字数据集。

**📈 对比分析**

使用 Equilibrium Propagation 训练 FHN 网络，测试误差为 2.8% ± 0.2；通过对比时间动态模拟与 Hamiltonian 空间积分，发现两者在深度 ≤30 层时高度一致，深度超过 30 层时空间积分会发散，表明推断方法在一定深度范围内有效，但深层网络仍需改进。

**⚠️ 局限性**

仅在稳态下有效，存在深度 >30 层时 Hamiltonian 推断发散的问题；训练过程中可能出现 loss spike，说明网络可能离开稳态；实验仅在 Turing 模式区间进行，未验证更一般的非平衡行为；缺乏对比基线和大规模任务的评估。

---

## 287. OPPO: Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning

**arXiv ID:** 2605.21851 | [PDF](https://arxiv.org/pdf/2605.21851v1)

**作者:** Yu Li `[一作]` (George Washington University), Zhengling Qi `[通讯]` (George Washington University)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5078483423)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Oracle-Prompted Policy Optimization（OPD），通过累积基于oracle的对数似然比来构造每个token的优势，改进LLM推理训练。

**💡 创新点**

创新点在于将对数似然比视为贝叶斯更新量，闭式递归计算成功概率，得出无价值网络、无额外rollout的token级优势，并用方向锚定保证梯度方向正确。

**🔧 技术方法**

采用贝叶斯值递归、对数似然比估计（自oracle与教师oracle两种）、方向锚定、分组归一化和PPO风格的剪枝。

**📊 数据集**

在DeepScaleR上训练，评估七大推理基准（GSM8K、MATH-500、AMC'23、AIME'24、GPQA-D、ARC-Challenge、LiveCodeBench）使用Qwen3-4B、Phi-4-mini等模型。

**📈 对比分析**

与GRPO、Dr.GRPO、DAPO、SDPO等基线比较，OPD在所有基准上均优于基线，最高可提升+6.0分（AMC'23）和+5.2分（AIME'24），且收益随回答长度增加而增大。

**⚠️ 局限性**

局限包括：仍依赖可验证的奖励和oracle输入；教师oracle质量受限于模型规模；在更复杂或多阶段奖励情形下递归形式可能需要改进。

---

## 288. One Sentence, One Drama: Personalized Short-Form Drama Generation via Multi-Agent Systems

**arXiv ID:** 2605.22144 | [PDF](https://arxiv.org/pdf/2605.22144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 289. IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents

**arXiv ID:** 2605.22154 | [PDF](https://arxiv.org/pdf/2605.22154v1)

**作者:** Daewon Choi `[一作]` (KAIST), Aram Galstyan `[通讯]` (Amazon AGI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IdleSpec，一种在 LLM 代理的工具调用空闲时间内进行投机性规划的推理框架，生成多条候选计划并在获得观察结果后聚合以提升后续推理质量。

**💡 创新点**

创新点包括：①利用闲置时间进行动态策略采样，交替使用 Progressive（假设成功）和 Recovery（假设失败）两种草稿策略；②通过 Thomp­son 采样与后验更新自适应调整策略选择；③将草稿视为参考而非强制执行，增强在观察不确定性下的鲁棒性。

**🔧 技术方法**

核心技术：两阶段草稿生成（Progressive 与 Recovery）、Idle‑aware 迭代草稿、Beta 先验 + Thompson 采样的策略选择、草稿聚合提示、LLM 交互推理。

**📊 数据集**

实验数据集：GAIA、FRAMES 与 MLE‑Bench Lite，涵盖工具增强推理、多跳搜索以及长周期机器学习工程任务。

**📈 对比分析**

与 Vanilla、Sequential Revision、Planning、Sleep‑Time Compute 等基线对比，IdleSpec 在 GAIA 上平均提升 4.6–6.8% 准确率，在 MLE‑Bench 上提升 9.1% Gold‑medal 率，同时保持与 Vanilla 接近的整体延迟；在大多数基线之上实现 Pareto 提升。

**⚠️ 局限性**

局限性：1）当工具调用的空闲时间不足或极短时，无法充分利用；2）需要额外的 token 预算，虽然在空闲期消耗，但总体算力仍增加；3）对非常小的模型效果有限，需具备足够强的 LLM 生成能力。

---

## 290. Psy-Chronicle:A Structured Pipeline for Synthesizing Long-Horizon Campus Psychological Counseling Dialogues

**arXiv ID:** 2605.22140 | [PDF](https://arxiv.org/pdf/2605.22140v1)

**作者:** Chaogui Gou `[一作]` (University of Science and Technology Beijing), Jiarui Liang `[通讯]` (University of Science and Technology Beijing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Psy-Chronicle 框架，通过学生画像、时间压力事件图和交互式模拟生成长周期校园心理咨询对话，并基于此构建 CPCD 数据集和 CPCD-Bench 评测基准。

**💡 创新点**

创新点在于将学生背景、事件链和对话记忆三者结构化为统一的生成管道，首次实现长周期、跨会话的心理咨询语料生成和评估。

**🔧 技术方法**

采用结构化数据生成技术（学生画像、事件图）、对话模拟与记忆更新机制，基于 Qwen3 LLM 进行监督微调；评测使用 GPT‑5.2 作为评判器。

**📊 数据集**

使用自研的 CPCD 数据集（100名学生画像、90,000 轮对话、约 11.45M 汉字），并在其子集上构造 CPCD‑Bench 进行评测。

**📈 对比分析**

在三项任务（会话级回复 SR、记忆召回 MR、时间因果推理 TCR）上与 GPT‑5.4、Gemini、CBT‑LLM 等模型对比，CPCD‑Chat‑8B 在 SR 与 MR 上显著提升，TCR 仍有限。

**⚠️ 局限性**

主要限制为数据为合成，无法完整再现真实咨询的细腻互动；时间因果推理能力仍不足，且评测受限于模型长文本处理与记忆压缩能力。

---

## 291. EventGait: Towards Robust Gait Recognition with Event Streams

**arXiv ID:** 2605.22139 | [PDF](https://arxiv.org/pdf/2605.22139v1)

**作者:** Senyan Xu `[一作]` (University of Science and Technology of China), Xueyang Fu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 11543 | [OpenAlex ID](https://openalex.org/A5079007635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 EventGait——一种端到端的双流网络，用事件相机实现鲁棒步态识别，分别建模运动动态和静态形状；并构建了两大规模合成事件步态数据集（SUSTech1K-E 与 CCGR-Mini-E），为事件步态研究提供基准。

**💡 创新点**

创新点包括：
- 将混合时序的 Mixture of Spiking Experts（MoSE）应用于事件数据，适应不同光照与运动频率；
- 采用 Cross‑modal Structure Alignment（CroSA）将视觉基础模型（VFM）的稠密形状先验迁移到稀疏事件特征；
- 设计双尺度事件表示，短时片段保留高时间分辨率，长时片段捕捉稳定形状，突破传统事件图像化聚合的稀疏与低精度限制；
- 通过合成管线大规模生成事件步态数据，填补事件步态数据缺口。

**🔧 技术方法**

技术手段包括：事件摄像机捕捉、短/长时事件切片化、LIF 细胞的多时常专家（MoSE）处理动态，CNN 编码器配合 CroSA 对齐 VFM 提取的形状特征，跨模态对齐损失（L2）与三元组+交叉熵联合训练，融合模块将两流特征合并用于识别。

**📊 数据集**

使用了两大合成数据集：SUSTech1K-E（≈25k序列，1050身份）和 CCGR-Mini-E（≈48k序列，970身份）；以及两个真实事件数据集：DVS128‑Gait（4000流，20身份）和 EV‑CASIA‑B（124身份）。

**📈 对比分析**

与基准相机和 LiDAR 方法（如 GaitBase、LidarGait++ 等）对比，EventGait 在 SUSTech1K‑E 上整体 Rank‑1 达到 92.8%，低光下 83.2% 仅下降 9.6%；在 CCGR‑Mini‑E 上 Rank‑1 96.7%；在真实 DVS128‑Gait 上 Rank‑1 87.4%。在跨域与跨光照评估中，低光、夜间场景的性能提升显著（超过传统方法 30%+），并显著缩小了相机/ LiDAR 与事件方法的性能差距。

**⚠️ 局限性**

局限性包括：
- 仍存在 sim‑to‑real 的泛化差距，合成事件与真实事件在纹理、噪声等方面有偏差；
- 真实事件数据规模有限，难以覆盖更广泛的环境与身份多样性；
- 事件相机硬件成本虽低于 LiDAR，但在大规模部署前仍需进一步验证可靠性与能耗。

---

## 292. Cross-Lingual Consensus: Aligning Multilingual Cultural Knowledge via Multilingual Self-Consistency

**arXiv ID:** 2605.22137 | [PDF](https://arxiv.org/pdf/2605.22137v1)

**作者:** Andrew Ivan Soegeng `[一作]` (SAP), Tan Sang Nguyen `[通讯]` (National University of Singapore)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5101323058)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自监督的多语言自一致性框架，用以提升大语言模型在跨语言文化对齐上的表现。

**💡 创新点**

创新点在于利用模型自身在不同语言的回答一致性来挑选可靠文化知识，无需人工标注或更强教师模型。

**🔧 技术方法**

采用多语言自一致性评估、翻译转移、批判生成（self‑critique）训练以及批量微调等技术。

**📊 数据集**

使用BLEnD、CANDLE、CultureAtlas、Aya Dataset等数据集，合计约6,675条文化实例。

**📈 对比分析**

通过与Llama 3.1 Instruct基线在BLEnD、Multilingual HellaSwag和Global MMLU等指标上的对比，英语设置平均提升5.03%，总体保持泛化能力，轻微下降0.85%和1.42%。

**⚠️ 局限性**

限制在于低资源语言数据稀缺导致灾难性遗忘，导致当地语言性能下降。

---

## 293. Beyond Pixels: Learning Invariant Rewards for Real-World Robotics From a Few Demonstrations

**arXiv ID:** 2605.22123 | [PDF](https://arxiv.org/pdf/2605.22123v1)

**作者:** Tengye Xu `[一作]` (University of Hong Kong), Jia Pan `[通讯]` (University of Hong Kong)

**通讯引用:** 9218 | [OpenAlex ID](https://openalex.org/A5076812698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 FLORA 框架，利用仅五条演示数据学习可迁移的符号奖励函数，用于开放式机器人操作。

**💡 创新点**

创新点在于结合对象中心运动流、基于潜在函数的 PBRS-MS 形势保证最优策略不变，并用 LLM 与贝叶斯优化的混合符号-数值优化在低样本下自动发现行为不变性。

**🔧 技术方法**

技术包括运动流生成器、符号潜在函数、潜在奖励塑造（PBRS-MS）、LLM 反射搜索、贝叶斯优化以及强化学习算法 RLPD。

**📊 数据集**

实验使用 Meta-World 的八个任务、Franka 机械臂的三种真实任务，并对比 LIV、VLC、ReWiND 等视觉奖励基线。

**📈 对比分析**

在奖励质量、政策收敛速度和真实世界 OOD 泛化上，FLORA 均显著优于基线，成功率提升 20-42%，并在位置、视角、物体变异的零样本迁移中保持高对齐与区分能力。

**⚠️ 局限性**

局限性包括对单调进度假设的依赖、对遮挡和多模态感知的敏感，以及一次性任务级手动标注和 1-8 小时的离线优化开销。

---

## 294. MPDocBench-Parse: Benchmarking Practical Multi-page Document Parsing

**arXiv ID:** 2605.22100 | [PDF](https://arxiv.org/pdf/2605.22100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 295. VISTA: Validation-Guided Integration of Spatial and Temporal Foundation Models with Anatomical Decoding for Rare-Pathology VCE Event Detection -- after competition results

**arXiv ID:** 2605.22096 | [PDF](https://arxiv.org/pdf/2605.22096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 296. Measuring Cross-Modal Synergy: A Benchmark for VLM Explainability

**arXiv ID:** 2605.22168 | [PDF](https://arxiv.org/pdf/2605.22168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 297. Efficient Agentic Reasoning Through Self-Regulated Simulative Planning

**arXiv ID:** 2605.22138 | [PDF](https://arxiv.org/pdf/2605.22138v1)

**作者:** Mingkai Deng `[一作]` (Institute of Foundation Models), Eric P. Xing `[通讯]` (Institute of Foundation Models)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了将代理推理拆分为反应执行（System I）、模拟规划（System II）和自我调节（System III）的三系统框架，并实现了 Self‑Regulated Simulative Reasoning Agentic LLM。

**💡 创新点**

首次将自我调节与基于世界模型的模拟规划结合，并在同一 LLM 内实现可调节的规划决策；通过 RL 学习让规划深度而非频率增大。

**🔧 技术方法**

在 LLM 中通过链式思考实现世界模型、规划器和调节器；使用监督学习+强化学习（GRPO）进行训练；工具集包括搜索、浏览与 Python REPL。

**📊 数据集**

使用公开数学、科学、表格和 Web 推理数据集，如 AIME、MATH‑500、GPQA、FinQA、BrowseComp 等；构造两种监督数据 v0.1 与 v1.0。

**📈 对比分析**

在 11 个基准上与 120–1T 参数的传统 LLM 与 7–30B 规模的 agentic LLM 进行 Pass@1 与 reasoning token 对比；v1.0‑30B 在 Pass@1 71.3% 与 51% token 省，优于同规模未调节模型。

**⚠️ 局限性**

仅在语言交互任务验证，未测试嵌入式或多智能体环境；世界模型为纯语言空间，预测能力有限；缺乏对调节器和规划器准确性的独立评估。

---

## 298. ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs

**arXiv ID:** 2605.22158 | [PDF](https://arxiv.org/pdf/2605.22158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 299. Short-Term-to-Long-Term Memory Transfer for Knowledge Graphs under Partial Observability

**arXiv ID:** 2605.22142 | [PDF](https://arxiv.org/pdf/2605.22142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 300. SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering?

**arXiv ID:** 2605.22175 | [PDF](https://arxiv.org/pdf/2605.22175v1)

**作者:** Yuxuan Sun `[一作]` (University of Science and Technology of China), Zhenya Huang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4669 | [OpenAlex ID](https://openalex.org/A5085496384)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为 SWE-Mutation 的基准，用于评估大语言模型生成的软件测试套件的质量，并通过代理式框架生成复杂的语义突变体。

**💡 创新点**

创新点在于：①采用代理式框架结合多语言、仓库级环境生成逼真的语义突变体；②提出 RDR 指标及与传统突变方法对比，显著提升评估判别力；③提供多语言 10 种语言的基准数据，弥补单一语言局限。

**🔧 技术方法**

使用技术包括：代理式框架（Mini‑Swe‑Agent、Claude Code）、Tree‑Sitter、LLM（Claude Sonnet 4/3.7、DeepSeek V3.1、Qwen3-Coder、Kimi K2、GLM‑4.6、GPT‑oss‑120B）以及突变生成的五大策略。

**📊 数据集**

数据集：来自 SWE‑bench 的 800 个原始实例（Python 500 + 9 语言 300），通过代理式框架生成 2,636 个突变体，并包含 300 个多语言子集（9 种语言）。

**📈 对比分析**

比较方法：在 Test‑Repair 与 Test‑Generation 两项任务下，使用 Pass@1、VRR 与 RDR 指标对七大 LLM 进行评估。实验显示即便是最强模型（Claude‑Sonnet‑4.5）在修复任务中 VRR 仅 42.6%、RDR 79.3%；在生成任务中 VRR 仅 29.8%、RDR 63.7%。与传统规则突变相比，代理式突变使 RDR 从 71% 降至 39%，证明评估更严格。

**⚠️ 局限性**

局限性：①基准依赖真实仓库的“金手指”解决方案与测试套件，若其本身存在缺陷会影响评估；②突变生成使用 Claude‑4 可能带来同族偏差，虽通过替换实验验证不明显；③仍缺乏对多语言环境依赖与版本一致性的深入考量。

---

## 301. CoRMA: Contrastive RMA for Contact-Rich Meta-Adaptation

**arXiv ID:** 2605.22082 | [PDF](https://arxiv.org/pdf/2605.22082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 302. QT-PUF: Quantum Tunneling Leakage Based PUF for Implantable IoMT Devices

**arXiv ID:** 2605.22113 | [PDF](https://arxiv.org/pdf/2605.22113v1)

**作者:** Yueqi Ma `[一作]` (Imperial College London), Emmanuel M. Drakakis `[通讯]` (Imperial College London)

**通讯引用:** 2489 | [OpenAlex ID](https://openalex.org/A5061192370)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并仿真实现了一种基于量子隧穿漏电的物理不可克隆功能（QT-PUF），用于低功耗医疗物联网设备的身份认证与密钥生成。

**💡 创新点**

创新点在于利用标准CMOS工艺中的门隧穿漏电（QMDT）作为熵源，并配备可调伪电阻的I‑V转换与差分放大读取电路，实现了无需外部激励、静态工作、几乎无温度依赖且极低功耗的PUF方案。

**🔧 技术方法**

核心技术包括：量子隧穿漏电分析、四晶体管PUF单元设计、伪电阻I‑V转换、差分放大与比较、挑战‑响应对（CRP）生成，全部在65 nm CMOS工艺上实现。

**📊 数据集**

使用的“数据集”是基于Monte‑Carlo仿真的工艺波动模型，覆盖16单元与64×64阵列的多周期（1500 次）仿真样本，统计了泄漏电流分布、响应比特、BER、FHD等指标。

**📈 对比分析**

与现有SRAM、环振荡、软氧化破坏等PUF方法相比，QT‑PUF在65 nm工艺下实现熵0.9999998、阵列间FHD0.5001、BER0.000163、能耗仅96 nW/bit（19.2 fJ/bit），且不需后处理或校准，显著优于多数同类设计。

**⚠️ 局限性**

局限性包括：单元面积略大于深亚微米SRAM/ NAND PUFs；目前仅通过仿真验证，尚缺乏实测硬件验证；对极端温度（>100 °C）与电压偏移的可靠性仍需进一步评估。

---

## 303. Knowledge Graph Re-engineering Along the Ontological Continuum (extended version)

**arXiv ID:** 2605.22093 | [PDF](https://arxiv.org/pdf/2605.22093v1)

**作者:** Enrico Daga `[一作]` (Open University), Terry Payne `[通讯]` (University of Liverpool)

**通讯引用:** 9658 | [OpenAlex ID](https://openalex.org/A5024107224)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“本体连续体”框架，并在十个代表性知识图谱的溯因案例中进行实证验证。

**💡 创新点**

创新点在于将语义/语用与属性/可用性四维区分整合为一个连续空间，强调可用性（affordance）视角，并用经验观察构建可实证的描述体系。

**🔧 技术方法**

核心技术包括经验式数据采集、二元特征矩阵构建，以及形式概念分析（FCA）对知识图谱进行离散投影和层级化。

**📊 数据集**

使用了十个公开溯因相关的知识图谱：Europeana、Google Data Commons、Bio2RDF、British Museum ResearchSpace、UniProt、Wikidata、EU Open Data Portal、DBpedia、LOV 与 Nanopublications。

**📈 对比分析**

比较方法为在四个维度上标注二元特征，生成概念图，进而通过概念层次结构直观展示图谱间的相似性与差异；目前仅提供定性结构分析，未给出量化性能指标。

**⚠️ 局限性**

局限性包括：依赖有限样本，缺乏正式的维度间依赖与成本建模，投影结果仅与特定任务和领域相关，未实现自动化变换工具或客观性能评估。

---

## 304. Balancing Uncertainty and Diversity of Samples: Leveraging Diversity of Least, High Confidence Samples for Effective Active Learning

**arXiv ID:** 2605.22169 | [PDF](https://arxiv.org/pdf/2605.22169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. Automated Repair of TEE Partitioning Issues via DSL-Guided and LLM-Assisted Patching

**arXiv ID:** 2605.22087 | [PDF](https://arxiv.org/pdf/2605.22087v1)

**作者:** Chengyan Ma `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31431 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出了一种针对可信执行环境（TEE）错误划分问题的自动修复框架TEERepair。

**💡 创新点**

通过设计域特定语言（DSL）编码安全模式并结合大型语言模型（LLM）填充上下文，解决了缺乏标准规范、语义提取困难和缺少验证方法的问题。

**🔧 技术方法**

采用DSL规则+LLM推理、静态分析工具DITING、自动生成测试客户端等技术实现自动修复与验证。

**📊 数据集**

使用了TEE Partitioning Errors Benchmark（TPEB）共89个漏洞案例以及从GitHub收集的TEE项目进行评估。

**📈 对比分析**

与基线工具（如TBar、Prophet、VulMaster、ChatRepair、SAN2PATCH）对比，TEERepair在修复成功率上达87.6%，显著优于所有基线。

**⚠️ 局限性**

主要局限在仅针对ARM TrustZone平台，缺少Intel SGX/AMD SEV等支持，并且仍需人工编写测试用例。

---

## 306. Do Factual Recall Mechanisms Carry over from Text to Speech in Multimodal Language Models?

**arXiv ID:** 2605.22170 | [PDF](https://arxiv.org/pdf/2605.22170v1)

**作者:** Luca Modica `[一作]` (Zenseact), Richard Johansson `[通讯]` (Chalmers University of Technology)

**通讯引用:** 2736 | [OpenAlex ID](https://openalex.org/A5079881269)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Speech‑Language Model（SpiritLM）在文本和语音输入下的事实记忆机制进行因果中介分析，探究其内部如何编码、存储与检索事实关联。

**💡 创新点**

首次将Causal Mediation Analysis（因果中介分析）与Speech‑LLM相结合，并通过语音到文本的强制对齐将语音token映射回文本token，实现跨模态的因果追踪。

**🔧 技术方法**

使用Causal Tracing（因果追踪）技术，对模型不同层级（Transformer层、MLP、注意力子层）进行干预与恢复，计算平均间接效应（AIE）以量化事实回忆的贡献。

**📊 数据集**

采用已公开的Known事实问答数据集，并使用MeloTTS将每条提示转为语音，随后通过HuBERT离散化为语音token，形成对应的语音‑文本对照数据集。

**📈 对比分析**

在文本→文本和语音→文本两种输入设置下比较AIE曲线，发现文本输入时中间层MLP和注意力子层的AIE显著高于语音输入；语音输入的因果信号更稀疏、幅度较弱，表明事实检索机制在语音模式下的激活更弱。

**⚠️ 局限性**

局限性包括：仅使用单一合成语音数据集；仅测试SpiritLM一个离散语音token模型；未检验其他多模态或连续语音表示模型；因果追踪结果对模型架构细节高度依赖，缺乏更广泛的泛化验证。

---

## 307. Human Vulnerability Assessment in Cybersecurity: A Systematic Literature Review of Methods, Models, and Instruments

**arXiv ID:** 2605.22119 | [PDF](https://arxiv.org/pdf/2605.22119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 308. Accelerating Vision Foundation Models with Drop-in Depthwise Convolution

**arXiv ID:** 2605.22132 | [PDF](https://arxiv.org/pdf/2605.22132v1)

**作者:** Carmelo Scribano `[一作]` (University of Modena and Reggio Emilia), Luc Van Gool `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在预训练视觉Transformer中，用深度可分离卷积直接替换部分注意力头的drop‑in加速方法；

**💡 创新点**

创新点在于将可学习的卷积核与原有值投影分离为深度卷积，既保留预训练权重，又显著降低计算开销，并提供基于输入不变性的头选择准则；

**🔧 技术方法**

主要技术包括：将多头自注意力拆解为深度卷积、使用输入不变性度量选择卷积化头、在TensorRT框架下进行推理加速、以及两阶段微调恢复性能；

**📊 数据集**

实验数据集覆盖ImageNet-1K分类、COCO和ADE20K语义分割，模型以Dino‑V2为基准，亦验证CLIP和MAE；

**📈 对比分析**

与原始ViT和全卷积替换方案对比，采用块级深度卷积实现约17–20%的推理速度提升，分类精度下降不足1%（mIoU下降≤1%），在Nvidia Orin Nano上实测；

**⚠️ 局限性**

局限性在于需要针对每个预训练模型单独计算输入不变性度量，且仅对卷积化头的性能影响相对较小，对更大比例替换可能导致显著精度损失。

---

## 309. OPERA: An Agent for Image Restoration with End-to-End Joint Planning-Execution Optimization

**arXiv ID:** 2605.22104 | [PDF](https://arxiv.org/pdf/2605.22104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. TextTeacher: What Can Language Teach About Images?

**arXiv ID:** 2605.22098 | [PDF](https://arxiv.org/pdf/2605.22098v1)

**作者:** Tobias Christian Nauen `[一作]` (RPTU University Kaiserslautern-Landau), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在训练阶段引入冻结的文本编码器产生的文本语义锚点，并通过辅助对齐损失引导视觉Transformer（ViT/DeiT等）在ImageNet上训练，从而提升分类准确率，同时保持推理时模型完全纯视觉。

**💡 创新点**

不需要多模态预训练或教师模型，仅在训练时加入轻量文本信息；文本编码器冻结、投影层简洁；将文本视为特征空间的预条件器，促进早期收敛与更好语义对齐。

**🔧 技术方法**

使用自动生成的图片说明（CoCa、BLIP-L等）与BERT/CLIP文本编码器、Whitening、CLIP式对比损失、可调λ_t（adaptive）以及常见视觉Transformer架构；对齐损失为轻量级的双头（分类+文本）训练。

**📊 数据集**

主数据集为ImageNet 1.3M；下游细粒度任务包括FGVC-Aircraft、Caltech-UCSD Birds、Stanford Cars、Oxford Flowers、Food-101、Oxford-IIIT Pets；所有模型均在ImageNet上预训练后微调这些任务。

**📈 对比分析**

与仅分类、视觉/语言预训练对齐、在线知识蒸馏、BorLan等方法对比；在ImageNet上平均提升0.6–1.5个百分点，DeiT-L最高+2.7个百分点；在细粒度任务平均+0.9个百分点；在计算预算相同的情况下，比在线蒸馏快约50%或节省约6 GPU‑h。

**⚠️ 局限性**

仅对从零训练有效，对预训练模型可能无益或有害；依赖高质量、无偏见的图像说明，生成与标签提取需额外计算；在医学影像、遥感等专业域中可能不适用或需定制说明。

---

## 311. RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching

**arXiv ID:** 2605.22083 | [PDF](https://arxiv.org/pdf/2605.22083v1)

**作者:** Jinhyeok Yang `[一作]` (Supertone Inc), Juheon Lee `[通讯]` (Supertone Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在流匹配式文本到语音模型中引入基于长度保持的重复与跳过噪声的对抗性训练，直接在潜在空间生成对应失配的“硬负样本”，并以此增强对齐鲁棒性。

**💡 创新点**

通过在潜在空间构造重复和跳过的失配负样本，结合对比流匹配（Contrastive Flow Matching）实现对齐错误的直接惩罚，从而显著提升内容保真与语义一致性。

**🔧 技术方法**

核心技术是对比流匹配（Contrastive Flow Matching）与潜在空间增强（长度保持的 repeat/skip 变换），并在训练时加入负样本对比正负样本的梯度正则化。

**📊 数据集**

使用内部约 10k 小时、5M 句子、80k 说话人（英语、韩语）的语音数据；评测采用公开 Seed‑TTS‑eval 以及自建多说话人、多文本、多语言的 ZERO500 benchmark。

**📈 对比分析**

与基线 SupertonicTTS、标准 ContrastiveFM 以及多款公开 TTS 系统（如 DiTAR、VoxCPM 等）进行对比；在 Seed‑TTS‑eval 上 WER 从 1.44 降至 1.38（参数仅 0.06B），在 ZERO500 benchmark 上 CER/WER 同样显著下降，尤其在低 NFE（12/24）场景下表现最为突出。

**⚠️ 局限性**

受限于压缩模型的容量，导致说话人相似度（SIM）略有下降；实验仅使用自动化 ASR 指标，缺乏主观 MOS 评估；且目前的硬负样本仅覆盖重复/跳过两类失配，未覆盖更广泛的对齐错误。

---

## 312. Algebraic Machine Learning for Small-to-Medium Datasets Is Competitive against Strong Standard Baselines

**arXiv ID:** 2605.22155 | [PDF](https://arxiv.org/pdf/2605.22155v1)

**作者:** David Mendez `[一作]`, Gonzalo G. de Polavieja `[通讯]` (Champalimaud Foundation)

**通讯引用:** 4998 | [OpenAlex ID](https://openalex.org/A5006594306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 AML（符号化代数学习）在小到中等样本图像和表格分类任务上的性能进行系统评估。

**💡 创新点**

提出 AML 在无交叉验证、无任务相关超参数的情况下，使用通用代数先验即可竞争强大基线。

**🔧 技术方法**

使用子直接分解的稀疏交叉算法构建半格代数，并以逻辑回归或最少漏失方式进行分类。

**📊 数据集**

使用 12 个图像数据集（MNIST、CIFAR-10 等）和 29 个表格数据集（多类别）在 50–2000 样本规模上进行实验。

**📈 对比分析**

与 CNN、XGBoost、LightGBM、随机森林、MLP、SVM 等交叉验证基线比较，AML 在 50–2000 图像样本下平均排名最优，在表格任务中与 LightGBM 和随机森林相当，XGBoost 在表格任务上略优。

**⚠️ 局限性**

缺点包括对大尺寸图像的计算开销、缺乏对非分类任务的适配、未充分消除读取器与代数模型的交互效应、未探索多模态集成等。

---

## 313. Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched Trajectory Reachability Metrics

**arXiv ID:** 2605.22164 | [PDF](https://arxiv.org/pdf/2605.22164v1)

**作者:** Liangyu Li `[一作]` (Tongji University), Qingwen Liu `[通讯]` (Tongji University)

**通讯引用:** 3229 | [OpenAlex ID](https://openalex.org/A5059157106)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于已训练的潜在世界模型的后置终点可达性度量（Trajectory Reachability Metric），通过改进终点评价方式来修正规划器在潜在空间中错误的距离度量，从而显著提升控制性能。

**💡 创新点**

创新点在于：①采用全规划期望时间尺度的监督（horizon‑matched）来训练对称的时间间隔标签；②仅改动终点排序成本而不改动编码器、动力学或采样器；③通过子空间干预和同一候选排序审计提供机制证据，验证了潜在空间内容与规划器接口的分离。

**🔧 技术方法**

使用的技术包括：视觉潜在世界模型（如joint‑embedding encoder + dynamics）、小型两层MLP距离头、平衡全历程时间段采样、标准化与混合终点成本、以及线性探针和子空间操作来定位关键信息。

**📊 数据集**

主要实验数据集为基于二维迷宫/障碍物环境的“n100”硬实例（包含同室与穿墙两种目标），以及更难的go25/go50/go75 障碍导航任务；训练与评估均使用同一套随机种子、缓存和预算协议。

**📈 对比分析**

与原始潜在距离（latent MSE）相比，后置度量在硬 n100 任务上将成功率从 7.0% 提升到 97.0%，并将基线模型从 32.7% 提升至 84.0%；在 go50/go75 等连续接触任务中，作为辅助（hybrid）成本可提升候选排序与最终距离，但闭环成功率提升有限，表明仍受动力学误差与执行瓶颈限制。

**⚠️ 局限性**

局限性包括：①仅在固定的世界模型和评估协议内验证，未展示跨域泛化；②需要大量同一环境轨迹以生成时间标签；③对短期或低覆盖采样的性能敏感；④在连续控制任务中仍需结合动力学校准与执行改进，单一终点度量不足以彻底解决问题。

---

## 314. Adversarial Trust Poisoning in Vehicular Collaborative Perception

**arXiv ID:** 2605.22122 | [PDF](https://arxiv.org/pdf/2605.22122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 315. MotionDPS: Motion-Compensated 3D Brain MRI Reconstruction

**arXiv ID:** 2605.22121 | [PDF](https://arxiv.org/pdf/2605.22121v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. One-Way Policy Optimization for Self-Evolving LLMs

**arXiv ID:** 2605.22156 | [PDF](https://arxiv.org/pdf/2605.22156v1)

**作者:** Shuo Yang `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18479 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的RLVR优化方法One-Way Policy Optimization (OWPO)，通过解耦优化方向与更新幅度实现更稳定的自我演化；

**💡 创新点**

创新点在于引入方向感知的一阶“单向信任区域”，对劣势偏差加速对齐，对优势偏差进行锁定，并通过迭代刷新参考策略实现“齿轮效应”，避免对称KL导致的方向反转；

**🔧 技术方法**

采用PPO/DAPO式的裁剪目标，结合优势重加权、参考策略对比、动态权重、以及多阶段迭代刷新等技术；

**📊 数据集**

主要使用数学推理验证集DAPO-Math-17k（含AIME24/25、AMC）和通用科学基准GPQA、MMLU-Pro；

**📈 对比分析**

与纯RLVR方法（GRPO、DAPO）以及参考引导蒸馏方法（OPD、MOPD）进行对比，OWPO在所有基准上实现显著提升（Pass@1平均提升约3.9%-7.7%，且在迭代自我演化中仅需4轮即可突破弱参考上限），并保持更高的收敛性能；

**⚠️ 局限性**

局限在于主要评估于可验证的数学推理任务，缺乏对代码、工具使用或交互式RLVR的验证；依赖参考策略与超参数（刷新频率、权重区间）；理论分析仅局限于局部一阶更新。

---

## 317. Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents

**arXiv ID:** 2605.22148 | [PDF](https://arxiv.org/pdf/2605.22148v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入 Ratchet 单循环，冻结 LLM 自动写检索、策划、保留和淘汰自然语言技能，解决技能库生命周期管理瓶颈。

**💡 创新点**

创新点在于通过结果驱动淘汰、有限容量、元技能写作约束和模式规范化等治理机制，确保自演化技能库的非发散性，并首次将 LLM 自创技能性能提升至接近人工级别。

**🔧 技术方法**

使用技术包括：冻结的 Claude Opus 4.7 LLM、Cohere embed‑v4 嵌入检索、Meta‑skill 约束、证据日志与回溯恢复、单轮推理、滚动评估与离线实验。

**📊 数据集**

数据集：MBPP+ hard‑100 代码生成子集（60 训练 / 40 评估）和 SWE‑bench Verified hard‑150 多步骤编码子集。

**📈 对比分析**

方法对比：与无技能基线、仅检索、无元技能、硬淘汰、无规范化等八个消融进行对比；默认配置在 MBPP+ 上实现滚动提升 +0.328，SWE‑bench 提升 +0.22，显著优于所有基线。

**⚠️ 局限性**

局限性：权重冻结导致无法学习新知识；实验仅在单一模型/供应商上，缺乏跨模型验证；Critic 噪声可能导致持久错误；退役基于相关性而非因果；样本规模和任务多样性有限。

---

## 318. A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering

**arXiv ID:** 2605.22099 | [PDF](https://arxiv.org/pdf/2605.22099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 319. ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning

**arXiv ID:** 2605.22106 | [PDF](https://arxiv.org/pdf/2605.22106v1)

**作者:** Yeqiu Chen `[一作]` (University of Science and Technology of China), Lei Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 44924 | [OpenAlex ID](https://openalex.org/A5100349482)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种树结构感知的 KV 缓存管理框架 ArborKV，专为 Tree-of-Thoughts（ToT）推理设计；

**💡 创新点**

创新点在于结合轻量级价值估计（MSVE）与树结构分配策略（TAE），实现了可逆（lazy rehydration）和 token-extractive 的 eviction，充分利用搜索动态和树拓扑；

**🔧 技术方法**

使用了多信号价值估计（搜索值、熵不确定性、累积注意力）与树结构预算分配、Token级高频子句选择、懒惰恢复等技术；

**📊 数据集**

在 GSM8K、MATH500、Game of 24 和 AIME 四个推理基准上进行评估；

**📈 对比分析**

与 H2O、StreamingLLM、ThinKV 等线性序列化 KV 管理方法对比，在相同内存预算下，ArborKV 在保持接近完整 KV 的情况下，内存峰值减少约 4 倍，推理准确率仅略有下降；

**⚠️ 局限性**

限制在于对树结构的依赖需要搜索控制器提供准确的前沿信息，且在极度回溯频繁的场景中，懒恢复的预填充仍可能产生额外延迟与资源开销。

---

## 320. Narrative Sharpens Gender Gaps: Surveying Film Characters with LLM Agents

**arXiv ID:** 2605.22091 | [PDF](https://arxiv.org/pdf/2605.22091v1)

**作者:** Vivienne Bihe Chi `[一作]` (University of Pennsylvania), Sharath Chandra Guntuku `[通讯]` (University of Pennsylvania)

**通讯引用:** 4204 | [OpenAlex ID](https://openalex.org/A5010646067)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建基于电影剧本的虚构角色LLM代理，模拟其在世界价值观调查中的性别态度。

**💡 创新点**

首次将叙事角色行为转化为可测量的性别态度，并展示电影叙事在性别态度上的放大与波动。

**🔧 技术方法**

使用LLM生成的专家视角反射进行人物个性凝练，并以GPT‑5‑mini为推理模型。

**📊 数据集**

使用MovieSum剧本数据和OMDb元数据构建角色库，并选取1990‑2019年美国电影160部共734个角色。

**📈 对比分析**

将模拟问卷回答与WVS实测数据对比，结果显示角色代理放大性别差距且波动更大。

**⚠️ 局限性**

局限包括女性角色样本不足、未区分叙事与模型先验影响、未使用混合效应模型等。

---

## 321. LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model

**arXiv ID:** 2605.22089 | [PDF](https://arxiv.org/pdf/2605.22089v1)

**作者:** Xiaodong Mei `[一作]` (Hong Kong University of Science and Technology), Dan Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5144 | [OpenAlex ID](https://openalex.org/A5100778603)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LVDrive框架，通过在VLA模型中加入隐式未来视觉表征预测任务，实现单次前向推理的未来感知与动作规划。

**💡 创新点**

创新点在于将未来场景预测迁移到高层隐空间，避免像素重建误导；同时设计两阶段轨迹解码器，使预测的语义特征直接细化轨迹；并通过预训练VQGAN-ImageNet提供稠密语义监督。

**🔧 技术方法**

利用Vision-Language-Action（VLA）大模型（Vicuna + LoRA）结合视觉编码器（EVA‑02‑L / QT‑Former），构建隐空间未来视觉表征学习与轨迹解码模块；使用跨模态注意力与VAE规划器实现两阶段解码。

**📊 数据集**

在CARLA 2.0仿真环境下的Bench2Drive基准数据集（包含1000条训练、50条验证、220条闭环测试路线）上进行评估。

**📈 对比分析**

与传统端到端规划器、VLA方法及世界模型方法对比，LVDrive在Bench2Drive闭环评测中获得最高的Driving Score 80.71点、成功率58.26%，显著优于UniDrive‑WM‑AR等对手。

**⚠️ 局限性**

局限在于未充分利用语言监督提升视觉/动作表征；所选的预训练视觉骨干为通用模型，若使用专为自动驾驶预训练的视觉基础模型可能进一步提升性能。

---

## 322. GenHAR: Generalizing Cross-domain Human Activity Recognition for Last-mile Delivery

**arXiv ID:** 2605.22086 | [PDF](https://arxiv.org/pdf/2605.22086v1)

**作者:** Zhiqing Hong `[一作]` (Hong Kong University of Science and Technology), Desheng Zhang `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 GenHAR 框架，用于解决跨域人类活动识别（HAR）的分布漂移问题，通过将 IMU 传感器数据转换到频域并学习幅值特征，结合传感器维度自注意力与选择性遮蔽，实现对源域数据的域不变表征学习，并在边缘设备上实现实时推理。

**💡 创新点**

创新点：
1) 仅使用频域幅值特征进行跨域泛化，证明幅值比相位更鲁棒；
2) 设计传感器维度自注意力（sensor‑wise attention）捕获不同传感器通道间的关联，显著提升泛化性能；
3) 引入选择性遮蔽机制减少不重要通道间的注意力计算，显著降低 FLOPs 与推理时延。

**🔧 技术方法**

技术手段：FFT 频域变换、幅值提取、Transformer 自注意力（在传感器通道维度上）、Selective Masking、LayerNorm、全局平均池化、线性投影分类器、PyTorch 实现，支持模型压缩与量化。

**📊 数据集**

使用四个公开 IMU 数据集：UCI、Shoaib、Motion、HHAR，每个数据集被视为一个域，用作源域训练和目标域测试。

**📈 对比分析**

对比 10 个基线模型（MLP、CNN、ResNet、LSTM、TSFCN、LIMUBERT、ContraTSC、FreHAR、SDMix、UniHAR）。
- 平均准确率：GenHAR 81.42%，优于第二名 UniHAR 71.45%（提升 9.97%）。
- 参数量 25k，FLOPs 52k，推理时间 0.0299 s，显著低于同类方法（如 LIMUBERT 1136k FLOPs、0.4089 s）。

**⚠️ 局限性**

局限性：
1) 仍无法进一步减少模型规模与计算量，尤其在高频成分裁剪后仍需实验；
2) 仅在源域训练，未利用目标域无标签数据，可能在极端噪声或设备差异上性能下降；
3) 依赖 FFT 预处理，对实时低延迟场景需进一步优化；
未来工作包括模型剪枝、量化、跨任务迁移（如异常检测、手势识别）以及更细粒度的域不变学习。

---

## 323. Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents

**arXiv ID:** 2605.22166 | [PDF](https://arxiv.org/pdf/2605.22166v1)

**作者:** Tianshi Xu `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 25687 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种生命周期感知的运行时工具包（lifecycle-aware runtime harness），在不更新 LLM 权重或环境的前提下，通过从训练轨迹中提取可重用的接口干预来提升冻结 LLM 代理的表现。

**💡 创新点**

创新点在于：①将代理适配视为接口适配而非参数适配；②设计四层可进化的运行时接口（环境契约层、程序化技能层、动作实现层、轨迹调控层），每层针对不同失败模式进行干预；③使用训练轨迹驱动的增量演化方法，使得生成的干预在跨模型、跨任务上具有广泛的可迁移性。

**🔧 技术方法**

技术包括：训练轨迹收集、基于错误诊断的干预生成、非参数技能检索（BM25）、动作合法性验证、轨迹模式检测与恢复机制；使用 Codex 等编程助手协助编写与演化工具包代码。

**📊 数据集**

数据集涵盖三大基准：τ‑bench、τ²‑bench、AgentBench，包含七种确定性任务（航空、零售、电信、ALFWorld、WebShop、操作系统、数据库）以及 18 种不同模型后端。

**📈 对比分析**

评估方法：在不修改模型权重和环境的情况下，使用 0.0 温度生成策略，对每个基准进行单次或三次运行（Pass@1、Pass³）。实验显示平均相对提升 88.5%，在 116/126 设定中获胜；与仅进行提示演化（prompt evolving）相比，提升约 120%；相较于专业工具训练模型，运行时工具包可在保持权重不变的情况下取得更高或可持续的改进。

**⚠️ 局限性**

局限性在于仅针对确定性、规则驱动的环境；在开放式、非确定性任务中，难以定义稳定的运行时接口或迁移可重用的干预。

---

## 324. Market-Analysis-Driven Methodology for Assessing Charging Station Cybersecurity

**arXiv ID:** 2605.22151 | [PDF](https://arxiv.org/pdf/2605.22151v1)

**作者:** Jakob Löw `[一作]` (Technische Hochschule Ingolstadt), Hans-Joachim Hof `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 565 | [OpenAlex ID](https://openalex.org/A5029363762)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于市场分析的可扩展方法，用以在全国范围内评估充电站的网络安全状况。

**💡 创新点**

创新点在于通过将充电站按运营商-制造商对聚类，从少量现场测试中外推到整个国家，从而实现大规模安全评估。

**🔧 技术方法**

使用了现场测试设备模拟 EV 侧通信，检测 TLS、ISO 15118、DIN SPEC 70121 等协议，并结合市场数据进行聚类与外推。

**📊 数据集**

数据集来自德国 Bundesnetzagentur 公开充电点列表、goingelectric.de 的社区数据，涵盖 43,913 个 CCS 充电点，其中 40,949 个可用制造商信息。

**📈 对比分析**

方法通过对每个 (运营商, 制造商) 集群进行抽样测试，再将结果外推到该集群内全部充电点。性能上，本文对德国 51.9% 的充电点进行评估，发现仅 27.4% 支持 TLS。

**⚠️ 局限性**

局限性包括假设充电站同一制造商、同一运营商配置一致，且样本数量有限；未考虑固件更新、地区差异或临时误配导致的变化。

---

## 325. A Coalgebraic Dijkstra Algorithm

**arXiv ID:** 2605.22149 | [PDF](https://arxiv.org/pdf/2605.22149v1)

**作者:** Takahiro Sanada `[一作]` (Fukui Prefectural University), Ichiro Hasuo `[通讯]` (National Institute of Informatics)

**通讯引用:** 1850 | [OpenAlex ID](https://openalex.org/A5013382452)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于 coalgebra 的通用最短路径框架（CSPP），并给出了相应的 coalgebraic Dijkstra 算法，能够在满足特定条件的状态转移系统中高效求解最优值。

**💡 创新点**

创新点在于：①用范畴论/coalgebra 把经典 Dijkstra 加速的本质抽象为“广义状态转移系统”；②给出了必要且充分的“可扩展性（expansiveness）”条件，精确刻画何时能使用 Dijkstra 风格加速；③在同一框架下统一处理路径、树、游戏、概率等多种优化问题。

**🔧 技术方法**

核心技术包括：coalgebra、终值域（pointed weight domain）与转换模态（transition modality）、固定点理论（最大固定点）、有限支持函数（finitely supported functor）以及 Fibonacci 堆实现的优先队列。

**📊 数据集**

本文没有使用具体实验数据集，而是通过理论证明和复杂度分析展示算法的正确性与效率；若需要实验可参考论文中给出的示例图与树形结构。

**📈 对比分析**

与传统 Dijkstra 的比较表明：在满足可扩展性条件时，coalgebraic Dijkstra 与经典 Dijkstra 的时间复杂度相当（O(E+VlogV)），并可通过 Fibonacci 堆进一步优化；在不满足条件时，传统算法不适用，而 coalgebraic Dijkstra 通过条件判断能避免错误收敛。

**⚠️ 局限性**

局限性包括：①算法的正确性依赖于 G 为弱拉回保持的、非空的有限支持 functor 以及 σ 的可扩展性；②对更广泛的范畴（如拓扑空间、名义集合）尚未推广；③复杂度分析仍基于单个顶点更新时间的假设，实际实现中可能需要针对特定 G 调整；④需要证明或验证可扩展性条件，对于某些实际模型可能难以检查。

---

## 326. AesFormer: Transform Everyday Photos into Beautiful Memories

**arXiv ID:** 2605.22126 | [PDF](https://arxiv.org/pdf/2605.22126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 327. Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution

**arXiv ID:** 2605.22147 | [PDF](https://arxiv.org/pdf/2605.22147v1)

**作者:** Jiangwei Mo `[一作]` (Beijing Foreign Studies University), Hanlin Wu `[通讯]` (Beijing Foreign Studies University)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5067703101)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FlowGS，利用流匹配生成高频细节潜变量并结合 2D Gaussian splatting，实现遥感图像任意尺度的单步超分辨率重建。

**💡 创新点**

创新点包括：① 在细节潜空间使用条件流匹配（FM）取代多步扩散；② 引入快捷一致性约束，实现一阶 ODE 单步推理；③ 将连续特征场与 Gaussian splatting 相结合，支持任意尺度查询。

**🔧 技术方法**

采用条件流匹配、单步 Euler ODE、2D Gaussian splatting、连续坐标渲染、LPIPS/对抗/KL 正则化、两阶段预训练与细节潜变量生成。

**📊 数据集**

在 AID、DOTA、DIOR 三个公开遥感数据集（含 AID-tiny 小样本集）上进行训练与测试。

**📈 对比分析**

与 LIIF、CiaoSR、GaussianSR（连续尺度）以及 HAT-L、TTST、SR3、EDiffSR、SPSR（固定尺度）对比，使用 LPIPS 与 FID 评估；FlowGS 在大尺度（×8）下 FID 最低，并且单步推理速度是扩散方法的 1/10 左右。

**⚠️ 局限性**

局限性在于：对极端大尺度或极端噪声的鲁棒性不足；单步推理在极端条件下可能出现失真；缺乏针对真实传感器降解（如压缩、光照变化）的适配与评估。

---

## 328. Latency in Real-Time 3D Volumetric Streaming: A Comprehensive Study

**arXiv ID:** 2605.22131 | [PDF](https://arxiv.org/pdf/2605.22131v1)

**作者:** Seungwoo Hong `[一作]` (Electronics and Telecommunications Research Institute), Inayat Ali `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5053948849)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在一个真实的3D体积直播会议系统中，对端到端延迟进行系统化测量和分析。

**💡 创新点**

首次将延迟拆分为应用层、传输层和网络层，量化每层瓶颈并提出跨层优化方案。

**🔧 技术方法**

使用Unity+Azure Kinect捕获、基于UDP的自定义协议、IEEE 1588v2精准时钟同步及10/100Gbps以太网硬件。

**📊 数据集**

使用现场实时采集的演示者3D点云（约3.52 Mbytes帧）进行实验，不依赖公开数据集。

**📈 对比分析**

通过对比服务延迟、帧延迟、协议延迟与网络延迟，发现应用层渲染时间占总延迟的58%，系统平均端到端延迟约50 ms。

**⚠️ 局限性**

受限于单一硬件配置、单一路径（1 km光纤）和自定义协议，缺乏跨网络环境和大规模多客户端的验证。

---

## 329. Aerodynamic force reconstruction using physics-informed Gaussian processes

**arXiv ID:** 2605.22111 | [PDF](https://arxiv.org/pdf/2605.22111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 330. Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?

**arXiv ID:** 2605.22109 | [PDF](https://arxiv.org/pdf/2605.22109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 331. ExComm: Exploration-Stage Communication for Error-Resilient Agentic Test-Time Scaling

**arXiv ID:** 2605.22102 | [PDF](https://arxiv.org/pdf/2605.22102v1)

**作者:** Woomin Song `[一作]` (KAIST), Aram Galstyan `[通讯]` (Amazon AGI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种面向探索阶段的代理测试时缩放通信协议，包含在线信念一致性模块和轨迹多样化模块，通过软更新纠正错误并防止错误传播。

**💡 创新点**

创新点在于：①在并行代理执行过程中主动检测跨代理事实冲突并使用工具验证解决；②使用软信念更新保留原始信息，避免误修正；③引入轨迹多样化模块以防止信息共享导致的路径坍塌。

**🔧 技术方法**

技术手段包括：基于ReAct循环的代理执行框架；在线一致性检测与工具验证循环；软信念更新策略；批量计划分析与指令化的轨迹重定向。

**📊 数据集**

使用的数据集为 AIME 2024、AIME 2025（配备代码解释器）以及 GAIA（代码解释器、文件管理工具和网络搜索）。

**📈 对比分析**

与单核、序列修订、独立并行、树搜索、以及适配的多代理辩论和混合代理基线进行对比。实验中 Gemini‑2.5‑Flash‑Lite 与 Qwen3.5‑4B 上平均提升 5.7%/5.0%，错误恢复率提升至 38.7%/61.8%，并保持或降低了 API 费用与延迟。

**⚠️ 局限性**

局限性包括：对验证器准确性的依赖，软更新可能导致信息累积；轨迹多样化策略需要手工调参；在某些无冲突错误场景下效果有限；整体推理延迟相对较高。

---

## 332. Astragalus: Automatic Configuration Repair for Production Networks

**arXiv ID:** 2605.22092 | [PDF](https://arxiv.org/pdf/2605.22092v1)

**作者:** Zhenrong Gu `[一作]` (Xi'an Jiaotong University), Xu Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 62674 | [OpenAlex ID](https://openalex.org/A5100355692)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Astragalus，一款基于语法驱动的自动网络配置修复工具，采用局部化-修复-验证的迭代流程来定位并修复配置错误。

**💡 创新点**

创新点包括：① 抛弃传统的基于SMT约束的语义驱动方法，转而使用“塑料手术假设”在网络配置中寻找可直接植入的修复片段；② 通过AST层面的语法操作（删除、插入、修改）生成候选修复；③ 采用谱系故障定位（SBFL）与快速验证相结合的高效搜索策略；④ 在大规模生产网络中实现秒级甚至分钟级的修复。

**🔧 技术方法**

核心技术：
- 把网络配置建模为抽象语法树（AST）并定义配置单元与容器单元；
- 采用 Spectrum‑Based Fault Localization（Ochiai）计算配置单元的可疑度；
- 生成候选修复操作（Remove/Insert/Modify）并按可疑度排序；
- 利用现有配置验证器（如BGP控制平面模拟器）快速验证候选修复；
- 深度优先搜索策略与状态去重，实现高效遍历。

**📊 数据集**

数据集：
- 真实数据中心网络：83台设备，约306行/设备；
- 合成fat‑tree网络（k=4,6,8,10,12）：每台设备97–163行配置；
- 对两种数据集分别注入15类误配置（共2000+实例），每类100个实例。

**📈 对比分析**

比较方法与性能：
- 与现有语义驱动工具 AED（SMT合成）和 CEL（最小修正集定位）进行对比；
- 在合成网络上，Astragalus 的修复时间平均低于 AED 和 CEL 4–5 个数量级，且在 k≥6 时仍能在 10 000 s 内完成；
- 在真实网络上，成功率达 97.5%，平均修复时间仅 6.93 s，绝大多数错误在 1 s 内修复；
- 成功率、修复时间与工具对比结果显示，语法驱动方法在规模与效率上明显优于语义驱动工具。

**⚠️ 局限性**

局限性：
- 仅修复 BGP 等路由协议配置，无法处理数据平面（NAT、PBR）或缺失路由源的错误；
- 对于仅出现一次的设备角色或唯一配置，塑料手术假设不一定成立，导致修复失败；
- 需要手动审查非根因修复（如“插入静态路由”）以确认真正根因；
- 目前缺乏对 WAN 等非数据中心网络的评估；
- 仍依赖手工生成的测试前缀集合，无法覆盖所有类型的误配置。

---

## 333. A Camera-Cooperative ISAC Framework for Multimodal Non-Cooperative UAVs Sensing

**arXiv ID:** 2605.22090 | [PDF](https://arxiv.org/pdf/2605.22090v1)

**作者:** Wenfeng Wu `[一作]` (Nanjing University), Kun Yang `[通讯]` (Nanjing University)

**通讯引用:** 31877 | [OpenAlex ID](https://openalex.org/A5058780924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种Camera‑Cooperative ISAC (CC‑ISAC)框架，用摄像头实现粗定位、ISAC实现精测，提升无人机检测与跟踪效率。

**💡 创新点**

创新点包括：①跨模态任务分配，将视觉用于空气域预判、ISAC用于细粒度感知；②视觉‑回波对齐模型 V2EDA，利用交叉注意力学习视觉到波束空间的非线性映射；③多模态融合估计模型 MMFE，融合历史回波与实时视觉，实现鲁棒角度预测。

**🔧 技术方法**

主要技术：深度学习交叉注意力、Transformer 时序编码、YOLOv4 目标检测、MUSIC 回波定位、层级波束搜索与扩散候选策略。

**📊 数据集**

使用 Arizona State University 的 DeepSense‑6G 数据集，包含同步 RGB 视频与 UAV GPS/回波信息。

**📈 对比分析**

与传统层级扫描、单模态回波、卡尔曼滤波基线对比，CC‑ISAC 在多种码本分辨率下平均减少 71% 的波束搜索开销，跟踪阶段降低 1.69–11.15% 的扫描开销，并保持与基线相近或更优的角度精度。

**⚠️ 局限性**

局限性：需摄像头与 ISAC 基站高度同步与校准，受光照/遮挡影响；实验基于仿真/单目标场景，未在实际硬件或多目标/NLoS 环境中验证；对极端天气或高动态目标的鲁棒性尚待进一步研究。

---

## 334. Information-Theoretic Decentralized Secure Aggregation with User Dropouts

**arXiv ID:** 2605.22261 | [PDF](https://arxiv.org/pdf/2605.22261v1)

**作者:** Zhou Li `[一作]` (Guangxi University), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在存在用户掉线和T‑协作攻击的去中心化安全聚合（DSA）问题，给出了其信息理论可行性阈值以及最优通信速率区域。

**💡 创新点**

创新点包括：①首次给出了DSA的可行性条件U>T+1；②通过基于(T+1)‑私密MDS矩阵构造的相关密钥实现了在两轮通信中同时满足可靠性与信息理论安全；③完全解析了最优速率为R1=1，R2=1/(U‑T‑1)，并证明该速率与用户总数K无关。

**🔧 技术方法**

主要技术手段为向量线性编码、MDS矩阵（含(T+1)‑私密性），以及信息熵不等式与Shannon级安全性分析。

**📊 数据集**

本文为理论研究，无使用实验数据集。

**📈 对比分析**

通过构造最优可行方案与严谨的对偶证明，证明所给速率区域是最优的；相较于现有聚合协议，所实现的通信成本仅取决于有效冗余U‑T‑1，显著降低了通信开销。

**⚠️ 局限性**

主要限制在于假设网络为全连接广播，且密钥需要跨用户全局协同生成；对稀疏网络或仅本地可生成密钥的情况尚未给出可行方案。

---

## 335. D3Seg: Dependency-Aware Diffusion for Brain Tumor Segmentation with Missing Modalities

**arXiv ID:** 2605.22249 | [PDF](https://arxiv.org/pdf/2605.22249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. An Evidence Hierarchy for Bayesian Object Classification via OSINT-Aided Heterogeneous Sensor Fusion

**arXiv ID:** 2605.22259 | [PDF](https://arxiv.org/pdf/2605.22259v1)

**作者:** Jan Nausner `[一作]` (Austrian Institute of Technology GmbH), Michael Hubner `[通讯]` (Austrian Institute of Technology GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于上下文感知与领域知识增强的贝叶斯多源传感器融合方法，用于CBRNE威胁类型的检测与分类。

**💡 创新点**

创新点包括：① 形式化的证据层级（直接、指示性、上下文）实现不同层级信息的统一建模；② 利用OSINT（如OpenStreetMap）构建地理区域先验，提升融合的上下文鲁棒性；③ 在贝叶斯MAP框架下分别推导直接与指示性传感器的似然函数，实现早期融合。

**🔧 技术方法**

使用的技术主要有贝叶斯推断、证据层级模型、OSINT/地理信息系统（GIS）处理、Monte Carlo仿真与基线投票比较。

**📊 数据集**

采用模拟实验生成的基本与CBRNE场景数据；未使用真实标签数据，全部为仿真产生的传感器检测与置信度。

**📈 对比分析**

与传统后期投票（多数表决）基线对比，实验表明该方法在基本场景准确率提升15.8个百分点、CBRNE场景提升9.7个百分点；早期融合显著优于后期融合，直接证据传感器对性能贡献显著。

**⚠️ 局限性**

局限性包括：传感器检测条件独立性假设可能不成立；传感器模型过于简化，需在真实数据上校准；目前仅在仿真中验证，缺乏实测数据支持。

---

## 337. Harder to Defend: Towards Chinese Toxicity Attacks via Implicit Enhancement and Obfuscation Rewriting

**arXiv ID:** 2605.22258 | [PDF](https://arxiv.org/pdf/2605.22258v1)

**作者:** Jingyi Kang `[一作]` (Dalian University of Technology), Hongfei Lin `[通讯]` (Dalian University of Technology)

**通讯引用:** 9083 | [OpenAlex ID](https://openalex.org/A5023931221)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了中文隐式毒性攻击框架CITA，用于生成隐式毒性样本并评估检测器鲁棒性；

**💡 创新点**

将意图保持、语义隐性化和表面混淆三阶段分离，系统化评估中文毒性检测器并为蓝队训练提供红队数据；

**🔧 技术方法**

采用有监督微调、GRPO强化学习、表面重写代理、人工评测与攻击成功率（ASR）评估等技术；

**📊 数据集**

使用12,242条语境-响应对（自构）以及五个公开中文毒性数据集（COLD、SWSR、SCCD、CNTP、ToxiCN）和生成的红队样本；

**📈 对比分析**

对七个检测器（腾讯、百度、Gemini、Claude、GPT、DeepSeek、Qwen3）进行ASR评估，CITA完整流水线平均ASR达69.48%，显著优于公共数据集；用生成样本训练的Qwen3模型在五个测试集上平均准确率91.97%；

**⚠️ 局限性**

仅评估单轮中文对话，缺少多轮和多模态情境；受限于模型与计算资源，未覆盖所有新模型；红队数据需人工筛选，生成质量与覆盖度仍待提升。

---

## 338. No Epoch Like the Present: Robust Climate Emulation Requires Out-of-Distribution Generalisation

**arXiv ID:** 2605.22248 | [PDF](https://arxiv.org/pdf/2605.22248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 339. Explainable AI for Data-Driven Design of High-Dimensional Predictive Studies

**arXiv ID:** 2605.22243 | [PDF](https://arxiv.org/pdf/2605.22243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 340. Unlocking Proactivity in Task-Oriented Dialogue

**arXiv ID:** 2605.22240 | [PDF](https://arxiv.org/pdf/2605.22240v1)

**作者:** Hongbin Zhang `[一作]` (Keeta AI, Meituan), Chaozheng Wang `[通讯]` (Keeta AI, Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了认知用户模拟器（CUS），并提出Simulator-Induced Asymmetric-View Policy Optimization（SI-AVPO）实现主动式任务导向对话的训练与优化。

**💡 创新点**

创新点在于通过显式建模用户隐藏的内部关注点（latent concerns），并将其作为训练时的特征，通过AOPD实现偏置视角的自蒸馏，STPR利用模拟器的状态转移为每轮对话提供细粒度信用信号，完成无需外部奖励模型的主动学习。

**🔧 技术方法**

技术手段包括：层次化用户画像（背景、外观和内部关注），自上而下的策略优化框架，Asymmetric On-Policy Self-Distillation、State-Transition Policy Refinement、基于LLM的用户模拟器、对话生成与强化学习（PPO、GRPO、SEAD等）对比实验。

**📊 数据集**

使用了100K+真实外卖平台的主动拨打记录来构建用户关注点库和人物画像，并在两项实际业务任务（商家推广、快递员地区奖励）上分别采样10,000训练、200测试的用户画像。

**📈 对比分析**

与五大旗舰LLM（GPT‑4.1、DeepSeek‑V3.2、GLM‑5.1、Kimi‑K2.5、Claude‑Sonnet‑4.5）以及多种RL基线（GRPO、PPO、DAPO、SEAD）对比，实验显示在接受率、关注点解决率、对话质量（沟通、逻辑、主动性）等指标上，基于SI-AVPO的Qwen3‑4B/8B模型实现了超过所有RL基线且接近或超过旗舰模型的性能。

**⚠️ 局限性**

局限性包括：依赖于手工构建的关注点库与规则，模型对不同用户画像的泛化仍需进一步验证；AOPD与STPR对训练成本和调参敏感；对话主动性的评价主要基于模拟器和LLM评审，缺乏大规模真实用户实验。

---

## 341. An Architecture for Decentralised Deployment and Operation of Blockchain Applications

**arXiv ID:** 2605.22239 | [PDF](https://arxiv.org/pdf/2605.22239v1)

**作者:** Fabian Stiehle `[一作]` (Technical University of Munich), Ingo Weber `[通讯]` (Technical University of Munich)

**通讯引用:** 8089 | [OpenAlex ID](https://openalex.org/A5071642549)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种去中心化的区块链应用部署与运维体系，集成治理、升级机制与DevOps最佳实践，设计并实现了Deterministic Registry，并提供了完整的开源实现与评估。

**💡 创新点**

① 将治理、可升级与CI/CD三大领域统一到一个去中心化架构；② 通过 CREATE2 计算确定性地址的 Deterministic Registry，实现部署时的版本真实性与可追溯性；③ 采用分布式文件存储与多方本地CI/CD管道，使每方可独立验证并投票。

**🔧 技术方法**

技术栈包括 Solidity 与 OpenZeppelin 的 Governance 与 BeaconProxy；Docker 化的 CI/CD 环境；GitHub Actions 与 IPFS 或其他去中心化文件系统；EIP‑1014 的 deterministic address 方案；Hardhat 本地链模拟；Process Mining（Petri Net、Token‑Based Replay）用于流程合规验证。

**📊 数据集**

评估以 Hub‑Portal‑Chat 开源 dApp 为真实案例，部署于两个地理位置的节点（AWS 与本地 Munich），使用其 GitHub 仓库中的合约与测试集；实验中包含多版本升级、投票、时间锁等完整流程。

**📈 对比分析**

方法：先用流程模型（Petri Net）定义合法路径，再通过事件日志回放（Token‑Based Replay）检查合规性；性能：一次完整部署约 51 万 gas，主网成本约 21 美元；在 Optimism 等 L2 上，成本可降至 <$0.01。性能瓶颈主要在存储与治理交易的 gas 消耗。

**⚠️ 局限性**

限制：在 L1 上成本较高；依赖区块链时间（区块间隔变动）导致时间锁与投票窗口的精度受限；治理参数（投票延迟、时间锁）一经设置后需通过治理变更，难以灵活调整；未涵盖委托投票、离线/链下治理等场景；若不使用 L2/L3，频繁升级成本仍不低。

---

## 342. A Robust Semantic Segmentation Pipeline for the CVPR 2026 8th UG2+ Challenge Track 2

**arXiv ID:** 2605.22216 | [PDF](https://arxiv.org/pdf/2605.22216v1)

**作者:** Jinming Chai `[一作]` (Xidian University), Fang Liu `[通讯]` (Xidian University)

**通讯引用:** 20433 | [OpenAlex ID](https://openalex.org/A5100453114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 UniMatch V2 的半监督语义分割管线，利用清晰天气图像做监督、降雨天气图像做一致性学习，并在推理时加入 TTA。

**💡 创新点**

创新点在于将清晰与降雨成对数据仅用于教师‑学生一致性训练，结合弱到强增强、特征级互补 Dropout，以及仅使用挑战提供的数据。

**🔧 技术方法**

使用了 UniMatch V2 结构、DINOv2 预训练编码器、EMA 教师、弱-强一致性、互补通道 Dropout、以及 Test‑Time Augmentation。

**📊 数据集**

使用 WeatherProof 数据集（10 类语义标注，清晰与降雨成对）。

**📈 对比分析**

通过 Clean only、Clean+Degraded、Clean+Degraded+TTA 三种设置对比，验证集 mIoU 分别为 0.69、0.79、0.80，最终评测 mIoU 达到 0.80。

**⚠️ 局限性**

局限性在于仅在单一数据集上验证，缺乏跨域泛化评估，且对 TTA 的提升有限，未利用外部大规模数据提升模型表现。

---

## 343. Towards a compositional semantics for quantitative confidence assessment in assurance arguments

**arXiv ID:** 2605.22213 | [PDF](https://arxiv.org/pdf/2605.22213v1)

**作者:** Benjamin Herd `[一作]` (Fraunhofer Iks), Lydia Gauerhof `[通讯]` (Robert Bosch Gmbh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种基于Subjective Logic的可组合语义，用于在安全保证（assurance）论证中量化并传播置信度，从而将传统的结构化论证（如GSN）转化为可分析的置信网络。

**💡 创新点**

创新点在于：①将论证元素（目标、证据、假设等）统一映射为Subjective Logic的意见；②把论证关系（支持、上下文）视为SL的算子（条件推导、融合、乘法/共乘法），实现端到端的置信度传播；③保留推理的来源与上下文，支持模块化与可追溯性；④提供了示例与实用指南，弥合定性论证与定量置信评估之间的差距。

**🔧 技术方法**

核心技术是Subjective Logic（SL）及其组合算子（条件推导、共乘法、乘法、累积融合等）和目标结构化论证框架（Goal Structuring Notation, GSN）。

**📊 数据集**

本文没有使用公开数据集，而是以一个简化的GSN安全论证示例（包含目标、策略、证据、假设等）进行演示和手工置信度传播。

**📈 对比分析**

方法通过构造SL置信网络手工计算置信度，未做实验性对比。文中展示了在不同输入置信度（完全不确定、完全可信、部分可信）以及不同假设置信度下的置信度传播结果，证明了方法的可行性和对上下文/假设敏感性。

**⚠️ 局限性**

局限性包括：①仅覆盖支持与上下文两类关系，未完整扩展至所有SACM关系；②SL算子假设证据来源独立，实际中常有共享信息导致依赖性；③需要手工推导和量化条件意见，工作量大；④缺乏自动化工具与标准化的定量表达方法，难以在工业实践中快速推广。

---

## 344. GALAR-TemporalNet v2: Anatomy-Guided Dual-Branch Temporal Classification with Bidirectional Mamba and Dual-Graph GCN for Video Capsule Endoscopy -- after competition results

**arXiv ID:** 2605.22209 | [PDF](https://arxiv.org/pdf/2605.22209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 345. Zero-Shot Temporal Action Localization Through Textual Guidance

**arXiv ID:** 2605.22201 | [PDF](https://arxiv.org/pdf/2605.22201v1)

**作者:** Benedetta Liberatori `[一作]` (University of Trento), Elisa Ricci `[通讯]` (University of Trento)

**通讯引用:** 11722 | [OpenAlex ID](https://openalex.org/A5065059558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于视频级和帧级文本引导的零样本时序动作定位方法，利用生成式语言模型和场景三元组在推理时自适应预训练的视觉-语言模型。

**💡 创新点**

创新点在于将文本信息分为动作描述和场景三元组，并在测试时利用最大间隔损失对模型进行自适应，以无训练数据完成动作定位。

**🔧 技术方法**

使用的技术包括大型视觉-语言模型（SigLIP-SoViT）、生成式语言模型（Gemini 1.5-flash）、图像字幕模型、场景图解析、句子BERT嵌入、测试时自适应（max-margin loss + temporal smoothness）以及非监督的最大间隔学习。

**📊 数据集**

实验数据集为 THUMOS14 和 ActivityNet‑v1.3。

**📈 对比分析**

与基于训练和无训练的最先进方法相比，在无训练数据设置下实现了大约 +1.8% ~ +4.2% 的 mAP 提升，超越了如 T3AL 等主流无监督方法。

**⚠️ 局限性**

主要限制是依赖生成式模型的质量、对字幕的歧义性和噪声敏感，以及在极端长/短时序动作的定位仍存在挑战，整体性能尚未饱和。

---

## 346. Reinforced Graph of Thoughts: RL-Driven Adaptive Prompting for LLMs

**arXiv ID:** 2605.22195 | [PDF](https://arxiv.org/pdf/2605.22195v1)

**作者:** Manuel Noah Riesen `[一作]` (Bern University of Applied Sciences), Peter Alfred von Niederhäusern `[通讯]` (Bern University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文提出了“Reinforced Graph of Thoughts（GoT）”框架，并对相关概念（如LLM、RL、MDP、RAG等）以及常用工具（PyTorch、Gymnasium、Stable Baselines3等）进行了整理与说明。

**💡 创新点**

创新点在于将思维图（Graph of Thoughts）与强化学习相结合，提出了一种新的思维建模与决策框架；同时系统化列出了关键术语与工具。

**🔧 技术方法**

采用的技术主要是大语言模型（LLM）、强化学习算法（DQN、A2C、PPO等）、深度学习框架（PyTorch）、RL环境（Gymnasium）以及稳定基线实现（Stable Baselines3）。

**📊 数据集**

未使用任何具体数据集，文章仅为概念性整理与框架阐述。

**📈 对比分析**

由于缺乏实验与对比，本文未提供性能评估或与其他方法的比较。

**⚠️ 局限性**

主要局限在于缺乏实证验证、实验数据与性能评估，本文更多侧重于概念与工具梳理，而非算法实现与效果展示。

---

## 347. From Sequential Nodes to GPU Batches: Parallel Branch and Bound for Optimal $k$-Sparse GLMs

**arXiv ID:** 2605.22188 | [PDF](https://arxiv.org/pdf/2605.22188v1)

**作者:** Jiachang Liu `[一作]` (Cornell University), Andrea Lodi `[通讯]` (Jacobs Technion-Cornell Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种CPU–GPU混合的分支定界框架，用批量化GPU计算加速稀疏GLM的精确求解和Rashomon集合收集。

**💡 创新点**

创新点在于将节点级计算批量化、使用填充技术统一数据结构、在GPU上实现下界、重优化、分支变量选择以及Rashomon集合收集，使得GPU实现可模块化且显著加速。

**🔧 技术方法**

采用GPU矩阵乘、排序、PAVA、收集收缩、稀疏梯度计算、双重下界等并行计算技术，并在CPU侧管理树逻辑。

**📊 数据集**

使用合成高相关性线性和逻辑回归实例，以及真实数据集Santander（线性）和DOROTHEA（逻辑）。

**📈 对比分析**

与Gurobi、MOSEK、OKGLM对比，零最优性间隙，速度提升一至两位数，尤其在高维、强相关、节点数百万时显著优于现有方法。

**⚠️ 局限性**

局限在于对GPU显存和批量填充的依赖，批量大小选择影响性能；在极大节点数或非常稀疏情形下仍需进一步优化。

---

## 348. LLM-Metrics: Measuring Research Impact Through Large Language Model Memory

**arXiv ID:** 2605.22176 | [PDF](https://arxiv.org/pdf/2605.22176v1)

**作者:** Si Shen `[一作]` (Nanjing University of Science and Technology), Danhao Zhu `[通讯]` (Jiangsu Police Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了 LLM-Metrics，一种基于大型语言模型（LLM）参数记忆的论文影响度量指标。

**💡 创新点**

创新点在于将 LLM 的内存作为实时、跨学科、与引用计数无关的研究影响评估工具，并证明了其在多模型、多供应商下的可复制性。

**🔧 技术方法**

采用多选探针（标题识别、作者识别、方法识别、期刊信息识别）对 LLM 进行查询，并使用五级评分（正确、部分正确、拒绝、错误、幻觉）转化为 0–1 的记忆分数。

**📊 数据集**

使用 549 篇 2023–2024 年计算机科学论文（从 Semantic Scholar 获取引用计数）作为评估样本，覆盖多领域和不同影响力层级；同时评估了 17 款公开 LLM（0.5B–72B 参数，六大供应商）。

**📈 对比分析**

通过 Spearman 相关系数与实际引用计数对比，整体 ρ = 0.1495（p = 0.0004）；15/17 模型产生正相关，其中 9 个显著（p < 0.05），显著性一致性 88.2%；模型规模与预测力呈非单调关系，3B 模型表现最佳；不同供应商模型差异明显。

**⚠️ 局限性**

局限性包括：样本量相对有限且仅涵盖计算机科学领域；训练数据来源不透明，可能导致偏见；作者/机构知名度等因素混入记忆信号；多选探针形式受限，未覆盖开放式回忆；短期引用窗口可能低估长期影响；模型选择与调优对结果影响显著。

---

## 349. Decomposing Ensemble Spread in Lorenz '96 With Learned Stochastic Parameterizations

**arXiv ID:** 2605.22242 | [PDF](https://arxiv.org/pdf/2605.22242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 350. Structure Retention in Embedding Spaces as a Predictor of Benchmark Performance

**arXiv ID:** 2605.22202 | [PDF](https://arxiv.org/pdf/2605.22202v1)

**作者:** Amanda Myntti `[一作]` (University of Turku), Filip Ginter `[通讯]` (University of Turku)

**通讯引用:** 7724 | [OpenAlex ID](https://openalex.org/A5019929457)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 25 种现代嵌入模型在五个 MTEB 任务（检索、双语挖掘、成对分类、摘要）中的嵌入空间进行结构分析，提出并评估了两种结构性探测指标：k 近邻重叠率（Neighborhood Retention）和独立成分分析（ICA）峰值显著性（Gini 系数），并将其与任务得分相关联。

**💡 创新点**

创新点在于：①首次系统地将局部邻域保持度和线性差异显著性作为轻量化的评估工具，用以解释嵌入模型在不同任务上的表现差异；②通过 ICA 的峰值显著性与打乱实验揭示了模型对局部（成对）结构和全局（数据集级）结构的捕捉差异；③展示了提示（prompt）对结构保持度与任务性能关系的影响，提供了对提示设计的潜在洞见。

**🔧 技术方法**

采用的技术主要包括：k-最近邻重叠率计算、ICA（FastICA）提取线性方向、Gini 系数衡量 ICA 峰值显著性、Spearman 相关系数评估结构指标与 MTEB 得分的关联、随机打乱实验验证局部与全局信号。

**📊 数据集**

使用的数据集为 MTEB Benchmark 下的 ARCChallenge、WebFAQ、Tatoeba、RTE3、SummEval，涵盖英语单语、跨语种（13 种语言）和不同任务类型，数据集均提供成对文本（问题-答案、翻译对、摘要对等）。

**📈 对比分析**

对比方法：对每个模型–数据集组合计算 Neighborhood Retention 与 ICA 峰值 Gini，并与官方 MTEB 得分进行 Spearman 相关。结果显示：Tatoeba 任务的两种指标与得分的相关系数高达 0.9 以上，ARCChallenge 在提示下相关性提升；RTE3 与 WebFAQ 的相关性相对弱，提示对 RTE3 的影响有限。整体而言，结构指标可作为轻量化的性能预测手段，且对模型排名的解释力显著。

**⚠️ 局限性**

局限性包括：①仅适用于成对（paired）任务，无法覆盖聚类、多类分类等嵌入常见任务；②仅关注空间结构对性能的关联，未直接解释具体概念如何编码，缺乏因果性验证；③实验受限于模型可访问性与计算资源，部分模型–任务组合缺失；④相关性分析未能给出训练目标的具体改进策略，需进一步实验验证。

---

## 351. Decision-Aware Quadratic ReLU Replacement for HE-Friendly Inference

**arXiv ID:** 2605.22237 | [PDF](https://arxiv.org/pdf/2605.22237v1)

**作者:** Rui Li `[一作]` (Chinese Academy of Sciences), Weijie Miao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向完全同态加密（FHE）的决策感知二次 ReLU 替代方案，利用已训练的单隐藏层 MLP 或固定特征提取器后的 MLP 头，在线不进行重训练，只在离线阶段对 ReLU 进行二次多项式替换，并在加密推理过程中保持低乘法深度。

**💡 创新点**

核心创新在于将激活替换转化为决策一致性问题：通过把模型的线性头固定后，将每个样本映射到二维升维空间（$Q,H$）并构造正负凸包，利用正边距分离理论得到精确的可行系数；当正边距不可行时，引入约束凸包与软边距松弛，得到近似可行解；此外提供量化鲁棒性证书，确保在 CKKS 编码下保持决策不变。

**🔧 技术方法**

技术手段包括：FHE（CKKS）实现、凸几何（正边距分离、约束凸包）、二次规划与拉格朗日对偶、软边距 SVM 风格的四维升维线性约束、离线系数构造算法（凸包最近点、凸二次规划）以及开源库 Quad4FHE。

**📊 数据集**

实验使用的多种数据集涵盖：tabular 数据（AG News、BWC、CIFAR‑10/100、Diabetes、Otto、Shuttle、SST‑5），视觉特征（DINOv2 对 CIFAR‑100、FGVC‑Aircraft、Stanford‑Cars、Tiny‑ImageNet），以及文本特征（Qwen3‑Embedding 对 SIB‑200、MASSIVE、Banking77）。还对小样本校准池（1%–20%）进行了敏感性评估。

**📈 对比分析**

与固定区间多项式（Square、Remez‑7）以及后训练近似（OLA、Precise）进行比较。指标包括上机精度（Top‑1、macro‑F1）、与原 ReLU 的决策一致性、加密推理的误匹配数和 CKKS 延迟。Quad4FHE 在多数任务上保持或接近原模型精度，并在激活模块实现上实现 3.7–4.1× 的速度提升、整体 1.18–1.68× 的加密推理加速，相比 Remez‑7 深度更低（4 级）。

**⚠️ 局限性**

局限性包括：精确可行性仅在校准集上保证，难以保证未见样本；对单隐藏层或头部结构有限；软边距解虽能提升一致性，但在边距极小或样本分布变化时可能失效；依赖离线校准池的大小与代表性；仅在 CKKS 深度 4 内评估，未考虑更深网络或混合 FHE/MPC 协议的场景。

---

## 352. Audience Engagement with Arabic Women's Social Empowerment and Wellbeing: A Decadal Corpus

**arXiv ID:** 2605.22204 | [PDF](https://arxiv.org/pdf/2605.22204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 353. SGR-Bench: Benchmarking Search Agents on State-Gated Retrieval

**arXiv ID:** 2605.22219 | [PDF](https://arxiv.org/pdf/2605.22219v1)

**作者:** Ningyuan Li `[一作]` (Beijing University of Technology), Yun Ma `[通讯]` (Peking University)

**通讯引用:** 78826 | [OpenAlex ID](https://openalex.org/A5100369226)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个专注于“状态门控检索”（state‑gated retrieval）的基准Benchmark，并构建了100个专家策划的任务

**💡 创新点**

创新点在于把检索状态的维持作为评价维度，提供了约束引导与目标导向的任务对，揭示了现有agent在保持网站检索状态方面的瓶颈

**🔧 技术方法**

采用大型语言模型与CLI检索工具进行实验，并通过手工与LLM辅助的数据标注管道构建数据集；在评估中使用了GPT‑5.5、Claude、Gemini、Qwen等多款LLM以及Google Search AI、Gemini Deep Research、OpenAI Deep Research等商业产品

**📊 数据集**

使用了SgR‑Bench数据集（100个任务，覆盖12个公开数据生态、6个来源族），每个任务提供约束/目标对与结构化输出格式

**📈 对比分析**

通过与8个CLI‑LLM系统及3个商业系统的对比，测得最高Item‑F1为66.18%，Row‑F1显著低于此，表明检索状态丢失是主要问题

**⚠️ 局限性**

局限在于仅覆盖相对稳定的公开结构化网站，缺少动态内容，且轨迹级别评估和跨平台一致性仍待完善

---

## 354. EvoIR-Agent: Self-Evolving Image Restoration Agentic System via Experience-Driven Learning

**arXiv ID:** 2605.22208 | [PDF](https://arxiv.org/pdf/2605.22208v1)

**作者:** Kailin Zhuang `[一作]` (Sun Yat-sen University), Zhi Jin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 7946 | [OpenAlex ID](https://openalex.org/A5088162640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的图像恢复代理 EvoIR-Agent，利用多模态大语言模型与分层经验池进行工具选择、去除顺序及视觉质量的全流程决策，并通过自演化机制不断更新经验库。

**💡 创新点**

① 将视觉质量纳入经验组件；② 设计了三级分层经验池（洞察层、粗粒度层、细粒度层）以实现从全局到局部的逐步细化指导；③ 引入基于 Bradley–Terry–Davidson 统计模型和批量自演化机制，使经验随时间自适应提升；④ 结合 CLIP 编码器与 LLM 对图像模式进行检索与文本化，形成可持续更新的经验记录。

**🔧 技术方法**

多模态 LLM（如 GPT‑style 模型）+ CLIP 编码器；优先驱动映射函数 p；Bradley–Terry–Davidson 统计模型；自演化批处理更新；多级检索与模式描述；人工智能辩论（MAD）机制用于模式聚类。

**📊 数据集**

MiO100 合成数据集（1,440 张低质量图像，包含 16 种两三种退化组合）；FoundIR 真实场景数据集（低光+噪声、模糊+JPEG，各 50 张对）。

**📈 对比分析**

与六种 AiOIR 模型（AirNet、PromptIR、MiOIR、DA‑CLIP、InstructIR、AutoDIR）和四种基于代理的 IR 方法（AgenticIR、MAIR、4KAgent、TIR‑Agent）进行对比。EvoIR‑Agent 在 PSNR、SSIM、LPIPS、MANIQA、CLIP‑IQA、MUSIQ 上均超越对手，平均 PSNR 提升 2.35 dB，工具调用次数减少约 70%，实现性能与效率的 Pareto‑optimal 取舍。

**⚠️ 局限性**

仍需依赖已有工具集合，对未知或极端退化模式的泛化能力受限；自演化过程对计算资源和时间有一定开销；经验池的规模和更新频率影响最终性能，若数据来源不充分可能导致经验不足。

---

## 355. Skill Weaving: Efficient LLM Improvement via Modular Skillpacks

**arXiv ID:** 2605.22205 | [PDF](https://arxiv.org/pdf/2605.22205v1)

**作者:** Zhuo Li `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 116393 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SkillWeave 框架，将大模型拆分成域专用的轻量 SkillPack 并通过 SkillZip 压缩，实现在固定内存预算下多域性能提升。

**💡 创新点**

创新点：① 全参数自适应后再压缩，兼顾精度与效率；② SkillZip 的全量化（权重+激活）双平滑压缩技术；③ 共享 backbone 与动态 SkillPack 选择的模块化推理实现。

**🔧 技术方法**

采用自生成指令数据 + 规则筛选 + DPO 自监督 fine‑tune；模型合并提取共享知识；全量化低比特压缩与双平滑；动态路由推理。

**📊 数据集**

使用自生成的指令数据进行训练；评测基准包括 GSM8k、MATH、HumanEval、MBPP、AlpacaEval、BBH、ARC‑C 等通用任务；AgentBench（DB、OS、KG、WS、WB）用于 agent‑as‑model 场景。

**📈 对比分析**

与多类基线（Open‑source LLM、LoRA、Delta‑压缩、Self‑Specialization、Self‑Rewarding、模型合并、MoE、Multi‑Teacher Distillation 等）对比。9B/10B SkillWeave 在多域评测中超越 32B 单体模型；推理速度比 32B 快 4×，比 5×7B 快 5.5×，性能保持与对手相近。

**⚠️ 局限性**

局限：验证主要在明确域边界的任务，开创性或混合域场景缺乏评估；规则式自动验证不适用于开放式任务；缺乏自动化技能发现与验证机制。

---

## 356. OSS: Open Suturing Skills Vision-Based Assessment Challenge 2024-2025

**arXiv ID:** 2605.22200 | [PDF](https://arxiv.org/pdf/2605.22200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 357. Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset

**arXiv ID:** 2605.22186 | [PDF](https://arxiv.org/pdf/2605.22186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 358. Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles

**arXiv ID:** 2605.22177 | [PDF](https://arxiv.org/pdf/2605.22177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 359. What are the Right Symmetries for Formal Theorem Proving?

**arXiv ID:** 2605.22257 | [PDF](https://arxiv.org/pdf/2605.22257v1)

**作者:** Krzysztof Olejniczak `[一作]` (University of Oxford), İsmail İlkan Ceylan `[通讯]` (TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM基正式定理证明对语义等价变形的敏感性，提出重写范畴框架和成功不变性，设计测试时重写集成提升鲁棒性。

**💡 创新点**

以重写范畴形式化语义等价变形的对称性，定义证明等价性和成功不变性，并提出无需重训练的测试时重写集成实现。

**🔧 技术方法**

结合范畴理论的重写范畴、LLM定理证明器、采样+评分+选择的在线重写生成、成功不变性理论证明等技术。

**📊 数据集**

miniF2F-rw（miniF2F+语义等价变形）、ProofNet、Ineq-Comp 以及官方 miniF2F benchmark。

**📈 对比分析**

在固定推理预算下对DeepSeek-Prover-V2、Goedel-Prover-DPO/SFT等模型进行PASS@k比较；重写集成显著提升PASS@k并缩小训练与验证差距，甚至超过部分推理模型。

**⚠️ 局限性**

仅在非推理型LLM证明器上验证，未检验对推理模型的影响；重写集成依赖于简单能量函数，未使用更精确的成功预测；重写覆盖可能不完整。

---

## 360. A Generative Adversarial Graph Neural Network for Synthetic Time Series Data

**arXiv ID:** 2605.22215 | [PDF](https://arxiv.org/pdf/2605.22215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 361. Emergence of agriculture in an artificial society of reinforcement learning agents

**arXiv ID:** 2605.22256 | [PDF](https://arxiv.org/pdf/2605.22256v1)

**作者:** Gautier Hamon `[一作]` (Inria), Ricard Solé `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 25109 | [OpenAlex ID](https://openalex.org/A5046878011)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在一个基于强化学习的人工社会中，探索并证明了农业（通过种植、灌溉等生态工程行为）可在没有明确指令的情况下自发出现并演化为稳定的农耕状态。

**💡 创新点**

首次将多智能体强化学习与动态生态环境耦合，揭示了规划（延迟奖励）、社会脆弱性、社会学习防火墙以及锁定效应等四大机制是农业出现的通用驱动因素。

**🔧 技术方法**

使用多智能体强化学习（MARL）与Transformer动作策略、PPO优化、仿真生态环境及基于平均场的数学模型相结合。

**📊 数据集**

使用完全自定义的模拟数据集：30×30格网、三种植物（P1、P2、P3）与固定水源，模拟季节循环与种子萌发、扩散、灌溉等生态过程。

**📈 对比分析**

通过参数空间扫描（折扣因子γ与P3生长率）、行为指标（植物丰度、移动、灌溉、剔除竞争者）以及社交学习（克隆）对比实验，发现高γ低P3时农业易出现，社交学习能突破种群规模阈值并实现非线性繁殖，最终表现出显著的锁定与不可逆性。

**⚠️ 局限性**

局限性包括：未考虑资源存储与季节波动缓冲、植物无遗传变异、社交学习仅采用克隆方式缺乏教学与创新组合、模型对复杂人类与昆虫生态系统的外推性有限。

---

## 362. Direct content-based retrieval from music scores images

**arXiv ID:** 2605.22255 | [PDF](https://arxiv.org/pdf/2605.22255v1)

**作者:** Noelia Luna-Barahona `[一作]` (University of Alicante), Jorge Calvo-Zaragoza `[通讯]` (University of Alicante)

**通讯引用:** 2115 | [OpenAlex ID](https://openalex.org/A5085151278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究音乐乐谱图像的内容检索方法

**💡 创新点**

创新点在于构建通用查询数据集并系统比较传统OMR+符号检索、端到端Transformer以及微调多模态LLM三种方案

**🔧 技术方法**

采用的技术包括基于CTC-CRNN的OMR转写、端到端的Transformer（SMT变体）以及微调的PaliGemma 2多模态LLM

**📊 数据集**

使用四个真实和合成数据集：FMT‑C、FMT‑M、Malaga、PrIMuS，以及它们的多域组合

**📈 对比分析**

通过宏/微F1分数比较，OMR管线在域内表现最佳，端到端和LLM在域外更鲁棒，但总体F1仍受限于数据质量和模型泛化

**⚠️ 局限性**

主要限制包括OMR对转写错误敏感、端到端方法对大规模训练数据依赖强、LLM泛化差且训练成本高

---

## 363. IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions

**arXiv ID:** 2605.22247 | [PDF](https://arxiv.org/pdf/2605.22247v1)

**作者:** Kai Golan Hashiloni `[一作]` (Reichman University), Kfir Bar `[通讯]` (Reichman University)

**通讯引用:** 325 | [OpenAlex ID](https://openalex.org/A5069195550)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了IdioLink检索基准，旨在评估语言模型能否在字面与比喻/惯用表达之间检索概念等价的文档；

**💡 创新点**

创新点在于：①设计了四类文档与查询，标注核心语义span；②构建了10,700+文档、2,140+查询的规模化数据集；③系统比较了指令增强与span embedding对检索效果的影响，并揭示模型在概念匹配上的局限；

**🔧 技术方法**

技术主要包括：使用Gemini、GPT‑4o mini等LLM生成文本；dense retrieval embeddings（SBERT、Contriever、E5、InstructOR、GTE等）；span‑level embedding（late chunking）；指令增强（explicit span instruction）；并对比BM25；对模型进行零样本与对比学习 fine‑tuning；

**📊 数据集**

数据集为IdioLink，涵盖107个惯用表达，在10个领域生成10,700+文档、2,140+查询；其中3,570条为人工金标准；

**📈 对比分析**

评估采用R‑Precision和nDCG@10两指标；在零样本下，指令+span embedding提升约22点，最佳R‑Precision≈55、nDCG≈73；fine‑tuned模型在标准配置下进一步提升，但指令或span embedding在训练时往往不利；整体仍显著低于理想水平，说明模型难以跨表面形式匹配概念；

**⚠️ 局限性**

局限包括：仅覆盖英语和107个PIE，跨语言推广受限；文本多为人工生成，可能存在自然性与分布偏差；查询与文档均短小，缺乏真实检索场景的噪声、多跳推理；训练样本仅440条，模型可能过拟合数据特定模式。

---

## 364. Evaluation of Chunking Strategies for Effective Text Embedding in Low-Resource Language on Agricultural Documents

**arXiv ID:** 2605.22203 | [PDF](https://arxiv.org/pdf/2605.22203v1)

**作者:** Sovandara Chhoun `[一作]` (Chungbuk National University), Saksonita Khoeurn `[通讯]` (BigDataLabs Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了四种文本分块策略（递归字符分块、语言感知分块、句子分块和LLM分块）在低资源语言柬埔寨农业文件中的嵌入检索性能。

**💡 创新点**

首次系统比较并证明基于结构的递归字符分块（300字符）在BGE‑M3+FAISS检索框架中显著优于其他分块方法，提出了低资源语言文档检索的新策略。

**🔧 技术方法**

采用BGE‑M3多语种嵌入、FAISS近邻检索、5折交叉验证、L2距离、余弦相似度以及字符级Khmer覆盖率和IoU等评估指标。

**📊 数据集**

使用了18条柬埔寨农业问答对组成的人工标注数据集（从PDF转换为文本）。

**📈 对比分析**

通过与四种分块方法的对比，递归300字符分块在L2距离最低（0.4295±0.0461）、答案相关性最高（0.8663±0.0199）并通过t检验在L2距离上显著优于句子分块；LLM分块压缩率最高但性能次之。

**⚠️ 局限性**

研究仅关注农业手册、仅使用BGE‑M3模型、LLM分块参数未进行优化、缺乏人工评估以及跨领域推广的验证。

---

## 365. Ultra-High-Definition Image Quality Assessment via Graph Representation Learning

**arXiv ID:** 2605.22192 | [PDF](https://arxiv.org/pdf/2605.22192v1)

**作者:** Shaode Yu `[一作]` (Communication University of China), Qiurui Sun `[通讯]` (Beijing Normal University)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5079661373)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了基于图卷积网络的UHD图像无参考质量评估框架，使用网格对齐补丁采样、混合KNN图构建、残差GCN传播和门控注意力池化，结合EMA归一化多目标损失训练。

**💡 创新点**

创新点包括：将UHD图像无参考评估建模为图级回归问题，将补丁采样对齐至原始长宽比并以混合空间+特征距离构造KNN图，使用残差GCN和门控注意力读出，并在多目标损失中引入EMA归一化实现稳定优化。

**🔧 技术方法**

使用技术包括：ResNet-50特征提取、KNN图构建、残差GCN、门控注意力池化、EMA归一化多目标损失、PyTorch与torch-geometric。

**📊 数据集**

使用数据集：UHD-IQA基准（6073张4K照片）。

**📈 对比分析**

与传统DL、UHD挑战、基于图的BIQA方法对比，获得最佳RMSE 0.0519，SRCC 0.8019，PLCC 0.7784，表现优于大多数方法但略逊于SJTU/UIQA与GS-PIQA。

**⚠️ 局限性**

局限性包括：模型参数虽小但MACs极高，推理耗时长；使用ResNet-50等较弱backbone，缺乏更强多尺度语义、审美与显著性特征；图构造与超参数需要分阶段调优，未全面验证跨数据集泛化。

---

## 366. Evaluating Large Language Models as Live Strategic Agents: Provider Performance, Hybrid Decomposition, and Operational Gaps in Timed Risk Play

**arXiv ID:** 2605.22238 | [PDF](https://arxiv.org/pdf/2605.22238v1)

**作者:** H. C. Ekne `[一作]` `[通讯]`, H. C. Ekne

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Risk 游戏环境中，评估多家供应商的 LLM 作为实时决策代理，在固定时间、格式和胜利目标下进行跨供应商比赛、规划与执行拆分、成本门实验以及轨迹分析。

**💡 创新点**

创新点：① 将传统静态基准转化为实时循环评估，② 通过拆分规划与执行展示系统设计差异占主导；③ 结合成本与轨迹数据揭示目标跟踪、执行转化、运行时可靠性对表现的决定性影响。

**🔧 技术方法**

技术手段：Risk 引擎 + 统一语法 + 90 秒规划/执行/放置计时器；Gemini、OpenAI GPT‑5.1/5.5、Claude Sonnet4、Kimi K2.6 等多模型；统计检验（配对胜负检验、均值差异、p 值）及轨迹可视化分析。

**📊 数据集**

数据集：32 局跨供应商锦标赛、16 局 OpenAI 生成阶梯、16 局 Kimi 锚定、15 局成本门、32 局规划对决、946 局规划/执行轨迹，全部来自 Risk 游戏回合日志。

**📈 对比分析**

比较方法：使用胜率、配对胜负检验、均值差异检验（p 值）与成本估算。结果显示 Gemini 在全栈赛中 20/32 胜；OpenAI 6/32；拆分规划后 32 局平等 p≈0.82；成本门混合方案将成本降低约 50% 仍保留大部分优势。

**⚠️ 局限性**

局限性：仅在单一游戏、单一提示语法、单一计时设置下实验；轨迹分析为观察性；无法保证结果推广至所有代理场景；规划层排名不确定；Kimi 时间滞后估计为近似。

---

## 367. Holomorphic Neural ODEs with Kolmogorov-Arnold Networks for Interpretable Discovery of Complex Dynamics

**arXiv ID:** 2605.22235 | [PDF](https://arxiv.org/pdf/2605.22235v1)

**作者:** Bhaskar Ranjan Karn `[一作]`, Dinesh Kumar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种“Holomorphic KAN‑ODE”框架，用可学习的B‑spline边激活函数和Cauchy–Riemann正则化，直接对复杂平面上的自守动力学进行可解释建模。

**💡 创新点**

创新点在于把Kolmogorov–Arnold网络（KAN）嵌入神经ODE中，并通过可微正则化强制满足复分析的Cauchy–Riemann条件，使模型既保持可解释性（可回归符号方程），又具备结构先验。

**🔧 技术方法**

核心技术包括：1）KAN架构的B‑spline边激活；2）神经ODE求解器与adjoint梯度；3）Cauchy–Riemann正则化与线性warmup；4）自动符号回归（spline‑to‑formula fitting）。

**📊 数据集**

实验数据集涵盖六类纯粹的复映射（如z²+c、z³+c、e^z+c、sin(z)+c、cos(z)+c、z·e^z+c）以及潜流模型Uz+Ua²/z。

**📈 对比分析**

与16倍参数的MLP基线对比，Holomorphic KAN‑ODE在R²>0.95、Julia集边界一致率最高达98%、噪声鲁棒性仅4%降解、迁移学习提升90.4%等方面表现优异，且参数量仅280。

**⚠️ 局限性**

局限性包括：1）对奇点或有理形式（如潜流）拟合不足；2）仅处理二维自守系统；3）符号回归受预设库限制；4）训练数据多为合成；5）在极端噪声或超域推断时仍需验证。

---

## 368. Temporal Coding as a Substrate for Sensorimotor Object Inference: A Spiking Reinterpretation of Thousand Brains Architecture

**arXiv ID:** 2605.22206 | [PDF](https://arxiv.org/pdf/2605.22206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 369. ARC-STAR: Auditable Post-Hoc Correction for PDE Foundation Models

**arXiv ID:** 2605.22222 | [PDF](https://arxiv.org/pdf/2605.22222v1)

**作者:** Chengze Li `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 136928 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出ARC-STAR框架，利用预训练的PDE基础模型（如Poseidon）进行后期修正；先用全局校正器去除大尺度误差，再用块级局部细化器在每个块内进行精细修正，并在部署时通过无标签风险评分在预算内选择高风险块；

**💡 创新点**

创新点在于：①冻结预训练模型不做再训练；②两阶段可审计的修正结构（全局+局部）可分离贡献；③局部细化采用halo‑read、center‑write块级接口，能够在任意计算预算下使用同一模块；④无标签风险评分可在部署时自适应路由，避免了昂贵的训练或额外标签；

**🔧 技术方法**

技术方法包括：全局残差校正器（TFNO）、局部ConvNeXt式块级细化器、汉宁窗平滑、基于速度创新与能量梯度的无标签评分、自动回归训练与预算路由；

**📊 数据集**

使用的基准数据集为五种不可压Navier–Stokes流（NS‑SL、KF、NS‑PwC、NS‑G、NS‑Sines），每种分为两种粘度（moderate、extreme），共十个测试细胞；还在DPOT‑Ti上验证跨主机迁移；

**📈 对比分析**

与13种外部基线（密集神经算子、迭代细化、后期修正、测试时自适应）以及九种路由策略进行对比；ARC‑STAR在十个细胞中平均至少将10步速度误差降低36倍，7/10细胞最优，优于参数高效微调且在共享计算预算下始终位于最优或近优位置；

**⚠️ 局限性**

局限性：仅在二维不可压、周期边界、速度通道修正；不适用于压缩性流、三维或非周期域、长滚动期以及需要主机参数修改的方法；未来需扩展至更广泛场景和更大规模基准。

---

## 370. Can Transformers Learn to Verify During Backtracking Search?

**arXiv ID:** 2605.22221 | [PDF](https://arxiv.org/pdf/2605.22221v1)

**作者:** Yin Jun Phua `[一作]` (Institute of Science Tokyo), Katsumi Inoue `[通讯]` (National Institute of Informatics)

**通讯引用:** 3862 | [OpenAlex ID](https://openalex.org/A5080458729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了在后向搜索（backtracking search）中，基于 Transformer 的模型是否能学习只依据当前搜索状态做出继续或回溯的判断，并提出了两个结构性修正来消除模型对搜索历史的依赖。

**💡 创新点**

创新点在于①将搜索轨迹序列化并通过“局部化（localization）”把状态信息集中到当前决策块；②设计了“选择性状态注意力（Selective State Attention, SSA）”掩码，使模型在做出判断时仅能看到当前状态和问题前缀，从而保证了状态等价性（state‑equivalence）并消除了历史纠缠（history entanglement）。

**🔧 技术方法**

技术包括：decoder‑only Transformer 结构（6 层、256 维、8 头），基于块相对位置编码的注意力掩码，累计轨迹（cumulative trace）与状态重建（state‑rebuilt）两种推理协议，以及对比实验中的直接指标（KL、AUROC、解算率）和诊断方法（history transplant、verifier‑only 评测）。

**📊 数据集**

使用的基准数据集包括：3‑SAT（n=50、n=75）、图着色（30 节点、4 颜色、边概率 0.35）、Blocks World 以及基于模糊表达式的回溯解析，所有实例均通过随机生成器并植入已知解。

**📈 对比分析**

比较方法：将 SSA 与默认因果注意力（causal）Transformer 进行对比，分别在累计推理和状态重建推理两种协议下评估。结果显示：在状态重建协议下，SSA 的求解率保持在 95%+（接近 MLP/图神经网络基准），而因果 Transformer 则显著下降；在累计推理下 SSA 也优于因果 Transformer，且在不同任务（SAT、图着色、Blocks World、解析）上表现一致，证明了 SSA 的普适性。

**⚠️ 局限性**

局限性：①只验证了被动回溯（reactive verification），未处理主动死角检测（proactive verification）；②SSA 依赖于轨迹的局部化格式，若无法提供明确的状态块则无效；③在极长或更复杂的搜索树上，位置外推（positional extrapolation）仍可能导致微小误差；④实验仅在中等规模实例上进行，未展示对极大规模实例或真正的符号求解器的效果。

---

## 371. CLORE: Content-Level Optimization for Reasoning Efficiency

**arXiv ID:** 2605.22211 | [PDF](https://arxiv.org/pdf/2605.22211v1)

**作者:** Yuyang Wu `[一作]` (Carnegie Mellon University), Olexandr Isayev `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18649 | [OpenAlex ID](https://openalex.org/A5011932992)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种内容级优化框架CLORE，用外部编辑模型在强化学习后期对已成功生成的推理轨迹进行删减，去除重复、不可读或无关的推理片段，并通过增强的原始-编辑对进行偏好学习，提升推理效率。

**💡 创新点**

创新点在于：①把推理质量从单纯的长度控制转向内容级优化；②利用外部LLM进行局部删减，保持与原始轨迹分布的相似性；③将参考自由DPO与策略梯度联合使用，兼容多种长度控制方法，实现内容与长度双重优化。

**🔧 技术方法**

核心技术包括：强化学习策略梯度（PPO/GRPO等）、Direct Preference Optimization (DPO) 无参考式偏好学习、外部编辑模型（如Qwen3-4B）实现推理轨迹的局部删除，以及与已有长度控制方法（GRPO、DAPO、Training Efficient、ThinkPrune）的融合。

**📊 数据集**

使用的训练集为DapO-Math-17K（约17K数学题目），评测数据集包括OlympiadBench、Minerva、MATH500、AMC2023、AIME2025。模型基于DeepSeek‑R1‑Distill‑Qwen‑7B和Qwen2.5‑Math‑7B。

**📈 对比分析**

通过与GRPO、DAPO、Training Efficient、ThinkPrune等基线的对比实验，CLORE在保持或提升准确率的同时显著缩短推理长度，AE（准确率-效率）分数平均提升约0.4–1+点；在Qwen2.5‑Math‑7B上更显著，平均提升超过1点。实验表明CLORE与长度控制方法兼容，可进一步提升推理效率。

**⚠️ 局限性**

局限性：①依赖外部编辑模型的质量，若编辑错误可能导致重要信息丢失；②实验仅在数学推理任务上验证，跨任务泛化需进一步评估；③引入编辑步骤增加额外算力和工程成本；④对长文本或不同语言的适用性尚未测试。

---

## 372. Bandit Convex Optimization with Gradient Prediction Adaptivity

**arXiv ID:** 2605.22191 | [PDF](https://arxiv.org/pdf/2605.22191v1)

**作者:** Shuche Wang `[一作]` (National University of Singapore), Vincent Y. F. Tan `[通讯]` (National University of Singapore)

**通讯引用:** 5025 | [OpenAlex ID](https://openalex.org/A5058345431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

在该论文中，作者提出了一种基于双阶段反馈的强化学习框架，并针对单智能体和多智能体的情形，设计了新的在线学习算法；

**💡 创新点**

创新点在于引入了“单阶段预测-双阶段估计”的结构，能够显著降低估计方差，实现了对梯度残差的自适应控制；

**🔧 技术方法**

主要技术包括基于自适应学习率的优化梯度下降（Optimistic Mirror Descent）、二次逼近法（variance reduction）以及多专家元学习策略；

**📊 数据集**

实验中主要使用了合成数据集（synthetic data），并在这些数据上验证了算法的有效性；

**📈 对比分析**

与传统的单阶段反馈方法比较，实验结果显示该算法在收敛速度和最终性能上均优于基线，尤其在高维场景下表现更为显著；

**⚠️ 局限性**

该方法的局限性在于需要已知梯度预测的可观测成分，并且对域的几何假设（如凸、有限直径）要求较高，且在非平稳环境下的鲁棒性尚未完全验证。

---

## 373. No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians from Unposed Multi-View Videos

**arXiv ID:** 2605.22190 | [PDF](https://arxiv.org/pdf/2605.22190v1)

**作者:** Matteo Balice `[一作]` (Politecnico di Milano), Sungwhan Hong `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 NoPo4D 系统，能够在单次前向推理中同时处理未知相机位姿、动态场景和多视角输入，实现实时 4D 场景重建。

**💡 创新点**

创新点包括：① 将高斯运动分解为像素平面位移与深度变化，直接利用光流监督；② 双向运动编码器实现跨视角与跨时间特征聚合；③ 视角相关的 SH 不透明度用于缓解跨视角和跨时间的不对齐误差。

**🔧 技术方法**

采用预训练的 Depth Anything 3 backbone、冻结的相机/深度头、4D 高斯 splatting、双向运动编码器、视角相关 SH 不透明度以及可选的后期优化。

**📊 数据集**

训练使用 ~2900 个 Ego-Exo4D 场景；在 ExoRecon、Immersive Light Field、Kubric、N3DV 四个多视角动态基准上进行评估。

**📈 对比分析**

与现有 feed-forward 及 per-scene 优化方法进行对比；在所有基准上 NoPo4D 的 PSNR/SSIM/LPIPS 均优于其它前向方法，并在后期优化后超越 MonoFusion 等传统优化方案，速度提升数十倍。

**⚠️ 局限性**

局限性：假设相机阵列保持静止；依赖预训练模型提供的伪 ground-truth，受其精度限制；仅适用于同步多视角输入，无法直接扩展到单目或异步场景；运动建模仅为一阶。

---

## 374. Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments

**arXiv ID:** 2605.22189 | [PDF](https://arxiv.org/pdf/2605.22189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 375. Action with Visual Primitives

**arXiv ID:** 2605.22183 | [PDF](https://arxiv.org/pdf/2605.22183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 376. Enhancing Multimodal Large Language Models for Safety-Critical Driving Video Analysis

**arXiv ID:** 2605.22185 | [PDF](https://arxiv.org/pdf/2605.22185v1)

**作者:** Tomaso Trinci `[一作]` (Verizon Connect), Leonardo Taccari `[通讯]` (Verizon Connect)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态流水线，将下采样视频帧与同步的IMU/GPS以及专用视觉模型的语义信息融合，以生成高质量的伪标签（描述性字幕和问答对），用于训练多模态大型语言模型识别和描述安全关键驾驶事件。

**💡 创新点**

创新点在于：①将车载传感器高频物理数据与视频同步映射并与语义层融合；②利用专家模型生成的语义信息作为教师模型的高质量伪标签；③通过DoRA低秩适配器在参数极低的情况下实现对SCE的强学习并实现迁移。

**🔧 技术方法**

使用技术包括：多模态同步处理（IMU与视频对齐），专用语义模型（碰撞检测、急刹车分类等），知识蒸馏/伪标签生成，DoRA低秩适配器微调，IMU dropout策略。

**📊 数据集**

使用的数据集：私有的7000条美国车载视频+遥测数据、Nexar 1500条SCE样本、BDD-X、LingoQA（用于泛化评测）。

**📈 对比分析**

与六个基线（Amazon Nova Pro、Claude Sonnet 4.5、Molmo 2、MiniCPM-V 4.5、InternVL 3.5、QwenVL 3.5）以及原始QwenVL-2.5在多项任务（字幕、闭合QA、三分类SCE、二分类SCE）进行比较。结果显示：在SCE字幕和分类任务上，DoRA适配的Qwen2.5-VL-7B在ROUGE‑L 0.44、BERTScore 0.51、SCE三分类准确率87.7%以及二分类准确率90%时优于所有基线，显著提升。

**⚠️ 局限性**

限制：需要同步的IMU/GPS数据；伪标签生成依赖专家模型；输入格式仍为文本序列，对高频时序信息的表达有限；仅在有传感器支持的镜头上可用，未能覆盖无传感器的普通行车摄像。

---

## 377. IKNO: Infinite-order Kernel Neural Operators

**arXiv ID:** 2605.22182 | [PDF](https://arxiv.org/pdf/2605.22182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation

**arXiv ID:** 2605.22252 | [PDF](https://arxiv.org/pdf/2605.22252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 379. How Many Different Outputs Can a Transformer Generate?

**arXiv ID:** 2605.22223 | [PDF](https://arxiv.org/pdf/2605.22223v1)

**作者:** Maxime Meyer `[一作]` (National University of Singapore), Vincent Y. F. Tan `[通讯]` (National University of Singapore)

**通讯引用:** 5025 | [OpenAlex ID](https://openalex.org/A5058345431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer在有限数值精度与嵌入空间几何约束下，可生成的不同序列的上限；提出可访问序列（accessible sequence）概念，并给出其最大长度随提示长度线性增长、以及超过临界长度后大多数序列不可访问的理论上界；通过实验验证这些结论。

**💡 创新点**

首次将数值精度与嵌入空间几何结合，提出可访问序列理论框架，给出精确的线性阈值上限，并用该框架解释Transformer在复制、cramming任务中的失效；同时提供了可直接用来预测模型表现的闭式公式。

**🔧 技术方法**

使用包络数（packing number）与Wasserstein精度的理论分析；基于mean‑field Transformer的概率度量；在实验中优化软提示进行cramming；利用复制任务评估可访问阈值；对嵌入空间使用椭球和锥形包络估计；对单词单元体积分布进行卷积估计。

**📊 数据集**

PG19文本片段、随机均匀采样的词序列；对Pythia、Qwen‑2.5、Llama‑3.2、Gemma‑3等多种预训练Transformer进行实验；通过采样10K随机提示估计嵌入空间形状。

**📈 对比分析**

用sigmoid拟合cramming成功率随序列长度变化的曲线，并提取n₅₀（成功率下降至50%时的长度）；将理论上界的斜率与实验得到的斜率比值，发现理论上界在5~10倍以内；复制任务中观察到精确复制准确率随长度急剧下降，吻合理论预测。

**⚠️ 局限性**

理论假设最坏情况：嵌入空间完全填充球形、各token对应的区间体积相等；对数值精度假设有限，未考虑隐藏状态可能发散的情况；cramming实验在长提示时成本高；结果主要针对标准Transformer，未涵盖所有新型架构。

---

## 380. REACH: Hand Pose Estimation from Room Corners

**arXiv ID:** 2605.22231 | [PDF](https://arxiv.org/pdf/2605.22231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 381. GHI: Graphormer over Conditioned Hypergraph Incidence for Aspect-Based Sentiment Analysis

**arXiv ID:** 2605.22228 | [PDF](https://arxiv.org/pdf/2605.22228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 382. Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL

**arXiv ID:** 2605.22217 | [PDF](https://arxiv.org/pdf/2605.22217v1)

**作者:** Sophia Xiao Pu `[一作]` (University of California, Santa Barbara), Xin Eric Wang `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 10258 | [OpenAlex ID](https://openalex.org/A5100327844)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在自我对弈强化学习框架中引入数据门控和奖励对比，系统研究了语言模型在自生成任务上的稳定性，发现数据门控是保持训练不崩溃的关键因素。

**💡 创新点**

创新点在于揭示奖励与数据门控的非对称作用，提出“Grounded Proposer Paradox”以及通过连续门控强度分析自我对弈的相变现象，证明数据级过滤才是限制崩溃的根本约束。

**🔧 技术方法**

使用了可验证奖励的自我对弈（GRPO）、Python和确定性DSL的输出预测任务、基于执行器的奖励（grounded）与自洽奖励（intrinsic）、数据门控函数以及连续泄漏率ε的实验设计。

**📊 数据集**

实验数据集包括基于Python的代码输出预测任务、15操作符DSL对照任务，以及CRUXEval-O/I、HumanEval+、MBPP+等公开评测集，预训练模型为基准。

**📈 对比分析**

通过构建七种不同奖励与门控组合的实验矩阵，比较验证准确率、训练-验证间的Intrinsic‑Grounded Gap等指标；结果显示，严格门控下无论奖励如何均能保持高达0.71的验证准确率，而无门控则在所有奖励设定下崩溃至0.00，证明门控的决定性作用。

**⚠️ 局限性**

局限性在于需预先设定极端严格的数据门控，未解决提议者容量瓶颈，实验仅覆盖确定性任务环境，对更复杂或噪声环境的推广仍有待验证。

---

## 383. Kernel-Based Safe Exploration in Deep Reinforcement Learning

**arXiv ID:** 2605.22207 | [PDF](https://arxiv.org/pdf/2605.22207v1)

**作者:** Rupak Majumdar `[一作]` (Max Planck Institute for Software Systems), Sadegh Soudjani `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 1976 | [OpenAlex ID](https://openalex.org/A5017334634)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Kernel‑Based Safe Exploration (KBSE) 算法，在线学习安全控制策略并同时构造基于核均值嵌入的障碍函数；

**💡 创新点**

创新点在于利用条件均值嵌入把障碍函数的期望约束转化为线性规划，支持在未知随机系统中使用概率阈值的安全约束，无需模型或海量数据；

**🔧 技术方法**

使用了 RKHS、核均值嵌入、线性回归、局部线性动力学估计以及深度强化学习框架 OMNISAFE；

**📊 数据集**

在 Gym 与 Mujoco 的连续控制 benchmark（如 SafetyPendulum、SafetyMountainCar、SafetyAnt 等）上进行实验；

**📈 对比分析**

与 DDPG+Lagrangian、SAC+Lagrangian、DDPG+PID‑Lagrangian 等基线对比，KBSE 在奖励更高、成本更低的同时提供安全概率下界，训练时间与安全违规次数呈正相关；

**⚠️ 局限性**

局限性包括训练时间随违规次数显著增长，低维环境提升有限，核函数选择与样本量对性能影响大，仅考虑概率阈值安全约束，未涵盖时序行为。

---

## 384. Modeling Pathology-Like Behavioral Patterns in Language Models Through Behavioral Fine-Tuning

**arXiv ID:** 2605.22356 | [PDF](https://arxiv.org/pdf/2605.22356v1)

**作者:** Nicola Milano `[一作]` (University of Naples Federico II), Davide Marocco `[通讯]` (University of Naples Federico II)

**通讯引用:** 1747 | [OpenAlex ID](https://openalex.org/A5027259197)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在大型语言模型上进行行为细调，强制模型在多种情景中持续选择与抑郁或偏执相关的非适应性行为，并观察其对生成分布的影响。

**💡 创新点**

提出了“行为诱导”框架，能够通过对行为决策的持续优化，在不依赖显式身份提示的情况下在模型内部诱发与心理疾病对应的长期行为与语言偏差。

**🔧 技术方法**

使用LoRA（低秩适配）在LLama‑3‑8B‑Instruct与Qwen‑2.5‑14B‑Instruct上进行监督细调，结合自回归语言建模损失。

**📊 数据集**

合成数据集：基于DSM‑5标准生成的“抑郁行为集”和“偏执行为集”，每个约1,000条样本，包含情境、两种适应性选择与两种非适应性选择。

**📈 对比分析**

对比方法包括：基线（无细调）、适应性细调、随机选择和纯负面细调；使用KL、JSD、概率质量分布、以及标准心理测量量表（BDI、GPTS、DASS）作为评估指标，结果显示抑郁细调模型在抑郁测量上显著高于基线，偏执细调模型在偏执测量上显著高于基线，且两者在其他测量上保持低值，证明行为细调产生了特异且可区分的病理性偏差。

**⚠️ 局限性**

局限包括：数据为模型生成的合成情境，缺乏真实临床语料的生态效度；未对内部表示机制进行直接解释；长期稳定性与可逆性未知；对安全机制的影响仍需进一步评估。

---

## 385. Sibyl-AutoResearch: Autonomous Research Needs Self-Evolving Trial-and-Error Harnesses, Not Paper Generators

**arXiv ID:** 2605.22343 | [PDF](https://arxiv.org/pdf/2605.22343v1)

**作者:** Chengcheng Wang `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22181 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了自演化的自动研究框架 Sibyl‑AutoResearch，并在文件化工作空间中实现了可审计的试验到行为、试验到 harness 的转换单元，形成 agent‑harness 的协同进化；

**💡 创新点**

提出并验证了两类可审计的转换单元以捕捉试验经验，并针对自动研究中六种经验丢失模式设计了七个正向 harness 功能，实现了从试验到决策、验证、声明、调度、批评、写作及 harness 自我演化的完整路径；

**🔧 技术方法**

使用 LLM 驱动代理、文件化工作空间、任务调度器、质量门控、验证任务、资源策略与自愈机制，结合日志追踪、反射与演化记录，实现了可审计的决策链；

**📊 数据集**

主要基于内部 AI/ML 研究项目（如 sparse‑autoencoder、diffusion‑language‑model、动态权重衰减等）的工作空间日志和生成的评审数据；未使用公开数据集；

**📈 对比分析**

通过回顾性审计而非对照实验，发现 8 个高置信转换事件，平均延迟 1 迭代、最大 3 迭代，展示了框架的可审计性；未与其他系统进行性能对比；

**⚠️ 局限性**

验证仅在作者自建 harness 上完成，审计为手工标注，使用自然失败而非注入，评审为生成而非真实同行评审，缺乏公开基准，未来需独立验证与更系统的对比实验。

---

## 386. Physics-Informed Generative Solver: Bridging Data-Driven Priors and Conservation Laws for Stable Spatiotemporal Field Reconstruction

**arXiv ID:** 2605.22338 | [PDF](https://arxiv.org/pdf/2605.22338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 387. A First Measurement Study on Authentication Security in Real-World Remote MCP Servers

**arXiv ID:** 2605.22333 | [PDF](https://arxiv.org/pdf/2605.22333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 388. PIU: Proximity-guided Identity Unlearning in ID-Conditioned Diffusion Models

**arXiv ID:** 2605.22311 | [PDF](https://arxiv.org/pdf/2605.22311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 389. PACT: Reducing Alert Fatigue in Low-Prevalence SOC Streams with Triggered Active Learning

**arXiv ID:** 2605.22324 | [PDF](https://arxiv.org/pdf/2605.22324v1)

**作者:** Samuel Ndichu `[一作]` (National Institute of Information and Communications Technology), Daisuke Inoue `[通讯]` (National Institute of Information and Communications Technology)

**通讯引用:** 9499 | [OpenAlex ID](https://openalex.org/A5028783978)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在低出现率SOC警报流中，提出PACT框架，将冻结的XGBoost-Focal屏幕器与基于分数平移的ADWIN触发器和混合查询规则结合，实施触发式主动学习以降低误报负担。

**💡 创新点**

创新点在于将ADWIN分数平移触发与阈值相对不确定性采样与高分抽样两部分混合的查询策略相结合，形成Pareto-aware的操作点选择，并通过匹配触发对比消除触发时序对性能的影响。

**🔧 技术方法**

技术包括：XGBoost-Focal作为冻结模型；ADWIN分数平移触发器；阈值不确定性+高分抽样的混合查询规则；热启动增量树更新；离线流式模拟器与滚动窗口指标评估。

**📊 数据集**

使用公开的低出现率SOC警报基准AIT-ADS和BOTSv1进行评估。

**📈 对比分析**

通过与冻结、周期性更新、ADWIN随机更新四种策略对比，PACT在两组数据上实现FP/1M benign下降约43%（AIT-ADS）和约21%（BOTSv1），正窗口召回仅下降约10个百分点，且查询量显著减少，显示出较优的误报降低与成本平衡。

**⚠️ 局限性**

局限性包括：仅针对两种低出现率基准，未验证高出现率或不同攻击模式；ADWIN触发仅基于分数平移，无标签漂移确认；批量大小、冷却时间等超参未做系统性敏感性分析；离线模拟未评估实时推理延迟与在线部署的实际效果。

---

## 390. Cross-domain benchmarks reveal when coordinated AI agents improve scientific inference from partial evidence

**arXiv ID:** 2605.22300 | [PDF](https://arxiv.org/pdf/2605.22300v1)

**作者:** Fiona Y. Wong `[一作]`, Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 54127 | [OpenAlex ID](https://openalex.org/A5011504360)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种跨学科的评估框架，用来比较协调化科学代理与传统单一方法在四个不同科学任务中的表现，涵盖分子音频化、范式转变检测、向量传播疾病出现预测和外行星候选验证。

**💡 创新点**

核心创新在于将“证据通道整合、可审计的中间工件与公共调查记录”与“冻结评测面板、基线对照与零控制”结合，形成可复现、可跨领域比较的“监管型多代理科学工作流”，并据此绘制了分布式证据、主导通道与表征映射三大运行模式。

**🔧 技术方法**

技术实现基于ScienceClaw × Infinite平台，利用内容地址化工件、可视化DAG、LLM推理器和领域专属工具（RDKit、music21、OpenAlex等）完成证据收集、特征变换、复合评分与统计检验。

**📊 数据集**

使用了四个数据集：16个药物化合物+6位作曲家（音频化）、16个历史范式转变+16个控制、12个向量病媒出现事件+12个稳定内在对照、12个确认行星+12个假阳性候选，所有数据均为冻结、公开可复现。

**📈 对比分析**

对比方法包括单通道脚本基线、单代理综合基线、通道消融与随机置换测试；在分布式证据任务上，联合通道实现AUROC分别达0.944和0.955，显著优于单通道；在主导通道任务上未获性能提升；在表征映射任务上虽检索表现不佳，但同类近邻一致性显著高于基线，说明结构恢复得到验证。

**⚠️ 局限性**

主要局限包括：样本量有限、评测面板仅为后向回顾、使用确定性流水线，未检验前瞻性预测性能；以及在不同领域的通用性和可扩展性仍需进一步验证。

---

## 391. Detection of Virus and Small Cell Patches in Foci Images Using Switchable Convolution and Feature Pyramid Networks

**arXiv ID:** 2605.22290 | [PDF](https://arxiv.org/pdf/2605.22290v1)

**作者:** Amrita Singh `[一作]`, Snehasis Mukherjee `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

改进了YOLOv2，结合FPN和可切换空洞卷积，实现了对病毒斑点和细胞斑点的精确检测和计数。

**💡 创新点**

创新点在于将FPN与可切换空洞卷积并入YOLOv2，以提升多尺度特征表达和自适应感受野。

**🔧 技术方法**

使用的技术包括YOLOv2骨干Darknet-19改造、特征金字塔网络、可切换空洞卷积以及多尺度训练。

**📊 数据集**

使用了病毒斑点（FFU）图像数据集和小细胞斑点图像数据集。

**📈 对比分析**

通过与YOLOv2、YOLOv7、SAC_YOLOv2等模型对比，实验显示在FFU斑点检测mAP达68.81%（YOLOv2_FPN_Switch），在小细胞斑点检测mAP达40.55%，均优于基线。

**⚠️ 局限性**

局限性在于对更复杂或更大规模数据集的泛化能力尚未验证，模型对极小或模糊目标的召回仍有提升空间。

---

## 392. Adaptive Measurement Allocation for Learning Kernelized SVMs Under Noisy Observations

**arXiv ID:** 2605.22275 | [PDF](https://arxiv.org/pdf/2605.22275v1)

**作者:** Artur Miroszewski `[一作]` `[通讯]` (European Space Agency), Artur Miroszewski (European Space Agency)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对噪声量子核矩阵的自适应测量分配方法，用于训练核化支持向量机（SVM），通过多轮迭代不断调整测量预算以提高分类器精度。

**💡 创新点**

创新点在于将几何敏感度（即核矩阵单个条目对分类器间距的影响）与活跃集不稳定性（支持向量状态变化的概率）结合，形成任务感知的分配指标；并引入早停机制与离散采样实现高效的多轮自适应测量。

**🔧 技术方法**

核心技术包括：核矩阵的贝尔努利测量模型、梯度式敏感度分析、支持向量不确定性估计、基于权重的最优分配公式、整数化多项式采样以及基于双重系数稳定性的早停判据。

**📊 数据集**

实验使用合成高斯混合数据集、经典 RBF 核以及从印度平原（Indian Pines）高光谱数据通过块编码映射得到的真实量子核矩阵，覆盖 2 到 100 个 qubit 的规模。

**📈 对比分析**

与传统均匀分配的基准相比，自适应方法在固定测量预算下显著降低支持向量重建误差、决策函数 RMSE 和间距误差；在早停情形下，仅使用原预算的 15–20% 即可获得与均匀分配相当甚至更优的性能。

**⚠️ 局限性**

局限性包括：需先行估计敏感度权重，噪声相关性模型假设简化；在权重高度均匀（低结构）或浓缩极端的量子核场景下，适应性优势减弱；多轮训练增加经典计算负担，需权衡量子测量成本与训练开销。

---

## 393. Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation

**arXiv ID:** 2605.22350 | [PDF](https://arxiv.org/pdf/2605.22350v1)

**作者:** Fabian Morelli `[一作]` (University of Tübingen), Stephan Eckstein `[通讯]` (University of Tübingen)

**通讯引用:** 190 | [OpenAlex ID](https://openalex.org/A5047214859)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出部分融合（Partial Fusion）方法，将模型融合与集成的权重平均相结合，允许在层内仅融合相似神经元，获得灵活的模型规模与性能折衷；同时将该方法视为对集成模型的泛化修剪（Generalized Pruning），并给出基于聚类的实现。

**💡 创新点**

创新点在于：①使用部分最优传输（Partial Optimal Transport）仅匹配最相似的神经元，从而在保持低参数量的同时提升性能；②将模型融合与修剪统一为一种框架，允许线性组合而非仅删除神经元；③提供多层全局匹配的可行策略，并对比不同匹配方法（固定点、贪心、激活）。

**🔧 技术方法**

核心技术包括：神经网络权重聚合、部分最优传输、聚类（K‑means/层级聚类）、固定点迭代与贪心匹配、层间特征映射（权重矩阵或激活向量）、大规模张量操作。

**📊 数据集**

使用标准基准数据集：MNIST（单层 MLP）、CIFAR‑10（ResNet‑18、VGG‑11、CIFAR‑10 CNNs）进行实验。

**📈 对比分析**

与纯权重平均、完整集成、无结构修剪以及基于聚类的修剪进行对比。实验表明：- 在层内仅保留少量孤立神经元即可显著提升精度；- 部分融合在参数量 1.45× 原模型时达到接近集成的准确率；- 基于聚类的泛化修剪在单模型压缩时比无结构修剪与 OT Fusion 后处理取得更优的精度/参数折衷；- 在有限数据微调后，部分融合模型往往超过单模型与集成的性能。

**⚠️ 局限性**

局限性包括：①对层间匹配的全局优化仍为近似；②聚类与部分最优传输求解成本较高，尤其在大模型上；③实验规模受限于小型网络，未验证在大型预训练模型（如 Transformer）上的可扩展性；④方法依赖于神经元相似度度量，可能对不同任务或网络架构表现不一致。

---

## 394. Exposing Vulnerabilities in Visible-Infrared VLMs: A Unified Geometric Adversarial Framework with Cross-Task Transferability

**arXiv ID:** 2605.22273 | [PDF](https://arxiv.org/pdf/2605.22273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 395. Building Europe's Quantum Shield: The Strategic view for a Continent-Wide Quantum Key Ditribution (QKD) Infrastructure

**arXiv ID:** 2605.22332 | [PDF](https://arxiv.org/pdf/2605.22332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 396. At What Cost? Software Developers' Well-Being in the Age of GenAI

**arXiv ID:** 2605.22349 | [PDF](https://arxiv.org/pdf/2605.22349v1)

**作者:** Mariam Guizani `[一作]` (Queen's University), Sofia Ouhbi `[通讯]` (Uppsala University)

**通讯引用:** 1887 | [OpenAlex ID](https://openalex.org/A5085468869)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出了一个跨学科的理论框架，旨在系统研究生成式人工智能（GenAI）对软件开发者福祉的正面与负面影响，并呼吁未来的实证研究方向

**💡 创新点**

创新点在于将组织双元性理论、ISO/IEC 25019 质量使用模型和工作需求-资源（JD‑R）模型三者结合，形成一套综合评估工具，兼顾技术使用效果与心理社会维度

**🔧 技术方法**

主要采用了已有的理论模型与测量量表（如NASA‑TLX、JD‑R量表等）进行概念化构建，而非新的算法或技术实现

**📊 数据集**

本研究未使用任何数据集，纯粹为概念性与理论性工作；呼吁后续研究收集开发者工作日志、问卷及访谈数据进行验证

**📈 对比分析**

由于缺乏实证实验，文中未对性能指标进行比较；提出的框架为未来的定量与定性研究提供了参考路线

**⚠️ 局限性**

局限性包括：1) 仅为理论与框架构建，缺乏数据验证；2) 可能未充分覆盖所有行业与文化背景；3) 对具体干预措施的可行性尚未评估

---

## 397. 4D-GSW: Kinematic-Aware Spatio-Temporal Consistent Watermarking for 4D Gaussian Splatting

**arXiv ID:** 2605.22342 | [PDF](https://arxiv.org/pdf/2605.22342v1)

**作者:** Sifan Zhou `[一作]` (Southeast University), Ming Li `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对 4D 高精度 Gaussian Splatting 动态场景，提出了 4D‑GSW 框架，在不破坏时空一致性的前提下把版权信息嵌入到 4D 体素的 0 阶 SH 颜色属性中。

**💡 创新点**

创新点：
- 通过 Spatio‑Temporal Curvature (STC) 量化动态瞬时曲率，动态门控水印梯度，避免 FVD 崩溃；
- 结合 HMM‑MRF 与 Optimal Transport 的联合能量最小化，实现跨帧全局时空一致性；
- 采用方向性梯度路由（anisotropic gradient routing）将水印信号仅注入曲率低、物理稳健的区域；
- 采用 360° 多视角监督，使水印可从任意视角解码。

**🔧 技术方法**

技术栈：4D Gaussian Splatting、Spatio‑Temporal Curvature、Hidden Markov Model、Gaussian Markov Random Field、Optimal Transport、Anisotropic Diffusion PDE、Wavelet 频域监督、LPIPS、CLIP、FVD 等评估指标。

**📊 数据集**

数据集：Consistent4D（七个动画资产）以及真实单目视频；使用 SC4D（Sparse‑Controlled 4D‑GS）作为骨干模型。

**📈 对比分析**

与 Consistent4D、4DGen、4Diffusion、L4GM、SC4D、SC4D+HiDDeN、SC4D+3D‑GSW 等基线对比：PSNR 28.36、SSIM 0.91、LPIPS 0.133、CLIP 0.904、FVD 1331，水印提取准确率 98.05%。相比基线，FVD 明显降低、PSNR 与 SSIM 提升，鲁棒性与可视化质量均得到提升。

**⚠️ 局限性**

局限：STC 假设 Gaussian 轨迹连续可导（C²），在极端非刚性或拓扑裂解事件中曲率估计失效，导致梯度路由失衡，可能削弱水印在这些极端区域的鲁棒性；未针对极端形变的自适应策略。

---

## 398. A Boundary-Layer Mechanism for One-Third Scaling in Online Softmax Classification

**arXiv ID:** 2605.22341 | [PDF](https://arxiv.org/pdf/2605.22341v1)

**作者:** Marcel Kühn `[一作]` (Leipzig University), Bernd Rosenow `[通讯]` (Leipzig University)

**通讯引用:** 5958 | [OpenAlex ID](https://openalex.org/A5020122979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

研究在线硬标签分类中softmax/交叉熵学习的极限行为，推导出以 1/3 幂律衰减的泛化误差，并分析学习率调度对该曲线的影响。

**💡 创新点**

首次在中心化变量框架下将softmax瓶颈归因于决策边界层，得到误差与梯度噪声、学习率调度等因素的解析关系。

**🔧 技术方法**

使用高维教师-学生统计力学模型、中心化闭包、边界层近似、在线SGD 的梯度噪声分析以及学习率调度解析。

**📊 数据集**

理论基于高斯输入；实验使用随机高斯样本和白化后的 ViT 特征等控制实验，没有使用公开大型数据集。

**📈 对比分析**

通过与仿真结果比较验证理论：D∝α^{1/3}，Δ趋于噪声底，泛化误差 ∝α^{-1/3}；学习率调度可提升至 ∝α^{-(2+γ)/6}；实验与理论吻合良好。

**⚠️ 局限性**

仅适用于单层固定特征、无标签噪声的硬标签、在线SGD 热力学极限，无法涵盖标签噪声、不可忽略的 Bayes 误差、特征学习、批量训练等更一般情形。

---

## 399. SepsisAI Orchestrator: A Containerized and Scalable Platform for Deploying AI Models and Real-Time Monitoring in Early Sepsis Detection

**arXiv ID:** 2605.22331 | [PDF](https://arxiv.org/pdf/2605.22331v1)

**作者:** Santiago Ospitia `[一作]` (University of Valle), John Garcia-Henao `[通讯]` (Balgrist University Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了SepsisAI-Orchestrator，一个面向早期败血症检测的开源容器化平台，集成数据预处理、HL7 FHIR兼容、NoSQL存储、LightGBM推理服务和Streamlit仪表盘，并对其在高并发环境下的水平扩展进行实测。

**💡 创新点**

提供了完整的临床AI交付参考架构，并首次量化了容器化模型推理在单节点CPU线程数下的U形延迟曲线，揭示副本数与CPU线程匹配是性能最优点。

**🔧 技术方法**

使用了Docker容器、Kubernetes编排、FastAPI+LightGBM推理服务、MongoDB NoSQL数据库、HL7 FHIR‑CDA预处理、Streamlit可视化、k6负载测试等技术。

**📊 数据集**

采用了PhysioNet/Computing in Cardiology Challenge 2019 ICU时间序列数据集。

**📈 对比分析**

通过k6模拟50-1000虚拟用户，比较不同AI服务副本数对吞吐量和p95延迟的影响；在12线程CPU上，12个副本时p95延迟1.41s、失败率0%，比3个副本提升57.3%，但仍未达到<500ms的目标，提示需进一步硬件或架构优化。

**⚠️ 局限性**

局限性包括未进行前瞻性临床验证、性能受CPU线程数限制，需更高核心或多节点；缺乏模型解释性、监管合规细节及实时监测的完整实现。

---

## 400. One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs

**arXiv ID:** 2605.22297 | [PDF](https://arxiv.org/pdf/2605.22297v1)

**作者:** Di He `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Shiwei Liu `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Heavy‑Tail Self‑Regularization（HT‑SR）理论的层级学习率分配策略（Layerwise Learning Rate，LLR），用于大规模语言模型（LLM）的预训练。

**💡 创新点**

创新点包括：①通过周期性估计每层权重谱的幂律指数α，动态将更大的学习率分配给heavy‑tail性弱的层；②嵌入层采用上限学习率；③使用软切换（Soft Switch）平滑LR变化；④仅在训练初20%阶段进行昂贵的谱计算，显著降低计算成本；⑤实现了低调优开销，直接继承统一LR的近优设置。

**🔧 技术方法**

主要技术：HT‑SR理论、经验谱密度（ESD）与幂律拟合、Hill估计、cosine decay调度、Soft Switch机制、Embedding特殊处理、主动阶段限制。

**📊 数据集**

数据集：FineWeb（LLM预训练）、GPT‑nano数据；七个commonsense reasoning基准（PIQA、SIQA、HellaSwag、Winogrande、ARC‑c、ARC‑e、OBQA）用于零样本评估。

**📈 对比分析**

对比方法：Uniform、LARS、LAMB、Sharpness‑based grid search、TempBalance 等；在60M–1B LLaMA、GPT‑nano模型以及 AdamW 与 Muon 优化器上，LLR 在相同训练 token 数下实现约1.5×速度提升、perplexity 降低、零样本准确率从 47.09% 提升至 49.02%（1.93%），在更大规模训练（3B/10B token）亦保持优势。

**⚠️ 局限性**

局限性：仍需在训练初期周期性计算谱，虽然已降到20%但仍存在额外计算；仅在 Transformer‑based LLM 上验证，极大规模（>10B）或其他模型架构的适用性待进一步研究；对不同任务（如下游微调）的泛化尚需更多实验。

---

## 401. Multi-Cell 6DMA: Cooperative Interference Management and Antenna Rotation Optimization

**arXiv ID:** 2605.22288 | [PDF](https://arxiv.org/pdf/2605.22288v1)

**作者:** Qijun Jiang `[一作]` (Chinese University of Hong Kong, Shenzhen), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 109099 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了多基站六维可动天线（6DMA）网络的两时尺度分布式协同优化框架，联合短期前向功率控制和长期天线旋转，以最大化平均加权总速率。

**💡 创新点**

创新点包括①提出低维度互细胞干扰功率约束（IPC）作为协同接口，分离短期与长期优化；②采用随机最大匹配+两阶段一维网格搜索实现边缘级IPC协调；③证明分布式算法对网络规模不显著依赖，并接近集中式全局最优。

**🔧 技术方法**

主要技术手段包括加权最小均方误差（WMMSE）重构、粒子群优化（PSO）搜索、样本平均逼近（SAA）、投影子梯度更新、随机最大匹配与两阶段网格搜索、局部凸优化。

**📊 数据集**

实验使用仿真数据：用户位置与路径损耗模型在三种网络拓扑（高、低IIC）以及多尺寸网络（M=3,6,10,15）上进行仿真，无使用公开数据集。

**📈 对比分析**

将所提分布式方案与集中式全局优化、固定旋转、仅IPC协同等基准进行对比，实验表明分布式方案与集中上界相差不足5%，并在不同IIC强度下均显著优于其他基准。

**⚠️ 局限性**

局限性包括：需要多次迭代，依赖统计CSI；IPC阈值设置仍需经验；在极端IIC条件下可能需进一步细化；目前仅通过仿真验证，实际天线模式与理论假设的差异需进一步研究。

---

## 402. SciCore-Mol: Augmenting Large Language Models with Pluggable Molecular Cognition Modules

**arXiv ID:** 2605.22287 | [PDF](https://arxiv.org/pdf/2605.22287v1)

**作者:** Yuxuan Chen `[一作]` (Peking University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 45509 | [OpenAlex ID](https://openalex.org/A5100320723)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出SciCore-Mol，一种在LLM主体上集成拓扑感知、扩散生成和反应感知的可插拔分子认知模块框架；

**💡 创新点**

创新点在于将分子拓扑、生成和反应推理等科学模块以隐藏状态接口深度集成到LLM内部，避免文本接口信息丢失；

**🔧 技术方法**

使用GVP网络做3D拓扑编码、DiT扩散生成器、Reaction Transformer以及跨模态对齐、连续强化学习等技术；

**📊 数据集**

利用400M化学领域文本、ORD反应数据库、SMolInstruct、MoleculeNet、ChemBench4K、MMLU-Chemistry、DrugR等多任务数据集；

**📈 对比分析**

通过与多种公开基线（GPT-4o、Intern-S1-mini、LlaSMol-Mistral-7B、Qwen3-8B等）对比，SciCore-Mol在分子理解、生成、反应预测和化学知识等五大维度上实现了更均衡且多项指标领先；

**⚠️ 局限性**

局限在于生成的分子结构精度仍有限、对大分子与蛋白质的支持不足、数值推理精度有待提升、训练流程复杂且需要多阶段协同训练。

---

## 403. EmoTrack: Robust Depression Tracking from Counseling Transcripts across Session Regimes

**arXiv ID:** 2605.22286 | [PDF](https://arxiv.org/pdf/2605.22286v1)

**作者:** Zhaomin Wu `[一作]` (National University of Singapore), Bingsheng He `[通讯]` (National University of Singapore)

**通讯引用:** 21645 | [OpenAlex ID](https://openalex.org/A5039946576)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出跨单会话和多会话的PHQ‑8抑郁评分预测框架 EmoTrack，结合LLM 提取的结构化临床特征、转录的语义嵌入以及压缩的跨会话记忆，完成对会话中抑郁症状的实时评估。

**💡 创新点**

创新点在于：①将 LLM 用作结构化特征提取器而非直接预测；②设计 Transformer encoder‑decoder 与症状查询相结合的自注意力机制，得到症状级别表征；③通过多槽注意力压缩先前会话并门控注入，减少噪声并捕获连续性；④构建标准化的多会话抑郁跟踪数据集，填补单会话与多会话评估的空白。

**🔧 技术方法**

使用 Qwen3‑Embedding‑8B 生成句子嵌入，Qwen3.5‑35B 提取 23 条 AIDA 结构化特征，Transformer 编码器‑解码器 + 症状查询 + 门控注意力，Huber 损失 + 逐症状辅助损失进行训练。

**📊 数据集**

使用新构造的多会话模拟数据集（约 3,600 条抑郁轨迹，17,122 条标注会话）以及公开的单会话临床访谈数据集（189 例）进行评估。

**📈 对比分析**

与 AIDA、LMIQ、Lau 等基线相比，EmoTrack 在单会话数据集上 MAE 从 2.82 降至 2.44（相对下降 13.5%），在多会话数据集上 MAE 与最强基线相当（2.65 vs 2.67），并且随会话数增加时记忆机制进一步提升性能。

**⚠️ 局限性**

局限包括：①基准使用合成数据，缺乏真实长期临床验证；②依赖 LLM 生成的特征，受限于提示质量；③仅使用最近一次会话的压缩记忆，未对完整历史进行建模；④缺乏完整的安全与伦理评估。

---

## 404. Bernini: Latent Semantic Planning for Video Diffusion

**arXiv ID:** 2605.22344 | [PDF](https://arxiv.org/pdf/2605.22344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. 3D LULC classification using multispectral LiDAR and deep learning: current and prospective schemes

**arXiv ID:** 2605.22328 | [PDF](https://arxiv.org/pdf/2605.22328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. Joint Communication and Computation Scheduling for MEC-enabled AIGC Services: A Game-Theoretic Stochastic Learning Approach

**arXiv ID:** 2605.22277 | [PDF](https://arxiv.org/pdf/2605.22277v1)

**作者:** Huaizhe Liu `[一作]` (Harbin Institute Of Technology), Lin Gao `[通讯]` (Harbin Institute Of Technology)

**通讯引用:** 91561 | [OpenAlex ID](https://openalex.org/A5087724445)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在移动边缘计算（MEC）支持的人工智能生成内容（AIGC）网络中，研究如何让用户设备（UE）自适应地选择通信基站（AP）、计算服务器（ES）以及模型推理步骤，以最小化服务完成时间并满足精度约束。

**💡 创新点**

创新点：①将 UE 决策建模为联合通信关联与计算卸载（JCACO）游戏，并证明在完全信息和随机信息两种情境下均为潜在博弈，从而保证存在纳什均衡；②提出分解为通信关联子博弈和计算卸载子博弈的潜在博弈框架；③设计分布式多智能体随机学习（MASL）算法，能够在无全局信息、环境动态变化的情况下收敛至纳什均衡，并给出严格的 ODE 收敛分析。

**🔧 技术方法**

技术：博弈理论（潜在博弈、纳什均衡）、多智能体随机学习、普通微分方程（ODE）收敛分析、通信与计算模型（Rayleigh 衰落、带宽分配、推理 FLOPs 计数）以及离散事件仿真。

**📊 数据集**

主要使用仿真数据：AP 数量 2~10、ES 数量 2~10、UE 数量 10~30、UE 活跃概率 0~1、推理步 FLOPs 0.1~0.5 TFLOPs/步、ES 计算容量 2~10 TFLOPs/s、传输数据量 2~10 MB、带宽 2~10 MHz 等；并未使用公开的真实 AIGC 数据集。

**📈 对比分析**

与四种基准方法比较：Best Response（BR）、Fictitious Play（mxFP）、Selfish、随机关联（RARO）。MASL 在总服务时延上均优于对比方案，提升幅度可达 6%–77%（在最多 ES 情况下）。实验还展示了学习率对收敛速度与最终性能的权衡。

**⚠️ 局限性**

局限性：①仿真模型假设 UE 活跃状态为独立伯努利分布，且网络状态在时隙内保持不变；②算法的理论收敛证明基于小学习率，实际实现中可能需要调参；③未考虑 UE 能耗、信道预测误差、模型迁移等实际细节；④缺乏在真实边缘部署环境中的实验验证。

---

## 407. From Snapshots to Trajectories: Learning Single-Cell Gene Expression Dynamics via Conditional Flow Matching

**arXiv ID:** 2605.22340 | [PDF](https://arxiv.org/pdf/2605.22340v1)

**作者:** Siyu Pu `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Xuezhi Wang `[通讯]` (Computer Network Information Center, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出单细胞 Flow Matching（scFM），通过在稀疏的单细胞RNA测序快照之间学习连续时间潜在动力学，生成未观测时点的细胞分布。

**💡 创新点**

创新点包括：① 双向时间相关速度场与 entropic OT 软监督相结合；② 利用速度场一致性细化 OT 耦合；③ 全局分布级对齐与潜在动力学正则化，显著降低长程漂移。

**🔧 技术方法**

采用 VAE 潜在空间编码、entropic optimal transport、模拟无耦合的流匹配回归、双向 Neural ODE 速度场、Wasserstein 对齐及 Top‑K Sinkhorn 近似等技术。

**📊 数据集**

实验使用 Zebrafish embryo（ZB）、Drosophila（DR）和 Schiebinger2019 细胞分化（SC）三大时间序列 scRNA‑seq 数据集。

**📈 对比分析**

在多时点 hold‑out 下与 OT‑WOT、scNODE、PRESCIENT、VGFM、Naive 等基准比较，使用 2‑Wasserstein 与平均 ℓ2 误差评估；scFM 在插值与外推任务上均显著低于基准，尤其在长时程外推中表现最佳。

**⚠️ 局限性**

局限性包括：仍依赖 VAE 的潜在空间假设；训练过程涉及多轮 OT 与 ODE 计算，成本较高；对极端分布漂移或极少观测时点的适应性仍需进一步验证。

---

## 408. Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action

**arXiv ID:** 2605.22283 | [PDF](https://arxiv.org/pdf/2605.22283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 409. Robustness of breast lesion segmentation under MRI undersampling improves with k-space-aware deep learning

**arXiv ID:** 2605.22327 | [PDF](https://arxiv.org/pdf/2605.22327v1)

**作者:** Lukas T. Rotkopf `[一作]` (University Hospital Essen), Moritz Rempe `[通讯]` (University Hospital Essen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究直接使用原始k‑space进行乳腺肿瘤分割，并评估其在加速采样或噪声环境下的鲁棒性。

**💡 创新点**

提出混合k‑space‑图像分割网络，利用k‑space信息在高加速和噪声下显著提升分割性能。

**🔧 技术方法**

采用3D U‑Net变体：混合k‑space‑图像模型、纯k‑space模型、幅值图像基线和复数图像基线；通过逆FFT桥接，使用Dice+Focal损失（纯k‑space模型额外加MSE），并采用随机下采样、梯度裁剪等训练技巧。

**📊 数据集**

使用fastMRI breast公开乳腺DCE‑MRI（原始k‑space）与MAMA‑MIA合成k‑space数据集，并在fastMRI上构建synthetic control。

**📈 对比分析**

通过5折交叉验证计算患者级Dice，混合模型与幅值基线在全采样下相当；在6×至48×加速以及噪声实验中，混合模型显著优于基线，基线在高加速下出现完全失败。

**⚠️ 局限性**

数据集规模有限、仅单一扫描平台、混合模型计算成本较高、下采样仅通过重格网实现、缺乏前瞻性验证。

---

## 410. Eliminating Premature Termination in Multihop Rendezvous for Cognitive Radio-based Emergency Response Network

**arXiv ID:** 2605.22325 | [PDF](https://arxiv.org/pdf/2605.22325v1)

**作者:** Zahid Ali `[一作]` (Atlantic Technological University), Saim Ghafoor `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了MR‑DMCA多跳可靠双模时钟协议，消除多跳会面中的过早终止，确保完整邻居发现与拓扑形成。

**💡 创新点**

创新点在于引入坐标辅助邻居验证与IDN验证的受控终止机制，解决传统N‑1终止导致的过早终止问题，实现100%拓扑匹配。

**🔧 技术方法**

采用双模时钟跳频、三路握手、坐标辅助邻居分类、坐标验证与受控终止策略，并在NS‑3中进行仿真。

**📊 数据集**

使用随机部署的20个CR节点（1000×1000m²区域），10/20个可用频道，频道相似度m=2/5，PR活跃度0%/85%，在NS‑3仿真中生成。

**📈 对比分析**

通过与RCS、MCA、EMCA、M‑DMCA在相同终止条件下的比较，利用ATTR、ATM、PTDD等指标评估；在高PR、低m条件下，MR‑DMCA的ATTR提升76%、37%、17%，且ATM始终为100%。

**⚠️ 局限性**

局限在于仅在静态节点、已知频道可用性场景下验证，未考虑节点移动、频谱漂移、硬件延迟等实际部署挑战，需要进一步扩展到大规模动态灾难环境。

---

## 411. Evaluation of Pipelines for Data Integration into Knowledge Graphs

**arXiv ID:** 2605.22304 | [PDF](https://arxiv.org/pdf/2605.22304v1)

**作者:** Marvin Hofer `[一作]` (ScaDS.AI), Erhard Rahm `[通讯]` (ScaDS.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于评估知识图谱（KG）集成管道的基准（KGI-Bench），并在电影领域（Movie）中实现了对应的数据集与评估指标；

**💡 创新点**

创新点在于：①将覆盖度、正确性和一致性三维质量指标统一到端到端管道评估；②提供可公开的基准数据集、参考KG以及多格式源（RDF、JSON、文本）模拟真实增量集成；③设计了可聚合成单一分数的加权排名方法；

**🔧 技术方法**

使用了多种开源工具（如PARIS、OpenIE、DBpedia Spotlight）和LLM（OpenAI gpt‑5‑mini）实现管道任务；在评估时引入实体对齐、三元组匹配、SHACL/OWL一致性检查等技术；

**📊 数据集**

数据集包括：①参考KG（约10k部电影及其关联实体），②种子KG，③分别为RDF、JSON、文本格式的三份重叠源；全部公开在GitHub与Zenodo；

**📈 对比分析**

通过对12个管道（6种单源配置 × 2 LLM/非LLM）在覆盖、正确率、F1、重复率、执行时间等指标进行量化比较；结果显示结构化RDF管道（尤其是LLM增强版）性能最优，JSON管道次之，文本管道表现最差；

**⚠️ 局限性**

限制包括：①LLM增强方案在非结构化源上效果不佳且成本高；②基准仅覆盖电影领域，泛化性待验证；③评价主要依赖参考KG，缺少对无参考KG场景的评估；

---

## 412. Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors

**arXiv ID:** 2605.22272 | [PDF](https://arxiv.org/pdf/2605.22272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 413. VEELA: A Clinically-Constrained Benchmark for Liver Vessel Segmentation in Computed Tomography Angiography

**arXiv ID:** 2605.22357 | [PDF](https://arxiv.org/pdf/2605.22357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering

**arXiv ID:** 2605.22269 | [PDF](https://arxiv.org/pdf/2605.22269v1)

**作者:** Junbin Xiao `[一作]` (University of Science and Technology of China), Angela Yao `[通讯]` (National University of Singapore)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5006278133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MuKV框架，用多粒度KV缓存压缩和半层级检索实现长流式视频问答的高效回答。

**💡 创新点**

创新点在于：① 采用自注意力与频率两信号融合的双信号KV压缩，② 以补丁/帧/片段三层粒度生成KV缓存，③ 设计半层级检索先并行检索后细粒度再排序的策略。

**🔧 技术方法**

技术包括：LLM预填充获取KV缓存、FFT频率分析、双信号加权、平均池化、余弦相似度检索、LLaVA-OV/ Qwen3-VL 作为多模LLM、半层级检索与重排序。

**📊 数据集**

使用RVSEgo、RVSMovie、StreamingBench三大长流视频QA数据集进行实验。

**📈 对比分析**

与ReKV、FVStream、LongVA、InfiniPot-V等方法对比，MuKV在RVSEgo和StreamingBench上显著提升准确率（最高提升约5.8%），同时保持或降低KV缓存大小和在线检索时延。

**⚠️ 局限性**

局限性：对需要精细计数的任务表现不佳，压缩可能会损失细粒度信息，且在极长视频或极高帧率场景下效果需要进一步验证。

---

## 415. Detecting Atypical Clients in Federated Learning via Representation-Level Divergence

**arXiv ID:** 2605.22266 | [PDF](https://arxiv.org/pdf/2605.22266v1)

**作者:** Cristian Pérez-Corral `[一作]` (Polytechnic University of Valencia), Enrique S. Quitana-Ortí `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于激活模式的轻量几何度量，用来量化联邦学习中各客户端与全局模型的功能差异，并通过该度量检测异常客户端。

**💡 创新点**

创新点在于不再关注模型参数或梯度，而是从ReLU网络激活模式的几何结构出发，构建层级加权的几何发散度指标，提供可解释且对重排不敏感的监控信号。

**🔧 技术方法**

使用激活模式、汉明距离、层级加权的Frobenius范数、MAD稳健z分数等技术；实验采用SGD、Flower框架，并在Probe集上评估几何度量。

**📊 数据集**

使用Fashion‑MNIST（MLP）和CIFAR‑10（ResNet‑18）作为实验数据集。

**📈 对比分析**

通过不同Dirichlet α值的异质性对比，度量能清晰区分iid与高异质性客户端；在模拟数据扰动的场景下，稳健z分数能持续识别异常客户端，验证了方法的有效性和鲁棒性。

**⚠️ 局限性**

局限性在于仅在轻量模型和小规模Probe集上验证，未在大规模联邦系统或真正对抗攻击场景下测试；还需进一步将该信号集成到聚合策略中以提升实用性。

---

## 416. QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks

**arXiv ID:** 2605.22351 | [PDF](https://arxiv.org/pdf/2605.22351v1)

**作者:** Haotong Qin `[一作]` (ETH Zurich), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 23713 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种低位量化图像超分辨率框架QuantSR+，用于在资源受限设备上实现高效且高精度的超分辨率任务。

**💡 创新点**

核心创新包括：Redistribution-driven Bit Determination (RBD)提升量化器表示能力；Quantized Slimmable Architecture (QSA)实现结构逐步剪枝与自适应；Slimming-guided Function-localized Distillation (SFD)在量化过程中实现局部功能对齐并加速收敛。

**🔧 技术方法**

技术实现涵盖：低位量化（2–4比特）权重/激活量化，基于残差的逐位决定；构建可动态演化的可剪枝量化网络；采用基于块的蒸馏损失配合渐进剪枝；结合STE改进梯度传递。

**📊 数据集**

数据集与评测：训练使用DIV2K；在Set5、Set14、B100、Urban100、Manga109等标准图像超分辨率基准上评估；对SwinIR、Transformer、Diffusion等多种骨干网络进行量化实验。

**📈 对比分析**

与DoReFa、PAMS、CADyQ、QuantSR、ODM、2DQuant等现有方法对比，QuantSR+在2比特量化下PSNR提升约0.29dB（×4 Urban100），4比特时达到或超过全精度水平；同时在2比特时运算减少约88%，存储减少约89%。

**⚠️ 局限性**

局限性包括：1比特量化效果尚不理想；需要大量训练与调参；对极低精度下的泛化性与不同Transformer结构的适配仍待进一步验证。

---

## 417. Riemannian geometry meets fMRI: the advantages of modeling correlation manifolds and eigenvector subspaces

**arXiv ID:** 2605.22334 | [PDF](https://arxiv.org/pdf/2605.22334v1)

**作者:** Mario Severino `[一作]` (University of Padova), Mattia Veronese `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文将功能连接矩阵视为Riemannian流形上的点，提出Off–log欧氏几何和Grassmannian子空间判别方法，提升fMRI数据的统计灵敏度和预测性能。

**💡 创新点**

创新点在于引入Permutation‑Invariant Off–log diffeomorphism，使关联矩阵在全局可逆坐标下可进行闭式距离、Fréchet均值和线性模型；以及将图拉普拉斯特征子空间映射到Grassmannian，解决基向量符号/旋转歧义。

**🔧 技术方法**

主要技术包括Riemannian几何、log‑euclidean映射、Tangent‑Space投影、Elastic‑Net回归、线性SVM、基于主角度的Grassmannian判别。

**📊 数据集**

使用了三组老化组（cam‑CAN、HCP‑Aging、NKI）、帕金森（DataPD）和非情感精神病（DataNAP）fMRI数据。

**📈 对比分析**

与Euclidean、ECM、LEC基线对比，Off–log在两类临床分类任务中实现最高AUC/准确率，Grassmannian判别在两组上均优于LDA；在脑龄回归中Off–log与Raw相当，ECM在部分数据集上稍优。

**⚠️ 局限性**

局限在于样本量有限、仅使用线性模型、未验证跨站/不同预处理的鲁棒性、需要对低秩或噪声矩阵做正则化，且对高维全局分布的近似性尚未充分评估。

---

## 418. ACCoRD: Actor-Critic Conflict Resolution with Deep learning for O-RAN xApps

**arXiv ID:** 2605.22306 | [PDF](https://arxiv.org/pdf/2605.22306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 419. How can reasoning capability empower the AI copilot robot in endoscopic surgery

**arXiv ID:** 2605.22322 | [PDF](https://arxiv.org/pdf/2605.22322v1)

**作者:** Guankun Wang `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17724 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出将多模态视觉-语言-动作（VLA）模型与推理能力结合，构建具备级别2-3自治的内镜手术 AI 助手机器人；

**💡 创新点**

通过链式推理实现多步骤逻辑推断、上下文感知和不确定性意识，从而让机器人从仅仅执行预定义指令转向像经验丰富外科医生般的认知协作；

**🔧 技术方法**

采用 VLA 结构、链式思维提示、强化学习以及多模态融合技术（视觉、影像、传感器等），并利用轻量化推理架构与边缘硬件加速；

**📊 数据集**

文章未提供具体数据集，主要基于现有大规模机器人视觉与语言数据和临床影像（如 CT/MRI、EUS、OCT）进行概念性论证；

**📈 对比分析**

未开展实验比较，本文以理论分析与架构设计为主，未给出性能指标；

**⚠️ 局限性**

面临实时计算约束、推理可靠性与安全性保证、与现有手术平台的兼容性、以及缺乏实证评估和标准化基准等限制。

---

## 420. Throughput and Delay Performance of Slotted Aloha in SmartBANs under Saturation Conditions

**arXiv ID:** 2605.22317 | [PDF](https://arxiv.org/pdf/2605.22317v1)

**作者:** Anastasios C. Politis `[一作]` (International Hellenic University), Constantinos S. Hilas `[通讯]` (International Hellenic University)

**通讯引用:** 507 | [OpenAlex ID](https://openalex.org/A5054211132)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文使用二维离散时间马尔可夫链（DTMC）对 ETSI SmartBAN 标准中 Slotted Aloha 的拥塞控制进行建模，并推导了在饱和条件下的最大吞吐量与平均端到端延迟。

**💡 创新点**

创新点在于：① 将 CP_min/CP_max 的递减规则与重传阶段一起嵌入二维 DTMC；② 在单一用户优先级（UP）均质网络上得到解析表达式，能够对不同优先级的吞吐量与延迟进行精确评估；③ 通过与自定义仿真对比验证模型的准确性。

**🔧 技术方法**

主要技术：二维 DTMC 建模、马尔可夫稳态分析、解析求解传输概率 τ 与碰撞概率 p、推导吞吐量 S 与平均延迟 E[D]；仿真使用 Python 进行离散时间 Slotted Aloha 仿真。

**📊 数据集**

未使用真实数据集；所有结果均来自理论推导与基于模型的仿真（10^5 个时间槽、节点数 1–16）。

**📈 对比分析**

通过将理论结果与仿真结果在吞吐量、传输概率、碰撞概率及延迟等指标上对比，发现两者吻合度极高；结果表明吞吐量随节点数增加而下降，尤其是高优先级 UP 的性能衰减最快；平均延迟在低至中等优先级时相对稳定，但在高优先级下快速恶化。

**⚠️ 局限性**

局限性：仅考虑同一 UP 的均质网络；假设信道无误差、无限重传；未涵盖 ACK/NACK 机制；缺乏对异质 UP 网络或实际无线信道衰落条件的分析。

---

## 421. Long-term Fairness with Selective Labels

**arXiv ID:** 2605.22291 | [PDF](https://arxiv.org/pdf/2605.22291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 422. Impact of Atmospheric Turbulence and Pointing Error on Earth Observation

**arXiv ID:** 2605.22268 | [PDF](https://arxiv.org/pdf/2605.22268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 423. Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation for LLM Reasoning

**arXiv ID:** 2605.22263 | [PDF](https://arxiv.org/pdf/2605.22263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 424. TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation

**arXiv ID:** 2605.22355 | [PDF](https://arxiv.org/pdf/2605.22355v1)

**作者:** Hanyu Guo `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5555 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TransitLM数据集和三任务基准，实现端到端无地图公共交通路线规划

**💡 创新点**

首次构建13M记录、120k站点的无地图路线规划数据，证明LLM可通过数据学习空间拓扑并完成结构化路线生成

**🔧 技术方法**

采用持续预训练（CPT）+监督微调（SFT）的大模型（Qwen3系列），并为每个站点注册专门令牌以保证结构化生成

**📊 数据集**

使用Amap四大城市（北京、上海、深圳、成都）的路线日志、站点与线路静态描述，构成13.9M条记录

**📈 对比分析**

与六大通用LLM在最优路线生成任务下对比，TransitLM 4B模型在连通率≥97%、站点定位≥99%、路线完全匹配≈74%，在仅提供GPS坐标的设定仍保持高性能；多任务联合训练无负迁移

**⚠️ 局限性**

仅覆盖四个城市且为静态网络，缺少实时动态和更广泛地理范围，模型训练成本高

---

## 425. Pattern-and-root inflectional morphology: the Arabic broken plural

**arXiv ID:** 2605.22310 | [PDF](https://arxiv.org/pdf/2605.22310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 426. Meta-Soft: Leveraging Composable Meta-Tokens for Context-Preserving KV Cache Compression

**arXiv ID:** 2605.22337 | [PDF](https://arxiv.org/pdf/2605.22337v1)

**作者:** Wei Luo `[一作]` (Guangdong Institute of Intelligence Science and Technology), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Meta-Soft框架，实现KV缓存压缩。

**💡 创新点**

创新点在于使用Meta-库生成动态软Token进行全局探测，并通过注意力流实现信息迁移，避免硬删除。

**🔧 技术方法**

采用Meta-Library与Gumbel-Softmax生成软Token，注意力流聚合以及两阶段训练。

**📊 数据集**

使用SlimPajama、PG19、OpenWebText2、LongBench、RULER等数据集。

**📈 对比分析**

与H2O、SnapKV、Judge Q、ZeroMerge、StreamingLLM等方法对比，Meta-Soft在PPL、LongBench平均分以及RULER精度上均优于基线，且保持接近Full KV性能。

**⚠️ 局限性**

局限在于仍需额外的Meta-Library训练，且在极长上下文下信息迁移可能导致轻微语义混淆。

---

## 427. Learning Causal Orderings for In-Context Tabular Prediction

**arXiv ID:** 2605.22335 | [PDF](https://arxiv.org/pdf/2605.22335v1)

**作者:** Sascha Xu `[一作]` (CISPA Helmholtz Center), Jilles Vreeken `[通讯]` (CISPA Helmholtz Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于Transformer的表格基础模型，利用可学习的因果变量顺序进行注意力约束，实现对缺失值的推断与稳健预测。

**💡 创新点**

创新点在于将因果顺序学习与无监督的最大似然目标结合，直接在模型内部学习因果拓扑，且对缺失数据的噪声增量机制提升因果识别。

**🔧 技术方法**

使用Transformer、因果顺序约束注意力、可微温度化排序、加性噪声模型、最大似然训练以及缺失值增量估计。

**📊 数据集**

在合成数据（加性、非加性机制）和真实世界数据（单细胞多参数、OpenML CC18/CTR23）上验证。

**📈 对比分析**

与传统因果发现算法（CAM、GES、PC等）以及表格基础模型、缺失值插补方法对比，因果顺序发现误差与最优方法相近，预测/插补误差在高缺失率下优于竞争者。

**⚠️ 局限性**

局限性：仅学习拓扑顺序而非完整DAG，依赖加性噪声模型假设，对训练数据的先验分布敏感，缺少对完整因果图结构的显式推断。

---

## 428. Benchmarking Autonomous Agents against Temporal, Spatial, and Semantic Evasions

**arXiv ID:** 2605.22321 | [PDF](https://arxiv.org/pdf/2605.22321v1)

**作者:** Jianan Ma `[一作]` (Hangzhou Dianzi University), Zhen Wang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 73267 | [OpenAlex ID](https://openalex.org/A5100460802)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向自主智能体的安全基准，评估其在多轮交互中对时空语义规避攻击的鲁棒性。

**💡 创新点**

提出了跨时空语义三维规避框架，并设计了包含20类风险的系统化风险分类法及对应的34种攻击技术。

**🔧 技术方法**

利用大型语言模型生成多轮会话、自动合成攻击实例，并通过 OpenClaw 平台实现真实工具调用与执行。

**📊 数据集**

使用自研的 Benchmark3Sigma，包含 2,254 条多轮对话（1,512 侵入案例、742 正常案例），涵盖六种使用场景和十类风险。

**📈 对比分析**

对十个主流 LLM（包括八大开源模型和两大专有模型）进行实验，比较 RTR@1、RTR@2、RTR@3、GSS 等指标；实验表明攻击成功率从基线 28.3% 提升至 52.6%，部分模型攻击率可达 77.7%。

**⚠️ 局限性**

实验仅在 OpenClaw 平台进行，评估方法依赖 LLM‑as‑judge 可能产生误判；缺乏跨平台验证、跨模态攻击探索以及对防御机制的更系统评估。

---

## 429. Automatic Contextual Audio Denoising

**arXiv ID:** 2605.22262 | [PDF](https://arxiv.org/pdf/2605.22262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 430. Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

**arXiv ID:** 2605.22305 | [PDF](https://arxiv.org/pdf/2605.22305v1)

**作者:** Stefan Huber `[一作]` (University of Applied Sciences), Jakob Rehrl `[通讯]` (University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Mountain Car问题进行解析求解并提出Chebyshev多项式策略

**💡 创新点**

解析求解36年未解的山车问题，并将Chebyshev多项式引入为可稠密逼近连续策略的通用模型

**🔧 技术方法**

采用多元Chebyshev多项式、PPO/ARS/REINFORCE等强化学习算法

**📊 数据集**

使用Mountain Car Continuous、Pendulum、Aero 2模拟与真实实验数据集

**📈 对比分析**

与MLP基准对比，Chebyshev策略将regret降低约4倍，参数量减少277倍，表现更佳

**⚠️ 局限性**

多项式基数指数增长导致高维或高阶时计算和数值不稳定

---

## 431. Don't Forget the Critic: Value-Based Data Rehearsal for Multi-Cyclic Continual Reinforcement Learning

**arXiv ID:** 2605.22454 | [PDF](https://arxiv.org/pdf/2605.22454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 432. A Constant-Time Implementation Methodology for Activation Functions on Microcontrollers

**arXiv ID:** 2605.22441 | [PDF](https://arxiv.org/pdf/2605.22441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 433. Assisted Counterspeech Writing at the Crossroads of Hate Speech and Misinformation

**arXiv ID:** 2605.22435 | [PDF](https://arxiv.org/pdf/2605.22435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 434. Characterizing the Fault Response of the Intel Neural Compute Stick 2 Under Single-Pulse Electromagnetic Fault Injection

**arXiv ID:** 2605.22437 | [PDF](https://arxiv.org/pdf/2605.22437v1)

**作者:** Štefan Kučerák `[一作]` (Slovak University Of Technology), Xiaolu Hou `[通讯]` (Slovak University Of Technology)

**通讯引用:** 630 | [OpenAlex ID](https://openalex.org/A5069420868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对Intel Neural Compute Stick 2进行单脉冲电磁失效注入实验，使用三种ImageNet卷积网络（ResNet‑18、ResNet‑50、VGG‑11）在OpenVINO运行时评估其故障响应。

**💡 创新点**

提出了四类故障分类法（无效、更改、持久性降级、设备失效），揭示了持久性SEU类和空闲状态下可诱发故障的新现象，并通过模型架构对可靠性影响进行量化。

**🔧 技术方法**

采用NewAE ChipSHOUTER单脉冲EMFI、1mm/4mm近场探头、3轴XYZ位移台、Optuna超参数搜索和Python客户端/服务器架构，并使用OpenVINO驱动推理与硬件看门狗。

**📊 数据集**

使用ImageNet验证集512张图像，分别加载ResNet‑18、ResNet‑50、VGG‑11三种训练好的模型进行实验。

**📈 对比分析**

通过对持久性降级率和设备失效率的比较，发现ResNet‑50的持久性降级率最高（≈31%），VGG‑11在空闲时的设备失效率最高（≈40%），显示模型架构显著影响可靠性。

**⚠️ 局限性**

实验仅在单台NCS2上进行，缺乏多设备复现，缺少层级定位、无无脉冲对照、仅限图像分类工作负载，且触发时延导致层级定位受限，限制了结论的通用性。

---

## 435. The Neglected Baseline in Model Interpretation

**arXiv ID:** 2605.22417 | [PDF](https://arxiv.org/pdf/2605.22417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 436. Multi-Stage Training for Abusive Comment Detection in Indic Languages

**arXiv ID:** 2605.22380 | [PDF](https://arxiv.org/pdf/2605.22380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 437. DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA

**arXiv ID:** 2605.22411 | [PDF](https://arxiv.org/pdf/2605.22411v1)

**作者:** Jianing Yin `[一作]` (Zhejiang University), Tan Tang `[通讯]` (Zhejiang University)

**通讯引用:** 840 | [OpenAlex ID](https://openalex.org/A5018322383)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种长程记忆框架DeferMem，先用轻量级的分段-链接结构进行高召回检索，再在查询时通过强化学习训练的记忆蒸馏器将噪声候选转换为查询条件的证据。

**💡 创新点**

创新点在于（1）将证据蒸馏推迟到查询时刻，解耦检索与蒸馏；（2）设计分段-链接结构实现可控的高召回检索；（3）提出DistillPO算法，结合结构化蒸馏动作、分解与门控奖励以及结构对齐的优势分配，显著提升蒸馏效果。

**🔧 技术方法**

采用的技术包括：LLM+轻量级分段-链接检索结构、强化学习（DAPO+Leaky Hierarchical Gating）进行蒸馏器训练、结构化奖励分解、结构对齐优势分配、全奖励锚点稳定训练等。

**📊 数据集**

使用的评测数据集为LoCoMo、LongMemEval‑S（以及LongMemEval‑M的扩展实验）。

**📈 对比分析**

与FullText、NaiveRAG、Mem0、A‑Mem、MemoryOS、MemGAS、LightMem、GAM、Memory‑R1等多组基线相比，DeferMem在两大数据集上实现了最高的问答准确率（LoCoMo 88.25%、LongMemEval‑S 70.0%），并且在内存操作的Token成本为0、运行时显著缩短（约3.1×-9.8×提升）。

**⚠️ 局限性**

局限性包括：对蒸馏器的强化学习训练依赖大量标注问答对；在极小但关键细节的提取上仍有错误；分段-链接阈值调节对召回与噪声平衡敏感；当历史极为长大时，检索候选量仍可能较大，影响效率。

---

## 438. GazePrior: Zero-Shot AR/VR Eye Tracking via Learned 3D Gaze Reconstruction

**arXiv ID:** 2605.22359 | [PDF](https://arxiv.org/pdf/2605.22359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 439. Nf-PEAK: Process-Based Energy Attribution for Nextflow Workflows on Kubernetes Clusters

**arXiv ID:** 2605.22393 | [PDF](https://arxiv.org/pdf/2605.22393v1)

**作者:** Philipp Thamm `[一作]` (Humboldt-Universität zu Berlin), Ulf Leser `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 8995 | [OpenAlex ID](https://openalex.org/A5055236937)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一种基于容器化的Nf-PEAK工具，用于在Kubernetes集群上对Nextflow工作流中的单个任务进行CPU和DRAM能耗归因。

**💡 创新点**

结合RAPL计数器与进程级性能计数，并采用非线性能耗信用模型，实现了在多租户、共享集群环境下的任务级能耗精准归因。

**🔧 技术方法**

利用Kubernetes API、cgroup元数据、RAPL能耗计数、CPU时间/内存占比、非线性γ指数归因模型、容器化部署与周期采样技术。

**📊 数据集**

在nf-core的RNASeq、Sarek、Rangeland三个工作流上进行实验，使用真实输入数据、不同节点数和受控CPU负载。

**📈 对比分析**

与Kepler在相同集群与负载条件下比较，采用MAPE指标；Nf-PEAK在无负载时平均MAPE 6.6%，有负载时10.9%，均显著低于Kepler的17.4%/22.5%。

**⚠️ 局限性**

对极短（<15s）物理任务的归因精度不足；仅覆盖CPU/DRAM能耗，未考虑存储、网络、GPU等；需宿主机权限，部署受限。

---

## 440. Cohesion-6K: An Arabic Dataset for Analyzing Social Cohesion and Conflict in Online Discourse

**arXiv ID:** 2605.22447 | [PDF](https://arxiv.org/pdf/2605.22447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 441. Towards Explainability of SLMs by investigating Token Level Activation

**arXiv ID:** 2605.22377 | [PDF](https://arxiv.org/pdf/2605.22377v1)

**作者:** Sayantani Ghosh `[一作]` (A.K. Choudhury School of Information Technology), Amlan Chakrabarti `[通讯]` (University of Calcutta)

**通讯引用:** 5302 | [OpenAlex ID](https://openalex.org/A5043543748)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对BERT-Base-uncased模型第8层隐藏状态的L2范数进行量化，并构建激活流网络（AFN）对Token进行重要性排名；

**💡 创新点**

创新点在于采用轻量化、模型无关的激活强度阈值分桶方法，直接用激活幅度揭示语义关键Token，并提出激活位移（Activation Shift）度量上下文或提示变化对Token表示的影响；

**🔧 技术方法**

技术包括BERT隐藏状态提取、L2范数计算、上四分位阈值分桶、激活位移（L2差值）以及可视化统计；

**📊 数据集**

使用约732条情感句子（sentiment-bearing sentences）以及若干案例对比句；

**📈 对比分析**

通过与第9层激活值、不同句子对与不同提示下的激活位移比较，实验显示第8层高激活Token主要为内容词，且激活位移聚焦于语义变更的Token，验证了方法能更准确捕捉语义信息；

**⚠️ 局限性**

局限性：仅聚焦单一层和单一模型，阈值选择经验性，未对多种Transformer架构或更大语料进行泛化验证，且未提供梯度级别的解释。

---

## 442. TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting

**arXiv ID:** 2605.22365 | [PDF](https://arxiv.org/pdf/2605.22365v1)

**作者:** Quang Duc Nguyen `[一作]` (Nanyang Technological University, Singapore), Dacheng Tao `[通讯]` (Nanyang Technological University, Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对时间序列预测中的后门攻击，系统评估了13种防御方法，并提出了基于通道级池训练的TimeGuard防御框架。

**💡 创新点**

创新点在于将训练过程从样本级转为通道级训练，采用时序感知的池初始化以及距离正则化损失选择，以解决数据混合和任务形式转移导致的信号稀释与损失退化问题。

**🔧 技术方法**

技术手段包括通道可靠池训练、时间感知池初始化策略和距离正则化损失选择机制。

**📊 数据集**

实验使用了三大公开时间序列数据集，并在三种不同的预测模型（如LSTM、Transformer、GRU）上进行验证。

**📈 对比分析**

与最强基线相比，TimeGuard在鲁棒性指标上提升了1.96倍，同时在干净数据上的预测误差仅提升不到5%。

**⚠️ 局限性**

局限性在于仍需要一定量的无污染训练数据来构建可靠池，且在极高维度或复杂多通道场景下的效果尚未充分验证。

---

## 443. Terminal Constraint Model Predictive Control for Image-Based Visual Servoing of UAVs with Kalman Filter-Based Moment Loss Compensation

**arXiv ID:** 2605.22443 | [PDF](https://arxiv.org/pdf/2605.22443v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 444. Perceived Safety of Workers in Encounters with Large Industrial AGVs

**arXiv ID:** 2605.22461 | [PDF](https://arxiv.org/pdf/2605.22461v1)

**作者:** Ansgar Howey `[一作]` (KION Supply Chain Solutions, Still GmbH), Achim J. Lilienthal `[通讯]` (Technische Universitaet Muenchen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了工业级大型AGV与工人近距离交互时的感知安全性，并在真实工厂现场与虚拟现实环境中进行实验。

**💡 创新点**

创新点在于将真实工人作为受试者，结合连续触发器压力测量和VR对比，首次探讨大载荷AGV在实际和虚拟场景下的安全感知差异，并让工人自行设定避让参数。

**🔧 技术方法**

使用的技术包括基于ROS2的AGV路径规划与Regulated Pure Pursuit控制、LED+音响eHMI信号、Unity实现的VR仿真（HTC XR Elite头显），以及手持触发器的实时压力测量。

**📊 数据集**

数据集由10名Still GmbH员工的触发器时序记录、距离测量及5点Likert量表问卷组成，总计约100多条交互记录。

**📈 对比分析**

通过触发器使用率、累计触发时长和曲线下面积三项指标与Likert评分对比，结果显示VR情境下感知威胁更高（AUC、t_total均升高），但两种模式在中等和缓和轨迹下相近，整体安全评分略低于真实场景。

**⚠️ 局限性**

主要局限包括样本量仅10人、仅使用静止受试者的单一交互情景、性别分布不均，且未充分验证不同速度或动态场景下的可推广性。

---

## 445. Diffusion-guided Generalizable Enhancer for Urban Scene Reconstruction

**arXiv ID:** 2605.22420 | [PDF](https://arxiv.org/pdf/2605.22420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 446. Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer Inference

**arXiv ID:** 2605.22416 | [PDF](https://arxiv.org/pdf/2605.22416v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 447. Real-Time Auto-Optimization in Unknown Environments via Structure-Exploiting Dual Control for Exploration and Exploitation

**arXiv ID:** 2605.22431 | [PDF](https://arxiv.org/pdf/2605.22431v1)

**作者:** Shiying Dong `[一作]` (Hong Kong Polytechnic University), Wen-Hua Chen `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 24876 | [OpenAlex ID](https://openalex.org/A5100428822)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了一种针对未知环境下自动优化控制的结构化数值双控制方法（DCEE），能够在实时操作中同时实现探索与利用；

**💡 创新点**

发现并利用DCEE的“凸‑非线性”结构，将非线性残差线性化而保留凸外层损失，从而把原始非凸问题转化为一系列高效的Gauss‑Newton最小二乘子问题；

**🔧 技术方法**

采用序贯凸优化（SCP）与广义Gauss‑Newton（GGN）近似、自动微分求雅可比、集成学习（ensemble）更新及非线性残差映射；

**📊 数据集**

通过仿真和硬件在环（HiL）实验验证，使用电动车的生态巡航控制作为案例，未使用公开数据集；

**📈 对比分析**

与经典DCEE、极值寻踪（ESC）以及直接非线性规划求解器进行对比，结果显示终端速度误差下降83.5%、累计速度误差下降66.7%、累计遗憾下降84.3%；计算时间平均提升约9.6倍，极端情况下提升约9.4倍；

**⚠️ 局限性**

仅在无约束情况下验证，缺乏闭环稳定性与收敛性分析；目前仅针对车辆生态巡航的具体案例，需进一步推广到更多受限系统与更复杂环境。

---

## 448. Exploiting Multicast for Accelerating Collective Communication

**arXiv ID:** 2605.22428 | [PDF](https://arxiv.org/pdf/2605.22428v1)

**作者:** Chao Xu `[一作]` (Huawei), Jingbin Zhou `[通讯]` (Huawei)

**通讯引用:** 91 | [OpenAlex ID](https://openalex.org/A5086356074)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出并实现了一种新的多写（MultiWrite）语义，用于在 AI 训练与推理集群中消除多对多通信中的冗余数据传输，从而显著降低 AllGather 与 AlltoAll 操作的端到端延迟。

**💡 创新点**

创新点在于：① 将多路复用机制迁移到写语义层，避免了传统硬件多播的控制平面开销与组爆炸问题；② 通过在包头中嵌入位图元数据，让中继节点自行复制转发，保持对现有框架和协议（如 UB/RoCE）的完整兼容；③ 在 Ascend NPU 生态上实现无硬件改造的多写，实测可达 33% 的延迟下降。

**🔧 技术方法**

技术手段包括：基于位图的目的地编码、递归的多写执行模型、软件层级的复制与转发实现、以及对现有 UB 传输栈的语义扩展；在实现上采用 Ascend 910B4 NPU 集群的全网格与 CLOS 互联，并利用 Ascend 910B 的 Cache Coherence System 与 RoCE 网络接口。

**📊 数据集**

数据集与实验环境：使用两台配备 8 个 Ascend 910B4 NPU 的服务器（共 16 个 NPU），在长时间压力测试中对 AllGather（每个 NPU 16 MB 数据）和 AlltoAll（MoE 预填/解码阶段的 token 批量）进行测评；并与官方 HCCL 及已优化的单播多路径方案进行对比。

**📈 对比分析**

比较方法：以端到端延迟（API 调用到返回）为指标，取所有参与节点的最大值；实验通过多轮 warm‑up 与长周期跑测，得到统计均值与方差。结果显示：MultiWrite AllGather 延迟平均降低 30%（相较官方基线）并比单播多路径方案低 17%；AlltoAll 在大批量预填阶段可降低 12%–27%，整体可实现最高 33% 的延迟提升。

**⚠️ 局限性**

局限性包括：① 对于消息体积极小（<2 MB）时，由于拆分与转发开销，延迟反而高于基线；② 需要在中继 NPU 上额外占用 AICPU 进行元数据解析与复制，虽对整体工作负载影响有限，但在 CPU 资源紧张时仍是潜在瓶颈；③ 位图编码在 64+ 设备规模下需使用协议预留字段或嵌入负载，导致占用比例略升高；④ 目前仅在 Ascend NPUs 上实现，跨平台可移植性需进一步验证。

---

## 449. Guiding Multi-Objective Genetic Programming with Description Length Improves Symbolic Regression Solutions

**arXiv ID:** 2605.22374 | [PDF](https://arxiv.org/pdf/2605.22374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 450. Towards Clinically Interpretable Ophthalmic VQA via Spatially-Grounded Lesion Evidence

**arXiv ID:** 2605.22414 | [PDF](https://arxiv.org/pdf/2605.22414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 451. Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent Higher-Area Rankings Across Human fMRI and Macaque Electrophysiology

**arXiv ID:** 2605.22401 | [PDF](https://arxiv.org/pdf/2605.22401v1)

**作者:** Nils Leutenegger `[一作]` `[通讯]` (Independent Researcher), Nils Leutenegger (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在相同的CNN权重上，对比人类fMRI与猕猴电生理数据，系统评估了五种学习规则（BP、FA、PC、STDP、随机权重）在跨物种层面上的表征相似度。

**💡 创新点**

首次实现跨物种、跨测量模式的多规则模型-脑对齐比较，发现早期视觉层中局部学习规则（STDP、PC）始终优于BP，表明该规律不受测量方式限制。

**🔧 技术方法**

采用代表性相似性分析（RSA）、Kendall’s τ排名比较、噪声上限估计、刺激控制分析以及以ResNet‑50为基准的容量控制。

**📊 数据集**

使用人类THINGS‑fMRI、猕猴MajajHong2015（V4/IT）、FreemanZiemba2013（V1/V2）以及预训练ResNet‑50作为对照模型。

**📈 对比分析**

通过RDM的Spearman相关度量模型层与神经数据的相似性，发现猕猴早期视觉层的ρ值高达0.15–0.30，远高于人类的0.01–0.08；STDP/PC在V1/V2中取得最高对齐；IT层的对齐受模型容量限制，ResNet‑50显著优于小CNN。

**⚠️ 局限性**

限制包括CNN架构过小、学习规则数量有限导致统计功效不足、跨物种刺激集合不匹配、噪声上限估计可能偏高，以及只用分割神经元而非试验的可靠性估算。

---

## 452. A Posterior-Predictive Variance Decomposition for Epistemic and Aleatoric Uncertainty in Wind Power Forecasting

**arXiv ID:** 2605.22390 | [PDF](https://arxiv.org/pdf/2605.22390v1)

**作者:** Yinsong Chen `[一作]` (Deakin University), Kashem M. Muttaqi `[通讯]` (University of Wollongong)

**通讯引用:** 19212 | [OpenAlex ID](https://openalex.org/A5000829452)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了后验预测方差分解框架，利用异方差神经网络与贝叶斯后验推断，将总不确定性分解为知识不确定性与噪声不确定性；

**💡 创新点**

首次将后验预测方差分解与β-NLL训练结合，实现可解释且可操作的AU/EU分离，并给出了评估协议；

**🔧 技术方法**

采用异方差两头神经网络、MC-DropConnect、Bayes by Backprop、深度集成等贝叶斯后验近似方法，并利用Monte Carlo采样与β-NLL损失实现训练与推理；

**📊 数据集**

在合成正弦数据、风机SCADA数据以及英国风电时间序列数据上进行实验验证；

**📈 对比分析**

通过噪声注入/分布移、数据属性验证与数据量扩展三类实验进行比较，结果显示AU随噪声与稀疏性上升而增大，EU随样本增大而递减，且不同后验近似方法在EU的细节表现上存在显著差异；

**⚠️ 局限性**

局限在于未对模型欠拟合或偏差项Δ_bias进行量化，且对非高斯噪声或模型失配场景的适用性有限。

---

## 453. Integrating Chain-of-Thought into Generative Retrieval: A Preliminary Study

**arXiv ID:** 2605.22358 | [PDF](https://arxiv.org/pdf/2605.22358v1)

**作者:** Wenhao Zhang `[一作]` (Shandong University), Pengjie Ren `[通讯]` (Shandong University)

**通讯引用:** 5312 | [OpenAlex ID](https://openalex.org/A5046700486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ThinkGR的生成式检索框架，将链式思考（CoT）与文档ID生成交叉迭代，支持复杂查询的多步推理。

**💡 创新点**

创新点在于：1）引入混合解码策略，在自由思考与受约束检索之间动态切换；2）采用两阶段训练：先通过监督微调对齐思考-检索模式，再通过检索强化学习提升思考质量。

**🔧 技术方法**

使用技术包括：大规模预训练语言模型、动态解码策略、思考-检索对齐的监督微调以及检索驱动的强化学习。

**📊 数据集**

实验使用了四个多跳检索基准数据集（例如HotpotQA、WikiHop、MedHop、COVID-QA）。

**📈 对比分析**

与现有检索方法对比，ThinkGR在四个基准上均实现了最优性能，平均提升约6.86%。

**⚠️ 局限性**

局限性包括：只在公开数据集上验证，缺乏对更大规模、多模态数据的测试；模型规模和推理成本高，且对非常长的链式思考仍存在泛化挑战。

---

## 454. Unified Data Selection for LLM Reasoning

**arXiv ID:** 2605.22389 | [PDF](https://arxiv.org/pdf/2605.22389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 455. From Recognition to Reasoning: Benchmarking and Enhancing MLLMs on Real-World Receipt Document Understanding

**arXiv ID:** 2605.22413 | [PDF](https://arxiv.org/pdf/2605.22413v1)

**作者:** Yandi Wang `[一作]` (Zhejiang University), Jun Chen `[通讯]` (Zhejiang University)

**通讯引用:** 116645 | [OpenAlex ID](https://openalex.org/A5100450146)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大规模真实收据基准ReceiptBench，构建了10,656张多类型、多语言收据的细粒度标注，并设计了四层级信息抽取子任务；

**💡 创新点**

创新点包括：①引入四层级抽取体系（感知、格式化、语义推理、结构解析）与混合评测协议；②提出Metric‑Aware Group Relative Policy Optimization (GRPO) 将评测指标直接转化为强化学习奖励，实现结构一致性与反假设的优化；

**🔧 技术方法**

采用了多模态大语言模型（如Qwen3‑VL、InternVL3）进行监督微调（SFT）+ GRPO强化学习；

**📊 数据集**

使用的主要数据集为ReceiptBench，涵盖10,656张真实收据，19个字段，4类文档类型；

**📈 对比分析**

在该基准上实验发现，SFT+GRPO的Qwen3‑VL‑8B实现整体F1≈0.795，明显优于GPT‑5、Gemini‑3‑Pro等专有模型，同时对结构解析和推理任务提升显著；

**⚠️ 局限性**

局限性：数据以英语为主（≈97%），低资源语言覆盖不足；未做严重视觉退化或对抗样本的系统评估；GRPO训练成本高，易在小模型上产生奖励坍塌。

---

## 456. Translating Signals to Languages for sEMG-Based Activity Recognition

**arXiv ID:** 2605.22403 | [PDF](https://arxiv.org/pdf/2605.22403v1)

**作者:** Ming Wang `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**通讯引用:** 39440 | [OpenAlex ID](https://openalex.org/A5100361885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种将 sEMG 信号映射为类似语言的“sEMG 语言”，并利用大型语言模型（LLaMA‑13B）进行活动识别。

**💡 创新点**

创新点在于将 VQ‑VAE 与 Lewis 信号游戏相结合，构建迭代学习的语言出现机制，并加入 Zipf 法则、上下文敏感性约束以及残差自适应 token 分配，以提升生成语言的表达力和判别性。

**🔧 技术方法**

主要技术包括 sEMG 导向的 VQ‑VAE 离散化、迭代学习与语言演化、Zipf 与上下文约束、残差自适应 token 分配，以及在 LLaMA‑13B 上的 LoRA 微调。

**📊 数据集**

实验使用公开的 GRABMyo 和 NinaPro DB2 两大手部/腕部肌电数据集。

**📈 对比分析**

在这两数据集上与 TCN、GRU、Informer、STET 等多种基线对比，LLM‑sEMG 在整体精度上分别提升约 4.4%（GRABMyo）和 4.0%（NinaPro DB2），达到最高 95.14%/93.17% 的准确率。

**⚠️ 局限性**

局限包括对大规模 sEMG 数据的依赖、生成语言与真实语言之间的可解释性有限，以及模型对新活动类别迁移能力尚未充分验证。

---

## 457. AgroTools: A Benchmark for Tool-Augmented Multimodal Agents in Agriculture

**arXiv ID:** 2605.22366 | [PDF](https://arxiv.org/pdf/2605.22366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 458. Cross-Subject EEG Emotion Recognition Based on Temporal Asynchronous Alignment Contrastive Learning

**arXiv ID:** 2605.22379 | [PDF](https://arxiv.org/pdf/2605.22379v1)

**作者:** Ying Xie `[一作]` (Sun Yat-sen University), Mengting Liu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3040 | [OpenAlex ID](https://openalex.org/A5100620509)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了基于时间异步对齐的对比学习框架（TA2CL），通过局部匹配与异步对比损失显著提升跨被试脑电情绪识别性能

**💡 创新点**

创新点在于：①将NLP中的ColBERT后期交互机制迁移至EEG，对齐方式从全局硬对齐转为局部软对齐；②设计了Async-InfoNCE损失，结合TopK匹配与均值聚合，解决时序偏移和噪声问题；③在Encoder层融合多尺度时空卷积与注意力，进一步增强情绪特征表达

**🔧 技术方法**

技术手段包括：深度卷积+多尺度扩张卷积、通道注意力、Temporal Token化、Late Interaction（MaxSim+TopK）局部匹配、Async-InfoNCE对比损失、预训练+轻量化分类器

**📊 数据集**

使用公开数据集FACED（9类/2类）、SEED（三类）和SEED‑V（五类）进行交叉被试验证

**📈 对比分析**

与DE+MLP、TCA、DS‑AGC、GCPL、SimCLR、CLISA、PPDA、DAEST等方法对比，TA2CL在所有任务均取得最高准确率：FACED-2 79.5%（vs 78.3% DAEST）、FACED-9 64.5%（vs 62.4%）、SEED 86.4%（vs 86.1%）、SEED‑V 70.1%（vs 68.2%）

**⚠️ 局限性**

局限性：对极细粒度情绪分类提升有限；实验仅涵盖三组数据集，未验证在更大规模或多模态环境下的泛化；模型在实时BCI场景中的推理速度与资源需求仍待进一步优化

---

## 459. VeriScale: Adversarial Test-Suite Scaling for Verifiable Code Generation

**arXiv ID:** 2605.22368 | [PDF](https://arxiv.org/pdf/2605.22368v1)

**作者:** Yifan Bai `[一作]` (Shanghai Jiao Tong University), Tao Luo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7128 | [OpenAlex ID](https://openalex.org/A5034211225)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VeriScale 框架，用对抗性实现扩展和精简测试集评估可验证代码生成

**💡 创新点**

利用对抗性实现驱动测试扩展与约简，实现高质量正负测试并保持判别力

**🔧 技术方法**

结合 LLM 种子生成、类型感知变异、预条件引导分类、对抗实现合成与集合覆盖约简

**📊 数据集**

基于原始可验证代码生成基准（如 3c），构建 3c×83 与 3c×14 两个扩展版数据集

**📈 对比分析**

通过与八大 LLM 的 SpecGen/CodeGen 评测对比，扩展版显著降低通过率（Spec下降 24% 以上），轻量版仅略增评测时长且保持判别力

**⚠️ 局限性**

依赖真预条件与参考实现；对抗实现质量受 LLM 推理与类型约束限制；扩展过程计算成本高

---

## 460. AMUSE: Anytime Muon with Stable Gradient Evaluation

**arXiv ID:** 2605.22432 | [PDF](https://arxiv.org/pdf/2605.22432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 461. BioFormer: Rethinking Cross-Subject Generalization via Spectral Structural Alignment in Biomedical Time-Series

**arXiv ID:** 2605.22468 | [PDF](https://arxiv.org/pdf/2605.22468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 462. Scaling Observation-aware Planning in Uncertain Domains

**arXiv ID:** 2605.22364 | [PDF](https://arxiv.org/pdf/2605.22364v1)

**作者:** Adrian Zvizdenco `[一作]` (Technical University of Denmark), Christoph Matheja `[通讯]` (Technical University of Denmark)

**通讯引用:** 781 | [OpenAlex ID](https://openalex.org/A5089189715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种基于 SMT 参数综合与子符号化技术的可扩展观察感知规划框架，用于解决不确定域中的感知器选择与定位观测问题（SSP 与 POP）。

**💡 创新点**

创新点在于将可观测性问题转化为带类型参数的马尔可夫链（tpMC）并通过本地布尔编码、约束松弛及增量修复策略，显著降低了 SMT 求解的非线性开销；同时设计了基于原子可区分组的启发式枚举算法（A_𝔊），实现了对可观测性函数空间的有效分解与搜索。

**🔧 技术方法**

主要技术包括 SMT 参数综合（EQR）实现、原子可区分组划分、布尔型卡迪纳利约束、预算修复的多射击求解、以及 POMDP 评估 oracle（基于 SMT 与 Storm）。

**📊 数据集**

实验使用了标准的 POMDP benchmark，包括 3×3 与 4×3 网格世界、迷宫模型（M(k)）、以及 G(k) 迷宫式网格，覆盖不同状态数、行动数与预算设置。

**📈 对比分析**

与原始 SMT 编码相比，改进方案在运行时间上提升约 1000 倍，实例规模提升 75 倍；而基于 A_𝔊 的分解策略在运行时间上提升 10^3 倍，能够求解规模约 100 倍更大的实例，显著优于先前工作。

**⚠️ 局限性**

局限性包括：A_𝔊 在预算低于最小可观测预算（B < B^*）时不完整；求解仍受 SMT 求解器不稳定性影响；且随着原子可区分组数量的指数增长，枚举规模可能不可行；此外，实验仍集中在特定网格与迷宫拓扑，未验证更复杂结构。

---

## 463. Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings

**arXiv ID:** 2605.22391 | [PDF](https://arxiv.org/pdf/2605.22391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 464. Minimum Description Length based Granular-Ball Tree Regularization for Spectral Clustering

**arXiv ID:** 2605.22410 | [PDF](https://arxiv.org/pdf/2605.22410v1)

**作者:** Zeqiang Xian `[一作]` (Gannan Normal University), Wenjing Qiu `[通讯]` (Gannan Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于最小描述长度（MDL）的颗粒球树正则化谱聚类方法（MDL-GBTRSC），通过MDL驱动的颗粒球划分和互惠邻居连续性正则化，构建样本级相似度图，并在此图上完成谱聚类。

**💡 创新点**

创新点在于：① 将颗粒球树与谱聚类统一在MDL框架下；② 通过互惠邻居连续性对颗粒球划分进行正则化，避免破坏可靠的局部连接；③ 使用稳定叶球的编码尺度和共享邻居桥码在不需要额外阈值的前提下调节弱桥连边；④ 使得局部区域信息能回馈到样本级图中，提升聚类鲁棒性。

**🔧 技术方法**

使用技术包括：最小描述长度（MDL）理论、颗粒球（Granular Ball）计算、互惠kNN图、稀疏相似度计算、谱聚类（图拉普拉斯特征分解）、连通分量分析、eigengap估计以及归一化处理。

**📊 数据集**

数据集：UCI 18个真实基准数据集（样本 106–17898，特征 4–617，类别 2–26）和 20 个人工合成数据集（包含非凸、嵌套、桥接、噪声等多种结构）。

**📈 对比分析**

与 SC-kNN、SC-RBF、GBCT、MDMSC、GBSC、GMM-SC 等方法进行比较。实验表明在真实数据集上 MDL-GBTRSC 的平均 ARI 0.5910、NMI 0.6419 均位于首位；在合成数据集上平均 ARI 0.9568、NMI 0.9639 亦为最优，显示出对复杂结构的高适应性。

**⚠️ 局限性**

限制：① 计算成本高于轻量级方法（如 SC-kNN、GBSC），尤其在邻接图构建和 MDL 评估上；② 依赖初始邻接图，对高维噪声或重叠数据敏感；③ 需要预先给定聚类数 K，自动估计仍待进一步研究；④ 对大规模或流式数据的可扩展性尚不成熟，需要近似邻接和并行化改进。

---

## 465. Making the Discrete Continuous: Synthetic RAW Augmentations for Fine-Grained Evaluation of Person Detection Performance in Low Light

**arXiv ID:** 2605.22455 | [PDF](https://arxiv.org/pdf/2605.22455v1)

**作者:** Valeria Pais `[一作]` (University of Glasgow), Bruno Sanguinetti `[通讯]` (Dotphoton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并使用基于物理的RAW图像低光衰减技术生成合成低光样本，评估并对比其在自动驾驶场景下对人行检测模型的性能表现。

**💡 创新点**

创新在于将RAW电荷映射与泊松-高斯噪声模型相结合，提出可持续的低光合成方法，并展示合成低光数据与真实低光数据在检测器指标上几乎不可区分。

**🔧 技术方法**

采用RAW光照衰减算法、Poisson-Gaussian噪声模型、Cascade R-CNN+ConvNeXt检测框架以及AP/mAP评估指标。

**📊 数据集**

使用AODRaw数据集（Sony A7M4 RAW图像）及其生成的合成低光样本。

**📈 对比分析**

通过将光照强度分段并分别计算真实与合成样本的AP/mAP进行对比，结果显示在极低光照下模型性能急剧下降，且合成数据与真实数据在大多数指标上统计无显著差异；但在高IOU阈值下仍出现区别。

**⚠️ 局限性**

限制包括真实低光样本稀缺导致评估分辨率有限；合成方法对标签质量和高IOU误差敏感；实验范围仅限于人行检测，未验证对其他目标类别的泛化。

---

## 466. Moment-Reenacting: Inverse Motion Degradation with Cross-shutter Guidance

**arXiv ID:** 2605.22423 | [PDF](https://arxiv.org/pdf/2605.22423v1)

**作者:** Ji Xiang `[一作]` (University of Tokyo), Zheng Yinqiang `[通讯]` (University of Tokyo)

**通讯引用:** 4567 | [OpenAlex ID](https://openalex.org/A5100698163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `8d10c613-917e-4880-9716-17789f50e119` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的双快门（全局快门 + 行滚动快门）成像框架，通过同时利用全局快门的模糊图像和行滚动快门的扭曲图像来逆向运动退化，恢复高帧率清晰视频。

**💡 创新点**

创新点包括：
1) 设计了同步的双快门拍摄系统与对应的真实数据集（realBR），克服仅靠合成数据导致的泛化问题；
2) 将模糊分解与滚动快门时间超分两项任务融合，利用两种退化互补信息；
3) 构建双流运动解释模块，分别提取全局时间上下文与局部运动细节，并通过遮挡对齐/聚合实现两流互促；
4) 引入自提示（motion‑residue prompt）与区域自适应蒸馏，显著提升运动估计与帧重建的精度与一致性；
5) 将双快门方案扩展到窄基线立体 Blur‑RS，兼顾硬件成本与性能。

**🔧 技术方法**

主要技术手段包括：
- 双快门同步捕捉与光学对齐；
- 基于时间位姿编码的滚动快门特征增强；
- 双流网络（Blur‑branch + RS‑branch）与光学变形对齐/聚合模块；
- 逐步迭代的运动解释与自提示重建模块；
- 区域自适应蒸馏（动态区、边缘区、低置信度区）实现无监督运动监督；
- 轻量化变体与多尺度融合以降低算力。

**📊 数据集**

使用了两类数据集：
- realBR：84条街景真实双快门（GS‑Blur + RS）对齐高帧率（500fps）GT 的三轴相机采集数据；
- GOPRO‑BR：基于GOPRO 240fps 视频合成的双快门对齐数据，做跨域验证；
- 第三方测试集（不同传感器、噪声、视差设置）用于评估泛化。

**📈 对比分析**

在 realBR、GOPRO‑BR 以及第三方数据上与现有最佳方法（如 RIFE_B、IFED_B、RSSR、CVR、LBCNet、RIFE_BR、IFED_BR 等）进行对比，实验显示：
- PSNR/SSIM/LPIPS 上平均提升 1–3 dB，远超单模态或合成数据的方案；
- 运动一致性指标（tOF、tLP）明显改善，帧间抖动减少；
- 轻量化版本在保持 96% 以上性能的同时，参数量仅为原始模型的 49%，推理速度提升至 1.3×；
- 在立体 Blur‑RS 扩展中仍保持对比优势。

**⚠️ 局限性**

局限性包括：
- 需要至少双摄像头或多相机光学布置，硬件成本和同步校准仍是部署壁垒；
- 对极端低照度或高噪声行滚动快门仍有一定性能下降；
- 立体方案在较大视差范围下仍会出现细节损失；
- 当前网络主要针对单帧输入的两种退化，尚未针对多帧动态变化或多模态（事件、光流）进一步融合；
- 训练过程需要高质量对齐的同步数据集，合成数据仍存在一定的分布偏差。

---

## 467. Target-Aligned Bellman Backup for Cross-domain Offline Reinforcement Learning

**arXiv ID:** 2605.22376 | [PDF](https://arxiv.org/pdf/2605.22376v1)

**作者:** Wei Liu `[一作]` (Jilin University), Ting Long `[通讯]` (Jilin University)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5107922744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对跨域离线强化学习，提出一种通过评估源域转移样本对目标域Bellman备份一致性的权重机制（Target‑Aligned Bellman Backup, TABB），实现对源域数据的筛选与加权，并将加权后的源域数据与目标域数据共同用于目标策略训练。

**💡 创新点**

创新点在于：①用目标域Bellman目标差异（TBM）替代传统的转移相似性度量来评估源样本的可转移性；②在两阶段学习中先联合训练编码器和目标域预测器，再仅用目标域数据微调预测器，从而使TBM更贴近目标域动态；③通过TBM对源样本进行软加权，仅在Critic更新中利用源样本，避免直接使用源域动态导致的负迁移。

**🔧 技术方法**

技术方法包括：隐空间编码器（ϕ、ψ）、多层感知器目标域预测器f_ref、TBM度量、基于指数函数的权重计算、IQL风格的Critic与Value更新以及优势加权回归的策略更新。

**📊 数据集**

实验使用D4RL的多种机器人控制数据集（HalfCheetah、Hopper、Walker2d、Ant、Pen、Door）在形态、运动学、摩擦等多种动态迁移场景下构建源域与目标域数据集；源域数据为1M条，目标域仅5K条。

**📈 对比分析**

与多种跨域离线RL基线（DARA、BOSA、SRPO、OTDF、DROCO、MOBODY）以及IQL混合数据基线对比，TABB在12个不同任务中表现最佳，平均提升约29%（对MOBODY的提升为29%），在大多数单项任务上超越其他方法；在鲁棒性实验中对不同摩擦强度和数据质量组合也表现出显著优势。

**⚠️ 局限性**

局限性在于：①假设源域与目标域共享相同的状态与动作空间；②依赖于对目标域Bellman目标的近似预测器，若目标域数据极度稀缺或预测误差较大，TBM可能失效；③仅在Critic更新中使用源样本，未充分探索源域数据对策略提升的潜在正面影响。

---

## 468. Boundary-targeted Membership Inference Attacks on Safety Classifiers

**arXiv ID:** 2605.22373 | [PDF](https://arxiv.org/pdf/2605.22373v1)

**作者:** Anthony Hughes `[一作]` (University of Sheffield), Niloofar Mireshghallah `[通讯]` (Carnegie Mellon University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5093022042)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出针对安全分类器的边界定向成员推断攻击，并系统评估其效果。

**💡 创新点**

创新点是将低置信度样本作为攻击目标，通过边界定向选择显著提升成员推断精度。

**🔧 技术方法**

采用LiRA、Loss-based评分、LoRA参数高效微调、Laplace输出噪声等技术。

**📊 数据集**

使用BeaverTails、X-Guard-Train、WildChat、ESConv、Psychotherapy Eval等公开数据集。

**📈 对比分析**

与传统全局MIA比较，边界定向在MI-AUC和TPR@5%FPR上提升约3.5倍，最高MI-AUC超过0.8。

**⚠️ 局限性**

局限包括仅针对二分类安全任务，未考虑多标签或更复杂隐私模型，噪声防御需调参且可能影响分类性能。

---

## 469. KAPPS: A knowledge-based CPPS Architecture for the Circular Factory

**arXiv ID:** 2605.22457 | [PDF](https://arxiv.org/pdf/2605.22457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 470. SADGE: Structure and Appearance Domain Gap Estimation of Synthetic and Real Data

**arXiv ID:** 2605.22467 | [PDF](https://arxiv.org/pdf/2605.22467v1)

**作者:** Patryk Bartkowiak `[一作]` (Adam Mickiewicz University), Wojtek Palubicki `[通讯]` (Adam Mickiewicz University)

**通讯引用:** 870 | [OpenAlex ID](https://openalex.org/A5016232337)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SADGE，一个无需训练即可评估合成图像数据集对下游视觉任务效能的零射击指标。

**💡 创新点**

创新点在于将外观相似度（如DINOv3嵌入）与几何一致性（如MASt3R匹配）通过受限双线性交互融合，显著提升对真实数据迁移性能的预测准确性。

**🔧 技术方法**

使用的技术包括：预训练视觉嵌入（DINOv3、CLIP、SigLIP等）、几何匹配器（MASt3R、LoFTR、SuperPoint+LightGlue）、统计归一化以及低容量双线性融合模型。

**📊 数据集**

实验基于五个公开合成‑真实对照基准：DIMO、VKITTI2、RarePlanes、TUD‑L 和 ASD，共 15 个数据集变体，涵盖光照、天气、姿态、视角等多种域差异。

**📈 对比分析**

评价方式：计算每个变体的下游任务性能（检测、分割、姿态估计）与SADGE分数的 Pearson 与 Spearman 相关系数。SADGE 在所有变体中实现 Pearson r=0.879、Spearman ρ=0.768，明显优于仅使用外观或几何单一指标。

**⚠️ 局限性**

局限性包括：基准覆盖有限，只涉及五个数据集；对预训练模型的依赖，若目标域与预训练数据差异过大可能降低可靠性；以及对大规模合成数据的实时计算仍存在一定开销。

---

## 471. In Silico Modeling of the RAMPHO Buffer: Dissociating Informational and Energetic Masking via Phonetic Entropy in Deep Neural Networks

**arXiv ID:** 2605.22465 | [PDF](https://arxiv.org/pdf/2605.22465v1)

**作者:** Stefan Bleeck `[一作]` (University of Southampton), Stefan Bleeck `[通讯]` (University of Southampton)

**通讯引用:** 1427 | [OpenAlex ID](https://openalex.org/A5005337719)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在多说话者环境下，利用自监督语音模型 wav2vec 2.0 的帧级语音熵作为 RAMPHO 缓冲区的代理，量化并区分信息遮蔽（IM）与能量遮蔽（EM）对听觉认知负担的影响。

**💡 创新点**

创新点在于将语音模型的熵值作为客观认知负荷指标，揭示在高信噪比下消除语义负载可显著降低信息遮蔽，而在低信噪比下相位随机化导致时间窗消失，反而提高能量遮蔽，从而提出认知‑声学 Pareto 优化的概念。

**🔧 技术方法**

使用技术包括自监督模型 wav2vec 2.0、CTC 解码器、帧级 Softmax 熵计算、相位随机化的“浓缩盾”掩蔽、SNR 梯度混合，以及语音能量与频谱处理。

**📊 数据集**

实验数据集为一段英文目标叙述，与三种掩蔽（原声可懂英语、相位随机化的浓缩盾、语音形状噪声）混合，SNR 范围覆盖 0、5、10、15、20 dB 与 100 dB 基准。

**📈 对比分析**

通过比较三种掩蔽在不同 SNR 下的语音熵曲线，验证了熵值能够精确区分 IM 与 EM：高 SNR 时浓缩盾熵值接近噪声基准，表明信息遮蔽被消除；低 SNR 时浓缩盾熵值最高，说明能量遮蔽占优。实验结果表明该熵指标可客观评估认知负荷。

**⚠️ 局限性**

局限性包括：wav2vec 2.0 采用双向 Transformer，缺乏实时和有限上下文的约束，未模拟人类回声记忆衰减和工作记忆容量；实验仅在模拟环境下进行，缺乏真实听力人群验证。

---

## 472. From Correlation to Cause: A Five-Stage Methodology for Feature Analysis in Transformer Language Models

**arXiv ID:** 2605.22462 | [PDF](https://arxiv.org/pdf/2605.22462v1)

**作者:** Caleb Munigety `[一作]` `[通讯]` (Independent Researcher), Caleb Munigety (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一个五阶段因果特征分析方法，包括探针设计、特征提取、因果验证、鲁棒性测试和部署集成，并在GPT‑2 small的间接宾语识别（IOI）任务上完整实现该流程；

**💡 创新点**

创新点是将激活补丁、稀疏自编码器、自然语言自动编码器等技术按因果导向顺序组合为系统化工作流，并揭示“中间因果介质”现象，即特征既具有选择性又仅部分因果、检测鲁棒性与因果鲁棒性不一致等。

**🔧 技术方法**

使用了激活补丁、稀疏自编码器（SAE）、自然语言自动编码器（NLA）评估框架、特征消融干预、ROC曲线、成本模型等多种技术。

**📊 数据集**

使用的数据集为GPT‑2 small（124M参数）模型的IOI任务提示，训练集约2000个IOI实例，评估集约500+实例，并构造了多种分布偏移（内容同义词、未见姓名、提示重构）。

**📈 对比分析**

通过与先前工作对比，展示单个特征消融可降低0.7 logits、整体准确率仅降至98%；FVE与可靠性评估显示所解释的特征仅覆盖约30%方差；在成本模型下，阈值15的SAE‑only监测器在假设成本下可节约99.1%费用。

**⚠️ 局限性**

主要限制包括模型规模较小、任务为模板化IOI、SAE规模与训练步数有限、未在大模型或更自然任务上验证、成本评估基于假设、未训练真正的NLA、鲁棒性测试覆盖面有限。

---

## 473. Steins;Gate Drive: Semantic Safety Arbitration over Structured Futures for Latency-Decoupled LLM Planning

**arXiv ID:** 2605.22456 | [PDF](https://arxiv.org/pdf/2605.22456v1)

**作者:** Anjie Qiu `[一作]` (RPTU University Kaiserslautern-Landau), Hans D. Schotten `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 Steins;Gate Drive，一种基于 LLM 的延迟解耦规划器-运行时架构，通过角色化世界线生成与可撤销的预测合同，使慢速语义推理能够在高速驾驶循环中可复用。

**💡 创新点**

创新点在于将 LLM 的推理包装成可回收的 Typed Forecast，配备有效期、失效条件、回退动作和运行时权限，从而在保持安全性的前提下实现推理延迟的折算。

**🔧 技术方法**

采用了角色化世界线生成器、两时尺度双代理 LLM（策略层与运行时层）、预测缓冲、原子条件检查（validity/abort）、基于 TTC 与 gap 的安全评估以及 GPT‑5.4 mini 语言模型。

**📊 数据集**

实验基于仿真器（如 Carla/Autoware）中的正常高速公路场景，使用 IDM 流量，10 个随机种子、20 步决策周期，采用现实高速公路速度配置，数据来源为匹配种子协议下的仿真轨迹。

**📈 对比分析**

通过匹配种子对比无缓冲反应式 LLM、确定性重放和双代理缓冲三种模式，评估指标包括无碰撞率、平均速度、动作切换率、TTC 风险、有效延迟和令牌成本。结果显示双代理模式在保持 100% 无碰撞的同时，将有效延迟从 +3.07 s 降至约 -0.01 s（H≈4 s），速度略下降但保持可接受，且低分支选择率显著提升。

**⚠️ 局限性**

局限性包括仅在简化 IDM 流量、无感知不确定性、单一正常跑道的仿真环境下评估，缺乏多场景、多模型和真实世界转移验证；未来工作需扩展种子数量、引入场景标签、验证预测合同的安全性以及结合学习式视觉世界模型。

---

## 474. FastTab: A Fast Table Recognizer with a Tiny Recursive Module and 1D Transformers

**arXiv ID:** 2605.22422 | [PDF](https://arxiv.org/pdf/2605.22422v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 475. S2ED: From Story to Executable Descriptions for Consistency-Aware Story Illustration

**arXiv ID:** 2605.22448 | [PDF](https://arxiv.org/pdf/2605.22448v1)

**作者:** Sijing Yin `[一作]` (University of Auckland), Qian Liu `[通讯]` (University of Auckland)

**通讯引用:** 3542 | [OpenAlex ID](https://openalex.org/A5037327117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、模型无关的Prompt‑Layer框架S2ED，用于将完整故事编译成可执行的帧级描述，从而在多帧故事插图中显式传递身份、布局和情感状态，显著降低跨帧漂移；

**💡 创新点**

核心创新在于：①把故事“编译”为可执行描述，构建显式的视觉状态接口；②通过分层Agent（情节分割、身份对齐、视觉丰富）实现对状态的递归传播；③在Prompt层而非模型内部维护状态，保持可编辑性与跨模型迁移性；

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）实现故事分割与实体解析；结构化知识库（角色库、外观属性库）进行身份对齐；Prompt‑Carried State Propagation 通过递归更新Prompt；以及常见的文本到图像生成器（如FLUX‑1）作为渲染后端；

**📊 数据集**

使用公开收集的《Flintstones》故事数据集（166个故事）与新构建的《Shakoo Maku》插画故事集（包含固定IP角色），并配备相应的角色属性库；

**📈 对比分析**

与多种基线（PlainPrompt、TokenInject、LayoutPrompt、全故事提示 GPT‑5 / Gemini‑2.5 Pro 以及基于训练的 StoryDiffusion）进行对比。S2ED 在自动指标（CLIPScore、Char‑F1、F‑Acc、mAP）以及人类评估（MOS 与配对偏好）上均优于所有基线，尤其在角色一致性与布局稳定性方面提升显著；

**⚠️ 局限性**

主要局限包括：①多实体场景下身份漂移更难控制；②属性泄漏导致不同角色间外观混淆；③评价数据集与指标偏向特定风格，缺乏更广泛的故事多样性与情感连贯性评测。

---

## 476. Monotone Erasure Codes

**arXiv ID:** 2605.22426 | [PDF](https://arxiv.org/pdf/2605.22426v1)

**作者:** Vivien Bammert `[一作]` (University of Bern), Christian Cachin `[通讯]` (University of Bern)

**通讯引用:** 11584 | [OpenAlex ID](https://openalex.org/A5054120489)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了能够满足任意单调访问结构的单调擦除码，以实现对不均匀信任模型下的可靠存储与通信

**💡 创新点**

创新点在于：1) 将单调访问结构映射为单调布尔公式并构造相应单调擦除码；2) 设计了两种线性单调擦除码构造方法，其中一种通过MDS码实现最优冗余；3) 针对分区访问结构给出了高效的最优片段分配算法；4) 将单调擦除码应用于通用异步可验证信息分发（GAVID），从而实现对任意拜占庭仲裁系统的通信高效可靠广播与共识。

**🔧 技术方法**

使用了有限域线性代数、Vandermonde矩阵与MDS码、阈值布尔公式、Kronecker积、线性规划以及可靠集合（kernel）和可靠集（reliable set）的概念

**📊 数据集**

未使用公开数据集；所有评估均为理论分析与复杂度证明

**📈 对比分析**

通过理论证明与复杂度分析比较传统阈值MDS码与已有的AVID协议，显示在冗余度（β）与消息/通信复杂度方面更优，尤其在分区访问结构中可实现接近零冗余且消息复杂度为O(n^2)

**⚠️ 局限性**

局限性包括：1) 对任意访问结构的最优构造仍需解线性规划，复杂度随访问集合数量呈指数增长；2) 对分区访问结构的最优算法仅适用于树形结构；3) 在实践中需要大域数与高计算开销；4) 依赖哈希碰撞难度保证一致性，若哈希弱则安全受限

---

## 477. ASAP: Attention Sink Anchored Pruning

**arXiv ID:** 2605.22372 | [PDF](https://arxiv.org/pdf/2605.22372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 478. Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts

**arXiv ID:** 2605.22446 | [PDF](https://arxiv.org/pdf/2605.22446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 479. Hybrid Kolmogorov-Arnold Network and XGBoost Framework for Week-Ahead Price Forecasting in Australia's National Electricity Market

**arXiv ID:** 2605.22387 | [PDF](https://arxiv.org/pdf/2605.22387v1)

**作者:** Houxuan Zhou `[一作]` (Monash University), Hao Wang `[通讯]` (Monash University)

**通讯引用:** 28683 | [OpenAlex ID](https://openalex.org/A5100446064)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种混合 Kolmogorov–Arnold 网络（KAN）与 XGBoost 的框架，用于澳大利亚全国电力市场（NEM）的周前电价预测。

**💡 创新点**

创新点在于将 KAN 的全局非线性表示能力与 XGBoost 的局部鲁棒性结合，通过加权融合同时捕获长期趋势与短期价格波动，并在实验中证明该混合模型优于单一模型。

**🔧 技术方法**

采用 KAN（两层、基于样条的可学习激活函数）、XGBoost（500棵树、深度6、学习率0.05）以及两者的加权融合（α=0.5）进行预测；使用 MAE、RMSE 与相对 MAE 评估性能，并进行极值事件（EVT/POT）分析。

**📊 数据集**

使用 AEMO 提供的 2024‑2025 年澳大利亚 NEM 区域（NSW、VIC、QLD、SA、TAS）的价格、负荷、净交换、可再生发电预测及天气变量（温度、风速、湿度、云量）构建的数据集。

**📈 对比分析**

与 Naïve、SARIMAX、LSTM、单独 KAN 与 XGBoost 进行对比。混合模型在整体 MAE 上比 XGBoost 降低约12%，比 Naïve 降低 50% 以上，且在所有州和极值事件预测中均表现最优。

**⚠️ 局限性**

局限性包括：依赖准确的外部变量（需预测）、仅覆盖一年时间周期、极端价格尖峰仍难以精确捕捉；未来工作建议加入概率预测、适应性集成与更丰富的领域特征。

---

## 480. Efficient Higher-order Subgraph Attribution via Message Passing

**arXiv ID:** 2605.22385 | [PDF](https://arxiv.org/pdf/2605.22385v1)

**作者:** Ping Xiong `[一作]` (Technische Universitaet Berlin), Shinichi Nakajima `[通讯]` (Technische Universitaet Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种线性时间计算子图归因的高阶解释方法 sGNN-LRP，显著降低了传统 GNN-LRP 的指数复杂度，并进一步定义了可折扣的子图归因 Rα，兼顾子图内部与外部路径贡献。

**💡 创新点**

创新点包括：① 将归因目标量化为信息流的边缘概率，利用消息传递（sum‑product）框架直接求解子图归因；② 通过“forward‑hook trick”实现高效实现；③ 引入折扣参数 α 的通用子图归因定义，提升解释的鲁棒性。

**🔧 技术方法**

技术手段主要是：图神经网络（GNN）、层级相关传播（LRP）、消息传递算法（belief propagation）、forward‑hook 以及基于折扣的子图归因公式。

**📊 数据集**

实验使用的公开数据集包括 BA‑2motif、MUTAG、Mutagenicity、REDDIT‑BINARY、Graph‑SST2，模型为 GIN 与 GCN，网络深度从 3 到 7 层不等。

**📈 对比分析**

与基线（Naïve GNN‑LRP、GNNExplainer、Gradient‑heatmap、Grad‑CAM）比较，sGNN‑LRP 在子图归因计算时间上提升了数百到数千倍，且在子图归因相关的节点排序任务（激活与修剪）中往往获得更高的 AUAC/AUPC 分数，证明了其优越性。

**⚠️ 局限性**

局限性包括：① 当前实现仍假设传播矩阵可归一化；② 折扣参数 α 的选择仍需经验或交叉验证，缺乏自动化调优；③ 对于极大图和极深网络，虽然复杂度已降至多项式，但实际内存与时间仍受限；④ 仅在实验中验证了部分任务，缺乏在更多领域（如自然语言、量子化学）上的广泛评估。

---

## 481. ImplicitTerrainV2: Wavelet-Guided Spatially Adaptive Neural Terrain Representation

**arXiv ID:** 2605.22556 | [PDF](https://arxiv.org/pdf/2605.22556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 482. Neural Flow Operators can Approximate any Operator: Abstract Frameworks and Universal Approcimations

**arXiv ID:** 2605.22557 | [PDF](https://arxiv.org/pdf/2605.22557v1)

**作者:** Shuang Chen `[一作]` (Tsinghua University), Xue-Cheng Tai `[通讯]` (Norwegian Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种抽象的连续深度流框架（Neural Flow Framework），可统一描述有限维神经网络和无限维神经算子，并通过两种结构（组合式与分离式）推导出 ResNet 与普通前馈网络及卷积网络和卷积算子的离散实现。

**💡 创新点**

创新点在于：①首次给出流模型在无限维 Hilbert 空间上的普适逼近定理；②在同一框架下实现组合式与分离式流模型的统一；③通过卷积线性层的限制，扩展到卷积算子，得到卷积算子的普适逼近；④通过时间离散化把连续模型自然映射到实际的有限深度网络，解释了 ResNet 与普通架构的本质关联。

**🔧 技术方法**

核心技术包括：连续深度流的微分方程建模、分离式与组合式两种向量场设计、激活函数（Leaky ReLU）满足的隐式关系、分裂式与显式欧拉离散化、ODE/PDE 的存在唯一性与参数连续性证明、Hilbert 空间中的泛化逼近证明。

**📊 数据集**

本文为理论研究，未使用具体数据集，全部工作基于数学证明与抽象构造。

**📈 对比分析**

由于是理论性论文，未进行实验对比；通过证明可知在给定误差 ε 下，存在足够宽度与深度的网络/算子可逼近任何连续映射，满足普适逼近性质。

**⚠️ 局限性**

局限性包括：①仅证明了 Leaky ReLU 类激活函数的可行性，未涵盖更一般激活；②仅给出定性逼近结果，缺乏量化误差界与逼近率；③未在实验中验证理论在实际训练中的可实现性；④对 Transformer 等更复杂架构的适用性尚待进一步扩展。

---

## 483. Relational Linear Properties in Language Models: An Empirical Investigation

**arXiv ID:** 2605.22532 | [PDF](https://arxiv.org/pdf/2605.22532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 484. "Refactoring Runaway": Understanding and Mitigating Tangled Refactorings in Coding Agents for Issue Resolution

**arXiv ID:** 2605.22526 | [PDF](https://arxiv.org/pdf/2605.22526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 485. Scene Abstraction for Lexical Semantics: Structured Representations of Situated Meaning

**arXiv ID:** 2605.22542 | [PDF](https://arxiv.org/pdf/2605.22542v1)

**作者:** Yejin Cho `[一作]` (University of Texas at Austin), Katrin Erk `[通讯]` (University of Massachusetts)

**通讯引用:** 3948 | [OpenAlex ID](https://openalex.org/A5067235438)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Scene Abstraction 框架，用 LLM 通过少量示例生成结构化场景表示（上下文场景和表达式配置），用于捕捉词义的情境化维度。

**💡 创新点**

创新点在于将词义视为可变的情境分布，构建两层结构（事件、实体、环境与针对词的参与事件、属性、情感）并通过 LLM 自动抽取，弥补传统词典、FrameNet 与分布式嵌入对情境维度的隐式编码不足。

**🔧 技术方法**

采用 GPT‑4 等大型语言模型进行少量示例提示生成场景结构，并用 Sentence‑BERT 对抽取的各子结构序列进行嵌入以实现向量空间操作。

**📊 数据集**

数据集包括 COCA‑Scenes（520句、26关键词、4种情境）用于 odd‑scene‑out 评估，以及 105 个关键词–句子对（来自 DWUG‑EN 与 COCA）用于人类偏好评估。

**📈 对比分析**

通过 odd‑scene‑out 任务验证模型，Scene‑Only 与 Text+Scene 在准确率上分别提升至 0.627 与 0.693（相较文本基线 0.575），在人类偏好评估中 Scene Profiles 在 3 个维度上被偏好率高达 86.4%，显著优于 ATOMIC‑based 常规知识图谱。

**⚠️ 局限性**

局限性包括依赖 LLM 的系统偏差与幻觉、情感与解释性推断的主观性、仅在英文小说语料上验证、缺乏多语言与跨文化的普适性。

---

## 486. Dynamic Hypergraph Representation Learning for Multivariate Time Series without Prior Knowledge

**arXiv ID:** 2605.22540 | [PDF](https://arxiv.org/pdf/2605.22540v1)

**作者:** Marco Gregnanin `[一作]` (IMT School for Advanced Studies Lucca), Maurizio Parton `[通讯]` (University of Chieti-Pescara)

**通讯引用:** 493 | [OpenAlex ID](https://openalex.org/A5014573834)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于无先验知识的多变量时序数据的动态超图表示，并使用超图注意力网络结合 LSTM 进行预测。

**💡 创新点**

创新点在于将随机矩阵理论滤波与自注意力社区检测相结合，实现无先验动态超图生成，并将超图卷积与 LSTM 融合为集成模型。

**🔧 技术方法**

采用随机矩阵理论、社区检测、GRU+自注意力、超图卷积网络和 LSTM 等技术。

**📊 数据集**

使用了股票回报、家用电器能耗和城市空气质量三个公开时序数据集。

**📈 对比分析**

与 AT‑LSTM、Transformer、GCN+LSTM 等基线模型比较，能耗与空气质量数据上取得最优或接近最优的 RMSE/MAE/MAPE，股票回报数据上表现略逊。

**⚠️ 局限性**

在高度随机的金融时序中超图信息贡献有限，模型复杂度较高，缺乏实时可解释性。

---

## 487. Quantifying Full-Body Immersion

**arXiv ID:** 2605.22521 | [PDF](https://arxiv.org/pdf/2605.22521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 488. LACO: Adaptive Latent Communication for Collaborative Driving

**arXiv ID:** 2605.22504 | [PDF](https://arxiv.org/pdf/2605.22504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 489. BeLink: Biomedical Entity Linking Meets Generative Re-Ranking

**arXiv ID:** 2605.22501 | [PDF](https://arxiv.org/pdf/2605.22501v1)

**作者:** Darya Shlyk `[一作]` (University of Milan), Lawrence Hunter `[通讯]` (University of Chicago)

**通讯引用:** 9429 | [OpenAlex ID](https://openalex.org/A5041860080)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了名为BeLink的端到端生物医学实体链接系统，采用生成式检索与基于指令微调的集合式重新排序；

**💡 创新点**

首次将集合式指令微调引入生成式重新排序框架，能一次性处理全部候选，大幅提升推理速度与准确率，并结合零样本查询改写提升检索召回；

**🔧 技术方法**

使用SapBERT做密集检索、Faiss向量索引、Qwen3生成式模型进行指令微调、SWIFT框架微调以及GenQR零样本查询改写技术；

**📊 数据集**

评估了8个公开BEL基准，覆盖疾病、化学品、基因、物种四个领域，并链接到CTD、NCBI Gene、NCBI Taxonomy等知识库；

**📈 对比分析**

与BERT跨编码器、零样本LLM重新排序和Qwen3点式重新排序对比，BeLink在Acc@1上提升3%–24%，推理速度约四倍加速，NIL敏感性更稳健；

**⚠️ 局限性**

跨域迁移性能不均衡，基因领域迁移效果差；需要在目标领域微调；模型规模仍较大，推理成本相对较高。

---

## 490. Training-Free Fine-Grained Semantic Segmentations in Low Data Regimes: A FungiTastic Baseline

**arXiv ID:** 2605.22492 | [PDF](https://arxiv.org/pdf/2605.22492v1)

**作者:** Sebastian Cavada `[一作]` (Covision Lab), Lapo Faggi `[通讯]` (Covision Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个无训练的两阶段细粒度语义分割框架：先用SAM3进行宏观类的无监督分割，再用DINOv3提取特征并通过PCA白化后的原型匹配完成细粒度分类。

**💡 创新点**

创新点在于将分割与分类完全解耦，避免在推理时对每个类别重复调用SAM3，同时通过简单的PCA白化显著提升了预训练特征的可比性，提供了低数据场景下细粒度分割的基线。

**🔧 技术方法**

使用的技术包括：SAM3（类无关分割模型）、DINOv3（自监督视觉特征提取器）、PCA+白化预处理、原型匹配（最近邻分类）以及基于宏观标签的阈值处理。

**📊 数据集**

使用FungiTastic数据集，包含约13k张带分割标注的图像，194个细粒度蘑菇种类，重点评估一、少量样本和低数据环境下的表现。

**📈 对比分析**

与原始DINOv3特征及单纯PCA对比，PCA白化后原型匹配在一次样本到数百样本的低数据区间提升了约20%至30%的平均分类准确率（mAcc），在50张样本时mIoU达到30%，并且性能在40–60张样本后趋于饱和。

**⚠️ 局限性**

局限性包括：依赖SAM3对宏观类别的分割质量，原型匹配在极少样本时仍受限；PCA白化虽然有效但需额外计算；当前方法未探索多类别交互或不同视觉骨干的影响；对长尾分布的处理仍不充分。

---

## 491. Represented Is Not Computed: A Causal Test of Candidate Algorithmic Intermediates in a Transformer

**arXiv ID:** 2605.22488 | [PDF](https://arxiv.org/pdf/2605.22488v1)

**作者:** Ishita Darade `[一作]` (MKSSS's Cummins College of Engineering for Women), Sushrut Thorat `[通讯]` (Osnabrück University)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5034511642)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文通过构造一个基数提取（base‑digit extraction）任务，训练并分析Transformer在该任务上的表现，探讨模型内部表示与实际计算之间的关系。

**💡 创新点**

创新点在于揭示了“可解码（represented）”并不必然等同于“被计算（computed）”，通过线性探针与因果干预（attention ablation、key/value patching、稀疏电路搜索）展示了表示与因果路径的解耦。

**🔧 技术方法**

使用的技术包括：Transformer（10 层 GPT‑style）模型、线性探针（probe）分析、注意力遮蔽（attention ablation）、关键/值向量补丁（key/value patching）以及贪婪稀疏电路搜索（sparse circuit search）。

**📊 数据集**

数据集为自定义的算术数据集：N∈{0,…,999}，B∈{2,…,30}，D 为基数位置，所有组合在训练/验证/测试中按数值–基数交叉拆分，保证 held‑out 交叉验证。

**📈 对比分析**

在 held‑out 交叉验证上模型达到 99.83%（±0.17%）的完全准确率；线性探针可在不同层解码出 B^D、N/B^D、⌊N/B^D⌋ 等中间量；但因果干预表明输出主要依赖早期 D‑选择性通信，最终路由与探针预测不一致，说明表示与计算路径不完全重合。

**⚠️ 局限性**

局限性包括：仅分析了基数提取的有限算术任务，未揭示输出层的非线性整合细节；稀疏电路搜索提供的是充分而非最小或唯一解；实验聚焦于单一模型规模，可能不适用于更大或更复杂的Transformer。

---

## 492. When Stronger Triggers Backfire: A High-Dimensional Theory of Backdoor Attacks

**arXiv ID:** 2605.22481 | [PDF](https://arxiv.org/pdf/2605.22481v1)

**作者:** Donald Flynn `[一作]` (University of Oxford), Inbar Seroussi `[通讯]` (Tel Aviv University)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5054816338)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析后门（trigger-based）中毒攻击在高维高比例下对正则化广义线性模型的影响，揭示了触发强度与模型性能之间的三种反直觉现象，并给出了精确理论解析。

**💡 创新点**

首次在 p/n→κ 的高维比例极限下提供触发强度对攻击成功率与干净准确率的定量描述，揭示“更强训练触发并不一定更有效”以及“低方差方向最易被利用”的机制。

**🔧 技术方法**

采用高维极限分析工具（Gaussian‑proxy 固定点系统、凸高斯最小最大定理等）、闭式解析（平方损失）与数值求解（Logistic 等凸损失）以及对齐量的解析。

**📊 数据集**

在 CIFAR‑10（选取两类子集）上实验，并用高斯混合仿真作为代理；还在 ResNet‑18 上验证了相同现象。

**📈 对比分析**

与理论预测对比，实验曲线与理论高度一致；在 CIFAR‑10 上，正则化广义线性模型的干净准确率随训练触发强度递增，攻击成功率先增后减；这些现象在深度网络上也能观察到。

**⚠️ 局限性**

主要局限是理论仅适用于高斯混合数据和凸/线性模型，未覆盖非凸特征映射、多类别情形；同时对触发方向与数据均值正交性做了简化假设。

---

## 493. Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking

**arXiv ID:** 2605.22538 | [PDF](https://arxiv.org/pdf/2605.22538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 494. Implicit Regularization of Mini-Batch Training in Graph Neural Networks

**arXiv ID:** 2605.22480 | [PDF](https://arxiv.org/pdf/2605.22480v1)

**作者:** Clement Wang `[一作]` (Institut Polytechnique de Paris), Thomas Bonald `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 5488 | [OpenAlex ID](https://openalex.org/A5020404511)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了随机节点采样（Random Node Sampling, RNS）在图神经网络（GNN）mini‑batch 训练中的效果，并通过反向误差分析揭示其隐式正则化机制。

**💡 创新点**

创新点在于把 GNN mini‑batch 训练视为优化被采样损失加上梯度方差正则化的“修改目标”，证明 RNS 在此框架下产生梯度方差最小、偏差低的子图，从而实现更稳定的隐式正则化；并将 RNS 提出为一种无结构、易调优的默认采样策略。

**🔧 技术方法**

采用了反向误差分析（Backward Error Analysis）与随机梯度下降（SGD/Adam）的离散动力学、子图采样策略比较、以及在 GraphSAGE、GCN、GAT、SGFormer 等架构上的大规模实验。

**📊 数据集**

实验使用 Open Graph Benchmark 中的 ogbn-arxiv、ogbn-products、pokec 三个大规模节点分类数据集，以及 GraphLand 的七个节点分类数据集（如 hm-categories、tolokers-2 等）。

**📈 对比分析**

通过与全图训练以及邻域采样、ClusterGCN、GraphSAINT、LADIES 等采样方法在相同架构和超参搜索下对比，RNS 在 8/10 个基准上与全图相当或更优；在训练时间上实现 2×–12× 加速，在 GPU 显存占用上实现 3× 以内的节省。

**⚠️ 局限性**

局限性包括需要调节批次数 m，m 对性能敏感；在全局注意力模型（如 SGFormer）或极度稠密/稀疏、极度不平衡任务上表现不佳；以及对 Adam 等自适应优化器的隐式正则化理论尚未完全建立。

---

## 495. Structured-Sparse Attention for Entity Tracking with Subquadratic Sequence Complexity

**arXiv ID:** 2605.22476 | [PDF](https://arxiv.org/pdf/2605.22476v1)

**作者:** Hangyue Zhao `[一作]` (ESPCI PSL), Alexandre Allauzen `[通讯]` (ESPCI PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对实体追踪任务，提出了基于块化评估的可逆注意力（Block‑ChaCAL），在保持多跳传播的同时降低了计算复杂度。

**💡 创新点**

创新点在于：①发现实体追踪中的注意力矩阵呈块状稀疏结构；②在块内部精确计算可逆注意力，跨块通过压缩系统高效路由；③实现了子二次（O(n^{4/3}d)）的时间复杂度。

**🔧 技术方法**

使用了可逆（resolvent）注意力操作、块化分解、下采样/上采样、前向替代求逆、以及对比实验中的稀疏化Top‑k诊断。

**📊 数据集**

主要在合成的Toy和Boxes实体追踪数据集；在语言模型任务上使用WikiText‑103和OpenWebText；在代码推理上使用Copilot数据集；在长序列序列到序列任务上使用SCROLLS。

**📈 对比分析**

与全密集Transformer、ChaCAL单层、滑动窗口和BigBird稀疏注意力等方法对比，Block‑ChaCAL在保持接近或等同的准确率/EM的同时，平均推理时间比ChaCAL快12–29%，比5层全密集Transformer快约2.4×；在从头训练的GPT‑2中可获得比密集注意力更低的困惑度。

**⚠️ 局限性**

局限包括：①对块化稀疏结构的依赖，非局部依赖任务效果有限；②块大小选择的权衡导致性能呈U形曲线；③需足够的注意力头数以匹配并行属性数，头数不足时准确率下降；④与预训练权重不匹配时需要额外适配，尤其在Encoder‑Decoder架构下效果不如预期。

---

## 496. Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers

**arXiv ID:** 2605.22471 | [PDF](https://arxiv.org/pdf/2605.22471v1)

**作者:** Maya Bechler-Speicher `[一作]` (Meta AI), Joan Bruna `[通讯]` (New York University)

**通讯引用:** 21789 | [OpenAlex ID](https://openalex.org/A5112569280)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究Transformer在图学习中tokenization对表达力的本质影响，并证明不同tokenization导致不同深度需求、不可逆性与信息丢失。

**💡 创新点**

提出tokenization是Transformer表达力的关键维度，给出深度分离、不可逆性和随机游走tokenization本质有损的理论定理。

**🔧 技术方法**

使用复杂性理论、谱分解、梯度条件分析结合Transformer架构，并在节点级tokenization（adjacency、spectral、random‑walk）上进行实验。

**📊 数据集**

使用GraphBench的MaxClique、TopoOrd、EC‑5，以及OGB的分子数据集HIV、BBBP、BACE、Tox21和ZINC。

**📈 对比分析**

与DeepSet、GIN等基线对比，评估不同tokenization在不同任务上的表现，发现局部任务偏好Adjacency，全局任务偏好Spectral，随机游走提供补充信息，组合tokenization可提升性能。

**⚠️ 局限性**

受限于理论证明在实际Transformer中可能被宽度、数值精度等因素影响；随机游走tokenization不可逆导致全局任务受限；截断导致不可恢复的损失；实验规模受GPU内存限制。

---

## 497. Meta-Learning for Rapid Adaptation in Reference Tracking of Uncertain Nonlinear Systems

**arXiv ID:** 2605.22513 | [PDF](https://arxiv.org/pdf/2605.22513v1)

**作者:** Jiaqi Yan `[一作]` (Beihang University), Alisa Rupenyan `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 1893 | [OpenAlex ID](https://openalex.org/A5058099719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了基于iMAML的元学习框架，旨在利用离线源系统数据快速适应并控制具有不确定非线性动力学的目标系统，支持间接模型识别+MPC和直接DQN两种控制实现；

**💡 创新点**

创新点在于：①设计了通用的元学习框架，兼容多种学习算法；②利用隐式MAML实现高效的二级优化与梯度计算；③将模型识别与控制策略直接嵌入元学习，实现快速少样本适应；

**🔧 技术方法**

核心技术包括iMAML、隐式函数定理实现的外梯度计算、神经状态空间模型（NSSM）、模型预测控制（MPC）以及深度Q网络（DQN）等；

**📊 数据集**

使用的实验数据集：离线源系统的10个参数可变的van der Pol振荡器和球在板上的模拟系统；目标系统数据为有限长度的真实/模拟轨迹；

**📈 对比分析**

通过与MAML、无预训练监督学习以及仅在目标系统上训练的DQN进行对比，实验表明iMAML+MPC在跟踪误差上显著优于基线，DQN预训练在实物板上可将平均成本降低约73%；

**⚠️ 局限性**

局限性包括：需要源目标系统高度相似以保证快速适应；对高维/复杂系统的可扩展性有限；DQN的理论稳定性尚未完全证明；对极端噪声和模型偏差的鲁棒性仍需进一步研究。

---

## 498. Exact Hidden Paths in Noisy High Dimensional Path Spaces

**arXiv ID:** 2605.22477 | [PDF](https://arxiv.org/pdf/2605.22477v1)

**作者:** Victor Duarte Melo `[一作]` `[通讯]`, Victor Duarte Melo

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于路径积分思想的离散高维路径空间中精确噪声路径恢复的理论框架。

**💡 创新点**

创新点在于将路径积分的全局近似与单条隐藏轨迹的精确恢复区分开来，并给出了信息量与可逆性关系的定量分析。

**🔧 技术方法**

使用了离散路径建模、投影与量化观测、信息论计数、线性与非线性观测的形式化方法，并讨论了相关的密码学搜索假设。

**📊 数据集**

未使用具体数据集，而是构造了抽象的参数空间与随机生成模型。

**📈 对比分析**

本文未给出实验比较或性能评估，主要通过理论定理和结构攻击分析展示可能的难度。

**⚠️ 局限性**

主要限制在于缺乏针对具体观测系统的硬件实现与完整的安全证明，仍需进一步验证其实际难度。

---

## 499. A Symbolic Homotopy Algorithm for Solving Composable Polynomial Systems

**arXiv ID:** 2605.22514 | [PDF](https://arxiv.org/pdf/2605.22514v1)

**作者:** Thi Xuan Vu `[一作]` `[通讯]` (University of Lille), Thi Xuan Vu (University of Lille)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在特征为零的域上，计算n个多项式方程组的孤立正则解的问题，特别关注可组合结构的系统。

**💡 创新点**

通过利用可组合结构，显著提高了符号解算法的效率，提出了一种概率算法，计算所有孤立正则解，复杂度在输入大小和解的数量上是多项式的。

**🔧 技术方法**

使用了几何分解和全局牛顿-亨塞尔（同伦）提升程序的组合，形成了一种概率符号算法。

**📊 数据集**

没有具体提到使用的数据集，但提到的应用包括对称群、超立方体群和有限反射群的多项式系统。

**📈 对比分析**

与直接求解组合系统相比，提出的方法在复杂度上依赖于内外映射的度数和直线程序的大小，显著降低了计算成本。

**⚠️ 局限性**

限制在于算法主要针对孤立正则解，未来的工作将扩展到处理正维度组件，并开发稀疏或多重同质变体以更好地利用单项式结构。

---

## 500. Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning

**arXiv ID:** 2605.22511 | [PDF](https://arxiv.org/pdf/2605.22511v1)

**作者:** Zihan Liang `[一作]`, Lingtao Mao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演化搜索增强推理框架 Search‑E1，交替使用 GRPO 与离线自蒸馏（OFSD），无需外部教师或额外模块；

**💡 创新点**

创新点在于将同一问题的相邻轨迹对比转化为基于前馈 KL 的逐词学习信号，实现从自身轨迹中获得稠密步骤级监督；

**🔧 技术方法**

采用标准 GRPO 作为主体优化器，OFSD 通过前馈 KL 与点截断实现教师-学生对齐，并通过 LoRA 适配器实现模型更新；

**📊 数据集**

使用 Natural Questions 与 HotpotQA 训练集，评估覆盖七个单跳和多跳 QA 基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle）；

**📈 对比分析**

在 Qwen2.5‑3B‑Instruct 下平均 EM 达 0.440，超过所有开源基线（单跳提升 0.6–1.1 点，多跳提升 2.2–12.0 点），尤其在多跳任务上显著优于过程监督与外部教师方法；

**⚠️ 局限性**

局限在于仅试验了两轮 GRPO+OFSD 循环，未验证更长自演化周期的收益；对 Bamboogle 的性能仍落后，可能因样本量小及桥实体查询的重复状态导致。

---

## 501. The Neural Compiler: Program-to-Network Translation for Hybrid Scientific Machine Learning

**arXiv ID:** 2605.22498 | [PDF](https://arxiv.org/pdf/2605.22498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 502. Reflecti-Mate: A Conversational Agent for Adaptive Decision-Making Support Through System 1 and System 2 Thinking

**arXiv ID:** 2605.22509 | [PDF](https://arxiv.org/pdf/2605.22509v1)

**作者:** Morita Tarvirdians `[一作]` (Delft University of Technology), Catharine Oertel `[通讯]` (Delft University of Technology)

**通讯引用:** 991 | [OpenAlex ID](https://openalex.org/A5013731783)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估一种基于用户思考模式的对话式反思助手（Reflecti‑Mate），用于促进高风险个人决策的整合式预决策反思。

**💡 创新点**

创新点在于：①构建实时反思模式模型（关注类别和深度），②利用探索-利用权衡动态调整问答策略，以扩展未被充分探讨的思维维度；并在保留用户主导思维风格的同时促进多模态（头脑、心灵、直觉）反思。

**🔧 技术方法**

技术包括：Phi‑4 14B 大语言模型（本地部署）、自定义用户模型（思维类别/深度计算）、ε‑贪婪探索策略、基于 LIWC‑22 的情感/认知/直觉语言分析。

**📊 数据集**

数据集为实验中收集的128名受试者的“无援助”与“辅助”反思文本，受试者自行选择高风险人生决策主题（如职业、搬迁、教育等）。

**📈 对比分析**

比较方法：对照实验（实验版 vs 基线版），通过MANOVA、Cohen's d 及聚类分析评估反思语言维度；主观评估采用5点Likert量表。实验版在情感与直觉维度上显著提升，认知维度保持稳定，且被感知为更具整体整合支持；基线版则显著偏向认知并产生同质化效果。

**⚠️ 局限性**

局限性：①探索性问题为脚本化，未完全个性化；②LIWC 仅反映语言表现，未能直接测量内在认知/情感状态；③仅在短期交互中评估，缺乏对长期决策结果的跟踪；④系统未提供具体决策建议，可能与部分用户期望不符。

---

## 503. Relay-Based Synchronization of Replicated Data Types in Opportunistic Networks

**arXiv ID:** 2605.22491 | [PDF](https://arxiv.org/pdf/2605.22491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 504. Fluid RIS (FRIS)-Assisted Index Modulation for 6G Wireless Communications

**arXiv ID:** 2605.22508 | [PDF](https://arxiv.org/pdf/2605.22508v1)

**作者:** Xusheng Zhu `[一作]` (University College London), Hyundong Shin `[通讯]` (Kyung Hee University)

**通讯引用:** 8503 | [OpenAlex ID](https://openalex.org/A5007557286)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究如何利用流体可重构智能表面（FRIS）实现空间索引调制，并提出基于响应可分离度的代码书设计方法。

**💡 创新点**

创新点在于将索引选择从表面布局多样性转向接收端响应可分离度，并引入驱动粒度调节以平衡多样性、耦合、训练开销与硬件可行性。

**🔧 技术方法**

采用FRIS模型、互耦仿真、响应可分离度度量、代码书优化、BER/吞吐量仿真以及响应-aware设计流程。

**📊 数据集**

使用基于毫米波/太赫兹信道模型的仿真数据（含互耦、硬件失真），并未使用公开实验数据集。

**📈 对比分析**

通过与传统RIS-IM、随机FRIS-IM和布局驱动FRIS-IM的BER曲线及净吞吐量对比，展示响应-aware代码书在相同码本大小下能显著降低误码率、提升净吞吐量。

**⚠️ 局限性**

局限在于需要精确的耦合校准、对硬件漂移敏感、对实时更新的需求以及多用户/多功能环境下的代码书冲突问题。

---

## 505. Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost

**arXiv ID:** 2605.22502 | [PDF](https://arxiv.org/pdf/2605.22502v1)

**作者:** Simon Dennis `[一作]` (University of Melbourne), Hao Guo `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将代理流程编译进模型权重（“地下代理”）来替代传统的外部编排架构，评估其在旅行预订、Zoom 支持和保险理赔三大流程中的表现；

**💡 创新点**

创新点在于系统性量化编译对质量、成本和灵活性的影响，并提供了完整的实验设计、基准与同模型对比，证明编译模型可在大幅降低成本的同时保持接近前沿模型的质量；

**🔧 技术方法**

技术上采用完整参数微调、流程图驱动的数据生成、vLLM 部署、DeepSpeed ZeRO‑3 并行训练以及多轮评估（LLM‑judge 和 GPT‑4.1 评审）；

**📊 数据集**

数据集为由 Claude Sonnet 4.5 合成的对话，覆盖旅行（14 节点）、Zoom 支持（14 节点）和保险理赔（55 节点）共计约 15k 以上对话；

**📈 对比分析**

对比方法包括：同模型编排 vs 编译、前沿 LangGraph 编排（约 70 倍参数）vs 编译、以及全流程在‑context 方案；性能方面，8B 编译模型达 87–98% 的 in‑context 前沿质量，且相较于 LangGraph 编排在自然度、连贯性、灵活性上往往更优；在成本上，编译模型每对话比 in‑context 低 128–462 倍、比 LangGraph 低 77–249 倍；

**⚠️ 局限性**

局限性包括：需对流程进行完整重编译才能适配变更（仍需 30–50 分钟训练周期），对高度动态或无结构任务适用性不明；此外，实验主要聚焦在预定义流程，未验证在更开放式对话场景中的表现。

---

## 506. Supervised Classification Heads as Semantic Prototypes: Unlocking Vision-Language Alignment via Weight Recycling

**arXiv ID:** 2605.22484 | [PDF](https://arxiv.org/pdf/2605.22484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 507. MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation

**arXiv ID:** 2605.22469 | [PDF](https://arxiv.org/pdf/2605.22469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 508. Understanding Multimodal Failure in Action-Chunking Behavioral Cloning

**arXiv ID:** 2605.22493 | [PDF](https://arxiv.org/pdf/2605.22493v1)

**作者:** Lorenzo Mazza `[一作]` (NCT/UCC Dresden), Stefanie Speidel `[通讯]` (NCT/UCC Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究多模态行为克隆的理论与实践，提出了对多模态保持的定量评估，并验证了不同正则化与生成器平滑度对多模态性能的影响

**💡 创新点**

将多模态定义为条件动作分布的有限分离模式，证明了CVAE的后验-先验正则化会压缩动作条件信息，从而导致模式崩塌；同时证明了行动空间生成器的Lipschitz常数限制了可覆盖模式的数量

**🔧 技术方法**

使用变分自编码器(CVAE)、聚合匹配(MMD/Sinkhorn)、向量量化VQ-VAE、动作空间流匹配（Act-Flow）和扩散模型（Act-Diff）等生成器；通过信息理论（Fano、互信息）和几何分析证明正则化与Lipschitz约束对多模态的影响

**📊 数据集**

在合成的二维多模态导航任务（K=2,4,8,16）和三大机器人模拟基准（Push‑T、UR3 BlockPush、Kitchen）上进行实验；使用专家演示数据生成动作-观察对

**📈 对比分析**

与单模态BCAT、点估计KL-CVAE、学习先验CVAE、聚合匹配CWAE、流匹配和扩散模型进行对比；结果显示：①多模态覆盖是成功的必要但不足条件；②强点估计KL会抑制模式信息，导致性能下降；③学习先验和聚合匹配能在保持多模态的同时提升成功率；④动作空间生成器在覆盖多模态时需要更大的Lipschitz或桥区域，速度较慢，但在某些任务表现最好

**⚠️ 局限性**

理论结果不依赖训练，未给出实际选择β或Lipschitz阈值的准则；训练过程可能无法达到理论中的多模态证书；过度正则化或过度平滑的生成器会导致模式崩塌；对任务模糊度、模式数量与条件熵的估计仍是开放问题

---

## 509. One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation

**arXiv ID:** 2605.22544 | [PDF](https://arxiv.org/pdf/2605.22544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 510. Generative Modeling by Value-Driven Transport

**arXiv ID:** 2605.22507 | [PDF](https://arxiv.org/pdf/2605.22507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 511. Case-Aware Medical Image Classification with Multimodal Knowledge Graphs and Reliability-Guided Refinement

**arXiv ID:** 2605.22547 | [PDF](https://arxiv.org/pdf/2605.22547v1)

**作者:** Yiming Xu `[一作]` (University of Science and Technology of China), Qi Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2223 | [OpenAlex ID](https://openalex.org/A5108088410)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于多模态知识图谱的病例感知医学图像诊断框架 MKG-CARE，能够检索相似病例构建层次化图谱，进行知识传播与注入，并通过置信度校准的决策细化实现可解释的分类结果。

**💡 创新点**

创新点包括：①构建包含图像、症状、疾病节点的层次化多模态知识图谱；②设计图像中心的图注意网络与双向跨模态注意机制，实现知识与视觉特征的深度融合；③引入置信度校准的决策细化模块，根据相似度与置信度动态加权检索案例，提升鲁棒性与可解释性。

**🔧 技术方法**

采用 MedMamba 作为视觉编码器，FAISS 进行相似图像检索，图注意网络（GAT）进行知识传播，双向 CrossModalBlock 进行跨模态对齐，多头注意与 MLP 进行特征融合与分类，并通过第二次前向推理计算置信度权重，最后用 AdamW 进行优化。

**📊 数据集**

实验使用五个医学图像分类数据集：BreastMNIST、DermaMNIST、Kvasir、PAD_UFES_20、RetinaMNIST。

**📈 对比分析**

与多类视觉模型（Swin-Transformer、EfficientNetV2、ConvNeXt、VMamba、MedMamba）、医学/通用大型视觉语言模型（HealthGPT、HuatuoGPT、MedGemma、GPT‑5.1、Gemini、Qwen3.5‑plus）以及知识增强方法 KEM 进行对比。MKG-CARE 在所有五个数据集上均实现了最优或接近最优的整体准确率（OA），宏观 AUC 也优于或与强基线持平，验证了病例感知与知识融合的有效性。

**⚠️ 局限性**

主要局限包括：①检索质量直接影响最终性能，检索不准会导致错误的案例引入；②目前仅处理二维图像，未扩展至三维 CT/MRI 等更复杂场景；③知识图谱构建依赖专家手工标注，规模与覆盖面有限；④虽然模型相对轻量，但仍需要显存支持，难以直接迁移至大规模部署。

---

## 512. EnCoR: An end-to-end architecture for simplifying cellular networks

**arXiv ID:** 2605.22524 | [PDF](https://arxiv.org/pdf/2605.22524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 513. TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks

**arXiv ID:** 2605.22535 | [PDF](https://arxiv.org/pdf/2605.22535v1)

**作者:** Zhaoyang Chu `[一作]` (University College London), He Ye `[通讯]` (University College London)

**通讯引用:** 3772 | [OpenAlex ID](https://openalex.org/A5060546219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套可扩展的数据引擎，利用公开的终端录音自动反演真实工作流，生成高保真评测任务；

**💡 创新点**

创新点在于：①以“录音→任务”逆向方式实现评测数据的自动化生成，消除人工策划的规模瓶颈；②结合LLM、Docker环境重构和试验驱动的测试套件迭代，保证任务可执行与判定的可靠性；③引入Verified子集与跨基准比较，揭示专家手工基准与真实工作流的差距；

**🔧 技术方法**

主要技术包括：LLM（如Claude Sonnet、Claude Code）用于意图提炼与脚本抽取；Docker镜像构建与执行反馈循环；试验驱动的测试生成与校正；评测时使用Terminal-Bench的Harbor框架；

**📊 数据集**

数据集来源于80,870条公开终端录音，经过过滤后生成9,492条高质量录音，最终合成1,530条验证任务，其中200条为人工验证的Verified子集；

**📈 对比分析**

与已公开的Terminal-Bench 2.0、八款前沿LLM及六款终端代理进行对照实验，最高通过率为62.5%，平均通过率约为54.8%；与Terminal-Bench的相关系数仅为0.20，显示评测目标差异；

**⚠️ 局限性**

局限性包括：仅覆盖纯CLI工作流，忽略TUI/GUI；过度依赖LLM生成与环境推理，可能引入幻觉；生成的测试套件需要多轮迭代，成本较高；以及在极端依赖网络/专有软件的场景中难以重现。

---

## 514. A Subjective Logic-based method for runtime confidence updates in safety arguments

**arXiv ID:** 2605.22530 | [PDF](https://arxiv.org/pdf/2605.22530v1)

**作者:** Benjamin Herd `[一作]` (Fraunhofer Institute for Cognitive Systems), João-Vitor Zacchi `[通讯]` (Fraunhofer Institute for Cognitive Systems)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5011540519)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

本文提出一种将安全性能指标（SPI）融入主观逻辑（SL）安全论证框架的方法，实现运行时动态的安全信心更新。通过对SPI窗口内正负观测计数构造SL观点，并将累计融合（CBF）与否定挑战（RC）相结合，缺陷出现时即时惩罚信心，缺陷缺失时逐步提升信心。

**💡 创新点**

创新点在于：①引入非贝叶斯的SL更新规则，使SPI违规可立即反驳而不需要累积更多证据；②使用SL的第一阶与第二阶置信度分离，实时追踪对安全论证的具体影响；③将SPI监测与SL论证天然统一，形成端到端可审计的动态安全保证流程。

**🔧 技术方法**

使用技术包括：主观逻辑（SL）及其累计融合与否定挑战运算；SPI监测（基于阈值的误检率窗口统计）；YOLOv8 目标检测模型；CARLA 仿真环境与 APIKS ROS2 平台；贝塔分布映射与期望计算。

**📊 数据集**

数据集：仿真产生的高速公路施工现场数据（构筑物、道路工位、交通锥位置）。通过 CARLA 生成的场景与 Ground Truth，结合 YOLOv8 预测框，计算误检率以构造 SPI 观测。

**📈 对比分析**

对比方法：文章未给出与传统贝叶斯更新或静态安全论证的定量比较，而是通过多场景仿真展示信心随时间的演化。实验表明，SPI 违规时信心迅速下降，违规后持续监测可逐步恢复；且通过累积融合能降低不确定度。性能评价主要是可信度曲线与 Beta 分布收窄程度，未给出数值指标。

**⚠️ 局限性**

限制：①需要准确的 SPI 监测与阈值设定，易受环境噪声影响；②假设 SPI 窗口独立且无重叠，实际系统中可能存在时间相关性；③仅考虑前导式窗口式 SPI，缺乏对滞后式 SPI 的处理；④实时计算开销和硬件适配性未详述；⑤若 SPI 误报频繁，信心会被过度惩罚，需合适的阈值与折衷策略。

---

## 515. MOTOR: A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding

**arXiv ID:** 2605.22550 | [PDF](https://arxiv.org/pdf/2605.22550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 516. Polite on the Surface, Wrong in Practice: A Curated Dataset for Fixing Honorific Failures in Multilingual Bangla Generation

**arXiv ID:** 2605.22487 | [PDF](https://arxiv.org/pdf/2605.22487v1)

**作者:** Md. Asaduzzaman Shuvo `[一作]`, Md. Shafayet Hossain Ovi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供ACL会议论文提交的格式与风格文件使用说明。

**💡 创新点**

在已有的格式指导基础上，增加了对style文件细节的说明，方便作者快速使用。

**🔧 技术方法**

使用LaTeX模板、style文件以及参考官方格式指南的网址等技术。

**📊 数据集**

没有使用具体的数据集，主要面向作者。

**📈 对比分析**

通过与官方格式指南对照，说明其兼容性和完整性，满足会议提交要求。

**⚠️ 局限性**

仅适用于ACL格式，其他会议需要自行调整。

---

## 517. Matching with Deliberation: Test-Time Evolutionary Hierarchical Multi-Agents for Zero-Shot Compositional Image Retrieval

**arXiv ID:** 2605.22478 | [PDF](https://arxiv.org/pdf/2605.22478v1)

**作者:** Xingtian Pei `[一作]`, Shibiao Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种层次化的感知到推理框架（PDF），在零样本合成图像检索中通过多视角感知、意图路由和经验演化的推理提升检索精度。

**💡 创新点**

创新点包括：① 采用三层多代理协作（语义想象、约束解析、参考一致）实现多视角先验；② 通过意图感知路由动态分配先验权重，避免单一先验的感知偏差；③ 引入训练无关的推理策略蒸馏与锦标赛式测试时扩展（T-TTS），实现推理密度自适应提升。

**🔧 技术方法**

使用CLIP作为基准检索器，GPT‑4o‑mini作为推理引擎，构建视觉语义代理、意图路由管理器、决策演化管理器；实现经验自进化的推理策略蒸馏和基于锦标赛的测试时缩放。

**📊 数据集**

在三大基准数据集 CIRR、CIRCO、FashionIQ 上进行实验。

**📈 对比分析**

与多类基准（训练依赖、生成式无监督、现有多代理）对比，PDF 在 38 项评估指标中 35 项（92.1%）取得最优或接近最优结果，显著超越最新同类方法。

**⚠️ 局限性**

局限性在于最终推理仍受视觉语义代理质量限制，且候选池构造与检索器感知上限会影响性能；未来需改进多粒度表征与动态资源分配。

---

## 518. Winner-Take-All bottlenecks enforce disentangled symbolic representations in multi-task learning

**arXiv ID:** 2605.22472 | [PDF](https://arxiv.org/pdf/2605.22472v1)

**作者:** Julian Gutheil `[一作]` (Graz University of Technology), Robert Legenstein `[通讯]` (Graz University of Technology)

**通讯引用:** 5822 | [OpenAlex ID](https://openalex.org/A5045622873)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在多任务学习中使用赢家通吃（WTA）瓶颈网络如何从高度非线性混合的潜在因子中提取符号化的类别表示。

**💡 创新点**

首次给出了严格的理论证明，证明在满足条件的多任务设置下，WTA瓶颈会产生与真实类别对应的结构化置换的符号表示，并在多种实验中验证其有效性。

**🔧 技术方法**

采用多层感知机作为编码器，WTA头实现离散化，使用多任务线性分类器（后验概率预测）作为读出层，训练时使用 Straight‑Through Gumbel‑Softmax 估计梯度。

**📊 数据集**

在合成数据（5 个 5 类潜变量经过 MLP 生成的 70 维输入）和公开 dsprites 图像数据（形状、位置、颜色四个潜变量）上进行实验。

**📈 对比分析**

与仅使用混合表征的 MLP 对比，结果显示基于符号表征的模型在小样本下 AUC 几乎完美（>0.99），而基于混合表征的模型即使样本量增大也难以达到相同的泛化。

**⚠️ 局限性**

主要限制在于需要足够多且足够多样化的任务、假设潜变量映射是可注入的、WTA 结构需与潜变量类别匹配；当 WTA 输出维度过大或任务数量不足时，符号化效果可能衰退。

---

## 519. Why Are Agentic Pull Requests Merged or Rejected? An Empirical Study

**arXiv ID:** 2605.22534 | [PDF](https://arxiv.org/pdf/2605.22534v1)

**作者:** Sien Reeve O. Peralta `[一作]` (Waseda University), Youmei Fan `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5103134243)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了11,048个AI编码代理提交的Pull Request，结合人类审查交互细节，揭示仅凭合并/拒绝标签无法准确评估代理能力。

**💡 创新点**

提出决策导向分析方法，区分代理失效、工作流约束与无决策依据的拒绝，并量化合并时人类介入比例。

**🔧 技术方法**

采用手工定性编码、分层抽样、Cohen's κ 评估一致性等技术。

**📊 数据集**

基于AIDev数据集，筛选出9,799个人工审查的PR，并对717个案例进行手工检查。

**📈 对比分析**

与传统基于合并率的指标对比，发现约35.7%的拒绝并非代理失效，15.4%的合并需要人工反馈，说明传统指标存在显著偏差。

**⚠️ 局限性**

局限在于手工编码的主观性、未知类别导致信息缺失、仅覆盖高星公开仓库，难以推广到小型或私有项目。

---

## 520. Towards Direct Evaluation of Harness Optimizers via Priority Ranking

**arXiv ID:** 2605.22505 | [PDF](https://arxiv.org/pdf/2605.22505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 521. Stabilising Explainability Fragility in Cybersecurity AI: The Impact and Mitigation of Multicollinearity in Public Benchmark Datasets

**arXiv ID:** 2605.22529 | [PDF](https://arxiv.org/pdf/2605.22529v1)

**作者:** Ioannis J. Vourganas `[一作]` (Netrity Ltd), Anna Lito Michala `[通讯]` (University of Glasgow)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5025418975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了安全领域数据集中的多重共线性如何导致XAI解释的脆弱性，并提出理论证明。

**💡 创新点**

创新点包括正式定理、解释脆弱度得分、后处理CAA‑过滤和训练时SHARP正则化。

**🔧 技术方法**

技术手段为VIF诊断、相关系数聚类、SHAP/LIME解释、Kendall τ评估以及正则化损失。

**📊 数据集**

实验以UNSW‑NB15入侵检测基准数据集为主。

**📈 对比分析**

对比结果显示，特征剔除不显著影响预测精度，但显著提升解释稳定性；CAA‑过滤提高Kendall τ约0.2，SHARP在保持或提升准确率的同时使解释稳定性趋近1。

**⚠️ 局限性**

局限性包括需在训练时重新训练模型、计算成本高、仅在UNSW‑NB15上验证、方法适用于基于特征归因的XAI。

---

## 522. The Signal in the Noise: OOD Detection Through Goodness-of-Fit Testing in Factorised Latent Spaces

**arXiv ID:** 2605.22496 | [PDF](https://arxiv.org/pdf/2605.22496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 523. Quantum Genetic Optimization for Negative Selection Algorithms in Anomaly Detection

**arXiv ID:** 2605.22527 | [PDF](https://arxiv.org/pdf/2605.22527v1)

**作者:** Giancarlo P. Gamberi `[一作]` (Mackenzie Presbyterian University), Calebe P. Bianchini `[通讯]` (Mackenzie Presbyterian University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了将量子遗传算法整合进EvoSeedRNSA的量子遗传负选择算法（QGNSA），用于高维异常检测。

**💡 创新点**

通过利用量子叠加和Y旋转调节实现检测器生成的量子优化，替代传统遗传优化，显著提升检测器搜索效率和准确率。

**🔧 技术方法**

采用Qiskit量子模拟器实现量子遗传算法，结合经典的EvoSeedRNSA框架，使用量子门旋转、叠加态测量等量子技术。

**📊 数据集**

使用Kaggle的Metaverse金融交易数据集（约78,600条记录、12个特征）进行异常检测实验。

**📈 对比分析**

采用5折交叉验证、25次重复实验对比QGNSA与经典EvoSeedRNSA，量子版在召回率和错误负率上优于经典，但假阳性率升高；经典版在特异性和总体准确率上更优。

**⚠️ 局限性**

受限于量子硬件的量子比特数量和退相干，实验仅在模拟器上完成，且算法对阈值、精度等参数敏感，未验证在更大规模或不同领域的数据集上的鲁棒性。

---

## 524. EnCAgg: Enhanced Clustering Aggregation for Robust Federated Learning against Dynamic Model Poisoning

**arXiv ID:** 2605.22506 | [PDF](https://arxiv.org/pdf/2605.22506v1)

**作者:** Tianyun Zhang `[一作]` (Beijing University of Posts and Telecommunications), Yongfeng Huang `[通讯]` (Tsinghua University)

**通讯引用:** 13238 | [OpenAlex ID](https://openalex.org/A5100768896)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于低维梯度聚类与伪梯度生成的 EnCAgg 聚合框架，能够在动态模型中毒攻击下实现高鲁棒性与保真度。

**💡 创新点**

创新点包括：① 利用少量已知正常客户端梯度做参考，投影梯度至二维空间后进行基于密度的聚类；② 引入生成模型产生靠近良好聚类边界的伪梯度，拉拢稀疏良好梯度；③ 通过重新聚类进一步恢复被误判为噪声的良好梯度，突破传统固定阈值和固定聚类数的限制。

**🔧 技术方法**

主要技术手段为：PCA 降维、DBSCAN 密度聚类、三层全连接伪梯度生成网络、生成模型的多样性与方向损失、再聚类筛选以及最终的无权重平均聚合。

**📊 数据集**

实验使用 MNIST、CIFAR‑10（图像分类）和 MIND（新闻推荐）三大数据集。

**📈 对比分析**

与 FedSGD、Krum、Median、Trimmed‑Mean、FLTrust、DPFLA、FLGuardian 等基线比较。EnCAgg 在无攻击和多种攻击（UA‑FedRec、LIE、Min‑Max、Adaptive）下，准确率/ AUC/MRR 等指标均优于或持平于其他方法，尤其在 60% 甚至 80% 侵入比例时仍保持 80%+ 的性能。

**⚠️ 局限性**

局限性：需要事先获得少量可信客户端梯度；在极高攻击比例下性能仍略有下降；生成模型训练增加计算开销；对高度可变异或完全未知攻击的鲁棒性尚待进一步验证。

---

## 525. SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation

**arXiv ID:** 2605.22536 | [PDF](https://arxiv.org/pdf/2605.22536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 526. Multi-Winner Voting Games in TU and NTU: When is the Core Always Non-Empty?

**arXiv ID:** 2605.22528 | [PDF](https://arxiv.org/pdf/2605.22528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 527. FashionLens: Toward Versatile Fashion Image Retrieval via Task-Adaptive Learning

**arXiv ID:** 2605.22552 | [PDF](https://arxiv.org/pdf/2605.22552v1)

**作者:** Haokun Wen `[一作]` (Harbin Institute of Technology), Weili Guan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2400 | [OpenAlex ID](https://openalex.org/A5075938343)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FashionLens统一框架，解决多模态、多任务时尚图像检索问题。

**💡 创新点**

创新点包括Proposal-Guided Spherical Query Calibrator（PGSQC）实现查询向量在球面上自适应旋转，以及Gradient-Guided Adaptive Sampling（GGAS）利用梯度归一化动态调整任务采样比例。

**🔧 技术方法**

技术手段包括多模态大型语言模型(Qwen3‑VL‑4B)作为编码器、低秩适配器、球面线性插值（Slerp）、梯度EMA评估任务难度、LoRA微调和可学习检索标记。

**📊 数据集**

使用U‑FIRE基准，整合15个时尚检索数据集（覆盖9个任务）并新增2个OOD任务，样本总量超40万。

**📈 对比分析**

与GUMR、Fashion‑VLP以及Fine‑tuned模型进行对比，FashionLens在所有训练任务的平均mR达到50.01，显著优于基线，并在未见任务上也保持强泛化。

**⚠️ 局限性**

局限性包括对极端稀缺任务的处理仍有限、对长序列视频检索支持不足，以及在跨域迁移场景下的进一步验证需求。

---

## 528. F-TIS: Harnessing Diverse Models in Collaborative GRPO

**arXiv ID:** 2605.22537 | [PDF](https://arxiv.org/pdf/2605.22537v1)

**作者:** Nikolay Blagoev `[一作]` (Gensyn), Lydia Yiyu Chen `[通讯]` (University of Neuchatel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种F-TIS方法，在异构分布式环境中使用GRPO训练LLM，支持不同规模、专长和可训练参数的模型共同学习。

**💡 创新点**

提出将截断重要性采样与过滤结合的F-TIS，既降低通信开销，又保持稳定的收敛，支持异构模型协同训练。

**🔧 技术方法**

技术包括Group Relative Policy Optimization (GRPO)、Truncated Importance Sampling (TIS)、过滤策略、垂直/水平分布式生成、LoRA参数高效微调。

**📊 数据集**

使用GSM8k数学题集进行训练和验证，并在MATH-500离散任务上评估模型的泛化能力。

**📈 对比分析**

与纯单模型、NoIS、VIS、TIS、F-NoIS、F-VIS、水平协作等方法对比，F-TIS在异构设置下收敛速度与单模型相当，且在离散任务上可提升12%性能。

**⚠️ 局限性**

局限包括对g阈值的经验选择、较慢初期收敛、水平协作效果不佳、仅在特定任务和模型尺寸下验证、对极端异构规模与参数的扩展未探究。

---

## 529. Summarizing Time-Varying Digital Image Correlation Strain Fields Using Sankey Diagrams

**arXiv ID:** 2605.22627 | [PDF](https://arxiv.org/pdf/2605.22627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 530. Disentanglement Beyond Generative Models with Riemannian ICA

**arXiv ID:** 2605.22531 | [PDF](https://arxiv.org/pdf/2605.22531v1)

**作者:** Edmond Cunningham `[一作]` (University of Massachusetts Amherst), Edmond Cunningham `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5073716546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于黎曼几何的局部解耦方法RICA，通过在数据空间上构造Riemannian度量与概率密度，利用正则化的坐标变换实现对数据局部变异因素的分离；

**💡 创新点**

核心创新是提出“解耦张量”𝒟 = ∇²logρ - ⅓Ricci，将点对点解耦转化为对该张量的对角化，从而无需全局可逆生成模型；

**🔧 技术方法**

使用Riemannian几何（指数/对数映射、正交坐标、Christoffel符号、Ricci曲率）与概率密度的二阶导数，求解广义特征值问题来得到本地解耦方向；

**📊 数据集**

实验采用在已知源、度量和密度的合成数据集（多种流形和坐标系下的点云），并在控制源恢复任务中验证；

**📈 对比分析**

与传统ICA（线性和非线性）基线比较。RICA能够在不同坐标系下均成功恢复真源，而ICA的成功率随坐标系变化显著；

**⚠️ 局限性**

主要限制包括：需要对数据密度闭式或可近似；在预训练编码器上计算解耦张量需第三阶自动微分，计算成本高；目前缺乏可扩展的实用实现。

---

## 531. More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts

**arXiv ID:** 2605.22641 | [PDF](https://arxiv.org/pdf/2605.22641v1)

**作者:** Víctor Yeste `[一作]` (Universitat Politècnica de València), Paolo Rosso `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 15559 | [OpenAlex ID](https://openalex.org/A5053947754)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对政治文本中Schwartz价值检测进行了系统实验，比较句子/窗口/全文上下文、检索增强、不同模型规模与融合策略的影响；

**💡 创新点**

创新点在于揭示上下文、检索知识、模型族、规模和融合方式的交互效应，指出信息增量的条件性与非统一性；

**🔧 技术方法**

采用监督式DeBERTa‑v3基/大编码器、零-shot指令调优LLM（Gemma‑12B、Qwen‑72B、Mistral‑123B）、检索增强（早期融合、后期融合、跨注意）以及FAISS向量检索；

**📊 数据集**

使用ValuesML/Touché24‑ValueEval数据集，包含19标签精细化Schwartz价值标签的政治文本句子；

**📈 对比分析**

通过宏F1比较句子、窗口、全文三种上下文以及检索与否的多因素实验，发现监督编码器在全文上下文与早期融合下表现最佳，宏F1最高约0.314；零-shotLLM表现较低；模型规模并非决定性，检索增强可提升约0.02‑0.04 F1；

**⚠️ 局限性**

局限包括仅限英语政治文本、检索KB手工构建且固定、零-shotLLM未尝试少样本/微调、模型族与规模比较不完全公平、稀有标签性能低、架构 Ablation 未覆盖全部可能的融合方案。

---

## 532. A Multi-Source Framework for Relational Validation of Large Language Models Using Expert-Curated Encyclopedic Sources

**arXiv ID:** 2605.22636 | [PDF](https://arxiv.org/pdf/2605.22636v1)

**作者:** Moses Boudourides `[一作]` (Northwestern University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个三层多源框架，用专家编纂的百科全书知识图谱验证大型语言模型的关系性知识。

**💡 创新点**

首次将专家百科全书视为金标准，对LLM生成的知识图谱进行图层级（图、节点、边）关系性匹配，揭示了LLM在关系重现上的系统缺陷。

**🔧 技术方法**

采用图结构相似度、节点中心度相关性、边恢复精度召回F1等网络科学指标，以及GPT‑4的提示式知识抽取。

**📊 数据集**

十个专业百科全书（包括哲学、政治学、文化研究、精神分析、古罗马食谱等），每本都提供概念节点与专家标注的关系边。

**📈 对比分析**

与参考图谱的结构相似度、关系覆盖率、社区一致性等多层度量对比，结果显示SSS指数平均仅为0.06，节点中心度相关系数约0.15，边恢复F1平均仅为0.04，表明LLM在关系重现方面表现极差。

**⚠️ 局限性**

局限性在于仅检验显式提及的关系，忽略语义推理与隐式关联；数据集覆盖度不均导致领域差异大；框架不评估关系生成的合理性与一致性。

---

## 533. Do Deep Ensembles Actually Capture Uncertainty in Graph Neural Networks?

**arXiv ID:** 2605.22593 | [PDF](https://arxiv.org/pdf/2605.22593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 534. Cell Phantom Video Generation in Elliptical Fourier Descriptor Domain

**arXiv ID:** 2605.22563 | [PDF](https://arxiv.org/pdf/2605.22563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 535. Benchmarking Machine Learning Architectures for Antimicrobial Stewardship in Pediatric ICUs

**arXiv ID:** 2605.22611 | [PDF](https://arxiv.org/pdf/2605.22611v1)

**作者:** Niklas Raehse `[一作]` (University of Zurich), Daphné Chopard `[通讯]` (University of Zurich)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5021195814)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过系统评估多种机器学习模型，在两个儿童重症监护病房数据集上预测抗菌药物使用的干预机会。

**💡 创新点**

首次将四种临床相关代理目标（IV→口服、降谱、停药、短程疗法）引入儿童ICU，并在统一框架下对表格、序列和图模型进行对比。

**🔧 技术方法**

采用表格模型（LightGBM、MLP、TabPFN）、序列模型（GRU、LSTM、TCN、Transformer、SSM）以及图模型（RAINDROP），并结合1小时嵌入和多任务学习。

**📊 数据集**

使用公开的浙江大学儿童重症监护数据库（PIC）和私有医院PICU数据，共计约8,460次住院记录。

**📈 对比分析**

在四个目标上，表格模型和序列模型AUROC相近，但序列模型在AUPRC和F1上略有提升；细粒度1小时特征带来小幅提升，校准方面表格模型优于序列/图模型。

**⚠️ 局限性**

主要局限包括目标为历史实践代理、极端类别不平衡导致罕见事件预测低精度、固定时间窗口可能掩盖重要时序模式，以及缺乏前瞻性临床验证。

---

## 536. Decoupling Ego-Motion from Target Dynamics via Dual-Interval Motion Cues for UAV Detection

**arXiv ID:** 2605.22605 | [PDF](https://arxiv.org/pdf/2605.22605v1)

**作者:** Liuyang Wang `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**通讯引用:** 662 | [OpenAlex ID](https://openalex.org/A5059772938)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种双间隔运动引导检测框架，用以在UAV视频中解决严重视角变动和小目标检测难题。

**💡 创新点**

创新点在于通过全局运动补偿与双间隔差分融合获取稳健运动掩码，并将其作为轻量化注意力引入FPN，实现运动信息与特征的动态融合。

**🔧 技术方法**

采用的技术包括基于SIFT/ORB的全局运动补偿、双间隔差分、运动引导注意力模块、YOLOv8n网络以及TensorRT加速。

**📊 数据集**

实验使用VisDrone-VID数据集进行评估。

**📈 对比分析**

与YOLOv8n、YOLOv8s、YOLO11n及RT-DETR-L等基线对比，mAP@0.5提升至27.4%，mAP@0.5:0.95提升至12.1%，同时保持与基线相同的参数量和GFLOPs，在Jetson Orin Nano上实现约72 FPS。

**⚠️ 局限性**

主要局限是对非平面背景的鲁棒性不足，以及在极端光照或高速运动下仍可能出现误检；且训练阶段仍需高精度SIFT匹配，推理阶段虽已加速但仍有额外计算开销。

---

## 537. Branch-Stochastic Model Predictive Control for Motion Planning under Multi-Modal Uncertainty with Scenario Clustering

**arXiv ID:** 2605.22600 | [PDF](https://arxiv.org/pdf/2605.22600v1)

**作者:** Zekun Xing `[一作]` (Technical University of Munich), Martin Buss `[通讯]` (Technical University of Munich)

**通讯引用:** 15426 | [OpenAlex ID](https://openalex.org/A5081223790)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了结合SMPC与BMPC的B-SMPC框架，用于多模不确定性下的自动驾驶运动规划。

**💡 创新点**

创新点在于基于高层操纵规划的情景聚类、使用DTW自适应分支时机以及显式轨迹不确定性的机会约束。

**🔧 技术方法**

使用SMPC、BMPC、IAIMM‑KF预测、多模概率预测、DBSCAN聚类、DTW距离、GMM与椭圆约束、CasADi/Ipopt求解。

**📊 数据集**

在CommonRoad高速公路追尾/超车基准场景中评估。

**📈 对比分析**

与NMPC、SMPC以及B‑SMPC的消除聚类、无DTW等消融版对比，B‑SMPC实现0%碰撞率、约118成本、平均计算时间≈46 ms，显著优于基线。

**⚠️ 局限性**

局限性包括对阈值与聚类参数的手工调优、对非关键车辆的近似处理，以及在车辆数量极多时仍可能出现计算瓶颈。

---

## 538. Why SGD is not Brownian Motion: A New Perspective on Stochastic Dynamics

**arXiv ID:** 2605.22644 | [PDF](https://arxiv.org/pdf/2605.22644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 539. An intensive vRAN deployment with OpenAirInterface

**arXiv ID:** 2605.22638 | [PDF](https://arxiv.org/pdf/2605.22638v1)

**作者:** Romain Beurdouche `[一作]` (EURECOM), Raymond Knopp `[通讯]` (EURECOM)

**通讯引用:** 6283 | [OpenAlex ID](https://openalex.org/A5006213949)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在三种工业级vRAN计算架构上实现并验证OpenAirInterface（OAI）vRAN的高密度部署，改造OAI高物理层（High‑PHY）以充分利用硬件加速器（HA）和SIMD指令，优化资源分配、线程调度与接口重构，实现多实例共享单服务器的低时延、高吞吐量服务。

**💡 创新点**

①将OAI LDPC编码/解码模块与DPDK BBDev AAL对接，实现对多种HA（T2 Telco、ACC 100、vRAN Boost）的兼容；②重新设计LDPC接口，将整个slot的CB批量处理一次，显著降低调用开销；③针对precoding和资源映射引入多线程和SIMD加速；④在不同CPU/HA架构下系统级性能和能效评测，为工业化部署提供基准。

**🔧 技术方法**

OpenAirInterface 5G软核、DPDK BBDev、AAL、SR‑IOV、VFIO、AVX512/AVX512‑FP16、Neon、Docker容器化、Linux实时内核、O‑RAN 7.2 fronthaul、OAI的phy‑test模式、以及对不同硬件平台（AMD EPYC 9575F、AMD EPYC 8534P + T2 Telco、Intel Xeon Gold 6433N vRAN Boost）的驱动与配置。

**📊 数据集**

本研究使用OAI自带的synthetic UE流量（phy‑test模式）及真实RU连接产生的5G FR1 100 MHz 4×4小区流量，未使用公开数据集，主要通过模拟UE报文和实验室网络链路进行性能测试。

**📈 对比分析**

对三套架构分别在1、3、5、7实例（HPP/EP‑RFSoC）和1、3实例（vRANP）进行吞吐量、延迟和能耗对比。结果显示：HPP在UL/DL吞吐量上领先，能耗约107 W/实例；EP‑RFSoC与vRANP在能耗方面更优（≈45–90 W/实例），但在多实例共享HA时出现解码/编码时延峰值，影响实时性；总体DL吞吐量可达1.2 Gbps，UL 90 Mbps。

**⚠️ 局限性**

①资源分配需要为每个实例预留固定CPU核心，难以实现真正共享式vRAN；②HA共享导致的时延峰值，尤其在EP‑RFSoC上，影响低时延服务；③实验仅覆盖FR1、numerology 1，未测试FR2、numerology 3或大规模MIMO；④缺乏对GPU/ARM‑64架构的支持；⑤缺乏完整的RAN‑RIC调度与AAL调度方案，难以保证多实例实时性。

---

## 540. Whole-Blood Boundary Analysis of BioFET-Based ctDNA Detection for Intravascular Sensing in Intrabody Nanonetworks

**arXiv ID:** 2605.22637 | [PDF](https://arxiv.org/pdf/2605.22637v1)

**作者:** Ida Kleger-Rudomin `[一作]` (Gdansk University of Technology), Ethungshan Shitiri `[通讯]` (Universitat Politecnica de Catalunya)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

建立简化的随机模拟模型评估 BioFET 在全血中检测循环肿瘤 DNA（ctDNA）的可行性，并分析了离子屏蔽、界面距离、非特异性吸附及低频噪声对检测性能的影响。

**💡 创新点**

首次将 Debye 屏蔽、界面几何、有限容量占位以及噪声等因素整合进端到端的 stochastic 模型，以阈值检测方式系统评估全血环境下 BioFET 的检测边界。

**🔧 技术方法**

使用 Debye 屏蔽公式、有限容量占位模型、泊松绑定、加噪声的电流平移计算，以及 95% 上分位阈值与多传感器 OR 融合的判决技术。

**📊 数据集**

采用生理参数（离子强度、Debye 长度、血液暴露体积等）和文献给出的实验参数（氧化层厚度、探针密度等）作为输入；未使用公开实验数据集。

**📈 对比分析**

通过在不同 Debye 长度、氧化层厚度、界面层厚度和目标浓度下进行 Monte Carlo 仿真，比较灵敏度与特异度；结果显示 1 fM 时灵敏度可接近 100%，但在 10 aM 或 100 aM 下仅 10–15%，特异度保持 90–93%。屏蔽与界面距离是主要限制因素。

**⚠️ 局限性**

模型简化，忽略时间依赖的传输与绑定动力学、漂移噪声、制造波动和真实 ctDNA/背景片段分布；仅评估边界行为，绝对数值不具备预测性；未通过实验验证。

---

## 541. Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents

**arXiv ID:** 2605.22608 | [PDF](https://arxiv.org/pdf/2605.22608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 542. The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution

**arXiv ID:** 2605.22635 | [PDF](https://arxiv.org/pdf/2605.22635v1)

**作者:** Erjian Zhang `[一作]` (Xinjiang University), Zhiqing Guo `[通讯]` (Xinjiang University)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5044056760)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究多任务医学影像报告生成中的梯度冲突，提出 CAME-Grad 优化器以解决“Double Dilemma”。

**💡 创新点**

创新点在于用 SDE 视角分析线性标量化失效，设计冲突规避方向校正、能量增强注入和适应融合三阶段梯度优化。

**🔧 技术方法**

采用梯度冲突分析、SDE 模型、冲突规避方向投影、幅度增强注入以及自适应梯度融合等技术。

**📊 数据集**

使用公开胸部 X 光影像报告数据集 MIMIC-CXR 与 IU X‑Ray。

**📈 对比分析**

与八种基线模型及七种多任务优化器对比，在 MIMIC-CXR 平均提升 2.3% 临床效能（CE），在 IU X‑Ray 平均提升 1.9%，显著优于传统方法。

**⚠️ 局限性**

局限包括依赖自动标签评估、超参数多、理论验证不足，实际临床部署需进一步安全验证。

---

## 543. UNAD+: An Explainable Hybrid Framework for Unknown Network Attack Detection

**arXiv ID:** 2605.22621 | [PDF](https://arxiv.org/pdf/2605.22621v1)

**作者:** Saif Alzubi `[一作]` (University of Exeter), Frederic Stahl `[通讯]` (German Research Center for Artificial Intelligence GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 UNAD+ 框架，结合无监督集成、加权多数投票、伪标签监督细化和后置可解释层，以检测未知网络攻击并显著降低误报；

**💡 创新点**

1) 引入加权多数投票去除默认善意偏差；2) 使用伪标签进行监督细化提升精度；3) 在IDS工作流中嵌入局部 LIME 与全局决策树解释，提升透明度；

**🔧 技术方法**

无监督集成（LOF+Isolation Forest）、加权投票、伪标签监督（Random Forest）、SMOTE、PCA、特征选择；可解释技术 LIME 与 Decision Tree surrogate；

**📊 数据集**

CICIDS2017 与 NSL-KDD 两个公开基准数据集；

**📈 对比分析**

与原 UNAD、加权集成、单独监督进行对比；在 CICIDS2017 上 F1 98.31%，在 NSL-KDD 上 98.25%；误报率从 11.34% 降至 0.18%（CICIDS2017），从 6.37% 降至 2.77%（NSL-KDD），整体性能明显优于基线；

**⚠️ 局限性**

对极少见攻击类别仍难以提升；伪标签误差可能影响细化；框架未针对加密流量进行验证；

---

## 544. Innovations in Cardless Artificial Intelligence Banking: A Comprehensive Framework for Cyber Secure and Fraud Mitigation using Machine Learning Algorithms

**arXiv ID:** 2605.22604 | [PDF](https://arxiv.org/pdf/2605.22604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 545. Factored Diffusion Policies:Compositionally Generalized Robot Control with a Single Score Network

**arXiv ID:** 2605.22596 | [PDF](https://arxiv.org/pdf/2605.22596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 546. MoSA: Motion-constrained Stress Adaptation for Mitigating Real-to-Sim Gap in Continuum Dynamics via Learning Residual Anisotropy

**arXiv ID:** 2605.22597 | [PDF](https://arxiv.org/pdf/2605.22597v1)

**作者:** Jiaxu Wang `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在近似各向同性基础上学习残差应力调节模块，MoSA能够补偿实际材料的轻微各向异性与异质性，实现更精确的实景到仿真动力学建模。

**💡 创新点**

创新点在于：①将物理先验（各向同性本构关系）与可学习的残差应力适配器分离，仅对残差进行学习；②采用微面重分配与层层递进的残差校正，实现对四阶刚度张量的可控近似；③利用动态三维重建提供的运动信息，对位移场的时空导数进行高阶约束，显著提升数据效率与泛化。

**🔧 技术方法**

主要技术包括：基于MPM的连续介质模拟；结构化残差应力适配器（含预/后线性校正和微面重分配矩阵）；动态3D Gaussian Splatting的两阶段重建；流场与尺度/旋转约束的高阶监督；以及全局材料参数与连续隐式空间场的耦合建模。

**📊 数据集**

实验使用了PAC-NeRF的合成多视角视频数据集以及自采集的七个弹性/塑性对象的实景多摄像头视频数据集。

**📈 对比分析**

与PAC-NeRF、DEL、GIC、NeuMA、Vid2Sim等基线对比，MoSA在合成数据上Chamfer Distance与Earth Mover Distance均优于其他方法；在实景数据上PSNR/SSIM均明显提升，特别是在不同初始姿态和下落高度的泛化实验中，表现出更高的渲染质量与长时序稳定性。

**⚠️ 局限性**

局限性包括：①对重建质量高度依赖，若动态3D重建误差过大可能影响残差学习；②模型仍基于各向同性先验，极端各向异性或大尺度异质性场景下可能不足；③需要较多计算资源进行多视角重建与MPM仿真；④在极端光照或遮挡情况下运动约束可能失效。

---

## 547. SCALE: Sensitivity-Aware Federated Unlearning with Information Freshness Optimization for Mobile Edge Computing

**arXiv ID:** 2605.22589 | [PDF](https://arxiv.org/pdf/2605.22589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 548. A Formal Basis for Quantum Cryptographic Exposure Measurement under HNDL Threat

**arXiv ID:** 2605.22569 | [PDF](https://arxiv.org/pdf/2605.22569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 549. SynAE: A Framework for Measuring the Quality of Synthetic Data for Tool-Calling Agent Evaluations

**arXiv ID:** 2605.22564 | [PDF](https://arxiv.org/pdf/2605.22564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 550. Symmetries Here and There, Combined Everywhere: Cross-space Symmetry Compositions in Robotics

**arXiv ID:** 2605.22639 | [PDF](https://arxiv.org/pdf/2605.22639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 551. Chinese sensorimotor and embodiment norms for 3,000 lexicalized concepts

**arXiv ID:** 2605.22616 | [PDF](https://arxiv.org/pdf/2605.22616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 552. Regret-Based $(ε,δ)$-optimal Stopping Criteria for Bayesian Optimization

**arXiv ID:** 2605.22561 | [PDF](https://arxiv.org/pdf/2605.22561v1)

**作者:** Haowei Wang `[一作]` (National University of Singapore), Qiyu Wei `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于即时回报上界的贝叶斯优化停止准则，能够在满足(ε,δ)-最优性保证的前提下提前终止搜索；

**💡 创新点**

创新点在于给出了GP-UCB的紧致即时回报上界，并通过优化子问题自适应选择置信区间参数，从而实现更严格的停止判定；

**🔧 技术方法**

主要技术包括高斯过程建模、UCB采集函数、概率置信区间分析、离散化与参数化置信区间、以及非线性优化求解；

**📊 数据集**

使用了经典的合成基准函数（Branin、Rosenbrock、Levy）、GP采样目标函数以及真实的CNN超参数调优任务（MNIST）进行实验；

**📈 对比分析**

与PRB、ΔCB、ΔES、Acq等现有停止方法以及不做停止的NOSTOP和Oracle_r 进行对比；在大多数任务上，新准则在保持高成功率（≥95%）的同时显著减少评估次数，且停止时的最终简单回报往往优于或与Oracle_r相近；

**⚠️ 局限性**

局限性包括：对β_t的取值仍采用传统形式，未进一步探索其对探索-利用平衡的影响；算法仍需在高维复杂问题或非光滑目标上进一步验证；此外，当前实现仅针对GP-UCB，尚未扩展到其他采集函数或多目标设置。

---

## 553. Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning

**arXiv ID:** 2605.22642 | [PDF](https://arxiv.org/pdf/2605.22642v1)

**作者:** Banghao Chi `[一作]` (University of Illinois Urbana Champaign), Hanchao Yu `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Spreadsheet‑RL框架，利用基于奖励的强化学习对大语言模型进行后期微调，训练出专门的电子表格代理，能够在真实的Excel环境中完成多步骤工作流。

**💡 创新点**

创新点包括：① 通过自动化数据代理从在线论坛收集大规模初始–最终电子表格对；② 设计Spreadsheet Gym与表格特定工具引导的交互 harness；③ 在此环境下采用GRPO实现稀疏、可验证的结果奖励的上游策略优化。

**🔧 技术方法**

使用技术包括：强化学习（GRPO）、多步对话与工具调用（ReAct‑style）、Microsoft Excel 365 真实运行时、Python 代码沙盒、KL 散度正则化以及异步奖励计算接口。

**📊 数据集**

数据集：1）从 ExcelForum 自动收集的 5,928 任务（含 18,855 讨论线程）用于训练；2）公开基准 SpreadsheetBench（912 任务）和新构建的 Domain‑Spreadsheet（1,660 任务，涵盖金融、供应链、人力资源、销售与房地产）用于评估。

**📈 对比分析**

通过与基线（无工具、仅工具、无 RL）及商业代理（ChatGPT‑Agent、Copilot）比较，结果显示：在 SpreadsheetBench 上，Qwen3‑4B‑Thinking‑2507 的 Pass@1 由 12.0% 提升至 23.4%；在 Domain‑Spreadsheet 上从 8.4% 提升至 17.2%，在金融子域提升尤为显著。

**⚠️ 局限性**

局限性包括：仅在 4B 参数规模模型上实验，性能仍落后于大型专有模型；对部分复杂领域（如房地产）仍表现不佳；训练需要依赖真实 Excel 环境和高昂的计算资源；且数据集主要来自论坛讨论，可能缺少更广泛的专业任务多样性。

---

## 554. Rethinking Noise-Robust Training for Frozen Vision Foundation Models: A Cross-Dataset Benchmark with a Case Study of Small-Loss Failure

**arXiv ID:** 2605.22591 | [PDF](https://arxiv.org/pdf/2605.22591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 555. Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework

**arXiv ID:** 2605.22620 | [PDF](https://arxiv.org/pdf/2605.22620v1)

**作者:** Shourov Joarder `[一作]` (Bangladesh University of Engineering and Technology), Prashnna Gyawali `[通讯]` (West Virginia University)

**通讯引用:** 781 | [OpenAlex ID](https://openalex.org/A5008803380)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多奖励内部反馈强化学习框架（RLIF），通过组合答案级聚类投票奖励和完成级自信奖励来改进大型语言模型的推理能力。

**💡 创新点**

创新点在于将两种互补的内部奖励（聚类投票和自信度）结合，并利用GDPO归一化和KL‑Cov正则化来防止奖励破解、熵坍塌及模型崩溃。

**🔧 技术方法**

技术实现包括：GRPO风格的策略优化、每个奖励通道的组内z‑score归一化、KL‑Cov针对高协方差token的局部KL惩罚，以及自回归token生成。

**📊 数据集**

使用数据集为未标注的MATH训练集进行无监督训练，评估指标覆盖七个基准：GSM8K、MATH500、MMLU‑Pro、AIME 2024/25、LiveCodeBench v6 和 CRUXEval‑O。

**📈 对比分析**

与单奖励RLIF（如Intuitor）以及监督RLVR（GRPO‑GT）比较，本文方法在大多数基准上达到了接近监督方法的性能，在1.5B和3B模型上分别超过Intuitor约30–50%并与GRPO‑GT相差不到7个百分点。

**⚠️ 局限性**

局限性包括：需额外调优的超参数（奖励权重、KL‑Cov阈值），以及对预训练模型推理能力的依赖，低基础模型的表现仍有限。

---

## 556. Healthcare LLM Benchmarks Are Only as Good as Their Explicit Assumptions

**arXiv ID:** 2605.22612 | [PDF](https://arxiv.org/pdf/2605.22612v1)

**作者:** Naveen Raman `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2917 | [OpenAlex ID](https://openalex.org/A5079207566)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了医疗行业中LLM评估与实际部署之间的差距，并提出通过显式记录评估假设（Task与Outcome两类）来缩小这一差距。提出BenchmarkCards记录假设，并给出分阶段评估流程，在实际RCT案例中验证方法。

**💡 创新点**

创新点在于：①将评估假设分为Task（仅需对话数据可检验）与Outcome（需真实结果和行为研究）两类；②引入BenchmarkCards这一结构化文档体系，让评估者透明记录假设；③设计分阶段评估流程，先验证Task假设，再针对Outcome假设进行行为实验或RCT，系统化地闭合评估-部署差距。

**🔧 技术方法**

主要技术手段包括：假设分类与假设检验框架、敏感性分析分解评估-部署差距、BenchmarkCards的设计与实现（基于现有Model Card、EvalCard等模板）、分阶段评估流程的制定。

**📊 数据集**

使用了一个真实的医疗RCT案例（2026年5月22日的研究）来演示方法，并引用了若干先前的医疗LLM评估与部署实验数据；此外，还基于公开的BenchmarkCards和GitHub项目（<https://github.com/naveenr414/benchmarkcards>）进行实验。

**📈 对比分析**

比较方法：先在基准上评估模型性能，再在真实部署场景中测量性能，利用敏感性分析将差距拆分为Task与Outcome两部分；通过该方法发现Task假设约占31个百分点，Outcome假设约占30个百分点。性能上，单轮基准表现95%，但真实部署仅达34%，显示显著差距。

**⚠️ 局限性**

局限性：①敏感性分析仅在单一案例中验证，结果不一定通用；②BenchmarkCards尚未得到广泛验证其实际影响；③假设分类为二元，现实中假设可能处于连续谱；④Outcome假设的验证往往需要昂贵的RCT或长期追踪，实施成本高。

---

## 557. Building an Open Source Operational Technology Pentesting Platform: Lessons from LINICS

**arXiv ID:** 2605.22590 | [PDF](https://arxiv.org/pdf/2605.22590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 558. Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion

**arXiv ID:** 2605.22579 | [PDF](https://arxiv.org/pdf/2605.22579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 559. Contractual Skills: A GovernSpec Design Framework for Enterprise AI Agents

**arXiv ID:** 2605.22634 | [PDF](https://arxiv.org/pdf/2605.22634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 560. A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models

**arXiv ID:** 2605.22586 | [PDF](https://arxiv.org/pdf/2605.22586v1)

**作者:** Jiayi Fu `[一作]` (INSAIT), Yuxia Wang `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以微分方程的视角系统阐述扩散模型的理论与实现，梳理了前向后向过程的ODE/SDE表述、训练目标、采样方法以及条件引导；

**💡 创新点**

创新点在于统一了DDPM、DDIM、SDE、ODE等多种实现，揭示它们在逆向动力学下的等价性，并将传统的噪声预测损失与流匹配、分数匹配两种视角关联起来；

**🔧 技术方法**

使用的技术包括高斯前向过程、Fokker–Planck连续性方程、随机微分方程、逆向SDE/ODE、变分推理、噪声预测、分类器/无分类器引导、以及数值求解器（Euler、Heun、RK45、DPM‑Solver）等；

**📊 数据集**

论文为教程性质，并未在实验中使用具体数据集；

**📈 对比分析**

没有实验对比，本文侧重理论分析与公式推导，未给出具体性能指标；

**⚠️ 局限性**

局限性在于缺乏实证评估，对不同超参数、模型架构和数据集的影响未做系统验证，仅为理论框架与方法的统一与说明。

---

## 561. AtomicMotion: Learning Human Motion From Different Human Parts

**arXiv ID:** 2605.22631 | [PDF](https://arxiv.org/pdf/2605.22631v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 562. SE3Kit: A Lightweight Python Library for Specialized Geometric Primitives in Robotics

**arXiv ID:** 2605.22633 | [PDF](https://arxiv.org/pdf/2605.22633v1)

**作者:** Daniyal Maroufi `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5055294307)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款轻量化的Python库SE3Kit，用于SE(3)和SO(3)的Lie群运算。

**💡 创新点**

核心创新在于仅依赖NumPy，提供严格的数学实现、统一表示、指数/对数映射，并消除数值漂移。

**🔧 技术方法**

技术上采用纯Python+NumPy实现Lie代数与群运算、旋转矩阵/四元数/欧拉角统一处理、切空间映射。

**📊 数据集**

未使用公开数据集，主要通过基准测试与现有库进行比较。

**📈 对比分析**

与SciPy、SpatialMath、PyPose、tf2_py等比较，SE3Kit在内存占用（≈MB级）和依赖轻量化上占优，适合嵌入式/快速原型；在性能上能够满足CPU控制循环的实时需求。

**⚠️ 局限性**

局限性包括缺乏GPU加速、可微功能、可视化工具，且对大规模点云/视觉算法支持有限。

---

## 563. H-Flow: Self-supervised Human Scene Flow via Physics-inspired Joint Multi-modal Learning

**arXiv ID:** 2605.22629 | [PDF](https://arxiv.org/pdf/2605.22629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 564. GLeVE: Graph-Guided Lesion Grounding with Proposal Verification in 3D CT

**arXiv ID:** 2605.22619 | [PDF](https://arxiv.org/pdf/2605.22619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 565. A note on convergence of Wasserstein policy optimization

**arXiv ID:** 2605.22622 | [PDF](https://arxiv.org/pdf/2605.22622v1)

**作者:** David Šiška `[一作]` (University of Edinburgh), Yufei Zhang `[通讯]` (Imperial College London)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5100324331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 Wasserstein Policy Optimization (WPO) 在熵正则化马尔可夫决策过程中的线性收敛性证明，说明该算法在连续动作空间下可实现指数级速度收敛。

**💡 创新点**

通过将 WPO 视为 Wasserstein 梯度流，利用对数-索博列夫不等式与能量耗散分析，首次将连续时间梯度流理论应用于策略优化问题，实现了对传统随机梯度方法的理论推广。

**🔧 技术方法**

使用了 Wasserstein 梯度流、熵正则化动态规划、对数-索博列夫不等式、能量耗散分析以及平均场（mean‑field）方法来证明收敛性。

**📊 数据集**

该工作为理论分析性质，未涉及具体实验或数据集。

**📈 对比分析**

没有进行实验对比，论文仅给出理论上的线性收敛证明。

**⚠️ 局限性**

主要局限在于对梯度流解的存在与光滑性、对数-索博列夫不等式的局部满足等假设要求较高；对离散状态、连续动作空间的广泛适用性尚待验证，并且未给出数值实验支持。

---

## 566. Evolutionary Multi-Task Optimization for LLM-Guided Program Discovery

**arXiv ID:** 2605.22613 | [PDF](https://arxiv.org/pdf/2605.22613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 567. Missing Links in Public Email and Covert Networks: A Comparative Evaluation of Link Prediction, Hyperlink Prediction, and ERGM Estimation

**arXiv ID:** 2605.22606 | [PDF](https://arxiv.org/pdf/2605.22606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 568. Think Thrice Before You Speak: Dual knowledge-enhanced Theory-of-Mind Reasoning for Persuasive Agents

**arXiv ID:** 2605.22602 | [PDF](https://arxiv.org/pdf/2605.22602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 569. Measuring Security Without Fooling Ourselves: Why Benchmarking Agents Is Hard

**arXiv ID:** 2605.22568 | [PDF](https://arxiv.org/pdf/2605.22568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 570. LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance

**arXiv ID:** 2605.22567 | [PDF](https://arxiv.org/pdf/2605.22567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 571. GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving

**arXiv ID:** 2605.22566 | [PDF](https://arxiv.org/pdf/2605.22566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 572. GeoWeaver: Grounding Visual Tokens with Geometric Evidence before Scene Reasoning

**arXiv ID:** 2605.22558 | [PDF](https://arxiv.org/pdf/2605.22558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 573. Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety

**arXiv ID:** 2605.22643 | [PDF](https://arxiv.org/pdf/2605.22643v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 574. SceneAligner: 3D-Grounded Floorplan Localization in the Wild

**arXiv ID:** 2605.22581 | [PDF](https://arxiv.org/pdf/2605.22581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 575. Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following

**arXiv ID:** 2605.22607 | [PDF](https://arxiv.org/pdf/2605.22607v1)

**作者:** Shijing Wang `[一作]` (Beijing Jiaotong University), Hyung Jin Chang `[通讯]` (University of Birmingham)

**通讯引用:** 5622 | [OpenAlex ID](https://openalex.org/A5004895698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于视觉基础模型的注意力适配方法，结合头部条件局部LoRA和外锥惩罚，提升了视线追踪的推理能力。

**💡 创新点**

创新点在于①将LoRA改为头部条件局部适配，只在头部区域进行可学习更新；②设计外锥惩罚，用柔性锥形掩码约束中间特征，促使模型关注真实视线而非语义显著性。

**🔧 技术方法**

使用冻结的DINOv2视觉基础模型，加入头部条件局部LoRA（HCLoRA）和外锥惩罚（OOC）作为训练机制，并配合轻量化的Gaze decoder和二分类损失。

**📊 数据集**

在公开的GazeFollow和VideoAttentionTarget两个视线追踪基准数据集上进行训练与评估。

**📈 对比分析**

与GazeLLE、Sharingan等前沿方法在AUC、L2距离及角度误差等指标上对比，获得SOTA表现，尤其在语义不显著的“inconsistent”子集平均L2下降约10%、角度误差下降约15%。

**⚠️ 局限性**

仍存在对语义显著性的小幅依赖，且在多摄像头或动态视频场景下的泛化能力尚未得到充分验证。

---

## 576. Beyond Chamfer Distance: Granular Order-aware Evaluation Metric For Online Mapping

**arXiv ID:** 2605.22578 | [PDF](https://arxiv.org/pdf/2605.22578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 577. SegGuidedNet: Sub-Region-Aware Attention Supervision for Interpretable Brain Tumor Segmentation

**arXiv ID:** 2605.22572 | [PDF](https://arxiv.org/pdf/2605.22572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 578. VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis

**arXiv ID:** 2605.22570 | [PDF](https://arxiv.org/pdf/2605.22570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 579. Reading Task Failure Off the Activations: A Sparse-Feature Audit of GPT-2 Small on Indirect Object Identification

**arXiv ID:** 2605.22719 | [PDF](https://arxiv.org/pdf/2605.22719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 580. TriSweep: A Four-Drone Swarm Framework for Electromagnetic Side-Channel Analysis

**arXiv ID:** 2605.22709 | [PDF](https://arxiv.org/pdf/2605.22709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 581. Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models

**arXiv ID:** 2605.22732 | [PDF](https://arxiv.org/pdf/2605.22732v1)

**作者:** Juergen Dietrich `[一作]` `[通讯]` (Democracy Intelligence gGmbH), Juergen Dietrich (Democracy Intelligence gGmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了语音情绪识别模型是否能作为政治演讲中Pathos维度的代理，并与LLM多模态分析结果进行对比；同时系统评估了EMO‑DB语料库的结构与适用性。

**💡 创新点**

提出将acoustic SER与post‑hoc Russell Circumplex投影结合，用LLM（Gemini）进行全音频+文本情绪与语篇功能的开放式多模态分析，并将三者与TRUST‑Pathos分数进行关联评估，揭示EMO‑DB在自然语境下的局限。

**🔧 技术方法**

使用emotion2vec_plus_large（SER），Gemini 2.5 Flash（多模态LLM），TRUST Pipeline（LLM监督集合），post‑hoc Russell Circumplex投影，以及Spearman相关分析。

**📊 数据集**

使用德国联邦议会的Felix Banaszak的全程演讲（51段，约245 s）以及EMO‑DB的535段情绪化说话片段。

**📈 对比分析**

通过Spearman相关系数比较各模态与TRUST‑Pathos的关联：Gemini Valence与TRUST‑Pathos高度相关（ρ = 0.664，p < 0.001），Gemini Arousal与TRUST‑Pathos中等负相关（ρ = ‑0.535），而emotion2vec Valence与TRUST‑Pathos几乎无关联（ρ = 0.097）；跨模态一致性低。EMO‑DB评估显示Gemini对Disgust和Boredom的匹配率极低，整体匹配率仅30%。

**⚠️ 局限性**

局限包括：仅单一说话者样本量小，emotion2vec的post‑hoc投影假设未被验证，EMO‑DB结构缺陷导致评估失真，TRUST‑Pathos的操作化仅适用于德语政治语境，LLM对情绪与语篇功能共用模型可能产生内部一致性偏差。

---

## 582. Live Music Diffusion Models: Efficient Fine-Tuning and Post-Training of Interactive Diffusion Music Generators

**arXiv ID:** 2605.22717 | [PDF](https://arxiv.org/pdf/2605.22717v1)

**作者:** Zachary Novack `[一作]` (University of California San Diego), Cheng-Zhi Anna Huang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为 Live Music Diffusion Models（LMDMs）的实时音乐生成框架，能够在消费者级硬件上通过扩散模型实现低延迟的交互式音乐创作。

**💡 创新点**

核心创新点包括：① 通过路由与定制注意力掩码实现 KV 缓存，使扩散模型的推理复杂度恢复甚至优于离散自回归（AR）模型；② 引入 ARC‑Forcing 这一无需 RL 或奖励模型的全局对抗性后训练策略，显著减轻多块生成过程中的误差累积；③ 将文本、草图（CQT 等）与伴奏等多种控制方式统一到同一框架。

**🔧 技术方法**

使用技术包括：流匹配（Flow Matching）/扩散模型、DiT Transformer、KV 缓存、Self‑Forcing、Adversarial Relativistic Contrastive（ARC）后训练、Ping‑Pong 采样器、ONNX 导出及 C++/JUCE 实时推理。

**📊 数据集**

主要数据集：MTG‑Jamendo（草图控制与音乐生成）、MusicCaps（文本标签）、MusDB18（评估）、FSD50k（ Foley 任务）、以及用于对比的 MusicGen‑Large、Stable Audio、Magenta RealTime 等公开模型。

**📈 对比分析**

与 LMM、MusicGen‑Large、Stable Audio 等基准相比，LMDMs 在参数量、训练成本、推理延迟方面均显著优越：文本条件下参数约为 LMM 的一半，推理延迟降低至 30–170 ms；ARC‑Forcing 后可在 8 步内完成，生成质量（FD/KL、CLAP、CoCoLA 等指标）与离散 AR 模型相当甚至略优；在伴奏和草图控制任务中亦表现出较强的控制遵从性。

**⚠️ 局限性**

局限性：① 受训练数据偏向（如 EDM）影响，模型对非主流音乐风格的生成效果不佳；② 对文本条件的反应性仍弱，尤其在实时场景中易回退到通用 EDM 声音；③ 延迟仍高于子秒级，需更高效的音频编解码器和模型架构改进；④ 与大型专有模型（如 Suno）相比，整体音质仍有差距。

---

## 583. AMEL: Accumulated Message Effects on LLM Judgments

**arXiv ID:** 2605.22714 | [PDF](https://arxiv.org/pdf/2605.22714v1)

**作者:** Sid-ali Temkit `[一作]` `[通讯]`, Sid-ali Temkit

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究大语言模型在多轮评估中的累积消息效应（AMEL），测量对模型判断的偏移并分析其强度、方向与饱和点。

**💡 创新点**

首次系统量化并揭示AMEL的普遍性、负面偏好不对称、对模型规模与不确定性敏感性，并提供基于实验的缓解建议。

**🔧 技术方法**

通过大规模API调用构造多轮对话历史，使用偏差分数（bias score）衡量模型输出变化，结合t检验、Spearman相关、OLS回归以及对概率分布、位置效应的对照实验。

**📊 数据集**

使用包含21条评测项目（代码评审、内容审核、营养评价）共63条条目，构造不同长度（5/10/20/50）与不同极性（正/负/中立）的对话历史，采集公开API响应。

**📈 对比分析**

以偏差分数对比不同极性、长度、模型规模，计算Cohen d和p值；整体d≈-0.17，最强在不确定项d≈-0.34，负极性比正性大1.6倍；模型规模越大效应越小。

**⚠️ 局限性**

仅覆盖二分类评判、仅三领域、作者编码的模糊项与内部不确定性不完全一致、温度/语言多样性不足、未在所有模型上重复对照实验、对未评测历史可能影响不完全。

---

## 584. Tokenization with Split Trees

**arXiv ID:** 2605.22705 | [PDF](https://arxiv.org/pdf/2605.22705v1)

**作者:** Craig W. Schmidt `[一作]` (Kensho Technologies), Chris Tanner `[通讯]` (Kensho Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于分割树的子词分词方法 ToaST。

**💡 创新点**

创新点在于将分割树与整数规划相结合，以全局优化实现最小化词元数量。

**🔧 技术方法**

使用字节 n-gram 统计构建分割树，并通过整数规划及 LP 松弛求解词表。

**📊 数据集**

在 175GB 英语 CulturaX 语料上进行训练。

**📈 对比分析**

与 BPE、WordPiece、UnigramLM 对比，ToaST 词元压缩率提升 11%+，模型 CORE 分数提高 2.6–7.6%。

**⚠️ 局限性**

仅在英语上验证，跨语言效果尚未评估。

---

## 585. What Does the Caption Really Say? Counterfactual Phrase Intervention for Compositional Data Selection in Vision-Language Pretraining

**arXiv ID:** 2605.22651 | [PDF](https://arxiv.org/pdf/2605.22651v1)

**作者:** Hyejin Go `[一作]` (Soongsil University), Hyesong Choi `[通讯]` (Soongsil University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5004848551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了 CLIP 预训练中全局匹配可能缺乏对字幕中单词、属性和关系的具体支持，提出一种基于受控词组置换的词组级筛选框架 Counterfactual Phrase Intervention（CPI），从而挑选更具组合语义信息的正样本。

**💡 创新点**

创新点在于把句子层匹配拆解为词组级的敏感性评估，使用三种不变置换协议在每个词组上计算相似度衰减，形成句子级平均 PAS 得分，从而在保留全局匹配的同时实现更细粒度的正样本筛选。

**🔧 技术方法**

采用的技术包括 CLIP ViT‑B/32 相似度计算、受控 nonce‑token 置换、Three‑Invariance Replacement Protocol、PAS 的平均聚合、两阶段粗细筛选流程。

**📊 数据集**

主要使用 CC3M（约 220 万图文对）做预训练实验，并在 CC3M 上评估 Zero‑shot、检索、线性探针、SugarCrepe、VL‑CheckList‑VG 等基准；对比实验还把 CPI 子集应用于 NegCLIP、CE‑CLIP。

**📈 对比分析**

通过与全数据、随机 50% 数据、单纯 top50 全局过滤等基线对比，CPI 在保持整体迁移性能的同时显著提升 SugarCrepe、VL‑CheckList‑VG Relation 等词组敏感度指标（如 Relation +1.91、SC 整体 +0.71），且在 NegCLIP、CE‑CLIP 中进一步提升组合语义指标。

**⚠️ 局限性**

局限包括 PAS 仅是相对的一阶敏感性度量，不能直接解释可视化定位或因果关系；其效果受 CLIP 评分器的依赖，可能不适用于其他 VLM；实验仅在 CC3M 规模下验证，未测试更大规模或不同架构；置换规则和词组抽取基于规则，可能存在残留噪声。

---

## 586. AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters

**arXiv ID:** 2605.22645 | [PDF](https://arxiv.org/pdf/2605.22645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 587. The Value of Covariance Matching in Gaussian DDPMs and the Lanczos Sampler

**arXiv ID:** 2605.22723 | [PDF](https://arxiv.org/pdf/2605.22723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 588. Improving Viewpoint-Invariance and Temporal Consistency for Action Detection

**arXiv ID:** 2605.22695 | [PDF](https://arxiv.org/pdf/2605.22695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 589. HarnessAPI: A Skill-First Framework for Unified Streaming APIs and MCP Tools

**arXiv ID:** 2605.22733 | [PDF](https://arxiv.org/pdf/2605.22733v1)

**作者:** Edwin Jose `[一作]` (Western Michigan University), Edwin Jose `[通讯]` (Western Michigan University)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5010259422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

构建并发布了 HarnessAPI，一个 Python 框架，能够从单一 typed skill 文件夹自动生成 HTTP REST API（支持 SSE）和 MCP 工具注册，消除双重维护。

**💡 创新点**

采用 skill-first 逆向依赖，将业务逻辑与 schema 统一在 skill 目录中，动态生成路由、MCP 包装器和内容协商，保证 HTTP 与 MCP 的 schema 一致性并实现单进程部署。

**🔧 技术方法**

结合 FastAPI、FastMCP、Pydantic、动态代码生成、异步 ASGI、SSE、OpenAPI/Swagger UI，使用工具链实现模块隔离和热替换。

**📊 数据集**

在评估中使用了自定义的六个示例技能（Echo, Greet, Summarize, VectorNorm, Classify, Translate），以及 12 个 agentskills.io 规范的技能目录。

**📈 对比分析**

通过比较手动双栈实现与 HarnessAPI 的框架 boilerplate 行数，发现 Boilerplate 减少 74%；功能对比表显示所有 FastAPI/FastMCP 需手动集成的特性已被一站式支持，单进程部署降低运维成本；未测量端到端延迟，预期继承 FastAPI 的性能。

**⚠️ 局限性**

缺乏端到端性能基准，MCP 包装器依赖动态编译且对复杂 Pydantic 模型不完善；热替换仅限本地开发；MCP 认证与多租户支持不足；未来需要使用 FastMCP 的显式 schema API。

---

## 590. Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation

**arXiv ID:** 2605.22731 | [PDF](https://arxiv.org/pdf/2605.22731v1)

**作者:** Dong Nie `[一作]` `[通讯]` (Independent Researcher), Dong Nie (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨并实验了后训练（SFT、RL、OPD）在自回归语言模型中对状态分布的影响，并通过Qwen3-0.6B模型在GSM8K、TruthfulQA、MMLU任务上的小规模实验验证了状态源对模型表现与遗忘的决定作用。

**💡 创新点**

提出将后训练视为状态分布塑造的统一框架，区分训练状态来源与监督信号来源，解释了为何OPD能超越退化教师、RL能保持能力以及SFT在不同压力下产生遗忘。

**🔧 技术方法**

采用LoRA适配器对Qwen3-0.6B进行微调，实施SFT（轻度与强度两种）、基于教师连续体的OPD、以及轻量化的对策性RL（GRPO），并用MMD等指标评估状态漂移。

**📊 数据集**

目标任务使用GSM8K，遗忘评估使用TruthfulQA和MMLU子集，教师数据来源于SFT训练的模型。

**📈 对比分析**

对比不同后训练方法在目标任务准确率、遗忘量（基于TruthfulQA/MMLU）和MMD漂移等指标，结果显示轻度SFT提升GSM8K且无遗忘，强度SFT导致显著遗忘，OPD可从退化教师中恢复并超越，RL轻量化亦能提升目标并保持遗忘极低，MMD漂移并未完全预测遗忘。

**⚠️ 局限性**

实验规模有限，仅使用单一模型和少数数据集，RL与OPD实现简单且漂移测度基于词法特征，缺乏大规模验证与深层特征分析，结果仅为机制性证据。

---

## 591. Slimmable ConvNeXt: Width-Adaptive Inference for Efficient Multi-Device Deployment

**arXiv ID:** 2605.22677 | [PDF](https://arxiv.org/pdf/2605.22677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 592. Can AI Make Conflicts Worse? An Alignment Failure in LLM Deployment Across Conflict Contexts

**arXiv ID:** 2605.22720 | [PDF](https://arxiv.org/pdf/2605.22720v1)

**作者:** Andrii Kryshtal `[一作]` `[通讯]` (BlueDot Impact Technical AI Safety Sprint), Andrii Kryshtal (BlueDot Impact Technical AI Safety Sprint)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并量化大型语言模型在冲突环境中的对齐性能，提出冲突敏感度评估框架。

**💡 创新点**

首次将冲突敏感度作为对齐属性纳入模型评估，并发现压迫性平衡请求导致模型普遍失效。

**🔧 技术方法**

使用基于Anthropic Bloom的多轮交互式评估流程，包含场景生成、模型推理、评判。

**📊 数据集**

构造90个多轮冲突场景（包括东DRC、乌克兰、缅甸等），并对五个维度进行变体。

**📈 对比分析**

对9种模型配置（OpenAI GPT-5.4-mini、Claude Sonnet 4、DeepSeek-V3.2、xAI Grok等）进行比较，失败率从6%到47%不等，压力框架下最高可达100%。

**⚠️ 局限性**

局限包括评判者为AI模型、仅英文场景、样本量有限、缺乏因果机制验证、非实时对话长度有限。

---

## 593. Maximum-Weight Two Boxes Symmetric Difference Problem

**arXiv ID:** 2605.22690 | [PDF](https://arxiv.org/pdf/2605.22690v1)

**作者:** José Fernández Goycoolea `[一作]` (Universidad de Magallanes), Carlos Seara `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 1767 | [OpenAlex ID](https://openalex.org/A5062044790)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于泛化 MCS-tree 的算法，在平面点集上寻找两条可能重叠的轴对齐矩形，使其对称差内点的权重和最大化。

**💡 创新点**

创新点在于将二维对称差优化问题拆解为一维连续子序列最大化问题，利用泛化 MCS-tree 处理多种相对位置（共 18 种情况），并给出了可扩展到 k 个矩形或并集的框架。

**🔧 技术方法**

核心技术包括：泛化 MCS-tree（最大连续子序列树）的构造与更新、四条水平线与九个 0‑1 区域激活矩阵的组合、O(n) 线性扫描与 O(log n) 更新的滑动窗口方法。

**📊 数据集**

论文未使用实验数据集，而是以理论分析为主，给出了算法时间空间复杂度与相关硬件假设。

**📈 对比分析**

与单箱最大加权问题 O(n²log n) 或 O(n²) 的比较显示，本文的 O(n⁴log n) 解决了更一般的两箱对称差问题；对 k>2 的情况复杂度随 k 指数增长，实测未给出。

**⚠️ 局限性**

主要限制包括：算法时间复杂度高（n⁴log n），常数因子大；对 k 作为输入时问题变为 NP‑hard；假设点集一般位置且权重可为负；未给出实验验证或实际应用案例。

---

## 594. Is Capability a Liability? More Capable Language Models Make Worse Forecasts When It Matters Most

**arXiv ID:** 2605.22672 | [PDF](https://arxiv.org/pdf/2605.22672v1)

**作者:** Nick Merrill `[一作]` (Forecasting Research Institute), Ezra Karger `[通讯]` (Forecasting Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在超线性增长且存在转型风险的时间序列预测任务中的逆向规模现象，系统验证其在模拟基准（FBSim）和多领域（SIR、COVID‑19、住房价格、通胀周期、美国麻疹）中的表现，并揭示分布式评分规则与阈值评分对结论的影响。

**💡 创新点**

首次提出并验证结构可辨识的逆向规模类：更大模型在分布式预测上会更不准确，且仅在使用覆盖整个分布的评分规则（CRPS、log‑score）时显现；同时揭示单阈值评分会误导模型能力评估。

**🔧 技术方法**

采用大语言模型（Llama‑3.1、OpenAI等）、分布式评分规则（CRPS、Pinball、Brier）、per‑quantile 分解、对照实验（2×2 scale/post‑training）、模拟基准 FBSim 以及合成 SIR 数据，系统性分析模型能力与预测质量的关系。

**📊 数据集**

数据集包括：1) FBSim 纯净模拟世界的多模板时间序列；2) 合成 SIR 流行病序列；3) 60 国 COVID‑19 日病例；4) 19 个美国城市的住房价格；5) 12 个超通胀周期的 CPI；6) 1928–1962 年美国麻疹状态季序列。

**📈 对比分析**

通过 Spearman ρ 评估模型能力（ECI）与不同评分规则（CRPS、Brier、Pinball）结果的相关性。结果显示：在 FBSim 的 H1 阶段 ρ≈+0.67，H7 阶段 ρ≈-0.42；SIR、COVID‑19、住房、通胀、麻疹等领域均出现类似的逆向规模。Brier 评分未显示负相关，证明阈值评分掩盖了问题。

**⚠️ 局限性**

局限性：①仅在已知存在转型风险的时间序列中验证，未探究其他结构的普适性；②样本量受历史记录限制，特别是超通胀和住房数据；③模型能力通过外部 ECI 指标评估，缺乏直接操纵；④未深入揭示内部机制导致的校准差距；⑤对实时部署预测的影响仍未知。

---

## 595. Imperfect Commitment in Maximal Extractable Value Auctions

**arXiv ID:** 2605.22667 | [PDF](https://arxiv.org/pdf/2605.22667v1)

**作者:** Aleksei Adadurov `[一作]` (nuconstruct), Arsenii Valitov `[通讯]` (nuconstruct)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了以Builder为主的Ethereum MEV 拍卖中承诺缺失的问题，并揭示了Builder可以在未履行拍卖结果时复制并获取 MEV 价值。

**💡 创新点**

创新点是将拍卖腐败文献与 MEV 复制渠道相结合，提出了类型特定的可复制份额 γ(τ) 并构建了分段贝叶斯纳什均衡，揭示了不同 MEV 类型下的承诺成本差异。

**🔧 技术方法**

采用了附属价值的首次价格封闭式拍卖模型，利用解析求解贝叶斯纳什均衡和边界条件，进一步通过右尾板块估计 γ̂(τ)。

**📊 数据集**

使用了 2024 年 9 月至 2025 年 8 月的 2.2 百万条 MEV 包交易数据集，包含交易哈希、块号、MEV 类型、Builder、tip 等信息。

**📈 对比分析**

通过估计 γ̂(τ) 并与贝格曼披露阈值对比，验证了不同 MEV 类型的破坏收益差异，整体回报差距可达 48.8%，表明承诺缺失对某些类型（如裸套利、清算）影响显著。

**⚠️ 局限性**

局限在于将 ε 视为外生参数，未建模 Builder 与搜索者的集成效应，也缺乏对稀疏类型（如清算）的精确估计和对不同 Builder 的个体差异分析。

---

## 596. Seeing the Poem: Image-Semantic Detection of AI-Generated Modern Chinese Poetry with MLLMs

**arXiv ID:** 2605.22654 | [PDF](https://arxiv.org/pdf/2605.22654v1)

**作者:** Shanshan Wang `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**通讯引用:** 3823 | [OpenAlex ID](https://openalex.org/A5101468579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估并提升大语言模型在现代中文诗歌检测中的表现，提出基于图像语义的 IMAGINE 检测框架。

**💡 创新点**

首次将与诗歌内容对应的图像作为辅助信息加入检测，通过图像语义引导显著提升检测准确率。

**🔧 技术方法**

利用多模态大语言模型（Gemini、Qwen 等）和示例驱动的提示，构建图像语义引导检测方法。

**📊 数据集**

构建 800 条人工诗歌与 3200 条 LLM 生成诗歌的检测数据集，并为 800 条人写诗歌配制对应图像。

**📈 对比分析**

与传统文本检测器（Fast-DetectGPT、RoBERTa 等）和 LLM 文本检测器对比，IMAGINE 在 Gemini 上实现宏观 F1 85.65%，超过所有基线。

**⚠️ 局限性**

方法主要适用于诗歌等富含形象化内容的文本，对新闻、论文等文本效果有限。

---

## 597. From Baseline to Follow-Up: Counterfactual Spine DXA Image Synthesis in UK Biobank Using a Causal Hierarchical Variational Autoencoder

**arXiv ID:** 2605.22649 | [PDF](https://arxiv.org/pdf/2605.22649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 598. The Secretary Problem with a Stochastic Precursor

**arXiv ID:** 2605.22653 | [PDF](https://arxiv.org/pdf/2605.22653v1)

**作者:** Franziska Eberle `[一作]` (Technische Universität Berlin), Alexander Lindermayr `[通讯]` (Technische Universität Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在经典的秘书问题中引入仅含时序信息的异步前导信号，分析其在随机顺序和对抗顺序下的最优停机策略，并证明该信号能提升成功概率。

**💡 创新点**

创新点在于提出并研究了仅靠信号到达时间而无内容的前导信号模型，展示时序信息本身就能显著突破经典 1/e 限界，并给出了完整的最优策略与稳健性分析；同时引入 α‑power 信号分布，统一处理不同早晚程度的信号，并研究多重信号历史对确定性策略的提升。

**🔧 技术方法**

主要技术包括：最优停机的贝尔曼递推与记录特性、Beta(α,1) 分布分析、渐进极限计算、随机与确定性策略的对比、对抗顺序下的整数线性规划（ILP）来刻画完整信号历史的最优解、以及仿真验证。

**📊 数据集**

使用人工合成数据：随机排列的排名序列和根据 α‑power 分布抽样的信号时间；实验规模从 n=1000 开始，并在不同 α、误差率和信号失真下进行多次重复。

**📈 对比分析**

与经典 (n/e) 规则和不使用信号的策略相比，实验显示即使单个均匀信号也可使成功概率提升至约 0.5，且随 α 增大可趋近 1；在对抗顺序中，随机策略可达到 n^α/∑j^α 的成功率，确定性策略为 1-(1-1/n)^α；多信号历史对确定性策略提升显著；整体表现与理论一致。

**⚠️ 局限性**

局限包括：仅考虑完美、始终提前最佳项的信号，未对噪声、延迟或误报的异步信号给出完整理论；实验依赖人工合成数据，缺乏真实世界验证；在更一般的秘书约束（如匹配、子模最优化）中的适用性尚未探究。

---

## 599. Abstraction for Offline Goal-Conditioned Reinforcement Learning

**arXiv ID:** 2605.22711 | [PDF](https://arxiv.org/pdf/2605.22711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 600. ChronoVAE-HOPE: Beyond Attention -- A Next-Generation VAE Foundation Model for Specialized Time Series Classification

**arXiv ID:** 2605.22684 | [PDF](https://arxiv.org/pdf/2605.22684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 601. Conceptualizing Embeddings: Sparse Disentanglement for Vision-Language Models

**arXiv ID:** 2605.22679 | [PDF](https://arxiv.org/pdf/2605.22679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 602. Self-Evolving Multi-Agent Systems via Decentralized Memory

**arXiv ID:** 2605.22721 | [PDF](https://arxiv.org/pdf/2605.22721v1)

**作者:** Guangya Hao `[一作]` (University of Cambridge), Zhuokai Zhao `[通讯]` (University of Chicago)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5101776767)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种去中心化的双池记忆框架，支持LLM多智能体系统在协作中持续自我进化。

**💡 创新点**

创新点包括：1）每个智能体拥有私有的利用池（E‑pool）和探索池（X‑pool）；2）利用LLM评判器在线重新加权池权重，自动平衡利用与探索；3) 在理论上证明实现全局可达性并达到O(log T)累计遗憾；4) 该框架与任意MAS结构兼容，且对不同LLM模型无关。

**🔧 技术方法**

采用的核心技术有：双池记忆结构、基于相似度检索与LLM生成相结合的在线路由、LLM‑as‑judge的阶段反馈与权重更新、图结构搜索与随机游走的混合策略，以及对抗式多臂赌博机分析。

**📊 数据集**

使用的评测数据集包括：数学推理（AIME25、AIME24）、代码生成（MBPP‑Plus）、问答（BBH）和具身决策（ALFWorld）。

**📈 对比分析**

与MetaGPT、ChatDev、G‑Memory以及无记忆基线进行对比。实验覆盖AutoGen、DyLAN、AgentNet三大MAS框架，并测试Qwen3（4B/8B/14B）与Gemma4（E2B/E4B）等多种LLM。结果显示：平均提升 8.6% 以上，相比最强基线可达 +23.8%，相对无记忆提升可达 +52.5%；同时在Token消耗上减少约 32%–49%。

**⚠️ 局限性**

局限性：实验仅覆盖四个任务域，缺乏对更高风险或专业领域（如法律、医学）的验证，未来工作需要在更广泛、多样化的高价值任务上进一步评估。

---

## 603. Swift Sampling: Selecting Temporal Surprises via Taylor Series

**arXiv ID:** 2605.22678 | [PDF](https://arxiv.org/pdf/2605.22678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 604. AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild

**arXiv ID:** 2605.22715 | [PDF](https://arxiv.org/pdf/2605.22715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 605. Posterior Collapse as Automatic Spectral Pruning

**arXiv ID:** 2605.22691 | [PDF](https://arxiv.org/pdf/2605.22691v1)

**作者:** Johannes Hirn `[一作]` (Universitat de València), Johannes Hirn `[通讯]` (Universitat de València)

**通讯引用:** 748 | [OpenAlex ID](https://openalex.org/A5025149839)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了β-VAE中后验崩塌现象，将其视为一种自动谱剪枝机制，分析了崩塌阈值与重构效用之间的关系，并在线性高斯VAE中与PCA谱建立对应；

**💡 创新点**

创新点在于把后验崩塌解释为损失函数的稳定性分岔，提出模式级别的崩塌阈值与效用谱概念，并证明在线性高斯模型下两者与PCA解释方差完全一致；

**🔧 技术方法**

使用了β-VAE的变分下界、Landau稳定性分析、PCA以及归一化的温度参数T来对比不同模式的崩塌和效用；

**📊 数据集**

实验数据集为WorldClim全球气候变量数据集；

**📈 对比分析**

通过将崩塌阈值、效用谱和PCA解释方差对齐，验证了三者在归一化单位下基本一致；实验显示β扫描能准确揭示有效模式数，与理论预期相符；

**⚠️ 局限性**

局限性在于结果仅在线性高斯VAE中解析可证，非线性解码器、非高斯似然或有限训练等情形下谱可能被旋转、混合或偏移，仍需进一步研究。

---

## 606. The efficiency-gain illusion: People underestimate the rate of AI use and overestimate its benefits on simple tasks

**arXiv ID:** 2605.22687 | [PDF](https://arxiv.org/pdf/2605.22687v1)

**作者:** Sunny Yu `[一作]` (Stanford University), Robert D. Hawkins `[通讯]` (Stanford University)

**通讯引用:** 31420 | [OpenAlex ID](https://openalex.org/A5041689299)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过三项预注册用户研究，检验人们在简单任务上使用大型语言模型（LLM）时的自我估计失准与效率误区，并探讨使用体验对后续AI采纳的影响。

**💡 创新点**

创新点在于首次系统量化人类对LLM效率的误判（自估低于实际使用率和速度提升幻觉），并揭示使用经验能加剧误判与AI过度依赖的反馈循环。

**🔧 技术方法**

采用了GPT‑4o作为LLM助手，结合NASA‑TLX自评量表、时间记录和多组对照设计。

**📊 数据集**

使用了基于TUNA分类的24道简单认知任务（信息寻求、加工、执行、创作）以及来自Prolific的美国成年样本共2691名参与者。

**📈 对比分析**

通过对比预测与实际使用率、时间与努力的差异，发现实际AI使用率比预测高约14%，但在时间和努力上几乎无节省，甚至在易任务上导致平均多耗时约10秒；相较传统人类代理辅助，AI未实现预期效率提升。

**⚠️ 局限性**

局限性包括任务仅限5分钟以内的简单任务、未考虑长期使用和动机激励、样本为线上众包人群、对AI使用方式缺乏细粒度测度，以及未检验跨文化差异。

---

## 607. N3P: Accelerated Automated Parking via a Learning-Based Naturalistic Three-Stage Scheme

**arXiv ID:** 2605.22722 | [PDF](https://arxiv.org/pdf/2605.22722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 608. WorldKV: Efficient World Memory with World Retrieval and Compression

**arXiv ID:** 2605.22718 | [PDF](https://arxiv.org/pdf/2605.22718v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 609. Beyond the Org Chart: AI and the Transformation of Invisible Work

**arXiv ID:** 2605.22707 | [PDF](https://arxiv.org/pdf/2605.22707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 610. From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model

**arXiv ID:** 2605.22671 | [PDF](https://arxiv.org/pdf/2605.22671v1)

**作者:** Bing Hu `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29944 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出BehaviorVLA框架，通过学习时序一致的行为表示，提升Vision‑Language‑Action（VLA）模型在分布迁移下的鲁棒性，核心组件为Visuomotor Behavior Encoder（VBE）和Phase‑conditioned Behavior Decoder（PBD）。

**💡 创新点**

创新点包括：① VBE采用因果三流（视觉、动作、行为）架构与Mamba时序建模，能够在长时序上捕获任务的全局结构；② PBD采用Predictor‑Corrector设计，利用Phase‑Attention实时对齐结构先验，解决动作生成的时间漂移问题；③ 通过将行为表示分解为时间不变的全局原型与时间变相位状态，实现从具体示范到通用抽象再到精确实例化的端到端流程。

**🔧 技术方法**

技术方法：Mamba（Selective State Space Model）实现线性复杂度的长时序建模；交叉注意力实现视觉与动作的跨模态融合；进化的Contrastive Loss与InfoNCE用于原型聚类和相位辨识；Conditional Flow Matching结合可学习的先验向量场完成动作细化；此外采用Progress‑Attention、Predictor‑Corrector、Guidance Strength调节等机制。

**📊 数据集**

数据集：模拟环境：RoboTwin 2.0、LIBERO、CALVIN；真实世界：GALAXEA R1 Lite（14‑DoF双臂机器人），涵盖Generalization和Long‑horizon两类任务。

**📈 对比分析**

与多种基线（RDT、π_0.5、OpenVLA‑OFT等）对比，BehaviorVLA在RoboTwin 2.0 Hard设置下平均成功率58%（比RDT提升37.7%），LIBERO平均成功率98%（跨所有子套件领先），CALVIN Avg.Len 4.36；在真实世界任务上平均成功率提升63%，且在sim‑to‑real迁移中仅使用50%示范数据即可匹配OpenVLA‑OFT，显著提升数据效率与泛化性能。

**⚠️ 局限性**

局限性：① 相位估计误差仍可能导致时间漂移，尤其在极端动态或高速操作中；② 对新任务的迁移依赖于丰富的行为原型库，原型缺失会影响抽象能力；③ 模型包含多模块与Flow Matching，推理时延较大，对实时性能有一定影响。

---

## 611. Multiple Neural Operators Achieve Near-Optimal Rates for Multi-Task Learning

**arXiv ID:** 2605.22724 | [PDF](https://arxiv.org/pdf/2605.22724v1)

**作者:** Adrien Weihs `[一作]` (University of California Los Angeles), Hayden Schaeffer `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文研究了多任务/多操作符学习中的近似与统计复杂性，并针对多神经算子（MNO）提出了近似与泛化的最优扩展；

**💡 创新点**

创新点在于通过改进构造和错误分析，消除了先前研究中出现的多任务特有指数复杂度，证明多任务学习的复杂度与单任务学习保持一致，并给出了匹配的下界；

**🔧 技术方法**

使用了深度 ReLU 网络的分层构造、分解与聚合技术、覆盖数与泛化误差分析、以及无限维立方体的低维嵌入与下界证明；

**📊 数据集**

由于论文为理论分析，未使用具体数据集；

**📈 对比分析**

通过与基于拼接的 DeepONet 进行理论比较，发现两种架构在最优近似和统计学习率上具有相同的渐近复杂度，但在实际实验中 MNO 通常表现更优；

**⚠️ 局限性**

局限性包括：仅针对 Lipschitz/可微多操作符类给出最优界限，缺乏对更光滑或 PDE 特定结构的分析；下界与上界在指数幂上仍不紧凑；未给出经验验证或特定任务的实验。

---

## 612. SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers

**arXiv ID:** 2605.22668 | [PDF](https://arxiv.org/pdf/2605.22668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 613. Parametric Modular Answer Set Programs Made Declarative

**arXiv ID:** 2605.22716 | [PDF](https://arxiv.org/pdf/2605.22716v1)

**作者:** Jorge Fandinno `[一作]` (University of Nebraska Omaha), Torsten Schaub `[通讯]` (University of Potsdam)

**通讯引用:** 8564 | [OpenAlex ID](https://openalex.org/A5058467603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出参数化模块化答案集程序（Parametric Modular Answer Set Programs）的形式化框架，并为其提供了宣言性语义，阐述了如何通过模块化逻辑程序和参数化子程序来实现ASP的collective控制。

**💡 创新点**

创新点在于：①引入参数化模块化逻辑程序，能够以参数化方式定义子程序；②使用简单强度语句细化谓词的强度/扩展性，保证模块之间的无冲突性；③通过构造合并子程序的集合作为整体程序，给出模块化ASP与传统ASP等价性的理论基础，实现在声明性层面对控制的解释。

**🔧 技术方法**

主要技术包括：第一阶逻辑和答案集语义；这里-那里逻辑（Quantified Logic of Here-and-There）及其稳定模型定义；模块化逻辑程序的定义与简单强度语句；参数化强度语句与子程序实例化；依赖图与连通分量的概念；以及归纳证明方式验证答案集唯一性。

**📊 数据集**

无实验数据集；论文侧重理论分析与形式化证明。

**📈 对比分析**

无实验比较或性能评估；工作本质为理论探讨，未给出系统实现或基准测试。

**⚠️ 局限性**

局限性：仅考虑collective控制，未覆盖包含外部原子（external atom）或更复杂控制机制的情况；框架在实际ASP系统中的实现细节与效率尚未验证；进一步研究需要扩展到更多控制形式与实际应用场景。

---

## 614. Cross-Domain Human Action Recognition from Multiview Motion and Textual Descriptions

**arXiv ID:** 2605.22697 | [PDF](https://arxiv.org/pdf/2605.22697v1)

**作者:** Yannick Porto `[一作]` (Université Bourgogne Europe), Cedric Demonceaux `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种利用多视角训练与视角感知编码、并结合文本提示实现零样本动作识别的方法。

**💡 创新点**

创新点包括：①双分支注意力的视角感知运动编码网络；②在训练阶段通过虚拟相机投影生成多视角数据；③将动作文本提示与视角信息融合，并使用对比学习与分类联合优化；④在推理时仅需单视角即可保持高性能。

**🔧 技术方法**

主要技术：ProtoGCN 时空骨架编码器、双分支多头交叉注意力、姿态角度位置编码、CLIP 文本编码器、GPT‑3.5 生成视角增强文本、对比损失与交叉熵联合训练。

**📊 数据集**

使用的数据集：BABEL、NTU‑RGB+D、NW‑UCLA、RHM‑HAR、MCAD 以及用于预训练的 Posetics。

**📈 对比分析**

与多种零样本/跨域基线（如 CADA‑VAE、JPoSE、ReViSE、DeViSE、PURLS、Neuron、ViA、MotionClip 等）对比，本文在 NTU‑60/NW‑UCLA 的零样本任务上分别提升 3.46% 与 14.62%，跨域 top‑1 均达到 28.48/69.09，平均 49.93，明显优于现有最优方法。

**⚠️ 局限性**

局限性：推理时需要先估计人体姿态角度，增加计算负担；方法依赖预训练的多视角投影和文本提示，对数据与视角的覆盖范围有一定要求。

---

## 615. Scout-Assisted Planning for Heterogeneous Robot Teams under Partially Known Environments

**arXiv ID:** 2605.22693 | [PDF](https://arxiv.org/pdf/2605.22693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 616. Forecasting Scientific Progress with Artificial Intelligence

**arXiv ID:** 2605.22681 | [PDF](https://arxiv.org/pdf/2605.22681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 617. Claw AI Lab: An Autonomous Multi-Agent Research Team

**arXiv ID:** 2605.22662 | [PDF](https://arxiv.org/pdf/2605.22662v1)

**作者:** Fan Wu `[一作]`, Fayao Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用离线 Q 学习对公共教育数据进行建模，目标是学习针对学生的干预决策策略。

**💡 创新点**

提出了泄漏安全的状态构造、支持感知贝尔曼回归以及在支持不足时的回退策略，形成了 SPARQ 框架。

**🔧 技术方法**

核心技术包括离线强化学习（Q‑学习）、贝尔曼回归、状态特征工程与支持检测机制。

**📊 数据集**

实验基于公开的学生学习记录数据集（例如美国教育统计数据或类似的公共教育数据集）。

**📈 对比分析**

与传统的预测模型和基线决策策略进行对比，SPARQ 在单次模拟实验中实现了与基线相当或略优的学生绩效提升，但缺乏多次实验验证与统计显著性检验。

**⚠️ 局限性**

主要局限在于仅执行一次实验、缺乏重复验证、数据分割和支持检测细节不透明，导致方法有效性尚未得到充分证明。

---

## 618. SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation

**arXiv ID:** 2605.22658 | [PDF](https://arxiv.org/pdf/2605.22658v1)

**作者:** Zhenyu Lu `[一作]` (Institutes of Advanced Technology, Chinese Academy of Sciences), Yaowei Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9238 | [OpenAlex ID](https://openalex.org/A5100631216)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端可解释的推理分割模型，利用稀疏自编码器将链式思考（CoT）与视觉特征映射到共享稀疏概念空间，并通过查询码本与多槽热图将概念空间可视化地对齐到空间位置信息，从而生成分割掩码。

**💡 创新点**

创新点在于将稀疏自编码器（SAE）作为可解释桥梁，构建“白盒”对齐路径，显式展示推理过程与分割结果之间的关联；同时将强化学习（GRPO）与监督分割联合训练，提升推理与生成的协同效果。

**🔧 技术方法**

技术包括：多模态大型语言模型（LLaVA、Qwen2.5-VL）生成CoT与浓缩令牌；SAE对语言与视觉隐藏状态进行稀疏编码；查询码本+Transformer聚合概念；槽映射（slot mapper）产生多槽热图；SAM风格的两路变压器分割解码器；GRPO强化学习、BCE、Dice等监督损失。

**📊 数据集**

使用数据集：RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO、ReasonSeg，以及200K OBELICS样本用于SAE预训练。

**📈 对比分析**

与27种基线（无LLM、隐式查询、文本定位）在cIoU/gIoU指标上对比，模型在五个基准上均达到或超过SOTA；在零样本ReasonSeg上也表现优于现有方法；强化学习与监督联合训练显著提升性能。

**⚠️ 局限性**

局限性包括：SAE预训练与超参数对性能影响显著；查询码本、槽数等需要手工调优；对更大规模模型或跨域场景的鲁棒性未知；GRPO采样与多槽热图导致推理速度受限。

---

## 619. Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals

**arXiv ID:** 2605.22703 | [PDF](https://arxiv.org/pdf/2605.22703v1)

**作者:** Shuo Yang `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**通讯引用:** 8052 | [OpenAlex ID](https://openalex.org/A5057864403)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种 Near-boundary Stochastic Rescue (NSR) 机制，用以在强化学习可验证奖励 (RLVR) 的硬裁剪设置中恢复边界附近的梯度信号，从而提升训练稳定性与性能。

**💡 创新点**

NSR 的创新点在于将硬裁剪的二值决策转变为对边界附近离群样本的随机救援，形成隐式的 O(1/r²) 软裁剪，同时保持策略更新的保守性。

**🔧 技术方法**

论文采用 RLVR 与可验证奖励、基于组的策略优化 (GRPO/DAPO) 以及在裁剪阈值外引入随机采样窗口 δ 的方式，实现了近边界的随机救援。

**📊 数据集**

实验使用数学推理基准 AIME24/25、AMC 以及通用推理数据集 GPQA、MMLU-Pro，在 Qwen 系列模型（7B、8B、30B）上评估。

**📈 对比分析**

与强基线 DAPO、GSPO 对比，NSR 在 Pass@1/Pass@16 上平均提升约 10% 以上，并在所有规模上显著改善策略熵的稳定性。

**⚠️ 局限性**

研究局限于裁剪型 RLVR 与可验证奖励，且引入了额外的超参数 δ；未对标准 RL、非裁剪目标或更广泛的模型架构进行验证，且对 MoE 路由与 token/sequence 层级裁剪的交互尚未深入探讨。

---

## 620. Therm-FM: Foundation Model is ALL YOU NEED for 3D-ICs Thermal Simulation

**arXiv ID:** 2605.22663 | [PDF](https://arxiv.org/pdf/2605.22663v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 621. Self-Policy Distillation via Capability-Selective Subspace Projection

**arXiv ID:** 2605.22675 | [PDF](https://arxiv.org/pdf/2605.22675v1)

**作者:** Guangya Hao `[一作]` (University of Cambridge), Hanxue Liang `[通讯]` (University of Cambridge)

**通讯引用:** 642 | [OpenAlex ID](https://openalex.org/A5027844642)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自我策略蒸馏（Self-Policy Distillation）框架，通过内部投影子钩子引导大型语言模型在自身生成的文本中选择与目标能力相关的信号，然后再使用这些自生成数据进行下一步微调，以提升模型在代码生成、数学推理和多项选择问答等任务上的表现。

**💡 创新点**

创新点包括：① 在无外部验证或奖励信号的条件下，利用模型自身梯度在“正确性对齐”位置上提取低秩关键/数值子空间；② 将该子空间作为投影子钩子嵌入注意力层，仅在生成阶段改变中间表示；③ 通过这种能力选择机制实现跨领域的可迁移提升。

**🔧 技术方法**

核心技术包括：梯度收集与 SVD 低秩子空间提取、KV 激活投影子钩子、基于 LoRA 的微调、以及使用“正确性对齐”损失对梯度进行聚焦。

**📊 数据集**

使用了三大能力域的数据集：代码生成（MBPP、CodeAlpaca-20k）、数学推理（GSM8K、SVAMP）和问答（MMLU、BBH），并在多种开源 LLM 主干（Qwen2.5‑0.5B/7B/14B、Qwen3‑4B、Llama‑3.1‑8B）上进行实验。

**📈 对比分析**

与基线（原始预训练模型、Plain Self‑Retraining、Simple Self‑Distillation）以及不同主干模型进行对比。实验表明 Self‑Policy Distillation 在所有任务和主干上平均提升约 9% 的性能，并在 in‑domain 及 out‑of‑domain 转移中均显著优于对手，最大提升可达 13% 的单项任务改进。

**⚠️ 局限性**

局限性在于只在三类能力域和少量模型上验证，未覆盖更高风险或多模态任务；同时对子空间维度、投影层选择等超参数的敏感性以及在更大模型上的可扩展性仍需进一步研究。

---

## 622. WorkstreamBench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance

**arXiv ID:** 2605.22664 | [PDF](https://arxiv.org/pdf/2605.22664v1)

**作者:** Thomson Yen `[一作]` (Columbia Business School), Hongseok Namkoong `[通讯]` (Columbia Business School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估LLM智能体在金融行业端到端电子表格任务中的表现，并提出多维质量评估体系。

**💡 创新点**

设计了基于准确性、公式和格式三大维度的细粒度评估分类，并使用LLM判定器实现可扩展的非精确匹配评估。

**🔧 技术方法**

采用LLM判定器进行人工校准的评估方法，结合人工扰动测试与多模型实验来验证评估可靠性。

**📊 数据集**

构建了新的金融电子表格端到端任务集合（包括模型建模、情景分析等），并与已有 SpreadsheetBench 等基准对比。

**📈 对比分析**

通过专家标注与LLM判定器对照，计算准确率、平衡准确率和F1；实验显示Claude Web在所有维度最高，整体得分仅达69.1/100，难度增大时性能显著下降。

**⚠️ 局限性**

仅聚焦金融行业端到端表格任务，部分评估标准与金融特定，缺乏最困难任务样本，且无法完全解释不同模型表现差异。

---

## 623. A Generalized Nash Equilibrium-Seeking Scheme for Trauma Resuscitation

**arXiv ID:** 2605.22661 | [PDF](https://arxiv.org/pdf/2605.22661v1)

**作者:** Promise Ekpo `[一作]` (Cornell University), Lekan Molu `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

将创伤复苏过程建模为一个带耦合不等式约束的分布式广义纳什均衡问题，并提出一种基于时变通信图的主从算法来实现医护人员技能、敏捷度、公平度和通信效率的协同优化

**💡 创新点**

首次将创伤复苏中的临床决策、工作负荷与通信效率等多维度指标融入游戏理论框架，利用广义纳什均衡对多主体系统进行理论分析和实用设计

**🔧 技术方法**

采用分布式主从优化算法、极值映射、线性拉格朗日乘子、连续时间动力学和信号平滑技术，实现了在耦合不等式约束下的自适应决策

**📊 数据集**

使用基于实际ALS（高级生命支持）流程的合成20分钟时间序列数据，模拟7名医护人员在不同技能、敏捷度和沟通条件下的工作场景，并对多种团队规模（5–50人）进行扩展实验

**📈 对比分析**

通过与固定点残差、拉格朗日乘子范数及目标值的收敛曲线对比，验证算法在约10⁻³容差下收敛，迭代次数随团队规模仅轻微上升；实验表明在保持通信图连通的前提下，算法能够在有限时间内实现均衡解

**⚠️ 局限性**

实验仅基于合成数据，未考虑真实临床中的方案偏差、压力诱导行为和非理性决策；模型假设理性半可观测性、通信完整性以及固定技能参数，限制了对真实临床环境的直接迁移

---

## 624. Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora

**arXiv ID:** 2605.22660 | [PDF](https://arxiv.org/pdf/2605.22660v1)

**作者:** Maciej Skorski `[一作]` (University of Luxembourg), Maciej Skorski `[通讯]` (University of Luxembourg)

**通讯引用:** 527 | [OpenAlex ID](https://openalex.org/A5011993100)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了基于LLM的英译波语道德价值语料库扩展管线，利用大规模社交媒体文本进行四阶段验证。

**💡 创新点**

提出了可复现的四方法验证流程（嵌入相似度、CKA、LLM-as-judge评估、分类器对齐）以及低成本LLM翻译方案，可直接把英语道德标签迁移到未标注语言。

**🔧 技术方法**

使用Claude Sonnet LLM进行翻译，LaBSE跨语言嵌入、Centered Kernel Alignment、mDeBERTa‑v3微调、线性分类头，以及LLM-as-judge评估。

**📊 数据集**

采用Moral Foundations Reddit Corpus (MFRC) 与 Moral Foundations Twitter Corpus (MFTC) 共约51,744条帖子。

**📈 对比分析**

通过平均余弦相似度0.889、CKA0.860、LLM-as-judge平均分9.1/10、分类器AUC差距≤0.02，表明翻译后波兰语料与英语在道德分类上的性能基本保持一致。

**⚠️ 局限性**

缺乏波兰本土重标注，可能存在跨文化标签偏移；仅验证Reddit/Twitter，域迁移至新闻或议会语料需要再评估；LLM翻译可能引入词法或文化误译。

---

## 625. Student programming behavior with and without phone notification suppression

**arXiv ID:** 2605.22657 | [PDF](https://arxiv.org/pdf/2605.22657v1)

**作者:** Gavin Eddington `[一作]` (Utah State University), John Edwards `[通讯]` (Utah State University)

**通讯引用:** 4573 | [OpenAlex ID](https://openalex.org/A5062252263)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在CS1课程中，通过对学生在使用和不使用手机通知抑制（如Do Not Disturb）时的键盘敲击日志进行分析，探讨通知抑制对编程任务中的分离与持续关注的影响。

**💡 创新点**

创新点在于将系统级手机通知状态与高分辨率键盘日志结合，构建了大样本（22名学生，47名控制组）并在真实学习环境下评估通知抑制的个体差异与群体效应。

**🔧 技术方法**

技术主要包括在PyCharm IDE中安装ShowYourWork插件记录键盘事件，以及自定义的Android/iOS手机日志应用记录通知抑制状态，并使用负二项混合效应模型（GLMM）分析断点次数。

**📊 数据集**

数据集为22名CS1学生的键盘事件与手机通知状态同步记录，共计201个学生-作业-条件观测点，包含47名仅记录键盘的对照组。

**📈 对比分析**

通过负二项GLMM将打断次数作为因变量，以按千次敲击归一化的打断率为度量，结果显示启用通知抑制时打断率平均下降约19%（IRR=0.81），但个体差异呈双峰分布。

**⚠️ 局限性**

局限性包括：学生自行决定开启抑制，可能与动机等未观测因素相关；仅测量打断频率而非任务完成质量；样本仅来自单一大学CS1课程，外推性有限。

---

## 626. Whose Voice Counts? Mapping Stakeholder Perspectives on AI Through Public Submissions to the U.S. Government

**arXiv ID:** 2605.22650 | [PDF](https://arxiv.org/pdf/2605.22650v1)

**作者:** Alina Karakanta `[一作]` (Leiden University), Aletta G. Dorst `[通讯]` (Leiden University)

**通讯引用:** 2167 | [OpenAlex ID](https://openalex.org/A5010900411)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用公开的10,068封政府征求意见信件，构建六类利益相关者语料库，并对AI行动计划文本进行主题建模与频率对比，评估各群体关注点在政策中的体现程度。

**💡 创新点**

创新点在于提供可复现的完整文本处理流水线，首次系统性地将公众与不同机构对AI的主题关注与官方行动计划进行量化对比。

**🔧 技术方法**

主要技术包括预训练Transformer的BERTopic进行主题建模、Stanza进行句子拆分、Sketch Engine做词频分析以及Python实现的整体Pipeline。

**📊 数据集**

使用的数据集为NITRD网站公开的10,068封英文回应信件（按六类分组）和2025年美国白宫发布的23页AI行动计划PDF。

**📈 对比分析**

通过统计每个主题代表词在AI行动计划中的出现频率，得到主题覆盖度排名；结果显示REST（行业、非盈利、协会等）主题与政策重合度高，而IND（个人）主题重合度低，表明政策偏向机构关注。

**⚠️ 局限性**

局限性包括：样本仅来自已公开提交的信件，可能缺乏边缘群体视角；主题模型受参数影响，可能缺乏细粒度；频率分析未捕捉短语或上下文细节。

---

## 627. MotiMotion: Motion-Controlled Video Generation with Visual Reasoning

**arXiv ID:** 2605.22818 | [PDF](https://arxiv.org/pdf/2605.22818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 628. Vector Policy Optimization: Training for Diversity Improves Test-Time Search

**arXiv ID:** 2605.22817 | [PDF](https://arxiv.org/pdf/2605.22817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 629. Integrable Elasticity via Neural Demand Potentials

**arXiv ID:** 2605.22820 | [PDF](https://arxiv.org/pdf/2605.22820v1)

**作者:** Carlos Heredia `[一作]` (DAMM), Daniel Roncel `[通讯]` (DAMM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Integrable Context-Dependent Demand Network（ICDN），一种基于需求表面学习的多品类零售需求模型；

**💡 创新点**

通过将对数需求视为平滑、可微的价格函数，使弹性可通过一阶导数直接获得，且在模型结构中引入了可学习的线性与非线性价格效应、定向交叉弹性以及稀疏注意力机制，确保弹性可解析且具备经济合理性；

**🔧 技术方法**

利用深度神经网络中的可微价格基底（立方截断样条）、上下文编码器、稀疏注意力和正则化项（平滑度、弹性范围约束）实现；

**📊 数据集**

在Dominick’s Finer Foods的啤酒类别扫描仪面板上进行实验，处理每周店-UPC销量、价格、促销等信息；

**📈 对比分析**

与传统的有向对数-对数线性回归基准相比，ICDN在时间展开交叉验证中均表现出更高的R²、更低的MAE/RMSE；弹性估计更稳定、符合理性（所有自身弹性为负，交叉弹性大多为正，且区间约束内），且置信区间和标准差明显减小；

**⚠️ 局限性**

主要局限在于对因果解释不具备完整性（仅为局部情境下的观测弹性），模型依赖于大量参数且训练过程复杂；同时跨品类弹性仍可能受到观测价格变动不足的影响，导致识别度低。

---

## 630. Remember to be Curious: Episodic Context and Persistent Worlds for 3D Exploration

**arXiv ID:** 2605.22814 | [PDF](https://arxiv.org/pdf/2605.22814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 631. ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning

**arXiv ID:** 2605.22734 | [PDF](https://arxiv.org/pdf/2605.22734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 632. CogAdapt: Transferring Clinical ECG Foundation Models to Wearable Cognitive Load Assessment via Lead Adaptation

**arXiv ID:** 2605.22774 | [PDF](https://arxiv.org/pdf/2605.22774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 633. AI-Driven Multi-Region Provisioning for Cloud Services Using Spot Fleets

**arXiv ID:** 2605.22778 | [PDF](https://arxiv.org/pdf/2605.22778v1)

**作者:** Javier Fabra `[一作]` (Universidad de Zaragoza), Pedro García-López `[通讯]` (Universitat Rovira i Virgili)

**通讯引用:** 4010 | [OpenAlex ID](https://openalex.org/A5101612028)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种AI驱动的多区域 Spot Fleet 预估服务，能够在部署前预测不同区域与分配策略下的 fleet 配置与成本。

**💡 创新点**

实现了跨区域成本预估与决策支持，使用区域与策略专属 LSTM 模型预测 Spot Fleet 行为，兼容 EC2 Spot Service 并在多区域上实现最大 64% 的成本节约。

**🔧 技术方法**

采用监控收集真实部署数据、LSTM 时序预测模型（在 SageMaker 上训练）、AWS S3 存储、EC2 Spot Service API 调用、RESTful API 暴露服务、Audit 反馈循环。

**📊 数据集**

基于 90 天内 9 个 AWS 区域共 720 条 Spot Fleet 监控记录，目标容量 64–1500 vCPU，涵盖多种实例类型与分配策略的数据集。

**📈 对比分析**

通过与 EC2 Spot Service 直接比较，预测准确率达 99.79%（匹配率 92.78% 以上），并在多区域评估中发现最高 64% 的成本节省。

**⚠️ 局限性**

仅在 AWS 平台与部分 x86 实例系列上验证，未覆盖其他云服务商或实例族，且对 Spot 价格波动的长期稳定性假设有限，模型需定期重新训练以适应市场变化。

---

## 634. Diversed Model Discovery via Structured Table Discovery

**arXiv ID:** 2605.22766 | [PDF](https://arxiv.org/pdf/2605.22766v1)

**作者:** Zhengyuan Dong `[一作]` (University of Waterloo), Renée J. Miller `[通讯]` (University Waterloo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于结构化表格的模型搜索框架（NL2Card2Tab2Card），通过在模型湖中对模型卡表进行联合、连接、关键词检索等表格检索操作，扩展检索结果集合，并将检索到的表格映射回模型卡，最后对检索结果进行表格融合与可视化；同时引入基于“nugget”（六属性元组）的评估协议，用以量化检索方法在覆盖用户信息需求上的表现。

**💡 创新点**

创新点包括：①将表格作为可检索和可集成的核心证据单元，突破传统文本相似度检索的同质化局限；②设计三种表格检索算子（unionability、joinability、keyword search）实现结构感知检索；③提出面向表格的方向感知集成算法，解决表格转置导致的匹配问题；④构建基于nugget的可审核评估流程，使得检索质量可在动态模型湖中持续度量。

**🔧 技术方法**

技术方法主要包括：
- 文本检索：基于Dense（Sentence‑BERT+FAISS）、Sparse（Pyserini）和Hybrid两种检索技术作为基线。
- 表格检索：使用Blend框架实现的表格关键词搜索、可连接表格搜索、可联合表格搜索。
- 表格映射与重排序：将表格关联到模型卡，并通过语义相似度重排序。
- 表格融合：采用Orientation‑Aware Integration（基于ALITE）对检索到的表格进行转置识别、对齐与融合。
- Nugget抽取与匹配：使用Prompt‑based抽取器提取六属性元组；Prompt‑based查询‑nugget映射；基于集合覆盖的nugget计分。

**📊 数据集**

数据集：
- 60K+模型卡与对应表格的ModelTables基准（HuggingFace模型湖的精炼版本）。
- 597条来自LitSearch的论文推荐查询，经Prompt重写转化为模型推荐查询，用于评估。

**📈 对比分析**

比较方法：将传统文本检索（Dense、Sparse、Hybrid）与结构化表格检索（Unionable、Joinable、Keyword）在同一top‑k预算下进行对比；对每组查询计算nugget覆盖计数，并统计各方法在不同排名位置的出现频率。实验结果显示：
- 在top‑1/3/5预算下，Unionable表格检索在nugget覆盖量和多样性上显著优于所有文本检索基线；
- 当top‑10预算提升时，差距减小，Sparse和Hybrid因词汇重叠仍具一定竞争力；
- 表格融合后生成的统一视图在可视化对比和信息检索方面比单纯文本检索更直观、丰富。

**⚠️ 局限性**

限制：
- 表格融合在面对转置或结构异质的表格时仍存在匹配误差，需要进一步改进。
- 许多模型卡缺乏完整或可表格化的证据，导致检索无法利用表格信息，需考虑缺失结构推断。
- 评估主要集中在“证据型”查询，对其它意图类型（对比、经验、争论等）的效果尚未系统验证。
- Nugget抽取依赖Prompt，可能在新的模型卡格式或语言多样性下失效，需持续维护抽取模板。

---

## 635. Lumberjack: Better Differentially Private Random Forests through Heavy Hitter Detection in Trees

**arXiv ID:** 2605.22756 | [PDF](https://arxiv.org/pdf/2605.22756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 636. Evaluating Commercial AI Chatbots as News Intermediaries

**arXiv ID:** 2605.22785 | [PDF](https://arxiv.org/pdf/2605.22785v1)

**作者:** Mirac Suzgun `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**通讯引用:** 40694 | [OpenAlex ID](https://openalex.org/A5005779176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对六大商业 AI 聊天机器人（Gemini 3 Flash/Pro、Grok 4、Claude 4.5 Sonnet、GPT‑5、GPT‑4o‑mini）在同一日 BBC 新闻的六个地区/语言（美加、阿拉伯语、法语、印地语、俄语、土耳其语）下进行 14 天的实时事实问答评估，测量其准确率、检索与引用行为、错误类型以及对含假前提的鲁棒性。

**💡 创新点**

创新点在于：①首次大规模、实时跨语言评估，揭示检索是错误主因；②发现显著区域不平等——Hindi 区域准确率仅 79%，并归因于检索偏向英语源；③提出“智能错误”现象（检索到相似但细节不同的来源）；④揭示检索层级的“对抗性悖论”，即检测与恢复能力部分独立。

**🔧 技术方法**

技术手段包括：用 Gemini 3 Flash 自动生成多选题；并行调用各模型的检索‑生成接口；对检索结果进行域名和来源分析；构造对抗性前提变体；进行无检索消融实验；对错误进行三人 LLM 评分的分类。

**📊 数据集**

数据集为 14 天内每日从 BBC 6 个区域服务各采集 15 篇文章，共 90 篇/日，生成 2,100 个五选项多选题，形成 12,600 条模型‑问题实例。

**📈 对比分析**

评估方法为多选准确率和自由响应（FR）对照，发现多选准确率 88.1%（不含 GPT‑4o‑mini 91.9%），顶级模型 Gemini 3 Flash 达 95.6% 以上；禁用检索导致 31–46% 减低；对抗前提时准确率从 88–96% 降至 19–70%，说明鲁棒性显著下降。检索错误占全部错误 71%，英语源占比高，Hindi 区域仅 79.3% 的准确率。

**⚠️ 局限性**

局限包括：多选格式可能夸大准确率；未对开放式对话进行评估；仅基于 BBC 这类高索引、可信源，结果可能对低索引源更差；对抗样本仅英语、四天；模型版本固定，未来更新可能改变结果；生成问卷的 LLM 与解答模型可能存在匹配偏差。

---

## 637. SeqLoRA: Bilevel Orthogonal Adaptation for Continual Multi-Concept Generation

**arXiv ID:** 2605.22743 | [PDF](https://arxiv.org/pdf/2605.22743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 638. Understanding Data Temporality Impact on Large Language Models Pre-training

**arXiv ID:** 2605.22769 | [PDF](https://arxiv.org/pdf/2605.22769v1)

**作者:** Pilchen Hippolyte `[一作]` (Kyutai), Grave Edouard `[通讯]` (Kyutai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM预训练时的数据排序进行研究，提出按时间顺序预训练的方式并构建了7,000+个时间感知问答基准

**💡 创新点**

展示时间顺序预训练能显著提升模型对近年事实的掌握，首次系统评估预训练数据排序对时间知识的影响

**🔧 技术方法**

使用6B参数Transformer解码器，RoPE位置编码，Grouped-Query Attention，SwiGLU激活，AdamW + Warmup-Stable-Decay优化器，构建连续与随机打乱两种训练策略

**📊 数据集**

使用经过滤的Common Crawl快照（2018-2025）以及从Wikidata抽取的时间属性三元组，经过进一步热门度与质量筛选，形成7,167个时间敏感问答样本

**📈 对比分析**

将顺序预训练模型与随机打乱模型在OLMES、Kairos（自建）和TAQA等评测集上进行对比，结果显示顺序模型在一般语言理解与常识任务基本相当，但在时间敏感问答（cloze和生成）上比随机模型提高20-30%，并在最近年份表现出显著优势；但在旧知识上出现一定遗忘

**⚠️ 局限性**

主要局限是顺序预训练导致旧知识衰退，需进一步机制保持历史事实；此外评测集主要基于Wikidata，覆盖面有限；模型规模相对中等，未验证更大模型效果

---

## 639. Cambrian-P: Pose-Grounded Video Understanding

**arXiv ID:** 2605.22819 | [PDF](https://arxiv.org/pdf/2605.22819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 640. Synthetic Data Alone is Enough? Rethinking Data Scarcity in Pediatric Rare Disease Recognition

**arXiv ID:** 2605.22767 | [PDF](https://arxiv.org/pdf/2605.22767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 641. Towards a General Intelligence and Interface for Wearable Health Data

**arXiv ID:** 2605.22759 | [PDF](https://arxiv.org/pdf/2605.22759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 642. Spectral Tail Auxiliary Learning for AI-Generated Image Detection

**arXiv ID:** 2605.22751 | [PDF](https://arxiv.org/pdf/2605.22751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 643. MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems

**arXiv ID:** 2605.22794 | [PDF](https://arxiv.org/pdf/2605.22794v1)

**作者:** Qianshu Cai `[一作]` (University of Science and Technology of China), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19001 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

MOSS实现了在生产级自主代理子系统中进行源代码级自我重写，能够修复宿主（harness）层的路由、状态管理等结构性错误；

**💡 创新点**

其创新点在于：①把自我进化扩展到代码层，突破了传统仅限文本可编辑层的限制；②采用多阶段 deterministic pipeline 与可插拔外部编码代理协同完成定位、规划、实现和验证；③在生产环境下使用即时批量故障证据驱动演化，保障改动针对真实用户失败；

**🔧 技术方法**

技术包括：源代码级重写（Turing-complete 代码搜索空间）、外部 coding‑agent CLI（如 Claude Code、OpenAI Codex 等）用于自动生成修补，容器化部署与 in‑place swap、临时 trial workers 进行运行时验证、Webhook 通知、CLI 控制面板、自动化扫描和批量处理；

**📊 数据集**

使用 OpenClaw 生产系统的实际用户会话日志作为故障批量，结合四个合规/库存检查任务作为评估基准；

**📈 对比分析**

评估方法为：在同一批任务上对比演化前后的平均 Grader 分数，演化后平均分从 0.25 提升至 0.61，单个任务最高提升至 0.90；说明系统在真实环境中实现了显著性能提升；

**⚠️ 局限性**

局限性包括：需要依赖外部 LLM 作为编码代理，改动质量受其能力限制；代码层面改动风险大，需精细验证；只针对可编程的宿主系统，无法自动处理非代码结构的错误；目前仅在 OpenClaw 上验证，跨平台通用性待进一步验证。

---

## 644. FAME: Failure-Aware Mixture-of-Experts for Message-Level Log Anomaly Detection

**arXiv ID:** 2605.22779 | [PDF](https://arxiv.org/pdf/2605.22779v1)

**作者:** Huanchi Wang `[一作]` (University of Toronto), Alberto Leon-Garcia `[通讯]` (University of Toronto)

**通讯引用:** 12572 | [OpenAlex ID](https://openalex.org/A5055726968)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了FAME框架，结合LLM一次性离线分组、K-shot标签、稀疏Mixture-of-Experts路由，实现消息级日志异常检测并输出失败域标签。

**💡 创新点**

创新点在于只使用一次LLM进行语义分区，并通过异向路由实现纯异常域直接检测，减少标签量且保持全本地推理；同时利用K-shot不对称置信度设计专门专家。

**🔧 技术方法**

技术包括Drain日志解析、LLM提示式语义分区、稀疏MoE（门DistilBERT+多分类器）、两阶段BERT预训练+微调、Focal Loss阈值校准、门与选择器的异向路由。

**📊 数据集**

使用Blue Gene/L (BGL) 约475万行日志和Thunderbird 约500万行日志进行评估。

**📈 对比分析**

与经典基线（Drain+RF、TF-IDF+IForest、SBERT+LR）、单模型BERT、前沿LLM推理、非LLM分区等进行对比，K=100时BGL F1达98.16、Thunderbird 99.95；LLM推理成本高，F1略低；单模型远逊；分区选择影响BGL性能但对Thunderbird影响小。

**⚠️ 局限性**

局限在于依赖离线分区需要重新运行以适应概念漂移、对K-shot样本的阈值敏感、对模板切换率高的系统可能难以保持；LLM分区不保证最优；仅在超算日志上验证，真实工业环境可能更复杂。

---

## 645. Advancing Mathematics Research with AI-Driven Formal Proof Search

**arXiv ID:** 2605.22763 | [PDF](https://arxiv.org/pdf/2605.22763v1)

**作者:** George Tsoukalas `[一作]` (Google DeepMind), Swarat Chaudhuri `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一个基于LLM的Lean证明生成框架，构建了四种不同级别的代理（基本、工具增强、进化、完整），并在353个开放研究层面的问题上进行了大规模评估，成功解决了9个正式证明、44个OEIS开放猜想，以及在优化、图论、代数几何、算术组合等领域取得多项新结果。

**💡 创新点**

创新点在于：①将LLM的自然语言推理与Lean编译器的自动验证反馈相结合，实现了可检验的证明草图生成；②通过Elo评级与P‑UCB采样的进化算法对多代理进行协同搜索，显著提升了在困难问题上的成功率；③首次在开放研究层面对数百个正式化猜想进行系统评估，并展示了LLM驱动的正式证明搜索在科研中的潜在价值。

**🔧 技术方法**

使用技术包括：Gemini 3.1 Pro/Flash LLM、Lean证明助手、可搜索脚本与工具（如Olympiad‑level Lean证明系统）、AlphaEvolve风格的进化算法、Elo评分与P‑UCB采样、以及定制的验证与重构框架。

**📊 数据集**

数据集主要包括：Bloom的Formal Conjectures仓库（353个正式化问题）、OEIS 492个开放问题的自动化正式化版本、Erdos Problems等；此外对每个问题都构造了Lean文件草图并加入必要的库和定义。

**📈 对比分析**

通过比较四种代理在同一问题集上的成功率与推理成本（美元）以及壁钟时间来评估性能。完整代理在最难的几个问题（如#138、#125）表现最佳，成本约为基本代理的2–5倍；基本代理在多数问题上同样能成功，成本最低；进化代理整体表现不佳，成本高且成功率低。结果显示，随着LLM能力提升，基本代理的竞争力将进一步增强。

**⚠️ 局限性**

局限性包括：①仍无法解决绝大多数高难度问题，受限于LLM质量与Lean库成熟度；②搜索过程高度随机，导致成本和成功率方差大；③缺乏生成全新理论的能力，主要依赖已有公式化知识；④对误形态（hallucination）的处理仍不完善；⑤当前实现对算子、库的依赖较强，迁移到其他证明助手需要额外工作。

---

## 646. Proxy-Based Approximation of Shapley and Banzhaf Interactions

**arXiv ID:** 2605.22738 | [PDF](https://arxiv.org/pdf/2605.22738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 647. Ternary Decision Trees with Locally-Adaptive Uncertainty Zones

**arXiv ID:** 2605.22740 | [PDF](https://arxiv.org/pdf/2605.22740v1)

**作者:** William Smits `[一作]` `[通讯]` (Avathon), William Smits (Avathon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种在每个节点自动生成局部阈值不确定区间的三元决策树，并在该区间内采用加权融合做预测，能够对边界附近实例标记为边界不确定。

**💡 创新点**

创新点在于：1）利用节点分割统计量（质量曲线、类别重叠、增益比、自助采样、SVM边距）五种方法自适应计算每个节点的δ；2）设计了概率路由与硬中间分支两种路由架构，并证明概率路由更优；3）首次在决策树内部实现局部不确定性处理，而非仅在输出层。

**🔧 技术方法**

技术实现基于CART（基尼/信息增益）分裂准则，结合节点级质量曲线、类别分布重叠、增益比、节点自助采样以及SVM边距来估计δ；采用概率路由将不确定区间内实例按距离加权融合子树输出；评估使用5折交叉验证、决定精度、边界不确定率、整体精度和F1。

**📊 数据集**

实验数据集包括72个OpenML‑CC18分类任务、3个已知Bayes误差的Breiman合成基准（waveform、twonorm、ringnorm）以及4个高风险医疗/金融数据集（Pima diabetes、German credit、Cleveland heart disease、Mammography）。

**📈 对比分析**

与标准CART对比，所有五种δ方法在决定精度上均显著提升（Wilcoxon p≤0.001）。Margin方法在决定精度提升0.4–0.7%、边界不确定率仅为16.7%且效率最高；在合成基准上Margin自校准且能匹配Bayes误差。整体精度与F1也保持竞争或略优。

**⚠️ 局限性**

局限性包括：自助采样方法在样本量>20k时需剔除；Margin方法在小样本或复杂经济特征数据（如German credit）上表现不如CART；δ估计与Bayes误差未完全校准，导致边界不确定率有时过高或过低；三元树仍只能给出预测及其不确定标记，无法完全替代拒绝选项。

---

## 648. Tokenisation via Convex Relaxations

**arXiv ID:** 2605.22821 | [PDF](https://arxiv.org/pdf/2605.22821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 649. On the Parameterized Complexity of Min-Sum-Radii

**arXiv ID:** 2605.22804 | [PDF](https://arxiv.org/pdf/2605.22804v1)

**作者:** Pankaj Kumar `[一作]` (University of Birmingham), Melanie Schmidt `[通讯]` (Heinrich Heine University Dusseldorf)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在由加权或无权无向图诱导的度量空间下，最小总半径聚类（Min‑Sum‑Radii）问题的参数化复杂度。

**💡 创新点**

创新点在于：①首次给出加权图情况下对参数 k+Δ、树宽、顶点覆数等多种结构参数的完整W[1]‑硬性与FPT分类；②针对无权图提出两种自然变体（Exact‑MS、Fixed‑MS），证明其W[1]‑硬性并给出按树深度的FPT算法；③通过细致的参数化归约与指数权重构造，深化了该聚类问题在图度量上的理论边界。

**🔧 技术方法**

使用的技术主要包括：参数化归约（从多彩团、支配集等经典W[1]‑硬问题），指数权重构造、树宽与邻域多样性/模组宽等结构参数的动态规划与搜索，以及对已有树宽算法的时间复杂度改写。

**📊 数据集**

本文没有使用实验数据集，全部工作基于理论证明与算法分析。

**📈 对比分析**

对比方法主要是与已有的k‑Center、k‑Median等经典聚类问题的复杂度研究进行对照；未给出实验性能指标，而是通过时间复杂度上界（如O(2^{3ω}(#(G))^{3ω+1}k^3)等）来展示算法的可行性。

**⚠️ 局限性**

局限性在于：①无权图原问题的P/NP‑判定仍未确定，只能对其变体给出复杂度结果；②部分W[1]‑硬性证明依赖于指数权重构造，实际实例中可能不具备；③对于大规模图的具体实现与性能评估仍缺乏实验验证。

---

## 650. Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving

**arXiv ID:** 2605.22809 | [PDF](https://arxiv.org/pdf/2605.22809v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 651. The Matching Principle: A Geometric Theory of Loss Functions for Nuisance-Robust Representation Learning

**arXiv ID:** 2605.22800 | [PDF](https://arxiv.org/pdf/2605.22800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 652. DecQ: Detail-Condensing Queries for Enhanced Reconstruction and Generation in Representation Autoencoders

**arXiv ID:** 2605.22777 | [PDF](https://arxiv.org/pdf/2605.22777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 653. Optimal Testing of Reed-Muller Codes with an Online Adversary

**arXiv ID:** 2605.22813 | [PDF](https://arxiv.org/pdf/2605.22813v1)

**作者:** Esty Kelman `[一作]` (Massachusetts Institute of Technology), Kai Zhe Zheng `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 10649 | [OpenAlex ID](https://openalex.org/A5100603979)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文提出了一种半采样（semi‑sample）测试器，用于在在线擦除模型下对 Reed‑Muller 码及其提升的仿射不变码进行属性检测；

**💡 创新点**

创新点在于设计半采样测试器，它在查询时先随机选取子空间再在子空间内随机采样，从而在抵御在线擦除的同时保持查询复杂度最优；

**🔧 技术方法**

主要技术包括层次化的超平面一致性引理（hyperplane agreement lemma）、采样引理、PCP 与一致性图分析的组合以及多维归纳证明；

**📊 数据集**

由于论文研究的是理论性质的错误检测，不依赖于任何具体数据集；

**📈 对比分析**

与先前 Minzer‑Zheng 等人提出的 O(q^{3d} (log t)^d) 查询复杂度的在线擦除测试器相比，本文实现了 O((log t/d)^d) 的查询复杂度，显著降低；

**⚠️ 局限性**

局限性在于需要先验知道子空间维度 k 足够大且对擦除率 t 有一定上限，且对只有已知良好局部测试的仿射不变族适用，尚未扩展到所有非线性仿射不变族。

---

## 654. GesVLA: Gesture-Aware Vision-Language-Action Model Embedded Representations

**arXiv ID:** 2605.22812 | [PDF](https://arxiv.org/pdf/2605.22812v1)

**作者:** Wenxuan Guo `[一作]` (Tsinghua University), Jianjiang Feng `[通讯]` (Tsinghua University)

**通讯引用:** 6329 | [OpenAlex ID](https://openalex.org/A5084679040)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GesVLA框架，将手势作为并行指令模态融入视觉‑语言‑动作模型，实现机器人在复杂场景中的空间指令执行。

**💡 创新点**

创新点在于：1）将手势直接编码进多模态潜在空间并通过双VLM架构实现手势与意图推理、动作生成的紧耦合；2）构建可扩展的半合成手势数据生成管道；3）采用两阶段训练策略，先预训练意图推理，再冻结或微调完成动作生成。

**🔧 技术方法**

使用了MediaPipe手势关键点提取、PaliGemma VLM、流式动作生成、双VLM结构、手势关键帧抽取与MLP嵌入、跨注意力机制、Flow匹配训练等技术。

**📊 数据集**

使用了约16k条半合成手势数据（含手势视频、语言指令、目标标注）以及真实机器人演示数据，并通过GroundingDINO、DepthAnythingV2等工具生成场景与目标。

**📈 对比分析**

通过与文本仅VLA、MLLM+VLA、几何管线+VLA等基线对比，GesVLA在意图推理准确率达94.3%，在Pick‑and‑Place、Select Jelly、Select Fruit/Vegetable等任务中平均成功率为83.3%，显著优于基线。

**⚠️ 局限性**

目前仅支持常见指点手势，未覆盖更复杂或多模态交互场景；系统对手势识别依赖MediaPipe与合成数据，跨场景泛化与鲁棒性仍有限。

---

## 655. MambaGaze: Bidirectional Mamba with Explicit Missing Data Modeling for Cognitive Load Assessment from Eye-Gaze Tracking Data

**arXiv ID:** 2605.22775 | [PDF](https://arxiv.org/pdf/2605.22775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 656. Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation

**arXiv ID:** 2605.22765 | [PDF](https://arxiv.org/pdf/2605.22765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 657. Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs

**arXiv ID:** 2605.22823 | [PDF](https://arxiv.org/pdf/2605.22823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 658. Deep Reinforcement Learning for Flexible Job Shop Scheduling with Random Job Arrivals

**arXiv ID:** 2605.22773 | [PDF](https://arxiv.org/pdf/2605.22773v1)

**作者:** Yu Tang `[一作]` (Zurich University of Applied Sciences), Alisa Rupenyan `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 1893 | [OpenAlex ID](https://openalex.org/A5058099719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用事件驱动的深度强化学习框架解决随机作业到达的柔性工作车间排程问题（FJSP）

**💡 创新点**

提出简洁的状态与动作表示，只通过选择预设的调度规则来决策，并用轻量级MLP实现PPO，兼顾可解释性与实时性

**🔧 技术方法**

事件式MDP建模、Proximal Policy Optimization（PPO）算法、轻量级多层感知器、预设调度规则集

**📊 数据集**

生成的同质数据集（作业/机台均为同分布，Poisson到达）与异质数据集（短/长作业、快慢机台、速率惩罚）

**📈 对比分析**

与到达触发的混合整数规划（AT‑MILP）和最佳规则组合（Best HH）比较；在同质场景AT‑MILP略优，异质场景DRL表现更好；总体能与最佳规则持平或略优

**⚠️ 局限性**

仍受限于规则选择的近似性，事件式决策导致局部最优；在同质环境中不如AT‑MILP；仅考虑了单一到达模式，未对参数空间做系统搜索

---

## 659. AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation

**arXiv ID:** 2605.22816 | [PDF](https://arxiv.org/pdf/2605.22816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 660. Quoridor is PSPACE-Hard

**arXiv ID:** 2605.22747 | [PDF](https://arxiv.org/pdf/2605.22747v1)

**作者:** Marius Drop `[一作]`, Finn van der Velde `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了通用 Quoridor 的决策问题是 PSPACE-完全的。

**💡 创新点**

首次将 Schaefer 的正 CNF 公式游戏归约到 Quoridor，构造了复杂的迷宫与通道，展示了两位玩家间的变量选择与路径争夺。

**🔧 技术方法**

使用了多阶段的图灵机式构造、迷宫加通道的细致布置、Railroading 以及 PSPACE 归约技术。

**📊 数据集**

无实验数据，全部为理论构造。

**📈 对比分析**

无实验对比，证明仅为理论复杂度分析。

**⚠️ 局限性**

证明仅适用于两玩家版本；对 Maze Attack、Pinko Pallino 等变体尚未覆盖，且构造相当复杂，未给出可实现的算法。

---

## 661. Plug-in Losses for Evidential Deep Learning: A Simplified Framework for Uncertainty Estimation that Includes the Softmax Classifier

**arXiv ID:** 2605.22746 | [PDF](https://arxiv.org/pdf/2605.22746v1)

**作者:** Berk Hayta `[一作]` (Technische Universitaet Munich), Felix Krahmer `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了简化的 Evidential Deep Learning (EDL) 框架，将传统 Dirichlet-期望损失近似为在 Dirichlet 均值处的插件损失，从而实现单前向传播、易于实现且与软最大（softmax）兼容；

**💡 创新点**

创新点在于证明当证据量增大时，插件损失与传统 EDL 损失差距可控制为 O(1/(α₀+1))，并将软最大视为简化 Evidential 分类器的特殊情况；

**🔧 技术方法**

主要技术包括：Dirichlet 期望损失的泰勒展开、插件损失的构造、KL 正则化的分析以及基于 Google Speech Commands 的关键词检测任务实验；

**📊 数据集**

使用 Google Speech Commands v1 数据集，进行 30 类完整分类任务；

**📈 对比分析**

通过与经典 EDL（含/不含 KL 正则）以及纯软最大模型在预测准确率、选择性预测（覆盖率-准确率）和不确定性指标（vacuum、熵）上的对比，简化模型在大多数阈值下能保持与经典 EDL 相近甚至更优的总准确率；

**⚠️ 局限性**

实验局限仅在单一关键词检测基准上进行，且只评估分布内的选择性预测，未检验对抗或分布外样本的鲁棒性。

---

## 662. GS-QA: A Benchmark for Geospatial Question Answering

**arXiv ID:** 2605.22811 | [PDF](https://arxiv.org/pdf/2605.22811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 663. Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

**arXiv ID:** 2605.22791 | [PDF](https://arxiv.org/pdf/2605.22791v1)

**作者:** Ali Hatamizadeh `[一作]` (Nvidia), Jan Kautz `[通讯]` (Nvidia)

**通讯引用:** 41990 | [OpenAlex ID](https://openalex.org/A5056503617)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Gated DeltaNet-2，一种将擦除与写入解耦的线性递归注意力层；

**💡 创新点**

创新点在于将原先单一标量β_t拆分为关键通道的erase门和值通道的write门，实现更细粒度的记忆编辑，并在保持WY型块级并行训练的同时保留高效的实现；

**🔧 技术方法**

采用了快速权重（fast‑weight）delta‑rule更新、通道级衰减、块级（chunkwise）WY算法、短时卷积+SiLU、L2归一化等技术；

**📊 数据集**

使用的训练数据为FineWeb‑Edu 100B tokens，评估数据集包括WikiText、LAMBADA、PIQA、BoolQ等通用语言模型和常识推理数据集，以及RULER的S‑NIAH和MK‑NIAH合成检索任务和真实检索任务（SQuAD、TriviaQA、DROP、NQ等）；

**📈 对比分析**

与Mamba、Gated DeltaNet、KDA、Mamba‑3等同参数（1.3B）模型对比，Gated DeltaNet‑2在语言建模、零样本常识推理、合成检索以及真实检索任务上均实现了显著提升，尤其在长上下文检索和多键检索中表现最为突出；在训练吞吐量方面，Gated DeltaNet‑2仅略低于KDA，保持了与递归混合模型相近的高效性；

**⚠️ 局限性**

局限性包括仍受固定状态尺寸限制，虽然解耦门提升了记忆控制但在极长上下文或高维值空间时仍可能出现干扰；门的通道化虽然提升了性能，却略微增加参数和计算开销；此外，实验主要聚焦于单头/单向设置，跨多头或双向结构的适用性尚待验证。

---

## 664. LCGuard: Latent Communication Guard for Safe KV Sharing in Multi-Agent Systems

**arXiv ID:** 2605.22786 | [PDF](https://arxiv.org/pdf/2605.22786v1)

**作者:** Sadia Asif `[一作]` (Rensselaer Polytechnic Institute), Karthikeyan Natesan Ramamurthy `[通讯]` (IBM Research)

**通讯引用:** 3862 | [OpenAlex ID](https://openalex.org/A5081874896)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LCGuard，设计一种在多智能体 LLM 系统中对 Transformer KV 缓存的隐式通信进行安全保护的框架；

**💡 创新点**

在表示层面定义泄露可恢复性，将共享 KV 视为可被逆向重构敏感信息的渠道，并通过对抗学习使通信函数最大化重构损失而保持任务性能；

**🔧 技术方法**

采用 Transformer KV 缓存、对抗解码器、双人极小化优化（minimax）、基于重构损失的隐私约束；

**📊 数据集**

在 PrivacyLens、AgentLeak、MAGPIE 三个多智能体隐私基准上评估，使用 Qwen3、Gemma‑2‑9B、LLaMA 等多种规模模型；

**📈 对比分析**

与 Vanilla KV、PrivAct、ADAPT 及 Per‑Agent LCGuard 对比，LCGuard 在保持接近或略低于 Vanilla KV 的任务准确率/帮助度的同时，显著降低攻击成功率（ASR）和泄露率，优于其他方法的隐私‑效能平衡；

**⚠️ 局限性**

仅适用于同构语言模型、缺乏对异构或多模态通信的支持，训练过程需要对抗解码器，且对系统规模和实时性仍有一定开销。

---

## 665. Reducing Political Manipulation with Consistency Training

**arXiv ID:** 2605.22771 | [PDF](https://arxiv.org/pdf/2605.22771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 666. DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback

**arXiv ID:** 2605.22781 | [PDF](https://arxiv.org/pdf/2605.22781v1)

**作者:** Yunpeng Dong `[一作]` (Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为 DeltaBox 的 OS‑级别沙盒，能够在 LLM 驱动的 AI 代理中以毫秒级别完成状态快照与恢复，支持高频树搜索与强化学习训练。

**💡 创新点**

创新点在于：①仅记录连续检查点之间的差异而非完整复制；②引入可动态切换的 OverlayFS 层与 XFS reflink 进行文件层级快照；③结合 CRIU 的增量 dump 与 warm‑template fork 实现低延迟恢复；④通过网络代理隔离 LLM I/O 以实现推理时的“隐藏式”快照。

**🔧 技术方法**

使用技术包括：Linux 6.8 自定义 OverlayFS 扩展、XFS reflink、CRIU 的 soft‑dirty + lazy‑pages、模板池 + LRU 回收、异步热页预热、网络代理（NPD）、基于搜索树的 GC 策略。

**📊 数据集**

评估数据集包括 SWE‑bench（Verified）、GitHub 开源项目（Django、SymPy、Astropy、Xarray 等）以及 RL 微基准，用于测试 MCTS 搜索和 RL 训练的吞吐量。

**📈 对比分析**

与传统文件复制、Docker commit、Firecracker VM snapshot、CRIU + 文件复制等基线对比；MCTS 中单节点恢复平均 5–6 ms，快照 14 ms；相比基线延迟低 1–3 个数量级；RL 训练多路并行时单 fork 延迟 ≤6 ms，整体训练吞吐提升数倍。

**⚠️ 局限性**

局限性包括：模板池大小限制导致失效时恢复回到 8 ms 的慢路径；未支持网络 I/O 的回滚；目前仅针对单进程/单线程代理，对多进程或多线程场景需进一步改进；依赖 XFS reflink 等文件系统特性。

---

## 667. SDPM: Survival Diffusion Probabilistic Model for Continuous-Time Survival Analysis

**arXiv ID:** 2605.22776 | [PDF](https://arxiv.org/pdf/2605.22776v1)

**作者:** Stanislav R. Kirpichenko `[一作]` (Peter Great Saint Petersburg Polytechnic University), Lev V. Utkin `[通讯]` (Peter Great Saint Petersburg Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于扩散概率模型的连续时间生存分析方法 SDPM，直接建模观察到的生存时间与删失指示符的条件联合分布，然后通过采样和 Kaplan‑Meier 估计生成个体生存函数。

**💡 创新点**

创新点在于：① 将生存分析转化为生成式建模，避免对风险函数或生存函数做参数化假设；② 在目标空间使用对数时间标准化和连续高斯混合表示删失指示符，提升数值稳定性和样本合法性；③ 通过可调样本数 K 控制估计精度。

**🔧 技术方法**

使用 Denoising Diffusion Probabilistic Model（DDPM）对标记为 (t,δ) 的数据进行训练，采样时加入条件编码和逆向扩散步骤；结合 AdaLN‑Zero、嵌入编码、Fourier 特征等深度学习技术。

**📊 数据集**

在十个公开生存数据集上评估：FLC、Ovarian、PBC、Retinopathy、Rotterdam、SEER、SUPPORT、TCGA‑GBM、VLBW、WHAS500。

**📈 对比分析**

与六种强基线（RSF、DeepSurv、DeepHit、GBM‑KM、GBM‑Weibull、XGBSE）在 C‑index、时间相关 AUC 与 IBS 进行比较，SDPM 在 IBS 上表现最佳、整体排名最优，且在大多数数据集的 C‑index 与 AUC 也保持竞争力。

**⚠️ 局限性**

主要限制是推断时需多次采样生成样本，导致计算开销较大；当前实现对逆向步数和样本数敏感，需在精度与速度间权衡。

---

## 668. Cyber-Physical Anomaly Detection in IoT-Enabled Smart Grids Using Machine Learning and Metaheuristic Feature Optimization

**arXiv ID:** 2605.22749 | [PDF](https://arxiv.org/pdf/2605.22749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 669. Superhuman Safe and Agile Racing through Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.22748 | [PDF](https://arxiv.org/pdf/2605.22748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 670. The Distillation Game: Adaptive Attacks & Efficient Defenses

**arXiv ID:** 2605.22737 | [PDF](https://arxiv.org/pdf/2605.22737v1)

**作者:** Youssef Allouah `[一作]` (Stanford University), Reza Shokri `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了模型提供者在蒸馏攻击与实用性之间的权衡，提出了一种基于最小‑最大博弈的蒸馏防御与评估框架，并实现了一种使用代理学生的Product‑of‑Experts（PoE）防御；

**💡 创新点**

创新在于将适应性学生评估和教师防御统一为指数倾斜的最佳响应，利用低成本代理价值估计实现了可在推理时执行的PoE防御，并通过适应性评估显著揭示传统防御的弱点；

**🔧 技术方法**

采用KL约束的最小‑最大优化、指数倾斜的最佳响应推导、梯度基价值函数与对数似然差代理价值、Token‑级Logit混合的PoE采样以及LoRA微调等技术；

**📊 数据集**

在数学推理基准GSM8K和MATH上进行实验，使用Llama‑2‑7B等大型语言模型及其代理学生；

**📈 对比分析**

通过比较标准教师、ADS防御和PoE防御三种设置，在被动与适应性学生下测量教师准确率与学生蒸馏准确率，结果显示适应性评估显著提升泄露率，PoE在相同教师准确率下适应性学生的准确率更低、运行时开销更小且保留更高的推理链质量；

**⚠️ 局限性**

仅考虑了基于重权重的适应性攻击，对更复杂攻击形式或多任务场景缺乏探讨，代理价值的简单性可能无法捕捉所有可训练信号，且实验仅覆盖两类推理任务，尚未验证在更大模型或更广泛任务上的普适性。

---

